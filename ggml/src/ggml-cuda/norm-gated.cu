#include "norm-gated.cuh"
#include "ggml-impl.h"

// Wave-per-row RMS norm for narrow rows (ncols <= 256): 8 rows per 256-thread block instead of one
// block per row (the model's per-head norms have ~786k rows of <= 256 -> the upstream launch is
// block-scheduling bound).  The sum reproduces norm.cu's rms_norm_f32<256,...> + block_reduce order
// exactly: per-warp xor trees over 32 consecutive columns, then a xor tree over the 8 partials
// padded with zeros -> bitwise identical results.
// Optional sigmoid gate: dst = (scale * x * w) * (1 / (1 + expf(-z)))  (== op_sigmoid then binbcast mul).
//
// Ported from the other solution's norm-gated.cu for wip/closing-the-gap Phase-1 item 4.
// Default ON (per the default-on policy); GGML_CUDA_DISABLE_NORM_ROWS=1 is the A/B kill switch.

static __device__ __forceinline__ float norm_rows_xor_tree(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) { v += __shfl_xor(v, off); }
    return v;
}

template <bool GATE>
static __global__ void __launch_bounds__(256) rms_rows_f32(const float * x, const float * w, const float * z, float * dst,
        const int ncols, const int64_t nrows, const int64_t nchannels, const int64_t total_rows,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps) {
    const int lane = threadIdx.x & 31;
    const int64_t g = (int64_t) blockIdx.x * 8 + (threadIdx.x >> 5);
    if (g >= total_rows) return;
    const int64_t row = g % nrows, channel = (g / nrows) % nchannels, sample = g / (nrows * nchannels);
    x += sample * stride_sample + channel * stride_channel + row * stride_row;
    dst += g * ncols; if (GATE) z += g * ncols;
    float part[8];
#pragma unroll
    for (int wv = 0; wv < 8; ++wv) {
        const int col = 32 * wv + lane;
        const float xi = col < ncols ? x[col] : 0.0f;
        part[wv] = norm_rows_xor_tree(xi * xi);
    }
    float v = 0.0f;
#pragma unroll
    for (int wv = 0; wv < 8; ++wv) { v = lane == wv ? part[wv] : v; }
    const float tmp = norm_rows_xor_tree(v);
    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);
#pragma unroll
    for (int wv = 0; wv < 8; ++wv) {
        const int col = 32 * wv + lane;
        if (col < ncols) {
            const float t = scale * x[col] * w[col];
            if (GATE) { const float s = 1.0f / (1.0f + expf(-z[col])); dst[col] = t * s; } else { dst[col] = t; }
        }
    }
}

static bool norm_rows_enabled() {
    static const bool enabled = []() {
        const char * env = getenv("GGML_CUDA_DISABLE_NORM_ROWS");
        return env == nullptr || std::atoi(env) == 0;
    }();
    return enabled;
}

// Diagnostic only (never a behaviour switch): GGML_CUDA_NORM_ROWS_DEBUG=1 logs one decision line per
// candidate node, fused or declined, so a gate can prove both that the fusion is live and that an
// arm really reached the aliasing guard below.
static bool norm_rows_debug() {
    static const bool enabled = []() {
        const char * env = getenv("GGML_CUDA_NORM_ROWS_DEBUG");
        return env != nullptr && std::atoi(env) != 0;
    }();
    return enabled;
}

// The byte extent a tensor's elements actually span.  ggml_nbytes() is the DENSE size and understates a
// padded view, which is a real shape here: x is frequently a strided reshape of a projection output.
static size_t norm_rows_span(const ggml_tensor * t) {
    size_t span = ggml_type_size(t->type);
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        span += (size_t) (t->ne[d] - 1) * t->nb[d];
    }
    return span;
}

static bool norm_rows_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    if (a == nullptr || b == nullptr || a->data == nullptr || b->data == nullptr) {
        return false;
    }
    const char * a0 = (const char *) a->data;
    const char * b0 = (const char *) b->data;
    return a0 < b0 + norm_rows_span(b) && b0 < a0 + norm_rows_span(a);
}

// The fused launch reads x, w and z (the gate output) and writes dst, all as the dense
// [ncols, rows] layout the kernel indexes, so it may only run when none of those buffers is behind
// another one -- with ONE exception: an exact alias of x with dst is a read-then-write of the same
// address by the same thread (both tensors then have x's dense layout), which is what the allocator
// hands out when it recycles x's block for the output (the common case for the x -> MUL form).
//
// issue #276: the gated form dispatches the gate MUL_MAT before the fused kernel runs, and in graph
// order x is dead as soon as the RMS_NORM has consumed it while the gate output has exactly x's
// shape, so the allocator's best fit is x's own block (measured on Ling-3.0-tiny: with the
// scheduler's alloc deps disabled the gate output takes x's block outright and the greedy text
// collapses from 172 to 52 characters).  The scheduler's optimize pass keeps x/w/z alive until dst
// (ggml_backend_cuda_graph_optimize, "add alloc deps ... for performance positive fusions"), which
// is what makes the shipped configuration safe -- but this function must not depend on a pass it
// cannot see: ggml_backend_graph_compute() can be driven with no scheduler and therefore no deps at
// all, and a fusion copied from this one inherits the hazard without inheriting the deps.  Verify the
// aliasing here instead and fall back to the unfused chain, which computes the same text.
static const char * norm_rows_alias_conflict(const ggml_cuda_norm_gated_match & m, const ggml_tensor * pre_out) {
    if (pre_out != nullptr &&
            (norm_rows_overlap(pre_out, m.x) || norm_rows_overlap(pre_out, m.w) || norm_rows_overlap(pre_out, m.dst))) {
        return "gate-out";
    }
    if (norm_rows_overlap(m.dst, m.x) && !(m.x->data == m.dst->data && ggml_are_same_layout(m.x, m.dst) && ggml_is_contiguous(m.x))) {
        return "dst-x";
    }
    if (norm_rows_overlap(m.dst, m.w)) {
        return "dst-w";
    }
    return nullptr;
}

static bool rows_shape_ok(const ggml_tensor * x, const ggml_tensor * w, const ggml_tensor * out) {
    if (x->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || out->type != GGML_TYPE_F32) return false;
    if (x->ne[0] > 256 || x->ne[0] % 32 != 0 || x->nb[0] != 4 || !ggml_is_contiguous(w) || !ggml_is_contiguous(out)) return false;
    if (w->ne[0] != x->ne[0] || w->ne[1] != 1 || w->ne[2] != 1 || w->ne[3] != 1) return false;
    if (x->ne[1] * x->ne[2] * x->ne[3] < 4096) return false;
    return ggml_nelements(out) == ggml_nelements(x);
}

int ggml_cuda_norm_gated_match_at(const ggml_cgraph * cgraph, int i, ggml_cuda_norm_gated_match & m) {
    if (!norm_rows_enabled() || i + 3 >= cgraph->n_nodes) return 0;
    const ggml_tensor * rms = cgraph->nodes[i], * mul = cgraph->nodes[i+1];
    if (rms->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL) return 0;
    // issue #285: `pre` is a LOCAL until the match succeeds.  This matcher can record the gate's
    // MUL_MAT and still decline (the UNARY below is not SIGMOID -- Qwen3.x's delta-net output norm is
    // SILU-gated), and ggml_cuda_norm_rows_try_fuse() then reuses the same struct for the plain match,
    // which owns no gate at all.  Writing m.pre only on success means a declined gated match cannot
    // leave a gate index behind for the caller, whatever the caller does next.
    int k = i + 2, pre = -1;
    while (k + 1 < cgraph->n_nodes && k < i + 6) {
        const ggml_op op = cgraph->nodes[k]->op;
        if (op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE) { ++k; continue; }
        if (op == GGML_OP_MUL_MAT && pre < 0) { pre = k; ++k; continue; }
        break;
    }
    if (k + 1 >= cgraph->n_nodes) return 0;
    const ggml_tensor * sig = cgraph->nodes[k]; ggml_tensor * out = cgraph->nodes[k+1];
    if (sig->op != GGML_OP_UNARY || out->op != GGML_OP_MUL) return 0;
    if (ggml_get_unary_op(sig) != GGML_UNARY_OP_SIGMOID) return 0;
    const ggml_tensor * x = rms->src[0];
    const ggml_tensor * w = mul->src[0] == rms ? mul->src[1] : (mul->src[1] == rms ? mul->src[0] : nullptr);
    if (!w) return 0;
    const ggml_tensor * z = sig->src[0];
    if (pre >= 0) { const ggml_tensor * zr = z->view_src ? z->view_src : z; if (zr != cgraph->nodes[pre]) return 0; }
    if (!((out->src[0] == mul && out->src[1] == sig) || (out->src[0] == sig && out->src[1] == mul))) return 0;
    if (!ggml_node_has_n_uses(cgraph, i, 1) || !ggml_node_has_n_uses(cgraph, i+1, 1) || !ggml_node_has_n_uses(cgraph, k, 1)) return 0;
    if (!rows_shape_ok(x, w, out) || z->type != GGML_TYPE_F32 || !ggml_is_contiguous(z) || !ggml_is_contiguous(mul)) return 0;
    if (!ggml_are_same_shape(z, out) || !ggml_are_same_shape(mul, out) || !ggml_are_same_shape(rms, out)) return 0;
    float eps; memcpy(&eps, rms->op_params, sizeof(float)); if (eps < 0.0f) return 0;
    m.x = x; m.w = w; m.z = z; m.dst = out; m.eps = eps; m.pre = pre;
    return (k + 1) - i;
}

// plain RMS_NORM -> MUL(w) on narrow rows
int ggml_cuda_norm_rows_match_at(const ggml_cgraph * cgraph, int i, ggml_cuda_norm_gated_match & m) {
    if (!norm_rows_enabled() || i + 1 >= cgraph->n_nodes) return 0;
    const ggml_tensor * rms = cgraph->nodes[i]; ggml_tensor * mul = cgraph->nodes[i+1];
    if (rms->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL) return 0;
    const ggml_tensor * x = rms->src[0];
    const ggml_tensor * w = mul->src[0] == rms ? mul->src[1] : (mul->src[1] == rms ? mul->src[0] : nullptr);
    if (!w || !ggml_node_has_n_uses(cgraph, i, 1)) return 0;
    if (!rows_shape_ok(x, w, mul) || !ggml_are_same_shape(rms, mul)) return 0;
    float eps; memcpy(&eps, rms->op_params, sizeof(float)); if (eps < 0.0f) return 0;
    // A plain match has no gate (issue #285): clear the field explicitly, so this path cannot read or
    // pre-dispatch a MUL_MAT even if a caller hands it a struct a failed gated match already touched.
    m.x = x; m.w = w; m.z = nullptr; m.dst = mul; m.eps = eps; m.pre = -1;
    return 1;
}

void ggml_cuda_op_norm_gated(ggml_backend_cuda_context & ctx, const ggml_cuda_norm_gated_match & m) {
    const ggml_tensor * x = m.x;
    const int64_t total = x->ne[1] * x->ne[2] * x->ne[3];
    const dim3 grid((unsigned) ((total + 7) / 8)), block(256);
    const ggml_cuda_kernel_launch_params lp(grid, block, 0, ctx.stream());
    if (m.z) {
        ggml_cuda_kernel_launch(rms_rows_f32<true>, lp,
            (const float *) x->data, (const float *) m.w->data, (const float *) m.z->data, (float *) m.dst->data,
            (int) x->ne[0], x->ne[1], x->ne[2], total,
            (int64_t) (x->nb[1] / 4), (int64_t) (x->nb[2] / 4), (int64_t) (x->nb[3] / 4), m.eps);
    } else {
        ggml_cuda_kernel_launch(rms_rows_f32<false>, lp,
            (const float *) x->data, (const float *) m.w->data, (const float *) nullptr, (float *) m.dst->data,
            (int) x->ne[0], x->ne[1], x->ne[2], total,
            (int64_t) (x->nb[1] / 4), (int64_t) (x->nb[2] / 4), (int64_t) (x->nb[3] / 4), m.eps);
    }
}

// Narrow-row RMS norm (+ sigmoid-gated form) fusion entry point.  Returns the number of graph
// nodes to skip when fused, 0 otherwise.  A/B / bisect kill switch: GGML_CUDA_DISABLE_NORM_ROWS=1.
int ggml_cuda_norm_rows_try_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    static const bool disable_norm_rows = [] { const char * e = getenv("GGML_CUDA_DISABLE_NORM_ROWS"); return e != nullptr && std::atoi(e); }();
    if (disable_norm_rows) {
        return 0;
    }
    const ggml_tensor * node = cgraph->nodes[i];
    if (node->op != GGML_OP_RMS_NORM) {
        return 0;
    }
    ggml_cuda_norm_gated_match nm;
    int sk = ggml_cuda_norm_gated_match_at(cgraph, i, nm);
    if (sk == 0) {
        sk = ggml_cuda_norm_rows_match_at(cgraph, i, nm);
    }
    if (sk == 0) {
        return 0;
    }
    // Aliasing guard (issue #276): see norm_rows_alias_conflict().  Never fires on the delivery's own
    // path (the scheduler's alloc deps prevent it); it is the safety net for a caller that computes
    // this graph without them.
    const ggml_tensor * pre_out = nm.pre >= 0 ? cgraph->nodes[nm.pre] : nullptr;
    if (const char * conflict = norm_rows_alias_conflict(nm, pre_out)) {
        if (norm_rows_debug()) {
            GGML_LOG_INFO("norm-rows: i=%d declined=%s gated=%d x=%p z=%p dst=%p\n",
                    i, conflict, (int) (nm.pre >= 0), nm.x->data, pre_out ? pre_out->data : nullptr, nm.dst->data);
        }
        return 0;
    }
    if (pre_out != nullptr) {
        if (!ggml_cuda_compute_forward(ctx, cgraph->nodes[nm.pre])) {
            GGML_ABORT("norm-gated: gate MUL_MAT dispatch failed");
        }
    }
    if (norm_rows_debug()) {
        GGML_LOG_INFO("norm-rows: i=%d fused=1 gated=%d rows=%lld ncols=%lld x=%p z=%p dst=%p\n",
                i, (int) (nm.pre >= 0), (long long) (nm.x->ne[1] * nm.x->ne[2] * nm.x->ne[3]),
                (long long) nm.x->ne[0], nm.x->data, pre_out ? pre_out->data : nullptr, nm.dst->data);
    }
    ggml_cuda_op_norm_gated(ctx, nm);
    return sk;
}
