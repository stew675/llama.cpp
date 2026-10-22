#include "gdn-conv.cuh"
#include "unary.cuh"
#include "mmvq.cuh"
#include <cstdlib>
#include <algorithm>

// The direct kernel is bit-identical to the CONCAT+SSM_CONV+SILU reference: same 4-tap reduction
// order, same silu, and `gdn_concat_tail` still materializes the snapshot tail.
//
// It was gated to single-device graphs from 2026-09-23 (r13) because under a multi-device tensor split
// it produced a different greedy continuation on 3x R9700 (27B/35B), while the raw ops and the
// upstream SSM_CONV+SILU fusion agreed byte for byte.  That divergence was never root-caused and
// **no longer reproduces** on the r186 tree: the post-prefill logits are bit-identical with the
// fusion on and off at P=4096 on 1/2/3 devices, `-sm tensor` and `-sm layer`, dense (27B) and MoE
// (35B-A3B, including the `-ncmoe` server shape), and so is greedy text; the prefill-logit gate
// reads KLD 0.000000 / same-top-p 99.990 % for the fused arm against the unfused one on 3 GPU
// `-sm tensor`.  The likely fix is the r183 rework of exactly the suspect area (the snapshot write
// moved into the conv kernel and the conv-state gather was deferred into it), but that is unproven.
// Relative throughput of the fused arm on 3 GPU `-sm tensor`: 27B IQ3_S tg128 +5.7 % / pp2048
// +1.1 %, 35B-A3B Q4_K_M tg128 +3.0 % (prefill flat).
//
// So the device gate is gone and the fusion runs on every split.  GGML_CUDA_DISABLE_CONV_FUSION=1 is
// the off switch and the A/B arm; the old GGML_CUDA_CONV_FUSION_MULTI force-on knob is retired (it
// only ever countermanded the gate, and forcing it on is now the default).
static bool gdn_conv_enabled() {
    // Cache the switch: this runs once per conv launch, and getenv() is expensive on Windows.
    static const bool disabled = getenv("GGML_CUDA_DISABLE_CONV_FUSION") != nullptr;
    return !disabled;
}

// tail-only materialization of the concat: columns [tail_from, T+3) of every channel row
static __global__ void gdn_concat_tail(const float * __restrict__ state, const float * __restrict__ x, float * __restrict__ out,
                                       const int C, const int T, const int tail_from, const int row_stride) {
    const int ncols = T + 3 - tail_from;
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t) C * ncols) return;
    const int c = (int) (idx / ncols), j = tail_from + (int) (idx % ncols);
    out[(size_t) c * row_stride + j] = (j < 3) ? state[c * 3 + j] : x[(size_t) (j - 3) * C + c];
}

struct gdn_conv_snap_args {
    float * dst[ggml_cuda_gdn_conv_match::SNAP_MAX];
    int     col[ggml_cuda_gdn_conv_match::SNAP_MAX];
    int     n;
};

// The GDN q / k l2 norms (RMS_NORM + SCALE over 128-wide heads, rms_norm_scale_pair_f32<256>) folded into the conv: channels
// [0, nk*128) are the q heads, [nk*128, 2*nk*128) the k heads, and a 256-thread block holds two whole heads (warps 0-3, 4-7).
struct gdn_conv_norm_args {
    float * dst0 = nullptr;   // q: [128, nk, T] contiguous
    float * dst1 = nullptr;   // k
    int     nk = 0;
    float   eps = 0.0f;
    float   scale_out = 0.0f;
};

template <bool apply_silu, int TT, bool fuse_norm = false>
static __global__ void __launch_bounds__(256) gdn_conv_direct_kernel(const float * __restrict__ state, const float * __restrict__ x,
        const float * __restrict__ w, float * __restrict__ y, const int C, const int T, const gdn_conv_snap_args snap,
        const gdn_conv_norm_args norm, const int32_t * __restrict__ state_ids, const int64_t state_row) {
    const int c  = blockIdx.x * 256 + threadIdx.x;
    const int t0 = blockIdx.y * TT;
    if (c >= C) return;
    // a deferred conv-state gather (ggml_cuda_gdn_conv_gather_deferrable): the state is the cache row itself
    if (state_ids != nullptr) {
        state += (int64_t) state_ids[0] * state_row;
    }
    // this channel's state, read before the snapshot writes below (one of them may be the write-back to that row)
    float sr[3] = { 0.0f, 0.0f, 0.0f };
    if (t0 < 3) {
#pragma unroll
        for (int j = 0; j < 3; ++j) sr[j] = state[c * 3 + j];
    }
    const auto st = [&](int jp) { return jp == 0 ? sr[0] : (jp == 1 ? sr[1] : sr[2]); };
    // the snapshot CPYs: dst[c*3 + j] = concat[c][col + j], read straight from state / x
    if (blockIdx.y == 0) {
        for (int s = 0; s < snap.n; ++s) {
#pragma unroll
            for (int j = 0; j < 3; ++j) {
                const int jp = snap.col[s] + j;
                snap.dst[s][c * 3 + j] = (jp < 3) ? st(jp) : x[(size_t) (jp - 3) * C + c];
            }
        }
    }
    float wr[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) wr[j] = w[c * 4 + j];
    float xw[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) { const int jp = t0 + k; xw[k] = (jp < 3) ? st(jp) : x[(size_t) (jp - 3) * C + c]; }
    const float b = 0.0f;
    const int tend = min(T, t0 + TT);
    for (int t = t0; t < tend; ++t) {
        float sumf = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) sumf += xw[j] * wr[j];
        sumf += b;
        const float yv = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
        y[(size_t) t * C + c] = yv;
        if constexpr (fuse_norm) {
            // rms_norm_scale_pair_f32<256> on a 128-column row: thread col holds 0 + x*x (threads 128..255 hold 0), the
            // warp butterfly, then warp 0 butterflies the 8 warp sums (warps 4..7 are 0, lanes 8..31 the 0 sentinel).
            // Here the head's 4 warp sums sit in lanes 0..3 and 0 in the rest: the same values in the same lanes, so the
            // same mean, scale and outputs.
            __shared__ float s_w[256/WARP_SIZE];
            const int  lane = threadIdx.x % WARP_SIZE, warp = threadIdx.x / WARP_SIZE;
            const bool qk   = c < 2*norm.nk*128;
            float tmp = 0.0f;
            if (qk) {
                tmp += yv * yv;
            }
            tmp = warp_reduce_sum(tmp);
            if (lane == 0) {
                s_w[warp] = tmp;
            }
            __syncthreads();
            const int half = warp / (128/WARP_SIZE);
            float val = 0.0f;
            if (lane < 256/WARP_SIZE) {
                val = lane < 128/WARP_SIZE ? s_w[half*(128/WARP_SIZE) + lane] : 0.0f;
            }
            val = warp_reduce_sum(val);
            if (qk) {
                const float mean  = val / 128;
                const float scale = rsqrtf(mean + norm.eps);
                const int   is_k  = c >= norm.nk*128;
                const int   h     = c / 128 - is_k*norm.nk;
                float * dst = is_k ? norm.dst1 : norm.dst0;
                dst[((size_t) t*norm.nk + h)*128 + c % 128] = norm.scale_out * (scale * yv);
            }
            __syncthreads();
        }
        xw[0] = xw[1]; xw[1] = xw[2]; xw[2] = xw[3];
        const int jp = t + 4; xw[3] = (jp < 3) ? st(jp) : ((jp - 3) < T ? x[(size_t) (jp - 3) * C + c] : 0.0f);
    }
}

// byte span of a (possibly strided) tensor
static void gdn_conv_span(const ggml_tensor * t, uintptr_t & lo, uintptr_t & hi) {
    lo = (uintptr_t) t->data;
    size_t last = 0;
    for (int k = 0; k < 4; ++k) last += (size_t) (t->ne[k] - 1) * t->nb[k];
    hi = lo + last + ggml_type_size(t->type);
}

static bool gdn_conv_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    if (!a || !b || !a->data || !b->data) return false;
    uintptr_t alo, ahi, blo, bhi;
    gdn_conv_span(a, alo, ahi); gdn_conv_span(b, blo, bhi);
    return alo < bhi && blo < ahi;
}

static bool gdn_conv_noop(const ggml_tensor * t) {
    return t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_NONE;
}

// The fold moves the snapshot writes from the CPY nodes to the SSM_CONV node and reads state / x there.  It is taken only
// when nothing in between can see the difference: no other node writes over state / x or touches a snapshot destination,
// and the destinations do not overlap state / x or each other.  GGML_CUDA_DISABLE_GDN_CONV_SNAP=1: off.
static bool gdn_conv_snap_safe(const ggml_cgraph * cgraph, const ggml_cuda_gdn_conv_match & m) {
    static const bool disabled = getenv("GGML_CUDA_DISABLE_GDN_CONV_SNAP") != nullptr && atoi(getenv("GGML_CUDA_DISABLE_GDN_CONV_SNAP")) != 0;
    if (disabled) return false;
    for (int a = 0; a < m.n_snap; ++a) {
        // a skipped CPY also consumes the node after it (try_fuse returns the count of EXTRA nodes): that one must be a no-op
        const int nx = m.snap_idx[a] + 1;
        if (nx >= m.conv_idx || !gdn_conv_noop(cgraph->nodes[nx])) return false;
        if (gdn_conv_overlap(m.snap_dst[a], m.state) || gdn_conv_overlap(m.snap_dst[a], m.x)) return false;
        for (int b = a + 1; b < m.n_snap; ++b) if (gdn_conv_overlap(m.snap_dst[a], m.snap_dst[b])) return false;
    }
    for (int n = m.concat_idx + 1; n < m.conv_idx; ++n) {
        const ggml_tensor * t = cgraph->nodes[n];
        bool mine = false;
        for (int a = 0; a < m.n_snap; ++a) if (m.snap_idx[a] == n) mine = true;
        if (mine || gdn_conv_noop(t)) continue;
        if (gdn_conv_overlap(t, m.state) || gdn_conv_overlap(t, m.x)) return false;
        for (int a = 0; a < m.n_snap; ++a) {
            if (gdn_conv_overlap(t, m.snap_dst[a])) return false;
            for (int s = 0; s < GGML_MAX_SRC && t->src[s]; ++s) if (gdn_conv_overlap(t->src[s], m.snap_dst[a])) return false;
        }
    }
    return true;
}

static bool gdn_conv_check(const ggml_cgraph * cgraph, int i, ggml_cuda_gdn_conv_match & m) {
    if (!gdn_conv_enabled() || i < 0 || i + 2 >= cgraph->n_nodes) return false;
    const ggml_tensor * cc = cgraph->nodes[i];
    if (cc->op != GGML_OP_CONCAT || cc->type != GGML_TYPE_F32 || ggml_get_op_params_i32(cc, 0) != 0) return false;
    const ggml_tensor * st = cc->src[0]; const ggml_tensor * tr = cc->src[1];
    if (!st || !tr || st->type != GGML_TYPE_F32 || tr->type != GGML_TYPE_F32) return false;
    if (tr->op != GGML_OP_TRANSPOSE || !tr->view_src) return false;
    const ggml_tensor * x = tr->view_src;
    if (x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || x->ne[2] != 1 || x->ne[3] != 1) return false;
    const int64_t C = x->ne[0], T = x->ne[1];
    if (st->ne[0] != 3 || st->ne[1] != C || st->ne[2] != 1 || st->ne[3] != 1 || !ggml_is_contiguous(st)) return false;
    if (tr->ne[0] != T || tr->ne[1] != C || tr->view_offs != 0) return false;
    if (cc->ne[0] != T + 3 || cc->ne[1] != C || cc->ne[2] != 1 || cc->ne[3] != 1 || !ggml_is_contiguous(cc)) return false;
    int n_snap = 0; bool snap_ok = true;
    // On RDNA4 and RDNA3_0 the 1..255-token batches (decode and the speculative verify step) take the fused path too; otherwise
    // their CONCAT is a generic non-contiguous copy per layer (decode: the ssm_conv_in epilogue, which steps aside for this path).
    // Same kernels, bit-identical.  GGML_CUDA_FUSE_GDN_CONV_VERIFY=0 turns it off.
    static const bool conv_verify = getenv("GGML_CUDA_FUSE_GDN_CONV_VERIFY") == nullptr || atoi(getenv("GGML_CUDA_FUSE_GDN_CONV_VERIFY")) != 0;
    const int  cc_verify = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const bool verify_ok = conv_verify && T >= 1 && (GGML_CUDA_CC_IS_RDNA4(cc_verify) || GGML_CUDA_CC_IS_RDNA3_0(cc_verify));
    if ((T < 256 && !verify_ok) || C % 256 != 0) return false;
    if (i + 1 >= cgraph->n_nodes || cgraph->nodes[i + 1]->op != GGML_OP_VIEW || cgraph->nodes[i + 1]->view_src != cc) return false;
    int64_t tail_from = T + 3; int conv = -1;
    for (int n = i + 1; n < cgraph->n_nodes && n < i + 64; ++n) {
        const ggml_tensor * t = cgraph->nodes[n];
        if (t->op == GGML_OP_SSM_CONV && t->src[0] == cc) { conv = n; break; }
        bool reads = false;
        for (int s = 0; s < GGML_MAX_SRC && t->src[s]; ++s) if (t->src[s] == cc || t->src[s]->view_src == cc) reads = true;
        if (!reads) continue;
        if (t->op == GGML_OP_VIEW) {
            if (t->ne[0] != 3 || t->ne[1] != C || t->nb[1] != cc->nb[1] || t->view_offs % sizeof(float)) return false;
            const int64_t col = (int64_t) (t->view_offs / sizeof(float)); if (col < 0 || col + 3 > T + 3) return false;
            tail_from = std::min(tail_from, col); continue;
        }
        if (t->op == GGML_OP_CONT && t->src[0]->op == GGML_OP_VIEW && t->src[0]->view_src == cc) { snap_ok = false; continue; }
        // the shared build_conv_state (qwen35moe/qwen35/qwen3next) writes the snapshot with a raw
        // view of the concat, while qwen4exp wraps it in a cont.  Accept the snapshot cpy when its
        // source is a 3-column view of the concat (already accounted for by tail_from above); reject
        // any other cpy that reads cc.
        if (t->op == GGML_OP_CPY) {
            const ggml_tensor * s0 = t->src[0];
            if (s0 == nullptr || s0->op != GGML_OP_VIEW || s0->view_src != cc ||
                    s0->ne[0] != 3 || s0->ne[1] != C || s0->ne[2] != 1) return false;
            const ggml_tensor * d = t->src[1];
            if (n_snap < ggml_cuda_gdn_conv_match::SNAP_MAX && d && d->type == GGML_TYPE_F32 && ggml_is_contiguous(d) &&
                    ggml_nelements(d) == 3 * C && s0->nb[1] == cc->nb[1] && s0->view_offs % sizeof(float) == 0) {
                m.snap_idx[n_snap] = n; m.snap_col[n_snap] = (int) (s0->view_offs / sizeof(float)); m.snap_dst[n_snap] = t->src[1]; ++n_snap;
            } else {
                snap_ok = false;
            }
            continue;
        }
        return false;
    }
    if (conv < 0) return false;
    const ggml_tensor * cv = cgraph->nodes[conv];
    const ggml_tensor * w = cv->src[1];
    if (cv->type != GGML_TYPE_F32 || !w || w->type != GGML_TYPE_F32 || w->ne[0] != 4 || w->ne[1] != C || !ggml_is_contiguous(w)) return false;
    if (cv->ne[0] != C || cv->ne[1] != T || cv->ne[2] != 1 || !ggml_is_contiguous(cv)) return false;
    if (conv + 1 >= cgraph->n_nodes) return false;
    const ggml_tensor * su = cgraph->nodes[conv + 1];
    if (su->op != GGML_OP_UNARY || ggml_get_unary_op(su) != GGML_UNARY_OP_SILU || su->src[0] != cv || su->type != GGML_TYPE_F32 || !ggml_is_contiguous(su)) return false;
    for (int n = conv + 2; n < cgraph->n_nodes; ++n) { const ggml_tensor * t = cgraph->nodes[n];
        for (int s = 0; s < GGML_MAX_SRC && t->src[s]; ++s) if (t->src[s] == cv || t->src[s]->view_src == cv) return false; }
    m.concat_idx = i; m.conv_idx = conv; m.x = x; m.state = st; m.concat = cc; m.w = w; m.conv_out = cgraph->nodes[conv + 1];
    m.C = C; m.T = T; m.tail_from = tail_from; m.silu = true;
    m.n_snap = n_snap;
    m.snap = snap_ok && n_snap > 0 && gdn_conv_snap_safe(cgraph, m);
    return true;
}

static bool gdn_conv_decode_ok(const ggml_cgraph * cgraph, const ggml_cuda_gdn_conv_match & m);

bool ggml_cuda_gdn_conv_match_at_concat(const ggml_cgraph * cgraph, int i, ggml_cuda_gdn_conv_match & m) {
    return gdn_conv_check(cgraph, i, m) && gdn_conv_decode_ok(cgraph, m);
}

// The structural match alone (graph_optimize, before allocation: no data pointers yet).  A superset of match_at_concat, so
// its alloc deps cover every case the compute-time matchers take.
bool ggml_cuda_gdn_conv_pattern_at_concat(const ggml_cgraph * cgraph, int i, ggml_cuda_gdn_conv_match & m) { return gdn_conv_check(cgraph, i, m); }

bool ggml_cuda_gdn_conv_match_at_conv(const ggml_cgraph * cgraph, int j, ggml_cuda_gdn_conv_match & m) {
    if (!gdn_conv_enabled() || j < 1 || cgraph->nodes[j]->op != GGML_OP_SSM_CONV) return false;
    const ggml_tensor * cc = cgraph->nodes[j]->src[0];
    for (int i = j - 1; i >= 0 && i >= j - 64; --i) if (cgraph->nodes[i] == cc) return gdn_conv_check(cgraph, i, m) && m.conv_idx == j && gdn_conv_decode_ok(cgraph, m);
    return false;
}

void ggml_cuda_gdn_conv_write_tail(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_conv_match & m) {
    {
        const ggml_tensor * root = m.state;
        while (root->view_src != nullptr) {
            root = root->view_src;
        }
        GGML_ASSERT(std::find(ctx.gdn_deferred_gathers.begin(), ctx.gdn_deferred_gathers.end(), root) == ctx.gdn_deferred_gathers.end() &&
                    "a deferred conv-state gather must take the snapshot-fold path");
    }
    const int ncols = (int) (m.T + 3 - m.tail_from);
    if (ncols <= 0) return;
    const int64_t n = m.C * ncols;
    gdn_concat_tail<<<(unsigned) ((n + 255) / 256), 256, 0, ctx.stream()>>>((const float *) m.state->data, (const float *) m.x->data,
        (float *) m.concat->data, (int) m.C, (int) m.T, (int) m.tail_from, (int) (m.concat->nb[1] / sizeof(float)));
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_gdn_conv_direct(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_conv_match & m, const gdn_conv_norm_args * norm) {
    static bool dbg = getenv("GGML_CUDA_CONV_DEBUG") != nullptr;
    if (dbg) { fprintf(stderr, "GDN_CONV_DIRECT C=%lld T=%lld tail_from=%lld silu=%d snap=%d n_snap=%d\n", (long long) m.C, (long long) m.T, (long long) m.tail_from, (int) m.silu, (int) m.snap, m.n_snap); dbg = false; }
    constexpr int TT = 128;
    dim3 grid((unsigned) (m.C / 256), (unsigned) ((m.T + TT - 1) / TT));
    gdn_conv_snap_args snap{};
    if (m.snap) {
        snap.n = m.n_snap;
        for (int a = 0; a < m.n_snap; ++a) { snap.dst[a] = (float *) m.snap_dst[a]->data; snap.col[a] = m.snap_col[a]; }
    }
    // the state's GET_ROWS was deferred by the graph loop (ggml_cuda_gdn_conv_gather_deferrable): read the cache row
    const float *   state     = (const float *) m.state->data;
    const int32_t * state_ids = nullptr;
    int64_t         state_row = 0;
    {
        const ggml_tensor * root = m.state;
        while (root->view_src != nullptr) {
            root = root->view_src;
        }
        if (std::find(ctx.gdn_deferred_gathers.begin(), ctx.gdn_deferred_gathers.end(), root) != ctx.gdn_deferred_gathers.end()) {
            GGML_ASSERT(m.snap && m.T <= 16 && m.state->data == root->data);
            state     = (const float *) root->src[0]->data;
            state_ids = (const int32_t *) root->src[1]->data;
            state_row = root->src[0]->nb[1] / sizeof(float);
        }
    }
    if (norm) {
        gdn_conv_direct_kernel<true, TT, true><<<grid, 256, 0, ctx.stream()>>>(state, (const float *) m.x->data,
            (const float *) m.w->data, (float *) m.conv_out->data, (int) m.C, (int) m.T, snap, *norm, state_ids, state_row);
    } else {
        gdn_conv_direct_kernel<true, TT><<<grid, 256, 0, ctx.stream()>>>(state, (const float *) m.x->data,
            (const float *) m.w->data, (float *) m.conv_out->data, (int) m.C, (int) m.T, snap, gdn_conv_norm_args{}, state_ids, state_row);
    }
    CUDA_CHECK(cudaGetLastError());
}

static bool gdn_conv_noop_op(const ggml_tensor * t) {
    return t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_NONE;
}

// RMS_NORM + SCALE (scale only) over a 128-wide head view of the conv output, the view starting at channel `ch0`
static bool gdn_conv_norm_one(const ggml_cgraph * cgraph, int j, const ggml_cuda_gdn_conv_match & m, int64_t ch0, int64_t nk,
                              float & eps, float & scale) {
    if (j + 1 >= cgraph->n_nodes) return false;
    const ggml_tensor * rn = cgraph->nodes[j];
    const ggml_tensor * sc = cgraph->nodes[j + 1];
    if (rn->op != GGML_OP_RMS_NORM || sc->op != GGML_OP_SCALE || sc->src[0] != rn || rn->type != GGML_TYPE_F32 ||
            sc->type != GGML_TYPE_F32 || !ggml_is_contiguous(sc)) return false;
    const ggml_tensor * v = rn->src[0];
    if (!v || v->op != GGML_OP_VIEW || v->view_src != m.conv_out || v->type != GGML_TYPE_F32) return false;
    if (v->ne[0] != 128 || v->ne[1] != nk || v->ne[2] != m.T || v->ne[3] != 1 || v->nb[0] != sizeof(float) ||
            v->nb[1] != 128*sizeof(float) || v->nb[2] != (size_t) m.C*sizeof(float) || v->view_offs != (size_t) ch0*sizeof(float)) return false;
    if (sc->ne[0] != 128 || sc->ne[1] != nk || sc->ne[2] != m.T || sc->ne[3] != 1) return false;
    float bias;
    memcpy(&eps,   rn->op_params, sizeof(float));
    memcpy(&scale, sc->op_params, sizeof(float));
    memcpy(&bias,  (const float *) sc->op_params + 1, sizeof(float));
    // only rms_norm / scale outputs that nothing else reads (the views between are no-ops)
    return bias == 0.0f && eps >= 0.0f && (rn->flags & GGML_TENSOR_FLAG_COMPUTE) && (sc->flags & GGML_TENSOR_FLAG_COMPUTE);
}

// The q / k l2 norm pair right after the conv (only no-op nodes between): the direct kernel also writes both normalized
// outputs.  Returns the index of the second SCALE, -1 when the pattern is not there.  GGML_CUDA_FUSE_GDN_CONV_NORM=0: off.
static int gdn_conv_norm_match(const ggml_cgraph * cgraph, const ggml_cuda_gdn_conv_match & m, gdn_conv_norm_args & na) {
    static const bool fuse = getenv("GGML_CUDA_FUSE_GDN_CONV_NORM") == nullptr || atoi(getenv("GGML_CUDA_FUSE_GDN_CONV_NORM")) != 0;
    if (!fuse || !m.silu || m.C % 256 != 0) return -1;
    int j = m.conv_idx + 2;
    while (j < cgraph->n_nodes && gdn_conv_noop_op(cgraph->nodes[j])) ++j;
    if (j >= cgraph->n_nodes || cgraph->nodes[j]->op != GGML_OP_RMS_NORM) return -1;
    const ggml_tensor * v0 = cgraph->nodes[j]->src[0];
    if (!v0 || v0->ne[0] != 128 || v0->view_offs != 0) return -1;
    const int64_t nk = v0->ne[1];
    if (nk < 1 || 2*nk*128 > m.C) return -1;
    float eps0, sc0, eps1, sc1;
    if (!gdn_conv_norm_one(cgraph, j, m, 0, nk, eps0, sc0)) return -1;
    int j2 = j + 2;
    while (j2 < cgraph->n_nodes && gdn_conv_noop_op(cgraph->nodes[j2])) ++j2;
    if (!gdn_conv_norm_one(cgraph, j2, m, nk*128, nk, eps1, sc1) || eps0 != eps1 || sc0 != sc1) return -1;
    // nothing after the pair reads the RMS_NORM outputs (only the SCALEs are produced)
    for (int n = j2 + 2; n < cgraph->n_nodes; ++n) {
        const ggml_tensor * t = cgraph->nodes[n];
        for (int s = 0; s < GGML_MAX_SRC && t->src[s]; ++s) {
            if (t->src[s] == cgraph->nodes[j] || t->src[s] == cgraph->nodes[j2] ||
                    t->src[s]->view_src == cgraph->nodes[j] || t->src[s]->view_src == cgraph->nodes[j2]) return -1;
        }
    }
    // the SCALE outputs are written while the conv still reads x / state / w straight from their buffers; the allocator
    // can place them over the state or x (both dead after the CONCAT in graph order), so the fold needs disjoint memory
    const auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
        const uintptr_t a0 = (uintptr_t) a->data, a1 = a0 + ggml_nbytes(a), b0 = (uintptr_t) b->data, b1 = b0 + ggml_nbytes(b);
        return a0 < b1 && b0 < a1;
    };
    for (const ggml_tensor * d : { cgraph->nodes[j + 1], cgraph->nodes[j2 + 1] }) {
        for (const ggml_tensor * src : { m.x, m.state, m.w, (const ggml_tensor *) m.conv_out }) {
            if (overlap(d, src)) return -1;
        }
    }
    na.dst0 = (float *) cgraph->nodes[j + 1]->data;
    na.dst1 = (float *) cgraph->nodes[j2 + 1]->data;
    na.nk = (int) nk; na.eps = eps0; na.scale_out = sc0;
    return j2 + 1;
}

// One token (decode; GGML_CUDA_FUSE_GDN_CONV_DECODE=0: off): the direct conv only where it also takes the q / k l2 norm pair (otherwise the ssm_conv_in epilogue,
// which writes the CONCAT inside the in-proj GEMV, is cheaper) and only where the decode-only ssm_prescan fusion would not
// take the conv.  Uses data pointers: compute time only.  Every compute-time caller (CONCAT, SSM_CONV, the ssm_conv_in
// stand-down, the snapshot and gather folds) goes through the two matchers above, so they all decide the same way.
static bool gdn_conv_decode_ok(const ggml_cgraph * cgraph, const ggml_cuda_gdn_conv_match & m) {
    if (m.T > 1) {
        return true;
    }
    static const bool decode = getenv("GGML_CUDA_FUSE_GDN_CONV_DECODE") == nullptr || atoi(getenv("GGML_CUDA_FUSE_GDN_CONV_DECODE")) != 0;
    if (!decode) {
        return false;
    }
    gdn_conv_norm_args na;
    return gdn_conv_norm_match(cgraph, m, na) > 0 && !ggml_cuda_ssm_prescan_match_at(cgraph, m.conv_idx);
}

// Depthwise causal conv1d fusion entry point (GDN).  Returns the number of graph nodes to skip
// when fused, 0 otherwise.  A/B / bisect kill switch: GGML_CUDA_DISABLE_CONV_FUSION=1.
int ggml_cuda_gdn_conv_try_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    static const bool disable_conv_fusion = getenv("GGML_CUDA_DISABLE_CONV_FUSION") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_CONV_FUSION"));
    if (disable_conv_fusion) {
        return 0;
    }
    const ggml_tensor * node = cgraph->nodes[i];
    if (node->op != GGML_OP_CONCAT && node->op != GGML_OP_SSM_CONV) {
        return 0;
    }
    ggml_cuda_gdn_conv_match gm;
    if (node->op == GGML_OP_CONCAT && !ctx.gdn_deferred_gathers.empty() && !ggml_cuda_gdn_conv_match_at_concat(cgraph, i, gm)) {
        const ggml_tensor * root = node->src[0];
        while (root != nullptr && root->view_src != nullptr) {
            root = root->view_src;
        }
        GGML_ASSERT(std::find(ctx.gdn_deferred_gathers.begin(), ctx.gdn_deferred_gathers.end(), root) == ctx.gdn_deferred_gathers.end() &&
                    "a CONCAT over a deferred conv-state gather must be taken by the direct conv");
        return 0;
    }
    if (node->op == GGML_OP_CONCAT && ggml_cuda_gdn_conv_match_at_concat(cgraph, i, gm)) {
        if (!gm.snap) {
            ggml_cuda_gdn_conv_write_tail(ctx, gm);
        }
        return 1;
    }
    if (node->op == GGML_OP_SSM_CONV && ggml_cuda_gdn_conv_match_at_conv(cgraph, i, gm)) {
        gdn_conv_norm_args na;
        const int last = gdn_conv_norm_match(cgraph, gm, na);
        if (last > 0) {
            ggml_cuda_gdn_conv_direct(ctx, gm, &na);
            return last - i;
        }
        ggml_cuda_gdn_conv_direct(ctx, gm);
        return 1;
    }
    return 0;
}

// A snapshot CPY whose write the direct kernel already makes (gm.snap): it launches nothing.  Must run before the batched-copy
// fusion, which would otherwise take the run of snapshot CPYs.  Returns 1 (the CPY and the no-op node after it, which
// gdn_conv_snap_safe requires) when the node is skipped, 0 otherwise.
int ggml_cuda_gdn_conv_snap_try_fuse(const ggml_cgraph * cgraph, int i) {
    const ggml_tensor * node = cgraph->nodes[i];
    if (node->op != GGML_OP_CPY || !node->src[0] || node->src[0]->op != GGML_OP_VIEW) return 0;
    const ggml_tensor * cc = node->src[0]->view_src;
    if (!cc || cc->op != GGML_OP_CONCAT) return 0;
    for (int j = i - 1; j >= 0 && j >= i - 64; --j) {
        if (cgraph->nodes[j] != cc) continue;
        ggml_cuda_gdn_conv_match gm;
        if (!ggml_cuda_gdn_conv_match_at_concat(cgraph, j, gm) || !gm.snap) return 0;
        for (int a = 0; a < gm.n_snap; ++a) if (gm.snap_idx[a] == i) return 1;
        return 0;
    }
    return 0;
}
