#include "gdn-conv.cuh"
#include "unary.cuh"
#include <cstdlib>

// The direct kernel is bit-identical to the CONCAT+SSM_CONV+SILU reference on a single-device graph
// (same 4-tap reduction order, same silu, `gdn_concat_tail` still materializes the snapshot tail).
// Under a multi-device tensor split it is NOT: with the fusion off, the raw ops and the upstream
// SSM_CONV+SILU fusion agree byte for byte, while the direct kernel produces a different greedy
// continuation (27B/35B, 3x R9700, -sm tensor; gfx1201 2026-09-23).  The sharded inputs themselves
// are consistent, so the divergence is still open; until it is root-caused keep the fusion on
// single-device graphs only.  GGML_CUDA_CONV_FUSION_MULTI=1 forces it back on (A/B / fix validation)
// and GGML_CUDA_DISABLE_CONV_FUSION=1 is the explicit off switch.
static bool gdn_conv_enabled() {
    // Cache the switches: this runs once per conv launch, and getenv() is expensive on Windows.
    static const bool disabled = getenv("GGML_CUDA_DISABLE_CONV_FUSION") != nullptr;
    if (disabled) return false;
    static const bool multi_ok = getenv("GGML_CUDA_CONV_FUSION_MULTI") != nullptr;
    return multi_ok || ggml_backend_cuda_get_device_count() <= 1;
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

template <bool apply_silu, int TT>
static __global__ void __launch_bounds__(256) gdn_conv_direct_kernel(const float * __restrict__ state, const float * __restrict__ x,
        const float * __restrict__ w, float * __restrict__ y, const int C, const int T) {
    const int c  = blockIdx.x * 256 + threadIdx.x;
    const int t0 = blockIdx.y * TT;
    if (c >= C) return;
    float wr[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) wr[j] = w[c * 4 + j];
    float xw[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) { const int jp = t0 + k; xw[k] = (jp < 3) ? state[c * 3 + jp] : x[(size_t) (jp - 3) * C + c]; }
    const float b = 0.0f;
    const int tend = min(T, t0 + TT);
    for (int t = t0; t < tend; ++t) {
        float sumf = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) sumf += xw[j] * wr[j];
        sumf += b;
        y[(size_t) t * C + c] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
        xw[0] = xw[1]; xw[1] = xw[2]; xw[2] = xw[3];
        const int jp = t + 4; xw[3] = (jp < 3) ? state[c * 3 + jp] : ((jp - 3) < T ? x[(size_t) (jp - 3) * C + c] : 0.0f);
    }
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
    // On RDNA4 the 2..255-token batches (the speculative verify step) take the fused path too; otherwise their CONCAT is a
    // generic non-contiguous copy per layer.  Same kernels, bit-identical.  GGML_CUDA_FUSE_GDN_CONV_VERIFY=0 turns it off.
    static const bool conv_verify = getenv("GGML_CUDA_FUSE_GDN_CONV_VERIFY") == nullptr || atoi(getenv("GGML_CUDA_FUSE_GDN_CONV_VERIFY")) != 0;
    const bool verify_ok = conv_verify && T >= 2 && GGML_CUDA_CC_IS_RDNA4(ggml_cuda_info().devices[ggml_cuda_get_device()].cc);
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
        if (t->op == GGML_OP_CONT && t->src[0]->op == GGML_OP_VIEW && t->src[0]->view_src == cc) continue;
        // the shared build_conv_state (qwen35moe/qwen35/qwen3next) writes the snapshot with a raw
        // view of the concat, while qwen4exp wraps it in a cont.  Accept the snapshot cpy when its
        // source is a 3-column view of the concat (already accounted for by tail_from above); reject
        // any other cpy that reads cc.
        if (t->op == GGML_OP_CPY) {
            const ggml_tensor * s0 = t->src[0];
            if (s0 == nullptr || s0->op != GGML_OP_VIEW || s0->view_src != cc ||
                    s0->ne[0] != 3 || s0->ne[1] != C || s0->ne[2] != 1) return false;
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
    return true;
}

bool ggml_cuda_gdn_conv_match_at_concat(const ggml_cgraph * cgraph, int i, ggml_cuda_gdn_conv_match & m) { return gdn_conv_check(cgraph, i, m); }

bool ggml_cuda_gdn_conv_match_at_conv(const ggml_cgraph * cgraph, int j, ggml_cuda_gdn_conv_match & m) {
    if (!gdn_conv_enabled() || j < 1 || cgraph->nodes[j]->op != GGML_OP_SSM_CONV) return false;
    const ggml_tensor * cc = cgraph->nodes[j]->src[0];
    for (int i = j - 1; i >= 0 && i >= j - 64; --i) if (cgraph->nodes[i] == cc) return gdn_conv_check(cgraph, i, m) && m.conv_idx == j;
    return false;
}

void ggml_cuda_gdn_conv_write_tail(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_conv_match & m) {
    const int ncols = (int) (m.T + 3 - m.tail_from);
    if (ncols <= 0) return;
    const int64_t n = m.C * ncols;
    gdn_concat_tail<<<(unsigned) ((n + 255) / 256), 256, 0, ctx.stream()>>>((const float *) m.state->data, (const float *) m.x->data,
        (float *) m.concat->data, (int) m.C, (int) m.T, (int) m.tail_from, (int) (m.concat->nb[1] / sizeof(float)));
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_gdn_conv_direct(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_conv_match & m) {
    static bool dbg = getenv("GGML_CUDA_CONV_DEBUG") != nullptr;
    if (dbg) { fprintf(stderr, "GDN_CONV_DIRECT C=%lld T=%lld tail_from=%lld silu=%d\n", (long long) m.C, (long long) m.T, (long long) m.tail_from, (int) m.silu); dbg = false; }
    constexpr int TT = 128;
    dim3 grid((unsigned) (m.C / 256), (unsigned) ((m.T + TT - 1) / TT));
    gdn_conv_direct_kernel<true, TT><<<grid, 256, 0, ctx.stream()>>>((const float *) m.state->data, (const float *) m.x->data,
        (const float *) m.w->data, (float *) m.conv_out->data, (int) m.C, (int) m.T);
    CUDA_CHECK(cudaGetLastError());
}
