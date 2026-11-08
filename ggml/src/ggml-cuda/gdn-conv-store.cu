// The Gated DeltaNet conv of a decode / verify step as one kernel on the graph of upstream #31660 (see gdn-conv-store.cuh).
//
// Per channel and sequence: the three state values (the cache row itself when the state's GET_ROWS is deferred), the four taps and the T
// inputs; out: the conv + silu row, the q / k rows normalized (the l2 norm pair's SCALE outputs), and the Kw snapshot windows of
// the concatenated input straight into the cache's rollback slots.  The conv and the norm keep the arithmetic of rdna-boosts'
// direct conv (gdn-conv.cu, MIT): the same 4-tap order and silu; the q / k sums are rms_norm_scale_pair_f32<256>'s (norm.cu: the
// warp butterflies, the 4 warp sums of a 128-wide head as ((s0 + s2) + (s1 + s3)), nothing contracted, a run-time divisor), so
// the outputs are bit-identical to the chain it replaces.
// Several sequences (a verify step of two slots in one batch, --kv-unified): one block row per sequence; their state is the
// GET_ROWS copy (not deferred: one sequence's cache rows may be another's source), the slots are the sequences' cells.
#include "gdn-conv-store.cuh"
#include "ggml-cuda-try-fuse.cuh"
#include "unary.cuh"
#include "ggml-impl.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

// Two switches, and the family switch has to cover this kernel: it replaces the same GDN conv step as the direct
// conv, so with only the direct conv stood down the "chain" reference arm still ran this kernel and the W=1 logits
// diverged from W=8 by 5.88 (measured, gfx1201, test-logits-width-probe P=3584, width_purity FAIL) -- i.e.
// GGML_CUDA_DISABLE_CONV_FUSION did not produce the chain it names.  With this, `GGML_CUDA_DISABLE_CONV_FUSION=1`
// is the chain again (width_purity PASS, the same W8/row-0 hashes as the previous release's build).
static bool gdn_store_enabled() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_FUSE_GDN_CONV_STORE");
        if (e != nullptr && atoi(e) == 0) {
            return false;
        }
        const char * d = getenv("GGML_CUDA_DISABLE_CONV_FUSION");
        return d == nullptr || atoi(d) == 0;
    }();
    return on;
}

static const ggml_tensor * gdn_store_root(const ggml_tensor * t) {
    while (t != nullptr && t->view_src != nullptr) t = t->view_src;
    return t;
}

static bool gdn_store_noop(const ggml_tensor * t) {
    return t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_NONE;
}

static float gdn_store_param_f(const ggml_tensor * t, int k) {
    float v;
    memcpy(&v, (const float *) t->op_params + k, sizeof(float));
    return v;
}

// byte span [lo, hi) of a (possibly strided) tensor; empty for a tensor without data
static void gdn_store_span(const ggml_tensor * t, uintptr_t & lo, uintptr_t & hi) {
    lo = (uintptr_t) t->data;
    size_t last = 0;
    for (int k = 0; k < GGML_MAX_DIMS; ++k) last += (size_t) (t->ne[k] - 1) * t->nb[k];
    hi = lo + last + ggml_type_size(t->type);
}

static bool gdn_store_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    if (a == nullptr || b == nullptr || a->data == nullptr || b->data == nullptr || ggml_nelements(a) == 0 || ggml_nelements(b) == 0) {
        return false;
    }
    uintptr_t alo, ahi, blo, bhi;
    gdn_store_span(a, alo, ahi);
    gdn_store_span(b, blo, bhi);
    return alo < bhi && blo < ahi;
}

// RMS_NORM + SCALE (scale only) over the 128-wide head view of y that starts at channel ch0
static bool gdn_store_norm_one(const ggml_cgraph * cgraph, int j, const ggml_tensor * y, int64_t C, int64_t T, int64_t S, int64_t ch0,
                               int64_t nk, float & eps, float & scale) {
    if (j + 1 >= cgraph->n_nodes) return false;
    const ggml_tensor * rn = cgraph->nodes[j];
    const ggml_tensor * sc = cgraph->nodes[j + 1];
    if (rn->op != GGML_OP_RMS_NORM || sc->op != GGML_OP_SCALE || sc->src[0] != rn || rn->type != GGML_TYPE_F32 ||
            sc->type != GGML_TYPE_F32 || !ggml_is_contiguous(sc)) return false;
    const ggml_tensor * v = rn->src[0];
    if (v == nullptr || v->op != GGML_OP_VIEW || v->view_src != y || v->type != GGML_TYPE_F32) return false;
    if (v->ne[0] != 128 || v->ne[1] != nk || v->ne[2] != T || v->ne[3] != S || (S > 1 && v->nb[3] != (size_t) (T*C)*sizeof(float)) ||
            v->nb[0] != sizeof(float) ||
            v->nb[1] != 128*sizeof(float) || v->nb[2] != (size_t) C*sizeof(float) || v->view_offs != (size_t) ch0*sizeof(float)) return false;
    if (sc->ne[0] != 128 || sc->ne[1] != nk || sc->ne[2] != T || sc->ne[3] != S) return false;
    eps   = gdn_store_param_f(rn, 0);
    scale = gdn_store_param_f(sc, 0);
    return gdn_store_param_f(sc, 1) == 0.0f && eps >= 0.0f;
}

bool ggml_cuda_gdn_store_match_at_concat(const ggml_cgraph * cgraph, int i, ggml_cuda_gdn_store_match & m, bool with_data) {
    if (!gdn_store_enabled() || i < 0 || i + 8 >= cgraph->n_nodes) return false;
    if (!GGML_CUDA_CC_IS_RDNA4(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) return false;
    const ggml_tensor * cc = cgraph->nodes[i];
    if (cc->op != GGML_OP_CONCAT || cc->type != GGML_TYPE_F32 || ggml_get_op_params_i32(cc, 0) != 0) return false;

    // CONCAT(state [3, C, S], x^T [T, C, S]): S sequences of T tokens
    const ggml_tensor * st = cc->src[0];
    const ggml_tensor * tr = cc->src[1];
    if (st == nullptr || tr == nullptr || tr->op != GGML_OP_TRANSPOSE || tr->view_src == nullptr) return false;
    // x^T is a view of the in-projection's output, the root of its views: [C, T*S] (or [C, T, S]) contiguous, so the shape is
    // the transposed view's own [T, C, S]
    const ggml_tensor * x = tr->view_src;
    const int64_t T = tr->ne[0], C = tr->ne[1], S = tr->ne[2];
    if (T < 1 || T > 16 || C % 256 != 0 || S < 1 || S > 8 || tr->ne[3] != 1 || tr->view_offs != 0) return false;
    if (tr->nb[0] != (size_t) C*sizeof(float) || tr->nb[1] != sizeof(float) || (S > 1 && tr->nb[2] != (size_t) (T*C)*sizeof(float))) return false;
    if (x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || x->ne[0] != C || ggml_nelements(x) != C*T*S) return false;
    if (st->type != GGML_TYPE_F32 || !ggml_is_contiguous(st) || st->ne[0] != 3 || st->ne[1] != C || st->ne[2] != S || st->ne[3] != 1) return false;
    if (!ggml_is_contiguous(cc) || cc->ne[0] != T + 3 || cc->ne[1] != C || cc->ne[2] != S || cc->ne[3] != 1) return false;

    // build_conv_state_store: the window view right after the CONCAT, then the slot order {Kw-1, ..., 0} (a view of
    // ARANGE -> SCALE -> CAST over K; built per layer, or once per graph and device with r241's LLAMA_SHARE_SLOT_REV), the
    // slots and the SET_ROWS
    const ggml_tensor * win = cgraph->nodes[i + 1];
    if (win->op != GGML_OP_VIEW || win->view_src != cc || win->type != GGML_TYPE_F32 || win->ne[0] != 3 || win->ne[2] != C ||
            win->ne[3] != S || win->nb[1] != sizeof(float) || win->nb[2] != cc->nb[1] || (S > 1 && win->nb[3] != cc->nb[2]) ||
            win->view_offs % sizeof(float) != 0) return false;
    const int64_t Kw = win->ne[1], col0 = (int64_t) (win->view_offs / sizeof(float));
    if (Kw < 1 || col0 < 0 || col0 + Kw + 1 != T + 2) return false;   // the last window ends at the last column
    int i_sr = -1;
    for (int n = i + 2; n < cgraph->n_nodes && n < i + 12; ++n) {
        if (cgraph->nodes[n]->op == GGML_OP_SET_ROWS && cgraph->nodes[n]->src[0] == win) { i_sr = n; break; }
    }
    if (i_sr < 0) return false;
    const ggml_tensor * sr = cgraph->nodes[i_sr];
    const ggml_tensor * so = sr->src[1];
    const ggml_tensor * sl = sr->src[2];
    const ggml_tensor * ca = so != nullptr ? so->view_src : nullptr;
    const ggml_tensor * sb = ca != nullptr ? ca->src[0] : nullptr;
    const ggml_tensor * ar = sb != nullptr ? sb->src[0] : nullptr;
    if (so == nullptr || so->op != GGML_OP_VIEW || ca == nullptr || ca->op != GGML_OP_CPY || ca->type != GGML_TYPE_I32 ||
            sb == nullptr || sb->op != GGML_OP_SCALE || sb->type != GGML_TYPE_F32 || ar == nullptr || ar->op != GGML_OP_ARANGE ||
            ar->type != GGML_TYPE_F32 || ar->ne[1] != 1 || ar->ne[2] != 1 || ar->ne[3] != 1) return false;
    const int64_t K = ar->ne[0];
    if (K < Kw || gdn_store_param_f(ar, 0) != 0.0f || gdn_store_param_f(ar, 1) != (float) K || gdn_store_param_f(ar, 2) != 1.0f) return false;
    if (sb->ne[0] != K || gdn_store_param_f(sb, 0) != -1.0f || gdn_store_param_f(sb, 1) != (float) (K - 1) || ca->ne[0] != K) return false;
    if (so->type != GGML_TYPE_I32 || so->ne[0] != Kw || so->ne[1] != 1 || so->view_offs != (size_t) (K - Kw) * sizeof(int32_t)) return false;
    const ggml_tensor * cache = gdn_store_root(sl);
    if (sl == nullptr || sl->op != GGML_OP_VIEW || cache == nullptr || cache->op != GGML_OP_NONE || sl->type != GGML_TYPE_F32 || sl->ne[0] != 3 ||
            sl->ne[1] != K || sl->ne[2] != C || sl->ne[3] != S || sl->nb[0] != sizeof(float) || sl->nb[2] != 3*sizeof(float)) return false;
    // between the window view and the SET_ROWS only the store's own nodes; the slot order chain is the layer's own (all three
    // in there, each read once) or a shared one, which is left to run for the later layers
    int n_chain = 0;
    for (int n = i + 2; n < i_sr; ++n) {
        const ggml_tensor * t = cgraph->nodes[n];
        if (t == ar || t == sb || t == ca) { ++n_chain; continue; }
        if (t != so && t != sl) return false;
    }
    bool private_chain = false;
    if (n_chain == 3) {
        int i_ar = -1, i_sb = -1, i_ca = -1;
        for (int n = i + 2; n < i_sr; ++n) {
            if (cgraph->nodes[n] == ar) i_ar = n;
            if (cgraph->nodes[n] == sb) i_sb = n;
            if (cgraph->nodes[n] == ca) i_ca = n;
        }
        private_chain = ggml_node_get_use_count(cgraph, i_ar) == 1 && ggml_node_get_use_count(cgraph, i_sb) == 1 &&
                        ggml_node_get_use_count(cgraph, i_ca) == 1;
    }
    // a shared chain: the CONCAT takes only itself and the window view, and the SET_ROWS takes itself and the no-op after it
    if (!private_chain && (i_sr + 1 >= cgraph->n_nodes || !gdn_store_noop(cgraph->nodes[i_sr + 1]))) return false;

    // the conv: nothing between the SET_ROWS and it reads the concat or its views
    int conv = -1;
    for (int n = i_sr + 1; n < cgraph->n_nodes && n < i_sr + 24; ++n) {
        const ggml_tensor * t = cgraph->nodes[n];
        if (t->op == GGML_OP_SSM_CONV && t->src[0] == cc) { conv = n; break; }
        for (int s = 0; s < GGML_MAX_SRC && t->src[s]; ++s) {
            const ggml_tensor * r = gdn_store_root(t->src[s]);
            if (r == cc || (private_chain && r == ca) || t->src[s] == x) return false;
        }
    }
    if (conv < 0 || conv + 1 >= cgraph->n_nodes) return false;
    const ggml_tensor * cv = cgraph->nodes[conv];
    const ggml_tensor * w  = cv->src[1];
    if (cv->type != GGML_TYPE_F32 || !ggml_is_contiguous(cv) || cv->ne[0] != C || cv->ne[1] != T || cv->ne[2] != S) return false;
    if (w == nullptr || w->type != GGML_TYPE_F32 || !ggml_is_contiguous(w) || w->ne[0] != 4 || w->ne[1] != C) return false;
    ggml_tensor * y = cgraph->nodes[conv + 1];
    if (y->op != GGML_OP_UNARY || ggml_get_unary_op(y) != GGML_UNARY_OP_SILU || y->src[0] != cv || y->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(y)) return false;

    // the q / k l2 norm pair over y
    int j = conv + 2;
    while (j < cgraph->n_nodes && gdn_store_noop(cgraph->nodes[j])) ++j;
    if (j >= cgraph->n_nodes || cgraph->nodes[j]->op != GGML_OP_RMS_NORM || cgraph->nodes[j]->src[0] == nullptr) return false;
    const int64_t nk = cgraph->nodes[j]->src[0]->ne[1];
    if (nk < 1 || 2*nk*128 > C) return false;
    float eps0, sc0, eps1, sc1;
    if (!gdn_store_norm_one(cgraph, j, y, C, T, S, 0, nk, eps0, sc0)) return false;
    int j2 = j + 2;
    while (j2 < cgraph->n_nodes && gdn_store_noop(cgraph->nodes[j2])) ++j2;
    if (!gdn_store_norm_one(cgraph, j2, y, C, T, S, nk*128, nk, eps1, sc1) || eps0 != eps1 || sc0 != sc1) return false;
    // the conv output is read by the SILU only, the RMS_NORM outputs by their SCALEs only
    for (int n = conv + 2; n < cgraph->n_nodes; ++n) {
        const ggml_tensor * t = cgraph->nodes[n];
        for (int s = 0; s < GGML_MAX_SRC && t->src[s]; ++s) {
            const ggml_tensor * r = t->src[s];
            if (r == cv || r->view_src == cv) return false;
            if (n != j + 1 && n != j2 + 1 && (r == cgraph->nodes[j] || r == cgraph->nodes[j2] ||
                    r->view_src == cgraph->nodes[j] || r->view_src == cgraph->nodes[j2])) return false;
        }
    }

    const ggml_tensor * gather = gdn_store_root(st);
    if (gather == nullptr || gather->op != GGML_OP_GET_ROWS || gdn_store_root(gather->src[0]) != cache) gather = nullptr;

    if (with_data) {
        const ggml_tensor * q_out = cgraph->nodes[j + 1], * k_out = cgraph->nodes[j2 + 1];
        if (!st->data || !x->data || !w->data || !y->data || !q_out->data || !k_out->data || !sl->data) return false;
        // what the kernel writes overlaps nothing it reads, nor each other
        for (const ggml_tensor * d : { (const ggml_tensor *) y, q_out, k_out, sl }) {
            for (const ggml_tensor * s : { st, x, w }) {
                if (gdn_store_overlap(d, s)) return false;
            }
        }
        if (gdn_store_overlap(y, q_out) || gdn_store_overlap(y, k_out) || gdn_store_overlap(q_out, k_out) ||
                gdn_store_overlap(y, sl) || gdn_store_overlap(q_out, sl) || gdn_store_overlap(k_out, sl)) return false;
        // the SET_ROWS writes move to the conv: nothing in between may touch the slots or write x / the state
        for (int n = i + 2; n < conv; ++n) {
            const ggml_tensor * t = cgraph->nodes[n];
            if (n == i_sr || gdn_store_noop(t)) continue;
            if (gdn_store_overlap(t, sl) || gdn_store_overlap(t, x) || gdn_store_overlap(t, st) || gdn_store_overlap(t, w)) return false;
            for (int s = 0; s < GGML_MAX_SRC && t->src[s]; ++s) {
                if (gdn_store_overlap(t->src[s], sl)) return false;
            }
        }
    }

    m.concat = i; m.set_rows = i_sr; m.conv = conv; m.last = j2 + 1; m.private_chain = private_chain;
    m.gather = gather; m.state = st; m.x = x; m.w = w; m.y = y;
    m.q_out = cgraph->nodes[j + 1]; m.k_out = cgraph->nodes[j2 + 1]; m.slots = sl;
    m.C = C; m.T = T; m.S = S; m.K = K; m.Kw = Kw; m.col0 = col0;
    m.nk = (int) nk; m.eps = eps0; m.scale = sc0;
    return true;
}

// One thread a channel, 256 a block, one block row a sequence; the q / k blocks (channels below 2 nk 128, a multiple of 256)
// hold two whole heads.
// state and slots may be one row (the deferred gather reads the cache row that slot 0 writes back): no __restrict__ on them,
// and each thread reads its three state values before it writes any slot.
static __global__ void __launch_bounds__(256) gdn_conv_store_kernel(
        const float * state, const int32_t * __restrict__ state_ids, const int64_t state_row,
        const float * __restrict__ x, const float * __restrict__ w, float * __restrict__ y,
        float * __restrict__ q_out, float * __restrict__ k_out, char * slots, const int64_t slot_nb1, const int64_t slot_nb3,
        const int Kw, const int col0, const int C, const int T, const int nk, const int ncols, const float eps, const float scale_out) {
    const int c = blockIdx.x * 256 + threadIdx.x;
    const int s = blockIdx.y;   // the sequence: its state row, its T tokens, its cell's slots
    if (state_ids != nullptr) {
        state += (int64_t) state_ids[0] * state_row;   // deferred gather: one sequence only
    } else {
        state += (int64_t) s * 3 * C;
    }
    x     += (size_t) s * T * C;
    y     += (size_t) s * T * C;
    q_out += (size_t) s * T * nk * 128;
    k_out += (size_t) s * T * nk * 128;
    slots += (int64_t) s * slot_nb3;
    const float s0 = state[c * 3 + 0], s1 = state[c * 3 + 1], s2 = state[c * 3 + 2];
    const auto col = [&](const int jp) -> float {
        return jp == 0 ? s0 : (jp == 1 ? s1 : (jp == 2 ? s2 : x[(size_t) (jp - 3) * C + c]));
    };

    // window v (columns col0 + v .. col0 + v + 2 of the concatenated input) is rollback slot Kw - 1 - v
    for (int v = 0; v < Kw; ++v) {
        float * d = (float *) (slots + (int64_t) (Kw - 1 - v) * slot_nb1) + (size_t) c * 3;
        d[0] = col(col0 + v + 0);
        d[1] = col(col0 + v + 1);
        d[2] = col(col0 + v + 2);
    }

    float wr[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) wr[k] = w[c * 4 + k];
    float xw[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) xw[k] = col(k);

    const bool qk = blockIdx.x * 256 < 2 * nk * 128;   // block-uniform
    const float b = 0.0f;
    for (int t = 0; t < T; ++t) {
        float sumf = 0.0f;
#pragma unroll
        for (int k = 0; k < 4; ++k) sumf += xw[k] * wr[k];
        sumf += b;
        const float yv = ggml_cuda_op_silu_single(sumf);
        y[(size_t) t * C + c] = yv;
        if (qk) {
            // the head's 32-lane sums of squares, then its 4 wave sums in lanes 0..3 (0 elsewhere) butterflied again.  As in
            // rms_norm_scale_pair_f32<256> (norm.cu): nothing contracted (0 + y*y would fuse with the first butterfly add) and a
            // run-time divisor (sum / 128 as a constant becomes sum * 2^-7, which would fuse with + eps)
#pragma clang fp contract(off)
            __shared__ float s_w[256/WARP_SIZE];
            const int lane = threadIdx.x % WARP_SIZE, warp = threadIdx.x / WARP_SIZE;
            float tmp = 0.0f;
            tmp += yv * yv;
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
            const float mean  = val / ncols;
            const float scale = rsqrtf(mean + eps);
            const int   is_k  = c >= nk*128;
            const int   h     = c / 128 - is_k*nk;
            float * dst = is_k ? k_out : q_out;
            dst[((size_t) t*nk + h)*128 + c % 128] = scale_out * (scale * yv);
            __syncthreads();
        }
        xw[0] = xw[1]; xw[1] = xw[2]; xw[2] = xw[3];
        const int jp = t + 4;
        xw[3] = (jp - 3) < T ? x[(size_t) (jp - 3) * C + c] : 0.0f;
    }
}

static void gdn_store_launch(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_store_match & m) {
    // GGML_CUDA_CONV_DEBUG: one line per process for each sequence count S, the same switch and shape as the direct
    // conv's GDN_CONV_DIRECT (gdn-conv.cu).  Without it a run of this path cannot be told from the chain, i.e. the gate
    // would be vacuous; per S, so that a multi-sequence ubatch shows up next to the single-sequence steps.
    static const bool dbg = getenv("GGML_CUDA_CONV_DEBUG") != nullptr;
    static bool seen[9] = {};
    if (dbg && !seen[m.S]) { fprintf(stderr, "GDN_CONV_STORE C=%lld T=%lld S=%lld Kw=%lld col0=%lld K=%lld private_chain=%d\n",
        (long long) m.C, (long long) m.T, (long long) m.S, (long long) m.Kw, (long long) m.col0, (long long) m.K, (int) m.private_chain); seen[m.S] = true; }
    const float *   state     = (const float *) m.state->data;
    const int32_t * state_ids = nullptr;
    int64_t         state_row = 0;
    if (m.gather != nullptr &&
            std::find(ctx.gdn_deferred_gathers.begin(), ctx.gdn_deferred_gathers.end(), m.gather) != ctx.gdn_deferred_gathers.end()) {
        GGML_ASSERT(m.state->data == m.gather->data);
        state     = (const float *) m.gather->src[0]->data;
        state_ids = (const int32_t *) m.gather->src[1]->data;
        state_row = (int64_t) (m.gather->src[0]->nb[1] / sizeof(float));
    }
    GGML_ASSERT(state_ids == nullptr || m.S == 1);
    gdn_conv_store_kernel<<<dim3((unsigned) (m.C / 256), (unsigned) m.S), 256, 0, ctx.stream()>>>(state, state_ids, state_row,
        (const float *) m.x->data, (const float *) m.w->data, (float *) m.y->data, (float *) m.q_out->data, (float *) m.k_out->data,
        (char *) m.slots->data, (int64_t) m.slots->nb[1], (int64_t) m.slots->nb[3], (int) m.Kw, (int) m.col0, (int) m.C, (int) m.T, m.nk, 128, m.eps, m.scale);
    CUDA_CHECK(cudaGetLastError());
}

int ggml_cuda_gdn_store_try_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    if (!gdn_store_enabled()) {
        return 0;
    }
    const ggml_tensor * node = cgraph->nodes[i];
    if (node->op == GGML_OP_CONCAT) {
        // the CONCAT and the window view launch nothing (with the layer's own slot order chain: through the SET_ROWS); the
        // conv writes the slots
        ggml_cuda_gdn_store_match m;
        if (!ggml_cuda_gdn_store_match_at_concat(cgraph, i, m, true)) return 0;
        return m.private_chain ? m.set_rows - i : 1;
    }
    if (node->op == GGML_OP_SET_ROWS && node->src[0] != nullptr && node->src[0]->op == GGML_OP_VIEW) {
        // the SET_ROWS of a store whose shared slot order chain ran: it and the no-op after it launch nothing
        const ggml_tensor * cc = node->src[0]->view_src;
        if (cc == nullptr || cc->op != GGML_OP_CONCAT) return 0;
        for (int j = i - 1; j >= 0 && j >= i - 12; --j) {
            if (cgraph->nodes[j] != cc) continue;
            ggml_cuda_gdn_store_match m;
            if (!ggml_cuda_gdn_store_match_at_concat(cgraph, j, m, true) || m.set_rows != i) return 0;
            GGML_ASSERT(!m.private_chain);
            return 1;
        }
        return 0;
    }
    if (node->op == GGML_OP_SSM_CONV) {
        for (int j = i - 1; j >= 0 && j >= i - 32; --j) {
            if (cgraph->nodes[j] != node->src[0]) continue;
            ggml_cuda_gdn_store_match m;
            if (!ggml_cuda_gdn_store_match_at_concat(cgraph, j, m, true)) return 0;
            GGML_ASSERT(m.conv == i);
            gdn_store_launch(ctx, m);
            return m.last - i;   // the SILU and the norm pair too
        }
    }
    return 0;
}

bool ggml_cuda_gdn_store_gather_deferrable(const ggml_cgraph * cgraph, int node_idx, const ggml_cuda_gdn_gather_index & idx) {
    if (!gdn_store_enabled()) {
        return false;
    }
    const ggml_tensor * gr = cgraph->nodes[node_idx];
    const auto g = idx.gathers.find(gr);
    if (g == idx.gathers.end() || g->second.n_gdn != 0 || g->second.n_concat != 1 || g->second.other_reader ||
            g->second.concat < node_idx) {
        return false;
    }
    ggml_cuda_gdn_store_match m;
    // one sequence only: with several, one sequence's cache rows may be the source of another's state
    if (!ggml_cuda_gdn_store_match_at_concat(cgraph, g->second.concat, m, true) || m.S != 1 || m.gather != gr || m.state->data != gr->data ||
            ggml_nelements(gr) != 3 * m.C || ggml_nrows(gr) != 1 || gr->src[1] == nullptr || gr->src[1]->ne[0] != 1) {
        return false;
    }
    // nothing but the folded SET_ROWS writes the cache between the gather and the conv
    const auto wr = idx.writers.find(gdn_store_root(gr->src[0]));
    if (wr != idx.writers.end()) {
        for (const int j : wr->second) {
            if (j > node_idx && j < m.conv && j != m.set_rows) {
                return false;
            }
        }
    }
    return true;
}
