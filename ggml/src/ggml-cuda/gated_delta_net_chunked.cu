#include "gated_delta_net_chunked.cuh"
#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

// ---------------------------------------------------------------------------------------------
// Fused chunked GGML_OP_GATED_DELTA_NET for prefill (n_tokens > 1, K == 1, non-KDA).
//
// The chunked recurrence is sequential in the state: chunk c consumes the state chunk c-1
// produced. Two launches handle that without a cross-block dependency:
//
//   mode 0 (state scan):  one block per (v-head, seq), looping the chunks, carrying the state
//                         through a per-chunk scratch buffer. Computes the KKT inverse A, the
//                         predicted-v correction and the state update; writes the state after
//                         each chunk to scratch[c] (the final one straight to state_out).
//   mode 1 (output):      one block per (v-head, seq), looping the chunks, reading the state
//                         at each chunk start from scratch[c-1] and writing the attention
//                         output o (and only o). The state work is recomputed here so the
//                         output pass is embarrassingly parallel over (head, seq, chunk).
//
// The per-chunk math is the algebra of llm_build_delta_net_base::build_delta_net_chunking,
// with gcs the chunk-local inclusive cumsum of the gate, K_b = K*beta, V_b = V*beta,
// Q_s = scale*Q, and decay[i][j] = e^{gcs[j]-gcs[i]} on the upper triangle (i <= j):
//
//     L    = strict_upper(K^T K_b . decay)                 (the gram, masked)
//     A    = (I + L)^-1                                    (unit upper; back substitution)
//     KQ   = upper_diag((K^T Q_s) . decay)
//     U    = K_b^T S
//     v_n  = V_b^T A - diag(e^g) A^T U                     (v_new = v_corr - predicted)
//     o    = e^g . S^T Q_s + KQ^T v_n
//     S'   = e^{g_last} S + K diag(e^{g_last - g}) v_n
//
// Decay is computed DIRECTLY per kept (i, j) pair, not split around a midpoint: the test
// harness feeds gates as low as -20 per token, so a chunk's cumsum can span thousands and any
// split-form half clamps. The direct exponent gcs[j]-gcs[i] is <= 0 on every kept pair (gates
// are negative; padding keeps the cumsum flat), so it can never overflow. Rows/columns past
// the sequence end are handled by clamping the token READ to the last real token (zero-padding
// would read out of bounds) and by bt[pad] = el[pad] = 0, which zero the gram entries and the
// state-update factors the way ggml_pad's zero columns do.
//
// Layouts (fp32, matching the sequential kernel):
//   q, k:  [S_v, H_k, n_tokens, n_seqs] contiguous rows, q/k share strides
//   v:     [S_v, H,   n_tokens, n_seqs] contiguous rows
//   g,beta:[1,  H,   n_tokens, n_seqs] contiguous (non-KDA scalar-per-head gate)
//   state: [S_v, S_v, H, n_seqs] contiguous, stored transposed: state[v*S_v + k] = S[k][v]
//   out:   attn [S_v, H, n_tokens, n_seqs] followed by the new state (K == 1: one slot)
//   scratch: [n_chunks][H][n_seqs][S_v*S_v] fp32, the per-chunk states (mode 0 writes, mode 1 reads)
//
// The v-head h reads q/k from k-head h % H_k (fastmodulo on the same magics as the sequential
// kernel) and its g/beta from head h.
// ---------------------------------------------------------------------------------------------

#define GDN_CHUNKED_CS 64
#define GDN_CHUNKED_NTHREADS 256
#define GDN_CHUNKED_VP 64   // v-rows per pass in phase 5 (U is [CS, 64] in LDS; 2 passes at S_v=128)

struct gdn_chunked_smem {
    float LU[GDN_CHUNKED_CS][GDN_CHUNKED_CS];  // gram L (phases 2-3), then U / v_new (phase 5)
    float A[GDN_CHUNKED_CS][GDN_CHUNKED_CS];   // the KKT inverse (unit upper)
    float KQ[GDN_CHUNKED_CS][GDN_CHUNKED_CS];  // intra-chunk attention gram (output pass only)
    // gate arrays (per chunk, per head)
    float gc[GDN_CHUNKED_CS];   // raw gate, pre-cumsum
    float gcs[GDN_CHUNKED_CS];  // chunk-local inclusive cumsum
    float eg[GDN_CHUNKED_CS];   // e^{gcs} (<= 1: gates are negative)
    float el[GDN_CHUNKED_CS];   // e^{g_last - gcs} (<= 1; 0 on padding)
    float bt[GDN_CHUNKED_CS];   // beta
};

template <int S_v>
__global__ static void __launch_bounds__(GDN_CHUNKED_NTHREADS)
gated_delta_net_chunked_cuda(
        const float * __restrict__ q,
        const float * __restrict__ k,
        const float * __restrict__ v,
        const float * __restrict__ g,
        const float * __restrict__ beta,
        const float * __restrict__ state_in,
        float * __restrict__ attn_out,
        float * __restrict__ state_out,
        float * __restrict__ scratch,
        int mode,   // 0 = state scan, 1 = output
        int64_t H, int64_t Hg, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        const uint3 neqk1_magic, const uint3 rq3_magic,
        float scale, int64_t state_seq_stride)
{
    __shared__ gdn_chunked_smem s;
    const int tid = threadIdx.x;
    const int h   = blockIdx.x;   // v-head
    const int seq = blockIdx.y;   // sequence (uniform length: no cu needed)

    const int n_chunks = (int) ((n_tokens + GDN_CHUNKED_CS - 1) / GDN_CHUNKED_CS);

    // q/k head for this v-head and the q/k seq index (matches the sequential kernel)
    const int64_t hq  = fastmodulo((uint32_t) h, neqk1_magic);
    const int64_t iq3 = fastdiv((uint32_t) seq, rq3_magic);

    // per-chunk state pointers: chunk c reads scratch[c-1] (or state_in for c == 0) and,
    // in scan mode, writes scratch[c] (state_out for the last chunk)
    const int64_t state_seq_base = (int64_t) seq * state_seq_stride + (int64_t) h * S_v * S_v;
    const int64_t scratch_seq_base = ((int64_t) h * n_seqs + seq) * S_v * S_v;

    ggml_cuda_pdl_sync();

    for (int c = 0; c < n_chunks; ++c) {
        const int c0 = c * GDN_CHUNKED_CS;
        const int nval = min(GDN_CHUNKED_CS, (int) n_tokens - c0);
        const int tok_lim = nval - 1;
#define TK(t) min((t), (tok_lim))

        const float * kbase = k + iq3 * sq3 + (int64_t) c0 * sq2 + hq * sq1;
        const float * qbase = q + iq3 * sq3 + (int64_t) c0 * sq2 + hq * sq1;
        const float * vbase = v + (int64_t) seq * sv3 + (int64_t) c0 * sv2 + h * sv1;
        const float * gbase = g + ((int64_t) seq * n_tokens + c0) * H + h;
        const float * bbase = beta + ((int64_t) seq * n_tokens + c0) * H + h;

        const float * sbase = (c == 0)
            ? state_in + state_seq_base
            : scratch + ((int64_t) (c - 1) * H * n_seqs) * (int64_t) (S_v * S_v) + scratch_seq_base;
        float * sobase = nullptr;   // scan mode: where this chunk's state goes
        if (mode == 0) {
            sobase = (c == n_chunks - 1)
                ? state_out + state_seq_base
                : scratch + ((int64_t) c * H * n_seqs) * (int64_t) (S_v * S_v) + scratch_seq_base;
        }
        float * obase = attn_out + ((int64_t) seq * n_tokens + c0) * H * S_v + h * S_v;

        // ---- phase 1: gate, beta, chunk-local inclusive cumsum ----------------------------
        if (tid < GDN_CHUNKED_CS) {
            if (tid < nval) {
                s.gc[tid] = gbase[(int64_t) tid * H];
                s.bt[tid] = bbase[(int64_t) tid * H];
            } else {
                s.gc[tid] = 0.0f;   // ggml_pad semantics: zero-padded gate keeps the cumsum flat
                s.bt[tid] = 0.0f;
            }
        }
        __syncthreads();
        if (tid == 0) {
            float acc = 0.0f;
            for (int t = 0; t < GDN_CHUNKED_CS; ++t) { acc += s.gc[t]; s.gcs[t] = acc; }
        }
        __syncthreads();
        if (tid < GDN_CHUNKED_CS) {
            const float gc = s.gcs[tid];
            s.eg[tid] = __expf(gc);
            // el feeds the state update S' += el[t] * K[k][t] * v_new[t][v]. K reads are clamped
            // to the last real token (zero-padding would read OOB), so the padding factor must
            // be 0 or the clamped K columns leak into the state.
            s.el[tid] = (tid < nval) ? __expf(s.gcs[GDN_CHUNKED_CS - 1] - gc) : 0.0f;
        }
        __syncthreads();

        // ---- phase 2: L[i][j] = (i<j) ? bt[j] * e^{gcs[j]-gcs[i]} * sum_d K[d][i] K[d][j] : 0
        // The exponent is <= 0 on every kept pair (gates negative, padding cumsum flat), so the
        // direct form cannot overflow no matter how far the gate span runs.
        {
            const int i0 = (tid & 15) * 4;
            const int j0 = (tid >> 4) * 4;
            float acc[4][4];
            float bj[4];
#pragma unroll
            for (int f = 0; f < 4; ++f) bj[f] = s.bt[j0 + f];
#pragma unroll
            for (int e = 0; e < 4; ++e)
#pragma unroll
                for (int f = 0; f < 4; ++f) acc[e][f] = 0.0f;
            for (int d0 = 0; d0 < S_v; d0 += 4) {
                float ki[4][4], kj[4][4];
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    *(float4*) &ki[e][0] = *(const float4 *) (kbase + (int64_t) d0 + (int64_t) TK(i0 + e) * sq2);
#pragma unroll
                for (int f = 0; f < 4; ++f)
                    *(float4*) &kj[f][0] = *(const float4 *) (kbase + (int64_t) d0 + (int64_t) TK(j0 + f) * sq2);
#pragma unroll
                for (int e = 0; e < 4; ++e)
#pragma unroll
                    for (int f = 0; f < 4; ++f)
#pragma unroll
                        for (int dd = 0; dd < 4; ++dd)
                            acc[e][f] += ki[e][dd] * kj[f][dd];   // pure gram; beta applied once at the store
            }
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int i = i0 + e;
#pragma unroll
                for (int f = 0; f < 4; ++f) {
                    const int j = j0 + f;
                    s.LU[i][j] = (i < j) ? bj[f] * __expf(s.gcs[j] - s.gcs[i]) * acc[e][f] : 0.0f;
                }
            }
        }
        __syncthreads();

        // ---- phase 3: A = (I+L)^-1, unit upper, by back substitution, lane per column --------
        // (I+L)x = e_col: bottom-up, x[i] = (i==col) - sum_{j>i} L[i][j] x[j]. The full column is
        // computed (upper entries included); the mask in the consumers is implicit in A itself.
        if (tid < GDN_CHUNKED_CS) {
            const int col = tid;
            float x[GDN_CHUNKED_CS];
#pragma unroll
            for (int i = GDN_CHUNKED_CS - 1; i >= 0; --i) {
                float acc = 0.0f;
#pragma unroll
                for (int j = i + 1; j < GDN_CHUNKED_CS; ++j)
                    acc += s.LU[i][j] * x[j];
                x[i] = ((i == col) ? 1.0f : 0.0f) - acc;
            }
#pragma unroll
            for (int i = 0; i < GDN_CHUNKED_CS; ++i)
                s.A[i][col] = x[i];
        }
        __syncthreads();

        // ---- phase 4 (output pass only): KQ[i][j] = (i<=j) ? e^{gcs[j]-gcs[i]} * sum_d K[d][i] (Q[d][j]*scale) : 0
        if (mode == 1) {
            const int i0 = (tid & 15) * 4;
            const int j0 = (tid >> 4) * 4;
            float acc[4][4];
#pragma unroll
            for (int e = 0; e < 4; ++e)
#pragma unroll
                for (int f = 0; f < 4; ++f) acc[e][f] = 0.0f;
            for (int d0 = 0; d0 < S_v; d0 += 4) {
                float ki[4][4], qj[4][4];
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    *(float4*) &ki[e][0] = *(const float4 *) (kbase + (int64_t) d0 + (int64_t) TK(i0 + e) * sq2);
#pragma unroll
                for (int f = 0; f < 4; ++f)
                    *(float4*) &qj[f][0] = *(const float4 *) (qbase + (int64_t) d0 + (int64_t) TK(j0 + f) * sq2);
#pragma unroll
                for (int e = 0; e < 4; ++e)
#pragma unroll
                    for (int f = 0; f < 4; ++f)
#pragma unroll
                        for (int dd = 0; dd < 4; ++dd)
                            acc[e][f] += ki[e][dd] * (qj[f][dd] * scale);
            }
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int i = i0 + e;
#pragma unroll
                for (int f = 0; f < 4; ++f) {
                    const int j = j0 + f;
                    s.KQ[i][j] = (i <= j) ? __expf(s.gcs[j] - s.gcs[i]) * acc[e][f] : 0.0f;
                }
            }
        }
        __syncthreads();

        // ---- phase 5: per v-pass (S_v/VP passes; U holds [CS, VP] at a time) -------------------
        const int npass = (S_v + GDN_CHUNKED_VP - 1) / GDN_CHUNKED_VP;
        for (int pass = 0; pass < npass; ++pass) {
            const int vrow = pass * GDN_CHUNKED_VP;
            const int vrows = min(GDN_CHUNKED_VP, S_v - vrow);
            const int t0 = (tid & 15) * 4;
            const int v0 = (tid >> 4) * 4;
            const bool vlive = (v0 < vrows);

            // 5a: U[j][v] = sum_k K_b[k][j] * S[k][v]   -> s.LU[j][v]
            if (vlive) {
                float acc[4][4];
                float bj[4];
#pragma unroll
                for (int f = 0; f < 4; ++f) bj[f] = s.bt[t0 + f];
#pragma unroll
                for (int e = 0; e < 4; ++e)
#pragma unroll
                    for (int f = 0; f < 4; ++f) acc[e][f] = 0.0f;
                for (int d0 = 0; d0 < S_v; d0 += 4) {
                    float kj[4][4], sv[4][4];
#pragma unroll
                    for (int f = 0; f < 4; ++f)
                        *(float4*) &kj[f][0] = *(const float4 *) (kbase + (int64_t) d0 + (int64_t) TK(t0 + f) * sq2);
#pragma unroll
                    for (int e = 0; e < 4; ++e)
                        *(float4*) &sv[e][0] = *(const float4 *) (sbase + (int64_t) (vrow + v0 + e) * S_v + d0);
#pragma unroll
                    for (int e = 0; e < 4; ++e)
#pragma unroll
                        for (int f = 0; f < 4; ++f)
#pragma unroll
                            for (int dd = 0; dd < 4; ++dd)
                                acc[e][f] += sv[e][dd] * (kj[f][dd] * bj[f]);
                }
#pragma unroll
                for (int e = 0; e < 4; ++e)
#pragma unroll
                    for (int f = 0; f < 4; ++f)
                        s.LU[t0 + f][v0 + e] = acc[e][f];
            }
            __syncthreads();

            // 5b: v_new[t][v] = sum_j A[j][t] V_b[j][v] - sum_j eg[j] A[j][t] U[j][v]
            // (A upper kills j > t; sign folded so the stored value is +v_new)
            float acc[4][4];
#pragma unroll
            for (int e = 0; e < 4; ++e)
#pragma unroll
                for (int f = 0; f < 4; ++f) acc[e][f] = 0.0f;
            if (vlive) {
                for (int j = 0; j < GDN_CHUNKED_CS; ++j) {
                    float arow[4], vv[4];
                    *(float4*) &arow[0] = *(const float4 *) &s.A[j][t0];
                    *(float4*) &vv[0]   = *(const float4 *) (vbase + (int64_t) TK(j) * sv2 + (vrow + v0));
                    const float bj = s.bt[j];
#pragma unroll
                    for (int e = 0; e < 4; ++e)
#pragma unroll
                        for (int f = 0; f < 4; ++f)
                            acc[e][f] += arow[e] * (vv[f] * bj);
                }
                for (int j = 0; j < GDN_CHUNKED_CS; ++j) {
                    float arow[4], urow[4];
                    *(float4*) &arow[0] = *(const float4 *) &s.A[j][t0];
                    *(float4*) &urow[0] = *(const float4 *) &s.LU[j][v0];
                    const float ej = s.eg[j];
#pragma unroll
                    for (int e = 0; e < 4; ++e)
#pragma unroll
                        for (int f = 0; f < 4; ++f)
                            acc[e][f] -= ej * arow[e] * urow[f];
                }
            }
            // every thread's last read of s.LU (the U data) must land before any thread
            // overwrites it with v_new: the barrier is OUTSIDE the vlive guard so every thread
            // reaches it (a barrier inside a divergent branch is UB)
            __syncthreads();
            if (vlive) {
#pragma unroll
                for (int e = 0; e < 4; ++e)
#pragma unroll
                    for (int f = 0; f < 4; ++f)
                        s.LU[t0 + e][v0 + f] = acc[e][f];
            }
            __syncthreads();

            // 5c (output pass only): o[v][t] = sum_k S[k][v] (Q_s[k][t] eg[t]) - sum_j KQ[j][t] v_new[j][v]
            if (mode == 1 && vlive) {
                float acc[4][4];
#pragma unroll
                for (int e = 0; e < 4; ++e)
#pragma unroll
                    for (int f = 0; f < 4; ++f) acc[e][f] = 0.0f;
                for (int d0 = 0; d0 < S_v; d0 += 4) {
                    float sv[4][4], qk[4][4];
#pragma unroll
                    for (int f = 0; f < 4; ++f)
                        *(float4*) &sv[f][0] = *(const float4 *) (sbase + (int64_t) (vrow + v0 + f) * S_v + d0);
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        *(float4*) &qk[e][0] = *(const float4 *) (qbase + (int64_t) d0 + (int64_t) TK(t0 + e) * sq2);
                        const float eg_ = s.eg[t0 + e] * scale;
#pragma unroll
                        for (int dd = 0; dd < 4; ++dd) qk[e][dd] *= eg_;
                    }
#pragma unroll
                    for (int e = 0; e < 4; ++e)
#pragma unroll
                        for (int f = 0; f < 4; ++f)
#pragma unroll
                            for (int dd = 0; dd < 4; ++dd)
                                acc[e][f] += qk[e][dd] * sv[f][dd];
                }
                for (int j = 0; j < GDN_CHUNKED_CS; ++j) {
                    float kqrow[4], vnrow[4];
                    *(float4*) &kqrow[0] = *(const float4 *) &s.KQ[j][t0];
                    *(float4*) &vnrow[0] = *(const float4 *) &s.LU[j][v0];
#pragma unroll
                    for (int e = 0; e < 4; ++e)
#pragma unroll
                        for (int f = 0; f < 4; ++f)
                            acc[e][f] += kqrow[e] * vnrow[f];   // o = attn_inter + KQ^T v_new
                }
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int t = t0 + e;
                    if (t >= nval) break;
#pragma unroll
                    for (int f = 0; f < 4; ++f)
                        obase[(int64_t) t * H * S_v + (vrow + v0 + f)] = acc[e][f];
                }
            }

            // 5d (scan pass only): S'[k][v] = e^{g_last} S[k][v] - sum_t el[t] K[k][t] v_new[t][v]
            if (mode == 0 && vlive) {
                const float elast = s.eg[GDN_CHUNKED_CS - 1];   // e^{g_last}
#pragma unroll
                for (int kt = 0; kt < (S_v + 63) / 64; ++kt) {
                    const int k0 = (tid & 15) * 4 + kt * 64;
                    if (k0 >= S_v) continue;
                    float acc[4][4];
#pragma unroll
                    for (int e = 0; e < 4; ++e)
#pragma unroll
                        for (int f = 0; f < 4; ++f) acc[e][f] = 0.0f;
                    for (int t = 0; t < GDN_CHUNKED_CS; ++t) {
                        float kk[4], vn[4];
                        *(float4*) &kk[0] = *(const float4 *) (kbase + (int64_t) k0 + (int64_t) TK(t) * sq2);
                        *(float4*) &vn[0] = *(const float4 *) &s.LU[t][v0];
                        const float elt = s.el[t];
#pragma unroll
                        for (int e = 0; e < 4; ++e)
#pragma unroll
                            for (int f = 0; f < 4; ++f)
                                acc[e][f] += elt * kk[e] * vn[f];
                    }
#pragma unroll
                    for (int f = 0; f < 4; ++f) {
                        float sv[4];
                        *(float4*) &sv[0] = *(const float4 *) (sbase + (int64_t) (vrow + v0 + f) * S_v + k0);
#pragma unroll
                        for (int e = 0; e < 4; ++e) acc[e][f] += elast * sv[e];
                    }
#pragma unroll
                    for (int f = 0; f < 4; ++f) {
                        float * dst = sobase + (int64_t) (vrow + v0 + f) * S_v + k0;
                        *(float4*) dst = make_float4(acc[0][f], acc[1][f], acc[2][f], acc[3][f]);
                    }
                }
            }
            __syncthreads();
        }
#undef TK
    }
}

// ---------------------------------------------------------------------------------------------

template <int S_v>
static void launch_gated_delta_net_chunked(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d, float * scratch_d,
        int mode, int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_seq_stride, cudaStream_t stream) {
    dim3 grid((unsigned) H, (unsigned) n_seqs);
    const dim3 block(GDN_CHUNKED_NTHREADS);
    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid, block, 0, stream);
    ggml_cuda_kernel_launch(gated_delta_net_chunked_cuda<S_v>, launch_params,
        q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, scratch_d, mode, H, neqk1, n_tokens, n_seqs,
        sq1, sq2, sq3, sv1, sv2, sv3, neqk1_magic, rq3_magic, scale, state_seq_stride);
}

void ggml_cuda_op_gated_delta_net_chunked(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;
    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;
    const float * s_d = (const float *) src_state->data;

    float * dst_d     = (float *) dst->data;
    float * state_d   = dst_d + S_v * H * n_tokens * n_seqs;   // K == 1: single slot

    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);
    const int64_t state_seq_stride = H * S_v * S_v;

    cudaStream_t stream = ctx.stream();

    // per-chunk state scratch: [n_chunks][H][n_seqs][S_v*S_v] fp32
    const int64_t n_chunks = (n_tokens + GDN_CHUNKED_CS - 1) / GDN_CHUNKED_CS;
    ggml_cuda_pool_alloc<float> scratch(ctx.pool(), (size_t) n_chunks * H * n_seqs * S_v * S_v);

    switch (S_v) {
        case 16:  launch_gated_delta_net_chunked<16 >(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, scratch.get(), 0,
                    H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, neqk1, rq3, scale, state_seq_stride, stream); break;
        case 32:  launch_gated_delta_net_chunked<32 >(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, scratch.get(), 0,
                    H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, neqk1, rq3, scale, state_seq_stride, stream); break;
        case 64:  launch_gated_delta_net_chunked<64 >(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, scratch.get(), 0,
                    H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, neqk1, rq3, scale, state_seq_stride, stream); break;
        case 128: launch_gated_delta_net_chunked<128>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, scratch.get(), 0,
                    H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, neqk1, rq3, scale, state_seq_stride, stream); break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
    switch (S_v) {
        case 16:  launch_gated_delta_net_chunked<16 >(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, scratch.get(), 1,
                    H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, neqk1, rq3, scale, state_seq_stride, stream); break;
        case 32:  launch_gated_delta_net_chunked<32 >(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, scratch.get(), 1,
                    H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, neqk1, rq3, scale, state_seq_stride, stream); break;
        case 64:  launch_gated_delta_net_chunked<64 >(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, scratch.get(), 1,
                    H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, neqk1, rq3, scale, state_seq_stride, stream); break;
        case 128: launch_gated_delta_net_chunked<128>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, scratch.get(), 1,
                    H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, neqk1, rq3, scale, state_seq_stride, stream); break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}
