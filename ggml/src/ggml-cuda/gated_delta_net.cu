#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"
#include <cstdlib>

// ================= chunked gated delta rule (prefill path) =================
//
// For n_tokens > 1 the per-token sequential loop of gated_delta_net_cuda is
// replaced by a chunked formulation (chunk size GDN_CHUNK = 64, K = V = 128):
//
//   phase A (gdn_chunk_prepare, one CTA per head x seq x chunk):
//     P_t      = exp(cumsum(gate))            per-chunk prefix products
//     L[i][j]  = P_i / P_j                    causal decay factors
//     w[i][j]  = -beta_i * L[i][j] * <k_i,k_j>
//     attn     = (I - w_lo)^{-1}              closure via forward substitution
//     k_cumsum   = attn @ (beta*v)
//     k_cumdecay = attn @ (beta*k*P)
//     attn_causal = L .* (q^T k)
//
//   phase B (gdn_chunk_state, one CTA per head x seq x V-slice of 32):
//     v_prime = k_cumdecay @ S
//     v_new   = k_cumsum - v_prime
//     o       = P .* (q @ S) + attn_causal @ v_new
//     S       = P_last * S + (k * (P_last / P)) @ v_new
//
// Matches gated_delta_net_cuda<128,false,false> numerically (validated in
// python, < 3e-4 relative on random K=V=128, T up to 512). Enabled only for
// scalar gates (KDA=false), K=1, K=V=128, n_tokens > 1; everything else keeps
// the sequential kernel.

#define GDN_CHUNK 64
#define GDN_CHUNK_VSLICE 32

// phase A: per (head, seq, chunk). Block = 256 threads.
__global__ void gdn_chunk_prepare(
        const float * __restrict__ q,
        const float * __restrict__ k,
        const float * __restrict__ v,
        const float * __restrict__ gate,
        const float * __restrict__ beta,
        float * __restrict__ k_cumsum,     // [n_chunks][GDN_CHUNK][V]
        float * __restrict__ k_cumdecay,   // [n_chunks][GDN_CHUNK][K]
        float * __restrict__ attn_causal,  // [n_chunks][GDN_CHUNK][GDN_CHUNK]
        float * __restrict__ decay_scratch, // [n_chunks][GDN_CHUNK]  cumsum(gate)
        float * __restrict__ delta_scratch,// [n_chunks][GDN_CHUNK]  exp(decay_last - decay_t)
        float * __restrict__ P_last_scratch, // [n_chunks] exp(decay_last)
        const int64_t n_tokens, const int64_t n_seqs, const int64_t H,
        const int64_t n_chunks,
        const int64_t sq1, const int64_t sq2, const int64_t sq3,
        const int64_t sv1, const int64_t sv2, const int64_t sv3,
        const int64_t sb1, const int64_t sb2, const int64_t sb3,
        const int64_t neqk1) {
    const int chunk = blockIdx.y;
    const int hs    = blockIdx.x;
    const int h     = hs % H;
    const int seq   = hs / H;
    const int iq    = h % neqk1;
    const int tid   = threadIdx.x;

    const int t0     = chunk * GDN_CHUNK;
    const int n_real = min((int) GDN_CHUNK, (int) n_tokens - t0);
    const int64_t hsc = (int64_t) hs * n_chunks;

    __shared__ float s_attn[GDN_CHUNK][GDN_CHUNK + 1]; // padded: column reads conflict-free
    __shared__ float s_k[GDN_CHUNK][128 + 1];          // padded row for conflict-free column reads
    __shared__ float s_bp[GDN_CHUNK];                  // beta_j * exp(decay_j)
    __shared__ float s_decay[GDN_CHUNK];
    __shared__ float s_A[GDN_CHUNK * (GDN_CHUNK - 1) / 2]; // preserved strictly-lower A for the closure

    const float * qb = q    + (int64_t) seq * sq3 + (int64_t) iq * sq1;
    const float * kb = k    + (int64_t) seq * sq3 + (int64_t) iq * sq1;
    const float * vb = v    + (int64_t) seq * sv3 + (int64_t) h  * sv1;
    const float * gb = gate + (int64_t) seq * sb3 + (int64_t) h  * sb1;
    const float * bb = beta + (int64_t) seq * sb3 + (int64_t) h  * sb1;

    // prefetch the chunk's k rows into registers: the L2 latency of the staging
    // loads overlaps the decay scan below (the phases read disjoint inputs)
    float4 kpre[8];
#pragma unroll
    for (int it = 0; it < 8; ++it) {
        const int e  = tid + it * 256;
        const int t  = e / 32;
        const int d4 = e % 32;
        kpre[it] = (t < n_real)
            ? *reinterpret_cast<const float4 *>(kb + (int64_t) (t0 + t) * sq2 + 4 * d4)
            : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // per-chunk prefix sums of the log-gates; decay factors are always taken
    // as exp(diff) so strong decay (cumsum < -88) underflows to 0 instead of
    // producing 0/0 NaN. Padded gates are 0. Warp-shuffle scan (2 syncs):
    // warps 0 and 1 scan their 32 elements, then warp 1 adds warp 0's total.
    {
        float v = (tid < n_real) ? gb[(int64_t) (t0 + tid) * sb2] : 0.0f;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const float up = __shfl_up_sync(0xffffffff, v, o, 32);
            if (tid % 32 >= o) {
                v += up;
            }
        }
        if (tid == 31) {
            s_decay[31] = v; // warp 0 total, needed by warp 1's fixup
        }
        __syncthreads();
        if (tid >= 32 && tid < GDN_CHUNK) {
            v += s_decay[31];
        }
        if (tid < GDN_CHUNK) {
            s_decay[tid] = v;
            decay_scratch[(hsc + chunk) * GDN_CHUNK + tid] = v;
        }
        __syncthreads();
    }

    // stage k into smem from the prefetched registers; padded rows are zero
    {
#pragma unroll
        for (int it = 0; it < 8; ++it) {
            const int e  = tid + it * 256;
            const int t  = e / 32;
            const int d4 = e % 32;
            s_k[t][4 * d4 + 0] = kpre[it].x;
            s_k[t][4 * d4 + 1] = kpre[it].y;
            s_k[t][4 * d4 + 2] = kpre[it].z;
            s_k[t][4 * d4 + 3] = kpre[it].w;
        }
        if (tid < GDN_CHUNK) {
            s_bp[tid] = (tid < n_real) ? bb[(int64_t) (t0 + tid) * sb2] * expf(s_decay[tid]) : 0.0f;
        }
        __syncthreads();
    }

    // gram = k @ k^T into s_attn; one 4x4 output tile per thread
    {
        const int i0 = 4 * (tid / 16);
        const int j0 = 4 * (tid % 16);
        float acc[4][4];
#pragma unroll
        for (int ii = 0; ii < 4; ++ii) {
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                acc[ii][jj] = 0.0f;
            }
        }
        for (int d = 0; d < 128; ++d) {
            float kr[4], kc[4];
#pragma unroll
            for (int ii = 0; ii < 4; ++ii) {
                kr[ii] = s_k[i0 + ii][d];
            }
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                kc[jj] = s_k[j0 + jj][d];
            }
#pragma unroll
            for (int ii = 0; ii < 4; ++ii) {
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    acc[ii][jj] += kr[ii] * kc[jj];
                }
            }
        }
#pragma unroll
        for (int ii = 0; ii < 4; ++ii) {
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                s_attn[i0 + ii][j0 + jj] = acc[ii][jj];
            }
        }
        __syncthreads();
    }

    // w[i][j] = -beta_i * exp(decay_i - decay_j) * gram[i][j] for i > j
    // (zero diagonal and upper triangle: the closure treats A as strictly lower)
    {
        for (int e = tid; e < GDN_CHUNK * (GDN_CHUNK - 1) / 2; e += 256) {
            const int i = (int) ((sqrtf(8.0f * e + 1.0f) + 1.0f) * 0.5f);
            const int j = e - i * (i - 1) / 2;
            const bool ok = (i < n_real) && (j < n_real);
            const float w = ok ? (-bb[(int64_t) (t0 + i) * sb2] * expf(s_decay[i] - s_decay[j]) * s_attn[i][j]) : 0.0f;
            s_attn[i][j] = w;
            s_A[e] = w; // preserved strictly-lower A (row-major packed) for the closure
        }
        for (int e = tid; e < GDN_CHUNK * GDN_CHUNK; e += 256) {
            const int i = e / GDN_CHUNK;
            const int j = e % GDN_CHUNK;
            if (j >= i) {
                s_attn[i][j] = 0.0f;
            }
        }
        __syncthreads();
    }

    // closure: attn = (I-A)^{-1} by 4x4-blocked forward substitution. Exact: a
    // 4x4 strictly-lower block has nilpotency index 4, so the diagonal blocks
    // D(i,i) = I + Aii + Aii^2 + Aii^3 are exact; the off-diagonal blocks follow
    // D(i,j) = D(i,i) @ (A(i,j) D(j,j) + sum_{j<k<i} A(i,k) D(k,j)) in row passes.
    // The row passes read A(i,k) from the preserved s_A copy: the in-place writes
    // of D(i,k) would otherwise clobber them. 16 syncs total (vs 146 before).
    {
        // level 0: diagonal blocks D(i,i), one 4x4 block per 16 threads
        {
            const int b    = tid / 16;               // diagonal block 0..15
            const int r    = (tid % 16) / 4;
            const int c    = tid % 4;
            const int base = 16 * ((tid % 32) / 16); // first lane of this block
            // row substitution D(rr,:) = e_rr + sum_{k<rr} A(rr,k) D(k,:); rows are
            // processed in order, D(k,c) broadcast from lane (4k+c), no smem round
            // trip inside the 4x4 solve
            float v = 0.0f;
#pragma unroll
            for (int rr = 0; rr < 4; ++rr) {
                const float a = s_attn[4 * b + r][4 * b + rr]; // A(r,rr), strictly lower
                if (rr == r) {
                    v += (rr == c) ? 1.0f : 0.0f;
                }
                const float dk = __shfl_sync(0xffffffff, v, base + 4 * rr + c, 32);
                if (rr < r) {
                    v += a * dk;
                }
            }
            s_attn[4 * b + r][4 * b + c] = v;
        }
        __syncthreads();

        // row passes i = 1..15: D(i,j) for j = 0..i-1, one 4x4 block per 16 threads
        for (int i = 1; i < 16; ++i) {
            if (tid / 16 < i) {
                const int blk = tid / 16;            // j
                const int r   = (tid % 16) / 4;
                const int c   = tid % 4;
                // X[l][c] = sum_{k=j}^{i-1} sum_m A(i,k)[l][m] D(k,j)[m][c]
                float X[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
                int rowoff[4];
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int row = 4 * i + l;
                    rowoff[l] = row * (row - 1) / 2; // packed strictly-lower offset
                }
                for (int k = blk; k < i; ++k) {
                    const float dk0 = s_attn[4 * k + 0][4 * blk + c];
                    const float dk1 = s_attn[4 * k + 1][4 * blk + c];
                    const float dk2 = s_attn[4 * k + 2][4 * blk + c];
                    const float dk3 = s_attn[4 * k + 3][4 * blk + c];
                    const int acol = 4 * k;
#pragma unroll
                    for (int l = 0; l < 4; ++l) {
                        X[l] += s_A[rowoff[l] + acol + 0] * dk0;
                        X[l] += s_A[rowoff[l] + acol + 1] * dk1;
                        X[l] += s_A[rowoff[l] + acol + 2] * dk2;
                        X[l] += s_A[rowoff[l] + acol + 3] * dk3;
                    }
                }
                // D(i,j)[r][c] = sum_l D(i,i)[r][l] X[l][c]
                float d = 0.0f;
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    d += s_attn[4 * i + r][4 * i + l] * X[l];
                }
                s_attn[4 * i + r][4 * blk + c] = d;
            }
            __syncthreads();
        }
    }

    // k_cumsum[t][v]   = sum_j attn[t][j] * beta_j * v[j][v]
    // k_cumdecay[t][k] = sum_j attn[t][j] * beta_j * exp(decay_j) * k[j][k]
    // one 4x8 output tile per thread (4 token rows x 8 v/k columns)
    {
        const int t0t = 4 * (tid / 16);
        const int v0  = 8 * (tid % 16);
        float accv[4][8], acck[4][8];
#pragma unroll
        for (int ii = 0; ii < 4; ++ii) {
#pragma unroll
            for (int vv = 0; vv < 8; ++vv) {
                accv[ii][vv] = 0.0f;
                acck[ii][vv] = 0.0f;
            }
        }
        // v[j][v0..v0+7] is 8-consecutive floats per row; load it as 2 float4
        // when the row stride and base pointer are 16-B aligned (the model's v
        // layout is [V][T][H] so sv2 = V*H), else fall back to scalars
        if (((sv2 & 3) == 0) && (((uintptr_t) vb) & 15) == 0) {
            for (int j = 0; j < GDN_CHUNK; ++j) {
                const bool okj = j < n_real;
                const float bj  = okj ? bb[(int64_t) (t0 + j) * sb2] : 0.0f;
                const float bpj = okj ? s_bp[j] : 0.0f;
                const float * vj = vb + (int64_t) (t0 + j) * sv2 + v0;
                const float4 v4a = okj ? *reinterpret_cast<const float4 *>(vj)     : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                const float4 v4b = okj ? *reinterpret_cast<const float4 *>(vj + 4) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                const float vval[8] = { v4a.x, v4a.y, v4a.z, v4a.w, v4b.x, v4b.y, v4b.z, v4b.w };
#pragma unroll
                for (int ii = 0; ii < 4; ++ii) {
                    const float a = s_attn[t0t + ii][j];
#pragma unroll
                    for (int vv = 0; vv < 8; ++vv) {
                        accv[ii][vv] += a * (bj * vval[vv]);
                        acck[ii][vv] += a * (bpj * s_k[j][v0 + vv]);
                    }
                }
            }
        } else {
            for (int j = 0; j < GDN_CHUNK; ++j) {
                const bool okj = j < n_real;
                const float bj  = okj ? bb[(int64_t) (t0 + j) * sb2] : 0.0f;
                const float bpj = okj ? s_bp[j] : 0.0f;
                const float * vj = vb + (int64_t) (t0 + j) * sv2 + v0;
#pragma unroll
                for (int ii = 0; ii < 4; ++ii) {
                    const float a = s_attn[t0t + ii][j];
#pragma unroll
                    for (int vv = 0; vv < 8; ++vv) {
                        const float vval = okj ? vj[vv] : 0.0f;
                        accv[ii][vv] += a * (bj * vval);
                        acck[ii][vv] += a * (bpj * s_k[j][v0 + vv]);
                    }
                }
            }
        }
        float * kc = k_cumsum   + ((hsc + chunk) * GDN_CHUNK + t0t) * 128 + v0;
        float * kd = k_cumdecay + ((hsc + chunk) * GDN_CHUNK + t0t) * 128 + v0;
#pragma unroll
        for (int ii = 0; ii < 4; ++ii) {
#pragma unroll
            for (int vv = 0; vv < 8; ++vv) {
                kc[ii * 128 + vv] = accv[ii][vv];
                kd[ii * 128 + vv] = acck[ii][vv];
            }
        }
        __syncthreads();
    }

    // attn_causal = L .* (q @ k^T), reusing s_attn (strictly causal: j <= i)
    {
        const int i0 = 4 * (tid / 16);
        const int j0 = 4 * (tid % 16);
        float acc[4][4];
#pragma unroll
        for (int ii = 0; ii < 4; ++ii) {
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                acc[ii][jj] = 0.0f;
            }
        }
        const float * qr0 = qb + (int64_t) (t0 + i0) * sq2;
        // q rows are sq2-strided, but consecutive along d: 4-wide unroll with
        // float4 loads when the row stride and base pointer are 16-B aligned
        if (((sq2 & 3) == 0) && (((uintptr_t) qb) & 15) == 0) {
            for (int d4 = 0; d4 < 32; ++d4) {
                const int d = 4 * d4;
                float4 qv4[4];
                float kv[4][4];
#pragma unroll
                for (int ii = 0; ii < 4; ++ii) {
                    qv4[ii] = (i0 + ii < n_real)
                        ? *reinterpret_cast<const float4 *>(qr0 + ii * sq2 + d)
                        : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    kv[jj][0] = s_k[j0 + jj][d + 0];
                    kv[jj][1] = s_k[j0 + jj][d + 1];
                    kv[jj][2] = s_k[j0 + jj][d + 2];
                    kv[jj][3] = s_k[j0 + jj][d + 3];
                }
#pragma unroll
                for (int ii = 0; ii < 4; ++ii) {
#pragma unroll
                    for (int jj = 0; jj < 4; ++jj) {
                        acc[ii][jj] += qv4[ii].x * kv[jj][0] + qv4[ii].y * kv[jj][1]
                                     + qv4[ii].z * kv[jj][2] + qv4[ii].w * kv[jj][3];
                    }
                }
            }
        } else {
            for (int d = 0; d < 128; ++d) {
                float qv[4], kv[4];
#pragma unroll
                for (int ii = 0; ii < 4; ++ii) {
                    qv[ii] = (i0 + ii < n_real) ? qr0[ii * sq2 + d] : 0.0f;
                }
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    kv[jj] = s_k[j0 + jj][d];
                }
#pragma unroll
                for (int ii = 0; ii < 4; ++ii) {
#pragma unroll
                    for (int jj = 0; jj < 4; ++jj) {
                        acc[ii][jj] += qv[ii] * kv[jj];
                    }
                }
            }
        }
#pragma unroll
        for (int ii = 0; ii < 4; ++ii) {
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                const int i = i0 + ii;
                const int j = j0 + jj;
                const float L = (j <= i && i < n_real) ? expf(s_decay[i] - s_decay[j]) : 0.0f;
                s_attn[i][j] = L * acc[ii][jj];
            }
        }
        __syncthreads();
        for (int e = tid; e < GDN_CHUNK * GDN_CHUNK; e += 256) {
            const int i = e / GDN_CHUNK;
            const int j = e % GDN_CHUNK;
            attn_causal[((hsc + chunk) * GDN_CHUNK + i) * GDN_CHUNK + j] = s_attn[i][j];
        }
    }

    // delta[t] = exp(decay_last - decay_t) for the phase-B state update
    {
        const int last = n_real - 1;
        const float d_last = s_decay[max(0, last)];
        if (tid == 0) {
            P_last_scratch[hsc + chunk] = expf(d_last);
        }
        for (int t = tid; t < GDN_CHUNK; t += 256) {
            delta_scratch[(hsc + chunk) * GDN_CHUNK + t] = expf(d_last - s_decay[t]);
        }
    }
}

// phase B: per (head, seq, V-slice of 32). Block = 256 threads, one state
// column per thread (32 cols per warp) so kd/q/k/ac loads are warp broadcasts:
//   v_new = k_cumsum - k_cumdecay @ S
//   o     = P .* (q @ S) + attn_causal @ v_new
//   S     = P_last * S + (k .* delta)^T @ v_new
// S stays resident in smem across the chunk chain. Thread (row8, v) owns 8
// token rows (steps 1/2) or 16 state rows (step 3) of column v.
__global__ void __launch_bounds__(256, 2) gdn_chunk_state(
        const float * __restrict__ q,
        const float * __restrict__ k,
        const float * __restrict__ curr_state,
        const float * __restrict__ k_cumsum,     // [n_chunks][GDN_CHUNK][V]
        const float * __restrict__ k_cumdecay,   // [n_chunks][GDN_CHUNK][K]
        const float * __restrict__ attn_causal,  // [n_chunks][GDN_CHUNK][GDN_CHUNK]
        const float * __restrict__ decay_scratch,// [n_chunks][GDN_CHUNK]  cumsum(gate)
        const float * __restrict__ delta_scratch,// [n_chunks][GDN_CHUNK]  exp(decay_last - decay_t)
        const float * __restrict__ P_last_scratch, // [n_chunks] exp(decay_last)
        float * __restrict__ dst,
        float * __restrict__ state_out,
        const int64_t n_tokens, const int64_t n_seqs, const int64_t H,
        const int64_t n_chunks, const int64_t V, const int64_t K,
        const int64_t sq1, const int64_t sq2, const int64_t sq3,
        const int64_t neqk1, const float scale) {
    const int n_slices = (int) (V / GDN_CHUNK_VSLICE);
    const int vslice   = (int) (blockIdx.x % n_slices);
    const int hs       = (int) (blockIdx.x / n_slices);
    const int h        = (int) (hs % H);
    const int seq      = (int) (hs / H);
    const int iq       = (int) (h % neqk1);
    const int tid      = threadIdx.x;
    const int v0       = vslice * GDN_CHUNK_VSLICE;
    const int64_t hsc  = (int64_t) hs * n_chunks;

    const int row8 = tid / GDN_CHUNK_VSLICE; // 8-row block, 0..7
    const int v    = tid % GDN_CHUNK_VSLICE; // state column within the slice

    __shared__ float s_S[128][GDN_CHUNK_VSLICE];
    __shared__ float s_vnew[GDN_CHUNK][GDN_CHUNK_VSLICE];

    const float * qb = q + (int64_t) seq * sq3 + (int64_t) iq * sq1;
    const float * kb = k + (int64_t) seq * sq3 + (int64_t) iq * sq1;
    const float * sbase = curr_state + ((int64_t) seq * H + h) * (K * V);

    // load this slice of the incoming state: S[k][v], stored flat v*K + k
    for (int e = tid; e < 128 * GDN_CHUNK_VSLICE / 4; e += 256) {
        const int vv  = e % GDN_CHUNK_VSLICE;
        const int kk4 = 4 * (e / GDN_CHUNK_VSLICE);
        const float4 sv = *reinterpret_cast<const float4 *>(sbase + (v0 + vv) * K + kk4);
        s_S[kk4 + 0][vv] = sv.x;
        s_S[kk4 + 1][vv] = sv.y;
        s_S[kk4 + 2][vv] = sv.z;
        s_S[kk4 + 3][vv] = sv.w;
    }
    __syncthreads();

    for (int c = 0; c < n_chunks; ++c) {
        const int t0     = (int) (c * GDN_CHUNK);
        const int n_real = min((int) GDN_CHUNK, (int) n_tokens - t0);
        const int64_t cc = hsc + c;
        const float * kc = k_cumsum    + cc * GDN_CHUNK * V;
        const float * kd = k_cumdecay  + cc * GDN_CHUNK * K;
        const float * ac = attn_causal + cc * GDN_CHUNK * GDN_CHUNK;
        const float * dc = decay_scratch + cc * GDN_CHUNK;
        const float * dlt = delta_scratch + cc * GDN_CHUNK;
        const float P_last = P_last_scratch[cc];

        // v_new = k_cumsum - k_cumdecay @ S ; o_q = q @ S. Merged K loop: the
        // s_S column is shared by the thread's 8 token rows.
        float acc_v[8], acc_q[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            acc_v[r] = 0.0f;
            acc_q[r] = 0.0f;
        }
        for (int kq = 0; kq < 128; kq += 4) {
            float sS[4];
#pragma unroll
            for (int k2 = 0; k2 < 4; ++k2) {
                sS[k2] = s_S[kq + k2][v];
            }
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const float4 kdv = *reinterpret_cast<const float4 *>(kd + (8 * row8 + r) * K + kq);
                float4 qv;
                if (8 * row8 + r < n_real) {
                    qv = *reinterpret_cast<const float4 *>(qb + (int64_t) (t0 + 8 * row8 + r) * sq2 + kq);
                } else {
                    qv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
                acc_v[r] += kdv.x * sS[0] + kdv.y * sS[1] + kdv.z * sS[2] + kdv.w * sS[3];
                acc_q[r] += qv.x  * sS[0] + qv.y  * sS[1] + qv.z  * sS[2] + qv.w  * sS[3];
            }
        }

        // v_new[t][v] = k_cumsum[t][v] - (k_cumdecay @ S)[t][v]
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            s_vnew[8 * row8 + r][v] = kc[(8 * row8 + r) * V + v0 + v] - acc_v[r];
        }
        __syncthreads();

        // o = P .* (q @ S) + attn_causal @ v_new
        float acc_o[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            acc_o[r] = expf(dc[8 * row8 + r]) * acc_q[r];
        }
        for (int jq = 0; jq < GDN_CHUNK; jq += 4) {
            float vn[4];
#pragma unroll
            for (int k2 = 0; k2 < 4; ++k2) {
                vn[k2] = s_vnew[jq + k2][v];
            }
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const float4 av = *reinterpret_cast<const float4 *>(ac + (8 * row8 + r) * GDN_CHUNK + jq);
                acc_o[r] += av.x * vn[0] + av.y * vn[1] + av.z * vn[2] + av.w * vn[3];
            }
        }

        // write o to dst
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int t = 8 * row8 + r;
            if (t < n_real) {
                dst[((int64_t) (seq * n_tokens + t0 + t) * H + h) * V + v0 + v] = scale * acc_o[r];
            }
        }

        // S = P_last * S + (k .* delta)^T @ v_new. k[t][k] is contiguous in k
        // (float2 per 2 rows), v_new[t][v] shared by the thread's 16 k rows.
        float acc_S[16];
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            acc_S[r] = 0.0f;
        }
        for (int t4 = 0; t4 < GDN_CHUNK / 4; ++t4) {
            const int t = 4 * t4;
            const float4 d4 = *reinterpret_cast<const float4 *>(dlt + t);
#pragma unroll
            for (int tt = 0; tt < 4; ++tt) {
                const int  tr = (t + tt < n_real) ? (t + tt) : (n_real - 1); // clamp: padded rows never contribute
                const float d  = (t + tt < n_real) ? (tt == 0 ? d4.x : tt == 1 ? d4.y : tt == 2 ? d4.z : d4.w) : 0.0f;
                const float vn = s_vnew[t + tt][v];
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    const float2 kv = *reinterpret_cast<const float2 *>(kb + (int64_t) (t0 + tr) * sq2 + 16 * row8 + 2 * r);
                    acc_S[2 * r + 0] += d * kv.x * vn;
                    acc_S[2 * r + 1] += d * kv.y * vn;
                }
            }
        }

        // in-place S update; the barrier below publishes it to the next chunk
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            s_S[16 * row8 + r][v] = P_last * s_S[16 * row8 + r][v] + acc_S[r];
        }
        __syncthreads();
    }

    // write the final state slice
    float * sbase_out = state_out + ((int64_t) seq * H + h) * (K * V);
    for (int e = tid; e < 128 * GDN_CHUNK_VSLICE / 4; e += 256) {
        const int vv  = e % GDN_CHUNK_VSLICE;
        const int kk4 = 4 * (e / GDN_CHUNK_VSLICE);
        const float4 sv = make_float4(s_S[kk4 + 0][vv], s_S[kk4 + 1][vv], s_S[kk4 + 2][vv], s_S[kk4 + 3][vv]);
        *reinterpret_cast<float4 *>(sbase_out + (v0 + vv) * K + kk4) = sv;
    }
}

template <int S_v, bool KDA, bool keep_rs_t>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        ggml_cuda_pool & pool,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, cudaStream_t stream) {
    // chunked path: prefill only, scalar gates, K=1, K=V=128. For K=1 the state
    // write layout is the same whether the gdn->cpy fusion fired (slot_stride 0)
    // or not (slot_stride = S_v*S_v*H*n_seqs). Default; the sequential kernel
    // stays for decode (n_tokens=1), KDA and keep_rs_t.
    static const bool gdn_chunked_enabled = []() {
        const char * env = getenv("GGML_CUDA_GDN_CHUNKED");
        return env == nullptr || std::atoi(env) != 0;
    }();
    if (gdn_chunked_enabled && !KDA && !keep_rs_t && S_v == 128 && n_tokens > 1 &&
        (state_slot_stride == 0 || state_slot_stride == S_v * S_v * H * n_seqs)) {
        const int64_t V = S_v;
        const int64_t KK = S_v;
        const int64_t n_chunks = (n_tokens + GDN_CHUNK - 1) / GDN_CHUNK;
        const int64_t n_slices = V / GDN_CHUNK_VSLICE;
        const int64_t per_hs = n_chunks * (GDN_CHUNK * (V + KK + GDN_CHUNK) + 2 * GDN_CHUNK) + n_chunks;
        const int64_t scratch_elems = H * n_seqs * per_hs;

        ggml_cuda_pool_alloc<float> scratch(pool, scratch_elems);
        float * k_cumsum    = scratch.get();
        float * k_cumdecay  = k_cumsum    + H * n_seqs * n_chunks * GDN_CHUNK * V;
        float * attn_causal = k_cumdecay  + H * n_seqs * n_chunks * GDN_CHUNK * KK;
        float * decay_scratch = attn_causal + H * n_seqs * n_chunks * GDN_CHUNK * GDN_CHUNK;
        float * delta_scratch = decay_scratch + H * n_seqs * n_chunks * GDN_CHUNK;
        float * P_last_scratch = delta_scratch + H * n_seqs * n_chunks * GDN_CHUNK;
        {
            const dim3 grid_a(H * n_seqs, n_chunks, 1);
            const dim3 block_a(256, 1, 1);
            const ggml_cuda_kernel_launch_params lp_a(grid_a, block_a, 0, stream);
            ggml_cuda_kernel_launch(gdn_chunk_prepare, lp_a,
                q_d, k_d, v_d, g_d, b_d,
                k_cumsum, k_cumdecay, attn_causal, decay_scratch, delta_scratch, P_last_scratch,
                n_tokens, n_seqs, H, n_chunks,
                sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1);
        }
        {
            const dim3 grid_b(H * n_seqs * n_slices, 1, 1);
            const dim3 block_b(256, 1, 1);
            const ggml_cuda_kernel_launch_params lp_b(grid_b, block_b, 0, stream);
            ggml_cuda_kernel_launch(gdn_chunk_state, lp_b,
                q_d, k_d, s_d,
                k_cumsum, k_cumdecay, attn_causal, decay_scratch, delta_scratch, P_last_scratch,
                dst_d, state_d,
                n_tokens, n_seqs, H, n_chunks, V, KK,
                sq1, sq2, sq3, neqk1, scale);
        }
        return;
    }

    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
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

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, ctx.pool(),
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, ctx.pool(),
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, ctx.pool(),
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, ctx.pool(),
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
