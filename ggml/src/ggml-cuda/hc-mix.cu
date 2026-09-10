// Fused hyper-connection mixer tail for qwen4exp (decode/verify band).
//
// Replaces the unfused decode chain (SCALE, SILU, MUL_MAT up, SIGMOID, MUL,
// collapse ADD/SCALE) with one op dispatch. The numerics replicate the unfused
// chain bit for bit:
//   lo_raw = w_down^T xn                       (mmvq Q8_0 dot, M = 1)
//   v      = silu(lo_raw / hc)                 (SCALE then SILU)
//   gate   = sigmoid(w_up^T v)                 (mmvq Q8_0 dot, M = 1)
//   mixed  = (1/hc) * sum_c xn * gate          (collapse of the hc streams)
// The xn -> Q8_1 quantization mirrors quantize_row_q8_1_cuda, and the per-row
// dots replicate mul_mat_vec_q<Q8_0, 1> (block (32, 8), rpb = 1) so the sums
// are bit-identical to the unfused mmvq path.
//
// Token dimension: the ops serve the whole decode/verify band, nt <=
// HC_FUSED_MAX_TOKENS, by mapping the token index onto blockIdx.y (the graph
// gates in src/models/qwen4exp.cpp use the same bound). Every token therefore
// runs exactly the per-token kernel sequence a single-token decode runs, which
// is what makes the band width-invariant; nt == 1 selects t == 0 only and its
// arithmetic is unchanged.

#include "hc-mix.cuh"

#include "convert.cuh"
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

// Upper bound of the decode/verify band these fused ops serve (see the header
// comment).  The graphs in src/models/qwen4exp.cpp gate the fused path on the
// same bound; a verify batch is --spec-draft-n-max + 1 tokens and the
// greedy-purity guarantee is defined over that band (GREEDY-PURITY.md).
static constexpr int64_t HC_FUSED_MAX_TOKENS = 8;


// Grouped RMSNorm over the hc streams plus the gamma scale (w_norm). One
// block of 1024 threads per stream row; the reduction mirrors rms_norm_f32
// (blockDim 1024, warp shuffles then one final warp pass over the 32 warp
// sums) and the gamma multiply keeps the rms output's separate rounding, so
// xn is bit-identical to the unfused RMS + MUL chain.
// grouped RMSNorm + gamma (the xn stream) with the Q8_1 quantize of xn merged
// in: one block per stream computes xn then quantizes its own q8_1 groups (the
// quantize pattern mirrors quantize_q8_1: one warp per 32-value group, amax
// reduction, d = amax/127, roundf(x/d), sum), so the op runs one fewer kernel.
// n_embd must be divisible by 32.
// Token dimension: `t` (blockIdx.y) selects the token; every per-token pointer is offset by
// that token's row in the packed scratch buffers (strides from the launcher).  At nt == 1 only
// t == 0 exists, so the arithmetic is unchanged.
static __global__ void hc_mix_rms_gamma_quant(
        const float * x, const float * w_norm, float * xn, block_q8_1 * y,
        const int n_embd, const float eps,
        const int64_t x_stride_t, const int64_t xn_stride_t, const int64_t y_stride_t) {
    const int c   = blockIdx.x;
    const int t   = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const float * xs = x + (int64_t) t*x_stride_t + (int64_t) c*n_embd;

    // thread tid owns columns tid + k*blockDim.x in all three passes (group g = tid/32 + k*blockDim.x/32 covers
    // exactly those columns), so x and xn stay in registers instead of being re-read from global memory
    constexpr int max_k = 8;
    const bool in_regs = n_embd <= max_k*(int) blockDim.x;
    float xr[max_k];
    float tmp = 0.0f;
#pragma unroll
    for (int k = 0; k < max_k; ++k) {
        const int col = tid + k*blockDim.x;
        if (in_regs && col < n_embd) {
            xr[k] = xs[col];
            tmp += xr[k]*xr[k];
        }
    }
    if (!in_regs) {
        for (int col = tid; col < n_embd; col += blockDim.x) {
            const float xi = xs[col];
            tmp += xi*xi;
        }
    }
    __shared__ float s_sum[32];
    tmp = block_reduce<block_reduce_method::SUM>(tmp, s_sum);

    const float scale = rsqrtf(tmp / (float) n_embd + eps);
    float * xo = xn + (int64_t) t*xn_stride_t + (int64_t) c*n_embd;
#pragma unroll
    for (int k = 0; k < max_k; ++k) {
        const int col = tid + k*blockDim.x;
        if (in_regs && col < n_embd) {
            const float r = scale * xr[k];
            xr[k] = r * w_norm[(int64_t) c*n_embd + col];
            xo[col] = xr[k];
        }
    }
    if (!in_regs) {
        for (int col = tid; col < n_embd; col += blockDim.x) {
            const float r = scale * xs[col];
            xo[col] = r * w_norm[(int64_t) c*n_embd + col];
        }
        __syncthreads();
    }

    // the BF16 variant (y == nullptr) only needs xn
    if (y == nullptr) {
        return;
    }

    // quantize this stream's q8_1 groups (n_embd/32 groups, one warp each)
    const int n_groups = n_embd / 32;
    const auto quant_group = [&](const int g, const float xv) {
        float amax = fabsf(xv);
        float sum  = xv;
        amax = warp_reduce_max<32>(amax);
        sum  = warp_reduce_sum<32>(sum);
        const float  d = amax / 127.0f;
        const int8_t q = amax == 0.0f ? 0 : (int8_t) roundf(xv / d);
        y[(int64_t) t*y_stride_t + (int64_t) c*n_groups + g].qs[lane] = q;
        if (lane == 0) {
            y[(int64_t) t*y_stride_t + (int64_t) c*n_groups + g].ds = make_half2(d, sum);
        }
    };
    if (in_regs) {
#pragma unroll
        for (int k = 0; k < max_k; ++k) {
            const int g = tid/32 + k*(blockDim.x/32);
            if (g < n_groups) {
                quant_group(g, xr[k]);
            }
        }
    } else {
        for (int g = tid / 32; g < n_groups; g += blockDim.x / 32) {
            quant_group(g, xo[g*32 + lane]);
        }
    }
}

// hc_combine (the residual-stream combine of the previous block) fused into the BF16 hc_mix norm that reads it: block
// (c, t) computes stream c of token t of the combine exactly as hc_combine_kernel does (w = 2*sigmoid(inject[c]/hc),
// res + bo*w), writes it (the combine result is still the next combine's residual), and keeps the values in registers
// for the grouped RMSNorm + gamma, whose per-thread columns and block reduction are hc_mix_rms_gamma_quant's.  Same
// values, same order: bit-identical to the two kernels.  Only the in-register shape (n_embd <= 8*blockDim.x).
static __global__ void hc_mix_combine_rms_gamma(
        const float * residual, const float * block_out, const float * inject, float * comb,
        const float * w_norm, float * xn, const int n_embd, const int hc, const float eps,
        const int64_t res_stride_t, const int64_t bo_stride_t, const int64_t inject_stride_t,
        const int64_t comb_stride_t, const int64_t xn_stride_t) {
    const int c   = blockIdx.x;
    const int t   = blockIdx.y;
    const int tid = threadIdx.x;

    const float inv_hc = 1.0f / (float) hc;
    const float s = inject[(int64_t) t*inject_stride_t + c] * inv_hc;
    const float w = (1.0f / (1.0f + expf(-s))) * 2.0f;

    const float * res = residual  + (int64_t) t*res_stride_t + (int64_t) c*n_embd;
    const float * bo  = block_out + (int64_t) t*bo_stride_t;
    float       * dt  = comb      + (int64_t) t*comb_stride_t + (int64_t) c*n_embd;

    constexpr int max_k = 8;
    float xr[max_k];
    float tmp = 0.0f;
#pragma unroll
    for (int k = 0; k < max_k; ++k) {
        const int col = tid + k*blockDim.x;
        if (col < n_embd) {
            // hc_combine_kernel's product is not contracted into the add (its separate pp loop): keep it out of
            // an fma here too
            {
#pragma clang fp contract(off)
                xr[k] = res[col] + bo[col] * w;
            }
            dt[col] = xr[k];
            tmp += xr[k]*xr[k];
        }
    }
    __shared__ float s_sum[32];
    tmp = block_reduce<block_reduce_method::SUM>(tmp, s_sum);

    const float scale = rsqrtf(tmp / (float) n_embd + eps);
    float * xo = xn + (int64_t) t*xn_stride_t + (int64_t) c*n_embd;
#pragma unroll
    for (int k = 0; k < max_k; ++k) {
        const int col = tid + k*blockDim.x;
        if (col < n_embd) {
            const float r = scale * xr[k];
            xo[col] = r * w_norm[(int64_t) c*n_embd + col];
        }
    }
}

// One Q8_0 matrix-vector product against the pre-quantized Q8_1 input, rpb =
// 1 (one row per block, K fills the thread groups). The grid covers the lo
// rows (w_down, K = hc_dim) and then the inject rows (w_inject, K = hc_dim);
// the accumulation is the mul_mat_vec_q rpb=1 clone, bit-identical to the
// unfused mmvq path.  `t` (blockIdx.y) is the token; the weights are shared, the
// per-token q8_1 input and the output rows are offset by that token's scratch rows
// (`lo` is packed at hc_lr, `inject` lives in the dst tail so it carries the dst row
// stride).
static __global__ void hc_mix_down_dots(
        const block_q8_0 * w_down, float * lo, const int nrows_down,
        const block_q8_0 * w_inject, float * inject, const int nrows_inject,
        const block_q8_1 * y, const int blocks_per_row,
        const int64_t y_stride_t, const int64_t lo_stride_t, const int64_t inject_stride_t,
        block_q8_1 * y_v, const int blocks_up, const float inv_hc, unsigned int * done) {
    const int row = blockIdx.x;
    const int t   = blockIdx.y;
    const bool is_inject = row >= nrows_down;
    const int r = is_inject ? row - nrows_down : row;
    const block_q8_0 * w = is_inject ? w_inject : w_down;
    float * dst = is_inject ? inject + (int64_t) t*inject_stride_t : lo + (int64_t) t*lo_stride_t;
    const block_q8_1 * y_t = y + (int64_t) t*y_stride_t;

    constexpr int qi  = QI8_0;             // 8
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    const int tid      = 32*threadIdx.y + threadIdx.x;
    const int n_groups = 8*32 / (qi/vdr);
    const int n_items  = blocks_per_row;

    float acc = 0.0f;
    const int kqs = vdr * (tid % (qi/vdr));
    for (int it = tid / (qi/vdr); it < n_items; it += n_groups) {
        acc += vec_dot_q8_0_q8_1(w + (int64_t) r*blocks_per_row, &y_t[it], it, kqs);
    }

    acc = warp_reduce_sum<32>(acc);
    __shared__ float tmp_shared[7];
    if (threadIdx.y > 0) {
        tmp_shared[threadIdx.y-1] = acc;
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }
#pragma unroll
    for (int l = 0; l < 7; ++l) {
        acc += tmp_shared[l];
    }
    if (threadIdx.x == 0) {
        dst[r] = acc;
    }
    if (y_v == nullptr || is_inject) {
        return;
    }
    // v = silu(lo/hc) quantized once here instead of in every up-dot block: the block that
    // completes the 32 lo rows of a q8_1 group (per-token counter) quantizes that group with
    // warp 0. Same arithmetic as the up-dot prologue, so y_v is bit-identical to it.
    const int kb = r / 32;
    unsigned int prev = 0;
    if (threadIdx.x == 0) {
        __threadfence();
        prev = atomicAdd(&done[t*blocks_up + kb], 1u);
    }
    prev = __shfl_sync(0xFFFFFFFF, prev, 0, 32);
    if (prev != 31) {
        return;
    }
    __threadfence();
    const int lane = threadIdx.x;
    const float x = ggml_cuda_op_silu_single(__builtin_nontemporal_load(&lo[(int64_t) t*lo_stride_t + kb*32 + lane]) * inv_hc);
    float amax = fabsf(x);
    float sum  = x;
    amax = warp_reduce_max<32>(amax);
    sum  = warp_reduce_sum<32>(sum);
    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : (int8_t) roundf(x / d);
    y_v[t*blocks_up + kb].qs[lane] = q;
    if (lane == 0) {
        y_v[t*blocks_up + kb].ds = make_half2(d, sum);
    }
}

// silu(lo/hc) and Q8_1 quantize of the low-rank vector, then the Q8_0 matrix-
// vector product w^T v (the "up" dot). The quantize kernel is merged into the
// dot kernel: every block quantizes all of v in its prologue (the writes are
// identical across blocks, so the global y buffer stays valid) and then dots,
// so the op runs one fewer kernel. The per-kblock reduction mirrors
// quantize_q8_1 (one warp over 32 consecutive values) and silu mirrors the
// standalone op, so the values are bit-identical to the unfused chain.
// `t` (blockIdx.y) is the token: the weights are shared, `lo`/`y`/`dst` are the
// packed per-token scratch rows.
template <int nwarps, int RPB, bool PRE = false>
static __global__ void hc_mix_up_silu_dot(
        const float * lo, const block_q8_0 * w, block_q8_1 * y,
        float * dst, const int nrows, const int blocks_per_row,
        const float inv_hc,
        const int64_t lo_stride_t, const int64_t y_stride_t, const int64_t dst_stride_t) {
    const int lane = threadIdx.x;
    const int t    = blockIdx.y;
    const float * lo_t = lo + (int64_t) t*lo_stride_t;
    block_q8_1 * y_t = y + (int64_t) t*y_stride_t;
    float * dst_t = dst + (int64_t) t*dst_stride_t;
    // prologue: v = silu(lo/hc) quantized to Q8_1, warps split the kblocks (PRE: done by down_dots)
    for (int kb = threadIdx.y; !PRE && kb < blocks_per_row; kb += nwarps) {
        const int col = kb*32 + lane;
        const float x = ggml_cuda_op_silu_single(lo_t[col] * inv_hc);
        float amax = fabsf(x);
        float sum  = x;
        amax = warp_reduce_max<32>(amax);
        sum  = warp_reduce_sum<32>(sum);
        const float  d = amax / 127.0f;
        const int8_t q = amax == 0.0f ? 0 : (int8_t) roundf(x / d);
        y_t[kb].qs[lane] = q;
        if (lane == 0) {
            y_t[kb].ds = make_half2(d, sum);
        }
    }
    if (!PRE) {
        __syncthreads();
    }

    // the dot body below is hc_mix_row_dot<8, RPB> unchanged
    const int row0 = RPB*blockIdx.x;
    constexpr int qi  = QI8_0;             // 8
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    const int tid      = 32*threadIdx.y + threadIdx.x;
    const int n_groups = nwarps*32 / (qi/vdr);
    const int n_items  = RPB * blocks_per_row;

    float tmp[RPB] = {0.0f};
    const int kqs = vdr * (tid % (qi/vdr));
    for (int it = tid / (qi/vdr); it < n_items; it += n_groups) {
        const int i   = it / blocks_per_row;
        const int kbx = it % blocks_per_row;
        if (row0 + i < nrows) {
            tmp[i] += vec_dot_q8_0_q8_1(w + (int64_t) (row0 + i) * blocks_per_row, &y_t[kbx], kbx, kqs);
        }
    }

    // rows touched by this warp (bit i); a warp with no item in row i holds +0.0f in every lane
    // for it, so skipping its butterfly leaves the +0.0f unchanged (bit-identical)
    uint32_t warp_rows = 0;
    for (int a = (32/(qi/vdr))*threadIdx.y; a < n_items; a += n_groups) {
        const int b = min(a + 32/(qi/vdr) - 1, n_items - 1);
        for (int i = a/blocks_per_row; i <= b/blocks_per_row; ++i) {
            warp_rows |= 1u << i;
        }
    }
    __shared__ float tmp_shared[nwarps > 1 ? nwarps-1 : 1][RPB];
    for (int i = 0; i < RPB; ++i) {
        if (warp_rows >> i & 1u) {
            tmp[i] = warp_reduce_sum<32>(tmp[i]);
        }
        if (threadIdx.y > 0) {
            tmp_shared[threadIdx.y-1][i] = tmp[i];
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }
    for (int i = 0; i < RPB; ++i) {
#pragma unroll
        for (int l = 0; l < nwarps-1; ++l) {
            tmp[i] += tmp_shared[l][i];
        }
        if (threadIdx.x == 0 && row0 + i < nrows) {
            dst_t[row0 + i] = tmp[i];
        }
    }
}

// Band variant of hc_mix_up_silu_dot: one block per RPB rows serves every token of the band
// (grid hc_dim/RPB instead of (hc_dim/RPB) x nt). v = silu(lo/hc) is quantized once per token
// into shared memory, each thread loads its weight blocks once and dots them against all NT
// token vectors, and the reductions run for all (token, row) pairs together. Per (token, row)
// the item -> thread mapping, the per-thread accumulation order (items ascending, from 0.0f),
// the warp butterfly and the warp-0 + warps 1..7 sum order are exactly those of
// hc_mix_up_silu_dot<8, RPB>, so every output is bit-identical to it at any nt.
// MAXI bounds the items per thread (ceil(RPB*blocks_per_row / n_groups)).
template <int nwarps, int RPB, int NT, int MAXI, int BPR, bool PRE>
static __global__ void hc_mix_up_silu_dot_band(
        const float * lo, const block_q8_1 * y_pre, const block_q8_0 * w,
        float * dst, const int nrows, const int blocks_per_row,
        const float inv_hc,
        const int64_t lo_stride_t, const int64_t dst_stride_t) {
    extern __shared__ block_q8_1 y_s[]; // [NT][blocks_per_row]
    const int lane = threadIdx.x;
    __shared__ float tmp_shared[nwarps > 1 ? nwarps-1 : 1][NT][RPB];
    __shared__ float tmp_w0[NT][RPB];
    // rows a warp has no item in contribute exactly +0.0 (see below): pre-zero every partial so the reduction
    // only visits the warp's own rows; made visible by the prologue barrier
    for (int k0 = 32*threadIdx.y + lane; k0 < (nwarps > 1 ? nwarps-1 : 1)*NT*RPB; k0 += 32*nwarps) {
        (&tmp_shared[0][0][0])[k0] = 0.0f;
    }
    for (int k0 = 32*threadIdx.y + lane; k0 < NT*RPB; k0 += 32*nwarps) {
        (&tmp_w0[0][0])[k0] = 0.0f;
    }
    // BPR == blocks_per_row. All lo loads of this warp are issued before any is used (one
    // memory latency instead of one per task); the per-task arithmetic is unchanged.
    if (PRE) {
        // y_pre holds v already quantized (down_dots tail): copy it to shared as 32-bit words
        const int * src = (const int *) y_pre;
        int * dstw = (int *) y_s;
        constexpr int nw = NT*BPR*sizeof(block_q8_1)/sizeof(int);
        for (int k = 32*threadIdx.y + lane; k < nw; k += 32*nwarps) {
            dstw[k] = src[k];
        }
        __syncthreads();
    }
    constexpr int NTK = PRE ? 1 : (NT*BPR + nwarps - 1) / nwarps;
    float xl[NTK];
#pragma unroll
    for (int j = 0; j < NTK; ++j) {
        const int tk = threadIdx.y + j*nwarps;
        xl[j] = PRE ? 0.0f : tk < NT*BPR ? lo[(int64_t) (tk / BPR)*lo_stride_t + (tk % BPR)*32 + lane] : 0.0f;
    }
#pragma unroll
    for (int j = 0; j < NTK; ++j) {
        const int tk = threadIdx.y + j*nwarps;
        if (PRE || tk >= NT*BPR) {
            break;
        }
        const float x = ggml_cuda_op_silu_single(xl[j] * inv_hc);
        float amax = fabsf(x);
        float sum  = x;
        amax = warp_reduce_max<32>(amax);
        sum  = warp_reduce_sum<32>(sum);
        const float  d = amax / 127.0f;
        const int8_t q = amax == 0.0f ? 0 : (int8_t) roundf(x / d);
        y_s[tk].qs[lane] = q;
        if (lane == 0) {
            y_s[tk].ds = make_half2(d, sum);
        }
    }
    __syncthreads();

    const int row0 = RPB*blockIdx.x;
    constexpr int qi  = QI8_0;
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    const int tid      = 32*threadIdx.y + threadIdx.x;
    const int n_groups = nwarps*32 / (qi/vdr);
    const int n_items  = RPB * blocks_per_row;
    const int kqs = vdr * (tid % (qi/vdr));
    const int it0 = tid / (qi/vdr);

    // this thread's items (it0, it0 + n_groups, ...): the dot of each against every token
    float val[MAXI][NT];
    int   row_of[MAXI];
#pragma unroll
    for (int k = 0; k < MAXI; ++k) {
        const int it = it0 + k*n_groups;
        row_of[k] = -1;
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            val[k][t] = 0.0f;
        }
        if (it < n_items) {
            const int i   = it / blocks_per_row;
            const int kbx = it % blocks_per_row;
            if (row0 + i < nrows) {
                row_of[k] = i;
                const block_q8_0 * wr = w + (int64_t) (row0 + i) * blocks_per_row;
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    val[k][t] = vec_dot_q8_0_q8_1(wr, &y_s[t*blocks_per_row + kbx], kbx, kqs);
                }
            }
        }
    }

    // rows touched by this warp: its items are it = 8*warp + [0, 8) + k*n_groups
    const int wit0 = (qi/vdr) == 4 ? 8*threadIdx.y : (32/(qi/vdr))*threadIdx.y;
    uint32_t warp_rows = 0; // bit i: some item of this warp is in row i
#pragma unroll
    for (int k = 0; k < MAXI; ++k) {
        const int a = wit0 + k*n_groups;
        const int b = min(a + 32/(qi/vdr) - 1, n_items - 1);
        if (a < n_items) {
            for (int i = a/blocks_per_row; i <= b/blocks_per_row; ++i) {
                warp_rows |= 1u << i;
            }
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        // a warp none of whose items fall in row i would reduce +0.0 values to +0.0 (no lane holds -0.0: tmp
        // starts at +0.0f), which is what the pre-zeroed partial already holds, so only the warp's own rows
        // (warp-uniform mask) are visited
        for (uint32_t m = warp_rows; m != 0; m &= m - 1) {
            const int i = __builtin_ctz(m);
            // tmp[i] of the reference: this thread's items of row i, ascending, from 0.0f
            float tmp = 0.0f;
#pragma unroll
            for (int k = 0; k < MAXI; ++k) {
                if (row_of[k] == i) {
                    tmp += val[k][t];
                }
            }
            tmp = warp_reduce_sum<32>(tmp);
            if (threadIdx.y > 0) {
                tmp_shared[threadIdx.y-1][t][i] = tmp;
            } else if (threadIdx.x == 0) {
                tmp_w0[t][i] = tmp;
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }
    for (int ti = lane; ti < NT*RPB; ti += 32) {
        const int t = ti / RPB;
        const int i = ti % RPB;
        float acc = tmp_w0[t][i];
#pragma unroll
        for (int l = 0; l < nwarps-1; ++l) {
            acc += tmp_shared[l][t][i];
        }
        if (row0 + i < nrows) {
            dst[(int64_t) t*dst_stride_t + row0 + i] = acc;
        }
    }
}

// Collapse the gated streams to their mean: mixed[j] = (1/hc) * sum_c xn*c*gate.
// gate holds the raw up projection; sigmoid is applied inline (same formula as
// the standalone op). The products are stored to an array before summing so the
// compiler cannot contract them into FMAs: the reference rounds each xn*gate
// product (a separate MUL op) and then adds the rounded values. One thread per
// output element in (256)-thread blocks so the stream reads coalesce; the adds
// follow the graph order (left-to-right ADD chain), then SCALE.
// collapse + F32 inject merged into one dispatch: grid = collapse blocks
// (n_embd/256) + hc inject blocks. Each path is unchanged from its separate
// kernel (the collapse products are stored before summing - no FMA - and the
// inject replicates the mmvf float2 accumulation), so the op runs one fewer
// kernel per call.  `t` (blockIdx.y) is the token: `xn`/`gate_raw`/`dst` and the dst
// inject tail are the packed per-token rows, while w_inject is shared.
static __global__ void hc_mix_collapse_inject(
        const float * xn, const float * gate_raw, float * dst,
        const int n_embd, const int hc, const float inv_hc,
        const float * w_inject, float * inject, const int ncols2,
        const int n_collapse_blocks,
        const int64_t xn_stride_t, const int64_t gate_stride_t, const int64_t dst_stride_t) {
    const int t = blockIdx.y;
    const float * xn_t = xn + (int64_t) t*xn_stride_t;
    const float * gate_t = gate_raw + (int64_t) t*gate_stride_t;
    float * dst_t = dst + (int64_t) t*dst_stride_t;
    float * inject_t = inject + (int64_t) t*dst_stride_t;
    if (blockIdx.x >= n_collapse_blocks) {
        // inject rows: one 256-thread block per inject row (mmvf numerics)
        const int r   = blockIdx.x - n_collapse_blocks;
        const int tid = threadIdx.x;
        const float2 * x2 = (const float2 *) xn_t;
        const float2 * w2 = (const float2 *) (w_inject + (int64_t) r*2*ncols2);
        float sumf = 0.0f;
        for (int col2 = tid; col2 < ncols2; col2 += blockDim.x) {
            const float2 tmpx = x2[col2];
            const float2 tmpw = w2[col2];
            sumf += tmpx.x*tmpw.x;
            sumf += tmpx.y*tmpw.y;
        }
        sumf = warp_reduce_sum<32>(sumf);
        __shared__ float buf[32];
        if (tid < 32) {
            buf[tid] = 0.0f;
        }
        __syncthreads();
        buf[tid/32] = sumf;
        __syncthreads();
        if (tid < 32) {
            sumf = buf[tid];
            sumf = warp_reduce_sum<32>(sumf);
            if (tid == 0) {
                inject_t[r] = sumf;
            }
        }
        return;
    }
    const int j = blockIdx.x*blockDim.x + threadIdx.x;
    if (j >= n_embd) {
        return;
    }
    float pp[8];
    for (int c = 0; c < hc; ++c) {
        const float g = 1.0f / (1.0f + expf(-gate_t[c*n_embd + j]));
        pp[c] = xn_t[c*n_embd + j] * g;
    }
    float sum = pp[0];
    for (int c = 1; c < hc; ++c) {
        sum = sum + pp[c];
    }
    dst_t[j] = sum * inv_hc;
}

// ---------------------------------------------------------------------------------------------------------------
// BF16 hc weights (Flash-Next GSQ-RCO). The unfused decode chain for BF16 weights is
//   rms_norm*gamma -> mul_mat_vec_f<bf16, 256> (down, K = hc_dim) -> scale+silu -> mul_mat_vec_f_vb<bf16, 160, 4>
//   (up, K = hc_lr) -> dsv4_hc_pre (sigmoid gate, collapse) -> mul_mat_vec_f<bf16, 256> (inject, K = hc_dim),
// six dispatches. This replays it in three, bit for bit:
//   1. hc_mix_rms_gamma_quant without the quantize (xn, as for Q8_0),
//   2. the down rows and the inject rows in one grid, each row the mul_mat_vec_f<bf16, float, nt, 256> block
//      (same per-thread K order, same warp butterfly, same butterfly over the per-warp sums),
//   3. the up rows of the hc streams of one column in one block (one warp per stream, the mul_mat_vec_f_vb
//      emulation of the 160-thread block), reading v = silu(lo/hc) as the scale+silu kernel computes it, and the
//      gated collapse of that column as dsv4_hc_pre_f32 computes it.
// The block sizes are what the mmvf launcher picks for these K (it depends on K only), so every nt in the band
// replays the unfused arithmetic.

template <int nt>
static __global__ void __launch_bounds__(256, 1) hc_mix_bf16_down_inject(
        const nv_bfloat16 * w_down, const nv_bfloat16 * w_inject, const float * xn, float * lo, float * inject,
        const int ncols2, const int nrows_down, const int64_t xn_stride_t, const int64_t lo_stride_t,
        const int64_t inject_stride_t) {
    constexpr int block_size = 256;
    constexpr int warp_size  = 32;
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const bool is_inject = row >= nrows_down;
    const int * x2 = (const int *) (is_inject ? w_inject + (int64_t) (row - nrows_down)*(2*ncols2)
                                              : w_down   + (int64_t) row*(2*ncols2));
    const float2 * y2 = (const float2 *) xn;
    const int stride_col_y2 = (int) (xn_stride_t/2);

    __shared__ float buf_iw[warp_size];
    if (tid < warp_size) {
        buf_iw[tid] = 0.0f;
    }
    __syncthreads();

    float sumf[nt] = {0.0f};
#pragma unroll 4
    for (int col2 = tid; col2 < ncols2; col2 += block_size) {
        const int tmpx = x2[col2];
#pragma unroll
        for (int j = 0; j < nt; ++j) {
            const float2 tmpy = y2[j*stride_col_y2 + col2];
            const float tmpx0 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[0]);
            const float tmpx1 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[1]);
            ggml_cuda_mad(sumf[j], tmpx0, tmpy.x);
            ggml_cuda_mad(sumf[j], tmpx1, tmpy.y);
        }
    }

    if constexpr (nt > 1) {
        __shared__ float buf_cols[nt][warp_size];
#pragma unroll
        for (int j = 0; j < nt; ++j) {
            sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);
            buf_cols[j][tid/warp_size] = sumf[j];
        }
        __syncthreads();
        if (tid < warp_size) {
#pragma unroll
            for (int j = 0; j < nt; ++j) {
                sumf[j] = warp_reduce_sum<warp_size>(tid < block_size/warp_size ? buf_cols[j][tid] : 0.0f);
            }
        }
    } else {
        sumf[0] = warp_reduce_sum<warp_size>(sumf[0]);
        buf_iw[tid/warp_size] = sumf[0];
        __syncthreads();
        if (tid < warp_size) {
            sumf[0] = buf_iw[tid];
            sumf[0] = warp_reduce_sum<warp_size>(sumf[0]);
        }
    }

    if (tid >= nt) {
        return;
    }
    const float value = sumf[tid];
    if (is_inject) {
        inject[tid*inject_stride_t + (row - nrows_down)] = value;
    } else {
        lo[tid*lo_stride_t + row] = value;
    }
}

template <int nt>
static __global__ void __launch_bounds__(4*32, 1) hc_mix_bf16_up_collapse(
        const nv_bfloat16 * w_up, const float * lo, const float * xn, float * dst,
        const int n_embd, const float inv_hc,
        const int64_t lo_stride_t, const int64_t xn_stride_t, const int64_t dst_stride_t) {
    constexpr int warp_size = 32;
    constexpr int hc        = 4;      // one warp per stream
    constexpr int vblock    = 160;    // the mmvf block for K = hc_lr = 320
    constexpr int nv        = vblock / warp_size;
    constexpr int ncols2    = vblock; // K/2: exactly one K iteration per virtual thread
    const int lane = threadIdx.x % warp_size;
    const int c    = threadIdx.x / warp_size;

    __shared__ float s_gate[hc][nt];
    __shared__ float2 s_v[nt][ncols2];

    // v = silu(lo/hc) once per block, as scale_unary_kernel<op_silu> computes it (scale*x + bias, bias 0)
    for (int i = threadIdx.x; i < nt*ncols2; i += blockDim.x) {
        const int j = i / ncols2;
        const int col2 = i % ncols2;
        const float lo0 = lo[j*lo_stride_t + 2*col2 + 0];
        const float lo1 = lo[j*lo_stride_t + 2*col2 + 1];
        const float bias = 0.0f;
        float2 v;
        v.x = ggml_cuda_op_silu_single(inv_hc * lo0 + bias);
        v.y = ggml_cuda_op_silu_single(inv_hc * lo1 + bias);
        s_v[j][col2] = v;
    }
    __syncthreads();

    for (int r = blockIdx.x; r < n_embd; r += gridDim.x) {
        const int row = c*n_embd + r;   // the up row of stream c, column r
        const int * x2 = (const int *) (w_up + (int64_t) row*(2*ncols2));

        float sumf[nv][nt];
#pragma unroll
        for (int v = 0; v < nv; ++v) {
#pragma unroll
            for (int j = 0; j < nt; ++j) {
                sumf[v][j] = 0.0f;
            }
        }
#pragma unroll
        for (int v = 0; v < nv; ++v) {
            const int col2 = v*warp_size + lane;
            const int tmpx = x2[col2];
#pragma unroll
            for (int j = 0; j < nt; ++j) {
                const float2 tmpy = s_v[j][col2];
                const float tmpx0 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[0]);
                const float tmpx1 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[1]);
                ggml_cuda_mad(sumf[v][j], tmpx0, tmpy.x);
                ggml_cuda_mad(sumf[v][j], tmpx1, tmpy.y);
            }
        }
#pragma unroll
        for (int j = 0; j < nt; ++j) {
            float total = 0.0f;
#pragma unroll
            for (int v = 0; v < nv; ++v) {
                const float wsum = warp_reduce_sum<warp_size>(sumf[v][j]);
                total = lane == v ? wsum : total;
            }
            total = warp_reduce_sum<warp_size>(total);
            if (lane == j) {
                s_gate[c][j] = total;
            }
        }
        __syncthreads();

        // gated collapse of column r, as dsv4_hc_pre_f32<gated> computes it
        if (threadIdx.x < nt) {
            const int t = threadIdx.x;
            float sum = 0.0f;
            for (int ih = 0; ih < hc; ++ih) {
                const float xv = xn[t*xn_stride_t + ih*n_embd + r];
                const float wr = s_gate[ih][t];
                const float wv = 1.0f / (1.0f + expf(-wr));
                sum += xv * wv;
            }
            const float v = inv_hc * sum;
            dst[t*dst_stride_t + r] = v;
        }
        __syncthreads();
    }
}

// The same rows and the same arithmetic as hc_mix_bf16_up_collapse, scheduled for latency: each warp computes the
// gates of all of its block's rows back to back (the next row's weights are loaded before the current row's
// butterflies, and there is no block barrier between rows), then one barrier, then the collapses of all the
// block's (row, token) pairs in parallel instead of nt threads per row.  Bit-identical to the original.
#define HC_UP2_MAXR 32
template <int nt>
static __global__ void __launch_bounds__(4*32, 1) hc_mix_bf16_up_collapse2(
        const nv_bfloat16 * w_up, const float * lo, const float * xn, float * dst,
        const int n_embd, const float inv_hc,
        const int64_t lo_stride_t, const int64_t xn_stride_t, const int64_t dst_stride_t) {
    constexpr int warp_size = 32;
    constexpr int hc        = 4;      // one warp per stream
    constexpr int vblock    = 160;    // the mmvf block for K = hc_lr = 320
    constexpr int nv        = vblock / warp_size;
    constexpr int ncols2    = vblock; // K/2: exactly one K iteration per virtual thread
    const int lane = threadIdx.x % warp_size;
    const int c    = threadIdx.x / warp_size;

    __shared__ float s_gate[HC_UP2_MAXR][hc][nt];
    __shared__ float2 s_v[nt][ncols2];

    // v = silu(lo/hc) once per block, as scale_unary_kernel<op_silu> computes it (scale*x + bias, bias 0)
    for (int i = threadIdx.x; i < nt*ncols2; i += blockDim.x) {
        const int j = i / ncols2;
        const int col2 = i % ncols2;
        const float lo0 = lo[j*lo_stride_t + 2*col2 + 0];
        const float lo1 = lo[j*lo_stride_t + 2*col2 + 1];
        const float bias = 0.0f;
        float2 v;
        v.x = ggml_cuda_op_silu_single(inv_hc * lo0 + bias);
        v.y = ggml_cuda_op_silu_single(inv_hc * lo1 + bias);
        s_v[j][col2] = v;
    }

    const int nr = (n_embd - (int) blockIdx.x + (int) gridDim.x - 1) / (int) gridDim.x;   // rows of this block

    int wnext[nv];
    if (nr > 0) {
        const int * x2 = (const int *) (w_up + (int64_t) (c*n_embd + blockIdx.x)*(2*ncols2));
#pragma unroll
        for (int v = 0; v < nv; ++v) {
            wnext[v] = x2[v*warp_size + lane];
        }
    }
    __syncthreads();

    for (int i = 0; i < nr; ++i) {
        int wcur[nv];
#pragma unroll
        for (int v = 0; v < nv; ++v) {
            wcur[v] = wnext[v];
        }
        if (i + 1 < nr) {
            const int * x2 = (const int *) (w_up + (int64_t) (c*n_embd + blockIdx.x + (i + 1)*gridDim.x)*(2*ncols2));
#pragma unroll
            for (int v = 0; v < nv; ++v) {
                wnext[v] = x2[v*warp_size + lane];
            }
        }

        float sumf[nv][nt];
#pragma unroll
        for (int v = 0; v < nv; ++v) {
#pragma unroll
            for (int j = 0; j < nt; ++j) {
                sumf[v][j] = 0.0f;
            }
        }
#pragma unroll
        for (int v = 0; v < nv; ++v) {
            const int col2 = v*warp_size + lane;
            const int tmpx = wcur[v];
#pragma unroll
            for (int j = 0; j < nt; ++j) {
                const float2 tmpy = s_v[j][col2];
                const float tmpx0 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[0]);
                const float tmpx1 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[1]);
                ggml_cuda_mad(sumf[v][j], tmpx0, tmpy.x);
                ggml_cuda_mad(sumf[v][j], tmpx1, tmpy.y);
            }
        }
        // The second butterfly of the original sums lane v = the warp sum of virtual warp v (v < 5) and zeros
        // elsewhere: its xor tree (16, 8, 4, 2, 1) reduces to ((w0 + w4) + w2) + (w1 + w3), and every lane holds
        // the same warp sums, so each lane computes that directly (x + 0 == x; only the sign of a zero total
        // could differ, and sigmoid(+-0) is the same gate).
        static_assert(nv == 5, "the closed-form second butterfly assumes 5 virtual warps");
#pragma unroll
        for (int j = 0; j < nt; ++j) {
            float w[nv];
#pragma unroll
            for (int v = 0; v < nv; ++v) {
                w[v] = warp_reduce_sum<warp_size>(sumf[v][j]);
            }
            const float total = ((w[0] + w[4]) + w[2]) + (w[1] + w[3]);
            if (lane == j) {
                s_gate[i][c][j] = total;
            }
        }
    }
    __syncthreads();

    // gated collapse of every (row, token) of the block, as dsv4_hc_pre_f32<gated> computes it
    for (int idx = threadIdx.x; idx < nr*nt; idx += blockDim.x) {
        const int i = idx / nt;
        const int t = idx % nt;
        const int r = blockIdx.x + i*gridDim.x;
        float sum = 0.0f;
        for (int ih = 0; ih < hc; ++ih) {
            const float xv = xn[t*xn_stride_t + ih*n_embd + r];
            const float wr = s_gate[i][ih][t];
            const float wv = 1.0f / (1.0f + expf(-wr));
            sum += xv * wv;
        }
        const float v = inv_hc * sum;
        dst[t*dst_stride_t + r] = v;
    }
}

static void ggml_cuda_op_hc_mix_bf16(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * comb = nullptr) {
    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * w_norm   = dst->src[1];
    const ggml_tensor * w_down   = dst->src[2];
    const ggml_tensor * w_up     = dst->src[3];
    const ggml_tensor * w_inject = dst->src[4];

    const int   hc  = ggml_get_op_params_i32(dst, 0);
    const float eps = ggml_get_op_params_f32(dst, 1);
    GGML_ASSERT(hc == 4);
    GGML_ASSERT(x->type == GGML_TYPE_F32 && w_norm->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(w_down->type == GGML_TYPE_BF16 && w_up->type == GGML_TYPE_BF16);
    GGML_ASSERT(w_inject == nullptr || w_inject->type == GGML_TYPE_BF16);

    const int64_t n_embd   = x->ne[0];
    const int64_t hc_dim   = n_embd * hc;
    const int64_t n_tokens = x->ne[2];
    const int64_t hc_lr    = w_down->ne[1];
    const int64_t out_n    = n_embd + (w_inject ? hc : 0);

    GGML_ASSERT(n_tokens >= 1 && n_tokens <= HC_FUSED_MAX_TOKENS);
    GGML_ASSERT(hc_lr == 320);                       // the up kernel is the K = 320 (160-thread) mmvf block
    GGML_ASSERT(hc_dim % 2 == 0 && n_embd % 32 == 0);
    GGML_ASSERT(x->ne[1] == hc && x->nb[2] == hc_dim*sizeof(float));
    GGML_ASSERT(ggml_is_contiguous(w_down) && ggml_is_contiguous(w_up) && (w_inject == nullptr || ggml_is_contiguous(w_inject)));
    GGML_ASSERT(w_up->ne[0] == hc_lr && w_up->ne[1] == hc_dim);
    GGML_ASSERT(dst->ne[0] == out_n && dst->ne[1] == n_tokens && dst->nb[1] == out_n*sizeof(float));

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<float> xn_alloc(pool, hc_dim*n_tokens);
    ggml_cuda_pool_alloc<float> lo_alloc(pool, hc_lr*n_tokens);
    float * xn    = xn_alloc.get();
    float * lo    = lo_alloc.get();
    float * dst_d = (float *) dst->data;
    // planar (ggml_hc_mix_set_planar): mixed rows [n_embd, T], then inject rows [hc, T]
    const bool    planar          = ggml_get_op_params_i32(dst, 2) != 0;
    float *       inject_d        = w_inject ? (planar ? dst_d + n_embd*n_tokens : dst_d + n_embd) : nullptr;
    const int64_t inject_stride_t = planar ? hc     : out_n;
    const int64_t mixed_stride_t  = planar ? n_embd : out_n;

    if (comb != nullptr) {
        // x == comb (see ggml_cuda_hc_combine_mix_fusable): the combine writes x and feeds the norm
        const ggml_tensor * residual  = comb->src[0];
        const ggml_tensor * block_out = comb->src[1];
        const ggml_tensor * inject    = comb->src[2];
        const ggml_cuda_kernel_launch_params launch_params = {dim3(hc, n_tokens), dim3(1024), 0, stream};
        ggml_cuda_kernel_launch(hc_mix_combine_rms_gamma, launch_params,
                (const float *) residual->data, (const float *) block_out->data, (const float *) inject->data,
                (float *) comb->data, (const float *) w_norm->data, xn, (int) n_embd, hc, eps,
                (int64_t) (residual->nb[2] / sizeof(float)),
                block_out->ne[1] == 1 ? (int64_t) 0 : (int64_t) (block_out->nb[1] / sizeof(float)),
                inject->ne[1]    == 1 ? (int64_t) 0 : (int64_t) (inject->nb[1]    / sizeof(float)),
                (int64_t) (comb->nb[2] / sizeof(float)), hc_dim);
    } else {
        const ggml_cuda_kernel_launch_params launch_params = {dim3(hc, n_tokens), dim3(1024), 0, stream};
        ggml_cuda_kernel_launch(hc_mix_rms_gamma_quant, launch_params,
                (const float *) x->data, (const float *) w_norm->data, xn, (block_q8_1 *) nullptr, (int) n_embd, eps,
                hc_dim, hc_dim, (int64_t) 0);
    }
    {
        const int nrows_inject = w_inject ? hc : 0;
        const ggml_cuda_kernel_launch_params launch_params = {dim3(hc_lr + nrows_inject), dim3(256), 0, stream};
#define HC_BF16_DOWN(NT) ggml_cuda_kernel_launch(hc_mix_bf16_down_inject<NT>, launch_params, \
        (const nv_bfloat16 *) w_down->data, w_inject ? (const nv_bfloat16 *) w_inject->data : nullptr, xn, lo, \
        inject_d, (int) (hc_dim/2), (int) hc_lr, hc_dim, hc_lr, inject_stride_t)
        switch (n_tokens) {
            case 1: HC_BF16_DOWN(1); break;
            case 2: HC_BF16_DOWN(2); break;
            case 3: HC_BF16_DOWN(3); break;
            case 4: HC_BF16_DOWN(4); break;
            case 5: HC_BF16_DOWN(5); break;
            case 6: HC_BF16_DOWN(6); break;
            case 7: HC_BF16_DOWN(7); break;
            case 8: HC_BF16_DOWN(8); break;
            default: GGML_ABORT("hc_mix bf16: unexpected n_tokens %d\n", (int) n_tokens);
        }
#undef HC_BF16_DOWN
    }
    {
        // GGML_HC_UP_V2=0: the original row-at-a-time kernel (v2 is bit-identical)
        static const bool up_v2 = !getenv("GGML_HC_UP_V2") || atoi(getenv("GGML_HC_UP_V2")) != 0;
        // at most 448 blocks of 4 warps (the gfx1201 grid-size stall, as mul_mat_vec_f_vb); v2: 256 blocks (flat from
        // 128 to 320 at n_embd 2560, slower above), at most HC_UP2_MAXR rows per block
        const int nblk = up_v2 ? (int) std::max<int64_t>(std::min<int64_t>(n_embd, 256), (n_embd + HC_UP2_MAXR - 1)/HC_UP2_MAXR)
                               : (int) std::min<int64_t>(n_embd, 448);
        const ggml_cuda_kernel_launch_params launch_params = {dim3(nblk), dim3(hc*32), 0, stream};
#define HC_BF16_UP(NT) do { if (up_v2) { ggml_cuda_kernel_launch(hc_mix_bf16_up_collapse2<NT>, launch_params, \
        (const nv_bfloat16 *) w_up->data, lo, xn, dst_d, (int) n_embd, 1.0f / (float) hc, hc_lr, hc_dim, mixed_stride_t); } else { \
        ggml_cuda_kernel_launch(hc_mix_bf16_up_collapse<NT>, launch_params, \
        (const nv_bfloat16 *) w_up->data, lo, xn, dst_d, (int) n_embd, 1.0f / (float) hc, hc_lr, hc_dim, mixed_stride_t); } } while (0)
        switch (n_tokens) {
            case 1: HC_BF16_UP(1); break;
            case 2: HC_BF16_UP(2); break;
            case 3: HC_BF16_UP(3); break;
            case 4: HC_BF16_UP(4); break;
            case 5: HC_BF16_UP(5); break;
            case 6: HC_BF16_UP(6); break;
            case 7: HC_BF16_UP(7); break;
            case 8: HC_BF16_UP(8); break;
            default: GGML_ABORT("hc_mix bf16: unexpected n_tokens %d\n", (int) n_tokens);
        }
#undef HC_BF16_UP
    }
}

void ggml_cuda_op_hc_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (dst->src[2]->type == GGML_TYPE_BF16) {
        ggml_cuda_op_hc_mix_bf16(ctx, dst);
        return;
    }
    const ggml_tensor * x         = dst->src[0];
    const ggml_tensor * w_norm    = dst->src[1];
    const ggml_tensor * w_down    = dst->src[2];
    const ggml_tensor * w_up      = dst->src[3];
    const ggml_tensor * w_inject  = dst->src[4];

    GGML_ASSERT(x->type        == GGML_TYPE_F32);
    GGML_ASSERT(w_norm->type   == GGML_TYPE_F32);
    GGML_ASSERT(w_down->type   == GGML_TYPE_Q8_0);
    GGML_ASSERT(w_up->type     == GGML_TYPE_Q8_0);
    GGML_ASSERT(w_inject == nullptr || w_inject->type == GGML_TYPE_F32 || w_inject->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(dst->type      == GGML_TYPE_F32);
    GGML_ASSERT(ggml_get_op_params_i32(dst, 2) == 0);   // planar layout: BF16 path only

    const int   hc  = ggml_get_op_params_i32(dst, 0);
    const float eps = ggml_get_op_params_f32(dst, 1);
    GGML_ASSERT(hc > 0 && hc <= 8);

    const int64_t n_embd   = x->ne[0];
    const int64_t hc_dim   = n_embd * hc;
    const int64_t n_tokens = x->ne[2];
    const int64_t hc_lr    = w_down->ne[1];

    GGML_ASSERT(n_tokens >= 1 && n_tokens <= HC_FUSED_MAX_TOKENS);  // decode/verify band
    GGML_ASSERT(hc_dim % 32 == 0 && hc_lr % 32 == 0);
    GGML_ASSERT(x->ne[1] == hc);
    GGML_ASSERT(x->nb[2] == hc_dim*sizeof(float));  // streams contiguous
    GGML_ASSERT(ggml_nelements(w_norm) == hc_dim);
    GGML_ASSERT(w_up->ne[0] == hc_lr && w_up->ne[1] == hc_dim);
    GGML_ASSERT(w_inject == nullptr || (w_inject->ne[0] == hc_dim && w_inject->ne[1] == hc));
    GGML_ASSERT(dst->ne[0] == n_embd + (w_inject ? hc : 0));
    // dst is [out_n, n_tokens]; its row is the per-token stride for every packed
    // scratch row this op writes (mixed head + F32 inject tail)
    const int64_t out_n = n_embd + (w_inject ? hc : 0);
    GGML_ASSERT(dst->ne[1] == n_tokens);
    GGML_ASSERT(dst->nb[1] == out_n*sizeof(float));

    // F32 inject: the mmvf float2 tail of the collapse launch; Q8_0 inject:
    // extra rows on the down-dots grid (mmvq, bit-identical to the unfused path)
    const bool inject_f32  = w_inject == nullptr || w_inject->type == GGML_TYPE_F32;
    const int  n_inject_q8 = w_inject && !inject_f32 ? (int) hc : 0;

    const float * x_d   = (const float *) x->data;
    const float * wn_d  = (const float *) w_norm->data;
    float * dst_d       = (float *) dst->data;
    float * inject      = w_inject ? dst_d + n_embd : nullptr;  // the inject tail

    // per-token element strides: x streams are contiguous within a token (asserted
    // above), the packed scratch rows are contiguous within their own arrays, and
    // both the mixed head and the inject tail live in a dst row of `out_n` floats
    const int64_t x_stride_t   = hc_dim;
    const int64_t dst_stride_t = out_n;

    cudaStream_t stream = ctx.stream();

    const int blocks_down = hc_dim / 32; // xn Q8_1 blocks for the down dots
    const int blocks_up   = hc_lr  / 32; // v  Q8_1 blocks for the up dots

    ggml_cuda_pool & pool = ctx.pool();

    ggml_cuda_pool_alloc<float>      xn_alloc(pool, hc_dim*n_tokens);
    ggml_cuda_pool_alloc<block_q8_1> y_xn_alloc(pool, blocks_down*n_tokens);
    ggml_cuda_pool_alloc<float>      lo_alloc(pool, hc_lr*n_tokens);
    ggml_cuda_pool_alloc<block_q8_1> y_v_alloc(pool, blocks_up*n_tokens);
    ggml_cuda_pool_alloc<float>      gate_alloc(pool, hc_dim*n_tokens);

    float      * xn   = xn_alloc.get();
    block_q8_1 * y_xn = y_xn_alloc.get();
    float      * lo   = lo_alloc.get();
    block_q8_1 * y_v  = y_v_alloc.get();
    float      * gate = gate_alloc.get();

    // rows-per-block as the mmvq dispatch chooses: 1 when the K-blocks fill the
    // thread groups, or the short-K override (RDNA2+) that packs RPB rows per
    // block so the item loop is bit-identical to the unfused path.
    constexpr int warp_size = 32;
    constexpr int qi  = QI8_0;
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    const auto calc_rpb = [&](int blocks_per_row) {
        const int n_groups = 8 * warp_size * vdr / qi;   // nwarps = 8
        int rpb = 1;
        if (blocks_per_row > 0 && blocks_per_row < n_groups) {
            int fill = (n_groups + blocks_per_row - 1) / blocks_per_row;
            int a = blocks_per_row, b = n_groups;
            while (b) { int t = a % b; a = b; b = t; }
            rpb = std::max(fill, n_groups / a);
            int pp = 1;
            while (pp < rpb) { pp <<= 1; }
            rpb = std::min(pp, 16);
        }
        return rpb;
    };
    const int rpb_down = calc_rpb(blocks_down);
    const int rpb_up   = calc_rpb(blocks_up);

    // xn = rms(x) * w_norm with the xn Q8_1 quantize merged in (grid hc x 1024 x nt)
    {
        const dim3 block_nums(hc, n_tokens);
        const dim3 block_dims(1024);
        const ggml_cuda_kernel_launch_params launch_params = {block_nums, block_dims, 0, stream};
        ggml_cuda_kernel_launch(hc_mix_rms_gamma_quant, launch_params,
                x_d, wn_d, xn, y_xn, (int) n_embd, eps,
                x_stride_t, hc_dim, blocks_down);
    }

    // lo_raw = w_down^T xn: 320 rows x 10240 dots (rpb is always 1 here);
    // a Q8_0 inject appends its hc rows to the same grid
    if (rpb_down != 1) {
        GGML_ABORT("hc_mix: unexpected down rpb %d\n", rpb_down);
    }

    // v = silu(lo/hc) quantized once, by the down-dots blocks, instead of in every up-dot
    // block's prologue (same arithmetic, bit-identical); GGML_CUDA_HC_MIX_PREQ=0 restores the
    // prologue. The band kernel serves all tokens of a >1-token band from one block per RPB rows
    // (bit-identical); GGML_CUDA_HC_MIX_BAND=0 restores the one-block-per-token launch.
    static const bool preq    = !getenv("GGML_CUDA_HC_MIX_PREQ") || std::atoi(getenv("GGML_CUDA_HC_MIX_PREQ"));
    static const bool up_band = !getenv("GGML_CUDA_HC_MIX_BAND") || std::atoi(getenv("GGML_CUDA_HC_MIX_BAND"));
    const int n_groups_up = 8 * warp_size * vdr / qi;
    const int maxi_up     = (rpb_up*blocks_up + n_groups_up - 1) / n_groups_up;
    const bool band       = up_band && n_tokens >= 1 && rpb_up == 16 && maxi_up <= 4 && blocks_up == 10;
    const bool pre        = preq && (band || rpb_up == 16);

    ggml_cuda_pool_alloc<unsigned int> done_alloc(pool);  // per (token, q8_1 group) row counters
    if (pre) {
        done_alloc.alloc(n_tokens*blocks_up);
        CUDA_CHECK(cudaMemsetAsync(done_alloc.get(), 0, n_tokens*blocks_up*sizeof(unsigned int), stream));
    }
    {
        const dim3 block_nums(hc_lr + n_inject_q8, n_tokens);
        const dim3 block_dims(32, 8);
        const ggml_cuda_kernel_launch_params launch_params = {block_nums, block_dims, 0, stream};
        ggml_cuda_kernel_launch(hc_mix_down_dots, launch_params,
                (const block_q8_0 *) w_down->data, lo, hc_lr,
                n_inject_q8 ? (const block_q8_0 *) w_inject->data : nullptr, inject, n_inject_q8,
                y_xn, blocks_down,
                blocks_down, hc_lr, dst_stride_t,
                pre ? y_v : nullptr, blocks_up, 1.0f / (float) hc, pre ? done_alloc.get() : nullptr);
    }

    // gate_raw = w_up^T v: 10240 rows x 320 dots (short K -> rpb override)
    if (band) {
        const dim3 block_nums((hc_dim + rpb_up - 1) / rpb_up);
        const dim3 block_dims(32, 8);
        const size_t smem = (size_t) n_tokens*blocks_up*sizeof(block_q8_1);
        const ggml_cuda_kernel_launch_params launch_params = {block_nums, block_dims, smem, stream};
#define HC_UP_BAND(NT) (pre \
        ? ggml_cuda_kernel_launch(hc_mix_up_silu_dot_band<8, 16, NT, 4, 10, true>,  launch_params, lo, y_v, (const block_q8_0 *) w_up->data, gate, (int) hc_dim, blocks_up, 1.0f / (float) hc, hc_lr, hc_dim) \
        : ggml_cuda_kernel_launch(hc_mix_up_silu_dot_band<8, 16, NT, 4, 10, false>, launch_params, lo, y_v, (const block_q8_0 *) w_up->data, gate, (int) hc_dim, blocks_up, 1.0f / (float) hc, hc_lr, hc_dim))
        switch (n_tokens) {
            case 1: HC_UP_BAND(1); break;
            case 2: HC_UP_BAND(2); break;
            case 3: HC_UP_BAND(3); break;
            case 4: HC_UP_BAND(4); break;
            case 5: HC_UP_BAND(5); break;
            case 6: HC_UP_BAND(6); break;
            case 7: HC_UP_BAND(7); break;
            case 8: HC_UP_BAND(8); break;
            default: GGML_ABORT("hc_mix: unexpected n_tokens %d\n", (int) n_tokens); break;
        }
#undef HC_UP_BAND
    } else {
        const dim3 block_nums((hc_dim + rpb_up - 1) / rpb_up, n_tokens);
        const dim3 block_dims(32, 8);
        const ggml_cuda_kernel_launch_params launch_params = {block_nums, block_dims, 0, stream};
        switch (rpb_up) {
            case 1:  ggml_cuda_kernel_launch(hc_mix_up_silu_dot<8, 1>,  launch_params, lo, (const block_q8_0 *) w_up->data, y_v, gate, hc_dim, blocks_up, 1.0f / (float) hc, hc_lr, blocks_up, hc_dim); break;
            case 2:  ggml_cuda_kernel_launch(hc_mix_up_silu_dot<8, 2>,  launch_params, lo, (const block_q8_0 *) w_up->data, y_v, gate, hc_dim, blocks_up, 1.0f / (float) hc, hc_lr, blocks_up, hc_dim); break;
            case 4:  ggml_cuda_kernel_launch(hc_mix_up_silu_dot<8, 4>,  launch_params, lo, (const block_q8_0 *) w_up->data, y_v, gate, hc_dim, blocks_up, 1.0f / (float) hc, hc_lr, blocks_up, hc_dim); break;
            case 8:  ggml_cuda_kernel_launch(hc_mix_up_silu_dot<8, 8>,  launch_params, lo, (const block_q8_0 *) w_up->data, y_v, gate, hc_dim, blocks_up, 1.0f / (float) hc, hc_lr, blocks_up, hc_dim); break;
            case 16:
                if (pre) {
                    ggml_cuda_kernel_launch(hc_mix_up_silu_dot<8, 16, true>, launch_params, lo, (const block_q8_0 *) w_up->data, y_v, gate, hc_dim, blocks_up, 1.0f / (float) hc, hc_lr, blocks_up, hc_dim);
                } else {
                    ggml_cuda_kernel_launch(hc_mix_up_silu_dot<8, 16>,       launch_params, lo, (const block_q8_0 *) w_up->data, y_v, gate, hc_dim, blocks_up, 1.0f / (float) hc, hc_lr, blocks_up, hc_dim);
                }
                break;
            default: GGML_ABORT("hc_mix: unexpected up rpb %d\n", rpb_up); break;
        }
    }

    // mixed at the dst head + the F32 inject at the dst tail in one dispatch:
    // the collapse blocks (n_embd/256) and the hc inject rows share the grid.
    // A Q8_0 inject is already in the tail from the down-dots launch above
    {
        const int n_collapse_blocks = (int) ((n_embd + 255) / 256);
        const int n_inject_blocks   = w_inject && inject_f32 ? hc : 0;
        const dim3 block_nums(n_collapse_blocks + n_inject_blocks, n_tokens);
        const dim3 block_dims(256);
        const ggml_cuda_kernel_launch_params launch_params = {block_nums, block_dims, 0, stream};
        ggml_cuda_kernel_launch(hc_mix_collapse_inject, launch_params,
                xn, gate, dst_d, n_embd, hc, 1.0f / (float) hc,
                w_inject && inject_f32 ? (const float *) w_inject->data : nullptr, inject,
                (int) (hc_dim / 2), n_collapse_blocks,
                hc_dim, hc_dim, dst_stride_t);
    }
}

// Fused hyper-connection residual combine for qwen4exp (decode/verify band).
// out[r, c, t] = residual[r, c, t] + block_out[r, t] * w[c, t], with
// w[c, t] = 2 * sigmoid(inject[c, t] / hc) (the SCALE+SIGMOID+SCALE chain of
// build_hc_combine; the 1/hc and 2.0 scalars are exact in f32 so only the
// sigmoid rounds). The product block_out[r]*w[c] and the residual add keep
// their own roundings because the reference runs separate MUL and ADD ops:
// the products go to an array first so the compiler cannot contract them
// into FMAs. One thread per row handles the hc columns, mirroring the
// collapse kernel. The token index is blockIdx.y and w is recomputed per token
// in registers (identical expression, so the nt == 1 result is unchanged);
// `inject` is often a view into the mix output, so it is read with its own
// per-token stride.
// Token dimension: the op serves the whole decode/verify band, nt <=
// HC_FUSED_MAX_TOKENS (the graph gates in src/models/qwen4exp.cpp use the same
// bound), which is what makes the band width-invariant.
static __global__ void hc_combine_kernel(
        const float * residual, const float * block_out, const float * inject,
        float * dst, const int n_embd, const int hc,
        const int64_t res_stride_t, const int64_t bo_stride_t,
        const int64_t inject_stride_t, const int64_t dst_stride_t) {
    const float inv_hc = 1.0f / (float) hc;
    const int t = blockIdx.y;

    // w[c] = 2*sigmoid(inject[c]/hc); per-thread registers rather than shared
    // memory so the early return below stays barrier-free
    const float * inj = inject + (int64_t) t*inject_stride_t;
    float w[8];
    for (int c = 0; c < hc; ++c) {
        const float s = inj[c] * inv_hc;
        w[c] = (1.0f / (1.0f + expf(-s))) * 2.0f;
    }

    const int r = blockIdx.x*blockDim.x + threadIdx.x;
    if (r >= n_embd) {
        return;
    }
    const float * res = residual + (int64_t) t*res_stride_t + r;
    const float   bo  = block_out[(int64_t) t*bo_stride_t + r];
    float       * dt  = dst + (int64_t) t*dst_stride_t + r;
    float pp[8];
    for (int c = 0; c < hc; ++c) {
        pp[c] = bo * w[c];
    }
    for (int c = 0; c < hc; ++c) {
        dt[c*n_embd] = res[c*n_embd] + pp[c];
    }
}

// HC_COMBINE immediately followed by the BF16 HC_MIX that reads its result: one kernel for the combine and the norm
// (hc_mix_combine_rms_gamma), then the mixer as usual.  The combine checks are ggml_cuda_op_hc_combine's.
bool ggml_cuda_hc_combine_mix_fusable(const ggml_tensor * comb, const ggml_tensor * mix) {
    if (comb->op != GGML_OP_HC_COMBINE || mix->op != GGML_OP_HC_MIX || mix->src[0] != comb) {
        return false;
    }
    const ggml_tensor * residual  = comb->src[0];
    const ggml_tensor * block_out = comb->src[1];
    const ggml_tensor * inject    = comb->src[2];
    const int64_t n_embd   = residual->ne[0];
    const int64_t n_tokens = residual->ne[2];
    const int     hc       = ggml_get_op_params_i32(comb, 0);
    return mix->src[2]->type == GGML_TYPE_BF16 && ggml_get_op_params_i32(mix, 0) == hc && hc == 4 &&
        residual->type == GGML_TYPE_F32 && block_out->type == GGML_TYPE_F32 && inject->type == GGML_TYPE_F32 &&
        comb->type == GGML_TYPE_F32 && residual->ne[1] == hc && n_tokens >= 1 && n_tokens <= HC_FUSED_MAX_TOKENS &&
        block_out->ne[0] == n_embd && inject->ne[0] == hc &&
        (block_out->ne[1] == n_tokens || block_out->ne[1] == 1) && (inject->ne[1] == n_tokens || inject->ne[1] == 1) &&
        residual->nb[1] == n_embd*sizeof(float) && residual->nb[2] == n_embd*hc*sizeof(float) &&
        comb->nb[1] == n_embd*sizeof(float) && comb->nb[2] == n_embd*hc*sizeof(float) &&
        n_embd <= 8*1024;
}

void ggml_cuda_op_hc_combine_mix(ggml_backend_cuda_context & ctx, ggml_tensor * comb, ggml_tensor * mix) {
    ggml_cuda_op_hc_mix_bf16(ctx, mix, comb);
}

void ggml_cuda_op_hc_combine(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * residual  = dst->src[0];
    const ggml_tensor * block_out = dst->src[1];
    const ggml_tensor * inject    = dst->src[2];

    GGML_ASSERT(residual->type  == GGML_TYPE_F32);
    GGML_ASSERT(block_out->type == GGML_TYPE_F32);
    GGML_ASSERT(inject->type    == GGML_TYPE_F32);
    GGML_ASSERT(dst->type       == GGML_TYPE_F32);

    const int hc = ggml_get_op_params_i32(dst, 0);
    GGML_ASSERT(hc > 0 && hc <= 8);

    const int64_t n_embd   = residual->ne[0];
    const int64_t n_tokens = residual->ne[2];

    GGML_ASSERT(n_tokens >= 1 && n_tokens <= HC_FUSED_MAX_TOKENS);  // decode/verify band
    GGML_ASSERT(residual->ne[1] == hc);
    GGML_ASSERT(block_out->ne[0] == n_embd);
    GGML_ASSERT(inject->ne[0] == hc);
    GGML_ASSERT(residual->nb[1] == n_embd*sizeof(float));   // contiguous rows
    GGML_ASSERT(dst->nb[1]      == n_embd*sizeof(float));
    GGML_ASSERT(residual->nb[2] == n_embd*hc*sizeof(float));  // contiguous tokens
    GGML_ASSERT(dst->nb[2]      == n_embd*hc*sizeof(float));
    GGML_ASSERT(block_out->ne[1] == n_tokens || block_out->ne[1] == 1);  // broadcast allowed
    GGML_ASSERT(inject->ne[1]    == n_tokens || inject->ne[1]    == 1);
    // block_out and inject may be broadcast (ne[1] == 1) and inject is often a
    // view into the mix output, whose row stride is the mix dst stride rather
    // than hc*4: a zero per-token stride expresses the broadcast, otherwise the
    // tensor's own nb[1] is the per-token step
    const int64_t res_stride_t    = residual->nb[2] / sizeof(float);
    const int64_t dst_stride_t    = dst->nb[2]      / sizeof(float);
    const int64_t bo_stride_t     = block_out->ne[1] == 1 ? 0 : block_out->nb[1] / sizeof(float);
    const int64_t inject_stride_t = inject->ne[1]    == 1 ? 0 : inject->nb[1]    / sizeof(float);

    cudaStream_t stream = ctx.stream();

    const dim3 block_nums((n_embd + 255) / 256, n_tokens);
    const dim3 block_dims(256);
    const ggml_cuda_kernel_launch_params launch_params = {block_nums, block_dims, 0, stream};
    ggml_cuda_kernel_launch(hc_combine_kernel, launch_params,
            (const float *) residual->data, (const float *) block_out->data,
            (const float *) inject->data, (float *) dst->data,
            (int) n_embd, hc,
            res_stride_t, bo_stride_t, inject_stride_t, dst_stride_t);
}
