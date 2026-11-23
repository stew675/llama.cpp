#include "tq.cuh"
#include "mma.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

// See tq.cuh.  The CPU reference: ggml-quants.c quantize_row_q8_K_ref (plain SSE2 build: mul, then add, no fma) and
// ggml-cpu/arch/x86/quants.c ggml_vec_dot_tq{1,2}_0_q8_K (AVX2: per block, 16-bit lanes k from byte pairs 2k, 2k+1 of
// every 32-value vector, minus bsums[k], then pairs of them as eight 32-bit lanes; sumf = add(mul(lane, y.d*x.d), sumf);
// hsum_float_8 at the end).  Lane L of a block holds the values whose index mod 32 is 4L..4L+3, minus the sum of the
// Q8_K values 32L..32L+31.
//
// NOTE: the CPU's `_mm256_add_ps(_mm256_mul_ps(..), sumf)` is NOT contracted -- a Release clang/GCC build of ggml-cpu
// emits vmulps + vaddps, with no vfmadd in ggml_vec_dot_tq{1,2}_0_q8_K (verified in the disassembly) -- so the block
// accumulation below is an UNFUSED mul and add.  `fmaf` there is a 1-ULP-per-block difference from the CPU and makes
// test-backend-ops' NMSE nonzero on every k > 256 (`GREEDY-PURITY.md` section 108).

// Products and sums that must not become an fma.  clang (HIP): plain operators in a scope with fp contract(off) -- HIP's
// __fmul_rn / __fadd_rn are inline header functions whose operators keep their own contractable flags and would still be
// fused after inlining.  nvcc: __fmul_rn / __fadd_rn, which it never contracts.
#if defined(__clang__)
#define TQ_FMUL(a, b) ((a)*(b))
#define TQ_FADD(a, b) ((a)+(b))
#else
#define TQ_FMUL(a, b) __fmul_rn(a, b)
#define TQ_FADD(a, b) __fadd_rn(a, b)
#endif // defined(__clang__)

#define TQ_LANES       8          // the CPU's 32-bit lanes per block
#define TQ_SMALL_NCOLS 8          // small-batch kernel: src1 columns per thread
#define TQ_SMALL_MAX   16         // batches up to this many columns take the small-batch kernel, larger ones the MMA kernel
#define TQ_VEC_THREADS 512        // vec kernel (decode): threads per block; every block quantizes src1 once
#define TQ_VEC_BPSM    2          // vec kernel: thread blocks per SM at most (they take turns on the row groups)
#define TQ_VEC_SMEM    (48*1024)  // vec kernel: the largest quantized src1 it keeps in shared memory
#define TQ_VEC_MAX     3          // vec kernel: weight matrices per launch
#define TQ_MMA_WARPS   4          // MMA kernel: warps per thread block, 16 weight rows each
#define TQ_MMA_BM      (16*TQ_MMA_WARPS) // MMA kernel: weight rows per thread block
#define TQ_MMA_BN      16         // MMA kernel: src1 columns per thread block

// Weight blocks the small-batch and vec kernels load ahead: more only crowds the caches (TQ1_0 has more decoding to cover).
template <ggml_type type> static constexpr __device__ int tq_unroll() { return type == GGML_TYPE_TQ1_0 ? 2 : 1; }

// src1 as quantize_row_q8_K_ref quantizes it, stored lane-major: q[32L + 4k + r] is value 32k + 4L + r, bsum[L] the sum of
// values 32L..32L+31 (bsums[2L] + bsums[2L + 1]).
struct block_q8_tq {
    int8_t q[QK_K];
    int    bsum[TQ_LANES];
    float  d;
    int    pad[3];
};
static_assert(sizeof(block_q8_tq) == 304, "block_q8_tq: 16-byte multiple");

// Q8_K exactly as quantize_row_q8_K_ref, by one warp: 256 values from xb (thread t holds values 8t..8t+7) into yb.
static __device__ __forceinline__ void tq_quantize_warp(const float * __restrict__ xb, block_q8_tq * __restrict__ yb, const int t) {
#if defined(__clang__)
#pragma clang fp contract(off) // see TQ_FMUL
#endif // defined(__clang__)
    static_assert(WARP_SIZE == 32, "8 values a thread");
    float v[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        v[k] = xb[8*t + k];
    }
    // the largest magnitude; 0 when every value is zero or NaN, as the CPU's scan from 0 with a strict '>'
    float amax = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        amax = fmaxf(amax, fabsf(v[k]));
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(GGML_CUDA_FULL_THREAD_MASK, amax, off, WARP_SIZE));
    }
    // the value at its first position (the CPU keeps the first one): the lowest thread with a match, its lowest k
    float vk  = 0.0f;
    bool  hit = false;
#pragma unroll
    for (int k = 7; k >= 0; --k) {
        if (fabsf(v[k]) == amax) {
            vk  = v[k];
            hit = true;
        }
    }
    const int   src  = __ffsll((long long) __ballot_sync(GGML_CUDA_FULL_THREAD_MASK, hit)) - 1;
    const float vmax = __shfl_sync(GGML_CUDA_FULL_THREAD_MASK, vk, src, WARP_SIZE);

    // values 8t..8t+3 and 8t+4..8t+7 are 4 consecutive bytes each of the lane-major order
    int * yq = (int *) yb->q;
    if (amax == 0.0f) {
        yq[8*(2*(t % 4) + 0) + t/4] = 0;
        yq[8*(2*(t % 4) + 1) + t/4] = 0;
        if (t < TQ_LANES) {
            yb->bsum[t] = 0;
        }
        if (t == 0) {
            yb->d = 0.0f;
        }
        return;
    }

    const float iscale = __fdiv_rn(-127.0f, vmax);
    int sum   = 0;
    int pk[2] = {0, 0};
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        // nearest_int(iscale*x): the float add of 1.5*2^23 rounds to an integer in the mantissa
        const float val = TQ_FADD(TQ_FMUL(iscale, v[k]), 12582912.0f);
        const int   q   = min(127, (__float_as_int(val) & 0x007fffff) - 0x00400000);
        pk[k/4] |= (q & 0xFF) << (8*(k % 4));
        sum += q;
    }
    yq[8*(2*(t % 4) + 0) + t/4] = pk[0];
    yq[8*(2*(t % 4) + 1) + t/4] = pk[1];
    // bsum[L]: values 32L..32L+31, held by threads 4L..4L+3
    sum += __shfl_xor_sync(GGML_CUDA_FULL_THREAD_MASK, sum, 1, WARP_SIZE);
    sum += __shfl_xor_sync(GGML_CUDA_FULL_THREAD_MASK, sum, 2, WARP_SIZE);
    if (t % 4 == 0) {
        yb->bsum[t/4] = sum;
    }
    if (t == 0) {
        yb->d = __fdiv_rn(1.0f, iscale);
    }
}

// src1 to block_q8_tq: one warp per 256 values.
static __global__ void quantize_q8_tq(
        const float * __restrict__ x, block_q8_tq * __restrict__ y, const int nb,
        const int64_t s11, const int64_t s12, const int64_t s13, const int64_t ne11, const int64_t ne12) {
    const int     ib  = blockIdx.x*blockDim.y + threadIdx.y;
    const int64_t i11 = blockIdx.y;
    const int64_t i12 = blockIdx.z % ne12;
    const int64_t i13 = blockIdx.z / ne12;
    if (ib >= nb) {
        return;
    }
    tq_quantize_warp(x + i13*s13 + i12*s12 + i11*s11 + (int64_t) ib*QK_K, y + ((int64_t) blockIdx.z*ne11 + i11)*nb + ib, threadIdx.x);
}

template <ggml_type type> struct tq_block;
template <> struct tq_block<GGML_TYPE_TQ1_0> { typedef block_tq1_0 t; };
template <> struct tq_block<GGML_TYPE_TQ2_0> { typedef block_tq2_0 t; };

// The bytes of lane L of one weight block, loaded ahead of their decoding.
template <ggml_type type> struct tq_raw;
template <> struct tq_raw<GGML_TYPE_TQ2_0> { int w0; int w1; float d; };       // qs[4L..], qs[32 + 4L..]
template <> struct tq_raw<GGML_TYPE_TQ1_0> { int a; int b; int h; float d; };  // qs[4L..], qs[32 + 4L%16..], qh

static __device__ __forceinline__ tq_raw<GGML_TYPE_TQ2_0> tq_load(const block_tq2_0 * xb, const int L) {
    return { get_int_b2(xb->qs, L), get_int_b2(xb->qs + 32, L), __half2float(xb->d) };
}

static __device__ __forceinline__ tq_raw<GGML_TYPE_TQ1_0> tq_load(const block_tq1_0 * xb, const int L) {
    return { get_int_b2(xb->qs, L), get_int_b2(xb->qs + 32, L % 4), get_int_b2(xb->qh, 0), __half2float(xb->d) };
}

template <ggml_type type>
static __device__ __forceinline__ tq_raw<type> tq_load_block(const char * __restrict__ xr, const int ib, const int L) {
    if constexpr (type == GGML_TYPE_TQ2_0) {
        return tq_load((const block_tq2_0 *) xr + ib, L);
    } else {
        return tq_load((const block_tq1_0 *) xr + ib, L);
    }
}

// digit n (0..4) of each byte of w, as ggml-quants.c: ((uint8_t) (b*3^n) * 3) >> 8, packed back into bytes; p = 3^n.  The
// even and the odd bytes go through in two 16-bit fields at once: b*3^n < 2^15, so neither field carries into the other.
static __device__ __forceinline__ int tq1_0_digits4(const int w, const uint32_t p) {
    const uint32_t xe = (((uint32_t) w        & 0x00FF00FFu)*p & 0x00FF00FFu)*3;
    const uint32_t xo = ((((uint32_t) w >> 8) & 0x00FF00FFu)*p & 0x00FF00FFu)*3;
    return (int) (((xe >> 8) & 0x00030003u) | (xo & 0x03000300u));
}

// The trits 0..2 of lane L: t[k] packs values 32k + 4L + 0..3 (k = 0..7).
static __device__ __forceinline__ void tq_decode(const tq_raw<GGML_TYPE_TQ2_0> & raw, const int L, int * t) {
    GGML_UNUSED(L);
    // value 128h + 32l + m is bits 2l..2l+1 of qs[32h + m]
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        t[l]     = (raw.w0 >> (2*l)) & 0x03030303;
        t[4 + l] = (raw.w1 >> (2*l)) & 0x03030303;
    }
}

static __device__ __forceinline__ void tq_decode(const tq_raw<GGML_TYPE_TQ1_0> & raw, const int L, int * t) {
    // values 0..159: digit e/32 of qs[e%32]; 160..239: digit (e-160)/16 of qs[32 + (e-160)%16]; 240..255: digit (e-240)/4
    // of qh[(e-240)%4].  Digits 0..4 of qs[4L..]: x <- x*3 mod 256 steps b*3^n, and (x*3) >> 8 is digit n (16-bit fields
    // as in tq1_0_digits4).
    uint32_t xe = (uint32_t) raw.a        & 0x00FF00FFu;
    uint32_t xo = ((uint32_t) raw.a >> 8) & 0x00FF00FFu;
#pragma unroll
    for (int k = 0; k < 5; ++k) {
        const uint32_t ye = xe*3;
        const uint32_t yo = xo*3;
        t[k] = (int) (((ye >> 8) & 0x00030003u) | (yo & 0x03000300u));
        xe = ye & 0x00FF00FFu;
        xo = yo & 0x00FF00FFu;
    }
    // lanes 0..3: digits 0, 2, 4 of qs[32 + 4L..]; lanes 4..7: digits 1, 3 of qs[32 + 4(L-4)..] and digit L-4 of qh
    const bool lo = L < 4;
    t[5] = tq1_0_digits4(raw.b, lo ? 1 : 3);
    t[6] = tq1_0_digits4(raw.b, lo ? 9 : 27);
    t[7] = tq1_0_digits4(lo ? raw.b : raw.h, lo ? 81 : (0x1B090301u >> (8*(L & 3))) & 0xFF);
}

// The integer dot of one block and CPU lane L with the Q8 values of that lane, minus the lane's Q8 sum.
static __device__ __forceinline__ int tq_lane_dot(const int * t, const int4 ya, const int4 yc, const int bsum) {
    int s = -bsum;
    s = ggml_cuda_dp4a(t[0], ya.x, s);
    s = ggml_cuda_dp4a(t[1], ya.y, s);
    s = ggml_cuda_dp4a(t[2], ya.z, s);
    s = ggml_cuda_dp4a(t[3], ya.w, s);
    s = ggml_cuda_dp4a(t[4], yc.x, s);
    s = ggml_cuda_dp4a(t[5], yc.y, s);
    s = ggml_cuda_dp4a(t[6], yc.z, s);
    s = ggml_cuda_dp4a(t[7], yc.w, s);
    return s;
}

// hsum_float_8 over the TQ_LANES threads of a row: ((l4 + l0) + (l6 + l2)) + ((l5 + l1) + (l7 + l3))
static __device__ __forceinline__ float tq_lane_hsum(float v) {
    v += ggml_cuda_shfl_xor(v, 4, TQ_LANES);
    v += ggml_cuda_shfl_xor(v, 2, TQ_LANES);
    v += ggml_cuda_shfl_xor(v, 1, TQ_LANES);
    return v;
}

// Small batches: TQ_LANES threads per weight row (one per CPU lane), up to TQ_SMALL_NCOLS src1 columns per thread.
template <ggml_type type>
static __global__ void mul_mat_tq_small(
        const char * __restrict__ vx, const block_q8_tq * __restrict__ y, float * __restrict__ dst,
        const int nrows, const int nb, const int ncols, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne12, const int r2, const int r3, const int64_t s1, const int64_t s2, const int64_t s3) {
#if defined(__clang__)
#pragma clang fp contract(off) // see TQ_FMUL
#endif // defined(__clang__)
    constexpr int U   = tq_unroll<type>();
    const int     L   = threadIdx.x % TQ_LANES;
    const int     row = blockIdx.x*(blockDim.x/TQ_LANES) + threadIdx.x/TQ_LANES;
    const int     c0  = blockIdx.y*TQ_SMALL_NCOLS;
    const int     nc  = min(TQ_SMALL_NCOLS, ncols - c0);
    const int64_t i12 = blockIdx.z % ne12;
    const int64_t i13 = blockIdx.z / ne12;
    if (row >= nrows) {
        return;
    }

    const block_q8_tq * yz = y + ((int64_t) blockIdx.z*ncols + c0)*nb;
    const char        * xr = vx + (i13/r3)*s03 + (i12/r2)*s02 + (int64_t) row*s01;
    float acc[TQ_SMALL_NCOLS] = {0.0f};
    for (int ib0 = 0; ib0 < nb; ib0 += U) {
        tq_raw<type> raw[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            if (ib0 + u < nb) {
                raw[u] = tq_load_block<type>(xr, ib0 + u, L);
            }
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            if (ib0 + u >= nb) {
                break;
            }
            int t[8];
            tq_decode(raw[u], L, t);
#pragma unroll
            for (int c = 0; c < TQ_SMALL_NCOLS; ++c) {
                if (c < nc) {
                    const block_q8_tq * yb = yz + (int64_t) c*nb + ib0 + u;
                    const int4 ya = ((const int4 *) (yb->q + 32*L))[0];
                    const int4 yc = ((const int4 *) (yb->q + 32*L))[1];
                    acc[c] = TQ_FADD(TQ_FMUL((float) tq_lane_dot(t, ya, yc, yb->bsum[L]), TQ_FMUL(yb->d, raw[u].d)), acc[c]);
                }
            }
        }
    }

#pragma unroll
    for (int c = 0; c < TQ_SMALL_NCOLS; ++c) {
        const float v = tq_lane_hsum(acc[c]);
        if (L == 0 && c < nc) {
            dst[i13*s3 + i12*s2 + (int64_t) (c0 + c)*s1 + row] = v;
        }
    }
}

// Decode (one src1 column): up to TQ_VEC_MAX weight matrices over the same src1 in one launch, their rows one after
// another (row_end: the running row counts); or with glu the gate (0) and up (1) rows of an SWIGLU, written as
// up*silu(gate); or (n == 1) the result plus add's element (a residual ADD).  Every thread block quantizes src1 into
// shared memory first, which saves a quantize launch, then takes groups of blockDim.x/TQ_LANES rows, gridDim.x apart.
// Every row's dot product is the small kernel's.
struct tq_vec_args {
    const char  * x[TQ_VEC_MAX];
    float       * dst[TQ_VEC_MAX];
    int64_t       s01[TQ_VEC_MAX];
    int           row_end[TQ_VEC_MAX];
    int           n;
    const float * add; // or nullptr
};

template <ggml_type type, bool glu>
static __global__ void mul_mat_tq_vec(
        const tq_vec_args a, const float * __restrict__ x1, const int nb, const int64_t s02, const int64_t s03,
        const int64_t ne12, const int r2, const int r3, const int64_t s12_1, const int64_t s13_1, const int64_t s2,
        const int64_t s3) {
#if defined(__clang__)
#pragma clang fp contract(off) // see TQ_FMUL
#endif // defined(__clang__)
    constexpr int U   = tq_unroll<type>();
    constexpr int R   = glu ? 2 : 1; // weight rows a thread: gate and up
    const int     L   = threadIdx.x % TQ_LANES;
    const int     rpb = blockDim.x/TQ_LANES;
    const int64_t i12 = blockIdx.z % ne12;
    const int64_t i13 = blockIdx.z / ne12;

    extern __shared__ int4 tq_smem[];
    block_q8_tq * ys = (block_q8_tq *) tq_smem;
    {
        const float * xc = x1 + i13*s13_1 + i12*s12_1;
        for (int qb = threadIdx.x/WARP_SIZE; qb < nb; qb += blockDim.x/WARP_SIZE) {
            tq_quantize_warp(xc + (int64_t) qb*QK_K, ys + qb, threadIdx.x % WARP_SIZE);
        }
    }
    __syncthreads();

    const int64_t woff  = (i13/r3)*s03 + (i12/r2)*s02;
    const int     total = glu ? a.row_end[0] : a.row_end[a.n - 1];
    for (int row = blockIdx.x*rpb + threadIdx.x/TQ_LANES; row < total; row += gridDim.x*rpb) {
        const char * xr[R];
        float      * d;
        int          lr;
        if constexpr (glu) {
            lr    = row;
            xr[0] = a.x[0] + woff + (int64_t) lr*a.s01[0];
            xr[1] = a.x[1] + woff + (int64_t) lr*a.s01[1];
            d     = a.dst[0];
        } else {
            const int m = row < a.row_end[0] ? 0 : row < a.row_end[1] ? 1 : 2;
            lr    = row - (m == 0 ? 0 : a.row_end[m - 1]);
            xr[0] = (m == 0 ? a.x[0] : m == 1 ? a.x[1] : a.x[2]) + woff + (int64_t) lr*(m == 0 ? a.s01[0] : m == 1 ? a.s01[1] : a.s01[2]);
            d     = m == 0 ? a.dst[0] : m == 1 ? a.dst[1] : a.dst[2];
        }

        float acc[R] = {0.0f};
        for (int ib0 = 0; ib0 < nb; ib0 += U) {
            tq_raw<type> raw[R][U];
#pragma unroll
            for (int u = 0; u < U; ++u) {
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    if (ib0 + u < nb) {
                        raw[r][u] = tq_load_block<type>(xr[r], ib0 + u, L);
                    }
                }
            }
#pragma unroll
            for (int u = 0; u < U; ++u) {
                if (ib0 + u >= nb) {
                    break;
                }
                const block_q8_tq * yb = ys + ib0 + u;
                const int4  ya = ((const int4 *) (yb->q + 32*L))[0];
                const int4  yc = ((const int4 *) (yb->q + 32*L))[1];
                const int   bs = yb->bsum[L];
                const float yd = yb->d;
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    int t[8];
                    tq_decode(raw[r][u], L, t);
                    acc[r] = TQ_FADD(TQ_FMUL((float) tq_lane_dot(t, ya, yc, bs), TQ_FMUL(yd, raw[r][u].d)), acc[r]);
                }
            }
        }

        float v = tq_lane_hsum(acc[0]);
        if constexpr (glu) {
            // as the GLU kernel computes it (the product is the same either way round)
            v = ggml_cuda_op_glu_single(GGML_GLU_OP_SWIGLU, v, tq_lane_hsum(acc[1]), 0.0f);
        } else if (a.add != nullptr) {
            v += a.add[lr]; // as the ADD kernel computes it (one rounding, either order)
        }
        if (L == 0) {
            d[i13*s3 + i12*s2 + lr] = v;
        }
    }
}

// The weight bytes one thread of the MMA kernel needs per block: its row (lane % 16), its half h (lane / 16) of the
// lane-major 8 ints t[0..7] of every CPU lane L, i.e. t[4h..4h + 3].
template <ggml_type type> struct tq_a_raw;
template <> struct tq_a_raw<GGML_TYPE_TQ2_0> { int w[8]; float d; };                 // w[L] = qs[32h + 4L..]

static __device__ __forceinline__ void tq_a_load(const block_tq2_0 * xb, const int h, tq_a_raw<GGML_TYPE_TQ2_0> & r) {
#pragma unroll
    for (int L = 0; L < TQ_LANES; ++L) {
        r.w[L] = get_int_b2(xb->qs + 32*h, L);
    }
    r.d = __half2float(xb->d);
}

// t[4h + l], l = 0..3, of lane L (see tq_decode)
static __device__ __forceinline__ void tq_a_tile(const tq_a_raw<GGML_TYPE_TQ2_0> & r, const int h, const int L, int * x) {
    GGML_UNUSED(h);
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        x[l] = (r.w[L] >> (2*l)) & 0x03030303;
    }
}

// Larger batches (RDNA4 WMMA): one int8 MMA (K = 32) per block, CPU lane and 16x16 output tile gives exactly that lane's
// integer dot, the accumulator starting at minus the lane's Q8_K sum; the float part is the small kernel's, element by
// element.  Every warp works on its own 16 weight rows and 16 src1 columns straight from global memory into the WMMA
// registers (no shared memory, no barrier): thread i + 16h holds weight row i and src1 column i, half h of each lane's
// 32 values (16x8 int tiles, I-major on WMMA128), and output rows 8h..8h + 7 of column i (the J-major result tile).  The
// next block's data is loaded while the current one is computed.  TQ2_0 only: TQ1_0 is converted to it first.
template <ggml_type type>
static __global__ void __launch_bounds__(TQ_MMA_WARPS*WARP_SIZE, 1) mul_mat_tq_mma(
        const char * __restrict__ vx, const block_q8_tq * __restrict__ y, float * __restrict__ dst,
        const int nrows, const int nb, const int ncols, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne12, const int r2, const int r3, const int64_t s1, const int64_t s2, const int64_t s3) {
#if defined(AMD_WMMA_AVAILABLE) && defined(RDNA4)
#if defined(__clang__)
#pragma clang fp contract(off) // see TQ_FMUL
#endif // defined(__clang__)
    static_assert(type == GGML_TYPE_TQ2_0, "TQ1_0 is converted to TQ2_0 first");
    using namespace ggml_cuda_mma;
    typedef tile<16,  8, int, DATA_LAYOUT_I_MAJOR> tile_ab;
    typedef tile<16, 16, int, DATA_LAYOUT_J_MAJOR> tile_d;
    static_assert(tile_ab::ne == 4 && tile_d::ne == 8, "WMMA128 tiles");

    const int     i    = threadIdx.x % 16;
    const int     h    = threadIdx.x / 16;
    const int     c0   = blockIdx.x*16;
    const int     wr0  = blockIdx.y*TQ_MMA_BM + threadIdx.y*16;
    const int64_t i12  = blockIdx.z % ne12;
    const int64_t i13  = blockIdx.z / ne12;
    const bool    a_ok = wr0 + i < nrows;
    const bool    b_ok = c0 + i < ncols;

    const char        * xr = vx + (i13/r3)*s03 + (i12/r2)*s02 + (int64_t) (wr0 + i)*s01;
    const block_q8_tq * yc = y + (int64_t) blockIdx.z*ncols*nb + (int64_t) (c0 + i)*nb;

    // the next block, in registers
    tq_a_raw<type> an = {};
    int4  bn[TQ_LANES];
    int4  sn0 = make_int4(0, 0, 0, 0);
    int4  sn1 = make_int4(0, 0, 0, 0);
    float yn  = 0.0f;
#pragma unroll
    for (int L = 0; L < TQ_LANES; ++L) {
        bn[L] = make_int4(0, 0, 0, 0);
    }
#define TQ_MMA_LOAD(ib_)                                                                                     \
    {                                                                                                        \
        if (a_ok) {                                                                                          \
            tq_a_load((const typename tq_block<type>::t *) xr + (ib_), h, an);                               \
        }                                                                                                    \
        if (b_ok) {                                                                                          \
            const block_q8_tq * yb = yc + (ib_);                                                             \
            _Pragma("unroll")                                                                                \
            for (int L = 0; L < TQ_LANES; ++L) {                                                             \
                bn[L] = *(const int4 *) (yb->q + 32*L + 16*h);                                               \
            }                                                                                                \
            sn0 = *(const int4 *) (yb->bsum + 0);                                                            \
            sn1 = *(const int4 *) (yb->bsum + 4);                                                            \
            yn  = yb->d;                                                                                     \
        }                                                                                                    \
    }

    float acc[TQ_LANES][tile_d::ne];
#pragma unroll
    for (int L = 0; L < TQ_LANES; ++L) {
#pragma unroll
        for (int l = 0; l < tile_d::ne; ++l) {
            acc[L][l] = 0.0f;
        }
    }

    TQ_MMA_LOAD(0);
    for (int ib = 0; ib < nb; ++ib) {
        const tq_a_raw<type> ac = an;
        int4 bc[TQ_LANES];
#pragma unroll
        for (int L = 0; L < TQ_LANES; ++L) {
            bc[L] = bn[L];
        }
        const int   bs[TQ_LANES] = { sn0.x, sn0.y, sn0.z, sn0.w, sn1.x, sn1.y, sn1.z, sn1.w };
        const float yd = yn;
        if (ib + 1 < nb) {
            TQ_MMA_LOAD(ib + 1); // in flight while this block is computed
        }

        // dd = y.d*x.d for output rows 8h + l (weight row 8h + l is held by thread 8h + l) of column i
        const float dx = a_ok ? ac.d : 0.0f;
        float dd[tile_d::ne];
#pragma unroll
        for (int l = 0; l < tile_d::ne; ++l) {
            dd[l] = TQ_FMUL(yd, __shfl_sync(GGML_CUDA_FULL_THREAD_MASK, dx, 8*h + l, WARP_SIZE));
        }
#pragma unroll
        for (int L = 0; L < TQ_LANES; ++L) {
            tile_ab A;
            tile_ab B;
            int x[4] = {0, 0, 0, 0};
            if (a_ok) {
                tq_a_tile(ac, h, L, x);
            }
            A.x[0] = x[0]; A.x[1] = x[1]; A.x[2] = x[2]; A.x[3] = x[3];
            B.x[0] = bc[L].x; B.x[1] = bc[L].y; B.x[2] = bc[L].z; B.x[3] = bc[L].w;
            tile_d D;
#pragma unroll
            for (int l = 0; l < tile_d::ne; ++l) {
                D.x[l] = -bs[L];
            }
            mma(D, A, B);
#pragma unroll
            for (int l = 0; l < tile_d::ne; ++l) {
                acc[L][l] = TQ_FADD(TQ_FMUL((float) D.x[l], dd[l]), acc[L][l]);
            }
        }
    }

#pragma unroll
    for (int l = 0; l < tile_d::ne; ++l) {
        const int row = wr0 + 8*h + l;
        const int col = c0 + i;
        // hsum_float_8: ((l4 + l0) + (l6 + l2)) + ((l5 + l1) + (l7 + l3))
        const float v = ((acc[4][l] + acc[0][l]) + (acc[6][l] + acc[2][l])) + ((acc[5][l] + acc[1][l]) + (acc[7][l] + acc[3][l]));
        if (row < nrows && col < ncols) {
            dst[i13*s3 + i12*s2 + (int64_t) col*s1 + row] = v;
        }
    }
#undef TQ_MMA_LOAD
#else
    GGML_UNUSED_VARS(vx, y, dst, nrows, nb, ncols, s01, s02, s03, ne12, r2, r3, s1, s2, s3);
    NO_DEVICE_CODE;
#endif // defined(AMD_WMMA_AVAILABLE) && defined(RDNA4)
}

// TQ1_0 -> TQ2_0, value for value (the same trits, the same d), so the MMA kernel reads two bits a value instead of decoding
// base 3 once per column tile.  One thread per output byte: value 128h + 32l + m goes to bits 2l..2l+1 of qs[32h + m].
static __global__ void convert_tq1_0_tq2_0(const block_tq1_0 * __restrict__ x, block_tq2_0 * __restrict__ y, const int64_t nblocks) {
    const int64_t ib = blockIdx.x*(int64_t) (blockDim.x/64) + threadIdx.x/64;
    const int     b  = threadIdx.x % 64;
    if (ib >= nblocks) {
        return;
    }
    const int h = b / 32;
    const int m = b % 32;
    int q = 0;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        q |= ggml_cuda_tq1_0_trit(x + ib, 128*h + 32*l + m) << (2*l);
    }
    y[ib].qs[b] = q;
    if (b == 0) {
        y[ib].d = x[ib].d;
    }
}

bool ggml_cuda_mul_mat_tq_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    if (src0->type != GGML_TYPE_TQ1_0 && src0->type != GGML_TYPE_TQ2_0) {
        return false;
    }
    return src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        src0->nb[0] == ggml_type_size(src0->type) && src1->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float) &&
        src0->ne[0] % QK_K == 0 && src1->ne[2] % src0->ne[2] == 0 && src1->ne[3] % src0->ne[3] == 0 &&
        src1->ne[1] <= 65535;
}

template <ggml_type type>
static void ggml_cuda_mul_mat_tq_launch(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const block_q8_tq * y, const int nb) {
    GGML_TENSOR_BINARY_OP_LOCALS;
    cudaStream_t stream = ctx.stream();

    const int  cc  = ggml_cuda_info().devices[ctx.device].cc;
    // elsewhere the small-batch kernel takes every batch; TQ1_0's MMA path converts the whole weight flat
    // (below), so a weight that is not contiguous keeps the small-batch kernel there instead of asserting.
    const bool mma = ne11 > TQ_SMALL_MAX && GGML_CUDA_CC_IS_RDNA4(cc) &&
        (type != GGML_TYPE_TQ1_0 || ggml_is_contiguous(src0));
    const int  r2  = ne12/ne02;
    const int  r3  = ne13/ne03;
    ctx.tq_census[mma ? 1 : 0]++;
    if (mma) {
        const dim3 grid((ne11 + TQ_MMA_BN - 1)/TQ_MMA_BN, (ne01 + TQ_MMA_BM - 1)/TQ_MMA_BM, ne12*ne13);
        const dim3 block(WARP_SIZE, TQ_MMA_WARPS, 1);
        if constexpr (type == GGML_TYPE_TQ1_0) {
            // the same values as TQ2_0, contiguous over all of src0's matrices (rows of nb02 / nb03 strides kept)
            const int64_t nblocks = ggml_nelements(src0)/QK_K;
            ggml_cuda_pool_alloc<block_tq2_0> x2(ctx.pool(), nblocks);
            convert_tq1_0_tq2_0<<<(nblocks + 3)/4, 256, 0, stream>>>((const block_tq1_0 *) src0->data, x2.get(), nblocks);
            CUDA_CHECK(cudaGetLastError());
            const int64_t s01_2 = nb*sizeof(block_tq2_0);
            mul_mat_tq_mma<GGML_TYPE_TQ2_0><<<grid, block, 0, stream>>>((const char *) x2.get(), y, (float *) dst->data,
                ne01, nb, ne11, s01_2, s01_2*ne01, s01_2*ne01*ne02, ne12, r2, r3, nb1/sizeof(float), nb2/sizeof(float), nb3/sizeof(float));
        } else {
            mul_mat_tq_mma<type><<<grid, block, 0, stream>>>((const char *) src0->data, y, (float *) dst->data,
                ne01, nb, ne11, nb01, nb02, nb03, ne12, r2, r3, nb1/sizeof(float), nb2/sizeof(float), nb3/sizeof(float));
        }
    } else {
        const int  rows_per_block = 256/TQ_LANES;
        const dim3 grid((ne01 + rows_per_block - 1)/rows_per_block, (ne11 + TQ_SMALL_NCOLS - 1)/TQ_SMALL_NCOLS, ne12*ne13);
        mul_mat_tq_small<type><<<grid, 256, 0, stream>>>((const char *) src0->data, y, (float *) dst->data,
            ne01, nb, ne11, nb01, nb02, nb03, ne12, r2, r3, nb1/sizeof(float), nb2/sizeof(float), nb3/sizeof(float));
    }
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_mul_mat_tq_vec_ok(const ggml_tensor * mm) {
    const ggml_tensor * src0 = mm->src[0];
    const ggml_tensor * src1 = mm->src[1];
    return mm->op == GGML_OP_MUL_MAT && ggml_cuda_mul_mat_tq_supported(src0, src1, mm) && src1->ne[1] == 1 &&
        (size_t) (src0->ne[0]/QK_K)*sizeof(block_q8_tq) <= TQ_VEC_SMEM;
}

template <ggml_type type, bool glu>
static void ggml_cuda_mul_mat_tq_vec_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * const * mm, const int n,
        const tq_vec_args & a) {
    const ggml_tensor * src0 = mm[0]->src[0];
    const ggml_tensor * src1 = mm[0]->src[1];
    const ggml_tensor * dst  = mm[0];
    const int nb   = src0->ne[0]/QK_K;
    const int rows = glu ? a.row_end[0] : a.row_end[n - 1];
    // the thread blocks take turns on the row groups, so that fewer of them quantize src1
    const int  rows_f = TQ_VEC_THREADS/TQ_LANES;
    const int  nsm    = ggml_cuda_info().devices[ctx.device].nsm;
    const dim3 grid(std::min((rows + rows_f - 1)/rows_f, TQ_VEC_BPSM*nsm), 1, src1->ne[2]*src1->ne[3]);
    mul_mat_tq_vec<type, glu><<<grid, TQ_VEC_THREADS, nb*sizeof(block_q8_tq), ctx.stream()>>>(a,
        (const float *) src1->data, nb, src0->nb[2], src0->nb[3], src1->ne[2], src1->ne[2]/src0->ne[2], src1->ne[3]/src0->ne[3],
        src1->nb[2]/sizeof(float), src1->nb[3]/sizeof(float), dst->nb[2]/sizeof(float), dst->nb[3]/sizeof(float));
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_mul_mat_tq_vec(ggml_backend_cuda_context & ctx, const ggml_tensor * const * mm, const int n, ggml_tensor * glu,
        ggml_tensor * add) {
    GGML_ASSERT(n >= 1 && n <= TQ_VEC_MAX && (glu == nullptr || n == 2) && (add == nullptr || (n == 1 && glu == nullptr)));
    tq_vec_args a = {};
    a.n = n;
    if (add != nullptr) {
        const ggml_tensor * res = add->src[0] == mm[0] ? add->src[1] : add->src[0];
        GGML_ASSERT(ggml_is_vector(mm[0]) && ggml_are_same_shape(add, mm[0]) && ggml_are_same_shape(res, add) &&
                    ggml_is_contiguous(add) && ggml_is_contiguous(res) && res->type == GGML_TYPE_F32 && add->type == GGML_TYPE_F32);
        a.add = (const float *) res->data;
    }
    for (int k = 0; k < n; ++k) {
        GGML_ASSERT(ggml_cuda_mul_mat_tq_vec_ok(mm[k]) && mm[k]->src[0]->type == mm[0]->src[0]->type &&
                    mm[k]->src[1] == mm[0]->src[1] && (n == 1 || (ggml_is_matrix(mm[k]->src[0]) && ggml_is_vector(mm[k]))));
        a.x[k]       = (const char *) mm[k]->src[0]->data;
        a.dst[k]     = (float *) (glu ? glu : add ? add : mm[k])->data;
        a.s01[k]     = mm[k]->src[0]->nb[1];
        a.row_end[k] = (k == 0 ? 0 : a.row_end[k - 1]) + (int) mm[k]->src[0]->ne[1];
    }
    ctx.tq_census[3]++;
    ctx.tq_census[4] += n - 1 + (add != nullptr);
    const bool tq1 = mm[0]->src[0]->type == GGML_TYPE_TQ1_0;
    if (glu != nullptr) {
        GGML_ASSERT(mm[0]->src[0]->ne[1] == mm[1]->src[0]->ne[1] && ggml_is_contiguous(glu) && glu->ne[0] == mm[0]->ne[0]);
        tq1 ? ggml_cuda_mul_mat_tq_vec_launch<GGML_TYPE_TQ1_0, true>(ctx, mm, n, a)
            : ggml_cuda_mul_mat_tq_vec_launch<GGML_TYPE_TQ2_0, true>(ctx, mm, n, a);
    } else {
        tq1 ? ggml_cuda_mul_mat_tq_vec_launch<GGML_TYPE_TQ1_0, false>(ctx, mm, n, a)
            : ggml_cuda_mul_mat_tq_vec_launch<GGML_TYPE_TQ2_0, false>(ctx, mm, n, a);
    }
}

void ggml_cuda_mul_mat_tq(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS;
    GGML_ASSERT(ggml_cuda_mul_mat_tq_supported(src0, src1, dst));

    cudaStream_t stream = ctx.stream();

    const int     nb      = ne00 / QK_K;
    const int64_t nrows_y = ne11*ne12*ne13;

    // one src1 column (decode): the vec kernel quantizes it itself, into shared memory
    if (ggml_cuda_mul_mat_tq_vec_ok(dst)) {
        const ggml_tensor * mm = dst;
        ggml_cuda_mul_mat_tq_vec(ctx, &mm, 1, nullptr, nullptr);
        return;
    }

    ggml_cuda_pool_alloc<block_q8_tq> y_q8(ctx.pool(), nrows_y*nb);
    {
        const int  wpb = 4; // Q8_K blocks (warps) per thread block
        const dim3 grid((nb + wpb - 1)/wpb, ne11, ne12*ne13);
        quantize_q8_tq<<<grid, dim3(WARP_SIZE, wpb, 1), 0, stream>>>((const float *) src1->data, y_q8.get(), nb,
            nb11/sizeof(float), nb12/sizeof(float), nb13/sizeof(float), ne11, ne12);
        CUDA_CHECK(cudaGetLastError());
    }

    if (src0->type == GGML_TYPE_TQ1_0) {
        ggml_cuda_mul_mat_tq_launch<GGML_TYPE_TQ1_0>(ctx, src0, src1, dst, y_q8.get(), nb);
    } else {
        ggml_cuda_mul_mat_tq_launch<GGML_TYPE_TQ2_0>(ctx, src0, src1, dst, y_q8.get(), nb);
    }
}
