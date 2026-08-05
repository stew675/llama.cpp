// FP8 E4M3 mul_mat kernels. Native FP8 hardware only (RDNA4 WMMA or CUDA sm_89+).
//
// Weight layout is the self-contained block_f8_e4m3: [f32 scale][128 fp8].
// Activations are pre-quantized by quantize_fp8 into a transposed staging layout
// ([k][n] fp8, k-major/token-minor, plus per-token-block scales), so the hot
// mul_mat kernel reads contiguous fp8 fragments and never re-quantizes.
// Mirrors the quantize_mmq_q8_1_cuda + mul_mat_q pattern in this codebase.

#pragma once

#include "common.cuh"

#include <cstdint>

#define GGML_FP8_TILE_M 16
#define GGML_FP8_TILE_N 16
#define GGML_FP8_TILE_K 128
#define GGML_FP8_NWARPS 4
#define GGML_FP8_NTHREADS (32 * GGML_FP8_NWARPS) // 128
#define GGML_FP8_GEMV_MAX_N 16 // use the dot4 GEMV below this token count

// fp8 e4m3fn (OCP) decode: value = (-1)^s * 2^(e-7) * (1.m), max 448, NaN = 0x7F/0xFF
__device__ __forceinline__ float fp8_e4m3_to_f32(uint8_t x) {
    const uint32_t sign = ((uint32_t)(x & 0x80)) << 24;
    const uint32_t exp  = (x >> 3) & 0x0F;
    const uint32_t man  = x & 0x07;

    uint32_t bits;
    if (exp == 0) {
        // subnormal or zero: value = man * 2^-9
        if (man == 0) {
            bits = sign;
        } else {
            const uint32_t k = man >= 4 ? 2 : man >= 2 ? 1 : 0;
            bits = sign | ((k + 118) << 23) | ((man - (1u << k)) << (23 - k));
        }
    } else if (exp == 15 && man == 7) {
        bits = 0x7FC00000u; // NaN
    } else {
        bits = sign | ((exp + 120) << 23) | (man << 20);
    }

    return __uint_as_float(bits);
}

// fp8 e4m3fn encode, round-to-nearest-even, saturating to +/-448
__device__ __forceinline__ uint8_t fp8_e4m3_from_f32(float x) {
    const uint32_t bits = __float_as_uint(x);
    const uint32_t sign = (bits >> 31) & 1;
    const uint32_t exp  = (bits >> 23) & 0xFF;
    const uint32_t man  = bits & 0x7FFFFF;

    if (exp == 0xFF) {
        return sign ? 0xFF : 0x7F; // NaN/inf input
    }
    if (exp == 0) {
        return sign ? 0x80 : 0x00; // f32 subnormal is far below fp8 min
    }

    int E = (int) exp - 120;
    if (E >= 16) {
        return sign ? 0xFE : 0x7E; // well beyond e4m3 range
    }
    if (E <= 0) {
        // subnormal range: M = rne(value * 512), 0..7, 8 carries to E=1
        const int shift = 141 - (int) exp;
        uint32_t M = 0;
        if (shift < 31) {
            const uint64_t val = (1ull << 23) | man;
            M = (uint32_t)(val >> shift);
            const uint64_t frac = val & ((1ull << shift) - 1);
            const uint64_t half = 1ull << (shift - 1);
            if (frac > half || (frac == half && (M & 1))) {
                M++;
            }
        }
        if (M >= 8) {
            return (uint8_t)((sign << 7) | 0x08); // smallest normal (E=1, M=0)
        }
        return (uint8_t)((sign << 7) | M);
    }

    // normal: E in 1..15, round mantissa to 3 bits (RNE)
    uint32_t man_3 = man >> 20;
    const uint32_t frac = man & 0xFFFFF;
    if (frac > 0x80000 || (frac == 0x80000 && (man_3 & 1))) {
        man_3++;
    }
    if (man_3 == 8) {
        man_3 = 0;
        E++;
    }
    if (E == 15 && man_3 == 7) {
        man_3 = 6; // 480 is NaN in e4m3 -> clamp to 448
    }
    if (E >= 16) {
        return sign ? 0xFE : 0x7E; // saturate to 448
    }
    return (uint8_t)((sign << 7) | (E << 3) | man_3);
}

// ---- activation quantization: F32 [k, n] -> fp8 staging [n][k] + scales [n][k/128] ----
// one CTA of 128 threads per (token, 128-col block); block reduce for amax
// staging is token-major/k-minor so the wmma B fragment (8 k-values of one token)
// is one contiguous 8-byte load per lane
__global__ void quantize_fp8(
        const float * __restrict__ x, uint8_t * __restrict__ y_q, float * __restrict__ y_s,
        const int64_t k, const int64_t n, const int64_t n_col_blocks, const int64_t n_pad) {
    __shared__ float red[4];

    const int col0 = blockIdx.x * GGML_FP8_TILE_K;
    const int token = blockIdx.y;
    const int tid = threadIdx.x;

    const float v = x[(int64_t) token * k + col0 + tid];

    // block reduce: max abs over the 128 values
    float m = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o, 32));
    }
    if (tid % 32 == 0) {
        red[tid / 32] = m;
    }
    __syncthreads();
    if (tid == 0) {
        float amax = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            amax = fmaxf(amax, red[i]);
        }
        red[0] = amax / 448.0f; // scale
        red[1] = amax != 0.0f ? 448.0f / amax : 0.0f; // inverse
    }
    __syncthreads();

    y_s[(int64_t) token * n_col_blocks + blockIdx.x] = red[0];
    // staging: [n][k], token-major/k-minor
    y_q[(int64_t) token * k + col0 + tid] = fp8_e4m3_from_f32(v * red[1]);
}

// C fragment layout (wmma f32 16x16x16 fp8, wave32, empirically verified on gfx1201):
//   lane l: token column l%16, rows (l/16)*8 + slot
//   A fragment: lane l byte e = A[l%16][(l/16)*8+e]
//   B fragment: lane l byte e = B[(l/16)*8+e][l%16]
__device__ __forceinline__ int fp8_wmma_row(int l, int s) {
    return (l / 16) * 8 + s;
}
__device__ __forceinline__ int fp8_wmma_col(int l, int s) {
    return l % 16;
}

// Wave32 wmma fragments (gfx12 fp8, verified on gfx1201):
//   A: lane l byte e = A[l%16][(l/16)*8+e]  -> weight row (m0 + l%16), 8 contiguous qs bytes
//   B: lane l byte e = B[(l/16)*8+e][l%16]  -> staging[(n0 + l%16)][...], 8 contiguous bytes
//   C: slot s = C[(l/16)*8+s][l%16]
//
// Each warp computes a 2x2 grid of 16x16 wmma tiles (32 weight rows x 32 tokens):
// per k-step the same A/B fragments feed 4 wmma instructions, reducing the shared
// memory read traffic per wmma. 8 warps per CTA -> 64 rows x 128 tokens, which also
// halves the redundant re-staging of the activation block across the m dimension
// (the staging bandwidth was the bottleneck after the smem port fix).
// Grid: (ceil(n/128), ceil(m/64)).
//
// The 32 weight rows (scales and fp8 bytes in separate arrays so the per-lane
// fragment loads are 8-byte aligned) and the 128-token x 128-byte activation block
// are staged to shared memory with coalesced loads once per k-block.
__launch_bounds__(GGML_FP8_NTHREADS, 1)
__global__ void mul_mat_fp8_wmma(
        const char * __restrict__ src0, const uint8_t * __restrict__ src1_q, const float * __restrict__ src1_s, float * __restrict__ dst,
        const int64_t k, const int64_t m, const int64_t n, const int64_t n_col_blocks, const int64_t n_pad) {
#if defined(GGML_USE_HIP) && defined(RDNA4)
    using fp8x8_t   = __attribute__((ext_vector_type(2))) int;
    using floatx8_t = __attribute__((ext_vector_type(8))) float;

    constexpr int TILE_M = 16;   // wmma tile rows
    constexpr int TILE_N = 16;   // wmma tile cols
    constexpr int MT = 2;        // m-tiles per warp
    constexpr int NT = 2;        // n-tiles per warp
    constexpr int MH = 1;        // m-halves per CTA (warp groups)
    constexpr int CTA_M = TILE_M * MT * MH;                   // 32 rows per CTA
    constexpr int CTA_N = GGML_FP8_NWARPS * TILE_N * NT / MH; // 128 tokens per CTA

    // sA: CTA_M weight rows x 132 B (f32 scale + 128 fp8); sB: CTA_N tokens x 136 B
    // (128 + pad so the per-lane 8-byte fragment reads hit distinct banks)
    __shared__ uint8_t sA[CTA_M][132];
    __shared__ uint8_t sB[CTA_N][GGML_FP8_TILE_K + 8];

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int tid  = threadIdx.x;
    const int wm = warp / (GGML_FP8_NWARPS / MH); // which 32-row half of the CTA
    const int wn = warp % (GGML_FP8_NWARPS / MH); // which 32-token quad of the CTA
    const int t0 = wn * (TILE_N * NT); // this warp's first token within the CTA
    const int rm = wm * (TILE_M * MT); // this warp's first row within the CTA

    const int m0 = blockIdx.y * CTA_M;
    const int n0 = blockIdx.x * CTA_N;

    const int row_lane = lane % 16; // weight row of this lane within a tile (also token column)
    const int k_half   = (lane / 16) * 8; // k offset of this lane's half of the 16-k step

    const int64_t row_stride = n_col_blocks * (int64_t) sizeof(block_f8_e4m3);

    floatx8_t acc00 = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
    floatx8_t acc01 = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
    floatx8_t acc10 = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
    floatx8_t acc11 = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };

    for (int64_t cb = 0; cb < n_col_blocks; ++cb) {
        const char * Abase = src0 + (int64_t) m0 * row_stride + cb * sizeof(block_f8_e4m3);
        const uint8_t * Bbase = src1_q + (int64_t) n0 * k + cb * GGML_FP8_TILE_K;

        // stage the weight tile: d (4 B) + 128 fp8 per row, coalesced 4-B loads
        if (tid < CTA_M) {
            *(float *) &sA[tid][0] = (m0 + tid < m) ? *(const float *) (Abase + (int64_t) tid * row_stride) : 0.f;
        }
        // CTA_M x 32 uint = 1024 / 128 threads = 8 each
#pragma unroll
        for (int it = 0; it < 8; ++it) {
            const int idx = tid + it * GGML_FP8_NTHREADS; // 0..1023
            const int r = idx >> 5;
            const int c = idx & 31;
            *(uint *) &sA[r][4 + c * 4] = (m0 + r < m) ? *(const uint *) (Abase + (int64_t) r * row_stride + 4 + c * 4) : 0u;
        }
        // stage the activation block: CTA_N tokens x 128 fp8, coalesced 16-B loads
        // CTA_N x 8 uint4 = 1024 / 128 threads = 8 each
#pragma unroll
        for (int it = 0; it < 8; ++it) {
            const int idx = tid + it * GGML_FP8_NTHREADS; // 0..1023
            const int t = idx >> 3;
            const int c = idx & 7;
            *(uint4 *) &sB[t][c * 16] = *(const uint4 *) (Bbase + (int64_t) t * k + c * 16);
        }
        __syncthreads();

        // 8 wmma k-steps per tile per 128-col block; fragments are shared by the
        // 2x2 tile grid so each smem byte feeds up to 4 wmma instructions
        floatx8_t t00 = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
        floatx8_t t01 = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
        floatx8_t t10 = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
        floatx8_t t11 = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
        // software-pipeline the fragment loads ahead of the wmma chain
        fp8x8_t a0 = *reinterpret_cast<const fp8x8_t *>(&sA[rm + row_lane][4 + k_half]);
        fp8x8_t a1 = *reinterpret_cast<const fp8x8_t *>(&sA[rm + TILE_M + row_lane][4 + k_half]);
        fp8x8_t b0 = *reinterpret_cast<const fp8x8_t *>(&sB[t0 + row_lane][k_half]);
        fp8x8_t b1 = *reinterpret_cast<const fp8x8_t *>(&sB[t0 + TILE_N + row_lane][k_half]);
#pragma unroll
        for (int kk = 0; kk < GGML_FP8_TILE_K / 16; ++kk) {
            const int kk_next = (kk + 1) * 16;
            const fp8x8_t a0_n = kk < 7 ? *reinterpret_cast<const fp8x8_t *>(&sA[rm + row_lane][4 + kk_next + k_half]) : a0;
            const fp8x8_t a1_n = kk < 7 ? *reinterpret_cast<const fp8x8_t *>(&sA[rm + TILE_M + row_lane][4 + kk_next + k_half]) : a1;
            const fp8x8_t b0_n = kk < 7 ? *reinterpret_cast<const fp8x8_t *>(&sB[t0 + row_lane][kk_next + k_half]) : b0;
            const fp8x8_t b1_n = kk < 7 ? *reinterpret_cast<const fp8x8_t *>(&sB[t0 + TILE_N + row_lane][kk_next + k_half]) : b1;
            t00 = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a0, b0, t00);
            t01 = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a0, b1, t01);
            t10 = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a1, b0, t10);
            t11 = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a1, b1, t11);
            a0 = a0_n; a1 = a1_n; b0 = b0_n; b1 = b1_n;
        }

        // apply the per-128x128 weight scale and per-128-block activation scale
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            const int r = fp8_wmma_row(lane, s);
            const int c = fp8_wmma_col(lane, s);
            const float wd0 = *(float *) &sA[rm + r][0];
            const float wd1 = *(float *) &sA[rm + TILE_M + r][0];
            const float ad0 = src1_s[(int64_t) (n0 + t0 + c) * n_col_blocks + cb];
            const float ad1 = src1_s[(int64_t) (n0 + t0 + TILE_N + c) * n_col_blocks + cb];
            acc00[s] += wd0 * ad0 * t00[s];
            acc01[s] += wd0 * ad1 * t01[s];
            acc10[s] += wd1 * ad0 * t10[s];
            acc11[s] += wd1 * ad1 * t11[s];
        }

        __syncthreads(); // protect sA/sB before the next block's stage-in
    }

#pragma unroll
    for (int s = 0; s < 8; ++s) {
        const int r = fp8_wmma_row(lane, s);
        const int c = fp8_wmma_col(lane, s);
        const int mr = m0 + rm + r;
        const int nc = n0 + t0 + c;
        if (mr < m && nc < n) {
            dst[(int64_t) nc * m + mr] = acc00[s];
        }
        if (mr < m && nc + TILE_N < n) {
            dst[(int64_t) (nc + TILE_N) * m + mr] = acc01[s];
        }
        if (mr + TILE_M < m && nc < n) {
            dst[(int64_t) nc * m + mr + TILE_M] = acc10[s];
        }
        if (mr + TILE_M < m && nc + TILE_N < n) {
            dst[(int64_t) (nc + TILE_N) * m + mr + TILE_M] = acc11[s];
        }
    }
#else
    // unreachable: the host pass only generates the launch stub
    (void) src0; (void) src1_q; (void) src1_s; (void) dst;
    (void) k; (void) m; (void) n; (void) n_col_blocks; (void) n_pad;
#endif
}

// dot4-based GEMV for small token batches (n <= GGML_FP8_GEMV_MAX_N), used during
// generation. One warp per output row; the 32 lanes read the 128 fp8 bytes of a
// weight block coalesced (4 bytes each) and dot4 them with the shared activation
// block. This avoids the wmma tile waste (16-token tiles) at batch 1.
__global__ void mul_mat_fp8_gemv(
        const char * __restrict__ src0, const uint8_t * __restrict__ src1_q, const float * __restrict__ src1_s, float * __restrict__ dst,
        const int64_t k, const int64_t m, const int64_t n, const int64_t n_col_blocks, const int64_t n_pad) {
#if defined(GGML_USE_HIP) && defined(RDNA4)
    (void) n_pad;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    const int64_t mi = blockIdx.x * 4 + warp;
    if (mi >= m) {
        return;
    }
    const int64_t ni = blockIdx.y;

    float acc = 0.0f;
    const int64_t row_stride = n_col_blocks * (int64_t) sizeof(block_f8_e4m3);
    const uint8_t * x = src1_q + ni * k;

    for (int64_t cb = 0; cb < n_col_blocks; ++cb) {
        const block_f8_e4m3 * wblk = (const block_f8_e4m3 *) (src0 + mi * row_stride + cb * sizeof(block_f8_e4m3));
        const uint32_t w4 = *reinterpret_cast<const uint32_t *>(wblk->qs + lane * 4);
        const uint32_t x4 = *reinterpret_cast<const uint32_t *>(x + cb * GGML_FP8_TILE_K + lane * 4);

        float dot = __builtin_amdgcn_dot4_f32_fp8_fp8(w4, x4, 0.0f);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            dot += __shfl_xor_sync(0xffffffff, dot, o, 32);
        }
        acc += wblk->d * src1_s[ni * n_col_blocks + cb] * dot;
    }

    dst[ni * m + mi] = acc;
#else
    (void) src0; (void) src1_q; (void) src1_s; (void) dst;
    (void) k; (void) m; (void) n; (void) n_col_blocks; (void) n_pad;
#endif
}

// Scalar fallback (CUDA, and any device without the RDNA4 WMMA path).
// One thread per output element, consumes the pre-quantized staging.
__global__ void mul_mat_fp8_scalar(
        const char * __restrict__ src0, const uint8_t * __restrict__ src1_q, const float * __restrict__ src1_s, float * __restrict__ dst,
        const int64_t k, const int64_t m, const int64_t n, const int64_t n_col_blocks, const int64_t n_pad) {
    const int64_t mi = blockIdx.y * blockDim.y + threadIdx.y;
    const int64_t ni = blockIdx.x * blockDim.x + threadIdx.x;
    if (mi >= m || ni >= n) {
        return;
    }

    float acc = 0.0f;
    for (int64_t cb = 0; cb < n_col_blocks; ++cb) {
        const block_f8_e4m3 * wblk = (const block_f8_e4m3 *) (src0 + (mi * n_col_blocks + cb) * sizeof(block_f8_e4m3));
        const float ad = src1_s[ni * n_col_blocks + cb];
        float dot = 0.0f;
        for (int c = 0; c < GGML_FP8_TILE_K; ++c) {
            dot += fp8_e4m3_to_f32(wblk->qs[c]) * fp8_e4m3_to_f32(src1_q[ni * k + cb * GGML_FP8_TILE_K + c]);
        }
        acc += wblk->d * ad * dot;
    }
    dst[ni * m + mi] = acc;
}
