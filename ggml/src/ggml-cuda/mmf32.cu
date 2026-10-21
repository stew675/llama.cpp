#if defined(GGML_USE_HIP)

// mmf32 -- the F32 "skinny" prefill GEMM (block 06).  See mmf32.cuh for why it exists and its scope.

#include "mmf32.cuh"

#include <atomic>

namespace {

// BM x BN output tile per block, k blocked by BK, and the k range additionally split across
// blockIdx.z.  A thread owns a TMR (rows) x TNC (columns) sub-tile; threadIdx.x walks the rows so
// that a warp's stores are contiguous.  Each chunk accumulates its own contiguous k range
// sequentially and the chunks are summed in increasing z, so the result is a fixed reduction tree --
// deterministic across runs by construction, and no atomics.
template<int BM, int BN, int BK, int TMR, int TNC>
static __global__ __launch_bounds__((BM/TMR)*(BN/TNC)) void mul_mat_f32_skinny_kernel(
        const float * __restrict__ x, const float * __restrict__ y, float * __restrict__ tmp,
        const int m, const int n, const int k, const int ldc, const int kchunk) {
    constexpr int GX = BM/TMR;     // thread row-groups
    constexpr int GY = BN/TNC;     // thread column-groups
    constexpr int THREADS = GX*GY;
    constexpr int PAD = 4;         // keeps both tile strides 16-byte friendly and off the wave stride

    __shared__ float xs[BK][BM + PAD];
    __shared__ float ys[BK][BN + PAD];

    const int txc = threadIdx.x;   // row group
    const int tyc = threadIdx.y;   // column group
    const int tid = tyc*GX + txc;
    const int n0  = blockIdx.y*BN;
    const int m0  = blockIdx.x*BM;
    const int k0b = blockIdx.z*kchunk;
    const int k1b = k0b + kchunk < k ? k0b + kchunk : k;

    float acc[TMR][TNC];
    #pragma unroll
    for (int i = 0; i < TMR; ++i) {
        #pragma unroll
        for (int j = 0; j < TNC; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (int k0 = k0b; k0 < k1b; k0 += BK) {
        // weight tile: rows m0.., k columns k0..  (global reads coalesced along k)
        for (int idx = tid; idx < BM*BK; idx += THREADS) {
            const int row = idx/BK;
            const int kk  = idx - row*BK;
            const int gm  = m0 + row;
            xs[kk][row] = (gm < m && k0 + kk < k1b) ? x[(size_t) gm*k + (k0 + kk)] : 0.0f;
        }
        // activation tile: rows k0.., columns n0..  (ggml src1 is ne={k,n} with nb0=4, i.e. n rows
        // of k contiguous elements -- so the coalesced read walks k, and the store transposes)
        for (int idx = tid; idx < BK*BN; idx += THREADS) {
            const int col = idx/BK;
            const int kk  = idx - col*BK;
            const int gn  = n0 + col;
            ys[kk][col] = (gn < n && k0 + kk < k1b) ? y[(size_t) gn*k + (k0 + kk)] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float a[TMR];
            float b[TNC];
            const float4 a4 = *reinterpret_cast<const float4 *>(&xs[kk][txc*TMR]);
            a[0] = a4.x; a[1] = a4.y; a[2] = a4.z; a[3] = a4.w;
            const float4 b0 = *reinterpret_cast<const float4 *>(&ys[kk][tyc*TNC]);
            const float4 b1 = *reinterpret_cast<const float4 *>(&ys[kk][tyc*TNC + 4]);
            b[0] = b0.x; b[1] = b0.y; b[2] = b0.z; b[3] = b0.w;
            b[4] = b1.x; b[5] = b1.y; b[6] = b1.z; b[7] = b1.w;

            #pragma unroll
            for (int i = 0; i < TMR; ++i) {
                #pragma unroll
                for (int j = 0; j < TNC; ++j) {
                    acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }

    // Store: consecutive `gm` is consecutive memory, so the TMR values of one column are a
    // contiguous run (float4 when the ldc/m alignment allows).
    float * out = tmp + (size_t) blockIdx.z*n*ldc;
    #pragma unroll
    for (int j = 0; j < TNC; ++j) {
        const int gn = n0 + tyc*TNC + j;
        if (gn >= n) {
            continue;
        }
        float * o = out + (size_t) gn*ldc + m0 + txc*TMR;
        if (m0 + txc*TMR + TMR <= m && (ldc % 4) == 0) {
            *reinterpret_cast<float4 *>(o) = make_float4(acc[0][j], acc[1][j], acc[2][j], acc[3][j]);
            continue;
        }
        #pragma unroll
        for (int i = 0; i < TMR; ++i) {
            if (m0 + txc*TMR + i < m) {
                o[i] = acc[i][j];
            }
        }
    }
}

// Sum the k chunks in increasing z: a fixed order, so the whole GEMM is a fixed reduction tree.
static __global__ void mmf32_reduce_kernel(const float * __restrict__ tmp, float * __restrict__ dst,
                                           const int total, const int per_chunk, const int ks) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= total) {
        return;
    }
    float s = tmp[i];
    for (int z = 1; z < ks; ++z) {
        s += tmp[(size_t) z*per_chunk + i];
    }
    dst[i] = s;
}


template<int BM, int BN, int BK, int TMR, int TNC>
static void mmf32_launch(const float * x, const float * y, float * dst, float * tmp,
                         int m, int n, int k, int ldc, int ks, cudaStream_t stream) {
    const dim3 block(BM/TMR, BN/TNC, 1);
    const dim3 grid((m + BM - 1)/BM, (n + BN - 1)/BN, ks);
    const int kchunk = (k + ks - 1)/ks;
    mul_mat_f32_skinny_kernel<BM, BN, BK, TMR, TNC><<<grid, block, 0, stream>>>(x, y, tmp, m, n, k, ldc, kchunk);
    if (ks == 1) {
        return;
    }
    // One chunk's output is n columns of ldc rows -- NOT ldc*max(m,n): when n < m the chunk stride must
    // follow the kernel's (blockIdx.z*n*ldc), or the reduce reads the wrong chunk.
    const int per_chunk = n*ldc;
    mmf32_reduce_kernel<<<(per_chunk + 255)/256, 256, 0, stream>>>(tmp, dst, per_chunk, per_chunk, ks);
}

} // namespace

bool ggml_cuda_mmf32_enabled(int cc) {
    // Default-on policy: the variable exists to DISABLE.  RDNA4 (gfx1201) and RDNA3_5 (gfx1151) are the
    // validated devices -- same four legs on both (exact, deterministic by construction, a prefill win, and a
    // --family-switch PASS at a confident point; GREEDY-PURITY.md §47).  On RDNA3 the kernel is NOT a win
    // (gfx1100: -1.3..-2.2 % at pp2048, one visit a tie), so it stays opt-in there until that box has a pinned
    // per-arch tile; the lever would be the tile (`BN`/`ks`), not this gate.
    static const int env = []() {
        const char * e = getenv("GGML_CUDA_MMF32");
        if (e == nullptr) {
            return -1;                      // unset: arch default
        }
        return atoi(e) != 0 ? 1 : 0;
    }();
    if (env >= 0) {
        return env == 1;
    }
    return GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc);
}

bool ggml_cuda_mmf32_supported(int cc, int64_t m, int64_t n, int64_t k, int64_t ldc) {
    // The hipBLASLt surface this kernel covers: F32 weights x a prefill-width batch with a small
    // output-row count.  Everything else keeps the rocBLAS path.
    return ggml_cuda_mmf32_enabled(cc) && m >= 1 && m <= 256 && n >= 17 && n <= 65536 &&
           k >= 256 && k <= 8192 && ldc == m;
}

void ggml_cuda_mul_mat_f32_skinny(ggml_backend_cuda_context & ctx, int64_t m_, int64_t n_, int64_t k_,
                                  int64_t ldc_, const float * x, const float * y, float * dst) {
    const int m = (int) m_, n = (int) n_, k = (int) k_, ldc = (int) ldc_;
    // Tiles and the k split are the campaign's measured picks (wip/hipblaslt-bypass, archived after r104):
    // 32 output rows per block, a 4x8 thread sub-tile, BN=256 for the skinny shapes (the GDN
    // ssm_alpha/ssm_beta GEMMs) and 64 for the wider ones (the MoE routers), and enough k chunks to fill the
    // card.  Measured on gfx1201: 32x512x2048 -> 0.032 ms (2.07 TFLOP/s) vs the best hipBLASLt class
    // 0.046 ms; the un-split kernel was 0.27 ms, i.e. the split is the whole win.
    const int bn  = m <= 64 ? 256 : 64;
    const int blocks_base = ((m + 31)/32) * ((n + bn - 1)/bn);
    int ks = 256/blocks_base;
    ks = ks < 1 ? 1 : (ks > 16 ? 16 : ks);
    if (k/ks < 32) {                      // never let a chunk be shorter than one BK block
        ks = k/32 > 0 ? k/32 : 1;
    }
    // Diagnostics: the first few resolved plans on stderr, gated on the env var (the same convention as
    // GGML_CUDA_MMB_LOG).  It reports what was actually used; it never changes what is computed.
    static const int log_n = getenv("GGML_CUDA_MMF32_LOG") ? atoi(getenv("GGML_CUDA_MMF32_LOG")) : 0;
    if (log_n != 0) {
        static std::atomic<int> printed{0};
        if (printed.fetch_add(1) < 16) {
            fprintf(stderr, "MMF32 m=%d n=%d k=%d ldc=%d tile=32x%d/4x8 ks=%d\n", m, n, k, ldc, bn, ks);
        }
    }
    ggml_cuda_pool_alloc<float> tmp(ctx.pool(), ks == 1 ? 1 : (size_t) ks*n*ldc);
    float * tmp_ptr = ks == 1 ? dst : tmp.get();
    if (m <= 64) {
        mmf32_launch<32, 256, 32, 4, 8>(x, y, dst, tmp_ptr, m, n, k, ldc, ks, ctx.stream());
    } else {
        mmf32_launch<32,  64, 32, 4, 8>(x, y, dst, tmp_ptr, m, n, k, ldc, ks, ctx.stream());
    }
    CUDA_CHECK(cudaGetLastError());
}

#endif // defined(GGML_USE_HIP)
