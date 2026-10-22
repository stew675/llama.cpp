#include "common.cuh"
#include "cp-async.cuh"
#include "mma.cuh"
#include "fattn-common.cuh"

using namespace ggml_cuda_mma;

// Config options for the MMA kernel.
// Should not affect results, only speed/register pressure/shared memory use.
struct fattn_mma_config {
    int  nthreads;       // Number of threads per CUDA block.
    int  occupancy;      // Targeted occupancy for the MMA kernel.
    int  nbatch_fa;      // Number of KV rows per softmax rescaling of KQ rowsums and VKQ accumulators.
    int  nbatch_K2;      // Number of K half2 values in direction of DKQ to load in parallel.
    int  nbatch_V2;      // Number of V half2 values in direction of DV to load in parallel.
    int  nbatch_combine; // Number of VKQ half2 values in direction of DV to combine in parallel.
    int  nstages_target; // Number of pipeline stages to use ideally, 1 == always load data synchronously, 2 == preload data if there is hardware support.
    bool Q_in_reg;       // Whether the Q values should be kept permanently in registers.

    constexpr __host__ __device__ fattn_mma_config(
            int nthreads, int occupancy, int nbatch_fa, int nbatch_K2, int nbatch_V2, int nbatch_combine, int nstages_target, bool Q_in_reg) :
        nthreads(nthreads), occupancy(occupancy), nbatch_fa(nbatch_fa), nbatch_K2(nbatch_K2), nbatch_V2(nbatch_V2), nbatch_combine(nbatch_combine),
        nstages_target(nstages_target), Q_in_reg(Q_in_reg) {}
};

#define GGML_CUDA_FATTN_MMA_CONFIG_CASE(DKQ_, DV_, ncols_, nthreads_, occupancy_, nbatch_fa_, nbatch_K2_, nbatch_V2_, nbatch_combine_, nstages_target_, Q_in_reg_) \
    if (DKQ == (DKQ_) && DV == (DV_) && ncols == (ncols_)) {                                                                                                       \
        static_assert((nthreads_)       % 32 == 0 && (nthreads_)       <= 512, "bad nthreads");                                                                    \
        static_assert(                               (occupancy_)      <=   8, "bad occupancy");                                                                   \
        static_assert((nbatch_fa_)      % 32 == 0 && (nbatch_fa_)      <= 256, "bad nbatch_fa");                                                                   \
        static_assert((nbatch_K2_)      %  4 == 0 && (nbatch_K2_)      <= 512, "bad nbatch_K2");                                                                   \
        static_assert((nbatch_V2_)      %  4 == 0 && (nbatch_V2_)      <= 256, "bad nbatch_V2");                                                                   \
        static_assert((nbatch_combine_) %  4 == 0 && (nbatch_combine_) <= 128, "bad nbatch_combine");                                                              \
        static_assert((nstages_target_)      >= 1 && (nstages_target_) <=   2, "bad nstages_target");                                                              \
        return fattn_mma_config{(nthreads_), (occupancy_), (nbatch_fa_), (nbatch_K2_), (nbatch_V2_), (nbatch_combine_), (nstages_target_), (Q_in_reg_)};           \
    }                                                                                                                                                              \

static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_ampere(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64,  8, 128, 2, 128,  32,  32,  32, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 16, 128, 2,  64,  32,  32,  32, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 32, 128, 2,  64,  32,  32,  32, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 64, 128, 2,  64,  32,  32,  32, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80,  8, 128, 2, 128,  40,  40,  40, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 16, 128, 2,  64,  40,  40,  40, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 32, 128, 2,  64,  40,  40,  40, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 64, 128, 2,  64,  40,  40,  40, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96,  8, 128, 2, 128,  48,  48,  48, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 16, 128, 2,  64,  48,  48,  48, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 32, 128, 2,  64,  48,  48,  48, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 64, 128, 2,  64,  48,  48,  48, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112,  8, 128, 2, 128,  56,  56,  56, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 16, 128, 2,  64,  56,  56,  56, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 32, 128, 2,  64,  56,  56,  56, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 64, 128, 2,  64,  56,  56,  56, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128,  8, 128, 2, 128,  64,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 16, 128, 2,  64,  64,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 32, 128, 2,  64,  64,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 64, 128, 2,  64,  64,  64,  64, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128,  8,  64, 4,  64,  96,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 16,  64, 4,  32,  96,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 32, 128, 2,  32,  96,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 64, 128, 2,  32,  96,  64,  64, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256,  8, 128, 2,  64, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 16, 256, 1,  64, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 32, 128, 2,  32, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 128, 2,  32, 128, 128, 128, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 32, 128, 2,  32, 128, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 64, 256, 1,  32, 128, 128, 128, 1, false);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512,  8, 128, 2,  64, 128, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 16, 256, 1,  64, 128, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 32, 256, 1,  32, 128, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 64, 256, 1,  32, 128, 128, 128, 1, false);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512,  8,  64, 4,  32, 288, 256, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 16,  64, 4,  32, 288, 256, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 32, 128, 2,  32, 160, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 256, 1,  32, 160, 128, 128, 1, false);

    return fattn_mma_config(32, 1, 0, 0, 0, 0, 0, false);
}

static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_turing(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256,  8, 128, 2,  64, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 16, 128, 2,  64, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 32, 128, 2,  64, 128, 128,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 128, 2,  64, 128, 128,  64, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 32, 128, 2,  32, 128, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 64, 256, 1,  32, 128, 128, 128, 1, false);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512,  8,  64, 4,  32,  96,  64, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 16,  64, 4,  32,  96,  64, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 32, 128, 2,  32, 128, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 64, 256, 1,  32, 128, 128, 128, 1, false);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512,  8,  64, 4,  32,  96,  64, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 16,  64, 4,  32,  96,  64, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 32, 128, 2,  32, 160, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 256, 1,  32, 160, 128, 128, 1, false);

    return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);
}

static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_volta(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 32, 128, 2,  32, 128, 128,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 64, 256, 1,  32, 128, 128,  64, 1, false);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512,  8,  64, 4,  32, 256, 256,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 16,  64, 4,  32, 256, 256,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 32, 128, 2,  32, 128, 128,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 64, 256, 1,  32, 128, 128,  64, 1, false);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512,  8,  64, 4,  32, 288, 256,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 16,  64, 4,  32, 288, 256,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 32, 128, 2,  32, 160, 128,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 256, 1,  32, 160, 128,  64, 1, false);

    // TODO tune specifically for Volta
    return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);
}

static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_rdna(const int DKQ, const int DV, const int ncols, const bool is_rdna3_5,
        const bool is_rdna4 = false, const int ncols2 = 1, const bool band_wide = false) {
    // RDNA4 GQA decode/verify band (ncols2 == 8, 256-dim heads; see ggml_cuda_fattn_band_wmma_applies).
    // K and V are staged in two 64-half2 batches instead of one 128-half2 batch: with native q8_0 K/V
    // this is 12-14 % faster on gfx1201 (kv 51200..98304, n_q 1..4) and f16 3-4 %.  The K batches are
    // iterated forward for 256-dim heads (see flash_attn_ext_f16_iter), so the result is bit-identical
    // to the single-batch row.
    // 8 warps per block (256 threads) hide more of the synchronous K/V load latency: another ~17 %
    // for q8_0.  The warps split each Q column's KV rows 4 ways instead of 2, which changes the VKQ
    // summation order (not bit-identical to 4 warps) but not its dependence on n_q (W = 1..8 stay pure).
    // This row is reachable ONLY from the band: ncols1 = 32/ncols2, which the dispatch chooses for
    // n_q <= 32/ncols2 (<= 4 with ncols2 = 8); prefill always takes ncols1 = 64/ncols2 (ncols 64).
    if (is_rdna4 && ncols2 == 8 && DKQ == 256 && DV == 256 && ncols == 32) {
        return fattn_mma_config(256, 1, 64, 64, 64, 64, 1, true);
    }
    // The band's 64-column block (query widths 5..8, one KV pass) keeps its OWN geometry, selected by
    // the compile-time band_wide flag, because ncols 64 is also every wide prefill's shape and
    // nthreads/occupancy feed __launch_bounds__, so a row keyed on ncols alone would be compiled into
    // the prefill kernel (a 16-warp band row there measured 26 % slower FA / 10 % slower end to end at
    // pp512 @ d98304).  The row below is the one the band needs: 16 warps so that the warps split each
    // Q column's KV rows 4 ways (np = nwarps*cols_per_warp/ncols = 4), exactly as the 32-column block
    // splits them with its 8 warps -- that per-column reduction split is part of the band's width-purity
    // contract (GREEDY-PURITY band invariant), and the ncols 64 block is reached by every verify step of
    // width 5..8, so a different split there makes greedy decode with a draft differ from decode without
    // one (issue #224).  nbatch_combine 32 keeps the block inside the 64 KiB LDS limit at 16 warps.
    // It is reachable ONLY from the band dispatch (fattn.cu, the band_wide case instantiation): prefill
    // keeps the 256-thread table row below.
    if (band_wide && is_rdna4 && ncols2 == 8 && DKQ == 256 && DV == 256 && ncols == 64) {
        return fattn_mma_config(512, 1, 64, 64, 64, 32, 1, true);
    }
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64,  8, 128, 2,  64,  32,  32,  32, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 16, 128, 2,  64,  32,  32,  32, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 32, 128, 2,  64,  32,  32,  32, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 64, 128, 2,  64,  32,  32,  32, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80,  8,  64, 2,  32,  40,  40,  40, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 16,  64, 2,  32,  40,  40,  40, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 32, 128, 2,  64,  40,  40,  40, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 64, 128, 2,  64,  40,  40,  40, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96,  8,  64, 2,  32,  48,  48,  48, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 16,  64, 2,  32,  48,  48,  48, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 32, 128, 2,  64,  48,  48,  48, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 64, 128, 2,  64,  48,  48,  48, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112,  8,  64, 2,  32,  56,  56,  56, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 16,  64, 2,  32,  56,  56,  56, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 32, 128, 2,  64,  56,  56,  56, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 64, 128, 2,  64,  56,  56,  56, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128,  8,  64, 2,  32,  64,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 16,  64, 2,  32,  64,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 32, 128, 2,  64,  64,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 64, 128, 2,  64,  64,  64,  64, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128,  8,  64, 2,  32,  96,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 16,  64, 2,  32,  96,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 32, 128, 2,  64,  96,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 64, 128, 2,  64,  96,  64,  64, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256,  8,  64, 2,  32, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 16,  64, 2,  32, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 32, 128, 2,  64, 128, 128,  64, 1, true);
    // Strix Halo (gfx1151) tuned row ("B's halo row", locally the `halo` machine): half the KV/V
    // tile with Q_in_reg=false.  Kept for RDNA3_5 where it was tuned, but it is the wrong shape for
    // the discrete cards -- it costs ~1.5x per attention cell on gfx1201 and ~11% at pp150k, so
    // RDNA4/RDNA3_0 take upstream's #28102 config below.
    if (is_rdna3_5 && DKQ == 256 && DV == 256 && ncols == 64) {
        return fattn_mma_config(256, 1, 32, 128, 64, 64, 1, false);
    }
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 256, 2,  64, 128, 128,  64, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 32, 256, 2,  64,  96,  16,  16, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 64, 256, 2,  64,  96,  16,  16, 1, true);

    // RDNA3_0 (gfx1100) head 512: the RDNA4 rows below keep Q in registers and spill 1.2-1.8k VGPRs per lane on
    // gfx1100 (2.2-3.4 KB scratch), which made WMMA ~30 % slower than the tile kernel there.  Q in shared memory,
    // one 128-thread block per SIMD and 64-wide K/V/combine batches spill 46-331 instead; gemma-4-26B-A4B
    // pp2048 @ d32768: q8_0 2793 vs tile 1891, f16 2841 vs 2000.  (AMD prefill takes <= 32 columns for heads
    // > 256, see ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1; the 64-column row is kept consistent.)
    if (!is_rdna3_5 && !is_rdna4 && DKQ == 512 && DV == 512) {
        if (ncols == 16) { return fattn_mma_config(128, 1, 64, 64, 64, 64, 1, false); }
        if (ncols == 32) { return fattn_mma_config(128, 1, 32, 64, 64, 64, 1, false); }
        if (ncols == 64) { return fattn_mma_config(128, 1, 32, 64, 64, 64, 1, false); }
    }
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512,  8, 128, 3,  64,  96,  64, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 16, 128, 2,  64,  96,  16,  16, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 32, 128, 2,  32, 128, 128,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 64, 128, 2,  32, 128, 128, 128, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512,  8, 128, 3,  64,  96,  64, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 16, 128, 2,  64,  96,  16,  16, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 32, 256, 2, 128,  96,  64,  16, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 128, 2,  32, 160, 128, 128, 1, true);

    return fattn_mma_config(32, 1, 0, 0, 0, 0, 0, false);
}

static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_cdna(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64,  8, 128, 1,  64,  32,  32,  32, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 16, 256, 2,  64,  32,  32,  32, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 32, 256, 2,  64,  32,  32,  32, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 64, 256, 3,  64,  32,  32,  32, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80,  8, 256, 2,  64,  40,  40,  40, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 16, 256, 2,  64,  40,  40,  40, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 32, 256, 2,  64,  40,  40,  40, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 64, 256, 2,  64,  40,  40,  40, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96,  8, 256, 2,  64,  48,  48,  48, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 16, 256, 2,  64,  48,  48,  48, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 32, 256, 2,  64,  48,  48,  48, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 64, 256, 2,  64,  48,  48,  48, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112,  8, 256, 2,  64,  56,  56,  56, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 16, 256, 2,  64,  56,  56,  56, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 32, 256, 2,  64,  56,  56,  56, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 64, 256, 2,  64,  56,  56,  56, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128,  8, 256, 2,  64,  64,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 16, 256, 2,  64,  64,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 32, 256, 2,  64,  64,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 64, 256, 2,  64,  64,  64,  64, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128,  8, 256, 1,  64,  64,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 16, 256, 1,  64,  64,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 32, 256, 1,  64,  64,  64,  64, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 64, 512, 1,  64,  64,  64,  64, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256,  8, 256, 1,  64, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 16, 256, 1,  64, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 32, 256, 1,  64, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 512, 1,  64, 128, 128,  64, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 32, 256, 1,  64, 160, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 64, 256, 1,  64, 160, 128, 128, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512,  8, 256, 1,  64, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 16, 256, 1,  64, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 32, 256, 1,  64, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 64, 256, 1,  64, 128, 128, 128, 1, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512,  8, 256, 1,  64, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 16, 256, 1,  64, 128, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 32, 256, 1,  64, 160, 128, 128, 1, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 256, 1,  64, 160, 128, 128, 1, true);

    return fattn_mma_config(32, 1, 0, 0, 0, 0, 0, false);
}

static __host__ fattn_mma_config ggml_cuda_fattn_mma_get_config(const int DKQ, const int DV, const int ncols, const int cc, const int ncols2,
        const bool band_wide) {
    if (ampere_mma_available(cc)) {
        return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);
    }
    if (turing_mma_available(cc)) {
        return ggml_cuda_fattn_mma_get_config_turing(DKQ, DV, ncols);
    }
    if (amd_mfma_available(cc)) {
        return ggml_cuda_fattn_mma_get_config_cdna(DKQ, DV, ncols);
    }
    if (amd_wmma_available(cc)) {
        return ggml_cuda_fattn_mma_get_config_rdna(DKQ, DV, ncols, GGML_CUDA_CC_IS_RDNA3_5(cc), GGML_CUDA_CC_IS_RDNA4(cc), ncols2, band_wide);
    }
    GGML_ASSERT(volta_mma_available(cc));
    return ggml_cuda_fattn_mma_get_config_volta(DKQ, DV, ncols);
}

static constexpr __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config(const int DKQ, const int DV, const int ncols, const int ncols2 = 1,
        const bool band_wide = false) {
    (void) ncols2;
#if defined(AMPERE_MMA_AVAILABLE)
    return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);
#elif defined(TURING_MMA_AVAILABLE)
    return ggml_cuda_fattn_mma_get_config_turing(DKQ, DV, ncols);
#elif defined(AMD_MFMA_AVAILABLE)
    return ggml_cuda_fattn_mma_get_config_cdna(DKQ, DV, ncols);
#elif defined(VOLTA_MMA_AVAILABLE)
    return ggml_cuda_fattn_mma_get_config_volta(DKQ, DV, ncols);
#elif defined(AMD_WMMA_AVAILABLE)
#if defined(RDNA3_5)
    return ggml_cuda_fattn_mma_get_config_rdna(DKQ, DV, ncols, true, false, ncols2, band_wide);
#elif defined(RDNA4)
    return ggml_cuda_fattn_mma_get_config_rdna(DKQ, DV, ncols, false, true, ncols2, band_wide);
#else
    return ggml_cuda_fattn_mma_get_config_rdna(DKQ, DV, ncols, false, false, ncols2, band_wide);
#endif
#else
    GGML_UNUSED_VARS(DKQ, DV, ncols, band_wide);
    return fattn_mma_config(32, 1, 0, 0, 0, 0, 0, false);
#endif // defined(AMPERE_MMA_AVAILABLE)
}

static __host__ int ggml_cuda_fattn_mma_get_nthreads(const int DKQ, const int DV, const int ncols, const int cc, const int ncols2, const bool band_wide) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc, ncols2, band_wide).nthreads;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nthreads(const int DKQ, const int DV, const int ncols, const int ncols2 = 1, const bool band_wide = false) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, ncols2, band_wide).nthreads;
}

static __host__ int ggml_cuda_fattn_mma_get_occupancy(const int DKQ, const int DV, const int ncols, const int cc, const int ncols2, const bool band_wide) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc, ncols2, band_wide).occupancy;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_occupancy(const int DKQ, const int DV, const int ncols, const int ncols2 = 1, const bool band_wide = false) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, ncols2, band_wide).occupancy;
}

static __host__ int ggml_cuda_fattn_mma_get_nbatch_fa(const int DKQ, const int DV, const int ncols, const int cc, const int ncols2, const bool band_wide) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc, ncols2, band_wide).nbatch_fa;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nbatch_fa(const int DKQ, const int DV, const int ncols, const int ncols2 = 1, const bool band_wide = false) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, ncols2, band_wide).nbatch_fa;
}

static __host__ int ggml_cuda_fattn_mma_get_nbatch_K2(const int DKQ, const int DV, const int ncols, const int cc, const int ncols2, const bool band_wide) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc, ncols2, band_wide).nbatch_K2;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nbatch_K2(const int DKQ, const int DV, const int ncols, const int ncols2 = 1, const bool band_wide = false) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, ncols2, band_wide).nbatch_K2;
}

static __host__ int ggml_cuda_fattn_mma_get_nbatch_V2(const int DKQ, const int DV, const int ncols, const int cc, const int ncols2, const bool band_wide) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc, ncols2, band_wide).nbatch_V2;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nbatch_V2(const int DKQ, const int DV, const int ncols, const int ncols2 = 1, const bool band_wide = false) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, ncols2, band_wide).nbatch_V2;
}

static __host__ int ggml_cuda_fattn_mma_get_nbatch_combine(const int DKQ, const int DV, const int ncols, const int cc, const int ncols2, const bool band_wide) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc, ncols2, band_wide).nbatch_combine;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nbatch_combine(const int DKQ, const int DV, const int ncols, const int ncols2 = 1, const bool band_wide = false) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, ncols2, band_wide).nbatch_combine;
}

static __host__ int ggml_cuda_fattn_mma_get_nstages_target(const int DKQ, const int DV, const int ncols, const int cc, const int ncols2, const bool band_wide) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc, ncols2, band_wide).nstages_target;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nstages_target(const int DKQ, const int DV, const int ncols, const int ncols2 = 1, const bool band_wide = false) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, ncols2, band_wide).nstages_target;
}

static __host__ bool ggml_cuda_fattn_mma_get_Q_in_reg(const int DKQ, const int DV, const int ncols, const int cc, const int ncols2, const bool band_wide) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc, ncols2, band_wide).Q_in_reg;
}

static constexpr __device__ bool ggml_cuda_fattn_mma_get_Q_in_reg(const int DKQ, const int DV, const int ncols, const int ncols2 = 1, const bool band_wide = false) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, ncols2, band_wide).Q_in_reg;
}

// The RDNA4 band rows that have a second kernel variant, which fetches q8_0 K/V ahead (see flash_attn_ext_f16_band_depth):
// the 32-column block (query widths 1..4) and the 64-column block (widths 5..8).  The variant is launched for q8_0 K and V;
// the other K/V types, and q8_0 with GGML_HIP_FA_BAND_FETCH_AHEAD=0, run the kernel without it.
static constexpr __host__ __device__ bool ggml_cuda_fattn_mma_band_has_fetch_ahead(const int DKQ, const int DV, const int ncols1, const int ncols2) {
#if defined(GGML_USE_HIP)
    constexpr bool is_hip = true;
#else
    constexpr bool is_hip = false;
#endif // defined(GGML_USE_HIP)
    return is_hip && DKQ == 256 && DV == 256 && ncols2 == 8 && (ncols1 == 4 || ncols1 == 8);
}

// In the fetch-ahead variant the 32-column block reads Q from shared memory: a register-resident Q is spilled there, and
// its scratch reloads count in the same load counter as the fetched K/V, so every reload would wait for the fetch-ahead
// (same values, so the result does not change).  gfx1201, KV 110592: 13 % faster with both than without either; the
// 64-column block is faster with Q in registers.
static constexpr __host__ __device__ bool ggml_cuda_fattn_mma_band_fetch_ahead_Q_in_lds(const int DKQ, const int DV, const int ncols1, const int ncols2) {
    return ggml_cuda_fattn_mma_band_has_fetch_ahead(DKQ, DV, ncols1, ncols2) && ncols1 == 4;
}

// Swizzling needs a tile stride that is a multiple of 32 half2 columns.
static constexpr __host__ __device__ bool ggml_cuda_fattn_mma_bank_aligned(const int nbatch_2) {
    return nbatch_2 >= 32 && nbatch_2 % 32 == 0;
}

// Swizzling needs ldmatrix, on other hardware the tiles keep the row padding.
static __host__ bool ggml_cuda_fattn_mma_get_swizzled(const int DKQ, const int DV, const int ncols1, const int ncols2, const int cc,
        const bool band_wide) {
    const fattn_mma_config cfg = ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols1*ncols2, cc, ncols2, band_wide);
    return turing_mma_available(cc) && ggml_cuda_fattn_mma_bank_aligned(cfg.nbatch_K2) && ggml_cuda_fattn_mma_bank_aligned(cfg.nbatch_V2);
}

static constexpr __device__ bool ggml_cuda_fattn_mma_get_swizzled(const int DKQ, const int DV, const int ncols1, const int ncols2,
        const bool band_wide = false) {
#if defined(TURING_MMA_AVAILABLE)
    const fattn_mma_config cfg = ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols1*ncols2, ncols2, band_wide);
    return ggml_cuda_fattn_mma_bank_aligned(cfg.nbatch_K2) && ggml_cuda_fattn_mma_bank_aligned(cfg.nbatch_V2);
#else
    GGML_UNUSED_VARS(DKQ, DV, ncols1, ncols2, band_wide);
    return false;
#endif // defined(TURING_MMA_AVAILABLE)
}

// Row padding is only needed if the tile is not swizzled.
static constexpr __host__ __device__ int ggml_cuda_fattn_mma_get_stride_tile(const int nbatch_2, const bool swizzled) {
    return swizzled ? nbatch_2 : nbatch_2 + 4;
}

static constexpr __device__ int get_cols_per_thread() {
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
    return 1; // AMD has a single column per thread.
#else
    return 2; // This is specifically KQ columns, Volta only has a single VKQ column.
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
}

static __host__ int get_cols_per_warp(const int cc) {
    if (turing_mma_available(cc) || amd_wmma_available(cc) || amd_mfma_available(cc)) {
        return 16;
    } else {
        // Volta
        return 32;
    }
}

// ------------------------------------------------------------------------------------------------------------------

static __host__ int ggml_cuda_fattn_mma_get_nstages(const int DKQ, const int DV, const int ncols1, const int ncols2, const int cc,
        const bool band_wide) {
    return cp_async_available(cc) && ncols2 >= 2 ? ggml_cuda_fattn_mma_get_nstages_target(DKQ, DV, ncols1*ncols2, cc, ncols2, band_wide) : 0;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nstages(
        const int DKQ, const int DV, const int ncols1, const int ncols2, const bool use_sparse, const bool band_wide = false) {
#ifdef CP_ASYNC_AVAILABLE
    const int nstages_target = ncols2 >= 2 ? ggml_cuda_fattn_mma_get_nstages_target(DKQ, DV, ncols1*ncols2, ncols2, band_wide) : 0;
    // sparse gather is not implemented for multi-stage loading
    return use_sparse && nstages_target > 1 ? 1 : nstages_target;
#else
    GGML_UNUSED_VARS(DKQ, DV, ncols1, ncols2, use_sparse);
    return 0;
#endif // CP_ASYNC_AVAILABLE
}

// ------------------------------------------------------------------------------------------------------------------

// V4: native q8_0/q4_0 K/V staging.  Same (i, k) thread/chunk mapping as the half2 loader above, but
// each 16-byte chunk is dequantized from the quantized source (ggml_cuda_fattn_dequantize_q8_0_chunk /
// _q4_0_chunk) instead of copied.  `KV` is the byte-addressed row base (the row index k_VKQ_0 + i is
// applied here, like the half2 loader's `KV`), `stride_KV` is the row stride in bytes and `el_off` is
// the element offset of the chunk slice within a row (0 for whole rows).
template<ggml_type type_KV, int stride_tile, bool swz, int nwarps, int nbatch_fa, bool oob_check, bool use_sparse>
static __device__ __forceinline__ void flash_attn_ext_f16_load_tile_native(
        const char * const __restrict__ KV, half2 * const __restrict__ tile_KV, const int D2, const int stride_KV,
        const int el_off, const int k_VKQ_0, const int i_sup, const int32_t * const __restrict__ indices) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    // The minimum granularity is 16 bytes, i.e. GGML_CUDA_FA_Q8_CHUNK elements.
    constexpr int h2_per_chunk = 16/sizeof(half2);
    static_assert(2*h2_per_chunk == GGML_CUDA_FA_Q8_CHUNK, "bad chunk size");
    const int chunks_per_row = D2 / h2_per_chunk;

    auto load = [&] __device__ (const int n) {
        const int stride_k = 32 >> n;
        const int k0_start = stride_k == 32 ? 0 : chunks_per_row - chunks_per_row % (2*stride_k);
        const int k0_stop  =                      chunks_per_row - chunks_per_row % (1*stride_k);
        const int stride_i = warp_size / stride_k;

        if (k0_start == k0_stop) {
            return;
        }

#pragma unroll
        for (int i0 = 0; i0 < nbatch_fa; i0 += nwarps*stride_i) {
            const int i = i0 + threadIdx.y*stride_i + (stride_k == warp_size ? 0 : threadIdx.x / stride_k);

            if (i0 + nwarps*stride_i > nbatch_fa && i >= nbatch_fa) {
                break;
            }

            int64_t i_KV;
            if constexpr (use_sparse) {
                // padded slots gather row 0, the -inf mask removes their contribution
                const int32_t index = i < i_sup ? indices[k_VKQ_0 + i] : -1;
                i_KV = index;
            } else {
                i_KV = k_VKQ_0 + i;
            }

#pragma unroll
            for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

                __align__(16) half2 tmp[h2_per_chunk];
                if ((use_sparse ? i_KV >= 0 : true) && (!oob_check || i < i_sup)) {
                    const char * const __restrict__ row = KV + (size_t) i_KV*stride_KV;
                    const int el = el_off + k*h2_per_chunk*2;
                    if constexpr (type_KV == GGML_TYPE_Q8_0) {
                        ggml_cuda_fattn_dequantize_q8_0_chunk(row, el, tmp);
                    } else if constexpr (type_KV == GGML_TYPE_Q4_0) {
                        ggml_cuda_fattn_dequantize_q4_0_chunk(row, el, tmp);
                    } else if constexpr (type_KV == GGML_TYPE_Q4_1) {
                        ggml_cuda_fattn_dequantize_q4_1_chunk(row, el, tmp);
                    } else if constexpr (type_KV == GGML_TYPE_Q5_0) {
                        ggml_cuda_fattn_dequantize_q5_0_chunk(row, el, tmp);
                    } else if constexpr (type_KV == GGML_TYPE_Q5_1) {
                        ggml_cuda_fattn_dequantize_q5_1_chunk(row, el, tmp);
                    } else {
                        ggml_cuda_fattn_dequantize_iq4_nl_chunk(row, el, tmp);
                    }
                } else {
#pragma unroll
                    for (int l = 0; l < h2_per_chunk; ++l) {
                        tmp[l] = make_half2(0.0f, 0.0f);
                    }
                }
                ggml_cuda_memcpy_1<16>(swizzle<stride_tile>(tile_KV, i*stride_tile + k*h2_per_chunk, i), tmp);
            }
        }
    };
    // 1: max 32*16=512 bytes, 256 half
    // 2: max 16*16=256 bytes, 128 half
    // 3: max  8*16=128 bytes,  64 half
    // 4: max  4*16= 64 bytes,  32 half
    // 5: max  2*16= 32 bytes,  16 half
    // 6: max  1*16= 16 bytes,   8 half
    ggml_cuda_unroll<6>{}(load);
}

// Block-15 amendment: native bf16 K/V staging.  A bf16 source row and the F16 tile it feeds have the
// same byte layout (2 bytes per element, 16-byte chunks), so this is the F16 loader with a register
// conversion per chunk instead of the copy: the staged values are exactly the ones the launcher's
// F16 staging would hold (ggml_cuda_cast<half>(nv_bfloat16) is a single bf16 -> f32 -> f16
// round-to-nearest), so the arithmetic is unchanged.  Same (i, k) thread/chunk mapping as the q8_0
// loader above.
template<int stride_tile, bool swz, int nwarps, int nbatch_fa, bool oob_check, bool use_sparse>
static __device__ __forceinline__ void flash_attn_ext_f16_load_tile_bf16(
        const char * const __restrict__ KV, half2 * const __restrict__ tile_KV, const int D2, const int stride_KV,
        const int el_off, const int k_VKQ_0, const int i_sup, const int32_t * const __restrict__ indices) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    // The minimum granularity is 16 bytes, i.e. GGML_CUDA_FA_Q8_CHUNK 2-byte elements.
    constexpr int h2_per_chunk = 16/sizeof(half2);
    static_assert(2*h2_per_chunk == GGML_CUDA_FA_Q8_CHUNK, "bad chunk size");
    const int chunks_per_row = D2 / h2_per_chunk;

    auto load = [&] __device__ (const int n) {
        const int stride_k = 32 >> n;
        const int k0_start = stride_k == 32 ? 0 : chunks_per_row - chunks_per_row % (2*stride_k);
        const int k0_stop  =                      chunks_per_row - chunks_per_row % (1*stride_k);
        const int stride_i = warp_size / stride_k;

        if (k0_start == k0_stop) {
            return;
        }

#pragma unroll
        for (int i0 = 0; i0 < nbatch_fa; i0 += nwarps*stride_i) {
            const int i = i0 + threadIdx.y*stride_i + (stride_k == warp_size ? 0 : threadIdx.x / stride_k);

            if (i0 + nwarps*stride_i > nbatch_fa && i >= nbatch_fa) {
                break;
            }

            int64_t i_KV;
            if constexpr (use_sparse) {
                // padded slots gather row 0, the -inf mask removes their contribution
                const int32_t index = i < i_sup ? indices[k_VKQ_0 + i] : -1;
                i_KV = index;
            } else {
                i_KV = k_VKQ_0 + i;
            }

#pragma unroll
            for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

                __align__(16) nv_bfloat162 tmp_bf[h2_per_chunk];
                __align__(16) half2         tmp   [h2_per_chunk];
                if ((use_sparse ? i_KV >= 0 : true) && (!oob_check || i < i_sup)) {
                    // el_off counts elements within the row and a bf16 element is 2 bytes, so the
                    // chunk at (el_off + k*h2_per_chunk*2) starts at 2*el_off + 16*k bytes.
                    ggml_cuda_memcpy_1<16>(tmp_bf, KV + (size_t) i_KV*stride_KV + (el_off + k*h2_per_chunk*2)*2);
#pragma unroll
                    for (int l = 0; l < h2_per_chunk; ++l) {
                        tmp[l] = __float22half2_rn(ggml_cuda_cast<float2>(tmp_bf[l]));
                    }
                } else {
#pragma unroll
                    for (int l = 0; l < h2_per_chunk; ++l) {
                        tmp[l] = make_half2(0.0f, 0.0f);
                    }
                }
                ggml_cuda_memcpy_1<16>(swizzle<stride_tile>(tile_KV, i*stride_tile + k*h2_per_chunk, i), tmp);
            }
        }
    };
    // 1: max 32*16=512 bytes, 256 half
    // 2: max 16*16=256 bytes, 128 half
    // 3: max  8*16=128 bytes,  64 half
    // 4: max  4*16= 64 bytes,  32 half
    // 5: max  2*16= 32 bytes,  16 half
    // 6: max  1*16= 16 bytes,   8 half
    ggml_cuda_unroll<6>{}(load);
}

template<int stride_tile, bool swz, int nwarps, int nbatch_fa, bool use_cp_async, bool oob_check, bool use_sparse>
static __device__ __forceinline__ void flash_attn_ext_f16_load_tile(
        const half2 * const __restrict__ KV, half2 * const __restrict__ tile_KV, const int D2, const int stride_KV,
        const int k_VKQ_0, const int i_sup, const int32_t * const __restrict__ indices,
        const char * const __restrict__ KV_native, const int stride_KV_native, const int el_off,
        const int kv_native_type) {
    if (kv_native_type != FATTN_KV_NATIVE_NONE && kv_native_type != FATTN_KV_NATIVE_BF16) {
        // V4 / issue #30 item 2: dequantize the raw quantized source instead of reading the F16
        // staging copy.  (bf16 has its own converting loader below.)
#define GGML_CUDA_FATTN_MMA_NATIVE_ARM(NATIVE, TYPE)                                                       \
        if (kv_native_type == NATIVE) {                                                                    \
            flash_attn_ext_f16_load_tile_native<TYPE, stride_tile, swz, nwarps, nbatch_fa, oob_check, use_sparse> \
                (KV_native, tile_KV, D2, stride_KV_native, el_off, k_VKQ_0, i_sup, indices);               \
            return;                                                                                        \
        }
        GGML_CUDA_FATTN_MMA_NATIVE_ARM(FATTN_KV_NATIVE_Q8_0,  GGML_TYPE_Q8_0)
        GGML_CUDA_FATTN_MMA_NATIVE_ARM(FATTN_KV_NATIVE_Q4_0,  GGML_TYPE_Q4_0)
        GGML_CUDA_FATTN_MMA_NATIVE_ARM(FATTN_KV_NATIVE_Q4_1,  GGML_TYPE_Q4_1)
        GGML_CUDA_FATTN_MMA_NATIVE_ARM(FATTN_KV_NATIVE_Q5_0,  GGML_TYPE_Q5_0)
        GGML_CUDA_FATTN_MMA_NATIVE_ARM(FATTN_KV_NATIVE_Q5_1,  GGML_TYPE_Q5_1)
        GGML_CUDA_FATTN_MMA_NATIVE_ARM(FATTN_KV_NATIVE_IQ4_NL, GGML_TYPE_IQ4_NL)
#undef GGML_CUDA_FATTN_MMA_NATIVE_ARM
    }
    if (kv_native_type == FATTN_KV_NATIVE_BF16) {
        // Block 15: convert the bf16 source while staging instead of reading the F16 staging copy.
        flash_attn_ext_f16_load_tile_bf16<stride_tile, swz, nwarps, nbatch_fa, oob_check, use_sparse>
            (KV_native, tile_KV, D2, stride_KV_native, el_off, k_VKQ_0, i_sup, indices);
        return;
    }
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    // K/V data is loaded with decreasing granularity for D for better memory bandwidth.
    // The minimum granularity is 16 bytes.
    constexpr int chunk_size = 16;
    constexpr int h2_per_chunk = chunk_size / sizeof(half2);
    const int chunks_per_row = D2 / h2_per_chunk;
    if constexpr (use_cp_async) {
        static_assert(warp_size == 32, "bad warp_size");
        static_assert(!oob_check || use_sparse, "OOB check not compatible with cp_async");
        constexpr int preload = 64;

        const unsigned int tile_KV_32 = ggml_cuda_cvta_generic_to_shared(tile_KV);

        auto load = [&] __device__ (auto n) {
            const int stride_k = warp_size >> n;
            const int k0_start = stride_k == warp_size ? 0 : chunks_per_row - chunks_per_row % (2*stride_k);
            const int k0_stop  =                             chunks_per_row - chunks_per_row % (1*stride_k);
            const int stride_i = warp_size / stride_k;

            if (k0_start == k0_stop) {
                return;
            }

#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += nwarps*stride_i) {
                const int i = i0 + threadIdx.y*stride_i + (stride_k == warp_size ? 0 : threadIdx.x / stride_k);

                if (i0 + nwarps*stride_i > nbatch_fa && i >= nbatch_fa) {
                    break;
                }

                int64_t i_KV;
                if constexpr (use_sparse) {
                    // padded slots gather row 0, the -inf mask removes their contribution
                    const int32_t index = i < i_sup ? indices[k_VKQ_0 + i] : 0;
                    i_KV = index >= 0 ? index : 0;
                } else {
                    i_KV = k_VKQ_0 + i;
                }

#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

                    cp_async_cg_16<preload>(tile_KV_32 + swizzle<stride_tile*sizeof(half2), char>(i*stride_tile*sizeof(half2) + k*chunk_size, i), KV + i_KV*stride_KV + k*h2_per_chunk);
                }
            }
        };
        // 1: max 32*16=512 bytes, 256 half
        // 2: max 16*16=256 bytes, 128 half
        // 3: max  8*16=128 bytes,  64 half
        // 4: max  4*16= 64 bytes,  32 half
        // 5: max  2*16= 32 bytes,  16 half
        // 6: max  1*16= 16 bytes,   8 half
        ggml_cuda_unroll<6>{}(load);
    } else {
        const half2 zero[4] = {{0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}};
        auto load = [&] __device__ (const int n) {
            const int stride_k = 32 >> n;
            const int k0_start = stride_k == 32 ? 0 : chunks_per_row - chunks_per_row % (2*stride_k);
            const int k0_stop  =                      chunks_per_row - chunks_per_row % (1*stride_k);
            const int stride_i = warp_size / stride_k;

            if (k0_start == k0_stop) {
                return;
            }

#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += nwarps*stride_i) {
                const int i = i0 + threadIdx.y*stride_i + (stride_k == warp_size ? 0 : threadIdx.x / stride_k);

                if (i0 + nwarps*stride_i > nbatch_fa && i >= nbatch_fa) {
                    break;
                }

#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

                    const half2 * src;
                    if constexpr (use_sparse) {
                        const int32_t index = i < i_sup ? indices[k_VKQ_0 + i] : -1;
                        src = index >= 0 ? KV + int64_t(index)*stride_KV + k*h2_per_chunk : zero;
                    } else {
                        src = !oob_check || i < i_sup ? KV + int64_t(k_VKQ_0 + i)*stride_KV + k*h2_per_chunk : zero;
                    }
                    // Keep the typed store when the tile is not swizzled (every AMD target): the
                    // byte-pointer form drops the half2 alignment and HIP then splits the 16-byte
                    // shared-memory store (issue #47).  The address is identical either way.
                    ggml_cuda_memcpy_1<16>(swizzle<stride_tile>(tile_KV, i*stride_tile + k*h2_per_chunk, i), src);
                }
            }
        };
        // 1: max 32*16=512 bytes, 256 half
        // 2: max 16*16=256 bytes, 128 half
        // 3: max  8*16=128 bytes,  64 half
        // 4: max  4*16= 64 bytes,  32 half
        // 5: max  2*16= 32 bytes,  16 half
        // 6: max  1*16= 16 bytes,   8 half
        ggml_cuda_unroll<6>{}(load);
    }
}

// kq_derived_t is defined in fattn-common.cuh (shared with the tile kernel).
template<int ncols1, int nwarps, int nbatch_fa, bool use_cp_async, bool oob_check, bool use_sparse>
static __device__ __forceinline__ void flash_attn_ext_f16_load_mask(
        const half * const __restrict__ mask_h, half * const __restrict__ tile_mask,
        const int stride_mask, const int k_VKQ_0, const int i_sup, const int j0, const uint3 ne01,
        const int32_t * const __restrict__ indices, const kq_derived_t derived) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    // V3 derived mask: nothing to load, compute 0 / -INFINITY per cell instead.  This is a runtime
    // branch and must precede the if constexpr chain below: the derived path cannot use cp_async.
    // The i_sup handling mirrors the packed path exactly (the oob case holds 0.0f out of bounds).
    //
    // Shape mirrors the packed fallback below (two cells per thread step, one half2 store): the first
    // cut of this loop did one cell per step with a scalar half store -- twice the iterations and
    // unvectorised shared stores for strictly *less* global traffic than the packed path -- and that
    // was a net loss wherever the saved mask bandwidth did not cover it (issue #30: -3.5% PP512 @
    // d98k on gfx1100, -6.0% on a 27B 2-GPU layer split on gfx1201).  The per-cell position does not
    // depend on the query row, so it is also hoisted out of the j1 loop: the packed path re-reads the
    // mask for every row, the derived path must not re-read cell_pos for every row.
    // nbatch_fa is a multiple of 32 and <= 256 (static_assert on the config table), so the even `i`
    // never makes `i + 1` leave the tile and no tail handling is needed.
    if (derived.cell_pos != nullptr) {
#pragma unroll
        for (int i0 = 0; i0 < nbatch_fa; i0 += 2*warp_size) {
            const int i = i0 + 2*threadIdx.x;

            if (i >= nbatch_fa) {
                continue; // the mask add and the dead-column detector never read past nbatch_fa
            }

            const int p0 = derived.cell_pos[k_VKQ_0 + i];
            const int p1 = derived.cell_pos[k_VKQ_0 + i + 1];

            const bool oob0 = oob_check && (i    ) >= i_sup;
            const bool oob1 = oob_check && (i + 1) >= i_sup;

#pragma unroll
            for (int j1 = 0; j1 < ncols1; j1 += nwarps) {
                const int j_sram = j1 + threadIdx.y;

                if (j1 + nwarps > ncols1 && j_sram >= ncols1) {
                    break;
                }

                const int j_vram = fastmodulo(j0 + j_sram, ne01);
                const int lo = derived.tok_lo[j_vram];
                const int hi = derived.tok_hi[j_vram];

                const half h0 = oob0 || (p0 >= lo && p0 <= hi) ? half(0.0f) : half(-INFINITY);
                const half h1 = oob1 || (p1 >= lo && p1 <= hi) ? half(0.0f) : half(-INFINITY);

                *reinterpret_cast<half2 *>(tile_mask + j_sram*(nbatch_fa + 8) + i) = __halves2half2(h0, h1);
            }
        }
        return;
    }

    if constexpr (use_cp_async) {
        static_assert(nbatch_fa <= 8*warp_size && nbatch_fa % 8 == 0, "bad nbatch_fa");
        static_assert(!oob_check, "OOB check incompatible with cp_async");
        static_assert(!use_sparse, "sparse gather incompatible with cp_async");
        constexpr int preload = nbatch_fa >= 32 ? nbatch_fa * sizeof(half) : 64;
        constexpr int cols_per_warp = 8*warp_size/nbatch_fa;
        constexpr int stride_j = nwarps * cols_per_warp;

        const unsigned int tile_mask_32 = ggml_cuda_cvta_generic_to_shared(tile_mask);

#pragma unroll
        for (int j1 = 0; j1 < ncols1; j1 += stride_j) {
            const int j_sram = j1 + threadIdx.y*cols_per_warp + threadIdx.x / (warp_size/cols_per_warp);
            const int j_vram = fastmodulo(j0 + j_sram, ne01);

            if (j1 + stride_j > ncols1 && j_sram >= ncols1) {
                break;
            }

            const int i = 8 * (threadIdx.x % (nbatch_fa/8));

            cp_async_cg_16<preload>(tile_mask_32 + j_sram*(nbatch_fa*sizeof(half) + 16) + i*sizeof(half), mask_h + int64_t(j_vram)*stride_mask + k_VKQ_0 + i);
        }
    } else if constexpr (oob_check || use_sparse) {
#pragma unroll
        for (int j1 = 0; j1 < ncols1; j1 += nwarps) {
            const int j_sram = j1 + threadIdx.y;
            const int j_vram = fastmodulo(j0 + j_sram, ne01);

            if (j1 + nwarps > ncols1 && j_sram >= ncols1) {
                break;
            }

#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                if constexpr (use_sparse) {
                    const int32_t index = i < i_sup ? indices[k_VKQ_0 + i] : -1;
                    tile_mask[j_sram*(nbatch_fa + 8) + i] = index >= 0 ? mask_h[int64_t(j_vram)*stride_mask + index] : half(-INFINITY);
                } else {
                    tile_mask[j_sram*(nbatch_fa + 8) + i] = i < i_sup ? mask_h[int64_t(j_vram)*stride_mask + k_VKQ_0 + i] : half(0.0f);
                }
            }
        }
    } else if constexpr (nbatch_fa < 2*warp_size) {
        constexpr int cols_per_warp = 2*warp_size/nbatch_fa;
        constexpr int stride_j = nwarps * cols_per_warp;
#pragma unroll
        for (int j1 = 0; j1 < ncols1; j1 += stride_j) {
            const int j_sram = j1 + threadIdx.y*cols_per_warp + threadIdx.x / (warp_size/cols_per_warp);
            const int j_vram = fastmodulo(j0 + j_sram, ne01);

            if (j1 + stride_j > ncols1 && j_sram >= ncols1) {
                break;
            }

            const int i = threadIdx.x % (warp_size/cols_per_warp);

            ggml_cuda_memcpy_1<sizeof(half2)>(tile_mask + j_sram*(nbatch_fa + 8) + 2*i, mask_h + int64_t(j_vram)*stride_mask + k_VKQ_0 + 2*i);
        }
    } else {
#pragma unroll
        for (int j1 = 0; j1 < ncols1; j1 += nwarps) {
            const int j_sram = j1 + threadIdx.y;
            const int j_vram = fastmodulo(j0 + j_sram, ne01);

            if (j1 + nwarps > ncols1 && j_sram >= ncols1) {
                break;
            }

#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += 2*warp_size) {
                const int i = i0 + 2*threadIdx.x;

                ggml_cuda_memcpy_1<sizeof(half2)>(tile_mask + j_sram*(nbatch_fa + 8) + i, mask_h + int64_t(j_vram)*stride_mask + k_VKQ_0 + i);
            }
        }
    }
}

// RDNA4 decode/verify band with q8_0 K and V: fetch ahead.  The synchronous loader issues the global loads of a K or V
// batch only once the previous batch has been computed and every warp has passed the barrier, then waits for them at
// once, so each of the four batches of a KV iteration exposes a full memory round trip (on gfx1201 about 70 % of the
// band's time is spent waiting for loads or at the barriers behind them).  Instead, the raw q8_0 bytes of a batch are
// fetched into registers ahead of use (all K batches at the start of the iteration, each V batch as soon as a K batch
// has been staged) and only converted and stored to shared memory when the batch is due.  The staged tiles hold the
// same values in the same slots as with flash_attn_ext_f16_load_tile_native, so the result is bit-identical.
template<int nwarps, int nbatch_fa, int D2>
struct flash_attn_ext_f16_band_batch {
    static constexpr int chunks_per_row    = D2 / (16/sizeof(half2));
    static constexpr int chunks_per_thread = nbatch_fa*chunks_per_row >= nwarps*32 ? nbatch_fa*chunks_per_row / (nwarps*32) : 1;

    ggml_cuda_fattn_q8_0_chunk_raw raw[chunks_per_thread];
    bool                           ok [chunks_per_thread];
};

template<int nwarps, int nbatch_fa, int D2, bool oob_check>
static __device__ __forceinline__ void flash_attn_ext_f16_band_fetch(
        flash_attn_ext_f16_band_batch<nwarps, nbatch_fa, D2> & batch, const char * const __restrict__ KV, const int stride_KV,
        const int el_off, const int k_VKQ_0, const int i_sup) {
    using batch_t = flash_attn_ext_f16_band_batch<nwarps, nbatch_fa, D2>;
    static_assert(nbatch_fa*batch_t::chunks_per_row % (nwarps*32) == 0, "bad band batch");
    const int tid = threadIdx.y*32 + threadIdx.x;
#pragma unroll
    for (int j = 0; j < batch_t::chunks_per_thread; ++j) {
        const int c = tid + j*nwarps*32;
        const int i = c / batch_t::chunks_per_row;
        const int k = c % batch_t::chunks_per_row;
        batch.ok[j] = !oob_check || i < i_sup;
        if (batch.ok[j]) {
            ggml_cuda_fattn_load_q8_0_chunk(KV + (size_t) (k_VKQ_0 + i)*stride_KV, el_off + k*GGML_CUDA_FA_Q8_CHUNK, batch.raw[j]);
        }
    }
}

template<int stride_tile, int nwarps, int nbatch_fa, int D2>
static __device__ __forceinline__ void flash_attn_ext_f16_band_stage(
        const flash_attn_ext_f16_band_batch<nwarps, nbatch_fa, D2> & batch, half2 * const __restrict__ tile_KV) {
    using batch_t = flash_attn_ext_f16_band_batch<nwarps, nbatch_fa, D2>;
    constexpr int h2_per_chunk = 16/sizeof(half2);
    const int tid = threadIdx.y*32 + threadIdx.x;
#pragma unroll
    for (int j = 0; j < batch_t::chunks_per_thread; ++j) {
        const int c = tid + j*nwarps*32;
        const int i = c / batch_t::chunks_per_row;
        const int k = c % batch_t::chunks_per_row;
        __align__(16) half2 tmp[h2_per_chunk];
        if (batch.ok[j]) {
            // Pin the conversion here: it is pure arithmetic on the fetched words, and without the opaque asm the
            // compiler moves it up to the fetch (through the merge after the runtime K/V type check), which waits for
            // the fetch right there and keeps the unpacked bytes live instead of the packed words.
            ggml_cuda_fattn_q8_0_chunk_raw raw = batch.raw[j];
            uint16_t d_bits;
            memcpy(&d_bits, &raw.d, sizeof(d_bits));
            asm volatile("" : "+v"(raw.q[0]), "+v"(raw.q[1]), "+v"(d_bits));
            memcpy(&raw.d, &d_bits, sizeof(d_bits));
            ggml_cuda_fattn_convert_q8_0_chunk(raw, tmp);
        } else {
#pragma unroll
            for (int l = 0; l < h2_per_chunk; ++l) {
                tmp[l] = make_half2(0.0f, 0.0f);
            }
        }
        ggml_cuda_memcpy_1<16>(swizzle<stride_tile>(tile_KV, i*stride_tile + k*h2_per_chunk, i), tmp);
    }
}

// The band's barriers only order shared-memory accesses (a KV iteration shares nothing through global memory), so the
// fetched-ahead loads stay in flight across them.  __syncthreads() would also wait for every outstanding global load.
template<bool band>
static __device__ __forceinline__ void flash_attn_ext_f16_iter_barrier() {
#if defined(AMD_WMMA_AVAILABLE)
    if constexpr (band) {
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
        return;
    }
#endif // defined(AMD_WMMA_AVAILABLE)
    __syncthreads();
}

// Fetch-ahead depth of the RDNA4 band: the number of K/V batches whose raw bytes are in flight in registers, 0 = no
// fetch-ahead (always 0 outside the fetch-ahead kernel variant, see ggml_cuda_fattn_mma_band_has_fetch_ahead).  The
// batches of an iteration (K first, then V) form a stream: when a batch is staged, the batch `depth` places later is
// fetched into the registers just freed.  Two batches where a thread fetches at most 2 chunks per batch (the 16-warp
// 64-column block); the first two batches of each iteration are then fetched at its start.  One batch where a thread
// fetches 4 (the 8-warp 32-column block has no registers for two: gfx1201 spills ~400 VGPRs); the stream then continues
// across the end of an iteration into the next one (fetched during the last V batch), so that the first batch of an
// iteration is not exposed either.  Measured best per row on gfx1201 (the 64-column block is slower with the
// cross-iteration stream: it keeps the next K live through the PV products).
template<int DKQ, int DV, int ncols1, int ncols2, int nwarps, bool V_is_K_view, bool use_sparse, bool fetch_ahead, bool band_wide>
static constexpr __device__ int flash_attn_ext_f16_band_depth() {
#if defined(AMD_WMMA_AVAILABLE) && defined(RDNA4)
    constexpr int ncols     = ncols1*ncols2;
    constexpr int nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa(DKQ, DV, ncols, ncols2, band_wide);
    constexpr int nbatch_K2 = ggml_cuda_fattn_mma_get_nbatch_K2(DKQ, DV, ncols, ncols2, band_wide);
    constexpr int nbatch_V2 = ggml_cuda_fattn_mma_get_nbatch_V2(DKQ, DV, ncols, ncols2, band_wide);
    constexpr int chunks    = flash_attn_ext_f16_band_batch<nwarps, nbatch_fa, nbatch_K2>::chunks_per_thread;
    if (!fetch_ahead || !ggml_cuda_fattn_mma_band_has_fetch_ahead(DKQ, DV, ncols1, ncols2) ||
            V_is_K_view || use_sparse || ggml_cuda_get_physical_warp_size() != 32 ||
            ggml_cuda_fattn_mma_get_nstages(DKQ, DV, ncols1, ncols2, use_sparse, band_wide) != 0 || nbatch_K2 != nbatch_V2 ||
            (DKQ/2) % nbatch_K2 != 0 || (DV/2) % nbatch_V2 != 0 || chunks > 4) {
        return 0;
    }
    return chunks <= 2 ? 2 : 1;
#else
    GGML_UNUSED_VARS(DKQ, DV, ncols1, ncols2, nwarps, V_is_K_view, use_sparse, fetch_ahead, band_wide);
    return 0;
#endif // defined(AMD_WMMA_AVAILABLE) && defined(RDNA4)
}

template<int DKQ, int DV, int ncols1, int ncols2, int nwarps,
    bool use_logit_softcap, bool V_is_K_view, bool use_sparse, bool needs_fixup, bool is_fixup, bool last_iter, bool oob_check,
    typename T_A_KQ, typename T_B_KQ, typename T_C_KQ, typename T_A_VKQ, typename T_B_VKQ, typename T_C_VKQ,
    bool band_fetch_ahead = false, bool band_wide = false>
static __device__ __forceinline__ void flash_attn_ext_f16_iter(
        const float2 * const __restrict__ Q_f2,
        const half2  * const __restrict__ K_h2,
        const half2  * const __restrict__ V_h2,
        const half   * const __restrict__ mask_h,
        const int32_t * const __restrict__ indices,
        float2       * const __restrict__ dstk,
        float2       * const __restrict__ dstk_fixup,
        const float scale,
        const float slope,
        const float logit_softcap,
        const uint3 ne01,
        const int ne02,
        const int stride_K,
        const int stride_V,
        const int stride_mask,
        half2        * const __restrict__ tile_Q,
        half2        * const __restrict__ tile_K,
        half2        * const __restrict__ tile_V,
        half         * const __restrict__ tile_mask,
        T_B_KQ       * const __restrict__ Q_B,
        T_C_VKQ      * const __restrict__ VKQ_C,
        float        * const __restrict__ KQ_max,
        float        * const __restrict__ KQ_rowsum,
        const int jt,
        const int kb0,
        const int k_VKQ_sup, const kq_derived_t derived, const fattn_kv_native_t kv_native,
        // RDNA4 band fetch-ahead (see flash_attn_ext_f16_band_depth): the caller's ring of fetched batches, which holds
        // the first batches of this iteration when band_on, and the next iteration's kb0 (-1: none).
        void * const band_ring_v = nullptr, const bool band_on = false, const int band_kb0_next = -1) {
#if defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
    constexpr int  warp_size       = ggml_cuda_get_physical_warp_size();
    constexpr int  ncols           = ncols1 * ncols2;
    constexpr int  cols_per_warp   = T_B_KQ::I;
    constexpr int  cols_per_thread = get_cols_per_thread();
    constexpr int  np              = cols_per_warp > ncols ? nwarps : nwarps * cols_per_warp/ncols; // Number of parallel CUDA warps per Q column.
    constexpr int  nbatch_fa       = ggml_cuda_fattn_mma_get_nbatch_fa(DKQ, DV, ncols, ncols2, band_wide);
    constexpr int  nbatch_K2       = ggml_cuda_fattn_mma_get_nbatch_K2(DKQ, DV, ncols, ncols2, band_wide);
    constexpr int  nbatch_V2       = ggml_cuda_fattn_mma_get_nbatch_V2(DKQ, DV, ncols, ncols2, band_wide);
    constexpr bool Q_in_reg        = ggml_cuda_fattn_mma_get_Q_in_reg (DKQ, DV, ncols, ncols2, band_wide) &&
        !(band_fetch_ahead && ggml_cuda_fattn_mma_band_fetch_ahead_Q_in_lds(DKQ, DV, ncols1, ncols2));
    constexpr int  nstages         = ggml_cuda_fattn_mma_get_nstages  (DKQ, DV, ncols1, ncols2, use_sparse, band_wide);

    constexpr bool swz = ggml_cuda_fattn_mma_get_swizzled(DKQ, DV, ncols1, ncols2, band_wide);
    constexpr int stride_tile_K = swz ? nbatch_K2 : nbatch_K2 + 4;
    constexpr int stride_tile_V = V_is_K_view ? stride_tile_K : (swz ? nbatch_V2 : nbatch_V2 + 4);

    // RDNA4 decode/verify band: q8_0 K/V are fetched ahead (see flash_attn_ext_f16_band_depth) and the iteration's
    // barriers only order shared memory.
    constexpr int  band_depth = flash_attn_ext_f16_band_depth<DKQ, DV, ncols1, ncols2, nwarps, V_is_K_view, use_sparse, band_fetch_ahead, band_wide>();
    constexpr bool band_fetch = band_depth > 0;
    constexpr int  n_band_K   = band_fetch ? (DKQ/2)/nbatch_K2 : 1;
    constexpr int  n_band_V   = band_fetch ? (DV/2)/nbatch_V2  : 1;
    static_assert(!band_fetch || !oob_check, "the band fetch-ahead assumes whole KV tiles");
    using band_batch_t = flash_attn_ext_f16_band_batch<nwarps, nbatch_fa, nbatch_K2>;
    constexpr bool band_cross = band_depth == 1; // the stream continues into the next iteration (see above)
    // Within one iteration the ring is this function's own; across iterations it is the caller's.
    band_batch_t   band_local[band_cross ? 1 : (band_depth > 0 ? band_depth : 1)];
    band_batch_t * const band_ring = band_cross ? (band_batch_t *) band_ring_v : band_local;
    const bool band_q8 = band_fetch && band_on;

    const int k_VKQ_0 = kb0 * nbatch_fa;
    if constexpr (band_fetch && !band_cross) {
        if (band_q8) {
#pragma unroll
            for (int t = 0; t < band_depth; ++t) {
                flash_attn_ext_f16_band_fetch<nwarps, nbatch_fa, nbatch_K2, false>
                    (band_ring[t], kv_native.K, kv_native.stride_K, 2*t*nbatch_K2, k_VKQ_0, nbatch_fa);
            }
        }
    }

    // Batch s of the iteration (K batches, then V batches) has been staged: fetch batch s + band_depth of the stream into
    // the ring slot it freed, from this iteration or from the next one.
    auto band_advance = [&](const int s) {
        if constexpr (band_fetch) {
            const int t  = s + band_depth;
            const int tb = t % (n_band_K + n_band_V);
            const int kv = t < n_band_K + n_band_V ? k_VKQ_0 : band_kb0_next*nbatch_fa;
            if (t >= n_band_K + n_band_V && (!band_cross || band_kb0_next < 0)) {
                return;
            }
            if (tb < n_band_K) {
                flash_attn_ext_f16_band_fetch<nwarps, nbatch_fa, nbatch_K2, false>
                    (band_ring[t % band_depth], kv_native.K, kv_native.stride_K, 2*tb*nbatch_K2, kv, nbatch_fa);
            } else {
                flash_attn_ext_f16_band_fetch<nwarps, nbatch_fa, nbatch_V2, false>
                    (band_ring[t % band_depth], kv_native.V, kv_native.stride_V, 2*(tb - n_band_K)*nbatch_V2, kv, nbatch_fa);
            }
        }
    };
#if defined(TURING_MMA_AVAILABLE)
    T_C_KQ KQ_C[nbatch_fa/(np*(cols_per_warp == 8 ? T_C_KQ::I : T_C_KQ::J))];
#elif defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
    T_C_KQ KQ_C[nbatch_fa/(np*T_C_KQ::J)];
#else // Volta
    T_C_KQ KQ_C[nbatch_fa/(np*T_C_KQ::J)];
#endif // defined(TURING_MMA_AVAILABLE)

    if constexpr (nstages > 1) {
        static_assert(!oob_check, "OOB check incompatible with multi-stage pipeline");
        static_assert(!V_is_K_view, "K data reuse not implemented multi-stage loading");
        static_assert(nbatch_K2 == DKQ/2, "batching not implemented for multi stage loading");
        constexpr bool use_cp_async = true;
        cp_async_wait_all();
        __syncthreads();
        flash_attn_ext_f16_load_tile<stride_tile_V, swz, nwarps, nbatch_fa, use_cp_async, oob_check, use_sparse>
            (V_h2, tile_V, nbatch_V2, stride_V, k_VKQ_0, k_VKQ_sup, nullptr,
             kv_native.V, kv_native.stride_V, 0, kv_native.type_V);
    } else {
        // the sparse mask values are gathered per element, always load them synchronously
        constexpr bool use_cp_async = nstages == 1 && !use_sparse;
        if (ncols2 > 1 || mask_h || derived.cell_pos != nullptr) {
            flash_attn_ext_f16_load_mask<ncols1, nwarps, nbatch_fa, use_cp_async, oob_check, use_sparse>
                (mask_h, tile_mask, stride_mask, k_VKQ_0, k_VKQ_sup, jt*ncols1, ne01, indices, derived);
        }
    }

    // For MLA K and V have the same data.
    // Therefore, iterate over K in reverse and later re-use the data if possible.
    // A 256-dim non-MLA head that is loaded in several K batches (the RDNA4 band row) iterates forward
    // instead, so the head-dimension accumulation order (and the result) is the same as with one
    // full-width batch.
    constexpr bool K_forward = !V_is_K_view && DKQ == 256;
    constexpr int  K_last    = (DKQ/2-1) - (DKQ/2-1) % nbatch_K2;
#pragma unroll
    for (int k0_iter = 0; k0_iter <= K_last; k0_iter += nbatch_K2) {
        const int k0_start = K_forward ? k0_iter : K_last - k0_iter;
        const int k0_stop = k0_start + nbatch_K2 < DKQ/2 ? k0_start + nbatch_K2 : DKQ/2;

        if constexpr (nstages <= 1) {
            const int k0_diff = k0_stop - k0_start;
            constexpr bool use_cp_async = nstages == 1;
            bool staged = false;
            if constexpr (band_fetch) {
                if (band_q8) {
                    const int s = k0_iter / nbatch_K2;
                    flash_attn_ext_f16_band_stage<stride_tile_K, nwarps, nbatch_fa, nbatch_K2>(band_ring[s % band_depth], tile_K);
                    band_advance(s);
                    staged = true;
                }
            }
            if (!staged) {
                flash_attn_ext_f16_load_tile<stride_tile_K, swz, nwarps, nbatch_fa, use_cp_async, oob_check, use_sparse>
                    (K_h2 + k0_start, tile_K, k0_diff, stride_K, k_VKQ_0, k_VKQ_sup, indices,
                     kv_native.K, kv_native.stride_K, 2*k0_start, kv_native.type_K);
                if (use_cp_async) {
                    cp_async_wait_all();
                }
            }
            flash_attn_ext_f16_iter_barrier<band_fetch>();
        }

        // Calculate tile of KQ:
        if constexpr (Q_in_reg) {
#pragma unroll
            for (int i_KQ_00 = 0; i_KQ_00 < nbatch_fa; i_KQ_00 += np*T_A_KQ::I) {
                const int i_KQ_0 = i_KQ_00 + (threadIdx.y % np)*T_A_KQ::I;
#pragma unroll
                for (int k_KQ_0 = k0_start; k_KQ_0 < k0_stop; k_KQ_0 += T_A_KQ::J) {
                    T_A_KQ K_A;
                    load_ldmatrix_swizzled<stride_tile_K>(K_A, tile_K, i_KQ_0*stride_tile_K + k_KQ_0-k0_start);
                    if constexpr (cols_per_warp == 8) {
                        mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], K_A, Q_B[k_KQ_0/T_A_KQ::J]);
                    } else {
                        // Wide version of KQ_C is column-major
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                        // AMD matrix C is column-major.
                        mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], K_A, Q_B[k_KQ_0/T_A_KQ::J]);
#else
                        // swap A and B for CUDA.
                        mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], Q_B[k_KQ_0/T_A_KQ::J], K_A);
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    }
                }
            }
        } else {
            constexpr int stride_tile_Q = DKQ/2 + 4;
#pragma unroll
            for (int k_KQ_0 = k0_start; k_KQ_0 < k0_stop; k_KQ_0 += T_A_KQ::J) {
                load_ldmatrix(Q_B[0], tile_Q + (threadIdx.y / np)*(T_B_KQ::I*stride_tile_Q) + k_KQ_0, stride_tile_Q);

#pragma unroll
                for (int i_KQ_00 = 0; i_KQ_00 < nbatch_fa; i_KQ_00 += np*T_A_KQ::I) {
                    const int i_KQ_0 = i_KQ_00 + (threadIdx.y % np)*T_A_KQ::I;

                    T_A_KQ K_A;
                    load_ldmatrix_swizzled<stride_tile_K>(K_A, tile_K, i_KQ_0*stride_tile_K + k_KQ_0-k0_start);

                    if constexpr (cols_per_warp == 8) {
                        mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], K_A, Q_B[0]);
                    } else {
                        // Wide version of KQ_C is column-major
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                        // AMD matrix C is column-major.
                        mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], K_A, Q_B[0]);
#else
                        // swap A and B for CUDA.
                        mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], Q_B[0], K_A);
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    }
                }
            }
        }

        if constexpr (nstages <= 1) {
            flash_attn_ext_f16_iter_barrier<band_fetch>(); // Only needed if tile_K == tile_V.
        }
    }

    if (use_logit_softcap) {
        constexpr int stride = cols_per_warp == 8 ? np*T_C_KQ::I : np*T_C_KQ::J;
        static_assert(nbatch_fa % stride == 0, "bad loop size");
#pragma unroll
        for (int i = 0; i < nbatch_fa/stride; ++i) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                KQ_C[i].x[l] = logit_softcap*tanhf(KQ_C[i].x[l]);
            }
        }
    }

    float KQ_max_new[cols_per_thread];
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_max_new[col] = KQ_max[col];
    }
    float KQ_rowsum_add[cols_per_thread] = {0.0f};

    if constexpr (cols_per_warp == 8) {
        if (ncols2 > 1 || mask_h || derived.cell_pos != nullptr) {
#pragma unroll
            for (int i00 = 0; i00 < nbatch_fa; i00 += np*T_C_KQ::I) {
                const int i0 = i00 + (threadIdx.y % np)*T_C_KQ::I;
#pragma unroll
                for (int l = 0; l < T_C_KQ::ne; ++l) {
                    const int i = i0 + T_C_KQ::get_i(l);
                    const int j = ((threadIdx.y / np)*T_C_KQ::J + T_C_KQ::get_j(l)) / ncols2;

                    KQ_C[i00/(np*T_C_KQ::I)].x[l] += slope * __half2float(tile_mask[j*(nbatch_fa + 8) + i]);
                }
            }
        }

        // Calculate softmax for each KQ column using the current max. value.
        // The divisor is stored in KQ_rowsum and will be applied at the end.
        static_assert(nbatch_fa % (np*T_C_KQ::I) == 0, "bad loop size");
#pragma unroll
        for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::I) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                if (!oob_check || k0 + (threadIdx.y % np)*T_C_KQ::I + T_C_KQ::get_i(l) < k_VKQ_sup) {
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    constexpr int KQ_idx = 0;
#else
                    // Turing + Volta:
                    const int KQ_idx = l % 2;
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    KQ_max_new[KQ_idx] = fmaxf(KQ_max_new[KQ_idx], KQ_C[k0/(np*T_C_KQ::I)].x[l] + FATTN_KQ_MAX_OFFSET);
                }
            }
        }

        // Values per KQ column are spread across 8 threads:
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
#pragma unroll
            for (int offset = 16; offset >= 4; offset >>= 1) {
                KQ_max_new[col] = fmaxf(KQ_max_new[col], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[col], offset, warp_size));
            }
        }

        static_assert(nbatch_fa % (np*T_C_KQ::I) == 0, "bad loop size");
#pragma unroll
        for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::I) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                if (!oob_check || k0 + (threadIdx.y % np)*T_C_KQ::I + T_C_KQ::get_i(l) < k_VKQ_sup) {
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    constexpr int KQ_idx = 0;
#else
                    // Turing + Volta:
                    const int KQ_idx = l % 2;
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    KQ_C[k0/(np*T_C_KQ::I)].x[l] = expf(KQ_C[k0/(np*T_C_KQ::I)].x[l] - KQ_max_new[KQ_idx]);
                    KQ_rowsum_add[KQ_idx] += KQ_C[k0/(np*T_C_KQ::I)].x[l];
                } else {
                    KQ_C[k0/(np*T_C_KQ::I)].x[l] = 0.0f;
                }
            }
        }
    } else { // not Turing mma or T_B_KQ::I > 8
        if (ncols2 > 1 || mask_h || derived.cell_pos != nullptr) {
#pragma unroll
            for (int i00 = 0; i00 < nbatch_fa; i00 += np*T_C_KQ::J) {
                const int i0 = i00 + (threadIdx.y % np)*T_C_KQ::J;

                // The mask is stored as 16 bit half values, loading them as 32 bit half2 values is preferred in terms of speed.
                // However, this is not possible for RDNA3 where 2 consecutive l indices are not consecutive in the mask memory layout.
#ifdef RDNA3
#pragma unroll
                for (int l = 0; l < T_C_KQ::ne; ++l) {
                    const int i = i0 + T_C_KQ::get_j(l);
                    const int j = ((threadIdx.y / np)*cols_per_warp + T_C_KQ::get_i(l)) / ncols2;

                    KQ_C[i00/(np*T_C_KQ::J)].x[l] += __half2float(tile_mask[j*(nbatch_fa + 8) + i]);
                }
#else
#pragma unroll
                for (int l0 = 0; l0 < T_C_KQ::ne; l0 += 2) {
                    const int i = (i0 + T_C_KQ::get_j(l0)) / 2;
                    const int j = ((threadIdx.y / np)*cols_per_warp + T_C_KQ::get_i(l0)) / ncols2;

                    const float2 tmp = __half22float2(((const half2 *)tile_mask)[j*(nbatch_fa/2 + 4) + i]);
                    KQ_C[i00/(np*T_C_KQ::J)].x[l0 + 0] += slope*tmp.x;
                    KQ_C[i00/(np*T_C_KQ::J)].x[l0 + 1] += slope*tmp.y;
                }
#endif // RDNA3
            }
        }

        // Calculate softmax for each KQ column using the current max. value.
        // The divisor is stored in KQ_rowsum and will be applied at the end.
        static_assert(nbatch_fa % (np*T_C_KQ::J) == 0, "bad loop size");
#pragma unroll
        for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::J) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                if (!oob_check || k0 + (threadIdx.y % np)*T_C_KQ::J + T_C_KQ::get_j(l) < k_VKQ_sup) {
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    constexpr int KQ_idx = 0;
#else
                    // Turing + Volta:
                    const int KQ_idx = (l/2) % 2;
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    KQ_max_new[KQ_idx] = fmaxf(KQ_max_new[KQ_idx], KQ_C[(k0/(np*T_C_KQ::J))].x[l] + FATTN_KQ_MAX_OFFSET);
                }
            }
        }

#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
#if defined(TURING_MMA_AVAILABLE)
            // Values per KQ column are spread across 4 threads:
            constexpr int offset_first = 2;
            constexpr int offset_last  = 1;
#elif defined(AMD_MFMA_AVAILABLE)
            // MFMA: 4 threads per Q column (threadIdx.x % 16 == col, spaced by 16).
            constexpr int offset_first = 32;
            constexpr int offset_last  = 16;
#elif defined(AMD_WMMA_AVAILABLE)
            // Values per KQ column are spread across 2 threads:
            constexpr int offset_first = 16;
            constexpr int offset_last  = 16;
#else // Volta
            // Values per KQ column are spread across 2 threads:
            constexpr int offset_first = 2;
            constexpr int offset_last  = 2;
#endif // defined(TURING_MMA_AVAILABLE)
#pragma unroll
            for (int offset = offset_first; offset >= offset_last; offset >>= 1) {
                KQ_max_new[col] = fmaxf(KQ_max_new[col], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[col], offset, warp_size));
            }
        }

        static_assert(nbatch_fa % (np*T_C_KQ::J) == 0, "bad loop size");
#pragma unroll
        for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::J) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                if (!oob_check || k0 + (threadIdx.y % np)*T_C_KQ::J + T_C_KQ::get_j(l) < k_VKQ_sup) {
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    constexpr int KQ_idx = 0;
#else
                    // Turing + Volta:
                    const int KQ_idx = (l/2) % 2;
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    KQ_C[(k0/(np*T_C_KQ::J))].x[l] = expf(KQ_C[(k0/(np*T_C_KQ::J))].x[l] - KQ_max_new[KQ_idx]);
                    KQ_rowsum_add[KQ_idx] += KQ_C[(k0/(np*T_C_KQ::J))].x[l];
                } else {
                    KQ_C[(k0/(np*T_C_KQ::J))].x[l] = 0.0f;
                }
            }
        }
    }

    {
        float KQ_max_scale[cols_per_thread];
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            const float KQ_max_diff = KQ_max[col] - KQ_max_new[col];
            KQ_max_scale[col] = expf(KQ_max_diff);
            KQ_max[col] = KQ_max_new[col];

            *((uint32_t *) &KQ_max_scale[col]) *= KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD;

            // Scale previous KQ_rowsum to account for a potential increase in KQ_max:
            KQ_rowsum[col] = KQ_max_scale[col]*KQ_rowsum[col] + KQ_rowsum_add[col];
        }

#if defined(TURING_MMA_AVAILABLE)
        if constexpr (cols_per_warp == 8) {
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[0], KQ_max_scale[cols_per_thread - 1]);
#pragma unroll
            for (int i = 0; i < DV/T_C_VKQ::I; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale_h2;
                }
            }
        } else {
#pragma unroll
            for (int col = 0; col < cols_per_thread; ++col) {
                const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[col], KQ_max_scale[col]);
#pragma unroll
                for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
                    for (int l0 = 0; l0 < T_C_VKQ::ne; l0 += 2) {
                        VKQ_C[i].x[l0 + col] *= KQ_max_scale_h2;
                    }
                }
            }
        }
#elif defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
        if constexpr (std::is_same_v<decltype(T_C_VKQ::x), half2[T_C_VKQ::ne]>) {
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[0], KQ_max_scale[0]);
#pragma unroll
            for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale_h2;
                }
            }
        } else {
            static_assert(std::is_same_v<decltype(T_C_VKQ::x), float[T_C_VKQ::ne]>, "bad VKQ type");
#pragma unroll
            for (int i = 0; i < DV/T_C_VKQ::J; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale[0];
                }
            }
        }
#else // Volta
        const half2 KQ_max_scale_h2 = make_half2(
            KQ_max_scale[(threadIdx.x / 2) % 2], KQ_max_scale[(threadIdx.x / 2) % 2]);
#pragma unroll
        for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
            for (int l = 0; l < T_C_VKQ::ne; ++l) {
                VKQ_C[i].x[l] *= KQ_max_scale_h2;
            }
        }
#endif // defined(TURING_MMA_AVAILABLE)
    }

    // Convert KQ C tiles into B tiles for VKQ calculation:
    T_B_VKQ B[nbatch_fa/(np*2*T_B_VKQ::J)];
    static_assert(nbatch_fa % (np*2*T_B_VKQ::J) == 0, "bad loop size");
    if constexpr (cols_per_warp == 8) {
#pragma unroll
        for (int k = 0; k < nbatch_fa/(np*2*T_B_VKQ::J); ++k) {
            B[k] = get_transposed(get_half2(KQ_C[k]));
        }
    } else {
        for (int k = 0; k < nbatch_fa/(np*2*T_B_VKQ::J); ++k) {
            B[k] = get_half2(KQ_C[k]);
        }
    }

    if constexpr (nstages > 1) {
        static_assert(!use_sparse, "sparse gather not implemented for multi-stage loading");
        static_assert(!V_is_K_view, "K data reuse not implemented multi-stage loading");
        // Preload K tile for next iteration:
        constexpr bool use_cp_async = true;
        cp_async_wait_all();
        __syncthreads();
        if (!last_iter) {
            if (ncols2 > 1 || mask_h || derived.cell_pos != nullptr) {
                flash_attn_ext_f16_load_mask<ncols1, nwarps, nbatch_fa, use_cp_async, oob_check, use_sparse>
                    (mask_h, tile_mask, stride_mask, k_VKQ_0 + nbatch_fa, k_VKQ_sup, jt*ncols1, ne01, nullptr, derived);
            }
            flash_attn_ext_f16_load_tile<stride_tile_K, swz, nwarps, nbatch_fa, use_cp_async, oob_check, use_sparse>
                (K_h2, tile_K, nbatch_K2, stride_K, k_VKQ_0 + nbatch_fa, k_VKQ_sup, nullptr,
                 kv_native.K, kv_native.stride_K, 0, kv_native.type_K);
        }
    }


    // Calculate VKQ tile, need to use logical rather than physical elements for i0 due to transposition of V:
#pragma unroll
    for (int i0_start = 0; i0_start < DV; i0_start += 2*nbatch_V2) {
        static_assert(DV % (2*nbatch_V2) == 0, "bad loop size");
        const int i0_stop = i0_start + 2*nbatch_V2;

        if constexpr (nstages <= 1) {
            const int i0_diff = i0_stop - i0_start;
            if (!V_is_K_view || i0_stop > 2*nbatch_K2) {
                constexpr bool use_cp_async = nstages == 1;
                bool staged = false;
                if constexpr (band_fetch) {
                    if (band_q8) {
                        const int s = n_band_K + i0_start/(2*nbatch_V2);
                        flash_attn_ext_f16_band_stage<stride_tile_V, nwarps, nbatch_fa, nbatch_V2>(band_ring[s % band_depth], tile_V);
                        band_advance(s);
                        staged = true;
                    }
                }
                if (!staged) {
                    flash_attn_ext_f16_load_tile<stride_tile_V, swz, nwarps, nbatch_fa, use_cp_async, oob_check, use_sparse>
                        (V_h2 + i0_start/2, tile_V, i0_diff/2, stride_V, k_VKQ_0, k_VKQ_sup, indices,
                         kv_native.V, kv_native.stride_V, i0_start, kv_native.type_V);
                    if (use_cp_async) {
                        cp_async_wait_all();
                    }
                }
                flash_attn_ext_f16_iter_barrier<band_fetch>();

                // gfx11 signed-zero leak: a KV row that is masked (P == +0.0) for every query
                // column of this block still feeds its V to the WMMA, where the masked cell's
                // x + (-0.0) products are accumulated inexactly on gfx11 and leak the cell's
                // (possibly stale/freed) content into the output. Never let such a row's V
                // reach the multiply: zero it after it is staged, using the mask tile that is
                // already in shared memory. Only the plain (non-swizzled) layout is handled;
                // the swizzled layout is NVIDIA-only where the hardware does not exhibit the
                // leak. V_is_K_view (MLA) reuses K data as V and needs the K tile zeroed
                // instead - not implemented here (RDNA caps route those heads to the tile kernel).
                if constexpr (!V_is_K_view && !swz) {
                    if (ncols2 > 1 || mask_h || derived.cell_pos != nullptr) { // tile_mask is valid iff the mask-add ran
                        const int tid = threadIdx.y * warp_size + threadIdx.x;
#pragma unroll
                        for (int i = tid; i < nbatch_fa; i += nwarps*warp_size) {
                            bool dead = true;
#pragma unroll
                            for (int j = 0; j < ncols1; ++j) {
                                dead = dead && (__half2float(tile_mask[j*(nbatch_fa + 8) + i]) <= -1e30f);
                            }
                            if (dead) {
                                half2 * const row = tile_V + i*stride_tile_V;
                                for (int k = 0; k < i0_diff/2; ++k) {
                                    row[k] = make_half2(0.0f, 0.0f);
                                }
                            }
                        }
                        flash_attn_ext_f16_iter_barrier<band_fetch>();
                    }
                }
            }
        }
        const int tile_V_offset_i = !V_is_K_view || i0_stop > 2*nbatch_K2 ? 0 : i0_start/2;

#if defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
#pragma unroll
        for (int i_VKQ_0 = i0_start; i_VKQ_0 < i0_stop; i_VKQ_0 += T_A_VKQ::I) {
            static_assert((nbatch_fa/2) % (np*T_A_VKQ::J) == 0, "bad loop size");
#pragma unroll
            for (int k00 = 0; k00 < nbatch_fa/2; k00 += np*T_A_VKQ::J) {
                const int k0 = k00 + (threadIdx.y % np)*T_A_VKQ::J;

                T_A_VKQ A; // Transposed in SRAM but not in registers, gets transposed on load.
                load_ldmatrix_trans_swizzled<stride_tile_V>(A, tile_V, tile_V_offset_i + 2*k0*stride_tile_V + (i_VKQ_0 - i0_start)/2);
                if constexpr (T_B_KQ::I == 8) {
                    mma(VKQ_C[i_VKQ_0/T_A_VKQ::I], A, B[k00/(np*T_A_VKQ::J)]);
                } else {
                    // Wide version of VKQ_C is column-major.
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                    // AMD matrix C is column-major.
                    mma(VKQ_C[i_VKQ_0/T_A_VKQ::I], A, B[k00/(np*T_A_VKQ::J)]);
#else
                    // swap A and B for CUDA.
                    mma(VKQ_C[i_VKQ_0/T_A_VKQ::I], B[k00/(np*T_A_VKQ::J)], A);
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
                }
            }
        }
#else // Volta
        constexpr int i0_stride = 2*T_C_VKQ::J;
#pragma unroll
        for (int i_VKQ_0 = i0_start; i_VKQ_0 < i0_stop; i_VKQ_0 += i0_stride) {
            static_assert(nbatch_fa % (np*T_A_VKQ::I) == 0, "bad loop size");
            static_assert(2*T_B_VKQ::J == T_A_VKQ::I, "bad tile sizes");
#pragma unroll
            for (int k00 = 0; k00 < nbatch_fa; k00 += np*T_A_VKQ::I) {
                const int k0 = k00 + (threadIdx.y % np)*T_A_VKQ::I;

                T_A_VKQ A; // Transposed in both SRAM and registers, load normally.
                load_ldmatrix_swizzled<stride_tile_V>(A, tile_V, tile_V_offset_i + k0*stride_tile_V + (i_VKQ_0 - i0_start)/2);
                mma(VKQ_C[i_VKQ_0/i0_stride], B[k00/(np*T_A_VKQ::I)], A);
            }
        }
#endif // defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)

        if constexpr (nstages <= 1) {
            flash_attn_ext_f16_iter_barrier<band_fetch>(); // Only needed if tile_K == tile_V.
        }
    }
#else
    GGML_UNUSED_VARS(Q_f2, K_h2, V_h2, mask_h, indices, dstk, dstk_fixup,
        scale, slope, logit_softcap, ne01, ne02,
        stride_K, stride_V, stride_mask,
        tile_Q, tile_K, tile_V, tile_mask,
        Q_B, VKQ_C, KQ_max, KQ_rowsum, kb0, derived);
    NO_DEVICE_CODE;
#endif // defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
}

#if defined(TURING_MMA_AVAILABLE)
template<int DKQ, int ncols> struct mma_tile_sizes {
    using T_A_KQ  = tile<16,  8, half2>; // row-major
    using T_B_KQ  = tile<16,  8, half2>; // column-major
    using T_C_KQ  = tile<16, 16, float>; // column-major
    using T_A_VKQ = tile<16,  8, half2>; // row-major
    using T_B_VKQ = tile<16,  8, half2>; // column-major
    using T_C_VKQ = tile<16,  8, half2>; // column-major
};
// If there are only 8 columns, use thinner B tiles to avoid wasting compute:
template<int DKQ> struct mma_tile_sizes<DKQ, 8> {
    using T_A_KQ  = tile<16,  8, half2>; // row-major
    using T_B_KQ  = tile< 8,  8, half2>; // column-major
    using T_C_KQ  = tile<16,  8, float>; // row-major
    using T_A_VKQ = tile<16,  8, half2>; // row-major
    using T_B_VKQ = tile< 8,  8, half2>; // column-major
    using T_C_VKQ = tile<16,  4, half2>; // row-major
};
// For very large head sizes, use thinner B tiles to reduce register pressure:
template<> struct mma_tile_sizes<256, 16> {
    using T_A_KQ  = tile<16,  8, half2>; // row-major
    using T_B_KQ  = tile< 8,  8, half2>; // column-major
    using T_C_KQ  = tile<16,  8, float>; // row-major
    using T_A_VKQ = tile<16,  8, half2>; // row-major
    using T_B_VKQ = tile< 8,  8, half2>; // column-major
    using T_C_VKQ = tile<16,  4, half2>; // row-major
};
template<> struct mma_tile_sizes<512, 16> {
    using T_A_KQ  = tile<16,  8, half2>; // row-major
    using T_B_KQ  = tile< 8,  8, half2>; // column-major
    using T_C_KQ  = tile<16,  8, float>; // row-major
    using T_A_VKQ = tile<16,  8, half2>; // row-major
    using T_B_VKQ = tile< 8,  8, half2>; // column-major
    using T_C_VKQ = tile<16,  4, half2>; // row-major
};
template<> struct mma_tile_sizes<512, 32> {
    using T_A_KQ  = tile<16,  8, half2>; // row-major
    using T_B_KQ  = tile< 8,  8, half2>; // column-major
    using T_C_KQ  = tile<16,  8, float>; // row-major
    using T_A_VKQ = tile<16,  8, half2>; // row-major
    using T_B_VKQ = tile< 8,  8, half2>; // column-major
    using T_C_VKQ = tile<16,  4, half2>; // row-major
};
#elif defined(AMD_WMMA_AVAILABLE)
#ifdef RDNA3
template<int DKQ, int ncols> struct mma_tile_sizes {
    using T_A_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
    using T_B_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // column-major
    using T_C_KQ  = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;          // column-major
    using T_A_VKQ = tile<32,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
    using T_B_VKQ = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // column-major
    using T_C_VKQ = tile<16, 16, half2, DATA_LAYOUT_I_MAJOR>;          // column-major
};
template<int ncols> struct mma_tile_sizes<80, ncols> {
    using T_A_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
    using T_B_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // column-major
    using T_C_KQ  = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;          // column-major
    using T_A_VKQ = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
    using T_B_VKQ = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // column-major
    using T_C_VKQ = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;          // column-major
};
template<int ncols> struct mma_tile_sizes<112, ncols> {
    using T_A_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
    using T_B_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // column-major
    using T_C_KQ  = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;          // column-major
    using T_A_VKQ = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
    using T_B_VKQ = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // column-major
    using T_C_VKQ = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;          // column-major
};
#else
template<int DKQ, int ncols> struct mma_tile_sizes {
    using T_A_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR>;           // row-major
    using T_B_KQ  = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR>;           // column-major
    using T_C_KQ  = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;           // column-major
    using T_A_VKQ = tile<32,  8, half2, DATA_LAYOUT_I_MAJOR>;           // row-major
    using T_B_VKQ = tile<16,  8, half2, DATA_LAYOUT_I_MAJOR>;           // column-major
    using T_C_VKQ = tile<16, 16, half2, DATA_LAYOUT_I_MAJOR_SCRAMBLED>; // column-major
};
template<int ncols> struct mma_tile_sizes<80, ncols> {
    using T_A_KQ  = tile<16,  8, half2>; // row-major
    using T_B_KQ  = tile<16,  8, half2>; // column-major
    using T_C_KQ  = tile<16, 16, float>; // column-major
    using T_A_VKQ = tile<16,  8, half2>; // row-major
    using T_B_VKQ = tile<16,  8, half2>; // column-major
    using T_C_VKQ = tile<16,  8, half2>; // column-major
};
template<int ncols> struct mma_tile_sizes<112, ncols> {
    using T_A_KQ  = tile<16,  8, half2>; // row-major
    using T_B_KQ  = tile<16,  8, half2>; // column-major
    using T_C_KQ  = tile<16, 16, float>; // column-major
    using T_A_VKQ = tile<16,  8, half2>; // row-major
    using T_B_VKQ = tile<16,  8, half2>; // column-major
    using T_C_VKQ = tile<16,  8, half2>; // column-major
};
#endif // RDNA3
#elif defined(AMD_MFMA_AVAILABLE)
template<int DKQ, int ncols> struct mma_tile_sizes {
    using T_A_KQ  = tile<16,  8, half2>; // row-major
    using T_B_KQ  = tile<16,  8, half2>; // column-major
    using T_C_KQ  = tile<16, 16, float>; // column-major
    using T_A_VKQ = tile<16,  8, half2>; // row-major
    using T_B_VKQ = tile<16,  8, half2>; // column-major
    using T_C_VKQ = tile<16, 16, float>; // column-major
};
#else // Volta
template<int DKQ, int ncols> struct mma_tile_sizes {
    using T_A_KQ  = tile< 8,  4, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
    using T_B_KQ  = tile<32,  4, half2, DATA_LAYOUT_I_MAJOR>;          // column-major
    using T_C_KQ  = tile<32,  8, float, DATA_LAYOUT_I_MAJOR>;          // column-major
    using T_A_VKQ = tile< 8,  4, half2, DATA_LAYOUT_J_MAJOR_MIRRORED>; // column-major
    using T_B_VKQ = tile<32,  4, half2, DATA_LAYOUT_I_MAJOR>;          // column-major
    using T_C_VKQ = tile<32,  4, half2, DATA_LAYOUT_I_MAJOR>;          // column-major
};
#endif // defined(TURING_MMA_AVAILABLE)

template<int DKQ, int DV, int ncols1, int ncols2, int nwarps, bool use_logit_softcap, bool V_is_K_view, bool use_sparse, bool needs_fixup, bool is_fixup,
    bool band_fetch_ahead = false, bool band_wide = false>
static __device__ __forceinline__ void flash_attn_ext_f16_process_tile(
        const float2 * const __restrict__ Q_f2,
        const half2  * const __restrict__ K_h2,
        const half2  * const __restrict__ V_h2,
        const half   * const __restrict__ mask_h,
        const int32_t * const __restrict__ indices,
        const uint32_t * const __restrict__ kq_blocks,
        const float  * const __restrict__ sinks_f,
        float2       * const __restrict__ dstk,
        float2       * const __restrict__ dstk_fixup,
        const float scale,
        const float slope,
        const float logit_softcap,
        const uint3 ne01,
        const int ne02,
        const int gqa_ratio,
        const int ne11,
        const int stride_Q1,
        const int stride_Q2,
        const int stride_K,
        const int stride_V,
        const int stride_mask,
        const int jt,
        const int zt_gqa,
        const int kb0_start,
        const int kb0_stop, const kq_derived_t derived, const fattn_kv_native_t kv_native,
        // wip/fa-stage-chunking: element-major (max, rowsum) target for the PARTIAL contract; non-null
        // means this launch publishes partials (no normalize, and the no-seam case must publish meta)
        // -- see the kernel signature.  `kv_start` is this launch's first KV row in the whole op's
        // cell space (0 unless KV-chunked).
        float2 * const meta_kv, const int kv_start,
        // > 1 only in the decode/verify band, where the block processes the KV iterations
        // kb0_start, kb0_start + kb0_step, ... (< kb0_stop) and may own none of them.
        const int kb0_step = 1) {
#if defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int ncols = ncols1 * ncols2;
    using     T_A_KQ    = typename mma_tile_sizes<DKQ, ncols>::T_A_KQ;
    using     T_B_KQ    = typename mma_tile_sizes<DKQ, ncols>::T_B_KQ;
    using     T_C_KQ    = typename mma_tile_sizes<DKQ, ncols>::T_C_KQ;
    using     T_A_VKQ   = typename mma_tile_sizes<DKQ, ncols>::T_A_VKQ;
    using     T_B_VKQ   = typename mma_tile_sizes<DKQ, ncols>::T_B_VKQ;
    using     T_C_VKQ   = typename mma_tile_sizes<DKQ, ncols>::T_C_VKQ;

    constexpr int  cols_per_warp   = T_B_KQ::I;
    constexpr int  cols_per_thread = get_cols_per_thread();
    constexpr int  np              = cols_per_warp > ncols ? nwarps : nwarps * cols_per_warp/ncols; // Number of parallel CUDA warps per Q column.
    constexpr int  nbatch_fa       = ggml_cuda_fattn_mma_get_nbatch_fa     (DKQ, DV, ncols, ncols2, band_wide);
    constexpr int  nbatch_K2       = ggml_cuda_fattn_mma_get_nbatch_K2     (DKQ, DV, ncols, ncols2, band_wide);
    constexpr int  nbatch_V2       = ggml_cuda_fattn_mma_get_nbatch_V2     (DKQ, DV, ncols, ncols2, band_wide);
    constexpr int  nbatch_combine  = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols, ncols2, band_wide);
    constexpr bool Q_in_reg        = ggml_cuda_fattn_mma_get_Q_in_reg      (DKQ, DV, ncols, ncols2, band_wide) &&
        !(band_fetch_ahead && ggml_cuda_fattn_mma_band_fetch_ahead_Q_in_lds(DKQ, DV, ncols1, ncols2));
    constexpr int  nstages         = ggml_cuda_fattn_mma_get_nstages       (DKQ, DV, ncols1, ncols2, use_sparse, band_wide);

    if (cols_per_warp > ncols) {
        NO_DEVICE_CODE;
        return;
    }

    static_assert(nwarps * (cols_per_warp/ncols2) % ncols1 == 0, "bad nwarps");

    constexpr bool swz = ggml_cuda_fattn_mma_get_swizzled(DKQ, DV, ncols1, ncols2, band_wide);
    constexpr int stride_tile_Q = DKQ/2 + 4;
    constexpr int stride_tile_K = swz ? nbatch_K2 : nbatch_K2 + 4;
    constexpr int stride_tile_V = V_is_K_view ? stride_tile_K : (swz ? nbatch_V2 : nbatch_V2 + 4);
    constexpr int stride_tile_KV_max = stride_tile_K > stride_tile_V ? stride_tile_K : stride_tile_V;

    extern __shared__ half2 tile_Q[];
    half2 * tile_K    = Q_in_reg              ? tile_Q                             : tile_Q + ncols     * stride_tile_Q;
    half2 * tile_V    =           nstages > 1 ? tile_K + nbatch_fa * stride_tile_K : tile_K;
    half  * tile_mask = (half *) (nstages > 1 ? tile_V + nbatch_fa * stride_tile_V : tile_V + nbatch_fa * stride_tile_KV_max);

    T_B_KQ    Q_B[(Q_in_reg ? DKQ/(2*T_B_KQ::J) : 1)];
#if defined(TURING_MMA_AVAILABLE)
    T_C_VKQ VKQ_C[cols_per_warp == 8 ? DV/T_C_VKQ::I : DV/(2*T_C_VKQ::J)];
#elif defined(AMD_WMMA_AVAILABLE) && defined(RDNA3)
    // An fp32 C tile spans T_C_VKQ::J output columns, an fp16 one spans 2*T_C_VKQ::J (it is written by
    // two wmma ops, one per half).  Sizing on DV % 32 alone is only right for the fp32 rows whose DV is
    // not a multiple of 32 (head 80/112); an fp32 tile at DV % 32 == 0 needs DV/T_C_VKQ::J, which is also
    // the bound the fp32 rescale and combine loops use.  Identical to the old expression for every fp16
    // row, i.e. for every row the delivery instantiates (GREEDY-PURITY.md §46).
    T_C_VKQ VKQ_C[std::is_same_v<decltype(T_C_VKQ::x), float[T_C_VKQ::ne]> ? DV/T_C_VKQ::J :
                  (DV % 32 != 0 ? DV/T_C_VKQ::J : DV/(2*T_C_VKQ::J))];
#elif defined(AMD_MFMA_AVAILABLE)
    T_C_VKQ VKQ_C[                                     DV/T_C_VKQ::J];
#elif defined(AMD_WMMA_AVAILABLE)
    T_C_VKQ VKQ_C[                                     DV/(2*T_C_VKQ::J)];
#else // Volta
    T_C_VKQ VKQ_C[                                     DV/(2*T_C_VKQ::J)];
#endif // defined(TURING_MMA_AVAILABLE)

    float KQ_rowsum[cols_per_thread] = {0.0f};
    float KQ_max[cols_per_thread];
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_max[col] = -FLT_MAX/2.0f;
    }

    // Load Q data into tile_Q, either temporarily or permanently.
    // Q in registers is faster, but register pressure is the biggest bottleneck.
    // The loading is done with decreasing granularity for D for better memory bandwidth.
    const half2 scale_h2 = make_half2(scale, scale);
#pragma unroll
    for (int stride_k : {warp_size, warp_size/2, warp_size/4, warp_size/8}) {
        const int k0_start  = stride_k == warp_size ? 0 : DKQ/2 - (DKQ/2) % (2*stride_k);
        const int k0_stop   =                             DKQ/2 - (DKQ/2) % (1*stride_k);
        const int stride_jc = warp_size / stride_k;

        if (k0_start == k0_stop) {
            continue;
        }

#pragma unroll
        for (int jc0 = 0; jc0 < ncols; jc0 += nwarps*stride_jc) {
            const int jc = jc0 + threadIdx.y*stride_jc + (stride_k == warp_size ? 0 : threadIdx.x / stride_k);

            if (jc0 + nwarps*stride_jc > ncols && jc >= ncols) {
                break;
            }

            const int j = jc / ncols2;
            const int c = jc % ncols2;

            if ((ncols1 == 1 || jt*ncols1 + j < int(ne01.z)) && (ncols2 == 1 || zt_gqa*ncols2 + c < gqa_ratio)) {
#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

                    const float2 tmp = Q_f2[(jt*ncols1 + j)*stride_Q1 + c*stride_Q2 + k];
                    tile_Q[jc*stride_tile_Q + k] = scale_h2 * make_half2(tmp.x, tmp.y);
                }
            } else {
#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

                    tile_Q[jc*stride_tile_Q + k] = make_half2(0.0f, 0.0f);
                }
            }
        }
    }

    __syncthreads();

    if (Q_in_reg) {
        const int j0 = (threadIdx.y / np) * cols_per_warp;

#pragma unroll
        for (int k0 = 0; k0 < DKQ/2; k0 += T_B_KQ::J) {
            load_ldmatrix(Q_B[k0/T_B_KQ::J], tile_Q + j0*stride_tile_Q + k0, stride_tile_Q);
        }
    }

    __syncthreads();

    int kb0 = kb0_start;

    // Preload mask and K data for first iteration when using cp_async with multiple stages:
    if constexpr (nstages > 1) {
        static_assert(!use_sparse, "sparse gather not implemented for multi-stage loading");
        static_assert(nbatch_K2 == DKQ/2, "batching not implemented for multi-stage pipeline");
        constexpr bool use_cp_async = true;
        constexpr bool oob_check    = false;
        constexpr int  k_VKQ_sup    = nbatch_fa;
        if (ncols2 > 1 || mask_h || derived.cell_pos != nullptr) {
            flash_attn_ext_f16_load_mask<ncols1, nwarps, nbatch_fa, use_cp_async, oob_check, use_sparse>
                (mask_h, tile_mask, stride_mask, kb0*nbatch_fa, k_VKQ_sup, jt*ncols1, ne01, nullptr, derived);
        }
        flash_attn_ext_f16_load_tile<stride_tile_K, swz, nwarps, nbatch_fa, use_cp_async, oob_check, use_sparse>
            (K_h2, tile_K, nbatch_K2, stride_K, kb0*nbatch_fa, k_VKQ_sup, nullptr,
             kv_native.K, kv_native.stride_K, 0, kv_native.type_K);
    }

    // kb0_start < kb0_stop for every stream-k block, so the last iter can be executed unconditionally.
    // Only a band block (kb0_step > 1, round-robin KV split) can own no iteration at a short KV; it then
    // skips the loop and writes the neutral partial (max -FLT_MAX/2, rowsum 0, VKQ 0) the fixup ignores.
    if (kb0 < kb0_stop) {
    if constexpr (ncols2 == 1 || use_sparse) {
        constexpr bool oob_check = true;
        if (kq_blocks != nullptr) {
            // Derived mask + fully-masked group bitmap (issue #48): skip the marked KV blocks.  The
            // last_iter distinction only exists to preload the next tile in the multi-stage pipeline;
            // the bitmap is produced only where that pipeline does not exist (AMD has no cp_async),
            // so every block can take last_iter=false and the accumulation is unchanged.
            for (; kb0 < kb0_stop; kb0 += kb0_step) {
                if (fattn_kq_group_masked(kq_blocks, kb0, nbatch_fa, kv_start)) {
                    continue;
                }
                constexpr bool last_iter = false;
                const     int  k_VKQ_sup = min(nbatch_fa, ne11 - kb0*nbatch_fa);
                flash_attn_ext_f16_iter
                    <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, use_sparse, needs_fixup, is_fixup, last_iter, oob_check,
                     T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ, band_fetch_ahead, band_wide>
                    (Q_f2, K_h2, V_h2, mask_h, indices, dstk, dstk_fixup, scale, slope, logit_softcap,
                     ne01, ne02, stride_K, stride_V, stride_mask, tile_Q, tile_K, tile_V, tile_mask, Q_B, VKQ_C,
                     KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup, derived, kv_native);
            }
        } else {
        for (; kb0 + kb0_step < kb0_stop; kb0 += kb0_step) {
            constexpr bool last_iter = false;
            constexpr int  k_VKQ_sup = nbatch_fa;
            flash_attn_ext_f16_iter
                <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, use_sparse, needs_fixup, is_fixup, last_iter, oob_check,
                 T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ, band_fetch_ahead, band_wide>
                (Q_f2, K_h2, V_h2, mask_h, indices, dstk, dstk_fixup, scale, slope, logit_softcap,
                 ne01, ne02, stride_K, stride_V, stride_mask, tile_Q, tile_K, tile_V, tile_mask, Q_B, VKQ_C,
                 KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup, derived, kv_native);
        }
        constexpr bool last_iter = true;
        const     int  k_VKQ_sup = ne11 - kb0*nbatch_fa;
        flash_attn_ext_f16_iter
            <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, use_sparse, needs_fixup, is_fixup, last_iter, oob_check,
              T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ, band_fetch_ahead, band_wide>
            (Q_f2, K_h2, V_h2, mask_h, indices, dstk, dstk_fixup, scale, slope, logit_softcap,
             ne01, ne02, stride_K, stride_V, stride_mask, tile_Q, tile_K, tile_V, tile_mask, Q_B, VKQ_C,
             KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup, derived, kv_native);
        }
    } else {
        constexpr bool oob_check = false;

        // RDNA4 band fetch-ahead (see flash_attn_ext_f16_band_depth): the ring of fetched K/V batches, carried from one
        // KV iteration to the next.  The first batches of the first iteration are fetched here.
        constexpr int band_depth = flash_attn_ext_f16_band_depth<DKQ, DV, ncols1, ncols2, nwarps, V_is_K_view, use_sparse, band_fetch_ahead, band_wide>();
        using band_batch_t = flash_attn_ext_f16_band_batch<nwarps, nbatch_fa, nbatch_K2>;
        band_batch_t band_ring[band_depth > 0 ? band_depth : 1];
        const bool band_q8 = band_depth > 0 && kv_native.type_K == FATTN_KV_NATIVE_Q8_0 && kv_native.type_V == FATTN_KV_NATIVE_Q8_0;
        // Always the ring's own address (a runtime choice between it and nullptr would keep the ring out of registers).
        void * const band_ring_v = band_ring;
        static_assert(band_depth <= (DKQ/2)/nbatch_K2, "the prologue fetches K batches only");
        auto band_prologue = [&](const int kb0_first) {
            if constexpr (band_depth == 1) { // only the cross-iteration stream starts here (see flash_attn_ext_f16_band_depth)
                if (band_q8) {
#pragma unroll
                    for (int t = 0; t < band_depth; ++t) {
                        flash_attn_ext_f16_band_fetch<nwarps, nbatch_fa, nbatch_K2, false>
                            (band_ring[t], kv_native.K, kv_native.stride_K, 2*t*nbatch_K2, kb0_first*nbatch_fa, nbatch_fa);
                    }
                }
            }
        };

        if (kq_blocks != nullptr) {
          if constexpr (band_depth == 1) {
            // The next iteration is the next KV block that is not fully masked (the stream continues into it).
            auto unmasked = [&](int k) {
                while (k < kb0_stop && fattn_kq_group_masked(kq_blocks, k, nbatch_fa, kv_start)) {
                    k += kb0_step;
                }
                return k;
            };
            kb0 = unmasked(kb0);
            if (kb0 < kb0_stop) {
                band_prologue(kb0);
            }
            while (kb0 < kb0_stop) {
                const int kb0_next = unmasked(kb0 + kb0_step);
                constexpr bool last_iter = false;
                constexpr int  k_VKQ_sup = nbatch_fa;
                flash_attn_ext_f16_iter
                    <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, use_sparse, needs_fixup, is_fixup, last_iter, oob_check,
                     T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ, band_fetch_ahead, band_wide>
                    (Q_f2, K_h2, V_h2, mask_h, indices, dstk, dstk_fixup, scale, slope, logit_softcap,
                     ne01, ne02, stride_K, stride_V, stride_mask, tile_Q, tile_K, tile_V, tile_mask, Q_B, VKQ_C,
                     KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup, derived, kv_native, band_ring_v, band_q8, kb0_next < kb0_stop ? kb0_next : -1);
                kb0 = kb0_next;
            }
          } else {
            for (; kb0 < kb0_stop; kb0 += kb0_step) {
                if (fattn_kq_group_masked(kq_blocks, kb0, nbatch_fa, kv_start)) {
                    continue;
                }
                constexpr bool last_iter = false;
                constexpr int  k_VKQ_sup = nbatch_fa;
                flash_attn_ext_f16_iter
                    <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, use_sparse, needs_fixup, is_fixup, last_iter, oob_check,
                     T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ, band_fetch_ahead, band_wide>
                    (Q_f2, K_h2, V_h2, mask_h, indices, dstk, dstk_fixup, scale, slope, logit_softcap,
                     ne01, ne02, stride_K, stride_V, stride_mask, tile_Q, tile_K, tile_V, tile_mask, Q_B, VKQ_C,
                     KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup, derived, kv_native, band_ring_v, band_q8, -1);
            }
          }
        } else {
        band_prologue(kb0);
        for (; kb0 + kb0_step < kb0_stop; kb0 += kb0_step) {
            constexpr bool last_iter = false;
            constexpr int  k_VKQ_sup = nbatch_fa;
            flash_attn_ext_f16_iter
                <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, use_sparse, needs_fixup, is_fixup, last_iter, oob_check,
                 T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ, band_fetch_ahead, band_wide>
                (Q_f2, K_h2, V_h2, mask_h, indices, dstk, dstk_fixup, scale, slope, logit_softcap,
                 ne01, ne02, stride_K, stride_V, stride_mask, tile_Q, tile_K, tile_V, tile_mask, Q_B, VKQ_C,
                 KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup, derived, kv_native, band_ring_v, band_q8, kb0 + kb0_step);
        }
        constexpr bool last_iter = true;
        constexpr int  k_VKQ_sup = nbatch_fa;
        flash_attn_ext_f16_iter
            <DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, use_sparse, needs_fixup, is_fixup, last_iter, oob_check,
             T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ, band_fetch_ahead, band_wide>
            (Q_f2, K_h2, V_h2, mask_h, indices, dstk, dstk_fixup, scale, slope, logit_softcap,
             ne01, ne02, stride_K, stride_V, stride_mask, tile_Q, tile_K, tile_V, tile_mask, Q_B, VKQ_C,
             KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup, derived, kv_native, band_ring_v, band_q8, -1);
        }
    }
    } // kb0 < kb0_stop

    // With multi-stage loading there is no __syncthreads at the end of the iter,
    //     there can be a race condition on shared memory access for combining/writing back results.
    if constexpr (nstages > 1 && nwarps*cols_per_warp > nbatch_fa) {
        __syncthreads();
    }

    // Finally, sum up partial KQ rowsums.
    {
#if defined(TURING_MMA_AVAILABLE)
        // The partial sums are spread across 8/4 threads.
        constexpr int offset_first = cols_per_warp == 8 ? 16 : 2;
        constexpr int offset_last  = cols_per_warp == 8 ?  4 : 1;
#elif defined(AMD_MFMA_AVAILABLE)
        // The partial sums are spread across 4 threads (wavefront64, 16 cols).
        constexpr int offset_first = 32;
        constexpr int offset_last  = 16;
#elif defined(AMD_WMMA_AVAILABLE)
        // The partial sums are spread across 2 threads.
        constexpr int offset_first = 16;
        constexpr int offset_last  = 16;
#else // Volta
        // The partial sums are spread across 2 threads.
        constexpr int offset_first = 2;
        constexpr int offset_last  = 2;
#endif // defined(TURING_MMA_AVAILABLE)
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
#pragma unroll
            for (int offset = offset_first; offset >= offset_last; offset >>= 1) {
                KQ_rowsum[col] += __shfl_xor_sync(0xFFFFFFFF, KQ_rowsum[col], offset, warp_size);
            }
        }
    }

    // If attention sinks are used, potentially re-scale if KQ_max is small.
    // Also add the sink as a value to KQ_rowsum, this is done after synchronization of KQ_rowsum
    //     so it's being done unconditionally for every thread.
    if (!is_fixup && (np == 1 || threadIdx.y % np == 0) && sinks_f) {
        float KQ_max_scale[cols_per_thread];
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            const int jc = (threadIdx.y/np)*cols_per_warp + (cols_per_warp == 8 ? T_C_KQ::get_j(col) : T_C_KQ::get_i(2*col));
            const float sink = sinks_f[jc % ncols2];

            const float KQ_max_new = fmaxf(KQ_max[col], sink);
            const float KQ_max_diff = KQ_max[col] - KQ_max_new;
            KQ_max_scale[col] = expf(KQ_max_diff);
            KQ_max[col] = KQ_max_new;

            *((uint32_t *) &KQ_max_scale[col]) *= KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD;

            const float KQ_max_add = expf(sink - KQ_max_new);
            KQ_rowsum[col] = KQ_max_scale[col]*KQ_rowsum[col] + KQ_max_add;
        }

#if defined(TURING_MMA_AVAILABLE)
        if constexpr (cols_per_warp == 8) {
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[0], KQ_max_scale[cols_per_thread - 1]);
#pragma unroll
            for (int i = 0; i < DV/T_C_VKQ::I; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale_h2;
                }
            }
        } else {
#pragma unroll
            for (int col = 0; col < cols_per_thread; ++col) {
                const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[col], KQ_max_scale[col]);
#pragma unroll
                for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
                    for (int l0 = 0; l0 < T_C_VKQ::ne; l0 += 2) {
                        VKQ_C[i].x[l0 + col] *= KQ_max_scale_h2;
                    }
                }
            }
        }
#elif defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
        if constexpr (std::is_same_v<decltype(T_C_VKQ::x), half2[T_C_VKQ::ne]>) {
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[0], KQ_max_scale[0]);
#pragma unroll
            for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale_h2;
                }
            }
        } else {
            static_assert(std::is_same_v<decltype(T_C_VKQ::x), float[T_C_VKQ::ne]>, "bad VKQ type");
#pragma unroll
            for (int i = 0; i < DV/T_C_VKQ::J; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale[0];
                }
            }
        }
#else // Volta
        const int col = (threadIdx.x / 2) % 2;
        const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[col], KQ_max_scale[col]);
#pragma unroll
        for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
            for (int l = 0; l < T_C_VKQ::ne; ++l) {
                VKQ_C[i].x[l] *= KQ_max_scale_h2;
            }
        }
#endif // defined(TURING_MMA_AVAILABLE)
    }

    // Combine VKQ accumulator values if np > 1.
    // It's also faster to do small writes to shared memory, then large write to VRAM than to do small writes to VRAM.
    // So also write VKQ accumulators to shared memory in column-major format if np == 1.

    constexpr int tile_stride = nbatch_combine + 4;
    static_assert((DV/2) % nbatch_combine == 0, "bad nbatch_combine");

    if constexpr (cols_per_warp == 8) {
        const int jc_cwmo = (threadIdx.x % (2*T_C_VKQ::J)) / T_C_VKQ::J; // jc combine write meta offset
        const int jc_cwm = threadIdx.y*(2*T_C_VKQ::J) + 2*T_C_VKQ::get_j(-1) + jc_cwmo; // jc combine write meta
        const float2 KQ_cmr = make_float2(KQ_max[jc_cwmo], KQ_rowsum[jc_cwmo]); // KQ combine max rowsum

        if constexpr (swz) {
            __syncthreads();
        }

        if (((!needs_fixup && !is_fixup) || np > 1) && threadIdx.x < 2*T_C_VKQ::J) {
            // Use the 16 bytes of padding in each row to store the meta data: KQ max, KQ rowsum, KQ max scale.
            ((float2 *) tile_Q)[jc_cwm*(tile_stride/2) + nbatch_combine/2] = KQ_cmr;
        }

        __syncthreads();

        if (np == 1) {
            // No combination is needed, the meta data can be directly written from registers to VRAM.
            if (needs_fixup && threadIdx.x < T_B_KQ::I) {
                float2 * dstk_fixup_meta = dstk_fixup + (blockIdx.x*gridDim.y + blockIdx.y)*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
            if (is_fixup && threadIdx.x < T_B_KQ::I) {
                float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x*gridDim.y + blockIdx.x*gridDim.y + blockIdx.y)*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
        }
    } else {
        // jc_cwm = jc combine write meta
        // KQ_cmr = KQ combine max rowsum
        // Use the 16 bytes of padding in each Q column to store the meta data: KQ max, KQ rowsum, KQ max scale.
#if defined(TURING_MMA_AVAILABLE)
        const int jc_cwm = threadIdx.y*cols_per_warp + T_C_VKQ::get_i(threadIdx.x % 4);
        const float2 KQ_cmr = make_float2(KQ_max[threadIdx.x % cols_per_thread], KQ_rowsum[threadIdx.x % cols_per_thread]);
        const bool thread_should_write = threadIdx.x % 4 < cols_per_thread;
#elif defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
        const int jc_cwm = threadIdx.y*cols_per_warp + T_C_VKQ::get_i(0);
        const float2 KQ_cmr = make_float2(KQ_max[0], KQ_rowsum[0]);
        const bool thread_should_write = threadIdx.x / 16 < cols_per_thread;
#else // Volta
        const int jc_cwm = threadIdx.y*cols_per_warp + T_C_KQ::get_i(threadIdx.x & 2);
        const float2 KQ_cmr = make_float2(KQ_max[(threadIdx.x & 2) / 2], KQ_rowsum[(threadIdx.x & 2) / 2]);
        const bool thread_should_write = T_C_KQ::J == 8 || T_C_KQ::get_j(threadIdx.x & 2) < 8;
#endif // defined(TURING_MMA_AVAILABLE)

        if constexpr (swz) {
            __syncthreads();
        }

        if (((!needs_fixup && !is_fixup) || np > 1) && thread_should_write) {
            ((float2 *) tile_Q)[jc_cwm*(tile_stride/2) + nbatch_combine/2] = KQ_cmr;
        }
        __syncthreads();

        if (np == 1) {
            // No combination is needed, the meta data can be directly written from registers to VRAM.
            if (needs_fixup && thread_should_write) {
                float2 * dstk_fixup_meta = dstk_fixup + (blockIdx.x*gridDim.y + blockIdx.y)*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
            if (is_fixup && thread_should_write) {
                float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x*gridDim.y + blockIdx.x*gridDim.y + blockIdx.y)*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
        }
    }

    if (np > 1) {
        constexpr int nmeta = np*cols_per_warp >= warp_size ? np*cols_per_warp/warp_size : 1;

        float KQ_cmn;
        float KQ_cms[nmeta];
        float KQ_crs;

        const int jc_meta = threadIdx.y*cols_per_warp + (np*cols_per_warp < warp_size ? threadIdx.x % (np*cols_per_warp) : threadIdx.x);
        float2 * const meta_ptr = ((float2 *) tile_Q) + jc_meta*(tile_stride/2) + nbatch_combine/2;

        if (threadIdx.y % np == 0) {
            // Combine the meta data for parallel warps via shared memory.
            float2 meta[nmeta];
#pragma unroll
            for (int imeta = 0; imeta < nmeta; ++imeta) {
                meta[imeta] = meta_ptr[imeta * warp_size * tile_stride/2];
            }

            KQ_cmn = meta[0].x; // KQ combine max new, max between all parallel warps.
#pragma unroll
            for (int imeta = 1; imeta < nmeta; ++imeta) {
                KQ_cmn = fmaxf(KQ_cmn, meta[imeta].x);
            }
#pragma unroll
            for (int offset = np*cols_per_warp/2; offset >= cols_per_warp; offset >>= 1) {
                if (offset < warp_size) {
                    KQ_cmn = fmaxf(KQ_cmn, __shfl_xor_sync(0xFFFFFFFF, KQ_cmn, offset, warp_size));
                }
            }

#pragma unroll
            for (int imeta = 0; imeta < nmeta; ++imeta) {
                KQ_cms[imeta] = expf(meta[imeta].x - KQ_cmn);
            }

            KQ_crs = KQ_cms[0]*meta[0].y; // KQ combine rowsum, scaled sum of all parallel warps.
#pragma unroll
            for (int imeta = 1; imeta < nmeta; ++imeta) {
                KQ_crs += KQ_cms[imeta]*meta[imeta].y;
            }
#pragma unroll
            for (int offset = np*cols_per_warp/2; offset >= cols_per_warp; offset >>= 1) {
                if (offset < warp_size) {
                    KQ_crs += __shfl_xor_sync(0xFFFFFFFF, KQ_crs, offset, warp_size);
                }
            }
        }

        __syncthreads();

        if (threadIdx.y % np == 0) {
            // Write back combined meta data:
#pragma unroll
            for (int imeta = 0; imeta < nmeta; ++imeta) {
                if (np*cols_per_warp >= warp_size || threadIdx.x < np*cols_per_warp) {
                    // Combined KQ max scale + rowsum.
                    meta_ptr[imeta * warp_size * tile_stride/2] = make_float2(KQ_cms[imeta], KQ_crs);
                }
            }
            // wip/fa-stage-chunking: with np > 1 the combined max (KQ_cmn) is not otherwise readable
            // by the data write below, and the no-seam partial contract has to publish it.  The row's
            // 16 bytes of padding hold one float2 of meta, so the 3rd float is free.
            if (meta_kv != nullptr && (np*cols_per_warp >= warp_size || threadIdx.x < np*cols_per_warp)) {
                ((float *) meta_ptr)[2] = KQ_cmn;
            }

            // Combined KQ max + rowsum.
            static_assert(cols_per_warp <= warp_size);
            if (needs_fixup && (cols_per_warp == warp_size || threadIdx.x < cols_per_warp)) {
                float2 * dstk_fixup_meta = dstk_fixup + (blockIdx.x*gridDim.y + blockIdx.y)*ncols;
                dstk_fixup_meta[(threadIdx.y/np)*cols_per_warp + threadIdx.x] = make_float2(KQ_cmn, KQ_crs);
            }
            if (is_fixup && (cols_per_warp == warp_size || threadIdx.x < cols_per_warp)) {
                float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x*gridDim.y + blockIdx.x*gridDim.y + blockIdx.y)*ncols;
                dstk_fixup_meta[(threadIdx.y/np)*cols_per_warp + threadIdx.x] = make_float2(KQ_cmn, KQ_crs);
            }
        }
    }

#pragma unroll
    for (int k00 = 0; k00 < DV/2; k00 += nbatch_combine) {
        if constexpr (cols_per_warp == 8) {
            static_assert(std::is_same_v<decltype(T_C_VKQ::x), half2[T_C_VKQ::ne]>, "bad VKQ type");
            const int jc_cwd = threadIdx.y*T_B_KQ::I + T_B_KQ::get_i(-1); // jc combine write data
#pragma unroll
            for (int k1 = 0; k1 < nbatch_combine; k1 += T_B_KQ::J) {
                const T_B_KQ B = get_transposed(VKQ_C[(k00 + k1)/T_B_KQ::J]); // Conversion of C to B matrix puts it in column-major format.

#pragma unroll
                for (int l = 0; l < T_B_KQ::ne; ++l) {
                    const int k = k1 + T_B_KQ::get_j(l);

                    tile_Q[jc_cwd*tile_stride + k] = B.x[l];
                }
            }
        } else {
            const int j0 = threadIdx.y*cols_per_warp;
            if constexpr (std::is_same_v<decltype(T_C_VKQ::x), half2[T_C_VKQ::ne]>) {
                if constexpr (T_C_VKQ::dl == DATA_LAYOUT_I_MAJOR) {
#pragma unroll
                    for (int k1 = 0; k1 < nbatch_combine; k1 += T_C_VKQ::J) {
#pragma unroll
                        for (int l = 0; l < T_C_VKQ::ne; ++l) {
                            const int j = j0 + T_C_VKQ::get_i(l);
                            const int k = k1 + T_C_VKQ::get_j(l);

                            tile_Q[j*tile_stride + k] = VKQ_C[(k00 + k1)/T_C_VKQ::J].x[l];
                        }
                    }
                } else {
                    static_assert(T_C_VKQ::dl == DATA_LAYOUT_I_MAJOR_SCRAMBLED, "bad T_C_VKQ data layout");
                    using T_C_VKQ_us = tile<T_C_VKQ::I, T_C_VKQ::J, half2, DATA_LAYOUT_I_MAJOR>; // us == unscrambled
#pragma unroll
                    for (int k1 = 0; k1 < nbatch_combine; k1 += T_C_VKQ::J) {
                        const T_C_VKQ_us VKQ_C_us = unscramble(VKQ_C[(k00 + k1)/T_C_VKQ::J]);
#pragma unroll
                        for (int l = 0; l < T_C_VKQ_us::ne; ++l) {
                            const int j = j0 + T_C_VKQ_us::get_i(l);
                            const int k = k1 + T_C_VKQ_us::get_j(l);

                            tile_Q[j*tile_stride + k] = VKQ_C_us.x[l];
                        }
                    }
                }
            } else {
                static_assert(std::is_same_v<decltype(T_C_VKQ::x), float[T_C_VKQ::ne]>, "bad VKQ type");
                half * tile_Q_h = (half *) tile_Q;
#pragma unroll
                for (int k1 = 0; k1 < nbatch_combine; k1 += T_C_VKQ::J/2) {
#pragma unroll
                    for (int l = 0; l < T_C_VKQ::ne; ++l) {
                        const int j = j0 + T_C_VKQ::get_i(l);
                        const int k = 2*k1 + T_C_VKQ::get_j(l);

                        tile_Q_h[j*(2*tile_stride) + k] = VKQ_C[(k00 + k1)/(T_C_VKQ::J/2)].x[l];
                    }
                }
            }
        }

        __syncthreads();

        if (np == 1 || threadIdx.y % np == 0) {
            // The first 2*2*gridDim.x*gridDim.y*ncols floats in dstk_fixup are for storing max. values and row sums.
            // The values after that are for the partial results of the individual blocks.
            float2 * dstk_fixup_data = dstk_fixup + gridDim.x*gridDim.y*(2*ncols) + (blockIdx.x*gridDim.y + blockIdx.y)*(ncols*(DV/2));

#pragma unroll
            for (int stride_k : {warp_size, warp_size/2, warp_size/4, warp_size/8}) {
                const int k0_start  = stride_k == warp_size ? 0 : nbatch_combine - nbatch_combine % (2*stride_k);
                const int k0_stop   =                             nbatch_combine - nbatch_combine % (1*stride_k);
                const int stride_jc = warp_size / stride_k;

                if (k0_start == k0_stop) {
                    continue;
                }

#pragma unroll
                for (int jc0_dst = 0; jc0_dst < ncols; jc0_dst += (nwarps/np)*stride_jc) {
                    const int jc_dst = jc0_dst + (threadIdx.y/np)*stride_jc + (stride_k == warp_size ? 0 : threadIdx.x / stride_k);

                    if (jc0_dst + (nwarps/np)*stride_jc > ncols && jc_dst >= ncols) {
                        break;
                    }

                    const int jc_tile_K = (jc_dst/cols_per_warp)*(np*cols_per_warp) + jc_dst % cols_per_warp;

                    const int j_dst = jc_dst / ncols2;
                    const int c_dst = jc_dst % ncols2;

                    if (!is_fixup && ((ncols1 > 1 && jt*ncols1 + j_dst >= int(ne01.z)) || (ncols2 > 1 && zt_gqa*ncols2 + c_dst >= gqa_ratio))) {
                        continue;
                    }

                    const float * meta_j = (const float *) tile_Q + jc_tile_K*tile_stride + nbatch_combine;
#pragma unroll
                    for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                        const int k = k0 + (stride_k == warp_size ? threadIdx.x : threadIdx.x % stride_k);

                        float2 dstk_val = make_float2(0.0f, 0.0f);
#pragma unroll
                        for (int ip = 0; ip < np; ++ip) {
                            const float KQ_crs = np == 1 ? 1.0f : meta_j[ip*cols_per_warp * tile_stride + 0];
                            const float2 dstk_val_add = __half22float2(tile_Q[(jc_tile_K + ip*cols_per_warp) * tile_stride + k]);
                            dstk_val.x += dstk_val_add.x*KQ_crs;
                            dstk_val.y += dstk_val_add.y*KQ_crs;
                        }

                        if (!needs_fixup && !is_fixup) {
                            const float KQ_rowsum_j = meta_j[1];
                            if (meta_kv == nullptr) {
                                dstk_val.x /= KQ_rowsum_j;
                                dstk_val.y /= KQ_rowsum_j;
                            }
                        }

                        // wip/fa-stage-chunking: a tile that sits entirely inside one block has no
                        // stream-K seam, so no fixup visits it -- it is the publisher of its own rows
                        // and must hand back (max, rowsum) too, or a chunked consumer has a value it
                        // cannot weight.  One thread per column writes it (idempotent value).
                        // `k == 0` selects the first DV element of the column whichever stride_k pass
                        // owns it (a small nbatch_combine, e.g. 16, leaves the warp_size pass empty).
                        if (meta_kv != nullptr && !needs_fixup && !is_fixup && k00 == 0 && k == 0) {
                            meta_kv[(jt*ncols1 + j_dst)*ne02 + c_dst] =
                                make_float2(np == 1 ? meta_j[0] : meta_j[2], meta_j[1]);
                        }

                        if (is_fixup) {
                            dstk_fixup_data[jc_dst*(DV/2) + k00 + k] = dstk_val;
                        } else {
                            dstk[((jt*ncols1 + j_dst)*ne02 + c_dst)*(DV/2) + k00 + k] = dstk_val;
                        }
                    }
                }
            }
        }
        // Two independent hazards share this barrier:
        //  * upstream #30103: when nbatch_combine != DKQ/2, some warps have not finished reading tile_Q.
        //  * our block 04: tile_Q is reused for the next k00 iteration, so all warps must sync before
        //    its data is overwritten (with np > 1 only some warps read back, but they all write).
        // Neither condition subsumes the other (e.g. DKQ=64/DV=128/nbatch_combine=32 needs ours;
        // nbatch_combine > DV/2 needs upstream's), so keep the union.
        if (np > 1 || nbatch_combine != DKQ/2 || k00 + nbatch_combine < DV/2) {
            __syncthreads();
        }
    }
#else
    GGML_UNUSED_VARS(Q_f2, K_h2, V_h2, mask_h, indices, kq_blocks, sinks_f, dstk, dstk_fixup,
        scale, slope, logit_softcap, ne01, ne02, gqa_ratio,
        stride_Q1, stride_Q2, stride_K, stride_V, stride_mask,
        jt, kb0_start, kb0_stop);
    NO_DEVICE_CODE;
#endif // defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
}

static constexpr __host__ __device__ bool ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(
        const int DKQ, const int DV, const int ncols1, const int ncols2) {
    return (DKQ == 512 && DV == 512 && ncols1 == 1 && ncols2 == 8) ||
           (DKQ == 576 && DV == 512 && ncols1 == 1 && ncols2 == 16) ||
           (DKQ == 256 && DV == 256 && ncols1 == 1 && ncols2 == 8) ||
           (DKQ == 256 && DV == 256 && ncols1 == 8 && ncols2 == 8);
}

template<int DKQ, int DV, int ncols1, int ncols2, bool use_logit_softcap, bool V_is_K_view, bool use_sparse, bool band_fetch_ahead = false,
    bool band_wide = false>
__launch_bounds__(ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, ncols1*ncols2, ncols2, band_wide),
                  ggml_cuda_fattn_mma_get_occupancy(DKQ, DV, ncols1*ncols2, ncols2, band_wide))
static __global__ void flash_attn_ext_f16(
        const char * Q_ptr,
        [[maybe_unused]] const char * Q_h2_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        const char * sinks_ptr,
        const int  * KV_max_ptr,
        const uint32_t * kq_blocks_ptr,
        float      * dst_ptr,
        float2     * dst_meta_ptr,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33,
        const int    * __restrict__ cell_pos_ptr,
        const int    * __restrict__ tok_lo_ptr,
        const int    * __restrict__ tok_hi_ptr,
        // Native (V4 q8_0 / block-15 bf16) K/V: byte-addressed rows, read natively by the loaders
        // (null / FATTN_KV_NATIVE_NONE when the launcher staged F16).
        const int kv_native_K, const int kv_native_V,
        // wip/fa-stage-chunking (TODO #114): non-null enables the per-row partial contract -- every
        // block publishes (unnormalized numerator, max, rowsum) for its rows here (element-major) and
        // the launcher merges KV chunks.  nullptr = the shipping behaviour, bit-identical.  `kv_start`
        // is the first KV row of the launched extent in the whole op's cell space (0 unless chunked).
        float2 * __restrict__ kv_partial_meta_ptr, const int kv_start) {
    ggml_cuda_pdl_sync(); // TODO optimize placement
#if defined(FLASH_ATTN_AVAILABLE) && (defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE))
    const char * GGML_CUDA_RESTRICT Q              = Q_ptr;
    const char * GGML_CUDA_RESTRICT K              = K_ptr;
    const char * GGML_CUDA_RESTRICT V              = V_ptr;
    const char * GGML_CUDA_RESTRICT mask           = mask_ptr;
    const char * GGML_CUDA_RESTRICT sinks          = sinks_ptr;
    // sparse: one index list per (sequence, query tile), the live count of each list follows the lists
    const int  * GGML_CUDA_RESTRICT sparse_indices = use_sparse ? KV_max_ptr : nullptr;

    // V3 derived kq mask: the mask tensor is absent when the host published these instead.
    const kq_derived_t derived = { cell_pos_ptr, tok_lo_ptr, tok_hi_ptr };
    // Fully-masked KV-group bitmap (issue #48): batch-wide on the derived path (mask == nullptr),
    // per (stream, query tile) on the packed path.  See launch_fattn.
    const uint32_t * GGML_CUDA_RESTRICT kq_blocks = kq_blocks_ptr;
    const int  * GGML_CUDA_RESTRICT KV_max        = KV_max_ptr;
    float      * GGML_CUDA_RESTRICT dst            = dst_ptr;
    float2     * GGML_CUDA_RESTRICT dst_meta       = dst_meta_ptr;
    // wip/fa-stage-chunking: element-major per-row partial meta (see the kernel signature).
    float2     * GGML_CUDA_RESTRICT kv_partial_meta = kv_partial_meta_ptr;

    // Skip unused kernel variants for faster compilation:
    if (use_logit_softcap && !(DKQ == 128 || DKQ == 256 || DKQ == 512)) {
        NO_DEVICE_CODE;
        return;
    }
    if (DKQ == 192 && ncols2 != 8 && ncols2 != 16) {
        NO_DEVICE_CODE;
        return;
    }

    if (!ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(DKQ, DV, ncols1, ncols2) && use_sparse) {
        NO_DEVICE_CODE;
        return;
    }
#ifdef VOLTA_MMA_AVAILABLE
    if (ncols1*ncols2 < 32) {
        NO_DEVICE_CODE;
        return;
    }
#endif // VOLTA_MMA_AVAILABLE

#if __CUDA_ARCH__ == GGML_CUDA_CC_TURING
    if (ncols1*ncols2 > 32) {
        NO_DEVICE_CODE;
        return;
    }
#endif // __CUDA_ARCH__ == GGML_CUDA_CC_TURING

#if defined(AMD_WMMA_AVAILABLE)
    if (ncols1*ncols2 < 16 || ncols2 == 1 || DKQ > 576) {
        NO_DEVICE_CODE;
        return;
    }
#endif // defined(AMD_WMMA_AVAILABLE)

#if defined(AMD_MFMA_AVAILABLE)
    if (ncols1*ncols2 < 16 || (DKQ > 256 && ncols1*ncols2 < 32)) {
        NO_DEVICE_CODE;
        return;
    }
#endif // defined(AMD_MFMA_AVAILABLE)

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int ncols     = ncols1 * ncols2;
    constexpr int nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa(DKQ, DV, ncols, ncols2, band_wide);
    constexpr int nthreads  = ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, ncols, ncols2, band_wide);
    constexpr int nwarps    = nthreads / warp_size;

    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.

    const int stride_Q1   = nb01 / sizeof(float2);
    const int stride_Q2   = nb02 / sizeof(float2);
    const int stride_K    = nb11 / sizeof(half2);
    const int stride_mask = nb31 / sizeof(half);

    const int stride_V = V_is_K_view ? stride_K : nb21 / sizeof(half2);

    const int iter_k     = (ne11      + (nbatch_fa - 1)) / nbatch_fa;
    const int iter_j     = (ne01.z    + (ncols1    - 1)) / ncols1;
    const int iter_z_gqa = (gqa_ratio + (ncols2    - 1)) / ncols2;

    if (use_sparse) {
        KV_max = KV_max_ptr + int64_t(iter_j)*ne33*ne11;
    }

#if defined(AMD_WMMA_AVAILABLE)
    // Decode/verify band, launched as gridDim = (output tiles, P) with P > 1.  Block (tile, i) takes
    // the KV iterations i, i+P, i+2P, ... of its tile; block P-1 writes the tile-finishing partial
    // (needs_fixup) and the others the fixup-buffer partials (is_fixup), which is exactly the layout
    // flash_attn_stream_k_fixup_uniform combines (b = tile*P + i).  Compiled only for the instances
    // the band dispatches to (head 256, GQA folded into ncols2 = 8).
    if constexpr (!use_sparse && DKQ == 256 && DV == 256 && ncols2 == 8) {
        if (gridDim.y > 1) {
            const int tile     = blockIdx.x;
            const int sequence =  tile /(iter_j*iter_z_gqa*ne12);
            const int z_KV     = (tile - iter_j*iter_z_gqa*ne12 * sequence)/(iter_j*iter_z_gqa);
            const int zt_gqa   = (tile - iter_j*iter_z_gqa*ne12 * sequence - iter_j*iter_z_gqa * z_KV)/iter_j;
            const int jt       =  tile - iter_j*iter_z_gqa*ne12 * sequence - iter_j*iter_z_gqa * z_KV - iter_j * zt_gqa;

            const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

            const float2 * Q_f2   = (const float2 *) (Q + nb03*sequence + nb02*zt_Q);
            const half2  * K_h2   = (const half2  *) (K + nb13*sequence + nb12*z_KV);
            const half   * mask_h = ncols2 == 1 && !mask ? nullptr :
                (const half *) (mask + nb33*(sequence % ne33));
            float2       * dstk   = ((float2 *) dst) + (sequence*ne01.z*ne02 + zt_Q) * (DV/2);

            const half2 * V_h2 = V_is_K_view ? K_h2 : (const half2 *) (V + nb23*sequence + nb22*z_KV);
            const fattn_kv_native_t kv_native = {
                kv_native_K ? K + nb13*sequence + nb12*z_KV : nullptr,
                kv_native_V ? V + nb23*sequence + nb22*z_KV : nullptr,
                nb11, nb21,
                kv_native_K, kv_native_V,
            };
            const float * sinks_f = sinks ? (const float *) sinks + zt_Q : nullptr;
            const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, zt_Q, n_head_log2, m0, m1) : 1.0f;
            // The packed fully-masked group bitmap of this (Q stream, query tile), when the launcher
            // built one (GGML_CUDA_FA_BAND_SKIP): the round-robin split stays, the masked iterations
            // in it are skipped.  Same layout the prefill path reads below.
            const uint32_t * kq_blocks_tile = kq_blocks != nullptr && mask != nullptr ?
                kq_blocks + int64_t(sequence*iter_j + jt)*(((ne11/FATTN_KQ_STRIDE) + 31)/32) : nullptr;

            const int kb0_start = blockIdx.y;
            const int kb0_stop  = iter_k;
            const int kb0_step  = gridDim.y;

            // wip/fa-stage-chunking: the band is a decode/verify path and is never launched with the
            // partial contract (the launcher only sets it for a wide, staged prefill); a device-side
            // assert would not be compiled there, so the band simply ignores it (see the call sites).

            if (blockIdx.y == gridDim.y - 1) {
                constexpr bool needs_fixup = true;
                constexpr bool is_fixup    = false;
                flash_attn_ext_f16_process_tile<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, use_sparse, needs_fixup, is_fixup, band_fetch_ahead, band_wide>
                    (Q_f2, K_h2, V_h2, mask_h, nullptr, kq_blocks_tile, sinks_f, dstk, dst_meta, scale, slope, logit_softcap,
                     ne01, ne02, gqa_ratio, ne11, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, jt, zt_gqa,
                     kb0_start, kb0_stop, derived, kv_native, nullptr, 0, kb0_step);
            } else {
                constexpr bool needs_fixup = false;
                constexpr bool is_fixup    = true;
                flash_attn_ext_f16_process_tile<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, use_sparse, needs_fixup, is_fixup, band_fetch_ahead, band_wide>
                    (Q_f2, K_h2, V_h2, mask_h, nullptr, kq_blocks_tile, sinks_f, dstk, dst_meta, scale, slope, logit_softcap,
                     ne01, ne02, gqa_ratio, ne11, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, jt, zt_gqa,
                     kb0_start, kb0_stop, derived, kv_native, nullptr, 0, kb0_step);
            }
            return;
        }
    }
#endif // defined(AMD_WMMA_AVAILABLE)

    // kbc == k block continuous, current index in continuous ijk space.
    int       kbc      = int64_t(blockIdx.x + 0)*(iter_k*iter_j*iter_z_gqa*ne12*ne03) / gridDim.x;
    const int kbc_stop = int64_t(blockIdx.x + 1)*(iter_k*iter_j*iter_z_gqa*ne12*ne03) / gridDim.x;

    // If the seams of 2 CUDA blocks fall within an output tile their results need to be combined.
    // For this we need to track both the block that starts the tile (needs_fixup) and the block that finishes the tile (is_fixup).
    // In the most general case >2 seams can fall into the same tile.

    // kb0 == k start index when in the output tile.
    int kb0_start = kbc % iter_k;
    int kb0_stop  = min(iter_k, kb0_start + kbc_stop - kbc);

    while (kbc < kbc_stop && kb0_stop == iter_k) {
        // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
        const int sequence =  kbc /(iter_k*iter_j*iter_z_gqa*ne12);
        const int z_KV     = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence)/(iter_k*iter_j*iter_z_gqa);
        const int zt_gqa   = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV)/(iter_k*iter_j);
        const int jt       = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV - iter_k*iter_j * zt_gqa) / iter_k;

        const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

        const float2 * Q_f2   = (const float2 *) (Q + nb03*sequence + nb02*zt_Q);
        const half2  * K_h2   = (const half2  *) (K + nb13*sequence + nb12*z_KV);
        const half   * mask_h = ncols2 == 1 && !mask ? nullptr :
            (const half *) (mask + nb33*(sequence % ne33));
        float2       * dstk   = ((float2 *) dst) + (sequence*ne01.z*ne02 + zt_Q) * (DV/2);
        // wip/fa-stage-chunking: this tile's rows of the element-major partial meta buffer.  The
        // per-sequence/per-head base is folded in here because process_tile has no `sequence`/`zt_Q`
        // (its dstk is pre-rebased the same way).
        float2 * const meta_kv = kv_partial_meta != nullptr ?
            kv_partial_meta + int64_t(sequence*ne01.z*ne02) + zt_Q : nullptr;

        const half2 * V_h2 = V_is_K_view ? K_h2 : (const half2 *) (V + nb23*sequence + nb22*z_KV);

        // Native (V4 q8_0 / block-15 bf16) K/V: byte-addressed rows for the loaders.
        const fattn_kv_native_t kv_native = {
            kv_native_K ? K + nb13*sequence + nb12*z_KV : nullptr,
            kv_native_V ? V + nb23*sequence + nb22*z_KV : nullptr,
            nb11, nb21,
            kv_native_K, kv_native_V,
        };
        const float * sinks_f = sinks ? (const float *) sinks + zt_Q : nullptr;
        const int32_t * indices = use_sparse ? sparse_indices + (int64_t(sequence % ne33)*iter_j + jt)*ne11 : nullptr;
        const uint32_t * kq_blocks_tile = kq_blocks;
        if (kq_blocks != nullptr && mask != nullptr) { // packed mask: per (Q stream, query tile)
            const int kq_words_per_tile = ((ne11/FATTN_KQ_STRIDE) + 31)/32;
            kq_blocks_tile = kq_blocks + int64_t(sequence*iter_j + jt)*kq_words_per_tile;
        }

        const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, zt_Q, n_head_log2, m0, m1) : 1.0f;

        if (use_sparse) {
            kb0_stop = min(kb0_stop, (KV_max[(sequence % ne33)*iter_j + jt] + nbatch_fa - 1) / nbatch_fa);
        } else if (KV_max) {
            // wip/fa-stage-chunking: KV_max is a whole-op column bound; a chunk's own bound is what
            // remains above its first row.  kv_start is a whole tile, so this stays exact.
            kb0_stop = min(kb0_stop, max(0, KV_max[sequence*iter_j + jt] - kv_start) / nbatch_fa);
        }
        constexpr bool is_fixup = false; // All but (potentially) the last iterations write their data to dst rather than the fixup buffer.
        if (kb0_start == 0) {
            constexpr bool needs_fixup = false; // CUDA block is working on an entire tile.
            flash_attn_ext_f16_process_tile<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, use_sparse, needs_fixup, is_fixup, band_fetch_ahead, band_wide>
                (Q_f2, K_h2, V_h2, mask_h, indices, kq_blocks_tile, sinks_f, dstk, dst_meta, scale, slope, logit_softcap,
                 ne01, ne02, gqa_ratio, ne11, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, jt, zt_gqa, kb0_start, kb0_stop, derived, kv_native,
                 meta_kv, kv_start);
        } else {
            constexpr bool needs_fixup = true; // CUDA block is missing the beginning of a tile.
            flash_attn_ext_f16_process_tile<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, use_sparse, needs_fixup, is_fixup, band_fetch_ahead, band_wide>
                (Q_f2, K_h2, V_h2, mask_h, indices, kq_blocks_tile, sinks_f, dstk, dst_meta, scale, slope, logit_softcap,
                 ne01, ne02, gqa_ratio, ne11, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, jt, zt_gqa, kb0_start, kb0_stop, derived, kv_native,
                 meta_kv, kv_start);
        }

        // The next process_tile call reuses the tile_Q buffer for its Q/K tiles, so all warps must
        // have finished reading the combined results before any of them starts the next call.
        // (With np == 1 the end-of-k00 barrier does not fire, so this is required for correctness.)
        __syncthreads();

        kbc += iter_k;
        kbc -= kbc % iter_k;

        kb0_start = 0;
        kb0_stop  = min(iter_k, kbc_stop - kbc);
    }

    if (kbc >= kbc_stop) {
        return;
    }

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index.
    const int sequence =  kbc /(iter_k*iter_j*iter_z_gqa*ne12);
    const int z_KV     = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence)/(iter_k*iter_j*iter_z_gqa);
    const int zt_gqa   = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV)/(iter_k*iter_j);
    const int jt       = (kbc - iter_k*iter_j*iter_z_gqa*ne12 * sequence - iter_k*iter_j*iter_z_gqa * z_KV - iter_k*iter_j * zt_gqa) / iter_k;

    const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    const float2 * Q_f2   = (const float2 *) (Q + nb03*sequence + nb02*zt_Q);
    const half2  * K_h2   = (const half2  *) (K + nb13*sequence + nb12*z_KV);
    const half   * mask_h = ncols2 == 1 && !mask ? nullptr :
        (const half *) (mask + nb33*(sequence % ne33));
    float2       * dstk   = ((float2 *) dst) + (sequence*ne01.z*ne02 + zt_Q) * (DV/2);
    // wip/fa-stage-chunking: this tile's rows of the element-major partial meta buffer (see above).
    float2 * const meta_kv = kv_partial_meta != nullptr ?
        kv_partial_meta + int64_t(sequence*ne01.z*ne02) + zt_Q : nullptr;

    const half2 * V_h2 = V_is_K_view ? K_h2 : (const half2 *) (V + nb23*sequence + nb22*z_KV);
    const fattn_kv_native_t kv_native = {
        kv_native_K ? K + nb13*sequence + nb12*z_KV : nullptr,
        kv_native_V ? V + nb23*sequence + nb22*z_KV : nullptr,
        nb11, nb21,
        kv_native_K, kv_native_V,
    };
    const float * sinks_f = sinks ? (const float *) sinks + zt_Q : nullptr;
    const int32_t * indices = use_sparse ? sparse_indices + (int64_t(sequence % ne33)*iter_j + jt)*ne11 : nullptr;
    const uint32_t * kq_blocks_tile = kq_blocks;
    if (kq_blocks != nullptr && mask != nullptr) { // packed mask: per (Q stream, query tile)
        const int kq_words_per_tile = ((ne11/FATTN_KQ_STRIDE) + 31)/32;
        kq_blocks_tile = kq_blocks + int64_t(sequence*iter_j + jt)*kq_words_per_tile;
    }

    const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, zt_Q, n_head_log2, m0, m1) : 1.0f;

    if (use_sparse) {
        kb0_stop = min(kb0_stop, (KV_max[(sequence % ne33)*iter_j + jt] + nbatch_fa - 1) / nbatch_fa);
    } else if (KV_max) {
        // wip/fa-stage-chunking: slice the whole-op column bound to this launch's extent.
        kb0_stop = min(kb0_stop, max(0, KV_max[sequence*iter_j + jt] - kv_start) / nbatch_fa);
    }

    constexpr bool is_fixup = true; // Last index writes its data to fixup buffer to avoid data races with other blocks.
    constexpr bool needs_fixup = false;
    flash_attn_ext_f16_process_tile<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, use_sparse, needs_fixup, is_fixup, band_fetch_ahead, band_wide>
        (Q_f2, K_h2, V_h2, mask_h, indices, kq_blocks_tile, sinks_f, dstk, dst_meta, scale, slope, logit_softcap,
         ne01, ne02, gqa_ratio, ne11, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, jt, zt_gqa, kb0_start, kb0_stop, derived, kv_native,
         meta_kv, kv_start);
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, KV_max_ptr, kq_blocks_ptr, dst_ptr, dst_meta_ptr, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33,
        kv_partial_meta_ptr, kv_start);
    NO_DEVICE_CODE;
#endif // defined(FLASH_ATTN_AVAILABLE) && (defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE))
}

bool ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(const int cc, const ggml_tensor * dst, const int ncols1, const int ncols2);

// The q8_0 fetch-ahead variant of an RDNA4 band kernel (see ggml_cuda_fattn_mma_band_has_fetch_ahead) is compiled in a
// translation unit of its own (template-instances/fattn-mma-f16-band-fetch-ahead-*.cu): next to the kernel without it,
// it changes how the compiler optimizes the code the two share, and with it the compiled code of that kernel (not its
// results).
template <int DKQ, int DV, int ncols1, int ncols2, bool band_wide = false>
fattn_kernel_t ggml_cuda_flash_attn_ext_mma_f16_band_fetch_ahead_kernel() {
    return flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, false, false, false, true, band_wide>;
}

#define DECL_FATTN_MMA_F16_BAND_FETCH_AHEAD(DKQ, DV, ncols1, ncols2)                \
    template fattn_kernel_t ggml_cuda_flash_attn_ext_mma_f16_band_fetch_ahead_kernel \
    <DKQ, DV, ncols1, ncols2>()                                                     \

#if defined(GGML_USE_HIP)
extern DECL_FATTN_MMA_F16_BAND_FETCH_AHEAD(256, 256, 4, 8);
extern DECL_FATTN_MMA_F16_BAND_FETCH_AHEAD(256, 256, 8, 8);
#endif // defined(GGML_USE_HIP)

// Width purity of the RDNA4 band is a property of its two config rows (issue #224): the 32-column block
// (query widths 1..4) and the 64-column block (widths 5..8) must reduce each Q column identically, i.e.
// split that column's KV rows the same number of ways (np) and use the same KV batch shape.  When they
// diverge, greedy decode with a draft differs from decode without one -- and no throughput number, no
// same-seed gate and no CPU-referenced op test sees it (the difference is a reordered reduction, ~1e-3
// relative), so a regression here is made a build error instead of a test failure.  r126's table row for
// ncols 64 (8 warps, np 2) against the band's ncols 32 row (8 warps, np 4) fails this.
// 16 = cols_per_warp for the RDNA4 256-dim WMMA tiles (T_B_KQ::I), 32 = the physical wavefront size.
static_assert([]() constexpr {
    constexpr fattn_mma_config band32 = ggml_cuda_fattn_mma_get_config_rdna(256, 256, 32, false, true, 8, false);
    constexpr fattn_mma_config band64 = ggml_cuda_fattn_mma_get_config_rdna(256, 256, 64, false, true, 8, true);
    constexpr int np32 = (band32.nthreads/32)*16/32;
    constexpr int np64 = (band64.nthreads/32)*16/64;
    return np32 == np64 &&
        band32.nbatch_fa      == band64.nbatch_fa      &&
        band32.nbatch_K2      == band64.nbatch_K2      &&
        band32.nbatch_V2      == band64.nbatch_V2      &&
        band32.nstages_target == band64.nstages_target &&
        band32.Q_in_reg       == band64.Q_in_reg;
}(), "the RDNA4 band's 32- and 64-column rows must reduce each Q column identically (issue #224)");

// band_wide: the RDNA4 band's 64-column block (query widths 5..8), which has its own geometry (see
// ggml_cuda_fattn_mma_get_config_rdna) and is dispatched from the band branch in fattn.cu.  It is a
// separate instantiation because a prefill that shares the instantiation would be compiled with the
// band's __launch_bounds__ (r126).
template <int DKQ, int DV, int ncols1, int ncols2, bool band_wide = false>
void ggml_cuda_flash_attn_ext_mma_f16_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    constexpr int ncols = ncols1 * ncols2;

    const int  nthreads       = ggml_cuda_fattn_mma_get_nthreads      (DKQ, DV, ncols, cc, ncols2, band_wide);
    const int  nbatch_fa      = ggml_cuda_fattn_mma_get_nbatch_fa     (DKQ, DV, ncols, cc, ncols2, band_wide);
    const int  nbatch_K2      = ggml_cuda_fattn_mma_get_nbatch_K2     (DKQ, DV, ncols, cc, ncols2, band_wide);
    const int  nbatch_V2      = ggml_cuda_fattn_mma_get_nbatch_V2     (DKQ, DV, ncols, cc, ncols2, band_wide);
    const int  nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols, cc, ncols2, band_wide);
    const int  nstages        = ggml_cuda_fattn_mma_get_nstages       (DKQ, DV, ncols1, ncols2, cc, band_wide);

    // RDNA4 decode/verify with q8_0 K and V read in place (launch_fattn reads them natively for up to 8 query rows): the
    // kernel variant that fetches K/V ahead (see ggml_cuda_fattn_mma_band_has_fetch_ahead).
    bool band_fetch_ahead = false;
    if constexpr (ggml_cuda_fattn_mma_band_has_fetch_ahead(DKQ, DV, ncols1, ncols2)) {
        band_fetch_ahead = GGML_CUDA_CC_IS_RDNA4(cc) && ggml_cuda_fattn_band_fetch_ahead_enabled() && KQV->src[0]->ne[1] <= 8 &&
            ggml_cuda_fattn_kv_native_type(KQV->src[1]) == FATTN_KV_NATIVE_Q8_0 &&
            ggml_cuda_fattn_kv_native_type(KQV->src[2]) == FATTN_KV_NATIVE_Q8_0;
    }
    const bool Q_in_reg       = ggml_cuda_fattn_mma_get_Q_in_reg      (DKQ, DV, ncols, cc, ncols2, band_wide) &&
        !(band_fetch_ahead && ggml_cuda_fattn_mma_band_fetch_ahead_Q_in_lds(DKQ, DV, ncols1, ncols2));

    const int cols_per_warp = std::min(ncols, get_cols_per_warp(cc));
    const int warp_size_host = ggml_cuda_info().devices[ctx.device].warp_size;
    const int nwarps         = nthreads / warp_size_host;

    constexpr bool V_is_K_view = DKQ == 576; // Guaranteed by the kernel selection logic in fattn.cu

    // KV tile strides must match flash_attn_ext_f16_iter / _process_tile.
    const bool swz = ggml_cuda_fattn_mma_get_swizzled(DKQ, DV, ncols1, ncols2, cc, band_wide);
    const int stride_tile_K = swz ? nbatch_K2 : nbatch_K2 + 4;
    const int stride_tile_V = V_is_K_view ? stride_tile_K : (swz ? nbatch_V2 : nbatch_V2 + 4);
    const size_t nbytes_shared_KV_1stage = nbatch_fa            * std::max(stride_tile_K,  stride_tile_V) * sizeof(half2);
    const size_t nbytes_shared_KV_2stage = nbatch_fa            *         (stride_tile_K + stride_tile_V) * sizeof(half2);
    const size_t nbytes_shared_Q         = ncols                * (DKQ/2 + 4)                             * sizeof(half2);
    const size_t nbytes_shared_mask      = ncols1               * (nbatch_fa/2 + 4)                       * sizeof(half2);
    const size_t nbytes_shared_combine   = nwarps*cols_per_warp * (nbatch_combine + 4)                    * sizeof(half2);

    const size_t nbytes_shared_KV = nstages <= 1 ? nbytes_shared_KV_1stage : nbytes_shared_KV_2stage;

    const size_t nbytes_shared_total = std::max(nbytes_shared_combine, Q_in_reg ?
        std::max(nbytes_shared_Q,  nbytes_shared_KV + nbytes_shared_mask) :
                 nbytes_shared_Q + nbytes_shared_KV + nbytes_shared_mask);

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

#if defined(GGML_USE_HIP)
    using fattn_kernel_ptr_t = const void*;
#else
    using fattn_kernel_ptr_t = fattn_kernel_t;
#endif // defined(GGML_USE_HIP)
    fattn_kernel_t fattn_kernel;
    bool use_sparse = false;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        if constexpr (ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(DKQ, DV, ncols1, ncols2)) {
            if (ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(cc, dst, ncols1, ncols2)) {
                constexpr bool use_sparse_kernel = true;
                fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, use_sparse_kernel, false, band_wide>;
                use_sparse = true;

                static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
                if (!shared_memory_limit_raised[id]) {
                    CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
                    shared_memory_limit_raised[id] = true;
                }
            } else {
                constexpr bool use_sparse_kernel = false;
                fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, use_sparse_kernel, false, band_wide>;

                static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
                if (!shared_memory_limit_raised[id]) {
                    CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
                    shared_memory_limit_raised[id] = true;
                }
            }
        } else
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        {
            constexpr bool use_sparse_kernel = false;
            fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, use_sparse_kernel, false, band_wide>;
            if constexpr (ggml_cuda_fattn_mma_band_has_fetch_ahead(DKQ, DV, ncols1, ncols2)) {
                if (band_fetch_ahead) {
                    fattn_kernel = ggml_cuda_flash_attn_ext_mma_f16_band_fetch_ahead_kernel<DKQ, DV, ncols1, ncols2, band_wide>();
                }
            }

            static bool shared_memory_limit_raised[2][GGML_CUDA_MAX_DEVICES] = {{false}};
            if (!shared_memory_limit_raised[band_fetch_ahead][id]) {
                CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
                shared_memory_limit_raised[band_fetch_ahead][id] = true;
            }
        }
    } else {
        constexpr bool use_logit_softcap = true;
        constexpr bool use_sparse_kernel = false;
        fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, use_sparse_kernel, false, band_wide>;

        static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
        if (!shared_memory_limit_raised[id]) {
            CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
            shared_memory_limit_raised[id] = true;
        }
    }

    const bool async_kv_preload = nstages == 2 && !use_sparse;
    launch_fattn<DV, ncols1, ncols2>
        (ctx, dst, fattn_kernel, nwarps, nbytes_shared_total, nbatch_fa, true, true, true, use_sparse, FATTN_KV_NATIVE_PER_OPERAND, warp_size_host, async_kv_preload,
         // q_h2: this kernel reads Q from the tensor (the tile kernel's pre-scaled half2 copy is
         // tile-only), and it is the only kernel with the per-row partial contract (wip/fa-stage-chunking).
         false, true);
}


#define DECL_FATTN_MMA_F16_CASE(DKQ, DV, ncols1, ncols2)                          \
    template void ggml_cuda_flash_attn_ext_mma_f16_case                           \
    <DKQ, DV, ncols1, ncols2>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

// The RDNA4 band's W = 5..8 64-column block: its own case + kernel instantiation (see
// ggml_cuda_fattn_mma_get_config_rdna), in a translation unit of its own.
#define DECL_FATTN_MMA_F16_BAND_WIDE_CASE(DKQ, DV, ncols1, ncols2)                \
    template void ggml_cuda_flash_attn_ext_mma_f16_case                           \
    <DKQ, DV, ncols1, ncols2, true>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

#define DECL_FATTN_MMA_F16_BAND_WIDE_FETCH_AHEAD(DKQ, DV, ncols1, ncols2)         \
    template fattn_kernel_t ggml_cuda_flash_attn_ext_mma_f16_band_fetch_ahead_kernel \
    <DKQ, DV, ncols1, ncols2, true>()                                          \

#define DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(DKQ, DV, ncols)   \
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/ 1,  1); \
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/ 2,  2); \
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/ 4,  4); \
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/ 8,  8); \
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/16, 16); \

DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 64,  64,   8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 80,  80,   8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 96,  96,   8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(112, 112,   8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(128, 128,   8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(256, 256,   8)

DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 64,  64,  16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 80,  80,  16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 96,  96,  16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(112, 112,  16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(128, 128,  16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(256, 256,  16)

DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 64,  64,  32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 80,  80,  32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 96,  96,  32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(112, 112,  32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(128, 128,  32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(256, 256,  32)

DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 64,  64,  64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 80,  80,  64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 96,  96,  64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(112, 112,  64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(128, 128,  64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(256, 256,  64)

extern DECL_FATTN_MMA_F16_CASE(512, 512,  4,  2);
extern DECL_FATTN_MMA_F16_CASE(512, 512,  8,  2);
extern DECL_FATTN_MMA_F16_CASE(512, 512, 16,  2);
extern DECL_FATTN_MMA_F16_CASE(512, 512, 32,  2);
extern DECL_FATTN_MMA_F16_CASE(512, 512,  2,  4);
extern DECL_FATTN_MMA_F16_CASE(512, 512,  4,  4);
extern DECL_FATTN_MMA_F16_CASE(512, 512,  8,  4);
extern DECL_FATTN_MMA_F16_CASE(512, 512, 16,  4);
extern DECL_FATTN_MMA_F16_CASE(512, 512,  1,  8);
extern DECL_FATTN_MMA_F16_CASE(512, 512,  2,  8);
extern DECL_FATTN_MMA_F16_CASE(512, 512,  4,  8);
extern DECL_FATTN_MMA_F16_CASE(512, 512,  8,  8);

// The number of viable configurations for Deepseek is very limited:
extern DECL_FATTN_MMA_F16_CASE(576, 512, 1, 16);
extern DECL_FATTN_MMA_F16_CASE(576, 512, 2, 16);
extern DECL_FATTN_MMA_F16_CASE(576, 512, 4, 16);

// Mistral Small 4 (DKQ=320, DV=256), GQA=32-only build:
extern DECL_FATTN_MMA_F16_CASE(320, 256,  1, 32);
extern DECL_FATTN_MMA_F16_CASE(320, 256,  2, 32);

// For GLM 4.7 Flash
extern DECL_FATTN_MMA_F16_CASE(576, 512,  4,  4);
extern DECL_FATTN_MMA_F16_CASE(576, 512,  8,  4);
extern DECL_FATTN_MMA_F16_CASE(576, 512, 16,  4);
extern DECL_FATTN_MMA_F16_CASE(576, 512,  1, 32);
extern DECL_FATTN_MMA_F16_CASE(576, 512,  2, 32);
