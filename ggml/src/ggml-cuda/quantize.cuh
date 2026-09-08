#pragma once

#include "common.cuh"
#include "mmq.cuh"

#include <cstdint>

#define CUDA_QUANTIZE_BLOCK_SIZE     256
#define CUDA_QUANTIZE_BLOCK_SIZE_MMQ 128

static_assert(MATRIX_ROW_PADDING %    CUDA_QUANTIZE_BLOCK_SIZE      == 0, "Risk of out-of-bounds access.");
static_assert(MATRIX_ROW_PADDING % (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ) == 0, "Risk of out-of-bounds access.");

// halo-box merge (gated to gfx1151 / Strix Halo, validated): quantize_mmq_q8_1 splits its
// gridDim.y work across ggml_cuda_quantize_mmq_q8_1_n_chunks(cc) consecutive 512-float slices
// per block. The mmq q8_1 feed tensors have up to 262144 rows, so the unchunked form launches
// tens of millions of tiny 128-thread blocks and is block-dispatch-bound on RDNA3.5; the 2x
// block-count reduction measured ~1.5x faster per call with byte-identical output (each slice
// is exactly the work one unchunked block did - per-slice element math is unchanged). GATED:
// only the exact validated SKU gfx1151 (Strix Halo) enables it. gfx1150 (Strix Point) shares the
// RDNA3.5 block dispatcher and likely benefits but is NOT validated; gfx120x/RDNA4 and all
// non-AMD targets keep n_chunks = 1 = the upstream unchunked launch — RDNA4 A/B measured flat
// on gfx1201 (2026-09-06, wip 1.2: pp8192/pp16384 r3 ON-vs-OFF within ±1% drift; gfx1201's
// dispatcher does not share gfx1151's small-block dispatch bottleneck). The chunk count is a
// runtime kernel argument (not a compile-time define): the HIP host pass does not see __gfx*__
// macros, so a define would disagree between the host (gridDim.y) and device (loop) passes.
static inline int ggml_cuda_quantize_mmq_q8_1_n_chunks(const int cc) {
    return GGML_CUDA_CC_IS_GFX1151(cc) ? 2 : 1;
}

static_assert((4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ) % QK8_1_MMQ == 0, "Quantization chunks must contain complete Q8_1 MMQ blocks.");

typedef void (*quantize_cuda_t)(
        const float * x, const int32_t * ids, void * vy,
        ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

void quantize_row_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy,
        ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

void quantize_mmq_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy,
        ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, const int n_chunks, cudaStream_t stream);

void quantize_mmq_q8_1_swiglu_cuda(
        const float * gate, const float * up, const int32_t * ids, void * vy, ggml_type type_src0,
        int64_t ne00, int64_t ne0, int64_t ne1, int logical_n1,
        int64_t gate_s1, int64_t gate_st, int64_t up_s1, int64_t up_st, cudaStream_t stream);

void quantize_mmq_fp4_cuda(const float *   x,
                             const int32_t * ids,
                             void *          vy,
                             float *         scale,
                             ggml_type       type_src0,
                             bool            use_aligned_float8,
                             int64_t         ne00,
                             int64_t         s01,
                             int64_t         s02,
                             int64_t         s03,
                             int64_t         ne0,
                             int64_t         ne1,
                             int64_t         ne2,
                             int64_t         ne3,
                             cudaStream_t    stream);

// quantize each token once and scatter the block to its compact rows (via the inverse map)
void quantize_scatter_mmq_fp4_cuda(const float *   x,
                                   const int32_t * ids_src1_inv,
                                   void *          vy,
                                   float *         scale,
                                   ggml_type       type_src0,
                                   bool            use_aligned_float8,
                                   int64_t         ne00,
                                   int64_t         stride_token,
                                   int64_t         ne0,
                                   int64_t         n_tokens,
                                   int64_t         nrows_dst,
                                   int             n_expert_used,
                                   cudaStream_t    stream);

void quantize_scatter_mmq_q8_1_cuda(const float *   x,
                                    const int32_t * ids_src1_inv,
                                    void *          vy,
                                    ggml_type       type_src0,
                                    int64_t         ne00,
                                    int64_t         stride_token,
                                    int64_t         ne0,
                                    int64_t         n_tokens,
                                    int64_t         nrows_dst,
                                    int             n_expert_used,
                                    const int       n_chunks,
                                    cudaStream_t    stream);
