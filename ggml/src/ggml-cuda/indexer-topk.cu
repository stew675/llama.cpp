#include "indexer-topk.cuh"
#include "common.cuh"

// Fused indexer expand + mask + top-k.
//
// The qwen4exp indexer scores blocks, then every cell of a block carries the
// block score, the attention mask is added per cell, and the top-k cells are
// selected.  The plain graph materializes the full [n_kv, n_tps] F32 expanded
// tensor (512 MB at 64K, mirrored on every GPU) only to feed a radix top-k
// that re-reads it 5 times.  This op computes
//
//     value(c) = score[cell_blk(c)] + additive(c)
//
// on the fly in each radix pass, so the expanded tensor never exists.
//
// Layouts:
//   score     [n_blocks, n_tps, n_stream] F32
//   cell_blk  [n_kv, n_stream] I32
//   additive  [n_kv, n_tps, n_stream] F16 or F32 (attention mask or bias)
//   dst       [k, n_tps, 1, n_stream] I32
//
// Each row (tps, stream) is an independent top-k over n_kv cells.  The
// score gather goes through cell_blk, which has only n_blocks distinct
// values (r cells per block), so the score column stays in L2.

static __device__ __forceinline__ uint32_t indexer_topk_float_to_ordered(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}

struct indexer_topk_radix_state {
    uint32_t prefix;
    uint32_t prefix_mask;
    int rank;
    int greater_count;
    int equal_count;
};

static __global__ void indexer_topk_radix_init(indexer_topk_radix_state * states, int nrows, int k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row] = {0, 0, k, 0, 0};
    }
}

// value(c) for row r: score[cell_blk(c, s), t, s] + additive(c, t, s)
template<typename kv_t>
static __device__ __forceinline__ float indexer_topk_value(
        const float * __restrict__ score,
        const int   * __restrict__ cell_blk,
        const kv_t  * __restrict__ additive,
        int c, int t, int s,
        int n_blocks, int n_tps, int n_kv) {
    const int b = cell_blk[c + s*n_kv];
    const float sc = score[b + t*n_blocks + s*n_blocks*n_tps];
    return sc + (float) additive[c + t*n_kv + s*n_kv*n_tps];
}

template<int BLOCK_SIZE, int RADIX_BITS, typename kv_t>
static __global__ void indexer_topk_radix_histogram(
        const float * __restrict__ score,
        const int   * __restrict__ cell_blk,
        const kv_t  * __restrict__ additive,
        const indexer_topk_radix_state * __restrict__ states,
        int * __restrict__ block_histograms,
        int ncols, int n_tps, int n_blocks, int n_kv,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const int t = row % n_tps;
    const int s = row / n_tps;
    __shared__ int histogram[NBINS];

    for (int i = tid; i < NBINS; i += BLOCK_SIZE) {
        histogram[i] = 0;
    }
    __syncthreads();

    const indexer_topk_radix_state state = states[row];
    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = indexer_topk_float_to_ordered(
                indexer_topk_value(score, cell_blk, additive, col, t, s, n_blocks, n_tps, n_kv));
        if ((key & state.prefix_mask) == state.prefix) {
            atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
        }
    }
    __syncthreads();

    const size_t histogram_offset =
        ((size_t) row * blocks_per_row + row_block) * NBINS;
    block_histograms[histogram_offset + tid] = histogram[tid];
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void indexer_topk_radix_select(
        const int * __restrict__ block_histograms,
        indexer_topk_radix_state * __restrict__ states,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int histogram[NBINS];

    int count = 0;
    for (int i = tid; i < NBINS; i += BLOCK_SIZE) {
        count = 0;
        for (int row_block = 0; row_block < blocks_per_row; ++row_block) {
            const size_t offset = ((size_t) row * blocks_per_row + row_block) * NBINS;
            count += block_histograms[offset + i];
        }
        histogram[i] = count;
    }
    __syncthreads();

    if (tid == 0) {
        indexer_topk_radix_state state = states[row];
        int bin = NBINS - 1;
        while (bin > 0 && histogram[bin] < state.rank) {
            state.rank -= histogram[bin--];
        }
        state.prefix |= (uint32_t) bin << shift;
        state.prefix_mask |= (uint32_t) (NBINS - 1) << shift;
        states[row] = state;
    }
}

static __global__ void indexer_topk_radix_reset_counters(indexer_topk_radix_state * states, int nrows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row].greater_count = 0;
        states[row].equal_count = 0;
    }
}

template<int BLOCK_SIZE, typename kv_t>
static __global__ void indexer_topk_radix_gather(
        const float * __restrict__ score,
        const int   * __restrict__ cell_blk,
        const kv_t  * __restrict__ additive,
        int * __restrict__ dst,
        indexer_topk_radix_state * __restrict__ states,
        int ncols, int n_tps, int n_blocks, int n_kv, int k,
        int blocks_per_row) {
    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const int t = row % n_tps;
    const int s = row / n_tps;
    int * row_dst = dst + (size_t) row * k;
    indexer_topk_radix_state * state = &states[row];

    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = indexer_topk_float_to_ordered(
                indexer_topk_value(score, cell_blk, additive, col, t, s, n_blocks, n_tps, n_kv));
        if (key > state->prefix) {
            const int pos = atomicAdd(&state->greater_count, 1);
            row_dst[pos] = col;
        } else if (key == state->prefix) {
            const int pos = atomicAdd(&state->equal_count, 1);
            if (pos < state->rank) {
                row_dst[k - state->rank + pos] = col;
            }
        }
    }
}

template<typename kv_t>
static void indexer_topk_radix_cuda(
        ggml_cuda_pool & pool,
        const float * score, const int * cell_blk, const kv_t * additive,
        int * dst, int ncols, int nrows, int n_tps, int n_blocks, int n_kv, int k,
        cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS = 1 << RADIX_BITS;
    const int blocks_per_row = std::min((ncols + 1023) / 1024, 8);

    ggml_cuda_pool_alloc<indexer_topk_radix_state> states_alloc(pool, nrows);
    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) nrows * blocks_per_row * NBINS);
    indexer_topk_radix_state * states = states_alloc.get();
    int * histograms = histograms_alloc.get();

    indexer_topk_radix_init<<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows, k);

    const dim3 row_grid(blocks_per_row * nrows);
    for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
        indexer_topk_radix_histogram<BLOCK_SIZE, RADIX_BITS, kv_t>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                score, cell_blk, additive, states, histograms,
                ncols, n_tps, n_blocks, n_kv, blocks_per_row, shift);
        indexer_topk_radix_select<BLOCK_SIZE, RADIX_BITS>
            <<<nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift);
    }

    indexer_topk_radix_reset_counters
        <<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows);
    indexer_topk_radix_gather<BLOCK_SIZE, kv_t>
        <<<row_grid, BLOCK_SIZE, 0, stream>>>(
            score, cell_blk, additive, dst, states,
            ncols, n_tps, n_blocks, n_kv, k, blocks_per_row);
}

void ggml_cuda_indexer_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * score    = dst->src[0];
    const ggml_tensor * cell_blk = dst->src[1];
    const ggml_tensor * additive = dst->src[2];
    const float * score_d    = (const float *) score->data;
    const int   * cell_blk_d = (const int  *) cell_blk->data;
    int *         dst_d      = (int *) dst->data;
    cudaStream_t  stream     = ctx.stream();

    GGML_ASSERT(score->type == GGML_TYPE_F32);
    GGML_ASSERT(cell_blk->type == GGML_TYPE_I32);
    GGML_ASSERT(additive->type == GGML_TYPE_F16 || additive->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(score));
    GGML_ASSERT(ggml_is_contiguous(cell_blk));
    GGML_ASSERT(ggml_is_contiguous(additive));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int n_blocks = score->ne[0];
    const int n_tps    = score->ne[1];
    const int n_stream = score->ne[2];
    const int n_kv     = cell_blk->ne[0];
    const int nrows    = n_tps * n_stream;
    const int k        = dst->ne[0];
    ggml_cuda_pool & pool = ctx.pool();

    if (additive->type == GGML_TYPE_F16) {
        indexer_topk_radix_cuda(pool, score_d, cell_blk_d, (const half *) additive->data,
                dst_d, n_kv, nrows, n_tps, n_blocks, n_kv, k, stream);
    } else {
        indexer_topk_radix_cuda(pool, score_d, cell_blk_d, (const float *) additive->data,
                dst_d, n_kv, nrows, n_tps, n_blocks, n_kv, k, stream);
    }
}

bool ggml_cuda_indexer_top_k_supported(int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);

    const ggml_tensor * score    = dst->src[0];
    const ggml_tensor * cell_blk = dst->src[1];
    const ggml_tensor * additive = dst->src[2];

    return score->type == GGML_TYPE_F32 &&
        cell_blk->type == GGML_TYPE_I32 &&
        (additive->type == GGML_TYPE_F16 || additive->type == GGML_TYPE_F32) &&
        dst->type == GGML_TYPE_I32;
}

