#include "common.cuh"
#include "mmid.cuh"

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The upper and lower bounds for the ith expert in the compact src1 tensor are stored in expert_bounds[i:i+1].
//
// An expert id may repeat within a token's row; every occurrence is an independent destination column and gets its own
// compact row (the Vulkan backend handles the same case in its prepass).  Pass 1 counts the columns of the lower-indexed
// experts (nex_prev), pass 2 writes this expert's columns in (token, iex) order directly to the compact arrays.  No
// per-token staging buffer is used, so the kernel's shared-memory need does not grow with n_tokens*n_expert_used.
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, const bool write_inverse) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int expert = blockIdx.x;
    const int lane   = threadIdx.x;

    // Pass 1: number of compact columns for the experts with a lower index.
    int nex_prev = 0;
    for (int it = 0; it < n_tokens; ++it) {
        for (int iex = lane; iex < n_expert_used; iex += warp_size) {
            nex_prev += ids[it*si1 + iex] < expert;
        }
    }
    nex_prev = warp_reduce_sum<warp_size>(nex_prev);

    // Pass 2: write this expert's columns, one per matching (token, iex) pair, in (token, iex) order.
    int it_compact = 0;
    for (int it = 0; it < n_tokens; ++it) {
        int lane_count = 0;
        for (int iex = lane; iex < n_expert_used; iex += warp_size) {
            lane_count += ids[it*si1 + iex] == expert;
        }
        const int tok_count = warp_reduce_sum<warp_size>(lane_count);
        const int lane_base = warp_prefix_inclusive_sum<int, warp_size>(lane_count) - lane_count;

        int j = 0;
        for (int iex = lane; iex < n_expert_used; iex += warp_size) {
            if (ids[it*si1 + iex] == expert) {
                const int pos = nex_prev + it_compact + lane_base + j;
                ids_dst[pos] = it*n_expert_used + iex;
                // ids_src1 holds the forward map, or the inverse map (token slot -> compact row) for quant dedup
                if (write_inverse) {
                    ids_src1[it*n_expert_used + iex] = pos;
                } else {
                    ids_src1[pos] = it*sis1 + iex % nchannels_y;
                }
                ++j;
            }
        }
        it_compact += tok_count;
    }

    if (lane == 0) {
        expert_bounds[expert] = nex_prev;
        if (expert == static_cast<int>(gridDim.x) - 1) {
            expert_bounds[gridDim.x] = nex_prev + it_compact;
        }
    }
}

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    GGML_ASSERT(n_tokens >= 0 && n_expert_used >= 0);
    GGML_ASSERT((int64_t) n_tokens * n_expert_used < (int64_t) 1 << 31);

    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;

    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, 1, 1);
    mm_ids_helper<<<num_blocks, block_size, 0, stream>>>(
        ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse);
}

