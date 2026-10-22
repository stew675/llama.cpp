#include "common.cuh"
#include "mmid.cuh"

// Qwen4exp 512-expert / 10-active fast path (ported from the halo-box tree). The generic helper
// below launches one warp per expert and every warp scans all token slots (O(n_experts*n_tokens*
// n_used)). Here one 1024-thread block histograms the slots, computes the 512 bounds, and
// scatters the two maps in O(n_tokens*n_used). Row ordering inside an expert is irrelevant:
// ids_dst maps every compact MMQ row back to its exact destination and there is no cross-row
// reduction, so results are identical to the generic kernel by construction.  Every occurrence of
// a repeated id is its own slot here, matching the generic kernel's duplicate handling.
__launch_bounds__(1024, 1)
static __global__ void mm_ids_helper_512_10(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst,
        int32_t * __restrict__ expert_bounds, const int n_tokens, const int nchannels_y,
        const int si1, const int sis1, const bool write_inverse) {
    __shared__ int counts[512];
    __shared__ int cursors[512];

    const int tid = threadIdx.x;
    if (tid < 512) {
        counts[tid] = 0;
        cursors[tid] = 0;
    }
    __syncthreads();

    const int n_slots = n_tokens * 10;
    for (int idx = tid; idx < n_slots; idx += blockDim.x) {
        const int token = idx / 10;
        const int iex   = idx - token * 10;
        atomicAdd(&counts[ids[token * si1 + iex]], 1);
    }
    __syncthreads();

    if (tid == 0) {
        int total = 0;
        expert_bounds[0] = 0;
        for (int e = 0; e < 512; ++e) {
            total += counts[e];
            expert_bounds[e + 1] = total;
        }
    }
    __syncthreads();

    for (int idx = tid; idx < n_slots; idx += blockDim.x) {
        const int token = idx / 10;
        const int iex   = idx - token * 10;
        const int expert = ids[token * si1 + iex];
        const int dst = expert_bounds[expert] + atomicAdd(&cursors[expert], 1);
        ids_dst[dst] = idx;
        if (write_inverse) {
            ids_src1[idx] = dst;
        } else {
            ids_src1[dst] = token * sis1 + iex % nchannels_y;
        }
    }
}

// the generic path passes 0, which needs no padding since it never groups lanes by token
template <int n> struct mm_ids_pow2 { static constexpr int value = 2*mm_ids_pow2<(n + 1)/2>::value; };
template <>      struct mm_ids_pow2<1> { static constexpr int value = 1; };
template <>      struct mm_ids_pow2<0> { static constexpr int value = 1; };

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The upper and lower bounds for the ith expert in the compact src1 tensor are stored in expert_bounds[i:i+1].
//
// An expert id may repeat within a token's row; every occurrence is an independent destination column and gets its own
// compact row (the Vulkan backend handles the same case in its prepass).  The compact columns are written in (token, iex)
// order: the offset within an expert is `it_compact (earlier tokens) + tok_base (earlier tokens in this warp iteration) +
// rank (this lane's occurrence index within its token)`.
//
// The common expert counts take the templated fast path: `warp_size/neu_padded` tokens are handled per warp iteration and
// the per-token match count / rank come from reductions within the token's padded lane group.  The r66 rewrite of this
// helper used one full warp iteration (and two warp reductions) per token; that cost ~20% of the 35B-A3B prefill
// (measured gfx1201, `-ncmoe 0 -ub 4096 -fa 1`: 8357 -> 6430 t/s).  The fast path restores the token batching while
// keeping every repeated occurrence.  `n_expert_used_template == 0` keeps the generic form as the fallback.
template <int n_expert_used_template>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;

    if constexpr (n_expert_used_template == 0) {
        // Generic implementation: one warp iteration (and two warp reductions) per token.  Kept for
        // expert counts without a template instantiation; needs no per-token staging buffer.
        const int lane = threadIdx.x;

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
        return;
    }

    // Fast implementation for a fixed number of experts used: a warp holds a whole number of token slots,
    // so the slot count is padded to a power of 2.
    constexpr int neu_padded   = mm_ids_pow2<n_expert_used_template>::value;
    constexpr int tok_per_iter = warp_size / neu_padded;
    static_assert(neu_padded <= warp_size && warp_size % neu_padded == 0, "bad n_expert_used");

    // Pass 1: number of compact columns for the experts with a lower index.
    int nex_prev = 0;
    for (int it0 = 0; it0 < n_tokens; it0 += tok_per_iter) {
        const int it  = it0 + threadIdx.x / neu_padded;
        const int iex = threadIdx.x % neu_padded;
        const int expert_used = (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ?
            ids[it*si1 + iex] : INT_MAX;
        nex_prev += expert_used < expert;
    }
    nex_prev = warp_reduce_sum<warp_size>(nex_prev);

    // Pass 2: write this expert's columns, one per matching (token, iex) pair, in (token, iex) order.
    int it_compact = 0;
    for (int it0 = 0; it0 < n_tokens; it0 += tok_per_iter) {
        const int it  = it0 + threadIdx.x / neu_padded;
        const int iex = threadIdx.x % neu_padded;
        const int expert_used = (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ?
            ids[it*si1 + iex] : INT_MAX;
        const bool match = expert_used == expert;

        // The number of occurrences of this expert in this token (duplicates allowed) and this lane's rank among them.
        const int tok_count = warp_reduce_sum<neu_padded>((int) match);
        const int lane_rank = warp_prefix_inclusive_sum<int, neu_padded>((int) match) - (int) match;

        // Scan over the tokens at lower positions in the warp for this expert's compact offset.
        int tok_base = 0;
#pragma unroll
        for (int offset = neu_padded; offset < warp_size; offset += neu_padded) {
            const int tmp = __shfl_up_sync(0xFFFFFFFF, tok_count, offset, warp_size);
            if (threadIdx.x >= static_cast<unsigned int>(offset)) {
                tok_base += tmp;
            }
        }

        if (match) {
            const int pos = nex_prev + it_compact + tok_base + lane_rank;
            ids_dst[pos] = it*n_expert_used + iex;
            // ids_src1 holds the forward map, or the inverse map (token slot -> compact row) for quant dedup
            if (write_inverse) {
                ids_src1[it*n_expert_used + iex] = pos;
            } else {
                ids_src1[pos] = it*sis1 + iex % nchannels_y;
            }
        }

        // The thread with the highest index in the warp has the sum over the whole warp.
        it_compact += __shfl_sync(0xFFFFFFFF, tok_base + tok_count, warp_size - 1, warp_size);
    }

    if (threadIdx.x == 0) {
        expert_bounds[expert] = nex_prev;
        if (expert == static_cast<int>(gridDim.x) - 1) {
            expert_bounds[gridDim.x] = nex_prev + it_compact;
        }
    }
}

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    static const bool mmid_512_disabled = getenv("GGML_CUDA_DISABLE_MMID_512") != nullptr;
    // mm_ids_helper_512_10: one 1024-thread block replaces the 512 one-warp expert blocks of the
    // generic helper for the 512-expert/10-used shape.  RDNA3_5 (source of record) + RDNA4
    // (2026-09-06 gfx1201 validation: prefill +4.5% pp16384 / +5.2% pp2048, same-seed text
    // byte-identical; decode unaffected - the helper feeds the mmq prefill path only).
    if ((GGML_CUDA_CC_IS_RDNA3_5(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) && n_experts == 512 && n_expert_used == 10 && !mmid_512_disabled) {
        mm_ids_helper_512_10<<<1, 1024, 0, stream>>>(
            ids, ids_src1, ids_dst, expert_bounds, n_tokens, nchannels_y, si1, sis1, write_inverse);
        return;
    }

    GGML_ASSERT(n_tokens >= 0 && n_expert_used >= 0);
    GGML_ASSERT((int64_t) n_tokens * n_expert_used < (int64_t) 1 << 31);

    const int id = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[id].warp_size;

    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, 1, 1);

    switch (n_expert_used) {
        case  2:
            mm_ids_helper< 2><<<num_blocks, block_size, 0, stream>>>(ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse);
            break;
        case  4:
            mm_ids_helper< 4><<<num_blocks, block_size, 0, stream>>>(ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse);
            break;
        case  6:
            mm_ids_helper< 6><<<num_blocks, block_size, 0, stream>>>(ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse);
            break;
        case  8:
            mm_ids_helper< 8><<<num_blocks, block_size, 0, stream>>>(ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse);
            break;
        case 10:
            mm_ids_helper<10><<<num_blocks, block_size, 0, stream>>>(ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse);
            break;
        case 16:
            mm_ids_helper<16><<<num_blocks, block_size, 0, stream>>>(ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse);
            break;
        case 32:
            mm_ids_helper<32><<<num_blocks, block_size, 0, stream>>>(ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse);
            break;
        default:
            mm_ids_helper< 0><<<num_blocks, block_size, 0, stream>>>(ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse);
            break;
    }
}
