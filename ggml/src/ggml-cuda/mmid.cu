#include "common.cuh"
#include "mmid.cuh"

// To reduce shared memory use, store "it" and "iex_used" with 22/10 bits each.
struct mm_ids_helper_store {
    uint32_t data;

    __device__ mm_ids_helper_store(const uint32_t it, const uint32_t iex_used) {
        data = (it & 0x003FFFFF) | (iex_used << 22);
    }

    __device__ uint32_t it() const {
        return data & 0x003FFFFF;
    }

    __device__ uint32_t iex_used() const {
        return data >> 22;
    }
};
static_assert(sizeof(mm_ids_helper_store) == 4, "unexpected size for mm_ids_helper_store");

// the generic path passes 0, which needs no padding since it never groups lanes by token
template <int n> struct mm_ids_pow2 { static constexpr int value = 2*mm_ids_pow2<(n + 1)/2>::value; };
template <>      struct mm_ids_pow2<1> { static constexpr int value = 1; };
template <>      struct mm_ids_pow2<0> { static constexpr int value = 1; };

template <bool increment_nex_prev, bool write_to_store>
static __device__ __forceinline__ void check_experts(
        mm_ids_helper_store * store, int & nex_prev, int & it_compact,
        const int expert, const int it, const int iex, const int expert_found) {
    if (increment_nex_prev) {
        nex_prev += expert_found < expert;
    }

    if (write_to_store) {
        const ggml_cuda_thread_mask_t iex_used = ggml_cuda_ballot(expert_found == expert);
        const ggml_cuda_thread_mask_t mask = (ggml_cuda_thread_mask_t(1) << threadIdx.x) - 1;
        const int it_compact_offset = ggml_cuda_popc(iex_used & mask);

        if (expert_found == expert) {
            store[it_compact + it_compact_offset] = mm_ids_helper_store(it, iex);
        }

        it_compact += ggml_cuda_popc(iex_used);
    }
}

static __device__ __forceinline__ void dump_store(
        int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst,
        const int n_expert_used, const int nchannels_y, const int sis1, const bool write_inverse,
        const mm_ids_helper_store * __restrict__ store, const int nex_prev, int & n_dumped, const int it_compact) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    for (int itc = threadIdx.x; itc < it_compact; itc += warp_size) {
        const mm_ids_helper_store store_it = store[itc];
        const int it       = store_it.it();
        const int iex_used = store_it.iex_used();
        ids_dst[nex_prev + n_dumped + itc] = it*n_expert_used + iex_used;
        // ids_src1 holds the forward map, or the inverse map (token slot -> compact row) for quant dedup
        if (write_inverse) {
            ids_src1[it*n_expert_used + iex_used] = nex_prev + n_dumped + itc;
        } else {
            ids_src1[nex_prev + n_dumped + itc] = it*sis1 + iex_used % nchannels_y;
        }
    }
    n_dumped += it_compact;
}

constexpr int n_unroll = 4; // Manually unroll this many loop iterations for better performance.

// Writes the tile_spec entries of one expert (thread 0 of its block only).
//
// Upstream (#31047) assigns one MUL_MAT_ID tile per tile_spec entry and launches exactly
//     ntj = ceil(n_tokens*n_expert_used / tile_width) + n_experts
// blocks in the grid: the entries at < ntj - n_experts are the tile-aligned tiles indexed by their column
// position, and the last n_experts entries ("between the seams") hold each expert's remaining columns.
static __device__ __forceinline__ void mm_ids_write_tile_spec(
        ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec, const int expert, const int nex_prev, const int n_dumped,
        const uint3 tile_width, const int n_tokens, const int n_expert_used) {
    int col_low  = nex_prev;
    int col_high = nex_prev + min(int(tile_width.z), n_dumped);

    // The while loop covers all tiles at the seams containing the column with % tile_width == 0.
    while (col_high > fastdiv(col_low + tile_width.z - 1, tile_width) * tile_width.z) {
        tile_spec[fastdiv(col_low + tile_width.z - 1, tile_width)]
            = {expert, col_low, col_high, /*padding =*/ 0};

        col_low  = col_high;
        col_high = min(col_high + int(tile_width.z), nex_prev + n_dumped);
    }

    // Finally, at most one extra tile_spec between the seams per expert.
    tile_spec[fastdiv(n_tokens*n_expert_used + tile_width.z - 1, tile_width) + expert]
        = {expert, col_low, col_high, /*padding =*/ 0};
}

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The first (n_tokens*n_expert_used + J - 1) / tile_width elements of tile_spec contain the specication for
//   matrix multiplication tiles at token positions which are exactly divided by tile_width.
// Additionally the following n_expert elements of tile_spec may contain additional tiles that fall between these seams.
template <int n_expert_used_template>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst,
        ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const uint3 tile_width,
        const int si1, const int sis1, const bool write_inverse, const int ne_store) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;

    const int expert = blockIdx.x; // The expert this CUDA block handles.

    extern __shared__ char data_mm_ids_helper[];
    mm_ids_helper_store * store = (mm_ids_helper_store *) data_mm_ids_helper;

    int nex_prev   = 0; // Number of columns for experts with a lower index.
    int n_dumped   = 0; // Number of data values that have been dumped to output buffers.
    int it_compact = 0; // Running index for the compact slice of this expert.

    // Step 1:   Iterate over n_tokens, increment nex_prev to determine the output indices, fill up store.
    // Step 1.5: If the store does not have enough space to hold all values, iterate over the rest of the tokens just to increment nex_prev.
    // Step 2:   Dump the stored values using nex_prev to get the correct output indices.
    // Step 3:   If we were not able to process all values, resume at the index up to which we were able to dump the output and re-do.

    if constexpr (n_expert_used_template == 0) {
        // Generic implementation:
        int it = 0;
        for (; it < n_tokens && it_compact + max(warp_size, n_expert_used) <= ne_store; ++it) {
            for (int iex0 = 0; iex0 < n_expert_used; iex0 += warp_size) {
                const int iex = iex0 + threadIdx.x;
                const int expert_found = iex < n_expert_used ? ids[it*si1 + iex] : INT_MAX;

                constexpr bool increment_nex_prev = true;
                constexpr bool write_to_store     = true;
                check_experts<increment_nex_prev, write_to_store>
                    (store, nex_prev, it_compact, expert, it, iex, expert_found);
            }
        }
        int it_dumped = it;
        for (; it < n_tokens; ++it) {
            for (int iex0 = 0; iex0 < n_expert_used; iex0 += warp_size) {
                const int iex = iex0 + threadIdx.x;
                const int expert_found = iex < n_expert_used ? ids[it*si1 + iex] : INT_MAX;

                constexpr bool increment_nex_prev = true;
                constexpr bool write_to_store     = false;
                check_experts<increment_nex_prev, write_to_store>
                    (store, nex_prev, it_compact, expert, it, iex, expert_found);
            }
        }

        nex_prev = warp_reduce_sum<warp_size>(nex_prev);
        ggml_cuda_syncwarp();
        dump_store(ids_src1, ids_dst, n_expert_used, nchannels_y, sis1, write_inverse, store, nex_prev, n_dumped, it_compact);
        it_compact = 0;

        for (int it = it_dumped; it < n_tokens; ++it) {
            for (int iex0 = 0; iex0 < n_expert_used; iex0 += warp_size) {
                const int iex = iex0 + threadIdx.x;
                const int expert_found = iex < n_expert_used ? ids[it*si1 + iex] : INT_MAX;

                constexpr bool increment_nex_prev = false;
                constexpr bool write_to_store     = true;
                check_experts<increment_nex_prev, write_to_store>
                    (store, nex_prev, it_compact, expert, it, iex, expert_found);
                if (it_compact + max(warp_size, n_expert_used) >= ne_store) {
                    ggml_cuda_syncwarp();
                    dump_store(ids_src1, ids_dst, n_expert_used, nchannels_y, sis1, write_inverse, store, nex_prev, n_dumped, it_compact);
                    it_compact = 0;
                }
            }
        }
        if (it_compact > 0) {
            ggml_cuda_syncwarp();
            dump_store(ids_src1, ids_dst, n_expert_used, nchannels_y, sis1, write_inverse, store, nex_prev, n_dumped, it_compact);
        }
    } else {
        // Implementation optimized for specific numbers of experts used:
        // a warp holds a whole number of token slots, so the slot count is padded to a power of 2
        constexpr int neu_padded = mm_ids_pow2<n_expert_used_template>::value;
        static_assert(neu_padded <= warp_size && warp_size % neu_padded == 0, "bad n_expert_used");
        const int iex = threadIdx.x % neu_padded; // The index at which the expert is used, if any.

        int it0 = 0;
        for (; it0 < n_tokens && it_compact + n_unroll*warp_size <= ne_store; it0 += n_unroll*warp_size/neu_padded) {
#pragma unroll
            for (int it1 = 0; it1 < n_unroll*warp_size/neu_padded; it1 += warp_size/neu_padded) {
                const int it = it0 + it1 + threadIdx.x / neu_padded;
                const int expert_found = (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ?
                    ids[it*si1 + iex] : INT_MAX;

                constexpr bool increment_nex_prev = true;
                constexpr bool write_to_store     = true;
                check_experts<increment_nex_prev, write_to_store>
                    (store, nex_prev, it_compact, expert, it, iex, expert_found);
            }
        }
        int it0_dumped = it0;
        for (; it0 < n_tokens; it0 += n_unroll*warp_size/neu_padded) {
#pragma unroll
            for (int it1 = 0; it1 < n_unroll*warp_size/neu_padded; it1 += warp_size/neu_padded) {
                const int it = it0 + it1 + threadIdx.x / neu_padded;
                const int expert_found = (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ?
                    ids[it*si1 + iex] : INT_MAX;

                constexpr bool increment_nex_prev = true;
                constexpr bool write_to_store     = false;
                check_experts<increment_nex_prev, write_to_store>
                    (store, nex_prev, it_compact, expert, it, iex, expert_found);
            }
        }

        nex_prev = warp_reduce_sum<warp_size>(nex_prev);
        ggml_cuda_syncwarp();
        dump_store(ids_src1, ids_dst, n_expert_used, nchannels_y, sis1, write_inverse, store, nex_prev, n_dumped, it_compact);
        it_compact = 0;

        for (it0 = it0_dumped; it0 < n_tokens; it0 += n_unroll*warp_size/neu_padded) {
#pragma unroll
            for (int it1 = 0; it1 < n_unroll*warp_size/neu_padded; it1 += warp_size/neu_padded) {
                const int it = it0 + it1 + threadIdx.x / neu_padded;
                const int expert_found = (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ?
                    ids[it*si1 + iex] : INT_MAX;

                constexpr bool increment_nex_prev = false;
                constexpr bool write_to_store     = true;
                check_experts<increment_nex_prev, write_to_store>
                    (store, nex_prev, it_compact, expert, it, iex, expert_found);
                if (it_compact + warp_size >= ne_store) {
                    ggml_cuda_syncwarp();
                    dump_store(ids_src1, ids_dst, n_expert_used, nchannels_y, sis1, write_inverse, store, nex_prev, n_dumped, it_compact);
                    it_compact = 0;
                }
            }
        }
        if (it_compact > 0) {
            ggml_cuda_syncwarp();
            dump_store(ids_src1, ids_dst, n_expert_used, nchannels_y, sis1, write_inverse, store, nex_prev, n_dumped, it_compact);
        }
    }

    if (threadIdx.x != 0) {
        return;
    }

    mm_ids_write_tile_spec(tile_spec, expert, nex_prev, n_dumped, tile_width, n_tokens, n_expert_used);
    if (expert_bounds) {
        // block 06 (MMB): per-expert compact-range bounds, [n_experts + 1] int32, a second reader of this
        // pass so MMQ/MMF (tile_spec) and MMB (bounds) share one compaction.
        expert_bounds[expert] = nex_prev;
        if (expert == static_cast<int>(gridDim.x) - 1) {
            expert_bounds[gridDim.x] = nex_prev + n_dumped;
        }
    }
}

// Replace the `n_experts` one-warp expert blocks of the helper with ONE 1024-thread block for the
// 512-experts / 10-used shape (RDNA3_5 source of record + RDNA4): one bucket-sort pass over
// n_tokens*n_expert_used slots with per-expert counters in shared memory, then the same tile_spec writer.
// Measured gfx1201: prefill +4.5 % pp16384 / +5.2 % pp2048, decode unaffected (the helper feeds the mmq
// prefill path only), same-seed text byte-identical.  Kill switch: GGML_CUDA_DISABLE_MMID_512.
//
// An expert id may repeat within a token's row: every occurrence is an independent destination column and
// gets its own compact row, which is why the scatter is per (token, iex) slot and not per token.
static __global__ void __launch_bounds__(1024) mm_ids_helper_512_10(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst,
        ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int nchannels_y,
        const uint3 tile_width, const int si1, const int sis1, const bool write_inverse) {
    constexpr int n_experts    = 512;
    constexpr int n_expert_used = 10;

    __shared__ int counts[n_experts];
    __shared__ int cursors[n_experts];

    const int tid = threadIdx.x;
    if (tid < n_experts) {
        counts[tid]  = 0;
        cursors[tid] = 0;
    }
    __syncthreads();

    const int n_slots = n_tokens * n_expert_used;
    for (int idx = tid; idx < n_slots; idx += blockDim.x) {
        const int token = idx / n_expert_used;
        const int iex   = idx - token * n_expert_used;
        atomicAdd(&counts[ids[token * si1 + iex]], 1);
    }
    __syncthreads();

    // The expert offsets are cumulative (thread 0 only: counts[] is read and rewritten in place).
    if (tid == 0) {
        int nex_prev = 0;
        for (int e = 0; e < n_experts; ++e) {
            const int c = counts[e];
            counts[e] = nex_prev;
            nex_prev += c;
        }
    }
    __syncthreads();

    for (int idx = tid; idx < n_slots; idx += blockDim.x) {
        const int token  = idx / n_expert_used;
        const int iex    = idx - token * n_expert_used;
        const int expert = ids[token * si1 + iex];
        const int dst    = counts[expert] + atomicAdd(&cursors[expert], 1);
        ids_dst[dst] = idx;
        if (write_inverse) {
            ids_src1[idx] = dst;
        } else {
            ids_src1[dst] = token * sis1 + iex % nchannels_y;
        }
    }
    __syncthreads();

    if (tid < n_experts) {
        mm_ids_write_tile_spec(tile_spec, tid, counts[tid], cursors[tid], tile_width, n_tokens, n_expert_used);
        if (expert_bounds) {
            // block 06 (MMB): counts[] holds the compact-row start, cursors[] the occurrence count
            expert_bounds[tid] = counts[tid];
            if (tid == n_experts - 1) {
                expert_bounds[n_experts] = counts[n_experts - 1] + cursors[n_experts - 1];
            }
        }
    }

}

template <int n_expert_used_template>
static void launch_mm_ids_helper(
        const int32_t * __restrict__ ids,
        int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec,
        int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int tile_width,
        const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    GGML_ASSERT(n_tokens          < (1 << 22) && "too few bits in mm_ids_helper_store");
    GGML_ASSERT(n_expert_used_var < (1 << 10) && "too few bits in mm_ids_helper_store");

    const int id = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
    CUDA_SET_SHARED_MEMORY_LIMIT(mm_ids_helper<n_expert_used_template>, smpbo);

    const uint3 tile_width_fd = init_fastdiv_values(tile_width);

    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, 1, 1);
    const size_t ne_shared_pad = n_expert_used_template == 0 ? std::max(n_expert_used_var, warp_size) : n_unroll*warp_size;
    const size_t nbytes_store_core = std::min(smpbo - ne_shared_pad * sizeof(mm_ids_helper_store), n_tokens * sizeof(mm_ids_helper_store));
    const size_t nbytes_shared = nbytes_store_core + ne_shared_pad * sizeof(mm_ids_helper_store);
    const size_t ne_store = nbytes_shared / sizeof(mm_ids_helper_store);
    mm_ids_helper<n_expert_used_template><<<num_blocks, block_size, nbytes_shared, stream>>>
        (ids, ids_src1, ids_dst, tile_spec, expert_bounds, n_tokens, n_expert_used_var, nchannels_y, tile_width_fd, si1, sis1, write_inverse, ne_store);
}

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * __restrict__ ids,
        int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec,
        int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int tile_width,
        const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    // The 512-expert / 10-used fast path (block 00): one 1024-thread block instead of 512 one-warp expert
    // blocks.  RDNA3_5 (source of record) + RDNA4 only -- the shape is a qwen4exp host-expert model's.
    {
        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        static const bool mmid_512_disabled = getenv("GGML_CUDA_DISABLE_MMID_512") != nullptr;
        if (!mmid_512_disabled && (GGML_CUDA_CC_IS_RDNA3_5(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) &&
                n_experts == 512 && n_expert_used == 10) {
            mm_ids_helper_512_10<<<1, 1024, 0, stream>>>(
                ids, ids_src1, ids_dst, tile_spec, expert_bounds, n_tokens, nchannels_y, init_fastdiv_values(tile_width),
                si1, sis1, write_inverse);
            return;
        }
    }

    switch (n_expert_used) {
        case  2:
            launch_mm_ids_helper< 2>(ids, ids_src1, ids_dst, tile_spec, expert_bounds,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
        case  4:
            launch_mm_ids_helper< 4>(ids, ids_src1, ids_dst, tile_spec, expert_bounds,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
        case  6:
            launch_mm_ids_helper< 6>(ids, ids_src1, ids_dst, tile_spec, expert_bounds,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
        case  8:
            launch_mm_ids_helper< 8>(ids, ids_src1, ids_dst, tile_spec, expert_bounds,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
        case 10:
            launch_mm_ids_helper<10>(ids, ids_src1, ids_dst, tile_spec, expert_bounds,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
        case 16:
            launch_mm_ids_helper<16>(ids, ids_src1, ids_dst, tile_spec, expert_bounds,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
        case 32:
            launch_mm_ids_helper<32>(ids, ids_src1, ids_dst, tile_spec, expert_bounds,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
        default:
            launch_mm_ids_helper< 0>(ids, ids_src1, ids_dst, tile_spec, expert_bounds,
                n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream);
            break;
    }
}
