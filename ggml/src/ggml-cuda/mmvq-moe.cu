#include "mmvq-moe.cuh"
#include "mmvq-vecdot.cuh"
#include "mmvq.cuh"
#include "moe-expert-cache.h"
#include "unary.cuh"
#include "vecdotq.cuh"

#include <algorithm>
#include <cstdint>

// Dedicated MoE multi-token kernel.
// Grid: (ceil(nrows_x / c_rows_per_block), nchannels_dst)
// Block: (warp_size, ncols_dst) - each warp handles one token independently.
// No shared memory reduction needed since each warp works alone.
// The block is (warp_size, ncols_dst) and ncols_dst spans the whole decode/verify band, and this
// kernel is the one that keeps the band on a single accumulation order (see the has_ids branch in
// mul_mat_vec_q_switch_ncols_dst), so its launch bound must cover the band rather than the per-type
// cap -- otherwise a launch with ncols_dst > cap fails outright.
template <ggml_type type, int c_rows_per_block, bool has_fusion = false>
__launch_bounds__(MMVQ_MOE_MAX_BATCH_SIZE*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_moe(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion,
        float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride, const uint32_t nchannels_dst,
        const void * cold_base, const uint32_t n_res_cold,
        const uint32_t cold_channel_stride, const uint32_t cold_row_stride,
        const void * cold_gate_base, const uint32_t cold_gate_n_res,
        const uint32_t cold_gate_channel_stride, const uint32_t cold_gate_row_stride) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type, true);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type, true);

    const bool shared_expert = has_fusion && fusion.shared_up && blockIdx.y == gridDim.y - 1;
    if (shared_expert) {
        vx = fusion.shared_up;
        dst = fusion.shared_dst;
        stride_col_dst = fusion.shared_stride_col_dst;
    }

    // fuse gate, bias, scales, and glu_op into the up projection
    bool use_gate = false;
    const void  * vgate      = nullptr;
    const float * x_bias     = nullptr;
    const float * gate_bias  = nullptr;
    const float * x_scale    = nullptr;
    const float * gate_scale = nullptr;
    ggml_glu_op   active_glu = GGML_GLU_OP_SWIGLU;
    float         glu_limit  = 0.0f;

    if constexpr (has_fusion) {
        use_gate   = fusion.gate != nullptr;
        vgate      = shared_expert ? fusion.shared_gate : fusion.gate;
        x_bias     = (const float *) fusion.x_bias;
        gate_bias  = (const float *) fusion.gate_bias;
        active_glu = fusion.glu_op;
        glu_limit  = fusion.glu_limit;
        if constexpr (type == GGML_TYPE_NVFP4) {
            x_scale    = (const float *) fusion.x_scale;
            gate_scale = (const float *) fusion.gate_scale;
        }
        // Per-(expert, token) MoE scale (down x topk weights). Indexed by
        // channel_dst + token_idx*nchannels_dst, one scalar per (expert, token).
        if (fusion.x_scale_channel_dst) {
            x_scale = (const float *) fusion.x_scale;
        }
    }

    const uint32_t token_idx   = threadIdx.y;
    const int      row0        = c_rows_per_block*blockIdx.x;
    if (row0 >= (int) nrows_x) {
        return;   // the padding block of ggml_cuda_grid_stall_pad (never block 0)
    }
    const int      blocks_per_row_x = ncols_x / qk;

    const uint32_t channel_dst = shared_expert ? 0 : blockIdx.y;

    if (token_idx >= ncols_dst) {
        return;
    }

    ggml_cuda_pdl_sync();
    // The shared-expert lane reads its own table at channel 0; its routing id is not in `ids`,
    // so the cache slot lookup and the UVA cold seam apply to the routed lanes only.
    const int32_t raw_id = shared_expert ? 0 : ids[channel_dst + token_idx * ids_stride];
    // The cache serves the routed lane through a materialized slot-remapped id (or the raw id for an
    // identity table); the shared-expert lane reads its own table at channel 0.
    const uint32_t channel_x_raw = (uint32_t) raw_id;
    // Phase 1b/3 UVA cold read.  An id >= n_res_cold is not a VRAM slot but host expert `id - n_res_cold`,
    // read in place from the pinned host alias.  The host geometry (per-expert stride `cold_channel_stride`,
    // per-row stride `cold_row_stride`) is returned by `moe_cache_get_cold`: for an unsplit or contiguous
    // axis-1 slice the row stride matches the arena (`cold_row_stride == 0`), while a strided axis-0 slice
    // uses the host pitch.  The default (cold_base == nullptr, n_res_cold == 0) leaves the resident-only
    // 1a contract untouched.
    const bool     cold      = !shared_expert && cold_base != nullptr && channel_x_raw >= n_res_cold;
    const void *   vx_use    = cold ? cold_base : vx;
    const uint32_t channel_x = shared_expert ? 0u : (cold ? (channel_x_raw - n_res_cold) : channel_x_raw);
    const uint32_t stride_channel_x_use = cold ? cold_channel_stride : stride_channel_x;
    const uint32_t stride_row_x_use     = (cold && cold_row_stride != 0) ? cold_row_stride : stride_row_x;
    // The fused gate lane is a SEPARATE expert table with its own host geometry (same raw routing id,
    // but its own slot count and pinned host master), so resolve it independently of the up lane.
    const bool     gate_cold        = !shared_expert && cold_gate_base != nullptr && channel_x_raw >= cold_gate_n_res;
    const void *   vgate_use        = gate_cold ? cold_gate_base : vgate;
    const uint32_t gate_channel_x   = shared_expert ? 0u : (gate_cold ? (channel_x_raw - cold_gate_n_res) : channel_x_raw);
    const uint32_t gate_channel_x_st = gate_cold ? cold_gate_channel_stride : stride_channel_x;
    const uint32_t gate_row_x_st     = (gate_cold && cold_gate_row_stride != 0) ? cold_gate_row_stride : stride_row_x;
    const uint32_t channel_y = fastmodulo(channel_dst, nchannels_y);

    const block_q8_1 * y = ((const block_q8_1 *) vy) + channel_y*stride_channel_y + token_idx*stride_col_y;
    const int kbx_offset      = channel_x*stride_channel_x_use + row0*stride_row_x_use;
    const int kbx_offset_gate = gate_channel_x*gate_channel_x_st + row0*gate_row_x_st;

    // partial sum for each thread
    float tmp[c_rows_per_block] = {0.0f};
    float tmp_gate[c_rows_per_block] = {0.0f};

    // split (row, kblock) items across the 8 thread groups: the K-split loop
    // leaves most groups idle on short-K MoE GEMMs (down K=512 -> 2 K-blocks)
    //
    // Row-loop clamp: the last row-block may hold fewer than `c_rows_per_block` valid rows, but the loop
    // below used to read all of them (only the write was guarded).  The dead loads walk `rpb - rem` rows
    // past the expert's last row -- into the next expert, and off the end of the tensor for the last
    // expert of the last host tensor (a cold read sources the pinned master in place).  Clamping the item
    // count skips them and leaves the valid-row accumulation order unchanged.
    const int rows_left  = row0 < (int) nrows_x ? (int) nrows_x - row0 : 0;
    const int rows_valid = rows_left < c_rows_per_block ? rows_left : c_rows_per_block;
    const int n_items    = rows_valid * blocks_per_row_x;
    const int n_groups = warp_size / (qi/vdr);
    const int kqs      = vdr * (threadIdx.x % (qi/vdr));
    // IQ2_XS is ALU/latency bound here: two items in flight per thread (-9 %); the other types lose from it
#pragma unroll type == GGML_TYPE_IQ2_XS ? 2 : 1
    for (int it = threadIdx.x / (qi/vdr); it < n_items; it += n_groups) {
        const int i   = it / blocks_per_row_x;
        const int kbx = it % blocks_per_row_x;
        const int kby = kbx * (qk/QK8_1);
        tmp[i] += vec_dot_q_cuda(vx_use, &y[kby], kbx_offset + i*stride_row_x_use + kbx, kqs);
        if constexpr (has_fusion) {
            if (use_gate) {
                tmp_gate[i] += vec_dot_q_cuda(vgate_use, &y[kby], kbx_offset_gate + i*gate_row_x_st + kbx, kqs);
            }
        }
    }

    ggml_cuda_pdl_lc();

    // Warp-level reduction only - no shared memory needed
#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
        if constexpr (has_fusion) {
            if (use_gate) {
                tmp_gate[i] = warp_reduce_sum<warp_size>(tmp_gate[i]);
            }
        }
    }

    // Write results
    if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
        float result = tmp[threadIdx.x];
        if constexpr (has_fusion) {
            const uint32_t bias_idx = channel_x*stride_channel_dst + row0 + threadIdx.x;

            if constexpr (type == GGML_TYPE_NVFP4) {
                if (x_scale) {
                    result *= x_scale[channel_x];
                }
            }
            if (fusion.x_scale_channel_dst) {
                result *= x_scale[channel_dst + token_idx * nchannels_dst];
            }
            if (x_bias) {
                result += x_bias[bias_idx];
            }
            if (use_gate) {
                float gate_value = tmp_gate[threadIdx.x];
                if constexpr (type == GGML_TYPE_NVFP4) {
                    if (gate_scale) {
                        gate_value *= gate_scale[channel_x];
                    }
                }
                if (gate_bias) {
                    gate_value += gate_bias[bias_idx];
                }
                switch (active_glu) {
                    case GGML_GLU_OP_SWIGLU:
                        result *= ggml_cuda_op_silu_single(gate_value);
                        break;
                    case GGML_GLU_OP_GEGLU:
                        result *= ggml_cuda_op_gelu_single(gate_value);
                        break;
                    case GGML_GLU_OP_SWIGLU_OAI:
                        result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                        break;
                    case GGML_GLU_OP_SWIGLU_CLAMP:
                        result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, glu_limit);
                        break;
                    default:
                        result = result * gate_value;
                        break;
                }
            }
        }
        dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = result;
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, tmp_gate, vgate, x_bias, gate_bias, active_glu, glu_limit, x_scale, gate_scale);
    } else if constexpr (type != GGML_TYPE_NVFP4) {
        GGML_UNUSED_VARS(x_scale, gate_scale);
    }
}

template <ggml_type type>
void mul_mat_vec_q_moe_launch(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride,
        const int warp_size, const int nchannels_dst, cudaStream_t stream) {

    // Phase 1b/3 UVA cold seam.  The slot-remap consumer redirects the op's `src0` to the cache arena, so
    // `vx` IS the arena base here; look the table up by it and, when it serves cold experts
    // (MOE_EXPERT_CACHE_COLD=uva), pass the pinned host alias + resident slot count + host geometry to the
    // kernel.  The default lookup misses and the kernel gets (nullptr, 0, 0, 0): the resident-only 1a
    // contract, unchanged.  The fused gate lane is a different table/arena, so resolve its cold geometry
    // too (without it the fused kernel would index the compact gate arena with a cold (host) expert id).
    void *  cold_base  = nullptr;
    int64_t n_res_cold = 0;
    size_t  cold_channel_bytes = 0;
    size_t  cold_row_bytes     = 0;
    void *  cold_gate_base     = nullptr;
    int64_t cold_gate_n_res    = 0;
    size_t  cold_gate_channel_bytes = 0;
    size_t  cold_gate_row_bytes     = 0;
    if (moe_cache_enabled()) {
        (void) moe_cache_get_cold(vx, &cold_base, &n_res_cold, &cold_channel_bytes, &cold_row_bytes);
        if (cold_base != nullptr && fusion.gate != nullptr) {
            (void) moe_cache_get_cold(fusion.gate, &cold_gate_base, &cold_gate_n_res,
                                      &cold_gate_channel_bytes, &cold_gate_row_bytes);
        }
    }
    // The kernel indexes in type-block units; the cold geometry is reported in bytes.
    const size_t   ts = ggml_type_size(type);
    const uint32_t cold_channel_stride      = cold_channel_bytes      > 0 ? (uint32_t) (cold_channel_bytes      / ts) : 0;
    const uint32_t cold_row_stride          = cold_row_bytes          > 0 ? (uint32_t) (cold_row_bytes          / ts) : 0;
    const uint32_t cold_gate_channel_stride = cold_gate_channel_bytes > 0 ? (uint32_t) (cold_gate_channel_bytes / ts) : 0;
    const uint32_t cold_gate_row_stride     = cold_gate_row_bytes     > 0 ? (uint32_t) (cold_gate_row_bytes     / ts) : 0;

    // rows_per_block scaled for short-K expert GEMMs: the item-split loop in
    // mul_mat_vec_q_moe spreads (row, kblock) items over the 8 thread groups,
    // so pick rows_per_block to fill them (e.g. the down K=512 -> rpb=4).
    const int blocks_per_row_x = ncols_x / ggml_cuda_type_traits<type>::qk;
    int rpb = 2;
    if (blocks_per_row_x > 0 && blocks_per_row_x < 8) {
        rpb = std::min((8 + blocks_per_row_x - 1) / blocks_per_row_x, 8);
    }
    // mul_mat_vec_q_moe is only instantiated for RPB 2/4/8, and nblocks_rows below must be sized with
    // the value the kernel actually launches with.  ceil(8/blocks_per_row_x) is 3 for
    // blocks_per_row_x == 3 (k == 3*qk), which armed the grid for RPB 3 and then fell through the
    // switch to the RPB 2 default: only the first 2/3 of the rows were computed and the rest of the
    // output was left unwritten (test-backend-ops MUL_MAT_ID, m=64/n=16/k=96, ERR ~0.5).  Snap to a
    // supported value before sizing the grid.
    if (rpb != 4 && rpb != 8) {
        rpb = 2;
    }
    // RDNA4: a short-K expert GEMM (fewer than 8 K-blocks per row, e.g. the Flash-Next down projection split under
    // -sm tensor, K = 320) gives each lane at most one dot product at 2 rows per block, so the warps are too short
    // to stream: 8 rows per block measured q2_0 43.0 -> 29.8 us per 4-token verify call (gfx1201, 16/32 rows were
    // slower again: one full-warp reduction per row), same greedy output.  GGML_CUDA_MOE_RPB_SHORT=0 restores the
    // ceil(8 / K-blocks) rule; 2/4/8 force a value.
    if (blocks_per_row_x > 0 && blocks_per_row_x < 8) {
        static const int rpb_env = getenv("GGML_CUDA_MOE_RPB_SHORT") ? atoi(getenv("GGML_CUDA_MOE_RPB_SHORT")) : -1;
        if (rpb_env == 2 || rpb_env == 4 || rpb_env == 8) {
            rpb = rpb_env;
        } else if (rpb_env != 0 && GGML_CUDA_CC_IS_RDNA4(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) {
            rpb = 8;
        }
    }
    const int64_t nblocks_rows = (nrows_x + rpb - 1) / rpb;
    // Upstream's fused shared-expert lane is the last blockIdx.y column.
    dim3 block_nums(nblocks_rows, nchannels_dst + (fusion.shared_up != nullptr));
    const dim3 block_dims(warp_size, ncols_dst);
    if (ggml_cuda_grid_stall_pad(block_nums, block_dims)) {
        block_nums.x++;   // the kernel returns for row0 >= nrows_x
    }
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr || fusion.x_scale_channel_dst;

    const auto launch = [&](auto rpb_c, auto fusion_c) {
        constexpr int  RPB   = decltype(rpb_c)::value;
        constexpr bool CFUSE = decltype(fusion_c)::value;
        ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, RPB, CFUSE>, launch_params,
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride, nchannels_dst,
            cold_base, (uint32_t) n_res_cold, cold_channel_stride, cold_row_stride,
            cold_gate_base, (uint32_t) cold_gate_n_res, cold_gate_channel_stride, cold_gate_row_stride);
    };
    const auto launch_rpb = [&](auto rpb_c) {
        if (has_fusion) {
            launch(rpb_c, std::true_type{});
        } else {
            launch(rpb_c, std::false_type{});
        }
    };
    switch (rpb) {
        case 2:  launch_rpb(std::integral_constant<int, 2>{});  break;
        case 4:  launch_rpb(std::integral_constant<int, 4>{});  break;
        case 8:  launch_rpb(std::integral_constant<int, 8>{});  break;
        default: launch_rpb(std::integral_constant<int, 2>{});  break;
    }
}

// The dense dispatch (mmvq.cu) instantiates the surrounding per-type switch, so every
// type it can reach must have its MoE launch specialization available here.
#define GGML_CUDA_MMVQ_MOE_INSTANTIATE(TYPE) \
    template void mul_mat_vec_q_moe_launch<TYPE>( \
        const void *, const void *, const int32_t *, const ggml_cuda_mm_fusion_args_device, float *, \
        const uint32_t, const uint3, const uint32_t, const uint32_t, const uint32_t, const uint32_t, \
        const uint32_t, const uint32_t, const uint32_t, const uint32_t, const uint32_t, const int, \
        const int, cudaStream_t);
GGML_CUDA_MMVQ_MOE_INSTANTIATE(GGML_TYPE_Q1_0)
GGML_CUDA_MMVQ_MOE_INSTANTIATE(GGML_TYPE_Q2_0)
GGML_CUDA_MMVQ_MOE_INSTANTIATE(GGML_TYPE_Q4_0)
GGML_CUDA_MMVQ_MOE_INSTANTIATE(GGML_TYPE_Q4_1)
GGML_CUDA_MMVQ_MOE_INSTANTIATE(GGML_TYPE_Q5_0)
GGML_CUDA_MMVQ_MOE_INSTANTIATE(GGML_TYPE_Q5_1)
GGML_CUDA_MMVQ_MOE_INSTANTIATE(GGML_TYPE_Q8_0)
GGML_CUDA_MMVQ_MOE_INSTANTIATE(GGML_TYPE_MXFP4)
GGML_CUDA_MMVQ_MOE_INSTANTIATE(GGML_TYPE_NVFP4)
GGML_CUDA_MMVQ_MOE_INSTANTIATE(GGML_TYPE_Q2_K)
GGML_CUDA_MMVQ_MOE_INSTANTIATE(GGML_TYPE_Q3_K)
GGML_CUDA_MMVQ_MOE_INSTANTIATE(GGML_TYPE_Q4_K)
GGML_CUDA_MMVQ_MOE_INSTANTIATE(GGML_TYPE_Q5_K)
GGML_CUDA_MMVQ_MOE_INSTANTIATE(GGML_TYPE_Q6_K)
GGML_CUDA_MMVQ_MOE_INSTANTIATE(GGML_TYPE_IQ2_XXS)
GGML_CUDA_MMVQ_MOE_INSTANTIATE(GGML_TYPE_IQ2_XS)
GGML_CUDA_MMVQ_MOE_INSTANTIATE(GGML_TYPE_IQ2_S)
GGML_CUDA_MMVQ_MOE_INSTANTIATE(GGML_TYPE_IQ3_XXS)
GGML_CUDA_MMVQ_MOE_INSTANTIATE(GGML_TYPE_IQ1_S)
GGML_CUDA_MMVQ_MOE_INSTANTIATE(GGML_TYPE_IQ1_M)
GGML_CUDA_MMVQ_MOE_INSTANTIATE(GGML_TYPE_IQ4_NL)
GGML_CUDA_MMVQ_MOE_INSTANTIATE(GGML_TYPE_IQ4_XS)
GGML_CUDA_MMVQ_MOE_INSTANTIATE(GGML_TYPE_IQ3_S)
#undef GGML_CUDA_MMVQ_MOE_INSTANTIATE
