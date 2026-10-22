#include "ggml.h"
#include "common.cuh"
#include "unary.cuh"
#include "mmvf.cuh"
#include "mmf.cuh"
#include "convert.cuh"
#include "fwht.cuh"

template <typename T, typename type_acc, int ncols_dst, int block_size, bool has_fusion = false, bool is_multi_token_id = false,
          bool gate_beta = false>
static __global__ void mul_mat_vec_f(
        const T * x_ptr, const float * y_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion, float * dst_ptr,
        const int ncols2, const uint3 nchannels_y, const int stride_row, const int stride_col_y2, const int stride_col_dst,
        const uint3 channel_ratio, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const uint3 sample_ratio, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, const int nrows) {
    // gate_beta: the second half of the rows is the beta projection (own weights and output, same shapes)
    const bool gb_beta = gate_beta && (int) blockIdx.x >= fusion.gb_nrows;
    const T       * GGML_CUDA_RESTRICT x   = gb_beta ? (const T *) fusion.gb_x2 : x_ptr;
    const float   * GGML_CUDA_RESTRICT y   = y_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = gb_beta ? fusion.gb_dst2 : dst_ptr;
    if ((int) blockIdx.x >= nrows) {
        return;   // the padding block of a HIP launch moved off a stalling grid size (see the launch)
    }
    const int row         = gb_beta ? (int) blockIdx.x - fusion.gb_nrows : (int) blockIdx.x;
    // for MUL_MAT_ID - blockIdx.y = n_expert_used, blockIdx.z = ncols_dst (tokens)
    const int channel_dst = blockIdx.y;
    const int tid         = threadIdx.x;

    int token_idx;
    int channel_x;
    int channel_y;
    int sample_dst;

    ggml_cuda_pdl_sync();
    if constexpr (is_multi_token_id) {
        // Multi-token MUL_MAT_ID path, adding these in the normal path causes a perf regression for n_tokens=1 case
        token_idx  = blockIdx.z;
        channel_x  = ids[channel_dst + token_idx * ids_stride];
        channel_y  = fastmodulo(channel_dst, nchannels_y);
        sample_dst = 0;
    } else {
        token_idx  = ids ? blockIdx.z                                          : 0;
        channel_x  = ids ? ids[blockIdx.y + token_idx * ids_stride]            : fastdiv((uint32_t) channel_dst, channel_ratio);
        channel_y  = ids ? fastmodulo(blockIdx.y, nchannels_y)                 : channel_dst;
        sample_dst = ids ? 0                                                   : blockIdx.z;
    }

    const int sample_x    = fastdiv((uint32_t) sample_dst, sample_ratio);
    const int sample_y    = sample_dst;

    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();

    x   += int64_t(sample_x)  *stride_sample_x   + channel_x  *stride_channel_x   + row*stride_row;
    y   += int64_t(sample_y)  *stride_sample_y   + channel_y  *stride_channel_y;
    dst += int64_t(sample_dst)*stride_sample_dst + channel_dst*stride_channel_dst;
    if constexpr (is_multi_token_id) {
        y   += token_idx*stride_col_y2*2;
        dst += token_idx*stride_col_dst;
    }

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    ggml_glu_op glu_op = ggml_glu_op::GGML_GLU_OP_SWIGLU;
    float glu_limit = 0.0f;
    const T * gate_x = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;

    if constexpr (has_fusion) {
        use_gate = fusion.gate != nullptr;
        use_bias = fusion.x_bias != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr;
        glu_op = fusion.glu_op;
        glu_limit = fusion.glu_limit;

        if (use_gate) {
            gate_x = static_cast<const T *>(fusion.gate);
        }
        if (use_bias) {
            x_bias = static_cast<const float *>(fusion.x_bias);
        }
        if (use_gate_bias) {
            gate_bias = static_cast<const float *>(fusion.gate_bias);
            use_gate_bias = use_gate;
        } else {
            use_gate_bias = false;
        }
    }

    if (use_gate) {
        gate_x += int64_t(sample_x)  *stride_sample_x   + channel_x  *stride_channel_x   + row*stride_row;
    }

    if constexpr (has_fusion) {
        const int channel_bias = ids ? channel_x : channel_dst;
        if (use_bias) {
            x_bias += int64_t(sample_dst)*stride_sample_dst + channel_bias*stride_channel_dst;
        }
        if (use_gate_bias) {
            gate_bias += int64_t(sample_dst)*stride_sample_dst + channel_bias*stride_channel_dst;
        }
    }

    const float2 * y2 = (const float2 *) y;

    extern __shared__ char data_mmv[];
    float * buf_iw = (float *) data_mmv;
    [[maybe_unused]] float * buf_iw_gate = nullptr;
    if constexpr (has_fusion) {
        buf_iw_gate = (float *) (data_mmv + warp_size*sizeof(float));
    }

    if (block_size > warp_size) {
        if (tid < warp_size) {
            buf_iw[tid] = 0.0f;
            if constexpr (has_fusion) {
                if (use_gate) {
                    buf_iw_gate[tid] = 0.0f;
                }
            }
        }
        __syncthreads();
    }

    float sumf[ncols_dst] = {0.0f};
    float sumf_gate[ncols_dst];
    if constexpr (has_fusion) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            sumf_gate[j] = 0.0f;
        }
    }

    if constexpr (std::is_same_v<T, float>) {
        const float2 * x2 = (const float2 *) x;
        [[maybe_unused]] const float2 * gate_x2 = nullptr;
        if constexpr (has_fusion) {
            if (use_gate) {
                gate_x2 = (const float2 *) gate_x;
            }
        }

#pragma unroll 4
        for (int col2 = tid; col2 < ncols2; col2 += block_size) {
            const float2 tmpx = x2[col2];
            float2 tmpx_gate = make_float2(0.0f, 0.0f);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmpx_gate = gate_x2[col2];
                }
            }

#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                const float2 tmpy = y2[j*stride_col_y2 + col2];
                ggml_cuda_mad(sumf[j], tmpx.x, tmpy.x);
                ggml_cuda_mad(sumf[j], tmpx.y, tmpy.y);

                if constexpr (has_fusion) {
                    if (use_gate) {
                        ggml_cuda_mad(sumf_gate[j], tmpx_gate.x, tmpy.x);
                        ggml_cuda_mad(sumf_gate[j], tmpx_gate.y, tmpy.y);
                    }
                }
            }
        }
    } else if constexpr (std::is_same_v<T, half>) {
        const half2 * x2 = (const half2 *) x;
        [[maybe_unused]] const half2 * gate_x2 = nullptr;
        if constexpr (has_fusion) {
            if (use_gate) {
                gate_x2 = (const half2 *) gate_x;
            }
        }

        if (std::is_same_v<type_acc, float>) {
            for (int col2 = tid; col2 < ncols2; col2 += block_size) {
                const float2 tmpx = __half22float2(x2[col2]);
                float2 tmpx_gate = make_float2(0.0f, 0.0f);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmpx_gate = __half22float2(gate_x2[col2]);
                    }
                }
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    const float2 tmpy = y2[j*stride_col_y2 + col2];
                    ggml_cuda_mad(sumf[j], tmpx.x, tmpy.x);
                    ggml_cuda_mad(sumf[j], tmpx.y, tmpy.y);

                    if constexpr (has_fusion) {
                        if (use_gate) {
                            ggml_cuda_mad(sumf_gate[j], tmpx_gate.x, tmpy.x);
                            ggml_cuda_mad(sumf_gate[j], tmpx_gate.y, tmpy.y);
                        }
                    }
                }
            }
        } else {
#ifdef FP16_AVAILABLE
            half2 sumh2[ncols_dst] = {{0.0f, 0.0f}};
            half2 sumh2_gate[ncols_dst] = {{0.0f, 0.0f}};

            for (int col2 = tid; col2 < ncols2; col2 += block_size) {
                const half2 tmpx = x2[col2];
                half2 tmpx_gate = make_half2(0.0f, 0.0f);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmpx_gate = gate_x2[col2];
                    }
                }
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    const float2 tmpy = y2[j*stride_col_y2 + col2];
                    sumh2[j] += tmpx * make_half2(tmpy.x, tmpy.y);

                    if constexpr (has_fusion) {
                        if (use_gate) {
                            sumh2_gate[j] += tmpx_gate * make_half2(tmpy.x, tmpy.y);
                        }
                    }
                }
            }

#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                sumf[j] = __low2float(sumh2[j]) + __high2float(sumh2[j]);
            }

            if constexpr (has_fusion) {
                if (use_gate) {
#pragma unroll
                    for (int j = 0; j < ncols_dst; ++j) {
                        sumf_gate[j] = __low2float(sumh2_gate[j]) + __high2float(sumh2_gate[j]);
                    }
                }
            }
#else
            NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
        }
    } else if constexpr (std::is_same_v<T, nv_bfloat16>) {
//TODO: add support for ggml_cuda_mad for hip_bfloat162
#if defined(GGML_USE_HIP)
        const int * x2 = (const int *) x;
        const int * gate_x2 = nullptr;
        if constexpr (has_fusion) {
            if (use_gate) {
                gate_x2 = (const int *) gate_x;
            }
        }
#pragma unroll 4
        for (int col2 = tid; col2 < ncols2; col2 += block_size) {
            const int tmpx = x2[col2];
            int tmpx_gate = 0;
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmpx_gate = gate_x2[col2];
                }
            }
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                const float2 tmpy = y2[j*stride_col_y2 + col2];
                const float tmpx0 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[0]);
                const float tmpx1 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[1]);
                ggml_cuda_mad(sumf[j], tmpx0, tmpy.x);
                ggml_cuda_mad(sumf[j], tmpx1, tmpy.y);

                if constexpr (has_fusion) {
                    if (use_gate) {
                        const float tmpx0_gate = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx_gate)[0]);
                        const float tmpx1_gate = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx_gate)[1]);
                        ggml_cuda_mad(sumf_gate[j], tmpx0_gate, tmpy.x);
                        ggml_cuda_mad(sumf_gate[j], tmpx1_gate, tmpy.y);
                    }
                }
            }
        }
#else
        const nv_bfloat162 * x2 = (const nv_bfloat162 *) x;
        [[maybe_unused]] const nv_bfloat162 * gate_x2 = nullptr;
        if constexpr (has_fusion) {
            if (use_gate) {
                gate_x2 = (const nv_bfloat162 *) gate_x;
            }
        }
        for (int col2 = tid; col2 < ncols2; col2 += block_size) {
            const nv_bfloat162 tmpx = x2[col2];
            [[maybe_unused]] nv_bfloat162 tmpx_gate;
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmpx_gate = gate_x2[col2];
                }
            }
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                const float2 tmpy = y2[j*stride_col_y2 + col2];
                ggml_cuda_mad(sumf[j], tmpx.x, tmpy.x);
                ggml_cuda_mad(sumf[j], tmpx.y, tmpy.y);

                if constexpr (has_fusion) {
                    if (use_gate) {
                        ggml_cuda_mad(sumf_gate[j], tmpx_gate.x, tmpy.x);
                        ggml_cuda_mad(sumf_gate[j], tmpx_gate.y, tmpy.y);
                    }
                }
            }
        }
#endif
    } else {
        static_assert(std::is_same_v<T, void>, "unsupported type");
    }

    ggml_cuda_pdl_lc();
    // multi-token: one cross-warp exchange for all columns instead of two barriers per column. Per column the
    // arithmetic is unchanged: the warp butterfly, then warp 0 butterflies the per-warp sums (0 past nwarps).
    constexpr bool batched_reduce = !has_fusion && ncols_dst > 1 && block_size > warp_size;
    if constexpr (batched_reduce) {
        __shared__ float buf_cols[ncols_dst][warp_size];
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);
            buf_cols[j][tid/warp_size] = sumf[j];
        }
        __syncthreads();
        if (tid < warp_size) {
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                sumf[j] = warp_reduce_sum<warp_size>(tid < block_size/warp_size ? buf_cols[j][tid] : 0.0f);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < (batched_reduce ? 0 : ncols_dst); ++j) {
        sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);

        if constexpr (has_fusion) {
            if (use_gate) {
                sumf_gate[j] = warp_reduce_sum<warp_size>(sumf_gate[j]);
            }
        }

        if (block_size > warp_size) {
            buf_iw[tid/warp_size] = sumf[j];
            if constexpr (has_fusion) {
                if (use_gate) {
                    buf_iw_gate[tid/warp_size] = sumf_gate[j];
                }
            }
            __syncthreads();
            if (tid < warp_size) {
                sumf[j] = buf_iw[tid];
                sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        sumf_gate[j] = buf_iw_gate[tid];
                        sumf_gate[j] = warp_reduce_sum<warp_size>(sumf_gate[j]);
                    }
                }
            }

            if (j < ncols_dst) {
                __syncthreads();
            }
        }
    }

    if (tid >= ncols_dst) {
        return;
    }

    float value = sumf[tid];

    if constexpr (has_fusion) {
        if (use_bias) {
            value += x_bias[tid*stride_col_dst + row];
        }

        if (use_gate) {
            float gate_value = sumf_gate[tid];
            if (use_gate_bias) {
                gate_value += gate_bias[tid*stride_col_dst + row];
            }
            switch (glu_op) {
                case GGML_GLU_OP_SWIGLU:
                    value *= ggml_cuda_op_silu_single(gate_value);
                    break;
                case GGML_GLU_OP_GEGLU:
                    value *= ggml_cuda_op_gelu_single(gate_value);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI: {
                    value = ggml_cuda_op_swiglu_oai_single(gate_value, value);
                    break;
                }
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    value = ggml_cuda_op_swiglu_clamp_single(gate_value, value, glu_limit);
                    break;
                default:
                    break;
            }
        }
    }

    if constexpr (gate_beta) {
        if (!gb_beta) {
            // add_softplus_mul_bcast_f32 (unary.cu) on the value the plain kernel would store: bit-identical
            const float xb = value + fusion.gb_dt[row];
            value = ((xb > 20.0f) ? xb : logf(1.0f + expf(xb))) * fusion.gb_a[row];
        }
    }

    dst[tid*stride_col_dst + row] = value;

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, glu_op, glu_limit, gate_x, x_bias, gate_bias, sumf_gate);
    }
}

// Short rows (few K iterations per thread) waste most of a multi-warp block on the cross-warp reduction and launch
// 5..8 waves per row. This kernel gives each row one warp that plays the vblock threads of mul_mat_vec_f: lane l
// accumulates virtual threads l, l + warp_size, ... in their own registers, in the same K order; each virtual warp's
// sum is the same butterfly, and the per-warp sums are combined by the same butterfly (0 past the virtual warps).
// So every output is bit-identical to mul_mat_vec_f<T, float, ncols_dst, vblock> (no fusion, no ids).
// Warp w of block b takes rows (b*rows_per_block + w) + k*gridDim.x*rows_per_block.
template <typename T, int ncols_dst, int vblock, int rows_per_block>
__launch_bounds__(rows_per_block*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_f_vb(
        const T * x_ptr, const float * y_ptr, float * dst_ptr, const int ncols2, const int nrows,
        const int stride_row, const int stride_col_y2, const int stride_col_dst,
        const uint3 channel_ratio, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const uint3 sample_ratio, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nv        = vblock / warp_size;
    static_assert(vblock % warp_size == 0 && nv > 1 && nv <= warp_size, "bad vblock");

    const int lane        = threadIdx.x % warp_size;
    const int channel_dst = blockIdx.y;
    const int sample_dst  = blockIdx.z;
    const int channel_x   = fastdiv((uint32_t) channel_dst, channel_ratio);
    const int sample_x    = fastdiv((uint32_t) sample_dst, sample_ratio);

    ggml_cuda_pdl_sync();
    const float2 * y2 = (const float2 *) (y_ptr + int64_t(sample_dst)*stride_sample_y + channel_dst*stride_channel_y);
    float * dst = dst_ptr + int64_t(sample_dst)*stride_sample_dst + channel_dst*stride_channel_dst;

    for (int row = blockIdx.x*rows_per_block + threadIdx.x/warp_size; row < nrows; row += gridDim.x*rows_per_block) {
        const T * x = x_ptr + int64_t(sample_x)*stride_sample_x + channel_x*stride_channel_x + int64_t(row)*stride_row;

        float sumf[nv][ncols_dst];
#pragma unroll
        for (int v = 0; v < nv; ++v) {
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                sumf[v][j] = 0.0f;
            }
        }

#pragma unroll
        for (int v = 0; v < nv; ++v) {
            if constexpr (std::is_same_v<T, float>) {
                const float2 * x2 = (const float2 *) x;
#pragma unroll 4
                for (int col2 = v*warp_size + lane; col2 < ncols2; col2 += vblock) {
                    const float2 tmpx = x2[col2];
#pragma unroll
                    for (int j = 0; j < ncols_dst; ++j) {
                        const float2 tmpy = y2[j*stride_col_y2 + col2];
                        ggml_cuda_mad(sumf[v][j], tmpx.x, tmpy.x);
                        ggml_cuda_mad(sumf[v][j], tmpx.y, tmpy.y);
                    }
                }
            } else {
                static_assert(std::is_same_v<T, nv_bfloat16>, "mul_mat_vec_f_vb: F32 and BF16 only");
                const int * x2 = (const int *) x;
#pragma unroll 4
                for (int col2 = v*warp_size + lane; col2 < ncols2; col2 += vblock) {
                    const int tmpx = x2[col2];
#pragma unroll
                    for (int j = 0; j < ncols_dst; ++j) {
                        const float2 tmpy = y2[j*stride_col_y2 + col2];
                        const float tmpx0 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[0]);
                        const float tmpx1 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[1]);
                        ggml_cuda_mad(sumf[v][j], tmpx0, tmpy.x);
                        ggml_cuda_mad(sumf[v][j], tmpx1, tmpy.y);
                    }
                }
            }
        }

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            float total = 0.0f;
#pragma unroll
            for (int v = 0; v < nv; ++v) {
                const float wsum = warp_reduce_sum<warp_size>(sumf[v][j]);
                total = lane == v ? wsum : total;
            }
            total = warp_reduce_sum<warp_size>(total);
            if (lane == j) {
                dst[j*stride_col_dst + row] = total;
            }
        }
    }
    ggml_cuda_pdl_lc();
}

template<typename T, typename type_acc, int ncols_dst, int block_size, bool is_multi_token_id = false>
static void mul_mat_vec_f_switch_fusion(
        const T * x, const float * y, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int64_t ncols, const uint3 nchannels_y,
        const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst,
        const uint3 channel_ratio, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const uint3 sample_ratio, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const dim3 & block_dims, const dim3 & block_nums, const int nbytes_shared, const int ids_stride, const int nrows, const cudaStream_t stream) {

    const ggml_cuda_kernel_launch_params launch_params = {block_nums, block_dims, nbytes_shared, stream};

    if (fusion.gb_x2 != nullptr) {
        if constexpr (!is_multi_token_id && std::is_same_v<T, nv_bfloat16> && std::is_same_v<type_acc, float>) {
            ggml_cuda_kernel_launch(mul_mat_vec_f<T, type_acc, ncols_dst, block_size, false, false, true>, launch_params,
                x, y, ids, fusion, dst, ncols, nchannels_y, stride_row, stride_col_y, stride_col_dst,
                channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, nrows);
            return;
        }
        GGML_ABORT("fatal error");
    }

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;
    if constexpr (ncols_dst == 1) {
        if (has_fusion) {
            ggml_cuda_kernel_launch(mul_mat_vec_f<T, type_acc, ncols_dst, block_size, true, is_multi_token_id>, launch_params,
                x, y, ids, fusion, dst, ncols, nchannels_y, stride_row, stride_col_y, stride_col_dst,
                channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, nrows);
            return;
       }
    }

    GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1");

    ggml_cuda_kernel_launch(mul_mat_vec_f<T, type_acc, ncols_dst, block_size, false, is_multi_token_id>, launch_params,
        x, y, ids, fusion, dst, ncols, nchannels_y, stride_row, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, nrows);

}

template <typename T, typename type_acc, int ncols_dst, bool is_multi_token_id = false>
void launch_mul_mat_vec_f_cuda(
        const T * x, const float * y, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int64_t ncols, const int64_t nrows,
        const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst,
        const int64_t nchannels_x, const int64_t nchannels_y, const int64_t nchannels_dst,
        const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst, const int64_t nsamples_x,
        const int64_t nsamples_dst, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
        const int64_t nsamples_or_ntokens, const int64_t ids_stride, cudaStream_t stream) {
    GGML_ASSERT(ncols        % 2 == 0);
    GGML_ASSERT(stride_row   % 2 == 0);
    GGML_ASSERT(stride_col_y % 2 == 0);
    GGML_ASSERT(ids || nchannels_dst % nchannels_x == 0);
    GGML_ASSERT(       nsamples_dst  % nsamples_x  == 0);
    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0) : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst  / nsamples_x);

    const int device = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[device].warp_size;

    int64_t block_size_best = warp_size;
    int64_t niter_best      = (ncols + 2*warp_size - 1) / (2*warp_size);
    int64_t max_block_size  = 256;
    if(ggml_cuda_info().devices[device].cc > GGML_CUDA_CC_OFFSET_AMD && ggml_cuda_info().devices[device].cc < GGML_CUDA_CC_RDNA1) {
        max_block_size = 128;
    }
    for (int64_t block_size = 2*warp_size; block_size <= max_block_size; block_size += warp_size) {
        const int64_t niter = (ncols + 2*block_size - 1) / (2*block_size);
        if (niter < niter_best) {
            niter_best      = niter;
            block_size_best = block_size;
        }
    }

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;

#if defined(GGML_USE_HIP)
    // RDNA4, short F32/BF16 rows (one K iteration per thread, K <= 512) and many of them: one warp per row through
    // mul_mat_vec_f_vb (bit-identical), 4 rows per block, at most 1792 warps in flight (the grid-size stall, see
    // GGML_MMVQ_RDNA4_DECODE_GRID). gfx1201 sweep (K 192..512 x 3072..10240 rows x 1..8 tokens): -1..-69 % while
    // (block_size/warp_size) * ncols_dst <= 36, losing up to 17 % above that; fewer rows lose up to 37 %.
    // E.g. the 10240 x 320 BF16 hyper-connection up-projection of Flash-Next GSQ: 23.6 -> 14.1 us per token.
    if constexpr (!is_multi_token_id && (std::is_same_v<T, float> || std::is_same_v<T, nv_bfloat16>) && std::is_same_v<type_acc, float>) {
        constexpr int vb_rpb = 4;
        constexpr int vb_max_warps = 1792;
        if (GGML_CUDA_CC_IS_RDNA4(ggml_cuda_info().devices[device].cc) && warp_size == 32 && ids == nullptr && !has_fusion &&
            fusion.gb_x2 == nullptr &&
            niter_best == 1 && nrows >= 3072 && (block_size_best/warp_size)*ncols_dst <= 36) {
            const int64_t nblk = (nrows + vb_rpb - 1) / vb_rpb;
            const dim3 vb_nums(std::min<int64_t>(nblk, vb_max_warps/vb_rpb), nchannels_dst, nsamples_or_ntokens);
            const ggml_cuda_kernel_launch_params vb_params = {vb_nums, dim3(vb_rpb*warp_size, 1, 1), 0, stream};
            const auto vb_launch = [&](auto vb_tag) {
                constexpr int c_vb = decltype(vb_tag)::value;
                if constexpr ((c_vb/32)*ncols_dst <= 36) {
                    ggml_cuda_kernel_launch(mul_mat_vec_f_vb<T, ncols_dst, c_vb, vb_rpb>, vb_params,
                        x, y, dst, (int) (ncols/2), (int) nrows, (int) stride_row, (int) (stride_col_y/2), (int) stride_col_dst,
                        channel_ratio_fd, (int) stride_channel_x, (int) stride_channel_y, (int) stride_channel_dst,
                        sample_ratio_fd, (int) stride_sample_x, (int) stride_sample_y, (int) stride_sample_dst);
                } else {
                    GGML_ABORT("fatal error");
                }
            };
            switch (block_size_best) {
                case  64: vb_launch(std::integral_constant<int,  64>{}); return;
                case  96: vb_launch(std::integral_constant<int,  96>{}); return;
                case 128: vb_launch(std::integral_constant<int, 128>{}); return;
                case 160: vb_launch(std::integral_constant<int, 160>{}); return;
                case 192: vb_launch(std::integral_constant<int, 192>{}); return;
                case 224: vb_launch(std::integral_constant<int, 224>{}); return;
                case 256: vb_launch(std::integral_constant<int, 256>{}); return;
                default: break;
            }
        }
    }
#endif // defined(GGML_USE_HIP)

    const int nbytes_shared = warp_size*sizeof(float) + (has_fusion ? warp_size*sizeof(float) : 0);
    dim3 block_nums(nrows, nchannels_dst, nsamples_or_ntokens);
    const dim3 block_dims(block_size_best, 1, 1);
    if (ggml_cuda_grid_stall_pad(block_nums, block_dims)) {
        block_nums.x++;   // the kernel returns for row >= nrows
    }
    switch (block_size_best) {
        case   32: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 32, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, nrows, stream);
        } break;
        case   64: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 64, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, nrows, stream);
        } break;
        case   96: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 96, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, nrows, stream);
        } break;
        case  128: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 128, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, nrows, stream);
        } break;
        case  160: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 160, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, nrows, stream);
        } break;
        case  192: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 192, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, nrows, stream);
        } break;
        case  224: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 224, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, nrows, stream);
        } break;
        case  256: {
            mul_mat_vec_f_switch_fusion<T, type_acc, ncols_dst, 256, is_multi_token_id>
                (x, y, ids, fusion, dst, ncols/2, nchannels_y_fd, stride_row, stride_col_y/2, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst, block_dims, block_nums, nbytes_shared, ids_stride, nrows, stream);
        } break;
        default:
            GGML_ABORT("fatal error");
    }
}

template <typename T, typename type_acc>
static void mul_mat_vec_f_cuda_switch_ncols_dst(
        const T * x, const float * y, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int64_t ncols, const int64_t nrows, const int64_t ncols_dst,
        const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst,
        const int64_t nchannels_x, const int64_t nchannels_y, const int64_t nchannels_dst,
        const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst, const int64_t nsamples_x,
        const int64_t nsamples_dst, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
        const int64_t ids_stride, cudaStream_t stream) {

    const bool has_ids = ids != nullptr;

    if (has_ids && ncols_dst > 1) {
        // Multi-token MUL_MAT_ID path only - single-token goes through regular path below
        constexpr int c_ncols_dst = 1;
        launch_mul_mat_vec_f_cuda<T, type_acc, c_ncols_dst, true>
            (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
             nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
             stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
             ncols_dst, ids_stride, stream);
        return;
    }

    if (has_ids) {
        // Single-token MUL_MAT_ID path
        constexpr int c_ncols_dst = 1;
        launch_mul_mat_vec_f_cuda<T, type_acc, c_ncols_dst>
            (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
             nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
             stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
             ncols_dst, ids_stride, stream);
        return;
    }

    switch (ncols_dst) {
        case 1:
            launch_mul_mat_vec_f_cuda<T, type_acc, 1>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 2:
            launch_mul_mat_vec_f_cuda<T, type_acc, 2>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 3:
            launch_mul_mat_vec_f_cuda<T, type_acc, 3>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 4:
            launch_mul_mat_vec_f_cuda<T, type_acc, 4>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 5:
            launch_mul_mat_vec_f_cuda<T, type_acc, 5>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 6:
            launch_mul_mat_vec_f_cuda<T, type_acc, 6>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 7:
            launch_mul_mat_vec_f_cuda<T, type_acc, 7>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 8:
            launch_mul_mat_vec_f_cuda<T, type_acc, 8>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 9:
            launch_mul_mat_vec_f_cuda<T, type_acc, 9>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 10:
            launch_mul_mat_vec_f_cuda<T, type_acc, 10>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 11:
            launch_mul_mat_vec_f_cuda<T, type_acc, 11>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 12:
            launch_mul_mat_vec_f_cuda<T, type_acc, 12>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 13:
            launch_mul_mat_vec_f_cuda<T, type_acc, 13>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 14:
            launch_mul_mat_vec_f_cuda<T, type_acc, 14>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 15:
            launch_mul_mat_vec_f_cuda<T, type_acc, 15>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 16:
            launch_mul_mat_vec_f_cuda<T, type_acc, 16>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 17:
            launch_mul_mat_vec_f_cuda<T, type_acc, 17>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 18:
            launch_mul_mat_vec_f_cuda<T, type_acc, 18>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 19:
            launch_mul_mat_vec_f_cuda<T, type_acc, 19>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 20:
            launch_mul_mat_vec_f_cuda<T, type_acc, 20>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 21:
            launch_mul_mat_vec_f_cuda<T, type_acc, 21>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 22:
            launch_mul_mat_vec_f_cuda<T, type_acc, 22>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 23:
            launch_mul_mat_vec_f_cuda<T, type_acc, 23>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 24:
            launch_mul_mat_vec_f_cuda<T, type_acc, 24>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 25:
            launch_mul_mat_vec_f_cuda<T, type_acc, 25>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 26:
            launch_mul_mat_vec_f_cuda<T, type_acc, 26>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 27:
            launch_mul_mat_vec_f_cuda<T, type_acc, 27>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 28:
            launch_mul_mat_vec_f_cuda<T, type_acc, 28>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 29:
            launch_mul_mat_vec_f_cuda<T, type_acc, 29>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 30:
            launch_mul_mat_vec_f_cuda<T, type_acc, 30>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 31:
            launch_mul_mat_vec_f_cuda<T, type_acc, 31>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        case 32:
            launch_mul_mat_vec_f_cuda<T, type_acc, 32>
                (x, y, ids, fusion, dst, ncols, nrows, stride_row, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                 stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,
                 nsamples_dst, ids_stride, stream);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

template<typename T>
static void mul_mat_vec_f_cuda(
        const T * x, const float * y, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int64_t ncols, const int64_t nrows, const int64_t ncols_dst,
        const int64_t stride_row, const int64_t stride_col_y, const int stride_col_dst,
        const int64_t nchannels_x, const int64_t nchannels_y, const int64_t nchannels_dst,
        const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst, const int64_t nsamples_x,
        const int64_t nsamples_dst, const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,
        const int64_t ids_stride, enum ggml_prec prec, cudaStream_t stream) {

    if constexpr(std::is_same_v<T, half>) {
        if (prec == GGML_PREC_DEFAULT) {
            mul_mat_vec_f_cuda_switch_ncols_dst<T, half>
                (x, y, ids, fusion, dst, ncols, nrows, ncols_dst, stride_row, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
                stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            return;
        }
    }
    mul_mat_vec_f_cuda_switch_ncols_dst<T, float>
        (x, y, ids, fusion, dst, ncols, nrows, ncols_dst, stride_row, stride_col_y, stride_col_dst,
        nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y,
        stride_channel_dst, nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
}

void ggml_cuda_mul_mat_vec_f(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
    const ggml_cuda_mm_fusion_args_host * fusion, const ggml_cuda_mm_fusion_args_device * gb) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(!ids ||  ids->type == GGML_TYPE_I32);
    GGML_ASSERT(         dst->type == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(!ids || ne12 <= MMVF_MAX_BATCH_SIZE);
    GGML_ASSERT(ne13 == ne3);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));
    GGML_ASSERT(        nb0        == ts_dst);

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const enum ggml_prec prec = fast_fp16_available(cc) ? ggml_prec(dst->op_params[0]) : GGML_PREC_F32;

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    ggml_cuda_mm_fusion_args_device fusion_local{};

    if (fusion) {
        GGML_ASSERT( !ids || dst->ne[2] == 1);
        GGML_ASSERT(  ids || dst->ne[1] == 1);
        if (fusion->x_bias) {
            GGML_ASSERT(fusion->x_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->x_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->x_bias->ne[1] == src0->ne[2]);
            fusion_local.x_bias = fusion->x_bias->data;
        }
        if (fusion->gate) {
            GGML_ASSERT(fusion->gate->type == src0->type && ggml_are_same_stride(fusion->gate, src0));
            fusion_local.gate = fusion->gate->data;
        }
        if (fusion->gate_bias) {
            GGML_ASSERT(fusion->gate_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->gate_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->gate_bias->ne[1] == src0->ne[2]);
            fusion_local.gate_bias = fusion->gate_bias->data;
        }
        fusion_local.glu_op = fusion->glu_op;
        fusion_local.glu_limit = fusion->glu_limit;
    }
    if (gb) {
        GGML_ASSERT(!fusion && !ids && src0->type == GGML_TYPE_BF16 && gb->gb_nrows == ne01);
        fusion_local.gb_x2    = gb->gb_x2;
        fusion_local.gb_dst2  = gb->gb_dst2;
        fusion_local.gb_dt    = gb->gb_dt;
        fusion_local.gb_a     = gb->gb_a;
        fusion_local.gb_nrows = gb->gb_nrows;
    }

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = src1->nb[1] / ts_src1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s12 = src1->nb[2] / ts_src1;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s13 = src1->nb[3] / ts_src1;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_col_y       = ids ? s12  : s11;
    const int64_t stride_channel_dst = ids ? s1   : s2;
    const int64_t stride_channel_y   = ids ? s11  : s12;

    const int64_t ids_stride = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;

    switch (src0->type) {
        case GGML_TYPE_F32: {
            const float * src0_d = (const float *) src0->data;
            mul_mat_vec_f_cuda(src0_d, src1_d, ids_d, fusion_local, dst_d, ne00, ne01, ncols_dst, s01, stride_col_y, stride_col_dst,
                ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
                ne03,              ne3,           s03, s13,              s3,                 ids_stride, prec, ctx.stream());
        } break;
        case GGML_TYPE_F16: {
            const half * src0_d = (const half *) src0->data;
            mul_mat_vec_f_cuda(src0_d, src1_d, ids_d, fusion_local, dst_d, ne00, ne01, ncols_dst, s01, stride_col_y, stride_col_dst,
                ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
                ne03,              ne3,           s03, s13,              s3,                 ids_stride, prec, ctx.stream());
        } break;
        case GGML_TYPE_BF16: {
            const nv_bfloat16 * src0_d = (const nv_bfloat16 *) src0->data;
            mul_mat_vec_f_cuda(src0_d, src1_d, ids_d, fusion_local, dst_d, ne00, gb ? 2*ne01 : ne01, ncols_dst, s01, stride_col_y, stride_col_dst,
                ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
                ne03,              ne3,           s03, s13,              s3,                 ids_stride, prec, ctx.stream());
        } break;
        default:
            GGML_ABORT("unsupported type: %s", ggml_type_name(src0->type));
    }
}

void ggml_cuda_op_mul_mat_vec_f(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];
    const int64_t ne0  =  dst->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    const enum ggml_prec prec = fast_fp16_available(cc) ? ggml_prec(dst->op_params[0]) : GGML_PREC_F32;

    // ggml_cuda_op provides single, contiguous matrices
    const int64_t stride_row         = ne00;
    const int64_t stride_col_y       = ne10;
    const int64_t stride_col_dst     = id == ctx.device ? ne0 : row_diff; // main device has larger memory buffer
    const int64_t nchannels_x        = 1;
    const int64_t nchannels_y        = 1;
    const int64_t nchannels_dst      = 1;
    const int64_t stride_channel_x   = 0;
    const int64_t stride_channel_y   = 0;
    const int64_t stride_channel_dst = 0;
    const int64_t nsamples_x         = 1;
    const int64_t nsamples_dst       = 1;
    const int64_t stride_sample_x    = 0;
    const int64_t stride_sample_y    = 0;
    const int64_t stride_sample_dst  = 0;

    ggml_cuda_mm_fusion_args_device empty{};
    switch (src0->type) {
        case GGML_TYPE_F32: {
            const float * src0_d = (const float *) src0_dd_i;
            mul_mat_vec_f_cuda(src0_d, src1_ddf_i, nullptr, empty, dst_dd_i, ne00, row_diff, src1_ncols, stride_row, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, 0, prec, stream);
        } break;
        case GGML_TYPE_F16: {
            const half * src0_d = (const half *) src0_dd_i;
            mul_mat_vec_f_cuda(src0_d, src1_ddf_i, nullptr, empty, dst_dd_i, ne00, row_diff, src1_ncols, stride_row, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, 0, prec, stream);
        } break;
        case GGML_TYPE_BF16: {
            const nv_bfloat16 * src0_d = (const nv_bfloat16 *) src0_dd_i;
            mul_mat_vec_f_cuda(src0_d, src1_ddf_i, nullptr, empty, dst_dd_i, ne00, row_diff, src1_ncols, stride_row, stride_col_y, stride_col_dst,
                nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, 0, prec, stream);
        } break;
        default:
            GGML_ABORT("unsupported type: %s", ggml_type_name(src0->type));
    }

    GGML_UNUSED_VARS(ctx, src1, dst, src1_ddq_i, src1_ncols, src1_padded_row_size);
}

bool ggml_cuda_should_use_mmvf(enum ggml_type type, int cc, int warp_size, const int64_t * src0_ne, const size_t * src0_nb, int64_t ne11) {
    if (src0_ne[0] % 2 != 0) {
        return false;
    }

    const size_t ts = ggml_type_size(type);
    if (src0_nb[0] != ts) {
        return false;
    }

    // Pointers not aligned to the size of half2/nv_bfloat162/float2 would result in a crash:
    for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
        if (src0_nb[i] % (2*ts) != 0) {
            return false;
        }
    }

    switch (type) {
        case GGML_TYPE_F32:
            if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
                if (ampere_mma_available(cc)) {
                    return ne11 <= 3;
                }
                if (cc >= GGML_CUDA_CC_TURING) {
                    return ne11 <= 4;
                }
                return ne11 <= 3;
            } else if (GGML_CUDA_CC_IS_AMD(cc)) {
                if (fp32_mma_hardware_available(cc)) {
                    return ne11 <= 3;
                }
                return ne11 <= MMVF_MAX_BATCH_SIZE;
            }
            return ne11 <= MMVF_MAX_BATCH_SIZE;
        case GGML_TYPE_F16:
            if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
                const bool src0_small = (src0_ne[1] <= 512 || src0_ne[2]*src0_ne[3] == 1);
                // MMF needs full row tiles, for other row counts MMVF still beats cuBLAS at small batch size
                if (src0_small && !ggml_cuda_should_use_mmf(type, cc, warp_size, src0_ne, src0_nb, ne11, /*mul_mat_id =*/ false)) {
                    return ne11 <= MMVF_MAX_BATCH_SIZE;
                }
                if (ampere_mma_available(cc)) {
                    return src0_small && ne11 == 1;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    return src0_small && ne11 <= 4;
                }
                if (fp16_mma_hardware_available(cc)) {
                    return src0_small && ne11 <= 3;
                }
                return ne11 <= MMVF_MAX_BATCH_SIZE;
            } else if (GGML_CUDA_CC_IS_AMD(cc)) {
                if (GGML_CUDA_CC_IS_RDNA(cc) && !ggml_cuda_should_use_mmf(type, cc, warp_size, src0_ne, src0_nb, ne11, /*mul_mat_id =*/ false)) {
                    return ne11 <= MMVF_MAX_BATCH_SIZE;
                }
                if (fp16_mma_hardware_available(cc)) {
                    if (GGML_CUDA_CC_IS_RDNA3(cc)) {
                        return ne11 <= 3;
                    }
                    if (GGML_CUDA_CC_IS_RDNA4(cc)) {
                        return ne11 <= 5;
                    }
                    return ne11 <= 2;
                }
                return ne11 <= MMVF_MAX_BATCH_SIZE;
            }
            return ne11 <= MMVF_MAX_BATCH_SIZE;
        case GGML_TYPE_BF16:
            if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
                const bool src0_small = (src0_ne[1] <= 512 || src0_ne[2]*src0_ne[3] == 1);
                // MMF needs full row tiles, for other row counts MMVF still beats cuBLAS at small batch size
                if (src0_small && !ggml_cuda_should_use_mmf(type, cc, warp_size, src0_ne, src0_nb, ne11, /*mul_mat_id =*/ false)) {
                    return ne11 <= MMVF_MAX_BATCH_SIZE;
                }
                if (ampere_mma_available(cc)) {
                    return src0_small && ne11 == 1;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    return src0_small && ne11 <= 4;
                }
                if (bf16_mma_hardware_available(cc)) {
                    return src0_small && ne11 <= 3;
                }
                return ne11 <= MMVF_MAX_BATCH_SIZE;
            } else if (GGML_CUDA_CC_IS_AMD(cc)) {
                if (GGML_CUDA_CC_IS_RDNA(cc) && !ggml_cuda_should_use_mmf(type, cc, warp_size, src0_ne, src0_nb, ne11, /*mul_mat_id =*/ false)) {
                    return ne11 <= MMVF_MAX_BATCH_SIZE;
                }
                if (bf16_mma_hardware_available(cc)) {
                    return ne11 <= 3;
                }
                return ne11 <= MMVF_MAX_BATCH_SIZE;
            }
            return ne11 <= MMVF_MAX_BATCH_SIZE;
        default:
            return false;
    }
}

// GDN gate/beta with BF16 projections (qwen4exp), decode and verify band (GGML_CUDA_FUSE_GATE_BETA_BF16=0: off):
//   gate = softplus(alpha_w x + dt) * a, beta = beta_w x (the sigmoid stays for the GATED_DELTA_NET fusion).
// One mul_mat_vec_f launch computes both projections (alpha rows, then beta rows) and applies the gate chain to the
// alpha rows before the store.  Each row runs the instantiation the unfused MUL_MAT would take, so both outputs are
// bit-identical.  The Q8_0 pair is ggml_cuda_gate_beta_verify_try_fuse (mmvq.cu).  Returns 7 when fused, 0 otherwise.
int ggml_cuda_gate_beta_bf16_try_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    static const bool fuse = getenv("GGML_CUDA_FUSE_GATE_BETA_BF16") == nullptr || atoi(getenv("GGML_CUDA_FUSE_GATE_BETA_BF16")) != 0;
    if (!fuse || i + 7 >= cgraph->n_nodes || cgraph->nodes[i]->op != GGML_OP_MUL_MAT) {
        return 0;
    }
    const ggml_op ops[8] = {
        GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_MUL,
        GGML_OP_RESHAPE, GGML_OP_MUL_MAT, GGML_OP_RESHAPE
    };
    const int out_nodes[] = { i + 5, i + 7 };
    if (!ggml_can_fuse_subgraph(cgraph, i, 8, ops, out_nodes, 2)) {
        return 0;
    }
    ggml_tensor * alpha_w  = cgraph->nodes[i];
    ggml_tensor * alpha_v  = cgraph->nodes[i + 1];
    ggml_tensor * bias     = cgraph->nodes[i + 2];
    ggml_tensor * softplus = cgraph->nodes[i + 3];
    ggml_tensor * gate_mul = cgraph->nodes[i + 4];
    ggml_tensor * beta_w   = cgraph->nodes[i + 6];
    ggml_tensor * beta_v   = cgraph->nodes[i + 7];
    if (alpha_v->src[0] != alpha_w || softplus->src[0] != bias || cgraph->nodes[i + 5]->src[0] != gate_mul ||
            beta_v->src[0] != beta_w || alpha_w->src[1] != beta_w->src[1] ||
            ggml_get_unary_op(softplus) != GGML_UNARY_OP_SOFTPLUS) {
        return 0;
    }
    const ggml_tensor * dt    = bias->src[0] == alpha_v ? bias->src[1] : (bias->src[1] == alpha_v ? bias->src[0] : nullptr);
    const ggml_tensor * ssm_a = gate_mul->src[0] == softplus ? gate_mul->src[1] : (gate_mul->src[1] == softplus ? gate_mul->src[0] : nullptr);
    const ggml_tensor * wa = alpha_w->src[0];
    const ggml_tensor * wb = beta_w->src[0];
    const ggml_tensor * y  = alpha_w->src[1];
    const int cc        = ggml_cuda_info().devices[ctx.device].cc;
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
    const int64_t nr    = wa->ne[1];
    if (!dt || !ssm_a || wa->type != GGML_TYPE_BF16 || wb->type != GGML_TYPE_BF16 || y->type != GGML_TYPE_F32 ||
            wa->view_src || wb->view_src || wa->ne[0] != wb->ne[0] || nr != wb->ne[1] || !ggml_are_same_stride(wa, wb) ||
            wa->ne[2] != 1 || wa->ne[3] != 1 || y->ne[2] != 1 || y->ne[3] != 1 ||
            y->ne[1] < 1 || y->ne[1] > MMVF_MAX_BATCH_SIZE || nr >= 3072 ||
            alpha_w->type != GGML_TYPE_F32 || beta_w->type != GGML_TYPE_F32 || bias->type != GGML_TYPE_F32 ||
            softplus->type != GGML_TYPE_F32 || gate_mul->type != GGML_TYPE_F32 ||
            dt->type != GGML_TYPE_F32 || ssm_a->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(dt) || !ggml_is_contiguous(ssm_a) || dt->ne[0] != nr || ssm_a->ne[0] != nr ||
            ggml_nelements(dt) != nr || ggml_nelements(ssm_a) != nr ||
            !ggml_is_contiguous(alpha_w) || !ggml_is_contiguous(beta_w) || !ggml_is_contiguous(gate_mul) ||
            !ggml_are_same_shape(alpha_w, beta_w) || ggml_nelements(gate_mul) != ggml_nelements(alpha_w) ||
            gate_mul->ne[0] != nr || alpha_w->ne[0] != nr ||
            ggml_cuda_op_mul_mat_use_fwht(alpha_w) || ggml_cuda_op_mul_mat_use_fwht(beta_w) ||
            !ggml_cuda_should_use_mmvf(wa->type, cc, warp_size, wa->ne, wa->nb, y->ne[1]) ||
            !ggml_cuda_should_use_mmvf(wb->type, cc, warp_size, wb->ne, wb->nb, y->ne[1])) {
        return 0;
    }
    ggml_cuda_mm_fusion_args_device gb{};
    gb.gb_x2    = wb->data;
    gb.gb_dst2  = (float *) beta_w->data;
    gb.gb_dt    = (const float *) dt->data;
    gb.gb_a     = (const float *) ssm_a->data;
    gb.gb_nrows = (int) nr;
    // Launch with the projection's own shape, not the gate MUL's.  The launch takes its column count from
    // dst->ne[1] and its channel count from dst->ne[2], but with more than one sequence the unfused chain's
    // shapes are [nr, n_seq_tokens, n_seqs] for the gate MUL (and for the beta store) while the projections
    // and y are flat [nr, n_tokens] / [ne10, n_tokens].  Taking the gate MUL's dst, a multi-sequence decode
    // (e.g. 2 sequences x 1 token: dst [nr, 1, 2] against y [ne10, 2], one channel) would run the beta rows
    // once per channel and write every channel from y's first n_seq_tokens columns - i.e. sequences 1..n-1
    // got sequence 0's alpha/beta gates (issue #204).  Both tensors are contiguous with the same element
    // count (checked above), so the projection's shape is the same memory, written flat.
    ggml_tensor dst_flat = *alpha_w;
    dst_flat.data = gate_mul->data;
    ggml_cuda_mul_mat_vec_f(ctx, wa, y, nullptr, &dst_flat, nullptr, &gb);
    return 7;
}
