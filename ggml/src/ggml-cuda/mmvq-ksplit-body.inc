// The body of mul_mat_vec_q_ksplit (mmvq.cu), also the body of mul_mat_vec_q_ksplit_body for the mixed-type row
// concat.  It is included into both rather than having the kernel call the device function: inlined as a function,
// the body compiles differently in the multi-column kernels (gfx1201: fewer VGPRs, longer code), which cost 6-7 %
// of MTP decode there.
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq_dense(type, get_device_table_id());
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int nwarps = calc_nwarps_weight(type, ncols_dst, table_id, long_k, small_m);
    constexpr int rows_per_cuda_block = rows_per_block > 0 ? rows_per_block : calc_rows_per_block_weight(type, ncols_dst, table_id, small_k, nwarps);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const     int tid = warp_size*threadIdx.y + threadIdx.x;
    // nrows_loop > 0 (row_loop types only): a smaller grid walks the rows, block b taking row group b, b + gridDim.x, ...
    // (groups of rows_per_cuda_block rows; see GGML_MMVQ_RDNA4_DECODE_GRID). Each row is still one warp's unchanged dot
    // product. For every other
    // instantiation the loop body runs once.
    constexpr bool row_loop = mmvq_decode_grid_loop_type(type) && ncols_dst == 1 && rows_per_cuda_block <= 2 && nwarps == 1;
    for (int row0_g = rows_per_cuda_block*blockIdx.x; ; row0_g += rows_per_cuda_block*gridDim.x) {
    if (nrows_loop != 0 && uint32_t(row0_g) >= nrows_loop) {
        break;   // a padding block (ggml_cuda_grid_stall_pad) or a row loop past the end
    }
    dst = dst_ptr;
    int row0 = row0_g;
    if constexpr (row_cat) {
        vx = vx_ptr;
        // row concat: the rows of the appended weights (segment boundaries are multiples of rows_per_cuda_block)
        if (fusion.cat_dst[1] != nullptr && uint32_t(row0) >= fusion.cat_row[1]) {
            vx    = fusion.cat_x[1];
            dst   = fusion.cat_dst[1];
            row0 -= fusion.cat_row[1];
        } else if (fusion.cat_dst[0] != nullptr && uint32_t(row0) >= fusion.cat_row[0]) {
            vx    = fusion.cat_x[0];
            dst   = fusion.cat_dst[0];
            row0 -= fusion.cat_row[0];
        }
    }
    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;

    const bool shared_expert = has_fusion && fusion.shared_up && blockIdx.y == gridDim.y - 1;
    const uint32_t channel_dst = shared_expert ? 0 : blockIdx.y;
    if (shared_expert) {
        vx = fusion.shared_up;
        dst = fusion.shared_dst;
        stride_col_dst = fusion.shared_stride_col_dst;
    }

    uint32_t channel_x;
    uint32_t channel_y;
    uint32_t sample_dst;

    ggml_cuda_pdl_sync();
    channel_x  = shared_expert ? 0 : ncols_dst == 1 && ids ? ids[channel_dst] : fastdiv(channel_dst, channel_ratio);
    channel_y  = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    sample_dst = blockIdx.z;

    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    bool use_scale = false;
    bool use_gate_scale = false;
    bool use_dst_gate = false;
    bool use_conv_input = false;
    [[maybe_unused]] const void * vgate = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    const float * x_scale = nullptr;
    const float * gate_scale = nullptr;
    [[maybe_unused]] const void * dst_gate = nullptr;
    [[maybe_unused]] float * conv_input = nullptr;
    [[maybe_unused]] const float * conv_states = nullptr;
    [[maybe_unused]] int conv_kernel_size = 0;
    ggml_glu_op active_glu;
    float glu_limit = 0.0f;

    if constexpr (has_fusion) {
        use_gate      = fusion.gate      != nullptr;
        use_bias      = fusion.x_bias    != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr && use_gate;
        vgate         = shared_expert ? fusion.shared_gate : fusion.gate;
        x_bias        = (const float *) fusion.x_bias;
        gate_bias     = (const float *) fusion.gate_bias;
        active_glu    = fusion.glu_op;
        glu_limit     = fusion.glu_limit;
        use_dst_gate  = fusion.dst_gate != nullptr && use_gate;
        if (use_dst_gate) {
            dst_gate = fusion.dst_gate;
        }
        use_conv_input = fusion.conv_input != nullptr && fusion.conv_states != nullptr;
        if (use_conv_input) {
            conv_input       = (float *) fusion.conv_input;
            conv_states      = (const float *) fusion.conv_states;
            conv_kernel_size = fusion.conv_kernel_size;
        }
        if constexpr (type == GGML_TYPE_NVFP4) {
            use_scale      = fusion.x_scale    != nullptr;
            use_gate_scale = fusion.gate_scale != nullptr && use_gate;
            x_scale        = (const float *) fusion.x_scale;
            gate_scale     = (const float *) fusion.gate_scale;
        }
        // Per-token scale (MoE down x topk weights). Indexed by channel_dst.
        if (fusion.x_scale_channel_dst) {
            use_scale = true;
            x_scale   = (const float *) fusion.x_scale;
        }
    }


    [[maybe_unused]] float x_biases[ncols_dst]    = { 0.0f };
    [[maybe_unused]] float gate_biases[ncols_dst] = { 0.0f };
    [[maybe_unused]] float x_scales = 1.0f;
    [[maybe_unused]] float gate_scales = 1.0f;
    if constexpr (has_fusion) {
        // 1. Hide latency by prefetching bias, gates and scales here
        // 2. load only on threads that won't die after partial sum calculation
        const uint32_t channel_bias = ids ? channel_x : channel_dst;
        if (threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
            (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
            if (use_bias) {
                x_bias = x_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    x_biases[j] = x_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if (use_gate_bias) {
                gate_bias = gate_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    gate_biases[j] = gate_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if (use_scale) {
                x_scales = fusion.x_scale_channel_dst ? x_scale[channel_dst] : x_scale[ids ? channel_x : 0];
            }
            if (use_gate_scale) {
                gate_scales = gate_scale[ids ? channel_x : 0];
            }
        }
    }

    // partial sum for each thread
    float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};
    float tmp_gate[ncols_dst][rows_per_cuda_block] = {{0.0f}};

    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

#if __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
        // start the next iterations' weight loads early
        if constexpr (mmvq_should_prefetch(type)) {
            // loop iterations, not blocks; q3_K and q6_K keep the distance in blocks they had with vdr 1
            constexpr int pf_dist = type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q6_K ? 1 : 2;
            const int kbx_pf = kbx + pf_dist*blocks_per_iter;
            if (kbx_pf < blocks_per_row_x) {
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    const size_t off = (size_t)(kbx_offset + i*stride_row_x + kbx_pf) * ggml_cuda_type_traits<type>::bs;
                    ggml_cuda_prefetch_l2((const char *) vx + off);
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            ggml_cuda_prefetch_l2((const char *) vgate + off);
                        }
                    }
                }
            }
        }
#endif

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp[j][i] += vec_dot_q_cuda(
                    vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += vec_dot_q_cuda(
                            vgate, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                    }
                }
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
    [[maybe_unused]] __shared__ float tmp_shared_gate[(has_fusion && (nwarps-1 > 0)) ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];

    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_shared_gate[threadIdx.y-1][j][i][threadIdx.x] = tmp_gate[j][i];
                    }
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;

    // Llama-Frankenstein R1: without fusion, reduce all ncols_dst*rows_per_cuda_block partial sums at once by
    // recursive halving instead of one butterfly per output (8 tokens x 4 rows: 31 lane exchanges instead of 160).
    // Every output is summed by the same pairing tree (lane ^ 16, ^ 8, ... ^ 1, own value first) as
    // warp_reduce_sum, only in a different lane, so the results are bit-identical.
    // One-wave blocks only: in the 8-wave Q8_0 short-K block the gathering warp's serial halving steps were
    // slower on small grids (gpt-oss k/v, 512 rows x 6 tokens: 6.7 -> 11.1 us per launch).
    constexpr int lf_nval = ncols_dst*rows_per_cuda_block;
    if constexpr (!has_fusion && nwarps == 1 && lf_nval >= 2 && lf_nval <= warp_size) {
        float lf_v[lf_nval];
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                float t = tmp[j][i];
#pragma unroll
                for (int l = 0; l < nwarps-1; ++l) {
                    t += tmp_shared[l][j][i][threadIdx.x];
                }
                lf_v[j*rows_per_cuda_block + i] = t;
            }
        }
        const float lf_sum = ggml_cuda_lf_reduce_scatter<lf_nval, warp_size>(lf_v);
        constexpr int lf_lanes = warp_size / ggml_cuda_lf_pow2_ceil(lf_nval); // lanes holding the same output
        const int lf_idx = threadIdx.x / lf_lanes;
        if (threadIdx.x % lf_lanes == 0 && lf_idx < lf_nval) {
            const int j = lf_idx / rows_per_cuda_block;
            const int i = lf_idx % rows_per_cuda_block;
            if (rows_per_cuda_block == 1 || uint32_t(row0 + i) < stride_col_dst) {
                dst[j*stride_col_dst + i] = lf_sum;
            }
        }
    } else {
    // sum up partial sums and write back result
    #pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
    #pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
    #pragma unroll
                for (int l = 0; l < nwarps-1; ++l) {
                    tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            tmp_gate[j][i] += tmp_shared_gate[l][j][i][threadIdx.x];
                        }
                    }
                }
                tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] = warp_reduce_sum<warp_size>(tmp_gate[j][i]);
                    }
                }

                float result_val = 0.0f;
                if (threadIdx.x == i && (rows_per_cuda_block == 1 || uint32_t(row0 + i) < stride_col_dst)) {
                    result_val = tmp[j][i];
                    if constexpr (has_fusion) {
                        if (use_scale) {
                            result_val *= x_scales;
                        }
                        result_val += x_biases[j];
                        if (use_gate) {
                            float gate_value = tmp_gate[j][i];
                            if constexpr (type == GGML_TYPE_NVFP4) {
                                gate_value *= gate_scales;
                            }
                            gate_value += gate_biases[j];
                            if (use_dst_gate) {
                                // separate output: write the gate result to its own destination
                                // (dst was already offset by sample/channel/row; apply the same to dst_gate)
                                float * dst_gate_row = (float *) dst_gate + sample_dst*stride_sample_dst +
                                                       channel_dst*stride_channel_dst + row0 + j*stride_col_dst;
                                dst_gate_row[i] = gate_value;
                            } else {
                                switch (active_glu) {
                                    case GGML_GLU_OP_SWIGLU:
                                        result_val *= ggml_cuda_op_silu_single(gate_value);
                                        break;
                                    case GGML_GLU_OP_GEGLU:
                                        result_val *= ggml_cuda_op_gelu_single(gate_value);
                                        break;
                                    case GGML_GLU_OP_SWIGLU_OAI:
                                        result_val = ggml_cuda_op_swiglu_oai_single(gate_value, result_val);
                                        break;
                                    case GGML_GLU_OP_SWIGLU_CLAMP:
                                        result_val = ggml_cuda_op_swiglu_clamp_single(gate_value, result_val, glu_limit);
                                        break;
                                    default:
                                        result_val = result_val * gate_value;
                                        break;
                                }
                            }
                        }
                    }
                }
                if (use_conv_input) {
                    // interleaved conv input: [state0, state1, ..., state_{cs-2}, result] per
                    // channel. conv_input [cs, C] has nb1 = cs floats, conv_states [(cs-1), C]
                    // has nb1 = cs-1 floats. Spread the 4 writes over threads 0..cs-1.
                    const float r_bcast = __shfl_sync(0xffffffff, result_val, i, warp_size);
                    const int c = row0 + i;
                    const int cs = conv_kernel_size;
                    if (threadIdx.x < cs) {
                        const int k = threadIdx.x;
                        const float v = k < cs - 1 ? conv_states[(cs-1)*c + k] : r_bcast;
                        conv_input[cs*c + k] = v;
                    }
                } else if (threadIdx.x == i && (rows_per_cuda_block == 1 || uint32_t(row0 + i) < stride_col_dst)) {
                    dst[j*stride_col_dst + i] = result_val;
                }
            }
        }
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, use_scale, use_gate_scale, use_dst_gate, use_conv_input, active_glu, glu_limit, gate_bias, x_bias, x_scale, gate_scale, tmp_gate, dst_gate, conv_input, conv_states, conv_kernel_size);
    }
    if constexpr (type != GGML_TYPE_NVFP4) {
        GGML_UNUSED_VARS(use_scale, use_gate_scale, x_scale, gate_scale, x_scales, gate_scales);
    }
    if constexpr (!row_loop) {
        GGML_UNUSED(nrows_loop);
        break;
    } else {
        if (uint32_t(row0_g) + rows_per_cuda_block*gridDim.x >= nrows_loop) {
            break;
        }
    }
    }
