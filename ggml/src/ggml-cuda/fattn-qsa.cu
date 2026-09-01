#include "common.cuh"
#include "fattn-qsa.cuh"
#include "fattn-common.cuh"
#include <vector>

// QSA sparse flash attention for qwen4exp: attend only over the cells the
// indexer's top-k names, instead of the whole KV cache.  The K/V cache is
// F16/BF16; the mask is the base kq_mask and is gathered at the idx positions.
//
// Layouts (after the same permutes the dense FA path applies):
//   q    [n_embd_head_q, n_tps, n_head_q, n_stream]  F32
//   k    [n_embd_head_k, n_kv,   n_head_kv, n_stream] F16/BF16
//   v    [n_embd_head_v, n_kv,   n_head_v,  n_stream] F16/BF16
//   idx  [n_top_k, n_tps, 1, n_stream] I32
//   mask [n_kv, n_tps, 1, n_stream] F16
//   dst  [n_embd_head_v, n_head_q, n_tps, n_stream] F32
//
// Each block handles ONE token column for ALL q-heads of a stream (one
// warp per head).  All heads read the same idx list and the same K/V cells
// (gqa: same kv-head), so the gathers hit the same L1 lines and the L2
// fetch is shared - the dense tile FA reads each K/V cell once for all
// heads, this kernel must too or the 12x re-gather saturates L2.
// Each warp walks the column's top-k list in TILES of WARP_SIZE cells.
// Within a tile the 32 cells are processed in 4 groups of 8 lanes
// (NTHREADS_KQ per cell): the 4 cells of a group-step are in flight at
// once (independent K gathers), each dot is reduced with a cheap 3-step
// shuffle, and the online-softmax max/sum/VKQ rescale happens once per
// tile.  This is the same structure as the vec FA kernel and breaks the
// per-cell serial dependency chain.

static constexpr __device__ int ggml_cuda_fattn_qsa_get_nthreads_device() {
    return 16*WARP_SIZE; // 1 warp per head; up to 16 heads per block
}

template<int D, ggml_type type_KV, bool use_logit_softcap> // D == head size
__launch_bounds__(ggml_cuda_fattn_qsa_get_nthreads_device(), 1)
static __global__ void flash_attn_qsa(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const int  * idx_ptr,
        const char * mask_ptr,
        float      * dst_ptr,
        const float scale,
        const float logit_softcap,
        const bool   identity,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
        const int32_t ne30, const int32_t ne31, const int32_t ne33,
                            const int32_t nb31, const int64_t nb33,
        const int32_t ne40, const int32_t ne41, const int32_t ne43,
                            const int32_t nb41, const int64_t nb43) {
    ggml_cuda_pdl_lc();
#ifdef FLASH_ATTN_AVAILABLE
    const char * GGML_CUDA_RESTRICT Q    = Q_ptr;
    const char * GGML_CUDA_RESTRICT K    = K_ptr;
    const char * GGML_CUDA_RESTRICT V    = V_ptr;
    const int  * GGML_CUDA_RESTRICT idx  = idx_ptr;
    const char * GGML_CUDA_RESTRICT mask = mask_ptr;
    float      * GGML_CUDA_RESTRICT dst  = dst_ptr;

    if constexpr (use_logit_softcap) {
        GGML_UNUSED_VARS(Q, K, V, idx, mask, dst, scale, logit_softcap,
            ne00, ne01, ne02, ne03, nb01, nb02, nb03,
            ne10, ne11, ne12, ne13, nb11, nb12, nb13,
            nb21, nb22, nb23, ne30, ne31, ne33, nb31, nb33,
            ne40, ne41, ne43, nb41, nb43);
        NO_DEVICE_CODE;
        return;
    }

    // One block per (column, stream); warp w = head w.
    const int col = blockIdx.x;
    const int tid = threadIdx.x;

    const int sequence = blockIdx.z;
    const int head     = threadIdx.y;
    const int gqa_ratio = ne02 / ne12;

    if (col >= int(ne01.z)) {
        return;
    }

    Q += nb03*sequence + nb02*head + nb01*col;
    K += nb13*sequence + nb12*(head / gqa_ratio);
    V += nb23*sequence + nb22*(head / gqa_ratio);
    idx += col*ne40 + sequence*ne40*ne41; // [n_top_k, n_tps, 1, n_stream]
    const half * maskh = (const half *) (mask + nb33*(sequence % ne33) + nb31*col);

    // Per-warp tile exp weights, staged in shared:
    constexpr int NHEADS_MAX = 16; // launch bounds: 16 warps max
    __shared__ float KQ_w[NHEADS_MAX][WARP_SIZE];
    float * KQ_warp = KQ_w[head];

    // 8 lanes per cell -> 4 cells in flight per warp per group-step.
    constexpr int NTHREADS_KQ = 8;
    constexpr int NCELLS      = WARP_SIZE / NTHREADS_KQ;          // 4 cells per step
    constexpr int NSTEPS      = NTHREADS_KQ;                      // steps per tile
    static_assert(NSTEPS*NCELLS == WARP_SIZE, "tile must cover WARP_SIZE cells");
    constexpr int nchunks_KQ = (D/2)/NTHREADS_KQ;                 // half2 per lane per cell
    constexpr int NSTEPS_Q   = nchunks_KQ/4;                      // Q stride steps (D/64)
    static_assert(nchunks_KQ == 4*NSTEPS_Q, "D/2 must be divisible by 4*NTHREADS_KQ");
    const int lane_in_group = tid & (NTHREADS_KQ - 1);            // 0..7

    if constexpr (type_KV == GGML_TYPE_F16) {
        // Q replicated per lane, strided like the vec kernel:
        //   lane p holds Q half2 at (p*4 + k) + 32*j for j in 0..3, k in 0..3
        half2 Q_h2[nchunks_KQ];
        const float2 * Q_col = (const float2 *) Q;
#pragma unroll
        for (int j = 0; j < NSTEPS_Q; ++j) {
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float2 qf = Q_col[lane_in_group*4 + k + 32*j];
                Q_h2[j*4 + k] = __float22half2_rn(make_float2(qf.x*scale, qf.y*scale));
            }
        }

        float KQ_max = -FLT_MAX/2.0f;
        float KQ_sum = 0.0f;
        float2 VKQ[D/(2*WARP_SIZE)] = {};

        const int n_top_k = ne40;
        for (int tile0 = 0; tile0 < n_top_k; tile0 += WARP_SIZE) {
            const int tile_len = min(WARP_SIZE, n_top_k - tile0);

            // Score pass: NSTEPS steps, NCELLS cells in flight per step.
            // The K gathers for the 4 cells of a step are independent.
            float KQ_max_new = KQ_max;
#pragma unroll
            for (int i = 0; i < NSTEPS; ++i) {
                const int cell_in_tile = (tid/NTHREADS_KQ)*NSTEPS + i;
                float partial = 0.0f;
                int cell = -1;
                if (cell_in_tile < tile_len) {
                    cell = identity ? tile0 + cell_in_tile : idx[tile0 + cell_in_tile];
                    const half2 * K_cell = (const half2 *) (K + cell*nb11);
#pragma unroll
                    for (int j = 0; j < NSTEPS_Q; ++j) {
#pragma unroll
                        for (int k = 0; k < 4; ++k) {
                            ggml_cuda_mad(partial, K_cell[lane_in_group*4 + k + 32*j], Q_h2[j*4 + k]);
                        }
                    }
                }
                partial = warp_reduce_sum<NTHREADS_KQ>(partial);

                const float score = (cell_in_tile < tile_len) ? partial + __half2float(maskh[cell]) : -FLT_MAX/2.0f;
                KQ_max_new = fmaxf(KQ_max_new, score + FATTN_KQ_MAX_OFFSET);
                KQ_warp[cell_in_tile] = score;
            }

            // Cross-group max reduction, then one softmax update per tile:
#pragma unroll
            for (int offset = NTHREADS_KQ; offset < WARP_SIZE; offset <<= 1) {
                KQ_max_new = fmaxf(KQ_max_new, __shfl_xor_sync(0xFFFFFFFF, KQ_max_new, offset, WARP_SIZE));
            }
            const float KQ_max_scale = expf(KQ_max - KQ_max_new);
            KQ_max = KQ_max_new;
            KQ_sum *= KQ_max_scale;
#pragma unroll
            for (int k = 0; k < D/(2*WARP_SIZE); ++k) {
                VKQ[k].x *= KQ_max_scale;
                VKQ[k].y *= KQ_max_scale;
            }

            // Exp weights per cell (lane tid rewrites its own cell's score):
            const float KQ_reg = (tid < tile_len) ? expf(KQ_warp[tid] - KQ_max) : 0.0f;
            KQ_warp[tid] = KQ_reg;
            KQ_sum += warp_reduce_sum(KQ_reg);

            // VKQ pass: for each cell of the tile, all lanes accumulate their
            // D chunk, weighted by the (broadcast) exp weight.  Independent
            // gathers -> they pipeline.
            for (int c = 0; c < WARP_SIZE; ++c) {
                const float w = (c < tile_len) ? KQ_warp[c] : 0.0f;
                if (w != 0.0f) {
                    const int cell_c = (c < tile_len) ? (identity ? tile0 + c : idx[tile0 + c]) : 0;
                    const half2 * V_cell = (const half2 *) (V + cell_c*nb21);
#pragma unroll
                for (int k = 0; k < D/(2*WARP_SIZE); ++k) {
                    const half2 v = V_cell[tid + k*WARP_SIZE];
                    VKQ[k].x += __half2float(v.x)*w;
                    VKQ[k].y += __half2float(v.y)*w;
                }
                }
            }
        }

        float * dst_col = dst + ((sequence*int(ne01.z) + col)*ne02 + head)*D;
        float2 * dst2 = (float2 *) dst_col;
#pragma unroll
        for (int k = 0; k < D/(2*WARP_SIZE); ++k) {
            dst2[tid + k*WARP_SIZE] = make_float2(VKQ[k].x / KQ_sum, VKQ[k].y / KQ_sum);
        }
    } else {
        // BF16 K/V
        nv_bfloat162 Q_bf16[nchunks_KQ];
        const float2 * Q_col = (const float2 *) Q;
#pragma unroll
        for (int j = 0; j < NSTEPS_Q; ++j) {
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float2 qf = Q_col[lane_in_group*4 + k + 32*j];
                Q_bf16[j*4 + k] = __float22bfloat162_rn(make_float2(qf.x*scale, qf.y*scale));
            }
        }

        float KQ_max = -FLT_MAX/2.0f;
        float KQ_sum = 0.0f;
        float2 VKQ[D/(2*WARP_SIZE)] = {};

        const int n_top_k = ne40;
        for (int tile0 = 0; tile0 < n_top_k; tile0 += WARP_SIZE) {
            const int tile_len = min(WARP_SIZE, n_top_k - tile0);

            float KQ_max_new = KQ_max;
            for (int i = 0; i < NSTEPS; ++i) {
                const int cell_in_tile = (tid/NTHREADS_KQ)*NSTEPS + i;
                float partial = 0.0f;
                int cell = -1;
                if (cell_in_tile < tile_len) {
                    cell = identity ? tile0 + cell_in_tile : idx[tile0 + cell_in_tile];
                    const nv_bfloat162 * K_cell = (const nv_bfloat162 *) (K + cell*nb11);
#pragma unroll
                    for (int j = 0; j < NSTEPS_Q; ++j) {
#pragma unroll
                        for (int k = 0; k < 4; ++k) {
                            ggml_cuda_mad(partial, K_cell[lane_in_group*4 + k + 32*j], Q_bf16[j*4 + k]);
                        }
                    }
                }
                partial = warp_reduce_sum<NTHREADS_KQ>(partial);

                const float score = (cell_in_tile < tile_len) ? partial + __half2float(maskh[cell]) : -FLT_MAX/2.0f;
                KQ_max_new = fmaxf(KQ_max_new, score + FATTN_KQ_MAX_OFFSET);
                KQ_warp[cell_in_tile] = score;
            }

#pragma unroll
            for (int offset = NTHREADS_KQ; offset < WARP_SIZE; offset <<= 1) {
                KQ_max_new = fmaxf(KQ_max_new, __shfl_xor_sync(0xFFFFFFFF, KQ_max_new, offset, WARP_SIZE));
            }
            const float KQ_max_scale = expf(KQ_max - KQ_max_new);
            KQ_max = KQ_max_new;
            KQ_sum *= KQ_max_scale;
#pragma unroll
            for (int k = 0; k < D/(2*WARP_SIZE); ++k) {
                VKQ[k].x *= KQ_max_scale;
                VKQ[k].y *= KQ_max_scale;
            }

            const float KQ_reg = (tid < tile_len) ? expf(KQ_warp[tid] - KQ_max) : 0.0f;
            KQ_warp[tid] = KQ_reg;
            KQ_sum += warp_reduce_sum(KQ_reg);

#pragma unroll
            for (int c = 0; c < WARP_SIZE; ++c) {
                const float w = (c < tile_len) ? KQ_warp[c] : 0.0f;
                if (w != 0.0f) {
                    const int cell_c = (c < tile_len) ? (identity ? tile0 + c : idx[tile0 + c]) : 0;
                    const nv_bfloat162 * V_cell = (const nv_bfloat162 *) (V + cell_c*nb21);
#pragma unroll
                for (int k = 0; k < D/(2*WARP_SIZE); ++k) {
                    const nv_bfloat162 v = V_cell[tid + k*WARP_SIZE];
                    VKQ[k].x += __bfloat162float(__low2bfloat16(v))*w;
                    VKQ[k].y += __bfloat162float(__high2bfloat16(v))*w;
                }
                }
            }
        }

        float * dst_col = dst + ((sequence*int(ne01.z) + col)*ne02 + head)*D;
        float2 * dst2 = (float2 *) dst_col;
#pragma unroll
        for (int k = 0; k < D/(2*WARP_SIZE); ++k) {
            dst2[tid + k*WARP_SIZE] = make_float2(VKQ[k].x / KQ_sum, VKQ[k].y / KQ_sum);
        }
    }
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, idx_ptr, mask_ptr, dst_ptr, scale, logit_softcap, identity,
        ne00, ne01, ne02, ne03, nb01, nb02, nb03,
        ne10, ne11, ne12, ne13, nb11, nb12, nb13,
        nb21, nb22, nb23, ne30, ne31, ne33, nb31, nb33,
        ne40, ne41, ne43, nb41, nb43);
    NO_DEVICE_CODE;
#endif // FLASH_ATTN_AVAILABLE
}

template <int D, ggml_type type_KV>
static void ggml_cuda_flash_attn_qsa_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const bool identity) {
    const ggml_tensor * Q   = dst->src[0];
    const ggml_tensor * K   = dst->src[1];
    const ggml_tensor * V   = dst->src[2];
    const ggml_tensor * idx = dst->src[3];
    const ggml_tensor * mask = dst->src[4];

    cudaStream_t main_stream = ctx.stream();

    float scale = 1.0f;
    float logit_softcap = 0.0f;
    memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 1, sizeof(float));

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const dim3 blocks_num(Q->ne[1], 1, Q->ne[3]);
    const dim3 block_dim(WARP_SIZE, Q->ne[2], 1);

    const uint3 ne01 = init_fastdiv_values(Q->ne[1]);

    ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num, block_dim, 0, main_stream);
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        ggml_cuda_kernel_launch(flash_attn_qsa<D, type_KV, use_logit_softcap>, launch_params,
            (const char *) Q->data,
            (const char *) K->data,
            (const char *) V->data,
            (const int  *) idx->data,
            (const char *) mask->data,
            (float *) dst->data,
            scale, logit_softcap, identity,
            Q->ne[0], ne01,     Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
            K->ne[0], K->ne[1], K->ne[2], K->ne[3], K->nb[1], K->nb[2], K->nb[3],
            V->nb[1], V->nb[2], V->nb[3],
            mask->ne[0], mask->ne[1], mask->ne[3], mask->nb[1], mask->nb[3],
            idx->ne[0], idx->ne[1], idx->ne[3], idx->nb[1], idx->nb[3]);
    } else {
        constexpr bool use_logit_softcap = true;
        ggml_cuda_kernel_launch(flash_attn_qsa<D, type_KV, use_logit_softcap>, launch_params,
            (const char *) Q->data,
            (const char *) K->data,
            (const char *) V->data,
            (const int  *) idx->data,
            (const char *) mask->data,
            (float *) dst->data,
            scale, logit_softcap, identity,
            Q->ne[0], ne01,     Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
            K->ne[0], K->ne[1], K->ne[2], K->ne[3], K->nb[1], K->nb[2], K->nb[3],
            V->nb[1], V->nb[2], V->nb[3],
            mask->ne[0], mask->ne[1], mask->ne[3], mask->nb[1], mask->nb[3],
            idx->ne[0], idx->ne[1], idx->ne[3], idx->nb[1], idx->nb[3]);
    }
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_flash_attn_qsa(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);

    const ggml_tensor * Q   = dst->src[0];
    const ggml_tensor * K   = dst->src[1];
    const ggml_tensor * V   = dst->src[2];
    const ggml_tensor * idx = dst->src[3];
    const ggml_tensor * mask = dst->src[4];

    // GGML_CUDA_QSA_IDENTITY=1 forces idx = 0..n_top_k-1 (dense-equivalent) for validation.
    GGML_ASSERT(Q->type   == GGML_TYPE_F32);
    GGML_ASSERT(Q->nb[0]  == ggml_element_size(Q));
    GGML_ASSERT(K->nb[0]  == ggml_element_size(K));
    GGML_ASSERT(V->nb[0]  == ggml_element_size(V));
    GGML_ASSERT(idx->type == GGML_TYPE_I32);

    const int D = Q->ne[0];

    const bool identity = getenv("GGML_CUDA_QSA_IDENTITY") != nullptr;
    if (K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16) {
        switch (D) {
            case 64:  ggml_cuda_flash_attn_qsa_case< 64, GGML_TYPE_F16>(ctx, dst, identity); break;
            case 128: ggml_cuda_flash_attn_qsa_case<128, GGML_TYPE_F16>(ctx, dst, identity); break;
            case 256: ggml_cuda_flash_attn_qsa_case<256, GGML_TYPE_F16>(ctx, dst, identity); break;
            default: GGML_ABORT("unsupported head size");
        }
    } else if (K->type == GGML_TYPE_BF16 && V->type == GGML_TYPE_BF16) {
        switch (D) {
            case 64:  ggml_cuda_flash_attn_qsa_case< 64, GGML_TYPE_BF16>(ctx, dst, identity); break;
            case 128: ggml_cuda_flash_attn_qsa_case<128, GGML_TYPE_BF16>(ctx, dst, identity); break;
            case 256: ggml_cuda_flash_attn_qsa_case<256, GGML_TYPE_BF16>(ctx, dst, identity); break;
            default: GGML_ABORT("unsupported head size");
        }
    } else {
        GGML_ABORT("unsupported K/V type");
    }
}

bool ggml_cuda_flash_attn_qsa_supported(int device, const ggml_tensor * dst) {
    GGML_ASSERT(dst->op == GGML_OP_FLASH_ATTN_QSA);

    const ggml_tensor * Q   = dst->src[0];
    const ggml_tensor * K   = dst->src[1];
    const ggml_tensor * V   = dst->src[2];
    const ggml_tensor * idx = dst->src[3];

    GGML_UNUSED(device);

    const bool kv_ok =
        (K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16) ||
        (K->type == GGML_TYPE_BF16 && V->type == GGML_TYPE_BF16);

    return Q->type == GGML_TYPE_F32 && idx->type == GGML_TYPE_I32 && kv_ok &&
        (Q->ne[0] == 64 || Q->ne[0] == 128 || Q->ne[0] == 256);
}
