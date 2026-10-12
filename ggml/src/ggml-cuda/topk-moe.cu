#include "ggml-cuda/common.cuh"
#include "ggml.h"
#include "topk-moe.cuh"

#include <cmath>
#include <initializer_list>

// Kernel config struct - passed by value to CUDA kernel
struct topk_moe_config {
    bool use_sigmoid;
    bool use_sqrt_softplus;
    bool with_norm;
    bool delayed_softmax;
    bool sorted;            // per-lane sorted candidates (see topk_moe_sort_lane); set by the launcher
};

// Warp-local softmax used for both the pre-top-k logits and the post-top-k delayed path.
//
// Must be bit-identical to the generic softmax -> block_reduce chain, otherwise whether this
// fusion fires (an address-overlap decision in ggml_cuda_check_fusion_memory_ranges) changes the
// model output.  The generic soft_max_f32 kernel launches one thread per column (a power of two
// >= ncols, capped at 1024) and reduces with a per-warp butterfly followed by a cross-warp
// butterfly.  Here lane l holds column `l + i*WARP_SIZE`, so the generic per-warp phase is exactly
// warp_reduce_sum(vals[i]); the cross-warp phase then re-reduces those per-warp results.  The
// fused kernel used to do a single flat warp reduction, which differs by ULPs.
template <int experts_per_thread, bool use_limit>
__device__ void softmax_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
    float max_val = -INFINITY;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            max_val = max(max_val, vals[i]);
        }
    }

    // max is associative and exact, so any reduction order gives the same result
    max_val = warp_reduce_max(max_val);

    if (experts_per_thread == 1) {
        // the generic kernel uses a single warp for ncols <= WARP_SIZE: one warp reduction
        const bool  active  = !use_limit || (lane < limit);
        vals[0]             = active ? expf(vals[0] - max_val) : 0.f;
        const float sum     = warp_reduce_sum(vals[0]);
        const float inv_sum = 1.0f / sum;

        if (active) {
            vals[0] *= inv_sum;
        }
        return;
    }

    // phase 1: per "virtual warp" (the generic kernel's consecutive 32-column group)
    float partial = 0.f;
#pragma unroll
    for (int w = 0; w < experts_per_thread; w++) {
        const int   idx    = lane + w * WARP_SIZE;
        const bool  active = !use_limit || (idx < limit);
        vals[w]            = active ? expf(vals[w] - max_val) : 0.f;
        const float rw     = warp_reduce_sum(vals[w]);
        if (lane == w) {
            partial = rw;
        }
    }

    // phase 2: re-reduce the per-warp results (lanes >= experts_per_thread contribute 0)
    const float sum     = warp_reduce_sum(lane < experts_per_thread ? partial : 0.f);
    const float inv_sum = 1.0f / sum;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            vals[i] *= inv_sum;
        }
    }
}

template <int experts_per_thread, bool use_limit>
__device__ void sigmoid_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        vals[i]           = active ? 1.f / (1.f + expf(-vals[i])) : -INFINITY;
    }
}

template <int experts_per_thread, bool use_limit>
__device__ void sqrt_softplus_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        vals[i]           = active ? sqrtf(vals[i] > 20.0f ? vals[i] : logf(1.0f + expf(vals[i]))) : -INFINITY;
    }
}

// Lane exchange of the top-k argmax butterfly.  On RDNA (GGML_CUDA_DPP_XOR) a DPP / permlanex16 move instead of
// __shfl_xor_sync, which lowers to ds_bpermute through LDS: the argmax runs n_expert_used rounds of 5 dependent
// stages with two exchanges each, so the LDS round trip was most of the kernel.  Same values, so the same result.
template <int mask, typename T>
static __device__ __forceinline__ T topk_moe_xor(const T x) {
#if defined(GGML_CUDA_DPP_XOR)
    return ggml_cuda_xor_lane<mask>(x);
#else
    return __shfl_xor_sync(0xFFFFFFFF, x, mask, WARP_SIZE);
#endif // defined(GGML_CUDA_DPP_XOR)
}

template <int mask>
static __device__ __forceinline__ void topk_moe_argmax_stage(float & max_val, int & max_expert) {
    const float val    = topk_moe_xor<mask>(max_val);
    const int   expert = topk_moe_xor<mask>(max_expert);
    if (val > max_val || (val == max_val && expert < max_expert)) {
        max_val    = val;
        max_expert = expert;
    }
}

template <int mask>
static __device__ __forceinline__ void topk_moe_argmax_stage_bias(float & max_val, float & max_val_s, int & max_expert) {
    const float val    = topk_moe_xor<mask>(max_val);
    const float val_s  = topk_moe_xor<mask>(max_val_s);
    const int   expert = topk_moe_xor<mask>(max_expert);
    if (val_s > max_val_s || (val_s == max_val_s && expert < max_expert)) {
        max_val    = val;
        max_val_s  = val_s;
        max_expert = expert;
    }
}

// The argmax rounds pick experts in the order (value descending, lower expert index first on ties).  With `sorted`,
// each candidate becomes one 64-bit key that orders the same way: the float's order-preserving bit pattern above the
// inverted expert index (unique, so no two keys are equal), and each lane sorts its keys once (bitonic network, E a
// power of 2).  A round is then a 5-stage max over the lane heads and a shift on the winning lane, with one unsigned
// compare per step instead of the float/index tie logic.  The keys order exactly as the float compare does for any
// non-NaN value except -0.0 vs +0.0, so the path is limited to the gating outputs (softmax, sigmoid, sqrt-softplus:
// never -0.0; NaN is already sanitized), and recovers the float bit-exactly from the key.
static __device__ __forceinline__ uint64_t topk_moe_key(const float v, const int expert) {
    const uint32_t u = __float_as_uint(v);
    const uint32_t o = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return ((uint64_t) o << 32) | (uint32_t) ~(uint32_t) expert;
}

static __device__ __forceinline__ float topk_moe_key_val(const uint64_t key) {
    const uint32_t o = (uint32_t) (key >> 32);
    return __uint_as_float((o & 0x80000000u) ? (o & 0x7FFFFFFFu) : ~o);
}

static __device__ __forceinline__ int topk_moe_key_expert(const uint64_t key) {
    return (int) ~(uint32_t) key;
}

template <int mask>
static __device__ __forceinline__ uint64_t topk_moe_key_max_stage(const uint64_t key) {
    const uint32_t lo = topk_moe_xor<mask>((int) (uint32_t) key);
    const uint32_t hi = topk_moe_xor<mask>((int) (uint32_t) (key >> 32));
    const uint64_t other = ((uint64_t) hi << 32) | lo;
    return other > key ? other : key;
}

template <int E>
static __device__ __forceinline__ void topk_moe_sort_keys(uint64_t (&key)[E]) {
    static_assert((E & (E - 1)) == 0, "bitonic network needs a power of 2");
#pragma unroll
    for (int k = 2; k <= E; k <<= 1) {
#pragma unroll
        for (int j = k >> 1; j > 0; j >>= 1) {
#pragma unroll
            for (int i = 0; i < E; ++i) {
                const int l = i ^ j;
                if (l > i) {
                    // descending within (i & k) == 0 blocks, ascending in the others: a descending sort overall
                    const uint64_t a = key[i];
                    const uint64_t b = key[l];
                    const bool swap = (i & k) == 0 ? b > a : a > b;
                    key[i] = swap ? b : a;
                    key[l] = swap ? a : b;
                }
            }
        }
    }
}

/*
    This kernel does the following:
    1. optionally softmax over the logits per token [n_experts, n_tokens]
    2. argmax reduce over the top-k (n_experts_used) logits
    3. write weights + ids to global memory
    4. optionally normalize the weights or apply softmax over the selected logits

    It is intended as fusion of softmax->top-k->get_rows pipeline for MoE models
*/
template <int n_experts, bool has_bias>
__launch_bounds__(TOPK_MOE_ROWS_PER_BLOCK * WARP_SIZE, 1)
__global__ void topk_moe_cuda(const float *         logits,
                              float *               weights,
                              int32_t *             ids,
                              float *               bias,
                              const int             n_rows,
                              const int             n_expert_used,
                              const float           clamp_val,
                              const float           scale_val,
                              const topk_moe_config config) {
#if defined(GGML_USE_MUSA)
    // MUSA: every warp of a partially filled block must reach the barrier below.
    const int row = MIN(blockIdx.x * blockDim.y + threadIdx.y, n_rows - 1);
#else
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
#endif // defined(GGML_USE_MUSA)
    if (row >= n_rows) {
        return;
    }

    logits += n_experts * row;
    weights += n_expert_used * row;
    ids += n_experts * row;

    constexpr int experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;

    float wt[experts_per_thread];

    // Initialize all slots to -INFINITY
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        wt[i] = -INFINITY;
    }

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int i = 0; i < n_experts; i += WARP_SIZE) {
        const int expert  = i + threadIdx.x;
        wt[i / WARP_SIZE] = (n_experts % WARP_SIZE == 0 || expert < n_experts) ? logits[expert] : -INFINITY;
    }

    // Weights and IDs can alias logits, so wait until every row in the block reads its logits.
    __syncthreads();

    if (!config.delayed_softmax) {
        if (config.use_sigmoid) {
           sigmoid_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        } else if (config.use_sqrt_softplus) {
           sqrt_softplus_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        } else {
           softmax_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        }
    }

    // Sanitize NaN to -FLT_MAX so the iterative argmax produces unique expert IDs.
    // NaN comparisons always return false, which would cause the same expert to be
    // selected repeatedly. -FLT_MAX compares normally and is still excluded by the
    // -INFINITY sentinel used after each selection round.
    // More relevant for the cuBLAS path. See https://github.com/ggml-org/llama.cpp/issues/19659
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        if (__isnanf(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    // selection_wt is only needed when bias is present (selection uses wt + bias)
    // when no bias, we use wt directly for both selection and weight values
    [[maybe_unused]] float selection_wt[has_bias ? experts_per_thread : 1];

    if constexpr (has_bias) {
#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            selection_wt[i] = -INFINITY;
        }
#pragma unroll
        for (int i = 0; i < n_experts; i += WARP_SIZE) {
            const int expert = i + threadIdx.x;
            selection_wt[i / WARP_SIZE] =
                (n_experts % WARP_SIZE == 0 || expert < n_experts) ? wt[i / WARP_SIZE] + bias[expert] : -INFINITY;
        }
    }

    //at this point, each thread holds either a portion of the softmax distribution
    //or the raw logits. We do the argmax reduce over n_expert_used, each time marking
    //the expert weight as -inf to exclude from the next iteration

    float wt_sum = 0.f;

    float output_weights[experts_per_thread];

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        output_weights[i] = 0.f;
    }

    ggml_cuda_pdl_lc();

    constexpr bool sorted_ok = !has_bias && experts_per_thread >= 2 && n_experts % WARP_SIZE == 0 &&
                               (experts_per_thread & (experts_per_thread - 1)) == 0;
    const bool use_sorted = sorted_ok && config.sorted && !config.delayed_softmax;
    if constexpr (sorted_ok) {
        if (use_sorted) {
            uint64_t key[experts_per_thread];
#pragma unroll
            for (int i = 0; i < experts_per_thread; i++) {
                key[i] = topk_moe_key(wt[i], threadIdx.x + i * WARP_SIZE);
            }
            topk_moe_sort_keys<experts_per_thread>(key);

            for (int k = 0; k < n_expert_used; k++) {
                static_assert(WARP_SIZE == 32, "five butterfly stages");
                uint64_t best = key[0];
                best = topk_moe_key_max_stage<16>(best);
                best = topk_moe_key_max_stage<8>(best);
                best = topk_moe_key_max_stage<4>(best);
                best = topk_moe_key_max_stage<2>(best);
                best = topk_moe_key_max_stage<1>(best);

                if (key[0] == best) {
#pragma unroll
                    for (int i = 0; i < experts_per_thread - 1; i++) {
                        key[i] = key[i + 1];
                    }
                    key[experts_per_thread - 1] = 0;   // below every candidate key
                }

                const float max_val    = topk_moe_key_val(best);
                const int   max_expert = topk_moe_key_expert(best);

                if ((k & (WARP_SIZE - 1)) == threadIdx.x) {
                    output_weights[k / WARP_SIZE] = max_val;
                }

                if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
                    ids[k] = max_expert;
                    if (config.with_norm) {
                        wt_sum += max_val;
                    }
                }
            }
        }
    }

    for (int k = 0; k < (use_sorted ? 0 : n_expert_used); k++) {
        float max_val    = wt[0];
        int   max_expert = threadIdx.x;

        if constexpr (has_bias) {
            float max_val_s = selection_wt[0];

#pragma unroll
            for (int i = 1; i < experts_per_thread; i++) {
                const int expert = threadIdx.x + i * WARP_SIZE;
                if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && selection_wt[i] > max_val_s) {
                    max_val    = wt[i];
                    max_val_s  = selection_wt[i];
                    max_expert = expert;
                }
            }

            static_assert(WARP_SIZE == 32, "five butterfly stages");
            topk_moe_argmax_stage_bias<16>(max_val, max_val_s, max_expert);
            topk_moe_argmax_stage_bias<8>(max_val, max_val_s, max_expert);
            topk_moe_argmax_stage_bias<4>(max_val, max_val_s, max_expert);
            topk_moe_argmax_stage_bias<2>(max_val, max_val_s, max_expert);
            topk_moe_argmax_stage_bias<1>(max_val, max_val_s, max_expert);

            if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
                selection_wt[max_expert / WARP_SIZE] = -INFINITY;
            }
        } else {
#pragma unroll
            for (int i = 1; i < experts_per_thread; i++) {
                const int expert = threadIdx.x + i * WARP_SIZE;
                if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && wt[i] > max_val) {
                    max_val    = wt[i];
                    max_expert = expert;
                }
            }

            static_assert(WARP_SIZE == 32, "five butterfly stages");
            topk_moe_argmax_stage<16>(max_val, max_expert);
            topk_moe_argmax_stage<8>(max_val, max_expert);
            topk_moe_argmax_stage<4>(max_val, max_expert);
            topk_moe_argmax_stage<2>(max_val, max_expert);
            topk_moe_argmax_stage<1>(max_val, max_expert);

            if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
                wt[max_expert / WARP_SIZE] = -INFINITY;
            }
        }

        if ((k & (WARP_SIZE - 1)) == threadIdx.x) {
            output_weights[k / WARP_SIZE] = max_val;
        }

        if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
            ids[k] = max_expert;
            if (config.with_norm) {
                wt_sum += max_val;
            }
        }
    }

    if (config.with_norm) {
        if (n_expert_used <= WARP_SIZE) {
            // match the generic sum_rows -> reduce_rows_f32 order exactly: lane j holds the
            // weight of selection j in output_weights[0] (all k < WARP_SIZE rounds write index 0),
            // and the generic kernel reduces with a warp butterfly over lanes 0..k-1
            wt_sum = warp_reduce_sum(threadIdx.x < n_expert_used ? output_weights[0] : 0.f);
        } else {
            wt_sum = warp_reduce_sum(wt_sum);
        }
        wt_sum = max(wt_sum, clamp_val);
    }

    if (config.delayed_softmax) {
        softmax_warp_inplace<experts_per_thread, true>(output_weights, n_expert_used, threadIdx.x);
    }

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int idx = i * WARP_SIZE + threadIdx.x;
        if (idx < n_expert_used) {
            float w = output_weights[i];
            if (config.with_norm) {
                w = w / wt_sum; // generic div (weights / weights_sum), not a reciprocal multiply
            }
            weights[idx] = w * scale_val;
        }
    }
}

template<bool has_bias>
static void launch_topk_moe_cuda(ggml_backend_cuda_context & ctx,
                                 const float *               logits,
                                 float *                     weights,
                                 int32_t *                   ids,
                                 float *                     bias,
                                 const int                   n_rows,
                                 const int                   n_expert,
                                 const int                   n_expert_used,
                                 const float                 clamp_val,
                                 const float                 scale_val,
                                 const topk_moe_config       config) {
    GGML_ASSERT(!(config.with_norm && config.delayed_softmax) &&
                "delayed softmax is not supported with weight normalization");
    const int    rows_per_block = TOPK_MOE_ROWS_PER_BLOCK;
    dim3         grid_dims((n_rows + rows_per_block - 1) / rows_per_block, 1, 1);
    dim3         block_dims(WARP_SIZE, rows_per_block, 1);
    cudaStream_t stream = ctx.stream();
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

    switch (n_expert) {
        case 1:
            ggml_cuda_kernel_launch(topk_moe_cuda<1, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 2:
            ggml_cuda_kernel_launch(topk_moe_cuda<2, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 4:
            ggml_cuda_kernel_launch(topk_moe_cuda<4, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 8:
            ggml_cuda_kernel_launch(topk_moe_cuda<8, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 16:
            ggml_cuda_kernel_launch(topk_moe_cuda<16, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 32:
            ggml_cuda_kernel_launch(topk_moe_cuda<32, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 64:
            ggml_cuda_kernel_launch(topk_moe_cuda<64, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 128:
            ggml_cuda_kernel_launch(topk_moe_cuda<128, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 256:
            ggml_cuda_kernel_launch(topk_moe_cuda<256, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 288: // StepFun 3.7
            ggml_cuda_kernel_launch(topk_moe_cuda<288, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 512:
            ggml_cuda_kernel_launch(topk_moe_cuda<512, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 576:
            ggml_cuda_kernel_launch(topk_moe_cuda<576, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        default:
            GGML_ASSERT(false && "fatal error");
            break;
    }
}

void ggml_cuda_op_topk_moe(ggml_backend_cuda_context &     ctx,
                           const ggml_tensor *             logits,
                           ggml_tensor *                   weights,
                           ggml_tensor *                   ids,
                           const ggml_tensor *             clamp,
                           const ggml_tensor *             scale,
                           const ggml_tensor *             bias,
                           const ggml_cuda_topk_moe_args & args) {
    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    const int n_experts = logits->ne[0];
    const int n_rows    = logits->ne[1];

    const float * logits_d  = (const float *) logits->data;
    float *       weights_d = (float *) weights->data;
    int32_t *     ids_d     = (int32_t *) ids->data;
    float *       bias_d    = bias ? (float *) bias->data : nullptr;

    float scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;

    GGML_ASSERT(ids->nb[1] / ggml_type_size(ids->type) == (size_t) n_experts);

    const int n_expert_used = weights->ne[1];

    const bool with_norm = clamp != nullptr;

    float clamp_val = -INFINITY;
    if (clamp) {
        clamp_val = ggml_get_op_params_f32(clamp, 0);
    }

    topk_moe_config config;
    config.use_sigmoid       = args.sigmoid;
    config.use_sqrt_softplus = args.sqrt_softplus;
    config.with_norm         = with_norm;
    config.delayed_softmax   = args.delayed_softmax;
    // GGML_CUDA_TOPK_SORTED=0: the per-round scan of every candidate (same result)
    static const bool sorted = !getenv("GGML_CUDA_TOPK_SORTED") || atoi(getenv("GGML_CUDA_TOPK_SORTED")) != 0;
    config.sorted            = sorted;

    if (bias) {
        launch_topk_moe_cuda<true>(ctx, logits_d, weights_d, ids_d, bias_d, n_rows, n_experts, n_expert_used, clamp_val,
                             scale_val, config);
    } else {
        launch_topk_moe_cuda<false>(ctx, logits_d, weights_d, ids_d, bias_d, n_rows, n_experts, n_expert_used, clamp_val,
                             scale_val, config);
    }
}

bool ggml_cuda_should_use_topk_moe(const ggml_tensor * gating_op,
                                   const ggml_tensor * weights,
                                   const ggml_tensor * logits,
                                   const ggml_tensor * ids) {
    // must match an instantiation of launch_topk_moe_cuda: a power of 2 up to 512,
    // or one of the non-power-of-2 expert counts of supported models
    const int n_expert = ids->nb[1] / ids->nb[0];
    if (((n_expert & (n_expert - 1)) != 0 || n_expert > 512) && n_expert != 288 && n_expert != 576) {
        return false;
    }

    if (!ggml_is_contiguous(weights) || !ggml_is_contiguous(logits)) {
        return false;
    }

    if (gating_op->op == GGML_OP_SOFT_MAX) {
        const ggml_tensor * softmax  = gating_op;
        float               scale    = 1.0f;
        float               max_bias = 0.0f;

        memcpy(&scale, (const float *) softmax->op_params + 0, sizeof(float));
        memcpy(&max_bias, (const float *) softmax->op_params + 1, sizeof(float));

        if (!ggml_is_contiguous(softmax->src[0])) {
            return false;
        }

        if (scale != 1.0f || max_bias != 0.0f) {
            return false;
        }

        // don't fuse when masks or sinks are present
        if (softmax->src[1] || softmax->src[2]) {
            return false;
        }
    } else if (gating_op->op == GGML_OP_UNARY) {
        ggml_unary_op op = ggml_get_unary_op(gating_op);

        if (op != GGML_UNARY_OP_SIGMOID && op != GGML_UNARY_OP_SOFTPLUS) {
            return false;
        }
    }

    return true;
}
