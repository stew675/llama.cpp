#include "moe-weighted-reduction.cuh"

// float4 quad variant: only valid when every expert row starts 16B-aligned, i.e. when
// n_embd % 4 == 0 (checked by the launcher). Real models have n_embd % 4 == 0, so the
// aligned path below covers them unchanged; the scalar kernel is the fallback for
// n_embd % 4 != 0, where a vectorized kernel could neither cover the trailing 1-3
// columns nor assume per-row alignment.
static __global__ void moe_weighted_reduction_f32_vec4(const float * __restrict__ experts,
                                                       const float * __restrict__ expert_scale,
                                                       const float * __restrict__ weights,
                                                       float * __restrict__ dst,
                                                       const int64_t n_embd,
                                                       const int     n_expert_used) {
    const int64_t n_embd4 = n_embd / 4;
    const int64_t token   = blockIdx.x;
    const int64_t i4      = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (i4 >= n_embd4) {
        return;
    }

    const int64_t col        = i4 * 4;
    const uint64_t first_row = (uint64_t) token * n_expert_used;
    const float    first_scale = expert_scale != nullptr ? expert_scale[first_row] : 1.0f;
    float4 sum;
    {
        const float4 e = *reinterpret_cast<const float4 *>(experts + first_row * n_embd + col);
        const float  w = weights[first_row];
        sum.x = (e.x * first_scale) * w;
        sum.y = (e.y * first_scale) * w;
        sum.z = (e.z * first_scale) * w;
        sum.w = (e.w * first_scale) * w;
    }

    for (int expert = 1; expert < n_expert_used; ++expert) {
        const uint64_t row   = first_row + expert;
        const float   scale = expert_scale != nullptr ? expert_scale[row] : 1.0f;
        const float4  e     = *reinterpret_cast<const float4 *>(experts + row * n_embd + col);
        const float   w     = weights[row];
        sum.x += (e.x * scale) * w;
        sum.y += (e.y * scale) * w;
        sum.z += (e.z * scale) * w;
        sum.w += (e.w * scale) * w;
    }
    reinterpret_cast<float4 *>(dst + token * n_embd + col)[0] = sum;
}

// scalar kernel, bounds-checked per column (upstream body): correct for any n_embd.
static __global__ void moe_weighted_reduction_f32(const float * __restrict__ experts,
                                                  const float * __restrict__ expert_scale,
                                                  const float * __restrict__ weights,
                                                  float * __restrict__ dst,
                                                  const int64_t n_embd,
                                                  const int     n_expert_used) {
    const int64_t token = blockIdx.x;
    const int64_t col   = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= n_embd) {
        return;
    }

    const uint64_t first_row   = (uint64_t) token * n_expert_used;
    const float    first_scale = expert_scale != nullptr ? expert_scale[first_row] : 1.0f;
    float          sum         = (experts[first_row * n_embd + col] * first_scale) * weights[first_row];

    for (int expert = 1; expert < n_expert_used; ++expert) {
        const uint64_t row   = first_row + expert;
        const float   scale = expert_scale != nullptr ? expert_scale[row] : 1.0f;
        sum += (experts[row * n_embd + col] * scale) * weights[row];
    }
    dst[token * n_embd + col] = sum;
}

static void launch_moe_weighted_reduction(const float * experts,
                                          const float * expert_scale,
                                          const float * weights,
                                          float *       dst,
                                          int64_t       n_embd,
                                          int64_t       n_tokens,
                                          int           n_expert_used,
                                          cudaStream_t  stream) {
    constexpr int threads = 256;
    if (n_embd % 4 == 0) {
        const int64_t n_embd4 = n_embd / 4;
        const dim3 blocks(n_tokens, (n_embd4 + threads - 1) / threads, 1);
        moe_weighted_reduction_f32_vec4
            <<<blocks, threads, 0, stream>>>(experts, expert_scale, weights, dst, n_embd, n_expert_used);
    } else {
        const dim3 blocks(n_tokens, (n_embd + threads - 1) / threads, 1);
        moe_weighted_reduction_f32
            <<<blocks, threads, 0, stream>>>(experts, expert_scale, weights, dst, n_embd, n_expert_used);
    }
}

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst) {
    GGML_ASSERT(experts->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(expert_scale == nullptr || expert_scale->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(experts));
    GGML_ASSERT(ggml_is_contiguous(weights));
    GGML_ASSERT(expert_scale == nullptr || ggml_is_contiguous(expert_scale));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t n_embd        = experts->ne[0];
    const int64_t n_expert_used = experts->ne[1];
    const int64_t n_tokens      = experts->ne[2] * experts->ne[3];
    cudaStream_t  stream        = ctx.stream();

    launch_moe_weighted_reduction((const float *) experts->data,
                                  expert_scale ? (const float *) expert_scale->data : nullptr,
                                  (const float *) weights->data,
                                  (float *) dst->data, n_embd, n_tokens, (int) n_expert_used, stream);
    CUDA_CHECK(cudaGetLastError());
}
