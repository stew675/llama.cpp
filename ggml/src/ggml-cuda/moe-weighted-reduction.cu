#include "moe-weighted-reduction.cuh"
#include "mmb.cuh"

// HIP's float4 is a struct, and __builtin_nontemporal_load/_store only accept builtin scalar or
// vector types, so use an ext_vector_type view of the same 16 bytes for the streaming access.
//
// The expert-row read is the streaming bulk (each element read once), so it uses a non-temporal
// LOAD; the small weights/expert_scale stay cached, and the dst STORE stays a normal store (a
// non-temporal store here measured +32 ms: it evicts the output the next op wants).  Measured
// -40 ms on this kernel at gfx1151 pp8192 (session 14).
typedef float moe_nt_f4 __attribute__((ext_vector_type(4)));

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
        const moe_nt_f4 e = __builtin_nontemporal_load(reinterpret_cast<const moe_nt_f4 *>(experts + first_row * n_embd + col));
        const float  w = weights[first_row];
        sum.x = (e[0] * first_scale) * w;
        sum.y = (e[1] * first_scale) * w;
        sum.z = (e[2] * first_scale) * w;
        sum.w = (e[3] * first_scale) * w;
    }

    for (int expert = 1; expert < n_expert_used; ++expert) {
        const uint64_t row   = first_row + expert;
        const float   scale = expert_scale != nullptr ? expert_scale[row] : 1.0f;
        const moe_nt_f4 e   = __builtin_nontemporal_load(reinterpret_cast<const moe_nt_f4 *>(experts + row * n_embd + col));
        const float   w     = weights[row];
        sum.x += (e[0] * scale) * w;
        sum.y += (e[1] * scale) * w;
        sum.z += (e[2] * scale) * w;
        sum.w += (e[3] * scale) * w;
    }
    reinterpret_cast<float4 *>(dst + token * n_embd + col)[0] = sum;
}

// BF16 expert outputs (GGML_CUDA_MMB_DOWN16): identical arithmetic, the inputs are the BF16-rounded
// down-GEMM results.  The producer (MUL_MAT_ID) stored BF16 in place over its F32 buffer, so the
// tensor metadata stays F32 while the bytes are BF16 - the caller checks ggml_cuda_mmb_is_bf16_only.
static __device__ __forceinline__ float moe_bf2f(const uint16_t h) { return __uint_as_float(((uint32_t) h) << 16); }

static __global__ void moe_weighted_reduction_bf16_v4(const uint16_t * __restrict__ experts,
                                                      const float * __restrict__ expert_scale,
                                                      const float * __restrict__ weights,
                                                      float * __restrict__ dst,
                                                      const int64_t n_embd,
                                                      const int     n_expert_used) {
    const int64_t token = blockIdx.x;
    const int64_t col4  = ((int64_t) blockIdx.y * blockDim.x + threadIdx.x) * 4;
    if (col4 >= n_embd) {
        return;
    }

    const uint64_t first_row   = (uint64_t) token * n_expert_used;
    const float    first_scale = expert_scale != nullptr ? expert_scale[first_row] : 1.0f;
    const float    w0          = weights[first_row];
    const ushort4  h0          = *(const ushort4 *)(experts + first_row * n_embd + col4);
    float4 sum;
    sum.x = (moe_bf2f(h0.x) * first_scale) * w0;
    sum.y = (moe_bf2f(h0.y) * first_scale) * w0;
    sum.z = (moe_bf2f(h0.z) * first_scale) * w0;
    sum.w = (moe_bf2f(h0.w) * first_scale) * w0;

    for (int expert = 1; expert < n_expert_used; ++expert) {
        const uint64_t row   = first_row + expert;
        const float    scale = expert_scale != nullptr ? expert_scale[row] : 1.0f;
        const float    w     = weights[row];
        const ushort4  h     = *(const ushort4 *)(experts + row * n_embd + col4);
        sum.x += (moe_bf2f(h.x) * scale) * w;
        sum.y += (moe_bf2f(h.y) * scale) * w;
        sum.z += (moe_bf2f(h.z) * scale) * w;
        sum.w += (moe_bf2f(h.w) * scale) * w;
    }
    reinterpret_cast<float4 *>(dst + token * n_embd + col4)[0] = sum;
}

__device__ __forceinline__ uint16_t moe_f2bf(const float f) {
    uint32_t u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}

// BF16 in, BF16 out: the only consumer is the fused HC combine, which reads block_out as BF16.
// `merge` (the other operand of the shared-expert ADD, F32) is folded into the same pass when set.
static __global__ void moe_weighted_reduction_bf16_v4_out(const uint16_t * __restrict__ experts,
                                                          const float * __restrict__ expert_scale,
                                                          const float * __restrict__ weights,
                                                          uint16_t * __restrict__ dst,
                                                          const float * __restrict__ merge,
                                                          const int64_t n_embd,
                                                          const int     n_expert_used) {
    const int64_t token = blockIdx.x;
    const int64_t col4  = ((int64_t) blockIdx.y * blockDim.x + threadIdx.x) * 4;
    if (col4 >= n_embd) {
        return;
    }
    const uint64_t first_row   = (uint64_t) token * n_expert_used;
    const float    first_scale = expert_scale != nullptr ? expert_scale[first_row] : 1.0f;
    const float    w0          = weights[first_row];
    const ushort4  h0          = *(const ushort4 *)(experts + first_row * n_embd + col4);
    float4 sum;
    sum.x = (moe_bf2f(h0.x) * first_scale) * w0;
    sum.y = (moe_bf2f(h0.y) * first_scale) * w0;
    sum.z = (moe_bf2f(h0.z) * first_scale) * w0;
    sum.w = (moe_bf2f(h0.w) * first_scale) * w0;

    for (int expert = 1; expert < n_expert_used; ++expert) {
        const uint64_t row   = first_row + expert;
        const float    scale = expert_scale != nullptr ? expert_scale[row] : 1.0f;
        const float    w     = weights[row];
        const ushort4  h     = *(const ushort4 *)(experts + row * n_embd + col4);
        sum.x += (moe_bf2f(h.x) * scale) * w;
        sum.y += (moe_bf2f(h.y) * scale) * w;
        sum.z += (moe_bf2f(h.z) * scale) * w;
        sum.w += (moe_bf2f(h.w) * scale) * w;
    }
    if (merge != nullptr) {
        const float4 m = *(const float4 *)(merge + token * n_embd + col4);
        sum.x += m.x; sum.y += m.y; sum.z += m.z; sum.w += m.w;
    }
    *(ushort4 *)(dst + token * n_embd + col4) =
        make_ushort4(moe_f2bf(sum.x), moe_f2bf(sum.y), moe_f2bf(sum.z), moe_f2bf(sum.w));
}

// F32 experts, BF16 out (same merge semantics as above)
static __global__ void moe_weighted_reduction_f32in_bf16out_v4(const float * __restrict__ experts,
                                                               const float * __restrict__ expert_scale,
                                                               const float * __restrict__ weights,
                                                               uint16_t * __restrict__ dst,
                                                               const float * __restrict__ merge,
                                                               const int64_t n_embd,
                                                               const int     n_expert_used) {
    const int64_t token = blockIdx.x;
    const int64_t col4  = ((int64_t) blockIdx.y * blockDim.x + threadIdx.x) * 4;
    if (col4 >= n_embd) {
        return;
    }
    const uint64_t first_row   = (uint64_t) token * n_expert_used;
    const float    first_scale = expert_scale != nullptr ? expert_scale[first_row] : 1.0f;
    const float    w0          = weights[first_row];
    const float4   e0          = *(const float4 *)(experts + first_row * n_embd + col4);
    float4 sum;
    sum.x = (e0.x * first_scale) * w0;
    sum.y = (e0.y * first_scale) * w0;
    sum.z = (e0.z * first_scale) * w0;
    sum.w = (e0.w * first_scale) * w0;

    for (int expert = 1; expert < n_expert_used; ++expert) {
        const uint64_t row   = first_row + expert;
        const float    scale = expert_scale != nullptr ? expert_scale[row] : 1.0f;
        const float    w     = weights[row];
        const float4   e     = *(const float4 *)(experts + row * n_embd + col4);
        sum.x += (e.x * scale) * w;
        sum.y += (e.y * scale) * w;
        sum.z += (e.z * scale) * w;
        sum.w += (e.w * scale) * w;
    }
    if (merge != nullptr) {
        const float4 m = *(const float4 *)(merge + token * n_embd + col4);
        sum.x += m.x; sum.y += m.y; sum.z += m.z; sum.w += m.w;
    }
    *(ushort4 *)(dst + token * n_embd + col4) =
        make_ushort4(moe_f2bf(sum.x), moe_f2bf(sum.y), moe_f2bf(sum.z), moe_f2bf(sum.w));
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
    float          sum         = (__builtin_nontemporal_load(experts + first_row * n_embd + col) * first_scale) * weights[first_row];

    for (int expert = 1; expert < n_expert_used; ++expert) {
        const uint64_t row   = first_row + expert;
        const float   scale = expert_scale != nullptr ? expert_scale[row] : 1.0f;
        sum += (__builtin_nontemporal_load(experts + row * n_embd + col) * scale) * weights[row];
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
                                         ggml_tensor *               dst,
                                         const ggml_tensor *         merge) {
    GGML_ASSERT(experts->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(expert_scale == nullptr || expert_scale->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(merge == nullptr || merge->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(experts));
    GGML_ASSERT(ggml_is_contiguous(weights));
    GGML_ASSERT(expert_scale == nullptr || ggml_is_contiguous(expert_scale));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(merge == nullptr || ggml_is_contiguous(merge));

    const int64_t n_embd        = experts->ne[0];
    const int64_t n_expert_used = experts->ne[1];
    const int64_t n_tokens      = experts->ne[2] * experts->ne[3];
    cudaStream_t  stream        = ctx.stream();

    // BF16 routed-down output for the BF16 HC block_out stream (LLAMA_HC_BLK16, default OFF): the
    // producer stored BF16 in place over the F32 buffer and the graph marked the tensor bf16-only,
    // and the fused HC combine reads it as BF16.  A shared-expert ADD after the reduction is folded
    // in as `merge` (the other ADD operand).
    if (ggml_cuda_mmb_blk16() && ggml_cuda_mmb_is_bf16_only(dst)) {
        GGML_ASSERT(n_embd % 4 == 0);
        const bool ein = ggml_cuda_mmb_is_bf16_only(experts);
        constexpr int threads = 256;
        const dim3 blocks(n_tokens, (n_embd / 4 + threads - 1) / threads, 1);
        const float * mrg = merge ? (const float *) merge->data : nullptr;
        if (ein) {
            moe_weighted_reduction_bf16_v4_out<<<blocks, threads, 0, stream>>>((const uint16_t *) experts->data,
                    expert_scale ? (const float *) expert_scale->data : nullptr,
                    (const float *) weights->data, (uint16_t *) dst->data, mrg,
                    n_embd, (int) n_expert_used);
        } else {
            moe_weighted_reduction_f32in_bf16out_v4<<<blocks, threads, 0, stream>>>((const float *) experts->data,
                    expert_scale ? (const float *) expert_scale->data : nullptr,
                    (const float *) weights->data, (uint16_t *) dst->data, mrg,
                    n_embd, (int) n_expert_used);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    GGML_ASSERT(merge == nullptr && "shared-expert merge is only fused on the BF16 output path");

    // BF16 routed-down output (GGML_CUDA_MMB_DOWN16, default OFF): the producer stored BF16 in place
    // over the F32 buffer and the graph marked the tensor bf16-only.
    if (ggml_cuda_mmb_down16() && ggml_cuda_mmb_is_bf16_only(experts)) {
        GGML_ASSERT(n_embd % 4 == 0);
        constexpr int threads = 256;
        const dim3 blocks(n_tokens, (n_embd / 4 + threads - 1) / threads, 1);
        moe_weighted_reduction_bf16_v4<<<blocks, threads, 0, stream>>>((const uint16_t *) experts->data,
                expert_scale ? (const float *) expert_scale->data : nullptr,
                (const float *) weights->data, (float *) dst->data,
                n_embd, (int) n_expert_used);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    launch_moe_weighted_reduction((const float *) experts->data,
                                  expert_scale ? (const float *) expert_scale->data : nullptr,
                                  (const float *) weights->data,
                                  (float *) dst->data, n_embd, n_tokens, (int) n_expert_used, stream);
    CUDA_CHECK(cudaGetLastError());
}
