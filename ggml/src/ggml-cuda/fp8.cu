// FP8 E4M3 mul_mat host launcher (native FP8 hardware only).

#include "common.cuh"
#include "fp8.cuh"

#include <cstdint>

// Pre-quantize F32 activations to fp8 staging, then run the native FP8 mul_mat
// kernel. Mirrors the quantize_mmq_q8_1_cuda + mul_mat_q flow.
void ggml_cuda_mul_mat_fp8(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F8_E4M3);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src1));

    GGML_TENSOR_BINARY_OP_LOCALS;

    GGML_ASSERT(ne00 % GGML_FP8_TILE_K == 0);
    GGML_ASSERT(ne02 == 1 && ne03 == 1); // weights are unbatched (broadcast over src1 batches)

    const int64_t k = ne00;
    const int64_t m = ne01;
    const int64_t n = ne1;
    const int64_t n_batch = ne12 * ne13;

    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const char  * src0_d = (const char  *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float       *)  dst->data;

    const int64_t n_col_blocks = k / GGML_FP8_TILE_K;

    // token padding: the wmma tiles span 16 tokens; out-of-range reads must be zeros
    const int64_t n_pad = (n + (GGML_FP8_NWARPS * GGML_FP8_TILE_N) - 1) / (GGML_FP8_NWARPS * GGML_FP8_TILE_N) * (GGML_FP8_NWARPS * GGML_FP8_TILE_N);

    ggml_cuda_pool_alloc<uint8_t> src1_q(ctx.pool(), k * n_pad);
    ggml_cuda_pool_alloc<float>   src1_s(ctx.pool(), n_pad * n_col_blocks);
    uint8_t * src1_q_d = src1_q.get();
    float   * src1_s_d = src1_s.get();
    GGML_ASSERT(src1_q_d != nullptr);
    GGML_ASSERT(src1_s_d != nullptr);
    if (n_pad > n) {
        CUDA_CHECK(cudaMemsetAsync(src1_q_d + k * n, 0, k * (n_pad - n), stream));
        CUDA_CHECK(cudaMemsetAsync(src1_s_d + n * n_col_blocks, 0, (n_pad - n) * n_col_blocks * sizeof(float), stream));
    }

    for (int64_t ib = 0; ib < n_batch; ++ib) {
        const float * src1_b = src1_d + ib * k * n;
        float       * dst_b  = dst_d  + ib * m * n;

        {
            const dim3 num_blocks(n_col_blocks, n, 1);
            const dim3 block_size(GGML_FP8_NTHREADS, 1, 1);
            quantize_fp8<<<num_blocks, block_size, 0, stream>>>(src1_b, src1_q_d, src1_s_d, k, n, n_col_blocks, n_pad);
            CUDA_CHECK(cudaGetLastError());
        }

        // the fp8 kernels are compiled for the RDNA4 device pass only; on the host
        // side the launch stubs always exist, so dispatch on the runtime cc
        if (GGML_CUDA_CC_IS_RDNA4(cc)) {
            const dim3 block_dims(GGML_FP8_NTHREADS, 1, 1);
            if (n <= GGML_FP8_GEMV_MAX_N) {
                // dot4 GEMV: one warp per output row, one CTA per 4 rows and token
                const dim3 grid_dims((m + 3) / 4, n, 1);
                const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
                ggml_cuda_kernel_launch(mul_mat_fp8_gemv, launch_params, src0_d, src1_q_d, src1_s_d, dst_b, k, m, n, n_col_blocks, n_pad);
            } else {
                const dim3 grid_dims((n + (GGML_FP8_NWARPS * GGML_FP8_TILE_N) - 1) / (GGML_FP8_NWARPS * GGML_FP8_TILE_N), (m + GGML_FP8_TILE_M - 1) / GGML_FP8_TILE_M, 1);
                const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
                ggml_cuda_kernel_launch(mul_mat_fp8_wmma, launch_params, src0_d, src1_q_d, src1_s_d, dst_b, k, m, n, n_col_blocks, n_pad);
            }
            continue;
        }

        // Scalar fallback (CUDA sm_89+)
        const dim3 block_dims(32, 8, 1);
        const dim3 grid_dims((n + 31) / 32, (m + 7) / 8, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
        ggml_cuda_kernel_launch(mul_mat_fp8_scalar, launch_params, src0_d, src1_q_d, src1_s_d, dst_b, k, m, n, n_col_blocks, n_pad);
    }
}
