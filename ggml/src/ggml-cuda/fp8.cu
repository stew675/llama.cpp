// FP8 E4M3 mul_mat host launcher (native FP8 hardware only).

#include "common.cuh"
#include "fp8.cuh"

#include <cstdint>

// Pre-quantize F32 activations to fp8 staging, then run the native FP8 mul_mat
// kernel. Mirrors the quantize_mmq_q8_1_cuda + mul_mat_q flow.

// Lazily repack the block_f8_e4m3 weight layout into the wmma-friendly layout
// (fp8 bytes contiguous [m_pad][k] + separate scales [m_pad][n_col_blocks]), cached
// per tensor in the backend context. The wmma kernel reads 16-B aligned staging
// loads this way; the GEMV and scalar paths keep reading the original block layout.
// Safe under CUDA graphs: the first call always happens during the direct-execution
// warmup (capture only starts after warmup completes), so cudaMalloc is legal here.
static const ggml_backend_cuda_context::fp8_repack_buf & ggml_cuda_fp8_repack(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, cudaStream_t stream) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    const int64_t k = src0->ne[0];
    const int64_t m = src0->ne[1];
    const int64_t n_col_blocks = k / GGML_FP8_TILE_K;
    const int64_t m_pad = (m + GGML_FP8_CTA_M - 1) / GGML_FP8_CTA_M * GGML_FP8_CTA_M;

    auto & cache = ctx.fp8_repacks;
    auto it = cache.find(src0->data);
    if (it != cache.end()) {
        const ggml_backend_cuda_context::fp8_repack_buf * buf = it->second.get();
        if (buf->m == m && buf->k == k && buf->n_col_blocks == n_col_blocks) {
            return *buf;
        }
        cache.erase(it); // shape changed - rebuild
    }

    ggml_cuda_set_device(ctx.device);

    auto buf = std::make_unique<ggml_backend_cuda_context::fp8_repack_buf>();
    buf->m = m;
    buf->k = k;
    buf->n_col_blocks = n_col_blocks;

    CUDA_CHECK(cudaMalloc(&buf->q, m_pad * k));
    CUDA_CHECK(cudaMalloc(&buf->s, m_pad * n_col_blocks * sizeof(float)));

    // zero the padded tail rows: the wmma staging has no bounds predicates
    if (m_pad > m) {
        CUDA_CHECK(cudaMemsetAsync(buf->q + m * k, 0, (m_pad - m) * k, stream));
        CUDA_CHECK(cudaMemsetAsync(buf->s + m * n_col_blocks, 0, (m_pad - m) * n_col_blocks * sizeof(float), stream));
    }

    // one warp per (row, k-block); grid.y covers the rows in chunks of 65535
    const dim3 grid_dims(n_col_blocks, std::min<int64_t>(m, 65535), 1);
    const dim3 block_dims(32, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    ggml_cuda_kernel_launch(fp8_repack_weights, launch_params, (const char *) src0->data, buf->q, buf->s, k, n_col_blocks, m);

    const ggml_backend_cuda_context::fp8_repack_buf * ret = buf.get();
    cache.emplace(src0->data, std::move(buf));
    return *ret;
}

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

    // token padding: the wmma CTA spans GGML_FP8_CTA_N tokens; out-of-range reads must be zeros
    const int64_t n_pad = (n + GGML_FP8_CTA_N - 1) / GGML_FP8_CTA_N * GGML_FP8_CTA_N;

    ggml_cuda_pool_alloc<uint8_t> src1_q(ctx.pool(), k * n_pad);
    ggml_cuda_pool_alloc<float>   src1_s(ctx.pool(), n_pad * n_col_blocks);
    uint8_t * src1_q_d = src1_q.get();
    float   * src1_s_d = src1_s.get();
    GGML_ASSERT(src1_q_d != nullptr);
    GGML_ASSERT(src1_s_d != nullptr);
    // only the wmma path reads the padded token region (its CTA spans
    // GGML_FP8_CTA_N tokens); the GEMV path reads tokens 0..n-1 only
    if (n_pad > n && n > GGML_FP8_GEMV_MAX_N) {
        CUDA_CHECK(cudaMemsetAsync(src1_q_d + k * n, 0, k * (n_pad - n), stream));
        CUDA_CHECK(cudaMemsetAsync(src1_s_d + n * n_col_blocks, 0, (n_pad - n) * n_col_blocks * sizeof(float), stream));
    }

    for (int64_t ib = 0; ib < n_batch; ++ib) {
        const float * src1_b = src1_d + ib * k * n;
        float       * dst_b  = dst_d  + ib * m * n;

        {
            const dim3 num_blocks(n_col_blocks, n, 1);
            const dim3 block_size(GGML_FP8_QUANT_NTHREADS, 1, 1);
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
                const dim3 gemv_block_dims(GGML_FP8_GEMV_NTHREADS, 1, 1);
                const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, gemv_block_dims, 0, stream);
                ggml_cuda_kernel_launch(mul_mat_fp8_gemv, launch_params, src0_d, src1_q_d, src1_s_d, dst_b, k, m, n, n_col_blocks, n_pad);
            } else {
                // CTA: 128 weight rows x 64 tokens (2x2 wmma tiles per warp, 8 warps),
                // register-staged k-block pipelining; grouped-M pid swizzle (aiter recipe)
                // weights are read from the repacked layout (16-B staging loads)
                const ggml_backend_cuda_context::fp8_repack_buf & rp = ggml_cuda_fp8_repack(ctx, src0, stream);
                const dim3 grid_dims(((m + GGML_FP8_CTA_M - 1) / GGML_FP8_CTA_M) * ((n + GGML_FP8_CTA_N - 1) / GGML_FP8_CTA_N), 1, 1);
                const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
                ggml_cuda_kernel_launch(mul_mat_fp8_wmma, launch_params, rp.q, rp.s, src1_q_d, src1_s_d, dst_b, k, m, n, n_col_blocks, n_pad);
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
