#include "common.cuh"

#define CUDA_ROPE_BLOCK_SIZE 256

void ggml_cuda_op_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rope_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * set_rows);

void ggml_cuda_op_rms_norm_mul_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * rms_norm, ggml_tensor * mul, ggml_tensor * rope, ggml_tensor * set_rows,
        ggml_tensor * fwht = nullptr, ggml_tensor * k_set_rows = nullptr, ggml_tensor * v_set_rows = nullptr, void * capture = nullptr,
        ggml_tensor * v_fwht = nullptr);

// Quantized-KV form: Q norm+rope+FWHT (q_had's rotated rows) and the K norm+rope+FWHT + q8_0 K / V cache writes (V rotated
// in the kernel when v_had != nullptr) of the same layer in one launch.
void ggml_cuda_op_rms_norm_mul_rope_qk_pair_fwht(ggml_backend_cuda_context & ctx,
        ggml_tensor * q_norm, ggml_tensor * q_mul, ggml_tensor * q_rope, ggml_tensor * q_had,
        ggml_tensor * k_norm, ggml_tensor * k_mul, ggml_tensor * k_rope, ggml_tensor * k_had,
        ggml_tensor * k_set_rows, ggml_tensor * v_set_rows, ggml_tensor * v_had);

void ggml_cuda_op_rms_norm_mul_rope_qk_pair(ggml_backend_cuda_context & ctx,
        ggml_tensor * q_norm, ggml_tensor * q_mul, ggml_tensor * q_rope,
        ggml_tensor * k_norm, ggml_tensor * k_mul, ggml_tensor * k_rope, ggml_tensor * k_set_rows, ggml_tensor * v_set_rows);
