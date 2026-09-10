#pragma once

#include "ggml.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda/common.cuh"

void ggml_cuda_indexer_score(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_indexer_score_supported(int device, const ggml_tensor * dst);
void ggml_cuda_indexer_fill(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_indexer_fill_supported(int device, const ggml_tensor * dst);

// Prefill indexer head reduction: relu each head's block score, then sum the heads in graph order.
// Replaces the RELU (+ optional RESHAPE) + CONT + ADD chain the qwen4exp prefill emits, reading the
// scores once instead of H times.  Bit-identical (fmaxf + the same left-to-right add order).
struct ggml_cuda_idx_relu_sum_args {
    const ggml_tensor * score = nullptr;   // pre-relu block scores, physical [n_blocks, H, nt, ns] layout
    ggml_tensor *       dst   = nullptr;   // [n_blocks, nt*ns] F32 contiguous
    int                 heads = 0;
    int64_t             rows  = 0;         // nt * ns
};
bool ggml_cuda_idx_relu_sum_enabled();
void ggml_cuda_op_idx_relu_sum(ggml_backend_cuda_context & ctx, const ggml_cuda_idx_relu_sum_args & args);
