#pragma once

#include "common.cuh"

// Fused indexer expand + mask + top-k (see indexer-topk.cu for layouts).
void ggml_cuda_indexer_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_indexer_top_k_supported(int device, const ggml_tensor * dst);
