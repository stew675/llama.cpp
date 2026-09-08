#pragma once

#include "ggml.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda/common.cuh"

void ggml_cuda_indexer_score(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_indexer_score_supported(int device, const ggml_tensor * dst);
void ggml_cuda_indexer_fill(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_indexer_fill_supported(int device, const ggml_tensor * dst);
