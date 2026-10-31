#pragma once

#include "common.cuh"

// wip/kv-residency (TODO #118) B-A: resolve a QSA selection through a page table (GGML_OP_KV_POOL).
// See kv-pool.cu for the writer rule this stands on.
void ggml_cuda_kv_pool(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
