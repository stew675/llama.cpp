// The delivery's CUDA graph fusion dispatcher (matchers + per-concern try-fuse arms).
// Split out of ggml-cuda.cu (an upstream hub); declared here, defined in ggml-cuda-try-fuse.cu.
#pragma once

#include "common.cuh"

int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i);

// Also used by the graph optimizer in ggml-cuda.cu.
bool ggml_cuda_gdn_state_gather_deferrable(const ggml_cgraph * cgraph, int node_idx);
struct ggml_cuda_topk_moe_args;
bool ggml_cuda_topk_moe_fusion(const struct ggml_cgraph * cgraph, int node_idx, struct ggml_cuda_topk_moe_args & args);
