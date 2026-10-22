#include "common.cuh"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always the state row size, set in-kernel)
struct ggml_cuda_ssm_scan_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots
};

void ggml_cuda_op_ssm_scan(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// same op, but writes the state snapshot(s) into the cache instead of dst (see ggml_cuda_try_ssm_scan_cache_fusion)
void ggml_cuda_op_ssm_scan_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                       ggml_cuda_ssm_scan_fused_cache cache);
int ggml_cuda_ssm_scan_cache_cpy_try_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);

int ggml_cuda_try_ssm_scan_cache_fusion(const ggml_cgraph * cgraph, int node_idx, ggml_cuda_ssm_scan_fused_cache & fused_state_cpy);
