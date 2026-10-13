#include "common.cuh"

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst,
                                         const ggml_tensor *         merge = nullptr);
int ggml_cuda_moe_weighted_reduction_try_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
