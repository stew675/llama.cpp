#include "common.cuh"

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst,
                                         const ggml_tensor *         merge = nullptr);
int ggml_cuda_moe_weighted_reduction_try_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
void ggml_cuda_mwr_fold_mark(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
// GGML_CUDA_MWR_EXACT (default 1; =0 restores the pre-r324 fma-contracted kernels): no fma contraction in the expert
// weighted sum, so the fold and the standalone kernels round exactly as the graph's MUL / VIEW / ADD chain does.
// ONE knob for both paths on purpose -- they replace each other at the same node, so an arm where only one is exact
// would compare two different arithmetics and blame the wrong one.
bool ggml_cuda_mwr_exact();
// GGML_CUDA_MWR_DEBUG is on (the fold census: FOLD / FOLD ASIDE / FOLD DECLINE / FUSED / EXACT)
bool ggml_cuda_mwr_debug();
