#include "common.cuh"
#include "ggml.h"

void ggml_cuda_op_hc_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_hc_combine(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_hc_combine_mix_fusable(const ggml_tensor * comb, const ggml_tensor * mix);
void ggml_cuda_op_hc_combine_mix(ggml_backend_cuda_context & ctx, ggml_tensor * comb, ggml_tensor * mix);
