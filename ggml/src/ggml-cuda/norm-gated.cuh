#pragma once

#include "common.cuh"

// Narrow-row RMS norm (and its sigmoid-gated form), ported from the other solution's
// norm-gated.cu (Phase-1 item 4 of wip/closing-the-gap).  The kernel processes 8 rows per
// 256-thread block instead of the one-block-per-row rms_norm_f32<256>, which is
// block-scheduling bound for the model's per-head norms (~786k rows of <= 256 columns).
struct ggml_cuda_norm_gated_match {
    const ggml_tensor * x;
    const ggml_tensor * w;
    const ggml_tensor * z;
    ggml_tensor * dst;
    float eps;
    int pre = -1;   // node index of the gate MUL_MAT to compute first (or -1)
};

int  ggml_cuda_norm_gated_match_at(const ggml_cgraph * cgraph, int i, ggml_cuda_norm_gated_match & m);
int  ggml_cuda_norm_rows_match_at (const ggml_cgraph * cgraph, int i, ggml_cuda_norm_gated_match & m);
void ggml_cuda_op_norm_gated(ggml_backend_cuda_context & ctx, const ggml_cuda_norm_gated_match & m);
