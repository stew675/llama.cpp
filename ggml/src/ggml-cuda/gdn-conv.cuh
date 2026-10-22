#pragma once
#include "common.cuh"
struct ggml_cuda_gdn_conv_match {
    int concat_idx = -1, conv_idx = -1;   // node indices
    const ggml_tensor * x = nullptr;      // [C, T] F32 contiguous (root of the transpose view)
    const ggml_tensor * state = nullptr;  // [3, C] F32 contiguous
    const ggml_tensor * concat = nullptr; // [T+3, C] F32
    const ggml_tensor * w = nullptr;
    ggml_tensor * conv_out = nullptr;
    int64_t C = 0, T = 0, tail_from = 0;  // first concat column that must be materialized
    bool silu = false;
    // snapshot fold: the direct kernel also writes the 3-column snapshot tails, CONCAT and the snapshot CPYs launch nothing
    static constexpr int SNAP_MAX = 8;
    bool snap = false;
    int  n_snap = 0;
    int  snap_idx[SNAP_MAX] = {};         // node indices of the snapshot CPYs
    int  snap_col[SNAP_MAX] = {};         // first concat column each one copies
    ggml_tensor * snap_dst[SNAP_MAX] = {};
};
bool ggml_cuda_gdn_conv_match_at_concat(const ggml_cgraph * cgraph, int i, ggml_cuda_gdn_conv_match & m);
bool ggml_cuda_gdn_conv_pattern_at_concat(const ggml_cgraph * cgraph, int i, ggml_cuda_gdn_conv_match & m);
bool ggml_cuda_gdn_conv_match_at_conv(const ggml_cgraph * cgraph, int j, ggml_cuda_gdn_conv_match & m);
void ggml_cuda_gdn_conv_write_tail(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_conv_match & m);
struct gdn_conv_norm_args;
void ggml_cuda_gdn_conv_direct(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_conv_match & m, const gdn_conv_norm_args * norm = nullptr);
int  ggml_cuda_gdn_conv_try_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
int  ggml_cuda_gdn_conv_snap_try_fuse(const ggml_cgraph * cgraph, int i);
