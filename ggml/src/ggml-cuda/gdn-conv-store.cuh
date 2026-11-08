#pragma once
#include "common.cuh"

// The Gated DeltaNet conv of a decode / verify step (1..16 tokens a sequence, 1..8 sequences, RDNA4) as one kernel, on the graph that stores
// the rollback snapshots with one SET_ROWS (upstream #31660, build_conv_state_store):
//
//   GET_ROWS (state) -> CONCAT(state, x^T) -> VIEW (windows) + VIEW (slot order, of an ARANGE/SCALE/CAST chain built per layer
//   or once per graph) + VIEW (slots) -> SET_ROWS
//   ... -> SSM_CONV -> SILU -> RMS_NORM/SCALE (q) + RMS_NORM/SCALE (k)
//
// The kernel reads the state (the cache row itself when the GET_ROWS is deferred), x and the taps; writes the conv + silu
// output, the normalized q / k (the two SCALE outputs) and the snapshot windows straight into the cache slots.  The rest of the
// chain launches nothing.  Bit-identical to the chain (the conv and the norm in rdna-boosts' direct conv arithmetic).
// GGML_CUDA_FUSE_GDN_CONV_STORE=0: off.
struct ggml_cuda_gdn_store_match {
    int concat = -1, set_rows = -1, conv = -1, last = -1;   // node indices; last = the k norm's SCALE
    const ggml_tensor * gather = nullptr;   // the GET_ROWS the state roots at (nullptr: none)
    const ggml_tensor * state  = nullptr;   // [3, C, S] F32 contiguous
    const ggml_tensor * x      = nullptr;   // [C, T, S] F32 contiguous
    const ggml_tensor * w      = nullptr;   // [4, C] F32 contiguous
    ggml_tensor *       y      = nullptr;   // the SILU output [C, T, S]
    ggml_tensor *       q_out  = nullptr;   // the q norm's SCALE [128, nk, T, S]
    ggml_tensor *       k_out  = nullptr;   // the k norm's SCALE
    const ggml_tensor * slots  = nullptr;   // the SET_ROWS destination [3, K, C, S]: slot planes nb[1] apart, cells nb[3]
    int64_t C = 0, T = 0, S = 1, K = 0, Kw = 0, col0 = 0;
    int     nk = 0;
    float   eps = 0.0f, scale = 0.0f;
    bool    private_chain = false;   // the slot order chain is this layer's own (taken with the CONCAT) or shared
};

// with_data = false: the structure only (graph_optimize, before allocation)
bool ggml_cuda_gdn_store_match_at_concat(const ggml_cgraph * cgraph, int i, ggml_cuda_gdn_store_match & m, bool with_data);
int  ggml_cuda_gdn_store_try_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
struct ggml_cuda_gdn_gather_index;
// the conv-state GET_ROWS at node_idx feeds a store match that reads the cache row itself
bool ggml_cuda_gdn_store_gather_deferrable(const ggml_cgraph * cgraph, int node_idx, const ggml_cuda_gdn_gather_index & idx);
