#pragma once

#include "common.cuh"
#include "fattn-common.cuh"

// QSA sparse flash attention for qwen4exp: attend only over the cells the
// indexer's top-k names, instead of the whole KV cache.  The K/V cache is
// F16/BF16, Q8_0 or one of the q4_0/q4_1/q5_0/q5_1 nibble types (those are
// dequantized to F16 while a tile is staged into shared memory); the mask is the
// base kq_mask and is gathered at the idx positions.
//
// Layouts (after the same permutes the dense FA path applies):
//   q    [n_embd_head_q, n_tps, n_head_q, n_stream]  F32
//   k    [n_embd_head_k, n_kv,   n_head_kv, n_stream] F16/BF16/Q8_0/Q4_0/Q4_1/Q5_0/Q5_1
//   v    [n_embd_head_v, n_kv,   n_head_v,  n_stream] F16/BF16/Q8_0/Q4_0/Q4_1/Q5_0/Q5_1
//   idx  [n_top_k, n_tps, 1, n_stream] I32
//   mask [n_kv, n_tps, 1, n_stream] F16
//   dst  [n_embd_head_v, n_head_q, n_tps, n_stream] F32
//
// Each block handles ONE token column (up to QSA_MAX_HEADS=16 q-heads of a
// stream, one warp per head); each warp walks that column's top-k list in tiles.

void ggml_cuda_flash_attn_qsa(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_flash_attn_qsa_supported(int device, const ggml_tensor * dst);
