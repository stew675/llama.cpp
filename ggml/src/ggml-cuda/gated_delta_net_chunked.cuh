// Fused chunked kernel for GGML_OP_GATED_DELTA_NET prefill (n_tokens > 1, K == 1, non-KDA).
//
// The chunked recurrence is sequential in the state, so it runs as TWO launches (see the .cu):
// a state-scan pass (block per (v-head, seq), looping chunks, state through a scratch buffer)
// and an output pass (same grid, reading the per-chunk states, writing only the attention out).
//
// The per-chunk math is the algebra of llm_build_delta_net_base::build_delta_net_chunking, with
// gcs the chunk-local inclusive cumsum of the gate, K_b = K . beta, V_b = V . beta, Q_s = scale . Q,
// and decay[i][j] = e^{gcs[j] - gcs[i]} on the upper triangle (i <= j):
//
//     A    = (I + strict_upper(K^T K_b . decay))^-1      (unit upper, the KKT solve)
//     U    = K_b^T S                                     (state times keys)
//     v_n  = V_b^T A - diag(e^g) A^T U                   (v_new = v_corr - predicted)
//     o    = e^g . S^T Q_s + KQ^T v_n                    (decay + intra-chunk attention)
//     S'   = e^{g_last} S + K diag(e^{g_last - g}) v_n   (state update)
//
// Decay is computed DIRECTLY per kept pair: the exponent gcs[j]-gcs[i] is <= 0 on every kept
// pair (gates are negative, padding keeps the cumsum flat), so it can never overflow even for
// the test harness's pathological gates. Padding is handled by clamping token reads to the last
// real token and bt[pad] = el[pad] = 0, which zero the gram entries and the state-update
// factors the way ggml_pad's zero columns do.

#pragma once
#include "common.cuh"

void ggml_cuda_op_gated_delta_net_chunked(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// bf16/WMMA tensor-core variant (S_v == 128 only; near-lossless). Selected by
// GGML_CUDA_GDN_CHUNKED_BF16=1 in the dispatch seam; the fp32 chunked path stays the default.
void ggml_cuda_op_gated_delta_net_chunked_bf16(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
