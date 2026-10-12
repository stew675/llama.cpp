#pragma once

#include "common.cuh"

// RDNA mmvq launch-parameter policy (part 2): per-type/per-shape overrides that build on the
// upstream calc_nwarps()/calc_rows_per_block() kept in mmvq.cu.  mmvq.cu includes this after
// calc_rows_per_block() so the calls resolve.

// nwarps for a dense mmvq *weight* launch (the ksplit kernel).  A Q8_0 weight with a
// short K (the MoE attention qkv/gate and the lm_head, K < 4096) wins from the
// pre-2026-09-11 wider block; the long-K Q8_0 hybrid projections (dense 27B
// attn_v/ssm_out, K >= 5120) lose from it and shift the MTP arithmetic, so they stay
// at 1.  The choice is per tensor shape (K is fixed for a weight), so W = 1..8 of the
// same tensor still agree.  Fusion ops (GDN/SSM, shared-expert, the gate fusions)
// MUST keep plain calc_nwarps -- their nwarps is a single-token reduction-order anchor.
static constexpr __host__ __device__ int calc_nwarps_weight(ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id, bool long_k) {
    if (table_id == MMVQ_PARAMETERS_RDNA4 && type == GGML_TYPE_Q8_0 && !long_k && ncols_dst <= MMVQ_MAX_BATCH_SIZE) {
        return 8;
    }
    return calc_nwarps(type, ncols_dst, table_id);
}

// rows_per_block override for dispatch-selected kernels (e.g. the small-K MoE
// down projection on RDNA, where a single 544 B row per block underutilizes the
// memory system). 0 = use calc_rows_per_block().
static constexpr __host__ __device__ int calc_rows_per_block_override(int rows_per_block, int ncols_dst, int table_id, bool small_k, int nwarps) {
    return rows_per_block > 0 ? rows_per_block : calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
}

// rows per block for a dense mmvq *weight* launch (the ksplit kernel).  On RDNA4 a
// multi-token (verify) batch computes several rows per block so the ncols_dst q8_1
// activation columns are loaded once per block instead of once per row.  Each row's
// accumulation (thread mapping, K order, warp reduction) is unchanged, so W = 1..8
// stay bit-identical.
#ifndef GGML_MMVQ_RDNA4_WEIGHT_RPB
#define GGML_MMVQ_RDNA4_WEIGHT_RPB 4
#endif
// Multi-row blocks only pay off while the launch still has enough blocks to fill the GPU:
// with fewer than this many blocks (e.g. 1024-row K/V or 48-row SSM projections at 4 rows)
// the one-row launch is faster.
#ifndef GGML_MMVQ_RDNA4_WEIGHT_MIN_BLOCKS
#define GGML_MMVQ_RDNA4_WEIGHT_MIN_BLOCKS 512
#endif
// Largest grid of the single-token dense decode launch; larger weights loop over their rows (see launch_ksplit_rpb).
#ifndef GGML_MMVQ_RDNA4_DECODE_GRID
#define GGML_MMVQ_RDNA4_DECODE_GRID 1792
#endif
// The types that take that row loop (Q3_K loses 8-35 % from it); the others compile to the one-row kernel as before.
static constexpr __host__ __device__ bool mmvq_decode_grid_loop_type(ggml_type type) {
    return type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K || type == GGML_TYPE_IQ4_XS ||
           type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ2_S;
}
// gfx1100 (RDNA3_0) rows per block for a Q4_K verify batch (ncols_dst 2..8), same row-block
// decomposition as the RDNA4 rule below: each row's accumulation is unchanged (bit-identical).
// gfx1100 sweep (nwarps 1/2/4/8 x rows 1/2/4, Qwen3.8-27B Q4_K_M, MTP n-max 3): 4 rows at
// nwarps 1 was the fastest bit-identical point (~+7 % MTP decode), greedy text unchanged;
// nwarps > 1 changes the K-split order and was no faster.  With the 32-element dot product
// (vec_dot_q4_K_q8_1_vdr4_dense) 2 rows is faster: W = 4 kernel times vs 4 rows, ffn_gate/up
// 71.4 -> 69.4 us, attn_qkv 41.8 -> 39.3, attn_q 48.3 -> 46.5; 8 rows is slower everywhere.
// The multi-row guard in mmvq.cu is only compiled for types where RDNA4 also takes >1 row:
// extend it before adding other types here.
#ifndef GGML_MMVQ_RDNA3_Q4K_WEIGHT_RPB
#define GGML_MMVQ_RDNA3_Q4K_WEIGHT_RPB 2
#endif
static constexpr __host__ __device__ int calc_rows_per_block_weight(ggml_type type, int ncols_dst, int table_id, bool small_k, int nwarps) {
    if (table_id == MMVQ_PARAMETERS_RDNA3_0 && type == GGML_TYPE_Q4_K && ncols_dst >= 2 && ncols_dst <= MMVQ_MAX_BATCH_SIZE) {
        return GGML_MMVQ_RDNA3_Q4K_WEIGHT_RPB;
    }
    // Q6_K (8-warp blocks) peaks at 2 rows: gfx1100 W = 4 kernel times vs 1 row, Qwen3.8-27B: lm_head 248320 x 5120
    // -18 %, attn_qkv 10240 x 5120 -14 %, ffn_down 5120 x 17408 -3 %; 4 rows is slower on the last two, 8 spills.
    if (table_id == MMVQ_PARAMETERS_RDNA3_0 && type == GGML_TYPE_Q6_K && ncols_dst >= 2 && ncols_dst <= MMVQ_MAX_BATCH_SIZE) {
        return 2;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA4 && ncols_dst >= 2 && ncols_dst <= MMVQ_MAX_BATCH_SIZE) {
        // Multi-row blocks only pay off for one-wave blocks: the 8-wave Q8_0 short-K block
        // (calc_nwarps_weight) was measured slower at every row count (K = 2880, 8 tokens:
        // +21..41 %, issue #71). Two rows beat one from 4 tokens on (gfx1201, K 640 / 2560 / 2880, 512..10240
        // rows: -14..-17 % summed at 4..6 tokens, -3 % at 8, worst shape +3.7 %) but not at 2..3 tokens.
        // Single-token decode does not come here (see launch_ksplit).
        if (nwarps > 1) {
            return ncols_dst >= 4 ? 2 : 1;
        }
        constexpr int rpb_max = GGML_MMVQ_RDNA4_WEIGHT_RPB;
        switch (type) {
            // gfx1201 sweep (rows 1/2/4 x every type x n = 2..8 x 4 shapes): these gain
            // nothing from more rows (NVFP4 loses 40-60 %, Q1_0 ~10 %) and keep one row.
            case GGML_TYPE_NVFP4:
            case GGML_TYPE_Q1_0:
                return 1;
            // these peak at 2 rows
            case GGML_TYPE_IQ2_XXS:
            case GGML_TYPE_IQ2_XS:
            case GGML_TYPE_IQ3_XXS:
            case GGML_TYPE_Q2_0:
                return rpb_max < 2 ? rpb_max : 2;
            // q5_1 is the exception: two rows regress the narrow verify widths on ROCm 7.14.
            // One row is 6.6-7.1 % faster at n = 2..4 (gfx1201, m=4096 k=14336), while two rows
            // stay ahead from n = 5 (up to +16 % at n = 8), so the row count follows the width.
            // rows-per-block is a weight-row block decomposition, not a K reduction knob.
            case GGML_TYPE_Q5_1:
                return ncols_dst <= 4 ? 1 : (rpb_max < 2 ? rpb_max : 2);
            // Q2_K's dot product exceeds the 256-VGPR budget at 4 rows x 7..8 columns and
            // spills to scratch; it keeps 2 rows there.
            case GGML_TYPE_Q2_K:
                return ncols_dst > 6 && rpb_max > 2 ? 2 : rpb_max;
            default:
                return rpb_max;
        }
    }
    return calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
}
