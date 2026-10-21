#pragma once

#include "common.cuh"

// NOTE: this header relies on the includer (mmvq.cu) having already defined
// `enum mmvq_parameter_table_id` and `MMVQ_MAX_BATCH_SIZE`; it is included after the
// enum so that the RDNA policy can name the table ids.

// RDNA (gfx11/gfx12) mmvq launch-parameter policy.
//
// These helpers own the *delivery's* RDNA policy only; each returns 0 when the
// table_id/type is not ours, so mmvq.cu can call them at the top of its
// upstream-derived selector and otherwise leave upstream's own logic untouched.
// That keeps our rebase surface to a single call line instead of a rewrite of
// calc_nwarps()/calc_rows_per_block().
//
// Band uniformity (GREEDY-PURITY.md sections 19/24/25): nwarps participates in the
// K-split accumulation order, so decode (ncols_dst == 1) and the speculative verify
// batch (ncols_dst 2..MMVQ_MAX_BATCH_SIZE) must use the same value.  Every branch
// below therefore keys on the *band* (ncols_dst <= MMVQ_MAX_BATCH_SIZE), not just
// ncols_dst == 1.

// nwarps override for the block-selector table.  0 = no override.
//
// `small_m` (RDNA3_0 only, wip/nwarps): the ksplit weight kernel's wide (8-warp)
// block is a *large-M* win and a small-M loss on RDNA3_0 (35B-A3B / gemma-26B dense
// layers).  It is a weight-row-scoped choice -- M is fixed per tensor, so every
// width W = 1..8 of that tensor takes the same branch, which is what keeps the
// decode == verify invariant (GREEDY-PURITY.md section 19) intact.  The K-split
// order of mul_mat_vec_q_ksplit is `kbx = tid/(qi/vdr) + m*blocks_per_iter`, i.e.
// a per-thread residue class that does not depend on ncols_dst / rows_per_block,
// so one nwarps value per shape is exactly the width-invariance requirement.
// Pass the same value to the launch dims and the kernel (a host/kernel mismatch
// here launched 1 warp for an 8-warp kernel; GREEDY-PURITY.md section 20).
static constexpr __host__ __device__ int mmvq_rdna_nwarps(ggml_type type, int ncols_dst,
                                                          mmvq_parameter_table_id table_id,
                                                          bool small_m = false) {
    if (table_id == MMVQ_PARAMETERS_RDNA4) {
        // Band-uniform nwarps=1 for the whole mmvq band (ncols_dst 1..8).  The Q8_0
        // *weight* kernel gets a per-(type,K) override from calc_nwarps_weight(); every
        // other launch (including the GDN/SSM and shared-expert fusions) stays at 1.
        (void) type;
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA3_0) {
        // RDNA3 (W7900): stricter whitelist than RDNA4.  Q2_K / Q5_K / IQ4_XS regress in
        // full quant sweeps and stay 1; Q6_K uses 8 (gfx1100 sweep 2026-08-28, R5).
        if (ncols_dst <= MMVQ_MAX_BATCH_SIZE) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                    return small_m ? 1 : 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA3_5) {
        // gfx1151 (Strix Halo iGPU): nwarps=2 wins on Q8_0 (~+0.6% decode on
        // Qwen3.6-35B-A3B Q8_0, swept 2025-08); everything else stays 1.
        if (ncols_dst <= MMVQ_MAX_BATCH_SIZE) {
            switch (type) {
                case GGML_TYPE_Q8_0:
                    return 2;
                default:
                    return 1;
            }
        }
        return 1;
    }
    return 0;
}
