// CUDA graph fusion dispatcher: the delivery's matchers and the one-line-per-concern try-fuse
// arms.  Split out of ggml-cuda.cu so an upstream change to the shared graph/evaluate path
// collides with the single ggml_cuda_try_fuse() call, not with every arm body.

#include "ggml-cuda.h"
#include "ggml-impl.h"
#include "ggml-moe-weighted-reduction.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda/allreduce.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/acc.cuh"
#include "ggml-cuda/add-id.cuh"
#include "ggml-cuda/arange.cuh"
#include "ggml-cuda/argmax.cuh"
#include "ggml-cuda/argsort.cuh"
#include "ggml-cuda/binbcast.cuh"
#include "ggml-cuda/clamp.cuh"
#include "ggml-cuda/col2im-1d.cuh"
#include "ggml-cuda/concat.cuh"
#include "ggml-cuda/conv-transpose-1d.cuh"
#include "ggml-cuda/conv2d.cuh"
#include "ggml-cuda/conv2d-dw.cuh"
#include "ggml-cuda/conv2d-transpose.cuh"
#include "ggml-cuda/conv3d.cuh"
#include "ggml-cuda/convert.cuh"
#include "ggml-cuda/count-equal.cuh"
#include "ggml-cuda/cpy.cuh"
#include "ggml-cuda/cross-entropy-loss.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/diagmask.cuh"
#include "ggml-cuda/diag.cuh"
#include "ggml-cuda/fattn.cuh"
#include "ggml-cuda/fattn-qsa.cuh"
#include "ggml-cuda/indexer-topk.cuh"
#include "ggml-cuda/indexer-score.cuh"
#include "ggml-cuda/fwht.cuh"
#include "ggml-cuda/gdn-conv.cuh"
#include "ggml-cuda/ple-conv.cuh"
#include "ggml-cuda/getrows.cuh"
#include "ggml-cuda/im2col.cuh"
#include "ggml-cuda/mmf.cuh"
#include "ggml-cuda/mmb.cuh"
#include "ggml-cuda/moe-expert-cache.h"
#include "ggml-cuda/cpy-batch.cuh"
#include "ggml-cuda/mmq.cuh"
#include "ggml-cuda/mmvf.cuh"
#include "ggml-cuda/mmvq.cuh"
#include "ggml-cuda/moe-weighted-reduction.cuh"
#include "ggml-cuda/norm.cuh"
#include "ggml-cuda/norm-gated.cuh"
#include "ggml-cuda/opt-step-adamw.cuh"
#include "ggml-cuda/opt-step-sgd.cuh"
#include "ggml-cuda/out-prod.cuh"
#include "ggml-cuda/pad.cuh"
#include "ggml-cuda/pool2d.cuh"
#include "ggml-cuda/pool1d.cuh"
#include "ggml-cuda/quantize.cuh"
#include "ggml-cuda/rope.cuh"
#include "ggml-cuda/roll.cuh"
#include "ggml-cuda/scale.cuh"
#include "ggml-cuda/snake.cuh"
#include "ggml-cuda/softcap.cuh"
#include "ggml-cuda/softmax.cuh"
#include "ggml-cuda/ssm-conv.cuh"
#include "ggml-cuda/ssm-scan.cuh"
#include "ggml-cuda/sum.cuh"
#include "ggml-cuda/sumrows.cuh"
#include "ggml-cuda/top-k.cuh"
#include "ggml-cuda/mean.cuh"
#include "ggml-cuda/tsembd.cuh"
#include "ggml-cuda/topk-moe.cuh"
#include "ggml-cuda/unary.cuh"
#include "ggml-cuda/upscale.cuh"
#include "ggml-cuda/wkv.cuh"
#include "ggml-cuda/gla.cuh"
#include "ggml-cuda/gated_delta_net.cuh"
#include "ggml-cuda/dsv4-hc.cuh"
#include "ggml-cuda/hc-mix.cuh"
#include "ggml-cuda/hyperconn.cuh"
#include "ggml-cuda/set.cuh"
#include "ggml-cuda/set-rows.cuh"
#include "ggml-cuda/pad_reflect_1d.cuh"
#include "ggml-cuda/solve_tri.cuh"
#include "ggml-cuda/tri.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/fill.cuh"
#include "ggml-cuda/lightning-indexer.cuh"
#include "ggml-cuda/ggml-cuda-try-fuse.cuh"
#include "ggml.h"
#include <algorithm>
#include <numeric>
#include <array>
#include <atomic>
#include <charconv>
#include <cinttypes>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cfloat>
#include <initializer_list>
#include <limits>
#include <map>
#include <set>
#include <memory>
#include <mutex>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// RDNA3_5 (Strix Halo, gfx1151): the dense gate+up+GLU mmvq fusion is single-token-only
// (mmvq.cu restricts fusion to ncols_dst == 1) and its fused kernel does not reproduce the
// standalone mul_mat_vec_q arithmetic, so a 1-token decode and an n-token speculative verify
// batch of the same layer are not bit-identical - the decode==verify invariant greedy MTP
// depends on.  Measured 2026-09-12: W=1 8abc6206... vs W=8 453eaa61...; skipping it (together
// with the weighted-down MoE tail, gated in ggml_cuda_mul_mat_id_weighted_rdna3_5_ok) restores
// W=1..8 == 453eaa61... .  Skip it on that arch unless explicitly re-enabled for A/B.
static bool ggml_cuda_rdna3_5_dense_glu_disabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_ENABLE_RDNA3_5_SINGLE_TOKEN_FUSIONS");
        return env != nullptr && std::atoi(env) != 0;
    }();
    if (enabled) {
        return false;
    }
    return GGML_CUDA_CC_IS_RDNA3_5(ggml_cuda_info().devices[ggml_cuda_get_device()].cc);
}

static bool ggml_cuda_should_fuse_mul_mat_vec_f(const ggml_tensor * tensor) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool is_mul_mat_id = tensor->op == GGML_OP_MUL_MAT_ID;

    bool use_mul_mat_vec_f =
        (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16) &&
        src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

    const int cc        = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    use_mul_mat_vec_f = use_mul_mat_vec_f && ggml_cuda_should_use_mmvf(src0->type, cc, warp_size, src0->ne, src0->nb, is_mul_mat_id ? src1->ne[2] : src1->ne[1]);

    //we only support fusion for ncols_dst = 1
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] != 1) {
        return false;
    }


    return use_mul_mat_vec_f;
}

static bool ggml_cuda_should_fuse_rope_set_rows(const ggml_tensor * rope,
                                                const ggml_tensor * view,
                                                const ggml_tensor * set_rows) {
    // The fusion is selected by ggml_cuda_check_fusion_memory_ranges(), i.e. by buffer addresses, so
    // its rounding path must be identical to the unfused chain.  Keep a kill switch to bisect it.
    static const bool disabled = getenv("GGML_CUDA_DISABLE_ROPE_SET_ROWS") != nullptr;
    if (disabled) {
        return false;
    }

    if (rope->op != GGML_OP_ROPE || view->op != GGML_OP_VIEW || set_rows->op != GGML_OP_SET_ROWS) {
        return false;
    }
    // ne3 not tested
    if (rope->src[0]->ne[3] != 1) {
        return false;
    }

    if (set_rows->type != GGML_TYPE_F32 && set_rows->type != GGML_TYPE_F16 && set_rows->type != GGML_TYPE_BF16) {
        return false;
    }

    if (set_rows->src[1]->type != GGML_TYPE_I64) {
        return false;
    }

    // The view should flatten two dims of rope into one dim
    if (!ggml_is_contiguous(view) || view->ne[0] != rope->ne[0] * rope->ne[1]) {
        return false;
    }

    // Only norm/neox/multi shaders have the fusion code
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX &&
        mode != GGML_ROPE_TYPE_MROPE && mode != GGML_ROPE_TYPE_IMROPE) {
        return false;
    }

    return true;
}

static bool ggml_cuda_should_fuse_mul_q8_1(const ggml_tensor * mul,
                                           const ggml_tensor * mm) {
    if (mul->op != GGML_OP_MUL || (mm->op != GGML_OP_MUL_MAT && mm->op != GGML_OP_MUL_MAT_ID)) {
        return false;
    }

    // The matmul must run on the mmvq path (the only consumer of the arena).
    if (!ggml_cuda_should_fuse_mul_mat_vec_q(mm, true)) {
        return false;
    }

    if (mul->type != GGML_TYPE_F32 || !ggml_is_quantized(mm->src[0]->type)) {
        return false;
    }

    // The matmul's activation must be the MUL (or a no-op view of it).
    const ggml_tensor * mm_src1 = mm->src[1];
    const ggml_tensor * src1 = mm_src1;
    while (src1 != nullptr && src1->view_src != nullptr) {
        src1 = src1->view_src;
    }
    if (src1 != mul) {
        return false;
    }

    // Same-shape, contiguous F32 inputs; rows must align with Q8_1 blocks.
    if (mul->src[0]->type != GGML_TYPE_F32 || mul->src[1]->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(mul->src[0], mul->src[1]) ||
        !ggml_is_contiguous(mul->src[0]) || !ggml_is_contiguous(mul->src[1]) ||
        !ggml_is_contiguous(mul) || mul->ne[0] % QK8_1 != 0 ||
        ggml_nelements(mul) != ggml_nelements(mm_src1)) {
        return false;
    }

    return true;
}

// If the MUL output feeds a single mmvq matmul (directly or via a no-op
// reshape at the next node), return that matmul; otherwise return nullptr.
static const ggml_tensor * ggml_cuda_find_mul_q8_1_matmul(const ggml_cgraph * cgraph,
                                                          int mul_idx, const ggml_tensor * mul) {
    const int n = cgraph->n_nodes;
    const ggml_tensor * n1 = (mul_idx + 1 < n) ? cgraph->nodes[mul_idx + 1] : nullptr;
    if (n1 == nullptr) {
        return nullptr;
    }

    const ggml_tensor * mm = nullptr;
    if (n1->op == GGML_OP_MUL_MAT) {
        mm = n1;
    } else if (n1->op == GGML_OP_RESHAPE && mul_idx + 2 < n && cgraph->nodes[mul_idx + 2]->op == GGML_OP_MUL_MAT) {
        mm = cgraph->nodes[mul_idx + 2];
    } else {
        return nullptr;
    }

    // The MUL output must have exactly one consumer (the matmul or its view).
    int uses = 0;
    for (int j = 0; j < n; ++j) {
        const ggml_tensor * t = cgraph->nodes[j];
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (t->src[s] == mul) {
                uses++;
            }
        }
    }
    if (uses != 1) {
        return nullptr;
    }

    if (!ggml_cuda_should_fuse_mul_q8_1(mul, mm)) {
        return nullptr;
    }
    return mm;
}

static bool ggml_cuda_should_fuse_rms_norm_mul_rope(const ggml_tensor * rms_norm,
                                                    const ggml_tensor * mul,
                                                    const ggml_tensor * rope) {
    // Address-selected like the rope_set_rows fusion above; kill switch for bisection.
    static const bool disabled = getenv("GGML_CUDA_DISABLE_RMS_NORM_MUL_ROPE") != nullptr;
    if (disabled) {
        return false;
    }
    if (rms_norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || rope->op != GGML_OP_ROPE) {
        return false;
    }

    if (rms_norm->src[0]->type != GGML_TYPE_F32 || rms_norm->type != GGML_TYPE_F32 ||
        mul->src[0]->type != GGML_TYPE_F32 || mul->src[1]->type != GGML_TYPE_F32 ||
        mul->type != GGML_TYPE_F32 || rope->type != GGML_TYPE_F32) {
        return false;
    }

    if (rope->src[0] != mul) {
        return false;
    }

    //if rms norm is the B operand, then we don't handle broadcast
    if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
        return false;
    }

    if (!ggml_are_same_shape(rms_norm, mul)) {
        return false;
    }

    //rms_norm kernel assumes contiguous rows
    if (!ggml_is_contiguous_rows(rms_norm->src[0]) ||
        !ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
        return false;
    }

    // the fused kernel handles the norm/neox rope modes, and M-RoPE / IMROPE (rope_multi's section-selected
    // positions, NEOX pairing; not VISION).  GGML_CUDA_DISABLE_ROPE_MROPE_FUSE keeps M-RoPE on the unfused chain.
    static const bool mrope_disabled = getenv("GGML_CUDA_DISABLE_ROPE_MROPE_FUSE") != nullptr;
    const int mode = ((const int32_t *) rope->op_params)[2];
    const bool mrope = mode == GGML_ROPE_TYPE_MROPE || mode == GGML_ROPE_TYPE_IMROPE;
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX && !(mrope && !mrope_disabled)) {
        return false;
    }

    const int n_dims = ((const int32_t *) rope->op_params)[1];
    if (n_dims % 2 != 0 || rope->src[0]->ne[0] % 2 != 0) {
        return false;
    }

    // ggml_rope_set_offset is not yet supported in the fused kernel
    const int n_offs = ((const int32_t *) rope->op_params)[15];
    if (n_offs != 0) {
        return false;
    }

    return true;
}

// match gated_delta_net + the strided cpy that scatters its state snapshots into the cache
// (slot i -> rollback group i, slot 0 newest), so the kernel can write them and skip the cpy.

// match ssm_scan + the strided cpy that scatters its state snapshots into the cache, so the kernel writes them and skips the cpy

// GET_ROWS that gathers a GATED_DELTA_NET input state from the recurrent cache (build_rs): the GDN can read the cache
// rows itself (row ids = the gather's src[1]), so the gather is skipped (GGML_CUDA_FUSE_GDN_STATE_GATHER=0: off).
// Only when the row ids are a graph output (the allocator never reuses them; llama flags rs_s_copy so), the GDN is
// in this graph, in the decode/verify band (single sequence, <= 16 tokens: the sequential kernel), every other reader
// of the gathered rows is empty (the zero-row extra-states copy of a single sequence), and no node in between writes
// the cache tensor.
static const ggml_tensor * gdn_gather_root_of(const ggml_tensor * t) {
    while (t != nullptr && t->view_src != nullptr) {
        t = t->view_src;
    }
    return t;
}

// the gather-side conditions: the cache root, or nullptr when the gather does not qualify
static const ggml_tensor * gdn_gather_candidate(const ggml_tensor * gr) {
    if (gr->op != GGML_OP_GET_ROWS || gr->type != GGML_TYPE_F32 || (gr->flags & GGML_TENSOR_FLAG_OUTPUT) ||
        gr->src[0]->type != GGML_TYPE_F32 || gr->src[0]->nb[0] != sizeof(float) || gr->src[0]->ne[0] != gr->ne[0] ||
        gr->src[1]->type != GGML_TYPE_I32 || gr->src[1]->nb[0] != sizeof(int32_t) || !ggml_is_contiguous(gr)) {
        return nullptr;
    }
    // the row ids are read when the GDN runs, so they must outlive the gather: graph outputs are never reused
    if (!(gdn_gather_root_of(gr->src[1])->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return nullptr;
    }
    return gdn_gather_root_of(gr->src[0]);
}

const ggml_tensor * ggml_cuda_gdn_gather_candidate(const ggml_cgraph * cgraph, int node_idx) {
    return gdn_gather_candidate(cgraph->nodes[node_idx]);
}

// the GDN-side conditions
static bool gdn_gather_gdn_ok(const ggml_tensor * gr, const ggml_tensor * gdn) {
    if (gdn == nullptr || (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }
    const ggml_tensor * src_v = gdn->src[2];
    const ggml_tensor * state = gdn->src[5];
    return src_v->ne[2] <= 16 && src_v->ne[3] == 1 && state->data == gr->data &&
           ggml_nelements(state) == gr->ne[0] && gr->ne[0] == src_v->ne[0] * src_v->ne[0] * src_v->ne[1];
}

bool ggml_cuda_gdn_state_gather_deferrable(const ggml_cgraph * cgraph, int node_idx) {
    const ggml_tensor * gr = cgraph->nodes[node_idx];
    const ggml_tensor * cache_root = gdn_gather_candidate(gr);
    if (cache_root == nullptr) {
        return false;
    }
    const auto root_of = gdn_gather_root_of;

    const ggml_tensor * gdn = nullptr;
    for (int j = node_idx + 1; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_is_empty(n)) {
            continue;
        }
        bool reads_gr = n->view_src != nullptr && root_of(n) == gr;
        for (int s = 0; s < GGML_MAX_SRC && !reads_gr; ++s) {
            reads_gr = n->src[s] != nullptr && root_of(n->src[s]) == gr;
        }
        if (n->op == GGML_OP_GATED_DELTA_NET && root_of(n->src[5]) == gr) {
            if (gdn != nullptr) {
                return false;
            }
            gdn = n;
            continue;
        }
        if (reads_gr) {
            if (ggml_cuda_is_view_or_noop(n)) {
                continue; // the reshape/view chain into the GDN (a view reads nothing)
            }
            return false;
        }
        // nothing may write the cache before the GDN reads it
        if (gdn == nullptr && !ggml_cuda_is_view_or_noop(n) && root_of(n) == cache_root) {
            return false;
        }
    }
    return gdn_gather_gdn_ok(gr, gdn);
}

void ggml_cuda_gdn_gather_index_build(const ggml_cgraph * cgraph, ggml_cuda_gdn_gather_index & idx) {
    idx.gathers.clear();
    idx.writers.clear();
    idx.built = true;
    for (int j = 0; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * cache_root = gdn_gather_candidate(cgraph->nodes[j]);
        if (cache_root != nullptr) {
            idx.gathers.emplace(cgraph->nodes[j], ggml_cuda_gdn_gather_index::gather_info{});
            idx.writers.emplace(cache_root, std::vector<int>{});
        }
    }
    if (idx.gathers.empty()) {
        return;
    }

    const auto root_of = gdn_gather_root_of;
    for (int j = 0; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_is_empty(n)) {
            continue;
        }
        const bool view_or_noop = ggml_cuda_is_view_or_noop(n);
        const ggml_tensor * gdn_root =
            n->op == GGML_OP_GATED_DELTA_NET && n->src[5] != nullptr ? root_of(n->src[5]) : nullptr;
        const ggml_tensor * concat_root = n->op == GGML_OP_CONCAT && n->src[0] != nullptr ? root_of(n->src[0]) : nullptr;

        // a non-view node reading a gather is a reader that blocks the deferral, unless it is that gather's GDN
        if (!view_or_noop) {
            const ggml_tensor * read[GGML_MAX_SRC + 1];
            int n_read = 0;
            if (n->view_src != nullptr) {
                read[n_read++] = root_of(n);
            }
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                if (n->src[s] != nullptr) {
                    read[n_read++] = root_of(n->src[s]);
                }
            }
            for (int k = 0; k < n_read; ++k) {
                if (read[k] == gdn_root) {
                    continue;
                }
                // read[0] is a CONCAT's src[0] (a CONCAT is no view): its state operand
                if (k == 0 && concat_root != nullptr && read[k] == concat_root && n->view_src == nullptr &&
                        (n->src[1] == nullptr || root_of(n->src[1]) != concat_root)) {
                    continue;   // the CONCAT's state operand: counted below, decided by ggml_cuda_gdn_conv_gather_deferrable
                }
                auto it = idx.gathers.find(read[k]);
                if (it != idx.gathers.end()) {
                    it->second.other_reader = true;
                }
            }
        }
        if (gdn_root != nullptr) {
            auto it = idx.gathers.find(gdn_root);
            if (it != idx.gathers.end()) {
                it->second.gdn = j;
                it->second.n_gdn++;
            }
        }
        if (concat_root != nullptr) {
            auto it = idx.gathers.find(concat_root);
            if (it != idx.gathers.end()) {
                it->second.concat = j;
                it->second.n_concat++;
            }
        }
        if (!view_or_noop) {
            auto it = idx.writers.find(root_of(n));
            if (it != idx.writers.end()) {
                it->second.push_back(j);
            }
        }
    }
}

// same answer as the per-gather walk above, read from the index
bool ggml_cuda_gdn_state_gather_deferrable(const ggml_cgraph * cgraph, int node_idx, const ggml_cuda_gdn_gather_index & idx) {
    const ggml_tensor * gr = cgraph->nodes[node_idx];
    const auto g = idx.gathers.find(gr);
    if (g == idx.gathers.end() || g->second.n_gdn != 1 || g->second.n_concat != 0 || g->second.other_reader) {
        return false;
    }
    const int gdn_idx = g->second.gdn;
    if (gdn_idx < node_idx) {
        return false;
    }

    // nothing may write the cache between the gather and the GDN
    const auto w = idx.writers.find(gdn_gather_root_of(gr->src[0]));
    if (w != idx.writers.end()) {
        const auto first_after = std::upper_bound(w->second.begin(), w->second.end(), node_idx);
        if (first_after != w->second.end() && *first_after < gdn_idx) {
            return false;
        }
    }
    return gdn_gather_gdn_ok(gr, cgraph->nodes[gdn_idx]);
}

// expert group selection from build_moe_ffn (n_expert_groups > 1)
// src0/src1 index into this table, -1 is the (biased) probs, -2 is not checked
static const struct {
    ggml_op op;
    int     src0;
    int     src1;
} topk_moe_group_pattern[] = {
    { GGML_OP_RESHAPE,  -1, -2 }, //  0: [n_exp_per_group, n_expert_groups, n_tokens]
    { GGML_OP_RESHAPE,   0, -2 }, //  1: [1, n_exp_per_group, n_expert_groups, n_tokens]
    { GGML_OP_ARGSORT,   0, -2 }, //  2
    { GGML_OP_VIEW,      2, -2 }, //  3: top-2 per group
    { GGML_OP_GET_ROWS,  1,  3 }, //  4
    { GGML_OP_RESHAPE,   4, -2 }, //  5
    { GGML_OP_SUM_ROWS,  5, -2 }, //  6: group scores
    { GGML_OP_RESHAPE,   6, -2 }, //  7
    { GGML_OP_ARGSORT,   7, -2 }, //  8
    { GGML_OP_VIEW,      8, -2 }, //  9: top n_group_used groups
    { GGML_OP_GET_ROWS,  0,  9 }, // 10
    { GGML_OP_FILL,      0, -2 }, // 11
    { GGML_OP_SET_ROWS, 10,  9 }, // 12: unused groups -> -inf
    { GGML_OP_RESHAPE,  12, -2 }, // 13: [n_expert, n_tokens]
};

// The conv-state gather (build_rs's GET_ROWS of the recurrent conv cache) feeding a GDN conv CONCAT that the direct conv
// takes with its snapshot fold: the direct kernel reads the cache row itself (row id = the gather's src[1]) and the
// GET_ROWS launches nothing.  It holds each channel's three state values in registers before it writes the snapshots,
// one of which is the write-back to that same row.  Only for a single sequence in the decode / verify band (<= 16
// tokens), when the CONCAT is the gather's only reader and nothing but the folded snapshot CPYs writes the cache before
// the conv.  GGML_CUDA_FUSE_GDN_CONV_GATHER=0: off.
bool ggml_cuda_gdn_conv_gather_deferrable(const ggml_cgraph * cgraph, int node_idx, const ggml_cuda_gdn_gather_index & idx) {
    static const bool fuse = getenv("GGML_CUDA_FUSE_GDN_CONV_GATHER") == nullptr || atoi(getenv("GGML_CUDA_FUSE_GDN_CONV_GATHER")) != 0;
    const ggml_tensor * gr = cgraph->nodes[node_idx];
    const auto g = idx.gathers.find(gr);
    if (!fuse || g == idx.gathers.end() || g->second.n_gdn != 0 || g->second.n_concat != 1 || g->second.other_reader ||
            g->second.concat < node_idx) {
        return false;
    }
    ggml_cuda_gdn_conv_match m;
    if (!ggml_cuda_gdn_conv_match_at_concat(cgraph, g->second.concat, m) || !m.snap || m.T > 16 ||
            gdn_gather_root_of(m.state) != gr || m.state->data != gr->data || ggml_nelements(gr) != 3 * m.C ||
            gr->src[1]->ne[0] != 1 || ggml_nrows(gr) != 1) {
        return false;
    }
    const auto w = idx.writers.find(gdn_gather_root_of(gr->src[0]));
    if (w != idx.writers.end()) {
        for (const int j : w->second) {
            if (j <= node_idx || j >= m.conv_idx) {
                continue;
            }
            bool snap = false;
            for (int a = 0; a < m.n_snap; ++a) {
                snap = snap || m.snap_idx[a] == j;
            }
            if (!snap) {
                return false;
            }
        }
    }
    return true;
}

static bool ggml_cuda_topk_moe_fusion_disabled() {
    // A/B kill-switch for the fused MoE router.  The fused kernel is bit-identical to the generic
    // softmax -> argsort -> get_rows -> norm chain (both the softmax reduction/normalization and
    // the argsort tie-break are matched), so this only exists to compare the two execution paths.
    static const bool disabled = getenv("GGML_CUDA_DISABLE_TOPK_MOE_FUSION") != nullptr &&
                                 std::atoi(getenv("GGML_CUDA_DISABLE_TOPK_MOE_FUSION")) != 0;
    return disabled;
}

static bool ggml_cuda_topk_moe_group_fusion(const struct ggml_cgraph * cgraph, int node_idx, const ggml_tensor * selection, ggml_cuda_topk_moe_args & args) {
#if defined(GGML_USE_HIP) && HIP_VERSION < 60300000
    GGML_UNUSED_VARS(cgraph, node_idx, selection, args);
    return false;
#else
    const int n_ops = (int) std::size(topk_moe_group_pattern);
    if (node_idx + n_ops > cgraph->n_nodes) {
        return false;
    }

    ggml_tensor ** nodes = cgraph->nodes + node_idx;
    auto src_ok = [&](const ggml_tensor * src, int idx) {
        return idx == -2 || src == (idx == -1 ? selection : nodes[idx]);
    };

    for (int i = 0; i < n_ops; i++) {
        const auto & p = topk_moe_group_pattern[i];
        if (nodes[i]->op != p.op || !src_ok(nodes[i]->src[0], p.src0) || !src_ok(nodes[i]->src[1], p.src1)) {
            return false;
        }
    }

    const int n_groups     = nodes[0]->ne[1];
    const int n_group_used = nodes[9]->ne[0];

    if (nodes[3]->ne[0] != 2 || nodes[12]->src[2] != nodes[11] || ggml_get_op_params_f32(nodes[11], 0) != -INFINITY ||
            n_groups > WARP_SIZE) {
        return false;
    }

    args.n_groups     = n_groups;
    args.n_group_used = n_group_used;
    return true;
#endif // defined(GGML_USE_HIP) && HIP_VERSION < 60300000
}

bool ggml_cuda_topk_moe_fusion(const struct ggml_cgraph * cgraph, int node_idx, ggml_cuda_topk_moe_args & args) {
    args.sigmoid         = false;
    args.sqrt_softplus   = false;
    args.softmax         = false;
    args.delayed_softmax = false;
    args.prob_bias       = false;
    args.norm            = false;
    args.n_groups        = 0;
    args.n_group_used    = 0;

    const int      n_nodes = cgraph->n_nodes;
    ggml_tensor ** nodes   = cgraph->nodes;

    if (nodes[node_idx]->op == GGML_OP_SOFT_MAX) {
        args.softmax = true;
    }

    if (nodes[node_idx]->op == GGML_OP_UNARY) {
        const ggml_unary_op unary_op = ggml_get_unary_op(nodes[node_idx]);
        if (unary_op == GGML_UNARY_OP_SIGMOID) {
            args.sigmoid = true;
        } else if (unary_op == GGML_UNARY_OP_SOFTPLUS && node_idx + 1 < n_nodes &&
                   nodes[node_idx + 1]->op == GGML_OP_SQRT && nodes[node_idx + 1]->src[0] == nodes[node_idx]) {
            // sqrt(softplus(x)) scoring (DeepSeek-V4)
            args.sqrt_softplus = true;
            node_idx++;
        } else {
            return false;
        }
    }

    if (nodes[node_idx]->op == GGML_OP_ARGSORT) {
        args.delayed_softmax = true;
    }

    node_idx++;

    if (args.sigmoid || args.sqrt_softplus || args.softmax) {
        // SOFTMAX -> RESHAPE
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_RESHAPE ||
                nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx];
        node_idx++;

        if (node_idx >= n_nodes) {
            return false;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        const ggml_tensor * selection = nodes[node_idx - 2];
        if (nodes[node_idx]->op == GGML_OP_ADD && nodes[node_idx]->src[0] == selection) {
            args.prob_bias = true;
            selection = nodes[node_idx];
            node_idx++;
        }

        if (ggml_cuda_topk_moe_group_fusion(cgraph, node_idx, selection, args)) {
            node_idx += (int) std::size(topk_moe_group_pattern);
            selection = nodes[node_idx - 1];
        }

        // RESHAPE/ADD/group mask -> ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_ARGSORT || nodes[node_idx]->src[0] != selection) {
            return false;
        }

        node_idx++;

        // ARGSORT-> VIEW
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
                nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }

        // kept groups must hold all used experts
        if (args.n_groups > 0 && nodes[node_idx]->ne[0] > args.n_group_used * (probs_reshaped->ne[1] / args.n_groups)) {
            return false;
        }
        node_idx++;

        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_GET_ROWS) {
            return false;
        }

        // GET_ROWS
        if (nodes[node_idx]->src[0] != probs_reshaped || nodes[node_idx]->src[1] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;
    } else if (args.delayed_softmax) {
        if (node_idx - 2 < 0) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx - 2];

        // VIEW->ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
            nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;

        // GET_ROWS
        if (node_idx >= n_nodes || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
                nodes[node_idx]->src[0] != probs_reshaped) {
            return false;
        }
        node_idx++;

        static const std::vector<ggml_op> remaining_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

        for (const ggml_op op : remaining_ops) {
            if (node_idx >= n_nodes || nodes[node_idx]->op != op || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
                return false;
            }
            node_idx++;
        }
    }

    // At this point we can check for norm + scale. Everything is now at least valid till the norm
    if (node_idx >= n_nodes) {
        return true;
    }

    if (nodes[node_idx]->op == GGML_OP_RESHAPE) {
        //check RESHAPE->SUM_ROWS->CLAMP->DIV->RESHAPE
        static const std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP };

        args.norm = true;
        for (const ggml_op op : norm_ops) {
            if (nodes[node_idx]->op == op && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
                node_idx++;
            } else {
                args.norm = false;
                return true;
            }
        }

        // DIV <- CLAMP, RESHAPE
        if (nodes[node_idx]->op != GGML_OP_DIV || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
            nodes[node_idx]->src[0] != nodes[node_idx - 3]) {
            args.norm = false;
            return true;
        }
        node_idx++;

        if (nodes[node_idx]->op != GGML_OP_RESHAPE || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            args.norm = false;
            return true;
        }

        node_idx++;
    }

    if (nodes[node_idx]->op == GGML_OP_SCALE && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
        args.scale = true;
    }

    return true;
}

// returns whether the write (out) nodes overwrite the read nodes in operation
bool ggml_cuda_check_fusion_memory_ranges(const ggml_cgraph * cgraph,
                                                 const int           node_idx,
                                                 const int           node_count,
                                                 const int *         out_nodes,
                                                 const int           out_count,
                                                 const bool          is_topk_moe) {
    auto nodes_overlap = [&](const ggml_tensor * a, const ggml_tensor * b) {
        const int64_t a_start = (int64_t) a->data;
        const int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);

        const int64_t b_start = (int64_t) b->data;
        const int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        if ((b_start <= a_start && a_start < b_end) || (a_start <= b_start && b_start < a_end)) {
            return true;
        }

        return false;
    };

    bool is_ok = true;
    // one block reads all logits before it writes, so logits may alias the out nodes
    const ggml_tensor * logits_may_alias = nullptr;
    if (is_topk_moe && ggml_nrows(cgraph->nodes[node_idx]) <= TOPK_MOE_ROWS_PER_BLOCK) {
        logits_may_alias = cgraph->nodes[node_idx]->src[0];
    }

    for (int i = 0; i < out_count; ++i) {
        const ggml_tensor * dst = cgraph->nodes[out_nodes[i]];

        for (int j = node_idx; j < node_idx + node_count; ++j) {
            // Loop over all srcs of all nodes in the fusion. If the src overlaps
            // the destination and the src is not an intermediate node that's being
            // elided, then disable fusion.

            for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                const ggml_tensor * src = cgraph->nodes[j]->src[src_idx];

                if (!src || src->op == GGML_OP_NONE || src == logits_may_alias) {
                    continue;
                }

                if (nodes_overlap(dst, src)) {
                    bool found = false;

                    for (int k = node_idx; k < j; ++k) {
                        if (cgraph->nodes[k] == src) {
                            found = true;
                            break;
                        }
                    }

                    if (!found) {
                        is_ok = false;
                        break;
                    }
                }
            }
        }
    }

    return is_ok;
}

// The long form spans 2*k + 1 nodes. ggml_can_fuse_subgraph() accepts at most
// 31 nodes, so k <= 15; larger values use the per-operation path.
static constexpr int MOE_WEIGHTED_REDUCTION_MAX_EXPERTS = 15;

struct ggml_cuda_moe_weighted_reduction_match {
    const ggml_tensor * experts      = nullptr;
    const ggml_tensor * expert_scale = nullptr;
    const ggml_tensor * weights      = nullptr;
    ggml_tensor *       dst          = nullptr;
    int                 node_count   = 0;
};

static bool ggml_cuda_match_moe_weighted_reduction(
        const ggml_cgraph * cgraph,
        int node_idx,
        ggml_cuda_moe_weighted_reduction_match & match) {
    const ggml_tensor * first = cgraph->nodes[node_idx];
    if (first->op != GGML_OP_MUL || first->type != GGML_TYPE_F32 || !ggml_is_contiguous(first)) {
        return false;
    }

    auto split_mul = [](const ggml_tensor * mul, const ggml_tensor *& full, const ggml_tensor *& broadcast) {
        auto is_weights = [mul](const ggml_tensor * tensor) {
            return tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) && tensor->ne[0] == 1 &&
                tensor->ne[1] == mul->ne[1] && tensor->ne[2] == mul->ne[2] && tensor->ne[3] == mul->ne[3];
        };
        auto is_experts = [mul](const ggml_tensor * tensor) {
            return tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) &&
                ggml_are_same_shape(tensor, mul);
        };

        if (is_experts(mul->src[0]) && is_weights(mul->src[1])) {
            full      = mul->src[0];
            broadcast = mul->src[1];
            return true;
        }
        if (is_experts(mul->src[1]) && is_weights(mul->src[0])) {
            full      = mul->src[1];
            broadcast = mul->src[0];
            return true;
        }
        return false;
    };

    const ggml_tensor * weighted     = first;
    const ggml_tensor * experts      = nullptr;
    const ggml_tensor * expert_scale = nullptr;
    const ggml_tensor * weights      = nullptr;
    int                 mul_count    = 1;

    // Match both structural forms:
    //   (experts * expert_scale) * router_weight
    //   experts * router_weight
    // The matcher does not depend on the model or quantization type.
    if (node_idx + 1 < cgraph->n_nodes) {
        const ggml_tensor * second = cgraph->nodes[node_idx + 1];
        const ggml_tensor * scaled = nullptr;
        const ggml_tensor * route  = nullptr;
        const ggml_tensor * raw    = nullptr;
        const ggml_tensor * scale  = nullptr;
        if (second->op == GGML_OP_MUL && second->type == GGML_TYPE_F32 && ggml_is_contiguous(second) &&
                split_mul(second, scaled, route) && scaled == first && split_mul(first, raw, scale)) {
            weighted     = second;
            experts      = raw;
            expert_scale = scale;
            weights      = route;
            mul_count    = 2;
        }
    }

    if (experts == nullptr && !split_mul(first, experts, weights)) {
        return false;
    }

    const int     n_expert_used = (int) weighted->ne[1];
    const int64_t n_tokens      = weighted->ne[2] * weighted->ne[3];
    if (n_expert_used < 2 || n_expert_used > MOE_WEIGHTED_REDUCTION_MAX_EXPERTS) {
        return false;
    }

    const int node_count = 2 * n_expert_used + mul_count - 1;
    if (node_idx + node_count > cgraph->n_nodes) {
        return false;
    }

    std::vector<ggml_op> ops(node_count, GGML_OP_VIEW);
    ops[0] = GGML_OP_MUL;
    if (mul_count == 2) {
        ops[1] = GGML_OP_MUL;
    }
    std::vector<const ggml_tensor *> views;
    views.reserve(n_expert_used);
    const ggml_tensor * previous = nullptr;
    int n_adds = 0;
    for (int offset = mul_count; offset < node_count; ++offset) {
        const ggml_tensor * candidate = cgraph->nodes[node_idx + offset];
        ops[offset] = candidate->op;

        if (candidate->op == GGML_OP_VIEW) {
            const int expert = (int) views.size();
            if (expert >= n_expert_used || candidate->src[0] != weighted || candidate->view_src != weighted ||
                    candidate->type != GGML_TYPE_F32 || candidate->ne[0] != weighted->ne[0] ||
                    candidate->ne[1] != n_tokens || candidate->ne[2] != 1 || candidate->ne[3] != 1 ||
                    candidate->nb[0] != weighted->nb[0] || candidate->nb[1] != weighted->nb[2] ||
                    candidate->view_offs != (size_t) expert * weighted->nb[1]) {
                return false;
            }
            views.push_back(candidate);
            continue;
        }

        if (candidate->op != GGML_OP_ADD || views.size() < 2 || n_adds + 1 >= (int) views.size()) {
            return false;
        }
        const ggml_tensor * lhs = n_adds == 0 ? views[0] : previous;
        const ggml_tensor * rhs = views[n_adds + 1];
        if (candidate->src[0] != lhs || candidate->src[1] != rhs || candidate->type != GGML_TYPE_F32) {
            return false;
        }
        previous = candidate;
        ++n_adds;
    }

    if ((int) views.size() != n_expert_used || n_adds != n_expert_used - 1 || previous == nullptr) {
        return false;
    }
    if (!ggml_is_contiguous(previous) || previous->ne[0] != weighted->ne[0] ||
            previous->ne[1] != n_tokens || previous->ne[2] != 1 || previous->ne[3] != 1) {
        return false;
    }

    const int output_idx = node_idx + node_count - 1;
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, node_count, ops.data(), &output_idx, 1)) {
        return false;
    }

    match.experts      = experts;
    match.expert_scale = expert_scale;
    match.weights      = weights;
    match.dst          = cgraph->nodes[output_idx];
    match.node_count   = node_count;
    return true;
}


// RMS_NORM + MUL + ROPE followed by the KV rotation (RESHAPE + Hadamard MUL_MAT, run as fwht_cuda): one launch
// that writes the rotated rows.  The FWHT is the same code as fwht_cuda, so the result is the same either way.
static bool ggml_cuda_should_fuse_rms_norm_mul_rope_fwht(const ggml_cgraph * cgraph, const int node_idx) {
    static const bool disabled = getenv("GGML_CUDA_DISABLE_ROPE_FWHT") != nullptr;
    if (disabled || node_idx + 5 > cgraph->n_nodes) {
        return false;
    }
    const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
    const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
    const ggml_tensor * rope     = cgraph->nodes[node_idx + 2];
    const ggml_tensor * reshape  = cgraph->nodes[node_idx + 3];
    const ggml_tensor * mm       = cgraph->nodes[node_idx + 4];

    // ggml_can_fuse for the RMS_NORM .. ROPE part; the RESHAPE is a view of the ROPE, so its use count is
    // checked directly (ggml_node_has_n_uses rejects every view)
    if (!ggml_can_fuse(cgraph, node_idx, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE }) ||
        !ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope) ||
        !ggml_node_has_n_uses(cgraph, node_idx + 2, 1)) {
        return false;
    }
    if (reshape->op != GGML_OP_RESHAPE || reshape->src[0] != rope ||
        ggml_node_get_use_count(cgraph, node_idx + 3) != 1 || (reshape->flags & GGML_TENSOR_FLAG_OUTPUT) ||
        mm->op != GGML_OP_MUL_MAT || mm->src[1] != reshape || !ggml_cuda_op_mul_mat_use_fwht(mm) ||
        (mm->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        return false;
    }
    // one rotation block per head
    const int64_t n = mm->src[0]->ne[0];
    if (rope->ne[0] != n || (n != 64 && n != 128 && n != 256) || !ggml_is_contiguous(rope)) {
        return false;
    }

    // As ggml_cuda_check_fusion_memory_ranges, except that dst may alias the norm input x exactly (the
    // allocator hands the dead Q/K projection to the rotation output): each block reads its whole row of x
    // before a barrier and only then writes the same row of dst.
    const auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
        const int64_t a0 = (int64_t) a->data;
        const int64_t a1 = a0 + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);
        const int64_t b0 = (int64_t) b->data;
        const int64_t b1 = b0 + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);
        return a0 < b1 && b0 < a1;
    };
    const ggml_tensor * x = rms_norm->src[0];
    for (int j = node_idx; j < node_idx + 5; ++j) {
        for (int k = 0; k < GGML_MAX_SRC; ++k) {
            const ggml_tensor * src = cgraph->nodes[j]->src[k];
            if (!src || src->op == GGML_OP_NONE || !overlap(mm, src)) {
                continue;
            }
            if (j > node_idx && src == cgraph->nodes[j - 1]) {
                continue;   // intermediate node, not read by the fused kernel
            }
            // x has rope's shape and both are contiguous, as is mm with the same element count: same linear layout
            const bool exact_alias = src == x && x->data == mm->data && ggml_is_contiguous(x);
            if (!exact_alias) {
                return false;
            }
        }
    }
    return true;
}

static bool ggml_cuda_can_fuse(const struct ggml_cgraph * cgraph, int node_idx, std::initializer_list<enum ggml_op> ops,
                               std::initializer_list<enum ggml_unary_op> unary_ops);

// Q / K norm+rope pairing: the Q chain RMS_NORM, MUL, ROPE at i (plain F32 output) and the K chain RMS_NORM, MUL, ROPE, VIEW,
// SET_ROWS of the same layer at j (a few nodes later, after the K / V matmuls) with the same rope parameters and positions run
// as one launch at j.  Returns j, or 0.  The deferred Q kernel reads the Q chain's inputs at j, so nothing in between (and no
// cache row the K chain writes) may write over them or over the Q output, and nothing in between may read the Q chain.
// GGML_CUDA_ROPE_QK_PAIR=0: off.
static int ggml_cuda_qk_pair_k_at(const ggml_cgraph * cgraph, const int i, const bool allow) {
    static const bool on = getenv("GGML_CUDA_ROPE_QK_PAIR") == nullptr || atoi(getenv("GGML_CUDA_ROPE_QK_PAIR")) != 0;
    if (!on || !allow) {
        return 0;
    }
    const ggml_tensor * q_norm = cgraph->nodes[i];
    const ggml_tensor * q_mul  = cgraph->nodes[i + 1];
    const ggml_tensor * q_rope = cgraph->nodes[i + 2];
    if (q_rope->type != GGML_TYPE_F32 || q_norm->src[0]->ne[0] >= 1024 || q_rope->src[1] == nullptr) {
        return 0;
    }
    int j = -1;
    for (int n = i + 3; n < std::min(cgraph->n_nodes, i + 16); ++n) {
        if (cgraph->nodes[n]->op == GGML_OP_RMS_NORM) {
            j = n;
            break;
        }
    }
    if (j < 0 || j + 5 > cgraph->n_nodes ||
            !ggml_cuda_can_fuse(cgraph, j, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
        return 0;
    }
    const ggml_tensor * k_norm = cgraph->nodes[j];
    const ggml_tensor * k_rope = cgraph->nodes[j + 2];
    const ggml_tensor * k_sr   = cgraph->nodes[j + 4];
    if ((k_sr->type != GGML_TYPE_F16 && k_sr->type != GGML_TYPE_F32) || k_norm->src[0]->ne[0] >= 1024 ||
            memcmp(q_rope->op_params, k_rope->op_params, sizeof(q_rope->op_params)) != 0 ||
            q_rope->src[1] != k_rope->src[1] || q_rope->src[2] != k_rope->src[2] ||
            q_norm->src[0]->ne[2] != k_norm->src[0]->ne[2] || q_norm->src[0]->ne[3] != k_norm->src[0]->ne[3]) {
        return 0;
    }
    const ggml_tensor * q_mul_src = q_mul->src[0] == q_norm ? q_mul->src[1] : q_mul->src[0];
    const auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
        const int64_t a0 = (int64_t) a->data, a1 = a0 + ggml_nbytes(a);
        const int64_t b0 = (int64_t) b->data, b1 = b0 + ggml_nbytes(b);
        return a0 < b1 && b0 < a1;
    };
    const ggml_tensor * q_live[] = { q_norm->src[0], q_mul_src, q_rope->src[1], q_rope->src[2], q_rope };
    const int last = std::min(cgraph->n_nodes - 1, j + 6);
    for (int n = i + 3; n <= last; ++n) {
        const ggml_tensor * t = cgraph->nodes[n];
        // of the K chain the pair kernel writes only the SET_ROWS destinations (j + 4, and j + 6 with the V row)
        const bool written = n < j || n == j + 4 || n == j + 6;
        for (const ggml_tensor * l : q_live) {
            if (written && l != nullptr && t->data != nullptr && overlap(t, l)) {
                return 0;
            }
        }
        if (n < j) {
            for (int s = 0; s < GGML_MAX_SRC && t->src[s]; ++s) {
                const ggml_tensor * r = t->src[s];
                while (r->view_src != nullptr) {
                    r = r->view_src;
                }
                if (r == q_norm || r == q_mul || r == q_rope) {
                    return 0;
                }
            }
        }
    }
    return j;
}

// The plain (F32 / F16 cache) K path RMS_NORM, MUL, ROPE, VIEW, SET_ROWS at i, followed at i + 5 / i + 6 by the V row's
// VIEW and SET_ROWS into a cache of the same type with the same row length: the fused kernel's per-head blocks also write
// the V row (k_set_rows' cast), so the V set_rows launch goes away.  GGML_CUDA_ROPE_SET_ROWS_V=0: off.
static bool ggml_cuda_rope_set_rows_v_ok(const ggml_cgraph * cgraph, const int i) {
    static const bool on = getenv("GGML_CUDA_ROPE_SET_ROWS_V") == nullptr || atoi(getenv("GGML_CUDA_ROPE_SET_ROWS_V")) != 0;
    if (!on || i + 7 > cgraph->n_nodes) {
        return false;
    }
    const ggml_tensor * rope   = cgraph->nodes[i + 2];
    const ggml_tensor * k_sr   = cgraph->nodes[i + 4];
    const ggml_tensor * v_view = cgraph->nodes[i + 5];
    const ggml_tensor * v_sr   = cgraph->nodes[i + 6];
    const int64_t row_len = rope->ne[0]*rope->ne[1];
    const int64_t n_tok   = rope->ne[2];
    if (rope->ne[3] != 1 || k_sr->op != GGML_OP_SET_ROWS || (k_sr->type != GGML_TYPE_F16 && k_sr->type != GGML_TYPE_F32) ||
            v_view->op != GGML_OP_VIEW || v_view->type != GGML_TYPE_F32 || !ggml_is_contiguous(v_view) ||
            v_view->ne[0] != row_len || v_view->ne[1] != n_tok || v_view->ne[2] != 1 || v_view->ne[3] != 1 ||
            v_sr->op != GGML_OP_SET_ROWS || v_sr->src[0] != v_view || v_sr->type != k_sr->type ||
            v_sr->ne[0] != row_len || v_sr->ne[2] != 1 || v_sr->ne[3] != 1 ||
            ggml_node_get_use_count(cgraph, i + 5) != 1 || (v_view->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
        return false;
    }
    const ggml_tensor * idx = v_sr->src[1];
    if (idx->type != GGML_TYPE_I64 || !ggml_is_contiguous(idx) || ggml_nelements(idx) != n_tok) {
        return false;
    }
    // the V write must not touch anything the fused kernel reads or the K cache rows it writes
    const auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
        const int64_t a0 = (int64_t) a->data, a1 = a0 + ggml_nbytes(a);
        const int64_t b0 = (int64_t) b->data, b1 = b0 + ggml_nbytes(b);
        return a0 < b1 && b0 < a1;
    };
    const ggml_tensor * rms_norm = cgraph->nodes[i];
    const ggml_tensor * mul      = cgraph->nodes[i + 1];
    const ggml_tensor * mul_src  = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];
    for (const ggml_tensor * t : std::initializer_list<const ggml_tensor *>{ rms_norm->src[0], mul_src, rope->src[1], k_sr, k_sr->src[1], v_view }) {
        if (t != nullptr && overlap(v_sr, t)) {
            return false;
        }
    }
    return true;
}

// The K path's rms_norm_mul_rope + FWHT (above) followed by its KV-cache write, RESHAPE, VIEW, SET_ROWS (q8_0), and
// usually by the V write, VIEW, SET_ROWS (q8_0) of the V row rotated earlier: the fused kernel writes both cache rows
// with k_set_rows_q8_0_group's quantization, so the two set_rows launches go away.  Returns the number of fused nodes
// (10 with the V write, 8 without, 0 if the pattern doesn't match).  GGML_CUDA_DISABLE_ROPE_FWHT_SET_ROWS turns it off.
// allow_v: false inside a concurrent-stream region (GGML_CUDA_GRAPH_OPT=1), where the V row comes from another stream.
static int ggml_cuda_rms_norm_mul_rope_fwht_set_rows_count(const ggml_cgraph * cgraph, const int i, const bool allow_v) {
    static const bool disabled = getenv("GGML_CUDA_DISABLE_ROPE_FWHT_SET_ROWS") != nullptr;
    if (disabled || i + 8 > cgraph->n_nodes || !ggml_cuda_should_fuse_rms_norm_mul_rope_fwht(cgraph, i)) {
        return 0;
    }
    const ggml_tensor * rms_norm = cgraph->nodes[i];
    const ggml_tensor * mul      = cgraph->nodes[i + 1];
    const ggml_tensor * rope     = cgraph->nodes[i + 2];
    const ggml_tensor * mm       = cgraph->nodes[i + 4];
    const ggml_tensor * reshape  = cgraph->nodes[i + 5];
    const ggml_tensor * k_view   = cgraph->nodes[i + 6];
    const ggml_tensor * k_sr     = cgraph->nodes[i + 7];

    // one contiguous F32 row of nrows*ncols per token written as q8_0 cache rows through I64 indices
    const int64_t row_len = rope->ne[0]*rope->ne[1];
    const int64_t n_tok   = rope->ne[2];
    const auto is_row_write = [&](const ggml_tensor * view, const ggml_tensor * sr) {
        const ggml_tensor * idx = sr->src[1];
        return view->op == GGML_OP_VIEW && view->type == GGML_TYPE_F32 && ggml_is_contiguous(view) &&
               view->ne[0] == row_len && view->ne[1] == n_tok && view->ne[2] == 1 && view->ne[3] == 1 &&
               sr->op == GGML_OP_SET_ROWS && sr->src[0] == view && sr->type == GGML_TYPE_Q8_0 &&
               sr->ne[0] == row_len && sr->ne[2] == 1 && sr->ne[3] == 1 &&
               idx->type == GGML_TYPE_I64 && ggml_is_contiguous(idx) && idx->ne[0] == n_tok && ggml_nelements(idx) == n_tok;
    };
    const auto single_use = [&](const int j) {
        return ggml_node_get_use_count(cgraph, j) == 1 && (cgraph->nodes[j]->flags & GGML_TENSOR_FLAG_OUTPUT) == 0;
    };
    if (rope->ne[3] != 1 || row_len % QK8_0 != 0 ||
        reshape->op != GGML_OP_RESHAPE || reshape->src[0] != mm || k_view->src[0] != reshape ||
        !single_use(i + 4) || !single_use(i + 5) || !single_use(i + 6) || !is_row_write(k_view, k_sr)) {
        return 0;
    }

    bool with_v = false;
    if (allow_v && i + 10 <= cgraph->n_nodes) {
        const ggml_tensor * v_view = cgraph->nodes[i + 8];
        const ggml_tensor * v_sr   = cgraph->nodes[i + 9];
        with_v = is_row_write(v_view, v_sr) && single_use(i + 8);
    }

    // the cache writes must not touch anything the fused kernel reads, nor each other
    const auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
        const int64_t a0 = (int64_t) a->data;
        const int64_t a1 = a0 + ggml_nbytes(a);
        const int64_t b0 = (int64_t) b->data;
        const int64_t b1 = b0 + ggml_nbytes(b);
        return a0 < b1 && b0 < a1;
    };
    std::vector<const ggml_tensor *> reads = { rms_norm->src[0], mul->src[0] == rms_norm ? mul->src[1] : mul->src[0],
                                               rope->src[1], k_sr->src[1] };
    if (rope->src[2] != nullptr) {
        reads.push_back(rope->src[2]);
    }
    std::vector<const ggml_tensor *> writes = { k_sr };
    if (with_v) {
        const ggml_tensor * v_sr = cgraph->nodes[i + 9];
        reads.push_back(v_sr->src[0]);
        reads.push_back(v_sr->src[1]);
        writes.push_back(v_sr);
    }
    for (const ggml_tensor * w : writes) {
        for (const ggml_tensor * r : reads) {
            if (overlap(w, r)) {
                return 0;
            }
        }
    }
    if (with_v && overlap(writes[0], writes[1])) {
        with_v = false;
    }
    return with_v ? 10 : 8;
}

// Hadamard MUL_MAT (run as fwht_cuda) feeding an mmvq matmul through a RESHAPE (the attention-output rotation in
// front of the output projection): the FWHT kernel also writes the matmul's Q8_1 activation into the quantize
// cache, so the matmul skips its quantize_q8_1 launch.  The F32 output is still written.  Returns 1 (the RESHAPE
// is a no-op) when fused.  GGML_CUDA_DISABLE_FWHT_Q8_1 turns it off.
static int ggml_cuda_fwht_q8_1_try_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int i) {
    static const bool disabled = getenv("GGML_CUDA_DISABLE_FWHT_Q8_1") != nullptr;
    if (disabled || i + 2 >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor *       had     = cgraph->nodes[i];
    const ggml_tensor * reshape = cgraph->nodes[i + 1];
    const ggml_tensor * mm      = cgraph->nodes[i + 2];
    if (had->op != GGML_OP_MUL_MAT || !ggml_cuda_op_mul_mat_use_fwht(had) || had->src[1]->type != GGML_TYPE_F32 ||
        reshape->op != GGML_OP_RESHAPE || reshape->src[0] != had ||
        mm->op != GGML_OP_MUL_MAT || mm->src[1] != reshape || !ggml_is_quantized(mm->src[0]->type) ||
        !ggml_cuda_should_fuse_mul_mat_vec_q(mm, true)) {
        return 0;
    }
    // the FWHT rows (one warp each) tile the matmul's unpadded, contiguous activation rows
    const int64_t n    = had->ne[0];
    const int64_t ne10 = reshape->ne[0];
    if ((n != 64 && n != 128 && n != 256 && n != 512) || ne10 % n != 0 || ne10 % MATRIX_ROW_PADDING != 0 ||
        !ggml_is_contiguous(reshape)) {
        return 0;
    }

    // the mmvq launcher's key and layout: the view root of src1 (here the Hadamard output) and src1's shape
    const size_t  ts          = ggml_type_size(reshape->type);
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const size_t  q8_1_size   = reshape->ne[3]*reshape->ne[2] * reshape->ne[1]*ne10_padded * sizeof(block_q8_1)/QK8_1;
    bool cached = false;
    void * y = ctx.q8_1_cache_get(had, ctx.curr_stream_no, q8_1_size,
                                  ne10, reshape->ne[1], reshape->ne[2], reshape->ne[3],
                                  reshape->nb[1]/ts, reshape->nb[2]/ts, reshape->nb[3]/ts, cached);
    if (cached) {
        return 0;
    }
    // the cache entry now exists, so it must be filled
    GGML_ASSERT(ggml_cuda_op_fwht(ctx, had->src[1], had, y));
    return 1;
}

static bool ggml_cuda_can_fuse(const struct ggml_cgraph *                cgraph,
                               int                                       node_idx,
                               std::initializer_list<enum ggml_op>       ops,
                               std::initializer_list<enum ggml_unary_op> unary_ops) {
#ifndef NDEBUG
    const size_t num_unary = std::count(ops.begin(), ops.end(), GGML_OP_UNARY);
    GGML_ASSERT(unary_ops.size() == num_unary);
#endif

    const auto is_equal = [](const std::initializer_list<enum ggml_op> & list1,
                             const std::initializer_list<enum ggml_op> & list2) {
        return std::equal(list1.begin(), list1.end(), list2.begin(), list2.end());
    };

    std::initializer_list<enum ggml_op> mul_mat_bias_glu_ops    = { GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_GLU };
    std::initializer_list<enum ggml_op> mul_mat_id_bias_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_GLU };

    std::initializer_list<enum ggml_op> mul_mat_id_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU };
    std::initializer_list<enum ggml_op> mul_mat_glu_ops    = { GGML_OP_MUL_MAT,    GGML_OP_MUL_MAT,    GGML_OP_GLU };

    if ((is_equal(mul_mat_bias_glu_ops, ops) || is_equal(mul_mat_id_bias_glu_ops, ops)) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
        const ggml_tensor * ffn_gate      = cgraph->nodes[node_idx];
        const ggml_tensor * ffn_gate_bias = cgraph->nodes[node_idx + 1];
        const ggml_tensor * ffn_up        = cgraph->nodes[node_idx + 2];
        const ggml_tensor * ffn_up_bias   = cgraph->nodes[node_idx + 3];
        const ggml_tensor * glu           = cgraph->nodes[node_idx + 4];

        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu, ffn_up_bias, ffn_gate_bias)) {
            int out_nodes[] = { node_idx + 4 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if ((is_equal(mul_mat_id_glu_ops, ops) || is_equal(mul_mat_glu_ops, ops)) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * ffn_gate = cgraph->nodes[node_idx];
        const ggml_tensor * ffn_up   = cgraph->nodes[node_idx + 1];
        const ggml_tensor * glu      = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    std::initializer_list<enum ggml_op> rms_norm_mul_rope_ops          = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE };
    std::initializer_list<enum ggml_op> rms_norm_mul_rope_set_rows_ops = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    if (is_equal(rms_norm_mul_rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * rope     = cgraph->nodes[node_idx + 2];
        const ggml_tensor * view     = cgraph->nodes[node_idx + 3];
        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 4];

        // ggml_cuda_op_rms_norm_mul_rope_fused writes F32 / F16 rows only; a BF16 cache keeps rope's own set_rows path
        if (ggml_check_edges(cgraph, node_idx, {{1, 0, 0}, {2, 0, 1}, {3, 0, 2}, {4, 0, 3}}) &&
            set_rows->type != GGML_TYPE_BF16 &&
            ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope) &&
            ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
            int out_nodes[] = { node_idx + 4 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if (is_equal(rms_norm_mul_rope_ops, ops) && ggml_can_fuse(cgraph, node_idx, ops)) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * rope     = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
        return false;
    }

    std::initializer_list<enum ggml_op> rope_set_rows_ops = { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    if (is_equal(rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * rope     = cgraph->nodes[node_idx];
        const ggml_tensor * view     = cgraph->nodes[node_idx + 1];
        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if (!ggml_can_fuse(cgraph, node_idx, ops)) {
        return false;
    }

    if ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        const ggml_tensor *rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor *mul      = cgraph->nodes[node_idx+1];
        const ggml_tensor *add      = nullptr;

        if (ops.size() == 3 && ops.begin()[2] == GGML_OP_ADD) {
            add = cgraph->nodes[node_idx+2];
        }

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        //rms norm only supports F32
        if (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            return false;
        }

        if (add && (add->src[0]->type != GGML_TYPE_F32 ||
            add->src[1]->type != GGML_TYPE_F32 ||
            add->type != GGML_TYPE_F32) ) {
            return false;
        }

        //if rms norm is the B operand, then we don't handle broadcast
        if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
            return false;
        }

        //rms_norm kernel assumes contiguous rows
        if (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
            return false;
        }

        if (add && (!ggml_is_contiguous(add->src[0]) || !ggml_is_contiguous_rows(add->src[1]))) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_SCALE) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * scale    = cgraph->nodes[node_idx+1];

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        float bias;
        memcpy(&bias, (const float *) scale->op_params + 1, sizeof(float));

        return bias == 0.0f && scale->type == GGML_TYPE_F32;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_UNARY
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * silu     = cgraph->nodes[node_idx+1];
        if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            return false;
        }

        if (ssm_conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            return false;
        }

        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_ADD
     && ops.begin()[2] == GGML_OP_UNARY && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * add      = cgraph->nodes[node_idx+1];
        const ggml_tensor * silu     = cgraph->nodes[node_idx+2];
        if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            return false;
        }

        if (ssm_conv->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            return false;
        }

        // ADD must consume ssm_conv's output and broadcast a 1-D channel-wise bias.
        const ggml_tensor * bias = (add->src[0] == ssm_conv) ? add->src[1] : add->src[0];
        if (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias)) {
            return false;
        }
        if (ggml_nelements(bias) != ssm_conv->ne[0] || bias->ne[0] != ssm_conv->ne[0]) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL
     && unary_ops.size() == 1 && (unary_ops.begin()[0] == GGML_UNARY_OP_SILU || unary_ops.begin()[0] == GGML_UNARY_OP_SIGMOID || unary_ops.begin()[0] == GGML_UNARY_OP_SOFTPLUS)) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * mul   = cgraph->nodes[node_idx+1];

        if (ggml_get_unary_op(unary) != unary_ops.begin()[0]) {
            return false;
        }

        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16 && unary->type != GGML_TYPE_BF16) {
            return false;
        }

        if (unary->type != mul->type) {
            return false;
        }

        const ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
        if (other->type != unary->type) {
            return false;
        }
        if (!ggml_is_contiguous_1(other) || !ggml_is_contiguous_1(unary->src[0]) || !ggml_are_same_shape(other, unary)) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_SQR
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_RELU) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * sqr   = cgraph->nodes[node_idx+1];

        if (ggml_get_unary_op(unary) != GGML_UNARY_OP_RELU) {
            return false;
        }

        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
            return false;
        }

        if (unary->type != sqr->type) {
            return false;
        }

        if (!ggml_is_contiguous(unary->src[0])) {
            return false;
        }

        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SCALE && ops.begin()[1] == GGML_OP_UNARY && ops.begin()[2] == GGML_OP_SCALE
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_TANH) {
        const ggml_tensor *scale  = cgraph->nodes[node_idx];
        const ggml_tensor *tanh   = cgraph->nodes[node_idx+1];
        const ggml_tensor *scale2 = cgraph->nodes[node_idx+2];

        // the fused softcap kernel is F32 only
        if (scale->src[0]->type != GGML_TYPE_F32 || scale->type != GGML_TYPE_F32) {
            return false;
        }

        if (ggml_get_unary_op(tanh) != GGML_UNARY_OP_TANH) {
            return false;
        }

        // Check for bias
        if (ggml_get_op_params_f32(scale, 1) != 0.0f || ggml_get_op_params_f32(scale2, 1) != 0.0f) {
            return false;
        }

        return true;
    }

    return false;
}

// Structural identification of the qwen4exp hyper-connection combine+norm subgraph for the BF16 HC
// streams (LLAMA_HC_BLK16 / LLAMA_HC_RES16, both default OFF).  graph_optimize runs before the
// buffers are assigned, so this is structure/shape only (no alias checks); the fusion site
// re-checks everything.  Only the repeat-anchored form qwen4exp emits is recognised (the narrow
// block_out base is a graph output and the REPEAT is consumed inside the fused window); if it does
// not match, the caller simply does not mark and the two flags stay inert.

// Prefill indexer head reduction: relu + head-sum.  Anchored at the RELU.  Our qwen4exp graph (the
// L2a memory win) puts the relu BEFORE the 4-D reshape, so an optional RESHAPE between the relu and
// the head views is accepted; the reference's relu-on-4-D form matches with no reshape.  The fused
// kernel recomputes the relu from the pre-relu scores and sums in graph order, so it is bit-identical.
// wip/moe-expert-cache (H1): while the cache is active, only the fusions that read a routed
// expert table in the cache band stand down; every other backend fusion (router/topk, GDN, QSA,
// rope, norms, and the *prefill* MoE fusions) stays on.  The decode/verify MoE must run through
// `ggml_cuda_mul_mat_id`, which is where the slot-remap consumer lives; a fused MoE bypasses it
// and would read the redirected `input_cpy`, which the scheduler did not populate (the cache took
// the input over).  Prefill (n_tokens > band) is untouched: the cache does not take those inputs
// over, their `input_cpy` is fully copied, and their fusions keep firing bit-identically.
static void moe_fuse_trace(int ne2, bool blocked, const char * why) {
    static const bool on = getenv("GGML_CUDA_FUSE_TRACE") != nullptr;
    if (!on) {
        return;
    }
    static std::map<std::string, int> seen;
    const std::string k = std::to_string(ne2) + "/" + (blocked ? "block" : "allow") + "/" + why;
    if (seen.emplace(k, 1).second) {
        GGML_LOG_WARN("FUSE_TRACE ne2=%d %s (%s)\n", ne2, blocked ? "BLOCK" : "allow", why);
    }
}

static bool ggml_cuda_cache_blocks_fusion(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, int i) {
    if (!moe_cache_enabled()) {
        return false;
    }
    const ggml_tensor * node = cgraph->nodes[i];

    // gate+up+GLU, the routed pair, and the qwen4exp weighted-down all start at the routed
    // matmul; `ne[2]` is its token count, so gate on the cache band (prefill stays fused).
    if (node->op == GGML_OP_MUL_MAT_ID) {
        if (node->ne[2] > MOE_EXPERT_CACHE_MAX_TOK) {
            moe_fuse_trace((int) node->ne[2], false, "prefill");
            return false;   // prefill: the cache does not take these inputs over
        }
        // No routed expert table at all (the model is fully device-resident: `-ncmoe 0`, or a `-ncmoe`
        // that did not offload, e.g. on unified memory).  The cache can never take an input over, so its
        // fusions must behave exactly as if it were disabled -- otherwise enabling the cache changes the
        // output vs the cache-less oracle, because this stand-down changes the arithmetic.
        if (!moe_cache_has_tables()) {
            moe_fuse_trace((int) node->ne[2], false, "no-tables");
            return false;
        }
        // Tables exist but cannot serve (priming pass, every arena allocation failed, or a budget below
        // one expert): the cache takes no input over, so its cache-aware fusions MUST stay stood down.
        // The gate+up mmvq fusion reads the scheduler's copy through the fused call site, which is only
        // staged when the scheduler performs the copy itself; forcing the per-op path here is what keeps
        // the output byte-identical to the cache-less run.
        //
        // wip/moe-verify-fusions (OPEN 2 / #50): this used to be the GLOBAL `moe_cache_has_arena()`,
        // which requires EVERY table.  The movable-boundary slab evicts tables PER TABLE, so one
        // evicted table flipped the cache-band arithmetic of the whole graph -- and because the plain
        // and MTP arms evict at different points, the greedy streams diverged at a near-tie.  Ask THIS
        // table instead, under the exact `identity` condition the scheduler take-over uses,
        // so the fusion guard and the take-over agree per table (the code's own OPEN 2 invariant).
        const int device = cuda_ctx->device;
        // gate+up+GLU: one fused kernel reads both lanes, and it seeds the gate lane from the up
        // table's remap, so BOTH tables must serve or the pair stands down together.
        if (i + 2 < cgraph->n_nodes &&
                cgraph->nodes[i + 1]->op == GGML_OP_MUL_MAT_ID &&
                cgraph->nodes[i + 2]->op == GGML_OP_GLU) {
            const bool up_ok   = moe_cache_table_serves(node, node->src[0], device);
            const bool gate_ok = moe_cache_table_serves(cgraph->nodes[i + 1],
                                                        cgraph->nodes[i + 1]->src[0], device);
            moe_fuse_trace((int) node->ne[2], !(up_ok && gate_ok), up_ok && gate_ok ? "gate+up+glu" : "gate+up+glu-unserveable");
            return !(up_ok && gate_ok);   // allowed: the plain gate+up+GLU decode fusion
        }
        // down projection + per-(expert,token) routing-weight fold ([MUL_MAT_ID, MUL]): the
        // cache-aware variant in the epilogue below redirects the down table and the routing onto
        // the arena/remap, so it may fire.  The qwen4exp weighted-down chain (a 21-node tail,
        // opt-in via GGML_CUDA_ENABLE_RDNA3_5_SINGLE_TOKEN_FUSIONS and RDNA3_5-only) is NOT
        // redirect-safe; it is kept stood down by its own `!moe_cache_enabled()` gate at its site.
        if (i + 1 < cgraph->n_nodes && cgraph->nodes[i + 1]->op == GGML_OP_MUL &&
                cgraph->nodes[i + 1]->src[0] == node) {
            const bool ok = moe_cache_table_serves(node, node->src[0], device);
            // The down-fold is redirect-safe by construction: its call site (`moe_cache_redirect_fused`)
            // sends a served table to the arena and falls back to the raw, scheduler-copied table when
            // the cache does not serve, and the raw fallback is bit-identical to the per-op chain.  Do
            // NOT stand it down while the arena is still cold: that selected raw MUL_MAT_ID + MWR at
            // W=1 and the cache-aware down-fold at W>=2, which are not bit-identical (the qwen35moe
            // (35B-A3B) `-sm tensor -ncmoe` width impurity, TODO #53).
            moe_fuse_trace((int) node->ne[2], false, ok ? "down-fold" : "down-fold-unserveable-fallback");
            return false;   // allowed unconditionally: the call site redirects or falls back to the raw table
        }
        moe_fuse_trace((int) node->ne[2], true, "other");
        return true;
    }

    // swiglu -> routed-down fold (leading GLU, consuming the next MUL_MAT_ID).  Prefill-only in
    // practice, but guard it too so a cache-band variant can never read the redirected table.
    if (node->op == GGML_OP_GLU && i + 1 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_MUL_MAT_ID &&
            cgraph->nodes[i + 1]->ne[2] <= MOE_EXPERT_CACHE_MAX_TOK) {
        return true;
    }

    return false;
}

static int ggml_cuda_count_uses(const ggml_cgraph * cgraph, const ggml_tensor * t) {
    int uses = 0;
    for (int j = 0; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (n->src[s] == t || (n->src[s] && n->src[s]->view_src == t)) {
                uses++;
            }
        }
    }
    return uses;
}

// CONT(view) -> UNARY -> MUL -> decode matmul (the qwen4exp attention output gate: the gate half of each Q head):
// the gate -> Q8_1 kernel reads the row-strided view itself, so the CONT launches nothing.  The CONT is an exact
// copy, so the result is bit-identical.  GGML_CUDA_FUSE_CONT_GATE_Q8_1=0: off.  Returns the extra nodes consumed.
static int ggml_cuda_cont_gate_q8_1_try_fuse(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, int i) {
    static const bool fuse = getenv("GGML_CUDA_FUSE_CONT_GATE_Q8_1") == nullptr || atoi(getenv("GGML_CUDA_FUSE_CONT_GATE_Q8_1")) != 0;
    const ggml_tensor * c = cgraph->nodes[i];
    if (!fuse || c->op != GGML_OP_CONT || c->type != GGML_TYPE_F32 || i + 2 >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * un  = cgraph->nodes[i + 1];
    ggml_tensor * mul = cgraph->nodes[i + 2];
    const ggml_tensor * v = c->src[0];
    if (un->op != GGML_OP_UNARY || un->src[0] != c || ggml_get_unary_op(un) != GGML_UNARY_OP_SIGMOID ||
            !v || v->type != GGML_TYPE_F32 || v->nb[0] != sizeof(float) || !ggml_is_contiguous_1(v) ||
            ggml_nelements(v) != ggml_nelements(c) || !ggml_is_contiguous(c) ||
            !ggml_cuda_can_fuse(cgraph, i + 1, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SIGMOID })) {
        return 0;
    }
    const ggml_tensor * other = mul->src[0] == un ? mul->src[1] : mul->src[0];
    // a skipped GDN norm is computed only at its own gate node (ggml_cuda_gdn_norm_gate_try_fuse)
    if (!other || other == cuda_ctx->gdn_norm_pending || other->type != GGML_TYPE_F32 || !ggml_is_contiguous(other) ||
            ggml_nelements(other) != ggml_nelements(c) ||
            !ggml_are_same_shape(other, c) || ggml_cuda_count_uses(cgraph, c) != 1) {
        return 0;
    }
    const ggml_tensor * mm = ggml_cuda_find_mul_q8_1_matmul(cgraph, i + 2, mul);
    if (mm == nullptr) {
        return 0;
    }
    ggml_cuda_op_unary_mul_q8_1(*cuda_ctx, un, mul, mm, v);
    return cgraph->nodes[i + 3] == mm ? 2 : 3;
}

// GDN output: RMS_NORM(xn) -> MUL(w) at r, r+1, and later UNARY(sigmoid)(z) -> MUL(norm, .) -> decode matmul at s, s+1.
// Returns r when the gate -> Q8_1 kernel at s can compute the norm itself (same decision at both nodes), -1 otherwise.
static int ggml_cuda_gdn_norm_gate_match(const ggml_cgraph * cgraph, int s) {
    static const bool fuse = getenv("GGML_CUDA_FUSE_GDN_NORM_GATE") == nullptr || atoi(getenv("GGML_CUDA_FUSE_GDN_NORM_GATE")) != 0;
    if (!fuse || s < 2 || s + 1 >= cgraph->n_nodes) {
        return -1;
    }
    const ggml_tensor * un  = cgraph->nodes[s];
    const ggml_tensor * mul = cgraph->nodes[s + 1];
    if (un->op != GGML_OP_UNARY || ggml_get_unary_op(un) != GGML_UNARY_OP_SIGMOID || un->type != GGML_TYPE_F32 ||
            mul->op != GGML_OP_MUL || (mul->src[0] != un && mul->src[1] != un) ||
            !ggml_cuda_can_fuse(cgraph, s, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SIGMOID })) {
        return -1;
    }
    const ggml_tensor * nm = mul->src[0] == un ? mul->src[1] : mul->src[0];
    int m = -1;
    for (int j = s - 1; j >= 1 && j >= s - 32; --j) {
        if (cgraph->nodes[j] == nm) { m = j; break; }
    }
    if (m < 1) {
        return -1;
    }
    const ggml_tensor * rn = cgraph->nodes[m - 1];
    const ggml_tensor * w  = nm->src[1];
    const ggml_tensor * xn = rn->src[0];
    const ggml_tensor * z  = un->src[0];
    if (nm->op != GGML_OP_MUL || rn->op != GGML_OP_RMS_NORM || nm->src[0] != rn || rn->type != GGML_TYPE_F32 ||
            nm->type != GGML_TYPE_F32 || !w || w->type != GGML_TYPE_F32 || !ggml_is_contiguous(w) || w->ne[0] != 128 ||
            ggml_nelements(w) != 128 || !xn || xn->type != GGML_TYPE_F32 || xn->ne[0] != 128 || xn->nb[0] != sizeof(float) ||
            xn->ne[3] != 1 || !ggml_are_same_shape(rn, nm) || !ggml_are_same_shape(nm, z) || !ggml_is_contiguous(nm) ||
            !ggml_is_contiguous(z) || ggml_nelements(mul) % 256 != 0 || ggml_nelements(mul) != ggml_nelements(z) ||
            ggml_cuda_count_uses(cgraph, rn) != 1 || ggml_cuda_count_uses(cgraph, nm) != 1) {
        return -1;
    }
    // nothing between the norm and the gate writes over its input or weight
    const uintptr_t xlo = (uintptr_t) xn->data;
    const uintptr_t xhi = xlo + (size_t) (xn->ne[1] - 1)*xn->nb[1] + (size_t) (xn->ne[2] - 1)*xn->nb[2] + 128*sizeof(float);
    for (int j = m + 1; j < s; ++j) {
        const ggml_tensor * t = cgraph->nodes[j];
        if (t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE ||
                t->op == GGML_OP_NONE || !t->data) {
            continue;
        }
        const uintptr_t lo = (uintptr_t) t->data, hi = lo + ggml_nbytes(t);
        if ((lo < xhi && xlo < hi) || (lo < (uintptr_t) w->data + 128*sizeof(float) && (uintptr_t) w->data < hi)) {
            return -1;
        }
    }
    if (ggml_cuda_find_mul_q8_1_matmul(cgraph, s + 1, mul) == nullptr) {
        return -1;
    }
    return m - 1;
}

// At the RMS_NORM: skip RMS_NORM + MUL when the gate kernel below will compute them.  GGML_CUDA_FUSE_GDN_NORM_GATE=0: off.
static int ggml_cuda_gdn_norm_gate_skip(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, int i) {
    const ggml_tensor * rn = cgraph->nodes[i];
    if (rn->op != GGML_OP_RMS_NORM || i + 1 >= cgraph->n_nodes) {
        return 0;
    }
    for (int s = i + 2; s < cgraph->n_nodes && s <= i + 32; ++s) {
        const ggml_tensor * t = cgraph->nodes[s];
        if (t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == GGML_UNARY_OP_SIGMOID && ggml_cuda_gdn_norm_gate_match(cgraph, s) == i) {
            if (cuda_ctx->gdn_norm_pending != nullptr) {
                GGML_ABORT("GDN norm %s was skipped but never computed", cuda_ctx->gdn_norm_pending->name);
            }
            cuda_ctx->gdn_norm_pending = cgraph->nodes[i + 1];
            cuda_ctx->gdn_norm_gate_at = s;
            return 1;
        }
    }
    return 0;
}

// At the gate: the skipped norm computed by the gate -> Q8_1 kernel.  Returns the extra nodes consumed.
static int ggml_cuda_gdn_norm_gate_try_fuse(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, int s) {
    const ggml_tensor * un = cgraph->nodes[s];
    if (cuda_ctx->gdn_norm_pending == nullptr || un->op != GGML_OP_UNARY) {
        return 0;
    }
    const int r = ggml_cuda_gdn_norm_gate_match(cgraph, s);
    if (r < 0 || cgraph->nodes[r + 1] != cuda_ctx->gdn_norm_pending) {
        return 0;
    }
    cuda_ctx->gdn_norm_pending = nullptr;
    cuda_ctx->gdn_norm_gate_at = -1;
    ggml_tensor * mul = cgraph->nodes[s + 1];
    const ggml_tensor * mm = ggml_cuda_find_mul_q8_1_matmul(cgraph, s + 1, mul);
    ggml_cuda_op_unary_mul_q8_1(*cuda_ctx, cgraph->nodes[s], mul, mm, nullptr, cgraph->nodes[r]);
    return cgraph->nodes[s + 2] == mm ? 1 : 2;
}

struct ggml_cuda_topk_moe_match {
    ggml_cuda_topk_moe_args args;
    const ggml_tensor *     logits;
    ggml_tensor *           weights;
    ggml_tensor *           ids;
    const ggml_tensor *     bias;
    const ggml_tensor *     clamp;
    const ggml_tensor *     scale;
};

// The topk-moe chain starting at node i: the number of nodes it covers (0: no fusion) and its tensors.
static int ggml_cuda_topk_moe_match_at(const ggml_cgraph * cgraph, int i, ggml_cuda_topk_moe_match & m) {
    if (cgraph->nodes[i]->op != GGML_OP_UNARY && cgraph->nodes[i]->op != GGML_OP_SOFT_MAX &&
            cgraph->nodes[i]->op != GGML_OP_ARGSORT) {
        return 0;
    }
    ggml_cuda_topk_moe_args args;
    const bool              can_fuse = ggml_cuda_topk_moe_fusion(cgraph, i, args);
    std::vector<ggml_op>    ops;
    ops.reserve(13);  // max ops; avoids gcc -Wstringop-overflow false positive

    if (can_fuse) {
        const ggml_tensor * logits  = cgraph->nodes[i]->src[0];
        ggml_tensor *       weights = nullptr;
        ggml_tensor *       ids     = nullptr;
        const ggml_tensor * bias    = nullptr;
        const ggml_tensor * clamp   = nullptr;
        const ggml_tensor * scale   = nullptr;

        if (!args.delayed_softmax) {
            int out_nodes[2];  // nodes which can't be elided

            if (args.sigmoid) {
                ops.insert(ops.end(), { GGML_OP_UNARY });
            } else if (args.sqrt_softplus) {
                ops.insert(ops.end(), { GGML_OP_UNARY, GGML_OP_SQRT });
            } else {
                ops.insert(ops.end(), { GGML_OP_SOFT_MAX });
            }
            ops.push_back(GGML_OP_RESHAPE);
            if (args.prob_bias) {
                ops.push_back(GGML_OP_ADD);
                bias = cgraph->nodes[i + ops.size() - 1]->src[1];
            }
            if (args.n_groups > 0) {
                for (const auto & p : topk_moe_group_pattern) {
                    ops.push_back(p.op);
                }
            }
            ops.insert(ops.end(), { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
            out_nodes[0] = i + ops.size() - 2;
            ids = cgraph->nodes[out_nodes[0]];

            if (args.norm) {
                ops.insert(ops.end(),
                           { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
                clamp = cgraph->nodes[i + ops.size() - 3];
            }
            if (args.scale) {
                ops.insert(ops.end(), { GGML_OP_SCALE });
                scale = cgraph->nodes[i + ops.size() - 1];
            }

            weights      = cgraph->nodes[i + ops.size() - 1];
            out_nodes[1] = i + ops.size() - 1;

            if (!ggml_cuda_topk_moe_fusion_disabled() &&
                    ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                    ggml_cuda_should_use_topk_moe(cgraph->nodes[i], logits, weights, ids) &&
                    ggml_cuda_check_fusion_memory_ranges(cgraph, i, ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
                m = { args, logits, weights, ids, bias, clamp, scale };
                return (int) ops.size();
            }
        } else if (!args.norm && !args.prob_bias) {
            //special case gpt-oss, no norm, no bias.
            ops.insert(ops.end(), { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_RESHAPE,
                                    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE });
            weights                     = cgraph->nodes[i + 5];
            ids                         = cgraph->nodes[i + 1];
            const ggml_tensor * softmax = cgraph->nodes[i + 4];

            int out_nodes[2] = { i + 1, i + 5 };
            if (!ggml_cuda_topk_moe_fusion_disabled() &&
                    ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                    ggml_cuda_should_use_topk_moe(softmax, logits, weights, ids) &&
                    ggml_cuda_check_fusion_memory_ranges(cgraph, i, ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
                m = { args, logits, weights, ids, bias, clamp, scale };
                return (int) ops.size();
            }
        }
    }
    return 0;
}

int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {

    if (ggml_cuda_fusion_disabled()) {
        return 0;
    }
    // A deferred Q norm+rope whose K node was passed (taken by another fusion) is launched on its own before anything else.
    if (cuda_ctx->qk_pend_graph != nullptr && (cuda_ctx->qk_pend_graph != cgraph || i > cuda_ctx->qk_pend_j || i < cuda_ctx->qk_pend_i)) {
        if (cuda_ctx->qk_pend_graph != cgraph || i < cuda_ctx->qk_pend_i) {
            GGML_ABORT("deferred Q norm+rope was left in an earlier graph");
        }
        const int qi = cuda_ctx->qk_pend_i;
        cuda_ctx->qk_pend_graph = nullptr;
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, cgraph->nodes[qi], cgraph->nodes[qi + 1], cgraph->nodes[qi + 2], nullptr);
    }
    if (ggml_cuda_cache_blocks_fusion(cuda_ctx, cgraph, i)) {
        cuda_ctx->mwr_fold_experts = nullptr;
        return 0;
    }
    // MoE down + weighted reduction fold (owned by moe-weighted-reduction.cu): arms the down MUL_MAT_ID, fuses no node
    ggml_cuda_mwr_fold_mark(*cuda_ctx, cgraph, i);
    // a skipped GDN norm whose gate node was passed (taken by another fusion) or left in an earlier graph
    if (cuda_ctx->gdn_norm_pending != nullptr && (i == 0 || i > cuda_ctx->gdn_norm_gate_at)) {
        GGML_ABORT("GDN norm %s was skipped but its gate node was never reached", cuda_ctx->gdn_norm_pending->name);
    }
    // ahead of every other matcher, so the skip and the gate decide on the same nodes
    if (const int n_gdn_norm_skip = ggml_cuda_gdn_norm_gate_skip(cuda_ctx, cgraph, i)) {
        return n_gdn_norm_skip;
    }
    if (const int n_gdn_norm_gate = ggml_cuda_gdn_norm_gate_try_fuse(cuda_ctx, cgraph, i)) {
        return n_gdn_norm_gate;
    }
    // A skipped GDN output norm is recomputed by the gate kernel, which reads xn and w directly, so its own
    // MUL node is dead: skip it too instead of letting it run unfused over a norm output nobody wrote.
    if (cuda_ctx->gdn_norm_pending != nullptr && cgraph->nodes[i] == cuda_ctx->gdn_norm_pending) {
        return 1;
    }
    // While a norm is pending, no other fusion may start before its gate node: one that spanned the gate
    // would consume it, leaving the skipped norm uncomputed (the two aborts above then fire).  The window
    // is [MUL_MAT, RESHAPE] on the only graph this fires on (qwen4exp; measured span 4, no exceptions), so
    // the nodes run unfused and the cost is one plain matmul per GDN layer.  This makes the abort checks
    // unreachable rather than merely unlikely, so they stay as tripwires.
    if (cuda_ctx->gdn_norm_pending != nullptr && i < cuda_ctx->gdn_norm_gate_at) {
        return 0;
    }
    if (const int n_cont_gate = ggml_cuda_cont_gate_q8_1_try_fuse(cuda_ctx, cgraph, i)) {
        return n_cont_gate;
    }

    // fused gate+up+GLU MMQ (prefill): hard opt-out for A/B and regression testing
    static bool disable_moe_mmq = getenv("GGML_CUDA_DISABLE_MOE_MMQ_FUSION") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_MOE_MMQ_FUSION"));

    const int cc = ggml_cuda_info().devices[cuda_ctx->device].cc;

    ggml_tensor * node = cgraph->nodes[i];

    // HC_COMBINE -> BF16 HC_MIX (GGML_CUDA_FUSE_HC_COMBINE_MIX=0: off): the combine and the mixer's norm in one kernel
    if (const int n_hc_combine_mix = ggml_cuda_hc_combine_mix_try_fuse(*cuda_ctx, cgraph, i)) {
        return n_hc_combine_mix;
    }

    // GDN snapshot CPYs already written by the direct conv kernel (GGML_CUDA_DISABLE_GDN_CONV_SNAP=1: off)
    if (ggml_cuda_gdn_conv_snap_try_fuse(cgraph, i)) {
        return 1;
    }

    // batched copy (GGML_CUDA_FUSE_CPY_BATCH=0: off): consecutive same-layout f32 copies as one launch
    if (const int n_cpy_batch = ggml_cuda_cpy_batch_try_fuse(*cuda_ctx, cgraph, i)) {
        return n_cpy_batch;
    }

    // GLU -> Q8_1 (GGML_CUDA_FUSE_GLU_Q8_1=0: off): an F32 GLU whose next node (or the one after a reshape) is a verify-band
    // mmvq matmul over it is marked here (RDNA4 / RDNA3_0 only, through ggml_cuda_should_fuse_mul_mat_vec_q(mm, true)); the regular
    // GLU launcher then also writes the matmul's Q8_1 blocks into the quantize cache, which the matmul finds.
    // GLU -> Q8_1 marking (owned by mmvq.cu).
    ggml_cuda_glu_q8_1_mark(*cuda_ctx, cgraph, i);

    // Upstream fused shared-expert MMVQ (owned by mmvq.cu).
    if (const int n_shared_expert = ggml_cuda_match_shared_expert_try_fuse(*cuda_ctx, cgraph, i)) {
        return n_shared_expert;
    }

    // Prefill indexer head reduction (relu + head-sum) for the qwen4exp sparse-attention graph.
    if (const int n_idx_relu_sum = ggml_cuda_indexer_relu_sum_try_fuse(*cuda_ctx, cgraph, i)) {
        return n_idx_relu_sum;
    }

    // Depthwise causal conv1d fusions (qwen4exp GDN + PLE), ported from the halo-box
    // reference: the CONCAT(state, x) materialization and the SSM_CONV / tap-mul-add
    // chain are replaced by a direct kernel that reads state+x and writes the conv
    // output (+ silu).  The CONCAT is still partially materialized for the recurrent
    // snapshot copies (its tail only).  Prefill only (T >= 256, C % 256 == 0).
    // A/B / bisect kill switch: GGML_CUDA_DISABLE_CONV_FUSION=1.
    if (const int n_ple_conv = ggml_cuda_ple_conv_try_fuse(*cuda_ctx, cgraph, i)) {
        return n_ple_conv;
    }
    if (const int n_gdn_conv = ggml_cuda_gdn_conv_try_fuse(*cuda_ctx, cgraph, i)) {
        return n_gdn_conv;
    }

    // Narrow-row RMS norm (ncols <= 256, >= 4096 rows) and its sigmoid-gated form, ported from the
    // halo-box reference (Phase-1 item 4): 8 rows per 256-thread block instead of one block per row,
    // which is block-scheduling bound for the model's per-head norms.  Bit-identical to
    // rms_norm_f32<256,{true,false}> (same per-warp xor trees + 8-partial xor tree).
    // A/B / bisect kill switch: GGML_CUDA_DISABLE_NORM_ROWS=1.
    if (const int n_norm_rows = ggml_cuda_norm_rows_try_fuse(*cuda_ctx, cgraph, i)) {
        return n_norm_rows;
    }

    // qwen4exp IQ4_NL/Q8_0 routed down projection followed by the 10-expert weighted sum (ported
    // from halo-box/strix-llama.cpp): compute all selected experts for one output row in a wave and
    // apply their routing weights immediately, avoiding the [n_embd, n_used] intermediate and its
    // separate reduction launch. Hard opt-out (A/B, regression testing). Single-token only
    // (n_tokens == 1, gate in ggml_cuda_mul_mat_id_weighted_rdna3_5_ok).
    // qwen4exp routed down projection + 10-expert weighted sum (owned by mmvq.cu).
    // GGML_CUDA_DISABLE_WEIGHTED_DOWN=1 turns it off.
    if (const int n_weighted_down = ggml_cuda_mul_mat_id_weighted_down_try_fuse(*cuda_ctx, cgraph, i)) {
        return n_weighted_down;
    }

    // Two consecutive MUL_MAT(_ID) nodes sharing src1/ids (owned by mmq.cu).
    // GGML_PAIR_OFF / GGML_PAIR_DENSE_OFF are the A/B kill switches.
    if (const int n_mm_q_pair = ggml_cuda_mul_mat_q_pair_try_fuse(*cuda_ctx, cgraph, i)) {
        return n_mm_q_pair;
    }

    if (const int n_mwr = ggml_cuda_moe_weighted_reduction_try_fuse(*cuda_ctx, cgraph, i)) {
        return n_mwr;
    }

    // Verify band: the residual ADD in front of such a norm is folded into the norm kernel (norm surface).
    // GGML_CUDA_FUSE_ADD_RMS_Q8=0 turns it off.
    if (const int n_add_rms_q8 = ggml_cuda_add_rms_q8_1_try_fuse(*cuda_ctx, cgraph, i)) {
        return n_add_rms_q8;
    }

    // rms_norm + norm-weight MUL whose output feeds an mmvq matmul (norm surface).
    // GGML_CUDA_DISABLE_NORM_Q8_1=1 turns it off.
    if (const int n_norm_q8_1 = ggml_cuda_norm_q8_1_try_fuse(*cuda_ctx, cgraph, i)) {
        return n_norm_q8_1;
    }

    // UNARY(sigmoid) beta -> GATED_DELTA_NET (GGML_CUDA_FUSE_GDN_BETA_SIGMOID=0: off; gdn surface)
    if (const int n_gdn_beta_sig = ggml_cuda_gdn_beta_sigmoid_try_fuse(*cuda_ctx, cgraph, i)) {
        return n_gdn_beta_sig;
    }
    // gated_delta_net -> cpy: scatter recurrent-state snapshots into the cache (gdn surface)
    if (const int n_gdn_cpy = ggml_cuda_gdn_cache_cpy_try_fuse(*cuda_ctx, cgraph, i)) {
        return n_gdn_cpy;
    }

    // ssm_scan -> cpy: scatter recurrent-state snapshots into the cache (ssm-scan surface)
    if (const int n_ssm_cpy = ggml_cuda_ssm_scan_cache_cpy_try_fuse(*cuda_ctx, cgraph, i)) {
        return n_ssm_cpy;
    }

    // routed SWIGLU -> MMQ down feed (owned by mmq.cu; MMB stands it down for its fused GLU).
    if (const int n_swiglu_mmq = ggml_cuda_mul_mat_q_swiglu_try_fuse(*cuda_ctx, cgraph, i)) {
        return n_swiglu_mmq;
    }
    // dense SWIGLU -> MUL_MAT that runs as mmq (owned by mmq.cu). GGML_CUDA_FUSE_SWIGLU_MMQ=0 off.
    if (const int n_swiglu_mmq_dense = ggml_cuda_mul_mat_q_swiglu_dense_try_fuse(*cuda_ctx, cgraph, i)) {
        return n_swiglu_mmq_dense;
    }

    // 1-token router MUL_MAT + topk-moe in one launch (owned by topk-moe.cu)
    if (node->op == GGML_OP_MUL_MAT && i + 1 < cgraph->n_nodes) {
        ggml_cuda_topk_moe_match m;
        if (const int n_topk = ggml_cuda_topk_moe_match_at(cgraph, i + 1, m)) {
            if (ggml_cuda_router_topk_moe_ok(*cuda_ctx, node, m.logits, m.bias, m.args) && ggml_nrows(m.logits) == 1) {
                ggml_cuda_op_router_topk_moe(*cuda_ctx, node, m.weights, m.ids, m.clamp, m.scale, m.args);
                return n_topk;
            }
        }
    }

    //topk-moe
    {
        ggml_cuda_topk_moe_match m;
        if (const int n_topk = ggml_cuda_topk_moe_match_at(cgraph, i, m)) {
            ggml_cuda_op_topk_moe(*cuda_ctx, m.logits, m.weights, m.ids, m.clamp, m.scale, m.bias, m.args);
            return n_topk - 1;
        }
    }

    //RoPE + view + set-rows
    static const bool disable_rope_setrows = getenv("GGML_CUDA_DISABLE_ROPE_SETROWS") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_ROPE_SETROWS"));
    if (!disable_rope_setrows && ggml_cuda_can_fuse(cgraph, i, { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
        ggml_tensor * rope     = cgraph->nodes[i];
        ggml_tensor * set_rows = cgraph->nodes[i + 2];

        ggml_cuda_op_rope_fused(*cuda_ctx, rope, set_rows);
        return 2;
    }

    // Snake activation: y = x + sin(a*x)^2 * inv_b
    // Naive 5-op decomposition emitted by frontends: mul -> sin -> sqr -> mul -> add
    static const bool disable_snake = getenv("GGML_CUDA_DISABLE_SNAKE") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_SNAKE"));
    if (!disable_snake && ggml_can_fuse_subgraph(cgraph, i,
            { GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_ADD },
            { i + 4 })) {
        const ggml_tensor * mul0 = cgraph->nodes[i];
        const ggml_tensor * sqr  = cgraph->nodes[i + 2];
        const ggml_tensor * mul1 = cgraph->nodes[i + 3];
        ggml_tensor *       add  = cgraph->nodes[i + 4];

        // x carries the full activation shape, a is the broadcast operand
        const ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
        const ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

        // mul1 reads sqr and inv_b in either operand order
        const ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

        // closure check: the trailing add must read the same x as the leading mul
        const ggml_tensor * x_in_add = (add->src[0] == mul1) ? add->src[1] : add->src[0];

        // Kernel iterates over total = T * C, so x and add must be 2D and
        // a / inv_b must collapse to [1, C, 1, 1]. Higher dims are not handled.
        const bool dim_ok   = (x->ne[2]   == 1 && x->ne[3]   == 1) &&
                              (add->ne[2] == 1 && add->ne[3] == 1) &&
                              (a->ne[2]   == 1 && a->ne[3]   == 1);
        const bool shape_ok = ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];

        // x is in the supported whitelist and every chain intermediate shares
        // x's type. launch_snake reads a and inv_b as const float *, so they
        // stay F32.
        const ggml_tensor * sin1 = cgraph->nodes[i + 1];
        const bool types_ok = (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
                              (a->type    == GGML_TYPE_F32) && (inv_b->type == GGML_TYPE_F32) &&
                              (mul0->type == x->type) && (sin1->type  == x->type) &&
                              (sqr->type  == x->type) && (mul1->type  == x->type) &&
                              (add->type  == x->type);

        // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
        const bool contig_ok = ggml_is_contiguous(x) && ggml_is_contiguous(add) &&
                               ggml_is_contiguous(a) && ggml_is_contiguous(inv_b);

        if (types_ok && shape_ok && dim_ok && contig_ok && x_in_add == x) {
            ggml_cuda_op_snake_fused(*cuda_ctx, x, a, inv_b, add);
            return 4;
        }
    }

    // multi-(add or mul)
    static const bool disable_fused_addmul = getenv("GGML_CUDA_DISABLE_FUSED_ADDMUL") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_FUSED_ADDMUL"));
    if (!disable_fused_addmul && (node->op == GGML_OP_ADD || node->op == GGML_OP_MUL)) {
        int     n_fuse = 0;
        ggml_op ops[8];
        std::fill(ops, ops + 8, node->op);

        for (; n_fuse <= 6; ++n_fuse) {
            if (!ggml_can_fuse(cgraph, i + n_fuse, ops + n_fuse, 2)) {
                break;
            }
            if (cgraph->nodes[i + n_fuse] != cgraph->nodes[i + n_fuse + 1]->src[0]) {
                break;
            }
            if (!ggml_are_same_layout(cgraph->nodes[i + n_fuse]->src[1], cgraph->nodes[i + n_fuse + 1]->src[1])) {
                break;
            }
        }

        n_fuse++;

        if (n_fuse > 1) {
            ggml_tensor fused_node;
            memcpy(&fused_node, node, sizeof(ggml_tensor));
            for (int j = 0; j < n_fuse - 1; ++j) {
                fused_node.src[j + 2] = cgraph->nodes[i + j + 1]->src[1];
            }
            fused_node.data = cgraph->nodes[i + n_fuse - 1]->data;
            if (node->op == GGML_OP_ADD) {
                ggml_cuda_op_fused_add(*cuda_ctx, &fused_node, n_fuse);
            } else {
                ggml_cuda_op_fused_mul(*cuda_ctx, &fused_node, n_fuse);
            }
            return n_fuse - 1;
        }
    }

    bool fused_mul_mat_vec = false;
    int  fused_node_count  = 0;

    // SSM conv-input fusion (owned by mmvq.cu). GGML_CUDA_DISABLE_SSM_CONV_IN=1 turns it off.
    if (const int n_ssm_conv_in = ggml_cuda_ssm_conv_in_try_fuse(*cuda_ctx, cgraph, i)) {
        return n_ssm_conv_in;
    }

    auto get_mul_mat_scale = [](const ggml_tensor * scale_node, const ggml_tensor * mm_node) -> const ggml_tensor * {
        const bool scale_lhs_mm = scale_node->src[0] == mm_node;
        const bool scale_rhs_mm = scale_node->src[1] == mm_node;
        if (!scale_lhs_mm && !scale_rhs_mm) {
            return nullptr;
        }

        const ggml_tensor * scale = scale_lhs_mm ? scale_node->src[1] : scale_node->src[0];
        if (mm_node->src[0]->type != GGML_TYPE_NVFP4 || scale_node->type != GGML_TYPE_F32 ||
                scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || ggml_nelements(scale) != 1 ||
                !ggml_are_same_shape(scale_node, mm_node)) {
            return nullptr;
        }

        return scale;
    };

    auto get_mul_mat_id_scale = [](const ggml_tensor * reshape, const ggml_tensor * repeat, const ggml_tensor * getrows,
            const ggml_tensor * scale_node, const ggml_tensor * mm_node) -> const ggml_tensor * {
        if (repeat->src[0] != reshape || getrows->src[0] != repeat || getrows->src[1] != mm_node->src[2]) {
            return nullptr;
        }
        if (!((scale_node->src[0] == mm_node && scale_node->src[1] == getrows) ||
                (scale_node->src[0] == getrows && scale_node->src[1] == mm_node))) {
            return nullptr;
        }

        const ggml_tensor * scale = reshape->src[0];
        if (mm_node->src[0]->type != GGML_TYPE_NVFP4 || scale_node->type != GGML_TYPE_F32 ||
                scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || ggml_nelements(scale) != mm_node->src[0]->ne[2] ||
                !ggml_are_same_shape(scale_node, mm_node)) {
            return nullptr;
        }

        return scale;
    };

    auto get_bias_tensor = [](const ggml_tensor * bias_node, const ggml_tensor * mul_node, ggml_op op_bias) -> const ggml_tensor * {
        if (op_bias == GGML_OP_ADD) {
            if (bias_node->src[0] == mul_node) {
                return bias_node->src[1];
            }
            if (bias_node->src[1] == mul_node) {
                return bias_node->src[0];
            }
            return nullptr;
        }
        GGML_ASSERT(op_bias == GGML_OP_ADD_ID);
        GGML_ASSERT(bias_node->src[0] == mul_node);
        return bias_node->src[1];
    };

    // gate + glu + up, with optional scale/bias on both lanes.
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        if (op == GGML_OP_MUL_MAT) {
            for (const bool with_bias : { false, true }) {
                const int gate_idx       = i;
                const int gate_scale_idx = i + 1;
                const int gate_bias_idx  = with_bias ? i + 2 : -1;
                const int up_idx         = with_bias ? i + 3 : i + 2;
                const int up_scale_idx   = up_idx + 1;
                const int up_bias_idx    = with_bias ? up_idx + 2 : -1;
                const int glu_idx        = with_bias ? up_idx + 3 : up_idx + 2;

                const int out_nodes[] = { glu_idx };
                ggml_op ops[7];
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = bias_op;
                    ops[3] = op;
                    ops[4] = GGML_OP_MUL;
                    ops[5] = bias_op;
                    ops[6] = GGML_OP_GLU;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = op;
                    ops[3] = GGML_OP_MUL;
                    ops[4] = GGML_OP_GLU;
                }
                const int n_ops = with_bias ? 7 : 5;

                if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                        !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                    continue;
                }

                ggml_tensor * gate_n       = cgraph->nodes[gate_idx];
                ggml_tensor * gate_scale_n = cgraph->nodes[gate_scale_idx];
                ggml_tensor * gate_out_n   = with_bias ? cgraph->nodes[gate_bias_idx] : gate_scale_n;
                ggml_tensor * up_n         = cgraph->nodes[up_idx];
                ggml_tensor * up_scale_n   = cgraph->nodes[up_scale_idx];
                ggml_tensor * up_out_n     = with_bias ? cgraph->nodes[up_bias_idx] : up_scale_n;
                const ggml_tensor * glu = cgraph->nodes[glu_idx];

                if (!ggml_cuda_should_fuse_mul_mat(up_n, gate_n, glu,
                        with_bias ? up_out_n : nullptr, with_bias ? gate_out_n : nullptr, up_scale_n, gate_scale_n)) {
                    continue;
                }

                const ggml_tensor * gate_scale = get_mul_mat_scale(gate_scale_n, gate_n);
                const ggml_tensor * up_scale   = get_mul_mat_scale(up_scale_n, up_n);
                if (!gate_scale || !up_scale) {
                    continue;
                }

                const ggml_tensor * up_bias   = with_bias ? get_bias_tensor(up_out_n, up_scale_n, bias_op) : nullptr;
                const ggml_tensor * gate_bias = with_bias ? get_bias_tensor(gate_out_n, gate_scale_n, bias_op) : nullptr;
                if (with_bias && (!ggml_are_same_shape(gate_out_n->src[0], gate_out_n->src[1]) ||
                        !ggml_are_same_shape(up_out_n->src[0], up_out_n->src[1]))) {
                    continue;
                }

                const ggml_tensor * src0 = up_n->src[0];
                const ggml_tensor * src1 = up_n->src[1];
                const ggml_tensor * ids  = up_n->src[2];

                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate       = gate_n->src[0];
                fusion_data.x_bias     = up_bias;
                fusion_data.gate_bias  = gate_bias;
                fusion_data.x_scale    = up_scale;
                fusion_data.gate_scale = gate_scale;
                fusion_data.glu_op     = ggml_get_glu_op(glu);
                fusion_data.glu_limit  = ggml_get_op_params_f32(glu, 3);

                if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n) && (ids != nullptr || !ggml_cuda_rdna3_5_dense_glu_disabled())) {
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }
            }

            if (fused_mul_mat_vec) {
                break;
            }
        } else {
            for (const bool with_bias : { false, true }) {
                const int gate_idx       = i;
                const int gate_scale_idx = i + 4;
                const int gate_bias_idx  = with_bias ? i + 5 : -1;
                const int up_idx         = with_bias ? i + 6 : i + 5;
                const int up_scale_idx   = up_idx + 4;
                const int up_bias_idx    = with_bias ? up_idx + 5 : -1;
                const int glu_idx        = with_bias ? up_idx + 6 : up_idx + 5;

                const int out_nodes[] = { glu_idx };
                ggml_op ops[13];
                if (with_bias) {
                    ops[0]  = op;
                    ops[1]  = GGML_OP_RESHAPE;
                    ops[2]  = GGML_OP_REPEAT;
                    ops[3]  = GGML_OP_GET_ROWS;
                    ops[4]  = GGML_OP_MUL;
                    ops[5]  = bias_op;
                    ops[6]  = op;
                    ops[7]  = GGML_OP_RESHAPE;
                    ops[8]  = GGML_OP_REPEAT;
                    ops[9]  = GGML_OP_GET_ROWS;
                    ops[10] = GGML_OP_MUL;
                    ops[11] = bias_op;
                    ops[12] = GGML_OP_GLU;
                } else {
                    ops[0]  = op;
                    ops[1]  = GGML_OP_RESHAPE;
                    ops[2]  = GGML_OP_REPEAT;
                    ops[3]  = GGML_OP_GET_ROWS;
                    ops[4]  = GGML_OP_MUL;
                    ops[5]  = op;
                    ops[6]  = GGML_OP_RESHAPE;
                    ops[7]  = GGML_OP_REPEAT;
                    ops[8]  = GGML_OP_GET_ROWS;
                    ops[9]  = GGML_OP_MUL;
                    ops[10] = GGML_OP_GLU;
                }
                const int n_ops = with_bias ? 13 : 11;

                if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                        !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                    continue;
                }

                ggml_tensor * gate_n       = cgraph->nodes[gate_idx];
                ggml_tensor * gate_scale_n = cgraph->nodes[gate_scale_idx];
                ggml_tensor * gate_out_n   = with_bias ? cgraph->nodes[gate_bias_idx] : gate_scale_n;
                ggml_tensor * up_n         = cgraph->nodes[up_idx];
                ggml_tensor * up_scale_n   = cgraph->nodes[up_scale_idx];
                ggml_tensor * up_out_n     = with_bias ? cgraph->nodes[up_bias_idx] : up_scale_n;
                const ggml_tensor * glu = cgraph->nodes[glu_idx];

                if (!ggml_cuda_should_fuse_mul_mat(up_n, gate_n, glu,
                        with_bias ? up_out_n : nullptr, with_bias ? gate_out_n : nullptr, up_scale_n, gate_scale_n)) {
                    continue;
                }

                const ggml_tensor * gate_scale = get_mul_mat_id_scale(cgraph->nodes[gate_idx + 1], cgraph->nodes[gate_idx + 2],
                        cgraph->nodes[gate_idx + 3], gate_scale_n, gate_n);
                const ggml_tensor * up_scale = get_mul_mat_id_scale(cgraph->nodes[up_idx + 1], cgraph->nodes[up_idx + 2],
                        cgraph->nodes[up_idx + 3], up_scale_n, up_n);
                if (!gate_scale || !up_scale) {
                    continue;
                }

                const ggml_tensor * up_bias   = with_bias ? get_bias_tensor(up_out_n, up_scale_n, bias_op) : nullptr;
                const ggml_tensor * gate_bias = with_bias ? get_bias_tensor(gate_out_n, gate_scale_n, bias_op) : nullptr;

                const ggml_tensor * src0 = up_n->src[0];
                const ggml_tensor * src1 = up_n->src[1];
                const ggml_tensor * ids  = up_n->src[2];

                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate       = gate_n->src[0];
                fusion_data.x_bias     = up_bias;
                fusion_data.gate_bias  = gate_bias;
                fusion_data.x_scale    = up_scale;
                fusion_data.gate_scale = gate_scale;
                fusion_data.glu_op     = ggml_get_glu_op(glu);
                fusion_data.glu_limit  = ggml_get_op_params_f32(glu, 3);

                if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n) && (ids != nullptr || !ggml_cuda_rdna3_5_dense_glu_disabled())) {
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }
            }

            if (fused_mul_mat_vec) {
                break;
            }
        }

        if (ggml_cuda_can_fuse(cgraph, i, { op, bias_op, op, bias_op, GGML_OP_GLU }, {})) {
            ggml_tensor * glu         = cgraph->nodes[i + 4];
            ggml_tensor * gate_bias_n = glu->src[0];
            ggml_tensor * up_bias_n   = glu->src[1];

            //we don't assume the order for {gate, up}. Instead infer it from the bias tensor
            ggml_tensor * gate_n = nullptr;
            ggml_tensor * up_n   = nullptr;

            if (gate_bias_n->src[0] == cgraph->nodes[i] || gate_bias_n->src[1] == cgraph->nodes[i]) {
                gate_n = cgraph->nodes[i];
                up_n   = cgraph->nodes[i + 2];
            } else if (gate_bias_n->src[0] == cgraph->nodes[i + 2] || gate_bias_n->src[1] == cgraph->nodes[i + 2]) {
                gate_n = cgraph->nodes[i + 2];
                up_n   = cgraph->nodes[i];
            } else {
                continue;
            }

            const ggml_tensor * up_bias_tensor   = get_bias_tensor(up_bias_n, up_n, bias_op);
            const ggml_tensor * gate_bias_tensor = get_bias_tensor(gate_bias_n, gate_n, bias_op);

            if (!up_bias_tensor || !gate_bias_tensor) {
                continue;
            }

            // we don't support repeating adds
            if (bias_op == GGML_OP_ADD && (!ggml_are_same_shape(gate_bias_n->src[0], gate_bias_n->src[1]) ||
                                           !ggml_are_same_shape(up_bias_n->src[0], up_bias_n->src[1]))) {
                continue;
            }

            const ggml_tensor * src0 = up_n->src[0];
            const ggml_tensor * src1 = up_n->src[1];
            const ggml_tensor * ids  = up_n->src[2];

            if (ggml_cuda_should_fuse_mul_mat_vec_f(up_n) && (ids != nullptr || !ggml_cuda_rdna3_5_dense_glu_disabled())) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate_n->src[0];
                fusion_data.x_bias    = up_bias_tensor;
                fusion_data.gate_bias = gate_bias_tensor;
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 5;
                break;
            }

            if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n) && (ids != nullptr || !ggml_cuda_rdna3_5_dense_glu_disabled())) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate_n->src[0];
                fusion_data.x_bias    = up_bias_tensor;
                fusion_data.gate_bias = gate_bias_tensor;
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 5;
                break;
            }
        } else if (ggml_cuda_can_fuse(cgraph, i, { op, op, GGML_OP_GLU }, {})) {
            ggml_tensor * glu  = cgraph->nodes[i + 2];
            ggml_tensor * gate = glu->src[0];
            ggml_tensor * up   = glu->src[1];

            bool ok = (gate == cgraph->nodes[i] && up == cgraph->nodes[i + 1]) ||
                      (gate == cgraph->nodes[i + 1] && up == cgraph->nodes[i]);

            if (!ok) {
                continue;
            }

            const ggml_tensor * src0 = up->src[0];
            const ggml_tensor * src1 = up->src[1];
            const ggml_tensor * ids  = up->src[2];

            // MMB fused MoE gate+up+swiglu (pwilkin port). Prefill-only by the shared predicate.
            if (op == GGML_OP_MUL_MAT_ID && ids != nullptr &&
                    ggml_cuda_mmb_supported_glu(gate->src[0], up->src[0], src1, ids, glu)) {
                ggml_cuda_mul_mat_id_mmb_glu(*cuda_ctx, gate->src[0], up->src[0], src1, ids, glu);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }
            if (ggml_cuda_should_fuse_mul_mat_vec_f(up) && (ids != nullptr || !ggml_cuda_rdna3_5_dense_glu_disabled())) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }

            if (ggml_cuda_should_fuse_mul_mat_vec_q(up) && (ids != nullptr || !ggml_cuda_rdna3_5_dense_glu_disabled())) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                // wip/moe-expert-cache: the decode MoE may read a compact arena + slot-remapped ids
                // instead of the full expert tables.  Redirect BOTH lanes together (the cache took
                // both inputs over, skipping the scheduler copy); on a decline the scheduler's full
                // copy stands and the original tensors are used.
                ggml_tensor src0_c, ids_c, gate_c;
                ggml_cuda_mm_fusion_args_host fusion_local = fusion_data;
                if (moe_cache_redirect_fused(glu, src0, fusion_data.gate, ids, cuda_ctx->device, cuda_ctx->stream(),
                                             &src0_c, &ids_c, &gate_c)) {
                    fusion_local.gate = &gate_c;
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, &src0_c, src1, &ids_c, glu, &fusion_local);
                } else {
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                }
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }

            // Dense fused gate/up/GLU MMQ (upstream #29948; the write-back relies on the MMA accumulator layout).
            if (ggml_cuda_should_fuse_mul_mat_q(up, gate, glu)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }

            // Prefill MMQ path: the mmvq/mmvf fused kernels only handle decode
            // (n_tokens <= MMVQ_MAX_BATCH_SIZE) or F32/F16 src0. For the batched
            // quantized case the gate+up+GLU triple runs as separate ops; fuse it
            // into one MMQ kernel that reads both weight streams and applies the
            // GLU epilogue. The J tile-width caps in mul_mat_q_switch_J are tuned
            // on RDNA4 (gfx1201). RDNA3_5 (Strix Halo, gfx1151) was added after
            // validation 2026-09-05 (coherence IDENTICAL fused-on vs off, pp2048
            // +5.3% / pp16384 +4.6% on Qwen3.6-35B-A3B Q3_K_M ub 2048; the
            // RDNA4-tuned J caps transfer). RDNA3_0 (gfx1100, RX 7900XTX) was
            // added after the 2026-09-05 validation on this box: coherence
            // IDENTICAL, pp2048 +9.4% / pp16384 +7.8%, decode unchanged, and the
            // RDNA4-tuned J caps transfer there too (uncapping regressed pp2048
            // 5405->4819 / pp16384 4487->4070; a Q3_K@96 probe at 5094/4251 also
            // lost to the cap 64).
            const bool moe_mmq_type = src0->type == GGML_TYPE_Q3_K || src0->type == GGML_TYPE_Q4_K ||
                                      src0->type == GGML_TYPE_Q5_K || src0->type == GGML_TYPE_Q8_0 ||
                                      src0->type == GGML_TYPE_Q6_K;
            if (op == GGML_OP_MUL_MAT_ID && ids != nullptr && !disable_moe_mmq &&
                    (GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc) || GGML_CUDA_CC_IS_RDNA3_0(cc)) && moe_mmq_type &&
                    ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[2], /*n_experts=*/src0->ne[2])) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    // Dual-output mmvq fusion (owned by mmvq.cu).  The row concat runs first (the submitter's order, measured identical to
    // the reverse on gfx1100/gfx1201 -- see the block-15 notes in patches/README.md): it appends the rows of two more
    // same-src1 matmuls to the largest weight's launch, while the dual-output keeps the equal-height pair it owns.
    if (const int n_row_concat = ggml_cuda_mul_mat_row_concat_try_fuse(*cuda_ctx, cgraph, i)) {
        return n_row_concat;
    }
    if (const int n_dual_output = ggml_cuda_mul_mat_dual_output_try_fuse(*cuda_ctx, cgraph, i)) {
        return n_dual_output;
    }

    fused_mul_mat_vec = false;
    fused_node_count  = 0;

    // mul_mat + scale + optional bias
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        for (const bool with_bias : { false, true }) {
            const int n_ops = op == GGML_OP_MUL_MAT ? (with_bias ? 3 : 2) : (with_bias ? 6 : 5);
            const int out_nodes[] = { i + n_ops - 1 };
            ggml_op ops[6];
            if (op == GGML_OP_MUL_MAT) {
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = bias_op;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                }
            } else {
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_RESHAPE;
                    ops[2] = GGML_OP_REPEAT;
                    ops[3] = GGML_OP_GET_ROWS;
                    ops[4] = GGML_OP_MUL;
                    ops[5] = bias_op;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_RESHAPE;
                    ops[2] = GGML_OP_REPEAT;
                    ops[3] = GGML_OP_GET_ROWS;
                    ops[4] = GGML_OP_MUL;
                }
            }

            if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                    !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                continue;
            }

            ggml_tensor * mm_node    = cgraph->nodes[i];
            ggml_tensor * scale_node = op == GGML_OP_MUL_MAT ? cgraph->nodes[i + 1] : cgraph->nodes[i + 4];
            ggml_tensor * out_node   = with_bias ? cgraph->nodes[i + n_ops - 1] : scale_node;

            const ggml_tensor * scale = nullptr;
            if (op == GGML_OP_MUL_MAT) {
                scale = get_mul_mat_scale(scale_node, mm_node);
            } else {
                scale = get_mul_mat_id_scale(cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 3], scale_node, mm_node);
            }
            if (!scale) {
                continue;
            }

            const ggml_tensor * bias = with_bias ? get_bias_tensor(out_node, scale_node, bias_op) : nullptr;
            if (with_bias && !bias) {
                continue;
            }
            if (with_bias && bias_op == GGML_OP_ADD && !ggml_are_same_shape(out_node->src[0], out_node->src[1])) {
                continue;
            }
            if (with_bias && bias_op == GGML_OP_ADD_ID && out_node->src[2] != mm_node->src[2]) {
                continue;
            }

            const ggml_tensor * src0 = mm_node->src[0];
            const ggml_tensor * src1 = mm_node->src[1];
            const ggml_tensor * ids  = mm_node->src[2];

            ggml_cuda_mm_fusion_args_host fusion_data{};
            fusion_data.x_bias  = bias;
            fusion_data.x_scale = scale;

            if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_node)) {
                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, out_node, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = n_ops;
                break;
            }
        }
        if (fused_mul_mat_vec) {
            break;
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    // MoE down fold (owned by mmvq.cu). GGML_CUDA_DISABLE_MOE_DOWN_FOLD=1 turns it off.
    if (const int n_moe_down_fold = ggml_cuda_moe_down_fold_try_fuse(*cuda_ctx, cgraph, i)) {
        return n_moe_down_fold;
    }

    // Pair of L2 norms over two views of the same tensor (the SSM conv output
    // q/k slices). The view source is an external compute tensor, so the
    // subgraph helper cannot express the pattern; check the wiring manually.
    // Pair of L2 norms over two views of the same tensor (owned by norm.cu).
    // GGML_CUDA_DISABLE_L2_NORM_PAIR=1 turns it off.
    if (const int n_l2_norm_pair = ggml_cuda_l2_norm_pair_try_fuse(*cuda_ctx, cgraph, i)) {
        return n_l2_norm_pair;
    }

    // SSM gated delta net: gate = softplus(alpha*x + dt) * a, beta = sigmoid(beta*x).
    // Two small Q8_0 projections over the same input plus the gating chain, fused
    // into one kernel. Both outputs feed only the gated delta net kernel.
    // SSM gate/beta verify-band fusion (owned by mmvq.cu).
    if (const int n_gate_beta = ggml_cuda_gate_beta_verify_try_fuse(*cuda_ctx, cgraph, i)) {
        return n_gate_beta;
    }
    // The same pair with BF16 projections (owned by mmvf.cu).
    if (const int n_gate_beta_bf16 = ggml_cuda_gate_beta_bf16_try_fuse(*cuda_ctx, cgraph, i)) {
        return n_gate_beta_bf16;
    }
    // The shared-expert gate GEMV with the sigmoid / mul / add after it (owned by mmvf.cu).
    if (const int n_shexp_gate = ggml_cuda_shexp_gate_sigmoid_try_fuse(*cuda_ctx, cgraph, i)) {
        return n_shexp_gate;
    }

    // Shared-expert output chain: down projection + gate + gating + residual
    // adds. dst = down(swiglu) * sigmoid(gate(x)) + moe_out + ffn_residual.
    //
    // This is DISJOINT from upstream's fused shared-expert MMVQ
    // (ggml_cuda_match_shared_expert / the MUL_MAT_ID branch above, base bed0a8566),
    // which folds the routed+shared gate/up pair and writes the shared GLU; this
    // matcher consumes that GLU plus the shared gate_inp sigmoid.  Both fire on the
    // same layer (verified gfx1201 `-sm layer`, Qwen3.6-35B-A3B Q8_0: upstream
    // 200x + this 160x on one 2-token pass).  Under `-sm tensor` the routed expert
    // is sharded while the shared expert is mirrored, so the routed/shared n_ff
    // differs and the upstream arm stands down; this one still runs.  The
    // LLAMA_HC_BLK16 MWR merge (default off) consumes the ffn_out ADD and so
    // intentionally replaces this epilogue when it is armed.  (wip/shared-expert-fusion-reconcile.)
    //
    // Decode/verify band.  The fused gate reduction (shexp_gate_sigmoid) does not
    // reproduce the order of the standalone mmvq/MUL_MAT it replaces, so the fused and the
    // unfused chain are not bit-identical - and a 1-token decode and an n-token verify of
    // the same MoE layer must be.  The fused kernels are therefore token-generic and pinned
    // to the single-token reduction order, and the whole band (n_tokens <= MMVQ_MAX_BATCH_SIZE)
    // takes the fused path: the multi-token path no longer runs the unfused chain.  Worth +3.1%
    // decode on Qwen3.6-35B-A3B (tg128 101.6 vs 98.5 t/s), and the verify widths gain the same
    // epilogue.  The unfused chain remains the reference; set
    // GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1 to compare against it (then decode and verify differ,
    // as before the 2026-09-11 band amendment).  Hard opt-out for A/B too.
    // Shared-expert output chain (owned by mmvq.cu). GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1 off.
    if (const int n_shexp_down = ggml_cuda_shexp_down_gate_try_fuse(*cuda_ctx, cgraph, i)) {
        return n_shexp_down;
    }

    // mul_mat + add, with an optional view (reshape) node between the matmul and the add
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        // view (reshape) between the matmul and the add
        const bool has_view = i + 1 < cgraph->n_nodes && cgraph->nodes[i + 1]->op == GGML_OP_RESHAPE;

        if (has_view) {
            // use ggml_can_fuse_subgraph: views in the subgraph are allowed here
            const ggml_op ops[3] = { op, GGML_OP_RESHAPE, bias_op };
            const int out_nodes[] = { i + 2 };
            if (!ggml_can_fuse_subgraph(cgraph, i, 3, ops, out_nodes, 1) || cgraph->nodes[i + 1]->src[0] != cgraph->nodes[i]) {
                continue;
            }
        } else {
            if (!ggml_can_fuse(cgraph, i, { op, bias_op })) {
                continue;
            }
        }

        ggml_tensor * mm_node   = cgraph->nodes[i];
        ggml_tensor * bias_node = cgraph->nodes[has_view ? i + 2 : i + 1];

        // the add reads the matmul output directly, or through the view
        ggml_tensor * mm_or_view = has_view ? cgraph->nodes[i + 1] : mm_node;

        // The mmvq/mmvf fusion kernels are told to write into bias_node, but the
        // shape checks below (and ggml_cuda_should_fuse_mul_mat_vec_*) look at
        // mm_node.  Without a view those are the same tensor, so the guard is
        // sound.  With one they are not: a reshape can move the tokens between
        // dimensions, so a matmul that looks like a single-column GEMV
        // (ne = [n,1,2]) can be paired with an add whose destination is
        // ne = [n,2,1].  The kernels index the destination by ne[1]/ne[2] and
        // assert on exactly this (mmvq.cu: GGML_ASSERT(ids || dst->ne[1] == 1)).
        // Require the destination to satisfy the same constraint the kernels
        // assert before fusing through a view.  The single-sequence case is
        // unaffected.  (PR #15 / DanoPTT.)
        if (has_view) {
            const ggml_tensor * ids_node = mm_node->src[2];
            if (( ids_node && bias_node->ne[2] != 1) ||
                (!ids_node && bias_node->ne[1] != 1)) {
                continue;
            }
        }

        ggml_tensor * bias_tensor = nullptr;
        if (bias_op == GGML_OP_ADD) {
            if (bias_node->src[0] == mm_or_view) {
                bias_tensor = bias_node->src[1];
            } else if (bias_node->src[1] == mm_or_view) {
                bias_tensor = bias_node->src[0];
            } else {
                continue;
            }
        } else {
            if (bias_node->src[0] != mm_or_view) {
                continue;
            }
            bias_tensor = bias_node->src[1];
        }

        const ggml_tensor * src0 = mm_node->src[0];
        const ggml_tensor * src1 = mm_node->src[1];
        const ggml_tensor * ids  = mm_node->src[2];

        if (bias_op == GGML_OP_ADD_ID && bias_node->src[2] != ids) {
            continue;
        }

        if (bias_op == GGML_OP_ADD && !ggml_are_same_shape(bias_node->src[0], bias_node->src[1])) {
            continue;
        }

        ggml_cuda_mm_fusion_args_host fusion_data{};
        fusion_data.x_bias = bias_tensor;

        if (ggml_cuda_should_fuse_mul_mat_vec_f(mm_node)) {
            ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count  = has_view ? 3 : 2;
            break;
        }

        if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_node)) {
            ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count  = has_view ? 3 : 2;
            break;
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    if (const int n_fwht_q8_1 = ggml_cuda_fwht_q8_1_try_fuse(*cuda_ctx, cgraph, i)) {
        return n_fwht_q8_1;
    }

    const bool kv_allow_v = cuda_ctx->curr_stream_no == 0 && cuda_ctx->stream_context().concurrent_events.empty();
    if (const int n_kv = ggml_cuda_rms_norm_mul_rope_fwht_set_rows_count(cgraph, i, kv_allow_v)) {
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], nullptr, cgraph->nodes[i + 4],
                                             cgraph->nodes[i + 7], n_kv == 10 ? cgraph->nodes[i + 9] : nullptr);
        return n_kv - 1;
    }

    if (ggml_cuda_should_fuse_rms_norm_mul_rope_fwht(cgraph, i)) {
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], nullptr, cgraph->nodes[i + 4]);
        return 4;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
        if (cuda_ctx->qk_pend_graph == cgraph && cuda_ctx->qk_pend_j == i) {
            const int qi = cuda_ctx->qk_pend_i;
            const bool with_v = kv_allow_v && ggml_cuda_rope_set_rows_v_ok(cgraph, i);
            cuda_ctx->qk_pend_graph = nullptr;
            ggml_cuda_op_rms_norm_mul_rope_qk_pair(*cuda_ctx, cgraph->nodes[qi], cgraph->nodes[qi + 1], cgraph->nodes[qi + 2],
                                                   node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 4],
                                                   with_v ? cgraph->nodes[i + 6] : nullptr);
            return with_v ? 6 : 4;
        }
        if (kv_allow_v && ggml_cuda_rope_set_rows_v_ok(cgraph, i)) {
            ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 4],
                                                 nullptr, nullptr, cgraph->nodes[i + 6]);
            return 6;
        }
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 4]);
        return 4;
    }

    // GDN gate chain for the verify band: ADD(dt) -> SOFTPLUS -> MUL(a) with per-head broadcasts in one
    // kernel (decode fuses it into ssm_gate_beta).  GGML_CUDA_FUSE_GDN_GATE=0 turns it off.
    // GDN gate chain (owned by unary.cu). GGML_CUDA_FUSE_GDN_GATE=0 turns it off.
    if (const int n_gdn_gate = ggml_cuda_add_softplus_mul_try_fuse(*cuda_ctx, cgraph, i)) {
        return n_gdn_gate;
    }

    // NOTE: RMS_NORM + SCALE is fused upstream now (ggml_cuda_op_rms_norm_scale_fused,
    // handled by the ggml_cuda_can_fuse matcher below); block 08's older separate
    // kernel (GGML_CUDA_FUSE_RMS_SCALE) was retired in the r38 re-base.

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE }, {})) {
        if (const int j = ggml_cuda_qk_pair_k_at(cgraph, i, kv_allow_v); j > 0 && cuda_ctx->qk_pend_graph == nullptr) {
            cuda_ctx->qk_pend_graph = cgraph;
            cuda_ctx->qk_pend_i     = i;
            cuda_ctx->qk_pend_j     = j;
            return 2;
        }
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], nullptr);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD }, {})) {
        ggml_cuda_op_rms_norm_fused_add(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
        ggml_cuda_op_rms_norm_fused(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_SCALE }, {})) {
        // a second RMS_NORM + SCALE over a view of the same tensor with the same shape, strides, eps and scale (the GDN
        // q / k l2 norm), separated only by no-op nodes: both in one launch.  GGML_CUDA_DISABLE_NORM_PAIR=1: off.
        static const bool norm_pair_off = getenv("GGML_CUDA_DISABLE_NORM_PAIR") != nullptr && atoi(getenv("GGML_CUDA_DISABLE_NORM_PAIR")) != 0;
        const ggml_tensor * x0 = node->src[0];
        int j = i + 2;
        while (!norm_pair_off && j < cgraph->n_nodes && (cgraph->nodes[j]->op == GGML_OP_VIEW || cgraph->nodes[j]->op == GGML_OP_RESHAPE ||
                cgraph->nodes[j]->op == GGML_OP_PERMUTE || cgraph->nodes[j]->op == GGML_OP_TRANSPOSE || cgraph->nodes[j]->op == GGML_OP_NONE)) {
            ++j;
        }
        if (!norm_pair_off && j + 1 < cgraph->n_nodes && ggml_cuda_can_fuse(cgraph, j, { GGML_OP_RMS_NORM, GGML_OP_SCALE }, {})) {
            const ggml_tensor * n1 = cgraph->nodes[j];
            const ggml_tensor * x1 = n1->src[0];
            bool same = x0->type == GGML_TYPE_F32 && x1->type == GGML_TYPE_F32 && x0->view_src && x0->view_src == x1->view_src &&
                        x0->ne[0] < 1024 && memcmp(node->op_params, n1->op_params, sizeof(float)) == 0 &&
                        memcmp(cgraph->nodes[i + 1]->op_params, cgraph->nodes[j + 1]->op_params, 2 * sizeof(float)) == 0;
            for (int k = 0; same && k < 4; ++k) {
                same = x0->ne[k] == x1->ne[k] && x0->nb[k] == x1->nb[k];
            }
            // the second pair must not read the first one's output (it reads a view of the shared parent)
            same = same && x1 != cgraph->nodes[i + 1] && x1->view_src != cgraph->nodes[i + 1];
            if (same) {
                ggml_cuda_op_rms_norm_scale_pair_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[j], cgraph->nodes[j + 1]);
                return j + 1 - i;
            }
        }
        ggml_cuda_op_rms_norm_scale_fused(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    // Fuse the SSM pre-scan chain (owned by mmvq.cu). GGML_CUDA_DISABLE_SSM_PRESCAN=1 turns it off.
    if (const int n_ssm_prescan = ggml_cuda_ssm_prescan_try_fuse(*cuda_ctx, cgraph, i)) {
        return n_ssm_prescan;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_ADD, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        ggml_cuda_op_ssm_conv(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        ggml_cuda_op_ssm_conv(*cuda_ctx, node, /*bias_add_node=*/ nullptr, cgraph->nodes[i + 1]);
        return 1;
    }

    // prefill hyper-connection fusions (DEFAULT ON). Bit-exact vs the unfused chain, verified on the
    // deterministic substrate with a llama_decode harness (pp + 40 greedy steps, logits hashed per
    // step: fused == unfused bit for bit at 6..1400-token prompts). FIXED 2026-09-06: the comb
    // matcher unwrapped the block_out REPEAT to its input, whose buffer the allocator had reused
    // (e.g. for hc_inject) because the standalone REPEAT ran before the fused window - the kernel
    // then read inject's data as block_out. It now reads the live REPEAT output (mul operand).
    // Opt-outs: GGML_CUDA_DISABLE_HC_FUSION=1 (all), GGML_CUDA_DISABLE_HC_MIX=1 / GGML_CUDA_DISABLE_HC_COMB=1.

    // qwen4exp hyper-connection gate GEMM + sigmoid + stream mix: when the HC gate MUL_MAT's only
    // consumer is the sigmoid that opens an hc_mix window, fold the GEMM into the mix kernel - it
    // removes the gate tensor materialization and one launch per layer.  RDNA3 (gfx1100/gfx1151) and,
    // since 2026-09-24, RDNA4 (gfx120x) - the kernel dequantizes the IQ4_NL gate weight and replays
    // the GEMM's BF16 epilogue rounding via the arch-aware `mmb_frag_t`/`mmb_wmma_bf16`/`MMB_ACC_M`
    // shim, so the fusion is bit-identical to the GEMM + sigmoid + mix chain on both.  A/B / bisect:
    // LLAMA_HC_GATEMIX=0.
    // gfx1100 is a port candidate: the gfx11 WMMA builtin is shared with gfx1151, but `gatemix` stays
    // default OFF on RDNA3_0 (mmb_arch_defaults) until a gfx1151/gfx1201 session A/Bs qwen4exp
    // end-to-end; LLAMA_HC_GATEMIX=1 is the opt-in on gfx1100.
    // qwen4exp HC gate GEMM + sigmoid + stream mix (owned by hyperconn.cu). LLAMA_HC_GATEMIX=0 off.
    if (const int n_hc_gate_mix = ggml_cuda_hc_gate_mix_try_fuse(*cuda_ctx, cgraph, i)) {
        return n_hc_gate_mix;
    }
    // qwen4exp hyper-connection stream mix (HC_MIX, prefill): one kernel reads xn/gate once and
    // writes the [n_embd, T] mean of the gated streams.  (hc-mix surface, owner block 10.)
    if (const int n_hc_mix = ggml_cuda_hc_mix_try_fuse(*cuda_ctx, cgraph, i)) {
        return n_hc_mix;
    }
    // qwen4exp hyper-connection combine+norm fusions (owned by hyperconn.cu).
    if (const int n_hc_comb = ggml_cuda_hc_combine_norm_try_fuse(*cuda_ctx, cgraph, i)) {
        return n_hc_comb;
    }

    // UNARY(sigmoid) -> MUL -> ADD with a per-row gate (GGML_CUDA_FUSE_SIGMOID_MUL_ADD=0: off): one kernel
    if (const int n_sig_mul_add = ggml_cuda_sigmoid_mul_add_try_fuse(*cuda_ctx, cgraph, i)) {
        return n_sig_mul_add;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SILU }) ||
        ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SIGMOID }) ||
        ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SOFTPLUS })) {
        ggml_tensor * mul_node = cgraph->nodes[i + 1];

        // If the product feeds a single decode matmul (directly or via a
        // no-op reshape), write its Q8_1 quantized value instead of the F32
        // output; the matmul launcher finds it via the quantize cache.
        const ggml_tensor * mm = ggml_cuda_find_mul_q8_1_matmul(cgraph, i + 1, mul_node);
        if (mm != nullptr) {
            ggml_cuda_op_unary_mul_q8_1(*cuda_ctx, node, mul_node, mm);
            return cgraph->nodes[i + 2] == mm ? 1 : 2;
        }

        ggml_cuda_op_unary_mul(*cuda_ctx, node, mul_node);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_SQR }, { GGML_UNARY_OP_RELU })) {
        ggml_cuda_op_relu_sqr(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE }, { GGML_UNARY_OP_TANH })) {
        ggml_cuda_op_softcap(*cuda_ctx, cgraph->nodes[i + 2], node);
        return 2;
    }

    // scale -> silu/sigmoid unary (owned by unary.cu). Opt-out: GGML_CUDA_SCALE_UNARY=0.
    if (const int n_scale_unary = ggml_cuda_scale_unary_try_fuse(*cuda_ctx, cgraph, i)) {
        return n_scale_unary;
    }

    return 0;
}
