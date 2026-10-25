// Fold (r96, PR #189): std::find in llama_state_sync_all.
#include <algorithm>

#include "llama-context.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "llama-arch.h"
#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-memory.h"
#include "llama-kv-cache.h"
#include "llama-memory-hybrid.h"
#include "llama-kv-cache-iswa.h"
#include "llama-mmap.h"
#include "llama-model.h"
#include "llama-moe-cache.h"
#include "llama-ext.h"
#include "llama-sampler.h"
#include "llama.h"

#include <cinttypes>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>

#if defined(__SSE2__) || defined(_M_X64)
#define LLAMA_STATE_COPY_STREAM 1
#include <emmintrin.h>
#endif

//
// llama_context
//

// a batch of several sequences the derived kq mask cannot serve (see llama_kv_cache::kq_mask_derivable_seqs); a batch of
// at most KQ_DERIVED_MULTISEQ_MIN_TOKENS tokens has only ubatches that keep the packed mask, which the reserve covers
static bool batch_needs_packed_multiseq(const llama_batch_ext & batch, const std::vector<llama_kv_cache *> & kvs) {
    if (kvs.empty() || batch.tokens.size() <= llama_kv_cache::KQ_DERIVED_MULTISEQ_MIN_TOKENS) {
        return false;
    }

    // fixed-size, no allocation: this runs on every decode() batch until a latch fires
    llama_seq_id seqs[LLAMA_MAX_SEQ];
    bool seen[LLAMA_MAX_SEQ] = {};
    uint32_t n_seqs = 0;
    llama_pos pos_max = 0;

    for (const auto & tok : batch.tokens) {
        if (tok.seq_ids.size() > 1) {
            return true;
        }
        for (const llama_seq_id s : tok.seq_ids) {
            if (s < 0 || s >= LLAMA_MAX_SEQ) {
                return true;
            }
            if (!seen[s]) {
                seen[s] = true;
                seqs[n_seqs++] = s;
            }
        }
        pos_max = std::max(pos_max, tok.pos[0]);
    }

    if (n_seqs <= 1) {
        return false;
    }
    for (const auto * kv : kvs) {
        if (!kv->kq_mask_derivable_seqs(seqs, n_seqs, pos_max)) {
            return true;
        }
    }
    return false;
}

// issue #238: does this batch carry a non-degenerate 2-D (M-RoPE) position?  Only an embedding entry (an
// image/audio chunk, i.e. no token id) can: the batch allocator expands a text entry to [p, p, p, 0], so a
// text token is degenerate by construction.  This mirrors what llama_ubatch::is_pos_2d() + the check in
// llama_kv_cache::kq_mask_derivable() will decide for the resulting ubatch - an embedding whose sections
// differ, or whose later sections were left at 0, needs the packed mask.
static bool batch_has_pos_2d(const llama_batch_ext & batch) {
    for (const auto & tok : batch.tokens) {
        if (tok.id != LLAMA_TOKEN_NULL) {
            continue;
        }
        if (tok.pos[0] != tok.pos[1] || tok.pos[0] != tok.pos[2]) {
            return true;
        }
    }
    return false;
}

// check that each folded weight in the graph gets its Hadamard transform, and each latent lookup gets the inverse
// without this check, an arch that skips the transform helpers loads and computes wrong results
static void llama_verify_hadamard_graph(
        ggml_cgraph * gf,
        const llama_hadamard_rotations & rotations,
        const llama_hadamard_rotations & inverses,
        const llama_moe_cache * moe_cache) {
    // the MoE cache gives the matmul a copy of the expert weights, so the copy also needs the transform of its source
    llama_hadamard_rotations forward = rotations;
    for (const auto & [w, t] : rotations) {
        if (const ggml_tensor * cached = moe_cache ? moe_cache->get_experts(w) : nullptr) {
            forward.emplace(cached, t);
        }
    }

    auto unwrap = [](const ggml_tensor * t) {
        while (t && (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW)) {
            t = t->src[0];
        }
        return t;
    };

    std::map<const ggml_tensor *, bool> lookups; // get_rows results of latent tables

    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        const ggml_tensor * node = ggml_graph_node(gf, i);

        if (node->op == GGML_OP_GET_ROWS && inverses.count(node->src[0])) {
            lookups.emplace(node, false);
            continue;
        }

        if (node->op != GGML_OP_MUL_MAT && node->op != GGML_OP_MUL_MAT_ID) {
            continue;
        }

        if (node->op == GGML_OP_MUL_MAT && ((const int32_t *) node->op_params)[1] == GGML_HINT_SRC0_IS_HADAMARD) {
            const auto lk = lookups.find(unwrap(node->src[1]));
            if (lk != lookups.end()) {
                lk->second = true;
            }
            continue;
        }

        const auto it = forward.find(node->src[0]);
        if (it == forward.end()) {
            if (inverses.count(node->src[0])) {
                throw std::runtime_error(format(
                    "Hadamard-latent table '%s' is used as a head without a forward transform", node->src[0]->name));
            }
            continue;
        }
        const ggml_tensor * src = unwrap(node->src[1]);
        const bool transformed = src && src->op == GGML_OP_MUL_MAT &&
            ((const int32_t *) src->op_params)[1] == GGML_HINT_SRC0_IS_HADAMARD &&
            src->src[0] == it->second.rot;
        if (!transformed) {
            throw std::runtime_error(format(
                "Hadamard-folded weight '%s' is consumed without its activation transform; "
                "this graph's matmul path does not support prism.hadamard folding",
                node->src[0]->name));
        }
    }

    for (const auto & [node, ok] : lookups) {
        if (!ok) {
            for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
                const ggml_tensor * n2 = ggml_graph_node(gf, i);
                for (int s = 0; s < GGML_MAX_SRC && n2->src[s]; ++s) {
                    if (unwrap(n2->src[s]) == node) {
                        LLAMA_LOG_WARN("%s: latent lookup '%s' consumed by op=%s name='%s' src%d hint=%d\n",
                                __func__, node->name, ggml_op_name(n2->op), n2->name, s,
                                ((const int32_t *) n2->op_params)[1]);
                    }
                }
            }
            throw std::runtime_error(format(
                "Hadamard-latent table '%s' is read without the inverse transform",
                node->src[0]->name));
        }
    }
}

static llm_graph_type ctx_type_to_graph_type(llama_context_type ctx_type) {
    switch (ctx_type) {
        case LLAMA_CONTEXT_TYPE_DEFAULT: return LLM_GRAPH_TYPE_DEFAULT;
        case LLAMA_CONTEXT_TYPE_MTP    : return LLM_GRAPH_TYPE_DECODER_MTP;
    }
    throw std::runtime_error("Unsupported ctx type");
}

struct llm_fused_op_probe {
    llm_fused_op op;
    const char * name;
    uint32_t n_tokens_per_seq;

    // number of sequences to reserve the probe graph with (0 = the context's n_seq_max)
    uint32_t n_seqs = 0;

    // the fused node has to land on a GPU device (used by features that only make sense on a GPU)
    bool require_gpu = false;

    // ... and specifically on a backend that implements the derived kq mask (CUDA/HIP)
    bool require_kq_derived = false;

    // the probe graph has to actually contain the fused op - otherwise nothing was verified
    bool require_observed = false;
};

static const llm_fused_op_probe llm_fused_op_flash_attn_probe = {
    /*.op               =*/ LLM_FUSED_OP_FLASH_ATTN,
    /*.name             =*/ "Flash Attention",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_gdn_ar_probe = {
    /*.op               =*/ LLM_FUSED_OP_GDN_AR,
    /*.name             =*/ "fused Gated Delta Net (autoregressive)",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_gdn_ch_probe = {
    /*.op               =*/ LLM_FUSED_OP_GDN_CH,
    /*.name             =*/ "fused Gated Delta Net (chunked)",
    /*.n_tokens_per_seq =*/ 16,
};

static const llm_fused_op_probe llm_fused_op_lid_probe = {
    /*.op               =*/ LLM_FUSED_OP_LIGHTNING_INDEXER,
    /*.name             =*/ "Lightning Indexer",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_pre_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_PRE,
    /*.name             =*/ "fused DeepSeek V4 HC pre",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_comb_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_COMB,
    /*.name             =*/ "fused DeepSeek V4 HC comb",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_post_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_POST,
    /*.name             =*/ "fused DeepSeek V4 HC post",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_top_k_mask_probe = {
    /*.op               =*/ LLM_FUSED_OP_TOP_K_MASK,
    /*.name             =*/ "fused top-k mask",
    /*.n_tokens_per_seq =*/ 1,
};

// V3: the derived kq mask only changes the flash attention prefill path (the packed mask stays for
// decode and small batches), so the probe is a prefill-shaped single sequence
static const llm_fused_op_probe llm_fused_op_kq_derived_probe = {
    /*.op               =*/ LLM_FUSED_OP_FLASH_ATTN_DERIVED,
    /*.name             =*/ "derived kq mask flash attention",
    /*.n_tokens_per_seq =*/ 64,
    /*.n_seqs           =*/ 1,
    /*.require_gpu      =*/ true,
    /*.require_kq_derived =*/ true,
    /*.require_observed =*/ true,
};

// the derived kq mask is implemented by the CUDA/HIP backend only - any other backend that accepts
// the flash attention op would silently ignore the derived sources.  An iGPU (APU) is still the
// CUDA/HIP backend, so it implements the derived path too (the MMA kernel is the same one).
static bool ggml_backend_dev_is_cuda(ggml_backend_dev_t dev) {
    if (dev == nullptr) {
        return false;
    }

    const auto type = ggml_backend_dev_type(dev);
    if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
        return false;
    }

    const char * name = ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));

    return name != nullptr && (strcmp(name, "CUDA") == 0 || strcmp(name, "ROCm") == 0);
}

// Only a tensor split distributes a single attention op across devices; tell the HIP FA chooser
// so it can pick a per-GPU-bandwidth-friendly kernel config instead of the whole-card one.
static void llama_context_set_fa_tensor_parallel_hint(const llama_model & model) {
    int n_cuda_dev = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        if (ggml_backend_dev_is_cuda(ggml_backend_dev_get(i))) {
            n_cuda_dev++;
        }
    }
    ggml_set_fa_tensor_parallel(llm_split_mode_is_tensor_like(model.split_mode()) && n_cuda_dev > 1);
}

// true when this build exposes the CUDA/HIP backend at all
static bool ggml_backend_cuda_family_available() {
    for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
        const char * name = ggml_backend_reg_name(ggml_backend_reg_get(i));

        if (name != nullptr && (strcmp(name, "CUDA") == 0 || strcmp(name, "ROCm") == 0)) {
            return true;
        }
    }

    return false;
}

// the derived kq mask is only implemented by the CUDA/HIP backend.  A "meta" device (the
// tensor-parallel wrapper used for multi-GPU tensor split) does not expose its sub-devices here, but
// its supports_op forwards to all of them - which is exactly how the probed node ended up on it - so
// accepting it is only safe while this build has the CUDA/HIP backend that the check above enforces.
static bool ggml_backend_dev_implements_kq_derived(ggml_backend_dev_t dev) {
    if (dev == nullptr) {
        return false;
    }

    switch (ggml_backend_dev_type(dev)) {
        case GGML_BACKEND_DEVICE_TYPE_GPU:
        case GGML_BACKEND_DEVICE_TYPE_IGPU:
            return ggml_backend_dev_is_cuda(dev);
        case GGML_BACKEND_DEVICE_TYPE_META:
            return ggml_backend_cuda_family_available();
        default:
            return false;
    }
}

llama_context::llama_context(
        const llama_model & model,
              llama_context_params params) :
    model(model),
    cvec(std::make_unique<llama_adapter_cvec>()),
    loras(std::make_unique<llama_adapter_loras>()),
    // MTP uses the embd input for the hidden state
    balloc(std::make_unique<llama_batch_allocr>(model.hparams.n_pos_per_embd(),
                llm_arch_supports_mixed_batch(model.arch) && params.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT)) {
    // TODO warning when creating llama_context with awkward ctx size that is not a power of 2,
    //     may need to be backend-dependent
    LLAMA_LOG_INFO("%s: constructing llama_context\n", __func__);

    t_start_us = model.t_start_us;
    t_load_us  = model.t_load_us;

    const auto & hparams = model.hparams;

    llama_context_set_fa_tensor_parallel_hint(model);

    cparams.n_seq_max = std::max(1u, params.n_seq_max);
    if (cparams.n_seq_max > LLAMA_MAX_SEQ) {
        throw std::runtime_error("n_seq_max must be <= " + std::to_string(LLAMA_MAX_SEQ));
    }

    cparams.n_rs_seq = params.n_rs_seq;
    // TEMP diagnostic (wip/moe-verify-fusions): force the target's snapshot depth independently of the
    // spec config, to A/B the n_rs_seq=0 (plain) vs n_rs_seq>0 (MTP) GDN/conv path with the cache armed.
    if (const char * e = getenv("GGML_FORCE_N_RS_SEQ")) {
        cparams.n_rs_seq = (uint32_t) atoi(e);
    }
    if (cparams.n_rs_seq > 0 && !llm_arch_supports_rs_rollback(model.arch)) {
        LLAMA_LOG_DEBUG("%s: n_rs_seq=%u requested but model does not support recurrent partial rollback; clamping to 0\n",
                        __func__, cparams.n_rs_seq);
        cparams.n_rs_seq = 0;
    }

    // a rollback candidate decodes the sampled token plus a full draft, which can be longer than
    // the snapshot depth; never allow the bound to drop below the snapshot depth itself
    cparams.n_rs_batch = std::max(cparams.n_rs_seq + 1, params.n_rs_batch);

    cparams.n_threads               = params.n_threads;
    cparams.n_threads_batch         = params.n_threads_batch;
    cparams.yarn_ext_factor         = params.yarn_ext_factor  >= 0.0f ? params.yarn_ext_factor  : hparams.yarn_ext_factor;
    cparams.yarn_attn_factor        = params.yarn_attn_factor >= 0.0f ? params.yarn_attn_factor : hparams.yarn_attn_factor;
    cparams.yarn_beta_fast          = params.yarn_beta_fast   >= 0.0f ? params.yarn_beta_fast   : hparams.yarn_beta_fast;
    cparams.yarn_beta_slow          = params.yarn_beta_slow   >= 0.0f ? params.yarn_beta_slow   : hparams.yarn_beta_slow;
    cparams.embeddings              = params.embeddings;
    cparams.embeddings_nextn        = false;
    cparams.embeddings_nextn_masked = false;
    cparams.offload_kqv             = params.offload_kqv;
    cparams.no_perf                 = params.no_perf;
    cparams.warmup                  = false;

    // +1: id n_layer() taps the output of the last layer ("input" of the head)
    cparams.embeddings_layer_inp.resize(hparams.n_layer() + 1, false);
    embd_layer_inp.resize(hparams.n_layer() + 1);
    for (const int32_t lid : model.embeddings_layer_inp_preset) {
        if (lid >= 0 && (size_t) lid < cparams.embeddings_layer_inp.size()) {
            cparams.embeddings_layer_inp[lid] = true;
        }
    }

    cparams.ctx_type          = params.ctx_type;
    cparams.rope_scaling_type = params.rope_scaling_type;
    cparams.pooling_type      = params.pooling_type;

    cparams.n_ctx            = params.n_ctx           == 0    ? hparams.n_ctx_train           : params.n_ctx;
    cparams.rope_freq_base   = params.rope_freq_base  == 0.0f ? hparams.rope_freq_base_train  : params.rope_freq_base;
    cparams.rope_freq_scale  = params.rope_freq_scale == 0.0f ? hparams.rope_freq_scale_train : params.rope_freq_scale;

    cparams.n_ctx_orig_yarn  = params.yarn_orig_ctx    != 0 ? params.yarn_orig_ctx    :
                               hparams.n_ctx_orig_yarn != 0 ? hparams.n_ctx_orig_yarn :
                                                              hparams.n_ctx_train;

    cparams.cb_eval           = params.cb_eval;
    cparams.cb_eval_user_data = params.cb_eval_user_data;

    cparams.ctx_other = nullptr;

    // TODO: more generic
    if (model.arch == LLM_ARCH_GEMMA4_ASSISTANT) {
        if (params.ctx_other == nullptr) {
            // TODO: change from runtime_error to llama_exception to avoid printing error message
            throw std::runtime_error("Gemma4Assistant requires ctx_other to be set (this warning is normal during memory fitting)");
        }

        cparams.ctx_other = params.ctx_other;
    }

    if (model.arch == LLM_ARCH_EAGLE3 || model.arch == LLM_ARCH_DFLASH || model.arch == LLM_ARCH_QWEN4EXP) {
        if (model.tok_embd == nullptr || model.output == nullptr) {
            if (params.ctx_other == nullptr) {
                throw std::runtime_error(model.arch_name() + " requires ctx_other to be set (this warning is normal during memory fitting)");
            }
            cparams.ctx_other = params.ctx_other;
        }
    }

    if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED) {
        cparams.rope_scaling_type = hparams.rope_scaling_type_train;
    }

    if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_NONE) {
        cparams.rope_freq_scale = 1.0f; // never scale if scaling type is none
    }

    if (cparams.yarn_ext_factor < 0.0f) { // negative indicates 'not set'
        cparams.yarn_ext_factor = cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN ? 1.0f : 0.0f;
    }

    if (cparams.yarn_ext_factor != 0) {
        static auto get_mscale = [](float scale, float mscale) {
            return scale <= 1.0f ? 1.0f : (0.1f * mscale * logf(scale) + 1.0f);
        };

        const float factor = 1.0f / cparams.rope_freq_scale;

        // ref: https://github.com/huggingface/transformers/blob/6d00f6b0a5679c36510f203e4226e36f517c3032/src/transformers/modeling_rope_utils.py#L336-L348
        if (hparams.rope_yarn_log_mul != 0.0f) {
            // note: here we assume `mscale == 1.0f`
            // TODO: start reading the actual value of mscale and handle the case where it is not 1.0f
                  float mscale          = 1.0f;
            const float mscale_all_dims = hparams.rope_yarn_log_mul;

            // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
            // special-case DEEPSEEK v2:
            // https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/blob/main/config.json#L42-L43
            if (model.arch == LLM_ARCH_DEEPSEEK2 && mscale_all_dims != 1.0f) {
                mscale = mscale_all_dims;
            }

            cparams.yarn_attn_factor = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dims);

            LLAMA_LOG_WARN("%s: setting new yarn_attn_factor = %.4f (mscale == %.1f, mscale_all_dim = %.1f)\n",
                    __func__, cparams.yarn_attn_factor, mscale, mscale_all_dims);
        } else {
            cparams.yarn_attn_factor = get_mscale(factor, 1.0f);
        }

        // when YARN is applied with yarn_ext_factor != 0.0f, we need to cancel this factor:
        // https://github.com/ggml-org/llama.cpp/blob/a81a569577cc38b32558958b048228150be63eae/ggml/src/ggml-cpu/ops.cpp#L5541-L5544
        //
        // ref: https://github.com/ggml-org/llama.cpp/discussions/7416
        //      https://github.com/ggml-org/llama.cpp/pull/17945
        cparams.yarn_attn_factor *= 1.0f / (1.0f + 0.1f * logf(factor));
    }

    cparams.yarn_attn_factor *= hparams.rope_attn_factor;

    if (cparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
        if (hparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
            cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
        } else {
            cparams.pooling_type = hparams.pooling_type;
        }
    }

    if (params.attention_type == LLAMA_ATTENTION_TYPE_UNSPECIFIED) {
        cparams.causal_attn = hparams.causal_attn;
    } else {
        cparams.causal_attn = params.attention_type == LLAMA_ATTENTION_TYPE_CAUSAL;
    }

    cparams.flash_attn = params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.auto_fa    = params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO;

    cparams.kq_mask_derived = true;
    cparams.auto_kq_mask_derived = true;
    {
        const char * LLAMA_KQ_MASK_DERIVED = getenv("LLAMA_KQ_MASK_DERIVED");
        if (LLAMA_KQ_MASK_DERIVED) {
            // only an explicit 0 forces the packed mask - the probe below stays in charge of whether
            // the backend can actually run the derived form
            cparams.kq_mask_derived = atoi(LLAMA_KQ_MASK_DERIVED) != 0;
            cparams.auto_kq_mask_derived = cparams.kq_mask_derived;
            LLAMA_LOG_INFO("%s: derived kq mask = %d (env)\n", __func__, cparams.kq_mask_derived);
        }
    }

    // issue #238: the packed kq mask costs 2 bytes per KV cell per ubatch token (860 MiB of VRAM plus as
    // much pinned host memory at -c 220160 -ub 2048) and only a batch the derived form cannot serve needs
    // it.  For an M-RoPE model (Qwen3.x is IMROPE even for text) that is a batch carrying non-degenerate
    // 2-D positions, i.e. an image/audio chunk from a multimodal projector.  Two signals decide it:
    //   * `params.pos_2d` - the caller declares that 2-D positions can reach this context (`common` sets
    //     it when --mmproj is given), so a multimodal run keeps the worst-case reserve at load time and
    //     never has to grow later;
    //   * `kq_mask_pos_2d_seen` - latched in decode() when a 2-D batch actually arrives from a caller that
    //     did not declare it, which forces one re-reserve with the packed mask.
    // Without either, the much smaller derived layout is reserved: a text-only M-RoPE context never pays.
    // LLAMA_KQ_MASK_PACKED_RESERVE=1 forces the packed reserve for every M-RoPE context (the pre-r160
    // behaviour); =0 forces the derived one and disables the latch (the growth path is the fallback).
    cparams.kq_mask_pos_2d        = params.pos_2d;
    cparams.kq_mask_packed_reserve = -1;
    {
        const char * e = getenv("LLAMA_KQ_MASK_PACKED_RESERVE");
        if (e != nullptr) {
            cparams.kq_mask_packed_reserve = atoi(e) != 0 ? 1 : 0;
            LLAMA_LOG_INFO("%s: packed kq mask reserve = %d (env)\n", __func__, cparams.kq_mask_packed_reserve);
        } else if (cparams.kq_mask_pos_2d && hparams.n_pos_per_embd() > 1) {
            LLAMA_LOG_INFO("%s: packed kq mask reserve = 1 (2-D positions declared, e.g. a multimodal projector)\n",
                    __func__);
        }
    }

    cparams.fused_gdn_ar = true;
    cparams.fused_gdn_ch = true;
    cparams.auto_fgdn    = false;

    cparams.fused_lid = true;
    cparams.auto_flid = false;

    cparams.fused_dsv4_hc_pre  = true;
    cparams.fused_dsv4_hc_comb = true;
    // The hc_combine_norm chain spelled out (the plain scale->sigmoid->scale->repeat->mul->add->rms->mul
    // form of DSV4_HC_POST) is faster on ONE device: measured +0.9..+1.8 % prefill on qwen4exp IQ4_NL,
    // gfx1151, which is why it is the default there.  It is NOT correct under the Meta (tensor-split)
    // device -- the chain's broadcasts are split-state sensitive, and `test-llama-archs -a deepseek4`
    // on 3x gfx1201 read NMSE 1.74e-01 vs the CPU reference (-sm tensor) while the DSV4_HC_POST op is
    // rounding-level on one device, on -sm layer and (with this default) on -sm tensor (6.5e-06, the
    // same as the op arm has always measured there).  So the op is the default whenever a Meta device
    // is in play; `LLAMA_FUSED_DSV4_HC_POST` still forces either arm for A/B.
    bool meta_split_active = false;
    for (const auto & dev : model.devices) {
        meta_split_active = meta_split_active || dev.is_meta;
    }
    cparams.fused_dsv4_hc_post = meta_split_active;
    cparams.auto_fhc           = true;
    {
        // gap-closing WIP: LLAMA_FUSED_DSV4_HC_PRE/_POST=1 force the DSV4 HC ops (the env only ever
        // re-enables the slower op path, for A/B and bisection).
        const char * env_pre = getenv("LLAMA_FUSED_DSV4_HC_PRE");
        if (env_pre) {
            cparams.fused_dsv4_hc_pre = atoi(env_pre) != 0;
            LLAMA_LOG_INFO("%s: fused dsv4_hc_pre = %d (env)\n", __func__, cparams.fused_dsv4_hc_pre);
        }
        const char * env_post = getenv("LLAMA_FUSED_DSV4_HC_POST");
        if (env_post) {
            cparams.fused_dsv4_hc_post = atoi(env_post) != 0;
            LLAMA_LOG_INFO("%s: fused dsv4_hc_post = %d (env)\n", __func__, cparams.fused_dsv4_hc_post);
        }
    }

    cparams.fused_hc_mix = true;
    {
        const char * LLAMA_FUSED_HC_MIX = getenv("LLAMA_FUSED_HC_MIX");
        if (LLAMA_FUSED_HC_MIX) {
            cparams.fused_hc_mix = atoi(LLAMA_FUSED_HC_MIX) != 0;
            LLAMA_LOG_INFO("%s: fused hc_mix = %d (env)\n", __func__, cparams.fused_hc_mix);
        }
    }

    cparams.fused_hc_combine = true;
    {
        const char * LLAMA_FUSED_HC_COMBINE = getenv("LLAMA_FUSED_HC_COMBINE");
        if (LLAMA_FUSED_HC_COMBINE) {
            cparams.fused_hc_combine = atoi(LLAMA_FUSED_HC_COMBINE) != 0;
            LLAMA_LOG_INFO("%s: fused hc_combine = %d (env)\n", __func__, cparams.fused_hc_combine);
        }
    }

    // only the indexer models build this mask, skip the probe for the others
    cparams.fused_top_k_mask = true;
    cparams.auto_ftkm        = hparams.indexer_top_k > 0;

    // with causal attention, the batch size is limited by the context size
    cparams.n_batch = cparams.causal_attn ? std::min(cparams.n_ctx, params.n_batch) : params.n_batch;

    cparams.n_ubatch = std::min(cparams.n_batch, params.n_ubatch == 0 ? params.n_batch : params.n_ubatch);

    cparams.n_outputs_max = params.n_outputs_max == 0 || llama_model_has_encoder(&model) ? cparams.n_batch : params.n_outputs_max;
    cparams.n_outputs_max_per_seq = params.n_outputs_max_per_seq == 0 ?
            cparams.n_outputs_max : std::min(params.n_outputs_max_per_seq, cparams.n_outputs_max);

    // Initialize backend samplers here so they are part of the sampling graph
    // before the reserve passes run later in this function. This avoids a later
    // re-reserve when graph nodes change.
    if (params.samplers != nullptr && params.n_samplers > 0) {
        for (size_t i = 0; i < params.n_samplers; ++i) {
            const auto & config = params.samplers[i];

            if (llama_sampler_chain_get(config.sampler, -1) == nullptr) {
                throw std::runtime_error("the backend samplers must be of type llama_sampler_chain");
            }

            if (set_sampler(config.seq_id, config.sampler)) {
                const int n_samplers = llama_sampler_chain_n(config.sampler);

                LLAMA_LOG_INFO("%s: setting backend sampler for seq_id %d (n = %d)\n", __func__, config.seq_id, n_samplers);
            }
        }
    }

    cparams.op_offload           = params.op_offload;
    cparams.kv_unified           = params.kv_unified;
    cparams.moe_cache_size       = params.moe_cache_size; // upstream MoE cache (dormant unless --moe-cache-mib)
    cparams.drop_compute_buffers = params.drop_compute_buffers;

    // initialized later
    cparams.pipeline_parallel = false;
    cparams.training = false;

    {
        const char * LLAMA_GRAPH_REUSE_DISABLE = getenv("LLAMA_GRAPH_REUSE_DISABLE");
        graph_reuse_disable = LLAMA_GRAPH_REUSE_DISABLE ? (atoi(LLAMA_GRAPH_REUSE_DISABLE) != 0) : graph_reuse_disable;

        if (graph_reuse_disable) {
            LLAMA_LOG_WARN("%s: graph reuse disabled\n", __func__);
        }
    }

    // ref: https://github.com/ggml-org/llama.cpp/pull/17046#discussion_r2503085732
    cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256);

    if (cparams.kv_unified) {
        cparams.n_ctx_seq = cparams.n_ctx;
    } else {
        cparams.n_ctx_seq = cparams.n_ctx / cparams.n_seq_max;
        cparams.n_ctx_seq = GGML_PAD(cparams.n_ctx_seq, 256);

        if (cparams.n_ctx_seq == 0) {
            throw std::runtime_error("n_ctx_seq == 0");
        }

        if (cparams.n_ctx != cparams.n_ctx_seq * cparams.n_seq_max) {
            cparams.n_ctx =  cparams.n_ctx_seq * cparams.n_seq_max;
            LLAMA_LOG_WARN("%s: n_ctx is not divisible by n_seq_max - rounding down to %u\n", __func__, cparams.n_ctx);
        }
    }

    LLAMA_LOG_INFO("%s: n_seq_max             = %u\n",   __func__, cparams.n_seq_max);
    LLAMA_LOG_INFO("%s: n_ctx                 = %u\n",   __func__, cparams.n_ctx);
    LLAMA_LOG_INFO("%s: n_ctx_seq             = %u\n",   __func__, cparams.n_ctx_seq);
    LLAMA_LOG_INFO("%s: n_batch               = %u\n",   __func__, cparams.n_batch);
    LLAMA_LOG_INFO("%s: n_ubatch              = %u\n",   __func__, cparams.n_ubatch);
    LLAMA_LOG_INFO("%s: causal_attn           = %d\n",   __func__, cparams.causal_attn);
    LLAMA_LOG_INFO("%s: flash_attn            = %s\n",   __func__, llama_flash_attn_type_name(params.flash_attn_type));
    LLAMA_LOG_INFO("%s: kv_unified            = %s\n",   __func__, cparams.kv_unified ? "true" : "false");
    LLAMA_LOG_INFO("%s: freq_base             = %.1f\n", __func__, cparams.rope_freq_base);
    LLAMA_LOG_INFO("%s: freq_scale            = %g\n",   __func__, cparams.rope_freq_scale);
    LLAMA_LOG_INFO("%s: n_rs_seq              = %u\n",   __func__, cparams.n_rs_seq);
    LLAMA_LOG_INFO("%s: n_rs_batch            = %u\n",   __func__, cparams.n_rs_batch);
    LLAMA_LOG_INFO("%s: n_outputs_max         = %u\n",   __func__, cparams.n_outputs_max);
    LLAMA_LOG_INFO("%s: n_outputs_max_per_seq = %u\n",   __func__, cparams.n_outputs_max_per_seq);

    if (cparams.n_ctx_seq < hparams.n_ctx_train) {
        LLAMA_LOG_INFO("%s: n_ctx_seq (%u) < n_ctx_train (%u) -- the full capacity of the model will not be utilized\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (cparams.n_ctx_seq > hparams.n_ctx_train) {
        LLAMA_LOG_WARN("%s: n_ctx_seq (%u) > n_ctx_train (%u) -- possible training context overflow\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (!hparams.vocab_only) {
        // GPU backends
        for (const auto & dev : model.devices) {
            ggml_backend_t backend = ggml_backend_dev_init(dev.dev, nullptr);
            if (backend == nullptr) {
                throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev.dev)));
            }
            backends.emplace_back(backend);
        }

        // add ACCEL backends (such as BLAS)
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
                if (backend == nullptr) {
                    throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev)));
                }
                backends.emplace_back(backend);
            }
        }

        // add CPU backend
        backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (backend_cpu == nullptr) {
            throw std::runtime_error("failed to initialize CPU backend");
        }
        backends.emplace_back(backend_cpu);

        // create a list of the set_n_threads functions in the backends
        for (auto & backend : backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            if (reg) {
                auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
                if (ggml_backend_set_n_threads_fn) {
                    set_n_threads_fns.emplace_back(backend.get(), ggml_backend_set_n_threads_fn);
                }
            }
        }

        llama_set_abort_callback(this, params.abort_callback, params.abort_callback_data);

        // graph outputs buffer
        {
            if (output_reserve(params.n_seq_max) < params.n_seq_max) {
                throw std::runtime_error("failed to reserve initial output buffer");
            }

            LLAMA_LOG_INFO("%s: %10s  output buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buffer_name    (buf_output.get()),
                    ggml_backend_buffer_get_size(buf_output.get()) / 1024.0 / 1024.0);
        }
    }

    // init the memory module
    if (!hparams.vocab_only) {
        llama_memory_params params_mem = {
            /*.type_k    =*/ params.type_k,
            /*.type_v    =*/ params.type_v,
            /*.swa_full  =*/ params.swa_full,
            /*.ctx_type  =*/ cparams.ctx_type,
            /*.mem_other =*/ llama_get_memory(cparams.ctx_other),
        };
        // the graph builders that choose between quantized-aware attention arms need the same
        // types the memory module was given (see qwen4exp_qsa_sparse)
        cparams.type_k = params_mem.type_k;
        cparams.type_v = params_mem.type_v;

        memory.reset(model.create_memory(params_mem, cparams));
    }

    // init backends
    if (!hparams.vocab_only) {
        LLAMA_LOG_DEBUG("%s: enumerating backends\n", __func__);

        backend_buft.clear();
        backend_ptrs.clear();
        backend_buf_exp_size.clear();

        for (auto & backend : backends) {
            auto * buft = ggml_backend_get_default_buffer_type(backend.get());
            auto backend_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));

            if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU && !model.devices.empty()) {
                // use the host buffer of the first device CPU for faster transfer of the intermediate state
                const auto & dev = model.devices[0];
                auto * host_buft = ggml_backend_dev_host_buffer_type(dev.dev);
                if (host_buft) {
                    buft = host_buft;
                }
            }

            backend_buft.push_back(buft);
            backend_ptrs.push_back(backend.get());
            backend_buf_exp_size.push_back(0);
        }

        LLAMA_LOG_DEBUG("%s: backend_ptrs.size() = %zu\n", __func__, backend_ptrs.size());

        // TODO: move these checks to ggml_backend_sched
        // enabling pipeline parallelism in the scheduler increases memory usage, so it is only done when necessary
        bool pipeline_parallel =
            model.n_devices() > 1 &&
            model.n_gpu_layers() > model.hparams.n_layer_all &&
            model.split_mode() == LLAMA_SPLIT_MODE_LAYER &&
            cparams.offload_kqv &&
            !model.has_tensor_overrides() &&
            cparams.moe_cache_size == 0; // not supported by the MoE cache

        // pipeline parallelism requires support for async compute and events in all devices
        if (pipeline_parallel) {
            for (auto & backend : backends) {
                auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));
                if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    // ignore CPU backend
                    // TODO: should we ignore ACCEL types too?
                    continue;
                }
                auto * dev = ggml_backend_get_device(backend.get());
                ggml_backend_dev_props props;
                ggml_backend_dev_get_props(dev, &props);
                if (!props.caps.async || !props.caps.events) {
                    // device does not support async compute or events
                    pipeline_parallel = false;
                    break;
                }
            }
        }

        cparams.pipeline_parallel = pipeline_parallel;

        if (cparams.pipeline_parallel) {
            LLAMA_LOG_INFO("%s: pipeline parallelism enabled\n", __func__);
        }

        if (cparams.moe_cache_size > 0) {
            moe_cache = std::make_unique<llama_moe_cache>(model, backend_ptrs, backend_buft, cparams.moe_cache_size);
        }

        // wip/slab-ring-region (issue #164): auto-size the H2D staging ring (per GPU).  The ring only has to
        // keep the prefill staging fed; too large and it evicts more arena on each wide pass.  Measured
        // knees on gfx1201 (GSQ -ncmoe 48, 31.5k prompt): n=1 ub2048 ~3 GiB, n=1 ub8192 ~6 GiB, n=2 ub8192
        // ~1.5 GiB, n=3 ub8192 <1 GiB -- i.e. the per-device need scales with 1/n (slices shrink) and
        // 1/n^2 (each device feeds a smaller share of the prefill).  1800/n + ub/(1.7 n^2) fits those.
        // Capped at 1/4 of the smallest staging device's VRAM so it scales down on a small card, and the
        // ring is one global region, so the smallest cap wins.  `GGML_CUDA_SLAB_RING_MIB` is a verbatim
        // per-GPU override and always wins.  Runs before `sched_reserve`, i.e. before the slab exists.
        // issue #164: `GGML_CUDA_SLAB_RING_MIB` is a verbatim per-GPU override and always wins, but a value
        // below the slice floor makes the H2D staging fall back to a cudaMalloc buffer -- which the
        // optional-allocation cap may then refuse -- so say so (non-fatal: the override is still honoured).
        const char * ring_env = getenv("GGML_CUDA_SLAB_RING_MIB");
        // Only the target context actually streams host-resident expert tables through the H2D staging
        // ring, so only it may size the ring.  The MTP draft context is built against the same model
        // (`moe_host_expert_bytes` is non-empty) but its weights are fully resident and it never stages a
        // table; running the auto-size here would overwrite the shared global region with the draft's
        // (smaller, ubatch-derived) value -- last-writer-wins -- and, before the carve-follows fix, moved
        // the wide compute buffer inside the target's carved ring.  Same MTP exclusion as the narrow-floor
        // pin and the ring-arm blocks below.
        if (!model.moe_host_expert_bytes.empty() && cparams.ctx_type != LLAMA_CONTEXT_TYPE_MTP) {
            const ggml_backend_dev_t stage_dev = model.moe_host_expert_bytes.begin()->first;
            const bool slab_enabled = ggml_backend_dev_slab_enabled(stage_dev);
            // issue #164: the knee above only balances prefill speed against arena retention; it
            // assumes the ring can hold one staged upload per slot.  A model whose per-device
            // expert-table slice is larger than ring/slots quietly falls back to a cudaMalloc staging
            // buffer, and once the arena has grown that fallback is refused by the optional-allocation
            // cap -- a large prefill after a decode then streams at ~half speed (measured 1462 -> 755
            // t/s on Qwen3.8-Flash-Next IQ4_NL `-ncmoe 46`).  Require room for `ring_slots` slices of
            // the largest staged table; the throughput knee is the floor, this is the fit requirement.
            // `llama_model::max_host_expert_slice_bytes` and `ggml_backend_meta_stage_input` share the rule.
            size_t slice = 0, n_slots = 0, slice_floor = 0;
            if (slab_enabled) {
                slice = model.max_host_expert_slice_bytes();
                if (slice > 0) {
                    n_slots = ggml_backend_dev_slab_ring_slots(stage_dev);
                    if (n_slots == 0) {
                        n_slots = 8;
                    }
                    slice_floor = n_slots * (slice + 512);
                }
            }
            if (ring_env == nullptr) {
                const double n_stage = (double) model.moe_host_expert_bytes.size();
                if (n_stage > 0.0) {
                    size_t ring_bytes = (size_t) ((1800.0 / n_stage +
                            (double) cparams.n_ubatch / (1.7 * n_stage * n_stage)) * 1024.0 * 1024.0);
                    if (slice_floor > ring_bytes) {
                        ring_bytes = slice_floor;
                    }
                    size_t vram_cap = (size_t) -1;
                    for (const auto & kv : model.moe_host_expert_bytes) {
                        size_t free_b = 0, total_b = 0;
                        ggml_backend_dev_memory(kv.first, &free_b, &total_b);
                        if (total_b > 0) {
                            vram_cap = std::min(vram_cap, total_b / 4);
                        }
                    }
                    if (vram_cap != (size_t) -1) {
                        ring_bytes = std::min(ring_bytes, vram_cap);
                    }
                    ggml_backend_dev_slab_set_ring_cap(stage_dev, ring_bytes);
                }
            } else if (slab_enabled && slice_floor > 0) {
                const size_t env_bytes = (size_t) atoll(ring_env) << 20;
                if (env_bytes < slice_floor) {
                    LLAMA_LOG_ERROR("%s: GGML_CUDA_SLAB_RING_MIB=%zu MiB is below the H2D staging slice floor of "
                            "%zu MiB (%zu slots x (%zu MiB largest per-device expert slice + 512 B)); the H2D "
                            "staging ring falls back to a cudaMalloc buffer, which the optional-allocation cap "
                            "may refuse, so a large prefill can stream at roughly half speed.  Raise it to at "
                            "least %zu MiB, or unset it to auto-size.\n",
                            __func__, env_bytes >> 20, slice_floor >> 20, n_slots, slice >> 20,
                            (slice_floor + (1u << 20) - 1) >> 20);
                }
            }
        }

        // the draft vocabulary subset belongs to this context; contexts that ask for the same N share one copy
        if (cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP && params.mtp_draft_vocab > 0) {
            mtp_draft = model.mtp_draft_vocab_get(params.mtp_draft_vocab);
            if (mtp_draft == nullptr && model.output == nullptr && !model.hparams.no_alloc && cparams.ctx_other != nullptr) {
                // shared-NextN sidecar (nextn_shared_target_tensors): this draft owns no LM head and its graph scores
                // with the head it borrows from the target model (see qwen4exp.cpp) - build the subset from that same
                // head. Fit-probe contexts (no_alloc) never see an allocated head; skipping them keeps the fitting
                // pass from allocating the subset outside its estimate
                mtp_draft = llama_get_model(cparams.ctx_other)->mtp_draft_vocab_get(params.mtp_draft_vocab,
                        /*shared_target_head=*/true);
            }
            if (mtp_draft) {
                cparams.mtp_draft_vocab = mtp_draft->n_keep;
            }
        }

        sched_reserve();

        // wip/first-prefill-latency: reserve the wide prefill work region in the slab BEFORE the MoE
        // cache's priming sizing (during the warmup) places the arena.  `sched_reserve` has just measured
        // the widest compute layout, so its largest compute buffer is the wide view; pinning that much
        // above the narrow floor + ring keeps the load-time arena above the region the first wide prefill
        // will use, instead of taking it and being evicted back on the first request.
        if (!model.moe_host_expert_bytes.empty()) {
            size_t wide_bytes = 0;
            for (size_t i = 0; i < backend_buf_exp_size.size(); ++i) {
                if (backend_buf_exp_size[i] > wide_bytes) {
                    wide_bytes = backend_buf_exp_size[i];
                }
            }
            if (wide_bytes > 0) {
                for (const auto & kv : model.moe_host_expert_bytes) {
                    ggml_backend_dev_slab_reserve_wide(kv.first, wide_bytes);
                }
                for (size_t d = 0; d < model.devices.size(); ++d) {
                    if (!model.devices[d].is_meta) {
                        ggml_backend_dev_slab_reserve_wide(model.devices[d].dev, wide_bytes);
                    }
                }
            }
        }

        if (!cparams.flash_attn) {
            if (ggml_is_quantized(params.type_v)) {
                throw std::runtime_error("quantized V cache was requested, but this requires Flash Attention");
            }
        }
    }

    // Initialize the full vocabulary token ids for backend samplers.
    {
        const int n_vocab = model.vocab.n_tokens();

        sampling.token_ids_full_vocab.resize(n_vocab);
        for (int i = 0; i < n_vocab; ++i) {
            sampling.token_ids_full_vocab[i] = i;
        }
    }
}

llama_context::~llama_context() {
    if (lf_consumed_ev != nullptr) {
        ggml_backend_event_free(lf_consumed_ev);
        lf_consumed_ev = nullptr;
    }
    for (auto & be : state_pipe_evs) {
        for (ggml_backend_event_t ev : be.second) {
            if (ev != nullptr) {
                ggml_backend_event_free(ev);
            }
        }
    }
    state_pipe_evs.clear();
    // wait for any pending asynchronous copies into the output buffers before they are freed
    synchronize();

    // when training, ggml_opt allocates extra buffers through the scheduler, so the sizes no longer match the expectation
    if (!model.hparams.no_alloc && !opt_ctx) {
        for (size_t i = 0; i < backend_ptrs.size(); ++i) {
            ggml_backend_t             backend = backend_ptrs[i];
            ggml_backend_buffer_type_t buft    = backend_buft[i];

            const size_t size_exp = backend_buf_exp_size[i];
            const size_t size_act = ggml_backend_sched_get_buffer_size(sched.get(), backend);
            if (size_exp == size_act) {
                LLAMA_LOG_DEBUG("%s: %10s compute buffer size is %8.4f MiB, matches expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            } else if (size_act < size_exp) {
                // The buffers were RE-RESERVED smaller during the run (the post-prefill compute-buffer
                // drop, or a narrower later reserve).  A legitimate shrink is not an accounting error.
                LLAMA_LOG_DEBUG("%s: %10s compute buffer size of %8.4f MiB after a mid-run re-reserve "
                    "(the reserve-time expectation was %8.4f MiB)\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            } else {
                LLAMA_LOG_WARN("%s: %10s compute buffer size of %8.4f MiB, does not match expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            }
        }
    }
    ggml_opt_free(opt_ctx);
}

void llama_context::resolve_fused_ops(const llama_memory_context_i * mctx, uint32_t n_seqs) {
    const char * func = __func__;
    auto resolve = [&](const llm_fused_op_probe & probe, bool & enabled) {
        if (!enabled) {
            return;
        }

        const uint32_t n_seqs_probe    = probe.n_seqs > 0 ? probe.n_seqs : n_seqs;
        const uint32_t n_tokens_probe  = probe.n_tokens_per_seq*n_seqs_probe;

        auto * gf = graph_reserve(n_tokens_probe, n_seqs_probe, n_tokens_probe, mctx, true);
        if (!gf) {
            throw std::runtime_error(std::string("failed to reserve graph for ") + probe.name + " check");
        }

        bool device_mismatch = false;
        bool observed = false;
        for (const auto & node : get_gf_res_reserve()->get_fused_nodes()) {
            if (node.op != probe.op) {
                continue;
            }

            observed = true;

            GGML_ASSERT(node.il >= 0);

            ggml_backend_t backend_fused = ggml_backend_sched_get_tensor_backend(sched.get(), node.tensor);
            ggml_backend_dev_t device_fused = backend_fused ? ggml_backend_get_device(backend_fused) : nullptr;

            // TODO: make this descriptor-specific; model.dev_layer() preserves the current behavior,
            // but is still wrong for cases like --no-kv-offload.
            ggml_backend_dev_t device_layer = model.dev_layer(node.il);

            if (device_fused != device_layer) {
                LLAMA_LOG_WARN("%s: layer %d is assigned to device %s but %s "
                        "is assigned to device %s (usually due to missing support)\n",
                        func, node.il,
                        device_layer ? ggml_backend_dev_name(device_layer) : "none",
                        probe.name,
                        device_fused ? ggml_backend_dev_name(device_fused) : "none");
                device_mismatch = true;
                break;
            }

            const auto device_fused_type = ggml_backend_dev_type(device_fused);

            if (probe.require_gpu && device_fused_type != GGML_BACKEND_DEVICE_TYPE_GPU &&
                    device_fused_type != GGML_BACKEND_DEVICE_TYPE_IGPU &&
                    device_fused_type != GGML_BACKEND_DEVICE_TYPE_META) {
                LLAMA_LOG_WARN("%s: %s is not supported on a CPU device\n", func, probe.name);
                device_mismatch = true;
                break;
            }

            if (probe.require_kq_derived && !ggml_backend_dev_implements_kq_derived(device_fused)) {
                LLAMA_LOG_WARN("%s: %s is assigned to %s, but it is only implemented by the CUDA/HIP backend\n",
                        func, probe.name,
                        device_fused ? ggml_backend_dev_name(device_fused) : "none");
                device_mismatch = true;
                break;
            }
        }

        if (!observed && probe.require_observed) {
            LLAMA_LOG_WARN("%s: %s was not used in the probe graph, set to disabled\n", func, probe.name);
            device_mismatch = true;
        }

        if (device_mismatch) {
            enabled = false;
            LLAMA_LOG_WARN("%s: %s not supported, set to disabled\n", func, probe.name);

            // The derived kq mask is implemented by both kernels that can serve a prefill: the MMA
            // and the tile flash attention kernel (the vec kernel is decode/verify-only, and the
            // derived form only exists for prefill-shaped batches).  So a rejection here is no longer
            // a head-cap or kernel-tuning issue - it means neither kernel served this graph.  The
            // generic "missing support" text sends people looking at the device instead, hence the
            // pointer at the kernel selection.
            if (probe.require_kq_derived) {
                LLAMA_LOG_WARN("%s: note: the derived kq mask is implemented by the MMA and the tile "
                        "flash attention kernels; it is disabled here because neither served this "
                        "graph (or the node did not reach the GPU)\n", func);
            }
        } else {
            enabled = true;
            LLAMA_LOG_INFO("%s: %s enabled\n", func, probe.name);
        }
    };

    if (cparams.auto_fa) {
        resolve(llm_fused_op_flash_attn_probe, cparams.flash_attn);
        cparams.auto_fa = false;
    }

    // note: after the flash attention probe, so that cparams.flash_attn is already resolved
    if (cparams.auto_kq_mask_derived) {
        // The derived form needs a single KV stream: llama_kv_cache::kq_mask_derivable() rejects a
        // multi-stream cache.  That is `n_seq_max > 1` without kv_unified for the plain cache, and
        // always `n_seq_max > 1` for deepseek4, which keeps per-sequence streams even in unified
        // mode (llama_kv_cache_dsv4 pins unified_raw/unified_compressed to false).  Do not run the
        // probe there - it forces a single-sequence graph, and on a multi-stream cache that is a
        // shape the cache-layout-driven attention builders reject (deepseek4's lightning indexer
        // ties its query stream count to the cache K's ne[3] while its mask uses the plan's stream
        // count, so mask->ne[1] != q->ne[2] and ggml_lightning_indexer asserts).  Derived is dead
        // in that configuration anyway, so skipping the probe loses nothing.
        const bool multi_stream = n_seqs > 1 && (!cparams.kv_unified || model.arch == LLM_ARCH_DEEPSEEK4);

        if (multi_stream) {
            LLAMA_LOG_INFO("%s: derived kq mask disabled (multi-stream KV cache)\n", func);
            cparams.kq_mask_derived = false;
        } else {
            LLAMA_LOG_INFO("%s: resolving derived kq mask support:\n", func);
            resolve(llm_fused_op_kq_derived_probe, cparams.kq_mask_derived);
        }
        cparams.auto_kq_mask_derived = false;
    }

    if (cparams.auto_fgdn) {
        LLAMA_LOG_INFO("%s: resolving fused Gated Delta Net support:\n", func);
        resolve(llm_fused_op_gdn_ar_probe, cparams.fused_gdn_ar);
        resolve(llm_fused_op_gdn_ch_probe, cparams.fused_gdn_ch);
        cparams.auto_fgdn = false;
    }

    if (cparams.auto_flid) {
        LLAMA_LOG_INFO("%s: resolving fused Lightning Indexer support:\n", func);
        resolve(llm_fused_op_lid_probe, cparams.fused_lid);
        cparams.auto_flid = false;
    }

    if (cparams.auto_fhc) {
        LLAMA_LOG_INFO("%s: resolving fused DeepSeek V4 HC support:\n", func);
        resolve(llm_fused_op_dsv4_hc_pre_probe,  cparams.fused_dsv4_hc_pre);
        resolve(llm_fused_op_dsv4_hc_comb_probe, cparams.fused_dsv4_hc_comb);
        resolve(llm_fused_op_dsv4_hc_post_probe, cparams.fused_dsv4_hc_post);
        cparams.auto_fhc = false;
    }

    if (cparams.auto_ftkm) {
        LLAMA_LOG_INFO("%s: resolving fused top-k mask support:\n", func);
        resolve(llm_fused_op_top_k_mask_probe, cparams.fused_top_k_mask);
        cparams.auto_ftkm = false;
    }
}

static int llama_graph_n_input_tensors(ggml_cgraph * gf) {
    std::unordered_map<const ggml_tensor *, std::vector<ggml_tensor *>> users;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        if (node->flags & GGML_TENSOR_FLAG_INPUT) {
            users[node].push_back(node);
        }
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            ggml_tensor * src = node->src[j];
            if (!src) {
                break;
            }
            if (src->flags & GGML_TENSOR_FLAG_INPUT) {
                users[src].push_back(node);
            }
        }
    }

    for (const auto & [tensor, nodes] : users) {
        GGML_ASSERT(tensor->op == GGML_OP_NONE);
        for (const ggml_tensor * node : nodes) {
            LLAMA_LOG_DEBUG("%s: input tensor '%32s' [%s, ne = { %5" PRId64 ", %5" PRId64 ", %5" PRId64 ", %5" PRId64 " }] is used by node '%s' (%s)\n",
                    __func__, tensor->name, ggml_type_name(tensor->type),
                    tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
                    node->name, ggml_op_name(node->op));
        }
    }

    return (int) users.size();
}

// The derived kq mask is a *per-batch* optimization, not a property of the context, but the reverse is
// not true either: not every batch shape is reachable in every context.  This predicate answers the
// reserve-time question "can a batch the derived form cannot serve actually be decoded here?" - if it
// cannot, the reserve stays with the (much smaller) derived layout and keeps V3's memory win.
//
// The reserve's own measure batch is 1-D and holds `n_seq_max` sequences of `n_tokens` tokens each
// (graph_reserve() -> llama_batch_allocr::ubatch_reserve()), so the runtime triggers have to be reasoned
// about from the model/context, not from a real ubatch:
//   * 2-D (M-RoPE) batches - an image/audio chunk decode carries 2-D positions (mtmd), which is only
//     possible for an M-RoPE model: llama_batch::is_pos_2d() is `n_pos >= 3`, and n_pos_per_embd() is
//     4 only for MROPE/IMROPE.  The packed mask's ext-based causal clause cannot be derived.  This is
//     the reason the M-RoPE term is NOT "any M-RoPE model" (issue #238): Qwen3.x is IMROPE even for
//     text, so that form reserved ~2 bytes per cell per ubatch token for a batch a text-only context
//     can never produce.  It is now `cparams.kq_mask_pos_2d` (the caller declared 2-D positions, e.g.
//     a multimodal projector) or the `kq_mask_pos_2d_seen` latch that decode() sets on the first 2-D
//     batch - which re-reserves once with the packed mask instead of relying on a per-graph growth.
//   * multi-sequence batches (ubatch.n_seqs_unq != 1), e.g. `--kv-unified` with several slots, when the
//     derived cell encoding cannot separate them (r172): a cell that carries two of the batch's
//     sequences (a prefix copied with llama_memory_seq_cp), a token in more than one sequence, a position
//     past KQ_DERIVED_SEQ_STRIDE, or at most KQ_DERIVED_MULTISEQ_MIN_TOKENS tokens.  That is exactly
//     llama_kv_cache::kq_mask_derivable_seqs(), and a multi-sequence batch is the one shape whose refusal
//     depends on the *cache* rather than only on the batch - which is why the decode() latch exists.
//   * alibi, which encodes a *value* (the distance) rather than a predicate.
// Every other reason the derived form is skipped is already covered without this predicate:
//   * the `allow_derived == false` builders (MLA / lightning-indexer / MSA) never take the derived
//     branch at all, so their reserve graph already holds the packed mask;
//   * alibi, a multi-stream KV cache and a disabled flash attention disable the derived form globally
//     (resolve_fused_ops()'s probe), so those reserves are packed anyway - the alibi term above is
//     belt-and-braces for that path;
//   * `n_kv % LLM_KQ_MASK_DERIVED_KV_STRIDE == 0` is not a runtime variable: llama_kv_cache::get_n_kv()
//     pads to at least 256 cells, so the stride gate always holds (verified on gfx1100 with a
//     text-only append at an off-stride position: every prefill graph logged DERIVED);
//   * a `n_tokens <= 8` batch, and a multi-sequence batch of at most KQ_DERIVED_MULTISEQ_MIN_TOKENS (32)
//     tokens - the multi-slot verify steps - take a packed mask, but it is a few MiB and fits the reserved
//     layout's slack (the same was true before the derived form existed).
// If a new reason to reject the derived form is ever added to kq_mask_derivable(), it must be added
// here too - otherwise a later batch can force the growth this predicate exists to prevent.  The
// ggml-alloc layout invalidation + the compute-buffer growth log are the safety net if that happens.
// upstream #29943/#29887: (re)create the scheduler and (re)install the expert-copy and graph-realloc
// callbacks.  The delivery's scheduler keeps its own copy path, but the realloc callback is load-bearing.
void llama_context::sched_new(size_t max_nodes, bool parallel) {
    sched.reset(ggml_backend_sched_new(
            backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, parallel, cparams.op_offload));

    ggml_backend_sched_set_copy_callback(sched.get(), sched_copy_experts, this);
    ggml_backend_sched_set_realloc_callback(sched.get(), sched_graph_realloc, this);
}

void llama_context::sched_alt_select(bool alt) {
    if (alt == sched_alt_in_use) {
        return;
    }
    if (alt && !sched_alt) {
        ggml_backend_sched_ptr sched_main = std::move(sched);
        sched_new(gf_res_reserve->get_max_nodes(), cparams.pipeline_parallel);
        sched_alt = std::move(sched);
        sched     = std::move(sched_main);
    }
    std::swap(sched, sched_alt);
    std::swap(gf_res_prev_active, gf_res_prev_active_alt);
    sched_alt_in_use = alt;
    ggml_backend_meta_set_view_slot(alt ? 1 : 0);
}

bool llama_context::sched_alt_for(const llama_ubatch & ubatch) const {
    if (cparams.ctx_other == nullptr) {
        return false;
    }

    // read once: this runs for every ubatch; getenv() is expensive on Windows
    static const bool draft_sched2 = [] {
        const char * e = getenv("LLAMA_DRAFT_SCHED2");
        return e == nullptr || atoi(e) != 0;
    }();
    if (!draft_sched2) {
        return false;
    }

    if (model.arch == LLM_ARCH_DFLASH) {
        // A DFlash/DFlash2 draft context alternates two graphs and NEITHER marks outputs: the KV injection
        // (host feature rows, or the F1 device injection whose token ids are target rows), and the noise-block
        // draft step, whose result is read from h_nextn (the selector lattice) instead of the logits.  So
        // `n_outputs` cannot tell them apart -- the batch kind can: the injection is the one that carries the
        // target features.  Without this split both graphs share one scheduler and one `gf_res_prev` slot, the
        // draft context rebuilds and re-allocates both of them on every verify step and never reuses a graph
        // (issue #208).  MTP keeps `n_outputs`: its ingest is a token batch without outputs and its step graph
        // is the one with outputs, so the batch kind would merge the two graphs instead of separating them.
        return ubatch.embd == nullptr && !cparams.lf_dev_inject;
    }

    return n_outputs > 0;
}

// the attention KV caches that decide the derived kq mask: the cache itself, the attention part of a hybrid memory,
// or both caches of an iSWA memory; empty for any other memory (which then keeps the packed reserve).  Resolved once:
// `memory` is immutable for the lifetime of the context, and decode() asks this on every batch until a latch fires.
const std::vector<llama_kv_cache *> & llama_context::kq_mask_kvs() const {
    if (memory && !kq_mask_kvs_done) {
        kq_mask_kvs_store.clear();

        if (auto * kv = dynamic_cast<llama_kv_cache *>(memory.get())) {
            kq_mask_kvs_store = { kv };
        } else if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(memory.get())) {
            kq_mask_kvs_store = { hybrid->get_mem_attn() };
        } else if (auto * iswa = dynamic_cast<llama_kv_cache_iswa *>(memory.get())) {
            kq_mask_kvs_store = { iswa->get_base(), iswa->get_swa() };
        }

        kq_mask_kvs_done = true;
    }

    return kq_mask_kvs_store;
}

bool llama_context::kq_mask_packed_reachable() const {
    // issue #238: an explicit LLAMA_KQ_MASK_PACKED_RESERVE=1 restores the pre-r160 worst-case reserve for
    // every model (an A/B arm and a safety valve, not a policy)
    if (cparams.kq_mask_packed_reserve > 0) {
        return true;
    }

    // several sequences in one batch: on a single-stream cache they derive their mask too (each in its own position
    // range, llama_kv_cache::kq_mask_derivable_seqs); a batch they cannot serve latches `kq_mask_multiseq_seen` in
    // decode() and re-reserves once.  Without the multi-sequence form (LLAMA_KQ_MASK_DERIVED_MULTISEQ=0, or a memory
    // it does not cover) such batches take the packed mask.
    const bool multiseq_packed = cparams.n_seq_max > 1 &&
            (kq_mask_multiseq_seen || !cparams.kq_mask_derived || kq_mask_kvs().empty() ||
             !llama_kv_cache::kq_mask_derived_multiseq_enabled());

    if (model.hparams.n_pos_per_embd() > 1) {
        // 2-D (M-RoPE) positions.  Only a caller that can submit them (a multimodal projector) needs the
        // packed mask reserved up front; a 2-D batch that arrives on a context which did not declare it
        // latches `kq_mask_pos_2d_seen` and re-reserves once (see decode()).  A text-only context keeps
        // the derived layout, which every M-RoPE text batch satisfies (kq_mask_derivable() accepts
        // degenerate positions, and qwen35/qwen3.8 are IMROPE even for text).
        if (cparams.kq_mask_packed_reserve == 0) {
            return false;
        }
        return cparams.kq_mask_pos_2d || kq_mask_pos_2d_seen || multiseq_packed;
    }

    if (multiseq_packed) {
        return true;
    }

    return model.hparams.use_alibi;
}

uint32_t llama_context::verify_batch_width() const {
    // TODO #96: see the declaration.  The band is queried live (not cached) on purpose: it is 0 once the
    // slab declines or the arena is released, and the reserve probes below run after the first compute
    // buffer (i.e. after that decision), so the answer here is the one the run will really issue.
    const int band = llama_model_moe_cache_band(&model);
    if (band > 1 && (uint32_t) band < cparams.n_rs_batch) {
        return (uint32_t) band;
    }
    return cparams.n_rs_batch;
}

void llama_context::sched_reserve() {
    if (!sched_need_reserve) {
        return;
    }

    sched_need_reserve = false;

    LLAMA_LOG_INFO("%s: reserving ...\n", __func__);

    synchronize();

    const int64_t t_start_us_l = ggml_time_us();

    const uint32_t n_seqs = cparams.n_seq_max;
    const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

    // DFlash draft (issue #239): the draft context alternates two graph shapes - the target-feature KV
    // injection (embd rows, up to `n_ubatch`, on the main scheduler) and the noise-block decoder (at most
    // `n_seqs x block` tokens, on the alt scheduler).  The decoder applies the lm_head to every row, so
    // measuring it at `n_ubatch` reserves `result_output` as `[n_vocab, n_ubatch]` f32 - about 1.9 GiB at
    // 2048 tokens and a 248k vocabulary - for a width no graph ever uses.  Reserve the injection at
    // `n_ubatch` and the decoder at the only width it receives; the compute buffer keeps the larger of the
    // two, so neither shape can grow it mid-run (issue #42).
    // `block` covers the lilicorr block and the DSpark/Dflash `n_max + 1` rows per sequence.  A model that
    // does not declare `dflash.block_size` gets common's default 16, which is also the bound its `n_max`
    // clamp uses.
    const bool dflash_draft = model.arch == LLM_ARCH_DFLASH && cparams.ctx_other != nullptr;
    const uint32_t dflash_block = model.hparams.dflash_block_size > 0 ? model.hparams.dflash_block_size : 16;
    const uint32_t n_tokens_draft = dflash_draft
            ? std::min<uint32_t>(n_tokens, n_seqs*std::max<uint32_t>(dflash_block,
                    std::max<uint32_t>(cparams.n_outputs_max_per_seq, 1)))
            : n_tokens;

    // size the compute buffers for the packed kq mask when a batch the derived form cannot serve is
    // reachable in this context - see kq_mask_packed_reachable()
    const bool packed_kq_mask = kq_mask_packed_reachable();

    const size_t max_nodes = this->graph_max_nodes(n_tokens);

    LLAMA_LOG_DEBUG("%s: max_nodes = %zu\n", __func__, max_nodes);

    for (auto & res : gf_res_prev) {
        res.reset();
    }
    gf_res_reserve.reset(new llm_graph_result(max_nodes));
    gf_res_prev_active = nullptr;

    GGML_ASSERT(!sched_alt_in_use);
    sched_alt.reset();
    gf_res_prev_active_alt = nullptr;

    sched_new(max_nodes, cparams.pipeline_parallel);

    llama_memory_context_ptr mctx;
    if (memory) {
        LLAMA_LOG_DEBUG("%s: reserving full memory module\n", __func__);
        mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory module");
        }
    }

    // avoid reserving graphs with zero outputs - assume one output per sequence
    const int n_outputs_l = n_seqs;

    LLAMA_LOG_DEBUG("%s: worst-case: n_tokens = %d, n_seqs = %d, n_outputs = %d\n", __func__, n_tokens, n_seqs, n_outputs_l);

    resolve_fused_ops(mctx.get(), n_seqs);

    // reserve worst-case graph
    int n_splits_pp        = -1;
    int n_nodes_pp         = -1;
    int n_inputs_pp        = -1;
    int n_input_tensors_pp = -1;

    int n_splits_tg        = -1;
    int n_nodes_tg         = -1;
    int n_inputs_tg        = -1;
    int n_input_tensors_tg = -1;

    const uint32_t n_outputs_pp = std::min(n_tokens, cparams.n_outputs_max);
    const uint32_t n_outputs_draft = std::min(n_tokens_draft, cparams.n_outputs_max);

    // reserve pp (prompt processing) graph first so that buffers are only allocated once
    {
        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(),
                model.hparams.no_alloc, model.hparams.no_alloc ? backend_buf_exp_size.data() : nullptr,
                packed_kq_mask, /*reserve_injection =*/ dflash_draft);
        if (!gf) {
            if (cparams.pipeline_parallel) {
                LLAMA_LOG_WARN("%s: compute buffer allocation failed, retrying without pipeline parallelism\n", __func__);
                cparams.pipeline_parallel = false;
                sched_new(max_nodes, false);
                gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(), /*split_only =*/ false, nullptr, packed_kq_mask,
                        /*reserve_injection =*/ dflash_draft);
            }
            if (!gf) {
                throw std::runtime_error("failed to allocate compute pp buffers");
            }
        }

        n_splits_pp        = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_pp         = ggml_graph_n_nodes(gf);
        n_inputs_pp        = get_gf_res_reserve()->inputs.size();
        n_input_tensors_pp = this->n_input_tensors;

        graph_has_logits = get_gf_res_reserve()->get_logits() != nullptr;
    }

    // reserve with tg (token generation) graph to get the number of splits and nodes
    if (cparams.training) {
        // no tg graph for training
        n_splits_tg = n_splits_pp;
        n_nodes_tg  = n_nodes_pp;
    } else {
        // a DFlash draft's token graph is the noise-block decoder, whose width is the block (not `n_ubatch`)
        auto * gf = dflash_draft
                ? graph_reserve(n_tokens_draft, n_seqs, n_outputs_draft, mctx.get(), model.hparams.no_alloc, nullptr, packed_kq_mask)
                : graph_reserve(n_seqs,         n_seqs, n_seqs,           mctx.get(), model.hparams.no_alloc, nullptr, packed_kq_mask);
        if (!gf) {
            throw std::runtime_error("failed to allocate compute tg buffers");
        }

        n_splits_tg        = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_tg         = ggml_graph_n_nodes(gf);
        n_inputs_tg        = get_gf_res_reserve()->inputs.size();
        n_input_tensors_tg = this->n_input_tensors;

        // the pp (injection) reserve has no lm_head, so for a DFlash draft the token graph is the one that
        // decides whether logits exist
        if (dflash_draft) {
            graph_has_logits = get_gf_res_reserve()->get_logits() != nullptr;
        }
    }

    // reserve again with pp graph to avoid ggml-alloc reallocations during inference
    {
        // TODO: the worst case graph is not always reached for `n_seqs > 1`
        //       need to implement a more robust mechanism that tries a few different inputs and analyzes the results
        ggml_cgraph * gf = nullptr;
        switch (model.arch) {
            case LLM_ARCH_KIMI_LINEAR:
            case LLM_ARCH_MINIMAX_01:
                // [TAG_RESERVE_DIAG_DECAY]
                // the `inp_diag_decay` tensor size scales with `n_seq_tokens^2` which
                // makes `n_seqs == 1` use more memory for the compute graph compared to `n_seqs > 1`
                gf = graph_reserve(n_tokens, 1,      n_outputs_pp, mctx.get(), model.hparams.no_alloc, nullptr, packed_kq_mask);
                break;
            default:
                // a DFlash draft leaves the layout at the injection graph: that is the graph this scheduler
                // runs on every verify step, and the decoder width was already reserved above
                gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(), model.hparams.no_alloc, nullptr, packed_kq_mask,
                        /*reserve_injection =*/ dflash_draft);
        };

        if (!gf) {
            throw std::runtime_error("failed to allocate compute pp buffers");
        }
    }

    // the buffers now cover a backend sampler of this size for these sequences (see set_sampler)
    sampling.reserved.clear();
    for (const auto & entry : sampling.samplers) {
        sampling.reserved[entry.first] = llama_sampler_backend_n_nodes(entry.second);
    }

    for (size_t i = 0; i < backend_ptrs.size(); ++i) {
        ggml_backend_t             backend = backend_ptrs[i];
        ggml_backend_buffer_type_t buft    = backend_buft[i];
        if (!model.hparams.no_alloc) {
            backend_buf_exp_size[i] = ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
        if (backend_buf_exp_size[i] > 1) {
            LLAMA_LOG_INFO("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buft_name(buft),
                    backend_buf_exp_size[i] / 1024.0 / 1024.0);
        }
    }

    // wip/moe-verify-fusions (narrow-2): re-affirm the pinned narrow-2 floor for the MTP draft context
    // from its measured compute size.  The region is established lazily on the draft's first allocation
    // (its widest reserve); this keeps the pin at least as large as that layout.  A no-op for the target
    // and when the slab is not live on a device.
    if (cparams.ctx_other != nullptr && !model.hparams.no_alloc) {
        size_t narrow2_bytes = 0;
        for (size_t i = 0; i < backend_ptrs.size(); ++i) {
            if (backend_buf_exp_size[i] > narrow2_bytes) {
                narrow2_bytes = backend_buf_exp_size[i];
            }
        }
        if (narrow2_bytes > 0) {
            for (const auto & d : model.devices) {
                ggml_backend_dev_slab_narrow2_floor(d.dev, narrow2_bytes);
            }
        }
    }

    {
        const bool diff = n_nodes_pp != n_nodes_tg || n_splits_pp != n_splits_tg ||
                          n_inputs_pp != n_inputs_tg || n_input_tensors_pp != n_input_tensors_tg;

        const auto val = [diff](int v_pp, int v_tg) -> std::string {
            return diff ? format("%d / %d", v_pp, v_tg) : format("%d", v_pp);
        };

        LLAMA_LOG_INFO("%s: graph%s: nodes = %s, splits = %s, input objects = %s, input tensors = %s\n",
                __func__,
                diff ? format(" (pp bs=%d, tg bs=%d)", n_tokens, n_seqs).c_str() : "",
                val(n_nodes_pp, n_nodes_tg).c_str(),
                val(n_splits_pp, n_splits_tg).c_str(),
                val(n_inputs_pp, n_inputs_tg).c_str(),
                val(n_input_tensors_pp, n_input_tensors_tg).c_str());
    }

    const int64_t t_end_us = ggml_time_us();

    LLAMA_LOG_INFO("%s: reserve took %.2f ms, sched copies = %d\n",
            __func__, (t_end_us - t_start_us_l)/1000.0, ggml_backend_sched_get_n_copies(sched.get()));
}

void llama_context::drop_compute_buffers() {
    // WIP r42 (TODO #42): release the grow-only compute buffers so the next alloc_graph re-reserves them
    // from the graph at hand.  See ggml_backend_sched_drop_buffers().
    if (!sched) {
        return;
    }
    ggml_backend_sched_drop_buffers(sched.get());
}

void llama_context_drop_compute_buffers(struct llama_context * ctx) {
    if (ctx == nullptr) {
        return;
    }
    ctx->drop_compute_buffers();
}

void llama_context::synchronize() {
    if (!sched) {
        return;
    }

    ggml_backend_sched_synchronize(sched.get());

    // FIXME: if multiple single tokens are evaluated without a synchronization,
    // the stats will be added to the prompt evaluation stats
    // this should only happen when using batch size 1 to evaluate a batch

    // add the evaluation to the stats
    if (n_queued_tokens == 1) {
        if (!cparams.no_perf) {
            t_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_eval++;
    } else if (n_queued_tokens > 1) {
        if (!cparams.no_perf) {
            t_p_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_p_eval += n_queued_tokens;
    }

    // get a more accurate load time, upon first eval
    if (n_queued_tokens > 0 && !has_evaluated_once) {
        t_load_us = ggml_time_us() - t_start_us;
        has_evaluated_once = true;
    }

    n_queued_tokens = 0;
    t_compute_start_us = 0;
}

const llama_model & llama_context::get_model() const {
    return model;
}

const llama_cparams & llama_context::get_cparams() const {
    return cparams;
}

ggml_backend_sched_t llama_context::get_sched() const {
    return sched.get();
}

uint32_t llama_context::n_ctx() const {
    return cparams.n_ctx;
}

uint32_t llama_context::n_ctx_seq() const {
    return cparams.n_ctx_seq;
}

uint32_t llama_context::n_batch() const {
    return cparams.n_batch;
}

uint32_t llama_context::n_ubatch() const {
    return cparams.n_ubatch;
}

uint32_t llama_context::n_seq_max() const {
    return cparams.n_seq_max;
}

uint32_t llama_context::n_threads() const {
    return cparams.n_threads;
}

uint32_t llama_context::n_threads_batch() const {
    return cparams.n_threads_batch;
}

llama_memory_t llama_context::get_memory() const {
    return memory.get();
}

bool llama_context::memory_update(bool optimize) {
    if (!memory) {
        return false;
    }

    {
        const auto mctx = memory->init_update(this, optimize);
        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                    // noop
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    // no updates need to be performed
                    return false;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: failed to prepare memory update\n", __func__);
                    return false;
                }
        }

        // reset the previous graph results to make sure that they won't be reused
        // TODO: make mctx->apply() report if a graph reserve is needed, then reset graph results only if the memory module reset the scheduler
        for (auto & res : gf_res_prev) {
            if (res) {
                res->reset();
            }
        }
        gf_res_prev_active = nullptr;
        gf_res_prev_active_alt = nullptr;

        if (!mctx->apply()) {
            LLAMA_LOG_ERROR("%s: failed to apply memory update\n", __func__);
        }
    }

    // if the memory module did any computation, we have to reserve a new worst-case graph
    {
        const auto mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory context");
        }

        const uint32_t n_seqs = cparams.n_seq_max;
        const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

        const uint32_t n_outputs_max = std::min(n_tokens, cparams.n_outputs_max);

        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_max, mctx.get(), /*split_only =*/ false, nullptr, kq_mask_packed_reachable());
        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to reserve graph after the memory update\n", __func__);
        }
    }

    return true;
}

enum llama_pooling_type llama_context::pooling_type() const {
    return cparams.pooling_type;
}

float * llama_context::get_logits() {
    output_reorder();

    return logits.data;
}

int64_t llama_context::output_resolve_row(int32_t i) const {
    int64_t j = -1;

    // support negative indices (last output row)
    if (i < 0) {
        j = n_outputs + i;
        if (j < 0) {
            throw std::runtime_error(format("negative index out of range [0, %d)", n_outputs));
        }
    } else if ((size_t) i >= output_ids.size()) {
        throw std::runtime_error(format("out of range [0, %zu)", output_ids.size()));
    } else {
        // use output_ids to translate the batch token index into a row number
        // that holds this token's data.
        j = output_ids[i];
    }

    if (j < 0) {
        // the batch token was not configured to output anything
        throw std::runtime_error(format("batch.logits[%d] != true", i));
    }

    if (j >= n_outputs) {
        throw std::runtime_error(format("corrupt output buffer (j=%" PRId64 ", n_outputs=%d)", j, n_outputs));
    }

    return j;
}

float * llama_context::get_logits_ith(int32_t i) {
    output_reorder();

    try {
        if (logits.data == nullptr) {
            throw std::runtime_error("no logits");
        }

        const int64_t j = output_resolve_row(i);
        return logits.data + j*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid logits id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings() {
    output_reorder();

    return embd.data;
}

llama_token * llama_context::get_sampled_tokens()  const{
    return sampling.sampled.data;
}

float * llama_context::get_embeddings_ith(int32_t i) {
    output_reorder();

    try {
        if (embd.data == nullptr) {
            throw std::runtime_error("no embeddings");
        }

        const int64_t j = output_resolve_row(i);
        const uint32_t n_embd_out = model.hparams.n_embd_out();
        return embd.data + j*n_embd_out;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_seq(llama_seq_id seq_id) {
    auto it = embd_seq.find(seq_id);
    if (it == embd_seq.end()) {
        return nullptr;
    }

    return it->second.data();
}

float * llama_context::get_embeddings_nextn() {
    output_reorder();

    return embd_nextn.data;
}

float * llama_context::get_embeddings_nextn_ith(int32_t i) {
    output_reorder();

    try {
        if (embd_nextn.data == nullptr) {
            throw std::runtime_error("no nextn embeddings");
        }

        const uint32_t n_embd = model.hparams.n_embd_out();

        if (!cparams.embeddings_nextn_masked) {
            // unmasked: nextn rows are stored densely, indexed by raw token position.
            if (i < 0 || (size_t)(i + 1) * n_embd > embd_nextn.size) {
                throw std::runtime_error(format("out of range [0, %zu)", embd_nextn.size / n_embd));
            }
            return embd_nextn.data + (size_t) i * n_embd;
        }

        const int64_t j = output_resolve_row(i);
        return embd_nextn.data + j*n_embd;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid nextn embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_layer_inp(uint32_t lid) {
    output_reorder();

    GGML_ASSERT(lid < embd_layer_inp.size() && embd_layer_inp[lid].has_data());

    return embd_layer_inp[lid].data;
}

llama_token llama_context::get_sampled_token_ith(int32_t idx) {
    output_reorder();

    if (!sampling.sampled.has_data()) {
        return LLAMA_TOKEN_NULL;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        GGML_ASSERT(row < (int64_t) sampling.sampled.size);
        return sampling.sampled.data[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled token id %d, reason: %s\n", __func__, idx, err.what());
        return LLAMA_TOKEN_NULL;
    }
}

float * llama_context::get_sampled_probs_ith(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size() || sampling.probs_count[row] == 0) {
            return nullptr;
        }
        return sampling.probs.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

float * llama_context::get_sampled_logits_ith(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size() || sampling.logits_count[row] == 0) {
            return nullptr;
        }
        return sampling.logits.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

const llama_token * llama_context::get_sampled_candidates_ith(int32_t idx) {
    output_reorder();

    try {
        const int64_t row = output_resolve_row(idx);
        if (sampling.candidates.has_data() &&
            (size_t) row < sampling.candidates_count.size() &&
            sampling.candidates_count[row] > 0) {
            return sampling.candidates.data + row*model.vocab.n_tokens();
        }
    } catch (const std::exception & err) {
        // fallback to full vocab list
        GGML_UNUSED(err);
    }

    return sampling.token_ids_full_vocab.data();
}

size_t llama_context::get_sampled_candidates_count(int32_t idx) {
    output_reorder();

    if (!sampling.candidates.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.candidates_count.size()) {
            return 0;
        }
        return sampling.candidates_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled candidates count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_logits_count(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return model.vocab.n_tokens();
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size()) {
            return 0;
        }
        return sampling.logits_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_probs_count(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size()) {
            return 0;
        }
        return sampling.probs_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}


void llama_context::attach_threadpool(
           ggml_threadpool_t threadpool_l,
           ggml_threadpool_t threadpool_batch_l) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = threadpool_l;
    this->threadpool_batch = threadpool_batch_l ? threadpool_batch_l : threadpool_l;
}

void llama_context::detach_threadpool() {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = nullptr;
    this->threadpool_batch = nullptr;
}

void llama_context::set_n_threads(int32_t n_threads, int32_t n_threads_batch) {
    LLAMA_LOG_DEBUG("%s: n_threads = %d, n_threads_batch = %d\n", __func__, n_threads, n_threads_batch);

    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads_batch;
}

void llama_context::set_abort_callback(bool (*abort_callback_l)(void * data), void * abort_callback_data_l) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->abort_callback      = abort_callback_l;
    this->abort_callback_data = abort_callback_data_l;

    for (auto & backend : backends) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        if (reg) {
            auto * set_abort_callback_fn = (ggml_backend_set_abort_callback_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_abort_callback");
            if (set_abort_callback_fn) {
                set_abort_callback_fn(backend.get(), this->abort_callback, this->abort_callback_data);
            }
        }
    }
}

void llama_context::set_embeddings(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    // the toggle changes the graph shape (pooled: logits only; pooling NONE: the reserve graph outputs all tokens,
    // which crashed ggml_gallocr on the first /completion), so re-reserve on every toggle
    if (cparams.embeddings != value) {
        sched_need_reserve = true;
    }

    cparams.embeddings = value;
}

void llama_context::set_embeddings_nextn(bool value, bool masked) {
    LLAMA_LOG_DEBUG("%s: value = %d, masked = %d\n", __func__, value, masked);

    if (cparams.embeddings_nextn != value || cparams.embeddings_nextn_masked != masked) {
        // these flags change the graph shape
        sched_need_reserve = true;
    }

    cparams.embeddings_nextn        = value;
    cparams.embeddings_nextn_masked = masked;
}

void llama_context::set_embeddings_nextn_from(llama_seq_id seq_id, llama_pos pos) {
    if (seq_id >= 0 && seq_id < LLAMA_MAX_SEQ) {
        embd_nextn_from[seq_id] = std::max<llama_pos>(0, pos);
    }
}

void llama_context::set_embeddings_layer_inp(uint32_t lid, bool enable) {
    LLAMA_LOG_DEBUG("%s: lid = %d, enable = %d\n", __func__, lid, enable);

    GGML_ASSERT(lid <= model.hparams.n_layer());

    if (cparams.embeddings_layer_inp[lid] == enable) {
        // unchanged (preset at context creation): the reserved graph has it already, and a reserve would free and
        // allocate all compute buffers again
        return;
    }

    cparams.embeddings_layer_inp[lid] = enable;

    // note: without this reserve, the draft acceptance drops to zero. not sure why - this is unexpected
    sched_need_reserve = true;
}

void llama_context::set_nextn_layer_offset(int32_t offset) {
    cparams.nextn_layer_offset = offset;
}

void llama_context::set_causal_attn(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.causal_attn == value) {
        return;
    }

    cparams.causal_attn = value;

    // no scheduler reserve needed because graph shapes must not depend on causal_attn, a flip only rebuilds the graph
    //sched_need_reserve = true;
}

bool llama_context::get_causal_attn() const {
    return cparams.causal_attn;
}

void llama_context::set_warmup(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.warmup == value) {
        return;
    }

    cparams.warmup = value;

    // warmups are usually with small batches, so no need to reserve
    //sched_need_reserve = true;
}

// LLAMA_BACKEND_SAMPLER_RESERVE_ONCE=0: reserve the compute buffers again on every set or removal of a backend sampler,
// as before (see set_sampler)
static bool backend_sampler_reserve_once() {
    static const bool enabled = [] {
        const char * e = getenv("LLAMA_BACKEND_SAMPLER_RESERVE_ONCE");
        return e == nullptr || atoi(e) != 0;
    }();
    return enabled;
}

bool llama_context::set_sampler(llama_seq_id seq_id, llama_sampler * sampler) {
    if (!sampler && sampling.samplers.count(seq_id) == 0) {
        return true;
    }

    LLAMA_LOG_DEBUG("%s: seq_id = %d, sampler = %p\n", __func__, (int) seq_id, (void *) sampler);

    if (sampler && llm_split_mode_is_tensor_like(model.split_mode())) {
        static bool warned = false;
        if (!warned) {
            LLAMA_LOG_WARN("%s: backend sampling not supported with SPLIT_MODE_TENSOR; using CPU\n", __func__);
            warned = true;
        }
        if (sampling.samplers.count(seq_id) > 0) {
            sched_need_reserve = true;
        }
        sampling.samplers.erase(seq_id);
        return false;
    }

    const bool can_offload =
        sampler &&
        sampler->iface->backend_init &&
        sampler->iface->backend_apply &&
        llama_sampler_chain_n(sampler) > 0;

    if (sampler && can_offload) {
        auto * buft = ggml_backend_dev_buffer_type(model.dev_output());

        sampler->iface->backend_init(sampler, buft, cparams.n_outputs_max_per_seq);

        sampling.samplers[seq_id] = sampler;

        // A reserve re-creates the scheduler, which frees and allocates every compute buffer again, so ask for one only
        // when the buffers were not reserved with a backend sampler at least as wide as this one for the sequence.  A
        // chain set again (the server sets a new one for every request), a narrower one, or a removal needs none: the
        // graph is rebuilt anyway (its samplers are part of the reuse check), the output buffers are sized per decode,
        // the allocator grows a buffer if a chain needs more, and only a WIDER chain can exceed the node capacity that
        // graph_max_nodes() froze at the last reserve (ggml_build_forward asserts on it).
        const uint32_t n_nodes = llama_sampler_backend_n_nodes(sampler);
        const auto it = sampling.reserved.find(seq_id);
        if (!backend_sampler_reserve_once() || it == sampling.reserved.end() || it->second < n_nodes) {
            sched_need_reserve = true;
        }

        return true;
    }

    if (sampler && !can_offload) {
        LLAMA_LOG_WARN("%s: sampler '%s' for seq_id = %d, cannot be offloaded to the backend\n", __func__, llama_sampler_name(sampler), seq_id);

        // no reserve: the buffers reserved with the sampler are large enough without it
        if (!backend_sampler_reserve_once() && sampling.samplers.count(seq_id) > 0) {
            sched_need_reserve = true;
        }
        sampling.samplers.erase(seq_id);

        return false;
    }

    // no reserve: the buffers reserved with the sampler are large enough without it
    sampling.samplers.erase(seq_id);

    if (!backend_sampler_reserve_once()) {
        sched_need_reserve = true;
    }

    return true;
}

void llama_context::set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    if (adapters_lora_are_same(adapters, n_adapters, scales)) {
        return;
    }

    loras.reset(new llama_adapter_loras());

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] != 0.0f) {
            loras->insert({adapters[i], scales[i]});
        }
    }

    sched_need_reserve = true;
}

bool llama_context::adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    // Adapters with a zero scale are never added to `loras`, so also ignore them for the comparison.
    size_t n_non_zero = 0;

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] == 0.0f) {
            continue;
        }
        n_non_zero++;

        auto it = loras->find(adapters[i]);

        if (it == loras->end() || it->second != scales[i]) {
            return false;
        }
    }

    if (n_non_zero != loras->size()) {
        return false;
    }

    return true;
}

bool llama_context::set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end) {
    LLAMA_LOG_DEBUG("%s: il_start = %d, il_end = %d\n", __func__, il_start, il_end);

    bool res = cvec->apply(model, data, len, n_embd, il_start, il_end);

    sched_need_reserve = true;

    return res;
}

    // OPEN 2: how much wider than the last settled narrow layout the live work region must be before the
    // compute reserve is dropped and re-reserved.  A material difference only -- the narrow layout varies a
    // little between a 1-token decode and an `n_rs_batch`-token verify pass, and dropping for that would
    // thrash the layout every token.  The real cases are GiBs (a 6.75 GiB wide reserve vs a ~1 GiB narrow
    // one), so 512 MiB is comfortably below anything worth reclaiming.
    static const size_t GGML_DROP_RESERVE_SLACK_BYTES = (size_t) 512 << 20;

llm_graph_result * llama_context::process_ubatch(const llama_ubatch & ubatch, llm_graph_type gtype, llama_memory_context_i * mctx, ggml_status & ret) {
    if (mctx && !mctx->apply()) {
        LLAMA_LOG_ERROR("%s: failed to apply memory context\n", __func__);
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }

    // TODO #42 (OPEN 1): release the wide-prefill compute layout at the prefill -> decode transition so
    // the MoE expert-cache arena can be sized large.  ON by default for every tool now that the
    // movable-boundary slab can reclaim a wide layout later (the boundary move evicts only the arena
    // tables in the chunks the work pool takes), so a server with `-np 4` still benefits whenever fewer
    // streams are active.  `LLAMA_DROP_COMPUTE_BUFFERS=0` is the kill switch.
    // read once: this runs for every ubatch; getenv() is expensive on Windows
    static const int drop_env = [] {
        const char * e = getenv("LLAMA_DROP_COMPUTE_BUFFERS");
        return e != nullptr ? atoi(e) : -1;
    }();
    const bool drop_enabled = drop_env >= 0 ? drop_env != 0 : cparams.drop_compute_buffers;
    // OPEN 2: WHEN to drop.  The compute reserve is created for the WIDEST graph the parameters allow, and
    // the work region holds it until a drop releases it.  The classic condition below only drops after a
    // WIDE pass (`n_tokens_prev > 8`), so a workload that never prefills -- a chat of short prompts -- keeps
    // the full reserve forever, and the MoE arena (sized against that boundary) stays small: measured on
    // `-ub 4096 -c 163860`, arena 32263 MiB with the wide reserve held vs 34645 MiB once it was released.
    //
    // Under the movable-boundary slab a drop is cheap and safe (the layout comes back via a boundary move),
    // so the condition becomes "the live work region is materially wider than the narrow layout we last
    // settled on" -- self-calibrating, no history guess, and it fires on the FIRST narrow pass of a
    // short-prompt workload.  Without a slab the classic wide-pass condition is kept, because there
    // reclaiming the wide layout is exactly what used to abort.
    // The dsv4 compressed cache builds its raw attention from the per-stream index fanout, so its
    // synthetic reserve probe has always used ubatch.n_seqs (a multi-stream shape) and must keep
    // doing so: a one-stream reserve left the next real decode reading a one-stream sinfo and
    // aborting in cpy_k.  Every other arch uses the real sequence count below, because the 4-D kq
    // mask derives its stream dim from it and must agree with the single-stream query tensor.
    const bool dsv4_cache = model.arch == LLM_ARCH_DEEPSEEK4 ||
            (model.arch == LLM_ARCH_DFLASH && model.hparams.dsv4_hc_mult > 0);
    bool drop_now = false;
    // The work region actually observed before the drop.  The reference must be at least this: the narrow
    // layout varies with the graph (the MTP decode/verify graph is wider than the `n_rs_batch` reserve the
    // drop re-creates), and recording only the post-reserve size made every following pass look
    // "materially wider" -- one drop per decode step, each evicting the band from the arena.
    size_t drop_observed = 0;
    if (drop_enabled && sched && ubatch.n_tokens <= 8 && n_tokens_prev > 0 &&
            !model.moe_host_expert_bytes.empty() && cparams.ctx_type != LLAMA_CONTEXT_TYPE_MTP) {
        bool any_slab = false;
        for (const auto & kv : model.moe_host_expert_bytes) {
            const size_t cur = ggml_backend_dev_slab_work_size(kv.first);
            if (cur == 0) {
                continue;   // this device has no slab to reclaim the wide layout with
            }
            any_slab = true;
            drop_observed = std::max(drop_observed, cur);
            const auto it = slab_narrow_boundary.find(kv.first);
            const size_t was = it == slab_narrow_boundary.end() ? 0 : it->second;
            // A material difference only: the narrow layout varies a little between a 1-token and a
            // `n_rs_batch`-token verify pass, and dropping for that would thrash the layout every token.
            // A wide -> narrow transition ALWAYS drops.  `slab_narrow_boundary` is recorded as
            // `max(re-reserved narrow, drop_observed)`, and at a prefill -> decode transition
            // `drop_observed` is the WIDE prefill region, so an equally-wide SECOND prefill would compare
            // equal and never release the wide layout or re-arm the arena (r85 regression: the second
            // request keeps the wide work buffer and a small arena, decode ~half speed).
            if (n_tokens_prev > 8 || cur >= was + (size_t) GGML_DROP_RESERVE_SLACK_BYTES) {
                drop_now = true;
                break;
            }
        }
        if (!any_slab) {
            drop_now = n_tokens_prev > 8;
        }
    }

    // wip/slab-ring-region (early narrow floor): the ring is one meta-unit with the WIDE (prefill) work
    // view, and the slab can only place it once the always-resident NARROW (decode/verify) floor is pinned.
    // That used to happen only inside the `drop_now` block below (the prefill -> decode transition), so
    // the FIRST prefill -- exactly the pass the ring exists for -- ran with `ring_region == 0` and fell
    // back to a raw cudaMalloc bounded by the slab's optional-allocation cap (measured: the 70K-token
    // IQ4_NL prefill runs `WORK_ALLOC ... narrow_floor=0.0 ring=0.0`, and the ring is only carved
    // afterwards).  Measure the narrow layout once with a size-only reserve and pin the floor BEFORE the
    // first wide view is placed, so the ring is carved for that prefill too.  The size-only reserve is
    // the same probe the `drop_now` path already uses (`split_only`, no allocation).
    if (!slab_narrow_pinned && !model.moe_host_expert_bytes.empty() &&
            cparams.ctx_type != LLAMA_CONTEXT_TYPE_MTP) {
        slab_narrow_pinned = true;
        const uint32_t n_rs_seqs = std::max<uint32_t>(1, dsv4_cache ? ubatch.n_seqs : ubatch.n_seqs_unq);
        size_t narrow_sizes[16] = {0};
        // TODO #96: reserve the width the run will actually issue, not the unclamped safety bound
        const uint32_t n_verify = verify_batch_width();
        if (graph_reserve(n_verify, n_rs_seqs, n_verify, mctx, /*split_only=*/true,
                          narrow_sizes, kq_mask_packed_reachable()) != nullptr) {
            size_t narrow_bytes = 0;
            for (int i = 0; i < 16; i++) {
                if (narrow_sizes[i] > narrow_bytes) {
                    narrow_bytes = narrow_sizes[i];
                }
            }
            if (narrow_bytes > 0) {
                for (const auto & kv : model.moe_host_expert_bytes) {
                    ggml_backend_dev_slab_narrow_floor(kv.first, narrow_bytes);
                }
                for (size_t d = 0; d < model.devices.size(); ++d) {
                    if (!model.devices[d].is_meta) {
                        ggml_backend_dev_slab_narrow_floor(model.devices[d].dev, narrow_bytes);
                    }
                }
            }
        }
    }

    // wip/slab-ring-region: the H2D staging ring is one meta-unit with the WIDE (prefill) compute layout.
    // It is armed for a wide pass (>= the staging gate floor, 64 tokens) and disarmed for a narrow one, so
    // a decode-only stretch returns its hole to the arena.  This runs BEFORE the drop's cache re-arm, so a
    // post-prefill re-arm sizes against the FULL (disarmed) arena instead of the ring-reduced one.  It does
    // NOT piggyback on the compute-buffer drop: the drop only fires when the work boundary actually grew,
    // while the ring is used on every wide pass.  `ring_set` is a no-op when the state already matches.
    if (!model.moe_host_expert_bytes.empty() && cparams.ctx_type != LLAMA_CONTEXT_TYPE_MTP) {
        const bool ring_arm = ubatch.n_tokens >= 64;
        // wip/slab-ring-region (issue #164): the narrow -> wide drop below exists only to force the wide
        // view to re-place ABOVE a carved ring hole so the ring arms.  When the slab carved no hole
        // (`ring_region == 0`; e.g. the requested hole does not fit under `mapped`), the drop has nothing
        // to arm and merely churns the compute layout -- measured ~13-16 % decode on a pinned-host-expert
        // MoE config, where the replayed HIP graph lost its replay benefit.  Only drop when a hole exists.
        bool ring_carved = false;
        for (const auto & kv : model.moe_host_expert_bytes) {
            ggml_backend_dev_slab_ring_set(kv.first, ring_arm);
            ring_carved = ring_carved || ggml_backend_dev_slab_ring_region(kv.first) > 0;
        }
        for (size_t d = 0; d < model.devices.size(); ++d) {
            if (!model.devices[d].is_meta) {
                ggml_backend_dev_slab_ring_set(model.devices[d].dev, ring_arm);
                ring_carved = ring_carved || ggml_backend_dev_slab_ring_region(model.devices[d].dev) > 0;
            }
        }
        // wip/slab-ring-region (prefill arming): the WIDE (prefill) view is what places the ring -- the
        // slab arms it when `boundary >= narrow_floor + ring_region` -- but the scheduler REUSES the
        // previous compute buffer whenever the new layout fits, and for these models the prefill and the
        // narrow (decode/verify) layout are the same size.  A prefill that follows a decode therefore
        // reused the narrow buffer at base 0, never re-placed a wide view, and the carved ring stayed
        // unarmed (`ring_slot_capacity() == 0`) -- the H2D staging fell back to a capped cudaMalloc and
        // the prefill streamed.  Release the narrow layout on a narrow -> wide transition so the prefill
        // re-reserves above the ring (`ring_set` has already marked the request wide).  Symmetric with the
        // `drop_now` wide -> narrow transition below; consecutive prefill ubatches (`n_tokens_prev >= 64`)
        // keep the wide buffer.
        if (ring_carved && ring_arm && n_tokens_prev < 64) {
            ggml_backend_sched_drop_buffers(sched.get());
        }
    }
    if (drop_now) {
        ggml_backend_sched_drop_buffers(sched.get());
        // re-reserve the widest post-prefill (verify) layout so the first decode graphs fit.
        // The reserve must use the CURRENT ubatch's sequence count, not a fixed 1: the memory context
        // (`mctx`) spans this ubatch's streams, and `build_attn_mha`/`ggml_flash_attn_ext` require the
        // query and the cache to agree on the stream dim (`q->ne[3] == k->ne[3]`).  A hard-coded 1
        // against a multi-sequence batch took `q->ne[2]/n_stream` to zero and aborted (issue #48).
        const uint32_t n_rs_seqs = std::max<uint32_t>(1, dsv4_cache ? ubatch.n_seqs : ubatch.n_seqs_unq);
        // wip/slab-ring-region: measure the narrow (decode/verify) layout BEFORE allocating it, so the slab
        // can pin its narrow floor AHEAD of the cache's deferred sizing (which runs later in this same pass).
        // `split_only` + a `sizes` array is a size-only reserve (no allocation); the real reserve follows.
        // TODO #96: the width is the band-limited one (`verify_batch_width`), so a live arena reserves for
        // the clamped draft while `n_rs_batch` stays the unclamped correctness bound.
        const uint32_t n_verify = verify_batch_width();
        {
            size_t narrow_sizes[16] = {0};
            if (graph_reserve(n_verify, n_rs_seqs, n_verify, mctx, /*split_only=*/true,
                              narrow_sizes, kq_mask_packed_reachable()) != nullptr) {
                size_t narrow_bytes = 0;
                for (int i = 0; i < 16; i++) {
                    if (narrow_sizes[i] > narrow_bytes) {
                        narrow_bytes = narrow_sizes[i];
                    }
                }
                if (narrow_bytes > 0) {
                    for (const auto & kv : model.moe_host_expert_bytes) {
                        ggml_backend_dev_slab_narrow_floor(kv.first, narrow_bytes);
                    }
                    for (size_t d = 0; d < model.devices.size(); ++d) {
                        if (!model.devices[d].is_meta) {
                            ggml_backend_dev_slab_narrow_floor(model.devices[d].dev, narrow_bytes);
                        }
                    }
                }
            }
        }
        const bool narrow_ok = graph_reserve(n_verify, n_rs_seqs, n_verify, mctx, false) != nullptr;
        if (!narrow_ok) {
            LLAMA_LOG_WARN("%s: failed to re-reserve the post-prefill compute layout; a later"
                    " growth may not fit next to the MoE expert cache arena\n", __func__);
        } else if (!model.hparams.no_alloc) {
            // `~llama_context` compares the CURRENT sched buffer sizes against `backend_buf_exp_size`,
            // which `sched_reserve()` captured for the WIDEST layout.  The drop legitimately narrows the
            // buffers, so refresh the expectation to the layout this workload settled on -- otherwise
            // every dropped run reports a healthy buffer as a mismatch.  (The field's only other reader is
            // the no-alloc memory accounting, which this branch skips.)
            size_t narrow_bytes = 0;
            for (size_t i = 0; i < backend_ptrs.size(); ++i) {
                backend_buf_exp_size[i] = ggml_backend_sched_get_buffer_size(sched.get(), backend_ptrs[i]);
                if (backend_buf_exp_size[i] > narrow_bytes) {
                    narrow_bytes = backend_buf_exp_size[i];
                }
            }
            // wip/slab-ring-region: pin the narrow (decode/verify) work floor so the slab places the H2D
            // staging ring just above it (and reclaims it with the wide view on a decode stretch).
            if (narrow_bytes > 0) {
                for (const auto & kv : model.moe_host_expert_bytes) {
                    ggml_backend_dev_slab_narrow_floor(kv.first, narrow_bytes);
                }
                for (size_t d = 0; d < model.devices.size(); ++d) {
                    if (!model.devices[d].is_meta) {
                        ggml_backend_dev_slab_narrow_floor(model.devices[d].dev, narrow_bytes);
                    }
                }
            }
        }
        // Record the work region's size AFTER the narrow re-reserve: that is the steady state this workload
        // settled on, and the value the next pass is compared against.  Take the MAX with the region
        // actually observed before the drop, because the re-reserve here is not the widest narrow graph
        // (`n_rs_batch` vs the MTP decode/verify layout).
        for (const auto & kv : model.moe_host_expert_bytes) {
            const size_t cur = ggml_backend_dev_slab_work_size(kv.first);
            const size_t settled = std::max(cur, drop_observed);
            if (settled > 0) {
                slab_narrow_boundary[kv.first] = settled;
            }
        }
        // OPEN 2 (TODO #42): re-arm the MoE expert-cache arena with the VRAM the wide layout just
        // released.  The arena was culled to make room for the wide compute buffer; the surviving tables
        // were never freed and the cache is merely bypassed while any table is down (the
        // wholesale-fallback invariant), so re-allocating the stood-down tables restores the cache WITH
        // its residents -- not cold -- for the decode that follows.  Idempotent: the first call does the
        // work, the rest are no-ops.  The narrow layout is reserved first, so the arena takes only what is
        // left free.
        for (const auto & kv : model.moe_host_expert_bytes) {
            (void) ggml_backend_dev_moe_cache_rearm(kv.first);
        }
        for (size_t d = 0; d < model.devices.size(); ++d) {
            if (!model.devices[d].is_meta) {
                (void) ggml_backend_dev_moe_cache_rearm(model.devices[d].dev);
            }
        }
        // Extra arena reserve.  DEFAULT 0: the compute-buffer margin covers the small post-drop growth
        // and the layer-uniform re-size absorbs arena fragmentation, so holding VRAM out only costs
        // arena.  `LLAMA_DROP_EXTRA_RESERVE_MIB=N` raises it (per device) for A/B.
        const char * extra_mib_env = getenv("LLAMA_DROP_EXTRA_RESERVE_MIB");
        const int64_t extra_mib = extra_mib_env != nullptr ? atoll(extra_mib_env) : 0;
        if (extra_mib > 0) {
            // Under -sm tensor the scheduler's backends are Meta wrappers, so set the reserve on the
            // model's real (host-expert-owning) devices, which own the MoE expert cache.
            const size_t extra_bytes = (size_t) extra_mib * 1024 * 1024;
            for (const auto & kv : model.moe_host_expert_bytes) {
                ggml_backend_dev_moe_cache_set_reserve(kv.first, extra_bytes);
            }
            for (size_t d = 0; d < model.devices.size(); ++d) {
                if (!model.devices[d].is_meta) {
                    ggml_backend_dev_moe_cache_set_reserve(model.devices[d].dev, extra_bytes);
                }
            }
        }
    }

    const bool alt = sched_alt_for(ubatch);
    sched_alt_select(alt);

    auto * res = get_gf_res_prev(alt);
    auto * gf  = res->get_gf();

    // the new graph parameters
    // in order to correctly reuse a graph, it's full topology has to be uniquely determined by these parameters
    const auto gparams = graph_params(res, ubatch, mctx, gtype);

    if (!graph_reuse_disable && gf_res_prev_active == res && res->can_reuse(gparams)) {
        //LLAMA_LOG_DEBUG("%s: reusing previous graph\n", __func__);

        // with pipeline parallelism, the previous graph_compute_async may still be running
        // on the GPU. we must synchronize before set_inputs to avoid overwriting input tensors
        // that the previous compute is still reading.
        if (cparams.pipeline_parallel) {
            ggml_backend_sched_synchronize(sched.get());
        }

        n_reused++;
    } else {
        gf_res_prev_active = nullptr;
        res->reset();

        ggml_backend_sched_reset(sched.get());
        ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);

        //const auto t_start_us = ggml_time_us();

        gf = model.build_graph(gparams);

        //LLAMA_LOG_INFO("graph build time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);

        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to initialize graph\n", __func__);
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }

        // wip/moe-verify-fusions (narrow-2): route the MTP draft context's compute allocation to the
        // slab's pinned narrow-2 region (the reserve-time toggle in `graph_reserve` does not cover the
        // first `alloc_graph`).
        const bool is_draft = cparams.ctx_other != nullptr;
        if (is_draft) {
            for (const auto & d : model.devices) {
                ggml_backend_dev_slab_compute_narrow2(d.dev, true);
                // the second scheduler's small buffer is never served by the slab (narrow-2 holds one view)
                ggml_backend_dev_slab_compute_plain(d.dev, sched_alt_in_use);
            }
        }
        const bool alloc_ok = ggml_backend_sched_alloc_graph(sched.get(), gf);
        if (is_draft) {
            for (const auto & d : model.devices) {
                ggml_backend_dev_slab_compute_narrow2(d.dev, false);
                ggml_backend_dev_slab_compute_plain(d.dev, false);
            }
        }
        if (!alloc_ok) {
            LLAMA_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            ret = GGML_STATUS_ALLOC_FAILED;
            return nullptr;
        }

        gf_res_prev_active = res;
    }

    // set the input data for the input tensors
    {
        //const auto t_start_us = ggml_time_us();

        // FIXME this call causes a crash if any model inputs were not used in the graph and were therefore not allocated
        res->set_inputs(&ubatch);

        // F1 (issue #240): the layer-input stores are indexed by the token's row in this decode batch, and the
        // graph writes them (one SET_ROWS per extracted layer).  The row indices are the only per-ubatch data
        // there, so a graph with the same shape stays reusable across ubatches at different offsets.
        if (ggml_tensor * rows = res->get_lf_rows()) {
            GGML_ASSERT(rows->ne[0] == (int64_t) ubatch.n_tokens);
            lf_row_idx.resize(ubatch.n_tokens);
            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                lf_row_idx[i] = cparams.lf_row_offset + (int32_t) i;
            }
            ggml_backend_tensor_set(rows, lf_row_idx.data(), 0, ubatch.n_tokens*sizeof(int32_t));
        }

        //LLAMA_LOG_INFO("graph set inputs time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);
    }

    // F1 (issue #240): the graph writes the layer-input store rows itself, so the wait for the draft's read of the
    // previous batch has to be ordered before its writes, not before a post-graph copy.
    if (!cparams.lf_layer_inp_dev_store.empty()) {
        lf_wait_features_consumed();
    }

    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: failed to compute graph, compute status: %d\n", __func__, status);
        ret = status;
        return nullptr;
    }

    n_tokens_prev = ubatch.n_tokens;
    ret = GGML_STATUS_SUCCESS;

    return res;
}

int llama_context::encode(const llama_batch_ext & batch_inp) {
    // the draft context's second scheduler is swapped into `sched` only while this call runs
    struct sched_alt_restore {
        llama_context * ctx;
        ~sched_alt_restore() { ctx->sched_alt_select(false); }
    } sched_alt_restore_ { this };

    if (batch_inp.tokens.empty()) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    const auto & hparams = model.hparams;

    if (batch_inp.n_embd > 0 && batch_inp.n_embd != hparams.n_embd_inp_enc()) {
        LLAMA_LOG_ERROR("%s: embd row width %zu does not match the encoder input %u\n",
                __func__, batch_inp.n_embd, hparams.n_embd_inp_enc());
        return -1;
    }

    // eagle3/DFlash: features as encoder input, and non-draft paths fall back to model's input dim
    const int64_t n_vocab = model.vocab.n_tokens();

    // note: during encode, we always output all tokens and skip position continuity checks (output_all=true)
    if (!balloc->init(batch_inp, model.vocab, true)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens = balloc->get_n_tokens();

    // [TAG_NO_CACHE_PAD]
    // TODO: add new split mode where we pad the input sequences so that ubatch.equal_seqs == true
    const llama_ubatch ubatch = balloc->split_simple(n_tokens);

    // micro-batching is not possible for non-causal encoding, so we process the batch in a single shot
    GGML_ASSERT(cparams.n_ubatch >= n_tokens && "encoder requires n_ubatch >= n_tokens");

    // TODO: this clear of the buffer can easily be forgotten - need something better
    // sync first so any in-flight async copies into embd_seq complete before it is freed
    if (!embd_seq.empty()) {
        synchronize();
    }
    embd_seq.clear();

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }

    sched_reserve();

    n_queued_tokens += n_tokens;

    // reserve output buffer
    if (output_reserve(n_tokens) < n_tokens) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %u outputs\n", __func__, n_tokens);
        return -2;
    };

    for (uint32_t i = 0; i < n_tokens; ++i) {
        output_ids[i] = i;
    }

    n_outputs = n_tokens;

    const auto causal_attn_org = cparams.causal_attn;

    // always use non-causal attention for encoder graphs
    // TODO: this is a tmp solution until we have a proper way to support enc-dec models
    //       ref: https://github.com/ggml-org/llama.cpp/pull/12181#issuecomment-2730451223
    cparams.causal_attn = false;

    ggml_status status;
    const auto * res = process_ubatch(ubatch, LLM_GRAPH_TYPE_ENCODER, nullptr, status);

    cparams.causal_attn = causal_attn_org;

    if (!res) {
        switch (status) {
            case GGML_STATUS_ABORTED:      return  2;
            case GGML_STATUS_ALLOC_FAILED: return -2;
            case GGML_STATUS_FAILED:       return -3;
            case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
        }
    }

    auto * t_logits  = res->get_logits();
    auto * t_embd    = res->get_embd_pooled() ? res->get_embd_pooled() : res->get_embd();
    auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn() : nullptr;

    // extract logits
    if (logits.data && t_logits) {
        ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
        GGML_ASSERT(backend_res != nullptr);
        GGML_ASSERT(logits.data != nullptr);

        ggml_backend_tensor_get_async(backend_res, t_logits, logits.data, 0, n_tokens*n_vocab*sizeof(float));
    }

    // extract embeddings
    if (embd.data && t_embd) {
        ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
        GGML_ASSERT(backend_embd != nullptr);

        switch (cparams.pooling_type) {
            case LLAMA_POOLING_TYPE_NONE:
                {
                    // extract token embeddings
                    GGML_ASSERT(embd.data != nullptr);
                    const uint32_t n_embd_out = hparams.n_embd_out();

                    GGML_ASSERT(n_tokens*n_embd_out <= (int64_t) embd.size);
                    ggml_backend_tensor_get_async(backend_embd, t_embd, embd.data, 0, n_tokens*n_embd_out*sizeof(float));
                } break;
            case LLAMA_POOLING_TYPE_MEAN:
            case LLAMA_POOLING_TYPE_CLS:
            case LLAMA_POOLING_TYPE_LAST:
                {
                    // extract sequence embeddings
                    auto & embd_seq_out = embd_seq;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        embd_seq_out[seq_id].resize(n_embd_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_RANK:
                {
                    // extract the rerank score - n_cls_out floats per sequence
                    auto & embd_seq_out = embd_seq;

                    const uint32_t n_cls_out = hparams.n_cls_out;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        embd_seq_out[seq_id].resize(n_cls_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_UNSPECIFIED:
                {
                    GGML_ABORT("unknown pooling type");
                }
        }
    }

    // extract nextn embeddings (hidden state before the final output norm)
    if (embd_nextn.data && t_h_nextn && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
        ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), t_h_nextn);
        GGML_ASSERT(backend_h != nullptr);

        const uint32_t n_embd = hparams.n_embd_out();
        GGML_ASSERT(n_tokens*n_embd <= (int64_t) embd_nextn.size);
        ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn.data, 0, n_tokens*n_embd*sizeof(float));
    }

    // TODO: hacky solution
    if (model.arch == LLM_ARCH_T5 && t_embd) {
        //cross.t_embd = t_embd;

        synchronize();

        cross.n_embd = t_embd->ne[0];
        cross.n_enc  = t_embd->ne[1];
        cross.v_embd.resize(cross.n_embd*cross.n_enc);
        memcpy(cross.v_embd.data(), embd.data, ggml_nbytes(t_embd));

        const auto & batch = balloc->get_batch();

        // remember the sequence ids used during the encoding - needed for cross attention later
        cross.seq_ids_enc.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; i++) {
            cross.seq_ids_enc[i].clear();

            for (int s = 0; s < batch.n_seq_id[i]; s++) {
                const llama_seq_id seq_id = batch.seq_id[i][s];

                cross.seq_ids_enc[i].insert(seq_id);
            }
        }
    }

    return 0;
}

template<typename T>
static void copy_tensor_async_rows(
    const std::vector<ggml_tensor *> & tensors,
    const buffer_view<T> & dst,
    size_t stride,
    uint32_t row_offset,
    ggml_backend_sched_t sched,
    std::vector<uint32_t> * counts = nullptr) {
    if (!dst.has_data()) {
        return;
    }

    for (size_t i = 0; i < tensors.size(); ++i) {
        auto * tensor = tensors[i];
        if (tensor == nullptr) {
            continue;
        }

        const uint32_t row = row_offset + i;
        const size_t n_elements = ggml_nelements(tensor);
        GGML_ASSERT(ggml_is_contiguous(tensor) && "sampling tensor must be contiguous for async copy");
        GGML_ASSERT(n_elements <= stride);
        GGML_ASSERT((size_t) row * stride + n_elements <= dst.size);

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        T * row_ptr = dst.data + (size_t) row * stride;
        ggml_backend_tensor_get_async(backend, tensor, row_ptr, 0, ggml_nbytes(tensor));

        if (counts) {
            GGML_ASSERT(row < counts->size());
            (*counts)[row] = n_elements;
        }
    }
}

static bool needs_raw_logits(const llama_ubatch & ubatch, const std::map<llama_seq_id, llama_sampler *> & samplers) {
    for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
        if (!ubatch.output[i]) {
            continue;
        }

        // Check if the output token has at least one sequence without a backend sampler.
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
            llama_seq_id seq_id = ubatch.seq_id[i][j];
            if (samplers.find(seq_id) == samplers.end()) {
                return true;
            }
        }
    }
    return false; // all sequences use backend sampling
}

int llama_context::decode(const llama_batch_ext & batch_inp) {
    // the draft context's second scheduler is swapped into `sched` only while this call runs
    struct sched_alt_restore {
        llama_context * ctx;
        ~sched_alt_restore() { ctx->sched_alt_select(false); }
    } sched_alt_restore_ { this };

    if (!memory) {
        LLAMA_LOG_DEBUG("%s: cannot decode batches with this context (calling encode() instead)\n", __func__);
        return encode(batch_inp);
    }

    if (batch_inp.tokens.empty()) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    if (batch_inp.n_embd > 0 && batch_inp.n_embd != batch_inp.n_embd_inp) {
        LLAMA_LOG_ERROR("%s: embd row width %zu does not match the decoder input %zu\n",
                __func__, batch_inp.n_embd, batch_inp.n_embd_inp);
        return -1;
    }

    const auto & vocab   = model.vocab;
    const auto & hparams = model.hparams;

    const int64_t n_vocab = vocab.n_tokens();

    // when computing embeddings, all tokens are output
    const bool output_all   = cparams.embeddings;
    // non-causal attention needs the whole sequence in one ubatch
    const bool embd_all     = output_all || !cparams.causal_attn;
    const bool has_samplers = !sampling.samplers.empty();

    const uint32_t n_seq_max = cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max;

    // TODO: avoid this workaround in the future
    // embedding contexts output every token even when no token is explicitly marked as output
    if (has_samplers) {
        std::vector<int32_t> seq_output_count(n_seq_max, 0);

        for (const auto & tok : batch_inp.tokens) {
            if (!output_all && !tok.output) {
                continue;
            }

            for (auto seq_id : tok.seq_ids) {
                if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
                    continue;
                }

                seq_output_count[seq_id]++;
                auto sampler = sampling.samplers.find(seq_id);
                if (sampler != sampling.samplers.end() &&
                        seq_output_count[seq_id] > (int32_t) cparams.n_outputs_max_per_seq) {
                    LLAMA_LOG_ERROR("%s: backend sampling supports at most %u outputs per sequence "
                            "(seq_id %d had %d)\n", __func__, cparams.n_outputs_max_per_seq,
                            seq_id, seq_output_count[seq_id]);
                    return -1;
                }
            }
        }
    }

    if (!balloc->init(batch_inp, vocab, output_all)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens_all  = balloc->get_n_tokens();
    const uint32_t n_outputs_all = balloc->get_n_outputs();

    if (output_all) {
        // require that all tokens are output
        if (n_outputs_all != n_tokens_all) {
            LLAMA_LOG_ERROR("%s: pooled embedding requires that all tokens are output (n_outputs_all = %d, n_tokens_all = %d)\n",
                    __func__, n_outputs_all, n_tokens_all);
            return -1;
        }
    }

    GGML_ASSERT(n_tokens_all <= cparams.n_batch);

    GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >= n_tokens_all) && "non-causal attention requires n_ubatch >= n_tokens");

    // TODO: this clear of the buffer can easily be forgotten - need something better
    // sync first so any in-flight async copies into embd_seq complete before it is freed
    if (!embd_seq.empty()) {
        synchronize();
    }
    embd_seq.clear();

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }
    n_queued_tokens += n_tokens_all;

    output_swaps.clear();
    embd_batch_idxs.clear();
    lf_dev_rows.clear();

    // F1 (issue #240): the device layer-input stores must exist before the first graph that writes them, so that
    // the first reserve already sizes the compute buffers without the (much larger) output shape.  This is a
    // no-op after the first call.
    lf_layer_inp_dev_arm();

    // issue #238: a 2-D (M-RoPE) position can only come from a multimodal chunk, and the compute reserve
    // covers the packed kq mask for such a batch only when the caller declared it at context creation.
    // When one arrives from a caller that did not, re-reserve ONCE here (before this batch is split into
    // ubatches) instead of leaving the graph to grow a buffer mid-flight - the packed mask is 2 bytes per
    // KV cell per ubatch token.  A text-only context never reaches this.  See kq_mask_packed_reachable().
    if (!kq_mask_pos_2d_seen && !cparams.kq_mask_pos_2d && cparams.kq_mask_packed_reserve != 0 &&
            model.hparams.n_pos_per_embd() > 1 && batch_has_pos_2d(batch_inp)) {
        kq_mask_pos_2d_seen = true;
        sched_need_reserve = true;
        LLAMA_LOG_INFO("%s: 2-D (M-RoPE) positions in this batch - re-reserving with the packed kq mask\n", __func__);
    }

    // the same for a batch of several sequences the derived kq mask cannot serve together
    if (!kq_mask_multiseq_seen && cparams.kq_mask_packed_reserve != 0 && cparams.n_seq_max > 1 && cparams.kq_mask_derived &&
            llama_kv_cache::kq_mask_derived_multiseq_enabled() && batch_needs_packed_multiseq(batch_inp, kq_mask_kvs())) {
        kq_mask_multiseq_seen = true;
        sched_need_reserve = true;
        LLAMA_LOG_INFO("%s: a batch of several sequences the derived kq mask cannot serve - re-reserving with the packed kq mask\n", __func__);
    }

    sched_reserve();

    bool did_optimize = false;

    // handle any pending shifts/copies
    memory_update(false);

    llama_memory_context_ptr mctx;

    while (true) {
        mctx = memory->init_batch(*balloc, cparams.n_ubatch, embd_all);
        if (!mctx) {
            return -2;
        }

        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    LLAMA_LOG_ERROR("%s: unexpected memory context status: %d\n", __func__, mctx->get_status());

                    return -2;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
                {
                    if (!did_optimize) {
                        did_optimize = true;

                        if (memory_update(true)) {
                            LLAMA_LOG_DEBUG("%s: retrying batch size %d after cache optimization\n", __func__, balloc->get_n_tokens());

                            continue;
                        }
                    }

                    LLAMA_LOG_WARN("%s: failed to find a memory slot for batch of size %d\n", __func__, balloc->get_n_tokens());

                    return 1;
                }
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: compute failed while preparing batch of size %d\n", __func__, balloc->get_n_tokens());

                    return -2;
                }
        }

        break;
    }

    // reserve output buffer
    if (output_reserve(n_outputs_all) < n_outputs_all) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
        return -2;
    };

    // start a new sampling transaction for this logical batch
    for (const auto & entry : sampling.samplers) {
        llama_sampler_backend_begin(entry.second);
    }

    int64_t n_outputs_prev = 0;
    int64_t n_tokens_prev_l  = 0;

    // LLAMA_MTP_TAIL_READBACK=0: disable the prompt-tail readback skip on both halves (this ubatch's nextn copy and
    // the MTP hook's early return, which is what stops llama_get_embeddings_nextn() from draining the backends).  That
    // is the pre-r118 behaviour with LLAMA_MTP_INGEST_TAIL still set.  Bisect/debug kill-switch; the skip is on by
    // default, and without LLAMA_MTP_INGEST_TAIL no row's threshold is above 0, so it cannot trigger.
    static const bool nextn_readback_skip = [] {
        const char * e = getenv("LLAMA_MTP_TAIL_READBACK");
        return e == nullptr || atoi(e) != 0;
    }();

    // complete the async device->host readbacks issued for one ubatch
    auto sync_readback = [&](ggml_tensor * t) {
        if (t == nullptr) {
            return;
        }
        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched.get(), t);
        if (backend != nullptr) {
            ggml_backend_synchronize(backend);
        }
    };

    do {
        const auto & ubatch = mctx->get_ubatch();

        // count the outputs in this ubatch
        {
            int32_t n_outputs_new = 0;

            if (n_outputs_all == n_tokens_all) {
                n_outputs_new = ubatch.n_tokens;
            } else {
                for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
                    n_outputs_new += (int32_t) (ubatch.output[i] != 0);
                }
            }

            // needs to happen before the graph is built
            n_outputs = n_outputs_new;
        }

        // F1 (issue #240): first row of this ubatch in the decode batch = where its layer inputs are stored
        cparams.lf_row_offset = (int32_t) n_tokens_prev_l;

        ggml_status status;

        const auto * res = process_ubatch(ubatch, ctx_type_to_graph_type(cparams.ctx_type), mctx.get(), status);

        if (!res) {
            // the last ubatch failed or was aborted -> remove all positions of that ubatch from the memory module
            llama_pos pos_min[LLAMA_MAX_SEQ];
            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                pos_min[s] = std::numeric_limits<llama_pos>::max();
            }

            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                const auto & seq_id = ubatch.seq_id[i][0];

                pos_min[seq_id] = std::min(pos_min[seq_id], ubatch.pos[i]);
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (pos_min[s] == std::numeric_limits<llama_pos>::max()) {
                    continue;
                }

                LLAMA_LOG_WARN("%s: removing memory module entries for seq_id = %d, pos = [%d, +inf)\n", __func__, s, pos_min[s]);

                memory->seq_rm(s, pos_min[s], -1);
            }

            switch (status) {
                case GGML_STATUS_ABORTED:      return  2;
                case GGML_STATUS_ALLOC_FAILED: return -2;
                case GGML_STATUS_FAILED:       return -3;
                case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
            }
        }

        // plot the computation graph in dot format (for debugging purposes)
        //if (n_past%100 == 0) {
        //    ggml_graph_dump_dot(gf, NULL, "llama.dot");
        //}

        auto * t_logits  = res->get_logits();
        auto * t_embd    = cparams.embeddings       ? res->get_embd()     : nullptr;
        auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn()  : nullptr;

        if (t_embd && res->get_embd_pooled()) {
            t_embd = res->get_embd_pooled();
        }

        // extract logits
        if (logits.data && t_logits && n_outputs > 0 && needs_raw_logits(ubatch, sampling.samplers)) {
            ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
            GGML_ASSERT(backend_res != nullptr);
            GGML_ASSERT(logits.data != nullptr);

            float * logits_out = logits.data + n_outputs_prev*n_vocab;

            if (n_outputs) {
                GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                GGML_ASSERT((n_outputs_prev + n_outputs)*n_vocab <= (int64_t) logits.size);
                ggml_backend_tensor_get_async(backend_res, t_logits, logits_out, 0, n_outputs*n_vocab*sizeof(float));
            }
        }

        // extract embeddings
        if (embd.data && t_embd && n_outputs > 0) {
            ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
            GGML_ASSERT(backend_embd != nullptr);

            switch (cparams.pooling_type) {
                case LLAMA_POOLING_TYPE_NONE:
                    {
                        // extract token embeddings
                        GGML_ASSERT(embd.data != nullptr);
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        float * embd_out = embd.data + n_outputs_prev*n_embd_out;

                        if (n_outputs) {
                            GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                            GGML_ASSERT((n_outputs_prev + n_outputs)*n_embd_out <= (int64_t) embd.size);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_out, 0, n_outputs*n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_MEAN:
                case LLAMA_POOLING_TYPE_CLS:
                case LLAMA_POOLING_TYPE_LAST:
                    {
                        // extract sequence embeddings (cleared before processing each batch)
                        auto & embd_seq_out = embd_seq;

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_embd_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_RANK:
                    {
                        // extract the rerank score - n_cls_out floats per sequence
                        auto & embd_seq_out = embd_seq;

                        const uint32_t n_cls_out = hparams.n_cls_out;

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_cls_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_UNSPECIFIED:
                    {
                        GGML_ABORT("unknown pooling type");
                    }
            }
        }

        // [TAG_EXTRACT_TARGET_EMBEDDINGS]
        // F1 (issue #240): an embedding (image/audio) ubatch has no token ids, so the draft's device injection cannot
        // address its rows and it reads the host rows instead - fill them too, and reserve them if this is the first
        // such ubatch (the arm released the rows the target's warm-up had allocated).
        const bool layer_inp_host_fill = cparams.lf_layer_inp_dev_store.empty() || ubatch.embd != nullptr;
        bool extract_all_idxs = extract_layer_inputs(res, n_tokens_prev_l, ubatch.n_tokens, ubatch.embd != nullptr);
        const bool layer_inp_read = extract_all_idxs;
        bool nextn_read = false;

        // extract nextn embeddings before
        // only meaningful in LLAMA_POOLING_TYPE_NONE (per-token); other pooling modes are ignored.
        {
            const bool masked    = cparams.embeddings_nextn_masked;
            const int64_t n_rows = masked ? n_outputs       : (int64_t) ubatch.n_tokens;
            const int64_t offset = masked ? n_outputs_prev  : n_tokens_prev_l;
            const bool nextn_rows = embd_nextn.data && t_h_nextn && n_rows > 0 &&
                                    cparams.pooling_type == LLAMA_POOLING_TYPE_NONE;

            // set_embeddings_nextn_from: an unmasked ubatch whose every row is below its sequence's threshold is not
            // read back (its host rows stay stale).  Embedding (vision) rows are always read: under M-RoPE the text
            // after an image resumes at a jumped section-0 position, so the last image row can be the h(p - 1) of an
            // ingested row although its own position is below the threshold.  The scan runs only when the nextn rows
            // are extracted at all, so a decode without them (every non-MTP run) is untouched.
            bool nextn_unneeded = false;
            if (nextn_rows && !masked && nextn_readback_skip) {
                nextn_unneeded = true;
                for (uint32_t i = 0; i < ubatch.n_tokens && nextn_unneeded; ++i) {
                    if (ubatch.type ? ubatch.type[i] != 0 : ubatch.token == nullptr) {
                        nextn_unneeded = false;
                        break;
                    }
                    for (int32_t j = 0; j < ubatch.n_seq_id[i] && nextn_unneeded; ++j) {
                        const llama_seq_id s = ubatch.seq_id[i][j];
                        nextn_unneeded = s >= 0 && s < LLAMA_MAX_SEQ && ubatch.pos[i] < embd_nextn_from[s];
                    }
                }
            }

            if (nextn_rows && nextn_unneeded) {
                // the host row bookkeeping below is kept: embd_batch_idxs must stay aligned with the dense unmasked
                // host rows, or output_reorder()'s nextn permutation would place a later ubatch's rows wrongly
                extract_all_idxs = true;
            } else if (nextn_rows) {
                ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), t_h_nextn);
                GGML_ASSERT(backend_h != nullptr);

                const uint32_t n_embd  = hparams.n_embd_out();
                float * embd_nextn_out = embd_nextn.data + offset*n_embd;

                GGML_ASSERT((offset + n_rows)*n_embd <= (int64_t) embd_nextn.size);
                ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn_out, 0, n_rows*n_embd*sizeof(float));
                nextn_read = true;
                extract_all_idxs = extract_all_idxs || !masked;
            }
        }

        if (extract_all_idxs) {
            GGML_ASSERT(ubatch.data && ubatch.data->batch_idxs.size() == ubatch.n_tokens);
            GGML_ASSERT(embd_batch_idxs.size() == (size_t) n_tokens_prev_l);
            const auto & batch_idxs = ubatch.data->batch_idxs;
            embd_batch_idxs.insert(embd_batch_idxs.end(), batch_idxs.begin(), batch_idxs.end());
        }

        if (has_samplers) {
            const auto stride = n_vocab;

            // async copy the sampling data from the backend to the host
            copy_tensor_async_rows(res->t_sampled,        sampling.sampled,    1,      n_outputs_prev, sched.get());
            copy_tensor_async_rows(res->t_sampled_logits, sampling.logits,     stride, n_outputs_prev, sched.get(), &sampling.logits_count);
            copy_tensor_async_rows(res->t_sampled_probs,  sampling.probs,      stride, n_outputs_prev, sched.get(), &sampling.probs_count);
            copy_tensor_async_rows(res->t_candidates,     sampling.candidates, stride, n_outputs_prev, sched.get(), &sampling.candidates_count);
        }

        n_outputs_prev += n_outputs;
        n_tokens_prev_l  += ubatch.n_tokens;

        if (!mctx->next()) {
            break;
        }

        // Another ubatch follows and will reuse this graph's output tensors.  The readbacks above
        // copied them into the persistent host output buffers asynchronously, so make sure those
        // copies have completed before the next graph overwrites the source.  On backends whose
        // pinned async device->host copy is not ordered against the compute stream (observed on
        // ROCm) the next ubatch otherwise overwrites the source before an earlier copy reads it,
        // and the host rows end up holding a later ubatch's data (multi-ubatch per-token outputs:
        // perplexity / KLD, and the server's multi-sequence batches).  A single-ubatch decode has
        // nothing to order against and pays nothing.
        // The per-token readbacks (the DFlash layer inputs, the unmasked nextn embeddings) also run in a ubatch without
        // outputs (a prompt chunk that does not end its prompt), so they need the same wait there: the next ubatch's input
        // upload waits only for this graph's compute, not for the copies queued after it, and was observed (Windows, ROCm)
        // to overwrite a layer input before its copy had read it.
        static const bool readback_sync = getenv("GGML_READBACK_SYNC") == nullptr || atoi(getenv("GGML_READBACK_SYNC")) != 0;
        // F1 (issue #240): with the in-graph stores there is no layer-input readback to order (the copy that needed
        // it no longer exists, and syncing the device once per ubatch would serialize a multi-ubatch prefill) -
        // except for an embedding ubatch, which fills the host rows the draft reads.
        const bool layer_inp_host = layer_inp_read && layer_inp_host_fill;
        if (readback_sync && (n_outputs > 0 || layer_inp_host || nextn_read)) {
            sync_readback(t_logits);
            sync_readback(t_embd);
            sync_readback(t_h_nextn);
            if (layer_inp_host) {
                for (uint32_t il = 0; il < cparams.embeddings_layer_inp.size(); ++il) {
                    if (cparams.embeddings_layer_inp[il]) {
                        sync_readback(res->get_layer_inp((int) il));
                    }
                }
            }
            if (has_samplers) {
                for (ggml_tensor * t : res->t_sampled)        { sync_readback(t); }
                for (ggml_tensor * t : res->t_sampled_logits) { sync_readback(t); }
                for (ggml_tensor * t : res->t_sampled_probs)  { sync_readback(t); }
                for (ggml_tensor * t : res->t_candidates)     { sync_readback(t); }
            }
        }
    } while (true);

    // F1: the device buffers hold the rows in ubatch order; keep the inverse of embd_batch_idxs for the
    // draft (output_reorder() permutes only the host buffers, and clears embd_batch_idxs when it runs)
    if (cparams.lf_layer_inp_dev && !lf_layer_dev.empty()) {
        bool same = true;
        for (size_t d = 0; d < embd_batch_idxs.size() && same; ++d) {
            same = embd_batch_idxs[d] == (int32_t) d;
        }
        if (!same) {
            lf_dev_rows.assign(embd_batch_idxs.size(), -1);
            for (size_t d = 0; d < embd_batch_idxs.size(); ++d) {
                GGML_ASSERT(embd_batch_idxs[d] >= 0 && (size_t) embd_batch_idxs[d] < lf_dev_rows.size());
                lf_dev_rows[embd_batch_idxs[d]] = (int32_t) d;
            }
        }
    }

    // set to total number of outputs in the batch, for use in llama_get_logits_ith
    n_outputs = n_outputs_all;

    // set output mappings
    if (n_outputs > 0) {
        bool sorted_output = true;

        auto & out_ids = balloc->get_out_ids();

        GGML_ASSERT(out_ids.size() == (size_t) n_outputs);

        for (int64_t i = 0; i < n_outputs; ++i) {
            int64_t out_id = out_ids[i];
            output_ids[out_id] = i;
            if (out_id != i) {
                sorted_output = false;
            }
        }

        // make the outputs have the same order they had in the user-provided batch
        // note: this is mostly relevant for recurrent models atm
        if (!sorted_output && n_outputs > 1) {
            GGML_ASSERT((size_t) n_outputs == out_ids.size());

            // TODO: is there something more efficient which also minimizes swaps?
            // selection sort, to minimize swaps (from https://en.wikipedia.org/wiki/Selection_sort)
            for (uint32_t i = 0; i < n_outputs - 1; ++i) {
                uint32_t j_min = i;
                for (uint32_t j = i + 1; j < n_outputs; ++j) {
                    if (out_ids[j] < out_ids[j_min]) {
                        j_min = j;
                    }
                }
                if (j_min == i) {
                    continue;
                }
                std::swap(out_ids[i], out_ids[j_min]);

                // remember the swaps and apply them lazily upon logits/embeddings access
                output_swaps.push_back({ i, j_min });
            }

            std::fill(output_ids.begin(), output_ids.end(), -1);

            for (uint32_t i = 0; i < n_outputs; ++i) {
                output_ids[out_ids[i]] = i;
            }
        }
    }

    // wait for the computation to finish (automatically done when obtaining the model output)
    //synchronize();

    return 0;
}

//
// output
//

uint32_t llama_context::output_reserve(int32_t n_outputs_l) {
    const auto & hparams = model.hparams;
    const auto & vocab   = model.vocab;

    const int64_t n_outputs_max = std::max<int64_t>(n_outputs_l, n_seq_max());

    const auto n_batch    = cparams.n_batch;
    const auto n_vocab    = vocab.n_tokens();
    const auto n_embd_out = hparams.n_embd_out();

    bool has_logits     = !cparams.embeddings || graph_has_logits;
    bool has_embd       = cparams.embeddings;
    bool has_embd_nextn = cparams.embeddings_nextn;

    // TODO: hacky enc-dec support
    if (model.arch == LLM_ARCH_T5) {
        has_logits = true;
        has_embd   = true;
    }

    size_t backend_float_count = 0;
    size_t backend_token_count = 0;

    logits.size     = has_logits     ? n_vocab*n_outputs_max     : 0;
    embd.size       = has_embd       ? n_embd_out*n_outputs_max  : 0;
    embd_nextn.size = has_embd_nextn ? n_embd_out*n_outputs_max  : 0;

    if (has_embd_nextn && !cparams.embeddings_nextn_masked) {
        // unmasked: nextn row exists for every token in the batch, not just
        // those flagged via batch.logits[i] -> size by token count instead.
        embd_nextn.size = (size_t) n_embd_out * n_batch;
    }

    // issue #240: the host layer-input rows are NOT part of the output buffer.  They are reserved lazily, by
    // the host extraction path only (layer_inp_host_reserve): with F1 the layer inputs never leave the device,
    // and `n_embd x n_batch` floats of pinned memory per extracted layer (200 MiB at -b 2048 on a 27B) were
    // allocated on every DFlash run and never read.

    // Allocate backend sampling output buffers if there are backend samplers configured.
    const bool has_sampling = !sampling.samplers.empty();
    if (has_sampling) {
        backend_float_count = 2 * n_vocab * n_outputs_max;      // logits + probs
        backend_token_count = (1 + n_vocab) * n_outputs_max;    // sampled + candidates
    }

    if (output_ids.empty()) {
        // init, never resized afterwards
        output_ids.resize(n_batch);
    }

    const size_t prev_size = buf_output ? ggml_backend_buffer_get_size(buf_output.get()) : 0;
    const size_t new_size  =
        (logits.size + embd.size + embd_nextn.size + backend_float_count) * sizeof(float) +
        (                                               backend_token_count) * sizeof(llama_token);

    // alloc only when more than the current capacity is required
    // TODO: also consider shrinking the buffer
    if (!buf_output || prev_size < new_size) {
        if (buf_output) {
#ifndef NDEBUG
            // This doesn't happen often, but may be annoying in some cases (like the HellaSwag benchmark)
            LLAMA_LOG_DEBUG("%s: reallocating output buffer from size %.02f MiB to %.02f MiB\n", __func__, prev_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
#endif
            synchronize();

            // TODO: not needed?
            buf_output = nullptr;
            logits.data = nullptr;
            embd.data = nullptr;
            embd_nextn.data = nullptr;
        }

        auto * buft = output_host_buft();
        buf_output.reset(ggml_backend_buft_alloc_buffer(buft, new_size));
        if (buf_output == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to allocate output buffer of size %.2f MiB\n", __func__, new_size / (1024.0 * 1024.0));
            return 0;
        }
        ggml_backend_buffer_clear(buf_output.get(), 0);
    }

    float * output_base = (float *) ggml_backend_buffer_get_base(buf_output.get());

    size_t offset = 0;
    uint8_t * base = (uint8_t *) output_base;

    logits = has_logits ? buffer_view<float>{output_base, logits.size} : buffer_view<float>{nullptr, 0};
    offset += logits.size * sizeof(float);

    embd = has_embd ? buffer_view<float>{(float *) (base + offset), embd.size} : buffer_view<float>{nullptr, 0};
    offset += embd.size * sizeof(float);

    embd_nextn = has_embd_nextn ? buffer_view<float>{(float *) (base + offset), embd_nextn.size} : buffer_view<float>{nullptr, 0};
    offset += embd_nextn.size * sizeof(float);

    if (has_sampling) {
        sampling.logits = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.logits.size * sizeof(float);

        sampling.probs = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.probs.size * sizeof(float);

        sampling.sampled = {(llama_token *) (base + offset), (size_t)n_outputs_max};
        offset += sampling.sampled.size * sizeof(llama_token);

        sampling.candidates = {(llama_token *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.candidates.size * sizeof(llama_token);

        // The count vectors keep track of the actual number of logits/probs/candidates
        // copied from the backend for each output row.

        sampling.logits_count.resize(n_outputs_max);
        sampling.probs_count.resize(n_outputs_max);
        sampling.candidates_count.resize(n_outputs_max);

        std::fill(sampling.logits_count.begin(),     sampling.logits_count.end(),     0);
        std::fill(sampling.probs_count.begin(),      sampling.probs_count.end(),      0);
        std::fill(sampling.candidates_count.begin(), sampling.candidates_count.end(), 0);

        std::fill_n(sampling.sampled.data, sampling.sampled.size, LLAMA_TOKEN_NULL);
    } else {
        sampling.logits     = {nullptr, 0};
        sampling.probs      = {nullptr, 0};
        sampling.sampled    = {nullptr, 0};
        sampling.candidates = {nullptr, 0};

        sampling.logits_count.clear();
        sampling.probs_count.clear();
        sampling.candidates_count.clear();
    }

    // set all ids as invalid (negative)
    std::fill(output_ids.begin(), output_ids.end(), -1);

    this->n_outputs = 0;

    GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max);

    return n_outputs_max;
}

void llama_context::set_lf_layer_inp_dev(bool enable) {
    cparams.lf_layer_inp_dev = enable;
}

ggml_tensor * llama_context::get_lf_layer_inp_dev(uint32_t lid) const {
    return lid < lf_layer_dev.size() ? lf_layer_dev[lid] : nullptr;
}

const int32_t * llama_context::get_lf_dev_rows(int32_t * n_rows) const {
    *n_rows = (int32_t) lf_dev_rows.size();
    return lf_dev_rows.empty() ? nullptr : lf_dev_rows.data();
}

void llama_context::set_lf_dev_inject(bool enable) {
    cparams.lf_dev_inject = enable;
}

ggml_backend_t llama_context::lf_backend_for(ggml_backend_dev_t dev) const {
    for (const auto & b : backends) {
        if (ggml_backend_get_device(b.get()) == dev) {
            return b.get();
        }
    }
    return nullptr;
}

void llama_context::lf_set_consumed_event(ggml_backend_t backend_reader) {
    if (!lf_layer_buf || backend_reader == nullptr) {
        return;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(backend_reader);
    if (lf_consumed_ev == nullptr) {
        lf_consumed_ev = ggml_backend_event_new(dev);
    }
    if (lf_consumed_ev == nullptr) {
        // no event support: fall back to a host wait
        ggml_backend_synchronize(backend_reader);
        return;
    }
    ggml_backend_event_record(lf_consumed_ev, backend_reader);
    lf_consumed_pending = true;
}

void llama_context::lf_signal_features_consumed() {
    llama_context * tgt = cparams.ctx_other;
    if (tgt == nullptr || !tgt->lf_layer_buf) {
        return;
    }
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tgt->lf_layer_buf.get()));
    ggml_backend_t reader = lf_backend_for(dev);
    if (reader == nullptr) {
        synchronize();
        return;
    }
    tgt->lf_set_consumed_event(reader);
}

// issue #240: with the in-graph stores armed the target graph writes this ubatch's rows itself, so the only thing
// left to order is the previous batch's read: the draft may still be gathering those rows on its own stream.
void llama_context::lf_wait_features_consumed() {
    if (!lf_consumed_pending) {
        return;
    }
    lf_consumed_pending = false;
    if (!lf_layer_buf) {
        return;
    }
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(lf_layer_buf.get()));
    ggml_backend_t backend = lf_backend_for(dev);
    if (backend == nullptr) {
        synchronize();
        return;
    }
    ggml_backend_event_wait(backend, lf_consumed_ev);
}

// F1 (issue #240): create the persistent device layer-input stores BEFORE the first graph that writes them.
// The stores used to be created lazily inside extract_layer_inputs, i.e. after the first target graph had already
// run - which forced that graph (and its compute-buffer reserve) to keep every layer input alive as a graph output.
// Creating them here means the first graph already writes them from inside the graph, and the compute buffer is
// reserved for the smaller shape from the start.
bool llama_context::lf_layer_inp_dev_arm() {
    if (lf_layer_dev_armed) {
        return !lf_layer_dev.empty();
    }
    if (!cparams.lf_layer_inp_dev) {
        return false;
    }
    lf_layer_dev_armed = true;

    const auto & enabled = cparams.embeddings_layer_inp;
    int il_first = -1;
    for (size_t il = 0; il < enabled.size(); ++il) {
        if (enabled[il]) {
            il_first = (int) il;
            break;
        }
    }
    if (il_first < 0) {
        return false;
    }

    const auto keep_host_path = [this](const char * why) {
        LLAMA_LOG_INFO("%s: F1: %s, using the host path\n", __func__, why);
        cparams.lf_layer_inp_dev = false;
        lf_layer_dev.clear();
        lf_layer_ctx.reset();
        return false;
    };

    // The store lives on the device that owns the first extracted layer: a layer input is produced by the layer
    // before it (the same device, except exactly at a split boundary, where the scheduler adds the cross-device
    // copy the post-graph extraction used to issue) and the draft gathers it there in place.
    const int n_layer_all = model.hparams.n_layer_all;
    ggml_backend_dev_t dev = il_first < n_layer_all ? model.dev_layer(il_first) : model.dev_output();
    if (dev == nullptr) {
        return keep_host_path("no device for the first extracted layer");
    }
    if (ggml_backend_dev_is_meta(dev)) {
        // `-sm tensor`: the layer inputs live in a Meta buffer, where one device pointer does not exist, and the
        // SET_ROWS would have to be served by the Meta split state.  The draft then gathers these features from
        // the host buffers instead (byte-identical output).
        return keep_host_path("layer inputs are in a Meta buffer (tensor split)");
    }
    if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        // a CPU backend is not guaranteed to implement SET_ROWS, and on the host the device path has nothing to win
        return keep_host_path("the first extracted layer is on the CPU");
    }
    ggml_backend_t backend = lf_backend_for(dev);
    if (backend == nullptr) {
        return keep_host_path("no backend for the first extracted layer's device");
    }

    const size_t n_layers = enabled.size();
    lf_layer_dev.assign(n_layers, nullptr);
    ggml_init_params ip = { ggml_tensor_overhead() * n_layers, nullptr, true };
    lf_layer_ctx.reset(ggml_init(ip));
    size_t need = 0;
    for (size_t l = 0; l < n_layers; ++l) {
        if (!enabled[l]) {
            continue;
        }
        lf_layer_dev[l] = ggml_new_tensor_2d(lf_layer_ctx.get(), GGML_TYPE_F32, model.hparams.n_embd, cparams.n_batch);
        ggml_format_name(lf_layer_dev[l], "lf_layer_inp_dev-%u", (uint32_t) l);
        need += ggml_nbytes(lf_layer_dev[l]);
    }
    lf_layer_buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(lf_layer_ctx.get(), ggml_backend_get_default_buffer_type(backend)));
    if (!lf_layer_buf) {
        // not enough device memory: stay on the host path for the rest of this context (the host buffers are
        // reserved on first use).  The draft sees llama_lf_get_layer_inp_dev() return null and gathers the
        // features from the host buffers instead, so the output does not change.
        LLAMA_LOG_WARN("%s: F1: could not allocate the device layer-input buffers (%.1f MiB), using the host path\n",
                __func__, need / (1024.0 * 1024.0));
        lf_layer_dev.clear();
        lf_layer_ctx.reset();
        cparams.lf_layer_inp_dev = false;
        return false;
    }
    LLAMA_LOG_INFO("%s: F1 device layer-input buffers: %.1f MiB\n", __func__,
            ggml_backend_buffer_get_size(lf_layer_buf.get()) / (1024.0 * 1024.0));

    // The graph the reserve measured kept every layer input alive as an output; with the stores the same graphs
    // are materially smaller, and a compute buffer only shrinks by dropping it first.
    static const bool lf_dev_graph = [] {
        const char * e = getenv("GGML_LF_DFLASH_DEV_GRAPH");
        return e == nullptr || atoi(e) != 0;
    }();
    if (lf_dev_graph) {
        cparams.lf_layer_inp_dev_store = lf_layer_dev;
    } else {
        cparams.lf_layer_inp_dev_store.clear();
    }
    synchronize();

    // issue #240: the target context is warmed up before the draft exists, and that warm-up took the host path (no
    // store yet).  Now that the layer inputs never leave the device, release those pinned rows instead of holding
    // them for the rest of the process.
    if (buf_layer_inp) {
        buf_layer_inp.reset();
        for (auto & layer_inp : embd_layer_inp) {
            layer_inp = buffer_view<float>{nullptr, 0};
        }
        LLAMA_LOG_DEBUG("%s: F1: released the unused host layer-input rows\n", __func__);
    }

    if (sched) {
        ggml_backend_sched_drop_buffers(sched.get());
    }
    sched_need_reserve = true;

    return true;
}

// issue #240: the pinned host rows of the host extraction path.  Reserved when that path runs, not at context
// creation: with F1 they are never read, and `n_embd x n_batch` floats per extracted layer is 200 MiB at -b 2048.
bool llama_context::layer_inp_host_reserve() {
    const size_t   n_embd  = model.hparams.n_embd;
    const uint32_t n_batch = cparams.n_batch;

    int il_first = -1;
    size_t n_bytes = 0;
    for (size_t il = 0; il < cparams.embeddings_layer_inp.size(); ++il) {
        if (!cparams.embeddings_layer_inp[il]) {
            continue;
        }
        if (il_first < 0) {
            il_first = (int) il;
        }
        n_bytes += (size_t) n_embd * n_batch * sizeof(float);
    }
    if (il_first < 0) {
        return true; // nothing enabled: nothing to reserve (and nothing to extract)
    }
    if (embd_layer_inp[il_first].has_data()) {
        return true; // already reserved (n_batch and n_embd are fixed for the context)
    }

    buf_layer_inp.reset(ggml_backend_buft_alloc_buffer(output_host_buft(), n_bytes));
    if (!buf_layer_inp) {
        LLAMA_LOG_ERROR("%s: failed to allocate the host layer-input buffer of size %.2f MiB\n",
                __func__, n_bytes / (1024.0 * 1024.0));
        return false;
    }
    ggml_backend_buffer_clear(buf_layer_inp.get(), 0);

    uint8_t * base = (uint8_t *) ggml_backend_buffer_get_base(buf_layer_inp.get());
    size_t offset = 0;
    for (uint32_t il = 0; il < embd_layer_inp.size(); ++il) {
        if (cparams.embeddings_layer_inp[il]) {
            embd_layer_inp[il] = buffer_view<float>{(float *) (base + offset), (size_t) n_embd * n_batch};
            offset += embd_layer_inp[il].size * sizeof(float);
        } else {
            embd_layer_inp[il] = buffer_view<float>{nullptr, 0};
        }
    }
    LLAMA_LOG_DEBUG("%s: host layer-input buffers: %.2f MiB\n", __func__, n_bytes / (1024.0 * 1024.0));

    return true;
}

// try to use the host buffer of the device where the output tensor is allocated for faster transfer to system memory
ggml_backend_buffer_type_t llama_context::output_host_buft() const {
    auto * buft = ggml_backend_cpu_buffer_type();

    auto * output_dev             = model.dev_output();
    auto * output_dev_host_buft   = output_dev ? ggml_backend_dev_host_buffer_type(output_dev) : nullptr;
    // A meta (tensor-split) device has no single host buffer type when its devices each have their own (the CUDA
    // per-device pinned types differ only in `device`).  The output buffer is never scheduled on, so the first
    // simple device's pinned type serves it: without it the per-step D2H reads of the outputs land in pageable
    // memory and each one blocks the host until the GPU is done.
    static const bool output_pinned = [] { const char * e = getenv("LLAMA_OUTPUT_PINNED"); return e == nullptr || atoi(e) != 0; }();
    if (output_pinned && output_dev_host_buft == nullptr && output_dev != nullptr && ggml_backend_dev_is_meta(output_dev) &&
            ggml_backend_meta_dev_n_devs(output_dev) > 0) {
        output_dev_host_buft = ggml_backend_dev_host_buffer_type(ggml_backend_meta_dev_simple_dev(output_dev, 0));
    }
    if (output_dev_host_buft) {
        buft = output_dev_host_buft;
    }

    return buft;
}

bool llama_context::extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens,
        bool host_also) {
    if (cparams.lf_layer_inp_dev && !lf_layer_dev.empty()) {
        if (cparams.lf_layer_inp_dev_store.empty()) {
            // `GGML_LF_DFLASH_DEV_GRAPH=0`: the pre-#240 shape.  The stores exist, but the layer inputs are still
            // graph outputs and are copied here, after the graph, as a device-to-device copy on the layer's own
            // backend.  Pure A/B/bisect arm.
            for (uint32_t il = 0; il < cparams.embeddings_layer_inp.size(); ++il) {
                if (!cparams.embeddings_layer_inp[il]) {
                    continue;
                }
                ggml_tensor * t = res->get_layer_inp((int) il);
                if (!t) {
                    GGML_ABORT("layer input tensor not found");
                }
                GGML_ASSERT(t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && t->ne[1] == (int64_t) n_tokens);
                ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched.get(), t);
                GGML_ASSERT(backend != nullptr);

                if (lf_consumed_pending) {
                    // the draft may still be reading the previous batch's rows on its own stream
                    ggml_backend_event_wait(backend, lf_consumed_ev);
                    lf_consumed_pending = false;
                }

                ggml_tensor * buf = lf_layer_dev[il];
                GGML_ASSERT(buf != nullptr && buf->ne[0] == t->ne[0] && token_offset + n_tokens <= (size_t) buf->ne[1]);
                ggml_tensor dst = *buf;
                dst.ne[1] = (int64_t) n_tokens;
                dst.nb[2] = dst.nb[1] * dst.ne[1];
                dst.nb[3] = dst.nb[2] * dst.ne[2];
                dst.data  = (char *) buf->data + token_offset * buf->nb[1];
                dst.view_src  = buf;
                dst.view_offs = token_offset * buf->nb[1];
                ggml_backend_tensor_copy_async(backend, backend, t, &dst);
            }
        } else if (!host_also) {
            // Llama-Frankenstein F1 (issue #240): the target graph wrote this ubatch's rows into the persistent
            // per-layer device stores itself (GGML_OP_SET_ROWS, emitted at the end of each extracted layer), so a
            // draft context reads them in place and there is nothing to copy.  The only ordering left is the wait
            // for the previous batch's reader, which now happens before the graph runs
            // (lf_wait_features_consumed), not before a copy that no longer exists.
            return true;
        }

        if (!host_also) {
            return true;
        }
        // an embedding ubatch: the draft reads the host rows, so fill them below as well
    }

    // no device store (arm() declined: Meta buffer, CPU layer, no backend, or the allocation failed) -> host

    bool extracted = false;

    if (!layer_inp_host_reserve()) {
        GGML_ABORT("output layer input buffer not allocated");
    }

    for (uint32_t il = 0; il < cparams.embeddings_layer_inp.size(); ++il) {
        if (!cparams.embeddings_layer_inp[il]) {
            continue;
        }
        if (!embd_layer_inp[il].has_data()) {
            GGML_ABORT("output layer input buffer not allocated");
        }
        ggml_tensor * t = res->get_layer_inp((int) il);
        if (!t) {
            GGML_ABORT("layer input tensor not found");
        }

        const size_t nbytes = ggml_nbytes(t);
        const size_t nfloats = nbytes / sizeof(float);
        GGML_ASSERT(n_tokens > 0);
        GGML_ASSERT(nfloats % n_tokens == 0);

        const size_t row_floats = nfloats / n_tokens;
        GGML_ASSERT(row_floats == model.hparams.n_embd);
        const size_t dst_offset = token_offset * row_floats;
        GGML_ASSERT(dst_offset + nfloats <= embd_layer_inp[il].size);

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched.get(), t);
        GGML_ASSERT(backend != nullptr);
        // Tensor-split backends require a zero source offset.
        ggml_backend_tensor_get_async(backend, t, embd_layer_inp[il].data + dst_offset, 0, nbytes);
        extracted = true;
    }
    return extracted;
}

void llama_context::output_reorder() {
    const uint64_t n_vocab     = model.vocab.n_tokens();
    const uint64_t n_embd      = model.hparams.n_embd;
    const uint64_t n_embd_out  = model.hparams.n_embd_out();

    for (size_t s = 0; s < output_swaps.size(); ++s) {
        const uint64_t i0 = output_swaps[s].i0;
        const uint64_t i1 = output_swaps[s].i1;

        if (logits.size > 0) {
            for (uint64_t k = 0; k < n_vocab; k++) {
                std::swap(logits.data[i0*n_vocab + k], logits.data[i1*n_vocab + k]);
            }
        }

        if (embd.size > 0) {
            for (uint64_t k = 0; k < n_embd_out; k++) {
                std::swap(embd.data[i0*n_embd_out + k], embd.data[i1*n_embd_out + k]);
            }
        }

        if (embd_nextn.size > 0 && cparams.embeddings_nextn_masked) {
            for (uint64_t k = 0; k < n_embd_out; k++) {
                std::swap(embd_nextn.data[i0*n_embd_out + k], embd_nextn.data[i1*n_embd_out + k]);
            }
        }

        if (!sampling.samplers.empty()) {
            assert(sampling.logits.size > 0);
            assert(sampling.probs.size > 0);
            assert(sampling.candidates.size > 0);
            assert(sampling.sampled.size > 0);
            assert(sampling.logits_count.size() > 0);
            assert(sampling.probs_count.size() > 0);
            assert(sampling.candidates_count.size() > 0);

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.logits.data[i0*n_vocab + k], sampling.logits.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.probs.data[i0*n_vocab + k], sampling.probs.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.candidates.data[i0*n_vocab + k], sampling.candidates.data[i1*n_vocab + k]);
            }

            std::swap(sampling.sampled.data[i0],     sampling.sampled.data[i1]);
            std::swap(sampling.logits_count[i0],     sampling.logits_count[i1]);
            std::swap(sampling.probs_count[i0],      sampling.probs_count[i1]);
            std::swap(sampling.candidates_count[i0], sampling.candidates_count[i1]);
        }
    }

    output_swaps.clear();

    // [TAG_EXTRACT_TARGET_EMBEDDINGS]
    // Layer inputs and unmasked NextN embeddings contain all token rows, independent of logits selection.
    for (size_t i = 0; i < embd_batch_idxs.size(); ++i) {
        while (embd_batch_idxs[i] != (int32_t) i) {
            const int32_t j = embd_batch_idxs[i];
            GGML_ASSERT(j >= 0 && (size_t) j < embd_batch_idxs.size());
            if (embd_nextn.has_data() && !cparams.embeddings_nextn_masked) {
                for (size_t k = 0; k < n_embd_out; ++k) {
                    std::swap(embd_nextn.data[i*n_embd_out + k], embd_nextn.data[j*n_embd_out + k]);
                }
            }
            for (auto & layer : embd_layer_inp) {
                // F1 keeps the rows on the device (its own row map, see decode()); the host rows are unused there
                if (layer.has_data() && !cparams.lf_layer_inp_dev) {
                    for (size_t k = 0; k < n_embd; ++k) {
                        std::swap(layer.data[i*n_embd + k], layer.data[j*n_embd + k]);
                    }
                }
            }
            std::swap(embd_batch_idxs[i], embd_batch_idxs[j]);
        }
    }
    embd_batch_idxs.clear();
}

//
// graph
//

uint32_t llama_context::graph_max_nodes(uint32_t n_tokens) const {
    uint32_t res;
    if (model.arch == LLM_ARCH_KIMI_K3 || model.arch == LLM_ARCH_GLM5_NEXT) {
        // the n_tokens*40 budget below is exhausted at ubatch 3840
        res = std::max<uint32_t>(n_tokens * 160, 64u * model.n_tensors());
    } else if (model.arch == LLM_ARCH_HRM_TEXT) {
        // the 128-slot looped graph needs roughly one stack per token budget
        res = std::max<uint32_t>(n_tokens * 80, 64u * model.n_tensors());
    } else if (model.arch == LLM_ARCH_QWEN3NEXT ||
        model.arch == LLM_ARCH_KIMI_LINEAR ||
        model.arch == LLM_ARCH_BAILINGMOE3 ||
        model.arch == LLM_ARCH_QWEN35 ||
        model.arch == LLM_ARCH_QWEN35MOE ||
        model.arch == LLM_ARCH_CLEF ||
        model.arch == LLM_ARCH_QWEN4EXP ||
        model.arch == LLM_ARCH_DEEPSEEK4 ||
        model.arch == LLM_ARCH_DEEPSEEK41 ||
        (model.arch == LLM_ARCH_DFLASH && model.hparams.dsv4_hc_mult > 0) ||
        model.arch == LLM_ARCH_NANBEIGE ||
        model.arch == LLM_ARCH_MINIMAX_01 ||
        model.arch == LLM_ARCH_MINIMAX_M3 ||
        model.arch == LLM_ARCH_HY_V4) {
        res = std::max<uint32_t>(n_tokens * 40, 32u * model.n_tensors());
    } else if (model.arch == LLM_ARCH_DFLASH && model.hparams.dflash_selector_rank > 0) {
        // DFlash2's convolutions and selector are shape work rather than matmuls,
        // so they cost ~8.6 nodes per tensor against ~5.9 for a plain DFlash draft
        res = std::max<uint32_t>(1024u, 12u*model.n_tensors());
    } else {
        res = std::max<uint32_t>(1024u, 8u*model.n_tensors());
        for (const auto & lora : model.loras) {
            res += lora->get_n_nodes();
        }
    }

    // every sampler builds n_outputs_max_per_seq chains, see llm_graph_context::build_sampling
    for (const auto & [seq_id, sampler] : sampling.samplers) {
        res += llama_sampler_backend_n_nodes(sampler) * cparams.n_outputs_max_per_seq;
    }

    // F1 (issue #240): one SET_ROWS per extracted layer plus the shared row-index input
    if (!cparams.lf_layer_inp_dev_store.empty()) {
        res += 1;
        for (bool enabled : cparams.embeddings_layer_inp) {
            res += enabled ? 1 : 0;
        }
    }

    if (cparams.training) {
        res *= 4;
    }

    return res;
}

llm_graph_result * llama_context::get_gf_res_reserve() const {
    return static_cast<llm_graph_result *>(gf_res_reserve.get());
}

llm_graph_result * llama_context::get_gf_res_prev(bool alt) {
    auto & res = gf_res_prev[alt];
    if (!res) {
        res.reset(new llm_graph_result(gf_res_reserve->get_max_nodes()));
    }
    return res.get();
}

// pack sampler outputs into as few sequences as possible before using sequences without samplers
static void ubatch_prepare_reserve(
              llama_ubatch                            & ubatch,
              uint32_t                                  n_outputs,
        const std::map<llama_seq_id, llama_sampler *> & samplers,
              uint32_t                                  n_outputs_max_per_seq) {
    const uint32_t n_seqs       = ubatch.n_seqs;
    const uint32_t n_seq_tokens = ubatch.n_seq_tokens;

    for (uint32_t s = 0; s < n_seqs; ++s) {
        for (uint32_t t = 0; t < n_seq_tokens; ++t) {
            const uint32_t i = s * n_seq_tokens + t;
            ubatch.n_seq_id[i] = 1;
            ubatch.seq_id[i] = &ubatch.seq_id_unq[s];
        }
    }

    // sequences with a sampler that fit in this ubatch
    std::vector<uint32_t> sampler_seqs;
    std::vector<bool> has_sampler(n_seqs, false);
    for (const auto & entry : samplers) {
        const llama_seq_id seq_id = entry.first;
        if (seq_id < 0 || (uint32_t) seq_id >= n_seqs) {
            continue;
        }

        sampler_seqs.push_back(seq_id);
        has_sampler[seq_id] = true;
    }

    uint32_t n_outputs_set = 0;

    const uint32_t n_outputs_per_seq = std::min(n_seq_tokens, n_outputs_max_per_seq);
    for (uint32_t s : sampler_seqs) {
        if (n_outputs_set >= n_outputs) {
            break;
        }

        for (uint32_t t = 0; t < n_outputs_per_seq && n_outputs_set < n_outputs; ++t) {
            ubatch.output[s * n_seq_tokens + t] = true;
            ++n_outputs_set;
        }
    }

    // use sequences without samplers for any remaining outputs
    for (uint32_t t = 0; t < n_seq_tokens && n_outputs_set < n_outputs; ++t) {
        for (uint32_t s = 0; s < n_seqs && n_outputs_set < n_outputs; ++s) {
            if (has_sampler[s]) {
                continue;
            }

            ubatch.output[s * n_seq_tokens + t] = true;
            ++n_outputs_set;
        }
    }
}

ggml_cgraph * llama_context::graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs_l, const llama_memory_context_i * mctx, bool split_only, size_t * sizes,
        bool packed_kq_mask, bool reserve_injection) {
    LLAMA_LOG_DEBUG("%s: reserving a graph for ubatch with n_tokens = %4u, n_seqs = %2u, n_outputs = %4u\n", __func__, n_tokens, n_seqs, n_outputs_l);
    GGML_ASSERT(n_outputs_l >= 1);

    if (n_tokens % n_seqs != 0) {
        n_tokens = ((n_tokens + (n_seqs - 1)) / n_seqs) * n_seqs; // round to next multiple of n_seqs
        LLAMA_LOG_DEBUG("%s: making n_tokens a multiple of n_seqs - n_tokens = %u, n_seqs = %u, n_outputs = %u\n", __func__, n_tokens, n_seqs, n_outputs_l);
    }

    ggml_backend_sched_reset(sched.get());

    // when the scheduler is reset, we cannot reuse old graphs, so we reset the previous graph results
    for (auto & res : gf_res_prev) {
        if (res) {
            res->reset();
        }
    }
    gf_res_prev_active = nullptr;
    gf_res_prev_active_alt = nullptr;

    // store the n_outputs as it is, and restore it afterwards
    // TODO: not sure if needed, might simplify in the future by removing this
    const auto save_n_outputs = this->n_outputs;

    this->n_outputs = n_outputs_l;

    llama_batch_allocr balloc_l(model.hparams.n_pos_per_embd());
    llama_ubatch ubatch = balloc_l.ubatch_reserve(n_tokens/n_seqs, n_seqs, reserve_injection);

    ubatch_prepare_reserve(ubatch, n_outputs_l, sampling.samplers, cparams.n_outputs_max_per_seq);

    auto * res = gf_res_reserve.get();

    auto gparams = graph_params(res, ubatch, mctx, ctx_type_to_graph_type(cparams.ctx_type));

    // The derived kq mask is a *per-batch* optimization, so a batch it cannot serve allocates the
    // packed mask, n_kv*n_tokens*2 bytes - and at a deep context that is hundreds of MiB, which a
    // reserve measured with the derived form on does not contain.  The first such batch then has to
    // *grow* the compute buffer mid-run, and on a --fit server whose target left less free memory than
    // that the growth fails with cudaMalloc out of memory and the request dies
    // (github.com/stew675/llama-cpp-rdna-boosts/issues/42).  Building the measure graph with the
    // packed mask makes the reserve (and therefore --fit) account for it up front, so the growth
    // cannot happen.  Callers pass kq_mask_packed_reachable(), which is true only when such a batch is
    // actually reachable in this context - see its definition.  Only the *reserve* changes: every
    // derivable batch still takes the derived path at runtime, and a non-derivable one takes the
    // packed-mask kernel upstream ships for it anyway.
    if (packed_kq_mask) {
        gparams.cparams.kq_mask_derived = false;
    }

    res->reset();

    auto * gf = model.build_graph(gparams);

    // check the graph before scheduling: cross-backend copies break the producer chain that the check follows
    if (!hadamard_verified && gf && (!model.hdmd.rot.empty() || !model.hdmd.inv.empty())) {
        llama_verify_hadamard_graph(gf, model.hdmd.rot, model.hdmd.inv, moe_cache.get());
        hadamard_verified = true;
    }

    this->n_input_tensors = llama_graph_n_input_tensors(gf);
    this->n_outputs = save_n_outputs;

    // initialize scheduler with the specified graph
    if (split_only) {
        if (sizes) {
            ggml_backend_sched_reserve_size(sched.get(), gf, sizes);
        } else {
            ggml_backend_sched_split_graph(sched.get(), gf);
        }
    } else {
        // wip/moe-verify-fusions (narrow-2): the MTP draft context (identified by `ctx_other`, set for
        // every speculative draft context) routes its COMPUTE buffer to the slab's pinned narrow-2 region
        // at the top of the reserved VA.  The slab used to hand a single narrow base to every narrow view,
        // so the draft's buffer aliased the target's narrow verify buffer at base 0 (the MTP-vs-plain
        // near-tie drift).  `GGML_CUDA_SLAB_NARROW2=0` is the A/B arm (the draft bypasses the slab).
        const bool is_draft = cparams.ctx_other != nullptr;
        // read once: this runs on every reserve; getenv() is expensive on Windows
        static const bool work_alloc_debug = getenv("GGML_CUDA_WORK_ALLOC_DEBUG") != nullptr;
        if (work_alloc_debug) {
            LLAMA_LOG_WARN("GRAPH_RESERVE ctx=%p is_draft=%d n_tokens=%u\n", (void *) this, (int) is_draft, n_tokens);
        }
        if (is_draft) {
            for (const auto & d : model.devices) {
                ggml_backend_dev_slab_compute_narrow2(d.dev, true);
            }
        }
        const bool reserved = ggml_backend_sched_reserve(sched.get(), gf);
        if (is_draft) {
            for (const auto & d : model.devices) {
                ggml_backend_dev_slab_compute_narrow2(d.dev, false);
            }
        }
        if (!reserved) {
            GGML_ASSERT(!sizes);
            LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
            return nullptr;
        }
    }

    return gf;
}

llm_graph_params llama_context::graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const {
    return {
        /*.arch        =*/ model.arch,
        /*.hparams     =*/ model.hparams,
        /*.cparams     =*/ cparams,
        /*.ubatch      =*/ ubatch,
        /*.gtype       =*/ gtype,
        /*.sched       =*/ sched.get(),
        /*.backend_cpu =*/ backend_cpu,
        /*.cvec        =*/ cvec.get(),
        /*.loras       =*/ loras.get(),
        /*.mctx        =*/ mctx,
        /*.cross       =*/ &cross,
        /*.moe_cache   =*/ moe_cache.get(),
        /*.prec_policy =*/ &model.prec_policy,
        /*.hdmd        =*/ model.hdmd.rot.empty() ? nullptr : &model.hdmd,
        /*.samplers    =*/ sampling.samplers,
        /*.mtp_draft   =*/ mtp_draft.get(),
        /*.n_outputs   =*/ n_outputs,
        /*.cb          =*/ graph_get_cb(),
        /*.res         =*/ res,
    };
}

void llama_context::sched_graph_realloc(ggml_backend_sched_t sched, ggml_cgraph * gf, bool backend_ids_changed, bool graph_size_changed, void * user_data) {
    // reallocating after the backend assignments changed is expected, trace only the unexpected ones
    if (backend_ids_changed) {
        return;
    }

    auto * lctx = static_cast<llama_context *>(user_data);

    if (graph_size_changed) {
        lctx->n_reallocs_diff++;
    } else {
        lctx->n_reallocs_same++;
    }

    LLAMA_LOG_INFO("%s: graph reallocation: nodes = %d (changed = %d), splits = %d, n_same = %d, n_diff = %d\n",
            __func__, ggml_graph_n_nodes(gf), graph_size_changed, ggml_backend_sched_get_n_splits(sched),
            lctx->n_reallocs_same, lctx->n_reallocs_diff);
}

ggml_status llama_context::graph_compute(
            ggml_cgraph * gf,
                   bool   batched) {
    int n_threads        = batched ? cparams.n_threads_batch : cparams.n_threads;
    ggml_threadpool_t tp = batched ? threadpool_batch        : threadpool;

    if (backend_cpu != nullptr) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
        auto * set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        if (set_threadpool_fn) {
            set_threadpool_fn(backend_cpu, tp);
        }
    }

    // set the number of threads for all the backends
    for (const auto & set_n_threads_fn : set_n_threads_fns) {
        set_n_threads_fn.second(set_n_threads_fn.first, n_threads);
    }

    auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: ggml_backend_sched_graph_compute_async failed with error %d\n", __func__, status);
    }

    // fprintf(stderr, "splits: %d\n", ggml_backend_sched_get_n_splits(sched));

    return status;
}

bool llama_context::sched_copy_experts(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph, void * user_data) {
    auto * lctx = static_cast<llama_context *>(user_data);

    // The slot maps of the upstream `llama_moe_cache` (`--moe-cache-mib`): plan the slots, upload
    // the missing experts and fill the slot map, so the GET_ROWS that follows reads fresh slot ids.
    // This is the ONLY thing this callback answers for.  The used-expert staging and pruning of
    // the delivery's own host-expert paths (op-offload / expert cache / slab) lives in the
    // scheduler (`ggml_backend_sched_copy_input`), which is the sole owner of that machinery --
    // keeping a second copy of it here would double-stage every host expert table.
    return lctx->moe_cache && lctx->moe_cache->copy(backend, src, dst, graph);
}

llm_graph_cb llama_context::graph_get_cb() const {
    return [&](const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il) {
        if (il >= 0) {
            ggml_format_name(cur, "%s-%d", name, il);
        } else {
            ggml_set_name(cur, name);
        }

        // - norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
        // - force the last op of the layer on the specified backend to avoid running it on the backend of the next layer due to scheduling
        // FIXME: fix in ggml_backend_sched
        const bool full_offload = model.n_gpu_layers() > model.hparams.n_layer_all;
        if (ubatch.n_tokens < 32 || full_offload) {
            if (il != -1 && (strcmp(name, "norm") == 0 || strcmp(name, "l_last") == 0)) {
                const auto & dev_layer = model.dev_layer(il);
                for (const auto & backend : backends) {
                    if (ggml_backend_get_device(backend.get()) == dev_layer) {
                        if (ggml_backend_supports_op(backend.get(), cur)) {
                            ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend.get());
                        }
                    }
                }
            }
        }
    };
}

//
// state save/load
//

class llama_io_write_dummy : public llama_io_write_i {
public:
    llama_io_write_dummy(bool skip_tensors) : skip_tensors(skip_tensors) {}

    void write(const void * /* src */, size_t size) override {
        size_written += size;
    }

    void write_tensor(ggml_tensor * /* tensor */, size_t /* offset */, size_t size) override {
        if (skip_tensors) {
            return;
        }

        size_written += size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    const bool skip_tensors;

    size_t size_written = 0;
};

// Batched state reads (LLAMA_STATE_ASYNC, default on; 0 = the per-tensor synchronous reads).  A context checkpoint
// of a recurrent model is ~75 small tensors (one sequence's cells), split over the devices under -sm tensor, and each
// synchronous read costs a device sync plus a bounce through the shared pinned chunk.  Instead, queue them all
// as async copies into one reused pinned staging area, synchronize each backend once, and copy to the caller's
// buffer on the host.  Restores are batched the same way through the same stage (llama_io_read_host, below;
// LLAMA_STATE_ASYNC_READ=0 keeps their per-tensor synchronous sets).  States above LLAMA_STATE_ASYNC_MAX_MIB (default
// 256) keep the old path in both directions, so a whole slot save or load never pins gigabytes.
//
// Fold (r96, PR #189): the staging buffer (`state_stage`) is kept by the context across saves and rounded up to
// 16 MiB, so a context holds the checkpoint size (never more than LLAMA_STATE_ASYNC_MAX_MIB) in pinned host memory
// for its lifetime.  It is ONE buffer per context, so state saves and restores on the same context must not run
// concurrently; the llama state API is not documented as thread-safe, and this makes that concrete.
static bool llama_state_async_enabled(size_t total) {
    static const bool on = [] { const char * e = getenv("LLAMA_STATE_ASYNC"); return e == nullptr || atoi(e) != 0; }();
    static const size_t max_bytes = [] {
        const char * e = getenv("LLAMA_STATE_ASYNC_MAX_MIB");
        return (size_t) (e != nullptr ? atoll(e) : 256) << 20;
    }();
    return on && total > 0 && total <= max_bytes;
}

// Pipelined batched state copies (fold r151, PR #233).  The batched copy ran the host memcpy into/out of the
// pinned `state_stage` and the device copies back to back, so a 112.6 MiB recurrent checkpoint paid ~5.3 ms of
// memcpy plus ~6.4 ms of device copies per restore (and 5.7 + 6.3 per save) serially.  The pieces are cut into
// up to LLAMA_STATE_PIPE_GROUPS (default 8; 1 = the unpipelined copy) size-balanced groups, so the host memcpy
// of one group overlaps the device copies of another.  Returns the first piece index of each group plus the
// end index.  Host DRAM bandwidth bounds the overlap (the memcpy moves twice the bytes the DMA does), so expect
// a fraction of the sum back, not the max.
//
// Invariants (a torn state is a wrong answer, not a crash): every group owns a distinct range of the stage, so
// the next group's pack never overwrites a range a queued copy still reads; the SAVE side records one event per
// (backend, group) AFTER that group's copies are queued and synchronizes it before the group is copied out (an
// event re-recorded for a later group would move past the earlier one, hence one event per group); and a
// backend with no events (the CPU backend, a device whose iface has no `event_new`) makes the whole copy take
// the single `llama_state_sync_all` path, exactly as before.  The restore side needs no events: it queues each
// group's `set_async` and keeps the one-per-backend sync at the end.
static std::vector<size_t> llama_state_pipe_groups(const std::vector<size_t> & sizes, size_t total) {
    static const size_t n_groups = [] {
        const char * e = getenv("LLAMA_STATE_PIPE_GROUPS");
        const long v = e != nullptr ? atol(e) : 8;
        return (size_t) std::max(1L, std::min(v, (long) llama_context::n_state_pipe_events));
    }();
    std::vector<size_t> first = {0};
    const size_t g = std::min(n_groups, sizes.size());
    size_t acc = 0;
    for (size_t i = 0; i < sizes.size() && first.size() < g; ++i) {
        acc += sizes[i];
        if (acc * g >= total * first.size() && i + 1 < sizes.size()) {
            first.push_back(i + 1);
        }
    }
    first.push_back(sizes.size());
    return first;
}

// Streaming (non-temporal) host copies for the batched state path (fold r152, PR #235).  Both directions move a
// checkpoint between the pinned `state_stage` and the caller's buffer with one `memcpy` per piece, and that copy is
// bound by host memory bandwidth.  Two things get in its way, and both are fixed here:
//   * a plain store first reads the destination line into the cache (read-for-ownership), so the copy pays a
//     destination read on top of its own read and write.  A non-temporal store writes straight to memory and drops
//     it.  Whether that is worth anything depends on the box: it is the whole win where the checkpoint does not fit
//     in the CPU's last-level cache (a 3900X + DDR4, the submitter's measurement: 19.4 -> 11.8 ms per 150 MiB) and
//     it is ~0 where it does (a 9950X3D has 192 MiB of L3, so a 164 MiB checkpoint's destination lines are mostly
//     resident and the read-for-ownership hits cache).
//   * one thread cannot saturate the memory controller on either box.  Measured on the 9950X3D, 164 MiB in the
//     pieces a 35B-A3B 5246-token checkpoint actually has (30 x 80 KiB, 30 x 2.6 MiB, 20 x 4.2 MiB): plain memcpy
//     1 thread 6.4 ms, streaming 1 thread 6.4 ms, streaming 2 threads 3.0 ms, streaming 3-4 threads 3.3-3.7 ms.  So
//     the threads are where this box's win is and the streaming stores are free; three or more threads lose to two
//     on a dual-channel DDR5 controller.
// The copy is per group (see llama_state_pipe_groups), and the group loop stays the caller's: a group's device
// copies are fenced by its event before it is copied out, so the pieces of ONE group are what the workers take.
// LLAMA_STATE_COPY_THREADS: 0 = the pre-r152 path (one plain memcpy per piece), 1 = streaming stores on one
// thread, N = streaming stores on N threads (default 2).  States below 16 MiB keep the plain copy: the whole copy
// is then a few hundred us and a thread hand-off costs more than it saves.
// Streaming stores are x86-only (`__SSE2__`, or `_M_X64` for MSVC's always-on SSE2); every other target keeps the
// memcpy and only gains the threads.  The bytes are the same, and nothing reads them until after the copy returns
// (the save hands them to the caller, the restore hands them to the device), so bypassing the cache costs nothing.
#ifdef LLAMA_STATE_COPY_STREAM
static constexpr size_t llama_state_copy_min_piece = 64u << 10;   // below this a piece is not worth a streamed run
#endif
static constexpr size_t llama_state_copy_min_total = 16u << 20;   // below this the whole state takes the plain copy

static int llama_state_copy_threads(size_t total) {
    static const int n = [] {
        const char * e = getenv("LLAMA_STATE_COPY_THREADS");
        return std::max(0, e != nullptr ? atoi(e) : 2);
    }();
    return total >= llama_state_copy_min_total ? n : 0;
}

// one contiguous run, self-fencing (so a worker thread needs nothing else)
static void llama_state_copy_run(uint8_t * dst, const uint8_t * src, size_t n) {
#ifdef LLAMA_STATE_COPY_STREAM
    if (n >= llama_state_copy_min_piece) {
        size_t i = 0;
        // aligned to a cache line, not just to the store width: a run that starts mid-line makes the two half-lines
        // each carry a partial write-combining buffer, which is slower than the memcpy it replaced
        for (; i < n && (((uintptr_t) (dst + i)) & 63) != 0; ++i) {
            dst[i] = src[i];
        }
        for (; i + 64 <= n; i += 64) {
            const __m128i a = _mm_loadu_si128((const __m128i *) (src + i));
            const __m128i b = _mm_loadu_si128((const __m128i *) (src + i + 16));
            const __m128i c = _mm_loadu_si128((const __m128i *) (src + i + 32));
            const __m128i d = _mm_loadu_si128((const __m128i *) (src + i + 48));
            _mm_stream_si128((__m128i *) (dst + i),      a);
            _mm_stream_si128((__m128i *) (dst + i + 16), b);
            _mm_stream_si128((__m128i *) (dst + i + 32), c);
            _mm_stream_si128((__m128i *) (dst + i + 48), d);
        }
        _mm_sfence();   // non-temporal stores are unordered until the fence: the bytes must be visible to the
                        // device DMA (restore) or to the state API's caller (save) before this returns
        if (i < n) {
            memcpy(dst + i, src + i, n - i);
        }
        return;
    }
#endif
    memcpy(dst, src, n);
}

// a contiguous run to copy, with its stage offset resolved
struct llama_state_copy_piece {
    uint8_t       * dst;
    const uint8_t * src;
    size_t          n;
};

// copy pieces [i0, i1) of `pcs` over `n_threads` threads, split by cumulative bytes (the pieces differ in size by
// two orders of magnitude, so an even piece count would be an uneven byte count)
static void llama_state_copy_pieces(const std::vector<llama_state_copy_piece> & pcs, size_t i0, size_t i1, int n_threads) {
    if (n_threads == 0) {   // LLAMA_STATE_COPY_THREADS=0: the pre-r152 path, one plain memcpy per piece
        for (size_t i = i0; i < i1; ++i) {
            memcpy(pcs[i].dst, pcs[i].src, pcs[i].n);
        }
        return;
    }
    auto copy_range = [&pcs](size_t a, size_t b) {
        for (size_t i = a; i < b; ++i) {
            llama_state_copy_run(pcs[i].dst, pcs[i].src, pcs[i].n);
        }
    };
    if (n_threads <= 1 || i1 - i0 < 2) {
        copy_range(i0, i1);
        return;
    }
    size_t total = 0;
    for (size_t i = i0; i < i1; ++i) {
        total += pcs[i].n;
    }
    std::vector<size_t> first = { i0 };
    size_t acc = 0;
    for (size_t i = i0; i < i1 && first.size() < (size_t) n_threads; ++i) {
        acc += pcs[i].n;
        if (acc * n_threads >= total * first.size() && i + 1 < i1) {
            first.push_back(i + 1);
        }
    }
    first.push_back(i1);
    std::vector<std::thread> workers;
    for (size_t w = 1; w + 1 < first.size(); ++w) {
        workers.emplace_back(copy_range, first[w], first[w + 1]);
    }
    copy_range(first[0], first[1]);
    for (auto & th : workers) {
        th.join();   // the join is also what makes a worker's streamed stores visible to the caller
    }
}

static void llama_state_sync_all(const std::vector<ggml_backend_t> & bks) {
    std::vector<ggml_backend_t> done;
    for (ggml_backend_t b : bks) {
        if (std::find(done.begin(), done.end(), b) == done.end()) {
            ggml_backend_synchronize(b);
            done.push_back(b);
        }
    }
}

// one piece of a device-state copy, as byte ranges into two tensors
struct llama_state_copy_job {
    ggml_tensor * src;
    ggml_tensor * dst;
    size_t        src_off;
    size_t        dst_off;
    size_t        n;
};

// Fold (r157, TODO #110): a copy between two meta buffers has to go through the host (a meta buffer has no
// `.cpy_tensor`), and `ggml_backend_tensor_copy`'s fallback does it one piece at a time through a *pageable*
// `malloc` with a synchronize per piece.  The device-state path copies one piece per (layer, K/V) per cell
// range, so a 1024-token 2-slot state is ~130 pieces: measured 2.23 s to save and 2.28 s to restore 213.6 MiB
// (~100 MB/s) against 21.9/38.9 ms for the same bytes over the host path.  Queue one band of pieces through the
// context's pinned state stage and synchronize once per band instead: 14.7/17.2 ms, i.e. 150x, and the
// single-device arm (which copies on the device and never reaches this) is unchanged at 3.0/3.2 ms.
// Returns false when the caller must take its own path (no context, no pinned host buffer type, LLAMA_STATE_
// STAGE_MIB=0 which is the A/B arm).
static bool llama_state_copy_staged(llama_context * lctx, const std::vector<llama_state_copy_job> & jobs) {
    static const size_t band_max = [] {
        const char * e = getenv("LLAMA_STATE_STAGE_MIB");
        return e != nullptr ? (size_t) strtoull(e, nullptr, 10) << 20 : (size_t) 64 << 20;
    }();
    if (lctx == nullptr || jobs.empty() || band_max == 0) {
        return false;
    }
    std::vector<ggml_backend_t> bks(2*jobs.size(), nullptr);
    for (size_t i = 0; i < jobs.size(); ++i) {
        if ((bks[2*i + 0] = lctx->state_backend(jobs[i].src)) == nullptr ||
            (bks[2*i + 1] = lctx->state_backend(jobs[i].dst)) == nullptr) {
            return false;
        }
    }
    struct band_piece {
        size_t   job;
        size_t   n;
        size_t   pos;
        uint8_t * data;
    };
    std::vector<band_piece>     band;
    std::vector<ggml_backend_t> band_bks;
    size_t pos = 0;
    // a band is a run of whole pieces: a piece is only ever a window the backends accept (a tensor or a view of
    // it), so it is never cut.  A single piece larger than the band gets a band of its own.
    auto flush = [&]() {
        size_t need = 0;
        for (const auto & b : band) {
            need += b.n;
        }
        uint8_t * stage = lctx->state_stage(need);
        if (stage == nullptr) {
            return false;
        }
        for (auto & b : band) {
            b.data = stage + b.pos;
        }
        band_bks.clear();
        for (const auto & b : band) {
            const auto & j = jobs[b.job];
            ggml_backend_tensor_get_async(bks[2*b.job + 0], j.src, b.data, j.src_off, b.n);
            if (std::find(band_bks.begin(), band_bks.end(), bks[2*b.job + 0]) == band_bks.end()) {
                band_bks.push_back(bks[2*b.job + 0]);
            }
        }
        llama_state_sync_all(band_bks);
        for (const auto & b : band) {
            const auto & j = jobs[b.job];
            ggml_backend_tensor_set_async(bks[2*b.job + 1], j.dst, b.data, j.dst_off, b.n);
            if (std::find(band_bks.begin(), band_bks.end(), bks[2*b.job + 1]) == band_bks.end()) {
                band_bks.push_back(bks[2*b.job + 1]);
            }
        }
        llama_state_sync_all(band_bks);   // also what makes the stage reusable by the next band
        band.clear();
        pos = 0;
        return true;
    };
    for (size_t i = 0; i < jobs.size(); ++i) {
        if (pos > 0 && pos + jobs[i].n > band_max && !flush()) {
            return false;
        }
        band.push_back({ i, jobs[i].n, pos, nullptr });
        pos += jobs[i].n;
    }
    return band.empty() || flush();
}

// whether a (tensor, offset, size) window is one the backend serves directly.  A meta buffer's setter/reader
// splices a window aligned to the tensor's own layout with one strided copy per device; a view of the piece is
// correct for anything else but walks its split state row by row (measured: 4.5 ms vs 0.15 ms for 2 MiB), so
// the copy stays on the tensor itself whenever it can.  `nb[1]` is the coarsest of those units, hence the test.
static bool llama_state_copy_window_ok(const ggml_tensor * t, size_t off, size_t n) {
    const size_t unit = t->nb[1] != 0 ? t->nb[1] : 1;
    return off % unit == 0 && n % unit == 0;
}

// whether a copy into `buf` has to be staged: the destination cannot copy on device (the meta backend), so
// `ggml_backend_tensor_copy` would take its own pageable bounce
static bool llama_state_copy_bounces(ggml_backend_buffer_t buf) {
    ggml_backend_dev_t dev = buf != nullptr ? ggml_backend_buft_get_device(ggml_backend_buffer_get_type(buf)) : nullptr;
    return dev != nullptr && ggml_backend_dev_is_meta(dev);
}

class llama_io_write_host : public llama_io_write_i {
public:
    llama_io_write_host(
            uint8_t * p, size_t len, llama_context * lctx = nullptr) : ptr(p), buf_size(len), lctx(lctx) {}

    // the batched path (see llama_state_async_enabled); false = not taken, nothing was copied
    bool flush_async() {
        size_t total = 0;
        for (const auto & w : winfos) {
            total += w.size;
        }
        if (lctx == nullptr || !llama_state_async_enabled(total)) {
            return false;
        }
        std::vector<ggml_backend_t> bks(winfos.size());
        for (size_t i = 0; i < winfos.size(); ++i) {
            if ((bks[i] = lctx->state_backend(winfos[i].tensor)) == nullptr) {
                return false;
            }
        }
        uint8_t * stage = lctx->state_stage(total);
        if (stage == nullptr) {
            return false;
        }
        std::vector<size_t> sizes(winfos.size());
        for (size_t i = 0; i < winfos.size(); ++i) {
            sizes[i] = winfos[i].size;
        }
        const std::vector<size_t> grp = llama_state_pipe_groups(sizes, total);
        const size_t n_grp = grp.size() - 1;
        // one event per (backend, group): an event re-recorded for a later group would move past the earlier one
        std::vector<std::vector<std::pair<ggml_backend_t, ggml_backend_event_t>>> grp_ev(n_grp);
        bool piped = n_grp > 1;
        for (size_t g = 0; piped && g < n_grp; ++g) {
            for (size_t i = grp[g]; piped && i < grp[g + 1]; ++i) {
                bool seen = false;
                for (const auto & be : grp_ev[g]) {
                    seen = seen || be.first == bks[i];
                }
                if (!seen) {
                    ggml_backend_event_t ev = lctx->state_pipe_event(bks[i], g);
                    piped = ev != nullptr;
                    grp_ev[g].emplace_back(bks[i], ev);
                }
            }
        }
        // LLAMA_STATE_TIMING=1 logs where the time of a batched state copy goes (a diagnostic; off by default)
        static const bool st_timing = getenv("LLAMA_STATE_TIMING") != nullptr;
        const int64_t t_enq = ggml_time_us();
        size_t off = 0;
        for (size_t g = 0; g < n_grp; ++g) {
            for (size_t i = grp[g]; i < grp[g + 1]; ++i) {
                ggml_backend_tensor_get_async(bks[i], winfos[i].tensor, stage + off, winfos[i].offset, winfos[i].size);
                off += winfos[i].size;
            }
            if (piped) {
                for (const auto & be : grp_ev[g]) {
                    ggml_backend_event_record(be.second, be.first);
                }
            }
        }
        if (!piped) {
            llama_state_sync_all(bks);
        }
        size_t stage_off = 0;
        std::vector<llama_state_copy_piece> pcs(winfos.size());
        for (size_t i = 0; i < winfos.size(); ++i) {
            pcs[i] = { winfos[i].ptr, stage + stage_off, winfos[i].size };
            stage_off += winfos[i].size;
        }
        const int n_copy_threads = llama_state_copy_threads(total);
        const int64_t t_copy = ggml_time_us();
        for (size_t g = 0; g < n_grp; ++g) {
            if (piped) {
                for (const auto & be : grp_ev[g]) {
                    ggml_backend_event_synchronize(be.second);
                }
            }
            llama_state_copy_pieces(pcs, grp[g], grp[g + 1], n_copy_threads);
        }
        if (st_timing) {
            const int64_t t_end = ggml_time_us();
            LLAMA_LOG_INFO("state get: %zu pieces, %.1f MiB in %zu group(s): enqueue %.2f ms, copy+wait %.2f ms "
                           "(%.1f GiB/s), total %.2f ms\n", winfos.size(), total / 1048576.0, n_grp,
                           (t_copy - t_enq) / 1e3, (t_end - t_copy) / 1e3,
                           total / 1073741824.0 / std::max(1e-6, (t_end - t_copy) / 1e6), (t_end - t_enq) / 1e3);
        }
        return true;
    }

    ~llama_io_write_host() {
        if (flush_async()) {
            return;
        }
        // Each ggml_backend_tensor_get is a synchronous device copy with a fixed cost (tens of us on ROCm), and a sequence's
        // KV cells are often many short ranges (concurrent decoding interleaves the slots' cells a few tokens at a time), so
        // the deferred reads of one tensor whose gaps are at most LLAMA_STATE_GET_MERGE_GAP bytes (default 256 KiB, 0 = off)
        // are read as one span and scattered on the host.  The bytes written are the same.
        static const size_t max_gap = [] {
            const char * e = getenv("LLAMA_STATE_GET_MERGE_GAP");
            return e != nullptr ? (size_t) strtoull(e, nullptr, 10) : (size_t) 256*1024;
        }();
        if (max_gap == 0) {
            for (const auto & winfo : winfos) {
                ggml_backend_tensor_get(winfo.tensor, winfo.ptr, winfo.offset, winfo.size);
            }
        } else {
            std::vector<uint8_t> span_buf;
            for (size_t i = 0; i < winfos.size(); ) {
                // [i, j): consecutive reads of one tensor, each starting at or after the previous one's end
                const write_info & first = winfos[i];
                size_t end = first.offset + first.size;
                bool   dst_contiguous = true;
                size_t j = i + 1;
                for (; j < winfos.size(); ++j) {
                    const write_info & w = winfos[j];
                    if (w.tensor != first.tensor || w.offset < end || w.offset - end > max_gap) {
                        break;
                    }
                    dst_contiguous = dst_contiguous && w.offset == end && w.ptr == winfos[j - 1].ptr + winfos[j - 1].size;
                    end = w.offset + w.size;
                }
                bool merged = j - i == 1 || dst_contiguous;
                if (merged) {
                    ggml_backend_tensor_get(first.tensor, first.ptr, first.offset, end - first.offset);
                } else {
                    try {
                        span_buf.resize(end - first.offset);
                        merged = true;
                    } catch (const std::bad_alloc &) {
                        // no host memory for the span (a destructor must not throw): read the pieces one by one
                    }
                    if (merged) {
                        ggml_backend_tensor_get(first.tensor, span_buf.data(), first.offset, end - first.offset);
                        for (size_t k = i; k < j; ++k) {
                            memcpy(winfos[k].ptr, span_buf.data() + (winfos[k].offset - first.offset), winfos[k].size);
                        }
                    } else {
                        for (size_t k = i; k < j; ++k) {
                            ggml_backend_tensor_get(winfos[k].tensor, winfos[k].ptr, winfos[k].offset, winfos[k].size);
                        }
                    }
                }
                i = j;
            }
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save the write for later during destruction
        winfos.push_back({tensor, ptr, size, offset});

        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;

    llama_context * lctx = nullptr;
};

class llama_io_read_host : public llama_io_read_i {
public:
    llama_io_read_host(const uint8_t * p, size_t len, llama_context * lctx = nullptr) : ptr(p), lctx(lctx), buf_size(len) {}

    // Batched restore (LLAMA_STATE_ASYNC_READ, default on; 0 = the per-tensor synchronous sets below).  Each
    // synchronous set bounces through a pinned chunk and synchronizes its device, once per tensor and device (a
    // hybrid model's checkpoint is ~75 tensors).  Instead pack the pieces into the context's pinned state stage,
    // queue every tensor as an async copy from it and synchronize each backend once.  Same bytes, same tensors.
    bool flush_async() {
        static const bool on = [] { const char * e = getenv("LLAMA_STATE_ASYNC_READ"); return e == nullptr || atoi(e) != 0; }();
        if (!on || lctx == nullptr || rinfos.empty()) {
            return false;
        }
        size_t total = 0;
        for (const auto & r : rinfos) {
            total += r.size;
        }
        if (!llama_state_async_enabled(total)) {
            return false;
        }
        std::vector<ggml_backend_t> bks(rinfos.size());
        for (size_t i = 0; i < rinfos.size(); ++i) {
            if ((bks[i] = lctx->state_backend(rinfos[i].tensor)) == nullptr) {
                return false;
            }
        }
        uint8_t * stage = lctx->state_stage(total);
        if (stage == nullptr) {
            return false;
        }
        // the pieces sit between small metadata fields in the caller's buffer, so pack them one by one; a group's device
        // copies are queued before the next group is packed, so they overlap its memcpy (see llama_state_pipe_groups)
        std::vector<size_t> sizes(rinfos.size());
        for (size_t i = 0; i < rinfos.size(); ++i) {
            sizes[i] = rinfos[i].size;
        }
        const std::vector<size_t> grp = llama_state_pipe_groups(sizes, total);
        // each piece's stage offset, resolved once (the pack loop and the queue loop below then read the same address)
        std::vector<llama_state_copy_piece> pcs(rinfos.size());
        size_t off = 0;
        for (size_t i = 0; i < rinfos.size(); ++i) {
            pcs[i] = { stage + off, rinfos[i].ptr, rinfos[i].size };
            off += rinfos[i].size;
        }
        static const bool st_timing = getenv("LLAMA_STATE_TIMING") != nullptr;
        const int n_copy_threads = llama_state_copy_threads(total);
        const int64_t t_pack = ggml_time_us();
        for (size_t g = 0; g + 1 < grp.size(); ++g) {
            llama_state_copy_pieces(pcs, grp[g], grp[g + 1], n_copy_threads);
            for (size_t i = grp[g]; i < grp[g + 1]; ++i) {
                ggml_backend_tensor_set_async(bks[i], rinfos[i].tensor, pcs[i].dst, rinfos[i].offset, rinfos[i].size);
            }
        }
        if (st_timing) {
            LLAMA_LOG_INFO("state set: %zu pieces, %.1f MiB in %zu group(s): pack+enqueue %.2f ms\n", rinfos.size(),
                           total / 1048576.0, grp.size() - 1, (ggml_time_us() - t_pack) / 1e3);
        }
        llama_state_sync_all(bks);   // the stage is reused by the next state copy on this context
        return true;
    }

    ~llama_io_read_host() {
        if (flush_async()) {
            return;
        }
        // flush the reads
        for (const auto & rinfo : rinfos) {
            ggml_backend_tensor_set(rinfo.tensor, rinfo.ptr, rinfo.offset, rinfo.size);
        }
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save for later during destruction
        rinfos.push_back({tensor, ptr, size, offset});

        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void discard() override {
        rinfos.clear();
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    llama_context * lctx = nullptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        const uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;
};

class llama_io_write_file : public llama_io_write_i {
public:
    llama_io_write_file(llama_file * f) : file(f) {}

    void write(const void * src, size_t size) override {
        file->write_raw(src, size);
        size_written += size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        ggml_backend_tensor_get(tensor, temp_buffer.data(), offset, size);
        write(temp_buffer.data(), temp_buffer.size());
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    llama_file * file;
    size_t size_written = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_read_file : public llama_io_read_i {
public:
    llama_io_read_file(llama_file * f) : file(f) {}

    void read(void * dst, size_t size) override {
        file->read_raw(dst, size);
        size_read += size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        read(temp_buffer.data(), size);
        ggml_backend_tensor_set(tensor, temp_buffer.data(), offset, size);
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    llama_file * file;
    size_t size_read = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_write_device : public llama_io_write_i {
public:
    llama_io_write_device(uint8_t * p, size_t len, llama_memory_buffers & mbufs, llama_context * lctx = nullptr)
        : ptr(p), buf_size(len), mbufs(mbufs), lctx(lctx) {
    }

    ~llama_io_write_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += winfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/ 2*mbuf.n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
            mbuf.cpy.reserve(mbuf.n_tensors);
        }

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            const int64_t n = winfo.size/ggml_element_size(winfo.tensor);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(ggml_view_1d      (mbuf.ctx.get(), winfo.tensor, n, winfo.offset));
            mbuf.cpy.push_back(ggml_new_tensor_1d(mbuf.ctx.get(), winfo.tensor->type, n));
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            auto & mbuf_cur = mbufs[buft];

            bool need_alloc = false;

            need_alloc = need_alloc || (!mbuf_cur.buf);
            need_alloc = need_alloc || (mbuf_cur.org.size() != mbuf.org.size());
            need_alloc = need_alloc || (mbuf_cur.total_size != mbuf.total_size);

            if (!need_alloc) {
                for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                    auto * org0 = mbuf_cur.org[i];
                    auto * org1 = mbuf.org[i];

                    if (!ggml_are_same_shape(org0, org1)) {
                        need_alloc = true;
                        break;
                    }

                    if (org0->view_src != org1->view_src || org0->view_offs != org1->view_offs) {
                        need_alloc = true;
                        break;
                    }
                }
            }

            if (need_alloc) {
                if (!mbuf_cur.buf || mbuf_cur.total_size != mbuf.total_size) {
                    mbuf_cur = std::move(mbuf);

                    mbuf_cur.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(mbuf_cur.ctx.get(), buft));

                    LLAMA_LOG_INFO("%s: allocated '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);
                } else {
                    //LLAMA_LOG_INFO("%s: reallocating tensors in '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);

                    // save the old buffer and allocate the new tensors in it
                    auto buf = std::move(mbuf_cur.buf);

                    mbuf_cur = std::move(mbuf);

                    ggml_tallocr talloc = ggml_tallocr_new(buf.get());

                    for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                        ggml_backend_view_init(mbuf_cur.org[i]);
                        ggml_tallocr_alloc(&talloc, mbuf_cur.cpy[i]);
                    }

                    mbuf_cur.buf = std::move(buf);
                }
            }

            // the pieces have to bounce through the host when the copy buffer is on a meta device: stage a band at
            // a time instead of one pageable allocation + synchronize per piece (r157, TODO #110)
            std::vector<llama_state_copy_job> jobs;
            if (llama_state_copy_bounces(mbuf_cur.buf.get())) {
                jobs.reserve(mbuf_cur.org.size());
                for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                    ggml_tensor * org = mbuf_cur.org[i];
                    ggml_tensor * src = org->view_src != nullptr ? org->view_src : org;
                    const size_t  off = org->view_src != nullptr ? org->view_offs : 0;
                    const size_t  n   = ggml_nbytes(org);
                    if (llama_state_copy_window_ok(src, off, n) && llama_state_copy_window_ok(mbuf_cur.cpy[i], 0, n)) {
                        jobs.push_back({ src, mbuf_cur.cpy[i], off, 0, n });
                    } else {
                        jobs.push_back({ org, mbuf_cur.cpy[i], 0, 0, n });
                    }
                }
            }
            if (!llama_state_copy_staged(lctx, jobs)) {
                for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                    ggml_backend_tensor_copy(mbuf_cur.org[i], mbuf_cur.cpy[i]);
                }
            }
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save the write for later during destruction
        winfos.push_back({tensor, ptr, size, offset});
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;

    llama_memory_buffers & mbufs;
    llama_context * lctx = nullptr;
};

class llama_io_read_device : public llama_io_read_i {
public:
    llama_io_read_device(const uint8_t * p, size_t len, const llama_memory_buffers & mbufs, llama_context * lctx = nullptr)
        : ptr(p), buf_size(len), mbufs(mbufs), lctx(lctx) {
    }

    ~llama_io_read_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += rinfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/ mbuf.n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
        }

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            const int64_t n = rinfo.size/ggml_element_size(rinfo.tensor);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(ggml_view_1d(mbuf.ctx.get(), rinfo.tensor, n, rinfo.offset));

            ggml_backend_view_init(mbuf.org.back());
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            const auto & mbuf_cur = mbufs.at(buft);

            if (!mbuf_cur.buf || mbuf_cur.total_size != mbuf.total_size) {
                GGML_ABORT("%s: memory buffer mismatch\n", __func__);
            }

            // copy the write-side data (mbuf_cur.cpy) into the read-side targets (mbuf.org) with a byte cursor.
            // Write and read enumerate the same logical data in the same order but may chunk it differently (even
            // with an equal number of tensors), e.g. save ranges [2,1] vs restore runs [1,2], so copy across
            // tensor boundaries rather than 1:1 by index.  The pieces are collected first: when the destination is
            // on a meta device the copy has to bounce through the host anyway, and staging a band of them through
            // the pinned state stage is what makes the restore fast (r157, TODO #110).
            const size_t total = mbuf_cur.total_size;

            ggml_init_params params_scratch = {
                /*.mem_size   =*/ 2*(mbuf_cur.cpy.size() + mbuf.org.size())*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };
            ggml_context * ctx_scratch = ggml_init(params_scratch);

            std::vector<llama_state_copy_job> jobs;
            jobs.reserve(mbuf_cur.cpy.size());

            size_t src_pos  = 0;
            size_t dst_pos  = 0;
            size_t src_j    = 0;
            size_t dst_i    = 0;
            size_t src_base = 0;
            size_t dst_base = 0;

            while (src_pos < total) {
                const auto & src_t = mbuf_cur.cpy[src_j];
                const auto & dst_t = mbuf.org[dst_i];

                const size_t src_size = ggml_nbytes(src_t);
                const size_t dst_size = ggml_nbytes(dst_t);

                const size_t src_off  = src_pos - src_base;
                const size_t dst_off  = dst_pos - dst_base;

                const size_t n_copy = std::min(src_size - src_off, dst_size - dst_off);

                // the piece is handed to the backend as the tensor itself plus an offset: a meta buffer splices
                // an aligned window with one strided copy per device, while a view of the piece walks its split
                // state per row (measured 4.5 ms vs 0.15 ms for 2 MiB)
                ggml_tensor * dst_root     = dst_t->view_src != nullptr ? dst_t->view_src : dst_t;
                const size_t  dst_root_off = dst_t->view_src != nullptr ? dst_t->view_offs : 0;

                jobs.push_back({ src_t, dst_root, src_off, dst_root_off + dst_off, n_copy });

                src_pos += n_copy;
                dst_pos += n_copy;

                if (src_pos - src_base == src_size) {
                    src_base = src_pos;
                    ++src_j;
                }
                if (dst_pos - dst_base == dst_size) {
                    dst_base = dst_pos;
                    ++dst_i;
                }
            }

            GGML_ASSERT(src_pos == total && dst_pos == total);
            // any tensors left unvisited hold no data
            for (size_t i = src_j; i < mbuf_cur.cpy.size(); ++i) {
                GGML_ASSERT(ggml_nbytes(mbuf_cur.cpy[i]) == 0);
            }
            for (size_t i = dst_i; i < mbuf.org.size(); ++i) {
                GGML_ASSERT(ggml_nbytes(mbuf.org[i]) == 0);
            }

            if (!llama_state_copy_bounces(mbuf_cur.buf.get()) || !llama_state_copy_staged(lctx, jobs)) {
                // a device that copies on its own needs two same-layout tensors, so a sub-range is expressed as
                // a view of the piece (the staged arm above hands the backend the window and its offset instead)
                for (const auto & job : jobs) {
                    if (job.src_off == 0 && job.dst_off == 0 &&
                        job.n == ggml_nbytes(job.src) && job.n == ggml_nbytes(job.dst)) {
                        ggml_backend_tensor_copy(job.src, job.dst);
                        continue;
                    }

                    const size_t   el   = ggml_element_size(job.src);
                    const int64_t n_el = (int64_t) (job.n / el);

                    auto * src_v = ggml_view_1d(ctx_scratch, job.src, n_el, job.src_off);
                    ggml_backend_view_init(src_v);
                    auto * dst_v = ggml_view_1d(ctx_scratch, job.dst, n_el, job.dst_off);
                    ggml_backend_view_init(dst_v);

                    ggml_backend_tensor_copy(src_v, dst_v);
                }
            }

            ggml_free(ctx_scratch);
        }

        GGML_ASSERT(buf_size == 0);
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save for later during destruction
        rinfos.push_back({tensor, ptr, size, offset});
    }

    void discard() override {
        rinfos.clear();
        buf_size = 0;
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        const uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;

    const llama_memory_buffers & mbufs;
    llama_context * lctx = nullptr;
};

uint8_t * llama_context::state_stage(size_t size) {
    if (buf_state_stage == nullptr || ggml_backend_buffer_get_size(buf_state_stage.get()) < size) {
        buf_state_stage.reset();
        // the pinned type of the output device (the first simple device of a meta device), as for the outputs
        // Fold (r96, PR #189): retained after the save (see the note above `llama_state_async_enabled`); it only
        // ever grows, so a small save after a large one keeps the large allocation.
        auto * dev = model.dev_output();
        ggml_backend_buffer_type_t buft = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
        if (buft == nullptr && dev != nullptr && ggml_backend_dev_is_meta(dev) && ggml_backend_meta_dev_n_devs(dev) > 0) {
            buft = ggml_backend_dev_host_buffer_type(ggml_backend_meta_dev_simple_dev(dev, 0));
        }
        if (buft == nullptr) {
            return nullptr;
        }
        const size_t gran = 16u << 20;
        buf_state_stage.reset(ggml_backend_buft_alloc_buffer(buft, (size + gran - 1) / gran * gran));
        if (buf_state_stage == nullptr) {
            return nullptr;
        }
    }
    return (uint8_t *) ggml_backend_buffer_get_base(buf_state_stage.get());
}

// Fold (r151, PR #233): the pipelined save's events, one per (backend, group), created on first use and kept
// for the context's lifetime (the backends it copies between do not change).  A device with no `event_new`
// (the CPU backend; any backend whose iface leaves events NULL) yields all-nullptr here, and the caller falls
// back to the single `llama_state_sync_all` - so this never turns a supported config into a failure.
ggml_backend_event_t llama_context::state_pipe_event(ggml_backend_t backend, size_t group) {
    GGML_ASSERT(group < n_state_pipe_events);
    for (auto & be : state_pipe_evs) {
        if (be.first == backend) {
            return be.second[group];
        }
    }
    std::array<ggml_backend_event_t, n_state_pipe_events> evs = {};
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    for (size_t g = 0; dev != nullptr && g < n_state_pipe_events; ++g) {
        if ((evs[g] = ggml_backend_event_new(dev)) == nullptr) {
            for (size_t k = 0; k < g; ++k) {
                ggml_backend_event_free(evs[k]);
                evs[k] = nullptr;
            }
            break;
        }
    }
    state_pipe_evs.emplace_back(backend, evs);   // all nullptr when the device has no events: the caller falls back
    return evs[group];
}

ggml_backend_t llama_context::state_backend(const ggml_tensor * t) {
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    if (buf == nullptr) {
        return nullptr;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf);
    for (auto & b : backends) {
        if (ggml_backend_supports_buft(b.get(), buft)) {
            return b.get();
        }
    }
    return nullptr;
}

size_t llama_context::state_get_size() {
    llama_io_write_dummy io(false);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_get_data(uint8_t * dst, size_t size) {
    llama_io_write_host io(dst, size, this);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_set_data(const uint8_t * src, size_t size) {
    llama_io_read_host io(src, size, this);
    try {
        return state_read_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        io.discard();
        return 0;
    }
}

static constexpr uint32_t io_magic = 0xaf143cd8;

size_t llama_context::state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags) {
    llama_io_write_dummy io(flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
    try {
        io.write(&io_magic, sizeof(io_magic));
        io.write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_get_data(llama_seq_id seq_id, uint8_t * dst, size_t size, llama_state_seq_flags flags) {
    std::unique_ptr<llama_io_write_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        io = std::make_unique<llama_io_write_device>(dst, size, mem_storage[seq_id], this);
    } else {
        io = std::make_unique<llama_io_write_host>(dst, size, this);
    }

    try {
        io->write(&io_magic, sizeof(io_magic));
        io->write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(*io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags) {
    std::unique_ptr<llama_io_read_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        // create a temporary io to read the magic and the src seq_id
        io = std::make_unique<llama_io_read_host>(src, size, this);

        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));

        GGML_ASSERT(mem_storage.find(seq_id_read) != mem_storage.end());

        io = std::make_unique<llama_io_read_device>(src, size, mem_storage[seq_id_read], this);
    } else {
        io = std::make_unique<llama_io_read_host>(src, size, this);
    }

    try {
        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));

        return state_seq_read_data(*io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        io->discard();
        return 0;
    }
}

bool llama_context::state_load_file(const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // sanity checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_SESSION_MAGIC || version != LLAMA_SESSION_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for session file: %08x, %08x\n", __func__, magic, version);
            return false;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in session file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return false;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t n_state_size_cur = file.size() - file.tell();

        llama_io_read_file io( &file);
        const size_t n_read = state_read_data(io);

        if (n_read != n_state_size_cur) {
            LLAMA_LOG_ERROR("%s: did not read all of the session file data! size %zu, got %zu\n", __func__, n_state_size_cur, n_read);
            return false;
        }
    }

    return true;
}

bool llama_context::state_save_file(const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_SESSION_MAGIC);
    file.write_u32(LLAMA_SESSION_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_write_data(io);

    return true;
}

size_t llama_context::state_seq_load_file(llama_seq_id seq_id, const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // version checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_STATE_SEQ_MAGIC || version != LLAMA_STATE_SEQ_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for sequence state file: %08x, %08x\n", __func__, magic, version);
            return 0;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (tokens_out == nullptr) {
            const size_t n_token_max = (file.size() - file.tell()) / sizeof(llama_token);
            if (n_token_count > n_token_max) {
                LLAMA_LOG_ERROR("%s: token count in sequence state file exceeds the file size! %u > %zu\n", __func__, n_token_count, n_token_max);
                return 0;
            }

            *n_token_count_out = n_token_count;
            return file.tell();
        }

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in sequence state file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return 0;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t state_size = file.size() - file.tell();
        llama_io_read_file io(&file);
        const size_t nread = state_seq_read_data(io, seq_id, 0);
        if (!nread) {
            LLAMA_LOG_ERROR("%s: failed to restore sequence state\n", __func__);
            return 0;
        }
        GGML_ASSERT(nread <= state_size);
        GGML_ASSERT(nread + sizeof(uint32_t) * 3 + sizeof(llama_token) * *n_token_count_out == file.tell());
    }

    return file.tell();
}

size_t llama_context::state_seq_save_file(llama_seq_id seq_id, const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_STATE_SEQ_MAGIC);
    file.write_u32(LLAMA_STATE_SEQ_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_seq_write_data(io, seq_id, 0);

    const size_t res = file.tell();
    GGML_ASSERT(res == sizeof(uint32_t) * 3 + sizeof(llama_token) * n_token_count + io.n_bytes());

    return res;
}

size_t llama_context::state_write_data(llama_io_write_i & io) {
    LLAMA_LOG_DEBUG("%s: writing state\n", __func__);

    // write model info
    {
        LLAMA_LOG_DEBUG("%s: - writing model info\n", __func__);

        const std::string arch_str = llm_arch_name(model.arch);
        io.write_string(arch_str);
        // TODO: add more model-specific info which should prevent loading the session file if not identical
    }

    if (memory != nullptr) {
        LLAMA_LOG_DEBUG("%s: - writing memory module\n", __func__);
        memory->state_write(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_read_data(llama_io_read_i & io) {
    LLAMA_LOG_DEBUG("%s: reading state\n", __func__);

    // read model info
    {
        LLAMA_LOG_DEBUG("%s: - reading model info\n", __func__);

        const std::string cur_arch_str = llm_arch_name(model.arch);

        std::string arch_str;
        io.read_string(arch_str);
        if (cur_arch_str != arch_str) {
            throw std::runtime_error(format("wrong model arch: '%s' instead of '%s'", arch_str.c_str(), cur_arch_str.c_str()));
        }
        // TODO: add more info which needs to be identical but which is not verified otherwise
    }

    if (memory) {
        LLAMA_LOG_DEBUG("%s: - reading memory module\n", __func__);

        memory->state_read(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if (memory) {
        memory->state_write(io, seq_id, flags);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_read_data(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if (memory) {
        memory->state_read(io, seq_id, flags);
    }

    return io.n_bytes();
}

//
// perf
//

llama_perf_context_data llama_context::perf_get_data() const {
    llama_perf_context_data data = {};

    data.t_start_ms  = 1e-3 * t_start_us;
    data.t_load_ms   = 1e-3 * t_load_us;
    data.t_p_eval_ms = 1e-3 * t_p_eval_us;
    data.t_eval_ms   = 1e-3 * t_eval_us;
    data.n_p_eval    = std::max(1, n_p_eval);
    data.n_eval      = std::max(1, n_eval);
    data.n_reused    = std::max(0, n_reused);

    data.n_reallocs_same = n_reallocs_same;
    data.n_reallocs_diff = n_reallocs_diff;

    return data;
}

void llama_context::perf_reset() {
    t_start_us  = ggml_time_us();
    t_eval_us   = n_eval = 0;
    t_p_eval_us = n_p_eval = 0;
    n_reused    = 0;

    // note: the reallocation counters are not reset - they track the lifetime totals for this context
}

// The FA prefill staging arena (block 15) is a raw device allocation outside the compute-graph
// reserve, so llama_get_memory_breakdown (and therefore --fit) does not include it.  A live context
// reports what the arenas have actually grown to; the no_alloc (--fit) path reserves the bounded
// worst case the runtime can request for this context, so the fit cannot under-provision a deep
// prefill.  The provider is the CUDA backend (the only backend with this arena); every other
// backend returns 0.  `f16_kv_bytes` is the F16 attention-KV size the context could stage.
static size_t llama_fattn_stage_accounting(ggml_backend_t backend, bool reserve, size_t f16_kv_bytes) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    if (reg == nullptr) {
        return 0;
    }
    if (!reserve) {
        auto used_fn = (size_t (*)(ggml_backend_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_fattn_stage_used");
        return used_fn != nullptr ? used_fn(backend) : 0;
    }
    auto bound_fn = (size_t (*)(ggml_backend_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_fattn_stage_bound");
    if (bound_fn == nullptr) {
        return 0;
    }
    const size_t bound = bound_fn(backend);
    if (bound == 0) {
        return 0;                   // this device never prefill-stages
    }
    if (bound == std::numeric_limits<size_t>::max()) {
        return f16_kv_bytes;        // unbounded cap: reserve the realistic high-water mark
    }
    return std::min(bound, f16_kv_bytes);
}

// The op-offload H2D staging ring (block 06) is also a raw device allocation outside the compute
// reserve.  `reserve` (the --fit / no_alloc path) uses the worst-case bound for a model whose largest
// host-resident weight tensor is `max_table` bytes (the ring holds one table per slot); the live path
// reports what the ring has actually grown to.  Every non-CUDA backend returns 0.
static size_t llama_h2d_stage_accounting(ggml_backend_t backend, bool reserve, size_t max_table) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    if (reg == nullptr) {
        return 0;
    }
    if (!reserve) {
        auto used_fn = (size_t (*)(ggml_backend_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_h2d_stage_used");
        return used_fn != nullptr ? used_fn(backend) : 0;
    }
    auto bound_fn = (size_t (*)(ggml_backend_t, size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_h2d_stage_bound");
    return bound_fn != nullptr ? bound_fn(backend, max_table) : 0;
}

llama_memory_breakdown llama_context::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> ret;
    for (const auto & [buft, size] : model.memory_breakdown()) {
        ret[buft].model += size;
    }
    if (memory) {
        for (const auto & [buft, size] : memory->memory_breakdown()) {
            ret[buft].context += size;
        }
    }
    // F16 attention-KV size the FA prefill staging could copy for this context (used only to bound
    // the fit's arena reserve; the cap is the hard bound).  Only the types the launcher stages
    // (anything a native-capable arm can read) are counted.
    const bool fa_stages =
            cparams.flash_attn && cparams.type_k != GGML_TYPE_F16 && cparams.type_k != GGML_TYPE_F32;
    size_t f16_kv_bytes = 0;
    if (model.hparams.no_alloc && fa_stages) {
        for (uint32_t il = 0; il < model.hparams.n_layer_all; ++il) {
            f16_kv_bytes += (size_t) cparams.n_ctx * model.hparams.n_embd_k_gqa(il) * sizeof(ggml_fp16_t);
            f16_kv_bytes += (size_t) cparams.n_ctx * model.hparams.n_embd_v_gqa(il) * sizeof(ggml_fp16_t);
        }
    }

    if (moe_cache) {
        for (const auto & [buft, size] : moe_cache->memory_breakdown()) {
            ret[buft].context += size;
        }
    }
    if (model.hparams.no_alloc) {
        const size_t max_host_table = model.max_host_weight_tensor_bytes();
        for (size_t i = 0; i < backends.size(); ++i) {
            ggml_backend_t             backend = backends[i].get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += backend_buf_exp_size[i] + llama_fattn_stage_accounting(backend, true, f16_kv_bytes)
                               + llama_h2d_stage_accounting(backend, true, max_host_table);
        }
    } else {
        for (const auto & backend_ptr : backends) {
            ggml_backend_t             backend = backend_ptr.get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += ggml_backend_sched_get_buffer_size(sched.get(), backend) + llama_fattn_stage_accounting(backend, false, 0)
                               + llama_h2d_stage_accounting(backend, false, 0);
        }
    }
    return ret;
}

//
// training
//

static void llama_set_param(struct ggml_tensor * tensor, llama_opt_param_filter param_filter, void * userdata) {
    if (!tensor || tensor->type != GGML_TYPE_F32) {
        return;
    }
    if (!param_filter(tensor, userdata)) {
        return;
    }
    if (strcmp(tensor->name, "token_embd.weight") == 0) {
        return; // FIXME
    }
    if (strcmp(tensor->name, "rope_freqs.weight") == 0) {
        return; // FIXME
    }
    ggml_set_param(tensor);
}

void llama_context::opt_init(struct llama_model * model_l, struct llama_opt_params lopt_params) {
    GGML_ASSERT(!opt_ctx);
    model_l->hparams.n_ctx_train = lopt_params.n_ctx_train > 0 ? lopt_params.n_ctx_train : n_ctx();
    const uint32_t n_batch     = std::min(this->n_batch(),  model_l->hparams.n_ctx_train);
    const uint32_t n_ubatch    = std::min(this->n_ubatch(), n_batch);
    GGML_ASSERT(model_l->hparams.n_ctx_train % n_batch  == 0);
    GGML_ASSERT(n_batch                    % n_ubatch == 0);

    if (cparams.flash_attn) {
        LLAMA_LOG_INFO("%s: disabling flash attention, FLASH_ATTN_EXT has no backward pass\n", __func__);
        cparams.flash_attn = false;
    }

    // gradients cannot flow through the KV cache, so the attention reads the K and V of the current ubatch directly
    if (n_ubatch == cparams.n_ctx) {
        cparams.training = true;
    } else {
        LLAMA_LOG_WARN("%s: n_ubatch (%u) != n_ctx (%u), the K and V projections will not receive gradients\n", __func__, n_ubatch, cparams.n_ctx);
    }

    // the training graph is different, need to reserve again
    sched_need_reserve = true;
    sched_reserve();

    ggml_opt_params opt_params = ggml_opt_default_params(sched.get(), GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
    opt_params.opt_period      = n_batch / n_ubatch;
    opt_params.get_opt_pars    = lopt_params.get_opt_pars;
    opt_params.get_opt_pars_ud = lopt_params.get_opt_pars_ud;
    opt_params.optimizer       = lopt_params.optimizer_type;
    opt_ctx = ggml_opt_init(opt_params);

    llama_opt_param_filter param_filter = lopt_params.param_filter;
    void * param_filter_ud              = lopt_params.param_filter_ud;

  //llama_set_param(model->tok_embd,        param_filter, param_filter_ud); // FIXME
    llama_set_param(model_l->type_embd,       param_filter, param_filter_ud);
    llama_set_param(model_l->pos_embd,        param_filter, param_filter_ud);
    llama_set_param(model_l->tok_norm,        param_filter, param_filter_ud);
    llama_set_param(model_l->tok_norm_b,      param_filter, param_filter_ud);
    llama_set_param(model_l->output_norm,     param_filter, param_filter_ud);
    llama_set_param(model_l->output_norm_b,   param_filter, param_filter_ud);
    llama_set_param(model_l->output,          param_filter, param_filter_ud);
    llama_set_param(model_l->output_b,        param_filter, param_filter_ud);
    llama_set_param(model_l->output_norm_enc, param_filter, param_filter_ud);
    llama_set_param(model_l->cls,             param_filter, param_filter_ud);
    llama_set_param(model_l->cls_b,           param_filter, param_filter_ud);
    llama_set_param(model_l->cls_out,         param_filter, param_filter_ud);
    llama_set_param(model_l->cls_out_b,       param_filter, param_filter_ud);
    llama_set_param(model_l->cls_norm,        param_filter, param_filter_ud);

    for (struct llama_layer & layer : model_l->layers) {
        for (size_t i = 0; i < sizeof(layer)/sizeof(struct ggml_tensor *); ++i) {
            llama_set_param(reinterpret_cast<struct ggml_tensor **>(&layer)[i], param_filter, param_filter_ud);
        }
    }
}

void llama_context::opt_epoch_iter(
        ggml_opt_dataset_t               dataset,
        ggml_opt_result_t                result,
        const std::vector<llama_token> & tokens,
        const std::vector<llama_token> & labels_sparse,
        llama_batch                    & batch,
        ggml_opt_epoch_callback          callback,
        bool                             train,
        int64_t                          idata_in_loop,
        int64_t                          ndata_in_loop,
        int64_t                          t_loop_start) {
    GGML_ASSERT(opt_ctx);
    const uint32_t n_ctx    = llama_model_n_ctx_train(&model);
    const uint32_t n_batch  = std::min(this->n_batch(),  n_ctx);
    const uint32_t n_ubatch = std::min(this->n_ubatch(), n_batch);

    memory->clear(true);

    for (uint32_t pos_ctx = 0; pos_ctx < n_ctx; pos_ctx += n_batch) {
        batch.n_tokens = n_batch;
        for (uint32_t pos_batch = 0; pos_batch < n_batch; ++pos_batch) {
            batch.token   [pos_batch]    = tokens[pos_ctx + pos_batch];
            batch.pos     [pos_batch]    = pos_ctx + pos_batch;
            batch.n_seq_id[pos_batch]    = 1;
            batch.seq_id  [pos_batch][0] = 0;
            batch.logits  [pos_batch]    = true;
        }

        // TODO: use llama_batch_ext here
        {
            llama_batch_compat compat(this, batch);
            if (!balloc->init(*compat.batch_ext, model.vocab, true)) {
                LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
                return;
            }
        }

        const uint32_t n_tokens_all = balloc->get_n_tokens();

        n_queued_tokens += n_tokens_all;

        embd_seq.clear();

        uint32_t n_outputs_all = n_tokens_all;

        auto mctx = memory->init_batch(*balloc, cparams.n_ubatch, true);
        if (!mctx || mctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: could not initialize batch\n", __func__);
            break;
        }

        // reserve output buffer
        if (output_reserve(n_outputs_all) < n_outputs_all) {
            LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
            GGML_ABORT("TODO: handle this error");
        };

        uint32_t pos_batch = 0;
        do {
            const auto & ubatch = mctx->get_ubatch();

            n_outputs = ubatch.n_tokens;

            if (!mctx->apply()) {
                LLAMA_LOG_ERROR("%s: failed to update the memory context\n", __func__);
                break;
            }

            // the trainer has no draft context and therefore no second scheduler: keep the slot the
            // output count selects, as before
            auto * res = get_gf_res_prev(n_outputs > 0);

            const auto gparams = graph_params(res, ubatch, mctx.get(), ctx_type_to_graph_type(cparams.ctx_type));

            // the optimizer graph is allocated outside sched, so the next decode must rebuild
            gf_res_prev_active = nullptr;
            gf_res_prev_active_alt = nullptr;
            res->reset();

            auto * gf = model.build_graph(gparams);

            struct ggml_context * ctx_compute_opt;
            {
                const size_t size_gf = ggml_graph_size(gf);
                const size_t size_meta = 4*size_gf*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(size_gf, /*grads = */ true);
                struct ggml_init_params params = {
                    /*.mem_size   =*/ size_meta,
                    /*.mem_buffer =*/ nullptr,
                    /*.no_alloc   =*/ true,
                };
                ctx_compute_opt = ggml_init(params);
            }
            ggml_opt_prepare_alloc(opt_ctx, ctx_compute_opt, gf, res->get_inp_tokens(), res->get_logits());
            ggml_opt_alloc(opt_ctx, train);

            res->set_inputs(&ubatch);
            {
                struct ggml_tensor * labels = ggml_opt_labels(opt_ctx);
                GGML_ASSERT(labels->ne[1] == n_ubatch);
                ggml_set_zero(labels);
                const float onef = 1.0f;
                for (uint32_t pos_ubatch = 0; pos_ubatch < n_ubatch; ++pos_ubatch) {
                    const uint32_t ilabel = pos_ctx + pos_batch + pos_ubatch;
                    GGML_ASSERT(labels_sparse[ilabel] < labels->ne[0]);
                    ggml_backend_tensor_set(labels, &onef, (pos_ubatch*labels->ne[0] + labels_sparse[ilabel])*sizeof(float), sizeof(float));
                }
            }
            ggml_opt_eval(opt_ctx, result);
            if (callback) {
                callback(train, opt_ctx, dataset, result, idata_in_loop + (pos_ctx + pos_batch)/n_ubatch + 1, ndata_in_loop, t_loop_start);
            }
            ggml_free(ctx_compute_opt);

            pos_batch += ubatch.n_tokens;
        } while (mctx->next());
    }
}

void llama_context::opt_epoch(
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    const uint32_t n_ctx    = this->n_ctx();
    const uint32_t n_batch  = std::min(cparams.n_batch,  n_ctx);
    const uint32_t n_ubatch = std::min(cparams.n_ubatch, n_batch);
    const  int64_t ndata    = ggml_opt_dataset_ndata(dataset);

    GGML_ASSERT(idata_split >= 0);
    GGML_ASSERT(idata_split <= ndata);

    const uint32_t ubatch_per_ctx = n_ctx / n_ubatch;

    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);
    std::vector<llama_token>        tokens(n_ctx);
    std::vector<llama_token> labels_sparse(n_ctx);

    int64_t idata = 0;

    int64_t t_loop_start = ggml_time_us();
    int64_t ndata_in_loop = idata_split*ubatch_per_ctx;
    for (; idata < idata_split; ++idata) {
        constexpr bool train = true;
        const int64_t idata_in_loop = idata*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_train, tokens, labels_sparse, batch,
            callback_train, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    t_loop_start = ggml_time_us();
    ndata_in_loop = (ndata - idata_split)*ubatch_per_ctx;
    for (; idata < ndata; ++idata) {
        constexpr bool train = false;
        const int64_t idata_in_loop = (idata - idata_split)*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_eval, tokens, labels_sparse, batch,
            callback_eval, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    llama_batch_free(batch);
}

//
// interface implementation
//

llama_context_params llama_context_default_params() {
    llama_context_params result = {
        /*.n_ctx                       =*/ 512,
        /*.n_batch                     =*/ 2048,
        /*.n_ubatch                    =*/ 512,
        /*.n_seq_max                   =*/ 1,
        /*.n_rs_seq                    =*/ 0,
        /*.n_rs_batch                  =*/ 0,
        /*.n_outputs_max               =*/ 0,
        /*.n_outputs_max_per_seq       =*/ 1,
        /*.n_threads                   =*/ GGML_DEFAULT_N_THREADS, // TODO: better default
        /*.n_threads_batch             =*/ GGML_DEFAULT_N_THREADS,
        /*.mtp_draft_vocab             =*/ 0,
        /*.ctx_type                    =*/ LLAMA_CONTEXT_TYPE_DEFAULT,
        /*.rope_scaling_type           =*/ LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
        /*.pooling_type                =*/ LLAMA_POOLING_TYPE_UNSPECIFIED,
        /*.attention_type              =*/ LLAMA_ATTENTION_TYPE_UNSPECIFIED,
        /*.flash_attn_type             =*/ LLAMA_FLASH_ATTN_TYPE_AUTO,
        /*.rope_freq_base              =*/ 0.0f,
        /*.rope_freq_scale             =*/ 0.0f,
        /*.yarn_ext_factor             =*/ -1.0f,
        /*.yarn_attn_factor            =*/ -1.0f,
        /*.yarn_beta_fast              =*/ -1.0f,
        /*.yarn_beta_slow              =*/ -1.0f,
        /*.yarn_orig_ctx               =*/ 0,
        /*.defrag_thold                =*/ -1.0f,
        /*.cb_eval                     =*/ nullptr,
        /*.cb_eval_user_data           =*/ nullptr,
        /*.type_k                      =*/ GGML_TYPE_F16,
        /*.type_v                      =*/ GGML_TYPE_F16,
        /*.moe_cache_size              =*/ 0,
        /*.abort_callback              =*/ nullptr,
        /*.abort_callback_data         =*/ nullptr,
        /*.embeddings                  =*/ false,
        /*.offload_kqv                 =*/ true,
        /*.no_perf                     =*/ true,
        /*.op_offload                  =*/ true,
        /*.swa_full                    =*/ true,
        /*.kv_unified                  =*/ false,
        /*.drop_compute_buffers        =*/ false,
        /*.pos_2d                      =*/ false,
        /*.sampler                     =*/ nullptr,
        /*.n_sampler                   =*/ 0,
        /*.ctx_other                   =*/ nullptr,
    };

    return result;
}

// True for the KV cache types whose flash-attention kernels have a native read path on the
// HIP/CUDA backends and can therefore be used with SPLIT_MODE_TENSOR (see the gate below).
// Keep in sync with ggml_cuda_fattn_kv_type_supported() in ggml/src/ggml-cuda/fattn.cu.
static bool llama_kv_type_has_native_fa(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ4_NL:
            return true;
        default:
            return false;
    }
}

llama_context * llama_init_from_model(
                 llama_model * model,
        llama_context_params   params) {
    if (!model) {
        LLAMA_LOG_ERROR("%s: model cannot be NULL\n", __func__);
        return nullptr;
    }

    if (params.n_batch == 0 && params.n_ubatch == 0) {
        LLAMA_LOG_ERROR("%s: n_batch and n_ubatch cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.n_ctx == 0 && model->hparams.n_ctx_train == 0) {
        LLAMA_LOG_ERROR("%s: n_ctx and model->hparams.n_ctx_train cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && model->arch == LLM_ARCH_GROK) {
        LLAMA_LOG_WARN("%s: flash_attn is not compatible with Grok - forcing off\n", __func__);
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }

    if (llm_split_mode_is_tensor_like(model->split_mode())) {
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            LLAMA_LOG_INFO("%s: enabling flash_attn since it is required for SPLIT_MODE_TENSOR\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
            LLAMA_LOG_ERROR("%s: SPLIT_MODE_TENSOR requires flash_attn to be enabled\n", __func__);
            return nullptr;
        }
        if (model->get_split_state_ud.n_devices == 1) {
            LLAMA_LOG_WARN("%s: SPLIT_MODE_TENSOR being used for a single device is not recommended\n", __func__);
        }

        // SPLIT_MODE_TENSOR runs the attention through the flash-attention kernels split across
        // the tensor-parallel devices. Those kernels have a native read path only for a subset of
        // quantized types (see llama_kv_type_has_native_fa below, which must mirror
        // ggml_cuda_fattn_kv_type_supported() in ggml/src/ggml-cuda/fattn.cu); for the rest
        // (the k-quants and the remaining i-quants) the graph cannot express a splittable
        // attention and the meta splitter aborts during reserve. This is only a problem when
        // the Meta device is actually used
        // (tensor split over >= 2 GPUs; a single-device "tensor" mode skips the Meta wrapper and
        // is fine).
        const bool use_meta_device =
            [&]() {
                for (const auto & dev : model->devices) {
                    if (dev.is_meta) {
                        return true;
                    }
                }
                return false;
            }();
        if (use_meta_device) {
            for (ggml_type type_kv : { params.type_k, params.type_v }) {
                if (ggml_is_quantized(type_kv) && !llama_kv_type_has_native_fa(type_kv)) {
                    LLAMA_LOG_ERROR("%s: KV cache type %s is not supported with tensor parallelism "
                            "(SPLIT_MODE_TENSOR, %zu GPUs): flash attention cannot read this type from a "
                            "split KV cache\n", __func__, ggml_type_name(type_kv), model->get_split_state_ud.n_devices);
                    LLAMA_LOG_ERROR("%s: use f32/f16/bf16/q4_0/q4_1/q5_0/q5_1/q8_0/iq4_nl KV cache types, "
                            "or switch to SPLIT_MODE_LAYER which supports every KV cache type\n", __func__);
                    return nullptr;
                }
            }
        }
    }

    // Every model requires the same K and V cache types (rdna-boosts, 2026-09-11).  Upstream enforces
    // this for MLA and DeepSeek4 only; here it is general: every mixed pair measured 1.7-3.6x slower
    // than the same-type equivalent and never smaller, and the attention path - including the
    // split/flash-attention one - assumes type_k == type_v, so a mixed configuration is rejected
    // rather than accepted as a slow-but-valid one.  Both types default to f16, so only an explicit
    // --cache-type-k (or --cache-type-v) can trigger this.
    if (params.type_k != params.type_v) {
        LLAMA_LOG_ERROR("%s: models require the same K and V cache types, got K=%s and V=%s; "
                "set --cache-type-v to match --cache-type-k (both default to f16)\n",
                __func__, ggml_type_name(params.type_k), ggml_type_name(params.type_v));
        return nullptr;
    }

    if (ggml_is_quantized(params.type_v) && params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            LLAMA_LOG_INFO("%s: enabling flash_attn since it is required for quantized V cache\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED) {
            LLAMA_LOG_ERROR("%s: quantized V cache requires flash_attn to be enabled\n", __func__);
            return nullptr;
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_k)) {
        const uint32_t blck_size = ggml_blck_size(params.type_k);
        for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
            if (model->hparams.n_embd_head_k(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: K cache type %s with block size %u does not divide n_embd_head_k=%u\n",
                    __func__, ggml_type_name(params.type_k), blck_size, model->hparams.n_embd_head_k(il));
                return nullptr;
            }
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_v)) {
        const uint32_t blck_size = ggml_blck_size(params.type_v);
        for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
            if (model->hparams.n_embd_head_v(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: V cache type %s with block size %u does not divide n_embd_head_v=%u\n",
                    __func__, ggml_type_name(params.type_v), blck_size, model->hparams.n_embd_head_v(il));
                return nullptr;
            }
        }
    }

    if (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED &&
        params.pooling_type != model->hparams.pooling_type) {
        //user-specified pooling-type is different from the model default
        LLAMA_LOG_WARN("%s: model default pooling_type is [%d], but [%d] was specified\n", __func__,
                       model->hparams.pooling_type, params.pooling_type);
    }

    // router_layer >= 0 means n_layer_nextn is repurposed for a router layer, not real MTP
    if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP &&
        (model->hparams.n_layer_nextn == 0 || model->hparams.router_layer >= 0)) {
        LLAMA_LOG_WARN("%s: context type MTP requested but model doesn't contain MTP layers\n", __func__);
        return nullptr;
    }

    try {
        auto * ctx = new llama_context(*model, params);
        const auto & cparams = ctx->get_cparams();

        if (cparams.rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN && cparams.rope_freq_scale != model->hparams.rope_freq_scale_train) {
            LLAMA_LOG_INFO("%s: custom YaRN scaling detected, re-adjusting n_ctx_train(%u)...\n", __func__, model->hparams.n_ctx_train);
            model->hparams.n_ctx_train = cparams.n_ctx_orig_yarn / cparams.rope_freq_scale;
            LLAMA_LOG_INFO("%s: n_ctx_train adjusted to %u\n", __func__, model->hparams.n_ctx_train);
        }

        return ctx;
    } catch (const std::exception & err) {
        // `--fit` probes the extra (draft) model by building its context BEFORE the target context exists, so a
        // model whose graph always needs the target's hidden state (ctx_other) fails here BY DESIGN; the fit
        // falls back to the derived reserve.  That case is expected, so report it as a warning; anything else
        // (including a probe that failed for a real reason) stays an error.  The message is matched rather
        // than the arch because the same arch can construct fine when it does not need ctx_other.
        if (err.what() != nullptr && strstr(err.what(), "requires ctx_other to be set") != nullptr) {
            LLAMA_LOG_WARN("%s: failed to initialize the context: %s\n", __func__, err.what());
        } else {
            LLAMA_LOG_ERROR("%s: failed to initialize the context: %s\n", __func__, err.what());
        }
    }

    return nullptr;
}

// deprecated
llama_context * llama_new_context_with_model(
                 llama_model * model,
        llama_context_params   params) {
    return llama_init_from_model(model, params);
}

void llama_free(llama_context * ctx) {
    delete ctx;
}

uint32_t llama_n_ctx(const llama_context * ctx) {
    return ctx->n_ctx();
}

uint32_t llama_n_ctx_seq(const llama_context * ctx) {
    return ctx->n_ctx_seq();
}

uint32_t llama_n_batch(const llama_context * ctx) {
    return ctx->n_batch();
}

uint32_t llama_n_ubatch(const llama_context * ctx) {
    return ctx->n_ubatch();
}

uint32_t llama_n_seq_max(const llama_context * ctx) {
    return ctx->n_seq_max();
}

uint32_t llama_n_rs_seq(const llama_context * ctx) {
    return ctx->get_cparams().n_rs_seq;
}

const llama_model * llama_get_model(const llama_context * ctx) {
    return &ctx->get_model();
}

enum llama_pooling_type llama_pooling_type(const llama_context * ctx) {
    return ctx->pooling_type();
}

void llama_attach_threadpool(
            llama_context * ctx,
        ggml_threadpool_t   threadpool,
        ggml_threadpool_t   threadpool_batch) {
    ctx->attach_threadpool(threadpool, threadpool_batch);
}

void llama_detach_threadpool(llama_context * ctx) {
    ctx->detach_threadpool();
}

void llama_set_n_threads(llama_context * ctx, int32_t n_threads, int32_t n_threads_batch) {
    ctx->set_n_threads(n_threads, n_threads_batch);
}

int32_t llama_n_threads(llama_context * ctx) {
    return ctx->n_threads();
}

int32_t llama_n_threads_batch(llama_context * ctx) {
    return ctx->n_threads_batch();
}

void llama_set_abort_callback(llama_context * ctx, bool (*abort_callback)(void * data), void * abort_callback_data) {
    ctx->set_abort_callback(abort_callback, abort_callback_data);
}

void llama_set_embeddings(llama_context * ctx, bool embeddings) {
    ctx->set_embeddings(embeddings);
}

void llama_set_causal_attn(llama_context * ctx, bool causal_attn) {
    ctx->set_causal_attn(causal_attn);
}

bool llama_get_causal_attn(const llama_context * ctx) {
    return ctx->get_causal_attn();
}

void llama_set_warmup(llama_context * ctx, bool warmup) {
    ctx->set_warmup(warmup);
}

void llama_synchronize(llama_context * ctx) {
    ctx->synchronize();
}

float * llama_get_logits(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_logits();
}

float * llama_get_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    float * res = nullptr;

    res = ctx->get_sampled_logits_ith(i);

    if (!res) {
        res = ctx->get_logits_ith(i);
    }

    return res;
}

float * llama_get_embeddings(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings();
}

float * llama_get_embeddings_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_ith(i);
}

float * llama_get_embeddings_seq(llama_context * ctx, llama_seq_id seq_id) {
    ctx->synchronize();

    return ctx->get_embeddings_seq(seq_id);
}

void llama_set_embeddings_nextn_from(llama_context * ctx, llama_seq_id seq_id, llama_pos pos) {
    ctx->set_embeddings_nextn_from(seq_id, pos);
}

void llama_set_embeddings_nextn(llama_context * ctx, bool value, bool masked) {
    ctx->set_embeddings_nextn(value, masked);
}

void llama_set_embeddings_layer_inp(llama_context * ctx, uint32_t lid, bool value) {
    ctx->set_embeddings_layer_inp(lid, value);
}

void llama_set_nextn_layer_offset(llama_context * ctx, int32_t offset) {
    ctx->set_nextn_layer_offset(offset);
}

llama_memory_t llama_get_memory(const struct llama_context * ctx) {
    if (!ctx) {
        return nullptr;
    }

    return ctx->get_memory();
}

float * llama_get_embeddings_nextn(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings_nextn();
}

float * llama_get_embeddings_nextn_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_nextn_ith(i);
}

void llama_lf_set_layer_inp_dev(llama_context * ctx, bool enable) {
    ctx->set_lf_layer_inp_dev(enable);
}

ggml_tensor * llama_lf_get_layer_inp_dev(llama_context * ctx, uint32_t lid) {
    return ctx->get_lf_layer_inp_dev(lid);
}

const int32_t * llama_lf_get_dev_rows(llama_context * ctx, int32_t * n_rows) {
    return ctx->get_lf_dev_rows(n_rows);
}

void llama_lf_set_dev_inject(llama_context * ctx, bool enable) {
    ctx->set_lf_dev_inject(enable);
}

void llama_lf_signal_features_consumed(llama_context * ctx_dft) {
    ctx_dft->lf_signal_features_consumed();
}

float * llama_get_embeddings_layer_inp(llama_context * ctx, uint32_t lid) {
    ctx->synchronize();

    return ctx->get_embeddings_layer_inp(lid);
}

bool llama_set_sampler(llama_context * ctx, llama_seq_id seq_id, llama_sampler * smpl) {
    return ctx->set_sampler(seq_id, smpl);
}

llama_token llama_get_sampled_token_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_token_ith(i);
}

float * llama_get_sampled_probs_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_probs_ith(i);
}

float * llama_get_sampled_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_logits_ith(i);
}

llama_token * llama_get_sampled_candidates_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return const_cast<llama_token *>(ctx->get_sampled_candidates_ith(i));
}

uint32_t llama_get_sampled_candidates_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_candidates_count(i));
}

uint32_t llama_get_sampled_logits_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_logits_count(i));
}

uint32_t llama_get_sampled_probs_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_probs_count(i));
}

struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs) {
    auto memory = ctx->get_memory();
    llama_memory_context_ptr mctx;
    if (memory) {
        mctx = memory->init_full();
    }
    return ctx->graph_reserve(n_tokens, n_seqs, n_outputs, mctx.get());
}

// llama adapter API

int32_t llama_set_adapters_lora(
            llama_context * ctx,
            llama_adapter_lora ** adapters,
            size_t n_adapters,
            float * scales) {
    if (adapters == nullptr || scales == nullptr) {
        GGML_ASSERT(n_adapters == 0 && "invalid llama_set_adapters_lora call");
    }

    ctx->set_adapters_lora(adapters, n_adapters, scales);

    return 0;
}

int32_t llama_set_adapter_cvec(
        llama_context * ctx,
          const float * data,
               size_t   len,
              int32_t   n_embd,
              int32_t   il_start,
              int32_t   il_end) {
    bool res = ctx->set_adapter_cvec(data, len, n_embd, il_start, il_end);

    return res ? 0 : -1;
}

//
// memory
//

void llama_memory_clear(llama_memory_t mem, bool data) {
    if (!mem) {
        return;
    }

    mem->clear(data);
}

bool llama_memory_seq_rm(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return true;
    }

    return mem->seq_rm(seq_id, p0, p1);
}

void llama_memory_seq_cp(
        llama_memory_t mem,
          llama_seq_id seq_id_src,
          llama_seq_id seq_id_dst,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return;
    }

    mem->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_seq_keep(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return;
    }

    mem->seq_keep(seq_id);
}

void llama_memory_seq_add(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
             llama_pos delta) {
    if (!mem) {
        return;
    }

    mem->seq_add(seq_id, p0, p1, delta);
}

void llama_memory_seq_div(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
                   int d) {
    if (!mem) {
        return;
    }

    mem->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_seq_pos_min(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_min(seq_id);
}

llama_pos llama_memory_seq_pos_max(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_max(seq_id);
}

bool llama_memory_can_shift(llama_memory_t mem) {
    if (!mem) {
        return false;
    }

    return mem->get_can_shift();
}

// llama state API

// deprecated
size_t llama_get_state_size(llama_context * ctx) {
    return llama_state_get_size(ctx);
}

// deprecated
size_t llama_copy_state_data(llama_context * ctx, uint8_t * dst) {
    return llama_state_get_data(ctx, dst, -1);
}

// deprecated
size_t llama_set_state_data(llama_context * ctx, const uint8_t * src) {
    return llama_state_set_data(ctx, src, -1);
}

// deprecated
bool llama_load_session_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    return llama_state_load_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out);
}

// deprecated
bool llama_save_session_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    return llama_state_save_file(ctx, path_session, tokens, n_token_count);
}

// Returns the *actual* size of the state.
// Intended to be used when saving to state to a buffer.
size_t llama_state_get_size(llama_context * ctx) {
    return ctx->state_get_size();
}

size_t llama_state_get_data(llama_context * ctx, uint8_t * dst, size_t size) {
    ctx->synchronize();

    return ctx->state_get_data(dst, size);
}

// Sets the state reading from the specified source address
size_t llama_state_set_data(llama_context * ctx, const uint8_t * src, size_t size) {
    ctx->synchronize();

    return ctx->state_set_data(src, size);
}

bool llama_state_load_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_load_file(path_session, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading session file: %s\n", __func__, err.what());
        return false;
    }
}

bool llama_state_save_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_save_file(path_session, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving session file: %s\n", __func__, err.what());
        return false;
    }
}

size_t llama_state_seq_get_size(llama_context * ctx, llama_seq_id seq_id) {
    return llama_state_seq_get_size_ext(ctx, seq_id, 0);
}

size_t llama_state_seq_get_data(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_get_data_ext(ctx, dst, size, seq_id, 0);
}

size_t llama_state_seq_set_data(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_set_data_ext(ctx, src, size, seq_id, 0);
}

size_t llama_state_seq_get_size_ext(llama_context * ctx, llama_seq_id seq_id, llama_state_seq_flags flags) {
    return ctx->state_seq_get_size(seq_id, flags);
}

size_t llama_state_seq_get_data_ext(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_get_data(seq_id, dst, size, flags);
}
size_t llama_state_seq_set_data_ext(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_set_data(seq_id, src, size, flags);
}

size_t llama_state_seq_save_file(llama_context * ctx, const char * filepath, llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_seq_save_file(seq_id, filepath, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_state_seq_load_file(llama_context * ctx, const char * filepath, llama_seq_id dest_seq_id, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_seq_load_file(dest_seq_id, filepath, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

// compat: llama_batch -> llama_batch_ext -> encode/decode

int llama_context::encode(const llama_batch & batch_inp) {
    llama_batch_compat compat(this, batch_inp, model.hparams.n_embd_inp_enc());
    return encode(*compat.batch_ext);
}

int llama_context::decode(const llama_batch & batch_inp) {
    llama_batch_compat compat(this, batch_inp);
    return decode(*compat.batch_ext);
}

///

int32_t llama_encode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->encode(batch);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: failed to encode, ret = %d\n", __func__, ret);
    }

    return ret;
}

int32_t llama_decode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->decode(batch);
    if (ret != 0 && ret != 1) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
    }

    return ret;
}

//
// perf
//

llama_perf_context_data llama_perf_context(const llama_context * ctx) {
    llama_perf_context_data data = {};

    if (ctx == nullptr) {
        return data;
    }

    data = ctx->perf_get_data();

    return data;
}

void llama_perf_context_print(const llama_context * ctx) {
    const auto data = llama_perf_context(ctx);

    const double t_end_ms = 1e-3 * ggml_time_us();

    LLAMA_LOG_INFO("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
    LLAMA_LOG_INFO("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
    LLAMA_LOG_INFO("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
    LLAMA_LOG_INFO("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
    LLAMA_LOG_INFO("%s:    graphs reused = %10d\n", __func__, data.n_reused);
    LLAMA_LOG_INFO("%s:   graph reallocs = %10d    / %5d        (same size) / (changed size)\n", __func__, data.n_reallocs_same, data.n_reallocs_diff);
}

void llama_perf_context_reset(llama_context * ctx) {
    ctx->perf_reset();
}

//
// training
//

bool llama_opt_param_filter_all(const struct ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(tensor);
    GGML_UNUSED(userdata);
    return true;
}

void llama_opt_init(struct llama_context * ctx, struct llama_model * model, struct llama_opt_params lopt_params) {
    ctx->opt_init(model, lopt_params);
}

void llama_opt_epoch(
        struct llama_context    * ctx,
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    ctx->opt_epoch(
        dataset,
        result_train,
        result_eval,
        idata_split,
        callback_train,
        callback_eval);
}

int32_t llama_process(llama_context * ctx, llama_process_type type, llama_batch_ext * batch) {
    switch (type) {
        case LLAMA_PROCESS_TYPE_ENCODE: return ctx->encode(*batch);
        case LLAMA_PROCESS_TYPE_DECODE: return ctx->decode(*batch);
    }
    return -1;
}

//
// ext
//

llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx) {
    return ctx->memory_breakdown();
}

llama_context * llama_get_ctx_other(struct llama_context * ctx) {
    return ctx->get_cparams().ctx_other;
}
