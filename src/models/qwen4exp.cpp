#include "models.h"
#include "llama-impl.h"
#include "llama-memory-hybrid-idx.h"
#include "llama-memory-recurrent.h"

#include <algorithm>
#include <cinttypes>
#include <cstdlib>
#if defined(__linux__) || defined(__APPLE__)
#include <sys/mman.h>
#endif

#if LLAMA_LAZY_READER_POSIX
#include <unistd.h> // dup
#endif

// qsa3 (the packed-block WMMA QSA prefill path) is gated at COMPILE TIME, not by an environment
// variable, on purpose.  The rocprofiler-register shipped with ROCm 7.14 calls setenv() during
// early init (publishing GLOG_* vars), which can reallocate the process environment under a
// concurrent getenv() in the target application and intermittently make an env gate read as unset
// (ROCm issue #10196; fixed upstream 2026-09-15 in rocm-systems PR #11620).  An env gate therefore
// makes qsa3 profiling non-reproducible.  Build with -DLLAMA_QSA3_ENABLE=0 to compile it out.
#ifndef LLAMA_QSA3_ENABLE
#define LLAMA_QSA3_ENABLE 1
#endif

// Upper bound of the decode/verify band served by the fused hyper-connection ops
// (ggml_cuda_op_hc_mix / _hc_combine, which assert the same bound - HC_FUSED_MAX_TOKENS
// in ggml-cuda/hc-mix.cu). A verify batch is --spec-draft-n-max + 1 tokens and the
// greedy-purity guarantee is defined over that band (GREEDY-PURITY.md), so routing the
// whole band through the fused path is what keeps decode and verify bit-identical; the
// prefill chunks keep the unfused chain (mmq-era numerics, untouched).
static constexpr int64_t HC_FUSED_MAX_TOKENS = 8;

// The same band for the QSA decode arm policy: the "dense decode below the arch crossover"
// arm must not be gated on n_tokens == 1, or a W=1 decode and a W=(--spec-draft-n-max + 1)
// verify of the same state take different attention regimes (dense vs sparse top-k
// selection) and the greedy stream diverges - measured 2026-09-11 with an arm trace: at
// n_kv = 2304 > width = indexer_top_k + r - 1 = 2051 the plain run took the dense arm and
// the verify the sparse one from the first decode graph on.  Prefill (n_tokens >> the band)
// keeps the sparse selection (the arch policy: "prefill is untouched, QSA always").
// The effective band is extended to the widest speculative verify batch (cparams.n_rs_batch,
// i.e. common_speculative_n_max() + 1) in build_layer_attn: this constant is only the
// no-speculation floor.  With --spec-draft-n-max clamped at 7 the verify is <= 8 tokens and
// the floor covers it, but a deeper draft (n_max > 7, LLAMA_SPEC_DRAFT_N_MAX_CLAMP=0) makes
// the verify W = n_max + 1 > 8 and the exact same dense-vs-sparse flip returns - measured
// 2026-09-13 on qwen4exp-moe at n_kv = 2501 (W = 1..8 `5009c55bca5e01ca`, W = 9..16
// `596ec8bf7461da1a`, `LLAMA_QSA_OFF=1` pure at the dense hash).
static constexpr int64_t QSA_DECODE_BAND = 8;

// LLAMA_QSA_DENSE_* values: a token count, optionally K/M/G suffixed.  -1 when unset or invalid.
static int64_t qsa_env_tokens(const char * name) {
    const char * env = getenv(name);
    if (env != nullptr && env[0] != '\0') {
        char * end = nullptr;
        const long long v = std::strtoll(env, &end, 10);
        if (end != env && v >= 0) {
            int64_t mult = 1;
            switch (*end) {
                case 'K': case 'k': mult = 1024;            break;
                case 'M': case 'm': mult = 1024*1024;       break;
                case 'G': case 'g': mult = 1024*1024*1024;  break;
                default: break;
            }
            return (int64_t) v * mult;
        }
    }
    return -1;
}

// The gfx id of the first non-CPU, non-Meta back-end device's description (the HIP/CUDA back-ends
// carry "(gfxNNNN)" for AMD parts), or -1 if none was found.  The QSA arm policy is per-arch.
static int qsa_arch_gfx() {
    int gfx = -1;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const enum ggml_backend_dev_type ty = ggml_backend_dev_type(dev);
        if (ty == GGML_BACKEND_DEVICE_TYPE_CPU || ty == GGML_BACKEND_DEVICE_TYPE_META) {
            continue;   // GPU / IGPU / ACCEL all qualify (IGPU = integrated parts)
        }
        const char * desc = ggml_backend_dev_description(dev);
        if (desc != nullptr) {
            const char * p = strstr(desc, "gfx");
            if (p != nullptr) {
                gfx = (int) strtol(p + 3, nullptr, 16);
            }
        }
        break;   // homogeneous GPU set in practice
    }
    return gfx;
}

// The qwen4exp MTP draft can attend sparsely (QSA) like the trunk.  The memory type
// (create_memory) and the graph routing (graph_mtp) must agree, so both read this one switch.
//
// **ON by default**: the MTP draft uses the trunk's hybrid-idx memory and attends its prefill
// sparsely (QSA) above the depth gate (LLAMA_MTP_SPARSE_MIN_KV, default 32768).  Measured on
// gfx1151 (qwen4exp IQ4_NL, f16 KV, MTP n3, -b/-ub 2048 unless noted): pp150K 937.1 -> 1007.9 t/s
// (+7.6 %), pp16K -0.3 %, 8K decode parity (56.5 -> 56.4 t/s), 40K text byte-identical with
// -b/-ub 4096.  The old default-OFF reason -- the HC16 F32-elision making the sparse draft's
// depth text diverge -- was fixed by the per-context HC16 state + whole-graph consumer scan
// (patches/0023), so the feature is pure with HC16 on.  LLAMA_MTP_SPARSE=0 disables it (plain KV
// cache + dense draft attention); the decode/verify arm stays opt-in (LLAMA_MTP_SPARSE_DECODE=1).
static bool qwen4exp_mtp_sparse_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("LLAMA_MTP_SPARSE");
        return env == nullptr || std::atoi(env) != 0;
    }();
    return enabled;
}

// LLAMA_QSA_OFF=1 forces the dense no-indexer regime everywhere (a plain-dense reference: no
// indexer store, scoring or sparse selection at any layer).  graph_mtp consults it so the draft
// matches the trunk's arm.
static bool qwen4exp_qsa_off() {
    static const bool off = [] {
        const char * env = getenv("LLAMA_QSA_OFF");
        return env != nullptr && std::atoi(env) != 0;
    }();
    return off;
}

static bool qwen4exp_want_derived_vis(const llama_model & model, const llama_hparams & hparams, int il, const llama_cparams & cparams, int64_t n_tokens);

// Sparse-draft attention arm.  The draft's dense attention is O(n_q*n_kv) while the sparse cap is
// top_k + r - 1, but the indexer score + top-k cost is fixed per query, so the sparse arm only
// pays once the cache is much deeper than the selection budget.  Measured on gfx1151 (qwen4exp
// IQ4_NL, f16 KV, MTP n3, -b/-ub 2048, this campaign):
//   prefill  5K -0.9 %, 16K -0.2 %, 40K +1.6 %, 150K +6.9 %  -> sparse above ~32K tokens
//   decode   16K -13 %, 40K -15 %, 150K -16 %                -> OFF (opt-in for A/B)
// Both arms still store the indexer keys when they attend dense, so the cache stays complete.
// LLAMA_MTP_SPARSE_MIN_KV overrides the prefill depth; LLAMA_MTP_SPARSE_DECODE=1 opts the decode/
// verify band back in (LLAMA_MTP_SPARSE_DECODE_UNTIL then gates it on depth, 0 = whole band).
static bool qwen4exp_mtp_sparse_arm(int64_t n_tokens, int64_t n_kv,
        const llama_memory_hybrid_idx_context * mctx,
        const llama_cparams & cparams, const llama_hparams & hparams) {
    if (!cparams.flash_attn) {
        return false;   // the sparse FA op is the whole point; without it this is wasted indexer work
    }
    if (n_tokens >= 128) {
        static const int64_t min_kv = []() -> int64_t {
            const int64_t env = qsa_env_tokens("LLAMA_MTP_SPARSE_MIN_KV");
            return env >= 0 ? env : 32768;
        }();
        return n_kv >= min_kv;
    }
    // decode/verify: the reference's <= 8-token guard, off by default because every measured depth
    // lost (the SIMT selected-cell decode plus the per-step indexer costs more than the dense
    // attention it replaces at this geometry).
    static const bool decode = [] {
        const char * env = getenv("LLAMA_MTP_SPARSE_DECODE");
        return env != nullptr && std::atoi(env) != 0;
    }();
    if (!decode) {
        return false;
    }
    if (n_tokens <= QSA_DECODE_BAND && mctx->get_n_stream() == 1 &&
            cparams.offload_kqv && hparams.f_max_alibi_bias == 0.0f && !hparams.attn_soft_cap) {
        const int64_t until = qsa_env_tokens("LLAMA_MTP_SPARSE_DECODE_UNTIL");
        return until < 0 || until == 0 || n_kv >= until;
    }
    return false;
}

// The prefill half of the QSA arm policy is the static lambda next to the decode gate in
// build_layer_attn (default 0 = QSA prefill always, per the documented 2026-09-07 arch policy;
// LLAMA_QSA_DENSE_PREFILL_UNTIL is the opt-in A/B).

// True when the device that will execute layer il accepts the fused sparse QSA op for this K/V
// cache type.  This delegates to the back-end's own predicate (ggml_cuda_flash_attn_qsa_supported
// through ggml_backend_dev_supports_op) instead of mirroring its type list here: a stale mirror is
// not a fallback but a hard abort, because an unsupported qsa op is never split across
// tensor-parallel devices while the attention gate is (meta splitter, 2026-09-11).  Under
// -sm tensor model.dev_layer() is the Meta device, whose supports_op() requires EVERY
// tensor-parallel device to accept the op, so this query *is* the meta-split safety condition -
// which the device-mismatch fused-op probe cannot express, because a QSA node only exists once the
// cache is deeper than the indexer selection width, so a reserve-time probe graph never has one.
static bool qsa_op_supported(const llama_model & model, const llama_hparams & hparams, int il, ggml_type type_kv) {
    ggml_backend_dev_t dev = model.dev_layer(il);
    if (dev == nullptr) {
        return false;
    }

    const ggml_init_params params = {
        /*.mem_size   =*/ 8*ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return false;
    }

    // minimal shapes: the back-end predicate reads the source types and the head size (D) only
    const int64_t D = hparams.n_embd_head_k(il);
    ggml_tensor * q   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D,       1, 1, 1);
    ggml_tensor * k   = ggml_new_tensor_4d(ctx, type_kv,       D,       1, 1, 1);
    ggml_tensor * v   = ggml_new_tensor_4d(ctx, type_kv,       D,       1, 1, 1);
    ggml_tensor * idx = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, 1,       1, 1, 1);
    ggml_tensor * msk = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1,       1, 1, 1);

    // mask form (cell_vis/q_vis are the derived-visibility alternative, not needed for a probe)
    const bool ok = ggml_backend_dev_supports_op(dev,
            ggml_flash_attn_qsa(ctx, q, k, v, idx, msk, 1.0f, 0.0f, nullptr, nullptr));
    ggml_free(ctx);
    return ok;
}

// True when the acting device accepts the *packed* (qsa3) QSA op for this layer's cache type.
// The packed form is only taken when the packed WMMA kernel is available (the arch policy lives
// inside ggml_cuda_flash_attn_qsa3_supported, reached here through ggml_backend_dev_supports_op
// when src[7]/src[8] are set).  This matters for memory: the natural F16 K/V packs the graph
// materialises are two full F16 copies of the cache, and building them on a device whose kernel
// will not read them is pure waste (issue #60's r17+qsa3-off headroom).  Under -sm tensor
// model.dev_layer() is the Meta device, whose supports_op() requires EVERY child to accept, so
// the probe is also the tensor-parallel safety condition.
static bool qsa3_op_supported(const llama_model & model, const llama_hparams & hparams, int il, ggml_type type_kv) {
    ggml_backend_dev_t dev = model.dev_layer(il);
    if (dev == nullptr) {
        return false;
    }

    const ggml_init_params params = {
        /*.mem_size   =*/ 16*ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return false;
    }

    // Minimal shapes that satisfy the packed predicate: D = 256, the layer's GQA ratio, a prefill
    // width of 128 and an FATTN_KQ_STRIDE-aligned n_kv.  The predicate is a pure field check, so
    // these shapes exercise it without touching memory (no_alloc).
    const int64_t D    = hparams.n_embd_head_k(il);
    const int64_t gqa  = hparams.n_head_kv(il) != 0 ? (int64_t) hparams.n_head(il) / hparams.n_head_kv(il) : 0;
    const int64_t n_kv = 4;
    ggml_tensor * q   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D,       128, gqa, 1);
    ggml_tensor * k   = ggml_new_tensor_4d(ctx, type_kv,       D,       n_kv, 1,  1);
    ggml_tensor * v   = ggml_new_tensor_4d(ctx, type_kv,       D,       n_kv, 1,  1);
    ggml_tensor * idx = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, 1,       128, 1,  1);
    ggml_tensor * msk = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv,    128, 1,  1);
    ggml_tensor * pk  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D,       1,   n_kv, 1);
    ggml_tensor * pv  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D,       1,   n_kv, 1);

    ggml_tensor * op = ggml_flash_attn_qsa(ctx, q, k, v, idx, msk, 1.0f, 0.0f, nullptr, nullptr);
    ggml_flash_attn_qsa_set_packed(op, pk, pv);

    const bool ok = ggml_backend_dev_supports_op(dev, op);
    ggml_free(ctx);
    return ok;
}

// True when the acting device accepts the fused 4-head lightning-indexer op (the qwen4exp QSA
// prefill score).  The probe is a pure capability query: the op has a generic fallback on every
// backend, so this is true wherever the backend implements LIGHTNING_INDEXER.  Whether the model
// should *use* the fused op over the unfused mul_mat+relu+head_sum chain is a separate policy
// (issue #59/#60) decided by use_wmma below, because on RDNA3_0 the generic vec fallback is
// slower than the chain at shallow/medium prefill yet is what keeps a deep prefill inside the
// compute-graph reserve.  The probe still guards a truly unsupported backend (and, under
// -sm tensor, the Meta device's all_of() is the split-safety condition).
static bool lightning_indexer_op_supported(const llama_model & model, int il, int64_t n_tps) {
    ggml_backend_dev_t dev = model.dev_layer(il);
    if (dev == nullptr) {
        return false;
    }

    const ggml_init_params params = {
        /*.mem_size   =*/ 8*ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return false;
    }

    const int64_t D    = 128;
    const int64_t H    = 4;
    const int64_t n_kv = 512;
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, H,    n_tps, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, 1,    n_kv,  1);
    ggml_tensor * w = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, H, n_tps, 1,    1);
    ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, n_tps, 1, 1);

    const bool ok = ggml_backend_dev_supports_op(dev, ggml_lightning_indexer(ctx, q, k, w, m));
    ggml_free(ctx);
    return ok;
}

// bad metadata must be catchable: GGML_ASSERT aborts the whole process
static void qwen4exp_require_nonzero(const llama_model_loader & ml, llm_kv kid, uint32_t value) {
    if (value == 0) {
        throw std::runtime_error(format("%s must be greater than zero, got %u", ml.llm_kv(kid).c_str(), value));
    }
}

// get_arr() copies a short array as-is, leaving a zero tail the n-gram hash silently drops
static void qwen4exp_require_arr_len(llama_model_loader & ml, llm_kv kid, uint32_t n_min) {
    uint32_t n_arr = 0;
    ml.get_arr_n(kid, n_arr, true);
    if (n_arr < n_min) {
        throw std::runtime_error(format("%s has %u entries, but at least %u are required",
                                        ml.llm_kv(kid).c_str(), n_arr, n_min));
    }
}

// qwen4exp sets mtp_use_dedicated_embeddings=false, so a draft head ships without its own
// embedding table / LM head only if the converter omitted them; borrow from the target then
static const llama_model & qwen4exp_shared_model(const llama_cparams & cparams, const llama_model & model, const char * name) {
    if (cparams.ctx_other == nullptr) {
        throw std::runtime_error(format("QWEN4EXP MTP: this draft head has no '%s' of its own; "
                                        "load it as a draft of its target model, not on its own", name));
    }
    const llama_model & other = *llama_get_model(cparams.ctx_other);
    if (other.hparams.n_embd != model.hparams.n_embd || other.vocab.n_tokens() != model.vocab.n_tokens()) {
        throw std::runtime_error(format("QWEN4EXP MTP: draft and target disagree on the shape of '%s'", name));
    }
    return other;
}

void llama_model_qwen4exp::load_arch_hparams(llama_model_loader & ml) {
    // NextN/MTP: an extra trunk-shaped block appended past the trunk. Read this first, since
    // n_layer() == n_layer_all - n_layer_nextn feeds every per-layer array below.
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);
    GGML_ASSERT(hparams.n_layer_nextn < hparams.n_layer_all && "n_layer_nextn must be < block_count");

    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all, false);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);

    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,    hparams.rope_sections, 4, true);

    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    // HC; low_rank is qwen4exp-specific, DeepSeek-V4 leaves it absent (full rank)
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,    hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_LOW_RANK, hparams.hc_low_rank);
    // a count of 1 has nothing to mix: transformers configuration_qwen4_exp.py:196, vLLM
    // config.py:49 and SGLang configs/qwen4_exp.py:38 all raise on hc_count <= 1
    if (hparams.dsv4_hc_mult <= 1) {
        throw std::runtime_error(format("%s must be greater than one, got %u",
                                        ml.llm_kv(LLM_KV_HYPER_CONNECTION_COUNT).c_str(), hparams.dsv4_hc_mult));
    }
    qwen4exp_require_nonzero(ml, LLM_KV_HYPER_CONNECTION_LOW_RANK, hparams.hc_low_rank);
    hparams.n_embd_out_impl = hparams.dsv4_hc_mult * hparams.n_embd;

    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);
    ml.get_key_or_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS, hparams.dsv4_compress_ratios, hparams.n_layer_all, false);

    {   // The converted GGUF leaves the nextn/MTP layer's compress ratio at 0, but the MTP sidecar
        // ships blk.N.indexer.* and runs that layer with the same sparse attention as the trunk, so
        // inherit the trunk's ratio.  With the sparse draft off (the default) the layer stays dense
        // (no ratio), matching the pre-change behaviour.
        if (qwen4exp_mtp_sparse_enabled() && hparams.n_layer_nextn > 0 && hparams.indexer_head_size > 0) {
            int32_t trunk_r = 0;
            for (uint32_t j = 0; j < hparams.n_layer(); ++j) {
                if (hparams.dsv4_compress_ratios[j] > 0) { trunk_r = hparams.dsv4_compress_ratios[j]; }
            }
            for (uint32_t j = hparams.n_layer(); j < hparams.n_layer_all && trunk_r > 0; ++j) {
                if (hparams.dsv4_compress_ratios[j] == 0) {
                    hparams.dsv4_compress_ratios[j] = trunk_r;
                    LLAMA_LOG_INFO("%s: nextn layer %u compress ratio 0 -> %d\n", __func__, j, trunk_r);
                }
            }
        }
    }

    // PLE n-gram hash embeddings; if the key group is absent every field stays zero
    hparams.is_ple_impl.reset();
    hparams.ple_n_heads = 0;

    uint32_t n_ple = 0;
    ml.get_arr_n(LLM_KV_PLE_LAYERS, n_ple, false);
    if (n_ple > 0) {
        std::vector<uint32_t> ple_layers;
        ml.get_arr(LLM_KV_PLE_LAYERS, ple_layers);
        if (n_ple != 1) {
            // hparams holds one set of hash constants, so several PLE modules cannot be represented
            throw std::runtime_error(format("%s lists %u layers, but only one PLE layer is supported",
                                            ml.llm_kv(LLM_KV_PLE_LAYERS).c_str(), n_ple));
        }
        for (uint32_t il : ple_layers) {
            if (il >= hparams.n_layer_all) {
                throw std::runtime_error(format("PLE layer %u is out of range", il));
            }
            hparams.is_ple_impl.set(il);
        }

        ml.get_key(LLM_KV_PLE_NGRAM_SIZE,      hparams.ple_ngram_size);
        ml.get_key(LLM_KV_PLE_HEADS_PER_NGRAM, hparams.ple_heads_per_ngram);
        ml.get_key(LLM_KV_PLE_CONV_KERNEL,     hparams.ple_conv_kernel);
        ml.get_key(LLM_KV_PLE_EOS_TOKEN_ID,    hparams.ple_eos_token_id);
        // optional: files written before this key fall back to the EOS token
        ml.get_key(LLM_KV_PLE_IMAGE_TOKEN_ID,  hparams.ple_image_token_id, false);
        ml.get_key(LLM_KV_EMBEDDING_LENGTH_PER_LAYER, hparams.n_embd_per_layer);
        qwen4exp_require_nonzero(ml, LLM_KV_PLE_CONV_KERNEL,             hparams.ple_conv_kernel);
        qwen4exp_require_nonzero(ml, LLM_KV_EMBEDDING_LENGTH_PER_LAYER,  hparams.n_embd_per_layer);

        hparams.ple_n_heads  = (hparams.ple_ngram_size - 1) * hparams.ple_heads_per_ngram;
        hparams.ple_head_dim = hparams.n_embd_per_layer;
        if (hparams.ple_ngram_size < 2 || hparams.ple_ngram_size > LLAMA_MAX_PLE_NGRAM) {
            throw std::runtime_error(format("PLE n-gram size %u is out of range", hparams.ple_ngram_size));
        }
        if (hparams.ple_n_heads == 0 || hparams.ple_n_heads > LLAMA_MAX_PLE_HEADS) {
            throw std::runtime_error(format("PLE head count %u is out of range", hparams.ple_n_heads));
        }

        qwen4exp_require_arr_len(ml, LLM_KV_PLE_LAYER_MULTIPLIERS, hparams.ple_ngram_size);
        qwen4exp_require_arr_len(ml, LLM_KV_PLE_HEAD_OFFSETS,      hparams.ple_n_heads);
        qwen4exp_require_arr_len(ml, LLM_KV_PLE_HEAD_VOCAB_SIZES,  hparams.ple_n_heads);

        ml.get_arr(LLM_KV_PLE_LAYER_MULTIPLIERS, hparams.ple_layer_multipliers);

        // the file stores the head ranges as uint64, so read at that width and narrow to the int32 the gather uses
        std::array<uint64_t, LLAMA_MAX_PLE_HEADS> head_offsets     = {};
        std::array<uint64_t, LLAMA_MAX_PLE_HEADS> head_vocab_sizes = {};
        ml.get_arr(LLM_KV_PLE_HEAD_OFFSETS,     head_offsets);
        ml.get_arr(LLM_KV_PLE_HEAD_VOCAB_SIZES, head_vocab_sizes);
        for (uint32_t h = 0; h < hparams.ple_n_heads; ++h) {
            if (head_vocab_sizes[h] == 0 ||
                head_offsets[h]     > INT32_MAX ||
                head_vocab_sizes[h] > INT32_MAX ||
                head_offsets[h] + head_vocab_sizes[h] > INT32_MAX) {
                throw std::runtime_error(format("PLE head %u range does not fit the int32 row index", h));
            }
            hparams.ple_head_offsets[h]     = (uint32_t) head_offsets[h];
            hparams.ple_head_vocab_sizes[h] = (uint32_t) head_vocab_sizes[h];
        }
    }

    // linear attention everywhere except every full_attention_interval-th layer
    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 4;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        qwen4exp_require_nonzero(ml, LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval);
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
        }
    }

    // the PLE conv history is a row of the recurrent cache, which linear layers alone have
    for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
        if (hparams.is_ple(i) && !hparams.is_recr(i)) {
            throw std::runtime_error(format("PLE layer %u is not a linear attention layer", i));
        }
    }

    switch (hparams.n_layer()) {
        case 48: type = LLM_TYPE_A3B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen4exp::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    const int64_t hc_lr  = hparams.hc_low_rank;

    const bool mtp_only    = (hparams.n_layer_nextn > 0) && (ml.get_weight("blk.0.hc_attn_norm.weight") == nullptr);
    const int  trunk_flags = mtp_only ? TENSOR_NOT_REQUIRED : 0;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, trunk_flags);

    // there is no output_norm: the final hyper-connection mixer carries it
    // the gammas load as [n_embd, hc] so the grouped norm multiplies them without a graph reshape
    hc_head_norm = create_tensor(tn(LLM_TENSOR_HC_HEAD_NORM, "weight"), { n_embd, hc }, TENSOR_ALLOW_RESHAPE | trunk_flags);
    hc_head_down = create_tensor(tn(LLM_TENSOR_HC_HEAD_DOWN, "weight"), { hc_dim, hc_lr }, trunk_flags);
    hc_head_up   = create_tensor(tn(LLM_TENSOR_HC_HEAD_UP,   "weight"), { hc_lr, hc_dim }, trunk_flags);

    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
    // tie_word_embeddings is false here: never tie to a token_embd a borrowing draft lacks
    if (output == NULL && tok_embd != NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    // flat [ple_head_dim, n_rows] gather target
    if (hparams.ple_n_heads > 0) {
        // the head ranges are what the gather indexes, so they set the minimum row count
        int64_t ple_rows = 0;
        for (uint32_t h = 0; h < hparams.ple_n_heads; ++h) {
            ple_rows = std::max(ple_rows, (int64_t) hparams.ple_head_offsets[h] + hparams.ple_head_vocab_sizes[h]);
        }

        // the converter pads the table; a model synthesised from metadata has no tensor to ask
        const std::string ple_name = tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight").str();
        const llama_model_loader::llama_tensor_weight * ple_w = ml.get_weight(ple_name.c_str());
        if (ple_w != nullptr) {
            if (ple_w->tensor->ne[1] < ple_rows) {
                throw std::runtime_error(format("%s has %" PRId64 " rows, too few for the PLE head ranges (%" PRId64 ")",
                                                ple_name.c_str(), ple_w->tensor->ne[1], ple_rows));
            }
            ple_rows = ple_w->tensor->ne[1];
        }

#if LLAMA_LAZY_READER_POSIX
        if (ml.lazy.managed_budget > 0 && ple_w != nullptr) {
            // managed path: cache the rows on demand in a fixed-size host buffer.
            // the tensor itself is never materialized; the reader reads straight
            // from the file with pread()
            const int64_t row_nelems = ple_w->tensor->ne[0];
            const size_t  row_bytes  = (size_t) row_nelems * ggml_type_size(ple_w->tensor->type)
                                                    / ggml_blck_size(ple_w->tensor->type);
            GGML_ASSERT(row_nelems % ggml_blck_size(ple_w->tensor->type) == 0);
            lazy_reader = std::make_shared<llama_lazy_reader>(llama_lazy_reader::config{
                /*fd*/         dup(ml.files[ple_w->idx]->file_id()),
                /*data_offs*/  ple_w->offs,
                /*row_bytes*/  row_bytes,
                /*n_rows*/     (uint64_t) ple_rows,
                /*type*/       ple_w->tensor->type,
                /*row_nelems*/ row_nelems,
                /*budget*/     ml.lazy.managed_budget,
            });
            // keep the ggml tensor for metadata/counts only; its data is never loaded.
            // the range must also stay out of the load-time WILLNEED prefetch (the
            // kernel would fault the whole table in), so record it as a lazy range
            ml.lazy.add_range(ple_name, *ple_w);
            per_layer_tok_embd = create_tensor(tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight"),
                                               { hparams.ple_head_dim, ple_rows }, TENSOR_SKIP_MANAGED);
        } else
#endif // LLAMA_LAZY_READER_POSIX
        {
            per_layer_tok_embd = create_tensor(tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight"),
                                               { hparams.ple_head_dim, ple_rows },
                                               ml.files.empty() ? 0 : TENSOR_READ_LAZY);
        }
    }

    // MTP tensors sit in the trailing block(s); skip them entirely unless a draft head was asked for
    const int mtp_flags = !ml.load_mtp ? TENSOR_SKIP : 0;

    for (int il = 0; il < (int) hparams.n_layer_all; ++il) {
        auto & layer = layers[il];

        // the MTP block is structurally a trunk block: is_recr()/is_ple() are both false past
        // the trunk, so it takes the full-attention + MoE path below with no special casing
        const int flags = il < n_layer ? trunk_flags : mtp_flags;

        const int64_t n_ff_exp   = hparams.n_ff_exp() ? hparams.n_ff_exp() : n_ff / n_expert_used;
        const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;

        const int64_t head_k_dim = hparams.ssm_d_state;
        const int64_t head_v_dim = hparams.ssm_d_state;
        const int64_t n_k_heads  = hparams.ssm_n_group;
        const int64_t n_v_heads  = hparams.ssm_dt_rank;
        const int64_t key_dim    = head_k_dim * n_k_heads;
        const int64_t value_dim  = head_v_dim * n_v_heads;
        const int64_t conv_dim   = key_dim * 2 + value_dim;

        // two HC modules per layer: before the token mixer, before the MoE
        layer.hc_attn_norm   = create_tensor(tn(LLM_TENSOR_HC_ATTN_NORM,   "weight", il), { n_embd, hc }, TENSOR_ALLOW_RESHAPE | flags);
        layer.hc_attn_down   = create_tensor(tn(LLM_TENSOR_HC_ATTN_DOWN,   "weight", il), { hc_dim, hc_lr }, flags);
        layer.hc_attn_up     = create_tensor(tn(LLM_TENSOR_HC_ATTN_UP,     "weight", il), { hc_lr, hc_dim }, flags);
        layer.hc_attn_inject = create_tensor(tn(LLM_TENSOR_HC_ATTN_INJECT, "weight", il), { hc_dim, hc }, flags);
        layer.hc_ffn_norm    = create_tensor(tn(LLM_TENSOR_HC_FFN_NORM,    "weight", il), { n_embd, hc }, TENSOR_ALLOW_RESHAPE | flags);
        layer.hc_ffn_down    = create_tensor(tn(LLM_TENSOR_HC_FFN_DOWN,    "weight", il), { hc_dim, hc_lr }, flags);
        layer.hc_ffn_up      = create_tensor(tn(LLM_TENSOR_HC_FFN_UP,      "weight", il), { hc_lr, hc_dim }, flags);
        layer.hc_ffn_inject  = create_tensor(tn(LLM_TENSOR_HC_FFN_INJECT,  "weight", il), { hc_dim, hc }, flags);

        if (!hparams.is_recr(il)) {
            // full attention: wq holds [q|gate] interleaved per head
            create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, flags);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k * n_head, n_embd }, flags);

            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, flags);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, flags);

            const int64_t idx_dim = hparams.indexer_head_size;
            layer.index_q_proj = create_tensor(tn(LLM_TENSOR_INDEXER_Q_PROJ, "weight", il), { n_embd, hparams.indexer_n_head * idx_dim }, flags);
            layer.index_k_proj = create_tensor(tn(LLM_TENSOR_INDEXER_K_PROJ, "weight", il), { n_embd, idx_dim }, flags);
            layer.index_q_norm = create_tensor(tn(LLM_TENSOR_INDEXER_Q_NORM, "weight", il), { idx_dim }, flags);
            layer.index_k_norm = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "weight", il), { idx_dim }, flags);
        } else {
            layer.wqkv       = create_tensor(tn(LLM_TENSOR_ATTN_QKV,   "weight", il), { n_embd, key_dim * 2 + value_dim }, flags);
            layer.wqkv_gate  = create_tensor(tn(LLM_TENSOR_ATTN_GATE,  "weight", il), { n_embd, value_dim }, flags);
            layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", il), { hparams.ssm_d_conv, conv_dim }, flags);
            layer.ssm_dt     = create_tensor(tn(LLM_TENSOR_SSM_DT,     "bias",   il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_a      = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,         il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_beta   = create_tensor(tn(LLM_TENSOR_SSM_BETA,   "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_alpha  = create_tensor(tn(LLM_TENSOR_SSM_ALPHA,  "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_norm   = create_tensor(tn(LLM_TENSOR_SSM_NORM,   "weight", il), { head_v_dim }, flags);
            layer.ssm_out    = create_tensor(tn(LLM_TENSOR_SSM_OUT,    "weight", il), { value_dim, n_embd }, flags);
        }

        if (hparams.is_ple(il)) {
            layer.ple_key        = create_tensor(tn(LLM_TENSOR_PLE_KEY,        "weight", il), { n_embd, hc_dim }, flags);
            layer.ple_value      = create_tensor(tn(LLM_TENSOR_PLE_VALUE,      "weight", il), { n_embd, n_embd }, flags);
            layer.ple_norm_key   = create_tensor(tn(LLM_TENSOR_PLE_NORM_KEY,   "weight", il), { n_embd, hc }, TENSOR_ALLOW_RESHAPE | flags);
            layer.ple_norm_query = create_tensor(tn(LLM_TENSOR_PLE_NORM_QUERY, "weight", il), { n_embd, hc }, TENSOR_ALLOW_RESHAPE | flags);
            layer.ple_norm_conv  = create_tensor(tn(LLM_TENSOR_PLE_NORM_CONV,  "weight", il), { n_embd, hc }, TENSOR_ALLOW_RESHAPE | flags);
            layer.ple_conv1d     = create_tensor(tn(LLM_TENSOR_PLE_CONV1D,     "weight", il), { hparams.ple_conv_kernel, hc_dim }, flags);
        }

        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", il), { n_embd, n_expert }, flags);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, flags);
        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, flags);

        layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", il), { n_embd }, flags);
        layer.ffn_gate_shexp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP,     "weight", il), { n_embd, n_ff_shexp }, flags);
        layer.ffn_up_shexp       = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,       "weight", il), { n_embd, n_ff_shexp }, flags);
        layer.ffn_down_shexp     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP,     "weight", il), { n_ff_shexp, n_embd }, flags);

        if (il < n_layer) {
            continue;
        }

        // NextN/MTP head. enorm/hnorm gate the two inputs; eh_proj is the checkpoint's
        // fc_embedding and fc_hidden fused side by side, so one matmul over
        // concat(e, h) computes fc_embedding@e + fc_hidden@h.
        layer.nextn.enorm   = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,   "weight", il), { n_embd }, flags);
        layer.nextn.hnorm   = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,   "weight", il), { hc_dim }, flags);
        layer.nextn.eh_proj = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", il), { 2 * n_embd, n_embd }, flags);

        // the head's own output mixer, mirroring the trunk's hc_head_*: it collapses the
        // hc streams and stands in for the output norm, of which qwen4exp has none
        layer.nextn.hc_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_NORM, "weight", il), { n_embd, hc }, TENSOR_ALLOW_RESHAPE | flags);
        layer.nextn.hc_head_down = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_DOWN, "weight", il), { hc_dim, hc_lr }, flags);
        layer.nextn.hc_head_up   = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_UP,   "weight", il), { hc_lr, hc_dim }, flags);

        // qwen4exp sets mtp_use_dedicated_embeddings=false, so these are absent and the
        // head falls back to the trunk's embedding table and LM head
        layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", il), { n_embd, n_vocab }, flags | TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", il), { n_embd, n_vocab }, flags | TENSOR_NOT_REQUIRED);
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen4exp::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    // without this a head-only draft loads, then walks the null trunk and segfaults
    if (hc_head_norm == nullptr) {
        throw std::runtime_error("this model is an MTP draft head without a trunk; "
                                 "load it as a draft of its target model, not on its own");
    }
    return std::make_unique<graph>(*this, params);
}

// Hyper-connections keep hc parallel residual streams [n_embd, hc, T] in place of layer norms.
// Returns the mixed [n_embd, T] stream; `inject` gets the [hc, T] scatter weights.
ggml_tensor * llama_model_qwen4exp::graph::build_hc_mix(
        ggml_tensor *  x,
        ggml_tensor *  w_norm,
        ggml_tensor *  w_down,
        ggml_tensor *  w_up,
        ggml_tensor *  w_inject,
        ggml_tensor ** inject,
        int            il) {
    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    const int64_t nt     = x->ne[2];

    // decode/verify band (nt <= HC_FUSED_MAX_TOKENS): fuse the whole mixer - grouped
    // RMSNorm, gamma scale, both LoRA products with the silu/sigmoid gating, the
    // stream collapse AND the inject product - into one dispatch; the model views the
    // mixed and inject outputs out of the one result. Serving the whole band (not just
    // nt == 1) is required for greedy purity: a verify batch must run the same
    // per-token arithmetic as a single-token decode. The prefill path keeps the
    // unfused chain so its numerics (mmq vs mmvq accumulation) are untouched.
    // The fused op is Q8_0-specific, so other weight types (synthetic/test
    // models) fall back to the generic chain. Inject may be F32 (the mmvf
    // tail of the collapse launch) or Q8_0 (mmvq rows on the down grid).
    // BF16 hc weights (Flash-Next GSQ-RCO): the BF16 variant of the op replays the BF16 chain bit for bit
    // (LLAMA_HC_MIX_BF16=0 keeps the unfused chain)
    static const bool hc_mix_bf16 = !getenv("LLAMA_HC_MIX_BF16") || atoi(getenv("LLAMA_HC_MIX_BF16")) != 0;
    // only the GPU backend implements the BF16 variant: hc weights in a host buffer (CPU-only, -ot) keep the chain
    const bool fused_bf16 = hc_mix_bf16 && hc == 4 && w_down->ne[1] == 320 && w_down->type == GGML_TYPE_BF16 && w_up->type == GGML_TYPE_BF16 &&
                            (w_inject == nullptr || w_inject->type == GGML_TYPE_BF16) &&
                            w_down->buffer != nullptr && !ggml_backend_buffer_is_host(w_down->buffer);
    const bool fused_ok = fused_bf16 || (w_down->type == GGML_TYPE_Q8_0 && w_up->type == GGML_TYPE_Q8_0 &&
                          (w_inject == nullptr || w_inject->type == GGML_TYPE_F32 || w_inject->type == GGML_TYPE_Q8_0));
    // nt == 0 appears in the reservation-only graphs (the old nt == 1 gate excluded them).
    // The band covers decode AND short prefill chunks: a <= HC_FUSED_MAX_TOKENS-token batch
    // cannot be told apart from a verify batch, and both must use the decode arithmetic.
    if (nt > 0 && nt <= HC_FUSED_MAX_TOKENS && cparams.fused_hc_mix && fused_ok) {
        // w_inject == nullptr only for the head call (il = -1): no tail
        ggml_tensor * dst_t = ggml_hc_mix(ctx0, x, w_norm, w_down, w_up, w_inject,
                hc, hparams.f_norm_rms_eps);
        const int64_t out_n = n_embd + (w_inject ? hc : 0);
        // planar output (BF16 path): mixed [n_embd, nt] then inject [hc, nt], both contiguous, so the verify band
        // needs no ggml_cont of the mixed head; byte-identical at nt == 1.  LLAMA_HC_MIX_PLANAR=0 disables it.
        static const bool planar_env = !getenv("LLAMA_HC_MIX_PLANAR") || atoi(getenv("LLAMA_HC_MIX_PLANAR")) != 0;
        const bool planar = planar_env && fused_bf16 && w_inject != nullptr;
        if (planar) {
            ggml_hc_mix_set_planar(dst_t);
        }
        ggml_tensor * mixed;
        if (planar) {
            mixed = ggml_view_2d(ctx0, dst_t, n_embd, nt, ggml_row_size(GGML_TYPE_F32, n_embd), 0);
        } else if (w_inject) {
            // mixed = the head of dst [n_embd]; inject = the tail [hc]
            mixed = ggml_view_2d(ctx0, dst_t, n_embd, nt,
                    ggml_row_size(GGML_TYPE_F32, out_n), 0);
            // the head view is strided (its row also holds the inject tail), and the
            // batched consumers (build_moe_ffn) reshape their input, so the verify
            // band needs a contiguous copy; nt == 1 does not take that path, so the
            // decode graph keeps the view untouched.  A contiguous copy preserves
            // the values bit for bit, so decode and verify stay identical.
            if (nt > 1) {
                mixed = ggml_cont(ctx0, mixed);
            }
        } else {
            mixed = dst_t;
        }
        cb(mixed, "hc_mixed", il);
        if (inject && w_inject) {
            *inject = planar
                ? ggml_view_2d(ctx0, dst_t, hc, nt, ggml_row_size(GGML_TYPE_F32, hc), ggml_row_size(GGML_TYPE_F32, n_embd*nt))
                : ggml_view_2d(ctx0, dst_t, hc, nt,
                    ggml_row_size(GGML_TYPE_F32, out_n), ggml_row_size(GGML_TYPE_F32, n_embd));
            cb(*inject, "hc_inject", il);
        } else if (inject) {
            *inject = nullptr;
        }
        return mixed;
    }

    // grouped RMSNorm: reduce over one stream, then scale all streams with the [n_embd, hc] gamma
    // the converter folded each gamma to (1 + w)
    ggml_tensor * xn = ggml_mul(ctx0, ggml_rms_norm(ctx0, x, hparams.f_norm_rms_eps), w_norm);
    xn = ggml_reshape_2d(ctx0, xn, hc_dim, nt);
    cb(xn, "hc_norm", il);

    ggml_tensor * lo = build_lora_mm(w_down, xn);
    if (inject) {
        // emit the inject product right after the down product: both are BF16 GEMMs over the same xn, and the
        // CUDA backend converts a shared F32 src1 to BF16 once when the two run back to back (bf16_src1_cache);
        // emitted at the combine, the block's own BF16 GEMMs in between made it convert the whole xn twice
        ggml_build_forward_expand(gf, lo);
        *inject = build_lora_mm(w_inject, xn);
        cb(*inject, "hc_inject", il);
        ggml_build_forward_expand(gf, *inject);
    }
    lo = ggml_silu(ctx0, ggml_scale(ctx0, lo, 1.0f / (float) hc));
    ggml_tensor * gate = build_lora_mm(w_up, lo);
    cb(gate, "hc_gate", il);

    ggml_tensor * mixed = nullptr;
    if (cparams.fused_dsv4_hc_pre && il >= 0) {
        // sigmoid gate and mean over the streams in one op
        mixed = ggml_dsv4_hc_pre_gated(ctx0,
                ggml_reshape_3d(ctx0, xn,   n_embd, hc, nt),
                ggml_reshape_3d(ctx0, gate, n_embd, hc, nt), 1.0f / (float) hc);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_PRE, mixed, il});
    } else {
        ggml_tensor * gated = ggml_mul(ctx0, xn, ggml_sigmoid(ctx0, gate));
        gated = ggml_reshape_3d(ctx0, gated, n_embd, hc, nt);

        // collapse the streams by their mean
        mixed = ggml_view_2d(ctx0, gated, n_embd, nt,
                ggml_row_size(gated->type, n_embd) * hc, 0);
        mixed = ggml_cont(ctx0, mixed);
        for (int64_t c = 1; c < hc; ++c) {
            ggml_tensor * s = ggml_view_2d(ctx0, gated, n_embd, nt,
                    ggml_row_size(gated->type, n_embd) * hc,
                    ggml_row_size(gated->type, n_embd) * c);
            mixed = ggml_add(ctx0, mixed, s);
        }
        mixed = ggml_scale(ctx0, mixed, 1.0f / (float) hc);
    }
    cb(mixed, "hc_mixed", il);

    return mixed;
}

ggml_tensor * llama_model_qwen4exp::graph::build_hc_combine(
        ggml_tensor * residual,
        ggml_tensor * block_out,
        ggml_tensor * inject,
        int           il) {
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = residual->ne[2];

    // decode/verify band (nt <= HC_FUSED_MAX_TOKENS): fuse the whole residual
    // combine into one dispatch, so a verify batch runs the same per-token
    // arithmetic as a single-token decode (see HC_FUSED_MAX_TOKENS above).
    // The prefill path keeps the unfused chain (its mmq-era numerics are
    // irrelevant here - combine is elementwise - but the chain must stay
    // identical for the prefill gates).
    // nt == 0: reservation-only graph (see build_hc_mix)
    if (nt > 0 && nt <= HC_FUSED_MAX_TOKENS && cparams.fused_hc_combine) {
        ggml_tensor * cur = ggml_hc_combine(ctx0, residual, block_out, inject, hc);
        cb(cur, "hc_combine", il);
        return cur;
    }

    // block_out and inject die at this combine; keep their buffers from being handed to the
    // next grouped norm output (the backend fuses combine + norm into one kernel whose blocks
    // all read block_out/inject while writing their own stream of the norm output, so the two
    // must not alias). Ported from the halo-box reference: with the buffers pinned, the fused
    // kernel can read the NARROW block_out base directly (block_out_hc=false) and the standalone
    // hc-wide REPEAT materialization (a ~80MB tensor per combine at pp2048) is absorbed into the
    // fusion window instead of dispatching.
    // Pinning block_out as a graph output keeps every layer's block output allocated until the end
    // of the graph (~20 MiB per layer at -ub 2048, ~1.9 GiB in all), which is what limited -ub at
    // 256K.  The CUDA combine+norm matcher checks the fused outputs against block_out at run time
    // and reads block_out from a copy when one of them reuses its buffer, so the pin is not needed.
    // LLAMA_HC_PIN_BLOCK_OUT=1 restores it for A/B.  inject is tiny and stays pinned.
    static const bool pin_block_out = getenv("LLAMA_HC_PIN_BLOCK_OUT") != nullptr && atoi(getenv("LLAMA_HC_PIN_BLOCK_OUT")) != 0;
    if (pin_block_out) {
        ggml_set_output(block_out);
    }
    ggml_set_output(inject);

    // 2*sigmoid centres the scatter weights on 1, so a zero injection is a plain residual add
    ggml_tensor * w = ggml_sigmoid(ctx0, ggml_scale(ctx0, inject, 1.0f / (float) hc));
    w = ggml_scale(ctx0, w, 2.0f);

    // Emit the block output first and the scatter-weight chain right after it, so that
    // scale -> sigmoid -> scale -> repeat -> mul -> add is one contiguous node run that
    // backends can fuse into a single kernel.
    ggml_build_forward_expand(gf, block_out);
    ggml_build_forward_expand(gf, w);

    ggml_tensor * cur = nullptr;
    if (cparams.fused_dsv4_hc_post && il >= 0) {
        // identity comb: every stream adds the same block output, scaled by its own weight
        cur = ggml_dsv4_hc_post(ctx0, block_out, residual, w, nullptr);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_POST, cur, il});
    } else {
        w = ggml_reshape_3d(ctx0, w, 1, hc, nt);
        // Emit the scatter-weight VIEW before the block_out REPEAT so the fusion window
        // repeat -> mul -> add -> rms -> mul(gamma) is contiguous.  Otherwise the w view lands
        // between the repeat and the mul (mul expands src[0]=repeat before src[1]=w) and
        // ggml_can_fuse_subgraph_ext rejects the window on its external view_src, so the
        // hc_combine_norm matcher never fires.
        ggml_build_forward_expand(gf, w);

        ggml_tensor * b = ggml_reshape_3d(ctx0, block_out, n_embd, 1, nt);
        b = ggml_repeat_4d(ctx0, b, n_embd, hc, nt, 1);

        cur = ggml_add(ctx0, residual, ggml_mul(ctx0, b, w));
    }
    cb(cur, "hc_combine", il);

    return cur;
}

llama_model_qwen4exp::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    const int64_t hc = hparams.dsv4_hc_mult;

    GGML_ASSERT(hparams.n_embd_head_v() == hparams.n_embd_head_k());

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "model.input_embed", -1);
    ggml_build_forward_expand(gf, inpL);

    auto * inp = build_inp_mem_hybrid();

    // qwen4exp always builds llama_memory_hybrid_idx, so this downcast is safe
    // the indexer cache inside it is absent when the GGUF has no indexer tensors
    const auto * mctx_hyb = static_cast<const llama_memory_hybrid_idx_context *>(inp->mctx);

    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();
    if (mctx_idx) {
        GGML_ASSERT(mctx_idx->get_n_kv() == inp->mctx->get_attn()->get_n_kv() &&
                "the indexer cache must track the attention cache cell for cell");
    }

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * ple_emb = nullptr;
    if (hparams.ple_n_heads > 0) {
        ple_emb = build_inp_ple(mctx_hyb);
        // make sure ple_emb and build_inp_embd are in the same graph split
        ggml_build_forward_expand(gf, ple_emb);
    }

    // the wide residual starts as hc identical copies of the embedding
    ggml_tensor * res_hc = ggml_repeat_4d(ctx0,
            ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens),
            n_embd, hc, n_tokens, 1);
    cb(res_hc, "hc_init", -1);
    // make sure hc_init is in the same graph split as the first layer (-sm tensor).  Upstream
    // 10f340d1a; without it the REPEAT stays on the CPU when the PLE gather is a CPU node and the
    // meta splitter views a host-resident reshaped node.
    ggml_build_forward_expand(gf, res_hc);

    // set on the last layer when the full-row residual for the unmasked MTP export is
    // built as a second tail, so the logits tail can stay gathered (see below)
    ggml_tensor * res_hc_export = nullptr;

    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = res_hc;

        if (hparams.is_ple(il)) {
            res_hc = build_ple(inp->get_recr(), ple_emb, res_hc, il);
        }

        ggml_tensor * inject = nullptr;
        ggml_tensor * cur = build_hc_mix(res_hc,
                model.layers[il].hc_attn_norm,
                model.layers[il].hc_attn_down,
                model.layers[il].hc_attn_up,
                model.layers[il].hc_attn_inject,
                &inject, il);

        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recr(il)) {
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            cur = build_layer_attn(inp->get_attn(), mctx_hyb, cur, inp_pos, sections, il);
        }

        // The last layer's tail is always computed on the gathered output rows, so the
        // logits path is bit-identical to `--spec-type none` (a plain decode gathers here
        // too).  An unmasked MTP export additionally needs a hidden row for *every* token,
        // so when this prefill chunk drops rows it gets a second, full-row tail whose
        // result is exported as t_h_nextn.  Running the logits off the full-row tail
        // instead shifts the last-position logits by a ULP (the wide ffn's reduction order
        // depends on the batch width).  A decode/verify batch drops no rows
        // (n_outputs == n_tokens), so nothing is duplicated there.
        const bool mtp_export_defer = il == n_layer - 1 && inp_out_ids &&
                cparams.embeddings_nextn && !cparams.embeddings_nextn_masked && n_outputs < n_tokens;

        ggml_tensor * res_hc_full = nullptr;
        ggml_tensor * cur_full    = nullptr;
        ggml_tensor * inject_full = nullptr;
        if (mtp_export_defer) {
            res_hc_full = res_hc;
            cur_full    = cur;
            inject_full = inject;
        }

        if (il == n_layer - 1 && inp_out_ids) {
            // everything below is per token, so drop the rows that produce no output
            cur    = ggml_get_rows(ctx0, cur,    inp_out_ids);
            inject = ggml_get_rows(ctx0, inject, inp_out_ids);

            res_hc = ggml_reshape_2d(ctx0, res_hc, n_embd*hc, res_hc->ne[2]);
            res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
            res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
        }

        res_hc = build_hc_combine(res_hc, cur, inject, il);

        cur = build_hc_mix(res_hc,
                model.layers[il].hc_ffn_norm,
                model.layers[il].hc_ffn_down,
                model.layers[il].hc_ffn_up,
                model.layers[il].hc_ffn_inject,
                &inject, il);

        cur = build_layer_ffn(cur, il);
        cb(cur, "ffn_out", il);

        res_hc = build_hc_combine(res_hc, cur, inject, il);

        // "l_last" is the layer output name that build_cvec and imatrix look for
        cb(res_hc, "l_last", il);

        if (mtp_export_defer) {
            // second, full-row tail: the unmasked MTP export only.  It does not feed the
            // logits (those come from the gathered tail above).
            ggml_tensor * res_hc_x = build_hc_combine(res_hc_full, cur_full, inject_full, il);
            ggml_tensor * cur_x    = build_hc_mix(res_hc_x,
                    model.layers[il].hc_ffn_norm,
                    model.layers[il].hc_ffn_down,
                    model.layers[il].hc_ffn_up,
                    model.layers[il].hc_ffn_inject,
                    &inject_full, il);
            cur_x = build_layer_ffn(cur_x, il);
            res_hc_export = build_hc_combine(res_hc_x, cur_x, inject_full, il);
        }
    }

    // The MTP head consumes the wide residual before the head mixer collapses it. Export the
    // combine result itself rather than a reshape of it: a pure view gets no backend
    // assignment from the scheduler, and the readback in llama_context looks one up. It is
    // contiguous, so [n_embd, hc, rows] already has the [n_embd_out, rows] layout the reader
    // expects.  A masked export uses the gathered tail (rows already collapsed); an
    // unmasked one uses the separate full-row tail built above.
    if (cparams.embeddings_nextn) {
        ggml_tensor * h_nextn = res_hc_export ? res_hc_export : res_hc;
        cb(h_nextn, "h_nextn", -1);
        // the separate full-row export tail is not on the logits path, so it has to be
        // expanded explicitly (ggml_set_output() alone does not add it to the graph)
        ggml_build_forward_expand(gf, h_nextn);
        res->t_h_nextn = h_nextn;
    }

    // the final mixer is the output norm: there is no separate one
    ggml_tensor * cur = build_hc_mix(res_hc,
            model.hc_head_norm, model.hc_head_down, model.hc_head_up,
            nullptr, nullptr, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// LLM_GRAPH_TYPE_DECODER_MTP draft head for qwen4exp.
//
// The head folds the next token's embedding into the trunk's wide hyper-connection residual,
// runs one trunk-shaped block (dense attention + MoE, wrapped in hyper-connections) over it,
// and collapses the result with its own mixer before reusing the trunk's LM head. The wide
// post-block residual is exported as t_h_nextn so the speculative driver can feed it straight
// back in for the next draft step.
//
// The block attends densely (the GGUF gives it compress_ratio 0; the indexer tensors it
// carries are dead weight), so this mirrors the trunk's dense full-attention branch. The head
// still rides the fused decode ops: build_hc_mix/build_hc_combine collapse to GGML_OP_HC_MIX /
// GGML_OP_HC_COMBINE for single-token decodes, so a draft step is one fused dispatch per HC
// module.
llama_model_qwen4exp::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params) :
    graph(model, params, no_build_t{}) {
    GGML_ASSERT(hparams.n_layer_nextn > 0 && "QWEN4EXP MTP requires n_layer_nextn > 0");
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "QWEN4EXP MTP currently only supports a single MTP block");
    GGML_ASSERT(ubatch.token && "QWEN4EXP MTP requires token input");

    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    GGML_ASSERT(hparams.n_embd_out() == (uint32_t) hc_dim && "QWEN4EXP MTP hidden width mismatch");

    const int il = hparams.n_layer();
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj     && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm       && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm       && "MTP block missing nextn.hnorm");
    GGML_ASSERT(layer.nextn.hc_head_norm && "MTP block missing nextn.hc_head_norm");
    GGML_ASSERT(layer.hc_attn_norm      && "MTP block missing hc_attn_norm");

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    auto inp = std::make_unique<llm_graph_input_embd_h>(hc_dim);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hc_dim, n_tokens);
    ggml_set_input(inp->embd);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hc_dim, n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;
    if (tok_embd_w == nullptr) {
        tok_embd_w = qwen4exp_shared_model(cparams, model, "token_embd.weight").tok_embd;
    }
    ggml_tensor * tok_embd   = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
    cb(tok_embd, "mtp_tok_embd", il);

    ggml_tensor * h_state = ggml_reshape_3d(ctx0, inp->h, n_embd, hc, n_tokens);
    cb(h_state, "mtp_h_state", il);

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // The MTP context is a hybrid-idx memory when the sparse draft is on (create_memory reads the
    // same switch), so the draft head can use the same sparse attention path as the trunk - one QSA
    // call per target chunk, as the reference does.  Otherwise it stays a plain KV cache.
    llm_graph_input_attn_kv * inp_attn = nullptr;
    const llama_memory_hybrid_idx_context * mctx_hyb = nullptr;
    if (qwen4exp_mtp_sparse_enabled() && hparams.indexer_head_size > 0) {
        auto * inp_hyb = build_inp_mem_hybrid();
        const auto * m = static_cast<const llama_memory_hybrid_idx_context *>(inp_hyb->mctx);
        if (m->get_idx() != nullptr) {
            mctx_hyb = m;
            inp_attn = inp_hyb->get_attn();
            // the MTP graph has no recurrent layers, so the hybrid input's recurrent tensors are
            // never used and the allocator skips them - set_input would then hit a null buffer.
            // Give them a trivial use.
            auto * rs = inp_hyb->get_recr();
            for (ggml_tensor * t : { rs->s_copy, rs->s_copy_main, rs->s_copy_tail }) {
                if (t) { ggml_build_forward_expand(gf, ggml_scale(ctx0, ggml_cast(ctx0, t, GGML_TYPE_F32), 0.0f)); }
            }
        }
    }
    if (!inp_attn) { inp_attn = build_attn_inp_kv(); }

    // grouped RMSNorm over the wide stream: normalise each hc stream, then scale the flattened
    // [hc_dim] vector with the head's gamma, exactly as build_hc_mix does
    ggml_tensor * h_norm = ggml_rms_norm(ctx0, h_state, hparams.f_norm_rms_eps);
    h_norm = ggml_reshape_2d(ctx0, h_norm, hc_dim, n_tokens);
    h_norm = ggml_mul(ctx0, h_norm, layer.nextn.hnorm);
    h_norm = ggml_reshape_3d(ctx0, h_norm, n_embd, hc, n_tokens);
    cb(h_norm, "mtp_hnorm", il);

    // the token embedding is shared across the streams, so broadcast it to hc copies
    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    e_norm = ggml_repeat_4d(ctx0,
            ggml_reshape_3d(ctx0, e_norm, n_embd, 1, n_tokens),
            n_embd, hc, n_tokens, 1);
    cb(e_norm, "mtp_enorm", il);

    // eh_proj holds fc_embedding and fc_hidden side by side, so this one matmul is
    // fc_embedding @ e_norm + fc_hidden @ h_norm, applied to each stream independently.
    // Keeping the streams distinct here is the point of the hyper-connection residual:
    // pooling them before the projection would throw that away.
    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0);
    cb(concat, "mtp_concat", il);

    ggml_tensor * res_hc = build_lora_mm(layer.nextn.eh_proj, concat, layer.nextn.eh_proj_s);
    cb(res_hc, "mtp_eh_proj", il);

    ggml_tensor * inject = nullptr;
    ggml_tensor * cur = build_hc_mix(res_hc,
            layer.hc_attn_norm, layer.hc_attn_down, layer.hc_attn_up, layer.hc_attn_inject,
            &inject, il);
    cb(cur, "mtp_hc_attn_pre", il);

    // ---- dense attention, mirroring the trunk's full-attention branch ----
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * Qcur_full = build_lora_mm(layer.wq, cur, layer.wq_s);
    cb(Qcur_full, "mtp_Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "mtp_Qcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "mtp_gate", il);

    ggml_tensor * Kcur = build_lora_mm(layer.wk, cur, layer.wk_s);
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "mtp_Kcur_normed", il);

    ggml_tensor * Vcur = build_lora_mm(layer.wv, cur, layer.wv_s);
    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
    cb(Vcur, "mtp_Vcur", il);

    // IMRoPE, same convention and freq_base as the trunk
    Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(Qcur, "mtp_Qcur", il);
    cb(Kcur, "mtp_Kcur", il);

    const float kq_scale = hparams.f_attention_scale == 0.0f
            ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    if (mctx_hyb) {
        const int64_t r        = hparams.dsv4_compress_ratios[il];
        const int64_t n_kv_idx = mctx_hyb->get_idx()->get_n_kv();
        const int64_t width    = (int64_t) hparams.indexer_top_k + r - 1;
        ggml_tensor * top_k = nullptr;
        if (!qwen4exp_qsa_off() && r > 0 && n_kv_idx > width &&
                qwen4exp_mtp_sparse_arm(n_tokens, n_kv_idx, mctx_hyb, cparams, hparams)) {
            // the mask may be dropped when the sparse FA derives the cell visibility (the same
            // n_ubatch*n_ctx*2 win the trunk takes); build_qsa_top_k expects nullptr exactly there
            const bool want_derived_vis = qwen4exp_want_derived_vis(model, hparams, il, cparams, n_tokens);
            top_k = build_qsa_top_k(mctx_hyb, cur, inp_pos,
                    want_derived_vis ? nullptr : inp_attn->get_kq_mask(), sections, il);
        } else if (!qwen4exp_qsa_off() && r > 0) {
            // keep the indexer cache complete for the first selective ubatch
            build_qsa_store_k(mctx_hyb, cur, il);
        }
        if (top_k) {
            cur = build_attn_qsa(inp_attn, Qcur, Kcur, Vcur, top_k, kq_scale, il);
        } else {
            cur = build_attn(inp_attn, nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
        }
    } else {
        cur = build_attn(inp_attn,
                nullptr, nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    }
    cb(cur, "mtp_attn_pregate", il);

    cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
    cb(cur, "mtp_attn_gated", il);

    cur = build_lora_mm(layer.wo, cur, layer.wo_s);
    cb(cur, "mtp_attn_out", il);

    if (inp_out_ids) {
        cur    = ggml_get_rows(ctx0, cur,    inp_out_ids);
        inject = ggml_get_rows(ctx0, inject, inp_out_ids);

        res_hc = ggml_reshape_2d(ctx0, res_hc, hc_dim, res_hc->ne[2]);
        res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
        res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
    }

    res_hc = build_hc_combine(res_hc, cur, inject, il);
    cb(res_hc, "mtp_hc_attn_post", il);

    // ---- MoE, identical to the trunk's build_layer_ffn ----
    cur = build_hc_mix(res_hc,
            layer.hc_ffn_norm, layer.hc_ffn_down, layer.hc_ffn_up, layer.hc_ffn_inject,
            &inject, il);
    cb(cur, "mtp_hc_ffn_pre", il);

    cur = build_layer_ffn(cur, il);
    cb(cur, "mtp_ffn_out", il);

    res_hc = build_hc_combine(res_hc, cur, inject, il);
    cb(res_hc, "mtp_hc_ffn_post", il);

    // The next draft step re-enters here, so export the wide stream before it is collapsed.
    // As in the trunk, export the combine result rather than a reshape view of it.
    cb(res_hc, "h_nextn", -1);
    res->t_h_nextn = res_hc;

    // the head's own mixer collapses the streams and doubles as the output norm
    cur = build_hc_mix(res_hc,
            layer.nextn.hc_head_norm, layer.nextn.hc_head_down, layer.nextn.hc_head_up,
            nullptr, nullptr, -1);
    cb(cur, "mtp_hc_head", -1);

    // deliberately no res->t_embd: it would be n_embd wide while the context sizes its
    // embedding buffer by n_embd_out (the wide stream). The driver reads t_h_nextn instead.

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    ggml_tensor * head_s = layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : model.output_s;
    if (head_w == nullptr) {
        const llama_model & other = qwen4exp_shared_model(cparams, model, "output.weight");
        head_w = other.output;
        head_s = other.output_s;
        GGML_ASSERT(head_w && "QWEN4EXP MTP: the target model has no LM head to borrow");
    }

    cur = build_lora_mm(head_w, cur, head_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_qwen4exp::graph::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input, model.layers[il].wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llama_model_qwen4exp::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    // the one numerical difference from Qwen3.5's GDN: sigmoid output gate, not silu
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated = ggml_sigmoid(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated);
}

// store-only subset of the QSA layout inputs: the dense shortcut below the selection width still
// caches this ubatch's raw indexer keys (build_input_k_idxs + cpy_k), so scoring can start
// seamlessly once the context passes top_k + ratio - 1
class llama_model_qwen4exp::llm_graph_input_qsa_k : public llm_graph_input_i {
public:
    llm_graph_input_qsa_k(const llama_memory_hybrid_idx_context * mctx) : mctx(mctx) {}
    virtual ~llm_graph_input_qsa_k() = default;

    void set_input(const llama_ubatch * ubatch) override {
        mctx->get_idx()->set_input_k_idxs(k_idxs, ubatch);
    }

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx);

        const auto * idx = mctx->get_idx();
        if (idx == nullptr) {
            return false;
        }

        return k_idxs->ne[0] == params.ubatch.n_tokens;
    }

    ggml_tensor * k_idxs = nullptr;   // I32 [n_tokens]

    const llama_memory_hybrid_idx_context * mctx;
};

void llama_model_qwen4exp::graph::build_qsa_store_k(
        const llama_memory_hybrid_idx_context * mctx_hyb,
        ggml_tensor *                           cur,
        int                                     il) {
    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();

    const int64_t idx_dim = hparams.indexer_head_size;

    if (qsa_k_inp == nullptr) {
        auto inp = std::make_unique<llm_graph_input_qsa_k>(mctx_hyb);
        inp->k_idxs = mctx_idx->build_input_k_idxs(ctx0, ubatch);
        qsa_k_inp = inp.get();
        res->add_input(std::move(inp));
    }

    ggml_tensor * k_raw = build_lora_mm(model.layers[il].index_k_proj, cur);
    k_raw = ggml_reshape_3d(ctx0, k_raw, idx_dim, 1, n_tokens);
    cb(k_raw, "indexer_k_raw", il);

    ggml_build_forward_expand(gf, mctx_idx->cpy_k(ctx0, k_raw, qsa_k_inp->k_idxs, il));
}

// QSA attends to a budget of whole blocks of compress_ratio tokens, plus the incomplete tail
// one mean-pooled indexer key scores each block; set_input resolves the cache layout
//
// The per-block half of the bias is derived in-kernel from blk_idx/blk_tail by default (see
// build_qsa_top_k); GGML_QSA_DERIVED_BIAS=0 restores the uploaded tensor for A/B validation.
static int qwen4exp_derived_bias_mode() {
    static const int mode = [] {
        const char * env = getenv("GGML_QSA_DERIVED_BIAS");
        return env == nullptr ? 1 : std::atoi(env);
    }();
    return mode;
}

// The QSA indexer score chain materialises an [n_blocks * n_idx_h, n_tps, n_stream] F32 tensor,
// which at ctx 204800 / ub 2048 is 1.6 GB per layer - the largest single allocation in the graph.
// build_qsa_top_k keeps the relu before the 4-D reshape and assembles the block dimension in
// chunks by default; GGML_QSA_SCORE_MEM=0 restores the plain chain for A/B.
static bool qwen4exp_score_mem_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_QSA_SCORE_MEM");
        return env == nullptr || std::atoi(env) != 0;
    }();
    return enabled;
}

// The per-cell half of the same bias (the attention mask) is derived from two compact keys as
// well: cell_vis is a cell's compaction key (-1 for an empty or foreign cell) and q_vis is the
// query's key, and a cell is visible iff 0 <= cell_vis <= q_vis.  GGML_QSA_DERIVED_VIS=0 keeps
// the uploaded mask as the additive for A/B.  The mask tensor stays allocated until the flash
// attention derives the same value, so this is a drop-in equivalent on its own.
static bool qwen4exp_derived_vis_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_QSA_DERIVED_VIS");
        return env == nullptr || std::atoi(env) != 0;
    }();
    return enabled;
}

// The prefill derived-visibility gate, expressed WITHOUT the mask tensor, because the derived
// path must not create the mask at all (that is the -800 MiB win: an unreachable mask is left
// unallocated by the gallocr and then aborted on by a buffer query elsewhere).  It is exactly
// the predicate blk_bias checks against the mask's shape - the mask is always
// [n_kv, n_tps, 1, n_stream] for this model - so both paths agree by construction.
// the QSA sparse (fused, top-k only) flash attention is the DEFAULT path; LLAMA_QSA_SPARSE_FA=0
// selects the dense masked path, which still reads the kq mask through build_attn_mha and must
// therefore keep it alive.  Both paths need flash attention enabled.
static bool qwen4exp_qsa_sparse(const llama_model & model, const llama_hparams & hparams, int il, const llama_cparams & cparams) {
    static const bool enabled = []() {
        const char * env = getenv("LLAMA_QSA_SPARSE_FA");
        return env == nullptr || std::atoi(env) != 0;
    }();
    // The fused sparse kernel reads the K/V cache rows itself - F16/BF16 natively, and
    // Q8_0/q4_0/q4_1/q5_0/q5_1/iq4_nl by dequantizing them while a tile is staged.  Whether the
    // acting device accepts the op is asked of the back-end itself (qsa_op_supported, which
    // reaches ggml_cuda_flash_attn_qsa_supported through ggml_backend_dev_supports_op) rather
    // than mirrored here: a stale mirror is not a fallback but an abort, because an unsupported
    // qsa op is not split across the tensor-parallel devices, so its output stays mirrored while
    // the attention gate remains hidden-split and the meta splitter aborts on the attn_gated MUL
    // (observed with qwen4exp + --cache-type-k q4_0/q4_1 + -sm tensor before the query replaced
    // the hand-maintained list).  Under -sm tensor model.dev_layer(il) is the Meta device, whose
    // supports_op requires EVERY tensor-parallel device to accept the op, so the query covers
    // the meta split as well.  Any cache type it rejects takes the dense masked path instead
    // (the same path as LLAMA_QSA_SPARSE_FA=0): that path reads the kq mask, so the
    // derived-visibility/mask-elision arm is off with it (qwen4exp_want_derived_vis below).
    const bool kv_native = qsa_op_supported(model, hparams, il, cparams.type_k);
    return enabled && cparams.flash_attn && kv_native;
}

// the mask may be dropped (never created, hence never allocated) only when nothing in the graph
// reads it: the derived-visibility path plus the sparse FA that derives the cell visibility in
// its kernel.  This is the -800 MiB win; the predicate is exactly the one blk_bias would have
// checked against the mask's own shape (the mask is always [n_kv, n_tps, 1, n_stream] here).
static bool qwen4exp_want_derived_vis(const llama_model & model, const llama_hparams & hparams, int il, const llama_cparams & cparams, int64_t n_tokens) {
    return qwen4exp_derived_bias_mode() != 0 && qwen4exp_derived_vis_enabled() &&
            qwen4exp_qsa_sparse(model, hparams, il, cparams) &&
            n_tokens > 1 && cparams.causal_attn && !hparams.use_alibi;
}

// [QSA_SCORE_BOUNDS] prefill queries are scored in strips of at most this many tokens; each strip is
// then trimmed to the blocks it can actually see (qsa_score_key_limits), so the indexer scorer runs
// on roughly half its columns.  Only the single-sequence prefill case is stripped; anything else
// (decode, verify, multi-stream, M-RoPE images) keeps the whole batch and stays bit-identical.
// LLAMA_QSA_SCORE_STRIP=0 disables the strip + bound for A/B.
static int64_t qwen4exp_query_strip(int64_t n_tokens, int64_t n_stream) {
    static const int64_t env_strip = []() -> int64_t {
        const char * env = getenv("LLAMA_QSA_SCORE_STRIP");
        // 1024 measured best on gfx1151 at -b/-ub 8192 (pp8192 +0.7 %, pp32768 neutral):
        // smaller strips multiply the per-strip matmul/top-k launches, larger ones trim less
        return env == nullptr ? 1024 : atoll(env);
    }();
    if (env_strip <= 0) {
        return n_tokens;
    }
    return n_stream == 1 ? std::min<int64_t>(n_tokens, env_strip) : n_tokens;
}

// The bound is only meaningful where the score columns are complete-block ordinals and the tail is
// carried as cells - the derived-bias path.  n_tokens < 128 keeps the whole decode/verify band on
// the unbounded graph, so W = 1..8 stays byte-identical by construction.
// LLAMA_QSA_SCORE_BOUNDS=0 keeps the strips but scores every strip at full width (isolates the
// strip/launch overhead from the trim savings in an A/B).
static std::vector<int64_t> qwen4exp_score_key_limits(const llama_memory_hybrid_idx_context * mctx,
        const llama_ubatch & ubatch, bool derived_bias, bool derived_vis, int64_t n_blocks,
        int64_t strip, uint32_t ratio, int64_t budget) {
    static const bool enabled = []() {
        const char * env = getenv("LLAMA_QSA_SCORE_STRIP");
        if (env != nullptr && atoll(env) <= 0) {
            return false;
        }
        const char * bounds = getenv("LLAMA_QSA_SCORE_BOUNDS");
        return bounds == nullptr || atoi(bounds) != 0;
    }();
    // the fused top-k skips trimmed blocks by cell index, which needs the derived-visibility
    // cell-clamped kernel path - keep the bound there only
    if (!enabled || !derived_bias || !derived_vis || ubatch.n_tokens < 128 || strip <= 0) {
        return {};
    }
    return mctx->qsa_score_key_limits(ubatch, n_blocks, strip, ratio, budget);
}

class llama_model_qwen4exp::llm_graph_input_qsa : public llm_graph_input_i {
public:
    llm_graph_input_qsa(const llama_memory_hybrid_idx_context * mctx, uint32_t ratio, bool blk_bias, bool derived_bias, bool derived_vis) :
        mctx(mctx), ratio(ratio), blk_bias(blk_bias), derived_bias(derived_bias), derived_vis(derived_vis) {}
    virtual ~llm_graph_input_qsa() = default;

    void set_input(const llama_ubatch * ubatch) override {
        mctx->get_idx()->set_input_k_idxs(k_idxs, ubatch);

        // the QSA mapping tensors (cell_blk/blk_cells/blk_pos + the per-block and per-cell bias
        // forms) are reachable only from the indexer top-k node.  The policies that store K only
        // (decode and short-context graphs) do not build it, so the allocator leaves these
        // unallocated - and there is nothing to fill for a tensor no node reads.
        if (cell_blk == nullptr || cell_blk->buffer == nullptr) {
            return;
        }

        if (rng != nullptr) {
            // derived-cache decode: emit the per-step fill range [from, lim) into the leaf
            int32_t * d = (int32_t *) rng->data;
            mctx->set_input_qsa(cell_blk, blk_cells, blk_pos, bias, blk_idx, blk_tail, cell_vis, q_vis, ubatch, ratio, blk_bias, d, d + 1);
        } else {
            mctx->set_input_qsa(cell_blk, blk_cells, blk_pos, bias, blk_idx, blk_tail, cell_vis, q_vis, ubatch, ratio, blk_bias);
        }
    }

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx);

        const auto * idx = mctx->get_idx();
        if (idx == nullptr) {
            return false;
        }

        const int64_t n_kv     = idx->get_n_kv();
        const int64_t n_stream = mctx->get_n_stream();
        // blocks are keyed by position, which can run ahead of the occupied cells (M-RoPE image + MTP)
        const int64_t n_blocks = ((int64_t) mctx->qsa_n_kv_window() + ratio - 1)/ratio;

        bool res = true;

        res &= params.ubatch.n_tokens % n_stream == 0;

        // a graph built for one form of the per-block bias cannot serve the other
        const int  dbias_mode   = qwen4exp_derived_bias_mode();
        const bool want_derived = dbias_mode != 0 && blk_bias && params.ubatch.n_tokens > 1;
        const bool want_vis     = want_derived && qwen4exp_derived_vis_enabled();
        res &= derived_bias == want_derived;
        res &= derived_vis  == want_vis;

        res &= k_idxs->ne[0]    == params.ubatch.n_tokens;
        res &= cell_blk->ne[0]  == n_kv;
        res &= cell_blk->ne[1]  == n_stream;
        res &= blk_cells->ne[0] == (int64_t) ratio*n_blocks;
        res &= blk_pos->ne[0]   == 4*n_blocks*n_stream;
        if (derived_bias) {
            res &= blk_idx->ne[0]  == n_blocks;
            res &= blk_idx->ne[1]  == n_stream;
            res &= blk_tail->ne[0] == params.ubatch.n_tokens/n_stream;
            res &= blk_tail->ne[1] == n_stream;
        }
        if (derived_vis) {
            res &= cell_vis->ne[0] == n_kv;
            res &= cell_vis->ne[1] == n_stream;
            res &= q_vis->ne[0]    == params.ubatch.n_tokens/n_stream;
            res &= q_vis->ne[1]    == n_stream;
        }
        if (!derived_bias) {
            res &= bias->ne[0] == (blk_bias ? n_blocks : n_kv);
            res &= bias->ne[1] == params.ubatch.n_tokens/n_stream;
        }

        // [QSA_SCORE_BOUNDS] the trimmed scorer widths are baked into the graph, so a reused graph
        // must agree on the strip size and the per-strip column limits.
        const int64_t next_strip  = qwen4exp_query_strip(params.ubatch.n_tokens/n_stream, n_stream);
        const auto    next_limits = qwen4exp_score_key_limits(mctx, params.ubatch, derived_bias, derived_vis, n_blocks,
                next_strip, ratio, (int64_t) params.hparams.indexer_top_k/ratio);
        res &= score_strip == next_strip;
        res &= score_key_limits == next_limits;

        return res;
    }

    // per stream: a cell index names a different token in each stream
    ggml_tensor * k_idxs    = nullptr;   // I32 [n_tokens]
    ggml_tensor * cell_blk  = nullptr;   // I32 [n_kv, n_stream]
    ggml_tensor * blk_cells = nullptr;   // I32 [ratio*n_blocks, n_stream]
    ggml_tensor * blk_pos   = nullptr;   // I32 [4*n_blocks*n_stream]
    ggml_tensor * bias      = nullptr;   // F32 [n_blocks or n_kv, n_tokens/n_stream, n_stream]
    ggml_tensor * rng       = nullptr;   // I32 [2*n_stream] derived-cache range leaf (decode only)

    // derived_bias: the per-block half of the bias is not uploaded at all - the top-k derives
    // it in-kernel from this compact pair (4 bytes per block instead of n_tokens/n_stream)
    ggml_tensor * blk_idx   = nullptr;   // I32 [n_blocks, n_stream]
    ggml_tensor * blk_tail  = nullptr;   // I32 [n_tokens/n_stream, n_stream]

    // derived_vis: the per-cell half (the attention mask) is not uploaded either - a cell carries
    // a compaction key and a token a query key, and the cell is visible iff 0 <= cell_vis <= q_vis
    ggml_tensor * cell_vis  = nullptr;   // I32 [n_kv, n_stream]
    ggml_tensor * q_vis     = nullptr;   // I32 [n_tokens/n_stream, n_stream]

    const llama_memory_hybrid_idx_context * mctx;
    const uint32_t ratio;

    // the per-cell half of the bias is the attention mask, so only the per-block half is uploaded
    const bool blk_bias;

    // ... unless it is derived from blk_idx/blk_tail (prefill only: the fused decode score op
    // applies the bias itself, and 2d positions select the ranked tail metric)
    const bool derived_bias;

    // ... and the per-cell half is derived from the visibility keys
    const bool derived_vis;

    // [QSA_SCORE_BOUNDS] strip size and per-strip score column limits baked into this graph
    int64_t score_strip = 0;
    std::vector<int64_t> score_key_limits;
};

ggml_tensor * llama_model_qwen4exp::graph::build_qsa_top_k(
        const llama_memory_hybrid_idx_context * mctx_hyb,
        ggml_tensor *                           cur,
        ggml_tensor *                           inp_pos,
        ggml_tensor *                           kq_mask,
        int *                                   sections,
        int                                     il) {
    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();

    const int64_t idx_dim  = hparams.indexer_head_size;
    const int64_t n_idx_h  = hparams.indexer_n_head;
    const int64_t r        = hparams.dsv4_compress_ratios[il];
    const int64_t n_kv     = mctx_idx->get_n_kv();

    GGML_ASSERT(r > 0);

    // blocks are keyed by position, which can run ahead of the occupied cells (see qsa_n_kv_window)
    const int64_t n_blocks = ((int64_t) mctx_hyb->qsa_n_kv_window() + r - 1)/r;
    // build_attn_qsa and the KQ mask need the tokens to divide evenly across the streams
    const int64_t n_stream = mctx_hyb->get_n_stream();
    GGML_ASSERT(n_tokens % n_stream == 0);
    const int64_t n_tps = n_tokens/n_stream;

    // only the "which block is visible" half of the bias varies per block
    // the rest is the visible/not test the attention mask already carries, so upload the per-block half only: 1/ratio of the cells
    // alibi writes distances instead of a mask and non-causal keeps future cells, so both opt out
    // the mask also holds an mrope rule for the query's own position, but only 2d image positions can differ there
    // kq_mask is null on the derived-visibility path (the mask is never created there); the
    // derived gate is the same predicate the mask's own shape would have satisfied
    const bool blk_bias = kq_mask != nullptr
            ? (kq_mask->ne[0] == n_kv && kq_mask->ne[1] == n_tps && kq_mask->ne[3] == n_stream &&
               cparams.causal_attn && !hparams.use_alibi)
            : qwen4exp_want_derived_vis(model, hparams, il, cparams, n_tokens);

    // The per-block half of the bias is a pure function of the block bookkeeping (blk_idx) and
    // the token's tail start (blk_tail), so the prefill graph has the top-k derive it in-kernel
    // instead of uploading an [n_blocks, n_tps] F32 tensor (400 MiB at ctx 204800 / ub 2048).
    // Decode keeps the tensor: the fused INDEXER_SCORE op applies the bias internally and its
    // rows are one token wide anyway.
    //
    // Note: not gated on !ubatch.is_pos_2d().  This model is IMROPE, so n_pos_per_embd() is 4 and
    // is_pos_2d() is always true, but that only describes the position *tensor*, not the data.
    // Both sides of the bias test come from the same code path - the memory layer computes
    // bid_idx and the query's tail start in rank units when it ranked the cells (mrope repeats a
    // position across an image) and in position units otherwise - so the comparison is exact
    // either way and the per-sequence half is redundant with the visibility.
    const int  dbias_mode   = qwen4exp_derived_bias_mode();
    const bool derived_bias = dbias_mode != 0 && blk_bias && n_tokens > 1;
    // the per-cell half of the same bias (the attention mask) is derived from the visibility
    // keys too; the mask tensor stays until the flash attention reads the keys as well
    const bool derived_vis  = derived_bias && qwen4exp_derived_vis_enabled();

    // env gates (per-process, read once): the fused INDEXER_SCORE op + the incremental derived
    // cache are the DEFAULT decode path (parity-verified byte-identical to the per-op chain);
    // =0 disables them for A/B.  The derived cache is what makes the QSA decode hold up at
    // depth (the 2026-09-07 crossover tables measured decode with both on).
    static const bool idx_score_fused = [] {
        const char * env = getenv("GGML_CUDA_QSA_INDEXER_SCORE");
        return env == nullptr || std::atoi(env) != 0;
    }();
    static const int idx_cache = [] {
        // 0 = off, 1 = on (default), 2 = probe-2 (pool passed to the score WITHOUT the fill)
        const char * env = getenv("GGML_CUDA_QSA_INDEXER_CACHE");
        return env == nullptr ? 1 : std::atoi(env);
    }();
    // [QSA_SCORE_WMMA] prefill indexer score -> one fused lightning-indexer op (all-ones weights,
    // zero mask) instead of the mul_mat + relu + head-sum chain.  RDNA3_5/RDNA4's 4-head WMMA
    // kernel is the fast path; elsewhere the generic vec kernel backs it.  The fused op casts q/k
    // to F16, so this is a *prefill re-baseline* (decode/verify width purity is untouched: the band
    // is n_tps >= 128).  Tri-state: unset = auto (see use_wmma; on RDNA3_0 auto picks the fused op
    // once the unfused score would exceed the threshold below, so the deep-prefill reserve stays
    // inside the device while shallow/medium prefill keeps the faster chain); =0 forces the per-op
    // chain; =1 forces the fused op.  (Issue #59/#60.)
    static const int idx_score_wmma = [] {
        const char * env = getenv("LLAMA_QSA_SCORE_WMMA");
        return env == nullptr ? -1 : (std::atoi(env) != 0 ? 1 : 0);
    }();
    // [issue #60] Auto's fused-vs-chain score-size threshold, in MiB.  The unfused chain's score
    // buffer is `score_blocks * n_idx_h * n_query * n_stream * 4`, and below the chunked branch
    // (also 128 MiB) it materialises the mul_mat result and its separate relu, so its reserve
    // peaks at twice the score size just under the old 128 MiB cutoff.  The reporter's 2x7900XTX
    // 196K profile needs the peak bounded well before that, and measures the chain as only ~7%
    // faster than the fused vec in the affected band, so auto switches at 64 MiB by default
    // (n_kv ~64K for their r=4 geometry).  LLAMA_QSA_SCORE_WMMA_MB=N overrides it, 0 = always fused.
    static const int64_t idx_score_wmma_threshold_bytes = []() -> int64_t {
        const char * env = getenv("LLAMA_QSA_SCORE_WMMA_MB");
        const long mb = env != nullptr ? atol(env) : 64;
        return mb <= 0 ? 0 : (int64_t) mb * 1024 * 1024;
    }();

    // nothing above depends on the layer, so the layers sharing a ratio share one input set
    llm_graph_input_qsa * inp = nullptr;

    const auto it = qsa_inps.find((uint32_t) r);
    if (it != qsa_inps.end()) {
        inp = it->second;
    } else {
        auto qsa = std::make_unique<llm_graph_input_qsa>(mctx_hyb, (uint32_t) r, blk_bias, derived_bias, derived_vis);

        qsa->k_idxs    = mctx_idx->build_input_k_idxs(ctx0, ubatch);
        qsa->cell_blk  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, n_stream);
        qsa->blk_cells = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, r*n_blocks, n_stream);
        qsa->blk_pos   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4*n_blocks*n_stream);
        if (derived_bias) {
            qsa->blk_idx  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_blocks, n_stream);
            qsa->blk_tail = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_tps,    n_stream);
        }
        if (derived_vis) {
            qsa->cell_vis = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv,  n_stream);
            qsa->q_vis    = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_tps, n_stream);
        }
        if (!derived_bias) {
            qsa->bias = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, blk_bias ? n_blocks : n_kv, n_tps, n_stream);
        }
        if (idx_cache && n_stream == 1) {
            // per-step derived fill range leaf: [from_s, lim_s] (one stream in decode)
            qsa->rng = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 2);
        }

        ggml_set_input(qsa->cell_blk);
        ggml_set_input(qsa->blk_cells);
        ggml_set_input(qsa->blk_pos);
        if (derived_bias) {
            ggml_set_input(qsa->blk_idx);
            ggml_set_input(qsa->blk_tail);
        }
        if (derived_vis) {
            ggml_set_input(qsa->cell_vis);
            ggml_set_input(qsa->q_vis);
        }
        if (!derived_bias) {
            ggml_set_input(qsa->bias);
        }
        if (qsa->rng != nullptr) {
            ggml_set_input(qsa->rng);
        }

        // [QSA_SCORE_BOUNDS] the prefill scorer is striped and causally trimmed; the widths are
        // baked into this graph and re-checked by can_reuse above.
        qsa->score_strip      = qwen4exp_query_strip(n_tps, n_stream);
        qsa->score_key_limits = qwen4exp_score_key_limits(mctx_hyb, ubatch, derived_bias, derived_vis, n_blocks,
                qsa->score_strip, (uint32_t) r, (int64_t) hparams.indexer_top_k/r);

        inp = qsa.get();
        res->add_input(std::move(qsa));
        qsa_inps.emplace((uint32_t) r, inp);
    }

    // cached indexer keys are raw: pooling precedes norm and rotation, so apply neither
    ggml_tensor * k_raw = build_lora_mm(model.layers[il].index_k_proj, cur);
    k_raw = ggml_reshape_3d(ctx0, k_raw, idx_dim, 1, n_tokens);
    cb(k_raw, "indexer_k_raw", il);

    ggml_build_forward_expand(gf, mctx_idx->cpy_k(ctx0, k_raw, inp->k_idxs, il));

    // one key head, so rows are contiguous. get_k gives [idx_dim, n_head_kv, n_kv, n_stream].
    ggml_tensor * k_all = mctx_idx->get_k(ctx0, il);
    k_all = ggml_view_3d(ctx0, k_all, idx_dim, n_kv, n_stream, k_all->nb[2], k_all->nb[3], 0);

    // query side: project + norm + rotate (shared by the fused and per-op paths)
    ggml_tensor * q = build_lora_mm(model.layers[il].index_q_proj, cur);
    q = ggml_reshape_3d(ctx0, q, idx_dim, n_idx_h, n_tokens);
    q = build_norm(q, model.layers[il].index_q_norm, nullptr, LLM_NORM_RMS, il);
    q = ggml_rope_multi(ctx0, q, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(q, "indexer_q", il);

    // the fused score ops read the raw cache rows natively - the kernels' load dispatch is
    // F32/BF16/F16 only, and the ggml_indexer_score/fill constructors assert the same, so a
    // quantized indexer key cache (e.g. --cache-type-k q8_0, which is applied to the indexer
    // store too) must NOT reach them.  Fall back to the per-op chain below, whose get_rows
    // dequantizes any cache type on gather (the BF16/f32 fused decode path is unchanged).
    const bool idx_key_float = k_all->type == GGML_TYPE_F32 ||
                               k_all->type == GGML_TYPE_BF16 ||
                               k_all->type == GGML_TYPE_F16;

    // every token of a block gets the block score; the budget is whole blocks, so top-k cuts on a block boundary
    // the reference returns indexer_top_k + compress_ratio - 1: whole blocks plus the tail
    const int64_t width = std::min<int64_t>(n_kv, (int64_t) hparams.indexer_top_k + r - 1);

    // fused expand + mask + top-k: value[c] = score[cell_blk[c]] + additive[c]
    // avoids materializing the [n_kv, n_tps] F32 expanded tensor (512MB at 64K) per layer
    // the mask is [n_kv, n_tps, 1, n_stream]; the size-1 dim is a no-op stride, so the
    // kernel reads it as [n_kv, n_tps, n_stream] and no per-layer copy is needed
    ggml_tensor * additive = blk_bias ? kq_mask : inp->bias;

    // [QSA_SCORE_BOUNDS] the query-indexed inputs (additive, q_vis, blk_tail) are sliced to the
    // strip so each per-strip top-k sees the same shapes the whole-batch decode call does.
    auto run_top_k = [&](ggml_tensor * sc, int64_t first, int64_t n_query, int64_t score_blocks) -> ggml_tensor * {
        ggml_tensor * add_s = nullptr;
        if (additive != nullptr) {
            add_s = ggml_n_dims(additive) == 4
                    ? ggml_view_4d(ctx0, additive, additive->ne[0], n_query, 1, n_stream,
                            additive->nb[1], additive->nb[2], additive->nb[3], first*additive->nb[1])
                    : ggml_view_3d(ctx0, additive, additive->ne[0], n_query, n_stream,
                            additive->nb[1], additive->nb[2], first*additive->nb[1]);
        }
        ggml_tensor * qpos_s = derived_vis ? ggml_view_2d(ctx0, inp->q_vis, n_query, n_stream,
                inp->q_vis->nb[1], first*inp->q_vis->nb[0]) : nullptr;
        ggml_tensor * tail_s = derived_bias ? ggml_view_2d(ctx0, inp->blk_tail, n_query, n_stream,
                inp->blk_tail->nb[1], first*inp->blk_tail->nb[0]) : nullptr;
        // [QSA_SCORE_BOUNDS] a trimmed score holds only the visible prefix; the block fast path
        // then skips the trimmed blocks by cell index.  The bound only applies with n_stream == 1,
        // so these views stay contiguous even though the source's stream stride is the full width.
        ggml_tensor * blk_idx_s = derived_bias
                ? (score_blocks < n_blocks
                        ? ggml_view_2d(ctx0, inp->blk_idx, score_blocks, n_stream, inp->blk_idx->nb[1], 0)
                        : inp->blk_idx)
                : nullptr;
        ggml_tensor * blk_cells_s = score_blocks < n_blocks
                ? ggml_view_2d(ctx0, inp->blk_cells, (int64_t) r*score_blocks, n_stream, inp->blk_cells->nb[1], 0)
                : inp->blk_cells;
        return ggml_indexer_top_k(ctx0, sc, inp->cell_blk, add_s,
                derived_vis ? inp->cell_vis : nullptr,
                qpos_s,
                blk_idx_s,
                tail_s,
                blk_cells_s,
                (int) width);
    };

    ggml_tensor * top_k = nullptr;

    ggml_tensor * score = nullptr;
    if (idx_score_fused && idx_key_float && n_tokens == 1 && blk_bias && n_idx_h <= 8 &&
            rope_type == GGML_ROPE_TYPE_IMROPE) {
        // FUSED PROBE (env-gated, GGML_CUDA_QSA_INDEXER_SCORE=1): ONE kernel replaces the
        // per-token decode chain gather + r-slice pool + scale + rms_norm + rope_multi +
        // score mul_mat + relu + head-sum + bias (~12 kernels/layer) with the fused
        // INDEXER_SCORE op.  The kernel replicates the per-op F32 arithmetic byte-
        // identically (same gather addresses, add order, 256-thread rms_norm reduction,
        // IMROPE half-pair rope, mmvf F32 vec-dot order) - see ggml-cuda/indexer-score.cu.
        ggml_tensor * score_pool = nullptr;   // derived-cache pool view (fill dst), if used
        if (idx_cache && inp->rng != nullptr) {
            // DERIVED-CACHE PROBE (env-gated, GGML_CUDA_QSA_INDEXER_CACHE=1, on top of the fused
            // score): a fill op first writes the completed blocks' pooled+normed+ROTATED vectors
            // into the memory-layer pool (rows [from, lim) of this step's range leaf), then the
            // score op reads the pool rows for blocks below the limit instead of re-pooling the
            // raw cache - the same F32 arithmetic at fill time, so the score is byte-identical.
            ggml_tensor * pool_v = mctx_hyb->get_pool(ctx0, il, (uint32_t) n_blocks);
            if (pool_v != nullptr) {
                // idx_cache == 2 is a DEBUG PROBE: pass the pool to the score (lim = n_bid)
                // WITHOUT building the fill - if the output diverges, the score genuinely reads
                // the (unfilled) pool rows; if it stays identical, the derived read is a no-op.
                if (idx_cache != 2) {
                    score_pool = ggml_indexer_fill(ctx0, k_all, inp->blk_cells, inp->blk_pos,
                            model.layers[il].index_k_norm, inp->rng, pool_v,
                            (int) r, hparams.f_norm_rms_eps,
                            (int) n_rot, sections, (int) rope_type, (int) n_ctx_orig,
                            freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                    cb(score_pool, "indexer_fill", il);
                } else {
                    // probe-2: the score reads pool_v directly, rows are garbage (never filled)
                    score_pool = pool_v;
                }
            }
        }
        // pass the fill's dst (a view of the pool buffer) so the score reads the rows the fill
        // just wrote: the graph orders the score after the fill through the src dependency
        score = ggml_indexer_score(ctx0, k_all, inp->blk_cells, inp->blk_pos,
                ggml_reshape_3d(ctx0, q, idx_dim, n_idx_h*n_tps, n_stream),
                model.layers[il].index_k_norm, inp->bias,
                (int) r, hparams.f_norm_rms_eps,
                (int) n_rot, sections, (int) rope_type, (int) n_ctx_orig,
                freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow,
                score_pool, score_pool != nullptr ? inp->rng : nullptr);
        // register the fused indexer score so resolve_fused_ops()'s Lightning Indexer probe can
        // report a layer/device mismatch.  cparams.auto_flid is false in this tree, so the probe
        // is inert for now, but the registration keeps us aligned with upstream 889edf43d.
        res->add_fused_node({LLM_FUSED_OP_LIGHTNING_INDEXER, score, il});
    } else {
        // gathers per stream: blk_cells row s indexes stream s's own cells
        ggml_tensor * members = ggml_get_rows(ctx0, k_all, inp->blk_cells);
        members = ggml_reshape_4d(ctx0, members, idx_dim, r, n_blocks, n_stream);

        ggml_tensor * pooled = nullptr;
        {
            // mean over the block members; r is small, so summing slices beats a transpose plus sum_rows
            for (int64_t i = 0; i < r; ++i) {
                ggml_tensor * slice = ggml_cont(ctx0,
                        ggml_view_3d(ctx0, members, idx_dim, n_blocks, n_stream,
                                members->nb[2], members->nb[3], i*members->nb[1]));
                pooled = pooled ? ggml_add(ctx0, pooled, slice) : slice;
            }
            pooled = ggml_scale(ctx0, pooled, 1.0f/(float) r);
            // count blocks along ne1: rms_norm launches gridDim.y = ne2, capped at 65535, and 262144/4 = 65536
            pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, n_blocks*n_stream, 1);
            pooled = build_norm(pooled, model.layers[il].index_k_norm, nullptr, LLM_NORM_RMS, il);
            pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, n_blocks, n_stream);
        }
        cb(pooled, "indexer_k_pooled", il);

        // rope wants [n_dims, n_head, n_tokens]: lay every stream's blocks flat, split after.
        pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, 1, n_blocks*n_stream);
        pooled = ggml_rope_multi(ctx0, pooled, inp->blk_pos, nullptr,
                n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, n_blocks, n_stream);
        cb(pooled, "indexer_k", il);

        // rectify each head dot product before the sum, as in the DeepSeek lightning indexer
        // mul_mat matches ne[2], so the queries of stream s only meet the blocks of stream s
        ggml_tensor * q3 = ggml_reshape_3d(ctx0, q, idx_dim, n_idx_h*n_tps, n_stream);

        // The score chain is the largest F32 tensor in the graph, so it carries two memory fixes
        // that together are the L2 win (GGML_QSA_SCORE_MEM, default on - =0 restores the plain
        // reference chain for A/B):
        //
        //   L2a: relu BEFORE the 4-D reshape.  relu is elementwise, so this is bit-identical, but
        //        it makes the relu's parent the mul_mat result instead of a reshape view.  The
        //        allocator cannot reuse a view parent in place (a view's data is NULL at planning
        //        time, so it is treated as external), which otherwise keeps the
        //        [n_blocks, n_idx_h, n_tps] F32 result and its relu alive simultaneously
        //        (2 x 1600 MB at ctx 204800 / ub 2048).
        //   L2m: M-chunked assembly.  n_blocks is an independent output (M) dimension of the
        //        matmul (the reduction is over idx_dim only), so slicing it leaves the per-element
        //        arithmetic untouched; the slices are assembled with ggml_concat (a produced tensor
        //        frees normally, unlike cpy into a view of an allocator-owned tensor, which leaks).
        //        The threshold is internal policy, not the A/B knob.
        //
        // [QSA_SCORE_BOUNDS] prefill is additionally scored in `strip`-token query strips, each
        // trimmed to the blocks it can see (qwen4exp_score_key_limits): a complete block's ordinal
        // cannot exceed its logical block number, so columns past (max_query_pos+1)/ratio are
        // invisible.  The trimmed score is handed straight to the fused top-k, whose cell range is
        // clamped to n_blocks*ratio so the trimmed cells (cell_blk[c] >= n_blocks) are never read;
        // the padded-blocks variant was measured slower because it re-materialises the full score.
        // A strip that is not trimmed (decode/verify, no bound) builds exactly the pre-strip graph,
        // so W = 1..8 stays byte-identical.
        const bool    score_mem = qwen4exp_score_mem_enabled();
        const int64_t strip     = inp->score_strip > 0 ? inp->score_strip : n_tps;

        // sum the indexer heads: ne[1] holds them side by side and there are only a few
        auto head_sum = [&](ggml_tensor * sc, int64_t nb, int64_t n_query) {
            ggml_tensor * summed = nullptr;
            for (int64_t h = 0; h < n_idx_h; ++h) {
                ggml_tensor * slice = ggml_view_3d(ctx0, sc, nb, n_query, n_stream,
                        sc->nb[2], sc->nb[3], h*sc->nb[1]);
                summed = summed ? ggml_add(ctx0, summed, slice) : ggml_cont(ctx0, slice);
            }
            return summed;
        };

        auto compute_score = [&](int64_t first, int64_t n_query, int64_t score_blocks) {
            ggml_tensor * pooled_s = score_blocks == n_blocks ? pooled :
                    ggml_view_3d(ctx0, pooled, idx_dim, score_blocks, n_stream,
                            pooled->nb[1], pooled->nb[2], 0);
            ggml_tensor * q_s = (first == 0 && n_query == n_tps) ? q3 :
                    ggml_view_3d(ctx0, q, idx_dim, n_idx_h*n_query, n_stream,
                            q->nb[1], q->nb[2]*n_tps, first*q->nb[2]);

            // [QSA_SCORE_WMMA] the fused op needs 128 dims and exactly 4 heads (the AMD WMMA
            // kernel's shape), and only pays in the prefill band; decode/verify keep the per-op
            // chain so the W=1..8 reduction order is unchanged.
            //
            // [issue #59/#60] On RDNA3_0 the generic-vec fused op is slower than the unfused
            // chain at shallow/medium prefill (measured ~7% in the 64K band) but its score buffer
            // is bounded, where the unfused chain's is not: below its own 128 MiB chunked branch
            // the chain materialises the mul_mat result and a separate relu, so its reserve peaks
            // at twice the score size, and past that the chunked assembly still reserves more than
            // the fused op.  The 196K prefill lives on the wrong side of that, so auto switches to
            // the fused op once the score passes idx_score_wmma_threshold_bytes (64 MiB by default,
            // LLAMA_QSA_SCORE_WMMA_MB overrides); the chain stays below it, and everywhere on
            // RDNA3_5/RDNA4, whose 4-head WMMA arm is the tuned path.  LLAMA_QSA_SCORE_WMMA=0/1
            // overrides the policy, so the knob works on gfx1100 again.
            const int64_t score_bytes = score_blocks * n_idx_h * n_query * n_stream * (int64_t) sizeof(float);
            const int  gfx      = qsa_arch_gfx();
            const bool rdna3_0  = gfx >= 0x1100 && gfx < 0x1150;   // gfx1100/1101/1102/1103 (RDNA3_5 is 0x1150+)
            const bool wmma_wanted = idx_score_wmma < 0
                    ? (!rdna3_0 || score_bytes > idx_score_wmma_threshold_bytes)
                    : (idx_score_wmma != 0);
            const bool use_wmma = wmma_wanted && n_tps >= 128 && idx_dim == 128 && n_idx_h == 4 &&
                    lightning_indexer_op_supported(model, il, n_tps);

            // ones-weights / zero-mask, shared across layers by name (the graph context outlives a
            // layer build); the mask is sized by the untrimmed block count and viewed per strip.
            auto score_wmma_weights = [&]() -> ggml_tensor * {
                char name[64];
                snprintf(name, sizeof(name), "qsa_score_w_%lld_%lld_%lld",
                        (long long) n_idx_h, (long long) strip, (long long) n_stream);
                ggml_tensor * t = ggml_get_tensor(ctx0, name);
                if (t == nullptr) {
                    t = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_idx_h, strip, 1, n_stream), 1.0f);
                    ggml_set_name(t, name);
                }
                return t;
            };
            auto score_wmma_mask = [&]() -> ggml_tensor * {
                char name[64];
                snprintf(name, sizeof(name), "qsa_score_m_%lld_%lld_%lld",
                        (long long) n_blocks, (long long) strip, (long long) n_stream);
                ggml_tensor * t = ggml_get_tensor(ctx0, name);
                if (t == nullptr) {
                    t = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F16, n_blocks, strip, 1, n_stream), 0.0f);
                    ggml_set_name(t, name);
                }
                return t;
            };

            ggml_tensor * strip_score = nullptr;
            if (use_wmma) {
                ggml_tensor * w_full = score_wmma_weights();
                ggml_tensor * m_full = score_wmma_mask();
                ggml_tensor * query = ggml_view_4d(ctx0, q, idx_dim, n_idx_h, n_query, n_stream,
                        q->nb[1], q->nb[2], q->nb[2]*n_tps, first*q->nb[2]);
                ggml_tensor * key = ggml_view_4d(ctx0, pooled, idx_dim, 1, score_blocks, n_stream,
                        pooled->nb[1], pooled->nb[1], pooled->nb[2], 0);
                ggml_tensor * weights_v = ggml_view_4d(ctx0, w_full, n_idx_h, n_query, 1, n_stream,
                        w_full->nb[1], w_full->nb[2], w_full->nb[3], 0);
                // leading-rows view: the all-zero mask's ne0 is n_blocks, the op needs score_blocks
                ggml_tensor * mask_v = ggml_view_4d(ctx0, m_full, score_blocks, n_query, 1, n_stream,
                        m_full->nb[1], m_full->nb[2], m_full->nb[3], 0);
                ggml_tensor * li = ggml_lightning_indexer(ctx0, query, key, weights_v, mask_v);
                res->add_fused_node({LLM_FUSED_OP_LIGHTNING_INDEXER, li, il});
                strip_score = ggml_reshape_3d(ctx0, li, score_blocks, n_query, n_stream);
            } else if (score_mem && score_bytes > 128*1024*1024) {
                const int64_t chunk = std::max<int64_t>(4096, (score_blocks + 15)/16);
                ggml_tensor * acc = nullptr;
                for (int64_t b0 = 0; b0 < score_blocks; b0 += chunk) {
                    const int64_t cb = std::min<int64_t>(chunk, score_blocks - b0);

                    ggml_tensor * pooled_c = ggml_view_3d(ctx0, pooled_s, idx_dim, cb, n_stream,
                            pooled_s->nb[1], pooled_s->nb[2], b0*pooled_s->nb[1]);

                    ggml_tensor * sc = ggml_mul_mat(ctx0, pooled_c, q_s);
                    // in-place: the relu shares the mul_mat buffer, so the chunked assembly does
                    // not hold the score and its relu at once (issue #60; bit-identical)
                    sc = ggml_relu_inplace(ctx0, sc);
                    sc = ggml_reshape_4d(ctx0, sc, cb, n_idx_h, n_query, n_stream);

                    ggml_tensor * summed = head_sum(sc, cb, n_query);

                    acc = acc ? ggml_concat(ctx0, acc, summed, 0) : summed;
                }
                strip_score = acc;
            } else if (score_mem) {
                strip_score = ggml_mul_mat(ctx0, pooled_s, q_s);
                // in-place: below the chunked branch the score is the whole strip, so sharing the
                // buffer with its relu halves the chain's reserve peak (issue #60; bit-identical)
                strip_score = ggml_relu_inplace(ctx0, strip_score);
                strip_score = ggml_reshape_4d(ctx0, strip_score, score_blocks, n_idx_h, n_query, n_stream);
                strip_score = head_sum(strip_score, score_blocks, n_query);
            } else {
                strip_score = ggml_mul_mat(ctx0, pooled_s, q_s);
                strip_score = ggml_reshape_4d(ctx0, strip_score, score_blocks, n_idx_h, n_query, n_stream);
                strip_score = ggml_relu(ctx0, strip_score);
                strip_score = head_sum(strip_score, score_blocks, n_query);
            }

            // one value per block, so it is cheaper to bias here than after the cells are expanded
            // (the derived path drops this tensor and lets the top-k add the bias in-kernel instead)
            if (blk_bias && !derived_bias) {
                ggml_tensor * bias_s = ggml_view_3d(ctx0, inp->bias, n_blocks, n_query, n_stream,
                        inp->bias->nb[1], inp->bias->nb[2], first*inp->bias->nb[1]);
                strip_score = ggml_add(ctx0, strip_score, bias_s);
            }

            return strip_score;
        };

        std::vector<ggml_tensor *> strips;
        for (int64_t first = 0; first < n_tps; first += strip) {
            const int64_t n_query = std::min(strip, n_tps - first);
            const int64_t score_blocks = inp->score_key_limits.empty()
                    ? n_blocks : inp->score_key_limits[first/strip];
            GGML_ASSERT(score_blocks > 0 && score_blocks <= n_blocks);

            ggml_tensor * sc = compute_score(first, n_query, score_blocks);
            cb(sc, "indexer_score", il);

            ggml_tensor * tk = run_top_k(sc, first, n_query, score_blocks);
            cb(tk, "indexer_top_k", il);
            strips.push_back(tk);
        }

        // balanced concatenation along the query dimension keeps the strips in order
        while (strips.size() > 1) {
            ggml_tensor * right = strips.back(); strips.pop_back();
            ggml_tensor * left  = strips.back(); strips.pop_back();
            ggml_tensor * joined = ggml_concat(ctx0, left, right, 1);
            ggml_build_forward_expand(gf, joined);
            strips.push_back(joined);
        }
        top_k = strips.front();
    }
    if (score != nullptr) {
        cb(score, "indexer_score", il);
        top_k = run_top_k(score, 0, n_tps, n_blocks);
        cb(top_k, "indexer_top_k", il);
    }

    return top_k;
}

// qsa3 packed-block layouts (the F16 layouts the WMMA prefill kernel reads, ggml/src/ggml-cuda/
// fattn-qsa3.cu).  The pack itself is done by the launcher in one fused pass per tensor
// (qsa3_pack_{keys,values}_kernel, session 7); the graph only materialises the natural contiguous F16
// view [D, n_head_kv, n_kv] the launcher packs from.  Keys -> [16, 4, 16, n_kv/4*n_kvh]; Values ->
// [4, 256, n_kv/4*n_kvh, 1].  Both are built only for the prefill band and any cache type the graph
// can materialize as F16 (the native ones, or a quantized cache cast on the way in); anything else
// leaves the VEC kernel in charge.
// The cache types the qsa3 pack can consume: the native pair plus every type the VEC kernel can
// dequantize (the cast below materializes them).  Kept in sync with the VEC kernel's type list.
static bool qwen4exp_qsa3_kv_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_IQ4_NL:
            return true;
        default:
            return false;
    }
}

// qsa3 works on F16 packs.  Materialize the cache view as F16 from its *natural* (contiguous)
// layout before any permute: a quantized tensor cannot be permuted, and the backend dup only
// dequantizes to F32, so those route through an F32 intermediate.  The permute/cont that follows
// is then a plain F16 copy.
static ggml_tensor * qsa3_f16_cast(ggml_context * ctx0, ggml_tensor * raw) {
    if (raw->type == GGML_TYPE_F16)  { return raw; }
    if (raw->type == GGML_TYPE_BF16) { return ggml_cast(ctx0, raw, GGML_TYPE_F16); }
    return ggml_cast(ctx0, ggml_cast(ctx0, raw, GGML_TYPE_F32), GGML_TYPE_F16);
}

static ggml_tensor * qsa3_f16_natural(ggml_context * ctx0, ggml_tensor * raw) {
    ggml_tensor * t = qsa3_f16_cast(ctx0, raw);
    if (!ggml_is_contiguous(t)) { t = ggml_cont(ctx0, t); }
    return t;
}

// Dense GQA self-attention restricted to the cells that top_k names.
// The mask build below copies the MLA sparse path in llm_graph_context::build_attn.
ggml_tensor * llama_model_qwen4exp::graph::build_attn_qsa(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             q_cur,
        ggml_tensor *             k_cur,
        ggml_tensor *             v_cur,
        ggml_tensor *             top_k,
        float                     kq_scale,
        int                       il) {
    // rotate q/k/v before they reach a quantized cache, as the dense path does. the indexer
    // has already scored with its own query in build_qsa_top_k, so top_k is unaffected.
    if (inp->self_k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, inp->self_k_rot);
        k_cur = llama_mul_mat_hadamard(ctx0, k_cur, inp->self_k_rot);
    }

    if (inp->self_v_rot) {
        v_cur = llama_mul_mat_hadamard(ctx0, v_cur, inp->self_v_rot);
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();
        const auto & v_idxs = inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    // Fused sparse FA (top-k cells only) vs dense masked FA: see qwen4exp_qsa_sparse().
    const bool qsa_sparse = qwen4exp_qsa_sparse(model, hparams, il, cparams);

    // the derived visibility keys live in this layer's QSA graph input set (null on the tensor
    // path, where the kernel keeps gathering the mask); the mask src stays until the prunable form
    ggml_tensor * qsa_cell_vis = nullptr;
    ggml_tensor * qsa_q_vis    = nullptr;
    {
        const auto it_qsa = qsa_inps.find((uint32_t) hparams.dsv4_compress_ratios[il]);

        if (it_qsa != qsa_inps.end()) {
            qsa_cell_vis = it_qsa->second->cell_vis;
            qsa_q_vis    = it_qsa->second->q_vis;
        }
    }

    // the derived-visibility path creates no mask at all (nothing in this graph reads it: the FA
    // derives from the keys and the dense fallback below is off) - this is the -800 MiB win
    const bool qsa_derive_vis = qsa_sparse && qsa_cell_vis != nullptr;
    ggml_tensor * kq_mask = qsa_derive_vis ? nullptr : inp->get_kq_mask();

    ggml_tensor * kq_mask_top_k = nullptr;

    if (kq_mask != nullptr) {
    // prepare new kq mask - starts filled with -INFINITY
    ggml_tensor * kq_mask_all = ggml_fill(ctx0, kq_mask, -INFINITY);

    // reshape KQ mask into tensor with rows of size 1:
    // [n_kv, n_batch, 1, n_stream] -> [1, n_kv, n_batch, n_stream]
    kq_mask_all = ggml_view_4d(ctx0, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3], kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

    // reshape top_k indices: [n_top_k, n_batch, 1, n_stream] -> [n_top_k, n_batch, n_stream, 1]
    ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1, top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);

    // prepare zero-filled tensor with rows of size 1: [1, n_top_k, n_batch, n_stream]
    // this will be our source of zero values for unmasking top k mask elements
    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
    zeros = ggml_fill(ctx0, zeros, 0.0f);

    // modify KQ mask by unmasking elements that are in top_k indices
    // ggml_set_rows([1, n_kv, n_batch, n_stream], [1, n_top_k, n_batch, n_stream], [n_top_k, n_batch, n_stream, 1])
    kq_mask_top_k = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);

    // reshape to restore the original shape of KQ mask:
    // [1, n_kv, n_batch, n_stream] -> [n_kv, n_batch, 1, n_stream]
    kq_mask_top_k = ggml_view_4d(ctx0, kq_mask_top_k, kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3], kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);

    // combine with the original kq mask
    kq_mask_top_k = ggml_add(ctx0, kq_mask_top_k, kq_mask);
    }

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);
    // the natural cache views, kept for the qsa3 packs (which must cast before permuting)
    ggml_tensor * k_raw = k;
    ggml_tensor * v_raw = v;
    ggml_tensor * cur;

    if (qsa_sparse) {
        // same view/permute as build_attn_mha does internally
        const int64_t n_stream = k->ne[3];
        ggml_tensor * q_p = ggml_view_4d(ctx0, q, q->ne[0], q->ne[1], q->ne[2]/n_stream, n_stream,
                q->nb[1], q->nb[2], q->nb[3]/n_stream, 0);
        q_p = ggml_permute(ctx0, q_p, 0, 2, 1, 3);
        k = ggml_permute(ctx0, k, 0, 2, 1, 3);
        v = ggml_permute(ctx0, v, 0, 2, 1, 3);

        // kq_mask is null on the derived-visibility path (the graph never creates it): the kernel
        // derives each cell's visibility from cell_vis/q_vis instead - see findings 2c
        cur = ggml_flash_attn_qsa(ctx0, q_p, k, v, top_k, kq_mask, kq_scale, 0.0f,
                qsa_cell_vis, qsa_q_vis);
        // qsa3 (packed-block WMMA) needs the packed F16 layouts; it is prefill-only, so the whole
        // W=1..8 decode/verify band keeps the VEC kernel and width purity holds by construction.
        // Compile-time gate (see LLAMA_QSA3_ENABLE at the top of this file, default 1) plus the
        // backend probe: never materialise the F16 packs when the acting device's qsa3 predicate
        // will not read them (the runtime arch/env policy lives there; issue #59/#60).
        constexpr bool qsa3_pack = LLAMA_QSA3_ENABLE != 0;
        if (qsa3_pack && n_stream == 1 && q_p->ne[1] >= 128 && k_raw->ne[0] == 256 && k_raw->ne[2] % 4 == 0 &&
            qwen4exp_qsa3_kv_type(k_raw->type) && qsa3_op_supported(model, hparams, il, k_raw->type)) {
            // The launcher packs these into the block layouts itself (one fused pass, session 7);
            // the graph only materialises the natural contiguous F16 view [D, n_head_kv, n_kv].
            ggml_tensor * k16 = qsa3_f16_natural(ctx0, k_raw);
            ggml_tensor * v16 = qsa3_f16_natural(ctx0, v_raw);
            ggml_flash_attn_qsa_set_packed(cur, k16, v16);
        }

        ggml_flash_attn_qsa_set_prec(cur, GGML_PREC_F32);
        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
        cb(cur, "kqv_out", il);

        // the rotation is its own inverse, so undo it on the value side of the output
        if (inp->self_v_rot) {
            cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
        }

        return cur;
    }

    cur = build_attn_mha(q, k, v, nullptr, kq_mask_top_k, nullptr, nullptr, 0, kq_scale, il);
    cb(cur, "kqv_out", il);

    // the rotation is its own inverse, so undo it on the value side of the output
    if (inp->self_v_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
    }

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        const llama_memory_hybrid_idx_context * mctx_hyb,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // indexer reads the same block input as q/k/v; no context, no cache, or no ratio means dense.
    // LLAMA_QSA_OFF=1 forces the dense no-indexer regime everywhere (a plain-dense reference:
    // no indexer store, scoring or sparse selection at any layer) - a gate knob, default OFF.
    const bool qsa_off = qwen4exp_qsa_off();
    const bool qsa = !qsa_off && mctx_hyb != nullptr && mctx_hyb->get_idx() != nullptr && hparams.dsv4_compress_ratios[il] > 0;

    ggml_tensor * top_k = nullptr;
    if (qsa) {
        // The selection keeps min(n_kv, top_k + ratio - 1) cells. While the cache holds no more
        // cells than that, every cell is selected and the top-k mask is a no-op on the causal
        // mask, so attend dense (same result) and only store this ubatch's indexer keys for
        // when the context grows past the budget. Mirrors the reference "seqlen <= topk" path.
        const int64_t r     = hparams.dsv4_compress_ratios[il];
        const int64_t n_kv  = mctx_hyb->get_idx()->get_n_kv();
        const int64_t width = (int64_t) hparams.indexer_top_k + r - 1;
        // LLAMA_QSA_DENSE_SHORTCUT: attend dense while every cell is still selected (n_kv <= the
        // selection width).  **DEFAULT OFF as of 2026-09-19 = always QSA** (the env name matches B
        // for cross-testing; LLAMA_QSA_DENSE_SHORTCUT=1 restores the dense shortcut).
        //
        // It was DEFAULT ON for B parity, on the reasoning that below the width the top-k selects
        // every cell, so sparse attention saves no work while still paying the indexer + top-k.
        // That stopped being true once qsa3 landed: measured at pp2048 (single ubatch, every cell
        // selected, gfx1151) the qsa3 attention kernel is 137.8 ms vs 149.9 ms for the dense
        // flash_attn_ext_f16 - i.e. qsa3 is already 8 % faster on the attention even when nothing
        // is skipped.  The path only still lost because the indexer + top-k (20.9 ms) costs more
        // than the kernel saves (12.1 ms), so the shortcut's net value had fallen to ~1 % at short
        // prefill and ~0 at depth.  It also left a numerics discontinuity at n_kv == width and kept
        // qsa3 out of every <= 2051-token validation (a `-c 2048` PPL exercise was silently dense).
        // Removing the split makes the prefill attention path uniform and always exercised; the
        // remaining gap is an indexer/top-k cost to attack, not a reason to keep two regimes.
        //
        // History (kept because it documents the alternative): the shortcut was once opt-in OFF
        // while a llama-bench-only multi-ubatch artifact (pp4096+@0 slower with it on) lacked a root
        // cause. Root-caused 2026-09-08/09: llama-bench's sync-free decode pipeline x ggml-gallocr's
        // single-layout alloc-fallback doing an unconditional full-device sync on every dense/sparse
        // topology flip (record: benchmarks/2026-09-10-*-shortcut-fix.md).  NOTE (2026-09-06): the
        // follow-up that removed that sync (ggml_gallocr_reserve_n_probe - sync only when a buffer
        // must grow) was UNSAFE across backends and has been reverted: a re-reserve re-points tensor
        // addresses while the previous ubatch's kernels may still be in flight on other GPUs
        // (cross-GPU race -> in-kernel spin / memory faults at multi-ubatch prefill, reproduced on
        // 3x R9700 gfx1201).  The per-flip full-device sync is back (upstream behavior).  Note the
        // dense path's numerics below the width equal the LLAMA_QSA_SPARSE_FA=0 masked-dense path -
        // useful as a cross-check, and still reachable with LLAMA_QSA_DENSE_SHORTCUT=1.  Decode is
        // unaffected by this default either way: it stays dense via qsa_dense_decode_until below.
        //
        // ARCH POLICY (2026-09-21, gfx1201 port): the WIP's always-QSA default (shortcut OFF) was
        // measured on gfx1151, where qsa3 makes the sparse attention competitive below the
        // selection width.  On gfx1201 the VEC QSA path is NOT: measured 2026-09-21, 3x R9700
        // -sm tensor, IQ4_XS, -b/-ub 2048, restoring the shortcut took pp2048 1539.6 -> 2097.4,
        // pp8192 2169.8 -> 2429.5, pp32768 2359.4 -> 2420.5 (+36 % / +12 % / +2.6 %).  The
        // shortcut also gates the first ~2051 tokens of EVERY prefill, so it is not a pp2048-only
        // effect.  Default the shortcut ON on every arch not measured to prefer always-QSA;
        // gfx1151 keeps the WIP's measured always-QSA.  Revisit per arch once qsa3 (the
        // packed-WMMA prefill) is ported - it is what made always-QSA viable on gfx1151.
        static const bool shortcut = [] {
            const char * env = getenv("LLAMA_QSA_DENSE_SHORTCUT");
            if (env != nullptr) return atoi(env) != 0;
            return qsa_arch_gfx() != 0x1151;
        }();
        // ARCH POLICY (re-measured 2026-09-22, fork `gap-closing-r13`, qwen4exp IQ4_NL f16 KV):
        // decode uses the plain dense attend while n_kv < the per-arch crossover depth and
        // QSA (sparse) at/above it.  gfx1151 (Strix Halo, 1 GPU) - interleaved forced-dense vs
        // forced-sparse, `-n 200 --ctx-checkpoints 0`: 16K dense 30.4 / sparse 29.4 (dense +3.4 %),
        // 32K 28.7 / 28.6 (parity), 48K 27.4 / 27.9 (sparse +1.8 %), 64K 26.1 / 27.3 (sparse
        // +4.6 %) -> the gfx1151 crossover is now **32K**, down from the 64K of the 2026-09-07
        // tables.  The move is the derived indexer cache (patches/0021): with it ON the sparse
        // decode stops re-pooling the raw indexer cache every step, which is what the old 64K
        // threshold was sized around.  gfx1201 (3x R9700) - dense wins at every measured depth
        // (8K-160K, flat ~8 %), so it never switches.  PREFILL has its own crossover
        // (qsa_dense_prefill_until, below).  The indexer keys are stored as the width-shortcut
        // does, so the sparse path takes over seamlessly above the depth.
        // LLAMA_QSA_DENSE_DECODE_UNTIL overrides the threshold in tokens (0 disables the
        // gate = QSA decode always, for A/B).
        static const int64_t qsa_dense_decode_until = []() -> int64_t {
            const int64_t env = qsa_env_tokens("LLAMA_QSA_DENSE_DECODE_UNTIL");
            if (env >= 0) {
                return env;
            }
            return qsa_arch_gfx() == 0x1151 ? 32768 : ((int64_t) 1 << 62);  // gfx1151 crossover ~32K; else dense always
        }();
        // the prefill counterpart: dense below `qsa_dense_prefill_until`, sparse above.  DEFAULT 0 =
        // **QSA prefill always**, the documented ARCH POLICY (2026-09-07 crossover tables;
        // beta/qwen4exp/README.md + wip/archive/qwen4exp/discovery/2026-09-07-qsa-dense-crossover-
        // tables-soar-halo.md): "decode uses the dense attend below a per-arch depth and QSA above;
        // **prefill is always QSA**" - on Soar (gfx1201, 3x R9700 tensor) QSA wins prefill from ~8K
        // monotonically to +181 % @160K, on Halo from ~16K, so there is no dense-prefill regime to
        // default to on either arch.  The knob exists because the crossover was not
        // depth-configurable (TODO item 9); it is opt-in only (LLAMA_QSA_DENSE_PREFILL_UNTIL=<tokens>,
        // K/M/G suffixes), 0 = the default = sparse prefill always.
        //
        // If it is ever turned on by default, measure it in the *comparable* shape: the 2026-09-07
        // tables are pp2048 *at depth*, and that record explicitly rejects the whole-prompt banner
        // shape ("the old \"dense wins prefill at 30K\" record is obsolete ... also a non-comparable
        // whole-prompt llama-cli banner").  The 2026-09-12 dev-box A/B below is that non-comparable
        // shape, so it documents what the knob does rather than setting a default: whole-prompt pp,
        // arm at `=1000000000` vs `=0`, r2, interleaved, gfx1151 IQ4_XS f16, b=ub=2048 -
        // pp4096 754.6/740.8 (dense +2.4 %), pp8192 747.3/737.8 (dense +1.6 %),
        // pp16384 712.4/735.1 (sparse +3.0 %), pp32768 604.5/709.5 (sparse +17.4 %).
        // Full data, gates and caveats: wip/strix-halo/qsa-item9/RECORD-2026-09-12-qsa-prefill-crossover.md
        static const int64_t qsa_dense_prefill_until = []() -> int64_t {
            const int64_t env = qsa_env_tokens("LLAMA_QSA_DENSE_PREFILL_UNTIL");
            return env >= 0 ? env : 0;
        }();
        // the dense decode band and the verify width: a speculative verify batch must take the
        // dense decode arm whenever the W=1 decode does (see QSA_DECODE_BAND above).  The widest
        // batch that can be a verify is cparams.n_rs_batch (the rollback bound = the longest
        // enabled draft + 1), so the effective band is at least that wide; it is 1 without
        // speculation, which leaves the plain QSA_DECODE_BAND in charge.
        const int64_t qsa_decode_band = std::max<int64_t>(QSA_DECODE_BAND, (int64_t) cparams.n_rs_batch);
        if (shortcut && n_kv <= width) {
            build_qsa_store_k(mctx_hyb, cur, il);
        } else if (qsa_dense_decode_until > 0 && n_tokens <= qsa_decode_band && n_kv < qsa_dense_decode_until) {
            // arch policy gate: dense decode below the crossover depth (store keys only).
            // qsa_decode_band, not n_tokens == 1: the verify batch must take the same arm as
            // the W=1 decode (see QSA_DECODE_BAND above).
            build_qsa_store_k(mctx_hyb, cur, il);
        } else if (qsa_dense_prefill_until > 0 && n_tokens > qsa_decode_band &&
                   n_kv < qsa_dense_prefill_until) {
            // arch policy, prefill half: dense attend below the prefill crossover (store keys
            // only).  n_tokens > qsa_decode_band keeps this arm disjoint from the decode one, so
            // a W=1 decode and a W=(--spec-draft-n-max + 1) verify of the same state still take
            // the same arm; a prefill chunk is > the band by definition (a chunk <= the band
            // cannot be distinguished from a verify batch - that is the point).
            build_qsa_store_k(mctx_hyb, cur, il);
        } else {
            // the derived path never creates the kq mask; the qsa input's can_reuse below carries
            // the same gate, so a prefill graph (no mask) and a decode graph (mask) never mix
            const bool want_derived_vis = qwen4exp_want_derived_vis(model, hparams, il, cparams, n_tokens);
            top_k = build_qsa_top_k(mctx_hyb, cur, inp_pos,
                    want_derived_vis ? nullptr : inp->get_kq_mask(), sections, il);
        }
    }

    // Qwen3Next uses a single Q projection that outputs query + gate
    ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s); // [ (n_embd_head * 2) * n_head, n_tokens ]
    cb(Qcur_full, "Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "Qcur_reshaped", il);

    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
    cb(Kcur, "Kcur", il);

    ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
    cb(Vcur, "Vcur", il);

    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply IMRoPE
    Qcur = ggml_rope_multi(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    Kcur = ggml_rope_multi(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    if (top_k) {
        cur = build_attn_qsa(inp, Qcur, Kcur, Vcur, top_k, kq_scale, il);
    } else {
        cur = build_attn(inp,
                    nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    }
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = hparams.ssm_d_state;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);
    GGML_ASSERT(head_v_dim * num_v_heads == d_inner);

    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    ggml_tensor * beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(beta, "beta", il);

    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    ggml_tensor * alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    cb(alpha, "alpha", il);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];

    // the channels must match how load_arch_tensors sizes wqkv, not ssm_d_inner
    const int64_t conv_channels    = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;

    ggml_tensor * conv_input = build_conv_state_at(inp, conv_states_all, qkv_mixed,
            conv_kernel_size - 1, conv_channels, il);

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, conv_channels);

    // Extract the convolved Q, K, V from conv_output
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);


    const float eps_norm = hparams.f_norm_rms_eps;

    q_conv = build_gdn_l2_norm(ctx0, q_conv, eps_norm);
    k_conv = build_gdn_l2_norm(ctx0, k_conv, eps_norm);

    // repeat to match shapes when head keys != value keys; unneeded with the fused GDN
    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);

    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // gated normalization, as self.norm(core_attn_out, z) in the reference
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cb(final_output, "final_output", il);

    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    cb(cur, "linear_attn_out", il);

    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    GGML_ASSERT(model.layers[il].ffn_gate_inp != nullptr);

    ggml_tensor * moe_out =
        build_moe_ffn(cur,
            model.layers[il].ffn_gate_inp,
            model.layers[il].ffn_up_exps,
            model.layers[il].ffn_gate_exps,
            model.layers[il].ffn_down_exps,
            nullptr,
            n_expert, n_expert_used,
            LLM_FFN_SILU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
            nullptr, model.layers[il].ffn_gate_up_exps,
            model.layers[il].ffn_up_exps_s,
            model.layers[il].ffn_gate_exps_s,
            model.layers[il].ffn_down_exps_s);
    cb(moe_out, "ffn_moe_out", il);

    // shared experts, as in the Qwen3Next reference
    if (model.layers[il].ffn_up_shexp != nullptr) {
        ggml_tensor * ffn_shexp =
            build_ffn(cur,
                model.layers[il].ffn_up_shexp, NULL, model.layers[il].ffn_up_shexp_s,
                model.layers[il].ffn_gate_shexp, NULL, model.layers[il].ffn_gate_shexp_s,
                model.layers[il].ffn_down_shexp, NULL, model.layers[il].ffn_down_shexp_s,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        // shared expert has its own sigmoided gate (ffn_gate_inp_shexp, one value per token)
        ggml_tensor * shared_gate = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
        cb(shared_gate, "shared_expert_gate", il);

        shared_gate = ggml_sigmoid(ctx0, shared_gate);
        cb(shared_gate, "shared_expert_gate_sigmoid", il);

        ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
        cb(ffn_shexp, "ffn_shexp_gated", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);
    } else {
        cur = moe_out;
    }

    return cur;
}

// PLE n-gram hash embedding: each token gathers ple_n_heads rows of a shared table.
//   mixed_n = (t[p]*m[0]) ^ ... ^ (t[p-n+1]*m[n-1]);  row = mixed_n % vocab[h] + offset[h]
// The hash runs host-side because ggml has no int64 and no xor. EOS resets the window.

class llm_graph_input_ple : public llm_graph_input_i {
public:
    llm_graph_input_ple(const llama_model_qwen4exp & pmodel,
                        const llama_kv_cache_context * mctx) : pmodel(pmodel), mctx(mctx) {}
    virtual ~llm_graph_input_ple() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx)->get_attn();
        if (emb != nullptr) {
            return emb->ne[1] == params.ubatch.n_tokens;
        }
        return rows->ne[0] == (int64_t) pmodel.hparams.ple_n_heads * params.ubatch.n_tokens;
    }

    ggml_tensor * rows = nullptr;   // I32 [ple_n_heads * n_tokens] (device-table path: graph get_rows)
    ggml_tensor * emb  = nullptr;   // F32 [ple_head_dim * n_heads, n_tokens] (host-table path: set_input gathers)

    const llama_model_qwen4exp & pmodel;

    // the predecessor tokens live in the attention KV cells (ext.tok)
    const llama_kv_cache_context * mctx;

    // scratch, reused across set_input() calls
    std::vector<llama_token> prev;
    std::vector<float> emb_scratch;
};

void llm_graph_input_ple::set_input(const llama_ubatch * ubatch) {
    const auto & hp = pmodel.hparams;

    // an image arrives as an embd batch, so ubatch->token is null, but every position still needs a row for ggml_get_rows
    // stand in the image token id that the reference hashes, or EOS if the file has no such key
    // gemma3n and gemma4 do the same with a hardcoded row 0 of per_layer_token_embd.
    const llama_token img_tok = hp.ple_image_token_id != 0
        ? (llama_token) hp.ple_image_token_id
        : (llama_token) hp.ple_eos_token_id;
    auto tok_of = [&](int64_t k) -> llama_token {
        return ubatch->token ? ubatch->token[k] : img_tok;
    };

    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t n_gram   = hp.ple_ngram_size;
    const int64_t n_heads  = hp.ple_n_heads;
    const int64_t per_gram = hp.ple_heads_per_ngram;
    const int64_t eos      = hp.ple_eos_token_id;
    const int64_t n_prev   = n_gram - 1;

    std::vector<int32_t> idx(n_heads * n_tokens);

    GGML_ASSERT(mctx != nullptr);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // the preceding tokens would be ambiguous, see get_prev_tokens()
        GGML_ASSERT(ubatch->n_seq_id[i] == 1 && "PLE n-gram embeddings do not support tokens shared by multiple sequences");
    }

    // predecessors come from the KV cells (ext.tok); apply_ubatch() already stored this ubatch, so its own tokens count too
    mctx->get_prev_tokens(*ubatch, n_prev, prev);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // an EOS in the window resets everything at or before it
        // a missing predecessor (before the sequence start, or no cached cell) reads as EOS
        // the EOS of the token itself does not cut its own context, as in the reference
        std::vector<int64_t> ctx(n_gram);
        ctx[0] = tok_of(i);
        bool cut = false;
        for (int64_t s = 1; s < n_gram; ++s) {
            // predecessor s positions back; prev[] is oldest-first, missing entries are LLAMA_TOKEN_NULL
            const llama_token t = cut ? LLAMA_TOKEN_NULL : prev[i*n_prev + (n_prev - s)];
            cut = cut || t < 0 || t == eos;
            ctx[s] = cut ? eos : t;
        }

        for (int64_t n = 2; n <= n_gram; ++n) {
            uint64_t mixed = (uint64_t) ctx[0] * hp.ple_layer_multipliers[0];
            for (int64_t j = 1; j < n; ++j) {
                mixed ^= (uint64_t) ctx[j] * hp.ple_layer_multipliers[j];
            }
            const int64_t base = (n - 2) * per_gram;
            for (int64_t g = 0; g < per_gram; ++g) {
                const int64_t h_i = base + g;
                idx[i * n_heads + h_i] =
                    (int32_t) (mixed % hp.ple_head_vocab_sizes[h_i] + hp.ple_head_offsets[h_i]);
            }
        }
    }

    if (pmodel.lazy_reader) {
        // managed path: dequantize the needed rows straight into the graph input.
        // gather() lays dst out as dst[(i*n_heads + h)*ple_head_dim + d], which is
        // exactly the flat layout of emb ([ple_head_dim*n_heads, n_tokens])
        emb_scratch.resize(idx.size() * pmodel.hparams.ple_head_dim);
        pmodel.lazy_reader->gather(idx.data(), idx.size(), emb_scratch.data());
        ggml_backend_tensor_set(emb, emb_scratch.data(), 0, emb_scratch.size()*sizeof(float));

        if (debug) {
            fprintf(stderr, "ple managed: %" PRId64 " tokens, cache hits=%" PRIu64 " misses=%" PRIu64 " bytes=%" PRIu64 "\n",
                    n_tokens, pmodel.lazy_reader->hits(), pmodel.lazy_reader->misses(),
                    pmodel.lazy_reader->bytes_read());
        }
    } else if (emb != nullptr) {
        // host-resident table (memory-mapped or in RAM): gather it here, exactly as the
        // CPU get_rows would (same to_float dequant), so that no CPU get_rows node cuts
        // the graph into GPU/CPU/GPU splits with a launch + sync per ubatch. The mmap
        // pages are random-access, so queue every distinct page with one WILLNEED batch
        // instead of taking the faults one after another (~0.2 ms each from NVMe).
        const ggml_tensor * tab = pmodel.per_layer_tok_embd;
        const size_t  row_size = ggml_row_size(tab->type, tab->ne[0]);
        const int64_t dim      = tab->ne[0];
        GGML_ASSERT(dim == hp.ple_head_dim);
        GGML_ASSERT(tab->data != nullptr);

        const auto * traits = ggml_get_type_traits(tab->type);

#if defined(__linux__) || defined(__APPLE__)
        {
            const long page = sysconf(_SC_PAGESIZE);
            if (page > 0) {
                const uintptr_t base = (uintptr_t) tab->data;
                const uintptr_t mask = ~((uintptr_t) page - 1);
                uintptr_t prev_page = 0;
                for (const int32_t r : idx) {
                    const uintptr_t a0 = (base + (size_t) r * row_size) & mask;
                    const uintptr_t a1 = base + (size_t) r * row_size + row_size;
                    if (a0 != prev_page) {
                        madvise((void *) a0, a1 - a0, MADV_WILLNEED);
                        prev_page = a0;
                    }
                }
            }
        }
#endif

        emb_scratch.resize(idx.size() * (size_t) dim);
        for (size_t k = 0; k < idx.size(); ++k) {
            const char * src = (const char *) tab->data + (size_t) idx[k] * row_size;
            float *      dst = emb_scratch.data() + k * (size_t) dim;
            if (tab->type == GGML_TYPE_F32) {
                memcpy(dst, src, dim * sizeof(float));
            } else {
                traits->to_float(src, dst, dim);
            }
        }
        ggml_backend_tensor_set(emb, emb_scratch.data(), 0, emb_scratch.size()*sizeof(float));
    } else {
        ggml_backend_tensor_set(rows, idx.data(), 0, idx.size()*ggml_element_size(rows));
    }
}

// Read a conv history out of its own recurrent row and write the new tail back.
// The shared build_conv_state cannot do this: qwen4exp has two such rows per layer.
ggml_tensor * llama_model_qwen4exp::graph::build_conv_state_at(
        llm_graph_input_rs * inp,
        ggml_tensor *        conv_states_all,
        ggml_tensor *        x,
        int64_t              state_cols,
        int64_t              channels,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const auto kv_head = mctx_cur->get_head();

    const int64_t n_seqs    = ubatch.n_seqs;
    const int64_t row_total = conv_states_all->ne[0];

    // the row is exactly this convolution's state, so the gather is reused as a whole
    GGML_ASSERT(state_cols * channels == row_total);

    auto it = rs_rows.find(conv_states_all);
    if (it == rs_rows.end()) {
        it = rs_rows.emplace(conv_states_all, build_rs(inp, conv_states_all, row_total, n_seqs)).first;
    }
    ggml_tensor * rows = it->second;

    ggml_tensor * state = ggml_reshape_3d(ctx0, rows, state_cols, channels, n_seqs);
    cb(state, "conv_state_at", il);

    ggml_tensor * conv_input = ggml_concat(ctx0, state, ggml_transpose(ctx0, x), 0);

    // [TAG_RECURRENT_ROLLBACK_SPLITS] keep the last state_cols columns once per rollback slot,
    // slot s ending s tokens earlier so a rollback of s tokens reads a history that never saw them
    const size_t row_size = ggml_row_size(conv_states_all->type, row_total);
    const uint32_t mem_size = mctx_cur->get_size();

    const int64_t n_slots = (int64_t) cparams.n_rs_seq + 1;

    for (int64_t slot = 0; slot < n_slots; ++slot) {
        const int64_t s_idx = std::max<int64_t>(0, conv_input->ne[0] - state_cols - slot);

        ggml_tensor * tail = ggml_view_3d(ctx0, conv_input,
                state_cols, channels, n_seqs,
                conv_input->nb[1], conv_input->nb[2],
                ggml_row_size(conv_input->type, s_idx));

        ggml_tensor * dst = ggml_view_2d(ctx0, conv_states_all,
                state_cols * channels, n_seqs,
                conv_states_all->nb[1],
                (slot * mem_size + kv_head) * row_size);

        // copy the strided tail straight into the slot (one copy, a 2D memcpy on CUDA) instead of cont + cpy;
        // LLAMA_CONV_TAIL_CONT=1 restores the intermediate contiguous copy for A/B
        static const bool tail_cont = getenv("LLAMA_CONV_TAIL_CONT") != nullptr && atoi(getenv("LLAMA_CONV_TAIL_CONT")) != 0;
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, tail_cont ? ggml_cont(ctx0, tail) : tail, dst));
    }

    return conv_input;
}

ggml_tensor * llama_model_qwen4exp::graph::build_inp_ple(
        const llama_memory_hybrid_idx_context * mctx_hyb) {
    const int64_t n_heads = hparams.ple_n_heads;

    // the attention cells see every ubatch regardless of the layer types
    auto ple_inp = std::make_unique<llm_graph_input_ple>(
            static_cast<const llama_model_qwen4exp &>(model), mctx_hyb->get_attn());

    const auto & pmodel = static_cast<const llama_model_qwen4exp &>(model);
    if (pmodel.lazy_reader) {
        // managed path: no gather node in the graph; set_input fills this tensor
        // with the dequantized rows (the layout matches the get_rows output below)
        ple_inp->emb = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32,
                                          hparams.ple_head_dim * n_heads, n_tokens);
        ggml_set_input(ple_inp->emb);
        ggml_tensor * emb = ple_inp->emb;
        res->add_input(std::move(ple_inp));
        cb(emb, "ple_embd", -1);
        return emb;
    }

    // A host-resident table (memory-mapped or loaded into RAM) is gathered on the host in
    // set_input too: a get_rows node on the CPU backend would cut the graph into three splits
    // (GPU, CPU, GPU) and, for a big random-access mmap, fault one 4 KB page at a time from
    // disk (the input-layer tables stay CPU-pinned). Only a device-resident table is gathered
    // by a graph get_rows node. LLAMA_QSA_PLE_HOSTGATHER=0 restores the graph get_rows path.
    const char * env_hg = getenv("LLAMA_QSA_PLE_HOSTGATHER");
    const bool host_gather = model.per_layer_tok_embd && model.per_layer_tok_embd->buffer &&
        ggml_backend_buffer_is_host(model.per_layer_tok_embd->buffer) &&
        (env_hg == nullptr || env_hg[0] != '0');
    if (host_gather) {
        ple_inp->emb = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32,
                                          hparams.ple_head_dim * n_heads, n_tokens);
        ggml_set_input(ple_inp->emb);
        ggml_tensor * emb = ple_inp->emb;
        res->add_input(std::move(ple_inp));
        cb(emb, "ple_embd", -1);
        return emb;
    }

    ple_inp->rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_heads * n_tokens);
    ggml_set_input(ple_inp->rows);
    ggml_tensor * rows = ple_inp->rows;
    res->add_input(std::move(ple_inp));

    // gather then flatten the heads: get_rows lays the head dimension out slowest, as the reference does
    ggml_tensor * emb = ggml_get_rows(ctx0, model.per_layer_tok_embd, rows);
    emb = ggml_reshape_2d(ctx0, emb, hparams.ple_head_dim * n_heads, n_tokens);
    cb(emb, "ple_embd", -1);

    return emb;
}

ggml_tensor * llama_model_qwen4exp::graph::build_ple(
        llm_graph_input_rs * inp,
        ggml_tensor *        emb,
        ggml_tensor *        hidden,
        int                  il) {
    const int64_t hc      = hparams.dsv4_hc_mult;
    const int64_t hc_dim  = hc * n_embd;

    ggml_tensor * key   = build_lora_mm(model.layers[il].ple_key,   emb);
    ggml_tensor * value = build_lora_mm(model.layers[il].ple_value, emb);

    // both norms group over one hc stream, with a [n_embd, hc] weight
    auto grouped_norm = [&](ggml_tensor * x, ggml_tensor * w) {
        ggml_tensor * t = ggml_reshape_3d(ctx0, x, n_embd, hc, n_tokens);
        return ggml_mul(ctx0, ggml_rms_norm(ctx0, t, hparams.f_norm_rms_eps), w);
    };

    key = grouped_norm(key, model.layers[il].ple_norm_key);
    ggml_tensor * query = grouped_norm(hidden, model.layers[il].ple_norm_query);

    // per-stream dot product, then a signed square root before the sigmoid
    ggml_tensor * s = ggml_sum_rows(ctx0, ggml_mul(ctx0, key, query));
    s = ggml_scale(ctx0, s, 1.0f / sqrtf((float) n_embd));

    ggml_tensor * mag  = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, s), 1e-6f, INFINITY));
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, ggml_sgn(ctx0, s), mag));
    cb(gate, "ple_gate", il);

    // [n_embd, 1, T] value broadcast across the hc streams, scaled by the gate
    ggml_tensor * v3 = ggml_reshape_3d(ctx0, value, n_embd, 1, n_tokens);
    v3 = ggml_repeat_4d(ctx0, v3, n_embd, hc, n_tokens, 1);

    ggml_tensor * gated = ggml_mul(ctx0, v3, gate);
    cb(gated, "ple_gated_value", il);

    ggml_tensor * normalized = grouped_norm(
            ggml_reshape_2d(ctx0, gated, hc_dim, n_tokens),
            model.layers[il].ple_norm_conv);
    normalized = ggml_reshape_2d(ctx0, normalized, hc_dim, n_tokens);

    // depthwise causal conv, dilated by the n-gram size, as a sum of shifted copies
    // ggml_conv_1d_dw is documented as unreliable:
    //   out[c, t] = sum_k w[k, c] * x[c, t - (K-1-k)*dilation]
    // The history of the earlier ubatches is prepended, so a chunked prefill matches a single-shot one.
    const int64_t kern = hparams.ple_conv_kernel;
    const int64_t dil  = hparams.ple_ngram_size;
    const int64_t hist = (kern - 1) * dil;

    // the conv history is per sequence, so the input carries the sequence axis too
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    // [hist + n_seq_tokens, hc_dim, n_seqs], tokens on ne[0]
    ggml_tensor * padded = build_conv_state_at(inp, inp->mctx->get_p_l(il),
            ggml_reshape_3d(ctx0, normalized, hc_dim, n_seq_tokens, n_seqs),
            hist, hc_dim, il);

    ggml_tensor * conv_out = nullptr;
    for (int64_t k = 0; k < kern; ++k) {
        // tap k reads (kern-1-k)*dilation positions back
        const int64_t start = hist - (kern - 1 - k) * dil;

        ggml_tensor * shifted = ggml_cont(ctx0,
                ggml_transpose(ctx0,
                        ggml_view_3d(ctx0, padded, n_seq_tokens, hc_dim, n_seqs,
                                padded->nb[1], padded->nb[2],
                                ggml_row_size(padded->type, start))));

        // column k of the [kern, hc_dim] kernel is one weight per channel
        ggml_tensor * wk = ggml_cont(ctx0,
                ggml_view_2d(ctx0, model.layers[il].ple_conv1d, 1, hc_dim,
                        model.layers[il].ple_conv1d->nb[1],
                        k * model.layers[il].ple_conv1d->nb[0]));
        // this kernel keeps the file type, so cast it before it multiplies an f32 activation
        wk = ggml_reshape_1d(ctx0, wk, hc_dim);
        if (wk->type != GGML_TYPE_F32) {
            wk = ggml_cast(ctx0, wk, GGML_TYPE_F32);
        }

        ggml_tensor * term = ggml_mul(ctx0, shifted, wk);
        conv_out = conv_out ? ggml_add(ctx0, conv_out, term) : term;
    }

    conv_out = ggml_silu(ctx0, conv_out);
    conv_out = ggml_reshape_3d(ctx0, ggml_cont(ctx0, conv_out), n_embd, hc, n_tokens);
    cb(conv_out, "ple_conv_out", il);

    return ggml_add(ctx0, hidden, ggml_add(ctx0, gated, conv_out));
}
