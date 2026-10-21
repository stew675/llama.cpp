#include "draft-reserve.h"

#include "log.h"

#include "gguf.h"

#include <algorithm>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------------------------
// wip/slab-transient-absorption (step 1): the draft reserve, derived instead of assumed.
//
// The slab holds back `headroom + aux` at creation; the speculative draft context is created afterwards,
// so its VRAM comes out of that reserve (measured: 3 584 MiB on the field config, 4 000 MiB for the same
// draft at `-c 204800`).  The flat constant is exactly the first and short for the second, and the reserve
// cannot be revised once the slab is mapped -- so an UNDER-estimate is what must be avoided (an outside
// `cudaMalloc` aborting is the failure mode; a surplus is reclaimed by `slab_extend`).
//
// Terms, and how each is obtained:
//   weights : the draft GGUF's total tensor bytes, EXACT (mmap mode, all layers on the device).  This is
//             the dominant term (2 647 MiB of the 3 584 measured) and the part that differs between draft
//             models, so it is the piece worth deriving per model.  A draft with some layers on the CPU
//             only over-estimates, which is the safe direction.
//   kv      : `n_ctx` tokens x (k + v) per KV layer, at the REAL element size of the KV types in use, plus
//             any declared indexer key cache at f32 (the QSA/indexer caches are not covered by `-ctk`).
//             KV layers = `nextn_predict_layers` when the GGUF declares an MTP head (the head only ever
//             holds its own layer's KV), else `block_count` for a standalone draft model.
//   compute : the draft context's compute buffer.  NOT derivable without building a graph; measured at
//             512 MiB (`-ub 4096`) and 768 MiB (`-ub 6144`), so the estimate uses a 1 024 MiB tier.
//   slack   : the draft's own load/steady transients, measured at ~135 MiB -> 192 MiB.
//
// Calibration (2026-10-17, gfx1201, `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`, qwen4exp MTP head,
// q8_0 KV): weights 2 647 (GGUF) + kv 300/459 (this function, `-c 131072`/`-c 204800`) + compute 1 024 +
// slack 192 = 4 163 / 4 322 MiB against a MEASURED 3 584 / 4 000.  It over-estimates by ~550 / ~320 MiB:
// the 1 024 MiB compute tier is the biggest single reason (the real buffer is 512-768), and over-estimating
// is the intended bias -- `slab_extend` reclaims a surplus once the draft is resident.
// ---------------------------------------------------------------------------------------------

namespace {

// A conservative f32 element size for a cache whose dtype the GGUF does not pin (the indexer K cache is
// not covered by `-ctk`; treating it as f32 keeps the term an upper bound).
constexpr size_t F32_BYTES = 4;

// Bytes per element of a block-quantised row: keep the fraction (q8_0 is 34/32 = 1.0625, not 2 -- rounding
// the block ratio up to a whole byte doubles the KV term for every quantised cache type).
double type_bytes_per_elem(ggml_type t) {
    const size_t bs = ggml_blck_size(t);
    if (bs == 0) {
        return (double) F32_BYTES;
    }
    return (double) ggml_type_size(t) / (double) bs;
}

// uint-ish scalar or the max of an array (per-layer `head_count_kv` is an array in some models).
bool read_u32_max(const gguf_context * g, const std::string & key, uint32_t & out) {
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0) {
        return false;
    }
    switch (gguf_get_kv_type(g, id)) {
        case GGUF_TYPE_UINT8:  out = gguf_get_val_u8 (g, id); return true;
        case GGUF_TYPE_UINT16: out = gguf_get_val_u16(g, id); return true;
        case GGUF_TYPE_UINT32: out = gguf_get_val_u32(g, id); return true;
        case GGUF_TYPE_UINT64: out = (uint32_t) gguf_get_val_u64(g, id); return true;
        case GGUF_TYPE_INT32:  out = (uint32_t) gguf_get_val_i32(g, id); return true;
        case GGUF_TYPE_INT64:  out = (uint32_t) gguf_get_val_i64(g, id); return true;
        case GGUF_TYPE_ARRAY: {
            const uint32_t n = (uint32_t) gguf_get_arr_n(g, id);
            if (n == 0) {
                return false;
            }
            uint32_t mx = 0;
            switch (gguf_get_arr_type(g, id)) {
                case GGUF_TYPE_UINT32: {
                    const uint32_t * d = (const uint32_t *) gguf_get_arr_data(g, id);
                    for (uint32_t i = 0; i < n; ++i) mx = std::max(mx, d[i]);
                    break;
                }
                case GGUF_TYPE_INT32: {
                    const int32_t * d = (const int32_t *) gguf_get_arr_data(g, id);
                    for (uint32_t i = 0; i < n; ++i) mx = std::max(mx, (uint32_t) d[i]);
                    break;
                }
                default:
                    return false;
            }
            out = mx;
            return true;
        }
        default:
            return false;
    }
}

} // namespace

common_draft_reserve common_draft_reserve_estimate(const std::string & path,
                                                   uint32_t n_ctx,
                                                   uint32_t n_ubatch,
                                                   ggml_type type_k,
                                                   ggml_type type_v) {
    common_draft_reserve r;

    if (path.empty()) {
        r.note = "no draft GGUF (shared / embedded MTP head): the weights are the target's, so this estimate"
                 " does not apply";
        return r;
    }

    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (g == nullptr) {
        r.note = "could not read the draft GGUF metadata";
        return r;
    }

    const int64_t ka = gguf_find_key(g, "general.architecture");
    if (ka < 0) {
        r.note = "draft GGUF has no general.architecture";
        gguf_free(g);
        return r;
    }
    const std::string arch = gguf_get_val_str(g, ka);

    // --- weights: exact, the dominant term -----------------------------------------------------
    size_t weights = 0;
    const int64_t n_tensors = gguf_get_n_tensors(g);
    for (int64_t i = 0; i < n_tensors; ++i) {
        weights += gguf_get_tensor_size(g, i);
    }
    r.weights = weights;

    // `n_ctx == 0` means "from the model": estimating the KV with 0 would UNDER-estimate the reserve (the
    // dangerous direction), so fall back to the context the draft itself declares as its trained maximum.
    if (n_ctx == 0) {
        uint32_t ctx_train = 0;
        if (read_u32_max(g, arch + ".context_length", ctx_train) && ctx_train > 0) {
            n_ctx = ctx_train;
        }
    }

    // --- KV: needs the layer geometry ----------------------------------------------------------
    uint32_t block_count = 0;
    uint32_t n_nextn      = 0;
    (void) read_u32_max(g, arch + ".nextn_predict_layers", n_nextn);   // absent for a standalone draft
    if (!read_u32_max(g, arch + ".block_count", block_count) || block_count == 0) {
        r.note = "draft GGUF has no usable block_count";
        gguf_free(g);
        return r;
    }
    const uint32_t kv_layers = n_nextn > 0 ? n_nextn : block_count;

    uint32_t k_len = 0;
    uint32_t v_len = 0;
    bool have_lens = read_u32_max(g, arch + ".attention.key_length", k_len) &&
                     read_u32_max(g, arch + ".attention.value_length", v_len);
    uint32_t n_head_kv = 1;
    if (!read_u32_max(g, arch + ".attention.head_count_kv", n_head_kv) || n_head_kv == 0) {
        n_head_kv = 1;
    }
    if (!have_lens) {
        // fall back to n_embd / n_head (the per-head width) when the GGUF does not pin them
        uint32_t n_embd = 0, n_head = 0;
        if (!read_u32_max(g, arch + ".embedding_length", n_embd) ||
            !read_u32_max(g, arch + ".attention.head_count", n_head) || n_head == 0) {
            r.note = "draft GGUF does not declare attention key/value lengths";
            gguf_free(g);
            return r;
        }
        k_len = n_embd / n_head;
        v_len = k_len;
    }

    // declared indexer key cache (QSA / DFlash-style sparse attention): one more k row per layer
    uint32_t idx_len = 0, idx_heads = 0;
    const bool have_indexer = read_u32_max(g, arch + ".attention.indexer.key_length", idx_len) &&
                             read_u32_max(g, arch + ".attention.indexer.head_count", idx_heads);

    // KV layers the draft OWNS.  A `shared_kv_layers` draft (gemma4-assistant) does not allocate its own K/V at
    // all: llama_kv_cache's layer_share_cb reuses the TARGET's tensors (`layers.push_back(layer_share)`), and
    // kv_size is overridden to the source cache's.  Charging the full KV for it reserved 18 022 MiB for a
    // drafter that allocates 0 -- at the default 262144 context that is what starved the MoE arena to
    // 13 752 MiB / 4564 slots (h=0.897, 41.0 t/s against 93.1 with the context pinned).
    uint32_t n_shared = 0;
    (void) read_u32_max(g, arch + ".attention.shared_kv_layers", n_shared);
    const bool shares_kv = arch == "gemma4-assistant" || n_shared >= kv_layers;
    const uint32_t kv_own = shares_kv ? 0 : kv_layers - n_shared;

    gguf_free(g);

    const double attn_per_tok = (double) kv_own * (double) n_head_kv *
                                ((double) k_len * type_bytes_per_elem(type_k) +
                                 (double) v_len * type_bytes_per_elem(type_v));
    const double idx_per_tok  = have_indexer
                              ? (double) kv_own * (double) idx_heads * (double) idx_len * (double) F32_BYTES
                              : 0.0;
    // +10 % for per-row block padding and the KQ-mask padding llama.cpp adds to the cell count.
    r.kv = (size_t) ((double) n_ctx * (attn_per_tok + idx_per_tok) * 1.10);

    // --- compute + slack: measured tiers -------------------------------------------------------
    // The draft context's compute buffer (pinned into the slab's narrow-2 region, and its mapping still comes
    // out of the OUTSIDE pool) measured 512 MiB at the target's `-ub 4096` and 768 MiB at `-ub 6144`.  That
    // is the one term that genuinely scales with the target's ubatch, so it is keyed on it (the wide tier is
    // used above 4096); it is a measured tier, not a derivation.
    r.compute = (n_ubatch > 4096 ? (size_t) 768 : (size_t) 512) << 20;
    r.slack   = (size_t) 192 << 20;   // measured ~135

    r.total = r.weights + r.kv + r.compute + r.slack;
    r.ok    = r.total > 0;

    LOG_INF("%s: draft reserve estimate for '%s': weights %.0f + kv %.0f (%u x %u layers, %u kv heads,"
            " %u/%u head dims, indexer %s%s) + compute %.0f + slack %.0f = %.0f MiB (n_ctx %u)\n",
            __func__, path.c_str(),
            (double) r.weights / 1048576.0, (double) r.kv / 1048576.0, kv_own, kv_layers, n_head_kv,
            k_len, v_len, have_indexer ? "yes" : "no",
            shares_kv ? ", KV shared with the target (0 owned)" : "",
            (double) r.compute / 1048576.0, (double) r.slack / 1048576.0, (double) r.total / 1048576.0,
            n_ctx);

    return r;
}
