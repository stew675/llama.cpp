#include "llama-memory-hybrid-idx.h"

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-model.h"


#include <algorithm>
#include <cassert>
#include <cmath>
#include <iterator>
#include <stdexcept>

//
// llama_memory_hybrid_idx
//

// The QSA indexer scores blocks of keys and never reads a stored value, so its cache is created
// keys-only by default: no V tensor exists, no V-side op may be issued against it.  That saves
// -637 MiB of VRAM at ctx 204800 (the store is triplicated across the three GPUs);
// LLAMA_QSA_KEYS_ONLY=0 restores the (dead) V buffer for A/B.
static bool qwen4exp_keys_only_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("LLAMA_QSA_KEYS_ONLY");
        return env == nullptr || std::atoi(env) != 0;
    }();
    return enabled;
}

// A qwen4exp MTP context is built with a recurrent filter that matches nothing (the nextn layer is
// not recurrent): it carries the indexer cache for sparse draft attention but has no recurrent
// layers.  An empty recurrent cache still refuses a partial seq_rm (llama_memory_recurrent::seq_rm's
// per-token rollback path needs a state snapshot that was never written), which would abort the
// server on the first speculative cache trim, so the recurrent child is skipped whenever no layer
// bound a state tensor.
static bool hybrid_idx_no_recr(const llama_memory_recurrent * r) {
    if (!r) {
        return true;
    }
    for (ggml_tensor * t : r->r_l) { if (t) { return false; } }
    for (ggml_tensor * t : r->s_l) { if (t) { return false; } }
    return true;
}

llama_memory_hybrid_idx::llama_memory_hybrid_idx(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                 uint32_t   n_rs_batch,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
    const layer_filter_cb & filter_idx) :
    llama_memory_hybrid(
        model,
        type_k, type_v, v_trans, kv_size, n_pad, n_swa, swa_type,
        type_r, type_s, rs_size,
        n_seq_max, n_rs_seq, n_rs_batch, offload, unified,
        filter_attn, filter_recr),
    hparams_idx(model.hparams),
    mem_idx(filter_idx == nullptr ? nullptr : [&] {
        // MQA with a single key head of indexer_head_size, as llama_kv_cache_dsa shapes its own
        std::fill(hparams_idx.n_head_kv_arr.begin(), hparams_idx.n_head_kv_arr.end(), 1);
        // a k-pool indexer caches its per-token rows and the pooled key side by side
        // (glm5-next: key | gate | pooled, qwen4exp: key | pooled)
        hparams_idx.n_embd_head_k_full = model.hparams.indexer_head_size * (model.hparams.indexer_kpool > 0 ? model.hparams.indexer_kpool_row : 1);

        // the cached indexer keys are raw, rotation happens after pooling at read time, so a
        // K-shift must not rotate them while the stream copies in the same update still apply
        hparams_idx.rope_type = LLAMA_ROPE_TYPE_NONE;

        // fool llama_kv_cache into thinking this is a MLA cache, so it won't cache V tensors
        hparams_idx.n_embd_head_k_mla_impl = model.hparams.indexer_head_size;
        hparams_idx.n_embd_head_v_mla_impl = model.hparams.indexer_head_size;

        LLAMA_LOG_INFO("%s: creating indexer KV cache, size = %u cells\n", __func__, kv_size);

        // the QSA indexer scores blocks of keys and never reads a stored value, so its cache is
        // created keys-only (LLAMA_QSA_KEYS_ONLY=0 restores the dead V buffer for A/B)
        return new llama_kv_cache(
            model, hparams_idx, type_k, type_v, v_trans, offload, unified,
            kv_size, n_seq_max, n_pad, n_swa, swa_type,
            nullptr, filter_idx, nullptr, nullptr, "idx_", /* v_enabled */ !qwen4exp_keys_only_enabled());
    }()) {
    // incremental block-vector cache (default ON; GGML_CUDA_QSA_INDEXER_CACHE=0 disables)
    //
    // The graph side (build_qsa_top_k) has always defaulted its idx_cache to ON, so the pool is the
    // only piece that was off - and with it off the fused decode score re-pools/rotates/scores the
    // WHOLE raw indexer cache every step.  Measured on gfx1151 qwen4exp IQ4_NL f16, plain (non-MTP)
    // decode with the pool ON vs OFF: pp80K **24.3 -> 26.5 t/s (+9.1 %)** and pp150K **20.6 ->
    // 23.6 t/s (+14.6 %)**, byte-identical text at both (6e1e56284fc9 / 1e871dd70a62).  The pool is
    // only allocated for the fused decode path (F32/BF16/F16 indexer keys) - see pool_create.
    const char * env_cache = getenv("GGML_CUDA_QSA_INDEXER_CACHE");
    derived_enabled = env_cache == nullptr || std::atoi(env_cache) != 0;

    if (mem_idx) {
        pool_create(model, filter_idx);
    }
}

void llama_memory_hybrid_idx::pool_create(
        const llama_model & model,
        const layer_filter_cb & filter_idx) {
    const llama_hparams & hparams = model.hparams;

    if (hparams.indexer_head_size == 0) {
        // no QSA geometry (should not happen when the indexer cache exists)
        return;
    }

    // the pool backs the fused INDEXER_FILL -> INDEXER_SCORE derived decode path only, and
    // only when that path can actually run:
    //  - the fused ops read the raw indexer keys natively (F32/BF16/F16 only - quantized
    //    indexer-key caches, e.g. --cache-type-k q8_0, always take the per-op chain), and
    //  - the memory-layer derived cache is engaged (GGML_CUDA_QSA_INDEXER_CACHE, default ON;
    //    =0 restores the raw-cache re-pool).  With it off, qsa_derived_limits emits an empty fill
    //    range every step and the pool is never written or read (each decode step would launch an
    //    empty fill for nothing).
    // Skip the allocation otherwise - get_pool() then returns nullptr and build_qsa_top_k
    // runs the fused score with no pool (pools the raw cache: the same F32 arithmetic and
    // byte-identical output, minus the dead buffer) or the per-op chain for quantized keys.
    if (!derived_enabled || (mem_idx->type_k() != GGML_TYPE_F32 &&
                             mem_idx->type_k() != GGML_TYPE_BF16 &&
                             mem_idx->type_k() != GGML_TYPE_F16)) {
        LLAMA_LOG_INFO("%s: derived indexer cache pool skipped (%s keys, derived cache %s)\n", __func__,
                ggml_type_name(mem_idx->type_k()), derived_enabled ? "enabled" : "disabled");
        return;
    }

    // guard on ANY indexer layer carrying a ratio: the QSA ratio starts above layer 0
    bool any_ratio = false;
    for (uint32_t il = 0; il < hparams.n_layer_all && !any_ratio; ++il) {
        if (filter_idx && filter_idx(il) && hparams.dsv4_compress_ratios[il] > 0) {
            any_ratio = true;
        }
    }
    if (!any_ratio) {
        return;
    }

    const uint32_t idx_dim  = hparams.indexer_head_size;
    const uint32_t kv_size  = mem_idx->get_size();
    const uint32_t n_stream = mem_idx->get_n_stream();
    const uint32_t n_layer  = hparams.n_layer_all;

    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };
    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ size_t(2u*(1 + n_stream)*n_layer*ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };
            ggml_context * ctx = ggml_init(params);
            if (ctx == nullptr) {
                throw std::runtime_error("failed to create ggml context for the derived indexer cache");
            }
            ctx_map.emplace(buft, ggml_context_ptr(ctx));
            return ctx;
        }
        return it->second.get();
    };

    for (uint32_t il = 0; il < n_layer; ++il) {
        if (!filter_idx || !filter_idx(il)) {
            continue;
        }

        const uint32_t ratio = hparams.dsv4_compress_ratios[il];
        if (ratio == 0) {
            continue;
        }

        ggml_backend_dev_t dev = model.dev_layer((int) il);
        ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
        ggml_context * ctx = ctx_for_buft(buft);

        const uint32_t n_blocks = (kv_size + ratio - 1)/ratio;
        ggml_tensor * pool = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, idx_dim, n_blocks, n_stream);
        ggml_format_name(pool, "cache_idx_pool_l%u", il);

        pool_layers.push_back({ il, ratio, pool });
    }

    // allocate the per-buft contexts and clear the buffers (no NaNs in the padding)
    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for the derived indexer cache");
        }
        ggml_backend_buffer_clear(buf, 0);
        pool_ctxs.push_back(std::move(ctx));
        pool_bufs.emplace_back(buf);
    }

    // one watermark per (pool layer, stream); layers sharing a ratio share the cell occupancy so
    // their watermarks advance together (qsa_derived_limits writes the whole ratio group)
    pool_wm.assign(pool_layers.size() * n_stream, 0);

    LLAMA_LOG_INFO("%s: derived indexer cache (pool) = %zu layers x %u dims x %u streams, %s\n", __func__,
            pool_layers.size(), idx_dim, n_stream, derived_enabled ? "ENABLED" : "disabled");
}

void llama_memory_hybrid_idx::pool_invalidate_all() {
    // rows above the watermark are never read, so dropping it to zero IS the invalidation
    std::fill(pool_wm.begin(), pool_wm.end(), 0);
}

ggml_tensor * llama_memory_hybrid_idx::get_pool(ggml_context * ctx, int32_t il, uint32_t n_blocks) const {
    for (const auto & pl : pool_layers) {
        if ((int32_t) pl.il == il) {
            GGML_ASSERT(n_blocks <= pl.pool->ne[1]);
            return ggml_view_3d(ctx, pl.pool,
                    pl.pool->ne[0], n_blocks, pl.pool->ne[2],
                    pl.pool->nb[1], pl.pool->nb[2], 0);
        }
    }
    return nullptr;
}

void llama_memory_hybrid_idx::qsa_derived_limits(
        int32_t * dst_fill_from, int32_t * dst_limit, int n_stream, uint32_t ratio,
        const uint32_t * n_bid, bool advance) const {
    // the indexer cells are shared across the layers, so n_bid per stream is the same for every
    // pool layer of this ratio group; write + advance the whole group's watermarks together
    for (size_t i = 0; i < pool_layers.size(); ++i) {
        if (pool_layers[i].ratio != ratio) {
            continue;
        }
        for (int s = 0; s < n_stream; ++s) {
            // without advance the fill op does not run, so the score must not read the pool
            const uint32_t lim = advance ? n_bid[s] : 0;  // full blocks this step (host grouping)
            const uint32_t from = advance ? pool_wm[i*n_stream + s] : 0;
            dst_fill_from[s] = (int32_t) std::min(from, lim);
            dst_limit[s]     = (int32_t) lim;
            if (advance) {
                // the fill op of this step writes [from, lim), so the watermark follows to lim
                pool_wm[i*n_stream + s] = lim;
            }
        }
    }
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    // note: repeats llama_memory_hybrid::init_batch, as the indexer needs the attention slot infos that the base context hides
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (get_mem_attn()->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = get_mem_recr()->n_rs_seq;

                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!hybrid_idx_no_recr(get_mem_recr()) && !get_mem_recr()->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = get_mem_attn()->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // the indexer uses the attention cache's slot layout; a separate one can drift from it
        llama_kv_cache::slot_info_vec_t heads_idx;
        if (mem_idx) {
            heads_idx = heads_attn;
        }

        return std::make_unique<llama_memory_hybrid_idx_context>(
                this, std::move(heads_attn), std::move(heads_idx), std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_full() {
    return std::make_unique<llama_memory_hybrid_idx_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_idx_context>(this, lctx, optimize);
}

void llama_memory_hybrid_idx::clear(bool data) {
    llama_memory_hybrid::clear(data);

    if (mem_idx) {
        mem_idx->clear(data);
        mem_idx_stale_set(-1, 0);
    }

    // the derived vectors derive from the raw cells, so any cache mutation invalidates them
    pool_invalidate_all();
}

// A pooled key is only valid while the grouping that produced it holds. Grouping is sequence relative,
// so an edit at p0 leaves every pool that ends before p0 alone.
void llama_memory_hybrid_idx::mem_idx_stale_set(llama_seq_id seq_id, llama_pos p0) {
    p0 = std::max<llama_pos>(p0, 0);

    if (seq_id < 0) {
        for (auto & p : mem_idx_stale) {
            p = std::min(p, p0);
        }

        return;
    }

    GGML_ASSERT(seq_id < (llama_seq_id) LLAMA_MAX_SEQ);

    mem_idx_stale[seq_id] = std::min(mem_idx_stale[seq_id], p0);
}

// An edit at or below the first position moves pos_min, which regroups the whole sequence.
llama_pos llama_memory_hybrid_idx::mem_idx_stale_pos(llama_seq_id seq_id, llama_pos p0) const {
    if (seq_id < 0 || p0 <= mem_idx->seq_pos_min(seq_id)) {
        return 0;
    }

    return p0;
}

bool llama_memory_hybrid_idx::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // same order as llama_memory_hybrid::seq_rm: the recurrent cache can refuse, so try it first
    if (!hybrid_idx_no_recr(get_mem_recr()) && !get_mem_recr()->seq_rm(seq_id, p0, p1)) {
        return false;
    }

    if (mem_idx) {
        const llama_pos stale = mem_idx_stale_pos(seq_id, p0);
        mem_idx->seq_rm(seq_id, p0, p1);
        mem_idx_stale_set(seq_id, stale);

        // removing a sequence can free cells another sequence shared, but only this one is marked stale, so the
        // survivor would keep shared = true and pin cache_safe off forever; stale every sequence to re-derive it
        if (kpool_layout_shared()) {
            mem_idx_stale_set(-1, 0);
        }
    }

    // the raw cells of the removed range are gone, so the derived rows over them are stale
    pool_invalidate_all();

    return get_mem_attn()->seq_rm(seq_id, p0, p1);
}

void llama_memory_hybrid_idx::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    llama_memory_hybrid::seq_cp(seq_id_src, seq_id_dst, p0, p1);

    if (mem_idx) {
        mem_idx->seq_cp(seq_id_src, seq_id_dst, p0, p1);
        // the copy shares cells, which cannot hold two groupings, so both sides drop their cached keys
        mem_idx_stale_set(seq_id_src, 0);
        mem_idx_stale_set(seq_id_dst, 0);
    }

    pool_invalidate_all();
}

void llama_memory_hybrid_idx::seq_keep(llama_seq_id seq_id) {
    llama_memory_hybrid::seq_keep(seq_id);

    if (mem_idx) {
        mem_idx->seq_keep(seq_id);
        // cells shared with the dropped sequences become exclusive again, their keys were never cached
        mem_idx_stale_set(-1, 0);
    }

    pool_invalidate_all();
}

void llama_memory_hybrid_idx::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    llama_memory_hybrid::seq_add(seq_id, p0, p1, shift);

    if (mem_idx) {
        // a negative shift moves the cells below p0, so they regroup as well
        const llama_pos stale = mem_idx_stale_pos(seq_id, shift < 0 ? p0 + shift : p0);
        mem_idx->seq_add(seq_id, p0, p1, shift);
        mem_idx_stale_set(seq_id, stale);
    }

    pool_invalidate_all();
}

void llama_memory_hybrid_idx::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    llama_memory_hybrid::seq_div(seq_id, p0, p1, d);

    if (mem_idx) {
        mem_idx->seq_div(seq_id, p0, p1, d);
        mem_idx_stale_set(seq_id, 0);
    }

    pool_invalidate_all();
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_idx::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = llama_memory_hybrid::memory_breakdown();

    if (mem_idx) {
        for (const auto & buft_size : mem_idx->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }

    return mb;
}

void llama_memory_hybrid_idx::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    // mirrors llama_memory_hybrid::state_write with the recurrent child skipped when it is empty
    // (the MTP context's recurrent filter matches nothing - see hybrid_idx_no_recr)
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        get_mem_attn()->state_write(io, seq_id, flags);
    }
    if (!hybrid_idx_no_recr(get_mem_recr())) {
        get_mem_recr()->state_write(io, seq_id, flags);
    }

    // [TAG_HYBRID_IDX_STATE] the indexer section goes last, so it is a pure suffix: an old reader stops early instead of misparsing it
    // The indexer mirrors the attention cache, so it uses the same PARTIAL_ONLY gate.
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        if (mem_idx) {
            mem_idx->state_write(io, seq_id, flags);
        }
    }

}

void llama_memory_hybrid_idx::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // note: repeats llama_memory_hybrid::state_read
    // the indexer needs the attention cache's cells, and a half-failed restore must leave all three caches alike

    // [TAG_HYBRID_IDX_SINFO]
    // the indexer restore adopts the attention cache's layout instead of searching for cells of its own
    // two find_slot calls agree only while both caches see the same occupancy, which a restore cannot promise
    llama_kv_cache::slot_info_vec_t sinfos_attn;

    try {
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            get_mem_attn()->state_read_sinfo(io, seq_id, flags, mem_idx ? &sinfos_attn : nullptr, nullptr);
        }

        if (!hybrid_idx_no_recr(get_mem_recr())) {
            get_mem_recr()->state_read(io, seq_id, flags);
        }

        // [TAG_HYBRID_IDX_STATE] must mirror the write order in state_write
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            if (mem_idx) {
                mem_idx->state_read_sinfo(io, seq_id, flags, nullptr, &sinfos_attn);
                // the restore rewrites the cells behind the pool layout's back
                mem_idx_stale_set(seq_id, 0);
                // it can also change which cells are shared; re-derive sharing for every sequence, as seq_rm does
                if (kpool_layout_shared()) {
                    mem_idx_stale_set(-1, 0);
                }
            }
        }

    } catch (...) {
        // a half-restored context is the one state the indexer cannot fix by itself: attention holds new cells, the indexer old ones
        // drop what was being restored from all of them, which is a state they do agree on.
        state_drop(seq_id);

        throw;
    }
}

void llama_memory_hybrid_idx::state_drop(llama_seq_id seq_id) {
    // dropped directly, not via seq_rm: the recurrent cache may refuse it and then only the other two get cleared
    if (seq_id < 0) {
        clear(true);

        return;
    }

    get_mem_attn()->state_clear(seq_id);
    if (!hybrid_idx_no_recr(get_mem_recr())) {
        get_mem_recr()->seq_rm(seq_id, -1, -1);
    }

    if (mem_idx) {
        mem_idx->state_clear(seq_id);
        mem_idx_stale_set(seq_id, 0);
        // clearing this sequence can end a sharing the survivor would otherwise keep flagged (see seq_rm)
        if (kpool_layout_shared()) {
            mem_idx_stale_set(-1, 0);
        }
    }

    pool_invalidate_all();
}

llama_kv_cache * llama_memory_hybrid_idx::get_mem_idx() const {
    return mem_idx.get();
}

void llama_memory_hybrid_idx::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        ggml_tensor * blk_idx,
        ggml_tensor * blk_tail,
        ggml_tensor * cell_vis,
        ggml_tensor * q_vis,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias,
        int32_t * dst_derived_from,
        int32_t * dst_derived_lim) const {
    GGML_ASSERT(ratio > 0);
    GGML_ASSERT(get_mem_idx() != nullptr);

    GGML_ASSERT(ggml_backend_buffer_is_host(cell_blk->buffer));

    const int64_t n_kv     = cell_blk->ne[0];
    const int64_t n_ns     = cell_blk->ne[1];        // streams in this ubatch
    const int64_t n_blocks = blk_pos->ne[0]/(4*n_ns);
    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t r        = ratio;

    GGML_ASSERT(n_tokens % n_ns == 0);
    const int64_t n_tps = n_tokens/n_ns;             // tokens per stream

    int32_t * dst_cell_blk  = (int32_t *) cell_blk->data;
    int32_t * dst_blk_cells = (int32_t *) blk_cells->data;
    int32_t * dst_blk_pos   = (int32_t *) blk_pos->data;
    float   * dst_bias      = bias     != nullptr ? (float   *) bias->data     : nullptr;
    int32_t * dst_blk_idx   = blk_idx  != nullptr ? (int32_t *) blk_idx->data  : nullptr;
    int32_t * dst_blk_tail  = blk_tail != nullptr ? (int32_t *) blk_tail->data : nullptr;
    int32_t * dst_cell_vis  = cell_vis != nullptr ? (int32_t *) cell_vis->data : nullptr;
    int32_t * dst_q_vis     = q_vis    != nullptr ? (int32_t *) q_vis->data    : nullptr;

    // exactly one form of the per-block bias is asked for: the uploaded tensor, or the compact
    // pair the top-k derives it from
    GGML_ASSERT((dst_blk_idx != nullptr) == (dst_blk_tail != nullptr));
    GGML_ASSERT(dst_blk_idx == nullptr || blk_bias);
    GGML_ASSERT(dst_bias != nullptr || dst_blk_idx != nullptr);
    GGML_ASSERT(dst_bias == nullptr || dst_blk_idx == nullptr);

    // the derived visibility travels with blk_bias (it is the per-cell half of the same bias)
    GGML_ASSERT((dst_cell_vis != nullptr) == (dst_q_vis != nullptr));
    GGML_ASSERT(dst_cell_vis == nullptr || blk_bias);

    if (dst_blk_idx != nullptr) {
        GGML_ASSERT(blk_idx->type  == GGML_TYPE_I32 && blk_idx->ne[0]  == n_blocks && blk_idx->ne[1]  == n_ns);
        GGML_ASSERT(blk_tail->type == GGML_TYPE_I32 && blk_tail->ne[0] == n_tps    && blk_tail->ne[1] == n_ns);
    }

    if (dst_cell_vis != nullptr) {
        GGML_ASSERT(cell_vis->type == GGML_TYPE_I32 && cell_vis->ne[0] == n_kv && cell_vis->ne[1] == n_ns);
        GGML_ASSERT(q_vis->type    == GGML_TYPE_I32 && q_vis->ne[0]    == n_tps && q_vis->ne[1]    == n_ns);
    }

    // a block is keyed on (sequence set, index bucket): a unified cache counts every sequence
    // from zero, so the bucket alone would pool two sequences into one block
    GGML_ASSERT(r <= 64);
    const uint64_t slots_full = r == 64 ? ~uint64_t(0) : ((uint64_t(1) << r) - 1);

    // TODO: this runs per ubatch and is O(n_kv) per stream, about 865 us at 33k context. the cost
    //       is the per-cell scan rather than these allocations, so hoisting them buys nothing
    std::vector<int32_t>  blk_of(n_kv);
    std::vector<int32_t>  cell_grp(n_kv);
    std::vector<uint32_t> n_bid_s(n_ns);
    std::vector<int32_t>  grp_head(n_blocks);
    std::vector<int32_t>  grp_next;
    std::vector<int32_t>  grp_first;
    std::vector<int32_t>  grp_slot0;
    std::vector<uint64_t> grp_slots;
    std::vector<int32_t>  grp_bid;
    std::vector<int32_t>  bid_idx;
    std::vector<int32_t>  bid_cell;
    std::vector<int32_t>  bid_slot0;

    std::vector<int32_t> order;
    std::vector<int32_t> rank;

    std::fill(dst_blk_pos, dst_blk_pos + 4*n_blocks*n_ns, 0);

    for (int64_t s = 0; s < n_ns; ++s) {
        // ubatch index s*n_tps belongs to this stream; ask which cells array it uses
        const llama_seq_id seq_of_stream = ubatch->seq_id[s*n_tps][0];
        const auto & cells = get_mem_idx()->get_cells(seq_of_stream);

        int32_t * cur_cell_blk  = dst_cell_blk  + s*n_kv;
        int32_t * cur_blk_cells = dst_blk_cells + s*(r*n_blocks);

        std::fill(cur_blk_cells, cur_blk_cells + r*n_blocks, 0);

        bid_idx  .clear();
        bid_cell .clear();
        bid_slot0.clear();

        int n_seq_present = 0;

        for (int sq = 0; sq < LLAMA_MAX_SEQ && n_seq_present < 2; ++sq) {
            if (cells.seq_pos_min(sq) >= 0) {
                n_seq_present++;
            }
        }

        const bool one_seq = n_seq_present <= 1;

        // a cell no block covers needs its own -inf, which a per-block bias cannot carry
        // every cache path keeps the position below the cell window, so this stays false
        bool oor = false;

        bool dup = false;

        bool ranked = false;

        auto group_cells = [&]() {
            // -1 means no usable block: an incomplete or short group cannot be pooled
            std::fill(blk_of.begin(),   blk_of.end(),   -1);
            std::fill(cell_grp.begin(), cell_grp.end(), -1);
            std::fill(grp_head.begin(), grp_head.end(), -1);

            grp_next .clear();
            grp_first.clear();
            grp_slot0.clear();
            grp_slots.clear();
            grp_bid  .clear();

            oor = false;
            dup = false;

            for (int64_t j = 0; j < n_kv; ++j) {
                if (cells.is_empty(j)) {
                    continue;
                }

                const int64_t idx = ranked ? rank[j] : cells.pos_get(j);
                const int64_t pb  = idx/r;

                if (pb >= n_blocks) {
                    oor = true;
                    continue;
                }

                int32_t g = -1;

                for (int32_t c = grp_head[pb]; c >= 0; c = grp_next[c]) {
                    if (one_seq || cells.seq_get_all((uint32_t) grp_first[c]) == cells.seq_get_all((uint32_t) j)) {
                        g = c;
                        break;
                    }
                }

                if (g < 0) {
                    g = (int32_t) grp_first.size();

                    grp_next .push_back(grp_head[pb]);
                    grp_first.push_back((int32_t) j);
                    grp_slot0.push_back(-1);
                    grp_slots.push_back(0);
                    grp_bid  .push_back(-1);

                    grp_head[pb] = g;
                }

                const uint64_t bit = uint64_t(1) << (idx%r);

                dup |= (grp_slots[g] & bit) != 0;

                cell_grp[j]   = g;
                grp_slots[g] |= bit;

                if (idx%r == 0) {
                    grp_slot0[g] = (int32_t) j;
                }
            }
        };

        group_cells();

        // mrope repeats one position across an image, so rank cells instead of using the position
        if (dup && ubatch->is_pos_2d() && one_seq) {
            order.clear();
            order.reserve(n_kv);

            for (int64_t j = 0; j < n_kv; ++j) {
                if (!cells.is_empty(j)) {
                    order.push_back((int32_t) j);
                }
            }

            // same total order the mrope causal mask uses: pos, then ext.y, then ext.x
            std::sort(order.begin(), order.end(), [&cells](int32_t a, int32_t b) {
                const llama_pos pa = cells.pos_get(a);
                const llama_pos pb = cells.pos_get(b);

                if (pa != pb) {
                    return pa < pb;
                }

                const auto & ea = cells.ext_get(a);

                return cells.ext_get(b).is_2d_gt(ea.x, ea.y);
            });

            rank.assign(n_kv, -1);

            for (int64_t k = 0; k < (int64_t) order.size(); ++k) {
                rank[order[k]] = (int32_t) k;
            }

            ranked = true;

            group_cells();
        }

        // the per-cell half of the visibility as state: the compaction key of every cell of
        // this stream's cache, -1 for a cell that is empty or owned by another sequence.
        // a query token with key q_vis sees exactly the cells with 0 <= key <= q_vis, which is
        // set_input_kq_mask_impl's predicate without its causal part - the same comparison the
        // per-cell bias above already makes (idx = ranked ? rank[j] : pos_get(j) against q).
        if (dst_cell_vis != nullptr) {
            int32_t * cur_cell_vis = dst_cell_vis + s*n_kv;

            for (int64_t j = 0; j < n_kv; ++j) {
                cur_cell_vis[j] = cells.is_empty(j) || !cells.seq_has((uint32_t) j, seq_of_stream)
                        ? -1 : (int32_t) (ranked ? rank[j] : cells.pos_get(j));
            }
        }

        GGML_ASSERT((!blk_bias || !oor) && "qsa: cell position runs past the cell window");

        int32_t n_bid = 0;

        for (int64_t pb = 0; pb < n_blocks; ++pb) {
            for (int32_t g = grp_head[pb]; g >= 0; g = grp_next[g]) {
                if (grp_slots[g] != slots_full) {
                    continue;
                }

                grp_bid[g] = n_bid++;

                bid_idx  .push_back((int32_t) (pb*r));
                bid_cell .push_back(grp_first[g]);
                bid_slot0.push_back(grp_slot0[g]);
            }
        }

        GGML_ASSERT(n_bid <= n_blocks);
        n_bid_s[s] = (uint32_t) n_bid;

        for (int32_t b = 0; b < n_bid; ++b) {
            int32_t sec_pos[4] = { bid_idx[b], bid_idx[b], bid_idx[b], bid_idx[b] };

            if (ranked) {
                const int32_t   c = bid_slot0[b];
                const llama_pos p = cells.pos_get(c);
                const auto &    e = cells.ext_get(c);

                sec_pos[0] = p;
                sec_pos[1] = e.y;
                sec_pos[2] = e.x;
                sec_pos[3] = p;
            }

            for (int64_t sec = 0; sec < 4; ++sec) {
                dst_blk_pos[sec*(n_blocks*n_ns) + s*n_blocks + b] = sec_pos[sec];
            }
        }

        // unpooled cells all point at one spare block. a spare block exists only when some
        // cell is unpooled: n_bid == n_blocks means every cell sits in a full block.
        const bool     have_dead = n_bid < n_blocks;
        const int32_t  dead_bid  = have_dead ? n_bid : n_blocks - 1;

        for (int64_t j = 0; j < n_kv; ++j) {
            const int32_t g = cell_grp[j];

            blk_of[j] = g < 0 ? -1 : grp_bid[g];

            if (blk_of[j] >= 0) {
                const int64_t idx = ranked ? rank[j] : cells.pos_get(j);

                cur_blk_cells[blk_of[j]*r + (idx%r)] = (int32_t) j;
            }

            cur_cell_blk[j] = blk_of[j] < 0 ? dead_bid : blk_of[j];
        }

        if (dst_blk_idx != nullptr) {
            // the per-block half of the bias as state: the top-k derives the -inf / 0 / 1e9
            // value from this against the per-token tail start.  the per-sequence half of the
            // original test is deliberately absent - the attention mask (or the derived cell
            // positions) already drops every cell of a foreign block, and -inf + -inf is
            // still -inf, so the values and the selection are identical.
            int32_t * cur_blk_idx = dst_blk_idx + s*n_blocks;

            for (int64_t b = 0; b < n_blocks; ++b) {
                cur_blk_idx[b] = have_dead && b == dead_bid ? INT32_MAX
                               : b >= n_bid                 ? -1
                               :                              bid_idx[b];
            }
        }

        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const int64_t      i      = s*n_tps + ii;
            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            int64_t q = ubatch->pos[i];

            if (ranked) {
                const llama_pos qt = ubatch->pos[i];
                const llama_pos qy = ubatch->pos[i + n_tokens];
                const llama_pos qx = ubatch->pos[i + n_tokens*2];

                int64_t lo = 0;
                int64_t hi = (int64_t) order.size();

                while (lo < hi) {
                    const int64_t   mid = (lo + hi)/2;
                    const int32_t   c   = order[mid];
                    const llama_pos pc  = cells.pos_get(c);

                    if (pc < qt || (pc == qt && !cells.ext_get(c).is_2d_gt(qx, qy))) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }

                q = lo - 1;
            }

            // the tail is an incomplete block and is always visible, as in the reference
            const int64_t tail_start = (q + 1)/r*r;

            if (dst_q_vis != nullptr) {
                dst_q_vis[s*n_tps + ii] = (int32_t) q;
            }

            if (blk_bias) {
                if (dst_blk_tail != nullptr) {
                    // compact form: the tail start is the only per-token part of the block bias
                    dst_blk_tail[i] = (int32_t) tail_start;

                    continue;
                }

                // a block sits wholly inside or outside the tail, so one value covers it
                // the caller adds the attention mask, which drops empty, foreign and future cells
                float * cur_blk_bias = dst_bias + i*n_blocks;

                for (int64_t b = 0; b < n_blocks; ++b) {
                    if (b >= n_bid || !cells.seq_has((uint32_t) bid_cell[b], seq_id)) {
                        cur_blk_bias[b] = -INFINITY;
                        continue;
                    }

                    // finite, so it can never meet a -inf and produce a nan
                    cur_blk_bias[b] = bid_idx[b] >= tail_start ? 1e9f : 0.0f;
                }

                // the spare block holds the unpooled cells, which are the incomplete tail, so
                // it gets the tail value. it must stay finite: a sequence with fewer than
                // `ratio` cells owns no full block, and a row of -inf only gives a nan.
                if (have_dead) {
                    cur_blk_bias[dead_bid] = 1e9f;
                }

                continue;
            }

            float * cur_bias = dst_bias + i*n_kv;

            for (int64_t j = 0; j < n_kv; ++j) {
                float v = -INFINITY;

                if (!cells.is_empty(j) && cells.seq_has(j, seq_id)) {
                    const int64_t idx = ranked ? rank[j] : cells.pos_get(j);

                    if (idx <= q) {
                        // finite, so it can never meet a -inf and produce a nan
                        v = idx >= tail_start ? 1e9f : (blk_of[j] < 0 ? -INFINITY : 0.0f);
                    }
                }

                cur_bias[j] = v;
            }
        }
    }

    // decode-only derived-cache limits: on the single-token append path the fill op of this step
    // writes [wm, n_bid) into the pool rows, so emit the range leaves and advance the watermarks.
    // Any other step emits an empty range (the score then pools raw, as without the derived cache).
    if (dst_derived_from != nullptr && dst_derived_lim != nullptr) {
        const bool advance = derived_enabled && ubatch->n_tokens == 1;
        qsa_derived_limits(dst_derived_from, dst_derived_lim, (int) n_ns, ratio, n_bid_s.data(), advance);
    }
}

//
// llama_memory_hybrid_idx_context
//

// streams in each ubatch's slot info, matching get_k/get_v's `ns`
static std::vector<uint32_t> llama_memory_hybrid_idx_ns(const llama_kv_cache::slot_info_vec_t & sinfos) {
    std::vector<uint32_t> res;
    res.reserve(sinfos.size());

    for (const auto & sinfo : sinfos) {
        res.push_back(sinfo.s1 - sinfo.s0 + 1);
    }

    return res;
}

// Which cells of a sequence make up which pool, for the whole cache.
struct llama_memory_hybrid_idx::kpool_layout {
    struct seq {
        llama_pos pos_min = 0;
        uint32_t  strm    = 0; // Stream holding this sequence's cells
        std::vector<std::pair<llama_pos, uint32_t>> cells; // Position and stream local cell pairs, sorted by position.
        std::vector<uint32_t> pools;

        // Where the pool scan stopped, so an append resumes instead of starting over.
        size_t j_next = 0;

        // Whether any cell also carries another sequence, which rules out caching this sequence's pooled keys.
        bool shared = false;
    };

    std::array<seq, LLAMA_MAX_SEQ> seqs;

    uint32_t n_pool_real = 0;
    bool cache_safe      = true;
};

// Which pools of the layout the current ubatch must re-pool, in the layout's pool order.
struct llama_memory_hybrid_idx_context::kpool_state {
    std::vector<uint32_t> is_new;
    uint32_t generation = 0;

    uint32_t n_pool_real = 0;
    uint32_t n_new       = 0;
    uint32_t n_new_g     = 1; // graph size of the new pool list, stable across decode steps
    bool     cache_safe  = true;
};

namespace {

// The last padded pool is always unused.
uint32_t kpool_pad(uint32_t n_pool) {
    return std::max<uint32_t>(64u, GGML_PAD(n_pool + 1, 64u));
}

// Rank of (pos, cell) in a sequence's cells sorted by position then cell, or -1 when absent.
// In order mode the rank alone places a token: cells sharing a position (M-RoPE images) have distinct ranks.
int64_t kpool_rank(const std::vector<std::pair<llama_pos, uint32_t>> & cells, llama_pos pos, uint32_t cell) {
    auto it = std::lower_bound(cells.begin(), cells.end(), std::make_pair(pos, cell));
    return it != cells.end() && it->second == cell && it->first == pos ? it - cells.begin() : -1;
}

}

llama_memory_hybrid_idx::~llama_memory_hybrid_idx() = default;

const llama_memory_hybrid_idx::kpool_layout & llama_memory_hybrid_idx::kpool_layout_get() const {
    GGML_ASSERT(kpool_lay != nullptr);

    return *kpool_lay;
}

bool llama_memory_hybrid_idx::kpool_layout_shared() const {
    return kpool_lay && !kpool_lay->cache_safe;
}

// Pools are fixed by the positions relative to the sequence's first one, so the layout survives a plain
// append. A sequence edit can regroup them, and mem_idx_stale tells us it happened.
const llama_memory_hybrid_idx::kpool_layout & llama_memory_hybrid_idx::kpool_layout_update() {
    GGML_ASSERT(mem_idx != nullptr);

    if (!kpool_lay) {
        kpool_lay = std::make_unique<kpool_layout>();
    }

    auto & lay = *kpool_lay;

    const uint32_t kpool       = get_kpool();
    const uint32_t n_stream_kv = mem_idx->get_n_stream();
    const bool     unified     = n_stream_kv == 1;

    lay.n_pool_real = 0;
    lay.cache_safe  = true;

    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
        auto & sq = lay.seqs[s];

        // a non unified cache gives each sequence its own stream, with stream local cell indices
        if (!unified && s >= (llama_seq_id) n_stream_kv) {
            sq = kpool_layout::seq();
            continue;
        }

        const auto & cells = mem_idx->get_cells(unified ? 0 : s);
        const auto & sp    = cells.seq_pos_get(s);

        sq.strm = unified ? 0 : mem_idx->get_stream(s);

        size_t n_kept = 0;
        if (mem_idx_stale[s] == POS_CLEAN && !sq.cells.empty() && !sp.empty() &&
                sq.pos_min == sp.begin()->first) {
            n_kept = sq.cells.size();
            for (auto it = sp.upper_bound(sq.cells.back()); it != sp.end(); ++it) {
                sq.cells.push_back(*it);
            }
        }

        // the appended tail accounts for every cell only if nothing before it was dropped, but an edit can
        // regroup a sequence without changing its cell count, so a stale sequence must rebuild regardless
        if (sq.cells.size() != sp.size() || mem_idx_stale[s] != POS_CLEAN) {
            sq.cells.assign(sp.begin(), sp.end());
            sq.pools.clear();
            sq.j_next  = 0;
            sq.shared  = false;
            sq.pos_min = sp.empty() ? 0 : sp.begin()->first;
            n_kept     = 0;
        }

        // sharing starts with a seq_cp; it ends with an edit, or a seq_rm/state_drop/state_read that frees the
        // shared cells - each stales every sequence so the rebuild above re-derives it, so once set it holds
        // until then and the rescan can be skipped
        if (unified && !sq.shared) {
            for (size_t j = n_kept; j < sq.cells.size(); ++j) {
                if (cells.seq_count(sq.cells[j].second) > 1) {
                    sq.shared = true;
                    break;
                }
            }
        }

        // Pools start at the first valid token
        size_t j = sq.j_next;
        if (hparams_idx.indexer_kpool_by_order) {
            // consecutive cells in sequence order, whatever their positions
            for (; j + kpool <= sq.cells.size(); j += kpool) {
                sq.pools.push_back((uint32_t) j);
            }
        } else {
            while (j + kpool <= sq.cells.size()) {
                const llama_pos p0 = sq.cells[j].first;
                if ((p0 - sq.pos_min) % (llama_pos) kpool != 0) {
                    ++j;
                    continue;
                }
                bool ok = true;
                for (uint32_t k = 1; k < kpool; ++k) {
                    if (sq.cells[j + k].first != p0 + (llama_pos) k) {
                        ok = false;
                        break;
                    }
                }
                if (ok) {
                    sq.pools.push_back((uint32_t) j);
                    j += kpool;
                } else {
                    ++j;
                }
            }
        }
        sq.j_next = j;

        lay.n_pool_real += (uint32_t) sq.pools.size();
        lay.cache_safe   = lay.cache_safe && !sq.shared;
    }

    return lay;
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_status status) :
    llama_memory_hybrid_context(status) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem) :
    llama_memory_hybrid_context(mem),
    mem(mem),
    // graph reservation walks a full context, and qwen4exp builds the sparse attention only when this is set
    // without it the reserved worst case is the dense graph, so ggml-alloc must grow the buffer on the first decode
    ns_ubatch(mem->get_mem_idx() == nullptr ?
        std::vector<uint32_t>() : std::vector<uint32_t>{ mem->get_mem_idx()->get_n_stream() }),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx())) {
    if (kpool_track()) {
        mem->kpool_layout_update();
        auto st = kpool_build_sizes();
        const auto * idx = mem->get_mem_idx();
        const uint64_t n_pool_max = uint64_t(idx->get_size() / mem->get_kpool()) * idx->get_n_seq_max();
        GGML_ASSERT(n_pool_max <= UINT32_MAX - 64);
        st.n_pool_real = std::max(st.n_pool_real, uint32_t(n_pool_max));
        st.n_new   = st.n_pool_real;
        st.n_new_g = std::max(st.n_new, 1u);
        kpool_st = std::make_unique<kpool_state>(std::move(st));
        i_kpool  = 0;
    }
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                  llama_context * lctx,
                           bool   optimize) :
    llama_memory_hybrid_context(mem, lctx, optimize),
    mem(mem),
    // update() applies a pending cross-stream seq_cp, else the copy keeps stale indexer keys
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        mem->get_mem_idx()->init_update(lctx, optimize)) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                slot_info_vec_t   sinfos_attn,
                slot_info_vec_t   sinfos_idx,
      std::vector<llama_ubatch>   ubatches) :
    // note: the base copies the ubatches; ctx_idx gets a copy of its own
    llama_memory_hybrid_context(mem, std::move(sinfos_attn), ubatches),
    mem(mem),
    ns_ubatch(llama_memory_hybrid_idx_ns(sinfos_idx)),
    sinfos_kpool(mem->get_mem_idx() != nullptr && mem->get_kpool() > 0 && mem->get_kpool_by_order() ? sinfos_idx : slot_info_vec_t()),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx(), std::move(sinfos_idx), ubatches)) {
    // Sequence edits force the touched positions to re-pool.
    mem_idx_stale_batch = mem->mem_idx_stale_get();
}

llama_memory_hybrid_idx_context::~llama_memory_hybrid_idx_context() = default;

bool llama_memory_hybrid_idx_context::next() {
    // Clear only after a successful ubatch.
    if (i_cur == 0 && mem != nullptr) {
        mem->mem_idx_stale_clear();
    }

    if (ctx_idx) {
        ctx_idx->next();
    }

    ++i_cur;

    return llama_memory_hybrid_context::next();
}

bool llama_memory_hybrid_idx_context::apply() {
    bool res = llama_memory_hybrid_context::apply();

    if (ctx_idx) {
        res = res & ctx_idx->apply();
    }

    // Extend the pool layout with this ubatch's cells, then pick what it must re-pool.
    if (res && kpool_track()) {
        mem->kpool_layout_update();
        if (!kpool_st) {
            kpool_st = std::make_unique<kpool_state>();
        }
        kpool_build_state(get_ubatch());
        i_kpool  = i_cur;
    }

    return res;
}

bool llama_memory_hybrid_idx_context::kpool_track() const {
    // Derived from mem instead of being cached.
    return mem != nullptr && mem->get_mem_idx() != nullptr && mem->get_kpool() > 0 && !ns_ubatch.empty();
}

const llama_kv_cache_context * llama_memory_hybrid_idx_context::get_idx() const {
    return static_cast<const llama_kv_cache_context *>(ctx_idx.get());
}

uint32_t llama_memory_hybrid_idx_context::get_n_stream() const {
    GGML_ASSERT(i_cur < ns_ubatch.size());

    return ns_ubatch[i_cur];
}

llama_memory_hybrid_idx_context::kpool_access::kpool_access(ggml_context * ctx, ggml_tensor * k, int64_t n_embd) : ctx(ctx) {
    // rows are the per-token part (glm5-next: key | gate, qwen4exp: key), then the pooled key
    const int64_t n_tok = k->ne[0] - n_embd;
    GGML_ASSERT(n_tok > 0 && n_tok % n_embd == 0);

    const int64_t n_cells = k->ne[1]*k->ne[2];

    // Pool indices can refer to other streams. Revisit these full-storage views if that changes:
    // https://github.com/ggml-org/llama.cpp/pull/27773#discussion_r4130905603
    key_gate = ggml_view_2d(ctx, k, n_tok,  n_cells, k->nb[1], 0);
    pooled   = ggml_view_2d(ctx, k, n_embd, n_cells, k->nb[1], ggml_row_size(k->type, n_tok));
}

ggml_tensor * llama_memory_hybrid_idx_context::kpool_access::gather_key_gate(ggml_tensor * idxs) const {
    return ggml_get_rows(ctx, key_gate, idxs);
}

ggml_tensor * llama_memory_hybrid_idx_context::kpool_access::scatter_pooled(ggml_tensor * values, ggml_tensor * idxs) const {
    return ggml_set_rows(ctx, pooled, values, idxs);
}

ggml_tensor * llama_memory_hybrid_idx_context::kpool_access::gather_pooled(ggml_tensor * idxs) const {
    return ggml_get_rows(ctx, pooled, idxs);
}

llama_memory_hybrid_idx_context::kpool_access llama_memory_hybrid_idx_context::get_kpool_access(
        ggml_context * ctx, int32_t il, int64_t n_embd) const {
    GGML_ASSERT(mem != nullptr && mem->get_mem_idx() != nullptr);

    return kpool_access(ctx, mem->get_mem_idx()->get_k_storage(il), n_embd);
}

ggml_tensor * llama_memory_hybrid_idx_context::gather_mla_rows(
        ggml_context * ctx, ggml_tensor * idxs, int64_t n_rows, int64_t n_embd, int32_t il) const {
    GGML_ASSERT(mem != nullptr);
    ggml_tensor * k = mem->get_mem_attn()->get_k_storage(il);
    GGML_ASSERT(k->ne[0] == n_embd);

    ggml_tensor * rows = ggml_view_2d(ctx, k, k->ne[0], k->ne[1]*k->ne[2], k->nb[1], 0);
    return ggml_get_rows(ctx, rows, ggml_reshape_1d(ctx, idxs, n_rows));
}

// k-pool DSA indexer (glm5-next, qwen4exp QSA)

// Sizes only, used by the full cache context so get_n_kpool() works during graph reserve.
llama_memory_hybrid_idx_context::kpool_state llama_memory_hybrid_idx_context::kpool_build_sizes() const {
    const auto & lay = mem->kpool_layout_get();

    kpool_state st;
    st.n_pool_real = lay.n_pool_real;
    st.cache_safe  = lay.cache_safe;

    return st;
}

// Which pools this ubatch must re-pool.
// Pool cache lifecycle:
// 1. cpy_k writes each token's key | gate into its idx cache row, pooled slot are zeroed.
// 2. This marks the pools the ubatch touches or completes as new, during decode that's one pool every kpool tokens, zero elsewise.
// 3. The graph pools only the new pools and set_rows each result into the pooled slot of the pool's last member row.
// 4. All pools are gathered in one get_rows via pool_cells, fresh ones just written, older ones from whatever batch last wrote them.
// A seq_* edit regroups the pools from the edited position on, so it stales them and the first ubatch of the next batch
// rebuilds them from the still-valid key | gate rows, rewriting the (possibly different) rep rows.
// Orphaned pooled slots are never cleared, a slot is only ever read through pool_cells, which follows the current grouping.
void llama_memory_hybrid_idx_context::kpool_build_state(const llama_ubatch & ubatch) {
    const auto & lay = mem->kpool_layout_get();
    auto & st = *kpool_st;

    st.n_pool_real = lay.n_pool_real;
    st.cache_safe  = lay.cache_safe;
    st.n_new       = 0;
    if (++st.generation == 0) {
        std::fill(st.is_new.begin(), st.is_new.end(), 0);
        st.generation = 1;
    }
    st.is_new.resize(lay.n_pool_real, 0);

    auto mark = [&](uint32_t ip) {
        if (st.is_new[ip] != st.generation) {
            st.is_new[ip] = st.generation;
            ++st.n_new;
        }
    };

    const uint32_t kpool = mem->get_kpool();
    std::array<uint32_t, LLAMA_MAX_SEQ> pool_start;
    uint32_t ip = 0;
    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
        const auto & sq = lay.seqs[s];
        pool_start[s] = ip;
        ip += (uint32_t) sq.pools.size();

        if (!st.cache_safe) {
            continue;
        }

        // A sequence edit invalidates only pools ending after the edited position.
        const llama_pos stale_from = i_cur == 0 ?
            mem_idx_stale_batch[s] : llama_memory_hybrid_idx::POS_CLEAN;
        if (stale_from == llama_memory_hybrid_idx::POS_CLEAN) {
            continue;
        }

        auto first = std::lower_bound(sq.pools.begin(), sq.pools.end(), stale_from,
                [&](uint32_t j, llama_pos p) { return sq.cells[j + kpool - 1].first < p; });
        for (auto it = first; it != sq.pools.end(); ++it) {
            mark(pool_start[s] + (uint32_t) (it - sq.pools.begin()));
        }
    }
    GGML_ASSERT(ip == st.is_new.size());

    if (!st.cache_safe) {
        std::fill(st.is_new.begin(), st.is_new.end(), st.generation);
        st.n_new   = st.n_pool_real;
        st.n_new_g = std::max(st.n_new, 1u);
        return;
    }

    // in order mode a token's cell gives its rank, and the rank its pool: positions cannot, as an image shares one
    const bool by_order = mem->get_kpool_by_order();
    const auto *   sinfo = by_order ? &sinfos_kpool[i_cur] : nullptr;
    const uint32_t n_tps = by_order ? (uint32_t) sinfo->size() : 0;

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        const llama_pos p = ubatch.pos[i];
        for (int32_t k = 0; k < ubatch.n_seq_id[i]; ++k) {
            const llama_seq_id s = ubatch.seq_id[i][k];
            const auto & sq = lay.seqs[s];
            if (by_order) {
                const int64_t r = kpool_rank(sq.cells, p, sinfo->idxs[i / n_tps][i % n_tps]);
                GGML_ASSERT(r >= 0);
                if ((size_t) r / kpool < sq.pools.size()) {
                    mark(pool_start[s] + (uint32_t) (r / kpool));
                }
                continue;
            }
            auto it = std::upper_bound(sq.pools.begin(), sq.pools.end(), p,
                    [&](llama_pos pos, uint32_t j) { return pos < sq.cells[j].first; });
            if (it == sq.pools.begin()) {
                continue;
            }
            --it;
            if (p <= sq.cells[*it + kpool - 1].first) {
                mark(pool_start[s] + (uint32_t) (it - sq.pools.begin()));
            }
        }
    }

    // a ubatch touches at most t_s/kpool + 1 pools per sequence, pad to that bound so the graph keeps its shape
    // as the count moves; reserve sizes the list for every pool the cache can hold, so never pad past n_pool_max
    const auto *   idx        = mem->get_mem_idx();
    const uint32_t n_pool_max = idx->get_size() / kpool * idx->get_n_seq_max();
    const uint32_t bound = ubatch.n_tokens/kpool + ubatch.n_seqs_unq;
    st.n_new_g = std::max({st.n_new, 1u, std::min({bound, kpool_pad(st.n_pool_real) - 1, n_pool_max})});
}

const llama_memory_hybrid_idx_context::kpool_state & llama_memory_hybrid_idx_context::kpool_cur() const {
    GGML_ASSERT(kpool_st != nullptr && i_kpool == i_cur && "k-pool state read before apply()");

    return *kpool_st;
}

uint32_t llama_memory_hybrid_idx_context::get_n_kpool() const {
    return kpool_pad(kpool_cur().n_pool_real);
}

uint32_t llama_memory_hybrid_idx_context::get_n_kpool_new() const {
    return kpool_cur().n_new_g;
}

bool llama_memory_hybrid_idx_context::get_kpool_cache_safe() const {
    return kpool_cur().cache_safe;
}

void llama_memory_hybrid_idx_context::set_input_kpool(ggml_tensor * pool_cells, ggml_tensor * pool_idxs, ggml_tensor * pool_mask, ggml_tensor * tail_idxs,
        ggml_tensor * gather_mask, bool gather, ggml_tensor * new_pool_idxs, ggml_tensor * new_pool_rep,
        const llama_ubatch * ubatch, ggml_tensor * new_pool_pos) const {
    GGML_ASSERT(mem != nullptr && mem->get_mem_idx() != nullptr);
    GGML_ASSERT(ggml_backend_buffer_is_host(pool_cells->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(pool_idxs->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(pool_mask->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(tail_idxs->buffer));

    const uint32_t kpool = mem->get_kpool();
    const uint32_t n_kv  = get_idx()->get_n_kv();

    const auto & st  = kpool_cur();
    const auto & lay = mem->kpool_layout_get();

    const uint32_t n_tokens = ubatch->n_tokens;
    const uint32_t n_pool   = (uint32_t) pool_cells->ne[0];
    const uint32_t n_new    = st.n_new;
    // the graph always pools at least one entry, padded to a stable bound, see kpool_build_state
    const uint32_t n_new_g  = st.n_new_g;

    const bool by_order = mem->get_kpool_by_order();

    GGML_ASSERT(n_pool == kpool_pad(st.n_pool_real));
    GGML_ASSERT(st.is_new.size() == st.n_pool_real);
    GGML_ASSERT(pool_mask->ne[0] == (int64_t) n_pool && pool_mask->ne[1] == (int64_t) n_tokens);
    GGML_ASSERT(tail_idxs->ne[0] == (int64_t) kpool - 1 && tail_idxs->ne[1] == (int64_t) n_tokens);
    GGML_ASSERT(pool_idxs->ne[0] == (int64_t) kpool && pool_idxs->ne[1] == (int64_t) n_pool);
    GGML_ASSERT(st.cache_safe == (new_pool_rep != nullptr));
    GGML_ASSERT(ggml_backend_buffer_is_host(new_pool_idxs->buffer));
    GGML_ASSERT(new_pool_idxs->ne[0] == (int64_t) kpool && new_pool_idxs->ne[1] == (int64_t) n_new_g);
    if (new_pool_rep != nullptr) {
        GGML_ASSERT(ggml_backend_buffer_is_host(new_pool_rep->buffer));
        GGML_ASSERT(new_pool_rep->ne[0] == (int64_t) n_new_g);
    }
    if (new_pool_pos != nullptr) {
        GGML_ASSERT(ggml_backend_buffer_is_host(new_pool_pos->buffer));
        GGML_ASSERT(new_pool_pos->ne[0] == 4*(int64_t) n_new_g);
    }

    const uint32_t kv_size = mem->get_mem_idx()->get_size();
    const uint32_t n_stream_kv = mem->get_mem_idx()->get_n_stream();

    auto gcell = [&](const llama_memory_hybrid_idx::kpool_layout::seq & sq, uint32_t cell) {
        return (int64_t) sq.strm*kv_size + cell;
    };

    // Sequences present in this ubatch, pools of absent sequences must fall on the scatter sentinel row.
    std::vector<uint8_t> seq_in_ub(LLAMA_MAX_SEQ, 0);
    for (uint32_t i = 0; i < n_tokens; ++i) {
        for (int32_t k = 0; k < ubatch->n_seq_id[i]; ++k) {
            seq_in_ub[ubatch->seq_id[i][k]] = 1;
        }
    }

    // Use the first ubatch cell for padded gathers.
    int64_t dummy_cell = 0;
    {
        const llama_seq_id s = ubatch->seq_id[0][0];
        const auto & sq = lay.seqs[s];
        auto it = std::lower_bound(sq.cells.begin(), sq.cells.end(), std::make_pair(ubatch->pos[0], 0u));
        GGML_ASSERT(it != sq.cells.end() && it->first == ubatch->pos[0]);
        dummy_cell = gcell(sq, it->second);
    }

    // in order mode a token sees the pools and the tail up to its own rank in the sequence, which its cell pins down
    std::vector<int64_t> rank;
    if (by_order) {
        const auto &   sinfo = sinfos_kpool[i_cur];
        const uint32_t n_tps = (uint32_t) sinfo.size();

        rank.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; ++i) {
            rank[i] = kpool_rank(lay.seqs[ubatch->seq_id[i][0]].cells, ubatch->pos[i], sinfo.idxs[i / n_tps][i % n_tps]);
            GGML_ASSERT(rank[i] >= 0);
        }
    }

    // Gather maps padding to a real cell and masks it separately.
    const int32_t sentinel = gather ? (int32_t) dummy_cell : (int32_t) n_kv;

    float *  gm    = nullptr;
    uint32_t n_sel = 0;
    uint32_t n_top = 0; // Pools per token in the selection.
    if (gather_mask != nullptr) {
        GGML_ASSERT(ggml_backend_buffer_is_host(gather_mask->buffer));
        GGML_ASSERT(gather_mask->type == GGML_TYPE_F32);
        GGML_ASSERT(gather_mask->ne[3] == (int64_t) n_tokens && gather_mask->ne[1] == 1 && gather_mask->ne[2] == 1);
        n_sel = (uint32_t) gather_mask->ne[0];
        // The tail slots, when selected, are the n_sel % kpool != 0 remainder.
        n_top = n_sel / kpool;
        GGML_ASSERT(n_sel % kpool == 0 || n_sel % kpool == kpool - 1);
        gm = (float *) gather_mask->data;
    }

    // pools are laid out per sequence
    std::vector<uint32_t>  seq_pool_start(LLAMA_MAX_SEQ, 0);
    std::vector<llama_pos> pool_end;
    pool_end.reserve(n_pool);

    int32_t * pcell = (int32_t *) pool_cells->data;
    int32_t * pidx  = (int32_t *) pool_idxs->data;
    int32_t * nidx  = (int32_t *) new_pool_idxs->data;
    int64_t * nrep  = new_pool_rep != nullptr ? (int64_t *) new_pool_rep->data : nullptr;
    int32_t * npos  = new_pool_pos != nullptr ? (int32_t *) new_pool_pos->data : nullptr;

    if (npos != nullptr) {
        std::fill(npos, npos + 4*n_new_g, 0);
    }

    uint32_t i_new = 0;
    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
        const auto & sq = lay.seqs[s];
        seq_pool_start[s] = (uint32_t) pool_end.size();

        const bool inert = !gather && n_stream_kv > 1 && !seq_in_ub[s];

        for (size_t pi = 0; pi < sq.pools.size(); ++pi) {
            const uint32_t j  = sq.pools[pi];
            const uint32_t ip = (uint32_t) pool_end.size();
            GGML_ASSERT(ip + 1 < n_pool);

            // The pooled key lives in the last member's row.
            const uint32_t rep = sq.cells[j + kpool - 1].second;
            pcell[ip] = (int32_t) gcell(sq, rep);

            for (uint32_t k = 0; k < kpool; ++k) {
                pidx[(size_t) ip*kpool + k] = inert ? sentinel :
                    (int32_t) (gather ? gcell(sq, sq.cells[j + k].second) : (int64_t) sq.cells[j + k].second);
            }

            if (st.is_new[ip] == st.generation) {
                GGML_ASSERT(i_new < n_new);
                for (uint32_t k = 0; k < kpool; ++k) {
                    nidx[(size_t) i_new*kpool + k] = (int32_t) gcell(sq, sq.cells[j + k].second);
                }
                if (nrep != nullptr) {
                    nrep[i_new] = gcell(sq, rep);
                }
                if (npos != nullptr) {
                    // a pooled key is rotated to the M-RoPE position of its first member
                    const uint32_t c = sq.cells[j].second;
                    const auto &   e = mem->get_mem_idx()->get_cells(s).ext_get(c);
                    npos[0*n_new_g + i_new] = sq.cells[j].first;
                    npos[1*n_new_g + i_new] = e.y;
                    npos[2*n_new_g + i_new] = e.x;
                    npos[3*n_new_g + i_new] = sq.cells[j].first;
                }
                ++i_new;
            }

            pool_end.push_back(sq.cells[j + kpool - 1].first);
        }
    }
    GGML_ASSERT(i_new == n_new);

    // Padded entries re-pool cells whose pooled slot is never read: only the reps of complete pools are read.
    // Each entry takes its own cell, entries sharing one would write it from several threads in the scatter.
    if (n_new_g > n_new) {
        std::vector<int64_t> reps(pcell, pcell + pool_end.size());
        std::sort(reps.begin(), reps.end());

        int64_t pad_cell = 0;
        for (uint32_t i = n_new; i < n_new_g; ++i, ++pad_cell) {
            while (std::binary_search(reps.begin(), reps.end(), pad_cell)) {
                ++pad_cell;
            }
            GGML_ASSERT(pad_cell < (int64_t) kv_size*n_stream_kv);
            for (uint32_t k = 0; k < kpool; ++k) {
                nidx[(size_t) i*kpool + k] = (int32_t) pad_cell;
            }
            if (nrep != nullptr) {
                nrep[i] = pad_cell;
            }
        }
    }

    const uint32_t n_pool_real = (uint32_t) pool_end.size();
    for (uint32_t ip = n_pool_real; ip < n_pool; ++ip) {
        pcell[ip] = (int32_t) dummy_cell; // pool_cells always addresses the K storage
        for (uint32_t k = 0; k < kpool; ++k) {
            pidx[(size_t) ip*kpool + k] = sentinel;
        }
    }

    // a pool is visible when it belongs to the token's sequence and ends at or before it
    auto fill_mask = [&](auto * data) {
        using T = std::remove_pointer_t<decltype(data)>;
        const T keep = llama_cast<T>(0.0f);
        const T drop = llama_cast<T>(-INFINITY);

        for (uint32_t i = 0; i < n_tokens; ++i) {
            const llama_seq_id s = ubatch->seq_id[i][0];
            const llama_pos    p = ubatch->pos[i];

            T * row = data + (size_t) i*n_pool;
            std::fill(row, row + n_pool, drop);

            const uint32_t p0 = seq_pool_start[s];
            const uint32_t p1 = p0 + (uint32_t) lay.seqs[s].pools.size();
            const uint32_t nv = by_order ? std::min(p1 - p0, (uint32_t) ((rank[i] + 1)/kpool)) :
                (uint32_t) (std::upper_bound(pool_end.begin() + p0, pool_end.begin() + p1, p) - (pool_end.begin() + p0));
            std::fill(row + p0, row + p0 + nv, keep);

            // Finite visible pools occupy the first min(nv, n_top) ranked slots.
            if (gm != nullptr) {
                const uint32_t nvc = std::min(nv, n_top);
                float * grow = gm + (size_t) i*n_sel;
                std::fill(grow,                        grow + (size_t) nvc*kpool,  0.0f);
                std::fill(grow + (size_t) nvc*kpool,   grow + (size_t) n_top*kpool, -INFINITY);
            }
        }
    };
    if (pool_mask->type == GGML_TYPE_F16) {
        fill_mask((ggml_fp16_t *) pool_mask->data);
    } else {
        fill_mask((float *) pool_mask->data);
    }

    int32_t * tidx = (int32_t *) tail_idxs->data;
    for (uint32_t i = 0; i < n_tokens; ++i) {
        const llama_seq_id s = ubatch->seq_id[i][0];
        const llama_pos    p = ubatch->pos[i];
        const auto & sq = lay.seqs[s];

        const uint32_t n_tail = by_order ?
            (uint32_t) ((rank[i] + 1) % kpool) :
            (uint32_t) ((p - sq.pos_min + 1) % (llama_pos) kpool);

        for (uint32_t k = 0; k < kpool - 1; ++k) {
            int32_t cell = sentinel;
            bool    real = false;
            if (k < n_tail && by_order) {
                const uint32_t c = sq.cells[rank[i] - k].second;
                cell = (int32_t) (gather ? gcell(sq, c) : (int64_t) c);
                real = true;
            } else if (k < n_tail) {
                const llama_pos pt = p - (llama_pos) k;
                auto it = std::lower_bound(sq.cells.begin(), sq.cells.end(), std::make_pair(pt, 0u));
                if (it != sq.cells.end() && it->first == pt) {
                    cell = (int32_t) (gather ? gcell(sq, it->second) : (int64_t) it->second);
                    real = true;
                }
            }
            tidx[(size_t) i*(kpool - 1) + k] = cell;

            if (gm != nullptr && n_sel % kpool != 0) {
                gm[(size_t) i*n_sel + (size_t) n_top*kpool + k] = real ? 0.0f : -INFINITY;
            }
        }
    }
}

uint32_t llama_memory_hybrid_idx_context::qsa_n_kv_window() const {
    const uint32_t n_kv = get_idx() ? get_idx()->get_n_kv() : 0;
    const llama_kv_cache * idx = mem ? mem->get_mem_idx() : nullptr;
    if (idx == nullptr) {
        return n_kv;
    }
    // seq_to_stream is LLAMA_MAX_SEQ-sized in the unified (n_stream == 1) case and n_stream-sized
    // otherwise, and get_cells() asserts, so bound the seq sweep to what that vector holds.
    const uint32_t n_stream = idx->get_n_stream();
    const llama_seq_id n_seq = n_stream > 1 ? (llama_seq_id) n_stream : (llama_seq_id) LLAMA_MAX_SEQ;
    llama_pos pos_max = -1;
    for (llama_seq_id s = 0; s < n_seq; ++s) {
        pos_max = std::max(pos_max, idx->get_cells(s).seq_pos_max(s));
    }
    if (pos_max < 0) {
        return n_kv;
    }
    const uint32_t window = ((uint32_t) pos_max + 1 + 255) / 256 * 256;
    return std::max(n_kv, window);
}

// [QSA_SCORE_BOUNDS] true when the occupied cells of `seq` hold a unique set of non-negative
// positions, one sequence only.  Complete blocks are enumerated in ascending logical block order
// (see set_input_qsa's grouping), so under this precondition a complete block's ordinal cannot
// exceed its logical block number and the scorer can be trimmed to a strip's visible prefix.
static bool qsa_single_sequence_prefix(const llama_kv_cells & cells, uint32_t count, llama_seq_id seq) {
    if (count > cells.size()) {
        return false;
    }

    std::vector<llama_pos> positions;
    positions.reserve(count);

    for (uint32_t i = 0; i < count; ++i) {
        if (cells.is_empty(i)) {
            continue;
        }
        // the scorer trim is by cell index, so the cell index must equal the position (no
        // permutation); a hole keeps the check valid, an empty cell contributes nothing
        if (cells.seq_get_all(i).count() != 1 || !cells.seq_has(i, seq) || cells.pos_get(i) != (llama_pos) i) {
            return false;
        }
        positions.push_back(cells.pos_get(i));
    }

    std::sort(positions.begin(), positions.end());
    return std::adjacent_find(positions.begin(), positions.end()) == positions.end();
}

bool llama_memory_hybrid_idx_context::qsa_position_prefix(const llama_ubatch & ubatch) const {
    if (get_n_stream() != 1 || !get_idx() || mem == nullptr) {
        return false;
    }
    if (!ubatch.token || !ubatch.pos || !ubatch.n_tokens || !ubatch.n_pos ||
        !ubatch.seq_id || !ubatch.n_seq_id) {
        return false;
    }
    if (ubatch.n_seq_id[0] < 1 || !ubatch.seq_id[0]) {
        return false;
    }

    const llama_seq_id seq = ubatch.seq_id[0][0];

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.n_seq_id[i] < 1 || !ubatch.seq_id[i] || ubatch.seq_id[i][0] != seq ||
            ubatch.pos[i] < 0 || ubatch.pos[i] >= 16777216) {
            return false;
        }
        // M-RoPE repeats a position across an image, which breaks the unique-position ordering
        for (uint32_t axis = 1; axis < ubatch.n_pos; ++axis) {
            if (ubatch.pos[i + axis*ubatch.n_tokens] != ubatch.pos[i]) {
                return false;
            }
        }
    }

    const llama_kv_cache * idx = mem->get_mem_idx();
    if (idx == nullptr) {
        return false;
    }

    return qsa_single_sequence_prefix(idx->get_cells(seq), get_idx()->get_n_kv(), seq);
}

std::vector<int64_t> llama_memory_hybrid_idx_context::qsa_score_key_limits(
        const llama_ubatch & ubatch, int64_t n_blocks, int64_t strip, uint32_t ratio, int64_t budget) const {
    const int64_t n_tokens = ubatch.n_tokens;

    if (!ubatch.pos || n_tokens <= 0 || strip <= 0 || ratio == 0 || budget <= 0 || n_blocks < budget) {
        return {};
    }

    // llama_context::graph_reserve builds a worst-case graph from a synthetic ubatch whose positions
    // are all 0.  Bounding that graph would reserve compute buffers for a 4%-wide scorer and then
    // execute full-width ones, so a many-token ubatch whose positions are all identical is treated
    // as synthetic and left unbounded.
    bool degenerate_pos = n_tokens > 1;
    for (int64_t i = 1; degenerate_pos && i < n_tokens; ++i) {
        if (ubatch.pos[i] != ubatch.pos[0]) {
            degenerate_pos = false;
        }
    }
    if (degenerate_pos || !qsa_position_prefix(ubatch)) {
        return {};
    }

    std::vector<int64_t> limits;
    limits.reserve((n_tokens + strip - 1)/strip);

    for (int64_t first = 0; first < n_tokens; first += strip) {
        int64_t maximum = -1;
        for (int64_t i = first; i < std::min(n_tokens, first + strip); ++i) {
            if (ubatch.pos[i] < 0 || ubatch.pos[i] >= 16777216) {
                return {};
            }
            maximum = std::max(maximum, (int64_t) ubatch.pos[i]);
        }
        // (maximum+1)/ratio complete blocks lie fully inside the strip's causal prefix.  The +1
        // keeps the incomplete tail block - the fused top-k carries it as ordinary cells, whereas
        // the reference appends it separately - inside the scored prefix.
        limits.push_back(std::min(n_blocks, std::max(budget, (maximum + 1)/ratio) + 1));
    }

    if (getenv("LLAMA_QSA_SCORE_STRIP_DEBUG") != nullptr) {
        fprintf(stderr, "QSA score bounds: n_tokens=%lld n_blocks=%lld strip=%lld ratio=%u budget=%lld limits=[%lld, %lld]\n",
                (long long) n_tokens, (long long) n_blocks, (long long) strip, ratio, (long long) budget,
                (long long) limits.front(), (long long) limits.back());
    }

    return limits;
}

void llama_memory_hybrid_idx_context::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        ggml_tensor * blk_idx,
        ggml_tensor * blk_tail,
        ggml_tensor * cell_vis,
        ggml_tensor * q_vis,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias,
        int32_t * dst_derived_from,
        int32_t * dst_derived_lim) const {
    GGML_ASSERT(mem != nullptr);

    mem->set_input_qsa(cell_blk, blk_cells, blk_pos, bias, blk_idx, blk_tail, cell_vis, q_vis,
                       ubatch, ratio, blk_bias, dst_derived_from, dst_derived_lim);
}

ggml_tensor * llama_memory_hybrid_idx_context::get_pool(ggml_context * ctx, int32_t il, uint32_t n_blocks) const {
    GGML_ASSERT(mem != nullptr);

    return mem->get_pool(ctx, il, n_blocks);
}
