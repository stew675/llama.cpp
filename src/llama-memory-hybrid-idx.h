#pragma once

#include "llama-memory-hybrid.h"

#include <array>
#include <limits>
#include <memory>
#include <vector>

//
// llama_memory_hybrid_idx
//

// llama_memory_hybrid plus a third cache with one indexer key per token, for block-sparse attention (qwen4exp QSA)
// the indexer is a side buffer over the attention cells: same size, padding, streams and slots, so cell j is one token in both

class llama_memory_hybrid_idx : public llama_memory_hybrid {
public:
    llama_memory_hybrid_idx(
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
                            /* the indexer cache exists only if this is given */
    const layer_filter_cb & filter_idx);

    // Defined out of line because kpool_layout is incomplete here.
    ~llama_memory_hybrid_idx();

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0)       override;

    //
    // llama_memory_hybrid_idx specific API
    //

    llama_kv_cache * get_mem_idx() const;   // nullptr when the model carries no indexer

    // The model's indexer pool size.
    uint32_t get_kpool() const { return hparams_idx.indexer_kpool; }

    // Whether pools are kpool consecutive cells in sequence order (qwen4exp) instead of kpool consecutive positions.
    bool get_kpool_by_order() const { return hparams_idx.indexer_kpool_by_order; }

    // Which cells of a sequence make up which pool of kpool consecutive positions (or cells, in order mode).
    // It is kept here because it outlives the batch: pools are fixed by the positions relative to the
    // sequence's first one, so a ubatch only ever appends to it. Sequence edits drop it, see mem_idx_stale.
    struct kpool_layout;

    const kpool_layout & kpool_layout_update();
    const kpool_layout & kpool_layout_get() const;

    // The pooled keys persist in the idx cache across batches. A sequence edit can regroup the pools
    // from some position on, which stales every pooled key at or after it. POS_CLEAN means none.
    using stale_pos_t = std::array<llama_pos, LLAMA_MAX_SEQ>;

    static constexpr llama_pos POS_CLEAN = std::numeric_limits<llama_pos>::max();

    static stale_pos_t stale_pos_clean() {
        stale_pos_t res;
        res.fill(POS_CLEAN);
        return res;
    }

    const stale_pos_t & mem_idx_stale_get() const { return mem_idx_stale; }
    void mem_idx_stale_clear() { mem_idx_stale.fill(POS_CLEAN); }
    // incremental block-vector cache access: the F32 pool tensor of indexer layer `il`
    // (nullptr when the model has no indexer); view [idx_dim x n_blocks x n_stream],
    // n_blocks = ceil(n_kv/ratio), rows [0, watermark) valid
    ggml_tensor * get_pool(ggml_context * ctx, int32_t il, uint32_t n_blocks) const;

    // fill the per-step derived-cache host leaves: fill range [from, to) with the score limit == to.
    // n_bid = count of full blocks per stream (from set_input_qsa's grouping).  Called by the const
    // set_input_qsa on the decode append path (advance = the derived path is live this step).
    void qsa_derived_limits(int32_t * dst_fill_from, int32_t * dst_limit, int n_stream, uint32_t ratio,
                            const uint32_t * n_bid, bool advance) const;

    // block-compressed sparse attention (qwen4exp QSA) over the cells of the indexer cache.
    // Blocks cut the position line, not the cell array, so no caller assumes a contiguous layout:
    //   cell_blk  I32 [n_kv, ns]           block each cell belongs to
    //   blk_cells I32 [ratio*n_blocks, ns] cells making up each block
    //   blk_pos   I32 [4*n_blocks*ns]      mrope position rows of each block's first token
    //   bias      F32 [n_kv, n_tokens/ns, ns] -inf where invisible, large where always visible
    // blk_bias asks for the bias per block instead: [n_blocks, n_tokens/ns, ns]
    // the caller then adds the attention mask, the only part of the bias that varies within a block
    //
    // blk_idx/blk_tail are the compact (derived) alternative to the bias tensor: they let the
    // top-k derive the per-block half of the bias in-kernel from 4 bytes per block instead of
    // n_tokens/ns.  blk_idx is -1 for a block that is not complete for this stream, INT32_MAX
    // for the spare block holding the unpooled tail cells, else the position of the block's
    // first cell; blk_tail holds the per-token tail start.  The per-sequence half of the bias
    // is not folded in: the visibility (the attention mask, or the derived cell positions)
    // already drops every cell of a foreign block, so the values stay identical.  A caller
    // passing blk_idx must not add the bias into the block score itself.
    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * bias, ggml_tensor * blk_idx, ggml_tensor * blk_tail,
                       ggml_tensor * cell_vis, ggml_tensor * q_vis,
                       const llama_ubatch * ubatch, uint32_t ratio,
                       bool blk_bias,
                       int32_t * dst_derived_from = nullptr,
                       int32_t * dst_derived_lim  = nullptr) const;

private:
    // forget seq_id (all of it if seq_id < 0) in every cache at once, so a failed restore cannot leave the caches out of step
    // seq_id < 0 drops the whole context, as the caches themselves do on a failed restore
    void state_drop(llama_seq_id seq_id);

    // the indexer cache holds one key head per layer, so it needs its own hparams:
    // llama_kv_cache keeps a reference to what it is given
    llama_hparams hparams_idx;

    const std::unique_ptr<llama_kv_cache> mem_idx;

    // unique_ptr because kpool_layout is incomplete here
    std::unique_ptr<kpool_layout> kpool_lay;

    // whether the current layout has cells shared between sequences (kpool_layout is incomplete here, so out of line)
    bool kpool_layout_shared() const;

    // seq_id < 0 stales every sequence, p0 < 0 stales the sequence from its first position
    void mem_idx_stale_set(llama_seq_id seq_id, llama_pos p0);

    // the position an edit at p0 stales the sequence from
    llama_pos mem_idx_stale_pos(llama_seq_id seq_id, llama_pos p0) const;

    stale_pos_t mem_idx_stale = stale_pos_clean();
    //
    // incremental block-vector cache ("derived cache", qwen4exp QSA decode waste fix)
    //
    // One F32 [indexer_head_size x ceil(kv_size/ratio) x n_stream] tensor per indexer layer
    // holding the pooled + rms-normed + ROTATED vector of each FULL block, computed once when
    // the block completes instead of re-pooling the raw cache every decode token.  Lifecycle =
    // a per-layer host watermark: rows [0, watermark) are valid; any sequence mutation drops the
    // watermarks (the rows are never read above the watermark, so nothing needs memsetting).
    // The graph ops (fill + the derived score path) receive the range via per-step host leaves;
    // set_input_qsa fills them and advances the watermarks (decode-only, env-gated).
    struct llama_mem_pool_layer {
        uint32_t il;                  // model layer id (dense-attention, indexer-carrying)
        uint32_t ratio;               // compress ratio of this layer (blocks = ceil(kv_size/ratio))
        ggml_tensor * pool = nullptr; // F32 [idx_dim, n_blocks, n_stream]
    };

    // per-layer derived tensors + the contexts/buffers owning their memory
    std::vector<llama_mem_pool_layer> pool_layers;
    std::vector<ggml_context_ptr>      pool_ctxs;
    std::vector<ggml_backend_buffer_ptr> pool_bufs;

    // watermark per pool layer (rows [0, wm) are valid); decode advances it, seq ops drop it
    // mutable: set_input_qsa (const) advances it on the decode append path
    mutable std::vector<uint32_t> pool_wm;

    // the derived path is decode-only; ON by default, GGML_CUDA_QSA_INDEXER_CACHE=0 disables
    // (the constructor overwrites this from the env)
    bool derived_enabled = false;

    void pool_invalidate_all();
    void pool_create(const llama_model & model, const layer_filter_cb & filter_idx);
};

class llama_memory_hybrid_idx_context : public llama_memory_hybrid_context {
public:
    class kpool_access {
    public:
        ggml_tensor * gather_key_gate(ggml_tensor * idxs) const;
        ggml_tensor * scatter_pooled(ggml_tensor * values, ggml_tensor * idxs) const;
        ggml_tensor * gather_pooled(ggml_tensor * idxs) const;

    private:
        friend class llama_memory_hybrid_idx_context;

        kpool_access(ggml_context * ctx, ggml_tensor * k, int64_t n_embd);

        ggml_context * ctx;
        ggml_tensor  * key_gate;
        ggml_tensor  * pooled;
    };

    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // used for errors
    explicit llama_memory_hybrid_idx_context(llama_memory_status status);

    // used to create a full-cache context
    explicit llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem);

    // used to create an update context
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                      llama_context * lctx,
                               bool   optimize);

    // used to create a batch processing context from a batch
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                    slot_info_vec_t   sinfos_attn,
                    slot_info_vec_t   sinfos_idx,
          std::vector<llama_ubatch>   ubatches);

    ~llama_memory_hybrid_idx_context(); // Defined out of line because kpool_state is incomplete here.

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    //
    // llama_memory_hybrid_idx_context specific API
    //

    // nullptr with no indexer
    const llama_kv_cache_context * get_idx() const;

    // streams in the current slot info, the `ns` of get_k/get_v; 1 if unified
    uint32_t get_n_stream() const;

    // glm5-next and qwen4exp, complete pools of kpool cells per sequence, scored as whole pools.
    uint32_t get_n_kpool    () const; // Padded pool count, where the last pool is always unused.
    uint32_t get_n_kpool_new() const; // Pools to re-pool this ubatch, padded to a stable bound, never below 1.
    bool get_kpool_cache_safe() const;
    kpool_access get_kpool_access(ggml_context * ctx, int32_t il, int64_t n_embd) const;
    ggml_tensor * gather_mla_rows(ggml_context * ctx, ggml_tensor * idxs, int64_t n_rows, int64_t n_embd, int32_t il) const;
    // new_pool_pos (I32 [4*n_new]): M-RoPE position of each new pool's first member, for pooled keys rotated at pooling time
    void set_input_kpool(ggml_tensor * pool_cells, ggml_tensor * pool_idxs, ggml_tensor * pool_mask, ggml_tensor * tail_idxs,
                         ggml_tensor * gather_mask, bool gather, ggml_tensor * new_pool_idxs, ggml_tensor * new_pool_rep,
                         const llama_ubatch * ubatch, ggml_tensor * new_pool_pos = nullptr) const;
    // Cells the QSA block metadata must cover.  The KV view is sized by OCCUPIED cells, but blocks
    // are keyed by POSITION, and a cache whose positions run ahead of its cells has blocks past that
    // view: the MTP draft context never receives the cells an M-RoPE image pins to one position, so
    // after an image its highest position leads its cell count by the image's grid size.  Sizing the
    // block tensors from get_n_kv() then makes the fill walk past the window (assert / corrupt read);
    // use max(get_n_kv(), highest stored position + 1), padded to 256 like get_n_kv() so graph reuse
    // keeps its cadence.  (Ported from the other solution's b0f31f587.)
    uint32_t qsa_n_kv_window() const;

    // [QSA_SCORE_BOUNDS] precondition for trimming the indexer scorer to the columns a query strip
    // can actually see: one sequence whose occupied cache cells carry unique non-negative positions
    // (so the complete blocks are enumerated in ascending logical block order and a block's ordinal
    // cannot exceed its logical block number).  M-RoPE images pin several cells to one position, so
    // they fail the uniqueness check and stay unbounded.
    bool qsa_position_prefix(const llama_ubatch & ubatch) const;

    // [QSA_SCORE_BOUNDS] per query strip, the number of leading score columns the strip can see
    // (the complete-block ordinals that are fully inside its causal prefix, plus the incomplete
    // tail block the fused top-k carries as cells), or an empty vector when the bound does not
    // apply.  `strip` is in tokens, `budget` = indexer_top_k / ratio.
    std::vector<int64_t> qsa_score_key_limits(const llama_ubatch & ubatch, int64_t n_blocks,
            int64_t strip, uint32_t ratio, int64_t budget) const;

    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * bias, ggml_tensor * blk_idx, ggml_tensor * blk_tail,
                       ggml_tensor * cell_vis, ggml_tensor * q_vis,
                       const llama_ubatch * ubatch, uint32_t ratio,
                       bool blk_bias,
                       int32_t * dst_derived_from = nullptr,
                       int32_t * dst_derived_lim  = nullptr) const;

    // F32 derived-cache view of indexer layer `il` ([idx_dim x n_blocks x n_stream]); nullptr
    // when the model carries no indexer or the layer is not a pool layer
    ggml_tensor * get_pool(ggml_context * ctx, int32_t il, uint32_t n_blocks) const;

private:
    llama_memory_hybrid_idx * mem = nullptr;

    // streams per ubatch, read from the slot infos before ctx_idx takes them
    // declared first, so it is initialised while sinfos_idx is still intact
    const std::vector<uint32_t> ns_ubatch;

    // the indexer cells of each ubatch, kept for pools in cache order (qwen4exp): token s*n + i of ubatch u
    // sits in cell idxs[s][i] of stream strm[s] of sinfos_kpool[u], and several cells can share a position
    const slot_info_vec_t sinfos_kpool;

    // null unless the model has an indexer
    const llama_memory_context_ptr ctx_idx;

    // mirrors the base class's ubatch cursor, which is private there
    size_t i_cur = 0;

    // Which pools of the layout this ubatch must re-pool. The layout itself belongs to the memory.
    struct kpool_state;
    kpool_state kpool_build_sizes() const;
    void kpool_build_state(const llama_ubatch & ubatch);
    const kpool_state & kpool_cur() const;

    // unique_ptr because kpool_state is incomplete here.
    std::unique_ptr<kpool_state> kpool_st;

    // The ubatch kpool_st was built for, guards against reads before apply.
    size_t i_kpool = SIZE_MAX;

    // Whether this context tracks k-pool states.
    bool kpool_track() const;

    // Positions each sequence must re-pool from, cleared only after the first ubatch succeeds
    llama_memory_hybrid_idx::stale_pos_t mem_idx_stale_batch = llama_memory_hybrid_idx::stale_pos_clean();
};
