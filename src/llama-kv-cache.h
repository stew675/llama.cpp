#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-kv-cells.h"
#include "llama-memory.h"

#include <map>
#include <unordered_map>
#include <vector>

struct llama_cparams;
struct llama_hparams;
struct llama_model;
struct llama_context;

//
// llama_kv_cache
//

class llama_kv_cache : public llama_memory_i {
public:
    struct stream_copy_info {
        bool empty() const {
            assert(ssrc.size() == sdst.size());
            return ssrc.empty();
        }

        std::vector<uint32_t> ssrc;
        std::vector<uint32_t> sdst;
    };

    // for each ubatch, create a slot_info that contains information about where the ubatch should be inserted in the
    //   KV cells. for example, cell indices for each token, such that: token[i] -> goes to cells[idxs[i]]
    struct slot_info {
        // data for ggml_set_rows
        using idx_vec_t = std::vector<uint32_t>;

        // number of streams: ns = s1 - s0 + 1
        uint32_t s0;
        uint32_t s1;

        std::vector<llama_seq_id> strm; // [ns]
        std::vector<idx_vec_t>    idxs; // [ns]

        uint32_t head() const {
            GGML_ASSERT(idxs.size() == 1);
            GGML_ASSERT(!idxs[0].empty());

            return idxs[0][0];
        }

        void resize(size_t n) {
            strm.resize(n);
            idxs.resize(n);
        }

        size_t size() const {
            GGML_ASSERT(idxs.size() == strm.size());
            GGML_ASSERT(!idxs.empty());

            return idxs[0].size();
        }

        size_t n_stream() const {
            return strm.size();
        }

        bool empty() const {
            return idxs.empty();
        }

        void clear() {
            idxs.clear();
        }

        // check if indices are contiguous starting from head()
        bool is_contiguous() const {
            if (idxs.empty() || idxs[0].empty()) {
                return true;
            }
            if (idxs.size() > 1) {
                return false;
            }
            const uint32_t h = idxs[0][0];
            for (size_t i = 0; i < idxs[0].size(); ++i) {
                if (idxs[0][i] != h + i) {
                    return false;
                }
            }
            return true;
        }
    };

    using slot_info_vec_t = std::vector<slot_info>;

    // TODO: refactor the memory instances to not depend on `llama_model`
    //       instead pass all necessary info (e.g. hparams, dev layers, arch, etc.) directly
    //       likely through `struct llama_memory_params`
    llama_kv_cache(
            const llama_model & model,
          const llama_hparams & hparams,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   unified,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_pad,
                     uint32_t   n_swa,
               llama_swa_type   swa_type,
               llama_memory_t   mem_other,
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse,
        const  layer_share_cb & share,
        // a model can hold more than one cache, so the tensor names have to stay unique
                 const char *   name_tag = "",
        // set to false for keys-only caches (e.g. the qwen4exp QSA indexer store):
        // no V tensor is allocated, and no V-side op may be issued against the cache
                         bool   v_enabled = true,
        // wip/kv-residency (TODO #118) B-A: may this cache take the ATTENDANCE POOL
        // (`LLAMA_KV_REGION_POOL`)?  Only a cache whose decode read is the sparse QSA arm can use it;
        // the MTP draft's attention is dense, so the draft cache must not take it.
                         bool   kv_pool_capable = false);

    ~llama_kv_cache();

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    // wip/kv-residency (TODO #117): map the growable KV region so the first `n_kv` cells of every KV tensor are
    // backed by physical memory.  `n_kv` must be the value the graph views use -- the PADDED one from
    // `get_n_kv()` -- because a kernel that reads a cell above the mapped extent faults the GPU rather than
    // returning an error.  A no-op unless the KV is region-backed (LLAMA_KV_REGION).
    void grow_regions(uint32_t n_kv) const;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;

    // parking (unified cache only): the cells of an idle sequence move to the top of the cache, stacked below the
    // sequences parked before it, and new cells are only taken below the lowest parked cell.  get_n_kv() stops below it
    // too, so a sequence in the low cells attends over the same range, at the same cell indices, as in a cache that
    // holds nothing else -- its results do not depend on what is parked.  seq_unpark() moves a parked sequence back
    // to the lowest free run.  Both return false, changing nothing, when the move is not possible (not unified, the
    // cells are shared or not contiguous, no room).  The K/V data are copied on the device.
    bool seq_park  (llama_seq_id seq_id);
    bool seq_unpark(llama_seq_id seq_id);
    bool seq_parked(llama_seq_id seq_id) const;

    // whether this cache supports parking at all: one stream, cells it does not share, V not transposed, no sliding window
    bool can_park() const;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    //
    // llama_kv_cache specific API
    //

    uint32_t get_size()     const;
    uint32_t get_n_seq_max() const;
    uint32_t get_n_stream() const;

    // wip/kv-residency (TODO #118) B2: is this cache's KV STREAMING at the extent `n_kv` -- i.e. is a
    // resident window active and has the used extent passed it, so that the tail of every tensor lives in
    // pinned host RAM?  A read of a spilled cell comes over the PCIe link (~55 GB/s) instead of VRAM
    // (~565 GB/s), which is what makes the attention ARM worth switching (a streamed cache prefers the
    // sparse QSA decode, a resident one the dense arm; wip/kv-residency/B1-STRATA-PLAN.md §3.2).
    bool is_streaming(uint32_t n_kv) const;

    // wip/kv-residency (TODO #118) B-A: the ATTENDANCE POOL.
    //
    // The attention KV is kept twice: the MIRROR (`layers[].k`/`v`, pinned host memory, authoritative and
    // cell-indexed -- every existing path, the writers included, works on it unchanged) and a compact
    // DEVICE POOL of `n_pages` pages of `page_cells` cells each (`page_cells` == one indexer block, so the
    // selection's own granularity), with a page table `tbl[block] -> slot`.  A decode step resolves its
    // selection through the table (`build_kv_pool_resolve`): the missed pages are copied from the mirror,
    // and the reader gathers K/V at the pool slots (`ggml_flash_attn_qsa_set_kv_idx`).  See
    // `wip/kv-residency/BA-POOL-DESIGN.md`; the writer rule (mirror always, no invalidation ever) is in
    // `ggml/src/ggml-cuda/kv-pool.cu`.
    bool has_kv_pool() const;
    uint32_t kv_pool_page_cells() const;
    // the block count one selection can span, i.e. the widest pool a single attention row needs
    uint32_t kv_pool_blocks_per_row() const;
    // may a resolve of `n_rows` selections be served without evicting a page it just claimed?
    bool kv_pool_serves(uint32_t n_rows) const;
    // the compact pool's K/V views (same 4-D layout as get_k/get_v, over the pool's cells)
    ggml_tensor * get_k_pool(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_v_pool(ggml_context * ctx, int32_t il) const;
    // the page-table resolve for one layer's selection: dst = the selected cells' POOL positions
    ggml_tensor * build_kv_pool_resolve(ggml_context * ctx, ggml_tensor * ids, ggml_tensor * wids, int32_t il) const;

    bool get_has_shift() const;

    ggml_type type_k() const;
    ggml_type type_v() const;

    std::vector<uint32_t> get_layer_ids() const;
    ggml_tensor * get_k_storage(int32_t il) const;

    const llama_kv_cells & get_cells(llama_seq_id seq_id) const;

    // The stream holding seq_id's cells.
    uint32_t get_stream(llama_seq_id seq_id) const;

    // state_read, plus the cells the restored tokens were placed in
    // a cache that mirrors another one (the qwen4exp indexer) must not search for its own cells: two searches agree only by luck
    //   sinfos_out: if set, filled with the layout used; a stream with no cells leaves an empty entry
    //   sinfos_in : if set, the layout to use instead of searching. one entry per stream, cell count must match the blob
    void state_read_sinfo(
            llama_io_read_i & io,
               llama_seq_id   seq_id,
      llama_state_seq_flags   flags,
          slot_info_vec_t *   sinfos_out,
    const slot_info_vec_t *   sinfos_in);

    // undo a state_read() of seq_id (-1 for the whole cache) that another memory module failed to complete
    void state_clear(llama_seq_id seq_id);

    //
    // graph_build API
    //

    uint32_t get_n_kv(const slot_info & sinfo, uint32_t n_tokens = 0) const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;

    // store k_cur and v_cur in the cache based on the provided head location
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const;

    //
    // preparation API
    //

    // find places for the provided ubatches in the cache, returns the slot infos
    // return empty vector on failure
    slot_info_vec_t prepare(const std::vector<llama_ubatch> & ubatches);

    bool update(llama_context * lctx, bool do_shift, const stream_copy_info & sc_info);

    // find a slot of kv cells that can hold the ubatch
    // if cont == true, then the slot must be continuous
    // return empty slot_info on failure
    slot_info find_slot(const llama_ubatch & ubatch, bool cont) const;

    // emplace the ubatch context into slot: [sinfo.idxs[0...ubatch.n_tokens - 1]]
    void apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch);

    //
    // input API
    //

    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;
    ggml_tensor * build_input_v_rot(ggml_context * ctx) const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;

    void set_input_k_shift(ggml_tensor * dst) const;

    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    // V3 derived kq mask: true when the packed mask for this batch can be replaced by the compact
    // per-cell state (see set_input_kq_derived).  The predicate is exactly the shape the phase-1
    // host oracle proved bit-exact against the packed fill - extend it only with new coverage.
    bool kq_mask_derivable(const llama_ubatch & ubatch) const;

    // the derived kq mask over several sequences (LLAMA_KQ_MASK_DERIVED_MULTISEQ, on unless 0): each sequence of the
    // batch gets its own position range, so it needs every cell to carry at most one of `seqs` and every position below
    // the range stride (see kq_mask_derivable_seqs in the .cpp)
    static bool kq_mask_derived_multiseq_enabled();
    // a batch of several sequences takes the derived form only above this many tokens: the multi-slot verify batches below
    // it (2 x 6, 4 x 6, 2 x 16 rows) keep the packed mask, whose few MiB the reserve covers, and the per-ubatch checks
    static constexpr uint32_t KQ_DERIVED_MULTISEQ_MIN_TOKENS = 32;
    bool kq_mask_derivable_seqs(const llama_seq_id * seqs, uint32_t n_seqs, llama_pos pos_max) const;
    void set_input_kq_derived(
            ggml_tensor * cell_pos, ggml_tensor * tok_lo, ggml_tensor * tok_hi,
            const llama_ubatch * ubatch, bool causal_attn) const;

    void set_input_k_rot(ggml_tensor * dst) const;
    void set_input_v_rot(ggml_tensor * dst) const;

    // true if llama_kv_cell_ext holds information that has to survive a state save/restore
    bool has_cell_ext() const;

    // for every token of the ubatch, the ids of the n tokens that precede it in its sequence
    // example for M-RoPE image case: tokens A B X X X C, where X is a 3-token image at pos 2 spanning positions 2..4:
    //   tok: A B X X X C
    //   pos: 0 1 2 2 2 5
    //   prev, n=2: A -> [NULL, NULL], B -> [NULL, A], 3rd X -> [X, X], C -> [X, X]
    // note: used by n-gram input embeddings
    void get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const;

private:
    const llama_model & model;
    const llama_hparams & hparams;

    struct kv_layer {
        // layer index in the model
        // note: can be different from the layer index in the KV cache
        uint32_t il;

        ggml_tensor * k;
        ggml_tensor * v;

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;
    };

    bool v_trans = true;  // the value tensor is transposed

    const uint32_t n_seq_max = 1;
    const uint32_t n_stream  = 1;

    // required padding
    const uint32_t n_pad = 1;

    // env: LLAMA_KV_N_PAD_PREFILL (0 when inactive)
    const uint32_t n_pad_prefill = 0;

    // SWA
    const uint32_t n_swa = 0;

    // env: LLAMA_ATTN_ROT_DISABLE
    uint32_t n_rot_k = 0;
    uint32_t n_rot_v = 0;

    // if all layers participating in the cache have constant head size, the value is stored here
    // otherwise the value is -1
    int32_t n_embd_head_k_all = 0;
    int32_t n_embd_head_v_all = 0;

    // pre-computed hadamard martrices
    std::unordered_map<int64_t, std::vector<float>> attn_rot_hadamard;

    // env: LLAMA_KV_CACHE_DEBUG
    int debug = 0;

    // this is the SWA type of the cache - not to be confused with the model SWA type
    const llama_swa_type swa_type = LLAMA_SWA_TYPE_NONE;

    // ggml contexts for the KV cache along with the allocated backend buffers:
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;

    // wip/kv-residency (TODO #117): is a buffer of this cache backed by a growable VMM region (provider P1)?
    // Set once at construction from the backend's answer, and `grow_regions()` is then called with the extent
    // the graph is about to read/write.  Zero cost (and no proc lookup) when the feature is off.
    bool kv_region = false;

    // wip/kv-residency (TODO #118) A2/A3: the windowed KV's resident window, in CELLS (0 = no window).
    // `memory_breakdown()` reports only the resident part against the device buffer type, because that is
    // all the buffer ever COMMITS -- the rest is pinned host memory.
    size_t kv_resident_cells = 0;

    // wip/kv-residency (TODO #118) B-A: the attendance pool (see has_kv_pool()).  `pool_layers` is
    // parallel to `layers`; `layers[].k`/`v` are the MIRROR in this mode.
    struct kv_pool_layer {
        ggml_tensor * pool_k = nullptr;   // [n_embd_k_gqa, page_cells*n_pages]  device
        ggml_tensor * pool_v = nullptr;
        ggml_tensor * tbl    = nullptr;   // I32 [n_blocks]  block -> slot | -1
        ggml_tensor * inv    = nullptr;   // I32 [n_pages]   slot -> block | -1
        ggml_tensor * use    = nullptr;   // I32 [n_pages]   the epoch that last marked the slot in use
    };
    std::vector<kv_pool_layer> pool_layers;
    ggml_tensor * pool_ctr = nullptr;      // I32 [8] device counters (claim cursor, miss list, ...)
    uint32_t kv_pool_page       = 0;       // page size in cells (0 = pool off)
    uint32_t kv_pool_n_pages    = 0;
    uint32_t kv_pool_n_blocks   = 0;
    uint32_t kv_pool_blocks_row = 0;       // blocks one selection can span

    // drop every residency claim (the pages stay valid copies, they just stop being authoritative for
    // their block).  Needed by any path that rewrites cells in place WITHOUT going through the writers:
    // the K-shift (rope/per-token moves) and a state restore.
    void kv_pool_invalidate();

    // the current index from where we start searching for a free slot in the ring buffer of KV cells (see find_slot())
    // note: this is not part of the KV state and it's only used to speed-up the find_slot() method
    std::vector<uint32_t> v_heads;

    // TODO: temporary until we refactor to be able to share the same cells between 2 kv caches [TAG_KV_CACHE_SHARE_CELLS]
    llama_kv_cache * other;

    std::shared_ptr<llama_kv_cells_vec> v_cells_impl;

    llama_kv_cells_vec & v_cells;

    // maps from a sequence id to a stream id
    std::vector<uint32_t> seq_to_stream;

    // pending stream copies that will be applied during the next update
    stream_copy_info sc_info;

    std::vector<kv_layer> layers;

    // parked sequences (stream 0 of a unified cache): [first cell, last cell + 1), see seq_park()
    std::map<llama_seq_id, std::pair<uint32_t, uint32_t>> parked;

    // the lowest parked cell, or the cache size
    uint32_t park_bound() const;

    // drops parked entries whose sequence is gone and narrows the others to the cells they still hold
    void park_refresh();

    // moves cells [src, src + n) to [dst, dst + n) on stream 0, K/V data and cell state, overlap-safe
    void park_move(uint32_t src, uint32_t dst, uint32_t n);

    // moves the parked sequences up so that they sit together at the top of the cache, in their order: a sequence that
    // leaves the stack would otherwise leave a hole that the next one to come down cannot use
    void park_compact();

    // model layer id -> KV cache layer id
    std::unordered_map<int32_t, int32_t> map_layer_ids;

    size_t total_size() const;

    size_t size_k_bytes() const;
    size_t size_v_bytes() const;

    ggml_tensor * build_rope_shift(
            const llama_cparams & cparams,
                   ggml_context * ctx,
                    ggml_tensor * cur,
                    ggml_tensor * shift,
                    ggml_tensor * rot,
                    ggml_tensor * factors,
                          float   freq_base,
                          float   freq_scale,
                       uint32_t   il) const;

    ggml_cgraph * build_graph_shift(
               llm_graph_result * res,
                  llama_context * lctx) const;

    struct cell_ranges_t {
        uint32_t strm;

        std::vector<std::pair<uint32_t, uint32_t>> data; // ranges, from inclusive, to exclusive
    };

    void state_write_meta(llama_io_write_i & io, const cell_ranges_t & cr, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const cell_ranges_t & cr) const;

    // sinfo_in, when set, replaces the find_slot call: the cells are given by the caller
    bool state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count,       slot_info & sinfo, llama_seq_id dest_seq_id = -1, const slot_info * sinfo_in = nullptr);
    bool state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo);

    void state_clear(llama_seq_id seq_id, uint32_t strm, const slot_info & sinfo);
};

class llama_kv_cache_context : public llama_memory_context_i {
public:
    // some shorthands
    using slot_info_vec_t  = llama_kv_cache::slot_info_vec_t;
    using stream_copy_info = llama_kv_cache::stream_copy_info;

    // used for errors
    llama_kv_cache_context(llama_memory_status status);

    // used to create a full-cache context
    llama_kv_cache_context(
            llama_kv_cache * kv);

    // used to create an update context
    llama_kv_cache_context(
            llama_kv_cache * kv,
            llama_context * lctx,
            bool do_shift,
            stream_copy_info sc_info);

    // used to create a batch processing context from a batch
    llama_kv_cache_context(
            llama_kv_cache * kv,
            slot_info_vec_t sinfos,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_kv_cache_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_kv_cache_context specific API
    //

    uint32_t get_n_kv() const;

    // wip/kv-residency (TODO #118) B2: see llama_kv_cache::is_streaming
    bool is_streaming(uint32_t n_kv) const;

    // wip/kv-residency (TODO #118) B-A: see llama_kv_cache::has_kv_pool
    bool has_kv_pool() const;
    // the K/V views of the compact pool (same 4-D layout as get_k/get_v, n_kv = the pool's cells)
    ggml_tensor * get_k_pool(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_v_pool(ggml_context * ctx, int32_t il) const;
    // emit the page-table resolve for this layer's selection: dst = the selected cells' POOL positions.
    // `ids` is the selection (cell ids, the shape indexer_top_k returns), `wids` the cells this batch just
    // WROTE (the pages holding them are re-filled from the mirror), or nullptr.
    ggml_tensor * build_kv_pool_resolve(ggml_context * ctx, ggml_tensor * ids, ggml_tensor * wids, int32_t il) const;
    bool kv_pool_serves(uint32_t n_rows) const;

    ggml_type type_k() const;
    ggml_type type_v() const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il) const;

    // store k_cur and v_cur in the cache based on the provided head location
    // note: the heads in k_cur and v_cur should be laid out contiguously in memory
    //   - k_cur  [n_embd_head_k, n_head_k, n_tokens]
    //   - k_idxs [n_tokens]
    //   - v_cur  [n_embd_head_v, n_head_v, n_tokens]
    //   - v_idxs [n_tokens] or [n_tokens*n_embd_v_gqa] depending if V cache is transposed
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const;

    // create destination indices for each head of the current batch for where it would be written in the KV cache
    // the indices address the global KV cache (not per stream) - this is not relevant for the user of this API, but
    //   helps understand the implementation logic of cpy_k and cpy_v
    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;
    ggml_tensor * build_input_v_rot(ggml_context * ctx) const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    void set_input_k_shift   (ggml_tensor * dst) const;
    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    // V3 derived kq mask - see llama_kv_cache::kq_mask_derivable
    bool kq_mask_derivable(const llama_ubatch & ubatch) const;
    void set_input_kq_derived(
            ggml_tensor * cell_pos, ggml_tensor * tok_lo, ggml_tensor * tok_hi,
            const llama_ubatch * ubatch, bool causal_attn) const;

    void set_input_k_rot(ggml_tensor * dst) const;
    void set_input_v_rot(ggml_tensor * dst) const;

    // see llama_kv_cache::get_prev_tokens()
    void get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const;

private:
    llama_memory_status status;

    llama_kv_cache * kv;
    llama_context * lctx;

    //
    // update context
    //

    bool do_shift = false;

    stream_copy_info sc_info;

    //
    // batch processing context
    //

    // the index of the cur ubatch to process
    size_t i_cur = 0;

    slot_info_vec_t sinfos;

    std::vector<llama_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    //

    // a heuristic, to avoid attending the full cache if it is not yet utilized
    // as the cache gets filled, the benefit from this heuristic disappears
    int32_t n_kv;
};
