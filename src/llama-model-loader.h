#pragma once

#include "llama.h"

#include "llama-impl.h"
#include "llama-arch.h"
#include "llama-hparams.h"
#include "llama-mmap.h"

#include "ggml-cpp.h"

#include <cstddef>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_map>

// lists of buffer types used for each layer
using buft_list_t = std::vector<std::pair<ggml_backend_dev_t, ggml_backend_buffer_type_t>>;

enum llama_fver {
    GGUF_FILE_VERSION_V1 = 1,
    GGUF_FILE_VERSION_V2 = 2,
    GGUF_FILE_VERSION_V3 = 3,
};

const char * llama_file_version_name(llama_fver version);

struct llama_model_loader {
    // Holds information on a model weight
    struct llama_tensor_weight {
        uint16_t  idx; // source file index
        size_t   offs; // tensor data offset in the original file

        ggml_tensor * tensor;

        llama_tensor_weight(const llama_file * file, uint16_t idx, const struct gguf_context * gguf_ctx, ggml_tensor * tensor) : idx(idx), tensor(tensor) {
            const int tensor_idx = gguf_find_tensor(gguf_ctx,  ggml_get_name(tensor));
            if (tensor_idx < 0) {
                throw std::runtime_error(format("tensor '%s' not found in the model", ggml_get_name(tensor)));
            }

            offs = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, tensor_idx);
            if (offs + ggml_nbytes(tensor) < offs || offs + ggml_nbytes(tensor) > file->size()) {
                throw std::runtime_error(format("tensor '%s' data is not within the file bounds, model is corrupted or incomplete", ggml_get_name(tensor)));
            }
        }
    };

    // custom comparator to sort weights more nicely by layer
    struct weight_name_comparer {
        bool operator()(const std::string & a, const std::string & b) const {
            int a_layer = -1;
            int b_layer = -1;
            sscanf(a.c_str(), "blk.%d.", &a_layer);
            sscanf(b.c_str(), "blk.%d.", &b_layer);
            if (a_layer != b_layer) {
                return a_layer < b_layer;
            }
            return a < b;
        }
    };

    static const int TENSOR_NOT_REQUIRED    = 1 << 0;
    static const int TENSOR_DUPLICATED      = 1 << 1;
    static const int TENSOR_SKIP            = 1 << 2;
    static const int TENSOR_SKIP_IF_VIRTUAL = 1 << 3;
    static const int TENSOR_ALLOW_RESHAPE   = 1 << 4;
    static const int TENSOR_READ_LAZY       = 1 << 5; // read rows on demand instead of loading whole tensor; requires mmap for now
    static const int TENSOR_SKIP_MANAGED    = 1 << 6; // like TENSOR_SKIP, but for a tensor whose data is read on demand elsewhere (managed lazy reader); no "unused tensor" warning

    int n_kv      = 0;
    int n_tensors = 0;
    int n_created = 0;

    uint64_t n_elements = 0;
    size_t   n_bytes    = 0;

    bool use_mmap = false;
    bool use_direct_io = false;
    // Host-resident expert weights (`-ncmoe`/`-cmoe`) are always pinned (`ROCm_Host`) unless the caller
    // selected `--host-experts pool`: in that mode the backend may offer a pageable, file-mapping-backed
    // master buffer type and every GPU read is served through the bounded pinned host pool (TODO #55).
    // The pageable `--host-experts mmap` mode was removed (issue #116).
    llama_host_experts_mode host_experts_mode = LLAMA_HOST_EXPERTS_MODE_PINNED;
    // wip/host-expert-tier: `-lme, --limit-host-experts` (-1 = unset, 0 = auto, >0 = GiB).
    int64_t host_experts_limit_gib = -1;
    // A shared host/device memory pool (an integrated GPU, e.g. gfx1151): "host" and "device" are ONE
    // physical pool, so a PINNED host expert master buys no capacity AND the tier that serves a host master
    // is denied on such a device (`moe-expert-cache.cu` refuses the arena on a shared pool), which leaves the
    // consumer reading a buffer the GPU cannot address -- a GPU page fault.  Measured on halo/gfx1151 (r236):
    // `-ncmoe 48` on Qwen3.8-Flash-Next GSQ-IQ3_XXS faulted in `mul_mat_q<Q2_0,16>` at a host address, and
    // `-ncmoe 24` too, while `-ncmoe 0` ran.
    // Set once from the model's device list before any placement decision.
    // wip/apu-expert-pool: the topology is not a verdict on the PLACEMENT MODE.  A `pool`/`auto` placement is
    // a different mechanism (the tensors stay in the read-only model mapping and every GPU read is served
    // through the bounded pinned pool), it is what makes a host expert set larger than the pool runnable,
    // and on a shared pool it is the *only* meaningful tier -- so it is allowed; only `pinned` is refused.
    // `MOE_HOST_EXPERTS_UMA=0` restores the blanket refusal (A/B and kill switch).
    bool host_experts_umem = false;         // the weights are going to a shared host/device pool
    bool host_experts_umem_refuse = false;  // ...and host-expert placement is refused on it (see above)
    // The resolved placement plan (see `plan_host_expert_placement`).  `host_experts_plan_done` false means
    // no plan was made and `host_expert_buft` keeps the historical mode-keyed decision.
    bool  host_experts_plan_done = false;
    std::set<std::string> host_experts_pinned;      // tensor names that get a pinned host master
    std::set<std::string> host_experts_holes;       // tensor names that stay in the model mapping
    size_t host_experts_total_bytes  = 0;           // Sigma of every host expert tensor the placement will host
    size_t host_experts_cap_bytes    = 0;           // the budget the plan may spend (SIZE_MAX = the whole set)
    size_t host_experts_pinned_bytes = 0;           // actual pinned bytes, accumulated at load
    size_t host_experts_holes_bytes  = 0;           // actual pageable bytes, accumulated at load
    bool   host_experts_pool_holes   = false;        // the holes are served by the bounded pinned pool
    std::string host_experts_policy;                // the resolved policy, for the census line
    bool   host_experts_heat_seen    = false;        // the heat prior answered for at least one tensor
    // issue #274: how many host-expert tensors actually got an on-disk source registered for the pool
    // (`moe_cache_set_host_source`), against how many were ELIGIBLE (a pageable, non-pinned expert bank).
    // Without a source the pool is never built for that tensor and every miss falls back to the per-table
    // bounce ring, so this pair is the first thing a "the pool never appears" report needs -- and it must be
    // visible at the default log level, not only under `-v`.
    size_t host_experts_sources      = 0;
    size_t host_experts_sourceable   = 0;

    // The buffer type a host-resident expert weight (`-ncmoe`/`-cmoe`, a `MUL_MAT_ID` table) should use on
    // `dev`: the pinned host buffer by default, or the backend's pool-backed master buffer in POOL mode
    // when the load is file-mapped (a non-mmap load has no mapping to place the tensors in).
    //
    // wip/host-expert-tier: `name` is the tensor's GGUF name.  With a placement plan (`plan_host_expert_
    // placement`) the decision is per TENSOR: the tensors inside `-lme`'s budget get the pinned master, the
    // rest keep the read-only model mapping and are served through the bounded pinned pool.  Without a plan
    // (a caller that never ran one, or a load with no mmap) this is the historical mode-keyed decision.
    ggml_backend_buffer_type_t host_expert_buft(ggml_backend_dev_t dev, const std::string & name) const;
    ggml_backend_buffer_type_t host_expert_buft(ggml_backend_dev_t dev) const {
        return host_expert_buft(dev, std::string());
    }

    // wip/host-expert-tier (the `-lme` budget): decide, before ANY host expert tensor is placed, how many
    // bytes of pinned host masters the budget buys, and WHICH tensors get them.
    //
    // The transport is uniform (a pinned master is the fastest at every device count: prefill 645.9/1 105.4
    // and decode 53.5/72.0 t/s at 1/2 devices against the pool's 616.6/707.1 and 49.0/51.3), but a pinned
    // master is a SECOND COPY of the bytes, so a set that does not fit in RAM must not be pinned wholesale.
    // The budget therefore buys whole tensors in DESCENDING hotness (the persisted heat prior, so the bytes
    // that are actually read are the ones that are pinned):
    //
    //   `--host-experts pinned` : the whole set (the historical meaning; no budget)
    //   `-lme auto`             : the whole set when it fits MOE_HOST_LIMIT_FRAC % of MemAvailable, else NO
    //                             pinned master -- every tensor stays in the model mapping and the bounded
    //                             pinned pool serves the reads (the not-fitting band, unchanged)
    //   `-lme N`                : N GiB of whole tensors, hottest first; the rest stays in the mapping
    //
    // Called from `llama_model_base::load_tensors` (it needs `n_layer_all`), after the buft lists exist and
    // before the arch creates any tensor.
    void plan_host_expert_placement(int n_layer_trunk);

    // Tell the loader that the weights are going to a shared host/device memory pool (an integrated GPU).
    // Called before `plan_host_expert_placement`.  A `pool`/`auto` placement is allowed there; `pinned` (and
    // `MOE_HOST_EXPERTS_UMA=0`) refuses every host-expert placement with one message.
    void set_host_experts_umem(bool on);

    // The `-ncmoe` tail rule / caller CPU-override match for one tensor (the placement the loader will make).
    bool host_expert_selected(const std::string & name, int layer, int n_layer_trunk) const;

    // True when this tensor keeps the read-only model mapping (a hole in the -lme budget).
    bool host_expert_is_hole(const std::string & name) const {
        return host_experts_plan_done && !name.empty() && host_experts_holes.count(name) > 0;
    }

    // True when this tensor got a pinned host master under the plan.
    bool host_expert_is_pinned(const char * name) const {
        return host_experts_plan_done && name != nullptr && host_experts_pinned.count(std::string(name)) > 0;
    }

    // Accumulate the actual tier bytes as the tensors are placed (the census line reports them).
    void account_host_expert_tier(const std::string & name, size_t nbytes) {
        if (!host_experts_plan_done) {
            return;
        }
        if (host_experts_pinned.count(name) > 0) {
            host_experts_pinned_bytes += nbytes;
        } else if (host_experts_holes.count(name) > 0) {
            host_experts_holes_bytes += nbytes;
        }
    }

    // The one-line census of the resolved tier, after the load: `pinned X MiB of Y MiB (cap C MiB),
    // pageable Z MiB`, plus which tensors were pinned and why, plus the pool-source count.
    // issue #274: this is WARN, not INFO.  `common_log_get_verbosity()` maps GGML_LOG_LEVEL_INFO to
    // LOG_LEVEL_TRACE, so an INFO line is INVISIBLE at the default verbosity (`-lv 3`) -- the reporter of
    // issue #274 could not see the tier census, the pool's "enabled"/fill-mode lines or the pool's per-table
    // build line at all, which is what made a pool that had FAILED look like a pool that was never requested.
    // It is a one-off decision line, the same class as `MoE expert cache (auto): arena ...` (also WARN).
    void report_host_expert_placement();
    bool check_tensors;
    bool no_alloc;
    bool load_mtp;

    // handle TENSOR_READ_LAZY
    // use case: keep PLE / engrams embd tensors on disk, read them on demand
    struct lazy_read {
        // set by the caller before the create_tensor() calls
        enum llama_lazy_mode mode = LLAMA_LAZY_MODE_OFF;

        // managed buffer size in bytes for the managed LAZY_MODE_AUTO PLE reader
        // (qwen4exp per_layer_token_embd); 0 = no managed reader (fall back to the mmap
        // lazy path / full load).  Set by the caller before the create_tensor() calls.
        size_t managed_budget = 0;

        // decide whether this tensor is read lazily
        // pass w to also record it, or nullptr to only ask
        bool add(const std::string & name, const ggml_tensor * t, const llama_tensor_weight * w);

        // record the on-disk range of a tensor whose data is loaded on demand by
        // other means (the managed lazy reader). Unlike add(), this does not
        // consult the lazy mode or the tensor size: the range must be excluded
        // from the load-time WILLNEED prefetch either way, or the kernel would
        // fault the whole table in
        void add_range(const std::string & name, const llama_tensor_weight & w);

        // record a tensor that must be loaded into host RAM even though the load is
        // file-mapped.  An explicit `--lazy-mode off` overrides the mmap-backed buffer for
        // the tensors the arch marks TENSOR_READ_LAZY (the PLE / engram tables): without
        // this the mapping wins and the table is only faulted in on access, so `off` does
        // not deliver the residency it promises.  The bulk of the model stays mapped, so
        // the load keeps mmap's speed.
        void add_resident(const std::string & name, const llama_tensor_weight & w);

        // --lazy-mode mlock: pull every recorded table into the page cache and pin it.  Must be
        // called once the mappings exist (the tables keep the lazy/mapped data path -- nothing is
        // copied, rows are still read from the mapping on demand -- only their residency is
        // guaranteed).  The lock is a property of this process: the kernel releases it when the
        // mapping is unmapped or the process exits, after which the pages are ordinary
        // reclaimable page cache again.
        void mlock_ranges(const llama_mmaps & mappings);

        // mlock objects created by mlock_ranges(): they must outlive the mappings, so the caller
        // moves them into the model (the loader itself is a temporary)
        llama_mlocks locks;

        bool any() const {
            // a `--lazy-mode mlock` range is deliberately not in `ranges` (no MADV_RANDOM), so the
            // recorded tensors count too: the table still needs the mapping under `--load-mode none`
            return !ranges.empty() || !tensors.empty();
        }

        bool has(const ggml_tensor * t) const {
            return tensors.count(ggml_get_name(t)) > 0;
        }

        // true for a tensor loaded into its own host buffer instead of the model mapping
        bool is_resident(const ggml_tensor * t) const {
            return resident.count(ggml_get_name(t)) > 0;
        }

        const llama_mmap::ranges & for_file(uint32_t idx) const {
            static const llama_mmap::ranges none;

            const auto it = ranges.find(idx);
            return it == ranges.end() ? none : it->second;
        }

        // lazy tensors are gathered on the host, so no offload setting applies to them
        static ggml_backend_buffer_type_t buft();

    private:
        // ranges excluded from the load-time prefetch (lazy and managed-reader tables)
        std::map<uint32_t, llama_mmap::ranges> ranges;
        std::set<std::string>                  tensors;
        // tensors read into a private host buffer (not served from the mapping)
        std::set<std::string>                  resident;
        // `--lazy-mode mlock`: on-disk ranges to populate and pin in the page cache.  Kept apart
        // from `ranges`, which is handed to llama_mmap and gets POSIX_MADV_RANDOM.
        std::map<uint32_t, llama_mmap::ranges> pin;
    } lazy;

    llama_files files;
    llama_ftype ftype;
    llama_fver  fver;

    llama_mmaps mappings;

    std::map<std::string, llama_tensor_weight, weight_name_comparer> weights_map;
    std::unordered_map<std::string, llama_model_kv_override> kv_overrides;
    const llama_model_tensor_buft_override * tensor_buft_overrides;
    // Allow a CPU expert override to match an appended nextn/MTP layer (the SPECULATIVE head).  False
    // for the main model so `-ncmoe`/`-cmoe` cannot offload the head; true for a separately loaded MTP
    // draft model so its own `--spec-draft-n-cpu-moe` is honoured.
    bool allow_nextn_cpu_offload = false;
    // `-ncmoe N` tail placement: N > 0 hosts the LAST N trunk layers `[n_layer - N, n_layer)` instead of
    // the caller's forward `blk.<i>` patterns (`LLAMA_HOST_EXPERTS_FROM_END=0` keeps N == 0).
    int  n_cpu_moe_from_end       = 0;

    gguf_context_ptr metadata_ptr;
    struct gguf_context * metadata; // either metadata_ptr.get() or externally set
    llama_model_set_tensor_data_t set_tensor_data;
    void * set_tensor_data_ud;
    std::vector<ggml_context_ptr> contexts;

    std::string arch_name;
    LLM_KV      llm_kv    = LLM_KV(LLM_ARCH_UNKNOWN);

    size_t size_done = 0;
    size_t size_data = 0;
    // exact byte ranges of each mapped file that are backed directly by a host-pointer (mmap) buffer.
    // A tensor duplicated host-side and device-side (a tied embedding) shares its range, so the device
    // copy must not unmap pages the host copy still points into.
    std::vector<llama_mmap::ranges> mmaps_used;

    // wip/moe-cache-autosize: bytes of host-resident MoE expert weights per device, accumulated by
    // `create_tensor` (works under `no_alloc`, unlike a scan of the allocated tensors).
    std::map<ggml_backend_dev_t, size_t> moe_host_expert_bytes;

    // define a comparator for the buft -> ctx map to ensure that the order is well-defined:
    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };

    // lazy and resident tensors need dedicated contexts
    struct ctx_key {
        ggml_backend_buffer_type_t buft;
        bool lazy;
        bool resident;
    };

    struct ctx_key_comparator {
        bool operator()(const ctx_key & lhs, const ctx_key & rhs) const {
            if (lhs.lazy != rhs.lazy) {
                return lhs.lazy < rhs.lazy;
            }
            if (lhs.resident != rhs.resident) {
                return lhs.resident < rhs.resident;
            }
            const int c = strcmp(ggml_backend_buft_name(lhs.buft), ggml_backend_buft_name(rhs.buft));
            if (c != 0) {
                return c < 0;
            }
            // Distinct buffer types can share a name (e.g. the per-device pinned host buffers: one
            // name, a per-device `device`).  They must not share a context, or every device's host
            // tensors land in the first device's buffer (and so in its ops' device).
            ggml_backend_dev_t lhs_dev = ggml_backend_buft_get_device(lhs.buft);
            ggml_backend_dev_t rhs_dev = ggml_backend_buft_get_device(rhs.buft);
            const char * lhs_name = lhs_dev ? ggml_backend_dev_name(lhs_dev) : "";
            const char * rhs_name = rhs_dev ? ggml_backend_dev_name(rhs_dev) : "";
            return strcmp(lhs_name, rhs_name) < 0;
        }
    };

    std::map<ctx_key, ggml_context_ptr, ctx_key_comparator> ctx_map;

    // track tensors that had to be moved for debugging:
    size_t n_tensors_moved = 0;
    std::string first_tensor_moved_name;
    std::string first_tensor_moved_type_name;
    ggml_backend_buffer_type_t first_moved_from_buft = nullptr;
    ggml_backend_buffer_type_t first_moved_to_buft = nullptr;

    llama_model_loader(
        struct gguf_context * metadata,
        llama_model_set_tensor_data_t set_tensor_data,
        void * set_tensor_data_ud,
        const std::string & fname,
        std::vector<std::string> & splits, // optional, only need if the split does not follow naming scheme
        FILE * file,
        llama_load_mode load_mode,
        llama_host_experts_mode host_experts_mode,
        int64_t host_experts_limit_gib,
        bool check_tensors,
        bool no_alloc,
        bool load_mtp,
        bool allow_nextn_cpu_offload,
        int  n_cpu_moe_from_end,
        const llama_model_kv_override * param_overrides_p,
        const llama_model_tensor_buft_override * param_tensor_buft_overrides_p);

    template<typename T>
    typename std::enable_if<std::is_integral<T>::value, bool>::type
    get_arr_n(const std::string & key, T & result, bool required = true);

    template<typename T>
    typename std::enable_if<std::is_integral<T>::value, bool>::type
    get_arr_n(enum llm_kv kid, T & result, bool required = true);

    template<typename T>
    bool get_arr(const std::string & key, std::vector<T> & result, bool required = true);

    template<typename T, size_t N_MAX>
    bool get_arr(const std::string & key, std::array<T, N_MAX> & result, bool required = true);

    template<typename T>
    bool get_arr(enum llm_kv kid, T & result, bool required = true);

    template<typename T>
    bool get_key(const std::string & key, T & result, bool required = true);

    template<typename T>
    bool get_key(enum llm_kv kid, T & result, bool required = true);

    template<typename T, size_t N_MAX>
    bool get_key_or_arr(const std::string & key, std::array<T, N_MAX> & result, uint32_t n, bool required = true);

    template<typename T>
    bool get_key_or_arr(enum llm_kv kid, T & result, uint32_t n, bool required = true);

    bool get_key_or_arr(enum llm_kv kid, uint32_t & result, bool required = true);

    std::string get_arch_name() const;

    enum llm_arch get_arch() const;

    const llama_tensor_weight * get_weight(const char * name) const;

    const llama_tensor_weight & require_weight(const char * name) const;

    struct ggml_tensor * get_tensor_meta(const char * name) const;

    struct ggml_tensor * require_tensor_meta(const std::string & name) const;

    const struct ggml_tensor * check_tensor_dims(
            const std::string & name,
            const std::vector<int64_t> & ne,
            bool required,
            bool allow_reshape) const;

    struct ggml_tensor * create_tensor(
        const llama_hparams & hparams, const buft_list_t * buft_list_cpu, const buft_list_t * buft_list_input, const buft_list_t * buft_list_output,
        const buft_list_t * buft_list_layer, const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags);

    void done_getting_tensors(bool partial = false) const;

    void init_mappings(bool prefetch = true, llama_mlocks * mlock_mmaps = nullptr);

    // byte range within a mapped file
    struct mapping_range {
        size_t first; // inclusive
        size_t last;  // exclusive
    };

    // sorted, non-overlapping ranges of file 'idx' that back the tensors of 'ctx'
    // ranges owned by other contexts are excluded, so that a buffer wrapping them won't pin
    // data that this context never reads (lazy tensors, -cmoe tensors, ...)
    std::vector<mapping_range> get_mapping_ranges(uint32_t idx, ggml_context * ctx) const;

    // record an exact mmap-backed byte range of file `idx` (a tensor held in a host-pointer buffer)
    void mark_mmap_used(uint32_t idx, size_t offs, size_t size);

    // release a weight's mmap pages
    void unmap_weight(const llama_tensor_weight & w) const;

    // read a byte range of a weight's data
    // with mmap, returns a pointer into the mapping, otherwise reads into buf and returns buf
    const void * load_data_range(const llama_tensor_weight & w, size_t offs, size_t size, void * buf) const;

    // Returns false if cancelled by progress_callback
    bool load_all_data(
            struct ggml_context * ctx,
            std::vector<ggml_backend_buffer_t> & bufs,
            llama_mlocks * lmlocks,
            llama_progress_callback progress_callback,
            void * progress_callback_user_data);

    std::string ftype_name() const;

    void print_info() const;
};
