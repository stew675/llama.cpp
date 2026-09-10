#pragma once

// ggml-backend internal header

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

    #define GGML_BACKEND_API_VERSION 3

    //
    // Backend buffer type
    //

    struct ggml_backend_buffer_type_i {
        const char *          (*get_name)        (ggml_backend_buffer_type_t buft);
        // allocate a buffer of this type
        ggml_backend_buffer_t (*alloc_buffer)    (ggml_backend_buffer_type_t buft, size_t size);
        // (optional) allocate tensors from a list into a buffer of this type (defaults to alloc_buffer + linear allocator)
        ggml_backend_buffer_t (*alloc_buffer_n)  (ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors);
        // tensor alignment
        size_t                (*get_alignment)   (ggml_backend_buffer_type_t buft);
        // (optional) max buffer size that can be allocated (defaults to SIZE_MAX)
        size_t                (*get_max_size)    (ggml_backend_buffer_type_t buft);
        // (optional) data size needed to allocate the tensor, including padding (defaults to ggml_nbytes)
        size_t                (*get_alloc_size)  (ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor);
        // (optional) total data size needed to allocate the given tensors, including padding and splitting (defaults to per-tensor get_alloc_size)
        size_t                (*get_alloc_size_n)(ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors);
        // (optional) check if tensor data is in host memory and uses standard ggml tensor layout (defaults to false)
        bool                  (*is_host)         (ggml_backend_buffer_type_t buft);
        // (optional) APPENDED, so the positional iface initializers elsewhere in the tree stay valid and
        // every backend that does not set it is zero-initialized: extra slack, in percent, added to each
        // COMPUTE buffer allocation (defaults to 0 = no size change at all).  The graph allocator sizes
        // the compute buffer from a *measure* graph, but a runtime graph can have a different live-tensor
        // set and need a little more; growing it is a free-then-allocate-larger, so it needs a contiguous
        // block BIGGER than the one just released, which fails once free VRAM belongs to a low-priority
        // cache.  A backend that can be that tight opts in so the slack is taken while memory is plentiful.
        size_t                (*get_compute_margin_pct)(ggml_backend_buffer_type_t buft);
        // (optional) APPENDED like the field above.  Allocate a buffer knowing how it will be used, so a
        // backend can pick a different allocator for the COMPUTE buffer than for model weights.  The
        // graph allocator is the only caller and it always knows the usage.  Defaults to alloc_buffer.
        ggml_backend_buffer_t (*alloc_buffer_usage)(ggml_backend_buffer_type_t buft, size_t size, enum ggml_backend_buffer_usage usage);
        // (optional) APPENDED.  Uniform chunk size, in bytes, for the COMPUTE buffer, 0 = unset.  When set
        // the graph allocator rounds each compute-buffer allocation up to whole chunks and adds ONE spare
        // chunk, so a later graph whose layout grows within that chunk does not trigger a
        // free-then-allocate-larger -- the re-alloc only fires when the layout crosses the next chunk
        // high-water mark.  That makes the dangerous free/alloc (and any arena yield it causes) rare
        // instead of per-ubatch, and the spare chunk doubles as the over-read guard at the buffer's end.
        // When 0, `get_compute_margin_pct` still applies.
        size_t                (*get_compute_chunk_bytes)(ggml_backend_buffer_type_t buft);
    };

    struct ggml_backend_buffer_type {
        struct ggml_backend_buffer_type_i  iface;
        ggml_backend_dev_t device;
        void * context;
    };

    // [TAG_ALLOC_SIZE_EXPAND]
    // returns true for ops that may require additional memory for fleeting data on some backends,
    // i.e. the backend buffer type's get_alloc_size may return more than ggml_nbytes for the output tensor
    GGML_API bool ggml_op_alloc_size_may_expand(enum ggml_op op);

    //
    // Backend buffer
    //

    struct ggml_backend_buffer_i {
        // (optional) free the buffer
        void         (*free_buffer)  (ggml_backend_buffer_t buffer);
        // base address of the buffer
        void *       (*get_base)     (ggml_backend_buffer_t buffer);
        // (optional) initialize a tensor in the buffer (eg. add tensor extras)
        enum ggml_status (*init_tensor)(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);
        // tensor data access
        void         (*memset_tensor)(ggml_backend_buffer_t buffer,       struct ggml_tensor * tensor,     uint8_t value, size_t offset, size_t size);
        void         (*set_tensor)   (ggml_backend_buffer_t buffer,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
        void         (*get_tensor)   (ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
        // (optional) 2d data copies
        void         (*set_tensor_2d)(ggml_backend_buffer_t buffer,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
        void         (*get_tensor_2d)(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);

        // (optional) tensor copy: dst is in the buffer, src may be in any buffer, including buffers from a different backend (return false if not supported)
        bool         (*cpy_tensor)   (ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst);
        // clear the entire buffer
        void         (*clear)        (ggml_backend_buffer_t buffer, uint8_t value);
        // (optional) reset any internal state due to tensor initialization, such as tensor extras
        void         (*reset)        (ggml_backend_buffer_t buffer);
    };

    struct ggml_backend_buffer {
        struct ggml_backend_buffer_i  iface;
        ggml_backend_buffer_type_t    buft;
        void * context;
        size_t size;
        enum ggml_backend_buffer_usage usage;
    };

    GGML_API ggml_backend_buffer_t ggml_backend_buffer_init(
                   ggml_backend_buffer_type_t buft,
            struct ggml_backend_buffer_i      iface,
                   void *                     context,
                   size_t                     size);

    // do not use directly, use ggml_backend_tensor_copy instead
    GGML_API bool ggml_backend_buffer_copy_tensor(const struct ggml_tensor * src, struct ggml_tensor * dst);

    // multi-buffer
    // buffer that contains a collection of buffers
    GGML_API ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers);
    GGML_API bool                  ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer);
    GGML_API void                  ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage);
    GGML_API void                  ggml_backend_meta_buffer_set_usage (ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage);

    //
    // Backend (meta)
    //

    GGML_API bool ggml_backend_is_meta       (ggml_backend_t backend);
    GGML_API bool ggml_backend_buffer_is_meta(ggml_backend_buffer_t buf);
    GGML_API bool ggml_backend_buft_is_meta  (ggml_backend_buffer_type_t buft);
    // Optional per-buffer-type compute-buffer slack in percent; 0 when the buffer type does not opt in.
    GGML_API size_t ggml_backend_buft_get_compute_margin_pct(ggml_backend_buffer_type_t buft);
    // Allocate a buffer for the given usage; falls back to alloc_buffer when the type does not care.
    GGML_API ggml_backend_buffer_t ggml_backend_buft_alloc_buffer_usage(ggml_backend_buffer_type_t buft, size_t size, enum ggml_backend_buffer_usage usage);
    // Optional uniform COMPUTE-buffer chunk size in bytes; 0 when the type does not opt in.
    GGML_API size_t ggml_backend_buft_get_compute_chunk_bytes(ggml_backend_buffer_type_t buft);

    GGML_API size_t         ggml_backend_meta_n_backends    (ggml_backend_t meta_backend);
    GGML_API ggml_backend_t ggml_backend_meta_simple_backend(ggml_backend_t meta_backend, size_t index);

    //
    // Backend (stream)
    //

    // passed to graph_optimize so the backend can add allocation dependencies:
    // if the backend executes parts of the graph out of order (e.g. on concurrent streams),
    // it must keep the affected tensors allocated until a node where execution is known to have joined
    struct ggml_backend_graph_optimize_params {
        // keep `tensor` allocated at least until `until` (a node of the same graph) has been computed
        // can be called multiple times for the same tensor: the longest lifetime applies
        void (*add_alloc_dep)(void * user_data, struct ggml_tensor * tensor, struct ggml_tensor * until);
        void * user_data;
        // the scheduler will run the graph in sub-graphs split at every eval-callback node, so a
        // backend optimisation that relies on state living for one whole compute (e.g. eliding a
        // producer's F32 output and re-reading it from a per-graph cache) must stand down
        bool has_eval_callback;
        // the *whole* scheduled graph.  graph_optimize is called per split, but a consumer of a
        // tensor may live in another split (the per-compute activation cache does not span splits),
        // so a backend optimisation that elides a producer's output must consult the whole graph to
        // find every consumer.  NULL means the graph is not split (full_graph == the split graph).
        const struct ggml_cgraph * full_graph;
        // Two-pass mode used by the meta (tensor-split) backend.  The meta owns the whole graph, so
        // the scheduler never calls a child backend's graph_optimize; the meta forwards it twice
        // instead -- once over the whole graph before allocation (alloc-deps only) and once per
        // per-device subgraph of simple tensors after allocation (markings only), because the marks
        // must be keyed by the tensor pointers the child's compute actually sees.  Both default to
        // false (the single-backend scheduler runs the complete pass in one call).
        bool marks_only;
        bool allocs_only;
    };

    struct ggml_backend_i {
        const char * (*get_name)(ggml_backend_t backend);

        void (*free)(ggml_backend_t backend);

        // (optional) asynchronous tensor data access
        void (*set_tensor_async)   (ggml_backend_t backend,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
        void (*get_tensor_async)   (ggml_backend_t backend, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
        void (*set_tensor_2d_async)(ggml_backend_t backend,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
        void (*get_tensor_2d_async)(ggml_backend_t backend, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
        bool (*cpy_tensor_async)(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst);

        // (optional) complete all pending operations (required if the backend supports async operations)
        void (*synchronize)(ggml_backend_t backend);

        // (optional) graph plans (not used currently)
        // compute graph with a plan
        ggml_backend_graph_plan_t (*graph_plan_create) (ggml_backend_t backend, const struct ggml_cgraph * cgraph);
        void                      (*graph_plan_free)   (ggml_backend_t backend, ggml_backend_graph_plan_t plan);
        // update the plan with a new graph - this should be faster than creating a new plan when the graph has the same topology
        void                      (*graph_plan_update) (ggml_backend_t backend, ggml_backend_graph_plan_t plan, const struct ggml_cgraph * cgraph);
        // compute the graph with the plan
        enum ggml_status          (*graph_plan_compute)(ggml_backend_t backend, ggml_backend_graph_plan_t plan);

        // compute graph (always async if supported by the backend)
        enum ggml_status          (*graph_compute)     (ggml_backend_t backend, struct ggml_cgraph * cgraph);

        // (optional) event synchronization
        // record an event on this stream
        void (*event_record)(ggml_backend_t backend, ggml_backend_event_t event);
        // wait for an event on on a different stream
        void (*event_wait)  (ggml_backend_t backend, ggml_backend_event_t event);

        // (optional) op-offload H2D staging (issue #50 WIP): overlap a whole-tensor host->device
        // weight upload with the previous split's compute.  `stage_buffer` returns a device buffer of
        // at least `size` bytes for ring `slot` (NULL on allocation failure); `stage_upload` issues the
        // host->device copy into `dst` on the backend's auxiliary copy stream and records `ev` there;
        // `stage_wait` makes that copy stream wait for `ev` (recorded on the main stream); `stage_d2d`
        // copies the staged bytes to their destination on the main stream.  A backend that does not
        // implement these leaves all four NULL and the scheduler keeps the in-order copy path.
        void * (*stage_buffer)(ggml_backend_t backend, int slot, size_t size);
        void   (*stage_upload)(ggml_backend_t backend, void * dst, const void * data, size_t size, ggml_backend_event_t ev);
        // (optional) like `stage_upload`, but for a split upload whose device slice is strided in the
        // source weight: the device owns `n_copies` blocks of `width` bytes, `stride_src` apart, starting
        // `offset` bytes into the contiguous host source `src`.  The implementation assembles the compacted
        // slice into ring `slot` (a host gather for a small `n_copies`, else a whole-range H2D plus a device
        // 2-D compaction) and records `ev` on the copy stream.  Returns false if it cannot stage.
        // `src_pinned` is true when `src` is device-accessible pinned host memory (a GPU host buffer, not
        // the pageable model mmap), which lets a backend copy the strided slice directly with a 2-D H2D
        // instead of staging the whole contiguous range.
        bool   (*stage_gather)(ggml_backend_t backend, int slot, const void * src, size_t offset, size_t width, size_t stride_src, size_t n_copies, ggml_backend_event_t ev, bool src_pinned);
        void   (*stage_wait)  (ggml_backend_t backend, ggml_backend_event_t ev);
        void   (*stage_d2d)   (ggml_backend_t backend, void * dst, const void * src, size_t size);
        // (optional) measured H2D bandwidth in GB/s (one-off calibration, cached); 0 if unknown.  The
        // scheduler uses it to pick the staging gate (a narrow link needs a wider batch).
        float  (*stage_h2d_gbps)(ggml_backend_t backend);

        // (optional) op-offload H2D staging owned by the split's backend.  The four hooks above
        // assume one destination and a device event on the split backend's device; under `-sm tensor`
        // neither exists -- one logical upload is spliced across N devices, and the split's consumers
        // read per-device "simple" tensors rather than the split tensor's `data`, so a redirect of
        // that pointer can never reach the op.  Such a backend stages the input itself instead: it is
        // called with the source `input` (a host weight) and the split input `input_cpy`, lands each
        // device's chunk in that device's own ring, and returns true; the scheduler then skips its own
        // copy path for this input.  Only consulted when the staging gate is open (the backend must
        // implement `stage_h2d_gbps` so the gate is calibrated, and must advertise this hook so the
        // scheduler does not report staging as unsupported).
        bool   (*stage_input)(ggml_backend_t backend, struct ggml_tensor * input, struct ggml_tensor * input_cpy);

        // (optional) sort/optimize the nodes in the graph
        void                      (*graph_optimize)    (ggml_backend_t backend, struct ggml_cgraph * cgraph, struct ggml_backend_graph_optimize_params * params);

        // (optional) MoE expert cache (wip/moe-expert-cache): called by the scheduler when a
        // host-resident `MUL_MAT_ID` weight is about to be uploaded, with the *host master*
        // (`weight`), the scheduler's redirected device tensor the op will read (`weight_cpy`),
        // and a contiguous host copy of the routing (`ids`, `n_used x n_tok` int32).  This is the
        // only place both the master and the routing are available: the op-offload redirect makes
        // the op's `src0->data` point at `weight_cpy`.  Lets the backend drive its residency
        // policy, fill its own compact slots from the true master, and stage the slot-remapped
        // ids.  Returns true when it took the input over, in which case the scheduler skips its
        // own expert copy and the op reads the compact arena (via `moe_cache_get_table`).  A
        // backend that does not implement it leaves it NULL and the scheduler is unchanged.
        //
        // `slice_off`/`split_axis` describe the device's slice of the host master under `-sm tensor`
        // (wip/moe-expert-cache Phase 3): `split_axis` is the axis the master is split on (0/1, or -1
        // for an unsplit whole-expert table) and `slice_off` is the byte offset of this device's slice
        // inside one host expert blob.  When the split backend is the Meta backend it does not
        // implement the policy itself; it computes these per device from the meta split state and
        // forwards to each simple backend's hook with that device's simple tensor as `weight_cpy`.
        bool (*moe_cache_update)(ggml_backend_t backend, const struct ggml_tensor * weight, const struct ggml_tensor * weight_cpy, const int32_t * ids, int64_t n_used, int64_t n_tok, size_t ids_nb0, size_t ids_nb1, size_t slice_off, int split_axis);

        // wip/moe-expert-cache (session 7) identity fast path.  Return true iff this expert input is
        // cache-managed and the consumer can read the arena WITHOUT the host routing (an identity table:
        // slot == expert, raw ids).  The scheduler calls this BEFORE the ids readback; a true lets it
        // skip the readback, the full device synchronize it forces, the used-expert pruning and the
        // copy.  The Meta backend forwards to each simple backend and returns true only if every device
        // took over.
        bool (*moe_cache_take_over)(ggml_backend_t backend, const struct ggml_tensor * weight, const struct ggml_tensor * weight_cpy);

        // wip/moe-expert-cache (B2): device-side host-weight expert gather for an offloaded `MUL_MAT_ID`
        // prefill.  Copy only the routed experts from the host master (`weight`) into the device tensor
        // the op reads (`weight_cpy`), reading the routing (`ids`, device, strided) on the device.  Lets
        // the scheduler skip the per-op ids readback + full device synchronize it forces.  `slice_off`/
        // `split_axis` are the per-device slice geometry (0/-1 for an unsplit table).  The Meta backend
        // forwards to each simple backend and returns true only if every device took it.
        // MoE expert cache: the widest routed `MUL_MAT_ID` batch (in tokens) this backend's cache takes over
        // (its decode/verify band).  The scheduler keys its band decisions (the take-over before the ids
        // readback, the per-layer split grouping and rebalance) off it.  NULL = the historical 8.
        int64_t (*moe_cache_band)(ggml_backend_t backend);
        bool (*moe_cache_gather)(ggml_backend_t backend, const struct ggml_tensor * weight, const struct ggml_tensor * weight_cpy, const struct ggml_tensor * ids, size_t slice_off, int split_axis);
    };

    struct ggml_backend {
        ggml_guid_t guid;
        struct ggml_backend_i iface;
        ggml_backend_dev_t device;
        void * context;
    };

    struct ggml_backend_event {
        struct ggml_backend_device * device;
        void * context;
    };

    //
    // Backend device
    //

    // Note: if additional properties are needed, we should add a struct with all of them
    //       the current functions to obtain the properties can remain, since they are more convenient for often used properties
    struct ggml_backend_device_i {
        // device name: short identifier for this device, such as "CPU" or "CUDA0"
        const char * (*get_name)(ggml_backend_dev_t dev);

        // device description: short informative description of the device, could be the model name
        const char * (*get_description)(ggml_backend_dev_t dev);

        // device memory in bytes: 0 bytes to indicate no memory to report
        void         (*get_memory)(ggml_backend_dev_t dev, size_t * free, size_t * total);

        // device type
        enum ggml_backend_dev_type (*get_type)(ggml_backend_dev_t dev);

        // device properties
        void (*get_props)(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props);

        // backend (stream) initialization
        ggml_backend_t (*init_backend)(ggml_backend_dev_t dev, const char * params);

        // preferred buffer type
        ggml_backend_buffer_type_t (*get_buffer_type)(ggml_backend_dev_t dev);

        // (optional) host buffer type (in system memory, typically this is a pinned memory buffer for faster transfers between host and device)
        ggml_backend_buffer_type_t (*get_host_buffer_type)(ggml_backend_dev_t dev);

        // (optional) buffer from pointer: create a buffer from a host pointer (useful for memory mapped models and importing data from other libraries)
        ggml_backend_buffer_t (*buffer_from_host_ptr)(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size);

        // check if the backend can compute an operation
        bool (*supports_op)(ggml_backend_dev_t dev, const struct ggml_tensor * op);

        // check if the backend can use tensors allocated in a buffer type
        bool (*supports_buft)(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft);

        // (optional) check if the backend wants to run an operation, even if the weights are allocated in an incompatible buffer
        // these should be expensive operations that may benefit from running on this backend instead of the CPU backend
        bool (*offload_op)(ggml_backend_dev_t dev, const struct ggml_tensor * op);

        // (optional) event synchronization
        ggml_backend_event_t (*event_new)         (ggml_backend_dev_t dev);
        void                 (*event_free)        (ggml_backend_dev_t dev, ggml_backend_event_t event);
        void                 (*event_synchronize) (ggml_backend_dev_t dev, ggml_backend_event_t event);

        // (optional) MoE expert cache early auto-sizing (wip/moe-cache-autosize): called once by the
        // model layer after the target context's memory is allocated and before an auxiliary (MTP draft)
        // context is created, with the bytes of host-resident expert weights on this device.  Lets a
        // backend make its auto-enable / floor decision before an auxiliary context is sized for a cache
        // that will never serve it (measured ~3.5x slower under MTP).
        bool (*moe_cache_preflight)(ggml_backend_dev_t dev, size_t host_expert_bytes, size_t aux_reserve_bytes, size_t max_host_table_bytes);

        // (optional) MoE expert cache: hold `bytes` of the free VRAM for the post-prefill compute
        // layout, so the auto-sized arena does not take the space a later compute growth needs.
        // WIP r42 (TODO #42).
        void (*moe_cache_set_reserve)(ggml_backend_dev_t dev, size_t bytes);

        // (optional) MoE expert cache: aggregate the arena's cumulative hit/miss counters (WIP r42,
        // per-turn logging).  Returns false when the cache is disabled.
        bool (*moe_cache_stats)(ggml_backend_dev_t dev, int64_t * hits, int64_t * misses, int64_t * arena_bytes);

        // (optional) OPEN 2 (TODO #42): re-arm the expert-cache arena after a compute-buffer DROP returned
        // the VRAM.  Called at the prefill -> decode transition, after the wide compute layout has been
        // released and the narrow one re-reserved, so the stood-down tables can be re-allocated (the
        // survivors were never freed).  Returns true when anything was re-armed.  Appended like the
        // fields above, so every backend that does not set it is zero-initialized.
        bool (*moe_cache_rearm)(ggml_backend_dev_t dev);

        // (optional) OPEN 2 (TODO #42): the size of this device's movable-boundary slab WORK region
        // (0 when the device has no slab).  The reserve is sized for the WIDEST graph the parameters
        // allow and the work region holds it until a drop releases it; this lets the context tell whether
        // the live region is still wider than the narrow layout the workload actually settled on, so the
        // reserve can follow the workload instead of a worst case.  Appended like the fields above, so
        // every backend that does not set it is zero-initialized.
        size_t (*slab_work_size)(ggml_backend_dev_t dev);

        // (optional) wip/fit-slab-accounting: the VRAM this device's allocator must keep free OUTSIDE the
        // movable-boundary slab (0 when there is nothing the slab cannot serve).  `--fit` reserves it as a
        // hard floor: a thin headroom corrupts, because hipBLASLt's Tensile code objects and workspace are
        // allocated from real free VRAM.  Appended like the fields above, so every backend that does not
        // set it is zero-initialized.
        size_t (*slab_headroom_bytes)(ggml_backend_dev_t dev);

        // (optional) wip/slab-ring-region: arm/disarm the allocator's reserved H2D staging-ring hole.
        // Armed during prefill (the hole is not the arena's), disarmed at the prefill -> decode transition
        // (returned to the arena, so a decode-only stretch gets the VRAM back).  Appended like the fields
        // above, so every backend that does not set it is zero-initialized.
        void (*slab_ring_set)(ggml_backend_dev_t dev, bool arm);

        // (optional) wip/slab-ring-region: pin the always-resident narrow (decode/verify) work size, so
        // the allocator can place the staging ring above it.  Appended like the fields above.
        void (*slab_narrow_floor)(ggml_backend_dev_t dev, size_t bytes);

        // (optional) wip/moe-verify-fusions (narrow-2): pin the MTP draft's COMPUTE region at the top of
        // the slab's reserved VA (idempotent), so it cannot alias the target's narrow verify buffer at
        // base 0.  Appended like the fields above, so backends that do not set it are zero-initialized.
        void (*slab_narrow2_floor)(ggml_backend_dev_t dev, size_t bytes);

        // (optional) wip/moe-verify-fusions (narrow-2): route this context's COMPUTE allocations to the
        // pinned narrow-2 region (the MTP draft).  Appended like the fields above.
        void (*slab_compute_narrow2)(ggml_backend_dev_t dev, bool enable);
    };

    struct ggml_backend_device {
        struct ggml_backend_device_i iface;
        ggml_backend_reg_t reg;
        void * context;
    };

    //
    // Backend (reg)
    //

    struct ggml_backend_reg_i {
        const char * (*get_name)(ggml_backend_reg_t reg);

        // enumerate available devices
        size_t             (*get_device_count)(ggml_backend_reg_t reg);
        ggml_backend_dev_t (*get_device)(ggml_backend_reg_t reg, size_t index);

        // (optional) get a pointer to a function in the backend
        // backends can add custom functions that are not part of the standard ggml-backend interface
        void * (*get_proc_address)(ggml_backend_reg_t reg, const char * name);
    };

    struct ggml_backend_reg {
        int api_version; // initialize to GGML_BACKEND_API_VERSION
        struct ggml_backend_reg_i iface;
        void * context;
    };

    // Add backend dynamic loading support to the backend

    // Initialize the backend
    typedef ggml_backend_reg_t (*ggml_backend_init_t)(void);
    // Optional: obtain a score for the backend based on the system configuration
    // Higher scores are preferred, 0 means the backend is not supported in the current system
    typedef int                (*ggml_backend_score_t)(void);

#ifdef GGML_BACKEND_DL
#    ifdef __cplusplus
#        define GGML_BACKEND_DL_IMPL(reg_fn)                             \
            extern "C" {                                                 \
            GGML_BACKEND_API ggml_backend_reg_t ggml_backend_init(void); \
            }                                                            \
            ggml_backend_reg_t ggml_backend_init(void) {                 \
                return reg_fn();                                         \
            }
#        define GGML_BACKEND_DL_SCORE_IMPL(score_fn)       \
            extern "C" {                                   \
            GGML_BACKEND_API int ggml_backend_score(void); \
            }                                              \
            int ggml_backend_score(void) {                 \
                return score_fn();                         \
            }
#    else
#        define GGML_BACKEND_DL_IMPL(reg_fn)                              \
            GGML_BACKEND_API ggml_backend_reg_t ggml_backend_init(void);  \
            ggml_backend_reg_t                  ggml_backend_init(void) { \
                return reg_fn();                                          \
            }
#        define GGML_BACKEND_DL_SCORE_IMPL(score_fn)        \
            GGML_BACKEND_API int ggml_backend_score(void);  \
            int                  ggml_backend_score(void) { \
                return score_fn();                          \
            }
#    endif
#else
#    define GGML_BACKEND_DL_IMPL(reg_fn)
#    define GGML_BACKEND_DL_SCORE_IMPL(score_fn)
#endif

#ifdef  __cplusplus
}
#endif
