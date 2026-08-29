#pragma once

// wip/moe-expert-cache — Phase 1a foundation (2026-09-28).
//
// The `device_alias()` seam: every MoE expert blob address the compute kernel reads must go
// through `moe_cache_alias_get()`.  Under one GPU / `-sm layer` the resident alias is a VRAM slot
// holding the whole expert; under `-sm tensor` it is a VRAM slot holding *this device's slice*,
// and the cold alias is the host master (Phase 1b: a UVA view of the pinned host slice).  Nothing
// in this module assumes a whole-expert blob — it only knows "n_experts blobs of expert_bytes".
//
// Default OFF: `MOE_EXPERT_CACHE_MIB` unset/0 -> every entry point is a cheap no-op and backend
// behaviour (and output) is bit-identical to a cache-less build.
//
// Env (all read once):
//   MOE_EXPERT_CACHE_MIB=N     per-device VRAM budget in MiB; 0/unset disables (default 0).  Under
//                              `-sm layer` each device owns a disjoint set of layers, so N is
//                              allocated on EVERY device that owns cache tables.
//   MOE_EXPERT_CACHE_SLOTS=S   uniform slots/table override; 0 (default) derives it from the budget
//                              and the *measured* total expert bytes (adaptive and uniform)
//   MOE_EXPERT_CACHE_PERIOD=P  LFRU decay period in decode steps (default 32)
//   MOE_EXPERT_CACHE_TOUCH=T   re-touch threshold before admission (default 2; floored at 2)
//   MOE_EXPERT_CACHE_FILL=0    disable the master->slot copy (default 1; the kernel consumer
//                              reads the slot, so the copy must be on whenever the cache takes
//                              an op over)
//   MOE_EXPERT_CACHE_RESERVE_MIB=R  VRAM held back from the arena (default 1024; the arena is sized
//                              from *free* memory, so `--fit` need not know about it)
//   MOE_EXPERT_CACHE_PREFILL_SEED=0  disable the prompt-routing seed (default 1)
//   MOE_EXPERT_CACHE_PREFILL_SEED_N=N  cap the seeded experts/table (0 = all slots)
//   MOE_EXPERT_CACHE_PROVISIONAL=0  disable reclaiming pre-filled/seed slots before first hit
//                              (default 1)
//   MOE_HOST_POOL_MIB=N        PROCESS-WIDE pinned host pool (L2) size in MiB; 0/unset disables it
//                              (default 0).  One pool per host tensor, shared by every device; the budget
//                              is split over the registered host tensors.  The pool is a GPU-readable
//                              BOUNCE BUFFER: the GPU reads it (pinned), and it is refilled from the GGUF
//                              BUFFERED, through the page cache, so a miss is a RAM read, not disk.
//                              Phase 2b of wip/host-expert-dio-cache.
//   MOE_HOST_POOL_DIO=1        fill the pool with O_DIRECT instead of buffered (debug fallback only:
//                              bypasses the page cache)
//   MOE_HOST_POOL_PREWARM=0    do not DIO/buffered-fill every pool slot when it is built (default on)

// Phase 1a status: registration + LFRU policy + arena + alias + fill + report.  The
// `mul_mat_id` dispatch already calls `moe_cache_observe()` (live hit-rate measurement) and the
// alias seam is the next commit's kernel consumer; see wip/moe-expert-cache/README.md §3.3.

#include <cstddef>
#include <cstdint>

struct ggml_tensor;

// The decode/verify band the cache serves.  Prefill (n_tokens > this) keeps the full-table path
// and its MoE fusions; the offload relaxation and the fusion guard both key off this value.
// The cache serves a routed MUL_MAT_ID only through the routed-expert MMVQ kernel (it reads the arena
// through the slot remap; MMQ/MMF do not), so the band is that kernel's band on the device: the narrowest
// get_mmvq_mmid_max_batch over the quantized types (16 on RDNA4, 8 on RDNA3 / NVIDIA), never below the
// historical 8.  GGML_MOE_CACHE_MAX_TOK lowers it (e.g. 8 = the pre-extension band).  This is the single
// source: the scheduler asks the backend through the `moe_cache_band` iface hook.
int moe_cache_max_tok_dev(int device);
int moe_cache_max_tok();   // the current device's band

#define MOE_EXPERT_CACHE_MAX_TOK (moe_cache_max_tok())

struct moe_cache_alias {
    void * ptr;        // device-usable address; nullptr if unknown
    bool   resident;   // true iff backed by a VRAM slot
};

bool moe_cache_enabled();
// true once the cache is sized with host-expert tables: the workspace pool then keeps a free-VRAM floor on its
// first attempt (ggml_cuda_pool_leg::alloc)
bool moe_cache_floor_active();

// Early auto-sizing decision (wip/moe-cache-autosize): called once by the model layer, after the target
// context's memory is allocated and before any auxiliary (MTP draft) context is created, with the
// bytes of host-resident expert weights on `device` and a per-device `aux_reserve_bytes` estimate for
// the auxiliary context's own memory (measured ~3.7 GiB for the MTP draft).  An auto cache whose
// projected arena would fall below the floor is disabled HERE, so an auxiliary context is never sized
// for a cache that will not serve it.  An explicit MOE_EXPERT_CACHE_MIB is left untouched.  Latched on
// the first call; returns true when the cache remains active.
bool moe_cache_preflight(int device, size_t host_expert_bytes, size_t aux_reserve_bytes);

// WIP r42 (TODO #42): hold `bytes` of the free VRAM on `device` for the post-prefill compute layout,
// so the auto-sized arena does not take the space a later compute growth needs.  Called at the
// prefill -> decode transition, before the arena is sized.
void moe_cache_set_extra_reserve(int device, size_t bytes);

// WIP r42: aggregate the arena's cumulative counters for per-turn logging.  Returns false when the
// cache is disabled.  The counters are process-global (one shared cache), so a before/after delta is
// whole-process activity during the turn.
bool moe_cache_get_stats(int64_t * hits, int64_t * misses, int64_t * arena_bytes);

// WIP r42: fail-soft guard -- free the whole arena (once) so a compute allocation that needs the VRAM
// can succeed; the cache is disabled for the rest of the run.  Returns true when it released anything.
bool moe_cache_release_arena();

// WIP r42: free the largest table arenas until `need_bytes` (+margin) is released, so a compute
// allocation that is only a little short of a contiguous block can succeed while most of the cache
// survives.  Returns true when anything was freed.
bool moe_cache_shrink_arena(size_t need_bytes);

// WIP r42: free ONE (the largest) table arena; the caller retries the failed allocation and calls this
// again only if it is still short.  Returns false when there is nothing left to free.
bool moe_cache_shrink_step();

// OPEN 2 (TODO #42): disable the cache outright (stream the experts from the host) because the
// movable-boundary slab cannot give it a usable arena.  MUST run before the first cache-consulting graph.
void moe_cache_disable_streaming(const char * why);

// OPEN 2 (TODO #42): re-arm the arena after a compute-buffer drop returned the VRAM.  Re-allocates the
// stood-down tables to the slot count the sizing settled on; the surviving tables were never freed, so
// the cache resumes with its residents.  Run it with no graph in flight, after the compute buffer has
// been dropped, so the next graph's take-over decisions all see the same residency.  Returns true when
// anything was re-armed.
bool moe_cache_rearm();

// OPEN 2 (TODO #42): evict every table whose arena lies inside the slab range `[lo, hi)`, so the slab can
// reassign those chunks to the work pool (the movable-boundary design's boundary move).  This is the ONLY
// loss of resident entries: the tables that live in the taken chunks go non-resident -- their storage is
// now the work pool -- while every other table keeps its bytes.  Their slabs are handed back through
// `ggml_cuda_slab_arena_free`.  Returns the bytes released.
size_t moe_cache_evict_slab_range(int device, void * lo, void * hi);

// OPEN 2 debug validator: `MOE_EXPERT_CACHE_VALIDATE=1` checks the structural invariants every consumer
// relies on (slot/arena consistency, `g_arena_bytes`, remap/identity coherence) and logs any
// violation; `=2` additionally D2H-reads each resident slot's head and flags a non-finite f16 scale --
// the precondition of the repeated-`/` MMQ over-read corruption.  A no-op when the env var is unset, so
// a normal run pays nothing.  Call it after anything that changes the arena (a stand-down, a release,
// a sizing pass) and before the next graph consumes it.
void moe_cache_validate(const char * where);

// OPEN 2: evict the tables living in a slab range so the work pool can take those chunks.

// True when the in-place UVA cold transport is active.  Both the per-op consumer and the cache-band
// gate+up+GLU / down-fold kernels do the cold-region lookup, so no fusion has to stand down.
bool moe_cache_cold_active();

// True once the arena sizing has run (or an explicit slot count was given).  Before this the per-op read
// source can still change, so CUDA graph capture must wait; after it the source is a constant per table.
bool moe_cache_ready();

// True only when the cache has at least one registered routed expert table (i.e. it observed
// host-resident expert weights and can participate in this run).  When false, the cache can never take
// an input over, so its fusions must behave exactly as if it were disabled.
bool moe_cache_has_tables();

// True only when EVERY table has a usable arena.  The cache-band fusion guard is global (one answer
// for the whole graph) and the gate+up fused kernel indexes the gate lane with the up table's remap, so
// a partially-failed cache (any allocation failed, or a budget below one expert) must stand its fusions
// down wholesale and let the per-op path run.  Returns false before sizing and for an empty cache.
bool moe_cache_has_arena();

// Per-table form of the fusion guard: true when THIS table can serve the cache-aware fused path, i.e.
// it is an identity table (fully resident; `moe_cache_take_over` serves it with the raw ids).  The
// global `moe_cache_has_arena()` requires ALL tables, so one evicted table flips the arithmetic of
// every other table in the graph; this lets a resident table keep the fused path while only the
// unserveable op(s) stand down.  `op` is the `MUL_MAT_ID` (for the layer suffix), `weight` is the
// tensor the fused call site would redirect (the scheduler's copy or the original).
bool moe_cache_table_serves(const struct ggml_tensor * op, const struct ggml_tensor * weight, int device);

// wip/slab-ring-region: the deferred arena sizing has run.  The slab's H2D-ring hole must stay ARMED
// through sizing, so the auto-sized arena never allocates its highest tables into the hole (a disarm
// before that makes the next arm evict those hot tables, and the cold re-arm corrupts a wide consumer).
bool moe_cache_is_sized();

// Register (or look up) the expert table backing `src0` and return its id.  `layer`/`role` are for
// reporting.  `host` is the master the cache fills from (may be null).  Idempotent per `(src0, device)`.
// `device` is the CUDA ordinal whose kernels will read this table's arena (the layer's owner under
// `-sm layer`); a table's arena is a per-device cudaMalloc, so it must be allocated on that device.
//
// Phase 3 (`-sm tensor`): `expert_bytes` is *this device's* slice, while `host_bytes` is the full
// per-expert stride in the host master, `src_off` is the byte offset of the slice within a host expert
// and `split_axis` is the master's split axis (0/1, or -1 for an unsplit whole-expert table).  For a
// contiguous axis-1 slice `host_pitch` is 0 and the fill is one 1-D copy; for a strided axis-0 slice
// `host_pitch` is the host row pitch and the fill is a 2-D copy.
int moe_cache_table(const void * src0, int layer, const char * role, int n_experts, size_t expert_bytes,
                    size_t host_bytes, size_t src_off, size_t host_pitch, int split_axis,
                    const void * host, int device);

// wip/host-expert-dio-cache Phase 1 (plumbing, inert): register the on-disk source of a host-resident
// expert tensor, keyed by its host data pointer (== the `host` passed to `moe_cache_table`).  Called by
// the model loader with the GGUF path, the tensor's file offset and its whole-tensor geometry; the
// per-device slice geometry is merged in `moe_cache_table`.  A no-op unless the host pool is enabled
// (`--host-experts pool` or `MOE_HOST_POOL_MIB`), so with the pool off no state is created and behaviour
// is byte-identical.
void moe_cache_set_host_source(const void * tensor_data, const char * path, size_t offs,
                               int n_experts, size_t host_bytes, size_t total_bytes);

// `--host-experts pool`: enable the bounded pinned host pool (size from MOE_HOST_POOL_MIB, default
// MOE_HOST_POOL_FRAC % of the MoE host experts).  Called by the loader before tables are registered.
void moe_cache_set_host_pool(bool on);

// `--host-experts pool`: enable the bounded pinned host pool (size from MOE_HOST_POOL_MIB, default
// MOE_HOST_POOL_FRAC % of the MoE host experts).  Called by the loader before tables are registered.
void moe_cache_set_host_pool(bool on);

// LFRU residency decision + alias for expert `expert` of `table`.  The single compute-path seam.
moe_cache_alias moe_cache_alias_get(int table, int expert);

// Observe the routing of one `MUL_MAT_ID` (live hit-rate measurement / warm-up).  `op` is the op
// output (its name carries the `-<layer>` suffix); `src0` is the expert weight table.  The
// gate/up/down ops of one (layer, token) are coalesced so the policy sees one reach per token.
void moe_cache_observe(const struct ggml_tensor * op, const struct ggml_tensor * src0, const struct ggml_tensor * ids);

// Drive the LFRU policy + slot fill from the scheduler, which is the only place the *host master*
// (`weight`/`weight->data`) and the host-readable routing are both available (the op-offload
// redirect makes `src0->data` point at the device `input_cpy`).  `weight_cpy` is the scheduler's
// redirected device tensor the op will actually read; the cache aliases it to the same table so
// `moe_cache_get_table()` can find the residency from the op.  `ids` is the contiguous host copy
// of the routing, `n_used x n_tok` int32.  Registers the table on first call and stages the
// slot-remapped ids (see `moe_cache_get_table`).  Returns true when it took the input over, i.e.
// the scheduler must skip its own expert copy and let the op read the compact arena.
// Called once per gate/up/down op per (layer, token); each weight tensor is its own table.
//
// `device` is the CUDA ordinal running this upload (the backend adapter's `ctx->device`); it is the
// device that will read the arena, and the cache allocates/migrates this table's arena onto it.
//
// `slice_off`/`split_axis` are the per-device slice geometry under `-sm tensor` (see `moe_cache_table`);
// they are `0`/`-1` for an unsplit table (1 GPU and `-sm layer`).  When the split backend is the Meta
// backend it computes them per device and forwards here once per simple device.
bool moe_cache_update_host(const struct ggml_tensor * weight, const struct ggml_tensor * weight_cpy,
                           const int32_t * ids, int64_t n_used, int64_t n_tok,
                           size_t ids_nb0, size_t ids_nb1, void * stream,
                           int device, size_t slice_off, int split_axis);

// Host-weight expert gather (B2, session 15).  For a `-ncmoe` `MUL_MAT_ID` prefill whose expert upload is
// NOT staged, copy only the routed experts from the host master into the scheduler's device `input_cpy`,
// on the device, instead of reading the routing back to the host and copying per-run.  Independent of the
// cache (it is a `-ncmoe` prefill feature; the tables need not be cache-managed).  `weight` is the host
// master, `weight_cpy` the device tensor the op reads, `ids` the device routing (strided).
// `slice_off`/`split_axis` are the per-device slice geometry (`0`/`-1` for an unsplit table).  `ids` is
// the routing tensor (device resident; `ne[0] x ne[1]`, strided).
bool moe_cache_gather_host(const struct ggml_tensor * weight, const struct ggml_tensor * weight_cpy,
                           const struct ggml_tensor * ids, void * stream, int device,
                           size_t slice_off, int split_axis);

// Kernel consumer seam.  Given the op (`op`) and the (redirected) expert table the op will read
// (`weight_cpy`) plus the CUDA ordinal running it, return the compact arena the op must read instead:
// `arena` holds `n_slots` experts of `expert_bytes`, and `remap_dev` holds the slot-remapped ids for
// the current routing (`n_used x n_tok` int32, device).  Returns false when this tensor is not
// cache-managed (caller uses the normal path).
bool moe_cache_get_table(const struct ggml_tensor * op, const struct ggml_tensor * weight_cpy, int device,
                         void ** arena, int64_t * n_slots,
                         size_t * expert_bytes, void ** remap_dev, int64_t * n_used, int64_t * n_tok);

// Structural takeover for the decode/verify band: true when the table is cache-managed and the consumer
// can read the arena without the host routing (an identity table, served with the raw ids).  Every other
// table takes the eager host-routing path (`moe_cache_update_host`), which materializes the remap on the
// host and republishes it before the graph.
bool moe_cache_take_over(const struct ggml_tensor * weight, const struct ggml_tensor * weight_cpy, int device);

// wip/moe-expert-cache: redirect a fused decode MoE (gate+up+GLU) onto the cache arenas.  When the
// routed up table (`src0`) and the gate table (`gate`) are both cache-managed for the current routing,
// fills the caller-owned shallow copies `src0_cpy`/`ids_cpy`/`gate_cpy` (arena base + slot-remapped ids)
// and returns true - the caller then passes them to the mmvq gate+up+GLU fusion, which reads the compact
// arena exactly as the unfused per-op consumer does.  Returns false when the cache is not managing this
// op (declined for the shape, or disabled), in which case the caller keeps the original tensors - which
// the scheduler copied in full.  Both tables must be redirected together: the hook takes over each of
// them independently and skips the scheduler's copy, so a mixed redirect would read a stale table.
bool moe_cache_redirect_fused(const struct ggml_tensor * op,
                              const struct ggml_tensor * src0,
                              const struct ggml_tensor * gate,
                              const struct ggml_tensor * ids,
                              int device,
                              void * stream,
                              struct ggml_tensor * src0_cpy,
                              struct ggml_tensor * ids_cpy,
                              struct ggml_tensor * gate_cpy);

// Phase 1b/3 UVA cold seam.  Given the arena base the op is reading (`src0->data` after the slot-remap
// redirect), return the pinned host master's device-accessible alias, the resident slot count, and the
// host geometry a cold read must use.  A remapped id `< *n_res` is a VRAM slot read with the arena's
// strides; `>= *n_res` is a COLD expert whose host index is `id - *n_res`, read from
// `*cold_base + (id - *n_res) * *cold_channel_bytes` with `*cold_channel_bytes` as the per-expert stride
// and `*cold_row_bytes` as the per-row stride (0 => the device slice's own row stride).  `*cold_base`
// already folds in the per-device tensor-split slice offset (`src_off`), so a Phase 3 slice reads the
// right host bytes.  The strides are in BYTES; the caller converts them to the kernel's type-block
// units.  Returns false when cold reads are disabled or this arena is not cache-managed, in which case
// every id is a resident slot (the 1a contract).
bool moe_cache_get_cold(const void * arena, void ** cold_base, int64_t * n_res,
                        size_t * cold_channel_bytes, size_t * cold_row_bytes);

void moe_cache_init(int device);   // lazy, per device; safe to call repeatedly
void moe_cache_selftest();         // synthetic arena fill/verify
void moe_cache_report();           // hit/miss/fill/eviction summary
