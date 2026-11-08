#pragma once

#include <stddef.h>
#include <stdbool.h>

// OPEN 2 / r42 (TODO #42): the movable-boundary slab interface (implemented in ggml-cuda.cu).
//
// This is the ONLY allocator here: an earlier per-allocation VMM pool (reserve/alloc/map/unmap with
// physical handed back on demand) was retired -- ROCm rejects a sub-range `hipMemUnmap`, so a table's
// tail could not be given back, and the pool could never satisfy a `cudaMalloc` outside itself.  The
// slab's boundary move replaces all of it, and the plain `cudaMalloc` path remains for any device where
// the slab declines.

// --- OPEN 2: the movable-boundary slab -----------------------------------------------------------------
//
// ONE slab per device holds BOTH the work pool (the compute buffer, the LOW region `[0, boundary)`) and
// the MoE expert-cache arena (the HIGH region `[boundary, size)`).  The slab is reserved and mapped
// EXACTLY ONCE; growing the work pool is a boundary move inside the already-mapped slab (the lowest arena
// chunks are reassigned and the arena tables there are evicted), so HIP is touched only at slab creation
// and at shutdown.  The work region's base VA never moves, so a growing layout keeps its addresses.

bool   ggml_cuda_slab_enabled   ();
// The env kill switch, ignoring whether the slab is armed (`g_slab_armed`): "the user asked for the slab".
// Used to warn when the MoE expert cache is enabled but the slab never armed (issue #136, item B).
bool   ggml_cuda_slab_configured();
// Is the slab live on THIS device?  (The gate can be on while a device failed to create one.)  Used to
// keep the generic OOM path from churning the arena, which cannot help under the slab.
bool   ggml_cuda_slab_active    (int device);
// May `[dst, dst+nbytes)` be the destination of a raw cross-device peer copy?  A peer copy into a slab range
// the driver's grants do not cover FAULTS the GPU instead of returning an error (the grant is best effort:
// three ~29 GiB slabs leave the peers no free VRAM to build it), and the slab check is process-wide because
// a copy fails while ANY live slab's grants are incomplete -- even between two fully granted devices.  A
// destination outside every slab is always fine (plain memory peer copies stage inside the driver).
// false => stage the copy through pinned host memory.
bool   ggml_cuda_slab_peer_copy_ok(const void * dst, size_t nbytes);
// The slab declined because `work estimate + cache floor` does not fit.  The expert cache must then stream
// the experts from the host instead of building an arena a wide prefill would have to evict wholesale.
bool   ggml_cuda_slab_cache_unusable(int device);
// Issue #136 (item F): the slab was armed for a host-expert model but could not be created on this device
// at all (VMM unavailable, bad chunk geometry, cudaMemGetInfo failure, or too little free VRAM for the
// reserve).  Distinguished from `cache_unusable`, which means "creatable, but work + minimum arena do not
// fit".  `ggml_cuda_slab_decline_reason` is a static-lifetime string for the diagnostic.
bool   ggml_cuda_slab_declined  (int device);
const char * ggml_cuda_slab_decline_reason(int device);
// The hard cache floor, in MiB (`GGML_CUDA_SLAB_MIN_ARENA_MIB`, default 2048).
size_t ggml_cuda_slab_min_arena_mib();
// The slab's chunk unit on this device (0 when the slab is disabled).  Arena allocations are rounded up
// to it, so a table's storage is a whole number of chunks.
size_t ggml_cuda_slab_chunk_size(int device);
void * ggml_cuda_slab_work_base (int device);
// The work region's current size (the boundary): what a compute buffer should report, so growth inside it
// does not look like a realloc to the graph allocator.
size_t ggml_cuda_slab_work_size (int device);
// The arena region's total size on this device (the cache sizes itself against this, not the free VRAM).
size_t ggml_cuda_slab_arena_total (int device);
// Grow every slab into the VA it reserved but left unmapped, down to `GGML_CUDA_SLAB_HEADROOM_MIB` of free
// VRAM -- the reserve the fitting probe made the slab hold back is a guess, and this is the measured
// correction once the weights / KV / draft are resident.  Safe to call repeatedly (a no-op when there is
// nothing to gain); moves no address, so every work view and arena table keeps its base.
void   ggml_cuda_slab_extend_all();
// raise the free VRAM the slab extension leaves (never lowers an explicit GGML_CUDA_SLAB_HEADROOM_MIB)
void   ggml_cuda_slab_headroom_at_least_mib(size_t mib);
// fit-slab-accounting revival (G3): the draft-context VRAM the MoE preflight projected; the slab reserve
// is sized from it (plus the headroom) because the slab is created before the draft context.
void   ggml_cuda_slab_set_aux_reserve(size_t bytes);
// The arena's allocation unit (fine: the slab is one mapping, so this is bookkeeping only).
size_t ggml_cuda_slab_arena_unit  (int device);
// Base of the work region for a `need`-byte compute buffer, moving the boundary up (evicting the arena
// tables in the taken chunks) if required.  nullptr when the slab cannot be created or has no room.
void * ggml_cuda_slab_work_alloc(int device, size_t need);
// A work-region view was dropped; the next, smaller work need may shrink the boundary and hand the slack
// back to the arena.
void   ggml_cuda_slab_work_release(int device, size_t off, size_t need);
// wip/moe-verify-fusions (narrow-2): pin (idempotently) the MTP draft's COMPUTE region at the top of the
// slab's reserved VA from a known narrow size.  No-op when the slab is not live.
void   ggml_cuda_slab_narrow2_floor(int device, size_t bytes);
// Chunk-aligned arena allocation from `[boundary, size)`; nullptr when the arena region is full.
void * ggml_cuda_slab_arena_alloc(int device, size_t size);
// Serve a large TRANSIENT from the slab's reserved, NON-EVICTABLE ring hole (a fixed region at the top of
// the mapped slab).  Bump-allocates within the hole; returns nullptr when the slab has no ring region, when
// the hole is exhausted, or when it could not be armed.  No ROCm call (pure bookkeeping inside the one slab
// mapping).  wip/slab-ring-region.
void * ggml_cuda_slab_arena_alloc_transient(int device, size_t size);
// Per-slot capacity of the slab's ring hole (0 when there is none).  The op-offload H2D ring allocates one
// full-capacity slot so every slot gets a stable VA and the ring never reallocs a slot.
size_t ggml_cuda_slab_ring_slot_capacity(int device);
// The ring hole's reserved size (0 when the slab carved none): `llama_context` uses it to skip the
// narrow -> wide compute-buffer drop when there is no ring to arm.  wip/slab-ring-region (issue #164).
size_t ggml_cuda_slab_ring_region(int device);
// wip/slab-transient-absorption (F5, step 0): was the ring hole REQUESTED for this model?  The ring
// fallback is refused only when it was (an operator can still leave the ring outside the slab with
// GGML_CUDA_SLAB_RING_MIB=0, which is the pre-slab behaviour).
bool   ggml_cuda_slab_ring_requested();
// wip/slab-transient-absorption: the WIDE-PHASE auxiliary region (the FA prefill staging arena's home).
// Bump-allocates above the live work boundary while the wide view is armed; released with the ring on the
// prefill -> decode transition.  Returns nullptr when there is no wide view or no room (the caller falls
// back).  `gen_out` receives the region generation, which the caller must compare against
// `ggml_cuda_slab_wide_aux_gen` before reusing a cached pointer.
void * ggml_cuda_slab_wide_aux_alloc(int device, size_t size, size_t * gen_out);
void   ggml_cuda_slab_wide_aux_free(int device, void * ptr, size_t size);

bool   ggml_cuda_slab_wide_available(int device);
size_t ggml_cuda_slab_wide_aux_gen(int device);
// One-line steady-state layout + transient accounting, gated on GGML_CUDA_SLAB_STATS (diagnostic only).
void   ggml_cuda_slab_log_stats(int device, const char * tag);
// wip/slab-transient-absorption: physical retained by the VMM workspace pool on this device (diagnostic).
size_t ggml_cuda_pool_mapped_bytes(int device);
// Largest single allocation made through `ggml_cuda_device_malloc` AFTER the slab became active, per
// device: everything it reports is OUTSIDE the slab by construction (scheduler buffers, the Meta
// all-reduce temp, the q8_1 arena, the pool).  Diagnostic only (GGML_CUDA_SLAB_STATS).
size_t ggml_cuda_outside_alloc_max_bytes(int device);
// Largest allocation the same choke point REFUSED after the slab became active (the outside-reserve
// shortfall signal).  Diagnostic only (GGML_CUDA_SLAB_STATS).
size_t ggml_cuda_outside_fail_max_bytes(int device);
// Low-water mark of the free VRAM outside the slab, per device (diagnostic only).
size_t ggml_cuda_free_min_bytes(int device);
void   ggml_cuda_free_min_sample(int device, size_t free_b);
// wip/slab-transient-absorption (step 2b): bytes and block count the workspace pools have currently handed
// out from the slab's wide band on this device.  The block count is the LEAK TRIPWIRE: it must be 0 at the
// prefill -> decode drop (a non-zero value means a pool allocation outlived the band, i.e. a retained site
// that was not pinned; `ggml_cuda_slab_ring_set` warns `POOL_WIDE_LEAK`).
size_t ggml_cuda_pool_wide_bytes (int device);
size_t ggml_cuda_pool_wide_blocks(int device);
// Diagnostic: bytes a large pool allocation asked for while the band was armed and the band REFUSED (it
// then stays outside the slab as pool high-water).  Reset at the wide arm, reported at the wide drop.
size_t ggml_cuda_pool_wide_refused(int device);
void   ggml_cuda_pool_wide_refused_reset(int device);
// Arm/disarm the ring hole: armed during prefill (excluded from the arena), disarmed at the
// prefill -> decode transition (returned to the arena, so a decode-only stretch gets the VRAM back).
// Arming evicts the arena tables the hole had picked up; called with NO lock held.
void   ggml_cuda_slab_ring_set(int device, bool arm);
// Size the ring hole the slab reserves (0 = the ring stays a cudaMalloc outside the slab).  Set once by the
// model preflight, before the slab is created, from the largest host-resident weight table.
void   ggml_cuda_slab_set_ring_region(size_t bytes);
void   ggml_cuda_slab_arena_free (int device, void * ptr, size_t size);

// Phase-2 (item 6b): slab state accessors used by the device wrappers in ggml-cuda.cu.
void   ggml_cuda_slab_arm();
void   ggml_cuda_slab_disarm();
bool   ggml_cuda_slab_compute_enabled();
void   ggml_cuda_slab_set_compute_narrow2(bool enable);
void   ggml_cuda_slab_set_compute_plain(bool enable);
bool   ggml_cuda_slab_compute_plain();
size_t ggml_cuda_slab_headroom_bytes();
void   ggml_cuda_slab_set_narrow_floor(int device, size_t bytes);
void   ggml_cuda_slab_reserve_wide(int device, size_t bytes);
size_t ggml_cuda_slab_seed_floor(int device);
