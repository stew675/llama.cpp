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
// Is the slab live on THIS device?  (The gate can be on while a device failed to create one.)  Used to
// keep the generic OOM path from churning the arena, which cannot help under the slab.
bool   ggml_cuda_slab_active    (int device);
// The slab declined because `work estimate + cache floor` does not fit.  The expert cache must then stream
// the experts from the host instead of building an arena a wide prefill would have to evict wholesale.
bool   ggml_cuda_slab_cache_unusable(int device);
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
// The arena's allocation unit (fine: the slab is one mapping, so this is bookkeeping only).
size_t ggml_cuda_slab_arena_unit  (int device);
// Base of the work region for a `need`-byte compute buffer, moving the boundary up (evicting the arena
// tables in the taken chunks) if required.  nullptr when the slab cannot be created or has no room.
void * ggml_cuda_slab_work_alloc(int device, size_t need);
// A work-region view was dropped; the next, smaller work need may shrink the boundary and hand the slack
// back to the arena.
void   ggml_cuda_slab_work_release(int device, size_t reported);
// Chunk-aligned arena allocation from `[boundary, size)`; nullptr when the arena region is full.
void * ggml_cuda_slab_arena_alloc(int device, size_t size);
// Serve a large TRANSIENT (a workspace or a draft buffer) from the arena, EVICTING the tables in the top
// band it needs.  The freed range is contiguous by construction (it reuses the boundary move's range
// eviction), which looping per-table stand-downs cannot guarantee.  Call with NO lock held.
void * ggml_cuda_slab_arena_alloc_transient(int device, size_t size);
void   ggml_cuda_slab_arena_free (int device, void * ptr, size_t size);
