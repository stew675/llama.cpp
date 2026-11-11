// wip/apu-expert-gtt P2: the UMA **GTT expert store** -- one registered anonymous region that holds the
// resident MoE expert set, read in place by `MUL_MAT_ID`.
//
// THE SHAPE (wip/apu-expert-gtt/README.md §0): ONE `mmap(PROT_READ|PROT_WRITE, MAP_NORESERVE)` +
// `MADV_HUGEPAGE`, registered once with `cudaHostRegister(Mapped|ReadOnly)`, sliced into per-(tensor, expert)
// slots.  There is no pinned master, no host pool, no arena and no slab in this tier: on an APU host RAM and
// "VRAM" are one DRAM, so the registered region *is* the resident expert set and residency is the whole
// policy.  The measurements this rests on are `wip/apu-expert-gtt/MEASUREMENTS.md` §13 (P1.0):
//
//   * THPs survive `hipHostRegister` (AnonHugePages 1024.0/1024 before and after);
//   * a registered READ-ONLY range reads at 232.1 GB/s, the same range registered WRITABLE at 60
//     (`MOE_EXPERT_GTT_RO=0` is the A/B that keeps that claim honest on the real region, not just a probe);
//   * `pread` INTO the registered region is accepted and the GPU reads the fresh bytes (including on the
//     reuse arm), so the host may fill the slots through its own mapping with no flush;
//   * with `iommu=off` the host VA **is** the device VA, which is what lets ONE pointer be both the fill
//     destination and the kernel's source.  That property is ASSERTED at creation and the GTT store declines
//     when it does not hold (no fault-and-retry exists on this part: an unregistered page is not in the
//     GPU's page tables at all, so a second address would be a fatal fault, not a slow path).
//
// The region machinery lives here; the buffer type that carves tensors out of it (the delivery's KV-region
// pattern: a device-carrying buft with the plain device type's vtable contract) lives in `ggml-cuda.cu`,
// which is where the private `ggml_backend_cuda_buffer_type_context` and the device registry are.
//
// Windows: `mmap`/`MADV_HUGEPAGE`-shaped residency is POSIX-only here and the single-address premise does
// not hold on WDDM, so the GTT store DECLINES there and the run keeps today's behaviour.  Nothing in this tier
// is reachable on a discrete device (`prop.integrated` gates it), and it changes no shared code path.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>

// `MOE_EXPERT_GTT` (default 1): the kill switch.  0 = never offer a GTT store, i.e. today's behaviour on an
// integrated device verbatim (the expert cache stays disabled, nothing is registered, nothing is logged).
bool ggml_cuda_gtt_store_wanted();

// `MOE_EXPERT_GTT_RO` (default 1): register the region ReadOnly (the 232 GB/s arm).  0 = register it
// writable (the measured 60 GB/s arm) -- the A/B that proves the flag does something on THIS region.
bool ggml_cuda_gtt_store_ro();

// `MOE_EXPERT_GTT_MLOCK` (default 1): `mlock()` the region, which is what makes it a RESERVATION.  Probe 16
// (`wip/apu-expert-gtt/MEASUREMENTS.md` §15) measured that `cudaHostRegister` alone does NOT pin an `mmap`
// region: under memory pressure the kernel splits its huge pages and swaps 0-436 MiB per 2 GiB out, and the
// next GPU read pays a one-off 12-37x stall for the driver's re-fault.  `mlock` is also the only thing that
// makes the tier visible to the memory accounting (`Unevictable`/`Mlocked`, 0 without it).  0 = do not lock:
// the tier is then reclaimable by design, which is the kill switch for A/B and bisection.
bool ggml_cuda_gtt_store_wants_lock();

// How many bytes this process may lock (`RLIMIT_MEMLOCK`), or SIZE_MAX when unlimited.  Returns SIZE_MAX too
// when locking is disabled, because the caller's budget policy must then behave exactly as it did before
// locking existed.  The GTT store's own extent is CLAMPED to this by the model loader, and a model that only
// fits because of the store is refused when it cannot be locked at all.
size_t ggml_cuda_gtt_store_lockable();

// Open the GTT store on `device` with `extent_bytes` of capacity and record the budget's source string (the
// census prints it verbatim).  Does NOT allocate or register yet: `--fit` loads with `no_alloc`, and a probe
// load must not reserve 40 GiB of non-swappable RAM.  The region is created at the first real carve.
// Returns false (and fills `err`) when the GTT store is unavailable, already open with a different extent, or
// the platform cannot give it the one address the design needs.
bool ggml_cuda_gtt_store_open(int device, size_t extent_bytes, const char * source, char * err, size_t errn);

// Carve `size` bytes out of the GTT store (first fit; page-aligned; never released).  NULL when the GTT store is
// not open or the request does not fit -- the caller then declines the tensor rather than over-committing.
// This is the ALL-RESIDENT placement arm (P2): one carve per load, holding whole expert tensors.
//
// `reserve`/`release` are the ARENA arm (P3): the expert cache allocates a table's slot storage here at sizing
// time and gives it back when the table stands down or its arena is evicted, so the region is a real allocator
// and not a bump pointer.  Both arms share the one region, so a mode switch never needs a second reservation.
void * ggml_cuda_gtt_store_carve(int device, size_t size);
void * ggml_cuda_gtt_store_reserve(int device, size_t size);
void   ggml_cuda_gtt_store_release(int device, void * p, size_t size);

// The largest single run still free (0 when the store is not open).  The cache's arena sizing asks this to
// avoid a failed allocation, and the census prints it.
size_t ggml_cuda_gtt_store_largest_free(int device);

// For a SECOND load in the same process: how many bytes it may still place in the live store (SIZE_MAX when no
// store is open, i.e. "size your own").  See `ggml_cuda_gtt_store_free_bytes` in the .cu.
size_t ggml_cuda_gtt_store_free_bytes(int device);

// The one `-lv 4` census line: extent requested vs realised, the budget source, the registration mode, the
// huge-page granularity, the slots and the bytes the loader read from the GGUF into the GTT store.
void ggml_cuda_gtt_store_report(int device, size_t filled_bytes, int n_tables, int64_t n_slots);

// Unregister and release every GTT store (idempotent; also runs at process exit).  The region is `mmap`ed so the
// kernel reclaims it anyway, but the registration is a HOLD (probe 09) and must be dropped explicitly.
void ggml_cuda_gtt_store_free();
