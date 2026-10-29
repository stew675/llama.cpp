#pragma once

// wip/kv-residency (TODO #117): a grow-only VMM region for the KV cache -- the O1 "fixed region" primitive.
//
// WHY THIS EXISTS
//   The KV is a plain `cudaMalloc` (one allocation per buffer type, `llama-kv-cache.cpp:301-320`) sized for
//   the WHOLE context a user asked for, made BEFORE the movable-boundary slab is created, so `SLAB_INIT`
//   maps `free VRAM - reserve` and every token of context is taken out of the MoE expert arena 1:1
//   (measured: +3840 MiB of `-c` = -3.75 GiB of arena on 35B-A3B).  A region that RESERVES its VA once and
//   maps physical only for the cells the run actually reaches keeps the arithmetic exact at any context
//   length, and it makes the KV's size and address known before the graph is built.
//
// THE DRIVER FACTS THIS IS BUILT ON (measured, `wip/kv-residency/VMM-PROBE.md`)
//   * VA can be reserved without mapping, and a chunk can be mapped into it later  -> the whole point.
//   * ONE mapping may host several independent handles, and `hipMemUnmap` works on a chunk's own mapping
//     (neighbours stay intact) -- but a SUB-range of a mapping is refused (`hipErrorInvalidValue`).
//     Hence: map in whole `unit` chunks, and grow MONOTONICALLY.  Nothing here ever unmaps part of a chunk.
//   * create+map and unmap cost ~3-4 us regardless of the chunk size (it is lazy bookkeeping), so a grow
//     between two decode steps is free.
//   * a fresh mapping can contain STALE bytes and the first write to it runs at VRAM speed (490 GB/s), so
//     `grow()` memsets what it maps.  Never rely on the driver zeroing a mapping.
//   * READING RESERVED-BUT-UNMAPPED VA FAULTS THE GPU (it does not return an error).  The caller must never
//     let a kernel or a copy touch a byte that is not mapped, which is the same contract `llama_kv_cache`
//     already keeps for the graph view (it sizes it by the USED cell count, not `kv_size`).
//
// WHY THE MAPPED EXTENT IS A SET OF SPANS AND NOT ONE WATERMARK
//   The consumer is the KV buffer type, and a KV buffer lays its tensors out CONSECUTIVELY: layer 0's K, then
//   layer 0's V, then layer 1's K, ...  A prefix watermark covering `n_kv` cells therefore has to reach the
//   LAST tensor's offset, i.e. almost the whole region, even for a 4-token run (measured on a 27B at
//   -c 65536: 4092 of 4096 MiB for 256 cells).  What a run actually needs is the first `n_kv` cells of EACH
//   tensor -- 32 disjoint spans -- which is 64 MiB instead of 4 GiB at 8 MiB units.  So `grow_span()` maps
//   whole units inside one arbitrary byte range and `is_mapped()` answers for a range; `grow()` is kept as
//   the contiguous-waiter convenience form (`grow_span(r, 0, used)`).
//
// All entry points fail soft: a driver refusal returns false/nullptr and leaves nothing half-created, so a
// caller can fall back to the plain allocation path.  No `GGML_ASSERT` ends the process.

#include <stddef.h>
#include <stdint.h>

struct ggml_cuda_vmm_region;

// Reserve `bytes` of VA on `device` with nothing mapped.  `unit` is the map/unmap granularity (0 = 64 MiB,
// rounded up to the driver's recommended granularity).  Returns nullptr when VMM is unavailable or the
// driver refuses; `err` receives a one-line reason (may be nullptr).
ggml_cuda_vmm_region * ggml_cuda_vmm_region_create(int device, size_t bytes, size_t unit, char * err, size_t errsz);

// Map whole units until at least `used_bytes` are mapped from the base, and clear what it maps.  Monotone: a
// smaller `used_bytes` than what is already mapped is a no-op, and the region never shrinks.  false = the
// driver refused (or `used_bytes` exceeds the reservation); the region is left with what did map (those
// bytes are valid and cleared, so a caller can still run on a shorter extent).
bool ggml_cuda_vmm_region_grow(ggml_cuda_vmm_region * r, size_t used_bytes);

// Map whole units covering [off, off + bytes) and clear what it maps; a unit already mapped is left alone
// (and never re-cleared, so data survives).  Same monotone/false semantics as `grow()`.  This is the form a
// consumer with a strided layout uses: the mapped extent is a SET of spans, not a single watermark.
bool ggml_cuda_vmm_region_grow_span(ggml_cuda_vmm_region * r, size_t off, size_t bytes);

// Are all the units touching [off, off + bytes) mapped?  This is the read/write legality check: ANY access
// to a byte outside the mapped set faults the GPU, so a caller gates every copy on it.  A zero-length range
// is trivially "mapped" only when it does not itself start outside the reservation.
bool ggml_cuda_vmm_region_is_mapped(const ggml_cuda_vmm_region * r, size_t off, size_t bytes);

// memset every mapped byte (a buffer type's `clear` must not touch unmapped VA).
void ggml_cuda_vmm_region_clear(ggml_cuda_vmm_region * r, uint8_t value);

// Total bytes mapped (not a watermark -- see `is_mapped`) and the reservation size.
size_t ggml_cuda_vmm_region_mapped(const ggml_cuda_vmm_region * r);
size_t ggml_cuda_vmm_region_size  (const ggml_cuda_vmm_region * r);
size_t ggml_cuda_vmm_region_unit  (const ggml_cuda_vmm_region * r);
void * ggml_cuda_vmm_region_base  (const ggml_cuda_vmm_region * r);

// Unmap what is mapped, free the VA, destroy the object.
void   ggml_cuda_vmm_region_free  (ggml_cuda_vmm_region * r);

// Is VMM usable on this device at all (reserve + one map + unmap succeeded once)?  Cached per device.
bool   ggml_cuda_vmm_region_available(int device);

// Diagnostic (GGML_CUDA_VMM_REGION_SELFTEST=1, run from the CUDA backend init): exercise the primitive on
// this device -- reserve, grow in steps, verify a pattern per unit, verify the watermark contract, reach
// past the reservation (must refuse, not abort), and drain.  Logs one PASS/FAIL line.  Returns false on a
// failed check.  This exists because the primitive lives inside the backend library and nothing above it
// can call it before the KV buft lands.
bool   ggml_cuda_vmm_region_selftest(int device);
