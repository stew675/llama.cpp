// Compute-buffer sizing helpers (margin / chunk).  Split out of ggml-cuda.cu.
#include "ggml-cuda-arena.cuh"
#include "common.cuh"
#include "ggml-cuda-vmm.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>

// RDNA/ROCm (TODO #42): the graph allocator reserves the compute buffer from a *measure* graph, but a
// runtime graph can carry a different live-tensor set (host-expert staging, MTP taps) and need a little
// more -- measured +3.3 % (6564 -> 6780 MiB; the cli case was a 11765.52 MiB allocation).  Growing it is
// a free-then-allocate-larger, so it needs a contiguous block BIGGER than the one just released, which
// fails once the leftover VRAM belongs to the MoE expert-cache arena.  Taking the slack up front moves
// it to *before* the arena is sized.  Deliberately HIP-only: this repo is RDNA/ROCm-scoped and no other
// backend's allocation sizes change at all.  `GGML_COMPUTE_BUFFER_MARGIN_PCT=0` disables it.
size_t ggml_backend_cuda_buffer_type_get_compute_margin_pct(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
#if defined(GGML_USE_HIP)
    static int margin_pct = -1;
    if (margin_pct < 0) {
        const char * env = getenv("GGML_COMPUTE_BUFFER_MARGIN_PCT");
        margin_pct = env != NULL ? atoi(env) : 10;
        if (margin_pct < 0) {
            margin_pct = 0;
        }
    }
    return (size_t) margin_pct;
#else
    return 0;
#endif
}

// OPEN 2: uniform COMPUTE-buffer chunk size (bytes).  Chunk-quantizing the allocation (+ one spare chunk)
// absorbs a later graph's growth without a free-then-allocate-larger, and makes workspace and arena memory
// interchangeable units for the VMM pool.  HIP-only, like the margin above.  `GGML_COMPUTE_BUFFER_CHUNK_MIB`
// (MiB, default 256) sets it; 0 falls back to the percentage margin.
size_t ggml_backend_cuda_buffer_type_get_compute_chunk_bytes(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
#if defined(GGML_USE_HIP)
    static size_t chunk_mib = SIZE_MAX;
    if (chunk_mib == SIZE_MAX) {
        const char * env = getenv("GGML_COMPUTE_BUFFER_CHUNK_MIB");
        chunk_mib = env != NULL ? (size_t) atoll(env) : 256;
    }
    // Chunk-quantizing the compute buffer (`(ceil(need/C)+1)*C`) exists to keep the SLAB's work region in
    // whole units, so it makes sense only when the slab is in use.  With no slab (a dense model) it just
    // inflates small buffers -- a 178 MiB layout becomes 512 MiB -- so fall back to the percentage margin,
    // which is the pre-slab behaviour.  This is what removes the dense +739 MiB of issue #120.
    if (!ggml_cuda_slab_enabled() || ggml_cuda_slab_compute_plain()) {
        return 0;
    }
    return chunk_mib * 1024 * 1024;
#else
    return 0;
#endif
}
