// MoE expert-cache callbacks + op-offload staging hooks (block 06/09) and the FA/H2D staging
// accounting.  Split out of ggml-cuda.cu (an upstream hub).

#include "ggml-cuda-moe-cache-hooks.cuh"
#include "common.cuh"
#include "ggml-backend-impl.h"
#include "ggml-cuda-vmm.h"
#include "moe-expert-cache.h"

#include <algorithm>
#include <cstdint>
#include <cstring>

void * ggml_backend_cuda_stage_buffer(ggml_backend_t backend, int slot, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_cuda_set_device(cuda_ctx->device);
    return cuda_ctx->h2d_stage_buffer(slot, size);
}

// MoE expert cache: whole-table staging from the cache arena.  Fills the ring slot from the immutable
// prefill residency snapshot (an arena slot for a resident expert, the pinned host master for the rest),
// so only the non-resident experts cross PCIe; declines (false) when the table is not
// cache-managed/usable, so the caller falls back to the plain upload.  Kill switch
// `GGML_MOE_CACHE_STAGE=0`.  The deferred (in-place) read of the r36 design is not used: `defer` is
// ignored and the destination is always filled.
static bool ggml_backend_cuda_stage_from_cache(ggml_backend_cuda_context * cuda_ctx, const void * src, size_t offset,
        size_t width, size_t stride_src, size_t n_copies, void * dst, ggml_backend_event_t ev, bool defer) {
    GGML_UNUSED(defer);
    if (!moe_cache_enabled()) {
        return false;
    }
    static cudaEvent_t order_ev[GGML_CUDA_MAX_DEVICES] = {};
    const int dev = cuda_ctx->device;
    if (order_ev[dev] == nullptr && cudaEventCreateWithFlags(&order_ev[dev], cudaEventDisableTiming) != cudaSuccess) {
        (void) cudaGetLastError();
        order_ev[dev] = nullptr;
        return false;
    }
    cudaStream_t s = cuda_ctx->copy_stream();
    if (!moe_cache_stage_table(dev, src, offset, width, stride_src, n_copies, dst,
                               cuda_ctx->stream(), s, order_ev[dev])) {
        return false;
    }
    CUDA_CHECK(cudaEventRecord((cudaEvent_t) ev->context, s));
    return true;
}

void ggml_backend_cuda_stage_upload(ggml_backend_t backend, void * dst, const void * data, size_t size, ggml_backend_event_t ev) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_cuda_set_device(cuda_ctx->device);
    // Deferred unless the scheduler's redirect mode (GGML_SCHED_STAGE_MODE=1) points the input at the slot: every
    // other caller (the scheduler's ring in its default mode, the meta backend's mirrored upload) copies the slot
    // into the input with stage_d2d, where the deferral is resolved.
    static const bool redirect = getenv("GGML_SCHED_STAGE_MODE") != nullptr && atoi(getenv("GGML_SCHED_STAGE_MODE")) != 0;
    if (ggml_backend_cuda_stage_from_cache(cuda_ctx, data, 0, size, size, 1, dst, ev, /*defer =*/ !redirect)) {
        return;
    }
    cudaStream_t s = cuda_ctx->copy_stream();
    CUDA_CHECK(cudaMemcpyAsync(dst, data, size, cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaEventRecord((cudaEvent_t) ev->context, s));
}

// wip/tensor-split-expert-split: a split device's slice is strided in the source weight.  Assemble it
// into the ring slot on the copy stream -- no pageable H2D 2-D copy (pathologically slow on ROCm) and no
// host-blocking transfer.  Two shapes, chosen by how fine-grained the slice is:
//   * coarse (`n` small, e.g. ffn_gate/up: one block per expert): gather on the host into pinned staging
//     and issue ONE 1-D H2D.
//   * fine (`n` huge, e.g. ffn_down: a block per (n_embd, expert) row, 500k+ of them): a per-block host
//     gather is hundreds of thousands of `memcpy` calls, so H2D the contiguous range once into the device
//     scratch and let a device D2D 2-D copy do the compaction.  Prefer the host gather where it is cheap
//     because it moves only the device's half, while the scratch path moves the whole range.
bool ggml_backend_cuda_stage_gather(ggml_backend_t backend, int slot, const void * src, size_t offset, size_t width, size_t stride_src, size_t n_copies, ggml_backend_event_t ev, bool src_pinned) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_cuda_set_device(cuda_ctx->device);
    const size_t total = width * n_copies;                 // the device's compacted slice
    void * dst = cuda_ctx->h2d_stage_buffer(slot, total);
    if (dst == nullptr) {
        return false;
    }
    if (ggml_backend_cuda_stage_from_cache(cuda_ctx, src, offset, width, stride_src, n_copies, dst, ev, /*defer =*/ true)) {
        return true;
    }
    cudaStream_t s = cuda_ctx->copy_stream();
    // WIP r42 (stage-1 item 2): a pinned source lets the compacted slice be copied straight with one
    // 2-D H2D, moving only `total` bytes instead of the whole contiguous range.  The pageable 2-D H2D
    // was the pathological ROCm case (the campaign's `hipMemcpy2DAsync` fault); this is the pinned
    // form the source-pinning fix made possible.  `GGML_STAGE_GATHER_SCRATCH=1` forces the old path.
    static const bool force_scratch = getenv("GGML_STAGE_GATHER_SCRATCH") != nullptr && atoi(getenv("GGML_STAGE_GATHER_SCRATCH")) != 0;
    if (src_pinned && !force_scratch) {
        CUDA_CHECK(cudaMemcpy2DAsync(dst, width, (const char *) src + offset, stride_src, width, n_copies, cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaEventRecord((cudaEvent_t) ev->context, s));
        return true;
    }
    // A host gather when the per-block compaction is cheap (few copies), else stage the whole
    // contiguous range and pick this device's `width`-byte block out of every `stride_src` on the copy
    // stream (the F2D compaction hides with the H2D).
    const bool host_gather = n_copies <= 4096;
    if (host_gather) {
        void * pin = cuda_ctx->h2d_pin_buffer(slot, total);
        if (pin == nullptr) {
            return false;
        }
        // the pinned slot may still be read by the copy that last targeted this ring slot
        CUDA_CHECK(cudaEventSynchronize(cuda_ctx->h2d_pin_ev[slot]));
        for (size_t i = 0; i < n_copies; ++i) {
            memcpy((char *) pin + i*width, (const char *) src + offset + i*stride_src, width);
        }
        CUDA_CHECK(cudaMemcpyAsync(dst, pin, total, cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaEventRecord(cuda_ctx->h2d_pin_ev[slot], s));
    } else {
        // stage the whole contiguous range, then pick this device's `width`-byte block out of every
        // `stride_src` (the F2D compaction runs on the copy stream, so it hides with the H2D)
        const size_t whole = stride_src * n_copies;
        void * scratch = cuda_ctx->h2d_scratch(whole);
        if (scratch == nullptr) {
            return false;
        }
        CUDA_CHECK(cudaMemcpyAsync(scratch, src, whole, cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaMemcpy2DAsync(dst, width, (const char *) scratch + offset, stride_src, width, n_copies, cudaMemcpyDeviceToDevice, s));
    }
    CUDA_CHECK(cudaEventRecord((cudaEvent_t) ev->context, s));
    return true;
}

void ggml_backend_cuda_stage_wait(ggml_backend_t backend, ggml_backend_event_t ev) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_cuda_set_device(cuda_ctx->device);
    CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->copy_stream(), (cudaEvent_t) ev->context, 0));
}

void ggml_backend_cuda_stage_d2d(ggml_backend_t backend, void * dst, const void * src, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_cuda_set_device(cuda_ctx->device);
    CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, cuda_ctx->stream()));
}

float ggml_backend_cuda_stage_h2d_gbps(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_cuda_set_device(cuda_ctx->device);

    // One-off H2D bandwidth calibration (issue #50 WIP), cached per device.  Sized past the Infinity
    // Cache (a 64 MiB probe reads ~25 GB/s on a x4 link because the L3 serves it).  Timed with a
    // synchronous copy so it needs no event API (this toolchain does not alias cudaEventCreate /
    // cudaEventElapsedTime).
    static float bw[GGML_CUDA_MAX_DEVICES];
    static bool  done[GGML_CUDA_MAX_DEVICES] = {};
    const int dev = cuda_ctx->device;
    if (done[dev]) {
        return bw[dev];
    }
    done[dev] = true;
    bw[dev] = 0.0f;

    const size_t sz = 512u << 20;
    void * h = malloc(sz);
    void * d = nullptr;
    // wip/slab-transient-absorption: a ONE-OFF 512 MiB probe run at load, AFTER the slab exists.  Serving
    // it from the slab's arena (allocated before the cache sizes itself, then freed) removes a 512 MiB
    // spike from the outside reserve -- the single largest reason a low GGML_CUDA_SLAB_HEADROOM_MIB is
    // unsafe today.  The arena is the right home here (the ring/wide band is not armed at load); the block
    // is released immediately, so it cannot be evicted mid-flight.
    bool d_slab = false;
    if (ggml_cuda_slab_active(dev)) {
        d = ggml_cuda_slab_arena_alloc(dev, sz);
        d_slab = d != nullptr;
    }
    if (h == nullptr || (d == nullptr && cudaMalloc(&d, sz) != cudaSuccess)) {
        (void) cudaGetLastError(); // clear the sticky error
        if (d != nullptr) {
            if (d_slab) {
                ggml_cuda_slab_arena_free(dev, d, sz);
            } else {
                CUDA_CHECK(cudaFree(d));
            }
        }
        free(h);
        return bw[dev];
    }
    memset(h, 1, sz);
    CUDA_CHECK(cudaMemcpy(d, h, sz, cudaMemcpyHostToDevice)); // warmup + fault the pages
    CUDA_CHECK(cudaDeviceSynchronize());

    const int64_t t0 = ggml_time_us();
    for (int i = 0; i < 3; ++i) {
        CUDA_CHECK(cudaMemcpy(d, h, sz, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const int64_t t1 = ggml_time_us();

    const double sec = double(t1 - t0) / 1e6;
    bw[dev] = sec > 0.0 ? float(3.0*double(sz)/1e9 / sec) : 0.0f;
    GGML_LOG_INFO("%s: H2D bandwidth calibration: %.1f GB/s (%zu MiB x3 in %.2f ms, from %s)\n", __func__, double(bw[dev]), sz >> 20, 1000.0*sec, d_slab ? "the slab arena" : "cudaMalloc");
    if (d_slab) {
        ggml_cuda_slab_arena_free(dev, d, sz);
    } else {
        CUDA_CHECK(cudaFree(d));
    }
    free(h);
    return bw[dev];
}

bool ggml_backend_cuda_moe_cache_update(ggml_backend_t backend, const ggml_tensor * weight, const ggml_tensor * weight_cpy, const int32_t * ids, int64_t n_used, int64_t n_tok, size_t ids_nb0, size_t ids_nb1, size_t slice_off, int split_axis, const struct ggml_backend_moe_slice * slices, int n_slices) {
    ggml_backend_cuda_context * ctx = (ggml_backend_cuda_context *) backend->context;
    return moe_cache_update_host(weight, weight_cpy, ids, n_used, n_tok, ids_nb0, ids_nb1, ctx->stream(), ctx->device, slice_off, split_axis, slices, n_slices);
}

// Session 7 identity fast path (see moe-expert-cache.h).  Called by the scheduler before it reads the
// routing ids back to the host: a true lets it skip the readback, the full device synchronize it forces,
// the used-expert pruning and the copy, because the consumer reads the compact arena with the raw ids.
bool ggml_backend_cuda_moe_cache_take_over(ggml_backend_t backend, const ggml_tensor * weight, const ggml_tensor * weight_cpy) {
    ggml_backend_cuda_context * ctx = (ggml_backend_cuda_context *) backend->context;
    return moe_cache_take_over(weight, weight_cpy, ctx->device);
}

// B2: device-side host-weight expert gather for an offloaded `MUL_MAT_ID` prefill (see the iface comment).
int64_t ggml_backend_cuda_moe_cache_band(ggml_backend_t backend) {
    const ggml_backend_cuda_context * cuda_ctx = (const ggml_backend_cuda_context *) backend->context;
    return moe_cache_max_tok_dev(cuda_ctx->device);
}

bool ggml_backend_cuda_moe_cache_gather(ggml_backend_t backend, const ggml_tensor * weight, const ggml_tensor * weight_cpy, const ggml_tensor * ids, size_t slice_off, int split_axis) {
    ggml_backend_cuda_context * ctx = (ggml_backend_cuda_context *) backend->context;
    return moe_cache_gather_host(weight, weight_cpy, ids, ctx->stream(), ctx->device, slice_off, split_axis);
}

// --fit / memory-breakdown accounting for the FA prefill staging arena (block 15).  The arena is a
// raw device allocation outside the compute-graph reserve, so `llama_get_memory_breakdown` (and
// therefore --fit) cannot see it and can under-provision a deep prefill.  `bound` returns the
// worst-case per-launch staging request for the device: 0 when this device never prefill-stages
// (RDNA3_5 reads natively), SIZE_MAX when the cap is unbounded (the caller supplies an estimate),
// otherwise 2*cap because K and V are capped independently.  `used` is what the arena has grown to.
size_t ggml_backend_cuda_fattn_stage_bound(ggml_backend_t backend) {
    const ggml_backend_cuda_context * ctx = (const ggml_backend_cuda_context *) backend->context;
    if (ctx == nullptr) {
        return 0;
    }
    const int cc = ggml_cuda_info().devices[ctx->device].cc;
    if (GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        return 0;
    }
    const size_t cap = ggml_cuda_fattn_stage_max_bytes();
    if (cap == 0) {
        return SIZE_MAX;
    }
    return 2*cap;
}

size_t ggml_backend_cuda_fattn_stage_used(ggml_backend_t backend) {
    const ggml_backend_cuda_context * ctx = (const ggml_backend_cuda_context *) backend->context;
    return ctx != nullptr ? ctx->fattn_stage_bytes() : 0;
}

// --fit / memory-breakdown accounting for the op-offload H2D staging ring (block 06).  Like the FA
// arena it is a raw device allocation outside the compute-graph reserve, so
// `llama_get_memory_breakdown` (and therefore --fit) would not see it; the auto-sized budget can grow
// it to several GiB for a large host-resident expert table.  `used` is the live ring; `bound` is
// `slots x largest host table`, the worst case auto-sizing can request for this model.
size_t ggml_backend_cuda_h2d_stage_used(ggml_backend_t backend) {
    const ggml_backend_cuda_context * ctx = (const ggml_backend_cuda_context *) backend->context;
    return ctx != nullptr ? ctx->h2d_stage_bytes() : 0;
}

size_t ggml_backend_cuda_h2d_stage_bound(ggml_backend_t backend, size_t max_table) {
    const ggml_backend_cuda_context * ctx = (const ggml_backend_cuda_context *) backend->context;
    if (ctx == nullptr) {
        return 0;
    }
    // wip/slab-ring-region: when the movable-boundary slab is enabled (default) the ring lives INSIDE it
    // as a reserved hole sized from this same bound, so counting it here as an outside allocation would
    // reserve it twice.  `GGML_CUDA_SLAB=0` (or a dense model, max_table == 0) keeps the old accounting.
    const char * env = getenv("GGML_CUDA_SLAB");
    if ((env == NULL || atoi(env) != 0) && max_table > 0) {
        return 0;
    }
    return ggml_backend_cuda_context::h2d_stage_bound(max_table);
}