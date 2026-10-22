#include "ggml-cuda.h"
#include "ggml-impl.h"
#include "ggml-moe-weighted-reduction.h"
#include "ggml-backend-impl.h"

#include "ggml-cuda/allreduce.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/acc.cuh"
#include "ggml-cuda/add-id.cuh"
#include "ggml-cuda/arange.cuh"
#include "ggml-cuda/argmax.cuh"
#include "ggml-cuda/argsort.cuh"
#include "ggml-cuda/binbcast.cuh"
#include "ggml-cuda/clamp.cuh"
#include "ggml-cuda/col2im-1d.cuh"
#include "ggml-cuda/concat.cuh"
#include "ggml-cuda/conv-transpose-1d.cuh"
#include "ggml-cuda/conv2d.cuh"
#include "ggml-cuda/conv2d-dw.cuh"
#include "ggml-cuda/conv2d-transpose.cuh"
#include "ggml-cuda/conv3d.cuh"
#include "ggml-cuda/convert.cuh"
#include "ggml-cuda/count-equal.cuh"
#include "ggml-cuda/cpy.cuh"
#include "ggml-cuda/cross-entropy-loss.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/diagmask.cuh"
#include "ggml-cuda/diag.cuh"
#include "ggml-cuda/fattn.cuh"
#include "ggml-cuda/fattn-qsa.cuh"
#include "ggml-cuda/indexer-topk.cuh"
#include "ggml-cuda/indexer-score.cuh"
#include "ggml-cuda/fwht.cuh"
#include "ggml-cuda/gdn-conv.cuh"
#include "ggml-cuda/ple-conv.cuh"
#include "ggml-cuda/getrows.cuh"
#include "ggml-cuda/im2col.cuh"
#include "ggml-cuda/mmf.cuh"
#include "ggml-cuda/mmb.cuh"
#include "ggml-cuda/moe-expert-cache.h"
#include "ggml-cuda/cpy-batch.cuh"
#include "ggml-cuda/mmq.cuh"
#include "ggml-cuda/mmvf.cuh"
#include "ggml-cuda/mmvq.cuh"
#include "ggml-cuda/moe-weighted-reduction.cuh"
#include "ggml-cuda/mmf32.cuh"
#include "ggml-cuda/norm.cuh"
#include "ggml-cuda/norm-gated.cuh"
#include "ggml-cuda/opt-step-adamw.cuh"
#include "ggml-cuda/opt-step-sgd.cuh"
#include "ggml-cuda/out-prod.cuh"
#include "ggml-cuda/pad.cuh"
#include "ggml-cuda/pool2d.cuh"
#include "ggml-cuda/pool1d.cuh"
#include "ggml-cuda/quantize.cuh"
#include "ggml-cuda/rope.cuh"
#include "ggml-cuda/roll.cuh"
#include "ggml-cuda/scale.cuh"
#include "ggml-cuda/snake.cuh"
#include "ggml-cuda/softcap.cuh"
#include "ggml-cuda/softmax.cuh"
#include "ggml-cuda/ssm-conv.cuh"
#include "ggml-cuda/ssm-scan.cuh"
#include "ggml-cuda/sum.cuh"
#include "ggml-cuda/sumrows.cuh"
#include "ggml-cuda/top-k.cuh"
#include "ggml-cuda/mean.cuh"
#include "ggml-cuda/tsembd.cuh"
#include "ggml-cuda/topk-moe.cuh"
#include "ggml-cuda/unary.cuh"
#include "ggml-cuda/upscale.cuh"
#include "ggml-cuda/wkv.cuh"
#include "ggml-cuda/gla.cuh"
#include "ggml-cuda/gated_delta_net.cuh"
#include "ggml-cuda/dsv4-hc.cuh"
#include "ggml-cuda/hc-mix.cuh"
#include "ggml-cuda/hyperconn.cuh"
#include "ggml-cuda/set.cuh"
#include "ggml-cuda/set-rows.cuh"
#include "ggml-cuda/pad_reflect_1d.cuh"
#include "ggml-cuda/solve_tri.cuh"
#include "ggml-cuda/tri.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/fill.cuh"
#include "ggml-cuda/lightning-indexer.cuh"
#include "ggml-cuda/top-k-mask.cuh"
#include "ggml.h"

#include <algorithm>
#include <numeric>
#include <array>
#include <atomic>
#include <charconv>
#include <cinttypes>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cfloat>
#include <initializer_list>
#include <limits>
#include <map>
#include <set>
#include <memory>
#include <mutex>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>
#include "ggml-cuda-try-fuse.cuh"
#include "ggml-cuda-vmm.h"
#include "ggml-cuda-moe-cache-hooks.cuh"
#include "ggml-cuda-arena.cuh"

static_assert(sizeof(half) == sizeof(ggml_fp16_t), "wrong fp16 size");

#define GGML_LOG_WARN_ONCE(str) \
    { static std::once_flag warn_flag; std::call_once(warn_flag, []() { GGML_LOG_WARN(str); }); }

[[noreturn]]
void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    int id = -1; // in case cudaGetDevice fails
    (void)cudaGetDevice(&id);

    GGML_LOG_ERROR(GGML_CUDA_NAME " error: %s\n", msg);
    GGML_LOG_ERROR("  current device: %d, in function %s at %s:%d\n", id, func, file, line);
    GGML_LOG_ERROR("  %s\n", stmt);
    // abort with GGML_ABORT to get a stack trace
    GGML_ABORT(GGML_CUDA_NAME " error");
}

// map a (possibly virtual) device id to the physical CUDA device that backs it
int ggml_cuda_get_physical_device(int device) {
    const ggml_cuda_device_info & info = ggml_cuda_info();
    GGML_ASSERT(device >= 0 && device < info.device_count);
    return info.devices[device].physical_device;
}

// this is faster on Windows
// probably because the Windows CUDA libraries forget to make this check before invoking the drivers
void ggml_cuda_set_device(int device) {
    // translate the (possibly virtual) device id to the physical CUDA device that backs it
    const int physical_device = ggml_cuda_get_physical_device(device);

    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));

    if (physical_device == current_device) {
        return;
    }

    CUDA_CHECK(cudaSetDevice(physical_device));
}

int ggml_cuda_get_device() {
    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

static bool ggml_cuda_env_flag(const char * name) {
    const char * env = getenv(name);
    return env != nullptr && std::atoi(env) != 0;
}

static int ggml_cuda_env_int(const char * name, int default_value) {
    const char * env = getenv(name);
    return env != nullptr ? std::atoi(env) : default_value;
}

static bool ggml_cuda_env_set(const char * name) {
    return getenv(name) != nullptr;
}

static bool ggml_cuda_unified_memory_enabled() {
    static const bool enabled = ggml_cuda_env_flag("GGML_CUDA_ENABLE_UNIFIED_MEMORY");
    return enabled;
}

static bool ggml_cuda_pinned_memory_disabled() {
    static const bool disabled = ggml_cuda_env_flag("GGML_CUDA_NO_PINNED");
    return disabled;
}

static bool ggml_cuda_register_host_enabled() {
    static const bool enabled = ggml_cuda_env_flag("GGML_CUDA_REGISTER_HOST");
    return enabled;
}

bool ggml_cuda_fusion_disabled() {
    static const bool disabled = ggml_cuda_env_flag("GGML_CUDA_DISABLE_FUSION");
    return disabled;
}

static bool ggml_cuda_p2p_enabled() {
    static const bool enabled = ggml_cuda_env_set("GGML_CUDA_P2P");
    return enabled;
}

static int ggml_cuda_op_offload_min_batch() {
    static const int min_batch = ggml_cuda_env_int("GGML_OP_OFFLOAD_MIN_BATCH", 32);
    return min_batch;
}

// Issue/TODO #42: shared device-allocation helper.  It yields the lowest-priority MoE expert-cache
// arena when an allocation cannot otherwise be satisfied, so EVERY device allocation inherits the same
// fail-soft policy (defined in ggml-cuda.cu).  The slab query is declared here because the slab itself
// is defined further down this TU.
bool ggml_cuda_slab_active(int device);
bool ggml_cuda_slab_enabled();

cudaError_t ggml_cuda_device_malloc(void ** ptr, size_t size, int device);

// Opt-in device budget: refuse an allocation that would leave the device with less than
// GGML_CUDA_MIN_FREE_MIB free (0 = off, the default).  The failure is reported exactly like a driver
// out-of-memory, so every fail-soft path above it (the MoE arena's shrink/release, the workspace pool's
// flush, the FA staging stand-down) runs unchanged and the run ends with a clear message.
//
// WHY: on Windows/WDDM a device allocation that over-commits is NOT refused -- it is paged into shared
// system memory and the loss surfaces later, as `unspecified launch failure` or a bugcheck inside dxgkrnl
// (issue #219: a 16 GiB card, a 12 GiB model and -c 65536).  `cudaMemGetInfo` still reports the real
// budget there, so a floor set below the card's usable VRAM keeps the process off that path.  On Linux
// the same config does not fault: it aborts with `ROCm error: out of memory`.  (The "~30x slower in a
// narrow band" that r135 saw here is NOT an over-commit: HIP refuses an allocation that does not fit and
// never demotes a resident working set, and what a squeezed driver does instead is migrate another
// client's buffers out of VRAM, which costs this run PCIe time -- measured and closed as TODO #103,
// archive/work/vram-limit-103/.)  Either way the floor makes the outcome deterministic instead.
//
// Skipped while the movable-boundary slab is armed: the slab parks nearly all of the device's free VRAM
// by design, so a floor read from `cudaMemGetInfo` would refuse every legitimate outside-the-slab
// allocation on an MoE config.
static bool ggml_cuda_min_free_refuses(size_t size, int device) {
    // read the value once: a cached getenv() pointer is not safe on Windows (the CRT buffer is reused)
    static const long long floor_mib = [] {
        const char * env = getenv("GGML_CUDA_MIN_FREE_MIB");
        return env != nullptr ? atoll(env) : 0;
    }();
    if (floor_mib <= 0 || ggml_cuda_slab_enabled()) {
        return false;
    }
    size_t f = 0;
    size_t t = 0;
    if (cudaMemGetInfo(&f, &t) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;   // fail-soft: never invert a working run on a driver hiccup
    }
    const size_t floor = (size_t) floor_mib << 20;
    if (f >= size + floor) {
        return false;
    }
    static bool warned = false;
    if (!warned) {
        warned = true;
        GGML_LOG_WARN("%s: refusing a %.2f MiB device allocation on device %d: it would leave %.2f MiB free, "
                      "below the GGML_CUDA_MIN_FREE_MIB budget of %lld MiB.  Lower the context / -ngl, or "
                      "lower the budget.  (Reported as an out-of-memory error; the fail-soft paths above "
                      "this point still run.)\n",
                      __func__, size / 1024.0 / 1024.0, device, f / 1024.0 / 1024.0, floor_mib);
    }
    return true;
}

// Diagnostic (GGML_CUDA_LOW_FREE_MIB, default 512, 0 = off): warn ONCE when a device allocation leaves
// less than that much VRAM free.  Nothing is refused -- this is the number a "it is slow" / "it aborted"
// report is missing, and the state it names is the one issue #219 reported: at a few MiB of headroom the
// next growth fails, Windows/WDDM pages the over-committing allocation instead of refusing it (a later
// access then faults -- bugcheck 0x7E inside dxgkrnl), and on any OS a squeezed driver may have to
// migrate another client's buffers out of VRAM while the run is live, which costs this run PCIe time.
//
// WHY the low-water mark is kept on every config since r167: until then it was sampled only while the
// movable-boundary slab was active and printed only under GGML_CUDA_SLAB_STATS, so a dense/plain run --
// the shape TODO #103 was found on -- recorded nothing about how close it came.  Skipped while the slab
// is armed: the slab parks nearly all of the device's free VRAM by design, so the reading says nothing.
static void ggml_cuda_low_free_warn(int device, size_t size, size_t free_b, size_t total_b) {
    static const long long low_mib = [] {
        const char * env = getenv("GGML_CUDA_LOW_FREE_MIB");
        return env != nullptr ? atoll(env) : 512;
    }();
    if (low_mib <= 0 || ggml_cuda_slab_enabled()) {
        return;
    }
    if (free_b >= ((size_t) low_mib << 20)) {
        return;
    }
    static bool warned[GGML_CUDA_MAX_DEVICES] = {};
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES || warned[device]) {
        return;
    }
    warned[device] = true;
    GGML_LOG_WARN("%s: device %d is at its VRAM limit: %.2f MiB free of %.2f MiB after a %.2f MiB allocation.  "
                  "The next growth can fail with out-of-memory (on Windows/WDDM the allocation is paged "
                  "instead and a later access can fault -- issue #219), and a driver that has to migrate "
                  "another client's buffers to make room costs this run PCIe time.  Lower -c / -ub / -ngl or "
                  "use a smaller quant; GGML_CUDA_MIN_FREE_MIB=<MiB> turns this into a deterministic refusal "
                  "first.  (Diagnostic only -- silence it with GGML_CUDA_LOW_FREE_MIB=0.)\n",
                  __func__, device, free_b / 1024.0 / 1024.0, total_b / 1024.0 / 1024.0, size / 1024.0 / 1024.0);
}

// One-shot report for a device allocation the driver refused after the MoE-arena yield above it has already
// tried.  It names the free/total and the run's low-water mark, which is what turns "ROCm error: out of
// memory" into something actionable.  The slab's own message is more specific, so this stands down for a
// slab-armed device.
static void ggml_cuda_oom_warn(int device, size_t size) {
    if (ggml_cuda_slab_enabled()) {
        return;
    }
    static bool warned[GGML_CUDA_MAX_DEVICES] = {};
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES || warned[device]) {
        return;
    }
    warned[device] = true;
    size_t f = 0;
    size_t t = 0;
    const bool measured = cudaMemGetInfo(&f, &t) == cudaSuccess;
    if (!measured) {
        (void) cudaGetLastError();
    }
    GGML_LOG_WARN("%s: a %.2f MiB device allocation failed on device %d: %.2f MiB free of %.2f MiB, run "
                  "low-water mark %.2f MiB.  Lower -c / -ub / -ngl or use a smaller quant; "
                  "GGML_CUDA_MIN_FREE_MIB=<MiB> refuses earlier with a message like this one, and "
                  "GGML_CUDA_LOW_FREE_MIB (default 512) warns before the run reaches this state.\n",
                  __func__, size / 1024.0 / 1024.0, device, f / 1024.0 / 1024.0, t / 1024.0 / 1024.0,
                  ggml_cuda_free_min_bytes(device) / 1024.0 / 1024.0);
}

// Diagnostic (GGML_CUDA_SLAB_STATS): the largest single allocation made through the device-allocation
// choke point AFTER the slab became active.  Everything counted here is outside the slab by construction,
// so it is the number to compare against the reserve: the 400 MiB Meta all-reduce temp that a 70k-token
// prefill needs was invisible in the old accounting.  Relaxed max, one update per allocation.
static std::atomic<size_t> g_outside_alloc_max[GGML_CUDA_MAX_DEVICES];
// Largest allocation the choke point REFUSED after the slab became active: this is the number to compare
// against the outside reserve (a failed 400 MiB Meta all-reduce temp is the symptom of a short reserve).
static std::atomic<size_t> g_outside_fail_max[GGML_CUDA_MAX_DEVICES];
// Low-water mark of the free VRAM OUTSIDE the slab, per device, sampled at every large device allocation
// and at every slab stats point (GGML_CUDA_SLAB_STATS).  This is the real margin the reserve gives: the
// wide-arm / wide-drop lines only bracket the prefill, the dip can be in the middle of it.
static std::atomic<size_t> g_free_min[GGML_CUDA_MAX_DEVICES];
static std::atomic<bool>   g_free_min_set[GGML_CUDA_MAX_DEVICES];

size_t ggml_cuda_free_min_bytes(int device) {
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES || !g_free_min_set[device].load(std::memory_order_relaxed)) {
        return 0;
    }
    return g_free_min[device].load(std::memory_order_relaxed);
}

void ggml_cuda_free_min_sample(int device, size_t free_b) {
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) {
        return;
    }
    if (!g_free_min_set[device].load(std::memory_order_relaxed)) {
        size_t first = 0;
        (void) g_free_min[device].compare_exchange_strong(first, free_b, std::memory_order_relaxed);
        g_free_min_set[device].store(true, std::memory_order_relaxed);
    }
    size_t prev = g_free_min[device].load(std::memory_order_relaxed);
    while (free_b < prev &&
           !g_free_min[device].compare_exchange_weak(prev, free_b, std::memory_order_relaxed)) {
    }
}

size_t ggml_cuda_outside_alloc_max_bytes(int device) {
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) {
        return 0;
    }
    return g_outside_alloc_max[device].load(std::memory_order_relaxed);
}

size_t ggml_cuda_outside_fail_max_bytes(int device) {
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) {
        return 0;
    }
    return g_outside_fail_max[device].load(std::memory_order_relaxed);
}

static void ggml_cuda_outside_fail_note(int device, size_t size) {
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES || size < ((size_t) 1 << 20)) {
        return;
    }
    if (!ggml_cuda_slab_active(device)) {
        return;
    }
    size_t prev = g_outside_fail_max[device].load(std::memory_order_relaxed);
    while (size > prev && !g_outside_fail_max[device].compare_exchange_weak(prev, size, std::memory_order_relaxed)) {
    }
}

static void ggml_cuda_outside_alloc_note(int device, size_t size) {
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES || size < ((size_t) 1 << 20)) {
        return;
    }
    if (!ggml_cuda_slab_active(device)) {
        return;   // weights / KV load before the slab: not part of the outside cushion
    }
    size_t prev = g_outside_alloc_max[device].load(std::memory_order_relaxed);
    while (size > prev && !g_outside_alloc_max[device].compare_exchange_weak(prev, size, std::memory_order_relaxed)) {
    }
}

cudaError_t ggml_cuda_device_malloc(void ** ptr, size_t size, int device) {
    ggml_cuda_set_device(device);   // the floor reads the free VRAM of the CURRENT device
    if (ggml_cuda_min_free_refuses(size, device)) {
        return cudaErrorMemoryAllocation;
    }
    cudaError_t err;
    if (ggml_cuda_unified_memory_enabled()) {
        err = cudaMallocManaged(ptr, size);
#if defined(GGML_USE_HIP)
        if (err == hipSuccess) {
            // hipMemAdviseSetCoarseGrain is an optional performance hint;
            // ignore errors (e.g. hipErrorInvalidValue on some APU/iGPU configs).
            (void)cudaMemAdvise(*ptr, size, hipMemAdviseSetCoarseGrain, device);
            (void)hipGetLastError(); // clear any error
        }

        // fall back to cudaMalloc if not supported (e.g. on Windows)
        if (err == hipErrorNotSupported) {
            static bool warned_unsupported = false;
            if (!warned_unsupported) {
                GGML_LOG_WARN("hipMallocManaged unsupported, falling back to hipMalloc.\n");
                warned_unsupported = true;
            }

            err = cudaMalloc(ptr, size);
        }
#endif // defined(GGML_USE_HIP)
    } else {
        err = cudaMalloc(ptr, size);
    }
    if (err == cudaErrorMemoryAllocation) {
        (void) cudaGetLastError();
        // OPEN 2 slab: with the movable-boundary slab the arena's physical is MAPPED INTO the slab, so
        // freeing a table returns its bytes to the slab's free list -- never to the driver -- and ROCm
        // will not unmap a sub-range of the slab's single mapping.  Churning the cache here cannot make
        // the retry succeed and only throws away residency, so say what actually helps instead: whatever
        // this allocation needs must have been reserved OUTSIDE the slab at creation time.
        // Issue #136 (item D): guard on the predicate that chose the arena backing (`use_slab` is the
        // global `ggml_cuda_slab_enabled()`), not the failing device's own slab.  When the slab is armed
        // the arena's physical is mapped into it on every device, so freeing tables cannot return bytes to
        // the driver and the stand-down loop below only throws residency away.  (`ggml_cuda_slab_active`
        // let a device whose slab failed to initialise enter that loop even though every table is
        // slab-backed.)
        if (ggml_cuda_slab_enabled()) {
            GGML_LOG_WARN("%s: %.2f MiB allocation failed on device %d while the movable-boundary slab holds "
                          "the VRAM; the arena cannot yield physical back to the driver.  Raise "
                          "GGML_CUDA_SLAB_RESERVE_MIB (or disable the slab with GGML_CUDA_SLAB=0) so the "
                          "weights / KV cache / workspaces stay outside it.\n",
                          __func__, size / 1024.0 / 1024.0, device);
            moe_cache_validate("device-malloc-yield");
            ggml_cuda_outside_fail_note(device, size);
            return err;
        }
        // WIP r42 (TODO #42) fail-soft policy at the ONE choke point every device allocation can share.
        // The MoE expert-cache arena is the LOWEST-priority VRAM consumer, so it yields -- largest table
        // first (usually a few hundred MiB is enough for a grow-in-place realloc), then whole -- before an
        // allocation is allowed to fail.  Measured: the compute buffer's growth, the mmq workspace pool
        // (`ggml_cuda_pool_leg::alloc`) and the Q8_1 cache arena (`q8_1_cache_get`) can each be the
        // allocation that runs when the arena holds the last free VRAM, so patching them one by one is a
        // losing game; the policy belongs here.  Every allocator that already has its own recovery (the
        // workspace pool flushes its cached blocks) keeps it -- this simply adds the arena to the pool of
        // things that can be given back.  Nothing changes when the cache is off.
        while (moe_cache_shrink_step(device)) {
            GGML_LOG_WARN("%s: %.2f MiB allocation failed on device %d; shrank the MoE arena and retrying\n",
                          __func__, size / 1024.0 / 1024.0, device);
            err = cudaMalloc(ptr, size);
            if (err == cudaSuccess) {
                break;
            }
            (void) cudaGetLastError();
        }
        if (err != cudaSuccess) {
            (void) cudaGetLastError();
            if (moe_cache_release_arena()) {
                GGML_LOG_WARN("%s: %.2f MiB allocation still failed on device %d; released the MoE arena and retrying\n",
                              __func__, size / 1024.0 / 1024.0, device);
                err = cudaMalloc(ptr, size);
            }
        }
        // OPEN 2 debug validator (no-op unless MOE_EXPERT_CACHE_VALIDATE is set): catch a cache the
        // stand-down left inconsistent before the next graph reads it.
        moe_cache_validate("device-malloc-yield");
    }
    if (err == cudaSuccess) {
        ggml_cuda_outside_alloc_note(device, size);
        // Low-water mark of the free VRAM, sampled on every config (until r167 only a slab-armed MoE run
        // tracked it, and only GGML_CUDA_SLAB_STATS printed it), plus the one-shot GGML_CUDA_LOW_FREE_MIB
        // warning from the same reading.  The 16 MiB gate is deliberate: `cudaMemGetInfo` is a driver call,
        // and the workspace pool's misses (the only frequent caller) are mostly small transients -- a card
        // reaches the limit through its large allocations, which are what this samples.  Measured cost of
        // the gate: a 15.8k-token prefill on the 27B GSQ arm is neutral within run-to-run noise versus the
        // pre-r167 binary (archive/work/vram-limit-103/).
        if (size >= ((size_t) 16 << 20)) {
            size_t fb = 0, tb = 0;
            if (cudaMemGetInfo(&fb, &tb) == cudaSuccess) {
                ggml_cuda_free_min_sample(device, fb);
                ggml_cuda_low_free_warn(device, size, fb, tb);
            } else {
                (void) cudaGetLastError();
            }
        }
    } else {
        ggml_cuda_outside_fail_note(device, size);
        ggml_cuda_oom_warn(device, size);
    }
    return err;
}

#if defined(GGML_USE_HIP)
static int ggml_cuda_parse_id(char devName[]) {
    // A list of possible Target IDs can be found under the rocclr/clr repo in device.cpp
    // these values are not stable so this is susceptible to breakage
    // https://github.com/ROCm/clr/blob/amd-staging/rocclr/device/device.cpp
    int archMajor = 0x0;
    int archMinor = 0x0;
    int archNum = GGML_CUDA_CC_OFFSET_AMD;
    int archLen = strlen(devName);
    char archName[archLen + 1];

    // strip leading 'gfx' while copying into our buffer
    if (archLen > 3) {
        strcpy(archName, &devName[3]);
        archLen -= 3;
    }

    // trim trailing :xnack- or :sramecc- statuses
    archLen = strcspn(archName, ":");
    archName[archLen] = '\0';

    // tease out the version information
    if (archLen > 8) {
        // versions labeled generic use '-' as delimiter
        // strip the trailing "-generic" then iterate through what remains
        if ((strstr(archName, "-generic"))) {
            archName[archLen - 8] = '\0';
            char * pch;
            if ((pch = strtok(archName, "-"))) {
                archMajor = (int)strtoul(pch, 0, 16);
                if ((pch = strtok(NULL, "-"))) {
                    archMinor = 0x10 * (int)strtoul(pch, 0, 16);
                }
            }
        }
    } else if (archLen >= 3) {
        // last two digits should be the minor * 0x10 + stepping
        archMinor = (int)strtoul(&archName[archLen - 2], 0, 16);
        archName[archLen - 2] = '\0';

        // only the major version remains
        archMajor = (int)strtoul(archName, 0, 16);
    }
    archNum += archMajor * 0x100;
    archNum += archMinor;

    return archNum;
}
#endif // defined(GGML_USE_HIP)

static ggml_cuda_device_info ggml_cuda_init() {
    ggml_cuda_device_info info = {};

    cudaError_t err = cudaGetDeviceCount(&info.physical_device_count);
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: failed to initialize " GGML_CUDA_NAME ": %s\n", __func__, cudaGetErrorString(err));
        return info;
    }

    GGML_ASSERT(info.physical_device_count <= GGML_CUDA_MAX_DEVICES);

    // by default expose exactly the physical devices; GGML_CUDA_DEVICES can request a different
    // number of (virtual) devices to emulate multi-GPU systems on a machine with fewer GPUs
    info.device_count = info.physical_device_count;

    const char * devices_env = getenv("GGML_CUDA_DEVICES");
    if (devices_env != nullptr && info.physical_device_count > 0) {
        const int requested = atoi(devices_env);
        if (requested > 0) {
            info.device_count = requested;
        } else {
            GGML_LOG_WARN("%s: ignoring invalid GGML_CUDA_DEVICES=\"%s\"\n", __func__, devices_env);
        }
    }

    if (info.device_count > GGML_CUDA_MAX_DEVICES) {
        GGML_LOG_WARN("%s: requested %d devices, clamping to GGML_CUDA_MAX_DEVICES=%d\n",
                      __func__, info.device_count, GGML_CUDA_MAX_DEVICES);
        info.device_count = GGML_CUDA_MAX_DEVICES;
    }

    // map each (virtual) device to a backing physical device (round-robin), assign each its index
    // among the (virtual) devices sharing that physical GPU, and store the per-physical share count
    int physical_share_count[GGML_CUDA_MAX_DEVICES] = {};
    GGML_ASSERT(info.device_count == 0 || info.physical_device_count > 0);
    for (int id = 0; id < info.device_count; ++id) {
        info.devices[id].physical_device = id % info.physical_device_count;
        info.devices[id].virtual_index  = physical_share_count[info.devices[id].physical_device]++;
    }

    int64_t total_vram = 0;
    for (int id = 0; id < info.physical_device_count; ++id) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
        total_vram += prop.totalGlobalMem;
    }
    GGML_LOG_INFO("%s: found %d " GGML_CUDA_NAME " devices (Total VRAM: %zu MiB):\n",
                  __func__, info.physical_device_count, (size_t)(total_vram / (1024 * 1024)));
    if (info.device_count != info.physical_device_count) {
        GGML_LOG_INFO("%s: emulating %d virtual device(s) on %d physical device(s) (GGML_CUDA_DEVICES)\n",
                      __func__, info.device_count, info.physical_device_count);
    }
    total_vram = 0;

    std::vector<std::pair<int, std::string>> turing_devices_without_mma;
    for (int id = 0; id < info.device_count; ++id) {
        const int physical_id = info.devices[id].physical_device;

        int device_vmm = 0;

        // r42 (OPEN 2): detect VMM support whenever the platform exposes the entry points, not only when
        // the workspace pool opts into VMM (GGML_USE_VMM / GGML_HIP_NO_VMM=OFF).  The compute-buffer VMM
        // pool is independent of that switch and must be able to see the capability on a default HIP
        // build.  `new_pool_for_device` keeps its own GGML_USE_VMM guard, so the workspace pool is
        // unaffected by `devices[].vmm` becoming true here.
#if defined(GGML_USE_VMM) || defined(GGML_USE_HIP)
        CUdevice device;
        CU_CHECK(cuDeviceGet(&device, physical_id));
        CU_CHECK(cuDeviceGetAttribute(&device_vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device));

        if (device_vmm) {
            CUmemAllocationProp alloc_prop = {};
            alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            alloc_prop.location.id = physical_id;
            CU_CHECK(cuMemGetAllocationGranularity(&info.devices[id].vmm_granularity, &alloc_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        }
#endif // defined(GGML_USE_VMM) || defined(GGML_USE_HIP)
        info.devices[id].vmm = !!device_vmm;

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, physical_id));

        // a virtual device owns only a share of its physical GPU's memory; report that share so the
        // logged per-device VRAM sums to the physical total above.
        GGML_ASSERT(physical_share_count[physical_id] > 0);
        info.devices[id].physical_share_count = physical_share_count[physical_id];
        const size_t device_vram = prop.totalGlobalMem / info.devices[id].physical_share_count;
        const size_t device_vram_mib = device_vram / (1024 * 1024);

        info.default_tensor_split[id] = total_vram;
        total_vram += device_vram;
        // Fork divergence from PR #24233 (restored prop.integrated on HIP builds): the CUDA
        // host-buffer path (zero-copy UMA weights) it enables on APUs corrupts full-model
        // results under async execution on this box (PPL 5.9243 -> 8.51+ without
        // HIP_LAUNCH_BLOCKING).  Upstream reverted #24233 in #28604 (2026-09-08), making
        // forced-integrated-false the upstream default again.
        //
        // Re-tested 2026-09-23 (closing-the-gap host-buffer investigation): the corruption was
        // the scheduler reading a host-resident graph input in place; see the host-input guard
        // in ggml_backend_sched_buffer_supported().  With that guard the real flag is stable on
        // this box, and it is what lets an APU gather the input embeddings on the GPU (no CPU
        // backend dispatch) while the embedding weights stay in zero-copy host memory.
        // GGML_FORCE_NO_INTEGRATED=1 restores the previous default for A/B and bisection.
#if defined(GGML_USE_HIP)
        static const bool force_no_integrated = getenv("GGML_FORCE_NO_INTEGRATED") != nullptr;
        info.devices[id].integrated = force_no_integrated ? false : prop.integrated;
#else
        info.devices[id].integrated = false; // Temporarily disabled due to issues with corrupted output (e.g. #15034)
#endif
        info.devices[id].nsm        = prop.multiProcessorCount;
        info.devices[id].smpb       = prop.sharedMemPerBlock;
        info.devices[id].warp_size  = prop.warpSize;

        int supports_coop_launch = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&supports_coop_launch, cudaDevAttrCooperativeLaunch, physical_id));
        info.devices[id].supports_cooperative_launch = !!supports_coop_launch;

#if defined(GGML_USE_HIP)
        info.devices[id].smpbo = prop.sharedMemPerBlock;

        info.devices[id].cc = ggml_cuda_parse_id(prop.gcnArchName);
        if ((info.devices[id].cc & 0xff00) == 0x0) {
            GGML_LOG_WARN("invalid architecture ID received for device %d %s: %s  cc %d.%d\n",
                            id, prop.name, prop.gcnArchName, prop.major, prop.minor);

            // Fallback to prop.major and prop.minor
            if (prop.major > 0) {
                info.devices[id].cc = GGML_CUDA_CC_OFFSET_AMD + prop.major * 0x100;
                info.devices[id].cc += prop.minor * 0x10;
            }
        }
        GGML_LOG_INFO("  Device %d: %s, %s (0x%x), VMM: %s, Wave Size: %d, VRAM: %zu MiB\n",
                      id, prop.name, prop.gcnArchName, info.devices[id].cc & 0xffff,
                      device_vmm ? "yes" : "no", prop.warpSize,
                      device_vram_mib);
#elif defined(GGML_USE_MUSA)
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = GGML_CUDA_CC_OFFSET_MTHREADS + prop.major * 0x100;
        info.devices[id].cc += prop.minor * 0x10;
        GGML_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s, VRAM: %zu MiB\n",
                      id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no",
                      device_vram_mib);
#else
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = 100*prop.major + 10*prop.minor;
        GGML_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s, VRAM: %zu MiB\n",
                      id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no",
                      device_vram_mib);
        std::string device_name(prop.name);
        if (device_name == "NVIDIA GeForce MX450") {
            turing_devices_without_mma.push_back({ id, device_name });
        } else if (device_name == "NVIDIA GeForce MX550") {
            turing_devices_without_mma.push_back({ id, device_name });
        } else if (device_name.substr(0, 21) == "NVIDIA GeForce GTX 16") {
            turing_devices_without_mma.push_back({ id, device_name });
        }

        // Temporary performance fix:
        // Setting device scheduling strategy for iGPUs with cc121 to "spinning" to avoid delays in cuda synchronize calls.
        // TODO: Check for future drivers the default scheduling strategy and
        // remove this call again when cudaDeviceScheduleSpin is default.
        if (prop.major == 12 && prop.minor == 1) {
            CUDA_CHECK(cudaSetDevice(physical_id));
            CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
        }

#endif  // defined(GGML_USE_HIP)
    }

    if (ggml_cuda_highest_compiled_arch(GGML_CUDA_CC_TURING) >= GGML_CUDA_CC_TURING && !turing_devices_without_mma.empty()) {
        GGML_LOG_INFO("The following devices will have suboptimal performance due to a lack of tensor cores:\n");
        for (size_t device_pos = 0; device_pos < turing_devices_without_mma.size(); device_pos++) {
            GGML_LOG_INFO(
                "  Device %d: %s\n", turing_devices_without_mma[device_pos].first, turing_devices_without_mma[device_pos].second.c_str());
        }
        GGML_LOG_INFO(
            "Consider compiling with CMAKE_CUDA_ARCHITECTURES=61-virtual;80-virtual and DGGML_CUDA_FORCE_MMQ to force the use of the Pascal code for Turing.\n");
    }

    for (int id = 0; id < info.device_count; ++id) {
        info.default_tensor_split[id] /= total_vram;
    }

    // configure logging to stdout
    // CUBLAS_CHECK(cublasLoggerConfigure(1, 1, 0, nullptr));

    if (ggml_cuda_p2p_enabled()) {
        for (int id = 0; id < info.physical_device_count; ++id) {
            CUDA_CHECK(cudaSetDevice(id));
            for (int id_other = 0; id_other < info.physical_device_count; ++id_other) {
                if (id == id_other) {
                    continue;
                }
                int can_access_peer;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, id, id_other));
                if (can_access_peer) {
                    CUDA_CHECK(cudaDeviceEnablePeerAccess(id_other, 0));
                }
            }
        }
    }

    return info;
}

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = ggml_cuda_init();
    return info;
}



// #define DEBUG_CUDA_MALLOC

// buffer pool for cuda (legacy)
// TEMP INSTRUMENT (exp7): pool alloc/free cost (cudaMalloc/cudaFree on this path are synchronizing).
static int64_t g_pool_us = 0, g_pool_n = 0, g_pool_free_us = 0, g_pool_free_n = 0, g_pool_oom_n = 0;

static std::atomic<uint64_t> g_cuda_graph_mem_gen[GGML_CUDA_MAX_DEVICES];

uint64_t ggml_cuda_graph_mem_gen(int device) {
    return g_cuda_graph_mem_gen[device].load(std::memory_order_relaxed);
}

void ggml_cuda_graph_mem_freed(int device) {
    g_cuda_graph_mem_gen[device]++;
}

// Free-VRAM floor for the workspace pool's first attempt (the current device).  With host-expert tables in the
// MoE expert cache, letting the pool take a device down to ~40 MiB free left the slab extension's arena tables
// overwritten with live data (pinned -sm layer, the greedy decode right after a 61k prefill).  We have not found
// what writes there; keeping 512 MiB free avoids the state.  GGML_CUDA_POOL_MIN_FREE_MIB sets the floor (0 = off).
static bool ggml_cuda_pool_floor_refuses(size_t size) {
    // read the value once: a cached getenv() pointer is not safe on Windows (the CRT buffer is reused)
    static const long long floor_env_mib = [] {
        const char * env = getenv("GGML_CUDA_POOL_MIN_FREE_MIB");
        return env != nullptr ? atoll(env) : -1;
    }();
    const size_t floor = floor_env_mib >= 0 ? (size_t) floor_env_mib << 20 : (moe_cache_floor_active() ? (size_t) 512 << 20 : 0);
    if (floor == 0 || size < ((size_t) 1 << 20)) {
        return false;
    }
    size_t f = 0;
    size_t t = 0;
    if (cudaMemGetInfo(&f, &t) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    return f < size + floor;
}

static bool ggml_cuda_graph_capture_active();

// wip/slab-transient-absorption: physical retained by the workspace pools, per device.  `pool_leg` keeps
// every freed block cached (up to 256) and the VMM pool never unmaps, so this is the pool high-water --
// one of the outside-slab consumers the slab headroom pays for.  Diagnostic only (GGML_CUDA_SLAB_STATS).
static std::atomic<size_t> g_pool_mapped[GGML_CUDA_MAX_DEVICES];

size_t ggml_cuda_pool_mapped_bytes(int device) {
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) {
        return 0;
    }
    return g_pool_mapped[device].load(std::memory_order_relaxed);
}

// wip/slab-transient-absorption (step 2b): pool blocks currently served from the slab's wide band, per
// device.  `blocks` is the leak tripwire (see `ggml_cuda_slab_ring_set`); `bytes` is reported by SLAB_STATS.
// Neither is counted in `g_pool_mapped`: band bytes are the slab's, not the pool's.
static std::atomic<size_t> g_pool_wide_bytes[GGML_CUDA_MAX_DEVICES];
static std::atomic<size_t> g_pool_wide_blocks[GGML_CUDA_MAX_DEVICES];
// bytes the armed band REFUSED (diagnostic: those stay outside the slab as pool high-water).
static std::atomic<size_t> g_pool_wide_refused[GGML_CUDA_MAX_DEVICES];

size_t ggml_cuda_pool_wide_bytes(int device) {
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) {
        return 0;
    }
    return g_pool_wide_bytes[device].load(std::memory_order_relaxed);
}

size_t ggml_cuda_pool_wide_blocks(int device) {
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) {
        return 0;
    }
    return g_pool_wide_blocks[device].load(std::memory_order_relaxed);
}

size_t ggml_cuda_pool_wide_refused(int device) {
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) {
        return 0;
    }
    return g_pool_wide_refused[device].load(std::memory_order_relaxed);
}

void ggml_cuda_pool_wide_refused_reset(int device) {
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) {
        return;
    }
    g_pool_wide_refused[device].store(0, std::memory_order_relaxed);
}

struct ggml_cuda_pool_leg : public ggml_cuda_pool {
    static const int MAX_BUFFERS = 256;
    // wip/slab-transient-absorption (step 2b): concurrent wide-band blocks one pool can hold.  The band
    // serves the LARGE prefill temporaries (>= GGML_CUDA_POOL_WIDE_MIN_MIB), a handful at a time; when the
    // table is full the block is handed straight back to the band instead of being handed out untracked.
    static const int MAX_WIDE = 64;

    ggml_backend_cuda_context & ctx;
    int device;
    struct ggml_cuda_buffer {
        void * ptr = nullptr;
        size_t size = 0;
    };

    ggml_cuda_buffer buffer_pool[MAX_BUFFERS] = {};
    size_t pool_size = 0;
    // GGML_CUDA_POOL_CACHE_MB: cap on the bytes of FREED buffers the pool keeps cached (unset/0 = unlimited).
    // Temporaries that grow with the KV length (long prefills) otherwise leave every smaller predecessor cached,
    // and that idle VRAM starves allocations made outside the pool (graph/compute buffers, FA staging).
    size_t cached_size = 0;
    size_t cache_cap   = 0;

    // wip/slab-transient-absorption (step 2b): the WIDE-PHASE redirect.  While the slab is live and its wide
    // band is armed (a prefill), a large allocation is bump-allocated from the band instead of `cudaMalloc`-ed
    // outside the slab, and returned to the band on free (never cached -- see `free`).  `set_wide_routing`
    // pins it off for a retained allocation (the MMB cache/slots, the allreduce scratch).
    bool wide_routing = true;
    struct ggml_cuda_wide_block {
        void * ptr  = nullptr;
        size_t size = 0;   // the REQUESTED size, i.e. what `free` is handed back
        size_t gen  = 0;   // the band generation the block was served in (stale detection)
    };
    ggml_cuda_wide_block wide_blocks[MAX_WIDE];
    int n_wide = 0;

    ggml_cuda_pool_leg(ggml_backend_cuda_context & ctx, int device) :
        ctx(ctx),
        device(device) {
        const char * cap = getenv("GGML_CUDA_POOL_CACHE_MB");
        const long long mb = cap ? atoll(cap) : 0;
        cache_cap = mb > 0 ? (size_t) mb << 20 : 0;
    }

    bool set_wide_routing(bool on) override {
        const bool prev = wide_routing;
        wide_routing = on;
        return prev;
    }

    static bool wide_enabled() {
        // Kill switch / A-B knob (default ON; set to 0 to keep every pool allocation outside the slab).
        static const bool on = [] {
            const char * e = getenv("GGML_CUDA_POOL_WIDE");
            return e == nullptr || atoi(e) != 0;
        }();
        return on;
    }

    static bool wide_debug() {
        static const bool on = getenv("GGML_CUDA_POOL_WIDE_DEBUG") != nullptr;
        return on;
    }

    static size_t wide_min_bytes() {
        // The band rounds a block up to the slab unit (the VMM granularity, 2 MiB here), so a tiny
        // allocation would cost a whole unit of the band.  Only sizes at or above one unit are worth
        // redirecting; the many small prefill scratch blocks stay in the pool.
        static const size_t v = [] {
            const char * e = getenv("GGML_CUDA_POOL_WIDE_MIN_MIB");
            const long long mb = e ? atoll(e) : 1;
            return (size_t) (mb > 0 ? mb : 1) << 20;
        }();
        return v;
    }

    void wide_account(int delta_blocks, long long delta_bytes) {
        if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) {
            return;
        }
        if (delta_blocks > 0) {
            g_pool_wide_blocks[device].fetch_add((size_t) delta_blocks, std::memory_order_relaxed);
        } else if (delta_blocks < 0) {
            g_pool_wide_blocks[device].fetch_sub((size_t) -delta_blocks, std::memory_order_relaxed);
        }
        if (delta_bytes != 0) {
            g_pool_wide_bytes[device].fetch_add((size_t) delta_bytes, std::memory_order_relaxed);
        }
    }

    ~ggml_cuda_pool_leg() {
        clear_pool();
        // A live wide-band block here is a pool allocation outliving its pool: nothing to free (the band
        // owns the bytes), but it must be visible -- the `POOL_WIDE_LEAK` tripwire at the wide drop is the
        // primary diagnostic and this is its late counterpart.
        if (n_wide != 0) {
            GGML_LOG_WARN(GGML_CUDA_NAME " pool[%d]: %d wide-band block(s) (%.2f MiB) outlived the pool\n",
                          device, n_wide, (double) ggml_cuda_pool_wide_bytes(device) / 1048576.0);
            wide_account(-n_wide, 0);
            n_wide = 0;
        }
        GGML_ASSERT(pool_size == 0);
    }

    void clear_pool() {
        ggml_cuda_set_device(device);
        ggml_cuda_graph_mem_freed(device);
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer & b = buffer_pool[i];
            if (b.ptr != nullptr) {
                CUDA_CHECK(cudaFree(b.ptr));
                pool_size -= b.size;
                if (device >= 0 && device < GGML_CUDA_MAX_DEVICES) {
                    g_pool_mapped[device].fetch_sub(b.size, std::memory_order_relaxed);
                }
                b.ptr  = nullptr;
                b.size = 0;
            }
        }
        cached_size = 0;
        // wip/slab-transient-absorption (step 2b): `clear_pool` drops CACHED (freed) buffers, and the OOM
        // retry calls it mid-graph -- `wide_blocks` therefore stays as it is.  Those entries describe LIVE
        // blocks (never cached: `free` returns a band block to the band); forgetting one would send its
        // later free down the cudaMalloc path.  A leftover at destruction is reported there.
    }

    // captured CUDA graphs can still use freed buffers, so they must be captured again
    void invalidate_cuda_graphs() {
#ifdef USE_CUDA_GRAPH
        int n_captured = 0;
        for (const auto & [key, graph] : ctx.cuda_graphs) {
            n_captured += graph->instances.size();
        }
        if (n_captured > 0) {
            GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[%d]: memory freed, %d captured CUDA graphs must be captured again\n", device, n_captured);
        }
        ctx.cuda_graphs.clear();
#endif // USE_CUDA_GRAPH
    }

    void * alloc(size_t size, size_t * actual_size) override {
#ifdef DEBUG_CUDA_MALLOC
        int nnz = 0;
        size_t max_size = 0;
#endif
        // wip/slab-transient-absorption (step 2b): serve a LARGE allocation from the slab's wide band while
        // it is armed (a prefill).  The band sits in the work region the prefill already claims, so these
        // bytes leave the headroom budget and go back to the arena at the prefill -> decode drop instead of
        // becoming a permanent pool high-water mark.  Deliberately BEFORE the cached-block scan: reusing a
        // cached block during a wide phase would keep the pool's outside high-water alive for no gain.
        //
        // Guards, in order: the pin (`set_wide_routing`), the size floor (the band rounds up to the slab
        // unit), the kill switch, an active graph capture (a captured graph's pointers must stay valid
        // forever, which a reclaimed band cannot promise) and the band's own availability (slab live AND
        // wide view armed).
        const bool wide_large = size >= wide_min_bytes();
        const bool wide_ok    = wide_routing && wide_large && wide_enabled() &&
                                !ggml_cuda_graph_capture_active();
        if (wide_ok && ggml_cuda_slab_wide_available(device)) {
            size_t gen = 0;
            void * p = ggml_cuda_slab_wide_aux_alloc(device, size, &gen);
            // Diagnostic: a LARGE allocation the armed band refused stays outside the slab and becomes
            // pool high-water, which is what this redirect exists to prevent -- count it (SLAB_STATS).
            if (p == nullptr && device >= 0 && device < GGML_CUDA_MAX_DEVICES) {
                g_pool_wide_refused[device].fetch_add(size, std::memory_order_relaxed);
            }
            if (p != nullptr) {
                if (n_wide < MAX_WIDE) {
                    wide_blocks[n_wide].ptr  = p;
                    wide_blocks[n_wide].size = size;
                    wide_blocks[n_wide].gen  = gen;
                    n_wide++;
                    wide_account(+1, (long long) size);
                    if (wide_debug()) {
                        GGML_LOG_WARN("%s pool[%d]: wide-band alloc %.3f MiB -> %p (live %d)\n",
                                      __func__, device, (double) size / 1048576.0, p, n_wide);
                    }
                    // The REQUESTED size, not the 5 % + 256 B look-ahead: `free` pops exactly this many
                    // band bytes (`ggml_cuda_slab_wide_aux_free` rounds it up the same way the alloc did).
                    *actual_size = size;
                    return p;
                }
                // No bookkeeping slot: hand the block straight back rather than return a pointer this pool
                // could later `cudaFree`.
                ggml_cuda_slab_wide_aux_free(device, p, size);
            }
        }
        if (wide_debug() && size >= ((size_t) 1 << 20)) {
            GGML_LOG_WARN("%s pool[%d]: OUTSIDE %.3f MiB (routing=%d enabled=%d capture=%d armed=%d "
                          "pinned=%d small=%d)\n", __func__, device, (double) size / 1048576.0,
                          (int) wide_routing, (int) wide_enabled(), (int) ggml_cuda_graph_capture_active(),
                          (int) ggml_cuda_slab_wide_available(device), (int) !wide_routing,
                          (int) !wide_large);
        }
        size_t best_diff = 1ull << 36;
        int ibest = -1;
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr != nullptr) {
#ifdef DEBUG_CUDA_MALLOC
                ++nnz;
                if (b.size > max_size) max_size = b.size;
#endif
                if (b.size >= size) {
                    size_t diff = b.size - size;
                    if (diff < best_diff) {
                        best_diff = diff;
                        ibest = i;
                        if (!best_diff) {
                            void * ptr = b.ptr;
                            *actual_size = b.size;
                            cached_size -= b.size;
                            b.ptr = nullptr;
                            b.size = 0;
                            return ptr;
                        }
                    }
                }
            }
        }
        if (ibest >= 0) {
            ggml_cuda_buffer& b = buffer_pool[ibest];
            void * ptr = b.ptr;
            *actual_size = b.size;
            cached_size -= b.size;
            b.ptr = nullptr;
            b.size = 0;
            return ptr;
        }
        void * ptr;
        size_t look_ahead_size = (size_t) (1.05 * size);
        look_ahead_size = 256 * ((look_ahead_size + 255)/256);
        ggml_cuda_set_device(device);
        // never flush while a CUDA graph is being captured: the sync is illegal and the capture may reference the
        // cached buffers; the next allocation outside capture flushes instead
        if (cache_cap && cached_size > cache_cap && !ggml_cuda_graph_capture_active()) {
            GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[%d]: %.2f MiB cached exceeds GGML_CUDA_POOL_CACHE_MB, flushing\n",
                           device, cached_size/1024.0/1024.0);
            CUDA_CHECK(cudaDeviceSynchronize());
            clear_pool();
        }
        static const bool gc_dbg = getenv("GGML_CUDA_GCDBG") != nullptr;
        const int64_t t_pl = gc_dbg ? ggml_time_us() : 0;
        // With the MoE expert cache holding host-expert tables, the first attempt keeps a free-VRAM floor and the
        // flush + retry below runs instead (see ggml_cuda_pool_floor_refuses).  The retry has no floor, so this
        // never turns into an abort.
        // A graph capture must never reach the driver's synchronization/flush path: cudaDeviceSynchronize
        // (and the cache flush's cudaFree) is illegal inside a capture.  The graph-capture guard in
        // `ggml_backend_cuda_graph_compute` keeps captures off the armed wide band and gives them a
        // disarmed warmup, so a capture's temporaries come from the pool; this is the belt-and-braces
        // for the case it does not (and it makes the failure report the real OOM, not a capture error).
        const bool capturing = ggml_cuda_graph_capture_active();
        cudaError_t err = cudaErrorMemoryAllocation;
        if (capturing) {
            // The free-VRAM floor is a soft check that is normally resolved by the flush + retry below;
            // that path is unavailable here, so go straight to the allocation without the floor.
            err = ggml_cuda_device_malloc(&ptr, look_ahead_size, device);
        } else if (!ggml_cuda_pool_floor_refuses(look_ahead_size)) {
            err = ggml_cuda_device_malloc(&ptr, look_ahead_size, device);
        }
        if (t_pl) { g_pool_us += ggml_time_us() - t_pl; g_pool_n++; }
        if (err == cudaErrorMemoryAllocation && capturing) {
            g_pool_oom_n++;
            (void)cudaGetLastError();
            GGML_LOG_WARN(GGML_CUDA_NAME " pool[%d]: %.2f MiB allocation failed during graph capture; a "
                          "captured graph's temporaries must come from the pool (no synchronize/flush is "
                          "possible inside a capture)\n", device, look_ahead_size/1024.0/1024.0);
        } else if (err == cudaErrorMemoryAllocation) {
            g_pool_oom_n++;
            (void)cudaGetLastError();
            const size_t cached_bytes = pool_size;
            GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[%d]: alloc of %.2f MiB failed, flushing %.2f MiB of cached buffers and retrying\n",
                           device, look_ahead_size/1024.0/1024.0, cached_bytes/1024.0/1024.0);
            CUDA_CHECK(cudaDeviceSynchronize());
            clear_pool();
            invalidate_cuda_graphs();
            // The MoE-arena yield lives inside `ggml_cuda_device_malloc`, so this retry inherits it.
            err = ggml_cuda_device_malloc(&ptr, look_ahead_size, device);
            if (err == cudaSuccess) {
                GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[%d]: retry succeeded\n", device);
            }
        }
        CUDA_CHECK(err);
        *actual_size = look_ahead_size;
        pool_size += look_ahead_size;
        if (device >= 0 && device < GGML_CUDA_MAX_DEVICES) {
            g_pool_mapped[device].fetch_add(look_ahead_size, std::memory_order_relaxed);
        }
#ifdef DEBUG_CUDA_MALLOC
        GGML_LOG_INFO("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, device, nnz,
                           (uint32_t)(max_size / 1024 / 1024), (uint32_t)(pool_size / 1024 / 1024), (uint32_t)(size / 1024 / 1024));
#endif
        return ptr;
    }

    void free(void * ptr, size_t size) override {
        // wip/slab-transient-absorption (step 2b): a wide-band block is not pool memory.  It must never
        // enter the free list (a cached band pointer would be reused after the drop) and must not touch
        // `pool_size`.  The table is tiny (one entry per concurrent large prefill temporary).
        for (int i = 0; i < n_wide; ++i) {
            if (wide_blocks[i].ptr != ptr) {
                continue;
            }
            // The BOOKKEPT size is authoritative: it is what `alloc` handed the band, so the pop is exact
            // even if a caller passes something other than the `actual_size` the pool returned.
            const size_t size_b = wide_blocks[i].size;
            const size_t gen    = wide_blocks[i].gen;
            n_wide--;
            wide_blocks[i] = wide_blocks[n_wide];
            wide_blocks[n_wide] = {};
            wide_account(-1, -(int64_t) size_b);
            if (wide_debug()) {
                GGML_LOG_WARN("%s pool[%d]: wide-band free  %.3f MiB <- %p (live %d, gen %zu/%zu)\n",
                              __func__, device, (double) size_b / 1048576.0, ptr, n_wide, gen,
                              ggml_cuda_slab_wide_aux_gen(device));
            }
            if (gen != ggml_cuda_slab_wide_aux_gen(device)) {
                // The band was released (prefill -> decode drop) or moved under a live block: a retained
                // allocation was served from it.  Give nothing back (the region is gone) and say so once;
                // the `POOL_WIDE_LEAK` tripwire at the drop is the primary diagnostic.
                static bool warned = false;
                if (!warned) {
                    warned = true;
                    GGML_LOG_WARN(GGML_CUDA_NAME " pool[%d]: freeing a wide-band block (%.2f MiB) from a dead "
                                  "generation: a pool allocation outlived the prefill band\n",
                                  device, (double) size / 1048576.0);
                }
                return;
            }
            ggml_cuda_slab_wide_aux_free(device, ptr, size_b);
            return;
        }
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr = ptr;
                b.size = size;
                cached_size += size;
                return;
            }
        }
        GGML_LOG_DEBUG(GGML_CUDA_NAME " buffer pool full, increase MAX_CUDA_BUFFERS\n");
        ggml_cuda_set_device(device);
        static const bool gc_dbg = getenv("GGML_CUDA_GCDBG") != nullptr;
        const int64_t t_pf = gc_dbg ? ggml_time_us() : 0;
        ggml_cuda_graph_mem_freed(device);
        CUDA_CHECK(cudaFree(ptr));
        if (t_pf) { g_pool_free_us += ggml_time_us() - t_pf; g_pool_free_n++; }
        pool_size -= size;
        invalidate_cuda_graphs();
        if (device >= 0 && device < GGML_CUDA_MAX_DEVICES) {
            g_pool_mapped[device].fetch_sub(size, std::memory_order_relaxed);
        }
    }
};

// pool with virtual memory
#if defined(GGML_USE_VMM)
struct ggml_cuda_pool_vmm : public ggml_cuda_pool {
    static const size_t CUDA_POOL_VMM_MAX_SIZE = 1ull << 35; // 32 GB

    int device;
    int physical_device;
    CUdeviceptr pool_addr = 0;
    size_t pool_used = 0;
    size_t pool_size = 0;
    size_t granularity;
#if defined(GGML_USE_HIP)
    std::vector<std::pair<CUdeviceptr, size_t>> mappings;
#endif

    explicit ggml_cuda_pool_vmm(int device) :
        device(device),
        physical_device(ggml_cuda_get_physical_device(device)),
        granularity(ggml_cuda_info().devices[device].vmm_granularity) {
    }

    ~ggml_cuda_pool_vmm() {
        if (pool_addr != 0) {
            if (device >= 0 && device < GGML_CUDA_MAX_DEVICES) {
                g_pool_mapped[device].fetch_sub(pool_size, std::memory_order_relaxed);
            }
#if defined(GGML_USE_HIP)
            // Workaround for https://github.com/ROCm/ROCR-Runtime/issues/285
            for (std::pair<CUdeviceptr, size_t> & mapping : mappings) {
                CU_CHECK(cuMemUnmap(mapping.first, mapping.second));
            }
#else
            CU_CHECK(cuMemUnmap(pool_addr, pool_size));
#endif
            CU_CHECK(cuMemAddressFree(pool_addr, CUDA_POOL_VMM_MAX_SIZE));
        }
    }

    void * alloc(size_t size, size_t * actual_size) override {
        // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
        const size_t alignment = 128;
        size = alignment * ((size + alignment - 1) / alignment);

        size_t avail = pool_size - pool_used;

        if (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_size = size - avail;
            reserve_size = granularity * ((reserve_size + granularity - 1) / granularity);

            GGML_ASSERT(pool_size + reserve_size <= CUDA_POOL_VMM_MAX_SIZE);

            // allocate more physical memory
            CUmemAllocationProp prop = {};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = physical_device;
            CUmemGenericAllocationHandle handle;
            CU_CHECK(cuMemCreate(&handle, reserve_size, &prop, 0));

            // reserve virtual address space (if not already reserved)
            if (pool_addr == 0) {
                CU_CHECK(cuMemAddressReserve(&pool_addr, CUDA_POOL_VMM_MAX_SIZE, 0, 0, 0));
            }

            // map at the end of the pool
            CUdeviceptr start_ptr = (CUdeviceptr)((char *)(pool_addr) + pool_size);
            CU_CHECK(cuMemMap(start_ptr, reserve_size, 0, handle, 0));
#if defined(GGML_USE_HIP)
            mappings.push_back({start_ptr, reserve_size});
#endif

            // the memory allocation handle is no longer needed after mapping
            CU_CHECK(cuMemRelease(handle));

            // VMM Bug fix for P2P access if GGML_CUDA_P2P is set, or if NCCL build
            bool use_peer_access = ggml_cuda_p2p_enabled();
#if defined(GGML_USE_NCCL)
            use_peer_access = true;
#endif // defined(GGML_USE_NCCL)

            if (use_peer_access) {
                // NCCL implicitly enables peer access (cudaDeviceEnablePeerAccess), and
                // GGML_CUDA_P2P enables it explicitly. Unlike cudaMalloc buffers, VMM
                // allocations do not become peer-accessible from that alone, so access
                // must be granted explicitly here. With virtual devices, grant access
                // on the backing *physical* devices (deduplicated, since several
                // virtual devices can map to the same physical GPU).
                std::vector<CUmemAccessDesc> access_descs;
                bool physical_seen[GGML_CUDA_MAX_DEVICES] = {};
                const int device_count = ggml_cuda_info().device_count;
                for (int id = 0; id < device_count; ++id) {
                    const int id_physical = ggml_cuda_get_physical_device(id);
                    if (id_physical != physical_device) {
                        int can_access_peer = 0;
                        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, id_physical, physical_device));
                        if (!can_access_peer) {
                            continue;
                        }
                    }
                    if (physical_seen[id_physical]) {
                        continue;
                    }
                    physical_seen[id_physical] = true;
                    CUmemAccessDesc access = {};
                    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                    access.location.id = id_physical;
                    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                    access_descs.push_back(access);
                }
                CU_CHECK(cuMemSetAccess(start_ptr, reserve_size, access_descs.data(), access_descs.size()));
            } else {
                // set access for non P2P
                CUmemAccessDesc access = {};
                access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                access.location.id = physical_device;
                access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                CU_CHECK(cuMemSetAccess(start_ptr, reserve_size, &access, 1));
            }

            // add to the pool
            pool_size += reserve_size;
            if (device >= 0 && device < GGML_CUDA_MAX_DEVICES) {
                g_pool_mapped[device].fetch_add(reserve_size, std::memory_order_relaxed);
            }

            //printf("cuda pool[%d]: size increased to %llu MB (reserved %llu MB)\n",
            //       device, (unsigned long long) (pool_size/1024/1024),
            //       (unsigned long long) (reserve_size/1024/1024));
        }

        GGML_ASSERT(pool_addr != 0);

        void * ptr = (void *) ((CUdeviceptr)((char *)(pool_addr) + pool_used));
        *actual_size = size;
        pool_used += size;

#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: allocated %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        return ptr;
    }

    void free(void * ptr, size_t size) override {
#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: freed %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        pool_used -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == (void *) ((char *)(pool_addr) + pool_used));
    }
};
#endif // defined(GGML_USE_VMM)

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(ggml_backend_cuda_context & ctx,
                                                                               int                         device,
                                                                               [[maybe_unused]] int        stream_no) {
#if defined(GGML_USE_VMM)
    if (ggml_cuda_info().devices[device].vmm) {
        return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_vmm(device));
    }
#endif // defined(GGML_USE_VMM)
    return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_leg(ctx, device));
}

// destroying a cuBLAS handle while a graph is being captured in a different thread can result in a CUDA error
// this lock is used to ensure that no cuBLAS handle is destroyed while a graph is being captured

static std::mutex ggml_cuda_lock;
static std::condition_variable ggml_cuda_lock_cv;
static std::atomic<int> ggml_cuda_lock_counter;

static bool ggml_cuda_graph_capture_active() {
    return ggml_cuda_lock_counter.load(std::memory_order_relaxed) > 0;
}

static void ggml_cuda_async_input_flush_done(int device, cudaStream_t stream);   // GGML_CUDA_ASYNC_INPUT_LAZY_DONE, below

ggml_backend_cuda_context::~ggml_backend_cuda_context() {
    ggml_cuda_mmb_release_all();
    std::unique_lock<std::mutex> lock(ggml_cuda_lock);
    ggml_cuda_lock_cv.wait(lock, []{ return ggml_cuda_lock_counter.load(std::memory_order_relaxed) == 0; });

    // wip/slab-transient-absorption: steady-state accounting for the headroom campaign (gated on
    // GGML_CUDA_SLAB_STATS).  Printed before the transients are released, so it reports the live layout.
    ggml_cuda_slab_log_stats(device, "context-exit");

    if (q8_1_arena != nullptr) {
        CUDA_CHECK(cudaFree(q8_1_arena));
    }
    for (char * p : q8_1_arena_retired) {
        CUDA_CHECK(cudaFree(p));
    }
    if (hc_q8_cnt != nullptr) {
        CUDA_CHECK(cudaFree(hc_q8_cnt));
    }
    if (moe_glu_q8_cnt != nullptr) {
        CUDA_CHECK(cudaFree(moe_glu_q8_cnt));
    }
    if (moe_red_cnt != nullptr) {
        CUDA_CHECK(cudaFree(moe_red_cnt));
    }
    h2d_stage_free();

    for (int i = 0; i < GGML_CUDA_MAX_STREAMS; ++i) {
        if (fattn_stage[i] != nullptr && !fattn_stage_slab[i]) {
            CUDA_CHECK(cudaFree(fattn_stage[i]));
        }
    }

    if (copy_event != nullptr) {
        CUDA_CHECK(cudaEventDestroy(copy_event));
    }
    // Release the splice-gather ring before its streams go away: the events were recorded on them.
    for (auto & s : splice_pin_ring) {
        if (s.ev != nullptr) { (void) cudaEventSynchronize(s.ev); CUDA_CHECK(cudaEventDestroy(s.ev)); }
        if (s.buf != nullptr) { CUDA_CHECK(hipHostFree(s.buf)); }
    }
    splice_pin_ring.clear();
    // a lazy done event still pending on one of these streams is recorded before the stream goes away
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        for (int j = 0; j < GGML_CUDA_MAX_STREAMS; ++j) {
            if (streams[i][j] != nullptr) {
                ggml_cuda_async_input_flush_done(i, streams[i][j]);
            }
        }
    }
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        for (int j = 0; j < GGML_CUDA_MAX_STREAMS; ++j) {
            if (streams[i][j] != nullptr) {
                CUDA_CHECK(cudaStreamDestroy(streams[i][j]));
            }
            if (cublas_handles[i][j] != nullptr) {
                CUBLAS_CHECK(cublasDestroy(cublas_handles[i][j]));
            }
            if (cublas_workspaces[i][j] != nullptr) {
                CUDA_CHECK(cudaFree(cublas_workspaces[i][j]));
            }
        }
    }
}


// cuda buffer

struct ggml_backend_cuda_buffer_context {
    int device;
    void * dev_ptr = nullptr;
    // OPEN 2: the buffer is a VIEW of the movable-boundary slab's work region (base = slab base).  Freeing
    // it must NOT release anything -- the region belongs to the slab, which lives until the process exits.
    bool slab_view = false;
    // The size this view REQUESTED (what bounds its tensor layout) and its base offset in the slab work
    // region; together they form the boundary contribution `ggml_cuda_slab_work_release` removes.
    size_t slab_need = 0;
    size_t slab_off  = 0;
    std::string name;

    ggml_backend_cuda_buffer_context(int device, void * dev_ptr) :
        device(device), dev_ptr(dev_ptr),
        name(GGML_CUDA_NAME + std::to_string(device)) {
    }

    ~ggml_backend_cuda_buffer_context() {
        if (slab_view) {
            ggml_cuda_slab_work_release(device, slab_off, slab_need);   // the slab owns the region; the slack may go to the arena
            return;
        }
        // A compute buffer allocated from the slab is a VIEW (handled above); everything else here is a
        // plain cudaMalloc pointer.
        CUDA_CHECK(cudaFree(dev_ptr));
    }
};

static void ggml_cuda_async_input_drain(int device);   // GGML_CUDA_ASYNC_INPUT_BATCH, defined with the async input ring below

static void ggml_backend_cuda_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    ggml_cuda_async_input_drain(ctx->device);   // a queued upload into this buffer must not run after it is freed
    delete ctx;
}

static bool ggml_backend_buffer_is_cuda(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_cuda_buffer_free_buffer;
}

static void * ggml_backend_cuda_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    return ctx->dev_ptr;
}

static enum ggml_status ggml_backend_cuda_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    if (tensor->view_src != NULL) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        return GGML_STATUS_SUCCESS;
    }

    if (ggml_is_quantized(tensor->type) && tensor->view_src == nullptr && ggml_backend_buffer_get_usage(buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        // initialize padding to 0 to avoid possible NaN values
        const size_t original_size = ggml_nbytes(tensor);
        const size_t padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size) {
            ggml_cuda_set_device(ctx->device);
            CUDA_CHECK(cudaMemsetAsync((char *)tensor->data + original_size, 0, padded_size - original_size, cudaStreamPerThread));
            CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        }
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cuda_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    ggml_cuda_async_input_drain(ctx->device);
    CUDA_CHECK(cudaMemsetAsync((char *) tensor->data + offset, value, size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

// GGML_CUDA_ASYNC_INPUT=1: graph inputs (small host -> device writes into a compute buffer) are staged through a
// per-device pinned ring and copied asynchronously on cudaStreamPerThread (the stream the synchronous path uses, so a
// later get_tensor stays ordered after it); an event is recorded per copy and the next graph makes its compute stream
// wait for it (ggml_cuda_async_input_wait), so the host never blocks per copy.  The ring bytes are reused only after
// the streams that read them have drained (once per lap).
//
// The ORDERING state is per ORIGIN THREAD, and the origin is a host thread -- not a device and not a stream: a
// `llama_context`'s compute stream is a different object this state never keys on, and MTP may run the draft and
// the target context concurrently on the same devices, each from its own host thread.  A single per-device
// `pending`/`q`/`done_ev` let one context consume the other's queued uploads, and then the other context's graph
// replayed without waiting for the copy that feeds it -- measured 2026-10-21 (gemma4 26B-A4B + MTP + `-ncmoe` +
// `-sm layer`, 2 devices): two identical runs produced different greedy text, and the same run became stable once
// a full two-device synchronize restored the ordering the fast path had removed.
//
// The key is the HOST THREAD ID.  The first version of this state (r101-r104) keyed it on `cudaStreamPerThread`,
// which is a SENTINEL, not a handle -- `#define hipStreamPerThread ((hipStream_t)2)` (hip_runtime_api.h; CUDA
// uses `(cudaStream_t)0x2`).  The runtime resolves it to the calling thread's stream per API call, but the VALUE
// the code holds and hashes on is `2` in every thread, so that map could only ever hold one entry and the state
// was shared exactly as the pre-r101 per-device struct was (issue #201, TODO #84).  Uploads are still ISSUED on
// `cudaStreamPerThread` (the stream the synchronous path uses, so a later get_tensor stays ordered after it), so
// `origin::stream` means "the stream of whichever thread is issuing", NOT an identity.  That is sound because the
// ordering is carried by the EVENTS -- `ev` for a per-copy upload, `flush_ev` for a batch -- which are recorded
// where the work was issued, and a consumer always waits on the EVENT, never on a stream handle.
//
// The ring BYTES stay per device, so they are guarded separately: a slot is reused only after every origin's
// issued copies have drained AND every origin's queued (not yet issued) uploads have been issued and drained.
struct ggml_cuda_async_input_origin {
    cudaStream_t stream       = nullptr;   // the issuing stream (`cudaStreamPerThread`: a sentinel, not an identity)
    cudaEvent_t  ev           = nullptr;   // recorded after the last per-copy upload on `stream`
    bool         pending      = false;
    cudaEvent_t  done_ev      = nullptr;   // recorded on `done_stream` when that graph_compute returns
    bool         has_done     = false;
    // Lazy done event (GGML_CUDA_ASYNC_INPUT_LAZY_DONE; default on with GGML_CUDA_ASYNC_INPUT_BATCH=1): a graph_compute
    // only marks it pending, and the next upload records it on that graph's stream.
    bool         done_pending = false;
    cudaStream_t done_stream  = nullptr;
    // GGML_CUDA_ASYNC_INPUT_BATCH: uploads queued until the next graph on this origin
    struct queued { char * dst; size_t off; size_t size; };
    std::vector<queued> q;
    cudaEvent_t  flush_ev     = nullptr;   // recorded after the last batch kernel: the ring is free once it completes
    bool         has_flush    = false;
    int64_t      hits         = 0;
};

struct ggml_cuda_async_input_dev {
    std::mutex  mtx;                 // serializes the ring bytes for this device
    char *      ring     = nullptr;  // pinned host ring
    size_t      cap      = 0;
    size_t      head     = 0;
    char *      ring_dev = nullptr;  // device alias of `ring` (batched uploads)
    std::map<std::thread::id, ggml_cuda_async_input_origin> origins;   // one entry per host thread that uploads
};
static ggml_cuda_async_input_dev g_async_input[GGML_CUDA_MAX_DEVICES];
// Default ON (GGML_CUDA_ASYNC_INPUT=0 disables): the synchronous path blocks the host on every small
// graph-input copy, which is a measurable cost on a slow host and invisible on a fast one.  The async path
// stages through a pinned ring and orders the graph on that ring with an event, so the host does not wait
// per copy.
// This was opt-in up to r89 because a run was NOT start-to-start deterministic when the consuming graph was
// captured.  That was NOT the ring: the uploads verify byte-correct at the graph's first read, consecutive
// graphs do not overlap, and HIP's wait-before-replay / event-after-replay ordering is correct (two
// standalone reproducers).  The defect was in `ggml_backend_cuda_set_tensor_async` /
// `_set_tensor_2d_async`: they issued a pinned-source `cudaMemcpyAsync` straight from the caller's live host
// pointer, and the scheduler uses that path for a host-resident *split* input -- a CPU split's output that
// the next ubatch rewrites in place -- so the copy could read the next ubatch's bytes (a rare single-graph
// corruption that propagates to the end of the chunk).  Both now snapshot through the pinned ring (below),
// so no upload reads memory the caller may still be writing.
// Record: wip/async-input-graph-ordering/FINDINGS-2026-10-19b.md, WORKLOG 261019.007.
static const bool g_async_input_on = [] {
    const char * e = getenv("GGML_CUDA_ASYNC_INPUT");
    return e == nullptr || atoi(e) != 0;
}();
static constexpr size_t GGML_CUDA_ASYNC_INPUT_MAX  = 1u << 20;
static constexpr size_t GGML_CUDA_ASYNC_INPUT_RING = 16u << 20;

// Default ON inside the (opt-in) async path; GGML_CUDA_ASYNC_INPUT_BATCH=0 restores the per-copy uploads.
static const bool g_async_input_batch = [] {
    const char * e = getenv("GGML_CUDA_ASYNC_INPUT_BATCH");
    return e == nullptr || atoi(e) != 0;
}();
static constexpr int GGML_CUDA_AIN_BATCH = 64;
struct ggml_cuda_ain_batch {
    int          n;
    char *       dst[GGML_CUDA_AIN_BATCH];
    const char * src[GGML_CUDA_AIN_BATCH];
    unsigned int size[GGML_CUDA_AIN_BATCH];
};
// blockIdx.x = the upload, blockIdx.y = a 64 KiB chunk of it; byte-exact copy from the mapped pinned ring
static __global__ void ggml_cuda_ain_batch_kernel(const ggml_cuda_ain_batch b) {
    const int i = blockIdx.x;
    if (i >= b.n) {
        return;
    }
    char * __restrict__       d = b.dst[i];
    const char * __restrict__ s = b.src[i];
    const size_t n  = b.size[i];
    for (size_t c0 = (size_t) blockIdx.y * 65536; c0 < n; c0 += (size_t) gridDim.y * 65536) {
        const size_t c1 = c0 + 65536 < n ? c0 + 65536 : n;
        if ((((uintptr_t) (d + c0)) | ((uintptr_t) (s + c0)) | (c1 - c0)) % 16 == 0) {
            for (size_t k = c0 + 16 * threadIdx.x; k < c1; k += 16 * blockDim.x) {
                *(int4 *) (d + k) = *(const int4 *) (s + k);
            }
        } else {
            for (size_t k = c0 + threadIdx.x; k < c1; k += blockDim.x) {
                d[k] = s[k];
            }
        }
    }
}

// GGML_CUDA_ASYNC_INPUT_DEBUG=1: diagnostic only -- it reports what the ring actually did and never changes
// what is computed.  First N prints only, same convention as GGML_CUDA_MMF32_LOG: every origin CREATED (the
// map key, the calling thread, the resulting per-device origin count) and the first few waits (origin count +
// how many queued uploads THAT wait issued).  It is the instrument that makes the origin count a measurement
// instead of an assumption: two contexts decoding on one device from two host threads must log origins=2, and
// a wait must issue its OWN queue (issued=1), never another thread's.
static const int g_async_input_dbg = [] {
    const char * e = getenv("GGML_CUDA_ASYNC_INPUT_DEBUG");
    return e == nullptr ? 0 : atoi(e);
}();
// The map key: the origin, i.e. the HOST THREAD.  This one line is the whole fix for issue #201 -- the previous
// key was `cudaStreamPerThread`, the sentinel `((hipStream_t)2)` that is `2` in every thread.
static std::thread::id ggml_cuda_async_input_origin_key() {
    return std::this_thread::get_id();
}
static unsigned long long ggml_cuda_ain_tid(std::thread::id id) {
    return (unsigned long long) std::hash<std::thread::id>()(id);   // stable within a process
}

// this thread's ordering state (caller holds d.mtx); created on first use
static ggml_cuda_async_input_origin & ggml_cuda_async_input_origin_locked(ggml_cuda_async_input_dev & d, int device) {
    const std::thread::id key = ggml_cuda_async_input_origin_key();
    auto it = d.origins.find(key);
    if (it == d.origins.end()) {
        it = d.origins.emplace(key, ggml_cuda_async_input_origin()).first;
        it->second.stream = cudaStreamPerThread;
        if (g_async_input_dbg) {
            fprintf(stderr, "AIN origin: dev=%d key=%llx thread=%llx origins=%zu\n", device,
                    ggml_cuda_ain_tid(key), ggml_cuda_ain_tid(std::this_thread::get_id()), d.origins.size());
        }
    }
    return it->second;
}

// Allocate the ring (per device) and this origin's events (caller holds d.mtx).  false = fall back to the
// synchronous path for this copy.
static bool ggml_cuda_async_input_arm_locked(ggml_cuda_async_input_dev & d, ggml_cuda_async_input_origin & o) {
    if (d.ring == nullptr) {
        if (cudaMallocHost((void **) &d.ring, GGML_CUDA_ASYNC_INPUT_RING) != cudaSuccess) {
            (void) cudaGetLastError();
            d.ring = nullptr;
            return false;
        }
        d.cap = GGML_CUDA_ASYNC_INPUT_RING;
        if (g_async_input_batch) {
            void * dp = nullptr;
            if (cudaHostGetDevicePointer(&dp, d.ring, 0) == cudaSuccess) {
                d.ring_dev = (char *) dp;
                GGML_LOG_INFO("%s: device ring: batched graph-input uploads on (GGML_CUDA_ASYNC_INPUT_BATCH=0 disables)\n", __func__);
            } else {
                (void) cudaGetLastError();
                d.ring_dev = nullptr;
            }
        }
    }
    if (o.ev == nullptr && cudaEventCreateWithFlags(&o.ev, cudaEventDisableTiming) != cudaSuccess) {
        (void) cudaGetLastError();
        o.ev = nullptr;
        return false;
    }
    if (o.done_ev == nullptr && cudaEventCreateWithFlags(&o.done_ev, cudaEventDisableTiming) != cudaSuccess) {
        (void) cudaGetLastError();
        o.done_ev = nullptr;
        return false;
    }
    if (g_async_input_batch && o.flush_ev == nullptr &&
        cudaEventCreateWithFlags(&o.flush_ev, cudaEventDisableTiming) != cudaSuccess) {
        (void) cudaGetLastError();
        o.flush_ev = nullptr;
    }
    return true;
}

// the old path for one upload already in the ring (caller holds d.mtx)
static void ggml_cuda_async_input_copy_locked(ggml_cuda_async_input_dev & d, ggml_cuda_async_input_origin & o,
                                              void * dst, const char * slot, size_t size) {
    if (o.done_pending) {
        CUDA_CHECK(cudaEventRecord(o.done_ev, o.done_stream));
        o.done_pending = false;
        o.has_done     = true;
    }
    if (o.has_done) {
        // the previous graph on this origin may still read this input: copy only after it (a GPU-side wait)
        CUDA_CHECK(cudaStreamWaitEvent(o.stream, o.done_ev, 0));
    }
    CUDA_CHECK(cudaMemcpyAsync(dst, slot, size, cudaMemcpyHostToDevice, o.stream));
    CUDA_CHECK(cudaEventRecord(o.ev, o.stream));
    o.pending = true;
}

// issue the queued uploads of ONE origin through the old per-copy path, in order (caller holds d.mtx)
static void ggml_cuda_async_input_drain_locked(ggml_cuda_async_input_dev & d, ggml_cuda_async_input_origin & o) {
    for (const auto & e : o.q) {
        ggml_cuda_async_input_copy_locked(d, o, e.dst, d.ring + e.off, e.size);
    }
    o.q.clear();
}

// issue an origin's queued uploads as one batch kernel on its own stream (caller holds d.mtx)
static void ggml_cuda_async_input_batch_locked(ggml_cuda_async_input_dev & d, ggml_cuda_async_input_origin & o) {
    if (o.done_pending) {
        CUDA_CHECK(cudaEventRecord(o.done_ev, o.done_stream));
        o.done_pending = false;
        o.has_done     = true;
    }
    if (o.has_done) {
        CUDA_CHECK(cudaStreamWaitEvent(o.stream, o.done_ev, 0));
    }
    ggml_cuda_ain_batch b;
    b.n = (int) o.q.size();
    size_t mx = 0;
    for (int i = 0; i < b.n; ++i) {
        b.dst[i]  = o.q[i].dst;
        b.src[i]  = d.ring_dev + o.q[i].off;
        b.size[i] = (unsigned int) o.q[i].size;
        mx = std::max(mx, o.q[i].size);
    }
    const unsigned int ny = (unsigned int) std::min<size_t>(16, std::max<size_t>(1, (mx + 65535) / 65536));
    ggml_cuda_ain_batch_kernel<<<dim3(b.n, ny), 256, 0, o.stream>>>(b);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(o.flush_ev, o.stream));
    o.has_flush = true;
    o.q.clear();
}

// any access to a CUDA buffer other than a graph-input upload or a graph compute calls this first.  EVERY
// origin's queue is issued (on that origin's own stream), because the buffer being freed or read synchronously
// may have been filled by any of them.
static void ggml_cuda_async_input_drain(int device) {
    if (!g_async_input_batch || device < 0 || device >= GGML_CUDA_MAX_DEVICES) {
        return;
    }
    ggml_cuda_async_input_dev & d = g_async_input[device];
    std::lock_guard<std::mutex> lock(d.mtx);
    bool armed = false;
    for (auto & kv : d.origins) {
        ggml_cuda_async_input_origin & o = kv.second;
        if (o.q.empty() && !o.has_flush) {
            continue;
        }
        if (!armed) {
            ggml_cuda_set_device(device);
            armed = true;
        }
        if (o.has_flush) {
            // the synchronous path shares this origin's stream with the old-path uploads: keep it after the last batch kernel too
            CUDA_CHECK(cudaStreamWaitEvent(o.stream, o.flush_ev, 0));
        }
        if (!o.q.empty()) {
            if (d.ring_dev != nullptr) {
                ggml_cuda_async_input_batch_locked(d, o);
            } else {
                ggml_cuda_async_input_drain_locked(d, o);
            }
        }
    }
}

static bool ggml_cuda_async_input_set(int device, void * dst, const void * src, size_t size) {
    ggml_cuda_async_input_dev & d = g_async_input[device];
    std::lock_guard<std::mutex> lock(d.mtx);
    ggml_cuda_async_input_origin & o = ggml_cuda_async_input_origin_locked(d, device);
    if (!ggml_cuda_async_input_arm_locked(d, o)) {
        return false;
    }
    const size_t need = (size + 255) & ~(size_t) 255;
    if (d.head + need > d.cap) {
        // The ring bytes are per DEVICE while the ordering state is per origin thread, so the head may only wrap
        // once NO origin can still read the slots: another origin's copies may be in flight (`ev`/`flush_ev`), and
        // its uploads may still be QUEUED -- not even issued -- in which case their bytes would be overwritten
        // before the copy ever reads them.  So issue every origin's queue first, then let the device drain, and
        // only then reset the head.  This is the only place the ring is reused, it happens once per 16 MiB of
        // small uploads, and it is host-blocking by design: correctness outranks that cost.
        for (auto & kv : d.origins) {
            ggml_cuda_async_input_origin & e = kv.second;
            if (e.q.empty()) {
                continue;
            }
            if (d.ring_dev != nullptr) {
                ggml_cuda_async_input_batch_locked(d, e);
            } else {
                ggml_cuda_async_input_drain_locked(d, e);
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        for (auto & kv : d.origins) {
            kv.second.pending   = false;
            kv.second.has_flush = false;
        }
        d.head = 0;
    }
    char * slot = d.ring + d.head;
    d.head += need;
    memcpy(slot, src, size);
    if (d.ring_dev != nullptr && size <= 0xffffffffu) {
        char * const dc = (char *) dst;
        for (const auto & e : o.q) {
            if (dc < e.dst + e.size && e.dst < dc + size) {   // overlaps a queued upload: keep the write order
                ggml_cuda_async_input_drain_locked(d, o);
                break;
            }
        }
        if ((int) o.q.size() >= GGML_CUDA_AIN_BATCH) {
            ggml_cuda_async_input_batch_locked(d, o);
        }
        o.q.push_back({ dc, (size_t) (slot - d.ring), size });
    } else {
        ggml_cuda_async_input_copy_locked(d, o, dst, slot, size);
    }
    if (o.hits++ == 0) {
        GGML_LOG_INFO("%s: device %d: async graph-input uploads on (GGML_CUDA_ASYNC_INPUT=1 enables)\n", __func__, device);
    }
    return true;
}

struct ggml_cuda_async_input_done_guard {
    int          device;
    cudaStream_t stream;
    ~ggml_cuda_async_input_done_guard() {
        // unconditional: the ring also carries the set_tensor_async / set_tensor_2d_async snapshots, which
        // are taken with GGML_CUDA_ASYNC_INPUT=0 too (see those functions)
        ggml_cuda_async_input_dev & d = g_async_input[device];
        std::lock_guard<std::mutex> lock(d.mtx);
        ggml_cuda_async_input_origin & o = ggml_cuda_async_input_origin_locked(d, device);
        if (o.done_ev == nullptr && cudaEventCreateWithFlags(&o.done_ev, cudaEventDisableTiming) != cudaSuccess) {
            (void) cudaGetLastError();
            o.done_ev = nullptr;
            return;
        }
        // default on only with GGML_CUDA_ASYNC_INPUT_BATCH=1 (i.e. the default): with per-copy uploads alone it measured
        // ~1 % slower on one GPU
        static const bool lazy = getenv("GGML_CUDA_ASYNC_INPUT_LAZY_DONE") ? atoi(getenv("GGML_CUDA_ASYNC_INPUT_LAZY_DONE")) != 0 : g_async_input_batch;
        if (lazy) {
            o.done_pending = true;
            o.done_stream  = stream;
            return;
        }
        CUDA_CHECK(cudaEventRecord(o.done_ev, stream));
        o.has_done = true;
    }
};

static void ggml_cuda_async_input_flush_done(int device, cudaStream_t stream) {
    // Unconditional, exactly like the done guard and ggml_cuda_async_input_wait: since r90 the ring also
    // carries the `set_tensor_async` / `set_tensor_2d_async` snapshots, which are taken with
    // GGML_CUDA_ASYNC_INPUT=0 too.  Returning early here left a pending done-event naming a stream that the
    // context destructor destroys right after this call, so the next context on that device recorded the
    // event on a dead handle ("invalid resource handle", ggml_cuda_async_input_copy_locked) the first time it
    // reused the device -- reproduced by test-llama-archs -a deepseek4 with GGML_CUDA_ASYNC_INPUT=0, whose
    // Meta row is the first one to reuse a device after the per-device row's context is gone.  A no-op when
    // nothing is pending.  Scans EVERY origin: the destroying thread need not be the decoding thread.
    ggml_cuda_async_input_dev & d = g_async_input[device];
    std::lock_guard<std::mutex> lock(d.mtx);
    for (auto & kv : d.origins) {
        ggml_cuda_async_input_origin & o = kv.second;
        if (o.done_pending && o.done_stream == stream) {
            CUDA_CHECK(cudaEventRecord(o.done_ev, o.done_stream));
            o.done_pending = false;
            o.has_done     = true;
        }
    }
}

static void ggml_cuda_async_input_wait(int device, cudaStream_t stream) {
    // unconditional (see the done guard): a no-op when nothing is pending.  Every origin is waited for / issued
    // here: a graph input may have been queued by another thread (the loader, or a second context), and an upload
    // must never be left queued when the graph that reads it runs.
    ggml_cuda_async_input_dev & d = g_async_input[device];
    std::lock_guard<std::mutex> lock(d.mtx);
    const std::thread::id self = std::this_thread::get_id();
    int issued = 0, ev_waits = 0, flush_waits = 0;
    for (auto & kv : d.origins) {
        ggml_cuda_async_input_origin & o = kv.second;
        // Only the OWNING thread may CONSUME this origin's state.  A foreign wait must still order itself behind
        // the events (it may read the buffer), but if it cleared `pending`/`has_flush` the owner's next graph
        // would find nothing to wait on and could read the buffer before its copy lands -- which is exactly the
        // defect a shared origin produced (issue #201).
        const bool owned = kv.first == self;
        // the events are waited for even when the queue is EMPTY: a queue another thread drained leaves this
        // origin unordered behind that drain's flush, and the flush must be consumed before the early-out.
        if (o.pending) {
            CUDA_CHECK(cudaStreamWaitEvent(stream, o.ev, 0));
            ev_waits++;
            if (owned) {
                o.pending = false;
            }
        }
        if (o.has_flush) {
            CUDA_CHECK(cudaStreamWaitEvent(stream, o.flush_ev, 0));
            flush_waits++;
            if (owned) {
                o.has_flush = false;
            }
        }
        if (o.q.empty()) {
            continue;
        }
        // after the waits above: an older old-path copy to the same bytes must land first.  Like each old-path
        // copy, wait for the last graph on this origin (it may run on another stream and still read these inputs).
        issued += (int) o.q.size();
        if (d.ring_dev != nullptr) {
            // batched: issued on THIS graph's stream (o.stream resolves to the caller's); the origin's flush event
            // orders its later graphs behind it
            ggml_cuda_async_input_batch_locked(d, o);
            CUDA_CHECK(cudaStreamWaitEvent(stream, o.flush_ev, 0));
            flush_waits++;
            o.has_flush = !owned;   // a foreign drain leaves the flush for the owner to consume
        } else {
            ggml_cuda_async_input_drain_locked(d, o);
            if (o.pending) {
                CUDA_CHECK(cudaStreamWaitEvent(stream, o.ev, 0));
                ev_waits++;
                o.pending = owned;
            }
        }
    }
    if (g_async_input_dbg) {
        static std::atomic<int> printed{0};
        if (printed.fetch_add(1) < 16) {
            fprintf(stderr, "AIN wait: dev=%d key=%llx thread=%llx origins=%zu issued=%d ev=%d flush=%d\n",
                    device, ggml_cuda_ain_tid(ggml_cuda_async_input_origin_key()), ggml_cuda_ain_tid(self),
                    d.origins.size(), issued, ev_waits, flush_waits);
        }
    }
}

// One reused pinned chunk per device for the bounce of pageable host copies (ggml_cuda_bounce_min).  The caller holds
// `lock` for as long as it uses the chunk.  Returns nullptr if the chunk
// cannot be allocated (pinned host memory exhausted; the error is cleared).  A failed allocation is not retried: the
// bounce runs on every small copy, so a retry per call would put a failing 64 MiB hipHostMalloc on the hot path.
static constexpr size_t ggml_cuda_pin_chunk_bytes = 64ull << 20;

static char * ggml_cuda_pin_chunk(int device, std::unique_lock<std::mutex> & lock) {
    static std::mutex mutex[GGML_CUDA_MAX_DEVICES];
    static void *     buf[GGML_CUDA_MAX_DEVICES] = {};
    static bool       failed[GGML_CUDA_MAX_DEVICES] = {};
    lock = std::unique_lock<std::mutex>(mutex[device]);
    if (buf[device] == nullptr && !failed[device] && cudaMallocHost(&buf[device], ggml_cuda_pin_chunk_bytes) != cudaSuccess) {
        (void) cudaGetLastError();
        buf[device]    = nullptr;
        failed[device] = true;
    }
    return (char *) buf[device];
}

// A copy between the device and pageable host memory is slow on ROCm at any size: a large one runs at ~2 GB/s through
// the runtime's own staging, and a small one still costs ~25 us (1-D) / ~57 us (2-D) on gfx1201.  The server's context
// checkpoint (~118 MB of recurrent state in 512 KiB tensors) is saved after every prompt, before the first token, and
// its prompt-cache save of a slot reads the KV / recurrent state as thousands of small row ranges (twice as many under
// -sm tensor, one per device), so all such copies bounce through the reused pinned chunk instead.
// Only a pageable host pointer bounces: a pinned one (e.g. a CUDA_Host graph input) already copies at full speed, so
// the extra memcpy would be pure overhead.  If the chunk cannot be allocated the copy stays direct.
// GGML_CUDA_BOUNCE_MIN sets the threshold in bytes (default 1 = every size, 0 or an unparsable value = off).
static size_t ggml_cuda_bounce_min() {
    static const size_t v = [] {
        const char * e = getenv("GGML_CUDA_BOUNCE_MIN");
        const size_t n = e != nullptr ? (size_t) atoll(e) : (size_t) 1;
        return n != 0 ? n : SIZE_MAX;
    }();
    return v;
}

#if defined(GGML_USE_HIP) || defined(GGML_USE_CUDA)
static bool ggml_cuda_host_ptr_pageable(const void * ptr) {
    cudaPointerAttributes attr;
    if (cudaPointerGetAttributes(&attr, ptr) != cudaSuccess) {
        (void) cudaGetLastError();   // an unregistered host pointer is an error on some runtimes
        return true;
    }
    return attr.type != cudaMemoryTypeHost && attr.type != cudaMemoryTypeDevice && attr.type != cudaMemoryTypeManaged;
}
#else
// vendors/musa.h has no cudaPointerGetAttributes/cudaMemoryType* aliases; keep the MUSA transport direct rather
// than add untested defines (the bounce is a ROCm win).  The threshold also makes the call sites short-circuit.
static bool ggml_cuda_host_ptr_pageable(const void * /*ptr*/) { return false; }
#endif

static void ggml_backend_cuda_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    if (g_async_input_on && size <= GGML_CUDA_ASYNC_INPUT_MAX &&
        ggml_backend_buffer_get_usage(buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
        ggml_cuda_async_input_set(ctx->device, (char *) tensor->data + offset, data, size)) {
        return;
    }
    ggml_cuda_async_input_drain(ctx->device);
    // see ggml_cuda_bounce_min (the context checkpoint restore).  A model weight is NOT special: a
    // WEIGHTS buffer is just a destination.  The exemption here used to keep the loader's mmap/pinned
    // uploads on the direct path, but it also let a genuinely pageable source reach the bare
    // `cudaMemcpyAsync` below -- and on ROCm the copy engine then reads non-resident anonymous pages and
    // faults (issue #137-class: `[gfxhub] page fault ... Faulty UTCL2 client ID: SDMA0 (0xd)`, seen on the
    // `--load-mode none` + `-sm tensor` Meta upload, whose source is the loader's pageable `read_buf`).
    // Pageability is the only thing that decides: a pinned source (the loader's async ring, a CUDA_Host
    // graph input) still copies directly, so both the host-pinned and the bounce path are correct.
    std::unique_lock<std::mutex> lock;
    char * pin = nullptr;
    if (size >= ggml_cuda_bounce_min() &&
        ggml_cuda_host_ptr_pageable(data) && (pin = ggml_cuda_pin_chunk(ctx->device, lock)) != nullptr) {
        for (size_t done = 0; done < size; done += ggml_cuda_pin_chunk_bytes) {
            const size_t n = std::min(ggml_cuda_pin_chunk_bytes, size - done);
            memcpy(pin, (const char *) data + done, n);
            CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset + done, pin, n, cudaMemcpyHostToDevice, cudaStreamPerThread));
            CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        }
        return;
    }
    CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    ggml_cuda_async_input_drain(ctx->device);
    // see ggml_cuda_bounce_min (the server's context checkpoint save)
    std::unique_lock<std::mutex> lock;
    char * pin = nullptr;
    if (size >= ggml_cuda_bounce_min() && ggml_cuda_host_ptr_pageable(data) &&
        (pin = ggml_cuda_pin_chunk(ctx->device, lock)) != nullptr) {
        for (size_t done = 0; done < size; done += ggml_cuda_pin_chunk_bytes) {
            const size_t n = std::min(ggml_cuda_pin_chunk_bytes, size - done);
            CUDA_CHECK(cudaMemcpyAsync(pin, (const char *) tensor->data + offset + done, n, cudaMemcpyDeviceToHost, cudaStreamPerThread));
            CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
            memcpy((char *) data + done, pin, n);
        }
        return;
    }
    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

// Load-path 2-D staging: a grow-only pinned buffer per device, reused across the tensors of a model
// load instead of one `cudaMallocHost`+`cudaFreeHost` pair per tensor.  The loader uploads a `-sm
// tensor` model one tensor at a time through `set_tensor_2d`; each call used to allocate ~30 MB of
// pinned memory, copy, sync and free it, and a 27B Q8_0 at 3 devices does that 678 times (measured
// `hipHostMalloc` 2.2 s + `hipHostFree` 1.1 s of a ~10 s load on gfx1201).  Grow-only with a cap so
// the retained pinned footprint stays bounded; above the cap the copy falls back to the one-shot
// allocation.  `GGML_CUDA_STAGE_PIN=0` restores the per-call allocation.
static constexpr size_t ggml_cuda_stage_pin_max = 64ull << 20;
static const bool ggml_cuda_stage_pin_on = [] {
    const char * e = getenv("GGML_CUDA_STAGE_PIN");
    return e == nullptr || atoi(e) != 0;
}();

static char * ggml_cuda_stage_pin(int device, size_t need, std::unique_lock<std::mutex> & lock) {
    static std::mutex mutex[GGML_CUDA_MAX_DEVICES];
    static void *     buf[GGML_CUDA_MAX_DEVICES] = {};
    static size_t     cap[GGML_CUDA_MAX_DEVICES] = {};
    lock = std::unique_lock<std::mutex>(mutex[device]);
    if (cap[device] >= need) {
        return (char *) buf[device];
    }
    if (buf[device] != nullptr) {
        (void) cudaFreeHost(buf[device]);
        buf[device] = nullptr;
        cap[device] = 0;
    }
    if (cudaMallocHost(&buf[device], need) != cudaSuccess) {
        (void) cudaGetLastError();
        buf[device] = nullptr;
        return nullptr;
    }
    cap[device] = need;
    return (char *) buf[device];
}

static void ggml_backend_cuda_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    ggml_cuda_async_input_drain(ctx->device);
    // NEVER H2D straight from a (potentially pageable) host pointer with a 2-D copy: on ROCm a
    // pageable-source `hipMemcpy2DAsync` is pathologically slow AND can fault ("Page not present").
    // It crashed the `-sm tensor` + host-expert load, whose master is the pageable `CPU_REPACK`
    // buffer (unlike `-sm layer`'s pinned `ROCm_Host`, or the mmap path).  Gather the source rows
    // into a pinned staging buffer and copy from there -- the same pattern the per-ubatch splice
    // (`ggml_backend_cuda_set_tensor_2d_async`) already uses.
    if (n_copies > 1) {
        const size_t full    = (n_copies - 1) * stride_data   + size;
        const size_t compact = (n_copies - 1) * stride_tensor + size;
        const size_t need    = (stride_tensor == size) ? compact : full;
        void * pin = nullptr;
        // Reuse the per-device staging buffer for the repeated same-size uploads a model load makes
        // (see ggml_cuda_stage_pin); a one-shot allocation is the fallback and the path for a size
        // beyond the retention cap.
        std::unique_lock<std::mutex> lock;
        if (ggml_cuda_stage_pin_on && need <= ggml_cuda_stage_pin_max) {
            pin = ggml_cuda_stage_pin(ctx->device, need, lock);
        }
        const bool one_shot = pin == nullptr;
        if (one_shot) {
            CUDA_CHECK(cudaMallocHost(&pin, need));
        }
        if (stride_tensor == size) {
            for (size_t e = 0; e < n_copies; ++e) {
                memcpy((char *) pin + e * size, (const char *) data + e * stride_data, size);
            }
            CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, pin, compact, cudaMemcpyHostToDevice, cudaStreamPerThread));
        } else {
            memcpy(pin, data, full);
            CUDA_CHECK(cudaMemcpy2DAsync((char *) tensor->data + offset, stride_tensor, pin, stride_data, size, n_copies,
                    cudaMemcpyHostToDevice, cudaStreamPerThread));
        }
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        if (one_shot) {
            CUDA_CHECK(cudaFreeHost(pin));   // a cached staging buffer is kept for the next tensor
        }
        return;
    }
    CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_get_tensor_2d(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    ggml_cuda_async_input_drain(ctx->device);
    // A 2-D D2H into pageable memory is slow on ROCm even when small (~57 us per call on gfx1201): under -sm tensor the
    // server's prompt-cache save reads every split KV / recurrent-state row range this way, thousands of calls per slot.
    // Pack the rows into the pinned chunk and scatter them on the host instead (see ggml_cuda_bounce_min).
    std::unique_lock<std::mutex> lock;
    char * pin = nullptr;
    if (size > 0 && n_copies > 0 && size <= ggml_cuda_pin_chunk_bytes && size * n_copies >= ggml_cuda_bounce_min() &&
        ggml_cuda_host_ptr_pageable(data) && (pin = ggml_cuda_pin_chunk(ctx->device, lock)) != nullptr) {
        const size_t rows = ggml_cuda_pin_chunk_bytes / size;
        for (size_t e0 = 0; e0 < n_copies; e0 += rows) {
            const size_t n = std::min(rows, n_copies - e0);
            CUDA_CHECK(cudaMemcpy2DAsync(pin, size, (const char *) tensor->data + offset + e0 * stride_tensor, stride_tensor,
                    size, n, cudaMemcpyDeviceToHost, cudaStreamPerThread));
            CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
            for (size_t e = 0; e < n; ++e) {
                memcpy((char *) data + (e0 + e) * stride_data, pin + e * size, size);
            }
        }
        return;
    }
    CUDA_CHECK(cudaMemcpy2DAsync(
        data, stride_data, (const char *) tensor->data + offset, stride_tensor, size, n_copies, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

// defined further down: the staged (pinned host bounce) cross-device copy, used for the slab peer pairs the
// driver refused to grant access to (a peer copy into those faults the GPU)
static void ggml_cuda_peer_copy_staged(
        int src_device, cudaStream_t src_stream,
        int dst_device, cudaStream_t dst_stream,
        void * dst, const void * src, size_t nbytes);

static bool ggml_backend_cuda_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_cuda(src->buffer)) {
        ggml_backend_cuda_buffer_context * src_ctx = (ggml_backend_cuda_buffer_context *)src->buffer->context;
        ggml_backend_cuda_buffer_context * dst_ctx = (ggml_backend_cuda_buffer_context *)dst->buffer->context;
        ggml_cuda_async_input_drain(src_ctx->device);
        ggml_cuda_async_input_drain(dst_ctx->device);
        // compare the backing physical devices: distinct virtual devices may share one physical GPU,
        // in which case a same-device copy (not a peer copy) is required
        const int src_physical = ggml_cuda_get_physical_device(src_ctx->device);
        const int dst_physical = ggml_cuda_get_physical_device(dst_ctx->device);
        if (src_physical == dst_physical) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(src), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            if (!ggml_cuda_slab_peer_copy_ok(dst->data, ggml_nbytes(src))) {
                ggml_cuda_peer_copy_staged(src_ctx->device, cudaStreamPerThread,
                                           dst_ctx->device, cudaStreamPerThread,
                                           dst->data, src->data, ggml_nbytes(src));
            } else {
                CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_physical, src->data, src_physical, ggml_nbytes(src), cudaStreamPerThread));
            }
#endif
        }
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cuda_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    ggml_cuda_async_input_drain(ctx->device);
    CUDA_CHECK(cudaMemsetAsync(ctx->dev_ptr, value, buffer->size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static const ggml_backend_buffer_i ggml_backend_cuda_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_cuda_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cuda_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cuda_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_cuda_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cuda_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cuda_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_cuda_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_cuda_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_cuda_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cuda_buffer_clear,
    /* .reset           = */ NULL,
};

// cuda buffer type
struct ggml_backend_cuda_buffer_type_context {
    int device;
    std::string name;
};

// See ggml-cuda-arena.cuh: the compute-buffer chunk decision needs this buffer type's device, because
// the slab can decline on one device while another keeps a live slab.
int ggml_backend_cuda_buffer_type_device(ggml_backend_buffer_type_t buft) {
    return ((const ggml_backend_cuda_buffer_type_context *) buft->context)->device;
}

static const char * ggml_backend_cuda_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_buffer_type_context * ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static bool ggml_backend_buft_is_cuda(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_buffer_type_get_name;
}

static ggml_backend_buffer_t ggml_backend_cuda_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    ggml_cuda_set_device(buft_ctx->device);

    void * dev_ptr;
    // WIP r42 (TODO #42): the fail-soft MoE-arena yield lives inside `ggml_cuda_device_malloc` (the
    // choke point shared with the workspace pool and the Q8_1 cache arena), so this path needs no retry
    // loop of its own.
    cudaError_t err = ggml_cuda_device_malloc(&dev_ptr, size, buft_ctx->device);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        GGML_LOG_ERROR("%s: allocating %.2f MiB on device %d: cudaMalloc failed: %s\n", __func__, size / 1024.0 / 1024.0, buft_ctx->device, cudaGetErrorString(err));
        return nullptr;
    }

    ggml_backend_cuda_buffer_context * ctx = new ggml_backend_cuda_buffer_context(buft_ctx->device, dev_ptr);

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_buffer_interface, ctx, size);
}

// r42 (OPEN 2): allocate the COMPUTE buffer from the per-device VMM pool when enabled.  Only the graph
// allocator calls this (with the real usage); model weights go through alloc_buffer and are untouched.
static ggml_backend_buffer_t ggml_backend_cuda_buffer_type_alloc_buffer_usage(ggml_backend_buffer_type_t buft, size_t size, enum ggml_backend_buffer_usage usage) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    if (usage == GGML_BACKEND_BUFFER_USAGE_COMPUTE && ggml_cuda_work_alloc_debug()) {
        GGML_LOG_WARN("COMPUTE_ALLOC %.1f MiB dev=%d slab_on=%d flag=%d\n", (double) size/1048576.0,
                      buft_ctx->device, (int) ggml_cuda_slab_enabled(), (int) ggml_cuda_slab_compute_enabled());
    }

    if (usage == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        // OPEN 2: the movable-boundary slab.  The compute buffer is a VIEW of the slab's work region: its
        // base is the slab base (so a growing layout keeps its tensor addresses) and its REPORTED size is
        // the boundary (so growth inside the region does not look like a realloc to the graph allocator).
        // Growing past the boundary moves the split -- evicting the arena tables in the taken chunks -- and
        // HIP is NOT called at runtime.
        if (ggml_cuda_slab_enabled() && ggml_cuda_slab_compute_enabled()) {
            ggml_cuda_set_device(buft_ctx->device);
            void * dev_ptr = ggml_cuda_slab_work_alloc(buft_ctx->device, size);
            if (dev_ptr != NULL) {
                // REPORT the size that was actually REQUESTED, not the slab's boundary.  The boundary is
                // shared by every view (the widest one pins it), so reporting it makes a small buffer
                // claim the whole region: `ggml_vbuffer_size()` feeds the graph allocator AND the
                // `--fit` memory accounting, and a fit probe buffer claiming ~11.5 GiB instead of its
                // requested 512 MiB sends the fit into seven extra rounds, where a pre-existing Meta
                // backend assert can fire (measured: 1-2 starts in 5 crashed with the boundary reported,
                // 5/5 clean with the request reported -- and 5/5 clean with the slab off entirely).
                // Growth is unaffected: the requested size already carries the chunk rounding and the
                // spare chunk, and beyond that a "realloc" under the slab re-uses the SAME base (the
                // work region's VA never moves), so it costs a layout pass, not a move.
                GGML_LOG_INFO("%s: compute buffer %.2f MiB from the slab work region (boundary %.2f MiB) on device %d\n",
                              __func__, size / 1024.0 / 1024.0, ggml_cuda_slab_work_size(buft_ctx->device) / 1024.0 / 1024.0,
                              buft_ctx->device);
                ggml_backend_cuda_buffer_context * ctx = new ggml_backend_cuda_buffer_context(buft_ctx->device, dev_ptr);
                ctx->slab_view     = true;
                ctx->slab_need     = size;
                ctx->slab_off      = (size_t) ((char *) dev_ptr - (char *) ggml_cuda_slab_work_base(buft_ctx->device));
                return ggml_backend_buffer_init(buft, ggml_backend_cuda_buffer_interface, ctx, size);
            }
            GGML_LOG_WARN("%s: slab work region has no room for %.2f MiB on device %d; falling back\n",
                          __func__, size / 1024.0 / 1024.0, buft_ctx->device);
            // The slab can also fail because `work estimate + cache floor` does not fit at all, which is
            // not a "no room right now" case: the arena and a wide prefill can never coexist here, so the
            // cache must STREAM instead of building one.  This is the earliest point the answer exists and
            // it is before any cache-consulting graph, which is what makes the disable safe (see
            // `moe_cache_disable_streaming`); the slab lock is already released here.
            if (ggml_cuda_slab_cache_unusable(buft_ctx->device)) {
                moe_cache_disable_streaming("the movable-boundary slab cannot hold the work buffers plus the "
                                            "minimum MoE cache on this device (MOE_EXPERT_CACHE_MIB / "
                                            "GGML_CUDA_SLAB_MIN_ARENA_MIB set the floor, GGML_CUDA_SLAB_RING_MIB "
                                            "the H2D-ring hole); lower GGML_CUDA_SLAB_RESERVE_MIB if the reserve "
                                            "is too large");
            } else if (ggml_cuda_slab_configured() && ggml_cuda_slab_declined(buft_ctx->device)) {
                // Issue #136 (item F): the slab was armed for a host-expert model but could not be CREATED on
                // this device at all (VMM unavailable, bad chunk geometry, cudaMemGetInfo failure, or too
                // little free VRAM for the reserve).  Nothing was committed (`!inited`; the single
                // mid-init failure frees its VA), so streaming is safe -- but never silently build a raw
                // arena.  This is the earliest point the answer exists and is before any cache-consulting
                // graph, which is what makes the disable safe (see `moe_cache_disable_streaming`).  If it
                // is somehow too late, that would be an inconsistent run: refuse to continue.
                const char * why = ggml_cuda_slab_decline_reason(buft_ctx->device);
                if (moe_cache_is_sized()) {
                    GGML_ABORT("%s: device %d: the movable-boundary slab was armed but could not be created "
                               "(%s), and the MoE expert cache is already sized -- it is too late to stream "
                               "without an inconsistent run.  Set GGML_CUDA_SLAB=0 and restart, or report the "
                               "reproduction at https://github.com/stew675/llama-cpp-rdna-boosts/issues\n",
                               __func__, buft_ctx->device, why != nullptr ? why : "unknown reason");
                }
                static bool warned[GGML_CUDA_MAX_DEVICES] = {};
                if (!warned[buft_ctx->device]) {
                    warned[buft_ctx->device] = true;
                    GGML_LOG_ERROR("%s: device %d: the movable-boundary slab was armed for this host-expert "
                                   "model but could not be created (%s).  Disabling the MoE expert cache and "
                                   "streaming the experts from the host; the run is correct but slower.  "
                                   "Workarounds: free VRAM / lower GGML_CUDA_SLAB_RESERVE_MIB (or "
                                   "GGML_CUDA_SLAB_RING_MIB, MOE_EXPERT_CACHE_MIB), or set "
                                   "GGML_CUDA_SLAB=0 to silently take the plain path.  If this is unexpected, "
                                   "please report the reproduction at "
                                   "https://github.com/stew675/llama-cpp-rdna-boosts/issues\n",
                                   __func__, buft_ctx->device, why != nullptr ? why : "unknown reason");
                }
                moe_cache_disable_streaming("the movable-boundary slab was armed but could not be created on "
                                            "this device");
            }
        }
    }

    return ggml_backend_cuda_buffer_type_alloc_buffer(buft, size);
}

static size_t ggml_backend_cuda_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_cuda_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *) buft->context;

    size_t size = tensor->op == GGML_OP_FLASH_ATTN_EXT
        ? ggml_cuda_flash_attn_ext_get_alloc_size(buft_ctx->device, tensor)
        : ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    // [TAG_ALLOC_SIZE_EXPAND]
    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            GGML_ASSERT(tensor->nb[0] == ggml_element_size(tensor));
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;
}


static const ggml_backend_buffer_type_i ggml_backend_cuda_buffer_type_interface = {    /* .get_name            = */ ggml_backend_cuda_buffer_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_cuda_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ NULL,
    /* .get_alignment       = */ ggml_backend_cuda_buffer_type_get_alignment,
    /* .get_max_size        = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size      = */ ggml_backend_cuda_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ NULL,
    /* .is_host             = */ NULL,
    /* .get_compute_margin_pct = */ ggml_backend_cuda_buffer_type_get_compute_margin_pct,
    /* .alloc_buffer_usage  = */ ggml_backend_cuda_buffer_type_alloc_buffer_usage,
    /* .get_compute_chunk_bytes = */ ggml_backend_cuda_buffer_type_get_compute_chunk_bytes,
};

ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device >= ggml_backend_cuda_get_device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type ggml_backend_cuda_buffer_types[GGML_CUDA_MAX_DEVICES];

    static bool ggml_backend_cuda_buffer_type_initialized = false;

    if (!ggml_backend_cuda_buffer_type_initialized) {
        for (int i = 0; i < ggml_backend_cuda_get_device_count(); i++) {
            ggml_backend_cuda_buffer_types[i] = {
                /* .iface    = */ ggml_backend_cuda_buffer_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), i),
                /* .context  = */ new ggml_backend_cuda_buffer_type_context{i, GGML_CUDA_NAME + std::to_string(i)},
            };
        }
        ggml_backend_cuda_buffer_type_initialized = true;
    }

    return &ggml_backend_cuda_buffer_types[device];
}

// Communication context for multi-GPU AllReduce during tensor parallelism.
//
// Created once per meta backend instance.  Resources for the selected mode
// (NCCL communicators or the internal AllReduce pipeline) are initialised
// eagerly during comm_init so any init failure surfaces at startup rather
// than mid-run.
struct ggml_backend_cuda_comm_context {
    using try_allreduce_fn = bool(*)(ggml_backend_cuda_comm_context *, struct ggml_tensor **);

    std::vector<ggml_backend_t> backends;
    std::vector<int>            dev_ids;

    // Set by the init chain (comm_init_{nccl, internal, none}) to one of
    // try_allreduce_{nccl, internal, butterfly}.  nccl needs `comms`,
    // internal needs `ar_pipeline`, butterfly needs nothing.  Per-call
    // failures return false; the meta backend's generic implementation then
    // handles that call.
    try_allreduce_fn            try_allreduce = nullptr;

    ggml_cuda_ar_pipeline *     ar_pipeline = nullptr;

    // --- copy-engine (SDMA) P2P AllReduce scratch (GGML_CUDA_ALLREDUCE=ce; opt-in) ---
    // Lazily grown bf16 staging per rank + four events per rank for cross-device ordering:
    //   ce_ev_send : phase-1 (reduce-scatter) sends drained
    //   ce_ev_done : phase-1 local reduce complete (this rank's tmp may be overwritten)
    //   ce_ev_recv : phase-2 (all-gather) sends drained
    //   ce_ev_out  : final output conversion complete (cross-call tmp-reuse guard)
    std::vector<void *>         ce_buf;
    std::vector<void *>         ce_tmp;    // reduce-scatter receive regions (sender-indexed)
    std::vector<void *>         ce_tmp2;   // all-gather receive regions (sender-indexed)
    std::vector<cudaEvent_t>    ce_ev_send;
    std::vector<cudaEvent_t>    ce_ev_done;
    std::vector<cudaEvent_t>    ce_ev_recv;
    std::vector<cudaEvent_t>    ce_ev_out;
    std::vector<std::pair<int, void *>> ce_old;   // buffers retired on growth, freed at teardown
    size_t                      ce_bytes = 0;
    // Issue #136 (item C): set when a scratch grow failed (OOM) or exceeded the optional budget
    // (`GGML_CUDA_AR_CE_SCRATCH_MIB`), so later calls fall back to the internal pipeline without
    // re-attempting or re-logging.
    bool                        ce_grow_disabled  = false;
    bool                        ce_fallback_warned = false;

    // Set if NCCL fails at runtime (e.g. RCCL refusing kernel dispatch on a
    // root port without AtomicOp completer support; see ROCm/ROCm#6520).
    // Once set, NCCL is never retried: AllReduce falls back to the internal
    // pipeline (if available) or the meta backend's butterfly.  Always
    // defined (stays false) so the dispatcher needs no #ifdef in
    // non-NCCL builds.
    bool                        nccl_failed = false;

    // r165 default chain state (see ggml_backend_cuda_comm_init_auto / _select_arm).
    bool                        ce_ready   = false;   // the copy-engine large arm came up
    bool                        nccl_ready = false;   // the NCCL/RCCL large arm came up
    bool                        auto_select = false;  // the large arm has not been chosen yet
    try_allreduce_fn            nccl_arm    = nullptr; // the NCCL/RCCL large arm, kept for the selection

#ifdef GGML_USE_NCCL
    std::vector<ncclComm_t>     comms;
#endif // GGML_USE_NCCL

    ~ggml_backend_cuda_comm_context() {
        for (size_t i = 0; i < ce_buf.size(); ++i) {
            ggml_cuda_set_device(dev_ids[i]);
            // The AR's cross-device writes land in THIS rank's scratch from the PEER's stream, so
            // quiesce every rank before freeing anything.
            (void) cudaDeviceSynchronize();
        }
        for (size_t i = 0; i < ce_buf.size(); ++i) {
            ggml_cuda_set_device(dev_ids[i]);
            if (ce_ev_send[i] != nullptr) (void) cudaEventDestroy(ce_ev_send[i]);
            if (ce_ev_done[i] != nullptr) (void) cudaEventDestroy(ce_ev_done[i]);
            if (ce_ev_recv[i] != nullptr) (void) cudaEventDestroy(ce_ev_recv[i]);
            if (ce_ev_out[i]  != nullptr) (void) cudaEventDestroy(ce_ev_out[i]);
            if (ce_buf[i] != nullptr) (void) cudaFree(ce_buf[i]);
            if (ce_tmp[i] != nullptr) (void) cudaFree(ce_tmp[i]);
            if (ce_tmp2[i] != nullptr) (void) cudaFree(ce_tmp2[i]);
            (void) cudaGetLastError();
        }
        for (auto & p : ce_old) { ggml_cuda_set_device(p.first); (void) cudaFree(p.second); (void) cudaGetLastError(); }
#ifdef GGML_USE_NCCL
        for (ncclComm_t comm : comms) {
            // Not fatal: after a runtime NCCL failure the comm state is
            // unknown and destroy may report it.
            if (ncclCommDestroy(comm) != ncclSuccess) {
                GGML_LOG_WARN("failed to destroy NCCL comm (state unknown?)\n");
            }
        }
#endif // GGML_USE_NCCL
        ggml_cuda_ar_pipeline_free(ar_pipeline);
    }
};

// Shared size heuristic: tensors below these element counts are latency-bound
// (token generation), above them bandwidth-bound (prefill).  The internal
// host-staged pipeline wins on latency; NCCL/RCCL P2P wins on bandwidth.
//
// The two paths are NOT bit-identical (different summation order), so a tensor
// whose size straddles the crossover gets a different result depending on its
// SHAPE.  Under -sm tensor the reduced tensors scale with the batch width
// (ne = ne0 * n_tokens), so the crossover must sit above the whole decode +
// speculative-verify family: with ne0 = 5120 a 7-token verify batch is 35840
// elements, which used to cross the old 2-device 32768 limit and reduce via
// NCCL while 1..6-token decode stayed on the internal pipeline -- a 7-token
// verify batch then disagreed with 1-token decode in the last bits and greedy
// near-ties flipped (blocks 02/13 aligned the kernels; this completed it).
// Necessary and sufficient: the largest verify batch (--spec-draft-n-max 16
// -> 17 tokens = 87040 elements) must stay below it; 131072 covers 25 tokens
// and is still far below the internal pipeline's own 1 MB (262144 element)
// cap, so nothing is pushed off the fast path.
static bool ggml_backend_cuda_comm_is_small(int64_t ne, size_t n_backends) {
    return (n_backends <= 2 && ne < 131072) ||
           (n_backends == 3 && ne < 131072) ||
           (n_backends >= 4 && ne < 262144);
}

#ifdef GGML_USE_NCCL
static bool ggml_backend_cuda_comm_try_allreduce_internal(ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors);

// NCCL failed at runtime.  The communicators are in an unknown state, so
// never retry them: clear the sticky HIP errors the failed dispatch left on
// each AR device, re-route subsequent AllReduce to the internal pipeline
// when it is available, and warn once.  The call that failed returns false,
// so the meta backend's butterfly handles that one.
//
// Known trigger: RCCL >= 2.30.4 refuses to dispatch its generic kernels
// (hipErrorIllegalState: "the operation cannot be performed in the present
// state") when the upstream PCIe root port lacks 32/64-bit AtomicOp
// completer support, e.g. a GPU behind a chipset/PCH root port.  Init
// (ncclCommInitAll) succeeds, so this only surfaces on the first collective.
// No NCCL_* env var helps (the refusal happens at kernel dispatch, before
// any transport is used).  Verify: `dmesg | grep -i atomic`; see
// ROCm/ROCm#6520.
static void ggml_backend_cuda_comm_nccl_failed(ggml_backend_cuda_comm_context * comm_ctx, const char * err) {
    if (comm_ctx->nccl_failed) {
        return;
    }
    comm_ctx->nccl_failed = true;
    for (const auto & backend : comm_ctx->backends) {
        ggml_cuda_set_device(((ggml_backend_cuda_context *) backend->context)->device);
        (void) cudaGetLastError(); // clear the sticky error from the failed dispatch
    }
    if (comm_ctx->ar_pipeline != nullptr) {
        comm_ctx->try_allreduce = ggml_backend_cuda_comm_try_allreduce_internal;
    }
    GGML_LOG_WARN("NCCL AllReduce failed (%s) - not retrying NCCL, falling back to %s for the rest of this run. "
                  "If the error is hipErrorIllegalState, the PCIe root port most likely lacks AtomicOp completer "
                  "support (check: dmesg | grep -i atomic; see ROCm/ROCm#6520). Run with NCCL_DEBUG=INFO for details.\n",
                  err, comm_ctx->ar_pipeline != nullptr ? "the internal AllReduce pipeline" : "butterfly AllReduce");
}

// AllReduce via NCCL. Reduces as FP32 for small tensors and BF16 for large
// tensors (bandwidth-bound), then converts back to FP32.
static bool ggml_backend_cuda_comm_allreduce_nccl(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    const int64_t ne = ggml_nelements(tensors[0]);
    // FIXME the input of llm_graph_context::build_in_out_ids can produce a tensor with 0 elements if n_outputs == 0
    // This then causes a crash in this function
    if (ne == 0) {
        return true;
    }

    const size_t n_backends = comm_ctx->backends.size();

    for (size_t i = 0; i < n_backends; ++i) {
        GGML_ASSERT(tensors[i] != nullptr);
        GGML_ASSERT(ggml_nelements(tensors[i]) == ne);
        GGML_ASSERT(ggml_is_contiguously_allocated(tensors[i]));
    }

    // A failure here is terminal for NCCL but not for the run: see
    // ggml_backend_cuda_comm_nccl_failed().
    const auto nccl_try = [&](ncclResult_t rc) {
        if (rc != ncclSuccess) {
            ggml_backend_cuda_comm_nccl_failed(comm_ctx, ncclGetErrorString(rc));
            return false;
        }
        return true;
    };

    // For small tensors, simply reduce them as FP32.
    // The following heuristic for how "small" a tensor should be is based on RTX 4090s connected via 16x PCIe 4.0.
    if (ggml_backend_cuda_comm_is_small(ne, n_backends)) {
        for (size_t i = 0; i < n_backends; ++i) {
            if ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
                ggml_cuda_set_device(cuda_ctx->device);
                CUDA_CHECK(cudaMemsetAsync(tensors[i]->data, 0, ggml_nbytes(tensors[i]), cuda_ctx->stream()));
            }
        }
        if (!nccl_try(ncclGroupStart())) {
            return false;
        }
        for (size_t i = 0; i < n_backends; ++i) {
            ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
            if (!nccl_try(ncclAllReduce(tensors[i]->data, tensors[i]->data, ne, ncclFloat, ncclSum, comm_ctx->comms[i], cuda_ctx->stream()))) {
                return false;
            }
        }
        if (!nccl_try(ncclGroupEnd())) {
            return false;
        }
        return true;
    }

    // For large tensors it's faster to compress them to BF16 for the reduction:
    to_bf16_cuda_t to_bf16 = ggml_get_to_bf16_cuda(GGML_TYPE_F32);
    to_fp32_cuda_t to_fp32 = ggml_get_to_fp32_cuda(GGML_TYPE_BF16);

    ggml_cuda_pool_alloc<nv_bfloat16> tmp[GGML_CUDA_MAX_DEVICES];
    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
        // wip/slab-transient-absorption (step 2b): held across the grouped NCCL calls -- never in the band.
        ggml_cuda_pool_wide_pin wide_pin(cuda_ctx->pool());
        tmp[i].pool = &cuda_ctx->pool();
        tmp[i].alloc(ne);

        ggml_cuda_set_device(cuda_ctx->device);
        if (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) {
            to_bf16(tensors[i]->data, tmp[i].get(), ne, cuda_ctx->stream());
        } else {
            CUDA_CHECK(cudaMemsetAsync(tmp[i].get(), 0, ne * sizeof(nv_bfloat16), cuda_ctx->stream()));
        }
        CUDA_CHECK(cudaGetLastError());
    }

    if (!nccl_try(ncclGroupStart())) {
        return false;
    }
    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
        if (!nccl_try(ncclAllReduce(tmp[i].get(), tmp[i].get(), ne, ncclBfloat16, ncclSum, comm_ctx->comms[i], cuda_ctx->stream()))) {
            return false;
        }
    }
    if (!nccl_try(ncclGroupEnd())) {
        return false;
    }

    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;

        ggml_cuda_set_device(cuda_ctx->device);
        to_fp32(tmp[i].get(), (float *) tensors[i]->data, ne, cuda_ctx->stream());
        CUDA_CHECK(cudaGetLastError());
    }

    return true;
}
#endif // GGML_USE_NCCL

// Run the internal AR pipeline.  Returns false on unsupported / failed input
// -- the caller decides whether to abort (env-forced) or fall back silently.
static bool ggml_backend_cuda_comm_allreduce_internal(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    GGML_ASSERT(comm_ctx->ar_pipeline != nullptr);

    const size_t n_backends = comm_ctx->backends.size();
    GGML_ASSERT(n_backends >= 2);
    GGML_ASSERT(tensors[0] != nullptr);

    const int64_t   ne   = ggml_nelements(tensors[0]);
    const ggml_type type = tensors[0]->type;

    if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16 && type != GGML_TYPE_BF16) {
        GGML_LOG_DEBUG("%s: internal unsupported: type=%d\n", __func__, (int) type);
        return false;
    }

    if (ne == 0) {
        return true;
    }

    for (size_t i = 0; i < n_backends; ++i) {
        if (tensors[i] == nullptr) {
            GGML_LOG_ERROR("%s: internal failed: tensor[%zu] is null\n", __func__, i);
            return false;
        }
        if (ggml_nelements(tensors[i]) != ne || tensors[i]->type != type) {
            GGML_LOG_ERROR("%s: internal failed: tensor[%zu] ne=%" PRId64 " type=%d expected ne=%" PRId64 " type=%d\n",
                           __func__, i, ggml_nelements(tensors[i]), (int) tensors[i]->type, ne, (int) type);
            return false;
        }
        if (!ggml_is_contiguously_allocated(tensors[i])) {
            GGML_LOG_DEBUG("%s: internal unsupported: tensor[%zu] is not contiguously allocated: ne=%" PRId64 " nbytes=%zu packed=%zu type=%d\n",
                           __func__, i, ne, ggml_nbytes(tensors[i]),
                           (size_t) ne * ggml_type_size(type) / ggml_blck_size(type), (int) type);
            return false;
        }
        if (((uintptr_t) tensors[i]->data & 0xF) != 0) {
            GGML_LOG_DEBUG("%s: internal unsupported: tensor[%zu] data pointer is not 16-byte aligned: %p type=%d ne=%" PRId64 "\n",
                           __func__, i, tensors[i]->data, (int) type, ne);
            return false;
        }
        GGML_ASSERT((ggml_nbytes(tensors[i]) & 0xF) == 0);
    }

    return ggml_cuda_ar_allreduce(comm_ctx->ar_pipeline, comm_ctx->backends.data(), tensors);
}

// ---------------------------------------------------------------------------
// Per-call dispatch -- three variants, one per backend.  Each is set as
// comm_ctx->try_allreduce by the matching init step.  Per-call failure
// returns false; the meta backend's generic implementation handles that call.
// ---------------------------------------------------------------------------

#ifdef GGML_USE_NCCL
static bool ggml_backend_cuda_comm_try_allreduce_nccl(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    return ggml_backend_cuda_comm_allreduce_nccl(comm_ctx, tensors);
}
#endif // GGML_USE_NCCL

static bool ggml_backend_cuda_comm_try_allreduce_internal(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    return ggml_backend_cuda_comm_allreduce_internal(comm_ctx, tensors);
}

static bool ggml_backend_cuda_comm_try_allreduce_butterfly(
        ggml_backend_cuda_comm_context *, struct ggml_tensor **) {
    return false;
}

static void ggml_backend_cuda_comm_free(void * comm_ctx_v) {
    if (comm_ctx_v == nullptr) {
        return;
    }
    delete static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
}

// ---------------------------------------------------------------------------
// Init -- chained nccl -> internal -> none.  Each step tries to bring up its
// resource; on failure it warns and recurses into the next step.
// ---------------------------------------------------------------------------
static void ggml_backend_cuda_comm_init_none(ggml_backend_cuda_comm_context * ret) {
    ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_butterfly;
}

// Try to bring up the internal host-staged AR pipeline (2 GPUs only).  Returns
// true on success.  On failure it does NOT clobber ret->try_allreduce, so a
// hybrid setup can keep the NCCL path.
static bool ggml_backend_cuda_comm_init_internal(ggml_backend_cuda_comm_context * ret) {
    ret->ar_pipeline = ggml_cuda_ar_pipeline_init(ret->dev_ids.data(), ret->dev_ids.size());
    if (ret->ar_pipeline) {
        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_internal;
        return true;
    }

    // Clear sticky CUDA error from the failed init.
    (void) cudaGetLastError();
    GGML_LOG_DEBUG("internal AllReduce init failed (n_devices != 2?); "
                   "not using the internal path\n");
    return false;
}

// Try to bring up the NCCL/RCCL comms.  Returns true on success.
static bool ggml_backend_cuda_comm_init_nccl(ggml_backend_cuda_comm_context * ret) {
#ifdef GGML_USE_NCCL
    // Disabling NCCL path when CUDA virtual devices are in use since NCCL requires one distinct physical GPU per rank.
    const ggml_cuda_device_info & info = ggml_cuda_info();
    if (info.device_count > info.physical_device_count) {
        GGML_LOG_WARN("NCCL disabled: virtual devices in use\n");
        return false;
    }

    const size_t n = ret->dev_ids.size();
    ret->comms.resize(n);
    ncclResult_t rc = ncclCommInitAll(ret->comms.data(), (int) n, ret->dev_ids.data());
    if (rc == ncclSuccess) {
        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_nccl;
        return true;
    }

    ret->comms.clear();
    GGML_LOG_WARN("NCCL init failed (%s)\n", ncclGetErrorString(rc));
#else // GGML_USE_NCCL
#ifndef GGML_USE_HIP
    GGML_LOG_WARN("NCCL not compiled in.  "
                  "Recompile with -DGGML_CUDA_NCCL=ON for best multi-GPU performance.\n");
#endif // !GGML_USE_HIP
#endif // GGML_USE_NCCL
    return false;
}

// ---------------------------------------------------------------------------------------------
// Copy-engine (SDMA) P2P AllReduce (GGML_CUDA_ALLREDUCE=ce; opt-in, 2 GPUs).  The platform
// default stays the hybrid NCCL + internal pipeline described above.
//
// Same dtype story as the NCCL large-tensor path (fp32 -> bf16 -> reduce -> fp32) but the peer
// exchange is hipMemcpyPeerAsync on the compute streams, ordered with cross-device events, instead
// of NCCL's SM-driven kernels.  Rationale (measured): copy-engine transfers
// hide completely behind a WMMA-saturating GEMM while SM-driven transfers steal ~84-95% of the
// compute; the store here is therefore the prerequisite for any overlapped all-reduce.  2 ranks only.
static __global__ void ggml_cuda_ce_add_bf16(const nv_bfloat16 * __restrict__ a,
                                             const nv_bfloat16 * __restrict__ b,
                                             nv_bfloat16 * __restrict__ o, long n) {
    for (long i = blockIdx.x * (long) blockDim.x + threadIdx.x; i < n; i += (long) gridDim.x * blockDim.x) {
        o[i] = __float2bfloat16(__bfloat162float(a[i]) + __bfloat162float(b[i]));
    }
}

static bool ggml_backend_cuda_comm_allreduce_ce(ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    const int64_t ne = ggml_nelements(tensors[0]);
    if (ne == 0) {
        return true;
    }
    const size_t nb = comm_ctx->backends.size();
    if (nb < 2) {
        return false;
    }
    for (size_t i = 0; i < nb; ++i) {
        GGML_ASSERT(tensors[i] != nullptr && ggml_nelements(tensors[i]) == ne &&
                    ggml_is_contiguously_allocated(tensors[i]));
    }

    const int n = (int) nb;

    // Chunk c covers [off[c], off[c+1]); the first `rem` chunks take one extra element when ne is
    // not divisible by n.  Direct sends (no ring relay): for n = 3 every transfer is one hop.
    std::vector<int64_t> off(n + 1);
    for (int c = 0; c <= n; ++c) {
        off[c] = (int64_t) c * (ne / n) + std::min<int64_t>(c, ne % n);
    }
    const int64_t chunk_max = (ne + n - 1) / n;   // padded tmp region (sender-indexed)

    // ce_buf[i] : this rank's staged bf16 copy (ne elements)
    // ce_tmp[i] : n receive regions of chunk_max elements, region <sender>; used for the
    //             reduce-scatter slices and then for the all-gather of the reduced chunks.
    const size_t need = (size_t) n * (size_t) chunk_max * sizeof(nv_bfloat16);
    if (comm_ctx->ce_bytes < need) {
        // Optional scratch budget (`GGML_CUDA_AR_CE_SCRATCH_MIB`; 0 = unlimited, the default): an
        // oversized prefill degrades to the internal pipeline instead of eating the slab reserve.  CE is
        // an opt-in (+2..4 % prefill), so a graceful fallback is strictly better than an abort.
        static const size_t ce_budget = []() {
            const char * e = getenv("GGML_CUDA_AR_CE_SCRATCH_MIB");
            return e != nullptr ? (size_t) atoll(e) << 20 : (size_t) 0;
        }();
        if (comm_ctx->ce_grow_disabled || (ce_budget > 0 && need > ce_budget)) {
            // Issue #136 (item C): the scratch could not be grown (or exceeds the budget).  Return false
            // BEFORE any collective phase runs, so the dispatcher falls back cleanly to the internal
            // pipeline (`ggml_backend_cuda_comm_allreduce_tensor`).  Never a hard abort.
            if (!comm_ctx->ce_fallback_warned) {
                comm_ctx->ce_fallback_warned = true;
                GGML_LOG_WARN("%s: copy-engine all-reduce scratch %.2f MiB unavailable%s; falling back to the "
                              "internal all-reduce for the rest of this run (GGML_CUDA_AR_CE_SCRATCH_MIB=%zu)\n",
                              __func__, need / 1024.0 / 1024.0,
                              comm_ctx->ce_grow_disabled ? " (allocation failed)" : " (over the scratch budget)",
                              ce_budget >> 20);
            }
            return false;
        }
        // Grow without ever freeing mid-run: a previous call's peer copy may still be writing the old
        // scratch from the OTHER device's stream (a cross-device use-after-free), and cudaFree /
        // cudaDeviceSynchronize are not safe under graph capture.  Retire the old buffers to teardown at
        // the END, and only if every rank got all three buffers.  Issue #136 (item C): every allocation
        // goes through `ggml_cuda_device_malloc` (the fail-soft choke point) instead of a bare cudaMalloc,
        // so an OOM retires the partial attempt and returns false instead of aborting the process.
        std::vector<void *> new_buf (nb, nullptr);
        std::vector<void *> new_tmp (nb, nullptr);
        std::vector<void *> new_tmp2(nb, nullptr);
        bool ok = true;
        for (size_t i = 0; i < nb && ok; ++i) {
            const int dev = (int) comm_ctx->dev_ids[i];
            ok = ggml_cuda_device_malloc(&new_buf [i], need, dev) == cudaSuccess &&
                 ggml_cuda_device_malloc(&new_tmp [i], need, dev) == cudaSuccess &&
                 ggml_cuda_device_malloc(&new_tmp2[i], need, dev) == cudaSuccess;
            if (!ok) {
                (void) cudaGetLastError();
                comm_ctx->ce_grow_disabled = true;
                for (size_t j = 0; j <= i; ++j) {
                    const int d = (int) comm_ctx->dev_ids[j];
                    if (new_buf [j] != nullptr) comm_ctx->ce_old.push_back({ d, new_buf [j] });
                    if (new_tmp [j] != nullptr) comm_ctx->ce_old.push_back({ d, new_tmp [j] });
                    if (new_tmp2[j] != nullptr) comm_ctx->ce_old.push_back({ d, new_tmp2[j] });
                }
                if (!comm_ctx->ce_fallback_warned) {
                    comm_ctx->ce_fallback_warned = true;
                    GGML_LOG_WARN("%s: could not allocate the copy-engine all-reduce scratch (%.2f MiB on "
                                  "device %d); falling back to the internal all-reduce for the rest of this "
                                  "run\n", __func__, need / 1024.0 / 1024.0, dev);
                }
                return false;
            }
        }
        // Success: retire the old buffers (freed at teardown) and install the new ones.
        for (size_t i = 0; i < nb; ++i) {
            const int d = (int) comm_ctx->dev_ids[i];
            if (comm_ctx->ce_buf [i] != nullptr) comm_ctx->ce_old.push_back({ d, comm_ctx->ce_buf [i] });
            if (comm_ctx->ce_tmp [i] != nullptr) comm_ctx->ce_old.push_back({ d, comm_ctx->ce_tmp [i] });
            if (comm_ctx->ce_tmp2[i] != nullptr) comm_ctx->ce_old.push_back({ d, comm_ctx->ce_tmp2[i] });
            comm_ctx->ce_buf [i] = new_buf [i];
            comm_ctx->ce_tmp [i] = new_tmp [i];
            comm_ctx->ce_tmp2[i] = new_tmp2[i];
        }
        comm_ctx->ce_bytes = need;
    }

    to_bf16_cuda_t to_bf16 = ggml_get_to_bf16_cuda(GGML_TYPE_F32);
    to_fp32_cuda_t to_fp32 = ggml_get_to_fp32_cuda(GGML_TYPE_BF16);

    auto cctx = [&](int i) -> ggml_backend_cuda_context * {
        return (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
    };
    auto cdev = [&](int i) -> int { return (int) comm_ctx->dev_ids[i]; };
    auto cbuf  = [&](int i) -> char * { return (char *) comm_ctx->ce_buf[i]; };
    auto ctmp1 = [&](int i) -> char * { return (char *) comm_ctx->ce_tmp[i]; };
    auto ctmp2 = [&](int i) -> char * { return (char *) comm_ctx->ce_tmp2[i]; };
    const size_t esz = sizeof(nv_bfloat16);

    // phase 0: stage each rank's contribution as bf16
    for (int i = 0; i < n; ++i) {
        ggml_cuda_set_device(cctx(i)->device);
        if (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) {
            to_bf16(tensors[i]->data, (nv_bfloat16 *) comm_ctx->ce_buf[i], ne, cctx(i)->stream());
        } else {
            CUDA_CHECK(cudaMemsetAsync(comm_ctx->ce_buf[i], 0, need, cctx(i)->stream()));
        }
    }

    // phase 1a: dev i ships its chunk c to dev c, which owns the reduced chunk c.
    // tmp1 is reused across calls, so wait for the peer's previous phase-1b read -- that record is
    // from the previous call, so this wait does not stall.
    for (int i = 0; i < n; ++i) {
        ggml_cuda_set_device(cctx(i)->device);
        cudaStream_t s = cctx(i)->stream();
        for (int c = 0; c < n; ++c) {
            if (c != i) {
                CUDA_CHECK(cudaStreamWaitEvent(s, comm_ctx->ce_ev_done[c], 0));
            }
        }
        for (int c = 0; c < n; ++c) {
            if (c == i) {
                continue;
            }
            CUDA_CHECK(cudaMemcpyPeerAsync(ctmp1(c) + (size_t) i * chunk_max * esz, cdev(c),
                                           cbuf(i)  + (size_t) off[c] * esz,     cdev(i),
                                           (size_t) (off[c + 1] - off[c]) * esz, s));
        }
        CUDA_CHECK(cudaEventRecord(comm_ctx->ce_ev_send[i], s));
    }

    // phase 1b: reduce the received slices into the own chunk in place
    for (int i = 0; i < n; ++i) {
        ggml_cuda_set_device(cctx(i)->device);
        cudaStream_t s = cctx(i)->stream();
        for (int c = 0; c < n; ++c) {
            if (c != i) {
                CUDA_CHECK(cudaStreamWaitEvent(s, comm_ctx->ce_ev_send[c], 0));
            }
        }
        const int64_t w = off[i + 1] - off[i];
        for (int c = 0; c < n; ++c) {
            if (c == i) {
                continue;
            }
            ggml_cuda_ce_add_bf16<<<256, 256, 0, s>>>(
                (const nv_bfloat16 *) (cbuf(i)  + (size_t) off[i] * esz),
                (const nv_bfloat16 *) (ctmp1(i) + (size_t) c * chunk_max * esz),
                (nv_bfloat16 *)       (cbuf(i)  + (size_t) off[i] * esz), w);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaEventRecord(comm_ctx->ce_ev_done[i], s));
    }

    // phase 2a: all-gather -- dev i ships its reduced chunk i to every peer.
    // tmp2 is a SEPARATE buffer from tmp1, so this needs no wait on the peers' phase-1b adds: the
    // only dependency is the peer's previous phase-2b read (cross-call guard, non-stalling).
    // That removes a full barrier between the reduce-scatter and the all-gather.
    for (int i = 0; i < n; ++i) {
        ggml_cuda_set_device(cctx(i)->device);
        cudaStream_t s = cctx(i)->stream();
        for (int c = 0; c < n; ++c) {
            if (c != i) {
                CUDA_CHECK(cudaStreamWaitEvent(s, comm_ctx->ce_ev_out[c], 0));
            }
        }
        for (int p = 0; p < n; ++p) {
            if (p == i) {
                continue;
            }
            CUDA_CHECK(cudaMemcpyPeerAsync(ctmp2(p) + (size_t) i * chunk_max * esz, cdev(p),
                                           cbuf(i)   + (size_t) off[i] * esz,     cdev(i),
                                           (size_t) (off[i + 1] - off[i]) * esz, s));
        }
        CUDA_CHECK(cudaEventRecord(comm_ctx->ce_ev_recv[i], s));
    }

    // phase 2b: own reduced chunk from buf, every other chunk from tmp
    for (int i = 0; i < n; ++i) {
        ggml_cuda_set_device(cctx(i)->device);
        cudaStream_t s = cctx(i)->stream();
        for (int c = 0; c < n; ++c) {
            if (c != i) {
                CUDA_CHECK(cudaStreamWaitEvent(s, comm_ctx->ce_ev_recv[c], 0));
            }
        }
        to_fp32((const nv_bfloat16 *) (cbuf(i) + (size_t) off[i] * esz),
                (float *) tensors[i]->data + off[i], off[i + 1] - off[i], s);
        for (int c = 0; c < n; ++c) {
            if (c == i) {
                continue;
            }
            to_fp32((const nv_bfloat16 *) (ctmp2(i) + (size_t) c * chunk_max * esz),
                    (float *) tensors[i]->data + off[c], off[c + 1] - off[c], s);
        }
        CUDA_CHECK(cudaEventRecord(comm_ctx->ce_ev_out[i], s));
        CUDA_CHECK(cudaGetLastError());
    }

    return true;
}

static bool ggml_backend_cuda_comm_init_ce(ggml_backend_cuda_comm_context * ret) {
    const size_t nb = ret->backends.size();
    if (nb < 2) {
        return false;
    }
    ret->ce_buf.assign(nb, nullptr);
    ret->ce_tmp.assign(nb, nullptr);
    ret->ce_tmp2.assign(nb, nullptr);
    ret->ce_ev_send.assign(nb, nullptr);
    ret->ce_ev_done.assign(nb, nullptr);
    ret->ce_ev_recv.assign(nb, nullptr);
    ret->ce_ev_out.assign(nb, nullptr);
    for (size_t i = 0; i < nb; ++i) {
        ggml_cuda_set_device(ret->dev_ids[i]);
        // Peer access is OPT-IN (`GGML_CUDA_AR_CE_PEER=1`) since r165, and OFF is the measured default.
        //
        // A device that has peer access enabled charges the HOST submission path, not the GPU: with the
        // AR path otherwise untouched, enabling it on three R9700s costs 27B Q8_0 `-sm tensor tg128`
        // 42.25 -> 38.77 t/s, and `cudaDeviceDisablePeerAccess` gives the whole of it back (42.24).  It
        // is a per-device state (one device's pairs already cost most of it), it is invisible to a
        // per-token-synced loop (41.1 vs 41.4 -- the GPU critical path is unchanged) and to kernel
        // microbenchmarks, and it only appears once decode graphs replay, i.e. once the host has to run
        // ahead.  That is `TODO #109`.
        //
        // Nothing is lost by leaving it off: a staged `cudaMemcpyPeerAsync` is driver-managed and still
        // runs on the copy engines, and it measured the same as the direct P2P arm even at 8k-token
        // prefill (pp2048 2212 vs 2206, pp8192 2100 vs 2096 t/s on the same 3-device config), because
        // the CE slices are only `ne/n` elements and each hop is already one transfer.  Enable the knob
        // on a topology where a direct peer copy is genuinely cheaper (xGMI/NVLink-like links).
        static const bool ce_peer = []() {
            const char * e = getenv("GGML_CUDA_AR_CE_PEER");
            return e != nullptr && atoi(e) != 0;
        }();
        if (ce_peer) {
            for (size_t j = 0; j < nb; ++j) {
                if (j == i) {
                    continue;
                }
                cudaError_t e = cudaDeviceEnablePeerAccess(ret->dev_ids[j], 0);
                if (e == cudaErrorPeerAccessAlreadyEnabled) {
                    // Benign: peer access is already on (e.g. a previous comm context on this device).
                    // It is still recorded as a sticky "last error", so clear it or the next kernel
                    // launch's error check will pick it up and abort.
                    (void) cudaGetLastError();
                } else if (e != cudaSuccess) {
                    (void) cudaGetLastError();
                    return false;
                }
            }
        }
        if (cudaEventCreateWithFlags(&ret->ce_ev_send[i], cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&ret->ce_ev_done[i], cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&ret->ce_ev_recv[i], cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&ret->ce_ev_out[i],  cudaEventDisableTiming) != cudaSuccess) {
            return false;
        }
    }
    ret->try_allreduce = ggml_backend_cuda_comm_allreduce_ce;
    return true;
}

// Copy-engine first chain (the Linux/HIP default since r165): the internal host-staged pipeline for
// latency-bound small tensors plus the copy-engine (SDMA) arm for bandwidth-bound large ones, and
// **RCCL is not brought up at all**.  The per-size routing is the same one the hybrid chain uses
// (ggml_backend_cuda_comm_allreduce_tensor sends smalls to the internal pipeline and everything else
// to try_allreduce), so the decode/verify band is served by exactly the same internal kernel as
// `hybrid` -- only the prefill-sized arm changes, from RCCL's P2P kernels to our copy-engine one.
//
// Why not RCCL (TODO #109, r165): RCCL's P2P bring-up (`ncclCommInitAll` on a PCIe topology, which
// exports/imports every channel buffer as a direct peer pointer) measurably slows the HOST submission
// path -- 27B Q8_0 3x R9700 `-sm tensor tg128` 41.5 -> 38.6 t/s -- while the GPU critical path is
// unchanged (a per-token sync removes the difference: 41.1 vs 41.4).  It is the presence of the P2P
// setup, not the per-call choice of arm: routing every tensor to the internal pipeline with RCCL up
// loses identically, and `NCCL_P2P_DISABLE=1` restores the full speed.  On the measured shapes the CE
// arm is also *faster* than RCCL's large-tensor path (pp2048 2227 vs 2187 t/s on the same 3-device
// config), so the large-tensor win is kept without paying the host-side cost.
//
// `GGML_CUDA_ALLREDUCE=ce` selects this chain explicitly: internal pipeline for smalls, copy engine for
// large tensors, RCCL never brought up.  A topology where CE cannot be set up falls back to RCCL.
static bool ggml_backend_cuda_comm_init_ce_chain(ggml_backend_cuda_comm_context * ret) {
    // Latency-bound smalls: the internal pipeline, exactly as in the hybrid chain.
    const bool has_internal = ggml_backend_cuda_comm_init_internal(ret);
    if (has_internal) {
        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_internal;
    }
    if ((ret->ce_ready = ggml_backend_cuda_comm_init_ce(ret))) {
        // Large tensors now go to the copy-engine arm; smalls keep the internal pipeline.
        return true;
    }
    // The copy engine is unavailable (fewer than 2 devices, or event/scratch setup failed): fall back to
    // the RCCL large-tensor arm rather than losing it (only NCCL -- the internal pipeline is already up,
    // and re-running it would leak a pipeline).
    GGML_LOG_WARN("copy-engine all-reduce unavailable; using the RCCL/NCCL large-tensor arm instead\n");
    if ((ret->nccl_ready = ggml_backend_cuda_comm_init_nccl(ret))) {
        return true;
    }
    if (!has_internal) {
        ret->try_allreduce = nullptr;
    }
    return ret->try_allreduce != nullptr;
}

// Set once the host-expert machinery knows this model offloads experts to host memory (the slab block's
// `ggml_cuda_slab_arm()` sets it; a dense model never does).  A plain flag rather than a call so this
// block does not depend on a symbol a later block introduces -- every block of the set must build on its
// own (scripts/gate-per-block-build.sh).  When it is false the large all-reduce uses the copy-engine arm
// and peer access is never enabled, which is what the dense shapes want (see below); when it is true the
// large arm is RCCL's ring, which is what a PCIe-saturated host-expert prefill wants.
bool g_comm_host_experts = false;

// The Linux/HIP default chain (r165).  Both large-tensor arms are brought up eagerly and the arm is
// chosen once, at the FIRST large all-reduce, from the only signal that separates the two regimes:
// whether the model offloads experts to host memory (g_comm_host_experts, set by the slab block).
//
//   * host experts in play -> RCCL's ring.  A host-offloaded prefill is PCIe-saturated by the expert
//     streaming and RCCL is far ahead there, at every offload depth we measured and at both ubatch
//     widths: Flash-Next IQ4_NL 2 GPU `-ncmoe 20` pp2048 809 vs 604 t/s (CE), `-ncmoe 32` 555 vs 414,
//     `-b/-ub 4096 -ncmoe 20` pp4096 2901 vs 2344; 35B-A3B 2 GPU `-ncmoe 40` 1257 vs 1257.
//   * no host experts -> the copy-engine arm, which is the faster one on every dense shape measured
//     (27B Q8_0 3 GPU pp2048 2178 vs 2018, 2 GPU 2245 vs 2091; Gemma4 31B 3 GPU 1998 vs 1944) and,
//     because peer access is then never enabled, it is also the only configuration that escapes the
//     TODO #109 host-submission cost (dense decode 42.2 vs 38.6 t/s).
//
// RCCL is brought up with peer access disabled (`NCCL_P2P_DISABLE`, only when the user has not set it).
// Its P2P bring-up is the host-submission cost, and it is not reversible: disabling peer access after
// `ncclCommInitAll` leaves the cost in place (measured: every `cudaDeviceDisablePeerAccess` returned
// `no error` and the dense decode stayed at 37.9 t/s).  The cost of that on the host-expert prefill is
// shape-dependent and small: RCCL over SHM measures the same as over P2P on Flash-Next GSQ `-ncmoe 48`
// (551.8 vs 552.7 t/s) and on 35B-A3B `-ncmoe 40` (1257 vs 1232), and 8 % behind on Flash-Next IQ4_NL
// `-ncmoe 20` (747 vs 809).  `GGML_CUDA_ALLREDUCE=hybrid` keeps the old P2P chain for a topology with
// genuinely cheap peer links (xGMI/scale-up).
static bool ggml_backend_cuda_comm_init_auto(ggml_backend_cuda_comm_context * ret) {
    const bool has_internal = ggml_backend_cuda_comm_init_internal(ret);
    if (has_internal) {
        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_internal;
    }
    // RCCL must not enable peer access: its P2P bring-up is what costs the host submission path
    // (TODO #109) and the cost SURVIVES a later `cudaDeviceDisablePeerAccess` (the mappings RCCL
    // created stay), so it has to be prevented at init.  Only if the user has not set the knob.
    static const bool p2p_env = []() {
        // A host-side call in a delivery-added path must build on Windows too (clang-cl + the MSVC STL
        // have no `setenv`), and it must not overwrite a value the user set.
#if defined(_WIN32)
        if (getenv("NCCL_P2P_DISABLE") == nullptr) {
            _putenv_s("NCCL_P2P_DISABLE", "1");
        }
#else
        setenv("NCCL_P2P_DISABLE", "1", 0);
#endif
        return true;
    }();
    (void) p2p_env;
    ret->nccl_ready = ggml_backend_cuda_comm_init_nccl(ret);
    if (ret->nccl_ready) {
        ret->nccl_arm = ret->try_allreduce;
    }
    ret->ce_ready   = ggml_backend_cuda_comm_init_ce(ret);
    // A valid arm has to exist before the first large call arrives (the CE arm if it came up, otherwise
    // whatever the two inits left as `try_allreduce`); the selection below overrides it once.
    if (ret->ce_ready) {
        ret->try_allreduce = ggml_backend_cuda_comm_allreduce_ce;
    }
    if (!ret->ce_ready && !ret->nccl_ready && !has_internal) {
        ret->try_allreduce = nullptr;
        return false;
    }
    ret->auto_select = true;
    return true;
}

// Choose the large-tensor arm once (see ggml_backend_cuda_comm_init_auto).
static void ggml_backend_cuda_comm_select_arm(ggml_backend_cuda_comm_context * comm_ctx) {
    comm_ctx->auto_select = false;
    const bool host_experts = g_comm_host_experts;

    if (host_experts && comm_ctx->nccl_ready) {
        GGML_LOG_INFO("all-reduce: host-resident experts -- large tensors use the RCCL ring "
                      "(GGML_CUDA_ALLREDUCE=ce forces the copy-engine arm)\n");
        comm_ctx->try_allreduce = comm_ctx->nccl_arm;
        return;
    }

    // Dense (or the slab declined): the copy-engine arm.  Peer access is not enabled at all in this
    // chain (see the init), and CE measures the same speed with a staged peer copy as with a direct one
    // (pp8192 2100 vs 2096 t/s), so nothing is given up.
    if (comm_ctx->ce_ready) {
        comm_ctx->try_allreduce = ggml_backend_cuda_comm_allreduce_ce;
        GGML_LOG_INFO("all-reduce: no host-resident experts -- large tensors use the copy-engine arm "
                      "(GGML_CUDA_ALLREDUCE=hybrid restores the RCCL P2P arm)\n");
    }
}

// Hybrid: NCCL/RCCL (P2P, BF16 round-trip) for bandwidth-bound large tensors,
// plus the internal host-staged pipeline (low per-call latency) for
// latency-bound small tensors.  The per-size routing happens in
// ggml_backend_cuda_comm_allreduce_tensor: small tensors go to the internal
// pipeline directly; everything else falls through to try_allreduce, which we
// set to NCCL when available.  `GGML_CUDA_ALLREDUCE=hybrid` still selects this
// chain; the Linux/HIP default is the copy-engine chain above (r165).
static void ggml_backend_cuda_comm_init_hybrid(ggml_backend_cuda_comm_context * ret) {
    // Bring the internal pipeline up first -- it serves every decode-sized tensor directly.
    const bool has_internal = ggml_backend_cuda_comm_init_internal(ret);
    if (has_internal) {
        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_internal;
    }
    // Bring NCCL up EAGERLY.  Deferring `ncclCommInitAll` to the first large tensor (the old
    // `nccl_lazy` path) runs it mid-forward, and RCCL's init side effects perturb the layout/state:
    // the `-sm tensor -ncmoe` split compute flips a near-tie and the field is no longer MTP-pure
    // (`none` != `draft-mtp n3`) even with a fully-resident expert cache.  Eager init is
    // bit-identical to the pure-RCCL path; the small-tensor arm is unchanged (the dispatch still
    // routes smalls to the internal pipeline).  The cost is RCCL's presence, which r34 measured as a
    // decode-only slowdown of the internal pipeline (the large-tensor path is bandwidth-bound and
    // unaffected); any run with a prefill inits RCCL during the prefill either way.
    if (!ggml_backend_cuda_comm_init_nccl(ret)) {
        // No NCCL -- the large arm falls through to the internal pipeline when one exists,
        // otherwise the meta backend's butterfly handles it.
        if (!has_internal) {
            ret->try_allreduce = nullptr;
        }
    }
}

// Top-level init.  Picks a comm setup based on GGML_CUDA_ALLREDUCE (or the
// platform default) and lets the chain handle any fallback.  Unrecognised env
// values warn and fall through to the platform default.
static void * ggml_backend_cuda_comm_init(ggml_backend_t * backends, size_t n_backends) {
    for (size_t i = 0; i < n_backends; i++) {
        if (!ggml_backend_is_cuda(backends[i])) {
            return nullptr;
        }
    }

    auto * ret = new ggml_backend_cuda_comm_context;
    ret->backends.assign(backends, backends + n_backends);
    ret->dev_ids.reserve(n_backends);
    for (size_t i = 0; i < n_backends; i++) {
        ret->dev_ids.push_back(static_cast<ggml_backend_cuda_context *>(backends[i]->context)->device);
    }

    const char * env = getenv("GGML_CUDA_ALLREDUCE");
    bool ok = false;
    if (!env) {
        // Platform default: on Linux/HIP the dual-arm chain (r165, TODO #109): host experts -> RCCL ring,
        // dense -> copy-engine, peer access never enabled by us.  On CUDA Linux the hybrid chain (NCCL
        // for large, internal for small); otherwise (generally Windows) internal only.
#if defined(__linux__) && defined(GGML_USE_HIP)
        ok = ggml_backend_cuda_comm_init_auto(ret);
#elif defined(__linux__)
        ggml_backend_cuda_comm_init_hybrid(ret);
        ok = ret->try_allreduce != nullptr;
#else
        ok = ggml_backend_cuda_comm_init_internal(ret);
#endif // defined(__linux__)
    } else {
        std::string env_str(env);
        if (env_str == "hybrid") {
            ggml_backend_cuda_comm_init_hybrid(ret);
            ok = ret->try_allreduce != nullptr;
        } else if (env_str == "nccl") {
            ok = ggml_backend_cuda_comm_init_nccl(ret) || ggml_backend_cuda_comm_init_internal(ret);
        } else if (env_str == "internal") {
            ok = ggml_backend_cuda_comm_init_internal(ret);
        } else if (env_str == "ce") {
            // The copy-engine chain: internal pipeline for latency-bound small tensors, CE for
            // bandwidth-bound large ones, no RCCL (see ggml_backend_cuda_comm_init_ce_chain).  The
            // Linux/HIP default (r165) is the dual-arm chain that picks between this and RCCL per model;
            // `ce` pins this half.  If CE cannot be set up the chain keeps the RCCL large-tensor arm, so
            // a CE failure is still a no-op downgrade.
            ok = ggml_backend_cuda_comm_init_ce_chain(ret);
        } else if (env_str == "none") {
            ok = false;
        } else {
            GGML_LOG_WARN("unknown GGML_CUDA_ALLREDUCE value: %s\n", env);
            ok = false;
        }
    }

    if (!ok) {
        ggml_backend_cuda_comm_init_none(ret);
    }

    return ret;
}

// Top-level dispatch -- calls the function pointer chosen by comm_init, with
// one hybrid rule: when the internal host-staged pipeline is available,
// latency-bound small tensors (token generation) go through it directly, and
// bandwidth-bound large tensors (prefill) fall through to NCCL/RCCL P2P.
// Returns false to let the meta-backend's butterfly run.
static bool ggml_backend_cuda_comm_allreduce_tensor(void * comm_ctx_v, struct ggml_tensor ** tensors) {
    if (comm_ctx_v == nullptr) {
        return false;
    }
    auto * comm_ctx = static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
    const int64_t ne = ggml_nelements(tensors[0]);
    const size_t n_backends = comm_ctx->backends.size();
    if (comm_ctx->nccl_failed) {
        // NCCL is dead for this run: use the internal pipeline for all sizes
        // it can serve, otherwise let the butterfly handle this call.
        return comm_ctx->ar_pipeline != nullptr
               ? ggml_backend_cuda_comm_try_allreduce_internal(comm_ctx, tensors)
               : false;
    }
    if (comm_ctx->ar_pipeline != nullptr && ggml_backend_cuda_comm_is_small(ne, n_backends)) {
        return ggml_backend_cuda_comm_try_allreduce_internal(comm_ctx, tensors);
    }
    if (comm_ctx->auto_select) {
        // r165 default chain: the first large tensor chooses the large arm (host experts -> RCCL,
        // otherwise the copy-engine arm).  See ggml_backend_cuda_comm_select_arm.
        ggml_backend_cuda_comm_select_arm(comm_ctx);
    }
    // First-call NCCL failure failover (issue #86): the selected function may
    // fail (NCCL/RCCL refusing to dispatch, e.g. hipErrorIllegalState on a PCIe
    // root port without AtomicOp completer support) and re-route itself to the
    // internal pipeline.  When that happens, serve this same call through the
    // internal pipeline as well instead of returning false and letting the meta
    // backend's butterfly run -- on a RCCL-broken host the butterfly is what
    // hangs.
    auto fn = comm_ctx->try_allreduce;
    const bool ok = fn(comm_ctx, tensors);
    if (!ok && comm_ctx->ar_pipeline != nullptr &&
        fn != ggml_backend_cuda_comm_try_allreduce_internal) {
        return ggml_backend_cuda_comm_try_allreduce_internal(comm_ctx, tensors);
    }
    return ok;
}

// host buffer type

static const char * ggml_backend_cuda_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_CUDA_NAME "_Host";

    GGML_UNUSED(buft);
}

static bool ggml_backend_buft_is_cuda_host(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
}

static void ggml_backend_cuda_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    CUDA_CHECK(cudaFreeHost(buffer->context));
}

static void * ggml_cuda_host_malloc(size_t size) {
    if (ggml_cuda_pinned_memory_disabled()) {
        return nullptr;
    }

    void * ptr = nullptr;
    cudaError_t err = cudaMallocHost((void **) &ptr, size);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
        GGML_LOG_DEBUG("%s: failed to allocate %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return nullptr;
    }

    return ptr;
}

static ggml_backend_buffer_t ggml_backend_cuda_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * ptr = ggml_cuda_host_malloc(size);

    if (ptr == nullptr) {
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_cuda_host_buffer_free_buffer;

    return buffer;
}

static size_t ggml_backend_cuda_host_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    // very large pinned allocations can fail (e.g. on Windows) and fall back to pageable memory, smaller ones succeed
    return 4ull*1024*1024*1024;
}

// Per-device pinned host buffer types.  A buffer type's `device` is the device the scheduler places
// the consumers of its tensors on, so a single device-0 host buffer type routes every host-resident
// MoE expert (MUL_MAT_ID) op to device 0 under `-sm layer`, leaving the other GPUs idle on the expert
// half.  The pinned allocation itself is device-agnostic (UVA), so only `device` differs; the public
// `ggml_backend_cuda_host_buffer_type()` keeps returning device 0's for callers that mean "the host
// buffer type" (and for API compatibility).
static ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type_dev(int device) {
    GGML_ASSERT(device >= 0 && device < GGML_CUDA_MAX_DEVICES);

    static struct ggml_backend_buffer_type bufts[GGML_CUDA_MAX_DEVICES];
    static bool init[GGML_CUDA_MAX_DEVICES] = {};

    if (!init[device]) {
        bufts[device] = {
            /* .iface    = */ {
                /* .get_name            = */ ggml_backend_cuda_host_buffer_type_name,
                /* .alloc_buffer        = */ ggml_backend_cuda_host_buffer_type_alloc_buffer,
                /* .alloc_buffer_n      = */ NULL,
                /* .get_alignment       = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
                /* .get_max_size        = */ ggml_backend_cuda_host_buffer_type_get_max_size,
                /* .get_alloc_size      = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
                /* .get_alloc_size_n    = */ NULL,
                /* .is_host             = */ ggml_backend_cpu_buffer_type()->iface.is_host,
            },
            /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), device),
            /* .context  = */ nullptr,
        };
        init[device] = true;
    }

    return &bufts[device];
}

ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type() {
    return ggml_backend_cuda_host_buffer_type_dev(0);
}

//static bool ggml_backend_buffer_is_cuda_host(ggml_backend_buffer_t buffer) {
//    return buffer->buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
//}

/// kernels

typedef void (*ggml_cuda_op_mul_mat_t)(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

static __global__ void k_compute_batched_ptrs(
        const void * src0_as_f16, const void * src1_as_f16, char * dst,
        const void ** ptrs_src, void ** ptrs_dst,
        int64_t ne12, int64_t ne13,
        int64_t ne23,
        size_t  nb02, size_t  nb03,
        size_t  nb12, size_t  nb13,
        size_t  nbd2, size_t  nbd3,
        int64_t r2,   int64_t r3) {
    const int64_t i13 = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t i12 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i13 >= ne13 || i12 >= ne12) {
        return;
    }

    const int64_t i03 = i13 / r3;
    const int64_t i02 = i12 / r2;

    ptrs_src[0*ne23 + i12 + i13*ne12] = (const char *) src0_as_f16 + i02*nb02 + i03*nb03;
    ptrs_src[1*ne23 + i12 + i13*ne12] = (const char *) src1_as_f16 + i12*nb12 + i13*nb13;
    ptrs_dst[0*ne23 + i12 + i13*ne12] = (      char *)         dst + i12*nbd2 + i13*nbd3;
}

// Type traits for mapping ggml types to CUDA/cuBLAS types
template<ggml_type T>
struct batched_mul_mat_traits;

template<>
struct batched_mul_mat_traits<GGML_TYPE_F32> {
    using cuda_type = float;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_32F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F32;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_fp32_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_fp32_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_BF16> {
    using cuda_type = nv_bfloat16;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_16BF;
    static inline const ggml_type ggml_type_val = GGML_TYPE_BF16;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_bf16_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_bf16_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_F16> {
    using cuda_type = half;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
    static inline const cudaDataType_t data_type = CUDA_R_16F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F16;
    static inline const half alpha = 1.0;
    static inline const half beta = 0.0;
    static inline const void* get_alpha() { static const half val = alpha; return &val; }
    static inline const void* get_beta() { static const half val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_fp16_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_fp16_nc_cuda(src_type); }
};

// TEMP INSTRUMENT (wip/tensor-split-expert-split): per-call host cost + node count of the CUDA
// (exp6/exp7) per-call host cost + node count of the CUDA backend graph_compute,
// the per-op / per-matmul-branch / per-cublas-section / pool breakdown.
static int64_t g_cb_us[6] = {0}, g_cb_n[6] = {0};
static char    g_cb_worst[160] = {0}; static int64_t g_cb_worst_us = 0;
// backend's graph_compute.  Under the meta backend this is called once per (subgraph, device), i.e.
// ~648 times per pass with op-offload; with a single device it is called once per pass.
static int64_t g_cgc_us = 0, g_cgc_nodes = 0, g_cgc_calls = 0;
static bool    g_cgc_reg = false;
// finer breakdown (wip/tensor-split-expert-split, exp6): where the per-node host cost goes.
static const bool g_cgc_on       = getenv("GGML_CUDA_GCDBG") != nullptr;
static const bool g_op_timing_on = getenv("GGML_CUDA_OP_TIMING") != nullptr;
static const bool g_stream_dbg_on = getenv("GGML_STREAMDBG") != nullptr;
static const int  g_mmb_mark_log = getenv("GGML_CUDA_MMB_MARK_LOG") ? atoi(getenv("GGML_CUDA_MMB_MARK_LOG")) : 0;
// Meta/op-offload A/B switch.  This sits on the per-copy staging path (thousands of calls per pass
// under -sm tensor), so it is resolved once instead of per call.
static const bool g_cuda_splice_env_on = [] {
    const char * e = getenv("GGML_CUDA_SPLICE_GATHER");
    return e == nullptr || atoi(e) != 0;
}();
static int64_t g_gc_pre_us = 0;    // graph_compute before evaluate_and_capture (q8_1 clear, graph-key probe)
static int64_t g_ev_pre_us = 0;    // evaluate_and_capture before the main node loop (concurrent events, alloc)
static int64_t g_ev_rest_us = 0;   // the main node loop + evaluate_and_capture's tail
static int64_t g_fuse_us = 0;      // inside ggml_cuda_try_fuse
static int64_t g_fwd_us = 0;       // inside ggml_cuda_compute_forward
static int64_t g_fuse_calls = 0, g_fwd_calls = 0, g_loop_start = 0;
// per-op-type attribution of the forward cost
// MUL_MAT branch attribution: which branch of ggml_cuda_mul_mat, and how long its call takes.
// The remainder of the op's time is its own predicate chain (should_use_mmvf/mmf/mmvq/mmq, mmb).
static int64_t g_mm_us[8] = {0};
static int64_t g_mm_n[8]  = {0};
#define MMB_T(call, k) do { const int64_t t_ = g_cgc_on ? ggml_time_us() : 0; call; \
    if (t_) { g_mm_us[k] += ggml_time_us() - t_; g_mm_n[k]++; } } while (0)
static int64_t g_opus[GGML_OP_COUNT] = {0};
static int64_t g_opn[GGML_OP_COUNT] = {0};
static void g_cgc_dump(void) {
    fprintf(stderr, "CUDAGC calls=%lld nodes=%lld host_total=%.1fms per_node=%.1fus per_call=%.1fus\n",
        (long long) g_cgc_calls, (long long) g_cgc_nodes, g_cgc_us/1000.0,
        g_cgc_nodes ? double(g_cgc_us)/double(g_cgc_nodes) : 0.0,
        g_cgc_calls ? double(g_cgc_us)/double(g_cgc_calls) : 0.0);
    {
        int idx[16]; double best[16];
        for (int k = 0; k < 16; ++k) { idx[k] = -1; best[k] = -1.0; }
        for (int o = 0; o < GGML_OP_COUNT; ++o) {
            if (g_opn[o] == 0) continue;
            double v = g_opus[o]/1000.0;
            for (int k = 0; k < 16; ++k) {
                if (v > best[k]) {
                    for (int m = 15; m > k; --m) { best[m] = best[m-1]; idx[m] = idx[m-1]; }
                    best[k] = v; idx[k] = o; break;
                }
            }
        }
        fprintf(stderr, "CUDAGC_OPS");
        for (int k = 0; k < 16 && idx[k] >= 0; ++k) {
            fprintf(stderr, " %s=%.1fms/%lld(%.1fus)", ggml_op_name((enum ggml_op) idx[k]),
                    best[k], (long long) g_opn[idx[k]], best[k]*1000.0/(double) g_opn[idx[k]]);
        }
        fprintf(stderr, "\n");
    }
    {
        static const char * cbn[5] = {"src0alloc+dequant","src1alloc+cvt","dst_temp","gemm","to_fp32"};
        fprintf(stderr, "CUDAGC_CUBLAS");
        for (int k = 0; k < 5; ++k) if (g_cb_n[k]) fprintf(stderr, " %s=%.1fms/%lld", cbn[k], g_cb_us[k]/1000.0, (long long) g_cb_n[k]);
        fprintf(stderr, " worst=%.1fms(%s)\n", g_cb_worst_us/1000.0, g_cb_worst);
        fprintf(stderr, "CUDAGC_POOL malloc=%.1fms/%lld free=%.1fms/%lld oom=%lld\n",
            g_pool_us/1000.0, (long long) g_pool_n, g_pool_free_us/1000.0, (long long) g_pool_free_n, (long long) g_pool_oom_n);
        static const char * nm[8] = {"mmb","mmvf","mmvfT","mmf","mmvq","mmq","cublas","fwht"};
        fprintf(stderr, "CUDAGC_MATMUL");
        for (int k = 0; k < 8; ++k) {
            if (g_mm_n[k]) fprintf(stderr, " %s=%.1fms/%lld(%.1fus)", nm[k], g_mm_us[k]/1000.0,
                (long long) g_mm_n[k], g_mm_us[k]/1000.0/g_mm_n[k]);
        }
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "CUDAGC_BREAK pre=%.1fms ev_pre=%.1fms ev_rest=%.1fms | fuse=%.1fms/%lld fwd=%.1fms/%lld\n",
        g_gc_pre_us/1000.0, g_ev_pre_us/1000.0, g_ev_rest_us/1000.0,
        g_fuse_us/1000.0, (long long) g_fuse_calls, g_fwd_us/1000.0, (long long) g_fwd_calls);
}
struct cuda_gc_timer {
    int64_t t0; ggml_cgraph * cg;
    cuda_gc_timer(ggml_cgraph * c) : t0(g_cgc_on ? ggml_time_us() : 0), cg(c) {
        if (t0 && !g_cgc_reg) { g_cgc_reg = true; atexit(g_cgc_dump); }
    }
    ~cuda_gc_timer() { if (t0) { g_cgc_us += ggml_time_us() - t0; g_cgc_nodes += cg->n_nodes; g_cgc_calls += 1; } }
};

// GGML_CUDA_BF16_SRC1_CACHE=0 disables the per-graph BF16 src1 reuse (bf16_src1_cache in common.cuh).

template<ggml_type compute_type>

static void ggml_cuda_mul_mat_cublas_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    using traits = batched_mul_mat_traits<compute_type>;
    using cuda_t = typename traits::cuda_type;

    // F32 chunks can have padding between rows to preserve the original destination stride.
    GGML_ASSERT(ggml_is_contiguous(dst) ||
        (compute_type == GGML_TYPE_F32 && ggml_is_contiguous_1(dst)));

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t ldc = nb1 / sizeof(float);
    const int64_t ne_dst = ggml_nelements(dst);
    cudaStream_t main_stream = ctx.stream();
    cublasHandle_t cublas_h = ctx.cublas_handle();

    const size_t src0_ts = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == src0_ts);
    int64_t s01 = nb01 / src0_ts;
    int64_t s02 = nb02 / src0_ts;
    int64_t s03 = nb03 / src0_ts;

    const size_t src1_ts = ggml_type_size(src1->type);
    GGML_ASSERT(nb10 == src1_ts);
    int64_t s11 = nb11 / src1_ts;
    int64_t s12 = nb12 / src1_ts;
    int64_t s13 = nb13 / src1_ts;

    float * dst_ddf = (float *) dst->data;

    const cuda_t * src0_ptr = nullptr;
    const cuda_t * src1_ptr = nullptr;

    ggml_cuda_pool_alloc<cuda_t> src0_alloc(ctx.pool());
    ggml_cuda_pool_alloc<cuda_t> src1_alloc(ctx.pool());

    bool is_src0_cont_2 = ggml_is_contiguous_2(src0);
    bool is_src1_cont_2 = ggml_is_contiguous_2(src1);

    const int64_t t_cb0 = g_cgc_on ? ggml_time_us() : 0;
    if (src0->type == compute_type) {
        src0_ptr = (const cuda_t *) src0->data;
    } else {
        src0_alloc.alloc(ggml_nelements(src0));

        if (ggml_is_contiguously_allocated(src0)) {
            const auto convert_func = traits::convert(src0->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src0->data, src0_alloc.get(), ggml_nelements(src0), main_stream);
            const size_t src0_bs = ggml_blck_size(src0->type);
            s01 *= src0_bs;
            s02 *= src0_bs;
            s03 *= src0_bs;
        } else {
            const auto convert_func = traits::convert_nc(src0->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src0->data, src0_alloc.get(), ne00, ne01, ne02, ne03, s01, s02, s03, main_stream);
            s01 = ne00;
            s02 = ne01*s01;
            s03 = ne02*s02;
            is_src0_cont_2 = true;
        }
        src0_ptr = src0_alloc.get();
    }
    if (t_cb0) { const int64_t d = ggml_time_us() - t_cb0; g_cb_us[0] += d; g_cb_n[0]++;
                 if (d > g_cb_worst_us) { g_cb_worst_us = d; snprintf(g_cb_worst, sizeof(g_cb_worst), "src0=%s n=%lld type=%s", src0->name, (long long) ggml_nelements(src0), ggml_type_name(src0->type)); } }

    const int64_t t_cb1 = g_cgc_on ? ggml_time_us() : 0;
    // BF16 GEMMs of a multi-token graph that share an F32 src1 convert it once, and a fused producer may have
    // written the BF16 copy already (see bf16_src1_cache).  Keyed by the view root, so a reshape of a cached
    // tensor (hc_norm is a reshape of the combine+norm output) finds it.
    bool src1_bf16_cached = false;
    if constexpr (compute_type == GGML_TYPE_BF16) {
        const ggml_tensor * root = src1;
        while (root->view_src) {
            root = root->view_src;
        }
        const int64_t n = ggml_nelements(src1);
        if (ggml_cuda_bf16_src1_cache_on() && src1->type == GGML_TYPE_F32 && ggml_is_contiguously_allocated(src1) &&
                ne11*ne12*ne13 >= 64 && src1->data == root->data && ggml_nelements(root) == n) {
            const auto & c = ctx.bf16_src1_cache;
            if (c.buf && c.src1 == root && c.data == root->data && c.stream_no == ctx.curr_stream_no && c.n == n) {
                src1_ptr = (const cuda_t *) c.buf->get();
            } else {
                nv_bfloat16 * buf = ctx.bf16_src1_cache_put(root, n);
                const auto convert_func = traits::convert(src1->type);
                GGML_ASSERT(convert_func != nullptr);
                convert_func(src1->data, buf, n, main_stream);
                src1_ptr = (const cuda_t *) buf;
            }
            src1_bf16_cached = true;
        }
    }
    if (src1_bf16_cached) {
        // F32 -> BF16 keeps the element strides (block size 1)
    } else if (src1->type == compute_type) {
        src1_ptr = (const cuda_t *) src1->data;
    } else {
        src1_alloc.alloc(ggml_nelements(src1));

        if (ggml_is_contiguously_allocated(src1)) {
            const auto convert_func = traits::convert(src1->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src1->data, src1_alloc.get(), ggml_nelements(src1), main_stream);
            const size_t src1_bs = ggml_blck_size(src1->type);
            s11 *= src1_bs;
            s12 *= src1_bs;
            s13 *= src1_bs;
        } else {
            const auto convert_func = traits::convert_nc(src1->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src1->data, src1_alloc.get(), ne10, ne11, ne12, ne13, s11, s12, s13, main_stream);
            s11 = ne10;
            s12 = ne11*s11;
            s13 = ne12*s12;
            is_src1_cont_2 = true;
        }
        src1_ptr = src1_alloc.get();
    }
    if (t_cb1) { g_cb_us[1] += ggml_time_us() - t_cb1; g_cb_n[1]++; }

    const int64_t t_cb2 = g_cgc_on ? ggml_time_us() : 0;
    ggml_cuda_pool_alloc<cuda_t> dst_temp(ctx.pool());
    char * dst_ptr;
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    cublasComputeType_t cu_compute_type = traits::compute_type;
    cudaDataType_t cu_data_type = traits::data_type;
    cudaDataType_t cu_data_type_a = traits::data_type;
    cudaDataType_t cu_data_type_b = traits::data_type;
    const void * alpha = traits::get_alpha();
    const void * beta = traits::get_beta();

    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    bool prefer_f32_output = false;
    if (compute_type == GGML_TYPE_F16) {
        prefer_f32_output = cc == GGML_CUDA_CC_VOLTA || GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_CDNA(cc);
    } else if (compute_type == GGML_TYPE_BF16) {
        prefer_f32_output = !GGML_CUDA_CC_IS_RDNA3(cc) && !GGML_CUDA_CC_IS_CDNA(cc);
    }

    if (prefer_f32_output) {
        dst_ptr = (char *) dst_ddf;
        cu_compute_type = batched_mul_mat_traits<GGML_TYPE_F32>::compute_type;
        cu_data_type = batched_mul_mat_traits<GGML_TYPE_F32>::data_type;
        alpha = batched_mul_mat_traits<GGML_TYPE_F32>::get_alpha();
        beta = batched_mul_mat_traits<GGML_TYPE_F32>::get_beta();
    } else {
        if constexpr (compute_type == GGML_TYPE_F32) {
            dst_ptr = (char *) dst_ddf;  // Direct F32 output
        } else {
            dst_ptr = (char *) dst_temp.alloc(ne_dst);
            nbd2 /= sizeof(float) / sizeof(cuda_t);
            nbd3 /= sizeof(float) / sizeof(cuda_t);
        }
    }

    if (t_cb2) { g_cb_us[2] += ggml_time_us() - t_cb2; g_cb_n[2]++; }

    const int64_t t_cb3 = g_cgc_on ? ggml_time_us() : 0;
    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

    // Theoretically cublasGemmStridedBatchedEx would always work, even for a single matrix.
    // However, for some old NVIDIA and AMD GPUs the strided/Ex GEMM is much slower,
    //     probably because the internal kernel selection logic is suboptimal.
    if (compute_type == GGML_TYPE_F32 && ne12 == 1 && ne13 == 1) {
#if defined(GGML_USE_HIP)
        // block 06: the F32 skinny prefill GEMM.  Taking this shape out of rocBLAS/hipBLASLt is what makes
        // the prefill logits reproducible on 35B-A3B and gemma4 (GREEDY-PURITY.md §44/§47), and it is
        // faster, because the heuristic's solution for these shapes is the slowest in the pool.  The
        // guards are the layouts the kernel assumes; anything else keeps the rocBLAS call below.
        if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
                s01 == ne10 && s11 == ne10 && dst->nb[0] == sizeof(float) &&
                ggml_cuda_mmf32_supported(cc, ne01, ne11, ne10, ldc)) {
            ggml_cuda_mul_mat_f32_skinny(ctx, ne01, ne11, ne10, ldc,
                                         (const float *) src0_ptr, (const float *) src1_ptr, (float *) dst_ptr);
        } else
#endif
        CUBLAS_CHECK(
            cublasSgemm(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                    ne01, ne11, ne10,
                    (const float *) alpha, (const float *) src0_ptr, s01,
                                           (const float *) src1_ptr, s11,
                    (const float *) beta,  (float       *)  dst_ptr, ldc));
    } else if (ne12 == 1 && ne13 == 1) {
        CUBLAS_CHECK(
            cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                    ne01, ne11, ne10,
                    alpha, src0_ptr, cu_data_type_a, s01,
                           src1_ptr, cu_data_type_b, s11,
                    beta,   dst_ptr, cu_data_type,   ldc,
                    cu_compute_type,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else if (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
        // with a [0, 2, 1, 3] perm. and ne02==1 the matrix strides need to be determined from dim 3:
        const int64_t sma = ne02 == 1 ? s03 : s02;
        const int64_t smb = ne12 == 1 ? s13 : s12;

        // there is no broadcast and src0, src1 are contiguous across dims 2, 3
        // use cublasGemmStridedBatchedEx
        CUBLAS_CHECK(
        cublasGemmStridedBatchedEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, src0_ptr, cu_data_type_a, s01, sma,     // strideA
                       src1_ptr, cu_data_type_b, s11, smb,     // strideB
                beta,   dst_ptr, cu_data_type,   ldc, ne1*ldc, // strideC
                ne12*ne13,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
        // use cublasGemmBatchedEx
        const int64_t ne23 = ne12*ne13;

        ggml_cuda_pool_alloc<const void *> ptrs_src(ctx.pool(), 2*ne23);
        ggml_cuda_pool_alloc<      void *> ptrs_dst(ctx.pool(), 1*ne23);

        const size_t src_type_size = sizeof(cuda_t);

        const int threads_x = 16;
        const int threads_y = 16;
        const dim3 block_dims(threads_x, threads_y);

        const dim3 grid_dims(
            (ne13 + threads_x - 1) / threads_x,
            (ne12 + threads_y - 1) / threads_y
        );
        k_compute_batched_ptrs<<<grid_dims, block_dims, 0, main_stream>>>(
                src0_ptr, src1_ptr, dst_ptr,
                ptrs_src.get(), ptrs_dst.get(),
                ne12, ne13,
                ne23,
                s02*src_type_size, s03*src_type_size,
                s12*src_type_size, s13*src_type_size,
                nbd2, nbd3,
                r2, r3);

        CUDA_CHECK(cudaGetLastError());

        CUBLAS_CHECK(
        cublasGemmBatchedEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, (const void **) (ptrs_src.get() + 0*ne23), cu_data_type_a, s01,
                       (const void **) (ptrs_src.get() + 1*ne23), cu_data_type_b, s11,
                beta,  (      void **) (ptrs_dst.get() + 0*ne23), cu_data_type,   ldc,
                ne23,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    if (t_cb3) { g_cb_us[3] += ggml_time_us() - t_cb3; g_cb_n[3]++; }

    const int64_t t_cb5 = g_cgc_on ? ggml_time_us() : 0;
    // Convert output back to F32 if needed
    if (cu_data_type != CUDA_R_32F) {
        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(traits::ggml_type_val);
        to_fp32_cuda(dst_temp.get(), dst_ddf, ne_dst, main_stream);
    }
    if (t_cb5) { g_cb_us[4] += ggml_time_us() - t_cb5; g_cb_n[4]++; }
}

static void ggml_cuda_mul_mat_cublas(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const ggml_prec prec = (ggml_prec) ggml_get_op_params_i32(dst, 0);
    ggml_type compute_type = src0->type;
    if (ggml_is_quantized(compute_type)) {
        compute_type = fast_fp16_hardware_available(cc) ? GGML_TYPE_F16 : GGML_TYPE_F32;
    } else if (compute_type == GGML_TYPE_F16 && !fast_fp16_hardware_available(cc)) {
        compute_type = GGML_TYPE_F32;
    } else if (compute_type == GGML_TYPE_BF16 && !fast_bf16_hardware_available(cc)) {
        if (GGML_CUDA_CC_IS_AMD(cc) && src1->ne[1] > 32) {
            compute_type = GGML_TYPE_F32;
        }
        if (GGML_CUDA_CC_IS_NVIDIA(cc) && src1->ne[1] > (cc >= GGML_CUDA_CC_VOLTA ? 8 : 128)) {
            compute_type = GGML_TYPE_F32;
        }
    }
    // F16 is the only compute type that can not satisfy a request for BF16
    if (prec == GGML_PREC_BF16 && compute_type == GGML_TYPE_F16) {
        compute_type = fast_bf16_hardware_available(cc) ? GGML_TYPE_BF16 : GGML_TYPE_F32;
    } else if (prec == GGML_PREC_F32) {
        compute_type = GGML_TYPE_F32;
    }

    // Parse GGML_CUDA_CUBLAS_COMPUTE_TYPE once: this runs for every cuBLAS matmul, and getenv() is
    // expensive on Windows (whose getenv returns a pointer that must not be retained across calls).
    static const ggml_type cublas_compute_type_env = []() -> ggml_type {
        const char * env_c = getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE");
        if (env_c == nullptr) {
            return GGML_TYPE_COUNT;
        }
        std::string env_cpp = env_c;
        for (char & c : env_cpp) {
            c = std::tolower(c);
        }
        if (env_cpp == "f32" || env_cpp == "fp32") {
            return GGML_TYPE_F32;
        } else if (env_cpp == "f16" || env_cpp == "fp16") {
            return GGML_TYPE_F16;
        } else if (env_cpp == "bf16") {
            return GGML_TYPE_BF16;
        } else if (env_cpp != "auto") {
            GGML_LOG_WARN("%s: unknown value for GGML_CUDA_CUBLAS_COMPUTE_TYPE: %s", __func__, env_cpp.c_str());
        }
        return GGML_TYPE_COUNT;
    }();
    if (cublas_compute_type_env != GGML_TYPE_COUNT) {
        compute_type = cublas_compute_type_env;
    }

    constexpr size_t max_src0_convert_size = 512ull * 1024 * 1024;
    const size_t src0_f32_size = ggml_nelements(src0) * sizeof(float);

    if (compute_type == GGML_TYPE_F32 &&
            (src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16) &&
            src0_f32_size > max_src0_convert_size) {
        const size_t f32_row_size = src0_f32_size / src0->ne[1];
        const int64_t rows_per_chunk = std::max<int64_t>(1, (int64_t) (max_src0_convert_size / f32_row_size));

        if (rows_per_chunk < src0->ne[1]) {
            ggml_tensor src0_chunk = *src0;
            ggml_tensor dst_chunk = *dst;

            for (int64_t i01 = 0; i01 < src0->ne[1]; i01 += rows_per_chunk) {
                src0_chunk.ne[1] = std::min(rows_per_chunk, src0->ne[1] - i01);
                src0_chunk.data = (char *) src0->data + i01*src0->nb[1];

                dst_chunk.ne[0] = src0_chunk.ne[1];
                dst_chunk.data = (char *) dst->data + i01*dst->nb[0];

                ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F32>(ctx, &src0_chunk, src1, &dst_chunk);
            }
            return;
        }
    }

    switch (compute_type) {
        case GGML_TYPE_F32:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F32>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_BF16:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_BF16>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_F16:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F16>(ctx, src0, src1, dst);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

bool ggml_cuda_should_fuse_mul_mat(const ggml_tensor * ffn_up,
                                          const ggml_tensor * ffn_gate,
                                          const ggml_tensor * glu,
                                          const ggml_tensor * ffn_up_bias,
                                          const ggml_tensor * ffn_gate_bias,
                                          const ggml_tensor * ffn_up_scale,
                                          const ggml_tensor * ffn_gate_scale) {
    const bool has_bias = ffn_up_bias != nullptr || ffn_gate_bias != nullptr;
    const bool has_scale = ffn_up_scale != nullptr || ffn_gate_scale != nullptr;

    if (has_bias && (!ffn_up_bias || !ffn_gate_bias)) {
        return false;
    }
    if (has_scale && (!ffn_up_scale || !ffn_gate_scale)) {
        return false;
    }

    const bool is_mul_mat     = ffn_up->op == GGML_OP_MUL_MAT     && ffn_gate->op == GGML_OP_MUL_MAT     && glu->op == GGML_OP_GLU;
    const bool is_mul_mat_id  = ffn_up->op == GGML_OP_MUL_MAT_ID  && ffn_gate->op == GGML_OP_MUL_MAT_ID  && glu->op == GGML_OP_GLU;

    GGML_ASSERT(ffn_up && ffn_gate && glu);

    if (!is_mul_mat && !is_mul_mat_id) {
        return false;
    }

    const ggml_op expected_bias_op = is_mul_mat ? GGML_OP_ADD : GGML_OP_ADD_ID;
    const ggml_tensor * ffn_up_bias_src   = has_scale ? ffn_up_scale   : ffn_up;
    const ggml_tensor * ffn_gate_bias_src = has_scale ? ffn_gate_scale : ffn_gate;
    const ggml_tensor * ffn_up_out        = has_bias ? ffn_up_bias     : ffn_up_bias_src;
    const ggml_tensor * ffn_gate_out      = has_bias ? ffn_gate_bias   : ffn_gate_bias_src;

    if (glu->src[0] != ffn_gate_out || glu->src[1] != ffn_up_out) {
        return false;
    }

    if (has_scale) {
        if (ffn_up_scale->op != GGML_OP_MUL || ffn_gate_scale->op != GGML_OP_MUL) {
            return false;
        }
        const bool up_has_mm   = ffn_up_scale->src[0] == ffn_up || ffn_up_scale->src[1] == ffn_up;
        const bool gate_has_mm = ffn_gate_scale->src[0] == ffn_gate || ffn_gate_scale->src[1] == ffn_gate;
        if (!up_has_mm || !gate_has_mm) {
            return false;
        }
    }

    if (has_bias) {
        if (ffn_up_bias->op != expected_bias_op || ffn_gate_bias->op != expected_bias_op) {
            return false;
        }

        if (expected_bias_op == GGML_OP_ADD) {
            const bool up_has_mul   = ffn_up_bias->src[0] == ffn_up_bias_src || ffn_up_bias->src[1] == ffn_up_bias_src;
            const bool gate_has_mul = ffn_gate_bias->src[0] == ffn_gate_bias_src || ffn_gate_bias->src[1] == ffn_gate_bias_src;
            if (!up_has_mul || !gate_has_mul) {
                return false;
            }
        } else { // GGML_OP_ADD_ID
            if (ffn_up_bias->src[0] != ffn_up_bias_src || ffn_gate_bias->src[0] != ffn_gate_bias_src) {
                return false;
            }
            if (ffn_up_bias->src[2] != ffn_up->src[2] || ffn_gate_bias->src[2] != ffn_gate->src[2]) {
                return false;
            }
        }
    }

    if (ffn_up->src[0]->type != ffn_gate->src[0]->type || !ggml_are_same_shape(ffn_up->src[0], ffn_gate->src[0]) ||
        !ggml_are_same_stride(ffn_up->src[0], ffn_gate->src[0])) {
        return false;
    }

    if (ffn_up->src[1] != ffn_gate->src[1]) {
        return false;
    }

    if (is_mul_mat_id && ffn_up->src[2] != ffn_gate->src[2]) {
        return false;
    }

    static constexpr std::array<ggml_glu_op, 4> valid_glu_ops = { GGML_GLU_OP_SWIGLU, GGML_GLU_OP_GEGLU, GGML_GLU_OP_SWIGLU_OAI, GGML_GLU_OP_SWIGLU_CLAMP };

    if (std::find(valid_glu_ops.begin(), valid_glu_ops.end(), ggml_get_glu_op(glu)) == valid_glu_ops.end()) {
        return false;
    }

    if (const bool swapped = ggml_get_op_params_i32(glu, 1); swapped) {
        return false;
    }

    return true;
}


// verify_band: callers that only pre-fill the mmvq Q8_1 activation cache (norm -> Q8_1, gated unary -> Q8_1)
// and leave the matmul itself unchanged.  Their kernels are row-generic, so on RDNA4 and RDNA3_0 the verify band
// (2..8 tokens) takes them too; the quantized values are the ones quantize_q8_1 would write, so W = 1..8
// stay bit-identical.  GGML_CUDA_FUSE_Q8_1_VERIFY=0 turns it off.

// Buffer- and batch-size independent, so graph_optimize produces the same graph topology for every ubatch size (see #29986).
static bool ggml_cuda_mul_mat_q_fusion_matches(const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * glu, const int device) {
    const ggml_tensor * x_up = up->src[0];
    const ggml_tensor * y    = up->src[1];

    if (up->op != GGML_OP_MUL_MAT || !ggml_cuda_mmq_fusion_supported(x_up->type) || y->type != GGML_TYPE_F32) {
        return false;
    }

    // The fused kernel is only compiled for weights whose rows fill whole tiles (no fallback).
    if (ggml_cuda_mmq_needs_fallback(x_up->ne[1])) {
        return false;
    }

    if (!ggml_cuda_should_fuse_mul_mat(up, gate, glu)) {
        return false;
    }

    // The fused write-back applies SWIGLU_OAI with the default alpha and limit of ggml_cuda_op_swiglu_oai_single.
    if (ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU_OAI && (ggml_get_op_params_f32(glu, 2) != 1.702f || ggml_get_op_params_f32(glu, 3) != 7.0f)) {
        return false;
    }

    // NVIDIA only: the fused write-back relies on the MMA accumulator layout.
    return turing_mma_available(ggml_cuda_info().devices[device].cc);
}

bool ggml_cuda_should_fuse_mul_mat_q(const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * glu) {
    // A hinted matmul can be fused, but its unfused path may be faster (e.g. FWHT for GGML_HINT_SRC0_IS_HADAMARD).
    if (ggml_get_op_params_i32(up, 1) != GGML_HINT_NONE || ggml_get_op_params_i32(gate, 1) != GGML_HINT_NONE) {
        return false;
    }

    // The padding of a compute-buffer weight that is a view cannot be cleared, see ggml_cuda_mul_mat.
    const auto bad_padding_clear = [](const ggml_tensor * x) {
        return ggml_backend_buffer_get_usage(x->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
            ggml_nbytes(x) != ggml_backend_buffer_get_alloc_size(x->buffer, x) && x->view_src;
    };
    const int device = ggml_cuda_get_device();
    if (!ggml_cuda_mul_mat_q_fusion_matches(up, gate, glu, device) || bad_padding_clear(up->src[0]) || bad_padding_clear(gate->src[0])) {
        return false;
    }

    // The batch size checks are done here, not in ggml_cuda_mul_mat_q_fusion_matches.
    const ggml_tensor * x_up = up->src[0];
    const ggml_tensor * y    = up->src[1];
    const int           cc   = ggml_cuda_info().devices[device].cc;
    if (ggml_cuda_should_use_mmvq(x_up->type, cc, y->ne[1]) || !ggml_cuda_should_use_mmq(x_up->type, cc, y->ne[1], 0)) {
        return false;
    }

    // The fused kernel needs one block per tile (no stream-k split), so only fuse if the tiling is efficient.
    return ggml_cuda_mmq_fusion_is_efficient(x_up->type, x_up->ne[1], y->ne[1], y->ne[2]*y->ne[3], device);
}

// True iff ggml_cuda_mul_mat() below would run this MUL_MAT through ggml_cuda_mul_mat_q: the same
// predicate chain in the same order.  Keep the two in sync.
bool ggml_cuda_mul_mat_takes_mmq(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    if (ggml_get_op_params_i32(dst, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
        return false;
    }
    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE
        && ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) && src0->view_src;
    if (bad_padding_clear || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    const int cc        = ggml_cuda_info().devices[ctx.device].cc;
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
    const int64_t ne11 = src1->ne[1];
    if (ggml_cuda_mmb_supported_mm(src0, src1, dst)) {
        return false;
    }
    const int64_t ne11_mmvf = ne11 <= MMVF_MAX_BATCH_SIZE_FLAT ? 1 : ne11;
    if (ggml_cuda_should_use_mmvf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11_mmvf) || src0->ne[1] == 1) {
        return false;
    }
    if (ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11, /*mul_mat_id =*/ false)) {
        return false;
    }
    bool use_mmvq = ggml_cuda_should_use_mmvq(src0->type, cc, ne11);
    static const bool dense_band_off = getenv("GGML_CUDA_DISABLE_MMVQ_DENSE_BAND") != nullptr;
    if (!use_mmvq && !dense_band_off && (GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc)) && ggml_is_quantized(src0->type) && ne11 <= MMVQ_MOE_MAX_BATCH_SIZE && src0->ne[1] % 128 != 0) {
        use_mmvq = true;
    }
    return !use_mmvq && ggml_cuda_should_use_mmq(src0->type, cc, ne11, /*n_experts =*/ 0);
}

static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS

    if (ggml_cuda_op_mul_mat_use_fwht(dst) && ggml_cuda_op_fwht(ctx, src1, dst)) {
        return;
    }

    // If src0 is a temporary compute buffer it may have some padding that needs to be cleared for mul_mat_vec_q or mul_mat_q.
    // But if src0 is also a view of another tensor then this cannot be done safely because it may overwrite valid tensor data.
    // Therefore, in such cases use cuBLAS.
    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE
        && ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) && src0->view_src;
    if (bad_padding_clear || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        MMB_T(ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst), 6);
        return;
    }

    const int cc        = ggml_cuda_info().devices[ctx.device].cc;
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;

    // MMB: bf16-WMMA dequant weight GEMM (pwilkin strix-halo port). Prefill-only: the predicate
    // requires T >= GGML_CUDA_MMB_MIN_T (default 512), so the whole decode/verify band
    // (n_tokens <= 8) stays on the existing kernels and W = 1..8 is bit-identical either way.
    if (ggml_cuda_mmb_supported_mm(src0, src1, dst)) {
        MMB_T(ggml_cuda_mul_mat_mmb(ctx, src0, src1, dst), 0);
        return;
    }

    // Speculative verify batches (ne11 = n_q <= 8) must run the same kernel as
    // decode (ne11 = 1): decode uses the MMVF kernel, while a larger batch can
    // fall through to MMF, which accumulates differently and produces different
    // logits. Use the decode (ne11 = 1) config for all small batches.
    // The QSA indexer score flattens heads*tokens into ne11 (4*n_tps for qwen4exp), so the
    // decode/verify band can reach ne11 = MMVF_MAX_BATCH_SIZE_FLAT while its token count is
    // still <= MMVF_MAX_BATCH_SIZE.  Keep the whole band on the decode family (MMVF) or the
    // verify batch falls through to MMF and accumulates differently.
    const int64_t ne11_mmvf = ne11 <= MMVF_MAX_BATCH_SIZE_FLAT ? 1 : ne11;
    // upstream #29633 added `warp_size` to the predicate (it dispatches to should_use_mmf)
    if (ggml_cuda_should_use_mmvf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11_mmvf)) {
        // The custom F16 vector kernel can be used over batched cuBLAS GEMM.
        // But this is only faster for GPUs without tensor cores or with a thin src0 matrix (particularly KQV in attention)
        MMB_T(ggml_cuda_mul_mat_vec_f(ctx, src0, src1, nullptr, dst), 1);
        return;
    }
    // A transposed vector can still use MMVQ (i.e. ne01 == 1)
    if (ne01 == 1 && ne11 > MMVF_MAX_BATCH_SIZE && ne2 == 1 && ne3 == 1
            && src0->type == GGML_TYPE_F32
            && ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst)
            && ggml_cuda_should_use_mmvf(src1->type, cc, warp_size, src1->ne, src1->nb, /*ne11 =*/ 1)) {
        ggml_tensor dst_vec = *dst;
        dst_vec.ne[0] = ne11;
        dst_vec.ne[1] = 1;
        dst_vec.nb[1] = dst_vec.nb[0]*ne11;
        dst_vec.nb[2] = dst_vec.nb[1];
        dst_vec.nb[3] = dst_vec.nb[1];
        MMB_T(ggml_cuda_mul_mat_vec_f(ctx, src1, src0, nullptr, &dst_vec), 2);
        return;
    }
    if (ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11, /*mul_mat_id =*/ false)) {
        MMB_T(ggml_cuda_mul_mat_f(ctx, src0, src1, nullptr, dst), 3);
        return;
    }
    bool use_mmvq = ggml_cuda_should_use_mmvq(src0->type, cc, ne11);
    // RDNA4/RDNA3_5: a dense weight whose output-row count is not a multiple of 128 takes MMQ's slow
    // non-128-row "fallback" config (~3x per launch versus the ksplit MMVQ kernel at these small
    // column counts; the qwen4exp n_tokens = 9 verify step).  Keep the ksplit band for those rows.
    // GGML_CUDA_DISABLE_MMVQ_DENSE_BAND=1 is the bisect/A-B kill-switch (default on).
    static const bool dense_band_off = getenv("GGML_CUDA_DISABLE_MMVQ_DENSE_BAND") != nullptr;
    if (!use_mmvq && !dense_band_off && (GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc)) && ggml_is_quantized(src0->type) && ne11 <= MMVQ_MOE_MAX_BATCH_SIZE && src0->ne[1] % 128 != 0) {
        use_mmvq = true;
    }
    if (use_mmvq) {
        MMB_T(ggml_cuda_mul_mat_vec_q(ctx, src0, src1, nullptr, dst), 4);
        return;
    }
    if (ggml_cuda_should_use_mmq(src0->type, cc, ne11, /*n_experts =*/ 0)) {
        MMB_T(ggml_cuda_mul_mat_q(ctx, src0, src1, nullptr, dst), 5);
        return;
    }
    MMB_T(ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst), 6);
}

// returns true when ggml_cuda_mul_mat_id takes the fallback path that requires stream synchronization
// [TAG_MUL_MAT_ID_CUDA_GRAPHS]
static bool ggml_cuda_mul_mat_id_needs_sync(const ggml_tensor * dst, const int cc) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;

    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return true;
    }

    if (ggml_is_quantized(src0->type)) {
        if (dst->ne[2] <= get_mmvq_mmid_max_batch(src0->type, cc)) {
            return false;
        }
    } else if (dst->ne[2] <= MMVQ_MAX_BATCH_SIZE && GGML_CUDA_CC_IS_AMD(cc)) {
        return false;
    }

    if (ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[2], /*n_experts=*/src0->ne[2])) {
        return false;
    }

    if (ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
        return false;
    }

    return true;
}

static void ggml_cuda_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    // wip/moe-heatmap-persistence (R2): the device-side prefill routing tally.  Run at the OUTER entry
    // only: the cache's slot-remap redirect below re-enters this function with a stack `src0` whose
    // `buffer` is null and whose `dst->src[2]` is the slot remap (not the routing), while the outer
    // entry always carries the graph's true raw ids - for every path (identity, eager, staged prefill).
    // The tally itself is band-gated (prefill only) and a no-op unless MOE_EXPERT_HEATMAP is set.
    if (moe_cache_enabled() && src0->buffer != nullptr && ids != nullptr) {
        moe_cache_heat_observe(dst, src0, ids, ctx.stream(), ctx.device);
    }

    // wip/moe-expert-cache slot-remap consumer: if the scheduler handed this expert table to the
    // cache (the backend iface `moe_cache_update` returned true), read the compact arena and the
    // slot-remapped ids instead of the full expert table.  `src0` is the redirected `input_cpy`,
    // which the cache aliased; `moe_cache_get_table` maps it to the arena.  The shallow `src0`/
    // `ids` copies are re-dispatched through this same function, whose cache check then misses
    // (the copies are stack pointers), so the normal kernel selection runs on the compact table.
    if (moe_cache_enabled()) {
        void *    arena   = nullptr;
        void *    remap   = nullptr;
        int64_t   n_slots = 0;
        int64_t   nu = 0;
        int64_t   nt = 0;
        size_t    eb = 0;
        if (moe_cache_get_table(dst, src0, ctx.device, &arena, &n_slots, &eb, &remap, &nu, &nt) &&
            (remap == nullptr || (nu == ids->ne[0] && nt == ids->ne[1]))) {
            // Keep `ne[2]`/`nb` exactly as the full table so the dispatcher's kernel-family
            // heuristics (which read `ne[2]`/`n_experts`) are unchanged and the arithmetic stays
            // bit-identical; only the base pointer moves to the arena, and the remapped ids stay
            // below `slots`, so the kernel never reads an expert outside the arena.  Unused expert
            // slots are never referenced by the ids and are only skipped over.
            GGML_UNUSED(n_slots);
            GGML_UNUSED(eb);
            ggml_tensor src0c = *src0;
            src0c.data = arena;
            // The arena is a raw cudaMalloc, not a ggml buffer; drop `buffer` so the mmvq
            // compute-buffer padding clear (which would offset by `ggml_nbytes(src0)` = the FULL
            // expert count) is skipped and cannot memset past the compact arena.
            src0c.buffer = nullptr;
            ggml_tensor idsc = *ids;
            if (remap != nullptr) {
                idsc.data   = remap;
                // the remap buffer is contiguous (`n_used x n_tok`), while `ids` is a strided view
                idsc.nb[0]  = sizeof(int32_t);
                idsc.nb[1]  = (size_t) ids->ne[0] * sizeof(int32_t);
            }   // else: identity fast path - the raw routing ids already index the arena

            ggml_tensor * saved_s0 = dst->src[0];
            ggml_tensor * saved_s2 = dst->src[2];
            dst->src[0] = &src0c;
            dst->src[2] = &idsc;
            ggml_cuda_mul_mat_id(ctx, dst);
            dst->src[0] = saved_s0;
            dst->src[2] = saved_s2;
            return;
        }
    }

    GGML_TENSOR_BINARY_OP_LOCALS

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;

    // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
    if (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        static_assert(MMVQ_MAX_BATCH_SIZE == MMVF_MAX_BATCH_SIZE);
        // Quantized routed experts take the dedicated MoE MMVQ kernel over the whole verify band
        // (MMVQ_MOE_MAX_BATCH_SIZE); the F32/F16 expert path keeps the 8-column MMVF band.
        if (ggml_is_quantized(src0->type)) {
            const int mmvq_mmid_max = get_mmvq_mmid_max_batch(src0->type, cc);
            if (ne2 <= mmvq_mmid_max) {
                ggml_cuda_mul_mat_vec_q(ctx, src0, src1, ids, dst);
                return;
            }
        } else if (ne2 <= MMVQ_MAX_BATCH_SIZE) {
            if (GGML_CUDA_CC_IS_AMD(cc)) {
                ggml_cuda_mul_mat_vec_f(ctx, src0, src1, ids, dst);
                return;
            }
        }

        if (ggml_cuda_mmb_supported_mmid(src0, src1, ids, dst)) {
            ggml_cuda_mul_mat_id_mmb(ctx, src0, src1, ids, dst);
            return;
        }

        if (ggml_cuda_should_use_mmq(src0->type, cc, ne12, /*n_experts=*/ne02)) {
            ggml_cuda_mul_mat_q(ctx, src0, src1, ids, dst);
            return;
        }

        if (ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
            ggml_cuda_mul_mat_f(ctx, src0, src1, ids, dst);
            return;
        }
    }

    // note: this path should not be reached when recording CUDA graphs, because it requires stream synchronization
    GGML_ASSERT(ggml_cuda_mul_mat_id_needs_sync(dst, cc));
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const ggml_type type_src1_sorted = (src0->type == GGML_TYPE_F16 && !fast_fp16_hardware_available(cc))
        || ggml_is_quantized(src0->type) ? GGML_TYPE_F32 : src0->type;
    const ggml_type type_dst_sorted  = GGML_TYPE_F32;
    const size_t ts_src1_sorted = ggml_type_size(type_src1_sorted);
    const size_t ts_dst_sorted  = ggml_type_size(type_dst_sorted);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;

    std::vector<int32_t> ids_to_sorted_host;
    ids_to_sorted_host.reserve(2*ne_get_rows);
    std::vector<int32_t> ids_from_sorted_host(ne_get_rows);

    ggml_cuda_pool_alloc<int32_t> ids_buf_dev(ctx.pool(), 2*ne_get_rows);

    std::vector<int32_t> tokens_per_expert(ne02);

    ggml_cuda_pool_alloc<char> src1_sorted(ctx.pool(), ne12*n_expert_used*ne10*ts_src1_sorted);
    ggml_cuda_pool_alloc<char>  dst_sorted(ctx.pool(), ne2 *n_expert_used* ne0*ts_dst_sorted);

    std::vector<char> ids_host(ggml_nbytes(ids));
    CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids->data, ggml_nbytes(ids), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int64_t i02 = 0; i02 < ne02; ++i02) { // expert matrices
        for (int64_t i12 = 0; i12 < ne12; ++i12) { // tokens
            for (int64_t iex = 0; iex < n_expert_used; ++iex) {
                const int32_t expert_to_use = *(const int32_t *)(ids_host.data() + i12*ids->nb[1] + iex*ids->nb[0]);
                assert(expert_to_use >= 0 && expert_to_use < ne02);
                // an id may repeat within a row: each occurrence is a separate dst column with its own src1 column
                if (expert_to_use == i02) {
                    ids_from_sorted_host[i12*n_expert_used + iex] = ids_to_sorted_host.size();
                    ids_to_sorted_host.push_back(i12*ne11 + iex % ne11);
                    tokens_per_expert[i02]++;
                }
            }
        }
    }
    GGML_ASSERT(ids_to_sorted_host.size() == size_t(ne_get_rows));

    ids_to_sorted_host.insert(ids_to_sorted_host.end(), ids_from_sorted_host.begin(), ids_from_sorted_host.end());

    CUDA_CHECK(cudaMemcpyAsync(ids_buf_dev.ptr, ids_to_sorted_host.data(), 2*ne_get_rows*sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const int32_t * ids_to_sorted   = ids_buf_dev.ptr + 0*ne_get_rows;
    const int32_t * ids_from_sorted = ids_buf_dev.ptr + 1*ne_get_rows;

    get_rows_cuda(src1->data, src1->type, ids_to_sorted, src1_sorted.ptr, type_src1_sorted,
        ne10, nb11, nb12, nb13,
        ne_get_rows, 1, 1, sizeof(int32_t), ne_get_rows*sizeof(int32_t), ne_get_rows*sizeof(int32_t),
        ne10*ts_src1_sorted, ne_get_rows*ne10*ts_src1_sorted, ne_get_rows*ne10*ts_src1_sorted, stream);
    CUDA_CHECK(cudaGetLastError());

    char * src1_data_cur = (char *) src1_sorted.ptr;
    char *  dst_data_cur = (char *)  dst_sorted.ptr;
    for (int64_t i02 = 0; i02 < ne02; ++i02) {
        if (tokens_per_expert[i02] == 0) {
            continue;
        }

        ggml_tensor src0_slice = *src0;
        src0_slice.ne[2]    = 1;
        src0_slice.nb[3]    = src0_slice.nb[2];
        src0_slice.op       = GGML_OP_VIEW;
        src0_slice.view_src = dst->src[0]; // non-const pointer to src0
        src0_slice.data     = (char *) src0->data + i02*nb02;

        ggml_tensor src1_slice;
        memset(&src1_slice, 0, sizeof(src1_slice));
        src1_slice.buffer = src1->buffer;
        src1_slice.type   = type_src1_sorted;
        src1_slice.ne[0]  = ne10;
        src1_slice.ne[1]  = tokens_per_expert[i02];
        src1_slice.ne[2]  = 1;
        src1_slice.ne[3]  = 1;
        src1_slice.nb[0]  = ts_src1_sorted;
        src1_slice.nb[1]  = src1_slice.ne[0] * src1_slice.nb[0];
        src1_slice.nb[2]  = src1_slice.ne[1] * src1_slice.nb[1];
        src1_slice.nb[3]  = src1_slice.ne[2] * src1_slice.nb[2];
        src1_slice.data   = src1_data_cur;

        ggml_tensor dst_slice;
        memset(&dst_slice, 0, sizeof(dst_slice));
        memcpy(dst_slice.op_params, dst->op_params, sizeof(dst_slice.op_params));
        dst_slice.buffer = dst->buffer;
        dst_slice.type   = type_dst_sorted;
        dst_slice.ne[0]  = ne0;
        dst_slice.ne[1]  = tokens_per_expert[i02];
        dst_slice.ne[2]  = 1;
        dst_slice.ne[3]  = 1;
        dst_slice.nb[0]  = ts_dst_sorted;
        dst_slice.nb[1]  = dst_slice.ne[0] * dst_slice.nb[0];
        dst_slice.nb[2]  = dst_slice.ne[1] * dst_slice.nb[1];
        dst_slice.nb[3]  = dst_slice.ne[2] * dst_slice.nb[2];
        dst_slice.data   = dst_data_cur;

        ggml_cuda_mul_mat(ctx, &src0_slice, &src1_slice, &dst_slice);
        CUDA_CHECK(cudaGetLastError());

        src1_data_cur += src1_slice.nb[2];
        dst_data_cur  +=  dst_slice.nb[2];
    }

    get_rows_cuda(dst_sorted.ptr, type_dst_sorted, ids_from_sorted, dst->data, dst->type,
        ne0, ne0*ts_dst_sorted, ne_get_rows*ne0*ts_dst_sorted, ne_get_rows*ne0*ts_dst_sorted,
        ne_get_rows, 1, 1, sizeof(int32_t), ne_get_rows*sizeof(int32_t), ne_get_rows*sizeof(int32_t),
        nb1, nb2, nb3, stream);
}

bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    switch (dst->op) {
        case GGML_OP_ARGMAX:
            ggml_cuda_argmax(ctx, dst);
            break;
        case GGML_OP_COUNT_EQUAL:
            ggml_cuda_count_equal(ctx, dst);
            break;
        case GGML_OP_REPEAT:
            ggml_cuda_op_repeat(ctx, dst);
            break;
        case GGML_OP_REPEAT_BACK:
            ggml_cuda_op_repeat_back(ctx, dst);
            break;
        case GGML_OP_GET_ROWS:
            ggml_cuda_op_get_rows(ctx, dst);
            break;
        case GGML_OP_GET_ROWS_BACK:
            ggml_cuda_op_get_rows_back(ctx, dst);
            break;
        case GGML_OP_SET_ROWS:
            ggml_cuda_op_set_rows(ctx, dst);
            break;
        case GGML_OP_SET:
            ggml_cuda_op_set(ctx, dst);
            break;
        case GGML_OP_DUP:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_CPY:
            ggml_cuda_cpy(ctx, dst->src[0], dst->src[1]);
            break;
        case GGML_OP_CONT:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_ADD:
        case GGML_OP_ADD1: // TODO: more efficient implementation
            ggml_cuda_op_add(ctx, dst);
            break;
        case GGML_OP_ADD_ID:
            ggml_cuda_op_add_id(ctx, dst);
            break;
        case GGML_OP_SUB:
            ggml_cuda_op_sub(ctx, dst);
            break;
        case GGML_OP_ACC:
            ggml_cuda_op_acc(ctx, dst);
            break;
        case GGML_OP_MUL:
            ggml_cuda_op_mul(ctx, dst);
            break;
        case GGML_OP_DIV:
            ggml_cuda_op_div(ctx, dst);
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_ABS:
                    ggml_cuda_op_abs(ctx, dst);
                    break;
                case GGML_UNARY_OP_SGN:
                    ggml_cuda_op_sgn(ctx, dst);
                    break;
                case GGML_UNARY_OP_NEG:
                    ggml_cuda_op_neg(ctx, dst);
                    break;
                case GGML_UNARY_OP_STEP:
                    ggml_cuda_op_step(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU:
                    ggml_cuda_op_gelu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SILU:
                    ggml_cuda_op_silu(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_ERF:
                    ggml_cuda_op_gelu_erf(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    ggml_cuda_op_gelu_quick(ctx, dst);
                    break;
                case GGML_UNARY_OP_TANH:
                    ggml_cuda_op_tanh(ctx, dst);
                    break;
                case GGML_UNARY_OP_RELU:
                    ggml_cuda_op_relu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SIGMOID:
                    ggml_cuda_op_sigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    ggml_cuda_op_hardsigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    ggml_cuda_op_hardswish(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXP:
                    ggml_cuda_op_exp(ctx, dst);
                    break;
                case GGML_UNARY_OP_ELU:
                    ggml_cuda_op_elu(ctx, dst);
                    break;
                case GGML_UNARY_OP_XIELU:
                    ggml_cuda_op_xielu(ctx, dst);
                    break;
                case GGML_UNARY_OP_FLOOR:
                    ggml_cuda_op_floor(ctx, dst);
                    break;
                case GGML_UNARY_OP_CEIL:
                    ggml_cuda_op_ceil(ctx, dst);
                    break;
                case GGML_UNARY_OP_ROUND:
                    ggml_cuda_op_round(ctx, dst);
                    break;
                case GGML_UNARY_OP_TRUNC:
                    ggml_cuda_op_trunc(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXPM1:
                    ggml_cuda_op_expm1(ctx, dst);
                    break;
                case GGML_UNARY_OP_SOFTPLUS:
                    ggml_cuda_op_softplus(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(dst)) {
                case GGML_GLU_OP_REGLU:
                    ggml_cuda_op_reglu(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU:
                    ggml_cuda_op_geglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU:
                    ggml_cuda_op_swiglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI:
                    ggml_cuda_op_swiglu_oai(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_ERF:
                    ggml_cuda_op_geglu_erf(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_QUICK:
                    ggml_cuda_op_geglu_quick(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    ggml_cuda_op_swiglu_clamp(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            ggml_cuda_op_norm(ctx, dst);
            break;
        case GGML_OP_GROUP_NORM:
            ggml_cuda_op_group_norm(ctx, dst);
            break;
        case GGML_OP_L2_NORM:
            ggml_cuda_op_l2_norm(ctx, dst);
            break;
        case GGML_OP_CONCAT:
            ggml_cuda_op_concat(ctx, dst);
            break;
        case GGML_OP_UPSCALE:
            ggml_cuda_op_upscale(ctx, dst);
            break;
        case GGML_OP_PAD:
            ggml_cuda_op_pad(ctx, dst);
            break;
        case GGML_OP_PAD_REFLECT_1D:
            ggml_cuda_op_pad_reflect_1d(ctx, dst);
            break;
        case GGML_OP_ARANGE:
            ggml_cuda_op_arange(ctx, dst);
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggml_cuda_op_timestep_embedding(ctx, dst);
            break;
        case GGML_OP_LEAKY_RELU:
            ggml_cuda_op_leaky_relu(ctx, dst);
            break;
        case GGML_OP_SILU_BACK:
            ggml_cuda_op_silu_back(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            ggml_cuda_op_rms_norm(ctx, dst);
            break;
        case GGML_OP_RMS_NORM_BACK:
            ggml_cuda_op_rms_norm_back(ctx, dst);
            break;
        case GGML_OP_MUL_MAT:
            ggml_cuda_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            break;
        case GGML_OP_MUL_MAT_ID:
            ggml_cuda_mul_mat_id(ctx, dst);
            break;
        case GGML_OP_OUT_PROD:
            ggml_cuda_out_prod(ctx, dst);
            break;
        case GGML_OP_SCALE:
            ggml_cuda_op_scale(ctx, dst);
            break;
        case GGML_OP_SQR:
            ggml_cuda_op_sqr(ctx, dst);
            break;
        case GGML_OP_SQRT:
            ggml_cuda_op_sqrt(ctx, dst);
            break;
        case GGML_OP_SIN:
            ggml_cuda_op_sin(ctx, dst);
            break;
        case GGML_OP_COS:
            ggml_cuda_op_cos(ctx, dst);
            break;
        case GGML_OP_CLAMP:
            ggml_cuda_op_clamp(ctx, dst);
            break;
        case GGML_OP_LOG:
            ggml_cuda_op_log(ctx, dst);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
                break;
        case GGML_OP_DIAG:
            ggml_cuda_op_diag(ctx, dst);
            break;
        case GGML_OP_DIAG_MASK_INF:
            ggml_cuda_op_diag_mask_inf(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX:
            ggml_cuda_op_soft_max(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX_BACK:
            ggml_cuda_op_soft_max_back(ctx, dst);
            break;
        case GGML_OP_ROPE:
            ggml_cuda_op_rope(ctx, dst);
            break;
        case GGML_OP_ROPE_BACK:
            ggml_cuda_op_rope_back(ctx, dst);
            break;
        case GGML_OP_ROLL:
            ggml_cuda_op_roll(ctx, dst);
            break;
        case GGML_OP_IM2COL:
            ggml_cuda_op_im2col(ctx, dst);
            break;
        case GGML_OP_IM2COL_3D:
            ggml_cuda_op_im2col_3d(ctx, dst);
            break;
        case GGML_OP_CONV_2D:
            ggml_cuda_op_conv2d(ctx, dst);
            break;
        case GGML_OP_CONV_3D:
            ggml_cuda_op_conv3d(ctx, dst);
            break;
        case GGML_OP_CONV_2D_DW:
            ggml_cuda_op_conv2d_dw(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            ggml_cuda_conv_2d_transpose_p0(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            ggml_cuda_op_conv_transpose_1d(ctx,dst);
            break;
        case GGML_OP_COL2IM_1D:
            ggml_cuda_op_col2im_1d(ctx, dst);
            break;
        case GGML_OP_POOL_2D:
            ggml_cuda_op_pool2d(ctx, dst);
            break;
        case GGML_OP_POOL_1D:
            ggml_cuda_op_pool1d(ctx, dst);
            break;
        case GGML_OP_SUM:
            ggml_cuda_op_sum(ctx, dst);
            break;
        case GGML_OP_CUMSUM:
            ggml_cuda_op_cumsum(ctx, dst);
            break;
        case GGML_OP_SUM_ROWS:
            ggml_cuda_op_sum_rows(ctx, dst);
            break;
        case GGML_OP_MEAN:
            ggml_cuda_op_mean(ctx, dst);
            break;
        case GGML_OP_SSM_CONV:
            ggml_cuda_op_ssm_conv(ctx, dst);
            break;
        case GGML_OP_SSM_SCAN:
            ggml_cuda_op_ssm_scan(ctx, dst);
            break;
        case GGML_OP_TOP_K:
            ggml_cuda_op_top_k(ctx, dst);
            break;
        case GGML_OP_ARGSORT:
            ggml_cuda_op_argsort(ctx, dst);
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            ggml_cuda_flash_attn_ext(ctx, dst);
            break;
        case GGML_OP_FLASH_ATTN_QSA:
            ggml_cuda_flash_attn_qsa(ctx, dst);
            break;
        case GGML_OP_INDEXER_TOPK:
            ggml_cuda_indexer_top_k(ctx, dst);
            break;
            break;
        case GGML_OP_INDEXER_SCORE:
            ggml_cuda_indexer_score(ctx, dst);
            break;
        case GGML_OP_INDEXER_FILL:
            ggml_cuda_indexer_fill(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_cuda_cross_entropy_loss(ctx, dst);
            break;
        case GGML_OP_TRI:
            ggml_cuda_op_tri(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV6:
            ggml_cuda_op_rwkv_wkv6(ctx, dst);
            break;
        case GGML_OP_GATED_LINEAR_ATTN:
            ggml_cuda_op_gated_linear_attn(ctx, dst);
            break;
        case GGML_OP_GATED_DELTA_NET:
            ggml_cuda_op_gated_delta_net(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_COMB:
            ggml_cuda_op_dsv4_hc_comb(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_PRE:
            ggml_cuda_op_dsv4_hc_pre(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_POST:
            ggml_cuda_op_dsv4_hc_post(ctx, dst);
            break;
        case GGML_OP_HC_MIX:
            ggml_cuda_op_hc_mix(ctx, dst);
            break;
        case GGML_OP_HC_COMBINE:
            ggml_cuda_op_hc_combine(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV7:
            ggml_cuda_op_rwkv_wkv7(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            ggml_cuda_cross_entropy_loss_back(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_ADAMW:
            ggml_cuda_opt_step_adamw(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_SGD:
            ggml_cuda_opt_step_sgd(ctx, dst);
            break;
        case GGML_OP_SOLVE_TRI:
            ggml_cuda_op_solve_tri(ctx, dst);
            break;
        case GGML_OP_FILL:
            ggml_cuda_op_fill(ctx, dst);
            break;
        case GGML_OP_LIGHTNING_INDEXER:
            ggml_cuda_lightning_indexer(ctx, dst);
            break;
        case GGML_OP_TOP_K_MASK:
            ggml_cuda_op_top_k_mask(ctx, dst);
            break;
        default:
            return false;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: %s failed\n", __func__, ggml_op_desc(dst));
        CUDA_CHECK(err);
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////

// backend

static const char * ggml_backend_cuda_get_name(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    return cuda_ctx->name.c_str();
}

static void ggml_backend_cuda_free(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    delete cuda_ctx;
    delete backend;
}

static void ggml_backend_cuda_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    // A pinned-source cudaMemcpyAsync is read by the copy engine when the stream reaches it, not when it is
    // issued, so the caller must not modify `data` until the copy has completed -- and this path's caller
    // cannot promise that: the scheduler uses it for a host-resident *split* input, and the one that bites is
    // a CPU split's output (`node_1` / `model.input_embed` in a ROCm_Host buffer), which the next ubatch's
    // CPU split rewrites in place.  The copy could read the next ubatch's bytes, or a torn mix.
    // Snapshot through the pinned upload ring instead: a ring slot is not reused until the ring wraps, and
    // the wrap is guarded by a device sync.  Unconditional -- this is a correctness fix, not an async-path
    // optimisation.
    if (size > 0 && size <= GGML_CUDA_ASYNC_INPUT_MAX &&
        ggml_backend_buffer_get_usage(buf) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        // the ring is per device, and the meta (tensor-split) backend forwards set_tensor_async per child,
        // so do not assume the caller left this backend's device current
        ggml_cuda_set_device(cuda_ctx->device);
        if (ggml_cuda_async_input_set(cuda_ctx->device, (char *) tensor->data + offset, data, size)) {
            return;
        }
    }
    ggml_cuda_async_input_drain(cuda_ctx->device);

    CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice, cuda_ctx->stream()));
}

static void ggml_backend_cuda_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");
    ggml_cuda_async_input_drain(cuda_ctx->device);

    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

static void ggml_backend_cuda_set_tensor_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");
    ggml_cuda_async_input_drain(cuda_ctx->device);

    ggml_cuda_set_device(cuda_ctx->device);

    if (g_stream_dbg_on) {
        static int n = 0;
        if (n++ < 24) fprintf(stderr, "STREAMDBG up2d dev=%d stream=%p size=%zu nc=%zu dst=%p name=%s\n",
            cuda_ctx->device, (void *) cuda_ctx->stream(), size, n_copies, (void *)((char *) tensor->data + offset), tensor->name);
    }
    // exp32/33: avoid the faulting pageable H2D 2-D copy.  Gather this device's compacted slice on
    // the host into a hipHostMalloc (pinned) buffer, then issue ONE 1-D H2D from pinned -- no 2-D
    // copy, half the volume, and the transfer is genuinely asynchronous.  (stride_tensor == size in
    // the splice, so the compacted layout is contiguous.)
    // Production default (wip/tensor-split-expert-split): a COMPACTED strided host->device upload --
    // the tensor-split splice, which takes `chunk_size_j` bytes out of every `chunk_size_full` -- goes
    // through a pinned host gather plus ONE queued 1-D H2D instead of the pageable `hipMemcpy2DAsync`.
    // The 2-D copy both faults from a pageable source (\u00a722) and is ~5-7x slower than the gather+1D
    // even from an already-pinned one, which is what made the split lose below the staging gate.
    // GGML_CUDA_SPLICE_GATHER=0 selects the strided 2-D upload instead of the compacted 1-D one.
    // `splice_gather` therefore only chooses the *shape* of the queued copy: the pinned source snapshot
    // below is unconditional, because the caller's pointer may be rewritten at any time (see
    // ggml_backend_cuda_set_tensor_async, same hazard) and a pageable-source 2-D H2D also faults on ROCm.
    const bool splice_gather = g_cuda_splice_env_on && n_copies > 1 && stride_tensor == size;
    {
        // wip/fix-meta-split-copy: the H2D MUST be ordered after the main stream.  The scheduler calls
        // `wait_before_overwrite()` (a wait on the split backend's previous-split event) on the MAIN
        // stream before this upload, because the graph allocator reuses this split-input buffer across
        // splits -- the previous split may still be reading it.  The plain 2-D path below issues its
        // copy on `cuda_ctx->stream()` and is therefore ordered; an earlier version of this gather
        // issued the 1-D H2D on `copy_stream()` and only ordered the *consumer*
        // (`cudaStreamWaitEvent(main, s.ev)`), so the H2D could overwrite the buffer while the previous
        // split still read it -- it corrupts the `-sm tensor` + `-ncmoe` below-gate path (measured: 35B-A3B
        // `3333…`, gemma4 26B-A4B `<unused32>…`, `GGML_META_SPLIT_COPY=0`/`GGML_CUDA_SPLICE_GATHER=0`
        // clean).  The copy stream buys nothing here (the transfer must wait for that read anyway), so
        // the gather stays a host pinned gather + one 1-D H2D, just on the main stream.
        const cudaStream_t h2d_stream = cuda_ctx->stream();
        const size_t full    = (n_copies > 0 ? (size_t)(n_copies-1)*stride_data    : 0) + size;
        const size_t compact = (n_copies > 0 ? (size_t)(n_copies-1)*stride_tensor : 0) + size;
        // the compact path only ever materializes the gathered slice; the strided path memcpys the whole range
        const size_t need    = (stride_tensor == size) ? compact : (full > compact ? full : compact);
        // A ring of pinned slots so the host can gather copy N+1 while the device still reads the
        // pinned buffer of copy N (a single shared buffer corrupts/faults -- and indeed did).
        // Depth is 8: more slots = fewer host waits = closer to the
        // no-wait ceiling, at `depth x span` bytes of pinned memory.
        // The ring is owned by the context (one context == one device), so its events and buffers
        // are destroyed with the context's streams.  It used to be a function-local static, which a
        // temporary backend could outlive: the loader's `-lm none` async uploader builds its own meta
        // backend (and CUDA contexts) and frees them after the load, leaving ring slots whose events
        // were recorded on now-destroyed streams -- the decode's next reuse then failed
        // `hipEventSynchronize` with `hipErrorUnknown`.
        auto & ring = cuda_ctx->splice_pin_ring;
        if (ring.empty()) {
            ring.resize(8, ggml_backend_cuda_context::splice_pin_slot{});
        }
        const int k = cuda_ctx->splice_pin_idx++ % (int) ring.size();
        auto & s = ring[k];
        if (need > s.sz) {
            if (s.ev != nullptr) { (void) hipEventSynchronize(s.ev); (void) hipEventDestroy(s.ev); s.ev = nullptr; }
            if (s.buf != nullptr) { (void) hipHostFree(s.buf); s.buf = nullptr; }
            CUDA_CHECK(hipHostMalloc(&s.buf, need, hipHostMallocDefault));
            CUDA_CHECK(hipEventCreateWithFlags(&s.ev, hipEventDisableTiming));
            s.sz = need;
        } else if (s.ev != nullptr) {
            CUDA_CHECK(hipEventSynchronize(s.ev));
        }
        // gather (modes 1..4); mode 5 returned above
        if (stride_tensor == size) {
            for (size_t e = 0; e < n_copies; ++e) {
                memcpy((char *) s.buf + e*size, (const char *) data + e*stride_data, size);
            }
        } else {
            memcpy(s.buf, data, full);
        }
        if (stride_tensor == size) {
            CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, s.buf, compact, cudaMemcpyHostToDevice, h2d_stream));
        } else {
            CUDA_CHECK(cudaMemcpy2DAsync(
                (char *) tensor->data + offset, stride_tensor, s.buf, stride_data, size, n_copies, cudaMemcpyHostToDevice, h2d_stream));
        }
        CUDA_CHECK(hipEventRecord(s.ev, h2d_stream));
        // order the consumer (the split's compute, on the main stream) after the queued H2D
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), s.ev, 0));
        return;
    }
    CUDA_CHECK(cudaMemcpy2DAsync(
        (char *) tensor->data + offset, stride_tensor, data, stride_data, size, n_copies, cudaMemcpyHostToDevice, cuda_ctx->stream()));
}

static void ggml_backend_cuda_get_tensor_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");
    ggml_cuda_async_input_drain(cuda_ctx->device);

    CUDA_CHECK(cudaMemcpy2DAsync(
        data, stride_data, (const char *) tensor->data + offset, stride_tensor, size, n_copies, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

// Staged cross-device copy (pinned host bounce).
//
// WHY THIS EXISTS: a `cudaMemcpyPeerAsync` into a movable-boundary-slab range only works when the driver
// granted that PEER device access to the mapping.  That grant is best effort and does get refused on a box
// whose slabs take nearly all VRAM (three ~29 GiB slabs: the peers have no free VRAM left to build a 29 GiB
// peer mapping, and `cuMemSetAccess` returns `invalid argument`).  The failure is not catchable: without
// peer access enabled HIP returns an error host-side, and with `GGML_CUDA_P2P=1` the copy is issued and the
// GPU FAULTS (`HSA_STATUS_ERROR_MEMORY_FAULT` in `__amd_rocclr_copyBuffer`, "Page not present"), which
// kills the process.  So the copy path asks `ggml_cuda_slab_peer_copy_ok` first and stages the copy through
// pinned host memory for the pairs the driver refused.
//
// One grow-only buffer and one done-event per (src, dst) pair.  The D2H runs on the source stream and the
// H2D on the DESTINATION stream, so the consumer that follows on that stream is ordered behind it by the
// stream itself; `done` guards a reuse of the buffer until the previous read has completed (an event wait on
// the source stream, not a host block).
struct ggml_cuda_peer_stage {
    void *      buf       = nullptr;
    size_t      size      = 0;
    cudaEvent_t d2h       = nullptr;   // D2H completed, recorded on the source stream
    cudaEvent_t done      = nullptr;   // H2D completed, recorded on the destination stream
    bool        have_done = false;
};

static std::mutex        g_peer_stage_mtx;
static ggml_cuda_peer_stage g_peer_stage[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_DEVICES];

static void ggml_cuda_peer_copy_staged(
        int src_device, cudaStream_t src_stream,
        int dst_device, cudaStream_t dst_stream,
        void * dst, const void * src, size_t nbytes) {
    if (nbytes == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_peer_stage_mtx);

    ggml_cuda_peer_stage & st = g_peer_stage[src_device][dst_device];

    if (st.buf == nullptr || st.size < nbytes) {
        if (st.buf != nullptr) {
            // the previous read of the staging buffer must have completed before it is handed back
            if (st.have_done) {
                ggml_cuda_set_device(dst_device);
                CUDA_CHECK(cudaEventSynchronize(st.done));
            }
            ggml_cuda_set_device(src_device);
            (void) cudaFreeHost(st.buf);
            st.buf  = nullptr;
            st.size = 0;
        }
        ggml_cuda_set_device(src_device);
        CUDA_CHECK(cudaMallocHost(&st.buf, nbytes));
        st.size = nbytes;
    }
    if (st.d2h == nullptr) {
        ggml_cuda_set_device(src_device);
        CUDA_CHECK(cudaEventCreateWithFlags(&st.d2h, cudaEventDisableTiming));
    }
    if (st.done == nullptr) {
        ggml_cuda_set_device(dst_device);
        CUDA_CHECK(cudaEventCreateWithFlags(&st.done, cudaEventDisableTiming));
    }

    ggml_cuda_set_device(src_device);
    if (st.have_done) {
        CUDA_CHECK(cudaStreamWaitEvent(src_stream, st.done, 0));
    }
    CUDA_CHECK(cudaMemcpyAsync(st.buf, src, nbytes, cudaMemcpyDeviceToHost, src_stream));
    CUDA_CHECK(cudaEventRecord(st.d2h, src_stream));

    ggml_cuda_set_device(dst_device);
    CUDA_CHECK(cudaStreamWaitEvent(dst_stream, st.d2h, 0));
    CUDA_CHECK(cudaMemcpyAsync(dst, st.buf, nbytes, cudaMemcpyHostToDevice, dst_stream));
    CUDA_CHECK(cudaEventRecord(st.done, dst_stream));
    st.have_done = true;

    ggml_cuda_set_device(src_device);
}

static bool ggml_backend_cuda_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    if (!ggml_backend_is_cuda(backend_src) || !ggml_backend_is_cuda(backend_dst)) {
        return false;
    }
    ggml_cuda_async_input_drain(((ggml_backend_cuda_context *) backend_src->context)->device);
    ggml_cuda_async_input_drain(((ggml_backend_cuda_context *) backend_dst->context)->device);

    if (!ggml_backend_buffer_is_cuda(buf_src) || !ggml_backend_buffer_is_cuda(buf_dst)) {
        return false;
    }

    // device -> device copy
    ggml_backend_cuda_context * cuda_ctx_src = (ggml_backend_cuda_context *) backend_src->context;
    ggml_backend_cuda_context * cuda_ctx_dst = (ggml_backend_cuda_context *) backend_dst->context;

    ggml_backend_cuda_buffer_context * buf_ctx_src = (ggml_backend_cuda_buffer_context *) buf_src->context;
    ggml_backend_cuda_buffer_context * buf_ctx_dst = (ggml_backend_cuda_buffer_context *) buf_dst->context;

    if (cuda_ctx_src->device != buf_ctx_src->device || cuda_ctx_dst->device != buf_ctx_dst->device) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: backend and buffer devices do not match\n", __func__);
#endif // NDEBUG
        return false;
    }

    // an async graph-input upload into src may still be in flight (GGML_CUDA_ASYNC_INPUT=1)
    ggml_cuda_async_input_wait(cuda_ctx_src->device, cuda_ctx_src->stream());

    if (backend_src != backend_dst) {
        // copy on src stream
        // compare the backing physical devices: distinct virtual devices may share one physical GPU,
        // in which case a same-device copy (not a peer copy) is required
        const int src_physical = ggml_cuda_get_physical_device(cuda_ctx_src->device);
        const int dst_physical = ggml_cuda_get_physical_device(cuda_ctx_dst->device);
        if (src_physical == dst_physical) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            // a peer copy the driver's grants do not cover faults the GPU, so ask first
            if (!ggml_cuda_slab_peer_copy_ok(dst->data, ggml_nbytes(dst))) {
                ggml_cuda_peer_copy_staged(cuda_ctx_src->device, cuda_ctx_src->stream(),
                                           cuda_ctx_dst->device, cuda_ctx_dst->stream(),
                                           dst->data, src->data, ggml_nbytes(dst));
            } else {
                CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_physical, src->data, src_physical, ggml_nbytes(dst), cuda_ctx_src->stream()));
            }
#endif // GGML_CUDA_NO_PEER_COPY
        }

        // record event on src stream after the copy
        if (!cuda_ctx_src->copy_event) {
            ggml_cuda_set_device(cuda_ctx_src->device);
            CUDA_CHECK(cudaEventCreateWithFlags(&cuda_ctx_src->copy_event, cudaEventDisableTiming));
        }

        CUDA_CHECK(cudaEventRecord(cuda_ctx_src->copy_event, cuda_ctx_src->stream()));

        // wait on dst stream for the copy to complete
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx_dst->stream(), cuda_ctx_src->copy_event, 0));
    } else {
        // src and dst are on the same backend
        CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
    }
    return true;
}

static void ggml_backend_cuda_synchronize(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;
    ggml_cuda_async_input_drain(cuda_ctx->device);

    CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));

    GGML_UNUSED(backend);
}

#ifdef USE_CUDA_GRAPH
static bool ggml_cuda_graph_debug_enabled() {
    static const bool enabled = ggml_cuda_env_flag("GGML_CUDA_GRAPH_DEBUG");
    return enabled;
}

static bool ggml_cuda_graph_check_compability(ggml_cgraph * cgraph) {

    // wip/moe-expert-cache: the slot-remap redirect is a HOST decision made in `ggml_cuda_mul_mat_id`,
    // which a CUDA graph only executes during CAPTURE.  The captured graph bakes in, per op, whether it
    // reads the compact arena or the full `input_cpy`, while the scheduler's per-token hook decides
    // whether to take the input over (and skip the `input_cpy` copy).  That is safe only if the decision
    // is CONSTANT for a given graph shape, so capture is blocked until the arena is sized.  After that
    // the hook always takes over whenever `slots >= n_used * n_tok` (the current token's experts are
    // protected from each other) and always declines below it - in both cases a per-shape constant.
    //
    // Only block a graph that ACTUALLY contains a cache-manageable routed MoE op.  A graph whose every
    // `MUL_MAT_ID` is above the decode band (e.g. a 16-sequence batched decode, `ne[2] == 16`) never
    // involves the cache, so its redirect decision is trivially constant and capture is safe - blocking
    // it left those workloads running graph-less (measured: `llama-batched-bench -npl 16` 137 -> 52 t/s).
    if (moe_cache_enabled() && !moe_cache_ready()) {
        for (int i = 0; i < cgraph->n_nodes; i++) {
            const ggml_tensor * n = cgraph->nodes[i];
            if (n->op == GGML_OP_MUL_MAT_ID && n->ne[2] <= MOE_EXPERT_CACHE_MAX_TOK) {
                return false;
            }
        }
    }

    bool use_cuda_graph = true;
    // Loop over nodes in GGML graph to obtain info needed for CUDA graph

    // The arch is a property of the device this graph runs on, not of the node, so read it once (and only
    // if a MUL_MAT_ID is found): `ggml_cuda_get_device()` is a cudaGetDevice() runtime call, and a decode
    // split carries several MUL_MAT_IDs.
    int cc = -1;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if (ggml_cuda_is_view_or_noop(node)) {
            continue;
        }

        // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
        if (node->op == GGML_OP_MUL_MAT_ID) {
            if (cc < 0) {
                cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
            }
            if (ggml_cuda_mul_mat_id_needs_sync(node, cc)) {
                // the mul_mat_id fallback path synchronizes the stream, so we cannot use CUDA graphs
                // ref: https://github.com/ggml-org/llama.cpp/pull/18958
                use_cuda_graph = false;
                if (ggml_cuda_graph_debug_enabled()) {
                    GGML_LOG_DEBUG("%s: disabling CUDA graphs due to unsupported node type\n", __func__);
                }
            }
        }

        if (!use_cuda_graph) {
            break;
        }
    }

    return use_cuda_graph;
}

// The batch token count a CUDA/HIP graph carries.  The first node's ne[1] is NOT a
// reliable token count: with expert offload (-ncmoe) the scheduler splits the graph
// around the CPU-resident experts, so a one-token decode split routinely starts with
// an expert-path tensor of shape [n_ff, n_expert_used, n_tokens] and ne[1] ==
// n_expert_used (10) even at one token.  Read it from the first op that actually
// carries it instead:
//   - MUL_MAT_ID  -> result is [n_out, n_expert_used, n_tokens], so ne[2] is n_tokens
//   - MUL_MAT     -> result is [src0->ne[1], src1->ne[1], ...], so src1->ne[1] is n_tokens
// The MUL_MAT arm requires a constant, unbatched weight (src0 is op NONE and 2-D) so the
// probe reads a layer's activation batch; attention score matmuls are skipped.
static int64_t ggml_cuda_graph_n_tokens(const ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * node = cgraph->nodes[i];

        if (node->op == GGML_OP_MUL_MAT_ID) {
            return node->ne[2];
        }
        if (node->op == GGML_OP_MUL_MAT && node->src[0] != nullptr && node->src[1] != nullptr &&
            node->src[0]->op == GGML_OP_NONE && node->src[0]->ne[2] == 1) {
            return node->src[1]->ne[1];
        }
    }

    // No weight matmul found in this split: fall back to the first node's second dim.
    return cgraph->n_nodes > 0 ? cgraph->nodes[0]->ne[1] : 0;
}

// One graph per (first node, token count): decode and each speculative verify width keep
// their own captured graph instead of invalidating one another's warmup.
//
// The first node's identity is its DATA pointer, not its `ggml_tensor` object address.  Under the
// Meta (tensor-split) backend every subgraph the scheduler hands a device is built from *simple
// tensors* that the meta re-creates on each rebuild (`ggml_backend_meta_graph_compute` resets the
// per-buffer `stc_compute` containers and maps every node to a fresh object), so the object address
// changes on every reallocation even when the graph is structurally identical and points at the same
// device buffer.  Keyed on the object address the cache therefore never warms: measured 3x R9700,
// Qwen3.8-27B Q8_0, `-sm tensor`, 300-token decode, 3954 graph keys / 1206 captures / 484 ms of
// `hipGraphInstantiate` (vs 33 keys / 8 captures / 15 ms on one device with `-sm none`, and 67 keys
// with `-sm layer`).  Keyed on `data` the same run takes 144 keys / 45 captures / 15 ms and decodes
// +41 % faster at 3 devices (+6 % at 2, unchanged at 1), with byte-identical text.  `data` is stable
// across a meta rebuild (it is the meta buffer's per-device slice) and it is what the captured graph
// actually bakes in.
//
// Two different graphs can only collide here if their first node resolves to the same data pointer
// AND the same token count; that is not trusted -- `ggml_cuda_graph_update_required` re-checks every
// node's op/type/shape/data pointers before a replay and recaptures on any difference, and a fresh
// `cgraph->uid` (the meta assigns one per rebuild) forces that check.  So a collision costs a
// recapture, never a mismatched replay.  `GGML_CUDA_GRAPH_KEY_DATA=0` restores the object-address
// key (A/B and bisect only).
//
// A first-node-data key alone is not enough on a *split* shape, and that is a correctness-of-speed
// issue, not a nicety: every layer's subgraph of a split graph can begin on the same activation buffer
// (the allocator reuses it: the subgraphs are structurally identical and their inputs do not overlap in
// lifetime), so one cache entry served all of them and each layer was compared against the *previous
// layer's* properties -- a source `data` pointer that "moved" by one layer's worth of weights.  The
// graph was then either run directly or recaptured on almost every call.  Measured 2x R9700 (x8),
// Flash-Next IQ4_NL, `-sm tensor -ncmoe 48`, 2000-token decode: `direct=549499 replay=31696
// capture=149 properties_changed=452439` (37.7 t/s, and `GGML_CUDA_DISABLE_GRAPHS=1` the same 37.3, i.e.
// the captures were worth nothing) -> `direct=3882 replay=574826 capture=2636 properties_changed=28`
// (**42.8 t/s, +13.5 %**).  The same shape reproduced at `direct=94659 ... properties_changed=78215` on
// ONE device (35B-A3B, a much smaller win there), so it is not a multi-device-only defect.
//
// A layer's weights are fixed and distinct, so they tell the subgraphs apart.  `GGML_CUDA_GRAPH_KEY_WEIGHT=0`
// restores the (first node, token count) key (A/B and bisect only).
static bool ggml_cuda_graph_key_by_weight() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_GRAPH_KEY_WEIGHT");
        return e == nullptr || atoi(e) != 0;
    }();
    return on;
}

// The data of the first device-resident weight any node of the subgraph reads.  Under a split mode the
// layers' subgraphs share their first activation buffer but never their weights.
static const void * ggml_cuda_graph_first_weight(const ggml_cgraph * cgraph) {
    if (!ggml_cuda_graph_key_by_weight()) {
        return nullptr;
    }
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * node = cgraph->nodes[i];
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            const ggml_tensor * src = node->src[j];
            if (src != nullptr && src->buffer != nullptr && src->data != nullptr &&
                    ggml_backend_buffer_get_usage(src->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                return src->data;
            }
        }
    }
    // No device-resident weight (e.g. a routed-MoE split whose expert table lives in host memory): the
    // first node's own first source is the next most layer-specific pointer (the expert table, or the
    // arena that backs it).
    const ggml_tensor * n0 = cgraph->n_nodes > 0 ? cgraph->nodes[0] : nullptr;
    return n0 != nullptr && n0->src[0] != nullptr ? n0->src[0]->data : nullptr;
}

// A cheap structural signature for the key, also gated by GGML_CUDA_GRAPH_KEY_WEIGHT: two subgraphs of one
// layer can share the first node's data AND the first weight (e.g. the two expert-width MoE splits of a
// host-expert layer, [384, n_used, n_tok] and [256, n_used, n_tok], both starting on the same output
// buffer) and would evict each other every step without it.  The node names carry the layer index
// ("ffn_moe_gate-5"), so they separate the routed-MoE splits of different layers whose expert arena is
// shared.
static uint64_t ggml_cuda_graph_shape_sig(const ggml_cgraph * cgraph) {
    if (!ggml_cuda_graph_key_by_weight() || cgraph->n_nodes <= 0) {
        return 0;
    }
    const ggml_tensor * n0 = cgraph->nodes[0];
    const ggml_tensor * nl = cgraph->nodes[cgraph->n_nodes - 1];
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](uint64_t v) { h ^= v; h *= 1099511628211ull; };
    mix((uint64_t) cgraph->n_nodes);
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        mix((uint64_t) n0->ne[d]);
    }
    mix((uint64_t) (uintptr_t) (n0->src[0] != nullptr ? n0->src[0]->data : nullptr));
    mix((uint64_t) (uintptr_t) nl->data);
    for (const char * c = n0->name; c < n0->name + GGML_MAX_NAME && *c; ++c) {
        mix((uint64_t) (unsigned char) *c);
    }
    for (const char * c = nl->name; c < nl->name + GGML_MAX_NAME && *c; ++c) {
        mix((uint64_t) (unsigned char) *c);
    }
    return h;
}

static ggml_cuda_graph_key ggml_cuda_graph_get_key(ggml_cgraph * cgraph) {
    static const bool key_by_data = [] {
        const char * e = getenv("GGML_CUDA_GRAPH_KEY_DATA");
        return e == nullptr || atoi(e) != 0;
    }();
    if (key_by_data && cgraph->n_nodes > 0 && cgraph->nodes[0] != nullptr &&
            cgraph->nodes[0]->data != nullptr) {
        // only the pointer values are used (hash / equality); they are never dereferenced as tensors
        return ggml_cuda_graph_key { cgraph->nodes[0]->data, ggml_cuda_graph_n_tokens(cgraph),
                                     ggml_cuda_graph_first_weight(cgraph), ggml_cuda_graph_shape_sig(cgraph) };
    }
    return ggml_cuda_graph_key { cgraph->nodes[0], ggml_cuda_graph_n_tokens(cgraph) };
}

// Whether a CUDA/HIP graph should skip the graph-capture path.  Only a true PRE-FILL (a
// varying ubatch, where capture never amortises) is skipped; single-token decode and the
// spec-verify widths (n_tokens <= MMVQ_MAX_BATCH_SIZE, a fixed shape every step) keep HIP
// graph replay.  GGML_CUDA_DISABLE_VERIFY_GRAPHS=1 restores the pre-r20 behaviour where
// every multi-token graph skipped the graph path.
// Diagnostic trace for the graph capture state machine (GGML_CUDA_GRAPH_TRACE=1).  Off by default.
static bool ggml_cuda_graph_trace() {
    static const bool on = getenv("GGML_CUDA_GRAPH_TRACE") != nullptr;
    return on;
}

static bool ggml_cuda_graph_is_multi_token(const ggml_cgraph * cgraph) {
    static const bool verify_graphs_off = getenv("GGML_CUDA_DISABLE_VERIFY_GRAPHS") != nullptr;
    return ggml_cuda_graph_n_tokens(cgraph) > (verify_graphs_off ? 1 : MMVQ_MAX_BATCH_SIZE);
}

// `GGML_CUDA_GRAPH_TRACE` also reports how each `graph_compute` call was served.  The distinction is
// the only way to tell "graphs are on" from "graphs are captured and then never replayed": a healthy
// decode replays for essentially every call, while a large `direct` with a matching `propchg` means the
// replay identity is churning (see `ggml_cuda_graph_update_required`).  Measured on 2x R9700,
// Flash-Next IQ4_NL, `-sm tensor -ncmoe 48`: direct=56171 replay=3152 capture=21 propchg=55321 over 200
// tokens, i.e. 93 % of calls ran the node loop directly and `GGML_CUDA_DISABLE_GRAPHS=1` measured the
// same speed.  Counters only; they cost two increments per call.
static int64_t g_gc_n_direct = 0, g_gc_n_replay = 0, g_gc_n_capture = 0, g_gc_n_propchg = 0;

static void ggml_cuda_graph_stats_report() {
    fprintf(stderr, "%s: graph_compute calls: direct=%lld replay=%lld capture=%lld properties_changed=%lld\n",
            __func__, (long long) g_gc_n_direct, (long long) g_gc_n_replay, (long long) g_gc_n_capture,
            (long long) g_gc_n_propchg);
}

static void ggml_cuda_graph_stats_arm() {
    // Read the env directly rather than through `ggml_cuda_graph_trace()`: that helper is defined far
    // below this point in the file (a later block moves it), and this block must compile on its own.
    static const bool armed = [] {
        const char * e = getenv("GGML_CUDA_GRAPH_TRACE");
        if (e != nullptr && e[0] != '\0') {
            atexit(ggml_cuda_graph_stats_report);
        }
        return true;
    }();
    (void) armed;
}

// Does the stored property set still describe `cgraph` exactly?  This is a replay's only guard: it
// compares everything a capture baked into its executable graph -- the node's op/type/op_params/geometry/
// data/flags (and the `src[]`/`view_src`/`buffer` object identities), then each source's data pointer and
// geometry.  It reads the live tensors in place rather than assembling a temporary per node, and stops at
// the first difference: on a healthy decode it is the only per-node work `graph_compute` does.
//
// The source run is looked up by source INDEX, so it is equivalent to the old fixed GGML_MAX_SRC compare:
// the node-header compare has already established that the null/non-null layout of `src[]` is unchanged.
static bool ggml_cuda_graph_props_match(const ggml_cuda_graph & graph, const ggml_cgraph * cgraph) {
    if ((int) graph.prop_nodes.size() != cgraph->n_nodes) {
        return false;
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (memcmp(&graph.prop_nodes[i], node, sizeof(ggml_tensor)) != 0) {
            return false;
        }

        uint32_t k = graph.prop_src_off[i];
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            const ggml_tensor * src = node->src[j];
            if (src == nullptr) {
                continue;
            }
            GGML_ASSERT(k < graph.prop_src_off[i + 1]);
            const ggml_cuda_graph::src_properties & p = graph.prop_srcs[k++];
            if (p.data != src->data ||
                memcmp(p.ne, src->ne, sizeof(p.ne)) != 0 ||
                memcmp(p.nb, src->nb, sizeof(p.nb)) != 0) {
                return false;
            }
        }
        GGML_ASSERT(k == graph.prop_src_off[i + 1]);
    }

    return true;
}

// Record `cgraph` as the graph's replay identity.  Only the recapture path pays for this; the vectors keep
// their capacity across calls, so a churning shape does not re-allocate every step.
static void ggml_cuda_graph_props_store(ggml_cuda_graph & graph, const ggml_cgraph * cgraph) {
    graph.prop_nodes.resize(cgraph->n_nodes);
    graph.prop_src_off.resize((size_t) cgraph->n_nodes + 1);
    graph.prop_srcs.clear();

    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * node = cgraph->nodes[i];
        memcpy(&graph.prop_nodes[i], node, sizeof(ggml_tensor));
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            const ggml_tensor * src = node->src[j];
            if (src == nullptr) {
                continue;
            }
            ggml_cuda_graph::src_properties p = {};
            p.data = src->data;
            memcpy(p.ne, src->ne, sizeof(p.ne));
            memcpy(p.nb, src->nb, sizeof(p.nb));
            graph.prop_srcs.push_back(p);
        }
        graph.prop_src_off[i + 1] = (uint32_t) graph.prop_srcs.size();
    }
}

// `graph` is the entry `ggml_cuda_graph_get_key(cgraph)` resolves to, resolved ONCE per graph_compute by
// the caller: the key computation walks the subgraph (a virtual `get_usage` call per source), and it used
// to be repeated here and in the graph_optimize hook for the same graph.
static bool ggml_cuda_graph_update_required(ggml_cuda_graph * graph, ggml_cgraph * cgraph) {
    bool res = false;

    if (cgraph->uid != 0 &&
        cgraph->uid == graph->uid) {
        if (ggml_cuda_graph_debug_enabled()) {
            GGML_LOG_DEBUG("CUDA Graph id %zu reused\n", cgraph->uid);
        }
        GGML_ASSERT((int)graph->prop_nodes.size() == cgraph->n_nodes);
        return false;
    }

    graph->uid = cgraph->uid;

    // A difference anywhere means "recapture", so the store is rewritten from the top rather than node by
    // node.  `had` keeps the counter's meaning: a difference on a graph we already had is a refused replay;
    // a first sight of a shape is not.
    const bool had = (int) graph->prop_nodes.size() == cgraph->n_nodes;
    if (!ggml_cuda_graph_props_match(*graph, cgraph)) {
        if (had) { g_gc_n_propchg++; }
        ggml_cuda_graph_props_store(*graph, cgraph);
        res = true;
    }

    return res;
}

static void ggml_cuda_graph_update_executable(cudaGraphExec_t & instance, cudaGraph_t cuda_graph) {
#ifdef GGML_USE_HIP
    // HIP/ROCm <= 10.0 leaks device memory in hipGraphExecUpdate: the driver's
    // GraphKernelArgManager bump-allocates a fresh kernel-argument slot on every update
    // and only reclaims slots when the exec is destroyed (ROCm/rocm-systems#10713; driver
    // fix in PR #11434).  A split-moe decode recaptures the graph on the order of once per
    // few tokens, so a long-lived exec grows by a few KB per token.  Destroying and
    // re-instantiating is the only reclaim path that exists today and was measured flat
    // over a soak; it also cannot leave a stale executable behind, which the update path
    // can silently do when the driver drops an error.  Cheap in practice: this runs only
    // on the recapture path, not per token.  Set GGML_HIP_GRAPH_FORCE_UPDATE=1 to take the
    // update path anyway (e.g. on a ROCm that has the driver fix).
    static const bool force_update = getenv("GGML_HIP_GRAPH_FORCE_UPDATE") != nullptr;
    if (!force_update) {
        if (instance != nullptr) {
            CUDA_CHECK(cudaGraphExecDestroy(instance));
            instance = nullptr;
        }
        CUDA_CHECK(cudaGraphInstantiate(&instance, cuda_graph, nullptr, nullptr, 0));
        return;
    }
#endif // GGML_USE_HIP

#if CUDART_VERSION >= 12000
    cudaGraphExecUpdateResultInfo result_info;
    cudaError_t stat = cudaGraphExecUpdate(instance, cuda_graph, &result_info);
#else
    cudaGraphNode_t errorNode;
    cudaGraphExecUpdateResult result_info;
    cudaError_t stat = cudaGraphExecUpdate(instance, cuda_graph, &errorNode, &result_info);
#endif // CUDART_VERSION >= 12000

    if (stat == cudaErrorGraphExecUpdateFailure) {
        if (ggml_cuda_graph_debug_enabled()) {
            GGML_LOG_DEBUG("%s: CUDA graph update failed\n", __func__);
        }

        // The pre-existing graph exec cannot be updated due to violated constraints
        // so instead clear error and re-instantiate
        (void)cudaGetLastError();
        CUDA_CHECK(cudaGraphExecDestroy(instance));
        instance = nullptr;
        CUDA_CHECK(cudaGraphInstantiate(&instance, cuda_graph, nullptr, nullptr, 0));
    } else {
        GGML_ASSERT(stat == cudaSuccess);
    }
}

// number of ggml nodes per CUDA graph chunk, 0 captures the whole ggml graph as one CUDA graph
static int ggml_cuda_graph_chunk_size() {
    static const int chunk_size = [] {
        const char * env = getenv("GGML_CUDA_GRAPH_CHUNK_SIZE");
        return env != nullptr ? atoi(env) : 512;
    }();
    return chunk_size;
}

// end the capture of chunk `idx`, update or create its executable and launch it
static void ggml_cuda_graph_launch_chunk(ggml_backend_cuda_context * cuda_ctx, ggml_cuda_graph * graph, size_t idx) {
    cudaGraph_t cuda_graph = nullptr;
    CUDA_CHECK(cudaStreamEndCapture(cuda_ctx->stream(), &cuda_graph));

    if (idx < graph->instances.size()) {
        ggml_cuda_graph_update_executable(graph->instances[idx], cuda_graph);
    } else {
        GGML_ASSERT(idx == graph->instances.size());
        graph->instances.emplace_back(nullptr);
        CUDA_CHECK(cudaGraphInstantiate(&graph->instances[idx], cuda_graph, nullptr, nullptr, 0));
    }
    CUDA_CHECK(cudaGraphDestroy(cuda_graph));

    CUDA_CHECK(cudaGraphLaunch(graph->instances[idx], cuda_ctx->stream()));
}
#endif // USE_CUDA_GRAPH


static void ggml_cuda_graph_evaluate_and_capture(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const bool use_cuda_graph, const bool cuda_graph_update_required, ggml_cuda_graph * graph) {
    int64_t t_ev = g_cgc_on ? ggml_time_us() : 0;
    bool graph_evaluated_or_captured = false;

    // per-op timing instrumentation (env-gated, diagnostic only)
    const bool op_timing = g_op_timing_on;
    std::vector<cudaEvent_t> op_ev0;
    std::vector<cudaEvent_t> op_ev1;
    std::vector<std::tuple<const ggml_tensor *, int, bool>> op_nodes;
    if (op_timing) {
        op_ev0.resize(cgraph->n_nodes);
        op_ev1.resize(cgraph->n_nodes);
        op_nodes.reserve(cgraph->n_nodes);
        for (int i = 0; i < cgraph->n_nodes; i++) {
#ifdef GGML_USE_HIP
            CUDA_CHECK(cudaEventCreateWithFlags(&op_ev0[i], hipEventDefault));
            CUDA_CHECK(cudaEventCreateWithFlags(&op_ev1[i], hipEventDefault));
#else
            CUDA_CHECK(cudaEventCreateWithFlags(&op_ev0[i], cudaEventDefault));
            CUDA_CHECK(cudaEventCreateWithFlags(&op_ev1[i], cudaEventDefault));
#endif
        }
    }

    // flag used to determine whether it is an integrated_gpu
    const bool integrated            = ggml_cuda_info().devices[cuda_ctx->device].integrated;

    ggml_cuda_stream_context & stream_ctx = cuda_ctx->stream_context();
    bool                         is_concurrent_event_active = false;
    ggml_cuda_concurrent_event * concurrent_event           = nullptr;
    bool                         should_launch_concurrent_events = false;

#ifdef USE_CUDA_GRAPH
    // resolved by the caller from the graph key it already computed; null in a build without graph support
    // or when this call runs the node loop directly
    graph = use_cuda_graph ? graph : nullptr;
    const int         chunk_size = ggml_cuda_graph_chunk_size();
    int               chunk_end  = chunk_size;
    size_t            n_chunks   = 0;
#endif // USE_CUDA_GRAPH

    const auto try_launch_concurrent_event = [&](const ggml_tensor * node) {
        if (stream_ctx.concurrent_events.find(node) != stream_ctx.concurrent_events.end()) {
            concurrent_event = &stream_ctx.concurrent_events[node];

            is_concurrent_event_active = true;

            GGML_LOG_DEBUG("Launching %d streams at %s\n", concurrent_event->n_streams, node->name);

            cudaStream_t main_stream = cuda_ctx->stream();  // this should be stream 0
            GGML_ASSERT(cuda_ctx->curr_stream_no == 0);
            CUDA_CHECK(cudaEventRecord(concurrent_event->fork_event, main_stream));

            for (int i = 1; i <= concurrent_event->n_streams; ++i) {
                cudaStream_t stream = cuda_ctx->stream(cuda_ctx->device, i);
                CUDA_CHECK(cudaStreamWaitEvent(stream, concurrent_event->fork_event));
            }
        }
    };

    while (!graph_evaluated_or_captured) {
        // Only perform the graph execution if CUDA graphs are not enabled, or we are capturing the graph.
        // With the use of CUDA graphs, the execution will be performed by the graph launch.
        if (!use_cuda_graph || cuda_graph_update_required) {
            [[maybe_unused]] int prev_i = 0;

            if (stream_ctx.concurrent_events.size() > 0) {
                should_launch_concurrent_events = true;
                for (const auto & [tensor, event] : stream_ctx.concurrent_events) {
                    should_launch_concurrent_events = should_launch_concurrent_events && event.is_valid();
                }
            }

            if (should_launch_concurrent_events) {
                // Restore original node order within each concurrent region to enable fusion within streams

                std::unordered_map<const ggml_tensor *, int> node_to_idx;
                node_to_idx.reserve(cgraph->n_nodes);
                for (int i = 0; i < cgraph->n_nodes; ++i) {
                    node_to_idx[cgraph->nodes[i]] = i;
                }

                for (auto & [fork_node, event] : stream_ctx.concurrent_events) {
                    // Find positions of all nodes from this event in the current graph
                    std::vector<int> positions;
                    positions.reserve(event.original_order.size());

                    bool all_found = true;
                    for (const ggml_tensor * orig_node : event.original_order) {
                        auto it = node_to_idx.find(orig_node);
                        if (it != node_to_idx.end()) {
                            positions.push_back(it->second);
                        } else {
                            all_found = false;
                            break;
                        }
                    }

                    if (!all_found || positions.size() != event.original_order.size()) {
                        continue;
                    }

                    // Sort positions to get contiguous range
                    std::vector<int> sorted_positions = positions;
                    std::sort(sorted_positions.begin(), sorted_positions.end());

                    bool is_contiguous = true;
                    for (size_t i = 1; i < sorted_positions.size(); ++i) {
                        if (sorted_positions[i] != sorted_positions[i-1] + 1) {
                            is_contiguous = false;
                            break;
                        }
                    }

                    if (!is_contiguous) {
                        continue;
                    }

                    // Restore original order at the sorted positions
                    int start_pos = sorted_positions[0];
                    for (size_t i = 0; i < event.original_order.size(); ++i) {
                        cgraph->nodes[start_pos + i] = const_cast<ggml_tensor *>(event.original_order[i]);
                    }
                }
            } else {
                stream_ctx.concurrent_events.clear();
            }

            if (t_ev) { g_ev_pre_us += ggml_time_us() - t_ev; g_loop_start = ggml_time_us(); }
            static const bool fuse_gdn_state_gather =
                (getenv("GGML_CUDA_FUSE_GDN_STATE_GATHER") == nullptr || atoi(getenv("GGML_CUDA_FUSE_GDN_STATE_GATHER")) != 0) &&
                (getenv("GGML_CUDA_DISABLE_FUSION") == nullptr || atoi(getenv("GGML_CUDA_DISABLE_FUSION")) == 0);
            // GGML_CUDA_GDN_GATHER_INDEX=0: per-gather walk instead of the per-graph index (A/B);
            // =2: both, abort on a different answer
            static const int gdn_gather_index_mode =
                getenv("GGML_CUDA_GDN_GATHER_INDEX") ? atoi(getenv("GGML_CUDA_GDN_GATHER_INDEX")) : 1;
            ggml_cuda_gdn_gather_index gdn_gather_index;
            cuda_ctx->gdn_deferred_gathers.clear();
            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
#ifdef USE_CUDA_GRAPH
                // fused nodes are skipped together, so a cut never splits a fusion
                // a concurrent region must join before its capture can end
                if (use_cuda_graph && chunk_size > 0 && i >= chunk_end && !is_concurrent_event_active) {
                    ggml_cuda_graph_launch_chunk(cuda_ctx, graph, n_chunks++);
                    CUDA_CHECK(cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed));
                    chunk_end = i + chunk_size;
                }
#endif // USE_CUDA_GRAPH
                if (is_concurrent_event_active) {
                    GGML_ASSERT(concurrent_event);

                    if (node == concurrent_event->join_node) {
                        cuda_ctx->curr_stream_no = 0;
                        for (int i = 1; i <= concurrent_event->n_streams; ++i) {
                            // Wait on join events of forked streams in the main stream
                            CUDA_CHECK(cudaEventRecord(concurrent_event->join_events[i - 1],
                                                       cuda_ctx->stream(cuda_ctx->device, i)));
                            CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), concurrent_event->join_events[i - 1]));
                        }

                        is_concurrent_event_active = false;
                        concurrent_event           = nullptr;
                    } else {
                        GGML_ASSERT (concurrent_event->stream_mapping.find(node) != concurrent_event->stream_mapping.end());
                        cuda_ctx->curr_stream_no = concurrent_event->stream_mapping[node];
                        GGML_LOG_DEBUG("Setting stream no to %d for node %s\n", cuda_ctx->curr_stream_no, node->name);
                    }
                } else if (i - prev_i > 1) {
                    //the previous node was fused
                    const ggml_tensor * prev_node = cgraph->nodes[i - 1];
                    try_launch_concurrent_event(prev_node);

                    if (is_concurrent_event_active) {
                        cuda_ctx->curr_stream_no = concurrent_event->stream_mapping[node];
                        GGML_LOG_DEBUG("Setting stream no to %d for node %s\n", cuda_ctx->curr_stream_no, node->name);
                    }
                }

                prev_i = i;

                if (ggml_cuda_is_view_or_noop(node)) {
                    continue;
                }

                if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                    continue;
                }

                if (fuse_gdn_state_gather && node->op == GGML_OP_GET_ROWS &&
                    ggml_cuda_gdn_gather_candidate(cgraph, i) != nullptr) {
                    bool deferrable;
                    if (gdn_gather_index_mode == 0) {
                        deferrable = ggml_cuda_gdn_state_gather_deferrable(cgraph, i);
                    } else {
                        if (!gdn_gather_index.built) {
                            ggml_cuda_gdn_gather_index_build(cgraph, gdn_gather_index);
                        }
                        deferrable = ggml_cuda_gdn_state_gather_deferrable(cgraph, i, gdn_gather_index);
                        if (gdn_gather_index_mode == 2 && deferrable != ggml_cuda_gdn_state_gather_deferrable(cgraph, i)) {
                            GGML_ABORT("GDN gather index mismatch at node %d (%s): index %d, walk %d",
                                       i, node->name, (int) deferrable, (int) !deferrable);
                        }
                    }
                    if (!deferrable && gdn_gather_index_mode != 0) {
                        deferrable = ggml_cuda_gdn_conv_gather_deferrable(cgraph, i, gdn_gather_index);
                    }
                    if (deferrable) {
                        cuda_ctx->gdn_deferred_gathers.push_back(node);
                        continue;
                    }
                }

                if (op_timing) {
                    // bracket the fusion dispatch: a matching try_fuse launches its own
                    // fused kernel and the per-node events below are skipped
                    CUDA_CHECK(cudaEventRecord(op_ev0[i], cuda_ctx->stream()));
                }

                const int64_t t_f = g_cgc_on ? ggml_time_us() : 0;
                int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);
                if (t_f) { g_fuse_us += ggml_time_us() - t_f; g_fuse_calls++; }

                if (nodes_to_skip != 0) {
                    if (op_timing) {
                        CUDA_CHECK(cudaEventRecord(op_ev1[i], cuda_ctx->stream()));
                        op_nodes.emplace_back(node, i, true);
                    }
#ifdef GGML_CUDA_DEBUG
                    const int last_fused = i + nodes_to_skip;
                    GGML_LOG_INFO("nodes_fused: %d, first: %s (%s), last: %s (%s)\n",
                            nodes_to_skip + 1, ggml_op_name(node->op), node->name,
                            ggml_op_name(cgraph->nodes[last_fused]->op), cgraph->nodes[last_fused]->name);
#endif
                    i += nodes_to_skip;
                    continue;
                }

#ifndef NDEBUG
                // On integrated GPUs (APUs, e.g. RDNA3.5) the scheduler may place a
                // node's output on the host-visible buffer, which the compute path
                // handles. Allow that here, mirroring the src-tensor check below.
                assert(node->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) ||
                       (integrated && ggml_backend_buft_is_cuda_host(node->buffer->buft)));
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    if (node->src[j] != nullptr) {
                        assert(node->src[j]->buffer);
                        assert(node->src[j]->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) ||
                               (integrated && ggml_backend_buft_is_cuda_host(node->src[j]->buffer->buft)));
                    }
                }
#else
                GGML_UNUSED(integrated);
#endif  // NDEBUG

                if (op_timing) {
                    CUDA_CHECK(cudaEventRecord(op_ev0[i], cuda_ctx->stream()));
                }

                const int64_t t_fw = g_cgc_on ? ggml_time_us() : 0;
                bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);
                if (t_fw) { const int64_t d = ggml_time_us() - t_fw; g_fwd_us += d; g_fwd_calls++;
                            const int o = (int) node->op; if (o >= 0 && o < GGML_OP_COUNT) { g_opus[o] += d; g_opn[o]++; } }
                if (!ok) {
                    GGML_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
                }
                GGML_ASSERT(ok);
                if (op_timing) {
                    CUDA_CHECK(cudaEventRecord(op_ev1[i], cuda_ctx->stream()));
                    op_nodes.emplace_back(node, i, false);
                }

                if (!is_concurrent_event_active) {
                    try_launch_concurrent_event(node);
               }
            }
        }

#ifdef USE_CUDA_GRAPH
        if (use_cuda_graph && cuda_graph_update_required) { // End CUDA graph capture
            // WIP fused-stage: capture the AR stage kernel as the LAST node of
            // the LAST chunk so each device's wire staging + arrival token are
            // ready at subgraph-end instead of after the separate AR kernel's
            // dispatch (which carries a per-device graph->kernel premium).
            // No-op when the internal AR pipeline isn't in fused mode.
            if (cgraph->n_nodes > 0 && cgraph->nodes[cgraph->n_nodes-1] != nullptr) {
                const ggml_tensor * last = cgraph->nodes[cgraph->n_nodes-1];
                ggml_cuda_ar_stage_hook_run(cuda_ctx->device, cuda_ctx->stream(),
                                            static_cast<const float *>(last->data),
                                            ggml_nelements(last));
            }

            // ends the capture of this (last) chunk, then updates/instantiates and launches it
            ggml_cuda_graph_launch_chunk(cuda_ctx, graph, n_chunks++);

            // drop the chunks left over from a longer graph
            for (size_t i = n_chunks; i < graph->instances.size(); ++i) {
                CUDA_CHECK(cudaGraphExecDestroy(graph->instances[i]));
            }
            graph->instances.resize(n_chunks);
            graph_evaluated_or_captured = true; // CUDA graph has been captured

            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            if (ggml_cuda_lock_counter.fetch_sub(1, std::memory_order_relaxed) == 1) {
                ggml_cuda_lock_cv.notify_all();
            }
        } else {
            graph_evaluated_or_captured = true; // ggml graph has been directly evaluated
        }
    }

    if (use_cuda_graph && !cuda_graph_update_required) {
        // the chunks were launched during capture, otherwise launch them here
        for (cudaGraphExec_t instance : graph->instances) {
            CUDA_CHECK(cudaGraphLaunch(instance, cuda_ctx->stream()));
        }
#else
        GGML_UNUSED(graph);
        graph_evaluated_or_captured = true;
#endif  // USE_CUDA_GRAPH
    }

    if (op_timing) {
        CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));
        static std::map<std::string, double> op_ms_total;
        static std::map<std::string, int>    op_cnt_total;
        std::map<std::string, double> op_ms;
        std::map<std::string, int>    op_cnt;
        for (const auto & [node, idx, fused] : op_nodes) {
            float ms = 0.0f;
#ifdef GGML_USE_HIP
            CUDA_CHECK(hipEventElapsedTime(&ms, (hipEvent_t) op_ev0[idx], (hipEvent_t) op_ev1[idx]));
#else
            CUDA_CHECK(cudaEventElapsedTime(&ms, op_ev0[idx], op_ev1[idx]));
#endif
            std::string key = fused ? "FUSED " : "";
            key += ggml_op_name(node->op);
            key += " ";
            key += node->name;
            if (node->op == GGML_OP_MUL_MAT && node->src[0] != nullptr && node->src[1] != nullptr) {
                char buf[64];
                snprintf(buf, sizeof(buf), " [%lldx%lldx%lld]",
                         (long long) node->src[0]->ne[0], (long long) node->src[0]->ne[1],
                         (long long) node->src[1]->ne[1]);
                key += buf;
            }
            op_ms[key] += ms;
            op_cnt[key]++;
            op_ms_total[key] += ms;
            op_cnt_total[key]++;
        }
        std::vector<std::pair<std::string, double>> sorted(op_ms.begin(), op_ms.end());
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto & a, const auto & b) { return a.second > b.second; });
        double total = 0.0;
        for (const auto & [k, v] : sorted) {
            total += v;
        }
        GGML_LOG_INFO("%s: op timing: total %.2f ms over %zu nodes:\n", __func__, total, op_nodes.size());
        for (const auto & [k, v] : sorted) {
            GGML_LOG_INFO("  %8.3f ms %5.1f%%  x%-4d %s\n", v, 100.0 * v / total, op_cnt[k], k.c_str());
        }
        GGML_LOG_INFO("%s: op timing cumulative: %.2f ms over %d nodes\n", __func__,
                      std::accumulate(op_ms_total.begin(), op_ms_total.end(), 0.0,
                                      [](double acc, const auto & p) { return acc + p.second; }),
                      std::accumulate(op_cnt_total.begin(), op_cnt_total.end(), 0,
                                      [](int acc, const auto & p) { return acc + p.second; }));
        for (cudaEvent_t e : op_ev0) {
            CUDA_CHECK(cudaEventDestroy(e));
        }
        for (cudaEvent_t e : op_ev1) {
            CUDA_CHECK(cudaEventDestroy(e));
        }
    }
}

#ifdef USE_CUDA_GRAPH
static bool ggml_cuda_graph_set_enabled(ggml_backend_cuda_context * cuda_ctx, ggml_cuda_graph * graph) {
    if (graph->instances.empty()) {
        if (ggml_cuda_info().devices[cuda_ctx->device].cc < GGML_CUDA_CC_VOLTA) {
            if (!graph->disable_due_to_gpu_arch && ggml_cuda_graph_debug_enabled()) {
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to GPU architecture\n", __func__);
            }
            graph->disable_due_to_gpu_arch = true;
        }
    }

    return graph->is_enabled();
}
#endif // USE_CUDA_GRAPH



static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    const cuda_gc_timer cuda_gc_timer_inst(cgraph);
    int64_t t_gc = g_cgc_on ? ggml_time_us() : 0;
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    ggml_cuda_set_device(cuda_ctx->device);
    ggml_cuda_async_input_wait(cuda_ctx->device, cuda_ctx->stream());
    const ggml_cuda_async_input_done_guard async_input_done { cuda_ctx->device, cuda_ctx->stream() };

    if (g_stream_dbg_on) {
        static int n = 0;
        if (n++ < 24) fprintf(stderr, "STREAMDBG compute dev=%d stream=%p first=%s nodes=%d\n",
            cuda_ctx->device, (void *) cuda_ctx->stream(), cgraph->n_nodes ? cgraph->nodes[0]->name : "-", cgraph->n_nodes);
    }

    // The Q8_1 input cache is only valid within one graph execution.
    cuda_ctx->q8_1_cache_clear();
    cuda_ctx->mwr_done_graph = nullptr;
    cuda_ctx->bf16_src1_cache_clear();
    ggml_cuda_mmb_set_active_ctx(cuda_ctx);
    ggml_cuda_mmb_begin_graph();

    // wip/slab-transient-absorption (step 2b-2): the two per-graph activation caches must not OUTLIVE the
    // graph.  They are cleared at the start of the next compute anyway, so releasing them at scope exit (on
    // every return path) loses no hit the next graph would have made -- and it means no MMB/BF16 pool block
    // is live ACROSS a graph boundary, which is what makes them safe to serve from the slab's wide band: the
    // band is released at the prefill -> decode drop, i.e. between graphs, and a block still live there would
    // be a silent use-after-free (the `POOL_WIDE_LEAK` tripwire in `ggml_cuda_slab_ring_set`).
    struct per_graph_cache_yield {
        ggml_backend_cuda_context * c;
        ~per_graph_cache_yield() {
            ggml_cuda_mmb_yield_caches(c);
            c->bf16_src1_cache_clear();
        }
    } per_graph_cache_yield_inst{cuda_ctx};

    bool use_cuda_graph             = false;
    bool cuda_graph_update_required = false;
    ggml_cuda_graph_key graph_key = {};
    // the entry `graph_key` resolves to, looked up ONCE per call: the capture trigger below and
    // `ggml_cuda_graph_evaluate_and_capture` used to look it up again
    ggml_cuda_graph * graph = nullptr;

    // op timing instruments each node with stream events, which is not possible during capture
    const bool op_timing = g_op_timing_on;

#ifdef USE_CUDA_GRAPH
    graph_key = ggml_cuda_graph_get_key(cgraph);

    graph = cuda_ctx->cuda_graph(graph_key);
    ggml_cuda_graph_set_enabled(cuda_ctx, graph);

    if (!op_timing && graph->is_enabled()) {
        const bool graph_compatible = ggml_cuda_graph_check_compability(cgraph);
        if (graph_compatible) {
            // PRE-FILL graphs use varying ubatch sizes, so each is a separate graph
            // key and CUDA-graph capture never amortizes: the per-call update_required
            // probe + failed capture is pure overhead. Measured pp512 is ~6.7% faster
            // with graphs OFF. Only single-token decode (stable shape) benefits from
            // graph replay. Skip the whole graph path (incl. the update_required probe)
            // for multi-token graphs. Note this must not use nodes[0]->ne[1] directly:
            // a split-MoE decode split can start on an expert tensor whose ne[1] is
            // n_expert_used (see ggml_cuda_graph_is_multi_token).
            if (ggml_cuda_graph_is_multi_token(cgraph)) {
                use_cuda_graph = false;
            } else {
            const bool properties_changed = ggml_cuda_graph_update_required(graph, cgraph);

            // A capture must NEVER start while the slab's wide band is armed.  The band is reclaimed at
            // the prefill -> decode drop, so it cannot hold a captured graph's pointers (that is why
            // `pool_leg::alloc` skips it during capture), and skipping it would place the capture's
            // temporaries OUTSIDE the slab -- the exact VRAM the band exists to save.  When that outside
            // cushion is short the allocation fails and the pool's OOM reclaim path tries to
            // cudaDeviceSynchronize(), which is illegal inside a capture (the "operation not permitted
            // when stream is capturing" abort).  REPLAY of an already captured graph stays safe (its
            // pointers are outside the slab); only capture/recapture waits for a disarmed pass.
            const bool band_armed = ggml_cuda_slab_enabled() &&
                                    ggml_cuda_slab_wide_available(cuda_ctx->device);

            if (!graph->warmup_complete) {
                // Warmup: need at least 2 calls with no property change on the 2nd call
                if (!properties_changed) {
                    if (band_armed) {
                        // This direct run is served from the band, so it leaves nothing in the pool for a
                        // capture: warm up again on the first disarmed pass (capture_deferred).
                        graph->capture_deferred = true;
                    } else if (graph->capture_deferred) {
                        // First disarmed pass after an armed stretch: run directly so the pool really
                        // holds this graph's temporaries, then capture on the next call.
                        graph->capture_deferred = false;
                        graph->warmup_complete = true;
                        if (ggml_cuda_graph_trace()) {
                            GGML_LOG_WARN("%s: device %d: deferred-capture warmup (direct run)\n", __func__, cuda_ctx->device);
                        }
                    } else {
                        graph->warmup_complete = true;
                        GGML_LOG_DEBUG("%s: CUDA graph warmup complete\n", __func__);
                        use_cuda_graph = true;
                        cuda_graph_update_required = true;
                    }
                }
                // else: properties changed or first call - execute directly (use_cuda_graph stays false)
            } else {
                // Post-warmup: normal CUDA graph operation
                if (properties_changed) {
                    // Properties changed - reset warmup, execute directly until stable again
                    graph->warmup_complete = false;
                    if (ggml_cuda_graph_debug_enabled()) {
                        GGML_LOG_DEBUG("%s: CUDA graph warmup reset\n", __func__);
                    }
                } else {
                    use_cuda_graph = true;
                    // recapture a graph whose pool temporaries or scratch buffers were freed since its capture
                    // (GGML_CUDA_GRAPH_MEM_GEN=0 disables the check, A/B only)
                    static const bool mem_gen_check = !getenv("GGML_CUDA_GRAPH_MEM_GEN") || atoi(getenv("GGML_CUDA_GRAPH_MEM_GEN")) != 0;
                    cuda_graph_update_required = graph->instances.empty() ||
                        (mem_gen_check && graph->mem_gen != ggml_cuda_graph_mem_gen(cuda_ctx->device));
                }
            }

            if (use_cuda_graph && cuda_graph_update_required && band_armed) {
                // Defer the capture (or recapture): run directly this time.  Do NOT leave a stale instance
                // for a replay -- `update_required` means its pointers may no longer be valid.
                GGML_LOG_DEBUG("%s: capture deferred, the slab wide band is armed\n", __func__);
                if (ggml_cuda_graph_trace()) {
                    GGML_LOG_WARN("%s: device %d: capture deferred, the slab wide band is armed\n", __func__, cuda_ctx->device);
                }
                use_cuda_graph = false;
                cuda_graph_update_required = false;
                graph->warmup_complete = false;
                graph->capture_deferred = true;
            }
            if (band_armed && !use_cuda_graph) {
                // Belt-and-braces: ANY direct run in an armed stretch served this graph's pool temporaries
                // from the band, so the next capture must be preceded by a disarmed direct run.  This also
                // covers the case where the band drops between two armed runs (properties changed, or the
                // warmup was already complete) and the very next run would otherwise capture immediately.
                graph->warmup_complete = false;
                graph->capture_deferred = true;
            }
            } // else: not prefill
        }
    }
#endif // USE_CUDA_GRAPH

    ggml_cuda_graph_stats_arm();
    if (!use_cuda_graph) {
        g_gc_n_direct++;
    } else if (cuda_graph_update_required) {
        g_gc_n_capture++;
    } else {
        g_gc_n_replay++;
    }

    if (use_cuda_graph && cuda_graph_update_required) {
        // Start CUDA graph capture
        if (ggml_cuda_graph_trace()) {
            GGML_LOG_WARN("%s: device %d: capturing graph (n_tokens=%d)\n", __func__, cuda_ctx->device,
                          (int) ggml_cuda_graph_n_tokens(cgraph));
        }
        {
            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            ggml_cuda_lock_counter.fetch_add(1, std::memory_order_relaxed);
        }

        graph->mem_gen = ggml_cuda_graph_mem_gen(cuda_ctx->device);
        CUDA_CHECK(cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed));
    }

    if (t_gc) { g_gc_pre_us += ggml_time_us() - t_gc; }
    ggml_cuda_graph_evaluate_and_capture(cuda_ctx, cgraph, use_cuda_graph, cuda_graph_update_required,
                                        use_cuda_graph ? graph : nullptr);
    if (t_gc) { g_ev_rest_us += ggml_time_us() - g_loop_start; }

    ggml_cuda_mmb_compute_done();
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cuda_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    CUDA_CHECK(cudaEventRecord((cudaEvent_t)event->context, cuda_ctx->stream()));
}

static void ggml_backend_cuda_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    if (ggml_backend_is_cuda(backend)) {
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), (cudaEvent_t)event->context, 0));
    } else {
#if 0
        // untested
        auto wait_fn = [](void * user_data) {
            ggml_backend_event_t event = (ggml_backend_event_t)user_data;
            ggml_backend_event_synchronize(event);
        };

        CUDA_CHECK(cudaLaunchHostFunc(cuda_ctx->stream(), wait_fn, event));
#endif
        GGML_ABORT("fatal error");
    }
}

static void ggml_backend_cuda_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    // The arch-keyed predicates this pass consults (MMB's mmb_enabled()/mmb_cfg() and the marks they
    // gate) read the *current* HIP device, and they must agree with the launch path, where the device
    // is always this context's.  The scheduler calls this pass for a split of whichever backend, and
    // under the meta backend it is forwarded per child from meta graph_compute, so the current device
    // can still be the previously computed one (mixed-arch box).  ggml_cuda_set_device() is a no-op
    // when the device already matches.
    ggml_cuda_set_device(cuda_ctx->device);

    // Two-pass mode for the meta (tensor-split) backend.  The scheduler calls this pass once per
    // split before allocation, but under the meta backend the graph a child actually computes is a
    // per-device subgraph of *simple* tensors built inside the meta graph_compute, so the marks this
    // pass sets must be keyed by those simple tensors, while the fusion alloc deps must still be
    // registered here, before allocation.  The meta therefore forwards the pass twice:
    //   - allocs_only: only the add_alloc_dep part (whole graph / original tensors, `params` carries
    //     the scheduler's collector);
    //   - marks_only:  only the marking part (per-device subgraph / simple tensors).
    // The single-backend scheduler leaves both false and runs the complete pass once.
    const bool marks_only  = params != nullptr && params->marks_only;
    const bool allocs_only = params != nullptr && params->allocs_only;

    if (!allocs_only) {
    // BF16-only mark lifetime: clear on the first optimize after a compute, or when the first split
    // of a new graph is seen (the scheduler optimizes every split of one graph before computing any
    // of them, so marks set below must survive the remaining splits' optimize calls).
    const int mark_log = g_mmb_mark_log;
    ggml_cuda_mmb_set_active_ctx(cuda_ctx);
    // BF16-only marks are per backend context and (re)built for the graph being optimised.  The
    // scheduler optimises every split of one graph before computing any of them, so the first
    // optimize pass of a graph clears and the remaining splits add to the same set.  Cleared when a
    // compute happened since the last optimize, or when the same graph's first split is seen again.
    const void * opt_key = cgraph->n_nodes ? cgraph->nodes[0] : nullptr;
    const bool do_clear = ggml_cuda_mmb_optimize_begin(opt_key);
    if (mark_log >= 2) fprintf(stderr, "MMB_OPT backend=%p key=%s n_nodes=%d clear=%d marks=%zu\n",
            (void *) backend, opt_key ? ((const ggml_tensor *) opt_key)->name : "-", cgraph->n_nodes,
            (int) do_clear, ggml_cuda_mmb_marks_count());

    // The HC16 "BF16-only" marks below let a producer elide its F32 output and have every consumer
    // re-read a BF16 copy out of the per-graph activation cache.  That contract only holds while the
    // whole graph runs as one backend compute: ggml_cuda_mmb_begin_graph() resets the cache at the
    // start of EVERY compute, so with an eval callback (llama-imatrix, common/debug) the scheduler
    // runs each MUL_MAT as its own sub-graph and the cache is cleared between the producer and the
    // GEMM -- which then re-converts the never-written F32 buffer and reads garbage (imatrix:
    // "non-finite values detected in blk.N.*.weight").  Keep the F32 outputs (downgrade to the
    // "wants a BF16 copy" mark) whenever the scheduler will split at callback nodes.
    const bool elide_f32 = params == nullptr || !params->has_eval_callback;

    // MMB HC16 (step 1 of the bf16-producer port): mark the gate producer of a gated DSV4_HC_PRE as
    // BF16-only when MMB will produce it and every consumer reads the BF16 copy.  The gate is a
    // dense MUL_MAT [320 x 10240] whose only consumer is the fused pre op, so this is the contained
    // first win.  STRUCTURE ONLY: inside graph_optimize the data pointers are not assigned yet, so
    // the check must not look at alias/buffer state.  LLAMA_MMB_HC16>=1 (== GGML_CUDA_MMB_HC16).
    {
        static const int hc16 = getenv("GGML_CUDA_MMB_HC16") ? atoi(getenv("GGML_CUDA_MMB_HC16")) : 1;
        if (hc16 >= 1 && ggml_cuda_mmb_active() &&
                GGML_CUDA_CC_IS_RDNA3_5(ggml_cuda_info().devices[cuda_ctx->device].cc)) {
            auto reads = [](const ggml_tensor * t, const ggml_tensor * x) {
                for (int s = 0; s < GGML_MAX_SRC && t->src[s]; ++s) if (t->src[s] == x || t->src[s]->view_src == x) return true;
                return false;
            };
            for (int i = 0; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * pre = cgraph->nodes[i];
                if (pre->op != GGML_OP_DSV4_HC_PRE || pre->src[1] == nullptr) continue;
                if (ggml_get_op_params_i32(pre, 1) == 0) continue;   // un-gated pre has no sigmoid gate
                const ggml_tensor * gr = pre->src[1]->view_src ? pre->src[1]->view_src : pre->src[1];
                if (gr->op != GGML_OP_MUL_MAT) continue;
                // the gate weight is [K=320, M=10240]; gr itself is [M=10240, T]
                if (gr->src[0]->ne[0] != 320 || gr->src[0]->ne[1] != 10240) continue;
                if (ggml_nrows(gr) < 512) continue;
                if (!ggml_cuda_mmb_supported_mm(gr->src[0], gr->src[1], gr)) continue;
                int gi = -1;
                for (int k = 0; k <= i; ++k) if (cgraph->nodes[k] == gr) { gi = k; break; }
                if (gi < 0) continue;   // producer not in this split -- cannot guarantee the BF16 copy
                bool ok = true; int nread = 0;
                for (int n = gi + 1; n < cgraph->n_nodes && ok; ++n) {
                    const ggml_tensor * t = cgraph->nodes[n];
                    if (!reads(t, gr)) continue;
                    ++nread;
                    if (t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE) continue;
                    if (t == pre) continue;
                    ok = false;
                }
                if (ok && nread > 0 && elide_f32) {
                    ggml_cuda_mmb_mark_bf16_only(gr);
                    static const int lg = getenv("GGML_CUDA_MMB_LOG") ? atoi(getenv("GGML_CUDA_MMB_LOG")) : 0;
                    if (lg) { static unsigned hits = 0; if (hits++ < 2) fprintf(stderr, "MMB_HC16 gate marked BF16-only: %s (M=%d K=%d T=%d)\n",
                            gr->name, (int) gr->ne[1], (int) gr->ne[0], (int) (gr->ne[2] * gr->ne[3])); }
                }
            }
        }
    }

    // MMB HC16 step 3: mark the activation (src1) of every MMB dense prefill GEMM "wants a BF16
    // copy".  Producers that can emit one (the fused rms_norm+mul, the fused sigmoid+mul, and
    // dsv4_hc_pre) write it into slot 0 in addition to the F32 output, and the GEMM's activation
    // conversion finds and skips it.  The F32 output stays valid for every other consumer, and the
    // copy is RNE-rounded exactly as mmb_cvt_f32_bf16, so the GEMM arithmetic is bit-identical.
    // Dense MUL_MAT only; F32-weight GEMMs read the activation directly (tiny-M / f32split).
    {
        static const int hc16 = getenv("GGML_CUDA_MMB_HC16") ? atoi(getenv("GGML_CUDA_MMB_HC16")) : 1;
        if (hc16 >= 1 && ggml_cuda_mmb_active() &&
                GGML_CUDA_CC_IS_RDNA3_5(ggml_cuda_info().devices[cuda_ctx->device].cc)) {
            auto rootof = [](const ggml_tensor * t) { while (t->view_src) t = t->view_src; return t; };
            // The marking pass runs per split, but a tensor's consumers may live in another split,
            // where the per-compute BF16 activation cache (cleared by ggml_cuda_mmb_begin_graph at
            // the start of every backend compute) is not populated.  Scan the whole scheduled graph
            // and treat any consumer outside this split as a non-BF16 consumer.
            const ggml_cgraph * full = (params && params->full_graph) ? params->full_graph : cgraph;
            const ptrdiff_t split_off = (full->nodes && cgraph->nodes) ? (cgraph->nodes - full->nodes) : 0;
            const ptrdiff_t split_end = split_off + cgraph->n_nodes;
            // classify the consumers of `root` (through views), looking at the whole graph:
            //  0 = no consumer at all, 1 = every consumer is an in-split BF16 reader, 2 = otherwise
            // (a non-BF16 consumer, any consumer in another split, or a routed MMB reader whose node
            // a mmq fusion may consume instead -- see the MUL_MAT_ID arm below) -- 2 means the F32
            // output must be kept (mark copy), 0 means nothing to do.
            auto classify_consumers = [&](const ggml_tensor * root) -> int {
                bool any = false;
                bool routed_reader = false;   // a MUL_MAT_ID consumer MMB claims: F32 must stay (TODO #115)
                for (int i = 0; i < full->n_nodes; ++i) {
                    const ggml_tensor * t = full->nodes[i];
                    bool uses = false;
                    for (int s = 0; s < GGML_MAX_SRC && t->src[s]; ++s) if (rootof(t->src[s]) == root) { uses = true; break; }
                    if (!uses) continue;
                    if (t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE) continue;
                    any = true;
                    if (i < split_off || i >= split_end) return 2;   // consumer in another split
                    if (t->op == GGML_OP_MUL_MAT && t->src[1] && rootof(t->src[1]) == root && t->src[0] &&
                            ggml_cuda_mmb_reads_bf16_act(t->src[0], t->src[1], t)) continue;
                    // A routed MMB reader does NOT authorise eliding the F32 output (TODO #115): the
                    // gate+up MUL_MAT_ID pair that reads this activation may be served by the mmq
                    // triple (block 08's routed gate/up+GLU fusion) or by mul_mat_q_pair instead of
                    // by the two individual MMB muls, and both read the F32 tensor.  Keep the F32
                    // output (mark the bf16 copy instead) so a preempting fusion is still correct.
                    if (t->op == GGML_OP_MUL_MAT_ID && t->src[1] && rootof(t->src[1]) == root &&
                            ggml_cuda_mmb_supported_mmid(t->src[0], t->src[1], t->src[2], t)) { routed_reader = true; continue; }
                    // dsv4_hc_pre reads src[0] (the HC normalized stream) through the bf16 cache when
                    // the activation is marked BF16-only (dsv4-hc.cu's xbf16 arm).  It is the one
                    // consumer that runs well after the producer, so give xn a dedicated slot: the
                    // generic slot 0 is reused by every other activation copy in between (and
                    // dsv4_hc_pre writes its own output there).
                    if (t->op == GGML_OP_DSV4_HC_PRE && t->src[0] && rootof(t->src[0]) == root) {
                        ggml_cuda_mmb_mark_bf16_slot(root, 4);
                        continue;
                    }
                    return 2;
                }
                return (any && !routed_reader) ? 1 : (routed_reader ? 2 : 0);
            };
            for (int i = 0; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * t = cgraph->nodes[i];
                if (t->op != GGML_OP_MUL_MAT || t->src[0] == nullptr || t->src[1] == nullptr) continue;
                if (t->src[0]->type == GGML_TYPE_F32) continue;
                if (!ggml_cuda_mmb_supported_mm(t->src[0], t->src[1], t)) continue;
                const ggml_tensor * ar = rootof(t->src[1]);
                // every consumer in this split reading through the BF16 cache (and none elsewhere)
                // -> the F32 output is dead: emit BF16 only.  otherwise keep the F32 output and add
                // the BF16 copy for the GEMMs that want it (a tensor with no in-split consumer at
                // all is left alone).
                const int c = classify_consumers(ar);
                if (c == 1 && elide_f32) ggml_cuda_mmb_mark_bf16_only(ar);
                else if (c != 0)         ggml_cuda_mmb_mark_bf16_copy(ar);
            }
        }
    }

    // MMB DOWN16 (GGML_CUDA_MMB_DOWN16, default OFF): mark the routed-down GEMM output BF16-only so
    // the producer stores BF16 in place and the weighted reduction reads it as BF16.  Only the
    // IQ4_NL MMB MMID path can emit a BF16 output; every other weight format keeps F32.  The
    // reduction op dispatches on ggml_cuda_mmb_is_bf16_only(experts).
    if (ggml_cuda_mmb_down16()) {
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            ggml_moe_weighted_reduction_match match;
            if (!ggml_match_moe_weighted_reduction(cgraph, i, match)) {
                continue;
            }
            const ggml_tensor * ex = match.experts;
            int xi = -1;
            for (int k = 0; k < i; ++k) {
                if (cgraph->nodes[k] == ex) { xi = k; break; }
            }
            if (xi < 0 || ex->op != GGML_OP_MUL_MAT_ID || ex->type != GGML_TYPE_F32) {
                continue;
            }
            if (!ggml_cuda_mmb_supported_mmid(ex->src[0], ex->src[1], ex->src[2], const_cast<ggml_tensor *>(ex))) {
                continue;
            }
            if (ex->src[0]->type != GGML_TYPE_IQ4_NL) {
                continue;
            }
            if (!ggml_node_has_n_uses(cgraph, xi, 1)) {
                continue;
            }
            if (ex->ne[0] % 8 != 0 || ggml_nrows(ex) < 512) {
                continue;
            }
            if (elide_f32) ggml_cuda_mmb_mark_bf16_only(ex);
        }
    }

    // HC BF16 streams (LLAMA_HC_BLK16 / LLAMA_HC_RES16, both default OFF).  Mark the combine+norm's
    // block_out (attention out-proj / MoE reduction) and residual tensors BF16-only so their
    // producers store BF16 in place and the fused combine reads/writes BF16.  Guards mirror the
    // reference: prefill only (>= 512 rows) and only when every consumer can read a BF16 copy.
    if (ggml_cuda_mmb_blk16() || ggml_cuda_mmb_res16()) {
        auto reads_any = [](const ggml_tensor * t, const ggml_tensor * x) {
            for (int s = 0; s < GGML_MAX_SRC && t->src[s]; ++s) if (t->src[s] == x || t->src[s]->view_src == x) return true;
            return false;
        };
        std::vector<ggml_cuda_hc_combine_norm_args> comb;
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            ggml_cuda_hc_combine_norm_args ca;
            if (ggml_cuda_hc_combine_norm_identify(cgraph, i, ca)) comb.push_back(ca);
        }
        static const int blk_dbg = getenv("GGML_CUDA_HC_BLK16_DEBUG") ? atoi(getenv("GGML_CUDA_HC_BLK16_DEBUG")) : 0;
        if (blk_dbg) fprintf(stderr, "HC_BLK16 comb=%zu\n", comb.size());
        if (ggml_cuda_mmb_blk16()) {   // block_out (attention out-proj / MoE merge) is read only by the fused combine
            std::vector<const ggml_tensor *> blks;
            for (const auto & ca : comb) blks.push_back(ca.block_out->view_src ? ca.block_out->view_src : ca.block_out);
            for (const ggml_tensor * b : blks) {
                if (ggml_nrows(b) < 512 || b->ne[0] % 8 != 0) continue;
                // the producer must honour a BF16 mark (MMB dense GEMM or the MoE reduction, possibly
                // through the shared-expert ADD the fusion folds in).  Our qwen4exp graph keeps the
                // shared-expert ADD apart from the reduction chain, so only the MUL_MAT block_out
                // arms are taken here (the reduction/merge arm is retained for trees where they are
                // adjacent, as in the reference).
                bool producer_ok = false;
                if (b->op == GGML_OP_MUL_MAT && ggml_cuda_mmb_supported_mm(b->src[0], b->src[1], b)) producer_ok = true;
                if (blk_dbg) fprintf(stderr, "HC_BLK16 blk %s op=%s prod=%d\n", b->name, ggml_op_name(b->op), (int) producer_ok);
                if (!producer_ok) {
                    for (int k = 0; k < cgraph->n_nodes && !producer_ok; ++k) {
                        ggml_moe_weighted_reduction_match mm;
                        if (cgraph->nodes[k]->op != GGML_OP_MUL) continue;
                        if (!ggml_match_moe_weighted_reduction(cgraph, k, mm)) continue;
                        if (mm.dst == b) producer_ok = true;
                        else if (k + mm.node_count < cgraph->n_nodes &&
                                 cgraph->nodes[k + mm.node_count] == b && b->op == GGML_OP_ADD &&
                                 ggml_node_has_n_uses(cgraph, k + mm.node_count - 1, 1) &&
                                 (b->src[0] == mm.dst || b->src[1] == mm.dst)) producer_ok = true;
                    }
                }
                if (!producer_ok) continue;
                int bi = -1;
                for (int k = 0; k < cgraph->n_nodes; ++k) if (cgraph->nodes[k] == b) { bi = k; break; }
                if (bi < 0) continue;
                bool ok = true; int nread = 0;
                for (int n = bi + 1; n < cgraph->n_nodes && ok; ++n) {
                    const ggml_tensor * t = cgraph->nodes[n];
                    if (!reads_any(t, b)) continue;
                    ++nread;
                    if (t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_REPEAT) continue;
                    if (t->op == GGML_OP_MUL || t->op == GGML_OP_ADD) continue;
                    ok = false;
                }
                if (ok && nread > 0 && elide_f32) ggml_cuda_mmb_mark_bf16_only(b);
            }
        }
        if (ggml_cuda_mmb_res16()) {   // residual stream: readers are the fused norm and the next combine
            std::vector<std::pair<const ggml_tensor *, const ggml_tensor *>> combp;
            for (const auto & ca : comb) combp.emplace_back(ca.residual, ca.out_res);
            for (const auto & c : combp) {
                const ggml_tensor * r = c.second;
                if (ggml_nrows(r) < 512) continue;
                int ri = -1;
                for (int k = 0; k < cgraph->n_nodes; ++k) if (cgraph->nodes[k] == r) { ri = k; break; }
                if (ri < 0) continue;
                bool ok = true; int nread = 0;
                for (int n = ri + 1; n < cgraph->n_nodes && ok; ++n) {
                    const ggml_tensor * t = cgraph->nodes[n];
                    if (!reads_any(t, r)) continue;
                    ++nread;
                    if (t->op == GGML_OP_RMS_NORM) continue;
                    if (t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE) continue;
                    bool next_combine = false;
                    for (const auto & d : combp) {
                        const ggml_tensor * dr = d.first->view_src ? d.first->view_src : d.first;
                        if (dr == r && d.second == t) { next_combine = true; break; }
                    }
                    if (next_combine) continue;
                    ok = false;
                }
                if (ok && nread > 0 && elide_f32) ggml_cuda_mmb_mark_bf16_only(r);
            }
        }
    }
    }   // !allocs_only (marking passes)

    if (marks_only) {
        return;
    }

    // The in-place residual stream (LLAMA_HC_RES16) needs the producer/consumer on different
    // buffers.  It is a *marking* dependency, so it must be registered by the alloc-deps pass the
    // meta backend runs before allocation -- the marking pass alone cannot reach the graft
    // allocator.  Re-run the structural matcher here so it lands in the same place as the other
    // fusion deps.
    if (ggml_cuda_mmb_res16()) {
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            ggml_cuda_hc_combine_norm_args ca;
            if (ggml_cuda_hc_combine_norm_identify(cgraph, i, ca)) {
                ggml_tensor * rr = const_cast<ggml_tensor *>(ca.residual->view_src ? ca.residual->view_src : ca.residual);
                params->add_alloc_dep(params->user_data, rr, ca.out_res);
                params->add_alloc_dep(params->user_data, rr, ca.out_xn);
            }
        }
    }

    // Prefill indexer relu-sum: keep the pre-relu scores alive until the summed output, so the
    // allocator cannot reuse that buffer for the fused destination.
    {
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            if (cgraph->nodes[i]->op != GGML_OP_UNARY) continue;
            ggml_cuda_idx_relu_sum_args ia;
            if (ggml_cuda_match_idx_relu_sum(cgraph, i, ia) > 0 && ia.score && ia.dst) {
                ggml_tensor * root = ia.score->view_src ? ia.score->view_src : const_cast<ggml_tensor *>(ia.score);
                params->add_alloc_dep(params->user_data, root, ia.dst);
            }
        }
    }

    static const bool disable_fusion = getenv("GGML_CUDA_DISABLE_FUSION") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_FUSION"));

    auto add_alloc_deps = [&](size_t start, size_t last_node) {

        for (size_t i = start; i < last_node; ++i) {
            params->add_alloc_dep(params->user_data, cgraph->nodes[i], cgraph->nodes[last_node]);

            for (int j = 0; j < GGML_MAX_SRC; ++j) {
                if (cgraph->nodes[i]->src[j]) {
                    params->add_alloc_dep(params->user_data, cgraph->nodes[i]->src[j], cgraph->nodes[last_node]);
                }
            }
        }
    };

    if (!disable_fusion) {
        // Q-norm attention (Qwen3, Qwen3-MoE): the model graph runs the Q projection and its RMS_NORM, MUL, ROPE before
        // the V and K projections, so the Q GEMV cannot join the V / K row concat and the allocator hands the Q chain's
        // freed buffers to V / K, which stops the Q / K norm+rope pair (ggml_cuda_qk_pair_k_at).  Move the V / K
        // projections (MUL_MAT + RESHAPE / VIEW, which read none of the Q nodes) in front of the Q projection: the same
        // nodes, one stream, so the results are bit-identical.  Structural only (no data pointers yet), every ubatch
        // size.  GGML_CUDA_QK_ROTATE=0: off.
        static const bool qk_rotate = getenv("GGML_CUDA_QK_ROTATE") == nullptr || atoi(getenv("GGML_CUDA_QK_ROTATE")) != 0;
        for (int i = 0; qk_rotate && i + 8 < cgraph->n_nodes; ++i) {
            ggml_tensor ** n = cgraph->nodes;
            if (n[i]->op != GGML_OP_MUL_MAT || n[i + 1]->op != GGML_OP_RESHAPE || n[i + 1]->src[0] != n[i] ||
                    n[i + 2]->op != GGML_OP_RMS_NORM || n[i + 2]->src[0] != n[i + 1] ||
                    n[i + 3]->op != GGML_OP_MUL      || n[i + 3]->src[0] != n[i + 2] ||
                    n[i + 4]->op != GGML_OP_ROPE     || n[i + 4]->src[0] != n[i + 3]) {
                continue;
            }
            // the K chain RMS_NORM, MUL, ROPE with the same rope parameters and positions; between them only V / K
            // projections and views that read none of the Q nodes (the Q nodes come first, so they read none of them)
            int j = -1;
            bool ok = true;
            for (int k = i + 5; k < std::min(cgraph->n_nodes - 2, i + 16) && ok; ++k) {
                const ggml_tensor * t = n[k];
                if (t->op == GGML_OP_RMS_NORM) {
                    j = k;
                    break;
                }
                ok = t->op == GGML_OP_MUL_MAT || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW;
                for (int s = 0; s < GGML_MAX_SRC && t->src[s] && ok; ++s) {
                    const ggml_tensor * r = t->src[s];
                    while (r->view_src) {
                        r = r->view_src;
                    }
                    for (int q = i; q <= i + 4; ++q) {
                        ok = ok && r != n[q];
                    }
                }
            }
            if (!ok || j < 0 || j == i + 5 || n[j + 1]->op != GGML_OP_MUL || n[j + 1]->src[0] != n[j] ||
                    n[j + 2]->op != GGML_OP_ROPE || n[j + 2]->src[0] != n[j + 1] ||
                    n[j + 2]->src[1] != n[i + 4]->src[1] || n[j + 2]->src[2] != n[i + 4]->src[2] ||
                    memcmp(n[j + 2]->op_params, n[i + 4]->op_params, sizeof(n[i + 4]->op_params)) != 0) {
                continue;
            }
            std::rotate(n + i, n + i + 5, n + j);
            i = j + 2;
        }

        // Small projections and a larger one over the same input (Qwen3.5/3.6 GDN: ssm_beta and ssm_alpha, 32 rows, then
        // the attn_gate z projection a few nodes later): from the first MUL_MAT of <= 256 rows, gather the later MUL_MATs
        // with the same activation, weight type and K (any number of small ones, one larger one) directly after it, so
        // ggml_cuda_mul_mat_row_concat_try_fuse runs them as one launch.  Up to 3 per group (the row concat cap), nothing in
        // between (row concat does not check memory ranges).  Never a MUL_MAT whose output feeds a CONCAT within 8 nodes
        // (the ssm_conv_in fusion skips the nodes up to its CONCAT, e.g. attn_qkv), not Q8_0 (the GDN prescan matches
        // positions).  Only nodes reading that activation and a weight move earlier: bit-identical.  Structural only, every
        // ubatch size, idempotent.  GGML_CUDA_MMVQ_CONCAT_REORDER=0: off.
        static const bool cat_reorder = getenv("GGML_CUDA_MMVQ_CONCAT_REORDER") == nullptr || atoi(getenv("GGML_CUDA_MMVQ_CONCAT_REORDER")) != 0;
        const int cc_reorder = ggml_cuda_info().devices[cuda_ctx->device].cc;
        if (cat_reorder && (GGML_CUDA_CC_IS_RDNA4(cc_reorder) || GGML_CUDA_CC_IS_RDNA3_0(cc_reorder))) {
            ggml_tensor ** n = cgraph->nodes;
            const auto is_weight_mm = [](const ggml_tensor * t) {
                return t->op == GGML_OP_MUL_MAT && t->src[2] == nullptr && ggml_is_quantized(t->src[0]->type) &&
                       t->src[0]->type != GGML_TYPE_Q8_0 && t->src[0]->op == GGML_OP_NONE && ggml_is_matrix(t->src[0]) &&
                       (t->src[0]->buffer == nullptr || ggml_backend_buffer_get_usage(t->src[0]->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            };
            const auto feeds_concat = [&](int i) {
                for (int k = i + 1; k < std::min(cgraph->n_nodes, i + 9); ++k) {
                    if (n[k]->op != GGML_OP_CONCAT) {
                        continue;
                    }
                    for (int s = 0; s < GGML_MAX_SRC && n[k]->src[s]; ++s) {
                        const ggml_tensor * r = n[k]->src[s];
                        while (r->view_src) {
                            r = r->view_src;
                        }
                        if (r == n[i]) {
                            return true;
                        }
                    }
                }
                return false;
            };
            for (int i = 0; i + 1 < cgraph->n_nodes; ++i) {
                const ggml_tensor * a = n[i];
                if (!is_weight_mm(a) || a->src[0]->ne[1] > 256 || feeds_concat(i)) {
                    continue;
                }
                int  pos   = i + 1;
                bool large = false;
                for (int j = i + 1; j < std::min(cgraph->n_nodes, i + 25) && pos < i + 3; ++j) {
                    const ggml_tensor * b = n[j];
                    if (!is_weight_mm(b) || b->src[1] != a->src[1] || b->src[0]->type != a->src[0]->type ||
                            b->src[0]->ne[0] != a->src[0]->ne[0] || b->src[0]->nb[1] != a->src[0]->nb[1] ||
                            (b->src[0]->ne[1] > 256 && large) || feeds_concat(j)) {
                        continue;
                    }
                    large = large || b->src[0]->ne[1] > 256;
                    if (j != pos) {
                        std::rotate(n + pos, n + j, n + j + 1);
                    }
                    ++pos;
                }
                i = pos - 1;
            }
        }

        // add alloc deps for performance positive fusions. This may increase the overall compute buffer size.
        // TODO: consolidate fusion paths in graph_optimize and graph_compute
        ggml_cuda_set_device(cuda_ctx->device);
        for (int i = 0; i + 5 < cgraph->n_nodes; ++i) {
            if (cgraph->nodes[i]->op != GGML_OP_MUL_MAT_ID) {
                continue;
            }
            for (int j = i + 3; j + 2 < cgraph->n_nodes; ++j) {
                if (cgraph->nodes[j]->op == GGML_OP_MUL_MAT_ID && cgraph->nodes[j + 1]->op == GGML_OP_MUL_MAT_ID) {
                    break;
                }
                if (cgraph->nodes[j]->op != GGML_OP_MUL_MAT || !ggml_cuda_match_shared_expert(cgraph, i, j)) {
                    continue;
                }
                // Group both outputs before allocation so the shared result cannot alias intervening nodes.
                std::rotate(cgraph->nodes + i + 3, cgraph->nodes + j, cgraph->nodes + j + 3);
                ggml_tensor * up = cgraph->nodes[i + 2]->src[1];
                params->add_alloc_dep(params->user_data, up->src[1], cgraph->nodes[i + 5]);
                params->add_alloc_dep(params->user_data, up->src[2], cgraph->nodes[i + 5]);
                i += 5;
                break;
            }
        }
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            // Keep the inputs of the fused gate/up/GLU MMQ kernel alive until the GLU so that they cannot alias its output (under op offload the weights are buffer copies).
            if (i + 2 < cgraph->n_nodes && cgraph->nodes[i + 2]->op == GGML_OP_GLU) {
                ggml_tensor * glu  = cgraph->nodes[i + 2];
                ggml_tensor * gate = glu->src[0];
                ggml_tensor * up   = glu->src[1];
                const bool ok = (gate == cgraph->nodes[i] && up == cgraph->nodes[i + 1]) || (gate == cgraph->nodes[i + 1] && up == cgraph->nodes[i]);
                if (ok && ggml_cuda_mul_mat_q_fusion_matches(up, gate, glu, cuda_ctx->device)) {
                    add_alloc_deps(i, i + 2);
                    i += 2;
                    continue;
                }
            }

            if (cgraph->nodes[i]->op == GGML_OP_CONCAT) {
                ggml_cuda_ple_conv_match pm;
                if (ggml_cuda_ple_conv_match_at_concat(cgraph, i, pm)) {
                    params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(pm.x), pm.out);
                    params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(pm.state), pm.out);
                    params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(pm.w), pm.out);
                    continue;
                }
                ggml_cuda_gdn_conv_match gm;
                if (ggml_cuda_gdn_conv_pattern_at_concat(cgraph, i, gm)) {
                    params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(gm.x), cgraph->nodes[gm.conv_idx]);
                    params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(gm.state), cgraph->nodes[gm.conv_idx]);
                }
                continue;
            }
            if (cgraph->nodes[i]->op == GGML_OP_RMS_NORM) {
                ggml_cuda_norm_gated_match nm;
                const int sk = ggml_cuda_norm_gated_match_at(cgraph, i, nm);
                if (sk > 0 && nm.pre >= 0) {
                    params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(nm.x), nm.dst);
                    params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(nm.w), nm.dst);
                    params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(nm.z->view_src ? nm.z->view_src : nm.z), nm.dst);
                }
                continue;
            }


            ggml_moe_weighted_reduction_match match;
            if (ggml_match_moe_weighted_reduction(cgraph, i, match)) {
                params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(match.experts), match.dst);
                params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(match.weights), match.dst);
                if (match.expert_scale != nullptr) {
                    params->add_alloc_dep(
                        params->user_data, const_cast<ggml_tensor *>(match.expert_scale), match.dst);
                }
                i += match.node_count - 1;
            }

            if (cgraph->nodes[i]->op == GGML_OP_UNARY || cgraph->nodes[i]->op == GGML_OP_SOFT_MAX ||
                    cgraph->nodes[i]->op == GGML_OP_ARGSORT) {
                ggml_cuda_topk_moe_args args;
                const bool              can_fuse = ggml_cuda_topk_moe_fusion(cgraph, i, args);
                std::vector<ggml_op>    ops;
                ops.reserve(13);  // max ops; avoids gcc -Wstringop-overflow false positive

                const ggml_tensor * node = cgraph->nodes[i];

                if (can_fuse) {
                    const ggml_tensor * logits  = node->src[0];
                    ggml_tensor *       weights = nullptr;
                    ggml_tensor *       ids     = nullptr;

                    if (!args.delayed_softmax) {
                        int out_nodes[2];  // nodes which can't be elided

                        if (args.sigmoid) {
                            ops.insert(ops.end(), { GGML_OP_UNARY });
                        } else if (args.sqrt_softplus) {
                            ops.insert(ops.end(), { GGML_OP_UNARY, GGML_OP_SQRT });
                        } else {
                            ops.insert(ops.end(), { GGML_OP_SOFT_MAX });
                        }
                        const int i_probs = i + (int) ops.size() - 1;  // last node of the gating activation

                        if (args.prob_bias) {
                            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW,
                                                    GGML_OP_GET_ROWS });
                            out_nodes[0] = i_probs + 4;
                        } else {
                            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
                            out_nodes[0] = i_probs + 3;
                        }
                        ids = cgraph->nodes[out_nodes[0]];

                        if (args.norm) {
                            ops.insert(ops.end(),
                                       { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
                        }
                        if (args.scale) {
                            ops.insert(ops.end(), { GGML_OP_SCALE });
                        }

                        weights      = cgraph->nodes[i + ops.size() - 1];
                        out_nodes[1] = i + ops.size() - 1;

                        if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                                ggml_cuda_should_use_topk_moe(node, logits, weights, ids)) {

                            add_alloc_deps(i, i + ops.size());
                            i += ops.size() - 1;
                        }
                    }
                }
            }
        }
    }

    if (allocs_only) {
        return;
    }

    static bool enable_graph_optimization = [] {
        const char * env     = getenv("GGML_CUDA_GRAPH_OPT");
        return env != nullptr && atoi(env) == 1;
    }();

    if (!enable_graph_optimization) {
        return;
    }

    // Only compute the key when this hook will use it: the key walks the subgraph (a virtual `get_usage`
    // call per source), and `GGML_CUDA_GRAPH_OPT` is off by default, so this used to be a full key walk
    // per graph_compute whose result was discarded by the early return above.  The `set_enabled` side
    // effect (flagging a pre-Volta device) is redundant here: `ggml_backend_cuda_graph_compute` does it.
#ifdef USE_CUDA_GRAPH
    const ggml_cuda_graph_key graph_key = ggml_cuda_graph_get_key(cgraph);
    const bool use_cuda_graph = ggml_cuda_graph_set_enabled(cuda_ctx, cuda_ctx->cuda_graph(graph_key));
#else
    const bool use_cuda_graph = false;
    GGML_UNUSED(cuda_ctx);
    GGML_UNUSED(cgraph);
#endif

    ggml_cuda_stream_context & stream_context = cuda_ctx->stream_context();
    stream_context.reset();

    if (!use_cuda_graph) {
        return;
    }

    ggml_cuda_set_device(cuda_ctx->device);

    // number of out-degrees for a particular node
    std::unordered_map<const ggml_tensor *, int> fan_out;
    // reverse mapping of node to index in the cgraph
    std::unordered_map<const ggml_tensor *, int> node_indices;

    const auto & is_noop = [](const ggml_tensor * node) -> bool {
        return ggml_is_empty(node) || node->op == GGML_OP_NONE || node->op == GGML_OP_RESHAPE ||
               node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE;
    };

    const auto & depends_on = [](const ggml_tensor * dst, const ggml_tensor * src) -> bool {
        for (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
            if (dst->src[s] == src) {
                return true;
            }
        }
        // implicit dependency if they view the same tensor
        const ggml_tensor * dst2 = dst->view_src ? dst->view_src : dst;
        const ggml_tensor * src2 = src->view_src ? src->view_src : src;
        if (dst2 == src2) {
            return true;
        }
        return false;
    };

    for (int node_idx = 0; node_idx < cgraph->n_nodes; node_idx++) {
        const ggml_tensor * node = cgraph->nodes[node_idx];
        node_indices[node]       = node_idx;

        if (is_noop(node)) {
            continue;
        }
        for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
            const ggml_tensor * src = cgraph->nodes[node_idx]->src[src_idx];
            //TODO: check why nrows > 1 fails
            if (node && !is_noop(node) && ggml_nrows(node) <= 1) {
                fan_out[src] += 1;
            }
        }
    }

    // Target Q, K, V for concurrency
    // this is a more general way to find nodes which can be candidates for concurrency (although it has not been tested for anything else):
    // 1. find fan-out (fork) nodes where the same input is used at least N times (in QKV, it would be "attn-norm")
    // 2. find the join node, where 2 or more of the outputs are required (in QKV, this would "KQ" or "flash-attn")
    // 3. account for all branches from the fork to the join
    // 4. To extend lifetimes of the tensors, we interleave the branches (see below for more details)
    // 5. save the original cgraph and restore it in graph_compute, to enable fusion within streams
    // See discussion: https://github.com/ggml-org/llama.cpp/pull/16991#issuecomment-3522620030

    const int min_fan_out = 3;
    const int max_fan_out = 3;

    // store {fork_idx, join_idx}
    std::vector<std::pair<int, int>> concurrent_node_ranges;

    for (const auto & [root_node, count] : fan_out) {
        if (count >= min_fan_out && count <= max_fan_out) {
            const int root_node_idx = node_indices[root_node];

            // only optimize for attn_norm
            // TODO: make this more generic
            if (!strstr(root_node->name, "attn_norm")) {
                continue;
            }

            bool is_part_of_event = false;
            for (const auto & [start, end] : concurrent_node_ranges) {
                if (root_node_idx >= start && root_node_idx <= end) {
                    is_part_of_event = true;
                }
            }

            if (is_part_of_event) {
                continue;
            }

            std::vector<std::vector<const ggml_tensor *>> nodes_per_branch;
            for (int i = root_node_idx + 1; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * node = cgraph->nodes[i];
                if (!is_noop(node) && depends_on(node, root_node)) {
                    nodes_per_branch.push_back({ node });
                }
            }

            GGML_ASSERT(nodes_per_branch.size() == (size_t) count);

            //find the join point
            const ggml_tensor * join_node = nullptr;

            const auto & belongs_to_branch = [&](const ggml_tensor *                      node,
                                                 const std::vector<const ggml_tensor *> & branch) -> bool {
                for (const ggml_tensor * n : branch) {
                    if (depends_on(node, n)) {
                        return true;
                    }
                }
                return false;
            };

            for (int i = root_node_idx + 1; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * curr_node = cgraph->nodes[i];

                int num_joins = 0;
                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    if (belongs_to_branch(curr_node, nodes_per_branch[branch_idx])) {
                        num_joins++;
                    }
                }

                if (num_joins >= 2) {
                    join_node = curr_node;
                    break;
                }

                bool found_branch = false;
                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    std::vector<const ggml_tensor *> & branch_vec = nodes_per_branch[branch_idx];
                    if (belongs_to_branch(curr_node, branch_vec)) {
                        //continue accumulating
                        if (std::find(branch_vec.begin(), branch_vec.end(), curr_node) == branch_vec.end()) {
                            branch_vec.push_back(curr_node);
                        }
                        found_branch = true;
                    }
                }

                if (!found_branch && is_noop(curr_node)) {
                    // we can put it in any branch because it will be ignored
                    nodes_per_branch[0].push_back({ curr_node });
                }
            }

            if (join_node) {
                //Create ggml_cuda_concurrent_event
                ggml_cuda_concurrent_event concurrent_event(nodes_per_branch.size());
                concurrent_event.join_node = join_node;

                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    for (const ggml_tensor * n : nodes_per_branch[branch_idx]) {
                        concurrent_event.stream_mapping[n] = branch_idx + 1;
                    }
                }

                int fork_node_idx = node_indices[root_node];
                int join_node_idx = node_indices[join_node];

                int       current_branch_idx = 0;
                int       current_node_idx   = fork_node_idx + 1;
                const int n_branches         = nodes_per_branch.size();

                int total_branch_nodes = 0;
                for (std::vector<const ggml_tensor *> branch_nodes : nodes_per_branch) {
                    total_branch_nodes += branch_nodes.size();
                }

                // there are other nodes in the middle which are unaccounted for
                // usually (cpy) nodes, then ignore this fork
                if (join_node_idx - fork_node_idx - 1 != total_branch_nodes) {
                    GGML_LOG_DEBUG(
                        "Skipping %s because the number of nodes in the middle is not equal to the total number of "
                        "branch nodes %d != %d\n",
                        root_node->name, join_node_idx - fork_node_idx - 1, total_branch_nodes);
                    continue;
                }

                // Save the original order of nodes in this region before interleaving
                // This is used later to restore grouping for fusion within streams
                concurrent_event.original_order.reserve(total_branch_nodes);
                for (int i = fork_node_idx + 1; i < join_node_idx; ++i) {
                    concurrent_event.original_order.push_back(cgraph->nodes[i]);
                }

                std::unordered_map<const ggml_tensor *, ggml_cuda_concurrent_event> & concurrent_events = cuda_ctx->stream_context().concurrent_events;
                GGML_ASSERT(concurrent_events.find(root_node) == concurrent_events.end());
                concurrent_events.emplace(root_node, std::move(concurrent_event));
                GGML_LOG_DEBUG("Adding stream at node %s %p\n", root_node->name, root_node);
                concurrent_node_ranges.emplace_back(fork_node_idx, join_node_idx);

                // interleave tensors to extend lifetimes so that ggml graph doesn't recycle them
                // example transformation:
                // [attn-norm, QMul, QNorm, QRope, KMul, KNorm, KRope, VMul, attn] ->
                // [attn-norm, QMul, KMul, VMul, QNorm, VNorm, QRope, KRope, attn]
                while (current_node_idx < join_node_idx) {
                    std::vector<const ggml_tensor *> & branch_nodes = nodes_per_branch[current_branch_idx];

                    bool has_node = false;
                    for (std::vector<const ggml_tensor *> branch_node : nodes_per_branch) {
                        has_node |= branch_node.size() > 0;
                    }

                    GGML_ASSERT(has_node);

                    if (branch_nodes.empty()) {
                        current_branch_idx = (current_branch_idx + 1) % n_branches;
                        continue;
                    }

                    cgraph->nodes[current_node_idx] = const_cast<ggml_tensor *>(branch_nodes.front());
                    current_node_idx++;
                    branch_nodes.erase(branch_nodes.begin());

                    // append all empty nodes
                    while (!branch_nodes.empty() && is_noop(branch_nodes.front())) {
                        cgraph->nodes[current_node_idx] = const_cast<ggml_tensor *>(branch_nodes.front());
                        current_node_idx++;
                        branch_nodes.erase(branch_nodes.begin());
                    }

                    current_branch_idx = (current_branch_idx + 1) % n_branches;
                }
            }
        }
    }
}


static const ggml_backend_i ggml_backend_cuda_interface = {
    /* .get_name                = */ ggml_backend_cuda_get_name,
    /* .free                    = */ ggml_backend_cuda_free,
    /* .set_tensor_async        = */ ggml_backend_cuda_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_cuda_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_cuda_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_cuda_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_cuda_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_cuda_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_cuda_graph_compute,
    /* .event_record            = */ ggml_backend_cuda_event_record,
    /* .event_wait              = */ ggml_backend_cuda_event_wait,
    /* .stage_buffer            = */ ggml_backend_cuda_stage_buffer,
    /* .stage_upload            = */ ggml_backend_cuda_stage_upload,
    /* .stage_gather            = */ ggml_backend_cuda_stage_gather,
    /* .stage_wait              = */ ggml_backend_cuda_stage_wait,
    /* .stage_d2d               = */ ggml_backend_cuda_stage_d2d,
    /* .stage_h2d_gbps          = */ ggml_backend_cuda_stage_h2d_gbps,
    /* .stage_input             = */ nullptr, // the CUDA backend stages into a single ring: stage_buffer
    /* .graph_optimize          = */ ggml_backend_cuda_graph_optimize,
    /* .moe_cache_update        = */ ggml_backend_cuda_moe_cache_update,
    /* .moe_cache_take_over     = */ ggml_backend_cuda_moe_cache_take_over,
    /* .moe_cache_band          = */ ggml_backend_cuda_moe_cache_band,
    /* .moe_cache_gather        = */ ggml_backend_cuda_moe_cache_gather,
};

static ggml_guid_t ggml_backend_cuda_guid() {
    static ggml_guid guid = { 0x2c, 0xdd, 0xe8, 0x1c, 0x65, 0xb3, 0x65, 0x73, 0x6a, 0x12, 0x88, 0x61, 0x1c, 0xc9, 0xdc, 0x25 };
    return &guid;
}

bool ggml_backend_is_cuda(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_cuda_guid());
}

int ggml_backend_cuda_get_device_count() {
    return ggml_cuda_info().device_count;
}

static std::string ggml_cuda_device_description(int device) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(device)));

    const ggml_cuda_device_info & info = ggml_cuda_info();
    std::string description = prop.name;
    if (info.device_count > info.physical_device_count) {
        description += " (dev p" + std::to_string(info.devices[device].physical_device) +
                       "/v" + std::to_string(info.devices[device].virtual_index) + ")";
    }
    // expose the AMD gfx id in the description: host-side (model-layer) code keys per-arch
    // policies (e.g. the qwen4exp QSA dense-vs-sparse decode crossover) off the device string
    const int cc = info.devices[device].cc;
    if (cc >= GGML_CUDA_CC_OFFSET_AMD) {
        char buf[16];
        snprintf(buf, sizeof(buf), " (gfx%x)", cc & 0xffff);
        description += buf;
    }
    return description;
}

void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size) {
    snprintf(description, description_size, "%s", ggml_cuda_device_description(device).c_str());
}

static int ggml_cuda_physical_device_share_count(int device) {
    const ggml_cuda_device_info & info = ggml_cuda_info();
    GGML_ASSERT(device >= 0 && device < info.device_count);
    return info.devices[device].physical_share_count;
}

void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total) {
    ggml_cuda_set_device(device);

    CUDA_CHECK(cudaMemGetInfo(free, total));

    // virtual devices sharing one physical GPU share its memory pool; split it between them
    const int share_count = ggml_cuda_physical_device_share_count(device);
    *free  /= share_count;
    *total /= share_count;
}

bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size) {
    if (!ggml_cuda_register_host_enabled()) {
        return false;
    }

#if CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA) || defined(GGML_USE_HIP)
    cudaError_t err = cudaHostRegister(buffer, size, cudaHostRegisterPortable | cudaHostRegisterReadOnly);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();

        GGML_LOG_DEBUG("%s: failed to register %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return false;
    }
    return true;
#else
    GGML_UNUSED(buffer);
    GGML_UNUSED(size);
    return false;
#endif // CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA)
}

void ggml_backend_cuda_unregister_host_buffer(void * buffer) {
    if (!ggml_cuda_register_host_enabled()) {
        return;
    }

    cudaError_t err = cudaHostUnregister(buffer);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
    }
}


// backend device

struct ggml_backend_cuda_device_context {
    int device;
    std::string name;
    std::string description;
    std::string pci_bus_id;
    int op_offload_min_batch_size;
};

static const char * ggml_backend_cuda_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_cuda_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ctx->description.c_str();
}

#if defined(__linux__)
// Helper function to get available memory from /proc/meminfo for UMA systems
static bool ggml_backend_cuda_get_available_uma_memory(long * available_memory_kb, long * free_swap_kb) {
    FILE * meminfo_file = nullptr;
    // 2KB buffer for reading /proc/meminfo since it does not report size info, should be enough
    const size_t BUFFER_SIZE = 2048;
    auto file_buffer = std::make_unique<char[]>(BUFFER_SIZE);
    size_t bytes_read = 0;
    long huge_tlb_total_pages = -1;
    long huge_tlb_free_pages = -1;
    long huge_tlb_page_size = -1;

    if (available_memory_kb == nullptr || free_swap_kb == nullptr) {
        return false;
    }

    meminfo_file = fopen("/proc/meminfo", "r");
    if (meminfo_file == nullptr) {
        GGML_LOG_ERROR("%s: failed to open /proc/meminfo\n", __func__);
        return false;
    }

    // Read file into buffer
    bytes_read = fread(file_buffer.get(), 1, BUFFER_SIZE - 1, meminfo_file);
    fclose(meminfo_file);

    if (bytes_read == 0) {
        GGML_LOG_ERROR("%s: failed to read from /proc/meminfo\n", __func__);
        return false;
    }
    file_buffer[bytes_read] = '\0';

    *available_memory_kb = -1;
    *free_swap_kb = -1;

    // Parse the file buffer line by line
    char * line = file_buffer.get();
    char * line_next;
    while (line < file_buffer.get() + bytes_read) {
        // Find the end of the current line
        line_next = strchr(line, '\n');
        if (line_next != nullptr) {
            *line_next = '\0';
            line_next++;
        } else {
            line_next = file_buffer.get() + bytes_read;
        }

        long value;
        if (sscanf(line, "MemAvailable: %ld kB", &value) == 1) {
            *available_memory_kb = value;
        } else if (sscanf(line, "SwapFree: %ld kB", &value) == 1) {
            *free_swap_kb = value;
        } else if (sscanf(line, "HugePages_Total: %ld", &value) == 1) {
            huge_tlb_total_pages = value;
        } else if (sscanf(line, "HugePages_Free: %ld", &value) == 1) {
            huge_tlb_free_pages = value;
        } else if (sscanf(line, "Hugepagesize: %ld kB", &value) == 1) {
            huge_tlb_page_size = value;
        }

        line = line_next;
    }

    if (huge_tlb_total_pages != 0 && huge_tlb_total_pages != -1) {
        *available_memory_kb = huge_tlb_free_pages * huge_tlb_page_size;

        // Hugetlbfs pages are not swappable.
        *free_swap_kb = 0;
    }

    GGML_LOG_DEBUG("%s: final available_memory_kb: %ld\n", __func__, *available_memory_kb);
    return true;
}
#endif // defined(__linux__)

static void ggml_backend_cuda_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    ggml_cuda_set_device(ctx->device);
    cudaError_t err = cudaMemGetInfo(free, total);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        GGML_LOG_WARN("%s: cudaMemGetInfo failed (%s), returning 0/0\n", __func__, cudaGetErrorString(err));
        *free = 0;
        *total = 0;
        return;
    }

// ref: https://github.com/ggml-org/llama.cpp/pull/17368
#if defined(__linux__) && !defined(GGML_USE_HIP)
    // Check if this is a UMA (Unified Memory Architecture) system
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(ctx->device)));

    // Check if UMA is explicitly enabled via environment variable
    bool is_uma = prop.integrated > 0 || ggml_cuda_unified_memory_enabled();

    if (is_uma) {
        // For UMA systems (like DGX Spark), use system memory info
        long available_memory_kb = 0;
        long free_swap_kb = 0;

        if (ggml_backend_cuda_get_available_uma_memory(&available_memory_kb, &free_swap_kb) && available_memory_kb > 0) {
            *free = (size_t)available_memory_kb * 1024;
        } else {
            GGML_LOG_ERROR("%s: /proc/meminfo reading failed, using cudaMemGetInfo\n", __func__);
        }
    }
#endif // defined(__linux__) && !defined(GGML_USE_HIP)

    // virtual devices sharing one physical GPU share its memory pool; split it between them
    const int share_count = ggml_cuda_physical_device_share_count(ctx->device);
    *free  /= share_count;
    *total /= share_count;
}

static enum ggml_backend_dev_type ggml_backend_cuda_device_get_type(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *) dev->context;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(ctx->device)));

    return prop.integrated
        ? GGML_BACKEND_DEVICE_TYPE_IGPU
        : GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_cuda_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;

    props->name        = ggml_backend_cuda_device_get_name(dev);
    props->description = ggml_backend_cuda_device_get_description(dev);
    props->type        = ggml_backend_cuda_device_get_type(dev);
    props->device_id   = ctx->pci_bus_id.empty() ? nullptr : ctx->pci_bus_id.c_str();
    ggml_backend_cuda_device_get_memory(dev, &props->memory_free, &props->memory_total);

    const bool host_buffer = !ggml_cuda_pinned_memory_disabled();
#ifdef GGML_CUDA_NO_PEER_COPY
    bool events = false;
#else
    bool events = true;
#endif

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ host_buffer,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ events,
        /* .mmap_support          = */ props->type != GGML_BACKEND_DEVICE_TYPE_IGPU,
    };
}

static ggml_backend_t ggml_backend_cuda_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ggml_backend_cuda_init(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cuda_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ggml_backend_cuda_buffer_type(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cuda_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *) dev->context;
    return ggml_backend_cuda_host_buffer_type_dev(ctx->device);
}

// TODO: move these functions here
static bool ggml_backend_cuda_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;

    // check if all the sources are allocated on this device
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op->src[i] && op->src[i]->buffer && ggml_backend_buft_is_cuda(op->src[i]->buffer->buft)) {
            ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)op->src[i]->buffer->buft->context;
            if (buft_ctx->device != dev_ctx->device) {
                return false;
            }
        }
    }

    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_EXPM1:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_XIELU:
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                    if (op->src[0]->type == GGML_TYPE_BF16 && ggml_get_unary_op(op) == GGML_UNARY_OP_XIELU) {
                        return false;
                    }
                    return op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_BF16;
                default:
                    return false;
            }
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    if (op->src[0]->type == GGML_TYPE_BF16 &&
                            (ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU_OAI || ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU_CLAMP)) {
                        return false;
                    }
                    if (op->src[0]->type == GGML_TYPE_F16 && ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU_OAI) {
                        return false;
                    }
                    return ggml_is_contiguous_1(op->src[0]);
                default:
                    return false;
            }
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            {
                struct ggml_tensor * a = op->src[0];
                struct ggml_tensor * b = op->src[1];
                if (a->nb[0] != ggml_element_size(a) || b->nb[0] != ggml_element_size(b)) {
                    return false; // TODO this could in principle be implemented though currently there is no use case.
                }
                if (b->type == GGML_TYPE_F16 && a->type != GGML_TYPE_F16 && !ggml_cuda_op_mul_mat_use_fwht(op)) {
                    return false;
                }
                if (op->op == GGML_OP_MUL_MAT_ID && ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
                    return false;
                }
#ifdef GGML_USE_MUSA
                const int cc = ggml_cuda_info().devices[dev_ctx->device].cc;
                if (b->ne[2]*b->ne[3] > 1 && !ggml_is_transposed(a) && !ggml_is_transposed(b)) {
                    if (GGML_CUDA_CC_IS_QY1(cc) && op->op == GGML_OP_MUL_MAT &&
                            a->type == GGML_TYPE_F16 && b->type == GGML_TYPE_F16) {
                        return false;
                    }
                    if (GGML_CUDA_CC_IS_QY2(cc) && op->op == GGML_OP_MUL_MAT_ID &&
                            a->type == GGML_TYPE_Q2_K && b->type == GGML_TYPE_F32) {
                        return false;
                    }
                }
#endif // GGML_USE_MUSA
                switch (a->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q2_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_MXFP4:
                    case GGML_TYPE_NVFP4:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_Q8_K:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_IQ4_XS:
                    case GGML_TYPE_BF16:
                        return true;
                    default:
                        return false;
                }
            }
        case GGML_OP_OUT_PROD:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_GET_ROWS:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F16:
                    case GGML_TYPE_F32:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_I32:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q2_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ4_XS:
                        return true;
                    case GGML_TYPE_IQ4_NL:
                        // 32-value sub-blocks: the QK_K super-block path needs ne00 % QK_K == 0,
                        // while the sub-block path (dequantize_q4_nl) handles any row width
                        return op->src[0]->ne[0] % QK4_NL == 0;
                    case GGML_TYPE_MXFP4:
                        // 32-value sub-blocks, the row size does not guarantee
                        // the QK_K super-blocks the get_rows kernel iterates on
                        return op->src[0]->ne[0] % QK_K == 0;
                    default:
                        return false;
                }
            }
        case GGML_OP_GET_ROWS_BACK:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->ne[2] == 1 && op->ne[3] == 1;
        case GGML_OP_SET_ROWS:
            {
                return (
                           (
                               (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_BF16 ||
                               op->type == GGML_TYPE_Q4_0 || op->type == GGML_TYPE_Q4_1 || op->type == GGML_TYPE_Q5_0 ||
                               op->type == GGML_TYPE_Q5_1 || op->type == GGML_TYPE_Q8_0 || op->type == GGML_TYPE_IQ4_NL) &&
                               op->src[0]->type == GGML_TYPE_F32
                           ) || (
                               op->type == GGML_TYPE_F16 && op->src[0]->type == GGML_TYPE_F16
                           )
                       ) &&
                       (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32);
            }
        case GGML_OP_SET:
            {
                const ggml_type t = op->type;
                return (t == GGML_TYPE_F32 || t == GGML_TYPE_I32) &&
                    t == op->src[0]->type &&
                    t == op->src[1]->type;
            }
        case GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if ((src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_BF16 || src0_type == GGML_TYPE_F16) &&
                    (src1_type == GGML_TYPE_F32 || src1_type == GGML_TYPE_BF16 || src1_type == GGML_TYPE_F16)
                ) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q8_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q8_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_IQ4_NL) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_I32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_I32) {
                    return true;
                }
                if (src0_type == src1_type && ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                    return true;
                }
                return false;
            }
        case GGML_OP_DUP:
                return true;
        case GGML_OP_ARGMAX:
        case GGML_OP_COUNT_EQUAL:
            return true;
        case GGML_OP_REPEAT:
            {
                // the CUDA REPEAT path only implements F32/F16; other types assert at runtime
                ggml_type src0_type = op->src[0]->type;
                return src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_F16;
            }
        case GGML_OP_REPEAT_BACK:
                return op->type == GGML_TYPE_F32 && (op->src[0]->ne[2]*op->src[0]->ne[3]) <= (1 << 15);
        case GGML_OP_CONCAT:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                const int32_t dim = op->op_params[0];
                return src0_type == src1_type &&
                       src0_type == op->type &&
                       (
                           (
                               ggml_is_quantized(src0_type) &&
                               (
                                   (
                                       dim == 3 &&
                                       ggml_is_contiguous(op->src[0]) &&
                                       ggml_is_contiguous(op->src[1])
                                   ) || (
                                       dim != 3 &&
                                       ggml_is_contiguous_to_3(op->src[0]) &&
                                       ggml_is_contiguous_to_3(op->src[1])
                                   )
                               ) &&
                               op->src[0]->ne[0] % ggml_blck_size(src0_type) == 0 &&
                               op->src[1]->ne[0] % ggml_blck_size(src0_type) == 0
                           ) || (
                               !ggml_is_quantized(src0_type) &&
                               ggml_blck_size(src0_type) == 1 &&
                               (
                                   ggml_type_size(src0_type) == 1 ||
                                   ggml_type_size(src0_type) == 2 ||
                                   ggml_type_size(src0_type) == 4 ||
                                   ggml_type_size(src0_type) == 8
                               )
                           )
                       );
            }
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                return false;
            }
        case GGML_OP_COL2IM_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                return (src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_F16 || src0_type == GGML_TYPE_BF16) &&
                    op->type == src0_type &&
                    ggml_is_contiguous(op->src[0]) &&
                    ggml_is_contiguous(op);
            }
        case GGML_OP_SILU_BACK:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_L2_NORM:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_RMS_NORM_BACK:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_ADD_ID:
        case GGML_OP_ADD1:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_LOG:
            return true;
        case GGML_OP_SCALE:
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_BF16) && op->type == op->src[0]->type;
        case GGML_OP_CLAMP:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            if (op->src[0]->type == GGML_TYPE_BF16 || op->src[1]->type == GGML_TYPE_BF16 || op->type == GGML_TYPE_BF16) {
                return op->src[0]->type == GGML_TYPE_BF16 && op->type == GGML_TYPE_BF16 &&
                    (op->src[1]->type == GGML_TYPE_BF16 || op->src[1]->type == GGML_TYPE_F32);
            }
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) &&
                   (op->type         == GGML_TYPE_F32 || op->type         == GGML_TYPE_F16);
        case GGML_OP_SSM_SCAN: {
            const int32_t K = ggml_get_op_params_i32(op, 0);

            if (op->src[3]->ne[0] == 1) {
                // Mamba2
                // (kernel only supports (d_state == 96 || d_state == 128 || d_state == 256) && d_head % 16 == 0)
                const int64_t d_state = op->src[0]->ne[0];
                return (d_state == 96 || d_state == 128 || d_state == 256) && op->src[0]->ne[1] % 16 == 0;
            } else {
                if (K > 1) {
                    return false;
                }

                // Mamba
                // (kernel only supports d_state == 16, d_head == 1, n_head % 128 == 0, n_group == 1)
                return op->src[0]->ne[0] == 16 && op->src[0]->ne[1] == 1 && op->src[0]->ne[2] % 128 == 0 && op->src[4]->ne[1] == 1;
            }
        }
        case GGML_OP_SSM_CONV: {
            // assumes d_inner % threads == 0
            return op->src[0]->ne[1] % 128 == 0;
        }
        case GGML_OP_CONT:
            return true;
        case GGML_OP_DIAG_MASK_INF:
            return true;
        case GGML_OP_SOFT_MAX:
            return true;
        case GGML_OP_SOFT_MAX_BACK: {
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));
            return max_bias == 0.0f;
        }
        case GGML_OP_ROLL:
            if(op->src[0]->type == GGML_TYPE_F32) {
                return true;
            }
            return false;
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK: {
            return op->src[0]->nb[0] == ggml_type_size(op->src[0]->type) && ggml_is_contiguous_2(op->src[0]);
        }
        case GGML_OP_IM2COL:
        case GGML_OP_IM2COL_3D:
        case GGML_OP_CONV_2D:
            return (ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]));
        case GGML_OP_CONV_3D:
            return (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
                   op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && ggml_is_contiguous(op);
        case GGML_OP_CONV_2D_DW:
            return (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
                   op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_CONV_TRANSPOSE_2D:
        case GGML_OP_POOL_1D:
        case GGML_OP_POOL_2D:
            return true;
        case GGML_OP_ACC:
            // TODO: extend support like so:
            //return ggml_is_contiguous_rows(op->src[0]) && ggml_is_contiguous_rows(op->src[1]);
            return ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
        case GGML_OP_SUM:
            return ggml_is_contiguously_allocated(op->src[0]);
        case GGML_OP_TOP_K:
            return op->src[0]->ne[0] <= INT_MAX;
        case GGML_OP_ARGSORT:
#ifndef GGML_CUDA_USE_CUB
            {
                // bitonic path: the padded row must fit in shared memory
                int64_t ncols_pad = 1;
                while (ncols_pad < op->src[0]->ne[0]) {
                    ncols_pad *= 2;
                }
                return ncols_pad * sizeof(int) <= ggml_cuda_info().devices[dev_ctx->device].smpb;
            }
#else
            return op->src[0]->ne[0] <= INT_MAX;
#endif
        case GGML_OP_SUM_ROWS:
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_MEAN:
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_GROUP_NORM:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_PAD:
            return true;
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_GATED_LINEAR_ATTN:
        case GGML_OP_RWKV_WKV7:
            return true;
        case GGML_OP_GATED_DELTA_NET:
            return true;
        case GGML_OP_DSV4_HC_COMB:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_HC_PRE:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_HC_POST:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && (op->src[3] == nullptr || op->src[3]->type == GGML_TYPE_F32) &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_HC_MIX:
            if (op->src[2]->type == GGML_TYPE_BF16) {
                // the BF16 arm replays the K = 320 mmvf block (hc_lr == 320); other low-ranks
                // take the unfused chain rather than aborting in the kernel
                return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                    op->src[3]->type == GGML_TYPE_BF16 && op->src[0]->ne[1] == 4 &&
                    op->src[2]->ne[1] == 320 && op->src[3]->ne[0] == 320 &&
                    (op->src[4] == nullptr || op->src[4]->type == GGML_TYPE_BF16) &&
                    op->type == GGML_TYPE_F32;
            }
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_Q8_0 && op->src[3]->type == GGML_TYPE_Q8_0 &&
                (op->src[4] == nullptr || op->src[4]->type == GGML_TYPE_F32 || op->src[4]->type == GGML_TYPE_Q8_0) &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_HC_COMBINE:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_FLASH_ATTN_EXT:
            return ggml_cuda_flash_attn_ext_supported(dev_ctx->device, op);
        case GGML_OP_FLASH_ATTN_QSA:
            return ggml_cuda_flash_attn_qsa_supported(dev_ctx->device, op);
        case GGML_OP_INDEXER_TOPK:
            return ggml_cuda_indexer_top_k_supported(dev_ctx->device, op);
        case GGML_OP_INDEXER_SCORE:
            return ggml_cuda_indexer_score_supported(dev_ctx->device, op);
        case GGML_OP_INDEXER_FILL:
            return ggml_cuda_indexer_fill_supported(dev_ctx->device, op);
        case GGML_OP_CROSS_ENTROPY_LOSS:
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
        case GGML_OP_FILL:
        case GGML_OP_CUMSUM:
        case GGML_OP_TRI:
        case GGML_OP_DIAG:
        case GGML_OP_SOLVE_TRI:
            return true;
        case GGML_OP_LIGHTNING_INDEXER:
            return ggml_cuda_lightning_indexer_supported(dev_ctx->device, op);
        case GGML_OP_TOP_K_MASK:
            return op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32;

        default:
            return false;
    }
}

static bool ggml_backend_cuda_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;
    const bool integrated = ggml_cuda_info().devices[dev_ctx->device].integrated;
    return (ggml_backend_buft_is_cuda(buft) && buft->device == dev) || (integrated && ggml_backend_buft_is_cuda_host(buft));
}

static int64_t get_op_batch_size(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_GET_ROWS:
            return 0;
        case GGML_OP_MUL_MAT:
            return op->ne[1];
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
            return op->ne[2];
        default:
            return ggml_nrows(op);
    }
}

static bool ggml_backend_cuda_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;

    // wip/moe-expert-cache: a cache-managed MoE op has its resident experts on the device even at
    // a one-token decode (where `get_op_batch_size` is 1), so offload it so the slot-remap
    // consumer can read the compact arena.  ONLY within the decode band: above it the cache is not
    // involved, and forcing offload there took the >8-token batched decode off its normal path
    // (measured: `llama-batched-bench -npl 16` 137 -> 52 t/s with the cache merely enabled).
    if (moe_cache_enabled() && op->op == GGML_OP_MUL_MAT_ID &&
        op->ne[2] <= MOE_EXPERT_CACHE_MAX_TOK) {
        return true;
    }

    return get_op_batch_size(op) >= dev_ctx->op_offload_min_batch_size;
}

static ggml_backend_event_t ggml_backend_cuda_device_event_new(ggml_backend_dev_t dev) {
#ifdef GGML_CUDA_NO_PEER_COPY
    GGML_UNUSED(dev);
    return nullptr;
#else
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *)dev->context;

    ggml_cuda_set_device(dev_ctx->device);

    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

    return new ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ event,
    };
#endif
}

static void ggml_backend_cuda_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);

    CUDA_CHECK(cudaEventDestroy((cudaEvent_t)event->context));
    delete event;
}

static void ggml_backend_cuda_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    CUDA_CHECK(cudaEventSynchronize((cudaEvent_t)event->context));
}

// wip/slab-ring-region (issue #164): the H2D staging-ring cap.  `GGML_CUDA_SLAB_RING_MIB` is a verbatim
// per-GPU override; otherwise the context auto-sizes it (`slab_set_ring_cap`) from the ubatch and the
// host-expert device count.  `g_ring_max_table` is stashed by the preflight so the recompute has the model
// bound the cap clamps against.  6144 MiB is the fallback for a consumer that never auto-sizes.
static size_t g_h2d_ring_cap_bytes = 0;
static size_t g_ring_max_table    = 0;

size_t ggml_cuda_h2d_ring_cap_bytes() {
    const char * e = getenv("GGML_CUDA_SLAB_RING_MIB");
    if (e != nullptr) {
        return (size_t) atoll(e) << 20;
    }
    return g_h2d_ring_cap_bytes != 0 ? g_h2d_ring_cap_bytes : (size_t) 6144 << 20;
}

void ggml_cuda_h2d_ring_cap_set(size_t bytes) {
    g_h2d_ring_cap_bytes = bytes;
}

static bool ggml_backend_cuda_device_moe_cache_preflight(ggml_backend_dev_t dev, size_t host_expert_bytes, size_t aux_reserve_bytes, size_t max_host_table_bytes) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *) dev->context;
    const bool on = moe_cache_preflight(ctx->device, host_expert_bytes, aux_reserve_bytes);
    if (on) {
        // fit-slab-accounting revival (G3): remember the draft-context estimate for the slab reserve (it
        // is created BEFORE the draft context, so the reserve has to leave room for it).
        ggml_cuda_slab_set_aux_reserve(aux_reserve_bytes);
        // wip/slab-ring-region: size the slab's reserved, non-evictable H2D-ring hole from the largest
        // host-resident weight table.  Done here, before the slab is created, so the hole is part of the
        // slab from birth and never competes with the post-slab consumers for free VRAM.  The context may
        // later re-apply an auto-size cap (`slab_set_ring_cap`) before the slab actually exists.
        g_ring_max_table = max_host_table_bytes;
        ggml_cuda_slab_set_ring_region(ggml_backend_cuda_context::h2d_stage_region_bytes(max_host_table_bytes));
        // This model HAS host-resident experts, so the work/arena split is meaningful -- arm the slab.
        // Dense models never reach here (`llama_model_moe_cache_preflight` skips an empty host-expert map),
        // which is what keeps the slab off them.  Runs before the context, so before any compute buffer.
        ggml_cuda_slab_arm();
    } else {
        // The preflight now runs once at model load (universal, aux 0) and again from `common` with the
        // draft/MTP aux; if the better-informed call declines the cache, undo the earlier arm so a
        // cache-off run does not still withhold VRAM for an arena it will never use.
        ggml_cuda_slab_disarm();
    }
    return on;
}

static void ggml_backend_cuda_device_moe_cache_set_reserve(ggml_backend_dev_t dev, size_t bytes) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *) dev->context;
    moe_cache_set_extra_reserve(ctx->device, bytes);
}

static bool ggml_backend_cuda_device_moe_cache_stats(ggml_backend_dev_t dev, int64_t * hits, int64_t * misses, int64_t * arena_bytes) {
    (void) dev;
    return moe_cache_get_stats(hits, misses, arena_bytes);
}

// TODO #59: the expert cache's decode/verify band for this device (0 = the cache is not active, so a
// speculative verify batch has no band to respect).  `common` clamps the draftless speculators' draft
// length to `band - 1` so every verify batch stays inside it: above the band the routed `MUL_MAT_ID`
// leaves the arena and its experts are streamed from the pinned host master every step, which on a
// host-expert config costs far more than the extra accepted tokens buy.
static int ggml_backend_cuda_device_moe_cache_band(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *) dev->context;
    return moe_cache_enabled() ? moe_cache_max_tok_dev(ctx->device) : 0;
}

static bool ggml_backend_cuda_device_moe_cache_rearm(ggml_backend_dev_t dev) {
    // OPEN 2 (TODO #42): re-arm the expert-cache arena after a compute-buffer drop returned the VRAM.
    (void) dev;
    const bool rearmed = moe_cache_rearm();
    // Debug validator (no-op unless MOE_EXPERT_CACHE_VALIDATE is set); runs outside the cache lock.
    moe_cache_validate("after-rearm");
    return rearmed;
}

// issue #130: route the host-pool entry points through the device API so that libllama does not
// forward-declare and call a ggml-cuda symbol directly (a backend-DL build, and a Windows DLL, only
// export what is marked GGML_API).  The pool is global in ggml-cuda, so any CUDA device serves.
static void ggml_backend_cuda_device_moe_cache_set_host_pool(ggml_backend_dev_t dev, bool on) {
    (void) dev;
    moe_cache_set_host_pool(on);
}

static void ggml_backend_cuda_device_moe_cache_set_host_source(ggml_backend_dev_t dev,
        const void * tensor_data, const char * path, size_t offs,
        int n_experts, size_t host_bytes, size_t total_bytes) {
    (void) dev;
    moe_cache_set_host_source(tensor_data, path, offs, n_experts, host_bytes, total_bytes);
}

static size_t ggml_backend_cuda_device_slab_work_size(ggml_backend_dev_t dev) {
    // OPEN 2 (TODO #42): the movable-boundary slab WORK region's current size (the boundary), or 0 when this
    // device has no slab.  `llama_context` compares it against the narrow size it last settled on to decide
    // whether the compute reserve should follow the workload instead of holding the widest-graph reserve.
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *) dev->context;
    return ggml_cuda_slab_active(ctx->device) ? ggml_cuda_slab_work_size(ctx->device) : 0;
}

static size_t ggml_backend_cuda_device_slab_ring_region(ggml_backend_dev_t dev) {
    // wip/slab-ring-region (issue #164): the reserved H2D staging-ring hole size, or 0 when the slab
    // carved none.  Valid on the narrow side of a narrow -> wide transition, where the ring is disarmed
    // but its region exists, so `llama_context` can skip the drop that would only churn the layout.
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *) dev->context;
    return ggml_cuda_slab_active(ctx->device) ? ggml_cuda_slab_ring_region(ctx->device) : 0;
}

static bool ggml_backend_cuda_device_slab_enabled(ggml_backend_dev_t dev) {
    // issue #164: the slab's global arm state (`GGML_CUDA_SLAB` env AND the MoE preflight's arm).
    GGML_UNUSED(dev);
    return ggml_cuda_slab_enabled();
}

static size_t ggml_backend_cuda_device_slab_ring_slots(ggml_backend_dev_t dev) {
    // issue #164: the H2D staging-ring depth; the context sizes the slab ring region so one staged table
    // fits per slot.  Model-global, so the device is not consulted.
    GGML_UNUSED(dev);
    return (size_t) ggml_backend_cuda_context::h2d_stage_slots();
}

static void ggml_backend_cuda_device_slab_set_ring_cap(ggml_backend_dev_t dev, size_t bytes) {
    // wip/slab-ring-region (issue #164): apply the context's auto-sized ring cap and recompute the region
    // the slab will carve.  Runs BEFORE the slab exists (context reserve), so the whole layout sees it.
    GGML_UNUSED(dev);
    ggml_cuda_h2d_ring_cap_set(bytes);
    ggml_cuda_slab_set_ring_region(ggml_backend_cuda_context::h2d_stage_region_bytes(g_ring_max_table));
}

static size_t ggml_backend_cuda_device_slab_headroom_bytes(ggml_backend_dev_t dev) {
    // wip/fit-slab-accounting (Phase 1, re-cut for r37): the configured slab headroom, reported even
    // though the slab is not armed yet.  `--fit` runs BEFORE the MoE preflight arms the slab, and r37
    // makes `ggml_cuda_slab_enabled()` false until then (`env_on && g_slab_armed`), so it cannot gate
    // this getter.  Report the headroom whenever the slab could be created (the env kill switch is on);
    // the fit only consults it for a model that actually has host-resident experts.
    GGML_UNUSED(dev);
    const char * env = getenv("GGML_CUDA_SLAB");
    if (env != NULL && atoi(env) == 0) {
        return 0;
    }
    return ggml_cuda_slab_headroom_bytes();
}

static void ggml_backend_cuda_device_slab_ring_set(ggml_backend_dev_t dev, bool arm) {
    // wip/slab-ring-region (issue #164): arm/disarm the reserved H2D staging-ring hole.  One meta-unit
    // with the wide compute layout: armed at a wide (prefill) pass, disarmed at the prefill -> decode
    // transition so a decode-only stretch returns the ring (and the wide slack) to the arena, which lets
    // the deferred `alloc_all_locked` size the decode arena across it (measured GSQ -ncmoe 48:
    // 8371 -> 33571 MiB, hit 0.94 -> 0.985).  A prefill that follows re-arms the hole and evicts only the
    // tables that filled it while decode-only -- the designed trade-off.
    //
    // A disarm used to be held back until the cache had sized, because the arena could place its hottest
    // tables in the hole and the next arm evicted them -- the "cold re-arm" MTP NaN of
    // archive/work/slab-ring-region §9.4.  That consumer (whole-table staging FROM the cache,
    // `stage_from_cache`) has since been removed and the NaN no longer reproduces (GSQ and the IQ4_NL
    // field config, 0 NaN); the hold-until-sized gate is retired with it.
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *) dev->context;
    ggml_cuda_slab_ring_set(ctx->device, arm);
}

static void ggml_backend_cuda_device_slab_narrow_floor(ggml_backend_dev_t dev, size_t bytes) {
    // wip/slab-ring-region: pin the narrow (decode/verify) work size; the ring is placed just above it.
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *) dev->context;
    ggml_cuda_slab_set_narrow_floor(ctx->device, bytes);
}

static void ggml_backend_cuda_device_slab_reserve_wide(ggml_backend_dev_t dev, size_t bytes) {
    // wip/first-prefill-latency: reserve the wide (prefill) work region above the narrow floor + ring.
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *) dev->context;
    ggml_cuda_slab_reserve_wide(ctx->device, bytes);
}

static void ggml_backend_cuda_device_slab_narrow2_floor(ggml_backend_dev_t dev, size_t bytes) {
    // wip/moe-verify-fusions (narrow-2): pin (idempotently) the MTP draft's compute region from its
    // measured narrow size.  No-op when the slab is not live on this device.
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *) dev->context;
    ggml_cuda_slab_narrow2_floor(ctx->device, bytes);
}

static void ggml_backend_cuda_device_slab_compute_narrow2(ggml_backend_dev_t dev, bool enable) {
    // wip/moe-verify-fusions (narrow-2): `enable` marks the MTP draft context's COMPUTE allocations.
    // Default: route them to the slab's pinned second narrow region.  `GGML_CUDA_SLAB_NARROW2=0` is the
    // A/B arm -- the draft bypasses the slab entirely (raw cudaMalloc), the session's quick fix.
    GGML_UNUSED(dev);
    ggml_cuda_slab_set_compute_narrow2(enable);
}

static void ggml_backend_cuda_device_slab_compute_plain(ggml_backend_dev_t dev, bool enable) {
    GGML_UNUSED(dev);
    ggml_cuda_slab_set_compute_plain(enable);
}

static const ggml_backend_device_i ggml_backend_cuda_device_interface = {
    /* .get_name                = */ ggml_backend_cuda_device_get_name,
    /* .get_description         = */ ggml_backend_cuda_device_get_description,
    /* .get_memory              = */ ggml_backend_cuda_device_get_memory,
    /* .get_type                = */ ggml_backend_cuda_device_get_type,
    /* .get_props               = */ ggml_backend_cuda_device_get_props,
    /* .init_backend            = */ ggml_backend_cuda_device_init_backend,
    /* .get_buffer_type         = */ ggml_backend_cuda_device_get_buffer_type,
    /* .get_host_buffer_type    = */ ggml_backend_cuda_device_get_host_buffer_type,
    /* .buffer_from_host_ptr    = */ NULL,
    /* .supports_op             = */ ggml_backend_cuda_device_supports_op,
    /* .supports_buft           = */ ggml_backend_cuda_device_supports_buft,
    /* .offload_op              = */ ggml_backend_cuda_device_offload_op,
    /* .event_new               = */ ggml_backend_cuda_device_event_new,
    /* .event_free              = */ ggml_backend_cuda_device_event_free,
    /* .event_synchronize       = */ ggml_backend_cuda_device_event_synchronize,
    /* .moe_cache_preflight     = */ ggml_backend_cuda_device_moe_cache_preflight,
    /* .moe_cache_set_reserve   = */ ggml_backend_cuda_device_moe_cache_set_reserve,
    /* .moe_cache_stats         = */ ggml_backend_cuda_device_moe_cache_stats,
    /* .moe_cache_rearm         = */ ggml_backend_cuda_device_moe_cache_rearm,
    /* .slab_work_size          = */ ggml_backend_cuda_device_slab_work_size,
    /* .slab_headroom_bytes     = */ ggml_backend_cuda_device_slab_headroom_bytes,
    /* .slab_ring_set           = */ ggml_backend_cuda_device_slab_ring_set,
    /* .slab_narrow_floor       = */ ggml_backend_cuda_device_slab_narrow_floor,
    /* .slab_reserve_wide       = */ ggml_backend_cuda_device_slab_reserve_wide,
    /* .slab_narrow2_floor      = */ ggml_backend_cuda_device_slab_narrow2_floor,
    /* .slab_compute_narrow2    = */ ggml_backend_cuda_device_slab_compute_narrow2,
    /* .moe_cache_set_host_pool   = */ ggml_backend_cuda_device_moe_cache_set_host_pool,
    /* .moe_cache_set_host_source = */ ggml_backend_cuda_device_moe_cache_set_host_source,
    /* .slab_compute_plain        = */ ggml_backend_cuda_device_slab_compute_plain,
    /* .slab_ring_region          = */ ggml_backend_cuda_device_slab_ring_region,
    /* .slab_set_ring_cap         = */ ggml_backend_cuda_device_slab_set_ring_cap,
    /* .slab_enabled              = */ ggml_backend_cuda_device_slab_enabled,
    /* .slab_ring_slots           = */ ggml_backend_cuda_device_slab_ring_slots,
    /* .moe_cache_band            = */ ggml_backend_cuda_device_moe_cache_band,
};

// backend reg

struct ggml_backend_cuda_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_cuda_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_CUDA_NAME;
}

static size_t ggml_backend_cuda_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_cuda_reg_context * ctx = (ggml_backend_cuda_reg_context *)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_cuda_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_cuda_reg_context * ctx = (ggml_backend_cuda_reg_context *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static ggml_backend_feature * ggml_backend_cuda_get_features(ggml_backend_reg_t reg) {
    static std::vector<ggml_backend_feature> features = []() {
        std::vector<ggml_backend_feature> features;
    #define _STRINGIFY(...) #__VA_ARGS__
    #define STRINGIFY(...) _STRINGIFY(__VA_ARGS__)

    #ifdef __CUDA_ARCH_LIST__
        features.push_back({ "ARCHS", STRINGIFY(__CUDA_ARCH_LIST__) });
    #endif

    #ifdef GGML_CUDA_FORCE_MMQ
        features.push_back({ "FORCE_MMQ", "1" });
    #endif

    #ifdef GGML_CUDA_FORCE_CUBLAS
        features.push_back({ "FORCE_CUBLAS", "1" });
    #endif

    #ifndef GGML_USE_VMM
        features.push_back({ "NO_VMM", "1" });
    #endif

    #ifdef GGML_CUDA_NO_PEER_COPY
        features.push_back({ "NO_PEER_COPY", "1" });
    #endif

    #ifdef GGML_CUDA_USE_GRAPHS
        features.push_back({ "USE_GRAPHS", "1" });
    #endif

    #ifdef GGML_CUDA_FA_QUANTS
        features.push_back({ "FA_QUANTS", GGML_CUDA_FA_QUANTS });
    #endif

    {
        const auto & info = ggml_cuda_info();
        for (int id = 0; id < info.device_count; ++id) {
            if (blackwell_mma_available(info.devices[id].cc)) {
                features.push_back({ "BLACKWELL_NATIVE_FP4", "1"});
                break;
            }
        }
    }

    #undef _STRINGIFY
    #undef STRINGIFY

        features.push_back({ nullptr, nullptr });

        return features;
    }();

    return features.data();

    GGML_UNUSED(reg);
}


static void * ggml_backend_cuda_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (strcmp(name, "ggml_backend_cuda_fattn_stage_bound") == 0) {
        return (void *) ggml_backend_cuda_fattn_stage_bound;
    }
    if (strcmp(name, "ggml_backend_cuda_fattn_stage_used") == 0) {
        return (void *) ggml_backend_cuda_fattn_stage_used;
    }
    if (strcmp(name, "ggml_backend_cuda_h2d_stage_bound") == 0) {
        return (void *) ggml_backend_cuda_h2d_stage_bound;
    }
    if (strcmp(name, "ggml_backend_cuda_h2d_stage_used") == 0) {
        return (void *) ggml_backend_cuda_h2d_stage_used;
    }
    if (strcmp(name, "ggml_backend_comm_init") == 0) {
        return (void *)ggml_backend_cuda_comm_init;
    }
    if (strcmp(name, "ggml_backend_comm_free") == 0) {
        return (void *)ggml_backend_cuda_comm_free;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        return (void *)ggml_backend_cuda_comm_allreduce_tensor;
    }
    if (strcmp(name, "ggml_backend_register_host_buffer") == 0) {
        return (void *)ggml_backend_cuda_register_host_buffer;
    }
    if (strcmp(name, "ggml_backend_unregister_host_buffer") == 0) {
        return (void *)ggml_backend_cuda_unregister_host_buffer;
    }
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_cuda_get_features;
    }
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_cuda_reg_interface = {
    /* .get_name          = */ ggml_backend_cuda_reg_get_name,
    /* .get_device_count  = */ ggml_backend_cuda_reg_get_device_count,
    /* .get_device        = */ ggml_backend_cuda_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_cuda_reg_get_proc_address,
};

// backend registry
ggml_backend_reg_t ggml_backend_cuda_reg() {
    static ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_backend_cuda_reg_context * ctx = new ggml_backend_cuda_reg_context;
            const int min_batch_size = ggml_cuda_op_offload_min_batch();

            const ggml_cuda_device_info & info = ggml_cuda_info();
            const bool virtual_devices = info.device_count > info.physical_device_count;

            for (int i = 0; i < info.device_count; i++) {
                const int physical_id = info.devices[i].physical_device;

                ggml_backend_cuda_device_context * dev_ctx = new ggml_backend_cuda_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_CUDA_NAME + std::to_string(i);
                dev_ctx->description = ggml_cuda_device_description(i);

                char pci_bus_id[32] = {};
                CUDA_CHECK(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), physical_id));
                dev_ctx->pci_bus_id = pci_bus_id;
                if (virtual_devices) {
                    // make the pci bus id unique for virtual devices
                    dev_ctx->pci_bus_id += "-v" + std::to_string(i);
                }
                for (char & c : dev_ctx->pci_bus_id) {
                    c = std::tolower(c);
                }
                dev_ctx->op_offload_min_batch_size = min_batch_size;

                ggml_backend_dev_t dev = new ggml_backend_device {
                    /* .iface   = */ ggml_backend_cuda_device_interface,
                    /* .reg     = */ &reg,
                    /* .context = */ dev_ctx
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_cuda_reg_interface,
                /* .context     = */ ctx
            };
        }

        initialized = true;
    }

    return &reg;
}

ggml_backend_t ggml_backend_cuda_init(int device) {
    if (device < 0 || device >= ggml_backend_cuda_get_device_count()) {
        GGML_LOG_ERROR("%s: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_cuda_context * ctx = new ggml_backend_cuda_context(device);
    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: failed to allocate context\n", __func__);
        return nullptr;
    }

    // wip/moe-expert-cache Phase 1a: parse the env, allocate/report the arena (no-op if disabled).
    moe_cache_init(device);

    ggml_backend_t cuda_backend = new ggml_backend {
        /* .guid    = */ ggml_backend_cuda_guid(),
        /* .iface   = */ ggml_backend_cuda_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), device),
        /* .context = */ ctx,
    };

    return cuda_backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cuda_reg)
