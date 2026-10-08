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
#include <vector>

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
static int ggml_cuda_get_physical_device(int device) {
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

// Issue/TODO #42: shared device-allocation helper.  It yields the lowest-priority MoE expert-cache
// arena when an allocation cannot otherwise be satisfied, so EVERY device allocation inherits the same
// fail-soft policy (defined in ggml-cuda.cu).  The slab query is declared here because the slab itself
// is defined further down this TU.
bool ggml_cuda_slab_active(int device);

cudaError_t ggml_cuda_device_malloc(void ** ptr, size_t size, int device);

cudaError_t ggml_cuda_device_malloc(void ** ptr, size_t size, int device) {
    ggml_cuda_set_device(device);
    cudaError_t err;
    if (getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") != nullptr) {
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
        if (ggml_cuda_slab_active(device)) {
            GGML_LOG_WARN("%s: %.2f MiB allocation failed on device %d while the movable-boundary slab holds "
                          "the VRAM; the arena cannot yield physical back to the driver.  Raise "
                          "GGML_CUDA_SLAB_RESERVE_MIB (or disable the slab with GGML_CUDA_SLAB=0) so the "
                          "weights / KV cache / workspaces stay outside it.\n",
                          __func__, size / 1024.0 / 1024.0, device);
            moe_cache_validate("device-malloc-yield");
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
        while (moe_cache_shrink_step()) {
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

    if (getenv("GGML_CUDA_P2P") != nullptr) {
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


// Map physical into an already-reserved VA range.  No slab lock (no shared state).
//
// NOTE: the range is mapped as ONE mapping.  ROCm rejects `hipMemUnmap` of a SUB-range (measured:
// hipErrorInvalidValue), so a per-table TAIL prune would need one mapping per granularity unit -- and a
// 11776 MiB compute buffer mapped as ~5900 unit mappings then fails an unrelated `hipMemcpy2DAsync`
// (cpy.cu:479, hipErrorInvalidValue) in-tree, even though a standalone 6144-unit map of the same size
// succeeds.  That is why the design maps the whole slab ONCE and never gives physical back: the boundary
// moves at runtime instead, and the arena reclaim is per-table (the tables in the taken chunks).
static bool ggml_cuda_vmm_map_phys(int device, CUdeviceptr addr, size_t aligned) {
    const int phys = ggml_cuda_get_physical_device(device);

    CUmemAllocationProp prop = {};
    prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id   = phys;

    CUmemGenericAllocationHandle handle;
    if (cuMemCreate(&handle, aligned, &prop, 0) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    // Every step below is checked explicitly rather than with CU_CHECK: a failure must leave the caller
    // able to fall back to the plain cudaMalloc path, and CU_CHECK ends the process.  The slab depends
    // on that (it is offered to every device, including ones whose driver refuses part of this sequence).
    if (cuMemMap(addr, aligned, 0, handle, 0) != cudaSuccess) {
        (void) cudaGetLastError();
        (void) cuMemRelease(handle);
        return false;
    }
    if (cuMemRelease(handle) != cudaSuccess) {
        (void) cudaGetLastError();
        (void) cuMemUnmap(addr, aligned);
        return false;
    }

    // Grant READWRITE to THIS device and every PEER device.  The scheduler copies a split input
    // device-to-device (`hipMemcpyPeerAsync`) into the compute buffer, which is a view of this same
    // slab mapping; granting only the owning device left the peer's copy engine with no access to the
    // destination and it faulted `Page not present or supervisor privilege` (the parked `-sm tensor`
    // + partial-arena fault).  Devices that cannot peer-map the range keep their own access (best
    // effort): a failed peer descriptor must not fail the mapping.
    const int n_dev = ggml_backend_cuda_get_device_count();
    std::vector<CUmemAccessDesc> access;
    access.reserve((size_t) n_dev);
    {
        CUmemAccessDesc own = {};
        own.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        own.location.id   = phys;
        own.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        access.push_back(own);
    }
    for (int d = 0; d < n_dev; d++) {
        const int p = ggml_cuda_get_physical_device(d);
        if (p == phys) {
            continue;
        }
        CUmemAccessDesc peer = {};
        peer.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        peer.location.id   = p;
        peer.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        access.push_back(peer);
    }
    if (cuMemSetAccess(addr, aligned, access.data(), access.size()) != cudaSuccess) {
        // Retry with only the owning device: a peer descriptor rejection is not a mapping failure.
        (void) cudaGetLastError();
        CUmemAccessDesc own = access.front();
        if (cuMemSetAccess(addr, aligned, &own, 1) != cudaSuccess) {
            (void) cudaGetLastError();
            (void) cuMemUnmap(addr, aligned);
            return false;
        }
    }
    return true;
}


// ---------------------------------------------------------------------------------------------
// OPEN 2: the movable-boundary slab -- ONE per device.
//
// The whole allocatable VRAM is reserved and MAPPED EXACTLY ONCE as a single slab, then split by a
// BOUNDARY: the LOW region `[0, boundary)` is the work pool (the compute buffer) and the HIGH region
// `[boundary, size)` is the MoE expert-cache arena.  Growing the work pool is a BOUNDARY MOVE inside the
// already-mapped slab -- the lowest arena chunks are reassigned to the work pool and the arena tables
// that live there are dropped -- so `cuMemCreate`/`cuMemMap`/`cuMemUnmap` are NOT called at runtime: HIP
// is touched only when the slab is created and when it is destroyed.  Because the work region's base VA
// never moves, a growing layout keeps every tensor address it had.
//
// LOCKING: `g_slab_mutex` guards the map.  The boundary move must NOT hold it across the arena eviction
// (which re-enters via `ggml_cuda_slab_arena_free`), so `work_alloc` computes the range, unlocks, evicts,
// relocks.  Ordering is cache lock -> slab lock everywhere (the cache allocates arena slabs under its own
// lock); nothing takes the slab lock and then the cache lock.
struct ggml_cuda_slab {
    bool                     inited   = false;
    int                      device   = -1;
    CUdeviceptr              base     = 0;
    size_t                   size     = 0;      // VA reserved for the slab (chunk-aligned).  Physical is
                                                //  mapped into it in two steps -- `mapped` at creation and
                                                //  the rest by `ggml_cuda_slab_extend` -- so the arena can
                                                //  grow into reserve the model turned out not to need.
    size_t                   mapped   = 0;      // how much of `size` has physical behind it: the ARENA's
                                                //  top is `mapped`, NOT `size`
    size_t                   chunk    = 0;      // chunk unit for the BOUNDARY move
    size_t                   unit     = 0;      // finer unit for ARENA allocations (chunk-aligned slabs
                                                //  would waste up to chunk-1 bytes per table: ~9 GiB over
                                                //  288 tables, which made most tables fail to allocate)
    size_t                   boundary = 0;      // work = [0, boundary), arena = [boundary, size)
    std::multiset<size_t>    work_needs;        // REQUESTED size of every live work view -- the floor the
                                                //  boundary may shrink to (see `work_alloc`)
    bool                     cache_unusable = false; // the slab was declined because `work + min arena`
                                                //  does not fit: the cache must stream instead
    std::map<size_t, size_t> arena_free;        // arena free runs: offset -> size (chunk-aligned)
};

static std::mutex    g_slab_mutex;
static ggml_cuda_slab g_slabs[GGML_CUDA_MAX_DEVICES];

static size_t ggml_cuda_slab_chunk_bytes() {
    static size_t chunk_mib = SIZE_MAX;
    if (chunk_mib == SIZE_MAX) {
        const char * env = getenv("GGML_CUDA_SLAB_CHUNK_MIB");
        chunk_mib = env != NULL ? (size_t) atoll(env) : 64;
    }
    return chunk_mib * 1024 * 1024;
}

static size_t ggml_cuda_slab_reserve_bytes(size_t total_b) {
    static size_t reserve_mib = SIZE_MAX;
    if (reserve_mib == SIZE_MAX) {
        const char * env = getenv("GGML_CUDA_SLAB_RESERVE_MIB");
        if (env != NULL) {
            reserve_mib = (size_t) atoll(env);
        } else {
            // Left OUT of the slab for everything that is NOT the work pool or the arena: the GPU-resident
            // model weights, the KV cache, the draft model and the various workspace pools are allocated
            // AFTER the slab is created (the slab is born during the fitting probe), and under the slab
            // they can no longer borrow from the arena -- ROCm will not unmap a sub-range of the slab's
            // single mapping, so a freed arena table returns a byte to the slab's free list, never to the
            // driver.  A reserve that is too small therefore ENDS THE RUN (measured: 4096 MiB -> a failed
            // hipBLASLt workspace allocation at load, exit 134), which is why the default is deliberately
            // generous and scales with the card rather than being a flat constant: ~1/4 of the device, and
            // never less than 8 GiB (the measured need of the primary Qwen3.8-Flash-Next IQ4_XS config on
            // a 32 GiB card).
            reserve_mib = (size_t) std::max<size_t>(8192, total_b / 4 / (1024 * 1024));
        }
    }
    return reserve_mib * 1024 * 1024;
}

// MiB form for messages.
static size_t ggml_cuda_slab_reserve_mib(size_t total_b) {
    return ggml_cuda_slab_reserve_bytes(total_b) / (1024 * 1024);
}

// HARD MINIMUM for the slab: the estimated maximum work buffer PLUS a floor for the MoE expert cache.// Below the floor the cache's fixed per-op cost outweighs the host bytes it saves (measured: a small arena
// is slower than the plain host-expert path, MTP especially), so a slab that cannot provide
// `work + floor` is not worth creating -- the cache falls back to STREAMING the experts from the host
// (the stock `-ncmoe` path) instead.  `work` is the first work need, which is the widest layout the
// fitting probe reserves, so it IS the estimate of the maximum work buffer.
//
// NOTE this is a floor at the ESTIMATE, not an absolute cap: a later graph needing more than the estimate
// moves the boundary again and eats into the floor.  The cache's own auto floor at sizing
// (`MOE_EXPERT_CACHE_MIN_MIB`) is the second, later guard for that case.
static size_t ggml_cuda_slab_min_arena_bytes() {
    static size_t mib = SIZE_MAX;
    if (mib == SIZE_MAX) {
        const char * env = getenv("GGML_CUDA_SLAB_MIN_ARENA_MIB");
        mib = env != NULL ? (size_t) atoll(env) : 2048;
    }
    return mib * 1024 * 1024;
}

// How much VRAM to leave OUTSIDE the slab once the model's own buffers are resident, when the slab GROWS
// into the reserve the fitting probe made it hold back (`ggml_cuda_slab_extend`).  The slab is created
// before the weights exist, so `GGML_CUDA_SLAB_RESERVE_MIB` has to be a guess; this is the measured
// correction -- whatever the weights / KV / draft did NOT use comes back to the arena, minus this much for
// the allocations that can still appear later.  `0` disables the extension (the slab then keeps exactly
// the reserve it took at creation).
//
// DEFAULT 4096 MiB, and do NOT treat it as slack: this is not just the steady-state weights/KV, it must also
// cover the largest TRANSIENT workspace any graph allocates, and those are allocated by
// `ggml_cuda_pool_leg` / `ggml_backend_cuda_buffer_type_alloc_buffer` with CUDA_CHECK -- i.e. a failure
// ABORTS.  The transient grows with the context, so the required headroom does too.  Measured on the
// 27B/16k FA path (`ggml_cuda_flash_attn_qsa3`) at `-ub 4096 -c 163860`: 2048 MiB aborts the run on a 16k
// prompt (after a short prompt, with the arena right-sized to 42018 MiB); 4096 MiB is reliable and still
// leaves a 37861 MiB (66.7 %) arena.  At a smaller context (`-c 32768`, `-ub 8192`) 2048 MiB was fine.  The
// real fix for the cliff is to let those transients draw on the slab's YIELDABLE arena (see the handover,
// Session 5), which would allow this default to go back down.
#define GGML_CUDA_SLAB_HEADROOM_MIB_DEFAULT 4096

static size_t ggml_cuda_slab_headroom_bytes() {
    static size_t mib = SIZE_MAX;
    if (mib == SIZE_MAX) {
        const char * env = getenv("GGML_CUDA_SLAB_HEADROOM_MIB");
        mib = env != NULL ? (size_t) atoll(env) : GGML_CUDA_SLAB_HEADROOM_MIB_DEFAULT;
    }
    return mib * 1024 * 1024;
}

// Public MiB form, for the cache's own fallback message.
size_t ggml_cuda_slab_min_arena_mib() {
    return ggml_cuda_slab_min_arena_bytes() / (1024 * 1024);
}

static size_t ggml_cuda_slab_up(size_t x, size_t a) {
    return a * ((x + a - 1) / a);
}

// The arena's allocation unit: fine (the VMM granularity).  The slab is ONE mapping, so this is pure
// bookkeeping -- the sub-slab ranges are not separately mapped -- which is why it can be much finer than
// the boundary's chunk without any extra HIP work.
static size_t ggml_cuda_slab_unit_bytes(int device) {
    const size_t g = ggml_cuda_info().devices[device].vmm_granularity;
    return g != 0 ? g : (2u * 1024 * 1024);
}

size_t ggml_cuda_slab_arena_unit(int device) {
    std::lock_guard<std::mutex> lock(g_slab_mutex);
    const ggml_cuda_slab & s = g_slabs[device];
    return s.inited ? s.unit : ggml_cuda_slab_unit_bytes(device);
}

bool ggml_cuda_slab_enabled() {
#if defined(GGML_USE_HIP)
    static int enabled = -1;
    if (enabled < 0) {
        // DEFAULT ON (RDNA/ROCm scope): the movable-boundary slab is what keeps a wide prefill and a large
        // expert-cache arena able to coexist, and without it a later wide prefill on a server aborts
        // (`cudaMalloc failed` -> assert) -- see WORKLOG / the OPEN 2 handover.  It costs a few percent of
        // decode because the arena region is smaller than the double-booked one the plain path gets away
        // with, so it is a kill switch, not an opt-in: GGML_CUDA_SLAB=0 restores the plain allocation path
        // (and the abort).  The slab declines per device where VMM is unavailable, and falls back cleanly.
        const char * env = getenv("GGML_CUDA_SLAB");
        enabled = env != NULL ? atoi(env) : 1;
    }
    return enabled != 0;
#else
    return false;   // RDNA/ROCm-scoped
#endif
}

// Is the slab live on THIS device?  (The gate can be on while a device failed to create one.)
bool ggml_cuda_slab_active(int device) {
    std::lock_guard<std::mutex> lock(g_slab_mutex);
    return g_slabs[device].inited;
}

// The slab declined because `work estimate + cache floor` does not fit on this device (see
// `ggml_cuda_slab_min_arena_bytes`).  The expert cache must then stream from the host rather than build an
// arena that a wide prefill would have to evict wholesale.
bool ggml_cuda_slab_cache_unusable(int device) {
    std::lock_guard<std::mutex> lock(g_slab_mutex);
    return g_slabs[device].cache_unusable;
}

size_t ggml_cuda_slab_chunk_size(int device) {
    std::lock_guard<std::mutex> lock(g_slab_mutex);
    const ggml_cuda_slab & s = g_slabs[device];
    return s.inited ? s.chunk : ggml_cuda_slab_chunk_bytes();
}

void * ggml_cuda_slab_work_base(int device) {
    std::lock_guard<std::mutex> lock(g_slab_mutex);
    const ggml_cuda_slab & s = g_slabs[device];
    return s.inited ? (void *) s.base : nullptr;
}

size_t ggml_cuda_slab_work_size(int device) {
    std::lock_guard<std::mutex> lock(g_slab_mutex);
    const ggml_cuda_slab & s = g_slabs[device];
    return s.inited ? s.boundary : 0;
}

// The arena region's total size on this device -- what the expert cache may size itself against (the
// slab, not the free VRAM, owns it).  Free runs may be smaller than this total; it is the CAP.
// `mapped`, not `size`: the VA reserved above `mapped` has no physical behind it (yet).
size_t ggml_cuda_slab_arena_total(int device) {
    std::lock_guard<std::mutex> lock(g_slab_mutex);
    const ggml_cuda_slab & s = g_slabs[device];
    return s.inited ? s.mapped - s.boundary : 0;
}

// GROW the slab into the VA it reserved at creation but left unmapped, down to `headroom` of free VRAM.
//
// The reserve had to be a guess (the slab is born before the weights exist); once they ARE resident this
// measures what is actually free and maps it, so the arena gets the VRAM the model turned out not to need
// instead of it sitting idle (measured: ~1.9 GiB/card idle with an 8 GiB reserve).  Called once, from the
// cache's sizing; harmless to call again (it is a no-op when there is no room or nothing to map).
//
// This is the design's second (and last) HIP touch: the extension maps a NEW range above everything in
// use, so it moves no address -- the work region and every arena table keep theirs -- and nothing needs
// unmapping, which is what ROCm cannot do at sub-range granularity.
bool ggml_cuda_slab_extend(int device) {
    std::lock_guard<std::mutex> lock(g_slab_mutex);
    ggml_cuda_slab & s = g_slabs[device];
    if (!s.inited) {
        return false;
    }
    const size_t headroom = ggml_cuda_slab_headroom_bytes();
    if (headroom == 0) {
        return false;
    }
    ggml_cuda_set_device(device);
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    if (free_b <= headroom + s.chunk) {
        return false;
    }
    // `free_b` is what is free OUTSIDE the slab, so the gain is a DELTA: map that much MORE VA, bounded by
    // what the creation reservation left spare.  (Treating it as a total extent would compare a few GiB of
    // free VRAM against a 20+ GiB `mapped` and never fire.)
    size_t add = (free_b - headroom) / s.chunk * s.chunk;
    const size_t room = s.size - s.mapped;               // VA left unmapped by the creation reservation
    if (add > room) {
        add = room;
    }
    if (add < s.chunk) {
        return false;                                    // less than a chunk to gain: not worth a mapping
    }
    if (!ggml_cuda_vmm_map_phys(device, (CUdeviceptr) ((uintptr_t) s.base + s.mapped), add)) {
        GGML_LOG_WARN("%s: device %d: could not map a further %.2f GiB into the slab; the arena stays at "
                      "%.2f GiB (raise GGML_CUDA_SLAB_HEADROOM_MIB if this repeats)\n",
                      __func__, device, (double) add / (1ull << 30),
                      (double) (s.mapped - s.boundary) / (1ull << 30));
        return false;
    }
    // Hand the new range to the arena (coalescing with the run below it if there is one; the arena fills
    // top-down, so normally the top of the region IS the free run).
    const size_t lo = s.mapped;
    const size_t hi = s.mapped + add;
    s.mapped = hi;
    auto next = s.arena_free.lower_bound(lo);
    size_t new_lo = lo, new_hi = hi;
    if (next != s.arena_free.end() && next->first == new_hi) {
        new_hi = next->first + next->second;
        s.arena_free.erase(next);
    }
    if (s.arena_free.find(new_lo) != s.arena_free.end()) {   // not expected, but never clobber an entry
        new_lo = lo;
        new_hi = hi;
    }
    s.arena_free[new_lo] = new_hi - new_lo;
    GGML_LOG_WARN("%s: device %d: slab grew by %.2f GiB into the reserve the model did not need "
                  "(now %.2f GiB: work %.2f GiB + arena %.2f GiB; %.2f GiB left free AFTER this mapping, "
                  "which is GGML_CUDA_SLAB_HEADROOM_MIB -- not extra headroom)\n",
                  __func__, device, (double) add / (1ull << 30), (double) s.mapped / (1ull << 30),
                  (double) s.boundary / (1ull << 30), (double) (s.mapped - s.boundary) / (1ull << 30),
                  (double) (free_b - add) / (1ull << 30));
    return true;
}

// Extend every device's slab (see above).  Called by the expert cache before it sizes itself, which is the
// first moment the weights / KV / draft are all resident.
void ggml_cuda_slab_extend_all() {
    if (!ggml_cuda_slab_enabled()) {
        return;
    }
    for (int d = 0; d < GGML_CUDA_MAX_DEVICES; d++) {
        ggml_cuda_slab_extend(d);
    }
}

// Reserve + map the whole slab ONCE.  `work_min` sets the initial boundary (the first work need).
static bool ggml_cuda_slab_init_locked(int device, ggml_cuda_slab & s, size_t work_min) {
    // VMM capability first: on a device that does not report it the driver API calls below are not
    // merely useless, they abort (CU_CHECK), so the slab must decline before touching them.  This is
    // what lets the slab be offered by default without endangering other devices/backends.
    if (!ggml_cuda_info().devices[device].vmm) {
        return false;
    }
    s.device = device;
    s.chunk  = ggml_cuda_slab_chunk_bytes();
    s.unit   = ggml_cuda_slab_unit_bytes(device);
    if (s.chunk == 0 || s.unit == 0 || s.chunk % s.unit != 0) {
        return false;
    }
    // `cuMemAddressReserve` requires a power-of-two alignment; a hand-set chunk that is not one would
    // otherwise fail after the fact, so decline now and let the plain path serve the buffer.
    if ((s.chunk & (s.chunk - 1)) != 0) {
        return false;
    }
    ggml_cuda_set_device(device);
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    const size_t reserve = ggml_cuda_slab_reserve_bytes(total_b);
    if (free_b <= reserve + 2 * s.chunk) {
        return false;
    }
    const size_t want = (free_b - reserve) / s.chunk * s.chunk;   // (W + A), chunk-aligned
    const size_t w    = ggml_cuda_slab_up(work_min, s.chunk);
    if (w == 0 || w + s.chunk > want) {
        return false;
    }
    // HARD MINIMUM (see `ggml_cuda_slab_min_arena_bytes`): the work estimate plus a usable cache floor.
    // Report it and decline the slab -- `cache_unusable` then makes the expert cache stream from the host
    // instead of building an arena that a wide prefill would have to evict wholesale (the r21 abort).
    const size_t min_arena = ggml_cuda_slab_min_arena_bytes();
    if (want < w + min_arena) {
        s.cache_unusable = true;
        GGML_LOG_ERROR("%s: device %d: the movable-boundary slab does NOT fit the MoE expert cache here: "
                       "it needs the estimated work buffer (%.0f MiB) plus a %.0f MiB cache floor, but only "
                       "%.0f MiB is available after the %zu MiB reserve (free %.0f MiB of %.0f MiB total).  "
                       "Not using the slab; the experts will STREAM from the host (the stock -ncmoe path).  "
                       "Lower GGML_CUDA_SLAB_RESERVE_MIB if the reserve is larger than the weights/KV need.\n",
                       __func__, device, (double) w / (1024 * 1024), (double) min_arena / (1024 * 1024),
                       (double) want / (1024 * 1024), ggml_cuda_slab_reserve_mib(total_b),
                       (double) free_b / (1024 * 1024), (double) total_b / (1024 * 1024));
        return false;
    }
    // Reserve VA for the whole slab, but only MAP `want`: the reserve has to be a guess because the
    // weights do not exist yet, so `ggml_cuda_slab_extend` later maps the part of it the model turned out
    // not to need (down to `headroom`) and the arena uses it.  A VA reservation holds no physical, so
    // reserving wide costs nothing and cannot deprive the weights' cudaMalloc of anything.
    const size_t va = ggml_cuda_slab_headroom_bytes() > 0
                    ? (free_b - ggml_cuda_slab_headroom_bytes()) / s.chunk * s.chunk
                    : want;
    s.size   = va > want ? va : want;
    s.mapped = want;
    if (cuMemAddressReserve(&s.base, s.size, s.chunk, 0, 0) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    if (!ggml_cuda_vmm_map_phys(device, s.base, s.mapped)) {
        (void) cuMemAddressFree(s.base, s.size);
        return false;
    }
    s.boundary = w;
    s.arena_free.clear();
    s.arena_free[w] = s.mapped - w;
    s.inited = true;
    GGML_LOG_WARN("%s: device %d: slab %.2f GiB (chunk %.1f MiB): work %.2f GiB + arena %.2f GiB "
                  "(%.2f GiB reserve, %.2f GiB VA spare for `ggml_cuda_slab_extend`)\n",
                  __func__, device, (double) s.mapped / (1ull << 30), (double) s.chunk / (1024 * 1024),
                  (double) s.boundary / (1ull << 30), (double) (s.mapped - s.boundary) / (1ull << 30),
                  (double) ggml_cuda_slab_reserve_bytes(total_b) / (1ull << 30),
                  (double) (s.size - s.mapped) / (1ull << 30));
    return true;
}

// A work-region view was dropped (`reported` = the size the buffer reported, i.e. the boundary at its
// alloc).  Removing it lowers the floor the boundary may shrink to; the actual shrink happens on the next
// `work_alloc`, which knows the new need.
//
// This is a MULTISET, not a flag: more than one compute buffer is live at a time (the main context and the
// MTP draft context each own one), so a release must not make the slab believe the work region is idle.
// Tracking each view's REPORTED size is also required, not its request: a narrow view allocated while the
// boundary was wide was handed the wide range, so the graph allocator may have laid its tensors out
// anywhere in it.
// A work-region view was dropped (`need` = the size it requested).  Removing it lowers the floor the
// boundary may shrink to; the actual shrink happens on the next `work_alloc`, which knows the new need.
//
// This is a MULTISET, not a flag: more than one compute buffer is live at a time (the main context and the
// MTP draft context each own one), so a release must not make the slab believe the work region is idle --
// doing that is what corrupted the no-drop runs (a release shrank the boundary under a live view and the
// arena re-took chunks that view was still using: MTP acceptance 0.010, `////` output).
void ggml_cuda_slab_work_release(int device, size_t need) {
    std::lock_guard<std::mutex> lock(g_slab_mutex);
    ggml_cuda_slab & s = g_slabs[device];
    if (!s.inited) {
        return;
    }
    const auto it = s.work_needs.find(need);
    if (it != s.work_needs.end()) {
        s.work_needs.erase(it);
    }
}

// The work pool's base for a `need`-byte compute buffer, moving the boundary up if required.  The arena
// tables in the taken range are evicted (the callback frees their slabs back through `arena_free`).
void * ggml_cuda_slab_work_alloc(int device, size_t need) {
    size_t lo = 0, hi = 0;
    bool   move = false;
    CUdeviceptr base = 0;
    {
        std::lock_guard<std::mutex> lock(g_slab_mutex);
        ggml_cuda_slab & s = g_slabs[device];
        if (!s.inited) {
            if (!ggml_cuda_slab_init_locked(device, s, need)) {
                return nullptr;
            }
            s.work_needs.insert(need);
            return (void *) s.base;
        }
        // The boundary may shrink only as far as the TALLEST live view's REQUESTED size (the floor).  That
        // request is what bounds the view's tensor layout (`ggml_vbuffer_alloc` passes the tallocr's
        // `max_size`, rounded up plus one spare unit, as the size to allocate), and the graph allocator
        // keeps its offsets inside it; the buffer's REPORTED size is only the boundary we advertise so a
        // slightly larger layout reuses this buffer instead of asking for a new one.  Using the reported
        // size instead would pin the floor at the widest boundary any view was ever handed -- the MTP
        // draft context allocates while the boundary is wide -- and the shrink would never fire again
        // (measured: arena 17128 MiB, decode 48.5 t/s instead of 36584 MiB / 74.2 t/s).
        //
        // Getting this wrong in the other direction -- letting a release shrink the boundary under a
        // still-live view -- hands the arena chunks that view is still using: measured as the no-drop
        // corruption (MTP acceptance 0.010, `////` output) when a bool "is a view live" flag did exactly
        // that.
        const size_t floor  = s.work_needs.empty() ? 0 : *s.work_needs.rbegin();
        const size_t target = ggml_cuda_slab_up(need > floor ? need : floor, s.chunk);
        if (target <= s.boundary) {
            // Shrink to the live need so the slack goes back to the arena (the narrow post-prefill layout
            // should not hold a wide region).  Safe while a view is live: the target never goes below the
            // floor, so every live view still fits.
            if (target < s.boundary) {
                const size_t old = s.boundary;
                s.boundary = target;
                // return [target, old) to the arena free list (coalescing)
                size_t lo2 = target, hi2 = old;
                auto next = s.arena_free.lower_bound(target);
                if (next != s.arena_free.end() && next->first == hi2) {
                    hi2 = next->first + next->second;
                    next = s.arena_free.erase(next);
                }
                if (next != s.arena_free.begin()) {
                    auto prev = std::prev(next);
                    if (prev->first + prev->second == lo2) {
                        lo2 = prev->first;
                        s.arena_free.erase(prev);
                    }
                }
                s.arena_free[lo2] = hi2 - lo2;
            }
            s.work_needs.insert(need);
            return (void *) s.base;
        }
        lo   = s.boundary;
        hi   = target;
        base = s.base;
        move = true;
    }
    if (move) {
        // Evict everything the arena still holds in [lo, hi).  MUST run without `g_slab_mutex`.
        moe_cache_evict_slab_range(device, (void *) (uintptr_t) ((char *) base + lo), (void *) (uintptr_t) ((char *) base + hi));
        std::lock_guard<std::mutex> lock(g_slab_mutex);
        ggml_cuda_slab & s = g_slabs[device];
        if (s.inited && hi > s.boundary) {
            s.boundary = hi;
            // drop/clip every arena free run at or below the new boundary
            for (auto it = s.arena_free.begin(); it != s.arena_free.end(); ) {
                const size_t off = it->first;
                const size_t end = off + it->second;
                if (end <= s.boundary) {
                    it = s.arena_free.erase(it);
                } else if (off < s.boundary) {
                    const size_t keep_off = s.boundary;
                    const size_t keep_sz  = end - keep_off;
                    s.arena_free.erase(it);
                    s.arena_free[keep_off] = keep_sz;
                    break;
                } else {
                    ++it;
                }
            }
        }
        if (s.inited) {
            s.work_needs.insert(need);
        }
        return (void *) base;
    }
    return nullptr;
}

// Chunk-aligned arena allocation from [boundary, size).
//
// PLACEMENT IS THE EVICTION POLICY.  A boundary move can only take a CONTIGUOUS range of chunks at the
// bottom of the arena, so there is no "evict the coldest table" decision to make at move time -- the
// only lever is where a table sits.  Filling from the TOP DOWN therefore keeps the band just above the
// boundary free for as long as the arena has slack, and a boundary move walks through empty chunks and
// evicts nothing.  A bottom-up fill would put the very first table hard against the boundary and make
// every growth of the work pool cost a table even when the arena is nearly empty.
void * ggml_cuda_slab_arena_alloc(int device, size_t size) {
    if (size == 0) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_slab_mutex);
    ggml_cuda_slab & s = g_slabs[device];
    if (!s.inited) {
        return nullptr;
    }
    const size_t want = ggml_cuda_slab_up(size, s.unit);
    // Highest run that fits, and take its TOP.  (Erase by key: the run's offset is unique.)
    for (auto it = s.arena_free.rbegin(); it != s.arena_free.rend(); ++it) {
        if (it->second < want) {
            continue;
        }
        const size_t run_off = it->first;
        const size_t run_sz  = it->second;
        s.arena_free.erase(run_off);
        if (run_sz > want) {
            s.arena_free[run_off] = run_sz - want;
        }
        return (void *) (uintptr_t) ((char *) s.base + (run_off + run_sz - want));
    }
    return nullptr;   // the arena region is full (the work pool may take more, or the cache is capped)
}

void ggml_cuda_slab_arena_free(int device, void * ptr, size_t size) {
    if (ptr == nullptr || size == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_slab_mutex);
    ggml_cuda_slab & s = g_slabs[device];
    if (!s.inited) {
        return;
    }
    const size_t want = ggml_cuda_slab_up(size, s.unit);
    const size_t off  = (size_t) ((char *) ptr - (char *) s.base);
    if (off < s.boundary) {
        return;   // already reassigned to the work pool: nothing to give back
    }
    size_t lo = off;
    size_t hi = off + want;
    auto next = s.arena_free.lower_bound(off);
    if (next != s.arena_free.end() && next->first == hi) {
        hi = next->first + next->second;
        next = s.arena_free.erase(next);
    }
    if (next != s.arena_free.begin()) {
        auto prev = std::prev(next);
        if (prev->first + prev->second == lo) {
            lo = prev->first;
            s.arena_free.erase(prev);
        }
    }
    s.arena_free[lo] = hi - lo;
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

struct ggml_cuda_pool_leg : public ggml_cuda_pool {
    static const int MAX_BUFFERS = 256;

    int device;
    struct ggml_cuda_buffer {
        void * ptr = nullptr;
        size_t size = 0;
    };

    ggml_cuda_buffer buffer_pool[MAX_BUFFERS] = {};
    size_t pool_size = 0;

    explicit ggml_cuda_pool_leg(int device) :
        device(device) {
    }

    ~ggml_cuda_pool_leg() {
        clear_pool();
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
                b.ptr  = nullptr;
                b.size = 0;
            }
        }
    }

    void * alloc(size_t size, size_t * actual_size) override {
#ifdef DEBUG_CUDA_MALLOC
        int nnz = 0;
        size_t max_size = 0;
#endif
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
            b.ptr = nullptr;
            b.size = 0;
            return ptr;
        }
        void * ptr;
        size_t look_ahead_size = (size_t) (1.05 * size);
        look_ahead_size = 256 * ((look_ahead_size + 255)/256);
        ggml_cuda_set_device(device);
        static const bool gc_dbg = getenv("GGML_CUDA_GCDBG") != nullptr;
        const int64_t t_pl = gc_dbg ? ggml_time_us() : 0;
        cudaError_t err = ggml_cuda_device_malloc(&ptr, look_ahead_size, device);
        if (t_pl) { g_pool_us += ggml_time_us() - t_pl; g_pool_n++; }
        if (err == cudaErrorMemoryAllocation) {
            g_pool_oom_n++;
            (void)cudaGetLastError();
            const size_t cached_bytes = pool_size;
            GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[%d]: alloc of %.2f MiB failed, flushing %.2f MiB of cached buffers and retrying\n",
                           device, look_ahead_size/1024.0/1024.0, cached_bytes/1024.0/1024.0);
            CUDA_CHECK(cudaDeviceSynchronize());
            clear_pool();
            // The MoE-arena yield lives inside `ggml_cuda_device_malloc`, so this retry inherits it.
            err = ggml_cuda_device_malloc(&ptr, look_ahead_size, device);
            if (err == cudaSuccess) {
                GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[%d]: retry succeeded\n", device);
            }
        }
        CUDA_CHECK(err);
        *actual_size = look_ahead_size;
        pool_size += look_ahead_size;
#ifdef DEBUG_CUDA_MALLOC
        GGML_LOG_INFO("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, device, nnz,
                           (uint32_t)(max_size / 1024 / 1024), (uint32_t)(pool_size / 1024 / 1024), (uint32_t)(size / 1024 / 1024));
#endif
        return ptr;
    }

    void free(void * ptr, size_t size) override {
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr = ptr;
                b.size = size;
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
            bool use_peer_access = getenv("GGML_CUDA_P2P") != nullptr;
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

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int                  device,
                                                                               [[maybe_unused]] int stream_no) {
#if defined(GGML_USE_VMM)
    if (ggml_cuda_info().devices[device].vmm) {
        return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_vmm(device));
    }
#endif // defined(GGML_USE_VMM)
    return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_leg(device));
}

// destroying a cuBLAS handle while a graph is being captured in a different thread can result in a CUDA error
// this lock is used to ensure that no cuBLAS handle is destroyed while a graph is being captured

static std::mutex ggml_cuda_lock;
static std::condition_variable ggml_cuda_lock_cv;
static std::atomic<int> ggml_cuda_lock_counter;

ggml_backend_cuda_context::~ggml_backend_cuda_context() {
    ggml_cuda_mmb_release_all();
    std::unique_lock<std::mutex> lock(ggml_cuda_lock);
    ggml_cuda_lock_cv.wait(lock, []{ return ggml_cuda_lock_counter.load(std::memory_order_relaxed) == 0; });

    if (q8_1_arena != nullptr) {
        CUDA_CHECK(cudaFree(q8_1_arena));
    }
    for (char * p : q8_1_arena_retired) {
        CUDA_CHECK(cudaFree(p));
    }
    h2d_stage_free();

    for (int i = 0; i < GGML_CUDA_MAX_STREAMS; ++i) {
        if (fattn_stage[i] != nullptr) {
            CUDA_CHECK(cudaFree(fattn_stage[i]));
        }
    }

    if (copy_event != nullptr) {
        CUDA_CHECK(cudaEventDestroy(copy_event));
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
    // The size this view REQUESTED (what bounds its tensor layout); `ggml_cuda_slab_work_release` removes
    // it from the live set so the boundary can shrink once a wide view is gone.
    size_t slab_need = 0;
    std::string name;

    ggml_backend_cuda_buffer_context(int device, void * dev_ptr) :
        device(device), dev_ptr(dev_ptr),
        name(GGML_CUDA_NAME + std::to_string(device)) {
    }

    ~ggml_backend_cuda_buffer_context() {
        if (slab_view) {
            ggml_cuda_slab_work_release(device, slab_need);   // the slab owns the region; the slack may go to the arena
            return;
        }
        // A compute buffer allocated from the slab is a VIEW (handled above); everything else here is a
        // plain cudaMalloc pointer.
        CUDA_CHECK(cudaFree(dev_ptr));
    }
};

static void ggml_backend_cuda_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
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
            CUDA_CHECK(cudaMemset((char *)tensor->data + original_size, 0, padded_size - original_size));
        }
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cuda_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync((char *) tensor->data + offset, value, size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}


static void ggml_backend_cuda_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    // NEVER H2D straight from a (potentially pageable) host pointer with a 2-D copy: on ROCm a
    // pageable-source `hipMemcpy2DAsync` is pathologically slow AND can fault ("Page not present").
    // It crashed the `-sm tensor` + host-expert load, whose master is the pageable `CPU_REPACK`
    // buffer (unlike `-sm layer`'s pinned `ROCm_Host`, or the mmap path).  Gather the source rows
    // into a pinned staging buffer and copy from there -- the same pattern the per-ubatch splice
    // (`ggml_backend_cuda_set_tensor_2d_async`) already uses.  Load-path only, so a per-call pinned
    // staging buffer is fine (no ring needed).
    if (n_copies > 1) {
        const size_t full    = (n_copies - 1) * stride_data   + size;
        const size_t compact = (n_copies - 1) * stride_tensor + size;
        const size_t need    = (stride_tensor == size) ? compact : full;
        void * pin = nullptr;
        CUDA_CHECK(cudaMallocHost(&pin, need));
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
        CUDA_CHECK(cudaFreeHost(pin));
        return;
    }
    CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_get_tensor_2d(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpy2DAsync(
        data, stride_data, (const char *) tensor->data + offset, stride_tensor, size, n_copies, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static bool ggml_backend_cuda_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_cuda(src->buffer)) {
        ggml_backend_cuda_buffer_context * src_ctx = (ggml_backend_cuda_buffer_context *)src->buffer->context;
        ggml_backend_cuda_buffer_context * dst_ctx = (ggml_backend_cuda_buffer_context *)dst->buffer->context;
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
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_physical, src->data, src_physical, ggml_nbytes(src), cudaStreamPerThread));
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

    if (usage == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        // OPEN 2: the movable-boundary slab.  The compute buffer is a VIEW of the slab's work region: its
        // base is the slab base (so a growing layout keeps its tensor addresses) and its REPORTED size is
        // the boundary (so growth inside the region does not look like a realloc to the graph allocator).
        // Growing past the boundary moves the split -- evicting the arena tables in the taken chunks -- and
        // HIP is NOT called at runtime.
        if (ggml_cuda_slab_enabled()) {
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
                                            "minimum MoE cache on this device (GGML_CUDA_SLAB_MIN_ARENA_MIB); "
                                            "lower GGML_CUDA_SLAB_RESERVE_MIB if the reserve is too large");
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

// RDNA/ROCm (TODO #42): the graph allocator reserves the compute buffer from a *measure* graph, but a
// runtime graph can carry a different live-tensor set (host-expert staging, MTP taps) and need a little
// more -- measured +3.3 % (6564 -> 6780 MiB; the cli case was a 11765.52 MiB allocation).  Growing it is
// a free-then-allocate-larger, so it needs a contiguous block BIGGER than the one just released, which
// fails once the leftover VRAM belongs to the MoE expert-cache arena.  Taking the slack up front moves
// it to *before* the arena is sized.  Deliberately HIP-only: this repo is RDNA/ROCm-scoped and no other
// backend's allocation sizes change at all.  `GGML_COMPUTE_BUFFER_MARGIN_PCT=0` disables it.
static size_t ggml_backend_cuda_buffer_type_get_compute_margin_pct(ggml_backend_buffer_type_t buft) {
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
static size_t ggml_backend_cuda_buffer_type_get_compute_chunk_bytes(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
#if defined(GGML_USE_HIP)
    static size_t chunk_mib = SIZE_MAX;
    if (chunk_mib == SIZE_MAX) {
        const char * env = getenv("GGML_COMPUTE_BUFFER_CHUNK_MIB");
        chunk_mib = env != NULL ? (size_t) atoll(env) : 256;
    }
    return chunk_mib * 1024 * 1024;
#else
    return 0;
#endif
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

    // Set if NCCL fails at runtime (e.g. RCCL refusing kernel dispatch on a
    // root port without AtomicOp completer support; see ROCm/ROCm#6520).
    // Once set, NCCL is never retried: AllReduce falls back to the internal
    // pipeline (if available) or the meta backend's butterfly.  Always
    // defined (stays false) so the dispatcher needs no #ifdef in
    // non-NCCL builds.
    bool                        nccl_failed = false;

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
        // Grow without ever freeing mid-run: a previous call's peer copy may still be writing the old
        // scratch from the OTHER device's stream (a cross-device use-after-free), and cudaFree /
        // cudaDeviceSynchronize are not safe under graph capture.  Retire the old buffers to teardown.
        for (size_t i = 0; i < nb; ++i) {
            ggml_cuda_set_device(comm_ctx->dev_ids[i]);
            if (comm_ctx->ce_buf[i] != nullptr) comm_ctx->ce_old.push_back({ (int) comm_ctx->dev_ids[i], comm_ctx->ce_buf[i] });
            if (comm_ctx->ce_tmp[i] != nullptr) comm_ctx->ce_old.push_back({ (int) comm_ctx->dev_ids[i], comm_ctx->ce_tmp[i] });
            if (comm_ctx->ce_tmp2[i] != nullptr) comm_ctx->ce_old.push_back({ (int) comm_ctx->dev_ids[i], comm_ctx->ce_tmp2[i] });
            CUDA_CHECK(cudaMalloc(&comm_ctx->ce_buf[i], need));
            CUDA_CHECK(cudaMalloc(&comm_ctx->ce_tmp[i], need));
            CUDA_CHECK(cudaMalloc(&comm_ctx->ce_tmp2[i], need));
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

// Hybrid: NCCL/RCCL (P2P, BF16 round-trip) for bandwidth-bound large tensors,
// plus the internal host-staged pipeline (low per-call latency) for
// latency-bound small tensors.  The per-size routing happens in
// ggml_backend_cuda_comm_allreduce_tensor: small tensors go to the internal
// pipeline directly; everything else falls through to try_allreduce, which we
// set to NCCL when available.
static void ggml_backend_cuda_comm_init_hybrid(ggml_backend_cuda_comm_context * ret) {
    const bool has_nccl     = ggml_backend_cuda_comm_init_nccl(ret);
    const bool has_internal = ggml_backend_cuda_comm_init_internal(ret);
#ifdef GGML_USE_NCCL
    if (has_nccl) {
        // Large tensors -> NCCL (P2P).  Small tensors are routed to the
        // internal pipeline by the dispatcher regardless of this pointer.
        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_nccl;
    } else if (!has_internal) {
        // Neither path came up; butterfly fallback below (try_allreduce stays
        // as-is until comm_init_none is called by the caller).
        ret->try_allreduce = nullptr;
    }
#else
    // No NCCL/RCCL compiled in (has_nccl is always false); only the internal
    // pipeline can serve AR.
    (void) has_nccl;
    if (!has_internal) {
        ret->try_allreduce = nullptr;
    }
#endif
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
        // Platform default: Linux uses the hybrid (NCCL for large, internal
        // for small); otherwise (generally Windows) internal only.
#if defined(__linux__)
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
            // Start from the hybrid setup (NCCL for large tensors, internal pipeline for
            // latency-bound small tensors) so that a CE failure is a no-op downgrade, then swap
            // the large-tensor arm to CE.  If CE cannot be set up (e.g. no peer access) we keep
            // the hybrid path -- falling all the way back to the butterfly would cost ~2.5x.
            ggml_backend_cuda_comm_init_hybrid(ret);
            if (ggml_backend_cuda_comm_init_ce(ret)) {
                ok = true;
            } else {
                GGML_LOG_WARN("GGML_CUDA_ALLREDUCE=ce unavailable (needs >= 2 peer-accessible devices); "
                              "using the hybrid (NCCL + internal) path\n");
                ok = ret->try_allreduce != nullptr;
            }
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
    if (getenv("GGML_CUDA_NO_PINNED") != nullptr) {
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
                /* .get_max_size        = */ NULL, // defaults to SIZE_MAX
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
static const char * g_cuda_splice_env      = getenv("GGML_CUDA_SPLICE_GATHER");
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

template<ggml_type compute_type>

static void ggml_cuda_mul_mat_cublas_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    using traits = batched_mul_mat_traits<compute_type>;
    using cuda_t = typename traits::cuda_type;

    GGML_ASSERT(ggml_is_contiguous(dst));

    // Byte offsets and tensor dimensions are currently used in an inconsistent way for dst.
    // As long as dst is contiguous this does not matter though.

    GGML_TENSOR_BINARY_OP_LOCALS

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
    if (src1->type == compute_type) {
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
        CUBLAS_CHECK(
            cublasSgemm(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                    ne01, ne11, ne10,
                    (const float *) alpha, (const float *) src0_ptr, s01,
                                           (const float *) src1_ptr, s11,
                    (const float *) beta,  (float       *)  dst_ptr, ne0));
    } else if (ne12 == 1 && ne13 == 1) {
        CUBLAS_CHECK(
            cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                    ne01, ne11, ne10,
                    alpha, src0_ptr, cu_data_type_a, s01,
                           src1_ptr, cu_data_type_b, s11,
                    beta,   dst_ptr, cu_data_type,   ne0,
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
                beta,   dst_ptr, cu_data_type,   ne0, ne1*ne0, // strideC
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
                beta,  (      void **) (ptrs_dst.get() + 0*ne23), cu_data_type,   ne0,
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

    const char * env_c = getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE");
    if (env_c != nullptr) {
        std::string env_cpp = env_c;
        for (char & c : env_cpp) {
            c = std::tolower(c);
        }
        if (env_cpp == "f32" || env_cpp == "fp32") {
            compute_type = GGML_TYPE_F32;
        } else if (env_cpp == "f16" || env_cpp == "fp16") {
            compute_type = GGML_TYPE_F16;
        } else if (env_cpp == "bf16") {
            compute_type = GGML_TYPE_BF16;
        } else if (env_cpp != "auto") {
            GGML_LOG_WARN("%s: unknown value for GGML_CUDA_CUBLAS_COMPUTE_TYPE: %s", __func__, env_cpp.c_str());
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

static bool ggml_cuda_should_fuse_mul_mat(const ggml_tensor * ffn_up,
                                          const ggml_tensor * ffn_gate,
                                          const ggml_tensor * glu,
                                          const ggml_tensor * ffn_up_bias = nullptr,
                                          const ggml_tensor * ffn_gate_bias = nullptr,
                                          const ggml_tensor * ffn_up_scale = nullptr,
                                          const ggml_tensor * ffn_gate_scale = nullptr) {
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

// RDNA3_5 (Strix Halo, gfx1151): the dense gate+up+GLU mmvq fusion is single-token-only
// (mmvq.cu restricts fusion to ncols_dst == 1) and its fused kernel does not reproduce the
// standalone mul_mat_vec_q arithmetic, so a 1-token decode and an n-token speculative verify
// batch of the same layer are not bit-identical - the decode==verify invariant greedy MTP
// depends on.  Measured 2026-09-12: W=1 8abc6206... vs W=8 453eaa61...; skipping it (together
// with the weighted-down MoE tail, gated in ggml_cuda_mul_mat_id_weighted_rdna3_5_ok) restores
// W=1..8 == 453eaa61... .  Skip it on that arch unless explicitly re-enabled for A/B.
static bool ggml_cuda_rdna3_5_dense_glu_disabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_ENABLE_RDNA3_5_SINGLE_TOKEN_FUSIONS");
        return env != nullptr && std::atoi(env) != 0;
    }();
    if (enabled) {
        return false;
    }
    return GGML_CUDA_CC_IS_RDNA3_5(ggml_cuda_info().devices[ggml_cuda_get_device()].cc);
}

static bool ggml_cuda_should_fuse_mul_mat_vec_f(const ggml_tensor * tensor) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool is_mul_mat_id = tensor->op == GGML_OP_MUL_MAT_ID;

    bool use_mul_mat_vec_f =
        (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16) &&
        src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

    const int cc      = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    use_mul_mat_vec_f = use_mul_mat_vec_f && ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, is_mul_mat_id ? src1->ne[2] : src1->ne[1]);

    //we only support fusion for ncols_dst = 1
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] != 1) {
        return false;
    }


    return use_mul_mat_vec_f;
}

// verify_band: callers that only pre-fill the mmvq Q8_1 activation cache (norm -> Q8_1, gated unary -> Q8_1)
// and leave the matmul itself unchanged.  Their kernels are row-generic, so on RDNA4 the verify band
// (2..8 tokens) takes them too; the quantized values are the ones quantize_q8_1 would write, so W = 1..8
// stay bit-identical.  GGML_CUDA_FUSE_Q8_1_VERIFY=0 turns it off.
static bool ggml_cuda_should_fuse_mul_mat_vec_q(const ggml_tensor * tensor, const bool verify_band = false) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
                                   ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) &&
                                   src0->view_src;

    bool use_mul_mat_vec_q = ggml_is_quantized(src0->type) && !bad_padding_clear && src1->type == GGML_TYPE_F32 &&
                             dst->type == GGML_TYPE_F32 && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;

    // fusion is not universally faster on Pascal
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (cc <= GGML_CUDA_CC_PASCAL) {
        return false;
    }
    //we only support fusion for ncols_dst = 1 (the RDNA4 verify band too for verify_band callers)
    static const bool q8_1_verify = getenv("GGML_CUDA_FUSE_Q8_1_VERIFY") == nullptr || atoi(getenv("GGML_CUDA_FUSE_Q8_1_VERIFY")) != 0;
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1 &&
            !(verify_band && q8_1_verify && GGML_CUDA_CC_IS_RDNA4(cc) && dst->ne[1] <= MMVQ_MAX_BATCH_SIZE)) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] > get_mmvq_mmid_max_batch(src0->type, cc)) {
        return false;
    }

    return use_mul_mat_vec_q;
}

static bool ggml_cuda_match_shared_expert(const ggml_cgraph * graph, int routed_idx, int shared_idx) {
    if (routed_idx + 2 >= graph->n_nodes || shared_idx + 2 >= graph->n_nodes || shared_idx < routed_idx + 3) {
        return false;
    }
    const int nodes[] = { routed_idx, routed_idx + 1, routed_idx + 2, shared_idx, shared_idx + 1, shared_idx + 2 };
    const ggml_op ops[] = { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU,
                           GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU };
    const int outputs[] = { routed_idx + 2, shared_idx + 2 };
    if (!ggml_can_fuse_subgraph_ext(graph, nodes, 6, ops, outputs, 2)) {
        return false;
    }

    const ggml_tensor * routed = graph->nodes[routed_idx + 2];
    const ggml_tensor * shared = graph->nodes[shared_idx + 2];
    const ggml_tensor * gate = routed->src[0];
    const ggml_tensor * up = routed->src[1];
    const ggml_tensor * shared_gate = shared->src[0];
    const ggml_tensor * shared_up = shared->src[1];
    const auto is_pair = [&](const ggml_tensor * a, const ggml_tensor * b, int idx) {
        return (a == graph->nodes[idx] && b == graph->nodes[idx + 1]) ||
               (b == graph->nodes[idx] && a == graph->nodes[idx + 1]);
    };
    if (!is_pair(gate, up, routed_idx) || !is_pair(shared_gate, shared_up, shared_idx) ||
            !ggml_cuda_should_fuse_mul_mat(up, gate, routed) ||
            !ggml_cuda_should_fuse_mul_mat(shared_up, shared_gate, shared) ||
            !up->src[0]->buffer ||
            !ggml_cuda_should_fuse_mul_mat_vec_q(up)) {
        return false;
    }
    const ggml_tensor * input = up->src[1];
    const ggml_tensor * weight = up->src[0];
    const ggml_tensor * shared_weight = shared_up->src[0];
    if (input->op != GGML_OP_RESHAPE || input->src[0] != shared_up->src[1] ||
            input->ne[1] != 1 || input->ne[3] != 1 || !ggml_is_contiguous(input) ||
            !ggml_is_contiguous(shared_up->src[1]) || !ggml_is_matrix(shared_up->src[1]) ||
            weight->type != shared_weight->type || weight->ne[0] != shared_weight->ne[0] ||
            weight->ne[1] != shared_weight->ne[1] || weight->nb[1] != shared_weight->nb[1] || weight->ne[3] != 1 ||
            !ggml_is_matrix(shared_weight) || !ggml_is_contiguous(shared_weight) ||
            !ggml_is_contiguous(shared_gate->src[0]) || !ggml_is_contiguous(routed) || !ggml_is_contiguous(shared)) {
        return false;
    }
    if (shared_weight->op != GGML_OP_NONE || shared_gate->src[0]->op != GGML_OP_NONE ||
            ggml_get_glu_op(routed) != ggml_get_glu_op(shared) ||
            ggml_get_op_params_f32(routed, 3) != ggml_get_op_params_f32(shared, 3)) {
        return false;
    }
    return true;
}

// True iff ggml_cuda_mul_mat() below would run this MUL_MAT through ggml_cuda_mul_mat_q: the same
// predicate chain in the same order.  Keep the two in sync.
static bool ggml_cuda_mul_mat_takes_mmq(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
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
    if (ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, ne11_mmvf) || src0->ne[1] == 1) {
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
    if (ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, ne11_mmvf)) {
        // The custom F16 vector kernel can be used over batched cuBLAS GEMM.
        // But this is only faster for GPUs without tensor cores or with a thin src0 matrix (particularly KQV in attention)
        MMB_T(ggml_cuda_mul_mat_vec_f(ctx, src0, src1, nullptr, dst), 1);
        return;
    }
    // A transposed vector can still use MMVQ (i.e. ne01 == 1)
    if (ne01 == 1 && ne11 > MMVF_MAX_BATCH_SIZE && ne2 == 1 && ne3 == 1
            && src0->type == GGML_TYPE_F32
            && ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst)
            && ggml_cuda_should_use_mmvf(src1->type, cc, src1->ne, src1->nb, /*ne11 =*/ 1)) {
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

    if (ggml_cuda_should_use_mmf(src0->type, cc, WARP_SIZE, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
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

    // wip/moe-expert-cache prefill-routing seed (MOE_EXPERT_CACHE_PREFILL_SEED=1): tally the prompt's
    // routing on the device so the first decode token can bulk-admit its hottest experts.  Under
    // `-sm tensor` the scheduler's block-06 staging intercepts the prefill upload before the host
    // `moe_cache_update` hook, so this is the one place the routing device tensor is in hand.  Prefill
    // only (the cache owns the decode band); no host readback, no sync - the histogram lands on the
    // same compute stream and the first decode flush reads it after the inter-token synchronize.
    if (moe_cache_enabled() && ids->ne[1] > MOE_EXPERT_CACHE_MAX_TOK) {
        moe_cache_tally_prefill(dst, src0, ids, ctx.device, ctx.stream());
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
        moe_cache_devmap dm{};
        if (moe_cache_get_table(dst, src0, ctx.device, &arena, &n_slots, &eb, &remap, &nu, &nt, &dm) &&
            (dm.slot_dev != nullptr || remap == nullptr ||
             (nu == ids->ne[0] && nt == ids->ne[1]))) {
            if (dm.slot_dev != nullptr) {
                // Device-remap mode: build the slot-remapped ids from the routing + the device slot map.
                moe_cache_launch_remap(ids->data, ids->nb[0], ids->nb[1], &dm,
                                       ids->ne[0], ids->ne[1], (int32_t *) remap, ctx.stream());
            }
            // Keep `ne[2]`/`nb` exactly as the full table so the dispatcher's kernel-family
            // heuristics (which read `ne[2]`/`n_experts`) are unchanged and the arithmetic stays
            // bit-identical; only the base pointer moves to the arena, and the remapped ids stay
            // below `slots`, so the kernel never reads an expert outside the arena.  Unused expert
            // slots are never referenced by the ids and are only skipped over.
            // NOTE: `moe_cache_read_check` is a bring-up diagnostic; call it only with CUDA graphs
            // disabled (it stream-synchronizes, which aborts during capture).
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

        if (ggml_cuda_should_use_mmf(src0->type, cc, WARP_SIZE, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
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
                if (expert_to_use == i02) {
                    ids_from_sorted_host[i12*n_expert_used + iex] = ids_to_sorted_host.size();
                    ids_to_sorted_host.push_back(i12*ne11 + iex % ne11);
                    tokens_per_expert[i02]++;
                    break;
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

static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
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

    CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice, cuda_ctx->stream()));
}

static void ggml_backend_cuda_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

static void ggml_backend_cuda_set_tensor_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

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
    // GGML_CUDA_SPLICE_GATHER=0 restores the plain 2-D copy.
    const char * splice_env  = g_cuda_splice_env;
    const bool splice_gather = (splice_env == nullptr || atoi(splice_env) != 0) && n_copies > 1 && stride_tensor == size;
    if (splice_gather) {
        const cudaStream_t h2d_stream = cuda_ctx->copy_stream();
        const size_t full    = (n_copies > 0 ? (size_t)(n_copies-1)*stride_data    : 0) + size;
        const size_t compact = (n_copies > 0 ? (size_t)(n_copies-1)*stride_tensor : 0) + size;
        // the compact path only ever materializes the gathered slice; the strided path memcpys the whole range
        const size_t need    = (stride_tensor == size) ? compact : (full > compact ? full : compact);
        // A ring of pinned slots so the host can gather copy N+1 while the device still reads the
        // pinned buffer of copy N (a single shared buffer corrupts/faults -- and indeed did).
        // Depth is 8: more slots = fewer host waits = closer to the
        // no-wait ceiling, at `depth x span` bytes of pinned memory.
        struct pin_slot { void * buf; size_t sz; hipEvent_t ev; };
        // exp36: keep one ring per device.  A shared ring recorded device 0's event on device 1's copy
        // stream (and vice versa), which either errors or is a no-op -- so the queued H2D (PINHOST=2)
        // never actually overlapped.  Cheap to fix and it makes the queue semantics honest.
        static thread_local std::vector<std::vector<pin_slot>> rings;   // [device][slot]
        static thread_local std::vector<int> pin_idx_dev;               // [device]
        static thread_local int ring_n = 0;
        if (ring_n == 0) {
            ring_n = 8;
        }
        const int dev = cuda_ctx->device;
        if ((int) rings.size()       <= dev) rings.resize(dev + 1);
        if ((int) pin_idx_dev.size() <= dev) pin_idx_dev.resize(dev + 1, 0);
        if (rings[dev].empty()) rings[dev].resize(ring_n, pin_slot{nullptr, 0, nullptr});
        const int k = pin_idx_dev[dev]++ % ring_n;
        pin_slot & s = rings[dev][k];
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

    CUDA_CHECK(cudaMemcpy2DAsync(
        data, stride_data, (const char *) tensor->data + offset, stride_tensor, size, n_copies, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

static bool ggml_backend_cuda_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    if (!ggml_backend_is_cuda(backend_src) || !ggml_backend_is_cuda(backend_dst)) {
        return false;
    }

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
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_physical, src->data, src_physical, ggml_nbytes(dst), cuda_ctx_src->stream()));
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

    CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));

    GGML_UNUSED(backend);
}

static bool ggml_cuda_is_view_or_noop(const ggml_tensor * t) {
    return ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;
}

#ifdef USE_CUDA_GRAPH
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

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if (ggml_cuda_is_view_or_noop(node)) {
            continue;
        }

        // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
        if (node->op == GGML_OP_MUL_MAT_ID) {
            const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
            if (ggml_cuda_mul_mat_id_needs_sync(node, cc)) {
                // the mul_mat_id fallback path synchronizes the stream, so we cannot use CUDA graphs
                // ref: https://github.com/ggml-org/llama.cpp/pull/18958
                use_cuda_graph = false;
#ifndef NDEBUG
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to unsupported node type\n", __func__);
#endif
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
static ggml_cuda_graph_key ggml_cuda_graph_get_key(ggml_cgraph * cgraph) {
    return ggml_cuda_graph_key { cgraph->nodes[0], ggml_cuda_graph_n_tokens(cgraph) };
}

// Whether a CUDA/HIP graph should skip the graph-capture path.  Only a true PRE-FILL (a
// varying ubatch, where capture never amortises) is skipped; single-token decode and the
// spec-verify widths (n_tokens <= MMVQ_MAX_BATCH_SIZE, a fixed shape every step) keep HIP
// graph replay.  GGML_CUDA_DISABLE_VERIFY_GRAPHS=1 restores the pre-r20 behaviour where
// every multi-token graph skipped the graph path.
static bool ggml_cuda_graph_is_multi_token(const ggml_cgraph * cgraph) {
    static const bool verify_graphs_off = getenv("GGML_CUDA_DISABLE_VERIFY_GRAPHS") != nullptr;
    return ggml_cuda_graph_n_tokens(cgraph) > (verify_graphs_off ? 1 : MMVQ_MAX_BATCH_SIZE);
}

static bool ggml_cuda_graph_update_required(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph) {
    bool res = false;

    const ggml_cuda_graph_key graph_key = ggml_cuda_graph_get_key(cgraph);
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

    if (cgraph->uid != 0 &&
        cgraph->uid == graph->uid) {
        GGML_LOG_DEBUG("CUDA Graph id %zu reused\n", cgraph->uid);
        GGML_ASSERT((int)graph->node_props.size() == cgraph->n_nodes);
        return false;
    }

    graph->uid = cgraph->uid;

    // Check if the graph size has changed
    if ((int)graph->node_props.size() != cgraph->n_nodes) {
        res = true;
        graph->node_props.resize(cgraph->n_nodes);
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_cuda_graph::node_properties prop = {};
        memcpy(&prop.node, cgraph->nodes[i], sizeof(ggml_tensor));

        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (cgraph->nodes[i]->src[j]) {
                prop.node_src_data_ptrs[j] = cgraph->nodes[i]->src[j]->data;
                memcpy(prop.node_src_ne[j], cgraph->nodes[i]->src[j]->ne, sizeof(prop.node_src_ne[j]));
                memcpy(prop.node_src_nb[j], cgraph->nodes[i]->src[j]->nb, sizeof(prop.node_src_nb[j]));
            }
        }

        if (res || memcmp(&graph->node_props[i], &prop, sizeof(prop)) != 0) {
            graph->node_props[i] = prop;
            res = true;
        }
    }

    return res;
}

static void ggml_cuda_graph_update_executable(ggml_backend_cuda_context * cuda_ctx, const ggml_cuda_graph_key & graph_key) {
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

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
        if (graph->instance != nullptr) {
            CUDA_CHECK(cudaGraphExecDestroy(graph->instance));
            graph->instance = nullptr;
        }
        CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));
        return;
    }
#endif // GGML_USE_HIP

#if CUDART_VERSION >= 12000
    cudaGraphExecUpdateResultInfo result_info;
    cudaError_t stat = cudaGraphExecUpdate(graph->instance, graph->graph, &result_info);
#else
    cudaGraphNode_t errorNode;
    cudaGraphExecUpdateResult result_info;
    cudaError_t stat = cudaGraphExecUpdate(graph->instance, graph->graph, &errorNode, &result_info);
#endif // CUDART_VERSION >= 12000

    if (stat == cudaErrorGraphExecUpdateFailure) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: CUDA graph update failed\n", __func__);
#endif

        // The pre-existing graph exec cannot be updated due to violated constraints
        // so instead clear error and re-instantiate
        (void)cudaGetLastError();
        CUDA_CHECK(cudaGraphExecDestroy(graph->instance));
        graph->instance = nullptr;
        CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));
    } else {
        GGML_ASSERT(stat == cudaSuccess);
    }
}
#endif // USE_CUDA_GRAPH

static bool ggml_cuda_should_fuse_rope_set_rows(const ggml_tensor * rope,
                                                const ggml_tensor * view,
                                                const ggml_tensor * set_rows) {
    // The fusion is selected by ggml_cuda_check_fusion_memory_ranges(), i.e. by buffer addresses, so
    // its rounding path must be identical to the unfused chain.  Keep a kill switch to bisect it.
    static const bool disabled = getenv("GGML_CUDA_DISABLE_ROPE_SET_ROWS") != nullptr;
    if (disabled) {
        return false;
    }

    if (rope->op != GGML_OP_ROPE || view->op != GGML_OP_VIEW || set_rows->op != GGML_OP_SET_ROWS) {
        return false;
    }
    // ne3 not tested
    if (rope->src[0]->ne[3] != 1) {
        return false;
    }

    if (set_rows->type != GGML_TYPE_F32 && set_rows->type != GGML_TYPE_F16 && set_rows->type != GGML_TYPE_BF16) {
        return false;
    }

    if (set_rows->src[1]->type != GGML_TYPE_I64) {
        return false;
    }

    // The view should flatten two dims of rope into one dim
    if (!ggml_is_contiguous(view) || view->ne[0] != rope->ne[0] * rope->ne[1]) {
        return false;
    }

    // Only norm/neox/multi shaders have the fusion code
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX && mode != GGML_ROPE_TYPE_IMROPE) {
        return false;
    }

    return true;
}

static bool ggml_cuda_should_fuse_mul_q8_1(const ggml_tensor * mul,
                                           const ggml_tensor * mm) {
    if (mul->op != GGML_OP_MUL || (mm->op != GGML_OP_MUL_MAT && mm->op != GGML_OP_MUL_MAT_ID)) {
        return false;
    }

    // The matmul must run on the mmvq path (the only consumer of the arena).
    if (!ggml_cuda_should_fuse_mul_mat_vec_q(mm, true)) {
        return false;
    }

    if (mul->type != GGML_TYPE_F32 || !ggml_is_quantized(mm->src[0]->type)) {
        return false;
    }

    // The matmul's activation must be the MUL (or a no-op view of it).
    const ggml_tensor * mm_src1 = mm->src[1];
    const ggml_tensor * src1 = mm_src1;
    while (src1 != nullptr && src1->view_src != nullptr) {
        src1 = src1->view_src;
    }
    if (src1 != mul) {
        return false;
    }

    // Same-shape, contiguous F32 inputs; rows must align with Q8_1 blocks.
    if (mul->src[0]->type != GGML_TYPE_F32 || mul->src[1]->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(mul->src[0], mul->src[1]) ||
        !ggml_is_contiguous(mul->src[0]) || !ggml_is_contiguous(mul->src[1]) ||
        !ggml_is_contiguous(mul) || mul->ne[0] % QK8_1 != 0 ||
        ggml_nelements(mul) != ggml_nelements(mm_src1)) {
        return false;
    }

    return true;
}

// If the MUL output feeds a single mmvq matmul (directly or via a no-op
// reshape at the next node), return that matmul; otherwise return nullptr.
static const ggml_tensor * ggml_cuda_find_mul_q8_1_matmul(const ggml_cgraph * cgraph,
                                                          int mul_idx, const ggml_tensor * mul) {
    const int n = cgraph->n_nodes;
    const ggml_tensor * n1 = (mul_idx + 1 < n) ? cgraph->nodes[mul_idx + 1] : nullptr;
    if (n1 == nullptr) {
        return nullptr;
    }

    const ggml_tensor * mm = nullptr;
    if (n1->op == GGML_OP_MUL_MAT) {
        mm = n1;
    } else if (n1->op == GGML_OP_RESHAPE && mul_idx + 2 < n && cgraph->nodes[mul_idx + 2]->op == GGML_OP_MUL_MAT) {
        mm = cgraph->nodes[mul_idx + 2];
    } else {
        return nullptr;
    }

    // The MUL output must have exactly one consumer (the matmul or its view).
    int uses = 0;
    for (int j = 0; j < n; ++j) {
        const ggml_tensor * t = cgraph->nodes[j];
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (t->src[s] == mul) {
                uses++;
            }
        }
    }
    if (uses != 1) {
        return nullptr;
    }

    if (!ggml_cuda_should_fuse_mul_q8_1(mul, mm)) {
        return nullptr;
    }
    return mm;
}

static bool ggml_cuda_should_fuse_rms_norm_mul_rope(const ggml_tensor * rms_norm,
                                                    const ggml_tensor * mul,
                                                    const ggml_tensor * rope) {
    // Address-selected like the rope_set_rows fusion above; kill switch for bisection.
    static const bool disabled = getenv("GGML_CUDA_DISABLE_RMS_NORM_MUL_ROPE") != nullptr;
    if (disabled) {
        return false;
    }
    if (rms_norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || rope->op != GGML_OP_ROPE) {
        return false;
    }

    if (rms_norm->src[0]->type != GGML_TYPE_F32 || rms_norm->type != GGML_TYPE_F32 ||
        mul->src[0]->type != GGML_TYPE_F32 || mul->src[1]->type != GGML_TYPE_F32 ||
        mul->type != GGML_TYPE_F32 || rope->type != GGML_TYPE_F32) {
        return false;
    }

    if (rope->src[0] != mul) {
        return false;
    }

    //if rms norm is the B operand, then we don't handle broadcast
    if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
        return false;
    }

    if (!ggml_are_same_shape(rms_norm, mul)) {
        return false;
    }

    //rms_norm kernel assumes contiguous rows
    if (!ggml_is_contiguous_rows(rms_norm->src[0]) ||
        !ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
        return false;
    }

    // the fused kernel handles the norm/neox rope modes only
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) {
        return false;
    }

    const int n_dims = ((const int32_t *) rope->op_params)[1];
    if (n_dims % 2 != 0 || rope->src[0]->ne[0] % 2 != 0) {
        return false;
    }

    // ggml_rope_set_offset is not yet supported in the fused kernel
    const int n_offs = ((const int32_t *) rope->op_params)[15];
    if (n_offs != 0) {
        return false;
    }

    return true;
}

// match gated_delta_net + the strided cpy that scatters its state snapshots into the cache
// (slot i -> rollback group i, slot 0 newest), so the kernel can write them and skip the cpy.
static int ggml_cuda_try_gdn_cache_fusion(
        const ggml_cgraph * cgraph, int node_idx, ggml_cuda_gated_delta_net_fused_cache & fused_state_cpy) {
    const ggml_tensor * gdn = cgraph->nodes[node_idx];
    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    if (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 ||
        (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }

    const ggml_tensor * src_v     = gdn->src[2];
    const int64_t       S_v       = src_v->ne[0];
    const int64_t       H         = src_v->ne[1];
    const int64_t       n_tokens  = src_v->ne[2];
    const int64_t       n_seqs    = src_v->ne[3];
    const int64_t       D         = S_v * S_v * H;
    const int64_t       K         = ggml_get_op_params_i32(gdn, 0); // snapshot slot count
    const int64_t       n_written = std::min<int64_t>(n_tokens, K); // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    const size_t tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    // snapshot cpy is the first real node after the gdn (skip views/no-ops)
    const ggml_tensor * cpy  = nullptr;
    int                 skip = 0;
    for (int j = node_idx + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        if (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
        cpy  = n;
        skip = j - node_idx;
    }
    if (cpy == nullptr) {
        return 0;
    }

    const ggml_tensor * src = cpy->src[0]; // view of the gdn snapshot tail
    const ggml_tensor * dst = cpy->src[1]; // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    if (src->op != GGML_OP_VIEW || src->view_src != gdn || src->view_offs != tail_off ||
        !ggml_is_contiguous(src)) {
        return 0;
    }

    // dst is the [D, n_seqs, n_written] cache view; require nb[1] == D (the per-seq stride the kernel
    // assumes). ggml_cpy pins src to the same element count.
    const std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    if (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->data == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) || dst->nb[1] != (size_t) ggml_row_size(GGML_TYPE_F32, D)) {
        return 0;
    }

    fused_state_cpy.data        = (float *) dst->data; // rollback group 0 (newest)
    fused_state_cpy.slot_stride = K > 1 ? (int64_t) (dst->nb[2] / sizeof(float)) : 0;
    return skip;
}

static bool ggml_cuda_topk_moe_fusion_disabled() {
    // A/B kill-switch for the fused MoE router.  The fused kernel is bit-identical to the generic
    // softmax -> argsort -> get_rows -> norm chain (both the softmax reduction/normalization and
    // the argsort tie-break are matched), so this only exists to compare the two execution paths.
    static const bool disabled = getenv("GGML_CUDA_DISABLE_TOPK_MOE_FUSION") != nullptr &&
                                 std::atoi(getenv("GGML_CUDA_DISABLE_TOPK_MOE_FUSION")) != 0;
    return disabled;
}

static bool ggml_cuda_topk_moe_fusion(const struct ggml_cgraph * cgraph, int node_idx, ggml_cuda_topk_moe_args & args) {
    args.sigmoid         = false;
    args.sqrt_softplus   = false;
    args.softmax         = false;
    args.delayed_softmax = false;
    args.prob_bias       = false;
    args.norm            = false;

    const int      n_nodes = cgraph->n_nodes;
    ggml_tensor ** nodes   = cgraph->nodes;

    if (nodes[node_idx]->op == GGML_OP_SOFT_MAX) {
        args.softmax = true;
    }

    if (nodes[node_idx]->op == GGML_OP_UNARY) {
        const ggml_unary_op unary_op = ggml_get_unary_op(nodes[node_idx]);
        if (unary_op == GGML_UNARY_OP_SIGMOID) {
            args.sigmoid = true;
        } else if (unary_op == GGML_UNARY_OP_SOFTPLUS && node_idx + 1 < n_nodes &&
                   nodes[node_idx + 1]->op == GGML_OP_SQRT && nodes[node_idx + 1]->src[0] == nodes[node_idx]) {
            // sqrt(softplus(x)) scoring (DeepSeek-V4)
            args.sqrt_softplus = true;
            node_idx++;
        } else {
            return false;
        }
    }

    if (nodes[node_idx]->op == GGML_OP_ARGSORT) {
        args.delayed_softmax = true;
    }

    node_idx++;

    if (args.sigmoid || args.sqrt_softplus || args.softmax) {
        // SOFTMAX -> RESHAPE
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_RESHAPE ||
                nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx];
        node_idx++;

        if (node_idx >= n_nodes) {
            return false;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        if (nodes[node_idx]->op == GGML_OP_ADD && nodes[node_idx]->src[0] == nodes[node_idx - 2]) {
            args.prob_bias = true;
            node_idx++;
        }
        // RESHAPE/ADD -> ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_ARGSORT) {
            return false;
        }

        if (args.prob_bias && nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        } else if (!args.prob_bias && nodes[node_idx]->src[0] != nodes[node_idx - 2]) {
            return false;
        }

        node_idx++;

        // ARGSORT-> VIEW
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
                nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;

        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_GET_ROWS) {
            return false;
        }

        // GET_ROWS
        if (nodes[node_idx]->src[0] != probs_reshaped || nodes[node_idx]->src[1] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;
    } else if (args.delayed_softmax) {
        if (node_idx - 2 < 0) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx - 2];

        // VIEW->ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
            nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;

        // GET_ROWS
        if (node_idx >= n_nodes || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
                nodes[node_idx]->src[0] != probs_reshaped) {
            return false;
        }
        node_idx++;

        static const std::vector<ggml_op> remaining_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

        for (const ggml_op op : remaining_ops) {
            if (node_idx >= n_nodes || nodes[node_idx]->op != op || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
                return false;
            }
            node_idx++;
        }
    }

    // At this point we can check for norm + scale. Everything is now at least valid till the norm
    if (node_idx >= n_nodes) {
        return true;
    }

    if (nodes[node_idx]->op == GGML_OP_RESHAPE) {
        //check RESHAPE->SUM_ROWS->CLAMP->DIV->RESHAPE
        static const std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP };

        args.norm = true;
        for (const ggml_op op : norm_ops) {
            if (nodes[node_idx]->op == op && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
                node_idx++;
            } else {
                args.norm = false;
                return true;
            }
        }

        // DIV <- CLAMP, RESHAPE
        if (nodes[node_idx]->op != GGML_OP_DIV || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
            nodes[node_idx]->src[0] != nodes[node_idx - 3]) {
            args.norm = false;
            return true;
        }
        node_idx++;

        if (nodes[node_idx]->op != GGML_OP_RESHAPE || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            args.norm = false;
            return true;
        }

        node_idx++;
    }

    if (nodes[node_idx]->op == GGML_OP_SCALE && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
        args.scale = true;
    }

    return true;
}

// returns whether the write (out) nodes overwrite the read nodes in operation
static bool ggml_cuda_check_fusion_memory_ranges(const ggml_cgraph * cgraph,
                                                 const int           node_idx,
                                                 const int           node_count,
                                                 const int *         out_nodes,
                                                 const int           out_count,
                                                 const bool          is_topk_moe = false) {
    auto nodes_overlap = [&](const ggml_tensor * a, const ggml_tensor * b) {
        const int64_t a_start = (int64_t) a->data;
        const int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);

        const int64_t b_start = (int64_t) b->data;
        const int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        if ((b_start <= a_start && a_start < b_end) || (a_start <= b_start && b_start < a_end)) {
            return true;
        }

        return false;
    };

    bool is_ok = true;
    // one block reads all logits before it writes, so logits may alias the out nodes
    const ggml_tensor * logits_may_alias = nullptr;
    if (is_topk_moe && ggml_nrows(cgraph->nodes[node_idx]) <= TOPK_MOE_ROWS_PER_BLOCK) {
        logits_may_alias = cgraph->nodes[node_idx]->src[0];
    }

    for (int i = 0; i < out_count; ++i) {
        const ggml_tensor * dst = cgraph->nodes[out_nodes[i]];

        for (int j = node_idx; j < node_idx + node_count; ++j) {
            // Loop over all srcs of all nodes in the fusion. If the src overlaps
            // the destination and the src is not an intermediate node that's being
            // elided, then disable fusion.

            for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                const ggml_tensor * src = cgraph->nodes[j]->src[src_idx];

                if (!src || src->op == GGML_OP_NONE || src == logits_may_alias) {
                    continue;
                }

                if (nodes_overlap(dst, src)) {
                    bool found = false;

                    for (int k = node_idx; k < j; ++k) {
                        if (cgraph->nodes[k] == src) {
                            found = true;
                            break;
                        }
                    }

                    if (!found) {
                        is_ok = false;
                        break;
                    }
                }
            }
        }
    }

    return is_ok;
}

static bool ggml_cuda_can_fuse(const struct ggml_cgraph *                cgraph,
                               int                                       node_idx,
                               std::initializer_list<enum ggml_op>       ops,
                               std::initializer_list<enum ggml_unary_op> unary_ops) {
#ifndef NDEBUG
    const size_t num_unary = std::count(ops.begin(), ops.end(), GGML_OP_UNARY);
    GGML_ASSERT(unary_ops.size() == num_unary);
#endif

    const auto is_equal = [](const std::initializer_list<enum ggml_op> & list1,
                             const std::initializer_list<enum ggml_op> & list2) {
        return std::equal(list1.begin(), list1.end(), list2.begin(), list2.end());
    };

    std::initializer_list<enum ggml_op> mul_mat_bias_glu_ops    = { GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_GLU };
    std::initializer_list<enum ggml_op> mul_mat_id_bias_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_GLU };

    std::initializer_list<enum ggml_op> mul_mat_id_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU };
    std::initializer_list<enum ggml_op> mul_mat_glu_ops    = { GGML_OP_MUL_MAT,    GGML_OP_MUL_MAT,    GGML_OP_GLU };

    if ((is_equal(mul_mat_bias_glu_ops, ops) || is_equal(mul_mat_id_bias_glu_ops, ops)) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
        const ggml_tensor * ffn_gate      = cgraph->nodes[node_idx];
        const ggml_tensor * ffn_gate_bias = cgraph->nodes[node_idx + 1];
        const ggml_tensor * ffn_up        = cgraph->nodes[node_idx + 2];
        const ggml_tensor * ffn_up_bias   = cgraph->nodes[node_idx + 3];
        const ggml_tensor * glu           = cgraph->nodes[node_idx + 4];

        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu, ffn_up_bias, ffn_gate_bias)) {
            int out_nodes[] = { node_idx + 4 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if ((is_equal(mul_mat_id_glu_ops, ops) || is_equal(mul_mat_glu_ops, ops)) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * ffn_gate = cgraph->nodes[node_idx];
        const ggml_tensor * ffn_up   = cgraph->nodes[node_idx + 1];
        const ggml_tensor * glu      = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    std::initializer_list<enum ggml_op> rms_norm_mul_rope_ops          = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE };
    std::initializer_list<enum ggml_op> rms_norm_mul_rope_set_rows_ops = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    if (is_equal(rms_norm_mul_rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * rope     = cgraph->nodes[node_idx + 2];
        const ggml_tensor * view     = cgraph->nodes[node_idx + 3];
        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 4];

        if (ggml_check_edges(cgraph, node_idx, {{1, 0, 0}, {2, 0, 1}, {3, 0, 2}, {4, 0, 3}}) &&
            ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope) &&
            ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
            int out_nodes[] = { node_idx + 4 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if (is_equal(rms_norm_mul_rope_ops, ops) && ggml_can_fuse(cgraph, node_idx, ops)) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * rope     = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
        return false;
    }

    std::initializer_list<enum ggml_op> rope_set_rows_ops = { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    if (is_equal(rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * rope     = cgraph->nodes[node_idx];
        const ggml_tensor * view     = cgraph->nodes[node_idx + 1];
        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if (!ggml_can_fuse(cgraph, node_idx, ops)) {
        return false;
    }

    if ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        const ggml_tensor *rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor *mul      = cgraph->nodes[node_idx+1];
        const ggml_tensor *add      = nullptr;

        if (ops.size() == 3 && ops.begin()[2] == GGML_OP_ADD) {
            add = cgraph->nodes[node_idx+2];
        }

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        //rms norm only supports F32
        if (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            return false;
        }

        if (add && (add->src[0]->type != GGML_TYPE_F32 ||
            add->src[1]->type != GGML_TYPE_F32 ||
            add->type != GGML_TYPE_F32) ) {
            return false;
        }

        //if rms norm is the B operand, then we don't handle broadcast
        if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
            return false;
        }

        //rms_norm kernel assumes contiguous rows
        if (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
            return false;
        }

        if (add && (!ggml_is_contiguous(add->src[0]) || !ggml_is_contiguous_rows(add->src[1]))) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_SCALE) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * scale    = cgraph->nodes[node_idx+1];

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        float bias;
        memcpy(&bias, (const float *) scale->op_params + 1, sizeof(float));

        return bias == 0.0f && scale->type == GGML_TYPE_F32;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_UNARY
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * silu     = cgraph->nodes[node_idx+1];
        if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            return false;
        }

        if (ssm_conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            return false;
        }

        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_ADD
     && ops.begin()[2] == GGML_OP_UNARY && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * add      = cgraph->nodes[node_idx+1];
        const ggml_tensor * silu     = cgraph->nodes[node_idx+2];
        if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            return false;
        }

        if (ssm_conv->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            return false;
        }

        // ADD must consume ssm_conv's output and broadcast a 1-D channel-wise bias.
        const ggml_tensor * bias = (add->src[0] == ssm_conv) ? add->src[1] : add->src[0];
        if (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias)) {
            return false;
        }
        if (ggml_nelements(bias) != ssm_conv->ne[0] || bias->ne[0] != ssm_conv->ne[0]) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL
     && unary_ops.size() == 1 && (unary_ops.begin()[0] == GGML_UNARY_OP_SILU || unary_ops.begin()[0] == GGML_UNARY_OP_SIGMOID || unary_ops.begin()[0] == GGML_UNARY_OP_SOFTPLUS)) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * mul   = cgraph->nodes[node_idx+1];

        if (ggml_get_unary_op(unary) != unary_ops.begin()[0]) {
            return false;
        }

        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16 && unary->type != GGML_TYPE_BF16) {
            return false;
        }

        if (unary->type != mul->type) {
            return false;
        }

        const ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
        if (other->type != unary->type) {
            return false;
        }
        if (!ggml_is_contiguous_1(other) || !ggml_is_contiguous_1(unary->src[0]) || !ggml_are_same_shape(other, unary)) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_SQR
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_RELU) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * sqr   = cgraph->nodes[node_idx+1];

        if (ggml_get_unary_op(unary) != GGML_UNARY_OP_RELU) {
            return false;
        }

        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
            return false;
        }

        if (unary->type != sqr->type) {
            return false;
        }

        if (!ggml_is_contiguous(unary->src[0])) {
            return false;
        }

        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SCALE && ops.begin()[1] == GGML_OP_UNARY && ops.begin()[2] == GGML_OP_SCALE
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_TANH) {
        const ggml_tensor *scale  = cgraph->nodes[node_idx];
        const ggml_tensor *tanh   = cgraph->nodes[node_idx+1];
        const ggml_tensor *scale2 = cgraph->nodes[node_idx+2];

        // the fused softcap kernel is F32 only
        if (scale->src[0]->type != GGML_TYPE_F32 || scale->type != GGML_TYPE_F32) {
            return false;
        }

        if (ggml_get_unary_op(tanh) != GGML_UNARY_OP_TANH) {
            return false;
        }

        // Check for bias
        if (ggml_get_op_params_f32(scale, 1) != 0.0f || ggml_get_op_params_f32(scale2, 1) != 0.0f) {
            return false;
        }

        return true;
    }

    return false;
}

// try and fuse nodes and return the number of nodes to skip
static int ggml_cuda_match_hc_mix(const ggml_cgraph * cgraph, const int i, ggml_cuda_hc_mix_args & args, enum ggml_op * ops) {
    const ggml_tensor * node = cgraph->nodes[i];
    if (!(node->op == GGML_OP_UNARY && ggml_get_unary_op(node) == GGML_UNARY_OP_SIGMOID && node->type == GGML_TYPE_F32 &&
        i + 5 < cgraph->n_nodes &&
        cgraph->nodes[i + 1]->op == GGML_OP_MUL && cgraph->nodes[i + 2]->op == GGML_OP_RESHAPE &&
        cgraph->nodes[i + 3]->op == GGML_OP_VIEW && cgraph->nodes[i + 4]->op == GGML_OP_CONT)) {
        return 0;
    }
    const ggml_tensor * gate  = node->src[0];
    const ggml_tensor * mul   = cgraph->nodes[i + 1];
    const ggml_tensor * view0 = cgraph->nodes[i + 3];
    const ggml_tensor * cont  = cgraph->nodes[i + 4];
    const ggml_tensor * xn    = mul->src[0] == node ? mul->src[1] : mul->src[0];

    const int64_t n_embd   = cont->ne[0];
    const int64_t n_tokens = cont->ne[1];
    const int64_t hc_dim   = mul->ne[0];
    const int     hc       = n_embd > 0 ? (int) (hc_dim / n_embd) : 0;
    const size_t  row_size = hc_dim * sizeof(float);

    auto stream_view_ok = [&](const ggml_tensor * v, int c) {
        return v->op == GGML_OP_VIEW && v->view_src == mul && v->type == GGML_TYPE_F32 &&
            v->view_offs == (size_t) c * n_embd * sizeof(float) &&
            v->ne[0] == n_embd && v->ne[1] == n_tokens && v->ne[2] == 1 && v->ne[3] == 1 &&
            v->nb[0] == sizeof(float) && v->nb[1] == row_size;
    };

    bool ok = (mul->src[0] == node || mul->src[1] == node) && xn != node &&
        mul->type == GGML_TYPE_F32 && xn->type == GGML_TYPE_F32 && gate->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(xn) && ggml_is_contiguous(gate) && ggml_are_same_shape(xn, gate) &&
        ggml_are_same_shape(mul, xn) && ggml_are_same_shape(mul, node) &&
        mul->ne[2] == 1 && mul->ne[3] == 1 && ggml_nrows(mul) == n_tokens &&
        hc >= 2 && hc <= 16 && hc_dim == (int64_t) hc * n_embd &&
        cgraph->nodes[i + 2]->view_src == mul && cont->src[0] == view0 && stream_view_ok(view0, 0) &&
        cont->type == GGML_TYPE_F32 && ggml_is_contiguous(cont) && cont->ne[2] == 1 && cont->ne[3] == 1;

    const int n_ops = ok ? 5 + 2 * (hc - 1) + 1 : 0;
    ok = ok && i + n_ops <= cgraph->n_nodes;
    if (!ok) {
        return 0;
    }

    ops[0] = GGML_OP_UNARY; ops[1] = GGML_OP_MUL; ops[2] = GGML_OP_RESHAPE; ops[3] = GGML_OP_VIEW; ops[4] = GGML_OP_CONT;
    const ggml_tensor * prev = cont;
    for (int c = 1; ok && c < hc; ++c) {
        const ggml_tensor * v   = cgraph->nodes[i + 5 + 2 * (c - 1)];
        const ggml_tensor * add = cgraph->nodes[i + 6 + 2 * (c - 1)];
        ok = stream_view_ok(v, c) && add->op == GGML_OP_ADD && add->type == GGML_TYPE_F32 &&
            add->src[0] == prev && add->src[1] == v && ggml_are_same_shape(add, cont);
        ops[5 + 2 * (c - 1)] = GGML_OP_VIEW;
        ops[6 + 2 * (c - 1)] = GGML_OP_ADD;
        prev = add;
    }
    ggml_tensor * scale = cgraph->nodes[i + n_ops - 1];
    ops[n_ops - 1] = GGML_OP_SCALE;
    ok = ok && scale->op == GGML_OP_SCALE && scale->src[0] == prev && scale->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(scale) && ggml_are_same_shape(scale, cont);
    if (!ok) {
        return 0;
    }

    args.xn    = xn;
    args.gate  = gate;
    args.dst   = scale;
    args.hc    = hc;
    args.scale = ggml_get_op_params_f32(scale, 0);
    args.bias  = ggml_get_op_params_f32(scale, 1);
    return n_ops;
}

// the hc_mix window must be closed (its only external consumer is the fused output) before the
// gate GEMM can be folded into it: otherwise the gate tensor the GEMM no longer materializes could
// still be read by another node.
static int ggml_cuda_hc_mix_closed(const ggml_cgraph * cgraph, const int i, ggml_cuda_hc_mix_args & args) {
    enum ggml_op ops[5 + 2 * 15 + 1];
    const int n_ops = ggml_cuda_match_hc_mix(cgraph, i, args, ops);
    if (n_ops == 0) {
        return 0;
    }
    int node_idxs[5 + 2 * 15 + 1];
    for (int j = 0; j < n_ops; ++j) {
        node_idxs[j] = i + j;
    }
    const int out_nodes[] = { i + n_ops - 1 };
    return ggml_can_fuse_subgraph_ext(cgraph, node_idxs, n_ops, ops, out_nodes, 1) ? n_ops : 0;
}

// Structural identification of the qwen4exp hyper-connection combine+norm subgraph for the BF16 HC
// streams (LLAMA_HC_BLK16 / LLAMA_HC_RES16, both default OFF).  graph_optimize runs before the
// buffers are assigned, so this is structure/shape only (no alias checks); the fusion site
// re-checks everything.  Only the repeat-anchored form qwen4exp emits is recognised (the narrow
// block_out base is a graph output and the REPEAT is consumed inside the fused window); if it does
// not match, the caller simply does not mark and the two flags stay inert.
static bool ggml_cuda_hc_combine_norm_identify(const ggml_cgraph * cgraph, const int i, ggml_cuda_hc_combine_norm_args & args) {
    const ggml_tensor * rep = cgraph->nodes[i];
    if (rep->op != GGML_OP_REPEAT || rep->type != GGML_TYPE_F32 || rep->ne[3] != 1) {
        return false;
    }
    const int64_t n_embd = rep->ne[0], hc = rep->ne[1], n_tok = rep->ne[2];
    if (n_embd <= 0 || hc < 2 || hc > 8 || n_tok < 1) {
        return false;
    }
    const ggml_tensor * base_root = rep->src[0];
    while (base_root != nullptr && base_root->view_src != nullptr) {
        base_root = base_root->view_src;
    }
    if (base_root == nullptr || base_root->type != GGML_TYPE_F32 || !(base_root->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }
    const int limit = std::min(cgraph->n_nodes, i + 24);
    int k = i + 1;
    while (k < limit && !(cgraph->nodes[k]->op == GGML_OP_MUL &&
            (cgraph->nodes[k]->src[0] == rep || cgraph->nodes[k]->src[1] == rep))) {
        ++k;
    }
    if (k >= limit) {
        return false;
    }
    const ggml_tensor * mul = cgraph->nodes[k];
    int m = k + 1;
    while (m < limit && ggml_cuda_is_view_or_noop(cgraph->nodes[m]) && !ggml_is_empty(cgraph->nodes[m])) {
        ++m;
    }
    if (m >= limit || cgraph->nodes[m]->op != GGML_OP_ADD) {
        return false;
    }
    const ggml_tensor * add = cgraph->nodes[m];
    int q = m + 1;
    while (q < limit && ggml_cuda_is_view_or_noop(cgraph->nodes[q]) && !ggml_is_empty(cgraph->nodes[q])) {
        ++q;
    }
    if (q >= limit || cgraph->nodes[q]->op != GGML_OP_RMS_NORM || cgraph->nodes[q]->src[0] == nullptr) {
        return false;
    }
    const ggml_tensor * rms = cgraph->nodes[q];
    const ggml_tensor * rms_src = rms->src[0];
    while (rms_src->view_src != nullptr) {
        rms_src = rms_src->view_src;
    }
    if (rms_src != add) {
        return false;
    }
    int g = q + 1;
    while (g < limit && ggml_cuda_is_view_or_noop(cgraph->nodes[g]) && !ggml_is_empty(cgraph->nodes[g])) {
        ++g;
    }
    if (g >= limit || cgraph->nodes[g]->op != GGML_OP_MUL) {
        return false;
    }
    const ggml_tensor * mulg = cgraph->nodes[g];
    const ggml_tensor * res = add->src[0] == mul ? add->src[1] : add->src[0];
    if (res == nullptr || res == mul ||
            !ggml_are_same_shape(add, mul) || !ggml_are_same_shape(add, res) || !ggml_are_same_shape(add, rms)) {
        return false;
    }
    args.inject       = nullptr;
    args.residual     = res;
    args.block_out    = base_root;
    args.block_out_hc = false;
    args.gamma        = nullptr;
    args.out_res      = const_cast<ggml_tensor *>(add);
    args.out_xn       = const_cast<ggml_tensor *>(mulg);
    return true;
}

// Fill the BF16 HC-stream pointers for a matched combine+norm window.  Each pointer is left null when
// the graph did not mark the matching tensor, so an unmarked stream keeps its F32 read/write.
static void ggml_cuda_hc_combine_norm_set_bf16(ggml_backend_cuda_context & ctx, ggml_cuda_hc_combine_norm_args & args) {
    // HC16 completion: the graph optimizer already marks out_xn BF16-only (all its consumers read a
    // BF16 copy: the MMB GEMMs through the activation cache, dsv4_hc_pre through its x16 arm).  The
    // fused combine is the one producer that did not emit the copy, so every consumer reconverted the
    // F32 (the hc_norm half of the mmb_cvt_f32_bf16 traffic).  Writing it here is bit-identical to
    // the on-the-fly conversion (same RNE rounding) and lets the F32 store be dropped.
    if (ggml_cuda_mmb_active() && ggml_cuda_mmb_is_bf16_only(args.out_xn)) {
        args.out_xn_bf16  = ggml_cuda_mmb_reserve_auto(ctx, args.out_xn, (size_t) ggml_nelements(args.out_xn));
        args.store_xn_f32 = args.out_xn_bf16 == nullptr;
    }
    // BF16 HC streams (LLAMA_HC_BLK16 / LLAMA_HC_RES16, both default OFF).
    if (!ggml_cuda_mmb_blk16() && !ggml_cuda_mmb_res16()) {
        return;
    }
    const ggml_tensor * rblk = args.block_out->view_src ? args.block_out->view_src : args.block_out;
    if (ggml_cuda_mmb_blk16() && !args.block_out_hc && args.block_out->view_offs == 0 &&
            ggml_cuda_mmb_is_bf16_only(rblk)) {
        args.blk_in_bf16 = (const uint16_t *) args.block_out->data;
    }
    if (ggml_cuda_mmb_res16()) {
        const ggml_tensor * rin  = args.residual->view_src ? args.residual->view_src : args.residual;
        const ggml_tensor * rout = args.out_res->view_src  ? args.out_res->view_src  : args.out_res;
        const bool in16  = ggml_cuda_mmb_is_bf16_only(rin)  && args.residual->view_offs == 0;
        const bool out16 = ggml_cuda_mmb_is_bf16_only(rout) && args.out_res->view_offs == 0;
        if (args.out_res->data == args.residual->data && in16 != out16) {
            GGML_ABORT("hc_combine_norm: in-place residual with mismatched BF16 marks (%s)", args.out_res->name);
        }
        args.res_in_bf16  = in16  ? (const uint16_t *) args.residual->data : nullptr;
        args.res_out_bf16 = out16 ? (uint16_t *) args.out_res->data : nullptr;
    }
}

// Prefill indexer head reduction: relu + head-sum.  Anchored at the RELU.  Our qwen4exp graph (the
// L2a memory win) puts the relu BEFORE the 4-D reshape, so an optional RESHAPE between the relu and
// the head views is accepted; the reference's relu-on-4-D form matches with no reshape.  The fused
// kernel recomputes the relu from the pre-relu scores and sums in graph order, so it is bit-identical.
static int ggml_cuda_match_idx_relu_sum(const ggml_cgraph * g, int i, ggml_cuda_idx_relu_sum_args & a) {
    if (!ggml_cuda_idx_relu_sum_enabled() || i + 4 >= g->n_nodes) {
        return 0;
    }
    const ggml_tensor * relu = g->nodes[i];
    if (relu->op != GGML_OP_UNARY || ggml_get_unary_op(relu) != GGML_UNARY_OP_RELU) {
        return 0;
    }
    if (relu->type != GGML_TYPE_F32 || !ggml_is_contiguous(relu)) {
        return 0;
    }
    const ggml_tensor * src = relu->src[0];
    if (!src || src->type != GGML_TYPE_F32 || !ggml_is_contiguous(src)) {
        return 0;
    }
    // our L2a form: RELU([nb, H*nt, ns]) -> RESHAPE_4D([nb, H, nt, ns]) -> head views.  ggml
    // collapses a view-of-a-view, so the head views' view_src is the RELU root while their strides
    // come from the reshape (`sc`).
    const ggml_tensor * sc = relu;
    int j = i + 1;
    if (g->nodes[j]->op == GGML_OP_RESHAPE && g->nodes[j]->src[0] == relu &&
            g->nodes[j]->view_src == relu && g->nodes[j]->type == GGML_TYPE_F32 && ggml_is_contiguous(g->nodes[j])) {
        sc = g->nodes[j];
        ++j;
    }
    const int64_t nb = sc->ne[0], H = sc->ne[1], nt = sc->ne[2], ns = sc->ne[3];
    if (H < 2 || H > 32 || nb < 64 || nt * ns < 64 || j + 1 >= g->n_nodes) {
        return 0;
    }
    auto is_slice = [&](const ggml_tensor * v, int64_t h) {
        return v->op == GGML_OP_VIEW && v->view_src == relu && v->type == GGML_TYPE_F32 &&
               v->ne[0] == nb && v->ne[1] == nt && v->ne[2] == ns && v->ne[3] == 1 &&
               v->nb[0] == sizeof(float) && v->nb[1] == sc->nb[2] && v->nb[2] == sc->nb[3] &&
               v->view_offs == (size_t) h * sc->nb[1];
    };
    const ggml_tensor * cont = g->nodes[j + 1];
    if (!is_slice(g->nodes[j], 0) || cont->op != GGML_OP_CONT || cont->src[0] != g->nodes[j]) {
        return 0;
    }
    const ggml_tensor * prev = cont;
    int k = j + 2;
    for (int64_t h = 1; h < H; ++h) {
        if (k + 1 >= g->n_nodes) {
            return 0;
        }
        const ggml_tensor * v = g->nodes[k];
        const ggml_tensor * add = g->nodes[k + 1];
        if (!is_slice(v, h) || add->op != GGML_OP_ADD || add->type != GGML_TYPE_F32) {
            return 0;
        }
        if (add->src[0] != prev || add->src[1] != v || !ggml_are_same_shape(add, cont) || !ggml_is_contiguous(add)) {
            return 0;
        }
        prev = add;
        k += 2;
    }
    const int count = k - i;
    if (count > 32) {
        return 0;
    }
    int indices[32];
    enum ggml_op ops[32];
    for (int q = 0; q < count; ++q) {
        indices[q] = i + q;
        ops[q] = g->nodes[i + q]->op;
    }
    const int output = i + count - 1;
    if (!ggml_can_fuse_subgraph_ext(g, indices, count, ops, &output, 1)) {
        return 0;
    }
    if (src->data && prev->data) {
        const uintptr_t av = (uintptr_t) src->data, bv = (uintptr_t) prev->data;
        const bool overlap = av <= bv ? bv - av < ggml_nbytes(src) : av - bv < ggml_nbytes(prev);
        if (overlap) {
            return 0;
        }
    }
    a.score = src;
    a.dst   = (ggml_tensor *) prev;
    a.heads = (int) H;
    a.rows  = nt * ns;
    return count;
}

// wip/moe-expert-cache (H1): while the cache is active, only the fusions that read a routed
// expert table in the cache band stand down; every other backend fusion (router/topk, GDN, QSA,
// rope, norms, and the *prefill* MoE fusions) stays on.  The decode/verify MoE must run through
// `ggml_cuda_mul_mat_id`, which is where the slot-remap consumer lives; a fused MoE bypasses it
// and would read the redirected `input_cpy`, which the scheduler did not populate (the cache took
// the input over).  Prefill (n_tokens > band) is untouched: the cache does not take those inputs
// over, their `input_cpy` is fully copied, and their fusions keep firing bit-identically.
static bool ggml_cuda_cache_blocks_fusion(const ggml_cgraph * cgraph, int i) {
    if (!moe_cache_enabled()) {
        return false;
    }
    const ggml_tensor * node = cgraph->nodes[i];

    // gate+up+GLU, the routed pair, and the qwen4exp weighted-down all start at the routed
    // matmul; `ne[2]` is its token count, so gate on the cache band (prefill stays fused).
    if (node->op == GGML_OP_MUL_MAT_ID) {
        if (node->ne[2] > MOE_EXPERT_CACHE_MAX_TOK) {
            return false;   // prefill: the cache does not take these inputs over
        }
        // No routed expert table at all (the model is fully device-resident: `-ncmoe 0`, or a `-ncmoe`
        // that did not offload, e.g. on unified memory).  The cache can never take an input over, so its
        // fusions must behave exactly as if it were disabled -- otherwise enabling the cache changes the
        // output vs the cache-less oracle, because this stand-down changes the arithmetic.
        if (!moe_cache_has_tables()) {
            return false;
        }
        // Tables exist but cannot serve (priming pass, every arena allocation failed, or a budget below
        // one expert): the cache takes no input over, so its cache-aware fusions MUST stay stood down.
        // The gate+up mmvq fusion reads the scheduler's copy through the fused call site, which is only
        // staged when the scheduler performs the copy itself; forcing the per-op path here is what keeps
        // the output byte-identical to the cache-less run.
        if (!moe_cache_has_arena()) {
            return true;
        }
        // The decode gate+up+GLU triple is cache-aware: its mmvq call site redirects both expert
        // tables and the routing onto the compact arenas/remap (`moe_cache_redirect_fused`), so it
        // may fire.  Every other routed-expert fusion still reads the scheduler's `input_cpy`, which
        // the cache does NOT populate when it takes the input over - so they stay stood down until
        // each is taught the arena.
        if (i + 2 < cgraph->n_nodes &&
                cgraph->nodes[i + 1]->op == GGML_OP_MUL_MAT_ID &&
                cgraph->nodes[i + 2]->op == GGML_OP_GLU) {
            return false;   // allowed: the plain gate+up+GLU decode fusion
        }
        // down projection + per-(expert,token) routing-weight fold ([MUL_MAT_ID, MUL]): the
        // cache-aware variant in the epilogue below redirects the down table and the routing onto
        // the arena/remap, so it may fire.  The qwen4exp weighted-down chain (a 21-node tail,
        // opt-in via GGML_CUDA_ENABLE_RDNA3_5_SINGLE_TOKEN_FUSIONS and RDNA3_5-only) is NOT
        // redirect-safe; it is kept stood down by its own `!moe_cache_enabled()` gate at its site.
        if (i + 1 < cgraph->n_nodes && cgraph->nodes[i + 1]->op == GGML_OP_MUL &&
                cgraph->nodes[i + 1]->src[0] == node) {
            return false;   // allowed: the cache-aware down projection fold
        }
        return true;
    }

    // swiglu -> routed-down fold (leading GLU, consuming the next MUL_MAT_ID).  Prefill-only in
    // practice, but guard it too so a cache-band variant can never read the redirected table.
    if (node->op == GGML_OP_GLU && i + 1 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_MUL_MAT_ID &&
            cgraph->nodes[i + 1]->ne[2] <= MOE_EXPERT_CACHE_MAX_TOK) {
        return true;
    }

    return false;
}

static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {

    static bool disable_fusion = getenv("GGML_CUDA_DISABLE_FUSION") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_FUSION"));
    if (disable_fusion) {
        return 0;
    }
    if (ggml_cuda_cache_blocks_fusion(cgraph, i)) {
        return 0;
    }

    // fused gate+up+GLU MMQ (prefill): hard opt-out for A/B and regression testing
    static bool disable_moe_mmq = getenv("GGML_CUDA_DISABLE_MOE_MMQ_FUSION") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_MOE_MMQ_FUSION"));

    // prefill hyper-connection (qwen4exp) elementwise-chain fusions (ported from
    // halo-box/strix-llama.cpp): opt-out for A/B and regression testing
    static bool disable_hc_fusion = getenv("GGML_CUDA_DISABLE_HC_FUSION") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_HC_FUSION"));

    const int cc = ggml_cuda_info().devices[cuda_ctx->device].cc;

    ggml_tensor * node = cgraph->nodes[i];

    // HC_COMBINE -> BF16 HC_MIX (GGML_CUDA_FUSE_HC_COMBINE_MIX=0: off): the combine and the mixer's norm in one kernel
    static const bool fuse_hc_combine_mix = getenv("GGML_CUDA_FUSE_HC_COMBINE_MIX") == nullptr || atoi(getenv("GGML_CUDA_FUSE_HC_COMBINE_MIX")) != 0;
    if (fuse_hc_combine_mix && node->op == GGML_OP_HC_COMBINE && i + 1 < cgraph->n_nodes &&
            ggml_cuda_hc_combine_mix_fusable(node, cgraph->nodes[i + 1])) {
        ggml_cuda_op_hc_combine_mix(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    // batched copy (GGML_CUDA_FUSE_CPY_BATCH=0: off): consecutive same-layout f32 copies as one launch
    if (node->op == GGML_OP_CPY && cuda_ctx->stream_context().concurrent_events.empty()) {
        const int taken = ggml_cuda_cpy_batch(*cuda_ctx, cgraph, i);
        if (taken >= 2) {
            return taken - 1;
        }
    }

    // GLU -> Q8_1 (GGML_CUDA_FUSE_GLU_Q8_1=0: off): an F32 GLU whose next node (or the one after a reshape) is a verify-band
    // mmvq matmul over it is marked here (RDNA4 only, through ggml_cuda_should_fuse_mul_mat_vec_q(mm, true)); the regular
    // GLU launcher then also writes the matmul's Q8_1 blocks into the quantize cache, which the matmul finds.
    static const bool fuse_glu_q8_1 = getenv("GGML_CUDA_FUSE_GLU_Q8_1") == nullptr || atoi(getenv("GGML_CUDA_FUSE_GLU_Q8_1")) != 0;
    cuda_ctx->glu_q8_1_node = nullptr;
    cuda_ctx->glu_q8_1_mm   = nullptr;
    if (node->op == GGML_OP_GLU && fuse_glu_q8_1 && node->type == GGML_TYPE_F32 &&
            node->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(node) && node->ne[0] % QK8_1 == 0) {
        const ggml_tensor * mm = nullptr;
        if (i + 1 < cgraph->n_nodes && cgraph->nodes[i + 1]->op == GGML_OP_MUL_MAT) {
            mm = cgraph->nodes[i + 1];
        } else if (i + 2 < cgraph->n_nodes && cgraph->nodes[i + 1]->op == GGML_OP_RESHAPE &&
                   cgraph->nodes[i + 2]->op == GGML_OP_MUL_MAT) {
            mm = cgraph->nodes[i + 2];
        }
        if (mm != nullptr) {
            const ggml_tensor * a = mm->src[1];
            const ggml_tensor * a_root = a;
            while (a_root->view_src != nullptr) {
                a_root = a_root->view_src;
            }
            if (a_root == node && ggml_is_contiguous(a) && a->ne[0] % QK8_1 == 0 && ggml_nelements(a) == ggml_nelements(node) &&
                    a->ne[1] >= 2 && ggml_is_quantized(mm->src[0]->type) && ggml_cuda_should_fuse_mul_mat_vec_q(mm, true)) {
                cuda_ctx->glu_q8_1_node = node;
                cuda_ctx->glu_q8_1_mm   = mm;
            }
        }
    }

    if (node->op == GGML_OP_MUL_MAT_ID && cuda_ctx->stream_context().concurrent_events.empty() &&
            ggml_cuda_match_shared_expert(cgraph, i, i + 3)) {
        const int outputs[] = { i + 2, i + 5 };
        if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, 6, outputs, 2)) {
            ggml_tensor * routed = cgraph->nodes[i + 2];
            ggml_tensor * shared = cgraph->nodes[i + 5];
            const ggml_tensor * up = routed->src[1];
            ggml_cuda_mm_fusion_args_host fusion{};
            fusion.gate = routed->src[0]->src[0];
            fusion.glu_op = ggml_get_glu_op(routed);
            fusion.glu_limit = ggml_get_op_params_f32(routed, 3);
            fusion.shared_up = shared->src[1]->src[0];
            fusion.shared_gate = shared->src[0]->src[0];
            fusion.shared_dst = shared;
            ggml_cuda_mul_mat_vec_q(*cuda_ctx, up->src[0], up->src[1], up->src[2], routed, &fusion);
            return 5;
        }
    }

    // Prefill indexer head reduction (relu + head-sum) for the qwen4exp sparse-attention graph.
    if (node->op == GGML_OP_UNARY && GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        ggml_cuda_idx_relu_sum_args args;
        const int count = ggml_cuda_match_idx_relu_sum(cgraph, i, args);
        if (count > 0) {
            ggml_cuda_op_idx_relu_sum(*cuda_ctx, args);
            return count - 1;
        }
    }

    // Depthwise causal conv1d fusions (qwen4exp GDN + PLE), ported from the halo-box
    // reference: the CONCAT(state, x) materialization and the SSM_CONV / tap-mul-add
    // chain are replaced by a direct kernel that reads state+x and writes the conv
    // output (+ silu).  The CONCAT is still partially materialized for the recurrent
    // snapshot copies (its tail only).  Prefill only (T >= 256, C % 256 == 0).
    // A/B / bisect kill switch: GGML_CUDA_DISABLE_CONV_FUSION=1.
    static const bool disable_conv_fusion = getenv("GGML_CUDA_DISABLE_CONV_FUSION") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_CONV_FUSION"));
    if (!disable_conv_fusion && (node->op == GGML_OP_CONCAT || node->op == GGML_OP_CONT)) {
        ggml_cuda_ple_conv_match pm;
        if (node->op == GGML_OP_CONCAT && ggml_cuda_ple_conv_match_at_concat(cgraph, i, pm)) {
            ggml_cuda_ple_conv_write_tail(*cuda_ctx, pm);
            return 1;
        }
        if (node->op == GGML_OP_CONT && ggml_cuda_ple_conv_match_at_tap(cgraph, i, pm)) {
            ggml_cuda_ple_conv_direct(*cuda_ctx, pm);
            return pm.silu_idx - i;
        }
    }
    if (!disable_conv_fusion && (node->op == GGML_OP_CONCAT || node->op == GGML_OP_SSM_CONV)) {
        ggml_cuda_gdn_conv_match gm;
        if (node->op == GGML_OP_CONCAT && ggml_cuda_gdn_conv_match_at_concat(cgraph, i, gm)) {
            ggml_cuda_gdn_conv_write_tail(*cuda_ctx, gm);
            return 1;
        }
        if (node->op == GGML_OP_SSM_CONV && ggml_cuda_gdn_conv_match_at_conv(cgraph, i, gm)) {
            ggml_cuda_gdn_conv_direct(*cuda_ctx, gm);
            return 1;
        }
    }

    // Narrow-row RMS norm (ncols <= 256, >= 4096 rows) and its sigmoid-gated form, ported from the
    // halo-box reference (Phase-1 item 4): 8 rows per 256-thread block instead of one block per row,
    // which is block-scheduling bound for the model's per-head norms.  Bit-identical to
    // rms_norm_f32<256,{true,false}> (same per-warp xor trees + 8-partial xor tree).
    // A/B / bisect kill switch: GGML_CUDA_DISABLE_NORM_ROWS=1.
    static const bool disable_norm_rows = getenv("GGML_CUDA_DISABLE_NORM_ROWS") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_NORM_ROWS"));
    if (!disable_norm_rows && node->op == GGML_OP_RMS_NORM) {
        ggml_cuda_norm_gated_match nm;
        int sk = ggml_cuda_norm_gated_match_at(cgraph, i, nm);
        if (sk > 0) {
            if (nm.pre >= 0) {
                if (!ggml_cuda_compute_forward(*cuda_ctx, cgraph->nodes[nm.pre])) {
                    GGML_ABORT("norm-gated: gate MUL_MAT dispatch failed");
                }
            }
            ggml_cuda_op_norm_gated(*cuda_ctx, nm);
            return sk;
        }
        sk = ggml_cuda_norm_rows_match_at(cgraph, i, nm);
        if (sk > 0) {
            ggml_cuda_op_norm_gated(*cuda_ctx, nm);
            return sk;
        }
    }

    // qwen4exp IQ4_NL/Q8_0 routed down projection followed by the 10-expert weighted sum (ported
    // from halo-box/strix-llama.cpp): compute all selected experts for one output row in a wave and
    // apply their routing weights immediately, avoiding the [n_embd, n_used] intermediate and its
    // separate reduction launch. Hard opt-out (A/B, regression testing). Single-token only
    // (n_tokens == 1, gate in ggml_cuda_mul_mat_id_weighted_rdna3_5_ok).
    static const bool disable_weighted_down = getenv("GGML_CUDA_DISABLE_WEIGHTED_DOWN") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_WEIGHTED_DOWN"));
    // wip/moe-expert-cache: this 21-node weighted-down chain reads the expert table directly and
    // is not redirect-safe, so it stands down whenever the cache is active (the cache consumer
    // then serves the down op).  It is opt-in (RDNA3_5 single-token fusions) anyway.
    if (!disable_weighted_down && !moe_cache_enabled() && node->op == GGML_OP_MUL_MAT_ID && i + 20 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_MUL) {
        constexpr int n_used = 10;
        constexpr int n_ops  = 2 + n_used + (n_used - 1);
        ggml_tensor * mul = cgraph->nodes[i + 1];
        const ggml_tensor * experts = ggml_are_same_shape(mul, mul->src[0]) ? mul->src[0] : mul->src[1];
        const ggml_tensor * weights = experts == mul->src[0] ? mul->src[1] : mul->src[0];
        const int output_idx = i + n_ops - 1;
        ggml_tensor * output = cgraph->nodes[output_idx];
        bool valid = experts == node && node->ne[1] == n_used &&
            ggml_cuda_mul_mat_id_weighted_rdna3_5_ok(experts, weights, output);

        for (int j = 0; valid && j < n_used; ++j) {
            const ggml_tensor * view = cgraph->nodes[i + 2 + j];
            valid = view->op == GGML_OP_VIEW && view->src[0] == mul &&
                view->ne[0] == mul->ne[0] && view->ne[1] == mul->ne[2] && view->ne[2] == 1 && view->ne[3] == 1 &&
                view->nb[0] == sizeof(float) && view->nb[1] == mul->nb[2] && view->view_offs == size_t(j) * mul->nb[1];
        }
        const int add_start = i + 2 + n_used;
        if (valid) {
            const ggml_tensor * first = cgraph->nodes[add_start];
            valid = first->op == GGML_OP_ADD && first->src[0] == cgraph->nodes[i + 2] && first->src[1] == cgraph->nodes[i + 3];
        }
        for (int j = 2; valid && j < n_used; ++j) {
            const ggml_tensor * add = cgraph->nodes[add_start + j - 1];
            valid = add->op == GGML_OP_ADD && add->src[0] == cgraph->nodes[add_start + j - 2] && add->src[1] == cgraph->nodes[i + 2 + j];
        }
        if (valid) {
            std::vector<ggml_op> ops = { GGML_OP_MUL_MAT_ID, GGML_OP_MUL };
            ops.insert(ops.end(), n_used, GGML_OP_VIEW);
            ops.insert(ops.end(), n_used - 1, GGML_OP_ADD);
            const int out_nodes[] = { output_idx };
            if (ggml_can_fuse_subgraph(cgraph, i, n_ops, ops.data(), out_nodes, 1) &&
                    ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                ggml_cuda_mul_mat_id_weighted_rdna3_5(*cuda_ctx, experts, weights, output);
                return n_ops - 1;
            }
        }
    }

    // two consecutive MUL_MAT(_ID) nodes sharing src1/ids (qwen4exp sparse-MoE gate+up pair,
    // shared-expert up+gate pair): merge into one shared activation quantize + one mm_ids_helper
    // (halo-box port ggml_cuda_mul_mat_q_pair). Only the mmq q8_1-feed path (prefill): both types
    // mmq-eligible for these cols/experts, identical weight/dst shapes, equal ds layout, same
    // src1 (and ids). The per-node dedup scatter quantize used to run 2x/layer on identical
    // rows (grid 262144x3); the pair runs it once and both muls read the same q8_1 buffer.
    static const bool pair_off        = getenv("GGML_PAIR_OFF") != nullptr;
    static const bool disable_pair_dense = getenv("GGML_PAIR_DENSE_OFF") != nullptr;
    // Stand the pair fusion down only when MMB will actually take BOTH muls (checklist #5): the
    // global gate used to disable our Q4_K MoE pair on a model whose experts MMB cannot accelerate.
    const bool mmb_pair_taken = ggml_cuda_mmb_active() && i + 1 < cgraph->n_nodes && node->src[0] && cgraph->nodes[i + 1]->src[0] &&
        (node->op == GGML_OP_MUL_MAT_ID
            ? (ggml_cuda_mmb_routed_will_take(node->src[0]) && ggml_cuda_mmb_routed_will_take(cgraph->nodes[i + 1]->src[0]))
            : (ggml_cuda_mmb_dense_will_take(node->src[0])  && ggml_cuda_mmb_dense_will_take(cgraph->nodes[i + 1]->src[0])));
    if (!pair_off && !mmb_pair_taken && (node->op == GGML_OP_MUL_MAT_ID || (node->op == GGML_OP_MUL_MAT && !disable_pair_dense)) && i + 1 < cgraph->n_nodes) {
        ggml_tensor * next = cgraph->nodes[i + 1];
        const ggml_tensor * src0 = node->src[0];
        const ggml_tensor * src0_next = next->src[0];
        const bool has_ids = node->op == GGML_OP_MUL_MAT_ID;
        const bool valid_sources = next->op == node->op && src0 && src0_next && node->src[1] && next->src[1] &&
            (!has_ids || (node->src[2] && next->src[2]));
        const bool shared_inputs = valid_sources && node->src[1] == next->src[1] &&
            // The MUL_MAT_ID arm of ggml_cuda_mul_mat_q_pair assumes the standard sparse-MoE
            // activation layout (src1 = [n_embd, 1, n_tokens]) with more than one routed
            // expert, and asserts ne11 == 1 && n_expert_used > 1. Require those preconditions
            // here too, so MUL_MAT_ID pairs in other layouts (e.g. a per-expert gathered
            // activation, src1->ne[1] > 1, or top-1 routing, ids->ne[0] == 1) fall back to
            // the per-node path instead of aborting.
            (!has_ids || (node->src[2] == next->src[2] &&
                          node->src[1]->ne[1] == 1 && node->src[2]->ne[0] > 1));
        const int64_t mmq_cols = shared_inputs ? (has_ids ? node->src[1]->ne[2] : node->src[1]->ne[1]) : 0;
        const int64_t n_experts = shared_inputs && has_ids ? src0->ne[2] : 0;
        const bool use_mmq = shared_inputs && node->src[1]->type == GGML_TYPE_F32 &&
            node->type == GGML_TYPE_F32 && next->type == GGML_TYPE_F32 &&
            src0->type != GGML_TYPE_NVFP4 && src0->type != GGML_TYPE_MXFP4 &&
            src0_next->type != GGML_TYPE_NVFP4 && src0_next->type != GGML_TYPE_MXFP4 &&
            ggml_are_same_shape(src0, src0_next) && ggml_are_same_shape(node, next) &&
            ggml_cuda_should_use_mmq(src0->type, cc, mmq_cols, n_experts) &&
            ggml_cuda_should_use_mmq(src0_next->type, cc, mmq_cols, n_experts) &&
            mmq_get_q8_1_ds_layout(src0->type) == mmq_get_q8_1_ds_layout(src0_next->type);
        // single-token/gathered (mmvq territory, e.g. the blk.47 gather in prefill) must keep the
        // per-node decode path: the merged mmq kernels would change numerics there (B's !use_mmvq).
        const int64_t ncols_dst = has_ids ? node->ne[2] : node->ne[1];
        // The routed-expert (MUL_MAT_ID) path takes the dedicated MoE MMVQ kernel over the whole
        // verify band; RDNA4/RDNA3_5 dense rows that MMQ would run in its non-128-row fallback
        // config do the same.  The pair fusion stands down so the per-node MMVQ dispatch handles them.
        static const bool dense_band_off = getenv("GGML_CUDA_DISABLE_MMVQ_DENSE_BAND") != nullptr;
        const bool mmvq_extended_rows = !dense_band_off && !has_ids && (GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc)) && src0->ne[1] % 128 != 0;
        const int64_t mmvq_band = (has_ids || mmvq_extended_rows) ? MMVQ_MOE_MAX_BATCH_SIZE : MMVQ_MAX_BATCH_SIZE;
        const bool use_mmvq = ncols_dst <= mmvq_band &&
            (!has_ids || ncols_dst <= get_mmvq_mmid_max_batch(src0->type, cc) ||
                         ncols_dst <= get_mmvq_mmid_max_batch(src0_next->type, cc));
        if (use_mmq && !use_mmvq) {
            ggml_cuda_mul_mat_q_pair(*cuda_ctx, node, next);
            return 1;
        }
    }

    static const bool disable_mwr = getenv("GGML_CUDA_DISABLE_MWR") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_MWR"));
    if (!disable_mwr && node->op == GGML_OP_MUL) {
        ggml_moe_weighted_reduction_match match;
        static const bool mwr_dbg2 = getenv("GGML_CUDA_MWR_DEBUG") != nullptr;
        if (ggml_match_moe_weighted_reduction(cgraph, i, match)) {
            int count = match.node_count;
            const ggml_tensor * merge = nullptr;
            // shared-expert merge (LLAMA_HC_BLK16, default OFF): the ADD(reduction, ffn_shexp_gated)
            // right after is folded into the reduction, which then writes the BF16 block_out stream.
            if (ggml_cuda_mmb_blk16()) {
                const int nadd = i + match.node_count;
                if (nadd < cgraph->n_nodes && ggml_node_has_n_uses(cgraph, i + match.node_count - 1, 1)) {
                    const ggml_tensor * add = cgraph->nodes[nadd];
                    if (add->op == GGML_OP_ADD && add->type == GGML_TYPE_F32 && ggml_is_contiguous(add) &&
                            ggml_are_same_shape(add, match.dst) && ggml_cuda_mmb_is_bf16_only(add)) {
                        const ggml_tensor * o = add->src[0] == match.dst ? add->src[1] :
                                                (add->src[1] == match.dst ? add->src[0] : nullptr);
                        if (o && o->type == GGML_TYPE_F32 && ggml_is_contiguous(o) && ggml_are_same_shape(o, match.dst)) {
                            merge = o;
                            match.dst = const_cast<ggml_tensor *>(add);
                            count = match.node_count + 1;
                        }
                    }
                }
            }
            const int output_idx = i + count - 1;
            if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, count, &output_idx, 1)) {
                if (mwr_dbg2) GGML_LOG_INFO("MWR FUSED %s\n", cgraph->nodes[i]->name);
                ggml_cuda_op_moe_weighted_reduction(
                    *cuda_ctx, match.experts, match.expert_scale, match.weights, match.dst, merge);
                return count - 1;
            }
        }
    }

    // Verify band: the residual ADD in front of such a norm is a standalone k_bin_bcast (one token
    // folds it into the mmvq epilogue instead); fold it into the norm kernel as well.
    // GGML_CUDA_FUSE_ADD_RMS_Q8=0 turns it off.
    static const bool fuse_add_rms_q8 = getenv("GGML_CUDA_FUSE_ADD_RMS_Q8") == nullptr || atoi(getenv("GGML_CUDA_FUSE_ADD_RMS_Q8")) != 0;
    if (fuse_add_rms_q8 && node->op == GGML_OP_ADD && i + 2 < cgraph->n_nodes &&
            node->ne[1] > 1 && node->ne[1] <= MMVQ_MAX_BATCH_SIZE &&
            cgraph->nodes[i + 1]->op == GGML_OP_RMS_NORM && cgraph->nodes[i + 1]->src[0] == node &&
            cgraph->nodes[i + 2]->op == GGML_OP_MUL && cgraph->nodes[i + 2]->src[0] == cgraph->nodes[i + 1]) {
        ggml_tensor * norm = cgraph->nodes[i + 1];
        const ggml_tensor * mul = cgraph->nodes[i + 2];
        const int scan_end = std::min(cgraph->n_nodes, i + 33);
        for (int j = i + 3; j < scan_end; ++j) {
            const ggml_tensor * n = cgraph->nodes[j];
            const ggml_tensor * n_src1 = n->src[1];
            while (n_src1 != nullptr && n_src1->view_src != nullptr) {
                n_src1 = n_src1->view_src;
            }
            if (!(n->src[0] == mul || n_src1 == mul)) {
                continue;
            }
            if (n->op == GGML_OP_MUL_MAT && ggml_cuda_should_fuse_mul_mat_vec_q(n, true) &&
                    ggml_cuda_op_add_rms_norm_q8_1(*cuda_ctx, node, norm, mul)) {
                return 2;
            }
            break;
        }
    }

    // rms_norm + norm-weight MUL whose output feeds an mmvq matmul: fold the
    // Q8_1 quantize into the norm kernel and pre-fill the matmul quantize
    // cache. Consumes only the norm+MUL pair; the matmul dispatch that follows
    // finds the cached blocks and skips its own quantize. The matmul need not
    // be adjacent (unrelated nodes may sit between in DFS order).
    static const bool disable_norm_q8_1 = getenv("GGML_CUDA_DISABLE_NORM_Q8_1") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_NORM_Q8_1"));
    if (!disable_norm_q8_1 && node->op == GGML_OP_RMS_NORM && i + 1 < cgraph->n_nodes) {
        const ggml_tensor * mul = cgraph->nodes[i + 1];
        if (mul->op == GGML_OP_MUL && mul->src[0] == node) {
            // The consumers of the norm output are mostly mmvq matmuls; skip
            // over unrelated nodes and non-mmvq consumers (they read the F32
            // output, which the fused kernel still writes) until an mmvq
            // matmul over the norm output is found.
            const int scan_end = std::min(cgraph->n_nodes, i + 32);
            for (int j = i + 2; j < scan_end; ++j) {
                const ggml_tensor * n = cgraph->nodes[j];
                // A matmul over the norm output may use a view of it; the mmvq
                // quantize cache keys on the view root, so both match the mul.
                const ggml_tensor * n_src1 = n->src[1];
                while (n_src1 != nullptr && n_src1->view_src != nullptr) {
                    n_src1 = n_src1->view_src;
                }
                const bool consumes_mul = n->src[0] == mul || n_src1 == mul;
                if (!consumes_mul) {
                    continue;
                }
                // Multi-token MUL_MAT_ID (MoE expert decode for a batch of tokens,
                // e.g. the speculative verify step / server batch): the moe-kernel
                // path does not consume the cached Q8_1 y correctly, breaking the
                // verify==decode numerics invariant (MTP acceptance collapses to 0).
                // Keep the fold for single-token MMID and plain MUL_MAT consumers.
                const bool mmid_single = n->op != GGML_OP_MUL_MAT_ID || n->ne[2] == 1;
                if ((n->op == GGML_OP_MUL_MAT || n->op == GGML_OP_MUL_MAT_ID) &&
                        node->ne[0] % QK8_1 == 0 &&
                        ggml_cuda_should_fuse_mul_mat_vec_q(n, true) &&
                        mmid_single) {
                    ggml_cuda_op_rms_norm_q8_1(*cuda_ctx, node, mul);
                    return 1;
                }
            }
        }
    }

    // gated_delta_net -> cpy: scatter recurrent-state snapshots into the cache
    static const bool disable_gdn_cpy = getenv("GGML_CUDA_DISABLE_GDN_CPY") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_GDN_CPY"));

    // UNARY(sigmoid) beta -> GATED_DELTA_NET (GGML_CUDA_FUSE_GDN_BETA_SIGMOID=0: off): the sequential GDN kernel applies the
    // sigmoid to the beta it loads.  Decode/verify band only (single sequence, <= 16 tokens: never the chunked kernels).
    static const bool fuse_gdn_beta_sig = getenv("GGML_CUDA_FUSE_GDN_BETA_SIGMOID") == nullptr || atoi(getenv("GGML_CUDA_FUSE_GDN_BETA_SIGMOID")) != 0;
    if (fuse_gdn_beta_sig && node->op == GGML_OP_UNARY && ggml_get_unary_op(node) == GGML_UNARY_OP_SIGMOID &&
            node->type == GGML_TYPE_F32 && node->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(node) &&
            ggml_is_contiguous(node->src[0]) && ggml_are_same_shape(node, node->src[0])) {
        // view/no-op nodes that do not read the sigmoid may sit between it and the GDN (the state reshape does)
        int j = i + 1;
        while (j < cgraph->n_nodes && j <= i + 4 && ggml_cuda_is_view_or_noop(cgraph->nodes[j]) &&
               cgraph->nodes[j]->src[0] != node && cgraph->nodes[j]->view_src != node) {
            ++j;
        }
        if (j < cgraph->n_nodes && cgraph->nodes[j]->op == GGML_OP_GATED_DELTA_NET && cgraph->nodes[j]->src[4] == node &&
                cgraph->nodes[j]->src[2]->ne[2] <= 16 && cgraph->nodes[j]->src[2]->ne[3] == 1 &&
                ggml_node_get_use_count(cgraph, i) == 1 && !(node->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            ggml_tensor * gdn = cgraph->nodes[j];
            ggml_cuda_gated_delta_net_fused_cache fused_state_cpy;
            const int nodes_to_skip = disable_gdn_cpy ? 0 : ggml_cuda_try_gdn_cache_fusion(cgraph, j, fused_state_cpy);
            ggml_cuda_op_gated_delta_net_beta_sigmoid(*cuda_ctx, gdn, node->src[0], nodes_to_skip > 0 ? &fused_state_cpy : nullptr);
            return (j - i) + nodes_to_skip;
        }
    }
    if (!disable_gdn_cpy && node->op == GGML_OP_GATED_DELTA_NET) {
        ggml_cuda_gated_delta_net_fused_cache fused_state_cpy;
        const int nodes_to_skip = ggml_cuda_try_gdn_cache_fusion(cgraph, i, fused_state_cpy);
        if (nodes_to_skip > 0) {
#ifdef GGML_CUDA_DEBUG
            GGML_LOG_INFO("%s: fused gated_delta_net snapshot copies for %s (skipped %d nodes)\n",
                          __func__, node->name, nodes_to_skip);
#endif
            ggml_cuda_op_gated_delta_net_fused_cache(*cuda_ctx, node, fused_state_cpy);
            return nodes_to_skip;
        }
    }

    // swiglu -> mul_mat_q: fold silu(gate)*up into the mmq activation quantize (the qwen4exp
    // MoE down feed; the GLU output is never materialized). mmq only engages at prefill, so
    // the decode MoE tails keep their own fusions.
    // The swiglu->mmq fold is the routed (MUL_MAT_ID) MoE down feed; MMB's fused GLU covers it only
    // for IQ4_NL experts, so stand it down only then (checklist #5).
    const bool mmb_glu_taken = ggml_cuda_mmb_active() && i + 1 < cgraph->n_nodes &&
        cgraph->nodes[i + 1]->op == GGML_OP_MUL_MAT_ID && cgraph->nodes[i + 1]->src[0] &&
        ggml_cuda_mmb_routed_will_take(cgraph->nodes[i + 1]->src[0]);
    if (node->op == GGML_OP_GLU && i + 1 < cgraph->n_nodes && !mmb_glu_taken && ggml_get_glu_op(node) == GGML_GLU_OP_SWIGLU && node->src[1]) {
        ggml_tensor * next = cgraph->nodes[i + 1];
        const bool has_ids = next->op == GGML_OP_MUL_MAT_ID;
        if (has_ids || next->op == GGML_OP_MUL_MAT) {
            const ggml_tensor * weights = next->src[0];
            const ggml_tensor * ids = has_ids ? next->src[2] : nullptr;
            const ggml_tensor * gate = node->src[0];
            const ggml_tensor * up = node->src[1];
            const int cc = ggml_cuda_info().devices[cuda_ctx->device].cc;
            const int64_t mmq_cols = has_ids ? node->ne[2] : node->ne[1];
            const int64_t n_experts = has_ids ? weights->ne[2] : 0;
            const bool bad_padding_clear = ggml_backend_buffer_get_usage(weights->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
                ggml_nbytes(weights) != ggml_backend_buffer_get_alloc_size(weights->buffer, weights) && weights->view_src;

            const bool target_qwen36 = node->ne[0] == 512 && next->ne[0] == 2048 &&
                (!has_ids || (gate->ne[1] == 8 && weights->ne[2] == 256));
            const bool target_qwen4exp = node->ne[0] == 640 && next->ne[0] == 2560 &&
                (!has_ids || (gate->ne[1] == 10 && weights->ne[2] == 512));
            const bool target_shape = target_qwen36 || target_qwen4exp;
            const bool weight_type_ok = weights->type == GGML_TYPE_Q8_0 ||
                (target_qwen4exp && weights->type == GGML_TYPE_IQ4_NL);
            const bool shape_ok = target_shape && next->src[1] == node && gate->type == GGML_TYPE_F32 && up->type == GGML_TYPE_F32 &&
                node->type == GGML_TYPE_F32 && next->type == GGML_TYPE_F32 && weight_type_ok &&
                ggml_are_same_shape(gate, up) && ggml_are_same_shape(gate, node) && gate->nb[0] == sizeof(float) && up->nb[0] == sizeof(float) &&
                gate->ne[3] == 1 && (has_ids ? gate->ne[1] == ids->ne[0] && gate->ne[2] == ids->ne[1] : gate->ne[2] == 1);
            const bool use_mmq = shape_ok && !bad_padding_clear && GGML_CUDA_CC_IS_RDNA3_5(cc) &&
                ggml_cuda_should_use_mmq(weights->type, cc, mmq_cols, n_experts);

            if (use_mmq && ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_GLU, next->op }, { i + 1 })) {
                ggml_cuda_mul_mat_q_swiglu(*cuda_ctx, weights, ids, next, node);
                return 1;
            }
        }
    }

    // Dense SWIGLU -> MUL_MAT that runs as mmq (prefill FFN down projection, any weight type): the
    // mmq activation quantize computes silu(gate) * up itself, so the GLU output is never written and
    // read back.  Same float values as unary_gated_op_kernel<op_silu> + quantize_mmq_q8_1: bit-identical.
    // GGML_CUDA_FUSE_SWIGLU_MMQ=0 turns it off.
    static const bool fuse_swiglu_mmq = getenv("GGML_CUDA_FUSE_SWIGLU_MMQ") == nullptr || atoi(getenv("GGML_CUDA_FUSE_SWIGLU_MMQ")) != 0;
    if (fuse_swiglu_mmq && node->op == GGML_OP_GLU && ggml_get_glu_op(node) == GGML_GLU_OP_SWIGLU && i + 1 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_MUL_MAT && cgraph->nodes[i + 1]->src[1] == node) {
        ggml_tensor * next = cgraph->nodes[i + 1];
        const ggml_tensor * a = node->src[0];
        const ggml_tensor * b = node->src[1];
        const auto rows_ok = [](const ggml_tensor * t) {
            return t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float) && t->nb[1] % 16 == 0 &&
                   ((uintptr_t) t->data) % 16 == 0 && t->ne[2] == 1 && t->ne[3] == 1;
        };
        const bool shape_ok = node->type == GGML_TYPE_F32 && ggml_is_contiguous(node) && node->ne[0] % 4 == 0 &&
            node->ne[2] == 1 && node->ne[3] == 1 && rows_ok(a) && (b == nullptr || (rows_ok(b) && ggml_are_same_shape(a, b))) &&
            a->ne[0] == (b ? node->ne[0] : 2*node->ne[0]) && a->ne[1] == node->ne[1];
        if (shape_ok && ggml_is_quantized(next->src[0]->type) && ggml_cuda_mul_mat_takes_mmq(*cuda_ctx, next->src[0], node, next) &&
                ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_GLU, GGML_OP_MUL_MAT }, { i + 1 })) {
            ggml_cuda_mul_mat_q_swiglu_dense(*cuda_ctx, next->src[0], next, node);
            return 1;
        }
    }

    //topk-moe
    if (cgraph->nodes[i]->op == GGML_OP_UNARY || cgraph->nodes[i]->op == GGML_OP_SOFT_MAX ||
            cgraph->nodes[i]->op == GGML_OP_ARGSORT) {
        ggml_cuda_topk_moe_args args;
        const bool              can_fuse = ggml_cuda_topk_moe_fusion(cgraph, i, args);
        std::vector<ggml_op>    ops;
        ops.reserve(13);  // max ops; avoids gcc -Wstringop-overflow false positive

        if (can_fuse) {
            const ggml_tensor * logits  = node->src[0];
            ggml_tensor *       weights = nullptr;
            ggml_tensor *       ids     = nullptr;
            const ggml_tensor * bias    = nullptr;
            const ggml_tensor * clamp   = nullptr;
            const ggml_tensor * scale   = nullptr;

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
                    bias = cgraph->nodes[i_probs + 2]->src[1];
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
                    clamp = cgraph->nodes[i + ops.size() - 3];
                }
                if (args.scale) {
                    ops.insert(ops.end(), { GGML_OP_SCALE });
                    scale = cgraph->nodes[i + ops.size() - 1];
                }

                weights      = cgraph->nodes[i + ops.size() - 1];
                out_nodes[1] = i + ops.size() - 1;

                if (!ggml_cuda_topk_moe_fusion_disabled() &&
                        ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                        ggml_cuda_should_use_topk_moe(node, logits, weights, ids) &&
                        ggml_cuda_check_fusion_memory_ranges(cgraph, i, ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
                    ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, clamp, scale, bias, args);
                    return ops.size() - 1;
                }
            } else if (!args.norm && !args.prob_bias) {
                //special case gpt-oss, no norm, no bias.
                ops.insert(ops.end(), { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_RESHAPE,
                                        GGML_OP_SOFT_MAX, GGML_OP_RESHAPE });
                weights                     = cgraph->nodes[i + 5];
                ids                         = cgraph->nodes[i + 1];
                const ggml_tensor * softmax = cgraph->nodes[i + 4];

                int out_nodes[2] = { i + 1, i + 5 };
                if (!ggml_cuda_topk_moe_fusion_disabled() &&
                        ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                        ggml_cuda_should_use_topk_moe(softmax, logits, weights, ids) &&
                        ggml_cuda_check_fusion_memory_ranges(cgraph, i, ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
                    ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, clamp, scale, bias, args);
                    return ops.size() - 1;
                }
            }
        }
    }

    //RoPE + view + set-rows
    static const bool disable_rope_setrows = getenv("GGML_CUDA_DISABLE_ROPE_SETROWS") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_ROPE_SETROWS"));
    if (!disable_rope_setrows && ggml_cuda_can_fuse(cgraph, i, { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
        ggml_tensor * rope     = cgraph->nodes[i];
        ggml_tensor * set_rows = cgraph->nodes[i + 2];

        ggml_cuda_op_rope_fused(*cuda_ctx, rope, set_rows);
        return 2;
    }

    // Snake activation: y = x + sin(a*x)^2 * inv_b
    // Naive 5-op decomposition emitted by frontends: mul -> sin -> sqr -> mul -> add
    static const bool disable_snake = getenv("GGML_CUDA_DISABLE_SNAKE") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_SNAKE"));
    if (!disable_snake && ggml_can_fuse_subgraph(cgraph, i,
            { GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_ADD },
            { i + 4 })) {
        const ggml_tensor * mul0 = cgraph->nodes[i];
        const ggml_tensor * sqr  = cgraph->nodes[i + 2];
        const ggml_tensor * mul1 = cgraph->nodes[i + 3];
        ggml_tensor *       add  = cgraph->nodes[i + 4];

        // x carries the full activation shape, a is the broadcast operand
        const ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
        const ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

        // mul1 reads sqr and inv_b in either operand order
        const ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

        // closure check: the trailing add must read the same x as the leading mul
        const ggml_tensor * x_in_add = (add->src[0] == mul1) ? add->src[1] : add->src[0];

        // Kernel iterates over total = T * C, so x and add must be 2D and
        // a / inv_b must collapse to [1, C, 1, 1]. Higher dims are not handled.
        const bool dim_ok   = (x->ne[2]   == 1 && x->ne[3]   == 1) &&
                              (add->ne[2] == 1 && add->ne[3] == 1) &&
                              (a->ne[2]   == 1 && a->ne[3]   == 1);
        const bool shape_ok = ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];

        // x is in the supported whitelist and every chain intermediate shares
        // x's type. launch_snake reads a and inv_b as const float *, so they
        // stay F32.
        const ggml_tensor * sin1 = cgraph->nodes[i + 1];
        const bool types_ok = (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
                              (a->type    == GGML_TYPE_F32) && (inv_b->type == GGML_TYPE_F32) &&
                              (mul0->type == x->type) && (sin1->type  == x->type) &&
                              (sqr->type  == x->type) && (mul1->type  == x->type) &&
                              (add->type  == x->type);

        // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
        const bool contig_ok = ggml_is_contiguous(x) && ggml_is_contiguous(add) &&
                               ggml_is_contiguous(a) && ggml_is_contiguous(inv_b);

        if (types_ok && shape_ok && dim_ok && contig_ok && x_in_add == x) {
            ggml_cuda_op_snake_fused(*cuda_ctx, x, a, inv_b, add);
            return 4;
        }
    }

    // multi-(add or mul)
    static const bool disable_fused_addmul = getenv("GGML_CUDA_DISABLE_FUSED_ADDMUL") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_FUSED_ADDMUL"));
    if (!disable_fused_addmul && (node->op == GGML_OP_ADD || node->op == GGML_OP_MUL)) {
        int     n_fuse = 0;
        ggml_op ops[8];
        std::fill(ops, ops + 8, node->op);

        for (; n_fuse <= 6; ++n_fuse) {
            if (!ggml_can_fuse(cgraph, i + n_fuse, ops + n_fuse, 2)) {
                break;
            }
            if (cgraph->nodes[i + n_fuse] != cgraph->nodes[i + n_fuse + 1]->src[0]) {
                break;
            }
            if (!ggml_are_same_layout(cgraph->nodes[i + n_fuse]->src[1], cgraph->nodes[i + n_fuse + 1]->src[1])) {
                break;
            }
        }

        n_fuse++;

        if (n_fuse > 1) {
            ggml_tensor fused_node;
            memcpy(&fused_node, node, sizeof(ggml_tensor));
            for (int j = 0; j < n_fuse - 1; ++j) {
                fused_node.src[j + 2] = cgraph->nodes[i + j + 1]->src[1];
            }
            fused_node.data = cgraph->nodes[i + n_fuse - 1]->data;
            if (node->op == GGML_OP_ADD) {
                ggml_cuda_op_fused_add(*cuda_ctx, &fused_node, n_fuse);
            } else {
                ggml_cuda_op_fused_mul(*cuda_ctx, &fused_node, n_fuse);
            }
            return n_fuse - 1;
        }
    }

    bool fused_mul_mat_vec = false;
    int  fused_node_count  = 0;

    // SSM conv-input fusion: qkv_mixed MUL_MAT output feeds the last row of an
    // interleaved [conv_kernel_size, channels] conv input (via view nodes and a
    // dim-0 CONCAT). Fold the concat into the mmvq epilogue: the kernel writes
    // conv_input[cs*c + cs-1] = result and copies the (cs-1) conv states rows
    // from the GET_ROWS output into conv_input[cs*c + k]. The CONCAT is skipped.
    static const bool disable_ssm_conv_in = getenv("GGML_CUDA_DISABLE_SSM_CONV_IN") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_SSM_CONV_IN"));
    if (!disable_ssm_conv_in && node->op == GGML_OP_MUL_MAT && (node->flags & GGML_TENSOR_FLAG_COMPUTE) &&
            ggml_cuda_should_fuse_mul_mat_vec_q(node)) {
        const int scan_end = std::min(cgraph->n_nodes, i + 8);
        for (int j = i + 1; j < scan_end; ++j) {
            const ggml_tensor * n = cgraph->nodes[j];
            if (n->op != GGML_OP_CONCAT || n->op_params[0] != 0) {
                continue;
            }
            if (n->type != GGML_TYPE_F32) {
                continue;
            }
            const ggml_tensor * s1 = n->src[1];
            while (s1 != nullptr && s1->view_src != nullptr) {
                s1 = s1->view_src;
            }
            if (s1 != node) {
                continue;
            }
            const ggml_tensor * s0 = n->src[0];
            while (s0 != nullptr && s0->view_src != nullptr) {
                s0 = s0->view_src;
            }
            if (s0 == nullptr || s0->op != GGML_OP_GET_ROWS || s0->type != GGML_TYPE_F32) {
                continue;
            }
            // conv_input [cs, C] = states [(cs-1)*C] + qkv [C], dim 0
            const int64_t C = n->ne[1] * n->ne[2] * n->ne[3];
            const int64_t cs = n->ne[0];
            if (cs < 2 || n->ne[1] != s1->ne[0]) {
                continue;
            }
            const int64_t src0_elems = s0->ne[0] * s0->ne[1] * s0->ne[2];
            const int64_t src1_elems = s1->ne[0] * s1->ne[1] * s1->ne[2];
            if (src0_elems != (cs - 1) * C || src1_elems != C) {
                continue;
            }
            int out_nodes[] = { j };
            if (!ggml_cuda_check_fusion_memory_ranges(cgraph, i, j - i + 1, out_nodes, 1)) {
                continue;
            }
            // the conv states GET_ROWS must be scheduled before this matmul so
            // the epilogue reads fresh states (the read is not a graph edge)
            bool states_ready = false;
            for (int k = 0; k < i; ++k) {
                if (cgraph->nodes[k] == s0) {
                    states_ready = true;
                    break;
                }
            }
            if (!states_ready) {
                continue;
            }
            ggml_cuda_mm_fusion_args_host fusion_data{};
            fusion_data.conv_input       = n;
            fusion_data.conv_states      = s0;
            fusion_data.conv_kernel_size = cs;
            ggml_cuda_mul_mat_vec_q(*cuda_ctx, node->src[0], node->src[1], node->src[2], node, &fusion_data);
            return j - i;
        }
    }

    auto get_mul_mat_scale = [](const ggml_tensor * scale_node, const ggml_tensor * mm_node) -> const ggml_tensor * {
        const bool scale_lhs_mm = scale_node->src[0] == mm_node;
        const bool scale_rhs_mm = scale_node->src[1] == mm_node;
        if (!scale_lhs_mm && !scale_rhs_mm) {
            return nullptr;
        }

        const ggml_tensor * scale = scale_lhs_mm ? scale_node->src[1] : scale_node->src[0];
        if (mm_node->src[0]->type != GGML_TYPE_NVFP4 || scale_node->type != GGML_TYPE_F32 ||
                scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || ggml_nelements(scale) != 1 ||
                !ggml_are_same_shape(scale_node, mm_node)) {
            return nullptr;
        }

        return scale;
    };

    auto get_mul_mat_id_scale = [](const ggml_tensor * reshape, const ggml_tensor * repeat, const ggml_tensor * getrows,
            const ggml_tensor * scale_node, const ggml_tensor * mm_node) -> const ggml_tensor * {
        if (repeat->src[0] != reshape || getrows->src[0] != repeat || getrows->src[1] != mm_node->src[2]) {
            return nullptr;
        }
        if (!((scale_node->src[0] == mm_node && scale_node->src[1] == getrows) ||
                (scale_node->src[0] == getrows && scale_node->src[1] == mm_node))) {
            return nullptr;
        }

        const ggml_tensor * scale = reshape->src[0];
        if (mm_node->src[0]->type != GGML_TYPE_NVFP4 || scale_node->type != GGML_TYPE_F32 ||
                scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || ggml_nelements(scale) != mm_node->src[0]->ne[2] ||
                !ggml_are_same_shape(scale_node, mm_node)) {
            return nullptr;
        }

        return scale;
    };

    auto get_bias_tensor = [](const ggml_tensor * bias_node, const ggml_tensor * mul_node, ggml_op op_bias) -> const ggml_tensor * {
        if (op_bias == GGML_OP_ADD) {
            if (bias_node->src[0] == mul_node) {
                return bias_node->src[1];
            }
            if (bias_node->src[1] == mul_node) {
                return bias_node->src[0];
            }
            return nullptr;
        }
        GGML_ASSERT(op_bias == GGML_OP_ADD_ID);
        GGML_ASSERT(bias_node->src[0] == mul_node);
        return bias_node->src[1];
    };

    // gate + glu + up, with optional scale/bias on both lanes.
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        if (op == GGML_OP_MUL_MAT) {
            for (const bool with_bias : { false, true }) {
                const int gate_idx       = i;
                const int gate_scale_idx = i + 1;
                const int gate_bias_idx  = with_bias ? i + 2 : -1;
                const int up_idx         = with_bias ? i + 3 : i + 2;
                const int up_scale_idx   = up_idx + 1;
                const int up_bias_idx    = with_bias ? up_idx + 2 : -1;
                const int glu_idx        = with_bias ? up_idx + 3 : up_idx + 2;

                const int out_nodes[] = { glu_idx };
                ggml_op ops[7];
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = bias_op;
                    ops[3] = op;
                    ops[4] = GGML_OP_MUL;
                    ops[5] = bias_op;
                    ops[6] = GGML_OP_GLU;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = op;
                    ops[3] = GGML_OP_MUL;
                    ops[4] = GGML_OP_GLU;
                }
                const int n_ops = with_bias ? 7 : 5;

                if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                        !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                    continue;
                }

                ggml_tensor * gate_n       = cgraph->nodes[gate_idx];
                ggml_tensor * gate_scale_n = cgraph->nodes[gate_scale_idx];
                ggml_tensor * gate_out_n   = with_bias ? cgraph->nodes[gate_bias_idx] : gate_scale_n;
                ggml_tensor * up_n         = cgraph->nodes[up_idx];
                ggml_tensor * up_scale_n   = cgraph->nodes[up_scale_idx];
                ggml_tensor * up_out_n     = with_bias ? cgraph->nodes[up_bias_idx] : up_scale_n;
                const ggml_tensor * glu = cgraph->nodes[glu_idx];

                if (!ggml_cuda_should_fuse_mul_mat(up_n, gate_n, glu,
                        with_bias ? up_out_n : nullptr, with_bias ? gate_out_n : nullptr, up_scale_n, gate_scale_n)) {
                    continue;
                }

                const ggml_tensor * gate_scale = get_mul_mat_scale(gate_scale_n, gate_n);
                const ggml_tensor * up_scale   = get_mul_mat_scale(up_scale_n, up_n);
                if (!gate_scale || !up_scale) {
                    continue;
                }

                const ggml_tensor * up_bias   = with_bias ? get_bias_tensor(up_out_n, up_scale_n, bias_op) : nullptr;
                const ggml_tensor * gate_bias = with_bias ? get_bias_tensor(gate_out_n, gate_scale_n, bias_op) : nullptr;
                if (with_bias && (!ggml_are_same_shape(gate_out_n->src[0], gate_out_n->src[1]) ||
                        !ggml_are_same_shape(up_out_n->src[0], up_out_n->src[1]))) {
                    continue;
                }

                const ggml_tensor * src0 = up_n->src[0];
                const ggml_tensor * src1 = up_n->src[1];
                const ggml_tensor * ids  = up_n->src[2];

                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate       = gate_n->src[0];
                fusion_data.x_bias     = up_bias;
                fusion_data.gate_bias  = gate_bias;
                fusion_data.x_scale    = up_scale;
                fusion_data.gate_scale = gate_scale;
                fusion_data.glu_op     = ggml_get_glu_op(glu);
                fusion_data.glu_limit  = ggml_get_op_params_f32(glu, 3);

                if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n) && (ids != nullptr || !ggml_cuda_rdna3_5_dense_glu_disabled())) {
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }
            }

            if (fused_mul_mat_vec) {
                break;
            }
        } else {
            for (const bool with_bias : { false, true }) {
                const int gate_idx       = i;
                const int gate_scale_idx = i + 4;
                const int gate_bias_idx  = with_bias ? i + 5 : -1;
                const int up_idx         = with_bias ? i + 6 : i + 5;
                const int up_scale_idx   = up_idx + 4;
                const int up_bias_idx    = with_bias ? up_idx + 5 : -1;
                const int glu_idx        = with_bias ? up_idx + 6 : up_idx + 5;

                const int out_nodes[] = { glu_idx };
                ggml_op ops[13];
                if (with_bias) {
                    ops[0]  = op;
                    ops[1]  = GGML_OP_RESHAPE;
                    ops[2]  = GGML_OP_REPEAT;
                    ops[3]  = GGML_OP_GET_ROWS;
                    ops[4]  = GGML_OP_MUL;
                    ops[5]  = bias_op;
                    ops[6]  = op;
                    ops[7]  = GGML_OP_RESHAPE;
                    ops[8]  = GGML_OP_REPEAT;
                    ops[9]  = GGML_OP_GET_ROWS;
                    ops[10] = GGML_OP_MUL;
                    ops[11] = bias_op;
                    ops[12] = GGML_OP_GLU;
                } else {
                    ops[0]  = op;
                    ops[1]  = GGML_OP_RESHAPE;
                    ops[2]  = GGML_OP_REPEAT;
                    ops[3]  = GGML_OP_GET_ROWS;
                    ops[4]  = GGML_OP_MUL;
                    ops[5]  = op;
                    ops[6]  = GGML_OP_RESHAPE;
                    ops[7]  = GGML_OP_REPEAT;
                    ops[8]  = GGML_OP_GET_ROWS;
                    ops[9]  = GGML_OP_MUL;
                    ops[10] = GGML_OP_GLU;
                }
                const int n_ops = with_bias ? 13 : 11;

                if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                        !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                    continue;
                }

                ggml_tensor * gate_n       = cgraph->nodes[gate_idx];
                ggml_tensor * gate_scale_n = cgraph->nodes[gate_scale_idx];
                ggml_tensor * gate_out_n   = with_bias ? cgraph->nodes[gate_bias_idx] : gate_scale_n;
                ggml_tensor * up_n         = cgraph->nodes[up_idx];
                ggml_tensor * up_scale_n   = cgraph->nodes[up_scale_idx];
                ggml_tensor * up_out_n     = with_bias ? cgraph->nodes[up_bias_idx] : up_scale_n;
                const ggml_tensor * glu = cgraph->nodes[glu_idx];

                if (!ggml_cuda_should_fuse_mul_mat(up_n, gate_n, glu,
                        with_bias ? up_out_n : nullptr, with_bias ? gate_out_n : nullptr, up_scale_n, gate_scale_n)) {
                    continue;
                }

                const ggml_tensor * gate_scale = get_mul_mat_id_scale(cgraph->nodes[gate_idx + 1], cgraph->nodes[gate_idx + 2],
                        cgraph->nodes[gate_idx + 3], gate_scale_n, gate_n);
                const ggml_tensor * up_scale = get_mul_mat_id_scale(cgraph->nodes[up_idx + 1], cgraph->nodes[up_idx + 2],
                        cgraph->nodes[up_idx + 3], up_scale_n, up_n);
                if (!gate_scale || !up_scale) {
                    continue;
                }

                const ggml_tensor * up_bias   = with_bias ? get_bias_tensor(up_out_n, up_scale_n, bias_op) : nullptr;
                const ggml_tensor * gate_bias = with_bias ? get_bias_tensor(gate_out_n, gate_scale_n, bias_op) : nullptr;

                const ggml_tensor * src0 = up_n->src[0];
                const ggml_tensor * src1 = up_n->src[1];
                const ggml_tensor * ids  = up_n->src[2];

                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate       = gate_n->src[0];
                fusion_data.x_bias     = up_bias;
                fusion_data.gate_bias  = gate_bias;
                fusion_data.x_scale    = up_scale;
                fusion_data.gate_scale = gate_scale;
                fusion_data.glu_op     = ggml_get_glu_op(glu);
                fusion_data.glu_limit  = ggml_get_op_params_f32(glu, 3);

                if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n) && (ids != nullptr || !ggml_cuda_rdna3_5_dense_glu_disabled())) {
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }
            }

            if (fused_mul_mat_vec) {
                break;
            }
        }

        if (ggml_cuda_can_fuse(cgraph, i, { op, bias_op, op, bias_op, GGML_OP_GLU }, {})) {
            ggml_tensor * glu         = cgraph->nodes[i + 4];
            ggml_tensor * gate_bias_n = glu->src[0];
            ggml_tensor * up_bias_n   = glu->src[1];

            //we don't assume the order for {gate, up}. Instead infer it from the bias tensor
            ggml_tensor * gate_n = nullptr;
            ggml_tensor * up_n   = nullptr;

            if (gate_bias_n->src[0] == cgraph->nodes[i] || gate_bias_n->src[1] == cgraph->nodes[i]) {
                gate_n = cgraph->nodes[i];
                up_n   = cgraph->nodes[i + 2];
            } else if (gate_bias_n->src[0] == cgraph->nodes[i + 2] || gate_bias_n->src[1] == cgraph->nodes[i + 2]) {
                gate_n = cgraph->nodes[i + 2];
                up_n   = cgraph->nodes[i];
            } else {
                continue;
            }

            const ggml_tensor * up_bias_tensor   = get_bias_tensor(up_bias_n, up_n, bias_op);
            const ggml_tensor * gate_bias_tensor = get_bias_tensor(gate_bias_n, gate_n, bias_op);

            if (!up_bias_tensor || !gate_bias_tensor) {
                continue;
            }

            // we don't support repeating adds
            if (bias_op == GGML_OP_ADD && (!ggml_are_same_shape(gate_bias_n->src[0], gate_bias_n->src[1]) ||
                                           !ggml_are_same_shape(up_bias_n->src[0], up_bias_n->src[1]))) {
                continue;
            }

            const ggml_tensor * src0 = up_n->src[0];
            const ggml_tensor * src1 = up_n->src[1];
            const ggml_tensor * ids  = up_n->src[2];

            if (ggml_cuda_should_fuse_mul_mat_vec_f(up_n) && (ids != nullptr || !ggml_cuda_rdna3_5_dense_glu_disabled())) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate_n->src[0];
                fusion_data.x_bias    = up_bias_tensor;
                fusion_data.gate_bias = gate_bias_tensor;
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 5;
                break;
            }

            if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n) && (ids != nullptr || !ggml_cuda_rdna3_5_dense_glu_disabled())) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate_n->src[0];
                fusion_data.x_bias    = up_bias_tensor;
                fusion_data.gate_bias = gate_bias_tensor;
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 5;
                break;
            }
        } else if (ggml_cuda_can_fuse(cgraph, i, { op, op, GGML_OP_GLU }, {})) {
            ggml_tensor * glu  = cgraph->nodes[i + 2];
            ggml_tensor * gate = glu->src[0];
            ggml_tensor * up   = glu->src[1];

            bool ok = (gate == cgraph->nodes[i] && up == cgraph->nodes[i + 1]) ||
                      (gate == cgraph->nodes[i + 1] && up == cgraph->nodes[i]);

            if (!ok) {
                continue;
            }

            const ggml_tensor * src0 = up->src[0];
            const ggml_tensor * src1 = up->src[1];
            const ggml_tensor * ids  = up->src[2];

            // MMB fused MoE gate+up+swiglu (pwilkin port). Prefill-only by the shared predicate.
            if (op == GGML_OP_MUL_MAT_ID && ids != nullptr &&
                    ggml_cuda_mmb_supported_glu(gate->src[0], up->src[0], src1, ids, glu)) {
                ggml_cuda_mul_mat_id_mmb_glu(*cuda_ctx, gate->src[0], up->src[0], src1, ids, glu);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }
            if (ggml_cuda_should_fuse_mul_mat_vec_f(up) && (ids != nullptr || !ggml_cuda_rdna3_5_dense_glu_disabled())) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }

            if (ggml_cuda_should_fuse_mul_mat_vec_q(up) && (ids != nullptr || !ggml_cuda_rdna3_5_dense_glu_disabled())) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                // wip/moe-expert-cache: the decode MoE may read a compact arena + slot-remapped ids
                // instead of the full expert tables.  Redirect BOTH lanes together (the cache took
                // both inputs over, skipping the scheduler copy); on a decline the scheduler's full
                // copy stands and the original tensors are used.
                ggml_tensor src0_c, ids_c, gate_c;
                ggml_cuda_mm_fusion_args_host fusion_local = fusion_data;
                if (moe_cache_redirect_fused(glu, src0, fusion_data.gate, ids, cuda_ctx->device, cuda_ctx->stream(),
                                             &src0_c, &ids_c, &gate_c)) {
                    fusion_local.gate = &gate_c;
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, &src0_c, src1, &ids_c, glu, &fusion_local);
                } else {
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                }
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }

            // Prefill MMQ path: the mmvq/mmvf fused kernels only handle decode
            // (n_tokens <= MMVQ_MAX_BATCH_SIZE) or F32/F16 src0. For the batched
            // quantized case the gate+up+GLU triple runs as separate ops; fuse it
            // into one MMQ kernel that reads both weight streams and applies the
            // GLU epilogue. The J tile-width caps in mul_mat_q_switch_J are tuned
            // on RDNA4 (gfx1201). RDNA3_5 (Strix Halo, gfx1151) was added after
            // validation 2026-09-05 (coherence IDENTICAL fused-on vs off, pp2048
            // +5.3% / pp16384 +4.6% on Qwen3.6-35B-A3B Q3_K_M ub 2048; the
            // RDNA4-tuned J caps transfer). RDNA3_0 (gfx1100, RX 7900XTX) was
            // added after the 2026-09-05 validation on this box: coherence
            // IDENTICAL, pp2048 +9.4% / pp16384 +7.8%, decode unchanged, and the
            // RDNA4-tuned J caps transfer there too (uncapping regressed pp2048
            // 5405->4819 / pp16384 4487->4070; a Q3_K@96 probe at 5094/4251 also
            // lost to the cap 64).
            const bool moe_mmq_type = src0->type == GGML_TYPE_Q3_K || src0->type == GGML_TYPE_Q4_K ||
                                      src0->type == GGML_TYPE_Q5_K || src0->type == GGML_TYPE_Q8_0 ||
                                      src0->type == GGML_TYPE_Q6_K;
            if (op == GGML_OP_MUL_MAT_ID && ids != nullptr && !disable_moe_mmq &&
                    (GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc) || GGML_CUDA_CC_IS_RDNA3_0(cc)) && moe_mmq_type &&
                    ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[2], /*n_experts=*/src0->ne[2])) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    // Dual-output mmvq fusion: two matmuls over the same activation with the
    // same output shape (e.g. the K and V projections of an attention layer).
    // The first matmul computes both results; the gate result is written to the
    // second matmul's destination. Only view/noop nodes may sit between the pair.
    if (cgraph->nodes[i]->op == GGML_OP_MUL_MAT) {
        ggml_tensor * mm_a = cgraph->nodes[i];
        if ((mm_a->flags & GGML_TENSOR_FLAG_COMPUTE) && ggml_cuda_should_fuse_mul_mat_vec_q(mm_a)) {
            for (int j = i + 1; j < std::min(cgraph->n_nodes, i + 8); ++j) {
                ggml_tensor * mid = cgraph->nodes[j];
                if (ggml_cuda_is_view_or_noop(mid)) {
                    continue;
                }
                if (mid->op != GGML_OP_MUL_MAT || !(mid->flags & GGML_TENSOR_FLAG_COMPUTE) ||
                        mid->src[1] != mm_a->src[1] || mid->ne[0] != mm_a->ne[0] ||
                        mid->ne[1] != mm_a->ne[1] || mid->ne[2] != mm_a->ne[2] ||
                        mid->src[0] == mm_a->src[0] || mid->src[0]->type != mm_a->src[0]->type ||
                        !ggml_cuda_should_fuse_mul_mat_vec_q(mid)) {
                    break;
                }
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate     = mid->src[0];
                fusion_data.dst_gate = mid;
                ggml_cuda_mul_mat_vec_q(*cuda_ctx, mm_a->src[0], mm_a->src[1], mm_a->src[2], mm_a, &fusion_data);
                return j - i;
            }
        }
    }

    fused_mul_mat_vec = false;
    fused_node_count  = 0;

    // mul_mat + scale + optional bias
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        for (const bool with_bias : { false, true }) {
            const int n_ops = op == GGML_OP_MUL_MAT ? (with_bias ? 3 : 2) : (with_bias ? 6 : 5);
            const int out_nodes[] = { i + n_ops - 1 };
            ggml_op ops[6];
            if (op == GGML_OP_MUL_MAT) {
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = bias_op;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                }
            } else {
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_RESHAPE;
                    ops[2] = GGML_OP_REPEAT;
                    ops[3] = GGML_OP_GET_ROWS;
                    ops[4] = GGML_OP_MUL;
                    ops[5] = bias_op;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_RESHAPE;
                    ops[2] = GGML_OP_REPEAT;
                    ops[3] = GGML_OP_GET_ROWS;
                    ops[4] = GGML_OP_MUL;
                }
            }

            if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                    !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                continue;
            }

            ggml_tensor * mm_node    = cgraph->nodes[i];
            ggml_tensor * scale_node = op == GGML_OP_MUL_MAT ? cgraph->nodes[i + 1] : cgraph->nodes[i + 4];
            ggml_tensor * out_node   = with_bias ? cgraph->nodes[i + n_ops - 1] : scale_node;

            const ggml_tensor * scale = nullptr;
            if (op == GGML_OP_MUL_MAT) {
                scale = get_mul_mat_scale(scale_node, mm_node);
            } else {
                scale = get_mul_mat_id_scale(cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 3], scale_node, mm_node);
            }
            if (!scale) {
                continue;
            }

            const ggml_tensor * bias = with_bias ? get_bias_tensor(out_node, scale_node, bias_op) : nullptr;
            if (with_bias && !bias) {
                continue;
            }
            if (with_bias && bias_op == GGML_OP_ADD && !ggml_are_same_shape(out_node->src[0], out_node->src[1])) {
                continue;
            }
            if (with_bias && bias_op == GGML_OP_ADD_ID && out_node->src[2] != mm_node->src[2]) {
                continue;
            }

            const ggml_tensor * src0 = mm_node->src[0];
            const ggml_tensor * src1 = mm_node->src[1];
            const ggml_tensor * ids  = mm_node->src[2];

            ggml_cuda_mm_fusion_args_host fusion_data{};
            fusion_data.x_bias  = bias;
            fusion_data.x_scale = scale;

            if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_node)) {
                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, out_node, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = n_ops;
                break;
            }
        }
        if (fused_mul_mat_vec) {
            break;
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    // MoE: ffn_moe_weighted = moe_down * topk_weights. The down projection
    // output is scaled per token (the topk softmax weights); fold the MUL into
    // the matmul epilogue. The pattern is [MUL_MAT_ID, MUL] with the MUL's
    // src1 being a contiguous per-channel F32 vector (a view of the
    // normalized weights).
    static const bool disable_moe_down_fold = getenv("GGML_CUDA_DISABLE_MOE_DOWN_FOLD") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_MOE_DOWN_FOLD"));
    if (!disable_moe_down_fold && i + 1 < cgraph->n_nodes && cgraph->nodes[i]->op == GGML_OP_MUL_MAT_ID) {
        ggml_tensor * mm_node  = cgraph->nodes[i];
        ggml_tensor * mul_node = cgraph->nodes[i + 1];

        const int out_nodes[] = { i + 1 };
        // The x_scale_channel_dst kernel path scales by a per-(expert, token)
        // vector of mm_node->ne[1]*mm_node->ne[2] values (topk weights).
        if (mul_node->op == GGML_OP_MUL &&
                mul_node->src[0] == mm_node &&
                (mm_node->flags & GGML_TENSOR_FLAG_COMPUTE) &&
                (mul_node->flags & GGML_TENSOR_FLAG_COMPUTE) &&
                ggml_cuda_check_fusion_memory_ranges(cgraph, i, 2, out_nodes, 1)) {
            const ggml_tensor * weights = mul_node->src[1];
            if (weights->type == GGML_TYPE_F32 && ggml_is_contiguous(weights) &&
                    weights->ne[0] == 1 && weights->ne[1] == mm_node->ne[1] &&
                    weights->ne[2] == mm_node->ne[2] &&
                    ggml_are_same_shape(mm_node, mul_node) &&
                    ggml_cuda_should_fuse_mul_mat_vec_q(mm_node)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.x_scale             = weights;
                fusion_data.x_scale_channel_dst = true;
                // wip/moe-expert-cache: the down table may be cache-managed; redirect the weight
                // table and the routing onto the compact arena/remap exactly as the gate+up+GLU
                // site does.  The redirect and the scheduler hook key off the same table lookup,
                // so a taken-over input is always redirected and a declined one is fully copied.
                ggml_tensor src0_c, ids_c;
                if (moe_cache_redirect_fused(mm_node, mm_node->src[0], nullptr, mm_node->src[2],
                                             cuda_ctx->device, cuda_ctx->stream(), &src0_c, &ids_c, nullptr)) {
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, &src0_c, mm_node->src[1], &ids_c, mul_node, &fusion_data);
                } else {
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, mm_node->src[0], mm_node->src[1], mm_node->src[2], mul_node, &fusion_data);
                }
                return 1;
            }
        }
    }

    // Pair of L2 norms over two views of the same tensor (the SSM conv output
    // q/k slices). The view source is an external compute tensor, so the
    // subgraph helper cannot express the pattern; check the wiring manually.
    static const bool disable_l2_norm_pair = getenv("GGML_CUDA_DISABLE_L2_NORM_PAIR") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_L2_NORM_PAIR"));
    if (!disable_l2_norm_pair && i + 2 < cgraph->n_nodes &&
            cgraph->nodes[i]->op == GGML_OP_L2_NORM &&
            cgraph->nodes[i + 1]->op == GGML_OP_VIEW &&
            cgraph->nodes[i + 2]->op == GGML_OP_L2_NORM &&
            cgraph->nodes[i + 2]->src[0] == cgraph->nodes[i + 1] &&
            cgraph->nodes[i + 1]->view_src == cgraph->nodes[i]->src[0]->view_src) {
        ggml_tensor * norm0 = cgraph->nodes[i];
        ggml_tensor * view  = cgraph->nodes[i + 1];
        ggml_tensor * norm1 = cgraph->nodes[i + 2];

        float eps0;
        float eps1;
        memcpy(&eps0, norm0->op_params, sizeof(float));
        memcpy(&eps1, norm1->op_params, sizeof(float));

        const bool ok =
            (norm0->flags & GGML_TENSOR_FLAG_COMPUTE) &&
            (norm1->flags & GGML_TENSOR_FLAG_COMPUTE) &&
            norm0->src[0]->type == GGML_TYPE_F32 && norm1->src[0]->type == GGML_TYPE_F32 &&
            ggml_are_same_shape(norm0->src[0], norm1->src[0]) &&
            ggml_are_same_stride(norm0->src[0], norm1->src[0]) &&
            eps0 == eps1;

        if (ok) {
            ggml_cuda_op_l2_norm_pair(*cuda_ctx, norm0->src[0], norm0, norm1->src[0], norm1, eps0);
            return 2;
        }
    }

    // SSM gated delta net: gate = softplus(alpha*x + dt) * a, beta = sigmoid(beta*x).
    // Two small Q8_0 projections over the same input plus the gating chain, fused
    // into one kernel. Both outputs feed only the gated delta net kernel.
    static const bool fuse_gate_beta_verify = getenv("GGML_CUDA_FUSE_GATE_BETA_VERIFY") == nullptr || atoi(getenv("GGML_CUDA_FUSE_GATE_BETA_VERIFY")) != 0;
    if (i + 8 < cgraph->n_nodes && cgraph->nodes[i]->op == GGML_OP_MUL_MAT) {
        const ggml_op ops[9] = {
            GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_MUL,
            GGML_OP_RESHAPE, GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_UNARY
        };
        const int out_nodes[] = { i + 5, i + 8 };

        if (ggml_can_fuse_subgraph(cgraph, i, 9, ops, out_nodes, 2)) {
            ggml_tensor * alpha_w  = cgraph->nodes[i];
            ggml_tensor * alpha_v  = cgraph->nodes[i + 1];
            ggml_tensor * bias     = cgraph->nodes[i + 2];
            ggml_tensor * softplus = cgraph->nodes[i + 3];
            ggml_tensor * gate_mul = cgraph->nodes[i + 4];
            ggml_tensor * gate_v   = cgraph->nodes[i + 5];
            ggml_tensor * beta_w   = cgraph->nodes[i + 6];
            ggml_tensor * beta_v   = cgraph->nodes[i + 7];
            ggml_tensor * beta_sig = cgraph->nodes[i + 8];

            const bool wiring_ok =
                alpha_v->src[0] == alpha_w &&
                softplus->src[0] == bias &&
                gate_v->src[0] == gate_mul &&
                beta_v->src[0] == beta_w &&
                beta_sig->src[0] == beta_v &&
                alpha_w->src[1] == beta_w->src[1];

            const ggml_tensor * dt    = nullptr;
            const ggml_tensor * ssm_a = nullptr;
            if (wiring_ok) {
                if (bias->src[0] == alpha_v) {
                    dt = bias->src[1];
                } else if (bias->src[1] == alpha_v) {
                    dt = bias->src[0];
                }
                if (gate_mul->src[0] == softplus) {
                    ssm_a = gate_mul->src[1];
                } else if (gate_mul->src[1] == softplus) {
                    ssm_a = gate_mul->src[0];
                }
            }

            const bool type_ok =
                dt && ssm_a &&
                dt->type == GGML_TYPE_F32 && ssm_a->type == GGML_TYPE_F32 &&
                ggml_get_unary_op(softplus) == GGML_UNARY_OP_SOFTPLUS &&
                ggml_get_unary_op(beta_sig)  == GGML_UNARY_OP_SIGMOID &&
                alpha_w->src[0]->type == GGML_TYPE_Q8_0 && beta_w->src[0]->type == GGML_TYPE_Q8_0 &&
                alpha_w->src[1]->type == GGML_TYPE_F32 &&
                alpha_w->src[0]->ne[0] == beta_w->src[0]->ne[0] &&
                alpha_w->src[0]->ne[1] == beta_w->src[0]->ne[1] &&
                // decode, and the verify band unless GGML_CUDA_FUSE_GATE_BETA_VERIFY=0
                alpha_w->src[1]->ne[2] == 1 && alpha_w->src[1]->ne[3] == 1 &&
                (alpha_w->src[1]->ne[1] == 1 || (fuse_gate_beta_verify && alpha_w->src[1]->ne[1] <= MMVQ_MAX_BATCH_SIZE)) &&
                ggml_is_contiguous(gate_mul) && ggml_is_contiguous(beta_sig) &&
                ggml_nelements(gate_mul) == alpha_w->src[0]->ne[1]*alpha_w->src[1]->ne[1] &&
                ggml_nelements(beta_sig) == alpha_w->src[0]->ne[1]*alpha_w->src[1]->ne[1];

            if (wiring_ok && type_ok) {
                ggml_cuda_op_ssm_gate_beta(*cuda_ctx, alpha_w->src[0], beta_w->src[0], alpha_w->src[1], dt, ssm_a, gate_mul, beta_sig);
                return 8;
            }
        }
    }

    // Shared-expert output chain: down projection + gate + gating + residual
    // adds. dst = down(swiglu) * sigmoid(gate(x)) + moe_out + ffn_residual.
    //
    // This is DISJOINT from upstream's fused shared-expert MMVQ
    // (ggml_cuda_match_shared_expert / the MUL_MAT_ID branch above, base bed0a8566),
    // which folds the routed+shared gate/up pair and writes the shared GLU; this
    // matcher consumes that GLU plus the shared gate_inp sigmoid.  Both fire on the
    // same layer (verified gfx1201 `-sm layer`, Qwen3.6-35B-A3B Q8_0: upstream
    // 200x + this 160x on one 2-token pass).  Under `-sm tensor` the routed expert
    // is sharded while the shared expert is mirrored, so the routed/shared n_ff
    // differs and the upstream arm stands down; this one still runs.  The
    // LLAMA_HC_BLK16 MWR merge (default off) consumes the ffn_out ADD and so
    // intentionally replaces this epilogue when it is armed.  (wip/shared-expert-fusion-reconcile.)
    //
    // Decode/verify band.  The fused gate reduction (shexp_gate_sigmoid) does not
    // reproduce the order of the standalone mmvq/MUL_MAT it replaces, so the fused and the
    // unfused chain are not bit-identical - and a 1-token decode and an n-token verify of
    // the same MoE layer must be.  The fused kernels are therefore token-generic and pinned
    // to the single-token reduction order, and the whole band (n_tokens <= MMVQ_MAX_BATCH_SIZE)
    // takes the fused path: the multi-token path no longer runs the unfused chain.  Worth +3.1%
    // decode on Qwen3.6-35B-A3B (tg128 101.6 vs 98.5 t/s), and the verify widths gain the same
    // epilogue.  The unfused chain remains the reference; set
    // GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1 to compare against it (then decode and verify differ,
    // as before the 2026-09-11 band amendment).  Hard opt-out for A/B too.
    static const bool disable_shexp_down_gate =
        getenv("GGML_CUDA_DISABLE_SHEXP_DOWN_GATE") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_SHEXP_DOWN_GATE"));
    if (!disable_shexp_down_gate && i + 5 < cgraph->n_nodes && cgraph->nodes[i]->op == GGML_OP_MUL_MAT) {
        const ggml_op ops[6] = {
            GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_ADD, GGML_OP_ADD
        };
        const int out_nodes[] = { i + 5 };

        if (ggml_can_fuse_subgraph(cgraph, i, 6, ops, out_nodes, 1)) {
            ggml_tensor * down_mm = cgraph->nodes[i];
            ggml_tensor * gate_mm = cgraph->nodes[i + 1];
            ggml_tensor * sigmoid = cgraph->nodes[i + 2];
            ggml_tensor * gated   = cgraph->nodes[i + 3];
            ggml_tensor * ffn_out = cgraph->nodes[i + 4];
            ggml_tensor * l_out   = cgraph->nodes[i + 5];

            const bool wiring_ok =
                sigmoid->src[0] == gate_mm &&
                ggml_get_unary_op(sigmoid) == GGML_UNARY_OP_SIGMOID &&
                ((gated->src[0] == down_mm && gated->src[1] == sigmoid) ||
                 (gated->src[0] == sigmoid && gated->src[1] == down_mm)) &&
                ((ffn_out->src[0] == gated && ffn_out->src[1] != gated) ||
                 (ffn_out->src[1] == gated && ffn_out->src[0] != gated)) &&
                l_out->src[0] == ffn_out && l_out->src[1] != ffn_out;

            const ggml_tensor * moe_out = nullptr;
            const ggml_tensor * ffn_residual = nullptr;
            if (wiring_ok) {
                moe_out = ffn_out->src[0] == gated ? ffn_out->src[1] : ffn_out->src[0];
                ffn_residual = l_out->src[1];
            }

            const bool type_ok =
                wiring_ok && moe_out && ffn_residual &&
                down_mm->src[0]->type == GGML_TYPE_Q8_0 &&
                down_mm->src[1]->type == GGML_TYPE_F32 &&
                gate_mm->src[0]->type == GGML_TYPE_F32 &&
                gate_mm->src[1]->type == GGML_TYPE_F32 &&
                // decode/verify band (n_tokens 1..MMVQ_MAX_BATCH_SIZE), both matmuls the same width
                down_mm->src[1]->ne[1] >= 1 && down_mm->src[1]->ne[1] <= MMVQ_MAX_BATCH_SIZE &&
                down_mm->src[1]->ne[1] == gate_mm->src[1]->ne[1] &&
                // the three epilogue operands are plain [n_embd, n_tokens] F32 tensors, so the
                // kernel can address token t as o = t*nrows + row (ggml_cuda_op_shexp_down_gate
                // asserts the same); the gate input x carries its stride explicitly
                ggml_is_contiguous(moe_out) && ggml_is_contiguous(ffn_residual) && ggml_is_contiguous(l_out);

            if (wiring_ok && type_ok) {
                ggml_cuda_op_shexp_down_gate(*cuda_ctx,
                    down_mm->src[0], down_mm->src[1], gate_mm->src[0], gate_mm->src[1],
                    moe_out, ffn_residual, l_out);
                return 5;
            }
        }
    }

    // mul_mat + add, with an optional view (reshape) node between the matmul and the add
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        // view (reshape) between the matmul and the add
        const bool has_view = i + 1 < cgraph->n_nodes && cgraph->nodes[i + 1]->op == GGML_OP_RESHAPE;

        if (has_view) {
            // use ggml_can_fuse_subgraph: views in the subgraph are allowed here
            const ggml_op ops[3] = { op, GGML_OP_RESHAPE, bias_op };
            const int out_nodes[] = { i + 2 };
            if (!ggml_can_fuse_subgraph(cgraph, i, 3, ops, out_nodes, 1) || cgraph->nodes[i + 1]->src[0] != cgraph->nodes[i]) {
                continue;
            }
        } else {
            if (!ggml_can_fuse(cgraph, i, { op, bias_op })) {
                continue;
            }
        }

        ggml_tensor * mm_node   = cgraph->nodes[i];
        ggml_tensor * bias_node = cgraph->nodes[has_view ? i + 2 : i + 1];

        // the add reads the matmul output directly, or through the view
        ggml_tensor * mm_or_view = has_view ? cgraph->nodes[i + 1] : mm_node;

        // The mmvq/mmvf fusion kernels are told to write into bias_node, but the
        // shape checks below (and ggml_cuda_should_fuse_mul_mat_vec_*) look at
        // mm_node.  Without a view those are the same tensor, so the guard is
        // sound.  With one they are not: a reshape can move the tokens between
        // dimensions, so a matmul that looks like a single-column GEMV
        // (ne = [n,1,2]) can be paired with an add whose destination is
        // ne = [n,2,1].  The kernels index the destination by ne[1]/ne[2] and
        // assert on exactly this (mmvq.cu: GGML_ASSERT(ids || dst->ne[1] == 1)).
        // Require the destination to satisfy the same constraint the kernels
        // assert before fusing through a view.  The single-sequence case is
        // unaffected.  (PR #15 / DanoPTT.)
        if (has_view) {
            const ggml_tensor * ids_node = mm_node->src[2];
            if (( ids_node && bias_node->ne[2] != 1) ||
                (!ids_node && bias_node->ne[1] != 1)) {
                continue;
            }
        }

        ggml_tensor * bias_tensor = nullptr;
        if (bias_op == GGML_OP_ADD) {
            if (bias_node->src[0] == mm_or_view) {
                bias_tensor = bias_node->src[1];
            } else if (bias_node->src[1] == mm_or_view) {
                bias_tensor = bias_node->src[0];
            } else {
                continue;
            }
        } else {
            if (bias_node->src[0] != mm_or_view) {
                continue;
            }
            bias_tensor = bias_node->src[1];
        }

        const ggml_tensor * src0 = mm_node->src[0];
        const ggml_tensor * src1 = mm_node->src[1];
        const ggml_tensor * ids  = mm_node->src[2];

        if (bias_op == GGML_OP_ADD_ID && bias_node->src[2] != ids) {
            continue;
        }

        if (bias_op == GGML_OP_ADD && !ggml_are_same_shape(bias_node->src[0], bias_node->src[1])) {
            continue;
        }

        ggml_cuda_mm_fusion_args_host fusion_data{};
        fusion_data.x_bias = bias_tensor;

        if (ggml_cuda_should_fuse_mul_mat_vec_f(mm_node)) {
            ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count  = has_view ? 3 : 2;
            break;
        }

        if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_node)) {
            ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count  = has_view ? 3 : 2;
            break;
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 4]);
        return 4;
    }

    // GDN gate chain for the verify band: ADD(dt) -> SOFTPLUS -> MUL(a) with per-head broadcasts in one
    // kernel (decode fuses it into ssm_gate_beta).  GGML_CUDA_FUSE_GDN_GATE=0 turns it off.
    static const bool fuse_gdn_gate = getenv("GGML_CUDA_FUSE_GDN_GATE") == nullptr || atoi(getenv("GGML_CUDA_FUSE_GDN_GATE")) != 0;
    if (fuse_gdn_gate && node->op == GGML_OP_ADD && i + 2 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_UNARY && cgraph->nodes[i + 2]->op == GGML_OP_MUL &&
            ggml_can_fuse(cgraph, i, { GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_MUL }) &&
            ggml_cuda_op_add_softplus_mul(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2])) {
        return 2;
    }

    // NOTE: RMS_NORM + SCALE is fused upstream now (ggml_cuda_op_rms_norm_scale_fused,
    // handled by the ggml_cuda_can_fuse matcher below); block 08's older separate
    // kernel (GGML_CUDA_FUSE_RMS_SCALE) was retired in the r38 re-base.

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE }, {})) {
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], nullptr);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD }, {})) {
        ggml_cuda_op_rms_norm_fused_add(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
        ggml_cuda_op_rms_norm_fused(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_SCALE }, {})) {
        ggml_cuda_op_rms_norm_scale_fused(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    // Fuse the SSM pre-scan chain: conv+silu, l2_norm(q,k) and the gate/beta
    // projections (pattern built by qwen35moe, decode only). One kernel instead
    // of ssm_conv, l2_norm_pair and ssm_gate_beta.
    static const bool disable_ssm_prescan = getenv("GGML_CUDA_DISABLE_SSM_PRESCAN") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_SSM_PRESCAN"));
    if (!disable_ssm_prescan && node->op == GGML_OP_SSM_CONV && node->type == GGML_TYPE_F32 && node->ne[1] == 1 &&
            i + 15 < cgraph->n_nodes) {
        const ggml_tensor * silu     = cgraph->nodes[i + 1];
        const ggml_tensor * q_view   = cgraph->nodes[i + 2];
        const ggml_tensor * q_norm   = cgraph->nodes[i + 3];
        const ggml_tensor * k_view   = cgraph->nodes[i + 4];
        const ggml_tensor * k_norm   = cgraph->nodes[i + 5];
        const ggml_tensor * v_view   = cgraph->nodes[i + 6];
        const ggml_tensor * alpha_w  = cgraph->nodes[i + 7];
        const ggml_tensor * alpha_v  = cgraph->nodes[i + 8];
        const ggml_tensor * bias     = cgraph->nodes[i + 9];
        const ggml_tensor * softplus = cgraph->nodes[i + 10];
        const ggml_tensor * gate_mul = cgraph->nodes[i + 11];
        const ggml_tensor * gate_v   = cgraph->nodes[i + 12];
        const ggml_tensor * beta_w   = cgraph->nodes[i + 13];
        const ggml_tensor * beta_v   = cgraph->nodes[i + 14];
        const ggml_tensor * beta_sig = cgraph->nodes[i + 15];

        const bool wiring_ok =
            silu->op == GGML_OP_UNARY && ggml_get_unary_op(silu) == GGML_UNARY_OP_SILU && silu->src[0] == node &&
            q_view->op == GGML_OP_VIEW && q_view->view_src == silu &&
            q_norm->op == GGML_OP_L2_NORM && q_norm->src[0] == q_view &&
            k_view->op == GGML_OP_VIEW && k_view->view_src == silu &&
            k_norm->op == GGML_OP_L2_NORM && k_norm->src[0] == k_view &&
            v_view->op == GGML_OP_VIEW && v_view->view_src == silu &&
            alpha_w->op == GGML_OP_MUL_MAT &&
            alpha_v->op == GGML_OP_RESHAPE && alpha_v->src[0] == alpha_w &&
            bias->op == GGML_OP_ADD && (bias->src[0] == alpha_v || bias->src[1] == alpha_v) &&
            softplus->op == GGML_OP_UNARY && ggml_get_unary_op(softplus) == GGML_UNARY_OP_SOFTPLUS && softplus->src[0] == bias &&
            gate_mul->op == GGML_OP_MUL && (gate_mul->src[0] == softplus || gate_mul->src[1] == softplus) &&
            gate_v->op == GGML_OP_RESHAPE && gate_v->src[0] == gate_mul &&
            beta_w->op == GGML_OP_MUL_MAT &&
            beta_v->op == GGML_OP_RESHAPE && beta_v->src[0] == beta_w &&
            beta_sig->op == GGML_OP_UNARY && ggml_get_unary_op(beta_sig) == GGML_UNARY_OP_SIGMOID && beta_sig->src[0] == beta_v &&
            alpha_w->src[1] == beta_w->src[1];

        const ggml_tensor * dt    = nullptr;
        const ggml_tensor * ssm_a = nullptr;
        if (wiring_ok) {
            if (bias->src[0] == alpha_v) {
                dt = bias->src[1];
            } else if (bias->src[1] == alpha_v) {
                dt = bias->src[0];
            }
            if (gate_mul->src[0] == softplus) {
                ssm_a = gate_mul->src[1];
            } else if (gate_mul->src[1] == softplus) {
                ssm_a = gate_mul->src[0];
            }
        }

        // the conv output is split into q [head_k_dim, n_qk_heads], k and v; the
        // channel bases and dims must match the kernel layout
        const int64_t n_qk_ch = q_view->ne[0] * q_view->ne[1];
        const bool type_ok =
            dt && ssm_a &&
            node->src[0]->type == GGML_TYPE_F32 && node->src[1]->type == GGML_TYPE_F32 &&
            node->src[1]->ne[0] >= 2 && node->src[1]->ne[0] <= 15 &&
            q_norm->type == GGML_TYPE_F32 && k_norm->type == GGML_TYPE_F32 &&
            silu->type == GGML_TYPE_F32 &&
            alpha_w->src[0]->type == GGML_TYPE_Q8_0 && beta_w->src[0]->type == GGML_TYPE_Q8_0 &&
            alpha_w->src[1]->type == GGML_TYPE_F32 && alpha_w->src[1]->ne[1] == 1 &&
            gate_mul->type == GGML_TYPE_F32 && beta_sig->type == GGML_TYPE_F32 &&
            alpha_w->src[0]->ne[0] == beta_w->src[0]->ne[0] &&
            alpha_w->src[0]->ne[1] == beta_w->src[0]->ne[1] &&
            q_view->ne[0] == 128 && k_view->ne[0] == 128 && v_view->ne[0] == 128 &&
            q_view->ne[1] == k_view->ne[1] &&
            q_view->view_offs == 0 && k_view->view_offs == n_qk_ch*sizeof(float) &&
            v_view->view_offs == 2*n_qk_ch*sizeof(float) &&
            silu->ne[0] == 2*n_qk_ch + v_view->ne[0]*v_view->ne[1];

        if (wiring_ok && type_ok) {
            const int out_nodes[] = { i + 1, i + 3, i + 5, i + 11, i + 15 };
            if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, 16, out_nodes, 5)) {
                float eps;
                memcpy(&eps, q_norm->op_params, sizeof(float));
                ggml_cuda_op_ssm_conv_l2_gatebeta(*cuda_ctx,
                    node->src[0], node->src[1],
                    alpha_w->src[0], beta_w->src[0], alpha_w->src[1], dt, ssm_a,
                    (ggml_tensor *) q_norm, (ggml_tensor *) k_norm, (ggml_tensor *) silu,
                    (ggml_tensor *) gate_mul, (ggml_tensor *) beta_sig,
                    q_view->ne[0], q_view->ne[1], v_view->ne[0], v_view->ne[1], eps);
                return 15;
            }
        }
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_ADD, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        ggml_cuda_op_ssm_conv(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        ggml_cuda_op_ssm_conv(*cuda_ctx, node, /*bias_add_node=*/ nullptr, cgraph->nodes[i + 1]);
        return 1;
    }

    // prefill hyper-connection fusions (DEFAULT ON). Bit-exact vs the unfused chain, verified on the
    // deterministic substrate with a llama_decode harness (pp + 40 greedy steps, logits hashed per
    // step: fused == unfused bit for bit at 6..1400-token prompts). FIXED 2026-09-06: the comb
    // matcher unwrapped the block_out REPEAT to its input, whose buffer the allocator had reused
    // (e.g. for hc_inject) because the standalone REPEAT ran before the fused window - the kernel
    // then read inject's data as block_out. It now reads the live REPEAT output (mul operand).
    // Opt-outs: GGML_CUDA_DISABLE_HC_FUSION=1 (all), GGML_CUDA_DISABLE_HC_MIX=1 / GGML_CUDA_DISABLE_HC_COMB=1.
    static const bool hc_optin  = !disable_hc_fusion && (!getenv("GGML_CUDA_DISABLE_HC_FUSION") || !std::atoi(getenv("GGML_CUDA_DISABLE_HC_FUSION")));
    static const bool hc_mix_on  = hc_optin && !disable_hc_fusion && (!getenv("GGML_CUDA_DISABLE_HC_MIX")  || !std::atoi(getenv("GGML_CUDA_DISABLE_HC_MIX")));
    static const bool hc_comb_on = hc_optin && !disable_hc_fusion && (!getenv("GGML_CUDA_DISABLE_HC_COMB") || !std::atoi(getenv("GGML_CUDA_DISABLE_HC_COMB")));
    // Diagnostic flag for the hc_combine_norm matchers.  Read once: these paths run per graph
    // node and getenv() takes a lock and rescans the environment block on Windows.
    static const bool hc_cn_dbg = getenv("LLAMA_HC_CN_DEBUG") != nullptr;

    // qwen4exp hyper-connection gate GEMM + sigmoid + stream mix: when the HC gate MUL_MAT's only
    // consumer is the sigmoid that opens an hc_mix window, fold the GEMM into the mix kernel - it
    // removes the gate tensor materialization and one launch per layer.  RDNA3 (gfx1100/gfx1151) and,
    // since 2026-09-24, RDNA4 (gfx120x) - the kernel dequantizes the IQ4_NL gate weight and replays
    // the GEMM's BF16 epilogue rounding via the arch-aware `mmb_frag_t`/`mmb_wmma_bf16`/`MMB_ACC_M`
    // shim, so the fusion is bit-identical to the GEMM + sigmoid + mix chain on both.  A/B / bisect:
    // LLAMA_HC_GATEMIX=0.
    // gfx1100 is a port candidate: the gfx11 WMMA builtin is shared with gfx1151, but `gatemix` stays
    // default OFF on RDNA3_0 (mmb_arch_defaults) until a gfx1151/gfx1201 session A/Bs qwen4exp
    // end-to-end; LLAMA_HC_GATEMIX=1 is the opt-in on gfx1100.
    if (hc_mix_on && ggml_cuda_mmb_gatemix() && (GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) &&
            node->op == GGML_OP_MUL_MAT && i + 1 < cgraph->n_nodes) {
        const ggml_tensor * w  = node->src[0];
        const ggml_tensor * lo = node->src[1];
        if (w != nullptr && lo != nullptr && ggml_is_quantized(w->type) && ggml_node_has_n_uses(cgraph, i, 1)) {
            // (a) unfused mix chain: sigmoid(gate) -> mul -> collapse -> scale
            ggml_cuda_hc_mix_args ma;
            const int count = ggml_cuda_hc_mix_closed(cgraph, i + 1, ma);
            if (count > 0 && ma.gate == node &&
                    ggml_cuda_hc_gate_mix(*cuda_ctx, w, lo, ma.xn, ma.dst, ma.hc, ma.scale, ma.bias)) {
                return count;
            }
            // (b) the explicit ggml_dsv4_hc_pre op (the qwen4exp default): MUL_MAT -> RESHAPE(view)
            //     -> DSV4_HC_PRE.  Lower it to the same kernel so the GEMM is folded either way.
            if (i + 2 < cgraph->n_nodes) {
                const ggml_tensor * gate3 = cgraph->nodes[i + 1];
                const ggml_tensor * pre   = cgraph->nodes[i + 2];
                if (gate3->op == GGML_OP_RESHAPE && gate3->view_src == node &&
                        ggml_node_get_use_count(cgraph, i + 1) == 1 &&
                        pre->op == GGML_OP_DSV4_HC_PRE && pre->type == GGML_TYPE_F32 &&
                        pre->src[1] == gate3 && pre->src[0] != nullptr &&
                        ggml_get_op_params_i32(pre, 1) != 0) {
                    const int hc = (int) pre->src[0]->ne[1];
                    const int64_t M = (int64_t) hc * pre->src[0]->ne[0];
                    const int64_t T = pre->src[0]->ne[2];
                    // the kernel needs the contiguous [hc*n_embd, T] activation the gate GEMM consumed
                    // (the tensor the bf16 cache is keyed on).  The pre op's src[0] is a reshaped view
                    // whose chain may not expose it, so fall back to the gate GEMM's activation input:
                    // the gate's activation chain is silu(scale(MUL_MAT(w_down, xn))).
                    const ggml_tensor * xn = pre->src[0];
                    while (xn->view_src != nullptr && (xn->ne[0] != M || ggml_nrows(xn) != T)) {
                        xn = xn->view_src;
                    }
                    if (xn->ne[0] != M || ggml_nrows(xn) != T) {
                        const ggml_tensor * a = node->src[1];
                        while (a != nullptr && a->op != GGML_OP_MUL_MAT) {
                            a = a->src[0];
                        }
                        if (a != nullptr && a->src[1] != nullptr && a->src[1]->ne[0] == M && ggml_nrows(a->src[1]) == T) {
                            xn = a->src[1];
                        }
                    }
                    if (ggml_cuda_hc_gate_mix(*cuda_ctx, w, lo, xn, (ggml_tensor *) pre, hc,
                            ggml_get_op_params_f32(pre, 0), 0.0f)) {
                        return 2;
                    }
                }
            }
        }
    }
    if (hc_mix_on) {
    // qwen4exp hyper-connection stream mix:
    //     sigmoid(gate) -> mul(xn, .) -> reshape -> view(stream 0) -> cont -> (view(stream c) -> add) x (hc-1) -> scale
    // one kernel reads xn/gate once and writes the [n_embd, T] mean of the gated streams
    {
        ggml_cuda_hc_mix_args args;
        enum ggml_op ops[5 + 2 * 15 + 1];
        int n_ops = ggml_cuda_match_hc_mix(cgraph, i, args, ops);
        if (n_ops > 0 && args.dst->ne[1] == 1) {
            n_ops = 0; // Decode preserves bit-identical routing. Prefill uses the fused reduction.
        }
        if (n_ops > 0) {
            int node_idxs[5 + 2 * 15 + 1];
            for (int j = 0; j < n_ops; ++j) {
                node_idxs[j] = i + j;
            }
            const int out_nodes[] = { i + n_ops - 1 };
            if (ggml_can_fuse_subgraph_ext(cgraph, node_idxs, n_ops, ops, out_nodes, 1)) {
                ggml_cuda_op_hc_mix_reduce(*cuda_ctx, args);
                return n_ops - 1;
            }
        }
    }
    }
    if (hc_comb_on) {
    // halo-box merge (repeat-anchored combine+norm fusion): the model pre-expands block_out/w so
    // the combine run is scale -> sigmoid -> scale -> [reshape] -> repeat -> mul -> add, with the
    // hc-wide block_out REPEAT between the scale chain and the mul. A scale-anchored window would
    // have to include the [n_embd,1,T] reshape view, whose external (non-constant) view_src fails
    // the fuse check, so anchor the window AT the repeat: [repeat, mul, add, rms, (view,) mul(gamma)].
    // The SCALE/SIGMOID/SCALE nodes dispatch standalone (tiny) and their op params feed the kernel.
    // The kernel broadcasts the NARROW base (block_out_hc=false), so the ~80MB materialization never
    // dispatches. Reading the base at the window is safe without pinning it: the nodes between the
    // repeat and mul(gamma) are skipped when fused, so only add and mulg are written, and when
    // either of them reuses the freed base the kernel reads the base from a pool copy (below).
    if (node->op == GGML_OP_REPEAT && node->type == GGML_TYPE_F32 && i + 1 < cgraph->n_nodes) {
        const ggml_tensor * rep = node;
        const int64_t n_embd = rep->ne[0];
        const int64_t hc     = rep->ne[1];
        const int64_t n_tok  = rep->ne[2];
        const auto root_of = [](const ggml_tensor * t) -> const ggml_tensor * {
            while (t != nullptr && t->view_src != nullptr) {
                t = t->view_src;
            }
            return t;
        };
        if (n_embd > 0 && hc >= 2 && hc <= 8 && n_tok >= 1 && rep->ne[3] == 1) {
            const ggml_tensor * base = rep->src[0];
            const ggml_tensor * base_root = base;
            while (base_root != nullptr && base_root->view_src != nullptr) {
                base_root = base_root->view_src;
            }
            if (hc_cn_dbg) {
                static unsigned d = 0;
                if (d++ < 24) fprintf(stderr, "HC_CN[repeat] anchor n_embd=%lld hc=%lld n_tok=%lld root=%s out=%d\n",
                    (long long) n_embd, (long long) hc, (long long) n_tok,
                    base_root ? ggml_type_name(base_root->type) : "null",
                    base_root ? (int) (base_root->flags & GGML_TENSOR_FLAG_OUTPUT) : -1);
            }
            if (base_root != nullptr && base_root->type == GGML_TYPE_F32) {
                const int limit = std::min(cgraph->n_nodes, i + 24);
                // the consuming mul, then the chain: add(residual, mul) -> rms -> (view) -> mul(gamma)
                int k = i + 1;
                while (k < limit && !(cgraph->nodes[k]->op == GGML_OP_MUL &&
                        (cgraph->nodes[k]->src[0] == rep || cgraph->nodes[k]->src[1] == rep))) {
                    ++k;
                }
                if (k + 1 < limit && cgraph->nodes[k]->op == GGML_OP_MUL) {
                    const ggml_tensor * mul = cgraph->nodes[k];
                    int m = k + 1;
                    while (m < limit && ggml_cuda_is_view_or_noop(cgraph->nodes[m]) && !ggml_is_empty(cgraph->nodes[m])) {
                        ++m;
                    }
                    if (m < limit && cgraph->nodes[m]->op == GGML_OP_ADD) {
                        const ggml_tensor * add = cgraph->nodes[m];
                        int q = m + 1;
                        while (q < limit && ggml_cuda_is_view_or_noop(cgraph->nodes[q]) && !ggml_is_empty(cgraph->nodes[q])) {
                            ++q;
                        }
                        if (q < limit && cgraph->nodes[q]->op == GGML_OP_RMS_NORM) {
                            const ggml_tensor * rms = cgraph->nodes[q];
                            int g = q + 1;
                            while (g < limit && ggml_cuda_is_view_or_noop(cgraph->nodes[g]) && !ggml_is_empty(cgraph->nodes[g])) {
                                ++g;
                            }
                            if (g < limit && cgraph->nodes[g]->op == GGML_OP_MUL) {
                                const ggml_tensor * mulg = cgraph->nodes[g];
                                const int64_t hc_dim = n_embd * hc;
                                // the other mul operand roots at the inject scale chain
                                const ggml_tensor * w = nullptr;
                                for (int s2i = 0; s2i < 2; ++s2i) {
                                    const ggml_tensor * src = s2i == 0 ? mul->src[0] : mul->src[1];
                                    if (src != nullptr && src != rep && root_of(src) != root_of(rep)) {
                                        w = src;
                                    }
                                }
                                const ggml_tensor * scale2 = w != nullptr ? root_of(w) : nullptr;
                                // chain: scale1(inject) -> sigmoid -> scale2  (scale2 is the w root)
                                const ggml_tensor * sigm  = scale2 != nullptr ? scale2->src[0] : nullptr;
                                const ggml_tensor * scale1 = sigm   != nullptr ? sigm->src[0]   : nullptr;
                                const ggml_tensor * inject = scale1 != nullptr ? scale1->src[0] : nullptr;
                                const ggml_tensor * res = add->src[0] == mul ? add->src[1] : add->src[0];
                                const ggml_tensor * gamma = nullptr;
                                if (root_of(mulg->src[0]) == (const ggml_tensor *) rms) {
                                    gamma = mulg->src[1];
                                } else if (root_of(mulg->src[1]) == (const ggml_tensor *) rms) {
                                    gamma = mulg->src[0];
                                }
                                const bool w_ok = w != nullptr && scale2 != nullptr && scale1 != nullptr && sigm != nullptr &&
                                    scale2->op == GGML_OP_SCALE && scale1->op == GGML_OP_SCALE &&
                                    sigm->op == GGML_OP_UNARY && ggml_get_unary_op(sigm) == GGML_UNARY_OP_SIGMOID &&
                                    sigm->src[0] == scale1 && scale1->src[0] == inject &&
                                    w->type == GGML_TYPE_F32 && w->ne[0] == 1 && w->ne[1] == hc &&
                                    ggml_nelements(w) == hc * n_tok && ggml_is_contiguous(w);
                                const bool b_ok = base_root->ne[0] == n_embd && ggml_is_contiguous(base_root) &&
                                    ggml_nelements(base_root) == n_embd * n_tok;
                                const bool ok_a = add->ne[3] == 1 && n_tok <= 65535 &&
                                    inject != nullptr && inject->type == GGML_TYPE_F32 &&
                                    ggml_is_contiguous(inject) && ggml_nelements(inject) == hc * n_tok;
                                const bool ok_b = (add->src[0] == mul || add->src[1] == mul) && res != nullptr && res != mul &&
                                    root_of((ggml_tensor *) rms->src[0]) == (const ggml_tensor *) add && rms->src[0] != nullptr;
                                const bool ok_c = mul->type == GGML_TYPE_F32 && add->type == GGML_TYPE_F32 && res->type == GGML_TYPE_F32 &&
                                    rms->type == GGML_TYPE_F32 && mulg->type == GGML_TYPE_F32 && gamma != nullptr && gamma->type == GGML_TYPE_F32;
                                const bool ok_d = ggml_are_same_shape(mul, add) && ggml_are_same_shape(res, add) && ggml_are_same_shape(rms, add);
                                const bool ok_e = ggml_is_contiguous(res) && ggml_is_contiguous(add) && ggml_is_contiguous(rms) &&
                                    ggml_is_contiguous(mulg) && ggml_is_contiguous(gamma);
                                const bool shape3 = mulg->ne[0] == n_embd && mulg->ne[1] == hc && mulg->ne[2] == n_tok && mulg->ne[3] == 1 &&
                                    gamma->ne[0] == n_embd && gamma->ne[1] == hc;
                                const bool shape2 = mulg->ne[0] == hc_dim && mulg->ne[1] == n_tok && mulg->ne[2] == 1 && mulg->ne[3] == 1 &&
                                    gamma->ne[0] == hc_dim;
                                const bool ok_f = (shape3 || shape2) && ggml_nelements(gamma) == hc_dim &&
                                    ggml_nelements(mulg) == hc_dim * n_tok && w_ok && b_ok;
                                const bool ok = ok_a && ok_b && ok_c && ok_d && ok_e && ok_f;
                                if (!ok && hc_cn_dbg) {
                                    static unsigned dbg = 0;
                                    if (dbg++ < 40) fprintf(stderr, "HC_CN[repeat] ok=%d abcdef=%d%d%d%d%d%d n_embd=%lld hc=%lld n_tok=%lld mulg=[%lld,%lld,%lld,%lld] gamma=[%lld,%lld] out=%d\n",
                                        (int) ok, ok_a, ok_b, ok_c, ok_d, ok_e, ok_f,
                                        (long long) n_embd, (long long) hc, (long long) n_tok,
                                        (long long) mulg->ne[0], (long long) mulg->ne[1], (long long) mulg->ne[2], (long long) mulg->ne[3],
                                        (long long) gamma->ne[0], (long long) gamma->ne[1], (int) (base_root->flags & GGML_TENSOR_FLAG_OUTPUT));
                                }
                                if (ok) {
                                    ggml_cuda_hc_combine_norm_args args;
                                    args.inject       = inject;
                                    args.residual     = res;
                                    args.block_out    = base_root;
                                    args.block_out_hc = false;
                                    args.gamma        = gamma;
                                    args.out_res      = (ggml_tensor *) add;
                                    args.out_xn       = (ggml_tensor *) mulg;
                                    args.s1           = ggml_get_op_params_f32(scale1, 0);
                                    args.b1           = ggml_get_op_params_f32(scale1, 1);
                                    args.s2           = ggml_get_op_params_f32(scale2, 0);
                                    args.b2           = ggml_get_op_params_f32(scale2, 1);
                                    memcpy(&args.eps, rms->op_params, sizeof(float));
                                    auto overlap = [](const ggml_tensor * p, const ggml_tensor * q) {
                                        const uintptr_t p0 = (uintptr_t) p->data, p1 = p0 + ggml_backend_buft_get_alloc_size(p->buffer->buft, p);
                                        const uintptr_t q0 = (uintptr_t) q->data, q1 = q0 + ggml_backend_buft_get_alloc_size(q->buffer->buft, q);
                                        return p0 < q1 && q0 < p1;
                                    };
                                    const bool res_inplace = res->data == add->data && ggml_are_same_layout(res, add);
                                    const bool alias_ok_but_base = !overlap(add, inject) && (res_inplace || !overlap(add, res)) &&
                                        !overlap(mulg, inject) && !overlap(mulg, res) && !overlap(mulg, add) &&
                                        !overlap(add, gamma) && !overlap(mulg, gamma);
                                    bool alias_ok = alias_ok_but_base && !overlap(add, base_root) && !overlap(mulg, base_root);
                                    // block_out is not pinned: it dies at the repeat, so add or mulg may reuse its buffer.
                                    // Nothing in the window runs before the fused kernel, so the base is still intact here:
                                    // read it from a pool copy instead of falling back to the unfused nodes.
                                    ggml_cuda_pool_alloc<char> base_copy_buf(cuda_ctx->pool());
                                    ggml_tensor base_copy;
                                    if (!alias_ok && alias_ok_but_base && n_tok > 1 && ggml_is_contiguous(base_root)) {
                                        base_copy = *base_root;
                                        base_copy.data = base_copy_buf.alloc(ggml_nbytes(base_root));
                                        CUDA_CHECK(cudaMemcpyAsync(base_copy.data, base_root->data, ggml_nbytes(base_root),
                                            cudaMemcpyDeviceToDevice, cuda_ctx->stream()));
                                        args.block_out = &base_copy;
                                        alias_ok = true;
                                        if (hc_cn_dbg) {
                                            static unsigned d3 = 0;
                                            if (d3++ < 24) fprintf(stderr, "HC_CN[repeat] base copy n_tok=%lld\n", (long long) n_tok);
                                        }
                                    }
                                    if (!alias_ok && n_tok == 1 && hc <= 4) {
                                        args.single_block = true;
                                        alias_ok = true;
                                    }
                                    int          idxs[32];
                                    enum ggml_op ops[32];
                                    int          count = 0;
                                    for (int kk = i; kk <= g; ++kk) {
                                        idxs[count] = kk;
                                        ops[count]  = cgraph->nodes[kk]->op;
                                        ++count;
                                    }
                                    const int out_nodes[] = { m, g };
                                    const bool supp = ggml_cuda_hc_combine_norm_supported(args, ggml_cuda_info().devices[cuda_ctx->device].warp_size);
                                    const bool canf = ggml_can_fuse_subgraph_ext(cgraph, idxs, count, ops, out_nodes, 2);
                                    if (!(alias_ok && supp && canf) && hc_cn_dbg) {
                                        static unsigned d2 = 0;
                                        if (d2++ < 24) fprintf(stderr, "HC_CN[repeat] dispatch alias=%d supp=%d canf=%d n_tok=%lld hc=%lld %s\n",
                                            (int) alias_ok, (int) supp, (int) canf, (long long) n_tok, (long long) hc,
                                            ggml_can_fuse_subgraph_ext(cgraph, idxs, count, ops, out_nodes, 2) ? "" : "(canf)");
                                    }
                                    if (alias_ok && supp && canf) {
                                        ggml_cuda_hc_combine_norm_set_bf16(*cuda_ctx, args);
                                        ggml_cuda_op_hc_combine_norm(*cuda_ctx, args);
                                        return g - i;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // qwen4exp hyper-connection combine + next grouped rms norm (A-chain):
    //   scale(inject) -> sigmoid -> scale -> [view w] -> mul(block_out x w) -> add(residual)
    //     -> [view] -> rms_norm -> [view] -> mul(xn, gamma)
    // A's graph has no REPEAT node: block_out is already [n_embd, 1, T] and the mul broadcasts.
    // One kernel (block per stream+token) writes out_res = the add ([n_embd, hc, T]) and
    // out_xn = the gamma mul ([n_embd*hc, T]); the two share one row-major layout.
    if (node->op == GGML_OP_SCALE && node->type == GGML_TYPE_F32 && i + 3 < cgraph->n_nodes &&
        cgraph->nodes[i + 1]->op == GGML_OP_UNARY && ggml_get_unary_op(cgraph->nodes[i + 1]) == GGML_UNARY_OP_SIGMOID &&
        cgraph->nodes[i + 1]->src[0] == node &&
        cgraph->nodes[i + 2]->op == GGML_OP_SCALE && cgraph->nodes[i + 2]->src[0] == cgraph->nodes[i + 1]) {
        const ggml_tensor * scale1 = node;
        const ggml_tensor * scale2 = cgraph->nodes[i + 2];
        const ggml_tensor * inject = scale1->src[0];
        const auto root_of = [](const ggml_tensor * t) -> const ggml_tensor * {
            while (t != nullptr && t->view_src != nullptr) {
                t = t->view_src;
            }
            return t;
        };

        const int limit = std::min(cgraph->n_nodes, i + 24);
        int k = i + 3;
        while (k < limit && ggml_cuda_is_view_or_noop(cgraph->nodes[k]) && !ggml_is_empty(cgraph->nodes[k])) {
            ++k;
        }
        if (k + 1 < limit && cgraph->nodes[k]->op == GGML_OP_MUL) {
            const ggml_tensor * mul = cgraph->nodes[k];
            int m = k + 1;
            while (m < limit && ggml_cuda_is_view_or_noop(cgraph->nodes[m]) && !ggml_is_empty(cgraph->nodes[m])) {
                ++m;
            }
            if (m < limit && cgraph->nodes[m]->op == GGML_OP_ADD) {
                const ggml_tensor * add = cgraph->nodes[m];
                int q = m + 1;
                while (q < limit && ggml_cuda_is_view_or_noop(cgraph->nodes[q]) && !ggml_is_empty(cgraph->nodes[q])) {
                    ++q;
                }
                if (q < limit && cgraph->nodes[q]->op == GGML_OP_RMS_NORM) {
                    const ggml_tensor * rms = cgraph->nodes[q];
                    int g = q + 1;
                    while (g < limit && ggml_cuda_is_view_or_noop(cgraph->nodes[g]) && !ggml_is_empty(cgraph->nodes[g])) {
                        ++g;
                    }
                    if (g < limit && cgraph->nodes[g]->op == GGML_OP_MUL) {
                        const ggml_tensor * mulg = cgraph->nodes[g];
                        const int64_t n_embd = add->ne[0];
                        const int64_t hc     = add->ne[1];
                        const int64_t n_tok  = add->ne[2];
                        const int64_t hc_dim = n_embd * hc;

                        const ggml_tensor * w = nullptr;
                        const ggml_tensor * b = nullptr;
                        for (int s = 0; s < 2; ++s) {
                            const ggml_tensor * src = s == 0 ? mul->src[0] : mul->src[1];
                            if (src == nullptr) { continue; }
                            if (root_of(src) == (const ggml_tensor *) scale2) {
                                w = src;
                            } else {
                                b = src;
                            }
                        }
                        // block_out reaches the mul either as a REPEAT output ([n_embd, hc, T], copies of
                        // the broadcast base) or directly ([n_embd, 1, T]-style, ggml broadcast). Pass the
                        // mul's actual operand: it is consumed inside the fused window, so its buffer is
                        // live there. NEVER unwrap to the repeat INPUT - the standalone REPEAT may run
                        // before the window (its input then dies and the allocator reuses the buffer,
                        // e.g. for hc_inject), leaving the fused kernel reading the wrong data.
                        const ggml_tensor * bo = b;
                        // if bo is itself a REPEAT op's output, that op must have run already (producer
                        // index < i); a repeat inside the window cannot be materialized by this fusion
                        if (bo != nullptr && bo->op == GGML_OP_REPEAT) {
                            bool producer_before = false;
                            for (int j = 0; j < i && !producer_before; ++j) {
                                producer_before = cgraph->nodes[j] == bo;
                            }
                            if (!producer_before) {
                                bo = nullptr;
                            }
                        }
                        const ggml_tensor * res = add->src[0] == mul ? add->src[1] : add->src[0];
                        // the rms output may reach the gamma MUL through a reshape view: identify the
                        // operand whose root is rms and take the other side as gamma
                        const ggml_tensor * gamma = nullptr;
                        if (root_of(mulg->src[0]) == (const ggml_tensor *) rms) {
                            gamma = mulg->src[1];
                        } else if (root_of(mulg->src[1]) == (const ggml_tensor *) rms) {
                            gamma = mulg->src[0];
                        }

                        const bool w_ok = w != nullptr && w->type == GGML_TYPE_F32 && w->ne[0] == 1 && w->ne[1] == hc &&
                            ggml_nelements(w) == hc * n_tok && ggml_is_contiguous(w);
                        // the mul operand has either hc stacked copies (a REPEAT output, rows of n_embd
                        // per (c, t)) or a single copy broadcast over c (rows of n_embd per token)
                        const bool bo_hc = bo != nullptr && ggml_nelements(bo) == n_embd * hc * n_tok;
                        const bool b_ok = bo != nullptr && bo->type == GGML_TYPE_F32 && bo->ne[0] == n_embd &&
                            ggml_is_contiguous(bo) &&
                            (bo_hc || ggml_nelements(bo) == n_embd * n_tok);
                        const bool ok_a = add->ne[3] == 1 && n_tok >= 1 && n_tok <= 65535 &&
                            inject->type == GGML_TYPE_F32 && ggml_is_contiguous(inject) &&
                            ggml_nelements(inject) == hc * n_tok && ggml_nrows(inject) == n_tok &&
                            ggml_are_same_shape(scale1, inject) && ggml_are_same_shape(scale2, inject);
                        const bool ok_b = (add->src[0] == mul || add->src[1] == mul) && res != nullptr && res != mul &&
                            root_of((ggml_tensor *) rms->src[0]) == (const ggml_tensor *) add && rms->src[0] != nullptr;
                        const bool ok_c = mul->type == GGML_TYPE_F32 && add->type == GGML_TYPE_F32 && res->type == GGML_TYPE_F32 &&
                            rms->type == GGML_TYPE_F32 && mulg->type == GGML_TYPE_F32 && gamma != nullptr && gamma->type == GGML_TYPE_F32;
                        const bool ok_d = ggml_are_same_shape(mul, add) && ggml_are_same_shape(res, add) && ggml_are_same_shape(rms, add);
                        const bool ok_e = ggml_is_contiguous(res) && ggml_is_contiguous(add) && ggml_is_contiguous(rms) &&
                            ggml_is_contiguous(mulg) && ggml_is_contiguous(gamma);
                        const bool shape3 = mulg->ne[0] == n_embd && mulg->ne[1] == hc && mulg->ne[2] == n_tok && mulg->ne[3] == 1 &&
                            gamma->ne[0] == n_embd && gamma->ne[1] == hc;
                        const bool shape2 = mulg->ne[0] == hc_dim && mulg->ne[1] == n_tok && mulg->ne[2] == 1 && mulg->ne[3] == 1 &&
                            gamma->ne[0] == hc_dim;
                        const bool ok_f = (shape3 || shape2) && ggml_nelements(gamma) == hc_dim &&
                            ggml_nelements(mulg) == hc_dim * n_tok && w_ok && b_ok;
                        const bool ok = ok_a && ok_b && ok_c && ok_d && ok_e && ok_f;
                        if (!ok && hc_cn_dbg) {
                            static unsigned dbg = 0;
                            if (dbg++ < 40) fprintf(stderr, "HC_CN[chain] ok=%d abcdef=%d%d%d%d%d%d n_embd=%lld hc=%lld n_tok=%lld mulg=[%lld,%lld,%lld,%lld] gamma=[%lld,%lld]\n",
                                (int) ok, ok_a, ok_b, ok_c, ok_d, ok_e, ok_f,
                                (long long) n_embd, (long long) hc, (long long) n_tok,
                                (long long) mulg->ne[0], (long long) mulg->ne[1], (long long) mulg->ne[2], (long long) mulg->ne[3],
                                (long long) gamma->ne[0], (long long) gamma->ne[1]);
                        }
                        if (ok) {
                            ggml_cuda_hc_combine_norm_args args;
                            args.inject    = inject;
                            args.residual  = res;
                            args.block_out = bo;
                            args.block_out_hc = bo_hc;
                            args.gamma     = gamma;
                            args.out_res   = (ggml_tensor *) add;
                            args.out_xn    = (ggml_tensor *) mulg;
                            args.s1        = ggml_get_op_params_f32(scale1, 0);
                            args.b1        = ggml_get_op_params_f32(scale1, 1);
                            args.s2        = ggml_get_op_params_f32(scale2, 0);
                            args.b2        = ggml_get_op_params_f32(scale2, 1);
                            memcpy(&args.eps, rms->op_params, sizeof(float));

                            // every block reads block_out and inject and writes its own stream of both outputs:
                            // the outputs must not overlap those, and the residual may only alias out_res in place
                            auto overlap = [](const ggml_tensor * p, const ggml_tensor * q) {
                                const uintptr_t p0 = (uintptr_t) p->data, p1 = p0 + ggml_backend_buft_get_alloc_size(p->buffer->buft, p);
                                const uintptr_t q0 = (uintptr_t) q->data, q1 = q0 + ggml_backend_buft_get_alloc_size(q->buffer->buft, q);
                                return p0 < q1 && q0 < p1;
                            };
                            const bool res_inplace = res->data == add->data && ggml_are_same_layout(res, add);
                            // the kernel reads bo (block_out = the repeat input), inject, gamma and (via add)
                            // res; it writes add and mulg. All checks must use bo, NOT the repeat output.
                            bool alias_ok = !overlap(add, bo) && !overlap(add, inject) && (res_inplace || !overlap(add, res)) &&
                                !overlap(mulg, bo) && !overlap(mulg, inject) && !overlap(mulg, res) && !overlap(mulg, add) &&
                                !overlap(add, gamma) && !overlap(mulg, gamma);
                            // outputs overlapping the inputs (typically xn reusing the block_out buffer, which dies
                            // at the combine): single-block variant that reads everything before it writes
                            if (!alias_ok && n_tok == 1 && hc <= 4) {
                                args.single_block = true;
                                alias_ok = true;
                            }

                            int          idxs[32];
                            enum ggml_op ops[32];
                            int          count = 0;
                            for (int kk = i; kk <= g; ++kk) {
                                idxs[count] = kk;
                                ops[count]  = cgraph->nodes[kk]->op;
                                ++count;
                            }
                            const int out_nodes[] = { m, g };
                            const bool supp = ggml_cuda_hc_combine_norm_supported(args, ggml_cuda_info().devices[cuda_ctx->device].warp_size);
                            const bool canf = ggml_can_fuse_subgraph_ext(cgraph, idxs, count, ops, out_nodes, 2);
                            if (alias_ok && supp && canf) {
                                ggml_cuda_hc_combine_norm_set_bf16(*cuda_ctx, args);
                                ggml_cuda_op_hc_combine_norm(*cuda_ctx, args);
                                return g - i;
                            }
                        }
                    }
                }
            }
        }
    }

    }

    // UNARY(sigmoid) -> MUL -> ADD with a per-row gate (GGML_CUDA_FUSE_SIGMOID_MUL_ADD=0: off): one kernel
    static const bool fuse_sig_mul_add = getenv("GGML_CUDA_FUSE_SIGMOID_MUL_ADD") == nullptr || atoi(getenv("GGML_CUDA_FUSE_SIGMOID_MUL_ADD")) != 0;
    if (fuse_sig_mul_add && node->op == GGML_OP_UNARY && i + 2 < cgraph->n_nodes &&
            ggml_cuda_sigmoid_mul_add_fusable(node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]) &&
            ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_ADD }, { i + 2 })) {
        // memory overlap is checked by the matcher (the in-place ADD the allocator picks is allowed)
        ggml_cuda_op_sigmoid_mul_add(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SILU }) ||
        ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SIGMOID }) ||
        ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SOFTPLUS })) {
        ggml_tensor * mul_node = cgraph->nodes[i + 1];

        // If the product feeds a single decode matmul (directly or via a
        // no-op reshape), write its Q8_1 quantized value instead of the F32
        // output; the matmul launcher finds it via the quantize cache.
        const ggml_tensor * mm = ggml_cuda_find_mul_q8_1_matmul(cgraph, i + 1, mul_node);
        if (mm != nullptr) {
            ggml_cuda_op_unary_mul_q8_1(*cuda_ctx, node, mul_node, mm);
            return cgraph->nodes[i + 2] == mm ? 1 : 2;
        }

        ggml_cuda_op_unary_mul(*cuda_ctx, node, mul_node);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_SQR }, { GGML_UNARY_OP_RELU })) {
        ggml_cuda_op_relu_sqr(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE }, { GGML_UNARY_OP_TANH })) {
        ggml_cuda_op_softcap(*cuda_ctx, cgraph->nodes[i + 2], node);
        return 2;
    }

    // scale -> silu/sigmoid unary (qwen4exp hyper-connection low-rank gate,
    // e.g. silu(x / hc) in build_hc_mix's prefill path): apply the activation
    // while scaling - one kernel instead of scale_f32 + unary (halo-box port
    // ggml_cuda_op_scale_unary). Placed last so the larger hc windows
    // (scale->sigmoid->scale->mul->add->rms... chains) match first; only
    // standalone scale->unary pairs reach here. Numerically identical
    // (same per-element expression). Opt-out: GGML_CUDA_SCALE_UNARY=0.
    static const bool disable_scale_unary = getenv("GGML_CUDA_SCALE_UNARY") != nullptr && std::atoi(getenv("GGML_CUDA_SCALE_UNARY")) == 0;
    if (!disable_scale_unary && node->op == GGML_OP_SCALE && node->type == GGML_TYPE_F32 && i + 1 < cgraph->n_nodes) {
        const ggml_tensor * next = cgraph->nodes[i + 1];
        if (next->op == GGML_OP_UNARY && next->type == GGML_TYPE_F32 && next->src[0] == node &&
                node->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(node->src[0]) &&
                (ggml_get_unary_op(next) == GGML_UNARY_OP_SILU || ggml_get_unary_op(next) == GGML_UNARY_OP_SIGMOID) &&
                ggml_can_fuse(cgraph, i, (const enum ggml_op[]) { GGML_OP_SCALE, GGML_OP_UNARY }, 2)) {
            // No ggml_cuda_check_fusion_memory_ranges gate here: the fused kernel is
            // PURELY ELEMENTWISE (dst[i] = op(scale*x[i] + bias)), so even when the
            // unary dst aliases the scale's src (in-place scale: the allocator reuses
            // the src buffer for the unary dst) the kernel is in-place-safe - each
            // thread reads only its own index. Whole-buffer allocator reuse means an
            // overlap is base-aligned (dst == src), never shifted. (The sigmoid arm
            // of this window passes the general check; the silu pairs below, whose
            // scale runs in-place on its 10K-wide input, need this relaxation - the
            // census showed mem_ok=0 for all 190 of them.)
            ggml_cuda_op_scale_unary(*cuda_ctx, node, cgraph->nodes[i + 1]);
            return 1;
        }
    }

    return 0;
}

static void ggml_cuda_graph_evaluate_and_capture(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const bool use_cuda_graph, const bool cuda_graph_update_required, const ggml_cuda_graph_key & graph_key) {
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
            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
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
        ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
        if (use_cuda_graph && cuda_graph_update_required) { // End CUDA graph capture
            if (graph->graph != nullptr) {
                CUDA_CHECK(cudaGraphDestroy(graph->graph));
                graph->graph = nullptr;
            }

            // WIP fused-stage: capture the AR stage kernel as the graph's LAST
            // node so each device's wire staging + arrival token are ready at
            // subgraph-end instead of after the separate AR kernel's dispatch
            // (which carries a per-device graph->kernel premium).  No-op when
            // the internal AR pipeline isn't in fused mode.
            if (cgraph->n_nodes > 0 && cgraph->nodes[cgraph->n_nodes-1] != nullptr) {
                const ggml_tensor * last = cgraph->nodes[cgraph->n_nodes-1];
                ggml_cuda_ar_stage_hook_run(cuda_ctx->device, cuda_ctx->stream(),
                                            static_cast<const float *>(last->data),
                                            ggml_nelements(last));
            }

            CUDA_CHECK(cudaStreamEndCapture(cuda_ctx->stream(), &graph->graph));
            graph_evaluated_or_captured = true; // CUDA graph has been captured

            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            if (ggml_cuda_lock_counter.fetch_sub(1, std::memory_order_relaxed) == 1) {
                ggml_cuda_lock_cv.notify_all();
            }
        } else {
            graph_evaluated_or_captured = true; // ggml graph has been directly evaluated
        }
    }

    if (use_cuda_graph) {
        ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
        if (graph->instance == nullptr) { // Create executable graph from captured graph.
            CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));
        }
        if (cuda_graph_update_required) { // Update graph executable
            ggml_cuda_graph_update_executable(cuda_ctx, graph_key);
        }
        // Launch graph
        CUDA_CHECK(cudaGraphLaunch(graph->instance, cuda_ctx->stream()));
#else
        GGML_UNUSED(graph_key);
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
static bool ggml_cuda_graph_set_enabled(ggml_backend_cuda_context * cuda_ctx, const ggml_cuda_graph_key & graph_key) {
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

    if (graph->graph == nullptr) {
        if (ggml_cuda_info().devices[cuda_ctx->device].cc < GGML_CUDA_CC_VOLTA) {
            if (!graph->disable_due_to_gpu_arch) {
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

    if (g_stream_dbg_on) {
        static int n = 0;
        if (n++ < 24) fprintf(stderr, "STREAMDBG compute dev=%d stream=%p first=%s nodes=%d\n",
            cuda_ctx->device, (void *) cuda_ctx->stream(), cgraph->n_nodes ? cgraph->nodes[0]->name : "-", cgraph->n_nodes);
    }

    // The Q8_1 input cache is only valid within one graph execution.
    cuda_ctx->q8_1_cache_clear();
    ggml_cuda_mmb_set_active_ctx(cuda_ctx);
    ggml_cuda_mmb_begin_graph();

    bool use_cuda_graph             = false;
    bool cuda_graph_update_required = false;
    ggml_cuda_graph_key graph_key = {};

    // op timing instruments each node with stream events, which is not possible during capture
    const bool op_timing = g_op_timing_on;

#ifdef USE_CUDA_GRAPH
    graph_key = ggml_cuda_graph_get_key(cgraph);

    ggml_cuda_graph_set_enabled(cuda_ctx, graph_key);

    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
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
            const bool properties_changed = ggml_cuda_graph_update_required(cuda_ctx, cgraph);

            if (!graph->warmup_complete) {
                // Warmup: need at least 2 calls with no property change on the 2nd call
                if (!properties_changed) {
                    graph->warmup_complete = true;
                    GGML_LOG_DEBUG("%s: CUDA graph warmup complete\n", __func__);
                    use_cuda_graph = true;
                    cuda_graph_update_required = true;
                }
                // else: properties changed or first call - execute directly (use_cuda_graph stays false)
            } else {
                // Post-warmup: normal CUDA graph operation
                if (properties_changed) {
                    // Properties changed - reset warmup, execute directly until stable again
                    graph->warmup_complete = false;
                    GGML_LOG_DEBUG("%s: CUDA graph warmup reset\n", __func__);
                } else {
                    use_cuda_graph = true;
                    // recapture a graph whose pool temporaries or scratch buffers were freed since its capture
                    // (GGML_CUDA_GRAPH_MEM_GEN=0 disables the check, A/B only)
                    static const bool mem_gen_check = !getenv("GGML_CUDA_GRAPH_MEM_GEN") || atoi(getenv("GGML_CUDA_GRAPH_MEM_GEN")) != 0;
                    cuda_graph_update_required = graph->instance == nullptr ||
                        (mem_gen_check && graph->mem_gen != ggml_cuda_graph_mem_gen(cuda_ctx->device));
                }
            }
            } // else: not prefill
        }
    }
#endif // USE_CUDA_GRAPH

    if (use_cuda_graph && cuda_graph_update_required) {
        // Start CUDA graph capture
        {
            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            ggml_cuda_lock_counter.fetch_add(1, std::memory_order_relaxed);
        }

        cuda_ctx->cuda_graph(graph_key)->mem_gen = ggml_cuda_graph_mem_gen(cuda_ctx->device);
        CUDA_CHECK(cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed));
    }

    if (t_gc) { g_gc_pre_us += ggml_time_us() - t_gc; }
    ggml_cuda_graph_evaluate_and_capture(cuda_ctx, cgraph, use_cuda_graph, cuda_graph_update_required, graph_key);
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
            // (a non-BF16 consumer, or any consumer in another split) -- 2 means the F32 output must
            // be kept (mark copy), 0 means nothing to do.
            auto classify_consumers = [&](const ggml_tensor * root) -> int {
                bool any = false;
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
                    if (t->op == GGML_OP_MUL_MAT_ID && t->src[1] && rootof(t->src[1]) == root &&
                            ggml_cuda_mmb_supported_mmid(t->src[0], t->src[1], t->src[2], t)) continue;
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
                return any ? 1 : 0;
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
            if (cgraph->nodes[i]->op == GGML_OP_CONCAT) {
                ggml_cuda_ple_conv_match pm;
                if (ggml_cuda_ple_conv_match_at_concat(cgraph, i, pm)) {
                    params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(pm.x), pm.out);
                    params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(pm.state), pm.out);
                    params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(pm.w), pm.out);
                    continue;
                }
                ggml_cuda_gdn_conv_match gm;
                if (ggml_cuda_gdn_conv_match_at_concat(cgraph, i, gm)) {
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

#ifdef USE_CUDA_GRAPH
    const ggml_cuda_graph_key graph_key = ggml_cuda_graph_get_key(cgraph);
    const bool use_cuda_graph = ggml_cuda_graph_set_enabled(cuda_ctx, graph_key);
#else
    const bool use_cuda_graph = false;
    GGML_UNUSED(cuda_ctx);
    GGML_UNUSED(cgraph);
#endif

    static bool enable_graph_optimization = [] {
        const char * env     = getenv("GGML_CUDA_GRAPH_OPT");
        return env != nullptr && atoi(env) == 1;
    }();

    if (!enable_graph_optimization) {
        return;
    }

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

static void * ggml_backend_cuda_stage_buffer(ggml_backend_t backend, int slot, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_cuda_set_device(cuda_ctx->device);
    return cuda_ctx->h2d_stage_buffer(slot, size);
}

static void ggml_backend_cuda_stage_upload(ggml_backend_t backend, void * dst, const void * data, size_t size, ggml_backend_event_t ev) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_cuda_set_device(cuda_ctx->device);
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
static bool ggml_backend_cuda_stage_gather(ggml_backend_t backend, int slot, const void * src, size_t offset, size_t width, size_t stride_src, size_t n_copies, ggml_backend_event_t ev, bool src_pinned) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_cuda_set_device(cuda_ctx->device);
    const size_t total = width * n_copies;                 // the device's compacted slice
    void * dst = cuda_ctx->h2d_stage_buffer(slot, total);
    if (dst == nullptr) {
        return false;
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

static void ggml_backend_cuda_stage_wait(ggml_backend_t backend, ggml_backend_event_t ev) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_cuda_set_device(cuda_ctx->device);
    CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->copy_stream(), (cudaEvent_t) ev->context, 0));
}

static void ggml_backend_cuda_stage_d2d(ggml_backend_t backend, void * dst, const void * src, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_cuda_set_device(cuda_ctx->device);
    CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, cuda_ctx->stream()));
}

static float ggml_backend_cuda_stage_h2d_gbps(ggml_backend_t backend) {
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
    if (h == nullptr || cudaMalloc(&d, sz) != cudaSuccess) {
        (void) cudaGetLastError(); // clear the sticky error
        if (d != nullptr) CUDA_CHECK(cudaFree(d));
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
    GGML_LOG_INFO("%s: H2D bandwidth calibration: %.1f GB/s (%zu MiB x3 in %.2f ms)\n", __func__, double(bw[dev]), sz >> 20, 1000.0*sec);
    CUDA_CHECK(cudaFree(d));
    free(h);
    return bw[dev];
}

static bool ggml_backend_cuda_moe_cache_update(ggml_backend_t backend, const ggml_tensor * weight, const ggml_tensor * weight_cpy, const int32_t * ids, int64_t n_used, int64_t n_tok, size_t ids_nb0, size_t ids_nb1, size_t slice_off, int split_axis) {
    ggml_backend_cuda_context * ctx = (ggml_backend_cuda_context *) backend->context;
    return moe_cache_update_host(weight, weight_cpy, ids, n_used, n_tok, ids_nb0, ids_nb1, ctx->stream(), ctx->device, slice_off, split_axis);
}

// Session 7 identity/device-remap fast path (see moe-expert-cache.h).  Called by the scheduler before
// it reads the routing ids back to the host: a true lets it skip the readback, the full device
// synchronize it forces, the used-expert pruning and the copy, because the consumer will read the
// compact arena with its own routing ids (identity) or a device-built remap (device-remap).
static bool ggml_backend_cuda_moe_cache_take_over(ggml_backend_t backend, const ggml_tensor * weight, const ggml_tensor * weight_cpy, bool * need_promote) {
    ggml_backend_cuda_context * ctx = (ggml_backend_cuda_context *) backend->context;
    return moe_cache_take_over(weight, weight_cpy, ctx->device, need_promote);
}

// Deferred promotion (device-remap tables), called by the scheduler once per token after the graph.
// `weight == nullptr` is the end-of-pass flush that runs the batched device-side admission policy.
static bool ggml_backend_cuda_moe_cache_promote(ggml_backend_t backend, const ggml_tensor * weight, const ggml_tensor * weight_cpy, const int32_t * ids, int64_t n_used, int64_t n_tok, size_t ids_nb0, size_t ids_nb1) {
    ggml_backend_cuda_context * ctx = (ggml_backend_cuda_context *) backend->context;
    if (weight == nullptr) {
        return moe_cache_policy_flush(ctx->device, ctx->stream());
    }
    return moe_cache_promote_host(weight, weight_cpy, ids, n_used, n_tok, ids_nb0, ids_nb1, ctx->stream(), ctx->device);
}

// B2: device-side host-weight expert gather for an offloaded `MUL_MAT_ID` prefill (see the iface comment).
static int64_t ggml_backend_cuda_moe_cache_band(ggml_backend_t backend) {
    const ggml_backend_cuda_context * cuda_ctx = (const ggml_backend_cuda_context *) backend->context;
    return moe_cache_max_tok_dev(cuda_ctx->device);
}

static bool ggml_backend_cuda_moe_cache_gather(ggml_backend_t backend, const ggml_tensor * weight, const ggml_tensor * weight_cpy, const ggml_tensor * ids, size_t slice_off, int split_axis) {
    ggml_backend_cuda_context * ctx = (ggml_backend_cuda_context *) backend->context;
    return moe_cache_gather_host(weight, weight_cpy, ids, ctx->stream(), ctx->device, slice_off, split_axis);
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
    /* .moe_cache_promote       = */ ggml_backend_cuda_moe_cache_promote,
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
    if (getenv("GGML_CUDA_REGISTER_HOST") == nullptr) {
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
    if (getenv("GGML_CUDA_REGISTER_HOST") == nullptr) {
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
    bool uma_env = getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") != nullptr;
    bool is_uma = prop.integrated > 0 || uma_env;

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

    bool host_buffer = getenv("GGML_CUDA_NO_PINNED") == nullptr;
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
                    // TODO: should become:
                    //return ggml_is_contiguous_rows(op->src[0]);
                    return ggml_is_contiguous(op->src[0]);
                default:
                    return false;
            }
            break;
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
                    return ggml_is_contiguous_1(op->src[0]);
                default:
                    return false;
            }
            break;
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
            } break;
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
            } break;
        case GGML_OP_GET_ROWS_BACK:
            {
                return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->ne[2] == 1 && op->ne[3] == 1;
            } break;
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
            } break;
        case GGML_OP_SET:
            {
                const ggml_type t = op->type;
                return (t == GGML_TYPE_F32 || t == GGML_TYPE_I32) &&
                    t == op->src[0]->type &&
                    t == op->src[1]->type;
            } break;
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
            } break;
        case GGML_OP_DUP:
                return true;
        case GGML_OP_ARGMAX:
        case GGML_OP_COUNT_EQUAL:
            {
                return true;
            } break;
        case GGML_OP_REPEAT:
            {
                // the CUDA REPEAT path only implements F32/F16; other types assert at runtime
                ggml_type src0_type = op->src[0]->type;
                return src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_F16;
            } break;
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
            } break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                return false;
            } break;
        case GGML_OP_COL2IM_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                return (src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_F16 || src0_type == GGML_TYPE_BF16) &&
                    op->type == src0_type &&
                    ggml_is_contiguous(op->src[0]) &&
                    ggml_is_contiguous(op);
            } break;
        case GGML_OP_SILU_BACK:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
            break;
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_L2_NORM:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_RMS_NORM_BACK:
            return ggml_is_contiguous(op->src[0]);
            break;
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
        case GGML_OP_CLAMP:
        case GGML_OP_LOG:
            return true;
        case GGML_OP_SCALE:
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_BF16) && op->type == op->src[0]->type;
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
            if(op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0])) {
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
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_TOP_K:
#if defined(GGML_USE_HIP) || defined(GGML_CUDA_USE_CUB)
            return true;
#else
            return op->src[0]->ne[0] <= 1024;
#endif // defined(GGML_USE_HIP) || defined(GGML_CUDA_USE_CUB)
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
            return true;
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

static bool ggml_backend_cuda_device_moe_cache_preflight(ggml_backend_dev_t dev, size_t host_expert_bytes, size_t aux_reserve_bytes) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *) dev->context;
    return moe_cache_preflight(ctx->device, host_expert_bytes, aux_reserve_bytes);
}

static void ggml_backend_cuda_device_moe_cache_set_reserve(ggml_backend_dev_t dev, size_t bytes) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *) dev->context;
    moe_cache_set_extra_reserve(ctx->device, bytes);
}

static bool ggml_backend_cuda_device_moe_cache_stats(ggml_backend_dev_t dev, int64_t * hits, int64_t * misses, int64_t * arena_bytes) {
    (void) dev;
    return moe_cache_get_stats(hits, misses, arena_bytes);
}

static bool ggml_backend_cuda_device_moe_cache_rearm(ggml_backend_dev_t dev) {
    // OPEN 2 (TODO #42): re-arm the expert-cache arena after a compute-buffer drop returned the VRAM.
    (void) dev;
    const bool rearmed = moe_cache_rearm();
    // Debug validator (no-op unless MOE_EXPERT_CACHE_VALIDATE is set); runs outside the cache lock.
    moe_cache_validate("after-rearm");
    return rearmed;
}

static size_t ggml_backend_cuda_device_slab_work_size(ggml_backend_dev_t dev) {
    // OPEN 2 (TODO #42): the movable-boundary slab WORK region's current size (the boundary), or 0 when this
    // device has no slab.  `llama_context` compares it against the narrow size it last settled on to decide
    // whether the compute reserve should follow the workload instead of holding the widest-graph reserve.
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *) dev->context;
    return ggml_cuda_slab_active(ctx->device) ? ggml_cuda_slab_work_size(ctx->device) : 0;
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

// --fit / memory-breakdown accounting for the FA prefill staging arena (block 15).  The arena is a
// raw device allocation outside the compute-graph reserve, so `llama_get_memory_breakdown` (and
// therefore --fit) cannot see it and can under-provision a deep prefill.  `bound` returns the
// worst-case per-launch staging request for the device: 0 when this device never prefill-stages
// (RDNA3_5 reads natively), SIZE_MAX when the cap is unbounded (the caller supplies an estimate),
// otherwise 2*cap because K and V are capped independently.  `used` is what the arena has grown to.
static size_t ggml_backend_cuda_fattn_stage_bound(ggml_backend_t backend) {
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

static size_t ggml_backend_cuda_fattn_stage_used(ggml_backend_t backend) {
    const ggml_backend_cuda_context * ctx = (const ggml_backend_cuda_context *) backend->context;
    return ctx != nullptr ? ctx->fattn_stage_bytes() : 0;
}

// --fit / memory-breakdown accounting for the op-offload H2D staging ring (block 06).  Like the FA
// arena it is a raw device allocation outside the compute-graph reserve, so
// `llama_get_memory_breakdown` (and therefore --fit) would not see it; the auto-sized budget can grow
// it to several GiB for a large host-resident expert table.  `used` is the live ring; `bound` is
// `slots x largest host table`, the worst case auto-sizing can request for this model.
static size_t ggml_backend_cuda_h2d_stage_used(ggml_backend_t backend) {
    const ggml_backend_cuda_context * ctx = (const ggml_backend_cuda_context *) backend->context;
    return ctx != nullptr ? ctx->h2d_stage_bytes() : 0;
}

static size_t ggml_backend_cuda_h2d_stage_bound(ggml_backend_t backend, size_t max_table) {
    const ggml_backend_cuda_context * ctx = (const ggml_backend_cuda_context *) backend->context;
    if (ctx == nullptr) {
        return 0;
    }
    return ggml_backend_cuda_context::h2d_stage_bound(max_table);
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
            const int min_batch_size = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;

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
