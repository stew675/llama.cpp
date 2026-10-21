// Free device memory on Windows (WDDM).
//
// hipMemGetInfo counts only this process's allocations, so when other processes hold memory on the card it reports more
// free memory than WDDM can give.  WDDM does not refuse an allocation past that point: it moves memory to system RAM
// ("shared GPU memory"), which the GPU then reads over PCIe -- slow, and on a large prefill it can end in a device loss
// (issue #216).  The adapter's own statistics count every process: the local (non-aperture) segments' commit limit minus
// their resident bytes.  cudaMemGetInfo (vendors/hip.h) returns the smaller of the two free figures; the total is HIP's.
// GGML_CUDA_WDDM_FREE=0 returns hipMemGetInfo's figures unchanged.
//
// The adapter is found by the LUID HIP reports; D3DKMTQueryStatistics is resolved from gdi32 at run time, so nothing new
// is linked.  The approach follows the WDDM free-memory query of Avalon (github.com/freriadoc/Avalon, MIT).

#include "common.cuh"

#if defined(GGML_USE_HIP) && defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winternl.h>   // NTSTATUS, for d3dkmthk.h
#include <d3dkmthk.h>

#include <cstdlib>
#include <cstring>
#include <mutex>

namespace {

using wddm_query_fn = NTSTATUS (APIENTRY *)(const D3DKMT_QUERYSTATISTICS *);

constexpr int WDDM_MAX_DEVICES = 64;

wddm_query_fn wddm_query() {
    static const wddm_query_fn fn = [] {
        HMODULE gdi = LoadLibraryA("gdi32.dll");
        return gdi ? (wddm_query_fn) GetProcAddress(gdi, "D3DKMTQueryStatistics") : nullptr;
    }();
    return fn;
}

bool wddm_enabled() {
    static const bool on = [] {
        const char * v = getenv("GGML_CUDA_WDDM_FREE");
        return v == nullptr || atoi(v) != 0;
    }();
    return on;
}

// the adapter LUID of each HIP device, read once (hipGetDeviceProperties is slow)
struct wddm_device {
    std::once_flag luid_once;
    bool           luid_ok = false;
    LUID           luid    = {};
    std::once_flag log_once;
};

wddm_device g_wddm_device[WDDM_MAX_DEVICES];

bool wddm_luid(int device, LUID & out) {
    wddm_device & d = g_wddm_device[device];
    std::call_once(d.luid_once, [&] {
        hipDeviceProp_t prop;
        if (hipGetDeviceProperties(&prop, device) != hipSuccess) {
            (void) hipGetLastError();
            return;
        }
        static_assert(sizeof(d.luid) == sizeof(prop.luid), "LUID size");
        memcpy(&d.luid, prop.luid, sizeof(d.luid));
        d.luid_ok = d.luid.LowPart != 0 || d.luid.HighPart != 0;
    });
    out = d.luid;
    return d.luid_ok;
}

// every process's free memory in the adapter's local segments
bool wddm_free(int device, size_t & free_out) {
    const wddm_query_fn query = wddm_query();
    LUID luid;
    if (query == nullptr || !wddm_luid(device, luid)) {
        return false;
    }
    D3DKMT_QUERYSTATISTICS q = {};
    q.Type        = D3DKMT_QUERYSTATISTICS_ADAPTER;
    q.AdapterLuid = luid;
    if (query(&q) != 0) {
        return false;
    }
    unsigned long long limit    = 0;
    unsigned long long resident = 0;
    for (ULONG i = 0; i < q.QueryResult.AdapterInformation.NbSegments; ++i) {
        D3DKMT_QUERYSTATISTICS s = {};
        s.Type                   = D3DKMT_QUERYSTATISTICS_SEGMENT;
        s.AdapterLuid            = luid;
        s.QuerySegment.SegmentId = i;
        if (query(&s) != 0) {
            return false;
        }
        const D3DKMT_QUERYSTATISTICS_SEGMENT_INFORMATION & seg = s.QueryResult.SegmentInformation;
        if (seg.Aperture) {
            continue;   // system memory seen through the GPU's aperture
        }
        limit    += seg.CommitLimit;
        resident += seg.BytesResident;
    }
    if (limit == 0) {
        return false;
    }
    free_out = resident < limit ? (size_t) (limit - resident) : 0;
    return true;
}

} // namespace

hipError_t ggml_hip_mem_get_info(size_t * free, size_t * total) {
    const hipError_t err = hipMemGetInfo(free, total);
    if (err != hipSuccess || !wddm_enabled()) {
        return err;
    }
    int device = 0;
    if (hipGetDevice(&device) != hipSuccess) {
        (void) hipGetLastError();
        return err;
    }
    if (device < 0 || device >= WDDM_MAX_DEVICES) {
        return err;
    }
    size_t free_wddm = 0;
    if (!wddm_free(device, free_wddm)) {
        return err;
    }
    std::call_once(g_wddm_device[device].log_once, [&] {
        GGML_LOG_INFO("%s: device %d: free memory %zu MiB by WDDM, %zu MiB by HIP (the smaller is used; "
                      "GGML_CUDA_WDDM_FREE=0 keeps HIP's)\n", __func__, device, free_wddm >> 20, *free >> 20);
    });
    if (free_wddm < *free) {
        *free = free_wddm;
    }
    return err;
}

// Available system RAM on Windows: the counterpart of Linux's MemAvailable (free plus standby pages), which the
// host expert pool (its fill mode and its RAM ceiling) and the windowed KV spill ceiling read from /proc/meminfo.
// Without it they took Windows for a memory-tight box: the pool chose O_DIRECT fills (no MemAvailable) and read
// every miss from disk.  -1 when the call fails, as on a platform without the figure.
int64_t ggml_hip_win_available_ram_bytes() {
    MEMORYSTATUSEX st = {};
    st.dwLength = sizeof(st);
    if (!GlobalMemoryStatusEx(&st) || st.ullAvailPhys == 0) {
        return -1;
    }
    return (int64_t) st.ullAvailPhys;
}

#endif // defined(GGML_USE_HIP) && defined(_WIN32)
