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

#include <cstdio>
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

// The adapter's SHARED (aperture) segments, all processes: where a pinned host allocation (`hipHostMalloc`,
// `cudaHostAlloc`, `cudaHostRegister`) lands on Windows.  They are NOT VRAM, and `ggml_hip_mem_get_info`
// deliberately ignores them, but they are the ceiling the host-expert pool actually hits: on issue #274's box
// the shared usage stops at ~9.3 GiB whether the machine has 47 or 64 GiB of RAM, far below the commit limit and
// far below `(RAM/2)/visible GPUs`.  Same statistics call as `wddm_free`, the other segment class.  Returns
// false (and leaves the outputs at -1) when the query is unavailable, so a caller can report "unknown" rather
// than a made-up number.
// The HIP runtime module that is ACTUALLY loaded, on Windows.  `amdhip64_7.dll` is normally resolved from
// the directory of the executable (the ROCm/TheRock tree dropped beside llama-server), but Windows will
// silently load the Adrenalin driver's copy from `C:\Windows\System32` instead when that directory has no
// copy.  The two are NOT interchangeable for the pinned host tier: the driver-shipped runtime (measured
// 10.0.3661.0 on issue #278's box) refuses a pinned allocation at ~9.3 GiB of adapter shared (aperture)
// usage even on a 64 GiB machine, latches the pinned tier off, and on that box spins two threads inside its
// own yield-wait loop after the post-prefill re-arm; the same binary with the TheRock ROCm 10.1 runtime
// (`amdhip64_7.dll` 10.0.3684.0) beside it pins 16-40 GiB and runs the same arm clean.  So the loaded module
// is the thing to report -- a HIP version alone cannot tell them apart.
//
// Fills `path_out` with the module file name, `system_dir_out` with true when it came from the Windows
// system directory (the driver copy), and `version_out` with `hipRuntimeGetVersion()` in the HIP_VERSION
// encoding (major*1e7 + minor*1e5 + patch).  Returns false when no HIP runtime module can be resolved;
// every output is then left at its neutral value.
bool ggml_hip_runtime_module_info(char * path_out, size_t path_cap, bool * system_dir_out, int * version_out) {
    if (path_out != nullptr && path_cap > 0) { path_out[0] = '\0'; }
    if (system_dir_out != nullptr) { *system_dir_out = false; }
    if (version_out    != nullptr) { *version_out    = 0; }

    static const char * const names[] = { "amdhip64_7.dll", "amdhip64.dll", "libamdhip64.dll" };
    HMODULE mod = nullptr;
    for (const char * name : names) {
        mod = GetModuleHandleA(name);
        if (mod != nullptr) {
            break;
        }
    }
    if (mod == nullptr) {
        return false;
    }

    char path[MAX_PATH] = {};
    const DWORD n = GetModuleFileNameA(mod, path, (DWORD) sizeof(path));
    if (n == 0 || n >= sizeof(path)) {
        return false;
    }
    if (path_out != nullptr && path_cap > 0) {
        snprintf(path_out, path_cap, "%s", path);
    }

    if (system_dir_out != nullptr) {
        char sys[MAX_PATH] = {};
        const UINT sd = GetSystemDirectoryA(sys, (UINT) sizeof(sys));
        if (sd > 0 && sd < sizeof(sys)) {
            const size_t sl = strlen(sys);
            // `GetSystemDirectoryA` gives `...\System32`; a case-insensitive prefix plus a separator is the
            // whole test (no other Windows directory is the driver's runtime location).  ASCII-only fold, so
            // this does not depend on the CRT's `_strnicmp`.
            bool prefix = sl > 0;
            for (size_t i = 0; i < sl && prefix; ++i) {
                char a = path[i];
                char b = sys[i];
                if (a >= 'A' && a <= 'Z') { a = (char) (a - 'A' + 'a'); }
                if (b >= 'A' && b <= 'Z') { b = (char) (b - 'A' + 'a'); }
                prefix = a == b;
            }
            *system_dir_out = prefix && (path[sl] == '\\' || path[sl] == '/');
        }
    }

    int ver = 0;
    if (hipRuntimeGetVersion(&ver) == hipSuccess && version_out != nullptr) {
        *version_out = ver;
    } else {
        (void) hipGetLastError();
    }
    return true;
}

bool ggml_hip_win_shared_mem_info(int64_t * limit_mib, int64_t * resident_mib) {
    if (limit_mib    != nullptr) { *limit_mib    = -1; }
    if (resident_mib != nullptr) { *resident_mib = -1; }
    const wddm_query_fn query = wddm_query();
    int device = 0;
    if (query == nullptr || hipGetDevice(&device) != hipSuccess) {
        (void) hipGetLastError();
        return false;
    }
    if (device < 0 || device >= WDDM_MAX_DEVICES) {
        return false;
    }
    LUID luid;
    if (!wddm_luid(device, luid)) {
        return false;
    }
    D3DKMT_QUERYSTATISTICS q = {};
    q.Type        = D3DKMT_QUERYSTATISTICS_ADAPTER;
    q.AdapterLuid = luid;
    if (query(&q) != 0) {
        return false;
    }
    unsigned long long limit = 0;
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
        if (!seg.Aperture) {
            continue;   // the local VRAM segments are `wddm_free`'s half
        }
        limit    += seg.CommitLimit;
        resident += seg.BytesResident;
    }
    if (limit == 0) {
        return false;
    }
    if (limit_mib    != nullptr) { *limit_mib    = (int64_t) (limit >> 20); }
    if (resident_mib != nullptr) { *resident_mib = (int64_t) (resident >> 20); }
    return true;
}

#endif // defined(GGML_USE_HIP) && defined(_WIN32)
