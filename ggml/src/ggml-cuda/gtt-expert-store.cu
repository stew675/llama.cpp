// wip/apu-expert-gtt P2: the GTT expert store's region.  See `gtt-expert-store.cuh` for the design and the
// measurements it rests on; the buffer type that carves tensors out of this region lives in `ggml-cuda.cu`.
//
// This file owns exactly three things:
//   1. the region's lifetime -- `mmap` + `MADV_HUGEPAGE` + ONE `cudaHostRegister(Mapped[|ReadOnly])`, and at
//      teardown `cudaHostUnregister` THEN `munmap` (probe 09: the registration is a hold that outlives the
//      unregister while the VMA lives, so both are required to actually release the pages);
//   2. the bump allocator the buffer type carves from;
//   3. the census.
//
// It deliberately contains no policy: `-lme` -> bytes, the heat ranking and which tensors are resident are
// the model loader's (`plan_host_expert_placement`), so the discrete path and this one share one decision.

#include "gtt-expert-store.cuh"

#include "common.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <vector>

#if !defined(_WIN32)
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#include <cerrno>
#endif

struct gtt_block_t {
    size_t off;
    size_t len;
};

// One GTT store per device.  A `-sm tensor` split over several APUs would open one per device; gfx1151 is one
// device here, and the plan only ever asks the device it found.
struct gtt_store_t {
    int      device    = -1;
    bool     opened    = false;   // a budget was accepted (the region may not exist yet: `no_alloc`/`--fit`)
    size_t   extent    = 0;       // the planned extent (the budget), page rounded down
    void *   base      = nullptr; // the region: host VA == device VA (asserted)
    size_t   created   = 0;       // the extent actually mapped and registered
    bool     ro        = false;   // registered ReadOnly
    size_t   used      = 0;       // bytes currently reserved (sum of live allocations)
    // The free list: sorted, coalesced byte ranges of the region nobody holds.  Created as one range when the
    // region is made; a reserve splits it, a release merges back.  The expert cache makes ~n_tables allocations
    // of 100-450 MiB each, so a first-fit linear scan over a handful of ranges is the whole data structure.
    std::vector<gtt_block_t> free_ranges;
    size_t   carve_min = 0;
    size_t   carve_max = 0;
    double   create_ms = 0.0;     // mmap + MADV_HUGEPAGE + the ONE registration + the mlock
    bool     want_lock = false;   // MOE_EXPERT_GTT_MLOCK
    size_t   locked    = 0;       // bytes mlock() actually took (0 = reclaimable tier)
    int      n_carves  = 0;
    size_t   fill      = 0;       // bytes the loader read from the GGUF into the GTT store
    int      n_tables  = 0;
    int64_t  n_slots   = 0;
    char     source[192] = {0};   // the budget's source, verbatim from the plan
};

static gtt_store_t      g_stores[GGML_CUDA_MAX_DEVICES];
static std::mutex       g_store_mutex;

// ---------------------------------------------------------------------------------------------------------
// environment
// ---------------------------------------------------------------------------------------------------------

bool ggml_cuda_gtt_store_wanted() {
    static const bool on = [] {
        const char * e = getenv("MOE_EXPERT_GTT");
        return e == nullptr || atoi(e) != 0;
    }();
    return on;
}

bool ggml_cuda_gtt_store_ro() {
    static const bool ro = [] {
        const char * e = getenv("MOE_EXPERT_GTT_RO");
        return e == nullptr || atoi(e) != 0;
    }();
    return ro;
}

bool ggml_cuda_gtt_store_wants_lock() {
    static const bool on = [] {
        const char * e = getenv("MOE_EXPERT_GTT_MLOCK");
        return e == nullptr || atoi(e) != 0;
    }();
    return on;
}

// `RLIMIT_MEMLOCK` in bytes, or SIZE_MAX when unlimited (and SIZE_MAX too when locking is disabled, so the
// caller's policy is exactly what it was before locking existed).
size_t ggml_cuda_gtt_store_lockable() {
    if (!ggml_cuda_gtt_store_wants_lock()) {
        return SIZE_MAX;
    }
#if defined(_WIN32)
    return SIZE_MAX;   // no RLIMIT_MEMLOCK; the GTT store declines on Windows anyway
#else
    struct rlimit rl;
    if (getrlimit(RLIMIT_MEMLOCK, &rl) != 0) {
        return SIZE_MAX;
    }
    if (rl.rlim_cur == RLIM_INFINITY) {
        return SIZE_MAX;
    }
    return (size_t) rl.rlim_cur;
#endif
}

// ---------------------------------------------------------------------------------------------------------
// small /proc helpers (the census and the budget assertion)
// ---------------------------------------------------------------------------------------------------------

static size_t gtt_meminfo_kib(const char * key) {
#if defined(_WIN32)
    GGML_UNUSED(key);
    return 0;
#else
    FILE * f = fopen("/proc/meminfo", "r");
    if (f == nullptr) {
        return 0;
    }
    char line[256];
    const size_t klen = strlen(key);
    size_t out = 0;
    while (fgets(line, sizeof(line), f) != nullptr) {
        if (strncmp(line, key, klen) == 0 && line[klen] == ':') {
            out = (size_t) strtoull(line + klen + 1, nullptr, 10);
            break;
        }
    }
    fclose(f);
    return out;
#endif
}

// AnonHugePages + Rss of the VMA that starts at `base`.  The GTT store is ONE `mmap`, so one header line matches.
static void gtt_smaps_of(const void * base, size_t * huge_kib, size_t * rss_kib) {
    *huge_kib = 0;
    *rss_kib  = 0;
#if defined(_WIN32)
    GGML_UNUSED(base);
#else
    FILE * f = fopen("/proc/self/smaps", "r");
    if (f == nullptr || base == nullptr) {
        if (f != nullptr) {
            fclose(f);
        }
        return;
    }
    char want[32];
    snprintf(want, sizeof(want), "%lx", (unsigned long) (uintptr_t) base);
    char line[512];
    bool in = false;
    while (fgets(line, sizeof(line), f) != nullptr) {
        // a header line looks like `7f...-7f... rw-p 00000000 00:00 0 ...`
        if (strchr(line, '-') != nullptr && (line[0] >= '0' && line[0] <= '9') &&
                strncmp(line, want, strlen(want)) == 0 && line[strlen(want)] == '-') {
            in = true;
            continue;
        }
        if (line[0] >= '0' && line[0] <= '9' && strchr(line, '-') != nullptr) {
            in = false;   // the next mapping
            continue;
        }
        if (!in) {
            continue;
        }
        if (strncmp(line, "AnonHugePages:", 14) == 0) {
            *huge_kib = (size_t) strtoull(line + 14, nullptr, 10);
        } else if (strncmp(line, "Rss:", 4) == 0) {
            *rss_kib = (size_t) strtoull(line + 4, nullptr, 10);
        }
    }
    fclose(f);
#endif
}

// ---------------------------------------------------------------------------------------------------------
// lifetime
// ---------------------------------------------------------------------------------------------------------

static void gtt_store_release_locked(gtt_store_t & s) {
#if !defined(_WIN32)
    if (s.base != nullptr) {
        // unregister FIRST, then unmap: probe 09 measured the hold surviving `hipHostUnregister` while the
        // VMA lives, so releasing in the other order would leave the pages pinned with no way back.
        const cudaError_t err = cudaHostUnregister(s.base);
        if (err != cudaSuccess) {
            (void) cudaGetLastError();
            GGML_LOG_WARN("%s: device %d: cudaHostUnregister of the %.2f GiB GTT expert store failed: %s\n", __func__,
                    s.device, (double) s.created / 1073741824.0, cudaGetErrorString(err));
        }
        if (s.locked != 0) {
            (void) munlock(s.base, s.created);
        }
        if (munmap(s.base, s.created) != 0) {
            GGML_LOG_WARN("%s: device %d: munmap of the GTT expert store failed: %s\n", __func__, s.device,
                    strerror(errno));
        }
    }
#endif
    s.base    = nullptr;
    s.created = 0;
    s.used    = 0;
    s.locked  = 0;
    s.free_ranges.clear();
    s.opened  = false;
}

void ggml_cuda_gtt_store_free() {
    std::lock_guard<std::mutex> lock(g_store_mutex);
    for (auto & s : g_stores) {
        if (s.opened || s.base != nullptr) {
            gtt_store_release_locked(s);
        }
    }
}

// Create the region: `mmap` + `MADV_HUGEPAGE` + ONE registration.  Registration of an UNTOUCHED region is
// 30.5 ms/GiB (probe 11) and the kernel forms the huge pages there; the loader's fill then writes straight
// into them with no page faults at all, which is why this is not pre-faulted with a memset first.
static bool gtt_store_create_locked(gtt_store_t & s, char * err, size_t errn) {
#if defined(_WIN32)
    // POSIX-only shape: `mmap`/`MADV_HUGEPAGE` residency plus one flat address.  WDDM gives a distinct device
    // alias for host memory, and this design refuses to run on two addresses (see the header).  Decline.
    snprintf(err, errn, "the GTT expert store is a POSIX/mmap shape and needs one address for host and device");
    GGML_UNUSED(s);
    return false;
#else
    struct timespec t0 = {0, 0}, t1 = {0, 0};
    clock_gettime(CLOCK_MONOTONIC, &t0);

    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    size_t extent = s.extent & ~(page - 1);
    if (extent < page) {
        snprintf(err, errn, "the %.2f MiB budget rounds to nothing at this page size", (double) s.extent / 1048576.0);
        return false;
    }

    void * p = mmap(nullptr, extent, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) {
        snprintf(err, errn, "mmap of %.2f MiB failed: %s", (double) extent / 1048576.0, strerror(errno));
        return false;
    }
    // Best-effort: without THP the GTT store is still correct, just a 4 KiB tier (and the read rate is the thing
    // the census reports, so a machine that refused THP is visible rather than silent).
    if (madvise(p, extent, MADV_HUGEPAGE) != 0) {
        GGML_LOG_WARN("%s: device %d: MADV_HUGEPAGE on the %.2f GiB GTT expert store failed: %s (the GTT store will be "
                      "4 KiB pages; the census reports the realised granularity)\n", __func__, s.device,
                      (double) extent / 1073741824.0, strerror(errno));
    }

    const unsigned int flags = cudaHostRegisterMapped | (s.ro ? cudaHostRegisterReadOnly : 0u);
    cudaError_t reg = cudaHostRegister(p, extent, flags);
    if (reg != cudaSuccess) {
        const char * msg = cudaGetErrorString(reg);
        (void) cudaGetLastError();
        // A driver that refuses the ReadOnly arm is not a reason to lose the GTT store: fall back to the writable
        // registration ONCE and say so (the census then reports `writable`, which is the 60 GB/s arm).
        if (s.ro) {
            GGML_LOG_WARN("%s: device %d: cudaHostRegister(Mapped|ReadOnly) of the %.2f GiB GTT expert store was "
                          "refused (%s); registering it WRITABLE instead -- expect the writable read rate\n",
                          __func__, s.device, (double) extent / 1073741824.0, msg);
            reg = cudaHostRegister(p, extent, cudaHostRegisterMapped);
            if (reg == cudaSuccess) {
                s.ro = false;
            }
        }
        if (reg != cudaSuccess) {
            msg = cudaGetErrorString(reg);
            (void) cudaGetLastError();
            (void) munmap(p, extent);
            snprintf(err, errn, "cudaHostRegister of %.2f MiB failed: %s", (double) extent / 1048576.0, msg);
            return false;
        }
    }

    // ONE address, or nothing.  With `iommu=off` the runtime returns the host pointer unchanged (measured);
    // if it ever returns a second address, the loader's fill (`read_raw` into `tensor->data`) and the
    // kernel's read would disagree, and there is no fault-and-retry on this part to catch it.
    void * d = p;
    if (cudaHostGetDevicePointer(&d, p, 0) != cudaSuccess) {
        (void) cudaGetLastError();
        d = p;
    }
    if (d != p) {
        (void) cudaHostUnregister(p);
        (void) munmap(p, extent);
        snprintf(err, errn, "the runtime gave the registered region a SECOND address (host %p, device %p); the "
                 "GTT store needs one (iommu=off)", p, d);
        return false;
    }

    // `mlock` is what makes the tier a RESERVATION, and it is the only thing that makes it visible to the
    // memory accounting (probe 16: `Unevictable`/`Mlocked` report 0 for a merely-registered region, which is
    // reclaimable all the same).  The pages are already resident -- the registration faulted them in -- so
    // this is a bookkeeping pass over the VMA, not a second walk of DRAM.  The model loader has already
    // clamped the extent to `RLIMIT_MEMLOCK`, so a failure here means something else took the limit between
    // the check and this call: lock what we can, keep going, and let the census say the true number.
    s.locked = 0;
    if (s.want_lock) {
        if (mlock(p, extent) == 0) {
            s.locked = extent;
        } else {
            // Partial locking is possible (mlock fails on the page it cannot charge), so report the truth
            // rather than a boolean: the tier is then partly a reservation and partly reclaimable.
            GGML_LOG_WARN("%s: device %d: mlock of the %.2f GiB GTT store failed: %s -- the tier is "
                          "RECLAIMABLE and will stall under memory pressure (raise RLIMIT_MEMLOCK; "
                          "`ulimit -l` shows it)\n", __func__, s.device, (double) extent / 1073741824.0,
                          strerror(errno));
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    s.create_ms = (double) (t1.tv_sec - t0.tv_sec) * 1000.0 + (double) (t1.tv_nsec - t0.tv_nsec) / 1.0e6;

    s.base    = p;
    s.created = extent;
    s.used    = 0;
    s.free_ranges.clear();
    s.free_ranges.push_back(gtt_block_t{0, extent});
    static std::once_flag once_exit;
    std::call_once(once_exit, [] {
        // The registration is a HOLD (probe 09) and the hold outlives `cudaHostUnregister` while the VMA
        // lives, so release both explicitly rather than relying on process teardown order.
        atexit(ggml_cuda_gtt_store_free);
    });
    return true;
#endif
}

bool ggml_cuda_gtt_store_open(int device, size_t extent_bytes, const char * source, char * err, size_t errn) {
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) {
        snprintf(err, errn, "device %d is out of range", device);
        return false;
    }
    std::lock_guard<std::mutex> lock(g_store_mutex);
    gtt_store_t & s = g_stores[device];
    if (s.opened && s.base != nullptr) {
        // Idempotent for the same request (a second `llama_model` in one process, e.g. llama-bench reloading
        // because its params differ).  A DIFFERENT extent under a live region cannot be honoured: the carved
        // tensors are already placed.
        if (s.extent == extent_bytes) {
            snprintf(s.source, sizeof(s.source), "%s", source != nullptr ? source : "");
            return true;
        }
        // A SECOND load may ADOPT the live region for a smaller request that fits in what is still free.  This
        // is the separate-MTP-drafter shape: the target's store is sized and carved first, and the drafter's
        // head (one nextn block, 3 expert tables) is placed in the same region on top of it.  The caller has
        // already capped its budget by `ggml_cuda_gtt_store_free_bytes()`, so this is a share, not a grow: the
        // region, its extent, its registration and its source string all stay the first model's.
        if (extent_bytes < s.extent) {
            size_t best = 0;
            for (const auto & r : s.free_ranges) {
                best = r.len > best ? r.len : best;
            }
            if (best >= extent_bytes) {
                GGML_LOG_WARN("%s: device %d: this load adopts the live %.2f MiB GTT store, placing %.2f MiB "
                              "of expert tables inside it (%.2f MiB still free; the region, its extent and its "
                              "registration stay the first model's)\n", __func__, device,
                              (double) s.extent / 1048576.0, (double) extent_bytes / 1048576.0,
                              (double) best / 1048576.0);
                return true;
            }
            snprintf(err, errn, "a GTT store of %.2f MiB is already live on device %d and only %.2f MiB of it is "
                     "free; this load asks for %.2f MiB", (double) s.extent / 1048576.0, device,
                     (double) best / 1048576.0, (double) extent_bytes / 1048576.0);
            return false;
        }
        snprintf(err, errn, "a GTT store of %.2f MiB is already live on device %d; %.2f MiB cannot replace it in "
                 "place", (double) s.extent / 1048576.0, device, (double) extent_bytes / 1048576.0);
        return false;
    }
    s.device  = device;
    s.opened  = true;
    s.extent  = extent_bytes;
    s.ro      = ggml_cuda_gtt_store_ro();
    s.want_lock = ggml_cuda_gtt_store_wants_lock();
    s.locked  = 0;
    s.fill    = 0;
    s.n_tables = 0;
    s.n_slots  = 0;
    s.carve_min = 0;
    s.carve_max = 0;
    s.n_carves  = 0;
    s.free_ranges.clear();
    snprintf(s.source, sizeof(s.source), "%s", source != nullptr ? source : "");
    // The region is created HERE, before a single expert tensor is placed, so a refusal is a clean decline
    // (the loader keeps today's placement) instead of a half-placed model.  A `--fit` / `--no-alloc` probe
    // load never reaches this point -- the loader skips the GTT store plan entirely there -- so nothing reserves
    // non-swappable RAM for a load that allocates no weights.
    if (!gtt_store_create_locked(s, err, errn)) {
        s.opened = false;
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------------------------------------
// the bump allocator
// ---------------------------------------------------------------------------------------------------------

// First fit over the free list.  `align` is a page: a GGUF expert slab is 0.45-0.92 MiB, so the padding costs
// well under 1 %.  Caller holds the lock.
static void * gtt_store_reserve_locked(gtt_store_t & s, size_t size, size_t align) {
    if (size == 0 || s.base == nullptr) {
        return nullptr;
    }
    for (size_t i = 0; i < s.free_ranges.size(); i++) {
        const gtt_block_t r = s.free_ranges[i];
        const size_t off = (r.off + align - 1) & ~(align - 1);
        const size_t pad = off - r.off;
        if (r.len < pad || size > r.len - pad) {
            continue;
        }
        // Take [off, off + size): the head (if any) stays free, the tail (if any) stays free.
        gtt_block_t head{r.off, pad};
        gtt_block_t tail{off + size, r.len - pad - size};
        s.free_ranges.erase(s.free_ranges.begin() + (long) i);
        if (tail.len > 0) {
            s.free_ranges.insert(s.free_ranges.begin() + (long) i, tail);
        }
        if (head.len > 0) {
            s.free_ranges.insert(s.free_ranges.begin() + (long) i, head);
        }
        s.used += size;
        s.carve_min = s.carve_min == 0 ? size : (size < s.carve_min ? size : s.carve_min);
        s.carve_max = size > s.carve_max ? size : s.carve_max;
        s.n_carves += 1;
        return (char *) s.base + off;
    }
    return nullptr;
}

void * ggml_cuda_gtt_store_reserve(int device, size_t size) {
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES || size == 0) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_store_mutex);
    gtt_store_t & s = g_stores[device];
    if (!s.opened || s.base == nullptr) {
        return nullptr;
    }
    void * p = gtt_store_reserve_locked(s, size, 4096);
    if (p == nullptr && s.base != nullptr) {
        size_t best = 0;   // NOT `ggml_cuda_gtt_store_largest_free()`: that takes this same lock
        for (const auto & r : s.free_ranges) {
            best = r.len > best ? r.len : best;
        }
        GGML_LOG_ERROR("%s: device %d: a %.2f MiB reserve does not fit the %.2f MiB GTT store (%.2f MiB held, "
                       "largest free run %.2f MiB)\n", __func__, device, (double) size / 1048576.0,
                       (double) s.created / 1048576.0, (double) s.used / 1048576.0, (double) best / 1048576.0);
    }
    return p;
}

void ggml_cuda_gtt_store_release(int device, void * p, size_t size) {
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES || p == nullptr || size == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_store_mutex);
    gtt_store_t & s = g_stores[device];
    if (!s.opened || s.base == nullptr) {
        return;
    }
    const size_t off = (size_t) ((char *) p - (char *) s.base);
    if (off >= s.created || size > s.created - off) {
        GGML_LOG_ERROR("%s: device %d: release of %p + %.2f MiB is outside the GTT store\n", __func__, device,
                       p, (double) size / 1048576.0);
        return;
    }
    // Insert in offset order and coalesce with both neighbours (the list is sorted and disjoint by invariant).
    size_t i = 0;
    while (i < s.free_ranges.size() && s.free_ranges[i].off < off) {
        i++;
    }
    // a double release would corrupt the list: refuse an overlap instead of coalescing garbage.
    const bool overlaps = (i < s.free_ranges.size() && off + size > s.free_ranges[i].off) ||
                          (i > 0 && s.free_ranges[i - 1].off + s.free_ranges[i - 1].len > off);
    if (overlaps) {
        GGML_LOG_ERROR("%s: device %d: release of %p + %.2f MiB overlaps a free range (double release?)\n",
                       __func__, device, p, (double) size / 1048576.0);
        return;
    }
    s.free_ranges.insert(s.free_ranges.begin() + (long) i, gtt_block_t{off, size});
    s.used = s.used >= size ? s.used - size : 0;
    i = 0;
    while (i + 1 < s.free_ranges.size()) {
        if (s.free_ranges[i].off + s.free_ranges[i].len == s.free_ranges[i + 1].off) {
            s.free_ranges[i].len += s.free_ranges[i + 1].len;
            s.free_ranges.erase(s.free_ranges.begin() + (long) (i + 1));
        } else {
            i++;
        }
    }
}

// How many bytes a SECOND load may still place in the live GTT store: SIZE_MAX when no store is open on this
// device (the caller sizes its own), otherwise the largest free run.  A second model in one process (a separate
// MTP drafter's head, or llama-bench reloading the same model) must not size or grow the region: its extent is
// the FIRST model's budget, and the arena tier reserves its slots out of the same free list.
size_t ggml_cuda_gtt_store_free_bytes(int device) {
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) {
        return SIZE_MAX;
    }
    std::lock_guard<std::mutex> lock(g_store_mutex);
    const gtt_store_t & s = g_stores[device];
    if (!s.opened || s.base == nullptr) {
        return SIZE_MAX;
    }
    size_t best = 0;
    for (const auto & r : s.free_ranges) {
        best = r.len > best ? r.len : best;
    }
    return best;
}

size_t ggml_cuda_gtt_store_largest_free(int device) {
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(g_store_mutex);
    const gtt_store_t & s = g_stores[device];
    size_t best = 0;
    for (const auto & r : s.free_ranges) {
        best = r.len > best ? r.len : best;
    }
    return best;
}

void * ggml_cuda_gtt_store_carve(int device, size_t size) {
    // The ALL-RESIDENT placement arm: one allocation per load, never released.
    return ggml_cuda_gtt_store_reserve(device, size);
}

// ---------------------------------------------------------------------------------------------------------
// the census
// ---------------------------------------------------------------------------------------------------------

// One line, at INFO (so it is read at `-lv 4`, like every other census) -- and it says what this session does
// NOT do: the GTT store is an ALL-RESIDENT tier here.  Eviction, the warm copy-engine arm and prefill streaming
// are P3/P4/P5, so a run whose expert set does not fit the budget is a *hot subset* whose non-resident
// experts stay on their layer's device, not a streaming tier.
void ggml_cuda_gtt_store_report(int device, size_t filled_bytes, int n_tables, int64_t n_slots) {
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) {
        return;
    }
    size_t extent = 0, created = 0, used = 0, carve_min = 0, carve_max = 0;
    int    n_carves = 0;
    double create_ms = 0.0;
    bool   locked_any = false;
    size_t locked = 0;
    size_t lockable = ggml_cuda_gtt_store_lockable();
    bool ro = false;
    char source[192] = {0};
    void * base = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_store_mutex);
        gtt_store_t & s = g_stores[device];
        if (!s.opened || s.base == nullptr) {
            return;
        }
        extent    = s.extent;
        created   = s.created;
        used      = s.used;
        carve_min = s.carve_min;
        carve_max = s.carve_max;
        n_carves  = s.n_carves;
        create_ms = s.create_ms;
        locked     = s.locked;
        locked_any = s.want_lock;
        ro        = s.ro;
        base      = s.base;
        snprintf(source, sizeof(source), "%s", s.source);
        s.fill     = filled_bytes;
        s.n_tables = n_tables;
        s.n_slots  = n_slots;
    }

    const double mib = 1024.0 * 1024.0;
    size_t huge_kib = 0;
    size_t rss_kib  = 0;
    gtt_smaps_of(base, &huge_kib, &rss_kib);

    // The budget assertion (P1.2): the GTT store is registered, non-swappable host memory, so it must stay under
    // the machine's own memory -- the measured over-commit this replaces (2026-10-19: a 123 GiB box went to
    // 121 GiB resident + 13 GiB swap) is exactly what an unchecked GTT store would repeat.
    const size_t avail_kib = gtt_meminfo_kib("MemAvailable");
    const size_t total_kib = gtt_meminfo_kib("MemTotal");
    const double thp = created > 0 ? 100.0 * (double) huge_kib / ((double) created / 1024.0) : 0.0;
    // The reservation's own line: what we asked to lock, what we got, and the limit that decides it.
    char lim[32];
    if (lockable == SIZE_MAX) {
        snprintf(lim, sizeof(lim), "unlimited");
    } else {
        snprintf(lim, sizeof(lim), "%.1f MiB", (double) lockable / mib);
    }
    char lock_note[224];
    if (!locked_any) {
        snprintf(lock_note, sizeof(lock_note), "NOT locked (MOE_EXPERT_GTT_MLOCK=0): RECLAIMABLE under pressure");
    } else if (locked == created) {
        snprintf(lock_note, sizeof(lock_note), "mlocked %.1f MiB (RLIMIT_MEMLOCK %s): a real reservation",
                 (double) locked / mib, lim);
    } else {
        snprintf(lock_note, sizeof(lock_note), "mlocked only %.1f of %.1f MiB (RLIMIT_MEMLOCK %s): the rest is "
                 "RECLAIMABLE under pressure", (double) locked / mib, (double) created / mib, lim);
    }

    GGML_LOG_INFO("%s: device %d: GTT expert store %.1f MiB realised of %.1f MiB requested (%s), registered %s, "
                  "%.1f%% huge pages (%.1f MiB AnonHugePages, %.1f MiB Rss), region created+registered in %.0f ms "
                  "(%.1f ms/GiB), carved %.1f MiB in %d tables / %lld "
                  "expert slots, in %d buffer carve%s of %.1f-%.1f MiB; %.1f MiB read from the GGUF into it; "
                  "%s; MemAvailable %.1f MiB of %.1f MiB total after the load\n",
                  __func__, device, (double) created / mib, (double) extent / mib, source,
                  ro ? "ReadOnly" : "WRITABLE", thp, (double) huge_kib / 1024.0, (double) rss_kib / 1024.0,
                  create_ms, created > 0 ? create_ms / ((double) created / 1073741824.0) : 0.0,
                  (double) used / mib, n_tables, (long long) n_slots,
                  n_carves, n_carves == 1 ? "" : "s", (double) carve_min / mib, (double) carve_max / mib,
                  (double) filled_bytes / mib, lock_note, (double) avail_kib / 1024.0, (double) total_kib / 1024.0);

    // The resident set this session can serve.  Say so out loud rather than implying the >RAM case works.
    GGML_LOG_INFO("%s: device %d: GTT expert store scope: ALL-RESIDENT arm only.  The GTT store holds the hottest "
                  "%d expert tables that fit the budget; anything that did not fit keeps its layer's plain "
                  "device buffer.  There is no eviction, no streaming refill and no prefill streaming yet "
                  "(P3-P5), so a set larger than the budget is a hot subset, not a streaming tier.  Kill "
                  "switch: MOE_EXPERT_GTT=0.\n", __func__, device, n_tables);

    if (total_kib > 0 && created / 1024 > total_kib) {
        GGML_LOG_WARN("%s: device %d: the GTT store (%.1f MiB) is larger than the machine's MemTotal (%.1f MiB); the "
                      "budget is not bounded by MemAvailable on this host\n", __func__, device,
                      (double) created / mib, (double) total_kib / 1024.0);
    }
}
