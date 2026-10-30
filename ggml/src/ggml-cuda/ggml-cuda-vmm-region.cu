// wip/kv-residency (TODO #117): the grow-only VMM region -- the O1 "fixed KV region" primitive.
// Contract, driver facts and the two traps: ggml-cuda-vmm-region.h.
//
// Nothing in the delivery calls this yet: the KV buffer type that will consume it is the next slice, and
// until then the only reachable entry point is the GGML_CUDA_VMM_REGION_SELFTEST diagnostic below (the
// same pattern LLAMA_RS_ROLLBACK_DEBUG / MOE_EXPERT_CACHE_VALIDATE / GGML_META_SS_VERIFY use: the
// primitive lives inside the backend library, so a diagnostic is the only way to exercise it in-tree).

#include "ggml-cuda-vmm-region.h"

#include "common.cuh"
#include "ggml-impl.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>

// One mapping.  Mixed backing (A2) needs mappings of *different* sizes inside one reservation -- a 2 MiB
// bulk and a 4 KiB-granular segment boundary -- so the extent is a table, not a per-unit bitmap.  `maps` is
// kept sorted by `off` with no overlaps and no two entries touching (a request that would extend an existing
// mapping is coalesced), which is what makes `is_mapped` a scan and `free` one unmap per entry.
struct vmm_mapping {
    size_t off   = 0;
    size_t bytes = 0;
    bool   host  = false;   // backed by pinned system RAM instead of VRAM
    // DIAGNOSTIC (block §3.7.6): fingerprint of a HOST mapping's contents as the flip left them, so a later
    // mutation of the spill can be caught and localised without a full dump.
    uint64_t h_flip  = 0;
    size_t   nz_flip = 0;
    bool     reported = false;
};

struct ggml_cuda_vmm_region {
    int      device  = -1;
    int      phys    = -1;
    size_t   size    = 0;   // the VA reservation
    size_t   unit    = 0;   // bulk map granularity (`unit`-sized steps by default)
    size_t   gran    = 0;   // the driver's MINIMUM granularity (4 KiB): what a boundary must align to
    size_t   mapped  = 0;   // bytes mapped (device + host)
    size_t   mapped_host = 0;
    void *   base    = nullptr;
    std::vector<vmm_mapping> maps;
    // pinned staging for a remap (grow-only, one unit's worth is enough for the sliding window)
    void *   stage       = nullptr;
    size_t   stage_bytes = 0;
    // device scratch of the same size: a restore into a freshly mapped HOST-location chunk must go through a
    // device write (see `vmm_attach`)
    void *   dev_stage       = nullptr;
    size_t   dev_stage_bytes = 0;
    // DIAGNOSTIC-only scratch for `vmm_fingerprint`: it must NOT use `stage`/`dev_stage`, which hold a flip's
    // live payload (using them silently corrupted the staged bytes and made the instrument lie).
    void *   probe_host       = nullptr;
    size_t   probe_bytes      = 0;
    void *   probe_dev        = nullptr;
};

namespace {

__global__ void vmm_copy_kernel(uint4 * __restrict__ dst, const uint4 * __restrict__ src, size_t n4) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (size_t) gridDim.x * blockDim.x) {
        dst[i] = src[i];
    }
}

// Copy `bytes` between device addresses (either side may be a HOST-location mapping) with a kernel.
// `cudaMemcpy` is NOT usable here: an H2D into a freshly mapped host-location chunk, and a D2D whose source is
// one, both complete "successfully" and leave zeros (measured 2026-10-30: stage=0x53, read=0x00 at byte 0 of a
// 4 MiB restore).  A kernel read/write of that VA is what the probes measured working (device write 53 GB/s).
static bool vmm_device_copy(void * dst, const void * src, size_t bytes) {
    if (bytes % 16 != 0) {
        return false;   // every mapping is a multiple of the driver's minimum granularity (>= 4 KiB)
    }
    vmm_copy_kernel<<<512, 256>>>((uint4 *) dst, (const uint4 *) src, bytes / 16);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    return true;
}

size_t vmm_up(size_t x, size_t a) {
    return a == 0 ? x : ((x + a - 1) / a) * a;
}

// The backend already queried this once at init (`ggml_cuda_info().devices[].vmm_granularity`, the
// RECOMMENDED granularity) and latched whether the driver offers VMM at all, so there is no second query
// and no need for the `CUresult`/`CU_MEM_ALLOC_GRANULARITY_MINIMUM` macros -- which `vendors/hip.h` does
// not map (only the RECOMMENDED one), and adding them would touch a header every backend compiles.
// The driver's MINIMUM map granularity (4 KiB on gfx1201).  A mapping must be a whole multiple of it, and a
// mixed-backing segment boundary is aligned to it.  Cached: two driver round trips per process at most.
size_t vmm_granularity_min(int device) {
    static size_t cache[GGML_CUDA_MAX_DEVICES] = {};
    static bool   done [GGML_CUDA_MAX_DEVICES] = {};
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES || !ggml_cuda_info().devices[device].vmm) {
        return 0;
    }
    if (done[device]) {
        return cache[device];
    }
    done[device] = true;
    CUmemAllocationProp prop = {};
    prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id   = ggml_cuda_get_physical_device(device);
    size_t g = 0;
    if (cuMemGetAllocationGranularity(&g, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != cudaSuccess || g == 0) {
        (void) cudaGetLastError();
        g = (size_t) 2 << 20;   // conservative fallback: the recommended granularity
    }
    return cache[device] = g;
}

size_t vmm_granularity(int device) {
    const ggml_cuda_device_info & info = ggml_cuda_info();
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES || !info.devices[device].vmm) {
        return 0;
    }
    const size_t g = info.devices[device].vmm_granularity;
    return g != 0 ? g : ((size_t) 2 << 20);
}

// ----------------------------------------------------------------------------------------------------------
// THE VMM HOST-MAPPING HAZARD (`A2-MIXED-BACKING.md` §3.7.6 / §3.7.9, TODO #118, measured 2026-10-30)
//
// A kind flip is `cuMemUnmap` of the old mapping followed by `cuMemCreate`/`cuMemMap` of the new one at the
// same VA.  On this driver that pair can DESTROY THE CONTENTS OF A DIFFERENT, LIVE host-location mapping in
// the same reservation: the mapping flipped one step earlier reads back as all zeros (whole 4 MiB mappings,
// exhaustively, with a 0xAB canary overwritten, so it is a real write and not a stale read).  In the windowed
// KV that silently shifted the prefill logits (KLD 0.004973 / 98.761 % on a 27B q8_0 arm) and changed the
// greedy text on every arm that actually spilled -- including the 4B, whose earlier "immunity" was only
// because those arms never reached a flip.  Neither the flip's stage-out nor its restore was at fault: both
// compare byte-exact against an independent copy-engine read.
//
// The trigger is the RELEASE of physical by `cuMemUnmap`: holding the just-released block in the runtime
// allocator's free list for the duration of the following create makes the whole slide byte-exact (32/32
// flips, 0/8 corrupted mappings, KLD -0.000000 / 100.000 %).  Isolated reproducer and its discriminating
// arms: `wip/kv-residency/hostmap-hazard-probe.cc` (default arm FAILS, `--precreate` still fails, `--grab`
// passes).  So `vmm_map` takes a hold first -- on EVERY map site, not only the flip, because the hazard is a
// property of the create that follows a release, and the map-new path (`ensure_span`, `grow_for`, a state
// restore crossing into the spill) can follow one too.
//
// `LLAMA_KV_REGION_QUARANTINE=0` is the kill-switch (A/B and bisection only -- with it off the spill is NOT
// numerically transparent).  The hold is the size of the mapping being created, so it scales with the unit.
// A refused hold FAILS the mapping: proceeding without it is a silent wrong answer, and the region's contract
// is to refuse loudly and let the caller fall back to today's fit-or-fail.

namespace {

struct vmm_quarantine_hold {
    void * p = nullptr;

    bool acquire(size_t bytes) {
        if (cudaMalloc(&p, bytes) != cudaSuccess) {
            (void) cudaGetLastError();
            p = nullptr;
            return false;
        }
        return true;
    }
    ~vmm_quarantine_hold() {
        if (p != nullptr) {
            (void) cudaFree(p);
        }
    }
};

bool vmm_quarantine_enabled() {
    static const bool on = [] {
        const char * e = getenv("LLAMA_KV_REGION_QUARANTINE");
        return e == nullptr || e[0] == '\0' || strcmp(e, "0") != 0;
    }();
    return on;
}

}  // namespace

// Map `bytes` of fresh physical at `addr` and grant READWRITE to the owner, then each peer on its own
// (best effort -- a refused peer descriptor must not fail the mapping; the slab's helper documents the
// `-sm tensor` fault that taught that).  The handle is released after a successful map.
bool vmm_map(ggml_cuda_vmm_region * r, size_t addr_off, size_t bytes, bool host = false) {
    // Take the physical an imminent `cuMemUnmap` released out of the pool before creating this mapping: see
    // the hazard note above.  Held across `cuMemCreate`/`cuMemMap`/`cuMemSetAccess` and released on return.
    const bool quarantine = vmm_quarantine_enabled();
    vmm_quarantine_hold hold;
    if (quarantine && !hold.acquire(bytes)) {
        GGML_LOG_ERROR("%s: device %d: could not reserve a %.2f MiB hold for the map at %.2f MiB.  The hold is "
                       "what keeps a VMM host-location mapping in this reservation from being zeroed by the "
                       "driver (LLAMA_KV_REGION_QUARANTINE); mapping without it is a silent wrong answer, so the "
                       "window is refused instead.  Free %.2f MiB of VRAM, or unset LLAMA_KV_REGION.\n",
                       __func__, r->device, bytes / 1048576.0, addr_off / 1048576.0, bytes / 1048576.0);
        return false;
    }
    CUmemAllocationProp prop = {};
    prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
    // A host-location handle is pinned system RAM granted device access: the device can read AND write it
    // (measured: 28.4 GB/s, both directions, through the copy engine too), which is why mixed backing needs
    // no kernel change.  Same code path otherwise -- a VMM mapping is a VMM mapping.
    prop.location.type = host ? CU_MEM_LOCATION_TYPE_HOST : CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id   = host ? 0 : r->phys;

    CUmemGenericAllocationHandle handle = 0;
    if (cuMemCreate(&handle, bytes, &prop, 0) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    const CUdeviceptr addr = (CUdeviceptr) ((uintptr_t) r->base + addr_off);
    if (cuMemMap(addr, bytes, 0, handle, 0) != cudaSuccess) {
        (void) cudaGetLastError();
        (void) cuMemRelease(handle);
        return false;
    }

    CUmemAccessDesc own = {};
    own.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    own.location.id   = r->phys;
    own.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (cuMemSetAccess(addr, bytes, &own, 1) != cudaSuccess) {
        (void) cudaGetLastError();
        (void) cuMemUnmap(addr, bytes);
        (void) cuMemRelease(handle);
        return false;
    }
    const int n_dev = ggml_backend_cuda_get_device_count();
    for (int d = 0; d < n_dev; d++) {
        const int p = ggml_cuda_get_physical_device(d);
        if (p == r->phys) {
            continue;
        }
        CUmemAccessDesc peer = {};
        peer.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        peer.location.id   = p;
        peer.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        if (cuMemSetAccess(addr, bytes, &peer, 1) != cudaSuccess) {
            (void) cudaGetLastError();   // owner access stands; copies from that peer stage via host
        }
    }
    (void) cuMemRelease(handle);   // the mapping keeps the physical alive
    if (getenv("LLAMA_KV_REGION_TRACE") != nullptr) {
        fprintf(stderr, "MMAP   addr=%8.3f MiB bytes=%7.3f host=%d\n", addr_off / 1048576.0, bytes / 1048576.0, (int) host);
        fflush(stderr);
    }
    return true;
}

void vmm_unmap(ggml_cuda_vmm_region * r, size_t addr_off, size_t bytes) {
    if (bytes == 0) {
        return;
    }
    const CUdeviceptr addr = (CUdeviceptr) ((uintptr_t) r->base + addr_off);
    if (getenv("LLAMA_KV_REGION_TRACE") != nullptr) {
        fprintf(stderr, "MUNMAP addr=%8.3f MiB bytes=%7.3f\n", addr_off / 1048576.0, bytes / 1048576.0);
        fflush(stderr);
    }
    if (cuMemUnmap(addr, bytes) != cudaSuccess) {
        (void) cudaGetLastError();
    }
}

}  // namespace

ggml_cuda_vmm_region * ggml_cuda_vmm_region_create(int device, size_t bytes, size_t unit, char * err, size_t errsz) {
    auto fail = [&](const char * why) -> ggml_cuda_vmm_region * {
        if (err != nullptr && errsz > 0) {
            std::snprintf(err, errsz, "%s", why);
        }
        return nullptr;
    };
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) {
        return fail("bad device");
    }
    if (bytes == 0) {
        return fail("zero size");
    }

    const int phys = ggml_cuda_get_physical_device(device);
    const size_t gran = vmm_granularity(device);
    if (gran == 0) {
        return fail("VMM unavailable on this device");
    }
    size_t u = unit != 0 ? unit : ((size_t) 64 << 20);
    u = vmm_up(u, gran);

    auto * r = new ggml_cuda_vmm_region();
    r->device = device;
    r->phys   = phys;
    r->unit   = u;
    r->gran   = vmm_granularity_min(device);
    if (r->gran == 0) {
        r->gran = gran;   // the recommended granularity: always legal, just coarser
    }
    r->size   = vmm_up(bytes, u);
    r->maps.reserve(1024);

    CUdeviceptr base = 0;
    if (cuMemAddressReserve(&base, r->size, 0, 0, 0) != cudaSuccess) {
        (void) cudaGetLastError();
        delete r;
        return fail("cuMemAddressReserve refused the VA reservation");
    }
    r->base = (void *) base;

    if (err != nullptr && errsz > 0) {
        err[0] = '\0';
    }
    return r;   // nothing is mapped yet: mapped == 0
}

namespace {

// Is [off, off + bytes) covered by one mapping, and is that mapping host-backed?  (Mappings never overlap,
// so "covered" means exactly one entry contains the whole range.)
// The mapping whose range contains (or starts at) `off`, or nullptr.  `maps` is sorted by `off` and
// non-overlapping, so this is a binary search: the sliding window calls it once per cell per batch.
const vmm_mapping * vmm_at(const ggml_cuda_vmm_region * r, size_t off) {
    auto it = std::upper_bound(r->maps.begin(), r->maps.end(), off,
            [](size_t v, const vmm_mapping & m) { return v < m.off; });
    if (it == r->maps.begin()) {
        return nullptr;
    }
    --it;
    return off - it->off < it->bytes ? &*it : nullptr;
}

// The first mapping that starts strictly after `off`, or nullptr.
const vmm_mapping * vmm_next(const ggml_cuda_vmm_region * r, size_t off) {
    auto it = std::upper_bound(r->maps.begin(), r->maps.end(), off,
            [](size_t v, const vmm_mapping & m) { return v < m.off; });
    return it == r->maps.end() ? nullptr : &*it;
}

bool vmm_find(const ggml_cuda_vmm_region * r, size_t off, size_t bytes, size_t * end, bool * host) {
    // "Is every byte of [off, off+bytes) mapped?" -- a range may span several mappings (they are one per
    // `cuMemMap`, so a multi-unit extent is a run of them).  `end`/`host` describe the FIRST mapping.
    size_t a = off;
    const size_t b = off + bytes;
    bool first = true;
    while (a < b) {
        const vmm_mapping * m = vmm_at(r, a);
        if (m == nullptr) {
            return false;
        }
        if (first) {
            if (end  != nullptr) { *end  = m->off + m->bytes; }
            if (host != nullptr) { *host = m->host; }
            first = false;
        }
        a = m->off + m->bytes;
    }
    return true;
}

// Insert a mapping and keep the table sorted.  Entries are NOT coalesced: one entry is exactly one driver
// mapping, i.e. exactly one `cuMemMap` -- and that is what a kind flip can safely rewrite.
//
// The table used to coalesce same-kind neighbours, which made a flip move (and re-map) a *range of driver
// mappings* as one: `cuMemUnmap`/`cuMemMap` over several prior map calls, and the H2D copy that restores the
// staged bytes then landed only in the first of them (measured 2026-10-30: a 24 MiB entry re-mapped and
// re-filled read back zeros everywhere except the first 8 MiB).  A table entry that matches the driver's own
// granularity keeps every flip a single, well-defined operation.
void vmm_insert(ggml_cuda_vmm_region * r, size_t off, size_t bytes, bool host) {
    // One entry per DRIVER mapping, i.e. per `cuMemMap`.  The table used to coalesce same-kind neighbours,
    // which looks harmless but makes a kind flip rewrite a *range* of prior map calls as one: the unmap/map
    // pair then spans several mappings and the H2D copy that restores the staged bytes lands only in the first
    // of them (measured 2026-10-30: an 8 MiB unit inside a coalesced 24 MiB entry read back all zeros after its
    // flip).  Keeping the entries 1:1 also makes "does this request coincide with a mapping?" -- the question a
    // flip has to answer -- exact.
    size_t i = 0;
    while (i < r->maps.size() && r->maps[i].off < off) { ++i; }
    r->maps.insert(r->maps.begin() + (long) i, vmm_mapping{off, bytes, host});
    r->mapped += bytes;
    if (host) { r->mapped_host += bytes; }
}

}  // namespace

bool ggml_cuda_vmm_region_map_span(ggml_cuda_vmm_region * r, size_t off, size_t bytes, bool host) {
    if (r == nullptr) {
        return false;
    }
    if (bytes == 0) {
        return true;
    }
    if (off > r->size || bytes > r->size - off) {
        return false;   // outside the reservation -- the caller sized the region for its own extent
    }
    // Round OUTWARD to the driver's minimum granularity (a host mapping must be a whole multiple of it, and a
    // mixed-backing boundary is 4 KiB-granular), then map in <=`unit` pieces.  Rounding outward can only touch
    // more VA than asked for, never a byte outside the reservation.
    const size_t g  = r->gran != 0 ? r->gran : (size_t) 4096;
    size_t a = (off / g) * g;
    size_t b = vmm_up(off + bytes, g);
    if (b > r->size) { b = r->size; }

    while (a < b) {
        if (const vmm_mapping * cur = vmm_at(r, a)) {
            a = cur->off + cur->bytes;   // already mapped: skip it (the driver refuses a re-map)
            continue;
        }
        size_t n = r->unit;
        if (const vmm_mapping * nxt = vmm_next(r, a)) {
            if (nxt->off - a < n) { n = nxt->off - a; }   // do not step over an existing mapping
        }
        if (b - a < n) { n = b - a; }
        if (!vmm_map(r, a, n, host)) {
            return false;             // leave what did map: those bytes are valid and cleared
        }
        vmm_insert(r, a, n, host);
        if (getenv("LLAMA_KV_REGION_TRACE") != nullptr) {
            fprintf(stderr, "MSETsp addr=%8.3f MiB bytes=%7.3f host=%d\n", a / 1048576.0, n / 1048576.0, (int) host);
            fflush(stderr);
        }
        // A fresh mapping can contain stale bytes (measured), so clear exactly what was just mapped.
        if (cudaMemset((char *) r->base + a, 0, n) != cudaSuccess) {
            (void) cudaGetLastError();
            return false;
        }
        a += n;
    }
    return true;
}

// Grow the pinned scratch to hold `bytes` (the staged bytes of one mapping).
static bool vmm_ensure_stage(ggml_cuda_vmm_region * r, size_t bytes) {
    if (bytes > r->stage_bytes) {
        if (r->stage != nullptr) { (void) cudaFreeHost(r->stage); r->stage = nullptr; r->stage_bytes = 0; }
        void * st = nullptr;
        if (cudaMallocHost(&st, bytes) != cudaSuccess) {
            (void) cudaGetLastError();
            return false;
        }
        r->stage = st;
        r->stage_bytes = bytes;
    }
    return true;
}

// Grow the device scratch (the staging copy's device-side leg).
static bool vmm_ensure_dev_stage(ggml_cuda_vmm_region * r, size_t bytes) {
    if (bytes > r->dev_stage_bytes) {
        if (r->dev_stage != nullptr) { (void) cudaFree(r->dev_stage); r->dev_stage = nullptr; r->dev_stage_bytes = 0; }
        void * d = nullptr;
        if (cudaMalloc(&d, bytes) != cudaSuccess) {
            (void) cudaGetLastError();
            return false;
        }
        r->dev_stage = d;
        r->dev_stage_bytes = bytes;
    }
    return true;
}

// Stage a mapping's bytes to the pinned scratch, unmap it and drop its table entry (updating the counters).
// A remap is unmap + map, and the driver hands back fresh physical with no relation to what was there
// (measured: stale on device chunks, zero on host ones), so everything that changes a mapping's KIND -- or
// cuts one in two -- goes through this.
static bool vmm_detach(ggml_cuda_vmm_region * r, const vmm_mapping & m) {
    if (!vmm_ensure_stage(r, m.bytes)) {
        return false;
    }
    // nothing may be in flight while the VA is unmapped (the VA stays valid for graphs, the *physical* does not)
    if (cudaDeviceSynchronize() != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    // mapping -> device scratch -> host stage.  A direct copy whose SOURCE is a host-location mapping is not
    // trusted either (see `vmm_attach`); the device-to-device leg is the one the probes measured as reliable.
    if (!vmm_ensure_dev_stage(r, m.bytes)) {
        return false;
    }
    if (!vmm_device_copy(r->dev_stage, (const char *) r->base + m.off, m.bytes) ||
        cudaMemcpy(r->stage, r->dev_stage, m.bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    vmm_unmap(r, m.off, m.bytes);
    const size_t off = m.off, bytes = m.bytes;
    const bool   was_host = m.host;
    r->maps.erase(std::find_if(r->maps.begin(), r->maps.end(),
            [off](const vmm_mapping & e) { return e.off == off; }));
    r->mapped      -= bytes;
    r->mapped_host -= was_host ? bytes : 0;
    return true;
}

// Map [off, off+bytes) from `host` and restore its bytes from the scratch (starting at `stage_off`).
static bool vmm_attach(ggml_cuda_vmm_region * r, size_t off, size_t bytes, bool host, size_t stage_off = 0) {
    if (!vmm_map(r, off, bytes, host)) {
        return false;   // left unmapped: the caller's next batch will refuse loudly rather than read garbage
    }
    // stage -> device scratch -> mapping.  A DIRECT H2D into a freshly mapped HOST-location chunk does not
    // land: measured 2026-10-30, the write is silent and the bytes read back as zeros (stage=0x53, read=0x00 at
    // byte 0 of a 4 MiB restore), which corrupted every spilled KV cell it touched -- a prefill-logit KLD of
    // 0.005 at ctx 8192 and a changed greedy text on the deep arm.  A DEVICE write to the same VA works (the
    // KV's own `set_rows` path writes host-backed cells), so the restore goes through a device scratch.
    if (!vmm_ensure_dev_stage(r, bytes)) {
        return false;
    }
    if (cudaMemcpy(r->dev_stage, (const char *) r->stage + stage_off, bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        !vmm_device_copy((char *) r->base + off, r->dev_stage, bytes)) {
        (void) cudaGetLastError();
        return false;
    }
    // The restore must be COMPLETE before any kernel touches the mapping.  The compute graph is enqueued on
    // the backend's own stream, and a copy that is only ordered on the legacy default stream is not ordered
    // against it -- so a batch could read a mapping whose staged bytes are still in flight.
    if (cudaDeviceSynchronize() != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    vmm_insert(r, off, bytes, host);   // re-insert
    if (getenv("LLAMA_KV_REGION_DEBUG") != nullptr) {
        // verify the restore landed (a debug-only readback: an H2D into a freshly mapped VA was measured to
        // land only partially in an earlier revision of this code)
        std::vector<char> chk(bytes);
        if (vmm_device_copy(r->dev_stage, (char *) r->base + off, bytes) &&
                cudaMemcpy(chk.data(), r->dev_stage, bytes, cudaMemcpyDeviceToHost) == cudaSuccess) {
            // EXHAUSTIVE (2026-10-30, block §3.7.6): the compare used to step every 512 bytes, which cannot
            // see a corruption smaller than that -- and the open finding is a *small*, deterministic numeric
            // change.  Compare every byte and report the first offset plus how many differ.
            const char * src = (const char *) r->stage + stage_off;
            size_t first_bad = bytes, n_bad = 0;
            for (size_t i = 0; i < bytes; i++) {
                if (chk[i] != src[i]) { if (first_bad == bytes) { first_bad = i; } n_bad++; }
            }
            GGML_LOG_INFO("%s: VERIFY device %d off=%.3f MiB bytes=%.3f MiB -> %s\n", __func__, r->device,
                    off / 1048576.0, bytes / 1048576.0, first_bad == bytes ? "ok (exhaustive)" : "MISMATCH");
            if (first_bad != bytes) {
                GGML_LOG_ERROR("%s: mismatch at byte %zu of %.3f MiB, %zu of %zu bytes differ (stage=%02x read=%02x)\n",
                        __func__, first_bad, bytes / 1048576.0, n_bad, bytes,
                        (unsigned char) src[first_bad], (unsigned char) chk[first_bad]);
            }
        }
    }
    return true;
}

// DIAGNOSTIC (block §3.7.6): read `bytes` at `off` through a device kernel into a host scratch and return a
// FNV-1a hash plus the zero-byte count.  Used to fingerprint a HOST mapping as the flip leaves it and to
// re-check it later, so a mutation of the spill can be caught and localised.
static bool vmm_fingerprint(ggml_cuda_vmm_region * r, size_t off, size_t bytes, uint64_t * h, size_t * nz) {
    if (bytes > r->probe_bytes) {
        if (r->probe_host != nullptr) { (void) cudaFreeHost(r->probe_host); r->probe_host = nullptr; }
        if (r->probe_dev  != nullptr) { (void) cudaFree(r->probe_dev);      r->probe_dev  = nullptr; }
        r->probe_bytes = 0;
        if (cudaMallocHost(&r->probe_host, bytes) != cudaSuccess) { (void) cudaGetLastError(); return false; }
        if (cudaMalloc(&r->probe_dev, bytes) != cudaSuccess) {
            (void) cudaGetLastError();
            (void) cudaFreeHost(r->probe_host); r->probe_host = nullptr;
            return false;
        }
        r->probe_bytes = bytes;
    }
    if (!vmm_device_copy(r->probe_dev, (const char *) r->base + off, bytes) ||
            cudaMemcpy(r->probe_host, r->probe_dev, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    // Cross-check: does the COPY ENGINE see the same bytes as a device kernel?  A disagreement would mean the
    // instrument (not the mapping) is wrong, which is exactly the trap that made the sliding-window debug
    // verify blind before.
    if (getenv("LLAMA_KV_REGION_DEBUG") != nullptr) {
        std::vector<unsigned char> ce(bytes);
        if (cudaMemcpy(ce.data(), (const char *) r->base + off, bytes, cudaMemcpyDeviceToHost) == cudaSuccess) {
            const unsigned char * k = (const unsigned char *) r->probe_host;
            size_t first = bytes, nb = 0;
            for (size_t i = 0; i < bytes; i++) {
                if (k[i] != ce[i]) { if (first == bytes) { first = i; } nb++; }
            }
            if (first != bytes) {
                fprintf(stderr, "READER_DIFF off=%.3f MiB: kernel vs copy-engine differ, %zu of %zu bytes, first at %zu "
                        "(kernel=%02x ce=%02x)\n", off / 1048576.0, nb, bytes, first, k[first], ce[first]);
                fflush(stderr);
            }
        } else {
            (void) cudaGetLastError();
        }
    }
    const unsigned char * p = (const unsigned char *) r->probe_host;
    uint64_t hv = 1469598103934665603ull;
    size_t   z  = 0;
    for (size_t i = 0; i < bytes; i++) {
        hv = (hv ^ p[i]) * 1099511628211ull;
        z += p[i] == 0;
    }
    if (h  != nullptr) { *h  = hv; }
    if (nz != nullptr) { *nz = z;  }
    return true;
}

// DIAGNOSTIC (block §3.7.6): is every host mapping still what the flip left there?  Called from the write
// paths and from the growth points, one line per divergence (which mapping, when, how many bytes changed).
void ggml_cuda_vmm_region_verify_host(ggml_cuda_vmm_region * r, const char * where) {
    if (r == nullptr || getenv("LLAMA_KV_REGION_DEBUG") == nullptr) {
        return;
    }
    for (const vmm_mapping & m : r->maps) {
        if (!m.host || m.h_flip == 0) {
            continue;
        }
        uint64_t h = 0; size_t nz = 0;
        if (!vmm_fingerprint(r, m.off, m.bytes, &h, &nz)) {
            fprintf(stderr, "HOSTMAP_CLOBBER dev=%d at=%s off=%.3f MiB: READBACK FAILED\n",
                    r->device, where, m.off / 1048576.0);
            continue;
        }
        if (h != m.h_flip) {
            vmm_mapping & mm = const_cast<vmm_mapping &>(m);
            if (!mm.reported) {
                mm.reported = true;
                fprintf(stderr, "HOSTMAP_CLOBBER dev=%d at=%s off=%.3f MiB bytes=%.3f MiB: changed (zeros %zu -> %zu)\n",
                        r->device, where, m.off / 1048576.0, m.bytes / 1048576.0, m.nz_flip, nz);
                fflush(stderr);
            }
        }
    }
}

// Preserve a mapping's bytes across a kind change.  Only the sliding window takes this path, once per unit
// per few thousand tokens, so a blocking round trip is the right trade for simplicity here.
//
// This is an `unmap` followed by a `map`, which is exactly the pair the hazard note above `vmm_map` describes;
// `vmm_map` takes the hold that keeps it safe, so nothing here has to.
bool vmm_remap(ggml_cuda_vmm_region * r, const vmm_mapping & m, bool host) {
    const vmm_mapping saved = m;
    const bool dbg = getenv("LLAMA_KV_REGION_DEBUG") != nullptr;
    if (dbg) {
        GGML_LOG_INFO("%s: FLIP device %d off=%.3f bytes=%.3f -> %s\n", __func__, r->device,
                saved.off / 1048576.0, saved.bytes / 1048576.0, host ? "host" : "vram");
    }
    // STAGE-OUT BLIND SPOT (block §3.7.6, plan step 1, now closed): the restore verify compares the STAGE with
    // the RESTORED bytes, which holds even if the stage itself was filled with the wrong data.  Take an
    // INDEPENDENT read of the mapping first -- the copy engine, the trusted path for a VRAM source -- and
    // compare it against what the flip's own kernel copy put in the stage.  Measured: 32/32 flips byte-exact,
    // so the flip was never the fault; it is kept as the regression test for the copy paths.
    std::vector<char> pre;
    bool pre_ok = false;
    if (dbg && !saved.host) {
        pre.resize(saved.bytes);
        if (cudaDeviceSynchronize() == cudaSuccess &&
                cudaMemcpy(pre.data(), (const char *) r->base + saved.off, saved.bytes,
                    cudaMemcpyDeviceToHost) == cudaSuccess) {
            pre_ok = true;
        } else {
            (void) cudaGetLastError();
        }
    }
    if (!vmm_detach(r, saved)) {
        return false;
    }
    if (pre_ok) {
        const char * st = (const char *) r->stage;
        size_t first_bad = saved.bytes, n_bad = 0;
        for (size_t i = 0; i < saved.bytes; i++) {
            if (pre[i] != st[i]) { if (first_bad == saved.bytes) { first_bad = i; } n_bad++; }
        }
        GGML_LOG_INFO("%s: STAGEOUT device %d off=%.3f MiB -> %s\n", __func__, r->device,
                saved.off / 1048576.0,
                first_bad == saved.bytes ? "ok (copy-engine vs kernel copy, exhaustive)" : "MISMATCH");
        if (first_bad != saved.bytes) {
            GGML_LOG_ERROR("%s: stage-out MISMATCH at byte %zu, %zu of %zu bytes differ (copy-engine=%02x stage=%02x)\n",
                    __func__, first_bad, n_bad, saved.bytes, (unsigned char) pre[first_bad],
                    (unsigned char) st[first_bad]);
        }
    }
    if (!vmm_attach(r, saved.off, saved.bytes, host, 0)) {
        return false;
    }
    if (dbg && host) {
        // CANARY (§3.7.6): overwrite the just-flipped mapping with a known non-zero pattern, so a later change
        // to it is proof that SOMETHING WROTE there, as opposed to a read returning stale bytes.  This is what
        // separated "the spill was written over" from "the reader saw stale bytes" in the hazard hunt.
        if (getenv("LLAMA_KV_REGION_CANARY") != nullptr) {
            if (vmm_ensure_dev_stage(r, saved.bytes)) {
                std::vector<unsigned char> ab(saved.bytes, 0xAB);
                if (cudaMemcpy(r->dev_stage, ab.data(), saved.bytes, cudaMemcpyHostToDevice) == cudaSuccess) {
                    (void) vmm_device_copy((char *) r->base + saved.off, r->dev_stage, saved.bytes);
                }
            }
        }
        for (vmm_mapping & e : r->maps) {
            if (e.off == saved.off) {
                e.h_flip = 0; e.nz_flip = 0; e.reported = false;
                (void) vmm_fingerprint(r, e.off, e.bytes, &e.h_flip, &e.nz_flip);
                break;
            }
        }
    }
    return true;
}

bool ggml_cuda_vmm_region_ensure_span(ggml_cuda_vmm_region * r, size_t off, size_t bytes, bool host) {
    if (r == nullptr) {
        return false;
    }
    if (getenv("LLAMA_KV_REGION_TRACE") != nullptr) {
        fprintf(stderr, "ENSURE off=%8.3f MiB bytes=%7.3f host=%d\n", off / 1048576.0, bytes / 1048576.0, (int) host);
        fflush(stderr);
    }
    if (bytes == 0) {
        return true;
    }
    if (off > r->size || bytes > r->size - off) {
        return false;
    }
    const size_t g = r->gran != 0 ? r->gran : (size_t) 4096;
    size_t a = (off / g) * g;
    size_t b = vmm_up(off + bytes, g);
    if (b > r->size) { b = r->size; }

    while (a < b) {
        const vmm_mapping * cur = vmm_at(r, a);
        if (cur != nullptr) {
            const size_t cur_end = cur->off + cur->bytes;
            const size_t seg_end = b < cur_end ? b : cur_end;   // the part of [a,b) this mapping covers
            if (cur->host != host) {
                // The kind is wrong for [a, b): flip the mapping, preserving its bytes.
                //
                // A flip is a WHOLE-mapping operation.  The driver refuses a sub-range unmap, and rewriting
                // one mapping as several pieces of mixed kinds (detach, then re-attach head/mid/tail) is NOT
                // safe here: the pieces after the first read back as zeros, i.e. the staged bytes are lost
                // (measured 2026-10-30).  The spill boundary therefore has to land ON a mapping boundary, and
                // it does by construction -- the window quantises the spilled BYTES down to a whole map unit
                // and every mapping in a windowed region is unit-aligned (`grow_cells`/`init_tensor` round
                // their extents up).  A request that still straddles is refused loudly: silently flipping the
                // whole mapping is how the window once spilled NOTHING at all (VRAM unchanged, `mapped_host`
                // 0) while its warning claimed the tail was in system RAM.
                if (cur->off != a || cur_end != b) {
                    GGML_LOG_ERROR("%s: device %d: refusing to flip [%.2f, %.2f) MiB: the mapping [%.2f, %.2f) "
                                   "MiB covers it only in part, and a partial rewrite of a mapping is not safe on "
                                   "this driver.  A windowed KV keeps every mapping on the map-unit grid; an "
                                   "unaligned request here is a caller bug.\n", __func__, r->device,
                                   a / 1048576.0, b / 1048576.0, cur->off / 1048576.0, cur_end / 1048576.0);
                    return false;
                }
                if (!vmm_remap(r, *cur, host)) {
                    return false;
                }
            }
            a = seg_end;                       // [a, seg_end) is the kind we want now
            continue;
        }
        size_t n = r->unit;
        if (const vmm_mapping * nxt = vmm_next(r, a)) {
            if (nxt->off - a < n) { n = nxt->off - a; }
        }
        if (b - a < n) { n = b - a; }
        if (!vmm_map(r, a, n, host)) {
            return false;
        }
        vmm_insert(r, a, n, host);
        if (getenv("LLAMA_KV_REGION_TRACE") != nullptr) {
            fprintf(stderr, "MSETmap addr=%8.3f MiB bytes=%7.3f host=%d\n", a / 1048576.0, n / 1048576.0, (int) host);
            fflush(stderr);
        }
        if (cudaMemset((char *) r->base + a, 0, n) != cudaSuccess) {
            (void) cudaGetLastError();
            return false;
        }
        a += n;
    }
    return true;
}

bool ggml_cuda_vmm_region_grow_span(ggml_cuda_vmm_region * r, size_t off, size_t bytes) {
    if (r == nullptr) {
        return false;
    }
    if (bytes == 0) {
        return true;
    }
    if (off > r->size || bytes > r->size - off) {
        return false;
    }
    // O1's form: whole `unit` chunks, rounded outward, VRAM-backed.  Kept for the O1 buffer type and the
    // selftest; mixed backing (A2) calls `map_span` directly with its own granularity.
    const size_t a = (off / r->unit) * r->unit;
    size_t b = vmm_up(off + bytes, r->unit);
    if (b > r->size) { b = r->size; }
    return ggml_cuda_vmm_region_map_span(r, a, b - a, false);
}

bool ggml_cuda_vmm_region_grow(ggml_cuda_vmm_region * r, size_t used_bytes) {
    if (r == nullptr) {
        return false;
    }
    if (used_bytes == 0) {
        return true;
    }
    return ggml_cuda_vmm_region_grow_span(r, 0, used_bytes);
}

bool ggml_cuda_vmm_region_is_mapped(const ggml_cuda_vmm_region * r, size_t off, size_t bytes) {
    if (r == nullptr) {
        return false;
    }
    if (bytes == 0) {
        return off <= r->size;
    }
    if (off > r->size || bytes > r->size - off) {
        return false;
    }
    return vmm_find(r, off, bytes, nullptr, nullptr);
}

size_t ggml_cuda_vmm_region_mapped_host(const ggml_cuda_vmm_region * r) { return r != nullptr ? r->mapped_host : 0; }

size_t ggml_cuda_vmm_region_granularity(int device) { return vmm_granularity_min(device); }

void ggml_cuda_vmm_region_clear(ggml_cuda_vmm_region * r, uint8_t value) {
    if (r == nullptr) {
        return;
    }
    for (const vmm_mapping & m : r->maps) {
        if (cudaMemset((char *) r->base + m.off, value, m.bytes) != cudaSuccess) {
            (void) cudaGetLastError();
            return;
        }
    }
}

size_t ggml_cuda_vmm_region_mapped(const ggml_cuda_vmm_region * r) { return r != nullptr ? r->mapped : 0; }
size_t ggml_cuda_vmm_region_size  (const ggml_cuda_vmm_region * r) { return r != nullptr ? r->size   : 0; }
size_t ggml_cuda_vmm_region_unit  (const ggml_cuda_vmm_region * r) { return r != nullptr ? r->unit   : 0; }
void * ggml_cuda_vmm_region_base  (const ggml_cuda_vmm_region * r) { return r != nullptr ? r->base   : nullptr; }

// DIAGNOSTIC (block §3.7.6): write every mapped byte of the region to `path`, at its own file offset, read
// back through a DEVICE KERNEL -- the access the probes measured as working for a host-location mapping, so
// the dump shows what a kernel (e.g. flash attention) would read rather than what a copy engine returns.
// The VA offset of every cell is identical in every arm, so two arms' dumps are byte-comparable: this is the
// instrument that separates "the bytes in the spill are wrong" from "the reader is wrong".
bool ggml_cuda_vmm_region_dump(ggml_cuda_vmm_region * r, const char * path) {
    if (r == nullptr || path == nullptr) {
        return false;
    }
    FILE * f = fopen(path, "wb");
    if (f == nullptr) {
        GGML_LOG_ERROR("%s: cannot open %s\n", __func__, path);
        return false;
    }
    size_t chunk = r->unit != 0 ? r->unit : (size_t) 4 << 20;
    size_t done = 0;
    for (const vmm_mapping & m : r->maps) {
        for (size_t o = 0; o < m.bytes; o += chunk) {
            const size_t n = m.bytes - o < chunk ? m.bytes - o : chunk;
            if (!vmm_ensure_stage(r, n) || !vmm_ensure_dev_stage(r, n)) {
                fprintf(stderr, "KVREGION_DUMP stage alloc failed (%zu)\n", n);
                fclose(f);
                return false;
            }
            if (!vmm_device_copy(r->dev_stage, (const char *) r->base + m.off + o, n) ||
                    cudaMemcpy(r->stage, r->dev_stage, n, cudaMemcpyDeviceToHost) != cudaSuccess) {
                (void) cudaGetLastError();
                fclose(f);
                return false;
            }
            if (fseek(f, (long) (m.off + o), SEEK_SET) != 0 || fwrite(r->stage, 1, n, f) != n) {
                fclose(f);
                return false;
            }
            done += n;
        }
    }
    fclose(f);
    fprintf(stderr, "KVREGION_DUMP dev=%d wrote=%.2f of mapped=%.2f (host %.2f) maps=%zu -> %s\n", r->device,
            done / 1048576.0, r->mapped / 1048576.0, r->mapped_host / 1048576.0, r->maps.size(), path);
    fflush(stderr);
    return true;
}

void ggml_cuda_vmm_region_free(ggml_cuda_vmm_region * r) {
    if (r == nullptr) {
        return;
    }
    if (const char * dump = getenv("LLAMA_KV_REGION_DUMP")) {
        if (dump[0] != '\0') {
            fprintf(stderr, "KVREGION_FREE dev=%d maps=%zu mapped=%.2f host=%.2f dump=%s\n", r->device,
                    r->maps.size(), r->mapped/1048576.0, r->mapped_host/1048576.0, dump);
            fflush(stderr);
            (void) ggml_cuda_vmm_region_dump(r, dump);
        }
    }
    // One unmap per mapping, each of a whole mapping -- never a sub-range (which the driver refuses,
    // VMM-PROBE T5).  Mixed backing's small boundary mappings are unmapped exactly like the bulk ones.
    for (const vmm_mapping & m : r->maps) {
        vmm_unmap(r, m.off, m.bytes);
    }
    r->maps.clear();
    r->mapped = 0;
    r->mapped_host = 0;
    if (r->stage != nullptr) {
        (void) cudaFreeHost(r->stage);
        r->stage = nullptr;
        r->stage_bytes = 0;
    }
    if (r->dev_stage != nullptr) {
        (void) cudaFree(r->dev_stage);
        r->dev_stage = nullptr;
        r->dev_stage_bytes = 0;
    }
    if (r->probe_host != nullptr) { (void) cudaFreeHost(r->probe_host); r->probe_host = nullptr; }
    if (r->probe_dev  != nullptr) { (void) cudaFree(r->probe_dev);      r->probe_dev  = nullptr; }
    if (r->base != nullptr) {
        if (cuMemAddressFree((CUdeviceptr) (uintptr_t) r->base, r->size) != cudaSuccess) {
            (void) cudaGetLastError();
        }
    }
    delete r;
}

bool ggml_cuda_vmm_region_available(int device) {
    // One real probe per device: reserve one unit, map it, unmap it, free it.  Cached, because the answer
    // cannot change inside a process and the probe costs two driver round trips.
    static bool     ok[GGML_CUDA_MAX_DEVICES] = {};
    static bool     done[GGML_CUDA_MAX_DEVICES] = {};
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) {
        return false;
    }
    if (done[device]) {
        return ok[device];
    }
    done[device] = true;
    const size_t g = vmm_granularity(device);
    if (g == 0) {
        return ok[device] = false;
    }
    const size_t u = g < ((size_t) 2 << 20) ? ((size_t) 2 << 20) : g;
    char err[128] = {0};
    ggml_cuda_vmm_region * r = ggml_cuda_vmm_region_create(device, u, u, err, sizeof(err));
    if (r == nullptr) {
        GGML_LOG_INFO("%s: device %d: VMM region unavailable (%s)\n", __func__, device, err);
        return ok[device] = false;
    }
    ok[device] = ggml_cuda_vmm_region_grow(r, u);
    ggml_cuda_vmm_region_free(r);
    GGML_LOG_INFO("%s: device %d: VMM region %s (map unit %zu MiB, driver granularity %zu MiB)\n", __func__,
            device, ok[device] ? "available" : "REFUSED a one-unit map", u >> 20, g >> 20);
    return ok[device];
}

// -----------------------------------------------------------------------------------------------
// GGML_CUDA_VMM_REGION_SELFTEST: the primitive's oracle while nothing in the delivery consumes it yet.
// -----------------------------------------------------------------------------------------------

namespace {

__global__ void st_fill_kernel(uint32_t * p, size_t n, uint32_t v) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        p[i] = v;
    }
}

__global__ void st_check_kernel(const uint32_t * __restrict__ p, size_t n, uint32_t v, int * bad) {
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        if (p[i] != v) {
            atomicAdd(bad, 1);
        }
    }
}

bool st_units_are(const ggml_cuda_vmm_region * r, size_t n_units, uint32_t v, int * d_bad, const char * what) {
    const size_t unit = ggml_cuda_vmm_region_unit(r);
    const size_t n = n_units * unit / 4;
    (void) cudaMemset(d_bad, 0, sizeof(int));
    st_check_kernel<<<256, 256>>>((const uint32_t *) ggml_cuda_vmm_region_base(r), n, v, d_bad);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        (void) cudaGetLastError();
        GGML_LOG_ERROR("%s: %s: check kernel failed\n", __func__, what);
        return false;
    }
    int bad = 0;
    (void) cudaMemcpy(&bad, d_bad, sizeof(int), cudaMemcpyDeviceToHost);
    if (bad != 0) {
        GGML_LOG_ERROR("%s: %s: %d mismatching words (expected 0x%x)\n", __func__, what, bad, v);
        return false;
    }
    return true;
}

}  // namespace

bool ggml_cuda_vmm_region_selftest(int device) {
    // 6 units of 64 MiB = 384 MiB reserved, 3 mapped -- small enough to run at every start, and enough to
    // exercise the watermark contract, the stale-content clear, and a growth step of more than one unit.
    const size_t unit = (size_t) 64 << 20;
    const size_t n_units = 6;

    char err[128] = {0};
    ggml_cuda_vmm_region * r = ggml_cuda_vmm_region_create(device, n_units * unit, unit, err, sizeof(err));
    if (r == nullptr) {
        GGML_LOG_ERROR("%s: device %d: create failed (%s)\n", __func__, device, err);
        return false;
    }
    bool ok = true;
    int * d_bad = nullptr;
    if (cudaMalloc(&d_bad, sizeof(int)) != cudaSuccess) {
        (void) cudaGetLastError();
        ggml_cuda_vmm_region_free(r);
        GGML_LOG_ERROR("%s: device %d: scratch alloc failed\n", __func__, device);
        return false;
    }

    auto check = [&](bool cond, const char * what) {
        if (!cond) {
            GGML_LOG_ERROR("%s: device %d: FAILED: %s\n", __func__, device, what);
            ok = false;
        }
    };

    check(ggml_cuda_vmm_region_size(r) == n_units * unit, "reservation size");
    check(ggml_cuda_vmm_region_unit(r) == unit, "unit honoured");
    check(ggml_cuda_vmm_region_mapped(r) == 0, "a fresh reservation maps nothing");

    // grow one unit: it must be mapped AND cleared (a fresh mapping can hold stale bytes)
    check(ggml_cuda_vmm_region_grow(r, 1), "grow to one unit");
    check(ggml_cuda_vmm_region_mapped(r) == unit, "watermark is one unit after grow(1)");
    if (ok) { check(st_units_are(r, 1, 0u, d_bad, "unit 0 cleared on map"), "unit 0 cleared on map"); }

    // write a distinct value into unit 0, grow to three units, and prove unit 0 survived and 1..2 are clear
    st_fill_kernel<<<256, 256>>>((uint32_t *) ggml_cuda_vmm_region_base(r), unit / 4, 0x5a5a5a5au);
    check(cudaDeviceSynchronize() == cudaSuccess, "fill unit 0");
    check(ggml_cuda_vmm_region_grow(r, 3 * unit), "grow to three units");
    check(ggml_cuda_vmm_region_mapped(r) == 3 * unit, "watermark is three units");
    if (ok) {
        check(st_units_are(r, 1, 0x5a5a5a5au, d_bad, "unit 0 preserved across a grow"), "unit 0 preserved across a grow");
        const size_t two_units = 2 * unit / 4;
        (void) cudaMemset(d_bad, 0, sizeof(int));
        st_check_kernel<<<256, 256>>>((const uint32_t *) ((char *) ggml_cuda_vmm_region_base(r) + unit), two_units, 0u, d_bad);
        int bad = -1;
        (void) cudaDeviceSynchronize();
        (void) cudaMemcpy(&bad, d_bad, sizeof(int), cudaMemcpyDeviceToHost);
        check(bad == 0, "newly mapped units 1..2 are cleared");
    }

    // a smaller grow is a no-op, and growing past the reservation must refuse rather than abort
    check(ggml_cuda_vmm_region_grow(r, unit), "a smaller grow is a no-op");
    check(ggml_cuda_vmm_region_mapped(r) == 3 * unit, "watermark unchanged by a smaller grow");
    check(!ggml_cuda_vmm_region_grow(r, n_units * unit + unit), "growing past the reservation is refused");

    // SPAN growth: a KV buffer lays its tensors out consecutively (K0, V0, K1, V1, ...), so what a batch needs
    // is the first n_kv cells of EACH tensor -- disjoint spans -- and never the gaps between them.  A span must
    // map whole units inside itself only, must not re-clear an already mapped unit, and must round a range that
    // starts mid-unit outwards.
    check(ggml_cuda_vmm_region_is_mapped(r, 0, 3 * unit), "a prefix grow maps its units");
    check(!ggml_cuda_vmm_region_is_mapped(r, 0, 4 * unit), "a prefix grow leaves the next unit unmapped");
    check(ggml_cuda_vmm_region_grow_span(r, 4 * unit, unit), "span grow");
    check(ggml_cuda_vmm_region_is_mapped(r, 4 * unit, unit), "the span is mapped");
    check(!ggml_cuda_vmm_region_is_mapped(r, 3 * unit, unit), "the gap between the spans is NOT mapped");
    check(ggml_cuda_vmm_region_mapped(r) == 4 * unit, "4 of 6 units are mapped");
    // data in the spanning unit survives a repeat (a grow is not a re-clear)
    st_fill_kernel<<<256, 256>>>((uint32_t *) ((char *) ggml_cuda_vmm_region_base(r) + 4 * unit), unit / 4, 0xa5a5a5a5u);
    check(cudaDeviceSynchronize() == cudaSuccess, "fill the span");
    check(ggml_cuda_vmm_region_grow_span(r, 4 * unit, unit), "a repeated span grow is a no-op");
    if (ok) {
        check(st_units_are(r, 1, 0x5a5a5a5au, d_bad, "prefix unit 0 is unchanged by a span grow"), "prefix unit 0 unchanged");
        (void) cudaMemset(d_bad, 0, sizeof(int));
        st_check_kernel<<<256, 256>>>((const uint32_t *) ((char *) ggml_cuda_vmm_region_base(r) + 4 * unit), unit / 4,
                0xa5a5a5a5u, d_bad);
        int bad = -1;
        (void) cudaDeviceSynchronize();
        (void) cudaMemcpy(&bad, d_bad, sizeof(int), cudaMemcpyDeviceToHost);
        check(bad == 0, "the spanning unit's data survives a repeated grow");
    }
    // a range that starts mid-unit rounds outwards, and a range past the reservation is refused
    check(ggml_cuda_vmm_region_grow_span(r, 5 * unit + 1, 8), "a mid-unit span grow");
    check(ggml_cuda_vmm_region_is_mapped(r, 5 * unit + 1, 8), "a mid-unit span is mapped");
    check(!ggml_cuda_vmm_region_grow_span(r, 5 * unit, 2 * unit), "a span past the reservation is refused");
    check(!ggml_cuda_vmm_region_is_mapped(r, 0, n_units * unit), "the reserved tail stays unmapped");

    // MIXED BACKING (A2): the same reservation, a span backed by PINNED SYSTEM RAM instead of VRAM.  The
    // device must be able to WRITE it (the KV's set_rows path) and read it back -- that is the whole
    // capability the windowed KV rests on (wip/kv-residency/mixed-backing-probe.cc measured 28.4 GB/s).
    {
        const size_t off = 3 * unit;   // the gap the span checks left unmapped
        const bool mapped_host = ggml_cuda_vmm_region_map_span(r, off, unit, /*host=*/ true);
        check(mapped_host, "a HOST-backed span maps into the same reservation");
        if (mapped_host) {
            check(ggml_cuda_vmm_region_is_mapped(r, off, unit), "the host span reads as mapped");
            check(ggml_cuda_vmm_region_mapped_host(r) == unit, "the host-backed byte count");
            st_fill_kernel<<<256, 256>>>((uint32_t *) ((char *) ggml_cuda_vmm_region_base(r) + off), unit / 4, 0x3c3c3c3cu);
            check(cudaDeviceSynchronize() == cudaSuccess, "the device writes the host-backed span");
            (void) cudaMemset(d_bad, 0, sizeof(int));
            st_check_kernel<<<256, 256>>>((const uint32_t *) ((char *) ggml_cuda_vmm_region_base(r) + off), unit / 4,
                    0x3c3c3c3cu, d_bad);
            int bad = -1;
            (void) cudaDeviceSynchronize();
            (void) cudaMemcpy(&bad, d_bad, sizeof(int), cudaMemcpyDeviceToHost);
            check(bad == 0, "the device reads back what it wrote to the host-backed span");
            // a host span must not disturb its VRAM neighbour
            if (ok) { check(st_units_are(r, 1, 0x5a5a5a5au, d_bad, "VRAM unit 0 after the host span"), "VRAM unit 0 intact"); }
        }
    }

    // SLIDING WINDOW (A2/A3): the spill boundary must land ON a mapping boundary, and a flip must keep every
    // byte.  The first version of the slide assumed the boundary was unit-quantised (it is) and that mappings
    // were too (they were NOT: the VRAM extent was mapped to its exact byte end, and the first growth of a KV
    // tensor is sub-unit, so a coalesced mapping straddled the boundary and the next VRAM step flipped it back
    // WHOLE -- the window silently spilled nothing at all, `mapped_host` 0, VRAM unchanged, measured on a
    // 4B/hybrid KV).  The fix is at the source: the window's resident extent and every growth are rounded up
    // to whole units, so no mapping straddles the (unit-quantised) boundary.  These arms pin both halves: the
    // aligned flip moves exactly its mapping, and a request that still straddles degrades loudly and keeps
    // every byte (never a wrong answer).
    {
        const size_t u2 = (size_t) 8 << 20;
        ggml_cuda_vmm_region * w = ggml_cuda_vmm_region_create(device, 4 * u2, u2, err, sizeof(err));
        check(w != nullptr, "window selftest: create");
        if (w != nullptr) {
            // three adjacent pieces, which `vmm_insert` coalesces into ONE mapping [0, 3*u2)
            check(ggml_cuda_vmm_region_map_span(w, 0,      u2, false), "window: span piece 0");
            check(ggml_cuda_vmm_region_map_span(w, u2,     u2, false), "window: span piece 1");
            check(ggml_cuda_vmm_region_map_span(w, 2 * u2, u2, false), "window: span piece 2");
            st_fill_kernel<<<256, 256>>>((uint32_t *) ggml_cuda_vmm_region_base(w), 3 * u2 / 4, 0x12345678u);
            check(cudaDeviceSynchronize() == cudaSuccess, "window: fill the coalesced span");

            // an ALIGNED flip (a whole unit, the shape the window produces): exactly that unit changes kind
            check(ggml_cuda_vmm_region_ensure_span(w, u2, u2, true), "window: flip a whole mapping to host");
            check(ggml_cuda_vmm_region_mapped_host(w) == u2, "window: only the flipped mapping is host");
            check(ggml_cuda_vmm_region_ensure_span(w, u2, u2, false), "window: flip it back");
            check(ggml_cuda_vmm_region_mapped_host(w) == 0, "window: nothing is host after flipping back");

            // every byte survives both flips, in every unit
            for (size_t at = 0; at < 3 * u2; at += u2) {
                (void) cudaMemset(d_bad, 0, sizeof(int));
                st_check_kernel<<<256, 256>>>((const uint32_t *) ((char *) ggml_cuda_vmm_region_base(w) + at),
                        u2 / 4, 0x12345678u, d_bad);
                int bad = -1;
                (void) cudaDeviceSynchronize();
                (void) cudaMemcpy(&bad, d_bad, sizeof(int), cudaMemcpyDeviceToHost);
                check(bad == 0, "window: every byte survives a flip");
            }

            // a request that straddles a mapping (a caller bug -- a windowed KV always lands on the unit grid)
            // is REFUSED: flipping only part of a mapping is not safe on this driver, and flipping all of it
            // silently is how the window once mapped nothing to host
            check(!ggml_cuda_vmm_region_ensure_span(w, u2 + 4096, 4096, true), "window: a straddling request is refused");
            check(ggml_cuda_vmm_region_mapped_host(w) == 0, "window: a refused request changes no kind");
            for (size_t at = 0; at < 3 * u2; at += u2) {
                (void) cudaMemset(d_bad, 0, sizeof(int));
                st_check_kernel<<<256, 256>>>((const uint32_t *) ((char *) ggml_cuda_vmm_region_base(w) + at),
                        u2 / 4, 0x12345678u, d_bad);
                int bad = -1;
                (void) cudaDeviceSynchronize();
                (void) cudaMemcpy(&bad, d_bad, sizeof(int), cudaMemcpyDeviceToHost);
                check(bad == 0, "window: a refused request moves no byte");
            }
            ggml_cuda_vmm_region_free(w);
        }
    }

    // and it must survive its own teardown (with holes in the mapped set)
    ggml_cuda_vmm_region_free(r);
    (void) cudaFree(d_bad);

    GGML_LOG_INFO("%s: device %d: %s (reserved %zu MiB in %zu x %zu MiB units, prefix + span growth, "
                  "a HOST-backed span, aligned window flips (a partial one is REFUSED), the watermark contract "
                  "and clear-on-map verified)\n", __func__, device, ok ? "PASS" : "FAIL",
                  (n_units * unit) >> 20, n_units, unit >> 20);
    return ok;
}
