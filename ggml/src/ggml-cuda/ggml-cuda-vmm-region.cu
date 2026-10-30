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
};

namespace {

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

// Map `bytes` of fresh physical at `addr` and grant READWRITE to the owner, then each peer on its own
// (best effort -- a refused peer descriptor must not fail the mapping; the slab's helper documents the
// `-sm tensor` fault that taught that).  The handle is released after a successful map.
bool vmm_map(ggml_cuda_vmm_region * r, size_t addr_off, size_t bytes, bool host = false) {
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
    return true;
}

void vmm_unmap(ggml_cuda_vmm_region * r, size_t addr_off, size_t bytes) {
    if (bytes == 0) {
        return;
    }
    const CUdeviceptr addr = (CUdeviceptr) ((uintptr_t) r->base + addr_off);
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
    const vmm_mapping * m = vmm_at(r, off);
    if (m == nullptr || bytes > m->bytes - (off - m->off)) {
        return false;
    }
    if (end  != nullptr) { *end  = m->off + m->bytes; }
    if (host != nullptr) { *host = m->host; }
    return true;
}

// Insert a mapping and keep the table sorted and coalesced.
void vmm_insert(ggml_cuda_vmm_region * r, size_t off, size_t bytes, bool host) {
    size_t i = 0;
    while (i < r->maps.size() && r->maps[i].off < off) { ++i; }
    r->maps.insert(r->maps.begin() + (long) i, vmm_mapping{off, bytes, host});
    // coalesce with the neighbours when they touch and share a kind
    for (size_t k = 0; k + 1 < r->maps.size(); ) {
        if (r->maps[k].host == r->maps[k+1].host && r->maps[k].off + r->maps[k].bytes >= r->maps[k+1].off) {
            r->maps[k].bytes = std::max(r->maps[k].bytes, r->maps[k+1].off + r->maps[k+1].bytes - r->maps[k].off);
            r->maps.erase(r->maps.begin() + (long) (k+1));
            if (k > 0) { --k; }
            continue;
        }
        ++k;
    }
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
        // A fresh mapping can contain stale bytes (measured), so clear exactly what was just mapped.
        if (cudaMemset((char *) r->base + a, 0, n) != cudaSuccess) {
            (void) cudaGetLastError();
            return false;
        }
        a += n;
    }
    return true;
}

// Preserve a mapping's bytes across a kind change.  A remap is unmap + map, and the driver hands back fresh
// physical with no relation to what was there (measured: stale on device chunks, zero on host ones), so the
// bytes are staged through a pinned scratch.  Only the sliding window takes this path, once per unit per
// few thousand tokens, so a blocking round trip is the right trade for simplicity here.
bool vmm_remap(ggml_cuda_vmm_region * r, const vmm_mapping & m, bool host) {
    if (m.bytes > r->stage_bytes) {
        if (r->stage != nullptr) { (void) cudaFreeHost(r->stage); r->stage = nullptr; r->stage_bytes = 0; }
        void * st = nullptr;
        if (cudaMallocHost(&st, m.bytes) != cudaSuccess) {
            (void) cudaGetLastError();
            return false;
        }
        r->stage = st;
        r->stage_bytes = m.bytes;
    }
    // nothing may be in flight while the VA is unmapped (the VA stays valid for graphs, the *physical* does not)
    if (cudaDeviceSynchronize() != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    if (cudaMemcpy(r->stage, (const char *) r->base + m.off, m.bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    vmm_unmap(r, m.off, m.bytes);
    // drop the table entry by hand (we are holding a reference into the vector)
    const size_t off = m.off, bytes = m.bytes;
    const bool was_host = m.host;
    r->maps.erase(std::find_if(r->maps.begin(), r->maps.end(),
            [off](const vmm_mapping & e) { return e.off == off; }));
    r->mapped      -= bytes;
    r->mapped_host -= was_host ? bytes : 0;

    if (!vmm_map(r, off, bytes, host)) {
        return false;   // left unmapped: the caller's next batch will refuse loudly rather than read garbage
    }
    if (cudaMemcpy((char *) r->base + off, r->stage, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    vmm_insert(r, off, bytes, host);   // re-insert (and coalesce with its new neighbours of the same kind)
    return true;
}

bool ggml_cuda_vmm_region_ensure_span(ggml_cuda_vmm_region * r, size_t off, size_t bytes, bool host) {
    if (r == nullptr) {
        return false;
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
        if (const vmm_mapping * cur = vmm_at(r, a)) {
            if (cur->host == host) {
                a = cur->off + cur->bytes;    // this piece is already the KIND we need
                continue;
            }
            // The kind is wrong for the whole mapping: flip it, preserving its bytes.  Flipping the whole
            // mapping (not just this range) keeps the table one-entry-per-mapping and avoids a sub-range
            // unmap, which the driver refuses.
            if (!vmm_remap(r, *cur, host)) {
                return false;
            }
            continue;                          // `a` is unchanged; the entry is gone/rewritten
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

void ggml_cuda_vmm_region_free(ggml_cuda_vmm_region * r) {
    if (r == nullptr) {
        return;
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

    // and it must survive its own teardown (with holes in the mapped set)
    ggml_cuda_vmm_region_free(r);
    (void) cudaFree(d_bad);

    GGML_LOG_INFO("%s: device %d: %s (reserved %zu MiB in %zu x %zu MiB units, prefix + span growth, "
                  "a HOST-backed span, the watermark contract and clear-on-map verified)\n", __func__, device, ok ? "PASS" : "FAIL",
                  (n_units * unit) >> 20, n_units, unit >> 20);
    return ok;
}
