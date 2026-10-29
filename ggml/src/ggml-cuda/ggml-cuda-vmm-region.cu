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
#include <vector>

struct ggml_cuda_vmm_region {
    int      device  = -1;
    int      phys    = -1;
    size_t   size    = 0;   // the VA reservation
    size_t   unit    = 0;   // map granularity (mapping is always a whole number of units)
    size_t   n_units = 0;   // size / unit -- the bitmap's length
    size_t   mapped  = 0;   // bytes mapped (a count, not a watermark: the set can have holes)
    void *   base    = nullptr;
    // One byte per unit: is that unit mapped (and therefore cleared)?  `size / unit` is a few KiB even for a
    // 16 GiB region at 2 MiB units, and it makes `is_mapped()` a pure range scan with no driver call.
    std::vector<uint8_t> unit_map;
};

namespace {

size_t vmm_up(size_t x, size_t a) {
    return a == 0 ? x : ((x + a - 1) / a) * a;
}

// The backend already queried this once at init (`ggml_cuda_info().devices[].vmm_granularity`, the
// RECOMMENDED granularity) and latched whether the driver offers VMM at all, so there is no second query
// and no need for the `CUresult`/`CU_MEM_ALLOC_GRANULARITY_MINIMUM` macros -- which `vendors/hip.h` does
// not map (only the RECOMMENDED one), and adding them would touch a header every backend compiles.
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
bool vmm_map(ggml_cuda_vmm_region * r, size_t addr_off, size_t bytes) {
    CUmemAllocationProp prop = {};
    prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id   = r->phys;

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
    r->size   = vmm_up(bytes, u);
    r->n_units = r->size / u;
    r->unit_map.assign(r->n_units, 0);

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

bool ggml_cuda_vmm_region_grow_span(ggml_cuda_vmm_region * r, size_t off, size_t bytes) {
    if (r == nullptr) {
        return false;
    }
    if (bytes == 0) {
        return true;
    }
    if (off > r->size || bytes > r->size - off) {
        return false;   // outside the reservation -- the caller sized the region for its own extent
    }
    // Whole units only: a sub-range of a mapping cannot be unmapped later (VMM-PROBE T5), so the mapping
    // granularity is the unit and every span is rounded OUTWARDS to it.  Rounding outwards can only touch
    // more VA than asked for, never a byte outside the reservation (`last` is clamped).
    const size_t first = off / r->unit;
    size_t       last  = (off + bytes + r->unit - 1) / r->unit;
    if (last > r->n_units) {
        last = r->n_units;
    }
    for (size_t u = first; u < last; ++u) {
        if (r->unit_map[u] != 0) {
            continue;   // already mapped (and cleared once): never re-clear, data must survive
        }
        const size_t addr_off = u * r->unit;
        if (!vmm_map(r, addr_off, r->unit)) {
            // Leave what did map: those bytes are valid and cleared, so the caller can still run on a shorter
            // extent instead of falling over.
            return false;
        }
        r->unit_map[u] = 1;
        r->mapped += r->unit;
        // A fresh mapping can contain stale bytes (measured), so clear exactly what was just mapped.  The
        // write runs at VRAM speed (~490 GB/s measured, no fault storm), so this is not a real cost.
        if (cudaMemset((char *) r->base + addr_off, 0, r->unit) != cudaSuccess) {
            (void) cudaGetLastError();
            return false;
        }
    }
    return true;
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
    const size_t first = off / r->unit;
    size_t       last  = (off + bytes + r->unit - 1) / r->unit;
    if (last > r->n_units) {
        last = r->n_units;
    }
    for (size_t u = first; u < last; ++u) {
        if (r->unit_map[u] == 0) {
            return false;
        }
    }
    return true;
}

void ggml_cuda_vmm_region_clear(ggml_cuda_vmm_region * r, uint8_t value) {
    if (r == nullptr) {
        return;
    }
    for (size_t u = 0; u < r->n_units; ++u) {
        if (r->unit_map[u] == 0) {
            continue;
        }
        if (cudaMemset((char *) r->base + u * r->unit, value, r->unit) != cudaSuccess) {
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
    // A mapping is a whole unit and units are disjoint, so the mapped set is unmapped unit by unit -- there is
    // never a sub-range unmap (which the driver refuses, VMM-PROBE T5).
    for (size_t u = 0; u < r->n_units; ++u) {
        if (r->unit_map[u] != 0) {
            vmm_unmap(r, u * r->unit, r->unit);
            r->unit_map[u] = 0;
        }
    }
    r->mapped = 0;
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

    // and it must survive its own teardown (with holes in the mapped set)
    ggml_cuda_vmm_region_free(r);
    (void) cudaFree(d_bad);

    GGML_LOG_INFO("%s: device %d: %s (reserved %zu MiB in %zu x %zu MiB units, prefix + span growth, "
                  "the watermark contract and clear-on-map verified)\n", __func__, device, ok ? "PASS" : "FAIL",
                  (n_units * unit) >> 20, n_units, unit >> 20);
    return ok;
}
