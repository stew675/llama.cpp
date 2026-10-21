// Movable-boundary slab (OPEN 2 / r42): the per-device VMM reserve + work/arena boundary move.
// Split out of ggml-cuda.cu (an upstream hub); the API is declared in ggml-cuda-vmm.h.

#include "ggml-cuda-vmm.h"
#include "common.cuh"
#include "ggml-cuda.h"
#include "ggml-impl.h"
#include "moe-expert-cache.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <set>
#include <vector>

// Map physical into an already-reserved VA range.  No slab lock (no shared state).
//
// NOTE: the range is mapped as ONE mapping.  ROCm rejects `hipMemUnmap` of a SUB-range (measured:
// hipErrorInvalidValue), so a per-table TAIL prune would need one mapping per granularity unit -- and a
// 11776 MiB compute buffer mapped as ~5900 unit mappings then fails an unrelated `hipMemcpy2DAsync`
// (cpy.cu:479, hipErrorInvalidValue) in-tree, even though a standalone 6144-unit map of the same size
// succeeds.  That is why the design maps the whole slab ONCE and never gives physical back: the boundary
// moves at runtime instead, and the arena reclaim is per-table (the tables in the taken chunks).
static bool ggml_cuda_vmm_map_phys(int device, CUdeviceptr addr, size_t aligned, uint32_t * granted_out) {
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
    // Grant the OWNER first, then each peer ON ITS OWN: one refused peer descriptor would otherwise poison
    // the whole call.  That is exactly what used to happen -- a single all-descriptors call failed on the
    // second and third ~29 GiB slab (no free VRAM left in the peers for the driver to build a 29 GiB peer
    // mapping), the owner-only retry papered over it, and every peer copy into those slabs then either
    // returned `invalid argument` or faulted the GPU.  A refused peer keeps owner access only, and copies
    // between those devices stage through host memory (see `ggml_cuda_slab_peer_copy_ok`).
    if (cuMemSetAccess(addr, aligned, &access.front(), 1) != cudaSuccess) {
        (void) cudaGetLastError();
        (void) cuMemUnmap(addr, aligned);
        return false;
    }
    uint32_t granted = 1u << (uint32_t) phys;
    for (size_t i = 1; i < access.size(); i++) {
        if (cuMemSetAccess(addr, aligned, &access[i], 1) != cudaSuccess) {
            (void) cudaGetLastError();
            static std::set<std::pair<int,int>> warned;
            if (warned.insert({ device, (int) access[i].location.id }).second) {
                GGML_LOG_WARN("%s: device %d: the driver refused peer access to the %.2f GiB slab range for "
                              "device %d; copies between those two devices stage through host memory\n",
                              __func__, device, (double) aligned / (1ull << 30), (int) access[i].location.id);
            }
            continue;
        }
        granted |= 1u << (uint32_t) access[i].location.id;
    }
    if (granted_out != nullptr) {
        *granted_out = granted;
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
    // wip/slab-ring-region (0 -> narrow -> RB -> wide -> arena): the work region is split so the
    // always-resident NARROW (decode/verify) compute view owns [0, `narrow_floor`), the H2D staging ring
    // owns [`narrow_floor`, `narrow_floor + ring_region`), and the transient WIDE (prefill) compute view
    // owns [`narrow_floor + ring_region`, boundary).  The arena is always [boundary, mapped), i.e. above
    // the work, so the ring is NEVER part of the arena and arming it is an ordinary work-boundary move
    // (it evicts the COLDEST tables, exactly like a prefill grow).  When the wide view is dropped the
    // boundary shrinks back to `narrow_floor` and the ring + wide regions return to the arena.
    size_t                   narrow_floor = 0;  // the fixed, always-resident narrow size (0 = not established)
    // wip/first-prefill-latency: the wide (prefill) compute size, recorded by the context.  The arena is
    // sized against the FULL region, but the seed is capped at `narrow_floor + ring + wide_reserve_bytes`
    // -- the band a wide prefill immediately reclaims -- so nothing is pre-seeded only to be evicted; the
    // hole fills lazily during decode.  See `ggml_cuda_slab_seed_floor`.
    size_t                   wide_reserve_bytes = 0;
    // wip/moe-verify-fusions (narrow-2): a SECOND static narrow view -- the MTP draft context's compute
    // buffer -- is pinned at the TOP of the reserved VA: `[narrow2_off, size)`.  With MTP there are TWO
    // live compute buffers (the target's verify view and the draft context's), so without this they both
    // aliased at base 0 (measured: plain != MTP, near-tie flip at decode token 69).  Pinning narrow-2 at
    // the top of `size` (NOT `mapped`) lets `slab_extend` grow the arena only up to `narrow2_off`: nothing
    // moves and no arena table is evicted to make room.  It is never part of the work pool or the arena.
    size_t                   narrow2_floor = 0; // the pinned narrow-2 size (0 = not established)
    size_t                   narrow2_off   = 0; // = size - narrow2_floor, fixed once established
    int                      narrow2_live  = 0; // live narrow-2 views (diagnostic/guard)
    size_t                   ring_region  = 0;  // ring bytes above the narrow floor (0 = no ring)
    bool                     ring_armed   = false; // the wide view (hence the ring region) is reserved
    // wip/slab-transient-absorption: the WIDE-PHASE auxiliary region (the FA prefill staging arena), bump
    // allocated just above the boundary so it is reclaimed with the ring/wide band on the prefill -> decode
    // transition.  `wide_aux_off` is the base offset, `wide_aux_size` the bytes reserved (0 = none), and
    // `wide_aux_gen` changes whenever the region is released or moves, so a cached pointer can be validated
    // (the FA staging arena keeps one per stream and must not read a stale region).
    size_t                   wide_aux_off  = 0;
    size_t                   wide_aux_size = 0;
    size_t                   wide_aux_gen  = 0;
    // wip/slab-transient-absorption (step 2b-2): a batch of wide-aux frees collected from the PER-GRAPH
    // caches (MMB activation cache/slots, BF16 src1) and replayed highest-address first.  The region is a
    // bump stack that only pops its top, so freeing a group of blocks in its owners' order leaves holes and
    // the region then ratchets upward (measured: wide_aux grew to 13 GiB and the arena to 0 in one 80K
    // prefill).  Address order makes the whole group pop cleanly.
    // wip/slab-transient-absorption (step 2b-2): the LIVE blocks of the wide-aux region, `off -> size`
    // (slab-relative, unit-rounded).  A bump stack that only pops its exact top cannot survive a group of
    // blocks released in another order: every non-top free leaves a hole that is never reclaimed before the
    // wide drop, so the region ratchets upward (measured with the per-graph caches in it: 7.4 GiB of holes in
    // ONE 6.6k prefill, wide_aux 7392 MiB and 149 of 2135 frees unable to pop).  With the live set,
    // `wide_aux_size` is the extent of the HIGHEST live block: holes below it are bounded and cost nothing,
    // and a group of frees in any order still returns the region to the live high-water.
    std::map<size_t, size_t> wide_live;
    // TODO #70: the wide-aux GROWTH decision is made once per wide pass.  A refused grow used to be retried
    // on every allocation that needed the region -- a prefill asks ~61 times per ubatch (one per FA op), so a
    // starved region produced thousands of `SLAB_WIDE_AUX` attempts, each of which could evict a band of
    // arena tables (or evict one and then refuse on a racing boundary move) while a graph that still held
    // the previous placement was queued.
    //
    // The two refusal classes are NOT the same decision:
    //   * a VA-LIMIT refusal (`target > aux_limit`: the pinned narrow-2 region, or the top of the mapping)
    //     is structural -- `aux_limit` cannot grow within the pass and any later grow only raises the
    //     extent -- so it is latched for the rest of the pass;
    //   * a SNAPSHOT-RACE refusal (the boundary moved while `g_slab_mutex` was dropped for the eviction)
    //     is recoverable -- the racing thread is the MTP draft context's reserve and it has usually
    //     finished by the next request -- so it is retried, but only `GGML_CUDA_SLAB_WIDE_AUX_RACE_MAX`
    //     times per pass, because each attempt costs an arena-band eviction and the old code's unbounded
    //     retry is what faulted the graph that still held the previous placement.
    // `wide_aux_denied_want` is the smallest request refused in this pass (0 = nothing latched yet); a
    // request at least that large is refused without touching the layout, a smaller one may still fit (a
    // grow only ever raises the extent).  It -- and the race counter -- are cleared by `ring_set` on a pass
    // transition, whenever the boundary shrinks (which hands VA back to the region), and on slab (re-)init.
    size_t                   wide_aux_denied_want = 0;
    size_t                   wide_aux_denied_n    = 0;   // refusals recorded in this pass (diagnostic)
    size_t                   wide_aux_race_n      = 0;   // snapshot-race refusals in this pass (bounded)
    // wip/slab-transient-absorption (step 2b-2) DIAGNOSTIC: why the wide-aux region grows.
    size_t                   wide_n_pop    = 0;   // frees that shrank the region
    size_t                   wide_n_hole   = 0;   // frees of a block that was not live (stale/hole)
    size_t                   wide_n_inval  = 0;   // `work_alloc` released the region (gen bump)
    size_t                   wide_n_rebase = 0;   // the base was re-established after the stack emptied
    bool                     ring_requested = false; // the context classified the live ubatch as wide (prefill)
                                                //  (`ring_set`, n_tokens >= 64).  `work_alloc` places
                                                //  such a view above the ring even when its buffer is no
                                                //  wider than the narrow floor (see the classification there)
    size_t                   ring_used    = 0;  // bump cursor within the ring region
    int                      ring_slots   = 0;  // slots the ring region was sized for (capacity = region/slots)
    std::multiset<size_t>    work_needs;        // REQUESTED size of every live work view -- the floor the
                                                //  boundary may shrink to (see `work_alloc`)
    bool                     cache_unusable = false; // the slab was declined because `work + min arena`
                                                //  does not fit: the cache must stream instead
    // Issue #136 (item F): the slab was armed/wanted for this device but could not be CREATED at all.
    // `decline_reason` is a static-lifetime string for the diagnostic.  Either decline means the cache
    // must stream; it must never fall back to a raw arena (the pre-slab crash mode).
    bool                     declined       = false;
    const char *             decline_reason = nullptr;
    // PHYSICAL devices granted READWRITE on the whole mapped range (bit p; the owner is always set).  The
    // grant is best effort: with three ~29 GiB slabs there is no free VRAM left in the peers for the driver
    // to grant a range that large, and `cuMemSetAccess` refuses it.  A peer copy into a range that was NOT
    // granted FAULTS the GPU (`HSA_STATUS_ERROR_MEMORY_FAULT`, `__amd_rocclr_copyBuffer`, "Page not
    // present"), it does not return an error, so every cross-device copy asks `ggml_cuda_slab_peer_copy_ok`
    // first and stages through pinned host memory when the grants are incomplete.
    uint32_t                 peer_mask     = 0;
    std::map<size_t, size_t> arena_free;        // arena free runs: offset -> size (chunk-aligned)
};

static std::mutex    g_slab_mutex;
static ggml_cuda_slab g_slabs[GGML_CUDA_MAX_DEVICES];

// The slab exists for ONE reason: to let a wide prefill's work region and the MoE expert-cache arena
// coexist on a driver that will not coalesce.  With no arena to arbitrate (a dense model, or `-ncmoe 0`)
// the movable boundary serves nothing and the slab only parks nearly all of the card's free VRAM in an
// "arena" that will never hold a table.  It is therefore ARMED by the MoE preflight, which runs before the
// context and only for models that actually have host-resident expert weights; the env var alone can no
// longer switch it on for a dense model.  `GGML_CUDA_SLAB=0` still kills it outright.
static bool g_slab_armed = false;

// wip/moe-verify-fusions: when false, COMPUTE buffers bypass the slab (raw cudaMalloc from the slab
// reserve).  The A/B arm (`GGML_CUDA_SLAB_NARROW2=0`) sets it for the MTP draft context so its compute
// buffer cannot alias the target's narrow verify buffer at slab base 0 (the session's "quick fix").
static bool g_slab_compute_enabled = true;

// wip/moe-verify-fusions (narrow-2): when true, the COMPUTE buffer being allocated belongs to the MTP
// draft context and is routed to the slab's second pinned narrow region (the top of the reserved VA).
// Set/cleared around the draft's reserve and alloc_graph calls by `llama_context`.
static bool g_slab_compute_narrow2 = false;

// when true, the COMPUTE buffer being allocated is one the slab never serves (the MTP draft's second scheduler):
// it bypasses the slab and the chunk quantization.  Set/cleared around that allocation by `llama_context`.
static bool g_slab_compute_plain = false;

// `GGML_CUDA_SLAB_NARROW2=0` is the A/B arm: the draft's compute buffers then bypass the slab entirely
// (raw cudaMalloc), which is the session's quick fix.  Default on.
static bool ggml_cuda_slab_narrow2_enabled() {
    static int on = -1;
    if (on < 0) {
        const char * e = getenv("GGML_CUDA_SLAB_NARROW2");
        on = e != nullptr ? atoi(e) : 1;
    }
    return on != 0;
}

// TODO #70: how many times ONE wide pass may retry a wide-aux grow whose layout snapshot was invalidated by
// another thread (the MTP draft context's reserve) before the pass gives up and lets the callers fall back to
// their capped raw allocations.  A retry costs an arena-band eviction, so the bound is small; 4 covers the one
// racing reserve a session actually performs with room to spare, and a bounded retry is deliberately kept (a
// blanket latch would write the pass off after one benign race).  No knob: not a tuning axis.
#define GGML_CUDA_SLAB_WIDE_AUX_RACE_MAX 4

// wip/slab-ring-region: the H2D staging-ring hole the slab must carve (0 = the ring stays a cudaMalloc
// outside the slab).  Set once by the model preflight, before the slab is created, from the largest
// host-resident weight table (so the per-slot capacity covers every upload the ring will see).
static size_t g_slab_ring_region_bytes = 0;
static size_t ggml_cuda_slab_up(size_t x, size_t a);   // defined below (used by the ring-floor pin)
void ggml_cuda_slab_set_ring_region(size_t bytes) {
    g_slab_ring_region_bytes = bytes;
}

// wip/slab-ring-region / first-prefill-latency: the chunk-aligned ring region to CARVE from the current
// request (0 when there is no room -- the ring then stays on the cudaMalloc path).  Used only at the point
// the region is actually carved into the work layout.  Caller holds `g_slab_mutex`.
static size_t ggml_cuda_slab_ring_candidate(const ggml_cuda_slab & s, size_t narrow_floor) {
    size_t ring = g_slab_ring_region_bytes;
    if (ring > 0) {
        ring = ggml_cuda_slab_up(ring, s.chunk);
        if (narrow_floor + ring + s.chunk > s.mapped) {
            ring = 0;   // no room for a meaningful region: leave the ring on the cudaMalloc path
        }
    }
    return ring;
}

// The base offset of the wide (prefill) view: narrow = [0, narrow_floor), ring = [narrow_floor,
// narrow_floor + ring_region), wide = [narrow_floor + ring_region, boundary), arena = [boundary, mapped).
//
// `s.ring_region` is the region the slab ACTUALLY reserved.  `g_slab_ring_region_bytes` is only the
// current REQUEST and is re-armed after the carve (the context applies its auto-size cap): using it here
// moves the wide view below the carved ring and the wide compute buffer overwrites the H2D staging ring
// (measured r74: ring carved 1216 MiB, request re-armed to 975 MiB -> wide placed at 1792 instead of 1984,
// a GPU VM fault in `k_get_rows_float` on the 35B-A3B + MTP path).  Placement must follow the carve.
static size_t ggml_cuda_slab_wide_offset(const ggml_cuda_slab & s) {
    return s.narrow_floor + s.ring_region;
}

// wip/slab-ring-region: pin the always-resident narrow (decode/verify) floor.  The context knows the real
// decode/verify compute size (from the narrow re-reserve), which is more reliable than inferring it from the
// reserve sizes -- a decode view can be LARGER than the first small reserve (measured: the field's target
// decode is 1792 MiB while the first small reserve is 768 MiB, and treating 1792 as a wide view kept the
// boundary high so the ring region was never reclaimed).  Only honored before the first wide view is placed
// above the ring, so the ring's VA does not move.
void ggml_cuda_slab_set_narrow_floor(int device, size_t bytes) {
    std::lock_guard<std::mutex> lock(g_slab_mutex);
    ggml_cuda_slab & s = g_slabs[device];
    if (!s.inited) {
        return;
    }
    // The `split_only` probe reports the LAYOUT size; the real compute allocation is rounded up to whole
    // `GGML_COMPUTE_BUFFER_CHUNK_MIB` units plus one spare unit (`ggml_vbuffer_chunk_alloc_size`).  Apply
    // the same formula so the pinned floor matches the size the first narrow reserve will actually request
    // (otherwise the floor is a chunk short and that reserve is misclassified as wide).
    static const size_t cchunk = []() {
        const char * e = getenv("GGML_COMPUTE_BUFFER_CHUNK_MIB");
        return (size_t) (e ? atoll(e) : 256) << 20;
    }();
    size_t nf = ggml_cuda_slab_up(bytes, s.chunk);
    if (cchunk > 0) {
        nf = ggml_cuda_slab_up(bytes, cchunk) + cchunk;
    }
    if (nf <= s.narrow_floor) {
        return;   // not a raise: nothing to apply and nothing to report
    }
    if (s.ring_used > 0) {
        // The ring holds blocks, so its VA must not move.  The floor is only an estimate at this point;
        // `work_alloc` raises it from the real graph the next time a narrow view is placed while the ring
        // is idle, so this is not fatal -- but a raise that arrives here means the estimate was too low, and
        // that is worth knowing (it is what kept the field's decode views above the ring).
        GGML_LOG_WARN("%s: device %d: narrow floor raise to %.1f MiB ignored (ring in use: %.1f MiB of %.1f MiB, "
                      "floor %.1f MiB); the band is learned from the next idle-ring narrow view instead\n",
                      __func__, device, (double) nf / (1ull << 20), (double) s.ring_used / (1ull << 20),
                      (double) s.ring_region / (1ull << 20), (double) s.narrow_floor / (1ull << 20));
        return;
    }
    s.narrow_floor = nf;
    s.ring_region = ggml_cuda_slab_ring_candidate(s, nf);
    s.ring_slots  = ggml_backend_cuda_context::h2d_stage_slots();
    s.ring_used   = 0;
}

// wip/first-prefill-latency: record the wide (prefill) compute size.  The arena is sized against the FULL
// region (the boundary is NOT raised), but this lets the seed stop at `narrow_floor + ring + wide` -- the
// band a wide prefill immediately reclaims -- so the seeded experts are never the ones about to be
// evicted.  The band's tables are left unseeded and fill lazily during decode.  The offset comes from
// `ggml_cuda_slab_wide_offset` (the CARVED ring region), never from the mutable request.
void ggml_cuda_slab_reserve_wide(int device, size_t bytes) {
    std::lock_guard<std::mutex> lock(g_slab_mutex);
    ggml_cuda_slab & s = g_slabs[device];
    if (!s.inited || bytes == 0) {
        return;
    }
    if (bytes > s.wide_reserve_bytes) {
        s.wide_reserve_bytes = bytes;
    }
}

// The offset below which the seed must NOT run: the reclaim band `[narrow_floor, narrow_floor + ring +
// wide)` a wide prefill takes back.  This is exactly the wide placement's layout, so the seed boundary and
// the wide placement cannot disagree.  Uses the CARVED ring region, never the mutable request.  Tables
// whose arena starts below it are the coldest (heat-ordered last) and are seeded only lazily.  0 when not
// established.
size_t ggml_cuda_slab_seed_floor(int device) {
    std::lock_guard<std::mutex> lock(g_slab_mutex);
    const ggml_cuda_slab & s = g_slabs[device];
    if (!s.inited || s.narrow_floor == 0) {
        return 0;
    }
    const size_t wide_off = ggml_cuda_slab_wide_offset(s);
    return wide_off + (s.wide_reserve_bytes > 0 ? ggml_cuda_slab_up(s.wide_reserve_bytes, s.chunk) : 0);
}

static size_t ggml_cuda_slab_chunk_bytes() {
    static size_t chunk_mib = SIZE_MAX;
    if (chunk_mib == SIZE_MAX) {
        const char * env = getenv("GGML_CUDA_SLAB_CHUNK_MIB");
        chunk_mib = env != NULL ? (size_t) atoll(env) : 64;
    }
    return chunk_mib * 1024 * 1024;
}

// fit-slab-accounting revival (G3): the MTP/DFlash draft context's VRAM, passed to the MoE preflight
// before the slab is armed.  The reserve uses it because the draft is created AFTER the slab.
static size_t g_slab_aux_reserve_bytes = 0;
void ggml_cuda_slab_set_aux_reserve(size_t bytes) {
    g_slab_aux_reserve_bytes = bytes;
}

// Forward declaration: the reserve policy reads the (single-sourced) headroom.
size_t ggml_cuda_slab_headroom_bytes();

static size_t ggml_cuda_slab_reserve_bytes(size_t total_b) {
    static size_t reserve_mib = SIZE_MAX;
    if (reserve_mib == SIZE_MAX) {
        const char * env = getenv("GGML_CUDA_SLAB_RESERVE_MIB");
        if (env != NULL) {
            reserve_mib = (size_t) atoll(env);
        } else {
            // fit-slab-accounting revival (G3): size the reserve from what actually appears AFTER the
            // slab, not a flat guess.  The slab is born during `sched_reserve`, and `llama_context`
            // allocates the KV cache BEFORE that (`create_memory` at llama-context.cpp:543 vs
            // `sched_reserve` at :609), so the reserve does NOT need to cover the weights or the KV --
            // they are already resident and are counted in `free_b`.  It only has to cover the
            // allocations that follow: the MTP draft context (the `aux_reserve_bytes` the preflight was
            // given, ~3.7 GiB when a draft is present, 0 otherwise) plus the slab's own headroom (the
            // workspaces hipBLASLt and the FA staging draw on).  A flat max(8192, total/4) still charged
            // a no-draft run 8 GiB, so under partial offload `want` fell below `work + min_arena` and the
            // slab DECLINED -- the cache streamed from the host (measured on a 32 GiB card, Q8_0 35B-A3B,
            // `-ncmoe 16`: 47 t/s, vs 55-59 t/s once the reserve fits).  `GGML_CUDA_SLAB_RESERVE_MIB`
            // remains the explicit override.
            const size_t headroom_mib = ggml_cuda_slab_headroom_bytes() / (1024 * 1024);
            const size_t aux_mib      = g_slab_aux_reserve_bytes / (1024 * 1024);
            reserve_mib = std::max<size_t>(headroom_mib + aux_mib, (size_t) 1024);
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
// DEFAULT: **640 MiB per GPU, plus 128 MiB for every additional GPU** (wip/slab-transient-absorption,
// 2026-10-18): 1 GPU -> 768, 2 -> 896, 3 -> 1024 MiB.  See `ggml_cuda_slab_headroom_bytes`.
//
// 2026-10-18: the base moved 512 -> 640 (+128 on every config).  The 2-GPU MTP measurement
// (`Qwen3.8-Flash-Next` IQ4_NL, `-ncmoe 48`, 131k ctx, 73k-token prefill) left the cards at **148 MiB free
// in the decode steady state** at 768 -- the decode-phase graph captures' pool temporaries are outside the
// slab by design and cached for the run, and the reduce temp takes its 400 MiB out of the same cushion.
// The old table below still measures correctly; every value is the OLD default's excess.  With 896 the same
// run settles at ~276 MiB (measured).  Keep raising the base rather than adding a per-device term: the
// decode pool high-water is a per-DEVICE cost, not a per-extra-GPU one (the per-GPU term is for the
// multi-GPU peer/reduce transients, which is where the 90 MiB/extra-device figure came from).
//
// The headroom is the STEADY-STATE FREE VRAM kept outside the slab: the cushion every allocation that cannot
// live in the slab draws on.  Since the transient work it is (measured) only `T` = the hipBLASLt/Tensile
// workspace (104 MiB, library-owned), the Q8_1 arena, and the transients that are not slab-served -- every
// other prefill transient now lives in the slab (H2D ring, FA staging, QSA3/GDN scratch, the calibration
// probe, the workspace pool's large temporaries) and the two per-graph caches are released at the graph end
// (wip/slab-transient-absorption §10.6-§10.8).  It is NOT slack to be maximised: it is permanently withheld
// from the arena, and the arena is where the expert cache lives (~1 pt of hit rate per GiB moved).
//
// T IS PER DEVICE (each card runs its own pools, Q8_1 arena and rocBLAS workspace), which is why the default
// scales with the device count instead of being flat.  Measured 2026-10-17 (derived draft reserve,
// `GGML_CUDA_SLAB_STATS=1`; "excess" = free VRAM outside the slab at the worst arm/drop of the run):
//
//   headroom | field GSQ seesaw (-ub 4096, 1 GPU) | wide IQ4_NL (-ub 6144) | 2-GPU tensor | 2-GPU layer
//   ---------+-------------------------------------+-----------------------+---------------+-------------
//       512  | floor 214.5 MiB                     | drop 471 MiB          | -             | -
//       600  | floor 346.5 MiB                     | -                     | -             | 209.5 MiB/dev
//       640  | floor 346.5 MiB  <- 1-GPU default   | ~559 (T=41)           | -             | -
//       768  | -                                   | -                     | -             | 401.5 MiB/dev <- 2-GPU default
//      1024  | floor 714.5 MiB                     | drop 983 MiB          | 1552.5 /dev   | 657.5 MiB/dev
//      2048  | -                                   | -                     | 2576.5 /dev   | 1681.5 MiB/dev
//
// So T ~= 300 MiB per device plus ~90 per extra device, and the shipped default leaves ~350-400 MiB of excess
// on every measured config (the target was ~300).  A FLAT 600 was measured and REJECTED: the 2-GPU
// `-sm layer` case drops to 209.5 MiB there, because T there is ~390, not ~300 -- that is the whole reason the
// default scales with the device count.  (512 flat is 214.5 on the 1-GPU field seesaw: also under target.)
//
// The effective quantum is the slab CHUNK (64 MiB): the reserve is `headroom + aux` mapped in chunk units, so
// values inside one chunk of each other measure identically (600 and 640 are the same on 1 GPU).
//
// A thin headroom can fail SILENTLY rather than loudly (see the 2048 table in
// `archive/work/fit-slab-accounting/README.md` §2.2: `!!!!!!!!`, 475 t/s, and the older `hipModuleLoad failed`
// abort at `-ub 4096 -c 163860`), which is why the cushion is kept at ~1.2x T per device rather than at the
// measured value.
//
// `-sm layer` with a partially host-resident MoE on more than one device raises this further
// (`ggml_cuda_slab_headroom_at_least_mib(6144)` from `alloc_all_locked`, the r36 255k needle floor) -- that
// floor only ever RAISES the value and only when the env is unset, so a lower default cannot weaken it.  A
// FULLY RESIDENT model (no `-ncmoe`) has no slab and no arena at all, so nothing is sized there.
//
// `GGML_CUDA_SLAB_HEADROOM_MIB` sets the value VERBATIM (no per-GPU addition), like every other explicit
// override; the per-GPU scaling applies only to the default.  The value is the steady-state free VRAM kept
// outside the slab after `ggml_cuda_slab_extend` reclaims the reserve.
#define GGML_CUDA_SLAB_HEADROOM_MIB_DEFAULT 640   // base; the default is 640 + 128 x n_gpu
#define GGML_CUDA_SLAB_HEADROOM_MIB_PER_GPU 128   // per device, so 1->768, 2->896, 3->1024 MiB

static size_t g_slab_headroom_min_mib = 0;
void ggml_cuda_slab_headroom_at_least_mib(size_t mib) {
    g_slab_headroom_min_mib = mib;
}
size_t ggml_cuda_slab_headroom_bytes() {
    static size_t mib = SIZE_MAX;
    static bool   env_set = false;
    if (mib == SIZE_MAX) {
        const char * env = getenv("GGML_CUDA_SLAB_HEADROOM_MIB");
        env_set = env != NULL;
        mib = env_set ? (size_t) atoll(env) : GGML_CUDA_SLAB_HEADROOM_MIB_DEFAULT;
    }
    // The outside transients are PER DEVICE, so the default grows with the number of cards (`640 + 128 x n`,
    // i.e. 768 / 896 / 1024 MiB for 1 / 2 / 3 GPUs); an explicit `GGML_CUDA_SLAB_HEADROOM_MIB` is taken
    // verbatim (an operator asks for a total, not a per-card value).
    size_t eff = mib;
    if (!env_set) {
        // `640 + 128 x n_devices`: 1 GPU -> 768, 2 -> 896, 3 -> 1024 MiB.
        const int n_phys = std::max(1, ggml_cuda_info().physical_device_count);
        eff += (size_t) n_phys * GGML_CUDA_SLAB_HEADROOM_MIB_PER_GPU;
    }
    if (!env_set && g_slab_headroom_min_mib > eff) {
        eff = g_slab_headroom_min_mib;
    }
    return eff * 1024 * 1024;
}

// Public MiB form, for the cache's own fallback message.
size_t ggml_cuda_slab_min_arena_mib() {
    return ggml_cuda_slab_min_arena_bytes() / (1024 * 1024);
}

static size_t ggml_cuda_slab_up(size_t x, size_t a) {
    return a * ((x + a - 1) / a);
}

// wip/moe-verify-fusions: the arena's VA ceiling and usable top when a narrow-2 region is pinned at the
// top of the reserved VA.  Before narrow-2 exists the ceiling is the whole reserved VA (`size`); after, it
// stops at `narrow2_off` so neither `slab_extend` nor `arena_alloc` can touch the pinned region.  The
// usable top is the mapped extent, clamped to the ceiling (narrow-2 is established only while
// `narrow2_off >= mapped`, so the clamp is defensive).
static size_t ggml_cuda_slab_arena_limit(const ggml_cuda_slab & s) {
    return s.narrow2_floor > 0 ? s.narrow2_off : s.size;
}
static size_t ggml_cuda_slab_arena_top(const ggml_cuda_slab & s) {
    const size_t lim = ggml_cuda_slab_arena_limit(s);
    return s.mapped < lim ? s.mapped : lim;
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

#if defined(GGML_USE_HIP)
// The env kill switch, read once.  Nonzero = the user wants the movable-boundary slab.
static int ggml_cuda_slab_env() {
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
    return enabled;
}
#endif

bool ggml_cuda_slab_enabled() {
#if defined(GGML_USE_HIP)
    // Armed only for host-resident-expert models (see `g_slab_armed`): a dense model has no arena, so the
    // slab would just withhold the free VRAM the model/KV/workspaces need.
    return ggml_cuda_slab_env() != 0 && g_slab_armed;
#else
    return false;   // RDNA/ROCm-scoped
#endif
}

// The env kill switch alone (ignoring `g_slab_armed`): "the user asked for the slab".  Needed because
// `ggml_cuda_slab_enabled()` is false until the MoE preflight arms it, so the not-armed diagnostic cannot
// use it to tell "configured off" from "configured on but never armed" (issue #136, item B).
bool ggml_cuda_slab_configured() {
#if defined(GGML_USE_HIP)
    return ggml_cuda_slab_env() != 0;
#else
    return false;
#endif
}

// Is the slab live on THIS device?  (The gate can be on while a device failed to create one.)
bool ggml_cuda_slab_active(int device) {
    std::lock_guard<std::mutex> lock(g_slab_mutex);
    return g_slabs[device].inited;
}

// May `[dst, dst+nbytes)` be the destination of a raw cross-device peer copy?  A peer copy into a slab range
// the driver's grants do not cover FAULTS the GPU instead of returning an error (the grant is best effort:
// three ~29 GiB slabs leave the peers no free VRAM to build it), and the slab check is process-wide because
// a copy fails while ANY live slab's grants are incomplete -- even between two fully granted devices.  A
// destination outside every slab is always fine (plain memory peer copies stage inside the driver).
// false => stage the copy through pinned host memory.
bool ggml_cuda_slab_peer_copy_ok(const void * dst, size_t nbytes) {
    if (dst == nullptr || nbytes == 0) {
        return true;
    }
    std::lock_guard<std::mutex> lock(g_slab_mutex);
    int      n_live  = 0;
    uint32_t pending = 0;                         // device ids that have a live slab
    uint32_t common  = ~0u;                       // INTERSECT of every live slab's grant mask
    bool     in_slab = false;
    const uintptr_t lo = (uintptr_t) dst;
    const uintptr_t hi = lo + nbytes - 1;
    for (int d = 0; d < GGML_CUDA_MAX_DEVICES; d++) {
        const ggml_cuda_slab & s = g_slabs[d];
        if (!s.inited || s.base == 0 || s.mapped == 0) {
            continue;
        }
        n_live++;
        pending |= 1u << (uint32_t) d;
        common  &= s.peer_mask;
        const uintptr_t s_lo = (uintptr_t) s.base;
        const uintptr_t s_hi = s_lo + s.mapped - 1;
        if (hi >= s_lo && lo <= s_hi) {
            in_slab = true;
        }
    }
    if (n_live == 0 || !in_slab) {
        // Plain `cudaMalloc` memory: a peer copy there stages inside the driver and always works.
        return true;
    }
    // The destination IS in a slab, so its peer-access grants decide it -- and the check is PROCESS-WIDE,
    // not per-pair: measured on three ~28 GiB slabs (masks {0,1,2} / {0,1} / {0}), a peer copy between the
    // TWO FULLY MUTUALLY GRANTED devices (0 -> 1) still fails `invalid argument` while a third slab's grants
    // are incomplete, and succeeds once all grants complete (three 20 GiB slabs).  A wrong "yes" faults the
    // GPU (uncatchable, process-killing), so the query is pessimistic on purpose.
    return (common & pending) == pending;
}

// The slab declined because `work estimate + cache floor` does not fit on this device (see
// `ggml_cuda_slab_min_arena_bytes`).  The expert cache must then stream from the host rather than build an
// arena that a wide prefill would have to evict wholesale.
bool ggml_cuda_slab_cache_unusable(int device) {
    std::lock_guard<std::mutex> lock(g_slab_mutex);
    return g_slabs[device].cache_unusable;
}

// Issue #136 (item F): the slab was armed for a host-expert model but could not be created on this device.
bool ggml_cuda_slab_declined(int device) {
    std::lock_guard<std::mutex> lock(g_slab_mutex);
    return g_slabs[device].declined;
}

const char * ggml_cuda_slab_decline_reason(int device) {
    std::lock_guard<std::mutex> lock(g_slab_mutex);
    return g_slabs[device].decline_reason;
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
    if (!s.inited) {
        return 0;
    }
    // wip/moe-verify-fusions: the arena is [boundary, arena_top) -- never past a pinned narrow-2 region.
    // wip/slab-ring-region: the cache's real allocations are bounded by `arena_free`, so this is the CAP
    // it sizes its budget against and it must reflect what is allocatable NOW (an inflated cap here makes
    // `alloc_table_locked` fail and self-disable).
    const size_t top = ggml_cuda_slab_arena_top(s);
    const size_t total = top > s.boundary ? top - s.boundary : 0;
    return total;
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
    // wip/moe-verify-fusions: never map into a pinned narrow-2 region -- the VA left for the arena.
    const size_t lim = ggml_cuda_slab_arena_limit(s);
    if (s.mapped >= lim) {
        return false;                                    // no VA left below narrow-2 (or the full reserved VA)
    }
    const size_t room = lim - s.mapped;                  // VA left unmapped by the creation reservation
    if (add > room) {
        add = room;
    }
    if (add < s.chunk) {
        return false;                                    // less than a chunk to gain: not worth a mapping
    }
    uint32_t granted = 0;
    if (!ggml_cuda_vmm_map_phys(device, (CUdeviceptr) ((uintptr_t) s.base + s.mapped), add, &granted)) {
        GGML_LOG_WARN("%s: device %d: could not map a further %.2f GiB into the slab; the arena stays at "
                      "%.2f GiB (raise GGML_CUDA_SLAB_HEADROOM_MIB if this repeats)\n",
                      __func__, device, (double) add / (1ull << 30),
                      (double) (s.mapped - s.boundary) / (1ull << 30));
        return false;
    }
    // A peer granted the first range is not automatically granted this one: keep a bit only when BOTH
    // ranges were granted, so the query can never over-report (a wrong "yes" is a GPU fault, a wrong "no"
    // only costs a staged copy).
    s.peer_mask &= granted;
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
        s.declined       = true;
        s.decline_reason = "the device does not report VMM (virtual memory management) support";
        return false;
    }
    s.device = device;
    s.chunk  = ggml_cuda_slab_chunk_bytes();
    s.unit   = ggml_cuda_slab_unit_bytes(device);
    if (s.chunk == 0 || s.unit == 0 || s.chunk % s.unit != 0) {
        s.declined       = true;
        s.decline_reason = "the slab chunk/unit geometry is invalid";
        return false;
    }
    // `cuMemAddressReserve` requires a power-of-two alignment; a hand-set chunk that is not one would
    // otherwise fail after the fact, so decline now and let the plain path serve the buffer.
    if ((s.chunk & (s.chunk - 1)) != 0) {
        s.declined       = true;
        s.decline_reason = "GGML_CUDA_SLAB_CHUNK_MIB is not a power of two";
        return false;
    }
    ggml_cuda_set_device(device);
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        (void) cudaGetLastError();
        s.declined       = true;
        s.decline_reason = "cudaMemGetInfo failed";
        return false;
    }
    const size_t reserve = ggml_cuda_slab_reserve_bytes(total_b);
    if (free_b <= reserve + 2 * s.chunk) {
        s.declined       = true;
        s.decline_reason = "there is not enough free VRAM for the slab reserve plus two chunks";
        return false;
    }
    const size_t want = (free_b - reserve) / s.chunk * s.chunk;   // (W + A), chunk-aligned
    const size_t w    = ggml_cuda_slab_up(work_min, s.chunk);
    if (w == 0 || w + s.chunk > want) {
        s.declined       = true;
        s.decline_reason = "the estimated work buffer does not fit after the reserve";
        return false;
    }
    // HARD MINIMUM (see `ggml_cuda_slab_min_arena_bytes`): the work estimate plus a usable cache floor.
    // Report it and decline the slab -- `cache_unusable` then makes the expert cache stream from the host
    // instead of building an arena that a wide prefill would have to evict wholesale (the r21 abort).
    // wip/slab-ring-region: the ring region sits ABOVE the narrow floor and its size only needs to be
    // reserved against the arena budget (it is work space while the wide view is reserved).
    size_t ring = g_slab_ring_region_bytes;
    if (ring > 0) {
        ring = ggml_cuda_slab_up(ring, s.chunk);
        if (ring + s.chunk > want) {
            ring = 0;   // no room for a meaningful hole: leave the ring on the cudaMalloc path
        }
    }
    const size_t min_arena = ggml_cuda_slab_min_arena_bytes();
    if (want < w + min_arena + ring) {
        s.cache_unusable = true;
        // TODO #96: name every knob that can make this fit, not only the reserve.  The reporter in issue
        // #214 had 3072 MiB available with work 768 + floor 2048 = 2816, i.e. the auto-sized ~1.6 GiB ring
        // hole was the binding term -- and the message named `GGML_CUDA_SLAB_RESERVE_MIB` alone.
        GGML_LOG_ERROR("%s: device %d: the movable-boundary slab does NOT fit the MoE expert cache here: "
                       "it needs the estimated work buffer (%.0f MiB) plus a %.0f MiB cache floor"
                       "%s (%.0f MiB), but only %.0f MiB is available after the %zu MiB reserve (free %.0f MiB of "
                       "%.0f MiB total).  Not using the slab; the experts will STREAM from the host (the stock "
                       "-ncmoe path).  Knobs, largest term first: MOE_EXPERT_CACHE_MIB / "
                       "GGML_CUDA_SLAB_MIN_ARENA_MIB lower the floor, GGML_CUDA_SLAB_RING_MIB lowers or "
                       "disables the H2D-ring hole, and GGML_CUDA_SLAB_RESERVE_MIB lowers the reserve if it "
                       "is larger than the weights/KV need.\n",
                       __func__, device, (double) w / (1024 * 1024), (double) min_arena / (1024 * 1024),
                       ring > 0 ? " plus the H2D-ring hole" : "", (double) ring / (1024 * 1024),
                       (double) want / (1024 * 1024), ggml_cuda_slab_reserve_mib(total_b),
                       (double) free_b / (1024 * 1024), (double) total_b / (1024 * 1024));
        s.declined       = true;
        s.decline_reason = "there is not enough room for the work region plus the minimum expert-cache arena";
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
    if (!ggml_cuda_vmm_map_phys(device, s.base, s.mapped, &s.peer_mask)) {
        (void) cuMemAddressFree(s.base, s.size);
        return false;
    }
    s.boundary     = w;
    s.narrow_floor = 0;   // established on the first narrow reserve or by `ggml_cuda_slab_set_narrow_floor`
    s.narrow2_floor = 0;  // established lazily on the draft context's first narrow compute allocation
    s.narrow2_off   = 0;
    s.narrow2_live  = 0;
    s.ring_region  = 0;
    s.ring_armed   = false;
    s.ring_used    = 0;
    s.wide_aux_off  = 0;
    s.wide_aux_size = 0;
    s.wide_aux_gen  = 0;
    s.wide_aux_denied_want = 0;   // TODO #70: this pass's wide-aux growth has not been refused
    s.wide_aux_denied_n    = 0;
    s.wide_aux_race_n      = 0;
    s.ring_slots   = ggml_backend_cuda_context::h2d_stage_slots();
    s.arena_free.clear();
    s.arena_free[w] = s.mapped - w;
    s.inited = true;
    // TEMP diagnostic (wip/moe-verify-fusions): the full layout accounting.
    GGML_LOG_WARN("SLAB_INIT dev=%d free=%.2f GiB reserve=%.2f GiB headroom=%.2f GiB aux=%.2f GiB "
                  "size(VA)=%.2f GiB mapped=%.2f GiB boundary=%.2f GiB va_spare=%.2f GiB chunk=%.1f MiB ring_req=%.2f GiB\n",
                  device, (double) free_b/(1ull<<30), (double) ggml_cuda_slab_reserve_bytes(total_b)/(1ull<<30),
                  (double) ggml_cuda_slab_headroom_bytes()/(1ull<<30), (double) g_slab_aux_reserve_bytes/(1ull<<30),
                  (double) s.size/(1ull<<30), (double) s.mapped/(1ull<<30), (double) s.boundary/(1ull<<30),
                  (double) (s.size - s.mapped)/(1ull<<30), (double) s.chunk/(1024*1024),
                  (double) g_slab_ring_region_bytes/(1ull<<30));
    GGML_LOG_WARN("%s: device %d: slab %.2f GiB (chunk %.1f MiB): work %.2f GiB + arena %.2f GiB "
                  "(%.2f GiB reserve, %.2f GiB VA spare for `ggml_cuda_slab_extend`; ring region %s)\n",
                  __func__, device, (double) s.mapped / (1ull << 30), (double) s.chunk / (1024 * 1024),
                  (double) s.boundary / (1ull << 30), (double) (s.mapped - s.boundary) / (1ull << 30),
                  (double) ggml_cuda_slab_reserve_bytes(total_b) / (1ull << 30),
                  (double) (s.size - s.mapped) / (1ull << 30),
                  g_slab_ring_region_bytes > 0 ? "pending (above the narrow floor)" : "off");
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
// wip/moe-verify-fusions (narrow-2): establish the MTP draft's pinned compute region at the top of the
// reserved VA and map its physical.  Idempotent: the draft reserves at construction, DROPS those buffers
// (`llama_context_drop_compute_buffers`, before the target's first prefill) and lazily re-reserves on its
// first real (n_max+1-token) graph, so this is called repeatedly.  Returns the region's base offset, or 0
// when it cannot be established (no room for the work region + a usable arena above `boundary`, the top
// overlaps already-mapped physical, or `ggml_cuda_vmm_map_phys` failed); the caller then falls back to
// the plain cudaMalloc path for that buffer.
//
// The size is FIXED at the first establishment: narrow-2 is anchored at the top (`off = size - floor`), so
// a later, larger request would move a live view's base.  In practice the first request is the draft's
// `sched_reserve` pp layout, which is its widest graph, so the region never needs to grow.
// Decide where the narrow-2 (draft compute) region goes.  Returns `off` (0 = nowhere) and reports whether
// the arena already holds mapped physical in [off, mapped) that has to be evicted first: the region is
// pinned at the TOP of the reserved VA, so anything the arena mapped above `off` is in the way.
//
// Two placements, in order of preference:
//   * ABOVE `mapped`: the VA the creation reservation left unmapped (`reserve - headroom`, i.e. the
//     draft-context estimate).  Nothing has to be moved and no arena byte is spent -- the MTP-draft case.
//   * BELOW `mapped`: the arena's mapped area.  The arena is the lowest-priority VRAM consumer (r20), so
//     it yields: the tables in [off, mapped) are evicted and the arena tops out at `off` from then on
//     (`ggml_cuda_slab_arena_limit`).  The physical between the two only stays idle -- bounded by the
//     shortfall of the estimate -- and is never handed out again.
//     NEEDED because a TENSOR-SPLIT drafter's per-device compute buffer is much larger than an MTP
//     draft's: measured 1280 MiB for a DFlash2 drafter under `-sm tensor` against the estimator's 512 MiB
//     tier, i.e. bigger than the whole unmapped spare, which used to leave the draft on the cudaMalloc
//     path with the slab holding the VRAM (`graph_reserve: failed to allocate compute buffers`).
static size_t ggml_cuda_slab_narrow2_pin_locked(int device, ggml_cuda_slab & s, size_t bytes, bool * evict) {
    *evict = false;
    if (!s.inited || bytes == 0) {
        return 0;
    }
    const size_t nf2 = ggml_cuda_slab_up(bytes, s.chunk);
    if (s.narrow2_floor == 0) {
        if (nf2 == 0 || nf2 >= s.size) {
            return 0;
        }
        const size_t off = s.size - nf2;
        const size_t min_arena = ggml_cuda_slab_min_arena_bytes();
        // The work region (`boundary`, which includes the wide prefill view and the ring) and a usable
        // arena must both fit below narrow-2, and enough VA must be left for the mapping.
        if (off < s.boundary + min_arena + s.chunk) {
            GGML_LOG_WARN("%s: device %d: no room for a %.2f GiB narrow-2 region above the work region "
                          "(%.2f GiB) and a %.2f GiB arena floor in a %.2f GiB slab (mapped %.2f GiB); "
                          "the draft compute buffer stays on the cudaMalloc path\n",
                          __func__, device, (double) nf2 / (1ull << 30), (double) s.boundary / (1ull << 30),
                          (double) min_arena / (1ull << 30), (double) s.size / (1ull << 30),
                          (double) s.mapped / (1ull << 30));
            return 0;
        }
        // mapped physical above `off` is in the way: it has to be evicted before the pin (the eviction
        // re-enters the slab through `arena_free`, so it cannot run under this lock).
        *evict = off < s.mapped;
        return off;
    }
    if (nf2 > s.narrow2_floor) {
        GGML_LOG_WARN("%s: device %d: narrow-2 request %.2f GiB exceeds the pinned %.2f GiB and the region "
                      "cannot grow without moving a live view; falling back\n",
                      __func__, device, (double) nf2 / (1ull << 30), (double) s.narrow2_floor / (1ull << 30));
        return 0;
    }
    return s.narrow2_off;
}

// Commit a pin decided above: map the region when it lies above `mapped`, otherwise cap the arena at it.
static void ggml_cuda_slab_narrow2_commit_locked(int device, ggml_cuda_slab & s, size_t bytes) {
    if (!s.inited || s.narrow2_floor != 0 || bytes == 0) {
        return;
    }
    const size_t nf2 = ggml_cuda_slab_up(bytes, s.chunk);
    if (nf2 == 0 || nf2 >= s.size) {
        return;
    }
    const size_t off = s.size - nf2;
    if (off >= s.mapped) {
        ggml_cuda_set_device(device);
        uint32_t granted = 0;
        if (!ggml_cuda_vmm_map_phys(device, (CUdeviceptr) ((uintptr_t) s.base + off), nf2, &granted)) {
            GGML_LOG_WARN("%s: device %d: could not map the %.2f GiB narrow-2 region; the draft compute "
                          "buffer stays on the cudaMalloc path\n",
                          __func__, device, (double) nf2 / (1ull << 30));
            return;
        }
        s.peer_mask &= granted;
    } else {
        // The arena gave the range up: drop every free run at or above the new top, so a later
        // `arena_alloc` cannot hand out memory that now belongs to the draft's compute view.
        for (auto it = s.arena_free.begin(); it != s.arena_free.end(); ) {
            const size_t lo = it->first;
            const size_t hi = lo + it->second;
            if (lo >= off) {
                it = s.arena_free.erase(it);
                continue;
            }
            if (hi > off) {
                it->second = off - lo;
            }
            ++it;
        }
    }
    s.narrow2_floor = nf2;
    s.narrow2_off   = off;
    GGML_LOG_WARN("%s: device %d: narrow-2 (draft compute) pinned at [%.2f, %.2f] GiB "
                  "(%.2f GiB, %s); the arena now tops out at %.2f GiB\n",
                  __func__, device, (double) off / (1ull << 30), (double) s.size / (1ull << 30),
                  (double) nf2 / (1ull << 30),
                  off >= s.mapped ? "unmapped VA spare" : "taken from the arena (tables evicted)",
                  (double) (ggml_cuda_slab_arena_top(s) - s.boundary) / (1ull << 30));
}

// The draft context's narrow-2 compute allocation.  Runs OUTSIDE `g_slab_mutex` because pinning may have to
// evict arena tables first (that callback re-enters the slab).
static void * ggml_cuda_slab_narrow2_alloc(int device, size_t need) {
    size_t off = 0;
    bool   evict = false;
    {
        std::lock_guard<std::mutex> lock(g_slab_mutex);
        ggml_cuda_slab & s = g_slabs[device];
        if (!s.inited) {
            if (s.declined) {
                return nullptr;
            }
            if (!ggml_cuda_slab_init_locked(device, s, need)) {
                return nullptr;
            }
        }
        // narrow-2 is ONE region: two simultaneous draft views would alias each other exactly as the
        // pre-fix defect did at base 0.  The MTP draft's buffers are sequential (the construction
        // reserve is released before the live verify buffer), but fall back to cudaMalloc rather than
        // alias if that ever changes.
        if (s.narrow2_live > 0) {
            GGML_LOG_WARN("%s: device %d: a narrow-2 view is already live (need %.2f MiB); falling back "
                          "to cudaMalloc to avoid an alias\n",
                          __func__, device, (double) need / (1ull << 20));
            return nullptr;
        }
        off = ggml_cuda_slab_narrow2_pin_locked(device, s, need, &evict);
    }
    if (off == 0) {
        return nullptr;
    }
    if (evict) {
        size_t base = 0, top = 0;
        {
            std::lock_guard<std::mutex> lock(g_slab_mutex);
            ggml_cuda_slab & s = g_slabs[device];
            base = (size_t) s.base;
            top  = s.mapped;
        }
        moe_cache_evict_slab_range(device, (void *) (uintptr_t) (base + off),
                                          (void *) (uintptr_t) (base + top));
    }
    void * ret = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_slab_mutex);
        ggml_cuda_slab & s = g_slabs[device];
        ggml_cuda_slab_narrow2_commit_locked(device, s, need);
        if (s.narrow2_floor != 0 && ggml_cuda_slab_up(need, s.chunk) <= s.narrow2_floor) {
            s.narrow2_live++;
            ret = (void *) (uintptr_t) ((char *) s.base + s.narrow2_off);
            if (ggml_cuda_work_alloc_debug()) {
                GGML_LOG_WARN("NARROW2_ALLOC dev=%d need=%.1f MiB off=%.1f MiB live=%d\n",
                    device, (double) need/1048576.0, (double) s.narrow2_off/1048576.0, s.narrow2_live);
            }
        }
    }
    return ret;
}

// Public form: pin narrow-2 from a known (worst-case) narrow size.  Called by the draft context after its
// reserve so the region covers the widest layout the scheduler reserved, not just the first graph.
void ggml_cuda_slab_narrow2_floor(int device, size_t bytes) {
    // Same placement decision as the allocation path, but never grows/evicts for a live view: if the
    // region is not pinned yet this pins it from the measured size.
    size_t off = 0;
    bool   evict = false;
    {
        std::lock_guard<std::mutex> lock(g_slab_mutex);
        ggml_cuda_slab & s = g_slabs[device];
        if (!s.inited) {
            return;
        }
        off = ggml_cuda_slab_narrow2_pin_locked(device, s, bytes, &evict);
    }
    if (off == 0) {
        return;
    }
    if (evict) {
        size_t base = 0, top = 0;
        {
            std::lock_guard<std::mutex> lock(g_slab_mutex);
            ggml_cuda_slab & s = g_slabs[device];
            base = (size_t) s.base;
            top  = s.mapped;
        }
        moe_cache_evict_slab_range(device, (void *) (uintptr_t) (base + off),
                                          (void *) (uintptr_t) (base + top));
    }
    std::lock_guard<std::mutex> lock(g_slab_mutex);
    ggml_cuda_slab_narrow2_commit_locked(device, g_slabs[device], bytes);
}

// This is a MULTISET, not a flag: more than one compute buffer is live at a time (the main context and the
// MTP draft context each own one), so a release must not make the slab believe the work region is idle --
// doing that is what corrupted the no-drop runs (a release shrank the boundary under a live view and the
// arena re-took chunks that view was still using: MTP acceptance 0.010, `////` output).
void ggml_cuda_slab_work_release(int device, size_t off, size_t need) {
    std::lock_guard<std::mutex> lock(g_slab_mutex);
    ggml_cuda_slab & s = g_slabs[device];
    if (!s.inited) {
        return;
    }
    // wip/moe-verify-fusions: a narrow-2 view owns no boundary/arena contribution -- the region is pinned
    // at the top of the reserved VA and never enters `work_needs`.
    if (s.narrow2_floor > 0 && off == s.narrow2_off) {
        if (s.narrow2_live > 0) {
            s.narrow2_live--;
        }
        if (ggml_cuda_work_alloc_debug()) {
            GGML_LOG_WARN("NARROW2_RELEASE dev=%d need=%.1f MiB off=%.1f MiB live=%d\n",
                device, (double) need/1048576.0, (double) off/1048576.0, s.narrow2_live);
        }
        return;
    }
    // Remove this view's boundary contribution (`off + up(need)`), the same value `work_alloc` inserted.
    const size_t contrib = off + ggml_cuda_slab_up(need, s.chunk);
    const auto it = s.work_needs.find(contrib);
    if (it != s.work_needs.end()) {
        s.work_needs.erase(it);
    }
    // TEMP diagnostic (wip/moe-verify-fusions): trace alloc/release overlap.
    if (ggml_cuda_work_alloc_debug()) {
        GGML_LOG_WARN("WORK_RELEASE dev=%d off=%.1f MiB need=%.1f MiB found=%d live_needs=%zu boundary=%.1f narrow_floor=%.1f\n",
            device, (double) off/1048576.0, (double) need/1048576.0, (int)(it != s.work_needs.end()),
            s.work_needs.size(), (double) s.boundary/1048576.0, (double) s.narrow_floor/1048576.0);
    }
}

// The work pool's base for a `need`-byte compute buffer, moving the boundary up if required.  The arena
// tables in the taken range are evicted (the callback frees their slabs back through `arena_free`).
//
// wip/slab-ring-region (0 -> narrow -> RB -> wide -> arena): TWO stable bases are handed out.  A NARROW
// request (the decode/verify compute view) always gets `s.base`; a WIDE request (the prefill view) always
// gets `s.base + narrow_floor + ring_region`, i.e. above the fixed narrow region and the ring.  The
// always-resident narrow view never moves; the ring and wide regions are transient and return to the arena
// when the wide view is dropped.  The ring is never part of the arena, so arming it is an ordinary
// boundary move (evicting the COLDEST tables), not an independent arena eviction.
void * ggml_cuda_slab_work_alloc(int device, size_t need) {
    if (g_slab_compute_narrow2) {
        // the draft context's compute buffers: a pinned region of their own (may evict arena tables, so it
        // runs outside the lock below)
        return ggml_cuda_slab_narrow2_alloc(device, need);
    }
    size_t lo = 0, hi = 0;
    bool   move = false;
    size_t ret_off = 0;
    CUdeviceptr base = 0;
    {
        std::lock_guard<std::mutex> lock(g_slab_mutex);
        ggml_cuda_slab & s = g_slabs[device];
        if (!s.inited) {
            if (s.declined) {
                return nullptr;   // declined once: do not retry init or repeat the diagnostic
            }
            if (!ggml_cuda_slab_init_locked(device, s, need)) {
                return nullptr;
            }
            s.work_needs.insert(ggml_cuda_slab_up(need, s.chunk));
            // Provisional base 0: the first reserve is the WIDE one, before the narrow floor exists.
            return (void *) s.base;
        }
        // Establish the fixed narrow floor: the first reserve smaller than the boundary (the
        // prefill -> decode drop re-reserves the decode/verify layout there).  Its ring region then sits
        // just above it, and the wide view above that.  The live wide view (the target/draft context keeps
        // one) stays at base 0 until the graph allocator re-creates it above the ring on the next wide
        // reserve; `work_needs` still floors the boundary at the wide need meanwhile.
        if (s.narrow_floor == 0 && need < s.boundary) {
            s.narrow_floor = ggml_cuda_slab_up(need, s.chunk);
            s.ring_region = ggml_cuda_slab_ring_candidate(s, s.narrow_floor);
            s.ring_slots  = ggml_backend_cuda_context::h2d_stage_slots();
            s.ring_used   = 0;
        }

        // wip/slab-ring-region (prefill arming): the context marks a prefill ubatch (`n_tokens >= 64`) via
        // `ring_set(true)`.  Such a view must go ABOVE the ring even when its buffer is no wider than the
        // narrow (decode/verify) floor -- the 35B-A3B and GSQ configs have a size-independent compute
        // layout, so a size-only test left the prefill at base 0, `boundary == narrow_floor`, and the
        // carved ring region was never armed (`ring_slot_capacity() == 0`): the H2D staging fell back to a
        // capped cudaMalloc and the prefill streamed.  Only force it when a region was actually carved, so
        // a no-ring model keeps the plain size classification.
        const bool   ring_wide = s.ring_region > 0 && s.ring_requested && s.narrow_floor > 0;
        // A view that is NOT a requested wide (prefill) pass belongs to the always-resident narrow band even
        // when it is wider than the floor established so far.  The ring region is only carved while a wide
        // pass is live, so when it is not (`boundary <= narrow_floor` and the ring is disarmed) a wider
        // narrow view simply grows the band; pushing it above the ring instead costs `ring_region` of arena,
        // and the prefill -> decode drop (which records the band it just reserved) then re-drops on every
        // pass.  Measured on the field config (MTP verify/draft graph 2304 MiB against a 1792 MiB floor):
        // 66 tables re-armed and ~3.65 GiB/device evicted per decode step, decode 17 t/s instead of 40+.
        // While a wide view IS live the ring VA must not move, so that case keeps the above-the-ring
        // placement (a later decode pass drops the wide view first, then the band grows here).
        // A live wide view is the only thing that makes the ring region part of the reserved layout:
        // `ring_armed` is set exactly when the boundary covers the carved ring.  `ring_used` is NOT a
        // usable test (it is a bump cursor that stays at the end after a prefill) and neither is
        // `boundary > narrow_floor` (the boundary can sit a chunk or two above the floor for an
        // unrelated narrow view) -- treating either as "the ring may not move" stopped the band from
        // being learned on a single device: measured gfx1100 `-ncmoe 40`, floor 1280 MiB against a
        // 1536 MiB decode view, one arena eviction per prefill -> decode transition.
        const bool   narrow   = s.narrow_floor > 0 && !ring_wide && (!s.ring_armed || need <= s.narrow_floor);
        if (narrow && need > s.narrow_floor) {
            // Learn the band from the real graph: the floor is estimated before the first carve from a
            // reserve that is not necessarily the widest narrow layout (the `n_rs_batch` probe measures
            // 1792 MiB where the MTP decode/verify graph needs 2304), and `ggml_cuda_slab_set_narrow_floor`
            // cannot correct it afterwards because the ring's VA is frozen once it holds blocks.  Here no
            // wide view is live and the ring is idle, so moving its base is safe.
            s.narrow_floor = ggml_cuda_slab_up(need, s.chunk);
            s.ring_region  = ggml_cuda_slab_ring_candidate(s, s.narrow_floor);
            s.ring_slots   = ggml_backend_cuda_context::h2d_stage_slots();
            s.ring_used    = 0;
            s.ring_armed   = false;
        }
        const size_t wide_off = ggml_cuda_slab_wide_offset(s);

        // The boundary may shrink only to the TALLEST live view's boundary contribution.  Each view's
        // contribution is `base_off + up(need)` (a narrow view at base 0, a wide view above the ring), so
        // the multiset stores that directly and a view placed before the narrow floor existed cannot
        // inflate the floor after the fact.
        ret_off = narrow ? 0 : wide_off;
        // TEMP diagnostic (wip/moe-verify-fusions): show the base offset each compute buffer gets.
        if (ggml_cuda_work_alloc_debug()) {
            GGML_LOG_WARN("WORK_ALLOC dev=%d need=%.1f MiB narrow=%d ret_off=%.1f MiB narrow_floor=%.1f ring=%.1f boundary=%.1f\n",
                device, (double) need/1048576.0, (int) narrow, (double) ret_off/1048576.0,
                (double) s.narrow_floor/1048576.0, (double) s.ring_region/1048576.0, (double) s.boundary/1048576.0);
        }
        // wip/slab-transient-absorption: the wide-phase auxiliary region sits just above the boundary,
        // i.e. above the wide view this request is about to place.  If the view now reaches into it,
        // release the region so the view cannot overwrite it.  The previous graph has already been
        // enqueued (the FA staging arena re-acquires the region on its next staging pass), and the
        // generation bump invalidates any pointer it cached.
        if (s.wide_aux_off != 0 && ret_off + ggml_cuda_slab_up(need, s.chunk) > s.wide_aux_off) {
            s.wide_aux_off  = 0;
            s.wide_aux_size = 0;
            s.wide_live.clear();
            s.wide_aux_gen++;
            s.wide_n_inval++;
        }
        const size_t contrib = ret_off + ggml_cuda_slab_up(need, s.chunk);
        size_t floor = s.narrow_floor;
        if (s.wide_aux_size > 0) {
            const size_t aux_end = s.wide_aux_off + s.wide_aux_size;
            if (aux_end > floor) {
                floor = aux_end;
            }
        }
        if (!s.work_needs.empty()) {
            const size_t mx = *s.work_needs.rbegin();
            if (mx > floor) {
                floor = mx;
            }
        }
        if (contrib > floor) {
            floor = contrib;
        }
        const size_t target = ggml_cuda_slab_up(floor, s.chunk);
        if (s.wide_aux_off != 0 && target < s.wide_aux_off) {
            // The boundary is about to shrink below the (empty but retained) wide-aux base, so those
            // bytes go back to the arena: drop the base so the next wide aux allocation re-bases.
            s.wide_aux_off  = 0;
            s.wide_aux_size = 0;
            s.wide_live.clear();
            s.wide_aux_gen++;
        }
        base    = s.base;
        // wip/moe-verify-fusions: never grow the work pool into a pinned narrow-2 region (the draft's
        // compute VA).  A request that does not fit falls back to cudaMalloc rather than corrupt it.
        if (target > ggml_cuda_slab_arena_limit(s)) {
            GGML_LOG_WARN("%s: device %d: work request %.2f MiB (target %.2f GiB) does not fit below the "
                          "pinned narrow-2 region at %.2f GiB; falling back to cudaMalloc\n",
                          __func__, device, (double) need / (1ull << 20), (double) target / (1ull << 30),
                          (double) s.narrow2_off / (1ull << 30));
            return nullptr;
        }
        if (target <= s.boundary) {
            // Shrink to the live need so the slack goes back to the arena (the decode layout should not hold
            // the wide region).  Safe while a view is live: the target never goes below the floor.
            if (target < s.boundary) {
                const size_t old = s.boundary;
                s.boundary = target;
                s.wide_aux_denied_want = 0;   // TODO #70: the VA above the boundary grew; allow one new grow
                s.wide_aux_denied_n    = 0;
                s.wide_aux_race_n      = 0;
                // ISSUE #187: a captured CUDA/HIP graph bakes the addresses of the work region's tensors
                // into the executable graph.  The bytes above `target` have just gone back to the arena,
                // so any graph captured against the old boundary must be recaptured before its next
                // replay -- `mem_gen` is the existing invalidation for exactly this ("recapture a graph
                // whose pool temporaries or scratch buffers were freed since its capture").  This is the
                // prefill -> decode drop: `ggml_backend_sched_drop_buffers` releases the wide compute
                // view and the narrow re-reserve shrinks the boundary here.  Without the bump a
                // 4-stream MTP decode replayed a graph whose KV `set_rows` source indices pointed into
                // the released wide region and faulted with
                // HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION.
                ggml_cuda_graph_mem_freed(device);
                if (ggml_cuda_work_alloc_debug()) {
                    GGML_LOG_WARN("WORK_SHRINK dev=%d old=%.1f MiB new=%.1f MiB (captured graphs invalidated)\n",
                            device, (double) old/1048576.0, (double) target/1048576.0);
                }
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
            s.ring_armed = s.ring_region > 0 && wide_off > s.narrow_floor && s.boundary >= wide_off;
            s.work_needs.insert(contrib);
            return (void *) (uintptr_t) ((char *) s.base + ret_off);
        }
        lo   = s.boundary;
        hi   = target;
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
            const size_t wide_off = s.narrow_floor + s.ring_region;
            s.ring_armed = s.ring_region > 0 && wide_off > s.narrow_floor && s.boundary >= wide_off;
            s.work_needs.insert(ret_off + ggml_cuda_slab_up(need, s.chunk));
        }
        return (void *) (uintptr_t) ((char *) base + ret_off);
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

// wip/slab-ring-region: the H2D staging ring.
//
// The ring is the sub-region `[narrow_floor, narrow_floor + ring_region)` of the work region, i.e. ABOVE the
// always-resident narrow (decode/verify) view and BELOW the transient wide (prefill) view.  It is not part
// of the arena, so there is no arm/disarm and no arena eviction: it is reserved exactly when the wide view
// is (`boundary >= narrow_floor + ring_region`), and when the wide view is dropped the boundary shrinks to
// `narrow_floor` and the ring region returns to the arena as its coldest space.  `arena_alloc_transient`
// bump-allocates the slots within it; the ring calls it once per slot with the full per-slot capacity, so
// each slot has a stable VA.
size_t ggml_cuda_slab_ring_slot_capacity(int device) {
    std::lock_guard<std::mutex> lock(g_slab_mutex);
    const ggml_cuda_slab & s = g_slabs[device];
    if (!s.inited || !s.ring_armed || s.ring_region == 0 || s.ring_slots <= 0) {
        return 0;
    }
    return (s.ring_region / (size_t) s.ring_slots) / s.unit * s.unit;
}

// The size of the reserved H2D staging-ring hole (0 when none was carved).  Unlike
// `ggml_cuda_slab_ring_slot_capacity`, this is valid BEFORE the ring is armed: the narrow -> wide drop
// in `llama_context` runs on the narrow side of the transition, when the hole exists but is disarmed.
size_t ggml_cuda_slab_ring_region(int device) {
    std::lock_guard<std::mutex> lock(g_slab_mutex);
    const ggml_cuda_slab & s = g_slabs[device];
    if (!s.inited) {
        return 0;
    }
    return s.ring_region;
}

// wip/slab-transient-absorption (F5, step 0): was the ring hole requested for this model?  Distinct from
// `ring_region > 0` (which is the CARVE): a requested-but-not-carved hole must refuse the ring fallback,
// while `GGML_CUDA_SLAB_RING_MIB=0` (requested == false) keeps the pre-slab cudaMalloc ring.
bool ggml_cuda_slab_ring_requested() {
    return g_slab_ring_region_bytes > 0;
}

// wip/slab-transient-absorption: one-line steady-state accounting for the campaign's headroom budget.
// Gated on GGML_CUDA_SLAB_STATS (the ring arm/disarm transitions and the destructor call it); it reports
// the layout and the live transient sizes so "what still lives outside the slab" is measured, not assumed.
// Caller holds `g_slab_mutex`.
static void ggml_cuda_slab_log_stats_locked(int device, const char * tag) {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_SLAB_STATS");
        return e != nullptr && atoi(e) != 0;
    }();
    if (!on) {
        return;
    }
    const ggml_cuda_slab & s = g_slabs[device];
    const size_t mapped       = s.mapped;
    const size_t boundary     = s.boundary;
    const size_t arena        = s.mapped > s.boundary ? s.mapped - s.boundary : 0;
    const size_t narrow_floor = s.narrow_floor;
    const size_t ring         = s.ring_region;
    const size_t narrow2      = s.narrow2_floor;
    const size_t aux          = s.wide_aux_size;
    const bool   inited       = s.inited;
    size_t free_b = 0, total_b = 0;
    // cudaMemGetInfo reports the free VRAM of the CURRENT device, so it must be selected here: without
    // this, every device's line repeats the first device's figure and a per-device shortfall is invisible.
    ggml_cuda_set_device(device);
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        (void) cudaGetLastError();
    }
    ggml_cuda_free_min_sample(device, free_b);
    GGML_LOG_WARN("SLAB_STATS %s dev=%d slab=%d free=%.1f MiB free_min=%.1f MiB total=%.1f MiB mapped=%.1f MiB "
                  "work=%.1f MiB (narrow=%.1f ring=%.1f wide_aux=%.1f narrow2=%.1f) arena=%.1f MiB "
                  "pool=%.1f MiB pool_wide=%.1f MiB/%zu refused=%.1f MiB outside_max=%.1f MiB "
                  "outside_failed=%.1f MiB headroom_cfg=%.1f MiB aux_cfg=%.1f MiB "
                  "wide[n_pop=%zu hole=%zu inval=%zu rebase=%zu aux_refused=%zu]\n",
                  tag, device, (int) inited,
                  (double) free_b / 1048576.0, (double) ggml_cuda_free_min_bytes(device) / 1048576.0,
                  (double) total_b / 1048576.0, (double) mapped / 1048576.0,
                  (double) boundary / 1048576.0, (double) narrow_floor / 1048576.0,
                  (double) ring / 1048576.0, (double) aux / 1048576.0, (double) narrow2 / 1048576.0,
                  (double) arena / 1048576.0,
                  (double) ggml_cuda_pool_mapped_bytes(device) / 1048576.0,
                  (double) ggml_cuda_pool_wide_bytes(device) / 1048576.0,
                  ggml_cuda_pool_wide_blocks(device),
                  (double) ggml_cuda_pool_wide_refused(device) / 1048576.0,
                  (double) ggml_cuda_outside_alloc_max_bytes(device) / 1048576.0,
                  (double) ggml_cuda_outside_fail_max_bytes(device) / 1048576.0,
                  (double) ggml_cuda_slab_headroom_bytes() / 1048576.0,
                  (double) g_slab_aux_reserve_bytes / 1048576.0,
                  s.wide_n_pop, s.wide_n_hole, s.wide_n_inval, s.wide_n_rebase, s.wide_aux_denied_n);
}

void ggml_cuda_slab_log_stats(int device, const char * tag) {
    std::lock_guard<std::mutex> lock(g_slab_mutex);
    ggml_cuda_slab_log_stats_locked(device, tag);
}

void * ggml_cuda_slab_arena_alloc_transient(int device, size_t size) {
    if (size == 0) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_slab_mutex);
    ggml_cuda_slab & s = g_slabs[device];
    if (!s.inited || !s.ring_armed || s.ring_region == 0) {
        return nullptr;
    }
    const size_t want = ggml_cuda_slab_up(size, s.unit);
    if (s.ring_used + want > s.ring_region) {
        return nullptr;   // region exhausted (sized for `ring_slots` full-capacity slots, so unexpected)
    }
    void * p = (void *) (uintptr_t) ((char *) s.base + s.narrow_floor + s.ring_used);
    s.ring_used += want;
    return p;
}

// wip/slab-transient-absorption: the WIDE-PHASE auxiliary region (currently the FA prefill staging arena).
//
// It is a bump allocation placed immediately ABOVE the live work boundary -- which already holds the wide
// (prefill) compute view -- and sized on demand.  Because it sits in the wide band, it is reserved only
// while a wide prefill is, and `ring_set(false)` releases it so a decode-only stretch returns the bytes to
// the arena exactly like the ring and the wide view.  The FA staging arena is only ever touched by a
// multi-token graph, which is never CUDA-graph captured, so its VA does not have to survive the release;
// the caller caches the pointer and validates it with `ggml_cuda_slab_wide_aux_gen`.
//
// Growing the region is an ordinary work-boundary move: the arena tables in the taken band are evicted
// (the same cold band a wide prefill already reclaims).  A refusal (no slab, ring disarmed, or the region
// would reach the pinned narrow-2 view) is fail-soft: the caller falls back to its capped raw allocation.
//
// TODO #70: the growth decision is made ONCE per wide pass (see `wide_aux_denied_want`).  A refusal is
// logged once per pass and the rest of the pass is served by the caller's fallback, so a starved region
// cannot turn into thousands of boundary moves under a graph that still holds the previous placement.
void * ggml_cuda_slab_wide_aux_alloc(int device, size_t size, size_t * gen_out) {
    if (gen_out != nullptr) {
        *gen_out = 0;
    }
    if (size == 0) {
        return nullptr;
    }
    size_t lo = 0, hi = 0, off = 0;
    CUdeviceptr base = 0;
    bool move = false;
    {
        std::lock_guard<std::mutex> lock(g_slab_mutex);
        ggml_cuda_slab & s = g_slabs[device];
        const size_t want = ggml_cuda_slab_up(size, s.unit);
        // One refusal log per wide pass, whatever the reason.  The no-slab / ring-disarmed case is NOT
        // latched: the ring can be armed later in the same pass (the first prefill ubatch of a session can
        // arrive before the narrow floor is pinned).
        const auto refuse_once = [&s, device, size](const char * why, size_t extra) {
            if (s.wide_aux_denied_n++ == 0) {
                GGML_LOG_WARN("SLAB_WIDE_AUX dev=%d req=%.1f MiB refused for this wide pass (%s%s%.0f MiB): "
                              "the callers fall back to their raw allocations until the pass ends\n",
                              device, (double) size/1048576.0, why, extra > 0 ? ", " : "",
                              extra > 0 ? (double) extra/1048576.0 : 0.0);
            }
        };
        if (!s.inited || !s.ring_armed) {
            if (getenv("GGML_CUDA_SLAB_STATS") != nullptr) {
                if (s.wide_aux_denied_n == 0) {
                    GGML_LOG_WARN("SLAB_WIDE_AUX dev=%d req=%.1f MiB refused: slab=%d ring_armed=%d\n",
                                  device, (double) size/1048576.0, (int) s.inited, (int) s.ring_armed);
                }
                s.wide_aux_denied_n++;
            }
            return nullptr;   // narrow-only stretch: the region belongs to the arena
        }
        if (s.wide_aux_denied_want != 0 && want >= s.wide_aux_denied_want) {
            s.wide_aux_denied_n++;
            return nullptr;   // this pass's growth was already refused: do not touch the layout again
        }
        if (s.wide_aux_size == 0 && s.wide_aux_off == 0) {
            s.wide_aux_off = s.boundary;   // just above the wide compute view
            s.wide_n_rebase++;
        }
        const size_t aux_end = s.wide_aux_off + s.wide_aux_size;
        const size_t target  = ggml_cuda_slab_up(aux_end + want, s.chunk);
        // Never grow past the MAPPED region (the arena's top): the slab's VA may carry spare unmapped
        // space above `mapped`, and a kernel writing there would fault.
        const size_t aux_limit = std::min(ggml_cuda_slab_arena_limit(s), s.mapped);
        if (getenv("GGML_CUDA_SLAB_STATS") != nullptr) {
            GGML_LOG_WARN("SLAB_WIDE_AUX dev=%d req=%.1f MiB want=%.1f MiB off=%.1f MiB aux=%.1f MiB "
                          "target=%.1f MiB boundary=%.1f MiB limit=%.1f MiB armed=%d\n",
                          device, (double) size/1048576.0, (double) want/1048576.0,
                          (double) aux_end/1048576.0, (double) s.wide_aux_size/1048576.0,
                          (double) target/1048576.0, (double) s.boundary/1048576.0,
                          (double) aux_limit/1048576.0, (int) s.ring_armed);
        }
        // Never grow into the pinned narrow-2 region (the draft's compute VA) or past the mapping.
        if (target > aux_limit) {
            if (s.wide_aux_size == 0) {
                s.wide_aux_off = 0;
                s.wide_live.clear();
            }
            s.wide_aux_denied_want = want;
            refuse_once("VA limit", target > aux_limit ? target - aux_limit : 0);
            return nullptr;
        }
        off  = aux_end;
        base = s.base;
        if (target > s.boundary) {
            lo   = s.boundary;
            hi   = target;
            move = true;
        } else {
            s.wide_live[off] = want;
            s.wide_aux_size += want;
            if (gen_out != nullptr) {
                *gen_out = s.wide_aux_gen;
            }
            return (void *) (uintptr_t) ((char *) s.base + off);
        }
    }
    if (move) {
        // The arena may hold tables in [lo, hi): evict them (this syncs the devices).  MUST run without
        // `g_slab_mutex` (the eviction callback re-enters via `ggml_cuda_slab_arena_free`).
        moe_cache_evict_slab_range(device, (void *) (uintptr_t) ((char *) base + lo),
                                   (void *) (uintptr_t) ((char *) base + hi));
        std::lock_guard<std::mutex> lock(g_slab_mutex);
        ggml_cuda_slab & s = g_slabs[device];
        // Another thread (the MTP draft context's reserve) may have moved the boundary or released the
        // region while the lock was dropped.  Only the exact snapshot we sized against is usable: `target`
        // was computed from `lo`, so ANY boundary change invalidates it.  (Comparing `s.boundary > off`
        // instead looks equivalent but is not: the base is set to the boundary BEFORE the move, and the
        // move then rounds the boundary up to `chunk`, so the retained base normally sits a whole chunk
        // below the boundary -- every later growth would be refused and fall back outside the slab.)
        if (!s.inited || !s.ring_armed || s.boundary != lo || s.wide_aux_off + s.wide_aux_size != off) {
            // TODO #70: the growth was ALREADY paid for (a band of arena tables was evicted) and the layout
            // it was sized against moved while the lock was dropped, so this snapshot is unusable.  This is
            // the RECOVERABLE class: the racing thread is the MTP draft context's reserve and retrying from
            // the new state normally succeeds.  Retry, but at most `GGML_CUDA_SLAB_WIDE_AUX_RACE_MAX` times
            // per pass -- each attempt evicts another band, which is what made the pre-r168 code fault a
            // queued graph that still held the previous placement -- and then latch the pass.
            const size_t want_denied = ggml_cuda_slab_up(size, s.unit);
            s.wide_aux_race_n++;
            if (s.wide_aux_race_n < GGML_CUDA_SLAB_WIDE_AUX_RACE_MAX) {
                return nullptr;   // next allocation re-reads the layout and tries again
            }
            s.wide_aux_denied_want = s.wide_aux_denied_want != 0
                ? std::min(s.wide_aux_denied_want, want_denied)
                : want_denied;
            if (s.wide_aux_denied_n++ == 0) {
                GGML_LOG_WARN("SLAB_WIDE_AUX dev=%d req=%.1f MiB refused for this wide pass (%zu layout moves "
                              "during the grow); the callers fall back to their raw allocations until the "
                              "pass ends\n", device, (double) size/1048576.0, s.wide_aux_race_n);
            }
            return nullptr;
        }
        if (hi > s.boundary) {
            s.boundary = hi;
            for (auto it = s.arena_free.begin(); it != s.arena_free.end(); ) {
                const size_t f_off = it->first;
                const size_t f_end = f_off + it->second;
                if (f_end <= s.boundary) {
                    it = s.arena_free.erase(it);
                } else if (f_off < s.boundary) {
                    const size_t keep_off = s.boundary;
                    const size_t keep_sz  = f_end - keep_off;
                    s.arena_free.erase(it);
                    s.arena_free[keep_off] = keep_sz;
                    break;
                } else {
                    ++it;
                }
            }
        }
        const size_t want2 = ggml_cuda_slab_up(size, s.unit);
        s.wide_live[off] = want2;
        s.wide_aux_size += want2;
        if (gen_out != nullptr) {
            *gen_out = s.wide_aux_gen;
        }
        return (void *) (uintptr_t) ((char *) s.base + off);
    }
    return nullptr;
}

// wip/slab-transient-absorption: is the wide band available (slab live AND the wide view armed)?  The
// prefill-only scratch users gate on this so a decode/verify graph (ring disarmed) keeps the workspace pool.
bool ggml_cuda_slab_wide_available(int device) {
    std::lock_guard<std::mutex> lock(g_slab_mutex);
    const ggml_cuda_slab & s = g_slabs[device];
    return s.inited && s.ring_armed;
}

// The current generation of the wide-phase auxiliary region.  A caller that caches a pointer from
// `ggml_cuda_slab_wide_aux_alloc` must re-acquire when this changes.
size_t ggml_cuda_slab_wide_aux_gen(int device) {
    std::lock_guard<std::mutex> lock(g_slab_mutex);
    return g_slabs[device].wide_aux_gen;
}

// LIFO release of a wide-phase auxiliary block (the QSA3 prefill workspace, allocated and freed around one
// op launch).  Only the block at the top of the stack is popped; anything else leaves a hole that the wide
// drop reclaims, so a non-LIFO caller is safe, just less efficient.  When the stack empties, the region is
// re-based on the next allocation and the generation is bumped (no live pointer can exist at that point).
// Release one wide-aux block.  `g_slab_mutex` must be held; no ROCm call.
//
// The block's OWN size comes from the live map, not from the caller, so a free can never corrupt the stack
// and the region shrinks to the extent of the highest block that is still live.  (A bump stack that only
// pops its exact top turns every out-of-order free into a permanent hole; see `wide_live`.)
static void ggml_cuda_slab_wide_aux_free_locked(ggml_cuda_slab & s, void * ptr, size_t size) {
    GGML_UNUSED(size);
    if (ptr == nullptr) {
        return;
    }
    if (!s.inited || s.wide_aux_size == 0) {
        return;
    }
    const size_t rel = (size_t) ((uintptr_t) ptr - (uintptr_t) s.base);
    const auto it = s.wide_live.find(rel);
    if (it == s.wide_live.end()) {
        s.wide_n_hole++;
        return;   // not a live block of this region (a stale pointer, or already released at the drop)
    }
    s.wide_live.erase(it);
    const size_t top = s.wide_live.empty()
                     ? 0
                     : s.wide_live.rbegin()->first + s.wide_live.rbegin()->second - s.wide_aux_off;
    if (top < s.wide_aux_size) {
        s.wide_aux_size = top;
        s.wide_n_pop++;
    }
    // The base is RETAINED when the block set empties: the next layer's blocks are the same sizes and land in
    // the same place, so the boundary does not walk a fresh chunk per layer (measured: 8064 -> 11136 MiB
    // over one 15.7k prefill before this, i.e. ~3 GiB of the arena churned per prefill).  `work_alloc`
    // invalidates the base if a wider view would overlap it or if the boundary shrinks below it.
}

void ggml_cuda_slab_wide_aux_free(int device, void * ptr, size_t size) {
    if (ptr == nullptr || size == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_slab_mutex);
    ggml_cuda_slab_wide_aux_free_locked(g_slabs[device], ptr, size);
}

// The context classifies each ubatch (prefill: `n_tokens >= 64`; decode/verify: below) and tells the slab
// here.  The ring itself is still carved with a wip view -- arming is just `boundary >= narrow_floor +
// ring_region` -- but the classification must NOT be inferred from the work request size: for the
// 35B-A3B / GSQ configs the whole compute layout is size-independent (`GRAPH_RESERVE n_tokens=2048` and
// the narrow probe both measure 1280 MiB), so a size test left the prefill at base 0 with
// `boundary == narrow_floor` and `ring_armed == false`.  `work_alloc` reads this flag and places the
// prefill view above the ring.  A no-op for the generic arm/disarm plumbing stays valid for a future
// non-slab allocator.
// The ring hole is released on the prefill -> decode transition (see `ggml_cuda_slab_ring_set`), and so is
// the wide-phase auxiliary region: it only ever serves a wide pass, so a decode-only stretch returns its
// bytes to the arena.  A cached FA staging pointer is validated against `wide_aux_gen` before use.
void ggml_cuda_slab_ring_set(int device, bool arm) {
    std::lock_guard<std::mutex> lock(g_slab_mutex);
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) {
        return;
    }
    ggml_cuda_slab & s = g_slabs[device];
    const bool was = s.ring_requested;
    s.ring_requested = arm;
    if (arm && !was) {
        ggml_cuda_pool_wide_refused_reset(device);   // per-pass diagnostic (reported at the drop)
    }
    const bool transition = was != arm;
    if (transition && s.inited) {
        // wip/slab-transient-absorption: the prefill -> decode transition is the point where the wide
        // band (ring + wide compute + FA staging) is about to be handed back to the arena, so it is the
        // right place to report what the prefill actually claimed and what is free outside the slab.  It
        // is logged BEFORE the auxiliary region is cleared so the wide_aux figure is the prefill's.
        // `ggml_cuda_slab_log_stats_locked` gates itself on GGML_CUDA_SLAB_STATS.
        ggml_cuda_slab_log_stats_locked(device, arm ? "wide-arm" : "wide-drop");
    }
    if (transition) {
        // TODO #70: one wide pass has ended -- the next one gets its own wide-aux growth decision, so a
        // region that was starved under this pass's layout is retried rather than written off for the run.
        s.wide_aux_denied_want = 0;
        s.wide_aux_denied_n    = 0;
        s.wide_aux_race_n      = 0;
    }
    if (!arm) {
        // The live set must not outlive the region it describes.
        s.wide_live.clear();
    }
    if (!arm && s.wide_aux_size > 0) {
        // The next `work_alloc` shrinks the boundary back to the narrow floor, returning the region to
        // the arena.  Bump the generation so a stream's cached FA staging pointer is re-acquired rather
        // than read after the arena has taken the bytes.
        s.wide_aux_off  = 0;
        s.wide_aux_size = 0;
        s.wide_aux_gen++;
    }
    if (!arm) {
        // wip/slab-transient-absorption (step 2b) TRIPWIRE: the workspace pool serves large prefill
        // allocations from this band and returns them on free, so NOTHING may still hold a band block
        // here -- the bytes are about to go back to arena tables that a later prefill overwrites.  A
        // non-zero count means a retained site (an activation cache, a comm scratch, ...) was not pinned
        // with `ggml_cuda_pool_wide_pin`: the delivery would be a silent use-after-free, so say so loudly
        // (not gated on GGML_CUDA_SLAB_STATS).
        const size_t live = ggml_cuda_pool_wide_blocks(device);
        if (live > 0) {
            GGML_LOG_WARN("POOL_WIDE_LEAK dev=%d blocks=%zu bytes=%.1f MiB still live at the wide drop: a "
                          "pool allocation outlived the prefill band (pin the retained site)\n",
                          device, live, (double) ggml_cuda_pool_wide_bytes(device) / 1048576.0);
        }
    }
}

// --- Phase-2 (item 6b): state accessors for the device wrappers kept in ggml-cuda.cu ---
// `g_slab_armed` / `g_slab_compute_enabled` / `g_slab_compute_narrow2` are module-local to the slab.
void ggml_cuda_slab_arm() {
    g_slab_armed = true;
    // r165: this model offloads experts to host memory, so the large all-reduce arm is RCCL's ring
    // rather than the copy engine (a host-offloaded prefill is PCIe-saturated by the expert streaming:
    // Flash-Next IQ4_NL 2 GPU `-ncmoe 20` pp2048 809 vs 604 t/s at `-ncmoe 32` 555 vs 414).  The flag is
    // never cleared: a dense model simply never arms the slab, and a later disarm (a declined cache)
    // does not change where the experts live.  See ggml-cuda.cu's comm_init_auto / select_arm.
    extern bool g_comm_host_experts;
    g_comm_host_experts = true;
}

// Undo a previous arm.  The MoE preflight runs at MODEL LOAD (universal, aux 0) and again from `common`
// with the draft/MTP aux; if the second, better-informed call declines the cache (auto floor), the slab
// must not stay armed for a cache that is off (it would withhold the VRAM the decline was protecting).
// Safe only before a slab exists; a live slab is left alone.
void ggml_cuda_slab_disarm() {
    {
        std::lock_guard<std::mutex> lock(g_slab_mutex);
        for (int d = 0; d < GGML_CUDA_MAX_DEVICES; d++) {
            if (g_slabs[d].inited) {
                return;   // a live slab owns its mappings; do not yank the gate out from under it
            }
        }
    }
    g_slab_armed = false;
}

bool ggml_cuda_slab_compute_enabled() {
    return g_slab_compute_enabled && !g_slab_compute_plain;
}

void ggml_cuda_slab_set_compute_plain(bool enable) {
    g_slab_compute_plain = enable;
}

bool ggml_cuda_slab_compute_plain() {
    return g_slab_compute_plain;
}

void ggml_cuda_slab_set_compute_narrow2(bool enable) {
    if (enable && !ggml_cuda_slab_narrow2_enabled()) {
        g_slab_compute_enabled = false;
        g_slab_compute_narrow2 = false;
    } else if (enable) {
        g_slab_compute_enabled = true;
        g_slab_compute_narrow2 = true;
    } else {
        g_slab_compute_enabled = true;
        g_slab_compute_narrow2 = false;
    }
}
