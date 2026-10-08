// wip/moe-expert-cache — Phase 1a foundation.  See moe-expert-cache.h for the contract.

#include "moe-expert-cache.h"
#include "ggml-cuda-vmm.h"

#include "common.cuh"
#include "mmvq.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#include <set>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ---------------------------------------------------------------------------------------------
// state
// ---------------------------------------------------------------------------------------------

// Device-side admission-policy descriptor (MOE_EXPERT_CACHE_DEVPOLICY=1).  One per device-remap table,
// built once at sizing and updated in place by `moe_cache_policy_kernel`.  All pointers are device
// addresses on the table's owner device; `used` is the persistent routing buffer the remap kernel
// wrote during the graph.  `arena`/`host_dev` drive the fill copy (device write, UVA host read).
struct moe_cache_policy_desc {
    int32_t *       slot_expert;
    int32_t *       slot;
    int32_t *       count;
    int32_t *       ghost;
    int32_t *       last;
    uint8_t *       slot_prov;   // slot -> provisional (pre-filled, never hit)
    const int32_t * used;
    void *          arena;
    const void *    host_dev;
    int64_t         host_bytes;
    int64_t         src_off;
    int64_t         expert_bytes;
    int64_t         host_pitch;
    int64_t         clock;
    int64_t         last_decay;
    int32_t         n_experts;
    int32_t         slots;
    int32_t         occupied;
    int32_t         split_axis;
    int64_t         acc_hits;
    int64_t         acc_misses;
    int64_t         acc_fills;
    int64_t         acc_evict;
    int64_t         pad0;
    int64_t         pad1;
};

// The kernel stages a token's admitted experts in shared memory before the block cooperatively copies them
// into the arena.  A table can see up to `n_used * n_tok` routed experts in one pass (16 x 16 = 256 at the
// widened band); the admission loop stops admitting once the list is full, so an expert is never marked
// resident without its fill (it stays cold and is served from the host alias instead).
#define MOE_CACHE_POLICY_MAX_FILL 256

namespace {

struct table_t {
    int         layer       = -1;
    std::string role;
    int         n_experts   = 0;
    size_t      expert_bytes= 0;    // THIS device's per-expert slice bytes (the arena stride)
    size_t      host_bytes  = 0;    // full per-expert stride in the host master (== expert_bytes when unsplit)
    size_t      src_off     = 0;    // byte offset of this device's slice within a host expert
    size_t      host_pitch  = 0;    // host row pitch (nb[1]) for a strided axis-0 slice; 0 => contiguous 1-D fill
    int         split_axis  = -1;   // -1 = whole expert; 0/1 = the master axis this slice was split on
    bool        cold_safe   = true; // false: the host geometry is not representable by a single UVA read
    // wip/host-expert-dio-cache Phase 1: the on-disk source of this table's host master, registered by
    // the loader (`moe_cache_set_host_source`).  Empty unless `MOE_HOST_POOL_MIB > 0`.
    std::string src_path;
    size_t      src_offs    = 0;
    bool        src_known   = false;
    const void * host       = nullptr;
    void *      host_dev    = nullptr;  // device-accessible alias of `host` (UVA), Phase 1b cold reads
    bool        host_dev_bound = false; // the alias was resolved (or ruled out) once
    int         device      = -1;       // CUDA ordinal that owns (allocates and reads) this table's arena

    int         slots       = 0;        // 0 => disabled (no budget / alloc failed)
    bool        primed      = false;    // registered on a previous (uncached) pass
    bool        allocated   = false;    // arena decided (allocated, or deliberately 0 slots)
    void *      arena       = nullptr;
    // OPEN 2: when the arena is VMM-backed, `arena` is a RESERVED VA range that stays valid for the
    // table's life (only physical is unmapped on a yield and re-mapped on a re-arm), and
    // `arena_reserved` is its size.  0 for a legacy cudaMalloc arena.
    size_t      arena_reserved = 0;
    std::vector<int32_t> slot_expert;   // slots -> expert id, -1 empty
    std::unordered_map<int32_t, int32_t> expert_slot;
    bool        slot_dirty  = false;    // a slot changed since the last device-map upload
    std::vector<int64_t> count;         // per expert, decaying LFU counter (uses while RESIDENT)
    std::vector<int64_t> ghost;         // per expert, decaying demand counter (uses while NON-resident)
    std::vector<int64_t> last;          // per expert, last use (per-table clock)

    int64_t     clock       = 0;        // decode steps seen by this table
    int64_t     last_decay  = 0;
    int64_t     hits        = 0;
    int64_t     misses      = 0;
    int64_t     fills       = 0;
    int64_t     evictions   = 0;
    bool        alloc_failed= false;
    // Session 7 identity fast path: slots == n_experts, slot == expert, the whole table was copied
    // in at sizing time, and the consumer reads the arena with the raw routing ids (no remap, no
    // per-token host routing readback).
    bool        identity    = false;

    // slot-remapped ids for the most recent routing (device), and its shape.
    int32_t *   remap_dev   = nullptr;
    int64_t     remap_cap   = 0;        // capacity in int32 elements
    int64_t     remap_n_used= 0;
    int64_t     remap_n_tok = 0;
    std::vector<int32_t> remap_host;    // persistent staging (the async H2D must not race its source)
    std::vector<int32_t> hook_experts;  // the routing the hook last consumed (read-check only)

    // Device-side remap (gentle-curve follow-up): a device `expert -> slot` map the consumer uses to
    // build the remap without a host routing readback.  Enabled for partial-residency tables.
    int32_t *   slot_dev    = nullptr;  // device int32[n_experts]
    int32_t *   used_dev    = nullptr;  // device int32[n_experts*8]: the routing the remap kernel last read
    int64_t     used_cap    = 0;
    int32_t *   used_host   = nullptr;  // PINNED staging for the used-list D2H
    int32_t *   used_host2  = nullptr;  // second pinned buffer: double-buffered PIPELINED readback
    int         used_toggle = 0;        // which pinned buffer the NEXT readback writes
    bool        used_pending= false;    // a readback was enqueued and has not been consumed yet
    int32_t *   slot_pin    = nullptr;  // PINNED staging for the slot-map H2D
    std::vector<int32_t> slot_dev_host; // staging for the async H2D
    bool        devmap      = false;
    // A sibling (the layer's routed `down`) built this table's remap in the gate+up redirect this pass;
    // the redirect skips its own launch and the per-table promotion clears the flag after the graph.
    bool        remap_fresh = false;

    // Device-side admission policy (MOE_EXPERT_CACHE_DEVPOLICY=1, default off).  When `policy` is set
    // the LFRU decision runs on the GPU (one batched kernel per device per token): these are the
    // device mirrors of `slot_expert`/`count`/`ghost`/`last`.  The per-table promote only records the
    // token's routing shape; `moe_cache_policy_flush` launches the kernel and copies admitted experts
    // into the arena.  The host mirrors above go stale while `policy` is on (refreshed for the report).
    bool        policy          = false;
    bool        policy_pending  = false;   // this table is in the current token's graph
    int         policy_n_used   = 0;       // shape recorded by the per-table promote
    int         policy_n_tok    = 0;
    int32_t *   slot_expert_dev = nullptr;  // device slot -> expert, -1 empty
    int32_t *   count_dev       = nullptr;  // device per-expert decaying LFU counter
    int32_t *   ghost_dev       = nullptr;  // device per-expert decaying demand counter
    int32_t *   last_dev        = nullptr;  // device per-expert last use (per-table clock)
    uint8_t *   slot_prov_dev   = nullptr;  // device slot -> provisional (pre-filled, never hit)
    // Provisional slots (MOE_EXPERT_CACHE_PROVISIONAL=1): a pre-filled/loaded expert is "provisional"
    // until it is hit.  While it is, the touch doorkeeper is bypassed for it, so the entry is evictable
    // like an empty slot while still serving hits - the model-both-ways behaviour arbitrary pre-fill
    // needs.  slot_prov_dev is the device mirror; the host vector drives the eager/remap path.
    std::vector<uint8_t> slot_prov;

    // Prefill seed (MOE_EXPERT_CACHE_PREFILL_SEED=1): per-expert tally of the experts the prefill
    // routing touched, accumulated in `moe_cache_update_host`'s prefill branch and bulk-admitted at
    // arena sizing by `apply_prefill_seed_locked`.  The decode band never uses it.
    std::vector<int64_t> prefill_count;     // [n_experts], 0 until a prefill touches this table
    int64_t     prefill_tokens = 0;         // prefill tokens tallied (reporting)
    // Device-side prefill tally (session 14): the host hook above is bypassed under `-sm tensor`
    // (block-06 staging intercepts the prefill upload), so the prompt's routing is histogrammed on the
    // GPU from `ggml_cuda_mul_mat_id`, where the routing device tensor is in hand.  One-shot: the first
    // decode-band flush after a non-empty tally bulk-admits it as provisional slots (`seed_prefill_lazy`).
    int32_t *   prefill_count_dev     = nullptr;   // device int32[n_experts]
    bool        prefill_tally_pending = false;     // a prefill op tallied since the last seed
    bool        prefill_seeded        = false;     // the prompt seed was applied (one-shot per table)
};

std::mutex                          g_mutex;
std::vector<table_t>                g_tables;
// Device-side admission policy: one descriptor array per CUDA device (indexed by ordinal).  Built
// lazily after sizing (all tables registered), then reused for the life of the process.
struct policy_dev_t {
    moe_cache_policy_desc * desc       = nullptr;
    int32_t *               shape_dev  = nullptr;   // device int32[n]: this token's n_used*n_tok, 0 = inactive
    std::vector<int>        ids;                     // global table indices, in descriptor order
    std::vector<int32_t>    shape_host;
    int                     n          = 0;
    int                     cap        = 0;
    bool                    initialized = false;     // device arrays seeded from the host mirrors
    // Periodic progress logging: one tick per token per device (the scheduler's end-of-pass flush).
    int64_t                 tokens      = 0;
    int64_t                 last_fills  = 0;
    int64_t                 last_evicts = 0;
    int64_t                 last_resident = 0;
    int64_t                 last_time_us = 0;
    int64_t                 last_hits   = 0;
    int64_t                 last_misses = 0;
};
std::vector<policy_dev_t>           g_policy_dev;
// Stable registration key: (host master weight, device).  It must not be the tensor the op reads
// (`weight_cpy`): the graph allocator reuses those pointers across layers, so the consumer alias map
// below is re-pointed on every hook call while registration stays fixed.
std::map<std::pair<const void*, int>, int> g_key_to_id;
// Consumer alias: the tensor the op actually reads (the redirected `input_cpy`, or a per-device simple
// tensor under `-sm tensor`) -> table.  Refreshed each hook call.
std::unordered_map<const void*, int>       g_alias_to_id;
std::set<std::pair<const void*, long long>> g_heads_zeroed;   // (buffer, expert_bytes) whose expert heads were zeroed once
// Semantic key (layer, normalized role, device) -> table.  Robust under `-sm tensor`, where the
// per-device simple tensor a hook saw may be a different pointer than the one the op reads after a meta
// graph rebuild: the consumer falls back to this when the pointer alias misses.
std::map<std::tuple<int, std::string, int>, int> g_sem_to_id;
// wip/host-expert-dio-cache Phase 1: host tensor data pointer -> on-disk source.  Populated by the
// loader only when `MOE_HOST_POOL_MIB > 0`; later phases open the file lazily and fill the pool from it.
struct host_src_t {
    std::string path;
    size_t      offs        = 0;   // byte offset of the tensor's data in the GGUF file
    int         n_experts   = 0;
    size_t      host_bytes  = 0;   // per-expert host stride (whole tensor)
    size_t      total_bytes = 0;   // whole tensor
};
std::unordered_map<const void *, host_src_t> g_host_src;
// wip/host-expert-dio-cache Phase 2: a bounded, pinned host tier (L2) that is the GPU-readable BOUNCE
// BUFFER between the page cache and the VRAM arena.  One pool per host tensor, keyed by `t.host`,
// holding WHOLE host experts (so a refill is one contiguous read and every per-device slice can be
// served from the same slot); LRU eviction.  Fills read the GGUF BUFFERED (through the page cache, the
// default) so a miss is a RAM read, not disk; `MOE_HOST_POOL_DIO=1` forces O_DIRECT as a debug fallback.
// The arena fill in `access_locked` sources a pool slot and the GPU never reads the pool (invisible L2);
// the full pinned master stays for now (Phase 3 removes it).
struct host_pool_t {
    char *      base        = nullptr;   // cudaMallocHost, slots * host_bytes
    int         slots       = 0;
    size_t      host_bytes  = 0;
    int         n_experts   = 0;
    std::string path;                    // GGUF path
    size_t      file_offs   = 0;         // tensor data offset in the file
    int         fd          = -1;
    size_t      align       = 4096;
    std::vector<int32_t>                 slot_expert;   // slot -> expert, -1 empty
    std::unordered_map<int32_t, int32_t> expert_slot;
    std::vector<cudaEvent_t>             slot_ev;       // one per slot: the last async arena fill copy out of it
    std::vector<int64_t>                 last;          // per expert, last use
    int64_t     clock       = 0;
    int64_t     hits        = 0;
    int64_t     misses      = 0;
    int64_t     fills       = 0;
    int64_t     evictions   = 0;
    bool        failed      = false;
};
// Keyed by (host tensor, device): the pool is PER DEVICE (the budget is per-device, per the campaign),
// so every slot's async fill copy is on one device's stream and its `slot_ev` is valid on that device.
std::map<std::pair<const void *, int>, host_pool_t> g_host_pools;   // node-based: pointers stay valid
bool    g_pool_enabled     = false;
bool    g_pool_prewarm     = true;   // MOE_HOST_POOL_PREWARM: fill every pool slot when it is built
bool    g_pool_dio         = false;  // MOE_HOST_POOL_DIO=1: O_DIRECT fill (debug fallback; bypasses the page cache)
int64_t g_pool_total_bytes = 0;   // per device

// An alias is keyed by a scheduler tensor address, and the graph allocator reuses those addresses across graphs - with
// several devices, for a split input on ANOTHER device.  A lookup that is not preceded by a registration in the same
// graph can therefore hit a stale alias: another layer's table, or another device's (its arena/counters are then read
// or written from the wrong GPU - a page fault in moe_cache_tally_kernel, or garbage, with -sm layer on 2 GPUs).
// Only trust an alias whose table is on the calling device and, when the op is known, belongs to the op's layer.
static int64_t g_alias_stale = 0;
static int alias_find_checked(const void * key, int device, int layer) {
    const auto it = g_alias_to_id.find(key);
    if (it == g_alias_to_id.end()) {
        return -1;
    }
    const int id = it->second;
    if (id < 0 || id >= (int) g_tables.size() ||
        (device >= 0 && g_tables[id].device != device) ||
        (layer >= 0 && g_tables[id].layer >= 0 && g_tables[id].layer != layer)) {
        if (g_alias_stale++ == 0 || (g_alias_stale % 10000) == 0) {
            // Warn on the first, then every 10000th: the guard is doing its job either way, but the
            // magnitude is what tells a one-off (a scheduler address reused across graphs, which is
            // routine) from a runaway (the graphs being re-planned constantly).
            GGML_LOG_WARN("%s: ignoring a stale MoE-cache alias (table device %d layer %d, op device %d layer %d); %lld refused so far\n",
                          __func__, id >= 0 && id < (int) g_tables.size() ? g_tables[id].device : -1,
                          id >= 0 && id < (int) g_tables.size() ? g_tables[id].layer : -1, device, layer,
                          (long long) g_alias_stale);
        }
        return -1;
    }
    return id;
}
bool                                g_enabled    = false;
bool                                g_init_done  = false;
bool                                g_report     = true;
bool                                g_fill       = false;   // opt-in: copy the master into the slot
int                                 g_device     = -1;
int64_t                             g_period     = 32;   // LFRU decay period, decode steps
size_t                              g_budget     = 0;    // bytes (per device); SIZE_MAX in auto mode
bool                                g_auto       = false;  // MOE_EXPERT_CACHE_MIB unset: size each device from its free VRAM
bool                                g_preflight_done = false; // the early auto floor decision has run
int64_t                             g_min_mib    = 0;      // auto: decline the arena below this many MiB/run
int                                 g_min_res_pct= 0;      // auto: decline the arena below this percent of the host experts
int                                 g_slots_hint = 0;    // 0 => derive from budget
int64_t                             g_arena_bytes= 0;
int64_t                             g_total_expert_bytes = 0;   // sum over primed tables
int                                 g_uniform_slots = -1;       // the decided per-table slot count
// OPEN 2 re-arm: the per-layer slot count `alloc_all_locked` settled on, so a later re-arm can restore
// it after a compute-buffer drop returned the VRAM.  -1 = the layer was not cacheable at sizing time.
int                                 g_rearm_slots[512];
bool                                g_sized           = false;  // the deferred sizing has run
bool                                g_cold_uva        = false;  // MOE_EXPERT_CACHE_COLD=uva: serve rejected misses from the host alias
bool                                g_devmap          = false;  // MOE_EXPERT_CACHE_DEVMAP (default ON): build the remap on the device
bool                                g_devpolicy       = false;  // MOE_EXPERT_CACHE_DEVPOLICY=1: LFRU admission on the device
bool                                g_kslot           = false;  // MOE_EXPERT_CACHE_KSLOT=1: resolve the slot map in the MoE ids consumer (B3)
// Device-remap transition arming.  Sizing happens mid-token inside the eager hook (the first decode-band
// role to reach `!g_sized`), so on that token some roles are filled by the eager hook and the rest would
// be taken over by devmap.  The two paths must not be mixed across the roles of a layer: the fused
// gate+up kernel indexes the gate lane with the UP table's remap, so gate/up/down must hold identical
// `expert_slot` maps.  Devmap takeover therefore stays disabled until a full decode pass has filled every
// devmap-capable table eagerly (one uniform pass), and only then flips on.
bool                                g_devmap_armed    = true;
int                                 g_devmap_arm_expected = 0;
int                                 g_devmap_arm_seen  = 0;
std::vector<uint8_t>                g_devmap_eager_seen;
int64_t                             g_cold_reaches    = 0;
// Phase 1b fill-vs-cold admission policy (only consulted when a cold read is available):
//   0 = always : fill every miss (LFRU eviction is the whole admission control)
//   1 = value  : fill only if the incoming's decaying DEMAND beats the victim's decaying VALUE
//   2 = touch  : fill only on the expert's 2nd+ use since it was last resident (doorkeeper)
int                                 g_admit           = 0;
int64_t                             g_cold_admits     = 0;  // misses served cold by the admission rule (not by a full/declined table)
int64_t                             g_touch           = 2;  // `admit=touch`: uses since last resident before a fill is allowed
int64_t                             g_takeover        = 0;
int64_t                             g_decline_all     = 0;
int64_t                             g_get_ok          = 0;
int64_t                             g_total_one_expert_bytes = 0;  // sum of one expert per table (the zero-slot reserve)
// Fail-soft / --fit interaction (issue #33).  The arena is the LOWEST-priority VRAM consumer: it is
// sized from what is actually FREE at sizing time (after --fit, the compute reserve and the KV cache
// have taken theirs) minus `g_reserve_mib`, so it cannot over-commit and `--fit` does not need to know
int64_t                             g_reserve_mib     = 1024;
// WIP r42 (TODO #42): extra reserve (bytes) for the post-prefill compute layout.  Set at the prefill ->
// decode transition (llama_context), so the arena does not take the VRAM a later compute growth needs.
// Device-global: the compute reserve is the same projection for every device, and the model's device
// list is a single Meta wrapper under -sm tensor, so a per-device set could miss a device entirely.
int64_t                             g_extra_reserve   = 0;
bool                                g_prefill_seed    = false; // MOE_EXPERT_CACHE_PREFILL_SEED=1: warm the arena from the prefill routing
bool                                g_prov_evict      = false; // MOE_EXPERT_CACHE_PROVISIONAL=1: pre-filled entries are evictable like empty slots until first hit
int                                 g_prefill_seed_n  = 0;     // MOE_EXPERT_CACHE_PREFILL_SEED_N: cap seeded experts/table (0 = all slots)
int64_t                             g_alloc_clamped   = 0;
int64_t                             g_alloc_failed    = 0;

// Deferred-promotion call accounting (used by the exit report).
int64_t                             g_promote_calls   = 0;
// Expert-traffic accounting (per access through the policy): a FILL is one host->device copy of
// `expert_bytes`; a COLD access is the MMVQ kernel reading those bytes in place over PCIe.  Both are
// host->device traffic; only the fill is managed by the cache.
int64_t                             g_acc_hits        = 0;
int64_t                             g_acc_fills       = 0;
int64_t                             g_acc_colds       = 0;
int64_t                             g_fill_bytes      = 0;
int64_t                             g_cold_bytes      = 0;
int64_t                             g_remap_launches  = 0;

int env_int(const char * name, int dflt) {
    const char * e = getenv(name);
    if (e == nullptr || e[0] == '\0') {
        return dflt;
    }
    return atoi(e);
}

// wip/host-expert-dio-cache Phase 1: MOE_HOST_POOL_MIB as a cached integer (0 = off).  Read directly
// rather than from `parse_env` so the loader can call `moe_cache_set_host_source` regardless of init
// ordering.  Unset or 0 leaves the bounded host pool off (the default) and every new code path inert.
static int64_t moe_host_pool_mib() {
    static const int64_t v = [] {
        const char *    e = getenv("MOE_HOST_POOL_MIB");
        const long long n = (e != nullptr && e[0] != '\0') ? atoll(e) : 0;
        return (int64_t) (n > 0 ? n : 0);
    }();
    return v;
}

// Scope guard: make `device` current for the allocations/computes inside, restore on exit.  Every
// cudaMalloc/cudaFree/cudaHostGetDevicePointer in this module is device-scoped, so with more than one
// GPU each table's arena must be created while its OWN device is current (otherwise a device-1 kernel
// reads a device-0 pointer and faults in mul_mat_vec_q_moe).  `device < 0` leaves the current device
// alone (used by the self-test before a device is known).
struct device_guard {
    int  prev    = -1;
    bool changed = false;
    explicit device_guard(int device) {
        if (device < 0) {
            return;
        }
        if (cudaGetDevice(&prev) != cudaSuccess) {
            prev = -1;
            return;
        }
        if (prev != device) {
            changed = (cudaSetDevice(device) == cudaSuccess);
        }
    }
    ~device_guard() {
        if (changed && prev >= 0) {
            (void) cudaSetDevice(prev);
        }
    }
};

// Resolve the pinned host master's device-accessible alias once.  `t.src_off` is folded in by
// `moe_cache_get_cold`, so the alias is always the base of the host tensor.
//
// Only a host allocation the runtime can map for device access may be used as an in-place kernel-read
// master, and `cudaHostGetDevicePointer` is the runtime's own answer.  A pageable model mapping
// (`--host-experts mmap` -> `CPU_Mapped`) maps to no device address; on a GPU without XNACK (RDNA
// under ROCm reports `XNACK enabled: NO`) a kernel read of it is a fatal "page not present" fault
// (issue #116, `mul_mat_vec_q_moe`'s cold read).  Leave `host_dev` null in that case: the in-place
// cold region, the device remap and the device-policy fill all require it and stand down, while the
// fill copies (which source `t.host` through cudaMemcpyAsync, pageable-safe) keep the cache correct.
void bind_host_dev_locked(table_t & t) {
    if (!(g_cold_uva && t.host != nullptr && !t.host_dev_bound)) {
        return;
    }
    t.host_dev_bound = true;
    device_guard dg(t.device);
    void * dev = nullptr;
    if (cudaHostGetDevicePointer(&dev, (void *) t.host, 0) == cudaSuccess && dev != nullptr) {
        t.host_dev = dev;
        return;
    }
    (void) cudaGetLastError();
    static bool warned = false;
    if (!warned) {
        warned = true;
        GGML_LOG_WARN("%s: the host expert master has no device mapping (pageable model mapping?); "
                      "the in-place cold read is disabled and misses fall back to the copy path\n", __func__);
    }
}

// True when the table can serve a miss by reading the host master in place: the host geometry must be
// representable AND the master must have a device mapping.  A pageable model mapping has none, so the
// table takes the fill-every-miss path instead (issue #116).
static inline bool table_cold_ok(const table_t & t) {
    return g_cold_uva && t.cold_safe && t.host_dev != nullptr;
}

// Human-readable cache budget for the log lines: "auto" when MOE_EXPERT_CACHE_MIB was unset (each
// device is sized from its free VRAM at `alloc_all_locked`), otherwise the requested MiB/device.
static const char * budget_desc() {
    static char buf[32];
    if (g_auto) {
        std::snprintf(buf, sizeof(buf), "auto");
    } else {
        std::snprintf(buf, sizeof(buf), "%zu", g_budget >> 20);
    }
    return buf;
}

void parse_env() {
    const char * mib = getenv("MOE_EXPERT_CACHE_MIB");
    g_auto = false;
    if (mib != nullptr && mib[0] != '\0') {
        // Explicit value wins and is used verbatim (locked semantics): `0` is the kill switch,
        // `>0` is a fixed per-device budget.
        const long long v = atoll(mib);
        g_enabled = v > 0;
        if (!g_enabled) {
            return;
        }
        // Hard floor: an explicit budget below 2048 MiB/device is insufficient for the expert cache.
        // A tiny arena leaves most routed experts non-resident on every token, and the resulting
        // residency churn (evictions, cold/2-D fills) has proven unsafe at these sizes. `0` still
        // disables the cache, and an unset value still auto-sizes from free VRAM.
        if (v < 2048) {
            GGML_ABORT("%s: MOE_EXPERT_CACHE_MIB=%lld MiB/device is too small; the expert cache needs at "
                       "least 2048 MiB/device (set 0 to disable the cache)\n", __func__, v);
        }
        g_budget = (size_t) v * 1024 * 1024;
    } else {
        // Unset == AUTO: enable and let `alloc_all_locked` size each device from its free VRAM
        // (free - reserve).  A model with no host-resident expert table (`-ncmoe 0`) registers no
        // table, so the whole cache stays inert and the run is byte-identical to cache-off.
        g_enabled = true;
        g_auto    = true;
        g_budget  = (size_t) -1;
    }
    // Phase 2: the bounded, pinned, O_DIRECT-filled host tier (L2).  `g_pool_total_bytes` is split over
    // the registered host tensors when the first pool is created; the full pinned master is KEPT for now
    // (Phase 3 removes it and the scheduler fallback).
    g_pool_enabled     = moe_host_pool_mib() > 0;
    g_pool_prewarm     = env_int("MOE_HOST_POOL_PREWARM", 1) != 0;
    g_pool_dio         = env_int("MOE_HOST_POOL_DIO", 0) != 0;
    g_pool_total_bytes = g_pool_enabled ? moe_host_pool_mib() * 1024 * 1024 : 0;
    if (g_pool_enabled) {
        GGML_LOG_INFO("%s: MoE host expert pool: %lld MiB total, pinned, O_DIRECT-filled (Phase 2; "
                      "full pinned master kept)\n", __func__, (long long) moe_host_pool_mib());
    }
    g_slots_hint  = env_int("MOE_EXPERT_CACHE_SLOTS",   0);
    g_period      = env_int("MOE_EXPERT_CACHE_PERIOD",  32);
    g_fill        = env_int("MOE_EXPERT_CACHE_FILL",     1) != 0;
    // Phase 1b/3 cold transport: the per-op MoE consumer AND the cache-band gate+up+GLU / down-fold
    // kernels do the cold-region lookup, so a cold id is safe with fusions on.  A miss served cold
    // reads the pinned host slice in place (the same bytes the H2D fill would have placed in the
    // slot), and it also lets a small arena represent an overflow (`id = slots + e`).
    g_cold_uva = true;
    // Admission: `touch` (measured best at every arena size): a first-touch expert is served cold and
    // only a re-touch is admitted, so one-shot experts stop churning the arena - which raises `h` as
    // well as cutting transferred bytes.
    g_admit = 2;
    g_touch = env_int("MOE_EXPERT_CACHE_TOUCH", 2);
    if (g_touch < 2) {
        g_touch = 2;
    }
    // Device-remap mode (item 3 / the "gentle curve"): build the slot remap on the device + deferred
    // post-graph promotion, instead of the eager per-op host routing readback + full device sync.  The
    // eager host path reads the routing back per layer and synchronizes, so at partial residency the
    // decode stalls to ~70 t/s; the device-remap path is +22 % over it at every h (session 10).  At h=1 the
    // identity fast path wins the lookup first, so this is a no-op at full residency, and it self-arms
    // after one uniform eager decode pass (see `g_devmap_armed`).  DEFAULT ON since session 19 (the
    // post-session-10 promotion gates - byte-identity, width purity, MTP, coherence, MUL_MAT_ID - were
    // re-run green on the device-remap path); `MOE_EXPERT_CACHE_DEVMAP=0` restores the eager host path.
    g_devmap      = env_int("MOE_EXPERT_CACHE_DEVMAP", 1) != 0;
    // Device-side admission policy: run the LFRU decision on the GPU (one batched kernel per device
    // per token) instead of the per-table host promotion.  Requires the device-remap path.  DEFAULT ON
    // when `DEVMAP` is on (it passed byte-identity, width purity, the policy self-test, MTP and the
    // concurrency sweep); `MOE_EXPERT_CACHE_DEVPOLICY=0` is the kill switch.
    g_devpolicy   = g_devmap && env_int("MOE_EXPERT_CACHE_DEVPOLICY", 1) != 0;
    // B3: the MoE ids consumer (mmvq MoE kernel) resolves `slot_dev[ids[i]]` itself and writes the raw
    // routing into the table's used-list, so the per-table `moe_cache_build_remap_kernel` launches are
    // not needed.  DEFAULT ON whenever DEVMAP is on (opt out with `MOE_EXPERT_CACHE_KSLOT=0`): it is
    // bit-exact on every split (1/2/3 GPU, layer and tensor) including the UVA cold encoding, and it is
    // neutral-to-positive at every arena size measured (tiny MIB=1024/2048) as well as high h; the only
    // effect is to remove remap launches from the graph.  Requires the device-remap path.
    g_kslot       = g_devmap && env_int("MOE_EXPERT_CACHE_KSLOT", 1) != 0;
    // Fail-soft / --fit: the reserve kept for the compute reserve / KV growth.
    g_reserve_mib      = env_int("MOE_EXPERT_CACHE_RESERVE_MIB", 1024);
    if (g_reserve_mib < 0) {
        g_reserve_mib = 0;
    }
    // Auto floor (opt-in): below this total arena the fixed per-op cost outweighs the CPU bytes it
    // saves, so disable the cache and take the CPU expert path.  Default 0 (always arm) because a
    // small arena is worse than the CPU path but the disable also has an MTP regression -- see the
    // warning below and `wip/moe-cache-autosize/README.md`.  Only meaningful in auto mode; an explicit
    // MIB is used verbatim.
    g_min_mib = env_int("MOE_EXPERT_CACHE_MIN_MIB", 0);
    if (g_min_mib < 0) {
        g_min_mib = 0;
    }
    // Auto preflight floor as a residency fraction: below this the arena is a net loss under MTP
    // (measured crossover ~18 % on IQ3_XXS, i.e. ~8 GiB of a 46 GiB host set).  `0` disables it.
    g_min_res_pct = env_int("MOE_EXPERT_CACHE_MIN_RES_PCT", g_auto ? 18 : 0);
    if (g_min_res_pct < 0) {
        g_min_res_pct = 0;
    }
    // Prefill seed: use the prompt's prefill routing to bulk-admit its hot experts as provisional slots
    // before decode starts (session 14).  Default ON (kill switch `MOE_EXPERT_CACHE_PREFILL_SEED=0`);
    // it passed the byte-identity / width-purity / MUL_MAT_ID / MTP / coherence gates, and the repo's
    // default-on policy says a beneficial feature ships on.  It rides the device-policy path, so the
    // tally is also gated on `g_devmap` below (with `DEVMAP=0` there is no consumer and it is inert).
    g_prefill_seed   = env_int("MOE_EXPERT_CACHE_PREFILL_SEED", 1) != 0;
    // Provisional pre-fill/seed slots: a pre-filled expert the policy has never hit is treated as an
    // empty slot for admission (the touch doorkeeper is bypassed for it) while still serving hits.  This
    // is what makes arbitrary load-time pre-fill behave like the empty arena PLUS its bonus hits, and it
    // is what lets the prompt seed's not-yet-hit entries be reclaimed (measured 2026-09-29 session 14:
    // seed `MIB=16384` 79.8 -> 80.6 t/s, `MIB=8192` 57.2 -> 57.8).  Default ON (kill switch
    // `MOE_EXPERT_CACHE_PROVISIONAL=0`); inert until something is pre-filled/seeded, so the default is
    // behaviourally a no-op for an empty arena.
    g_prov_evict     = env_int("MOE_EXPERT_CACHE_PROVISIONAL", 1) != 0;
    g_prefill_seed_n = env_int("MOE_EXPERT_CACHE_PREFILL_SEED_N", 0);
    if (g_prefill_seed_n < 0) {
        g_prefill_seed_n = 0;
    }
    if (g_period < 1) {
        g_period = 1;
    }
}

int name_layer(const ggml_tensor * t) {
    const char * n = (t != nullptr) ? t->name : "";
    const char * d = strrchr(n, '-');
    if (d != nullptr && d[1] != '\0') {
        return atoi(d + 1);       // op output name: ffn_moe_gate-12
    }
    if (strncmp(n, "blk.", 4) == 0) {
        return atoi(n + 4);       // weight name: blk.12.ffn_gate_exps.weight
    }
    return -1;
}

std::string name_role(const ggml_tensor * t) {
    std::string n = (t != nullptr) ? t->name : "?";
    const size_t d = n.rfind('-');
    if (d != std::string::npos) {
        n.resize(d);
    }
    return n;
}

// Canonical role token shared by the host master and the (meta-copy-named) simple tensor:
// `blk.5.ffn_gate_exps.weight` and `Meta(ROCm0,ROCm1)#blk.5.ffn_gate_exps.weight#0` both normalize to
// `ffn_gate_exps.weight`.  Used only as the semantic consumer-lookup key.
std::string sem_role(const ggml_tensor * t) {
    std::string n = (t != nullptr) ? t->name : "?";
    const size_t first = n.find('#');
    const size_t last  = n.rfind('#');
    if (first != std::string::npos && last > first + 1 && last + 1 < n.size()) {
        n = n.substr(first + 1, last - first - 1);
    }
    if (n.compare(0, 4, "blk.") == 0) {
        const size_t d = n.find('.', 4);
        if (d != std::string::npos) {
            n = n.substr(d + 1);
        }
    }
    const size_t dash = n.rfind('-');
    if (dash != std::string::npos && dash + 1 < n.size() && n[dash + 1] >= '0' && n[dash + 1] <= '9') {
        n.resize(dash);
    }
    return n;
}

void decay_locked(table_t & t) {
    if (t.clock - t.last_decay < g_period) {
        return;
    }
    for (auto it = t.count.begin(); it != t.count.end(); ++it) {
        *it >>= 1;
    }
    for (auto it = t.ghost.begin(); it != t.ghost.end(); ++it) {
        *it >>= 1;
    }
    t.last_decay = t.clock - (t.clock % g_period);
}

// LFRU admission + fill.  Mutates residency; returns the alias.  The fill is async on `stream`
// because this runs inside CUDA graph capture (the scheduler's own expert copies are async for the
// same reason); pass nullptr for a synchronous-context caller (the self-test).
// Invariant: `slot_expert` and `expert_slot` must describe the same occupancy.  A desync is exactly
// the class of bug that makes one expert's remap point at another expert's bytes under eviction.
void check_invariant(const table_t & t) {
    int occupied = 0;
    for (int s = 0; s < t.slots; s++) {
        const int32_t e = t.slot_expert[s];
        if (e < 0) {
            continue;
        }
        occupied++;
        const auto it = t.expert_slot.find(e);
        if (it == t.expert_slot.end() || it->second != s) {
            GGML_LOG_ERROR("%s: DESYNC layer=%d role=%s slot=%d e=%d map=%d\n",
                           __func__, t.layer, t.role.c_str(), s, e, it == t.expert_slot.end() ? -99 : it->second);
        }
    }
    if (occupied != (int) t.expert_slot.size()) {
        GGML_LOG_ERROR("%s: COUNT layer=%d role=%s occupied=%d map=%zu slots=%d\n",
                       __func__, t.layer, t.role.c_str(), occupied, t.expert_slot.size(), t.slots);
    }
}

// ---- wip/host-expert-dio-cache Phase 2: the bounded pinned host tier (L2) -------------------------

// One O_DIRECT read (the debug fallback; the default fill is a plain buffered `pread`) of `len` bytes at
// `off` into `dst`, through a process-global aligned bounce buffer (O_DIRECT needs an aligned buffer,
// length and offset).  Caller holds `g_mutex`.
static bool pool_dio_read(int fd, void * dst, size_t len, size_t off, size_t align) {
    if (align == 0 || (align & (align - 1)) != 0) {
        align = 4096;
    }
    if (off % align == 0 && len % align == 0 && ((uintptr_t) dst % align) == 0) {
        const ssize_t n = pread(fd, dst, len, (off_t) off);
        return n == (ssize_t) len;
    }
    const size_t aoff = off & ~(align - 1);
    const size_t skip = off - aoff;
    const size_t blen = (skip + len + align - 1) & ~(align - 1);
    static void * bounce      = nullptr;
    static size_t bounce_size = 0;
    if (bounce == nullptr || bounce_size < blen) {
        free(bounce);
        bounce = nullptr;
        bounce_size = 0;
        void * p = nullptr;
        if (posix_memalign(&p, align, blen) != 0) {
            return false;
        }
        bounce = p;
        bounce_size = blen;
    }
    const ssize_t n = pread(fd, bounce, blen, (off_t) aoff);
    if (n < 0 || (size_t) n < skip + len) {
        return false;
    }
    memcpy(dst, (const char *) bounce + skip, len);
    return true;
}

static char * pool_slot_locked(host_pool_t & p, int expert, int * out_slot, bool sync_on_evict);
static void   pool_prepopulate_locked(host_pool_t & p, const table_t & t);

// The pool for `t`'s host tensor, created lazily and shared by every per-device table of that tensor.
// Caller holds `g_mutex`.  Returns null (and every fill falls back to the full master) when the pool is
// off, the source is unknown, or the budget is too small for one expert.
static host_pool_t * table_pool_locked(table_t & t) {
    if (!g_pool_enabled || !t.src_known || t.host == nullptr || t.host_bytes == 0 || t.n_experts <= 0) {
        return nullptr;
    }
    const std::pair<const void *, int> key = { t.host, t.device };
    const auto it = g_host_pools.find(key);
    if (it != g_host_pools.end()) {
        return it->second.failed ? nullptr : &it->second;
    }
    std::set<const void *> hosts;
    for (const table_t & tt : g_tables) {
        if (tt.src_known && tt.host != nullptr && tt.device == t.device) {
            hosts.insert(tt.host);
        }
    }
    const size_t n_hosts  = hosts.empty() ? 1 : hosts.size();
    const size_t per_host = (size_t) g_pool_total_bytes / n_hosts;
    int slots = (int) (per_host / t.host_bytes);
    if (slots > t.n_experts) {
        slots = t.n_experts;
    }
    device_guard pdg(t.device);
    host_pool_t p;
    if (slots >= 1) {
        void * mem = nullptr;
        if (cudaMallocHost(&mem, (size_t) slots * t.host_bytes) == cudaSuccess) {
            p.base        = (char *) mem;
            p.slots       = slots;
            p.host_bytes  = t.host_bytes;
            p.n_experts   = t.n_experts;
            p.path        = t.src_path;
            p.file_offs   = t.src_offs;
            p.slot_expert.assign((size_t) slots, -1);
            p.last.assign((size_t) t.n_experts, 0);
            p.slot_ev.assign((size_t) slots, nullptr);
            for (int s = 0; s < slots; s++) {
                cudaEvent_t ev = nullptr;
                if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess) {
                    (void) cudaGetLastError();
                    ev = nullptr;
                }
                p.slot_ev[(size_t) s] = ev;
            }
        } else {
            (void) cudaGetLastError();
            p.failed = true;
        }
    } else {
        p.failed = true;
    }
    const auto res = g_host_pools.emplace(key, std::move(p));
    if (!res.first->second.failed && g_pool_prewarm) {
        pool_prepopulate_locked(res.first->second, t);   // DIO-fill the slots now, ranked by the prefill routing
    }
    return res.first->second.failed ? nullptr : &res.first->second;
}

// Ensure `expert` is in the pool (O_DIRECT fill on a miss, LRU eviction) and return its slot base (and
// the slot index), or null when the pool is unusable (the caller then reads the full master).  A slot's
// arena fill is an async copy, so before a slot is REUSED its previous copy is drained via `slot_ev`
// (a per-slot event; waits only for that copy, not the whole stream).  Caller holds `g_mutex`.
static char * pool_slot_locked(host_pool_t & p, int expert, int * out_slot, bool sync_on_evict) {
    if (expert < 0 || expert >= p.n_experts) {
        return nullptr;
    }
    const auto hit = p.expert_slot.find(expert);
    if (hit != p.expert_slot.end()) {
        p.hits++;
        p.last[(size_t) expert] = ++p.clock;
        if (out_slot != nullptr) {
            *out_slot = hit->second;
        }
        return p.base + (size_t) hit->second * p.host_bytes;
    }
    p.misses++;
    int slot = -1;
    for (int s = 0; s < p.slots; s++) {
        if (p.slot_expert[(size_t) s] < 0) {
            slot = s;
            break;
        }
    }
    if (slot < 0) {
        int64_t best   = 0;
        int     victim = -1;
        for (int s = 0; s < p.slots; s++) {
            const int32_t e = p.slot_expert[(size_t) s];
            if (e < 0) {
                continue;
            }
            if (victim < 0 || p.last[(size_t) e] < best) {
                best   = p.last[(size_t) e];
                victim = e;
                slot   = s;
            }
        }
        if (victim >= 0) {
            // Wait for this slot's previous copy before overwriting it -- but only when the current pass
            // could have enqueued that copy itself (a pass that fills more experts than the pool holds).
            // The scheduler synchronises the backend between tokens, so a slot whose last copy was a
            // previous token's is already drained; syncing here would wait for THIS token's graph too.
            if (sync_on_evict && p.slot_ev[(size_t) slot] != nullptr) {
                if (cudaEventSynchronize(p.slot_ev[(size_t) slot]) != cudaSuccess) {
                    (void) cudaGetLastError();
                }
            }
            p.expert_slot.erase(victim);
            p.slot_expert[(size_t) slot] = -1;
            p.evictions++;
        }
    }
    if (slot < 0) {
        return nullptr;
    }
    if (p.fd < 0) {
        p.fd = open(p.path.c_str(), O_RDONLY | (g_pool_dio ? O_DIRECT : 0));
        if (p.fd < 0) {
            p.failed = true;
            return nullptr;
        }
        struct stat st;
        if (fstat(p.fd, &st) == 0 && st.st_blksize > 0) {
            p.align = (size_t) st.st_blksize;
        }
    }
    const size_t off = p.file_offs + (size_t) expert * p.host_bytes;
    char *       dst = p.base + (size_t) slot * p.host_bytes;
    const bool ok = g_pool_dio ? pool_dio_read(p.fd, dst, p.host_bytes, off, p.align)
                               : (pread(p.fd, dst, p.host_bytes, (off_t) off) == (ssize_t) p.host_bytes);
    if (!ok) {
        p.failed = true;
        return nullptr;
    }
    p.slot_expert[(size_t) slot] = expert;
    p.expert_slot[expert]        = slot;
    p.last[(size_t) expert]      = ++p.clock;
    p.fills++;
    if (out_slot != nullptr) {
        *out_slot = slot;
    }
    return dst;
}

// Pre-warm a freshly created pool: fill every slot from the GGUF with O_DIRECT while there is no decode
// latency to lose.  Experts are ranked by the prefill routing (`t.prefill_count`) when it is available, so
// the decode's hot set is resident; otherwise experts are taken in order.  This is the whole point of a
// DIO tier: the SSD bandwidth is spent once, up front, not in per-token stalls.  Caller holds `g_mutex`.
static void pool_prepopulate_locked(host_pool_t & p, const table_t & t) {
    // Rank the experts to warm with.  The arena's current residents are the hot set the cache has already
    // learned (the prefill seed for the prompt, plus any admission), so put those first; a host prefill
    // tally refines the order; then everything else in id order.  This is what makes the prewarm useful:
    // a sequential warm would miss the prompt's experts and the decode would DIO on every token.
    std::vector<uint8_t> seen((size_t) p.n_experts, 0);
    std::vector<int32_t> rank;
    rank.reserve((size_t) p.n_experts);
    const bool have_tally = (int) t.prefill_count.size() == p.n_experts;

    std::vector<int32_t> residents;
    for (int32_t e : t.slot_expert) {
        if (e >= 0 && e < p.n_experts && !seen[(size_t) e]) {
            seen[(size_t) e] = 1;
            residents.push_back(e);
        }
    }
    if (have_tally) {
        std::sort(residents.begin(), residents.end(), [&](int32_t a, int32_t b) {
            if (t.prefill_count[(size_t) a] != t.prefill_count[(size_t) b]) {
                return t.prefill_count[(size_t) a] > t.prefill_count[(size_t) b];
            }
            return a < b;
        });
    }
    for (int32_t e : residents) {
        rank.push_back(e);
    }
    if (have_tally) {
        std::vector<int32_t> rest;
        for (int e = 0; e < p.n_experts; e++) {
            if (!seen[(size_t) e]) {
                rest.push_back(e);
            }
        }
        std::sort(rest.begin(), rest.end(), [&](int32_t a, int32_t b) {
            if (t.prefill_count[(size_t) a] != t.prefill_count[(size_t) b]) {
                return t.prefill_count[(size_t) a] > t.prefill_count[(size_t) b];
            }
            return a < b;
        });
        for (int32_t e : rest) {
            rank.push_back(e);
        }
    } else {
        for (int e = 0; e < p.n_experts; e++) {
            if (!seen[(size_t) e]) {
                rank.push_back(e);
            }
        }
    }
    const int k = p.slots < p.n_experts ? p.slots : p.n_experts;
    int filled = 0;
    for (int i = 0; i < k; i++) {
        if (pool_slot_locked(p, rank[(size_t) i], nullptr, false) == nullptr) {
            break;
        }
        filled++;
    }
    if (filled > 0) {
        GGML_LOG_INFO("%s: prewarmed host pool layer=%d role=%s: %d/%d slots (%zu arena-hot first)\n",
                      __func__, t.layer, t.role.c_str(), filled, p.slots, residents.size());
    }
}

// `protect`/`n_protect`: experts that belong to the token currently being staged and must NOT be chosen as
// an eviction victim.  Without this, an expert admitted earlier in the same token can be evicted by a later
// one (its freshly-seeded count is the smallest), which is what made the hook's `all` flag flip from token
// to token - and that variability is what broke CUDA graph replay.
moe_cache_alias access_locked(table_t & t, int32_t expert, void * stream,
                              const int32_t * protect = nullptr, int n_protect = 0,
                              bool * out_cold = nullptr) {
    // The cold read indexes the host master with the table's own host geometry (`moe_cache_get_cold`
    // returns the per-expert and per-row strides), so a Phase 3 slice is servable in place too.
    const bool cold_ok = table_cold_ok(t);
    const auto is_protected = [&](int32_t e) {
        for (int i = 0; i < n_protect; i++) {
            if (protect[i] == e) {
                return true;
            }
        }
        return false;
    };
    const int admit = g_admit;
    t.clock++;
    decay_locked(t);

    const auto hit_it = t.expert_slot.find(expert);
    if (hit_it != t.expert_slot.end()) {
        t.hits++;
        t.count[expert]++;
        t.last[expert] = t.clock;
        if (hit_it->second >= 0 && hit_it->second < (int) t.slot_prov.size()) {
            t.slot_prov[hit_it->second] = 0;   // hit: no longer provisional
        }
        g_acc_hits++;
        return { (char *) t.arena + (size_t) hit_it->second * t.expert_bytes, true };
    }

    t.misses++;
    // Every miss is a touch of an expert we do NOT hold, i.e. demand we are not serving.  The
    // admission rules below compare this demand against the value a resident has demonstrated
    // (`count`).  Counting the current touch in BEFORE the decision is what makes the rules
    // non-degenerate: without it a never-resident expert would always look like zero demand.
    if (expert >= 0 && expert < (int32_t) t.ghost.size()) {
        t.ghost[expert]++;
    }

    if (t.slots <= 0) {
        if (cold_ok && out_cold != nullptr) {
            *out_cold = true;
            g_acc_colds++;
            g_cold_bytes += (int64_t) t.expert_bytes;
        }
        return { nullptr, false };
    }
    int slot = -1;
    if ((int) t.expert_slot.size() < t.slots) {
        for (int s = 0; s < t.slots; s++) {
            if (t.slot_expert[s] < 0) { slot = s; break; }
        }
    }
    if (slot < 0) {
        // evict the resident with the smallest decaying count; tie -> oldest use.
        int32_t victim = -1;
        for (int s = 0; s < t.slots; s++) {
            const int32_t e = t.slot_expert[s];
            if (e < 0 || is_protected(e)) { continue; }
            if (victim < 0 || t.count[e] < t.count[victim] ||
                (t.count[e] == t.count[victim] && t.last[e] < t.last[victim])) {
                victim = e;
                slot   = s;
            }
        }
        // Phase 1b fill-vs-cold admission.  Only meaningful when a cold read is actually available
        // (`COLD=uva`); with no cold path a rejected miss could not be served at all, so every mode
        // fills and the policy is a no-op.  A rejected miss is still a miss and is served from the
        // pinned host slice, so the resident set converges on the hot experts without slot churn.
        bool   reject = false;
        if (slot >= 0 && cold_ok && out_cold != nullptr) {
            // Provisional pre-fill: a never-hit resident is treated as an empty slot for admission, so
            // the doorkeeper does not delay the real hot set behind it.
            const bool prov = g_prov_evict && slot < (int) t.slot_prov.size() && t.slot_prov[slot] != 0;
            if (!prov) {
                if (admit == 1) {
                    // value: hold the resident unless the incoming has demonstrated more demand.
                    reject = t.ghost[expert] <= t.count[victim];
                } else if (admit == 2) {
                    // touch: a first-touch expert is served cold; admit from the Nth touch on.
                    reject = t.ghost[expert] < g_touch;
                }
            }
        }
        if (reject) {
            g_cold_admits++;
            g_acc_colds++;
            g_cold_bytes += (int64_t) t.expert_bytes;
            *out_cold = true;
            return { nullptr, false };
        }
        if (slot >= 0) {
            t.expert_slot.erase(victim);
            t.slot_expert[slot] = -1;
            t.evictions++;
        }
    }
    if (slot < 0) {
        if (cold_ok && out_cold != nullptr) {
            *out_cold = true;
            g_acc_colds++;
            g_cold_bytes += (int64_t) t.expert_bytes;
        }
        return { nullptr, false };
    }

    t.slot_expert[slot] = expert;
    t.expert_slot[expert] = slot;
    t.slot_dirty = true;
    if (slot < (int) t.slot_prov.size()) {
        t.slot_prov[slot] = 0;   // a real (policy-chosen) admission is not provisional
    }
    t.count[expert]++;
    // The expert is resident now, so its outstanding demand is served: reset the doorkeeper.  A
    // later eviction starts the touch count from scratch, which is what makes `touch` a
    // "2nd use since last resident" rule rather than a lifetime-frequency rule.
    if (expert >= 0 && expert < (int32_t) t.ghost.size()) {
        t.ghost[expert] = 0;
    }
    t.last[expert] = t.clock;
    t.fills++;

    if (g_fill && t.host != nullptr) {
        g_acc_fills++;
        g_fill_bytes += (int64_t) t.expert_bytes;
        void * dst = (char *) t.arena + (size_t) slot * t.expert_bytes;
        // Phase 2 (invisible L2): source the fill from the bounded pinned host pool when one exists; a
        // pool miss is filled from the GGUF with O_DIRECT first (host-side, synchronous).  A pool slot
        // is host-written and a later fill in this same pass can evict it, so a pool-sourced copy is
        // SYNCHRONOUS (the async form could let the next fill overwrite the slot before the copy runs).
        // With the pool off this is the original `t.host` async copy.
        char *        pool_expert = nullptr;
        int           pool_slot   = -1;
        host_pool_t * pool        = table_pool_locked(t);
        if (pool != nullptr) {
            // A pass that cannot fill more experts than the pool holds cannot evict a slot it just filled,
            // so no per-slot copy drain is needed (the previous token's copies are already done).
            const bool pool_sync = (n_protect > pool->slots);
            pool_expert = pool_slot_locked(*pool, expert, &pool_slot, pool_sync);
        }
        const bool from_pool = pool_expert != nullptr;
        const char * src = from_pool
                               ? (const char *) pool_expert + t.src_off
                               : (const char *) t.host + (size_t) expert * t.host_bytes + t.src_off;
        // Phase 3 slice fill.  An unsplit table (split_axis < 0) and a contiguous axis-1 slice both
        // copy `expert_bytes` in one 1-D transfer; an axis-0 slice is `rows` small rows strided by the
        // host expert pitch, so it needs a 2-D copy (the prefill campaign's §26.3 geometry).
        cudaError_t err = cudaSuccess;
        if (t.split_axis != 0 || t.host_pitch == 0) {
            err = cudaMemcpyAsync(dst, src, t.expert_bytes, cudaMemcpyHostToDevice,
                                  (cudaStream_t) stream);
        } else {
            const int64_t rows = t.host_bytes > 0 ? (int64_t) (t.host_bytes / t.host_pitch) : 0;
            const size_t  row  = rows > 0 ? (t.expert_bytes / (size_t) rows) : 0;
            if (row > 0) {
                err = cudaMemcpy2DAsync(dst, row, src, t.host_pitch, row, (size_t) rows,
                                        cudaMemcpyHostToDevice, (cudaStream_t) stream);
            }
        }
        if (from_pool && pool != nullptr && pool_slot >= 0 && stream != nullptr &&
                pool->slot_ev[(size_t) pool_slot] != nullptr) {
            // mark when this slot's copy has drained, so a later eviction can safely overwrite it
            (void) cudaEventRecord(pool->slot_ev[(size_t) pool_slot], (cudaStream_t) stream);
        }
        if (err != cudaSuccess) {
            GGML_LOG_WARN("%s: async fill failed table layer=%d role=%s expert=%d: %s\n",
                          __func__, t.layer, t.role.c_str(), expert, cudaGetErrorString(cudaGetLastError()));
        }
    }
    return { (char *) t.arena + (size_t) slot * t.expert_bytes, true };
}

// wip/moe-expert-cache prefill seed (MOE_EXPERT_CACHE_PREFILL_SEED=1): one-shot bulk admission of the
// experts the prefill routing touched most, applied at arena sizing (after the arena/policy state exist).
// `t.prefill_count` was accumulated in `moe_cache_update_host`'s prefill branch; here it is ranked
// (count desc, expert id asc), the top `min(slots, cap)` are placed in slots 0.. and copied from the
// pinned host master - the same bytes a decode fill would fetch, just front-loaded.  The host mirrors are
// populated so both the eager remap and the device-policy seed (`build_policy_descs_locked`) pick them
// up.  No-op for an identity table (already wholly resident) or one with no prefill tally.
void apply_prefill_seed_locked(table_t & t) {
    const bool from_tally = g_prefill_seed;
    if (!g_prefill_seed || t.identity || t.slots <= 0 || t.host == nullptr ||
        !t.expert_slot.empty()) {
        return;
    }
    if (from_tally && (int) t.prefill_count.size() != t.n_experts) {
        return;
    }
    // Rank order.  From the tally: descending prefill frequency (expert id asc on a tie).
    std::vector<int32_t> rank((size_t) t.n_experts);
    for (int e = 0; e < t.n_experts; e++) {
        rank[(size_t) e] = e;
    }
    if (from_tally) {
        int64_t total = 0;
        for (int e = 0; e < t.n_experts; e++) {
            total += t.prefill_count[(size_t) e];
        }
        if (total <= 0) {
            return;
        }
        std::sort(rank.begin(), rank.end(), [&](int32_t a, int32_t b) {
            if (t.prefill_count[(size_t) a] != t.prefill_count[(size_t) b]) {
                return t.prefill_count[(size_t) a] > t.prefill_count[(size_t) b];
            }
            return a < b;
        });
    }
    int k = t.n_experts;
    if (g_prefill_seed_n > 0 && g_prefill_seed_n < k) {
        k = g_prefill_seed_n;
    }
    if (k > t.slots) {
        k = t.slots;
    }
    const int64_t rows = (t.split_axis == 0 && t.host_pitch > 0 && t.host_bytes > 0)
                             ? (int64_t) (t.host_bytes / t.host_pitch) : 0;
    const size_t  row  = rows > 0 ? (t.expert_bytes / (size_t) rows) : 0;
    int seeded = 0;
    bool ok = true;
    for (int i = 0; i < k; i++) {
        const int32_t e  = rank[(size_t) i];
        void *        dst = (char *) t.arena + (size_t) i * t.expert_bytes;
        const char *  src = (const char *) t.host + (size_t) e * t.host_bytes + t.src_off;
        cudaError_t err = cudaSuccess;
        if (t.split_axis != 0 || t.host_pitch == 0) {
            err = cudaMemcpyAsync(dst, src, t.expert_bytes, cudaMemcpyHostToDevice, (cudaStream_t) 0);
        } else if (row > 0) {
            err = cudaMemcpy2DAsync(dst, row, src, t.host_pitch, row, (size_t) rows,
                                    cudaMemcpyHostToDevice, (cudaStream_t) 0);
        } else {
            ok = false;   // the host geometry is not representable by a single fill: stop seeding
            break;
        }
        if (err != cudaSuccess) {
            (void) cudaGetLastError();
            ok = false;
            break;
        }
        t.slot_expert[i] = e;
        t.expert_slot[e] = i;
        t.slot_prov[i]  = 1;   // provisional until first hit (MOE_EXPERT_CACHE_PROVISIONAL=1)
        // `count` is "uses while RESIDENT", and a pre-filled expert has not been used yet - so seed it 0,
        // NOT 1.  Seeding 1 made the arbitrary experts outrank a freshly-admitted real hot expert (also
        // count 1, but with a newer `last`), so the wrong residents resisted eviction and the hit rate
        // lagged the lazy path for ~800 tokens.  With count 0 they are evicted first, exactly as if the
        // slot had been empty.
        t.count[e]       = 0;
        t.ghost[e]       = 0;
        t.last[e]        = 0;
        seeded++;
    }
    if (ok && seeded > 0 && cudaDeviceSynchronize() != cudaSuccess) {
        (void) cudaGetLastError();
        ok = false;
    }
    if (seeded > 0) {
        t.fills    += seeded;
        t.slot_dirty = true;
        GGML_LOG_WARN("%s: layer=%d role=%s prefill-%s %d/%d slots (%d experts; %lld prefill tokens)\n",
                      __func__, t.layer, t.role.c_str(), from_tally ? "seeded" : "loaded",
                      seeded, t.slots, t.n_experts, (long long) t.prefill_tokens);
    }
}

// Coalesced copy of `n` bytes with `nt` threads.  dst/src are assumed 8-byte aligned for the bulk
// (the arena slot, the host expert base and the per-expert strides are all quant-block aligned); a
// short byte tail handles any remainder.  2-D (strided axis-0) fills call this once per row.
static __device__ __forceinline__ void moe_cache_policy_copy(char * __restrict__ dst, const char * __restrict__ src,
                                                             size_t n, int tid, int nt) {
    // A strided axis-0 row can start at a non-8-byte offset; fall back to a byte copy there.
    if ((((uintptr_t) dst) & 7u) != 0 || (((uintptr_t) src) & 7u) != 0) {
        for (size_t j = (size_t) tid; j < n; j += (size_t) nt) {
            dst[j] = src[j];
        }
        return;
    }
    size_t i = (size_t) tid * 8;
    for (; i + 8 <= n; i += (size_t) nt * 8) {
        *reinterpret_cast<uint64_t *>(dst + i) = *reinterpret_cast<const uint64_t *>(src + i);
    }
    const size_t base = (n / 8) * 8;
    for (size_t j = base + (size_t) tid; j < n; j += (size_t) nt) {
        dst[j] = src[j];
    }
}

// Prefill-seed fill (session 14): one expert, from the device-accessible host alias into its arena slot.
// One tiny launch per seed entry (<= slots per table), reading the same UVA alias the policy kernel
// fills from, so a `-sm tensor` pageable host master (a host `cudaMemcpyAsync` here measured ~1 GB/s)
// is served at device-read speed.  `do2d` covers a strided axis-0 slice via `rows` rows of `row_bytes`.
// Modelled on the policy kernel's fill phase: `blockIdx.x` selects a seeded expert (one block per
// expert), `slots[blockIdx.x]` its arena slot and `experts[blockIdx.x]` its host master index.
static __global__ void moe_cache_seed_fill_kernel(char * __restrict__ arena, const char * __restrict__ host_dev,
                                                  const int32_t * __restrict__ slots,
                                                  const int32_t * __restrict__ experts, int n,
                                                  int64_t expert_bytes, int64_t host_bytes, int64_t src_off,
                                                  int64_t rows, int64_t row_bytes, int64_t host_pitch, int do2d) {
    const int i = blockIdx.x;
    if (i >= n) {
        return;
    }
    char *       dst = arena + (int64_t) slots[i] * expert_bytes;
    const char * src = host_dev + (int64_t) experts[i] * host_bytes + src_off;
    const int tid = threadIdx.x;
    const int nt  = blockDim.x;
    if (!do2d) {
        moe_cache_policy_copy(dst, src, (size_t) expert_bytes, tid, nt);
        return;
    }
    for (int64_t r = 0; r < rows; r++) {
        moe_cache_policy_copy(dst + (size_t) r * (size_t) row_bytes,
                              src + (size_t) r * (size_t) host_pitch, (size_t) row_bytes, tid, nt);
    }
}

// Host-weight expert gather (B2, session 15): for an offloaded `MUL_MAT_ID` expert upload, copy only the
// experts the routing selects, straight from the host master into the scheduler's device `input_cpy`,
// ON DEVICE.  This replaces the non-staging path's host ids readback (`ggml_backend_tensor_get_async` +
// a full device synchronize per op) plus its host-side `used_ids` scan and per-run `set_async` copies.
// `blockIdx.x` is the expert; each block scans the routing once (n_used*n_tok, <= 16k) and, if the expert
// is used, copies it via the same coalesced helper the seed/policy fills use (so a strided axis-0 slice
// is a 2-D copy).  The expert lands at its ORIGINAL offset in `input_cpy`, so the op's raw routing ids
// still index it and the output is bit-identical to the host path.
// Grid is `n_experts * n_split` blocks; `blockIdx.x % n_experts` is the expert, `blockIdx.x /
// n_experts` a byte/row chunk of it.  The split matters: a prefill ubatch uses only ~8 of the 256
// experts, so one block per expert would copy with ~8 resident blocks (almost no memory-level
// parallelism); n_split restores ~64-128 concurrent blocks.  The used-scan is thread-parallel (each
// thread checks a strided slice of the routing) - a per-thread full scan cost 256 redundant loads per
// thread and dominated the kernel.
static __global__ void moe_cache_gather_kernel(
        char * __restrict__ dst_base, const char * __restrict__ src_base,
        const int32_t * __restrict__ ids, size_t nb0, size_t nb1,
        int n_used, int n_tok, int n_experts, int n_split,
        int64_t expert_bytes, int64_t host_bytes, int64_t src_off, int64_t host_pitch, int split_axis) {
    const int e = blockIdx.x % n_experts;
    const int s = blockIdx.x / n_experts;

    __shared__ int used_s;
    if (threadIdx.x == 0) {
        used_s = 0;
    }
    __syncthreads();
    const int total = n_used * n_tok;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        const int tok = i / n_used;
        const int j   = i - tok * n_used;
        const int32_t x = *(const int32_t *) ((const char *) ids + (size_t) tok * nb1 + (size_t) j * nb0);
        if (x == e) { used_s = 1; break; }
    }
    __syncthreads();
    if (!used_s) {
        return;
    }

    const int tid = threadIdx.x;
    const int nt  = blockDim.x;
    char *       dst = dst_base + (int64_t) e * expert_bytes;
    const char * src = src_base + (int64_t) e * host_bytes + src_off;
    if (split_axis != 0 || host_pitch == 0) {
        const int64_t chunk = (expert_bytes + n_split - 1) / n_split;
        const int64_t off   = (int64_t) s * chunk;
        const int64_t len   = expert_bytes - off;
        if (len <= 0) {
            return;
        }
        moe_cache_policy_copy(dst + off, src + off, (size_t) (len < chunk ? len : chunk), tid, nt);
    } else {
        const int64_t rows = host_bytes > 0 ? host_bytes / host_pitch : 0;
        const size_t  row  = rows > 0 ? (size_t) expert_bytes / (size_t) rows : 0;
        const int64_t rpb  = (rows + n_split - 1) / n_split;
        const int64_t r0   = (int64_t) s * rpb;
        const int64_t r1   = r0 + rpb < rows ? r0 + rpb : rows;
        for (int64_t r = r0; r < r1; r++) {
            moe_cache_policy_copy(dst + (size_t) r * row, src + (size_t) r * host_pitch, row, tid, nt);
        }
    }
}

// Generalised seed placement for a NON-empty arena (session 14 live path).  `rank` is descending
// prefill frequency (expert id asc on a tie).  Free slots are used first; if none, the lowest-count
// resident that is NOT in the seed set is evicted - a provisional entry has count 0, so a stale
// pre-fill is reclaimed first, exactly like an empty slot.  The fills run on `stream` (one tiny kernel
// per expert, reading the device-accessible host alias) and are ordered before the following policy
// kernel.  A device fill rather than a host `cudaMemcpyAsync` because under `-sm tensor` the host master
// is a pageable mmap: a pageable H2D measured ~1 GB/s (25k calls = 8 s/device), while the GPU reading
// the same UVA alias is the fast path the policy kernel already uses.  Returns the number placed.
int apply_prefill_seed_rank_locked(table_t & t, const std::vector<int32_t> & rank, int k, void * stream) {
    if (t.identity || t.slots <= 0 || t.host == nullptr || k <= 0) {
        return 0;
    }
    std::vector<uint8_t> protect((size_t) t.n_experts, 0);
    for (int i = 0; i < k && i < (int) rank.size(); i++) {
        const int32_t e = rank[(size_t) i];
        if (e >= 0 && e < t.n_experts) {
            protect[(size_t) e] = 1;
        }
    }
    const int64_t rows = (t.split_axis == 0 && t.host_pitch > 0 && t.host_bytes > 0)
                             ? (int64_t) (t.host_bytes / t.host_pitch) : 0;
    const size_t  row  = rows > 0 ? (t.expert_bytes / (size_t) rows) : 0;
    std::vector<int32_t> pair_slot, pair_exp;
    pair_slot.reserve((size_t) k);
    pair_exp.reserve((size_t) k);
    int placed = 0;
    for (int i = 0; i < k && i < (int) rank.size(); i++) {
        const int32_t e = rank[(size_t) i];
        if (e < 0 || e >= t.n_experts) {
            continue;
        }
        if (t.expert_slot.find(e) != t.expert_slot.end()) {
            continue;   // already resident
        }
        int slot = -1;
        for (int s = 0; s < t.slots; s++) {
            if (t.slot_expert[(size_t) s] < 0) { slot = s; break; }
        }
        int victim = -1;
        if (slot < 0) {
            int64_t vcount = 0, vlast = 0;
            for (int s = 0; s < t.slots; s++) {
                const int32_t ce = t.slot_expert[(size_t) s];
                if (ce < 0 || ce >= t.n_experts || protect[(size_t) ce]) {
                    continue;
                }
                const int64_t c = (size_t) ce < t.count.size() ? t.count[(size_t) ce] : 0;
                const int64_t l = (size_t) ce < t.last.size()  ? t.last[(size_t) ce]  : 0;
                if (victim < 0 || c < vcount || (c == vcount && l < vlast)) {
                    victim = ce; slot = s; vcount = c; vlast = l;
                }
            }
            if (slot >= 0 && victim >= 0) {
                t.expert_slot.erase(victim);
                t.slot_expert[(size_t) slot] = -1;
                t.evictions++;
            }
        }
        if (slot < 0) {
            continue;   // every slot holds a protected seed expert already
        }
        if (t.host_dev == nullptr) {
            break;   // no device-accessible host alias: cannot fill
        }
        const bool do2d = (t.split_axis == 0 && t.host_pitch > 0);
        if (do2d && row == 0) {
            break;   // the host geometry is not representable by a single fill
        }
        pair_slot.push_back(slot);
        pair_exp.push_back(e);
        t.slot_expert[(size_t) slot] = e;
        t.expert_slot[e]             = slot;
        if ((int) t.slot_prov.size() == t.slots) {
            t.slot_prov[(size_t) slot] = 1;   // provisional until first hit (MOE_EXPERT_CACHE_PROVISIONAL=1)
        }
        t.count[(size_t) e] = 0;
        t.ghost[(size_t) e] = 0;
        t.last[(size_t) e]  = 0;
        placed++;
    }
    // ONE fill launch for the whole table (a block per seeded expert, `blockIdx.x` = expert).  A launch
    // per expert measured ~0.6 ms of host launch overhead x ~25k experts = ~15 s/device; one launch per
    // table is ~10 us.  The pair list is staged in `remap_dev`, which is free between graphs and is
    // overwritten by the next token's remap only after this fill (same stream, ordered).
    if (placed > 0 && 2 * placed <= (int) t.remap_cap) {
        std::vector<int32_t> staging((size_t) 2 * placed);
        for (int i = 0; i < placed; i++) {
            staging[(size_t) i]          = pair_slot[(size_t) i];
            staging[(size_t) placed + i] = pair_exp[(size_t) i];
        }
        const bool do2d = (t.split_axis == 0 && t.host_pitch > 0);
        (void) cudaMemcpyAsync(t.remap_dev, staging.data(), staging.size() * sizeof(int32_t),
                               cudaMemcpyHostToDevice, (cudaStream_t) stream);
        moe_cache_seed_fill_kernel<<<placed, 256, 0, (cudaStream_t) stream>>>(
            (char *) t.arena, (const char *) t.host_dev, t.remap_dev, t.remap_dev + placed, placed,
            (int64_t) t.expert_bytes, (int64_t) t.host_bytes, (int64_t) t.src_off,
            rows, (int64_t) row, (int64_t) t.host_pitch, do2d ? 1 : 0);
        if (cudaGetLastError() != cudaSuccess) {
            (void) cudaGetLastError();
            GGML_LOG_ERROR("%s: seed fill launch failed layer=%d role=%s; slots may be stale\n",
                           __func__, t.layer, t.role.c_str());
        }
    }
    if (placed > 0) {
        t.fills += placed;
    }
    return placed;
}

// D2H the device-policy residency/counters into the host mirrors.  Under DEVPOLICY the host mirrors go
// stale (the kernel mutates only the device arrays), so the live seed snapshots them before placing.
void policy_pull_host_locked(table_t & t, int device) {
    if (!t.policy || t.slot_expert_dev == nullptr || t.slots <= 0) {
        return;
    }
    device_guard dg(device);
    t.slot_expert.assign((size_t) t.slots, -1);
    t.slot_prov.assign((size_t) t.slots, 0);
    t.count.assign((size_t) t.n_experts, 0);
    t.ghost.assign((size_t) t.n_experts, 0);
    t.last.assign((size_t) t.n_experts, 0);
    (void) cudaMemcpy(t.slot_expert.data(), t.slot_expert_dev, (size_t) t.slots * sizeof(int32_t), cudaMemcpyDeviceToHost);
    (void) cudaMemcpy(t.count.data(),       t.count_dev,       (size_t) t.n_experts * sizeof(int32_t), cudaMemcpyDeviceToHost);
    (void) cudaMemcpy(t.ghost.data(),       t.ghost_dev,       (size_t) t.n_experts * sizeof(int32_t), cudaMemcpyDeviceToHost);
    (void) cudaMemcpy(t.last.data(),        t.last_dev,        (size_t) t.n_experts * sizeof(int32_t), cudaMemcpyDeviceToHost);
    if (t.slot_prov_dev != nullptr && (int) t.slot_prov.size() == t.slots) {
        (void) cudaMemcpy(t.slot_prov.data(), t.slot_prov_dev, (size_t) t.slots * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    }
    t.expert_slot.clear();
    for (int s = 0; s < t.slots; s++) {
        const int32_t e = t.slot_expert[(size_t) s];
        if (e >= 0 && e < t.n_experts) {
            t.expert_slot[e] = s;
        }
    }
    t.slot_dirty = true;
}

// One-shot bulk admission of the prompt's hottest experts into a LIVE arena (MOE_EXPERT_CACHE_PREFILL_SEED=1,
// the device-tally path).  Called from the first decode-band policy flush after the prefill tally is
// non-empty; the seed takes effect from the NEXT token (this call is post-graph, before the batched policy
// kernel, which then treats the seeded slots as provisional).  A no-op until a prefill has tallied.
bool seed_prefill_lazy_locked(int device, void * stream) {
    if (!g_prefill_seed || device < 0 || device >= (int) g_policy_dev.size()) {
        return false;
    }
    policy_dev_t & pd = g_policy_dev[device];
    bool any = false;
    for (int id : pd.ids) {
        if (g_tables[id].prefill_tally_pending && !g_tables[id].prefill_seeded) { any = true; break; }
    }
    if (!any) {
        return false;
    }
    // If the policy is already live the host mirrors are stale; snapshot the true residency so a victim is
    // never a slot the device still holds.  On the very first flush the mirrors ARE the source (the policy
    // prebuild seeds them from the same mirrors), so no snapshot is needed.
    if (pd.initialized && pd.desc != nullptr) {
        for (int id : pd.ids) {
            policy_pull_host_locked(g_tables[id], device);
        }
    }
    int seeded_tables = 0, seeded_slots = 0;
    device_guard dg(device);
    for (int id : pd.ids) {
        table_t & t = g_tables[id];
        if (!t.prefill_tally_pending || t.prefill_seeded || t.n_experts <= 0) {
            continue;
        }
        t.prefill_seeded        = true;
        t.prefill_tally_pending = false;
        if (t.prefill_count_dev == nullptr || t.identity || t.slots <= 0 || t.host == nullptr) {
            continue;
        }
        std::vector<int32_t> tally((size_t) t.n_experts, 0);
        if (cudaMemcpy(tally.data(), t.prefill_count_dev, (size_t) t.n_experts * sizeof(int32_t),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            (void) cudaGetLastError();
            continue;
        }
        (void) cudaMemsetAsync(t.prefill_count_dev, 0, (size_t) t.n_experts * sizeof(int32_t), (cudaStream_t) stream);
        int64_t total = 0;
        std::vector<int32_t> rank((size_t) t.n_experts);
        for (int e = 0; e < t.n_experts; e++) {
            total += tally[(size_t) e];
            rank[(size_t) e] = e;
        }
        if (total <= 0) {
            continue;
        }
        std::sort(rank.begin(), rank.end(), [&](int32_t a, int32_t b) {
            if (tally[(size_t) a] != tally[(size_t) b]) return tally[(size_t) a] > tally[(size_t) b];
            return a < b;
        });
        int k = t.n_experts;
        if (g_prefill_seed_n > 0 && g_prefill_seed_n < k) {
            k = g_prefill_seed_n;
        }
        if (k > t.slots) {
            k = t.slots;
        }
        const int placed = apply_prefill_seed_rank_locked(t, rank, k, stream);
        if (placed > 0) {
            seeded_tables++;
            seeded_slots += placed;
        }
    }
    if (seeded_tables > 0) {
        GGML_LOG_WARN("%s: prompt-routing seed dev=%d: %d tables, %d provisional slots\n",
                      __func__, device, seeded_tables, seeded_slots);
        pd.initialized = false;   // force `build_policy_descs_locked` to resync the device from the host
        return true;
    }
    return false;
}

// Allocate `slots` (capped at n_experts) for one table.  Fail-soft: a failed cudaMalloc disables
// just this table (the op falls back to the full-table path for it).
// OPEN 2: the byte size of a table's arena for `slots` resident slots (the stride plus the head guard's
// pad).  Used to unmap exactly what was mapped.
static size_t table_arena_bytes(const table_t & t, int slots) {
    const size_t head_pad = t.expert_bytes < 512 ? t.expert_bytes : 512;
    return (size_t) slots * t.expert_bytes + head_pad;
}

// OPEN 2: release a table's arena backing.  The slab OWNS the arena's VA (the table's slab is a range of
// the one mapped slab), so freeing it only returns the range to the slab's free list; the plain path is a
// cudaMalloc the table owns outright.
static void free_arena_backing(int device, void * arena, size_t reserved, int old_slots, const table_t & t) {
    GGML_UNUSED(old_slots);
    if (arena == nullptr) {
        return;
    }
    if (reserved != 0) {
        ggml_cuda_slab_arena_free(device, arena, reserved);
    } else {
        (void) cudaFree(arena);
    }
}

// WIP item 1 (TODO #42): release every buffer a table owns so it can be rebuilt at a different slot
// count (the layer-uniform re-size below).  Mirrors the allocations in `alloc_table_locked`; used only
// at sizing time, before the arena is read, but the identity/seed fills enqueue async copies on stream
// 0, so synchronize first.  `prefill_count` is deliberately KEPT: the re-seed needs it.
static void free_table_buffers_locked(table_t & t) {
    device_guard dg(t.device >= 0 ? t.device : 0);
    (void) cudaDeviceSynchronize();   // an identity/seed fill may still be in flight on stream 0
    if (t.arena != nullptr) {
        free_arena_backing(t.device, t.arena, t.arena_reserved, t.slots, t);
        t.arena_reserved = 0;
        t.arena = nullptr;
    }
    if (t.remap_dev != nullptr){ (void) cudaFree(t.remap_dev); t.remap_dev = nullptr; }
    if (t.slot_dev != nullptr) { (void) cudaFree(t.slot_dev); t.slot_dev = nullptr; }
    if (t.used_dev != nullptr) { (void) cudaFree(t.used_dev); t.used_dev = nullptr; }
    if (t.slot_expert_dev != nullptr) { (void) cudaFree(t.slot_expert_dev); t.slot_expert_dev = nullptr; }
    if (t.count_dev != nullptr)  { (void) cudaFree(t.count_dev);  t.count_dev  = nullptr; }
    if (t.ghost_dev != nullptr)  { (void) cudaFree(t.ghost_dev);  t.ghost_dev  = nullptr; }
    if (t.last_dev != nullptr)   { (void) cudaFree(t.last_dev);   t.last_dev   = nullptr; }
    if (t.slot_prov_dev != nullptr) { (void) cudaFree(t.slot_prov_dev); t.slot_prov_dev = nullptr; }
    if (t.prefill_count_dev != nullptr) { (void) cudaFree(t.prefill_count_dev); t.prefill_count_dev = nullptr; }
    if (t.used_host != nullptr)  { (void) cudaFreeHost(t.used_host);  t.used_host  = nullptr; }
    if (t.used_host2 != nullptr) { (void) cudaFreeHost(t.used_host2); t.used_host2 = nullptr; }
    if (t.slot_pin != nullptr)   { (void) cudaFreeHost(t.slot_pin);   t.slot_pin   = nullptr; }
    (void) cudaGetLastError();   // fail soft: a bad free must not abort a run

    g_arena_bytes -= (int64_t) t.slots * (int64_t) t.expert_bytes;
    if (g_arena_bytes < 0) {
        g_arena_bytes = 0;
    }
    t.slots          = 0;
    t.allocated      = false;
    t.identity       = false;
    t.devmap         = false;
    t.policy         = false;
    t.policy_pending = false;
    t.remap_fresh    = false;
    t.remap_cap      = 0;
    t.remap_n_used   = 0;
    t.remap_n_tok    = 0;
    t.used_cap       = 0;
    t.used_pending   = false;
    t.used_toggle    = 0;
    t.slot_dirty     = true;
    t.slot_expert.clear();
    t.slot_prov.clear();
    t.expert_slot.clear();
    t.slot_dev_host.clear();
    t.remap_host.clear();
    t.hook_experts.clear();
}

void alloc_table_locked(table_t & t, int slots) {
    // The arena and the remap staging are all raw cudaMalloc'd on
    // `t.device`, so make it current for the whole function.  `alloc_all_locked` runs the sweep on
    // the first table's owner device and this guard moves to each table's own device per iteration.
    device_guard dg(t.device);
    t.allocated = true;
    if (slots > t.n_experts) {
        slots = t.n_experts;
    }
    if (slots < 0) {
        slots = 0;
    }
    t.slots = slots;
    if (slots <= 0 || t.expert_bytes == 0) {
        return;
    }
    const int  arena_slots  = slots;
    // The quantized `MUL_MAT_ID` MMQ reads a full K tile, so the last row of a slot's expert over-reads
    // into the NEXT slot's head, and the head of an empty (never-filled) slot is uninitialized
    // `cudaMalloc` memory - usually NaN, and `NaN * 0 = NaN` poisons the tile -> the repeated-`/`
    // corruption.  Zero the first `min(expert_bytes, 512)` bytes of every slot once (the host upload
    // path's own guard value, `copy_experts`: `min(expert_size, 512)`), plus a 512-byte tail so the last
    // slot's over-read stays in-bounds.  Filled slots are overwritten with real expert data, and an
    // evicted slot keeps its finite bytes, so the invariant holds for the life of the arena with no
    // per-token work.  (This mirrors the gather's head guard; that guard alone did not cover the arena.)
    const size_t head_pad   = t.expert_bytes < 512 ? t.expert_bytes : 512;
    // WIP r42 (TODO #42): a failed slot allocation no longer drops the table to 0 slots.  Try the
    // requested size; then the exact largest size the free VRAM can hold (`cudaMemGetInfo` -- one call,
    // lands *at* the max); then a 0.95 geometric descent for the fragmentation case (a failed
    // cudaMalloc is cheap, so the extra attempts cost nothing measurable).  The table keeps the largest
    // slot count that actually allocates.
    const bool use_slab = ggml_cuda_slab_enabled();
    const auto arena_size_of = [&](int n) -> size_t {
        return (size_t) n * t.expert_bytes + head_pad;
    };
    int got_slots = 0;
    auto try_arena = [&](int n) -> bool {
        if (use_slab) {
            if (t.arena != nullptr) {
                return true;
            }
            const size_t want = arena_size_of(n);
            void * p = ggml_cuda_slab_arena_alloc(t.device, want);
            if (p == nullptr) {
                return false;   // the arena region below the slab boundary is full
            }
            t.arena = (char *) p;
            const size_t unit = ggml_cuda_slab_arena_unit(t.device);
            t.arena_reserved = unit > 0 ? unit * ((want + unit - 1) / unit) : want;
            return true;
        }
        if (cudaMalloc(&t.arena, (size_t) n * t.expert_bytes + head_pad) == cudaSuccess) {
            return true;
        }
        (void) cudaGetLastError();   // clear sticky error; fail soft (issue #33 lesson)
        t.arena = nullptr;
        return false;
    };
    if (use_slab) {
        // The slab rounds to arena units, so a smaller slot count can still fit a smaller free run.
        if (try_arena(arena_slots)) {
            got_slots = arena_slots;
        } else {
            for (int n = arena_slots * 95 / 100; got_slots == 0 && n >= 1; n = n * 95 / 100) {
                if (try_arena(n)) {
                    got_slots = n;
                }
            }
        }
    } else if (try_arena(arena_slots)) {
        got_slots = arena_slots;
    } else {
        size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess && t.expert_bytes > 0) {
            const size_t usable = free_b > head_pad ? free_b - head_pad : 0;
            const int fit = (int) std::min<size_t>(usable / t.expert_bytes, (size_t) arena_slots);
            if (fit > 0 && try_arena(fit)) {
                got_slots = fit;
            }
        }
        for (int n = arena_slots * 95 / 100; got_slots == 0 && n >= 1; n = n * 95 / 100) {
            if (try_arena(n)) {
                got_slots = n;
            }
        }
    }
    if (got_slots <= 0 || t.arena == nullptr) {
        g_alloc_failed++;
        GGML_LOG_WARN("%s: arena alloc failed for layer=%d role=%s (%d slots x %zu B); cache disabled for this table\n",
                      __func__, t.layer, t.role.c_str(), slots, t.expert_bytes);
        t.arena        = nullptr;
        t.slots        = 0;
        t.alloc_failed = true;
        return;
    }
    if (got_slots < arena_slots) {
        GGML_LOG_WARN("%s: layer=%d role=%s: %d slots requested, only %d fit; sizing the arena to %d slots\n",
                      __func__, t.layer, t.role.c_str(), arena_slots, got_slots, got_slots);
    }
    slots = got_slots;
    // `t.slots` is what every consumer (the remap, the fused kernels, `moe_cache_has_arena_locked`)
    // reads, so it MUST be the count actually allocated.  The r19 slot-count retry shrank the arena
    // but left `t.slots` at the requested count, so a shrunk table advertised more slots than its
    // arena holds -> the decode band read past the allocation (measured: `////` + MTP acc 0.007 on an
    // over-filled arena).
    t.slots        = got_slots;
    const size_t arena_size = (size_t) slots * t.expert_bytes + head_pad;
    (void) cudaMemset(t.arena, 0, arena_size);
    t.slot_expert.assign(slots, -1);
    t.slot_prov.assign(slots, 0);
    t.count.assign(t.n_experts, 0);
    t.ghost.assign(t.n_experts, 0);
    t.last.assign(t.n_experts, 0);
    // Allocate the remap staging HERE, not lazily in the hook: sizing runs once, for every table, so
    // the remap buffer is ready from the first hooked token.
    if (t.remap_dev == nullptr) {
        const int64_t cap = (int64_t) t.n_experts * 8;
        if (cudaMalloc((void **) &t.remap_dev, (size_t) cap * sizeof(int32_t)) == cudaSuccess) {
            t.remap_cap = cap;
        } else {
            (void) cudaGetLastError();   // fail soft: the hook falls back to the full-table path
            t.remap_dev = nullptr;
            t.remap_cap = 0;
        }
    }
    // Session 7 identity fast path.  A table whose arena holds EVERY expert needs no slot remap:
    // keep `slot == expert`, copy the whole table in once, and the consumer can index the arena with
    // the raw routing ids.  That removes the per-layer host ids readback + full device synchronize
    // that otherwise caps an all-resident cache far below the equivalent device-resident table.  The
    // copy is blocking and one-time (the sizing token pays it); it is ordered before any consumer
    // because the arena is not read until a later token.  A partial copy rolls the table back to the
    // normal remap path.
    t.identity = false;
    if (slots == t.n_experts && t.host != nullptr && t.expert_bytes > 0) {
        bool ok = true;
        const int64_t rows = (t.split_axis == 0 && t.host_pitch > 0 && t.host_bytes > 0)
                                 ? (int64_t) (t.host_bytes / t.host_pitch) : 0;
        const size_t  row  = (rows > 0) ? (t.expert_bytes / (size_t) rows) : 0;
        for (int e = 0; e < t.n_experts && ok; ++e) {
            void *       dst = (char *) t.arena + (size_t) e * t.expert_bytes;
            const char * src = (const char *) t.host + (size_t) e * t.host_bytes + t.src_off;
            cudaError_t err = cudaSuccess;
            if (t.split_axis != 0 || t.host_pitch == 0) {
                err = cudaMemcpyAsync(dst, src, t.expert_bytes, cudaMemcpyHostToDevice, (cudaStream_t) 0);
            } else if (row > 0) {
                err = cudaMemcpy2DAsync(dst, row, src, t.host_pitch, row, (size_t) rows,
                                        cudaMemcpyHostToDevice, (cudaStream_t) 0);
            } else {
                ok = false;
            }
            if (err != cudaSuccess) {
                (void) cudaGetLastError();
                ok = false;
            } else {
                t.slot_expert[e] = e;
                t.expert_slot[e] = e;
            }
        }
        if (ok && cudaDeviceSynchronize() != cudaSuccess) {
            (void) cudaGetLastError();
            ok = false;
        }
        if (ok) {
            t.identity = true;
            t.fills   += t.n_experts;
            GGML_LOG_WARN("%s: layer=%d role=%s is FULLY RESIDENT (%d/%d slots): identity fast path ON "
                          "(the decode band now reads the arena directly; no per-token routing readback)\n",
                          __func__, t.layer, t.role.c_str(), slots, t.n_experts);
        } else {
            t.expert_slot.clear();
            t.slot_expert.assign(slots, -1);
            GGML_LOG_WARN("%s: identity fill failed for layer=%d role=%s; keeping the remap path\n",
                          __func__, t.layer, t.role.c_str());
        }
    }
    // Device-remap mode (gentle-curve follow-up): a PARTIAL-residency table cannot keep `slot == expert`,
    // so allocate a device `expert -> slot` map (all -1 = cold) that the consumer's remap kernel reads.
    // The host refreshes it once per token from the deferred promotion pass.  Requirement: the table must
    // be COLD-SAFE, because the promotion lags one token and the current token's misses must be servable
    // through the UVA cold region.  Item 1 (session 8) made that true for a `-sm tensor` split whose host
    // slice geometry is representable (contiguous gate/up, strided axis-0 down), so split tables now
    if (g_devmap && !t.identity && table_cold_ok(t) && t.remap_dev != nullptr && t.n_experts > 0) {
        if (cudaMalloc((void **) &t.slot_dev, (size_t) t.n_experts * sizeof(int32_t)) == cudaSuccess) {
            const size_t used_cap = (size_t) t.n_experts * MOE_EXPERT_CACHE_MAX_TOK;
            if (cudaMalloc((void **) &t.used_dev, used_cap * sizeof(int32_t)) != cudaSuccess) {
                (void) cudaGetLastError();
                t.used_dev = nullptr;
            } else {
                t.used_cap = (int64_t) used_cap;
            }
            // The deferred promotion is per table per token; pageable D2H/H2D staging made it ~60 us
            // per table.  Pinned staging cuts that by ~5x and keeps the promotion off the critical path.
            if (t.used_dev != nullptr && cudaMallocHost((void **) &t.used_host, used_cap * sizeof(int32_t)) != cudaSuccess) {
                (void) cudaGetLastError();
                t.used_host = nullptr;
            }
            // Second pinned buffer for the PIPELINED (double-buffered) readback: each promote call
            // enqueues this token's async D2H and applies the policy to the previous call's data, whose
            // copy the backend synchronize between tokens has already completed.  A per-table
            // synchronous D2H on 240 tables cost ~4.8 ms/token (measured 2026-09-29).
            if (t.used_dev != nullptr && cudaMallocHost((void **) &t.used_host2, used_cap * sizeof(int32_t)) != cudaSuccess) {
                (void) cudaGetLastError();
                t.used_host2 = nullptr;
            }
            if (cudaMallocHost((void **) &t.slot_pin, (size_t) t.n_experts * sizeof(int32_t)) != cudaSuccess) {
                (void) cudaGetLastError();
                t.slot_pin = nullptr;
            }
            t.slot_dev_host.assign(t.n_experts, -1);
            (void) cudaMemcpy(t.slot_dev, t.slot_dev_host.data(), (size_t) t.n_experts * sizeof(int32_t),
                              cudaMemcpyHostToDevice);
            // The used-list buffer is what the deferred promotion reads (the routing tensor's own storage
            // is recycled once the graph completes, so reading it back later yields garbage).
            t.devmap = t.used_dev != nullptr && t.used_host != nullptr && t.used_host2 != nullptr && t.slot_pin != nullptr;
            // Prefill-seed tally (session 14, MOE_EXPERT_CACHE_PREFILL_SEED): allocate the device
            // histogram HERE, at sizing.  A lazy cudaMalloc on the first prefill op is a device
            // synchronize, and 240 of them interleaved with the prefill graph measured the prompt at
            // 207 -> 127 t/s (1.2 s).  Allocating once here keeps the prefill path allocation-free.
            if (g_prefill_seed && t.prefill_count_dev == nullptr) {
                if (cudaMalloc((void **) &t.prefill_count_dev, (size_t) t.n_experts * sizeof(int32_t)) == cudaSuccess) {
                    (void) cudaMemset(t.prefill_count_dev, 0, (size_t) t.n_experts * sizeof(int32_t));
                } else {
                    (void) cudaGetLastError();
                    t.prefill_count_dev = nullptr;
                }
            }
        } else {
            (void) cudaGetLastError();
            t.slot_dev = nullptr;
            t.devmap   = false;
        }
    }
    // Device-side admission policy state (MOE_EXPERT_CACHE_DEVPOLICY=1): the device mirrors of the
    // LFRU residency map + counters, updated in place by `moe_cache_policy_kernel`.  Only a devmap
    // table can use it, because the kernel replaces the host `access_locked` policy entirely.
    //
    // A SPLIT table (`split_axis >= 0`: a `-sm tensor` / Meta-split expert slice) must NOT use the
    // device policy.  Its policy kernel and its prefill seed are tuned for a whole, per-device expert
    // and measurably hurt the split case: 2 GPU IQ4_NL `-ncmoe 48`, MTP n3, n=1024 went 60.3 -> 70.7 t/s
    // (better than `-sm layer`) once the split tables fell back to the host promotion.  The unsplit
    // (`-sm layer`) case keeps the device policy -- that is where the seed pays off.  The split tables
    // still get the cache (the arena + slot remap); only the admission-policy engine changes.
    // `MOE_EXPERT_CACHE_DEVPOLICY_SPLIT=1` restores the old behaviour.
    static const bool devpolicy_split = [] {
        const char * e = getenv("MOE_EXPERT_CACHE_DEVPOLICY_SPLIT");
        return e != nullptr && atoi(e) != 0;
    }();
    // Phase 2: a pooled table must be filled on the host (the device-policy kernel fills from the full
    // master in-kernel and cannot DIO).  With the pool on, every table uses the host promotion path.
    if (g_devpolicy && !g_pool_enabled && t.devmap && (t.split_axis < 0 || devpolicy_split)) {
        device_guard pdg(t.device);
        bool ok = cudaMalloc((void **) &t.slot_expert_dev, (size_t) slots * sizeof(int32_t)) == cudaSuccess;
        ok = ok && cudaMalloc((void **) &t.count_dev, (size_t) t.n_experts * sizeof(int32_t)) == cudaSuccess;
        ok = ok && cudaMalloc((void **) &t.ghost_dev, (size_t) t.n_experts * sizeof(int32_t)) == cudaSuccess;
        ok = ok && cudaMalloc((void **) &t.last_dev,  (size_t) t.n_experts * sizeof(int32_t)) == cudaSuccess;
        ok = ok && cudaMalloc((void **) &t.slot_prov_dev, (size_t) slots * sizeof(uint8_t)) == cudaSuccess;
        if (ok) {
            (void) cudaMemset(t.slot_expert_dev, 0xFF, (size_t) slots * sizeof(int32_t));
            (void) cudaMemset(t.count_dev, 0, (size_t) t.n_experts * sizeof(int32_t));
            (void) cudaMemset(t.ghost_dev, 0, (size_t) t.n_experts * sizeof(int32_t));
            (void) cudaMemset(t.last_dev,  0, (size_t) t.n_experts * sizeof(int32_t));
            (void) cudaMemset(t.slot_prov_dev, 0, (size_t) slots * sizeof(uint8_t));
            t.policy = true;
        } else {
            (void) cudaGetLastError();
            if (t.slot_expert_dev != nullptr) { (void) cudaFree(t.slot_expert_dev); t.slot_expert_dev = nullptr; }
            if (t.count_dev       != nullptr) { (void) cudaFree(t.count_dev);       t.count_dev       = nullptr; }
            if (t.ghost_dev       != nullptr) { (void) cudaFree(t.ghost_dev);       t.ghost_dev       = nullptr; }
            if (t.last_dev        != nullptr) { (void) cudaFree(t.last_dev);        t.last_dev        = nullptr; }
            if (t.slot_prov_dev   != nullptr) { (void) cudaFree(t.slot_prov_dev);   t.slot_prov_dev   = nullptr; }
            GGML_LOG_WARN("%s: device-policy state alloc failed layer=%d role=%s; using the host promotion\n",
                          __func__, t.layer, t.role.c_str());
        }
    }
    // Prefill seed: bulk-admit the prompt's hottest experts before the first decode token (one-shot).
    // (The live, device-tally-driven seed runs later, from the first decode-band policy flush - see
    // `seed_prefill_lazy_locked`; this sizing-time call only serves the gate-off/eager path.)
    apply_prefill_seed_locked(t);
    // Account the count ACTUALLY allocated (`slots`/`t.slots`), not the requested `arena_slots`: on a
    // shrunk table the two differ and the old `arena_slots` accounting over-reported the arena.
    g_arena_bytes += (int64_t) slots * t.expert_bytes;
}

// Deferred, budget-adaptive, UNIFORM sizing.  Every table gets the same slot count
// `budget / total_expert_bytes`, so an expert resident for gate is resident for up and down too and
// the effective (all-roles) hit rate is set by the shared slot count, not by the smallest table.
// `total_expert_bytes` is known once the first decode pass has registered every table; that first
// pass runs uncached (the scheduler copies the full experts), the second sizes and starts filling.
void alloc_all_locked() {
    if (g_sized) {
        return;
    }
    g_sized = true;
    // Already disabled before the first graph (the slab could not give the cache a usable arena, or an
    // explicit `MOE_EXPERT_CACHE_MIB=0`): nothing to size.  See `moe_cache_disable_streaming` for why the
    // decision has to be made that early.
    if (!g_enabled) {
        return;
    }
    // The slab's reserve was a guess made before the weights existed (`GGML_CUDA_SLAB_RESERVE_MIB`); this is
    // the first moment the weights / KV / draft are all resident, so measure what they did NOT need and
    // hand it to the arena before sizing.  Without it that VRAM sits idle (~1.9 GiB/card measured) even
    // though the cache is capped by its region.
    ggml_cuda_slab_extend_all();
    // Fail-soft / --fit (§3.1 #5, issue #33): the arena is the LOWEST-priority VRAM consumer.  Size it
    // from what is actually free NOW - after --fit, the compute-graph reserve and the KV cache have all
    // taken theirs - minus the configured reserve for later growth (a bigger ubatch, more context, a
    // second pipeline).  This is why --fit does not need to learn about the arena: the arena yields to
    // whatever --fit already sized, it can never over-commit, and a user who asks for more than is free
    // simply gets less (warned).  A failed allocation still fails soft below.
    //
    // Phase 2: the budget is PER DEVICE.  Under `-sm layer` each device owns a disjoint set of layers,
    // so a second card must buy a second arena rather than split one budget in half (that split is what
    // made the second card buy no cache capacity).  A single-device run is unchanged: one device, one
    // budget.  Tables on a device still get a uniform slot count, so gate/up/down of a layer stay
    // aligned.
    int64_t dev_one_bytes [GGML_CUDA_MAX_DEVICES] = {0};
    int64_t dev_max_experts[GGML_CUDA_MAX_DEVICES] = {0};
    int     dev_n_tables [GGML_CUDA_MAX_DEVICES] = {0};
    int     max_dev = -1;
    for (const table_t & t : g_tables) {
        if (t.allocated || t.device < 0 || t.device >= GGML_CUDA_MAX_DEVICES) {
            continue;
        }
        dev_one_bytes [t.device] += (int64_t) t.expert_bytes;
        dev_n_tables  [t.device]++;
        if ((int64_t) t.n_experts > dev_max_experts[t.device]) {
            dev_max_experts[t.device] = t.n_experts;
        }
        if (t.device > max_dev) {
            max_dev = t.device;
        }
    }

    const size_t reserve = (size_t) g_reserve_mib * 1024 * 1024;
    int slots_d[GGML_CUDA_MAX_DEVICES];
    for (int d = 0; d < GGML_CUDA_MAX_DEVICES; d++) {
        slots_d[d] = 0;
    }
    for (int d = 0; d <= max_dev; d++) {
        if (dev_n_tables[d] == 0) {
            continue;
        }
        size_t budget = g_budget;
        {
            device_guard dg(d);
            // OPEN 2 slab: the slab's ARENA REGION owns the memory, so size against its capacity.  A
            // `cudaMemGetInfo` here reads only the small amount left outside the slab, which starves the
            // cache and drives the run to the host-expert path (measured: arena 2439 MiB, hit 0.47,
            // decode 21 t/s with 8 CPU cores busy and the GPUs 30 % idle).
            size_t free_b = 0, total_b = 0;
            if (ggml_cuda_slab_enabled()) {
                const size_t cap = ggml_cuda_slab_arena_total(d);
                // Each table's slab is rounded up to the arena unit, so hold back that rounding overhead
                // (up to one unit per table) or the sizing over-commits and the tail of the tables fails.
                const size_t rounding = (size_t) dev_n_tables[d] * ggml_cuda_slab_arena_unit(d);
                const size_t need = (size_t) g_extra_reserve + rounding;
                const size_t avail = cap > need ? cap - need : 0;
                if (budget > avail) {
                    budget = avail;
                    g_alloc_clamped++;
                }
            } else if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
                const size_t avail = free_b > reserve + (size_t) g_extra_reserve
                                   ? free_b - reserve - (size_t) g_extra_reserve : 0;
                if (budget > avail) {
                    if (!g_auto) {
                        GGML_LOG_WARN("%s: device %d: MOE_EXPERT_CACHE_MIB=%zu MiB exceeds the %.0f MiB free "
                                      "(free=%zu MiB - %zu MiB reserve); clamping to %.0f MiB so the compute "
                                      "reserve / KV cache are not starved\n",
                                      __func__, d, g_budget >> 20, (double) avail / (1024 * 1024),
                                      free_b >> 20, reserve >> 20, (double) avail / (1024 * 1024));
                    }
                    budget = avail;
                    g_alloc_clamped++;
                }
            }
        }
        int slots = dev_one_bytes[d] > 0 ? (int) (budget / (size_t) dev_one_bytes[d]) : 0;
        if (slots < 0) {
            slots = 0;
        }
        if ((int64_t) slots > dev_max_experts[d]) {
            slots = (int) dev_max_experts[d];
        }
        slots_d[d] = slots;
    }

    // Auto floor: below this total arena the fixed per-op cost outweighs the CPU bytes it saves, so
    // disable the cache entirely and take the CPU expert path.  This must be a full disable, NOT a
    // zero-slot arena: a registered table with no arena stands the cache-band MoE fusions down (the
    // per-op path), which is ~8x slower than the cache-less run's fused path (measured: MIB=64 ->
    // 4.3 t/s vs MIB=0 32.0).  Disabling here (before the first decode graph is built) leaves the run
    // byte-identical to `MOE_EXPERT_CACHE_MIB=0`.
    if (g_auto) {
        int64_t planned = 0;
        int64_t host_total = 0;
        for (int d = 0; d <= max_dev; d++) {
            planned += (int64_t) slots_d[d] * dev_one_bytes[d];
        }
        for (const table_t & t : g_tables) {
            if (!t.allocated) {
                host_total += (int64_t) t.n_experts * (int64_t) t.expert_bytes;
            }
        }
        const int64_t floor_bytes = std::max((int64_t) g_min_mib * 1024 * 1024,
                                             (g_min_res_pct > 0 && host_total > 0)
                                                 ? host_total * g_min_res_pct / 100 : (int64_t) 0);
        if (planned < floor_bytes) {
            // Do NOT disable here.  This runs AFTER the tables are registered and graphs have been planned
            // against them, and flipping the GLOBAL gate now plans the MTP draft and the target with
            // different kernels -- measured MTP acceptance 0.00874 where the streaming path gives 0.89506.
            // The floor is decided early, in `moe_cache_preflight` (before the context, so before any
            // graph).  If a config still reaches here, say so loudly and keep the small arena: correct,
            // just more host traffic.
            GGML_LOG_ERROR("%s: auto arena %.1f MiB is below the floor (%.1f MiB: MOE_EXPERT_CACHE_MIN_MIB=%lld, "
                          "_MIN_RES_PCT=%d of %.1f MiB host experts) -- too late to disable the cache safely "
                          "(the graphs already reference the tables), so continuing with the small arena. "
                          "Restart with MOE_EXPERT_CACHE_MIB=0 for the CPU expert path, or lower the floor.\n",
                          __func__, (double) planned / (1024 * 1024), (double) floor_bytes / (1024 * 1024),
                          (long long) g_min_mib, g_min_res_pct, (double) host_total / (1024 * 1024));
        }
        const double res = host_total > 0 ? (double) planned / (double) host_total : 1.0;
        GGML_LOG_WARN("%s: MoE expert cache (auto): arena %.1f MiB of %.1f MiB host experts (%.1f%% residency).  "
                      "MOE_EXPERT_CACHE_MIB=0 disables the cache; set a positive value to size it explicitly.\n",
                      __func__, (double) planned / (1024 * 1024), (double) host_total / (1024 * 1024), 100.0 * res);
        if (res < 0.20) {
            GGML_LOG_WARN("%s: auto arena residency is only %.1f%% -- a small arena is measured SLOWER "
                          "than the CPU expert path (MTP especially).  Set MOE_EXPERT_CACHE_MIB=0 to use "
                          "the CPU path, or free VRAM / raise MOE_EXPERT_CACHE_RESERVE_MIB=%lld to hold "
                          "more experts.\n",
                          __func__, 100.0 * res, (long long) g_reserve_mib);
        }
    }

    // Layer-uniform slot count.  gate/up/down of one layer still share a slot count (an expert
    // resident for gate must be resident for up and down, or the remap points a role at the wrong
    // expert).  Under `-sm tensor` a layer's tables live on *several* devices, so the count must also
    // be uniform ACROSS devices - otherwise device 0 and device 1 disagree on the residency and the
    // same remap id names different experts.  Take the minimum over the devices that own a layer;
    // for `-sm layer` every layer has one device, so this is exactly the old per-device count.
    int layer_slots[512];
    for (int i = 0; i < 512; i++) {
        layer_slots[i] = -1;
    }
    for (const table_t & t : g_tables) {
        if (t.allocated || t.layer < 0 || t.layer >= 512 || t.device < 0 || t.device >= GGML_CUDA_MAX_DEVICES) {
            continue;
        }
        const int s = slots_d[t.device];
        if (layer_slots[t.layer] < 0 || s < layer_slots[t.layer]) {
            layer_slots[t.layer] = s;
        }
    }
    // WIP item 1 (TODO #42): allocate each LAYER as a unit.  gate/up/down of a layer MUST expose the
    // same slot count (an expert resident for gate must be resident for up and down, or the shared
    // remap points a role at the wrong expert).  `alloc_table_locked` keeps the largest count that
    // actually allocates, so under fragmentation one table can fall short; instead of a per-table
    // shrink (which breaks the invariant and, before the `t.slots = got_slots` fix, over-read the
    // arena), free the layer, allocate the largest slices first, and if any table still falls short
    // take the achieved minimum and retry.  The count only decreases, so it converges.
    {
        std::map<int, std::vector<int>> by_layer;
        for (int i = 0; i < (int) g_tables.size(); ++i) {
            if (!g_tables[i].allocated) {
                by_layer[g_tables[i].layer].push_back(i);
            }
        }
        for (auto & kv : by_layer) {
            const int         layer = kv.first;
            std::vector<int> &ids   = kv.second;
            // largest expert slice first: a big table needs one contiguous block, and it gets the best
            // chance while the layer's own memory is still free
            std::stable_sort(ids.begin(), ids.end(), [](int a, int b) {
                return g_tables[a].expert_bytes > g_tables[b].expert_bytes;
            });
            int target = (layer >= 0 && layer < 512 && layer_slots[layer] >= 0) ? layer_slots[layer] : 0;
            const int requested = target;
            for (int attempt = 0; attempt < 8; ++attempt) {
                for (int i : ids) {
                    if (g_tables[i].allocated) {
                        free_table_buffers_locked(g_tables[i]);
                    }
                }
                for (int i : ids) {
                    alloc_table_locked(g_tables[i], target);
                }
                int mn = target;
                for (int i : ids) {
                    if (g_tables[i].slots < mn) {
                        mn = g_tables[i].slots;
                    }
                }
                if (mn == target) {
                    break;   // uniform
                }
                if (mn <= 0) {
                    if (target <= 1) {
                        target = 0;
                        break;
                    }
                    target = target / 2;   // a table could not hold even one slot: back off and retry
                    continue;
                }
                target = mn;
            }
            if (target <= 0) {
                // the layer cannot be made uniform above 0: run every table of this layer cache-less
                GGML_LOG_WARN("%s: layer %d cannot hold a uniform arena above 0 slots (requested %d); "
                              "running this layer uncached\n", __func__, layer, requested);
                for (int i : ids) {
                    if (g_tables[i].allocated && g_tables[i].slots > 0) {
                        free_table_buffers_locked(g_tables[i]);
                        alloc_table_locked(g_tables[i], 0);
                    }
                }
            } else if (target < requested) {
                GGML_LOG_WARN("%s: layer %d re-sized uniformly to %d slots (from %d) to keep gate/up/down "
                              "aligned\n", __func__, layer, target, requested);
            }
        }
    }
    // Safety net: by construction every layer is now slot-uniform.  If it is not (a bug in the re-size
    // above), never run with a non-uniform layer -- the shared remap would name different experts per
    // role.  Disable the cache (the CPU expert path) instead of risking silent corruption.
    {
        int got_min[512];
        for (int i = 0; i < 512; i++) {
            got_min[i] = -1;
        }
        for (const table_t & t : g_tables) {
            if (t.allocated && t.layer >= 0 && t.layer < 512) {
                if (got_min[t.layer] < 0 || t.slots < got_min[t.layer]) {
                    got_min[t.layer] = t.slots;
                }
            }
        }
        for (const table_t & t : g_tables) {
            if (!t.allocated || t.layer < 0 || t.layer >= 512 || got_min[t.layer] < 0) {
                continue;
            }
            if (t.slots != got_min[t.layer]) {
                GGML_LOG_WARN("%s: layer %d is not slot-uniform (%s has %d, layer min %d); disabling the cache "
                              "rather than risking a role mismatch\n",
                              __func__, t.layer, t.role.c_str(), t.slots, got_min[t.layer]);
                g_enabled = false;
                return;
            }
        }
    }
    // Arm the device-remap takeover only after one uniform eager decode pass (see `g_devmap_armed`).
    g_devmap_arm_expected = 0;
    for (const table_t & t : g_tables) {
        if (t.devmap) {
            g_devmap_arm_expected++;
        }
    }
    g_devmap_arm_seen = 0;
    g_devmap_eager_seen.assign(g_tables.size(), 0);
    g_devmap_armed = (g_devmap_arm_expected == 0);
    for (int d = 0; d <= max_dev; d++) {
        if (dev_n_tables[d] == 0) {
            continue;
        }
        GGML_LOG_INFO("%s: device %d: sized %d slots/table from %d tables / %.1f MiB per expert (budget %s MiB/device)\n",
                      __func__, d, slots_d[d], dev_n_tables[d],
                      (double) dev_one_bytes[d] / (1024 * 1024), budget_desc());
    }
    GGML_LOG_INFO("%s: per-device sizing: %d device(s), arena total %.1f MiB\n",
                  __func__, max_dev + 1, (double) g_arena_bytes / (1024 * 1024));
    g_uniform_slots = max_dev >= 0 ? slots_d[0] : -1;
    // OPEN 2 re-arm: record the per-layer slot count the sizing settled on (all a layer's tables are
    // uniform -- checked just above), so `moe_cache_rearm` can re-allocate stood-down tables to it.
    for (int i = 0; i < 512; i++) {
        g_rearm_slots[i] = -1;
    }
    for (const table_t & t : g_tables) {
        if (t.layer >= 0 && t.layer < 512 && t.slots > 0) {
            g_rearm_slots[t.layer] = t.slots;
        }
    }
}

} // namespace

// Device-side remap build (gentle-curve follow-up): remap[tok*n_used + j] = slot[ids[tok,j]], or
// `n_res + e` for a non-resident expert (the Phase 1b UVA cold region).  Tiny (n_used*n_tok <= 64);
// capture-safe so it can be baked into a decode CUDA graph.
static __global__ void moe_cache_build_remap_kernel(
        const int32_t * __restrict__ slot,
        const char * __restrict__ ids,
        int32_t * __restrict__ remap,
        int32_t * __restrict__ used,
        int32_t * __restrict__ used2,
        const int32_t * __restrict__ slot2,
        int32_t * __restrict__ remap2,
        int32_t * __restrict__ used3,
        int n_experts, int n_res, int n_used, int n_tok,
        size_t nb0, size_t nb1) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_used * n_tok) {
        return;
    }
    const int tok = i / n_used;
    const int j   = i - tok * n_used;
    const int32_t e = *(const int32_t *) (ids + (size_t) tok * nb1 + (size_t) j * nb0);
    int32_t s = -1;
    const int32_t ev = (e >= 0 && e < n_experts) ? e : -1;
    if (ev >= 0) {
        s = slot[ev];
    }
    if (used != nullptr) {
        used[i] = ev;
    }
    // used2 folds a second table's used-list write into the same launch (the fused gate+up path: the
    // gate lane's remap VALUE is unused - the fused kernel reads the up remap for both lanes - but the
    // deferred promotion needs the gate table's routing).  One kernel, both tables' used-lists.
    if (used2 != nullptr) {
        used2[i] = ev;
    }
    // slot2/remap2/used3 build a SIBLING table's remap from its OWN slot map (the layer's routed
    // `down`, whose redirect would otherwise be a separate launch).  Using slot2 - not the up map -
    // keeps it correct even if a role's map ever diverges; the down op reads this same routing `ids`.
    if (used3 != nullptr) {
        used3[i] = ev;
    }
    if (remap2 != nullptr) {
        int32_t s2 = -1;
        if (ev >= 0 && slot2 != nullptr) {
            s2 = slot2[ev];
        }
        remap2[i] = (s2 >= 0 && s2 < n_res) ? s2 : (n_res + (ev >= 0 ? ev : 0));
    }
    remap[i] = (s >= 0 && s < n_res) ? s : (n_res + (ev >= 0 ? ev : 0));
}

void moe_cache_launch_remap(const void * ids, size_t nb0, size_t nb1,
                            const moe_cache_devmap * dm, int64_t n_used, int64_t n_tok,
                            int32_t * remap_dev, void * stream, int32_t * used2,
                            const int32_t * slot2, int32_t * remap2, int32_t * used3) {
    if (ids == nullptr || dm == nullptr || dm->slot_dev == nullptr || remap_dev == nullptr) {
        return;
    }
    g_remap_launches++;
    const int n = (int) (n_used * n_tok);
    if (n <= 0) {
        return;
    }
    const int threads = 128;
    const int blocks  = (n + threads - 1) / threads;
    moe_cache_build_remap_kernel<<<blocks, threads, 0, (cudaStream_t) stream>>>(
        dm->slot_dev, (const char *) ids, remap_dev, dm->used_dev, used2, slot2, remap2, used3,
        dm->n_experts, dm->n_res, (int) n_used, (int) n_tok, nb0, nb1);
}

// ---------------------------------------------------------------------------------------------
// Prefill-routing seed (MOE_EXPERT_CACHE_PREFILL_SEED=1): device tally
// ---------------------------------------------------------------------------------------------

// One thread per (token, expert-slot) of the prefill routing.  `ids` is a strided device view
// (`n_used x n_tok`, strides `nb0`/`nb1`); the histogram lands in `count[n_experts]`.  Atomics are
// cheap here (top-8 x <=2048 tokens per op, one op per role per prefill ubatch) and the kernel is
// fire-and-forget on the compute stream, so the prefill keeps no host readback and no sync.
static __global__ void moe_cache_tally_kernel(
        const char * __restrict__ ids, size_t nb0, size_t nb1,
        int n_used, int n_tok, int n_experts, int32_t * __restrict__ count) {
    const int total = n_used * n_tok;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x) {
        const int tok = i / n_used;
        const int j   = i - tok * n_used;
        const int32_t e = *(const int32_t *) (ids + (size_t) tok * nb1 + (size_t) j * nb0);
        if (e >= 0 && e < n_experts) {
            atomicAdd(&count[e], 1);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Device-side admission policy kernel (MOE_EXPERT_CACHE_DEVPOLICY=1)
// ---------------------------------------------------------------------------------------------

// `moe_cache_policy_copy` and `moe_cache_seed_fill_kernel` live above (next to the seed placement) so the
// seed fill can use them; the policy kernel below reuses the same coalesced copy helper.

// One block per device-remap table.  Thread 0 replays the host `access_locked` LFRU logic sequentially
// over this token's routing (so the counters, decay cadence and victim tie-break are identical), staging
// each admitted `(slot, expert)` in shared memory; then the whole block copies the admitted experts from
// the pinned host alias into their arena slots.  The device slot map is updated in place, so the next
// token's remap kernel sees the new residency without any host readback.
//
// `shape[ti]` is this token's `n_used * n_tok` for table `ti`, or 0 when the table was not in the graph
// (its `used` buffer then holds stale routing and must not be replayed).
__global__ void moe_cache_policy_kernel(
        moe_cache_policy_desc * __restrict__ descs,
        const int32_t * __restrict__ shape,
        int n_tables, int admit, int touch, int period, int noevict, int do_fill, int prov_evict) {
    const int ti = blockIdx.x;
    if (ti >= n_tables) {
        return;
    }
    const int n = shape[ti];
    if (n <= 0) {
        return;
    }
    moe_cache_policy_desc & t = descs[ti];
    const int n_experts = t.n_experts;
    const int slots     = t.slots;

    __shared__ int32_t fill_slot[MOE_CACHE_POLICY_MAX_FILL];
    __shared__ int32_t fill_exp [MOE_CACHE_POLICY_MAX_FILL];
    __shared__ int     fill_n;

    if (threadIdx.x == 0) {
        fill_n = 0;
        if (slots > 0) {
            int64_t clock      = t.clock;
            int64_t last_decay = t.last_decay;
            int     occupied   = t.occupied;
            int64_t acc_hits   = t.acc_hits;
            int64_t acc_misses = t.acc_misses;
            int64_t acc_fills  = t.acc_fills;
            int64_t acc_evict  = t.acc_evict;
            for (int i = 0; i < n; i++) {
                const int32_t e = t.used[i];
                if (e < 0 || e >= n_experts) {
                    continue;
                }
                // host `access_locked`: clock++ then decay, per ACCESS
                clock++;
                if (period > 0 && clock - last_decay >= period) {
                    for (int x = 0; x < n_experts; x++) {
                        t.count[x] >>= 1;
                        t.ghost[x] >>= 1;
                    }
                    last_decay = clock - (clock % period);
                }
                if (t.slot[e] >= 0) {
                    t.count[e]++;
                    t.last[e] = (int32_t) clock;
                    if (t.slot_prov != nullptr) {
                        t.slot_prov[t.slot[e]] = 0;   // hit: no longer provisional
                    }
                    acc_hits++;
                    continue;
                }
                t.ghost[e]++;
                acc_misses++;
                if (noevict && occupied >= slots) {
                    continue;
                }
                if (do_fill && fill_n >= MOE_CACHE_POLICY_MAX_FILL) {
                    continue;   // fill list full: admitting would mark the expert resident with no fill staged
                }
                int slot = -1;
                if (occupied < slots) {
                    for (int s = 0; s < slots; s++) {
                        if (t.slot_expert[s] < 0) { slot = s; break; }
                    }
                }
                if (slot < 0) {
                    // evict the resident with the smallest decaying count; tie -> oldest use; never a
                    // victim used earlier in this same token (intra-token protect).
                    int32_t victim = -1;
                    int     vcount = 0;
                    int     vlast  = 0;
                    for (int s = 0; s < slots; s++) {
                        const int32_t ce = t.slot_expert[s];
                        if (ce < 0) {
                            continue;
                        }
                        bool prot = false;
                        for (int j = 0; j < n; j++) {
                            if (t.used[j] == ce) { prot = true; break; }
                        }
                        if (prot) {
                            continue;
                        }
                        if (victim < 0 || t.count[ce] < vcount ||
                            (t.count[ce] == vcount && t.last[ce] < vlast)) {
                            victim = ce;
                            slot   = s;
                            vcount = t.count[ce];
                            vlast  = t.last[ce];
                        }
                    }
                    bool reject = false;
                    if (slot >= 0) {
                        const bool prov = prov_evict && t.slot_prov != nullptr && t.slot_prov[slot] != 0;
                        if (!prov) {
                            if (admit == 1) {
                                reject = t.ghost[e] <= vcount;
                            } else if (admit == 2) {
                                reject = t.ghost[e] < touch;
                            }
                        }
                    }
                    if (reject) {
                        continue;
                    }
                    if (slot >= 0) {
                        t.slot[victim]       = -1;
                        t.slot_expert[slot]  = -1;
                        occupied--;
                        acc_evict++;
                    }
                }
                if (slot < 0) {
                    continue;
                }
                t.slot_expert[slot] = e;
                t.slot[e]           = slot;
                t.count[e]++;
                t.ghost[e]          = 0;
                t.last[e]           = (int32_t) clock;
                if (t.slot_prov != nullptr) {
                    t.slot_prov[slot] = 0;   // a real admission is not provisional
                }
                occupied++;
                acc_fills++;
                if (fill_n < MOE_CACHE_POLICY_MAX_FILL) {
                    fill_slot[fill_n] = slot;
                    fill_exp [fill_n] = e;
                    fill_n++;
                }
            }
            t.clock      = clock;
            t.last_decay = last_decay;
            t.occupied   = occupied;
            t.acc_hits   = acc_hits;
            t.acc_misses = acc_misses;
            t.acc_fills  = acc_fills;
            t.acc_evict  = acc_evict;
        }
    }
    __syncthreads();

    if (!do_fill || t.host_dev == nullptr || t.arena == nullptr) {
        return;
    }
    const int tid = threadIdx.x;
    const int nt  = blockDim.x;
    const int fn  = fill_n;
    for (int f = 0; f < fn; f++) {
        const int    slot = fill_slot[f];
        const int    e    = fill_exp[f];
        char *       dst  = (char *) t.arena + (size_t) slot * (size_t) t.expert_bytes;
        const char * src  = (const char *) t.host_dev + (size_t) e * (size_t) t.host_bytes + (size_t) t.src_off;
        if (t.split_axis != 0 || t.host_pitch == 0) {
            moe_cache_policy_copy(dst, src, (size_t) t.expert_bytes, tid, nt);
        } else {
            const int64_t rows = t.host_bytes > 0 ? (t.host_bytes / t.host_pitch) : 0;
            const size_t  row  = rows > 0 ? ((size_t) t.expert_bytes / (size_t) rows) : 0;
            for (int64_t r = 0; r < rows; r++) {
                moe_cache_policy_copy(dst + (size_t) r * row, src + (size_t) r * t.host_pitch, row, tid, nt);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// public API
// ---------------------------------------------------------------------------------------------

int moe_cache_max_tok_dev(int device) {
    static int band[GGML_CUDA_MAX_DEVICES] = {};   // 0 = not computed yet (idempotent, so a racy init is benign)
    if (band[device] == 0) {
        const char * e    = getenv("GGML_MOE_CACHE_MAX_TOK");
        const int    want = e != nullptr ? atoi(e) : MMVQ_MOE_MAX_BATCH_SIZE;
        const int    cc   = ggml_cuda_info().devices[device].cc;
        int kband = MMVQ_MOE_MAX_BATCH_SIZE;
        for (int t = 0; t < GGML_TYPE_COUNT; t++) {
            if (ggml_is_quantized((ggml_type) t)) {
                kband = std::min(kband, get_mmvq_mmid_max_batch((ggml_type) t, cc));
            }
        }
        band[device] = std::max(1, std::min(want, std::max(MMVQ_MAX_BATCH_SIZE, kband)));
    }
    return band[device];
}

int moe_cache_max_tok() {
    return moe_cache_max_tok_dev(ggml_cuda_get_device());
}

bool moe_cache_enabled() {
    return g_enabled;
}

bool moe_cache_cold_active() {
    return g_enabled && g_cold_uva;
}

// True once the arena sizing has run (or an explicit slot count was given), i.e. the per-op read source
// will no longer change.  CUDA graph capture must wait for this.
bool moe_cache_ready() {
    if (!g_enabled) {
        return false;
    }
    // No registered routed-expert table (a fully-resident model: `-ncmoe 0`, or a `-ncmoe` that did not
    // offload): the cache can never take an input over, so there is nothing to wait for and CUDA graph
    // capture must not be held off.  Tables register during the scheduler split of the first compute,
    // before any capture is attempted, so a host-expert model still holds capture until it is sized.
    if (!moe_cache_has_tables()) {
        return true;
    }
    // CUDA graphs must not be captured until the read source is constant.  Device-remap tables need one
    // more condition: the takeover decision flips when `g_devmap_armed` turns on (one uniform eager pass
    // after sizing), and a graph captured before that would bake the eager remap pointer.  Hold capture
    // off across the transition; the eager pass is short and runs uncaptured like the sizing token.
    if (g_devmap && !g_devmap_armed) {
        return false;
    }
    return g_slots_hint > 0 || g_sized;
}

bool moe_cache_preflight(int device, size_t host_expert_bytes, size_t aux_reserve_bytes) {
    if (!g_enabled) {
        return false;
    }
    if (!g_auto) {
        return true;   // an explicit MOE_EXPERT_CACHE_MIB is used verbatim; no early floor
    }
    if (g_preflight_done) {
        return g_enabled;   // latch: the first call (target context, before the draft) decides
    }
    g_preflight_done = true;
    size_t free_b = 0, total_b = 0;
    device_guard dg(device);
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        return true;   // fail-soft: keep the cache on
    }
    const size_t reserve = (size_t) g_reserve_mib * 1024 * 1024 + aux_reserve_bytes;
    const size_t avail   = free_b > reserve ? free_b - reserve : 0;
    int64_t floor_b = (int64_t) g_min_mib * 1024 * 1024;
    if (g_min_res_pct > 0 && host_expert_bytes > 0) {
        const int64_t res_b = (int64_t) host_expert_bytes * (int64_t) g_min_res_pct / 100;
        if (res_b > floor_b) {
            floor_b = res_b;
        }
    }
    if ((int64_t) avail < floor_b) {
        GGML_LOG_WARN("%s: auto arena on device %d would be %.1f MiB (free %.1f - reserve %.1f - aux %.1f) "
                      "< floor %.1f MiB (host experts %.1f MiB); disabling the MoE expert cache for this run "
                      "-- a starved arena is slower than the CPU expert path.  Set MOE_EXPERT_CACHE_MIB=0 to "
                      "silence, or free VRAM / lower MOE_EXPERT_CACHE_RESERVE_MIB=%lld.\n",
                      __func__, device, (double) avail / (1024 * 1024), (double) free_b / (1024 * 1024),
                      (double) g_reserve_mib, (double) aux_reserve_bytes / (1024 * 1024),
                      (double) floor_b / (1024 * 1024), (double) host_expert_bytes / (1024 * 1024),
                      (long long) g_reserve_mib);
        g_enabled = false;
        return false;
    }
    GGML_LOG_INFO("%s: auto cache preflight: device %d projected arena %.1f MiB (free %.1f - reserve %.1f - "
                  "aux %.1f) >= floor %.1f MiB (host experts %.1f MiB); cache stays on\n",
                  __func__, device, (double) avail / (1024 * 1024), (double) free_b / (1024 * 1024),
                  (double) g_reserve_mib, (double) aux_reserve_bytes / (1024 * 1024),
                  (double) floor_b / (1024 * 1024), (double) host_expert_bytes / (1024 * 1024));
    return true;
}

void moe_cache_set_extra_reserve(int device, size_t bytes) {
    // WIP r42 (TODO #42): hold this many bytes of the free VRAM for the post-prefill compute layout, so
    // the arena does not take the space a later compute growth (or the grow-in-place realloc) needs.
    // Called at the prefill -> decode transition, before the arena is sized.
    // device-global; the `device` argument is accepted so the per-device call site is unchanged
    (void) device;
    g_extra_reserve = (int64_t) bytes;
    GGML_LOG_INFO("%s: extra arena reserve %.1f MiB (post-prefill compute), all devices\n",
                  __func__, (double) bytes / (1024 * 1024));
}

bool moe_cache_get_stats(int64_t * hits, int64_t * misses, int64_t * arena_bytes) {    // Per-turn accounting for the server: aggregate the same counters `moe_cache_report` prints, so a
    // caller can snapshot before/after a turn and log the delta.  The counters are process-global (the
    // arena is one per-device cache shared by every slot), so with concurrent slots a delta is the
    // whole-process activity during the turn, not strictly one session's.
    if (!g_enabled) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    // the device-side admission policy owns the counters; pull them back first (as moe_cache_report does)
    for (int d = 0; d < (int) g_policy_dev.size(); d++) {
        policy_dev_t & pd = g_policy_dev[(size_t) d];
        if (pd.desc == nullptr || pd.n <= 0) {
            continue;
        }
        device_guard dg(d);
        std::vector<moe_cache_policy_desc> host((size_t) pd.n);
        if (cudaMemcpy(host.data(), pd.desc, (size_t) pd.n * sizeof(moe_cache_policy_desc),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            (void) cudaGetLastError();
            continue;
        }
        for (int i = 0; i < pd.n; i++) {
            table_t & t = g_tables[pd.ids[(size_t) i]];
            t.hits      = host[(size_t) i].acc_hits;
            t.misses    = host[(size_t) i].acc_misses;
            t.fills     = host[(size_t) i].acc_fills;
            t.evictions = host[(size_t) i].acc_evict;
        }
    }
    int64_t h = 0, m = 0, a = 0;
    for (const table_t & t : g_tables) {
        h += t.hits;
        m += t.misses;
        if (t.slots > 0) {
            a += (int64_t) t.slots * (int64_t) t.expert_bytes;
        }
    }
    if (hits != nullptr)        { *hits = h; }
    if (misses != nullptr)      { *misses = m; }
    if (arena_bytes != nullptr) { *arena_bytes = a; }
    return true;
}

// WIP r42 (TODO #42): stand a table down mid-run (its arena is being freed for a compute allocation).
// Mirrors the device-migration teardown -- free the arena AND the remap, reset the remap state and the
// residency maps, drop the scheduler alias so no fused op can resolve this table to a freed pointer.
// The table then behaves as a registered-but-uncached table (slots == 0); its consumers all check
// `slots <= 0` and fall back to the host path.  Caller holds g_mutex.
// WIP r42 (TODO #42): the MoE kernels receive the ARENA ADDRESS AS A KERNEL ARGUMENT -- the redirect
// (`moe_cache_redirect_fused`) copies the tensor descriptor and repoints `data` at the arena, so the
// pointer ends up in the launch parameters, not in a tensor the host can re-read.  The compute is
// asynchronous (the server launches a graph and returns; the alloc for a later/wider ubatch then runs
// while an earlier ubatch's kernels are still queued or running).  HIP does NOT wait for in-flight work
// when memory is freed, so releasing an arena that a running kernel still reads is a use-after-free on
// the device: ROCr reports `Memory access fault ... Page not present` and aborts the whole process.
// Measured: `mul_mat_vec_q_moe<(ggml_type)20,2,false>` faulting on a just-freed arena during a
// concurrent prefill.  Synchronize every device that owns a table before releasing anything.  This is a
// rare, exceptional path (an allocation already failed), so the sync cost is irrelevant.
static void moe_cache_sync_devices_locked() {
    std::vector<int> devs;
    for (const table_t & t : g_tables) {
        if (t.device < 0) {
            continue;
        }
        bool seen = false;
        for (int d : devs) {
            if (d == t.device) { seen = true; break; }
        }
        if (!seen) {
            devs.push_back(t.device);
        }
    }
    for (int d : devs) {
        device_guard dg(d);
        (void) cudaDeviceSynchronize();
    }
    (void) cudaGetLastError();   // fail soft: a bad sync must not abort a run
}

static void stand_down_table_locked(table_t & t, int idx) {
    const int    old_device = t.device;
    const int    old_slots  = t.slots;
    void * const old_arena  = t.arena;
    void * const old_remap  = t.remap_dev;

    // OPEN 2: a VMM arena keeps its reserved VA across a stand-down (only physical is unmapped), so a
    // re-arm maps at the same address; a cudaMalloc arena is freed and its pointer cleared.
    t.arena        = t.arena_reserved != 0 ? t.arena : nullptr;
    t.remap_dev    = nullptr;
    t.remap_cap    = 0;
    t.remap_n_used = 0;
    t.remap_n_tok  = 0;
    t.slots        = 0;
    t.slot_dirty   = true;
    t.slot_expert.clear();
    t.expert_slot.clear();
    t.slot_dev_host.clear();
    g_arena_bytes -= (int64_t) old_slots * (int64_t) t.expert_bytes;
    if (g_arena_bytes < 0) {
        g_arena_bytes = 0;
    }
    if (old_device >= 0) {
        moe_cache_sync_devices_locked();   // no kernel that reads this arena may still be in flight
        device_guard dg(old_device);
        if (old_arena != nullptr) {
            // The slab OWNS the arena's VA, so this only returns the range to the slab's free list (the
            // work pool can then take it, or the arena can hand it to another table); the plain path's
            // arena is a cudaMalloc the table owns outright.
            free_arena_backing(old_device, old_arena, t.arena_reserved, old_slots, t);
        }
        if (old_remap != nullptr) {
            (void) cudaFree(old_remap);
        }
        // The device-side expert->slot map still advertises the OLD residency, but the arena's bytes are
        // gone (unmapped, and zeroed when a re-arm maps it again).  Clear it, or the device-remap takeover
        // serves slots that no longer hold their expert -- measured as the repeated-`/` corruption once a
        // re-arm had re-mapped the arena.  The host `expert_slot`/`slot_expert` maps are cleared above.
        if (t.slot_dev != nullptr && t.n_experts > 0) {
            (void) cudaMemset(t.slot_dev, 0xff, (size_t) t.n_experts * sizeof(int32_t));
        }
        if (t.used_dev != nullptr && t.used_cap > 0) {
            (void) cudaMemset(t.used_dev, 0xff, (size_t) t.used_cap * sizeof(int32_t));
        }
        t.remap_fresh = false;
        (void) cudaGetLastError();   // fail soft: a bad free must not abort a run
    }
    for (auto it = g_alias_to_id.begin(); it != g_alias_to_id.end(); ) {
        if (it->second == idx) {
            it = g_alias_to_id.erase(it);
        } else {
            ++it;
        }
    }
}

bool moe_cache_shrink_step() {
    // WIP r42 (TODO #42): free ONE table arena -- the largest -- so the caller can retry the failed
    // allocation after each release and stop as soon as it fits.  The shortfall is usually a couple of
    // hundred MiB (a grow-in-place realloc only needs a block slightly larger than the one it freed), so
    // one or two tables is normally enough -- far better than releasing the whole cache.  The freed
    // table falls back to the host expert path; every other table keeps its arena and hit rate.
    if (!g_enabled) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_enabled) {
        return false;
    }
    int best = -1;
    int64_t best_b = 0;
    for (int i = 0; i < (int) g_tables.size(); ++i) {
        const table_t & t = g_tables[i];
        if (t.arena != nullptr && t.slots > 0) {
            const int64_t b = (int64_t) t.slots * (int64_t) t.expert_bytes;
            if (b > best_b) {
                best_b = b;
                best   = i;
            }
        }
    }
    if (best < 0) {
        g_enabled = false;   // nothing left -- behave like the full release
        return false;
    }
    table_t & t = g_tables[best];
    const int64_t freed_b = (int64_t) t.slots * (int64_t) t.expert_bytes;
    stand_down_table_locked(t, best);
    GGML_LOG_WARN("%s: stood down the largest table (layer=%d role=%s, %.1f MiB) for a compute allocation\n",
                  __func__, t.layer, t.role.c_str(), (double) freed_b / (1024.0 * 1024.0));
    return true;
}

bool moe_cache_shrink_arena(size_t need_bytes) {
    // WIP r42 (TODO #42): free the *largest* table arenas until `need_bytes` plus a margin is released,
    // so a compute allocation that is only a little short of a contiguous block can succeed without
    // giving up the whole cache.  The freed tables fall back to the host expert path; every other table
    // keeps its arena and its hit rate.  Returns true when anything was freed.
    if (!g_enabled) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_enabled) {
        return false;
    }
    const size_t target = need_bytes + (256u << 20);
    std::vector<int> order;
    for (int i = 0; i < (int) g_tables.size(); ++i) {
        if (g_tables[i].arena != nullptr && g_tables[i].slots > 0) {
            order.push_back(i);
        }
    }
    const size_t n_with_arena = order.size();
    std::sort(order.begin(), order.end(), [](int a, int b) {
        return (int64_t) g_tables[a].slots * g_tables[a].expert_bytes >
               (int64_t) g_tables[b].slots * g_tables[b].expert_bytes;
    });
    size_t freed = 0;
    size_t n_freed = 0;
    for (int i : order) {
        if (freed >= target) {
            break;
        }
        table_t & t = g_tables[i];
        device_guard dg(t.device);
        // A slab-backed arena must go back through the slab's free list, never `cudaFree`: the slab is a
        // single VMM mapping, and a `cudaFree` of a range inside it can unmap/corrupt the whole slab.
        // Mirrors `moe_cache_release_arena`.
        if (t.arena != nullptr) {
            freed += (size_t) t.slots * t.expert_bytes;
            free_arena_backing(t.device, t.arena, t.arena_reserved, t.slots, t);
            t.arena_reserved = 0;
        }
        t.arena = nullptr;
        t.slots = 0;
        t.slot_expert.clear();
        t.slot_prov.clear();
        t.expert_slot.clear();
        t.slot_dirty = true;
        n_freed++;
    }
    if (n_freed == 0) {
        return false;
    }
    GGML_LOG_WARN("%s: freed %zu of %zu table arena(s) (%.1f MiB) to make room for a compute allocation\n",
                  __func__, n_freed, n_with_arena, (double) freed / (1024.0 * 1024.0));
    if (n_freed == n_with_arena) {
        g_enabled = false;   // nothing left -- behave like the full release
    }
    return true;
}

bool moe_cache_release_arena() {
    // WIP r42 (TODO #42) fail-soft guard: the arena is the LOWEST-priority VRAM consumer.  The compute
    // buffer is grow-only, so a later, wider graph can need VRAM the arena owns; rather than abort, the
    // caller (the CUDA alloc path) frees the whole arena once and retries.  The cache is disabled for
    // the rest of the run (the CPU expert path serves the experts); correctness is unaffected -- only
    // the decode speed.  Returns true when an arena was actually released.
    if (!g_enabled) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_enabled) {
        return false;   // another thread released it first
    }
    size_t freed_b = 0;
    // Same hazard as the partial stand-down: the fused MoE kernels hold the arena address in their
    // launch parameters, so nothing may be freed while one is still in flight.
    moe_cache_sync_devices_locked();
    for (table_t & t : g_tables) {
        if (t.arena != nullptr && t.device >= 0) {
            device_guard dg(t.device);
            if (t.arena_reserved != 0) {
                // OPEN 2: hand the backing back (slab chunk, or VMM physical + reserved VA).  This path
                // disables the cache for the run.
                if (t.slots > 0) {
                    freed_b += (size_t) t.slots * t.expert_bytes;
                }
                free_arena_backing(t.device, t.arena, t.arena_reserved, t.slots, t);
                t.arena_reserved = 0;
            } else if (cudaFree(t.arena) == cudaSuccess) {
                freed_b += (size_t) t.slots * t.expert_bytes;
            } else {
                (void) cudaGetLastError();
            }
            t.arena = nullptr;
        }
        t.slots = 0;
        t.slot_expert.clear();
        t.expert_slot.clear();
        t.slot_dirty = true;
    }
    g_enabled = false;
    GGML_LOG_WARN("%s: released the MoE expert cache arena (%.1f MiB) to satisfy a compute allocation; "
                  "the expert cache is disabled for the rest of this run (experts come from the host)\n",
                  __func__, (double) freed_b / (1024.0 * 1024.0));
    return true;
}

// Disable the cache outright and STREAM the experts from the host (the stock `-ncmoe` path).
//
// MUST be called BEFORE the first graph that can consult the cache -- i.e. from the movable-boundary
// slab's own decision at the first compute-buffer allocation, not from the sizing.  `MOE_EXPERT_CACHE_MIB=0`
// is inert because `moe_cache_init` returns before registering anything; a LATE `g_enabled = false`
// instead leaves registered tables and the graphs already built against them, and the draft and target
// then disagree: measured MTP acceptance 0.00342 (2/585) vs 0.89506 for the same config disabled from the
// start, both coherent-looking.  `MOE_EXPERT_CACHE_MIN_MIB`'s auto floor still flips `g_enabled` late and
// is therefore subject to the same failure.
void moe_cache_disable_streaming(const char * why) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_enabled) {
        return;
    }
    g_enabled = false;
    GGML_LOG_ERROR("%s: disabling the MoE expert cache and STREAMING the experts from the host "
                   "(stock -ncmoe behaviour): %s\n", __func__, why);
}

// OPEN 2 (TODO #42): re-arm the arena after a compute-buffer DROP returned the VRAM.
//
// The slab's boundary move evicts the arena tables that live in the chunks the work pool takes, and a
// per-table stand-down (the plain path's fail-soft yield) leaves others down.  The survivors keep their
// bytes untouched -- a table that lost its arena is bypassed, not corrupt -- so when the wide compute
// buffer is dropped at the prefill -> decode transition the VRAM comes back and re-allocating only the
// stood-down tables restores the cache WITH its residents, not cold.  There is deliberately no second
// arena and no migration: the per-table address and slot layout are unchanged (the slab's arena region
// is a stable VA), the survivors are still valid, and only the missing tables are rebuilt.  `g_sized` is
// a one-shot, so this is the only path that re-grows the arena.
//
// Returns true when at least one table was re-armed.  The caller must run it at a point where no graph
// is in flight and the compute buffer has already been dropped (the drop site re-reserves the narrow
// layout first), so the CPU-side take-over decisions of the next graph all see the same residency.
bool moe_cache_rearm() {
    if (!g_enabled) {
        return false;   // released wholesale, or never sized: nothing to re-arm against
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_sized || g_tables.empty()) {
        return false;
    }

    // Which layers have stood-down tables?  A layer is re-armed as a unit: gate/up/down must stay
    // slot-uniform, or the shared remap points a role at the wrong expert.  A stood-down table is one
    // that was ALLOCATED but currently has no slots -- under VMM it still holds its reserved VA (only the
    // physical was unmapped), under cudaMalloc its pointer is null.
    std::map<int, std::vector<int>> by_layer;
    for (int i = 0; i < (int) g_tables.size(); ++i) {
        const table_t & t = g_tables[i];
        if (t.device >= 0 && t.allocated && t.slots == 0) {
            by_layer[t.layer].push_back(i);
        }
    }
    if (by_layer.empty()) {
        return false;
    }

    int n_rearmed = 0;
    for (auto & kv : by_layer) {
        const int         layer = kv.first;
        std::vector<int> &ids   = kv.second;
        int target = (layer >= 0 && layer < 512) ? g_rearm_slots[layer] : -1;
        if (target <= 0) {
            continue;   // the layer was not cacheable at sizing time
        }
        std::stable_sort(ids.begin(), ids.end(), [](int a, int b) {
            return g_tables[a].expert_bytes > g_tables[b].expert_bytes;
        });
        for (int attempt = 0; attempt < 8; ++attempt) {
            for (int i : ids) {
                if (g_tables[i].slots > 0) {
                    free_table_buffers_locked(g_tables[i]);   // undo a partial attempt (only mapped tables)
                }
            }
            for (int i : ids) {
                alloc_table_locked(g_tables[i], target);
            }
            int mn = target;
            for (int i : ids) {
                if (g_tables[i].slots < mn) {
                    mn = g_tables[i].slots;
                }
            }
            if (mn == target) {
                break;   // uniform
            }
            if (mn <= 0) {
                if (target <= 1) {
                    target = 0;
                    break;
                }
                target = target / 2;
                continue;
            }
            target = mn;
        }
        if (target > 0) {
            n_rearmed += (int) ids.size();
        }
    }

    if (n_rearmed > 0) {
        GGML_LOG_WARN("%s: re-armed %d stood-down expert-cache tables after a compute-buffer drop "
                      "(arena now %.1f MiB)\n", __func__, n_rearmed, (double) g_arena_bytes / (1024 * 1024));
    }
    return n_rearmed > 0;
}


// OPEN 2 (TODO #42): the movable-boundary slab's eviction hook.
//
// When the work pool needs more chunks the slab reassigns its LOWEST arena chunks to it.  A table whose
// storage lies in that range must be dropped -- those bytes are the work pool's now -- and each one's slab
// is handed back through `ggml_cuda_slab_arena_free`.  Every table outside the range keeps its bytes, so
// this is the design's ONLY eviction: no per-slot prune, no unmap of a table that survives.
//
// Called with `g_slab_mutex` NOT held (the slab releases it around this call) but WITH the cache lock, so
// the slab's own lock is taken here -- ordering is always cache -> slab.
size_t moe_cache_evict_slab_range(int device, void * lo, void * hi) {
    if (!g_enabled) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_sized || g_tables.empty() || lo == nullptr || hi == nullptr || lo >= hi) {
        return 0;
    }
    const size_t unit    = ggml_cuda_slab_arena_unit(device);
    const uintptr_t a_lo = (uintptr_t) lo;
    const uintptr_t a_hi = (uintptr_t) hi;

    // No kernel that holds an arena address in its launch parameters may still be in flight.
    moe_cache_sync_devices_locked();

    size_t freed = 0;
    for (table_t & t : g_tables) {
        if (t.arena == nullptr || t.device != device || t.arena_reserved == 0) {
            continue;
        }
        const size_t bytes = unit > 0 ? unit * ((t.arena_reserved + unit - 1) / unit) : t.arena_reserved;
        const uintptr_t a  = (uintptr_t) t.arena;
        const uintptr_t e  = a + bytes;
        if (a >= a_hi || e <= a_lo) {
            continue;   // outside the taken chunks
        }
        // Give the whole allocation back to the slab FIRST, then clear the table.  The small per-table
        // device buffers (remap_dev/slot_dev/...) are separate cudaMalloc allocations and are kept, so a
        // later re-size/re-arm can reuse them.
        ggml_cuda_slab_arena_free(device, t.arena, bytes);
        g_arena_bytes -= (int64_t) t.slots * (int64_t) t.expert_bytes;
        if (g_arena_bytes < 0) {
            g_arena_bytes = 0;
        }
        freed += bytes;

        t.arena          = nullptr;
        t.arena_reserved = 0;
        t.slots          = 0;
        t.identity       = false;
        t.remap_n_used   = 0;
        t.remap_fresh    = false;
        t.slot_dirty     = true;
        t.slot_expert.clear();
        t.expert_slot.clear();
        t.slot_dev_host.clear();
        t.allocated      = true;   // it was decided; a re-arm can rebuild it
    }
    if (freed > 0) {
        GGML_LOG_WARN("%s: evicted %.1f MiB of arena tables from the taken slab chunks (arena now %.1f MiB)\n",
                      __func__, (double) freed / (1024 * 1024), (double) g_arena_bytes / (1024 * 1024));
    }
    return freed;
}

// ALL tables must be servable.  The gate+up fused kernel indexes the gate lane with the up
// table's remap, and the fusion guard is global (one answer for the whole graph), so a single
// failed arena must stand the cache-band fusions down for every table: the per-op path is the
// correct fallback when a role's op would otherwise read the scheduler's copy while a sibling
// still redirects to an arena.  A partially-failed cache therefore falls back wholesale.
static bool moe_cache_has_arena_locked() {
    if (!g_enabled || g_tables.empty()) {
        return false;
    }
    for (const table_t & t : g_tables) {
        if (t.arena == nullptr || t.slots <= 0) {
            return false;
        }
    }
    return true;
}

bool moe_cache_has_arena() {
    if (!g_enabled) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    return moe_cache_has_arena_locked();
}

// ---------------------------------------------------------------------------------------------------
// OPEN 2 debug validator (`MOE_EXPERT_CACHE_VALIDATE`).  A no-op unless the env var is set, so a normal
// run pays nothing.  Level 1 checks the structural invariants every consumer relies on; level 2 also
// D2H-reads each resident slot's head and flags a non-finite f16 scale, which is exactly the
// precondition of the repeated-`/` MMQ over-read corruption described in `alloc_table_locked`.  The
// point is to catch a cache left inconsistent by a compute-buffer growth / arena stand-down BEFORE the
// next graph reads it, instead of inferring it from the generated text.
static int g_validate_level = -1;

static int moe_cache_validate_level() {
    if (g_validate_level < 0) {
        const char * env = getenv("MOE_EXPERT_CACHE_VALIDATE");
        g_validate_level = env != NULL ? atoi(env) : 0;
    }
    return g_validate_level;
}

void moe_cache_validate(const char * where) {
    const int level = moe_cache_validate_level();
    if (level <= 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mutex);

    size_t  n_res = 0, n_down = 0, n_bad = 0;
    int64_t bytes = 0;
    for (const table_t & t : g_tables) {
        const bool resident = t.slots > 0;
        if (resident) {
            n_res++;
            bytes += (int64_t) t.slots * (int64_t) t.expert_bytes;
            if (t.arena == nullptr && t.arena_reserved == 0) {
                n_bad++;
                GGML_LOG_ERROR("[validate %s] RESIDENT-NO-ARENA layer=%d role=%s slots=%d\n",
                               where, t.layer, t.role.c_str(), t.slots);
            }
        } else {
            // 0 slots is a LEGAL state: a VMM table keeps its reserved VA, a cudaMalloc table nulls the
            // pointer.  What matters is that no consumer serves it.
            n_down++;
        }
        if (resident && t.identity && t.slots < t.n_experts) {
            GGML_LOG_ERROR("[validate %s] IDENTITY-SHORT layer=%d slots=%d n_experts=%d\n",
                           where, t.layer, t.slots, t.n_experts);
        }
        if (resident && t.devmap && t.slot_dev == nullptr) {
            GGML_LOG_ERROR("[validate %s] DEVMAP-NULL-SLOTDEV layer=%d\n", where, t.layer);
        }
        if (t.remap_dev == nullptr && t.remap_cap != 0) {
            GGML_LOG_ERROR("[validate %s] DANGLING-REMAP layer=%d remap_dev=null remap_cap=%lld\n",
                           where, t.layer, (long long) t.remap_cap);
        }
    }
    if (bytes != g_arena_bytes) {
        GGML_LOG_ERROR("[validate %s] ARENA-BYTES g_arena_bytes=%lld sum(slots*expert_bytes)=%lld\n",
                       where, (long long) g_arena_bytes, (long long) bytes);
    }
    GGML_LOG_WARN("[validate %s] tables=%zu resident=%zu down=%zu inconsistent=%zu bytes=%lld g_arena_bytes=%lld enabled=%d has_arena=%d sized=%d\n",
                  where, g_tables.size(), n_res, n_down, n_bad, (long long) bytes, (long long) g_arena_bytes,
                  (int) g_enabled, (int) moe_cache_has_arena_locked(), (int) g_sized);

    if (level < 2) {
        return;
    }
    // Deep check: the head of a resident slot is the expert block's f16 scale.  Inf/NaN there is the
    // repeated-`/` poison.  A slot the gather never filled must still be zero (the head guard).
    for (table_t & t : g_tables) {
        if (t.arena == nullptr || t.slots <= 0 || t.device < 0) {
            continue;
        }
        device_guard dg(t.device);
        int n_nonfinite = 0, n_zero = 0;
        for (int i = 0; i < t.slots; i++) {
            uint16_t h = 0;
            const void * p = (const char *) t.arena + (size_t) i * t.expert_bytes;
            if (cudaMemcpy(&h, p, sizeof(h), cudaMemcpyDeviceToHost) != cudaSuccess) {
                (void) cudaGetLastError();
                break;
            }
            if ((h & 0x7c00u) == 0x7c00u) { n_nonfinite++; }
            if (h == 0) { n_zero++; }
        }
        if (n_nonfinite > 0) {
            GGML_LOG_ERROR("[validate %s] NONFINITE-HEAD layer=%d role=%s slots=%d nonfinite=%d zero=%d\n",
                           where, t.layer, t.role.c_str(), t.slots, n_nonfinite, n_zero);
        }
    }
}

// True only when the cache has at least one registered routed expert table.  An empty cache (the model
// is fully device-resident -- `-ncmoe 0`, or a `-ncmoe` that did not offload, e.g. on unified memory --
// can never take an input over, so the cache-band fusion guard must not stand anything down: a cache-on
// run has to stay byte-identical to the cache-less one.  Returns false before sizing.
bool moe_cache_has_tables() {
    if (!g_enabled) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    return !g_tables.empty();
}

// Attach the loader-registered on-disk source (if any) to `t`.  Caller holds `g_mutex`.
static void attach_host_src_locked(table_t & t, const void * host) {
    if (host == nullptr || !t.src_path.empty()) {
        return;
    }
    const auto it = g_host_src.find(host);
    if (it == g_host_src.end()) {
        return;
    }
    t.src_path  = it->second.path;
    t.src_offs  = it->second.offs;
    t.src_known = true;
}

// wip/host-expert-dio-cache Phase 1 (plumbing, inert).  See the header for the contract.
void moe_cache_set_host_source(const void * tensor_data, const char * path, size_t offs,
                               int n_experts, size_t host_bytes, size_t total_bytes) {
    if (moe_host_pool_mib() <= 0 || tensor_data == nullptr || path == nullptr || path[0] == '\0') {
        return;   // inert unless the bounded host pool is requested
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    host_src_t & s = g_host_src[tensor_data];
    s.path        = path;
    s.offs        = offs;
    s.n_experts   = n_experts;
    s.host_bytes  = host_bytes;
    s.total_bytes = total_bytes;
}

int moe_cache_table(const void * src0, int layer, const char * role, int n_experts, size_t expert_bytes,
                    size_t host_bytes, size_t src_off, size_t host_pitch, int split_axis,
                    const void * host, int device) {
    if (!g_enabled) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(g_mutex);

    // The cold read indexes the host master with the arena's row stride unless the table is a strided
    // axis-0 slice, in which case it uses the host row pitch.  A Phase 3 slice is therefore cold-safe
    // whenever the host geometry is representable: a contiguous per-expert blob (`host_pitch == 0`, any
    // `src_off`/`host_bytes`, covering the unsplit and contiguous axis-1 cases) or `rows` equal-width
    // rows (`host_pitch != 0`, `rows = host_bytes / host_pitch`, row width `expert_bytes / rows`).  A
    // geometry that cannot be expressed that way (the meta splitter never produces one) stays
    // cold-unsafe, so the hook fills every miss instead of emitting a bad cold id.
    bool cold_safe = true;
    if (host_pitch != 0) {
        cold_safe = host_bytes > 0 && host_bytes % host_pitch == 0 &&
                    expert_bytes > 0 && expert_bytes % (host_bytes / host_pitch) == 0;
    }

    const std::pair<const void*, int> key = { src0, device };
    const auto it = g_key_to_id.find(key);
    if (it != g_key_to_id.end()) {
        table_t & t = g_tables[it->second];
        if (t.device < 0) {
            t.device = device;   // bind the owner device once it is known
        }
        if (host != nullptr) {
            t.host = host;   // bind the master once it is known
        }
        attach_host_src_locked(t, t.host);
        bind_host_dev_locked(t);
        if (g_slots_hint > 0 && !t.allocated) {
            alloc_table_locked(t, g_slots_hint);
        }
        return it->second;
    }

    table_t t;
    t.layer        = layer;
    t.role         = (role != nullptr) ? role : "?";
    t.n_experts    = n_experts;
    t.expert_bytes = expert_bytes;
    t.host_bytes   = host_bytes > 0 ? host_bytes : expert_bytes;
    t.src_off      = src_off;
    t.host_pitch   = host_pitch;
    t.split_axis   = split_axis;
    t.cold_safe    = cold_safe;
    t.host         = host;
    t.device       = device;
    attach_host_src_locked(t, host);

    const int id = (int) g_tables.size();
    g_tables.push_back(std::move(t));
    g_key_to_id.emplace(key, id);
    g_sem_to_id.emplace(std::make_tuple(layer, sem_role((const ggml_tensor *) src0), device), id);
    bind_host_dev_locked(g_tables[id]);

    // An explicit `MOE_EXPERT_CACHE_SLOTS` is uniform and immediate (also what the self-test uses);
    // otherwise the arena is sized after the first (priming) pass, by `alloc_all_locked`.
    if (g_slots_hint > 0) {
        alloc_table_locked(g_tables[id], g_slots_hint);
    }

    return id;
}

moe_cache_alias moe_cache_alias_get(int table, int expert) {
    if (!g_enabled || table < 0 || table >= (int) g_tables.size()) {
        return { nullptr, false };
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    table_t & t = g_tables[table];
    const auto it = t.expert_slot.find(expert);
    if (it != t.expert_slot.end()) {
        return { (char *) t.arena + (size_t) it->second * t.expert_bytes, true };
    }
    if (t.host != nullptr && expert >= 0 && expert < t.n_experts && t.host_pitch == 0) {
        return { (void *) ((const char *) t.host + (size_t) expert * t.host_bytes + t.src_off), false };
    }
    return { nullptr, false };
}

void moe_cache_observe(const ggml_tensor * op, const ggml_tensor * src0, const ggml_tensor * ids) {
    if (!g_enabled || op == nullptr || src0 == nullptr || ids == nullptr) {
        return;
    }
    const int layer = name_layer(op);   // the op output carries the `-<layer>` suffix, not the weight
    if (layer < 0 || src0->ne[2] < 1) {
        return;
    }
    const int n_used = (int) ids->ne[0];
    const int n_tok  = (int) ids->ne[1];
    if (n_tok > MOE_EXPERT_CACHE_MAX_TOK) {
        return;   // decode/verify band only
    }

    const int    n_experts    = (int) src0->ne[2];
    // the expert stride; nb[2] for the contiguous expert tensor, falling back to the average.
    const size_t expert_bytes = src0->nb[2] > 0 ? (size_t) src0->nb[2]
                                                : (size_t) (ggml_nbytes(src0) / (size_t) n_experts);

    // register (idempotent) before taking the access lock
    const int table = moe_cache_table(src0, layer, name_role(src0).c_str(), n_experts, expert_bytes,
                                      expert_bytes, 0, 0, -1, src0->data, -1);
    if (table < 0) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_mutex);
    table_t & t = g_tables[table];
    for (int tok = 0; tok < n_tok; tok++) {
        const char * row = (const char *) ids->data + (size_t) tok * ids->nb[1];
        for (int j = 0; j < n_used; j++) {
            const int32_t e = *(const int32_t *) (row + (size_t) j * ids->nb[0]);
            if (e >= 0 && e < t.n_experts) {
                (void) access_locked(t, e, nullptr);
            }
        }
    }
}

// Device-side prefill tally (MOE_EXPERT_CACHE_PREFILL_SEED=1).  The scheduler's block-06 staging
// intercepts the prefill expert upload before `moe_cache_update_host` under `-sm tensor`, so the host
// hook never sees the prefill routing there; this is called from `ggml_cuda_mul_mat_id`, where the
// routing device tensor is in hand, and just histograms it on the compute stream.  Idempotent per
// (layer, role, device): the table is already registered by the load-time warmup decode.
bool moe_cache_tally_prefill(const ggml_tensor * op, const ggml_tensor * weight, const ggml_tensor * ids,
                             int device, void * stream) {
    // `g_devmap` because the seed is consumed by the device-policy flush; with `DEVMAP=0` the tally would
    // only waste a prefill kernel per op (the seed can never be applied).
    if (!g_enabled || !g_prefill_seed || !g_devmap || op == nullptr || weight == nullptr || ids == nullptr) {
        return false;
    }
    if (ids->type != GGML_TYPE_I32 || ids->ne[1] <= MOE_EXPERT_CACHE_MAX_TOK) {
        return false;   // decode/verify band (the band the cache serves), or a non-int routing
    }
    const int layer = name_layer(op);
    if (layer < 0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    int id = -1;
    // The op's `src0` is the per-device simple tensor (or the staged copy) whose name normalizes to the
    // host master's role, so the semantic key is the reliable lookup under `-sm tensor`.
    const auto sit = g_sem_to_id.find(std::make_tuple(layer, sem_role(weight), device));
    if (sit != g_sem_to_id.end()) {
        id = sit->second;
    } else {
        id = alias_find_checked(weight, device, layer);
    }
    if (id < 0 || id >= (int) g_tables.size()) {
        return false;
    }
    table_t & t = g_tables[id];
    if (t.prefill_seeded || t.n_experts <= 0) {
        return false;
    }
    device_guard dg(device);
    if (t.prefill_count_dev == nullptr) {
        if (cudaMalloc((void **) &t.prefill_count_dev, (size_t) t.n_experts * sizeof(int32_t)) != cudaSuccess) {
            (void) cudaGetLastError();
            return false;
        }
        (void) cudaMemsetAsync(t.prefill_count_dev, 0, (size_t) t.n_experts * sizeof(int32_t),
                               (cudaStream_t) stream);
    }
    const int n_used  = (int) ids->ne[0];
    const int n_tok   = (int) ids->ne[1];
    const int total   = n_used * n_tok;
    const int threads = 256;
    const int blocks  = (total + threads - 1) / threads;
    moe_cache_tally_kernel<<<blocks, threads, 0, (cudaStream_t) stream>>>(
        (const char *) ids->data, ids->nb[0], ids->nb[1], n_used, n_tok, t.n_experts, t.prefill_count_dev);
    t.prefill_tally_pending = true;
    return true;
}

// Device-side expert gather (B2).  `weight` is the host master, `weight_cpy` the device tensor the op
// will read, `ids` the routing (device, strided) and `slice_off`/`split_axis` the per-device slice
// geometry (`0`/`-1` for an unsplit table).  Launches one kernel on `stream`; true on success.  No host
// readback and no device sync - the caller has already ordered it on the compute stream.
// Expert-head guard (see `moe_cache_gather_host`): zero the first `nbytes` bytes of every expert
// slot ONCE per `input_cpy` buffer, so the MMQ's speculative read past a routed expert never sees a
// stale NaN.  Zero must only be finite; the gather overwrites the routed slots with real data.
static __global__ void moe_cache_gather_zero_heads_kernel(char * __restrict__ dst, int64_t expert_bytes,
                                                          int n_experts, int64_t nbytes) {
    const int e = blockIdx.x;
    if (e >= n_experts) {
        return;
    }
    char * p = dst + (int64_t) e * expert_bytes;
    for (int64_t i = threadIdx.x; i < nbytes; i += blockDim.x) {
        p[i] = 0;
    }
}

bool moe_cache_gather_host(const ggml_tensor * weight, const ggml_tensor * weight_cpy,
                           const ggml_tensor * ids, void * stream, int device,
                           size_t slice_off, int split_axis) {
    if (weight == nullptr || weight_cpy == nullptr || ids == nullptr || ids->data == nullptr) {
        return false;
    }
    const int64_t n_used = ids->ne[0];
    const int64_t n_tok  = ids->ne[1];
    if (n_used <= 0 || n_tok <= 0) {
        return false;
    }
    const int n_experts = (int) weight->ne[2];
    if (n_experts <= 0 || n_experts > 1 << 20) {
        return false;
    }
    const size_t host_bytes = weight->nb[2] > 0 ? (size_t) weight->nb[2]
                                                : (size_t) (ggml_nbytes(weight) / (size_t) n_experts);
    size_t       expert_bytes = host_bytes;
    size_t       host_pitch   = 0;
    if (split_axis >= 0 && weight_cpy->nb[2] > 0) {
        expert_bytes = (size_t) weight_cpy->nb[2];
        if (split_axis == 0) {
            host_pitch = weight->nb[1];
        }
    }
    if (host_bytes == 0 || expert_bytes == 0) {
        return false;
    }
    // Register the table with the cache even though this pass uploads through the gather.  The
    // deferred arena sizing (`alloc_all_locked`) latches on the first full decode pass and sizes every
    // table registered by then, so a table that is ONLY ever uploaded through the gather path would be
    // missing from that set - the arena then covers only the tables some other path happened to
    // register (observed: the last layer alone), and the decode band falls below the uncached path.
    // Registration is a cheap map insert and does not allocate (the deferred path leaves slots at 0).
    {
        const int layer = name_layer(weight);
        if (layer >= 0) {
            const int table = moe_cache_table(weight, layer, name_role(weight).c_str(), n_experts, expert_bytes,
                                              host_bytes, slice_off, host_pitch, split_axis, weight->data, device);
            // The gather kernel reads the host master IN PLACE (`weight->data` below).  That is only
            // safe when the master has a device mapping; for a pageable model mapping the kernel read
            // is a fatal page-not-present fault (issue #116).  Decline and let the scheduler's host
            // path stage the experts through its pinned scratch instead.
            bool accessible = false;
            if (table >= 0) {
                std::lock_guard<std::mutex> lock(g_mutex);
                accessible = g_tables[table].host_dev != nullptr;
            }
            if (!accessible) {
                return false;
            }
        }
    }
    // Finite-head guard (MMQ over-read): the quantized `MUL_MAT_ID` load speculatively reads past the
    // end of a routed expert into the NEXT expert slot's head.  The host path copies the whole table, so
    // those bytes are always finite; the pruned gather leaves them as the reused `input_cpy`'s stale
    // bytes (often NaN, and NaN * 0 = NaN poisons the tile).  Padding every routed expert on every gather
    // measured a ~3x loss, so instead establish the invariant ONCE: zero the first `head_pad` bytes of
    // every slot, below.  The gather's own copies then overwrite the routed slots with real, finite data,
    // so it holds for the life of this `input_cpy`; only the non-routed heads ever rely on the zero.
    // `head_pad` MUST match the host path's guard (`copy_experts`: `min(expert_size, 512)`) - the
    // over-read is quant-dependent, and 64 was measured to be enough for IQ4_NL only (IQ4_XS still
    // over-reads, corrupting the tile into the repeated-`/` output).  The one-time zero is free at any
    // size, so use the full 512.  The zero is keyed on `(buffer, expert_bytes)`, not the buffer alone:
    // the graph allocator reuses one `input_cpy` buffer across tables whose per-expert geometry differs,
    // and a zero laid down at one stride does not cover another's slots.
    const int64_t head_pad = expert_bytes < 512 ? (int64_t) expert_bytes : 512;
    device_guard dg(device);
    const int threads = 256;
    // 8 chunks/expert -> ~64 concurrent blocks for the usual 8 routed experts (see the kernel comment).
    const int n_split = 8;
    // The finite-head guard MUST run BEFORE the gather: the gather then overwrites every routed slot's
    // head with real data, so only the non-routed slots keep the zero (which is all the finite-byte
    // guard needs).  Launching it after the gather zeroes the routed experts' own first bytes and
    // corrupts the tile (the byte-identity regression).
    //
    // WIP r42 (stage-1 item 1): re-arm the guard on **every** gather, not once per
    // `(weight_cpy->data, expert_bytes)`.  The destination is the graph allocator's `input_cpy`, which
    // is reused across ubatches and across tables, so the once-only zero of r30/r31 ("Hole B") does
    // not survive a multi-ubatch prefill -- the non-routed heads then hold whatever the previous
    // graph left (often NaN), and the MMQ tail over-read poisons the tile.  The zero is `n_experts`
    // `head_pad`-byte writes (~256 KiB for 512 experts), i.e. free next to the expert upload it
    // guards.  `GGML_MOE_GATHER_ONCE=1` restores the once-only arm for A/B.
    if (head_pad > 0 && weight_cpy->data != nullptr) {
        static const bool once = getenv("GGML_MOE_GATHER_ONCE") != nullptr && atoi(getenv("GGML_MOE_GATHER_ONCE")) != 0;
        bool run = true;
        if (once) {
            std::lock_guard<std::mutex> lock(g_mutex);
            run = g_heads_zeroed.insert(std::make_pair((const void *) weight_cpy->data, (long long) expert_bytes)).second;
        }
        if (run) {
            moe_cache_gather_zero_heads_kernel<<<n_experts, 64, 0, (cudaStream_t) stream>>>(
                (char *) weight_cpy->data, (int64_t) expert_bytes, n_experts, head_pad);
        }
    }
    moe_cache_gather_kernel<<<n_experts * n_split, threads, 0, (cudaStream_t) stream>>>(
        (char *) weight_cpy->data, (const char *) weight->data,
        (const int32_t *) ids->data, ids->nb[0], ids->nb[1], (int) n_used, (int) n_tok, n_experts, n_split,
        (int64_t) expert_bytes, (int64_t) host_bytes, (int64_t) slice_off, (int64_t) host_pitch, split_axis);
    if (cudaGetLastError() != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    return true;
}

bool moe_cache_update_host(const ggml_tensor * weight, const ggml_tensor * weight_cpy,
                           const int32_t * ids, int64_t n_used, int64_t n_tok,
                           size_t ids_nb0, size_t ids_nb1, void * stream,
                           int device, size_t slice_off, int split_axis) {
    if (!g_enabled || weight == nullptr || ids == nullptr || weight->ne[2] < 1) {
        return false;
    }
    const int layer = name_layer(weight);
    if (layer < 0) {
        return false;
    }
    // Decode/verify band only, unchanged: when the prefill seed is off a prefill ubatch returns before
    // the table is even registered, exactly as before this feature existed.
    if (n_tok > MOE_EXPERT_CACHE_MAX_TOK && !g_prefill_seed) {
        return false;
    }
    const int    n_experts    = (int) weight->ne[2];
    // `host_bytes` is the full host-master expert stride; `expert_bytes` is THIS device's slice, which
    // the op reads and the arena stores.  They are equal for an unsplit table (1 GPU / `-sm layer`).
    const size_t host_bytes   = weight->nb[2] > 0 ? (size_t) weight->nb[2]
                                                  : (size_t) (ggml_nbytes(weight) / (size_t) n_experts);
    size_t       expert_bytes = host_bytes;
    size_t       host_pitch   = 0;
    if (split_axis >= 0 && weight_cpy != nullptr && weight_cpy->nb[2] > 0) {
        expert_bytes = (size_t) weight_cpy->nb[2];
        if (split_axis == 0) {
            host_pitch = weight->nb[1];   // a strided axis-0 slice needs a 2-D fill
        }
    }
    const int table = moe_cache_table(weight, layer, name_role(weight).c_str(), n_experts, expert_bytes,
                                      host_bytes, slice_off, host_pitch, split_axis, weight->data, device);
    if (table < 0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    table_t & t = g_tables[table];

    // wip/moe-expert-cache prefill seed (MOE_EXPERT_CACHE_PREFILL_SEED=1).  A prefill ubatch is never
    // cache-managed (the arena serves the decode band only), but its routing is the strongest available
    // prior for the decode hot set, and the scheduler has already read it back to the host for its own
    // op-offload pruning, so tallying it here is cheap.  The tally is bulk-admitted at arena sizing by
    // `apply_prefill_seed_locked`.  Only while unsized: after that the arena is fixed and the tally would
    // be wasted CPU.  NOTE: under the default block-06 prefill staging (`stage_consumed`) the expert
    // upload is intercepted before this hook, so the tally is inert there - see the campaign README.
    if (n_tok > MOE_EXPERT_CACHE_MAX_TOK) {
        if (g_prefill_seed && !g_sized && t.n_experts > 0) {
            if ((int) t.prefill_count.size() != t.n_experts) {
                t.prefill_count.assign((size_t) t.n_experts, 0);
            }
            const auto id_at = [&](int64_t tok, int64_t j) -> int32_t {
                return *(const int32_t *) ((const char *) ids + (size_t) tok * ids_nb1 + (size_t) j * ids_nb0);
            };
            for (int64_t tok = 0; tok < n_tok; tok++) {
                for (int64_t j = 0; j < n_used; j++) {
                    const int32_t e = id_at(tok, j);
                    if (e >= 0 && e < t.n_experts) {
                        t.prefill_count[(size_t) e]++;
                    }
                }
            }
            t.prefill_tokens += n_tok;
        }
        return false;   // decode/verify band only
    }

    // A pageable axis-0 slice (a `-sm tensor` `down_exps` split) has no device mapping and its only
    // cache fill is a strided 2-D H2D -- the pageable `cudaMemcpy2DAsync` that faults on ROCm 7.14
    // (issue #116's copy-side twin).  Decline the partial-residency case so the scheduler's
    // pageable-safe host path serves the op.  An identity (whole-resident) table needs no fill and is
    // left alone.
    if (t.host_dev == nullptr && t.split_axis == 0 && t.host_pitch > 0 && !t.identity) {
        g_decline_all++;
        return false;
    }

    // Per-device arena backstop (Phase 2).  Normally a table is allocated on its owner device (the
    // device is learned on the priming pass, before `alloc_all_locked` runs), so this never fires.
    // It covers the paths where the owner device is learned only after allocation: free the old
    // device's arena/remap and re-create them on `device`.  The old arena's bytes are gone, so the
    // residency map is reset (the next accesses refill from the host master).
    if (t.device >= 0 && t.device != device && t.allocated) {
        const int    old_device   = t.device;
        const int    old_slots    = t.slots;
        const size_t old_reserved = t.arena_reserved;
        void *       old_arena    = t.arena;
        void *       old_remap    = t.remap_dev;
        t.arena        = nullptr;
        t.remap_dev    = nullptr;
        t.remap_cap    = 0;
        t.slots        = 0;
        t.arena_reserved = 0;
        g_arena_bytes -= (int64_t) old_slots * (int64_t) t.expert_bytes;
        if (g_arena_bytes < 0) {
            g_arena_bytes = 0;
        }
        {
            device_guard dg(old_device);
            // Slab-backed arenas go back to the slab's free list; `cudaFree` can unmap the slab.
            if (old_arena != nullptr) { free_arena_backing(old_device, old_arena, old_reserved, old_slots, t); }
            if (old_remap != nullptr) { (void) cudaFree(old_remap); }
            (void) cudaGetLastError();   // fail soft: a bad free must not abort a run
        }
        t.device = device;
        alloc_table_locked(t, old_slots);
        t.slot_expert.assign(t.slots, -1);
        t.expert_slot.clear();
        GGML_LOG_WARN("%s: migrated layer=%d role=%s arena from device %d to %d (%d slots)\n",
                      __func__, t.layer, t.role.c_str(), old_device, device, t.slots);
    } else if (t.device < 0) {
        t.device = device;
    }

    // Invalidate the staged remap for this table until this call succeeds.  If the call returns false
    // (priming, no arena, or `!all` below), the scheduler copies the FULL expert table and the op must
    // read that copy - not the previous token's remap.  The consumer only redirects while
    // `remap_n_used > 0`, so zeroing it here turns "hook declined" into "op reads `input_cpy`".
    t.remap_n_used = 0;

    // Sizing and priming (see `alloc_all_locked`): with no explicit slot override, the first pass
    // only registers the tables and runs uncached (the scheduler copies the full experts); the
    // second pass sizes every table uniformly and starts filling.
    if (g_slots_hint <= 0) {
        if (!t.primed) {
            t.primed = true;
            g_total_expert_bytes += (int64_t) expert_bytes;
            g_total_one_expert_bytes += (int64_t) expert_bytes;
            return false;   // priming pass
        }
        if (!g_sized) {
            // OPEN 2: size ONLY on a decode-band pass, never mid-prefill.  The arena's capacity is
            // `mapped - boundary`, and the boundary carries the WIDE prefill work layout while that layout
            // is live.  Without this guard the sizing fires on the SECOND prefill UBatch (the first one
            // only primes the tables), i.e. before the drop releases the wide layout, and the cache is
            // permanently under-sized: measured on `-ub 4096 -c 163860` as `work 6.75 GiB + arena 12.31 GiB`
            // where the post-drop boundary gives `work ~1.5 GiB + arena ~17.5 GiB` -- ~5 GiB PER DEVICE left
            // unused, and the reason a 16k request failed at a small headroom.  The drop happens at the
            // prefill -> decode transition, so the first decode-band pass (n_tok <= 8, matching the drop's
            // own condition) always sees the narrow boundary.  With the drop DISABLED the boundary is still
            // wide at that point, which is correct there: the work region really is occupied.
            if (n_tok <= 8) {
                alloc_all_locked();
            }
        }
        if (t.arena == nullptr || t.slots <= 0) {
            return false;   // this table has no arena (fail-soft)
        }
    }

    // OPEN 2: the wholesale-fallback invariant (see `moe_cache_take_over`) must hold for EVERY consumer,
    // not just the fusion guard and the take-over hook.  While any table is stood down (a compute-buffer
    // OPEN 2 slab: the gate is per TABLE, not global.  The movable-boundary slab evicts the tables in the
    // chunks it hands to the work pool, so a PARTIAL cache is the normal state and the survivors must keep
    // serving.  This hook, the take-over hook and `moe_cache_get_table` all require THIS table to have a
    // usable arena (`t.arena`/`t.slots` below), so they still agree per table; the only global decision is
    // the fusion guard (`moe_cache_has_arena()`), which stands the cache-aware fusions down.

    // Alias the scheduler's redirected tensor so the op can find this table from its `src0`.  This
    // must OVERWRITE: the graph allocator reuses the `input_cpy` device pointers, so the same address
    // can serve a different (layer, role) across graphs; `emplace` would keep the first table and the
    // op would read the wrong arena/remap (observed as a residency-dependent divergence after ~100
    // tokens, once the allocator reshuffles).
    if (weight_cpy != nullptr) {
        g_alias_to_id[weight_cpy] = table;
    }

    if (t.remap_dev == nullptr) {
        const int64_t cap = (int64_t) n_experts * 8;
        if (cudaMalloc((void **) &t.remap_dev, (size_t) cap * sizeof(int32_t)) == cudaSuccess) {
            t.remap_cap = cap;
        } else {
            (void) cudaGetLastError();
            t.remap_dev = nullptr;
        }
    }
    if (t.remap_dev == nullptr || n_used * n_tok > t.remap_cap) {
        return false;   // cannot stage the remap: the op keeps the full-table path
    }
    // No arena (allocation failed, or a budget too small for a single slot).  The cold read could still
    // serve every expert, but the CONSUMER resolves its read source from the arena base
    // (`moe_cache_get_cold`), which is ambiguous when several tables have a null arena - so a takeover
    // here would make the consumer decline and read the un-staged `input_cpy`.  Decline instead: the
    // scheduler's full/used-expert copy is the correct path.
    if (t.arena == nullptr) {
        g_decline_all++;
        return false;
    }
    // A role that cannot produce a remap can never serve the table, so re-check readiness here too
    // (the sizing-time allocation above normally covers it).

    // `ids` is now a strided view (the scheduler fetched the raw tensor bytes), so index it with
    // the tensor's own strides rather than assuming a contiguous `n_used x n_tok` block.
    const auto id_at = [&](int64_t tok, int64_t j) -> int32_t {
        return *(const int32_t *) ((const char *) ids + (size_t) tok * ids_nb1 + (size_t) j * ids_nb0);
    };

    std::vector<int32_t> & remap = t.remap_host;
    if (remap.size() < (size_t) n_used * n_tok) {
        remap.resize((size_t) n_used * n_tok);
    }
    std::fill(remap.begin(), remap.begin() + (size_t) n_used * n_tok, -1);
    bool all = true;

    std::vector<int32_t> used;
    used.reserve((size_t) n_used * n_tok);
    for (int64_t tok = 0; tok < n_tok; tok++) {
        for (int64_t j = 0; j < n_used; j++) {
            const int32_t e = id_at(tok, j);
            if (e < 0 || e >= t.n_experts) {
                all = false;
                continue;
            }
            used.push_back(e);
        }
    }

    // A cache too small to hold this token's experts cannot guarantee a complete remap; decline
    // DETERMINISTICALLY (this bound is a constant per shape) so a shape never captures the arena and then
    // declines it on a later replay.  With UVA cold reads the overflow is representable (id = slots+e),
    // so a small arena is fine and the decline is not needed.
    if (!table_cold_ok(t) && t.slots < (int) (n_used * n_tok)) {
        g_decline_all++;
        return false;
    }

    // Pass 1: admit every used expert, protecting this token's experts from each other.  A later admission
    // used to be able to evict an earlier one (its freshly-seeded count is the smallest), which is what
    // left an earlier position pointing at a slot a later fill had refilled; the two-pass remap below
    // fixes the value, and the protection makes the success/failure of this hook CONSTANT per table
    // (true whenever `slots >= used.size()`), which is what a CUDA graph replay needs - a decision that
    // flips per token cannot be baked into one captured graph.
    for (int32_t e : used) {
        bool cold = false;
        (void) access_locked(t, e, stream, used.data(), (int) used.size(), &cold);
    }

    // Pass 2: build the remap from the FINAL map.  Nothing evicts between here and the read.
    //
    const int  n_res_slots = t.slots;
    for (int64_t tok = 0; tok < n_tok; tok++) {
        for (int64_t j = 0; j < n_used; j++) {
            const int32_t e = id_at(tok, j);
            const auto it = t.expert_slot.find(e);
            if (it != t.expert_slot.end()) {
                remap[(size_t) tok * n_used + j] = it->second;
            } else if (g_cold_uva && t.cold_safe && t.host_dev != nullptr && e >= 0 && e < t.n_experts) {
                // Phase 1b: not resident (never admitted / rejected / evicted).  Encode it in the COLD
                // region of the id space - the kernel reads `id - n_res_slots` from the host alias.
                remap[(size_t) tok * n_used + j] = n_res_slots + e;
                g_cold_reaches++;
            } else {
                all = false;   // evicted and not re-admitted, or slots == 0: fall back
            }
        }
    }
    if (!all) {
        g_decline_all++;
        return false;
    }
    // Async on the compute stream: the op reads the remap on that stream, and a synchronous copy on
    // the legacy stream is NOT ordered with the (non-blocking) compute stream, so the next graph's
    // hook could overwrite the remap before this graph's op reads it (observed as stale remaps under
    // eviction pressure).  The source is a persistent table member and the scheduler synchronizes the
    // backend before the next hook, so the async read does not race.
    (void) cudaMemcpyAsync(t.remap_dev, remap.data(), (size_t) n_used * n_tok * sizeof(int32_t),
                           cudaMemcpyHostToDevice, (cudaStream_t) stream);
    t.hook_experts.resize((size_t) n_used * n_tok);
    for (int64_t tok = 0; tok < n_tok; tok++) {
        for (int64_t j = 0; j < n_used; j++) {
            t.hook_experts[(size_t) tok * n_used + j] = id_at(tok, j);
        }
    }
    t.remap_n_used = n_used;
    t.remap_n_tok  = n_tok;
    g_takeover++;
    // Devmap transition arming: record that this devmap table has completed one eager fill pass.  Once
    // every devmap table has, the takeover fast path may switch on (next token).  Only a decode-band
    // eager stage counts, so the pass is a real decode pass.
    if (g_devmap && !g_devmap_armed && t.devmap && n_tok <= MOE_EXPERT_CACHE_MAX_TOK &&
        table >= 0 && table < (int) g_devmap_eager_seen.size() && !g_devmap_eager_seen[table]) {
        g_devmap_eager_seen[table] = 1;
        if (++g_devmap_arm_seen >= g_devmap_arm_expected) {
            g_devmap_armed = true;
            GGML_LOG_WARN("%s: device-remap fast path armed after a uniform eager pass over %d tables\n",
                          __func__, g_devmap_arm_expected);
        }
    }
    return true;
}

bool moe_cache_get_cold(const void * arena, void ** cold_base, int64_t * n_res,
                        size_t * cold_channel_bytes, size_t * cold_row_bytes) {
    if (!g_enabled || !g_cold_uva || arena == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    for (const table_t & t : g_tables) {
        if (t.arena != arena || t.slots <= 0 || t.host_dev == nullptr || !t.cold_safe) {
            continue;
        }
        // Fold in this device's slice offset so the kernel never needs to know the tensor-split
        // geometry: a cold expert `e` is at `*cold_base + e * *cold_channel_bytes`.
        *cold_base = (char *) t.host_dev + t.src_off;
        // The resident region is `t.slots`, so the cold encoding starts above it.
        *n_res     = t.slots;
        if (cold_channel_bytes) {
            *cold_channel_bytes = t.host_bytes;
        }
        if (cold_row_bytes) {
            *cold_row_bytes = t.host_pitch;   // 0 => the device slice's own row stride
        }
        return true;
    }
    return false;
}

bool moe_cache_kslot_active() {
    return g_enabled && g_kslot;
}

bool moe_cache_get_slot(const void * arena, const int32_t ** slot_dev, int32_t ** used_dev, int32_t * n_experts) {
    if (!g_enabled || !g_kslot || arena == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    // OPEN 2: per-table, like the other consumers (the loop below already skips a table with no arena).
    for (const table_t & t : g_tables) {
        if (t.arena != arena || !t.devmap || t.slots <= 0 || t.slot_dev == nullptr) {
            continue;
        }
        if (slot_dev)  *slot_dev  = t.slot_dev;
        if (used_dev)  *used_dev  = t.used_dev;
        if (n_experts) *n_experts = t.n_experts;
        return true;
    }
    return false;
}

bool moe_cache_get_table(const ggml_tensor * op, const ggml_tensor * weight_cpy, int device,
                         void ** arena, int64_t * n_slots,
                         size_t * expert_bytes, void ** remap_dev, int64_t * n_used, int64_t * n_tok,
                         moe_cache_devmap * dm) {
    if (!g_enabled || weight_cpy == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    int id = alias_find_checked(weight_cpy, device, op != nullptr ? name_layer(op) : -1);
    // Fallback for `-sm tensor`: a meta graph rebuild can hand the op a different simple-tensor pointer
    // than the one the hook saw, so resolve by the semantic (layer, role, device) key.  The buffer
    // check skips our own re-dispatch copy (`src0c.buffer = nullptr` in the consumer), whose name
    // pointer is inherited and which would otherwise recurse forever.
    if (id < 0 && op != nullptr && weight_cpy->buffer != nullptr) {
        const int layer = name_layer(op);
        const auto sit = g_sem_to_id.find(std::make_tuple(layer, sem_role(weight_cpy), device));
        if (sit != g_sem_to_id.end()) {
            id = sit->second;
        }
    }
    if (id < 0) {
        return false;
    }
    // OPEN 2: per-table, like the other consumers.  Every branch below requires THIS table's arena and
    // slots, and the caller re-checks the remap shape, so a partially-evicted cache still serves its
    // residents.
    table_t & t = g_tables[id];
    // Identity fast path: slot == expert, the whole table is resident, so the arena is indexed by the
    // raw routing ids and no remap (and no per-token host routing readback) is needed.  Restricted to
    // the decode/verify band: prefill keeps the normal path (its MMQ fusions and the MMQ dispatch read
    // the real copied buffer, and the shallow arena copy carries a null `buffer`).
    if (t.identity && t.arena != nullptr && t.slots >= t.n_experts &&
        op != nullptr && op->ne[2] <= MOE_EXPERT_CACHE_MAX_TOK) {
        g_get_ok++;
        if (arena)        *arena        = t.arena;
        if (n_slots)      *n_slots      = t.slots;
        if (expert_bytes) *expert_bytes = t.expert_bytes;
        if (remap_dev)    *remap_dev    = nullptr;
        if (n_used)       *n_used       = 0;
        if (n_tok)        *n_tok        = 0;
        return true;
    }
    // Device-remap mode (partial residency): the caller builds the remap from `dm->slot_dev` with
    // `moe_cache_launch_remap` before dispatching.  Same decode-band restriction as identity.
    if (t.devmap && t.arena != nullptr && t.slots > 0 && t.slot_dev != nullptr &&
        op != nullptr && op->ne[2] <= MOE_EXPERT_CACHE_MAX_TOK) {
        g_get_ok++;
        if (arena)        *arena        = t.arena;
        if (n_slots)      *n_slots      = t.slots;
        if (expert_bytes) *expert_bytes = t.expert_bytes;
        // B3: under KSLOT the consumer resolves the slot in-kernel, so there is no remap buffer to
        // consume.  Returning null here (rather than a stale buffer) makes the existing `remap !=
        // nullptr` guards at the redirect sites naturally keep the raw routing ids.
        if (remap_dev)    *remap_dev    = g_kslot ? nullptr : t.remap_dev;
        if (n_used)       *n_used       = 0;
        if (n_tok)        *n_tok        = 0;
        if (dm) {
            dm->slot_dev    = t.slot_dev;
            dm->used_dev    = t.used_dev;
            dm->n_experts   = t.n_experts;
            dm->n_res       = t.slots;
            dm->remap_fresh = t.remap_fresh;
        }
        return true;
    }
    if (t.arena == nullptr || t.slots <= 0 || t.remap_dev == nullptr || t.remap_n_used <= 0) {
        return false;
    }
    g_get_ok++;
    if (arena)        *arena        = t.arena;
    if (n_slots)      *n_slots      = t.slots;
    if (expert_bytes) *expert_bytes = t.expert_bytes;
    if (remap_dev)    *remap_dev    = t.remap_dev;
    if (n_used)       *n_used       = t.remap_n_used;
    if (n_tok)        *n_tok        = t.remap_n_tok;
    return true;
}

bool moe_cache_take_over(const ggml_tensor * weight, const ggml_tensor * weight_cpy, int device,
                         bool * need_promote) {
    if (need_promote != nullptr) {
        *need_promote = false;
    }
    if (!g_enabled || weight == nullptr || weight_cpy == nullptr || weight->ne[2] < 1) {
        return false;
    }
    if (name_layer(weight) < 0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    // A partially-failed cache MUST fall back wholesale, exactly as the fusion guard assumes.  The guard
    // stands the cache-aware fusions down as soon as any table is unserveable (`!moe_cache_has_arena()`),
    // and the scheduler hook must agree: if it still took the input over, the scheduler would skip
    // populating `input_cpy` and the fallback per-op path would read a tensor that was never filled --
    // after an arena release its stale `data` can even point into the freed arena (measured: a GPU VM
    // fault in `mul_mat_vec_q_moe` on an address inside a just-freed table arena).
    //
    // OPEN 2 slab: gate per TABLE.  The `identity`/`devmap` checks below already require THIS table's arena
    // and slots, so this hook, the update hook and `moe_cache_get_table` agree per table; only the fusion
    // guard is global (`moe_cache_has_arena()`), so a partial cache loses the fusions but keeps serving.
    const auto it = g_key_to_id.find({ (const void *) weight, device });
    if (it == g_key_to_id.end()) {
        return false;   // not registered yet (the first, priming token): the normal path sizes it
    }
    const int id = it->second;
    if (id < 0 || id >= (int) g_tables.size()) {
        return false;
    }
    table_t & t = g_tables[id];
    const bool identity = t.identity && t.arena != nullptr && t.slots >= t.n_experts;
    const bool devmap   = g_devmap_armed && t.devmap && t.arena != nullptr && t.slots > 0 && t.slot_dev != nullptr;
    if (!identity && !devmap) {
        return false;
    }
    // Overwrite: the graph allocator reuses these device pointers across graphs (see
    // `moe_cache_update_host`), so the alias must track the current tensor.
    g_alias_to_id[weight_cpy] = id;
    if (need_promote != nullptr && !identity) {
        *need_promote = true;   // a partial-residency table needs the deferred promotion pass
    }
    return true;
}

bool moe_cache_promote_host(const ggml_tensor * weight, const ggml_tensor * weight_cpy,
                            const int32_t * ids, int64_t n_used, int64_t n_tok,
                            size_t ids_nb0, size_t ids_nb1, void * stream, int device) {
    if (!g_enabled || !g_devmap || weight == nullptr || weight->ne[2] < 1) {
        return false;
    }
    if (n_tok > MOE_EXPERT_CACHE_MAX_TOK || name_layer(weight) < 0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    const auto it = g_key_to_id.find({ (const void *) weight, device });
    if (it == g_key_to_id.end()) {
        return false;
    }
    table_t & t = g_tables[it->second];
    if (!t.devmap || t.arena == nullptr || t.slots <= 0 || t.slot_dev == nullptr) {
        return false;
    }
    // this pass is over for this table: the next redirect must (re)build its remap
    t.remap_fresh = false;
    // Device-side admission policy: this per-table call only records the token's routing shape; the
    // batched kernel runs once per device in `moe_cache_policy_flush` (called by the scheduler after
    // this per-table loop).  No host policy, no used-list D2H, no slot-map H2D - the kernel updates
    // the device map in place.
    if (t.policy) {
        t.policy_n_used  = (int) n_used;
        t.policy_n_tok   = (int) n_tok;
        t.policy_pending = true;
        return true;
    }
    // The promotion allocates/fills into and copies to `t.arena`/`t.slot_dev`, which live on `t.device`,
    // so make that device current for the async fills and the device-map update (a stream from another
    // device does not make the pointer's device current).
    device_guard dg(t.device);
    if (weight_cpy != nullptr) {
        g_alias_to_id[weight_cpy] = it->second;
    }
    std::vector<int32_t> used;
    used.reserve((size_t) n_used * n_tok);
    // Read the routing from the table's persistent used-list buffer, which the remap kernel wrote during
    // the graph.  The graph's own routing tensor must NOT be read here: its storage is recycled once the
    // graph completes, so a deferred readback returns garbage (and the subsequent promotions no-op).
    if (t.used_dev != nullptr && t.used_host != nullptr && t.used_host2 != nullptr && t.used_cap >= n_used * n_tok) {
        // PIPELINED readback (2026-09-29): a per-table *synchronous* D2H on 240 tables measured
        // ~4.8 ms/token (81 % of the whole deferred pass).  Instead enqueue this token's readback
        // async (pinned dst, backend stream) and apply the policy to the PREVIOUS call's buffer, whose
        // copy the backend synchronize between tokens has already completed.  One token of extra lag,
        // which `touch` admission tolerates; the number of D2H API calls is unchanged, but none of
        // them blocks.  `used_host`/`used_host2` alternate, so a call never reads the buffer it is
        // about to overwrite (and the in-flight copy is ordered on the stream, not by the host).
        int32_t *       write_buf = t.used_toggle ? t.used_host2 : t.used_host;
        const int32_t * read_buf  = t.used_toggle ? t.used_host : t.used_host2;
        if (cudaMemcpyAsync(write_buf, t.used_dev, (size_t) n_used * n_tok * sizeof(int32_t),
                            cudaMemcpyDeviceToHost, (cudaStream_t) stream) != cudaSuccess) {
            (void) cudaGetLastError();
            return false;
        }
        if (t.used_pending) {
            for (int64_t i = 0; i < n_used * n_tok; i++) {
                const int32_t e = read_buf[(size_t) i];
                if (e >= 0 && e < t.n_experts) {
                    used.push_back(e);
                }
            }
        }
        t.used_toggle ^= 1;
        t.used_pending = true;
    } else if (ids != nullptr) {
        for (int64_t tok = 0; tok < n_tok; tok++) {
            for (int64_t j = 0; j < n_used; j++) {
                const int32_t e = *(const int32_t *) ((const char *) ids + (size_t) tok * ids_nb1 + (size_t) j * ids_nb0);
                if (e >= 0 && e < t.n_experts) {
                    used.push_back(e);
                }
            }
        }
    }
    if (used.empty()) {
        g_promote_calls++;
        return true;
    }
    // LFRU admission + fills for this token's used experts, protected from each other.  The admission
    // rule is the same active policy as the eager hook's (`touch` by default): `access_locked` only
    // applies it when `out_cold` is non-null.
    for (int32_t e : used) {
        bool cold = false;
        (void) access_locked(t, e, stream, used.data(), (int) used.size(), &cold);
    }
    if (t.slot_dirty) {
        for (int i = 0; i < t.n_experts; i++) {
            t.slot_pin[i] = -1;
        }
        for (const auto & kv : t.expert_slot) {
            if (kv.first >= 0 && kv.first < t.n_experts) {
                t.slot_pin[kv.first] = kv.second;
            }
        }
        (void) cudaMemcpyAsync(t.slot_dev, t.slot_pin, (size_t) t.n_experts * sizeof(int32_t),
                               cudaMemcpyHostToDevice, (cudaStream_t) stream);
        t.slot_dirty = false;
    }
    g_promote_calls++;
    return true;
}

// Build (or rebuild) the per-device device-policy descriptor array.  Runs after sizing, when every
// table is registered; the device arrays are seeded from the host mirrors so the residency the arming
// eager pass built is not lost when the device policy takes over.
static void build_policy_descs_locked(int device) {
    std::vector<int> ids;
    for (int i = 0; i < (int) g_tables.size(); i++) {
        const table_t & t = g_tables[i];
        if (t.policy && t.device == device) {
            ids.push_back(i);
        }
    }
    if (device >= (int) g_policy_dev.size()) {
        g_policy_dev.resize(device + 1);
    }
    policy_dev_t & pd = g_policy_dev[device];
    if (pd.initialized && pd.n == (int) ids.size()) {
        return;   // the set of policy tables on this device is stable
    }
    device_guard dg(device);
    if (pd.desc != nullptr)      { (void) cudaFree(pd.desc);      pd.desc = nullptr; }
    if (pd.shape_dev != nullptr) { (void) cudaFree(pd.shape_dev); pd.shape_dev = nullptr; }
    pd.ids         = ids;
    pd.n           = (int) ids.size();
    pd.cap         = pd.n;
    pd.initialized = true;
    pd.shape_host.assign((size_t) pd.n, 0);
    if (pd.n == 0) {
        return;
    }
    std::vector<moe_cache_policy_desc> host((size_t) pd.n);
    for (int i = 0; i < pd.n; i++) {
        table_t & t = g_tables[ids[i]];
        moe_cache_policy_desc & d = host[(size_t) i];
        d.slot_expert  = t.slot_expert_dev;
        d.slot         = t.slot_dev;
        d.count        = t.count_dev;
        d.ghost        = t.ghost_dev;
        d.last         = t.last_dev;
        d.slot_prov    = t.slot_prov_dev;
        d.used         = t.used_dev;
        d.arena        = t.arena;
        d.host_dev     = t.host_dev;
        d.host_bytes   = (int64_t) t.host_bytes;
        d.src_off      = (int64_t) t.src_off;
        d.expert_bytes = (int64_t) t.expert_bytes;
        d.host_pitch   = (int64_t) t.host_pitch;
        d.clock        = 0;
        d.last_decay   = 0;
        d.n_experts    = t.n_experts;
        d.slots        = t.slots;
        d.occupied     = (int32_t) t.expert_slot.size();
        d.split_axis   = t.split_axis;
        d.acc_hits     = 0;
        d.acc_misses   = 0;
        d.acc_fills    = 0;
        d.acc_evict    = 0;
        // Seed the device counters/residency from the host mirrors (the arming eager pass).  The host
        // clock restarts on the device, which only shifts the decay cadence, never correctness.
        std::vector<int32_t> sc((size_t) t.n_experts, 0), sg((size_t) t.n_experts, 0), sl((size_t) t.n_experts, 0);
        for (int e = 0; e < t.n_experts; e++) {
            sc[(size_t) e] = (size_t) e < t.count.size() ? (int32_t) t.count[(size_t) e] : 0;
            sg[(size_t) e] = (size_t) e < t.ghost.size() ? (int32_t) t.ghost[(size_t) e] : 0;
            sl[(size_t) e] = (size_t) e < t.last.size()  ? (int32_t) t.last[(size_t) e]  : 0;
        }
        (void) cudaMemcpy(t.count_dev, sc.data(), (size_t) t.n_experts * sizeof(int32_t), cudaMemcpyHostToDevice);
        (void) cudaMemcpy(t.ghost_dev, sg.data(), (size_t) t.n_experts * sizeof(int32_t), cudaMemcpyHostToDevice);
        (void) cudaMemcpy(t.last_dev,  sl.data(), (size_t) t.n_experts * sizeof(int32_t), cudaMemcpyHostToDevice);
        std::vector<int32_t> se((size_t) t.slots, -1);
        for (const auto & kv : t.expert_slot) {
            if (kv.second >= 0 && kv.second < t.slots) {
                se[(size_t) kv.second] = kv.first;
            }
        }
        std::vector<int32_t> sm((size_t) t.n_experts, -1);
        for (const auto & kv : t.expert_slot) {
            if (kv.first >= 0 && kv.first < t.n_experts) {
                sm[(size_t) kv.first] = kv.second;
            }
        }
        (void) cudaMemcpy(t.slot_expert_dev, se.data(), (size_t) t.slots * sizeof(int32_t), cudaMemcpyHostToDevice);
        (void) cudaMemcpy(t.slot_dev, sm.data(), (size_t) t.n_experts * sizeof(int32_t), cudaMemcpyHostToDevice);
        if (t.slot_prov_dev != nullptr && (int) t.slot_prov.size() == t.slots) {
            (void) cudaMemcpy(t.slot_prov_dev, t.slot_prov.data(), (size_t) t.slots * sizeof(uint8_t), cudaMemcpyHostToDevice);
        }
        t.slot_dirty = false;
    }
    void * desc_dev = nullptr;
    if (cudaMalloc(&desc_dev, host.size() * sizeof(moe_cache_policy_desc)) != cudaSuccess) {
        (void) cudaGetLastError();
        pd.desc        = nullptr;
        pd.initialized = false;   // retry on the next flush (fail-soft)
        return;
    }
    (void) cudaMemcpy(desc_dev, host.data(), host.size() * sizeof(moe_cache_policy_desc), cudaMemcpyHostToDevice);
    pd.desc = (moe_cache_policy_desc *) desc_dev;
    if (cudaMalloc((void **) &pd.shape_dev, (size_t) pd.n * sizeof(int32_t)) != cudaSuccess) {
        (void) cudaGetLastError();
        (void) cudaFree(pd.desc);
        pd.desc        = nullptr;
        pd.shape_dev   = nullptr;
        pd.initialized = false;   // retry on the next flush (fail-soft)
    }
}

// Per-token progress log: cumulative admissions (fills) vs evictions.  While the cache is warming up
// admits > evicts (empty slots are still being filled); once the arena is full and the working set is
// stable the per-interval deltas match.  Reads the device counters under DEVPOLICY, the host counters
// otherwise.
static void moe_cache_progress_locked(int device, policy_dev_t & pd, int64_t now_us) {
    int64_t fills = 0, evicts = 0, hits = 0, misses = 0, occupied = 0;
    int64_t slots = 0;
    int     n_tab = 0;
    if (g_devpolicy && pd.desc != nullptr && pd.n > 0) {
        device_guard dg(device);
        std::vector<moe_cache_policy_desc> host((size_t) pd.n);
        if (cudaMemcpy(host.data(), pd.desc, (size_t) pd.n * sizeof(moe_cache_policy_desc),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            (void) cudaGetLastError();
            return;
        }
        for (int i = 0; i < pd.n; i++) {
            fills    += host[(size_t) i].acc_fills;
            evicts   += host[(size_t) i].acc_evict;
            hits     += host[(size_t) i].acc_hits;
            misses   += host[(size_t) i].acc_misses;
            occupied += host[(size_t) i].occupied;
            slots    += host[(size_t) i].slots;
            n_tab++;
        }
    } else {
        for (const table_t & t : g_tables) {
            if (t.device != device || t.slots <= 0) {
                continue;
            }
            fills    += t.fills;
            evicts   += t.evictions;
            hits     += t.hits;
            misses   += t.misses;
            occupied += (int64_t) t.expert_slot.size();
            slots    += t.slots;
            n_tab++;
        }
    }
    const int64_t df      = fills   - pd.last_fills;
    const int64_t de      = evicts  - pd.last_evicts;
    const int64_t dres    = occupied - pd.last_resident;
    const int64_t dt_us   = (pd.last_time_us > 0 && now_us > pd.last_time_us) ? now_us - pd.last_time_us : 0;
    const double  dt_s    = (double) dt_us / 1e6;
    const double  a_rate  = dt_s > 0.0 ? (double) df   / dt_s : 0.0;
    const double  e_rate  = dt_s > 0.0 ? (double) de   / dt_s : 0.0;
    const double  r_rate  = dt_s > 0.0 ? (double) dres / dt_s : 0.0;
    const int64_t total   = hits + misses;
    const double  h       = total > 0 ? (double) hits / (double) total : 0.0;
    // Instantaneous (per-interval) hit rate: the cumulative `h` hides the warm-up crossover.
    const int64_t dh      = hits   - pd.last_hits;
    const int64_t dm      = misses - pd.last_misses;
    const double  h_int   = (dh + dm) > 0 ? (double) dh / (double) (dh + dm) : 0.0;
    // Instantaneous state from the RATES, not the running totals: QUIESCENT = no admissions at all
    // (the resident set has stopped growing); CHURN = admits == evicts > 0 (full, stable set);
    // FILLING = admitting more than evicting (empty slots remain, resident rising).
    const char * state = (df == 0 && de == 0) ? "QUIESCENT" : (df <= de ? "CHURN" : "FILLING");
    GGML_LOG_WARN("%s: dev %d tok=%lld dt=%.2fs admits=+%lld (%.1f/s) evicts=+%lld (%.1f/s) dRes=+%lld (%.1f/s) "
                  "resident=%lld/%lld (%.1f%%) h=%.4f h_int=%.4f tables=%d %s\n",
                  __func__, device, (long long) pd.tokens, dt_s,
                  (long long) df, a_rate, (long long) de, e_rate, (long long) dres, r_rate,
                  (long long) occupied, (long long) slots,
                  slots > 0 ? 100.0 * (double) occupied / (double) slots : 0.0, h, h_int, n_tab, state);
    pd.last_fills    = fills;
    pd.last_evicts   = evicts;
    pd.last_resident = occupied;
    pd.last_time_us  = now_us;
    pd.last_hits     = hits;
    pd.last_misses   = misses;
}

bool moe_cache_policy_flush(int device, void * stream) {
    if (!g_enabled || device < 0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    if (device >= (int) g_policy_dev.size()) {
        g_policy_dev.resize(device + 1);
    }
    policy_dev_t & pd = g_policy_dev[device];
    pd.tokens++;
    if (g_devpolicy) {
        build_policy_descs_locked(device);
        // Prompt-routing seed (MOE_EXPERT_CACHE_PREFILL_SEED=1): the first decode-band flush after the
        // prefill tally is non-empty bulk-admits the prompt's hottest experts; a true return means the
        // device state must be resynced from the (now seeded) host mirrors before the policy kernel.
        if (seed_prefill_lazy_locked(device, stream)) {
            build_policy_descs_locked(device);
        }
        if (pd.desc != nullptr && pd.shape_dev != nullptr && pd.n > 0) {
            for (int i = 0; i < pd.n; i++) {
                table_t & t = g_tables[pd.ids[(size_t) i]];
                pd.shape_host[(size_t) i] = t.policy_pending ? (int32_t) (t.policy_n_used * t.policy_n_tok) : 0;
                t.policy_pending = false;
            }
            device_guard dg(device);
            (void) cudaMemcpyAsync(pd.shape_dev, pd.shape_host.data(), (size_t) pd.n * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, (cudaStream_t) stream);
            const int threads = 256;
            const int admit_arg = g_admit;
            moe_cache_policy_kernel<<<pd.n, threads, 0, (cudaStream_t) stream>>>(
                pd.desc, pd.shape_dev, pd.n, admit_arg, (int) g_touch, (int) g_period,
                0, g_fill ? 1 : 0, g_prov_evict ? 1 : 0);
        }
    }
    return true;
}

// When a layer's gate+up pair is redirected, the SAME kernel can build the layer's routed-`down`
// table's remap too: the gate+up op runs before the down op, all three read the same routing, and the
// down's own slot map is handed to the kernel (so the result is correct even if a role's map diverges).
// The down redirect then skips its own - now redundant - launch this pass.  `remap_fresh` is cleared by
// the per-table promotion after the graph, so eager and capture/replay both see a consistent decision.
static bool moe_cache_sibling_down(const ggml_tensor * op, int device,
                                   const int32_t ** slot2, int32_t ** remap2, int32_t ** used3) {
    const int layer = name_layer(op);
    if (layer < 0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    for (table_t & t : g_tables) {
        if (t.layer != layer || t.device != device) {
            continue;
        }
        if (t.role.find("down_exps") == std::string::npos) {
            continue;
        }
        if (!(t.devmap && t.arena != nullptr && t.slots > 0 && t.slot_dev != nullptr &&
              t.remap_dev != nullptr && t.used_dev != nullptr)) {
            return false;   // a registered down role that cannot be served: do not fold
        }
        t.remap_fresh = true;
        if (slot2)  *slot2  = t.slot_dev;
        if (remap2) *remap2 = t.remap_dev;
        if (used3)  *used3  = t.used_dev;
        return true;
    }
    return false;
}

bool moe_cache_redirect_fused(const ggml_tensor * op, const ggml_tensor * src0, const ggml_tensor * gate,
                              const ggml_tensor * ids, int device, void * stream,
                              ggml_tensor * src0_cpy, ggml_tensor * ids_cpy, ggml_tensor * gate_cpy) {
    if (!g_enabled || ids == nullptr || src0 == nullptr) {
        return false;
    }
    void *    arena = nullptr;
    void *    remap = nullptr;
    int64_t   ns = 0, nu = 0, nt = 0;
    size_t    eb = 0;
    moe_cache_devmap dm{};
    if (!moe_cache_get_table(op, src0, device, &arena, &ns, &eb, &remap, &nu, &nt, &dm)) {
        return false;
    }
    // Resolve the gate lane BEFORE the up remap launch: when BOTH lanes are device-remap tables the
    // gate table's used-list write is folded into the up kernel (one launch for both tables), because
    // the fused kernel reads the up table's remap for both lanes and the gate's remap VALUE is unused.
    void *    gate_arena = nullptr;
    void *    gr         = nullptr;
    bool      gate_ok    = false;
    int64_t   gnu = 0, gnt = 0;
    moe_cache_devmap gdm{};
    if (gate != nullptr) {
        void *  ga = nullptr;
        int64_t gs = 0;
        size_t  geb = 0;
        gate_ok = moe_cache_get_table(op, gate, device, &ga, &gs, &geb, &gr, &gnu, &gnt, &gdm);
        if (!gate_ok) {
            return false;   // never leave one lane redirected and the other reading a stale copy
        }
        gate_arena = ga;
    }
    // Device-remap mode: build the remap on the device from the routing + the device slot map.  The
    // scheduler took this input over without populating the copied table, so this MUST succeed.  The
    // gate+up redirect (`gate != nullptr`) also builds the layer's routed `down` remap in the same
    // kernel (see `moe_cache_sibling_down`), so the down redirect can skip its own launch this pass.
    int32_t * gate_used2 = nullptr;
    if (dm.slot_dev != nullptr) {
        if (!g_kslot) {
            if (gate_ok && gdm.slot_dev != nullptr && gdm.used_dev != nullptr && gr != nullptr) {
                gate_used2 = gdm.used_dev;
            }
            if (!dm.remap_fresh) {
                const int32_t * slot2  = nullptr;
                int32_t *       remap2 = nullptr;
                int32_t *       used3  = nullptr;
                if (gate != nullptr) {
                    (void) moe_cache_sibling_down(op, device, &slot2, &remap2, &used3);
                }
                moe_cache_launch_remap(ids->data, ids->nb[0], ids->nb[1], &dm, ids->ne[0], ids->ne[1],
                                       (int32_t *) remap, stream, gate_used2, slot2, remap2, used3);
            }
        }
        nu = ids->ne[0];
        nt = ids->ne[1];
    }
    // remap == nullptr is the identity fast path: the raw routing ids already index the arena.
    if (remap != nullptr && (nu != ids->ne[0] || nt != ids->ne[1])) {
        return false;
    }
    // An eager (non-devmap) gate remap must match the up shape (unchanged semantics).
    if (gate_ok && gr != nullptr && gdm.slot_dev == nullptr && (gnu != nu || gnt != nt)) {
        return false;   // never leave one lane redirected and the other reading a stale copy
    }
    // The gate lane's remap VALUE is unused by the fused kernel, but the deferred promotion is per
    // table and needs the gate table's routing.  When the up lane is not a device-remap table there is
    // nothing to fold the write into, so launch the standby gate remap (it also fills the used-list).
    if (gate_ok && gdm.slot_dev != nullptr && gdm.used_dev != nullptr && gr != nullptr &&
        gate_used2 == nullptr) {
        moe_cache_launch_remap(ids->data, ids->nb[0], ids->nb[1], &gdm, ids->ne[0], ids->ne[1],
                               (int32_t *) gr, stream);
    }
    *src0_cpy = *src0;
    src0_cpy->data   = arena;
    src0_cpy->buffer = nullptr;
    *ids_cpy = *ids;
    if (remap != nullptr) {
        ids_cpy->data  = remap;
        ids_cpy->nb[0] = sizeof(int32_t);
        ids_cpy->nb[1] = (size_t) nu * sizeof(int32_t);
    }   // else: identity - keep the raw ids and their strides
    if (gate != nullptr) {
        *gate_cpy = *gate;
        gate_cpy->data   = gate_arena;
        gate_cpy->buffer = nullptr;
    }
    return true;
}

void moe_cache_read_check(const ggml_tensor * weight_cpy, const ggml_tensor * ids, void * stream) {
    if (!g_enabled || weight_cpy == nullptr || ids == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    const auto it = g_alias_to_id.find(weight_cpy);
    if (it == g_alias_to_id.end()) {
        return;
    }
    table_t & t = g_tables[it->second];
    if (t.policy) {
        return;   // host mirrors are stale under the device-side policy; the remap is built on the device
    }
    if (t.remap_dev == nullptr || t.remap_n_used <= 0) {
        return;
    }
    const int64_t n_used = ids->ne[0];
    const int64_t n_tok  = ids->ne[1];
    if (n_used != t.remap_n_used || n_tok != t.remap_n_tok) {
        GGML_LOG_ERROR("%s: SHAPE layer=%d role=%s ids=(%lld,%lld) remap=(%lld,%lld)\n",
                       __func__, t.layer, t.role.c_str(), (long long) n_used, (long long) n_tok,
                       (long long) t.remap_n_used, (long long) t.remap_n_tok);
        return;
    }
    (void) cudaStreamSynchronize((cudaStream_t) stream);
    // D2H the op's own routing (the device tensor), then compare it to the routing the hook consumed
    // and to the staged remap.  Reading `ids->data` on the host is only valid for host-mapped buffers,
    // so copy explicitly.
    const size_t id_bytes = (size_t) ids->nb[1] * (size_t) (ids->ne[1] > 0 ? ids->ne[1] : 1);
    std::vector<int32_t> idbuf(id_bytes / sizeof(int32_t) + 1, -1);
    (void) cudaMemcpy(idbuf.data(), ids->data, id_bytes, cudaMemcpyDeviceToHost);
    const char * ib = (const char *) idbuf.data();
    std::vector<int32_t> r((size_t) n_used * n_tok, -1);
    (void) cudaMemcpy(r.data(), t.remap_dev, (size_t) n_used * n_tok * sizeof(int32_t), cudaMemcpyDeviceToHost);
    std::vector<uint8_t> rd((size_t) t.expert_bytes);
    int bad = 0;
    for (int64_t tok = 0; tok < n_tok; tok++) {
        for (int64_t j = 0; j < n_used; j++) {
            const int32_t e_dev = *(const int32_t *) (ib + (size_t) tok * ids->nb[1] + (size_t) j * ids->nb[0]);
            const int32_t e_hook = t.hook_experts[(size_t) tok * n_used + j];
            const int32_t rv = r[(size_t) tok * n_used + j];
            if (e_dev != e_hook && bad++ < 6) {
                GGML_LOG_ERROR("%s: ROUTE layer=%d role=%s tok=%lld j=%lld dev=%d hook=%d\n",
                               __func__, t.layer, t.role.c_str(), (long long) tok, (long long) j, e_dev, e_hook);
            }
            const auto m = t.expert_slot.find(e_dev);
            const int32_t expect = (m != t.expert_slot.end()) ? m->second : -1;
            if (rv != expect && bad++ < 6) {
                GGML_LOG_ERROR("%s: STALE layer=%d role=%s tok=%lld j=%lld e=%d remap=%d map=%d\n",
                               __func__, t.layer, t.role.c_str(), (long long) tok, (long long) j, e_dev, rv, expect);
                continue;
            }
            if (rv >= 0 && rv < t.slots && t.host != nullptr) {
                (void) cudaMemcpy(rd.data(), (const char *) t.arena + (size_t) rv * t.expert_bytes,
                                  t.expert_bytes, cudaMemcpyDeviceToHost);
                if (memcmp(rd.data(), (const char *) t.host + (size_t) e_dev * t.expert_bytes, t.expert_bytes) != 0 && bad++ < 6) {
                    GGML_LOG_ERROR("%s: BYTES layer=%d role=%s tok=%lld j=%lld e=%d slot=%d\n",
                                   __func__, t.layer, t.role.c_str(), (long long) tok, (long long) j, e_dev, rv);
                }
            }
        }
    }
}

void moe_cache_init(int device) {
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!g_init_done) {
            g_init_done = true;
            parse_env();
            if (!g_enabled) {
                return;
            }
            g_device = device;
            if (g_report) {
                static bool reg = false;
                if (!reg) {
                    reg = true;
                    atexit(moe_cache_report);
                }
            }
            GGML_LOG_INFO("%s: MoE expert cache enabled on device %d: budget=%s MiB/device, period=%lld steps, %s\n",
                          __func__, device, budget_desc(), (long long) g_period,
                          g_slots_hint > 0 ? "explicit slots/table" : "adaptive uniform slots (deferred sizing)");
        }
    }
}

void moe_cache_report() {
    if (!g_enabled) {
        return;
    }    std::lock_guard<std::mutex> lock(g_mutex);
    // Device-side policy: the host counters are stale (the kernel owns them).  D2H each per-device
    // descriptor array and copy its cumulative counters back so the report and `h` are meaningful.
    for (int d = 0; d < (int) g_policy_dev.size(); d++) {
        policy_dev_t & pd = g_policy_dev[(size_t) d];
        if (pd.desc == nullptr || pd.n <= 0) {
            continue;
        }
        device_guard dg(d);
        std::vector<moe_cache_policy_desc> host((size_t) pd.n);
        if (cudaMemcpy(host.data(), pd.desc, (size_t) pd.n * sizeof(moe_cache_policy_desc),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            (void) cudaGetLastError();
            continue;
        }
        for (int i = 0; i < pd.n; i++) {
            table_t & t = g_tables[pd.ids[(size_t) i]];
            t.hits      = host[(size_t) i].acc_hits;
            t.misses    = host[(size_t) i].acc_misses;
            t.fills     = host[(size_t) i].acc_fills;
            t.evictions = host[(size_t) i].acc_evict;
        }
    }
    int64_t hits = 0, misses = 0, fills = 0, evictions = 0;
    int n_tables = 0, n_slots = 0;
    int64_t dev_slots[GGML_CUDA_MAX_DEVICES]  = {0};
    int64_t dev_arena[GGML_CUDA_MAX_DEVICES]  = {0};
    int     dev_tables[GGML_CUDA_MAX_DEVICES] = {0};
    for (const table_t & t : g_tables) {
        hits += t.hits; misses += t.misses; fills += t.fills; evictions += t.evictions;
        if (t.slots > 0) {
            n_tables++; n_slots += t.slots;
            if (t.device >= 0 && t.device < GGML_CUDA_MAX_DEVICES) {
                dev_slots [t.device] += t.slots;
                dev_arena [t.device] += (int64_t) t.slots * (int64_t) t.expert_bytes;
                dev_tables[t.device]++;
            }
        }
    }
    const int64_t total = hits + misses;
    const double h = total > 0 ? (double) hits / (double) total : 0.0;
    GGML_LOG_INFO("%s: MoE expert cache: h=%.4f (%lld/%lld reaches), fills=%lld evictions=%lld, tables=%d, slots=%d, arena=%.1f MiB total (budget %s MiB/device)\n",
                  __func__, h, (long long) hits, (long long) total, (long long) fills, (long long) evictions,
                  n_tables, n_slots, (double) g_arena_bytes / (1024 * 1024), budget_desc());
    if (g_pool_enabled) {
        int64_t p_slots = 0, p_hits = 0, p_misses = 0, p_fills = 0, p_evict = 0, p_bytes = 0;
        for (const auto & kv : g_host_pools) {
            const host_pool_t & p = kv.second;
            p_slots += p.slots; p_hits += p.hits; p_misses += p.misses;
            p_fills += p.fills; p_evict += p.evictions;
            p_bytes += (int64_t) p.slots * (int64_t) p.host_bytes;
        }
        const int64_t ptot = p_hits + p_misses;
        GGML_LOG_INFO("%s: host pool (L2): %zu tensors, %lld slots, %.1f MiB pinned; h=%.4f (%lld/%lld), "
                      "DIO fills=%lld evictions=%lld\n",
                      __func__, g_host_pools.size(), (long long) p_slots, (double) p_bytes / (1024 * 1024),
                      ptot > 0 ? (double) p_hits / (double) ptot : 0.0, (long long) p_hits, (long long) ptot,
                      (long long) p_fills, (long long) p_evict);
    }
    GGML_LOG_INFO("%s: takeover=%lld decline_all=%lld get_ok=%lld cold=%s cold_reaches=%lld\n",
                  __func__, (long long) g_takeover, (long long) g_decline_all, (long long) g_get_ok,
                  g_cold_uva ? "uva" : "off", (long long) g_cold_reaches);
    if (g_cold_uva) {
        static const char * const admit_names[] = { "always", "value", "touch" };
        GGML_LOG_INFO("%s: cold admission=%s, admission-rejected (served cold)=%lld\n",
                      __func__, admit_names[g_admit < 0 || g_admit > 2 ? 0 : g_admit], (long long) g_cold_admits);
    }
    for (int d = 0; d < GGML_CUDA_MAX_DEVICES; d++) {
        if (dev_tables[d] > 0) {
            GGML_LOG_INFO("%s: device %d: tables=%d slots/table(sum)=%lld arena=%.1f MiB (budget %s MiB/device)\n",
                          __func__, d, dev_tables[d], (long long) dev_slots[d],
                          (double) dev_arena[d] / (1024 * 1024), budget_desc());
            if (!g_auto && dev_arena[d] > (int64_t) g_budget) {
                GGML_LOG_WARN("%s: device %d arena %.1f MiB exceeds the %.1f MiB per-device budget "
                              "(an explicit MOE_EXPERT_CACHE_SLOTS override, or per-table expert_bytes varying)\n",
                              __func__, d, (double) dev_arena[d] / (1024 * 1024),
                              (double) g_budget / (1024 * 1024));
            }
        }
    }
    GGML_LOG_INFO("%s: VRAM budget: requested=%s MiB/device, free-clamped=%lld, arena alloc failures=%lld (fail-soft)\n",
                  __func__, budget_desc(), (long long) g_alloc_clamped, (long long) g_alloc_failed);    if (g_acc_hits + g_acc_fills + g_acc_colds > 0) {
        const int64_t acc   = g_acc_hits + g_acc_fills + g_acc_colds;
        const int64_t bytes = g_fill_bytes + g_cold_bytes;
        const int64_t tokens = (n_tables > 0 && g_promote_calls > 0) ? g_promote_calls / n_tables : 0;
        GGML_LOG_INFO("%s: remap-kernel launches=%lld (~%.1f/token)\n",
                      __func__, (long long) g_remap_launches,
                      tokens > 0 ? (double) g_remap_launches / (double) tokens : 0.0);
        GGML_LOG_INFO("%s: expert access: hits=%lld fills=%lld colds=%lld (hit=%.2f%% fill=%.2f%% cold=%.2f%%)\n",
                      __func__, (long long) g_acc_hits, (long long) g_acc_fills, (long long) g_acc_colds,
                      100.0 * (double) g_acc_hits / (double) acc, 100.0 * (double) g_acc_fills / (double) acc,
                      100.0 * (double) g_acc_colds / (double) acc);
        GGML_LOG_INFO("%s: expert H2D traffic: fills=%.1f MiB cold-reads=%.1f MiB total=%.1f MiB over ~%lld tokens = %.2f MiB/token\n",
                      __func__, (double) g_fill_bytes / (1024 * 1024), (double) g_cold_bytes / (1024 * 1024),
                      (double) bytes / (1024 * 1024), (long long) tokens,
                      tokens > 0 ? (double) bytes / (1024 * 1024) / (double) tokens : 0.0);
        if (tokens > 0) {
            GGML_LOG_INFO("%s: expert traffic per token: %.1f MiB/token = %.0f KiB/token (fills %.0f, cold %.0f)\n",
                          __func__, (double) bytes / 1048576.0 / (double) tokens,
                          (double) bytes / 1024.0 / (double) tokens,
                          (double) g_fill_bytes / 1024.0 / (double) tokens,
                          (double) g_cold_bytes / 1024.0 / (double) tokens);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// self-test: synthetic table, known pattern, fill + byte-verify.  Does not need a model.
// ---------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------
// Device-policy vs host-policy parity self-test (item A1).
//
// Byte-identity cannot catch an admission-policy divergence (a wrong victim still computes correct
// output), so this replays a deterministic synthetic routing sequence through BOTH `access_locked` and
// `moe_cache_policy_kernel` and compares the resulting residency, counters, occupancy and clock.  One
// scenario per call; `prefill` seeds experts 0..slots-1 as provisional (the newest logic).
// ---------------------------------------------------------------------------------------------
static bool devpolicy_selftest_case(const char * name, bool prefill) {
    const int    n_experts = 64;
    const int    slots     = 16;
    const int    n_used    = 8;
    const int    steps     = 512;
    const int    threads   = 256;
    const size_t eb        = 256;

    int prev = 0;
    (void) cudaGetDevice(&prev);
    if (g_device >= 0) { (void) cudaSetDevice(g_device); }

    std::vector<uint8_t> host((size_t) n_experts * eb);
    for (int e = 0; e < n_experts; e++) {
        for (size_t i = 0; i < eb; i++) {
            host[(size_t) e * eb + i] = (uint8_t) (e * 13 + i);
        }
    }
    void * arena = nullptr;
    if (cudaMalloc(&arena, (size_t) slots * eb) != cudaSuccess) {
        (void) cudaGetLastError();
        if (g_device >= 0) { (void) cudaSetDevice(prev); }
        return false;
    }

    table_t t;
    t.layer = -1; t.role = "selftest"; t.n_experts = n_experts; t.expert_bytes = eb; t.host_bytes = eb;
    t.slots = slots; t.allocated = true;
    t.host = host.data(); t.host_dev = host.data(); t.arena = arena; t.device = g_device; t.cold_safe = true;
    t.identity = false; t.policy = true;
    t.slot_expert.assign(slots, -1); t.slot_prov.assign(slots, 0);
    t.count.assign(n_experts, 0); t.ghost.assign(n_experts, 0); t.last.assign(n_experts, 0);
    if (prefill) {
        for (int i = 0; i < slots; i++) { t.slot_expert[i] = i; t.expert_slot[i] = i; t.slot_prov[i] = 1; }
    }

    int32_t * d_slot_expert = nullptr, * d_slot = nullptr, * d_count = nullptr, * d_ghost = nullptr;
    int32_t * d_last = nullptr, * d_used = nullptr, * d_shape = nullptr;
    uint8_t * d_prov = nullptr;
    moe_cache_policy_desc * d_desc = nullptr;
    auto alloc = [](void ** p, size_t bytes) { return cudaMalloc(p, bytes) == cudaSuccess; };
    bool aok = alloc((void **) &d_slot_expert, (size_t) slots * 4) && alloc((void **) &d_slot, (size_t) n_experts * 4) &&
               alloc((void **) &d_count, (size_t) n_experts * 4) && alloc((void **) &d_ghost, (size_t) n_experts * 4) &&
               alloc((void **) &d_last, (size_t) n_experts * 4) && alloc((void **) &d_used, (size_t) n_used * 4) &&
               alloc((void **) &d_shape, 4) && alloc((void **) &d_prov, (size_t) slots) &&
               alloc((void **) &d_desc, sizeof(moe_cache_policy_desc));
    if (!aok) {
        (void) cudaGetLastError();
        (void) cudaFree(arena);
        if (g_device >= 0) { (void) cudaSetDevice(prev); }
        return false;
    }

    {
        std::vector<int32_t> se((size_t) slots, -1), sm((size_t) n_experts, -1);
        for (int s = 0; s < slots; s++) { se[(size_t) s] = t.slot_expert[s]; }
        for (const auto & kv : t.expert_slot) { sm[(size_t) kv.first] = kv.second; }
        (void) cudaMemcpy(d_slot_expert, se.data(), (size_t) slots * 4, cudaMemcpyHostToDevice);
        (void) cudaMemcpy(d_slot, sm.data(), (size_t) n_experts * 4, cudaMemcpyHostToDevice);
        (void) cudaMemcpy(d_count, t.count.data(), (size_t) n_experts * 4, cudaMemcpyHostToDevice);
        (void) cudaMemcpy(d_ghost, t.ghost.data(), (size_t) n_experts * 4, cudaMemcpyHostToDevice);
        (void) cudaMemcpy(d_last, t.last.data(), (size_t) n_experts * 4, cudaMemcpyHostToDevice);
        (void) cudaMemcpy(d_prov, t.slot_prov.data(), (size_t) slots, cudaMemcpyHostToDevice);
        const int32_t shape0 = n_used;
        (void) cudaMemcpy(d_shape, &shape0, 4, cudaMemcpyHostToDevice);
    }
    moe_cache_policy_desc desc;
    memset(&desc, 0, sizeof(desc));
    desc.slot_expert = d_slot_expert; desc.slot = d_slot; desc.count = d_count; desc.ghost = d_ghost;
    desc.last = d_last; desc.slot_prov = d_prov; desc.used = d_used;
    desc.arena = arena; desc.host_dev = host.data(); desc.host_bytes = eb; desc.src_off = 0;
    desc.expert_bytes = eb; desc.host_pitch = 0; desc.clock = 0; desc.last_decay = 0;
    desc.n_experts = n_experts; desc.slots = slots; desc.occupied = (int32_t) t.expert_slot.size();
    desc.split_axis = -1;
    (void) cudaMemcpy(d_desc, &desc, sizeof(desc), cudaMemcpyHostToDevice);

    const int admit_arg = g_admit;
    uint32_t rng = 0x12345678u;
    auto next = [&]() { rng = rng * 1664525u + 1013904223u; return rng; };
    std::vector<int32_t> used((size_t) n_used);
    for (int step = 0; step < steps; step++) {
        for (int j = 0; j < n_used; j++) {
            used[(size_t) j] = (next() % 100 < 60) ? (int32_t) (next() % 16) : (int32_t) (next() % n_experts);
        }
        for (int j = 0; j < n_used; j++) {
            bool cold = false;
            (void) access_locked(t, used[(size_t) j], nullptr, used.data(), n_used, &cold);
        }
        (void) cudaMemcpy(d_used, used.data(), (size_t) n_used * 4, cudaMemcpyHostToDevice);
        // do_fill=0: the parity test only checks the admission policy, and a kernel-side fill would read
        // the selftest's pageable host buffer over UVA (which can fault/hang).
        moe_cache_policy_kernel<<<1, threads, 0, 0>>>(d_desc, d_shape, 1, admit_arg, (int) g_touch,
                                                      (int) g_period, 0, 0, g_prov_evict ? 1 : 0);
    }
    (void) cudaDeviceSynchronize();

    std::vector<int32_t> dse((size_t) slots, -2), dsm((size_t) n_experts, -2), dc((size_t) n_experts, 0);
    std::vector<int32_t> dg((size_t) n_experts, 0);
    uint8_t dp_dummy = 0; (void) dp_dummy;
    moe_cache_policy_desc rd;
    memset(&rd, 0, sizeof(rd));
    (void) cudaMemcpy(dse.data(), d_slot_expert, (size_t) slots * 4, cudaMemcpyDeviceToHost);
    (void) cudaMemcpy(dsm.data(), d_slot, (size_t) n_experts * 4, cudaMemcpyDeviceToHost);
    (void) cudaMemcpy(dc.data(), d_count, (size_t) n_experts * 4, cudaMemcpyDeviceToHost);
    (void) cudaMemcpy(dg.data(), d_ghost, (size_t) n_experts * 4, cudaMemcpyDeviceToHost);
    (void) cudaMemcpy(&rd, d_desc, sizeof(rd), cudaMemcpyDeviceToHost);
    int mism = 0;
    for (int s = 0; s < slots; s++) { if (dse[(size_t) s] != t.slot_expert[s]) { mism++; } }
    for (int e = 0; e < n_experts; e++) {
        const int32_t hs = (t.expert_slot.count(e) != 0) ? t.expert_slot[e] : -1;
        if (dsm[(size_t) e] != hs || dc[(size_t) e] != (int32_t) t.count[(size_t) e] ||
            dg[(size_t) e] != (int32_t) t.ghost[(size_t) e]) { mism++; }
    }
    const bool ok = mism == 0 && rd.acc_hits == t.hits && rd.acc_misses == t.misses && rd.clock == t.clock &&
                    rd.last_decay == t.last_decay;
    GGML_LOG_WARN("%s: %s: %s slots=%d resident=%zu hits(host=%lld dev=%lld) misses(host=%lld dev=%lld) "
                  "clock(host=%lld dev=%lld) mismatches=%d\n",
                  __func__, ok ? "PASS" : "FAIL", name, slots, t.expert_slot.size(),
                  (long long) t.hits, (long long) rd.acc_hits, (long long) t.misses, (long long) rd.acc_misses,
                  (long long) t.clock, (long long) rd.clock, mism);

    (void) cudaFree(d_slot_expert); (void) cudaFree(d_slot); (void) cudaFree(d_count);
    (void) cudaFree(d_ghost); (void) cudaFree(d_last); (void) cudaFree(d_used);
    (void) cudaFree(d_shape); (void) cudaFree(d_prov); (void) cudaFree(d_desc); (void) cudaFree(arena);
    if (g_device >= 0) { (void) cudaSetDevice(prev); }
    return ok;
}

static void moe_cache_devpolicy_selftest() {
    if (!g_devpolicy) {
        GGML_LOG_WARN("%s: skipped (MOE_EXPERT_CACHE_DEVPOLICY off)\n", __func__);
        return;
    }
    device_guard dg(g_device);
    const bool a = devpolicy_selftest_case("empty", false);
    const bool b = devpolicy_selftest_case("provisional-prefill", true);
    GGML_LOG_WARN("%s: %s (empty=%s provisional=%s)\n", __func__, (a && b) ? "PASS" : "FAIL",
                  a ? "PASS" : "FAIL", b ? "PASS" : "FAIL");
}

void moe_cache_selftest() {
    if (!g_enabled) {
        GGML_LOG_INFO("%s: skipped (cache disabled)\n", __func__);
        return;
    }
    // Save/restore the current device around the test.
    int prev = 0;
    (void) cudaGetDevice(&prev);
    if (g_device >= 0) {
        (void) cudaSetDevice(g_device);
    }

    const int    n_experts = 16;
    const size_t expert_bytes = 4096;
    const int    slots = 4;

    std::vector<uint8_t> host((size_t) n_experts * expert_bytes);
    for (int e = 0; e < n_experts; e++) {
        for (size_t i = 0; i < expert_bytes; i++) {
            host[(size_t) e * expert_bytes + i] = (uint8_t) (e * 31 + i);
        }
    }

    // Register with an explicit slot override for the test (a distinct src0 key).
    const int slot_env_prev = g_slots_hint;
    g_slots_hint = slots;
    const int table = moe_cache_table((const void *) &host, /*layer*/ -1, "selftest", n_experts, expert_bytes,
                                      expert_bytes, 0, 0, -1, host.data(), g_device);
    g_slots_hint = slot_env_prev;

    bool ok = table >= 0;
    if (!ok) {
        GGML_LOG_ERROR("%s: FAIL: table registration returned %d\n", __func__, table);
        if (g_device >= 0) { (void) cudaSetDevice(prev); }
        return;
    }

    // Access a classic scan pattern and then verify the resident slots byte-for-byte.
    {
        const bool fill_env_prev = g_fill;
        g_fill = true;   // the self-test exists to exercise the fill + byte-verify
        std::lock_guard<std::mutex> lock(g_mutex);
        table_t & t = g_tables[table];
        const int pattern[] = { 0, 1, 2, 3, 4, 5, 4, 5, 6, 7, 6, 0, 1, 2, 3, 4 };
        for (int e : pattern) {
            (void) access_locked(t, e, nullptr);
        }
        (void) cudaDeviceSynchronize();   // the fills are async
        std::vector<uint8_t> rd(expert_bytes);
        for (int s = 0; s < t.slots; s++) {
            const int32_t e = t.slot_expert[s];
            if (e < 0) { continue; }
            (void) cudaMemcpy(rd.data(), (char *) t.arena + (size_t) s * expert_bytes, expert_bytes, cudaMemcpyDeviceToHost);
            if (memcmp(rd.data(), host.data() + (size_t) e * expert_bytes, expert_bytes) != 0) {
                GGML_LOG_ERROR("%s: FAIL: slot %d expert %d mismatch\n", __func__, s, e);
                ok = false;
            }
        }
        GGML_LOG_INFO("%s: %s: table slots=%d resident=%zu hits=%lld misses=%lld fills=%lld evictions=%lld\n",
                      __func__, ok ? "PASS" : "FAIL", t.slots, t.expert_slot.size(),
                      (long long) t.hits, (long long) t.misses, (long long) t.fills, (long long) t.evictions);
        g_fill = fill_env_prev;
    }

    // item A1: device-policy vs host-policy parity (only when the device policy is enabled).
    moe_cache_devpolicy_selftest();

    if (g_device >= 0) {
        (void) cudaSetDevice(prev);
    }
}
