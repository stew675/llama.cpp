// wip/moe-expert-cache — Phase 1a foundation.  See moe-expert-cache.h for the contract.

#include "moe-expert-cache.h"
#include "ggml-cuda-vmm.h"

#include "common.cuh"
#include "mmvq.cuh"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdint.h>
#if !defined(_WIN32)
#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#else
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>   // the pool's disk tier: CreateFileA/ReadFile(OVERLAPPED)/FILE_FLAG_NO_BUFFERING
#include <malloc.h>    // _aligned_malloc/_aligned_free (the aligned bounce the unbuffered reads need)
#endif
#include <set>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// int64_t min/max without `std::min`/`std::max` (issue #217).  With HIP on Windows (clang + the MSVC STL)
// clang's force-included `cuda_wrappers/algorithm` and the MSVC STL `<utility>` each define an inline
// template `std::min<long long>` (int64_t is `__int64` there, so the mangled name is the same -- the
// `enable_if` that makes clang's version win overload resolution is not part of the mangling), and a
// translation unit that instantiates both fails to codegen with "definition with same mangled name".
static inline int64_t moe_cache_min_i64(int64_t a, int64_t b) { return a < b ? a : b; }
static inline int64_t moe_cache_max_i64(int64_t a, int64_t b) { return a > b ? a : b; }

// ---------------------------------------------------------------------------------------------
// state
// ---------------------------------------------------------------------------------------------


// The kernel stages a token's admitted experts in shared memory before the block cooperatively copies them
// into the arena.  A table can see up to `n_used * n_tok` routed experts in one pass (16 x 16 = 256 at the
// widened band); the admission loop stops admitting once the list is full, so an expert is never marked
// resident without its fill (it stays cold and is served from the host alias instead).
#define MOE_CACHE_POLICY_MAX_FILL 256

// wip/moe-heatmap-persistence: defined after the anonymous namespace (it is used by the global
// `moe_cache_get_stats`/`moe_cache_report`), but called from inside it too.
static void heat_pull_locked();
#if defined(GGML_USE_HIP) && defined(_WIN32)
int64_t ggml_hip_win_available_ram_bytes();   // GlobalMemoryStatusEx (wddm-mem.cu)
// The adapter's SHARED (aperture) segments, all processes: what a pinned host allocation is charged
// against on Windows, and NOT the same figure as `ggml_hip_win_available_ram_bytes()` (issue #274).
bool    ggml_hip_win_shared_mem_info(int64_t * limit_mib, int64_t * resident_mib);
// The loaded HIP runtime module (`amdhip64_7.dll`) and whether it is the Windows driver's System32 copy
// rather than the ROCm/TheRock runtime shipped beside the binary (issue #278, r233).
bool    ggml_hip_runtime_module_info(char * path_out, size_t path_cap, bool * system_dir_out, int * version_out);
#endif

namespace {

// A pool file descriptor: an `int` on POSIX and a pointer-sized `HANDLE` on Windows, where `int` is 32-bit
// and would truncate it.  Declared here (not with the I/O helpers below) because the table and pool structs
// carry one.
#if defined(_WIN32)
typedef intptr_t pool_fd_t;
#else
typedef int      pool_fd_t;
#endif

// wip/host-expert-tier: the loader opens the persisted prior BEFORE it places any host-expert tensor (see
// `moe_cache_heat_open` at the bottom of this file), which is earlier than the first
// `moe_cache_set_host_source` call.  The definition is in this anonymous namespace, so the declaration has
// to be too (a file-scope `static` declaration of the same name is a DIFFERENT entity, and every call
// becomes ambiguous).
static void heat_map_open_locked(const char * model_path);

// wip/host-expert-master-drop (TODO #55): one pinned, one-host-expert staging entry for a PAGEABLE master
// (see `expert_host_src`).  `busy` means a device copy is reading it, so it must not be rewritten until
// `ev` signals.
struct bounce_t {
    char *      host = nullptr;
    cudaEvent_t ev   = nullptr;
    bool        busy = false;
    uint64_t    seq  = 0;   // acquisition order (the bounded ring waits on the oldest entry)
};

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
    // wip/fix-gemma-4 (TODO #77/C): a FUSED table under `-sm tensor` (gemma4 `ffn_gate_up_exps`, whose
    // device slot is `[gate_slice | up_slice]`).  The slot is assembled by concatenating these host
    // ranges in order; `n_slices == 0` means the historical single slice (`src_off`/`host_pitch`).
    int         n_slices    = 0;
    struct { size_t src_off; size_t len; } slice[GGML_BACKEND_MOE_MAX_SLICES];
    // wip/host-expert-dio-cache Phase 1: the on-disk source of this table's host master, registered by
    // the loader (`moe_cache_set_host_source`).  Empty unless `MOE_HOST_POOL_MIB > 0`.
    std::string src_path;
    size_t      src_offs    = 0;
    bool        src_known   = false;
    // wip/host-expert-master-drop (TODO #55): the file the pool-backed master came from, opened lazily for
    // the bounce path in `MOE_HOST_POOL_DIO` mode (the only mode where the bounce reads the disk instead of
    // the mapped page cache).
    pool_fd_t   src_fd      = -1;
    size_t      src_align   = 4096;
    const void * host       = nullptr;
    void *      host_dev    = nullptr;  // device-accessible alias of `host` (UVA), Phase 1b cold reads
    bool        host_dev_bound = false; // the alias was resolved (or ruled out) once
    // TODO #271: this table's PREFILL STAGING decline has been recorded (once per table, for the exit
    // census).  A declined table's whole-table upload stays on the scheduler's own path, so it is not
    // staged and not counted in the staging census -- which is exactly how 50 of 144 tables stayed
    // invisible on a pinned master.  See `stage_decline_note`.
    bool        stage_declined = false;
    bool        host_pinned = false;    // `host` is pinned (cudaHostGetDevicePointer succeeded): the in-place reads need it
    // wip/host-expert-master-drop (TODO #55): pinned per-table bounce ring for a PAGEABLE master, grown on
    // demand (see `expert_host_src`).  Empty for a pinned master.
    std::vector<bounce_t> bounce;
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
    // issue #274: the once-per-table report of a FAILED fill (a failure used to be silent apart from a
    // per-occurrence warn, and the slot stayed resident).
    bool        fill_fail_warned = false;
    // A table whose fills have FAILED is not servable: the eager take-over and the consumer must decline it
    // wholesale from then on (the same policy `alloc_failed`/`!moe_cache_has_arena()` already use), because a
    // take-over that still emitted a remap would skip the scheduler's `input_cpy` copy AND the arena holds
    // nothing -- measured as corrupt greedy text on the issue-#274 forced-failure arm.  One failed fill is
    // enough: a source that failed once (no pool, no bounce, no mapping) does not come back.
    bool        fill_failed  = false;
    // Diagnostic counter for `MOE_HOST_POOL_FILL_FAULT=late` (see `pool_fill_fault`): how many times THIS
    // table's demand path asked for a source, so the fault can be "this table is already warm".
    int32_t     fill_src_calls = 0;
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
    // Sync-free decode device remap (GGML_MOE_DEVREMAP): expert -> slot map (slots + e = cold) on the device, its
    // pinned staging, and the previous call's routing written back by the remap kernel into mapped pinned memory.
    int         dr_device   = -1;
    int64_t     dr_cap      = 0;
    int32_t *   dr_map_dev  = nullptr;
    int32_t *   dr_map_host = nullptr;
    int32_t *   dr_lag_host = nullptr;
    int32_t *   dr_lag_dev  = nullptr;
    cudaEvent_t dr_map_ev   = nullptr;
    cudaEvent_t dr_lag_ev   = nullptr;
    int64_t     dr_lag_n    = 0;
    int32_t     dr_lag_seq  = 0;
    bool        dr_lag_flag = false;
    bool        dr_map_ok   = false;
    bool        dr_failed   = false;
    // TODO #269: this table's cold source at sizing was a host pool that COVERS it (every expert has a
    // resident pool slot), so the remap's cold ids can be `slots + pool_slot`.  Decided once, at sizing:
    // it is a per-shape constant for a graph capture, and (see `pool_rerank_from_heat_locked`) a covering
    // pool's layout is write-once, so a cold id stays valid for the life of the run.
    bool        dr_pool_ok  = false;
    const void * dr_last_ids    = nullptr;   // routing buffer and stream of the last take-over: a sibling joins a launch
    void *       dr_last_stream = nullptr;   // only when both match, so a launch never touches another context's table

    // Prefill residency snapshot (PRS): a small device copy of
    // `expert_slot` (-1 = not resident) consumed ONLY by the prefill staging kernel.  It is rebuilt from
    // the authoritative host map (immutable while in use), never mutated by the decode path, and is not
    // the removed device-remap policy.  Rebuilt when the cache write generation moves, the arena moves,
    // or the slot count changes (a slab stand-down/re-arm invalidates it).
    int32_t *   prs_dev     = nullptr;
    int64_t     prs_n       = 0;        // allocated elements
    uint64_t    prs_gen     = 0;        // `g_write_seq` the current snapshot reflects
    void *      prs_arena   = nullptr;  // arena base the snapshot was built against
    int         prs_slots   = 0;        // slot count the snapshot was built against
    std::vector<int32_t> prs_host;      // persistent staging (the async H2D must not race its source)

    // Gather residency snapshot (wip/host-pool-speed, mechanism (b)): the same two-tier encoding as `prs`
    // (`>= 0` arena slot, `<= -2` pool slot `-v-2`, `-1` neither), but refreshed on the COMPUTE stream so
    // the device gather can resolve an expert from the arena or the pool instead of only a device-readable
    // master.  Deliberately separate from `prs_*`: staging rebuilds that one on the copy stream, and one
    // buffer written from two streams is a race.  `gprs_last` is what the device copy currently holds, so
    // an unchanged map costs no H2D at all.
    int32_t *            gprs_dev  = nullptr;
    int64_t              gprs_n    = 0;
    void *               gprs_arena = nullptr;
    int                  gprs_slots = 0;
    std::vector<int32_t> gprs_host;
    std::vector<int32_t> gprs_last;

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

    // wip/moe-heatmap-persistence (R2): long-horizon per-expert heat, the persistence source.
    // Fed on the DECODE band by the eager path (`moe_cache_update_host`, host ids) and on the PREFILL
    // band by the device tally (`moe_cache_heat_observe`, so the staged path and the identity path
    // are covered).  NOT decayed (unlike `count`/`ghost`) - it is a workload prior, not eviction state.
    std::vector<uint32_t> heat;             // [n_experts], saturating
    // TODO #88: the DECODE-band part of `heat`.  `heat` is one counter fed by both bands, and a prefill
    // ubatch routes ~n_tok experts, so on a long prompt the prefill outweighs the decode by tens of times
    // and the ranking degenerates toward "nearly every expert is hot".  The ARENA serves the
    // decode/verify band only, so its seed is ranked by THIS vector (`heat_rank_vec`); the host pool
    // stages whole tables for prefill and keeps using the combined one.  Measured on a real trace: a
    // decode-only prior covers 95.3 % of the decode mass where the combined one covers 80.3 %.
    std::vector<uint32_t> heat_dec;         // [n_experts], saturating
    // The part of `heat` already folded into `g_heat_loaded`.  The persisted prior is loaded *into*
    // `heat` at registration, so without this the save's merge re-added the whole running total at every
    // flush: the map doubled per flush, saturated at 0xffffffff within ~6 flushes of a server session,
    // and every heat-driven ranking silently degenerated to expert-id order (measured 2026-10-17).
    std::vector<uint32_t> heat_merged;      // [n_experts], saturating
    std::vector<uint32_t> heat_dec_merged;  // [n_experts], saturating (TODO #88)
    uint32_t *  heat_dev  = nullptr;        // device mirror for the prefill tally (lazy, fail-soft)
    uint32_t *  heat_dec_dev = nullptr;     // device mirror for the sampled identity DECODE tally (TODO #88)
    int64_t     heat_dev_cap = 0;           // capacity in elements
};

// TODO #88: the band a heat-driven arena decision ranks by (defined with the heat-map code below).
static const std::vector<uint32_t> & heat_rank_vec(const table_t & t);

std::mutex                          g_mutex;
std::vector<table_t>                g_tables;
// Stable registration key: (host master weight, device).  It must not be the tensor the op reads
// (`weight_cpy`): the graph allocator reuses those pointers across layers, so the consumer alias map
// below is re-pointed on every hook call while registration stays fixed.
std::map<std::pair<const void*, int>, int> g_key_to_id;
// Consumer alias: the tensor the op actually reads (the redirected `input_cpy`, or a per-device simple
// tensor under `-sm tensor`) -> table.  Refreshed each hook call.
std::unordered_map<const void*, int>       g_alias_to_id;
static std::map<std::pair<const void *, int>, int> g_dr_key_to_id;   // (host master, device) -> table (device remap)
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
    char *      base        = nullptr;   // cudaHostAlloc(portable|mapped), slots * host_bytes
    void *      base_dev    = nullptr;   // UVA device alias of `base` (== base under UVA); the cross-device
                                         // fill source (a plain H2D from `base` on another ROCm context is
                                         // treated as pageable and crashes in the runtime's staging memmove)
    int         slots       = 0;
    size_t      host_bytes  = 0;
    int         n_experts   = 0;
    std::string path;                    // GGUF path
    size_t      file_offs   = 0;         // tensor data offset in the file
    pool_fd_t   fd          = -1;
    // wip/host-expert-master-drop (TODO #55): the host tensor this pool belongs to when it is backed by
    // the read-only model mapping.  A miss is then a plain memcpy out of the page cache (no syscall, no
    // second copy) instead of a pread; `MOE_HOST_POOL_DIO=1` still reads the file with O_DIRECT, which is
    // the only mode that bypasses the page cache.  It is the TENSOR BASE (expert `e` at `+ e*host_bytes`),
    // unlike `file_offs`, which is the tensor's offset in the GGUF.
    const void * src_host   = nullptr;
    size_t      align       = 4096;
    std::vector<int32_t>                 slot_expert;   // slot -> expert, -1 empty
    std::unordered_map<int32_t, int32_t> expert_slot;
    // Lifecycle per slot (TODO #87): 0 = free (never filled, or handed over and retired),
    // 1 = resident, 2 = retiring (a bulk seed copied it into the arena wholesale, so it is freed once every
    // in-flight copy drains).  There is no reserved/background state: the demand path fills and hands over
    // inline, never off the critical path.
    std::vector<uint8_t>                 slot_state;
    // wip/host-expert-split W2: 1 = this slot's content was DEMOTED from the arena (not a promotion or a
    // bulk-layout placement), so a run can answer the question that decides whether the demote pays: how
    // often is a demoted victim used again?  `copies` alone cannot -- a demote never read back is pure waste.
    std::vector<uint8_t>                 slot_demoted;
    // Per (slot, device) in-flight events.  The pool is PROCESS-WIDE, so the same slot can be copied to
    // several devices and every copy must drain before the slot is reused or retired.  Indexed
    // `slot * GGML_CUDA_MAX_DEVICES + device`, nullptr = never used; created lazily on the recording
    // device (a single event per slot + a cross-device record/sync fails with `invalid resource handle`).
    std::vector<cudaEvent_t>             slot_ev;
    std::vector<int64_t>                 last;          // per expert, last use
    int64_t     clock       = 0;
    int64_t     hits        = 0;
    int64_t     misses      = 0;
    int64_t     fills       = 0;
    int64_t     evictions   = 0;
    bool        failed      = false;
    // r215: nonzero = the backing is an `mmap`+`MADV_HUGEPAGE` region pinned with `cudaHostRegister`
    // (`MOE_HOST_POOL_ALLOC=thp`, the default), released with `cudaHostUnregister`+`munmap` rather than
    // `cudaFreeHost`.  Zero = the historical one-shot `cudaHostAlloc`/`cudaMallocHost`.
    size_t      mmap_bytes  = 0;
    bool        registered  = false;     // the mmap region above has been pinned
};
// Keyed by the host tensor only: the pool is PROCESS-WIDE (one copy of each whole expert for every
// device), and `MOE_HOST_POOL_MIB` is the total non-swappable budget, split equally over the registered
// host tensors.  Per-(slot, device) events make a shared slot safe to copy to several devices.
// Deliberately LEAKED (a heap object with no destructor): the pool is filled on the critical path, but a
// process-lifetime container that other threads may still be walking while the runtime tears down has
// bitten this code before (`double free or corruption` in `pool_reap_locked` at exit, once the pool became
// the ONLY host source in wip/host-expert-master-drop).  Leaking it costs one map.
std::map<const void *, host_pool_t> & g_host_pools = *new std::map<const void *, host_pool_t>();   // node-based: pointers stay valid
bool    g_pool_enabled     = false;
// wip/host-expert-pool-region (TODO #139): an explicit PROCESS-WIDE budget for the pinned pool, in bytes
// (-1 = no override; the MOE_HOST_POOL_* knobs decide).  The loader sets it from `-lme N` when N is below the
// host expert set: the same non-swappable budget, spent per EXPERT (residency) instead of per TENSOR.
int64_t g_pool_budget_override = -1;
bool    g_pool_requested   = false;  // --host-experts pool (set by the loader before tables are registered)
bool    g_pool_auto        = false;  // MOE_HOST_POOL_MIB unset/auto: size from the MoE host expert bytes
// r112: the defaults are tuned for the case POOL mode exists for -- a host expert set that does NOT fit in
// RAM, where the pool is the working set and the rest must come from disk.  80 % keeps most of the set
// hot in pinned RAM (the disk only serves the remainder, measured the knee: 80 % at 74.4 t/s / zero swap vs
// 90 % at 71.8 t/s / 0.3 GiB swap on a 30 GiB gfx1100 box) and, since r114, the FILL MODE is decided per box
// (buffered when the set fits beside the pool, O_DIRECT when it does not) instead of being pinned to DIO=1.
// On a box where the set fits in RAM, `--host-experts pinned` (the default mode) is still faster and
// cheaper than either.
int     g_pool_frac        = 80;     // MOE_HOST_POOL_FRAC: pool size as a % of the MoE host expert bytes
// wip/host-expert-split (2026-11-05): which persisted counter ranks the pool's CONTENT (the arena's
// complement).  `pool_rank_experts_locked` documents the measurement; 0 = the shipped combined counter.
int     g_pool_rank        = 0;      // MOE_HOST_POOL_RANK: 0 combined | 1 decode | 2 floor+decode
int     g_pool_rank_floor  = 50;     // MOE_HOST_POOL_RANK_FLOOR: the prefill-hot % kept ahead (mode 2)
int     g_pool_ram_pct     = 85;     // MOE_HOST_POOL_RAM_PCT: ceiling on the WHOLE pinned pool, as a % of
                                     //   MemAvailable at the first pool build (0 = no ceiling).  The FRAC is
                                     //   a percentage of the model's host expert set, which says nothing about
                                     //   whether the box can PIN that much: on a tight box an oversized pool
                                     //   makes each table's cudaHostAlloc fail and silently falls back to the
                                     //   per-table bounce (disk reads), so bound it by real RAM instead.
                                     //   85 % (r114, was 80) is the measured knee: on the 30 GiB gfx1100
                                     //   field box the pinned set is 32.6 GiB, so the FRAC wants more than the
                                     //   ceiling allows; 85 % of MemAvailable pins ~2 GiB more of the set and
                                     //   stays clear of swap (see ENVIRONMENT.md).
int64_t g_pool_ram_cap     = -1;     // bytes; -1 = not computed yet
int64_t g_pool_ram_used    = 0;      // bytes pinned by the pools built so far
// wip/host-pool-speed W3: the ceiling's DISTRIBUTION.  The clamp used to shave the REGISTRATION-ORDER TAIL
// down to the one-slot floor, which starved the last ~14 tables of a 144-table model at RAM_PCT 70 and cost
// 40 % of decode for 10 % fewer slots (README §14.6).  Instead the planned total is compared with the
// ceiling ONCE and every table's slot count is scaled by the same factor, so a tightened ceiling now costs
// coverage everywhere instead of killing a few tables.  -1 = the scale has not been decided yet.
double  g_pool_ram_scale   = -1.0;
int64_t g_preflight_host_bytes = 0;  // Σ host expert bytes over host-expert devices (the AUTO basis)
bool    g_pool_prewarm     = true;   // MOE_HOST_POOL_PREWARM: fill every pool slot when it is built
// TODO #87 (1): `MOE_HOST_POOL_DIO` is AUTO by default -- decided once, at the first pool build, from
// MemAvailable vs the host expert set (the same measurement `MOE_HOST_POOL_RAM_PCT` takes).  Buffered fills
// (DIO=0) let the page cache act as an L3, which is worth +90 % decode / +155 % prefill when the set fits;
// O_DIRECT (DIO=1) is right when it does not, because a page cache that cannot hold the set just competes
// with the pinned pool for the same RAM.  `MOE_HOST_POOL_DIO=0|1` forces either arm (and keeps the A/B
// reproducible); the decision and its inputs are logged.
bool    g_pool_dio         = false;  // the EFFECTIVE mode (forced, or decided by `pool_decide_dio_locked`)
int     g_pool_dio_req     = -1;      // MOE_HOST_POOL_DIO: -1 = auto, 0 = force buffered, 1 = force O_DIRECT
bool    g_pool_dio_decided = false;
// r212: the loader's resolved `--load-mode` as a fill mode: 1 = O_DIRECT, 0 = mapped/page-cache,
// -1 = nobody told us (a consumer that is not the CLI loader -> the historical heuristic).
int     g_pool_load_mode    = -1;
bool    g_pool_stage_batch = true;   // MOE_HOST_POOL_STAGE_BATCH=0: one cudaMemcpyAsync per expert for the
                                     //   pool-backed prefill staging (the pre-r114 arm; A/B only)
// wip/host-expert-split W4 (prefill read granularity): the experts of a run that is in NEITHER tier are
// contiguous in the file, so their bytes are ONE read instead of one per expert.  A single 0.6 MiB O_DIRECT
// read is latency-bound on this box (measured 520 MiB/s at depth 1 against 4.2 GiB/s for a 16 MiB read on the
// same drive), and a prefill stages the whole host set per ubatch, so read SIZE -- not the drive -- is the
// ceiling.  `MOE_HOST_POOL_STAGE_SPAN=0` is the kill switch (the pre-W4 per-expert arm) and
// `MOE_HOST_POOL_STAGE_SPAN_EXPERTS` is the chunk size (16 experts ~= 10.9 MiB on the field config).
bool    g_pool_stage_span  = true;   // MOE_HOST_POOL_STAGE_SPAN=0: kill switch for the coalesced read
int     g_pool_stage_span_n = 16;    // MOE_HOST_POOL_STAGE_SPAN_EXPERTS: experts per coalesced read
// The shortest run worth a span read is a CODE CONSTANT, not a knob: it is a property of the read-latency /
// copy-ordering geometry, it is load-bearing (measured at -lme 36: 1 -> 362 t/s, 2 -> 379, 4 -> 331), and a
// deployment that got it wrong would be silently 12 % slower with no symptom to point at.  2 experts is where a
// span stops reading about as many bytes as the single expert it replaces and starts amortising the ring
// bookkeeping; below it the per-expert path is the same shape and pays nothing.
static const int MOE_STAGE_SPAN_MIN = 2;
int64_t g_pool_total_bytes = 0;   // PROCESS-WIDE total budget (split over all registered host tensors)
// wip/host-pool-speed: per-table pool SIZING.  `MOE_HOST_POOL_FRAC` gives every host table the same slot count
// (`n_experts * FRAC/100`), so the pinned bytes go to a table with a long cold tail exactly as to one with a flat
// hot set.  What differs between tables is not their routed mass (a top-k router hands every table `n_used`
// experts per token, so the total is a constant) but its CONCENTRATION: `cov` sizing asks for the slots a table
// needs to hold `MOE_HOST_POOL_COV_PCT` of its own prior mass, fitted to the same total byte budget.  Offline on
// the 122B-A10B trace at FRAC 80 that takes the disk tier from 48.7 to 28.5 MiB per decode token with `h`
// unchanged (`wip/host-pool-speed/tools/pool-sizing-sim.py`, HANDOVER §4/F6).  0 = uniform (default until gated).
int     g_pool_sizing      = 0;      // MOE_HOST_POOL_SIZING: 0 = uniform, 1 = cov
int     g_pool_cov_pct     = 90;     // MOE_HOST_POOL_COV_PCT: the mass fraction `cov` sizing targets
bool    g_pool_cov_fit     = false;  // the one-shot fit has run
std::map<const void *, int> g_pool_cov_slots;   // host tensor -> pool slots (the fit's result)
// wip/host-pool-speed W3: the byte total the fit actually planned.  The RAM ceiling is applied as a
// proportional scale over the SIZING's own plan, so it needs that plan's total as its denominator --
// using the uniform FRAC total instead would scale a ceiling-fitted cov plan a second time.
int64_t g_pool_cov_bytes = 0;

// An alias is keyed by a scheduler tensor address, and the graph allocator reuses those addresses across graphs - with
// several devices, for a split input on ANOTHER device.  A lookup that is not preceded by a registration in the same
// graph can therefore hit a stale alias: another layer's table, or another device's (its arena/counters are then read
// or written from the wrong GPU - a page fault or garbage, with -sm layer on 2 GPUs).
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
// Teardown snapshot of the report's STRUCTURAL aggregates (tables/slots/arena, hits/misses/fills/evictions and
// the host pool's tensors/slots/pinned bytes).  `moe_cache_report` is an `atexit` hook and the model teardown
// (`moe_cache_forget_all`) un-sizes the cache first, so without this every run ends with
// "tables=0, slots=0, arena=0.0 MiB" and "host pool (L2): 0 tensors, 0 slots, 0.0 MiB pinned" -- in BOTH modes,
// which reads as "the pool never activated" (it cost one full investigation, 2026-11-02).  The snapshot is taken
// at the first forget, when every device's tables are still registered, and the report prefers it when present.
struct moe_report_snapshot {
    bool    have      = false;
    int64_t hits = 0, misses = 0, fills = 0, evictions = 0;
    int     n_tables = 0, n_slots = 0;
    int64_t dev_slots [GGML_CUDA_MAX_DEVICES] = {0};
    int64_t dev_arena [GGML_CUDA_MAX_DEVICES] = {0};
    int     dev_tables[GGML_CUDA_MAX_DEVICES] = {0};
    int64_t arena_bytes = 0;
    int64_t n_pools = 0, p_slots = 0, p_hits = 0, p_misses = 0, p_fills = 0, p_evict = 0, p_bytes = 0;
};
static moe_report_snapshot g_report_snap;
// issue #274 follow-up (@DanoPTT): the exit census is printed ONCE, by whichever teardown reaches it first --
// `moe_cache_forget_all` (while the caller's log is still open) or the `atexit` hook (a run with no teardown).
// On Windows the atexit hook runs after `main` has returned and llama-server's async log is already gone, so
// the forget path is the one an operator actually sees.
static bool g_report_printed = false;
bool                                g_fill       = false;   // opt-in: copy the master into the slot
int                                 g_device     = -1;
int64_t                             g_period     = 32;   // LFRU decay period, decode steps
size_t                              g_budget     = 0;    // bytes (per device); SIZE_MAX in auto mode
bool                                g_auto       = false;  // MOE_EXPERT_CACHE_MIB unset: size each device from its free VRAM
int                                 g_unified    = 0;      // MOE_EXPERT_CACHE_UNIFIED: allow the arena on a shared host/device pool
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
// WIP moe-upload-overlap: the per-device host-expert bytes the preflight reported, so the deferred
// arena sizing can WAIT until every host table has registered.  Without this, a pass that primes only a
// SUBSET of the host tables latches the sizing on that subset -- e.g. a prefill's bulk ubatch is served
// by the H2D staging ring (which does not register cache tables) while the final 1-token ubatch reaches
// the decode hook and primes only the last host layer.  See `moe_cache_tables_complete_locked`.
int64_t                             g_preflight_device_bytes[GGML_CUDA_MAX_DEVICES] = {0};
bool                                g_preflight_device_seen [GGML_CUDA_MAX_DEVICES] = {false};
// The aux (draft/MTP) estimate the early auto floor last used.  The preflight now runs once at model
// load (aux 0, before the speculative params are known) and again from `common` with the real estimate;
// a LARGER estimate re-runs the floor instead of latching the load-time answer.
static size_t                       g_preflight_aux = 0;

// WIP moe-upload-overlap: a host table the cache can NEVER hold (an unaddressable split slice) must not
// keep the deferred sizing waiting for bytes it will never see.  Keyed on the host master + device, so
// the repeated presentations of one table subtract once.
static std::set<std::pair<const void *, int>> g_unregisterable_keys;
static void moe_cache_note_unregisterable(const ggml_tensor * weight, int device, int n_experts, size_t expert_bytes) {
    if (weight == nullptr || device < 0 || device >= GGML_CUDA_MAX_DEVICES || n_experts <= 0 || expert_bytes == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_unregisterable_keys.insert(std::make_pair((const void *) weight->data, device)).second) {
        return;   // already subtracted
    }
    const int64_t b = (int64_t) n_experts * (int64_t) expert_bytes;
    if (g_preflight_device_bytes[device] > b) {
        g_preflight_device_bytes[device] -= b;
    } else {
        g_preflight_device_bytes[device] = 0;
    }
}
// Bumped by every entry point that can write an arena or a device slot map (the decode-band hooks, sizing, re-arm,
// eviction).  The cache-aware staging orders its copy stream after the compute stream only when this moved since its
// last staging on that device: during a prefill nothing writes, so the staging keeps overlapping the compute.
std::atomic<uint64_t>               g_write_seq{1};
bool                                g_cold_uva        = false;  // MOE_EXPERT_CACHE_COLD=uva: serve rejected misses from the host alias
// Whole-table staging from the arena (PRS) + pinned host alias.  DEFAULT ON (kill switch
// `GGML_MOE_CACHE_STAGE=0`): the staged bytes are unchanged, so the output is bit-identical and `0`
// restores the plain whole-table upload.
bool                                g_stage_cache     = true;
// Prefill staging accounting (per call = per table per ubatch).
int64_t                             g_stage_ok        = 0;
// Gather accounting (wip/host-pool-speed mechanism (b)): `calls` = gather invocations that returned true,
// `pool` = of those, the ones served from a POOL-backed table (the new third source), `miss_fills` = routed
// experts that were in neither tier and were copied into the destination before the kernel.
int64_t                             g_uploadp_calls   = 0;   // moe_cache_upload_pageable serviced ranges
int64_t                             g_uploadp_bytes   = 0;   // ... and the bytes that took the bounce path
int64_t                             g_bounce_bytes    = 0;   // pinned ring bytes ever allocated (freed at teardown)
// Demand-fill cost accounting (README §16): wall time spent INSIDE the synchronous fill, split by source, and
// the bytes it moved.  Read against the run's token count to size the overlap/coalesce candidate.
int64_t                             g_fill_read_calls   = 0;   // fills that hit storage (O_DIRECT / pread)
int64_t                             g_fill_read_us      = 0;
int64_t                             g_fill_memcpy_calls = 0;   // fills served by a memcpy out of the mapping
int64_t                             g_fill_memcpy_us    = 0;
int64_t                             g_fill_total_bytes  = 0;   // (bytes; `g_fill_bytes` is the pre-existing traffic counter)
static uint64_t                     g_bounce_seq      = 0;
// issue #274 follow-up (@DanoPTT): the PINNED tier is latched OFF by the first refused `cudaHostAlloc`.  A
// pinned-allocation failure is not a transient on every platform: on Windows ROCm a `hipHostMalloc` refusal
// surfaces as `hipErrorLaunchFailure` and LATCHES process-wide, so every later HIP call (including the fill
// copies and `hipHostFree`) fails too -- one refusal then looks like a cascade of unrelated defects
// (ROCM-33542 / rocm-systems#14578, ROCm/TheRock#9339).  The bounce ring is the last pinned consumer, so its
// first refusal latches: every later source resolution takes the pageable fallback or declines (slow but
// correct) instead of hammering a driver that is no longer answering.  Nothing clears it: a source that
// failed once does not come back (the same policy as `table_t::fill_failed`).
static bool                         g_pinned_off       = false;
// Why the tier is closed, when it was closed by a run-time decision rather than a refused allocation
// (`MOE_HOST_POOL_PIN=0`, r233).  Null for a real refusal, whose cause is `g_pinned_alloc_err`.
static const char *                 g_pinned_off_why   = nullptr;
static cudaError_t                  g_pinned_alloc_err = cudaSuccess;
// How much pinned memory the pool/ring/span tiers had obtained when a refusal happened: the number that
// names the platform's real ceiling (on the reporter's box it stops at ~9.3 GiB of shared usage whether the
// machine has 47 or 64 GiB of RAM, which is why MemAvailable is not consulted for it).  Monotone: a freed
// ring entry is not subtracted, because the question is "how big did the tier get", not "what is live now".
static int64_t                      g_pinned_tier_bytes = 0;
// issue #278: expert fills served from a host pool that was built BEFORE the pinned tier latched off.  This
// is the number that says the latch stops the tier GROWING rather than discarding what it already owns: a row
// of zeros after a `the pinned host tier is OFF` line means every pool that existed was abandoned and every
// copy after that point went through the pageable model mapping (the shape a Windows box hit at its ~9.3 GiB
// adaptor ceiling), which is what the gate arm `pinlate` asserts against.
static int64_t                      g_pool_after_latch   = 0;
int64_t                             g_gather_calls    = 0;
int64_t                             g_gather_pool     = 0;
int64_t                             g_gather_fills    = 0;
int64_t                             g_stage_arena_bytes = 0;
int64_t                             g_stage_host_bytes  = 0;
int64_t                             g_stage_decline_geom = 0;
int64_t                             g_stage_decline_usable = 0;
// Split out of `unusable` (TODO #271) so the number keeps its meaning, "this table will never be staged":
// the priming pass declines by design (the arena does not exist yet), and counting it as unusable made a
// prompt-only run's very first ubatch look like a defect.
int64_t                             g_stage_decline_unsized = 0;
// The deduped per-table record of a staging decline (layer, role, reason), for the exit census.  One entry
// per affected TABLE, not per call: a declined table declines on every ubatch of the prefill.
struct stage_decline_t {
    int         layer = -1;
    std::string role;
    std::string why;
    int         device = -1;   // a tensor split registers one table per device; the reason is the same
};
std::vector<stage_decline_t>        g_stage_declined;
int64_t                             g_stage_decline_nomatch = 0;
// W4 span-read accounting (see `moe_cache_stage_table`): `span_reads` = storage reads issued for a run,
// `span_bytes` = the bytes they covered, `span_experts` = experts a span served, `span_fallback` = chunks
// whose span read came up short and were then served one expert at a time (a short read never fails the
// stage: the mapping is still a valid source, so this is a slowdown and not a correctness event).
int64_t                             g_stage_span_reads    = 0;
int64_t                             g_stage_span_bytes    = 0;
int64_t                             g_stage_span_experts  = 0;
int64_t                             g_stage_span_fallback = 0;
int64_t                             g_stage_span_read_us  = 0;   // time inside the span pread itself
int64_t                             g_stage_span_wait_us  = 0;   // time waiting for a ring buffer's copies
// The per-expert storage read that the span path exists to replace: every `expert_host_src` fill that had to
// go to the file (O_DIRECT) instead of a pool slot or the mapping.  Reported beside the pool fills so the
// prefill band's storage traffic can be read off a counter instead of inferred from a wall clock.
int64_t                             g_bounce_read_calls = 0;
int64_t                             g_bounce_read_us    = 0;
int64_t                             g_bounce_read_bytes = 0;
int64_t                             g_prs_rebuilds     = 0;   // immutable snapshot refreshes (churn indicator)
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
bool                                g_heat_on         = false; // MOE_EXPERT_HEATMAP=1 (WIP R2): tally the long-horizon per-expert heat

// wip/moe-heatmap-persistence: the on-disk heat map (sidecar next to the model, geometry-keyed).
// `g_heat_loaded` is keyed by (layer, sem_role, n_experts, host_bytes) and consulted at table
// registration; `g_heat_path` is the chosen sidecar (model dir, else the cache-dir fallback).
// TODO #88: each entry carries both bands -- `all` (prefill + decode, the pool's ranking) and `dec` (the
// decode band alone, the arena seed's ranking).  A version-1 file loads with `dec = all`, which is exactly
// the pre-r115 behaviour; a version-2 file carries the real split.
struct heat_prior_t {
    std::vector<uint32_t> all;
    std::vector<uint32_t> dec;
};
std::map<std::tuple<int, std::string, int, size_t>, heat_prior_t> g_heat_loaded;
std::string                         g_heat_path;
std::string                         g_heat_path_fallback;   // used when the model dir is read-only
bool                                g_heat_path_set   = false;
bool                                g_heat_write_ok   = false; // the chosen sidecar was probed writable
bool                                g_heat_frozen     = false; // load-only: the map is read-only, never overwritten
bool                                g_heat_dead       = false; // too many soft-fails: off for this session
int                                 g_heat_fails      = 0;
int                                 g_heat_max_fails  = 3;
int64_t                             g_heat_debounce_s = 60;
bool                                g_heat_dirty      = false;
std::chrono::steady_clock::time_point g_heat_last_save{};
// TODO #64.2: a FULLY RESIDENT (identity) table's decode routing is never read back, so the eager host
// tally never sees it.  Tally it from the device entry instead, SAMPLED - one in this many decode ops -
// and weight the sample by the period so the counts stay on the same scale as the exact tallies.
// 0 disables it.  `g_heat_op_tick` is the global decode-op counter the sampling rides on.
int                                 g_heat_decode_sample = 64;
std::atomic<int64_t>                g_heat_op_tick{0};
// TODO #64.1: set once the host-pool warm order has been re-derived from the prompt's prefill tally.
bool                                g_pool_rank_done  = false;
// Counts the prefill-band ops the device tally has seen, so the pool re-rank can tell "no prefill yet"
// (the warmup pass) from "the prompt has been tallied".
std::atomic<int64_t>                g_heat_prefill_ops{0};
// Decode-band `MUL_MAT_ID` census.  A routed expert table is touched once per (layer, role, device) per
// token, so `g_band_ops / n_tables` is the token count the traffic report divides by.  The previous
// estimate used `g_takeover / n_tables`, which reads 0 whenever the hook did not take the input over -
// i.e. on the fully-resident identity fast path and on any band the cache serves without a remap - and
// printed the unusable `over ~0 tokens = 0.00 MiB/token` the campaign needed (2026-10-26).
std::atomic<int64_t>                g_band_ops{0};
// The same census's denominator, snapshotted at sizing (see `alloc_all_locked`).
std::atomic<int64_t>                g_band_n_tables{0};

// ---------------------------------------------------------------------------------------------
// TODO #61 P1: the offline routing trace, `MOE_EXPERT_TRACE=<path>`.  Opt-in diagnostic (never
// default-on): it writes every `mul_mat_id`'s raw routing so `scripts/moe-trace-replay.py` can score
// cache policies (the shipped F-LRU, static top-f, a per-layer quota, window dedup) against a REAL
// workload instead of the aggregate heat vector.  Two bands, two sources, and neither can disturb a
// captured graph:
//   * DECODE comes from `moe_cache_update_host`, where the scheduler has already read the routing back
//     to the host -- no readback of our own, no stream synchronize;
//   * PREFILL comes from the device hook (`moe_cache_heat_observe`), because the staged prefill path
//     never brings the ids to the host.  A prefill ubatch is never CUDA-graph captured (block 09's
//     staging invariant), so the readback there is safe.
// An IDENTITY (fully resident) table is not traced on decode: its routing is never read back at all.
// By construction it cannot miss, so the replayer models it as a guaranteed hit.
bool                                g_trace_on        = false;
bool                                g_trace_stopped   = false;
std::string                         g_trace_path;
int64_t                             g_trace_max_bytes = 1024LL * 1024 * 1024;
int64_t                             g_trace_bytes     = 0;
int64_t                             g_trace_records   = 0;
FILE *                              g_trace_file      = nullptr;
std::set<std::pair<int, std::string>> g_trace_described;   // (layer, role) table descriptors written
int                                 g_trace_prev_layer = -2;
char                                g_trace_prev_band = '?';
std::string                         g_trace_prev_role;
std::vector<int32_t>                g_trace_prev_ids;     // dedupe across a tensor split (same op, per device)
bool                                g_prov_evict      = false; // MOE_EXPERT_CACHE_PROVISIONAL=1: pre-filled entries are evictable like empty slots until first hit
int                                 g_prefill_seed_n  = 0;     // MOE_EXPERT_CACHE_PREFILL_SEED_N: cap seeded experts/table (0 = all slots)
// V2.0 block 09 (r74): size+seed the arena on the priming pass (the warmup), not the first real decode,
// so the first request does not pay the arena allocation + seed.  DEFAULT ON
// (`MOE_EXPERT_CACHE_SIZE_AT_PRIME=0` restores the deferred sizing).  See ENVIRONMENT.md §2.
bool                                g_size_at_prime   = true;
// V2.0 block 09 (r74): true only for the priming-time sizing, so it does not call `slab_extend_all`
// (which would map the reserve the wide prefill/transients still need).
bool                                g_alloc_priming   = false;
int64_t                             g_alloc_clamped   = 0;
int64_t                             g_alloc_failed    = 0;

// Expert-traffic accounting (per access through the policy): a FILL is one host->device copy of
// `expert_bytes`; a COLD access is the MMVQ kernel reading those bytes in place over PCIe.  Both are
// host->device traffic; only the fill is managed by the cache.
int64_t                             g_acc_hits        = 0;
int64_t                             g_acc_fills       = 0;
int64_t                             g_acc_colds       = 0;
int64_t                             g_fill_bytes      = 0;
int64_t                             g_cold_bytes      = 0;
// issue #274: fills whose source could not be resolved/filled.  A failed fill is rolled back (the slot is
// NOT resident) and the routed op is declined, so this is counted separately from `g_acc_fills`.
int64_t                             g_fill_fail       = 0;
// wip/host-expert-split W2 (MOE_HOST_POOL_SINK): the demote, with the exchange headroom INSIDE the pool.
// The design (maintainer's, 2026-11-06) removes the double copy: instead of staging the victim through a
// separate bounce buffer, the pool plans K slots of *headroom* below its budget and a demote writes the
// victim straight into a free one (ONE D2H).  The promoted expert's slot is retired back into the free list
// when the headroom would otherwise run dry, so the pool's occupancy is constant and nothing grows.  Net
// memory: the same budget, K slots less coverage (K=2 is ~1.7 MiB per 144-table model).
int                                 g_pool_sink       = 0;
int                                 g_pool_sink_spare = -1;   // absolute override; -1 = derive from the table
int                                 g_pool_sink_spare_pct = 4; // % of a table's slots kept free for the exchange (K=16 at 393 slots)
int                                 g_pool_sink_spare_min = 2;
int64_t                             g_sink_copies     = 0;
int64_t                             g_sink_bytes      = 0;
int64_t                             g_sink_declines   = 0;
// Split by CAUSE (issue #274's lesson: a counter that names the wrong cause sends the reader looking in
// the wrong place).  `geom` is structural -- a fused/multi-slice table, or one whose pool stride differs
// from its arena stride, can never demote however much headroom the pool has; only `noslot` is about K.
int64_t                             g_sink_decl_noslot = 0;
int64_t                             g_sink_decl_geom   = 0;
int64_t                             g_sink_decl_copy   = 0;
// W1's break-even is ~3 % on this box's NVMe and ~0.2 % on slow storage, so this ratio is the demote's
// verdict: uses of a demoted slot against demotions issued.
int64_t                             g_sink_uses        = 0;
// Tables whose pool can hold EVERY expert: the exchange is inert there (see `pool_sink_active`).
int64_t                             g_sink_mirror_tables = 0;
// Tables with a coverage gap under 4 % of the set: the exchange is DISENGAGED there (TODO #281 -- a near-mirror
// pool pays the decode-band churn at the next prefill and measures no decode gain).
int64_t                             g_sink_near_mirror_tables = 0;
// Slots reclaimed from a RESIDENT because the pool was saturated (see the demote path).
int64_t                             g_sink_evicted     = 0;
// TODO #281: PREFILL coverage fills that took a slot the decode-band exchange headroom would have kept free.
// Non-zero proves the band override fired; it is the counter that tells "the fix ran" from "the arm changed
// nothing".
int64_t                             g_sink_prefill_fills = 0;
int64_t                             g_sink_retire     = 0;
// Set when a demote could not find a free slot, cleared when the next successful fill retires its
// promoted slot.  Makes the retirement DEMAND-DRIVEN: the pool keeps every duplicate it can (the
// (#87 win) and only gives one up when a demote actually has a victim to park.
bool                                g_sink_need_slot  = false;

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
        const char * e = getenv("MOE_HOST_POOL_MIB");
        if (e == nullptr || e[0] == '\0') {
            return (int64_t) -2;  // unset: the pool is on only if MOE_HOST_POOL=1 (size defaults to FRAC %)
        }
        if (e[0] == 'a' || e[0] == 'A') {
            return (int64_t) -1;  // "auto" = MOE_HOST_POOL_FRAC % of the MoE host expert bytes
        }
        const long long n = atoll(e);
        return (int64_t) (n > 0 ? n : 0);   // 0 = disabled
    }();
    return v;
}

// The bounded host pool is an OPT-IN option for systems that cannot hold the weights resident.  Enabled
// by `MOE_HOST_POOL=1`, or by an explicit `MOE_HOST_POOL_MIB` (a positive MiB value or `auto`); an
// explicit `MOE_HOST_POOL_MIB=0` always disables.  When enabled without a size, the size defaults to
// `MOE_HOST_POOL_FRAC` % of the MoE host expert bytes.
static bool moe_host_pool_on() {
    if (g_pool_requested) {
        return true;   // --host-experts pool
    }
    const int64_t m = moe_host_pool_mib();
    return m != 0 && m != -2;   // explicit MiB / auto; unset off
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

// Sync-free decode device remap (GGML_MOE_DEVREMAP): free a table's map / lag buffers (defined here, inside the
// namespace, for the free sites below).
static void devremap_free_locked(table_t & t) {
    if (t.dr_device < 0) {
        return;
    }
    device_guard dg(t.dr_device);
    if (t.dr_map_dev  != nullptr) { (void) cudaFree(t.dr_map_dev); }
    if (t.dr_map_host != nullptr) { (void) cudaFreeHost(t.dr_map_host); }
    if (t.dr_lag_host != nullptr) { (void) cudaFreeHost(t.dr_lag_host); }
    if (t.dr_map_ev   != nullptr) { (void) cudaEventDestroy(t.dr_map_ev); }
    if (t.dr_lag_ev   != nullptr) { (void) cudaEventDestroy(t.dr_lag_ev); }
    (void) cudaGetLastError();
    t.dr_map_dev = t.dr_lag_dev = t.dr_map_host = t.dr_lag_host = nullptr;
    t.dr_map_ev = t.dr_lag_ev = nullptr;
    ggml_cuda_graph_mem_freed(t.dr_device);   // the map / lag buffers are kernel arguments of captured graphs
    t.dr_cap = 0; t.dr_device = -1; t.dr_lag_n = 0; t.dr_map_ok = false;
}

// Resolve the pinned host master's device-accessible alias once.  `t.src_off` is folded in by
// `moe_cache_get_cold`, so the alias is always the base of the host tensor.
//
// Only a host allocation the runtime can map for device access may be used as an in-place kernel-read
// master, and `cudaHostGetDevicePointer` is the runtime's own answer.  A pageable model mapping
// (`--host-experts mmap` -> `CPU_Mapped`) maps to no device address; on a GPU without XNACK (RDNA
// under ROCm reports `XNACK enabled: NO`) a kernel read of it is a fatal "page not present" fault
// (issue #116, `mul_mat_vec_q_moe`'s cold read).  Leave `host_dev` null in that case: the in-place
// cold region requires it and stands down, while the
// fill copies (which source `t.host` through cudaMemcpyAsync, pageable-safe) keep the cache correct.
void bind_host_dev_locked(table_t & t) {
    // Resolve ONCE, whatever the cold path is configured to do: `host_pinned` also decides whether a
    // device read of the master is legal at all, which the pool-backed master (TODO #55) depends on.
    if (t.host == nullptr || t.host_dev_bound) {
        return;
    }
    t.host_dev_bound = true;
    device_guard dg(t.device);
    void * dev = nullptr;
    if (cudaHostGetDevicePointer(&dev, (void *) t.host, 0) == cudaSuccess && dev != nullptr) {
        t.host_dev = dev;
        t.host_pinned = true;
        return;
    }
    (void) cudaGetLastError();
    // A pageable master (`--host-experts pool` places the tensors in the read-only model mapping): the
    // in-place cold read stands down and every GPU read is staged through the pool instead, so the miss
    // cost is a pool fill rather than a fault (issue #116).
    static bool warned = false;
    if (!warned && g_cold_uva) {
        warned = true;
        GGML_LOG_INFO("%s: the host expert master is pageable (pool-backed `--host-experts pool`); the "
                      "in-place cold read is disabled and misses are served from the pinned host pool\n", __func__);
    }
}

// True when the table can serve a miss by reading the host master in place: the host geometry must be
// representable AND the master must have a device mapping.  A pageable model mapping has none, so the
// table takes the fill-every-miss path instead (issue #116).
static inline bool table_cold_ok(const table_t & t) {
    return g_cold_uva && t.cold_safe && t.host_dev != nullptr;
}

// MOE_HOST_POOL_COLD (default 1): a pool-backed table (`--host-experts pool`, pageable master, so no in-place
// read of the master) serves a miss whose expert is RESIDENT in the pinned pool by reading the pool slot in place
// through the pool's device alias, instead of filling an arena slot from it first.  The bytes are the ones the
// fill would have copied, so the result is unchanged.  The slot stays valid for the read: the demand path never
// evicts a pool resident (TODO #87) and the scheduler synchronizes the backend before the next hook, i.e.
// before any rerank / retire could touch it.
//
// wip/host-expert-pool-region (TODO #139): the pool IS the expert-indexed region the design calls for -- a
// pinned per-host-tensor allocation addressed by (expert -> slot), with `slot_state` as the presence bitmap --
// so a pool-resident expert can be served IN PLACE by the device kernel at any device count, not only on one
// GPU.  The historical restriction was `device_count == 1 && src_off == 0`; both are geometry accidents, not
// properties of the pool: `base_dev` is a Portable|Mapped UVA alias (already the cross-device D2D source for
// fills), and a tensor split's slice offset is folded into the cold base exactly as it is for a pinned master
// (`moe_cache_get_cold`).  `MOE_HOST_POOL_COLD_MULTIDEV` (default ON) is the kill switch; with it off the old
// restriction holds, which is the A/B arm.
static bool g_pool_cold = true;
static bool g_pool_cold_multidev = true;
static host_pool_t * table_pool_locked(table_t & t);
static int64_t pool_ram_cap_locked();   // wip/host-pool-speed W3: the RAM_PCT ceiling, measured once
// The ONE predicate both the host-side cold decision (`pool_cold_slot_locked`) and the kernel-side cold
// description (`moe_cache_get_cold`) must agree on: if they disagree, the host emits a cold id that the kernel
// resolves from a different base, i.e. a silently wrong matmul.
static bool pool_cold_geometry_ok(const table_t & t, const host_pool_t * p) {
    if (!g_cold_uva || !g_pool_cold || p == nullptr || p->failed || p->base_dev == nullptr) {
        return false;
    }
    if (p->host_bytes != t.host_bytes || t.host == nullptr || t.cold_safe == false) {
        return false;
    }
    // A fused table's slot is the concatenation of several host ranges: one stride cannot express it.
    if (t.n_slices > 0) {
        return false;
    }
    if (t.src_off >= t.host_bytes || t.expert_bytes > t.host_bytes - t.src_off) {
        return false;   // the slice does not lie inside a whole run of the region
    }
    if (!g_pool_cold_multidev && (ggml_cuda_info().device_count != 1 || t.src_off != 0)) {
        return false;
    }
    return true;
}
static int pool_cold_slot_locked(table_t & t, int32_t expert) {
    if (t.host_dev != nullptr) {
        return -1;   // a pinned master serves its own cold ids; the pool is for a pageable master
    }
    host_pool_t * p = table_pool_locked(t);
    if (!pool_cold_geometry_ok(t, p) || expert < 0 || expert >= p->n_experts) {
        return -1;
    }
    const auto it = p->expert_slot.find(expert);
    if (it == p->expert_slot.end()) {
        return -1;
    }
    const int ps = it->second;
    return ps >= 0 && ps < p->slots && p->slot_state[(size_t) ps] == 1 ? ps : -1;
}

// wip/host-expert-eager-push (r210): can this table express EVERY expert as a remap id, i.e. is a
// miss always representable?  The eager take-over's completeness guard (`moe_cache_update_host`) needs
// that answer to be STATIC per shape, because a decision that flips per token cannot be baked into one
// captured graph.
//
// * A pinned master (`table_cold_ok`): yes -- the cold region is the master itself (`id = n_res_slots
//   + e`), so no arena bound applies at all.
// * A pool-backed table: `table_cold_ok` is FALSE (the master is the pageable model mapping, so
//   `host_dev` is null), which is why the guard bounded it by the arena -- and then a wide decode band
//   (`slots < n_used * n_tok`, measured: 56 slots against 8x8 at `-llama-batched-bench -npl 8`)
//   declined EVERY routed op and served the whole run from the pageable mapping (tg 49.2 against 270.5
//   for a pinned master).  But a pool whose capacity COVERS the tensor has the pinned master's
//   property: every expert is pool-resident, so the cold id `n_res_slots + slot` is valid for all of
//   them and the arena bound is irrelevant.  What makes that STABLE is the pool's layout policy: when
//   its capacity covers the tensor it never un-loads an expert merely because VRAM also holds it, so
//   coverage is a property of the budget (`slots`), not of the run.
//
// `p->slots` is fixed at the first pool build and `expert_slot` can never hold more than `n_experts`
// entries, so this is a static property; the residency half is re-checked here rather than trusted (LIVE
// slots only -- a slot retiring into the arena still has an index but cannot serve a cold read), and a pool
// with a genuine sub-complement budget (any `-lme N` below the set, any `MOE_HOST_POOL_FRAC` below 100) fails
// the first test and keeps the historical arena bound.  TODO #269: this is also the predicate that decides
// whether the decode remap may encode a cold id as `slots + pool slot`.  Caller holds `g_mutex`.
static int pool_live_slots(const host_pool_t & p) {
    int n = 0;
    for (int s = 0; s < p.slots && s < (int) p.slot_expert.size(); s++) {
        if (p.slot_expert[(size_t) s] >= 0 && p.slot_state[(size_t) s] == 1) {
            n++;
        }
    }
    return n;
}

// The pool that already serves this table, or null.  Unlike `table_pool_locked` this never CREATES a pool, so
// it is safe on the `check_only` remap path and on any pure query.  Caller holds `g_mutex`.
static host_pool_t * table_pool_peek_locked(table_t & t) {
    if (!g_pool_enabled || t.host_dev != nullptr || !t.src_known || t.host == nullptr) {
        return nullptr;
    }
    const auto it = g_host_pools.find(t.host);
    return (it != g_host_pools.end() && !it->second.failed) ? &it->second : nullptr;
}

static bool pool_covers_table(table_t & t) {
    if (t.host_dev != nullptr) {
        return false;   // a pinned master serves its own cold ids; the pool is for a pageable master
    }
    host_pool_t * p = table_pool_locked(t);
    if (!pool_cold_geometry_ok(t, p) || p->slots < t.n_experts) {
        return false;
    }
    return pool_live_slots(*p) >= t.n_experts;   // every expert is a resident cold source
}

// The same coverage test without the pool-creating lookup: `devremap_ok_locked` is a pure predicate
// (`check_only` must not mutate anything), so it answers only from a pool that already exists.  A table that
// has not built its pool yet has never filled from one either, so declining here cannot strand it -- the
// eager path builds the pool on its next pass and the remap is offered again.
static bool pool_covers_table_peek_locked(table_t & t) {
    host_pool_t * p = table_pool_peek_locked(t);
    if (!pool_cold_geometry_ok(t, p) || p->slots < t.n_experts) {
        return false;
    }
    return pool_live_slots(*p) >= t.n_experts;
}

// wip/host-expert-master-drop (TODO #55): the master is not pinned, so it lives in the read-only model
// mapping and every device-side byte must come through the pinned pool (or the per-table bounce).
static inline bool table_master_pageable(const table_t & t) {
    return t.host != nullptr && !t.host_pinned;
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

// Run `parse_env()` exactly once, whichever entry point runs first.  The backend may be asked for its
// host-expert preflight (which arms the movable-boundary slab) BEFORE `ggml_backend_cuda_init()` has run
// `moe_cache_init()`, so the env parse must not depend on that ordering (issue #136).  Caller holds
// `g_mutex` (or is single-threaded during startup).
void parse_env();
static void ensure_env() {
    static bool done = false;
    if (done) {
        return;
    }
    done = true;
    parse_env();
}

void parse_env() {
    // wip/umem-arena-guard: parsed before the early returns below so the shared-pool guard in
    // `moe_cache_preflight` sees it even when the cache itself is explicitly disabled.
    g_unified = env_int("MOE_EXPERT_CACHE_UNIFIED", 0) != 0;
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
    // The bounded, pinned host tier (L2).  PROCESS-WIDE: `g_pool_total_bytes` is the TOTAL budget, split
    // over the registered host tensors in proportion to their size; one copy of each whole expert serves
    // every device.  In `--host-experts pool` mode this is the ONLY pinned host tier (the master is the
    // read-only model mapping, r111); fills read the GGUF with O_DIRECT by default (r112), so the page
    // cache is not a second copy competing for RAM.
    g_pool_enabled     = moe_host_pool_on();
    g_pool_auto        = g_pool_enabled && (moe_host_pool_mib() < 0);
    // DEFAULT = 80 % of each host expert tensor, still subject to the `MOE_HOST_POOL_RAM_PCT` ceiling below.
    // `MOE_HOST_POOL_FRAC` is a percentage of the model's HOST EXPERT SET, not of RAM, and coverage is the
    // pool's decode lever: at `-ncmoe 48` on the 122B-A10B (1 x R9700, 96 GiB, 67.1 GiB of host experts)
    // decode was 17.76 t/s at 80 %, 20.59 at 100 % against 22.34 `pinned` (2026-11-02, wip/host-pool-speed).
    // Full coverage is nevertheless NOT the shipped default: (a) the ceiling below does not reserve the
    // model's own pageable footprint, so on a multi-device split 100 % pins enough to starve the page cache
    // and measured 9 % SLOWER than 95 % (4 GiB less pinned, 42 GiB less disk per run, 2026-11-03); (b) a
    // FRAC 100 pool over-committed a 91 GiB box once, stalling the prepopulate; (c) the 1-GPU srv cell
    // measured 100 % worse than 80 % (624.5 vs 646.8 prefill).  Raise it when the ceiling learns to reserve
    // the model, and measure per split then.
    g_pool_frac        = env_int("MOE_HOST_POOL_FRAC", 80);
    if (g_pool_frac < 1) {
        g_pool_frac = 1;
    }
    if (g_pool_frac > 100) {
        g_pool_frac = 100;
    }
    g_pool_sizing      = env_int("MOE_HOST_POOL_SIZING", 0);
    g_pool_cov_pct     = env_int("MOE_HOST_POOL_COV_PCT", 90);
    if (g_pool_cov_pct < 1)   { g_pool_cov_pct = 1;  }
    if (g_pool_cov_pct > 100) { g_pool_cov_pct = 100; }
    g_pool_ram_pct     = env_int("MOE_HOST_POOL_RAM_PCT", -1);   // -1 = auto (UMA-aware, see pool_ram_pct_eff)
    if (g_pool_ram_pct < -1) {
        g_pool_ram_pct = -1;
    }
    if (g_pool_ram_pct > 100) {
        g_pool_ram_pct = 100;
    }
    g_pool_prewarm     = env_int("MOE_HOST_POOL_PREWARM", 1) != 0;
    // TODO #87 (1): AUTO by default (see `g_pool_dio_req`).  An explicit `MOE_HOST_POOL_DIO=0|1` wins and is
    // decided here, so an A/B never depends on the machine.
    g_pool_dio_req     = env_int("MOE_HOST_POOL_DIO", -1);
    if (g_pool_dio_req < -1) {
        g_pool_dio_req = -1;
    }
    if (g_pool_dio_req >= 0) {
        // Apply the forced mode NOW (no fill may use the other one) but leave `g_pool_dio_decided` clear:
        // `pool_decide_dio_locked` reports the effective mode once, at the first pool build.  Pre-setting the
        // flag here is what made that report unreachable (a forced A/B arm logged no fill mode at all).
        g_pool_dio = g_pool_dio_req != 0;
    }
    g_pool_stage_batch = env_int("MOE_HOST_POOL_STAGE_BATCH", 1) != 0;
    g_pool_stage_span  = env_int("MOE_HOST_POOL_STAGE_SPAN", 1) != 0;
    g_pool_stage_span_n = env_int("MOE_HOST_POOL_STAGE_SPAN_EXPERTS", 16);
    if (g_pool_stage_span_n < 1) {
        g_pool_stage_span_n = 1;
    }
    if (g_pool_stage_span_n > 64) {
        g_pool_stage_span_n = 64;   // 64 experts is ~43 MiB: past that the pinned ring is the cost, not the read
    }
    g_pool_total_bytes = (g_pool_enabled && !g_pool_auto) ? moe_host_pool_mib() * 1024 * 1024 : 0;
    if (g_pool_enabled) {
        // Report the effective fill mode here as well as at the first pool build (`pool_decide_dio_locked`):
        // this line also serves the `MOE_HOST_POOL_MIB` shape, which never goes through `--host-experts pool`.
        const char * fill_mode = g_pool_dio_req < 0 ? "AUTO (MOE_HOST_POOL_DIO=0|1 forces it)"
                               : (g_pool_dio ? "O_DIRECT (forced by MOE_HOST_POOL_DIO=1)"
                                             : "buffered (forced by MOE_HOST_POOL_DIO=0)");
        if (g_pool_auto) {
            GGML_LOG_INFO("%s: MoE host expert pool: AUTO = %d%% of the MoE host expert bytes, pinned; fill "
                          "mode %s\n", __func__, g_pool_frac, fill_mode);
        } else {
            GGML_LOG_INFO("%s: MoE host expert pool: %lld MiB PROCESS-WIDE, pinned; fill mode %s\n",
                          __func__, (long long) moe_host_pool_mib(), fill_mode);
        }
    }
    g_slots_hint  = env_int("MOE_EXPERT_CACHE_SLOTS",   0);
    g_period      = env_int("MOE_EXPERT_CACHE_PERIOD",  32);
    g_fill        = env_int("MOE_EXPERT_CACHE_FILL",     1) != 0;
    // Phase 1b/3 cold transport: the per-op MoE consumer AND the cache-band gate+up+GLU / down-fold
    // kernels do the cold-region lookup, so a cold id is safe with fusions on.  A miss served cold
    // reads the pinned host slice in place (the same bytes the H2D fill would have placed in the
    // slot), and it also lets a small arena represent an overflow (`id = slots + e`).
    g_cold_uva = env_int("MOE_EXPERT_CACHE_COLD_UVA", 1) != 0;
    g_pool_cold = env_int("MOE_HOST_POOL_COLD", 1) != 0;
    g_pool_cold_multidev = env_int("MOE_HOST_POOL_COLD_MULTIDEV", 1) != 0;
    g_stage_cache = env_int("GGML_MOE_CACHE_STAGE", 1) != 0;
    // Admission: `touch` (measured best at every arena size): a first-touch expert is served cold and
    // only a re-touch is admitted, so one-shot experts stop churning the arena - which raises `h` as
    // well as cutting transferred bytes.
    g_admit = env_int("MOE_EXPERT_CACHE_ADMIT", 2);
    // ON by default (2026-11-06, maintainer's decision): safe in every cell measured -- +7.5 % single-stream,
    // +1.8/-0.9/+0.3 % at npl 1/4/8 -- and `MOE_HOST_POOL_SINK=0` is the kill switch.
    g_pool_sink = env_int("MOE_HOST_POOL_SINK", 1) != 0;
    // The headroom is a FRACTION of a table's slots, not a flat count: K=16 was optimal at 512 experts/table
    // (3.1 %) and K=32 there was already too much, so a flat 16 would silently be 6.25 % on a 256-expert
    // table.  MOE_HOST_POOL_SINK_SPARE is the absolute override (-1 = auto).
    g_pool_sink_spare = env_int("MOE_HOST_POOL_SINK_SPARE", -1);
    g_pool_sink_spare_pct = env_int("MOE_HOST_POOL_SINK_SPARE_PCT", 4);
    g_pool_sink_spare_min = env_int("MOE_HOST_POOL_SINK_SPARE_MIN", 2);
    g_touch = env_int("MOE_EXPERT_CACHE_TOUCH", 2);
    if (g_touch < 1) {
        g_touch = 1;
    }
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
    // before decode starts (session 14).  Default ON (kill switch `MOE_EXPERT_CACHE_PREFILL_SEED=0`).
    g_prefill_seed   = env_int("MOE_EXPERT_CACHE_PREFILL_SEED", 1) != 0;
    // wip/moe-heatmap-persistence -> V2.0 block 09 (r60): the long-horizon per-expert heat map.
    // DEFAULT ON (kill switch `MOE_EXPERT_HEATMAP=0`): it only has an effect when the cache is active on
    // host-expert tables, and it turns a cold start into a warm one (measured +34 % decode over a
    // 48-token turn, +10.3 % over 256, fills down 2.8-4x).  It is a placement prior only - the expert
    // bytes are unchanged, so the output is bit-identical and `MOE_EXPERT_HEATMAP=0` restores the exact
    // pre-change behaviour.
    // `MOE_EXPERT_HEATMAP=load` (alias `ro`, `read-only`) keeps the load+seed but never writes, so a
    // curated map can be frozen explicitly; a read-only map file (or a read-only model directory) is
    // also honoured as frozen at open time.
    {
        const char * hm = getenv("MOE_EXPERT_HEATMAP");
        if (hm != nullptr && (strcmp(hm, "load") == 0 || strcmp(hm, "ro") == 0 || strcmp(hm, "read-only") == 0)) {
            g_heat_on     = true;
            g_heat_frozen = true;
        } else {
            g_heat_on     = env_int("MOE_EXPERT_HEATMAP", 1) != 0;
            g_heat_frozen = false;
        }
    }
    g_heat_max_fails  = env_int("MOE_EXPERT_HEATMAP_MAX_FAILS", 3);
    g_heat_debounce_s = env_int("MOE_EXPERT_HEATMAP_DEBOUNCE_S", 60);
    if (g_heat_max_fails < 1)  { g_heat_max_fails = 1; }
    if (g_heat_debounce_s < 0) { g_heat_debounce_s = 0; }
    // Provisional pre-fill/seed slots: a pre-filled expert the policy has never hit is treated as an
    // empty slot for admission (the touch doorkeeper is bypassed for it) while still serving hits.  This
    // is what makes arbitrary load-time pre-fill behave like the empty arena PLUS its bonus hits, and it
    // is what lets the prompt seed's not-yet-hit entries be reclaimed (measured 2026-09-29 session 14:
    // seed `MIB=16384` 79.8 -> 80.6 t/s, `MIB=8192` 57.2 -> 57.8).  Default ON (kill switch
    // `MOE_EXPERT_CACHE_PROVISIONAL=0`); inert until something is pre-filled/seeded, so the default is
    // behaviourally a no-op for an empty arena.
    g_prov_evict     = env_int("MOE_EXPERT_CACHE_PROVISIONAL", 1) != 0;
    g_prefill_seed_n = env_int("MOE_EXPERT_CACHE_PREFILL_SEED_N", 0);
    g_size_at_prime  = env_int("MOE_EXPERT_CACHE_SIZE_AT_PRIME", 1) != 0;
    // TODO #64.2: sampled decode-band heat tally for a fully-resident (identity) table (0 = off).
    g_heat_decode_sample = env_int("MOE_EXPERT_HEATMAP_DECODE_SAMPLE", 64);
    if (g_heat_decode_sample < 0) {
        g_heat_decode_sample = 0;
    }
    // wip/host-expert-split: which counter ranks the host pool's content.  `combined` (default) is the
    // shipped behaviour; `decode` is the counter the arena seeds from; `floor` keeps the prefill-hot
    // MOE_HOST_POOL_RANK_FLOOR % ahead of the decode ranking so the pool's staging duty is preserved.
    {
        // DEFAULT `decode` (2026-11-06, maintainer's decision): +11.1 % single-stream for -5.1 % at npl 1 and
        // -13.2 % at npl 4.  The 4-slot cost is an ACCEPTED LIMITATION -- a single-user stream is the common
        // case -- and `MOE_HOST_POOL_RANK=combined` is the kill switch that restores the previous behaviour.
        const char * pr = getenv("MOE_HOST_POOL_RANK");
        if (pr == nullptr || pr[0] == '\0') {
            g_pool_rank = 1;
        } else if (strcmp(pr, "decode") == 0 || strcmp(pr, "dec") == 0 || strcmp(pr, "1") == 0) {
            g_pool_rank = 1;
        } else if (strcmp(pr, "floor") == 0 || strcmp(pr, "2") == 0) {
            g_pool_rank = 2;
        } else {
            g_pool_rank = 0;   // combined / all / anything else
        }
        g_pool_rank_floor = env_int("MOE_HOST_POOL_RANK_FLOOR", 50);
        if (g_pool_rank_floor < 0)   { g_pool_rank_floor = 0; }
        if (g_pool_rank_floor > 100) { g_pool_rank_floor = 100; }
    }
    // TODO #61 P1: the offline routing trace.  `MOE_EXPERT_TRACE_MIB` caps the file (default 1024); a run
    // that reaches the cap keeps the records it has and stops.
    {
        const char * tr = getenv("MOE_EXPERT_TRACE");
        g_trace_on = tr != nullptr && tr[0] != '\0';
        if (g_trace_on) {
            g_trace_path      = tr;
            g_trace_max_bytes = (int64_t) env_int("MOE_EXPERT_TRACE_MIB", 1024) * 1024 * 1024;
            if (g_trace_max_bytes < 1024 * 1024) {
                g_trace_max_bytes = 1024 * 1024;
            }
            GGML_LOG_WARN("%s: MoE routing trace ON -> %s (cap %.0f MiB; MOE_EXPERT_TRACE_MIB).  This is a "
                          "diagnostic: the prefill band pays a device readback per op, so expect a slower "
                          "run.  Replay it with scripts/moe-trace-replay.py.\n",
                          __func__, g_trace_path.c_str(), (double) g_trace_max_bytes / (1024 * 1024));
        }
    }
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

// Host-pool file I/O portability (issue #130, issue #274 follow-up).  The pool's DISK TIER is the read the
// page cache does not have to serve; every file call goes through these helpers so the platform difference
// lives in one place:
//
//   * POSIX:   `open(2)` (+ `O_DIRECT` in the DIO mode), `pread(2)`, `posix_memalign`.
//   * Windows: `CreateFileA` + one positional `ReadFile` per range + `_aligned_malloc`, with
//              `FILE_FLAG_NO_BUFFERING` for the DIO mode and `FILE_FLAG_OVERLAPPED` always (the reads are
//              positional, so the file pointer is never used).  This is the arm issue #274 asked for: on
//              Windows the buffered fallback is a demand fault per mapped page, measured at ~250 MiB/s with
//              the GPU ~98 % idle through a 4k-token prefill, while the same bytes read as large unbuffered
//              `ReadFile`s go at device speed and do not push the page cache around.
//
// The pool itself does NOT need the disk tier: a pool is only created for a table whose master is the
// read-only model mapping, so `src_host` (the mapping) can always serve a fill and the tier only decides
// whether the PAGE CACHE is populated.  Treating it as required is what broke issue #274: on Windows with
// `MOE_HOST_POOL_DIO=1` every pool fill failed, every pool marked itself failed, the bounce fallback asked
// for the same missing tier and failed too -- and the demand path still recorded the slot as RESIDENT, i.e.
// a garbage expert in the arena rather than a slow one.  The tier stays best-effort everywhere.
//
// `pool_fd_t` is pointer-sized: a Windows HANDLE does not fit in an `int` on Win64.  `MOE_HOST_POOL_DIO_FAULT=1`
// (diagnostic) reproduces "this platform has no pool disk tier" on ANY platform, which is how that arm stays
// reachable from a box with no Windows toolchain (gate arm `nodisk`).
typedef intptr_t pool_ssize_t;

#if defined(_WIN32)
static bool pool_file_io_available() {
    static const bool faulted = getenv("MOE_HOST_POOL_DIO_FAULT") != nullptr;
    return !faulted;
}
static pool_fd_t pool_open_file(const char * path, bool dio) {
    if (!pool_file_io_available()) {
        return (pool_fd_t) -1;
    }
    DWORD flags = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED | FILE_FLAG_SEQUENTIAL_SCAN;
    if (dio) {
        flags |= FILE_FLAG_NO_BUFFERING;
    }
    const HANDLE h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                 nullptr, OPEN_EXISTING, flags, nullptr);
    return h == INVALID_HANDLE_VALUE ? (pool_fd_t) -1 : (pool_fd_t) (intptr_t) h;
}
static void pool_fd_align(pool_fd_t fd, size_t * align) {
    // `FILE_FLAG_NO_BUFFERING` (the O_DIRECT counterpart) requires the offset, the length AND the buffer to
    // be multiples of the volume's PHYSICAL sector size.  4096 is a multiple of every sector size in use
    // (512e and 4Kn), and the caller treats a zero or non-power-of-two value as 4096 anyway.
    (void) fd;
    *align = 4096;
}
static pool_ssize_t pool_pread(pool_fd_t fd, void * dst, size_t len, size_t off) {
    const HANDLE h = (HANDLE) (intptr_t) fd;
    size_t done = 0;
    while (done < len) {
        // `ReadFile` takes a DWORD length; a coalesced prefill span is a few MiB, so this is the whole range.
        const size_t left = len - done;
        const DWORD  n    = (DWORD) (left > (size_t) (1u << 30) ? (size_t) (1u << 30) : left);
        OVERLAPPED ov = {};
        const unsigned long long o = (unsigned long long) (off + done);
        ov.Offset     = (DWORD) (o & 0xffffffffull);
        ov.OffsetHigh = (DWORD) (o >> 32);
        DWORD got = 0;
        if (!ReadFile(h, (char *) dst + done, n, &got, &ov)) {
            // The handle is FILE_FLAG_OVERLAPPED, so a queued read reports ERROR_IO_PENDING.
            if (GetLastError() != ERROR_IO_PENDING || !GetOverlappedResult(h, &ov, &got, TRUE)) {
                break;   // short/failed read: the caller falls back to the mapping
            }
        }
        done += got;
        if (got == 0) {
            break;
        }
    }
    return (pool_ssize_t) done;
}
static bool pool_aligned_alloc(size_t align, size_t size, void ** out) {
    void * p = _aligned_malloc(size, align);
    if (p == nullptr) {
        return false;
    }
    *out = p;
    return true;
}
static void pool_aligned_free(void * p) {
    if (p != nullptr) {
        _aligned_free(p);
    }
}
static void pool_close_file(pool_fd_t fd) {
    if (fd >= 0) {
        CloseHandle((HANDLE) (intptr_t) fd);
    }
}
#else
static bool pool_file_io_available() {
    static const bool faulted = getenv("MOE_HOST_POOL_DIO_FAULT") != nullptr;
    return !faulted;
}
static pool_fd_t pool_open_file(const char * path, bool dio) {
    if (!pool_file_io_available()) {
        return -1;
    }
    return open(path, O_RDONLY | (dio ? O_DIRECT : 0));
}
static void pool_fd_align(pool_fd_t fd, size_t * align) {
    struct stat st;
    if (fstat(fd, &st) == 0 && st.st_blksize > 0) {
        *align = (size_t) st.st_blksize;
    }
}
static pool_ssize_t pool_pread(pool_fd_t fd, void * dst, size_t len, size_t off) {
    return (pool_ssize_t) pread(fd, dst, len, (off_t) off);
}
static bool pool_aligned_alloc(size_t align, size_t size, void ** out) {
    return posix_memalign(out, align, size) == 0;
}
static void pool_aligned_free(void * p) {
    free(p);
}
static void pool_close_file(pool_fd_t fd) {
    close(fd);
}
#endif

// One O_DIRECT read (the debug fallback; the default fill is a plain buffered `pread`) of `len` bytes at
// `off` into `dst`, through a process-global aligned bounce buffer (O_DIRECT needs an aligned buffer,
// length and offset).  Caller holds `g_mutex`.
static bool pool_dio_read(pool_fd_t fd, void * dst, size_t len, size_t off, size_t align) {
    if (align == 0 || (align & (align - 1)) != 0) {
        align = 4096;
    }
    if (off % align == 0 && len % align == 0 && ((uintptr_t) dst % align) == 0) {
        const pool_ssize_t n = pool_pread(fd, dst, len, off);
        return n == (pool_ssize_t) len;
    }
    const size_t aoff = off & ~(align - 1);
    const size_t skip = off - aoff;
    const size_t blen = (skip + len + align - 1) & ~(align - 1);
    // THREAD-LOCAL: fills are issued under `g_mutex`, but a caller may hold the lock on one thread while
    // another path is inside this function (and the alignment tail copy must never share a bounce): a
    // shared static bounce was raced once and corrupted the tail of an expert (measured as a POOLREAD
    // MISMATCH from ~520 KiB into a 576 KiB expert, variably).  `MOE_HOST_POOL_DIO` is the disk-tier mode,
    // so this path must be correct on its own.
    static thread_local void * bounce      = nullptr;
    static thread_local size_t bounce_size = 0;
    if (bounce == nullptr || bounce_size < blen) {
        pool_aligned_free(bounce);
        bounce = nullptr;
        bounce_size = 0;
        void * p = nullptr;
        if (!pool_aligned_alloc(align, blen, &p)) {
            return false;
        }
        bounce = p;
        bounce_size = blen;
    }
    const pool_ssize_t n = pool_pread(fd, bounce, blen, aoff);
    if (n < 0 || (size_t) n < skip + len) {
        return false;
    }
    memcpy(dst, (const char *) bounce + skip, len);
    return true;
}

static char * pool_slot_locked(host_pool_t & p, int expert, int * out_slot, bool sync_on_evict);
static void   pool_prepopulate_locked(host_pool_t & p, const table_t & t);

// The (slot, device) in-flight event, created lazily on `device` (the device whose stream will record
// it).  Returns nullptr when the index is invalid or the event cannot be created (the caller then just
// skips the drain marker; the slot is conservatively left reusable only after a full sync).
static cudaEvent_t pool_slot_event(host_pool_t & p, int slot, int device) {
    if (slot < 0 || slot >= p.slots || device < 0 || device >= GGML_CUDA_MAX_DEVICES) {
        return nullptr;
    }
    const size_t idx = (size_t) slot * GGML_CUDA_MAX_DEVICES + (size_t) device;
    if (idx >= p.slot_ev.size()) {
        return nullptr;
    }
    if (p.slot_ev[idx] == nullptr) {
        device_guard dg(device);
        cudaEvent_t ev = nullptr;
        if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess) {
            (void) cudaGetLastError();
            return nullptr;
        }
        p.slot_ev[idx] = ev;
    }
    return p.slot_ev[idx];
}

// Drain every device's copy out of `slot` before the slot is overwritten or retired.
static void pool_slot_sync(host_pool_t & p, int slot) {
    if (slot < 0 || slot >= p.slots) {
        return;
    }
    for (int d = 0; d < GGML_CUDA_MAX_DEVICES; d++) {
        const size_t idx = (size_t) slot * GGML_CUDA_MAX_DEVICES + (size_t) d;
        if (idx < p.slot_ev.size() && p.slot_ev[idx] != nullptr) {
            if (cudaEventSynchronize(p.slot_ev[idx]) != cudaSuccess) {
                (void) cudaGetLastError();
            }
        }
    }
}

// TODO #87: how much can this box actually PIN?  `MOE_HOST_POOL_FRAC` is a percentage of the model's host
// expert set, not of RAM, so on a memory-tight box it can ask for more pinned memory than exists -- and an
// oversized pool does not degrade gracefully: each table's `cudaHostAlloc` fails, that table's pool is
// marked failed, and its reads silently fall back to the per-table bounce (i.e. the disk).  Better to trim
// the pool to what fits and say so.  Measured once, at the first pool build (after the model's own host
// mapping exists, before it is faulted in), so it is stable and order-independent.
static int64_t pool_available_ram_bytes() {
#if defined(__linux__)
    FILE * f = fopen("/proc/meminfo", "r");
    if (f == nullptr) {
        return -1;
    }
    char line[256];
    int64_t kb = -1;
    while (fgets(line, sizeof(line), f) != nullptr) {
        long long v = 0;
        if (sscanf(line, "MemAvailable: %lld kB", &v) == 1) {
            kb = (int64_t) v;
            break;
        }
    }
    fclose(f);
    return kb > 0 ? kb * 1024 : -1;
#elif defined(GGML_USE_HIP) && defined(_WIN32)
    return ggml_hip_win_available_ram_bytes();
#else
    return -1;
#endif
}

// wip/apu-expert-pool: on a shared host/device pool (a UMA APU) "VRAM" and RAM are one medium, so the two
// tiers must be sized from the SAME figure -- free system memory -- and they must not add up to all of it:
//   * VRAM (the arena / the slab ceiling) = 75 % of MemAvailable   (MOE_UMA_VRAM_PCT)
//   * the pinned expert pool (auto)       = 10 % of MemAvailable   (MOE_UMA_POOL_PCT)
//   * the remaining ~15 %                 = the OS, the page cache and the model mapping the pool fills from
// The percentages are ceilings, not reservations: a tier only takes what its tables need.  `MOE_HOST_POOL_RAM_PCT`
// and `MOE_EXPERT_CACHE_MIB` remain the operator's authoritative overrides on any platform.
static bool uma_any_device() {
    static const bool v = [] {
        for (int i = 0; i < GGML_CUDA_MAX_DEVICES; i++) {
            if (ggml_cuda_info().devices[i].integrated) {
                return true;
            }
        }
        return false;
    }();
    return v;
}

// The share of free system RAM an APU's "VRAM" is treated as (a percentage; the operator can move it).
static int uma_vram_pct() {
    static const int v = [] {
        const char * e = getenv("MOE_UMA_VRAM_PCT");
        const int p = e != nullptr ? atoi(e) : 75;
        return p > 0 && p <= 95 ? p : 75;
    }();
    return v;
}

// The share of free system RAM the pinned expert pool may take on a UMA box (auto mode only).
static int uma_pool_pct() {
    static const int v = [] {
        const char * e = getenv("MOE_UMA_POOL_PCT");
        const int p = e != nullptr ? atoi(e) : 10;
        return p > 0 && p <= 95 ? p : 10;
    }();
    return v;
}

// The effective `MOE_HOST_POOL_RAM_PCT`: the operator's value when set (0 = no ceiling), else the UMA default on a
// shared pool and the historical 85 elsewhere.
static int pool_ram_pct_eff() {
    if (g_pool_ram_pct >= 0) {
        return g_pool_ram_pct;
    }
    return uma_any_device() ? uma_pool_pct() : 85;
}

// wip/host-pool-speed: the slots `t` needs to hold `cov_pct` of its own prior mass, or -1 when there is no
// prior to rank by (the caller then falls back to the uniform ask).  `t.heat` is the persisted long-horizon
// prior applied at registration (`heat_apply_loaded_locked`) -- the same ranking the pool's content uses, so a
// table that has never been seen falls back rather than being sized from noise.  Caller holds `g_mutex`.
static int table_cov_slots_locked(const table_t & t, int cov_pct) {
    if (t.n_experts <= 0) {
        return 1;
    }
    if ((int) t.heat.size() != t.n_experts) {
        return -1;
    }
    std::vector<uint32_t> h(t.heat.begin(), t.heat.end());
    std::sort(h.begin(), h.end(), std::greater<uint32_t>());
    uint64_t tot = 0;
    for (uint32_t v : h) {
        tot += v;
    }
    if (tot == 0) {
        return -1;   // a prior with no mass at all: nothing to concentrate on
    }
    uint64_t cum = 0;
    for (int i = 0; i < t.n_experts; i++) {
        cum += h[(size_t) i];
        if (cum * 100 >= tot * (uint64_t) cov_pct) {
            return i + 1;
        }
    }
    return t.n_experts;
}

// One-shot fit of the per-table pool slots by concentration, to the SAME total byte budget the uniform rule
// would spend (`MOE_HOST_POOL_FRAC` of the host expert set).  Runs at the first pool build, when every host
// tensor is registered, so it is order-independent (which is what the per-tensor FRAC rule exists to guarantee).
// `MOE_HOST_POOL_MIB` (an explicit process-wide byte budget) keeps the uniform split.  Caller holds `g_mutex`.
static void pool_fit_cov_locked() {
    if (g_pool_cov_fit) {
        return;
    }
    g_pool_cov_fit = true;
    struct plan_t { const void * host; int k; int n_exp; size_t bytes; };
    std::vector<plan_t> plan;
    std::set<const void *> seen;
    int64_t budget = 0;
    int64_t uniform_slots = 0;
    for (const table_t & t : g_tables) {
        if (!t.src_known || t.host == nullptr || t.host_bytes == 0 || t.n_experts <= 0) {
            continue;
        }
        if (!seen.insert(t.host).second) {
            continue;   // one plan per host TENSOR (the pool is shared across a split's per-device tables)
        }
        const int uni = std::max(1, std::min(t.n_experts, (int) ((int64_t) t.n_experts * g_pool_frac / 100)));
        int k = table_cov_slots_locked(t, g_pool_cov_pct);
        if (k < 0) {
            k = uni;    // no prior: ask for what the uniform rule asks for
        }
        k = std::max(1, std::min(k, t.n_experts));
        budget         += (int64_t) uni * (int64_t) t.host_bytes;
        uniform_slots += uni;
        plan.push_back({ t.host, k, t.n_experts, t.host_bytes });
    }
    if (plan.empty() || budget <= 0) {
        return;
    }
    // wip/host-pool-speed W3: the concentration fit is a REDISTRIBUTION at a fixed byte total, so the ceiling
    // has to be applied to the TOTAL it fits to, not afterwards.  Scaling a fitted plan uniformly preserves
    // only the average and collapses the concentration straight back to the uniform rule (measured: FRAC 100
    // + `RAM_PCT=70` gave 144 x 208 slots with and without `MOE_HOST_POOL_SIZING=1`).
    {
        const int64_t cap = pool_ram_cap_locked();
        if (cap > 0 && budget > cap) {
            budget = cap;
        }
    }
    // slots_i = clamp(round(alpha * k_i), 1, n_exp_i); find the largest alpha whose byte total fits.
    auto alloc = [&](double alpha, std::vector<int> * out) {
        int64_t bytes = 0;
        if (out != nullptr) {
            out->assign(plan.size(), 0);
        }
        for (size_t i = 0; i < plan.size(); i++) {
            const plan_t & c = plan[i];
            int sl = (int) std::llround(alpha * (double) c.k);
            sl = std::max(1, std::min(c.n_exp, sl));
            bytes += (int64_t) sl * (int64_t) c.bytes;
            if (out != nullptr) {
                (*out)[i] = sl;
            }
        }
        return bytes;
    };
    double lo = 0.0, hi = 1.0;
    while (alloc(hi, nullptr) <= budget && hi < 1e9) {
        hi *= 2.0;
    }
    for (int it = 0; it < 80; it++) {
        const double mid = 0.5 * (lo + hi);
        if (alloc(mid, nullptr) <= budget) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    std::vector<int> slots;
    int64_t used = alloc(lo, &slots);
    // Hand the rounding leftovers to the tables that want the most slots (largest k first) while they fit, then
    // (if a floor overshot) take slots back from the smallest-k tables until the budget is respected.
    std::vector<size_t> order(plan.size());
    for (size_t i = 0; i < order.size(); i++) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return plan[a].k > plan[b].k; });
    for (size_t i : order) {
        while (slots[i] < plan[i].n_exp && used + (int64_t) plan[i].bytes <= budget) {
            slots[i]++;
            used += (int64_t) plan[i].bytes;
        }
    }
    for (auto it = order.rbegin(); it != order.rend() && used > budget; ++it) {
        const size_t i = *it;
        while (slots[i] > 1 && used > budget) {
            slots[i]--;
            used -= (int64_t) plan[i].bytes;
        }
    }
    int64_t kept = 0;
    int64_t kept_bytes = 0;
    for (size_t i = 0; i < plan.size(); i++) {
        g_pool_cov_slots[plan[i].host] = slots[i];
        kept += slots[i];
        kept_bytes += (int64_t) slots[i] * (int64_t) plan[i].bytes;
    }
    g_pool_cov_bytes = kept_bytes;
    GGML_LOG_INFO("%s: host pool SIZING = cov%d (concentration, %s): %zu host tensors, budget %.1f MiB, "
                  "alpha=%.3f, %lld slots vs %lld uniform (%.1f %%) -- per-table slots now follow each table's "
                  "own routing concentration instead of a flat FRAC (MOE_HOST_POOL_SIZING=0 restores it)\n",
                  __func__, g_pool_cov_pct,
                  "MOE_HOST_POOL_COV_PCT of each table's prior mass",
                  plan.size(), (double) budget / (1024.0 * 1024.0), lo,
                  (long long) kept, (long long) uniform_slots,
                  uniform_slots > 0 ? 100.0 * (double) kept / (double) uniform_slots : 0.0);
}

// TODO #87 (1): decide whether the pool fills from the page cache or with O_DIRECT.  Called ONCE, from the
// first pool build, so every pool and every bounce path in the process agrees.
//
// The trade is RAM, not I/O policy: a buffered fill leaves the set in the page cache, which then serves the
// bytes no tier holds (the FRAC remainder, and the whole set for a prefill ubatch's staging) at RAM speed --
// measured on the 2x R9700 field config, `--host-experts pool` went 287 -> 733 t/s prefill and 23.1 -> 53.7
// t/s decode for the switch alone.  But the page cache and the pinned pool compete for the same RAM, so on a
// box where the set does not fit in what the pool leaves, buffering thrashes and O_DIRECT wins (the 30 GiB
// gfx1100 box: DIO=1 is the stable disk-tier shape, FRAC 90 + buffered swapped).
//
// The test is the one `MOE_HOST_POOL_RAM_PCT` already takes: if the planned pinned pool PLUS the whole host
// expert set fits in MemAvailable, the page cache can hold the set and buffering is free; otherwise every
// buffered miss evicts the cache it needs.  The middle band (the set fits, the set + a full pool does not)
// is unmeasured -- a forced `MOE_HOST_POOL_DIO=0|1` is the escape hatch there.
// Caller holds `g_mutex`.
static bool pool_dio_effective(bool want) {
    if (!want || pool_file_io_available()) {
        return want;
    }
    // issue #274: O_DIRECT fills need `pool_open_file`, and a platform without the pool's file I/O reports
    // the tier absent (POSIX has it, Windows has it since the ReadFile arm, and `MOE_HOST_POOL_DIO_FAULT=1`
    // removes it deliberately).  Asking for the disk tier where it is absent used to fail EVERY pool fill,
    // fail every pool, and then fail the bounce fallback as well -- while the demand path still recorded the
    // slot as resident, so the run produced garbage rather than a slow answer.  The tier is an optimisation,
    // not a requirement (the pool's master is the read-only model mapping), so refuse it and keep the pool.
    static bool warned = false;
    if (!warned) {
        warned = true;
        GGML_LOG_WARN("%s: O_DIRECT fills are NOT available on this platform (the pool's file I/O is "
                      "disabled, or MOE_HOST_POOL_DIO_FAULT is set), so the host pool fills through the model "
                      "mapping / page cache instead.  Nothing is disabled: the pool, its pin, the in-place "
                      "cold read and the per-table bounce ring are unchanged.\n", __func__);
    }
    return false;
}

static void pool_decide_dio_locked(int64_t host_total, int64_t planned_pool) {
    if (g_pool_dio_decided) {
        return;
    }
    g_pool_dio_decided = true;
    if (g_pool_dio_req >= 0) {
        // Forced (`MOE_HOST_POOL_DIO=0|1`): `parse_env` already applied it; this is the one place that reports
        // it, because it runs once, at the first pool build, at a log level the operator can see.
        g_pool_dio = pool_dio_effective(g_pool_dio_req != 0);
        GGML_LOG_INFO("%s: host pool fill mode: %s (forced by MOE_HOST_POOL_DIO=%d)\n", __func__,
                      g_pool_dio ? "O_DIRECT (page cache bypassed)" : "buffered (page cache as L3)",
                      g_pool_dio_req);
        return;
    }
    // r212: mirror the operator's `--load-mode` when the loader told us what it was.  A pool filling from
    // a MAPPED master reads through the page cache, so `--load-mode mmap` is a request for the page cache;
    // bypassing it (the old heuristic: "the set does not fit beside the pool") is both a surprise and
    // slower on the only read that cannot be avoided (the population), measured on a 113 GB model at
    // `-ncmoe 48 -lme 60`: 95.20 GiB of storage reads at 2 785 MiB/s where the mapping was 4x faster and
    // stayed reusable.  `MOE_HOST_POOL_DIO=0|1` (handled above) still forces either arm.
    if (g_pool_load_mode >= 0) {
        g_pool_dio = pool_dio_effective(g_pool_load_mode != 0);
        // The mirror is the POLICY, but the trade-off is worth stating at the moment it is taken: when the
        // set does not fit beside the pinned pool, a buffered fill's page cache competes with the pool for
        // the same RAM (r112 measured O_DIRECT as the right arm there: 18-19 vs 3.2-3.3 t/s on a box whose
        // set overflowed).  An operator who asked for `--load-mode mmap` asked for the page cache, so this
        // is their call, not ours -- say what it costs and leave the override one env var away.
        const int64_t avail_m   = pool_available_ram_bytes();
        const int64_t planned_m = planned_pool;
        const bool    tight     = avail_m > 0 && planned_m > 0 && host_total > 0 &&
                                  planned_m + host_total > avail_m;
#if defined(_WIN32)
        // issue #274 follow-up (@DanoPTT): on Windows the "page cache as L3" model does not hold in the
        // memory-tight band, because a mapped-view fill is a DEMAND FAULT per page (measured ~250 MiB/s
        // while the GPU computed ~2 % of a 4k-token prefill, and the whole expert set re-read per pass)
        // rather than the kernel's readahead.  The direct tier reads the same bytes as large unbuffered
        // `ReadFile`s and does not push the page cache around, so it is the memory-tight arm HERE even when
        // the operator asked for `--load-mode mmap`.  `MOE_HOST_POOL_DIO=0` is the escape hatch.
        if (!g_pool_dio && tight) {
            g_pool_dio = pool_dio_effective(true);
            GGML_LOG_INFO("%s: host pool fill mode: O_DIRECT -- Windows takes the direct tier in the "
                          "memory-tight band even with --load-mode mmap (a mapped fill is a demand fault per "
                          "page there; MOE_HOST_POOL_DIO=0 forces the page cache)\n", __func__);
        }
#endif
        GGML_LOG_INFO("%s: host pool fill mode: %s -- mirroring --load-mode %s (MOE_HOST_POOL_DIO=0|1 "
                      "forces either arm)%s\n", __func__,
                      g_pool_dio ? "O_DIRECT (page cache bypassed)" : "buffered (page cache as L3)",
                      g_pool_dio ? "direct-io" : "mmap/mlock",
                      (!g_pool_dio && tight) ? "; NOTE MemAvailable does not hold the pool AND the host expert set, "
                      "so the page cache cannot keep it -- MOE_HOST_POOL_DIO=1 is the memory-tight arm" : "");
        if (!g_pool_dio && tight) {
            GGML_LOG_INFO("%s: host pool memory: MemAvailable %.1f MiB, planned pool %.1f MiB, host expert set "
                          "%.1f MiB (buffered fills follow --load-mode mmap; the fills may re-read after reclaim)\n",
                          __func__, (double) avail_m / (1024.0 * 1024.0), (double) planned_m / (1024.0 * 1024.0),
                          (double) host_total / (1024.0 * 1024.0));
        }
        return;
    }
    const int64_t avail = pool_available_ram_bytes();
    if (avail <= 0 || host_total <= 0) {
        g_pool_dio = pool_dio_effective(true);   // no MemAvailable to reason with: keep the memory-tight default (r112)
        GGML_LOG_INFO("%s: host pool fill mode: O_DIRECT (no MemAvailable, keeping the memory-tight default;"
                      " MOE_HOST_POOL_DIO=0 forces buffered)\n", __func__);
        return;
    }
    int64_t pool = planned_pool;
    if (pool_ram_pct_eff() > 0) {
        const int64_t cap = (int64_t) ((double) avail * (double) pool_ram_pct_eff() / 100.0);
        if (pool <= 0 || pool > cap) {
            pool = cap;
        }
    }
    g_pool_dio = pool_dio_effective(pool + host_total > avail);
    GGML_LOG_INFO("%s: host pool fill mode: %s -- MemAvailable %.1f MiB, planned pool %.1f MiB, host expert"
                  " set %.1f MiB (%s; MOE_HOST_POOL_DIO=0|1 forces either arm)\n", __func__,
                  g_pool_dio ? "O_DIRECT (page cache bypassed)" : "buffered (page cache as L3)",
                  (double) avail / (1024.0 * 1024.0), (double) pool / (1024.0 * 1024.0),
                  (double) host_total / (1024.0 * 1024.0),
                  g_pool_dio ? "the set does not fit beside the pool" : "the set fits beside the pool");
}

// Cap this table's `slots` so the WHOLE pinned pool stays under `MOE_HOST_POOL_RAM_PCT` % of MemAvailable.
// Caller holds `g_mutex` (it updates the process-wide running total).
//
// wip/host-pool-speed W3: the ceiling is applied as ONE proportional scale over every table, not as a shave
// of whichever tables happen to register last.  The old form gave the registration-order tail ONE slot each
// once the running total reached the cap -- measured on the 122B at `RAM_PCT=70`: 14 of 144 tables at 1 slot
// and **-40 % decode** for 10 % fewer slots (README §14.6), and on the 3-GPU pair at `FRAC=100` the same
// effect alone dropped the pool's hit rate from 0.91 to 0.78.  A proportional shave costs every table a
// little coverage instead, which is what a memory ceiling should do.
//
// `planned_total` is the pre-ceiling pinned size the caller was aiming for (FRAC / MIB / cov plan), i.e. the
// denominator the ceiling is compared against.  It is a process-wide number, so the scale is too.
// The `MOE_HOST_POOL_RAM_PCT` ceiling in bytes, measured once (0 = no ceiling: disabled, or no MemAvailable).
static int64_t pool_ram_cap_locked() {
    if (pool_ram_pct_eff() <= 0) {
        return 0;
    }
    if (g_pool_ram_cap < 0) {
        const int64_t avail = pool_available_ram_bytes();
        // issue #274 follow-up (@DanoPTT): on Windows this figure is NOT the pinned ceiling.  WDDM charges
        // pinned host pages against the adapter's SHARED (aperture) segment, and the published model for that
        // is roughly (total RAM / 2) / visible GPUs (ROCM-33542 / rocm-systems#14578, ROCm/TheRock#9339:
        // ~13.4 GiB on a 27.6 GiB box with one adapter, ~6.5 GiB with two).  The reporter's box REFUTES that
        // model as a predictor: shared usage stops at 9.3-9.4 GiB on both a 47 GiB and a 64 GiB machine with
        // one visible adapter, so something already counts against the budget before the pool does.  A cap
        // derived from it was considered and deliberately NOT shipped: it is unmeasured here, and getting it
        // wrong caps a healthy box.  The guards that do not guess are `g_pinned_off` (the first refused
        // cudaHostAlloc latches the pinned tier off with one WARN naming the platform cause) and the exit
        // census, which now reports the pinned tier's total AND, on Windows, the adapter's aperture
        // CommitLimit/resident through `ggml_hip_win_shared_mem_info`.  Add a ceiling here only once that
        // figure is understood.
        g_pool_ram_cap = avail > 0 ? (int64_t) ((double) avail * (double) pool_ram_pct_eff() / 100.0) : 0;
        if (g_pool_ram_cap > 0) {
            GGML_LOG_INFO("%s: host pool RAM ceiling: %d%% of MemAvailable = %.1f MiB (MOE_HOST_POOL_RAM_PCT)\n",
                          __func__, pool_ram_pct_eff(), (double) g_pool_ram_cap / (1024.0 * 1024.0));
        }
    }
    return g_pool_ram_cap;
}

static int pool_clamp_slots_to_ram(int slots, size_t host_bytes, int64_t planned_total, const table_t & t) {
    if (g_pool_ram_pct <= 0 || slots <= 0 || host_bytes == 0) {
        return slots;   // ceiling disabled
    }
    if (pool_ram_cap_locked() <= 0) {
        return slots;   // no MemAvailable (non-Linux, or unreadable): keep the configured size
    }
    // Decide the proportional scale once, from the process-wide plan vs the process-wide ceiling.
    if (g_pool_ram_scale < 0.0) {
        if (planned_total > 0 && planned_total > g_pool_ram_cap) {
            g_pool_ram_scale = (double) g_pool_ram_cap / (double) planned_total;
            if (g_pool_ram_scale > 1.0) {
                g_pool_ram_scale = 1.0;
            }
            GGML_LOG_INFO("%s: the host pool does not fit the RAM ceiling: planned %.1f MiB against %.1f MiB "
                          "(%d%% of MemAvailable), so EVERY table's slots are scaled by %.3f instead of "
                          "starving the tables that register last (MOE_HOST_POOL_FRAC is a percentage of the "
                          "model's HOST EXPERT SET, not of RAM; lower it, or raise MOE_HOST_POOL_RAM_PCT, to "
                          "size this deliberately)\n",
                          __func__, (double) planned_total / (1024.0 * 1024.0),
                          (double) g_pool_ram_cap / (1024.0 * 1024.0), g_pool_ram_pct, g_pool_ram_scale);
        } else {
            g_pool_ram_scale = 1.0;
        }
    }
    int out = slots;
    const int64_t spare = g_pool_ram_cap - g_pool_ram_used;
    if (g_pool_ram_scale < 1.0) {
        out = (int) ((double) slots * g_pool_ram_scale);
        if (out < 1) {
            out = 1;    // the floor is kept only where the ceiling cannot cover even one slot
        }
    }
    // Backstop: the proportional target can still overshoot (the one-slot floor, and slot rounding across
    // 144+ tables).  If this table would cross the ceiling, shave it to what actually fits; only a table that
    // cannot fit at all falls to the floor.
    if (spare <= 0 || (int64_t) out * (int64_t) host_bytes > spare) {
        const int fit = spare > 0 ? (int) (spare / (int64_t) host_bytes) : 0;
        if (fit < out) {
            static bool warned_trim = false;
            if (!warned_trim) {
                warned_trim = true;
                GGML_LOG_INFO("%s: the RAM ceiling still binds after the proportional scale (layer=%d "
                              "role=%s: %d slots wanted, %d fit in the %.1f MiB left); one-slot floor applies "
                              "to the remainder\n",
                              __func__, t.layer, t.role.c_str(), out, fit, (double) spare / (1024.0 * 1024.0));
            }
            out = fit < 1 ? 1 : fit;
        }
    }
    g_pool_ram_used += (int64_t) out * (int64_t) host_bytes;
    return out;
}

// The pool for `t`'s host tensor, created lazily and shared by every per-device table of that tensor.
// Caller holds `g_mutex`.  Returns null (and every fill falls back to the full master) when the pool is
// off, the source is unknown, or the budget is too small for one expert.
// r215: the pinned host pool's BACKING.  `cudaHostAlloc(Portable|Mapped)` pins page by page, so the
// 48 GiB mirror of a `-ncmoe all` model costs ~5.4 s of pure pinning -- and it pins a zero page for every
// 4 KiB page that the mirror fill then overwrites.  Two changes, both measured (gfx1201,
// Qwen3.8-Flash-Next IQ3_S, `-ncmoe 48 --load-mode mmap`, cold page cache):
//
//   * back the pool with an anonymous `mmap` + `MADV_HUGEPAGE` (2 MiB pages) and pin it with
//     `cudaHostRegister` AFTER the mirror has been filled, so the pin walks pages that already exist
//     (~1 ms instead of ~5.4 s: the fill's own page allocation is work that used to be wasted);
//   * the result still carries a UVA device alias for the in-place cold read and the cross-device D2D
//     fill source -- exactly the property `cudaHostAlloc(Portable|Mapped)` provided.
//
// `MOE_HOST_POOL_ALLOC=thp` selects it (the default); `portable` (the pre-r215 pair), `plain`
// (`cudaMallocHost`) and `mapped` (`cudaHostAllocMapped`) remain as the A/B arm and the escape hatch.
// Anonymous THP pages return to the buddy allocator on `munmap` (there is no hugetlbfs reservation to
// leak), and the pool's SLOTS are index bookkeeping inside this one allocation, so recycling a slot never
// pins or unpins anything: measured over a run with 35 559 slot fills, one register and one unregister
// per host tensor.
static void * pool_mmap_thp_alloc(size_t bytes) {
#if defined(__linux__)
    void * p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
        return nullptr;
    }
    (void) madvise(p, bytes, MADV_HUGEPAGE);   // best effort: no THP just means the historical pin cost
    return p;                                   // deliberately NOT pinned yet -- `pool_mmap_thp_pin`
#endif
    (void) bytes;
    return nullptr;
}

// Pin the (already populated) mirror and resolve its device alias.  Registering an EMPTY anonymous range
// makes the kernel allocate and pin a zero page per page, all of which the fill overwrites; registering it
// after the fill only walks pages that already exist.
static bool pool_mmap_thp_pin(host_pool_t & p) {
#if defined(__linux__)
    if (p.base == nullptr || p.mmap_bytes == 0 || p.registered) {
        return p.registered;
    }
    // wip/apu-expert-pool: a READ-ONLY device registration is the fast in-place read on a shared host/device
    // pool (measured on gfx1151: 233 GB/s against 60 GB/s for the writable one -- the coherent mapping; the
    // PCIe link on a discrete box masks the difference, so it needs an A/B there).  It is only LEGAL while
    // nothing writes the backing FROM THE DEVICE: the arena->pool demote (`access_locked`'s sink path, the
    // one `cudaMemcpyDeviceToHost` into `pool->base`) is that writer, and it cannot exist without an arena.
    // So: on when the expert cache is disabled for this run (every shared-pool APU today, and any
    // `MOE_EXPERT_CACHE_MIB=0` run), off when an arena is live.  `MOE_HOST_POOL_RO=1|0` forces either, and
    // forcing it on with a live arena is refused loudly rather than corrupting the pool.
    bool pool_ro = !g_enabled;
    if (const char * e = getenv("MOE_HOST_POOL_RO")) {
        const bool want = atoi(e) != 0;
        if (want && g_enabled) {
            GGML_LOG_WARN("%s: MOE_HOST_POOL_RO=1 refused: the expert arena is live, and the arena->pool demote "
                          "writes this backing from the device; a read-only registration would corrupt it\n",
                          __func__);
        } else {
            pool_ro = want;
        }
    }
    const unsigned reg_flags = cudaHostRegisterPortable | cudaHostRegisterMapped |
                               (pool_ro ? cudaHostRegisterReadOnly : 0);
    {
        static std::once_flag once_ro;
        std::call_once(once_ro, [pool_ro, reg_flags] {
            GGML_LOG_INFO("%s: host pool registered %s (flags=0x%x): a read-only device alias is the fast "
                          "in-place read on a shared host/device pool (MOE_HOST_POOL_RO)\n",
                          __func__, pool_ro ? "READ-ONLY" : "writable", reg_flags);
        });
    }
    if (cudaHostRegister(p.base, p.mmap_bytes, reg_flags) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    p.registered = true;
    void * dev = nullptr;
    if (cudaHostGetDevicePointer(&dev, p.base, 0) != cudaSuccess) {
        // UVA: the host pointer IS the device address (the delivered KV pool mirror relies on the same
        // property), so an unavailable alias is not a failure.
        (void) cudaGetLastError();
        dev = p.base;
    }
    p.base_dev = dev;
    return true;
#else
    (void) p;
    return false;
#endif
}

static void pool_mmap_thp_release(host_pool_t & p) {
#if defined(__linux__)
    if (p.base == nullptr || p.mmap_bytes == 0) {
        return;
    }
    if (p.registered) {
        (void) cudaHostUnregister(p.base);
        p.registered = false;
    }
    munmap(p.base, p.mmap_bytes);
    p.mmap_bytes = 0;
#else
    (void) p;
#endif
}

// Diagnostic (issue #278): emulate the platform's LATCHED pinned refusal -- what a Windows box does when WDDM
// refuses a pinned allocation against the adapter's shared-segment budget (see `g_pinned_off`).  Two shapes,
// because WHEN the tier goes is what the report is about:
//   =1      refuse every pinned allocation from the start: no pool is ever built and no ring ever grows (the
//           "the tier never came up" shape)
//   =late   let the load build its pools and let the arena size, then refuse the FIRST bounce-ring grow
//           after that -- the field's sequence: the pool works, a later fill needs one new ring entry, the
//           driver says no, the tier latches off, and every pool that already exists must keep serving from
//           the memory it already owns
//   =0/unset off.
// Diagnostic only: nothing here changes a default, and it is not part of the shipped environment surface.
static bool pinned_alloc_fault(bool grow) {
    static const char * mode = getenv("MOE_HOST_POOL_PIN_FAULT");
    static const int    kind = mode == nullptr ? 0
                             : (strcmp(mode, "late") == 0 ? 2 : (atoi(mode) != 0 ? 1 : 0));
    // `late` waits until the run holds its pinned tier AND has sized the arena, because that is the field's
    // shape: the pools and the arena exist, and a later grow is what is refused.  Refusing the very first
    // grow would instead reproduce the `=1` shape with extra steps.
    return kind == 1 || (kind == 2 && grow && g_sized);
}

// Is the latched-refusal SEMANTICS being emulated on a platform where production code does not latch (POSIX,
// where a refused pinned allocation is transient)?  Any non-zero `MOE_HOST_POOL_PIN_FAULT` asks for the
// Windows rule, which is that the tier refuses to GROW once it has been refused once.
static bool pinned_latch_emulated() {
    static const char * mode = getenv("MOE_HOST_POOL_PIN_FAULT");
    return mode != nullptr && strcmp(mode, "0") != 0;
}

// `MOE_HOST_POOL_PIN=0` (r233, issue #278): close the pinned host tier for the whole run on ANY platform --
// the same state a Windows driver refusal latches into.  Default on; this only ever DISABLES, so it is a
// kill-switch, not an opt-in.  It exists because on Windows the loaded HIP runtime may be the display
// driver's copy in System32, which refuses a pinned allocation at ~9.3 GiB of adapter shared usage and can
// spin in its own wait loop afterwards (see `ggml_hip_runtime_module_info`); a box stuck on that runtime can
// still run the host-expert path (correct, read through the mapping) without touching the pinned tier.
static bool pinned_tier_disabled() {
    static const bool off = [] {
        const char * e = getenv("MOE_HOST_POOL_PIN");
        return e != nullptr && atoi(e) == 0;
    }();
    return off;
}

static host_pool_t * table_pool_locked(table_t & t) {
    if (!g_pool_enabled || !t.src_known || t.host == nullptr || t.host_bytes == 0 || t.n_experts <= 0) {
        return nullptr;
    }
    // An ALREADY-BUILT pool is looked up BEFORE the latch guard below, and that order is the point: the
    // latch stops the pinned tier from GROWING, it does not throw away memory the driver has already
    // granted.  `bounce_acquire` has always drawn that line (it reuses a drained ring entry before its own
    // `g_pinned_off` check); this lookup did not, so ONE late refusal made every pool that had already been
    // built unreachable -- and every fill from then on went through the pageable model mapping, i.e. a
    // driver-staged copy, which is the exact thing the pool exists to avoid, on the exact box where the
    // pinned budget is the scarce resource.  Field shape (issue #278): the pools are built at load, the
    // ceiling is reached during the prefill, and from the latch on every expert copy is pageable.  A pool
    // slot that exists is already pinned, so serving out of it asks the driver for nothing.
    const void * key = t.host;
    const auto it = g_host_pools.find(key);
    if (it != g_host_pools.end()) {
        return it->second.failed ? nullptr : &it->second;
    }
#if defined(_WIN32)
    if (g_pinned_off) {
#else
    if (g_pinned_off && (pinned_latch_emulated() || pinned_tier_disabled())) {
        // POSIX: a refused pinned allocation is transient, so production code keeps trying (the guard above is
        // Windows-only).  The `MOE_HOST_POOL_PIN_FAULT` diagnostic asks for the Windows latch semantics here,
        // and `MOE_HOST_POOL_PIN=0` asks for the tier to be closed from the start on either platform.
#endif
        // issue #274 follow-up (@DanoPTT): the pinned tier was refused once (a latched HIP error on this
        // platform, see `g_pinned_off`) and a pool is entirely pinned memory, so no NEW pool can be built --
        // and asking again is what turned one refusal into a cascade of differently-worded failures.  A
        // table without one then takes the per-table bounce ring / model-mapping path, which is the same
        // policy the arena follows when an allocation fails.  One line, once.
        static bool warned = false;
        if (!warned) {
            warned = true;
            GGML_LOG_WARN("%s: the pinned host tier is OFF for this run, so no NEW host pool can be built; "
                          "layer=%d role=%s falls back to the bounce ring / model mapping, and so will every "
                          "table that has no pool yet (pools built before a refusal keep serving).  See the "
                          "pinned-tier line above for the cause.\n",
                          __func__, t.layer, t.role.c_str());
        }
        return nullptr;
    }
    // Process-wide: every distinct host tensor shares the one budget, regardless of which device owns
    // its table(s).
    std::set<const void *> hosts;
    for (const table_t & tt : g_tables) {
        if (tt.src_known && tt.host != nullptr) {
            hosts.insert(tt.host);
        }
    }
    const size_t n_hosts  = hosts.empty() ? 1 : hosts.size();
    // Budget: explicit MOE_HOST_POOL_MIB, else AUTO = MOE_HOST_POOL_FRAC % of the MoE host expert bytes.
    // The preflight accumulation is the full set; fall back to the registered tables if the preflight did
    // not run (a non-CLI consumer).  `g_pool_total_bytes` is set only for the explicit mode.
    //
    // CRITICAL: the budget must not depend on the REGISTRATION ORDER.  A pool is created lazily, on the
    // first fill of its host tensor, and that happens while the cache is still sizing (only a few tables
    // registered), so dividing the budget by the currently-known host tensor count over-sized every early
    // pool: measured `--host-experts pool` pinning ~75 % of the host expert set instead of 25 % (and then
    // the AUTO budget was a running sum on top).  Each tensor therefore gets the SAME SHARE OF ITS OWN
    // EXPERTS -- `frac %` of the host set in AUTO mode, or `MOE_HOST_POOL_MIB` scaled by this tensor's
    // share of the total -- which is exactly order-independent.
    int64_t host_total = g_preflight_host_bytes;
    if (host_total <= 0) {
        std::set<const void *> hs;
        for (const table_t & tt : g_tables) {
            if (tt.src_known && tt.host != nullptr && hs.insert(tt.host).second) {
                host_total += (int64_t) tt.host_bytes * tt.n_experts;
            }
        }
    }
    int slots = 0;
    int64_t planned_pool = g_pool_total_bytes > 0 ? g_pool_total_bytes
                                                 : (host_total * g_pool_frac) / 100;
    // TODO #87 (1): decide the fill mode before ANY read happens, from the process-wide numbers.
    pool_decide_dio_locked(host_total, planned_pool);
    if (g_pool_total_bytes > 0) {
        // explicit process-wide budget, split proportionally to each tensor's share of the host set
        if (host_total > 0) {
            slots = (int) ((int64_t) t.n_experts * g_pool_total_bytes / host_total);
        } else {
            slots = (int) (((int64_t) g_pool_total_bytes / (int64_t) n_hosts) / (int64_t) t.host_bytes);
        }
    } else if (g_pool_sizing == 1) {
        // wip/host-pool-speed: per-table sizing from routing concentration, same total byte budget.
        pool_fit_cov_locked();
        const auto cs = g_pool_cov_slots.find(key);
        slots = cs != g_pool_cov_slots.end() ? cs->second : (int) ((int64_t) t.n_experts * g_pool_frac / 100);
    } else {
        slots = (int) ((int64_t) t.n_experts * g_pool_frac / 100);
    }
    if (g_pool_total_bytes <= 0 && g_pool_sizing == 1 && g_pool_cov_bytes > 0) {
        // The fit is the plan that ran: the ceiling's denominator must be what it actually planned, or a
        // ceiling-fitted `cov` plan is scaled a second time here.
        planned_pool = g_pool_cov_bytes;
    }
    if (slots < 1) {
        slots = 1;   // a pool that cannot hold one expert cannot replace the master
    }
    if (slots > t.n_experts) {
        slots = t.n_experts;
    }
    if (g_pool_sink && slots >= t.n_experts) {
        g_sink_mirror_tables++;   // a full mirror: no exchange, no reserve, no demotions
    } else if (g_pool_sink && (int64_t) (t.n_experts - slots) * 25 < (int64_t) t.n_experts) {
        g_sink_near_mirror_tables++;   // gap < 4 % of the set: exchange DISENGAGED (TODO #281)
    }
    slots = pool_clamp_slots_to_ram(slots, t.host_bytes, planned_pool, t);
    GGML_LOG_INFO("%s: host pool (%s): %zu host tensors, %lld MiB host expert bytes; layer=%d role=%s -> "
                  "%d/%d slots of %zu B (%.1f MiB)\n",
                  __func__,
                  g_pool_total_bytes > 0 ? "MOE_HOST_POOL_MIB" : "MOE_HOST_POOL_FRAC of the host expert bytes",
                  n_hosts, (long long) (host_total >> 20), t.layer, t.role.c_str(), slots, t.n_experts,
                  t.host_bytes, (double) ((int64_t) slots * (int64_t) t.host_bytes) / (1024.0 * 1024.0));
    device_guard pdg(t.device);
    host_pool_t p;
    if (slots >= 1 && pinned_alloc_fault(/*grow*/ false)) {
        // Diagnostic `MOE_HOST_POOL_PIN_FAULT` (issue #278): the platform refuses the pinned pool, exactly as
        // a Windows box does when WDDM charges it against the adapter's shared-segment budget.
        p.failed = true;
        if (!g_pinned_off) {
            g_pinned_off       = true;
            g_pinned_alloc_err = cudaErrorMemoryAllocation;
            GGML_LOG_WARN("%s: MOE_HOST_POOL_PIN_FAULT: refusing the pinned host pool for layer=%d role=%s "
                          "(%.1f MiB) to emulate the platform's latched refusal (issue #278); the pinned host "
                          "tier is now OFF for the rest of this run\n", __func__, t.layer, t.role.c_str(),
                          (double) slots * (double) t.host_bytes / (1024.0 * 1024.0));
        }
    } else if (slots >= 1) {
        void * mem = nullptr;
        // PORTABLE + MAPPED pinned memory: the pool is process-wide, so a device other than the one that
        // allocated the pool copies out of it.  Plain `cudaMallocHost` is tied to the allocating ROCm
        // context; a cross-device H2D from it falls back to a pageable staging memmove that crashes on a
        // 2-D strided copy.  The mapped UVA alias is copied from with cudaMemcpyDeviceToDevice instead.
        //
        // wip/host-expert-pool-region: `MOE_HOST_POOL_ALLOC` is the A/B for the pool's MEMORY TYPE, because
        // the pinned master a pool must match is a plain `cudaMallocHost` (ggml_cuda_host_malloc) while the
        // pool is `cudaHostAlloc(Portable|Mapped)`.  On ROCm those are different driver objects and can be
        // mapped with different caching for the device, which would show up as per-byte cost on the in-place
        // cold reads (18 % of accesses, and that fraction DECAYS over a run as the arena warms, i.e. exactly
        // the observed "worse starting state that converges").  `plain` = cudaMallocHost (the master's
        // shape), `mapped` = cudaHostAllocMapped only, `portable` = the historical pair (default).
        static const char * pool_alloc = [] { const char * e = getenv("MOE_HOST_POOL_ALLOC"); return e != nullptr ? e : "thp"; }();
        const size_t pool_bytes = (size_t) slots * t.host_bytes;
        cudaError_t  alloc_err  = (cudaError_t) -1;   // for HIP, whose enum has no `cudaErrorUnknown`
        /**/ if (strcmp(pool_alloc, "thp") == 0) {
            mem = pool_mmap_thp_alloc(pool_bytes);
            if (mem != nullptr) {
                p.mmap_bytes = pool_bytes;
                alloc_err    = cudaSuccess;
            } else {
                // No `mmap`/THP (a non-POSIX build, or the reservation was refused): fall back to the
                // historical shape rather than leaving the host tier off.
                alloc_err = cudaHostAlloc(&mem, pool_bytes, cudaHostAllocPortable | cudaHostAllocMapped);
            }
        }
        else if (strcmp(pool_alloc, "plain") == 0) {
            alloc_err = cudaMallocHost(&mem, pool_bytes);
        } else if (strcmp(pool_alloc, "mapped") == 0) {
            alloc_err = cudaHostAlloc(&mem, pool_bytes, cudaHostAllocMapped);
        } else {
            alloc_err = cudaHostAlloc(&mem, pool_bytes, cudaHostAllocPortable | cudaHostAllocMapped);
        }
        if (alloc_err == cudaSuccess) {
            p.base        = (char *) mem;
            g_pinned_tier_bytes += (int64_t) pool_bytes;
            void * dev    = nullptr;
            if (p.mmap_bytes == 0) {
                if (cudaHostGetDevicePointer(&dev, mem, 0) != cudaSuccess) {
                    (void) cudaGetLastError();
                    dev = mem;
                }
            }
            p.base_dev    = dev;   // an mmap pool gets its alias from `pool_mmap_thp_pin`, after the fill
            p.slots       = slots;
            p.host_bytes  = t.host_bytes;
            p.n_experts   = t.n_experts;
            p.path        = t.src_path;
            p.file_offs   = t.src_offs;
            p.src_host    = t.host;
            p.slot_expert.assign((size_t) slots, -1);
            p.slot_state.assign((size_t) slots, 0);
            p.slot_demoted.assign((size_t) slots, 0);
            p.last.assign((size_t) t.n_experts, 0);
            p.slot_ev.assign((size_t) slots * GGML_CUDA_MAX_DEVICES, nullptr);   // lazily created per device
        } else {
            (void) cudaGetLastError();
            p.failed = true;
            // issue #274 follow-up: this used to be SILENT, so a pool that could not be pinned looked
            // exactly like a pool that was never requested (the reporter's "the pool never appeared").
            // Report the real allocation error once, with the size that was asked for.
            static bool warned_pool = false;
            if (!warned_pool) {
                warned_pool = true;
                GGML_LOG_WARN("%s: the pinned host pool could not be allocated (%.1f MiB for layer=%d "
                              "role=%s): %s.  %.1f MiB of the pinned tier was allocated before this refusal; "
                              "the table falls back to the per-table bounce ring / model mapping (correct, "
                              "slower).\n", __func__, (double) pool_bytes / (1024.0 * 1024.0), t.layer,
                              t.role.c_str(), cudaGetErrorString(alloc_err),
                              (double) g_pinned_tier_bytes / (1024.0 * 1024.0));
            }
#if defined(_WIN32)
            // On Windows a refused pinned allocation is not transient -- it can latch a process-wide HIP
            // error (ROCm/TheRock#9339), so the whole pinned tier is turned off for the run, exactly as the
            // bounce ring's first refusal does.
            if (!g_pinned_off) {
                g_pinned_off       = true;
                g_pinned_alloc_err = alloc_err;
            }
#endif
        }
    } else {
        p.failed = true;
    }
    const auto res = g_host_pools.emplace(key, std::move(p));
    if (!res.first->second.failed && g_pool_prewarm) {
        pool_prepopulate_locked(res.first->second, t);   // DIO-fill the slots now, ranked by the prefill routing
    }
    if (!res.first->second.failed && res.first->second.mmap_bytes > 0 &&
            !pool_mmap_thp_pin(res.first->second)) {
        // The pool's device alias is what the in-place cold read and the cross-device fill source need, so
        // an unpinned pool must not be handed to the scheduler: fall back to the per-table bounce ring.
        GGML_LOG_WARN("%s: could not pin the mmap-backed host pool of layer=%d role=%s (%.1f MiB); "
                      "falling back to the per-table bounce ring\n",
                      __func__, t.layer, t.role.c_str(),
                      (double) res.first->second.mmap_bytes / (1024.0 * 1024.0));
        res.first->second.failed = true;
    }
    return res.first->second.failed ? nullptr : &res.first->second;
}

// ADDITIVE lifecycle: free every pool slot whose arena copy has drained.  The pool holds the experts the
// arena does NOT, so once a block has been copied into the arena its pool entry is dead weight.  Slots
// still in transit (state 2) are skipped; non-blocking.  Caller holds `g_mutex`.
static void pool_reap_locked(host_pool_t & p) {
    for (int s = 0; s < p.slots; s++) {
        if (p.slot_state[(size_t) s] != 2) {
            continue;
        }
        bool drained = true;
        for (int d = 0; d < GGML_CUDA_MAX_DEVICES; d++) {
            const size_t idx = (size_t) s * GGML_CUDA_MAX_DEVICES + (size_t) d;
            if (idx >= p.slot_ev.size() || p.slot_ev[idx] == nullptr) {
                continue;
            }
#if defined(GGML_USE_HIP)
            const bool ready = hipEventQuery((hipEvent_t) p.slot_ev[idx]) != hipErrorNotReady;
#else
            const bool ready = cudaEventQuery(p.slot_ev[idx]) != cudaErrorNotReady;
#endif
            if (!ready) {
                drained = false;
                break;
            }
            (void) cudaGetLastError();
        }
        if (!drained) {
            continue;
        }
        const int32_t e = p.slot_expert[(size_t) s];
        if (e >= 0) {
            p.expert_slot.erase(e);
        }
        p.slot_expert[(size_t) s] = -1;
        p.slot_state[(size_t) s]  = 0;
    }
}

// wip/host-expert-master-drop (TODO #55): fill one pool slot with expert `expert`'s whole host blob.
// When the host tensor is backed by the read-only model mapping (the pool-backed master) that is a plain
// memcpy out of the page cache: no syscall, no second copy, and the pages stay reclaimable (the kernel drops
// them under pressure and re-faults from the GGUF -- the disk tier).  `MOE_HOST_POOL_DIO=1` keeps the file
// read with O_DIRECT, which is the only mode that bypasses the page cache.  The disk tier is an OPTIMISATION:
// when it is unavailable (a platform whose pool file I/O is a stub, a file that cannot be opened) the fill
// falls back to the mapping instead of failing, because failing here marks the whole pool dead (issue #274).
// Caller holds `g_mutex`.
static bool pool_fill_slot_locked(host_pool_t & p, int expert, char * dst) {
    // wip/host-pool-speed: the fill's WALL TIME, by source.  A demand fill is synchronous (the host thread
    // blocks on it), so this is the number that decides whether an overlapped/coalesced fill is worth
    // building at all (README §16).  Two clock reads per fill: fills are ~1 per miss, so this is free.
    const int64_t t0 = ggml_time_us();
    const size_t rel = (size_t) expert * p.host_bytes;
    const bool   use_disk = g_pool_dio && pool_file_io_available();
    if (!use_disk && p.src_host != nullptr) {
        memcpy(dst, (const char *) p.src_host + rel, p.host_bytes);
        g_fill_memcpy_calls++;
        g_fill_memcpy_us += ggml_time_us() - t0;
        g_fill_total_bytes += (int64_t) p.host_bytes;
        return true;
    }
    if (p.fd < 0) {
        p.fd = pool_open_file(p.path.c_str(), g_pool_dio);
        if (p.fd < 0) {
            if (p.src_host == nullptr) {
                return false;   // no source at all: a pool that cannot be filled must stay failed
            }
            // The disk tier is gone but the master is mapped: read it directly.  A one-time line, because
            // this is the arm that used to fail the pool outright (issue #274).
            static bool warned_fallback = false;
            if (!warned_fallback) {
                warned_fallback = true;
                GGML_LOG_WARN("%s: the host pool's disk tier is unavailable (%s); filling through the model "
                              "mapping / page cache instead.  The pool contents are identical, only the page "
                              "cache is populated.\n", __func__, g_pool_dio ? "O_DIRECT requested" : "no file handle");
            }
            memcpy(dst, (const char *) p.src_host + rel, p.host_bytes);
            g_fill_memcpy_calls++;
            g_fill_memcpy_us += ggml_time_us() - t0;
            g_fill_total_bytes += (int64_t) p.host_bytes;
            return true;
        }
        pool_fd_align(p.fd, &p.align);
    }
    const size_t off = p.file_offs + rel;
    bool ok = g_pool_dio ? pool_dio_read(p.fd, dst, p.host_bytes, off, p.align)
                         : (pool_pread(p.fd, dst, p.host_bytes, off) == (pool_ssize_t) p.host_bytes);
    if (!ok && p.src_host != nullptr) {
        // A failed disk read (a short pread, an O_DIRECT alignment refusal) is not a reason to fail the fill
        // OR the pool: the master holds the same bytes.  Retry through the mapping and say so once.
        static bool warned_read = false;
        if (!warned_read) {
            warned_read = true;
            GGML_LOG_WARN("%s: a host pool disk read failed (dio=%d); falling back to the model mapping for "
                          "the slot.  The pool stays live.\n", __func__, (int) g_pool_dio);
        }
        memcpy(dst, (const char *) p.src_host + rel, p.host_bytes);
        g_fill_memcpy_calls++;
        ok = true;
    }
    g_fill_read_calls++;
    g_fill_read_us += ggml_time_us() - t0;
    g_fill_total_bytes += (int64_t) p.host_bytes;
    // Diagnostic (`MOE_HOST_POOL_VERIFY=1`): the slot must hold exactly the expert's bytes as they appear in
    // the file mapping.  A mismatch means the pool read landed on the wrong offset/length -- the classic
    // O_DIRECT alignment bug -- and would otherwise surface as corrupt generated text much later.
    static const bool verify_pool = getenv("MOE_HOST_POOL_VERIFY") != nullptr;
    if (ok && verify_pool && p.src_host != nullptr) {
        static int64_t n_bad = 0;
        const char * ref = (const char *) p.src_host + rel;
        if (memcmp(dst, ref, p.host_bytes) != 0 && n_bad++ < 8) {
            size_t first = 0;
            while (first < p.host_bytes && dst[first] == ref[first]) {
                first++;
            }
            GGML_LOG_ERROR("POOLREAD MISMATCH expert=%d off=%zu len=%zu align=%zu dio=%d first=%zu got=%02x ref=%02x file_offs=%zu\n",
                           expert, off, p.host_bytes, p.align, (int) g_pool_dio, first,
                           (unsigned char) dst[first], (unsigned char) ref[first], p.file_offs);
        }
    }
    return ok;
}

// TODO #87: the DEMAND path never evicts a pool entry.  The pool and the arena are two tiers of one bounded
// cache (pinned host RAM / VRAM), so the only way a demand miss may change the pool is by consuming an
// EMPTY slot; a full pool means both tiers already cover what they can, and the miss is served from the
// model mapping / the disk (or, on the decode band, swapped with the arena victim).  That is what stops the
// old behaviour -- every miss evicting some other resident -- from turning one cold token into a storm of
// re-reads.  The eviction-capable helper below stays for the load-time prewarm and the one-shot
// prompt-tally re-rank (both of which build the working set once, off the demand path).

// A resident slot for `expert` (state 1), or null.  No state change beyond the LRU clock.
static char * pool_peek_locked(host_pool_t & p, int expert, int * out_slot) {
    if (expert < 0 || expert >= p.n_experts) {
        return nullptr;
    }
    const auto hit = p.expert_slot.find(expert);
    if (hit == p.expert_slot.end()) {
        return nullptr;
    }
    const int hs = hit->second;
    if (hs < 0 || hs >= p.slots || p.slot_state[(size_t) hs] != 1) {
        return nullptr;   // never published, or retiring (state 2)
    }
    p.hits++;
    if ((int) p.slot_demoted.size() == p.slots && p.slot_demoted[(size_t) hs] != 0) {
        g_sink_uses++;   // a demoted victim was wanted again: the demote earned its bandwidth
    }
    p.last[(size_t) expert] = ++p.clock;
    if (out_slot != nullptr) {
        *out_slot = hs;
    }
    return p.base + (size_t) hs * p.host_bytes;
}

// A resident-slot test that touches NO policy state (no hit counter, no LRU clock): the arena exchange
// asks "does the pool still hold this victim?" without making the victim look demanded.
static bool pool_resident_locked(const host_pool_t & p, int expert) {
    if (expert < 0 || expert >= p.n_experts) {
        return false;
    }
    const auto it = p.expert_slot.find(expert);
    if (it == p.expert_slot.end()) {
        return false;
    }
    const int s = it->second;
    return s >= 0 && s < p.slots && p.slot_state[(size_t) s] == 1;
}

// r143: drop a host pool's pinned storage.  Called only when the model that owned its master is going
// away, i.e. with no graph in flight (the caller synchronizes first): every slot is dead, the per-slot
// events are destroyed and the lazily opened source file is closed.  The map entry itself is erased by
// the caller (`g_host_pools` is the leaked container, not the entries).
static void pool_free_locked(host_pool_t & p) {
    for (cudaEvent_t ev : p.slot_ev) {
        if (ev != nullptr) {
            (void) cudaEventDestroy(ev);
        }
    }
    p.slot_ev.clear();
    if (p.base != nullptr) {
        // r215: an `mmap`+THP pool is released with unregister+munmap.  `cudaHostUnregister` ALONE does not
        // give the pages back on this stack (measured in `archive/work/host-expert-eager-push/`, P4 §5:
        // 2.0 of 256 MiB re-readable after unregister, i.e. still pinned; and it is not deferred) -- the pin
        // is tied to the VMA, so the `munmap` is what releases it.  Verified on teardown: AnonHugePages
        // 48.97 GiB -> 0 and MemAvailable back to baseline.
        if (p.mmap_bytes > 0) {
            pool_mmap_thp_release(p);
        } else {
            (void) cudaFreeHost(p.base);
        }
        p.base     = nullptr;
        p.base_dev = nullptr;
    }
    if (p.fd >= 0) {
        pool_close_file(p.fd);
        p.fd = -1;
    }
    (void) cudaGetLastError();   // fail soft: a bad free must not abort a model teardown
    p.slots = 0;
    p.failed = true;
}

// The first slot that holds no expert and is not being retired/reserved (-1 when the pool is full).
// `keep_free` is the number of slots the caller must leave untouched.  W2's exchange passes 0 (it may take
// the last one) while every fill path passes the reserve, so the headroom exists for the demote instead of
// being immediately consumed by the next incoming expert.  Measured why this matters: reserving headroom by
// shrinking `slots` instead does nothing at all, because `slots` is the capacity everywhere and the fill
// paths simply fill the smaller pool -- the decline rate was identical at K=2 and K=8.
// The exchange must be INERT against a full mirror.  When the pool can hold every expert of a table its
// coverage is exact, so there is nothing for a demote to fix and the copies are pure cost -- and worse, a
// demote there would find no free slot, ask for one, and RETIRE a resident to make room, destroying the
// mirror it was supposed to improve.  `pool + headroom <= a full mirror` holds by construction: `slots` is
// clamped to `n_experts` at sizing, so the headroom always comes out of that budget and never adds to it.
// TODO #281: a NEAR-MIRROR pool disengages the exchange.  Measured on a 59 GiB field box at `-lme auto`
// (97.7 % coverage = a 12-expert gap per 512-expert table, one R9700): decode 75.6 (`SINK=0`) against 76.2
// (`SINK=1`) -- no gain -- while the first-pass 50k prefill fell from 1 238 to 708 t/s, because the decode-band
// demote moves the pool's content away from the complement the next prefill has to read (the gather's
// "neither tier" volume went 10.1 k -> 32.5 k experts).  At `-lme 44` (93.7 %, a 32-expert gap) the same demote
// is worth **+21 %** decode and +10 % prefill.  So the boundary is the GAP, not the budget: below 4 % of the set
// the pool is a near-mirror whose few holes are cheaper to pay than to learn.
static bool pool_sink_near_mirror(const host_pool_t * p) {
    if (p == nullptr || p->n_experts <= 0 || p->slots >= p->n_experts) {
        return false;
    }
    return (int64_t) (p->n_experts - p->slots) * 25 < (int64_t) p->n_experts;   // gap < 4 % of the set
}

static bool pool_sink_active(const host_pool_t * p) {
    return g_pool_sink && p != nullptr && p->n_experts > 0 && p->slots < p->n_experts &&
           !pool_sink_near_mirror(p);   // TODO #281
}

// The exchange headroom for one pool: an absolute override if one was given, else ~3 % of that table's
// slots with a floor of 2 and a 1/8 cap (a reserve that large would starve the layout).
//
// TODO #281: the headroom is a DECODE-BAND notion, and it is capped by the pool's coverage gap.  Measured on a
// 59 GiB field box at `-lme auto` (97.7 % coverage, one R9700): the derived K was ~20 slots/table against a
// 12-expert gap, so the reserve held free the very slots the complement needed, every prefill fill was refused,
// and each prefill ubatch re-read the missing experts through the bounce ring -- first-pass 50k prefill
// 1837 t/s with `MOE_HOST_POOL_SINK=0` against 867 with it, decode flat (76.6 vs 76.1).  Two rules follow and
// both are load-bearing:
//   * a PREFILL fill passes `prefill=true` and ignores the headroom entirely (`pool_fill_empty_locked`,
//     `stage_span_fill`): it is the coverage path, and leaving slots free for a demote is meaningless while a
//     ubatch is staging the whole host table, and
//   * K never exceeds the gap, so a near-mirror pool cannot reserve slots it could never refill.
static int pool_sink_reserve(const host_pool_t & p) {
    if (!pool_sink_active(&p)) {
        return 0;   // no reserve against a mirror: the layout may use every slot
    }
    int k = g_pool_sink_spare;
    if (k < 0) {
        k = (int) ((int64_t) p.slots * g_pool_sink_spare_pct / 100);
        if (k < g_pool_sink_spare_min) {
            k = g_pool_sink_spare_min;
        }
    }
    if (k > p.slots / 8) {
        k = p.slots / 8;
    }
    const int gap = p.n_experts - p.slots;
    if (gap > 0 && k > gap) {
        k = gap;   // TODO #281: a reserve larger than the gap starves the complement it is meant to protect
    }
    return k;
}

static int pool_free_slot_locked(host_pool_t & p, int keep_free = 0) {
    pool_reap_locked(p);
    // Fast path: no reserve asked for, so the first free slot is the answer.  This MUST stay a single scan --
    // counting free slots first costs O(slots) twice on every fill (56 592 slots here, ~10^9 iterations over a
    // run) and it slowed the batched arms measurably at npl 8 while changing no decision at all.
    if (keep_free <= 0) {
        for (int s = 0; s < p.slots; s++) {
            if (p.slot_expert[(size_t) s] < 0) {
                return s;
            }
        }
        return -1;
    }
    int n_free = 0;
    for (int s = 0; s < p.slots; s++) {
        if (p.slot_expert[(size_t) s] < 0 && ++n_free > keep_free) {
            return s;   // more than `keep_free` spare: this one is usable
        }
    }
    return -1;
}

// Publish `expert` as the resident content of `slot` (replacing whatever the map said it held).  The SLOT
// BYTES are the caller's business -- used by `pool_fill_empty_locked` (filled from the file/mapping) and by
// the arena->pool exchange in `access_locked` (filled from the victim's arena slot).
static void pool_slot_assign_locked(host_pool_t & p, int slot, int32_t expert) {
    if (slot < 0 || slot >= p.slots || expert < 0) {
        return;
    }
    const int32_t old = p.slot_expert[(size_t) slot];
    if (old >= 0) {
        const auto it = p.expert_slot.find(old);
        if (it != p.expert_slot.end() && it->second == slot) {
            p.expert_slot.erase(it);
        }
    }
    const auto pre = p.expert_slot.find(expert);
    if (pre != p.expert_slot.end() && pre->second != slot) {
        p.expert_slot.erase(pre);   // never leave two indices for one expert
    }
    p.slot_expert[(size_t) slot] = expert;
    p.expert_slot[expert]        = slot;
    p.slot_state[(size_t) slot]  = 1;
    if ((int) p.slot_demoted.size() == p.slots) {
        p.slot_demoted[(size_t) slot] = 0;   // the demote path re-marks it straight after
    }
    p.last[(size_t) expert]      = ++p.clock;
}

// Fill the first EMPTY slot with `expert` and return its slot base (null when the pool is full or the read
// failed).  Never evicts.  `prefill` marks the PREFILL band's coverage fills, which ignore the exchange
// headroom (TODO #281 -- see `pool_sink_reserve`).  Caller holds `g_mutex`.
static char * pool_fill_empty_locked(host_pool_t & p, int expert, int * out_slot, bool prefill = false) {
    if (expert < 0 || expert >= p.n_experts || p.failed) {
        return nullptr;
    }
    if (p.expert_slot.find(expert) != p.expert_slot.end()) {
        return nullptr;   // already resident (or retiring): the caller must peek, not fill
    }
    const int slot = pool_free_slot_locked(p, prefill ? 0 : pool_sink_reserve(p));
    if (slot < 0) {
        return nullptr;
    }
    if (prefill && pool_sink_active(&p)) {
        g_sink_prefill_fills++;   // a coverage fill that the decode-band reserve would have refused
    }
    char * dst = p.base + (size_t) slot * p.host_bytes;
    if (!pool_fill_slot_locked(p, expert, dst)) {
        p.failed = true;
        return nullptr;
    }
    pool_slot_assign_locked(p, slot, expert);
    p.fills++;
    if (out_slot != nullptr) {
        *out_slot = slot;
    }
    return dst;
}

// Ensure `expert` is in the pool and return its slot base (and the slot index), or null when the pool is
// unusable (the caller then reads the full master / disk directly).  A miss is a memcpy out of the model
// mapping (or a buffered GGUF read).  This is the EVICTION-CAPABLE helper: only the load-time prewarm and
// the one-shot prompt-tally re-rank may call it (they build the working set once); the demand path uses
// `pool_peek_locked` + `pool_fill_empty_locked`.  Retiring slots (state 2, their arena copy is still in
// flight) are never reused or evicted.  Caller holds `g_mutex`.
static char * pool_slot_locked(host_pool_t & p, int expert, int * out_slot, bool sync_on_evict) {
    if (expert < 0 || expert >= p.n_experts) {
        return nullptr;
    }
    pool_reap_locked(p);
    const auto hit = p.expert_slot.find(expert);
    if (hit != p.expert_slot.end()) {
        const int hs = hit->second;
        if (hs >= 0 && hs < p.slots && p.slot_state[(size_t) hs] == 1) {
            p.hits++;
            p.last[(size_t) expert] = ++p.clock;
            if (out_slot != nullptr) {
                *out_slot = hs;
            }
            return p.base + (size_t) hs * p.host_bytes;
        }
        // State 2: retiring (its copy into the arena is still in flight).  Never duplicate the read; return
        // null so the caller serves this miss from the bounce instead.
        return nullptr;
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
            if (e < 0 || p.slot_state[(size_t) s] != 1) {
                continue;   // free, or retiring (its arena copy may still be reading it)
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
            if (sync_on_evict) {
                pool_slot_sync(p, slot);
            }
            p.expert_slot.erase(victim);
            p.slot_expert[(size_t) slot] = -1;
            p.evictions++;
        }
    }
    if (slot < 0) {
        return nullptr;
    }
    char *       dst = p.base + (size_t) slot * p.host_bytes;
    const bool ok = pool_fill_slot_locked(p, expert, dst);
    if (!ok) {
        p.failed = true;
        return nullptr;
    }
    p.slot_expert[(size_t) slot] = expert;
    p.expert_slot[expert]        = slot;
    p.slot_state[(size_t) slot]  = 1;
    if ((int) p.slot_demoted.size() == p.slots) {
        p.slot_demoted[(size_t) slot] = 0;   // the demote path re-marks it straight after
    }
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
// r215: a FULL-MIRROR pool (every expert of the tensor) is one contiguous allocation of
// `n_experts * host_bytes`, and the GGUF expert layout is expert-major at exactly that stride -- the slot
// for expert `e` is `base + e*host_bytes` at file offset `file_offs + e*host_bytes`.  So the whole mirror is
// ONE sequential pass over the tensor, the same read the `--load-mode none` host-master path issues,
// instead of one `host_bytes` memcpy per ranked slot out of the page-cache-backed model mapping.
//
// Measured (Qwen3.8-Flash-Next IQ3_S, `-ncmoe 48 -lm mmap`, cold page cache): the 48 GiB mirror went from
// ~10.8 s to ~7.4 s -- 4.5 -> 6.5 GB/s, the rate the direct load gets -- and it stops faulting a second
// 48 GiB into the page cache that cannot be kept beside the pinned mirror.  The fill ORDER is content-free
// here (`slots == n_experts`, so every expert is resident either way); the prior still decides the layout
// whenever the budget is smaller than the tensor, via the ranked path below.  `MOE_HOST_POOL_BULK=0`
// restores the per-slot fill (the A/B arm).
static bool pool_mirror_bulk_fill_locked(host_pool_t & p, const table_t & t) {
    if (p.slots != p.n_experts || p.host_bytes == 0 || p.fd >= 0 || p.base == nullptr || p.path.empty()) {
        return false;
    }
    p.fd = pool_open_file(p.path.c_str(), g_pool_dio);
    if (p.fd < 0) {
        return false;
    }
    pool_fd_align(p.fd, &p.align);
    const int64_t t0    = ggml_time_us();
    const size_t  total = (size_t) p.n_experts * p.host_bytes;
    const size_t  chunk = 32ull * 1024 * 1024;
    size_t        done  = 0;
    while (done < total) {
        const size_t want = std::min(chunk, total - done);
        // `pool_dio_read` handles the alignment (and the O_DIRECT bounce) when the disk tier is forced; the
        // default buffered arm is a plain bulk `pread`.
        const bool ok = g_pool_dio
                      ? pool_dio_read(p.fd, p.base + done, want, p.file_offs + done, p.align)
                      : pool_pread(p.fd, p.base + done, want, p.file_offs + done) == (pool_ssize_t) want;
        if (!ok) {
            // A half-filled mirror must never be handed to the scheduler: leave it unpublished and let the
            // caller fall back to the per-slot fill, which reports its own failures.
            GGML_LOG_WARN("%s: bulk mirror fill failed layer=%d role=%s at %zu/%zu B\n",
                          __func__, t.layer, t.role.c_str(), done, total);
            return false;   // the fd stays open: the per-slot fallback reuses it
        }
        done += want;
    }
    const int64_t us = ggml_time_us() - t0;
    for (int e = 0; e < p.n_experts; e++) {
        // slot == expert: the file layout IS the slot layout
        p.slot_expert[(size_t) e] = e;
        p.expert_slot[e]          = e;
        p.slot_state[(size_t) e]  = 1;
        p.last[(size_t) e]        = ++p.clock;
        p.fills++;
    }
    GGML_LOG_INFO("%s: host pool mirror layer=%d role=%s: %d experts, %.1f MiB sequential in %.0f ms "
                  "(%.0f MiB/s)\n", __func__, t.layer, t.role.c_str(), p.n_experts,
                  (double) total / (1024.0 * 1024.0), us / 1000.0,
                  (double) total / (1024.0 * 1024.0) / (double) std::max<int64_t>(us / 1000000, 1));
    return true;
}


// TODO #87: the ONE ranking that decides the host pool's content -- and therefore how much of the
// decode band is served from pinned RAM instead of the mapping/disk.  Order: the persisted long-horizon
// heat prior first (it is the only signal that has seen DECODE routing from previous sessions; a prefill
// tally is nearly flat, because a prefill routes every expert many times), then the in-run prefill tally
// as the tie-break, then expert id.  Measured offline on a real 10.5K-prompt trace (`scripts/moe-trace-
// replay.py --pool-frac 80 --pool-init sidecar-complement:<sidecar>`), same arena and pool:
//   pool ranked by expert id (the pre-r113 behaviour): h 0.9367, pool_h 0.7995, 3571 MiB L3 reads
//   pool ranked by the persisted heat prior:            h 0.9994, pool_h 0.3923,   237 MiB L3 reads
// i.e. the pool's CONTENT is worth ~15x the disk traffic, while every eviction/hand-over policy scores
// within 250 MiB of the others -- which is why the demand path below never churns it (r113).
static void pool_rank_experts_locked(const table_t & t, std::vector<int32_t> & out) {
    out.clear();
    out.reserve((size_t) t.n_experts);
    for (int e = 0; e < t.n_experts; e++) {
        out.push_back(e);
    }
    const bool have_heat  = (int) t.heat.size() == t.n_experts;
    const bool have_tally = (int) t.prefill_count.size() == t.n_experts;
    if (!have_heat && !have_tally) {
        return;   // expert id order (a cold start: nothing to rank by)
    }
    // wip/host-expert-split (2026-11-05): WHICH counter ranks the complement.  The historical choice is
    // `t.heat`, the combined prefill+decode counter -- defensible, because the pool also stages whole tables
    // for the prefill band.  But the ARENA seeds from the DECODE counter for exactly the reason this pool
    // exists (TODO #88), and measured on a Flash-Next prose trace the combined counter overlaps the
    // workload's own top-64 expert set at 10.5 % against a 12.5 % chance level -- WORSE than chance -- while
    // the decode counter is 24.2 %.  Modelled worth (same arena and pool, zero extra copies): 26.1 -> 23.3
    // MiB/token of L3 at `always` admission, 46.8 -> 39.0 at FRAC 40 (`wip/host-expert-split/
    // STEP1-REPLAY.md` §7).  Default 0 keeps the shipped behaviour byte-identical; the A/B is one env var.
    //   0 combined  `t.heat` (historical)
    //   1 decode    `heat_rank_vec()` -- the vector the arena seeds from
    //   2 floor     the prefill-hot `g_pool_rank_floor` % first (the staging duty), the rest by decode
    if (g_pool_rank != 0 || !have_heat) {
        const std::vector<uint32_t> & rv = heat_rank_vec(t);
        bool floor_hot_ok = false;
        uint32_t floor_thr = 0;
        if (g_pool_rank == 2 && have_heat && g_pool_rank_floor > 0) {
            std::vector<uint32_t> sorted = t.heat;
            const size_t keep = (size_t) t.n_experts * (size_t) std::min(100, g_pool_rank_floor) / 100;
            std::nth_element(sorted.begin(), sorted.begin() + (keep > 0 ? (long) keep - 1 : 0), sorted.end(),
                             std::greater<uint32_t>());
            floor_thr = keep > 0 ? sorted[keep - 1] : 0;
            floor_hot_ok = floor_thr > 0;   // an all-zero prior has no floor to keep
        }
        std::sort(out.begin(), out.end(), [&](int32_t a, int32_t b) {
            if (floor_hot_ok) {
                const bool ha = t.heat[(size_t) a] >= floor_thr;
                const bool hb = t.heat[(size_t) b] >= floor_thr;
                if (ha != hb) {
                    return ha;   // the prefill-hot floor stays ahead of the decode ranking
                }
            }
            if (rv[(size_t) a] != rv[(size_t) b]) {
                return rv[(size_t) a] > rv[(size_t) b];   // hottest first
            }
            if (have_tally && t.prefill_count[(size_t) a] != t.prefill_count[(size_t) b]) {
                return t.prefill_count[(size_t) a] > t.prefill_count[(size_t) b];
            }
            return a < b;
        });
        return;
    }
    std::sort(out.begin(), out.end(), [&](int32_t a, int32_t b) {
        if (have_heat) {
            const uint32_t ha = t.heat[(size_t) a];
            const uint32_t hb = t.heat[(size_t) b];
            if (ha != hb) {
                return ha > hb;
            }
        }
        if (have_tally) {
            const int64_t ta = t.prefill_count[(size_t) a];
            const int64_t tb = t.prefill_count[(size_t) b];
            if (ta != tb) {
                return ta > tb;
            }
        }
        return a < b;
    });
}

static void pool_prepopulate_locked(host_pool_t & p, const table_t & t) {
    static const bool bulk = [] { const char * e = getenv("MOE_HOST_POOL_BULK"); return e == nullptr || atoi(e) != 0; }();
    if (bulk && pool_mirror_bulk_fill_locked(p, t)) {
        return;
    }
    // Load-time warm: fill every slot from the ranked order.  It runs BEFORE the arena is sized/seeded,
    // so it can only use the persisted prior; `pool_rerank_from_heat_locked` re-derives the pool as the
    // arena's complement once the arena content is known.  `pool_slot_locked` is the eviction-capable
    // helper -- it is a bulk, one-shot layout decision, not demand churn.
    std::vector<int32_t> rank;
    pool_rank_experts_locked(t, rank);
    const int k = p.slots < p.n_experts ? p.slots : p.n_experts;
    int filled = 0;
    for (int i = 0; i < k; i++) {
        if (pool_fill_empty_locked(p, rank[(size_t) i], nullptr) == nullptr) {
            break;
        }
        filled++;
    }
    if (filled > 0) {
        const bool have_heat = (int) t.heat.size() == t.n_experts;
        GGML_LOG_INFO("%s: prewarmed host pool layer=%d role=%s: %d/%d slots (ranked by %s)\n",
                      __func__, t.layer, t.role.c_str(), filled, p.slots,
                      have_heat ? "the persisted heat prior" :
                      ((int) t.prefill_count.size() == t.n_experts ? "the prefill tally" : "expert id"));
    }
}

// TODO #87: put the pool where it belongs -- its slots hold the hottest experts the ARENA does not, ranked
// by `pool_rank_experts_locked`.  This is the layout the offline model says is worth ~15x the disk traffic
// (arena U pool then covers the set); it is a one-shot bulk decision at sizing (after the heat/prefill
// seeds) and at the first decode-band pass (after the prompt's routing is folded into the heat vector), so
// it may evict pool entries -- unlike the demand path, which never does.  Caller holds `g_mutex`.
static void pool_rerank_from_heat_locked(table_t & t) {
    if (t.n_experts <= 0 || t.identity) {
        return;   // an identity table holds everything: its pool is the empty complement
    }
    host_pool_t * pool = table_pool_locked(t);
    if (pool == nullptr || pool->failed || pool->slots <= 0) {
        return;
    }
    // Reap FIRST: a retirement whose copy has drained becomes a free slot (and drops its index), which is
    // what makes the coverage test below see the live set.  TODO #269 relies on it: the remap's cold ids are
    // built from this map, and "a covering pool's layout is write-once after sizing" is the invariant that
    // makes a one-call-late map safe without a device synchronize.
    pool_reap_locked(*pool);
    // A pool that COVERS the tensor has nothing to re-rank -- `want` is every expert, `keep` frees none and
    // the fill loop finds every expert resident, so the expert -> slot map would be unchanged.  Say so
    // instead of relying on that coincidence, and do it BEFORE any bookkeeping so no path (this one
    // included) can move a slot an in-flight remapped kernel may still be reading.
    if (pool->slots >= t.n_experts && pool_live_slots(*pool) >= t.n_experts) {
        return;
    }
    std::vector<int32_t> rank;
    pool_rank_experts_locked(t, rank);
    // Priority order, and it is ONE rule for every budget (wip/host-expert-pool-surplus):
    //   1. the COMPLEMENT -- every expert the arena does not hold -- in rank order, always first;
    //   2. then, while capacity remains, the arena-resident experts in the same rank order: the pool expands to
    //      mirror them, which is what "the budget is big enough" means once the complement is covered.
    // So `slots < |complement|` is a bounded complement (the historical layout), `slots == |complement|` is
    // exactly the complement, and `slots > |complement|` also keeps the hottest arena residents -- a full mirror
    // when the budget covers the tensor.  Un-loading an expert the arena holds forces its next re-admission to be
    // re-staged from the page cache (a synchronous 288 KiB memcpy), so it is only done when the complement needs
    // the capacity.
    std::vector<int32_t> want;
    want.reserve(rank.size());
    for (int32_t e : rank) {
        if (t.expert_slot.find(e) == t.expert_slot.end()) {
            want.push_back(e);
        }
    }
    for (int32_t e : rank) {
        if (want.size() >= (size_t) pool->slots) {
            break;
        }
        if (t.expert_slot.find(e) != t.expert_slot.end()) {
            want.push_back(e);
        }
    }
    const int n = std::min((int) want.size(), pool->slots);
    // 1. Free every slot whose expert is NOT wanted.  Freeing is bookkeeping only (no copy), and it MUST
    //    precede the fills: `pool_slot_locked`'s eviction picks the LRU, which while filling in rank order
    //    is the hottest entry just placed -- measured as a pool hit rate that got WORSE than expert order.
    std::unordered_set<int32_t> keep(want.begin(), want.begin() + n);
    int freed = 0;
    for (int sl = 0; sl < pool->slots; sl++) {
        const int32_t e = pool->slot_expert[(size_t) sl];
        if (e < 0 || pool->slot_state[(size_t) sl] != 1 || keep.count(e) != 0) {
            continue;
        }
        pool->expert_slot.erase(e);
        pool->slot_expert[(size_t) sl] = -1;
        pool->slot_state[(size_t) sl]  = 0;
        freed++;
    }
    // 2. Fill the wanted experts that are not resident (the pool was already prewarmed from the same
    //    prior, so on a warm start this is a handful of reads; slots freed in 1. are reused, never evicted).
    //
    // r231: the fills are issued in ASCENDING EXPERT ID, not in rank order.  One table's experts are
    // contiguous in the file (`src_offs + e*host_bytes`), so expert order IS file order, and `want` is in
    // rank order -- i.e. this loop used to walk the file at random.  It issues 1 500-6 400 single-expert
    // fills per band transition, all of it charged to the request that triggers the re-layout (measured on
    // the 56 GiB rig: 13.5-18.6 s of a 50k first pass at `-lme 44`, 0.6 s on a mirror, which never
    // re-lays-out).  File order recovers ~30 % of that (13.5-18.6 s -> 9.4-9.7 s, and the generated text is
    // unchanged); the rest is not read latency but page-cache/reclaim pressure under the memory cap -- a
    // microbenchmark of the same fill shape is 0.7-1.6 ms per expert, and `MOE_HOST_POOL_DIO=1` (which reads
    // around the page cache) removes ~90 % of the tail, which is the fill SOURCE's business, not this
    // loop's.  The SLOT an expert lands in is arbitrary by construction (`pool_fill_empty_locked` takes the
    // first empty slot), so this changes nothing but the order the reads are issued in.
    static const bool order_offset = [] {
        const char * e = getenv("MOE_HOST_POOL_FILL_ORDER");
        return e == nullptr || strcmp(e, "rank") != 0;   // `rank` = the pre-r231 order, for the A/B
    }();
    std::vector<int32_t> todo;
    todo.reserve((size_t) n);
    for (int i = 0; i < n; i++) {
        const int32_t e = want[(size_t) i];
        if (pool->expert_slot.find(e) == pool->expert_slot.end()) {
            todo.push_back(e);
        }
    }
    if (order_offset) {
        std::sort(todo.begin(), todo.end());
    }
    int filled = 0;
    for (int32_t e : todo) {
        if (pool_fill_empty_locked(*pool, e, nullptr) == nullptr) {
            break;
        }
        filled++;
    }
    if (freed > 0 || filled > 0) {
        GGML_LOG_INFO("%s: host pool laid out as the arena's %s layer=%d role=%s: "
                      "%d filled, %d freed, %zu wanted non-resident, ranked by the %s prior\n",
                      __func__, (int) want.size() > t.n_experts - (int) t.expert_slot.size()
                                ? "complement + mirrored residents" : "complement",
                      t.layer, t.role.c_str(), filled, freed, want.size(),
                      g_pool_rank == 1 ? "DECODE" : (g_pool_rank == 2 ? "prefill-floor + DECODE" : "combined"));
    // (`invert` -- the ascending-heat inversion probe that established the counter direction -- is deliberately
    // GONE as of 2026-11-06: it was a diagnostic, never a product option, and it must not ship.)
    }
}


// TODO #64.1 / #87: give the host pool the prompt's warm order.  `prefill_count` is only accumulated by
// the EAGER prefill path; the default staged prefill intercepts the expert upload before that hook, so the
// host tally stayed empty.  The DEVICE prefill tally does see every prefill, so fold it into the heat
// vector once (at the first decode-band pass after a prefill) and re-derive the pool's layout from it --
// combined with the persisted prior, which is what ranks better than a flat prompt tally (r113).  Caller
// holds `g_mutex`.
static void pool_rank_from_prefill_locked() {
    heat_pull_locked();   // one pass over the tables: fold + zero every device mirror
    for (table_t & t : g_tables) {
        if (t.n_experts <= 0 || t.identity || (int) t.heat.size() != t.n_experts) {
            continue;
        }
        bool any = false;
        for (uint32_t v : t.heat) {
            if (v != 0) {
                any = true;
                break;
            }
        }
        if (!any) {
            continue;
        }
        if ((int) t.prefill_count.size() != t.n_experts) {
            t.prefill_count.assign((size_t) t.n_experts, 0);
        }
        for (int e = 0; e < t.n_experts; e++) {
            const uint32_t h = t.heat[(size_t) e];
            if (t.prefill_count[(size_t) e] == 0) {
                t.prefill_count[(size_t) e] = h > 0x7fffffffu ? 0x7fffffff : (int32_t) h;
            }
        }
        pool_rerank_from_heat_locked(t);
    }
}

// TODO #87: reserve a pinned per-table bounce entry (reusing one whose copy has drained, else growing the
// ring).  Two users: the pageable-master source in `expert_host_src`, and the arena->pool hand-over in
// `access_locked`, which stages the victim through a bounce entry because its source slot is the arena slot
// the incoming copy is about to overwrite and its destination is the pool slot that copy is reading.
static int bounce_acquire(table_t & t) {
    for (int i = 0; i < (int) t.bounce.size(); i++) {
        if (t.bounce[(size_t) i].busy) {
#if defined(GGML_USE_HIP)
            const bool ready = hipEventQuery((hipEvent_t) t.bounce[(size_t) i].ev) != hipErrorNotReady;
#else
            const bool ready = cudaEventQuery(t.bounce[(size_t) i].ev) != cudaErrorNotReady;
#endif
            (void) cudaGetLastError();
            if (!ready) {
                continue;
            }
            t.bounce[(size_t) i].busy = false;
        }
        return i;
    }
    // wip/host-pool-speed: MEASURED, and deliberately NOT capped.  The grow-on-demand ring was suspected of
    // unbounded growth (charter §4.4 / PR #267's 0010), and it does hold 2.2-3.0 GiB of pinned bounce on the
    // 122B -- but the memory is the PREFILL MISS path's (one entry per expert in flight while a whole table's
    // miss set is read), and capping it costs 8-25 % of prefill at FRAC 80 (cap 8: prefill 573-644 vs 759-763;
    // cap 4: 870-891; cap 2: 846-839, all against 918-921 at FRAC 100 where nothing binds).  The scheduler-copy
    // path never exceeds ~5 entries, so a cap there is inert.  The fix for the memory is fewer, LARGER fills
    // (coalesce a run of consecutive missing experts into one pinned read), not a smaller ring -- see
    // wip/host-pool-speed README §16.
    if (g_pinned_off) {
        return -1;   // the pinned tier is latched off (see `g_pinned_off`): do not ask the driver again
    }
    if (pinned_alloc_fault(/*grow*/ true)) {
        // Diagnostic (issue #278): the driver refuses this grow, which latches the pinned tier off for the
        // rest of the run.  The refused fill takes the ordinary no-source path; what matters downstream is
        // that an EXISTING pool entry keeps serving (see `table_pool_locked`).
        g_pinned_off       = true;
        g_pinned_alloc_err = cudaErrorMemoryAllocation;
        GGML_LOG_WARN("%s: MOE_HOST_POOL_PIN_FAULT: refusing the pinned bounce grow (%.1f MiB for layer=%d "
                      "role=%s) to emulate the platform's latched refusal (issue #278).  The pinned host tier "
                      "is now OFF for the rest of this run; %.1f MiB of the pinned tier was allocated before "
                      "this refusal.\n", __func__, (double) t.host_bytes / (1024.0 * 1024.0), t.layer,
                      t.role.c_str(), (double) g_pinned_tier_bytes / (1024.0 * 1024.0));
        return -1;
    }
    device_guard dg(t.device);
    void *      mem = nullptr;
    cudaEvent_t ev  = nullptr;
    const cudaError_t alloc_err = cudaHostAlloc(&mem, t.host_bytes,
                                                cudaHostAllocPortable | cudaHostAllocMapped);
    if (alloc_err != cudaSuccess || cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess) {
        (void) cudaGetLastError();
        if (mem != nullptr) {
            (void) cudaFreeHost(mem);
        }
        if (ev != nullptr) {
            (void) cudaEventDestroy(ev);
        }
        // ONE line, then the tier is off.  This used to be silent apart from the fill-failure line, which
        // named a synthesized error instead of the allocation that actually failed (issue #274 follow-up).
        g_pinned_off       = true;
        g_pinned_alloc_err = alloc_err;
        GGML_LOG_WARN("%s: the pinned bounce ring could not grow (%.1f MiB for layer=%d role=%s): %s.  The "
                      "pinned host tier is now OFF for the rest of this run -- every fill that needs it is "
                      "served from the model mapping/pageable path, or the op is declined (correct, slower).  "
                      "%.1f MiB of the pinned tier was allocated before this refusal.\n",
                      __func__, (double) t.host_bytes / (1024.0 * 1024.0), t.layer, t.role.c_str(),
                      cudaGetErrorString(g_pinned_alloc_err),
                      (double) g_pinned_tier_bytes / (1024.0 * 1024.0));
#if defined(GGML_USE_HIP) && defined(_WIN32)
        GGML_LOG_WARN("%s: on Windows the pinned ceiling is the adapter's shared-memory (aperture) budget, "
                      "counted once per VISIBLE GPU; it is far below MemAvailable and it does NOT scale with "
                      "RAM (measured 9.3-9.4 GiB of shared usage on both a 47 GiB and a 64 GiB box, issue "
                      "#274), and a refused pinned allocation can latch a process-wide HIP error, which is "
                      "why the tier is not retried.  The exit census reports the adapter's aperture.",
                      __func__);
#endif
        return -1;
    }
    bounce_t b;
    b.host = (char *) mem;
    b.ev   = ev;
    b.busy = false;
    b.seq  = ++g_bounce_seq;
    t.bounce.push_back(b);
    g_bounce_bytes += (int64_t) t.host_bytes;
    g_pinned_tier_bytes += (int64_t) t.host_bytes;
    return (int) t.bounce.size() - 1;
}

// `protect`/`n_protect`: experts that belong to the token currently being staged and must NOT be chosen as
// an eviction victim.  Without this, an expert admitted earlier in the same token can be evicted by a later
// one (its freshly-seeded count is the smallest), which is what made the hook's `all` flag flip from token
// to token - and that variability is what broke CUDA graph replay.
// wip/host-expert-master-drop (TODO #55): resolve a PINNED, device-readable source for ONE whole host
// expert of `t`, so that no device copy and no kernel ever reads the pageable model mapping.
//
// * pinned master (the `pinned` default): the master itself, an H2D source.
// * pool-backed master (`--host-experts pool`): a pinned pool slot, a D2D source (the slot is filled from
//   the GGUF / page cache on a miss).
// * neither available (the pool is off, full, or the split geometry cannot be expressed in one slot): a
//   pinned per-table BOUNCE copy of the expert.  A pageable
//   `cudaMemcpyAsync` is NOT a safe fallback here -- ROCm stages a pageable upload through a shared
//   internal bounce, so back-to-back pageable copies can tear (measured: nondeterministic output from a
//   pageable master while every copy path verified byte-exact).  The delivery snapshots pageable uploads
//   for exactly this reason elsewhere (`set_tensor_2d_async`, the async-input ring).
//
// A source is "acquired" here and "released" by `expert_src_done` once the copy has been issued, which
// records the drain marker: a pool slot records its per-(slot,device) event (so a later hand-over to an
// arena victim, or a retire, waits for every in-flight read) and a bounce entry stays busy until its event
// signals, at which point it may be reused.
struct expert_src_t {
    const char *   base      = nullptr;   // start of the WHOLE host expert; add `t.src_off` at the copy
    cudaMemcpyKind kind      = cudaMemcpyHostToDevice;
    int            pool_slot = -1;
    bool           pool_hit  = false;     // the slot was already resident (not newly filled from disk)
    int            bounce    = -1;
};

// Diagnostic (issue #274): emulate "a pageable master has NO resolvable source for this expert", i.e. the
// bounce ring cannot be allocated.  The point of the arm is the SLOT state, not the fill: a failed fill must
// roll the admission back and decline the op (correct, slow) instead of leaving a resident slot full of stale
// bytes (fast, wrong).  Only the pool/bounce path is faulted; a pinned master is untouched.
//
//   =1        fault EVERY source, the issue's original shape (r219's arm).
//   =N (N>1)  start at the N-th call PROCESS-WIDE.  The landing point then depends on the box's arena/pool
//             sizing (free VRAM and MemAvailable move the fill count), so a run using it is not
//             reproducible across box states -- useful to reproduce a report, not to gate a text hash.
//   =late     fault from THIS TABLE's SECOND call on.  Every table gets exactly one successful source, then
//             loses it, which is the "fills that had already succeeded, then a failing call" shape
//             @briansp2020 reported -- deterministically, per table, on any box.
//   =0/unset  off.
static bool pool_fill_fault(table_t & t) {
    static const char * mode = getenv("MOE_HOST_POOL_FILL_FAULT");
    static const bool   late = mode != nullptr && strcmp(mode, "late") == 0;
    static const long long start = (mode == nullptr || late) ? 0LL : atoll(mode);
    static long long seen = 0;
    if (late) {
        return ++t.fill_src_calls >= 2;
    }
    return start > 0 && ++seen >= start;
}

// issue #274 for the DEMOTE (`gate-host-pool.sh`): make every demote copy fail, so the gate can prove that a
// failed demote DECLINES -- the victim is dropped exactly as before the exchange existed, no slot is published,
// and the admitted expert's own copy is unaffected.  A demote that "failed" but still published a victim would
// serve stale bytes from that slot forever, which is the same class of bug as #274's fill that was recorded as
// resident anyway.  Diagnostic only: nothing here changes a default, and `MOE_HOST_POOL_SINK_FAULT` is not part
// of the shipped environment surface.
static bool pool_sink_fault() {
    static const bool on = getenv("MOE_HOST_POOL_SINK_FAULT") != nullptr;
    return on;
}

// TODO #87: `allow_empty_fill` lets the demand path take an EMPTY pool slot for a miss (and then keep it --
// the pool is the other tier, not a bounce buffer).  It never evicts: a full pool misses through the bounce.
// Caller holds `g_mutex`.
static bool expert_host_src(table_t & t, int32_t expert, host_pool_t * pool, bool allow_empty_fill,
                            expert_src_t * out, const char ** why = nullptr, bool prefill = false) {
    *out = expert_src_t{};
    if (why != nullptr) {
        *why = nullptr;
    }
    if (t.host == nullptr || t.host_bytes == 0 || expert < 0 || expert >= t.n_experts) {
        if (why != nullptr) {
            *why = "the table has no usable host master";
        }
        return false;
    }
    if (!table_master_pageable(t)) {
        out->base = (const char *) t.host + (size_t) expert * t.host_bytes;
        out->kind = cudaMemcpyHostToDevice;
        return true;
    }
    if (pool_fill_fault(t)) {
        if (why != nullptr) {
            *why = "the diagnostic MOE_HOST_POOL_FILL_FAULT makes every pageable source unavailable";
        }
        return false;
    }
    if (pool != nullptr && t.src_off + t.expert_bytes <= t.host_bytes) {
        // wip/host-pool-speed: the copy KIND out of the pool.  Default keeps the historical D2D from the
        // mapped UVA alias (`base_dev`), which is required when the copying device is not the one that
        // allocated the pool (the portable+mapped alias is the only cross-device source that does not fall
        // back to a pageable staging memmove).  On a single-GPU arm the same pinned memory is a valid H2D
        // source, and `cudaMemcpyHostToDevice` is the DMA path the pinned master takes -- so this switch
        // measures how much of pool mode's decode deficit is the D2D-from-alias copy itself.
        static const bool pool_h2d = getenv("MOE_HOST_POOL_H2D") != nullptr;
        int    s = -1;
        char * p = pool_peek_locked(*pool, expert, &s);
        if (p != nullptr && s >= 0) {
            out->base      = pool_h2d ? (const char *) pool->base + (size_t) s * pool->host_bytes
                                      : (const char *) pool->base_dev + (size_t) s * pool->host_bytes;
            out->kind      = pool_h2d ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;
            out->pool_slot = s;
            out->pool_hit  = true;
            if (g_pinned_off) {
                g_pool_after_latch++;   // a pool built before the latch still serves (issue #278)
            }
            return true;
        }
        if (allow_empty_fill) {
            p = pool_fill_empty_locked(*pool, expert, &s, prefill);
            if (p != nullptr && s >= 0) {
                out->base      = pool_h2d ? (const char *) pool->base + (size_t) s * pool->host_bytes
                                          : (const char *) pool->base_dev + (size_t) s * pool->host_bytes;
                out->kind      = pool_h2d ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;
                out->pool_slot = s;
                out->pool_hit  = false;
                if (g_pinned_off) {
                    g_pool_after_latch++;   // as above: an EMPTY slot is memory the run already owns
                }
                return true;
            }
        }
        pool->misses++;
    }
    // Bounce: reuse an entry whose copy has drained, else grow the ring.  Copying the page-cache-backed
    // mapping into pinned memory is a plain host memcpy, so the transfer below is a pinned H2D.
    //
    // issue #274: the DISK TIER is an optimisation, never a requirement.  When it cannot serve (no file I/O on
    // this platform, the source was never registered, the file cannot be opened, a short read) the expert is
    // copied out of the model mapping instead -- the same bytes, only with the page cache populated.  A hard
    // failure here is a WRONG ANSWER, not a slow one: the demand path cannot serve the expert at all, and it
    // used to record the slot as resident anyway (`access_locked`).
    const int idx = bounce_acquire(t);
    if (idx < 0) {
        static bool warned_ring = false;
        if (!warned_ring) {
            warned_ring = true;
            GGML_LOG_WARN("%s: no pinned bounce entry for layer=%d role=%s (cudaHostAlloc failed); the host "
                          "pool and the bounce ring are both unavailable for this table\n", __func__, t.layer,
                          t.role.c_str());
        }
        if (why != nullptr) {
            *why = g_pinned_off ? (g_pinned_off_why != nullptr
                                       ? g_pinned_off_why
                                       : "the pinned host tier is off (the first cudaHostAlloc was refused; see the WARN above)")
                                : "the pinned bounce ring could not allocate an entry (cudaHostAlloc failed)";
        }
        return false;
    }
    char * buf = t.bounce[(size_t) idx].host;
    bool   filled = false;
    if (g_pool_dio && pool_file_io_available() && t.src_known && !t.src_path.empty()) {
        // DISK TIER (`MOE_HOST_POOL_DIO=1`): read the expert with O_DIRECT so the page cache is not
        // populated -- the whole point of the mode, and it must hold for STAGING too (a whole-table prefill
        // staging copy would otherwise fault the entire expert set into the page cache).
        device_guard dg(t.device);
        if (t.src_fd < 0) {
            t.src_fd = pool_open_file(t.src_path.c_str(), g_pool_dio);
            if (t.src_fd >= 0) {
                pool_fd_align(t.src_fd, &t.src_align);
            }
        }
        if (t.src_fd >= 0) {
            const size_t off = t.src_offs + (size_t) expert * t.host_bytes;
            const int64_t tr0 = ggml_time_us();
            filled = g_pool_dio ? pool_dio_read(t.src_fd, buf, t.host_bytes, off, t.src_align)
                                : (pool_pread(t.src_fd, buf, t.host_bytes, off) == (pool_ssize_t) t.host_bytes);
            if (filled) {
                g_bounce_read_calls++;
                g_bounce_read_us    += ggml_time_us() - tr0;
                g_bounce_read_bytes += (int64_t) t.host_bytes;
            }
        }
    }
    if (!filled && t.host != nullptr) {
        // the mapped master: a plain memcpy out of the page cache
        static bool warned_map = false;
        if (!warned_map && g_pool_dio) {
            warned_map = true;
            GGML_LOG_WARN("%s: the host pool's disk tier cannot serve a fill (layer=%d role=%s); staging "
                          "expert bytes through the model mapping instead.  The bytes are identical -- only "
                          "the page cache is populated.\n", __func__, t.layer, t.role.c_str());
        }
        memcpy(buf, (const char *) t.host + (size_t) expert * t.host_bytes, t.host_bytes);
        filled = true;
    }
    if (!filled) {
        return false;
    }
    out->base    = buf;
    out->kind    = cudaMemcpyHostToDevice;
    out->bounce  = idx;
    return true;
}

// Record the drain marker for a source acquired above, AFTER its copy has been issued.  Caller holds
// `g_mutex`.
// TODO #87: `retire` is only for a source whose copy is now DUPLICATED in VRAM (the bulk sizing/seeding
// fills, which copy a whole table in): the pool slot is freed once its in-flight reads drain, so the pool
// stays the arena's complement.  The demand path passes false -- there the slot is the other cache tier and
// is handed to the arena victim instead (see the exchange in `access_locked`).
static void expert_src_done(table_t & t, host_pool_t * pool, const expert_src_t & src, void * stream,
                            bool retire) {
    const cudaStream_t st = (cudaStream_t) stream;
    const int dev = t.device >= 0 ? t.device : 0;
    if (src.pool_slot >= 0 && pool != nullptr) {
        // Mark when this slot's copy to THIS device has drained, so a later hand-over to an arena victim
        // can safely overwrite it (per-device: a process-wide slot may serve several devices).
        cudaEvent_t ev = pool_slot_event(*pool, src.pool_slot, dev);
        if (ev != nullptr) {
            (void) cudaEventRecord(ev, st);
        }
        if (retire && src.pool_slot < (int) pool->slot_state.size()) {
            pool->slot_state[(size_t) src.pool_slot] = 2;
        }
    } else if (src.bounce >= 0 && src.bounce < (int) t.bounce.size()) {
        (void) cudaEventRecord(t.bounce[(size_t) src.bounce].ev, st);
        t.bounce[(size_t) src.bounce].busy = true;
    }
}

// wip/fix-gemma-4 (TODO #77/C): copy ONE expert's device slot from a pre-acquired host source.  One place
// for the three geometries a table can have, so the eager fill, the identity fill and any future fill cannot
// disagree: a FUSED table (multi-slice, concatenated in order), a strided axis-0 slice (2-D copy) and a
// contiguous slice (1-D copy).
// wip/host-expert-master-drop (TODO #55): the bytes come from `expert_host_src` (pinned master / pinned
// pool slot / pinned bounce), never from the pageable mapping.
static cudaError_t copy_slot_from_src(const table_t & t, void * dst, const expert_src_t & src,
                                      cudaStream_t stream) {
    if (t.n_slices > 0) {
        size_t dst_o = 0;
        for (int k = 0; k < t.n_slices; k++) {
            const cudaError_t err = cudaMemcpyAsync((char *) dst + dst_o, src.base + t.slice[k].src_off,
                                                    t.slice[k].len, src.kind, stream);
            if (err != cudaSuccess) {
                return err;
            }
            dst_o += t.slice[k].len;
        }
        return cudaSuccess;
    }
    const char * s = src.base + t.src_off;
    if (t.split_axis == 0 && t.host_pitch > 0) {
        const int64_t rows = (int64_t) (t.host_bytes / t.host_pitch);
        const size_t  row  = rows > 0 ? (t.expert_bytes / (size_t) rows) : 0;
        if (row == 0) {
            return cudaErrorInvalidDeviceFunction;
        }
        return cudaMemcpy2DAsync(dst, row, s, t.host_pitch, row, (size_t) rows, src.kind, stream);
    }
    return cudaMemcpyAsync(dst, s, t.expert_bytes, src.kind, stream);
}

// The SIZING-time bulk fill (identity table, prefill/heat seed): acquire, copy the whole table in, and
// retire the pool copy -- a seeded expert is wholly resident, so its pool slot is dead weight and is freed
// for a later `pool_fill_empty_locked` (the one-shot seeds are the only eviction this helper may cause).
//
// issue #274 follow-up: "no host source could be resolved" is a HOST-side answer, not a CUDA error, but this
// helper's contract returns `cudaError_t`.  The sentinel is never handed to `cudaGetErrorString` (every caller
// only tests it) and it deliberately is not a plausible device error: it used to be
// `cudaErrorInvalidDeviceFunction`, which printed `invalid device function` and sent the issue-#274 reporter
// looking at the device instead of at the refused `cudaHostAlloc` behind it.
static const cudaError_t k_no_host_source = cudaErrorMemoryAllocation;
static cudaError_t fill_slot_from_host(table_t & t, void * dst, size_t expert, cudaStream_t stream) {
    if (t.host == nullptr || t.host_bytes == 0 || t.expert_bytes == 0) {
        return k_no_host_source;
    }
    host_pool_t * pool = table_pool_locked(t);
    expert_src_t  src;
    const char *  why = nullptr;
    if (!expert_host_src(t, (int32_t) expert, pool, true, &src, &why)) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            GGML_LOG_WARN("%s: no pinned host source for layer=%d role=%s expert=%d (%s)\n", __func__, t.layer,
                          t.role.c_str(), (int) expert, why != nullptr ? why : "no source could be resolved");
        }
        return k_no_host_source;
    }
    const cudaError_t err = copy_slot_from_src(t, dst, src, stream);
    // wip/host-expert-pool-region: retire the source slot ONLY when the pool is a BOUNDED complement of the
    // arena (`slots < n_experts`), which is the case TODO #87's hand-over design was written for.  When the
    // budget covers the whole tensor the pool is a MIRROR of it, and retiring the slot here UN-LOADS an expert
    // that is already resident: the first arena eviction of that expert then has to re-read it from the page
    // cache through the bounce ring (a synchronous 288 KiB `memcpy` + an H2D) instead of a D2D from the pool.
    // Measured on a full-coverage run at 2 devices: 20 116 such bounce fills in the first 300 decode tokens
    // (5.5 GiB of host memcpy at ~7 GB/s = ~0.5 s), which is what made the pool's STARTING state worse than a
    // pinned master's and converged away as the arena settled.  A pinned master never has this: it holds every
    // expert for the whole run, so every fill is a direct H2D from pinned memory.
    // wip/host-expert-pool-surplus: keep the pool's copy whenever its capacity exceeds the COMPLEMENT (the
    // experts the arena does NOT hold).  The surplus is spent mirroring the arena-resident weights, so a budget
    // that is merely "big enough for the complement" already stops the un-load/re-read cycle, and a budget that
    // covers the whole tensor mirrors it completely (measured: retiring those copies cost 20 116 synchronous
    // page-cache memcpys in the first 300 decode tokens at full coverage).  A pool too small for the complement
    // retires it, so the space can hold a complement expert -- the historical bounded layout.
    const int  n_arena  = (int) t.expert_slot.size();
    // TODO #269: a pool whose capacity COVERS the tensor mirrors it, so it must never un-load an expert just
    // because the arena also holds it.  That is the same argument the surplus case below makes, taken to its
    // conclusion -- and it is also what makes the decode remap's cold map stable: a retirement is the ONLY way
    // an expert that is resident can lose its slot (the demand path never evicts a resident and a covering
    // pool's re-rank is a no-op), so with this condition a covering pool's layout is write-once after sizing.
    const bool keep_all = pool != nullptr && (pool->slots >= t.n_experts || pool->slots > t.n_experts - n_arena);
    expert_src_done(t, pool, src, stream, /*retire*/ !keep_all);
    return err;
}

moe_cache_alias access_locked(table_t & t, int32_t expert, void * stream,
                              const int32_t * protect = nullptr, int n_protect = 0,
                              bool * out_cold = nullptr) {
    // The cold read indexes the host master with the table's own host geometry (`moe_cache_get_cold`
    // returns the per-expert and per-row strides), so a Phase 3 slice is servable in place too.
    const bool cold_ok = table_cold_ok(t) || pool_cold_slot_locked(t, expert) >= 0;
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
    int     slot   = -1;
    int32_t victim = -1;   // arena resident evicted by this miss (also fetched into the pool: additive)
    if ((int) t.expert_slot.size() < t.slots) {
        for (int s = 0; s < t.slots; s++) {
            if (t.slot_expert[s] < 0) { slot = s; break; }
        }
    }
    if (slot < 0) {
        // evict the resident with the smallest decaying count; tie -> oldest use.
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
        char * dst = (char *) t.arena + (size_t) slot * t.expert_bytes;
        host_pool_t * pool = table_pool_locked(t);

        // TODO #87 -- the pool is a LAYOUT, not a churning cache.  Its content is decided in bulk by
        // `pool_rerank_from_heat_locked` (the arena's complement, ranked by the persisted heat prior), and
        // the demand path only ever reads it: a resident expert is the source (one D2D copy, needed anyway),
        // a miss takes an EMPTY slot (never evicting a resident) and otherwise falls through to the pinned
        // bounce.  No copy is spent moving the arena victim into the pool: measured on the IQ4_NL field
        // config, the hand-over cost 2x decode throughput while the offline model put every hand-over
        // policy within 250 MiB of the others' L3 reads, against a 15x difference for the pool's CONTENT.
        expert_src_t hsrc;
        cudaError_t  err = cudaSuccess;
        const char * why = nullptr;
        // issue #274 follow-up (@DanoPTT): `expert_host_src` failing is a HOST-side answer (no pool slot, no
        // pinned bounce, no mapping), so it has no CUDA error to report.  The old code synthesized
        // `cudaErrorInvalidDeviceFunction` here, which printed `invalid device function` and sent the
        // reporter looking at the device -- the actual cause was a refused `cudaHostAlloc`.  Keep the two
        // cases apart and report each one's real reason.
        // The incoming expert MAY take an empty pool slot, even with the exchange on: `pool_free_slot_locked`
        // already refuses to hand out one that the reserve is holding for the demote, which is what bounds the
        // competition.  Refusing empty fills outright was wrong and measurably harmful -- it turned the pool
        // into a demote-only cache that never read from disk, so at -lme 12 the hit rate collapsed to 4.4 %
        // and every miss took the synchronous bounce ring instead (the fill-time instrument read zero fills).
        const bool   no_src = !expert_host_src(t, expert, pool, /*allow_empty_fill*/ true, &hsrc, &why);
        // wip/host-expert-split W2: demote the victim BEFORE the fill overwrites it.  One D2H straight into a
        // free pool slot -- no staging buffer, no second copy.  `dst` still holds the victim's bytes here and
        // the fill's source is a different slot, so the only ordering requirement is D2H-then-H2D on this
        // stream.  A decline must never disturb the fill, and each decline names its CAUSE.
        if (pool_sink_active(pool) && victim >= 0 && !no_src) {
            if (t.n_slices != 0 || pool->host_bytes != t.expert_bytes || pool->base == nullptr) {
                g_sink_decl_geom++;   // structural: this table has no single contiguous pool image
            } else {
                int s_sink = pool_free_slot_locked(*pool, 0);
                if (s_sink < 0) {
                    // The pool is SATURATED.  Declining here freezes it: with no free slot no fill can
                    // happen, so nothing retires, so every later demote declines too and the pool stays
                    // frozen on whatever the prior guessed.  Measured at -lme 12: h=0.0439 with 15 212
                    // victims thrown away while every miss took the synchronous bounce ring (the disk path
                    // that runs at ~2.4 GB/s, latency-bound at one 0.6 MiB read per fill).  So reclaim a
                    // resident instead: prefer one that is a DUPLICATE -- its expert is in the arena too, so
                    // the pool copy is worth nothing right now -- else the least recently used.  Never the
                    // slot this fill is reading (`hsrc.pool_slot`), and never the victim itself.
                    // DUPLICATES ONLY.  Reclaiming the LRU resident instead was measured and is a net LOSS
                    // at mid budgets: at -lme 24 it evicted 10 220 residents, doubled the demotion count
                    // (8 019 -> 17 367) and lost a third of the throughput (18.7 -> 12.4 t/s), because a
                    // 4 GiB arena holds too few duplicates and the pool's ONLY copy of an expert was being
                    // thrown away.  A duplicate is the one eviction that costs nothing: the expert is in
                    // VRAM right now, so that pool copy is worth nothing.
                    int32_t best = -1, best_last = 0;
                    for (int s2 = 0; s2 < pool->slots; s2++) {
                        const int32_t e2 = pool->slot_expert[(size_t) s2];
                        if (e2 < 0 || e2 == victim || s2 == hsrc.pool_slot) {
                            continue;
                        }
                        if (t.expert_slot.find(e2) == t.expert_slot.end()) {
                            continue;   // not a duplicate: this is the pool's only copy -- keep it
                        }
                        const int32_t l2 = e2 < (int32_t) pool->last.size() ? (int32_t) pool->last[(size_t) e2] : 0;
                        if (best < 0 || l2 < best_last) {
                            best = s2;
                            best_last = l2;
                        }
                    }
                    if (best >= 0) {
                        s_sink = best;
                        g_sink_evicted++;
                    } else {
                        g_sink_decl_noslot++;
                        g_sink_need_slot = true;
                    }
                }
                if (s_sink >= 0) {
                    // `MOE_HOST_POOL_SINK_FAULT` short-circuits the copy into the SAME failure branch a real
                    // copy error takes: the demote is declined, nothing is published, the op proceeds.
                    const bool copy_ok = !pool_sink_fault() &&
                        cudaMemcpyAsync(pool->base + (size_t) s_sink * pool->host_bytes, dst, t.expert_bytes,
                                        cudaMemcpyDeviceToHost, (cudaStream_t) stream) == cudaSuccess;
                    if (!copy_ok) {
                        if (!pool_sink_fault()) {
                            (void) cudaGetLastError();
                        }
                        g_sink_decl_copy++;
                        g_sink_need_slot = true;
                    } else {
                        cudaEvent_t ev = pool_slot_event(*pool, s_sink, t.device >= 0 ? t.device : 0);
                    if (ev != nullptr) {
                            (void) cudaEventRecord(ev, (cudaStream_t) stream);
                        }
                        pool_slot_assign_locked(*pool, s_sink, victim);
                        if ((int) pool->slot_demoted.size() == pool->slots) {
                            pool->slot_demoted[(size_t) s_sink] = 1;
                        }
                        g_sink_copies++;
                        g_sink_bytes += (int64_t) t.expert_bytes;
                    }
                }
            }
            g_sink_declines = g_sink_decl_noslot + g_sink_decl_geom + g_sink_decl_copy;
        }
        if (!no_src) {
            err = copy_slot_from_src(t, dst, hsrc, (cudaStream_t) stream);
        }
        if (err == cudaSuccess && !no_src) {
            // Keep a pool-resident source resident: the pool holds the arena's complement, and this expert
            // being in VRAM now does not mean the pool should forget where to find it next time.
            //
            // wip/host-expert-split W2: EXCEPT when the demote needs the slot back.  With the exchange on,
            // the pool plans K slots of headroom; every demote spends one and nothing in this path ever
            // frees one (#87's demand path deliberately never retires), so the headroom would drain to zero
            // and the exchange would stall after K misses.  So when no free slot is left, retire this
            // expert's pool copy back into the free list: it is duplicated in VRAM now, and the vacancy it
            // creates is immediately filled by a just-evicted (hot) expert rather than left empty -- which
            // is what made #87's duplicate-retirement harmful, and what makes it harmless here.
            bool retire = false;
            if (pool_sink_active(pool)) {
                // ALWAYS retire under the exchange.  Measured why: every miss needs up to TWO free slots (one
                // for the incoming fill, one for the demote) and a demand-driven retirement returns only one,
                // so the steady state was a 66 % decline rate with declined == retired to the digit (7513 vs
                // 7513).  The duplicate this gives up has no value while the expert is arena-resident, and the
                // vacancy is taken by a just-evicted expert that does (#87's objection measured retirement with
                // nothing to refill the slot).
                retire = true;
                g_sink_retire++;
                g_sink_need_slot = false;
            }
            expert_src_done(t, pool, hsrc, stream, retire);
        }
        if (err != cudaSuccess || no_src) {
            // issue #274: a fill that failed must NOT leave the expert RESIDENT.  The remap is built from
            // `t.expert_slot`, so recording a slot whose bytes were never copied does not degrade the answer,
            // it CORRUPTS it (the reporter's run produced one token and stopped: every fill had failed and
            // every slot was still marked resident).  Roll the admission back: `moe_cache_update_host`'s pass 2
            // then finds no slot for this expert, cannot resolve a cold or pool id for it either, and DECLINES
            // the op, so the scheduler serves it from the host master (slow, correct).
            t.expert_slot.erase(expert);
            if (slot >= 0 && slot < t.slots && t.slot_expert[slot] == expert) {
                t.slot_expert[slot] = -1;
            }
            t.slot_dirty = true;
            t.fill_failed = true;
            t.count[expert] = 0;   // never used: a later admission must not inherit this attempt's count
            t.fills--;
            g_acc_fills--;
            g_fill_bytes -= (int64_t) t.expert_bytes;
            g_fill_fail++;
            const bool first_for_table = !t.fill_fail_warned;
            t.fill_fail_warned = true;
            if (first_for_table || (g_fill_fail % 10000) == 0) {
                // Report the REAL cause: `err` is what `copy_slot_from_src` returned, and `cudaGetLastError()`
                // is empty for a host-side failure (`"no error"` is what made this look like a driver problem
                // in issue #274).  A source that could not be resolved says so in words instead of naming a
                // device error it never had.
                if (no_src) {
                    GGML_LOG_WARN("%s: fill failed table layer=%d role=%s expert=%d: no pinned host source (%s); "
                                  "the slot is NOT resident (%lld failed fills so far)\n", __func__, t.layer,
                                  t.role.c_str(), expert, why != nullptr ? why : "no source could be resolved",
                                  (long long) g_fill_fail);
                } else {
                    GGML_LOG_WARN("%s: fill failed table layer=%d role=%s expert=%d: %s; the slot is NOT resident "
                                  "(%lld failed fills so far)\n", __func__, t.layer, t.role.c_str(), expert,
                                  cudaGetErrorString(err), (long long) g_fill_fail);
                }
            }
            if (cold_ok && out_cold != nullptr) {
                *out_cold = true;
                g_acc_colds++;
                g_cold_bytes += (int64_t) t.expert_bytes;
            }
            return { nullptr, false };
        }
        // NOTE: `expert_src_done` is called ONCE, above, inside the `err == cudaSuccess && !no_src` guard.
        // It used to be repeated here, outside the failure guard -- a redundant drain-event record while the
        // failure block always returns, but a trap: any future fallback that continued past that block would
        // publish this slot as drained although no copy was ever issued.
    }
    return { (char *) t.arena + (size_t) slot * t.expert_bytes, true };
}



// wip/moe-expert-cache prefill seed (MOE_EXPERT_CACHE_PREFILL_SEED=1): one-shot bulk admission of the
// experts the prefill routing touched most, applied at arena sizing (after the arena/policy state exist).
// wip/first-prefill-latency: the arena is sized against the full slab region, but a wide prefill reclaims
// the band `[narrow_floor, narrow_floor + ring + wide)`.  A table whose arena starts inside that band is
// the coldest (heat-ordered last) and must NOT be pre-seeded -- the seed would be thrown away on the very
// next prefill.  Leave it empty; it fills lazily during decode.  Returns true when seeding is allowed.
static bool heat_seed_above_floor(const table_t & t) {
    if (t.device < 0 || t.arena == nullptr) {
        return true;
    }
    const size_t floor = ggml_cuda_slab_seed_floor(t.device);
    if (floor == 0) {
        return true;   // layout not established (non-slab path): seed as before
    }
    const char * base = (const char *) ggml_cuda_slab_work_base(t.device);
    if (base == nullptr) {
        return true;
    }
    return (size_t) ((const char *) t.arena - base) >= floor;
}

// `t.prefill_count` was accumulated in `moe_cache_update_host`'s prefill branch; here it is ranked
// (count desc, expert id asc), the top `min(slots, cap)` are placed in slots 0.. and copied from the
// pinned host master - the same bytes a decode fill would fetch, just front-loaded.  The host mirrors are
// populated so the eager remap picks them up.  No-op for an identity table (already wholly resident)
// or one with no prefill tally.
void apply_prefill_seed_locked(table_t & t) {
    const bool from_tally = g_prefill_seed;
    if (!g_prefill_seed || t.identity || t.slots <= 0 || t.host == nullptr ||
        !t.expert_slot.empty() || !heat_seed_above_floor(t)) {
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
    int seeded = 0;
    bool ok = true;
    for (int i = 0; i < k; i++) {
        const int32_t e  = rank[(size_t) i];
        void *        dst = (char *) t.arena + (size_t) i * t.expert_bytes;
        // wip/host-expert-master-drop (TODO #55): one geometry-aware source (pinned master / pinned pool /
        // pageable fallback), so a seeded table is byte-identical to a lazily filled one.
        const cudaError_t err = fill_slot_from_host(t, dst, (size_t) e, (cudaStream_t) 0);
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
        GGML_LOG_INFO("%s: layer=%d role=%s prefill-%s %d/%d slots (%d experts; %lld prefill tokens)\n",
                      __func__, t.layer, t.role.c_str(), from_tally ? "seeded" : "loaded",
                      seeded, t.slots, t.n_experts, (long long) t.prefill_tokens);
    }
}

// wip/moe-heatmap-persistence: seed the arena at sizing from the PERSISTED heat map (loaded at model
// load), so a cold start begins with the previous run's hot set.  Same provisional bulk admission as
// the prefill seed, but the ranking is the cross-run prior and it is not gated on
// MOE_EXPERT_CACHE_PREFILL_SEED.  Call it BEFORE `apply_prefill_seed_locked` so the persisted prior
// wins the (empty-arena) race for slots 0...
void apply_heat_seed_locked(table_t & t) {
    if (!g_heat_on || t.identity || t.slots <= 0 || t.host == nullptr || !t.expert_slot.empty() ||
        !heat_seed_above_floor(t)) {
        return;
    }
    if ((int) t.heat.size() != t.n_experts || t.n_experts <= 0) {
        return;
    }
    // TODO #88: rank by the DECODE band (with the combined vector as the fallback), because the arena serves
    // the decode/verify band.  `heat`'s own total is what decides whether there is a prior at all.
    const std::vector<uint32_t> & rank_by = heat_rank_vec(t);
    uint64_t total = 0;
    for (uint32_t v : t.heat) {
        total += v;
    }
    if (total == 0) {
        return;
    }
    std::vector<int32_t> rank((size_t) t.n_experts);
    for (int e = 0; e < t.n_experts; e++) {
        rank[(size_t) e] = e;
    }
    std::sort(rank.begin(), rank.end(), [&](int32_t a, int32_t b) {
        if (rank_by[(size_t) a] != rank_by[(size_t) b]) {
            return rank_by[(size_t) a] > rank_by[(size_t) b];
        }
        return a < b;
    });
    int k = t.n_experts;
    if (g_prefill_seed_n > 0 && g_prefill_seed_n < k) {
        k = g_prefill_seed_n;
    }
    if (k > t.slots) {
        k = t.slots;
    }
    int  seeded = 0;
    bool ok = true;
    for (int i = 0; i < k; i++) {
        const int32_t e   = rank[(size_t) i];
        void *        dst = (char *) t.arena + (size_t) i * t.expert_bytes;
        // wip/host-expert-master-drop (TODO #55): the same geometry-aware source the eager fill uses.
        const cudaError_t err = fill_slot_from_host(t, dst, (size_t) e, (cudaStream_t) 0);
        if (err != cudaSuccess) {
            (void) cudaGetLastError();
            ok = false;
            break;
        }
        t.slot_expert[i] = e;
        t.expert_slot[e] = i;
        t.slot_prov[i]   = 1;
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
        t.fills     += seeded;
        t.slot_dirty = true;
        GGML_LOG_INFO("%s: layer=%d role=%s heat-seeded %d/%d slots from the persisted map (%d experts, "
                      "ranked by the %s band)\n", __func__, t.layer, t.role.c_str(), seeded, t.slots,
                      t.n_experts, &rank_by == &t.heat_dec ? "DECODE" : "combined");
    }
}

// ---------------------------------------------------------------------------------------------
// wip/moe-heatmap-persistence: the on-disk heat map (sidecar next to the model, geometry-keyed).
// Fixed-size records; a corrupt/truncated/mismatched file is ignored (a cold start), never fatal.
// ---------------------------------------------------------------------------------------------
#if !defined(_WIN32)
struct mht1_hdr_t {
    char     magic[4];
    uint32_t version;
    uint32_t n_tables;
    uint32_t reserved0;
    uint64_t model_bytes;
    uint32_t n_tensors;
    uint32_t reserved1;
};
struct mht1_rec_t {
    int32_t  layer;
    char     role[64];
    uint32_t n_experts;
    uint64_t host_bytes;
};
#endif

// The part of a table's `heat` NOT yet folded into `g_heat_loaded`.  `heat_merged` starts as the loaded
// prior (empty for a cold table), so the delta is exactly this session's routing since the last merge.
// Clamped: a vector re-assigned behind our back (a geometry change, a device-heat fold) must not
// underflow into a huge add.
static uint32_t heat_delta(const table_t & t, size_t e) {
    const uint32_t s = t.heat[e];
    const uint32_t m = e < t.heat_merged.size() ? t.heat_merged[e] : 0u;
    return s > m ? (uint32_t) (s - m) : 0u;
}

// TODO #88: the same delta for the decode-band counter.
static uint32_t heat_dec_delta(const table_t & t, size_t e) {
    if (e >= t.heat_dec.size()) {
        return 0u;
    }
    const uint32_t s = t.heat_dec[e];
    const uint32_t m = e < t.heat_dec_merged.size() ? t.heat_dec_merged[e] : 0u;
    return s > m ? (uint32_t) (s - m) : 0u;
}

// TODO #88: which counter a HEAT-DRIVEN ARENA DECISION ranks by.  The arena serves the decode/verify band
// (`moe_cache_update_host` returns early for a prefill ubatch, and the whole-table staging bypasses the
// arena), so the decode band is what its residents are worth.  `heat` remains the combined prior for the
// host pool's layout (`pool_rank_experts_locked`), which stages whole tables *for a prefill*: there the
// prefill mass is the right signal, not noise.  A map with no decode mass (a v1 sidecar, a run that never
// decoded, or `MOE_EXPERT_HEATMAP=0` in between) falls back to the combined vector, i.e. to the pre-#88
// behaviour.
static const std::vector<uint32_t> & heat_rank_vec(const table_t & t) {
    if ((int) t.heat_dec.size() == t.n_experts && t.n_experts > 0) {
        for (uint32_t v : t.heat_dec) {
            if (v != 0) {
                return t.heat_dec;
            }
        }
    }
    return t.heat;
}

// Fold every table's NEW heat (host) into `g_heat_loaded` so a save writes the merged prior.  The
// DELTA, not the running total: `heat` already contains the prior it was seeded from, so adding it
// whole would double the prior at every flush (see `heat_merged`).
// Caller holds `g_mutex`.
static void heat_merge_into_loaded_locked() {
    for (table_t & t : g_tables) {
        if ((int) t.heat.size() != t.n_experts || t.n_experts <= 0) {
            continue;
        }
        const auto key = std::make_tuple(t.layer, t.role, t.n_experts, t.host_bytes);
        heat_prior_t & prior = g_heat_loaded[key];
        if (prior.all.size() != (size_t) t.n_experts) {
            prior.all.assign((size_t) t.n_experts, 0);
        }
        if ((int) t.heat_dec.size() != t.n_experts && t.n_experts > 0) {
            t.heat_dec.assign((size_t) t.n_experts, 0);
        }
        if (prior.dec.size() != (size_t) t.n_experts) {
            prior.dec.assign((size_t) t.n_experts, 0);
        }
        for (int e = 0; e < t.n_experts; e++) {
            const uint64_t sa = (uint64_t) prior.all[(size_t) e] + (uint64_t) heat_delta(t, (size_t) e);
            prior.all[(size_t) e] = sa > 0xffffffffull ? 0xffffffffu : (uint32_t) sa;
            const uint64_t sd = (uint64_t) prior.dec[(size_t) e] + (uint64_t) heat_dec_delta(t, (size_t) e);
            prior.dec[(size_t) e] = sd > 0xffffffffull ? 0xffffffffu : (uint32_t) sd;
        }
        t.heat_merged     = t.heat;
        t.heat_dec_merged = t.heat_dec;
    }
}

// Atomic staged write: `<path>.tmp.<pid>` -> fsync -> rename over `<path>`.  Fail-soft; after
// `g_heat_max_fails` consecutive failures persistence is off for the session.  Caller holds `g_mutex`.
static void heat_map_save_locked() {
#if !defined(_WIN32)
    if (!g_heat_on || g_heat_dead || !g_heat_path_set || g_heat_path.empty()) {
        return;
    }
    // A frozen heat map is loaded and used to seed, but never overwritten.  The file is re-checked here
    // too, so making it read-only while a server is running freezes it at the next flush boundary.
    if (g_heat_frozen || !g_heat_write_ok ||
        (access(g_heat_path.c_str(), F_OK) == 0 && access(g_heat_path.c_str(), W_OK) != 0)) {
        g_heat_frozen = true;
        static bool warned = false;
        if (!warned) {
            warned = true;
            GGML_LOG_INFO("%s: heat map %s is read-only; seeding from it, not updating it\n",
                          __func__, g_heat_path.c_str());
        }
        return;
    }
    heat_merge_into_loaded_locked();
    std::vector<uint8_t> buf;
    mht1_hdr_t hdr{};
    // TODO #88: format v2 carries BOTH bands per table (`all` then `dec`).  A v1 reader ignores a v2 file
    // (it checks the version), which costs that build its prior but never corrupts anything; a v2 reader
    // accepts v1 and uses `all` for both, i.e. exactly the pre-r115 behaviour.
    memcpy(hdr.magic, "MHT2", 4);
    hdr.version   = 2;
    hdr.n_tables  = 0;
    for (const auto & kv : g_heat_loaded) {
        if (kv.second.all.size() == (size_t) std::get<2>(kv.first)) {
            hdr.n_tables++;
        }
    }
    buf.insert(buf.end(), (const uint8_t *) &hdr, (const uint8_t *) &hdr + sizeof(hdr));
    for (const auto & kv : g_heat_loaded) {
        const int         layer    = std::get<0>(kv.first);
        const std::string role     = std::get<1>(kv.first);
        const int         n_exp    = std::get<2>(kv.first);
        const uint64_t    host_b   = (uint64_t) std::get<3>(kv.first);
        if (kv.second.all.size() != (size_t) n_exp) {
            continue;
        }
        mht1_rec_t rec{};
        rec.layer      = layer;
        strncpy(rec.role, role.c_str(), sizeof(rec.role) - 1);
        rec.n_experts  = (uint32_t) n_exp;
        rec.host_bytes = host_b;
        buf.insert(buf.end(), (const uint8_t *) &rec, (const uint8_t *) &rec + sizeof(rec));
        buf.insert(buf.end(), (const uint8_t *) kv.second.all.data(),
                   (const uint8_t *) kv.second.all.data() + (size_t) n_exp * sizeof(uint32_t));
        if (kv.second.dec.size() == (size_t) n_exp) {
            buf.insert(buf.end(), (const uint8_t *) kv.second.dec.data(),
                       (const uint8_t *) kv.second.dec.data() + (size_t) n_exp * sizeof(uint32_t));
        } else {
            buf.insert(buf.end(), (size_t) n_exp * sizeof(uint32_t), 0);
        }
    }
    const std::string tmp = g_heat_path + ".tmp." + std::to_string((long) getpid());
    bool ok = true;
    int  fd = open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        ok = false;
    } else {
        size_t off = 0;
        while (off < buf.size()) {
            const ssize_t w = write(fd, buf.data() + off, buf.size() - off);
            if (w <= 0) { ok = false; break; }
            off += (size_t) w;
        }
        if (ok) { ok = fsync(fd) == 0; }
        close(fd);
    }
    if (ok) {
        ok = rename(tmp.c_str(), g_heat_path.c_str()) == 0;
        if (!ok) { (void) unlink(tmp.c_str()); }
    }
    if (ok) {
        g_heat_fails = 0;
        g_heat_last_save = std::chrono::steady_clock::now();
        GGML_LOG_INFO("%s: saved %u heat entries to %s\n", __func__, hdr.n_tables, g_heat_path.c_str());
    } else {
        g_heat_fails++;
        GGML_LOG_WARN("%s: could not write the heat map to %s (failure %d/%d)\n",
                      __func__, g_heat_path.c_str(), g_heat_fails, g_heat_max_fails);
        if (g_heat_fails >= g_heat_max_fails) {
            g_heat_dead = true;
            GGML_LOG_WARN("%s: heat-map persistence disabled for this session after %d failures\n",
                          __func__, g_heat_fails);
        }
    }
#endif
}

static void heat_map_load_locked(const std::string & path) {
#if !defined(_WIN32)
    FILE * f = fopen(path.c_str(), "rb");
    if (f == nullptr) {
        return;   // cold start
    }
    mht1_hdr_t hdr{};
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) {
        fclose(f);
        return;
    }
    // TODO #88: v1 carries one counter per expert (both bands mixed); v2 carries `all` then `dec`.  A v1
    // map is READ and used for both, which is the pre-r115 behaviour -- never an error, never a fallback to
    // "no prior".
    const bool v2 = memcmp(hdr.magic, "MHT2", 4) == 0 && hdr.version == 2;
    const bool v1 = memcmp(hdr.magic, "MHT1", 4) == 0 && hdr.version == 1;
    if (!v1 && !v2) {
        fclose(f);
        GGML_LOG_INFO("%s: ignoring an unreadable/stale heat map at %s\n", __func__, path.c_str());
        return;
    }
    size_t n = 0;
    for (uint32_t i = 0; i < hdr.n_tables; i++) {
        mht1_rec_t rec{};
        if (fread(&rec, sizeof(rec), 1, f) != 1) { break; }
        if (rec.n_experts == 0 || rec.n_experts > (1u << 22)) { break; }
        heat_prior_t h;
        h.all.assign((size_t) rec.n_experts, 0);
        if (fread(h.all.data(), sizeof(uint32_t), rec.n_experts, f) != rec.n_experts) { break; }
        if (v2) {
            h.dec.assign((size_t) rec.n_experts, 0);
            if (fread(h.dec.data(), sizeof(uint32_t), rec.n_experts, f) != rec.n_experts) { break; }
        } else {
            h.dec = h.all;
        }
        const std::string role(rec.role, strnlen(rec.role, sizeof(rec.role)));
        g_heat_loaded[std::make_tuple((int) rec.layer, role, (int) rec.n_experts, (size_t) rec.host_bytes)] =
            std::move(h);
        n++;
    }
    fclose(f);
    // Reject a degenerate prior.  A map that cannot rank is worse than no map: the seed's ties break by
    // expert id, so every table would pre-fill experts 0..slots-1 and the layer placement would fall back
    // to layer-id order - while the log still says "heat-seeded".  A saturated entry (0xffffffff) cannot
    // come from real routing (it needs 4.29e9 hits on one expert), and an all-tied map ranks nothing.
    // Seen in the wild: a doubling bug in this save path left 97-99% of entries saturated in the sidecar
    // of any long server session (2026-10-17, fixed above).  Policy: start neutral either way; re-record
    // when the file is writable, leave it alone when it is read-only.
    {
        size_t total = 0, sat = 0, nonzero = 0;
        uint32_t lo = 0xffffffffu, hi = 0;
        for (const auto & kv : g_heat_loaded) {
            for (uint32_t v : kv.second.all) {
                total++;
                if (v == 0xffffffffu) { sat++; }
                if (v > 0) {
                    nonzero++;
                    lo = v < lo ? v : lo;
                    hi = v > hi ? v : hi;
                }
            }
        }
        if (total > 0 && (sat > 0 || (nonzero > 0 && lo == hi))) {
            const bool frozen = g_heat_frozen || !g_heat_write_ok;
            GGML_LOG_WARN("%s: ignoring the heat map at %s: it is degenerate (%zu of %zu entries "
                          "saturated, %zu nonzero).  A prior that cannot rank would seed by expert id.  %s\n",
                          __func__, path.c_str(), sat, total, nonzero,
                          frozen ? "The file is read-only, so it is left as-is and this run starts neutral."
                                 : "This run starts neutral and re-records the map.");
            g_heat_loaded.clear();
            n = 0;
        }
    }
    GGML_LOG_INFO("%s: loaded %zu heat entries from %s\n", __func__, n, path.c_str());
#else
    GGML_UNUSED(path);
#endif
}

// V2.0 block 09: the shard-independent sidecar stem.  A sharded GGUF is handed to the loader as
// whichever shard owns the first host-expert tensor, so keying the sidecar on the raw path wrote one
// map per shard and which one existed depended on the load order (the observed mess: 00001/00003/
// 00004/00008 in one directory).  Normalise to the model base name: strip a trailing ".gguf", then the
// llama.cpp split marker "-NNNNN-of-NNNNN" (see `llama_split_path`, `src/llama.cpp`), so every shard of
// one model maps to exactly one name.  The ".moe-heatmap" extension is added by the caller.
static std::string heat_map_base_name(const std::string & name) {
    std::string s = name;
    static const char * const ext = ".gguf";
    const size_t ext_len = strlen(ext);
    if (s.size() > ext_len && s.compare(s.size() - ext_len, ext_len, ext) == 0) {
        s.resize(s.size() - ext_len);
    }
    // ``-NNNNN-of-NNNNN`` (exactly the `llama_split_path` format) at the very end.
    const size_t digits = 5;
    if (s.size() > 1 + digits + 4 + digits) {
        const size_t pos = s.size() - (1 + digits + 4 + digits);
        bool ok = s[pos] == '-';
        for (size_t i = 0; ok && i < digits; i++) {
            ok = s[pos + 1 + i] >= '0' && s[pos + 1 + i] <= '9';
        }
        ok = ok && s.compare(pos + 1 + digits, 4, "-of-") == 0;
        for (size_t i = 0; ok && i < digits; i++) {
            ok = s[pos + 1 + digits + 4 + i] >= '0' && s[pos + 1 + digits + 4 + i] <= '9';
        }
        if (ok) {
            s.resize(pos);
        }
    }
    return s;
}

// V2.0 block 09: the sidecar is one unified map per model, named by the shard-independent base.  Before
// this it was `<gguf>.moe-heatmap` -- one file per GGUF shard, and a redundant `.gguf` for a single-file
// model.  That layout is retired: it is never read or written now, so warn the user when stale ones are
// present next to the model so they can be removed (the map relearns from a cold start otherwise).
static void heat_map_warn_retired_locked(const std::string & dir, const std::string & base_name) {
#if !defined(_WIN32)
    DIR * d = opendir(dir.empty() ? "." : dir.c_str());
    if (d == nullptr) {
        return;   // no readable model directory (or no permission): nothing to report
    }
    static const char * const retired = ".moe-heatmap";
    const size_t retired_len = strlen(retired);
    int found = 0;
    while (struct dirent * e = readdir(d)) {
        const std::string name(e->d_name);
        if (name.size() <= retired_len || name.compare(name.size() - retired_len, retired_len, retired) != 0) {
            continue;   // not a heat map
        }
        const std::string bare = name.substr(0, name.size() - retired_len);   // the `<gguf>` it was keyed on
        if (bare == base_name) {
            continue;   // the current unified name (`<base>.moe-heatmap`)
        }
        if (heat_map_base_name(bare) == base_name) {
            found++;
            GGML_LOG_WARN("%s: ignoring retired heat map %s%s; heat maps are now one unified index per "
                          "model named %s%s%s (the retired `<gguf>.moe-heatmap` naming is no longer used)\n",
                          __func__, dir.c_str(), name.c_str(), dir.c_str(), base_name.c_str(), retired);
        }
    }
    closedir(d);
    if (found > 0) {
        GGML_LOG_WARN("%s: %d retired heat-map file(s) ignored; delete them to silence this warning\n",
                      __func__, found);
    }
#else
    GGML_UNUSED(dir);
    GGML_UNUSED(base_name);
#endif
}

// Resolve the sidecar path (model dir, else a writable cache dir) and load it.  Called once, from the
// loader's first host-source registration (which carries the GGUF path).  Caller holds `g_mutex`.
static void heat_map_open_locked(const char * model_path) {
    if (!g_heat_on || g_heat_path_set || model_path == nullptr || model_path[0] == '\0') {
        return;
    }
    g_heat_path_set  = true;
    g_heat_last_save = std::chrono::steady_clock::now();
    const std::string mp(model_path);
    const size_t slash = mp.find_last_of('/');
    const std::string stem = slash == std::string::npos ? mp : mp.substr(slash + 1);
    const std::string base_name = heat_map_base_name(stem);
    const std::string dir  = slash == std::string::npos ? std::string() : mp.substr(0, slash + 1);
    const std::string primary = dir + base_name + ".moe-heatmap";
    heat_map_warn_retired_locked(dir, base_name);
    const char * env_dir = getenv("MOE_EXPERT_HEATMAP_DIR");
#if !defined(_WIN32)
    // Is there a readable sidecar next to the model?  If so it is AUTHORITATIVE: a read-only one is
    // loaded and FROZEN rather than silently relocated (relocating would ignore the curated map and
    // keep learning in a cache directory, so "chmod the sidecar" and "read-only the model directory"
    // both freeze, as intended).
    const bool primary_readable = access(primary.c_str(), R_OK) == 0;
    const bool dir_writable = [&] {
        std::string probe = primary + ".probe";
        int fd = open(probe.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd >= 0) { close(fd); (void) unlink(probe.c_str()); return true; }
        return false;
    }();
    if (env_dir != nullptr && env_dir[0] != '\0') {
        g_heat_path     = std::string(env_dir) + "/" + base_name + ".moe-heatmap";
        g_heat_write_ok = access(env_dir, W_OK) == 0;
    } else if (g_heat_frozen) {
        g_heat_path     = primary;   // MOE_EXPERT_HEATMAP=load: load + seed, never write
        g_heat_write_ok = false;
    } else if (primary_readable && access(primary.c_str(), W_OK) != 0) {
        // A read-only heat map IS the freeze signal: load and seed from it, never overwrite it, and do
        // NOT fall back to a cache directory (that would ignore the curated map and keep learning).
        g_heat_path     = primary;
        g_heat_write_ok = false;
        g_heat_frozen   = true;
        GGML_LOG_INFO("%s: heat map %s is read-only; loading and seeding from it, not updating it\n",
                      __func__, g_heat_path.c_str());
    } else if (dir_writable) {
        g_heat_path     = primary;
        g_heat_write_ok = true;
    } else {
        // No sidecar and a read-only model directory (the container case): fall back to a writable
        // cache directory so the map can still be learned there.
        const char * xdg  = getenv("XDG_CACHE_HOME");
        const char * home = getenv("HOME");
        std::string base = (xdg != nullptr && xdg[0] != '\0') ? std::string(xdg)
                          : (home != nullptr && home[0] != '\0') ? std::string(home) + "/.cache"
                          : std::string("/tmp");
        base += "/llama-moe-heatmap";
        (void) mkdir(base.c_str(), 0755);
        g_heat_path     = base + "/" + base_name + ".moe-heatmap";
        g_heat_write_ok = access(base.c_str(), W_OK) == 0;
        if (!g_heat_write_ok) {
            g_heat_frozen = true;
        }
        GGML_LOG_INFO("%s: model directory is not writable and has no sidecar; heat map at %s%s\n",
                      __func__, g_heat_path.c_str(), g_heat_write_ok ? "" : " (frozen)");
    }
    if (g_heat_frozen) {
        g_heat_write_ok = false;
    }
#else
    g_heat_path = primary;
    g_heat_write_ok = false;
    GGML_UNUSED(env_dir);
#endif
    heat_map_load_locked(g_heat_path);
}

// Apply a loaded entry to a freshly registered table.  Caller holds `g_mutex`.
static void heat_apply_loaded_locked(table_t & t) {
    if (!g_heat_on || g_heat_loaded.empty() || t.n_experts <= 0) {
        return;
    }
    const auto it = g_heat_loaded.find(std::make_tuple(t.layer, t.role, t.n_experts, t.host_bytes));
    if (it == g_heat_loaded.end()) {
        return;
    }
    if ((int) it->second.all.size() != t.n_experts) {
        return;
    }
    t.heat             = it->second.all;
    t.heat_dec         = (int) it->second.dec.size() == t.n_experts ? it->second.dec
                                                                   : it->second.all;
    t.heat_merged      = t.heat;       // the prior is already IN g_heat_loaded: only this session's routing is new
    t.heat_dec_merged  = t.heat_dec;
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





// Cache-aware whole-table staging.  One block per expert fills the
// staged destination from two sources: a resident expert (`prs[e]` a valid slot) is copied from its
// arena slot (the device slice, contiguous `expert_bytes`); any other expert comes from the pinned host
// alias (a 2-D strided copy for an axis-0 slice).  The destination is the ordinary staged table in
// expert order, so every consumer is unchanged.  `prs` is the immutable prefill residency snapshot
// (expert -> slot, -1 = not usable); it is never the decode policy and never mutated by decode.
//
// TODO #87 (2): a POOL-BACKED master has a third source.  `pool_dev` is the pinned pool's device alias and
// a snapshot value `<= -2` means "pool slot `(-v - 2)`", so the experts that live in either VRAM tier are
// filled by ONE launch instead of one `cudaMemcpyAsync` per expert (measured: 733 -> 836-class prefill on the
// field config, where the pinned-master path already staged with a single launch).  `skip_miss` makes a `-1`
// entry a no-op: the host has already copied those experts (they are in neither tier, so they come from the
// mapping/the disk) into the destination before this kernel runs on the same stream.
static __global__ void moe_cache_stage_kernel(char * __restrict__ dst, const char * __restrict__ arena,
                                              const char * __restrict__ host_dev, const int32_t * __restrict__ prs,
                                              int slots, int64_t expert_bytes, int64_t host_bytes,
                                              int64_t src_off, int64_t rows, int64_t row_bytes, int64_t host_pitch,
                                              int do2d, const char * __restrict__ pool_dev,
                                              int64_t pool_stride, int skip_miss) {
    const int e   = blockIdx.x;
    const int tid = threadIdx.x;
    const int nt  = blockDim.x;
    char *    d   = dst + (int64_t) e * expert_bytes;
    const int s   = prs != nullptr ? prs[e] : -1;
    if (s >= 0 && s < slots) {
        moe_cache_policy_copy(d, arena + (int64_t) s * expert_bytes, (size_t) expert_bytes, tid, nt);
        return;
    }
    const char * src = host_dev + (int64_t) e * host_bytes + src_off;
    if (s <= -2) {
        if (pool_dev == nullptr) {
            return;   // can't happen: the encoding is only produced when a pool exists
        }
        src = pool_dev + (int64_t) (-s - 2) * pool_stride + src_off;
    } else if (skip_miss) {
        return;       // a pool-backed staging already served this expert from the mapping/the disk
    }
    if (!do2d) {
        moe_cache_policy_copy(d, src, (size_t) expert_bytes, tid, nt);
        return;
    }
    for (int64_t r = 0; r < rows; r++) {
        moe_cache_policy_copy(d + (size_t) r * (size_t) row_bytes, src + (size_t) r * (size_t) host_pitch,
                              (size_t) row_bytes, tid, nt);
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
// wip/host-pool-speed (mechanism (b)): the gather resolves a routed expert from THREE sources, all of
// them safe to read in a kernel on a no-XNACK device: the arena slot (`prs[e] >= 0`), a POOL slot
// (`prs[e] <= -2`, a `cudaHostAlloc`'d pinned mapping, so no hold is needed), and -- only when the table
// has no pool, i.e. the legacy pinned-master gather -- the device-readable master.  An expert in NEITHER
// tier (`prs[e] == -1`) is copied into `dst` by the caller before this kernel runs (`skip_miss`), so the
// kernel never reads the pageable master: that read is what `INVARIANTS.md` (device-visible host memory
// must be RESIDENT) forbids.
static __global__ void moe_cache_gather_kernel(
        char * __restrict__ dst_base, const char * __restrict__ src_base,
        const int32_t * __restrict__ ids, size_t nb0, size_t nb1,
        int n_used, int n_tok, int n_experts, int n_split,
        int64_t expert_bytes, int64_t host_bytes, int64_t src_off, int64_t host_pitch, int split_axis,
        const char * __restrict__ arena, const int32_t * __restrict__ prs, int slots,
        const char * __restrict__ pool_dev, int64_t pool_stride, int skip_miss) {
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
    // A slot source is already in the DESTINATION's layout (the arena slot and the pool slot both hold one
    // device expert, contiguous `expert_bytes`), so it is always a flat copy.  `src_off`/`host_pitch` only
    // describe the host master's slice geometry.
    bool flat = (split_axis != 0 || host_pitch == 0);
    if (prs != nullptr) {
        const int r = prs[e];
        if (r >= 0 && r < slots && arena != nullptr) {
            src  = arena + (int64_t) r * expert_bytes;
            flat = true;
        } else if (r <= -2 && pool_dev != nullptr) {
            src = pool_dev + (int64_t) (-r - 2) * pool_stride + src_off;
        } else if (skip_miss) {
            return;   // the caller already copied this expert into `dst` (it is in neither tier)
        }
    }
    if (flat) {
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
    // A captured graph bakes this arena's address into the MoE kernels' launch parameters (the slot table
    // is a raw device pointer, not a tensor source, so `ggml_cuda_graph_update_required` cannot see it): a
    // capture that baked it in must never be replayed.
    ggml_cuda_graph_mem_freed(device);
}

// wip/moe-heatmap-persistence (R2): fold this table's device heat mirror into `heat`, then free the
// mirror.  Called wherever a table is torn down or migrated devices.  Caller holds `g_mutex`.
static void free_heat_dev_locked(table_t & t) {
    if (t.heat_dev == nullptr && t.heat_dec_dev == nullptr) {
        t.heat_dev_cap = 0;
        return;
    }
    if (t.device >= 0 && t.heat_dev_cap > 0) {
        device_guard dg(t.device);
        if (t.heat_dev != nullptr) {
            std::vector<uint32_t> tmp((size_t) t.heat_dev_cap, 0);
            if (cudaMemcpy(tmp.data(), t.heat_dev, (size_t) t.heat_dev_cap * sizeof(uint32_t),
                           cudaMemcpyDeviceToHost) == cudaSuccess) {
                if ((int) t.heat.size() != t.n_experts) {
                    t.heat.assign((size_t) t.n_experts, 0);
                }
                const int cap = t.n_experts < (int) t.heat_dev_cap ? t.n_experts : (int) t.heat_dev_cap;
                for (int e = 0; e < cap; e++) {
                    const uint64_t s = (uint64_t) t.heat[(size_t) e] + (uint64_t) tmp[(size_t) e];
                    t.heat[(size_t) e] = s > 0xffffffffull ? 0xffffffffu : (uint32_t) s;
                }
            } else {
                (void) cudaGetLastError();
            }
        }
        // TODO #88: the identity DECODE tally has its own mirror; fold it into the decode counter before
        // the mirror goes away, exactly as the combined one is folded above.
        if (t.heat_dec_dev != nullptr) {
            std::vector<uint32_t> tmp((size_t) t.heat_dev_cap, 0);
            if (cudaMemcpy(tmp.data(), t.heat_dec_dev, (size_t) t.heat_dev_cap * sizeof(uint32_t),
                           cudaMemcpyDeviceToHost) == cudaSuccess) {
                if ((int) t.heat_dec.size() != t.n_experts) {
                    t.heat_dec.assign((size_t) t.n_experts, 0);
                }
                const int cap = t.n_experts < (int) t.heat_dev_cap ? t.n_experts : (int) t.heat_dev_cap;
                for (int e = 0; e < cap; e++) {
                    const uint64_t s = (uint64_t) t.heat_dec[(size_t) e] + (uint64_t) tmp[(size_t) e];
                    t.heat_dec[(size_t) e] = s > 0xffffffffull ? 0xffffffffu : (uint32_t) s;
                }
            } else {
                (void) cudaGetLastError();
            }
        }
    }
    if (t.heat_dev != nullptr) {
        device_guard dg(t.device >= 0 ? t.device : 0);
        (void) cudaFree(t.heat_dev);
        (void) cudaGetLastError();   // fail soft
    }
    if (t.heat_dec_dev != nullptr) {
        device_guard dg(t.device >= 0 ? t.device : 0);
        (void) cudaFree(t.heat_dec_dev);
        (void) cudaGetLastError();   // fail soft
    }
    t.heat_dev      = nullptr;
    t.heat_dec_dev  = nullptr;
    t.heat_dev_cap  = 0;
    // `moe_cache_heat_kernel` receives both mirrors as raw launch parameters, so a captured decode-band
    // graph holds them without any tensor source naming them: never replay a capture that baked them in.
    ggml_cuda_graph_mem_freed(t.device >= 0 ? t.device : 0);
}

// Ensure a table's device heat mirror exists.  Caller holds `g_mutex`.  MUST run outside a captured
// graph: `cudaMalloc` is illegal during capture, so the mirror is pre-allocated at sizing and the
// decode band may only ever *launch* the tally kernel (TODO #64.2's sampled identity tally sits on the
// captured decode path -- allocating there aborted with "operation failed due to a previous error
// during capture").  Fail-soft: a failed allocation leaves the mirror null and the tally disabled.
static bool heat_dev_ensure_locked(table_t & t, int device) {
    if (t.heat_dev != nullptr) {
        return true;
    }
    if (t.n_experts <= 0) {
        return false;
    }
    device_guard dg(device >= 0 ? device : 0);
    void * p = nullptr;
    // Raw cudaMalloc, not `ggml_cuda_device_malloc`: this is ~1 KiB and must NOT be able to yield the
    // whole expert arena (the OOM choke point's fail-soft ladder).
    if (cudaMalloc(&p, (size_t) t.n_experts * sizeof(uint32_t)) != cudaSuccess) {
        (void) cudaGetLastError();
        static bool warned = false;
        if (!warned) {
            warned = true;
            GGML_LOG_WARN("%s: heat tally buffer alloc failed; the device routing tally is disabled for "
                          "this table\n", __func__);
        }
        return false;
    }
    if (cudaMemset(p, 0, (size_t) t.n_experts * sizeof(uint32_t)) != cudaSuccess) {
        (void) cudaGetLastError();
        (void) cudaFree(p);
        (void) cudaGetLastError();
        return false;
    }
    t.heat_dev     = (uint32_t *) p;
    t.heat_dev_cap = t.n_experts;
    // TODO #88: the identity DECODE tally gets a mirror of its own, so the fold can tell the bands apart
    // (the same table also receives PREFILL tallies in `heat_dev`).  Same raw cudaMalloc, same fail-soft:
    // without it the decode band simply keeps tallying into the combined vector.
    void * q = nullptr;
    if (cudaMalloc(&q, (size_t) t.n_experts * sizeof(uint32_t)) == cudaSuccess) {
        if (cudaMemset(q, 0, (size_t) t.n_experts * sizeof(uint32_t)) == cudaSuccess) {
            t.heat_dec_dev = (uint32_t *) q;
        } else {
            (void) cudaGetLastError();
            (void) cudaFree(q);
            (void) cudaGetLastError();
        }
    } else {
        (void) cudaGetLastError();
    }
    return true;
}

// WIP item 1 (TODO #42): release every buffer a table owns so it can be rebuilt at a different slot
// count (the layer-uniform re-size below).  Mirrors the allocations in `alloc_table_locked`; used only
// at sizing time, before the arena is read, but the identity/seed fills enqueue async copies on stream
// 0, so synchronize first.  `prefill_count` is deliberately KEPT: the re-seed needs it.
static void free_table_buffers_locked(table_t & t) {
    device_guard dg(t.device >= 0 ? t.device : 0);
    (void) cudaDeviceSynchronize();   // an identity/seed fill may still be in flight on stream 0
    // Everything below is a raw device pointer a captured graph can hold as a launch parameter: the arena
    // and the heat mirrors bump for themselves, the remap/routing staging does not.  Bump once here so no
    // buffer this function frees can be replayed against.
    ggml_cuda_graph_mem_freed(t.device >= 0 ? t.device : 0);
    if (t.arena != nullptr) {
        free_arena_backing(t.device, t.arena, t.arena_reserved, t.slots, t);
        t.arena_reserved = 0;
        t.arena = nullptr;
    }
    if (t.remap_dev != nullptr){ (void) cudaFree(t.remap_dev); t.remap_dev = nullptr; }
    devremap_free_locked(t);
    if (t.prs_dev != nullptr) { (void) cudaFree(t.prs_dev); t.prs_dev = nullptr; }
    t.prs_n = 0; t.prs_gen = 0; t.prs_arena = nullptr; t.prs_slots = 0;
    if (t.gprs_dev != nullptr) { (void) cudaFree(t.gprs_dev); t.gprs_dev = nullptr; }
    t.gprs_n = 0; t.gprs_arena = nullptr; t.gprs_slots = 0;
    std::fill(t.gprs_last.begin(), t.gprs_last.end(), -1);
    free_heat_dev_locked(t);   // folds the prefill tally into `heat` before the mirror goes away
    (void) cudaGetLastError();   // fail soft: a bad free must not abort a run

    g_arena_bytes -= (int64_t) t.slots * (int64_t) t.expert_bytes;
    if (g_arena_bytes < 0) {
        g_arena_bytes = 0;
    }
    t.slots          = 0;
    t.allocated      = false;
    t.identity       = false;
    t.remap_cap      = 0;
    t.remap_n_used   = 0;
    t.remap_n_tok    = 0;
    t.slot_dirty     = true;
    t.slot_expert.clear();
    t.slot_prov.clear();
    t.expert_slot.clear();
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
    if (g_pool_sink && slots >= t.n_experts) {
        g_sink_mirror_tables++;   // a full mirror: no exchange, no reserve, no demotions
    } else if (g_pool_sink && (int64_t) (t.n_experts - slots) * 25 < (int64_t) t.n_experts) {
        g_sink_near_mirror_tables++;   // gap < 4 % of the set: exchange DISENGAGED (TODO #281)
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
    if (slots == t.n_experts && t.host != nullptr && t.expert_bytes > 0 && heat_seed_above_floor(t)) {
        bool ok = true;
        for (int e = 0; e < t.n_experts && ok; ++e) {
            void * dst = (char *) t.arena + (size_t) e * t.expert_bytes;
            const cudaError_t err = fill_slot_from_host(t, dst, (size_t) e, (cudaStream_t) 0);
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
    // Prefill seed: bulk-admit the prompt's hottest experts before the first decode token (one-shot).
    // This sizing-time call only serves the gate-off/eager path.
    // wip/moe-heatmap-persistence: the PERSISTED cross-run prior is applied FIRST, so on a warm start
    // the arena begins with the hot set the previous run learned (and the in-run prefill seed, which is
    // inert under staging, only fills what is left).
    apply_heat_seed_locked(t);
    // TODO #64.2: pre-allocate the device routing-tally mirror here, at sizing (outside any captured
    // graph), so the sampled decode-band tally on an identity table only ever *launches* its kernel.
    if (g_heat_on) {
        (void) heat_dev_ensure_locked(t, t.device);
    }
    apply_prefill_seed_locked(t);
    // TODO #87: NOW the arena content is known (heat seed + prefill seed), so lay the host pool out as its
    // complement, ranked by the same prior.  Doing it here (not only at the first decode pass) is what
    // makes the PREFILL's staging read from the pool instead of the file.
    pool_rerank_from_heat_locked(t);
    // Account the count ACTUALLY allocated (`slots`/`t.slots`), not the requested `arena_slots`: on a
    // shrunk table the two differ and the old `arena_slots` accounting over-reported the arena.
    g_arena_bytes += (int64_t) slots * t.expert_bytes;
}

// Set when the sizing runs with host-expert tables: the workspace pool then keeps a free-VRAM floor
// (moe_cache_floor_active).
bool g_floor_active = false;

// WIP moe-upload-overlap: true when every host expert table the preflight saw has registered.  A pass
// can reach the decode hook with only a SUBSET of the host tables registered (a prefill's staged bulk
// ubatch does not register cache tables, and a non-staged 1-token tail primes only the last host
// layer), and the deferred sizing must not latch on that subset or the not-yet-seen tables get no arena
// forever.
//
// The invariant is the TOTAL host-expert bytes, NOT the per-device bytes.  A table registers once per
// device and the sum of its per-device slices is its host bytes, so the registered total reaches the
// preflight total iff the full set has registered.  The per-device split is decided by the meta backend
// at graph-build time and is not necessarily even -- `moe_host_expert_account` can only guess
// `bytes / n_dev`, while the real split of a 640-wide expert FFN dim here is 256/384 (measured gfx1201,
// qwen4exp, `-sm tensor`: 40 %/60 %).  Comparing per device against that guess never completes on such
// a split, so the arena is never sized (g_sized stays false) and the cache never serves a byte.  A
// partial pass (one host layer = 6 tables) stays far below the total, so this still catches the
// prefill-tail latch it was written for.  An absent preflight is treated as complete (the pre-existing
// behaviour).
static bool moe_cache_tables_complete_locked() {
    int64_t registered = 0;
    for (const table_t & t : g_tables) {
        if (t.device >= 0 && t.device < GGML_CUDA_MAX_DEVICES) {
            registered += (int64_t) t.n_experts * (int64_t) t.expert_bytes;
        }
    }
    int64_t expected = 0;
    for (int d = 0; d < GGML_CUDA_MAX_DEVICES; d++) {
        if (g_preflight_device_seen[d] && g_preflight_device_bytes[d] > 0) {
            expected += g_preflight_device_bytes[d];
        }
    }
    if (expected <= 0) {
        return true;
    }
    // Small slack for rounding between the loader's account and the sum of the registered tables: the
    // account is per host tensor (`ggml_nbytes`) while the cache adds `n_experts x expert_bytes` per
    // device, and under `-sm tensor` the loader can only guess the per-device proportion
    // (`bytes / n_dev`) that the Meta backend later decides exactly.  The TOTALS agree exactly for a
    // fully registered set, which is what this compares (see the invariant above).
    //
    // It must stay well below one table: an earlier `expected/128` version silently accepted a 35B-A3B
    // sizing pass with one 36.4 MiB table still unregistered (240 -> 239 tables, -3.6 % tg128), which is
    // exactly the partial-set latch this guard exists to prevent.  The `moe_host_expert_account` filter
    // is what keeps the gemma4 class (`ffn_down_exps.scale`, 512 B/layer, the reason an EXACT comparison
    // once failed) out of the expectation in the first place.
    const int64_t slack = moe_cache_min_i64(
        moe_cache_max_i64((int64_t) 1 << 20, expected / 4096),
        (int64_t) 16 << 20);
    return registered + slack >= expected;
}

// Deferred, budget-adaptive, UNIFORM sizing.  Every table gets the same slot count
// `budget / total_expert_bytes`, so an expert resident for gate is resident for up and down too and
// the effective (all-roles) hit rate is set by the shared slot count, not by the smallest table.
// `total_expert_bytes` is known once the first decode pass has registered every table; that first
// pass runs uncached (the scheduler copies the full experts), the second sizes and starts filling.
// r117 (TODO #93): prefer whole-table identity placement -- see the pre-pass in `alloc_all_locked`.
// Default ON; `MOE_EXPERT_CACHE_IDENTITY_PREF=0` restores the pre-r117 uniform placement (A/B only).
static bool moe_cache_identity_pref() {
    static const bool on = [] {
        const char * e = getenv("MOE_EXPERT_CACHE_IDENTITY_PREF");
        return e == nullptr || atoi(e) != 0;
    }();
    return on;
}

void alloc_all_locked() {
    g_write_seq++;
    if (g_sized) {
        return;
    }
    g_floor_active = g_enabled && !g_tables.empty();
    // Already disabled before the first graph (the slab could not give the cache a usable arena, or an
    // explicit `MOE_EXPERT_CACHE_MIB=0`): nothing to size.  See `moe_cache_disable_streaming` for why the
    // decision has to be made that early.
    if (!g_enabled) {
        g_sized = true;
        return;
    }
    // WIP moe-upload-overlap: do not latch the sizing on a partial table set (see the helper above).
    if (!moe_cache_tables_complete_locked()) {
        return;   // defer: `g_sized` stays false, so a later decode-band pass re-enters
    }
    g_sized = true;
    // Issue #136 (item B): the invariant is "cache enabled => slab armed".  When it is broken -- the slab
    // was configured on but never armed (backend-init ordering, a missing preflight, a per-device
    // decline) -- the arena is a raw cudaMalloc outside the movable boundary, which is the pre-slab crash
    // mode.  Never let that state be silent.  One-time, non-fatal; a no-op on non-HIP and dense models.
    if (!g_tables.empty() && !ggml_cuda_slab_enabled() && ggml_cuda_slab_configured()) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            GGML_LOG_WARN("%s: the MoE expert cache is enabled but the movable-boundary slab is NOT armed; "
                          "the arena is a raw cudaMalloc and a wide prefill can abort (the pre-slab failure "
                          "mode, issue #136).  Set GGML_CUDA_SLAB=0 to acknowledge the plain path, and report "
                          "a reproduction if this is unexpected.\n", __func__);
        }
    }
    // The slab's reserve was a guess made before the weights existed (`GGML_CUDA_SLAB_RESERVE_MIB`); this is
    // the first moment the weights / KV / draft are all resident, so measure what they did NOT need and
    // hand it to the arena before sizing.  Without it that VRAM sits idle (~1.9 GiB/card measured) even
    // though the cache is capped by its region.
    // -sm layer (every table unsplit, tables on more than one device): each device runs its layers' attention over
    // the whole context, so leave 6 GiB, not the default, free after the extension (255k needle: the later-layer
    // device ran out of VRAM with 4 GiB of headroom).
    {
        bool all_unsplit = !g_tables.empty();
        int  dev_mask = 0;
        for (const table_t & t : g_tables) {
            all_unsplit = all_unsplit && t.split_axis < 0;
            if (t.device >= 0 && t.device < 31) {
                dev_mask |= 1 << t.device;
            }
        }
        if (all_unsplit && (dev_mask & (dev_mask - 1)) != 0) {
            ggml_cuda_slab_headroom_at_least_mib(6144);
        }
    }
    // wip/first-prefill-latency: at the priming sizing the context is empty, so `slab_extend_all` maps
    // the reserve the first wide prefill's work view and outside-slab transients still need (measured:
    // 3.88 GiB mapped, leaving only the 4 GiB headroom, then a decode capture OOMs).  Skip it here; the
    // arena is then capped to what was reserved before the request's real memory existed.
    if (!g_alloc_priming) {
        ggml_cuda_slab_extend_all();
    }
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
    // the post-clamp budget each device settled on (the identity pre-pass spends it exactly, rounding
    // included, instead of holding back one arena unit per table)
    size_t budget_d[GGML_CUDA_MAX_DEVICES];
    for (int d = 0; d < GGML_CUDA_MAX_DEVICES; d++) {
        budget_d[d] = 0;
    }
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
        budget_d[d] = budget;
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
        const int64_t floor_bytes = moe_cache_max_i64(
            (int64_t) g_min_mib * 1024 * 1024,
            (g_min_res_pct > 0 && host_total > 0) ? host_total * g_min_res_pct / 100 : (int64_t) 0);
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

    // r117 (TODO #93): prefer WHOLE-TABLE identity over a uniform shortfall.
    //
    // A table is on the identity fast path only at its FULL slot count (`t.slots == t.n_experts`): the
    // decode band then reads the arena directly.  One slot short is not "almost": the table takes the
    // remap path and pays the per-token routing readback exactly like a table with no arena at all, so a
    // uniform 511/512 costs the fast path for EVERY table.  Measured on the field config (GSQ
    // `-ncmoe 48 -sm tensor`, 3 devices, plain decode): arena 40845.1 of 40925.0 MiB and 0/432 identity ->
    // **46.7 t/s**, vs 40925.0 and 432/432 -> **65.0 t/s** (+39 %); the same cliff cost the 2-GPU arm
    // 46.3 vs 66.4 t/s at `-c 204800`.  The shortfall is REAL memory (each table's arena is rounded up to
    // the slab arena unit), so the fix is placement, not accounting: when at least 2/3 of the LAYERS can
    // have their full slot count, give those layers full residency and drop the coldest layers entirely
    // to 0 slots (an existing, supported state -- a layer that cannot be uniform above 0 already runs
    // uncached).  The dropped layers pay one host copy per op; the kept ones stop paying the readback.
    //
    // `MOE_EXPERT_CACHE_IDENTITY_PREF=0` restores the pre-r117 uniform placement.
    bool identity_mode = false;
    int  layer_full[512];
    for (int i = 0; i < 512; i++) {
        layer_full[i] = -1;
    }
    if (moe_cache_identity_pref() && max_dev >= 0) {
        for (const table_t & t : g_tables) {
            if (t.allocated || t.layer < 0 || t.layer >= 512 || t.device < 0) {
                continue;
            }
            const int full = (int) t.n_experts;
            if (layer_full[t.layer] < 0 || full < layer_full[t.layer]) {
                layer_full[t.layer] = full;
            }
        }
        bool any_short = false;
        int  n_layers  = 0;
        for (int i = 0; i < 512; i++) {
            if (layer_full[i] < 0) {
                continue;
            }
            n_layers++;
            if (layer_slots[i] < layer_full[i]) {
                any_short = true;
            }
        }
        if (any_short && n_layers > 0) {
            // heat order, same ranking the allocation below uses (allocation order IS the eviction
            // policy), so the layers dropped here are the coldest ones
            std::map<int, uint64_t> layer_heat;
            for (int i = 0; i < 512; i++) {
                if (layer_full[i] < 0) {
                    continue;
                }
                uint64_t s = 0;
                for (const table_t & t : g_tables) {
                    if (!t.allocated && t.layer == i) {
                        for (uint32_t v : heat_rank_vec(t)) {
                            s += v;
                        }
                    }
                }
                layer_heat[i] = s;
            }
            std::vector<int> order;
            order.reserve((size_t) n_layers);
            for (int i = 0; i < 512; i++) {
                if (layer_full[i] >= 0) {
                    order.push_back(i);
                }
            }
            if (g_heat_on) {
                std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
                    const uint64_t ha = layer_heat[a];
                    const uint64_t hb = layer_heat[b];
                    if (ha != hb) {
                        return ha > hb;
                    }
                    return a < b;
                });
            }
            size_t remain[GGML_CUDA_MAX_DEVICES];
            for (int d = 0; d <= max_dev; d++) {
                remain[d] = budget_d[d];
            }
            int layer_slots_prev[512];
            memcpy(layer_slots_prev, layer_slots, sizeof(layer_slots));
            int n_full = 0;
            for (int layer : order) {
                const int full = layer_full[layer];
                size_t need[GGML_CUDA_MAX_DEVICES] = {0};
                bool   ok = full > 0;
                for (const table_t & t : g_tables) {
                    if (t.allocated || t.layer != layer || t.device < 0 || t.device > max_dev) {
                        continue;
                    }
                    size_t b = (size_t) full * (size_t) t.expert_bytes;
                    if (ggml_cuda_slab_enabled()) {
                        const size_t unit = (size_t) ggml_cuda_slab_arena_unit(t.device);
                        if (unit > 1) {
                            b = ((b + unit - 1) / unit) * unit;
                        }
                    }
                    need[t.device] += b;
                }
                for (int d = 0; d <= max_dev; d++) {
                    if (need[d] > remain[d]) {
                        ok = false;
                    }
                }
                if (ok) {
                    layer_slots[layer] = full;
                    for (int d = 0; d <= max_dev; d++) {
                        remain[d] -= need[d];
                    }
                    n_full++;
                } else {
                    layer_slots[layer] = 0;
                }
            }
            // Only accept it when the drop is SMALL.  A dropped layer is not free: it serves every expert
            // through the scheduler's host copy, per op, whereas a 511/512 table still serves 511/512 of
            // them from VRAM and pays only the readback.  Measured: 1 dropped layer of 48 is +27.6 %
            // (46.7 -> 59.6 t/s, 3 devices, `MOE_EXPERT_CACHE_MIB=13800`), while 6 of 48 is -30 %
            // (65.7 vs 93.4 t/s, 2 devices, `-c 204800`) -- so cap the drop at ~4 % of the layers.
            const int max_drop = std::max(1, n_layers / 24);
            if (n_full * 3 >= n_layers * 2 && n_layers - n_full <= max_drop) {
                identity_mode = true;
                GGML_LOG_WARN("%s: identity-aware placement: %d of %d layers at their FULL slot count, "
                              "%d dropped to 0 (a uniform shortfall would leave every table off the identity "
                              "fast path).  MOE_EXPERT_CACHE_IDENTITY_PREF=0 restores the uniform placement\n",
                              __func__, n_full, n_layers, n_layers - n_full);
            } else {
                memcpy(layer_slots, layer_slots_prev, sizeof(layer_slots));
            }
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
        // wip/moe-arena-defrag phase 1: place the HOTTEST layers first.  `ggml_cuda_slab_arena_alloc`
        // stacks top-down and the boundary moves bottom-up, so the first-allocated table ends up
        // HIGHEST in VA and is the LAST the work pool reaches -- allocation order IS the eviction
        // policy.  The persisted/in-run heat gives the order; without it this is the historical
        // ascending-layer order (the sort is skipped when no heat is present).  Frozen at sizing
        // (maintainer decision, 2026-10-14): a re-arm does not reshuffle.
        std::vector<int> layer_order;
        layer_order.reserve(by_layer.size());
        for (const auto & kv : by_layer) {
            layer_order.push_back(kv.first);
        }
        if (g_heat_on) {
            std::map<int, uint64_t> layer_heat;
            for (const auto & kv : by_layer) {
                uint64_t s = 0;
                for (int i : kv.second) {
                    // TODO #88: the DECODE band (the arena's own band), same source as the seed ranking.
                    for (uint32_t v : heat_rank_vec(g_tables[i])) {
                        s += v;
                    }
                }
                layer_heat[kv.first] = s;
            }
            std::stable_sort(layer_order.begin(), layer_order.end(), [&](int a, int b) {
                const uint64_t ha = layer_heat[a];
                const uint64_t hb = layer_heat[b];
                if (ha != hb) {
                    return ha > hb;
                }
                return a < b;   // deterministic tie-break
            });
        }
        for (int layer : layer_order) {
            std::vector<int> &ids   = by_layer[layer];
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
                if (identity_mode && requested > 0) {
                    // this layer was placed at its FULL slot count: if the full count does not fit, run the
                    // layer uncached (0 slots) and leave every other layer at identity -- do NOT spread a
                    // smaller uniform count, which costs the identity fast path for all of them
                    target = 0;
                    break;
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
    g_uniform_slots = max_dev >= 0 ? slots_d[0] : -1;
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
    // The traffic report's denominator: one routed-MoE op per (layer, role, device) table per token, so
    // the decode-band census divided by this is the token count.  Snapshotted HERE (sizing) rather than
    // read from `g_tables` at report time, because `llama_model_free` now drops the tables before the
    // exit report runs (r143) and the report then had no denominator at all.
    {
        int64_t n_band_tables = 0;
        for (const table_t & t : g_tables) {
            if (t.slots > 0) {
                n_band_tables++;
            }
        }
        g_band_n_tables.store(n_band_tables, std::memory_order_relaxed);
    }
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
    // TODO #269: latch each table's remap cold SOURCE while the layout is final and `g_sized` is still
    // false, so `devremap_ok_locked` never has to ask a question whose answer could change later (a graph
    // capture needs a per-shape constant).  A host pool that covers the tensor gives every expert a
    // resident slot, so the remap can encode a cold read as `slots + pool slot` even though the master is
    // pageable and has no device mapping -- which is the default `-lme auto` layout on a mapped load
    // ("a pinned per-EXPERT pool of the whole host expert set (a MIRROR)").
    for (table_t & t : g_tables) {
        t.dr_pool_ok = t.slots > 0 && !table_cold_ok(t) && pool_covers_table_peek_locked(t);
    }
}

} // namespace

static void devremap_batch_clear_locked();   // device remap: drop the pending sibling batches (defined with g_dr_batch)
static void report_locked();                 // the whole exit census (defined with `moe_cache_report` below)



// ---------------------------------------------------------------------------------------------
// Prefill-routing seed (MOE_EXPERT_CACHE_PREFILL_SEED=1): device tally
// ---------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------
// Coalesced copy helper (used by the device-side expert gather).
// ---------------------------------------------------------------------------------------------


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

// wip/apu-expert-pool: is the bounded pinned pool actually armed?  The pool is what serves every GPU read of a
// pool-backed (model-mapping) host master, so a caller must not hand out that master unless this is true: with
// no pool the GPU would read the pageable mapping in place, which is a fatal fault without XNACK.
bool moe_cache_host_pool_enabled() {
    return g_pool_enabled;
}

bool moe_cache_enabled() {
    return g_enabled;
}

bool moe_cache_floor_active() {
    return g_floor_active;
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
    // CUDA graphs must not be captured until the read source is constant (the arena sizing).
    return g_slots_hint > 0 || g_sized;
}

// wip/umem-arena-guard: is this device's "VRAM" the same physical pool as host RAM?
//
// On an integrated GPU (an APU, e.g. gfx1151 / Strix Halo) `hipDeviceProp_t::totalGlobalMem` IS the
// shared GTT window -- this box reports "Total VRAM: 124000 MiB" on a 128 GiB machine -- so the arena's
// AUTO sizing ("free VRAM") reads the whole host pool, and the weights, the pinned host-expert master
// and the arena all compete for one budget.  Measured (2026-10-19, 35B-A3B Q4_K_M, `-ncmoe 40`, no
// explicit MIB): a 123 GiB box went to 121 GiB resident + 13 GiB of swap before the load finished.
//
// There is also nothing to win: an expert in the host master is already device-reachable through the
// same physical memory, so a slot in the arena would hold a copy of bytes the device can read anyway.
// The default is therefore OFF on a shared pool; the host-expert path (`-ncmoe`/`-cmoe`) is unaffected
// and simply runs without the arena.  `MOE_EXPERT_CACHE_UNIFIED=1` re-enables it for platforms that do
// partition the two pools (e.g. Windows, where an APU can be capped to a fixed VRAM carve-out), but
// AUTO cannot size a shared pool, so that arm also requires an explicit `MOE_EXPERT_CACHE_MIB`.
static bool moe_cache_shared_pool_device(int device) {
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) {
        return false;
    }
    // Query the live property rather than `ggml_cuda_info().devices[].integrated`: that flag is forced
    // false on CUDA builds (upstream #15034) and honours `GGML_FORCE_NO_INTEGRATED` on HIP, and both
    // concern the host-buffer/UMA weight path.  The question here is memory topology, which is always
    // `prop.integrated` -- the same value `ggml_backend_cuda_device_get_type()` (and therefore the fit
    // reserve's `GGML_BACKEND_DEVICE_TYPE_IGPU` check) uses.
    device_guard dg(device);
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;   // fail-soft: keep the previous (unguarded) behaviour
    }
    return prop.integrated;
}

bool moe_cache_preflight(int device, size_t host_expert_bytes, size_t aux_reserve_bytes) {
    // Issue #136: `parse_env()` must not depend on `ggml_backend_cuda_init()` having run first.  With
    // `--fit off` the fit probe load is skipped, so this preflight runs before any backend init and
    // `g_enabled`/`g_auto` are still false; the old early return below then left `g_slab_armed` unset and
    // the expert cache ran on a raw cudaMalloc arena (the pre-slab crash mode).  Parse (once) now.
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        ensure_env();
    }
    if (!g_enabled) {
        return false;
    }
    // wip/umem-arena-guard: refuse the arena on a shared host/device memory pool (see the helper above).
    // This runs at model load, before any table registers, so the whole cache stack (arena, pool, heat
    // map, cache-band fusions) stays inert and the run behaves like `MOE_EXPERT_CACHE_MIB=0`; the
    // backend wrapper then also disarms the movable-boundary slab.  Latches via `g_enabled`, so a second
    // preflight (the `common` draft/MTP aux call) does not repeat the message.
    if (moe_cache_shared_pool_device(device)) {
        std::lock_guard<std::mutex> lock(g_mutex);
        // wip/apu-expert-pool: the UMA arm.  The arena was refused here because AUTO sizing reads "free VRAM"
        // -- on a shared pool that is the whole machine -- NOT because the tier is meaningless: on an APU the
        // experts that fit in host RAM already fit the device, but an expert set LARGER than RAM has no tier
        // at all without the arena.  So size it from a RAM figure instead and cap the movable-boundary slab to
        // the same number, which makes the over-commit (measured 121 GiB resident + 13 GiB swap, and a
        // `SLAB_INIT size(VA)=98.44 GiB` on a 123 GiB box) unreachable by construction rather than by an
        // operator remembering GGML_CUDA_SLAB_RESERVE_MIB.
        //   * MOE_EXPERT_CACHE_UMA=0            -> the pre-campaign refusal, verbatim (kill switch / A/B)
        //   * MOE_EXPERT_CACHE_MIB=N            -> authoritative, never re-scaled
        //   * else MOE_EXPERT_CACHE_UMA_RAM_PCT (60) % of MemAvailable, floored at the slab's arena minimum
        // The slab cap is load-bearing: without it `ggml_cuda_slab_init_locked` maps `free VRAM - reserve`,
        // which on a shared pool is the machine.
        static const bool uma_off = [] {
            const char * e = getenv("MOE_EXPERT_CACHE_UMA");
            return e != nullptr && atoi(e) == 0;
        }();
        bool uma_sized = false;
        if (g_unified == 0 && !uma_off) {
            size_t budget = 0;
            bool   from_ram = false;
            // `g_budget` is NOT a source of truth here: the `--fit` pass pre-fills it from free VRAM (measured:
            // 74 489 620 070 B = 69.4 GiB on a 123 GiB box), so an "explicit budget wins" rule that tests
            // `!g_auto` would adopt the machine and cap nothing.  Only the OPERATOR's MOE_EXPERT_CACHE_MIB is
            // authoritative; anything else is re-derived from RAM.
            const char * mib_env = getenv("MOE_EXPERT_CACHE_MIB");
            if (mib_env != nullptr && atoi(mib_env) > 0) {
                budget = (size_t) atoi(mib_env) * 1024 * 1024;
            } else {
                const int64_t avail = pool_available_ram_bytes();
                if (avail <= 0) {
                    g_enabled = false;
                    GGML_LOG_WARN("%s: MoE expert cache OFF on device %d: MOE_EXPERT_CACHE_UMA needs a RAM "
                                  "figure and MemAvailable could not be read; set MOE_EXPERT_CACHE_MIB=N "
                                  "MiB/device to size it explicitly\n", __func__, device);
                    return false;
                }
                const int pct = uma_vram_pct();
                const size_t room = (size_t) avail * (size_t) pct / 100;
                const size_t floor_b = (size_t) ggml_cuda_slab_min_arena_mib() * 1024 * 1024;
                budget = room > floor_b ? room : floor_b;
                from_ram = true;
                g_budget = budget;
                g_auto   = false;
            }
            ggml_cuda_slab_set_total_cap(budget);
            uma_sized = true;
            static std::atomic<bool> uma_logged{false};
            if (!uma_logged.exchange(true)) {
                GGML_LOG_INFO("%s: MoE expert cache ON on integrated device %d: arena budget %zu MiB/device "
                              "(%s%s), slab capped to it.  A shared host/device pool has no separate VRAM, so "
                              "the arena is a disk-backed cache over the same RAM and the slab may not map "
                              "more than the budget.  MOE_EXPERT_CACHE_UMA=0 restores the refusal.\n",
                              __func__, device, budget >> 20,
                              from_ram ? (uma_any_device() ? "MOE_UMA_VRAM_PCT of MemAvailable"
                                                            : "MemAvailable")
                                       : "explicit MOE_EXPERT_CACHE_MIB",
                              from_ram ? "" : ", authoritative");
            }
        }
        if (g_unified == 0 && !uma_sized) {
            g_enabled = false;
            GGML_LOG_WARN("%s: MoE expert cache OFF on device %d: integrated GPU whose \"VRAM\" is the "
                          "same pool as host RAM.  The arena would only copy bytes the device can already "
                          "read through the host master, and AUTO sizing (free VRAM) would read the whole "
                          "shared pool.  Experts that fit in host memory already fit the device here, so "
                          "the host-expert path is used instead (-ncmoe/-cmoe still work).  Set "
                          "MOE_EXPERT_CACHE_MIB=N (or MOE_EXPERT_CACHE_UMA=1, the default) to size the arena "
                          "from a RAM budget on a shared pool, or MOE_EXPERT_CACHE_UNIFIED=1 plus an explicit "
                          "MOE_EXPERT_CACHE_MIB and GGML_CUDA_SLAB_RESERVE_MIB for a platform that really "
                          "partitions host and device memory (e.g. a Windows APU with a fixed carve-out).\n",
                          __func__, device);
            return false;
        }
        if (g_unified != 0 && !uma_sized) {
            if (g_auto) {
                g_enabled = false;
                GGML_LOG_WARN("%s: MoE expert cache OFF on device %d: MOE_EXPERT_CACHE_UNIFIED=1 needs an "
                              "explicit MOE_EXPERT_CACHE_MIB, because AUTO sizes from \"free VRAM\" and on a "
                              "shared host/device pool that is host RAM (it would size the arena against the "
                              "whole machine).  Set MOE_EXPERT_CACHE_MIB=N MiB/device to override.\n",
                              __func__, device);
                return false;
            }
            if (getenv("GGML_CUDA_SLAB_RESERVE_MIB") == nullptr) {
                g_enabled = false;
                GGML_LOG_WARN("%s: MoE expert cache OFF on device %d: MOE_EXPERT_CACHE_UNIFIED=1 also needs "
                              "an explicit GGML_CUDA_SLAB_RESERVE_MIB.  The cache requires the "
                              "movable-boundary slab, and the slab maps `free VRAM - reserve`; on a shared "
                              "host/device pool \"free VRAM\" is host RAM, so an unset reserve maps the whole "
                              "machine.  Set GGML_CUDA_SLAB_RESERVE_MIB=<MiB to keep OUTSIDE the slab> as "
                              "well.\n", __func__, device);
                return false;
            }
            static std::atomic<bool> unified_logged{false};
            if (!unified_logged.exchange(true)) {
                GGML_LOG_WARN("%s: MOE_EXPERT_CACHE_UNIFIED=1: expert cache ON with MOE_EXPERT_CACHE_MIB=%zu "
                              "MiB/device on integrated device %d (shared host/device pool requested by the "
                              "user); this is a copy tier over host RAM unless the platform caps VRAM\n",
                              __func__, g_budget >> 20, device);
            }
        }
    }
    // WIP moe-upload-overlap: remember the per-device host-expert bytes so the deferred arena sizing can
    // wait until every host table has registered (see `moe_cache_tables_complete_locked`).  Recorded for
    // the explicit-budget path too (`g_auto` false returns below).
    if (device >= 0 && device < GGML_CUDA_MAX_DEVICES && host_expert_bytes > 0) {
        std::lock_guard<std::mutex> lock(g_mutex);
        // `max`: a second model (a separate MTP/draft head) may preflight the same devices with a
        // smaller host-expert set; the cache is process-global and both models' tables register, so
        // never let the smaller one lower the bar the deferred sizing waits for.
        if ((int64_t) host_expert_bytes > g_preflight_device_bytes[device]) {
            g_preflight_device_bytes[device] = (int64_t) host_expert_bytes;
        }
        g_preflight_device_seen [device] = true;
    }
    // The pool's AUTO budget is a fraction of the MoE host expert bytes.  This runs once per host-expert
    // device, before the context -- but it is RE-RUN for the same devices (the `--fit` probe, and a
    // second model's preflight for a separate draft/MTP head), so a running sum sized the pool at N x the
    // configured fraction (measured 3 calls -> `--host-experts pool` pinned 75 % of the host set instead
    // of 25 %).  Derive it from the per-device MAX instead, exactly like the completeness gate above.
    if (g_pool_enabled) {
        int64_t host_total = 0;
        for (int d = 0; d < GGML_CUDA_MAX_DEVICES; d++) {
            if (g_preflight_device_seen[d]) {
                host_total += g_preflight_device_bytes[d];
            }
        }
        g_preflight_host_bytes = host_total;
    }
    if (!g_auto) {
        return true;   // an explicit MOE_EXPERT_CACHE_MIB is used verbatim; no early floor
    }
    // The floor is a one-time early decision, but the aux estimate is only known from `common` (after the
    // model load).  Re-run it when a larger aux arrives so the load-time aux-0 answer cannot silently
    // keep a cache that the with-draft projection would have declined -- and re-run it while the cache is
    // DECLINED even for a smaller aux: a windowed MTP draft (LLAMA_SPEC_DRAFT_WINDOW) derives a much
    // smaller reserve than the flat constant the load-time call can only use, so that call is the
    // pessimistic one, and a decline it reached must not survive the real number.
    if (g_preflight_done && aux_reserve_bytes <= g_preflight_aux && g_enabled) {
        return g_enabled;   // latch while the cache is on: the call with the largest aux so far decides
    }
    g_preflight_done = true;
    g_preflight_aux = std::max(g_preflight_aux, aux_reserve_bytes);
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
    // wip/moe-heatmap-persistence: the debounced flush.  This is the server's per-turn seam, so a long
    // run writes the heat map at most once per `MOE_EXPERT_HEATMAP_DEBOUNCE_S` (default 60 s); the
    // process-exit flush in `moe_cache_report` is the backstop.
    if (g_heat_on && !g_heat_dead && g_heat_dirty) {
        const auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - g_heat_last_save).count() >=
            g_heat_debounce_s) {
            heat_pull_locked();
            heat_map_save_locked();
            g_heat_dirty = false;
        }
    }
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
    const int    old_device   = t.device;
    const int    old_slots    = t.slots;
    const size_t old_reserved = t.arena_reserved;
    void * const old_arena    = t.arena;
    void * const old_remap    = t.remap_dev;

    // TODO #70: the arena's backing is RETURNED, so the table must not keep the pointer.  A slab-backed
    // table used to keep `t.arena` here (on the theory that a VMM arena keeps its reserved VA across a
    // stand-down), but `ggml_cuda_slab_arena_free` puts the range back on the slab's free list -- the work
    // pool or another table can take it -- while `alloc_table_locked` reads a non-null `t.arena` as
    // "already allocated" and reuses it at the requested slot count.  A re-arm of a table that an
    // allocation-OOM stand-down had dropped therefore claimed slots in a range it no longer owned
    // (measured as `[validate device-malloc-yield] ... layer=46 slots=0..512` right before an eviction +
    // re-arm).  Clear it exactly as `free_table_buffers_locked` and `moe_cache_evict_slab_range` do, so a
    // re-arm re-allocates from the free list.
    t.arena          = nullptr;
    t.arena_reserved = 0;
    t.remap_dev    = nullptr;
    t.remap_cap    = 0;
    free_heat_dev_locked(t);   // keep the tally across a stand-down (folds the mirror into `heat`)
    t.remap_n_used = 0;
    t.remap_n_tok  = 0;
    t.slots        = 0;
    t.slot_dirty   = true;
    t.slot_expert.clear();
    t.expert_slot.clear();
    g_arena_bytes -= (int64_t) old_slots * (int64_t) t.expert_bytes;
    if (g_arena_bytes < 0) {
        g_arena_bytes = 0;
    }
    if (old_device >= 0) {
        moe_cache_sync_devices_locked();   // no kernel that reads this arena may still be in flight
        device_guard dg(old_device);
        // The arena bumps for itself, the remap staging is a raw launch parameter: bump once for the
        // stand-down so neither can be replayed against a captured graph.
        ggml_cuda_graph_mem_freed(old_device);
        if (old_arena != nullptr) {
            // The slab OWNS the arena's VA, so this only returns the range to the slab's free list (the
            // work pool can then take it, or the arena can hand it to another table); the plain path's
            // arena is a cudaMalloc the table owns outright.
            free_arena_backing(old_device, old_arena, old_reserved, old_slots, t);
        }
        if (old_remap != nullptr) {
            (void) cudaFree(old_remap);
        }
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

bool moe_cache_shrink_step(int device) {
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
        // Device-scoped (issue #136, item D): a failed allocation on one device must not stand down
        // another device's residency.  Only tables whose kernels run on the failing device can help.
        if (t.device != device) {
            continue;
        }
        if (t.arena != nullptr && t.slots > 0) {
            const int64_t b = (int64_t) t.slots * (int64_t) t.expert_bytes;
            if (b > best_b) {
                best_b = b;
                best   = i;
            }
        }
    }
    if (best < 0) {
        // Nothing to free on THIS device.  Do NOT disable the whole cache: the caller then tries the
        // whole-arena release, which is the correct last resort (a fault on device 1 must not disable
        // device 0's cache).
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
                ggml_cuda_graph_mem_freed(t.device);
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
    if (freed_b == 0) {
        // Nothing was actually released, so there is nothing to report and no point in the caller's retry
        // (a dense model has no expert table at all: log-only noise of the same class as r125's gated
        // "flat draft reserve" line, issue #216).
        return false;
    }
    GGML_LOG_WARN("%s: released the MoE expert cache arena (%.1f MiB) to satisfy a compute allocation; "
                  "the expert cache is disabled for the rest of this run (experts come from the host)\n",
                  __func__, (double) freed_b / (1024.0 * 1024.0));
    return true;
}

// r143: drop the tables of a model that is being freed.
//
// The expert cache is PROCESS-GLOBAL state, but its tables point at host expert masters the MODEL owns, and
// nothing used to tell the cache when those masters went away.  `llama_model_free` frees them; the deferred
// arena sizing (`alloc_all_locked`, which fills EVERY registered table from its master) then read a freed
// pointer -- a host memcpy SIGSEGV in `fill_slot_from_host`, from a table belonging to a model that no
// longer existed.  Measured (Flash-Next GSQ-IQ3_XXS, 2 GPU `-sm layer -ts 59,41 -ncmoe 24`, gfx1201):
// llama-bench runs one test per `-ts` value, frees the model between them (the params differ) and the
// second test's FIRST band pass crashed while filling the first model's tables.  The same sequence happens
// on any model swap (`llama-server` reload, an API user freeing and reloading) -- and the host buffer free
// hook alone would not cover `--host-experts pool`, whose master is the read-only model mapping.
//
// So dropping a device's tables is a lifecycle operation, not a cache policy:
//   * free every buffer a table owns (arena slot range, remap staging, heat mirror, bounce ring) and the
//     host pool keyed by its master (that one holds PINNED memory);
//   * erase the tables and re-point every id map, so a `ggml_tensor*` the allocator reuses later registers
//     a FRESH table instead of inheriting a dead model's geometry;
//   * forget the per-model preflight accounting (expected host bytes, the unregisterable set, the latched
//     auto floor) and un-size the cache: the arena was sized -- and possibly seated in the movable-boundary
//     slab -- for the model that just went away, so the next model sizes on its own first band pass instead
//     of silently running uncached.
// Capture the report's structural aggregates before the tables (and their host pools) are dropped.  Taken once,
// at the first forget: a per-device teardown still sees every device's table, so the first snapshot is the whole
// run.  Caller holds `g_mutex`.
static void report_snapshot_locked() {
    if (g_report_snap.have) {
        return;
    }
    g_report_snap.have = true;
    for (const table_t & t : g_tables) {
        g_report_snap.hits += t.hits;
        g_report_snap.misses += t.misses;
        g_report_snap.fills += t.fills;
        g_report_snap.evictions += t.evictions;
        if (t.slots > 0) {
            g_report_snap.n_tables++;
            g_report_snap.n_slots += t.slots;
            if (t.device >= 0 && t.device < GGML_CUDA_MAX_DEVICES) {
                g_report_snap.dev_slots [t.device] += t.slots;
                g_report_snap.dev_arena [t.device] += (int64_t) t.slots * (int64_t) t.expert_bytes;
                g_report_snap.dev_tables[t.device]++;
            }
        }
    }
    g_report_snap.arena_bytes = g_arena_bytes;
    g_report_snap.n_pools     = (int64_t) g_host_pools.size();
    for (const auto & kv : g_host_pools) {
        const host_pool_t & p = kv.second;
        g_report_snap.p_slots  += p.slots;
        g_report_snap.p_hits   += p.hits;
        g_report_snap.p_misses += p.misses;
        g_report_snap.p_fills  += p.fills;
        g_report_snap.p_evict  += p.evictions;
        g_report_snap.p_bytes  += (int64_t) p.slots * (int64_t) p.host_bytes;
    }
}

void moe_cache_forget_all(int device) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_tables.empty()) {
        return;
    }
    report_snapshot_locked();   // before anything is un-sized: the exit-time report describes the RUN
    // issue #274 follow-up (@DanoPTT): PRINT the census here, not only from the `atexit` hook.  On Windows a
    // llama-server shutdown runs `atexit` from inside ggml-hip.dll after `main` has returned, by which time the
    // server's async log is gone -- the reporter's clean Ctrl+C printed every other teardown line and not one
    // MoE census line.  The model teardown is the last moment when the tables (hence the counters) exist AND
    // the log is still open, so the full report is emitted on the first teardown call; the hook then has
    // nothing left to print.  `report_locked` also pulls and saves the heat tallies, in that order, so the
    // "this session" coverage line below is computed before the prior absorbs it.
    if (!g_report_printed) {
        g_report_printed = true;
        report_locked();
    }
    // Flush the heat tallies HERE, while the tables they live on still exist.  The exit-time save in
    // `moe_cache_report` walks `g_tables`, so a teardown that dropped them first wrote a HEADER-ONLY sidecar --
    // and because that file is then a valid empty map, it silently TRUNCATED a curated writable prior to a
    // 32-byte stub on any run short enough that the 60 s debounce never fired (measured 2026-11-06: a 57 s
    // llama-cli run did exactly that).  The later flush finds nothing dirty and leaves the file alone.
    // (`report_locked` above has already done this and cleared the flag, so on the normal path this is inert.)
    if (g_heat_on && !g_heat_dead && g_heat_dirty) {
        heat_pull_locked();
        heat_map_save_locked();
        g_heat_dirty = false;
    }
    std::vector<char> dropped(g_tables.size(), 0);
    std::vector<int>  drop;
    for (int i = 0; i < (int) g_tables.size(); i++) {
        if (device < 0 || g_tables[i].device == device) {
            dropped[(size_t) i] = 1;
            drop.push_back(i);
        }
    }
    if (drop.empty()) {
        return;
    }
    // Same hazard as the arena release: the fused MoE kernels carry the arena address in their launch
    // parameters, so nothing may be freed while one is still in flight.
    moe_cache_sync_devices_locked();

    int64_t freed_b = 0;
    std::set<const void *> dead_hosts;
    for (int i : drop) {
        table_t & t = g_tables[i];
        freed_b += (int64_t) t.slots * (int64_t) t.expert_bytes;
        if (t.host != nullptr) {
            dead_hosts.insert(t.host);
        }
        free_table_buffers_locked(t);   // arena slot range, remap/prs staging, heat mirror, g_arena_bytes
        if (!t.bounce.empty()) {
            device_guard dg(t.device >= 0 ? t.device : 0);
            for (bounce_t & b : t.bounce) {
                if (b.ev != nullptr) {
                    (void) cudaEventDestroy(b.ev);
                }
                if (b.host != nullptr) {
                    (void) cudaFreeHost(b.host);
                }
            }
            (void) cudaGetLastError();
            t.bounce.clear();
        }
        if (t.src_fd >= 0) {
            pool_close_file(t.src_fd);
            t.src_fd = -1;
        }
    }
    // The host pool is keyed by the master and holds PINNED memory: it must not outlive the model either.
    for (const void * h : dead_hosts) {
        const auto it = g_host_pools.find(h);
        if (it != g_host_pools.end()) {
            pool_free_locked(it->second);
            g_host_pools.erase(it);
        }
        g_host_src.erase(h);
    }

    // Erase the tables, then re-point every id map (the ids are `g_tables` indices).
    std::vector<int>     new_id(g_tables.size(), -1);
    std::vector<table_t> kept;
    kept.reserve(g_tables.size() - drop.size());
    for (int i = 0; i < (int) g_tables.size(); i++) {
        if (dropped[(size_t) i]) {
            continue;
        }
        new_id[(size_t) i] = (int) kept.size();
        kept.push_back(std::move(g_tables[(size_t) i]));
    }
    g_tables = std::move(kept);
    for (auto it = g_key_to_id.begin(); it != g_key_to_id.end(); ) {
        const int nid = new_id[(size_t) it->second];
        if (nid < 0) { it = g_key_to_id.erase(it); } else { it->second = nid; ++it; }
    }
    for (auto it = g_sem_to_id.begin(); it != g_sem_to_id.end(); ) {
        const int nid = new_id[(size_t) it->second];
        if (nid < 0) { it = g_sem_to_id.erase(it); } else { it->second = nid; ++it; }
    }
    for (auto it = g_alias_to_id.begin(); it != g_alias_to_id.end(); ) {
        const int nid = (it->second >= 0 && (size_t) it->second < new_id.size()) ? new_id[(size_t) it->second] : -1;
        if (nid < 0) { it = g_alias_to_id.erase(it); } else { it->second = nid; ++it; }
    }
    for (auto it = g_dr_key_to_id.begin(); it != g_dr_key_to_id.end(); ) {
        const int nid = (it->second >= 0 && (size_t) it->second < new_id.size()) ? new_id[(size_t) it->second] : -1;
        if (nid < 0) { it = g_dr_key_to_id.erase(it); } else { it->second = nid; ++it; }
    }
    devremap_batch_clear_locked();   // the pending sibling ids are g_tables indices too
    // The expert-head zeroing ledger is keyed by a DEVICE buffer that just went away with the model, and a
    // new allocation can land on the same VA: a stale entry would skip the head zero (the NaN over-read the
    // guard exists for).  Clearing it only costs one redundant memset.
    g_heads_zeroed.clear();
    // Per-model preflight accounting (see `moe_cache_tables_complete_locked`): the expectations were the
    // dead model's host bytes and the unregisterable set is keyed by ITS tensor data pointers, which the
    // allocator will reuse.  The next model's load preflights again and re-derives them.
    for (int d = 0; d < GGML_CUDA_MAX_DEVICES; d++) {
        if (device < 0 || d == device) {
            g_preflight_device_bytes[d] = 0;
            g_preflight_device_seen [d] = false;
        }
    }
    g_unregisterable_keys.clear();
    g_preflight_done = false;
    g_preflight_aux  = 0;
    // The arena was sized for the model that just went away; un-size it so the next model sizes again.
    g_sized = false;

    GGML_LOG_WARN("%s: model teardown: dropped %zu expert-cache table(s) (%.1f MiB of arena) on device %d "
                  "and un-sized the cache, so the next model sizes on its own first band pass\n",
                  __func__, drop.size(), (double) freed_b / (1024.0 * 1024.0), device);
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
//
// TODO #70: a re-arm is LAYER-UNIFORM OR THE LAYER IS STOOD DOWN, never partial.  The arena a re-arm sizes
// against can be smaller than the one the sizing pass saw (that is the whole reason a table was evicted),
// so a layer's tables can each land on a different slot count; the gate+up fused kernel indexes the gate
// lane with the *up* table's remap, so a mixed-slot layer reads the wrong expert for one role.  The sizing
// path answers that with a safety net that disables the cache outright -- which a re-arm must NOT do (the
// graphs are already built against the tables, and a late `g_enabled = false` desynchronises the MTP draft
// from the target).  So: converge every table of the layer on one count (retrying at the largest count any
// of them reached), and if that cannot be made uniform stand the WHOLE layer down (every table at 0
// slots).  A stood-down table makes `moe_cache_has_arena_locked()` false, so the cache-aware fusions stand
// down for every table and each per-table consumer declines -- the fuse-guard invariant.

// issue #278: can this table resolve a host source for an armed slot AT ALL, without the pinned tier
// having to grow?  A pinned master is always a source (a plain H2D).  A PAGEABLE master needs the host
// pool -- a resident slot or an EMPTY one, both of which are memory the driver already granted -- or a
// bounce ring entry.  A latched pinned tier (`g_pinned_off`) can grow neither, so a table with no pool and
// no ring entry has no source at all.
//
// This exists for the re-arm: arming slots that no fill can reach takes arena VRAM, syncs the device once
// per table and then issues one doomed fill per seeded slot, and it changes NO downstream decision -- a
// table that cannot fill declines the op exactly as a stood-down one does.  Measured shape (issue #278,
// Windows): the pinned tier latches off during the prefill, and the post-prefill re-arm still re-takes
// 5.5 GiB of the arena for 39 tables, none of which can ever be filled.
static bool table_has_fill_source_locked(const table_t & t) {
    if (t.host == nullptr || t.host_bytes == 0) {
        return false;
    }
    if (!table_master_pageable(t)) {
        return true;   // a pinned master: an H2D source, no allocation needed
    }
    const auto it = g_host_pools.find(t.host);
    if (it != g_host_pools.end() && !it->second.failed) {
        return true;   // the pool owns pinned slots already: a hit or an empty slot asks for nothing new
    }
    if (!g_pinned_off || !t.bounce.empty()) {
        return true;   // the ring may still grow, or it already holds an entry that is reused
    }
    return false;
}

// TODO #70: bring ONE layer's tables to a uniform slot count, or stand the whole layer down.  Returns the
// count the layer ended at (0 = stood down), or -1 when nothing was attempted.
static int moe_cache_rearm_layer_locked(int layer, std::vector<int> & ids, int want) {
    // largest expert slice first: a big table needs one contiguous block, and it gets the best chance
    // while the layer's own memory is still free
    std::stable_sort(ids.begin(), ids.end(), [](int a, int b) {
        return g_tables[a].expert_bytes > g_tables[b].expert_bytes;
    });
    // Undo every table's current allocation so the next attempt starts from a clean arena.  A table that is
    // already at 0 slots owns no backing (a stand-down hands the range back and clears the pointer), so it is
    // left alone and `alloc_table_locked` re-allocates it from the slab's free list on this attempt.
    const auto reset_layer = [&ids]() {
        for (int i : ids) {
            if (g_tables[i].slots > 0) {
                free_table_buffers_locked(g_tables[i]);
            }
        }
    };
    int target = want;
    int got    = -1;
    for (int attempt = 0; attempt < 8; ++attempt) {
        reset_layer();
        for (int i : ids) {
            alloc_table_locked(g_tables[i], target);
        }
        int mn = target;
        int mx = 0;
        for (int i : ids) {
            mn = std::min(mn, g_tables[i].slots);
            mx = std::max(mx, g_tables[i].slots);
        }
        if (mn == mx) {
            got = mn;   // uniform (possibly 0): the layer converged
            break;
        }
        if (mx < target) {
            target = mx;            // every table came up short: retry at the best count any of them reached
        } else if (mn > 0) {
            target = mn;            // mixed with a non-zero floor: retry at the count all of them reached
        } else if (target > 1) {
            target = target / 2;    // a table could not hold even one slot: back off and retry
        } else {
            break;                  // nothing can hold a slot: stand the layer down
        }
    }
    if (got < 0) {
        // Attempts exhausted with the layer still mixed.  Never run it: stand every table of the layer down.
        reset_layer();
        for (int i : ids) {
            alloc_table_locked(g_tables[i], 0);
        }
        GGML_LOG_ERROR("%s: layer %d could not be re-armed slot-uniformly (wanted %d slots); standing the "
                       "whole layer down (%zu tables at 0 slots) rather than running a mixed-slot layer\n",
                       __func__, layer, want, ids.size());
        return 0;
    }
    if (got > 0 && got < want) {
        GGML_LOG_WARN("%s: layer %d re-armed uniformly at %d slots (the sizing pass had %d): the arena is "
                      "smaller than it was then\n", __func__, layer, got, want);
    }
    return got;
}

bool moe_cache_rearm() {
    g_write_seq++;
    if (!g_enabled) {
        return false;   // released wholesale, or never sized: nothing to re-arm against
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_sized || g_tables.empty()) {
        return false;
    }

    // Which layers have stood-down tables?  A layer is re-armed as a unit: gate/up/down must stay
    // slot-uniform, or the shared remap points a role at the wrong expert.  A stood-down table is one that
    // was ALLOCATED but currently has no slots (0 is a legal state: the table holds no backing and every
    // consumer declines it).
    //
    // TODO #70: a layer that landed at a SMALLER uniform count is deliberately NOT revisited here.  Nothing
    // lowers `g_rearm_slots[layer]`, so the target is re-read from the sizing count at every drop (there is
    // no arithmetic ratchet), and this re-arm runs at the prefill -> decode drop, i.e. with the arena at its
    // per-request maximum: a shortfall means the arena really is smaller from now on (a later context's KV or
    // compute layout took it, and those do not give it back), so a retry would only re-fill the layer for a
    // count it cannot hold.  Retrying a deficit layer every drop would also zero its residents each time.
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
    // The layer's tables that kept their slots are re-armed with it: left at their old size they would make the
    // layer non-uniform (the eviction drops only the tables whose storage lay in the taken range).
    for (int i = 0; i < (int) g_tables.size(); ++i) {
        const table_t & t = g_tables[i];
        const auto it = by_layer.find(t.layer);
        if (it != by_layer.end() && t.device >= 0 && t.allocated && t.slots > 0) {
            it->second.push_back(i);
        }
    }

    int n_rearmed   = 0;
    int n_stooddown = 0;
    int n_nosrc     = 0;
    int n_nosrc_layers = 0;
    for (auto & kv : by_layer) {
        const int         layer = kv.first;
        std::vector<int> &ids   = kv.second;
        const int want = (layer >= 0 && layer < 512) ? g_rearm_slots[layer] : -1;
        if (want <= 0) {
            continue;   // the layer was not cacheable at sizing time
        }
        // issue #278: never arm a layer that cannot fill.  The test is LAYER-level, not per table, because the
        // arm is layer-uniform: a skipped table sandwiched between armed ones is exactly the mixed-slot layer
        // that makes the fused gate+up kernel read the wrong expert.  So a layer with no fill source is stood
        // down WHOLESALE (the survivors included -- they cannot fill either, so their slots are dead weight),
        // which is uniform and is what the safety net below would do anyway.
        bool any_src = false;
        for (int i : ids) {
            if (table_has_fill_source_locked(g_tables[i])) {
                any_src = true;
                break;
            }
        }
        if (!any_src) {
            for (int i : ids) {
                if (g_tables[i].slots > 0) {
                    free_table_buffers_locked(g_tables[i]);
                    alloc_table_locked(g_tables[i], 0);   // the legal stood-down state (uniform 0 slots)
                }
            }
            // Latched for the rest of the run (the condition is `g_pinned_off` plus no pool and no ring entry),
            // so this layer can never be armed again: remember that, which also keeps the caller's second
            // re-arm pass (one per device loop) from repeating the line for a layer this pass already took down.
            g_rearm_slots[layer] = 0;
            n_nosrc += (int) ids.size();
            n_nosrc_layers++;
            continue;
        }
        const int got = moe_cache_rearm_layer_locked(layer, ids, want);
        if (got > 0) {
            n_rearmed += (int) ids.size();
        } else if (got == 0) {
            n_stooddown += (int) ids.size();
        }
    }
    // TODO #70 safety net (mirrors the sizing pass's, but LOCAL to the layer): the convergence above is
    // constructed to leave every touched layer uniform, so a non-uniform one is a bug.  Stand that layer
    // down instead of disabling the cache (the graphs already reference the tables) -- a mixed-slot layer
    // is the one state that silently reads the wrong expert.
    for (auto & kv : by_layer) {
        int mn = -1, mx = -1;
        for (int i : kv.second) {
            const int s = g_tables[i].slots;
            mn = mn < 0 ? s : std::min(mn, s);
            mx = mx < 0 ? s : std::max(mx, s);
        }
        if (mn < 0 || mn == mx) {
            continue;
        }
        GGML_LOG_ERROR("%s: layer %d is not slot-uniform after the re-arm (%d..%d); standing it down rather "
                       "than risking a role mismatch\n", __func__, kv.first, mn, mx);
        for (int i : kv.second) {
            if (g_tables[i].slots > 0) {
                free_table_buffers_locked(g_tables[i]);
            }
            alloc_table_locked(g_tables[i], 0);
        }
    }

    if (n_rearmed > 0 || n_stooddown > 0) {
        GGML_LOG_WARN("%s: re-armed %d stood-down expert-cache tables after a compute-buffer drop "
                      "(%d stood down: the arena could not hold the layer uniformly) (arena now %.1f MiB)\n",
                      __func__, n_rearmed, n_stooddown, (double) g_arena_bytes / (1024 * 1024));
    }
    if (n_nosrc > 0) {
        // issue #278: name it, so "the re-arm did nothing" is a decision in the log rather than a mystery.
        GGML_LOG_WARN("%s: left %d stood-down expert-cache tables in %d layer(s) DISARMED: %s, and none of "
                      "their tables has a host pool, so no armed slot could ever be filled (arming them would "
                      "take the arena and fail every seed fill; the ops decline exactly as they did while "
                      "stood down) (arena now %.1f MiB)\n",
                      __func__, n_nosrc, n_nosrc_layers,
                      g_pinned_off ? (g_pinned_off_why != nullptr ? g_pinned_off_why
                                                                 : "the pinned host tier is latched off")
                                   : "no host source can be resolved",
                      (double) g_arena_bytes / (1024 * 1024));
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
    g_write_seq++;
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
    int    n_down = 0;
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
        // device buffers are separate cudaMalloc allocations and are kept, so a
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
        t.slot_dirty     = true;
        t.slot_expert.clear();
        t.expert_slot.clear();
        t.allocated      = true;   // it was decided; a re-arm can rebuild it
        n_down++;
    }
    if (freed > 0) {
        // TODO #70: a captured graph bakes a table's arena address into the MoE kernels' launch parameters
        // (the arena is a raw device pointer, not a tensor source, so `ggml_cuda_graph_update_required`
        // cannot see it).  The bytes just went back to the slab's free list -- where the work pool or
        // another table can take them -- so no capture that baked them in may be replayed.  This is the
        // same invalidation `free_arena_backing` performs for the paths that go through it; this hook hands
        // the range back to the slab directly and so must bump for itself (belt and braces: a wide -> narrow
        // drop also shrinks the boundary and bumps through `work_alloc`, but a NARROW boundary move -- a
        // wider verify view, a wide-aux grow -- evicts without any other bump).
        ggml_cuda_graph_mem_freed(device);
        // Name the table count as well as the bytes (TODO #271): a stood-down table is not staged during the
        // prefill (it has no arena to read), so this number is the size of the prefill's switch to the
        // scheduler's own whole-table upload -- otherwise the two lines are in different parts of the log
        // and neither says the other exists.
        GGML_LOG_WARN("%s: evicted %.1f MiB of arena tables from the taken slab chunks (arena now %.1f MiB; "
                      "%d table(s) stood down until the compute-buffer drop re-arms them)\n",
                      __func__, (double) freed / (1024 * 1024), (double) g_arena_bytes / (1024 * 1024), n_down);
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


// Prefill-first workloads never armed the arena: llama-bench `-p`, llama-perplexity and the embedding
// tools have no <=8-token warmup pass, the deferred sizing is gated on one, and the staging ring takes a
// host-expert table away from the cache entirely (the ring's whole-table upload never calls the cache, so
// the table set is never even observed).  The scheduler asks this while deciding whether to stage a routed
// host-expert table: TRUE means "leave it on the host path for now, the cache must register, prime and
// size on this table set first".  It answers false for a device that expects no host expert tables (a
// dense model, `-ncmoe 0`), so the ring is untouched there.
bool moe_cache_pending(int device) {
    if (!g_enabled) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES || !g_preflight_device_seen[device]) {
        return false;
    }
    return !g_sized;
}

// wip/slab-ring-region: has the deferred arena sizing run?  The slab's H2D-ring hole must stay ARMED
// through sizing (so the auto-sized arena never allocates its highest tables into the hole); a disarm
// before that makes the next arm evict those hot tables, and the cold re-arm corrupts a wide consumer.
bool moe_cache_is_sized() {
    if (!g_enabled) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_sized;
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
        if (t.remap_dev == nullptr && t.remap_cap != 0) {
            GGML_LOG_ERROR("[validate %s] DANGLING-REMAP layer=%d remap_dev=null remap_cap=%lld\n",
                           where, t.layer, (long long) t.remap_cap);
        }
    }
    if (bytes != g_arena_bytes) {
        GGML_LOG_ERROR("[validate %s] ARENA-BYTES g_arena_bytes=%lld sum(slots*expert_bytes)=%lld\n",
                       where, (long long) g_arena_bytes, (long long) bytes);
    }
    // TODO #70: a layer's tables must agree on their slot count (and, under `-sm tensor`, across devices).
    // The gate+up fused kernel indexes the gate lane with the up table's remap, so two NON-ZERO counts in one
    // layer would read the wrong expert there; a table at 0 slots is legal (it declines every consumer, and
    // the identity-only fusion guard then stands the pair's fusion down).  Reported at WARN, not counted as
    // inconsistent, because `moe_cache_shrink_step` (the device-allocation OOM chain) stands ONE table down
    // on purpose, so a mixed layer is a legitimate emergency state -- what must not happen is a PARTIAL
    // count differing from its partners', which is what the re-arm's convergence now prevents.
    {
        int mn_of[512];
        int mx_of[512];
        int nz_of[512];   // the smallest NON-ZERO count, i.e. the count the served tables actually use
        for (int i = 0; i < 512; i++) {
            mn_of[i] = -1;
            mx_of[i] = -1;
            nz_of[i] = -1;
        }
        for (const table_t & t : g_tables) {
            if (t.device < 0 || t.layer < 0 || t.layer >= 512 || !t.allocated) {
                continue;
            }
            mn_of[t.layer] = mn_of[t.layer] < 0 ? t.slots : std::min(mn_of[t.layer], t.slots);
            mx_of[t.layer] = mx_of[t.layer] < 0 ? t.slots : std::max(mx_of[t.layer], t.slots);
            if (t.slots > 0) {
                nz_of[t.layer] = nz_of[t.layer] < 0 ? t.slots : std::min(nz_of[t.layer], t.slots);
            }
        }
        for (int l = 0; l < 512; l++) {
            if (mn_of[l] < 0 || mn_of[l] == mx_of[l]) {
                continue;
            }
            if (nz_of[l] >= 0 && nz_of[l] == mx_of[l]) {
                GGML_LOG_WARN("[validate %s] MIXED-LAYER layer=%d slots=%d..%d (a stood-down role; the served "
                              "tables still agree, the cache-aware fusions are per-table)\n",
                              where, l, mn_of[l], mx_of[l]);
                continue;
            }
            n_bad++;
            GGML_LOG_ERROR("[validate %s] NONUNIFORM-LAYER layer=%d slots=%d..%d (two DIFFERENT non-zero "
                           "counts: a fused gate/up lane would read the wrong expert)\n",
                           where, l, mn_of[l], mx_of[l]);
        }
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

    if (level < 3) {
        return;
    }
    // Level 3 (wip/moe-verify-fusions): byte-compare every resident slot with its expert's slice in the
    // host master.  A mismatch means the slot holds the wrong bytes -- the direct cause of a residency-
    // dependent wrong-expert read.  Geometry mirrors the host fill.
    static const int only_layer = [] {
        const char * vlay_e = getenv("MOE_EXPERT_CACHE_VALIDATE_LAYER");
        return vlay_e != nullptr ? atoi(vlay_e) : -1;
    }();
    for (table_t & t : g_tables) {
        if (t.arena == nullptr || t.slots <= 0 || t.device < 0 || t.host == nullptr) {
            continue;
        }
        if (only_layer >= 0 && t.layer != only_layer) {
            continue;
        }
        const bool    do2d = (t.split_axis == 0 && t.host_pitch > 0);
        const int64_t rows = do2d ? (int64_t) (t.host_bytes / t.host_pitch) : 0;
        const int64_t row  = rows > 0 ? (int64_t) (t.expert_bytes / rows) : 0;
        device_guard dg(t.device);
        const size_t cmp_bytes = only_layer >= 0 ? (size_t) t.expert_bytes
                                                 : (t.expert_bytes < 4096 ? (size_t) t.expert_bytes : 4096);
        std::vector<uint8_t> buf(cmp_bytes);
        int n_bad = 0;
        for (int s = 0; s < t.slots && n_bad < 8; s++) {
            const int e = t.slot_expert[(size_t) s];
            if (e < 0 || e >= t.n_experts) {
                continue;
            }
            const char * slotp = (const char *) t.arena + (size_t) s * t.expert_bytes;
            if (cudaMemcpy(buf.data(), slotp, cmp_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
                (void) cudaGetLastError();
                break;
            }
            const char * hsrc = (const char *) t.host + (size_t) e * t.host_bytes + t.src_off;
            bool bad = false;
            if (do2d && row > 0) {
                const int64_t nrows_cmp = (int64_t) (cmp_bytes / (size_t) row);
                for (int64_t r = 0; r < nrows_cmp && r < rows && !bad; r++) {
                    if (memcmp(buf.data() + r * row, hsrc + r * t.host_pitch, (size_t) row) != 0) {
                        bad = true;
                    }
                }
            } else {
                bad = memcmp(buf.data(), hsrc, cmp_bytes) != 0;
            }
            if (bad) {
                n_bad++;
                GGML_LOG_ERROR("[validate %s] SLOT-CONTENT-MISMATCH layer=%d role=%s slot=%d expert=%d expert_bytes=%lld host_bytes=%lld split_axis=%d host_pitch=%lld\n",
                               where, t.layer, t.role.c_str(), s, e, (long long) t.expert_bytes,
                               (long long) t.host_bytes, t.split_axis, (long long) t.host_pitch);
            }
        }
    }

}

// True only when the cache has at least one registered routed expert table.  An empty cache (the model
// is fully device-resident -- `-ncmoe 0`, or a `-ncmoe` that did not offload, e.g. on unified memory --
// can never take an input over, so the cache-band fusion guard must not stand anything down: a cache-on
// run has to stay byte-identical to the cache-less one.  Returns false before sizing.
bool moe_cache_table_is_pool(const struct ggml_tensor * weight, int device) {
    // wip/host-pool-speed (#131): a routed expert table whose master is the read-only model MAPPING
    // (`--host-experts pool`) is a "pool" table; one with a pinned, device-readable master is not.  Called
    // from the scheduler's split, once per routed expert op, so keep it cheap: a linear scan over the
    // registered tables (a few hundred) guarded by the same lock the registration uses.
    if (weight == nullptr || device < 0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    for (const table_t & t : g_tables) {
        if (t.device == device && t.host == (const void *) weight->data) {
            return !t.host_pinned && t.src_known;
        }
    }
    return false;
}

// Can the cache serve THIS routed host-expert table from the arena right now (sized + the table has slots)?
// The scheduler picks the transport with it: a table the cache cannot serve is left to the plain
// whole-table upload, because a ring round-trip that gains nothing still pays a ring slot + the
// slot -> input D2D.  Measured (35B-A3B Q8_0, `-ncmoe 48`, 1 GPU, `-p 4096`): a small mid-prefill arena
// staged 2658 t/s against 3446 for the plain upload, while a properly sized arena staged 1195 against
// 580 at `-ub 512`.
bool moe_cache_stage_ready(const struct ggml_tensor * weight, int device) {
    // TRUE means "staging this table is fine" -- including the case where the cache has no opinion at all
    // (disabled, or this device expects no host expert tables), so the ring behaves exactly as it did
    // before this query existed.  FALSE is the only case that changes the transport: the cache IS managing
    // host experts here and cannot fill this table from the arena, so a ring slot would buy nothing.
    if (weight == nullptr || device < 0 || !g_enabled) {
        return true;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    if (device >= GGML_CUDA_MAX_DEVICES || !g_preflight_device_seen[device]) {
        return true;
    }
    if (!g_sized) {
        return false;
    }
    for (const table_t & t : g_tables) {
        if (t.device == device && t.host == (const void *) weight->data) {
            return t.arena != nullptr && t.slots > 0;
        }
    }
    return false;
}

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
void moe_cache_set_host_pool(bool on) {
    // Enable the bounded host pool for the table's host experts that stay in the read-only model mapping
    // (the not-fitting band of `-lme auto`, i.e. the historical `--host-experts pool`).  The loader calls
    // this before any table is registered.  The size comes from MOE_HOST_POOL_MIB (default:
    // MOE_HOST_POOL_FRAC % of the MoE host expert bytes).  Idempotent, and inert until a table with a known
    // source is registered.
    //
    // wip/host-expert-tier: the loader decides the mode at MODEL LOAD, which can be after `parse_env()` has
    // already resolved the pool from the environment, so re-resolve it here.  Without this the pool stayed
    // off (parse_env had run with no request) and every hole fell back to the per-table bounce ring --
    // measured 37.4 t/s decode at 2 devices against the pool's 51.5.
    ensure_env();   // outside the lock, exactly as `moe_cache_set_host_source` does
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_pool_requested == on && g_pool_enabled == (moe_host_pool_on())) {
        return;
    }
    g_pool_requested = on;
    g_pool_enabled   = moe_host_pool_on();
    if (!g_pool_enabled) {
        return;
    }
    if (g_pool_budget_override >= 0) {
        // An explicit budget from the loader (`-lme N` below the host expert set): the pool IS the tier, so the
        // budget is exact rather than a percentage of the host bytes.
        g_pool_enabled     = g_pool_budget_override > 0;
        g_pool_auto        = false;
        g_pool_total_bytes = (size_t) g_pool_budget_override;
    } else {
        const int64_t mib  = moe_host_pool_mib();
        g_pool_auto        = mib < 0;
        g_pool_total_bytes = (!g_pool_auto && mib > 0) ? (size_t) mib * 1024 * 1024 : 0;
    }
    GGML_LOG_INFO("%s: host pool %s: %s\n", __func__, on ? "enabled" : "disabled",
                  g_pool_budget_override >= 0 ? "the loader's explicit budget (-lme N)"
                  : g_pool_total_bytes > 0 ? "MOE_HOST_POOL_MIB (explicit)"
                                           : "MOE_HOST_POOL_FRAC of the host bytes");
}

// wip/host-expert-tier: the persisted heat prior, opened by the model loader before it decides any
// host-expert placement.  `moe_cache_set_host_source` opens it too, but only when the first tensor's DATA
// is copied -- long after every buffer-type decision has been made, which would leave the pinned tier
// with no ranking at load time.
void moe_cache_set_host_pool_budget(size_t bytes) {
    ensure_env();
    std::lock_guard<std::mutex> lock(g_mutex);
    g_pool_budget_override = (int64_t) bytes;
}

void moe_cache_set_host_fill_mode(int direct) {
    ensure_env();
    std::lock_guard<std::mutex> lock(g_mutex);
    g_pool_load_mode = direct != 0 ? 1 : 0;
}

void moe_cache_heat_open(const char * model_path) {
    ensure_env();
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_heat_on || !g_enabled || model_path == nullptr || model_path[0] == '\0') {
        return;
    }
    heat_map_open_locked(model_path);
}

// The hotness prior of one host expert tensor (the sum of its persisted per-expert counts), or -1 when the
// sidecar carries no record of it.  `layer`/`name`/`n_experts`/`host_bytes` are the same key the residency
// prior uses at table registration, so a pinned set and a pool's contents are ranked by the same evidence.
int64_t moe_cache_heat_table_score(int layer, const char * name, int n_experts, size_t host_bytes) {
    if (name == nullptr || name[0] == '\0' || n_experts <= 0) {
        return -1;
    }
    ensure_env();
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_heat_on || g_heat_loaded.empty()) {
        return -1;
    }
    const auto it = g_heat_loaded.find(std::make_tuple(layer, std::string(name), n_experts, host_bytes));
    if (it == g_heat_loaded.end() || it->second.all.empty()) {
        return -1;
    }
    int64_t s = 0;
    for (const uint32_t v : it->second.all) {
        s += (int64_t) v;
    }
    return s;
}

void moe_cache_set_host_source(const void * tensor_data, const char * path, size_t offs,
                               int n_experts, size_t host_bytes, size_t total_bytes) {
    if (tensor_data == nullptr || path == nullptr || path[0] == '\0') {
        return;
    }
    ensure_env();
    std::lock_guard<std::mutex> lock(g_mutex);
    // wip/moe-heatmap-persistence: the first host-expert tensor gives the GGUF path, from which the
    // sidecar is derived and the prior loaded.  Independent of the host pool: recorded whenever the
    // heat map is on, so the loader needs no extra plumbing.
    if (g_heat_on && g_enabled && !g_heat_path_set) {
        heat_map_open_locked(path);
    }
    if (!moe_host_pool_on()) {
        return;   // the pool part is inert unless the bounded host pool is requested
    }
    host_src_t & s = g_host_src[tensor_data];
    s.path        = path;
    s.offs        = offs;
    s.n_experts   = n_experts;
    s.host_bytes  = host_bytes;
    s.total_bytes = total_bytes;
}

int moe_cache_table(const void * src0, int layer, const char * role, int n_experts, size_t expert_bytes,
                    size_t host_bytes, size_t src_off, size_t host_pitch, int split_axis,
                    const void * host, int device,
                    const struct ggml_backend_moe_slice * slices, int n_slices) {
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
    // A fused table's slot is several host ranges, so a single cold (UVA) read cannot express it: every
    // miss is filled from the master instead.  That is the same cost as a resident-set miss on any other
    // table, and it keeps the device-side cold encoding (`id - n_res_slots` -> one host stride) honest.
    const bool fused = slices != nullptr && n_slices > 0;
    if (fused) {
        cold_safe = false;
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
    t.n_slices     = fused ? std::min(n_slices, (int) GGML_BACKEND_MOE_MAX_SLICES) : 0;
    for (int k = 0; k < t.n_slices; k++) {
        t.slice[k].src_off = slices[k].src_off;
        t.slice[k].len     = slices[k].len;
    }
    // TEMP diagnostic (wip/moe-verify-fusions): print the per-table split geometry, to compare the
    // residency-sensitive qwen4exp tables against the residency-insensitive qwen35moe ones.
    if (ggml_cuda_cache_geom_debug()) {
        GGML_LOG_WARN("CACHE_GEOM layer=%d dev=%d role=%s n_experts=%d expert_bytes=%zu host_bytes=%zu src_off=%zu host_pitch=%zu split_axis=%d cold_safe=%d n_slices=%d\n",
                      layer, device, t.role.c_str(), n_experts, expert_bytes, host_bytes, src_off, host_pitch, split_axis, (int) cold_safe, t.n_slices);
    }
    t.host         = host;
    t.device       = device;
    attach_host_src_locked(t, host);
    heat_apply_loaded_locked(t);   // wip/moe-heatmap-persistence: adopt the persisted prior, if any

    // A MIRRORED expert table means the tensor split could not be expressed: the meta split is equal-division
    // only, rounded DOWN to the quant block, so an axis with fewer block-aligned chunks than devices leaves
    // one device with a zero extent -- and that device is handed the WHOLE tensor (MIRRORED) instead of being
    // skipped.  It is silent and expensive: measured on 35B-A3B at 3 devices, ~20 t/s against 72 at 2, for
    // EVERY transport (2x the host bytes on those layers plus the whole expert's compute on one device), which
    // is what made a transport investigation chase its own tail.  Detect it when the tables are alive.
    {
        static bool warned = false;
        const bool whole = t.host_bytes > 0 && t.expert_bytes == t.host_bytes;
        if (!warned) {
            for (const table_t & o : g_tables) {
                if (o.layer != t.layer || o.role != t.role || o.host_bytes == 0) {
                    continue;
                }
                if (whole != (o.expert_bytes == o.host_bytes)) {
                    warned = true;
                    // fprintf(stderr, ...) and not GGML_LOG_WARN: ggml's WARN is FILTERED at llama-cli's
                    // default log level, so a GGML_LOG_WARN here is invisible to the user -- which is how a
                    // 3.5x throughput cliff stayed undiagnosed.  Same rule the campaign's P1-PUSH messages use.
                    fprintf(stderr, "DEGENERATE TENSOR SPLIT: layer=%d role=%s has expert_bytes=%zu of "
                                  "host_bytes=%zu on device %d while the same tensor on device %d has "
                                  "expert_bytes=%zu -- one device holds the WHOLE expert (MIRRORED) and another "
                                  "a slice.  The split is equal-division only and rounded to the quant block, so "
                                  "this axis has fewer block-aligned chunks than devices.  Expect ~2x the host "
                                  "bytes on those layers and NO scaling, for every transport: use fewer devices "
                                  "(-sm layer) or a finer-block quant.\n", t.layer, t.role.c_str(),
                                  t.expert_bytes, t.host_bytes, t.device, o.device, o.expert_bytes);
                    break;
                }
            }
        }
    }
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
    if (t.host != nullptr && expert >= 0 && expert < t.n_experts && t.host_pitch == 0 && t.n_slices == 0 &&
        !table_master_pageable(t)) {
        return { (void *) ((const char *) t.host + (size_t) expert * t.host_bytes + t.src_off), false };
    }
    return { nullptr, false };
}

// ---------------------------------------------------------------------------------------------
// TODO #61 P1: the offline routing trace writer.  `emit` is the single sink both bands funnel into; it
// takes `g_mutex` itself, so a caller must NOT already hold it.  See the globals block above for the
// format and for why the two bands have different sources.
// ---------------------------------------------------------------------------------------------

// Copy a device-side routing tensor to the host.  Used by the PREFILL band only (never captured).
static void moe_trace_pull_device(const void * dev_ids, int64_t nb0, int64_t nb1, int n_tok, int n_used,
                                  void * stream, int device, std::vector<int32_t> & out) {
    out.assign((size_t) n_used * (size_t) n_tok, -1);
    device_guard dg(device);
    const cudaStream_t st = (cudaStream_t) stream;
    const bool         contiguous = nb0 == (int64_t) sizeof(int32_t) &&
                                    nb1 == (int64_t) n_used * (int64_t) sizeof(int32_t);
    const int rows = contiguous ? 1 : n_tok;
    for (int r = 0; r < rows; r++) {
        const char * src = (const char *) dev_ids + (size_t) r * (size_t) nb1;
        int32_t *    dst = out.data() + (size_t) r * (size_t) n_used;
        const size_t nby = (size_t) n_used * sizeof(int32_t);
        if (st != nullptr) {
            (void) cudaMemcpyAsync(dst, src, nby, cudaMemcpyDeviceToHost, st);
            (void) cudaStreamSynchronize(st);
        } else {
            (void) cudaMemcpy(dst, src, nby, cudaMemcpyDeviceToHost);
        }
    }
    (void) cudaGetLastError();   // a failed readback leaves -1s: the replayer ignores them
}

static void moe_trace_emit(char band, int layer, const char * role, int n_tok, int n_used,
                           const int32_t * ids, int n_experts, size_t expert_bytes) {
    if (!g_trace_on || g_trace_stopped) {
        return;
    }
    if (role == nullptr) {
        role = "?";
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_trace_stopped) {
        return;
    }
    // A tensor split runs the same op (same ids) once per device: the neighbours are identical, so drop
    // the repeat.  This is also what keeps the trace one record per (layer, role, pass).
    const size_t n = (size_t) n_used * (size_t) n_tok;
    if (layer == g_trace_prev_layer && band == g_trace_prev_band && g_trace_prev_role == role &&
        g_trace_prev_ids.size() == n && (n == 0 || memcmp(g_trace_prev_ids.data(), ids, n * sizeof(int32_t)) == 0)) {
        return;
    }
    if (g_trace_file == nullptr) {
        g_trace_file = fopen(g_trace_path.c_str(), "w");
        if (g_trace_file == nullptr) {
            GGML_LOG_ERROR("%s: cannot open MOE_EXPERT_TRACE=%s: %s; the routing trace is disabled\n",
                           __func__, g_trace_path.c_str(), strerror(errno));
            g_trace_stopped = true;
            return;
        }
        fputs("# moe-trace v1\n", g_trace_file);
    }
    if (g_trace_bytes >= g_trace_max_bytes) {
        GGML_LOG_WARN("%s: MOE_EXPERT_TRACE cap reached (%.0f MiB, %lld records); recording stopped.  Raise "
                      "MOE_EXPERT_TRACE_MIB for a longer capture.\n",
                      __func__, (double) g_trace_max_bytes / (1024 * 1024), (long long) g_trace_records);
        fflush(g_trace_file);
        g_trace_stopped = true;
        return;
    }
    // Table descriptor on first sight, so the replayer knows the expert geometry without the model.
    if (g_trace_described.emplace(layer, std::string(role)).second) {
        fprintf(g_trace_file, "T %d %s %d %zu\n", layer, role, n_experts, expert_bytes);
        g_trace_bytes += 32 + (int64_t) strlen(role);
    }
    std::string line;
    line.reserve(48 + n * 4);
    char buf[256];
    snprintf(buf, sizeof(buf), "%c %lld %d %s %d %d", band, (long long) g_trace_records, layer, role,
             n_tok, n_used);
    line += buf;
    for (size_t i = 0; i < n; i++) {
        snprintf(buf, sizeof(buf), " %d", ids[i]);
        line += buf;
    }
    line += '\n';
    fwrite(line.data(), 1, line.size(), g_trace_file);
    g_trace_bytes += (int64_t) line.size();
    g_trace_records++;
    g_trace_prev_layer = layer;
    g_trace_prev_band  = band;
    g_trace_prev_role  = role;
    g_trace_prev_ids.assign(ids, ids + n);
}

// wip/moe-heatmap-persistence (R2): device-side routing tally.  Reads the raw routing ids on the device
// (the OUTER `mul_mat_id` entry, before the cache's slot-remap redirect), so it counts the true expert
// ids for every path - including the staged path (where the host hook is skipped) and the identity path
// (where the host hook is short-circuited).  The PREFILL band is tallied here exactly; the DECODE band is
// tallied on the host by `moe_cache_update_host`, except for a fully-resident (identity) table, whose
// decode routing is never read back - that one is tallied here SAMPLED (`w` is the sample period, so the
// counts stay on the same scale as the exact tallies).
static __global__ void moe_cache_heat_kernel(const int32_t * __restrict__ ids, int64_t nb0, int64_t nb1,
                                             int n_used, int n_tok, int n_experts, uint32_t w,
                                             uint32_t * __restrict__ heat,
                                             uint32_t * __restrict__ heat_dec, int is_decode) {
    const int64_t n = (int64_t) n_used * (int64_t) n_tok;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (int64_t) gridDim.x * blockDim.x) {
        const int64_t tok = n_used > 0 ? i / n_used : 0;
        const int64_t j   = n_used > 0 ? i % n_used : 0;
        const int32_t e = *(const int32_t *) ((const char *) ids + (size_t) tok * (size_t) nb1 +
                                              (size_t) j * (size_t) nb0);
        if (e >= 0 && e < n_experts) {
            atomicAdd(&heat[e], w);
            // TODO #88: the DECODE band's own counter (the arena seed ranks by it).
            if (is_decode != 0 && heat_dec != nullptr) {
                atomicAdd(&heat_dec[e], w);
            }
        }
    }
}

void moe_cache_heat_observe(const ggml_tensor * op, const ggml_tensor * src0, const ggml_tensor * ids,
                            void * stream, int device) {
    if (!g_enabled || op == nullptr || src0 == nullptr || ids == nullptr) {
        return;
    }
    const int layer = name_layer(op);
    if (layer < 0 || src0->ne[2] < 1 || ids->ne[0] <= 0) {
        return;
    }
    const int    n_used = (int) ids->ne[0];
    const int    n_tok  = (int) ids->ne[1];
    const bool   prefill_band = n_tok > MOE_EXPERT_CACHE_MAX_TOK;

    // The one place every routed MoE op passes through, whatever path it then takes (identity, eager
    // remap, cold, declined), so it is the only honest token census.  Counted before the heat gate: the
    // traffic report needs it with the heat map off too.
    if (!prefill_band) {
        g_band_ops.fetch_add(1, std::memory_order_relaxed);
    }

    // TODO #61 P1: the prefill routing never reaches the host on the staged path, so the trace's
    // PREFILL records have to come from here.  A prefill ubatch is never CUDA-graph captured, so the
    // readback is safe; the whole trace is opt-in and the decode band is traced on the host instead.
    if (prefill_band && g_trace_on && !g_trace_stopped) {
        std::vector<int32_t> ids_host;
        moe_trace_pull_device(ids->data, (int64_t) ids->nb[0], (int64_t) ids->nb[1], n_tok, n_used,
                              stream, device, ids_host);
        const size_t expert_bytes = src0->nb[2] > 0 ? (size_t) src0->nb[2]
                                                    : (size_t) (ggml_nbytes(src0) / (size_t) src0->ne[2]);
        moe_trace_emit('P', layer, sem_role(src0).c_str(), n_tok, n_used, ids_host.data(),
                       (int) src0->ne[2], expert_bytes);
    }
    if (!g_heat_on) {
        return;
    }
    // TODO #64.2: the decode band only needs this tally for a table no host tally can see - a fully
    // resident (identity) one - and even then SAMPLED, so the latency-critical path pays one atomic
    // increment per op and a lock only every `g_heat_decode_sample` ops.
    uint32_t weight = 1;
    if (!prefill_band) {
        if (g_heat_decode_sample <= 0) {
            return;
        }
        const int64_t tick = g_heat_op_tick.fetch_add(1, std::memory_order_relaxed) + 1;
        if (tick % (int64_t) g_heat_decode_sample != 0) {
            return;
        }
        weight = (uint32_t) g_heat_decode_sample;
    } else {
        g_heat_prefill_ops.fetch_add(1, std::memory_order_relaxed);
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    const auto it = g_sem_to_id.find(std::make_tuple(layer, sem_role(src0), device));
    if (it == g_sem_to_id.end()) {
        return;   // not registered yet (the priming pass): nothing to attach the tally to
    }
    table_t & t = g_tables[it->second];
    if (t.n_experts <= 0) {
        return;
    }
    if (!prefill_band && !t.identity) {
        return;   // the eager host path tallies this table exactly; never double count it
    }
    // TODO #88: PREFILL touches the combined prior only; a DECODE touch goes to the decode counter as well
    // (it is the one the ARENA seed ranks by).
    g_heat_dirty = true;
    // The mirror is pre-allocated at sizing (alloc_table_locked).  Never allocate here on the decode
    // band: that path is a captured graph, where a `cudaMalloc` aborts the capture.
    if (t.heat_dev == nullptr && (!prefill_band || !heat_dev_ensure_locked(t, device))) {
        return;
    }
    const int64_t n     = (int64_t) n_used * (int64_t) n_tok;
    const int     threads = 128;
    int64_t       nb    = (n + threads - 1) / threads;
    if (nb > 4096) {
        nb = 4096;
    }
    if (nb < 1) {
        nb = 1;
    }
    moe_cache_heat_kernel<<<(unsigned) nb, threads, 0, (cudaStream_t) stream>>>(
        (const int32_t *) ids->data, (int64_t) ids->nb[0], (int64_t) ids->nb[1],
        n_used, n_tok, t.n_experts, weight, t.heat_dev,
        prefill_band ? nullptr : t.heat_dec_dev, prefill_band ? 0 : 1);
}

// Fold every table's device heat mirror into the host `heat` vector and zero the mirror.  Run at a
// non-hot boundary (the report, or a persistence flush).  Caller holds `g_mutex`.
static void heat_pull_locked() {
    for (table_t & t : g_tables) {
        if (t.heat_dev == nullptr || t.heat_dev_cap <= 0 || t.device < 0) {
            continue;
        }
        device_guard dg(t.device);
        std::vector<uint32_t> tmp((size_t) t.heat_dev_cap, 0);
        if (cudaMemcpy(tmp.data(), t.heat_dev, (size_t) t.heat_dev_cap * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            (void) cudaGetLastError();
            continue;
        }
        (void) cudaMemset(t.heat_dev, 0, (size_t) t.heat_dev_cap * sizeof(uint32_t));
        if ((int) t.heat.size() != t.n_experts) {
            t.heat.assign((size_t) t.n_experts, 0);
        }
        const int cap = t.n_experts < (int) t.heat_dev_cap ? t.n_experts : (int) t.heat_dev_cap;
        for (int e = 0; e < cap; e++) {
            const uint64_t s = (uint64_t) t.heat[(size_t) e] + (uint64_t) tmp[(size_t) e];
            t.heat[(size_t) e] = s > 0xffffffffull ? 0xffffffffu : (uint32_t) s;
        }
        // TODO #88: the identity DECODE tally has its own mirror, folded into the decode counter.
        if (t.heat_dec_dev != nullptr) {
            std::vector<uint32_t> tdec((size_t) t.heat_dev_cap, 0);
            if (cudaMemcpy(tdec.data(), t.heat_dec_dev, (size_t) t.heat_dev_cap * sizeof(uint32_t),
                           cudaMemcpyDeviceToHost) == cudaSuccess) {
                (void) cudaMemset(t.heat_dec_dev, 0, (size_t) t.heat_dev_cap * sizeof(uint32_t));
                if ((int) t.heat_dec.size() != t.n_experts) {
                    t.heat_dec.assign((size_t) t.n_experts, 0);
                }
                for (int e = 0; e < cap; e++) {
                    const uint64_t s = (uint64_t) t.heat_dec[(size_t) e] + (uint64_t) tdec[(size_t) e];
                    t.heat_dec[(size_t) e] = s > 0xffffffffull ? 0xffffffffu : (uint32_t) s;
                }
            } else {
                (void) cudaGetLastError();
            }
        }
    }
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

// fit-slab-accounting revival: a split host slice is only addressable when its byte range stays inside
// one host expert.  A meta view whose `nb` does not describe its slice (measured on `-sm tensor`: a
// layer registering `expert_bytes == host_bytes` with a nonzero `slice_off`) makes every cache fill read
// `slice_off` bytes past the end of the master -- a host memmove SIGSEGV in the identity fill, or a
// device page fault in the gather.  Returns false so the caller declines the table; the scheduler's
// full-table copy is always correct.  `host_pitch` is the strided axis-0 case (a 2-D fill whose span
// differs from `expert_bytes`).
static bool moe_cache_slice_addr_ok(size_t host_bytes, size_t expert_bytes, size_t src_off, size_t host_pitch) {
    if (host_bytes == 0 || expert_bytes == 0 || src_off > host_bytes) {
        return false;
    }
    size_t span = expert_bytes;
    if (host_pitch > 0) {
        if (host_bytes % host_pitch != 0) {
            return false;
        }
        const size_t rows = host_bytes / host_pitch;
        if (rows == 0 || expert_bytes % rows != 0) {
            return false;
        }
        span = (rows - 1) * host_pitch + expert_bytes / rows;
    }
    return span <= host_bytes - src_off;
}

// wip/fix-gemma-4 (TODO #77/C): every range of a fused table's slot must lie inside one host expert blob,
// and the ranges must tile the device slot exactly (they are concatenated into it).
static bool moe_cache_slices_addr_ok(size_t host_bytes, size_t expert_bytes,
                                    const struct ggml_backend_moe_slice * slices, int n_slices) {
    if (slices == nullptr || n_slices <= 0 || n_slices > GGML_BACKEND_MOE_MAX_SLICES) {
        return false;
    }
    if (host_bytes == 0 || expert_bytes == 0) {
        return false;
    }
    size_t total = 0;
    for (int k = 0; k < n_slices; k++) {
        const size_t len = slices[k].len;
        if (len == 0 || slices[k].src_off > host_bytes || len > host_bytes - slices[k].src_off) {
            return false;
        }
        total += len;
    }
    return total == expert_bytes;
}

// wip/host-expert-master-drop (TODO #55): serve a copy FROM a pageable host expert master (the model
// mapping) WITHOUT a pageable H2D.
//
// The scheduler's used-expert copy (`copy_experts`) and its whole-table fallback read the host master at
// a byte offset and hand the pointer to `ggml_backend_tensor_set_async`.  A pageable `cudaMemcpyAsync` is
// not safe for that (ROCm stages it through a shared internal bounce, and back-to-back pageable copies
// tear -- measured as nondeterministic greedy text while every copy verified byte-exact), so the bytes are
// copied into the per-table pinned bounce first and the transfer is a pinned H2D.  Returns false when
// `src_host` does not belong to a registered pool-backed master (the caller keeps its normal path).
//
// Only the 1:1 geometry is served (an unsplit table: the same byte offset in the master and in `dst`).  A
// split/meta table's copies are issued by the meta backend per device, where `set_tensor_2d_async` already
// snapshots a pageable source through its pinned ring.
bool moe_cache_upload_pageable(void * dst, const void * src_host, size_t size, void * stream, int device) {
    if (dst == nullptr || src_host == nullptr || size == 0 || !g_enabled) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    const char * s = (const char *) src_host;
    for (table_t & t : g_tables) {
        if (t.device != device || t.host == nullptr || t.host_pinned || t.n_experts <= 0 || t.host_bytes == 0) {
            continue;
        }
        const char * base = (const char *) t.host;
        const size_t span = (size_t) t.n_experts * t.host_bytes;
        if (s < base || s >= base + span) {
            continue;
        }
        const size_t off = (size_t) (s - base);
        // A byte range of the master, chunked through the per-table pinned bounce.  Deliberately NOT
        // indexed by expert: this must serve EVERY geometry a table can have (a fused `-sm tensor` slot, a
        // strided axis-0 slice, and the uneven-split "degenerate" slice the meta splitter can hand a device,
        // whose `src_off + expert_bytes` exceeds one host expert) -- the caller's range is just bytes.
        // Reading past the tensor (the caller's 512-byte MMQ tail pad on the LAST expert) is clamped: the
        // pinned master has no bytes there either, so the stale destination is the same either way, and a
        // pageable copy of that range would read past the mapping.
        const size_t n_total = std::min(size, span - off);
        g_uploadp_calls++;
        g_uploadp_bytes += (int64_t) n_total;
        size_t done = 0;
        while (done < n_total) {
            const size_t n = std::min(n_total - done, t.host_bytes);
            // Acquire a pinned bounce entry, fill it from the mapping, and issue a pinned H2D.  ONE place owns
            // the ring (`bounce_acquire`: reuse, BOUNDED growth, wait at the cap), so this path and the pool's
            // fill path cannot disagree about the depth or the pin budget.
            const int idx = bounce_acquire(t);
            if (idx < 0) {
                return false;
            }
            char * buf = t.bounce[(size_t) idx].host;
            memcpy(buf, s + done, n);
            const cudaError_t err = cudaMemcpyAsync((char *) dst + done, buf, n, cudaMemcpyHostToDevice,
                                                   (cudaStream_t) stream);
            if (err != cudaSuccess) {
                (void) cudaGetLastError();
                return false;
            }
            (void) cudaEventRecord(t.bounce[(size_t) idx].ev, (cudaStream_t) stream);
            t.bounce[(size_t) idx].busy = true;
            done += n;
        }
        return true;
    }
    return false;
}

// wip/host-expert-master-drop (TODO #55): is `p` inside a POOL-BACKED (pageable, file-mapping-backed)
// host expert master?  Used by the staging hooks to refuse a "pinned source" fast path for a pointer the
// runtime was never told to pin -- a 2-D pageable H2D is the pathological ROCm case.
bool moe_cache_source_pageable(const void * p) {
    if (p == nullptr || !g_enabled) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    const char * c = (const char *) p;
    for (const table_t & t : g_tables) {
        if (t.host == nullptr || t.host_pinned || t.host_bytes == 0 || t.n_experts <= 0) {
            continue;
        }
        const char * b = (const char *) t.host;
        if (c >= b && c < b + (size_t) t.n_experts * t.host_bytes) {
            return true;
        }
    }
    return false;
}

// wip/host-expert-split W4 (prefill read granularity): one pinned span buffer for the coalesced staging read.
// `host` is `align`-ALIGNED inside `raw` (an O_DIRECT read needs an aligned buffer AND an aligned offset, and
// the tensor's own file offset is only 32-byte aligned, so the read starts one partial block early and the
// device copy reads from `host + skip`).  Four entries per device: the ring is what lets the read of chunk
// N+1 overlap the copy of chunk N -- a TWO-deep ring spent 5.6 s of a 16 s prefill waiting on it (measured at
// -lme 36, where the runs are short enough that every chunk came back to a buffer whose copies were still in
// flight) -- and since both prefill paths hold `g_mutex` for the whole call, one process-wide ring per device
// is enough.  Deliberately LEAKED (a plain file-scope array): the buffers are pinned for the life of the
// process, exactly like the pool and the bounce rings, and a teardown-time free is the one place pinned
// allocations have already bitten this file.
struct span_buf_t {
    void *      raw   = nullptr;
    char *      host  = nullptr;   // `align`-aligned base inside `raw`
    size_t      cap   = 0;         // usable span bytes from `host`
    size_t      align = 0;         // the alignment `host` was cut to
    cudaEvent_t ev    = nullptr;
    bool        busy  = false;     // a device copy still reads `host`
};
// The ring is GROWN ON DEMAND (up to `SPAN_RING_MAX` entries per device) instead of being fixed, for the
// same measured reason the bounce ring is: the H2D copies are queued on the compute stream, so they only run
// as the layer's kernel queue reaches them, and a fixed 2-4 deep ring spent 5 s of a 16 s prefill WAITING on
// its own copies.  An entry that has drained is reused; only when every entry is still in flight does the
// caller wait, which then means the device is genuinely behind the reads.
#define SPAN_RING_MAX 32
static span_buf_t g_span_buf[GGML_CUDA_MAX_DEVICES][SPAN_RING_MAX];
static int        g_span_used[GGML_CUDA_MAX_DEVICES] = {};   // entries ever allocated (grows to the backlog)
static int        g_span_next[GGML_CUDA_MAX_DEVICES] = {};   // round-robin cursor
// A refused pinned allocation latches the coalesced path OFF for the run (the per-expert path is correct on
// its own, so there is nothing to report but the fallback itself).
static bool       g_span_off = false;

// (Re)allocate `b` so it holds `need` span bytes cut to `align`.  Caller has established that no copy is in
// flight out of `b`, and is inside a `device_guard`.
static span_buf_t * span_buf_ensure(span_buf_t & b, size_t need, size_t align) {
    if (align == 0 || (align & (align - 1)) != 0) {
        align = 4096;
    }
    if (pinned_tier_disabled()) {
        g_span_off = true;   // MOE_HOST_POOL_PIN=0: do not ask the driver for a pinned ring either
        return nullptr;
    }
    // One partial block ahead of the span (`skip` at the call site) plus the chunk itself; `skip` can be a
    // whole block, hence two.  `cap` is what the caller must not overrun.
    const size_t want = need + 2 * align;
    if (b.host != nullptr && b.align == align && b.cap >= need + align) {
        return &b;
    }
    if (b.raw != nullptr) {
        (void) cudaFreeHost(b.raw);
        if (b.ev != nullptr) {
            (void) cudaEventDestroy(b.ev);
            b.ev = nullptr;
        }
        b.raw  = nullptr;
        b.host = nullptr;
        b.cap  = 0;
    }
    void * mem = nullptr;
    if (cudaHostAlloc(&mem, want, cudaHostAllocPortable | cudaHostAllocMapped) != cudaSuccess || mem == nullptr) {
        (void) cudaGetLastError();
        g_span_off = true;
        GGML_LOG_WARN("%s: the coalesced prefill read is OFF for this run (cudaHostAlloc of %zu B was "
                      "refused; %.1f MiB of the pinned tier was already allocated); the per-expert read path "
                      "serves it as before\n", __func__, want,
                      (double) g_pinned_tier_bytes / (1024.0 * 1024.0));
        return nullptr;
    }
    if (cudaEventCreateWithFlags(&b.ev, cudaEventDisableTiming) != cudaSuccess) {
        (void) cudaGetLastError();
        (void) cudaFreeHost(mem);
        b.ev = nullptr;
        g_span_off = true;
        return nullptr;
    }
    b.raw   = mem;
    b.host  = (char *) (((uintptr_t) mem + align - 1) & ~((uintptr_t) align - 1));
    b.cap   = want - (size_t) (b.host - (char *) mem);
    b.align = align;
    g_pinned_tier_bytes += (int64_t) want;   // census only: how far the pinned tier grew (see `g_pinned_off`)
    return &b;
}

// Borrow a span buffer of >= `need` bytes.  Prefers an entry whose copies have drained; grows the ring while
// the backlog is deeper than the ring; only the full-ring case waits.  Returns null when the path cannot run
// (pinned allocation refused).  Caller holds `g_mutex` and is inside a `device_guard`.
static span_buf_t * span_buf_acquire(int device, size_t need, size_t align) {
    if (g_span_off || device < 0 || device >= GGML_CUDA_MAX_DEVICES || need == 0) {
        return nullptr;
    }
    const int used = g_span_used[device];
    for (int i = 0; i < used; i++) {
        const int idx = (g_span_next[device] + i) % used;
        if (g_span_buf[device][idx].busy) {
            continue;
        }
        g_span_next[device] = (idx + 1) % used;
        return span_buf_ensure(g_span_buf[device][idx], need, align);
    }
    if (used < SPAN_RING_MAX) {
        g_span_used[device] = used + 1;
        g_span_next[device] = 0;
        return span_buf_ensure(g_span_buf[device][used], need, align);
    }
    // Full ring and every entry still being read: wait on the next one in rotation.  This is the only stall,
    // and it means the device is a whole ring behind the reads.
    span_buf_t & b = g_span_buf[device][g_span_next[device]];
    const int64_t tw = ggml_time_us();
    if (b.ev != nullptr) {
        (void) cudaEventSynchronize(b.ev);
    }
    (void) cudaGetLastError();
    g_stage_span_wait_us += ggml_time_us() - tw;
    b.busy = false;
    g_span_next[device] = (g_span_next[device] + 1) % used;
    return span_buf_ensure(b, need, align);
}

// Serve every maximal RUN of consecutive experts that are in NEITHER tier from ONE storage read per chunk
// (`MOE_HOST_POOL_STAGE_SPAN_EXPERTS` experts), then one H2D per expert out of the pinned chunk.
//
// The read is a FILE read in both fill modes -- positional `pread`/`ReadFile` -- O_DIRECT (aligned) when the
// disk tier is on, buffered through the OS page cache otherwise.  It used to be file-only in the DIO arm; the
// buffered arm read the model MAPPING per expert instead, which is a demand fault per page on Windows and so
// cannot be coalesced away (issue #281).  A buffered `pread` reads the same page-cache pages the mapping
// does, so the fill's SOURCE is unchanged -- only its granularity (and, on Windows, its fault behaviour).
//
// `map[e]` is the two-tier residency encoding every prefill path already builds (`>= 0` arena slot, `<= -2`
// pool slot `-v-2`, `-1` in neither), and `dst_base + e*expert_bytes` is the destination, so this is the
// existing per-expert loop with the READ hoisted out of it: the bytes, the destination layout and the pool
// admission are all unchanged (an expert read here is still published into an EMPTY pool slot exactly as
// `expert_host_src` would, and is still counted by the caller).
//
// Why it matters: the experts in neither tier are contiguous in the file (`src_offs + e*host_bytes`) whenever
// the table is unsplit, and a prefill stages the whole host set per ubatch.  One ~0.7 MiB O_DIRECT read at a
// time measures this box's drive at ~2.1 GiB/s (latency-bound at depth 1 -- `dd iflag=direct` says 520 MiB/s
// for a 0.6 MiB read, 4.2 GiB/s for 16 MiB) while 16-expert chunks are ~10.9 MiB.
//
// A chunk whose read comes up short is left UNSERVED and counted: the caller's per-expert loop then handles
// those experts, and a read failure never fails the stage (the model mapping is still a valid source, so a
// short read is a slowdown, not a correctness event).  `served` (sized n_experts) is set to 1 for every
// expert written here so the caller's loop can skip it.  Caller holds `g_mutex` and a `device_guard`, and has
// already established the geometry this needs: unsplit (`host_bytes == expert_bytes`, `src_off == 0`), no
// `do2d`, no fused slices, a registered disk source and O_DIRECT mode.
static cudaError_t stage_span_fill(table_t & t, host_pool_t * pool, const int32_t * map, int device,
                                   char * dst_base, void * stream, std::vector<uint8_t> & served) {
    served.assign((size_t) t.n_experts, 0);
    if (t.src_fd < 0) {
        t.src_fd = pool_open_file(t.src_path.c_str(), g_pool_dio);
        if (t.src_fd >= 0) {
            pool_fd_align(t.src_fd, &t.src_align);
        }
    }
    if (t.src_fd < 0) {
        return cudaSuccess;   // nothing served: the caller's per-expert path runs exactly as before
    }
    const cudaStream_t st    = (cudaStream_t) stream;
    const size_t       chunk = (size_t) g_pool_stage_span_n;
    const int          SPAN_MIN = MOE_STAGE_SPAN_MIN < g_pool_stage_span_n ? MOE_STAGE_SPAN_MIN : g_pool_stage_span_n;
    cudaError_t        err   = cudaSuccess;
    int e = 0;
    while (e < t.n_experts && err == cudaSuccess) {
        if (map[e] != -1) {
            e++;
            continue;
        }
        int run_end = e + 1;
        while (run_end < t.n_experts && map[run_end] == -1) {
            run_end++;
        }
        // A SHORT run is left to the per-expert path: below `SPAN_MIN` experts a span reads about as many
        // bytes as the single expert it replaces and pays the ring bookkeeping on top.  Measured at -lme 36
        // (the pool holds 393 of 512 experts, so the mean missing run is 1.2 experts): coalescing them all
        // cost 12 % of prefill.  The pool size decides which arm a budget lands in, not this heuristic.
        if (run_end - e < SPAN_MIN) {
            e = run_end;
            continue;
        }
        while (e < run_end && err == cudaSuccess) {
            const size_t n     = std::min(chunk, (size_t) (run_end - e));
            if (n < SPAN_MIN) {
                e += (int) n;   // the leftover tail of a run: the per-expert path reads it
                continue;
            }
            const size_t off   = t.src_offs + (size_t) e * t.host_bytes;
            const size_t len   = n * t.expert_bytes;
            // O_DIRECT needs an aligned offset, length AND buffer, so a direct chunk is read padded and the
            // bytes start part-way into it.  A buffered `pread` needs none of that, and reading exactly
            // `len` at `off` also keeps the last chunk of the last table from padding past EOF (a short read
            // would decline the chunk that the per-expert path could serve).
            const size_t align = t.src_align != 0 && (t.src_align & (t.src_align - 1)) == 0 ? t.src_align : 4096;
            const size_t aoff  = g_pool_dio ? (off & ~(align - 1)) : off;
            const size_t skip  = off - aoff;
            const size_t blen  = g_pool_dio ? ((skip + len + align - 1) & ~(align - 1)) : len;
            span_buf_t * sb    = span_buf_acquire(device, len, align);
            const int64_t tr = ggml_time_us();
            const pool_ssize_t got = (sb != nullptr && blen <= sb->cap)
                                   ? pool_pread(t.src_fd, sb->host, blen, aoff)
                                   : (pool_ssize_t) -1;
            g_stage_span_read_us += ggml_time_us() - tr;
            if (sb == nullptr || got != (pool_ssize_t) blen) {
                g_stage_span_fallback++;   // this chunk takes the per-expert path
                e += (int) n;
                continue;
            }
            const char * base = sb->host + skip;
            for (size_t i = 0; i < n && err == cudaSuccess; i++) {
                const int    ee = e + (int) i;
                const char * s  = base + i * t.expert_bytes;
                if (pool != nullptr && !pool->failed &&
                    pool->expert_slot.find(ee) == pool->expert_slot.end()) {
                    // TODO #281: a PREFILL fill is the coverage path -- it ignores the exchange headroom
                    // (the decode-band reserve is what keeps a demote's landing zone).
                    const int slot = pool_free_slot_locked(*pool, 0);
                    if (slot >= 0 && pool_sink_active(pool)) {
                        g_sink_prefill_fills++;
                    }
                    if (slot >= 0) {
                        memcpy(pool->base + (size_t) slot * pool->host_bytes, s, t.expert_bytes);
                        pool_slot_assign_locked(*pool, slot, ee);
                    }
                }
                err = cudaMemcpyAsync(dst_base + (size_t) ee * t.expert_bytes, s, t.expert_bytes,
                                      cudaMemcpyHostToDevice, st);
                if (err == cudaSuccess) {
                    served[(size_t) ee] = 1;
                }
            }
            if (err == cudaSuccess) {
                g_stage_span_reads++;
                g_stage_span_bytes   += (int64_t) len;
                g_stage_span_experts += (int64_t) n;
                // Hold the buffer until its copies drain; the ring is two deep, so the read of the next
                // chunk overlaps this one's H2Ds.
                if (cudaEventRecord(sb->ev, st) == cudaSuccess) {
                    sb->busy = true;
                } else {
                    (void) cudaGetLastError();
                }
            }
            e += (int) n;
        }
    }
    return err;
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
    if (!moe_cache_slice_addr_ok(host_bytes, expert_bytes, slice_off, host_pitch)) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            GGML_LOG_WARN("%s: declining a MoE expert table with an unaddressable split slice "
                          "(layer=%d role=%s host_bytes=%zu expert_bytes=%zu slice_off=%zu host_pitch=%zu, "
                          "split_axis=%d); the scheduler's full-table copy will serve it, so decode uses "
                          "the host path for that table.  Please report this geometry.\n",
                          __func__, name_layer(weight), name_role(weight).c_str(),
                          host_bytes, expert_bytes, slice_off, host_pitch, split_axis);
        }
        // WIP moe-upload-overlap: this table can never enter the arena; drop it from the expected bytes
        // so the deferred sizing is not blocked waiting for it.
        moe_cache_note_unregisterable(weight, device, n_experts, expert_bytes);
        return false;
    }
    // Register the table with the cache even though this pass uploads through the gather.  The
    // deferred arena sizing (`alloc_all_locked`) latches on the first full decode pass and sizes every
    // table registered by then, so a table that is ONLY ever uploaded through the gather path would be
    // missing from that set - the arena then covers only the tables some other path happened to
    // register (observed: the last layer alone), and the decode band falls below the uncached path.
    // Registration is a cheap map insert and does not allocate (the deferred path leaves slots at 0).
    // wip/host-pool-speed (mechanism (b)): which sources this table has.  A pinned master is device-readable
    // (the legacy gather).  A POOL-backed table has no device-readable master at all -- its mapping is
    // pageable -- but it DOES have the pinned pool, whose slots are `cudaHostAlloc`'d and therefore sound to
    // read from a kernel, plus the arena.  Resolve arena -> pool -> (caller-filled dst), and never the
    // master.  The pool is looked up under the same lock as the snapshot below.
    bool          master_readable = false;
    host_pool_t * gpool           = nullptr;
    {
        const int layer = name_layer(weight);
        if (layer >= 0) {
            const int table = moe_cache_table(weight, layer, name_role(weight).c_str(), n_experts, expert_bytes,
                                              host_bytes, slice_off, host_pitch, split_axis, weight->data, device);
            // The gather kernel reads the host master IN PLACE (`weight->data` below).  That is only
            // safe when the master has a device mapping; for a pageable model mapping the kernel read
            // is a fatal page-not-present fault (issue #116).  Decline and let the scheduler's host
            // path stage the experts through its pinned scratch instead.
            if (table >= 0) {
                std::lock_guard<std::mutex> lock(g_mutex);
                table_t & t = g_tables[table];
                master_readable = t.host_dev != nullptr;
                if (!master_readable) {
                    host_pool_t * p = table_pool_locked(t);
                    if (p != nullptr && !p->failed && p->base_dev != nullptr &&
                        t.src_off + t.expert_bytes <= t.host_bytes) {
                        gpool = p;
                    }
                }
            }
            if (!master_readable && gpool == nullptr) {
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
        static const bool once = [] { const char * e = getenv("GGML_MOE_GATHER_ONCE"); return e != nullptr && atoi(e) != 0; }();
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
    // wip/host-pool-speed (mechanism (b)): build this call's two-tier residency map for the gather and, for
    // every routed expert in NEITHER tier, copy it into `dst` from a pinned source (`expert_host_src`:
    // a pool slot or the per-table bounce) BEFORE the kernel runs.  The kernel then skips those entries, so
    // no kernel ever reads the pageable mapping.  The snapshot is rebuilt on every call -- it depends on the
    // pool's slot map, which the demand path changes without bumping `g_write_seq` -- but the H2D is skipped
    // when the map is unchanged, so a steady ubatch pays one memcmp of `n_experts` int32s.
    int32_t *      gprs_dev    = nullptr;
    int            gprs_slots  = 0;
    const char *   garena      = nullptr;
    const char *   gpool_dev   = nullptr;
    int64_t        gpool_stride = 0;
    int            skip_miss   = 0;
    if (gpool != nullptr) {
        std::lock_guard<std::mutex> lock(g_mutex);
        // The table is identified by its MASTER POINTER (which is also the pool's key), not by geometry:
        // two tables of one layer share a geometry, and `table_pool_locked()` allocates on a miss, so it
        // must never be called on a candidate.
        table_t * tp = nullptr;
        for (table_t & t : g_tables) {
            if (t.host == weight->data && t.device == device && t.n_experts == n_experts &&
                t.host_bytes == host_bytes && (size_t) t.expert_bytes == expert_bytes &&
                t.src_off == slice_off && t.host_pitch == host_pitch && t.split_axis == split_axis &&
                t.host_dev == nullptr) {
                tp = &t;
                break;
            }
        }
        if (tp == nullptr) {
            return false;
        }
        table_t & t = *tp;
        if (t.gprs_dev == nullptr || t.gprs_n != (int64_t) n_experts) {
            if (t.gprs_dev != nullptr) {
                (void) cudaFree(t.gprs_dev);
                t.gprs_dev = nullptr;
            }
            if (cudaMalloc((void **) &t.gprs_dev, (size_t) n_experts * sizeof(int32_t)) != cudaSuccess) {
                (void) cudaGetLastError();
                t.gprs_dev = nullptr;
                return false;
            }
            t.gprs_n     = n_experts;
            t.gprs_arena = nullptr;
            t.gprs_slots = 0;
            t.gprs_last.assign((size_t) n_experts, INT32_MIN);   // force the first upload
        }
        if ((int) t.gprs_host.size() != n_experts) {
            t.gprs_host.assign((size_t) n_experts, -1);
        } else {
            std::fill(t.gprs_host.begin(), t.gprs_host.end(), -1);
        }
        for (const auto & kv : t.expert_slot) {
            const int e = kv.first;
            if (e >= 0 && e < n_experts && kv.second >= 0 && kv.second < t.slots) {
                t.gprs_host[(size_t) e] = kv.second;
            }
        }
        for (const auto & kv : gpool->expert_slot) {
            const int e = kv.first;
            const int s = kv.second;
            if (e < 0 || e >= n_experts || s < 0 || s >= gpool->slots ||
                (size_t) s >= gpool->slot_state.size() || gpool->slot_state[(size_t) s] != 1) {
                continue;
            }
            if (t.gprs_host[(size_t) e] < 0) {
                t.gprs_host[(size_t) e] = -(s + 2);
            }
        }
        if (t.gprs_arena != (void *) t.arena || t.gprs_slots != t.slots ||
            t.gprs_last.size() != t.gprs_host.size() ||
            !std::equal(t.gprs_last.begin(), t.gprs_last.end(), t.gprs_host.begin())) {
            if (cudaMemcpyAsync(t.gprs_dev, t.gprs_host.data(), (size_t) n_experts * sizeof(int32_t),
                                cudaMemcpyHostToDevice, (cudaStream_t) stream) != cudaSuccess) {
                (void) cudaGetLastError();
                return false;
            }
            t.gprs_last    = t.gprs_host;
            t.gprs_arena   = (void *) t.arena;
            t.gprs_slots   = t.slots;
        }
        // The experts in neither tier: a pinned bounce / a fresh pool slot, copied straight into the
        // destination.  Ordered before the kernel by the stream, so the kernel may skip them.
        //
        // wip/host-expert-split W4 (prefill read granularity): this is where a PREFILL actually reads
        // storage (the scheduler staging path does not run for this shape -- it declines), and it used to
        // issue one ~0.7 MiB read per expert.  The missing experts are contiguous in the file, so serve
        // each run of them from one read per chunk instead; anything a chunk cannot serve falls through
        // to the per-expert path below unchanged.
        //
        // A run is served from the FILE in BOTH fill modes (issue #281): O_DIRECT when the disk tier is on,
        // a plain buffered `pread` otherwise.  This predicate used to require `g_pool_dio`, so a buffered
        // --load-mode mmap run -- the field default since r212 -- could not coalesce at all and read every
        // missing expert out of the model MAPPING one at a time: ~0.65 MiB demand faults at 715 MiB/s on
        // the reporter's Windows box, a 13.9 GiB bounce ring, 185 k experts through the per-expert path and
        // 22 t/s prefill where the same box measured 70 t/s with O_DIRECT.  A buffered `pread` is the same
        // page cache `--load-mode mmap` asked for (the mapping and the file share its pages), so this is
        // not a fill-source change on Linux -- it only stops the read granularity from depending on it,
        // and it gives Windows the Cache Manager's readahead instead of a fault per 4 KiB page.
        std::vector<uint8_t> span_done;
        const bool span_geom = g_pool_stage_span && pool_file_io_available() &&
                               t.src_known && !t.src_path.empty() && t.n_slices == 0 &&
                               !(t.split_axis == 0 && t.host_pitch != 0) && t.src_off == 0 &&
                               t.host_bytes == expert_bytes && expert_bytes > 0 &&
                               gpool->base != nullptr && !gpool->failed &&
                               gpool->host_bytes == t.host_bytes;
        cudaError_t err = cudaSuccess;
        if (span_geom) {
            err = stage_span_fill(t, gpool, t.gprs_host.data(), device, (char *) weight_cpy->data, stream,
                                  span_done);
        }
        for (int e = 0; e < n_experts && err == cudaSuccess; e++) {
            if (!span_done.empty() && span_done[(size_t) e] != 0) {
                continue;   // already copied from a coalesced span read
            }
            if (t.gprs_host[(size_t) e] != -1) {
                continue;
            }
            char * d = (char *) weight_cpy->data + (size_t) e * expert_bytes;
            expert_src_t hsrc;
            if (!expert_host_src(t, e, gpool, /*allow_empty_fill*/ true, &hsrc, nullptr, /*prefill*/ true)) {
                return false;
            }
            if (t.split_axis == 0 && t.host_pitch != 0) {
                const int64_t rows      = (int64_t) (t.host_bytes / t.host_pitch);
                const int64_t row_bytes = rows > 0 ? (int64_t) (t.expert_bytes / (size_t) rows) : 0;
                err = cudaMemcpy2DAsync(d, (size_t) row_bytes, hsrc.base + t.src_off, t.host_pitch,
                                        (size_t) row_bytes, (size_t) rows, hsrc.kind, (cudaStream_t) stream);
            } else {
                err = cudaMemcpyAsync(d, hsrc.base + t.src_off, expert_bytes, hsrc.kind, (cudaStream_t) stream);
            }
            if (err == cudaSuccess) {
                expert_src_done(t, gpool, hsrc, stream, /*retire*/ false);
                g_gather_fills++;
            }
        }
        if (err != cudaSuccess) {
            (void) cudaGetLastError();
            return false;
        }
        g_gather_pool++;
        gprs_dev     = t.gprs_dev;
        gprs_slots   = t.slots;
        garena       = (const char *) t.arena;
        gpool_dev    = (const char *) gpool->base_dev;
        gpool_stride = (int64_t) gpool->host_bytes;
        skip_miss    = 1;
    }
    moe_cache_gather_kernel<<<n_experts * n_split, threads, 0, (cudaStream_t) stream>>>(
        (char *) weight_cpy->data, (const char *) weight->data,
        (const int32_t *) ids->data, ids->nb[0], ids->nb[1], (int) n_used, (int) n_tok, n_experts, n_split,
        (int64_t) expert_bytes, (int64_t) host_bytes, (int64_t) slice_off, (int64_t) host_pitch, split_axis,
        garena, gprs_dev, gprs_slots, gpool_dev, gpool_stride, skip_miss);
    if (cudaGetLastError() != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    g_gather_calls++;
    return true;
}



// TODO #271: record (once per table) why the prefill staging path declined it.  This was silent at every
// verbosity: the census counted the decline but named nothing, so "which tables, and why" needed a patch.
// A declined table is NOT a slower table by itself -- measured on 1 GPU with a pinned master, `-ub 6144`,
// 50 declined tables: 1826 t/s declined against 1824 t/s staged (ABBA, +-0.1 %), because the scheduler's
// own whole-table upload is a single bulk async copy on the same copy stream.  What it costs is the
// TRANSPORT ACCOUNTING (its bytes are missing from the staging census, and its engine is not the ring's)
// and, on a shape where the two transports really do differ, the wrong one being chosen without a line in
// the log saying so.
static void stage_decline_note(table_t & t, const char * why) {
    if (t.stage_declined) {
        return;
    }
    t.stage_declined = true;
    g_stage_declined.push_back({ t.layer, t.role, why, t.device });
}

// Fill a whole-table staging copy from the arena + the pinned host
// master.  `host_src`/`offset` identify the registered table (the same key the decode hooks use), and
// `width`/`stride_src`/`n_copies` describe the exact copy the caller asked for; a geometry mismatch
// declines and the caller falls back to the plain upload.  Returns true when the destination was
// filled and the caller's event recorded (the caller records `ev` after this returns).
bool moe_cache_stage_table(int device, const void * host_src, size_t offset, size_t width, size_t stride_src,
                           size_t n_copies, void * dst, void * compute_stream, void * copy_stream,
                           void * order_event) {
    if (!g_stage_cache || !g_enabled || host_src == nullptr || dst == nullptr || width == 0 || n_copies == 0) {
        return false;
    }
    static uint64_t ordered_seq[GGML_CUDA_MAX_DEVICES] = {};
    std::lock_guard<std::mutex> lock(g_mutex);
    for (table_t & t : g_tables) {
        if (t.device != device || t.host != host_src || t.src_off != offset) {
            continue;
        }
        if (t.n_experts <= 0 || t.host == nullptr) {
            g_stage_decline_usable++;
            stage_decline_note(t, "the cache holds no expert count/master for it");
            return false;
        }
        // The pinned-master path needs a built arena (the staging kernel reads it) and a device-readable
        // master.  A POOL-BACKED master needs neither: it is filled expert by expert from the pinned pool
        // (or the per-table bounce), so it also serves the pre-sizing prefill (the whole point -- a
        // pageable master must never reach the ring through the pageable upload).
        if (t.host_pinned) {
            // The arena requirement is the one that fires in practice, and it is TRANSIENT: the slab hands
            // the chunks a table's arena lived in to the compute layout, the table is stood down for the
            // rest of the prefill, and the drop at the prefill -> decode transition re-arms it (TODO #42).
            // Which of the four conditions fired, and for which tables, used to be unrecorded -- this is
            // the line that answers it (TODO #271).
            const char * why =
                !t.cold_safe         ? "its host geometry is not representable by a staged copy (a fused/sliced table)"
              : t.host_dev == nullptr ? "its host master has no device-readable (pinned) mapping"
              : t.arena == nullptr || t.slots <= 0
                  ? (t.alloc_failed ? "its arena allocation failed"
                     : t.allocated  ? "it is stood down (0 slots, no arena) -- a slab chunk reclamation or a layer stand-down"
                                    : "the cache has not sized yet")
              : nullptr;
            if (why != nullptr) {
                g_stage_decline_usable++;
                stage_decline_note(t, why);
                return false;
            }
            if (!g_sized) {
                // The priming pass (the arena does not exist yet).  Its own counter, so `unusable` keeps
                // meaning "a table that will never be staged".
                g_stage_decline_unsized++;
                return false;
            }
        }
        const bool do2d = t.split_axis == 0 && t.host_pitch != 0;
        int64_t rows = 0;
        int64_t row_bytes = 0;
        if (do2d) {
            rows      = (int64_t) (t.host_bytes / t.host_pitch);
            row_bytes = rows > 0 ? (int64_t) (t.expert_bytes / (size_t) rows) : 0;
            if (rows <= 0 || (size_t) row_bytes != width || stride_src != t.host_pitch ||
                    (size_t) (rows * row_bytes) != t.expert_bytes) {
                g_stage_decline_geom++;
                return false;
            }
        } else if (n_copies == 1) {
            if (t.host_bytes != t.expert_bytes || offset != 0) {
                g_stage_decline_geom++;
                return false;   // a single contiguous copy can only be an unsplit table
            }
        } else if (width != t.expert_bytes || stride_src != t.host_bytes || n_copies != (size_t) t.n_experts) {
            g_stage_decline_geom++;
            return false;
        }

        device_guard dg(device);

        // wip/host-expert-master-drop (TODO #55): a POOL-BACKED (pageable) master is not readable by the
        // staging kernel, so the bytes come from the two pinned tiers instead: the arena (a D2D copy) and
        // the pinned pool (a device-readable mapped alias).  Only the source of the bytes changes; the
        // destination layout is exactly the one the kernel produces, so every consumer is unchanged.
        //
        // TODO #87 (2): the tiers are filled by ONE kernel launch.  This loop used to issue one
        // `cudaMemcpyAsync` per expert for them (~200 per table per ubatch on the field config), and that
        // per-copy issue cost was the rest of the prefill gap to the pinned-master path, which has always
        // staged with a single launch.  The per-expert map encodes both tiers (`>= 0` arena slot,
        // `<= -2` pool slot `-v-2`, `-1` in neither), so the kernel needs no extra bookkeeping; the experts
        // in NEITHER tier are still copied here, one at a time, because each of those is a distinct read
        // from the mapping/disk.
        if (!t.host_pinned) {
            host_pool_t * pool = table_pool_locked(t);
            cudaError_t   err  = cudaSuccess;
            {
                const int64_t resident = (int64_t) t.expert_slot.size();
                g_stage_ok++;
                g_stage_arena_bytes += resident * (int64_t) t.expert_bytes;
                g_stage_host_bytes  += ((int64_t) t.n_experts - resident) * (int64_t) t.expert_bytes;
            }
            // The two-tier snapshot is rebuilt on EVERY pool-backed staging call: unlike the arena map it
            // also depends on the pool's slot map, which the demand path (and the bulk layout) changes
            // without bumping `g_write_seq`.  It is `n_experts` int32s (~1 KiB) against a whole-table
            // staging copy, so the freshness is free.
            if (t.prs_dev == nullptr || t.prs_n != t.n_experts) {
                if (t.prs_dev != nullptr) {
                    (void) cudaFree(t.prs_dev);
                    t.prs_dev = nullptr;
                }
                if (cudaMalloc((void **) &t.prs_dev, (size_t) t.n_experts * sizeof(int32_t)) != cudaSuccess) {
                    (void) cudaGetLastError();
                    t.prs_dev = nullptr;
                    return false;
                }
                t.prs_n   = t.n_experts;
                t.prs_gen = 0;
            }
            if ((int) t.prs_host.size() != t.n_experts) {
                t.prs_host.assign((size_t) t.n_experts, -1);
            } else {
                std::fill(t.prs_host.begin(), t.prs_host.end(), -1);
            }
            for (const auto & kv : t.expert_slot) {
                const int e = kv.first;
                if (e >= 0 && e < t.n_experts) {
                    t.prs_host[(size_t) e] = kv.second;
                }
            }
            const bool pool_dev_ok = pool != nullptr && pool->base_dev != nullptr &&
                                     t.src_off + t.expert_bytes <= t.host_bytes && !pool->failed;
            // Kill switch (`MOE_HOST_POOL_STAGE_BATCH=0`) for the single-launch arm: the pre-r114 behaviour
            // is one `cudaMemcpyAsync` per tier-resident expert, which is what the A/B is measured against.
            const bool batch = g_pool_stage_batch && pool_dev_ok;
            if (batch) {
                for (const auto & kv : pool->expert_slot) {
                    const int e = kv.first;
                    const int s = kv.second;
                    if (e < 0 || e >= t.n_experts || s < 0 || s >= pool->slots ||
                        (size_t) s >= pool->slot_state.size() || pool->slot_state[(size_t) s] != 1) {
                        continue;
                    }
                    if (t.prs_host[(size_t) e] < 0) {
                        t.prs_host[(size_t) e] = -(s + 2);
                    }
                }
            }
            if (cudaMemcpyAsync(t.prs_dev, t.prs_host.data(), (size_t) t.n_experts * sizeof(int32_t),
                                cudaMemcpyHostToDevice, (cudaStream_t) copy_stream) != cudaSuccess) {
                (void) cudaGetLastError();
                return false;
            }
            t.prs_gen   = g_write_seq.load();
            t.prs_arena = (void *) t.arena;
            t.prs_slots = t.slots;
            g_prs_rebuilds++;
            // wip/host-expert-split W4 (prefill read granularity): the experts in NEITHER tier are the only
            // ones this loop reads from STORAGE, and they are contiguous in the file.  Serve each run of them
            // with one read per ~16-expert chunk (`stage_span_fill`) and skip them below.  Only the single-
            // launch arm can do this: its snapshot is the one that encodes BOTH tiers (`-1` = neither), while
            // the per-expert arm's encodes only the arena.  The default (`MOE_HOST_POOL_STAGE_BATCH=1`) is the
            // arm that runs in practice; the per-expert arm is a debug A/B and keeps its own path.
            std::vector<uint8_t> span_done;
            // Issue #281: no `g_pool_dio` here either -- a buffered fill coalesces too (see the gather
            // path's note; `stage_span_fill` reads the file with the mode's own buffered/O_DIRECT open).
            const bool span_geom = batch && g_pool_stage_span && pool_file_io_available() &&
                                   t.src_known && !t.src_path.empty() && !do2d && t.n_slices == 0 &&
                                   t.src_off == 0 && t.host_bytes == t.expert_bytes && t.expert_bytes > 0 &&
                                   pool != nullptr && pool->base != nullptr &&
                                   pool->host_bytes == t.expert_bytes;
            if (span_geom) {
                err = stage_span_fill(t, pool, t.prs_host.data(), device, (char *) dst, copy_stream, span_done);
            }
            for (int e = 0; e < t.n_experts && err == cudaSuccess; e++) {
                if (!span_done.empty() && span_done[(size_t) e] != 0) {
                    continue;   // already copied from a coalesced span read above
                }
                if (batch && t.prs_host[(size_t) e] != -1) {
                    continue;   // served by the kernel (arena or pool)
                }
                char * d = (char *) dst + (size_t) e * t.expert_bytes;
                if (!batch) {
                    const auto hit = t.expert_slot.find(e);
                    if (hit != t.expert_slot.end() && hit->second >= 0 && hit->second < t.slots) {
                        err = cudaMemcpyAsync(d, (const char *) t.arena + (size_t) hit->second * t.expert_bytes,
                                              t.expert_bytes, cudaMemcpyDeviceToDevice,
                                              (cudaStream_t) copy_stream);
                        continue;
                    }
                }
                // TODO #87: the PREFILL band reads a pool slot but never evicts one, and a miss fills an
                // EMPTY slot (so the next ubatch finds it there).  Only an expert in neither tier is read
                // from the file/mapping -- which is the whole point: a prefill stages the whole host table
                // per ubatch, so without this every ubatch re-read the entire host expert set from disk
                // (measured on the IQ4_NL field config: 39 t/s prefill, 27 minutes of disk reads for one
                // 68K prompt).  The copy below keeps the pool entry (retire = false) because the staging
                // destination is a transient ring slot, not the arena.
                expert_src_t hsrc;
                if (!expert_host_src(t, e, pool, /*allow_empty_fill*/ true, &hsrc, nullptr, /*prefill*/ true)) {
                    err = k_no_host_source;   // see `fill_slot_from_host`: a host-side answer, never a device error
                    break;
                }
                if (do2d) {
                    err = cudaMemcpy2DAsync(d, (size_t) row_bytes, hsrc.base + t.src_off, t.host_pitch,
                                            (size_t) row_bytes, (size_t) rows, hsrc.kind,
                                            (cudaStream_t) copy_stream);
                } else {
                    err = cudaMemcpyAsync(d, hsrc.base + t.src_off, t.expert_bytes, hsrc.kind,
                                          (cudaStream_t) copy_stream);
                }
                if (err == cudaSuccess) {
                    expert_src_done(t, pool, hsrc, copy_stream, /*retire*/ false);
                }
            }
            if (err != cudaSuccess) {
                (void) cudaGetLastError();
                return false;
            }
            // The copies above are on `copy_stream`, so this kernel is ordered after every miss fill.
            if (batch && t.n_experts > 0) {
                moe_cache_stage_kernel<<<t.n_experts, 256, 0, (cudaStream_t) copy_stream>>>(
                    (char *) dst, (const char *) t.arena, (const char *) t.host_dev, t.prs_dev, t.slots,
                    (int64_t) t.expert_bytes, (int64_t) t.host_bytes, (int64_t) t.src_off,
                    rows, row_bytes, (int64_t) t.host_pitch, do2d ? 1 : 0,
                    pool_dev_ok && pool != nullptr ? (const char *) pool->base_dev : nullptr,
                    pool != nullptr ? (int64_t) pool->host_bytes : 0, /*skip_miss*/ 1);
                if (cudaGetLastError() != cudaSuccess) {
                    (void) cudaGetLastError();
                    return false;
                }
            }
            return true;
        }

        // Refresh the immutable snapshot when the cache wrote since we built it, or the arena moved.
        const uint64_t seq = g_write_seq.load();
        if (t.prs_dev == nullptr || t.prs_n != t.n_experts) {
            if (t.prs_dev != nullptr) {
                (void) cudaFree(t.prs_dev);
                t.prs_dev = nullptr;
            }
            if (cudaMalloc((void **) &t.prs_dev, (size_t) t.n_experts * sizeof(int32_t)) != cudaSuccess) {
                (void) cudaGetLastError();
                t.prs_dev = nullptr;
                return false;
            }
            t.prs_n   = t.n_experts;
            t.prs_gen = 0;
        }
        if (t.prs_gen != seq || t.prs_arena != (void *) t.arena || t.prs_slots != t.slots) {
            if ((int) t.prs_host.size() != t.n_experts) {
                t.prs_host.assign((size_t) t.n_experts, -1);
            } else {
                std::fill(t.prs_host.begin(), t.prs_host.end(), -1);
            }
            for (const auto & kv : t.expert_slot) {
                const int e = kv.first;
                if (e >= 0 && e < t.n_experts) {
                    t.prs_host[(size_t) e] = kv.second;
                }
            }
            // The snapshot covers the whole arena (no freeze partition): it is rebuilt whenever the cache
            // writes (`g_write_seq`), the arena moves or the slot count changes, and the decode path never
            // mutates it, so a resident expert can never change under the staging kernel.
            // Upload on the COPY stream: the staging kernel follows it on the same stream, so the map is
            // ordered where it is read, and the copy stream is NOT forced to wait for the compute stream
            // just because a map was (re)built.  Only an arena write orders against compute below.
            if (cudaMemcpyAsync(t.prs_dev, t.prs_host.data(), (size_t) t.n_experts * sizeof(int32_t),
                                cudaMemcpyHostToDevice, (cudaStream_t) copy_stream) != cudaSuccess) {
                (void) cudaGetLastError();
                return false;
            }
            t.prs_gen   = seq;
            t.prs_arena = (void *) t.arena;
            t.prs_slots = t.slots;
            g_prs_rebuilds++;
        }

        // Order the copy stream after the compute stream ONLY when the arena may have been written since
        // the last staging on this device (a decode admission between prefill ubatches).  During a pure
        // prefill nothing writes, so the staging keeps overlapping the previous split's compute.
        if (ordered_seq[device] != seq) {
            if (cudaEventRecord((cudaEvent_t) order_event, (cudaStream_t) compute_stream) != cudaSuccess ||
                cudaStreamWaitEvent((cudaStream_t) copy_stream, (cudaEvent_t) order_event, 0) != cudaSuccess) {
                (void) cudaGetLastError();
                return false;
            }
            ordered_seq[device] = seq;
        }

        {
            const int64_t resident = (int64_t) t.expert_slot.size();
            g_stage_ok++;
            g_stage_arena_bytes += resident * (int64_t) t.expert_bytes;
            g_stage_host_bytes  += ((int64_t) t.n_experts - resident) * (int64_t) t.expert_bytes;
        }
        moe_cache_stage_kernel<<<t.n_experts, 256, 0, (cudaStream_t) copy_stream>>>(
            (char *) dst, (const char *) t.arena, (const char *) t.host_dev, t.prs_dev, t.slots,
            (int64_t) t.expert_bytes, (int64_t) t.host_bytes, (int64_t) t.src_off,
            rows, row_bytes, (int64_t) t.host_pitch, do2d ? 1 : 0, nullptr, 0, /*skip_miss*/ 0);
        return cudaGetLastError() == cudaSuccess;
    }
    g_stage_decline_nomatch++;
    return false;
}

// Geometry-only registration: the half of `moe_cache_update_host` that needs no routing.
//
// The scheduler calls it for every routed host-expert input of EVERY split before the first split of a
// graph runs (gated on `moe_cache_pending`, see the backend iface).  Two things depended on this and
// neither worked before:
//   * a workload whose FIRST pass is wide (llama-bench `-p`, llama-perplexity, the embedding tools) never
//     armed the arena, because the deferred sizing waited for a <=8-token warmup pass;
//   * under `-sm tensor` the Meta `moe_cache_update` declines every prefill width, so a prefill-only run
//     never observed its table set at all.
// Registration is geometry-only, idempotent, and needs no routing, so it can run before the first ubatch.
// Returns the table id, or -1 when this table must stay on the scheduler's own copy path.
int moe_cache_register_host(const ggml_tensor * weight, const ggml_tensor * weight_cpy, int device,
                            size_t slice_off, int split_axis,
                            const struct ggml_backend_moe_slice * slices, int n_slices) {
    if (!g_enabled || weight == nullptr || weight->ne[2] < 1) {
        return -1;
    }
    const int layer = name_layer(weight);
    if (layer < 0) {
        return -1;
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
    const bool fused = slices != nullptr && n_slices > 0;
    if (fused) {
        // A fused table (`ffn_gate_up_exps` under `-sm tensor`) carries the whole geometry in `slices`;
        // reject a list that does not tile the device slot, so a caller bug declines instead of filling
        // an under-filled arena slot.
        if (!moe_cache_slices_addr_ok(host_bytes, expert_bytes, slices, n_slices)) {
            static bool warned_fused = false;
            if (!warned_fused) {
                warned_fused = true;
                GGML_LOG_WARN("%s: declining a fused MoE expert table with an inconsistent slice list "
                              "(layer=%d role=%s host_bytes=%zu expert_bytes=%zu n_slices=%d); the "
                              "scheduler's per-op copy will serve it.  Please report this geometry.\n",
                              __func__, layer, name_role(weight).c_str(), host_bytes, expert_bytes, n_slices);
            }
            moe_cache_note_unregisterable(weight, device, (int) n_experts, expert_bytes);
            return -1;
        }
    } else if (!moe_cache_slice_addr_ok(host_bytes, expert_bytes, slice_off, host_pitch)) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            GGML_LOG_WARN("%s: declining a MoE expert table with an unaddressable split slice "
                          "(layer=%d role=%s host_bytes=%zu expert_bytes=%zu slice_off=%zu host_pitch=%zu, "
                          "split_axis=%d); the scheduler's full-table copy will serve it, so decode uses "
                          "the host path for that table.  Please report this geometry.\n",
                          __func__, layer, name_role(weight).c_str(),
                          host_bytes, expert_bytes, slice_off, host_pitch, split_axis);
        }
        // WIP moe-upload-overlap: this table can never enter the arena; drop it from the expected bytes
        // so the deferred sizing is not blocked waiting for it.
        moe_cache_note_unregisterable(weight, device, n_experts, expert_bytes);
        return -1;
    }
    const int table = moe_cache_table(weight, layer, name_role(weight).c_str(), n_experts, expert_bytes,
                                      host_bytes, slice_off, host_pitch, split_axis, weight->data, device,
                                      fused ? slices : nullptr, fused ? n_slices : 0);
    if (table < 0) {
        return -1;
    }
    // Prime and arm.  Registering supersedes `moe_cache_update_host`'s priming pass (`t.primed`), and the
    // moment the table set is complete the arena sizes -- arming HERE means the first ubatch of this very
    // pass is already served from the arena, not the second pass onwards.  A warmup-first workload is
    // unchanged: its first graph is the 2-token warmup, whose boundary is the narrow one this sizing wanted
    // (the OPEN 2 caveat only applies to a prefill-first workload, where the wide work layout is live and
    // the arena comes out a little smaller than it would after the drop).
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        table_t & t = g_tables[table];
        if (!t.primed) {
            t.primed = true;
            g_total_expert_bytes += (int64_t) t.expert_bytes;
            g_total_one_expert_bytes += (int64_t) t.expert_bytes;
        }
        if (!g_sized && g_size_at_prime && moe_cache_tables_complete_locked()) {
            GGML_LOG_INFO("%s: the host-expert table set is complete at registration (%d tables); arming "
                          "the arena before the first ubatch of this pass\n", __func__, (int) g_tables.size());
            g_alloc_priming = true;
            alloc_all_locked();
            g_alloc_priming = false;
        }
    }
    return table;
}

bool moe_cache_update_host(const ggml_tensor * weight, const ggml_tensor * weight_cpy,
                           const int32_t * ids, int64_t n_used, int64_t n_tok,
                           size_t ids_nb0, size_t ids_nb1, void * stream,
                           int device, size_t slice_off, int split_axis,
                           const struct ggml_backend_moe_slice * slices, int n_slices) {
    g_write_seq++;
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

    // TODO #61 P1: the DECODE band's routing trace record.  The scheduler has already read these ids back
    // to the host (that is why this hook exists), so recording them costs no readback and no sync.  A
    // prefill ubatch is not traced from here: its ids never reach this hook under the default staging, so
    // the PREFILL records come from the device entry (`moe_cache_heat_observe`) instead.
    if (g_trace_on && !g_trace_stopped && n_tok <= MOE_EXPERT_CACHE_MAX_TOK) {
        std::vector<int32_t> ids_host((size_t) n_used * (size_t) n_tok, -1);
        for (int64_t tok = 0; tok < n_tok; tok++) {
            for (int64_t j = 0; j < n_used; j++) {
                ids_host[(size_t) (tok * n_used + j)] =
                    *(const int32_t *) ((const char *) ids + (size_t) tok * ids_nb1 + (size_t) j * ids_nb0);
            }
        }
        moe_trace_emit('D', layer, sem_role(weight).c_str(), (int) n_tok, (int) n_used, ids_host.data(),
                       (int) weight->ne[2], weight->nb[2] > 0 ? (size_t) weight->nb[2] : 0);
    }
    const int table = moe_cache_register_host(weight, weight_cpy, device, slice_off, split_axis, slices, n_slices);
    if (table < 0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    table_t & t = g_tables[table];
    t.dr_lag_n = 0;   // this path runs the policy on the current routing; a pending device-remap lag is stale
    if (t.fill_failed) {
        // issue #274: this table's fills have failed, so it has no arena it can serve from.  Decline BEFORE
        // anything is remapped: a take-over here would skip the scheduler's `input_cpy` copy and the consumer
        // would read an arena that was never filled.
        g_decline_all++;
        return false;
    }

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

    // TODO #64.1: the first decode-band pass after a prefill is the moment the prompt's routing is known
    // in full.  Re-derive the host pool's warm order from it once (the staged prefill never feeds
    // `prefill_count`, and the device tally is the only record of that routing).  Gated on
    // `g_heat_prefill_ops` so a pre-prompt warmup pass does not consume the one-shot decision, and on the
    // pool so a run without `--host-experts pool` pays nothing.
    if (g_heat_on && g_pool_enabled && !g_pool_rank_done &&
        g_heat_prefill_ops.load(std::memory_order_relaxed) > 0) {
        g_pool_rank_done = true;
        pool_rank_from_prefill_locked();
    }

    // A pageable axis-0 slice (a `-sm tensor` `down_exps` split) has no device mapping and its only
    // cache fill would be a strided 2-D H2D -- the pageable `cudaMemcpy2DAsync` that faults on ROCm 7.14
    // (issue #116's copy-side twin).  wip/host-expert-master-drop (TODO #55): a POOL-BACKED master is no
    // longer a reason to decline -- its fill is a 2-D copy from a pinned pool slot (or the pageable 1-D
    // per-row fallback), never a pageable 2-D transfer -- so the decline only covers a master the runtime
    // failed to pin for any other reason.
    if (t.n_slices == 0 && t.host != nullptr && !table_master_pageable(t) && t.host_dev == nullptr &&
        t.split_axis == 0 && t.host_pitch > 0 && !t.identity) {
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
        devremap_free_locked(t);
        free_heat_dev_locked(t);   // the mirror lives on the OLD device; fold + free before the move
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
            // The arena and the heat mirrors bump for themselves; the remap and the routing staging are
            // raw launch parameters, so bump once for the move.
            ggml_cuda_graph_mem_freed(old_device);
            // Slab-backed arenas go back to the slab's free list; `cudaFree` can unmap the slab.
            if (old_arena != nullptr) { free_arena_backing(old_device, old_arena, old_reserved, old_slots, t); }
            if (old_remap != nullptr) { (void) cudaFree(old_remap); }
            if (t.prs_dev != nullptr) { (void) cudaFree(t.prs_dev); t.prs_dev = nullptr; }
            t.prs_n = 0; t.prs_gen = 0; t.prs_arena = nullptr; t.prs_slots = 0;
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
    // Registration (and the priming-time sizing) already ran in `moe_cache_register_host`, which the
    // scheduler calls for every split before the first one computes -- so `t.primed` is set and the arena is
    // normally armed by the time this hook runs.  What is left here is the deferred sizing for a table whose
    // set never completed at registration (a backend without the register hook, or a partial graph), and
    // that keeps the original narrow-band rule.
    if (g_slots_hint <= 0) {
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
    g_dr_key_to_id[{ (const void *) weight, device }] = table;   // the device remap finds its table by the host master

    if (t.remap_dev == nullptr) {
        const int64_t cap = (int64_t) t.n_experts * 8;
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

    // wip/moe-heatmap-persistence (R2): the DECODE-band host tally.  This eager path is the only one
    // with the true routing on the host, so the tally is free (no extra launch) and exact.  The
    // PREFILL band is deliberately NOT tallied here: it is counted device-side by
    // `moe_cache_heat_observe`, which also covers the staged path that never reaches this hook.
    if (g_heat_on && n_tok <= MOE_EXPERT_CACHE_MAX_TOK && t.n_experts > 0) {
        if ((int) t.heat.size() != t.n_experts) {
            t.heat.assign((size_t) t.n_experts, 0);
        }
        // TODO #88: this IS the decode band (the prefill returns above), so it feeds the decode counter
        // too -- the one the arena seed ranks by.
        if ((int) t.heat_dec.size() != t.n_experts) {
            t.heat_dec.assign((size_t) t.n_experts, 0);
        }
        for (int32_t e : used) {
            if (e >= 0 && e < t.n_experts) {
                if (t.heat[(size_t) e] != 0xffffffffu) {
                    t.heat[(size_t) e]++;
                    g_heat_dirty = true;
                }
                if (t.heat_dec[(size_t) e] != 0xffffffffu) {
                    t.heat_dec[(size_t) e]++;
                    g_heat_dirty = true;
                }
            }
        }
    }

    // A cache too small to hold this token's experts cannot guarantee a complete remap; decline
    // DETERMINISTICALLY (this bound is a constant per shape) so a shape never captures the arena and then
    // declines it on a later replay.  With UVA cold reads the overflow is representable (id = slots+e),
    // so a small arena is fine and the decline is not needed.
    //
    // r210: a table whose pinned pool COVERS the tensor has the same property as a pinned master (every
    // expert is a cold source), so it must be exempt from the arena bound for the same reason -- the
    // bound is about REPRESENTABILITY, not about the arena.  Bounding a covering pool by the arena is
    // what made `llama-batched-bench -npl 8` collapse (8 x 8 routed experts against 56 slots: every op
    // declined and the run read the pageable mapping; tg 49.2 against 270.5 pinned, and 362.7 once the
    // arena was merely large enough to dodge the bound).  See `pool_covers_table`.
    if (!table_cold_ok(t) && !pool_covers_table(t) && t.slots < (int) (n_used * n_tok)) {
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

    // issue #274 follow-up (@briansp2020): a fill can fail DURING the pass above, and that is the moment the
    // table's `fill_failed` is set.  The guard at the top of this function only covers a table that had
    // ALREADY failed on a previous call, so the call that fails half way through used to carry on: pass 1 had
    // admitted the experts before the failure (and rolled the failed one back), pass 2 then found a valid
    // cold or pool id for the failed expert (a pool HIT is rolled back the same way but is still resident in
    // the pool), `all` stayed true and the hook returned TRUE.  The scheduler then skipped its `input_cpy`
    // copy -- while `moe_cache_get_table`, which the consuming op calls a few lines later, sees the SAME
    // `fill_failed` and returns false.  The op therefore read an `input_cpy` that was never populated, which
    // is the reported "correct until the pool starts failing, then `This!!!!!!`" output (NaN-shaped logits),
    // reproduced on stock r220 with `MOE_HOST_POOL_FILL_FAULT=200`.  Decline HERE, after the fills and before
    // anything is remapped, so the scheduler populates the copy the op will actually read.
    if (t.fill_failed) {
        g_decline_all++;
        return false;
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
            } else if (const int ps = pool_cold_slot_locked(t, e); ps >= 0) {
                // MOE_HOST_POOL_COLD: the expert's pinned pool slot, read in place (`moe_cache_get_cold` returns
                // the pool's base for this table)
                remap[(size_t) tok * n_used + j] = n_res_slots + ps;
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
    return true;
}

// ---- Sync-free decode device remap (GGML_MOE_DEVREMAP) ------------------------------------------------------------
// The eager path reads the routing back to the host (one synchronize per routed op), runs the residency policy and
// uploads the remap.  Here a kernel maps the device routing through a per-table expert -> slot map (slots + e = cold,
// read in place from the pinned master) and writes the routing into mapped pinned memory; the policy runs on that
// routing at the next call (one call late) and re-uploads the map when residency changed.  The bytes each expert is
// read from are the same (arena slot or the pinned master), so the output does not change.  Pinned masters only
// (`table_cold_ok`), single slice, no identity table.  GGML_MOE_DEVREMAP=0: off (the eager path).
#define MOE_DEVREMAP_MAX_JOBS 4
struct moe_devremap_jobs {
    const int32_t * map[MOE_DEVREMAP_MAX_JOBS];
    int32_t *       remap[MOE_DEVREMAP_MAX_JOBS];
    int32_t *       lag[MOE_DEVREMAP_MAX_JOBS];
    int32_t *       seq[MOE_DEVREMAP_MAX_JOBS];      // the lag completion flag (nullptr: the host uses the event)
    int32_t         seq_val[MOE_DEVREMAP_MAX_JOBS];
};

static __global__ void moe_cache_remap_kernel(const int32_t * __restrict__ ids, int64_t ids_s1, int n_used, int n_tok,
                                              int n_experts, const moe_devremap_jobs jobs) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_used * n_tok) {
        int32_t e = ids[(int64_t) (i / n_used) * ids_s1 + i % n_used];
        e = e >= 0 && e < n_experts ? e : 0;   // router ids are always in range; never index outside the map
        jobs.remap[blockIdx.y][i] = jobs.map[blockIdx.y][e];
        jobs.lag[blockIdx.y][i]   = e;
    }
    // single-block jobs only (the host sets seq only then): every lag write of the job is in this block
    if (jobs.seq[blockIdx.y] != nullptr) {
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence_system();
#if defined(GGML_USE_HIP)
            __hip_atomic_store(jobs.seq[blockIdx.y], jobs.seq_val[blockIdx.y], __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
            atomicExch(jobs.seq[blockIdx.y], jobs.seq_val[blockIdx.y]);
#endif
        }
    }
}

static bool g_devremap = [] {
    const char * e = getenv("GGML_MOE_DEVREMAP");
    return e == nullptr || atoi(e) != 0;
}();
static bool g_devremap_batch = [] {
    const char * e = getenv("GGML_MOE_DEVREMAP_BATCH");
    return e == nullptr || atoi(e) != 0;
}();
static int64_t g_devremap_calls = 0;
static int64_t g_devremap_offers = 0;   // hosted-expert inputs offered to the remap, and how many it declined
static int64_t g_devremap_declined = 0;

struct devremap_batch_t {
    const void *     ids    = nullptr;
    int64_t          s1     = 0;
    int64_t          n_used = 0;
    int64_t          n_tok  = 0;
    std::vector<int> pending;   // table ids already served by this layer's first call
};
static std::map<int, devremap_batch_t> g_dr_batch;

static void devremap_batch_clear_locked() {
    g_dr_batch.clear();
}


static bool devremap_alloc_locked(table_t & t) {
    device_guard dg(t.device);
    t.dr_device = t.device;
    t.dr_cap    = t.remap_cap;
    const size_t mb = (size_t) t.n_experts * sizeof(int32_t);
    const size_t lb = (size_t) (t.dr_cap + 1) * sizeof(int32_t);   // + the lag completion flag
    if (cudaMalloc((void **) &t.dr_map_dev, mb) != cudaSuccess ||
        cudaHostAlloc((void **) &t.dr_map_host, mb, cudaHostAllocPortable) != cudaSuccess ||
        cudaHostAlloc((void **) &t.dr_lag_host, lb, cudaHostAllocPortable | cudaHostAllocMapped) != cudaSuccess ||
        cudaHostGetDevicePointer((void **) &t.dr_lag_dev, t.dr_lag_host, 0) != cudaSuccess ||
        cudaEventCreateWithFlags(&t.dr_map_ev, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&t.dr_lag_ev, cudaEventDisableTiming) != cudaSuccess) {
        (void) cudaGetLastError();
        devremap_free_locked(t);
        t.dr_failed = true;
        GGML_LOG_WARN("%s: device remap buffers failed for layer=%d role=%s: the eager path serves it\n",
                      __func__, t.layer, t.role.c_str());
        return false;
    }
    t.dr_map_ok  = false;
    t.dr_lag_n   = 0;
    t.dr_lag_host[t.dr_cap] = 0;
    t.dr_lag_seq  = 0;
    t.dr_lag_flag = false;
    t.dr_last_ids    = nullptr;
    t.dr_last_stream = nullptr;
    return true;
}

// The remap's COLD SOURCE for a table: a pinned master (`table_cold_ok`, cold id = `slots + e`) or a host
// pool that covers the tensor (cold id = `slots + pool slot`).  Both give every expert a valid cold id, which
// is what a device-side map needs: it is built once per residency change and then read blindly by the kernel.
static bool devremap_cold_ok_locked(const table_t & t) {
    return table_cold_ok(t) || t.dr_pool_ok;
}

// Constant for a table and shape once the cache is sized (a graph replay needs a constant take-over decision).
static bool devremap_ok_locked(const table_t & t, int64_t n_used, int64_t n_tok, int device) {
    return g_sized && !t.dr_failed && !t.fill_failed && devremap_cold_ok_locked(t) && t.n_slices == 0 && !t.identity &&
           t.arena != nullptr && t.slots > 0 && t.device == device && t.n_experts > 0 &&
           n_used < t.n_experts && t.remap_dev != nullptr && n_used * n_tok <= t.remap_cap &&
           (t.dr_device < 0 || (t.dr_device == device && t.dr_cap == t.remap_cap));
}

// The one-call-late policy and the map upload for one table.  False: decline (the eager path serves the input).
static bool devremap_prep_locked(table_t & t, void * stream) {
    if (t.dr_device < 0 && !devremap_alloc_locked(t)) {
        return false;
    }
    const cudaStream_t s = (cudaStream_t) stream;
    // 1. The residency policy on the previous call's routing (never block: not back yet = skip this call's policy).
    if (t.dr_lag_n > 0) {
        const bool lag_ready = t.dr_lag_flag ? __atomic_load_n(&t.dr_lag_host[t.dr_cap], __ATOMIC_ACQUIRE) == t.dr_lag_seq
                                             : cudaEventQuery(t.dr_lag_ev) == cudaSuccess;
        if (!lag_ready) {
            (void) cudaGetLastError();
            t.dr_lag_n = 0;
        }
    }
    if (t.dr_lag_n > 0) {
        const int n = (int) t.dr_lag_n;
        t.dr_lag_n = 0;
        if (g_heat_on) {   // the decode-band tally the eager path keeps, one call late
            if ((int) t.heat.size() != t.n_experts)     { t.heat.assign((size_t) t.n_experts, 0); }
            if ((int) t.heat_dec.size() != t.n_experts) { t.heat_dec.assign((size_t) t.n_experts, 0); }
            for (int i = 0; i < n; i++) {
                const int32_t e = t.dr_lag_host[i];
                if (e >= 0 && e < t.n_experts) {
                    if (t.heat[(size_t) e] != 0xffffffffu)     { t.heat[(size_t) e]++;     g_heat_dirty = true; }
                    if (t.heat_dec[(size_t) e] != 0xffffffffu) { t.heat_dec[(size_t) e]++; g_heat_dirty = true; }
                }
            }
        }
        for (int i = 0; i < n; i++) {
            bool cold = false;
            (void) access_locked(t, t.dr_lag_host[i], stream, t.dr_lag_host, n, &cold);
        }
        if (t.arena == nullptr || t.slots <= 0) {
            return false;
        }
    }
    // 2. The device map, re-uploaded only after a residency change.
    if (t.slot_dirty || !t.dr_map_ok) {
        if (cudaEventQuery(t.dr_map_ev) != cudaSuccess) {   // the previous upload still reads the staging: decline
            (void) cudaGetLastError();
            return false;
        }
        const bool pinned = table_cold_ok(t);
        host_pool_t * pool = nullptr;
        if (!pinned) {
            // A covering host pool: the cold id is this expert's POOL slot, which is a host-side fact (the
            // policy runs here, on the host, one call late) while the arena slot would be per device.  Every
            // expert must have a live slot -- `t.dr_pool_ok` decided that at sizing, and a pool that has
            // since lost one (impossible for a covering pool after sizing) declines rather than emit an id
            // the kernel would resolve against the wrong bytes.
            pool = table_pool_peek_locked(t);
            if (pool == nullptr || !pool_cold_geometry_ok(t, pool) || pool->slots < t.n_experts) {
                t.dr_map_ok = false;
                return false;
            }
        }
        for (int e = 0; e < t.n_experts; e++) {
            t.dr_map_host[e] = t.slots + e;
        }
        if (pool != nullptr) {
            int n_cold = 0;
            for (int ps = 0; ps < pool->slots && ps < (int) pool->slot_expert.size(); ps++) {
                const int32_t e = pool->slot_expert[(size_t) ps];
                if (e >= 0 && e < t.n_experts && pool->slot_state[(size_t) ps] == 1) {
                    t.dr_map_host[e] = t.slots + ps;
                    n_cold++;
                }
            }
            if (n_cold < t.n_experts) {
                // Not every expert is pool-resident after all: the map would have a hole and the kernel
                // would read the pool at `expert * stride`.  Decline (the eager path serves the input).
                static bool warned = false;
                if (!warned) {
                    warned = true;
                    GGML_LOG_WARN("%s: layer=%d role=%s: the host pool no longer covers the table (%d of %d "
                                  "experts resident); the device remap stands down for it\n",
                                  __func__, t.layer, t.role.c_str(), n_cold, t.n_experts);
                }
                t.dr_map_ok = false;
                return false;
            }
        }
        for (int sl = 0; sl < t.slots && sl < (int) t.slot_expert.size(); sl++) {
            const int32_t e = t.slot_expert[sl];
            if (e >= 0 && e < t.n_experts) {
                t.dr_map_host[e] = sl;
            }
        }
        if (cudaMemcpyAsync(t.dr_map_dev, t.dr_map_host, (size_t) t.n_experts * sizeof(int32_t),
                            cudaMemcpyHostToDevice, s) != cudaSuccess) {
            (void) cudaGetLastError();
            t.dr_map_ok = false;
            return false;
        }
        (void) cudaEventRecord(t.dr_map_ev, s);
        t.slot_dirty = false;
        t.dr_map_ok  = true;
    }
    return true;
}

bool moe_cache_remap_dev(const ggml_tensor * weight, const ggml_tensor * weight_cpy, const ggml_tensor * ids,
                         void * stream, int device, bool check_only) {
    if (!g_enabled || !g_devremap || weight == nullptr || weight_cpy == nullptr || ids == nullptr ||
        ids->data == nullptr || stream == nullptr || ids->type != GGML_TYPE_I32) {
        return false;
    }
    const int64_t n_used = ids->ne[0];
    const int64_t n_tok  = ids->ne[1];
    if (n_used <= 0 || n_tok <= 0 || n_tok > MOE_EXPERT_CACHE_MAX_TOK ||
        ids->nb[0] != sizeof(int32_t) || ids->nb[1] % sizeof(int32_t) != 0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    const auto it = g_dr_key_to_id.find({ (const void *) weight, device });
    if (it == g_dr_key_to_id.end() || it->second < 0 || it->second >= (int) g_tables.size()) {
        return false;   // not seen by the eager path yet (priming / sizing run there)
    }
    const int id = it->second;
    table_t & t = g_tables[id];
    const bool ok = devremap_ok_locked(t, n_used, n_tok, device);
    if (check_only) {
        g_devremap_offers++;
        if (!ok) {
            g_devremap_declined++;
        }
        return ok;
    }
    g_write_seq++;
    const auto decline = [&]() {
        t.remap_n_used = 0;
        t.dr_lag_n     = 0;
        return false;
    };
    const int64_t n  = n_used * n_tok;
    const int64_t s1 = (int64_t) (ids->nb[1] / sizeof(int32_t));
    const auto publish = [&]() {
        t.dr_lag_n     = n;
        t.remap_n_used = n_used;
        t.remap_n_tok  = n_tok;
        g_alias_to_id[weight_cpy] = id;
        t.dr_last_ids    = ids->data;
        t.dr_last_stream = stream;
        g_takeover++;
        g_devremap_calls++;
        return true;
    };
    // A sibling of this layer's first call with the same routing: its remap was already written in that launch.
    devremap_batch_t & b = g_dr_batch[device];
    const auto pit    = std::find(b.pending.begin(), b.pending.end(), id);
    const bool served = pit != b.pending.end() && b.ids == ids->data && b.s1 == s1 && b.n_used == n_used &&
                        b.n_tok == n_tok;
    if (served) {
        b.pending.erase(pit);
    } else {
        b.pending.clear();
    }
    if (!ok) {
        return decline();
    }
    if (served) {
        return publish();
    }
    device_guard       dg(t.device);
    const cudaStream_t s = (cudaStream_t) stream;
    if (!devremap_prep_locked(t, stream)) {
        return decline();
    }
    moe_devremap_jobs jobs = {};
    int               job_id[MOE_DEVREMAP_MAX_JOBS];
    int               n_jobs = 0;
    const auto add_job = [&](table_t & u, int uid) {
        jobs.map[n_jobs]   = u.dr_map_dev;
        jobs.remap[n_jobs] = u.remap_dev;
        jobs.lag[n_jobs]   = u.dr_lag_dev;
        job_id[n_jobs++]   = uid;
    };
    add_job(t, id);
    // The layer's other tables on this device read the same routing (gate / up / down): serve them in this launch.
    if (g_devremap_batch && t.layer >= 0) {
        for (int k = 0; k < (int) g_tables.size() && n_jobs < MOE_DEVREMAP_MAX_JOBS; k++) {
            table_t & u = g_tables[k];
            if (k == id || u.layer != t.layer || u.device != device || u.n_experts != t.n_experts ||
                u.dr_last_ids != ids->data || u.dr_last_stream != stream ||
                !devremap_ok_locked(u, n_used, n_tok, device)) {
                continue;
            }
            if (!devremap_prep_locked(u, stream)) {
                u.remap_n_used = 0;
                u.dr_lag_n     = 0;
                continue;   // its own call retries
            }
            add_job(u, k);
        }
    }
    // Routing -> slot ids; the kernel also writes the routing into the mapped pinned lag buffer for the next call's
    // policy, with a completion flag (one block per job) or, for wider batches, an event per table.
    const bool use_flag = n <= 256;
    for (int j = 0; j < n_jobs; j++) {
        table_t & u = g_tables[job_id[j]];
        u.dr_lag_flag   = use_flag;
        jobs.seq[j]     = use_flag ? u.dr_lag_dev + u.dr_cap : nullptr;
        jobs.seq_val[j] = use_flag ? ++u.dr_lag_seq : 0;
    }
    moe_cache_remap_kernel<<<dim3((unsigned) ((n + 255) / 256), (unsigned) n_jobs), 256, 0, s>>>(
            (const int32_t *) ids->data, s1, (int) n_used, (int) n_tok, t.n_experts, jobs);
    if (cudaGetLastError() != cudaSuccess) {
        return decline();
    }
    if (!use_flag) {
        for (int j = 0; j < n_jobs; j++) {
            (void) cudaEventRecord(g_tables[job_id[j]].dr_lag_ev, s);
        }
    }
    b.ids    = ids->data;
    b.s1     = s1;
    b.n_used = n_used;
    b.n_tok  = n_tok;
    b.pending.assign(job_id + 1, job_id + n_jobs);
    return publish();
}

bool moe_cache_get_cold(const void * arena, void ** cold_base, int64_t * n_res,
                        size_t * cold_channel_bytes, size_t * cold_row_bytes) {
    if (!g_enabled || !g_cold_uva || arena == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    for (table_t & t : g_tables) {
        if (t.arena == arena && t.slots > 0 && t.host_dev == nullptr) {
            // MOE_HOST_POOL_COLD: a pool-backed table's cold region is the pinned pool, indexed by pool slot.
            // `pool_cold_geometry_ok` is the SAME predicate the host-side decision uses, and the slice offset
            // is folded in exactly as the master path does below.
            host_pool_t * p = table_pool_locked(t);
            if (pool_cold_geometry_ok(t, p)) {
                *cold_base = (char *) p->base_dev + t.src_off;
                *n_res     = t.slots;
                if (cold_channel_bytes) {
                    *cold_channel_bytes = p->host_bytes;
                }
                if (cold_row_bytes) {
                    *cold_row_bytes = t.host_pitch;
                }
                return true;
            }
        }
        if (t.arena != arena || t.slots <= 0 || t.host_dev == nullptr || !t.cold_safe || t.n_slices > 0) {
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



bool moe_cache_get_table(const ggml_tensor * op, const ggml_tensor * weight_cpy, int device,
                         void ** arena, int64_t * n_slots,
                         size_t * expert_bytes, void ** remap_dev, int64_t * n_used, int64_t * n_tok) {
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
    if (t.fill_failed) {
        return false;   // issue #274: never hand out the arena of a table whose fills have failed
    }
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

// Per-table fusion-guard query (wip/moe-verify-fusions OPEN 2).  Mirrors `moe_cache_take_over`'s
// `identity` decision for THIS table, with no side effects, so the fusion guard and the scheduler
// take-over agree per table instead of on the whole-graph `moe_cache_has_arena()`.
bool moe_cache_table_serves(const ggml_tensor * op, const ggml_tensor * weight, int device) {
    if (!g_enabled || weight == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    int id = alias_find_checked(weight, device, op != nullptr ? name_layer(op) : -1);
    if (id < 0 && op != nullptr && weight->buffer != nullptr) {
        const int layer = name_layer(op);
        const auto sit = g_sem_to_id.find(std::make_tuple(layer, sem_role(weight), device));
        if (sit != g_sem_to_id.end()) {
            id = sit->second;
        }
    }
    if (id < 0 || id >= (int) g_tables.size()) {
        return false;
    }
    const table_t & t = g_tables[id];
    return t.identity && t.arena != nullptr && t.slots >= t.n_experts;
}

bool moe_cache_take_over(const ggml_tensor * weight, const ggml_tensor * weight_cpy, int device) {
    g_write_seq++;
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
    // OPEN 2 slab: gate per TABLE.  The `identity` check below already requires THIS table's arena
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
    if (!identity) {
        return false;
    }
    // Overwrite: the graph allocator reuses these device pointers across graphs (see
    // `moe_cache_update_host`), so the alias must track the current tensor.
    g_alias_to_id[weight_cpy] = id;
    return true;
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
    if (!moe_cache_get_table(op, src0, device, &arena, &ns, &eb, &remap, &nu, &nt)) {
        return false;
    }
    // Resolve the gate lane first so a shape mismatch is caught before either lane is redirected.
    void *    gate_arena = nullptr;
    void *    gr         = nullptr;
    bool      gate_ok    = false;
    int64_t   gnu = 0, gnt = 0;
    if (gate != nullptr) {
        void *  ga = nullptr;
        int64_t gs = 0;
        size_t  geb = 0;
        gate_ok = moe_cache_get_table(op, gate, device, &ga, &gs, &geb, &gr, &gnu, &gnt);
        if (!gate_ok) {
            return false;   // never leave one lane redirected and the other reading a stale copy
        }
        gate_arena = ga;
    }
    // remap == nullptr is the identity fast path: the raw routing ids already index the arena.
    if (remap != nullptr && (nu != ids->ne[0] || nt != ids->ne[1])) {
        return false;
    }
    // A gate remap must match the up shape (both lanes read the same routing).
    if (gate_ok && gr != nullptr && (gnu != nu || gnt != nt)) {
        return false;   // never leave one lane redirected and the other reading a stale copy
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


void moe_cache_init(int device) {
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        ensure_env();
        if (!g_init_done) {
            g_init_done = true;
            // r233 (issue #278): a run-time kill switch for the pinned tier.  Setting `g_pinned_off` here is
            // the same latch every pinned consumer already honours, so no new code path decides "closed".
            // Applied even when the cache is disabled, so a later enable cannot outrun it.
            if (pinned_tier_disabled()) {
                g_pinned_off     = true;
                g_pinned_off_why = "MOE_HOST_POOL_PIN=0";
            }
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
#if defined(GGML_USE_HIP) && defined(_WIN32)
            {
                // r233 (issue #278): name the HIP runtime that is really loaded.  The driver's System32 copy
                // and the ROCm/TheRock runtime behave differently enough on the pinned tier that a report
                // without this line is ambiguous (see `ggml_hip_runtime_module_info`).
                char path[512]     = {};
                bool from_system   = false;
                int  hip_ver       = 0;
                if (ggml_hip_runtime_module_info(path, sizeof(path), &from_system, &hip_ver)) {
                    char ver[32] = "unknown";
                    if (hip_ver > 0) {
                        snprintf(ver, sizeof(ver), "%d.%d.%d",
                                 hip_ver / 10000000, (hip_ver / 100000) % 100, hip_ver % 100000);
                    }
                    GGML_LOG_INFO("%s: HIP runtime %s from %s\n", __func__, ver, path);
                    if (from_system) {
                        GGML_LOG_WARN("%s: the loaded HIP runtime is the Windows display driver's copy (%s), "
                                      "not the ROCm/TheRock runtime shipped beside the executable.  On this "
                                      "runtime a pinned host allocation is refused at ~9.3 GiB of adapter "
                                      "shared (aperture) usage even on a 64 GiB box (issues #274/#278), so "
                                      "the host pool is capped and the pinned tier latches off.  The same "
                                      "binary with the TheRock `amdhip64_7.dll` beside it pins 16-40 GiB and "
                                      "runs the same arm clean.  Put the ROCm/TheRock runtime DLLs next to "
                                      "llama-server.exe.  To stay on this runtime and still run the "
                                      "host-expert path, set MOE_HOST_POOL_PIN=0 (pinned tier off; correct, "
                                      "slower).\n", __func__, path);
                    }
                }
            }
#endif
            GGML_LOG_INFO("%s: MoE expert cache enabled on device %d: budget=%s MiB/device, period=%lld steps, %s\n",
                          __func__, device, budget_desc(), (long long) g_period,
                          g_slots_hint > 0 ? "explicit slots/table" : "adaptive uniform slots (deferred sizing)");
        }
    }
}

// The whole exit census, printed under `g_mutex` by whichever teardown gets there first: the first
// `moe_cache_forget_all` (the normal path, and the only one whose log an operator still sees on Windows) or
// the `atexit` hook below (a run with no teardown: `--help`, a signal, a crash handler).
static void report_locked() {
    if (!g_report_snap.have) {
        report_snapshot_locked();   // no teardown yet (`--help`, a crash handler, a server shutdown)
    }
    const int64_t hits      = g_report_snap.hits;
    const int64_t misses    = g_report_snap.misses;
    const int64_t fills     = g_report_snap.fills;
    const int64_t evictions = g_report_snap.evictions;
    const int     n_tables  = g_report_snap.n_tables;
    const int     n_slots   = g_report_snap.n_slots;
    const int64_t arena_b   = g_report_snap.arena_bytes;
    const int64_t * dev_slots  = g_report_snap.dev_slots;
    const int64_t * dev_arena  = g_report_snap.dev_arena;
    const int     * dev_tables = g_report_snap.dev_tables;
    const int64_t total = hits + misses;
    const double h = total > 0 ? (double) hits / (double) total : 0.0;
    GGML_LOG_INFO("%s: MoE expert cache: h=%.4f (%lld/%lld reaches), fills=%lld evictions=%lld, tables=%d, slots=%d, arena=%.1f MiB total (budget %s MiB/device)\n",
                  __func__, h, (long long) hits, (long long) total, (long long) fills, (long long) evictions,
                  n_tables, n_slots, (double) arena_b / (1024 * 1024), budget_desc());
    if (g_devremap_offers > 0 || g_devremap_calls > 0) {
        GGML_LOG_INFO("%s: device remap: %lld tables served, %lld of %lld offers declined\n", __func__,
                      (long long) g_devremap_calls, (long long) g_devremap_declined, (long long) g_devremap_offers);
    }
    if (g_fill_read_calls + g_fill_memcpy_calls > 0) {
        const double tb = (double) g_fill_total_bytes / (1024.0 * 1024.0 * 1024.0);
        GGML_LOG_INFO("%s: pool fills: %lld storage reads %.0f ms, %lld mapping memcpys %.0f ms, %.2f GiB total "
                      "(%.1f MiB/fill, %.0f MiB/s effective)\n",
                      __func__, (long long) g_fill_read_calls, g_fill_read_us / 1000.0,
                      (long long) g_fill_memcpy_calls, g_fill_memcpy_us / 1000.0, tb,
                      tb * 1024.0 / (double) (g_fill_read_calls + g_fill_memcpy_calls),
                      tb * 1024.0 / (double) std::max<int64_t>((g_fill_read_us + g_fill_memcpy_us) / 1000000, 1));
    }
    if (g_uploadp_calls > 0) {
        GGML_LOG_INFO("%s: scheduler pageable-master fallback: %lld ranges, %.1f MiB served through the per-table "
                      "bounce; pinned bounce ring total %.1f MiB\n",
                      __func__, (long long) g_uploadp_calls, (double) g_uploadp_bytes / (1024 * 1024),
                      (double) g_bounce_bytes / (1024 * 1024));
    }
    if (g_gather_calls > 0) {
        GGML_LOG_INFO("%s: device gather: %lld calls (%lld pool-backed), %lld routed experts copied into the "
                      "destination from the bounce/pool-fill path (neither tier)\n",
                      __func__, (long long) g_gather_calls, (long long) g_gather_pool, (long long) g_gather_fills);
    }
    if (g_pool_enabled) {
        const int64_t p_slots  = g_report_snap.p_slots;
        const int64_t p_hits   = g_report_snap.p_hits;
        const int64_t p_misses = g_report_snap.p_misses;
        const int64_t p_fills  = g_report_snap.p_fills;
        const int64_t p_evict  = g_report_snap.p_evict;
        const int64_t p_bytes  = g_report_snap.p_bytes;
        const int64_t ptot = p_hits + p_misses;
        // WARN, not INFO: llama-cli's default threshold filters INFO, so the pool summary was invisible on a
        // plain run (it needs `-lv 4`/`-v`).  The pool is a tuning surface, so its summary must be visible.
        GGML_LOG_WARN("%s: host pool (L2): %zu tensors, %lld slots, %.1f MiB pinned; h=%.4f (%lld/%lld), "
                      "fills=%lld evictions=%lld\n",
                      __func__, (size_t) g_report_snap.n_pools, (long long) p_slots, (double) p_bytes / (1024 * 1024),
                      ptot > 0 ? (double) p_hits / (double) ptot : 0.0, (long long) p_hits, (long long) ptot,
                      (long long) p_fills, (long long) p_evict);
    }
    if (g_pool_sink) {
        GGML_LOG_WARN("%s: sink (MOE_HOST_POOL_SINK): %lld demoted, %.1f MiB | declined: %lld no-free-slot, "
                      "%lld geometry (structural), %lld copy-error | %lld promoter slots retired (spare=%d)\n",
                      __func__, (long long) g_sink_copies, (double) g_sink_bytes / (1024 * 1024),
                      (long long) g_sink_decl_noslot, (long long) g_sink_decl_geom,
                      (long long) g_sink_decl_copy, (long long) g_sink_retire,
                      g_pool_sink_spare >= 0 ? g_pool_sink_spare : g_pool_sink_spare_pct);
        GGML_LOG_WARN("%s: sink reclaimed %lld saturated slots (evict duplicate/LRU instead of declining)\n",
                      __func__, (long long) g_sink_evicted);
        GGML_LOG_WARN("%s: sink headroom: %lld prefill fills took the exchange slots (a prefill never honours the "
                      "reserve; TODO #281)\n", __func__, (long long) g_sink_prefill_fills);
        GGML_LOG_WARN("%s: sink reuse: %lld uses against %lld demotions (%.1f %% used; W1 break-even ~3 %% "
                      "on NVMe, ~0.2 %% on slow storage)\n", __func__, (long long) g_sink_uses,
                      (long long) g_sink_copies,
                      g_sink_copies > 0 ? 100.0 * (double) g_sink_uses / (double) g_sink_copies : 0.0);
        GGML_LOG_WARN("%s: sink mirror tables (full coverage -> exchange INERT, spare=0): %lld\n", __func__,
                      (long long) g_sink_mirror_tables);
        if (g_sink_near_mirror_tables > 0) {
            GGML_LOG_WARN("%s: sink near-mirror tables (coverage gap < 4 %% of the set -> exchange DISENGAGED): "
                          "%lld\n", __func__, (long long) g_sink_near_mirror_tables);
        }
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
    // wip/moe-heatmap-persistence (R2): flush the device prefill tally into the host heat vectors and
    // report a concentration summary, so a run shows whether the tally is populated and how peaked it
    // is (the placement/persistence consumers need a peaked prior).
    if (g_heat_on) {
        heat_pull_locked();
        // TODO #61 P3 (metric parity): `h` alone cannot say how much of the routing distribution the arena
        // actually covers, and that is the quantity every coverage question is about.  Report, on THIS
        // SESSION's routing (`heat_delta`, i.e. `heat` minus the part already folded into the persisted
        // prior), the mass the resident set holds and the mass a GLOBAL-top-f resident set of the same slot
        // count would hold (the static bound, i.e. what a perfect static placement buys).  Measured BEFORE
        // the save below, which folds the delta away.  The `expert traffic per token` line further down
        // completes the pair the campaign's P3 asks for.
        {
            int64_t  n_slots       = 0;
            uint64_t total_mass    = 0;
            uint64_t resident_mass = 0;
            int64_t  n_resident    = 0;
            int64_t  n_expert_sl   = 0;
            std::vector<uint64_t> all_heat;
            for (const table_t & t : g_tables) {
                if ((int) t.heat.size() != t.n_experts) {
                    continue;
                }
                n_slots     += t.slots;
                n_expert_sl += t.n_experts;
                n_resident  += (int64_t) t.expert_slot.size();
                for (int e = 0; e < t.n_experts; e++) {
                    const uint64_t v = heat_delta(t, (size_t) e);
                    all_heat.push_back(v);
                    total_mass += v;
                    if (t.expert_slot.find(e) != t.expert_slot.end()) {
                        resident_mass += v;
                    }
                }
            }
            if (!all_heat.empty() && total_mass > 0) {
                std::sort(all_heat.begin(), all_heat.end(),
                          [](uint64_t a, uint64_t b) { return a > b; });
                const int64_t k = moe_cache_min_i64(n_slots, (int64_t) all_heat.size());
                uint64_t bound = 0;
                for (int64_t i = 0; i < k; i++) {
                    bound += all_heat[(size_t) i];
                }
                GGML_LOG_INFO("%s: heat coverage (this session): the resident set holds %.2f%% of the routed "
                              "mass; a global top-%lld placement of the same slots would hold %.2f%% "
                              "(static bound); resident %lld of %lld expert-slots (summed over tables/devices)\n",
                              __func__, 100.0 * (double) resident_mass / (double) total_mass,
                              (long long) k, 100.0 * (double) bound / (double) total_mass,
                              (long long) n_resident, (long long) n_expert_sl);
            }
        }
        heat_map_save_locked();   // final flush at process exit (the debounce must not lose the run)
        g_heat_dirty = false;     // the tallies are persisted; a later teardown flush is a no-op
        uint64_t total   = 0;
        int64_t  nonzero = 0, experts = 0;
        struct heat_top_t { uint64_t v; int layer; const char * role; int e; };
        std::vector<heat_top_t> top(5, { 0ull, -1, "", -1 });
        for (const table_t & t : g_tables) {
            if ((int) t.heat.size() != t.n_experts) {
                continue;
            }
            for (int e = 0; e < t.n_experts; e++) {
                const uint64_t v = t.heat[(size_t) e];
                total += v;
                experts++;
                if (v > 0) {
                    nonzero++;
                }
                if (v > top.back().v) {
                    top.back() = { v, t.layer, t.role.c_str(), e };
                    std::sort(top.begin(), top.end(),
                              [](const heat_top_t & a, const heat_top_t & b) { return a.v > b.v; });
                }
            }
        }
        GGML_LOG_INFO("%s: heat (R2): total=%llu (persisted prior + this session), nonzero=%lld/%lld experts "
                      "over %zu tables\n",
                      __func__, (unsigned long long) total, (long long) nonzero, (long long) experts,
                      g_tables.size());
        for (const heat_top_t & h : top) {
            if (h.v > 0) {
                GGML_LOG_INFO("%s: heat top: layer=%d role=%s expert=%d count=%llu\n",
                              __func__, h.layer, h.role, h.e, (unsigned long long) h.v);
            }
        }
    }
    // TODO #61 P1: flush (and close) the routing trace so a run always leaves a complete file.
    if (g_trace_on && g_trace_file != nullptr) {
        fflush(g_trace_file);
        fclose(g_trace_file);
        g_trace_file = nullptr;
        GGML_LOG_INFO("%s: routing trace: %lld records, %.1f MiB -> %s\n",
                      __func__, (long long) g_trace_records, (double) g_trace_bytes / (1024 * 1024),
                      g_trace_path.c_str());
    }

    if (!g_stage_declined.empty()) {
        // TODO #271: ONE line naming every table the prefill staging path declined, with the reason, plus
        // the per-table detail at INFO.  A WARN because it is a decision that changes which transport
        // carries the tier's prefill (the same level `report_host_expert_placement` uses for the same class
        // of decision) -- and once per RUN, because a declined table declines on every ubatch.
        const char * first_why = g_stage_declined.front().why.c_str();
        bool same_why = true;
        for (const stage_decline_t & d : g_stage_declined) {
            same_why = same_why && d.why == first_why;
        }
        // A tensor split registers one table per device, so the same layer/role can appear N times.  The
        // summary names each shape once (the reason is a property of the shape, not of the device); the
        // per-table INFO lines below carry the device.
        std::vector<std::pair<int, std::string>> named;
        for (const stage_decline_t & d : g_stage_declined) {
            if (std::find(named.begin(), named.end(), std::make_pair(d.layer, d.role)) == named.end()) {
                named.push_back({ d.layer, d.role });
            }
        }
        std::string tables;
        for (size_t i = 0; i < named.size() && i < 12; i++) {
            char buf[160];
            snprintf(buf, sizeof(buf), "%s%d/%s", i ? ", " : "", named[i].first, named[i].second.c_str());
            tables += buf;
        }
        if (named.size() > 12) {
            char buf[64];
            snprintf(buf, sizeof(buf), ", +%zu more (see the per-table lines below)", named.size() - 12);
            tables += buf;
        }
        GGML_LOG_WARN("%s: prefill staging declined %zu of %zu host table(s): %s%s.  Affected: %s.  They are "
                      "served by the scheduler's own whole-table upload instead (not staged, so their bytes "
                      "are missing from the staging census above); a stood-down one is re-armed at the "
                      "compute-buffer drop, so this is a prefill-only state\n",
                      __func__, g_stage_declined.size(), g_tables.size(),
                      same_why ? first_why : "several reasons --", same_why ? "" : " each listed per table below",
                      tables.c_str());
        for (const stage_decline_t & d : g_stage_declined) {
            GGML_LOG_INFO("%s: prefill staging declined: layer=%d role=%s dev=%d: %s\n", __func__, d.layer,
                          d.role.c_str(), d.device, d.why.c_str());
        }
    }
    if (g_stage_ok + g_stage_decline_geom + g_stage_decline_usable + g_stage_decline_nomatch > 0) {
        const double total = (double) g_stage_arena_bytes + (double) g_stage_host_bytes;
        GGML_LOG_INFO("%s: prefill cache staging: ok=%lld geom_decline=%lld unusable=%lld nomatch=%lld "
                      "unsized=%lld (the priming pass); arena=%.1f MiB host=%.1f MiB (H2D %.1f%% of staged); "
                      "PRS rebuilds=%lld\n",
                      __func__, (long long) g_stage_ok, (long long) g_stage_decline_geom,
                      (long long) g_stage_decline_usable, (long long) g_stage_decline_nomatch,
                      (long long) g_stage_decline_unsized,
                      (double) g_stage_arena_bytes / (1024 * 1024), (double) g_stage_host_bytes / (1024 * 1024),
                      total > 0 ? 100.0 * (double) g_stage_host_bytes / total : 0.0,
                      (long long) g_prs_rebuilds);
    }
    if (g_stage_span_reads + g_stage_span_fallback > 0) {
        // W4: the coalesced prefill read.  `MiB/read` is the chunk the run structure actually allowed -- if
        // it sits near 0.7 MiB the missing experts are too fragmented for a span to help (the pool covers
        // most of the table) and this is the per-expert shape with extra overhead.  `read` is the time inside
        // the pread, `wait` the time the two-deep ring had to wait for a previous chunk's copies to drain.
        GGML_LOG_INFO("%s: coalesced prefill reads (%s): %lld reads, %.1f MiB (%.2f MiB/read), %lld experts, "
                      "%lld us waiting for a ring buffer, fallbacks=%lld\n",
                      __func__, g_pool_dio ? "O_DIRECT" : "buffered pread",
                      (long long) g_stage_span_reads,
                      (double) g_stage_span_bytes / (1024 * 1024),
                      g_stage_span_reads > 0
                          ? (double) g_stage_span_bytes / 1048576.0 / (double) g_stage_span_reads : 0.0,
                      (long long) g_stage_span_experts, (long long) g_stage_span_wait_us,
                      (long long) g_stage_span_fallback);
        if (g_stage_span_bytes > 0) {
            GGML_LOG_INFO("%s: coalesced read rate: %.1f MiB/s over %.0f ms of actual reads\n", __func__,
                          (double) g_stage_span_bytes / 1048576.0 / ((double) g_stage_span_read_us / 1e6),
                          (double) g_stage_span_read_us / 1000.0);
        }
    }
    if (g_bounce_read_calls > 0) {
        // The prefill band's storage reads when they are NOT coalesced (a chunk that fell back, the 2-D
        // geometry, a pool with no free slot yet).  Read it against the span line above: `%.1f MiB/read`
        // near 0.7 MiB is the latency-bound shape this campaign set out to remove.
        GGML_LOG_INFO("%s: per-expert storage reads (bounce): %lld reads, %.1f MiB, %.0f ms, %.1f MiB/s, "
                      "%.2f MiB/read\n",
                      __func__, (long long) g_bounce_read_calls,
                      (double) g_bounce_read_bytes / (1024 * 1024), g_bounce_read_us / 1000.0,
                      g_bounce_read_us > 0
                          ? (double) g_bounce_read_bytes / 1048576.0 / ((double) g_bounce_read_us / 1e6) : 0.0,
                      (double) g_bounce_read_bytes / 1048576.0 / (double) g_bounce_read_calls);
    }
    GGML_LOG_INFO("%s: VRAM budget: requested=%s MiB/device, free-clamped=%lld, arena alloc failures=%lld (fail-soft)\n",
                  __func__, budget_desc(), (long long) g_alloc_clamped, (long long) g_alloc_failed);
    if (g_fill_fail > 0) {
        // issue #274: a failed fill rolls the admission back and declines the op, so the run is CORRECT but
        // slow.  Silence here is what let "every fill failed" reach the reporter as broken output instead of
        // a number: say it in the census, once, at the level the operator sees.
        GGML_LOG_WARN("%s: FAILED fills: %lld (each one was rolled back -- the slot is not resident and the "
                      "routed op was declined, so the scheduler served it from the host master)\n",
                      __func__, (long long) g_fill_fail);
    }
    if (g_pinned_tier_bytes > 0) {
        // How far the pinned tier got, whether or not it was refused: on Windows this is the number that names
        // the platform ceiling (the reporter's stops at ~9.3 GiB of adapter shared usage on both a 47 and a
        // 64 GiB box), and on POSIX it is how much of RAM the pool/ring/span tiers took.
        GGML_LOG_INFO("%s: pinned tier: %.1f MiB allocated in total (host pool + bounce rings + coalesced span "
                      "ring)%s\n", __func__, (double) g_pinned_tier_bytes / (1024.0 * 1024.0),
                      g_pinned_off ? (g_pinned_off_why != nullptr ? " -- the tier is closed, see the line below"
                                                                 : " -- the tier was refused after this, see the line below")
                                   : "");
    }
    if (g_pinned_off) {
        // issue #274 follow-up (@DanoPTT): the pinned tier is closed for the run, so the tier is being served
        // from the pageable mapping.  Say WHY -- a silent fallback reads as "the pool never appeared" all over
        // again.  A `MOE_HOST_POOL_PIN=0` run (r233) closed it deliberately; a refusal on Windows is the
        // platform's aperture budget, which the driver-shipped HIP runtime reaches where TheRock's does not
        // (issue #278).
        if (g_pinned_off_why != nullptr) {
            GGML_LOG_WARN("%s: the pinned host tier is OFF for this run (%s); every fill that needed it went "
                          "through the model mapping/pageable path or declined (correct, slower)\n",
                          __func__, g_pinned_off_why);
        } else {
            GGML_LOG_WARN("%s: the pinned host tier is OFF for this run (the first cudaHostAlloc was refused: "
                          "%s); every fill that needed it went through the model mapping/pageable path or "
                          "declined.  The ceiling is NOT MemAvailable: on Windows it is the adapter's "
                          "shared-memory (aperture) budget, and it does not scale with RAM (measured 9.3-9.4 "
                          "GiB of shared usage on both a 47 and a 64 GiB box, issue #274) -- and the "
                          "driver-shipped HIP runtime reaches it where the ROCm/TheRock runtime does not "
                          "(issue #278, see the runtime line at startup).  Lower MOE_HOST_POOL_RAM_PCT / -lme, "
                          "make one GPU visible (ROCm/TheRock#9339), or put the ROCm/TheRock runtime DLLs "
                          "beside the executable\n",
                          __func__, cudaGetErrorString(g_pinned_alloc_err));
        }
        if (g_pool_after_latch > 0) {
            // issue #278: the complement of a latched refusal.  A pool that was built before the ceiling was
            // hit is pinned memory the run already owns, so it keeps serving; only NEW pools and new ring
            // entries are refused.  Zero here with pools present is the defect (every existing pool thrown
            // away, every expert copy pageable).
            GGML_LOG_WARN("%s: %lld expert fills were served from a host pool built BEFORE the refusal (a pool "
                          "slot is already pinned, so an existing pool keeps serving; only growing the tier is "
                          "refused)\n", __func__, (long long) g_pool_after_latch);
        }
#if defined(GGML_USE_HIP) && defined(_WIN32)
        {
            int64_t shm_limit = -1, shm_resident = -1;
            if (ggml_hip_win_shared_mem_info(&shm_limit, &shm_resident)) {
                GGML_LOG_WARN("%s: adapter shared (aperture) segments, ALL processes: %lld MiB commit limit, "
                              "%lld MiB resident -- that is the budget the pinned tier is charged against "
                              "(it is not VRAM and not MemAvailable)\n",
                              __func__, (long long) shm_limit, (long long) shm_resident);
            } else {
                GGML_LOG_WARN("%s: the adapter's shared (aperture) segment budget could not be queried on this "
                              "platform; the pinned ceiling is still not MemAvailable\n", __func__);
            }
        }
#endif
    }
    if (g_acc_hits + g_acc_fills + g_acc_colds > 0) {
        const int64_t acc   = g_acc_hits + g_acc_fills + g_acc_colds;
        const int64_t bytes = g_fill_bytes + g_cold_bytes;
        // One census op per (layer, role, device) table per token; integer division loses at most
        // (n_band_tables - 1) ops.  The denominator comes from the sizing snapshot, not from the live
        // table set (which the model teardown has already freed by the time this report runs).
        const int64_t n_band_tables = g_band_n_tables.load(std::memory_order_relaxed);
        const int64_t tokens = n_band_tables > 0 ? g_band_ops.load(std::memory_order_relaxed) / n_band_tables : 0;
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

// The `atexit` hook (registered by `moe_cache_init`).  A run whose model was freed has already printed the
// whole report from `moe_cache_forget_all`; this is the backstop for a run that never tore down (a server
// killed before `llama_model_free`, `--help`, a crash handler), and it prints nothing in the normal case.
void moe_cache_report() {
    if (!g_enabled) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_report_printed) {
        return;
    }
    g_report_printed = true;
    report_locked();
}

// ---------------------------------------------------------------------------------------------
// self-test: synthetic table, known pattern, fill + byte-verify.  Does not need a model.
// ---------------------------------------------------------------------------------------------



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

    if (g_device >= 0) {
        (void) cudaSetDevice(prev);
    }
}
