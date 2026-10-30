#include "llama-kv-cache.h"

#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"
#include "llama-context.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_map>

static bool ggml_is_power_of_2(int n) {
    return (n & (n - 1)) == 0;
}

//
// wip/kv-residency (TODO #117): provider P1 for the KV cache -- a growable VMM region instead of one
// `cudaMalloc` of the whole context.
//
// The KV buffer is allocated BEFORE the movable-boundary slab exists, so `SLAB_INIT` maps `free VRAM -
// reserve` and every token of context comes out of the MoE expert arena 1:1.  A region reserves the KV's VA
// up front and maps physical only for the cells a run reaches, which keeps that arithmetic exact at any
// context length without touching a kernel: the region's base never moves, so a captured graph's baked
// pointers stay valid, and the graph's existing "view is [0, n_kv) cells" contract is exactly the mapped
// extent.
//
// Gated OFF (`LLAMA_KV_REGION=1` enables).  When the backend cannot provide a region -- no VMM, a driver
// refusal, or no such hook at all -- the plain device buffer type is used and the numbers are the old ones.
//

typedef ggml_backend_buffer_type_t (*llama_kv_region_buft_fn)(ggml_backend_dev_t dev);
typedef int                      (*llama_kv_region_grow_fn)(ggml_backend_buffer_t buffer, uint32_t n_cells);

static bool llama_kv_region_enabled() {
    static const bool on = [] {
        const char * s = getenv("LLAMA_KV_REGION");
        return s != nullptr && s[0] != '\0' && strcmp(s, "0") != 0;
    }();
    return on;
}

// The backend's region buffer type for this device, or nullptr (feature off / unsupported / driver refused).
static ggml_backend_buffer_type_t llama_kv_region_buft(ggml_backend_dev_t dev) {
    if (!llama_kv_region_enabled() || dev == nullptr) {
        return nullptr;
    }
    auto fn = (llama_kv_region_buft_fn) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev),
            "ggml_backend_cuda_kv_region_buft");
    return fn != nullptr ? fn(dev) : nullptr;
}

// wip/kv-residency (TODO #118) A2: the same buffer type with the windowed KV -- the newest `cells` cells of
// the used extent are in VRAM, everything older in pinned system RAM.
typedef ggml_backend_buffer_type_t (*llama_kv_region_buft_split_fn)(ggml_backend_dev_t dev, size_t cells);

static ggml_backend_buffer_type_t llama_kv_region_buft_split(ggml_backend_dev_t dev, size_t cells) {
    if (!llama_kv_region_enabled() || dev == nullptr || cells == 0) {
        return nullptr;
    }
    auto fn = (llama_kv_region_buft_split_fn) ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev), "ggml_backend_cuda_kv_region_buft_split");
    return fn != nullptr ? fn(dev, cells) : nullptr;
}

// wip/kv-residency (TODO #118): the windowed KV's resident window, in CELLS.  0/unset = off.
//
// The resident window is the LAST `W` cells of the context, so the spill is the low cells [0, kv_size - W).
// A3 (this session) makes `W` a decision `--fit` can take instead of the user's arithmetic:
//   - a number         -> that window, exactly as before;
//   - `auto`           -> `--fit` sizes it (`llama_kv_region_set_resident_cells` below).  Without `--fit`
//                         there is nobody to decide, so the cache declines the window and says so;
//   - unset / 0        -> no window (today's fully resident cache, today's OOM).
static uint32_t kv_split_resident_cells_env() {
    const char * e = getenv("LLAMA_KV_RESIDENT_CELLS");
    if (e == nullptr || e[0] == '\0') {
        return 0u;
    }
    if (strcmp(e, "auto") == 0) {
        return UINT32_MAX;
    }
    return (uint32_t) strtoul(e, nullptr, 10);
}

// The fit-produced window (`--fit` with `LLAMA_KV_RESIDENT_CELLS=auto`), 0 = none decided.  A process
// global on purpose: `common_params_fit` runs before the contexts exist and hands the decision down here,
// and one W covers every KV cache the process creates (the A2 rule: one window for every tensor and
// cache).
static uint32_t g_kv_resident_cells_override = 0;

void llama_kv_region_set_resident_cells(uint32_t cells) {
    g_kv_resident_cells_override = cells;
}

bool llama_kv_region_is_enabled(void) {
    return llama_kv_region_enabled();
}

// UINT32_MAX = `auto`: fit sizes it (see above).
uint32_t llama_kv_region_resident_cells_requested(void) {
    if (g_kv_resident_cells_override != 0) {
        return g_kv_resident_cells_override;
    }
    return kv_split_resident_cells_env();
}

// How much pinned host RAM the spill may take (bytes), 0 = no ceiling / unknown.  The backend owns the
// number (`LLAMA_KV_SPILL_RAM_PCT` % of MemAvailable, shared by every cache in the process).
typedef size_t (*llama_kv_spill_budget_fn)(void);

typedef float (*llama_kv_link_gbps_fn)(ggml_backend_dev_t dev);

// The PCIe link constant every spill cost model divides by: the backend's own calibrated H2D number
// (~28.8 GB/s when the campaign was written, ~55 GB/s after the 2026-10 PCIe rework).  0 = unknown.
static float llama_kv_region_link_gbps(ggml_backend_dev_t dev) {
    if (dev == nullptr) {
        return 0.0f;
    }
    auto fn = (llama_kv_link_gbps_fn) ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev), "ggml_backend_cuda_kv_region_link_gbps");
    return fn != nullptr ? fn(dev) : 0.0f;
}

typedef void (*llama_kv_spill_reserve_fn)(size_t bytes);

// What is left of the pinned-spill ceiling for a NEW cache to plan, in bytes (SIZE_MAX = no ceiling).
static size_t llama_kv_region_spill_budget(ggml_backend_dev_t dev) {
    if (dev == nullptr) {
        return SIZE_MAX;
    }
    auto fn = (llama_kv_spill_budget_fn) ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev), "ggml_backend_cuda_kv_region_spill_ram_budget");
    return fn != nullptr ? fn() : SIZE_MAX;
}

// Account this cache's planned worst-case spill against that ceiling.
static void llama_kv_region_spill_reserve(ggml_backend_dev_t dev, size_t bytes) {
    if (dev == nullptr || bytes == 0) {
        return;
    }
    auto fn = (llama_kv_spill_reserve_fn) ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev), "ggml_backend_cuda_kv_region_spill_ram_reserve");
    if (fn != nullptr) {
        fn(bytes);
    }
}

// The grow hook is the same for every device, so resolve it once.
static llama_kv_region_grow_fn llama_kv_region_grow_hook() {
    static llama_kv_region_grow_fn fn = nullptr;
    static bool done = false;
    if (!done) {
        done = true;
        if (llama_kv_region_enabled()) {
            for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
                ggml_backend_reg_t reg = ggml_backend_reg_get(i);
                fn = (llama_kv_region_grow_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_kv_region_grow");
                if (fn != nullptr) {
                    break;
                }
            }
        }
    }
    return fn;
}

// orthonormal Walsh-Hadamard rotation matrix
// note: res^2 == I
static void ggml_gen_hadamard(ggml_tensor * tensor) {
    assert(tensor->type == GGML_TYPE_F32);

    const int n = tensor->ne[0];

    assert(ggml_is_power_of_2(n));
    assert(tensor->ne[1] == n);
    assert(tensor->ne[2] == 1);
    assert(tensor->ne[3] == 1);

    std::vector<float> data_f32;

    float * data = (float *) tensor->data;

    if (tensor->type != GGML_TYPE_F32) {
        data_f32.resize(n*n);
        data = data_f32.data();
    }

    data[0*n + 0] = 1.0 / sqrtf(n);

    for (int s = 1; s < n; s *= 2) {
        for (int i = 0; i < s; i++) {
            for (int j = 0; j < s; j++) {
                const float val = data[i*n + j];

                data[(i + s)*n + (j    )] =  val;
                data[(i    )*n + (j + s)] =  val;
                data[(i + s)*n + (j + s)] = -val;
            }
        }
    }

    if (tensor->type != GGML_TYPE_F32) {
        ggml_quantize_chunk(tensor->type, data, tensor->data, 0, 1, n*n, nullptr);
    }
}

//
// llama_kv_cache
//

// LLAMA_KV_N_PAD_PREFILL: a coarser pad for prompt-sized ubatches (see get_n_kv).  The env value is rounded
// up to a power of two (GGML_PAD is a bit mask); 0 (or an unset value on a single device) turns it off.
// Default-on only when more than one GPU is in use: a single GPU rebuilds its graph cheaply and the
// coarser pad only adds masked attention work there (-0.35 % measured on a dense 27B), so the default
// stays inactive for that shape.  Under -sm tensor the model can be a single Meta device, hence the
// split-state count as well as the device list.
static uint32_t kv_n_pad_prefill_default(const llama_model & model) {
    if (const char * env = getenv("LLAMA_KV_N_PAD_PREFILL")) {
        const int e = atoi(env);
        if (e <= 0 || e > (1 << 24)) {
            return 0;
        }
        uint32_t p2 = 1;
        while (p2 < (uint32_t) e) {
            p2 <<= 1;
        }
        return p2;
    }

    const bool multi_device = model.get_split_state_ud.n_devices > 1 || model.n_devices() > 1;
    return multi_device ? 8192u : 0u;
}

llama_kv_cache::llama_kv_cache(
        const llama_model & model,
        const llama_hparams & hparams,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   offload,
                     bool   unified,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
           llama_memory_t   mem_other,
    const layer_filter_cb & filter,
    const  layer_reuse_cb & reuse,
    const  layer_share_cb & share,
             const char *   name_tag,
                     bool   v_enabled) :
    model(model), hparams(hparams), v_trans(v_trans),
    n_seq_max(n_seq_max), n_stream(unified ? 1 : n_seq_max), n_pad(n_pad),
    n_pad_prefill(kv_n_pad_prefill_default(model)), n_swa(n_swa), swa_type(swa_type),
    other(static_cast<llama_kv_cache *>(mem_other)),
    v_cells_impl(other ? other->v_cells_impl : std::make_shared<llama_kv_cells_vec>()),
    v_cells(*v_cells_impl) {

    // shared cells view the source cache's K/V tensors, so the cell count
    // follows the source allocation: a fitted target can be smaller than the
    // draft default and oversized views would overflow the source tensors
    if (other) {
        const uint32_t size_other = other->get_size();
        if (kv_size != size_other) {
            LLAMA_LOG_WARN("%s: kv_size = %u overridden to %u to match the shared source cache\n", __func__, kv_size, size_other);
            kv_size = size_other;
        }
    }

    GGML_ASSERT(kv_size % n_pad == 0);

    const uint32_t n_layer = hparams.n_layer_all;

    // define a comparator for the buft -> ctx map to ensure that the order is well-defined:
    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };
    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    // create a context for each buffer type
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ size_t(2u*(1 + n_stream)*n_layer*ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map.emplace(buft, ctx);

            return ctx;
        }

        return it->second.get();
    };

    GGML_ASSERT(n_stream == 1 || n_stream == n_seq_max);

    v_heads.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_heads[s] = 0;
    }

    v_cells.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].resize(kv_size);
    }

    // by default, all sequence ids are mapped to the 0th stream
    seq_to_stream.resize(LLAMA_MAX_SEQ, 0);

    if (n_stream > 1) {
        seq_to_stream.resize(n_stream, 0);
        for (uint32_t s = 0; s < n_stream; ++s) {
            seq_to_stream[s] = s;
        }
    }

    // [TAG_V_CACHE_VARIABLE]
    if (v_trans && hparams.is_n_embd_v_gqa_variable()) {
        LLAMA_LOG_WARN("%s: the V embeddings have different sizes across layers and FA is not enabled - padding V cache to %d\n",
                __func__, hparams.n_embd_v_gqa_max());
    }

    const bool is_mla = hparams.is_mla();

    // wip/kv-residency (TODO #118) A2: the windowed KV.  The RESIDENT WINDOW is the newest `W` cells of the
    // USED extent (`LLAMA_KV_RESIDENT_CELLS`), so the host/VRAM boundary slides forward as the context fills
    // and a token always attends the newest cells from VRAM.  The spill grows with the used extent up to
    // (kv_size - W) cells, i.e. the VRAM the cache never commits is what the expert arena gets.
    //   Default W = kv_size -> no spill, i.e. today's fully resident cache (the opt-in is what spills).
    //   One W covers every tensor, so the cell geometry has to be uniform across the layers this cache holds
    //   (K and V included).  Anything else DECLINES the window -- no spill, today's fit-or-fail -- not a guess.
    if (llama_kv_region_enabled() && other == nullptr && !v_trans && !hparams.no_alloc) {
        const uint32_t w = llama_kv_region_resident_cells_requested();
        if (w == UINT32_MAX) {
            // `LLAMA_KV_RESIDENT_CELLS=auto` with nobody to decide: `--fit` owns the sizing (it is the only
            // thing that knows the free-VRAM budget the ARENA also draws on, and the RAM budget the spill
            // draws on).  Declining keeps the run honest -- today's fully resident cache, today's OOM.
            LLAMA_LOG_WARN("%s: LLAMA_KV_RESIDENT_CELLS=auto is sized by --fit, which did not run (or could not "
                           "decide); the KV stays fully resident.  Pass the window explicitly "
                           "(LLAMA_KV_RESIDENT_CELLS=<cells>) or run with --fit.\n", __func__);
        }
        const uint32_t n_tail = (w != 0 && w != UINT32_MAX && kv_size > w) ? kv_size - w : 0;
        size_t cell = 0;
        size_t n_tens = 0;
        bool   uniform = n_tail != 0;
        for (uint32_t il = 0; uniform && il < n_layer; ++il) {
            if (!hparams.has_kv(il) || (filter && !filter(il))) {
                continue;
            }
            const size_t ck = ggml_row_size(type_k, hparams.n_embd_k_gqa(il));
            const size_t cv = (!is_mla && v_enabled) ? ggml_row_size(type_v, hparams.n_embd_v_gqa(il)) : ck;
            if (cell == 0) {
                cell = ck;
            }
            uniform = ck == cell && cv == cell;
            if (uniform) {
                n_tens += (!is_mla && v_enabled) ? 2 : 1;
            }
        }
        size_t spill_cap_bytes = 0;   // > 0 when the RAM budget clamped the window
        size_t w_eff = w;
        if (uniform) {
            // A3/item 3: the spill is PINNED, non-swappable RAM, so the window is clamped UP (fewer spilled
            // cells) until the worst case -- every tail cell of the context -- fits the RAM budget the
            // backend's `LLAMA_KV_SPILL_RAM_PCT` ceiling allows.  This is the creation-time decision; the
            // backend keeps a run-time check for RAM that disappears under a running process.
            const size_t cell_all = cell * n_tens * n_stream;              // one cell, every tensor
            const size_t budget   = llama_kv_region_spill_budget(model.dev_layer(0));
            const size_t spill_worst = (size_t) n_tail * cell_all;
            if (budget != SIZE_MAX && spill_worst > budget) {
                spill_cap_bytes = budget;
                const size_t tail_cells = budget / (cell_all != 0 ? cell_all : 1);
                w_eff = kv_size > tail_cells ? (size_t) kv_size - tail_cells : (size_t) kv_size;
                LLAMA_LOG_WARN("%s: the windowed KV's spill would need %.2f MiB of pinned, NON-SWAPPABLE RAM at "
                               "full context (%.0f KiB per cell x %u tail cells in the cache), but only %.2f MiB "
                               "is available (the LLAMA_KV_SPILL_RAM_PCT ceiling on pinned RAM).  The resident "
                               "window is raised from %u to %zu of %u cells, so the spill is %.2f MiB.\n",
                               __func__, (double) spill_worst / 1048576.0, (double) cell_all / 1024.0, n_tail,
                               (double) budget / 1048576.0,
                               w, w_eff, kv_size, (double) ((size_t) kv_size - w_eff) * cell_all / 1048576.0);
                if (w_eff >= kv_size) {
                    LLAMA_LOG_ERROR("%s: even with the whole KV resident the RAM budget leaves no room for this "
                                   "window; the KV stays fully resident (today's fit-or-fail).  Lower -c, raise "
                                   "LLAMA_KV_SPILL_RAM_PCT, or free RAM.\n", __func__);
                }
            }
            // account the plan, so a second cache (Flash-Next has two) sizes against what is left
            llama_kv_region_spill_reserve(model.dev_layer(0), (size_t) ((size_t) kv_size - w_eff) * cell_all);
        }
        if (!uniform) {
            LLAMA_LOG_WARN("%s: the windowed KV is unavailable for this cache (%s); the KV stays fully "
                           "resident\n", __func__,
                           n_tail == 0 ? "no spilled cells" : "non-uniform K/V cell geometry across layers");
        } else if (w_eff < kv_size) {
            const size_t tail = (size_t) kv_size - w_eff;
            const float  gbps = llama_kv_region_link_gbps(model.dev_layer(0));
            const size_t cell_all = cell * n_tens * n_stream;
            this->kv_resident_cells = w_eff;
            LLAMA_LOG_WARN("%s: WINDOWED KV ENABLED: the newest %zu of up to %u cells stay in VRAM, everything "
                           "older spills to pinned system RAM (%.1f KiB per cell%s).  The spill grows with the "
                           "used extent, up to %.2f MiB at full context, and every spilled cell is read over "
                           "the PCIe link per step (%s).  Unset LLAMA_KV_RESIDENT_CELLS or LLAMA_KV_REGION to "
                           "keep the KV fully resident.%s\n",
                           __func__, w_eff, kv_size, cell / 1024.0, n_stream > 1 ? ", per stream" : "",
                           (double) tail * cell_all / 1048576.0,
                           gbps > 0.0f ? (std::to_string((int) (gbps + 0.5f)) + " GB/s, calibrated").c_str()
                                       : "the link's measured rate",
                           spill_cap_bytes != 0 ? "  (The window was widened by the RAM budget -- see above.)" : "");
        }
    }

    for (uint32_t il = 0; il < n_layer; il++) {
        if (!hparams.has_kv(il)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: does not have KV cache\n", __func__, il);
            continue;
        }

        if (filter && !filter(il)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: filtered\n", __func__, il);
            continue;
        }

        if (share && other) {
            const int32_t il_share = share(il);

            if (il_share >= 0) {
                const auto & layer_share = other->layers[other->map_layer_ids[il_share]];

                LLAMA_LOG_WARN("%s: layer %3d: sharing with layer %d. k = %p, v = %p\n", __func__, il, il_share,
                        layer_share.k->data, layer_share.v->data);

                map_layer_ids[il] = layers.size();

                layers.push_back(layer_share);
                layers.back().il = il;

                continue;
            }
        }

        if (n_embd_head_k_all == 0) {
            n_embd_head_k_all = (int32_t) hparams.n_embd_head_k(il);
        } else if (n_embd_head_k_all > 0 && n_embd_head_k_all != (int32_t) hparams.n_embd_head_k(il)) {
            n_embd_head_k_all = -1;
        }

        if (!is_mla) {
            if (n_embd_head_v_all == 0) {
                n_embd_head_v_all = (int32_t) hparams.n_embd_head_v(il);
            } else if (n_embd_head_v_all > 0 && n_embd_head_v_all != (int32_t) hparams.n_embd_head_v(il)) {
                n_embd_head_v_all = -1;
            }
        }

        // [TAG_V_CACHE_VARIABLE]
        const uint32_t n_embd_k_gqa =            hparams.n_embd_k_gqa(il);
        const uint32_t n_embd_v_gqa = !v_trans ? hparams.n_embd_v_gqa(il) : hparams.n_embd_v_gqa_max();

        const char * dev_name = "CPU";

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

        if (offload) {
            auto * dev = model.dev_layer(il);
            buft = ggml_backend_dev_buffer_type(dev);

            dev_name = ggml_backend_dev_name(dev);

            // wip/kv-residency (TODO #117): take the growable KV region when the backend offers one.  It is a
            // device buffer type with the plain type's vtable contract (same device, same name predicate), so
            // everything downstream -- the scheduler's placement, `memory_breakdown()`, `--fit` -- sees a KV
            // buffer of the same size; only WHEN its physical is committed changes.
            ggml_backend_buffer_type_t buft_region = kv_resident_cells != 0 ? llama_kv_region_buft_split(dev, kv_resident_cells) : nullptr;
            if (kv_resident_cells != 0 && buft_region == nullptr) {
                // A silent fallback here means a warning above that promises a window the buffer does not have.
                static bool warned_split = false;
                if (!warned_split) {
                    warned_split = true;
                    LLAMA_LOG_WARN("%s: the backend declined the windowed KV buffer type for %zu cells; the KV is "
                                   "fully resident (no spill), so the VRAM saving does not happen\n",
                                   __func__, kv_resident_cells);
                }
            }
            if (buft_region == nullptr) {
                buft_region = llama_kv_region_buft(dev);
            }
            if (buft_region != nullptr) {
                buft = buft_region;
                kv_region = true;
            }
        }

        LLAMA_LOG_DEBUG("%s: layer %3d: dev = %s\n", __func__, il, dev_name);

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for kv cache");
        }

        const bool has_k = true;
        const bool has_v = !is_mla && v_enabled;

        ggml_tensor * k = has_k ? ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream) : nullptr;
        ggml_tensor * v = has_v ? ggml_new_tensor_3d(ctx, type_v, n_embd_v_gqa, kv_size, n_stream) : nullptr;

        has_k && ggml_format_name(k, "cache_%sk_l%d", name_tag, il);
        has_v && ggml_format_name(v, "cache_%sv_l%d", name_tag, il);

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;

        for (uint32_t s = 0; s < n_stream; ++s) {
            k_stream.push_back(has_k ? ggml_view_2d(ctx, k, n_embd_k_gqa, kv_size, k->nb[1], s*k->nb[2]) : nullptr);
            v_stream.push_back(has_v ? ggml_view_2d(ctx, v, n_embd_v_gqa, kv_size, v->nb[1], s*v->nb[2]) : nullptr);
        }

        map_layer_ids[il] = layers.size();

        layers.push_back({ il, k, v, k_stream, v_stream, });
    }

    if (reuse) {
        LLAMA_LOG_DEBUG("%s: reusing layers:\n", __func__);

        for (uint32_t il = 0; il < n_layer; il++) {
            const int32_t il_reuse = reuse(il);

            if (il_reuse < 0) {
                LLAMA_LOG_DEBUG("%s: - layer %3d: no reuse\n", __func__, il);
                continue;
            }

            if (filter && !filter(il)) {
                LLAMA_LOG_DEBUG("%s: - layer %3d: filtered\n", __func__, il);
                continue;
            }

            GGML_ASSERT(map_layer_ids.find(il_reuse) != map_layer_ids.end());

            map_layer_ids[il] = map_layer_ids[il_reuse];

            LLAMA_LOG_DEBUG("%s: - layer %3d: reuse layer %d, is_swa = %d\n", __func__, il, il_reuse, hparams.is_swa(il));
        }
    }

    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf;
        if (hparams.no_alloc) {
            buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                t->buffer = buf; // set dummy buffer for KV cache so that the backend scheduler won't try to allocate it
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft); // real buffer
        }
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for kv cache");
        }

        LLAMA_LOG_INFO("%s: %10s KV buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);

        ggml_backend_buffer_clear(buf, 0);
        ctxs_bufs.emplace_back(std::move(ctx), buf);
    }

    {
        const size_t memory_size_k = size_k_bytes();
        const size_t memory_size_v = size_v_bytes();

        LLAMA_LOG_INFO("%s: size = %7.2f MiB (%6u cells, %3d layers, %2u/%u seqs), K (%s): %7.2f MiB, V (%s): %7.2f MiB\n", __func__,
                (float)(memory_size_k + memory_size_v) / (1024.0f * 1024.0f), kv_size, (int) layers.size(), n_seq_max, n_stream,
                ggml_type_name(type_k), (float)memory_size_k / (1024.0f * 1024.0f),
                ggml_type_name(type_v), (float)memory_size_v / (1024.0f * 1024.0f));
    }

    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        n_embd_head_k_all = other->n_embd_head_k_all;
        n_embd_head_v_all = other->n_embd_head_v_all;

        n_rot_k = other->n_rot_k;
        n_rot_v = other->n_rot_v;
    } else {
        // two knobs turn the same thing off: the environment variable, for the whole process, and
        //   the model setting, for the contexts of one model - see llama_model_set_attn_rot
        const char * env_rot_disable = getenv("LLAMA_ATTN_ROT_DISABLE");

        const bool attn_rot_disable = !model.attn_rot || (env_rot_disable != nullptr && atoi(env_rot_disable) != 0);
        if (attn_rot_disable) {
            LLAMA_LOG_WARN("%s: attention rotation force disabled (%s)\n", __func__,
                    model.attn_rot ? "LLAMA_ATTN_ROT_DISABLE" : "llama_model_set_attn_rot");
        }

        bool attn_rot_k =
            !attn_rot_disable &&
            n_embd_head_k_all > 0 &&
            ggml_is_quantized(type_k) &&
            hparams.n_embd_head_k() % 64 == 0;

        // always create Hadamard rotation tensors for DeepSeek lightning indexers
        if ((model.arch == LLM_ARCH_DEEPSEEK32 || model.arch == LLM_ARCH_DEEPSEEK4 ||
                model.arch == LLM_ARCH_GLM_DSA || model.arch == LLM_ARCH_DOTS3NOTE) &&
                hparams.n_embd_head_k_full == hparams.indexer_head_size) {
            attn_rot_k = true;
        }

        const bool attn_rot_v =
            !attn_rot_disable &&
            n_embd_head_v_all > 0 &&
            ggml_is_quantized(type_v) &&
            hparams.n_embd_head_v() % 64 == 0;

        if (attn_rot_k) {
            n_rot_k = 64;

            // TODO: investigate if using the smallest rotation matrix is beneficial also for K (similar as for V)
            // ref: https://github.com/ggml-org/llama.cpp/pull/21038#issuecomment-4141323088
            while (n_embd_head_k_all % (2*n_rot_k) == 0) {
                n_rot_k *= 2;
            }
        }
        if (attn_rot_v) {
            n_rot_v = 64;
            // using smaller rotation matrices for V seems beneficial
            // ref: https://github.com/ggml-org/llama.cpp/pull/21038#issuecomment-4146397570
            //while (hparams.n_embd_head_v() % (2*n_rot_v) == 0) {
            //    n_rot_v *= 2;
            //}
        }
    }

    LLAMA_LOG_INFO("%s: n_rot_k = %u, n_embd_head_k_all = %d\n", __func__, n_rot_k, n_embd_head_k_all);
    LLAMA_LOG_INFO("%s: n_rot_v = %u, n_embd_head_k_all = %d\n", __func__, n_rot_v, n_embd_head_v_all);

    // pre-compute the haramard matrices and keep them in host memory
    // TODO: in the future, we can make copies in the backend buffers to avoid host -> device transfers
    if (n_rot_k || n_rot_v) {
        for (int64_t n = 64; n <= std::max(n_embd_head_k_all, n_embd_head_v_all); n *= 2) {
            attn_rot_hadamard[n] = std::vector<float>(n*n);

            ggml_init_params params = {
                /* .mem_size   = */ 1*ggml_tensor_overhead(),
                /* .mem_buffer = */ nullptr,
                /* .no_alloc   = */ true,
            };

            ggml_context_ptr ctx { ggml_init(params) };

            ggml_tensor * tmp = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n, n);
            tmp->data = attn_rot_hadamard[n].data();

            ggml_gen_hadamard(tmp);
        }
    }

    const char * LLAMA_KV_CACHE_DEBUG = getenv("LLAMA_KV_CACHE_DEBUG");
    debug = LLAMA_KV_CACHE_DEBUG ? atoi(LLAMA_KV_CACHE_DEBUG) : 0;
}

void llama_kv_cache::clear(bool data) {
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].reset();
        v_heads[s] = 0;
    }

    parked.clear();

    if (data) {
        for (auto & [_, buf] : ctxs_bufs) {
            ggml_backend_buffer_clear(buf.get(), 0);
        }
    }
}

bool llama_kv_cache::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return true;
    }

    // TODO: fix incosistent handling of `seq_id < 0` and `seq_id == -1` in the codebase [TAG_LLAMA_SEQ_ID_NEG]
    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    if (seq_id >= 0) {
        auto & cells = v_cells[seq_to_stream[seq_id]];
        auto & head  = v_heads[seq_to_stream[seq_id]];

        uint32_t new_head = cells.size();

        // only the used range can match: an empty cell's pos is -1 (bounds taken before the loop changes them)
        const uint32_t i_end = cells.used_max_p1();
        for (uint32_t i = cells.used_min(); i < i_end; ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id) && cells.seq_rm(i, seq_id)) {
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }

        // If we freed up a slot, set head to it so searching can start there.
        if (new_head != cells.size() && new_head < head) {
            head = new_head;
        }

        if (parked.count(seq_id) > 0) {
            park_refresh();
        }
    } else {
        // match any sequence
        for (uint32_t s = 0; s < n_stream; ++s) {
            auto & cells = v_cells[s];
            auto & head  = v_heads[s];

            uint32_t new_head = cells.size();

            const uint32_t i_end = cells.used_max_p1();
            for (uint32_t i = cells.used_min(); i < i_end; ++i) {
                if (!cells.pos_in(i, p0, p1)) {
                    continue;
                }

                cells.rm(i);

                if (new_head == cells.size()) {
                    new_head = i;
                }
            }

            // If we freed up a slot, set head to it so searching can start there.
            if (new_head != cells.size() && new_head < head) {
                head = new_head;
            }
        }

        if (!parked.empty()) {
            park_refresh();
        }
    }

    return true;
}

void llama_kv_cache::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id_src >= 0 && (size_t) seq_id_src < seq_to_stream.size());
    GGML_ASSERT(seq_id_dst >= 0 && (size_t) seq_id_dst < seq_to_stream.size());

    if (parked.count(seq_id_src) > 0 || parked.count(seq_id_dst) > 0) {
        parked.clear(); // a copy out of (or into) the parked cells: give up parking rather than hide cells from the range
    }

    const auto s0 = seq_to_stream[seq_id_src];
    const auto s1 = seq_to_stream[seq_id_dst];

    if (s0 == s1) {
        // since both sequences are in the same stream, no data copy is necessary
        // we just have to update the cells meta data

        auto & cells = v_cells[s0];

        if (seq_id_src == seq_id_dst) {
            return;
        }

        if (p0 < 0) {
            p0 = 0;
        }

        if (p1 < 0) {
            p1 = std::numeric_limits<llama_pos>::max();
        }

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id_src)) {
                cells.seq_add(i, seq_id_dst);
            }
        }

        return;
    }

    // cross-stream sequence copies require to copy the actual buffer data

    bool is_full = true;

    if (p0 > 0 && p0 + 1 < (int) get_size()) {
        is_full = false;
    }

    if (p1 > 0 && p1 + 1 < (int) get_size()) {
        is_full = false;
    }

    GGML_ASSERT(is_full && "seq_cp() is only supported for full KV buffers");

    // enqueue the copy operation - the buffer copy will be performed during the next update
    sc_info.ssrc.push_back(s0);
    sc_info.sdst.push_back(s1);

    v_cells[s1].reset();
    for (uint32_t i = 0; i < v_cells[s0].size(); ++i) {
        if (v_cells[s0].seq_has(i, seq_id_src)) {
            llama_pos pos   = v_cells[s0].pos_get(i);
            llama_pos shift = v_cells[s0].get_shift(i);

            llama_kv_cell_ext ext = v_cells[s0].ext_get(i);

            if (shift != 0) {
                pos -= shift;
                assert(pos >= 0);
            }

            v_cells[s1].pos_set(i, pos);
            v_cells[s1].seq_add(i, seq_id_dst);

            if (shift != 0) {
                v_cells[s1].pos_add(i, shift);
            }

            v_cells[s1].ext_set(i, ext);
        }
    }

    v_heads[s1] = v_heads[s0];

    //for (uint32_t s = 0; s < n_stream; ++s) {
    //    LLAMA_LOG_WARN("%s: seq %d: min = %d, max = %d\n", __func__, s, v_cells[s].seq_pos_min(s), v_cells[s].seq_pos_max(s));
    //}
}

void llama_kv_cache::seq_keep(llama_seq_id seq_id) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    auto & cells = v_cells[seq_to_stream[seq_id]];
    auto & head  = v_heads[seq_to_stream[seq_id]];

    uint32_t new_head = cells.size();

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cells.seq_keep(i, seq_id)) {
            if (new_head == cells.size()) {
                new_head = i;
            }
        }
    }

    if (!parked.empty()) {
        park_refresh();
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != cells.size() && new_head < head) {
        head = new_head;
    }
}

// LLAMA_KV_PARK_DEBUG=1: log every park, unpark and compaction move, and why a park or an unpark is refused
static bool llama_kv_park_debug() {
    static const bool v = getenv("LLAMA_KV_PARK_DEBUG") != nullptr && atoi(getenv("LLAMA_KV_PARK_DEBUG")) != 0;
    return v;
}

uint32_t llama_kv_cache::park_bound() const {
    uint32_t res = v_cells[0].size();
    for (const auto & [seq_id, range] : parked) {
        res = std::min(res, range.first);
    }
    return res;
}

void llama_kv_cache::park_refresh() {
    const auto & cells = v_cells[0];
    for (auto it = parked.begin(); it != parked.end(); ) {
        uint32_t lo = it->second.second;
        uint32_t hi = it->second.first;
        for (uint32_t i = it->second.first; i < it->second.second; ++i) {
            if (!cells.is_empty(i) && cells.seq_has(i, it->first)) {
                lo = std::min(lo, i);
                hi = i + 1;
            }
        }
        if (lo >= hi) {
            it = parked.erase(it);
        } else {
            it->second = { lo, hi };
            ++it;
        }
    }
}

void llama_kv_cache::park_move(uint32_t src, uint32_t dst, uint32_t n) {
    if (src == dst || n == 0) {
        return;
    }

    auto & cells = v_cells[0];

    // copy in chunks no longer than the distance, from the end when moving up and from the start when moving down, so
    // that no chunk overwrites cells that have not been moved yet
    const uint32_t dist  = src > dst ? src - dst : dst - src;
    const uint32_t chunk = std::min(n, dist);

    ggml_init_params params = {
        /*.mem_size   =*/ 4*layers.size()*ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    // The views below are created outside any graph, once per chunk, so their per-device simple tensors must not go to
    // the buffer's per-generation compute-view containers: those are reclaimed only when a graph is rebuilt, and this
    // move runs between graphs.  A long run parked into a small gap needs `n/chunk` chunks (chunk == the gap), each
    // registering 2 views per (layer, K/V), so a nearly full unified cache registered thousands of them in one move and
    // exhausted the container's fixed budget -- `ggml_new_object: not enough space in the context's memory pool`
    // (issue #232).  They go to the meta backend's scratch container instead, reset once this chunk's copies (which are
    // synchronous: a meta buffer has no `.cpy_tensor`) have completed.
    std::vector<ggml_backend_buffer_t> view_bufs;
    for (const auto & layer : layers) {
        for (ggml_tensor * t : { layer.k, layer.v }) {
            if (t != nullptr && t->buffer != nullptr) {
                view_bufs.push_back(t->buffer);
            }
        }
    }
    std::sort(view_bufs.begin(), view_bufs.end());
    view_bufs.erase(std::unique(view_bufs.begin(), view_bufs.end()), view_bufs.end());

    ggml_backend_meta_set_scratch_views(true);

    for (uint32_t done = 0; done < n; done += chunk) {
        const uint32_t len = std::min(chunk, n - done);
        const uint32_t off = dst > src ? n - done - len : done;

        ggml_context_ptr ctx { ggml_init(params) };

        for (const auto & layer : layers) {
            for (ggml_tensor * t : { layer.k, layer.v }) {
                if (t == nullptr) {
                    continue;
                }
                ggml_tensor * a = ggml_view_2d(ctx.get(), t, t->ne[0], len, t->nb[1], (size_t) (src + off)*t->nb[1]);
                ggml_tensor * b = ggml_view_2d(ctx.get(), t, t->ne[0], len, t->nb[1], (size_t) (dst + off)*t->nb[1]);
                GGML_ASSERT(ggml_backend_view_init(a) == GGML_STATUS_SUCCESS);
                GGML_ASSERT(ggml_backend_view_init(b) == GGML_STATUS_SUCCESS);
                ggml_backend_tensor_copy(a, b);
            }
        }

        if (dst > src) {
            for (uint32_t i = len; i-- > 0; ) {
                cells.mv(src + off + i, dst + off + i);
            }
        } else {
            for (uint32_t i = 0; i < len; ++i) {
                cells.mv(src + off + i, dst + off + i);
            }
        }

        // the copies are synchronous, so this chunk's views and their simple tensors are dead
        for (ggml_backend_buffer_t b : view_bufs) {
            ggml_backend_meta_reset_scratch_views(b);
        }
    }

    ggml_backend_meta_set_scratch_views(false);
}

// the cells of seq_id in stream 0, when they form one run that holds no other sequence: [lo, hi); else lo == hi
static std::pair<uint32_t, uint32_t> llama_kv_cache_seq_run(const llama_kv_cells & cells, llama_seq_id seq_id);

void llama_kv_cache::park_compact() {
    std::vector<std::pair<uint32_t, llama_seq_id>> order;
    for (const auto & [seq_id, range] : parked) {
        order.emplace_back(range.first, seq_id);
    }
    std::sort(order.rbegin(), order.rend());

    const auto & cells = v_cells[0];

    uint32_t top = cells.size();
    for (const auto & [lo_old, seq_id] : order) {
        GGML_UNUSED(lo_old);
        const auto [lo, hi] = llama_kv_cache_seq_run(cells, seq_id);
        if (lo == hi) {
            break; // not one run any more: leave the rest where it is
        }
        const uint32_t n = hi - lo;
        if (hi != top) {
            // between this sequence and the ones above it there are only free cells: the region above lo up to top
            // holds no active cell (those are all below the lowest parked one) and no other parked sequence
            park_move(lo, top - n, n);
            if (llama_kv_park_debug()) {
                LLAMA_LOG_WARN("%s: seq %d [%u, %u) -> [%u, %u)\n", __func__, seq_id, lo, hi, top - n, top);
            }
        }
        parked[seq_id] = { top - n, top };
        top -= n;
    }
}

static std::pair<uint32_t, uint32_t> llama_kv_cache_seq_run(const llama_kv_cells & cells, llama_seq_id seq_id) {
    uint32_t lo = 0;
    uint32_t hi = 0;
    uint32_t n  = 0;
    for (const auto & [pos, i] : cells.seq_pos_get(seq_id)) {
        GGML_UNUSED(pos);
        if (cells.seq_count(i) != 1) {
            return { 0, 0 };
        }
        lo = n == 0 ? i : std::min(lo, i);
        hi = n == 0 ? i + 1 : std::max(hi, i + 1);
        n++;
    }
    return n == hi - lo ? std::make_pair(lo, hi) : std::make_pair(0u, 0u);
}

bool llama_kv_cache::seq_park(llama_seq_id seq_id) {
    if (!can_park()) {
        return false;
    }

    // wip/kv-residency (TODO #118): a park MOVES cells, and park_bound() is the top of the CACHE -- above the cell
    // watermark a growable/windowed region has mapped.  The move is `ggml_backend_tensor_copy` (the copy engine),
    // so a write into unmapped VA is a GPU PAGE FAULT, not an error: measured on Flash-Next (both KV caches
    // windowed, `-c 65536 -ncmoe 48`) and reproduced with the window OFF (`LLAMA_KV_REGION=1` alone) and at a
    // depth where nothing spills, so it belongs to the region provider, not to the window
    // (`../llama-cpp-rdna-boosts/wip/kv-residency/A2-MIXED-BACKING.md` §3.7.11.3).
    //
    // Growing the region to cover the park instead is not free either: on a windowed cache the parked (cold)
    // cells would take the resident window while the ACTIVE cells fall below the spill boundary, so every decode
    // step would read the running context over the link -- a per-token cost against save-and-clear's one state
    // copy per idle transition.  So this DECLINES, which leaves every cell where it is and lets the caller use its
    // fallback: the server saves the slot to the prompt cache and clears it (server-context.cpp,
    // `park_idle_slots_for_launch`: "cannot park the idle slot: saving it to the prompt cache and clearing it"),
    // i.e. exactly the behaviour the parking gate compares against.  A region-backed cache is opt-in
    // (`LLAMA_KV_REGION=1`), so the default path is unchanged.
    if (kv_region) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            LLAMA_LOG_WARN("%s: the KV is a growable/windowed REGION (LLAMA_KV_REGION): a park moves cells above "
                    "the region's mapped watermark and a copy-engine move there faults the GPU, so idle slots are "
                    "saved to the prompt cache and cleared instead of parked (one state copy per idle transition)\n",
                    __func__);
        }
        return false;
    }
    if (seq_id < 0 || seq_id >= LLAMA_MAX_SEQ || parked.count(seq_id) > 0) {
        return seq_id >= 0 && seq_id < LLAMA_MAX_SEQ;
    }

    auto & cells = v_cells[0];

    const auto [lo, hi] = llama_kv_cache_seq_run(cells, seq_id);
    if (lo == hi) {
        if (llama_kv_park_debug() && !cells.seq_pos_get(seq_id).empty()) {
            LLAMA_LOG_WARN("%s: seq %d: its %zu cells are not one run of its own\n", __func__, seq_id, cells.seq_pos_get(seq_id).size());
        }
        return cells.seq_pos_get(seq_id).empty(); // nothing to park, or a run that cannot move
    }

    park_compact();

    const uint32_t n     = hi - lo;
    const uint32_t bound = park_bound();
    if (n > bound) {
        return false;
    }
    const uint32_t dst = bound - n;

    // the target run must hold nothing but free cells and cells of the run itself
    for (uint32_t i = dst; i < bound; ++i) {
        if (!cells.is_empty(i) && (i < lo || i >= hi)) {
            if (llama_kv_park_debug()) {
                LLAMA_LOG_WARN("%s: seq %d [%u, %u): cell %u of the target [%u, %u) is taken (seq %d)\n", __func__, seq_id, lo, hi,
                        i, dst, bound, cells.seq_count(i) == 1 ? cells.seq_get(i) : -1);
            }
            return false;
        }
    }

    park_move(lo, dst, n);
    parked[seq_id] = { dst, bound };
    if (llama_kv_park_debug()) {
        LLAMA_LOG_WARN("%s: seq %d [%u, %u) -> [%u, %u)\n", __func__, seq_id, lo, hi, dst, bound);
    }

    auto & head = v_heads[0];
    head = std::min(head, lo);

    return true;
}

bool llama_kv_cache::seq_unpark(llama_seq_id seq_id) {
    const auto it = parked.find(seq_id);
    if (it == parked.end()) {
        return true;
    }

    auto & cells = v_cells[0];

    const auto [lo, hi] = llama_kv_cache_seq_run(cells, seq_id);
    if (lo == hi) {
        return false;
    }
    const uint32_t n = hi - lo;

    const auto range = it->second;
    parked.erase(it);
    const uint32_t bound = park_bound();

    // the lowest run of n cells below the remaining parked ones that holds nothing but free cells and cells of the run
    uint32_t dst = 0;
    uint32_t len = 0;
    for (uint32_t i = 0; i < bound && len < n; ++i) {
        if (cells.is_empty(i) || (i >= lo && i < hi)) {
            if (len == 0) {
                dst = i;
            }
            len++;
        } else {
            len = 0;
        }
    }
    if (len < n) {
        if (llama_kv_park_debug()) {
            LLAMA_LOG_WARN("%s: seq %d [%u, %u): no run of %u free cells below %u\n", __func__, seq_id, lo, hi, n, bound);
        }
        parked[seq_id] = range;
        return false;
    }

    park_move(lo, dst, n);
    if (llama_kv_park_debug()) {
        LLAMA_LOG_WARN("%s: seq %d [%u, %u) -> [%u, %u)\n", __func__, seq_id, lo, hi, dst, dst + n);
    }

    park_compact();

    return true;
}

bool llama_kv_cache::seq_parked(llama_seq_id seq_id) const {
    return parked.count(seq_id) > 0;
}

bool llama_kv_cache::can_park() const {
    // shared cells (another cache reads them), a transposed V, or a sliding-window cache, which reuses the cells that fall
    // out of a sequence's window -- another sequence's included -- so it cannot keep cells off limits
    return n_stream == 1 && other == nullptr && v_cells_impl.use_count() == 1 && !v_trans && n_swa == 0;
}

void llama_kv_cache::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    GGML_ASSERT(hparams.n_pos_per_embd() == 1 && "seq_add() is only supported for n_pos_per_embd() == 1");

    auto & cells = v_cells[seq_to_stream[seq_id]];
    auto & head  = v_heads[seq_to_stream[seq_id]];

    if (shift == 0) {
        return;
    }

    uint32_t new_head = cells.size();

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over all cells.
    if (p0 == p1) {
        return;
    }

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            if (cells.pos_add(i, shift)) {
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    // Otherwise we just start the next search from the beginning.
    head = new_head != cells.size() ? new_head : 0;
}

void llama_kv_cache::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    GGML_ASSERT(hparams.n_pos_per_embd() == 1 && "seq_div() is only supported for n_pos_per_embd() == 1");

    auto & cells = v_cells[seq_to_stream[seq_id]];

    if (d == 1) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over the cache.
    if (p0 == p1) {
        return;
    }

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            cells.pos_div(i, d);
        }
    }
}

llama_pos llama_kv_cache::seq_pos_min(llama_seq_id seq_id) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return other->seq_pos_min(seq_id);
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const auto & cells = v_cells[seq_to_stream[seq_id]];

    return cells.seq_pos_min(seq_id);
}

llama_pos llama_kv_cache::seq_pos_max(llama_seq_id seq_id) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return other->seq_pos_max(seq_id);
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const auto & cells = v_cells[seq_to_stream[seq_id]];

    return cells.seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [ctx, buf] : ctxs_bufs) {
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf.get());

        if (hparams.no_alloc) {
            GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) == nullptr);
            ret[buft] += ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft);
        } else if (kv_resident_cells != 0 && get_size() != 0) {
            // wip/kv-residency (TODO #118) A3: a WINDOWED buffer only *commits* its resident window, so report
            // the split instead of the whole reservation (the buffer's size is the VA range).  The geometry is
            // uniform by construction (the ctor declines the window otherwise), so the split is proportional to
            // the cells.  The spill is charged to the host/CPU bucket, which is what makes `--fit` size a
            // context around the window and what makes the memory table show where the KV really lives.
            const size_t bytes  = ggml_backend_buffer_get_size(buf.get());
            const size_t res    = (size_t) std::min<size_t>(kv_resident_cells, get_size());
            const size_t vram   = bytes / get_size() * res + (bytes % get_size()) * res / get_size();
            ret[buft] += vram;
            if (bytes > vram) {
                ret[ggml_backend_cpu_buffer_type()] += bytes - vram;
            }
        } else {
            // GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) != nullptr); // multi_buffer does not have a defined base
            ret[buft] += ggml_backend_buffer_get_size(buf.get());
        }
    }

    return ret;
}

llama_memory_context_ptr llama_kv_cache::init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) {
    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = n_stream == 1 ? balloc.split_simple(n_ubatch) : embd_all ? balloc.split_seq(n_ubatch) : balloc.split_equal(n_ubatch, true, 0);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        auto sinfos = prepare(ubatches);
        if (sinfos.empty()) {
            break;
        }

        return std::make_unique<llama_kv_cache_context>(
                this, std::move(sinfos), std::move(ubatches));
    } while (false);

    return std::make_unique<llama_kv_cache_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_kv_cache::init_full() {
    return std::make_unique<llama_kv_cache_context>(this);
}

llama_memory_context_ptr llama_kv_cache::init_update(llama_context * lctx, bool optimize) {
    GGML_UNUSED(optimize);

    bool do_shift = get_has_shift();

    return std::make_unique<llama_kv_cache_context>(this, lctx, do_shift, std::move(sc_info));
}

llama_kv_cache::slot_info_vec_t llama_kv_cache::prepare(const std::vector<llama_ubatch> & ubatches) {
    llama_kv_cache::slot_info_vec_t res;

    // a batch for a parked sequence (the caller did not seq_unpark() it): parking is given up, so that the attention
    // range (get_n_kv) never stops below cells of a sequence in the batch.  No data move here: a decode may be in flight
    if (!parked.empty()) {
        for (const auto & ubatch : ubatches) {
            for (uint32_t s = 0; s < ubatch.n_seqs_unq && !parked.empty(); ++s) {
                if (parked.count(ubatch.seq_id_unq[s]) > 0) {
                    LLAMA_LOG_WARN("%s: a batch for parked seq %d: parking is given up\n", __func__, ubatch.seq_id_unq[s]);
                    parked.clear();
                }
            }
        }
    }

    struct state_t {
        slot_info sinfo; // slot info for the ubatch

        std::vector<uint32_t> v_heads_old; // old positions of the heads, before placing the ubatch

        std::vector<llama_kv_cells> v_cells; // copy of the old cells, before placing the ubatch
    };

    // remember the old state of the cells so we can restore it in the end
    std::vector<state_t> states;

    bool success = true;

    for (const auto & ubatch : ubatches) {
        // only find a suitable slot for the ubatch. don't modify the cells yet
        const auto sinfo_new = find_slot(ubatch, false);
        if (sinfo_new.empty()) {
            success = false;
            break;
        }

        // remember the position that we found
        res.push_back(sinfo_new);

        // store the old state of the cells in the recovery stack
        {
            state_t state = { sinfo_new, v_heads, {} };

            for (uint32_t s = 0; s < sinfo_new.n_stream(); ++s) {
                auto & cells = v_cells[sinfo_new.strm[s]];

                state.v_cells.push_back(cells.cp(sinfo_new.idxs[s]));
            }

            states.push_back(std::move(state));
        }

        // now emplace the ubatch
        apply_ubatch(sinfo_new, ubatch);
    }

    GGML_ASSERT(!states.empty() || !success);

    // iterate backwards and restore the cells to their original state
    for (auto it = states.rbegin(); it != states.rend(); ++it) {
        const auto & sinfo = it->sinfo;

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            auto & cells = v_cells[sinfo.strm[s]];
            auto & head  = v_heads[sinfo.strm[s]];

            cells.set(sinfo.idxs[s], it->v_cells[s]);
            head = it->v_heads_old[s];
        }
    }

    if (!success) {
        return {};
    }

    return res;
}

bool llama_kv_cache::update(llama_context * lctx, bool do_shift, const stream_copy_info & sc_info_l) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return true;
    }

    bool updated = false;

    auto * sched = lctx->get_sched();

    if (!sc_info_l.empty()) {
        assert(n_stream > 1 && "stream copy should never happen with a single stream");

        llama_synchronize(lctx);

        const size_t n_copy = sc_info_l.ssrc.size();

        for (size_t i = 0; i < n_copy; ++i) {
            const auto ssrc = sc_info_l.ssrc[i];
            const auto sdst = sc_info_l.sdst[i];

            assert(ssrc < n_stream);
            assert(sdst < n_stream);

            LLAMA_LOG_DEBUG("%s: copying KV buffer: stream %d to stream %d\n", __func__, ssrc, sdst);

            assert(ssrc != sdst);

            for (uint32_t il = 0; il < layers.size(); ++il) {
                const auto & layer = layers[il];

                ggml_backend_tensor_copy(layer.k_stream[ssrc], layer.k_stream[sdst]);

                if (layer.v_stream[ssrc]) {
                    ggml_backend_tensor_copy(layer.v_stream[ssrc], layer.v_stream[sdst]);
                }
            }
        }
    }

    if (do_shift) {
        if (!get_can_shift()) {
            GGML_ABORT("The current KV cache / model configuration does not support K-shift");
        }

        LLAMA_LOG_DEBUG("%s: applying K-shift\n", __func__);

        // apply K-shift if needed
        if (hparams.rope_type != LLAMA_ROPE_TYPE_NONE &&
                std::any_of(layers.begin(), layers.end(), [&](const auto & layer) {
                    return hparams.has_rope(layer.il);
                })) {
            ggml_backend_sched_reset(sched);

            auto * res = lctx->get_gf_res_reserve();

            res->reset();

            auto * gf = build_graph_shift(res, lctx);
            if (!ggml_backend_sched_alloc_graph(sched, gf)) {
                LLAMA_LOG_ERROR("%s: failed to allocate compute graph for K-shift\n", __func__);
                return updated;
            }

            res->set_inputs(nullptr);

            if (lctx->graph_compute(gf, false) != GGML_STATUS_SUCCESS) {
                LLAMA_LOG_ERROR("%s: failed to compute K-shift\n", __func__);
                return updated;
            }

            updated = true;
        }

        for (uint32_t s = 0; s < n_stream; ++s) {
            auto & cells = v_cells[s];

            cells.reset_shift();
        }
    }

    return updated;
}

llama_kv_cache::slot_info llama_kv_cache::find_slot(const llama_ubatch & ubatch, bool cont) const {

    if (debug > 0) {
        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
            const auto seq_id = ubatch.seq_id_unq[s];
            const auto stream_id = seq_to_stream[seq_id];
            const auto & cells = v_cells[stream_id];
            const uint32_t head_cur = v_heads[stream_id];

            LLAMA_LOG_DEBUG("%s: stream[%d], n = %5d, used = %5d, head = %5d, size = %5d, n_swa = %5d\n",
                    __func__, stream_id, cells.used_max_p1(), cells.get_used(), head_cur, get_size(), n_swa);

            if ((debug == 2 && n_swa > 0) || debug > 2) {
                std::string ss;
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    if (cells.is_empty(i)) {
                        ss += '.';
                    } else {
                        assert(cells.seq_count(i) >= 1);

                        if (cells.seq_count(i) == 1) {
                            ss += std::to_string(cells.seq_get(i));
                        } else {
                            ss += 'M';
                        }
                    }
                    if (i%256 == 255) {
                        ss += " *";
                        ss += '\n';
                    }
                }
                LLAMA_LOG_DEBUG("\n%s\n", ss.c_str());
            }

            if ((debug == 2 && n_swa > 0) || debug > 2) {
                std::string ss;
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    std::string cur;
                    if (cells.is_empty(i)) {
                        cur = '.';
                    } else {
                        cur = std::to_string(cells.pos_get(i));
                    }
                    const int n = cur.size();
                    for (int j = 0; j < 5 - n; ++j) {
                        cur += ' ';
                    }
                    ss += cur;
                    if (i%256 == 255) {
                        ss += " *";
                    }
                    if (i%64 == 63) {
                        ss += '\n';
                    }
                }
                LLAMA_LOG_DEBUG("\n%s\n", ss.c_str());
            }

            for (int s_seq = 0; s_seq < LLAMA_MAX_SEQ; ++s_seq) {
                if (cells.seq_pos_min(s_seq) < 0) {
                    continue;
                }

                LLAMA_LOG_DEBUG("%s_seq: stream[%d] min[%d] = %5d, max[%d] = %5d\n", __func__, stream_id, s_seq, cells.seq_pos_min(s_seq), s_seq, cells.seq_pos_max(s_seq));
            }
        }
    }

    uint32_t n_tokens = ubatch.n_tokens;
    uint32_t n_seqs   = 1;

    if (n_stream > 1) {
        GGML_ASSERT(n_tokens % ubatch.n_seqs_unq == 0);

        n_seqs   = ubatch.n_seqs_unq;
        n_tokens = n_tokens / n_seqs;
    }

    slot_info res = {
        /*.s0   =*/ LLAMA_MAX_SEQ,
        /*.s1   =*/ 0,
        /*.strm =*/ { },
        /*.idxs =*/ { },
    };

    res.resize(n_seqs);

    for (uint32_t s = 0; s < n_seqs; ++s) {
        const auto seq_id = ubatch.seq_id_unq[s];

        if (n_stream > 1) {
            GGML_ASSERT(ubatch.n_seq_id[s*n_tokens]    == 1);
            GGML_ASSERT(ubatch.seq_id  [s*n_tokens][0] == seq_id);
        }

        res.s0 = std::min<uint32_t>(res.s0, seq_to_stream[seq_id]);
        res.s1 = std::max<uint32_t>(res.s1, seq_to_stream[seq_id]);

        res.strm[s] = seq_to_stream[seq_id];
        res.idxs[s].reserve(n_tokens);

        const auto & cells = v_cells[seq_to_stream[seq_id]];

        uint32_t head_cur = v_heads[seq_to_stream[seq_id]];

        // with parked sequences (seq_park), the search covers only the cells below them
        const uint32_t n_cells = parked.empty() ? cells.size() : park_bound();

        // if we have enough unused cells before the current head ->
        //   better to start searching from the beginning of the cache, hoping to fill it
        if (head_cur > cells.get_used() + 2*n_tokens || head_cur >= n_cells) {
            head_cur = 0;
        }

        if (n_tokens > n_cells) {
            if (n_cells == cells.size()) {
                LLAMA_LOG_ERROR("%s: n_tokens = %d > size = %u\n", __func__, n_tokens, cells.size());
            }
            return { };
        }

        uint32_t n_tested = 0;

        // for continuous slots, we test that all tokens in the ubatch fit, starting from the current head
        // for non-continuous slots, we test the tokens one by one
        const uint32_t n_test = cont ? n_tokens : 1;

        while (true) {
            if (head_cur + n_test > n_cells) {
                n_tested += n_cells - head_cur;
                head_cur = 0;
                continue;
            }

            for (uint32_t i = 0; i < n_test; i++) {
                const auto idx = head_cur;

                head_cur++;
                n_tested++;

                //const llama_pos    pos    = ubatch.pos[i];
                //const llama_seq_id seq_id = ubatch.seq_id[i][0];

                // can we use this cell? either:
                //  - the cell is empty
                //  - the cell is occupied only by one sequence:
                //    - (disabled) mask causally, if the sequence is the same as the one we are inserting
                //    - mask SWA, using current max pos for that sequence in the cache
                //                always insert in the cell with minimum pos
                bool can_use = cells.is_empty(idx);

                if (!can_use && cells.seq_count(idx) == 1) {
                    const llama_pos pos_cell = cells.pos_get(idx);

                    // (disabled) causal mask
                    // note: it's better to purge any "future" tokens beforehand
                    //if (cells.seq_has(idx, seq_id)) {
                    //    can_use = pos_cell >= pos;
                    //}

                    if (!can_use) {
                        const llama_seq_id seq_id_cell = cells.seq_get(idx);

                        // SWA mask
                        if (llama_hparams::is_masked_swa(n_swa, swa_type, pos_cell, cells.seq_pos_max(seq_id_cell) + 1)) {
                            can_use = true;
                        }
                    }
                }

                if (can_use) {
                    res.idxs[s].push_back(idx);
                } else {
                    if (cont) {
                        break;
                    }
                }
            }

            if (res.idxs[s].size() == n_tokens) {
                break;
            }

            if (cont) {
                res.idxs[s].clear();
            }

            if (n_tested >= n_cells) {
                //LLAMA_LOG_ERROR("%s: failed to find a slot for %d tokens\n", __func__, n_tokens);
                return { };
            }
        }

        // we didn't find a suitable slot - return empty result
        if (res.idxs[s].size() < n_tokens) {
            return { };
        }
    }

    assert(res.s1 >= res.s0);

    return res;
}

void llama_kv_cache::apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    // keep track of the max sequence position that we would overwrite with this ubatch
    // for non-SWA cache, this would be always empty
    llama_seq_id seq_pos_max_rm[LLAMA_MAX_SEQ];
    for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        seq_pos_max_rm[s] = -1;
    }

    assert(ubatch.n_tokens == sinfo.n_stream()*sinfo.size());

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        for (uint32_t ii = 0; ii < sinfo.size(); ++ii) {
            const uint32_t i = s*sinfo.size() + ii;

            auto & cells = v_cells[sinfo.strm[s]];

            const auto idx = sinfo.idxs[s][ii];

            if (!cells.is_empty(idx)) {
                assert(cells.seq_count(idx) == 1);

                const llama_seq_id seq_id = cells.seq_get(idx);
                const llama_pos    pos    = cells.pos_get(idx);

                seq_pos_max_rm[seq_id] = std::max(seq_pos_max_rm[seq_id], pos);

                cells.rm(idx);
            }

            cells.pos_set(idx, ubatch.pos[i]);

            if (ubatch.is_pos_2d() || ubatch.token || hparams.ple_n_heads > 0) {
                llama_kv_cell_ext ext;

                if (ubatch.is_pos_2d()) {
                    ext.x = ubatch.pos[i + ubatch.n_tokens*2];
                    ext.y = ubatch.pos[i + ubatch.n_tokens];
                }

                const bool is_embd = !ubatch.token || (ubatch.is_mixed() && ubatch.type[i]);

                if (!is_embd) {
                    ext.tok = ubatch.token[i];
                } else if (hparams.ple_n_heads > 0) {
                    // embd batch (multimodal input) has no token ids, need to pad it with the correct ID for PLE layers
                    // TODO @ngxson : check if we can do the same as gemma 3n / gemma 4
                    ext.tok = hparams.ple_embd_token;
                }

                cells.ext_set(idx, ext);
            }

            for (int32_t s_seq = 0; s_seq < ubatch.n_seq_id[i]; s_seq++) {
                cells.seq_add(idx, ubatch.seq_id[i][s_seq]);
            }
        }
    }

    // note: we want to preserve the invariant that all positions between [pos_min, pos_max] for each sequence
    //       will be present in the cache. so we have to purge any position which is less than those we would overwrite
    //       ref: https://github.com/ggml-org/llama.cpp/pull/13746#issuecomment-2916057092
    for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        if (seq_pos_max_rm[s] == -1) {
            continue;
        }

        GGML_ASSERT(s < seq_to_stream.size());

        auto & cells = v_cells[seq_to_stream[s]];

        if (cells.seq_pos_min(s) <= seq_pos_max_rm[s]) {
            LLAMA_LOG_DEBUG("%s: purging positions [%d, %d] of sequence %d from KV cache\n",
                    __func__, cells.seq_pos_min(s), seq_pos_max_rm[s], s);

            seq_rm(s, cells.seq_pos_min(s), seq_pos_max_rm[s] + 1);
        }
    }

    // move the head at the end of the slot
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        auto & head = v_heads[sinfo.strm[s]];

        head = sinfo.idxs[s].back() + 1;
    }
}

bool llama_kv_cache::get_can_shift() const {
    // Step35 uses per-layer RoPE dims; K-shift assumes a single global n_rot.
    if (model.arch == LLM_ARCH_STEP35) {
        return false;
    }
    if (hparams.n_pos_per_embd() > 1) {
        return false;
    }
    return true;
}

uint32_t llama_kv_cache::get_size() const {
    const auto & cells = v_cells[seq_to_stream[0]];

    return cells.size();
}

uint32_t llama_kv_cache::get_n_seq_max() const {
    return n_seq_max;
}

uint32_t llama_kv_cache::get_n_stream() const {
    return n_stream;
}

bool llama_kv_cache::get_has_shift() const {
    bool result = false;

    for (uint32_t s = 0; s < n_stream; ++s) {
        result |= v_cells[s].get_has_shift();
    }

    return result;
}

ggml_type llama_kv_cache::type_k() const {
    return layers[0].k->type;
}

ggml_type llama_kv_cache::type_v() const {
    return layers[0].v->type;
}

std::vector<uint32_t> llama_kv_cache::get_layer_ids() const {
    std::vector<uint32_t> res;
    res.reserve(layers.size());

    for (const auto & layer : layers) {
        res.push_back(layer.il);
    }

    return res;
}

ggml_tensor * llama_kv_cache::get_k_storage(int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);

    return layers[ikv].k;
}

const llama_kv_cells & llama_kv_cache::get_cells(llama_seq_id seq_id) const {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    return v_cells[seq_to_stream[seq_id]];
}

uint32_t llama_kv_cache::get_stream(llama_seq_id seq_id) const {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    return seq_to_stream[seq_id];
}

void llama_kv_cache::grow_regions(uint32_t n_kv) const {
    if (!kv_region || ctxs_bufs.empty()) {
        return;
    }
    llama_kv_region_grow_fn fn = llama_kv_region_grow_hook();
    if (fn == nullptr) {
        return;
    }
    for (const auto & [_, buf] : ctxs_bufs) {
        if (fn(buf.get(), n_kv) < 0) {
            // A refused grow is not something to continue past: the graph this batch would build reads cells
            // above the mapped extent, and that faults the GPU (it does not return an error).  The backend has
            // already logged the numbers; refuse loudly instead of hanging the card.
            GGML_ABORT("%s: the KV region cannot be grown to %u cells and a batch that needs more than the "
                       "mapped extent would fault the GPU.  Lower -c, raise LLAMA_KV_REGION_MAP_MAX_MIB, or "
                       "unset LLAMA_KV_REGION.\n", __func__, n_kv);
        }
    }
}

uint32_t llama_kv_cache::get_n_kv(const slot_info & sinfo, uint32_t n_tokens) const {
    uint32_t result = 0;

    // pad the n_kv value so that the graph remains constant across batches and can be reused
    // note: this also helps some backends with performance (f.ex https://github.com/ggml-org/llama.cpp/pull/16812#issuecomment-3455112220)
    // LLAMA_KV_N_PAD_MIN raises the 256-cell floor: n_kv - and with it the graph layout - then stays constant for longer,
    // so fewer captured graphs are invalidated as the context grows (-sm tensor splits each token into ~100 graphs)
    static const uint32_t n_pad_min = getenv("LLAMA_KV_N_PAD_MIN") ? (uint32_t) atoi(getenv("LLAMA_KV_N_PAD_MIN")) : 256u;
    // LLAMA_KV_N_PAD_PREFILL: a coarser pad for prompt-sized ubatches (> 64 tokens) once the cache is past
    // LLAMA_KV_N_PAD_PREFILL_FROM cells (default 4096; compared with the decode-padded occupancy).  Every prompt ubatch grows the cache by its own size, so with the
    // decode pad n_kv - and the graph - changes at every ubatch and the whole graph is rebuilt (split state, allocation);
    // a pad several ubatches wide lets consecutive ubatches reuse one graph.  Below the threshold the decode pad stays, so
    // a model's small-context shortcuts keyed on n_kv (qwen4exp's dense attention for the first ~2051 tokens) still apply.
    // Decode and verify batches keep the decode pad (a coarse pad there slows every step and changes its numerics).
    // Default: 8192 on a multi-device split, 0 (off) on a single device; the value is rounded up to a power of two.
    // Not for training (its graphs assert n_kv == n_tokens).
    static const uint32_t n_pad_prefill_from = getenv("LLAMA_KV_N_PAD_PREFILL_FROM") ? (uint32_t) atoi(getenv("LLAMA_KV_N_PAD_PREFILL_FROM")) : 4096u;
    const uint32_t n_pad_cur = std::max(n_pad, n_pad_min);
    const bool     prefill   = n_tokens > 64 && n_pad_prefill > n_pad_cur;

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const auto & cells = v_cells[sinfo.strm[s]];

        // parked sequences (seq_park) sit above every sequence that is decoded, so the range stops below them
        const uint32_t used = parked.empty() ? cells.used_max_p1() : cells.used_max_p1_below(park_bound());
        const uint32_t pad  = prefill && GGML_PAD(used, n_pad_cur) > n_pad_prefill_from ? n_pad_prefill : n_pad_cur;

        result = std::max(std::min(cells.size(), std::max(pad, GGML_PAD(used, pad))), result);
    }

    return result;
}

uint32_t llama_kv_cache::get_kv_split_n_tail(uint32_t n_kv) const {
    const uint32_t w = llama_kv_region_resident_cells_requested();
    if (w == 0 || w == UINT32_MAX) {
        return 0; // default: no split, byte-identical to the single-base read (including the OOM)
    }
    if (v_trans) {
        return 0; // a transposed V cache has its cell axis on dim 0: not served, so no split
    }
    if (n_kv <= w) {
        return 0; // everything is resident: no tail to read from a second base
    }
    // the tail is rounded DOWN to the granularity every FA kernel tile satisfies, so a tile never
    // straddles the boundary and the split is a base selection rather than a numeric change
    return ((n_kv - w)/GGML_FA_KV_SPLIT_GRAN)*GGML_FA_KV_SPLIT_GRAN;
}

// wip/kv-residency (TODO #118) A1 NEGATIVE CONTROL.  `LLAMA_KV_SPLIT_POISON_CELLS=N` (default 0) shifts
// the TAIL view by N cells, so every kernel tile below n_tail reads the wrong rows.  A1's oracle is an
// equality (both segments are views of one tensor, so a wrong base choice is invisible to it), and this
// is what makes the equality meaningful: with the poison on the text MUST change, which is the only
// evidence that the kernels actually read the second base.  Never a shipping knob.
static uint32_t kv_split_poison_cells() {
    static const uint32_t v = []() {
        const char * e = getenv("LLAMA_KV_SPLIT_POISON_CELLS");
        return e != nullptr ? (uint32_t) strtoul(e, nullptr, 10) : 0u;
    }();
    return v;
}

ggml_tensor * llama_kv_cache::get_k_tail(ggml_context * ctx, int32_t il, uint32_t n_tail, const slot_info & sinfo) const {
    return get_k_cells(ctx, il, n_tail, kv_split_poison_cells(), sinfo);
}

ggml_tensor * llama_kv_cache::get_v_tail(ggml_context * ctx, int32_t il, uint32_t n_tail, const slot_info & sinfo) const {
    return get_v_cells(ctx, il, n_tail, kv_split_poison_cells(), sinfo);
}

ggml_tensor * llama_kv_cache::get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    return get_k_cells(ctx, il, n_kv, 0, sinfo);
}

ggml_tensor * llama_kv_cache::get_k_cells(ggml_context * ctx, int32_t il, uint32_t n_kv, uint32_t cell_off, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * k = layers[ikv].k;

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_k_gqa = k->ne[0];

    assert(n_embd_k_gqa == hparams.n_embd_k_gqa(il));

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;

    return ggml_view_4d(ctx, k,
            hparams.n_embd_head_k(il), hparams.n_head_kv(il), n_kv, ns,
            ggml_row_size(k->type, hparams.n_embd_head_k(il)),
            ggml_row_size(k->type, n_embd_k_gqa),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size)*sinfo.s0 +
            ggml_row_size(k->type, n_embd_k_gqa)*(int64_t) cell_off);
}

ggml_tensor * llama_kv_cache::get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    return get_v_cells(ctx, il, n_kv, 0, sinfo);
}

ggml_tensor * llama_kv_cache::get_v_cells(ggml_context * ctx, int32_t il, uint32_t n_kv, uint32_t cell_off, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * v = layers[ikv].v;

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_v_gqa = v->ne[0];

    // [TAG_V_CACHE_VARIABLE]
    assert(n_embd_v_gqa >= hparams.n_embd_v_gqa(il));

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;

    if (!v_trans) {
        // note: v->nb[1] <= v->nb[2]
        return ggml_view_4d(ctx, v,
                hparams.n_embd_head_v(il), hparams.n_head_kv(il), n_kv, ns,
                ggml_row_size(v->type, hparams.n_embd_head_v(il)),          // v->nb[1]
                ggml_row_size(v->type, n_embd_v_gqa),                   // v->nb[2]
                ggml_row_size(v->type, n_embd_v_gqa*kv_size),           // v->nb[3]
                ggml_row_size(v->type, n_embd_v_gqa*kv_size)*sinfo.s0 +
                ggml_row_size(v->type, n_embd_v_gqa)*(int64_t) cell_off);
    }

    // note: v->nb[1] > v->nb[2]
    return ggml_view_4d(ctx, v,
            n_kv, hparams.n_head_kv(il), hparams.n_embd_head_v(il), ns,
            ggml_row_size(v->type, kv_size*hparams.n_embd_head_v(il)),  // v->nb[1]
            ggml_row_size(v->type, kv_size),                        // v->nb[2]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa),           // v->nb[3]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa)*sinfo.s0 +
            ggml_row_size(v->type, kv_size*hparams.n_embd_head_v(il))*(int64_t) cell_off);
}

ggml_tensor * llama_kv_cache::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const {
    GGML_UNUSED(sinfo);

    const int32_t ikv = map_layer_ids.at(il);

    ggml_tensor * k = layers[ikv].k;

    const int64_t n_embd_head = k_cur->ne[0];
    const int64_t n_head      = k_cur->ne[1];
    const int64_t n_tokens    = k_cur->ne[2];

    const int64_t n_embd_gqa = n_embd_head*n_head;

    // we can merge dims 0 and 1
    // TODO: add ggml helper function for this?
    GGML_ASSERT(ggml_row_size(k_cur->type, n_embd_head) == k_cur->nb[1]);

    k_cur = ggml_view_2d(ctx, k_cur, n_embd_gqa, n_tokens, k_cur->nb[2], 0);

    const int64_t n_stream_l = k->ne[2];

    if (n_stream_l > 1) {
        const int64_t kv_size = get_size();

        assert(n_embd_gqa == k->ne[0]);
        assert(kv_size    == k->ne[1]);

        // merge the buffer across all streams because the idxs are global
        k = ggml_reshape_2d(ctx, k, n_embd_gqa, kv_size*n_stream_l);
    }

    // store the current K values into the cache
    return ggml_set_rows(ctx, k, k_cur, k_idxs);
}

ggml_tensor * llama_kv_cache::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const {
    GGML_UNUSED(sinfo);

    const int32_t ikv = map_layer_ids.at(il);

    auto * v = layers[ikv].v;

    const int64_t n_embd_head = v_cur->ne[0];
    const int64_t n_head      = v_cur->ne[1];
    const int64_t n_tokens    = v_cur->ne[2];

    const int64_t n_embd_gqa = n_embd_head*n_head;

    // we can merge dims 0 and 1
    GGML_ASSERT(ggml_row_size(v_cur->type, n_embd_head) == v_cur->nb[1]);

    const int64_t n_stream_l = v->ne[2];

    // take this branch when FA is enabled (the V cache is not transposed)
    if (!v_trans) {
        v_cur = ggml_view_2d(ctx, v_cur, n_embd_gqa, n_tokens, v_cur->nb[2], 0);

        if (n_stream_l > 1) {
            const int64_t kv_size = get_size();

            assert(n_embd_gqa == v->ne[0]);
            assert(kv_size    == v->ne[1]);

            // merge the buffer across all streams because the idxs are global
            v = ggml_reshape_2d(ctx, v, n_embd_gqa, kv_size*n_stream_l);
        }

        return ggml_set_rows(ctx, v, v_cur, v_idxs);
    }

    if (ggml_row_size(v_cur->type, n_embd_gqa) == v_cur->nb[2]) {
        // we can merge dims 0, 1 and 2
        v_cur = ggml_reshape_2d(ctx, v_cur, n_embd_gqa, n_tokens);
    } else {
        // otherwise -> make a copy to get contiguous data
        v_cur = ggml_cont_2d   (ctx, v_cur, n_embd_gqa, n_tokens);
    }

    // [TAG_V_CACHE_VARIABLE]
    if (n_embd_gqa < v->ne[0]) {
        v_cur = ggml_pad(ctx, v_cur, v->ne[0] - n_embd_gqa, 0, 0, 0);
    }

    // in this branch the v_idxs are constructed in such a way that each row is a single head element
    ggml_tensor * v_view = ggml_reshape_2d(ctx, v, 1, ggml_nelements(v));

    v_cur = ggml_reshape_2d(ctx, v_cur, 1, ggml_nelements(v_cur));

    return ggml_set_rows(ctx, v_view, v_cur, v_idxs);
}

ggml_tensor * llama_kv_cache::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    ggml_tensor * k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);

    ggml_set_input(k_idxs);
    ggml_set_name(k_idxs, "attn_inp_k_idxs");

    return k_idxs;
}

ggml_tensor * llama_kv_cache::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    ggml_tensor * v_idxs;

    if (!v_trans) {
        v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    } else {
        v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens*hparams.n_embd_v_gqa_max());
    }

    ggml_set_input(v_idxs);
    ggml_set_name(v_idxs, "attn_inp_v_idxs");

    return v_idxs;
}

ggml_tensor * llama_kv_cache::build_input_k_rot(ggml_context * ctx) const {
    ggml_tensor * res = nullptr;

    if (n_rot_k) {
        res = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rot_k, n_rot_k);
        ggml_set_input(res);
        ggml_set_name(res, "attn_inp_k_rot");
    }

    return res;
}

ggml_tensor * llama_kv_cache::build_input_v_rot(ggml_context * ctx) const {
    ggml_tensor * res = nullptr;

    if (n_rot_v) {
        res = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rot_v, n_rot_v);
        ggml_set_input(res);
        ggml_set_name(res, "attn_inp_v_rot");
    }

    return res;
}

void llama_kv_cache::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const int64_t offs = sinfo.strm[s]*get_size();

        for (uint32_t i = 0; i < sinfo.size(); ++i) {
            data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
        }
    }
}

void llama_kv_cache::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    if (!v_trans) {
        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const int64_t offs = sinfo.strm[s]*get_size();

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
            }
        }
    } else {
        // note: the V cache is transposed when not using flash attention
        const int64_t kv_size = get_size();

        const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa_max();

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const int64_t offs = sinfo.strm[s]*kv_size*n_embd_v_gqa;

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    data[s*sinfo.size()*n_embd_v_gqa + i*n_embd_v_gqa + j] = offs + j*kv_size + sinfo.idxs[s][i];
                }
            }
        }
    }
}

void llama_kv_cache::set_input_k_shift(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    int32_t * data = (int32_t *) dst->data;

    for (uint32_t s = 0; s < n_stream; ++s) {
        const auto & cells = v_cells[s];

        for (uint32_t i = 0; i < cells.size(); ++i) {
            data[s*cells.size() + i] = cells.is_empty(i) ? 0 : cells.get_shift(i);
        }
    }
}

struct args_set_input_kq_mask {
    const llama_hparams & hparams;
    const llama_ubatch  * ubatch;

    const std::vector<llama_kv_cells> & v_cells;
    const std::vector<uint32_t>       & seq_to_stream;

    uint32_t       n_swa;
    llama_swa_type swa_type;

    int64_t n_kv;
    int64_t n_stream;
    int64_t n_tps;
};

template<typename T, bool causal, bool swa, bool is_2d, bool alibi>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
  //const auto & hparams = args.hparams;
    const auto & ubatch  = args.ubatch;

    const auto & v_cells       = args.v_cells;
    const auto & seq_to_stream = args.seq_to_stream;

    const uint32_t       n_swa    = args.n_swa;
    const llama_swa_type swa_type = args.swa_type;

    const int64_t n_kv     = args.n_kv;
    const int64_t n_stream = args.n_stream;
    const int64_t n_tps    = args.n_tps;

    const T mask_keep = llama_cast<T>(0.0f);
    const T mask_drop = llama_cast<T>(-INFINITY);

    // the min position in the batch for each sequence
    llama_pos seq_pos_min[LLAMA_MAX_SEQ];
    std::fill(seq_pos_min, seq_pos_min + LLAMA_MAX_SEQ, INT32_MAX);

    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        const llama_seq_id seq_id = ubatch->seq_id[i][0];

        seq_pos_min[seq_id] = std::min(seq_pos_min[seq_id], ubatch->pos[i]);
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        // bookkeeping of the KQ mask cells that could change for other tokens of the same sequence
        std::unordered_map<llama_seq_id, uint32_t>              seq_srct;
        std::unordered_map<llama_seq_id, std::vector<uint32_t>> seq_idxs;

        for (uint32_t ii = 0; ii < n_tps; ++ii) {
            const uint32_t i = s*n_tps + ii;

            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            const auto & cells = v_cells.at(seq_to_stream[seq_id]);

                  llama_pos p0 = -1;
            const llama_pos p1 = ubatch->pos[i];

            // for M-RoPE
            const llama_pos p1_x = is_2d ? ubatch->pos[i + ubatch->n_tokens*2] : 0;
            const llama_pos p1_y = is_2d ? ubatch->pos[i + ubatch->n_tokens]   : 0;

            const uint64_t idst = n_kv*i;

            // for tokens of the same sequence, the mask is mostly the same, so we can reuse it
            // the only cells that could change are the ones that are with similar positions as the
            //   ones in the batch (i.e. due to causal masking, SWA, etc.)
            // keep track of those cells and shortcut the loop to save time
            // note: this optimization is not compatible with Alibi position encoding
            // ref:  https://github.com/ggml-org/llama.cpp/pull/18842
            bool prev = false;

            auto & idxs = seq_idxs[seq_id];

            if (!alibi) {
                if (seq_srct.find(seq_id) != seq_srct.end()) {
                    const uint32_t srct = seq_srct[seq_id];

                    const uint64_t idst_prev = n_kv*srct;

                    std::copy(data + idst_prev, data + idst_prev + n_kv, data + idst);

                    prev = true;
                } else {
                    idxs.clear();
                    idxs.reserve(ubatch->n_tokens + n_swa + 32);

                    seq_srct[seq_id] = i;
                }
            }

            for (uint32_t jj = 0; jj < n_kv; ++jj) {
                uint32_t j = jj;

                // we have an exiting mask for this sequence -> update just seq_idxs
                if (!alibi) {
                    if (prev) {
                        if (jj >= idxs.size()) {
                            break;
                        }

                        j = idxs[jj];
                    }
                }

                if (cells.is_empty(j)) {
                    goto skip;
                }

                // mask the token if not the same sequence
                if (!cells.seq_has(j, seq_id)) {
                    goto skip;
                }

                p0 = cells.pos_get(j);

                if (!alibi) {
                    if (!prev) {
                        // record all cells for which: p0 >= seq_pos_min[seq_id] - n_swa - 32
                        if (p0 + (int32_t) (n_swa + 32) >= seq_pos_min[seq_id]) {
                            idxs.push_back(j);
                        }
                    }
                }

                if (causal) {
                    // mask future tokens
                    if (p0 > p1) {
                        goto skip;
                    }

                    // M-RoPE causal mask
                    if (is_2d) {
                        if (p0 == p1) {
                            const auto & p0_ext = cells.ext_get(j);

                            if (p0_ext.is_2d_gt(p1_x, p1_y)) {
                                goto skip;
                            }
                        }
                    }
                }

                // apply SWA if any
                if (swa) {
                    // see llama_non_causal_type
                    const bool in_span = !causal && args.hparams.non_causal_type == LLAMA_NON_CAUSAL_TYPE_SWA_FULL && p0 >= seq_pos_min[seq_id];
                    if (!in_span && llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)) {
                        goto skip;
                    }
                }

                if (alibi) {
                    data[idst + j] = llama_cast<T>(static_cast<float>(-std::abs(p0 - p1)));
                } else {
                    data[idst + j] = mask_keep;
                }

                continue;
skip:
                data[idst + j] = mask_drop;
            }
        }
    }
}

template<typename T, bool causal, bool swa, bool is_2d>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool alibi = args.hparams.use_alibi;
    if (alibi) {
        set_input_kq_mask_impl<T, causal, swa, is_2d, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, swa, is_2d, false>(args, data);
    }
}

template<typename T, bool causal, bool swa>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool is_2d = args.ubatch->is_pos_2d();
    if (is_2d) {
        set_input_kq_mask_impl<T, causal, swa, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, swa, false>(args, data);
    }
}

template<typename T, bool causal>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool swa = args.swa_type != LLAMA_SWA_TYPE_NONE;
    if (swa) {
        set_input_kq_mask_impl<T, causal, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, false>(args, data);
    }
}

template<typename T>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data, bool causal_attn) {
    if (causal_attn) {
        set_input_kq_mask_impl<T, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, false>(args, data);
    }
}

void llama_kv_cache::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    const uint32_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const int64_t n_kv     = dst->ne[0];
    const int64_t n_stream_l = dst->ne[3]; // num streams in the current ubatch

    GGML_ASSERT(n_tokens%n_stream_l == 0);

    // n_tps == n_tokens_per_stream
    const int64_t n_tps = n_tokens/n_stream_l;

    // see llama_non_causal_type
    // only the SWA cache (or the SWA layers of a single cache) become non-causal
    if (!causal_attn && hparams.non_causal_type == LLAMA_NON_CAUSAL_TYPE_SWA_ONLY) {
        causal_attn = swa_type == LLAMA_SWA_TYPE_NONE;
    }

    //const int64_t t_start = ggml_time_us();

    const args_set_input_kq_mask args = {
        /*.hparams          =*/ hparams,
        /*.ubatch           =*/ ubatch,
        /*.v_cells          =*/ v_cells,
        /*.seq_to_stream    =*/ seq_to_stream,
        /*.n_swa            =*/ n_swa,
        /*.swa_type         =*/ swa_type,
        /*.n_kv             =*/ n_kv,
        /*.n_stream         =*/ n_stream_l,
        /*.n_tps            =*/ n_tps,
    };

    if (dst->type == GGML_TYPE_F16) {
        set_input_kq_mask_impl<ggml_fp16_t>(args, (ggml_fp16_t *) dst->data, causal_attn);
    } else {
        set_input_kq_mask_impl<float>(args, (float *) dst->data, causal_attn);
    }

    //const int64_t t_end = ggml_time_us();

    //LLAMA_LOG_ERROR("%s: kq mask time: %0.3f ms\n", __func__, (t_end - t_start)/1000.0);
}

// V3 derived kq mask ==========================================================================
//
// The packed mask costs n_kv x n_tps F16 per GPU *and* the same in host memory, and it is rewritten
// on every ubatch.  The derived form replaces it with two tiny per-token bounds and one per-cell
// position, and lets the flash attention kernel compute the same visibility on the fly:
//
//   visible(cell j, token i)  <=>  cell_pos[j] >= tok_lo[i] && cell_pos[j] <= tok_hi[i]
//
// cell_pos is INT32_MIN for a cell that is empty or belongs to another sequence (which is exactly
// what "not visible" needs: INT32_MIN fails the >= lo test for every lo >= 0), and otherwise the
// cell's position.  tok_hi is the causal bound (or INT32_MAX when not causal) and tok_lo is the
// visibility floor: 0, or the sliding-window floor, or (for LLAMA_NON_CAUSAL_TYPE_SWA_FULL) the
// minimum batch position of the sequence.
//
// The predicate below is the *only* shape the phase-1 host oracle proved value-identical to the
// packed fill (cell-by-cell, including both SWA cache variants and the non-causal path), so it
// deliberately stays narrow: one sequence, one stream, prefill-sized batches, no alibi, no 2-D
// (M-RoPE) positions with a live ext clause.  Everything else keeps the packed mask.  Note that
// this is not a correctness escape hatch - the host oracle has to be re-run on every model when
// this predicate is extended.
// Several sequences in one batch (`--kv-unified` with more than one slot: two slots verifying together, or one slot's
// prefill next to another's decode) derive their mask with the same predicate: sequence k of the ubatch (seq_idx) has
// its cells and its tokens' window shifted by k*KQ_DERIVED_SEQ_STRIDE, so a token's [tok_lo, tok_hi] never reaches
// another sequence's cells and the kernels need no change.  This holds when every cell carries at most one of the
// batch's sequences (a cell copied to two of them with llama_memory_seq_cp would have to be visible to both ranges)
// and every position stays below the stride; 2^23 positions per sequence keep all LLAMA_MAX_SEQ ranges inside int32.
static constexpr llama_pos KQ_DERIVED_SEQ_STRIDE = 1 << 23;
static_assert((int64_t) KQ_DERIVED_SEQ_STRIDE*LLAMA_MAX_SEQ - 1 <= INT32_MAX, "the shifted ranges must fit in int32");

bool llama_kv_cache::kq_mask_derived_multiseq_enabled() {
    static const bool enabled = [] {
        const char * e = getenv("LLAMA_KQ_MASK_DERIVED_MULTISEQ");
        return e == nullptr || atoi(e) != 0;
    }();
    return enabled;
}

bool llama_kv_cache::kq_mask_derivable_seqs(const llama_seq_id * seqs, uint32_t n_seqs, llama_pos pos_max) const {
    if (!kq_mask_derived_multiseq_enabled() || n_stream != 1 || n_seqs > LLAMA_MAX_SEQ || pos_max >= KQ_DERIVED_SEQ_STRIDE) {
        return false;
    }

    const auto & cells = v_cells[0];

    llama_kv_cells::seq_set_t in_batch;
    for (uint32_t k = 0; k < n_seqs; ++k) {
        if (seqs[k] < 0 || seqs[k] >= LLAMA_MAX_SEQ || cells.seq_pos_max(seqs[k]) >= KQ_DERIVED_SEQ_STRIDE) {
            return false;
        }
        in_batch.set(seqs[k]);
    }

    // no cell carries two sequences at all (the common case: nothing was copied with llama_memory_seq_cp)
    if (cells.get_n_multi_seq() == 0) {
        return true;
    }

    // O(used cells): about 2 ms at 245760 cells, so only when some cell is shared
    const uint32_t n_used = cells.used_max_p1();
    for (uint32_t j = 0; j < n_used; ++j) {
        if ((cells.seq_get_all(j) & in_batch).count() > 1) {
            return false;
        }
    }

    return true;
}

bool llama_kv_cache::kq_mask_derivable(const llama_ubatch & ubatch) const {
    // alibi writes a distance into the mask, that is a value, not a predicate
    if (hparams.use_alibi) {
        return false;
    }

    // a single KV stream too: build_attn_mha derives the attention stream count from the KV tensor
    // (k->ne[3] == the cache's n_stream), so a multi-stream cache (n_seq_max > 1 without kv_unified)
    // cannot use the derived form even when the ubatch holds a single sequence.  Without this the
    // derived tok_lo is built for one stream while Q is reshaped for n_seq_max streams and the
    // shape assert in ggml_flash_attn_ext_add_kq_derived fires at context creation.
    if (n_stream != 1) {
        return false;
    }

    // prefill-shaped batches only: below this the packed mask is a few hundred KiB, the launcher may
    // pick the vec/tile kernels and decode has to stay on the packed path
    if (ubatch.n_tokens <= 8) {
        return false;
    }

    // one sequence: every cell is either "own" or "foreign", which is what the INT32_MIN sentinel encodes.
    // Several sequences (one KV stream, as checked above): each gets its own position range, see
    // kq_mask_derivable_seqs - every token in one sequence, no cell shared between two of them
    if (ubatch.n_seqs_unq != 1) {
        if (ubatch.n_tokens <= KQ_DERIVED_MULTISEQ_MIN_TOKENS) {
            return false;
        }
        llama_pos pos_max = 0;
        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            if (ubatch.n_seq_id[i] != 1) {
                return false;
            }
            pos_max = std::max(pos_max, ubatch.pos[i]);
        }
        if (!kq_mask_derivable_seqs(ubatch.seq_id_unq, ubatch.n_seqs_unq, pos_max)) {
            return false;
        }
    }

    // M-RoPE: the derived form carries no 2-D ext clause, so it is only usable while the positions
    // are degenerate (x == y == pos).  qwen35 is IMROPE, so this is hit even for text input.
    if (ubatch.is_pos_2d()) {
        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            const llama_pos p = ubatch.pos[i];

            if (p != ubatch.pos[i + ubatch.n_tokens    ] ||
                p != ubatch.pos[i + 2*ubatch.n_tokens]) {
                return false;
            }
        }

        const auto & cells = v_cells.at(seq_to_stream[ubatch.seq_id[0][0]]);

        llama_kv_cells::seq_set_t in_batch;
        for (uint32_t k = 0; k < ubatch.n_seqs_unq; ++k) {
            in_batch.set(ubatch.seq_id_unq[k]);
        }

        for (uint32_t j = 0; j < cells.size(); ++j) {
            if (cells.is_empty(j) || (cells.seq_get_all(j) & in_batch).none()) {
                continue;
            }

            const auto & e = cells.ext_get(j);

            if (e.x != e.y || e.x != cells.pos_get(j)) {
                return false;
            }
        }
    }

    return true;
}

void llama_kv_cache::set_input_kq_derived(
        ggml_tensor * cell_pos, ggml_tensor * tok_lo, ggml_tensor * tok_hi,
        const llama_ubatch * ubatch, bool causal_attn) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(cell_pos->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(tok_lo->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(tok_hi->buffer));

    const int64_t n_kv     = cell_pos->ne[0];
    const int64_t n_stream_l = cell_pos->ne[1];
    const int64_t n_tps    = tok_lo->ne[0];

    GGML_ASSERT(tok_hi->ne[0] == n_tps && tok_lo->ne[1] == n_stream_l && tok_hi->ne[1] == n_stream_l);
    GGML_ASSERT(n_tps*n_stream_l == (int64_t) ubatch->n_tokens);

    // the derived form is only created for the shape kq_mask_derivable() accepts; the asserts above
    // and here keep the fill aligned with the packed fill in set_input_kq_mask
    GGML_ASSERT(n_stream_l == 1);

    // several sequences: sequence k of the ubatch lives in [k*KQ_DERIVED_SEQ_STRIDE, (k + 1)*KQ_DERIVED_SEQ_STRIDE)
    // (see kq_mask_derivable_seqs); one sequence keeps offset 0 and the unbounded non-causal window
    const bool multi_seq = ubatch->n_seqs_unq > 1;
    auto seq_offset = [&](llama_seq_id s) -> int32_t {
        return multi_seq ? ubatch->seq_idx[s]*KQ_DERIVED_SEQ_STRIDE : 0;
    };

    // see llama_non_causal_type - same adjustment as set_input_kq_mask
    if (!causal_attn && hparams.non_causal_type == LLAMA_NON_CAUSAL_TYPE_SWA_ONLY) {
        causal_attn = swa_type == LLAMA_SWA_TYPE_NONE;
    }

    // the min position in the batch for each sequence
    llama_pos seq_pos_min[LLAMA_MAX_SEQ];
    std::fill(seq_pos_min, seq_pos_min + LLAMA_MAX_SEQ, INT32_MAX);

    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        const llama_seq_id seq_id = ubatch->seq_id[i][0];

        seq_pos_min[seq_id] = std::min(seq_pos_min[seq_id], ubatch->pos[i]);
    }

    const bool swa = swa_type != LLAMA_SWA_TYPE_NONE;
    const bool swa_full = !causal_attn && hparams.non_causal_type == LLAMA_NON_CAUSAL_TYPE_SWA_FULL;

    int32_t * cell_pos_data = (int32_t *) cell_pos->data;
    int32_t * tok_lo_data   = (int32_t *) tok_lo->data;
    int32_t * tok_hi_data   = (int32_t *) tok_hi->data;

    // per-cell positions (INT32_MIN = never visible)
    if (!multi_seq) {
        const llama_seq_id seq_id = ubatch->seq_id[0][0];

        const auto & cells = v_cells.at(seq_to_stream[seq_id]);

        for (int64_t j = 0; j < n_kv; ++j) {
            const bool own = !cells.is_empty(j) && cells.seq_has(j, seq_id);

            cell_pos_data[j] = own ? (int32_t) cells.pos_get(j) : INT32_MIN;
        }
    } else {
        const auto & cells = v_cells.at(seq_to_stream[ubatch->seq_id[0][0]]);

        llama_kv_cells::seq_set_t in_batch;
        for (uint32_t k = 0; k < ubatch->n_seqs_unq; ++k) {
            in_batch.set(ubatch->seq_id_unq[k]);
        }

        for (int64_t j = 0; j < n_kv; ++j) {
            cell_pos_data[j] = INT32_MIN;

            if (cells.is_empty(j) || (cells.seq_get_all(j) & in_batch).none()) {
                continue;
            }

            // kq_mask_derivable_seqs: exactly one of the batch's sequences
            for (uint32_t k = 0; k < ubatch->n_seqs_unq; ++k) {
                if (cells.seq_has(j, ubatch->seq_id_unq[k])) {
                    cell_pos_data[j] = (int32_t) cells.pos_get(j) + seq_offset(ubatch->seq_id_unq[k]);
                    break;
                }
            }
        }
    }

    // per-token visibility window
    for (int64_t ii = 0; ii < n_tps; ++ii) {
        const llama_seq_id seq_id = ubatch->seq_id[ii][0];

        const llama_pos p1   = ubatch->pos[ii];
        const llama_pos hi   = causal_attn ? p1 : (multi_seq ? KQ_DERIVED_SEQ_STRIDE - 1 : INT32_MAX);

              llama_pos lo   = 0;

        if (swa) {
            switch (swa_type) {
                case LLAMA_SWA_TYPE_STANDARD:
                    lo = p1 - (llama_pos) n_swa + 1;
                    break;
                case LLAMA_SWA_TYPE_CHUNKED:
                    lo = (p1/(llama_pos) n_swa)*(llama_pos) n_swa;
                    break;
                default:
                    lo = 0;
                    break;
            }

            // LLAMA_NON_CAUSAL_TYPE_SWA_FULL keeps every cell at or after the batch's first position
            // of this sequence, i.e. a union of two ranges - the floor lowers to seq_pos_min
            if (swa_full) {
                lo = std::min(lo, seq_pos_min[seq_id]);
            }
        }

        if (lo < 0) {
            lo = 0;
        }

        tok_lo_data[ii] = (int32_t) lo + seq_offset(seq_id);
        tok_hi_data[ii] = (int32_t) hi + seq_offset(seq_id);
    }
}
// ==============================================================================================

void llama_kv_cache::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(n_stream == 1 && "TODO: support multiple streams");
    const auto & cells = v_cells[0];

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

    int32_t * data = (int32_t *) dst->data;

    const int32_t n_kv = dst->ne[0];

    for (int h = 0; h < 1; ++h) {
        for (int i = 0; i < n_tokens; ++i) {
            for (int j = 0; j < n_kv; ++j) {
                // the position when the cells is empty is irrelevant - it will be masked out later in the attention
                const llama_pos p0 = cells.is_empty(j) ? -1 : cells.pos_get(j);

                data[h*(n_kv*n_tokens) + i*n_kv + j] = llama_relative_position_bucket(p0, ubatch->pos[i], hparams.n_rel_attn_bkts, false);
            }
        }
    }
}

void llama_kv_cache::set_input_k_rot(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const auto n_rot = dst->ne[0];
    GGML_ASSERT(attn_rot_hadamard.count(dst->ne[0]));

    memcpy(dst->data, attn_rot_hadamard.at(n_rot).data(), ggml_nbytes(dst));
}

void llama_kv_cache::set_input_v_rot(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const auto n_rot = dst->ne[0];
    GGML_ASSERT(attn_rot_hadamard.count(dst->ne[0]));

    memcpy(dst->data, attn_rot_hadamard.at(n_rot).data(), ggml_nbytes(dst));
}

bool llama_kv_cache::has_cell_ext() const {
    // M-RoPE needs the 2D position, the PLE / Engram n-gram hashes need the token id
    return hparams.n_pos_per_embd() > 1 || hparams.ple_n_heads > 0;
}

void llama_kv_cache::get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    res.clear();
    res.resize(n_tokens*n, LLAMA_TOKEN_NULL);

    if (n == 0) {
        return;
    }

    // note: apply_ubatch() has already stored the current ubatch, so the cells cover the tokens
    //       of this very ubatch as well, which is what we want
    // the nearest cell at or before a position also resolves M-RoPE gaps, where multiple tokens
    // share the same temporal pos

    // an embd (multimodal) ubatch can repeat one position for a whole image, so positions
    // do not encode the token order; resolve its predecessors by ubatch order instead
    // same for a mixed ubatch
    const bool by_order = !ubatch.token || ubatch.is_mixed();

    std::vector<uint32_t> ord; // index among the ubatch tokens of the same seq
    std::unordered_map<llama_seq_id, std::vector<uint32_t>> seq_idx;

    if (by_order) {
        ord.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; ++i) {
            auto & v = seq_idx[ubatch.seq_id[i][0]];
            ord[i] = v.size();
            v.push_back(i);
        }
    }

    for (uint32_t i = 0; i < n_tokens; ++i) {
        // TODO: a token that belongs to more than one sequence has an ambiguous history.
        //       the n-gram architectures have to reject such batches
        const llama_seq_id seq_id = ubatch.seq_id[i][0];

        for (uint32_t j = 0; j < n; ++j) {
            const llama_pos d = (llama_pos) (n - j);

            llama_pos p;
            if (by_order) {
                const auto & v = seq_idx[seq_id];
                const int64_t k = (int64_t) ord[i] - d;
                // k >= 0: an earlier token of this very ubatch; k < 0: before the chunk
                p = k >= 0 ? ubatch.pos[v[k]] : ubatch.pos[v[0]] + (llama_pos) k;
            } else {
                p = ubatch.pos[i] - d;
            }

            if (p < 0) {
                continue;
            }

            res[i*n + j] = v_cells[seq_to_stream[seq_id]].seq_pos_tok_le(seq_id, p);
        }
    }
}

size_t llama_kv_cache::total_size() const {
    size_t size = 0;

    for (const auto & [_, buf] : ctxs_bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }

    return size;
}

size_t llama_kv_cache::size_k_bytes() const {
    size_t size_k_bytes = 0;

    for (const auto & layer : layers) {
        size_k_bytes += ggml_nbytes(layer.k);
    }

    return size_k_bytes;
}

size_t llama_kv_cache::size_v_bytes() const {
    size_t size_v_bytes = 0;

    for (const auto & layer : layers) {
        size_v_bytes += layer.v ? ggml_nbytes(layer.v) : 0;
    }

    return size_v_bytes;
}

ggml_tensor * llama_kv_cache::build_rope_shift(
        const llama_cparams & cparams,
               ggml_context * ctx,
                ggml_tensor * cur,
                ggml_tensor * shift,
                ggml_tensor * rot,
                ggml_tensor * factors,
                      float   freq_base,
                      float   freq_scale,
                   uint32_t   il) const {
    const auto & n_ctx_orig = cparams.n_ctx_orig_yarn;

    const auto & yarn_ext_factor  = cparams.yarn_ext_factor;
    const auto & yarn_beta_fast   = cparams.yarn_beta_fast;
    const auto & yarn_beta_slow   = cparams.yarn_beta_slow;
    const auto & yarn_attn_factor = cparams.yarn_attn_factor;

    const auto & n_rot     = hparams.n_rot(il);
    const auto & rope_type = hparams.rope_type == LLAMA_ROPE_TYPE_MROPE || hparams.rope_type == LLAMA_ROPE_TYPE_IMROPE
                                // @ngxson : this is a workaround
                                // for M-RoPE, we want to rotate the whole vector when doing KV shift
                                // a normal RoPE should work, we just need to use the correct ordering
                                // ref: https://github.com/ggml-org/llama.cpp/pull/13870
                                ? LLAMA_ROPE_TYPE_NEOX
                                : hparams.rope_type;
    ggml_tensor * tmp;

    if (ggml_is_quantized(cur->type)) {
        // dequantize to f32 -> RoPE -> quantize back
        tmp = ggml_cast(ctx, cur, GGML_TYPE_F32);

        // rotate back
        tmp = llama_mul_mat_hadamard(ctx, tmp, rot);

        tmp = ggml_rope_ext(ctx, tmp,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);

        // rotate fwd
        tmp = llama_mul_mat_hadamard(ctx, tmp, rot);

        tmp = ggml_cpy(ctx, tmp, cur);
    } else {
        // we rotate only the first n_rot dimensions
        tmp = ggml_rope_ext_inplace(ctx, cur,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);
    }

    return tmp;
}

class llm_graph_input_k_shift : public llm_graph_input_i {
public:
    llm_graph_input_k_shift(const llama_kv_cache * kv_self) : kv_self(kv_self) {}
    virtual ~llm_graph_input_k_shift() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * k_shift; // I32 [kv_size*n_stream]

    // note: assumes k_rot^2 == I
    ggml_tensor * k_rot = nullptr;

    const llama_kv_cache * kv_self;
};

void llm_graph_input_k_shift::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    if (k_shift && k_shift->buffer) {
        kv_self->set_input_k_shift(k_shift);
    }

    if (k_rot && k_rot->buffer) {
        kv_self->set_input_k_rot(k_rot);
    }
}

ggml_cgraph * llama_kv_cache::build_graph_shift(llm_graph_result * res, llama_context * lctx) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    GGML_ASSERT(!other);

    auto * ctx = res->get_ctx();
    auto * gf  = res->get_gf();

    auto inp = std::make_unique<llm_graph_input_k_shift>(this);

    inp->k_shift = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) get_size()*n_stream);
    ggml_set_input(inp->k_shift);

    inp->k_rot = build_input_k_rot(ctx);

    const auto & cparams = lctx->get_cparams();

    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        if (!hparams.has_rope(il)) {
            continue;
        }

        const int64_t n_head_kv    = hparams.n_head_kv(il);
        const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        const auto n_rot         = hparams.n_rot(il);
        const auto n_embd_head_k = hparams.n_embd_head_k(il);
        const auto n_embd_nope   = hparams.n_lora_kv > 0 ? n_embd_head_k - n_rot : 0;

        const float freq_base_l  = model.get_rope_freq_base (cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

        ggml_tensor * k =
            ggml_view_3d(ctx, layer.k,
                n_rot, n_head_kv, get_size()*n_stream,
                ggml_row_size(layer.k->type, n_embd_head_k),
                ggml_row_size(layer.k->type, n_embd_k_gqa),
                ggml_row_size(layer.k->type, n_embd_nope));

        ggml_tensor * cur = build_rope_shift(cparams, ctx, k, inp->k_shift, inp->k_rot, rope_factors, freq_base_l, freq_scale_l, il);

        ggml_build_forward_expand(gf, cur);
    }

    res->add_input(std::move(inp));

    return gf;
}

void llama_kv_cache::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) && swa_type == LLAMA_SWA_TYPE_NONE) {
        return;
    }

    io.write(&n_stream, sizeof(n_stream));

    for (uint32_t s = 0; s < n_stream; ++s) {
        cell_ranges_t cr { s, {} };

        uint32_t cell_count = 0;

        const auto & cells = v_cells[s];

        // Count the number of cells with the specified seq_id
        // Find all the ranges of cells with this seq id (or all, when -1)
        uint32_t cell_range_begin = cells.size();

        for (uint32_t i = 0; i < cells.size(); ++i) {
            bool add_cell = true;

            add_cell = add_cell && !cells.is_empty(i);
            add_cell = add_cell && (seq_id == -1 || cells.seq_has(i, seq_id));

            // check the cell is not SWA-masked
            if (add_cell && seq_id != -1) {
                const bool is_masked = llama_hparams::is_masked_swa(n_swa, swa_type, cells.pos_get(i), cells.seq_pos_max(seq_id));

                add_cell = !is_masked;
            }

            if (add_cell) {
                ++cell_count;
                if (cell_range_begin == cells.size()) {
                    cell_range_begin = i;
                }
            } else {
                if (cell_range_begin != cells.size()) {
                    cr.data.emplace_back(cell_range_begin, i);
                    cell_range_begin = cells.size();
                }
            }
        }

        if (cell_range_begin != cells.size()) {
            cr.data.emplace_back(cell_range_begin, cells.size());
        }

        // DEBUG CHECK: Sum of cell counts in ranges should equal the total cell count
        uint32_t cell_count_check = 0;
        for (const auto & range : cr.data) {
            cell_count_check += range.second - range.first;
        }
        GGML_ASSERT(cell_count == cell_count_check);

        io.write(&cell_count, sizeof(cell_count));

        // skip empty streams
        if (cell_count == 0) {
            continue;
        }

        state_write_meta(io, cr, seq_id);
        state_write_data(io, cr);
    }
}

void llama_kv_cache::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    state_read_sinfo(io, seq_id, flags, nullptr, nullptr);
}

void llama_kv_cache::state_read_sinfo(
        llama_io_read_i & io,
           llama_seq_id   seq_id,
  llama_state_seq_flags   flags,
      slot_info_vec_t *   sinfos_out,
const slot_info_vec_t *   sinfos_in) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) && swa_type == LLAMA_SWA_TYPE_NONE) {
        return;
    }

    // TODO: fix incosistent handling of `seq_id < 0` and `seq_id == -1` in the codebase [TAG_LLAMA_SEQ_ID_NEG]
    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    if (sinfos_out) {
        sinfos_out->assign(n_stream, slot_info{});
    }

    if (sinfos_in && sinfos_in->size() != n_stream) {
        throw std::runtime_error("failed to restore kv cache: mirrored slot layout has the wrong stream count");
    }

    uint32_t n_stream_cur;
    io.read(&n_stream_cur, sizeof(n_stream_cur));
    if (n_stream_cur != n_stream) {
        throw std::runtime_error("n_stream mismatch");
    }

    // a whole-context restore replaces every stream, so the cache is emptied once here
    // clear() resets all streams at once, so doing it per stream below would keep only the last one
    if (seq_id == -1) {
        clear(true);
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        uint32_t cell_count;
        io.read(&cell_count, sizeof(cell_count));

        if (cell_count == 0) {
            // a mirrored cache must be empty here as well, or the two no longer agree cell for cell
            if (sinfos_in && !(*sinfos_in)[s].empty()) {
                throw std::runtime_error("failed to restore kv cache: mirrored cache holds cells this one does not");
            }
            continue;
        }

        const uint32_t strm = seq_id == -1 ? s : seq_to_stream[seq_id];

        slot_info sinfo;

        bool res = true;
        res = res && state_read_meta(io, strm, cell_count, sinfo, seq_id, sinfos_in ? &(*sinfos_in)[s] : nullptr);

        try {
            res = res && state_read_data(io, strm, cell_count, sinfo);
        } catch (...) {
            res = false;
        }

        if (!res) {
            state_clear(seq_id, strm, sinfo);
            throw std::runtime_error("failed to restore kv cache");
        }

        if (sinfos_out) {
            (*sinfos_out)[s] = sinfo;
        }
    }
}

void llama_kv_cache::state_write_meta(llama_io_write_i & io, const cell_ranges_t & cr, llama_seq_id seq_id) const {
    const auto & cells = v_cells[cr.strm];

    for (const auto & range : cr.data) {
        for (uint32_t i = range.first; i < range.second; ++i) {
            std::vector<llama_seq_id> seq_ids;

            for (llama_seq_id cur = 0; cur < (int) n_seq_max; ++cur) {
                if (cur == seq_id || seq_id == -1) {
                    if (cells.seq_has(i, cur)) {
                        seq_ids.push_back(cur);
                    }
                }
            }

            const llama_pos pos     = cells.pos_get(i);
            const uint32_t n_seq_id = seq_ids.size();

            io.write(&pos,      sizeof(pos));
            io.write(&n_seq_id, sizeof(n_seq_id));

            if (has_cell_ext()) {
                const llama_kv_cell_ext ext = cells.ext_get(i);
                io.write(&ext, sizeof(ext));
            }

            for (const auto & seq_id_i : seq_ids) {
                io.write(&seq_id_i, sizeof(seq_id_i));
            }
        }
    }
}

void llama_kv_cache::state_write_data(llama_io_write_i & io, const cell_ranges_t & cr) const {
    const auto & cells = v_cells[cr.strm];

    const uint32_t v_trans_l = this->v_trans ? 1 : 0;
    const uint32_t n_layer = layers.size();

    io.write(&v_trans_l, sizeof(v_trans_l));
    io.write(&n_layer, sizeof(n_layer));
    io.write(&n_rot_k, sizeof(n_rot_k));
    io.write(&n_rot_v, sizeof(n_rot_v));

    // Iterate and write all the keys first, each row is a cell
    // Get whole range at a time
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        auto * k = layer.k_stream[cr.strm];

        // Write key type
        const int32_t k_type_i = (int32_t) k->type;
        io.write(&k_type_i, sizeof(k_type_i));

        // Write row size of key
        const uint64_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);
        io.write(&k_size_row, sizeof(k_size_row));

        // Read each range of cells of k_size length and write out
        for (const auto & range : cr.data) {
            const size_t range_size = range.second - range.first;
            const size_t buf_size = range_size * k_size_row;
            io.write_tensor(k, range.first * k_size_row, buf_size);
        }
    }

    if (!v_trans_l) {
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[cr.strm];
            if (!v) {
                continue;
            }

            // Write value type
            const int32_t v_type_i = (int32_t) v->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write row size of value
            const uint64_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);
            io.write(&v_size_row, sizeof(v_size_row));

            // Read each range of cells of v_size length and write out
            for (const auto & range : cr.data) {
                const size_t range_size = range.second - range.first;
                const size_t buf_size = range_size * v_size_row;
                io.write_tensor(v, range.first * v_size_row, buf_size);
            }
        }
    } else {
        // When v is transposed, we also need the element size and get the element ranges from each row
        const uint32_t kv_size = cells.size();

        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[cr.strm];
            if (!v) {
                continue;
            }

            // Write value type
            const int32_t v_type_i = (int32_t) v->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write element size
            const uint32_t v_size_el = ggml_type_size(v->type);
            io.write(&v_size_el, sizeof(v_size_el));

            // Write GQA embedding size
            io.write(&n_embd_v_gqa, sizeof(n_embd_v_gqa));

            // For each row, we get the element values of each cell
            for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                // Read each range of cells of v_size_el length and write out
                for (const auto & range : cr.data) {
                    const size_t range_size = range.second - range.first;
                    const size_t src_offset = (range.first + j * kv_size) * v_size_el;
                    const size_t buf_size = range_size * v_size_el;
                    io.write_tensor(v, src_offset, buf_size);
                }
            }
        }
    }
}

bool llama_kv_cache::state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, slot_info & sinfo, llama_seq_id dest_seq_id, const slot_info * sinfo_in) {
    auto & cells = v_cells[strm];
    auto & head  = v_heads[strm];

    if (dest_seq_id != -1) {
        // single sequence
        if (cell_count > cells.size()) {
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
            return false;
        }

        seq_rm(dest_seq_id, -1, -1);

        llama_batch_allocr balloc(hparams.n_pos_per_embd());

        llama_ubatch ubatch = balloc.ubatch_reserve(cell_count, 1);

        ubatch.seq_id_unq[0] = dest_seq_id;

        // the ext as it was saved, to put back after apply_ubatch()
        std::vector<llama_kv_cell_ext> exts;
        if (has_cell_ext()) {
            exts.resize(cell_count);
        }

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id != 1) {
                LLAMA_LOG_ERROR("%s: invalid seq_id-agnostic kv cell\n", __func__);
                return false;
            }

            if (has_cell_ext()) {
                llama_kv_cell_ext ext;
                io.read(&ext, sizeof(ext));

                if (hparams.n_pos_per_embd() > 1) {
                    ubatch.pos[i + ubatch.n_tokens]   = ext.y;
                    ubatch.pos[i + ubatch.n_tokens*2] = ext.x;
                }

                // apply_ubatch() below restores ext.tok from the ubatch tokens
                ubatch.token[i] = ext.tok;

                exts[i] = ext;
            }

            // read the sequence id, but directly discard it - we will use dest_seq_id instead
            {
                llama_seq_id seq_id;
                io.read(&seq_id, sizeof(seq_id));
            }

            ubatch.pos[i]      = pos;
            ubatch.n_seq_id[i] = n_seq_id;
            ubatch.seq_id[i]   = &dest_seq_id;
        }

        if (sinfo_in) {
            // this cache mirrors another one, so it takes that cache's layout instead of searching for its own cells
            if (sinfo_in->empty() || sinfo_in->n_stream() != 1 || sinfo_in->idxs[0].size() != cell_count) {
                LLAMA_LOG_ERROR("%s: mirrored slot layout holds %d cells, this cache restores %d\n", __func__,
                        sinfo_in->empty() ? 0 : (int) sinfo_in->idxs[0].size(), cell_count);
                return false;
            }

            sinfo = *sinfo_in;

            // the layout is cell indices, so it means the same in both caches only while their streams line up
            sinfo.s0 = strm;
            sinfo.s1 = strm;
            sinfo.strm[0] = strm;

            // seq_rm above freed exactly the cells this sequence held
            // anything else in the way is a cache that had already drifted, which this restore must not hide
            for (uint32_t i = 0; i < cell_count; ++i) {
                const uint32_t idx = sinfo.idxs[0][i];

                if (idx >= cells.size() || !cells.is_empty(idx)) {
                    LLAMA_LOG_ERROR("%s: cell %u of the mirrored slot layout is not free\n", __func__, idx);
                    return false;
                }
            }
        } else {
            sinfo = find_slot(ubatch, false);
            if (sinfo.empty()) {
                LLAMA_LOG_ERROR("%s: failed to find %d available cells in kv cache\n", __func__,  cell_count);
                return false;
            }
        }

        // note: apply_ubatch() rebuilds llama_kv_cell_ext from the ubatch
        //       only ext.tok and the M-RoPE 2D position round-trip through it
        //       see: https://github.com/ggml-org/llama.cpp/pull/16825#issuecomment-3460868350
        apply_ubatch(sinfo, ubatch);

        // apply_ubatch() takes the 2D position from the ubatch, and that ubatch is built with this
        // cache's own n_pos_per_embd. a cache that does not use M-RoPE itself but mirrors one that
        // does (the qwen4exp QSA indexer) would drop x and y. put the saved ext back instead, which
        // is what the whole-context path below already does.
        for (uint32_t i = 0; i < (uint32_t) exts.size(); ++i) {
            cells.ext_set(sinfo.idxs[0][i], exts[i]);
        }

        LLAMA_LOG_DEBUG("%s: cell_count = %d, dest_seq_id = %d\n", __func__, cell_count, dest_seq_id);

        // DEBUG CHECK: verify that all cells were allocated and have correct seq_id and pos values
        GGML_ASSERT(sinfo.n_stream() == 1);
        GGML_ASSERT(sinfo.idxs[0].size() == cell_count);
        for (uint32_t i = 0; i < cell_count; ++i) {
            const uint32_t idx = sinfo.idxs[0][i];
            GGML_ASSERT(cells.pos_get(idx) == ubatch.pos[i]);
            GGML_ASSERT(cells.seq_has(idx, dest_seq_id));
        }
    } else {
        // whole KV cache restore

        if (cell_count > cells.size()) {
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
            return false;
        }

        // the cells go in from 0, so a mirrored cache lands on the same ones as long as it restores the same count. the layout itself carries no more information here
        if (sinfo_in && (sinfo_in->empty() || sinfo_in->n_stream() != 1 || sinfo_in->idxs[0].size() != cell_count)) {
            LLAMA_LOG_ERROR("%s: mirrored slot layout holds %d cells, this cache restores %d\n", __func__,
                    sinfo_in->empty() ? 0 : (int) sinfo_in->idxs[0].size(), cell_count);
            return false;
        }

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t  n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            cells.pos_set(i, pos);

            if (has_cell_ext()) {
                llama_kv_cell_ext ext;
                io.read(&ext, sizeof(ext));
                cells.ext_set(i, ext);
            }

            for (uint32_t j = 0; j < n_seq_id; ++j) {
                llama_seq_id seq_id;
                io.read(&seq_id, sizeof(seq_id));

                if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
                    LLAMA_LOG_ERROR("%s: invalid seq_id, %d is out of range [0, %u)\n", __func__, seq_id, n_seq_max);
                    return false;
                }

                cells.seq_add(i, seq_id);
            }
        }

        // Create contiguous slot_info for whole cache restore
        sinfo.s0 = strm;
        sinfo.s1 = strm;
        sinfo.resize(1);
        sinfo.strm[0] = strm;
        sinfo.idxs[0].resize(cell_count);
        for (uint32_t i = 0; i < cell_count; ++i) {
            sinfo.idxs[0][i] = i;
        }

        head = 0;
    }

    return true;
}

bool llama_kv_cache::state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo) {
    auto & cells = v_cells[strm];

    // batch the scatter reads per contiguous run of destination indices
    // from inclusive, to exclusive - same convention as cell_ranges_t
    // contiguous cells yield a single run covering the whole block
    struct cell_run { uint32_t from; uint32_t to; };
    std::vector<cell_run> runs;
    if (cell_count > 0) {
        const auto & idxs = sinfo.idxs[0];
        uint32_t i0 = 0;
        while (i0 < cell_count) {
            uint32_t i1 = i0 + 1;
            while (i1 < cell_count && idxs[i1] == idxs[i1 - 1] + 1) {
                ++i1;
            }
            runs.push_back({idxs[i0], idxs[i1 - 1] + 1});
            i0 = i1;
        }
    }

    uint32_t v_trans_l;
    uint32_t n_layer;
    uint32_t n_rot_k_ref;
    uint32_t n_rot_v_ref;

    io.read(&v_trans_l, sizeof(v_trans_l));
    io.read(&n_layer, sizeof(n_layer));
    io.read(&n_rot_k_ref, sizeof(n_rot_k_ref));
    io.read(&n_rot_v_ref, sizeof(n_rot_v_ref));

    if (n_layer != layers.size()) {
        LLAMA_LOG_ERROR("%s: mismatched layer count (%u instead of %u)\n", __func__, n_layer, (uint32_t) layers.size());
        return false;
    }

    if (cell_count > cells.size()) {
        LLAMA_LOG_ERROR("%s: not enough cells in kv cache to restore state (%u > %u)\n", __func__, cell_count, cells.size());
        return false;
    }

    if (this->v_trans != (bool) v_trans_l) {
        LLAMA_LOG_ERROR("%s: incompatible V transposition\n", __func__);
        return false;
    }

    if (n_rot_k_ref != n_rot_k) {
        LLAMA_LOG_ERROR("%s: incompatible key rotation (%u instead of %u)\n", __func__, n_rot_k_ref, n_rot_k);
        return false;
    }

    if (n_rot_v_ref != n_rot_v) {
        LLAMA_LOG_ERROR("%s: incompatible value rotation (%u instead of %u)\n", __func__, n_rot_v_ref, n_rot_v);
        return false;
    }

    // For each layer, read the keys for each cell, one row is one cell, read as one contiguous block
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        auto * k = layer.k_stream[strm];

        // Read type of key
        int32_t k_type_i_ref;
        io.read(&k_type_i_ref, sizeof(k_type_i_ref));
        const int32_t k_type_i = (int32_t) k->type;
        if (k_type_i != k_type_i_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key type (%d != %d, layer %d)\n", __func__, k_type_i, k_type_i_ref, il);
            return false;
        }

        // Read row size of key
        uint64_t k_size_row_ref;
        io.read(&k_size_row_ref, sizeof(k_size_row_ref));
        const size_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);
        if (k_size_row != k_size_row_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key row size (%zu != %zu, layer %d)\n", __func__, k_size_row, (size_t) k_size_row_ref, il);
            return false;
        }

        for (const auto & r : runs) {
            io.read_tensor(k, (size_t) r.from * k_size_row, (size_t) (r.to - r.from) * k_size_row);
        }
    }

    if (!this->v_trans) {
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[strm];
            if (!v) {
                continue;
            }

            // Read type of value
            int32_t v_type_i_ref;
            io.read(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t) v->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read row size of value
            uint64_t v_size_row_ref;
            io.read(&v_size_row_ref, sizeof(v_size_row_ref));
            const size_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);
            if (v_size_row != v_size_row_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value row size (%zu != %zu, layer %d)\n", __func__, v_size_row, (size_t) v_size_row_ref, il);
                return false;
            }

            for (const auto & r : runs) {
                io.read_tensor(v, (size_t) r.from * v_size_row, (size_t) (r.to - r.from) * v_size_row);
            }
        }
    } else {
        // For each layer, read the values for each cell (transposed)
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[strm];
            if (!v) {
                continue;
            }

            // Read type of value
            int32_t v_type_i_ref;
            io.read(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t) v->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read element size of value
            uint32_t v_size_el_ref;
            io.read(&v_size_el_ref, sizeof(v_size_el_ref));
            const size_t v_size_el = ggml_type_size(v->type);
            if (v_size_el != v_size_el_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value element size (%zu != %zu, layer %d)\n", __func__, v_size_el, (size_t) v_size_el_ref, il);
                return false;
            }

            // Read GQA embedding size
            uint32_t n_embd_v_gqa_ref;
            io.read(&n_embd_v_gqa_ref, sizeof(n_embd_v_gqa_ref));
            if (n_embd_v_gqa != n_embd_v_gqa_ref) {
                LLAMA_LOG_ERROR("%s: mismatched GQA embedding size (%u != %u, layer %d)\n", __func__, n_embd_v_gqa, n_embd_v_gqa_ref, il);
                return false;
            }

            for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                for (const auto & r : runs) {
                    const size_t dst_offset = ((size_t) r.from + j * cells.size()) * v_size_el;
                    io.read_tensor(v, dst_offset, (size_t) (r.to - r.from) * v_size_el);
                }
            }
        }
    }

    return true;
}

void llama_kv_cache::state_clear(llama_seq_id seq_id) {
    if (seq_id == -1) {
        clear(true);
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const uint32_t strm = seq_to_stream[seq_id];

    const auto & cells = v_cells[strm];

    slot_info sinfo;
    sinfo.s0 = strm;
    sinfo.s1 = strm;
    sinfo.resize(1);
    sinfo.strm[0] = strm;

    // a cell that another sequence still uses keeps its data
    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cells.seq_has(i, seq_id) && cells.seq_count(i) == 1) {
            sinfo.idxs[0].push_back(i);
        }
    }

    state_clear(seq_id, strm, sinfo);
}

// the cleared ranges mirror the write pattern of state_read_data() - keep both in sync
void llama_kv_cache::state_clear(llama_seq_id seq_id, uint32_t strm, const slot_info & sinfo) {
    if (seq_id == -1) {
        clear(true);
        return;
    }

    seq_rm(seq_id, -1, -1);

    // zero the K/V data of the failed restore attempt - the attention can still read the data of free cells
    if (sinfo.empty() || sinfo.size() == 0) {
        return;
    }

    const auto & cells = v_cells[strm];

    const uint32_t cell_count = sinfo.size();

    const bool is_contiguous = sinfo.is_contiguous();

    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        auto * k = layer.k_stream[strm];

        const size_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);

        if (is_contiguous) {
            llama_clear_tensor_data(k, sinfo.head() * k_size_row, cell_count * k_size_row);
        } else {
            for (uint32_t i = 0; i < cell_count; ++i) {
                llama_clear_tensor_data(k, sinfo.idxs[0][i] * k_size_row, k_size_row);
            }
        }
    }

    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

        auto * v = layer.v_stream[strm];
        if (!v) {
            continue;
        }

        if (!v_trans) {
            const size_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);

            if (is_contiguous) {
                llama_clear_tensor_data(v, sinfo.head() * v_size_row, cell_count * v_size_row);
            } else {
                for (uint32_t i = 0; i < cell_count; ++i) {
                    llama_clear_tensor_data(v, sinfo.idxs[0][i] * v_size_row, v_size_row);
                }
            }
        } else {
            const size_t v_size_el = ggml_type_size(v->type);

            if (is_contiguous) {
                const uint32_t h = sinfo.head();

                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    llama_clear_tensor_data(v, (h + j * cells.size()) * v_size_el, cell_count * v_size_el);
                }
            } else {
                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    for (uint32_t i = 0; i < cell_count; ++i) {
                        llama_clear_tensor_data(v, (sinfo.idxs[0][i] + j * cells.size()) * v_size_el, v_size_el);
                    }
                }
            }
        }
    }
}

//
// llama_kv_cache_context
//

llama_kv_cache_context::llama_kv_cache_context(llama_memory_status status) : status(status) {}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv) {
    n_kv = kv->get_size();

    const uint32_t n_stream = kv->get_n_stream();

    // create a dummy slot info - the actual data is irrelevant. we just need to build the graph
    sinfos.resize(1);
    sinfos[0].s0 = 0;
    sinfos[0].s1 = n_stream - 1;
    sinfos[0].idxs.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        sinfos[0].strm.push_back(s);
        sinfos[0].idxs[s].resize(1, 0);
    }
}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv,
        llama_context * lctx,
        bool do_shift,
        stream_copy_info sc_info) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), lctx(lctx), do_shift(do_shift), sc_info(std::move(sc_info)) {
    if (!do_shift && this->sc_info.empty()) {
        status = LLAMA_MEMORY_STATUS_NO_UPDATE;
    }
}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv,
        llama_kv_cache::slot_info_vec_t sinfos,
        std::vector<llama_ubatch> ubatches) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), sinfos(std::move(sinfos)), ubatches(std::move(ubatches)) {
}

llama_kv_cache_context::~llama_kv_cache_context() = default;

bool llama_kv_cache_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (++i_cur >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_kv_cache_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    // no ubatches -> this is a KV cache update
    if (ubatches.empty()) {
        kv->update(lctx, do_shift, sc_info);

        return true;
    }

    kv->apply_ubatch(sinfos[i_cur], ubatches[i_cur]);
    n_kv = kv->get_n_kv(sinfos[i_cur], ubatches[i_cur].n_tokens);

    // wip/kv-residency (TODO #117): this is the last point before the graph is built, and `n_kv` is the PADDED
    // extent its views use -- so it is also exactly the extent the region must have mapped.  Growing to the
    // used cells instead would leave a kernel reading unmapped VA (a GPU fault, not an error).
    kv->grow_regions(n_kv);

    return true;
}

llama_memory_status llama_kv_cache_context::get_status() const {
    return status;
}

const llama_ubatch & llama_kv_cache_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_cur];
}

uint32_t llama_kv_cache_context::get_n_kv() const {
    return n_kv;
}

ggml_type llama_kv_cache_context::type_k() const {
    return kv->type_k();
}

ggml_type llama_kv_cache_context::type_v() const {
    return kv->type_v();
}

ggml_tensor * llama_kv_cache_context::get_k(ggml_context * ctx, int32_t il) const {
    return kv->get_k(ctx, il, n_kv, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::get_v(ggml_context * ctx, int32_t il) const {
    return kv->get_v(ctx, il, n_kv, sinfos[i_cur]);
}

uint32_t llama_kv_cache_context::get_kv_split_n_tail() const {
    return kv->get_kv_split_n_tail(n_kv);
}

ggml_tensor * llama_kv_cache_context::get_k_tail(ggml_context * ctx, int32_t il, uint32_t n_tail) const {
    return kv->get_k_tail(ctx, il, n_tail, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::get_v_tail(ggml_context * ctx, int32_t il, uint32_t n_tail) const {
    return kv->get_v_tail(ctx, il, n_tail, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const {
    return kv->cpy_k(ctx, k_cur, k_idxs, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const {
    return kv->cpy_v(ctx, v_cur, v_idxs, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_k_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_context::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_v_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_context::build_input_k_rot(ggml_context * ctx) const {
    return kv->build_input_k_rot(ctx);
}

ggml_tensor * llama_kv_cache_context::build_input_v_rot(ggml_context * ctx) const {
    return kv->build_input_v_rot(ctx);
}

void llama_kv_cache_context::set_input_k_shift(ggml_tensor * dst) const {
    kv->set_input_k_shift(dst);
}

void llama_kv_cache_context::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_k_idxs(dst, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_context::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_v_idxs(dst, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_context::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    kv->set_input_kq_mask(dst, ubatch, causal_attn);
}

bool llama_kv_cache_context::kq_mask_derivable(const llama_ubatch & ubatch) const {
    return kv->kq_mask_derivable(ubatch);
}

void llama_kv_cache_context::set_input_kq_derived(
        ggml_tensor * cell_pos, ggml_tensor * tok_lo, ggml_tensor * tok_hi,
        const llama_ubatch * ubatch, bool causal_attn) const {
    kv->set_input_kq_derived(cell_pos, tok_lo, tok_hi, ubatch, causal_attn);
}

void llama_kv_cache_context::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_pos_bucket(dst, ubatch);
}

void llama_kv_cache_context::set_input_k_rot(ggml_tensor * dst) const {
    kv->set_input_k_rot(dst);
}

void llama_kv_cache_context::set_input_v_rot(ggml_tensor * dst) const {
    kv->set_input_v_rot(dst);
}

void llama_kv_cache_context::get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const {
    kv->get_prev_tokens(ubatch, n, res);
}
