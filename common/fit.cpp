#include "fit.h"

#include "json.h"
#include "log.h"

#include "../src/llama-ext.h"

#include <array>
#include <cassert>
#include <stdexcept>
#include <cinttypes>
#include <set>
#include <string>
#include <vector>

// this enum is only used in llama_params_fit_impl but needs to be defined outside of it to fix a Windows compilation issue
// enum to identify part of a layer for distributing its tensors:
enum common_layer_fraction_t {
    LAYER_FRACTION_NONE = 0, // nothing
    LAYER_FRACTION_ATTN = 1, // attention
    LAYER_FRACTION_UP   = 2, // attention + up
    LAYER_FRACTION_GATE = 3, // attention + up + gate
    LAYER_FRACTION_MOE  = 4, // everything but sparse MoE weights
};

class common_params_fit_exception : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

// wip/fit-slab-accounting (Phase 1): the per-device reservation --fit must hold for the MoE expert cache
// and the movable-boundary slab.  See wip/fit-slab-accounting/README.md.
namespace {

struct common_fit_cache_reserve {
    bool    cache_auto    = true;   // MOE_EXPERT_CACHE_MIB unset
    bool    unified       = false;  // MOE_EXPERT_CACHE_UNIFIED: allow the arena on a shared host/device pool
    int64_t mib_bytes     = 0;      // explicit MOE_EXPERT_CACHE_MIB, in bytes
    int64_t min_mib_bytes = 0;      // MOE_EXPERT_CACHE_MIN_MIB, in bytes
    int     min_res_pct   = 18;     // MOE_EXPERT_CACHE_MIN_RES_PCT

    static common_fit_cache_reserve from_env() {
        constexpr int64_t MiB = 1024 * 1024;
        common_fit_cache_reserve r;
        const char * uni_env = getenv("MOE_EXPERT_CACHE_UNIFIED");
        r.unified = uni_env != nullptr && uni_env[0] != '\0' && atoi(uni_env) != 0;
        const char * mib_env = getenv("MOE_EXPERT_CACHE_MIB");
        if (mib_env != nullptr && mib_env[0] != '\0') {
            r.cache_auto = false;
            r.mib_bytes  = (int64_t) atoll(mib_env) * MiB;
        }
        const char * min_mib_env = getenv("MOE_EXPERT_CACHE_MIN_MIB");
        const char * min_res_env = getenv("MOE_EXPERT_CACHE_MIN_RES_PCT");
        r.min_mib_bytes = min_mib_env ? (int64_t) atoi(min_mib_env) * MiB : 0;
        r.min_res_pct   = min_res_env ? atoi(min_res_env) : 18;
        return r;
    }

    // wip/umem-arena-guard: mirror the backend's refusal to run the arena on a shared host/device memory
    // pool (an integrated GPU, where "VRAM" is host RAM).  The backend declines that combination unless
    // `MOE_EXPERT_CACHE_UNIFIED=1` *and* an explicit MIB is given, so reserving slab headroom / arena
    // bytes here would withhold device memory from a cache that will never be created.  Keep the two
    // predicates in lockstep (`moe_cache_shared_pool_device` / `moe_cache_preflight`).
    static bool shared_pool(ggml_backend_dev_t dev) {
        ggml_backend_dev_props props = {};
        ggml_backend_dev_get_props(dev, &props);
        return props.type == GGML_BACKEND_DEVICE_TYPE_IGPU;
    }

    bool arena_refused(ggml_backend_dev_t dev) const {
        // Mirrors `moe_cache_preflight`: a shared pool is refused outright, and the override needs BOTH an
        // explicit arena budget and an explicit slab reserve (the slab maps `free - reserve`).
        const char * slab_reserve = getenv("GGML_CUDA_SLAB_RESERVE_MIB");
        const bool   slab_bounded = slab_reserve != nullptr && slab_reserve[0] != '\0';
        return shared_pool(dev) && !(unified && !cache_auto && slab_bounded);
    }

    // The --fit margin for a device that owns `host_bytes` of host-resident experts, over the base
    // fit_params_target: max(base, slab headroom) + arena budget.  The headroom is additive-safe because
    // the arena lives INSIDE the slab while the headroom stays OUTSIDE it (both draw on the same free
    // VRAM); `max` rather than `+` avoids double-counting a large user target.
    int64_t margin(ggml_backend_dev_t dev, size_t host_bytes, int64_t base) const {
        int64_t m = base;
        if (host_bytes == 0 || arena_refused(dev)) {
            return m;
        }
        const int64_t headroom = (int64_t) ggml_backend_dev_slab_headroom_bytes(dev);
        if (headroom > m) {
            m = headroom;
        }
        int64_t arena;
        if (!cache_auto) {
            arena = mib_bytes;  // G1: the explicit budget is reserved verbatim
        } else {
            arena = min_mib_bytes;
            if (min_res_pct > 0) {
                const int64_t res = (int64_t) host_bytes * min_res_pct / 100;
                if (res > arena) {
                    arena = res;
                }
            }
        }
        return m + arena;
    }
};

} // namespace

// ---------------------------------------------------------------------------------------------
// wip/kv-residency (TODO #118) A3: the windowed KV, as `--fit` sees it.
//
// The KV buffer type is the only part of the context tier that scales with `-c`, and with a window only its
// RESIDENT part is ever committed to VRAM (the rest is pinned host memory).  Fit therefore needs the KV's own
// bytes, split from the rest of the context tier, which is what `kv_bytes_out` carries out of the measurement
// below (a `no_alloc` probe builds no window, so the numbers are the whole KV at that `-c`).
static bool fit_buft_is_kv_region(ggml_backend_buffer_type_t buft) {
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    if (dev == nullptr) {
        return false;
    }
    auto fn = (int (*)(ggml_backend_buffer_type_t)) ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev), "ggml_backend_cuda_kv_region_is_kv_buft");
    return fn != nullptr && fn(buft) != 0;
}

// The pinned-RAM ceiling on the spill, in bytes (SIZE_MAX = no ceiling / unknown), and the H2D link
// constant the cost model divides by, in GB/s (0 = unknown).  Both come from the backend so there is one
// number in the process.
static size_t fit_kv_spill_ram_budget(ggml_backend_dev_t dev) {
    auto fn = (size_t (*)(void)) ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev), "ggml_backend_cuda_kv_region_spill_ram_budget");
    return fn != nullptr ? fn() : SIZE_MAX;
}

static float fit_kv_link_gbps(ggml_backend_dev_t dev) {
    auto fn = (float (*)(ggml_backend_dev_t)) ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev), "ggml_backend_cuda_kv_region_link_gbps");
    return fn != nullptr ? fn(dev) : 0.0f;
}

static std::vector<llama_device_memory_data> common_get_device_memory_data_impl(
        const char * path_model,
        const llama_model_params * mparams,
        const llama_context_params * cparams,
        std::vector<ggml_backend_dev_t> & devs,
        uint32_t & hp_ngl,
        uint32_t & hp_n_ctx_train,
        uint32_t & hp_n_expert,
        ggml_log_level log_level,
        std::vector<size_t> * host_expert_bytes_out = nullptr,
        std::vector<std::pair<ggml_backend_dev_t, size_t>> * host_experts_out = nullptr,
        std::vector<size_t> * kv_bytes_out = nullptr) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } original_logger;
        ggml_log_level min_level; // prints below this log level go to debug log
    };
    user_data_t ud;
    llama_log_get(&ud.original_logger.callback, &ud.original_logger.user_data);
    ud.min_level = log_level;

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        const ggml_log_level level_eff = level >= ud->min_level ? level : GGML_LOG_LEVEL_DEBUG;
        ud->original_logger.callback(level_eff, text, ud->original_logger.user_data);
    }, &ud);

    llama_model_params mparams_copy = *mparams;
    mparams_copy.no_alloc  = true;
    mparams_copy.load_mode = LLAMA_LOAD_MODE_NONE;

    llama_model * model = llama_model_load_from_file(path_model, mparams_copy);
    if (model == nullptr) {
        llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);
        throw std::runtime_error("failed to load model");
    }

    llama_context * ctx = llama_init_from_model(model, *cparams);
    if (ctx == nullptr) {
        llama_model_free(model);
        llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);
        throw std::runtime_error("failed to create llama_context from model");
    }

    const size_t nd = llama_model_n_devices(model);
    std::vector<llama_device_memory_data> ret(nd + 1);

    llama_memory_breakdown memory_breakdown = llama_get_memory_breakdown(ctx);

    // wip/kv-residency (TODO #118) A3: the KV's own bytes, per device (the window's cell size comes from
    // these: a KV cell is `kv_bytes / kv_size` bytes across the whole cache).
    if (kv_bytes_out != nullptr) {
        kv_bytes_out->assign(nd, 0);
    }

    for (const auto & [buft, mb] : memory_breakdown) {
        if (ggml_backend_buft_is_host(buft)) {
            ret.back().mb.model   += mb.model;
            ret.back().mb.context += mb.context;
            ret.back().mb.compute += mb.compute;
            continue;
        }

        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (!dev) {
            continue;
        }
        for (size_t i = 0; i < nd; i++) {
            if (dev == llama_model_get_device(model, i)) {
                ret[i].mb.model   += mb.model;
                ret[i].mb.context += mb.context;
                ret[i].mb.compute += mb.compute;
                if (kv_bytes_out != nullptr && fit_buft_is_kv_region(buft)) {
                    // the KV cache's whole footprint at this `-c`: the reservation is its size, and the
                    // resident window is a subset of it (only what the buffer COMMITS is VRAM)
                    (*kv_bytes_out)[i] += mb.context;
                }
                break;
            }
        }
    }

    {
        ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu_dev == nullptr) {
            throw std::runtime_error("no CPU backend found");
        }
        size_t free;
        size_t total;
        ggml_backend_dev_memory(cpu_dev, &free, &total);
        ret.back().free  = free;
        ret.back().total = total;
    }
    for (size_t i = 0; i < nd; i++) {
        ggml_backend_dev_t dev = llama_model_get_device(model, i);

        size_t free;
        size_t total;
        ggml_backend_dev_memory(dev, &free, &total);

        // Some non-GPU accelerator backends, such as BLAS, report 0/0 and rely on
        // the host-memory fallback. For GPU-like backends, keep 0/0 so --fit does
        // not assign anything to a device with an unknown memory budget.
        if (free == 0 && total == 0) {
            const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                LOG_WRN("%s: device %s did not report memory; --fit will not use it\n",
                        __func__, ggml_backend_dev_name(dev));
            } else {
                free  = ret.back().free;
                total = ret.back().total;
            }
        }
        ret[i].free  = free;
        ret[i].total = total;
    }

    devs.clear();
    for (int i = 0; i < llama_model_n_devices(model); i++) {
        devs.push_back(llama_model_get_device(model, i));
    }

    hp_ngl         = llama_model_n_layer(model);
    if (mparams->load_mtp) {
        hp_ngl    += llama_model_n_layer_nextn(model);
    }
    hp_n_ctx_train = llama_model_n_ctx_train(model);
    hp_n_expert    = llama_model_n_expert(model);

    if (host_expert_bytes_out != nullptr) {
        host_expert_bytes_out->assign(nd, 0);
        for (size_t i = 0; i < nd; i++) {
            (*host_expert_bytes_out)[i] = llama_model_moe_host_expert_bytes(model, (uint32_t) i);
        }
    }

    // wip/fit-slab-accounting (Phase 1): the same host-expert bytes, but keyed by the REAL device that
    // owns the buffer.  `model->devices` is the scheduler's Meta wrapper under -sm tensor, so the
    // index-based lookup above sees 0 there; --fit needs the real devices to reserve per device.
    if (host_experts_out != nullptr) {
        host_experts_out->clear();
        const int32_t n_host = llama_model_moe_host_expert_dev_count(model);
        for (int32_t i = 0; i < n_host; i++) {
            ggml_backend_dev_t dev = nullptr;
            size_t bytes = 0;
            if (llama_model_moe_host_expert_dev(model, i, &dev, &bytes) && bytes > 0) {
                host_experts_out->emplace_back(dev, bytes);
            }
        }
    }

    common_memory_breakdown_print(ctx);

    llama_free(ctx);
    llama_model_free(model);
    llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);

    return ret;
}

common_device_memory_data_vec common_get_device_memory_data(
        const char * path_model,
        const llama_model_params * mparams,
        const llama_context_params * cparams,
        std::vector<ggml_backend_dev_t> & devs,
        uint32_t & hp_ngl,
        uint32_t & hp_n_ctx_train,
        uint32_t & hp_n_expert,
        ggml_log_level log_level) {
    std::vector<llama_device_memory_data> impl = common_get_device_memory_data_impl(
            path_model, mparams, cparams, devs, hp_ngl, hp_n_ctx_train, hp_n_expert, log_level);

    common_device_memory_data_vec ret(impl.size());
    for (size_t i = 0; i < impl.size(); i++) {
        ret[i].total   = impl[i].total;
        ret[i].free    = impl[i].free;
        ret[i].model   = impl[i].mb.model;
        ret[i].context = impl[i].mb.context;
        ret[i].compute = impl[i].mb.compute;
    }
    return ret;
}

// wip/slab-transient-absorption (step 2): the DERIVED reserve to use when the extra (draft) model cannot be
// MEASURED.  That is the normal MTP case: its context cannot be created while fitting (`qwen4exp requires
// ctx_other to be set`, and fitting runs before the target context exists), so the measurement throws and
// fit used to size the main model as if the draft did not exist -- the draft then came out of the slab's
// reserve, the outside cushion collapsed by ~D, and a lowered headroom aborted on an outside `cudaMalloc`.
//
// Layout: the weights + KV are ONE pool (they are tensor-split over the devices the draft shares with the
// main model) while the compute buffer (the draft's narrow-2 region) and its transients exist on EVERY
// device.  `n_dev` devices therefore need the compute tier each, plus the split of the rest.  Returns the
// TOTAL over `n_dev` devices.
static int64_t fit_extra_reserve_total(const common_fit_extra_model * extra, size_t n_dev) {
    if (extra == nullptr || extra->reserve_bytes == 0) {
        return 0;
    }
    if (n_dev == 0) {
        n_dev = 1;
    }
    const int64_t per_dev = (int64_t) 1024 << 20;   // the estimator's compute + slack tier, rounded up
    return (int64_t) extra->reserve_bytes + (int64_t) (n_dev - 1) * per_dev;
}

static void common_params_fit_impl(
        const char * path_model, struct llama_model_params * mparams, struct llama_context_params * cparams,
        float * tensor_split, struct llama_model_tensor_buft_override * tensor_buft_overrides,
        size_t * margins_s, uint32_t n_ctx_min, const common_fit_extra_model * extra, enum ggml_log_level log_level) {
    constexpr int64_t MiB = 1024*1024;
    typedef std::vector<llama_device_memory_data> dmds_t;
    const llama_model_params default_mparams = llama_model_default_params();

    std::vector<ggml_backend_dev_t> devs;
    uint32_t hp_ngl = 0; // hparams.n_gpu_layers
    uint32_t hp_nct = 0; // hparams.n_ctx_train
    uint32_t hp_nex = 0; // hparams.n_expert

    // with non-unified kv, we need to take into account n_streams
    // for example, if memory can hold more than model's trained context size, we must extend the n_ctx to hold enough n_streams
    const uint32_t n_streams  = cparams->kv_unified ? 1 : std::max<uint32_t>(1, cparams->n_seq_max);
    const bool     n_ctx_auto = cparams->n_ctx == 0;

    dmds_t   dmds_extra;       // memory of the extra model, laid out on the devices of the main model
    uint32_t n_ctx_extra = 0;  // context that memory was measured at

    // the extra model competes for the same memory as the main model, add it to every measurement
    // its memory is measured again whenever the context it follows changes
    auto add_extra_memory = [&](dmds_t & dmds) {
        if (extra == nullptr) {
            return;
        }

        if (dmds_extra.empty() || n_ctx_extra != cparams->n_ctx) {
            std::vector<ggml_backend_dev_t> devs_extra;
            uint32_t ngl_extra = 0;
            uint32_t nct_extra = 0;
            uint32_t nex_extra = 0;

            extra->cparams->n_ctx = cparams->n_ctx;

            LOG_TRC("%s: getting device memory data for the extra model at a context size of %" PRIu32 ":\n",
                __func__, cparams->n_ctx);

            dmds_t measured;
            try {
                measured = common_get_device_memory_data_impl(
                    extra->path_model, extra->mparams, extra->cparams, devs_extra, ngl_extra, nct_extra, nex_extra, log_level);
            } catch (const std::runtime_error & e) {
                // wip/slab-transient-absorption (step 2): the extra model could not be MEASURED.  That is the
                // normal MTP case (`qwen4exp requires ctx_other to be set` -- during fitting there is no target
                // context yet to point it at), and fitting the main model alone used to silently ignore the
                // draft's whole footprint: the draft then had to come out of the slab's reserve, the outside
                // cushion collapsed by ~D, and a lowered headroom aborted on an outside `cudaMalloc`.
                // Reserve the DERIVED estimate instead (`draft-reserve.h`) and split it over the devices the
                // draft shares with the main model.
                dmds_extra = dmds_t(devs.size() + 1);
                n_ctx_extra = cparams->n_ctx;
                if (extra->reserve_bytes == 0) {
                    LOG_WRN("%s: failed to measure the memory of the extra model, fitting without it: %s\n",
                            __func__, e.what());
                    return;
                }
                // Per device: the weights + KV follow the tensor split (the draft context uses the main model's
                // split), while the compute buffer (the draft's narrow-2 region) and its transients exist on
                // EVERY device it runs on.  So the portion of the estimate that is per-device is added to each
                // device, and the rest is distributed by the split.  For one device this is exactly the
                // estimate; for N > 1 it over-reserves by (N-1) x that portion, which is the safe direction.
                const size_t per_dev = (size_t) 1024 << 20;   // the estimator's compute + slack tier, rounded up
                const size_t split   = extra->reserve_bytes > per_dev ? extra->reserve_bytes - per_dev : 0;
                double share_sum = 0.0;
                std::vector<double> share(devs.size(), 0.0);
                for (size_t id = 0; id < devs.size(); id++) {
                    share[id] = tensor_split != nullptr && tensor_split[id] > 0.0f ? (double) tensor_split[id] : 0.0;
                    share_sum += share[id];
                }
                if (share_sum <= 0.0) {
                    for (size_t id = 0; id < devs.size(); id++) {
                        share[id] = 1.0 / (double) devs.size();
                    }
                    share_sum = 1.0;
                }
                for (size_t id = 0; id < devs.size(); id++) {
                    dmds_extra[id].mb.model   = (size_t) ((double) split * share[id] / share_sum);
                    dmds_extra[id].mb.compute = per_dev;
                }
                LOG_WRN("%s: the extra model cannot be measured during fitting (%s); reserving the DERIVED %.0f MiB "
                        "as a per-device-split reserve instead -- this is the MTP case, which needs ctx_other "
                        "while fitting has no target context yet\n", __func__, e.what(),
                        (double) extra->reserve_bytes / (double) (1024*1024));
                // `measured` is empty: the code below dereferences it, so the fallback must not fall through.
                return;
            }

            dmds_extra = dmds_t(devs.size() + 1);
            dmds_extra.back().mb = measured.back().mb;
            for (size_t je = 0; je < devs_extra.size(); je++) {
                for (size_t id = 0; id < devs.size(); id++) {
                    if (devs_extra[je] == devs[id]) {
                        dmds_extra[id].mb.model   += measured[je].mb.model;
                        dmds_extra[id].mb.context += measured[je].mb.context;
                        dmds_extra[id].mb.compute += measured[je].mb.compute;
                        break;
                    }
                }
            }
            if (extra->shares_model) {
                for (llama_device_memory_data & dmd : dmds_extra) {
                    dmd.mb.model = 0;
                }
            }

            n_ctx_extra = cparams->n_ctx;
        }

        for (size_t id = 0; id < dmds.size(); id++) {
            dmds[id].mb.model   += dmds_extra[id].mb.model;
            dmds[id].mb.context += dmds_extra[id].mb.context;
            dmds[id].mb.compute += dmds_extra[id].mb.compute;
        }
    };

    // step 1: get data for default parameters and check whether any changes are necessary in the first place

    LOG_TRC("%s: getting device memory data for initial parameters:\n", __func__);
    std::vector<size_t> host_expert_bytes;
    std::vector<std::pair<ggml_backend_dev_t, size_t>> host_experts;
    // wip/kv-residency (TODO #118) A3: the KV cache's own bytes, which is what the window is sized against
    std::vector<size_t> kv_bytes_full;
    dmds_t dmds_full = common_get_device_memory_data_impl(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level, &host_expert_bytes, &host_experts, &kv_bytes_full);

    // saturate instead of overflowing, this also preserves the UINT32_MAX sentinel of n_ctx_min:
    const uint32_t n_ctx_max       = (uint32_t) std::min<uint64_t>(uint64_t(hp_nct)    * n_streams, UINT32_MAX);
    const uint32_t n_ctx_min_total = (uint32_t) std::min<uint64_t>(uint64_t(n_ctx_min) * n_streams, UINT32_MAX);

    // llama_context would use only hp_nct in total for n_ctx == 0, resolve the context before measuring anything else:
    if (n_ctx_auto) {
        cparams->n_ctx = n_ctx_max;
        if (n_streams > 1) {
            LOG_TRC("%s: context size unset and KV cache not unified -> using %" PRIu32 " for %" PRIu32 " sequences:\n",
                __func__, n_ctx_max, n_streams);
            dmds_full = common_get_device_memory_data_impl(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level, nullptr, nullptr, &kv_bytes_full);
        }
    }
    add_extra_memory(dmds_full);

    const size_t nd = devs.size(); // number of devices

    std::vector<int64_t> margins; // this function uses int64_t rather than size_t for memory sizes to more conveniently handle deficits
    margins.reserve(nd);
    if (nd == 0) {
        margins.push_back(margins_s[0]);
    } else {
        for (size_t id = 0; id < nd; id++) {
            margins.push_back(margins_s[id]);
        }
    }

    // wip/fit-slab-accounting (Phase 1).  Reserve the MoE expert cache's arena budget AND the
    // movable-boundary slab's headroom in --fit's per-device margin, so --fit sizes the context around
    // them instead of letting both claim the free VRAM afterwards.
    //
    //   G1  an explicit MOE_EXPERT_CACHE_MIB is reserved verbatim (it used to be ignored: --fit sized the
    //       context as if the arena did not exist).
    //   G2  the slab headroom is a HARD floor: hipBLASLt loads its Tensile code objects and workspace
    //       BEHIND the application's back and the slab cannot serve them, so a thin headroom corrupts
    //       (measured: '!!!!!!!!' output, arena thrash, a 32.5 -> 13.0 GiB VRAM collapse).  See
    //       wip/fit-slab-accounting/README.md.
    //
    // Both apply only when the model has host-resident experts (the -ncmoe path); a fully resident model
    // is untouched.  The reservation is keyed on the REAL device that owns the host-expert buffer
    // (host_experts), not on model->devices: under -sm tensor the latter is the scheduler's Meta wrapper,
    // which is why the old code silently reserved nothing there.  The tensor-split path applies the same
    // reservation to its per-simple-device targets below.
    const common_fit_cache_reserve cache_reserve = common_fit_cache_reserve::from_env();
    int64_t host_total = 0;
    for (const auto & he : host_experts) {
        host_total += (int64_t) he.second;
    }
    if (host_total > 0) {
        int64_t reserve_total = 0;
        bool    refused_shared = false;
        for (size_t id = 0; id < nd; id++) {
            const int64_t before = margins[id];
            const bool refused = cache_reserve.arena_refused(devs[id]);
            refused_shared = refused_shared || (refused && host_expert_bytes[id] > 0);
            margins[id] = cache_reserve.margin(devs[id], host_expert_bytes[id], margins[id]);
            reserve_total += margins[id] - before;
        }
        if (refused_shared) {
            // wip/umem-arena-guard: the backend declines the arena on a shared host/device pool, so do
            // not withhold slab headroom or arena bytes from the fit budget either.
            LOG_WRN("%s: host-resident MoE experts on a shared host/device memory pool (integrated GPU): the "
                    "expert cache is not used there, so no slab headroom or arena is reserved in the --fit "
                    "margin.  Re-enabling it where the platform really partitions host/device memory needs "
                    "MOE_EXPERT_CACHE_UNIFIED=1 plus an explicit MOE_EXPERT_CACHE_MIB and "
                    "GGML_CUDA_SLAB_RESERVE_MIB.\n", __func__);
        }
        if (reserve_total > 0) {
            const char * reserve_env = getenv("MOE_EXPERT_CACHE_RESERVE_MIB");
            const int    reserve_mib = reserve_env ? atoi(reserve_env) : 1024;
            LOG_WRN("%s: host-resident MoE experts active: reserving the slab headroom plus the expert-cache arena "
                    "(%s) in the --fit margin before fitting the context (%.0f MiB this pass over %.0f MiB of host "
                    "experts).  --fit sizes the context around it; the arena then takes any remaining free device "
                    "memory keeping MOE_EXPERT_CACHE_RESERVE_MIB=%d MiB per device.\n",
                    __func__,
                    cache_reserve.cache_auto ? "auto floor: MOE_EXPERT_CACHE_MIN_MIB / MIN_RES_PCT"
                                             : "explicit MOE_EXPERT_CACHE_MIB",
                    (double) reserve_total / MiB, (double) host_total / MiB, reserve_mib);
        }
    }

    // ===== tensor split (-sm tensor / LLAMA_SPLIT_MODE_TENSOR) =====
    //
    // Under tensor split every GPU wraps into a single Meta device, so the layer-granular algorithm
    // below (whole layers per device, layer counts written into tensor_split) does not apply.  The
    // measurement above therefore yields one Meta entry whose components have mixed meaning:
    //   - model   is the whole offloaded tensor set, counted once (a per-device total once split)
    //   - context and compute are per-device: the Meta buffer reports its largest sub-buffer
    // The fit estimates total device memory as model + n_devices * (context + compute), assigns
    // tensor_split proportional to each device's free memory minus its margin (so the loader's
    // proportional sharding leaves every device at the same fill fraction), then reduces n_ctx and,
    // if needed, n_gpu_layers until the estimate fits the summed targets.
    if (mparams->split_mode == LLAMA_SPLIT_MODE_TENSOR || mparams->split_mode == LLAMA_SPLIT_MODE_RELAXED_TENSOR) {
        std::vector<ggml_backend_dev_t> tdevs; // the "simple" devices behind the Meta wrapper
        for (ggml_backend_dev_t dev : devs) {
            if (ggml_backend_dev_is_meta(dev)) {
                const size_t n_dev = ggml_backend_meta_dev_n_devs(dev);
                for (size_t i = 0; i < n_dev; i++) {
                    tdevs.push_back(ggml_backend_meta_dev_simple_dev(dev, i));
                }
            } else {
                tdevs.push_back(dev);
            }
        }
        if (tdevs.empty()) {
            throw common_params_fit_exception("no devices behind the Meta device, abort");
        }
        const size_t ntd = tdevs.size();
        if (mparams->n_gpu_layers != default_mparams.n_gpu_layers) {
            throw common_params_fit_exception("n_gpu_layers already set by user to " + std::to_string(mparams->n_gpu_layers) + ", abort");
        }
        if (!tensor_split) {
            throw common_params_fit_exception("did not provide a buffer to write the tensor_split to, abort");
        }
        // per-device target = free memory minus the requested margin
        std::vector<int64_t> ttarget(ntd);
        int64_t target_sum = 0;
        for (size_t id = 0; id < ntd; id++) {
            size_t free = 0;
            size_t total = 0;
            ggml_backend_dev_memory(tdevs[id], &free, &total);
            // wip/fit-slab-accounting: this branch consumes margins_s (per simple device), not `margins`,
            // so apply the cache/slab reservation here too, per real device.
            size_t host_bytes_t = 0;
            for (const auto & he : host_experts) {
                if (he.first == tdevs[id]) {
                    host_bytes_t = he.second;
                    break;
                }
            }
            const int64_t tmargin = host_total > 0
                ? cache_reserve.margin(tdevs[id], host_bytes_t, (int64_t) margins_s[id])
                : (int64_t) margins_s[id];
            ttarget[id] = std::max<int64_t>(0, (int64_t) free - tmargin);
            target_sum += ttarget[id];
            LOG_TRC("%s: tensor split: device %zu target %" PRId64 " MiB (free %" PRId64 " MiB, margin %zu MiB)\n",
                __func__, id, ttarget[id]/MiB, (int64_t) free/MiB, margins_s[id]/MiB);
        }

        // A user-pinned -ts is honored: the fit keeps the requested balance and only chooses
        // n_ctx / n_gpu_layers so that balance fits.  Without one, the split is pinned
        // proportional to the targets.
        bool ts_pinned = false;
        std::vector<double> ratio(ntd, 0.0);
        if (mparams->tensor_split) {
            double sum = 0.0;
            for (size_t id = 0; id < ntd; id++) {
                ratio[id] = std::max(0.0, (double) mparams->tensor_split[id]);
                sum += ratio[id];
            }
            if (sum > 0.0) {
                ts_pinned = true;
                for (size_t id = 0; id < ntd; id++) {
                    ratio[id] /= sum;
                }
                LOG_TRC("%s: tensor split: honoring the user tensor_split ratios\n", __func__);
            }
        }
        if (!ts_pinned) {
            // tensor_split proportional to the target, so the loader shards the model in the same
            // ratio.  Only the ratios matter; use MiB so the values stay small and print cleanly
            // (llama-fit-params renders tensor_split as uint32_t).
            double sum = 0.0;
            for (size_t id = 0; id < ntd; id++) {
                sum += (double) ttarget[id];
            }
            for (size_t id = 0; id < ntd; id++) {
                tensor_split[id] = (float) (ttarget[id] / MiB);
                ratio[id] = sum > 0.0 ? (double) ttarget[id] / sum : 0.0;
            }
            for (size_t id = ntd; id < llama_max_devices(); id++) {
                tensor_split[id] = 0.0f;
            }
            mparams->tensor_split = tensor_split;
        }

        // The estimate is a per-device total: a device holding ratio r gets r * D, so the fit must
        // satisfy r_i * D <= target_i for every used device, i.e. D <= min_i(target_i / r_i).
        // With the auto split (r_i = target_i / sum(target)) that is exactly sum(target), so the
        // two cases share one budget.
        int64_t budget = target_sum;
        for (size_t id = 0; id < ntd; id++) {
            if (ratio[id] > 0.0) {
                budget = std::min<int64_t>(budget, (int64_t) ((double) ttarget[id] / ratio[id]));
            }
        }

        // total device memory estimate; the Meta buffer reports the largest sub-buffer for
        // context/compute, so counting them once per device is the safe (slightly high) direction
        auto estimate = [&](const dmds_t & dmds) -> int64_t {
            int64_t model = 0;
            int64_t ctx   = 0;
            int64_t comp  = 0;
            for (size_t id = 0; id < nd; id++) {
                model += dmds[id].mb.model;
                ctx   += dmds[id].mb.context;
                comp  += dmds[id].mb.compute;
            }
            return model + (int64_t) ntd * (ctx + comp);
        };
        // the Meta devices of the main and extra models are distinct objects, so add_extra_memory()
        // could not attribute the extra model to the main Meta entry -- do it here
        auto extra_estimate = [&]() -> int64_t {
            if (extra == nullptr) {
                return 0;
            }
            extra->cparams->n_ctx = cparams->n_ctx;
            std::vector<ggml_backend_dev_t> devs_extra;
            uint32_t ngl_extra = 0;
            uint32_t nct_extra = 0;
            uint32_t nex_extra = 0;
            dmds_t measured;
            try {
                measured = common_get_device_memory_data_impl(
                    extra->path_model, extra->mparams, extra->cparams, devs_extra, ngl_extra, nct_extra, nex_extra, log_level);
            } catch (const std::runtime_error & e) {
                // wip/slab-transient-absorption (step 2): the derived reserve, not zero -- see
                // `fit_extra_reserve_total` (this is the MTP / Meta-device path, where the measurement always
                // fails because the draft context needs ctx_other).
                const int64_t fb = fit_extra_reserve_total(extra, devs.size());
                if (fb == 0) {
                    LOG_WRN("%s: failed to measure the memory of the extra model, fitting without it: %s\n",
                            __func__, e.what());
                    return 0;
                }
                LOG_WRN("%s: the extra model cannot be measured during fitting (%s); reserving the DERIVED "
                        "%.0f MiB instead (MTP needs ctx_other, which fitting does not have yet)\n", __func__,
                        e.what(), (double) fb / (double) (1024*1024));
                return fb;
            }
            int64_t model = 0;
            int64_t ctx   = 0;
            int64_t comp  = 0;
            for (size_t id = 0; id + 1 < measured.size(); id++) {
                model += measured[id].mb.model;
                ctx   += measured[id].mb.context;
                comp  += measured[id].mb.compute;
            }
            if (extra->shares_model) {
                model = 0; // the extra context runs on the main model's weights
            }
            return model + (int64_t) ntd * (ctx + comp);
        };

        int64_t used_total = estimate(dmds_full) + extra_estimate();
        if (ts_pinned) {
            LOG_TRC("%s: tensor split: %zu devices, estimated use %" PRId64 " MiB vs. %" PRId64 " MiB effective budget "
                "(sum of targets %" PRId64 " MiB, pinned tensor_split)\n",
                __func__, ntd, used_total/MiB, budget/MiB, target_sum/MiB);
        } else {
            LOG_TRC("%s: tensor split: %zu devices, estimated use %" PRId64 " MiB vs. %" PRId64 " MiB target\n",
                __func__, ntd, used_total/MiB, budget/MiB);
        }

        if (used_total <= budget) {
            LOG_TRC("%s: tensor split: no changes needed\n", __func__);
            return;
        }

        // reduce the context size first
        if (n_ctx_auto && n_ctx_max > n_ctx_min_total) {
            const uint32_t n_ctx_save = cparams->n_ctx;
            cparams->n_ctx = n_ctx_min_total;
            dmds_t dmds_min = common_get_device_memory_data_impl(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);
            const int64_t used_min = estimate(dmds_min) + extra_estimate();
            cparams->n_ctx = n_ctx_save;

            if (used_min <= budget && used_total > used_min) {
                const uint32_t n_ctx = n_ctx_min_total + (uint32_t) (
                    (int64_t) (n_ctx_max - n_ctx_min_total) * (budget - used_min) / (used_total - used_min));
                const uint32_t align = 256 * n_streams;
                cparams->n_ctx = std::max(std::min(n_ctx, n_ctx_max) - std::min(n_ctx, n_ctx_max) % align, n_ctx_min_total);
                LOG_TRC("%s: tensor split: context size reduced from %" PRIu32 " to %" PRIu32 "\n",
                    __func__, n_ctx_max, cparams->n_ctx);
                return;
            }
            LOG_TRC("%s: tensor split: context reduction alone is not enough (needs %" PRId64 " MiB at the minimum context), reducing the offloaded layers\n",
                __func__, used_min/MiB);
        } else if (!n_ctx_auto) {
            LOG_TRC("%s: context size set by user to %" PRIu32 " -> no change\n", __func__, cparams->n_ctx);
        }

        // reduce the number of offloaded layers with a binary search over the estimate
        uint32_t ngl_lo = 0;
        uint32_t ngl_hi = hp_ngl + 1;
        uint32_t ngl_best = 0;
        int64_t  used_best = 0;
        while (ngl_lo <= ngl_hi) {
            const uint32_t ngl_mid = ngl_lo + (ngl_hi - ngl_lo) / 2;
            mparams->n_gpu_layers = (int32_t) ngl_mid;
            dmds_t dmds_ngl = common_get_device_memory_data_impl(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);
            const int64_t used = estimate(dmds_ngl) + extra_estimate();
            LOG_TRC("%s: tensor split: n_gpu_layers=%" PRIu32 " -> %" PRId64 " MiB\n", __func__, ngl_mid, used/MiB);
            if (used <= budget) {
                ngl_best  = ngl_mid;
                used_best = used;
                ngl_lo    = ngl_mid + 1;
            } else if (ngl_mid == 0) {
                break;
            } else {
                ngl_hi = ngl_mid - 1;
            }
        }
        mparams->n_gpu_layers = (int32_t) ngl_best;
        LOG_TRC("%s: tensor split: offloading %" PRIu32 " layers, estimated use %" PRId64 " MiB\n",
            __func__, ngl_best, used_best/MiB);
        return;
    }

    // ===== wip/kv-residency (TODO #118) A3: the windowed KV =====
    //
    // A windowed KV keeps the newest W cells of the context in VRAM and puts everything older in pinned
    // system RAM, so the KV's VRAM cost becomes W cells instead of `-c` cells and the expert arena (or a
    // larger context) gets the difference.  It is NOT free: every spilled cell is read over the PCIe link on
    // every step, so the cost is `(used - W) x cell_bytes / link` per token -- linear in the resident
    // fraction, measured to a few percent (wip/kv-residency/A2-MIXED-BACKING.md §3.2).  That is why the
    // window is a DECISION and not a default:
    //
    //   the window pays when `(kv_size - depth) x cell_bytes` of VRAM is worth more than
    //   `(depth - W) x cell_bytes` of per-token spill -- i.e. when `-c` is much larger than the depth you run
    //   at.  At `-c` ~1.4x the depth it is a measured loss (Flash-Next: 0.39 GiB freed, -14.6 % decode), at
    //   `-c` ~4x it is a measured win (3.63 GiB freed, +13.1 % prefill, 0 % decode -- nothing spills at the
    //   run's depth).  So `LLAMA_KV_RESIDENT_CELLS=<cells>` states the window and `=auto` lets this function
    //   size it; `-c` close to what fits should keep the cache fully resident.
    //
    // Only one device is sized here: the multi-device (tensor/layer split) paths decide layer placement in
    // ways this campaign has no hardware to validate.
    if (llama_kv_region_is_enabled()) {
        const uint32_t kv_w_req = llama_kv_region_resident_cells_requested();
        const bool     kv_auto  = kv_w_req == UINT32_MAX;
        const bool     kv_fixed = kv_w_req != 0 && kv_w_req != UINT32_MAX;

        if (kv_auto || kv_fixed) {
            const bool have_kv = nd == 1 && kv_bytes_full.size() == 1 && kv_bytes_full[0] != 0;
            if (!have_kv) {
                LOG_WRN("%s: the windowed KV needs one region-backed KV cache to size against (this config has "
                        "%zu device(s)); the window is left as configured\n", __func__, nd);
            } else if (kv_auto && n_ctx_auto) {
                // `-c` is ours to choose: prefer a context that fits FULLY RESIDENT (no per-token cost) and say
                // how to get the bigger one, rather than silently trading decode speed for context length.
                LOG_WRN("%s: LLAMA_KV_RESIDENT_CELLS=auto only spills a context the user pinned; with -c unset "
                        "the fit will size a fully resident context instead (%" PRIu32 " cells).  Pass `-c <cells>` "
                        "next to `auto` to keep a bigger context and spill the tail.\n", __func__, n_ctx_max);
            } else {
                const size_t   kv_bytes   = kv_bytes_full[0];
                const uint32_t kv_cells   = cparams->n_ctx / n_streams;   // kv_size of one cache
                const size_t   cell_bytes = kv_cells != 0 ? kv_bytes / kv_cells : 0;   // every tensor, one cell
                const int64_t  used_other = (int64_t) dmds_full[0].mb.total() - (int64_t) kv_bytes;
                const int64_t  free_dev   = dmds_full[0].free;
                const int64_t  budget_kv  = free_dev - margins[0] - used_other;   // VRAM the window may take
                const size_t   ram_budget = fit_kv_spill_ram_budget(devs[0]);

                size_t w = kv_fixed ? (size_t) kv_w_req : 0;
                if (kv_auto) {
                    w = (budget_kv > 0 && cell_bytes != 0) ? (size_t) (budget_kv / (int64_t) cell_bytes) : 0;
                    if (w > kv_cells) {
                        w = kv_cells;
                    }
                }
                bool ram_clamped = false;
                if (ram_budget != SIZE_MAX && cell_bytes != 0 && kv_cells > w &&
                        (size_t) (kv_cells - w) * cell_bytes > ram_budget) {
                    // the spill is PINNED, non-swappable RAM: widen the window until the worst case fits
                    // (`llama_kv_cache` applies the same clamp at creation; doing it here keeps the report and
                    // the decision in one place)
                    const size_t tail_cells = ram_budget / cell_bytes;
                    const size_t w_ram      = kv_cells > tail_cells ? (size_t) kv_cells - tail_cells : (size_t) kv_cells;
                    if (w_ram > w) {
                        w = w_ram;
                        ram_clamped = true;
                    }
                }
                const size_t spill_bytes = kv_cells > w ? (size_t) (kv_cells - w) * cell_bytes : 0;
                const size_t res_bytes   = kv_cells > w ? (size_t) w * cell_bytes : kv_bytes;
                const int64_t projected  = used_other + (int64_t) res_bytes;
                const int64_t proj_free  = free_dev - projected;
                const float   gbps       = fit_kv_link_gbps(devs[0]);

                if (w >= kv_cells) {
                    // a zero-width spill saves nothing: either the whole KV already fits (the common case) or
                    // it does not fit at all, and today's fit-or-fail is then the honest answer
                    if (kv_auto) {
                        llama_kv_region_set_resident_cells(0);
                        LOG_WRN("%s: LLAMA_KV_RESIDENT_CELLS=auto: no window is needed -- the %" PRIu32
                                "-cell KV (%.1f MiB) fits fully resident (%.1f MiB of VRAM left over the %" PRId64
                                " MiB target)\n", __func__, kv_cells, (double) kv_bytes / MiB,
                                (double) budget_kv / MiB, margins[0] / MiB);
                    } else {
                        LOG_WRN("%s: the windowed KV is not active: W=%u leaves no spilled cells at %" PRIu32
                                " cells, so the KV stays fully resident\n", __func__, kv_w_req, kv_cells);
                    }
                } else if (proj_free >= margins[0]) {
                    // the window makes the configuration fit: keep `-c` and hand W down
                    llama_kv_region_set_resident_cells((uint32_t) w);
                    LOG_WRN("%s: the KV cache is WINDOWED: %zu of %u cells resident (%.1f MiB of %.1f MiB), "
                            "%.1f MiB spilled to pinned system RAM%s\n",
                            __func__, w, kv_cells, (double) res_bytes / MiB, (double) kv_bytes / MiB,
                            (double) spill_bytes / MiB, ram_clamped ? " (widened by the pinned-RAM ceiling)" : "");
                    LOG_WRN("%s:   -> it costs %.1f MiB less VRAM (%.1f MiB projected vs a %" PRId64 " MiB target, "
                            "%.1f MiB free); the spill reads %.1f KiB per cell over the link (%s) on every step, "
                            "so a token at the full context reads %.1f MiB = about %.1f ms at %s, and nothing "
                            "spills below depth %zu\n",
                            __func__, (double) (kv_bytes - res_bytes) / MiB, (double) projected / MiB,
                            margins[0] / MiB, (double) proj_free / MiB, (double) cell_bytes / 1024.0,
                            gbps > 0.0f ? (std::to_string((int) (gbps + 0.5f)) + " GB/s, calibrated").c_str()
                                        : "the measured link rate",
                            (double) spill_bytes / MiB,
                            gbps > 0.0f ? (double) spill_bytes / ((double) gbps * 1e6) : 0.0,
                            gbps > 0.0f ? "that rate" : "the link's own rate", w);
                    return;
                } else {
                    LOG_WRN("%s: the windowed KV does not make this fit either: %zu of %u cells resident leaves "
                            "%.1f MiB projected against a %" PRId64 " MiB target (%.1f MiB over); the window is "
                            "dropped and the context is fitted as before\n",
                            __func__, w, kv_cells, (double) projected / MiB, margins[0] / MiB,
                            (double) (margins[0] - proj_free) / MiB);
                    llama_kv_region_set_resident_cells(0);
                }
            }
        }
    }

    std::vector<std::string> dev_names;
    {
        dev_names.reserve(nd);
        size_t max_length = 0;
        for (const auto & dev : devs) {
            std::string name = ggml_backend_dev_name(dev);
            name += " (";
            name += ggml_backend_dev_description(dev);
            name += ")";
            dev_names.push_back(name);
            max_length = std::max(max_length, name.length());
        }
        for (std::string & dn : dev_names) {
            dn.insert(dn.end(), max_length - dn.length(), ' ');
        }
    }

    int64_t sum_free            = 0;
    int64_t sum_projected_free  = 0;
    int64_t sum_projected_used  = 0;
    int64_t sum_projected_model = 0;
    std::vector<int64_t> projected_free_per_device;
    projected_free_per_device.reserve(nd);

    if (nd == 0) {
        sum_projected_used = dmds_full.back().mb.total();
        sum_free           = dmds_full.back().total;
        sum_projected_free = sum_free - sum_projected_used;
        LOG_TRC("%s: projected to use %" PRId64 " MiB of host memory vs. %" PRId64 " MiB of total host memory\n",
            __func__, sum_projected_used/MiB, sum_free/MiB);
        if (sum_projected_free >= margins[0]) {
            LOG_TRC("%s: will leave %" PRId64 " >= %" PRId64 " MiB of system memory, no changes needed\n",
                __func__, sum_projected_free/MiB, margins[0]/MiB);
            return;
        }
    } else {
        if (nd > 1) {
            LOG_TRC("%s: projected memory use with initial parameters [MiB]:\n", __func__);
        }
        for (size_t id = 0; id < nd; id++) {
            const llama_device_memory_data & dmd = dmds_full[id];

            const int64_t projected_used = dmd.mb.total();
            const int64_t projected_free = dmd.free - projected_used;
            projected_free_per_device.push_back(projected_free);

            sum_free            += dmd.free;
            sum_projected_used  += projected_used;
            sum_projected_free  += projected_free;
            sum_projected_model += dmd.mb.model;

            if (nd > 1) {
                LOG_TRC("%s:   - %s: %6" PRId64 " total, %6" PRId64 " used, %6" PRId64 " free vs. target of %6" PRId64 "\n",
                    __func__, dev_names[id].c_str(), dmd.total/MiB, projected_used/MiB, projected_free/MiB, margins[id]/MiB);
            }
        }
        assert(sum_free >= 0 && sum_projected_used >= 0);
        LOG_TRC("%s: projected to use %" PRId64 " MiB of device memory vs. %" PRId64 " MiB of free device memory\n",
            __func__, sum_projected_used/MiB, sum_free/MiB);
        if (nd == 1) {
            if (projected_free_per_device[0] >= margins[0]) {
                LOG_TRC("%s: will leave %" PRId64 " >= %" PRId64 " MiB of free device memory, no changes needed\n",
                    __func__, projected_free_per_device[0]/MiB, margins[0]/MiB);
                return;
            }
        } else {
            bool changes_needed = false;
            for (size_t id = 0; id < nd; id++) {
                if (projected_free_per_device[id] < margins[id]) {
                    changes_needed = true;
                    break;
                }
            }
            if (!changes_needed) {
                LOG_TRC("%s: targets for free memory can be met on all devices, no changes needed\n", __func__);
                return;
            }
        }
    }

    // step 2: try reducing memory use by reducing the context size

    {
        int64_t global_surplus = sum_projected_free;
        if (nd == 0) {
            global_surplus -= margins[0];
        } else {
            for (size_t id = 0; id < nd; id++) {
                global_surplus -= margins[id];
            }
        }
        if (global_surplus < 0) {
            if (nd <= 1) {
                LOG_TRC("%s: cannot meet free memory target of %" PRId64 " MiB, need to reduce device memory by %" PRId64 " MiB\n",
                    __func__, margins[0]/MiB, -global_surplus/MiB);
            } else {
                LOG_TRC(
                    "%s: cannot meet free memory targets on all devices, need to use %" PRId64 " MiB less in total\n",
                    __func__, -global_surplus/MiB);
            }
            if (n_ctx_auto) {
                if (n_ctx_max > n_ctx_min_total) {
                    int64_t sum_used_target = sum_free;
                    if (nd == 0) {
                        sum_used_target -= margins[0];
                    } else {
                        for (size_t id = 0; id < nd; id++) {
                            sum_used_target -= margins[id];
                        }
                    }
                    if (nd > 1) {
                        // for multiple devices we need to be more conservative in terms of how much context we think can fit:
                        //   - for dense models only whole layers can be assigned to devices
                        //   - for MoE models only whole tensors can be assigned to devices, which we estimate to be <= 1/3 of a layer
                        //   - on average we expect a waste of 0.5 layers/tensors per device
                        //   - use slightly more than the expected average for nd devices to be safe
                        const int64_t model_per_layer = sum_projected_model / std::min(uint32_t(mparams->n_gpu_layers), hp_ngl);
                        sum_used_target -= (nd + 1) * model_per_layer / (hp_nex == 0 ? 2 : 6);
                    }

                    int64_t sum_projected_used_min_ctx = 0;
                    cparams->n_ctx = n_ctx_min_total;
                    dmds_t dmds_min_ctx = common_get_device_memory_data_impl(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);
                    add_extra_memory(dmds_min_ctx);
                    if (nd == 0) {
                        sum_projected_used_min_ctx = dmds_min_ctx.back().mb.total();
                    } else {
                        for (size_t id = 0; id < nd; id++) {
                            sum_projected_used_min_ctx += dmds_min_ctx[id].mb.total();
                        }
                    }
                    if (sum_used_target > sum_projected_used_min_ctx) {
                        // linear interpolation between minimum and maximum context size:
                        cparams->n_ctx += (n_ctx_max - n_ctx_min_total) * (sum_used_target - sum_projected_used_min_ctx)
                            / (sum_projected_used - sum_projected_used_min_ctx);
                        // round down context for CUDA backend, keep it divisible by the number of streams:
                        const uint32_t align = 256 * n_streams;
                        cparams->n_ctx = std::max(cparams->n_ctx - cparams->n_ctx % align, n_ctx_min_total);

                        const int64_t bytes_per_ctx = (sum_projected_used - sum_projected_used_min_ctx) / (n_ctx_max - n_ctx_min_total);
                        const int64_t memory_reduction = (n_ctx_max - cparams->n_ctx) * bytes_per_ctx;
                        LOG_TRC("%s: context size reduced from %" PRIu32 " to %" PRIu32 " -> need %" PRId64 " MiB less memory in total\n",
                            __func__, n_ctx_max, cparams->n_ctx, memory_reduction/MiB);
                        if (nd <= 1) {
                            LOG_TRC("%s: entire model can be fit by reducing context\n", __func__);
                            return;
                        }
                        LOG_TRC("%s: entire model should be fit across devices by reducing context\n", __func__);
                    } else {
                        const int64_t memory_reduction = sum_projected_used - sum_projected_used_min_ctx;
                        LOG_TRC("%s: context size reduced from %" PRIu32 " to %" PRIu32 " -> need %" PRId64 " MiB less memory in total\n",
                            __func__, n_ctx_max, cparams->n_ctx, memory_reduction/MiB);
                    }
                } else {
                    if (n_ctx_min == UINT32_MAX) {
                        LOG_TRC("%s: user has requested full context size of %" PRIu32 " -> no change\n", __func__, n_ctx_max);
                    } else {
                        LOG_TRC("%s: default model context size is %" PRIu32 " which is <= the min. context size of %" PRIu32 " -> no change\n",
                            __func__, n_ctx_max, n_ctx_min_total);
                    }
                }
            } else {
                LOG_TRC("%s: context size set by user to %" PRIu32 " -> no change\n", __func__, cparams->n_ctx);

                // The free-memory target cannot be met and the context is not ours to change, so the fit
                // ends here with the run still over budget.  Say so at ERROR with the arithmetic and the
                // ways out: this is issue #219's state (a 16 GiB card asked for more than it has), where
                // one W line was read as harmless and the run went on to a device fault.
                int64_t sum_margin = 0;
                if (nd == 0) {
                    sum_margin = margins[0];
                } else {
                    for (size_t id = 0; id < nd; id++) {
                        sum_margin += margins[id];
                    }
                }
                int64_t ctx_mib = 0;
                if (nd == 0) {
                    ctx_mib = dmds_full.back().mb.context;
                } else {
                    for (size_t id = 0; id < nd; id++) {
                        ctx_mib += dmds_full[id].mb.context;
                    }
                }
                const int64_t shortfall   = -global_surplus;
                const int64_t mib_per_1k  = cparams->n_ctx > 0 ? ctx_mib * 1024 / cparams->n_ctx : 0;
                LOG_ERR("%s: the requested configuration does NOT fit in device memory\n", __func__);
                LOG_ERR("%s:   projected %" PRId64 " MiB used vs. %" PRId64 " MiB free, target margin %" PRId64
                        " MiB -> over budget by %" PRId64 " MiB\n",
                        __func__, sum_projected_used/MiB, sum_free/MiB, sum_margin/MiB, shortfall/MiB);
                LOG_ERR("%s:   the context is fixed at %" PRIu32 " by the user (%" PRId64 " MiB, about %" PRId64
                        " MiB per 1024 tokens), so --fit will not reduce it\n",
                        __func__, (uint32_t) cparams->n_ctx, ctx_mib/MiB, mib_per_1k);
                if (shortfall < ctx_mib) {
                    LOG_ERR("%s:   about %" PRId64 " fewer context tokens would fit; lower -c, lower -ngl, or\n"
                            "%s:   unset -c/-ngl so --fit can choose.  Running over budget ends in an\n"
                            "%s:   allocator out-of-memory abort, and on Windows/WDDM a paged device\n"
                            "%s:   allocation can fault the GPU instead of failing cleanly\n",
                            __func__, shortfall * 1024 / std::max<int64_t>(1, mib_per_1k), __func__, __func__, __func__);
                } else {
                    LOG_ERR("%s:   even the smallest context cannot make this fit: lower -ngl (offload fewer\n"
                            "%s:   layers) or use a smaller quant.  Running over budget ends in an\n"
                            "%s:   allocator out-of-memory abort, and on Windows/WDDM a paged device\n"
                            "%s:   allocation can fault the GPU instead of failing cleanly\n",
                            __func__, __func__, __func__, __func__);
                }
            }
        }
    }
    if (nd == 0) {
        throw common_params_fit_exception("was unable to fit model into system memory by reducing context, abort");
    }

    if (mparams->n_gpu_layers != default_mparams.n_gpu_layers) {
        throw common_params_fit_exception("n_gpu_layers already set by user to " + std::to_string(mparams->n_gpu_layers) + ", abort");
    }
    if (nd > 1) {
        if (!tensor_split) {
            throw common_params_fit_exception("did not provide a buffer to write the tensor_split to, abort");
        }
        if (mparams->tensor_split) {
            for (size_t id = 0; id < nd; id++) {
                if (mparams->tensor_split[id] != 0.0f) {
                    throw common_params_fit_exception("model_params::tensor_split already set by user, abort");
                }
            }
        }
        if (mparams->split_mode == LLAMA_SPLIT_MODE_ROW) {
            throw common_params_fit_exception("changing weight allocation for LLAMA_SPLIT_MODE_ROW not implemented, abort");
        }
    }
    if (!tensor_buft_overrides) {
        throw common_params_fit_exception("did not provide buffer to set tensor_buft_overrides, abort");
    }
    if (mparams->tensor_buft_overrides && (mparams->tensor_buft_overrides->pattern || mparams->tensor_buft_overrides->buft)) {
        throw common_params_fit_exception("model_params::tensor_buft_overrides already set by user, abort");
    }

    // step 3: iteratively fill the back to front with "dense" layers
    //   - for a dense model simply fill full layers, giving each device a contiguous slice of the model
    //   - for a MoE model, same as dense model but with all MoE tensors in system memory

    // utility function that returns a static C string matching the tensors for a specific layer index and layer fraction:
    auto get_overflow_pattern = [&](const size_t il, const common_layer_fraction_t lf) -> const char * {
        constexpr size_t n_strings = 1000;
        if (il >= n_strings) {
            throw std::runtime_error("at most " + std::to_string(n_strings) + " model layers are supported");
        }
        switch (lf) {
            case LAYER_FRACTION_ATTN: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_(gate|up|gate_up|down).*";
                }
                return patterns[il].c_str();
            }
            case LAYER_FRACTION_UP: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_(gate|gate_up|down).*";
                }
                return patterns[il].c_str();
            }
            case LAYER_FRACTION_GATE: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_down.*";
                }
                return patterns[il].c_str();
            }
            case LAYER_FRACTION_MOE: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_(up|down|gate_up|gate)_(ch|)exps";
                }
                return patterns[il].c_str();
            }
            default:
                GGML_ABORT("fatal error");
        }
    };

    struct ngl_t {
        uint32_t n_layer = 0; // number of total layers
        uint32_t n_part  = 0; // number of partial layers, <= n_layer

        // for the first partial layer varying parts can overflow, all further layers use LAYER_FRACTION_MOE:
        common_layer_fraction_t overflow_type = LAYER_FRACTION_MOE;

        uint32_t n_full() const {
            assert(n_layer >= n_part);
            return n_layer - n_part;
        }
    };

    const size_t ntbo = llama_max_tensor_buft_overrides();

    // utility function to set n_gpu_layers and tensor_split
    auto set_ngl_tensor_split_tbo = [&](
            const std::vector<ngl_t> & ngl_per_device,
            const std::vector<ggml_backend_buffer_type_t> & overflow_bufts,
            llama_model_params & mparams) {
        mparams.n_gpu_layers = 0;
        for (size_t id = 0; id < nd; id++) {
            mparams.n_gpu_layers += ngl_per_device[id].n_layer;
            if (nd > 1) {
                tensor_split[id] = ngl_per_device[id].n_layer;
            }
        }
        assert(uint32_t(mparams.n_gpu_layers) <= hp_ngl + 1);
        uint32_t il0 = hp_ngl + 1 - mparams.n_gpu_layers; // start index for tensor buft overrides

        mparams.tensor_split = tensor_split;

        size_t itbo = 0;
        for (size_t id = 0; id < nd; id++) {
            il0 += ngl_per_device[id].n_full();
            for (uint32_t il = il0; il < il0 + ngl_per_device[id].n_part; il++) {
                if (itbo + 1 >= ntbo) {
                    tensor_buft_overrides[itbo].pattern = nullptr;
                    tensor_buft_overrides[itbo].buft    = nullptr;
                    itbo++;
                    mparams.tensor_buft_overrides = tensor_buft_overrides;
                    throw common_params_fit_exception("llama_max_tensor_buft_overrides() == "
                        + std::to_string(ntbo) + " is insufficient for model");
                }
                tensor_buft_overrides[itbo].pattern = get_overflow_pattern(il, il == il0 ? ngl_per_device[id].overflow_type : LAYER_FRACTION_MOE);
                tensor_buft_overrides[itbo].buft = il == il0 ? overflow_bufts[id] : ggml_backend_cpu_buffer_type();
                itbo++;
            }
            il0 += ngl_per_device[id].n_part;
        }
        tensor_buft_overrides[itbo].pattern = nullptr;
        tensor_buft_overrides[itbo].buft    = nullptr;
        itbo++;
        mparams.tensor_buft_overrides = tensor_buft_overrides;
    };

    // utility function that returns the memory use per device for given numbers of layers per device
    auto get_memory_for_layers = [&](
            const char * func_name,
            const std::vector<ngl_t> & ngl_per_device,
            const std::vector<ggml_backend_buffer_type_t> & overflow_bufts) -> std::vector<int64_t> {
        llama_model_params mparams_copy = *mparams;
        set_ngl_tensor_split_tbo(ngl_per_device, overflow_bufts, mparams_copy);

        dmds_t dmd_nl = common_get_device_memory_data_impl(
            path_model, &mparams_copy, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);
        add_extra_memory(dmd_nl);

        LOG_TRC("%s: memory for test allocation by device:\n", func_name);
        for (size_t id = 0; id < nd; id++) {
            const ngl_t & n = ngl_per_device[id];
            LOG_TRC(
                "%s: id=%zu, n_layer=%2" PRIu32 ", n_part=%2" PRIu32 ", overflow_type=%d, mem=%6" PRId64 " MiB\n",
                func_name, id, n.n_layer, n.n_part, int(n.overflow_type), dmd_nl[id].mb.total()/MiB);
        }

        std::vector<int64_t> ret;
        ret.reserve(nd);
        for (size_t id = 0; id < nd; id++) {
            ret.push_back(dmd_nl[id].mb.total());
        }
        return ret;
    };

    int64_t global_surplus_cpu_moe = 0;
    if (hp_nex > 0) {
        const static std::string pattern_moe_all = "blk\\.\\d+\\.ffn_(up|down|gate_up|gate)_(ch|)exps"; // matches all MoE tensors
        ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
        tensor_buft_overrides[0] = {pattern_moe_all.c_str(), cpu_buft};
        tensor_buft_overrides[1] = {nullptr, nullptr};
        mparams->tensor_buft_overrides = tensor_buft_overrides;

        LOG_TRC("%s: getting device memory data with all MoE tensors moved to system memory:\n", __func__);
        dmds_t dmds_cpu_moe = common_get_device_memory_data_impl(
            path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);
        add_extra_memory(dmds_cpu_moe);

        for (size_t id = 0; id < nd; id++) {
            global_surplus_cpu_moe += dmds_cpu_moe[id].free;
            global_surplus_cpu_moe -= int64_t(dmds_cpu_moe[id].mb.total()) + margins[id];
        }

        if (global_surplus_cpu_moe > 0) {
            LOG_TRC("%s: with only dense weights in device memory there is a total surplus of %" PRId64 " MiB\n",
                __func__, global_surplus_cpu_moe/MiB);
        } else {
            LOG_TRC("%s: with only dense weights in device memory there is still a total deficit of %" PRId64 " MiB\n",
                __func__, -global_surplus_cpu_moe/MiB);
        }

        // reset
        tensor_buft_overrides[0] = {nullptr, nullptr};
        mparams->tensor_buft_overrides = tensor_buft_overrides;
    }

    std::vector<int64_t> targets; // maximum acceptable memory use per device
    targets.reserve(nd);
    for (size_t id = 0; id < nd; id++) {
        targets.push_back(dmds_full[id].free - margins[id]);
        LOG_TRC("%s: id=%zu, target=%" PRId64 " MiB\n", __func__, id, targets[id]/MiB);
    }

    std::vector<ggml_backend_buffer_type_t> overflow_bufts; // which bufts the first partial layer of a device overflows to:
    overflow_bufts.reserve(nd);
    for (size_t id = 0; id < nd; id++) {
        overflow_bufts.push_back(ggml_backend_cpu_buffer_type());
    }

    std::vector<ngl_t> ngl_per_device(nd);
    std::vector<int64_t> mem = get_memory_for_layers(__func__, ngl_per_device, overflow_bufts);

    // optimize the number of layers per device using the method of false position:
    //   - ngl_per_device has 0 layers for each device, lower bound
    //   - try a "high" configuration where a device is given all unassigned layers
    //   - interpolate the memory use / layer between low and high linearly to get a guess where it meets our target
    //   - check memory use of our guess, replace either the low or high bound
    //   - once we only have a difference of a single layer, stop and return the lower bound that just barely still fits
    //   - the last device has the output layer, which cannot be a partial layer
    if (hp_nex == 0) {
        LOG_TRC("%s: filling dense layers back-to-front:\n", __func__);
    } else {
        LOG_TRC("%s: filling dense-only layers back-to-front:\n", __func__);
    }
    for (int id = nd - 1; id >= 0; id--) {
        uint32_t n_unassigned = hp_ngl + 1;
        for (size_t jd = id + 1; jd < nd; ++jd) {
            assert(n_unassigned >= ngl_per_device[jd].n_layer);
            n_unassigned -= ngl_per_device[jd].n_layer;
        }

        std::vector<ngl_t> ngl_per_device_high = ngl_per_device;
        ngl_per_device_high[id].n_layer = n_unassigned;
        if (hp_nex > 0) {
            ngl_per_device_high[id].n_part = size_t(id) < nd - 1 ? ngl_per_device_high[id].n_layer : ngl_per_device_high[id].n_layer - 1;
        }
        if (ngl_per_device_high[id].n_layer > 0) {
            std::vector<int64_t> mem_high = get_memory_for_layers(__func__, ngl_per_device_high, overflow_bufts);
            if (mem_high[id] > targets[id]) {
                assert(ngl_per_device_high[id].n_layer > ngl_per_device[id].n_layer);
                uint32_t delta = ngl_per_device_high[id].n_layer - ngl_per_device[id].n_layer;
                LOG_TRC("%s: start filling device %" PRIu32 ", delta=%" PRIu32 "\n", __func__, id, delta);
                while (delta > 1) {
                    uint32_t step_size = int64_t(delta) * (targets[id] - mem[id]) / (mem_high[id] - mem[id]);
                    step_size = std::max(step_size, uint32_t(1));
                    step_size = std::min(step_size, delta - 1);

                    std::vector<ngl_t> ngl_per_device_test = ngl_per_device;
                    ngl_per_device_test[id].n_layer += step_size;
                    if (hp_nex) {
                        ngl_per_device_test[id].n_part += size_t(id) == nd - 1 && ngl_per_device_test[id].n_part == 0 ?
                            step_size - 1 : step_size; // the first layer is the output layer which must always be full
                    }
                    const std::vector<int64_t> mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts);

                    if (mem_test[id] <= targets[id]) {
                        ngl_per_device = ngl_per_device_test;
                        mem            = mem_test;
                        LOG_TRC("%s: set ngl_per_device[%d].n_layer=%" PRIu32 "\n", __func__, id, ngl_per_device[id].n_layer);
                    } else {
                        ngl_per_device_high = ngl_per_device_test;
                        mem_high            = mem_test;
                        LOG_TRC("%s: set ngl_per_device_high[%d].n_layer=%" PRIu32 "\n", __func__, id, ngl_per_device_high[id].n_layer);
                    }
                    delta = ngl_per_device_high[id].n_layer - ngl_per_device[id].n_layer;
                }
            } else {
                assert(ngl_per_device_high[id].n_layer == n_unassigned);
                ngl_per_device = ngl_per_device_high;
                mem            = mem_high;
                LOG_TRC("%s: set ngl_per_device[%d].n_layer=%" PRIu32 "\n", __func__, id, ngl_per_device[id].n_layer);
            }
        }

        const int64_t projected_margin = dmds_full[id].free - mem[id];
        LOG_TRC(
            "%s:   - %s: %2" PRIu32 " layers, %6" PRId64 " MiB used, %6" PRId64 " MiB free\n",
            __func__, dev_names[id].c_str(), ngl_per_device[id].n_layer, mem[id]/MiB, projected_margin/MiB);
    }
    if (hp_nex == 0 || global_surplus_cpu_moe <= 0) {
        set_ngl_tensor_split_tbo(ngl_per_device, overflow_bufts, *mparams);
        return;
    }

    // step 4: for a MoE model where all dense tensors fit,
    //     convert the dense-only layers in the back to full layers in the front until all devices are full
    // essentially the same procedure as for the dense-only layers except front-to-back
    // also, try fitting at least part of one more layer to reduce waste for "small" GPUs with e.g. 24 GiB VRAM

    size_t id_dense_start = nd;
    for (int id = nd - 1; id >= 0; id--) {
        if (ngl_per_device[id].n_layer > 0) {
            id_dense_start = id;
            continue;
        }
        break;
    }
    assert(id_dense_start < nd);

    LOG_TRC("%s: converting dense-only layers to full layers and filling them front-to-back with overflow to next device/system memory:\n", __func__);
    for (size_t id = 0; id <= id_dense_start && id_dense_start < nd; id++) {
        std::vector<ngl_t> ngl_per_device_high = ngl_per_device;
        for (size_t jd = id_dense_start; jd < nd; jd++) {
            const uint32_t n_layer_move = jd < nd - 1 ? ngl_per_device_high[jd].n_layer : ngl_per_device_high[jd].n_layer - 1;
            ngl_per_device_high[id].n_layer += n_layer_move;
            ngl_per_device_high[jd].n_layer -= n_layer_move;
            ngl_per_device_high[jd].n_part = 0;
        }
        size_t id_dense_start_high = nd - 1;
        std::vector<int64_t> mem_high = get_memory_for_layers(__func__, ngl_per_device_high, overflow_bufts);

        if (mem_high[id] > targets[id]) {
            assert(ngl_per_device_high[id].n_full() >= ngl_per_device[id].n_full());
            uint32_t delta = ngl_per_device_high[id].n_full() - ngl_per_device[id].n_full();
            while (delta > 1) {
                uint32_t step_size = int64_t(delta) * (targets[id] - mem[id]) / (mem_high[id] - mem[id]);
                step_size = std::max(step_size, uint32_t(1));
                step_size = std::min(step_size, delta - 1);

                std::vector<ngl_t> ngl_per_device_test = ngl_per_device;
                size_t id_dense_start_test = id_dense_start;
                uint32_t n_converted_test = 0;
                for (;id_dense_start_test < nd; id_dense_start_test++) {
                    const uint32_t n_convert_jd = std::min(step_size - n_converted_test, ngl_per_device_test[id_dense_start_test].n_part);
                    ngl_per_device_test[id_dense_start_test].n_layer -= n_convert_jd;
                    ngl_per_device_test[id_dense_start_test].n_part -= n_convert_jd;
                    ngl_per_device_test[id].n_layer += n_convert_jd;
                    n_converted_test += n_convert_jd;

                    if (ngl_per_device_test[id_dense_start_test].n_part > 0) {
                        break;
                    }
                }
                const std::vector<int64_t> mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts);

                if (mem_test[id] <= targets[id]) {
                    ngl_per_device = ngl_per_device_test;
                    mem            = mem_test;
                    id_dense_start = id_dense_start_test;
                    LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part)=(%" PRIu32 ", %" PRIu32 "), id_dense_start=%zu\n",
                        __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
                } else {
                    ngl_per_device_high = ngl_per_device_test;
                    mem_high            = mem_test;
                    id_dense_start_high = id_dense_start_test;
                    LOG_TRC("%s: set ngl_per_device_high[%zu].(n_layer, n_part)=(%" PRIu32 ", %" PRIu32 "), id_dense_start_high=%zu\n",
                        __func__, id, ngl_per_device_high[id].n_layer, ngl_per_device_high[id].n_part, id_dense_start_high);
                }
                assert(ngl_per_device_high[id].n_full() >= ngl_per_device[id].n_full());
                delta = ngl_per_device_high[id].n_full() - ngl_per_device[id].n_full();
            }
        } else {
            ngl_per_device = ngl_per_device_high;
            mem            = mem_high;
            id_dense_start = id_dense_start_high;
            LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part)=(%" PRIu32 ", %" PRIu32 "), id_dense_start=%zu\n",
                __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
        }

        // try to fit at least part of one more layer
        if (ngl_per_device[id_dense_start].n_layer > (id < nd - 1 ? 0 : 1)) {
            std::vector<ngl_t> ngl_per_device_test = ngl_per_device;
            size_t id_dense_start_test = id_dense_start;
            ngl_per_device_test[id_dense_start_test].n_layer--;
            ngl_per_device_test[id_dense_start_test].n_part--;
            ngl_per_device_test[id].n_layer++;
            ngl_per_device_test[id].n_part++;
            if (ngl_per_device_test[id_dense_start_test].n_part == 0) {
                id_dense_start_test++;
            }
            ngl_per_device_test[id].overflow_type = LAYER_FRACTION_UP;
            std::vector<ggml_backend_buffer_type_t> overflow_bufts_test = overflow_bufts;
            if (id < nd - 1) {
                overflow_bufts_test[id] = ggml_backend_dev_buffer_type(devs[id + 1]);
            }
            LOG_TRC("%s: trying to fit one extra layer with overflow_type=LAYER_FRACTION_UP\n", __func__);
            std::vector<int64_t> mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts_test);
            if (mem_test[id] < targets[id] && (id + 1 == nd || mem_test[id + 1] < targets[id + 1])) {
                ngl_per_device = ngl_per_device_test;
                overflow_bufts = overflow_bufts_test;
                mem            = mem_test;
                id_dense_start = id_dense_start_test;
                LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part, overflow_type)=(%" PRIu32 ", %" PRIu32 ", UP), id_dense_start=%zu\n",
                    __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);

                ngl_per_device_test[id].overflow_type = LAYER_FRACTION_GATE;
                LOG_TRC("%s: trying to fit one extra layer with overflow_type=LAYER_FRACTION_GATE\n", __func__);
                mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts_test);
                if (mem_test[id] < targets[id] && (id + 1 == nd || mem_test[id + 1] < targets[id + 1])) {
                    ngl_per_device = ngl_per_device_test;
                    overflow_bufts = overflow_bufts_test;
                    mem            = mem_test;
                    id_dense_start = id_dense_start_test;
                    LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part, overflow_type)=(%" PRIu32 ", %" PRIu32 ", GATE), id_dense_start=%zu\n",
                        __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
                }
            } else {
                ngl_per_device_test[id].overflow_type = LAYER_FRACTION_ATTN;
                LOG_TRC("%s: trying to fit one extra layer with overflow_type=LAYER_FRACTION_ATTN\n", __func__);
                mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts_test);
                if (mem_test[id] < targets[id] && (id + 1 == nd || mem_test[id + 1] < targets[id + 1])) {
                    ngl_per_device = ngl_per_device_test;
                    overflow_bufts = overflow_bufts_test;
                    mem            = mem_test;
                    id_dense_start = id_dense_start_test;
                    LOG_TRC("%s: set ngl_per_device[%zu].(n_layer, n_part, overflow_type)=(%" PRIu32 ", %" PRIu32 ", ATTN), id_dense_start=%zu\n",
                        __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
                }
            }
        }

        const int64_t projected_margin = dmds_full[id].free - mem[id];
        LOG_TRC(
            "%s:   - %s: %2" PRIu32 " layers (%2" PRIu32 " overflowing), %6" PRId64 " MiB used, %6" PRId64 " MiB free\n",
            __func__, dev_names[id].c_str(), ngl_per_device[id].n_layer, ngl_per_device[id].n_part, mem[id]/MiB, projected_margin/MiB);
    }

    // print info for devices that were not changed during the conversion from dense only to full layers:
    for (size_t id = id_dense_start + 1; id < nd; id++) {
        const int64_t projected_margin = dmds_full[id].free - mem[id];
        LOG_TRC(
            "%s:   - %s: %2" PRIu32 " layers (%2" PRIu32 " overflowing), %6" PRId64 " MiB used, %6" PRId64 " MiB free\n",
            __func__, dev_names[id].c_str(), ngl_per_device[id].n_layer, ngl_per_device[id].n_part, mem[id]/MiB, projected_margin/MiB);
    }

    set_ngl_tensor_split_tbo(ngl_per_device, overflow_bufts, *mparams);
}

enum common_params_fit_status common_fit_params(
        const char * path_model,
        llama_model_params * mparams,
        llama_context_params * cparams,
        float * tensor_split,
        llama_model_tensor_buft_override * tensor_buft_overrides,
        size_t * margins,
        uint32_t n_ctx_min,
        const common_fit_extra_model * extra,
        ggml_log_level log_level) {
    const int64_t t0_us = llama_time_us();
    common_params_fit_status status = COMMON_PARAMS_FIT_STATUS_SUCCESS;
    try {
        common_params_fit_impl(path_model, mparams, cparams, tensor_split, tensor_buft_overrides, margins, n_ctx_min, extra, log_level);
        LOG_TRC("%s: successfully fit params to free device memory\n", __func__);
    } catch (const common_params_fit_exception & e) {
        LOG_WRN("%s: failed to fit params to free device memory: %s\n", __func__, e.what());
        status = COMMON_PARAMS_FIT_STATUS_FAILURE;
    } catch (const std::runtime_error & e) {
        LOG_ERR("%s: encountered an error while trying to fit params to free device memory: %s\n", __func__, e.what());
        status = COMMON_PARAMS_FIT_STATUS_ERROR;
    }
    const int64_t t1_us = llama_time_us();
    LOG_TRC("%s: fitting params to free memory took %.2f seconds\n", __func__, (t1_us - t0_us) * 1e-6);
    return status;
}

void common_memory_breakdown_print(const struct llama_context * ctx) {
    //const auto & devices = ctx->get_model().devices;
    const auto * model = llama_get_model(ctx);

    std::vector<ggml_backend_dev_t> devices;
    for (int i = 0; i < llama_model_n_devices(model); i++) {
        devices.push_back(llama_model_get_device(model, i));
    }

    llama_memory_breakdown memory_breakdown = llama_get_memory_breakdown(ctx);

    std::vector<std::array<std::string, 9>> table_data;
    table_data.reserve(devices.size());

    // same data as the table below, for --log-jsonl consumers
    common_json rows = common_json::array();
    const std::string template_header = "%s: | %s | %s   %s    %s   %s   %s   %s    %s |\n";
    const std::string template_gpu    = "%s: | %s | %s = %s + (%s = %s + %s + %s) + %s |\n";
    const std::string template_other  = "%s: | %s | %s   %s    %s = %s + %s + %s    %s |\n";

    table_data.push_back({template_header, "memory breakdown [MiB]", "total", "free", "self", "model", "context", "compute", "unaccounted"});

    constexpr size_t MiB = 1024 * 1024;
    const std::vector<std::string> desc_prefixes_strip = {"NVIDIA ", "GeForce ", "Tesla ", "AMD ", "Radeon ", "Instinct "};

    // track seen buffer types to avoid double counting:
    std::set<ggml_backend_buffer_type_t> seen_buffer_types;

    // accumulative memory breakdown for each device and for host:
    std::vector<llama_memory_breakdown_data> mb_dev(devices.size());
    llama_memory_breakdown_data              mb_host;

    for (const auto & buft_mb : memory_breakdown) {
        ggml_backend_buffer_type_t          buft = buft_mb.first;
        const llama_memory_breakdown_data & mb   = buft_mb.second;
        if (ggml_backend_buft_is_host(buft)) {
            mb_host.model   += mb.model;
            mb_host.context += mb.context;
            mb_host.compute += mb.compute;
            seen_buffer_types.insert(buft);
            continue;
        }
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (dev) {
            int i_dev = -1;
            for (size_t i = 0; i < devices.size(); i++) {
                if (devices[i] == dev) {
                    i_dev = i;
                    break;
                }
            }
            if (i_dev != -1) {
                mb_dev[i_dev].model   += mb.model;
                mb_dev[i_dev].context += mb.context;
                mb_dev[i_dev].compute += mb.compute;
                seen_buffer_types.insert(buft);
                continue;
            }
        }
    }

    // print memory breakdown for each device:
    for (size_t i = 0; i < devices.size(); i++) {
        ggml_backend_dev_t dev = devices[i];
        llama_memory_breakdown_data mb = mb_dev[i];

        const std::string name = ggml_backend_dev_name(dev);
        std::string desc = ggml_backend_dev_description(dev);
        for (const std::string & prefix : desc_prefixes_strip) {
            if (desc.length() >= prefix.length() && desc.substr(0, prefix.length()) == prefix) {
                desc = desc.substr(prefix.length());
            }
        }

        size_t free, total;
        ggml_backend_dev_memory(dev, &free, &total);

        const size_t self = mb.model + mb.context + mb.compute;
        const int64_t unaccounted = static_cast<int64_t>(total) - static_cast<int64_t>(free) - static_cast<int64_t>(self);

        table_data.push_back({
            template_gpu,
            "  - " + name + " (" + desc + ")",
            std::to_string(total / MiB),
            std::to_string(free / MiB),
            std::to_string(self / MiB),
            std::to_string(mb.model / MiB),
            std::to_string(mb.context / MiB),
            std::to_string(mb.compute / MiB),
            std::to_string(unaccounted / static_cast<int64_t>(MiB))});

        rows.push_back({
            {"kind",        "device"},
            {"name",        name},
            {"description", desc},
            {"total",       total / MiB},
            {"free",        free / MiB},
            {"self",        self / MiB},
            {"model",       mb.model / MiB},
            {"context",     mb.context / MiB},
            {"compute",     mb.compute / MiB},
            {"unaccounted", unaccounted / static_cast<int64_t>(MiB)},
        });
    }

    // print memory breakdown for host:
    {
        const size_t self = mb_host.model + mb_host.context + mb_host.compute;
        table_data.push_back({
            template_other,
            "  - Host",
            "", // total
            "", // free
            std::to_string(self / MiB),
            std::to_string(mb_host.model / MiB),
            std::to_string(mb_host.context / MiB),
            std::to_string(mb_host.compute / MiB),
            ""}); // unaccounted

        rows.push_back({
            {"kind",    "host"},
            {"name",    "Host"},
            {"self",    self / MiB},
            {"model",   mb_host.model / MiB},
            {"context", mb_host.context / MiB},
            {"compute", mb_host.compute / MiB},
        });
    }

    // print memory breakdown for all remaining buffer types:
    for (const auto & buft_mb : memory_breakdown) {
        ggml_backend_buffer_type_t          buft = buft_mb.first;
        const llama_memory_breakdown_data & mb   = buft_mb.second;
        if (seen_buffer_types.count(buft) == 1) {
            continue;
        }
        const std::string name = ggml_backend_buft_name(buft);
        const size_t self = mb.model + mb.context + mb.compute;
        table_data.push_back({
            template_other,
            "  - " + name,
            "", // total
            "", // free
            std::to_string(self / MiB),
            std::to_string(mb.model / MiB),
            std::to_string(mb.context / MiB),
            std::to_string(mb.compute / MiB),
            ""}); // unaccounted

        rows.push_back({
            {"kind",    "buffer_type"},
            {"name",    name},
            {"self",    self / MiB},
            {"model",   mb.model / MiB},
            {"context", mb.context / MiB},
            {"compute", mb.compute / MiB},
        });

        seen_buffer_types.insert(buft);
    }

    for (size_t j = 1; j < table_data[0].size(); j++) {
        size_t max_len = 0;
        for (const auto & td : table_data) {
            max_len = std::max(max_len, td[j].length());
        }
        for (auto & td : table_data) {
            td[j].insert(j == 1 ? td[j].length() : 0, max_len - td[j].length(), ' ');
        }
    }
    for (const auto & td : table_data) {
        LOG_TRC(td[0].c_str(),
            __func__, td[1].c_str(), td[2].c_str(), td[3].c_str(), td[4].c_str(), td[5].c_str(),
            td[6].c_str(), td[7].c_str(), td[8].c_str());
    }

    LOG_JSON("fit_memory_breakdown", common_json({
        {"unit", "MiB"},
        {"rows", rows},
    }));
}

void common_fit_print(
        const char * path_model,
        llama_model_params * mparams,
        llama_context_params * cparams) {
    std::vector<ggml_backend_dev_t> devs;
    uint32_t hp_ngl = 0; // hparams.n_gpu_layers
    uint32_t hp_nct = 0; // hparams.n_ctx_train
    uint32_t hp_nex = 0; // hparams.n_expert

    auto dmd = common_get_device_memory_data_impl(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, GGML_LOG_LEVEL_ERROR);
    GGML_ASSERT(dmd.size() == devs.size() + 1);

    for (size_t id = 0; id < devs.size(); id++) {
        printf("%s ",  ggml_backend_dev_name(devs[id]));
        printf("%zu ", dmd[id].mb.model/1024/1024);
        printf("%zu ", dmd[id].mb.context/1024/1024);
        printf("%zu ", dmd[id].mb.compute/1024/1024);
        printf("\n");
    }

    printf("Host ");
    printf("%zu ", dmd.back().mb.model/1024/1024);
    printf("%zu ", dmd.back().mb.context/1024/1024);
    printf("%zu ", dmd.back().mb.compute/1024/1024);
    printf("\n");
}
