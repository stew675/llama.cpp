#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"
#include "ggml-moe-weighted-reduction.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <memory_resource>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// getenv() is cheap on Linux but takes a lock and rescans the environment block on Windows.  These
// debug / A-B gates sit on per-graph, per-split and per-tensor paths, so resolve each one once per
// call site: the static inside the immediately-invoked lambda is unique to each macro expansion.
//
// Cache the *value*, not the getenv() pointer: the MSVC CRT returns a buffer that the next getenv()
// call overwrites, so a cached `const char *` can end up pointing at a different variable's value.
#define GGML_ENV_STR(name) ([]() -> const char * { \
    static const std::string v = []() { const char * e = getenv(name); return e != nullptr ? std::string(e) : std::string(); }(); \
    static const bool set = getenv(name) != nullptr; \
    return set ? v.c_str() : nullptr; \
}())

struct ggml_backend_meta_device;
struct ggml_backend_meta_buffer_type;
struct ggml_backend_meta_buffer;
struct ggml_backend_meta;

const char * ggml_backend_meta_split_axis_name(enum ggml_backend_meta_split_axis split_axis) {
    switch (split_axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
            return "0";
        case GGML_BACKEND_SPLIT_AXIS_1:
            return "1";
        case GGML_BACKEND_SPLIT_AXIS_2:
            return "2";
        case GGML_BACKEND_SPLIT_AXIS_3:
            return "3";
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            return "MIRRORED";
        case GGML_BACKEND_SPLIT_AXIS_PARTIAL:
            return "PARTIAL";
        case GGML_BACKEND_SPLIT_AXIS_PER_DEVICE:
            return "PER_DEVICE";
        case GGML_BACKEND_SPLIT_AXIS_NONE:
            return "NONE";
        case GGML_BACKEND_SPLIT_AXIS_UNKNOWN:
            return "UNKNOWN";
        default:
            GGML_ABORT("fatal error");
    }
}

//
// meta backend device
//

// Upper bound on the "rows" (elements of every dim above the split axis) an ALL_GATHER collective copies
// per (source, destination) pair.  The only producer is the DFlash2 selector's draft batch, whose row count
// is the draft block size times the number of sequences: 16 sequences of an 8-token block is far above any
// supported --parallel.  A larger batch trips the bound assert instead of overrunning the view pool.
#define GGML_META_GATHER_MAX_ROWS 128

struct ggml_backend_meta_device_context {
    std::vector<ggml_backend_dev_t>     simple_devs;
    ggml_backend_meta_get_split_state_t get_split_state;
    void *                              get_split_state_ud;

    std::string name;
    std::string description;

    ggml_backend_meta_device_context(
            std::vector<ggml_backend_dev_t> simple_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud) :
            simple_devs(std::move(simple_devs)), get_split_state(get_split_state), get_split_state_ud(get_split_state_ud) {
        name        = std::string("Meta(");
        description = std::string("Meta(");
        for (size_t i = 0; i < simple_devs.size(); i++) {
            if (i > 0) {
                name        += ",";
                description += ",";
            }
            name        += ggml_backend_dev_name       (simple_devs[i]);
            description += ggml_backend_dev_description(simple_devs[i]);
        }
        name        += ")";
        description += ")";
    }

    bool operator<(const ggml_backend_meta_device_context & other) const {
        return std::tie(simple_devs, get_split_state, get_split_state_ud)
            < std::tie(other.simple_devs, other.get_split_state, other.get_split_state_ud);
    }
};

bool ggml_backend_dev_is_meta(ggml_backend_dev_t dev);

static const char * ggml_backend_meta_device_get_name(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return meta_dev_ctx->name.c_str();
}

static const char * ggml_backend_meta_device_get_description(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return meta_dev_ctx->description.c_str();
}

static void ggml_backend_meta_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    *free  = 0;
    *total = 0;
    for (ggml_backend_dev_t dev : meta_dev_ctx->simple_devs) {
        size_t tmp_free, tmp_total;
        ggml_backend_dev_memory(dev, &tmp_free, &tmp_total);
        *free  += tmp_free;
        *total += tmp_total;
    }
}

static enum ggml_backend_dev_type ggml_backend_meta_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_META;

    GGML_UNUSED(dev);
}

static void ggml_backend_meta_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;

    // TODO replace placeholders
    props->name        = ggml_backend_meta_device_get_name(dev);
    props->description = ggml_backend_meta_device_get_description(dev);
    props->type        = ggml_backend_meta_device_get_type(dev);
    props->device_id   = 0;

    ggml_backend_meta_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ true,  // proxied via simple devices in get_host_buffer_type
        /* .buffer_from_host_ptr  = */ false, // Not implemented.
        /* .events                = */ true,  // proxied via simple devices (fan-out)
        /* .mmap_support          = */ true,
    };
    for (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_dev_props tmp_props;
        ggml_backend_dev_get_props(simple_dev, &tmp_props);
        props->caps.async                = props->caps.async                && tmp_props.caps.async;
        props->caps.host_buffer          = props->caps.host_buffer          && tmp_props.caps.host_buffer;
        props->caps.buffer_from_host_ptr = props->caps.buffer_from_host_ptr && tmp_props.caps.buffer_from_host_ptr;
        props->caps.events               = props->caps.events               && tmp_props.caps.events;
        props->caps.mmap_support         = props->caps.mmap_support         && tmp_props.caps.mmap_support;
    }
}

// UPSTREAM PR #22466 (adapted): the meta device proxies events to its simple devices so the loader's
// chunked async-upload pipeline (`upload_backend`) engages under `-sm tensor`.  Without this the loader
// falls back to a pageable `read_buf` and issues a raw async H2D from anonymous memory, which the SDMA
// engine can read directly and fault on (ROCm 7.14, gfx1201) -- the `--load-mode none` startup fault.
struct ggml_backend_meta_event_context {
    std::vector<ggml_backend_event_t> simple_events;
};

static ggml_backend_event_t ggml_backend_meta_device_event_new(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    auto * ev_ctx = new ggml_backend_meta_event_context();
    ev_ctx->simple_events.reserve(meta_dev_ctx->simple_devs.size());
    for (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_event_t simple_ev = ggml_backend_event_new(simple_dev);
        if (simple_ev == nullptr) {
            for (ggml_backend_event_t e : ev_ctx->simple_events) {
                ggml_backend_event_free(e);
            }
            delete ev_ctx;
            return nullptr;
        }
        ev_ctx->simple_events.push_back(simple_ev);
    }
    return new ggml_backend_event{dev, ev_ctx};
}

static void ggml_backend_meta_device_event_free(ggml_backend_dev_t /*dev*/, ggml_backend_event_t event) {
    auto * ev_ctx = (ggml_backend_meta_event_context *) event->context;
    for (ggml_backend_event_t e : ev_ctx->simple_events) {
        ggml_backend_event_free(e);
    }
    delete ev_ctx;
    delete event;
}

static void ggml_backend_meta_device_event_synchronize(ggml_backend_dev_t /*dev*/, ggml_backend_event_t event) {
    auto * ev_ctx = (ggml_backend_meta_event_context *) event->context;
    for (ggml_backend_event_t e : ev_ctx->simple_events) {
        ggml_backend_event_synchronize(e);
    }
}

static ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, const char * params);

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev);

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_host_buffer_type(ggml_backend_dev_t dev);

static bool ggml_backend_meta_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    // ALL_GATHER is the meta backend's own collective (cross-device copies between the simple backends);
    // no simple backend has to implement it for the meta device to run it.
    if (op->op == GGML_OP_ALL_GATHER) {
        return true;
    }
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return std::all_of(meta_dev_ctx->simple_devs.begin(), meta_dev_ctx->simple_devs.end(),
        [op](ggml_backend_dev_t simple_dev) { return ggml_backend_dev_supports_op(simple_dev, op); });
}

// Op offload (host-resident weights executed on the device) for the tensor split.  The meta backend
// is a first-class host of such ops: it runs the op on every device, and each device uploads its own
// share of the weights -- which is exactly what the scheduler's used-expert pruning and the H2D
// staging ring act on.  Every device has to accept the op on its own, since the meta backend cannot
// run a node that only some of its devices can.  Without this hook `ggml_backend_dev_offload_op()`
// is false here (the device iface defaults to no offload), so `ggml_backend_sched_backend_id_from_cur`
// never places an offloaded node on this backend and `-sm tensor` silently executes the whole MoE on
// the CPU.
static bool ggml_backend_meta_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    if (meta_dev_ctx->simple_devs.empty()) {
        return false;
    }
    // [TAG_META_OFFLOAD_MOE_ONLY]
    // Only the host-resident *MoE expert* op, which is what this hook exists for: its weights are the
    // ones the loader keeps in the host buffer (-ncmoe/-cmoe), and the split state of a MUL_MAT_ID is
    // defined for a host weight.
    //
    // A *dense* op that has a host weight is a different situation: a CPU-offloaded layer (`-ngl` / a
    // --fit that reduced n_gpu_layers) leaves the whole layer on the host, including its *state*
    // (KV cache, GDN conv/ssm state).  The meta split-state model has no representation for a
    // host-resident input of a split op (a GDN op reaches `handle_gated_delta_net`'s src_ss[5] assert,
    // and a flash-attention op would fail the same way on a host KV), so claiming such an op aborts
    // the context at the first wide prefill.  Upstream only offloads host-weight ops onto a *single*
    // device (`-sm layer`), where the op and its state share one backend; the tensor split needs that
    // design work first (issue #170).
    if (op->op != GGML_OP_MUL_MAT_ID) {
        return false;
    }
    return std::all_of(meta_dev_ctx->simple_devs.begin(), meta_dev_ctx->simple_devs.end(),
        [op](ggml_backend_dev_t simple_dev) { return ggml_backend_dev_offload_op(simple_dev, op); });
}

static bool ggml_backend_meta_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ggml_backend_dev_t dev_buft = ggml_backend_buft_get_device(buft);
    if (!ggml_backend_dev_is_meta(dev_buft)) {
        return false;
    }
    const ggml_backend_meta_device_context * meta_dev_ctx      = (const ggml_backend_meta_device_context *) dev->context;
    const ggml_backend_meta_device_context * meta_buft_dev_ctx = (const ggml_backend_meta_device_context *) dev_buft->context;
    if (meta_dev_ctx->simple_devs.size() != meta_buft_dev_ctx->simple_devs.size()) {
        return false;
    }
    for (size_t i = 0; i < meta_dev_ctx->simple_devs.size(); i++) {
        if (meta_dev_ctx->simple_devs[i] != meta_buft_dev_ctx->simple_devs[i]) {
            return false;
        }
    }
    return true;
}

static void ggml_backend_meta_device_slab_narrow2_floor(ggml_backend_dev_t dev, size_t bytes) {
    // forward to the simple devices: the meta device owns no slab of its own
    ggml_backend_meta_device_context * ctx = (ggml_backend_meta_device_context *) dev->context;
    for (ggml_backend_dev_t d : ctx->simple_devs) {
        ggml_backend_dev_slab_narrow2_floor(d, bytes);
    }
}

static void ggml_backend_meta_device_slab_compute_narrow2(ggml_backend_dev_t dev, bool enable) {
    // forward to the simple devices: the meta device owns no slab of its own
    ggml_backend_meta_device_context * ctx = (ggml_backend_meta_device_context *) dev->context;
    for (ggml_backend_dev_t d : ctx->simple_devs) {
        ggml_backend_dev_slab_compute_narrow2(d, enable);
    }
}

static void ggml_backend_meta_device_slab_compute_plain(ggml_backend_dev_t dev, bool enable) {
    // forward to the simple devices: the meta device owns no slab of its own
    ggml_backend_meta_device_context * ctx = (ggml_backend_meta_device_context *) dev->context;
    for (ggml_backend_dev_t d : ctx->simple_devs) {
        ggml_backend_dev_slab_compute_plain(d, enable);
    }
}

static const ggml_backend_device_i ggml_backend_meta_device_iface = {
    /* .get_name             = */ ggml_backend_meta_device_get_name,
    /* .get_description      = */ ggml_backend_meta_device_get_description,
    /* .get_memory           = */ ggml_backend_meta_device_get_memory,
    /* .get_type             = */ ggml_backend_meta_device_get_type,
    /* .get_props            = */ ggml_backend_meta_device_get_props,
    /* .init_backend         = */ ggml_backend_meta_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_meta_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_meta_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_meta_device_supports_op,
    /* .supports_buft        = */ ggml_backend_meta_device_supports_buft,
    /* .offload_op           = */ ggml_backend_meta_device_offload_op,
    /* .event_new            = */ ggml_backend_meta_device_event_new,
    /* .event_free           = */ ggml_backend_meta_device_event_free,
    /* .event_synchronize    = */ ggml_backend_meta_device_event_synchronize,
    /* .moe_cache_preflight  = */ nullptr,
    /* .moe_cache_set_reserve = */ nullptr,
    /* .moe_cache_stats      = */ nullptr,
    /* .moe_cache_rearm      = */ nullptr,
    /* .slab_work_size       = */ nullptr,
    /* .slab_headroom_bytes  = */ nullptr,
    /* .slab_ring_set        = */ nullptr,
    /* .slab_narrow_floor    = */ nullptr,
    /* .slab_reserve_wide    = */ nullptr,
    /* .slab_narrow2_floor   = */ ggml_backend_meta_device_slab_narrow2_floor,
    /* .slab_compute_narrow2 = */ ggml_backend_meta_device_slab_compute_narrow2,
    /* .moe_cache_set_host_pool   = */ nullptr,
    /* .moe_cache_set_host_source = */ nullptr,
    /* .slab_compute_plain   = */ ggml_backend_meta_device_slab_compute_plain,
    /* .slab_ring_region     = */ nullptr,
    /* .slab_set_ring_cap    = */ nullptr,
    /* .slab_enabled         = */ nullptr,
    /* .slab_ring_slots      = */ nullptr,
};

bool ggml_backend_dev_is_meta(ggml_backend_dev_t dev) {
    return dev != nullptr && dev->iface.get_name == ggml_backend_meta_device_iface.get_name;
}

size_t ggml_backend_meta_dev_n_devs(ggml_backend_dev_t meta_dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) meta_dev->context;
    return meta_dev_ctx->simple_devs.size();
}

ggml_backend_dev_t ggml_backend_meta_dev_simple_dev(ggml_backend_dev_t meta_dev, size_t index) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) meta_dev->context;
    GGML_ASSERT(index < meta_dev_ctx->simple_devs.size());
    return meta_dev_ctx->simple_devs[index];
}

// issue #164: the byte size of device `dev`'s slice of a tensor under the meta split.  THIS is the
// definition of the per-device split size.  `ggml_backend_meta_stage_input` sizes the H2D ring slot with
// it, and `llama_model::max_host_expert_slice_bytes` sizes the slab ring hole from the same rule, so the
// ring can always hold the largest upload the staging will ask for.  `extent` is the same per-segment sum
// `ggml_backend_meta_buffer_init_tensor_impl` uses to build each device's tensor (the split is
// granularity-rounded, so it need not be even), and the byte size scales linearly with it because the
// tensor is contiguous.
size_t ggml_backend_meta_split_dev_bytes(const struct ggml_tensor * tensor,
        const struct ggml_backend_meta_split_state * split_state, size_t n_devs, size_t dev) {
    if (tensor == nullptr || split_state == nullptr || dev >= n_devs) {
        return 0;
    }
    if (split_state->axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
        return ggml_nbytes(tensor);
    }
    if (split_state->axis < 0 || split_state->axis >= GGML_MAX_DIMS) {
        return 0;
    }
    const int64_t ne_axis = tensor->ne[split_state->axis];
    if (ne_axis <= 0) {
        return 0;
    }
    int64_t extent = 0;
    for (uint32_t s = 0; s < split_state->n_segments; s++) {
        extent += split_state->ne[s*n_devs + dev] * split_state->nr[s];
    }
    if (extent <= 0) {
        return 0;
    }
    const size_t nbytes = ggml_nbytes(tensor);
    if (nbytes % (size_t) ne_axis != 0) {
        return nbytes;   // non-contiguous tensor: the whole tensor is the safe bound
    }
    return (nbytes / (size_t) ne_axis) * (size_t) extent;
}

ggml_backend_dev_t ggml_backend_meta_device(
        ggml_backend_dev_t * devs, size_t n_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud) {
    GGML_ASSERT(n_devs <= GGML_BACKEND_META_MAX_DEVICES);
    // TODO: this is not thread-safe - needs to be fixed
    static std::vector<std::unique_ptr<ggml_backend_meta_device_context>>         ctxs;
    static std::map<ggml_backend_meta_device_context, struct ggml_backend_device> meta_devs;

    std::vector<ggml_backend_dev_t> simple_devs;
    simple_devs.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        simple_devs.push_back(devs[i]);
    }
    ggml_backend_meta_device_context ctx(simple_devs, get_split_state, get_split_state_ud);

    {
        auto it = meta_devs.find(ctx);
        if (it != meta_devs.end()) {
            return &it->second;
        }
    }
    ctxs.push_back(std::make_unique<ggml_backend_meta_device_context>(ctx));

    struct ggml_backend_device meta_dev = {
        /*iface  =*/ ggml_backend_meta_device_iface,
        /*reg    =*/ nullptr,
        /*ctx    =*/ ctxs.back().get(),
    };

    auto result = meta_devs.emplace(*ctxs.back(), meta_dev);
    return &result.first->second;
}

//
// meta backend buffer type
//

struct ggml_backend_meta_buffer_type_context {
    std::vector<ggml_backend_buffer_type_t> simple_bufts;

    std::string name;

    ggml_backend_meta_buffer_type_context(std::vector<ggml_backend_buffer_type_t> simple_bufts) : simple_bufts(std::move(simple_bufts)) {
        name = "Meta(";
        for (size_t i = 0; i < simple_bufts.size(); i++) {
            if (i > 0) {
                name += ",";
            }
            name += ggml_backend_buft_name(simple_bufts[i]);
        }
        name += ")";
    }

    bool operator<(const ggml_backend_meta_buffer_type_context & other) const {
        return simple_bufts < other.simple_bufts;
    }
};

static size_t ggml_backend_meta_buft_n_bufts(ggml_backend_buffer_type_t meta_buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) meta_buft->context;
    return meta_buft_ctx->simple_bufts.size();
}

static const char * ggml_backend_meta_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) buft->context;
    return meta_buft_ctx->name.c_str();
}

static ggml_backend_buffer_type_t ggml_backend_meta_buft_simple_buft(ggml_backend_buffer_type_t meta_buft, size_t index) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) meta_buft->context;
    GGML_ASSERT(index < meta_buft_ctx->simple_bufts.size());
    return meta_buft_ctx->simple_bufts[index];
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);
static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer_usage(ggml_backend_buffer_type_t buft, size_t size, enum ggml_backend_buffer_usage usage);

static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer_n(ggml_backend_buffer_type_t buft, ggml_tensor ** tensors, int n_tensors);

static size_t ggml_backend_meta_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alignment = 1;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        const size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_meta_buft_simple_buft(buft, i));
        max_alignment = std::max(max_alignment, alignment);
        GGML_ASSERT(max_alignment % alignment == 0);
    }
    return max_alignment;
}

static size_t ggml_backend_meta_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_size = SIZE_MAX;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        max_size = std::min(max_size, ggml_backend_buft_get_max_size(ggml_backend_meta_buft_simple_buft(buft, i)));
    }
    return max_size;
}

static size_t ggml_backend_meta_buffer_type_get_compute_margin_pct(ggml_backend_buffer_type_t buft) {
    // Conservative aggregation, matching get_max_size: the single meta allocation is split across the
    // constituents, so only ask for slack when every one of them does.  A constituent that does not opt
    // in (every non-AMD backend, and AMD builds without the opt-in) makes this 0 -- i.e. exactly the
    // allocation size this build produced before the feature existed.
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t margin_pct = SIZE_MAX;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        margin_pct = std::min(margin_pct, ggml_backend_buft_get_compute_margin_pct(ggml_backend_meta_buft_simple_buft(buft, i)));
    }
    return margin_pct == SIZE_MAX ? 0 : margin_pct;
}

static size_t ggml_backend_meta_buffer_type_get_compute_chunk_bytes(ggml_backend_buffer_type_t buft) {
    // Same conservative aggregation as the margin above: the one meta allocation is split across the
    // constituents, so use a chunk size only when every constituent agrees on one.
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t chunk = 0;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        const size_t c = ggml_backend_buft_get_compute_chunk_bytes(ggml_backend_meta_buft_simple_buft(buft, i));
        if (c == 0) {
            return 0;
        }
        chunk = chunk == 0 ? c : std::min(chunk, c);
    }
    return chunk;
}

static size_t ggml_backend_meta_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alloc_size = 0;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        const size_t alloc_size = ggml_backend_buft_get_alloc_size(ggml_backend_meta_buft_simple_buft(buft, i), tensor);
        max_alloc_size = std::max(max_alloc_size, alloc_size);
    }
    return max_alloc_size;
}

static bool ggml_backend_meta_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        if (!ggml_backend_buft_is_host(ggml_backend_meta_buft_simple_buft(buft, i))) {
            return false;
        }
    }
    return true;
}

static const struct ggml_backend_buffer_type_i ggml_backend_meta_buffer_type_iface = {
    /* .get_name            = */ ggml_backend_meta_buffer_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_meta_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ ggml_backend_meta_buffer_type_alloc_buffer_n,
    /* .get_alignment       = */ ggml_backend_meta_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_meta_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_meta_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ NULL,
    /* .is_host             = */ ggml_backend_meta_buffer_type_is_host,
    /* .get_compute_margin_pct = */ ggml_backend_meta_buffer_type_get_compute_margin_pct,
    /* .alloc_buffer_usage  = */ ggml_backend_meta_buffer_type_alloc_buffer_usage,
    /* .get_compute_chunk_bytes = */ ggml_backend_meta_buffer_type_get_compute_chunk_bytes,
};

bool ggml_backend_buft_is_meta(ggml_backend_buffer_type_t buft) {
    return buft != nullptr && buft->iface.get_name == ggml_backend_meta_buffer_type_iface.get_name;
}

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev) {
    static std::map<ggml_backend_dev_t, struct ggml_backend_buffer_type> meta_bufts;
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    {
        auto it = meta_bufts.find(dev);
        if (it != meta_bufts.end()) {
            return &it->second;
        }
    }

    const size_t n_devs = ggml_backend_meta_dev_n_devs(dev);
    std::vector<ggml_backend_buffer_type_t> simple_bufts;
    simple_bufts.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        simple_bufts.push_back(ggml_backend_dev_buffer_type(ggml_backend_meta_dev_simple_dev(dev, i)));
    }
    ggml_backend_meta_buffer_type_context * buft_ctx = new ggml_backend_meta_buffer_type_context(simple_bufts);

    struct ggml_backend_buffer_type meta_buft = {
        /*iface  =*/ ggml_backend_meta_buffer_type_iface,
        /*device =*/ dev,
        /*ctx    =*/ buft_ctx,
    };
    auto result = meta_bufts.emplace(dev, meta_buft);
    return &result.first->second;
}

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;

    ggml_backend_buffer_type_t host_buft = nullptr;
    for (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_buffer_type_t simple_host_buft = ggml_backend_dev_host_buffer_type(simple_dev);
        if (simple_host_buft == nullptr) {
            return nullptr;
        }
        if (host_buft == nullptr) {
            host_buft = simple_host_buft;
        } else if (host_buft != simple_host_buft) {
            // if different simple devices have different host buffer types,
            // we cannot provide a single host buffer type for the meta device
            return nullptr;
        }
    }
    return host_buft;
}

//
// meta backend buffer
//

// Container to hold the tensor slices per simple ggml backend buffer.
//
// One generation (everything registered between two graph rebuilds: a graph's own tensors and views, plus the
// views an out-of-graph operation creates) can register an unbounded number of simple tensors.  A state
// restore creates one view per KV cache chunk per layer (`llama_io_read_device`), a park move creates its
// views per chunk (`llama_kv_cache::park_move`), and every graph shape contributes its own.  A single
// fixed-size context per simple backend therefore has a cliff: the 128-views-per-static-tensor budget of the
// 48-tensor attention KV buffer (6144 views) was exhausted by a long-context checkpoint/prompt-cache restore
// plus the graph's own views, aborting in `ggml_new_object` with "not enough space in the context's memory
// pool" (issue #232, `needed 2261360, available 2260992`).  The context list is therefore growable: a full
// context is followed by a new one instead of aborting, and a rebuild resets (and releases) all of them.
struct ggml_backend_meta_simple_tensor_container {
    // [simple backend][context] -- more than one context per simple backend only once a generation registered
    // more tensors than one holds.  `used[j]` counts the tensors created in the last context of `j`.
    std::vector<std::vector<ggml_context_ptr>> ctxs;
    std::vector<size_t> used;
    ggml_init_params params = { /*.mem_size   =*/ 0, /*.mem_buffer =*/ nullptr, /*.no_alloc   =*/ true };
    size_t capacity = 1; // tensors that fit in one context

    std::unordered_map<const ggml_tensor *, std::vector<ggml_tensor *>> simple_tensors;

    ggml_backend_meta_simple_tensor_container(const ggml_init_params & params_l, const int n_simple) : params(params_l) {
        capacity = std::max<size_t>(1, params_l.mem_size/ggml_tensor_overhead());
        ctxs.resize(n_simple);
        used.assign(n_simple, 0);
        for (int i = 0; i < n_simple; i++) {
            ctxs[i].emplace_back(ggml_init(params));
            if (ctxs[i].back() == nullptr) {
                GGML_ABORT("%s: failed to create a ggml context for the simple tensors of simple backend %d\n", __func__, i);
            }
        }
    }
    ggml_backend_meta_simple_tensor_container() {}

    // the context of simple backend `j` the next simple tensor is created in, growing the list when the last
    // context is full
    ggml_context * ctx_for_new_tensor(size_t j) {
        GGML_ASSERT(j < ctxs.size() && !ctxs[j].empty());
        if (used[j] >= capacity) {
            ctxs[j].emplace_back(ggml_init(params));
            if (ctxs[j].back() == nullptr) {
                GGML_ABORT("%s: failed to grow the simple tensor container of simple backend %zu\n", __func__, j);
            }
            used[j] = 0;
            GGML_LOG_DEBUG("%s: simple backend %zu: tensor container grew to %zu contexts (%zu tensors each)\n",
                    __func__, j, ctxs[j].size(), capacity);
        }
        used[j]++;
        return ctxs[j].back().get();
    }

    // the first context of simple backend `j` (the one that holds the tensors allocated with the buffer)
    ggml_context * ctx_first(size_t j) const {
        GGML_ASSERT(j < ctxs.size() && !ctxs[j].empty() && ctxs[j][0] != nullptr);
        return ctxs[j][0].get();
    }

    // a rebuild: drop every tensor registered in this container and release the contexts grown past the first
    // one, so one large generation does not pin its peak for the rest of the run
    void reset() {
        for (size_t j = 0; j < ctxs.size(); j++) {
            for (ggml_context_ptr & ctx : ctxs[j]) {
                if (ctx != nullptr) {
                    ggml_reset(ctx.get());
                }
            }
            if (!ctxs[j].empty()) {
                ctxs[j].resize(1);
            }
            used[j] = 0;
        }
    }
};

// The external views of a buffer live in a rotating pair of containers that assumes ONE live graph: a rebuild
// clears the container the previous graph used.  A second live graph (a second scheduler of the same context)
// selects its own view slot, so the two graphs keep separate pairs and do not clear each other's views.
static int g_meta_view_slot = 0;

void ggml_backend_meta_set_view_slot(int slot) {
    GGML_ASSERT(slot >= 0 && slot < 2);
    g_meta_view_slot = slot;
}

// Views created *between* graph evals (KV parking, issue #232) go to a scratch container instead of the
// per-generation compute containers: they are not tied to a graph, so nothing would reclaim them until the next
// rebuild, and `llama_kv_cache::park_move` creates a fresh view pair per chunk of a moved run -- a nearly full
// unified cache moves a long run in chunks no larger than the gap, so one move can register hundreds of
// thousands of views.  The caller resets the scratch container once the chunk's copies are done.
static thread_local bool g_meta_scratch_views = false;

void ggml_backend_meta_set_scratch_views(bool on) {
    g_meta_scratch_views = on;
}

struct ggml_backend_meta_buffer_context {
    // FIXME
    // Most tensors can simply be stored statically in their own buffer.
    // Externally created views however also need a mapping to simple tensors but they use the buffer of the view source.
    // If external views are simply using that buffer they will slowly deplete its memory.
    // Current solution: rotating set of 2 "compute" containers to hold external views, works correctly for llama.cpp.
    // Long-term: tie the lifetime of external views to the meta backend executing the graph instead,
    //     currently not possible due to graph-external operations in the backend scheduler.
    // stc_compute[slot][2]: one rotating pair per view slot (see g_meta_view_slot); slot 1 is created on first use
    static constexpr int n_view_slots = 2;
    ggml_backend_meta_simple_tensor_container stc_static;
    ggml_backend_meta_simple_tensor_container stc_compute[n_view_slots][2];
    // views created between graph evals, reset by their creator once the copies that read them are done
    ggml_backend_meta_simple_tensor_container stc_scratch;
    int stc_compute_index     [n_view_slots] = {0};
    int stc_compute_index_next[n_view_slots] = {0};
    ggml_init_params stc_compute_params;
    std::vector<ggml_backend_buffer_ptr> bufs;

    // FIXME
    // The size of the split state cache is unbounded and can theoretically grow infinitely large.
    // However, it is also expensive to build and clearing it on every rebuild in ggml_backend_meta_graph_compute is too expensive.
    // Graphs of different shapes (e.g. speculative verify and draft graphs) reuse the same tensor addresses, so a single
    // cache was cleared on every alternation; keep a few versions and switch to the one the changed tensor matches.
    static constexpr size_t nbtc = GGML_TENSOR_SIZE - sizeof(ggml_tensor::padding);
    static constexpr int n_split_state_caches = 8;
    // A split state is ~2.2 KiB (ne has room for 16 segments x 16 devices); the cache keeps only the n_segments x n_bufs
    // entries in use, so that a lookup (~10k per decode step) touches a few cache lines instead of the whole struct.
    struct split_state_entry {
        ggml_backend_meta_split_axis axis;
        uint32_t n_segments;
        uint32_t nr[16];
        std::vector<int64_t> ne;
        char bytes[nbtc];   // the tensor as it was when the state was computed; a hit requires identical bytes

        void set(const ggml_backend_meta_split_state & ss, size_t n_bufs) {
            axis       = ss.axis;
            n_segments = ss.n_segments;
            memcpy(nr, ss.nr, sizeof(nr));
            ne.assign(ss.ne, ss.ne + std::max<size_t>(n_segments, 1)*n_bufs);
        }
        ggml_backend_meta_split_state get() const {
            ggml_backend_meta_split_state ss = {};
            ss.axis       = axis;
            ss.n_segments = n_segments;
            memcpy(ss.nr, nr, sizeof(nr));
            std::copy(ne.begin(), ne.end(), ss.ne);
            return ss;
        }
    };
    struct split_state_key_hash {
        size_t operator()(const std::pair<const ggml_tensor *, bool> & k) const {
            return std::hash<const void *>()(k.first) ^ (size_t) k.second;
        }
    };
    std::unordered_map<std::pair<const ggml_tensor *, bool>, split_state_entry, split_state_key_hash> split_state_cache[n_split_state_caches];
    int      split_state_cache_cur = 0;
    uint64_t split_state_cache_used[n_split_state_caches] = {0};
    uint64_t split_state_cache_tick = 0;

    ggml_backend_meta_buffer_context(
            ggml_backend_meta_simple_tensor_container & stc_static,
            ggml_backend_meta_simple_tensor_container & stc_compute_0,
            ggml_backend_meta_simple_tensor_container & stc_compute_1,
            const ggml_init_params & stc_compute_params,
            const std::vector<ggml_backend_buffer_t> & bufs)
            : stc_static(std::move(stc_static)), stc_compute_params(stc_compute_params) {
        stc_compute[0][0] = std::move(stc_compute_0);
        stc_compute[0][1] = std::move(stc_compute_1);
        this->bufs.reserve(bufs.size());
        for (ggml_backend_buffer_t buf : bufs) {
            this->bufs.emplace_back(buf);
        }
    }

    ggml_backend_meta_simple_tensor_container & get_simple_tensor_container(const ggml_tensor * tensor) {
        if (stc_static.simple_tensors.find(tensor) != stc_static.simple_tensors.end()) {
            return stc_static;
        }
        if (g_meta_scratch_views) {
            return scratch_container();
        }
        return compute_container(stc_compute_index[g_meta_view_slot]);
    }

    // created on first use from the compute params: one park chunk needs 2 views per (layer, K/V), well inside
    // that headroom -- and the container grows if a caller ever needs more (ggml_backend_meta_simple_tensor_container)
    ggml_backend_meta_simple_tensor_container & scratch_container() {
        if (stc_scratch.ctxs.empty()) {
            stc_scratch = ggml_backend_meta_simple_tensor_container(stc_compute_params, (int) bufs.size());
        }
        return stc_scratch;
    }

    // container `index` of the current view slot, created on the slot's first use
    ggml_backend_meta_simple_tensor_container & compute_container(int index) {
        const int slot = g_meta_view_slot;
        if (slot > 0 && stc_compute[slot][0].ctxs.empty()) {
            for (auto & stc : stc_compute[slot]) {
                stc = ggml_backend_meta_simple_tensor_container(stc_compute_params, (int) bufs.size());
            }
        }
        return stc_compute[slot][index];
    }
};

static void ggml_backend_meta_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    delete buf_ctx;
}

static size_t ggml_backend_meta_buffer_n_bufs(ggml_backend_buffer_t meta_buf) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    return buf_ctx->bufs.size();
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_simple_buffer(ggml_backend_buffer_t meta_buf, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    GGML_ASSERT(index < buf_ctx->bufs.size());
    return buf_ctx->bufs[index].get();
}

static struct ggml_tensor * ggml_backend_meta_buffer_simple_tensor(const struct ggml_tensor * tensor, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    GGML_ASSERT(index < buf_ctx->bufs.size());

    ggml_backend_meta_simple_tensor_container & stc = buf_ctx->get_simple_tensor_container(tensor);
    auto it = stc.simple_tensors.find(tensor);
    if (it == stc.simple_tensors.end()) {
        return nullptr;
    }
    return it->second[index];
}

static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(const struct ggml_tensor * tensor, bool assume_sync);

// [TAG_META_UNALLOC_ALIAS]
// A view tensor does not own its buffer: `ggml_gallocr_init_tensor()` gives it the buffer of the root
// of its view chain (`ggml_backend_view_init()`).  A *nested* view, however, is never visited by the
// allocator: the graph builder reaches an intermediate alias only through `view_src`, and the
// allocator initialises views from `view_src`, not from `src[0]`.  `build_attn_mha()` builds exactly
// such a pair -- `ggml_view_4d(Qcur, ...)` followed by `ggml_permute()` -- and the permuted tensor is a
// plain alias (GGML_OP_NONE, no node), so its `src[0]` (the intermediate view) stays unallocated for
// the whole run.  The split-state walk below does look at `src[]` of an alias, so it must resolve such
// a tensor the way the allocator eventually will: through its view chain to the buffer it inherits.
// This is reachable whenever an op with host-resident weights runs on the meta backend (a CPU-offloaded
// dense layer under `-sm tensor`, i.e. a --fit that reduced n_gpu_layers): the split-state walk then
// inspects the classic attention `ggml_view_4d` -> `ggml_permute` pair and aborted on the unallocated
// intermediate view (issue #170).
static ggml_backend_buffer_t ggml_backend_meta_view_chain_buffer(const struct ggml_tensor * tensor) {
    while (tensor != nullptr && tensor->buffer == nullptr && tensor->view_src != nullptr) {
        tensor = tensor->view_src;
    }
    return tensor != nullptr ? tensor->buffer : nullptr;
}

static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(
        ggml_backend_meta_simple_tensor_container & stc, const struct ggml_tensor * tensor, bool assume_sync) {
    // [TAG_META_UNALLOC_ALIAS] an alias tensor may not have been given its inherited buffer yet
    ggml_backend_buffer_t buffer = tensor->buffer;
    if (buffer == nullptr) {
        buffer = ggml_backend_meta_view_chain_buffer(tensor);
        if (buffer == nullptr || !ggml_backend_buffer_is_meta(buffer)) {
            // nothing in the view chain is in a meta buffer (host, another device's buffer, or not
            // allocated anywhere): the tensor is not split across the sub-devices
            return {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1};
        }
    }
    // FIXME Currently this function preserves/erases the information in n_segments and nr in an inconsistent way.
    // Since the operations in question are developed specifically for llama.cpp this currently does not manifest as a bug there.
    // However, in a broader ggml context with arbitrary ggml graphs this can lead to unexpected results.
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;

    auto split_states_equal = [&](const ggml_backend_meta_split_state & a, const ggml_backend_meta_split_state & b) -> bool {
        if (a.axis != b.axis) {
            return false;
        }
        for (size_t j = 0; j < n_bufs; j++) {
            int64_t sum_a = 0;
            for (size_t s = 0; s < a.n_segments; s++) {
                sum_a += a.ne[s*n_bufs + j] * a.nr[s];
            }
            int64_t sum_b = 0;
            for (size_t s = 0; s < b.n_segments; s++) {
                sum_b += b.ne[s*n_bufs + j] * b.nr[s];
            }
            if (sum_a != sum_b) {
                return false;
            }
        }
        return true;
    };

    auto handle_generic = [&](const std::vector<ggml_backend_meta_split_state> & src_ss, bool scalar_only) -> ggml_backend_meta_split_state {
        ggml_backend_meta_split_state ret = {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1};
        for (size_t i = 0; i < GGML_MAX_SRC; i++) {
            if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                continue;
            }
            if (ret.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
                ret = src_ss[i];
            } else if (!split_states_equal(src_ss[i], ret)) {
                ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
                break;
            }
        }
        if (ret.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
            ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        if (scalar_only && ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
            ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        if (ret.axis == GGML_BACKEND_SPLIT_AXIS_UNKNOWN) {
            // the operands' splits disagree and the op is not one that can reconcile them
            std::string ops;
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                if (tensor->src[i] != nullptr) {
                    ops += " src" + std::to_string(i) + "=" + ggml_backend_meta_split_axis_name(src_ss[i].axis);
                }
            }
            GGML_LOG_ERROR("%s: no split state for %s (%s)%s\n", __func__, tensor->name, ggml_op_name(tensor->op), ops.c_str());
        }
        GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        return ret;
    };

    // Some ops process data on a per-row bases:
    auto handle_per_row = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        // dim 0 is the reduction axis, so an axis-0 split is normally a CROSS-DEVICE reduction.  The exception is
        // a split that no device straddles: if every per-device extent is either empty or the WHOLE row, each
        // device owns complete rows and reduces them locally, and the state passes through unchanged.
        // gemma4's full-attention layers have a single KV head, so their 512-wide K/V projection row is placed
        // whole on one device (measured `ne_per_dev = [0, 512]`); the per-head K norm, the V rms_norm and the
        // rope then find an axis-0 split whose every row is whole.  A genuine straddle (e.g. [256, 256] over a
        // 512 row) is still rejected -- that one really would need a cross-device reduction.
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0) {
            bool whole_rows = true;
            for (size_t s = 0; s < src_ss[0].n_segments && whole_rows; s++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    const int64_t extent = src_ss[0].ne[s*n_bufs + j] * src_ss[0].nr[s];
                    if (extent != 0 && extent != tensor->src[0]->ne[0]) {
                        whole_rows = false;
                        break;
                    }
                }
            }
            GGML_ASSERT(whole_rows && "per-row op over a row STRADDLING an axis-0 split needs a cross-device "
                                      "reduction (not implemented)");
        }
        return src_ss[0];
    };

    // Some ops broadcast the src1 data across src0:
    auto handle_bin_bcast = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        // GGML_BACKEND_SPLIT_AXIS_PER_DEVICE (the MoE amax rescale, see the enum comment) is only ever
        // consumed elementwise and carries its own value on the backend that produced it, so an elementwise
        // op simply follows the other operand's split.
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS &&
                tensor->src[1]->ne[src_ss[0].axis] == 1 && (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED ||
                                                            src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_PER_DEVICE)) {
            return src_ss[0];
        }
        if (src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[0].axis == src_ss[1].axis ||
           (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL)))) {
            return src_ss[0]; // GGML_OP_ADD_ID
        }
        // A per-device scalar broadcast over any other split: the result follows the other operand.  This is
        // the MoE amax rescale's undo `y * e`, whose other operand is PARTIAL in the real graph (the partial
        // down-projection sum, the factor must be applied before the all-reduce) and MIRRORED in the
        // allocation-time `assume_sync` view.  PER_DEVICE and MIRRORED have identical per-backend extents, so
        // returning the other operand's state keeps the layout -- and the reduce point -- unchanged.
        if (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_PER_DEVICE && tensor->src[0] != nullptr &&
                ggml_can_repeat(tensor->src[1], tensor->src[0])) {
            return src_ss[0];
        }

        // An elementwise op whose OTHER operand is MIRRORED and of the same shape: the result follows the split
        // operand, whatever its split is, because nothing is reduced and the mirrored operand is present in full
        // on every backend.  A per-head norm weight is exactly this: gemma4's full-attention K/V row is 512 wide
        // with a single KV head, so the row lands whole on one device (axis-0 split, extents [0, 512]) while the
        // [512] norm weight is MIRRORED -- the broadcast case above only covers ne[axis] == 1.
        if ((src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED || src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_PER_DEVICE) &&
                tensor->src[1] != nullptr && ggml_are_same_shape(tensor->src[0], tensor->src[1])) {
            return src_ss[0];
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && tensor->src[1] != nullptr &&
                ggml_are_same_shape(tensor->src[0], tensor->src[1])) {
            return src_ss[1];
        }
        GGML_ASSERT(tensor->src[2] == nullptr || src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_concat = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        const ggml_backend_meta_split_axis concat_axis = ggml_backend_meta_split_axis(ggml_get_op_params_i32(tensor, 0));
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis >= 0 && src_ss[1].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[1].axis);
            return src_ss[1];
        }
        if (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[0].axis);
            return src_ss[0];
        }
        if (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis != concat_axis) {
            return src_ss[0];
        }
        return handle_generic(src_ss, /*scalar_only =*/ true);
    };

    auto handle_mul_mat = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            ggml_backend_meta_split_state ret = src_ss[0];
            ret.axis = GGML_BACKEND_SPLIT_AXIS_0;
            ret.nr[0] = 1;
            ret.n_segments = 1;
            return ret;
        }
        if (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[1];
        }
        // The rotated-KV cache (`-ctk q8_0`, n_rot_k > 0) rotates K with a MIRRORED n_rot_k x n_rot_k matrix:
        //     out[rot_row, i1] = sum_k rot[k, rot_row] * cur[k, i1],  cur = Kcur reshaped to [n_rot_k, rows]
        // so the contraction axis of `cur` is axis 0.  Usually the head axis gives that split (gemma4's
        // single-KV-head full-attention layers reshape a whole [512, 1] row set, i.e. axis-0 extents are
        // `[0, 512]`): no device straddles the contraction, the device holding the whole range computes the
        // complete dot product (every rot column is available to it, src0 is MIRRORED) and the others hold an
        // empty slice, so the reduction is local and nothing is summed across devices.  The output follows
        // `cur`'s split, which is exactly the state the unrotated Kcur has after the same reshape: gemma4's
        // 512-wide row is whole on one device, so the rotation's output is whole on that device too.
        // A *straddling* axis-0 split (e.g. [256, 256] over a 512-wide contraction) is a genuine cross-device
        // partial sum; it is rejected rather than silently wrong.
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_0) {
            const ggml_tensor * cur = tensor->src[1];
            GGML_ASSERT(tensor->ne[0] == cur->ne[0] &&
                    "mul_mat of a MIRRORED src0 over an axis-0-split src1 needs out->ne[0] == src1->ne[0] "
                    "(a square contraction) so each device's output slice maps 1:1 onto its contraction slice");
            for (size_t s = 0; s < src_ss[1].n_segments; s++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    const int64_t extent = src_ss[1].ne[s*n_bufs + j] * src_ss[1].nr[s];
                    GGML_ASSERT((extent == 0 || extent == cur->ne[0]) &&
                            "mul_mat of a MIRRORED src0 over an axis-0 split that STRADDLES the contraction "
                            "needs a cross-device reduction (not implemented)");
                }
            }
            return src_ss[1];
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(split_states_equal(src_ss[0], src_ss[1]));
            return {assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}, {1}, 1};
        }
        if (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis >= GGML_BACKEND_SPLIT_AXIS_2 &&
                src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(split_states_equal(src_ss[0], src_ss[1]));
            return src_ss[0];
        }
        // batched matmul with the batches split across devices and a replicated activation
        if (src_ss[0].axis >= GGML_BACKEND_SPLIT_AXIS_2 && src_ss[0].axis < GGML_MAX_DIMS &&
                src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        GGML_ABORT("unsupported mul_mat split states: node=%s src0=%s axis=%d src1=%s axis=%d",
            tensor->name, tensor->src[0]->name, (int) src_ss[0].axis, tensor->src[1]->name, (int) src_ss[1].axis);
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    auto handle_reshape = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1:
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3: {
                int64_t base_ne_in = 1;
                for (int dim = 0; dim <= src_ss[0].axis; dim++) {
                    base_ne_in *= tensor->src[0]->ne[dim];
                }
                if (src_ss[0].n_segments == 1) {
                    base_ne_in /= src_ss[0].nr[0];
                    if (src_ss[0].axis == ggml_n_dims(tensor->src[0]) - 1 && src_ss[0].nr[0] == 1) {
                        return {ggml_backend_meta_split_axis(ggml_n_dims(tensor) - 1), {0}, {1}, 1};
                    }
                    if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && tensor->ne[0] == tensor->src[0]->ne[0] &&
                            tensor->ne[1] == 1 && src_ss[0].nr[0] == 1) {
                        bool complete_rows = true;
                        for (size_t j = 0; j < n_bufs; j++) {
                            const int64_t ne = src_ss[0].ne[j];
                            complete_rows = complete_rows && (ne == 0 || ne == tensor->src[0]->ne[0]);
                        }
                        if (complete_rows) {
                            // Move a complete dim-0 split to the following singleton dimension.
                            return {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
                        }
                    }
                }
                // Reshape outputs use one segment; split-state propagation merges source segments.
                int64_t base_ne_out = 1;
                for (int dim = 0; dim < GGML_MAX_DIMS; dim++) {
                    base_ne_out *= tensor->ne[dim];
                    if (base_ne_out % base_ne_in == 0) {
                        return {ggml_backend_meta_split_axis(dim), {0}, {uint32_t(base_ne_out/base_ne_in)}, 1};
                    }
                    if (base_ne_out > base_ne_in) {
                        GGML_ASSERT(src_ss[0].n_segments == 1);
                        GGML_ASSERT(src_ss[0].nr[0]      == 1);
                        return {ggml_backend_meta_split_axis(dim), {0}, {1}, 1};
                    }
                }
                GGML_ABORT("shape mismatch for %s", ggml_op_name(tensor->op));
            }
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_cpy = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            return handle_reshape(src_ss);
        }
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_view = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (ggml_is_contiguous(tensor) && ggml_is_contiguous(tensor->src[0])) {
            return handle_reshape(src_ss);
        }
        const int axis = src_ss[0].axis;
        {
            bool all_strides_the_same = true;
            for (int dim = 0; dim < GGML_MAX_DIMS; dim++) {
                if (tensor->ne[dim] == 1 && tensor->src[0]->ne[dim] == 1) {
                    continue;
                }
                if (tensor->nb[dim] != tensor->src[0]->nb[dim]) {
                    all_strides_the_same = false;
                    break;
                }
            }
            if (all_strides_the_same) {
                return src_ss[0];
            }
        }
        if (!ggml_is_permuted(tensor) && !ggml_is_permuted(tensor->src[0]) && axis >= 0 && axis < GGML_MAX_DIMS-1) {
            for (int dim = 0; dim < GGML_MAX_DIMS-1; dim++) {
                if (tensor->nb[dim+1] == tensor->src[0]->nb[axis+1]) {
                    return {ggml_backend_meta_split_axis(dim), {0}, {1}, 1};
                }
            }
            GGML_ABORT("fatal error");
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED || src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
            return src_ss[0];
        }
        GGML_ABORT("view of permuted tensor not implemented");
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
    };

    auto handle_permute = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1:
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                return {ggml_backend_meta_split_axis(tensor->op_params[src_ss[0].axis]), {0}, {src_ss[0].nr[0]}, 1};
            }
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_transpose = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1: {
                GGML_ASSERT(src_ss[0].n_segments == 1 || src_ss[0].nr[0] == 1);
                return {ggml_backend_meta_split_axis(int(src_ss[0].axis) ^ 1), {0}, {src_ss[0].nr[0]}, 1};
            }
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3:
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            }
        }
    };

    auto handle_get_rows = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        return handle_generic(src_ss, /*scalar_only =*/ true);
    };

    // ALL_GATHER: the result is the whole tensor on every backend, so it is MIRRORED whatever the source
    // split is.  The collective itself runs between the subgraphs (see `graph_compute`): each device's
    // slice is written into every device's copy at that slice's own offset, which keeps the element
    // positions of a split axis-0 input (the Draft/DFlash2 selector's global vocabulary indices).
    auto handle_all_gather = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_PER_DEVICE &&
                "a per-device (relaxed-tensor) tensor cannot be gathered: every backend already holds a full,\n"
                "different copy -- the value is only defined for elementwise use on its own backend");
        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED ||
                (src_ss[0].axis >= GGML_BACKEND_SPLIT_AXIS_0 && src_ss[0].axis < GGML_MAX_DIMS));
        return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
    };

    auto handle_set_rows = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        GGML_ASSERT(split_states_equal(src_ss[0], src_ss[2]));
        return src_ss[0];
    };

    auto handle_rope = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        return src_ss[0];
    };

    auto handle_pad = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 0] == 0);
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 1] == 0);
        }
        return src_ss[0];
    };

    // The MoE amax rescale (`build_amax_src1_f16`) pools a whole matmul row to one value per [expert, token]:
    //     amax = pool_1d(abs(cur), POOL_MAX, k = s = cur->ne[0])   // [1, n_expert_used, n_tokens]
    // and the graph then divides `cur` by a power-of-two function of it and multiplies the down-projection
    // result back by the same value.  Over an axis-0 split no backend holds a complete row, so each backend
    // computes the max of its own slice (the kernels clamp the window to `src->ne[0]`, i.e. the slice).
    //
    // The result is therefore the SAME SHAPE everywhere with DIFFERENT values -> PER_DEVICE, never MIRRORED
    // (claiming MIRRORED here would be a false claim; upstream #31081 removed exactly such a case).  It is
    // sound because the value is only consumed elementwise on the backend that produced it and its inverse is
    // applied to that backend's own partial before the all-reduce: `e` is a power of two, so `sum(w . (x/e)) * e
    // == sum(w . x)`, exactly, and `e_local <= e_global` means every backend keeps at least the reference's
    // f16 headroom.  See GGML_BACKEND_SPLIT_AXIS_PER_DEVICE and `-sm relaxed-tensor`.
    auto handle_pool_1d = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0) {
            // Only the full-row MAX reduction that the amax rescale builds is understood.  A windowed pool
            // over a split axis would change the OUTPUT shape per backend, which no split state can express.
            const int32_t * p = (const int32_t *) tensor->op_params;
            GGML_ASSERT(p[0] == GGML_OP_POOL_MAX);
            GGML_ASSERT(p[1] == p[2] && p[1] >= tensor->src[0]->ne[0] + 2*p[3] &&
                    "POOL_1D over an axis-0 split is only supported for the whole-row amax max reduction");
            return {GGML_BACKEND_SPLIT_AXIS_PER_DEVICE, {0}, {1}, 1};
        }
        return src_ss[0];
    };

    auto handle_flash_attn_ext = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(tensor->src[3] == nullptr || src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            GGML_ASSERT(tensor->src[4] == nullptr || src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
        }

        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_2);
        const bool kv_split = src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_2 &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_2;
        const bool kv_mirrored = src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED;
        // A KV slice that is empty-or-whole on every device: gemma4's single-KV-head full-attention layers
        // put the whole 512-wide K/V row (granularity == the whole row) and therefore the whole KV cache on
        // ONE device.  Depending on how the cache is viewed that state arrives split on the kv-head axis
        // (`[0, 1]`, already accepted above) or on the cache's embd axis (`[0, 512]`, reached when the ubatch
        // carries several sequences and the view spans the cache's stream dimension -- llama-server
        // `-np > 1`, `llama-batched-bench -npl > 1`, `llama-perplexity --kl-divergence`).  Both mean the same
        // thing: nothing is reduced, the owning device computes the whole op, and a device with a zero-sized
        // source has its node disabled (`GGML_TENSOR_FLAG_COMPUTE` clear) exactly as in the axis-2 case.
        // The query must be empty-or-whole *on the same devices* as the KV: a device holding queries but no
        // K/V would compute rows nothing serves, which is the genuinely-broken combination this rejects.
        auto whole_mask = [&](const ggml_backend_meta_split_state & ss, const ggml_tensor * t, uint32_t & mask) -> bool {
            mask = 0;
            if (ss.axis < 0 || ss.axis >= GGML_MAX_DIMS || t == nullptr) {
                return false;
            }
            for (size_t j = 0; j < n_bufs; j++) {
                int64_t extent = 0;
                for (size_t s = 0; s < ss.n_segments; s++) {
                    extent += ss.ne[s*n_bufs + j] * ss.nr[s];
                }
                if (extent == 0) {
                    continue;
                }
                if (extent != t->ne[ss.axis]) {
                    return false;
                }
                mask |= 1u << j;
            }
            return mask != 0;
        };
        uint32_t mask_k = 0, mask_v = 0, mask_q = 0;
        const bool kv_whole = whole_mask(src_ss[1], tensor->src[1], mask_k) &&
                whole_mask(src_ss[2], tensor->src[2], mask_v) &&
                whole_mask(src_ss[0], tensor->src[0], mask_q) &&
                mask_k == mask_v && (mask_q & mask_k) == mask_k;
        GGML_ASSERT(kv_split || kv_mirrored || kv_whole);
        GGML_ASSERT(tensor->src[4] == nullptr || src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_0);
        return {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
    };

    // qwen4exp QSA sparse attention: same split as FLASH_ATTN_EXT (q by head,
    // K/V by kv-head, result by head); idx and mask are mirrored.
    auto handle_flash_attn_qsa = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        // src4 (mask) plus the optional derived-visibility keys are replicated when present; a
        // null src slot (the mask on the derived path) carries no split state
        for (size_t i = 4; i < 7; i++) {
            GGML_ASSERT(src_ss[i].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED ||
                    src_ss[i].axis == GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        }

        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_2);
        const bool kv_split = src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_2 &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_2;
        const bool kv_mirrored = src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED;
        GGML_ASSERT(kv_split || kv_mirrored);
        return {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
    };

    auto handle_lightning_indexer = [&](
            const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        for (size_t i = 0; i < 4; i++) {
            GGML_ASSERT(src_ss[i].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        }
        return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
    };

    auto handle_ssm_conv = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == src_ss[1].axis) {
            if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0) {
                return {GGML_BACKEND_SPLIT_AXIS_1, {0}, {1}, 1};
            }
            if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1) {
                return {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
            }
        }
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_gated_delta_net = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
                src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_1);
        // state shape is [S_v, S_v, H_v, n_seqs] (s0 only); the heads dim is its own axis 2,
        // so a head-aligned split on the input cache lands on axis 2 here.
        GGML_ASSERT(src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_2 || src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_1 || src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_0);
        return {GGML_BACKEND_SPLIT_AXIS_0, {0}, {1}, 1};
    };

    auto calculate_split_state = [&]() -> ggml_backend_meta_split_state {
        if (ggml_nelements(tensor) == 0) {
            return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
        }
        // Tensors the per-tensor policy can answer for: a weight held in a static buffer, or the
        // scheduler's split-input copy of one.  A host-resident weight that is op-offloaded reaches the
        // meta backend **only** as such a copy (`ggml_dup_tensor_layout` of the weight, so op NONE with
        // no view source, allocated in the split's COMPUTE buffer -- see
        // `ggml_backend_sched_alloc_splits`), and the policy has to be asked for it too: the COMPUTE
        // branch below would otherwise propagate it from `GGML_OP_NONE` and return mirrored, which
        // uploads the whole weight to every device.  A tensor the policy does not recognise (any
        // activation copy) keeps the mirrored default, so this is a no-op for them.
        const bool policy_tensor =
            tensor->view_src == nullptr &&
            (ggml_backend_buffer_get_usage(buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE ||
             tensor->op == GGML_OP_NONE);
        if (policy_tensor) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(buffer));
            const ggml_backend_meta_device_context * dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
            ggml_backend_meta_split_state ret = dev_ctx->get_split_state(tensor, dev_ctx->get_split_state_ud);
            if (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
                const int64_t granularity = ret.axis == GGML_BACKEND_SPLIT_AXIS_0 ? ggml_blck_size(tensor->type) : 1;
                int64_t ne_sum = 0;
                for (size_t s = 0; s < ret.n_segments; s++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        GGML_ASSERT(ret.ne[s*n_bufs + j] % granularity == 0);
                        ne_sum += ret.ne[s*n_bufs + j] * ret.nr[s];
                    }
                }
                GGML_ASSERT(ne_sum == tensor->ne[ret.axis]);
            } else if (ret.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                GGML_ASSERT(ret.n_segments == 1);
                GGML_ASSERT(ret.nr[0] == 1);
            }
            return ret;
        }

        std::vector<ggml_backend_meta_split_state> src_ss(GGML_MAX_SRC, {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1});
        for (size_t i = 0; i < GGML_MAX_SRC; i++) {
            if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                src_ss[i] = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
                continue;
            }
            src_ss[i] = ggml_backend_meta_get_split_state(stc, tensor->src[i], /*assume_sync =*/ true);
            GGML_ASSERT(src_ss[i].axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        }

        ggml_backend_meta_split_state split_state;
        switch (tensor->op) {
            case GGML_OP_NONE: {
                if (tensor->view_src != nullptr) {
                    // full-tensor view created with ggml_view_tensor, transparent for the split state
                    split_state = ggml_backend_meta_get_split_state(stc, tensor->view_src, assume_sync);
                } else {
                    split_state = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
                }
            } break;
            case GGML_OP_DUP: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_ADD:
            case GGML_OP_ADD_ID: {
                split_state = handle_bin_bcast(src_ss);
            } break;
            case GGML_OP_ADD1:
            case GGML_OP_ACC: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SUB:
            case GGML_OP_MUL:
            case GGML_OP_DIV: {
                split_state = handle_bin_bcast(src_ss);
            } break;
            case GGML_OP_SQR:
            case GGML_OP_SQRT:
            case GGML_OP_LOG:
            case GGML_OP_SIN:
            case GGML_OP_COS: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_SUM: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SUM_ROWS:
            case GGML_OP_CUMSUM:
            case GGML_OP_MEAN:
            case GGML_OP_ARGMAX:
            case GGML_OP_COUNT_EQUAL: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_REPEAT:
            case GGML_OP_REPEAT_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_CONCAT: {
                split_state = handle_concat(src_ss);
            } break;
            case GGML_OP_SILU_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_NORM:
            case GGML_OP_RMS_NORM:
            case GGML_OP_RMS_NORM_BACK:
            case GGML_OP_GROUP_NORM:
            case GGML_OP_L2_NORM: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_MUL_MAT:
            case GGML_OP_MUL_MAT_ID: {
                split_state = handle_mul_mat(src_ss);
            } break;
            case GGML_OP_OUT_PROD: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SCALE: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_SET: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_CPY: {
                split_state = handle_cpy(src_ss);
            } break;
            case GGML_OP_CONT:
            case GGML_OP_RESHAPE: {
                split_state = handle_reshape(src_ss);
            } break;
            case GGML_OP_VIEW: {
                split_state = handle_view(src_ss);
            } break;
            case GGML_OP_PERMUTE: {
                split_state = handle_permute(src_ss);
            } break;
            case GGML_OP_TRANSPOSE: {
                split_state = handle_transpose(src_ss);
            } break;
            case GGML_OP_GET_ROWS: {
                split_state = handle_get_rows(src_ss);
            } break;
            case GGML_OP_GET_ROWS_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SET_ROWS: {
                split_state = handle_set_rows(src_ss);
            } break;
            case GGML_OP_DIAG:
            case GGML_OP_DIAG_MASK_INF:
            case GGML_OP_DIAG_MASK_ZERO: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SOFT_MAX:
            case GGML_OP_SOFT_MAX_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_ROPE: {
                split_state = handle_rope(src_ss);
            } break;
            case GGML_OP_ROPE_BACK: {
                split_state = handle_rope(src_ss);
            } break;
            case GGML_OP_CLAMP: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_CONV_TRANSPOSE_1D:
            case GGML_OP_IM2COL:
            case GGML_OP_IM2COL_BACK:
            case GGML_OP_IM2COL_3D:
            case GGML_OP_CONV_2D:
            case GGML_OP_CONV_3D:
            case GGML_OP_CONV_2D_DW:
            case GGML_OP_CONV_TRANSPOSE_2D:
            case GGML_OP_POOL_2D:
            case GGML_OP_POOL_2D_BACK:
            case GGML_OP_UPSCALE: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_POOL_1D: {
                split_state = handle_pool_1d(src_ss);
            } break;
            case GGML_OP_PAD: {
                split_state = handle_pad(src_ss);
            } break;
            case GGML_OP_PAD_REFLECT_1D:
            case GGML_OP_ROLL:
            case GGML_OP_ARANGE:
            case GGML_OP_TIMESTEP_EMBEDDING: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_ARGSORT:
            case GGML_OP_TOP_K: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_ALL_GATHER: {
                split_state = handle_all_gather(src_ss);
            } break;
            case GGML_OP_LEAKY_RELU: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_TRI: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_FILL: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_FLASH_ATTN_EXT: {
                split_state = handle_flash_attn_ext(src_ss);
            } break;
            case GGML_OP_FLASH_ATTN_QSA: {
                split_state = handle_flash_attn_qsa(src_ss);
            } break;
            case GGML_OP_INDEXER_TOPK: {
                // like TOP_K: the row (tps, stream) is the independent unit.  every src is
                // mirrored: the cell map plus the optional derived position/bias compact srcs
                for (int i = 0; i < GGML_MAX_SRC; ++i) {
                    if (tensor->src[i] != nullptr) {
                        GGML_ASSERT(src_ss[i].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
                    }
                }
                split_state = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
            } break;
            case GGML_OP_INDEXER_SCORE: {
                // like INDEXER_TOPK: mirrored indexer cache view + cell map + positions +
                // query + norm weights + bias (each device holds full mirrored copies);
                // the optional derived-cache pool view + range leaf are mirrored too
                for (int i = 0; i < GGML_MAX_SRC; ++i) {
                    if (tensor->src[i] != nullptr) {
                        GGML_ASSERT(src_ss[i].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
                    }
                }
                split_state = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
            } break;
            case GGML_OP_INDEXER_FILL: {
                // mirrored like the other indexer ops: raw cache view + cell map + positions + norm
                // weights + the host range leaf + the pool buffer it writes
                for (int i = 0; i < GGML_MAX_SRC; ++i) {
                    if (tensor->src[i] != nullptr) {
                        GGML_ASSERT(src_ss[i].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
                    }
                }
                split_state = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
            } break;
            case GGML_OP_FLASH_ATTN_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SSM_CONV: {
                split_state = handle_ssm_conv(src_ss);
            } break;
            case GGML_OP_SSM_SCAN:
            case GGML_OP_WIN_PART:
            case GGML_OP_WIN_UNPART:
            case GGML_OP_GET_REL_POS:
            case GGML_OP_ADD_REL_POS:
            case GGML_OP_RWKV_WKV6:
            case GGML_OP_GATED_LINEAR_ATTN:
            case GGML_OP_RWKV_WKV7:
            case GGML_OP_SOLVE_TRI: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_GATED_DELTA_NET: {
                split_state = handle_gated_delta_net(src_ss);
            } break;
            case GGML_OP_LIGHTNING_INDEXER: {
                split_state = handle_lightning_indexer(src_ss);
            } break;
            case GGML_OP_DSV4_HC_COMB:
            case GGML_OP_DSV4_HC_PRE:
            case GGML_OP_DSV4_HC_POST:
            case GGML_OP_TOP_K_MASK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_HC_MIX: {
                // the hc weights and the per-token tensors are mirrored per
                // device (verified on the qwen4exp decode graph); each device
                // runs the full op and produces its own local copy. The output
                // carries both mixed and the inject scatter (the combine views
                // the tail), all mirrored. src[4] (w_inject) is null at the
                // head call (il = -1), which the splitter marks UNKNOWN.
                GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
                GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
                GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
                GGML_ASSERT(src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
                GGML_ASSERT(src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED || src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
                split_state = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
            } break;
            case GGML_OP_HC_COMBINE: {
                // the hc weights and the per-token tensors are mirrored per
                // device (verified on the qwen4exp decode graph); each device
                // runs the full op and produces its own local copy.
                GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
                GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
                GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
                split_state = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
            } break;
            case GGML_OP_UNARY: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_MAP_CUSTOM1:
            case GGML_OP_MAP_CUSTOM2:
            case GGML_OP_MAP_CUSTOM3:
            case GGML_OP_CUSTOM: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_CROSS_ENTROPY_LOSS:
            case GGML_OP_CROSS_ENTROPY_LOSS_BACK: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_OPT_STEP_ADAMW:
            case GGML_OP_OPT_STEP_SGD:
            case GGML_OP_GLU: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            default: {
                GGML_ABORT("ggml op not implemented: %s", ggml_op_name(tensor->op));
                split_state = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, {1}, 1};
            } break;
        }
        if (split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS) {
            bool first_src_split_by_axis = true;
            const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);

            for (size_t i = 0; i < GGML_MAX_SRC; i++) {
                if (tensor->src[i] == nullptr || src_ss[i].axis < 0 || src_ss[i].axis >= GGML_MAX_DIMS) {
                    continue;
                }
                if (first_src_split_by_axis) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        // Take over ratio from src:
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            split_state.ne[s*n_bufs + j] = 0;
                        }
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            split_state.ne[j] += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        split_state.ne[j] *= tensor->ne[split_state.axis];
                        if (split_state.ne[j] != 0 || tensor->src[i]->ne[src_ss[i].axis] != 0) {
                            const int64_t div = tensor->src[i]->ne[src_ss[i].axis] * split_state.nr[0];
                            GGML_ASSERT(split_state.ne[j] % div == 0);
                            split_state.ne[j] /= div;
                        }
                    }
                } else {
                    GGML_ASSERT(split_state.n_segments == 1);
                    for (size_t j = 0; j < n_bufs; j++) {
                        // Assert that ratio is consistent:
                        int64_t sum = 0;
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            sum += src_ss[i].ne[s*n_bufs + j] * src_ss[i].nr[s];
                        }
                        GGML_ASSERT(split_state.ne[j]*split_state.nr[0] * tensor->src[i]->ne[src_ss[i].axis]
                                                                 == sum * tensor->ne[split_state.axis]);
                    }
                }
                first_src_split_by_axis = false;
            }
            GGML_ASSERT(!first_src_split_by_axis);
        }
        return split_state;
    };

    // Non-recursive pre-warm of the per-buffer-context split-state cache, run only on a cache miss of
    // the tensor itself.  `calculate_split_state` recurses into every `src` and each frame is large; a
    // deep `src` chain (qwen35moe under `-sm tensor` with MTP builds a chain over 1200 nodes long, and
    // that many frames overflow the 8 MiB thread stack) crashed the allocator with a stack overflow.
    // The outermost call walks the not-yet-cached ancestors on an explicit heap stack and computes them
    // bottom-up, so each recursive lookup below is a cache hit and the C++ recursion depth stays at one.
    // Recursive calls see `ss_warm_active != 0` and skip this; a fully cached lookup never enters the
    // walk, so the steady-state hit path is unchanged.  The walk is cycle-safe and visits each node once.
    static thread_local int ss_warm_active = 0;
    struct ss_warm_guard {
        bool owner;
        ss_warm_guard() : owner(ss_warm_active == 0) { if (owner) { ss_warm_active = 1; } }
        ~ss_warm_guard() { if (owner) { ss_warm_active = 0; } }
    } ss_warm;

    // Current-version cache test, used only to prune the walk (a false negative just visits a node that
    // the nested lookup then resolves from another version, so pruning is always safe).
    auto ss_cached = [&](const ggml_tensor * t) -> bool {
        const std::pair k = std::make_pair(t, true);
        const auto & c = buf_ctx->split_state_cache[buf_ctx->split_state_cache_cur];
        const auto itc = c.find(k);
        return itc != c.end() && memcmp(itc->second.bytes, (const char *) t, sizeof(itc->second.bytes)) == 0;
    };
    if (ss_warm.owner && !ss_cached(tensor)) {
        // The walk's containers are scratch: one heap node per visited tensor (plus the rehash and
        // the frees) was measurable on a 4-stream server, where a graph re-allocation is frequent.
        // A reused per-thread arena, released per walk, keeps that allocation off the heap.  Only the
        // outermost call walks (ss_warm_active), so the resource is never re-entered; the walk visits
        // the same tensors in the same order and computes the same states, and if the arena runs out
        // the monotonic resource falls back to its upstream allocator, so nothing is capped.
        // GGML_META_SS_SCRATCH=0: the containers allocate from the default (heap) resource -- A/B and
        // bisection only.
        struct ss_warm_scratch {
            std::pmr::monotonic_buffer_resource res{1u << 20};
        };
        static const bool ss_scratch_on =
            GGML_ENV_STR("GGML_META_SS_SCRATCH") == nullptr || atoi(GGML_ENV_STR("GGML_META_SS_SCRATCH")) != 0;
        static thread_local ss_warm_scratch ss_scratch;
        std::pmr::memory_resource * ss_mr = std::pmr::get_default_resource();
        if (ss_scratch_on) {
            ss_scratch.res.release();
            ss_mr = &ss_scratch.res;
        }
        std::pmr::vector<std::pair<const ggml_tensor *, uint32_t>> wstack(ss_mr);
        std::pmr::unordered_set<const ggml_tensor *> wseen(ss_mr);
        wstack.emplace_back(tensor, 0);
        wseen.insert(tensor);
        while (!wstack.empty()) {
            const ggml_tensor * t = wstack.back().first;
            uint32_t & wi = wstack.back().second;
            if (wi < GGML_MAX_SRC) {
                const ggml_tensor * s = t->src[wi++];
                if (s != nullptr && s != t && wseen.insert(s).second && !ss_cached(s)) {
                    wstack.emplace_back(s, 0);
                }
                continue;
            }
            wstack.pop_back();
            if (t != tensor) {
                ggml_backend_meta_get_split_state(stc, t, /*assume_sync =*/ true);
            }
        }
    }

    const std::pair key = std::make_pair(tensor, assume_sync);
    auto * cache = &buf_ctx->split_state_cache[buf_ctx->split_state_cache_cur];
    auto it = cache->find(key);
    if (it != cache->end() && memcmp(it->second.bytes, (const char *) tensor, sizeof(it->second.bytes)) != 0) {
        // the tensor changed: switch to the cache version in which it is unchanged, else recycle the least recently used one
        int k_new = -1;
        for (int k = 0; k < buf_ctx->n_split_state_caches && k_new < 0; k++) {
            if (k == buf_ctx->split_state_cache_cur) {
                continue;
            }
            auto it_k = buf_ctx->split_state_cache[k].find(key);
            if (it_k != buf_ctx->split_state_cache[k].end() && memcmp(it_k->second.bytes, (const char *) tensor, sizeof(it_k->second.bytes)) == 0) {
                k_new = k;
            }
        }
        if (k_new < 0) {
            for (int k = 0; k < buf_ctx->n_split_state_caches; k++) {
                if (k != buf_ctx->split_state_cache_cur &&
                        (k_new < 0 || buf_ctx->split_state_cache_used[k] < buf_ctx->split_state_cache_used[k_new])) {
                    k_new = k;
                }
            }
            buf_ctx->split_state_cache[k_new].clear();
        }
        buf_ctx->split_state_cache_cur = k_new;
        buf_ctx->split_state_cache_used[k_new] = ++buf_ctx->split_state_cache_tick;
        cache = &buf_ctx->split_state_cache[k_new];
        it = cache->find(key);
    }

    // GGML_META_SS_VERIFY=1 (debug): recompute every outermost cache hit and abort on a stale entry
    static const bool ss_verify = getenv("GGML_META_SS_VERIFY") != nullptr;
    static thread_local bool ss_verify_active = false;
    const bool verify_hit = it != cache->end() && ss_verify && !ss_verify_active;
    if (it == cache->end() || verify_hit) {
        ss_verify_active = ss_verify_active || verify_hit;   // the recomputation's own source lookups are not verified
        const ggml_backend_meta_split_state c = verify_hit ? it->second.get() : ggml_backend_meta_split_state{};   // copy: the recursion may recycle this version
        const ggml_backend_meta_split_state ss = calculate_split_state();
        if (verify_hit) {
            bool same = ss.axis == c.axis && ss.n_segments == c.n_segments;
            for (uint32_t sg = 0; same && sg < ss.n_segments; sg++) {
                same = ss.nr[sg] == c.nr[sg];
                for (size_t j = 0; same && j < n_bufs; j++) {
                    same = ss.ne[sg*n_bufs + j] == c.ne[sg*n_bufs + j];
                }
            }
            if (!same) {
                GGML_ABORT("GGML_META_SS_VERIFY: stale cached split state for %s (op %s)", tensor->name, ggml_op_name(tensor->op));
            }
            ss_verify_active = false;
        }
        // the calculation recurses into the sources and may have switched the current version
        cache = &buf_ctx->split_state_cache[buf_ctx->split_state_cache_cur];
        auto & entry = (*cache)[key];
        entry.set(ss, n_bufs);
        memcpy(entry.bytes, tensor, sizeof(entry.bytes));
        it = cache->find(key);
    }

    const auto & entry = it->second;
    GGML_ASSERT(entry.axis != GGML_BACKEND_SPLIT_AXIS_NONE);
#ifndef NDEBUG
    if (entry.axis >= 0 && entry.axis < GGML_MAX_DIMS) {
        int64_t ne_ret = 0;
        for (size_t s = 0; s < entry.n_segments; s++) {
            for (size_t j = 0; j < n_bufs; j++) {
                ne_ret += entry.ne[s*n_bufs + j] * entry.nr[s];
            }
        }
        assert(ne_ret == tensor->ne[int(entry.axis)]);
    }
#endif // NDEBUG
    // Returned as a prvalue so the ~2 KiB state is built once, directly in the caller's object.  This is the hit path
    // of every lookup (every tensor of every graph allocation under -sm tensor); copying the state out of a named
    // local cost more than the hash lookup itself.
    return entry.get();
}

static bool ggml_backend_meta_is_host_view(const struct ggml_tensor * tensor) {
    return ggml_is_view(tensor) && ggml_backend_buffer_is_host(tensor->view_src->buffer);
}

// [TAG_META_UNALLOC_ALIAS] cold path of the split-state lookup -- a tensor that is *not* in a meta
// buffer: a host tensor/view, another device's buffer, or (issue #170) a nested alias that the
// allocator has not given its inherited buffer yet.  Out of line on purpose: the hot dispatcher below
// runs on every tensor of every graph allocation (~10k lookups per decode step under `-sm tensor`), so
// it keeps the exact pre-#170 two-test shape and this path -- which only the #170 alias case and host
// tensors take -- is kept out of it (issue #184).
#if defined(__GNUC__)
#  define GGML_META_COLD __attribute__((noinline, cold))
#else
#  define GGML_META_COLD
#endif
static GGML_META_COLD struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state_cold(
        const struct ggml_tensor * tensor, bool assume_sync) {
    // [TAG_META_UNALLOC_ALIAS] a nested alias may not have its inherited buffer yet: resolve the chain
    // to the buffer the allocator will give it, so it is classified like the tensor it aliases
    ggml_backend_buffer_t buffer = tensor->buffer;
    if (buffer == nullptr) {
        buffer = ggml_backend_meta_view_chain_buffer(tensor);
        if (buffer == nullptr || !ggml_backend_buffer_is_meta(buffer)) {
            // nothing in the view chain is in a meta buffer (host, another device's buffer, or not
            // allocated anywhere): the tensor is not split across the sub-devices
            return {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1};
        }
    }

    // host tensors are not part of the meta split (e.g. a reshaped token
    // embedding or a PLE gather); treat them as unsplit
    if (ggml_backend_buffer_is_host(buffer)) {
        return {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1};
    }

    // [TAG_META_HOST_VIEWS]
    // TODO: technically, this check should not be needed if the backend scheduler correctly prevents assigning
    //       such host-buffer views to the meta backend. figure out how to update the scheduler logic to achieve that
    //       ref: https://github.com/ggml-org/llama.cpp/pull/30217
    GGML_ASSERT(tensor->buffer == nullptr || ggml_backend_meta_is_host_view(tensor));

    // the view is not allocated in the meta buffer, it is not split across the sub-devices
    return { GGML_BACKEND_SPLIT_AXIS_NONE, {0}, {1}, 1 };
}
#undef GGML_META_COLD

static inline struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(const struct ggml_tensor * tensor, bool assume_sync) {
    // hot path: a tensor in a meta buffer -- unchanged from the pre-#170 dispatcher
    if (ggml_backend_buffer_is_meta(tensor->buffer)) {
        ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
        return ggml_backend_meta_get_split_state(buf_ctx->get_simple_tensor_container(tensor), tensor, assume_sync);
    }
    return ggml_backend_meta_get_split_state_cold(tensor, assume_sync);
}

static void * ggml_backend_meta_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return (void *) 0x1000000000000000; // FIXME
}

static enum ggml_status ggml_backend_meta_buffer_init_tensor_impl(ggml_backend_meta_simple_tensor_container & stc, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    const size_t n_simple_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(stc, tensor, /*assume_sync =*/ true);
    GGML_ASSERT(ggml_nelements(tensor) == 0 || split_state.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
    GGML_ASSERT(split_state.n_segments <= 16);

    int split_dim = split_state.axis;
    int64_t ne[GGML_MAX_DIMS];
    size_t  nb[GGML_MAX_DIMS];
    for (size_t k = 0; k < GGML_MAX_DIMS; k++) {
        ne[k] = tensor->ne[k];
        nb[k] = tensor->nb[k];
    }

    std::vector<ggml_tensor *> simple_tensors;
    simple_tensors.reserve(n_simple_bufs);
    for (size_t j = 0; j < n_simple_bufs; j++) {
        ggml_context          * simple_ctx = stc.ctx_for_new_tensor(j);
        ggml_backend_buffer_t   simple_buf = buf_ctx->bufs[j].get();

        if (split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
            // TODO: the following assert fails for llama-parallel even though the results are correct:
            // GGML_ASSERT(ggml_is_contiguously_allocated(tensor));
            ne[split_dim] = 0;
            for (size_t s = 0; s < split_state.n_segments; s++) {
                ne[split_dim] += split_state.ne[s*n_simple_bufs + j] * split_state.nr[s];
            }
            for (int i = 0; i < GGML_MAX_DIMS; i++) {
                if (tensor->nb[i] > tensor->nb[split_dim]) {
                    nb[i] = tensor->nb[i] * ne[split_dim]/tensor->ne[split_dim];
                }
            }
        }

        ggml_tensor * t_ij = ggml_new_tensor(simple_ctx, tensor->type, GGML_MAX_DIMS, ne);
        t_ij->op = tensor->op;
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            t_ij->nb[i] = nb[i];
        }
        t_ij->flags = tensor->flags;
        memcpy(t_ij->op_params, tensor->op_params, sizeof(tensor->op_params));
        ggml_set_name(t_ij, tensor->name);
        t_ij->buffer = simple_buf;
        t_ij->view_src = tensor->view_src;
        t_ij->view_offs = tensor->view_offs;
        if (t_ij->view_src != nullptr && ggml_backend_buffer_is_meta(t_ij->view_src->buffer)) {
            t_ij->view_src = ggml_backend_meta_buffer_simple_tensor(tensor->view_src, j);
            if (t_ij->view_offs > 0 && split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
                GGML_ASSERT(tensor->ne[split_dim] != 0);
                const int split_dim_view_src = ggml_backend_meta_get_split_state(tensor->view_src, /*assume_sync =*/ true).axis;
                GGML_ASSERT(split_dim_view_src >= 0 && split_dim_view_src < GGML_MAX_DIMS);

                // The offset can be internal to the data split, in those cases the view offset should not be scaled.
                // If however, the offset is larger than the data split then it needs to be scaled proportionally.
                bool split_internal_offset = t_ij->view_offs <= tensor->view_src->nb[split_dim_view_src];
                for (int i = 0; i < GGML_MAX_DIMS; i++) {
                    const size_t dim_size = tensor->ne[i] * tensor->nb[i];
                    if (tensor->view_offs <= dim_size && dim_size < tensor->nb[split_dim]) {
                        split_internal_offset = true;
                        break;
                    }
                }
                if (!split_internal_offset) {
                    t_ij->view_offs = t_ij->view_offs * ne[split_dim]/tensor->ne[split_dim];
                }
            }
        }
        // TODO: revisit once the graph allocator has been refactored, see https://github.com/ggml-org/llama.cpp/pull/25051#issuecomment-4842873396
        ggml_backend_buffer_t init_buf = simple_buf;
        if (t_ij->view_src != nullptr) {
            t_ij->data = (char *) t_ij->view_src->data + t_ij->view_offs;
            // views inherit the source slice's concrete sub-buffer (issue 22197)
            if (tensor->view_src != nullptr && ggml_backend_buffer_is_meta(tensor->view_src->buffer)
                    && t_ij->view_src->buffer != nullptr) {
                t_ij->buffer = t_ij->view_src->buffer;
                init_buf     = t_ij->view_src->buffer;
            }
        } else if (simple_buf != nullptr) {
            {
                const size_t off = size_t(tensor->data) - size_t(ggml_backend_buffer_get_base(tensor->buffer));
                GGML_UNUSED(off);
            }
            if (ggml_backend_buffer_is_multi_buffer(simple_buf)) {
                GGML_ABORT("multi buffers are not supported by the meta backend");
            }
            t_ij->data = (char *) ggml_backend_buffer_get_base(simple_buf)
                + size_t(tensor->data) - size_t(ggml_backend_buffer_get_base(tensor->buffer));
        }

        if (init_buf) {
            // the backend that owns the buffer will set .extra
            ggml_backend_buffer_init_tensor(init_buf, t_ij);
        } else {
            t_ij->extra = tensor->extra;
        }

        for (int i = 0; i < GGML_MAX_SRC; i++) {
            t_ij->src[i] = tensor->src[i];
            if (tensor->src[i] == tensor) {
                t_ij->src[i] = t_ij;
            } else if (t_ij->src[i] != nullptr && ggml_backend_buffer_is_meta(t_ij->src[i]->buffer)) {
                t_ij->src[i] = ggml_backend_meta_buffer_simple_tensor(tensor->src[i], j);
            }
        }

        simple_tensors.push_back(t_ij);
    }

    // If one of the sources has a zero-sized slice, disable the computation:
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i] == nullptr || !ggml_backend_buffer_is_meta(tensor->src[i]->buffer)) {
            continue;
        }

        const ggml_backend_meta_split_state split_state_src = ggml_backend_meta_get_split_state(tensor->src[i], /*assume_sync =*/ true);
        if (split_state_src.axis < 0 || split_state_src.axis >= GGML_MAX_DIMS) {
            continue;
        }
        for (size_t j = 0; j < n_simple_bufs; j++) {
            int64_t ne_sum = 0;
            for (size_t s = 0; s < split_state_src.n_segments; s++) {
                ne_sum += split_state_src.ne[s*n_simple_bufs + j] * split_state_src.nr[s];
            }
            if (ne_sum == 0) {
                simple_tensors[j]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
            }
        }
    }

    stc.simple_tensors[tensor] = std::move(simple_tensors);

    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_meta_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    buf_ctx->stc_compute_index[g_meta_view_slot] = buf_ctx->stc_compute_index_next[g_meta_view_slot];
    return ggml_backend_meta_buffer_init_tensor_impl(buf_ctx->get_simple_tensor_container(tensor), tensor);
}

static void ggml_backend_meta_buffer_memset_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    const ggml_backend_meta_split_state split_state =
            ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    if (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        std::vector<size_t> simple_offsets(n_bufs, 0);
        if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            const size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            const int64_t row_start = offset / row_stride;
            const int64_t row_count = size   / row_stride;
            GGML_ASSERT(row_start + row_count <= tensor->ne[1]);

            const int64_t blck_size = ggml_blck_size(tensor->type);
            for (size_t s = 0; s < split_state.n_segments; s++) {
                for (size_t r = 0; r < split_state.nr[s]; r++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                        const size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                        for (int64_t row = 0; row < row_count; row++) {
                            ggml_backend_tensor_memset(simple_tensor, value,
                                    simple_offsets[j] + (row_start + row)*simple_tensor->nb[1], nbytes);
                        }
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            return;
        }

        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        const size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        const int64_t row_start = offset / row_stride;
        const int64_t row_count = size   / row_stride;
        GGML_ASSERT(row_start + row_count <= tensor->ne[2]);

        for (size_t s = 0; s < split_state.n_segments; s++) {
            for (size_t r = 0; r < split_state.nr[s]; r++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    const size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    for (int64_t row = 0; row < row_count; row++) {
                        ggml_backend_tensor_memset(simple_tensor, value,
                                simple_offsets[j] + (row_start + row)*simple_tensor->nb[2], nbytes);
                    }
                    simple_offsets[j] += nbytes;
                }
            }
        }
        return;
    }

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        / chunk_size_full;
            const int64_t i_stop  = (offset + size) / chunk_size_full;
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size == 0) {
                    continue;
                }
                for (int64_t i = i_start; i < i_stop; i++) {
                    ggml_backend_tensor_memset(simple_tensor, value, i*chunk_size, chunk_size);
                }
            }
        } break;
        case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
            GGML_ASSERT(value == 0);
            [[fallthrough]];
        }
        case GGML_BACKEND_SPLIT_AXIS_PER_DEVICE: {
            GGML_ABORT("%s: a per-device (relaxed-tensor) tensor cannot be memset as one tensor", __func__);
        }
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ggml_backend_tensor_memset(simple_tensor, value, offset, size);
            }
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    if (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_0 ||
                    split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        // A segmented tensor (a fused tensor: several segments, or one segment repeated `nr[0]` times)
        // is spliced across the devices per segment/device/repeat.  The single-segment axis-0/1 sync
        // setters elsewhere only accept whole rows; here we accept an arbitrary `[offset, offset+size)`
        // byte range so `set_tensor_async` never has to classify a write by `offset % row_stride` (a
        // mid-row range would abort on axis 0 / write a bogus MMQ tail on axis 1).  Mirrors the
        // `write_partial_row` structure of the non-segmented switch below: head partial row, full rows
        // via one strided 2-D copy per segment chunk, then a tail partial row.
        const int  split_axis = (int) split_state.axis;
        const int  dst_dim    = split_axis + 1; // simple_tensor->nb[dst_dim] is the per-device row stride
        const bool axis0      = split_axis == GGML_BACKEND_SPLIT_AXIS_0;
        const int64_t blck_size = ggml_blck_size(tensor->type);
        const size_t row_stride = axis0 ? tensor->nb[1] : tensor->nb[2];
        const size_t ne_rows    = (size_t) (axis0 ? tensor->ne[1] : tensor->ne[2]);
        if (axis0) {
            GGML_ASSERT(tensor->ne[2] == 1);
        }

        auto chunk_bytes = [&](size_t s, size_t j) -> size_t {
            if (axis0) {
                GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                return (size_t) (split_state.ne[s*n_bufs + j] / blck_size) * tensor->nb[0];
            }
            return (size_t) split_state.ne[s*n_bufs + j] * tensor->nb[1];
        };

        // One row's `[col_off, col_off + len)` byte range, dispatched to each segment chunk that overlaps.
        auto write_partial_row = [&](int64_t R, size_t col_off, size_t len) {
            size_t p_src = 0; // source offset within the row's spliced layout
            std::vector<size_t> p_dst(n_bufs, 0); // per-device offset within the row's device slice
            for (size_t s = 0; s < split_state.n_segments; s++) {
                for (size_t r = 0; r < split_state.nr[s]; r++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        const size_t nbytes = chunk_bytes(s, j);
                        const size_t lo = std::max(p_src, col_off);
                        const size_t hi = std::min(p_src + nbytes, col_off + len);
                        if (lo < hi) {
                            ggml_backend_tensor_set(simple_tensor,
                                (const char *) data + ((size_t) R * row_stride + lo - offset),
                                p_dst[j] + (size_t) R * simple_tensor->nb[dst_dim] + (lo - p_src),
                                hi - lo);
                        }
                        p_src += nbytes;
                        p_dst[j] += nbytes;
                    }
                }
            }
        };

        // Whole rows: one strided 2-D copy per segment chunk (the historical fast path).
        auto write_full_rows = [&](int64_t row_start, int64_t row_count) {
            GGML_ASSERT(row_start + row_count <= (int64_t) ne_rows);
            size_t offset_data = 0;
            std::vector<size_t> simple_offsets(n_bufs, 0);
            for (size_t s = 0; s < split_state.n_segments; s++) {
                for (size_t r = 0; r < split_state.nr[s]; r++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        const size_t nbytes = chunk_bytes(s, j);
                        ggml_backend_tensor_set_2d(simple_tensor,
                            (const char *) data + ((size_t) row_start * row_stride + offset_data - offset),
                            simple_offsets[j] + (size_t) row_start * simple_tensor->nb[dst_dim],
                            nbytes, row_count, simple_tensor->nb[dst_dim], row_stride);
                        offset_data       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
        };

        size_t pos = offset;
        size_t remaining = size;
        if (pos % row_stride != 0) {
            const int64_t R = (int64_t) (pos / row_stride);
            const size_t col_off = pos % row_stride;
            const size_t len = std::min(row_stride - col_off, remaining);
            GGML_ASSERT(R < (int64_t) ne_rows);
            write_partial_row(R, col_off, len);
            pos += len;
            remaining -= len;
        }
        const int64_t n_rows = (int64_t) (remaining / row_stride);
        if (n_rows > 0) {
            write_full_rows((int64_t) (pos / row_stride), n_rows);
            const size_t consumed = (size_t) n_rows * row_stride;
            pos += consumed;
            remaining -= consumed;
        }
        if (remaining > 0) {
            GGML_ASSERT(pos % row_stride == 0);
            const int64_t R = (int64_t) (pos / row_stride);
            GGML_ASSERT(R < (int64_t) ne_rows);
            write_partial_row(R, 0, remaining);
        }
        return;
    }

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                const size_t simple_offset = i_start * chunk_size_j;
                ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_j, simple_offset, chunk_size_j, i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ggml_backend_tensor_set(simple_tensor, data, offset, size);
            }
        } break;
        case GGML_BACKEND_SPLIT_AXIS_PER_DEVICE: {
            GGML_ABORT("%s: a per-device (relaxed-tensor) tensor cannot be written as one tensor", __func__);
        }
        case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
            GGML_ASSERT(tensor->type == GGML_TYPE_F32);
            GGML_ASSERT(offset % sizeof(float) == 0);
            GGML_ASSERT(size   % sizeof(float) == 0);
            const size_t n_values = size / sizeof(float);
            size_t n_contributors = 0;
            for (size_t j = 0; j < n_bufs; j++) {
                n_contributors += split_state.ne[j] != 0;
            }
            const bool has_contributor_mask = n_contributors != 0;
            if (!has_contributor_mask) {
                n_contributors = n_bufs;
            }
            std::vector<float> tmp(n_values);
            for (size_t i = 0; i < n_values; i++) {
                tmp[i] = ((const float *) data)[i] / n_contributors;
            }
            std::vector<float> zero;
            if (has_contributor_mask) {
                zero.resize(n_values, 0.0f);
            }
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const float * partial = has_contributor_mask && split_state.ne[j] == 0 ? zero.data() : tmp.data();
                ggml_backend_tensor_set(simple_tensor, partial, offset, size);
            }
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_get_2d_any(ggml_backend_t backend, size_t j, const ggml_tensor * simple_tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    if (backend != nullptr) {
        ggml_backend_tensor_get_2d_async(ggml_backend_meta_simple_backend(backend, j), simple_tensor, data, offset, size,
            n_copies, stride_tensor, stride_data);
    } else {
        ggml_backend_tensor_get_2d(simple_tensor, data, offset, size, n_copies, stride_tensor, stride_data);
    }
}

// `backend` == nullptr: synchronous reads (the buffer interface).  Otherwise the same per-device reads are queued
// on the meta backend's simple backends (`get_tensor_async`), so a caller can batch many reads behind one sync.
// Fold (r96, PR #189): this shared implementation is what lets `get_tensor_async` read an arbitrary row range
// (a context checkpoint's per-sequence recurrent-state cells) instead of asserting `offset == 0` and one segment.
static void ggml_backend_meta_get_tensor_impl(ggml_backend_t backend, ggml_backend_buffer_t buffer, const ggml_tensor * tensor,
        void * data, size_t offset, size_t size) {
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(ggml_is_contiguous(tensor) || split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);

    // A 2-D tensor whose ROWS are split across the devices (axis 1 of a `[n_embd, n_vocab]` tensor -- e.g. a
    // vocabulary-split LM head): the rows of each block are contiguous inside their device, so an arbitrary row
    // range is spliced block by block, walking the segments/repeats in the order the split state defines
    // (seg 0 dev 0 r0, seg 0 dev 1 r0, seg 0 dev 0 r1, ..., seg 1 dev 0 r0, ...).  The chunked path below can
    // only serve a whole-tensor read of this shape, because its chunk is `nb[axis + 1]`, which for a 2-D tensor
    // is the whole tensor (spec-draft-mtp-vocab: the MTP draft head's row subset is copied out of exactly this
    // shape).
    if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_1 && tensor->ne[2] == 1 && tensor->ne[3] == 1) {
        const size_t row_stride = tensor->nb[1];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);

        const int64_t row_start = (int64_t) (offset / row_stride);
        const int64_t row_end   = row_start + (int64_t) (size / row_stride);
        GGML_ASSERT(row_end <= tensor->ne[1]);

        size_t  offset_data = 0;
        int64_t row_base    = 0; // first global row of the current segment
        // Per-device byte offset, advanced exactly as the setter's `simple_offsets` are: a device's chunks
        // (seg, repeat) are packed contiguously in its own slice, so repeat r of a segment does NOT start at
        // `r * rows` of that slice.
        std::vector<size_t> simple_offsets(n_bufs, 0);
        for (uint32_t s = 0; s < split_state.n_segments; s++) {
            int64_t rows_per_rep = 0;
            for (size_t j = 0; j < n_bufs; j++) {
                rows_per_rep += split_state.ne[s*n_bufs + j];
            }
            GGML_ASSERT(rows_per_rep > 0);
            for (uint32_t r = 0; r < split_state.nr[s]; r++) {
                int64_t base = row_base + (int64_t) r * rows_per_rep;
                for (size_t j = 0; j < n_bufs; j++) {
                    const int64_t rows_j = split_state.ne[s*n_bufs + j];
                    const size_t  chunk  = (size_t) rows_j * row_stride;
                    const int64_t r0 = std::max(row_start, base);
                    const int64_t r1 = std::min(row_end,   base + rows_j);
                    if (r1 > r0) {
                        const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        const size_t nbytes = (size_t) (r1 - r0) * row_stride;
                        ggml_backend_meta_get_2d_any(backend, j, simple_tensor, (char *) data + offset_data,
                            simple_offsets[j] + (size_t) (r0 - base) * simple_tensor->nb[1], nbytes, 1, nbytes, nbytes);
                        offset_data += nbytes;
                    }
                    simple_offsets[j] += chunk;
                    base += rows_j;
                }
            }
            row_base += rows_per_rep * split_state.nr[s];
        }
        GGML_ASSERT(row_base == tensor->ne[1]);
        GGML_ASSERT(offset_data == size);
        return;
    }

    if (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(split_state.nr[0] != 0);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t offset_data = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            const size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            const int64_t row_start = offset / row_stride;
            const int64_t row_count = size   / row_stride;
            GGML_ASSERT(row_start + row_count <= tensor->ne[1]);

            const int64_t blck_size = ggml_blck_size(tensor->type);
            for (size_t s = 0; s < split_state.n_segments; s++) {
                for (size_t r = 0; r < split_state.nr[s]; r++) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                        GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                        const size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                        ggml_backend_meta_get_2d_any(backend, j, simple_tensor, (char *) data + offset_data,
                            simple_offsets[j] + row_start * simple_tensor->nb[1], nbytes,
                            row_count, simple_tensor->nb[1], tensor->nb[1]);
                        offset_data       += nbytes;
                        simple_offsets[j] += nbytes;
                    }
                }
            }
            GGML_ASSERT(offset_data*row_count == size);
            return;
        }
        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        const size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        const int64_t row_start = offset / row_stride;
        const int64_t row_count = size   / row_stride;
        GGML_ASSERT(row_start + row_count <= tensor->ne[2]);

        for (size_t s = 0; s < split_state.n_segments; s++) {
            for (size_t r = 0; r < split_state.nr[s]; r++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    const size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                    ggml_backend_meta_get_2d_any(backend, j, simple_tensor, (char *) data + offset_data,
                        simple_offsets[j] + row_start * simple_tensor->nb[2], nbytes,
                        row_count, simple_tensor->nb[2], tensor->nb[2]);
                    offset_data       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
        }
        GGML_ASSERT(offset_data*row_count == size);
        return;
    }

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_bufs; j++){
                const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                if (chunk_size_j == 0) {
                    continue;
                }
                const size_t simple_offset = i_start * chunk_size_j;
                ggml_backend_meta_get_2d_any(backend, j, simple_tensor, (char *) data + offset_j, simple_offset, chunk_size_j, i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, 0);
            if (backend != nullptr) {
                ggml_backend_tensor_get_async(ggml_backend_meta_simple_backend(backend, 0), simple_tensor, data, offset, size);
            } else {
                ggml_backend_tensor_get(simple_tensor, data, offset, size);
            }
        } break;
        case GGML_BACKEND_SPLIT_AXIS_PER_DEVICE: {
            // There is no single value to return: every backend holds its own.
            GGML_ABORT("%s: a per-device (relaxed-tensor) tensor cannot be read back as one tensor", __func__);
        }
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_meta_get_tensor_impl(nullptr, buffer, tensor, data, offset, size);
}

static void ggml_backend_meta_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    const size_t n_buffers = ggml_backend_meta_buffer_n_bufs(buffer);
    for (size_t i = 0; i < n_buffers; i++) {
        ggml_backend_buffer_clear(ggml_backend_meta_buffer_simple_buffer(buffer, i), value);
    }
}

static void ggml_backend_meta_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    for (size_t i = 0; i < buf_ctx->bufs.size(); i++) {
        ggml_backend_buffer_reset(ggml_backend_meta_buffer_simple_buffer(buffer, i));
    }
}

static const ggml_backend_buffer_i ggml_backend_meta_buffer_iface = {
    /* .free_buffer     = */ ggml_backend_meta_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_meta_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_meta_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_meta_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_meta_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_meta_buffer_get_tensor,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_meta_buffer_clear,
    /* .reset           = */ ggml_backend_meta_buffer_reset,
};

bool ggml_backend_buffer_is_meta(ggml_backend_buffer_t buf) {
    return buf != nullptr && buf->iface.free_buffer == ggml_backend_meta_buffer_iface.free_buffer;
}

// drop the scratch container's views (and any context it grew), so the caller's next chunk starts clean
void ggml_backend_meta_reset_scratch_views(ggml_backend_buffer_t buffer) {
    if (!ggml_backend_buffer_is_meta(buffer)) {
        return;
    }
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    buf_ctx->stc_scratch.reset();
    buf_ctx->stc_scratch.simple_tensors.clear();
}

void ggml_backend_meta_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    for (size_t i = 0; i < buf_ctx->bufs.size(); i++) {
        if (buf_ctx->bufs[i]) {
            ggml_backend_buffer_set_usage(buf_ctx->bufs[i].get(), usage);
        }
    }
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer_usage(ggml_backend_buffer_type_t buft, size_t size, enum ggml_backend_buffer_usage usage) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    const ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024*ggml_tensor_overhead(), // FIXME
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_backend_meta_simple_tensor_container stc_static;
    ggml_backend_meta_simple_tensor_container stc_compute_0(params, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params, n_simple_bufts);

    size_t max_size = 0;
    std::vector<ggml_backend_buffer_t> bufs;
    bufs.reserve(n_simple_bufts);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        // pass the usage down so a simple buffer type can pick a different allocator for the COMPUTE
        // buffer (the RDNA/ROCm VMM compute pool); it defaults to the usage-agnostic allocation
        ggml_backend_buffer_type_t simple_buft = ggml_backend_meta_buft_simple_buft(buft, i);
        ggml_backend_buffer_t simple_buf = ggml_backend_buft_alloc_buffer_usage(simple_buft, size, usage);
        // [TAG_META_ALLOC_FAIL] a constituent allocation that fails is *not* fatal here: the caller
        // (`ggml_vbuffer_alloc` for a compute buffer, `ggml_backend_alloc_ctx_tensors_from_buft` for a
        // weight / KV buffer) handles a NULL and reports it, so a context that does not fit fails
        // loudly and cleanly instead of aborting with a core dump (issue #170).
        if (simple_buf == nullptr) {
            GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu (%zu of %zu sub-buffers allocated)\n",
                __func__, ggml_backend_buft_name(simple_buft), size, i, n_simple_bufts);
            for (ggml_backend_buffer_t b : bufs) {
                ggml_backend_buffer_free(b);
            }
            return nullptr;
        }
        bufs.push_back(simple_buf);
        max_size = std::max(max_size, ggml_backend_buffer_get_size(simple_buf));
    }
    ggml_backend_meta_buffer_context * buf_ctx = new ggml_backend_meta_buffer_context(stc_static, stc_compute_0, stc_compute_1, params, bufs);

    return ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, buf_ctx, max_size);
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    return ggml_backend_meta_buffer_type_alloc_buffer_usage(buft, size, GGML_BACKEND_BUFFER_USAGE_ANY);
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer_n(ggml_backend_buffer_type_t buft, ggml_tensor ** tensors, int n_tensors) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    // Views of the static tensors that are created between graph evals are stored in the compute
    // containers. The number of such views is proportional to the number of tensors in the graph
    // that share the buffer, which for hybrid recurrent models with n_rs_seq snapshotting can be
    // much larger than 16 per static tensor (e.g. Qwen35: ~2*(n_rs_seq+1) views per recurrent
    // layer are created for the conv-state snapshot copies). Size the headroom accordingly.
    //
    // The container grows past this budget when a generation registers more (see
    // ggml_backend_meta_simple_tensor_container), so this is a first-context size, not a limit.
    // GGML_META_COMPUTE_HEADROOM overrides it: a diagnostic/testing knob (a smaller value only makes
    // the container grow sooner, it never changes an output).
    static const size_t compute_headroom = [] {
        const char * e = getenv("GGML_META_COMPUTE_HEADROOM");
        const long v = e != nullptr ? atol(e) : 0;
        return v > 0 ? (size_t) v : (size_t) 128;
    }();
    const ggml_init_params params_static = {
        /*.mem_size   =*/ n_tensors * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    const ggml_init_params params_compute = {
        /*.mem_size   =*/ compute_headroom * n_tensors * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_backend_meta_simple_tensor_container stc_static   (params_static,  n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_0(params_compute, n_simple_bufts);
    ggml_backend_meta_simple_tensor_container stc_compute_1(params_compute, n_simple_bufts);

    std::vector<ggml_backend_buffer_t> bufs(n_simple_bufts, nullptr);
    ggml_backend_meta_buffer_context * meta_buf_ctx = new ggml_backend_meta_buffer_context(stc_static, stc_compute_0, stc_compute_1, params_compute, bufs);

    ggml_backend_buffer_t meta_buf = ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, meta_buf_ctx, 0);
    for (int i = 0; i < n_tensors; i++) {
        ggml_tensor * t = tensors[i];
        t->buffer = meta_buf;
        ggml_backend_meta_buffer_init_tensor_impl(meta_buf_ctx->stc_static, t);
        t->data = (void *) 0x2000000000000000; // FIXME
    }
    for (size_t i = 0; i < n_simple_bufts; i++) {
        ggml_context * ctx = meta_buf_ctx->stc_static.ctx_first(i);
        ggml_backend_buffer_type_t simple_buft = ggml_backend_meta_buft_simple_buft(buft, i);

        // If a ggml_context only has zero-sized tensors, ggml_backend_alloc_ctx_tensors_from_buft returns NULL.
        // For those edge cases, allocate a dummy buffer instead.
        bool any_nonzero_slice = false;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            if (ggml_nelements(t) != 0) {
                any_nonzero_slice = true;
                break;
            }
        }
        if (any_nonzero_slice) {
            meta_buf_ctx->bufs[i].reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx, simple_buft));
        } else {
            meta_buf_ctx->bufs[i].reset(ggml_backend_buft_alloc_buffer(simple_buft, 0));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                t->buffer = meta_buf_ctx->bufs[i].get();
            }
        }
        // [TAG_META_ALLOC_FAIL] a constituent allocation that fails (a KV cache, or a set of weights,
        // that does not fit) is reported, not fatal: the caller turns the NULL into a clean
        // "failed to allocate buffer" error instead of aborting (issue #170).  Tear down what was
        // already built -- the tensors must not keep pointing at the meta buffer we are about to free,
        // and freeing it releases the sub-buffers allocated so far.
        if (meta_buf_ctx->bufs[i] == nullptr) {
            GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu (%zu of %zu sub-buffers allocated)\n",
                __func__, ggml_backend_buft_name(simple_buft),
                ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, simple_buft), i, n_simple_bufts);
            for (int t = 0; t < n_tensors; t++) {
                tensors[t]->buffer = nullptr;
                tensors[t]->data   = nullptr;
            }
            ggml_backend_buffer_free(meta_buf);
            return nullptr;
        }
        meta_buf->size = std::max(meta_buf->size, ggml_backend_buffer_get_size(meta_buf_ctx->bufs[i].get()));
    }
    return meta_buf;
}

//
// meta backend
//

static ggml_guid_t ggml_backend_meta_guid() {
    static ggml_guid guid = {0xf1, 0x0e, 0x34, 0xcf, 0x9c, 0x6f, 0x43, 0xcb, 0x96, 0x92, 0xbe, 0x8e, 0xbb, 0x71, 0x3f, 0xda};
    return &guid;
}

struct ggml_backend_meta_context {
    struct cgraph_config {
        ggml_cgraph * cgraph_main = nullptr;
        int           offset      = 0; // Node offset vs. original graph

        std::vector<ggml_cgraph *> cgraphs_aux;
    };
    struct backend_config {
        ggml_backend_t backend;

        std::vector<cgraph_config>           cgraphs;
        std::vector<ggml_tensor *>           nodes;
        std::vector<ggml_backend_buffer_ptr> bufs;

        backend_config(ggml_backend_t backend, const size_t n_reduce_steps) : backend(backend) {
            bufs.resize(n_reduce_steps);
        }
    };
    std::string                 name;
    std::vector<backend_config> backend_configs;
    ggml_context_ptr            ctx;
    std::vector<ggml_cgraph *>  cgraphs_aux;
    std::vector<ggml_tensor *>  nodes_aux;
    // ALL_GATHER collectives: the node to run after each subgraph (or -1) and the scratch destination
    // views the copies write through, one per (source device, destination device) pair
    std::vector<int>            gather_nodes;
    std::vector<ggml_tensor *>  gather_views;
    size_t                      n_reduce_steps;
    int                         max_nnodes    = 0;
    size_t                      max_tmp_size  = 0;
    size_t                      max_subgraphs = 0;
    size_t                      n_subgraphs   = 0;
    uint64_t                    uid           = 0;
    // the scheduler's eval-callback flag, stashed by graph_optimize and replayed into the child
    // backends' per-device marking pass (graph_compute) so a callback-split graph stands the F32
    // elision down exactly as the single-backend path does
    bool                        opt_has_eval_callback = false;

    // op-offload H2D staging for the tensor split (issue #50 WIP).  The scheduler's own staging path
    // assumes one destination and an event on the split backend's device; under -sm tensor one logical
    // upload is spliced across N devices and the split's consumers read the per-device "simple"
    // tensors, so the meta backend owns the lifecycle: `stage_input` lands each device's chunk in that
    // device's own ring on its auxiliary copy stream (overlapping the previous split's compute), and
    // `graph_compute` waits for those copies, points the simple tensors at the slots and frees them
    // once the child graphs are launched.
    struct stage_chunk {
        ggml_backend_t backend = nullptr; // the simple backend that owns the slot
        size_t         device  = 0;
        void *         slot    = nullptr; // that backend's ring slot for this upload
        size_t         size    = 0;       // bytes per block (gather) or of the whole copy
        size_t         src     = 0;       // offset into the host source (0 for a MIRRORED split)
        size_t         stride  = 0;       // host-source stride between blocks (gather only)
        size_t         n_copies= 0;       // number of blocks to gather (0 = plain contiguous upload)
        void *         orig    = nullptr; // the simple tensor's data, restored after the launch
    };
    struct stage_entry {
        ggml_tensor * tensor = nullptr;   // the split input (a tensor of this backend)
        int            slot   = 0;
        std::vector<stage_chunk> chunks;  // one per device that holds part of the upload
    };
    std::vector<stage_entry> stage_pending; // staged by `stage_input`, drained by `graph_compute`
    int   stage_slot_next = 0;
    int   stage_n_slots   = 8;              // GGML_SCHED_STAGE_SLOTS, clamped to the ring's 16
    std::vector<std::vector<ggml_backend_event_t>> stage_done_ev; // [device][slot], upload complete
    std::vector<std::vector<ggml_backend_event_t>> stage_free_ev; // [device][slot], slot reusable

    void *                               comm_ctx       = nullptr;
    ggml_backend_comm_allreduce_tensor_t comm_allreduce = nullptr;

    ggml_backend_meta_context(ggml_backend_dev_t meta_dev, const char * params) {
        const size_t n_devs = ggml_backend_meta_dev_n_devs(meta_dev);
        n_reduce_steps = std::ceil(std::log2(n_devs));
        name = "Meta(";
        std::vector<ggml_backend_t> simple_backends;
        backend_configs.reserve(n_devs);
        simple_backends.reserve(n_devs);
        for (size_t i = 0; i < n_devs; i++) {
            ggml_backend_dev_t simple_dev = ggml_backend_meta_dev_simple_dev(meta_dev, i);
            if (i > 0) {
                name += ",";
            }
            name += ggml_backend_dev_name(simple_dev);
            simple_backends.push_back(ggml_backend_dev_init(simple_dev, params));
            backend_configs.emplace_back(simple_backends.back(), n_reduce_steps);
        }
        name += ")";

        // op-offload H2D staging (issue #50 WIP): the slot count mirrors the single-device ring, so
        // GGML_SCHED_STAGE_SLOTS means the same depth on both paths (the ring itself lives in each
        // simple backend's context and is bounded by its own GGML_SCHED_STAGE_MAX_MB).
        {
            const char * slots_env = GGML_ENV_STR("GGML_SCHED_STAGE_SLOTS");
            stage_n_slots = slots_env != nullptr ? atoi(slots_env) : 8;
            if (stage_n_slots < 1) {
                stage_n_slots = 1;
            }
            if (stage_n_slots > 16) {
                stage_n_slots = 16;
            }
            stage_done_ev.resize(n_devs);
            stage_free_ev.resize(n_devs);
            for (size_t j = 0; j < n_devs; j++) {
                stage_done_ev[j].resize(stage_n_slots, nullptr);
                stage_free_ev[j].resize(stage_n_slots, nullptr);
            }
        }

        if (n_devs > 1) {
            ggml_backend_comm_init_t comm_init = (ggml_backend_comm_init_t) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_init");
            if (comm_init != nullptr) {
                comm_ctx = comm_init(simple_backends.data(), simple_backends.size());
            }
        }
        if (comm_ctx != nullptr) {
            comm_allreduce = (ggml_backend_comm_allreduce_tensor_t)
                ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(
                    ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_allreduce_tensor");
            GGML_ASSERT(comm_allreduce != nullptr);
        }
    }

    ~ggml_backend_meta_context() {
        if (comm_ctx != nullptr) {
            ggml_backend_comm_free_t comm_free = (ggml_backend_comm_free_t) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_configs[0].backend)), "ggml_backend_comm_free");
            GGML_ASSERT(comm_free != nullptr);
            comm_free(comm_ctx);
        }
        for (size_t j = 0; j < stage_done_ev.size(); j++) {
            for (size_t s = 0; s < stage_done_ev[j].size(); s++) {
                ggml_backend_event_free(stage_done_ev[j][s]);
                ggml_backend_event_free(stage_free_ev[j][s]);
            }
        }
        for (auto & bc : backend_configs) {
            ggml_backend_free(bc.backend);
        }
    }
};

static const char * ggml_backend_meta_get_name(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) backend->context;
    return backend_ctx->name.c_str();
}

static void ggml_backend_meta_free(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    delete backend_ctx;
    delete backend;
}

static void ggml_backend_meta_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(ggml_is_contiguous(tensor));
    if (size == 0) {
        // A zero-element tensor has no meaningful meta split state (UNKNOWN); nothing to upload.
        return;
    }

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    if (split_state.n_segments != 1 || split_state.nr[0] != 1 || split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
        // A segmented tensor (a fused tensor: several segments, or a segment repeated `nr[0]` times) is
        // spliced across the devices by the synchronous setter, which accepts an arbitrary byte range
        // (head partial row + full rows + tail partial row).  Do NOT classify by `offset % row_stride`:
        // a mid-row chunk would trip the axis-0 `size % row_stride == 0` assert, and on axis 1 the sync
        // splitter would treat the remainder as a <=512-byte MMQ tail, so any other mid-row caller would
        // corrupt the tensor.  The reverted `size % row_stride <= 512` attempt tried to divert such a
        // range into a whole-tensor accumulator; a partial per-op upload never reached `total`, stranding
        // its bytes (the rear-placement hang).
        //
        // TODO #77 (fix-gemma-4 campaign): the HOT caller of this arm is the scheduler's per-op expert
        // upload (`copy_experts`), a ROW-ALIGNED range of a fused table (gemma4's `ffn_gate_up_exps`: two
        // segments, each split across the devices).  Handing it to the synchronous setter made every
        // 8-token run pay 3.8 s of host-blocking copies.  A row-aligned range is served here with the
        // same head/full/tail splice asynchronously (per device, on that device's stream, so the H2D
        // overlaps the compute that follows).  A MID-ROW range stays on the synchronous setter: that is
        // the loader's chunked upload (load-time, and its pinned staging ring may be reused as soon as
        // the call returns).  The async arm therefore requires (a) `offset` on a row boundary and (b) a
        // caller whose source outlives the queued copies -- which the scheduler guarantees: it uploads
        // from the immutable pinned host master.
        // Kill switch (`GGML_META_SEG_ASYNC=0`) for the async arm: it exists so the synchronous splice can
        // be the A/B reference and the bisect arm for a decode-path regression.
        static const bool seg_async_on = [] {
            const char * e = getenv("GGML_META_SEG_ASYNC");
            return e == nullptr || atoi(e) != 0;
        }();
        const int   split_axis_s = (int) split_state.axis;
        const bool  seg_axis0    = split_axis_s == GGML_BACKEND_SPLIT_AXIS_0;
        const int   seg_dst_dim  = split_axis_s + 1;
        const size_t seg_row_stride = seg_axis0 ? tensor->nb[1] : tensor->nb[2];
        if (seg_async_on && split_axis_s >= GGML_BACKEND_SPLIT_AXIS_0 &&
            split_axis_s <= GGML_BACKEND_SPLIT_AXIS_1 &&
            seg_row_stride != 0 && (offset % seg_row_stride) == 0) {
            const int64_t seg_blck     = ggml_blck_size(tensor->type);
            const size_t  seg_n_rows   = (size_t) (seg_axis0 ? tensor->ne[1] : tensor->ne[2]);

            auto seg_chunk_bytes = [&](size_t s, size_t j) -> size_t {
                if (seg_axis0) {
                    GGML_ASSERT(split_state.ne[s*n_backends + j] % seg_blck == 0);
                    return (size_t) (split_state.ne[s*n_backends + j] / seg_blck) * tensor->nb[0];
                }
                return (size_t) split_state.ne[s*n_backends + j] * tensor->nb[1];
            };

            size_t pos       = offset;
            size_t remaining = size;

            // Full rows: one strided 2-D async dispatch per (segment, repeat, device), exactly the
            // offsets the synchronous setter's `write_full_rows` uses.
            const int64_t seg_row0 = (int64_t) (pos / seg_row_stride);
            const int64_t seg_rows = (int64_t) (remaining / seg_row_stride);
            if (seg_rows > 0) {
                GGML_ASSERT(seg_row0 + seg_rows <= (int64_t) seg_n_rows);
                size_t offset_data = 0;
                std::vector<size_t> dst_off(n_backends, 0);
                for (size_t s = 0; s < split_state.n_segments; s++) {
                    for (size_t r = 0; r < split_state.nr[s]; r++) {
                        for (size_t j = 0; j < n_backends; j++) {
                            ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                            ggml_tensor *  simple_tensor  = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                            const size_t nbytes = seg_chunk_bytes(s, j);
                            ggml_backend_tensor_set_2d_async(simple_backend, simple_tensor,
                                (const char *) data + ((size_t) seg_row0 * seg_row_stride + offset_data - offset),
                                dst_off[j] + (size_t) seg_row0 * simple_tensor->nb[seg_dst_dim],
                                nbytes, (size_t) seg_rows, simple_tensor->nb[seg_dst_dim], seg_row_stride);
                            offset_data += nbytes;
                            dst_off[j]  += nbytes;
                        }
                    }
                }
                const size_t consumed = (size_t) seg_rows * seg_row_stride;
                pos       += consumed;
                remaining -= consumed;
            }

            // Tail: the remaining bytes at column 0 of the next row.  This is the `copy_experts` MMQ
            // guard pad (`min(expert_size, 512)`), not real data, so a 1-D async copy per device chunk
            // is sufficient and keeps its exact flat placement.
            if (remaining > 0) {
                GGML_ASSERT((pos % seg_row_stride) == 0);
                const int64_t R = (int64_t) (pos / seg_row_stride);
                GGML_ASSERT(R < (int64_t) seg_n_rows);
                size_t p_src = 0;
                std::vector<size_t> p_dst(n_backends, 0);
                for (size_t s = 0; s < split_state.n_segments; s++) {
                    for (size_t r = 0; r < split_state.nr[s]; r++) {
                        for (size_t j = 0; j < n_backends; j++) {
                            ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                            ggml_tensor *  simple_tensor  = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                            const size_t nbytes = seg_chunk_bytes(s, j);
                            const size_t lo = std::max(p_src, (size_t) 0);
                            const size_t hi = std::min(p_src + nbytes, remaining);
                            if (lo < hi) {
                                ggml_backend_tensor_set_async(simple_backend, simple_tensor,
                                    (const char *) data + ((size_t) R * seg_row_stride + lo - offset),
                                    p_dst[j] + (size_t) R * simple_tensor->nb[seg_dst_dim] + (lo - p_src),
                                    hi - lo);
                            }
                            p_src += nbytes;
                            p_dst[j] += nbytes;
                        }
                    }
                }
            }
            return;
        }

        ggml_backend_tensor_set(tensor, data, offset, size);
        return;
    }

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            // UPSTREAM PR #22466 (adapted): the loader's async-upload chunks may start or end mid-row.
            // `write_partial_row` dispatches one row's column range [col_off, col_off + len) per device.
            // It also subsumes the delivery's TODO #43 tail fix: each device gets its own
            // `min(rem, chunk_size_j)` guard, so devices 1..N-1 are not left with uninitialised
            // speculative-tail bytes (the `-sm tensor` + host-expert `////` corruption).
            auto write_partial_row = [&](int64_t R, size_t col_off, size_t len, const char * src) {
                size_t col_start_j = 0;
                for (size_t j = 0; j < n_backends; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                    const size_t col_end_j    = col_start_j + chunk_size_j;
                    const size_t s = std::max(col_start_j, col_off);
                    const size_t e = std::min(col_end_j,   col_off + len);
                    if (s < e) {
                        ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                        const size_t dst_off  = (size_t) R * chunk_size_j + (s - col_start_j);
                        const size_t src_off  = s - col_off;
                        const size_t copy_len = e - s;
                        ggml_backend_tensor_set_async(simple_backend, simple_tensor, src + src_off, dst_off, copy_len);
                    }
                    col_start_j = col_end_j;
                }
            };

            const char * src       = (const char *) data;
            size_t       pos       = offset;
            size_t       remaining = size;

            // Partial head: bytes from the current position up to the next row boundary.
            if ((pos % chunk_size_full) != 0) {
                const int64_t R       = (int64_t) (pos / chunk_size_full);
                const size_t  col_off = pos % chunk_size_full;
                const size_t  len     = std::min(chunk_size_full - col_off, remaining);
                write_partial_row(R, col_off, len, src);
                src       += len;
                pos       += len;
                remaining -= len;
            }

            // Aligned middle: full rows.  One strided 2-D async dispatch per device.
            const int64_t i_first = (int64_t) (pos / chunk_size_full);
            const int64_t n_rows  = (int64_t) (remaining / chunk_size_full);
            if (n_rows > 0) {
                size_t offset_j = 0;
                for (size_t j = 0; j < n_backends; j++) {
                    ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                    ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                    if (chunk_size_j == 0) {
                        continue;
                    }
                    ggml_backend_tensor_set_2d_async(simple_backend, simple_tensor, src + offset_j,
                        i_first * chunk_size_j, chunk_size_j,
                        n_rows, chunk_size_j, chunk_size_full);
                    offset_j += chunk_size_j;
                }
                GGML_ASSERT(offset_j == chunk_size_full);
                const size_t consumed = (size_t) n_rows * chunk_size_full;
                src       += consumed;
                pos       += consumed;
                remaining -= consumed;
            }

            // Partial tail: remaining bytes starting at column 0 of the current row.
            if (remaining > 0) {
                GGML_ASSERT((pos % chunk_size_full) == 0);
                const int64_t R = (int64_t) (pos / chunk_size_full);
                write_partial_row(R, 0, remaining, src);
            }
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // A mirrored tensor holds the same bytes on every device, so no chunk arithmetic is needed
            // and an arbitrary byte range can be forwarded as-is.  The scheduler's used-expert pruning
            // uploads just the experts a ubatch uses -- a range at a non-zero offset that need not be a
            // whole-chunk multiple -- which is what makes op-offload under -sm tensor work at all (see
            // ggml_backend_meta_device_offload_op).
            for (size_t j = 0; j < n_backends; j++) {
                ggml_backend_tensor_set_async(
                    ggml_backend_meta_simple_backend(backend, j), ggml_backend_meta_buffer_simple_tensor(tensor, j), data, offset, size);
            }
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    // the same split arithmetic as the synchronous read, with every per-device read queued on its simple backend
    // (an arbitrary row range, e.g. one sequence's recurrent-state cells for a context checkpoint).
    // Fold (r96, PR #189): calls that worked before (offset 0, one segment) issue the same copies as before.
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(ggml_backend_meta_buffer_n_bufs(buffer) == ggml_backend_meta_n_backends(backend));
    ggml_backend_meta_get_tensor_impl(backend, buffer, tensor, data, offset, size);
}

static void ggml_backend_meta_synchronize(ggml_backend_t backend) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    for (size_t i = 0; i < n_backends; i++) {
        ggml_backend_synchronize(ggml_backend_meta_simple_backend(backend, i));
    }
}

// op-offload H2D staging for the tensor split (issue #50 WIP).  The scheduler stages a whole host
// weight so its upload overlaps the previous split's compute; under -sm tensor that upload is spliced
// across N devices and the split's consumers read the per-device "simple" tensors rather than the
// split tensor's `data`, so neither the scheduler's single slot nor its data-pointer redirect can
// reach the op.  This backend therefore owns the whole lifecycle: each device's chunk goes into that
// device's own ring on the device's auxiliary copy stream, `graph_compute` waits for the copies on
// each device's main stream and points the simple tensors at the slots, and the slots are freed once
// the child graphs have been launched (see `ggml_backend_meta_stage_guard`).
static ggml_backend_event_t ggml_backend_meta_stage_ev(ggml_backend_meta_context * backend_ctx,
        std::vector<std::vector<ggml_backend_event_t>> & evs, size_t device, int slot) {
    if (evs[device][slot] == nullptr) {
        evs[device][slot] = ggml_backend_event_new(
            ggml_backend_get_device(backend_ctx->backend_configs[device].backend));
    }
    return evs[device][slot];
}

static bool ggml_backend_meta_stage_input(ggml_backend_t backend, ggml_tensor * input, ggml_tensor * input_cpy) {
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    const size_t n_backends = ggml_backend_meta_n_backends(backend);

    if (!ggml_backend_buffer_is_meta(input_cpy->buffer) || !ggml_is_contiguous(input_cpy)) {
        return false;
    }

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(input_cpy, /*assume_sync =*/ false);
    if (split_state.n_segments != 1 || split_state.nr[0] != 1) {
        return false;
    }
    const bool mirrored = split_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED;
    if (!mirrored && (split_state.axis < 0 || split_state.axis >= GGML_MAX_DIMS)) {
        return false;
    }

    // MIRRORED: a single contiguous copy per device covers the whole tensor.
    // Split op-offload (wip/tensor-split-expert-split): the tensor is uploaded whole and each device
    // owns a *strided* slice (every `chunk_size_full`, take `chunk_size_j`), so it is gathered on the
    // host into the ring slot -- which keeps the transfer on the copy stream (queueable) instead of the
    // in-order spliced path, whose pre-upload host synchronize exposed it (§25).
    const size_t size = ggml_nbytes(input_cpy);
    const size_t chunk_size_full = mirrored ? size : input_cpy->nb[split_state.axis + 1];
    if (chunk_size_full == 0) {
        return false;
    }
    const bool gather = !mirrored && chunk_size_full != size;
    if (!mirrored && !gather && chunk_size_full != size) {
        return false;
    }
    if (gather && (size % chunk_size_full) != 0) {
        return false;
    }
    const size_t n_chunks = gather ? size / chunk_size_full : 0;

    const int slot = backend_ctx->stage_slot_next;
    backend_ctx->stage_slot_next = (backend_ctx->stage_slot_next + 1) % backend_ctx->stage_n_slots;

    // Reserve every device's ring slot before anything is uploaded: a ring that cannot hold its chunk
    // aborts the whole attempt, because staging only part of an upload would leave the rest on the
    // pruned path (a full device synchronize plus an ids readback), which measured worse than either
    // path on its own.
    ggml_backend_meta_context::stage_entry entry;
    entry.tensor = input_cpy;
    entry.slot   = slot;
    size_t offset_j = 0;
    for (size_t j = 0; j < n_backends; j++) {
        ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
        ggml_tensor *  simple_tensor  = ggml_backend_meta_buffer_simple_tensor(input_cpy, j);
        if (simple_tensor == nullptr ||
            simple_backend->iface.stage_buffer == nullptr ||
            simple_backend->iface.stage_upload == nullptr ||
            simple_backend->iface.stage_wait == nullptr ||
            (gather && simple_backend->iface.stage_gather == nullptr)) {
            return false;
        }
        const size_t chunk_size_j = mirrored ? ggml_nbytes(simple_tensor) : simple_tensor->nb[split_state.axis + 1];
        if (chunk_size_j == 0) {
            continue;
        }
        // The ring slot request comes from the shared split-size rule, so the slab hole (sized from the
        // same rule, `llama_model::max_host_expert_slice_bytes`) always holds it.
        const size_t copy_bytes = ggml_backend_meta_split_dev_bytes(input_cpy, &split_state, n_backends, j);
        GGML_ASSERT(copy_bytes == (gather ? n_chunks * chunk_size_j : chunk_size_j));
        void * slot_ptr = simple_backend->iface.stage_buffer(simple_backend, slot, copy_bytes);
        if (slot_ptr == nullptr) {
            return false;
        }
        ggml_backend_meta_context::stage_chunk chunk;
        chunk.backend  = simple_backend;
        chunk.device   = j;
        chunk.slot     = slot_ptr;
        chunk.size     = chunk_size_j;
        chunk.src      = mirrored ? 0 : offset_j;
        chunk.stride   = gather ? chunk_size_full : 0;
        chunk.n_copies = gather ? n_chunks : 0;
        entry.chunks.push_back(chunk);
        if (!mirrored) {
            offset_j += chunk_size_j;
        }
    }
    if (entry.chunks.empty() || (!mirrored && offset_j != chunk_size_full)) {
        return false;
    }
    for (auto & chunk : entry.chunks) {
        if (ggml_backend_meta_stage_ev(backend_ctx, backend_ctx->stage_done_ev, chunk.device, slot) == nullptr ||
            ggml_backend_meta_stage_ev(backend_ctx, backend_ctx->stage_free_ev, chunk.device, slot) == nullptr) {
            return false;
        }
    }

    // Issue the copies on each device's auxiliary copy stream: they are independent of the main
    // streams, so they run concurrently with the previous split's compute.
    //
    // WIP r42 (stage-1 item 2): is the source weight pinned (a GPU host buffer) rather than the
    // pageable model mmap?  A pinned source lets `stage_gather` copy the strided slice directly with a
    // 2-D H2D instead of H2D-ing the whole contiguous range into scratch and compacting on device.
    // A device host buft carries its GPU device; the CPU / mmap bufts carry the CPU device.
    bool src_pinned = false;
    if (input->buffer != nullptr) {
        ggml_backend_dev_t src_dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(input->buffer));
        src_pinned = src_dev != nullptr && ggml_backend_dev_type(src_dev) != GGML_BACKEND_DEVICE_TYPE_CPU;
    }
    for (auto & chunk : entry.chunks) {
        chunk.backend->iface.stage_wait(chunk.backend, backend_ctx->stage_free_ev[chunk.device][slot]);
        if (chunk.n_copies > 0) {
            if (!chunk.backend->iface.stage_gather(chunk.backend, slot, (const char *) input->data, chunk.src,
                                              chunk.size, chunk.stride, chunk.n_copies,
                                              backend_ctx->stage_done_ev[chunk.device][slot], src_pinned)) {
                // A device could not stage its slice (e.g. `stage_gather`'s whole-range scratch is
                // bounded at 256 MiB and this range exceeds it).  Abort the whole attempt: the caller
                // then falls through to the normal copy path.  Reporting success here would leave the
                // ring slot unfilled and its done-event unrecorded, so the consuming kernels would read
                // whatever the slot last held -- silently correct only while the weights are static.
                return false;
            }
        } else {
            chunk.backend->iface.stage_upload(chunk.backend, chunk.slot, (const char *) input->data + chunk.src,
                                              chunk.size, backend_ctx->stage_done_ev[chunk.device][slot]);
        }
    }
    backend_ctx->stage_pending.push_back(std::move(entry));
    return true;
}

static float ggml_backend_meta_stage_h2d_gbps(ggml_backend_t backend) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    float bw = 0.0f;
    for (size_t j = 0; j < n_backends; j++) {
        ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
        if (simple_backend->iface.stage_h2d_gbps == nullptr) {
            return 0.0f;
        }
        const float b = simple_backend->iface.stage_h2d_gbps(simple_backend);
        if (b <= 0.0f) {
            return 0.0f;
        }
        // the slowest link gates: every staged chunk has to overlap the previous split's compute
        if (bw == 0.0f || b < bw) {
            bw = b;
        }
    }
    return bw;
}

// The fused per-device kernels (see ggml_cuda_try_fuse) must not write into a
// source they still read. The scheduler allocates the graph's tensors before
// the Meta backend dispatches the per-device subgraphs, so the CUDA-side
// graph_optimize (which registers the alloc deps against the scheduler's
// allocator) never runs for graphs owned by this backend. Walk the graph for
// the same structural patterns here so the gallocr keeps the sources alive
// until the fused kernel's destination is written.
static void ggml_backend_meta_graph_optimize(ggml_backend_t backend,
                                             struct ggml_cgraph * cgraph,
                                             struct ggml_backend_graph_optimize_params * params) {
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    backend_ctx->opt_has_eval_callback = params != nullptr && params->has_eval_callback;

    // The scheduler never calls a child backend's graph_optimize for a graph this backend owns, so
    // forward its alloc-deps half here, over the whole scheduled graph (original tensors), while the
    // scheduler's collector is still live and before allocation.  A child's full pass also sets its
    // MMB markings, but those must be keyed by the per-device *simple* tensors the child compute
    // sees; that half is deferred to graph_compute (`marks_only`).  `allocs_only` here keeps the two
    // halves from clobbering each other (the marks would otherwise be built for the original tensor
    // pointers and then never found at compute time).
    if (params != nullptr) {
        struct ggml_backend_graph_optimize_params alloc_params = *params;
        alloc_params.marks_only  = false;
        alloc_params.allocs_only = true;
        for (size_t j = 0; j < backend_ctx->backend_configs.size(); j++) {
            ggml_backend_t b = backend_ctx->backend_configs[j].backend;
            if (b->iface.graph_optimize != nullptr) {
                b->iface.graph_optimize(b, cgraph, &alloc_params);
            }
        }
    }

    static const bool disable_fusion = GGML_ENV_STR("GGML_CUDA_DISABLE_FUSION") != nullptr && std::atoi(GGML_ENV_STR("GGML_CUDA_DISABLE_FUSION"));
    if (disable_fusion) {
        return;
    }

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        if (cgraph->nodes[i]->op != GGML_OP_MUL) {
            continue;
        }

        ggml_moe_weighted_reduction_match match;
        if (!ggml_match_moe_weighted_reduction(cgraph, i, match)) {
            continue;
        }

        params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(match.experts), match.dst);
        params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(match.weights), match.dst);
        if (match.expert_scale != nullptr) {
            params->add_alloc_dep(
                params->user_data, const_cast<ggml_tensor *>(match.expert_scale), match.dst);
        }
        i += match.node_count - 1;
    }
}

// RAII: once a split's child graphs have been launched their kernels hold the staged slot pointers,
// so the simple tensors can be pointed back at their real data and the slots freed -- on every exit
// path, an early error return included.  Recording the free event on the device's main stream is what
// lets a later upload reuse the slot through `stage_wait`.
struct ggml_backend_meta_stage_guard {
    ggml_backend_meta_context * ctx;
    explicit ggml_backend_meta_stage_guard(ggml_backend_meta_context * ctx) : ctx(ctx) {}
    ~ggml_backend_meta_stage_guard() {
        // A/B: `GGML_STAGE_NO_RESTORE=1` leaves `simple_tensor->data` pointing at the ring slot.  If the
        // output then becomes coherent, the restore was clobbering the pointer before the kernels read it.
        static const bool no_restore = getenv("GGML_STAGE_NO_RESTORE") != nullptr;
        for (auto & entry : ctx->stage_pending) {
            for (auto & chunk : entry.chunks) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(entry.tensor, chunk.device);
                if (simple_tensor != nullptr && !no_restore) {
                    simple_tensor->data = chunk.orig;
                }
                ggml_backend_event_record(ctx->stage_free_ev[chunk.device][entry.slot], chunk.backend);
            }
        }
        ctx->stage_pending.clear();
    }
};

static enum ggml_status ggml_backend_meta_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(cgraph->grads == nullptr);
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;

    // If the previous cgraph had a defined UID it can be used to skip rebuilding the subgraphs per simple backend.
    const bool needs_rebuild = (cgraph->uid == 0) || (cgraph->uid != backend_ctx->uid);

    bool max_nnodes_raised = false;
    if (cgraph->n_nodes > backend_ctx->max_nnodes) {
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            bcj.nodes.resize(cgraph->n_nodes);
            bcj.cgraphs.resize(cgraph->n_nodes);
        }
        backend_ctx->max_nnodes = cgraph->n_nodes;
        max_nnodes_raised = true;
        assert(needs_rebuild);
    }

    if (needs_rebuild) {
        std::set<ggml_backend_buffer_t> used_buffers;
        for (int i = 0; i < cgraph->n_leafs; i++) {
            if (ggml_backend_buffer_is_meta(cgraph->leafs[i]->buffer)) {
                used_buffers.emplace(cgraph->leafs[i]->buffer);
            }
        }
        for (int i = 0; i < cgraph->n_nodes; i++) {
            if (ggml_backend_buffer_is_meta(cgraph->nodes[i]->buffer)) {
                used_buffers.emplace(cgraph->nodes[i]->buffer);
            }
        }
        for (ggml_backend_buffer_t buf : used_buffers) {
            ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buf->context;
            const int slot = g_meta_view_slot;
            buf_ctx->stc_compute_index_next[slot] = buf_ctx->stc_compute_index[slot] ^ 1;
            ggml_backend_meta_simple_tensor_container & stc = buf_ctx->compute_container(buf_ctx->stc_compute_index_next[slot]);
            stc.reset();
            stc.simple_tensors.clear();
        }
        size_t n_subgraphs  = 0;
        size_t max_tmp_size = 0;

        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];

            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (ggml_backend_buffer_is_host(node->buffer)) {
                    // FIXME host tensors are not part of the meta split: passing them to
                    // ggml_backend_meta_buffer_simple_tensor crashes. This covers views of
                    // NONE-op leafs (e.g. s_copy_main) as well as views of computed host
                    // tensors (e.g. a reshaped token embedding) and host op results (e.g.
                    // the PLE get_rows); for regular usage these are noops in the split
                    // graph, so keeping the original tensor is fine.
                    bcj.nodes[i] = node;
                    continue;
                }

                if (!ggml_backend_buffer_is_meta(node->buffer)) {
                    // [TAG_META_HOST_VIEWS]
                    GGML_ASSERT(ggml_backend_meta_is_host_view(node));

                    // keep the node as is, mapping it to a simple tensor is not possible
                    bcj.nodes[i] = node;
                    continue;
                }
                bcj.nodes[i] = ggml_backend_meta_buffer_simple_tensor(node, j);
                GGML_ASSERT(bcj.nodes[i]);
            }
        }

        // op-offload H2D staging (issue #50 WIP): the inputs `stage_input` took over are in flight on
        // each device's auxiliary copy stream.  Make that device's main stream wait for its copy and
        // point its simple tensor at the slot so the child graphs consume the staged bytes; both
        // happen before any child graph is launched, and the guard below undoes them once they all
        // are.  The lookup is redone here rather than cached because a rebuild replaces the simple
        // tensors.
        for (auto & entry : backend_ctx->stage_pending) {
            for (auto & chunk : entry.chunks) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(entry.tensor, chunk.device);
                GGML_ASSERT(simple_tensor != nullptr);
                ggml_backend_event_t done_ev = backend_ctx->stage_done_ev[chunk.device][entry.slot];
                GGML_ASSERT(done_ev != nullptr);
                ggml_backend_event_wait(chunk.backend, done_ev);
                chunk.orig = simple_tensor->data;
                // Copy the staged slot into the tensor's real buffer instead of repointing the tensor
                // at the slot.  The redirect is not safe for a host-offloaded expert table: its
                // consumer does not read it through `simple_tensor->data`, so the slot is ignored and
                // the (stale) real buffer is used instead -- a silent corruption of the whole layer.
                // Measured (2 x R9700, IQ3_XXS, `-sm tensor -ub 4096`, ~3.7k-token prompt): redirect ->
                // `////` x1 and MTP acc 1.00; this copy -> coherent, acc 3.88, and it keeps the
                // staging win (IQ4_XS 16k: ppN 1000 vs 875 with staging off, decode 43.6 vs 48.0).
                // `GGML_STAGE_META_REDIRECT=1` restores the redirect for A/B only -- it is not a
                // correct configuration.
                if (getenv("GGML_STAGE_META_REDIRECT") == nullptr && chunk.backend->iface.stage_d2d != NULL) {
                    chunk.backend->iface.stage_d2d(chunk.backend, chunk.orig, chunk.slot,
                                                   ggml_nbytes(simple_tensor));
                } else {
                    simple_tensor->data = chunk.slot;
                }
            }
        }
        const ggml_backend_meta_stage_guard stage_guard(backend_ctx);

        {
            // For MoE models it may make sense to delay the AllReduce in order to reduce I/O:
            auto get_i_delayed_branch = [&](const int i) -> int {
                int id = i; // i_delayed
                int idr = i; // i_delayed return, last safe return value

                ggml_tensor * node = cgraph->nodes[id];
                int32_t n_used = ggml_node_get_use_count(cgraph, id);

                // Skip MIRRORED nodes that don't consume node
                auto skip_unrelated = [&]() {
                    while (id + 1 < cgraph->n_nodes) {
                        ggml_tensor * next = cgraph->nodes[id+1];
                        if (ggml_backend_meta_get_split_state(next, false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                            break;
                        }
                        bool safe = true;
                        for (int s = 0; s < GGML_MAX_SRC; s++) {
                            if (next->src[s] == nullptr) {
                                continue;
                            }
                            if (next->src[s] == node) {
                                safe = false;
                                break;
                            }
                            if (ggml_backend_meta_get_split_state(next->src[s], false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                                safe = false;
                                break;
                            }
                        }
                        if (!safe) {
                            break;
                        }
                        id++;
                    }
                };

                skip_unrelated();
                if (id + 1 >= cgraph->n_nodes) {
                    return idr;
                }
                {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op == GGML_OP_ADD_ID && next->src[0] == node &&
                            ggml_backend_meta_get_split_state(next->src[1], false).axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL &&
                            ggml_backend_meta_get_split_state(next->src[2], false).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        node = next;
                        id++;
                        idr = id;
                        n_used = ggml_node_get_use_count(cgraph, id);
                    }
                }
                // Chain of MULs with MIRRORED src[1]
                while (true) {
                    skip_unrelated();
                    if (id + 1 >= cgraph->n_nodes) {
                        return idr;
                    }
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op == GGML_OP_MUL && next->src[0] == node &&
                            (ggml_backend_meta_get_split_state(next->src[1], false).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED ||
                             ggml_backend_meta_get_split_state(next->src[1], false).axis == GGML_BACKEND_SPLIT_AXIS_PER_DEVICE)) {
                        // A per-device operand (the MoE amax rescale's undo factor) must be included in the
                        // branch, so the reduce point stays AFTER it: each backend then applies its own
                        // factor to its own partial, which is what makes the per-device value cancel.
                        // Stopping before this MUL would reduce the unscaled partials and then multiply the
                        // sum by one backend's factor -- wrong, and silently so whenever every factor is 1.
                        node = next;
                        id++;
                        idr = id;
                        n_used = ggml_node_get_use_count(cgraph, id);
                    } else {
                        break;
                    }
                }

                if (n_used != node->ne[1] || id + 2*n_used-1 >= cgraph->n_nodes) {
                    return idr;
                }
                for (int32_t k = 0; k < n_used; k++) {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_VIEW || next->view_src != node || next->view_offs != k*node->nb[1] ||
                            next->ne[0] != node->ne[0] || next->ne[1] != node->ne[2] || next->nb[1] != node->nb[2] ||
                            ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_ADD || next->src[0] != cgraph->nodes[id - (n_used-1)] ||
                            next->src[1] != cgraph->nodes[id - (n_used-2)] || ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                for (int32_t k = 0; k < n_used - 2; k++) {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_ADD || next->src[0] != cgraph->nodes[id] ||
                            next->src[1] != cgraph->nodes[id - (n_used-2)] || ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                idr = id;
                return idr;
            };

            // AllReduce(a) + AllReduce(b) == AllReduce(a + b) for independent partial branches.
            auto get_i_delayed = [&](const int i) -> int {
                const int i_delayed = get_i_delayed_branch(i);
                ggml_tensor * node = cgraph->nodes[i_delayed];

                if (ggml_node_get_use_count(cgraph, i_delayed) != 1) {
                    return i_delayed;
                }

                for (int id = i_delayed + 1; id < cgraph->n_nodes; id++) {
                    ggml_tensor * next = cgraph->nodes[id];
                    if (next->view_src == node) {
                        return i_delayed;
                    }
                    for (int s = 0; s < GGML_MAX_SRC; s++) {
                        if (next->src[s] == node) {
                            return i_delayed;
                        }
                    }

                    if (next->view_src != nullptr && next->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(next->view_src->buffer)) {
                        continue;
                    }
                    if (ggml_backend_meta_get_split_state(next, false).axis != GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                        continue;
                    }

                    const int i_other = id;
                    const int i_other_delayed = get_i_delayed_branch(i_other);
                    ggml_tensor * other = cgraph->nodes[i_other_delayed];
                    if (ggml_node_get_use_count(cgraph, i_other_delayed) != 1 || i_other_delayed + 1 >= cgraph->n_nodes) {
                        return i_delayed;
                    }

                    ggml_tensor * sum = cgraph->nodes[i_other_delayed + 1];
                    if (sum->op != GGML_OP_ADD ||
                            !ggml_are_same_shape(node, other) || node->type != other->type || sum->type != node->type ||
                            !((sum->src[0] == node && sum->src[1] == other) ||
                              (sum->src[0] == other && sum->src[1] == node)) ||
                            ggml_backend_meta_get_split_state(sum, false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        return i_delayed;
                    }

                    for (size_t j = 0; j < n_backends; j++) {
                        auto & bcj = backend_ctx->backend_configs[j];
                        const bool compute       = bcj.nodes[i]->flags       & GGML_TENSOR_FLAG_COMPUTE;
                        const bool compute_other = bcj.nodes[i_other]->flags & GGML_TENSOR_FLAG_COMPUTE;
                        if (compute != compute_other) {
                            return i_delayed;
                        }
                    }
                    return i_other_delayed + 1;
                }
                return i_delayed;
            };

            int i_start = 0;
            std::vector<int> gather_nodes_new;
            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (node->view_src != nullptr && node->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(node->view_src->buffer)) {
                    continue;
                }
                const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(node, /*assume_sync =*/ false);
                if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                    max_tmp_size = std::max(max_tmp_size, ggml_nbytes(node));
                }
                // ALL_GATHER is a collective that replaces the node's own computation, so it is a boundary
                // of its own: the subgraph ends just before it, the gather runs, and the following subgraph
                // starts after it.  Without the cut the per-device kernels would each write their own slice
                // into the front of their copy.
                const bool is_gather = i + 1 < cgraph->n_nodes && cgraph->nodes[i + 1]->op == GGML_OP_ALL_GATHER;
                const bool new_subgraph = i + 1 == cgraph->n_nodes || split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL || is_gather;
                if (!new_subgraph) {
                    continue;
                }

                // the delayed-reduce search walks forward looking for a better reduce point and must not
                // step over the gather node
                const int i_delayed = is_gather ? i : get_i_delayed(i);

                // If we can delay the AllReduce we need to consider the interaction with zero-sized tensor slices.
                // A backend with such a slice would normally have valid data after participating in the AllReduce with a node that has
                //     its compute flag disabled and thus gets its data zeroed out.
                // If the AllReduce is delayed then the nodes until that point also need to have their compute flag disabled.
                if (i_delayed > i) {
                    for (size_t j = 0; j < n_backends; j++) {
                        auto & bcj = backend_ctx->backend_configs[j];
                        if ((bcj.nodes[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                            for (int ii = i + 1; ii <= i_delayed; ii++) {
                                bcj.nodes[ii]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
                            }
                        }
                    }
                }

                i = i_delayed;

                for (size_t j = 0; j < n_backends; j++) {
                    auto & bcj = backend_ctx->backend_configs[j];
                    bcj.cgraphs[n_subgraphs].offset = i_start;
                }
                gather_nodes_new.push_back(is_gather ? i + 1 : -1);
                n_subgraphs++;
                i_start = i + 1;
            }
            GGML_ASSERT(i_start == cgraph->n_nodes);
            backend_ctx->gather_nodes = std::move(gather_nodes_new);
        }

        backend_ctx->uid         = cgraph->uid;
        backend_ctx->n_subgraphs = n_subgraphs;

        if (max_tmp_size > backend_ctx->max_tmp_size) {
            for (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                for (size_t i = 0; i < backend_ctx->n_reduce_steps; i++) {
                    bcj.bufs[i].reset(ggml_backend_alloc_buffer(bcj.backend, max_tmp_size));
                }
            }
            backend_ctx->max_tmp_size = max_tmp_size;
        }

        if (max_nnodes_raised || n_subgraphs > backend_ctx->max_subgraphs) {
            backend_ctx->max_subgraphs = std::max(backend_ctx->max_subgraphs, n_subgraphs);
            const size_t n_nodes_per_device = 3 * backend_ctx->n_reduce_steps; // tmp + ADD (+zeroing) graph per step and device
            const size_t n_cgraphs_per_device = 2 * backend_ctx->n_reduce_steps; // ADD ( + zeroing) graph per step and device
            const size_t mem_per_device_graphs_main = backend_ctx->max_subgraphs*ggml_graph_overhead_custom(backend_ctx->max_nnodes, cgraph->grads);
            const size_t mem_per_device_graphs_aux = n_cgraphs_per_device*backend_ctx->max_subgraphs*ggml_graph_overhead_custom(1, cgraph->grads);
            const size_t mem_per_device_nodes_aux = n_nodes_per_device*backend_ctx->max_subgraphs*ggml_tensor_overhead()
                                                  + 2*n_backends*GGML_META_GATHER_MAX_ROWS*ggml_tensor_overhead(); // ALL_GATHER views
            const ggml_init_params params = {
                /*.mem_size   =*/ n_backends * (mem_per_device_graphs_main + mem_per_device_graphs_aux + mem_per_device_nodes_aux),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            backend_ctx->ctx.reset(ggml_init(params));
            for (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                for (size_t i = 0; i < backend_ctx->max_subgraphs; i++) {
                    bcj.cgraphs[i].cgraph_main = ggml_new_graph_custom(backend_ctx->ctx.get(), backend_ctx->max_nnodes, /*grads =*/ false);
                }
            }
            backend_ctx->cgraphs_aux.resize(n_backends*n_cgraphs_per_device*backend_ctx->max_subgraphs);
            for (size_t k = 0; k < backend_ctx->cgraphs_aux.size(); k++) {
                backend_ctx->cgraphs_aux[k] = ggml_new_graph_custom(backend_ctx->ctx.get(), 1, cgraph->grads);
            }
            backend_ctx->nodes_aux.resize(n_backends*n_nodes_per_device*backend_ctx->max_subgraphs);
            for (size_t k = 0; k < backend_ctx->nodes_aux.size(); k++) {
                backend_ctx->nodes_aux[k] = ggml_new_tensor_1d(backend_ctx->ctx.get(), GGML_TYPE_F32, 1);
            }
            // scratch views for the ALL_GATHER collectives: one (source, destination, row) triple takes
            // two 1-D view tensors.  The row count is a runtime property of the gathered tensor, so the
            // pool is sized for the largest token batch a split can carry plus slack, and `all_gather`
            // asserts rather than overrunning it.
            backend_ctx->gather_views.resize(2*n_backends*n_backends*GGML_META_GATHER_MAX_ROWS);
            for (size_t k = 0; k < backend_ctx->gather_views.size(); k++) {
                backend_ctx->gather_views[k] = ggml_new_tensor_1d(backend_ctx->ctx.get(), GGML_TYPE_F32, 1);
            }
        }

        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            for (size_t i_graph = 0; i_graph < n_subgraphs; i_graph++) {
                ggml_cgraph * cgraph_ij = bcj.cgraphs[i_graph].cgraph_main;
                const size_t i_node_start = bcj.cgraphs[i_graph].offset;
                const size_t i_node_stop = i_graph + 1 < n_subgraphs ? bcj.cgraphs[i_graph + 1].offset : cgraph->n_nodes;
                cgraph_ij->n_nodes = 0;
                ggml_hash_set_reset(&cgraph_ij->visited_hash_set);
                for (size_t i_node = i_node_start; i_node < i_node_stop; i_node++) {
                    if (cgraph->nodes[i_node]->op == GGML_OP_ALL_GATHER) {
                        // the meta collective fills this tensor (each device's slice written into every
                        // device's copy); a device-local execution would copy its own slice only
                        continue;
                    }
                    ggml_tensor * node_ij = bcj.nodes[i_node];
                    cgraph_ij->nodes[cgraph_ij->n_nodes++] = node_ij;
                    const size_t hash_pos_orig = ggml_hash_find(&cgraph->visited_hash_set, cgraph->nodes[i_node]);
                    const size_t hash_pos_ij = ggml_hash_insert(&cgraph_ij->visited_hash_set, node_ij);
                    cgraph_ij->use_counts[hash_pos_ij] = cgraph->use_counts[hash_pos_orig];
                }
                cgraph_ij->uid = ggml_graph_next_uid();
            }
        }

        // The child backends' marking half.  The scheduler never calls their graph_optimize under
        // this backend, so the MMB HC16 / DOWN16 / blk16 / res16 markings would be inert.  Run them
        // on the per-device subgraphs now, before any of them is computed (the mark lifetime assumes
        // every subgraph of a graph is optimized before any is computed).  `marks_only` skips the
        // alloc-deps half, which graph_optimize already registered before allocation, and
        // `full_graph == NULL` treats each subgraph as self-contained: a marked producer and every
        // consumer reading through its BF16 activation cache are always on the same device.
        {
            struct ggml_backend_graph_optimize_params mark_params = {
                /* .add_alloc_dep     = */ [](void *, ggml_tensor *, ggml_tensor *) {},
                /* .user_data         = */ nullptr,
                /* .has_eval_callback = */ backend_ctx->opt_has_eval_callback,
                /* .full_graph        = */ nullptr,
                /* .marks_only        = */ true,
                /* .allocs_only       = */ false,
            };
            for (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                if (bcj.backend->iface.graph_optimize == nullptr) {
                    continue;
                }
                for (size_t i_graph = 0; i_graph < n_subgraphs; i_graph++) {
                    bcj.backend->iface.graph_optimize(bcj.backend, bcj.cgraphs[i_graph].cgraph_main, &mark_params);
                }
            }
        }
    }

    size_t iga = 0; // i graph aux
    size_t ina = 0; // i node aux

    auto get_node_aux = [&](ggml_tensor * t) -> ggml_tensor * {
        // the pool is `3 * n_reduce_steps * max_subgraphs` tensors per device; a graph whose reduction
        // count exceeded what the sizing pass saw would otherwise write past it (the ALL_GATHER view
        // pool below asserts, this one did not)
        GGML_ASSERT(ina < backend_ctx->nodes_aux.size() && "node aux pool exhausted");
        ggml_tensor * ret = backend_ctx->nodes_aux[ina++];
        memset(ret, 0, sizeof(ggml_tensor));
        ret->op   = GGML_OP_NONE;
        ret->type = t->type;
        for (size_t k = 0; k < GGML_MAX_DIMS; k++) {
            ret->ne[k] = t->ne[k];
            ret->nb[k] = t->nb[k];
        }
        return ret;
    };
    auto set_tmp_data = [&](ggml_tensor * tensor, const size_t j, const size_t i_buf) {
        auto & bcj = backend_ctx->backend_configs[j];
        ggml_backend_buffer_ptr & buf_ptr = bcj.bufs[i_buf];
        if (!buf_ptr || ggml_backend_buffer_get_size(buf_ptr.get()) < backend_ctx->max_tmp_size) {
            buf_ptr.reset(ggml_backend_alloc_buffer(bcj.backend, backend_ctx->max_tmp_size));
        }
        tensor->buffer = buf_ptr.get();
        tensor->data   = ggml_backend_buffer_get_base(buf_ptr.get());
    };
    // FIXME usage_counts
    auto get_cgraph_aux = [&]() -> ggml_cgraph * {
        ggml_cgraph * ret = backend_ctx->cgraphs_aux[iga++];
        return ret;
    };

    // Preferentially use backend-specific allreduce_tensor_async (e.g. NCCL for CUDA), use a generic fallback if unavailable:
    auto allreduce_fallback = [&](size_t i) -> ggml_status {
        std::vector<ggml_cgraph *> step_cgraphs(n_backends, nullptr);

        // Zero out nodes that were disabled due to having a zero-sized slice:
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            ggml_tensor * node = bcj.cgraphs[i].cgraph_main->nodes[bcj.cgraphs[i].cgraph_main->n_nodes - 1];
            if (node->flags & GGML_TENSOR_FLAG_COMPUTE) {
                continue;
            }
            if (ggml_nelements(node) > 0) {
                ggml_tensor * node_zero = get_node_aux(node);
                node_zero->op = GGML_OP_FILL;
                node_zero->src[0] = node; // only used for the shape, the data is not read
                ggml_set_op_params_f32(node_zero, 0, 0.0f);
                node_zero->data = node->data;
                node_zero->buffer = node->buffer;
                node_zero->flags |= GGML_TENSOR_FLAG_COMPUTE;

                step_cgraphs[j] = get_cgraph_aux();
                step_cgraphs[j]->nodes[0] = node_zero;
                step_cgraphs[j]->n_nodes = 1;
                const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[j]);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
        }
        std::fill(step_cgraphs.begin(), step_cgraphs.end(), nullptr);

        auto push_data = [&](const size_t j_src, const size_t j_dst, const size_t i_buf) {
            assert(step_cgraphs[j_dst] == nullptr);
            auto & bcj_src = backend_ctx->backend_configs[j_src];
            auto & bcj_dst = backend_ctx->backend_configs[j_dst];

            ggml_tensor * node_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * node_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            GGML_ASSERT(ggml_is_contiguous(node_src));
            GGML_ASSERT(ggml_is_contiguous(node_dst));

            ggml_tensor * node_tmp = get_node_aux(node_dst);
            set_tmp_data(node_tmp, j_dst, i_buf);

            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, node_src, node_tmp);

            ggml_tensor * node_red = get_node_aux(node_dst);
            node_red->view_src = node_dst->view_src == nullptr ? node_dst : node_dst->view_src;
            node_red->view_offs = node_dst->view_offs;
            node_red->op = GGML_OP_ADD;
            node_red->src[0] = node_dst;
            node_red->src[1] = node_tmp;
            node_red->flags |= GGML_TENSOR_FLAG_COMPUTE;
            ggml_backend_view_init(node_red);

            ggml_cgraph * cgraph_aux = get_cgraph_aux();
            cgraph_aux->nodes[0] = node_red;
            cgraph_aux->n_nodes = 1;
            step_cgraphs[j_dst] = cgraph_aux;
        };

        size_t offset_j = n_backends/2;
        while ((offset_j & (offset_j - 1)) != 0) {
            offset_j--;
        }
        const size_t offset_j_max = offset_j;
        size_t i_buf = 0;

        // If n_backends is not a power of 2, fold in the excess prior to butterfly reduction:
        for (size_t j_src = 2*offset_j_max; j_src < n_backends; j_src++) {
            const size_t j_dst = j_src - 2*offset_j_max;
            push_data(j_src, j_dst, i_buf);
            const ggml_status status = ggml_backend_graph_compute_async(backend_ctx->backend_configs[j_dst].backend, step_cgraphs[j_dst]);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
            i_buf = 1;
        }

        // Butterfly reduction:
        for (; offset_j >= 1; offset_j /= 2) {
            std::fill(step_cgraphs.begin(), step_cgraphs.end(), nullptr);

            for (size_t j = 0; j < 2*offset_j_max; j++) {
                const size_t j_other = j ^ offset_j;
                if (j_other >= n_backends) {
                    continue;
                }
                push_data(j, j_other, i_buf);
            }

            for (size_t j = 0; j < 2*offset_j_max; j++) {
                if (step_cgraphs[j] == nullptr) {
                    continue;
                }
                auto & bcj = backend_ctx->backend_configs[j];
                const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[j]);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
            i_buf++;
        }
        assert(i_buf == backend_ctx->n_reduce_steps);

        // If n_backends is not a power of 2, copy back the reduced tensors to the excess:
        for (size_t j = 2*offset_j_max; j < n_backends; j++) {
            auto & bcj_src = backend_ctx->backend_configs[j - 2*offset_j_max];
            auto & bcj_dst = backend_ctx->backend_configs[j];

            ggml_tensor * node_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * node_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, node_src, node_dst);
        }

        return GGML_STATUS_SUCCESS;
    };


    // The ALL_GATHER collective: every device ends up holding the WHOLE input tensor.
    //
    // On a Meta device an op that reduces along an axis-0 split (the DFlash2 selector's `ggml_top_k`
    // over vocab-split draft logits, and the `ggml_get_rows` that reads the candidate values back)
    // cannot be computed per device: no device holds the whole row.  This op is the split state's way
    // of saying "I need all of it": each device's slice is copied into every device's full copy at
    // that slice's own offset.  The slices are disjoint (they tile the tensor) and nothing is summed,
    // so the element positions of the split axis are preserved across devices -- which is what makes a
    // vocabulary index produced by one device meaningful on the others.
    //
    // Only a plain axis-0 split (one segment, no repeats: each device holds a contiguous run of rows) is
    // implemented; anything else is a deliberate abort rather than a silently wrong gather.
    auto all_gather = [&](const int i_node) -> ggml_status {
        ggml_tensor * node = cgraph->nodes[i_node];
        ggml_tensor * src  = node->src[0];
        GGML_ASSERT(node->op == GGML_OP_ALL_GATHER && src != nullptr);

        int i_src = -1;
        for (int k = 0; k < i_node; k++) {
            if (cgraph->nodes[k] == src) {
                i_src = k;
                break;
            }
        }
        GGML_ASSERT(i_src >= 0);

        const ggml_backend_meta_split_state ss = ggml_backend_meta_get_split_state(src, /*assume_sync =*/ false);
        const bool split = ss.axis >= 0 && ss.axis < GGML_MAX_DIMS;
        if (split) {
            GGML_ASSERT(ss.axis == GGML_BACKEND_SPLIT_AXIS_0);
            GGML_ASSERT(ss.n_segments == 1 && ss.nr[0] == 1);
            GGML_ASSERT(ggml_is_contiguous(src) && ggml_is_contiguous(node));
        } else {
            GGML_ASSERT(ss.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        }

        // Each device's slice is contiguous in its OWN copy, but the destination is the full tensor, where
        // the split axis (0) is the slowest: the copies have to go one "row" (one element of every dim > 0)
        // at a time, exactly like the canonical host assembly in `ggml_backend_meta_get_tensor_impl`.
        const int64_t ne0     = src->ne[0];
        const int64_t n_rows  = src->ne[1] * src->ne[2] * src->ne[3];
        const size_t  elem    = ggml_element_size(src);

        const size_t n_src = split ? n_backends : 1;
        GGML_ASSERT(n_rows <= GGML_META_GATHER_MAX_ROWS && "ALL_GATHER row batch exceeds GGML_META_GATHER_MAX_ROWS");
        GGML_ASSERT(2*n_src*n_backends*n_rows <= backend_ctx->gather_views.size());

        size_t   i_view    = 0;
        int64_t  elems     = 0; // elements that precede this source's slice in the full tensor
        for (size_t s = 0; s < n_src; s++) {
            ggml_tensor * src_s = backend_ctx->backend_configs[s].nodes[i_src];
            GGML_ASSERT(src_s != nullptr);
            const int64_t ne_j  = split ? ss.ne[s] * ss.nr[0] : ne0;
            const size_t  nbytes_row = size_t(ne_j) * elem;
            GGML_ASSERT(elems + ne_j * n_rows <= ggml_nelements(node));
            for (size_t d = 0; d < n_backends; d++) {
                ggml_tensor * dst_d = backend_ctx->backend_configs[d].nodes[i_node];
                GGML_ASSERT(dst_d != nullptr);
                for (int64_t r = 0; r < n_rows; r++) {
                    ggml_tensor * row_src = backend_ctx->gather_views[i_view++];
                    ggml_tensor * row_dst = backend_ctx->gather_views[i_view++];
                    memset(row_src, 0, sizeof(ggml_tensor));
                    memset(row_dst, 0, sizeof(ggml_tensor));
                    for (ggml_tensor * t : { row_src, row_dst }) {
                        t->type    = src_s->type;
                        t->ne[0]   = ne_j;
                        t->nb[0]   = elem;
                        // ne/nb of the trailing dims must be a valid 1-element row: the copy sizes the
                        // transfer with `ggml_nbytes`, which returns 0 for a zero extent
                        for (int k = 1; k < GGML_MAX_DIMS; k++) {
                            t->ne[k] = 1;
                            t->nb[k] = nbytes_row;
                        }
                    }
                    // 1-D views: the source row of this device's slice, the destination row of the full tensor
                    row_src->view_src  = src_s;
                    row_src->view_offs = r * src_s->nb[1];
                    row_dst->view_src  = dst_d;
                    row_dst->view_offs = (r * ne0 + elems) * elem;
                    GGML_ASSERT(row_src->view_offs + nbytes_row <= ggml_nbytes(src_s));
                    GGML_ASSERT(row_dst->view_offs + nbytes_row <= ggml_nbytes(dst_d));
                    if (ggml_backend_view_init(row_src) != GGML_STATUS_SUCCESS ||
                        ggml_backend_view_init(row_dst) != GGML_STATUS_SUCCESS) {
                        return GGML_STATUS_FAILED;
                    }
                    ggml_backend_tensor_copy_async(backend_ctx->backend_configs[s].backend,
                            backend_ctx->backend_configs[d].backend, row_src, row_dst);
                }
            }
            elems += ne_j;
        }
        GGML_ASSERT(elems == ne0);

        return GGML_STATUS_SUCCESS;
    };

    for (size_t i = 0; i < backend_ctx->n_subgraphs; i++) {
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, bcj.cgraphs[i].cgraph_main);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
        }

        if (i < backend_ctx->n_subgraphs - 1 && i < backend_ctx->gather_nodes.size() && backend_ctx->gather_nodes[i] >= 0) {
            const ggml_status status = all_gather(backend_ctx->gather_nodes[i]);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
            continue;
        }

        if (n_backends > 1 && i < backend_ctx->n_subgraphs - 1) {
            bool backend_allreduce_success = false;
            if (backend_ctx->comm_ctx) {
                std::vector<ggml_tensor *> nodes;
                nodes.reserve(n_backends);
                for (size_t j = 0; j < n_backends; j++) {
                    auto & bcj = backend_ctx->backend_configs[j];
                    ggml_cgraph * cgraph_ij = bcj.cgraphs[i].cgraph_main;
                    nodes.push_back(cgraph_ij->nodes[cgraph_ij->n_nodes-1]);
                }
                backend_allreduce_success = backend_ctx->comm_allreduce(backend_ctx->comm_ctx, nodes.data());
            }

            if (!backend_allreduce_success) {
                const ggml_status status = allreduce_fallback(i);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
        }
    }
    return GGML_STATUS_SUCCESS;
}

// wip/moe-expert-cache Phase 3 (`-sm tensor`): the Meta backend is the split backend the scheduler
// calls for a host-resident expert upload, but the cache lives in the per-device CUDA backends.  This
// hook is a thin delegator: it derives each device's slice geometry from the meta split state and
// forwards to that simple backend's own `moe_cache_update`, passing the device's simple tensor (which
// is exactly the tensor the CUDA `MUL_MAT_ID` consumer will read) and the slice's byte offset inside a
// host expert.  Returns true only when EVERY device took the input over, so the scheduler's skip-copy
// decision stays all-or-nothing (a mixed outcome would leave some devices reading a buffer nobody
// filled).
static int64_t ggml_backend_meta_moe_cache_band(ggml_backend_t backend);

// wip/fix-gemma-4 (TODO #77/C): the per-device source geometry of a FUSED expert table.  The device's
// expert slot is the concatenation of one chunk per (segment, repeat) -- exactly the layout the
// synchronous segmented setter writes: `write_full_rows`/`write_partial_row` accumulate `chunk(s, j)` into
// the device row over (s, r) in order, while the host row walks (s, r, j).  `chunk(s, j)` is
// `ne[s*n_bufs + j]` elements, bytes `(ne / blck_size) * nb[0]` on an axis-0 split and `ne * nb[1]` on
// axis 1.  Returns false when the geometry is not expressible (too many slices, zero-length chunk, or a
// list that does not tile the device row), in which case the caller declines the table and the
// scheduler's per-op copy serves it.
static bool ggml_backend_meta_moe_slices(const ggml_backend_meta_split_state & ss, const ggml_tensor * host,
        const ggml_tensor * device, size_t j, size_t n_bufs,
        struct ggml_backend_moe_slice * slices, int * n_slices) {
    const int   axis  = (int) ss.axis;
    const bool  axis0 = axis == GGML_BACKEND_SPLIT_AXIS_0;
    const int64_t blck = ggml_blck_size(host->type);
    int    n     = 0;
    size_t p_src = 0;
    size_t total = 0;
    for (size_t s = 0; s < ss.n_segments; s++) {
        for (size_t r = 0; r < ss.nr[s]; r++) {
            for (size_t d = 0; d < n_bufs; d++) {
                const size_t ne_d = (size_t) ss.ne[s*n_bufs + d];
                const size_t nbytes = axis0 ? (size_t) (ne_d / (size_t) blck) * host->nb[0]
                                            : (size_t) ne_d * host->nb[1];
                if (d == j) {
                    if (nbytes == 0 || n >= GGML_BACKEND_MOE_MAX_SLICES) {
                        return false;
                    }
                    slices[n].src_off = p_src;
                    slices[n].len     = nbytes;
                    total            += nbytes;
                    n++;
                }
                p_src += nbytes;
            }
        }
    }
    if (n == 0 || total != (size_t) device->nb[axis + 1]) {
        return false;
    }
    *n_slices = n;
    return true;
}

static bool ggml_backend_meta_moe_cache_update(ggml_backend_t backend, const struct ggml_tensor * weight,
        const struct ggml_tensor * weight_cpy, const int32_t * ids, int64_t n_used, int64_t n_tok,
        size_t ids_nb0, size_t ids_nb1, size_t slice_off, int split_axis,
        const struct ggml_backend_moe_slice * slices_in, int n_slices_in) {
    GGML_UNUSED(slice_off);
    GGML_UNUSED(split_axis);
    GGML_UNUSED(slices_in);
    GGML_UNUSED(n_slices_in);
    // Only a routed expert table (a `MUL_MAT_ID` weight with an expert axis) is cacheable, and only the
    // decode/verify band (the narrowest band of the simple backends, see ggml_backend_meta_moe_cache_band).
    if (weight == nullptr || weight_cpy == nullptr || weight->ne[2] < 2 || n_tok > ggml_backend_meta_moe_cache_band(backend)) {
        return false;
    }
    if (!ggml_backend_buffer_is_meta(weight_cpy->buffer)) {
        return false;
    }
    const ggml_backend_meta_split_state ss = ggml_backend_meta_get_split_state(weight_cpy, /*assume_sync =*/ false);
    if (ss.axis < 0 || ss.axis >= GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
        // A mirrored table has no per-device slice geometry; leave it to the scheduler's normal copy.
        return false;
    }
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    if (ss.n_segments != 1 || ss.nr[0] != 1) {
        // FUSED table (gemma4 `ffn_gate_up_exps`): the device slot is `[seg0_slice | seg1_slice | ...]`.
        // Register it with an explicit per-device slice list so the cache can hold it at all -- without
        // this it never registers, the cache's completeness check never passes, no arena is sized and the
        // config pays a full per-op upload with zero residency (TODO #77).
        struct ggml_backend_moe_slice slice[GGML_BACKEND_MOE_MAX_SLICES];
        int    n_slices = 0;
        bool   all      = true;
        for (size_t j = 0; j < n_backends; j++) {
            ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
            ggml_tensor *  simple_tensor  = ggml_backend_meta_buffer_simple_tensor(weight_cpy, j);
            if (simple_tensor == nullptr || simple_backend->iface.moe_cache_update == nullptr ||
                !ggml_backend_meta_moe_slices(ss, weight, simple_tensor, j, n_backends, slice, &n_slices)) {
                return false;
            }
            const bool took = simple_backend->iface.moe_cache_update(
                    simple_backend, weight, simple_tensor, ids, n_used, n_tok, ids_nb0, ids_nb1,
                    0, (int) ss.axis, slice, n_slices);
            all = all && took;
        }
        return all;
    }
    size_t off = 0;
    bool all = true;
    for (size_t j = 0; j < n_backends; j++) {
        ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
        ggml_tensor *  simple_tensor  = ggml_backend_meta_buffer_simple_tensor(weight_cpy, j);
        if (simple_tensor == nullptr || simple_backend->iface.moe_cache_update == nullptr) {
            return false;
        }
        const bool took = simple_backend->iface.moe_cache_update(
                simple_backend, weight, simple_tensor, ids, n_used, n_tok, ids_nb0, ids_nb1, off, (int) ss.axis,
                nullptr, 0);
        all = all && took;
        off += simple_tensor->nb[ss.axis + 1];
    }
    return all;
}

// wip/moe-expert-cache (session 7) identity fast path: forward to every simple backend and take the
// input over only if ALL of them can (a mixed outcome would leave one device reading an arena nobody
// filled).
static bool ggml_backend_meta_moe_cache_take_over(ggml_backend_t backend, const struct ggml_tensor * weight,
        const struct ggml_tensor * weight_cpy) {
    if (weight == nullptr || weight_cpy == nullptr || weight->ne[2] < 2) {
        return false;
    }
    if (!ggml_backend_buffer_is_meta(weight_cpy->buffer)) {
        return false;
    }
    const ggml_backend_meta_split_state ss = ggml_backend_meta_get_split_state(weight_cpy, /*assume_sync =*/ false);
    if (ss.n_segments != 1 || ss.nr[0] != 1 || ss.axis < 0 || ss.axis >= GGML_MAX_DIMS) {
        return false;
    }
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    bool all = true;
    for (size_t j = 0; j < n_backends; j++) {
        ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
        ggml_tensor *  simple_tensor  = ggml_backend_meta_buffer_simple_tensor(weight_cpy, j);
        if (simple_tensor == nullptr || simple_backend->iface.moe_cache_take_over == nullptr) {
            return false;
        }
        all = all && simple_backend->iface.moe_cache_take_over(simple_backend, weight, simple_tensor);
    }
    return all;
}

// wip/moe-expert-cache (B2): forward the device-side expert gather to each simple device.  The gather
// reads the routing on-device, so every device must have its own device-resident routing: the ids must be
// a MIRRORED meta tensor (a simple per-device copy) or there must be a single device.  A single-device
// routing tensor on a multi-device split is refused (one device would read another's memory).  Returns
// true only if EVERY device gathered, so the scheduler's skip-copy decision stays all-or-nothing.
//
// The per-device source offset is the running sum of each device's slice stride
// (`off += simple_tensor->nb[ss.axis + 1]`).  That is exactly `device's slice size` when
// `n_segments == 1 && nr[0] == 1` (enforced above), because the split then takes a contiguous column
// block (axis 1, gate/up: `nb[2]` = one expert's slice) or the first rows of every row (axis 0,
// down: `nb[1]` = the row's slice width), and the source offset within a host expert is the sum of
// the preceding devices' slice sizes.  This holds for the 3-device qwen4exp (Qwen3.8-Flash-Next)
// uneven 640/3 expert split as well.  The session-16 corruption (`-sm tensor` prefill degenerating to
// one repeated token) was NOT this offset: it was the pruned gather leaving the MMQ's 512-byte
// expert-table tail over-read with the reused `input_cpy`'s stale/NaN bytes.  The CUDA gather now
// reproduces the host path's tail guard (see `moe_cache_gather_host`); the geometry was verified
// byte-exact for every routed expert on all three devices before the re-enable.
// The narrowest MoE cache band over the simple backends (a batch the meta backend forwards must be in every
// device's band); a simple backend without the hook counts as the historical 8.
static int64_t ggml_backend_meta_moe_cache_band(ggml_backend_t backend) {
    int64_t band = INT64_MAX;
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    for (size_t j = 0; j < n_backends; j++) {
        ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
        const int64_t b = simple_backend->iface.moe_cache_band != nullptr ? simple_backend->iface.moe_cache_band(simple_backend) : 8;
        band = std::min(band, b);
    }
    return n_backends > 0 ? band : 8;
}

static bool ggml_backend_meta_moe_cache_gather(ggml_backend_t backend, const struct ggml_tensor * weight,
        const struct ggml_tensor * weight_cpy, const struct ggml_tensor * ids,
        size_t slice_off, int split_axis) {
    GGML_UNUSED(slice_off);
    GGML_UNUSED(split_axis);
    if (weight == nullptr || weight_cpy == nullptr || ids == nullptr || weight->ne[2] < 2) {
        return false;
    }
    if (!ggml_backend_buffer_is_meta(weight_cpy->buffer)) {
        return false;
    }
    const size_t              n_backends = ggml_backend_meta_n_backends(backend);
    const ggml_backend_meta_split_state ss = ggml_backend_meta_get_split_state(weight_cpy, /*assume_sync =*/ false);
    if (ss.n_segments != 1 || ss.nr[0] != 1 || ss.axis < 0 || ss.axis >= GGML_MAX_DIMS) {
        return false;
    }
    const bool ids_meta = ggml_backend_buffer_is_meta(ids->buffer);
    if (!ids_meta && n_backends > 1) {
        return false;   // a single device routing tensor cannot be read by every device
    }
    size_t off = 0;
    bool   all = true;
    for (size_t j = 0; j < n_backends; j++) {
        ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
        ggml_tensor *  simple_tensor  = ggml_backend_meta_buffer_simple_tensor(weight_cpy, j);
        if (simple_tensor == nullptr || simple_backend->iface.moe_cache_gather == nullptr) {
            return false;
        }
        const ggml_tensor * ids_j = ids;
        if (ids_meta) {
            ggml_tensor * si = ggml_backend_meta_buffer_simple_tensor(ids, j);
            if (si == nullptr) {
                return false;
            }
            ids_j = si;
        }
        all = all && simple_backend->iface.moe_cache_gather(
                simple_backend, weight, simple_tensor, ids_j, off, (int) ss.axis);
        off += simple_tensor->nb[ss.axis + 1];
    }
    return all;
}

// UPSTREAM PR #22466 (adapted): fan the loader's per-upload event record/wait out to every simple
// backend so the meta backend can serve as the async-upload backend under `-sm tensor`.
static void ggml_backend_meta_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    auto * ev_ctx = (ggml_backend_meta_event_context *) event->context;
    const size_t n = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(ev_ctx->simple_events.size() == n);
    for (size_t j = 0; j < n; j++) {
        ggml_backend_event_record(ev_ctx->simple_events[j], ggml_backend_meta_simple_backend(backend, j));
    }
}

static void ggml_backend_meta_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    auto * ev_ctx = (ggml_backend_meta_event_context *) event->context;
    const size_t n = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(ev_ctx->simple_events.size() == n);
    for (size_t j = 0; j < n; j++) {
        ggml_backend_event_wait(ggml_backend_meta_simple_backend(backend, j), ev_ctx->simple_events[j]);
    }
}

static const ggml_backend_i ggml_backend_meta_i = {
    /* .get_name                = */ ggml_backend_meta_get_name,
    /* .free                    = */ ggml_backend_meta_free,
    /* .set_tensor_async        = */ ggml_backend_meta_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_meta_get_tensor_async,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ ggml_backend_meta_synchronize,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_meta_graph_compute,
    /* .event_record            = */ ggml_backend_meta_event_record,
    /* .event_wait              = */ ggml_backend_meta_event_wait,
    /* .stage_buffer            = */ nullptr, // the meta backend stages per device instead: stage_input
    /* .stage_upload            = */ nullptr,
    /* .stage_gather            = */ nullptr,
    /* .stage_wait              = */ nullptr,
    /* .stage_d2d               = */ nullptr,
    /* .stage_h2d_gbps          = */ ggml_backend_meta_stage_h2d_gbps,
    /* .stage_input             = */ ggml_backend_meta_stage_input,
    /* .graph_optimize          = */ ggml_backend_meta_graph_optimize,
    /* .moe_cache_update        = */ ggml_backend_meta_moe_cache_update,
    /* .moe_cache_take_over     = */ ggml_backend_meta_moe_cache_take_over,
    /* .moe_cache_band          = */ ggml_backend_meta_moe_cache_band,
    /* .moe_cache_gather        = */ ggml_backend_meta_moe_cache_gather,
};

bool ggml_backend_is_meta(ggml_backend_t backend) {
    return backend != nullptr && backend->iface.get_name == ggml_backend_meta_i.get_name;
}

static ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_meta_context * backend_ctx = new ggml_backend_meta_context(dev, params);

    ggml_backend_t backend = new struct ggml_backend;
    backend->guid    = ggml_backend_meta_guid();
    backend->iface   = ggml_backend_meta_i;
    backend->device  = dev;
    backend->context = backend_ctx;
    return backend;
}

size_t ggml_backend_meta_n_backends(ggml_backend_t meta_backend) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) meta_backend->context;
    return backend_ctx->backend_configs.size();
}

ggml_backend_t ggml_backend_meta_simple_backend(ggml_backend_t meta_backend, size_t index) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) meta_backend->context;
    return backend_ctx->backend_configs[index].backend;
}
