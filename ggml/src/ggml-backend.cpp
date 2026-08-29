// Note: porting this file to C++ is a work in progress

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"

#include <assert.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <unordered_map>
#include <vector>

// getenv() is cheap on Linux but takes a lock and rescans the environment block on Windows.  These
// debug / A-B gates sit on per-graph and per-split paths, so resolve each one once per call site:
// the static inside the immediately-invoked lambda is unique to each macro expansion.
#define GGML_ENV_STR(name) ([]() -> const char * { static const char * v = getenv(name); return v; }())

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#endif


// backend buffer type

const char * ggml_backend_buft_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_name(buft);
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    GGML_ASSERT(buft);
    if (size == 0) {
        // return a dummy buffer for zero-sized allocations
        return ggml_backend_buffer_init(buft, {}, NULL, 0);
    }
    return buft->iface.alloc_buffer(buft, size);
}

// shared planning logic for allocating a list of tensors into one or more buffers of the given type
struct ggml_backend_buft_alloc_buffer_n_plan_item {
    size_t size;  // total bytes for this buffer
    int    first; // first tensor index (inclusive)
    int    last;  // last tensor index (exclusive)
};

using ggml_backend_buft_alloc_buffer_n_plan_t = std::vector<ggml_backend_buft_alloc_buffer_n_plan_item>;

static ggml_backend_buft_alloc_buffer_n_plan_t ggml_backend_buft_alloc_buffer_n_plan(
        ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    ggml_backend_buft_alloc_buffer_n_plan_t plan;

    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    const size_t max_size  = ggml_backend_buft_get_max_size(buft);

    size_t cur_buf_size = 0;
    int    first        = 0;

    for (int i = 0; i < n_tensors; i++) {
        size_t this_size = 0;
        struct ggml_tensor * t = tensors[i];
        if (t->data == NULL && t->view_src == NULL) {
            this_size = GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);
        }

        // flush the current buffer if adding this tensor would exceed max_size
        if (cur_buf_size > 0 && (cur_buf_size + this_size) > max_size) {
            plan.push_back({ cur_buf_size, first, i });
            cur_buf_size = this_size;
            first        = i;
        } else {
            cur_buf_size += this_size;
        }
    }

    if (cur_buf_size > 0) {
        plan.push_back({ cur_buf_size, first, n_tensors });
    }

    return plan;
}

// default implementation of alloc_buffer_n
// allocates tensors from a list into one or more buffers of the given type
static ggml_backend_buffer_t ggml_backend_buft_alloc_buffer_n_default(ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    const ggml_backend_buft_alloc_buffer_n_plan_t plan = ggml_backend_buft_alloc_buffer_n_plan(buft, tensors, n_tensors);

    std::vector<ggml_backend_buffer_t> buffers;
    buffers.reserve(plan.size());

    for (const ggml_backend_buft_alloc_buffer_n_plan_item & item : plan) {
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, item.size);
        if (buffer == NULL) {
            GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), item.size);
            for (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            return NULL;
        }

        struct ggml_tallocr tallocr = ggml_tallocr_new(buffer);

        // allocate tensors in the current buffer
        struct ggml_tensor * t_failed = NULL;
        for (int j = item.first; j < item.last; j++) {
            struct ggml_tensor * t = tensors[j];
            if (t->data == NULL) {
                if (t->view_src == NULL) {
                    if (ggml_tallocr_alloc(&tallocr, t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        break;
                    }
                } else if (t->buffer == NULL) {
                    if (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        break;
                    }
                }
            } else {
                if (t->view_src != NULL && t->buffer == NULL) {
                    // view of a pre-allocated tensor
                    if (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        break;
                    }
                }
            }
        }
        if (t_failed != NULL) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t_failed->name);
            for (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            ggml_backend_buffer_free(buffer);
            return NULL;
        }

        buffers.push_back(buffer);
    }

    if (buffers.empty()) {
        return NULL;
    }

    if (buffers.size() == 1) {
        return buffers[0];
    }

    return ggml_backend_multi_buffer_alloc_buffer(buffers.data(), buffers.size());
}

// default implementation of get_alloc_size_n
// returns the total size that alloc_buffer_n_default would allocate for the given tensors
static size_t ggml_backend_buft_get_alloc_size_n_default(ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    const ggml_backend_buft_alloc_buffer_n_plan_t plan = ggml_backend_buft_alloc_buffer_n_plan(buft, tensors, n_tensors);

    size_t total = 0;
    for (const ggml_backend_buft_alloc_buffer_n_plan_item & item : plan) {
        total += item.size;
    }
    return total;
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer_n(ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    GGML_ASSERT(buft);
    if (buft->iface.alloc_buffer_n) {
        return buft->iface.alloc_buffer_n(buft, tensors, n_tensors);
    }
    return ggml_backend_buft_alloc_buffer_n_default(buft, tensors, n_tensors);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_alignment(buft);
}

size_t ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    // get_max_size is optional, defaults to SIZE_MAX
    if (buft->iface.get_max_size) {
        return buft->iface.get_max_size(buft);
    }
    return SIZE_MAX;
}

size_t ggml_backend_buft_get_compute_margin_pct(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    // optional, defaults to 0 (no slack: the buffer type is unaffected)
    if (buft->iface.get_compute_margin_pct) {
        return buft->iface.get_compute_margin_pct(buft);
    }
    return 0;
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer_usage(ggml_backend_buffer_type_t buft, size_t size, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buft);
    if (size == 0) {
        // same dummy zero-sized buffer ggml_backend_buft_alloc_buffer returns
        return ggml_backend_buffer_init(buft, {}, NULL, 0);
    }
    // optional, defaults to the usage-agnostic allocation
    if (buft->iface.alloc_buffer_usage) {
        return buft->iface.alloc_buffer_usage(buft, size, usage);
    }
    return buft->iface.alloc_buffer(buft, size);
}

size_t ggml_backend_buft_get_compute_chunk_bytes(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    // optional, defaults to 0 (no chunk quantization: the buffer type is unaffected)
    if (buft->iface.get_compute_chunk_bytes) {
        return buft->iface.get_compute_chunk_bytes(buft);
    }
    return 0;
}

size_t ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    GGML_ASSERT(buft);
    // get_alloc_size is optional, defaults to ggml_nbytes
    if (buft->iface.get_alloc_size) {
        size_t size = buft->iface.get_alloc_size(buft, tensor);
        assert(size >= ggml_nbytes(tensor));

        // [TAG_ALLOC_SIZE_EXPAND]
        // if you hit this assert, update ggml_backend_op_alloc_size_may_expand() accordingly
        GGML_ASSERT(size <= ggml_nbytes(tensor) ||
                    ggml_op_is_empty(tensor->op) ||
                    ggml_is_quantized(tensor->type) || // [TAG_ALLOC_SIZE_EXPAND]
                    ggml_op_alloc_size_may_expand(tensor->op));

        return size;
    }
    return ggml_nbytes(tensor);
}

size_t ggml_backend_buft_get_alloc_size_n(ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    GGML_ASSERT(buft);
    if (buft->iface.get_alloc_size_n) {
        return buft->iface.get_alloc_size_n(buft, tensors, n_tensors);
    }
    return ggml_backend_buft_get_alloc_size_n_default(buft, tensors, n_tensors);
}

bool ggml_backend_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    if (buft->iface.is_host) {
        return buft->iface.is_host(buft);
    }
    return false;
}

ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->device;
}

bool ggml_backend_dev_moe_cache_preflight(ggml_backend_dev_t dev, size_t host_expert_bytes, size_t aux_reserve_bytes) {
    if (dev == nullptr || dev->iface.moe_cache_preflight == nullptr) {
        return false;   // backend has no MoE expert cache
    }
    return dev->iface.moe_cache_preflight(dev, host_expert_bytes, aux_reserve_bytes);
}

void ggml_backend_dev_moe_cache_set_reserve(ggml_backend_dev_t dev, size_t bytes) {
    // WIP r42 (TODO #42): see ggml_backend_dev_moe_cache_preflight.
    if (dev == nullptr || dev->iface.moe_cache_set_reserve == nullptr) {
        return;
    }
    dev->iface.moe_cache_set_reserve(dev, bytes);
}

bool ggml_backend_dev_moe_cache_stats(ggml_backend_dev_t dev, int64_t * hits, int64_t * misses, int64_t * arena_bytes) {
    if (dev == nullptr || dev->iface.moe_cache_stats == nullptr) {
        return false;
    }
    return dev->iface.moe_cache_stats(dev, hits, misses, arena_bytes);
}

bool ggml_backend_dev_moe_cache_rearm(ggml_backend_dev_t dev) {
    // OPEN 2 (TODO #42): re-arm the expert-cache arena after a compute-buffer drop returned the VRAM.
    if (dev == nullptr || dev->iface.moe_cache_rearm == nullptr) {
        return false;
    }
    return dev->iface.moe_cache_rearm(dev);
}

size_t ggml_backend_dev_slab_work_size(ggml_backend_dev_t dev) {
    // OPEN 2 (TODO #42): 0 when this backend has no movable-boundary slab (or never set the hook), which is
    // how the caller tells "no slab to reclaim the wide layout with" from "slab, and it is this wide".
    if (dev == nullptr || dev->iface.slab_work_size == nullptr) {
        return 0;
    }
    return dev->iface.slab_work_size(dev);
}

// backend buffer

ggml_backend_buffer_t ggml_backend_buffer_init(
               ggml_backend_buffer_type_t buft,
        struct ggml_backend_buffer_i      iface,
               void *                     context,
               size_t                     size) {
    ggml_backend_buffer_t buffer = new ggml_backend_buffer {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ GGML_BACKEND_BUFFER_USAGE_ANY
    };

    return buffer;
}

const char * ggml_backend_buffer_name(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    if (buffer == NULL) {
        return;
    }

    if (buffer->iface.free_buffer != NULL) {
        buffer->iface.free_buffer(buffer);
    }
    delete buffer;
}

size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->size;
}

void * ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    // get_base is optional if the buffer is zero-sized
    if (!ggml_backend_buffer_is_meta(buffer) && buffer->size == 0) {
        return NULL;
    }

    // FIXME JG: a multi_buffer has a non-zero size, according to the above comment get_base is not optional,
    //     I don't know whether the above comment is correct
    if (!buffer->iface.get_base) {
        return NULL;
    }

    void * base = buffer->iface.get_base(buffer);

    GGML_ASSERT(base != NULL && "backend buffer base cannot be NULL");

    return base;
}

enum ggml_status ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    GGML_ASSERT(buffer);
    // init_tensor is optional
    if (buffer->iface.init_tensor) {
        return buffer->iface.init_tensor(buffer, tensor);
    }
    return GGML_STATUS_SUCCESS;
}

void ggml_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    // clear is optional if the buffer is zero-sized
    if (buffer->size == 0) {
        return;
    }

    buffer->iface.clear(buffer, value);
}

size_t ggml_backend_buffer_get_alignment(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_alignment(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_max_size(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_max_size(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor) {
    return ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(buffer), tensor);
}

bool ggml_backend_buffer_is_host(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    buffer->usage = usage;

    // FIXME: add a generic callback to the buffer interface
    if (ggml_backend_buffer_is_multi_buffer(buffer)) {
        ggml_backend_multi_buffer_set_usage(buffer, usage);
    } else if (ggml_backend_buffer_is_meta(buffer)) {
        ggml_backend_meta_buffer_set_usage(buffer, usage);
    }
}

enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->usage;
}

ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->buft;
}

void ggml_backend_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    if (buffer->iface.reset) {
        buffer->iface.reset(buffer);
    }
}

bool ggml_backend_buffer_copy_tensor(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
    if (dst_buf->iface.cpy_tensor) {
        return dst_buf->iface.cpy_tensor(dst_buf, src, dst);
    }
    return false;
}

// backend

ggml_guid_t ggml_backend_guid(ggml_backend_t backend) {
    if (backend == NULL) {
        return NULL;
    }
    return backend->guid;
}

const char * ggml_backend_name(ggml_backend_t backend) {
    if (backend == NULL) {
        return "NULL";
    }
    return backend->iface.get_name(backend);
}

void ggml_backend_free(ggml_backend_t backend) {
    if (backend == NULL) {
        return;
    }

    backend->iface.free(backend);
}

ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_buffer_type(backend->device);
}

ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size) {
    return ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    return ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend));
}

size_t ggml_backend_get_max_size(ggml_backend_t backend) {
    return ggml_backend_buft_get_max_size(ggml_backend_get_default_buffer_type(backend));
}

void ggml_backend_tensor_set_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    if (backend->iface.set_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_set(tensor, data, offset, size);
    } else {
        backend->iface.set_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_get_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    if (backend->iface.get_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(tensor, data, offset, size);
    } else {
        backend->iface.get_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_set_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.set_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set_async(backend, tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    backend->iface.set_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.get_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get_async(backend, tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");
    backend->iface.get_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_set_2d(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.set_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set(tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.get_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get(tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_memset(struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    if (size == 0) {
        return;
    }

    GGML_ASSERT(buf != NULL && "tensor buffer not set");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(buf->iface.memset_tensor != NULL && "memset not implemented by backend buffer");

    buf->iface.memset_tensor(buf, tensor, value, offset, size);
}

void ggml_backend_synchronize(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    if (backend->iface.synchronize == NULL) {
        return;
    }

    backend->iface.synchronize(backend);
}

ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_create != NULL);

    return backend->iface.graph_plan_create(backend, cgraph);
}

void ggml_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_free != NULL);

    backend->iface.graph_plan_free(backend, plan);
}

enum ggml_status ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_compute != NULL);

    return backend->iface.graph_plan_compute(backend, plan);
}

enum ggml_status ggml_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    enum ggml_status err = ggml_backend_graph_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    return err;
}

enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    return backend->iface.graph_compute(backend, cgraph);
}

bool ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_op(backend->device, op);
}

bool ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_buft(backend->device, buft);
}

bool ggml_backend_offload_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_offload_op(backend->device, op);
}

ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return backend->device;
}

// backend copy

void ggml_backend_tensor_copy(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    if (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
    } else if (ggml_backend_buffer_is_host(dst->buffer)) {
        ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));
    } else if (!ggml_backend_buffer_copy_tensor(src, dst)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_buffer_name(src->buffer), ggml_backend_buffer_name(dst->buffer));
#endif // NDEBUG
        size_t nbytes = ggml_nbytes(src);
        void * data = malloc(nbytes);
        ggml_backend_tensor_get(src, data, 0, nbytes);
        ggml_backend_tensor_set(dst, data, 0, nbytes);
        free(data);
    }
}

void ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    GGML_ASSERT(backend_dst);
    if (backend_dst->iface.cpy_tensor_async != NULL) {
        if (backend_dst->iface.cpy_tensor_async(backend_src, backend_dst, src, dst)) {
            return;
        }
    }

    // an async copy would normally happen after all the queued operations on both backends are completed
    // to simulate the same behavior, we need to synchronize both backends first, and do a blocking copy
    ggml_backend_synchronize(backend_src);
    ggml_backend_synchronize(backend_dst);
    ggml_backend_tensor_copy(src, dst);
}

// events

ggml_backend_event_t ggml_backend_event_new(ggml_backend_dev_t device) {
    // null device is allowed for the transition period to the device interface
    if (device == NULL || device->iface.event_new == NULL) {
        return NULL;
    }
    return device->iface.event_new(device);
}

void ggml_backend_event_free(ggml_backend_event_t event) {
    if (event == NULL) {
        return;
    }
    event->device->iface.event_free(event->device, event);
}

void ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_record != NULL);

    backend->iface.event_record(backend, event);
}

void ggml_backend_event_synchronize(ggml_backend_event_t event) {
    GGML_ASSERT(event);
    GGML_ASSERT(event->device->iface.event_synchronize);

    event->device->iface.event_synchronize(event->device, event);
}

void ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_wait != NULL);

    backend->iface.event_wait(backend, event);
}

static void ggml_backend_graph_optimize(ggml_backend_t backend, struct ggml_cgraph * cgraph, struct ggml_backend_graph_optimize_params * params) {
    GGML_ASSERT(backend);
    if (backend->iface.graph_optimize != NULL) {
        backend->iface.graph_optimize(backend, cgraph, params);
    }
}

// Backend device

const char * ggml_backend_dev_name(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_name(device);
}

const char * ggml_backend_dev_description(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_description(device);
}

void ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * free, size_t * total) {
    GGML_ASSERT(device);
    device->iface.get_memory(device, free, total);
}

enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_type(device);
}

void ggml_backend_dev_get_props(ggml_backend_dev_t device, struct ggml_backend_dev_props * props) {
    GGML_ASSERT(device);
    memset(props, 0, sizeof(*props));
    device->iface.get_props(device, props);
}

ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->reg;
}

ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, const char * params) {
    GGML_ASSERT(device);
    return device->iface.init_backend(device, params);
}

ggml_backend_buffer_type_t ggml_backend_dev_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_buffer_type(device);
}

ggml_backend_buffer_type_t ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    if (device->iface.get_host_buffer_type == NULL) {
        return NULL;
    }

    return device->iface.get_host_buffer_type(device);
}

ggml_backend_buffer_t ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_ASSERT(device);
    return device->iface.buffer_from_host_ptr(device, ptr, size, max_tensor_size);
}

bool ggml_backend_dev_supports_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    return device->iface.supports_op(device, op);
}

bool ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(device);
    return device->iface.supports_buft(device, buft);
}

bool ggml_backend_dev_offload_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    if (device->iface.offload_op != NULL) {
        return device->iface.offload_op(device, op);
    }

    return false;
}

// Backend (reg)

const char * ggml_backend_reg_name(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_name(reg);
}

size_t ggml_backend_reg_dev_count(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_device_count(reg);
}

ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(reg);
    return reg->iface.get_device(reg, index);
}

void * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_ASSERT(reg);
    if (!reg->iface.get_proc_address) {
        return NULL;
    }
    return reg->iface.get_proc_address(reg, name);
}

// multi-buffer buffer

struct ggml_backend_multi_buffer_context {
    ggml_backend_buffer_t * buffers;
    size_t n_buffers;
};

static void ggml_backend_multi_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_free(ctx->buffers[i]);
    }

    free(ctx->buffers);
    free(ctx);
}

static void ggml_backend_multi_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_clear(ctx->buffers[i], value);
    }
}

static const struct ggml_backend_buffer_i ggml_backend_multi_buffer_i = {
    /* .free_buffer     = */ ggml_backend_multi_buffer_free_buffer,
    /* .get_base        = */ NULL,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ NULL,
    /* .get_tensor      = */ NULL,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_multi_buffer_clear,
    /* .reset           = */ NULL,
};

ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers) {
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) malloc(sizeof(struct ggml_backend_multi_buffer_context));
    ctx->n_buffers = n_buffers;
    ctx->buffers = (ggml_backend_buffer_t *) malloc(n_buffers * sizeof(ggml_backend_buffer_t));

    GGML_ASSERT(ctx->buffers != NULL);

    size_t total_size = 0;
    for (size_t i = 0; i < n_buffers; i++) {
        ctx->buffers[i] = buffers[i];
        total_size += ggml_backend_buffer_get_size(buffers[i]);
    }

    return ggml_backend_buffer_init(buffers[0]->buft, ggml_backend_multi_buffer_i, ctx, total_size);
}

bool ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->iface.free_buffer == ggml_backend_multi_buffer_free_buffer;
}

void ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    GGML_ASSERT(ggml_backend_buffer_is_multi_buffer(buffer));
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_set_usage(ctx->buffers[i], usage);
    }
}

// creates a copy of the tensor with the same memory layout
static struct ggml_tensor * ggml_dup_tensor_layout(struct ggml_context * ctx, const struct ggml_tensor * tensor) {
    struct ggml_tensor * dup = ggml_dup_tensor(ctx, tensor);
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    return dup;
}

static bool ggml_is_view_op(enum ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

// scheduler

#ifndef GGML_SCHED_MAX_BACKENDS
#define GGML_SCHED_MAX_BACKENDS 16
#endif

#ifndef GGML_SCHED_MAX_SPLIT_INPUTS
#define GGML_SCHED_MAX_SPLIT_INPUTS 30
#endif

#ifndef GGML_SCHED_MAX_COPIES
#define GGML_SCHED_MAX_COPIES 4
#endif

struct ggml_backend_sched_split {
    int backend_id;
    int i_start;
    int i_end;
    struct ggml_tensor ** inputs;
    int n_inputs;
    int inputs_capacity;
    // graph view of this split
    struct ggml_cgraph graph;
};

// Ring depth for the op-offload H2D staging prototype (issue #50 WIP).  This is the compile-time
// maximum; the effective depth is GGML_SCHED_STAGE_SLOTS (default 6), clamped to this.
#define GGML_SCHED_STAGE_SLOTS 16
#define GGML_SCHED_STAGE_SLOTS_DEFAULT 8

struct ggml_backend_sched {
    bool is_reset; // true if the scheduler has been reset since the last graph split
    bool is_alloc;

    int n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    struct ggml_hash_set  hash_set;
    int                 * hv_tensor_backend_ids; // [hash_set.size]
    struct ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]

    int * node_backend_ids; // [graph_size]
    int * leaf_backend_ids; // [graph_size]

    int * prev_node_backend_ids; // [graph_size]
    int * prev_leaf_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    struct ggml_cgraph graph;

    // graph splits
    struct ggml_backend_sched_split * splits;
    int n_splits;
    int splits_capacity;

    // pipeline parallelism support
    int n_copies;
    int cur_copy;
    int next_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    struct ggml_tensor ** graph_inputs;
    int n_graph_inputs;
    int graph_inputs_capacity;

    struct ggml_context * ctx;

    ggml_backend_sched_eval_callback callback_eval;
    void * callback_eval_user_data;

    char * context_buffer;
    size_t context_buffer_size;

    bool op_offload;

    // Op-offload H2D staging ring (issue #50 WIP): overlap host->device weight uploads with compute.
    // GGML_SCHED_STAGE=1 enables it; a stage-capable backend is required.  stage_consumed is the
    // per-split list of staged inputs produced by sched_stage_issue and drained by the input loop.
    bool stage_enabled;
    int  stage_slot_next;
    int  stage_n_slots;
    int  stage_consumed_n;
    int  stage_mode; // 0 = stage then D2D into the split input, 1 = point the split input at the slot
    bool stage_split_ok; // this split passed the enable + width gates (a backend-owned `stage_input` reads it)
    // wip/moe-expert-cache (B2): device-side expert gather for an offloaded `MUL_MAT_ID` upload whose
    // width did not qualify for the staging ring.  GGML_SCHED_DEVGATHER=0 opts out.
    bool devgather_enabled;
    struct {
        struct ggml_tensor * dst;
        size_t size;
        int    slot;
        int    backend_id;
        void * orig; // dst->data to restore (redirect mode)
    } stage_consumed[GGML_SCHED_STAGE_SLOTS];
    struct ggml_backend_event * stage_done_ev[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_STAGE_SLOTS];
    struct ggml_backend_event * stage_free_ev[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_STAGE_SLOTS];

    int debug;

    // used for debugging graph reallocations [GGML_SCHED_DEBUG_REALLOC]
    // ref: https://github.com/ggml-org/llama.cpp/pull/17617
    int debug_realloc;
    int debug_graph_size;
    int debug_prev_graph_size;
};

#define hash_id(tensor) ggml_hash_find_or_insert(&sched->hash_set, tensor)
#define tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#define tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#define tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

static void ggml_backend_sched_split_inputs_grow(struct ggml_backend_sched_split * split) {
    int new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    if (split->inputs_capacity > 0) {
        new_cap = 2*split->inputs_capacity;
        GGML_LOG_DEBUG("%s: increasing split inputs capacity from %d to %d\n", __func__, split->inputs_capacity, new_cap);
    }
    auto * pnew = (struct ggml_tensor **) realloc((void *) split->inputs, new_cap * sizeof(struct ggml_tensor *));
    if (pnew == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * sizeof(struct ggml_tensor *));
        GGML_ABORT("failed to grow split inputs container");
    }
    split->inputs = pnew;
    split->inputs_capacity = new_cap;
}

static void ggml_backend_sched_graph_inputs_grow(ggml_backend_sched_t sched) {
    int new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    if (sched->graph_inputs_capacity > 0) {
        new_cap = 2*sched->graph_inputs_capacity;
        GGML_LOG_DEBUG("%s: increasing graph inputs capacity from %d to %d\n", __func__, sched->graph_inputs_capacity, new_cap);
    }
    auto * pnew = (struct ggml_tensor **) realloc((void *) sched->graph_inputs, new_cap * sizeof(struct ggml_tensor *));
    if (pnew == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * sizeof(struct ggml_tensor *));
        GGML_ABORT("failed to grow graph inputs container");
    }
    sched->graph_inputs = pnew;
    sched->graph_inputs_capacity = new_cap;
}

// returns the priority of the backend, lower id is higher priority
static int ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend) {
    for (int i = 0; i < sched->n_backends; i++) {
        if (sched->backends[i] == backend) {
            return i;
        }
    }
    return -1;
}

static int ggml_backend_sched_backend_from_buffer(ggml_backend_sched_t sched, const struct ggml_tensor * tensor, const struct ggml_tensor * op) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (buffer == NULL) {
        return -1;
    }

    // find highest prio backend that supports the buffer type and the op
    for (int i = 0; i < sched->n_backends; i++) {
        if (ggml_backend_supports_buft(sched->backends[i], buffer->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            return i;
        }
    }

#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), ggml_backend_buffer_name(buffer), tensor->name);
#endif

    return -1;
}

#if 0
#define GGML_SCHED_MAX_SPLITS_DEBUG 4096
static char causes[GGML_DEFAULT_GRAPH_SIZE*16 + GGML_SCHED_MAX_SPLITS_DEBUG*GGML_SCHED_MAX_SPLIT_INPUTS][128]; // debug only
#define SET_CAUSE(node, ...) sprintf(causes[hash_id(node)], __VA_ARGS__)
#define GET_CAUSE(node) causes[hash_id(node)]
#else
#define SET_CAUSE(node, ...)
#define GET_CAUSE(node) ""
#endif

// returns the backend that should be used for the node based on the current locations
static bool meta_dev_contains(ggml_backend_dev_t meta_dev, ggml_backend_dev_t simple_dev) {
    if (meta_dev == nullptr || !ggml_backend_dev_is_meta(meta_dev)) {
        return false;
    }
    const size_t n = ggml_backend_meta_dev_n_devs(meta_dev);
    for (size_t i = 0; i < n; i++) {
        if (ggml_backend_meta_dev_simple_dev(meta_dev, i) == simple_dev) {
            return true;
        }
    }
    return false;
}

static int ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, struct ggml_tensor * tensor) {
    // assign pre-allocated nodes to their backend
    int cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
    if (cur_backend_id != -1) {
        SET_CAUSE(tensor, "1.dst");
        return cur_backend_id;
    }

    // view_src
    if (tensor->view_src != NULL) {
        cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor->view_src, tensor);
        if (cur_backend_id != -1) {
            SET_CAUSE(tensor, "1.vsrc");
            return cur_backend_id;
        }
    }

    if (tensor->buffer || (tensor->view_src && tensor->view_src->buffer)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_buffer_name(buffer), ggml_op_name(tensor->op));
    }

    // graph input
    if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        SET_CAUSE(tensor, "1.inp");
        return cur_backend_id;
    }

    // operations with weights are preferably run on the same backend as the weights
    // TODO: there are exceptions (see below) - not an ideal solution
    bool allow = true;

    // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
    allow = allow && tensor->op != GGML_OP_ROPE;

    // skip FLASH_ATTN_EXT since the sinks tensor is too small to choose a based based on it
    allow = allow && tensor->op != GGML_OP_FLASH_ATTN_EXT;

    if (allow) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            const struct ggml_tensor * src = tensor->src[i];
            if (src == NULL) {
                continue;
            }
            if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                int src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
                // check if a backend with higher prio wants to offload the op
                if (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
                    // A per-device host buffer type names the device that owns the weight (the pinned
                    // host buffers are per device: ggml_backend_cuda_device_get_host_buffer_type).
                    // Offload to that device so a host-resident MoE expert op runs on its own layer's
                    // GPU; otherwise the first backend that can offload wins and `-sm layer` routes
                    // every host-expert op to device 0.
                    ggml_backend_dev_t src_buft_dev = ggml_backend_buft_get_device(src->buffer->buft);
                    for (int b = 0; b < src_backend_id; b++) {
                        ggml_backend_dev_t bdev = ggml_backend_get_device(sched->backends[b]);
                        // `-sm tensor`: the split backend is a Meta device that wraps the real GPUs, so
                        // a per-device host buffer's device is never the backend's own device.  Accept a
                        // Meta device that contains it, otherwise the split's offload is skipped and the
                        // host-resident MoE expert op falls back to the CPU (never what we want -- the
                        // CPU path is ~2x slower and the expert cache cannot engage).
                        if (src_buft_dev != nullptr && bdev != src_buft_dev && !meta_dev_contains(bdev, src_buft_dev)) {
                            continue;
                        }
                        if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                            SET_CAUSE(tensor, "1.off");
                            return b;
                        }
                    }
                }
                SET_CAUSE(tensor, "1.wgt%d", i);
                return src_backend_id;
            }
        }
    }

    return -1;
}

static char * fmt_size(size_t size) {
    static char buffer[128];
    if (size >= 1024*1024) {
        snprintf(buffer, sizeof(buffer), "%zuM", size/1024/1024);
    } else {
        snprintf(buffer, sizeof(buffer), "%zuK", size/1024);
    }
    return buffer;
}

static void ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    int cur_split = 0;
    for (int i = 0; i < graph->n_nodes; i++) {
        if (cur_split < sched->n_splits && i == sched->splits[cur_split].i_start) {
            ggml_backend_t split_backend = sched->backends[sched->splits[cur_split].backend_id];
            GGML_LOG_DEBUG("\n## SPLIT #%d: %s # %d inputs", cur_split, ggml_backend_name(split_backend),
                sched->splits[cur_split].n_inputs);
            for (int j = 0; j < sched->splits[cur_split].n_inputs; j++) {
                if (j == 0) {
                    GGML_LOG_DEBUG(": ");
                }
                GGML_LOG_DEBUG("[%s (%5.5s)] ", sched->splits[cur_split].inputs[j]->name,
                    fmt_size(ggml_nbytes(sched->splits[cur_split].inputs[j])));
            }
            GGML_LOG_DEBUG("\n");
            cur_split++;
        }
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        if (sched->debug > 1) {
            ggml_backend_t tensor_backend = ggml_backend_sched_get_tensor_backend(sched, node);
            GGML_LOG_DEBUG("node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:", i, ggml_op_desc(node), node->name,
                fmt_size(ggml_nbytes(node)), tensor_backend ? ggml_backend_name(tensor_backend) : "NULL", GET_CAUSE(node),
                graph->use_counts[ggml_hash_find(&graph->visited_hash_set, node)], node->flags & GGML_TENSOR_FLAG_COMPUTE ? 1 : 0);
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                ggml_backend_t src_backend = ggml_backend_sched_get_tensor_backend(sched, src);
                GGML_LOG_DEBUG(" %20.20s (%5.5s) [%5.5s %8.8s]", src->name,
                    fmt_size(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
            }
            GGML_LOG_DEBUG("\n");
        }
    }
}

// the graph input a tensor ultimately refers to, following view chains, or NULL if it is not (a
// view of) a graph input.  The recurrent-state copy, for one, is only ever read through views.
static struct ggml_tensor * ggml_backend_sched_graph_input(struct ggml_tensor * t) {
    while (t != NULL) {
        if (t->flags & GGML_TENSOR_FLAG_INPUT) {
            return t;
        }
        t = t->view_src;
    }
    return NULL;
}

static bool ggml_backend_sched_buffer_supported(ggml_backend_sched_t sched, struct ggml_tensor * t, int backend_id) {
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    ggml_backend_buffer_type_t buft = NULL;

    if (buf) {
        // the tensor is already allocated
        buft = buf->buft;
    } else {
        // see if the tensor already has a backend assigned, and use the buffer type of that backend
        int tensor_backend_id = tensor_backend_id(t);
        if (tensor_backend_id == -1 && t->view_src) {
            tensor_backend_id = tensor_backend_id(t->view_src);
        }
        if (tensor_backend_id != -1) {
            buft = sched->bufts[tensor_backend_id];
        }
    }

    if (buft != NULL && ggml_backend_buft_is_host(buft) && sched->n_copies <= 1 &&
            ggml_backend_sched_graph_input(t) != NULL) {
        // A graph input that lives in host memory is written by the host thread.  On a device
        // that accepts host buffers (an APU with info.devices[].integrated set), the scheduler
        // would otherwise let the compute backend read it in place: the next ubatch's
        // set_inputs then races the in-flight compute and a torn value can turn an index into an
        // out-of-bounds store (k_set_rows MEMORY_APERTURE_VIOLATION on gfx1151, and the #15034
        // corrupted output before that).  Force the split-input copy so the device reads a
        // stream-ordered device buffer.  Weights are unaffected: they are never
        // GGML_TENSOR_FLAG_INPUT, so zero-copy host weights (the input embeddings) keep working.
        // The view chain is resolved because an input can be reached only through a view - the
        // recurrent-state copy, for one, is.
        return false;
    }

    return buft != NULL && ggml_backend_supports_buft(sched->backends[backend_id], buft);
}

static void ggml_backend_sched_set_if_supported(ggml_backend_sched_t sched, struct ggml_tensor * node, int cur_backend_id, int * node_backend_id) {
    if (ggml_backend_supports_op(sched->backends[cur_backend_id], node)) {
        *node_backend_id = cur_backend_id;
        SET_CAUSE(node, "2.sup");
    }
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
// wip/moe-expert-cache: the layer index a weight tensor belongs to, parsed from its `blk.<N>.` name.
// Used only to find which device OWNS a layer when deciding where a host-weight op should run.

// The MoE expert cache's decode/verify band on `backend`: routed MUL_MAT_ID batches up to this many tokens are
// taken over by the cache instead of the ids readback + used-expert copy.  The backend owns the value (the CUDA
// cache clamps it to its routed-expert MMVQ band per device); a backend without the hook keeps the historical 8.
static int64_t sched_moe_cache_band(ggml_backend_t backend) {
    return backend->iface.moe_cache_band != NULL ? backend->iface.moe_cache_band(backend) : 8;
}

static int moe_name_layer(const struct ggml_tensor * t) {
    if (t->name[0] == '\0') {
        return -1;
    }
    const char * p = strstr(t->name, "blk.");
    if (p == NULL) {
        return -1;
    }
    return atoi(p + 4);
}

void ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    // reset splits
    sched->n_splits = 0;
    sched->n_graph_inputs = 0;
    sched->is_reset = false;

    struct ggml_init_params params = {
        /* .mem_size =   */ sched->context_buffer_size,
        /* .mem_buffer = */ sched->context_buffer,
        /* .no_alloc =   */ true
    };

    ggml_free(sched->ctx);

    sched->ctx = ggml_init(params);
    if (sched->ctx == NULL) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }

    graph->uid = ggml_graph_next_uid();

    // pass 1: assign backends to ops with pre-allocated inputs
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        int * leaf_backend_id = &tensor_backend_id(leaf);
        // do not overwrite user assignments
        if (*leaf_backend_id == -1) {
            *leaf_backend_id = ggml_backend_sched_backend_id_from_cur(sched, leaf);
        }
    }

    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * node_backend_id = &tensor_backend_id(node);
        // do not overwrite user assignments
        if (*node_backend_id == -1) {
            *node_backend_id = ggml_backend_sched_backend_id_from_cur(sched, node);

#if 0
            // src
            if (node->op == GGML_OP_NONE) {
                continue;
            }

            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                int * src_backend_id = &tensor_backend_id(src);
                if (*src_backend_id == -1) {
                    *src_backend_id = ggml_backend_sched_backend_id_from_cur(sched, src);
                }
            }
#endif
        }
    }

    // pass 2: expand current backend assignments
    // assign the same backend to adjacent nodes
    // expand gpu backends (i.e. non last prio) up and down, ignoring cpu (the lowest priority backend)
    // thus, cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops
    // ops unsupported by the backend being expanded will be left unassigned so that they can be assigned later when the locations of its inputs are known
    // expand gpu down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand gpu up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }

    // pass 3: upgrade nodes to higher prio backends with compatible buffer types
    // if the tensor is already in the same buffer type (*) as another higher priority backend, we should move it there
    // however, we also need to verify that the sources are in compatible buffer types
    // (*) the actual requirement is more relaxed, the buffer type of the backend should be supported by all the users of this tensor further down the graph
    // however, this is slow to verify, so we have a more strict requirement that the buffer type is the same
    // this is not uncommon since multiple backends can use host memory, with the same buffer type (eg. BLAS and CPU)
    // additionally, set remaining unassigned nodes to the backend with the most supported inputs
    // only nodes that could not be assigned during expansion due to the backend not supporting the op should be unassigned at this point
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        int * node_backend_id = &tensor_backend_id(node);
        if (*node_backend_id == -1) {
            // unassigned node: find the backend with the most supported inputs
            int n_supported_best = -1;
            for (int b = 0; b < sched->n_backends; b++) {
                if (ggml_backend_supports_op(sched->backends[b], node)) {
                    int n_supported = 0;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) && ggml_backend_sched_buffer_supported(sched, src, b)) {
                            n_supported++;
                        }
                    }
                    if (n_supported > n_supported_best) {
                        n_supported_best = n_supported;
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.best");
                    }
                }
            }
        } else {
            // assigned node: upgrade to higher prio backend if possible
            for (int b = 0; b < *node_backend_id; b++) {
                if (sched->bufts[b] == sched->bufts[*node_backend_id] && ggml_backend_supports_op(sched->backends[b], node)) {
                    bool supported = true;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if (!ggml_backend_sched_buffer_supported(sched, src, b)) {
                            supported = false;
                            break;
                        }
                    }
                    if (supported) {
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.upg");
                        break;
                    }
                }
            }
        }
    }

    // wip/moe-expert-cache: rebalance host-weight ops onto the device that OWNS their layer.
    //
    // A host-resident WEIGHT (the `-ncmoe` experts) makes its op runnable on ANY GPU, and pass 1 has to
    // pick one blindly - so the lowest-index GPU wins and every such op serialises onto device 0.
    // Observed under `-sm layer` with 2 GPUs: all 120 offloaded MoE ops on device 0, device 1 idle, and a
    // cross-device copy in and out for every layer-1 op - the cached 2-GPU path measured SLOWER than the
    // 1-GPU one (Q4_K_M 44.62 vs 56.21 t/s, Q8_0 42.39 vs 52.06).
    //
    // The owning device cannot be read from the op's own data inputs here: at this point they are still
    // unassigned (-1) and pass 4 below would only drag them onto whichever device pass 1 happened to
    // pick.  It is read from the layer's DEVICE-RESIDENT weights instead - the layer split places every
    // non-expert tensor of layer L on L's device, so their buffer type identifies the owner.  This is a
    // pure scheduling decision: no arithmetic changes.
    int layer_dev[512];
    for (int i = 0; i < 512; i++) {
        layer_dev[i] = -1;
    }
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * t = graph->nodes[i];
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * w = t->src[j];
            if (w == NULL || w->buffer == NULL || w->buffer->usage != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                continue;
            }
            if (ggml_backend_buffer_is_host(w->buffer)) {
                continue;   // the offloaded experts themselves
            }
            const int L = moe_name_layer(w);
            if (L < 0 || L >= 512 || layer_dev[L] >= 0) {
                continue;
            }
            for (int b = 0; b < sched->n_backends; b++) {
                if (sched->bufts[b] == w->buffer->buft) {
                    layer_dev[L] = b;
                    break;
                }
            }
        }
    }
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * node_backend_id = &tensor_backend_id(node);
        if (*node_backend_id < 0 || *node_backend_id == sched->n_backends - 1) {
            continue;   // unassigned (-1), or on the CPU (the last backend) - nothing to rebalance
        }
        int  layer = -1;
        bool host_weight = false;
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * s2 = node->src[j];
            if (s2 == NULL || s2->buffer == NULL || s2->buffer->usage != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                continue;
            }
            if (!ggml_backend_buffer_is_host(s2->buffer)) {
                host_weight = false;   // a device-resident weight: leave the normal choice alone
                break;
            }
            host_weight = true;
            layer = moe_name_layer(s2);
            break;
        }
        if (!host_weight || layer < 0) {
            continue;
        }
        // The rebalance serves the decode/verify band (the cache).  At prefill widths it splits the
        // offloaded expert uploads across devices and breaks the staging pipeline: measured -42 % on
        // `-sm layer` ncmoe 40 ub8192 (4511 -> 2625) and -15..-24 % on `-sm tensor`, against pass 1's
        // all-on-device-0 assignment.  Leave prefill on the pass-1 choice.
        if (node->op == GGML_OP_MUL_MAT_ID && node->ne[2] > sched_moe_cache_band(sched->backends[*node_backend_id])) {
            continue;
        }
        const int dst = layer_dev[layer];
        if (dst < 0 || dst == *node_backend_id || dst == sched->n_backends - 1 ||
            !ggml_backend_supports_op(sched->backends[dst], node) ||
            !ggml_backend_offload_op(sched->backends[dst], node)) {
            continue;
        }
        *node_backend_id = dst;
        SET_CAUSE(node, "4.reb");
    }

    // pass 4: assign backends to remaining src from dst and view_src
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * cur_backend_id = &tensor_backend_id(node);
        if (node->view_src != NULL && *cur_backend_id == -1) {
            *cur_backend_id = tensor_backend_id(node->view_src);
            SET_CAUSE(node, "4.vsrc");
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            int * src_backend_id = &tensor_backend_id(src);
            if (*src_backend_id == -1) {
                if (src->view_src != NULL) {
                    // views are always on the same backend as the source
                    *src_backend_id = tensor_backend_id(src->view_src);
                    SET_CAUSE(src, "4.vsrc");
                } else {
                    *src_backend_id = *cur_backend_id;
                    SET_CAUSE(src, "4.cur");
                }
            }
        }
        // if the node is still unassigned, assign it to the first backend that supports it
        for (int b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
            ggml_backend_sched_set_if_supported(sched, node, b, cur_backend_id);
        }
        GGML_ASSERT(*cur_backend_id != -1);
    }

    // pass 5: split graph, find tensors that need to be copied
    {
        int i_split = 0;
        struct ggml_backend_sched_split * split = &sched->splits[0];
        // find the backend of the first split, skipping view ops
        int i = 0;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (!ggml_is_view_op(node->op)) {
                split->backend_id = tensor_backend_id(node);
                break;
            }
        }
        split->i_start = 0;
        split->n_inputs = 0;
        int cur_backend_id = split->backend_id;
        // wip/moe-expert-cache: the layer of the routed expert weights the current split already owns.
        // Used to keep the gate/up/down (and the GLU between them) of ONE layer in one split - see the
        // `need_new_split` suppression below.
        int cur_moe_layer = -1;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];

            if (ggml_is_view_op(node->op)) {
                continue;
            }

            const int node_backend_id = tensor_backend_id(node);

            GGML_ASSERT(node_backend_id != -1); // all nodes should be assigned by now, this can happen if there is no CPU fallback

            // check if we should start a new split based on the sources of the current node
            bool need_new_split = false;
            int  next_moe_layer = -1;
            if (node_backend_id == cur_backend_id && split->n_inputs > 0) {
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    struct ggml_tensor * src = node->src[j];
                    if (src == NULL) {
                        continue;
                    }
                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                        int src_backend_id = tensor_backend_id(src);
                        if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                            // wip/moe-expert-cache: a cache-managed routed expert op (the decode MoE) reads
                            // the cache arena at runtime - the scheduler's `input_cpy` is never filled (the
                            // backend hook takes it over) and only exists to carry the tensor's split state.
                            // Splitting at every one of them fragments the graph into ~3 micro-splits per layer
                            // (122 vs 2 at `-ncmoe 0`), which both starves the GPU with per-op host dispatches
                            // and puts the gate and up in different child graphs so the gate+up+GLU fusion can
                            // never fire.  Keep one layer's routed ops together (the split boundary at the
                            // layer change still lets the input copies be reused across layers).
                            // `ggml_backend_offload_op` is true for a decode `MUL_MAT_ID` exactly when the
                            // cache is enabled (the CUDA offload relaxation), so it is the cache-active test.
                            const bool cache_op = node->op == GGML_OP_MUL_MAT_ID && node->ne[2] <= sched_moe_cache_band(sched->backends[cur_backend_id]) &&
                                ggml_backend_offload_op(sched->backends[cur_backend_id], node);
                            if (cache_op) {
                                const int L = moe_name_layer(src);
                                if (!(cur_moe_layer >= 0 && L == cur_moe_layer)) {
                                    need_new_split  = true;   // first routed op of a new layer
                                    next_moe_layer  = L;
                                }
                            } else {
                                need_new_split = true;
                            }
                            break;
                        }
                    }
                }
            }

            if (node_backend_id != cur_backend_id || need_new_split) {
                split->i_end = i;
                i_split++;
                if (i_split >= sched->splits_capacity) {
                    int old_cap = sched->splits_capacity;
                    sched->splits_capacity *= 2;
                    sched->splits = (ggml_backend_sched_split *)
                        realloc(sched->splits, sched->splits_capacity * sizeof(struct ggml_backend_sched_split));
                    GGML_ASSERT(sched->splits != NULL);
                    for (int k = old_cap; k < sched->splits_capacity; k++) {
                        memset(&sched->splits[k], 0, sizeof(struct ggml_backend_sched_split));
                    }
                }
                split = &sched->splits[i_split];
                split->backend_id = node_backend_id;
                split->i_start = i;
                split->n_inputs = 0;
                cur_backend_id = node_backend_id;
                cur_moe_layer = next_moe_layer;   // -1 unless this split starts at a routed op
            }

            // find inputs that are not on the same backend
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }

                size_t src_id = hash_id(src);
                const int src_backend_id = sched->hv_tensor_backend_ids[src_id];
                GGML_ASSERT(src_backend_id != -1); // all inputs should be assigned by now

                if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    if (tensor_id_copy(src_id, cur_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            if (sched->n_copies > 1) {
                                ggml_set_input(tensor_copy);
                                ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        int n_inputs = split->n_inputs++;
                        if (n_inputs >= split->inputs_capacity) {
                            ggml_backend_sched_split_inputs_grow(split);
                        }
                        split->inputs[n_inputs] = src;
                    } else if (node->op == GGML_OP_MUL_MAT_ID && j == 0 &&
                               src->buffer != NULL && ggml_backend_buffer_get_usage(src->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                        // The copy of these expert weights was created for a consumer in an earlier
                        // split.  That split staged only the experts its own routing selected (or let
                        // the expert cache take the copy over), so a second consumer in a later split
                        // (e.g. the qwen4exp unmasked MTP export, which recomputes the last layer's FFN
                        // on every row) would read experts that were never staged.  Register the
                        // weights as an input of this split as well, so they are staged again for
                        // this split's routing before it runs.
                        bool listed = false;
                        for (int k = 0; k < split->n_inputs; k++) {
                            if (split->inputs[k] == src) {
                                listed = true;
                                break;
                            }
                        }
                        if (!listed) {
                            int n_inputs = split->n_inputs++;
                            if (n_inputs >= split->inputs_capacity) {
                                ggml_backend_sched_split_inputs_grow(split);
                            }
                            split->inputs[n_inputs] = src;
                        }
                    }
                    node->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
                }
            }
        }
        split->i_end = graph->n_nodes;
        sched->n_splits = i_split + 1;
    }

    if (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }

    // pass 6: collect all input tensors into graph_inputs
    //         this includes inputs not consumed by any node (e.g. the embeddings input of a text-only batch) so that
    //         the graph composition does not depend on which inputs are used, which would otherwise cause graph
    //         reallocations when switching between different types of batches [GGML_SCHED_DEBUG_REALLOC]
    if (sched->n_copies > 1) {
        for (int i = 0; i < graph->n_leafs; i++) {
            struct ggml_tensor * leaf = graph->leafs[i];
            if ((leaf->flags & GGML_TENSOR_FLAG_INPUT) == 0) {
                continue;
            }

            const size_t leaf_id = hash_id(leaf);
            const int leaf_backend_id = tensor_backend_id(leaf);
            GGML_ASSERT(leaf_backend_id != -1); // all leafs should be assigned by now

            if (tensor_id_copy(leaf_id, leaf_backend_id, 0) == NULL) {
                ggml_backend_t backend = sched->backends[leaf_backend_id];
                for (int c = 0; c < sched->n_copies; c++) {
                    struct ggml_tensor * tensor_copy;
                    if (c == sched->cur_copy) {
                        tensor_copy = leaf; // use the original tensor as the current copy
                    } else {
                        tensor_copy = ggml_dup_tensor_layout(sched->ctx, leaf);
                        ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), leaf->name, c);
                    }
                    ggml_set_input(tensor_copy);
                    ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                    tensor_id_copy(leaf_id, leaf_backend_id, c) = tensor_copy;
                    SET_CAUSE(tensor_copy, "6.cpy");
                }
            }

            int n_graph_inputs = sched->n_graph_inputs++;
            if (n_graph_inputs >= sched->graph_inputs_capacity) {
                ggml_backend_sched_graph_inputs_grow(sched);
            }
            sched->graph_inputs[n_graph_inputs] = leaf;
        }
    }

    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        int * tmp = sched->node_backend_ids;
        sched->node_backend_ids = sched->prev_node_backend_ids;
        sched->prev_node_backend_ids = tmp;

        tmp = sched->leaf_backend_ids;
        sched->leaf_backend_ids = sched->prev_leaf_backend_ids;
        sched->prev_leaf_backend_ids = tmp;
    }

    // optimize the split graphs and collect the allocation dependencies added by the backends
    // this needs to happen before we make graph_copy, so they are in sync
    // TODO: this may create many small allocations in the scheduler, restructure to use a flat array
    std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> alloc_deps;

    struct ggml_backend_graph_optimize_params opt_params = {
        /* .add_alloc_dep = */ [](void * user_data, ggml_tensor * tensor, ggml_tensor * until) {
            auto & deps = *(std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> *) user_data;
            std::vector<ggml_tensor *> & keep = deps[until];
            if (std::find(keep.begin(), keep.end(), tensor) == keep.end()) {
                keep.push_back(tensor);
            }
        },
        /* .user_data     = */ &alloc_deps,
    };

    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];
        split->graph = ggml_graph_view(graph, split->i_start, split->i_end);

        ggml_backend_graph_optimize(sched->backends[split->backend_id], &split->graph, &opt_params);
    }

    // each dep is added to graph_copy as a GGML_OP_NONE node with the kept tensors as srcs
    int n_dep_nodes = 0;
    for (const auto & it : alloc_deps) {
        n_dep_nodes += (it.second.size() + GGML_MAX_SRC - 1) / GGML_MAX_SRC;
    }

    int total_inputs = sched->n_graph_inputs;
    for (int i = 0; i < sched->n_splits; i++) {
        total_inputs += sched->splits[i].n_inputs;
    }
    int graph_size = std::max(graph->n_nodes, graph->n_leafs) + total_inputs * 2 * sched->n_copies + n_dep_nodes;

    // remember the actual graph_size for performing reallocation checks later [GGML_SCHED_DEBUG_REALLOC]
    sched->debug_prev_graph_size = sched->debug_graph_size;
    sched->debug_graph_size = graph_size;

    if (sched->graph.size < graph_size) {
        sched->graph.size = graph_size;
        sched->graph.nodes = (ggml_tensor **) realloc(sched->graph.nodes, graph_size * sizeof(struct ggml_tensor *));
        sched->graph.leafs = (ggml_tensor **) realloc(sched->graph.leafs, graph_size * sizeof(struct ggml_tensor *));
        GGML_ASSERT(sched->graph.nodes != NULL);
        GGML_ASSERT(sched->graph.leafs != NULL);
    }
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;

    struct ggml_cgraph * graph_copy = &sched->graph;

    int n_dep_nodes_added = 0;

    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        for (int j = 0; j < split->n_inputs; j++) {
            assert(graph_copy->size > (graph_copy->n_nodes + 1));

            struct ggml_tensor * input = split->inputs[j];
            const size_t input_id = hash_id(input);
            struct ggml_tensor * input_cpy = tensor_id_copy(input_id, split->backend_id, sched->cur_copy);

            // add a dependency to the input source so that it is not freed before the copy is done
            struct ggml_tensor * input_dep = ggml_view_tensor(sched->ctx, input);
            input_dep->src[0] = input;
            sched->node_backend_ids[graph_copy->n_nodes] = sched->hv_tensor_backend_ids[input_id];
            graph_copy->nodes[graph_copy->n_nodes++] = input_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
            graph_copy->nodes[graph_copy->n_nodes++] = input_cpy;
        }

        for (int j = split->i_start; j < split->i_end; j++) {
            assert(graph_copy->size > graph_copy->n_nodes);
            sched->node_backend_ids[graph_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];

            if (alloc_deps.empty()) {
                continue;
            }

            // add a dependency node so that the kept tensors are not freed before this node is computed
            auto it = alloc_deps.find(graph->nodes[j]);
            if (it != alloc_deps.end()) {
                const std::vector<ggml_tensor *> & keep = it->second;
                for (size_t k = 0; k < keep.size(); k += GGML_MAX_SRC) {
                    struct ggml_tensor * dep = ggml_view_tensor(sched->ctx, keep[k]);
                    for (size_t s = 0; s < GGML_MAX_SRC && k + s < keep.size(); s++) {
                        dep->src[s] = keep[k + s];
                    }
                    assert(graph_copy->size > graph_copy->n_nodes);
                    sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
                    graph_copy->nodes[graph_copy->n_nodes++] = dep;
                    n_dep_nodes_added++;
                }
            }
        }
    }

    // a mismatch means a backend added a dep with an `until` tensor that is not a node of the optimized graph
    GGML_ASSERT(n_dep_nodes_added == n_dep_nodes);

    if (sched->n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        for (int i = 0; i < sched->n_graph_inputs; i++) {
            struct ggml_tensor * input = sched->graph_inputs[i];
            size_t id = hash_id(input);
            int backend_id = tensor_backend_id(input);
            for (int c = 0; c < sched->n_copies; c++) {
                struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                assert(graph_copy->size > graph_copy->n_leafs);
                graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
            }
        }

        for (int i = 0; i < sched->n_splits; i++) {
            struct ggml_backend_sched_split * split = &sched->splits[i];
            int backend_id = split->backend_id;
            for (int j = 0; j < split->n_inputs; j++) {
                struct ggml_tensor * input = split->inputs[j];
                size_t id = hash_id(input);
                for (int c = 0; c < sched->n_copies; c++) {
                    struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                    sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                    assert(graph_copy->size > graph_copy->n_leafs);
                    graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
                }
            }
        }
    }

    // add leafs from the original graph
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        sched->leaf_backend_ids[graph_copy->n_leafs] = tensor_backend_id(leaf);
        assert(graph_copy->size > graph_copy->n_leafs);
        graph_copy->leafs[graph_copy->n_leafs++] = leaf;
    }

    // set ids for all splits
    for (int i = 0; i < sched->n_splits; ++i) {
        sched->splits[i].graph.uid = ggml_graph_next_uid();
    }
}

static bool ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    bool backend_ids_changed = false;
    for (int i = 0; i < sched->graph.n_nodes; i++) {
        if (sched->node_backend_ids[i] != sched->prev_node_backend_ids[i] &&
            sched->bufts[sched->node_backend_ids[i]] != sched->bufts[sched->prev_node_backend_ids[i]]) {
            backend_ids_changed = true;
            break;
        }
    }
    if (!backend_ids_changed) {
        for (int i = 0; i < sched->graph.n_leafs; i++) {
            if (sched->leaf_backend_ids[i] != sched->prev_leaf_backend_ids[i] &&
                sched->bufts[sched->leaf_backend_ids[i]] != sched->bufts[sched->prev_leaf_backend_ids[i]]) {
                backend_ids_changed = true;
                break;
            }
        }
    }

    // allocate graph
    if (backend_ids_changed || !ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_changed);
#endif

        if (sched->debug_realloc > 0) {
            // we are interested only in situations where the graph was reallocated even though its size remained the same [GGML_SCHED_DEBUG_REALLOC]
            // example: https://github.com/ggml-org/llama.cpp/pull/17143
            const bool unexpected = !backend_ids_changed && sched->debug_prev_graph_size == sched->debug_graph_size;

            if (unexpected || sched->debug_realloc > 1) {
                GGML_ABORT("%s: unexpected graph reallocation (graph size = %d, nodes = %d, leafs = %d), debug_realloc = %d\n", __func__,
                        sched->debug_graph_size, sched->graph.n_nodes, sched->graph.n_leafs, sched->debug_realloc);
            }
        }

        // the re-allocation may cause the split inputs to be moved to a different address
        // synchronize without ggml_backend_sched_synchronize to avoid changing cur_copy
        for (int i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }

        if (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
            GGML_LOG_ERROR("%s: failed to reserve graph buffers\n", __func__);
            return false;
        }
        if (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            return false;
        }
    }

    return true;
}

static ggml_backend_event_t sched_stage_ev(ggml_backend_sched_t sched, int backend_id, int slot, bool done) {
    struct ggml_backend_event ** pev = done
        ? &sched->stage_done_ev[backend_id][slot]
        : &sched->stage_free_ev[backend_id][slot];
    if (*pev == NULL) {
        *pev = ggml_backend_event_new(sched->backends[backend_id]->device);
    }
    return *pev;
}

static bool sched_stage_is_host_weight(const struct ggml_tensor * input) {
    return input->buffer != NULL &&
           ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
           ggml_backend_buffer_is_host(input->buffer);
}

// Adaptive gate (issue #50 WIP): whole-tensor staging bypasses the used-expert pruning, so it only
// wins when the batch is wide enough that pruning would not prune much.  The crossover width is
// **link-dependent** (measured: ~2048 tokens at ~14.5 GB/s PCIe5 x4, ~700 at ~25 GB/s PCIe4 x16), so
// the default threshold is calibrated from the backend's measured H2D bandwidth.  An explicit
// GGML_SCHED_STAGE_MIN_TOKENS overrides it (0 = stage for every batch).
static int64_t sched_stage_min_tokens(ggml_backend_sched_t sched) {
    const char * e = getenv("GGML_SCHED_STAGE_MIN_TOKENS");
    if (e != nullptr) {
        return (int64_t) atoll(e);
    }
    static int64_t calibrated = -1;
    if (calibrated < 0) {
        float bw = 0.0f;
        for (int b = 0; b < sched->n_backends; b++) {
            if (sched->backends[b]->iface.stage_h2d_gbps != NULL) {
                bw = sched->backends[b]->iface.stage_h2d_gbps(sched->backends[b]);
                break;
            }
        }
        // Two measured crossover points (issue #50 WIP): ~14.5 GB/s (PCIe5 x4) crosses between 1024
        // and 2048 tokens, ~25 GB/s (PCIe4 x16) below 1024.  Anchor at 1536 for 14.5 GB/s and take a
        // first-order slope; GGML_SCHED_STAGE_MIN_TOKENS overrides it.
        const double t = 1536.0 - 132.0*(double(bw) - 14.5);
        // Floor the gate above the widest verify batch (16), so a decode/verify batch can never stage
        // whole expert tensors even on a link fast enough to push the crossover to 0.  An unknown link
        // (bw <= 0) falls to the same conservative floor, since t is then large anyway.
        calibrated = (int64_t) (t > 64.0 ? t : 64.0);
        GGML_LOG_INFO("%s: H2D staging calibration: %.1f GB/s -> min_tokens=%lld\n",
                      __func__, double(bw), (long long) calibrated);
        if (getenv("GGML_SCHED_STAGE") != nullptr) {
            // ggml's INFO level maps to TRACE verbosity, which is below llama.cpp's default threshold,
            // so a field log would not show which gate this host actually chose (only the messages
            // emitted before llama_log_set installs the filter get through by default).  An explicit
            // GGML_SCHED_STAGE=1 means the user asked for staging and wants to see the decision, so
            // that case also gets a notice at the level that survives by default.  If staging ever
            // becomes default-on this stays silent unless the variable is set.
            GGML_LOG_WARN("%s: H2D staging: %.1f GB/s link -> whole-weight uploads staged from %lld tokens\n",
                          __func__, double(bw), (long long) calibrated);
        }
    }
    return calibrated;
}

static int64_t sched_stage_batch_tokens(const struct ggml_backend_sched_split * split) {
    if (split->graph.n_nodes == 0) {
        return 0;
    }
    const struct ggml_tensor * node = split->graph.nodes[0];
    if (node->op == GGML_OP_MUL_MAT_ID && node->src[2] != NULL) {
        // MUL_MAT_ID: src[2] is the router's expert ids, ne[1] is the token count
        return node->src[2]->ne[1];
    }
    if (node->src[1] != NULL) {
        return node->src[1]->ne[1];
    }
    return 0;
}

// Size of the largest host-weight input of this split (0 when there is none).  The staging/gather
// crossover is a function of the table size, not just the link: the whole-table copy is bytes/bw, so a
// bigger table crosses over at a wider batch.
static size_t sched_stage_host_weight_bytes(const struct ggml_backend_sched_split * split) {
    size_t max_bytes = 0;
    for (int i = 0; i < split->n_inputs; i++) {
        if (!sched_stage_is_host_weight(split->inputs[i])) {
            continue;
        }
        const size_t bytes = ggml_nbytes(split->inputs[i]);
        if (bytes > max_bytes) {
            max_bytes = bytes;
        }
    }
    return max_bytes;
}

// Reference host-table size the link-calibrated base threshold was fitted at (the campaign's
// Qwen3.6-35B-A3B Q4_K_M tables, ~144 MiB).  **Default 0 = the table-size scaling is DISABLED**: the
// width-only, bandwidth-calibrated gate is used.  The scaling (issue #93) was added in the same change
// as the ring-budget fix, while the old fixed 2048 MiB budget was still disabling the ring mid-run, so
// the measurement it was fitted to was confounded.  Re-validated 2026-10-06 on gfx1201 x4 with the ring
// fix and the pinned 2-D H2D in place: staging beats the serial path at every `-ub` from 1024 to 8192
// for both a 450 MiB (Flash-Next IQ3_XXS `-sm layer`) and an 850 MiB (Flash-Next IQ4_XS `-sm tensor`)
// host table, so the scaling only ever turns a win into a loss.  `GGML_SCHED_STAGE_TABLE_REF_MB=<MiB>`
// restores the old scaling for A/B (0 is the same as the default).
static constexpr size_t SCHED_STAGE_TABLE_REF_BYTES = 0;

static int64_t sched_stage_min_tokens_for_bytes(ggml_backend_sched_t sched, size_t bytes) {
    const int64_t base = sched_stage_min_tokens(sched);
    if (base <= 0) {
        return base; // explicit "stage for every batch" (GGML_SCHED_STAGE_MIN_TOKENS=0)
    }
    const char * ref_env = GGML_ENV_STR("GGML_SCHED_STAGE_TABLE_REF_MB");
    if (ref_env != nullptr && atoll(ref_env) == 0) {
        return base; // scaling disabled
    }
    const size_t ref = ref_env != nullptr ? (size_t) atoll(ref_env) * 1024 * 1024
                                          : SCHED_STAGE_TABLE_REF_BYTES;
    if (ref == 0 || bytes == 0) {
        return base;
    }
    const int64_t scaled = base * (int64_t) bytes / (int64_t) ref;
    return scaled > 64 ? scaled : 64; // never below the widest verify batch
}

// The staging/gather crossover for this split (its largest host table).  Below it the gather wins: the
// whole-table ring copy cannot hide behind the short compute, and it forgoes the used-expert pruning.
// When the split has no host-resident weight there is nothing to stage, so return before
// `sched_stage_min_tokens()`: that call runs the one-off H2D bandwidth calibration, which cudaMallocs a
// large probe buffer and (on Windows) does not get the memory back after `cudaFree` (issue #97), so a
// dense full-offload run would strand the buffer for the whole session for no benefit.  The calibration
// is deferred until a split that actually carries a host weight.
static int64_t sched_stage_min_tokens_for(ggml_backend_sched_t sched, const struct ggml_backend_sched_split * split) {
    const size_t bytes = sched_stage_host_weight_bytes(split);
    if (bytes == 0) {
        return 0;
    }
    return sched_stage_min_tokens_for_bytes(sched, bytes);
}

// Issue this split's offloaded host-weight uploads into the staging ring on the copy stream, before
// the split's compute is enqueued.  The upload then overlaps the previous split's compute (the copy
// stream is independent).  The input loop below drains stage_consumed with a device-to-device copy
// into the actual split input after waiting on the per-slot upload event.
// Restore any split input a redirect-mode issue pointed at a ring slot.  The restore must happen
// after the previous split's graph_compute has been called -- the kernels copy the pointer value at
// launch -- and the top of the next issue is exactly that point.
static void sched_stage_restore(ggml_backend_sched_t sched) {
    for (int k = 0; k < sched->stage_consumed_n; k++) {
        if (sched->stage_consumed[k].orig != NULL) {
            sched->stage_consumed[k].dst->data = sched->stage_consumed[k].orig;
            sched->stage_consumed[k].orig       = NULL;
        }
    }
}

// wip/moe-expert-cache (B1): a routed expert table (a MUL_MAT_ID src0) is uploaded far more cheaply by
// the device-side expert gather -- it copies only the routed experts straight from the pinned host
// master into the split input, on the compute stream, with no ring and no whole-table upload -- than by
// whole-tensor H2D staging, which re-uploads all 512 experts on every ubatch (measured: 1814 vs 405
// t/s at `-p 8192 -ub 2048` on a single R9700).  When the backend can gather, defer these inputs to
// the input loop's gather branch instead of staging them.
//
// A *large* expert table was the case where the device gather was expected to win even above the
// staging width gate: whole-shard staging moves the whole table every ubatch while the gather moves
// only the routed experts.  The numbers that originally justified the 224 MiB threshold (qwen4exp
// 450 MiB table gather 3052 vs staging 1403 t/s, Q8_0 272 MiB 3595 vs 3552, Q4_K_M 144 MiB 4291 vs
// 5624) came from a *corrupted* gather pass: NaN routing made it skip expert work, so they are not
// valid.  With the gather made correct (zero every expert head on every gather) it loses to staging
// even for the 450 MiB table on a PCIe 5.0 x16 link (reporter's 2026-10-04 measurement, ~35 % slower),
// which is why the gather stays default-off (`GGML_SCHED_DEVGATHER=1` is an A/B switch only; see
// `wip/moe-mmq-overread/RESOLUTION.md`).  The threshold is retained so A/B runs still select the
// gather for large tables; it is not a claim that the gather wins.
static constexpr size_t SCHED_GATHER_TABLE_MIN_BYTES = (size_t) 224 * 1024 * 1024;

static bool sched_input_gatherable(ggml_backend_sched_t sched, struct ggml_backend_sched_split * split,
                                   const struct ggml_tensor * input_cpy) {
    if (!sched->devgather_enabled) {
        return false;
    }
    if (sched->backends[split->backend_id]->iface.moe_cache_gather == NULL) {
        return false;
    }
    // The gather serves the below-gate band, where `sched_stage_issue` skips whole-shard staging
    // entirely.  At or above the staging width gate, staging (the ring for `-sm layer`, the meta
    // `stage_input` for `-sm tensor`) is the faster path (Q4_K_M ub8192: staging 5283 vs gather 4035;
    // `-sm layer` ncmoe 40 4511 vs 2625), EXCEPT for a large expert table (see above).
    const bool below_gate = sched_stage_min_tokens(sched) > 0 &&
                            sched_stage_batch_tokens(split) < sched_stage_min_tokens(sched);
    const bool big_table  = ggml_nbytes(input_cpy) >= SCHED_GATHER_TABLE_MIN_BYTES;
    if (!below_gate && !big_table) {
        return false;
    }
    for (int ni = 0; ni < split->graph.n_nodes; ni++) {
        const struct ggml_tensor * cand = split->graph.nodes[ni];
        if (cand->op == GGML_OP_MUL_MAT_ID && cand->src[0] == input_cpy && cand->src[2] != NULL &&
            cand->src[2]->ne[1] > sched_moe_cache_band(sched->backends[split->backend_id])) {
            return true;
        }
    }
    return false;
}

// wip/moe-expert-cache (item 3): the device-side expert gather moves only the *routed* experts, while
// whole-tensor staging moves the device's entire shard every ubatch.  The ring path already prefers the
// gather for a routed expert table (`sched_input_gatherable` in `sched_stage_issue`); the meta
// backend's `stage_input` path defers routed expert tables to the gather there too.
static void sched_stage_issue(ggml_backend_sched_t sched, struct ggml_backend_sched_split * split) {
    sched_stage_restore(sched);
    sched->stage_consumed_n = 0;
    sched->stage_split_ok = false;
    if (!sched->stage_enabled || split->n_inputs == 0) {
        return;
    }
    if (sched_stage_min_tokens_for(sched, split) > 0 &&
        sched_stage_batch_tokens(split) < sched_stage_min_tokens_for(sched, split)) {
        return;
    }
    // A large expert table is always gathered (see `sched_input_gatherable`), so it must not be
    // staged above the width gate either: skip the ring/meta hand-off and let the input loop gather.
    for (int i = 0; i < split->n_inputs; i++) {
        if (!sched_stage_is_host_weight(split->inputs[i])) {
            continue;
        }
        struct ggml_tensor * in_cpy = tensor_copy(split->inputs[i], split->backend_id, sched->cur_copy);
        if (sched_input_gatherable(sched, split, in_cpy)) {
            return;
        }
    }
    // The split passed the enable and width gates.  A backend that owns its own staging (the meta
    // backend under -sm tensor) is driven from the input loop through `stage_input`; the ring below
    // needs a stage-capable backend plus one slot per *host-weight* input.  Count only the host-weight
    // inputs: a merged routed-MoE split carries one big expert weight plus dozens of tiny view/ids
    // inputs (measured 31 inputs on qwen4exp), so comparing the raw `n_inputs` to the ring slots
    // skipped staging for the whole split and left its 450 MiB weight to the serial host path.
    sched->stage_split_ok = true;
    ggml_backend_t backend = sched->backends[split->backend_id];
    int n_host_inputs = 0;
    for (int i = 0; i < split->n_inputs; i++) {
        if (!sched_stage_is_host_weight(split->inputs[i])) {
            continue;
        }
        // A routed expert table is deferred to the gather (see sched_input_gatherable).
        struct ggml_tensor * input_cpy = tensor_copy(split->inputs[i], split->backend_id, sched->cur_copy);
        if (sched_input_gatherable(sched, split, input_cpy)) {
            continue;
        }
        n_host_inputs++;
    }
    if (n_host_inputs == 0 || n_host_inputs > sched->stage_n_slots) {
        return;
    }
    if (backend->iface.stage_buffer == NULL) {
        return;
    }

    // Plan the whole split before issuing any upload: a partially staged split is worse than either
    // path (its non-staged input forces the ids readback + a full device sync).  Every host-weight
    // input gets a ring slot; if the ring cannot hold the whole split (GGML_SCHED_STAGE_MAX_MB, or a
    // cudaMalloc failure), skip this split **without disabling the ring** -- a later split may use
    // slots that do fit, so the fallback is a partial pipeline instead of the old cliff where one
    // failed growth killed staging for the rest of the run (issue #93).
    struct stage_plan {
        struct ggml_tensor * input;
        struct ggml_tensor * input_cpy;
        size_t               size;
        int                  slot;
    };
    stage_plan plan[GGML_SCHED_STAGE_SLOTS];
    int n_plan = 0;
    const int slot_first = sched->stage_slot_next;
    for (int i = 0; i < split->n_inputs; i++) {
        struct ggml_tensor * input = split->inputs[i];
        if (!sched_stage_is_host_weight(input)) {
            continue;
        }
        struct ggml_tensor * input_cpy = tensor_copy(input, split->backend_id, sched->cur_copy);
        if (sched_input_gatherable(sched, split, input_cpy)) {
            continue;
        }
        const size_t size = ggml_nbytes(input);
        const int slot = (slot_first + n_plan) % sched->stage_n_slots;
        // The free event must be waited on before the slot is grown/reused: in redirect mode the
        // previous split's kernels may still be reading it, and growing frees the old allocation.
        backend->iface.stage_wait(backend, sched_stage_ev(sched, split->backend_id, slot, false));
        if (backend->iface.stage_buffer(backend, slot, size) == NULL) {
            // Fall back for this split only.  Advance one slot so the next split starts somewhere
            // else; the slots that do fit keep being reused.
            static bool warned = false;
            if (!warned) {
                warned = true;
                GGML_LOG_WARN("%s: H2D staging shortfall: a %zu-byte upload does not fit "
                              "GGML_SCHED_STAGE_MAX_MB; staging what fits, serial for the rest\n",
                              __func__, size);
            }
            sched->stage_slot_next = (slot_first + 1) % sched->stage_n_slots;
            return;
        }
        plan[n_plan].input     = input;
        plan[n_plan].input_cpy = input_cpy;
        plan[n_plan].size      = size;
        plan[n_plan].slot      = slot;
        n_plan++;
    }

    for (int k = 0; k < n_plan; k++) {
        const int    slot = plan[k].slot;
        const size_t size = plan[k].size;
        void * buf = backend->iface.stage_buffer(backend, slot, size); // allocated in the plan pass
        GGML_ASSERT(buf != NULL);
        backend->iface.stage_upload(backend, buf, plan[k].input->data, size,
                                    sched_stage_ev(sched, split->backend_id, slot, true));

        sched->stage_consumed[sched->stage_consumed_n].dst        = plan[k].input_cpy;
        sched->stage_consumed[sched->stage_consumed_n].size       = size;
        sched->stage_consumed[sched->stage_consumed_n].slot       = slot;
        sched->stage_consumed[sched->stage_consumed_n].backend_id = split->backend_id;
        sched->stage_consumed[sched->stage_consumed_n].orig       = NULL;
        if (sched->stage_mode == 1) {
            // point the consuming op at the slot directly: the slot bytes then move once (H2D) instead
            // of twice (H2D + D2D).  Safe because the graph never captures (prefill is multi-token) and
            // the pointer is restored before the next split issues.
            sched->stage_consumed[sched->stage_consumed_n].orig = plan[k].input_cpy->data;
            plan[k].input_cpy->data = buf;
        }
        sched->stage_consumed_n++;
    }
    sched->stage_slot_next = (slot_first + n_plan) % sched->stage_n_slots;
    GGML_LOG_DEBUG("%s: batch_tokens=%lld n_inputs=%d staged=%d\n",
                   __func__, (long long) sched_stage_batch_tokens(split), split->n_inputs, sched->stage_consumed_n);
}

// r35 (default ON; this was the r33 opt-in A/B candidate).  A split input that is (a view of) a user
// graph input is written by the host on every ubatch, so the async H2D path below - which copies
// straight from that host pointer - can read the next ubatch's value, or a torn/racing one.  The
// recurrent-state copy (`rs_s_copy`) is always consumed through such views (`ggml_view_*` does not
// propagate GGML_TENSOR_FLAG_INPUT), so it was taking the async path and racing the host overwrite.
// This forces the synchronous copy for these tensors while leaving the host-resident weight uploads
// on the async/staged path.  Issue #87; the reporter confirmed the fix.
// `GGML_SCHED_SYNC_GRAPH_INPUTS=0` restores the r26 behaviour (A/B / bisect only).
static bool sched_sync_graph_inputs(void) {
    static const bool enabled = [] {
        const char * e = GGML_ENV_STR("GGML_SCHED_SYNC_GRAPH_INPUTS");
        return e == nullptr || atoi(e) != 0;
    }();
    return enabled;
}

static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    struct ggml_backend_sched_split * splits = sched->splits;

    ggml_tensor * prev_ids_tensor = nullptr;
    std::vector<int32_t> ids;
    std::vector<ggml_bitset_t> used_ids;

    // wip/moe-expert-cache: records for the deferred promotion pass (the device-remap fast path took
    // these inputs over before the host read the routing; the routing is read once after the graph and
    // feeds the LFRU admission for the NEXT token).
    struct moe_promote_rec {
        const struct ggml_tensor * weight;
        const struct ggml_tensor * weight_cpy;
        struct ggml_tensor *       ids;
        ggml_backend_t             ids_backend;
        ggml_backend_t             promote_backend;
        int64_t                    n_used;
        int64_t                    n_tok;
        size_t                     nb0;
        size_t                     nb1;
    };
    std::vector<moe_promote_rec> moe_promote_recs;

    int prev_backend_id = -1;

    for (int split_id = 0; split_id < sched->n_splits; split_id++) {
        struct ggml_backend_sched_split * split = &splits[split_id];
        int split_backend_id = split->backend_id;
        ggml_backend_t split_backend = sched->backends[split_backend_id];

        sched_stage_issue(sched, split);

        // ensure the previous split's async work has completed before we start
        // this split, the allocator may have reused buffer regions across splits
        if (split->n_inputs == 0 && prev_backend_id >= 0 && prev_backend_id != split_backend_id) {
            if (sched->events[prev_backend_id][sched->cur_copy] != NULL) {
                ggml_backend_event_synchronize(sched->events[prev_backend_id][sched->cur_copy]);
            } else {
                ggml_backend_synchronize(sched->backends[prev_backend_id]);
            }
        }

        // copy the input tensors to the split backend
        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            ggml_backend_t input_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[input_id]);
            struct ggml_tensor * input = split->inputs[input_id];
            struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);

            if ((input->flags & GGML_TENSOR_FLAG_INPUT) ||
                (sched_sync_graph_inputs() && ggml_backend_sched_graph_input(input) != nullptr)) {
                // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                    ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                } else {
                    ggml_backend_synchronize(split_backend);
                }
                ggml_backend_tensor_copy(input, input_cpy);
            } else {
                // staged input (issue #50 WIP): wait on the ring slot's upload event, D2D it into the
                // real split input, then free the slot.  Both the D2D and the compute are on the main
                // stream, so this stays ordered after the previous split's compute.
                {
                    int stage_k = -1;
                    for (int k = 0; k < sched->stage_consumed_n; k++) {
                        if (sched->stage_consumed[k].dst == input_cpy) {
                            stage_k = k;
                            break;
                        }
                    }
                    if (stage_k >= 0) {
                        const int    b    = sched->stage_consumed[stage_k].backend_id;
                        const int    slot = sched->stage_consumed[stage_k].slot;
                        const size_t size = sched->stage_consumed[stage_k].size;
                        ggml_backend_event_t done_ev = sched_stage_ev(sched, b, slot, true);
                        ggml_backend_event_wait(split_backend, done_ev);
                        if (sched->stage_mode == 0) {
                            // stage-then-D2D mode: copy the slot into the real split input, then free it
                            ggml_backend_event_t free_ev = sched_stage_ev(sched, b, slot, false);
                            void * buf = split_backend->iface.stage_buffer(split_backend, slot, size);
                            split_backend->iface.stage_d2d(split_backend, input_cpy->data, buf, size);
                            ggml_backend_event_record(free_ev, split_backend);
                        }
                        // redirect mode: input_cpy->data already points at the slot; the slot is freed
                        // after the split's compute below
                        continue;
                    }
                }

                // backend-owned staging (issue #50 WIP, `stage_input`): a backend whose consumers do
                // not read this tensor's `data` (the meta backend, where the upload is spliced across
                // devices) stages the input itself and takes over the ordering.  `callback_eval`
                // splits one split into several compute calls, which this hand-off does not model.
                if (sched->stage_split_ok && sched->callback_eval == NULL &&
                    split_backend->iface.stage_input != NULL && sched_stage_is_host_weight(input)) {
                    // Above the staging width gate (`sched_stage_min_tokens`) whole-shard staging
                    // beats the device gather (the campaign's own A/B: Q4_K_M ub8192 staging 5283 vs
                    // gather 4035).  The gather serves the below-gate band, where `sched_stage_issue`
                    // skips staging and the copy loop's gather branch runs.  Do not defer this table.
                    if (split_backend->iface.stage_input(split_backend, input, input_cpy)) {
                        continue;
                    }
                }

                // tripwire (issue #50 WIP): a split input that redirect mode pointed at a ring slot must
                // be consumed by the staged branch above, never by one of the copy paths below.  Catching
                // it here turns a future silent stale-slot read into an abort (reporters' suggestion,
                // PR #51).
                for (int k = 0; k < sched->stage_consumed_n; k++) {
                    GGML_ASSERT(sched->stage_consumed[k].dst != input_cpy &&
                                "H2D staging: a redirected split input reached a copy path");
                }

                // wait for the split backend to finish using the input before overwriting it.  exp37:
                // the meta backend has no event_record/event_wait, so this falls through to a FULL host
                // synchronize of both devices per expert upload, which keeps the host from running ahead.
                // wip/moe-expert-cache: a cache-managed expert input is NOT overwritten - the backend hook
                // fills its own arena and skips the copy - so the wait is deferred to the copy paths below
                // (an all-resident cached decode then no longer synchronizes both devices per MoE op, which
                // measured +25%).
                auto wait_before_overwrite = [&]() {
                    if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                        ggml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy]);
                    } else {
                        ggml_backend_synchronize(split_backend);
                    }
                };

                // when offloading MoE weights, we can reduce the amount of data copied by copying only the experts that are used
                // wip/moe-expert-cache: find THE `MUL_MAT_ID` consumer of this input copy, not just the
                // split's first node.  A cache-merged split holds the layer's routed gate/up/down
                // together, so `nodes[0]` is the gate and only the first input would ever be hooked.
                ggml_tensor * node = nullptr;
                for (int ni = 0; ni < split->graph.n_nodes; ni++) {
                    ggml_tensor * cand = split->graph.nodes[ni];
                    if (cand->op == GGML_OP_MUL_MAT_ID && cand->src[0] == input_cpy) {
                        node = cand;
                        break;
                    }
                }
                // wip/moe-expert-cache (session 7) takeover fast path: an identity table reads the
                // arena with its own routing ids, and a device-remap table builds the remap on the
                // device from a device slot map - either way the host needs no routing.  Take the
                // input over BEFORE the ids readback so the readback, the full device synchronize it
                // forces, the used-expert pruning and the copy are all skipped.  Restricted to the
                // decode/verify band: prefill keeps the staged upload and its direct-reading MMQ
                // fusions (which are not cache-aware).  A device-remap takeover records the pair for
                // the deferred promotion pass below.
                if (node != nullptr && node->ne[2] <= sched_moe_cache_band(split_backend) &&
                    ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                    ggml_backend_buffer_is_host(input->buffer) &&
                    split_backend->iface.moe_cache_take_over != NULL) {
                    bool need_promote = false;
                    if (split_backend->iface.moe_cache_take_over(split_backend, input, input_cpy, &need_promote)) {
                        if (need_promote && node->src[2] != NULL) {
                            // Resolve the routing tensor against the split's input copies exactly as the
                            // eager hook does: `node->src[2]` can be a stale copy-view of a split input,
                            // and reading that pointer back yields garbage (empty `used`, no promotions).
                            ggml_tensor *  ids_tensor  = node->src[2];
                            ggml_backend_t ids_backend = split_backend;
                            for (int i2 = input_id + 1; i2 < split->n_inputs; i2++) {
                                if (ids_tensor == tensor_copy(split->inputs[i2], split_backend_id, sched->cur_copy)) {
                                    ids_tensor  = split->inputs[i2];
                                    ids_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[i2]);
                                    break;
                                }
                            }
                            moe_promote_rec r;
                            r.weight          = input;
                            r.weight_cpy      = input_cpy;
                            r.ids             = ids_tensor;
                            r.ids_backend     = ids_backend;
                            r.promote_backend = split_backend;
                            r.n_used          = ids_tensor->ne[0];
                            r.n_tok           = ids_tensor->ne[1];
                            r.nb0             = ids_tensor->nb[0];
                            r.nb1             = ids_tensor->nb[1];
                            moe_promote_recs.push_back(r);
                        }
                        continue;
                    }
                }

                if (node != nullptr &&
                    ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                    ggml_backend_buffer_is_host(input->buffer)) {

                    const int64_t n_expert   = node->op == GGML_OP_MUL_MAT_ID ? input->ne[2] : input->ne[1];
                    const size_t expert_size = node->op == GGML_OP_MUL_MAT_ID ? input->nb[2] : input->nb[1];

                    // get the ids
                    ggml_tensor * ids_tensor = node->src[2];
                    ggml_backend_t ids_backend = split_backend;

                    if (ggml_nelements(ids_tensor) == 0) {
                        continue;
                    }

                    // if the ids tensor is also an input of the split, it may not have been copied yet to the split backend
                    // in that case, we use the original ids tensor
                    for (int i = input_id + 1; i < split->n_inputs; i++) {
                        if (ids_tensor == tensor_copy(split->inputs[i], split_backend_id, sched->cur_copy)) {
                            ids_tensor = split->inputs[i];
                            ids_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[i]);
                            break;
                        }
                    }

                    // wip/moe-expert-cache (B2): device-side expert gather.  Copy only the routed
                    // experts from the host master into `input_cpy`, reading the routing on the device,
                    // so the host ids readback and its full device synchronize are skipped.  A backend
                    // that cannot serve the layout (or the gate off) falls through to the host path.
                    if (sched_input_gatherable(sched, split, input_cpy) &&
                        ids_tensor->buffer != NULL && !ggml_backend_buffer_is_host(ids_tensor->buffer) &&
                        split_backend->iface.moe_cache_gather != NULL) {
                        // NO `wait_before_overwrite()`: the gather runs on the split backend's COMPUTE
                        // stream (the same one that read `input_cpy` for the previous pass), so the write
                        // is already ordered after that read.  The wait exists for the `set_async` copy
                        // path, whose copies are on the copy stream; skipping it here is what removes the
                        // per-op full device synchronize (SCHEDSYNC 1.12 -> ~0.1 ms/call on `-sm tensor`).
                        if (split_backend->iface.moe_cache_gather(
                                split_backend, input, input_cpy, ids_tensor, 0, -1)) {
                            continue;
                        }
                    }

                    // wip/moe-expert-cache (B2): fallback host path (the gather above did not fire).
                    // The input backend sync is deferred to here so the gather (which reads the static
                    // host master, not the device `input_cpy`) does not pay it: on `-sm tensor` that
                    // sync is a full meta synchronize (~1.5 ms/op when the pipeline is not drained).
                    ggml_backend_synchronize(input_backend);
                    if (ids_tensor != prev_ids_tensor) {
                        ids.resize(ggml_nbytes(ids_tensor) / sizeof(int32_t));
                        ggml_backend_tensor_get_async(ids_backend, ids_tensor, ids.data(), 0, ggml_nbytes(ids_tensor));
                        ggml_backend_synchronize(ids_backend);

                        // find the used experts
                        used_ids.clear();
                        used_ids.resize(ggml_bitset_size(n_expert));
                        for (int64_t i1 = 0; i1 < ids_tensor->ne[1]; i1++) {
                            for (int64_t i0 = 0; i0 < ids_tensor->ne[0]; i0++) {
                                int32_t id = ids[i1 * ids_tensor->nb[1]/sizeof(int32_t) + i0 * ids_tensor->nb[0]/sizeof(int32_t)];
                                GGML_ASSERT(id >= 0 && id < n_expert);
                                ggml_bitset_set(used_ids.data(), id);
                            }
                        }

                        prev_ids_tensor = ids_tensor;
                    }

                    // wip/moe-expert-cache: drive the backend's per-(layer, expert) residency
                    // policy from the true host master (`input`) and the routing (see the iface
                    // comment in ggml-backend-impl.h).  Inert unless the backend implements it and
                    // MOE_EXPERT_CACHE_MIB is set.
                    if (split_backend->iface.moe_cache_update != NULL) {
                        const bool took = split_backend->iface.moe_cache_update(
                                split_backend, input, input_cpy, ids.data(),
                                ids_tensor->ne[0], ids_tensor->ne[1],
                                ids_tensor->nb[0], ids_tensor->nb[1], 0, -1);
                        if (took) {
                            // the cache filled its compact slots and staged the slot-remapped ids; the
                            // consuming op reads the arena (see `moe_cache_get_table`), so the whole
                            // full-tensor expert copy below is skipped for this input.
                            continue;
                        }
                    }

                    // group consecutive experts and copy them together
                    auto copy_experts = [&](int32_t first_id, int32_t last_id) {
                        const size_t expert_offset = first_id * expert_size;
                        const size_t expert_size_copy =  (last_id - first_id + 1) * expert_size;
                        const size_t padding = std::min<size_t>(expert_size, 512);
                        const size_t padding_end = last_id < n_expert - 1 ? padding : 0;

                        ggml_backend_tensor_set_async(split_backend,
                            input_cpy,
                            (const uint8_t *)input->data + expert_offset, expert_offset,
                            // copy a bit extra at the to ensure there are no NaNs in the padding of the last expert
                            // this is necessary for MMQ in the CUDA backend
                            expert_size_copy + padding_end);
                    };

                    wait_before_overwrite();
                    int id = 0;
                    while (!ggml_bitset_get(used_ids.data(), id)) {
                        id++;
                    }
                    int32_t first_id = id;
                    int32_t last_id = first_id;

                    for (++id; id < n_expert; ++id) {
                        if (!ggml_bitset_get(used_ids.data(), id)) {
                            continue;
                        }

                        if (id == last_id + 1) {
                            last_id = id;
                            continue;
                        }

                        copy_experts(first_id, last_id);

                        first_id = id;
                        last_id = id;
                    }
                    copy_experts(first_id, last_id);
                } else {
                    wait_before_overwrite();
                    // A device-to-device input copy runs on the SOURCE backend's stream (cuda cpy_tensor_async), so it is
                    // not ordered after work already queued on the split backend - including an outbound copy of an
                    // earlier split's output, issued on this backend's stream AFTER that split's event was recorded.  With
                    // the allocator reusing that output's region for this input, the copy could overwrite the output
                    // before it was copied out.  Record a fresh event on the split backend and make the source wait on it.
                    if (input_backend != split_backend && input_backend->iface.event_wait != NULL &&
                        sched->events[split_backend_id][sched->cur_copy] != NULL &&
                        !(input->buffer != NULL && ggml_backend_buffer_is_host(input->buffer))) {
                        ggml_backend_event_record(sched->events[split_backend_id][sched->cur_copy], split_backend);
                        ggml_backend_event_wait(input_backend, sched->events[split_backend_id][sched->cur_copy]);
                    }
                    // A host-resident split input (a graph input or a small CPU-resident
                    // intermediate: `inp_pos`, `attn_inp_k_idxs`, ...) is re-copied on every split of a
                    // merged routed-MoE band (hundreds per pass).  The plain fallback below blocks the
                    // host on `ggml_backend_event_synchronize` for the whole previous split; enqueue the
                    // 1-D H2D on the split backend's compute stream after an in-stream event wait instead.
                    // A simple device backend (event_wait set) takes this; the meta backend's
                    // `set_tensor_async` only understands a whole split tensor, so it keeps the copy below.
                    const bool host_src = input->buffer != NULL && ggml_backend_buffer_is_host(input->buffer);
                    const bool dev_dst  = input_cpy->buffer != NULL && !ggml_backend_buffer_is_host(input_cpy->buffer);
                    if (host_src && dev_dst && split_backend->iface.set_tensor_async != NULL &&
                        split_backend->iface.event_wait != NULL) {
                        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                            ggml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy]);
                        } else {
                            ggml_backend_synchronize(split_backend);
                        }
                        ggml_backend_tensor_set_async(split_backend, input_cpy, input->data, 0, ggml_nbytes(input_cpy));
                    } else
                    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
                    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
                    if (!split_backend->iface.cpy_tensor_async || !split_backend->iface.cpy_tensor_async(input_backend, split_backend, input, input_cpy)) {
                        ggml_backend_synchronize(input_backend);
                        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                        } else {
                            ggml_backend_synchronize(split_backend);
                        }
                        ggml_backend_tensor_copy(input, input_cpy);
                    }
                }
            }
        }

        if (!sched->callback_eval) {
            enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
            if (ec != GGML_STATUS_SUCCESS) {
                return ec;
            }
        } else {
            // similar to ggml_backend_compare_graph_backend
            for (int j0 = 0; j0 < split->graph.n_nodes; j0++) {
                struct ggml_tensor * t = split->graph.nodes[j0];

                // check if the user needs data from this node
                bool need = sched->callback_eval(t, true, sched->callback_eval_user_data);

                int j1 = j0;

                // determine the range [j0, j1] of nodes that can be computed together
                while (!need && j1 < split->graph.n_nodes - 1) {
                    t = split->graph.nodes[++j1];
                    need = sched->callback_eval(t, true, sched->callback_eval_user_data);
                }

                struct ggml_cgraph gv = ggml_graph_view(&split->graph, j0, j1 + 1);

                enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &gv);
                if (ec != GGML_STATUS_SUCCESS) {
                    return ec;
                }

                // TODO: pass backend to the callback, then the user can decide if they want to synchronize
                ggml_backend_synchronize(split_backend);

                if (need && !sched->callback_eval(t, false, sched->callback_eval_user_data)) {
                    break;
                }

                j0 = j1;
            }
        }

        // record the event of this split
        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
            ggml_backend_event_record(sched->events[split_backend_id][sched->cur_copy], split_backend);
        }

        // redirect mode: the split's kernels have been launched, so the slots they read are reusable
        if (sched->stage_mode == 1) {
            for (int k = 0; k < sched->stage_consumed_n; k++) {
                ggml_backend_event_t free_ev = sched_stage_ev(sched, sched->stage_consumed[k].backend_id, sched->stage_consumed[k].slot, false);
                ggml_backend_event_record(free_ev, split_backend);
            }
        }

        prev_backend_id = split_backend_id;
    }

    // wip/moe-expert-cache: deferred promotion.  After the graph, run the LFRU admission + fills for
    // this token's routing; they apply to the NEXT token (the current token was already computed from
    // the pre-promotion device slot map, with a miss served through the UVA cold region).  The routing is
    // read from each table's persistent used-list (`used_dev`), which the remap kernel wrote during the
    // graph - NOT from the graph's routing tensor, whose storage is recycled once the graph completes
    // (a deferred readback of it returns garbage and the promotions no-op).  One synchronize per backend
    // is therefore enough, and there is no per-record D2H.
    if (!moe_promote_recs.empty()) {
        std::vector<ggml_backend_t> syncs;
        for (const moe_promote_rec & r : moe_promote_recs) {
            bool seen = false;
            for (ggml_backend_t b : syncs) {
                if (b == r.promote_backend) { seen = true; break; }
            }
            if (!seen) {
                syncs.push_back(r.promote_backend);
            }
        }
        for (ggml_backend_t b : syncs) {
            ggml_backend_synchronize(b);
        }
        for (const moe_promote_rec & r : moe_promote_recs) {
            if (r.promote_backend->iface.moe_cache_promote != NULL) {
                r.promote_backend->iface.moe_cache_promote(
                        r.promote_backend, r.weight, r.weight_cpy, NULL,
                        r.n_used, r.n_tok, r.nb0, r.nb1);
            }
        }
        // wip/moe-expert-cache: end-of-pass flush for the batched device-side admission policy
        // (MOE_EXPERT_CACHE_DEVPOLICY=1).  A no-op unless that policy is enabled; the per-table calls
        // above only recorded each table's routing shape.  One call per backend; the Meta backend
        // forwards it to each simple device, which runs one policy kernel for all its tables.
        for (ggml_backend_t b : syncs) {
            if (b->iface.moe_cache_promote != NULL) {
                b->iface.moe_cache_promote(b, NULL, NULL, NULL, 0, 0, 0, 0);
            }
        }
    }

    return GGML_STATUS_SUCCESS;
}

ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_buffer_type_t * bufts,
        int n_backends,
        size_t graph_size,
        bool parallel,
        bool op_offload) {
    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);

    struct ggml_backend_sched * sched = (ggml_backend_sched *) calloc(1, sizeof(struct ggml_backend_sched));

    const char * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;

    sched->debug_realloc = 0;
#ifdef GGML_SCHED_NO_REALLOC
    sched->debug_realloc = 1;
#endif
    const char * GGML_SCHED_DEBUG_REALLOC = getenv("GGML_SCHED_DEBUG_REALLOC");
    sched->debug_realloc = GGML_SCHED_DEBUG_REALLOC ? atoi(GGML_SCHED_DEBUG_REALLOC) : sched->debug_realloc;

    sched->n_backends = n_backends;
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
    sched->hash_set    = ggml_hash_set_new(graph_size);
    sched->hv_tensor_backend_ids = (int *) malloc(sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
    sched->hv_tensor_copies      = (ggml_tensor **) malloc(sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));

    const size_t ggml_sched_max_splits = graph_size; // at most there is one split for each node in the graph
    const size_t nodes_size = graph_size + ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2;
    sched->node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->node_backend_ids[0]));
    sched->leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->leaf_backend_ids[0]));
    sched->prev_node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_node_backend_ids[0]));
    sched->prev_leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_leaf_backend_ids[0]));

    sched->debug_graph_size = 0;
    sched->debug_prev_graph_size = 0;

    sched->context_buffer_size = ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*sizeof(struct ggml_tensor) + ggml_graph_overhead_custom(graph_size, false);
    sched->context_buffer = (char *) malloc(sched->context_buffer_size);

    const int initial_splits_capacity = 16;
    sched->splits = (ggml_backend_sched_split *) calloc(initial_splits_capacity, sizeof(sched->splits[0]));
    sched->splits_capacity = initial_splits_capacity;

    sched->graph_inputs_capacity = GGML_SCHED_MAX_SPLIT_INPUTS;
    sched->graph_inputs = (struct ggml_tensor **) calloc(sched->graph_inputs_capacity, sizeof(struct ggml_tensor *));

    for (int b = 0; b < n_backends; b++) {
        sched->backends[b] = backends[b];
        sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_buffer_type(backends[b]);
        GGML_ASSERT(ggml_backend_supports_buft(backends[b], sched->bufts[b]));

        // The per-backend events make the wait before a split's inputs are overwritten an in-stream
        // event wait instead of a FULL device synchronize (which also serialises op-offloaded weight
        // uploads behind the previous split's compute).  Ported from the reporter's PR #51.  Defaulted
        // ON (2026-09-30, r26): with a single graph copy the full synchronize ran thousands of times
        // per offloaded prefill pass (measured 5.2 s at `-ub 8192`, and 861 -> 1072 t/s once the
        // events are created).  `GGML_SCHED_EVENTS=0` opts out.
        static const bool sched_events = getenv("GGML_SCHED_EVENTS") == NULL || atoi(getenv("GGML_SCHED_EVENTS")) != 0;
        if (sched->n_copies > 1 || sched_events) {
            for (int c = 0; c < sched->n_copies; c++) {
                sched->events[b][c] = ggml_backend_event_new(backends[b]->device);
            }
        }
    }

    sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);
    sched->op_offload = op_offload;
    {
        const char * stage_env = getenv("GGML_SCHED_STAGE");
        sched->stage_enabled = stage_env != NULL && atoi(stage_env) != 0;
        const char * mode_env = getenv("GGML_SCHED_STAGE_MODE");
        sched->stage_mode = mode_env != NULL ? atoi(mode_env) : 1;
        const char * slots_env = getenv("GGML_SCHED_STAGE_SLOTS");
        sched->stage_n_slots = slots_env != NULL ? atoi(slots_env) : GGML_SCHED_STAGE_SLOTS_DEFAULT;
        if (sched->stage_n_slots < 1) {
            sched->stage_n_slots = 1;
        }
        if (sched->stage_n_slots > GGML_SCHED_STAGE_SLOTS) {
            sched->stage_n_slots = GGML_SCHED_STAGE_SLOTS;
        }
        // wip/moe-expert-cache (B2): device-side expert gather for offloaded `MUL_MAT_ID` uploads whose
        // width did not qualify for the staging ring.  **Default OFF (2026-10-03):** the gather's one-time
        // expert-head zero does not survive a multi-ubatch prefill - the graph allocator re-uses the
        // graph's `input_cpy` region between ubatches, so the MMQ's tail over-read reads stale NaN and
        // the gather's prefill numbers were corrupt (qwen4exp measured 2650-3060 t/s, which is above the
        // PCIe limit for the bytes actually copied, so it was doing less work).  The staging / host-copy
        // path copies the guard pad on every pass (`copy_experts`: `expert_size_copy + min(expert_size,
        // 512)`) and is correct; measured on the same benchmark it is also not slower: pp8192 `-ub 2048`
        // 655 vs the gather's true 683, `-ub 8192` 1398 vs 1532.  So the gather's apparent 2-4x prefill
        // win over staging was an artifact of the corruption.  `GGML_SCHED_DEVGATHER=1` re-enables it
        // (A/B only - it is not reliable for a multi-ubatch prefill until its destination is made
        // persistent/never-reused).
        const char * devgather_env = GGML_ENV_STR("GGML_SCHED_DEVGATHER");
        sched->devgather_enabled = devgather_env != NULL && atoi(devgather_env) != 0;
        if (sched->stage_enabled) {
            bool stage_capable = false;
            for (int b = 0; b < sched->n_backends; b++) {
                if (sched->backends[b]->iface.stage_buffer != NULL ||
                    sched->backends[b]->iface.stage_input  != NULL) {
                    stage_capable = true;
                    break;
                }
            }
            if (!stage_capable) {
                GGML_LOG_WARN("%s: GGML_SCHED_STAGE=1 but no backend supports it; staging inactive\n",
                              __func__);
            }
        }
    }

    ggml_backend_sched_reset(sched);

    return sched;
}

void ggml_backend_sched_free(ggml_backend_sched_t sched) {
    if (sched == NULL) {
        return;
    }
    for (int b = 0; b < sched->n_backends; b++) {
        for (int c = 0; c < sched->n_copies; c++) {
            ggml_backend_event_free(sched->events[b][c]);
        }
        for (int s = 0; s < GGML_SCHED_STAGE_SLOTS; s++) {
            ggml_backend_event_free(sched->stage_done_ev[b][s]);
            ggml_backend_event_free(sched->stage_free_ev[b][s]);
        }
    }
    ggml_gallocr_free(sched->galloc);
    ggml_free(sched->ctx);
    ggml_hash_set_free(&sched->hash_set);
    for (int i = 0; i < sched->splits_capacity; i++) {
        free(sched->splits[i].inputs);
    }
    free(sched->splits);
    free(sched->graph_inputs);
    free(sched->hv_tensor_backend_ids);
    free(sched->hv_tensor_copies);
    free(sched->node_backend_ids);
    free(sched->leaf_backend_ids);
    free(sched->prev_node_backend_ids);
    free(sched->prev_leaf_backend_ids);
    free(sched->context_buffer);
    free(sched->graph.nodes);
    free(sched->graph.leafs);
    free(sched);
}

void ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    // reset state for the next run
    if (!sched->is_reset) {
        ggml_hash_set_reset(&sched->hash_set);
        memset(sched->hv_tensor_backend_ids, -1, sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
        memset(sched->hv_tensor_copies,       0, sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));
        sched->is_reset = true;
    }
    sched->is_alloc = false;
}

void ggml_backend_sched_reserve_size(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph, size_t * sizes) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);
    GGML_ASSERT(sizes);

    ggml_backend_sched_reset(sched);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    ggml_gallocr_reserve_n_size(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids, sizes);
}

bool ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    if (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
        return false;
    }

    ggml_backend_sched_reset(sched);

    return true;
}

bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    sched->cur_copy = sched->next_copy;
    sched->next_copy = (sched->next_copy + 1) % sched->n_copies;

    ggml_backend_sched_split_graph(sched, graph);

    if (!ggml_backend_sched_alloc_splits(sched)) {
        return false;
    }

    sched->is_alloc = true;

    return true;
}

enum ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    enum ggml_status err = ggml_backend_sched_graph_compute_async(sched, graph);
    ggml_backend_sched_synchronize(sched);
    return err;
}

enum ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    if (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    if (!sched->is_alloc) {
        if (!ggml_backend_sched_alloc_graph(sched, graph)) {
            return GGML_STATUS_ALLOC_FAILED;
        }
    }

    // wip/moe-expert-cache: the routed expert tables read the cache arena, so no calibration pass is
    // needed here; the cache's own deferred promotion runs after the graph.
    return ggml_backend_sched_compute_splits(sched);
}

void ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    for (int i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
    }
    if (!sched->is_alloc) {
        // if the graph is not already allocated, always use copy 0 after a synchronization
        // this ensures that during generation the same copy is used every time,
        // which avoids changes in the graph that could cause CUDA or other graphs to be disabled
        sched->next_copy = 0;
    }
}

void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data) {
    GGML_ASSERT(sched);
    sched->callback_eval = callback;
    sched->callback_eval_user_data = user_data;
}

int ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_splits;
}

int ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_copies;
}

int ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_backends;
}

ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, int i) {
    GGML_ASSERT(sched);
    GGML_ASSERT(i >= 0 && i < sched->n_backends);
    return sched->backends[i];
}

ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return sched->bufts[backend_index];
}

size_t ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return ggml_gallocr_get_buffer_size(sched->galloc, backend_index);
}

void ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);
    tensor_backend_id(node) = backend_index;
    SET_CAUSE(node, "usr");
    sched->is_reset = false;
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node) {
    GGML_ASSERT(sched);
    int backend_index = tensor_backend_id(node);
    if (backend_index == -1) {
        return NULL;
    }
    return sched->backends[backend_index];
}

// utils

bool ggml_op_alloc_size_may_expand(enum ggml_op op) {
    switch (op) {
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_CUMSUM:
        case GGML_OP_ARGSORT:
        case GGML_OP_TOP_K:
            return true;
        default:
            return false;
    }
}

enum ggml_status ggml_backend_view_init(struct ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->view_src != NULL);
    GGML_ASSERT(tensor->view_src->buffer != NULL);
    GGML_ASSERT(tensor->view_src->data != NULL);

    tensor->buffer = tensor->view_src->buffer;
    tensor->data = (char *)tensor->view_src->data + tensor->view_offs;
    return ggml_backend_buffer_init_tensor(tensor->buffer, tensor);
}

enum ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->data == NULL);
    GGML_ASSERT(tensor->view_src == NULL);
    GGML_ASSERT(addr >= ggml_backend_buffer_get_base(buffer));
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer) ||
        (char *) addr + ggml_backend_buffer_get_alloc_size(buffer, tensor) <=
        (char *) ggml_backend_buffer_get_base(buffer) + ggml_backend_buffer_get_size(buffer));

    tensor->buffer = buffer;
    tensor->data = addr;
    return ggml_backend_buffer_init_tensor(buffer, tensor);
}

static struct ggml_tensor * graph_copy_dup_tensor(struct ggml_hash_set hash_set, struct ggml_tensor ** node_copies,
    struct ggml_context * ctx_allocated, struct ggml_context * ctx_unallocated, struct ggml_tensor * src) {

    GGML_ASSERT(src != NULL);
    GGML_ASSERT(src->data && "graph must be allocated");

    size_t id = ggml_hash_insert(&hash_set, src);
    if (id == GGML_HASHSET_ALREADY_EXISTS) {
        return node_copies[ggml_hash_find(&hash_set, src)];
    }

    struct ggml_tensor * dst = ggml_dup_tensor_layout(src->data && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    if (src->view_src != NULL) {
        dst->view_src = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    dst->flags = src->flags;
    memcpy(dst->op_params, src->op_params, sizeof(dst->op_params));
    ggml_set_name(dst, src->name);

    // copy src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        dst->src[i] = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, s);
    }

    node_copies[id] = dst;
    return dst;
}

static void graph_copy_init_tensor(struct ggml_hash_set * hash_set, struct ggml_tensor ** node_copies, bool * node_init, struct ggml_tensor * src) {
    size_t id = ggml_hash_find(hash_set, src);
    if (node_init[id]) {
        return;
    }
    node_init[id] = true;

    struct ggml_tensor * dst = node_copies[id];
    if (dst->view_src != NULL) {
        graph_copy_init_tensor(hash_set, node_copies, node_init, src->view_src);
        enum ggml_status status = ggml_backend_view_init(dst);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS);
    }
    else {
        ggml_backend_tensor_copy(src, dst);
    }

    // init src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        graph_copy_init_tensor(hash_set, node_copies, node_init, s);
    }
}

struct ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, struct ggml_cgraph * graph) {
    GGML_ASSERT(graph);
    struct ggml_hash_set hash_set = ggml_hash_set_new(graph->visited_hash_set.size);
    struct ggml_tensor ** node_copies = (ggml_tensor **) calloc(hash_set.size, sizeof(node_copies[0])); // NOLINT
    bool * node_init = (bool *) calloc(hash_set.size, sizeof(node_init[0]));

    struct ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*hash_set.size + ggml_graph_overhead_custom(graph->size, false),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true
    };

    struct ggml_context * ctx_allocated = ggml_init(params);
    struct ggml_context * ctx_unallocated = ggml_init(params);

    if (ctx_allocated == NULL || ctx_unallocated == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate context for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    // dup nodes
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, node);
    }

    // allocate nodes
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    if (buffer == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    //printf("copy buffer size: %zu MB\n", ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

    // copy data and init views
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_init_tensor(&hash_set, node_copies, node_init, node);
    }

    // build graph copy
    struct ggml_cgraph * graph_copy = ggml_new_graph_custom(ctx_allocated, graph->size, false);
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct ggml_tensor * node_copy = node_copies[ggml_hash_find(&hash_set, node)];
        graph_copy->nodes[i] = node_copy;
    }
    graph_copy->n_nodes = graph->n_nodes;

    ggml_hash_set_free(&hash_set);
    free(node_copies);
    free(node_init);

    return {
        /* .buffer           = */ buffer,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ graph_copy,
    };
}

void ggml_backend_graph_copy_free(struct ggml_backend_graph_copy copy) {
    ggml_backend_buffer_free(copy.buffer);
    ggml_free(copy.ctx_allocated);
    ggml_free(copy.ctx_unallocated);
}

bool ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, struct ggml_cgraph * graph, ggml_backend_eval_callback callback, void * user_data, struct ggml_tensor const * const * test_nodes, size_t num_test_nodes) {
    struct ggml_backend_graph_copy copy = ggml_backend_graph_copy(backend2, graph);
    if (copy.buffer == NULL) {
        return false;
    }

    struct ggml_cgraph * g1 = graph;
    struct ggml_cgraph * g2 = copy.graph;

    assert(g1->n_nodes == g2->n_nodes);

    if (num_test_nodes != 0) {
        GGML_ASSERT(test_nodes);
        // Compute the whole graph and only test the output for specific tensors
        ggml_backend_graph_compute(backend1, g1);
        ggml_backend_graph_compute(backend2, g2);

        bool verified = false;
        for (int i = 0; i < g1->n_nodes; i++) {
            for (size_t j = 0; j < num_test_nodes; ++j) {
                if (g1->nodes[i] == test_nodes[j]) {
                    callback(i, g1->nodes[i], g2->nodes[i], user_data);
                    verified = true;
                }
            }
        }
        GGML_ASSERT(verified);
    } else {
        for (int i = 0; i < g1->n_nodes; i++) {
            struct ggml_tensor * t1 = g1->nodes[i];
            struct ggml_tensor * t2 = g2->nodes[i];

            assert(t1->op == t2->op && ggml_are_same_layout(t1, t2));

            struct ggml_cgraph g1v = ggml_graph_view(g1, i, i + 1);
            struct ggml_cgraph g2v = ggml_graph_view(g2, i, i + 1);

            ggml_backend_graph_compute(backend1, &g1v);
            ggml_backend_graph_compute(backend2, &g2v);

            if (ggml_is_view_op(t1->op)) {
                continue;
            }

            // compare results, calculate rms etc
            if (!callback(i, t1, t2, user_data)) {
                break;
            }
        }
    }
    ggml_backend_graph_copy_free(copy);

    return true;
}

// CPU backend - buffer

static void * ggml_backend_cpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    uintptr_t data = (uintptr_t)buffer->context;

    // align the buffer
    if (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }

    return (void *)data;
}

static void ggml_backend_cpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_aligned_free(buffer->context, buffer->size);
}

static void ggml_backend_cpu_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memset((char *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy((char *)tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy(data, (const char *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static bool ggml_backend_cpu_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(src);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    memset(buffer->context, value, buffer->size);
}

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_i = {
    /* .free_buffer     = */ ggml_backend_cpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_from_ptr_i = {
    /* .free_buffer     = */ NULL, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

static const char * ggml_backend_cpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_cpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);

    if (data == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_cpu_buffer_i, data, size);
}

static size_t ggml_backend_cpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

static bool ggml_backend_cpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_buffer_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ NULL,
            /* .get_alignment       = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size        = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ NULL, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ NULL,
            /* .is_host             = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

static const char * ggml_backend_cpu_buffer_from_ptr_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU_Mapped";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_cpu_buffer_from_ptr_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_buffer_from_ptr_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ NULL,
            /* .get_alignment       = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size        = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ NULL, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ NULL,
            /* .is_host             = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size) {
    GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    return ggml_backend_buffer_init(ggml_backend_cpu_buffer_from_ptr_type(), ggml_backend_cpu_buffer_from_ptr_i, ptr, size);
}
