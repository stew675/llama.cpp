#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

// wip/flash-next-prefill: the H2D transport's buffer COUNT -- how many host-weight sets the staging pipeline
// holds.  ONE number, because the scheduler's pipeline depth, this backend's stage slots and the slab's transport
// region (and, through the `slab_stage_slots` device hook, the Meta backend's per-device depth) must agree: a
// region carved for fewer buffers than the pipeline uses makes `ggml_cuda_slab_arena_alloc_transient` refuse the
// fills, and every staged input falls back to the scheduler's serial upload.
//
// It is chosen LATE, from the MoE expert cache's SETTLED arena (the residency rule in
// `ggml_cuda_moe_transport_decide`), not from the free VRAM at startup: the buffers are a carve INSIDE the slab,
// so their cost is the decode hot set they take out of the arena.  `ggml_backend_cuda_transport_bufs_set` is the
// only writer and ignores a second, different value (a count that follows every arena re-size flaps).
// `GGML_SCHED_PIPELINE_BUFS` overrides the count -- a user override always wins, and `set` then declines.
GGML_BACKEND_API int  ggml_backend_cuda_transport_bufs(void);
GGML_BACKEND_API void ggml_backend_cuda_transport_bufs_set(int n);

#ifdef  __cplusplus
}
#endif
