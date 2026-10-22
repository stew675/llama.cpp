// MoE expert-cache callbacks + op-offload staging hooks and the FA/H2D staging accounting.
// Split out of ggml-cuda.cu (an upstream hub); declared here, defined in
// ggml-cuda-moe-cache-hooks.cu.  The thin device-API wrappers (which touch the private
// ggml_backend_cuda_device_context) stay in ggml-cuda.cu.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>

void * ggml_backend_cuda_stage_buffer(ggml_backend_t backend, int slot, size_t size);
void   ggml_backend_cuda_stage_upload(ggml_backend_t backend, void * dst, const void * data, size_t size, ggml_backend_event_t ev);
bool   ggml_backend_cuda_stage_gather(ggml_backend_t backend, int slot, const void * src, size_t offset, size_t width, size_t stride_src, size_t n_copies, ggml_backend_event_t ev, bool src_pinned);
void   ggml_backend_cuda_stage_wait(ggml_backend_t backend, ggml_backend_event_t ev);
void   ggml_backend_cuda_stage_d2d(ggml_backend_t backend, void * dst, const void * src, size_t size);
float  ggml_backend_cuda_stage_h2d_gbps(ggml_backend_t backend);
bool   ggml_backend_cuda_moe_cache_update(ggml_backend_t backend, const ggml_tensor * weight, const ggml_tensor * weight_cpy, const int32_t * ids, int64_t n_used, int64_t n_tok, size_t ids_nb0, size_t ids_nb1, size_t slice_off, int split_axis, const struct ggml_backend_moe_slice * slices, int n_slices);
bool   ggml_backend_cuda_moe_cache_take_over(ggml_backend_t backend, const ggml_tensor * weight, const ggml_tensor * weight_cpy);
int64_t ggml_backend_cuda_moe_cache_band(ggml_backend_t backend);
bool   ggml_backend_cuda_moe_cache_gather(ggml_backend_t backend, const ggml_tensor * weight, const ggml_tensor * weight_cpy, const ggml_tensor * ids, size_t slice_off, int split_axis);
size_t ggml_backend_cuda_fattn_stage_bound(ggml_backend_t backend);
size_t ggml_backend_cuda_fattn_stage_used(ggml_backend_t backend);
size_t ggml_backend_cuda_h2d_stage_used(ggml_backend_t backend);
size_t ggml_backend_cuda_h2d_stage_bound(ggml_backend_t backend, size_t max_table);
