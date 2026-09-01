#pragma once

#include "common.cuh"
#include "ggml-backend-impl.h"

#include <cstddef>

// Opaque pipeline context -- owns all pinned buffers, streams, and events.
struct ggml_cuda_ar_pipeline;

// Allocate a pipeline for n_devices GPUs.
// devices[] holds the CUDA device IDs in rank order.
// Returns nullptr on allocation failure.
ggml_cuda_ar_pipeline * ggml_cuda_ar_pipeline_init(
    const int * devices, size_t n_devices);

// Release all resources owned by the pipeline.
void ggml_cuda_ar_pipeline_free(ggml_cuda_ar_pipeline * pipeline);

// Execute an in-place AllReduce (sum) across tensors[0..n_devices-1].
// tensors[i] must live on the device managed by backends[i] and be
// contiguous F32, F16, or BF16.
// Preconditions are checked by the CUDA comm dispatcher before calling this.
// Returns true once the reduction work has been enqueued successfully.
bool ggml_cuda_ar_allreduce(
    ggml_cuda_ar_pipeline * pipeline,
    ggml_backend_t        * backends,
    ggml_tensor           ** tensors);

// ---------------------------------------------------------------------------
// WIP fused-stage hook: the CUDA backend's graph-capture path calls
// ggml_cuda_ar_stage_hook_run() right before cudaStreamEndCapture so the AR
// stage kernel (shard -> wire + arrival token) becomes the captured graph's
// LAST node.  With the stage inside the graph, the wire data + arrival token
// are ready at each device's subgraph-end instead of after the separate AR
// kernel's dispatch (which carries a per-device graph->kernel premium of up
// to ~12 us on the late card).  The reduce kernel validates that the stage
// actually ran and falls back to in-kernel staging otherwise, so prefill /
// warmup / graphs-off are never broken.  Not for integration.
typedef void (*ggml_cuda_ar_stage_fn)(int device, cudaStream_t stream,
                                      const float * data, int64_t count,
                                      void * user_data);

// Register (or clear, fn==nullptr) the stage hook for `device`.
void ggml_cuda_ar_stage_hook_set(int device, ggml_cuda_ar_stage_fn fn, void * user_data);

// Called by the CUDA backend during graph capture.  No-op when no hook is
// registered for `device` (the common case).
void ggml_cuda_ar_stage_hook_run(int device, cudaStream_t stream,
                                 const float * data, int64_t count);
