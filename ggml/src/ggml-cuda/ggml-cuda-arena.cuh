// Compute-buffer sizing for the graph allocator: the percentage margin and the slab-motivated chunk.
// Split out of ggml-cuda.cu (an upstream hub); declared here, defined in ggml-cuda-arena.cu.
// The MoE expert-cache arena itself lives in moe-expert-cache.cu.
#pragma once

#include "ggml-backend.h"

#include <cstddef>

size_t ggml_backend_cuda_buffer_type_get_compute_margin_pct(ggml_backend_buffer_type_t buft);
size_t ggml_backend_cuda_buffer_type_get_compute_chunk_bytes(ggml_backend_buffer_type_t buft);
