#pragma once

#include "common.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// The attention output's Hadamard rotation (64-wide FWHT) + Q8_1 publish for the output projection, offered by the fusion
// dispatcher to the FLASH_ATTN_EXT it runs next (reshape: the rotation's RESHAPE that the projection reads); done: the
// combine kernel did it.  Set and cleared around one ggml_cuda_flash_attn_ext call on the dispatching thread.
struct ggml_cuda_fa_had_req {
    const ggml_tensor * had     = nullptr;
    const ggml_tensor * reshape = nullptr;
    bool                done    = false;
};
extern thread_local ggml_cuda_fa_had_req ggml_cuda_fa_had;

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst);
