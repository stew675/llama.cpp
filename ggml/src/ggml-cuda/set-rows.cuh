#pragma once

#include "common.cuh"

#define CUDA_SET_ROWS_BLOCK_SIZE 256

// One q8_0 block quantized by a 32-lane group (lane = position in the block): amax by xor-shuffle fmaxf (an exact
// max, seeded with 0 like quantize_f32_q8_0_block), then the same d, id and roundf, so the bytes are identical.
// Used by k_set_rows_q8_0_group and by the KV-cache writes of the fused rope + FWHT kernel.
static __device__ __forceinline__ void ggml_cuda_quantize_q8_0_group(const float v, block_q8_0 * blk, const int lane) {
    float amax = fmaxf(0.0f, fabsf(v));
#pragma unroll
    for (int offset = QK8_0/2; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset, QK8_0));
    }

    const float d  = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    if (lane == 0) {
        blk->d = d;
    }
    blk->qs[lane] = roundf(v*id);
}

void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
