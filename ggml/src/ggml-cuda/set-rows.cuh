#pragma once

#include "common.cuh"
#include "cpy-utils.cuh"   // best_index_int8 / kvalues_iq4nl, the reference the iq4_nl group reproduces

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

// One q4_0 block from the 32 lanes of one wave (lane = element), bit-identical to quantize_f32_q4_0_block: the scale
// comes from the element its sequential `amax < fabsf(v)` scan keeps -- the FIRST of largest |v| on a tie (an fmaxf
// butterfly for the maximum, then a ballot for its lowest lane), never a NaN (fmaxf skips one and `NaN == amax` is
// false), and vmax = +0 when every |v| is 0 -- and every element is rounded with the reference's two statements,
// lane j packing element j with element j + QK4_0/2.  Used by the KV-cache writes of the fused rope + FWHT kernel
// for a q4_0 cache (GGML_CUDA_KV_FOLD_Q4_0).
// WAVE32 ONLY: the ballot and both shuffles are one 32-lane group, so a 64-lane wave would mix two blocks.  The host
// gates it (ggml_cuda_kv_fold_type_ok checks the running device's warp_size) and the launch sites assert it.
static __device__ __forceinline__ void ggml_cuda_quantize_q4_0_group(const float v, block_q4_0 * blk, const int lane) {
    float amax = fabsf(v);
#pragma unroll
    for (int offset = QK4_0/2; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset, QK4_0));
    }
    const uint64_t at  = (uint64_t) __ballot_sync(0xFFFFFFFFULL, amax > 0.0f && fabsf(v) == amax) & 0xFFFFFFFFULL;
    const int      src = at ? __ffsll((long long) at) - 1 : 0;   // the lowest lane holding the maximum (first on a tie)
    const float    bv  = __shfl_sync(0xffffffff, v, src, QK4_0);
    const float vmax = at ? bv : 0.0f;
    const float d  = vmax / -8;
    const float id = d ? 1.0f/d : 0.0f;

    const float   x     = v*id;   // the reference's two statements
    const uint8_t xi    = min(15, (int8_t)(x + 8.5f));
    const uint8_t xi_hi = __shfl_sync(0xffffffff, (int) xi, lane + QK4_0/2, QK4_0);   // element lane + QK4_0/2 (high nibble)
    if (lane == 0) {
        blk->d = d;
    }
    if (lane < QK4_0/2) {
        blk->qs[lane] = xi | (xi_hi << 4);
    }
}

// One q5_0 block from the 32 lanes of one wave (lane = element), bit-identical to quantize_f32_q5_0_block: the same
// scale as q4_0's group (the sequential scan's FIRST element of largest |v|, a NaN never winning, vmax = +0 when every
// |v| is 0) with the reference's -16 divisor, the same two rounding statements at 5 bits, lane j packing element j with
// element j + QK5_0/2 in qs[j], and the 5th bit of every element in the ONE 32-bit qh word -- which is exactly what a
// ballot over the 32 lanes gives (bit j = lane j's high bit), so the word is written whole by lane 0.  Used by the
// KV-cache writes of the fused rope + FWHT kernel for a q5_0 cache (GGML_CUDA_KV_FOLD_Q5_0).
// WAVE32 ONLY: the ballot and both shuffles are one 32-lane group, so a 64-lane wave would mix two blocks.  The host
// gates it (ggml_cuda_kv_fold_type_ok checks the running device's warp_size) and the launch sites assert it.
static __device__ __forceinline__ void ggml_cuda_quantize_q5_0_group(const float v, block_q5_0 * blk, const int lane) {
    float amax = fabsf(v);
#pragma unroll
    for (int offset = QK5_0/2; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset, QK5_0));
    }
    const uint64_t at  = (uint64_t) __ballot_sync(0xFFFFFFFFULL, amax > 0.0f && fabsf(v) == amax) & 0xFFFFFFFFULL;
    const int      src = at ? __ffsll((long long) at) - 1 : 0;   // the lowest lane holding the maximum (first on a tie)
    const float    bv  = __shfl_sync(0xffffffff, v, src, QK5_0);
    const float vmax = at ? bv : 0.0f;
    const float d  = vmax / -16;
    const float id = d ? 1.0f/d : 0.0f;

    const float   x     = v*id;   // the reference's two statements
    const uint8_t xi    = min(31, (int8_t)(x + 16.5f));
    const uint8_t xi_hi = __shfl_sync(0xffffffff, (int) xi, lane + QK5_0/2, QK5_0);   // element lane + QK5_0/2 (high nibble)
    const uint32_t qh   = (uint32_t) __ballot_sync(0xFFFFFFFFULL, (xi & 0x10) != 0); // bit j = element j's 5th bit
    if (lane == 0) {
        blk->d = d;
        memcpy(blk->qh, &qh, sizeof(qh));
    }
    if (lane < QK5_0/2) {
        blk->qs[lane] = (xi & 0xf) | ((xi_hi & 0xf) << 4);
    }
}

// One iq4_nl block from the 32 lanes of one wave (lane = element), bit-identical to quantize_f32_iq4_nl_block: the same
// first-of-largest-|v| scale seed (vmax / kvalues_iq4nl[0]), every element mapped through best_index_int8, lane j
// packing element j with element j + QK4_NL/2 -- and the reference's SCALE SEARCH, which is a sum of 16 terms
// accumulated in j order, so a butterfly would round differently: lane j computes term j and the 16 terms are then
// broadcast in order and added one at a time (the reference's serial association, reproduced exactly).  Used by the
// KV-cache writes of the fused rope + FWHT kernel for an iq4_nl cache (GGML_CUDA_KV_FOLD_IQ4_NL).
// WAVE32 ONLY: the ballot and every shuffle are one 32-lane group (see ggml_cuda_quantize_q5_0_group).
static __device__ __forceinline__ void ggml_cuda_quantize_iq4_nl_group(const float v, block_iq4_nl * blk, const int lane) {
    float amax = fabsf(v);
#pragma unroll
    for (int offset = QK4_NL/2; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset, QK4_NL));
    }
    const uint64_t at  = (uint64_t) __ballot_sync(0xFFFFFFFFULL, amax > 0.0f && fabsf(v) == amax) & 0xFFFFFFFFULL;
    const int      src = at ? __ffsll((long long) at) - 1 : 0;   // the lowest lane holding the maximum (first on a tie)
    const float    bv  = __shfl_sync(0xffffffff, v, src, QK4_NL);
    const float vmax = at ? bv : 0.0f;
    const float d0 = vmax / kvalues_iq4nl[0];
    const float id = d0 ? 1.0f/d0 : 0.0f;

    const uint8_t xi    = best_index_int8(16, kvalues_iq4nl, v*id);
    const uint8_t xi_hi = __shfl_sync(0xffffffff, (int) xi, lane + QK4_NL/2, QK4_NL);   // element lane + QK4_NL/2
    const float   v_hi  = __shfl_sync(0xffffffff, v, lane + QK4_NL/2, QK4_NL);
    if (lane < QK4_NL/2) {
        blk->qs[lane] = xi | (xi_hi << 4);
    }

    // the reference's scale search, term for term (lane j = the reference's loop index j, lanes >= QK4_NL/2 unused)
    const float v0 = kvalues_iq4nl[xi];
    const float v1 = kvalues_iq4nl[xi_hi];
    const float w0 = v*v;
    const float w1 = v_hi*v_hi;
    const float tqx = w0*v0*v + w1*v1*v_hi;
    const float tq2 = w0*v0*v0 + w1*v1*v1;
    float sumqx = 0.0f;
    float sumq2 = 0.0f;
#pragma unroll
    for (int j = 0; j < QK4_NL/2; ++j) {
        sumqx += __shfl_sync(0xffffffff, tqx, j, QK4_NL);
        sumq2 += __shfl_sync(0xffffffff, tq2, j, QK4_NL);
    }
    if (lane == 0) {
        blk->d = sumq2 > 0 ? sumqx/sumq2 : d0;
    }
}

void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
