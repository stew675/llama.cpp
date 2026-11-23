#pragma once

#include "common.cuh"

// TQ1_0 / TQ2_0 (ternary) MUL_MAT.  The result is bit-identical to the CPU backend's x86 AVX2 path
// (ggml_vec_dot_tq{1,2}_0_q8_K after quantize_row_q8_K_ref), for every batch size:
//   - src1 is quantized to Q8_K exactly as quantize_row_q8_K_ref does (no contraction: mul, then add of the rounding constant);
//   - per 256-value block, the integer dot is split into the CPU's eight 32-bit lanes (lane L takes the values whose index
//     mod 32 is 4L..4L+3, minus the Q8_K bsums of values 32L..32L+31), each lane accumulated over the blocks with one
//     UNFUSED mul and add by (y.d * x.d) -- the CPU's mul/add is not contracted into an fma, see tq.cu -- and the eight
//     lanes summed in hsum_float_8's order.
bool ggml_cuda_mul_mat_tq_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

void ggml_cuda_mul_mat_tq(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// Decode (one src1 column): whether mm can take the vec kernel, and the vec kernel for n = 1..3 such matmuls over the
// same src1 in one launch (the rows of each unchanged); or with glu (n = 2: gate, up) the SWIGLU of the two, written to
// glu; or with add (n = 1) the ADD of the result and add's other operand, written to add.  The fused results are those
// of the separate ops on this backend, bit for bit.
bool ggml_cuda_mul_mat_tq_vec_ok(const ggml_tensor * mm);

void ggml_cuda_mul_mat_tq_vec(ggml_backend_cuda_context & ctx, const ggml_tensor * const * mm, int n, ggml_tensor * glu,
                              ggml_tensor * add);
