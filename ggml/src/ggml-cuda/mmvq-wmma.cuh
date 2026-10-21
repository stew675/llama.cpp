#pragma once
#include "common.cuh"

// The speculative VERIFY band (2..16 columns) as a WMMA iu8 16x16x16 family on RDNA4 (block 07, r290).
//
// Each weight is dequantized once for all the columns, so the cost is nearly flat in the column count where
// the K-split mmvq kernel becomes instruction bound (7-8 columns).  Q4_K, Q5_K, Q6_K, IQ4_XS and Q8_0 are
// covered, read in their original GGUF layout, on the existing q8_1 activations.
//
// OPT-IN and width-split by design: 1 column keeps the K-split kernel.  With it enabled the delivery's
// byte-identical greedy contract no longer holds (the accumulation order differs, ~3e-7 relative), which is
// why the default is off.  Returns false for every case it does not cover (type, shape, GPU, floor); the
// caller then takes the usual path.
bool ggml_cuda_mmvq_wmma(ggml_type type, const void * vx, const void * vy, float * dst,
                         int64_t ncols_x, int64_t nrows_x, int64_t stride_row_x, int ncols_dst,
                         int64_t stride_col_y, int64_t stride_col_dst, cudaStream_t stream);

// The floor in columns: the first verify width that takes this family (0 = off).  Set from `--wmma-verify`
// through ggml_backend_dev_wmma_verify_set(); 2..16 after clamping (1 column is decode and always keeps the
// K-split kernel).  `GGML_CUDA_WMMA_MMVQ` is the contributor's environment alias and wins when it is set.
void ggml_cuda_mmvq_wmma_set_min_cols(int n);
