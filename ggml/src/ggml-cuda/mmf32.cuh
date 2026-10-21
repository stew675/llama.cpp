#pragma once

#include "common.cuh"

// mmf32 -- the F32 "skinny" prefill GEMM (block 06).
//
// WHY THIS EXISTS.  A non-quantized weight with more than 16 token columns cannot take `mmf` (there is no
// F32 MMA form on RDNA: `amd_mfma_available` is CDNA-only and `amd_wmma_available` covers f16/bf16), so on
// RDNA it falls through `ggml_cuda_mul_mat_cublas()` to rocBLAS, which dispatches to hipBLASLt.  That path
// costs twice: hipBLASLt picks its solution per process start from a heap-ordered container, so the prefill
// arithmetic -- and therefore the logits -- changes between fresh runs (GREEDY-PURITY.md §44: 3 logit
// classes in 8 starts on 35B-A3B), and the solution the heuristic lands on is the *slowest* one in the pool
// for these shapes (measured: 0.13-0.19 ms vs 0.046 ms for a well-chosen one).
//
// WHAT IT IS.  A plain SIMT tiled GEMM for exactly the shapes that hit that path: a small output-row count
// (m <= 256: the GDN ssm_alpha/ssm_beta projections and the MoE routers) against a prefill-width batch
// (n >= 17).  The k range is split across blocks into contiguous chunks and the chunks are summed in
// increasing z, so the whole GEMM is a fixed reduction tree -- no atomics, no data-dependent scheduling, and
// therefore the same bits on every run by construction.  There is no approximation anywhere: it is an F32
// FMA accumulation, the same arithmetic in a different (pinned) order.
//
// LAYOUTS (guaranteed by the caller, ggml_cuda_mul_mat_cublas_impl's F32 single-matrix branch):
//   x   : weight,      ggml src0, ne = {k, m}, nb0 = 4  -> element (row gm, k) at x[gm*k + k]
//   y   : activations, ggml src1, ne = {k, n}, nb0 = 4  -> element (k, column gn) at y[gn*k + k]
//   dst : output,      ggml dst,  ne = {m, n}, nb0 = 4  -> element (gm, gn) at dst[gn*ldc + gm], ldc == m
// beta is 0: dst is written, never read.
//
// SCOPE.  F32 weights only, and only the shape window in ggml_cuda_mmf32_supported(); every other mul_mat
// keeps the rocBLAS path unchanged.  BF16/f16 prefill (Flash-Next, block 05's MMB/FA surfaces) is untouched
// -- so is hipBLASLt's large advantage there.  A *full*-F32 model's FFN GEMMs (m >> 256) are also left on
// rocBLAS: that is a recorded limitation (patches/README.md, block 06).
//
// ENABLEMENT (default-on policy).  ON by default on RDNA4 (the validated device for this kernel) and
// disabled by `GGML_CUDA_MMF32=0`; on RDNA3/RDNA3_5 it is opt-in with `GGML_CUDA_MMF32=1` until those boxes
// have run the same determinism + bench pair (ENVIRONMENT.md §10).  `GGML_CUDA_MMF32_LOG=1` prints the
// resolved plan (a diagnostic; it does not change what is computed).

bool ggml_cuda_mmf32_enabled(int cc);

// True when this shape is served by the kernel (cheap: called once per F32 mul_mat).
bool ggml_cuda_mmf32_supported(int cc, int64_t m, int64_t n, int64_t k, int64_t ldc);

// Launch.  Requires ggml_cuda_mmf32_supported() to have returned true.
void ggml_cuda_mul_mat_f32_skinny(ggml_backend_cuda_context & ctx, int64_t m, int64_t n, int64_t k,
                                  int64_t ldc, const float * x, const float * y, float * dst);
