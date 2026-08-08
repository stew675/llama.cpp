# HIP backend notes

AMD HIP backend (GGML_HIP=ON) notes for kernel-level work. Focus: BF16
support on RDNA3 and newer, flash-attention kernel selection, and how KV
cache types flow through the attention kernels.

## BF16 on RDNA3 and newer (gfx110x, gfx115x, gfx120x)

AMD claims native BF16 on RDNA3+. That is only half true (the RDNA3 ISA
reference, February 2023, lists BF16 in the dot and matrix units but has
no BF16 FMA):

- The vector ALU has no native BF16 FMA. BF16 arithmetic is emulated by the
  compiler as FP32 math: BF16 -> FP32 is a left shift by 16, and the result
  is rounded back to BF16 with an explicit round-to-nearest-even sequence
  plus overflow/denormal handling. Measured on gfx1151 (clang 23): BF16
  scalar FMA is about 12x slower than FP16.
- The matrix and dot units do have native BF16, on RDNA3 as well as
  RDNA3.5/4 (`v_dot2_bf16_bf16` is in the RDNA3 ISA):
  - `v_dot2_f32_bf16` (BF16 dot product, FP32 accumulate)
  - `v_wmma_bf16_16x16x16_bf16` (BF16 matrix multiply)
  Both assemble for gfx1100/gfx1150/gfx1151/gfx1201 (not for gfx906/gfx1030).
  There is no native BF16<->FP16 conversion instruction on these targets.
  RDNA4 (gfx120x) adds sparse matrix variants (`v_swmmac_*_16x16x32_bf16`)
  and packed-BF16 atomic adds (image/flat/global/data-share, e.g.
  `ds_pk_add_bf16`) but still has no BF16 FMA in the vector ALU.

Microbenchmark (gfx1151, ROCm 7.14, 8 independent chains per thread):

| pattern           | FP16 | BF16 | FP32 |
|-------------------|-----:|-----:|-----:|
| scalar/vector FMA | 39.6 | 3.3  | 56.4 | TFLOP/s
| dot, fp32 accum   | 31.3 | 33.3 | -    | TFLOP/s
| WMMA 16x16x16     | ~54  | ~54  | -    | TFLOP/s

WMMA and fp32-accumulating dots run at FP16 parity; scalar BF16 math does
not. Prefer the matrix/dot units for BF16, never vector FMA.

## Flash-attention kernel selection

`ggml_cuda_get_best_fattn_kernel` (ggml/src/ggml-cuda/fattn.cu) picks
between VEC, TILE and MMA_F16:

- `gqa_opt_applies` = gqa_ratio >= 2 && mask && max_bias == 0
  && K->ne[1] % 256 == 0
- `can_use_vector_kernel` = Q->ne[0] <= 256 && Q->ne[0] % 64 == 0
  && Q->ne[0] != 192 && K->ne[1] % 256 == 0
- AMD WMMA gate: `amd_wmma_available && gqa_opt_applies && head_dim <= 128
  && Q->ne[1] * gqa_ratio_eff > 8`
- AMD MFMA gate: CDNA only, not RDNA
- Fallback: TILE

Consequences:

- During decode (ne[1] == 1) and MTP verification (ne[1] == 2..4) the KV
  length is never a multiple of 256, so both gqa_opt_applies and
  can_use_vector_kernel are false: everything runs the TILE kernel. The WMMA
  path only fires at 256-aligned KV lengths (e.g. prefill chunks of 512 or
  1024 tokens).
- There is no KV cache padding to FATTN_KQ_STRIDE (256) in the tree, so the
  alignment requirement is on the raw used length, not a padded buffer.

## Attention kernels and KV type handling

- TILE (fattn-tile.cuh): historically FP16-only. launch_fattn
  (fattn-common.cuh) converts K/V to FP16 with a `to_fp16` pass over the
  whole tensor on every call when need_f16_K/V is set. For BF16 KV that is
  a full KV read plus write per call: a DRAM round trip that scales with
  context.
- VEC (fattn-vec.cuh): BF16 K is converted to FP32 per element and the QK^T
  dot runs FP32 FMA (vec_dot_fattn_vec_KQ_bf16, upstream FIXME). A native
  alternative exists: convert Q to BF16 once per thread and use
  `v_dot2_f32_bf16`. It changes Q precision (FP16 -> BF16) and is not
  validated.
- MMA_F16 (fattn-mma-f16.cuh): FP16 tiles only, converts K/V on entry. The
  BF16 WMMA primitive already exists in mma.cuh (nv_bfloat162 overload,
  ~line 1260) but no BF16 MMA kernel instances are instantiated.

## Native BF16 tile loads (TILE)

The tile kernel reads BF16 K/V natively (no to_fp16 pass) on AMD
RDNA3/3.5/4 (guarded by V_DOT2_F32_BF16_AVAILABLE, explicit __gfxXXXX__
macros). K/V tiles stay BF16, Q is converted to BF16 at fill, and QK^T uses
v_dot2_f32_bf16 (exact BF16 products, FP32 accumulation). This preserves
the full BF16 exponent range through the attention math; the FP16 PV
accumulation bug (half2 VKQ) is also fixed by accumulating PV in FP32.

Two PV modes for BF16 K/V, selected by env var at dispatch time (both
variants compiled per kernel):

- Default (no env var): P stays FP32, PV runs FP32 math. Maximum
  precision; ~3% slower than F16 at deep context (e.g. 41.7 vs 42.95 t/s
  at 16K) where the PV cost scales with KV length.
- GGML_HIP_BF16_PV_DOT2=1: P rounded to BF16 (7-bit mantissa, ~0.4%
  added error - the BF16 design contract), V rows paired across
  consecutive KV rows via v_perm_b32, PV via v_dot2_f32_bf16 (one dot2 per
  head pair, FP32 accumulation). Recovers the 3%: exact F16 parity at
  depth. The KQ buffer still stores P as FP32; the BF16 pair conversion is
  amortized over the head dimension.

DV > 256 (rare 512/576-head models) and non-AMD platforms keep the
FP16-conversion tile path.

## Key files

- ggml/src/ggml-cuda/fattn.cu: kernel selection, need_f16_K/V
- ggml/src/ggml-cuda/fattn-tile.cuh/.cu: tile kernel and instances
- ggml/src/ggml-cuda/fattn-common.cuh: launch_fattn (to_fp16), vec dot
- ggml/src/ggml-cuda/fattn-vec.cuh: vec kernel
- ggml/src/ggml-cuda/fattn-mma-f16.cuh, mma.cuh: MMA kernel, BF16 WMMA primitive
- ggml/src/ggml-hip/CMakeLists.txt: fattn-mma*.cu glob

## Microbenchmark reproduction

A minimal HIP harness (FMA chains, fp32-accumulating dots, WMMA via
builtins) reproduces the table above. Build with `hipcc
--offload-arch=<arch> -O3` and inspect the device assembly with
`--save-temps`; verify native instructions assemble with llvm-mc.
