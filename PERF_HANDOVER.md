# PERF HANDOVER — FP8 prefill: state, levers, and next-session tracking

**Purpose**: single document for the next session's performance work. Everything
here was measured on this box (3x R9700 gfx1201, 1x used, -dev ROCm0) with the
fp8 GGUFs in `/llm/models/Qwen3.5/4B/StewFP8/` (Qwen3.5-4B StewFP8). Updated
2026-08-05 evening after the decode session (see section 10): generation now
matches Q8_0; prefill is the remaining gap.

Read `HANDOVER.md` for the project as a whole and `AITER_FINDINGS.md` for the
aiter inspection details. This file tracks ONLY the performance picture.

---

## 1. Scoreboard (measured, all pp512 unless noted)

| config | pp512 t/s | tg64 t/s | vs Q8_0 pp | vs Q8_0 tg |
|---|---|---|---|---|
| Q8_0 GGUF (target base) | 6177 | 90.9 | - | - |
| fp8 baseline (821d423ac) | 4930 | 71.0 | -20% | -22% |
| fp8 + aiter GEMM port (26b7be281) | 5546-5596 | 71.4 | -5.6% | -21% |
| fp8 now (decode session, commit below) | **5914** | **89.1** | **-4.3%** | **-2.0%** |
| target prefill (+50% over Q8_0) | 8770 | - | +50% | - |
| target generation (+10%) | - | ~100 | - | +10% |

pp256: 5014, pp1024: 5446, pp2048: 5292, pp128: 3661 (high variance, use -r 5).

PPL 6.2572 (fp8-ow) vs 6.2508 (fp8, no ow) vs 6.2464 (Q8_0) on the small
/tmp/corpus_pride.txt corpus (all within error bars); generation matches the
pre-change fp8 output on the same prompt/seed.

## 2. Where pp512 time goes (rocprofv3 kernel trace, decode session, fp8-ow GGUF)

| kernel | share of pp512 | note |
|---|---|---|
| mul_mat_fp8_wmma | ~48% | ~77 TFLOP/s (aiter port) |
| gated_delta_net_cuda | ~17% | linear-attention core, sequential per-token loop |
| Cijk (rocBLAS/CK BF16 GEMM) | ~6% | the delta-net in/out projections (BF16 in the GGUF) |
| quantize_fp8 | ~6% | activation staging pass |
| concat_non_cont | ~5% | delta-net plumbing (48 launches) |
| flash_attn_tile | ~4% | 8 attention layers |
| silu / norms / conv / copy | ~9% | ssm_conv_long_token_f32 ~1% |

Decode (tg64) breakdown, per ~11.1 ms step: mul_mat_fp8_gemv 8.95 ms (79%, 226
GEMMs incl. lm_head at ~496 GB/s), quantize_fp8 0.32 ms, gated_delta_net 0.41
ms, BF16 mmv 0.15 ms, norms/ops ~1.3 ms.

## 3. Ranked levers for the next session

### L1. gated_delta_net_cuda (linear attention) — HIGH value, MEDIUM effort
- 17% of pp512. aiter has a gfx1201 port:
  `/home/stew675/aiter/csrc/kernels/chunk_gated_delta_rule_fwd_h.cu` (explicit
  `__gfx1201__` support) + `op_tests/test_gated_delta_rule.py`.
- Our kernel is `gated_delta_net_cuda<128,false,false>` in ggml-cuda (runs the
  24 linear-attention layers' SSM). If 2-3x faster: pp512 +8-11% -> ~6000-6200.
- It is a port (different data flow than llama.cpp's), not a swap. Verify the
  gfx1201 path is used (there is a CK fallback; check dispatch) and that the
  f32 intermediate precision matches our current kernel (PPL check).
- This alone + the current GEMM gets close to Q8_0; combined with L2 it is the
  most credible route to +50%.

### L2. GEMM: weight-layout repack at load time — MEDIUM-HIGH value, MEDIUM effort
- The `block_f8_e4m3` rows are 132 B (f32 d + 128 fp8), so staging loads are
  4-B only (misaligned for 16-B). The wmma kernel spends ~30% of its time
  staging (weights 16 KB + activations 8 KB per k-block per CTA).
- Repack weights once at model load into a wmma-friendly layout: fp8 bytes
  contiguous `[m][k]` (16-B aligned rows), scales as a separate `[m/128][k/128]`
  f32 array. Then the kernel stages sA with 16-B vector loads.
- The repack must happen for BOTH the safetensors loader and the GGUF path, or
  in a shared in-memory step after load (best: do it in the backend when the
  tensor is first used, or in `ggml_cuda_mul_mat_fp8` into a cached buffer).
  Changes the kernel's Abase addressing only.
- Expected: +10-20% on the wmma kernel if staging bandwidth/latency is the
  limiter (it was ~23% at 55 TFLOP/s; at 77 the share may differ — re-profile
  first with the staging-loop removed to confirm it is still worth it).

### L3. GEMM: 3 CTAs/CU via 64x64 tiles — UNKNOWN, low confidence
- gfx1201: 196,608 regs/mp -> 3 CTAs of 256 threads fit by registers; LDS
  allows 3 x 17.7 KB = 53 KB. 24 warps/CU (vs 16 now).
- BUT 64x64 with 8 warps forces 2 accs/warp (2x1 or 1x2 tiles); the 16-warp
  experiment with 2-acc warps regressed badly (fragment-reuse loss). Try
  64x64 with 8 warps as 4m x 2n (warp 16x32 = 1x2 tiles, 2 accs) only if L2
  (repack) is done and the fragment-reuse cost is re-measured.
- Alternative: CTA 128x64 but 12-16 warps with 2x2 tiles... does not tile.

### L4. Fuse activation quantization into the GEMM — SMALL
- quantize_fp8 is only ~1.2% of pp512, but the staging round-trip also costs
  ~300 MB DRAM traffic per pp512 (~1.1 ms). Fusing reads f32 activations
  (4x the bytes) into the kernel — net win small; only do this after L1/L2.

### L5. Non-fp8 tensors -> Q8_0 in the GGUF — SMALL-MEDIUM, decoder-focused
- Only helps decode (memory-bound) and slightly reduces prefill staging.
- **token_embd constraint is now VOID**: the decode session added
  `--fp8-output-weight` (an fp8 output.weight copy of token_embd, token_embd
  stays BF16 for the input lookup) so lm_head runs on the fast fp8 GEMV. Do NOT
  quantize token_embd itself to fp8 without adding a get_rows fp8 kernel
  (getrows.cu has no F8_E4M3 case; CPU has no fp8 kernels either).
- The quantizable set is norms, conv1d, in_proj_a/b, A_log (small) -> gen maybe
  +5%, prefill ~0.

## 4. Kernel facts (established this session, do not rediscover)

- **Occupancy is the dominant GEMM lever on gfx1201**: 8 warps/CU = 58 TFLOP/s,
  16 warps/CU = 77 TFLOP/s. The aiter triton kernel's 121 TFLOP/s comes from
  its 24 KB smem tile running 2 CTAs/CU (16 warps) — NOT from its 2-stage
  pipeline. Double-buffered 128x64 (61 KB LDS) = 1 CTA/CU = the old 55-58.
- Raw wmma 16x16x16 fp8 ceiling: ~165 TFLOP/s (conservative microbench; codegen
  dependent up to ~430). Current kernel = 47% of the conservative ceiling.
- gfx1201 device props: smem/block = 65536, regs/mp = 196608, maxThreads = 2048.
- Current kernel shape: CTA 128 rows x 64 tokens, 8 warps (4m x 2n, 2x2 tiles),
  single-buffered smem 26.4 KB (sA 128x136 + sB 64x136 + sS 64), register-staged
  k-block pipelining (loads cb+1 into regs before cb's wmma chain, commit after
  the first of 2 barriers per k-block), grouped-M swizzle GROUP_SIZE_M=4.
- VGPR ~200, LDS 26,624 -> 2 CTAs/CU.

## 5. Failed experiments (do not retry without a reason)

| experiment | result |
|---|---|
| 16 warps/CTA, 2x1 tiles/warp (2 accs) | 4936 t/s (regression; fragment reuse 1.5 LDS/wmma vs 1.0) |
| Burst fragment array `fp8x8_t fa[4][4]` | **LLVM MISCOMPILE**: 16/32 wmma get v[0:1] operands under the single-buffer loop (correct in double-buffer variant). Reverted to software-pipelined loads. |
| GROUP_SIZE_M=8 | noise (5570 vs 5546) |
| Double-buffered smem (61 KB) | 1 CTA/CU -> 5110 t/s |
| (earlier session) 8-warp 64x128 CTA, register prefetch, aligned-A split, MT=4 | all < baseline; see HANDOVER.md section 8 |

## 6. Repro / tooling

```bash
# build + bench (after any kernel change)
cmake --build /tmp/llama-hip-full -j16 --target llama-bench llama-cli
/tmp/llama-hip-full/bin/llama-bench -m /llm/models/Qwen3.5/4B/StewFP8/stewfp8-ow.gguf -p 512 -n 64 -r 5 -dev ROCm0

# Q8_0 reference on the same box/day
/tmp/llama-hip-full/bin/llama-bench -m /llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf -p 512 -n 64 -r 5 -dev ROCm0

# correctness after kernel changes (fast gate)
/tmp/llama-hip-full/bin/test-backend-ops test -b ROCm0 -o MUL_MAT   # 1186 pass (no F8 cases! it only covers other types)
/tmp/llama-hip-full/bin/llama-cli -m /llm/models/Qwen3.5/4B/StewFP8/stewfp8-ow.gguf -f /tmp/prompt_pp.txt -n 24 --single-turn -s 42
# expected: prompt echo + "[Start thinking] Thinking Process: 1. Analyze the Request:..." (matches bf16 ref)
# note: test-backend-ops has NO F8_E4M3 mul_mat cases (CPU backend can't do fp8) - generation diff is the gate

# profile (decode-heavy runs crash rocprofv3 at teardown; the trace is still written)
rocprofv3 -r -d /tmp/rocp -f csv -- /tmp/llama-hip-full/bin/llama-bench -m /llm/models/Qwen3.5/4B/StewFP8/stewfp8-ow.gguf -p 512 -n 8 -r 1 -dev ROCm0
# per-kernel CSV in /tmp/rocp/soar/*kernel_trace.csv (Grid_Size_X = work-items = grid*256; VGPR/LDS columns)
# decode-only view: filter kernels after the last mul_mat_fp8_wmma start
```

## 7. PPL / parity (do not skip after any kernel or loader change)

```bash
/tmp/llama-hip-full/bin/llama-perplexity -m /llm/models/Qwen3.5/4B/StewFP8/stewfp8-ow.gguf -f /tmp/corpus_pride.txt -c 512 -b 2048 -n 4 -dev ROCm0 2>&1 | tr -d '\0' | grep "Final estimate"  # ~6.26 on this corpus
```

## 8. Git state

- This session (decode): ggml/src/ggml-cuda/fp8.cu + fp8.cuh (gemv rewrite +
  memset skip), convert_hf_to_gguf.py + conversion/base.py (--fp8-output-weight),
  docs (PERF_HANDOVER.md, vllm-vs-llamacpp-performance.md) committed as one unit.
- Branch `cllm`, origin git@github.com:stew675/llama.cpp.git.
- Commit style: concise subject, `Assisted-by:` line, no Co-authored-by.

## 9. Next-session suggested order

1. Confirm working tree matches this doc (git log, re-bench pp512 ~ 5925).
2. L1 (delta-net): the chunked kernel exists but is SLOWER than the sequential
   (pp512 4200 vs 5925) - its phase A/B matmuls are naive (untiled, global
   reads). Optimize: smem-stage k/v/q, tile the [64x64]@[64x128] matmuls, cut
   the closure's 126 barriers, then re-enable by default (see section 11).
3. L2 (weight repack): re-profile the staging share first; implement the
   in-memory repack; vectorize sA staging; PPL + bench.
4. Re-evaluate the +50% target with both in; then decide if L3/L4/L5 are worth it.

## 10. Decode session log (2026-08-05 evening)

Goal: close the fp8 generation gap to Q8_0 (tg64 was 71.0-71.4 vs Q8_0 ~90).
Result: tg64 72.0 -> 89.1 (98% of Q8_0's 90.9), pp512 unchanged ~5914.

Three changes, all gated by generation-parity + PPL:

1. fp8.cu: skip the n_pad zero-fill (cudaMemset x2 per GEMM, 450/step) when
   n <= GGML_FP8_GEMV_MAX_N; the GEMV path never reads the padded tokens.
   Removed ~0.5 ms/step.
2. fp8.cuh: rewrote mul_mat_fp8_gemv - fold the per-block scales into the lane
   partial (one deferred warp-reduce instead of 160 shuffles/row), 4 k-blocks
   per iteration for load ILP. Bandwidth 445 -> 496 GB/s (mmvf reference: 520).
3. convert_hf_to_gguf.py + conversion/base.py: --fp8-output-weight writes an
   fp8 output.weight copy of token_embd (same 128x128 convention as the model:
   d = BF16(amax/448), q = fp8(w/(amax/448))). lm_head moves from the fused
   BF16 mmv (2.44 ms/step) to the fp8 GEMV (~1.4 ms/step); token_embd stays
   BF16 for get_rows. +0.66 GiB storage, +0.63 B params.

New GGUF: /llm/models/Qwen3.5/4B/StewFP8/stewfp8-ow.gguf (5.36 GiB, 442
  tensors: token_embd BF16 + output.weight F8_E4M3). Regenerate with:
  python3 convert_hf_to_gguf.py /llm/models/Qwen3.5/4B/StewFP8 --outtype fp8_e4m3 --fp8-output-weight --outfile ...

Left on the table (decode): gemv 496 -> ~520 GB/s (vectorized x loads, 8-block
ILP), fuse quantize_fp8 into the gemv (226 launches/step), delta-net 0.41
ms/step. Roughly +4% if all taken.

## 11. Delta-net chunked session log (2026-08-05 night)

Goal: speed up gated_delta_net_cuda (17% of pp512) by replacing the sequential
per-token loop with a chunked formulation.

KEY FINDING: aiter's chunk_gated_delta_rule_fwd_h.cu does NOT run on gfx1201 -
op_tests/test_gated_delta_rule.py skips it on gfx12
("kernel does not support gfx12!"). The handover's "aiter has a gfx1201 port"
premise was wrong. The __gfx1201__ guard there is only for a buffer-resource
constant. So no port; we implemented our own chunked kernel.

Status: CORRECT but SLOWER. ggml-cuda/gated_delta_net.cu has gdn_chunk_prepare
(phase A: per head x seq x chunk of 64 - decay cumsum, closure (I-A)^-1 via
forward substitution, k_cumsum = attn @ (beta v), k_cumdecay = attn @ (beta k P),
attn_causal = L .* (q^T k)) and gdn_chunk_state (phase B: per head x seq x
V-slice of 8 - v_new = k_cumsum - k_cumdecay @ S, o = P .* (q @ S) + attn_causal
@ v_new, S = P_last S + (k .* delta) @ v_new). Opt-in via GGML_CUDA_GDN_CHUNKED=1;
off by default because pp512 = 4200 vs 5925 sequential.

Math: validated in python (/tmp/validate_chunked.py) against the op recurrence
(aiter's chunk_gated_delta_rule_ref structure, decay kept inside via
L[i][j] = exp(decay_i - decay_j)). All 45 GATED_DELTA_NET backend-ops tests pass
on BOTH paths (add 4 S_v=128 prefill cases to the eval set). PPL 6.30 vs 6.26
baseline; generation matches.

Bugs found and fixed (do not rediscover):
- P = exp(cumsum(gate)) underflows to 0 for strong decay (cumsum < -88) ->
  P_i/P_j = 0/0 = NaN. Keep the cumsum and always take exp(diff).
- Phase-A scratch was indexed per chunk only, not per (head, seq): heads raced
  on the same scratch (outputs looked like adjacent heads' data).
- The decay_scratch store used tid = 0..127 over a 64-float per-head region:
  threads 64-127 wrote garbage into the next head's region (nondeterministic
  per-head corruption; single-head tests passed, multi-head flaky).
- cudaStreamSynchronize / cudaEventCreate between the two kernels is ILLEGAL
  during CUDA graph capture ("operation not permitted when stream is
  capturing"). The kernels are on the same stream; no sync needed.

Why it is slow: ~2x the FLOPs of the sequential kernel (closure + materialized
k_cumsum/k_cumdecay/attn_causal) with untiled matmuls reading global memory and
126 barriers in the closure. Next session: tile the matmuls with smem staging
(phase A smem budget: s_attn 16.6 KB + s_k 32 KB fits one CTA/CU), reduce the
closure barrier count, then re-enable by default.
