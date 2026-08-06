# PERF HANDOVER - FP8 prefill: state, history, and operational reference

**Purpose**: the operational reference for performance work on this box: what
was done, why, and the standing rules (kernel facts, failed experiments,
tooling, git state, session logs). The FORWARD plan - ranked levers, expected
wins, validation steps, and the suggested order - lives in `LEVERS.md`;
PERF_HANDOVER.md intentionally does not duplicate it. Start a session with
`LEVERS.md`, come here for history and specifics.

Everything was measured on this box (1x R9700 gfx1201, 64 CUs, -dev ROCm0)
with the fp8 GGUFs in `/llm/models/Qwen3.5/4B/StewFP8/` (Qwen3.5-4B StewFP8).
Updated 2026-08-06 after the GDN correctness fix + phase B rewrite: the
chunked GDN is now faster than sequential and pp512 is at 6068.

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
| fp8 now (chunked GDN, 2026-08-06) | **6068-6072** | **89.8-90.0** | **-1.7%** | **-1.2%** |
| target prefill (+50% over Q8_0) | 8770 | - | +50% | - |
| target generation (+10%) | - | ~100 | - | +10% |

pp256: 5014, pp1024: 5446, pp2048: 5292, pp128: 3661 (high variance, use -r 5).

Current authoritative numbers (2026-08-06, chunked GDN default): see LEVERS.md
section 0. pp512 6068-6072, tg64 89.8-90.0.

PPL 6.2572 (fp8-ow) vs 6.2508 (fp8, no ow) vs 6.2464 (Q8_0) on the small
/tmp/corpus_pride.txt corpus (all within error bars); generation matches the
pre-change fp8 output on the same prompt/seed.

## 2. Where pp512 time goes (historical, decode session; current: LEVERS.md section 1)

| kernel | share of pp512 | note |
|---|---|---|
| mul_mat_fp8_wmma | ~48% | ~77 TFLOP/s (aiter port) |
| gated_delta_net_cuda | ~17% | sequential per-token loop; now replaced by the chunked path |
| Cijk (rocBLAS/CK BF16 GEMM) | ~6% | the delta-net in/out projections (BF16 in the GGUF) |
| quantize_fp8 | ~6% | activation staging pass |
| concat_non_cont | ~5% | delta-net plumbing |
| flash_attn_tile | ~4% | 8 attention layers |
| silu / norms / conv / copy | ~9% | ssm_conv_long_token_f32 ~1% |

The per-kernel shares are still representative; the current measured breakdown
(per-pp512, warmup removed) is in LEVERS.md section 1.

Decode (tg64) breakdown, per ~11.1 ms step: mul_mat_fp8_gemv 8.95 ms (79%, 226
GEMMs incl. lm_head at ~496 GB/s), quantize_fp8 0.32 ms, gated_delta_net 0.41
ms, BF16 mmv 0.15 ms, norms/ops ~1.3 ms.

## 3. Ranked levers - SEE LEVERS.md

The ranked levers, expected wins, and validation steps moved to `LEVERS.md`
(updated 2026-08-06 after the GDN work). Summary of what changed there vs the
old L1-L5 below:
- the delta-net port from aiter (old L1) is dead: aiter's chunk kernel does
  not run on gfx1201 (see section 11), and the chunked GDN now beats the
  sequential kernel anyway (section 12).
- the GDN levers are now internal: phase A (gdn_chunk_prepare) and phase B
  (gdn_chunk_state), both documented in LEVERS.md (L3/L4).
- the GEMM levers (occupancy via 24 KB smem tiles, weight repack, quantize
  fusion) survive as LEVERS.md L1/L2/L6.
- the non-fp8-tensor quantization lever (old L5) is folded into LEVERS.md L7
  (Cijk BF16 projections); the quantizable-set note below is still valid.

Old L5 note that still matters: **do NOT quantize token_embd to fp8** without
adding a get_rows fp8 kernel (getrows.cu has no F8_E4M3 case; CPU has no fp8
kernels either). lm_head already runs fp8 via the `--fp8-output-weight` copy
(token_embd stays BF16 for the input lookup). The quantizable set is norms,
conv1d, in_proj_a/b, A_log (small).

## 4. Kernel facts (established this session, do not rediscover)

- **Occupancy is the dominant GEMM lever on gfx1201**: 8 warps/CU = 58 TFLOP/s,
  16 warps/CU = 77 TFLOP/s. The aiter triton kernel's 121 TFLOP/s comes from
  its 24 KB smem tile running 2 CTAs/CU (16 warps) - NOT from its 2-stage
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

The GDN phase B failed designs (2026-08-06) are in section 12. The
consolidated do-not-rediscover list (both GEMM and GDN) is LEVERS.md section 4.

## 6. Repro / tooling

The full session protocol (build + bench + correctness gates + profile) is in
LEVERS.md section 3. Below: the fp8-specific extras that LEVERS.md does not
repeat.

```bash
# Q8_0 reference on the same box/day
/tmp/llama-hip-full/bin/llama-bench -m /llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf -p 512 -n 64 -r 5 -dev ROCm0

# GEMM correctness (fast gate)
/tmp/llama-hip-full/bin/test-backend-ops test -b ROCm0 -o MUL_MAT   # 1186 pass (no F8 cases! it only covers other types)
# note: test-backend-ops has NO F8_E4M3 mul_mat cases (CPU backend can't do fp8) - generation diff is the gate

# generation parity (fp8 changes)
/tmp/llama-hip-full/bin/llama-cli -m /llm/models/Qwen3.5/4B/StewFP8/stewfp8-ow.gguf -f /tmp/prompt_pp.txt -n 24 --single-turn -s 42
# expected: prompt echo + "[Start thinking] Thinking Process: 1. Analyze the Request:..." (matches bf16 ref)

# profile (decode-heavy runs crash rocprofv3 at teardown; the trace is still written)
rocprofv3 -r -d /tmp/rocp -f csv -- /tmp/llama-hip-full/bin/llama-bench -m /llm/models/Qwen3.5/4B/StewFP8/stewfp8-ow.gguf -p 512 -n 8 -r 1 -dev ROCm0
# per-kernel CSV in /tmp/rocp/soar/*kernel_trace.csv (Grid_Size_X = work-items = grid*256; VGPR/LDS columns)
# decode-only view: filter kernels after the last mul_mat_fp8_wmma start
# REMEMBER: llama-bench's warmup pass doubles every kernel count/time - divide by 2 for per-pp512
```

## 7. PPL / parity (do not skip after any kernel or loader change)

```bash
/tmp/llama-hip-full/bin/llama-perplexity -m /llm/models/Qwen3.5/4B/StewFP8/stewfp8-ow.gguf -f /tmp/corpus_pride.txt -c 512 -b 2048 -n 4 -dev ROCm0 2>&1 | tr -d '\0' | grep "Final estimate"  # ~6.26 on this corpus
```

## 8. Git state

- Branch `cllm`, origin git@github.com:stew675/llama.cpp.git.
- Commit style: concise subject, `Assisted-by:` line, no Co-authored-by.
- Recent commits (2026-08-06):
  - 56e7cb0c9 gated_delta_net : rewrite chunked phase B, 33.8ms -> 12.8ms
  - ef41a940a gated_delta_net : fix chunked phase A closure, enable by default
- Earlier (decode session, 2026-08-05): fp8.cu + fp8.cuh (gemv rewrite + memset
  skip), convert_hf_to_gguf.py + conversion/base.py (--fp8-output-weight),
  docs (PERF_HANDOVER.md, vllm-vs-llamacpp-performance.md).

## 9. Next-session suggested order - SEE LEVERS.md

The suggested order, and the current state to confirm first, moved to
LEVERS.md section 5 (re-bench 6068 -> L1 or L3 -> L2 -> L5 -> re-evaluate).

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

STATUS 2026-08-06: superseded - the chunked path is now default-on, correct
(closure fix, ef41a940a) and faster than sequential (phase B rewrite,
56e7cb0c9). See section 12 and LEVERS.md. The log below is historical.

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

## 12. Delta-net chunked: correctness fixed + phase B rewrite (2026-08-06)

The chunked GDN is now BOTH correct and faster than the sequential kernel.
Committed ef41a940a fixed the phase A closure (block forward substitution,
see GDN_DEBUG_HANDOVER.md). This session rewrote phase B.

Phase B before: gdn_chunk_state, 128 threads, V-slices of 8 (16 slices),
scalar serial reductions, stride-8KB global k loads in the S update.
33.82 ms per pp512 bench (48 launches x 704us, grid 32x16=512 CTAs).

Phase B now: 256 threads, V-slices of 32 (4 slices), one state column per
thread (32 cols per warp -> kd/q/k/ac loads are warp broadcasts, no L2
amplification), 8 token rows per thread (steps 1/2) or 16 state rows
(step 3). s_S (16 KB) + s_vnew (8 KB) smem -> 2 CTAs/CU, 1 wave (grid 128).
6.4 ms per pp512 (24 launches x 267us; the trace shows 48 x 267us = 12.8 ms
because llama-bench runs a warmup pass). VGPR 160, scratch 0, LDS 24.6 KB.

Scoreboard (pp512, stewfp8-ow, -p 512 -n 64 -r 5):
- chunked before this session: 5436 (broken-ish phase B)
- chunked now:               6068-6072  (sequential was 5878)
- tg64 unchanged:            89.8
PPL 6.2426 (sequential 6.2572), 47/47 GATED_DELTA_NET tests.

Failed experiments this session (do not retry without a reason; consolidated
with the GEMM list in LEVERS.md section 4):
- 16x16 thread grid with 4x2 register tiles + smem-staged kd/q/ac/k blocks:
  192 VGPRs + 764-1028 B scratch spills -> 274 scratch instructions vs 308
  FMAs in the ISA; 675-704us/launch. The register-tiled layout spreads the
  V dimension across threads, which both amplifies shared-input L2 reads 16x
  and blows the register budget.
- 16x16 grid, no staging (direct L2 float4): 591-644us/launch (L2-bound).
- __launch_bounds__(256,2) forcing 128 VGPRs: worse (more spills, 644us).
Lesson: for this kernel shape (64x128x32 matmuls, S resident), thread-per-
column with 32 cols/warp is the right layout - it makes every shared-input
load a warp broadcast and keeps live registers near 40.

Remaining GDN perf: the forward plan is LEVERS.md (L3 = phase A, L4 = phase B).
Summary: phase A (gdn_chunk_prepare) is now the bigger kernel (~7.1 ms/pp512,
~8x its FMA floor, ~154 dynamic closure barriers, 1 CTA/CU); phase B is ~10x
its FMA floor (chunk-loop latency).
