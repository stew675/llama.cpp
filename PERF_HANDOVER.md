# PERF HANDOVER — FP8 prefill: state, levers, and next-session tracking

**Purpose**: single document for the next session's performance work. Everything
here was measured on this box (3x R9700 gfx1201, 1x used, -dev ROCm0) with the
fp8 GGUF `/tmp/stewfp8-preserved.gguf` (Qwen3.5-4B StewFP8). Updated end of
session 2026-08-06 after the aiter GEMM port (commit pending, see section 8).

Read `HANDOVER.md` for the project as a whole and `AITER_FINDINGS.md` for the
aiter inspection details. This file tracks ONLY the performance picture.

---

## 1. Scoreboard (measured, all pp512 unless noted)

| config | pp512 t/s | vs Q8_0 | vs fp8 baseline |
|---|---|---|---|
| Q8_0 GGUF (target base) | 5847 | - | - |
| fp8 baseline (821d423ac) | 4929.8 | -16% | - |
| fp8 now (aiter port, this session) | **5546-5596** | **-5.6%** | **+12.5-13.5%** |
| tg64 (fp8, unchanged) | 71.0-71.4 | -21% | 0 |
| target prefill (+50% over Q8_0) | 8770 | +50% | +78% |
| target generation (+10%) | ~98.5 | +10% | +39% |

pp256: 5014, pp1024: 5446, pp2048: 5292, pp128: 3661 (high variance, use -r 5).

PPL 8.7152 (fp8) vs 8.6024 (bf16), 8.6126 (Q8_0) — unchanged by this port.
All 1186 MUL_MAT backend-ops tests pass; generation matches the BF16 reference.

## 2. Where pp512 time goes (rocprofv3 kernel trace, per ~1.8 bench runs)

| kernel | share of pp512 | note |
|---|---|---|
| mul_mat_fp8_wmma | ~50% | now ~77 TFLOP/s (was 55) |
| gated_delta_net_cuda | ~17% | linear-attention core, custom kernel |
| mul_mat_vec_f (bf16) | ~12.5% | incl. fused output_norm+lm_head (2.05 ms, m=248320) |
| flash_attn_tile | ~3.5% | |
| quantize_fp8 | ~1.2% | activation staging pass |
| norms/silu/rope/concat/copy | ~5% | |

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
- **token_embd.weight must STAY BF16**: it is type 30 (BF16) in the GGUF today
  and lm_head runs the fused BF16 mmv at 2.05 ms/launch; as fp8 it would be
  ~5.4 ms (slower). The quantizable set is norms, conv1d, in_proj_a/b, A_log
  (small) -> gen maybe +5%, prefill ~0.

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
/tmp/llama-hip-full/bin/llama-bench -m /tmp/stewfp8-preserved.gguf -p 512 -n 64 -r 3 -dev ROCm0

# correctness after kernel changes (fast gate)
/tmp/llama-hip-full/bin/test-backend-ops test -b ROCm0 -o MUL_MAT   # 1186 pass (no F8 cases! it only covers other types)
/tmp/llama-hip-full/bin/llama-cli -m /tmp/stewfp8-preserved.gguf -f /tmp/prompt_pp.txt -n 24 --single-turn -s 42
# expected: prompt echo + "[Start thinking] Thinking Process: 1. Analyze the Request:..." (matches bf16 ref)
# note: test-backend-ops has NO F8_E4M3 mul_mat cases (CPU backend can't do fp8) - generation diff is the gate

# profile
rocprofv3 -r -d /tmp/rocp -f csv -- /tmp/llama-hip-full/bin/llama-bench -m /tmp/stewfp8-preserved.gguf -p 512 -n 8 -r 1 -dev ROCm0
# per-kernel CSV in /tmp/rocp/soar/*kernel_trace.csv (Grid_Size_X = work-items = grid*256; VGPR/LDS columns)

# aiter reference numbers (M=512 gate/up etc. at 89-137 TFLOP/s)
PYTHONPATH=/home/stew675/aiter /tmp/aiter-venv/bin/python /tmp/bench_aiter_stable.py
```

## 7. PPL / parity (do not skip after any kernel or loader change)

```bash
/tmp/llama-hip-full/bin/llama-perplexity -m /tmp/stewfp8-preserved.gguf -f /tmp/corpus_pride.txt -c 512 -b 2048 -n 4 -dev ROCm0 2>&1 | tr -d '\0' | grep "Final estimate"  # ~8.715
```

## 8. Git state

- Kernel changes (ggml/src/ggml-cuda/fp8.cuh + fp8.cu) + docs (AITER_FINDINGS.md,
  HANDOVER.md, PERF_HANDOVER.md) are being committed as one unit this session.
- Branch `cllm`, origin git@github.com:stew675/llama.cpp.git.
- Commit style: concise subject, `Assisted-by:` line, no Co-authored-by.

## 9. Next-session suggested order

1. Confirm working tree matches this doc (git log, re-bench pp512 ~ 5550).
2. L1 (delta-net): profile `gated_delta_net_cuda` per-layer; check aiter's
   gfx1201 dispatch actually fires; port or optimize ours; PPL + bench.
3. L2 (weight repack): re-profile the staging share first; implement the
   in-memory repack; vectorize sA staging; PPL + bench.
4. Re-evaluate the +50% target with both in; then decide if L3/L4/L5 are worth it.
