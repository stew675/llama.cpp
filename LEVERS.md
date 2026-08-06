# LEVERS.md - remaining performance work, ranked, for the next session

**Purpose**: self-contained list of the remaining performance levers for the
fp8 GDN model on this box, with the current state, the approach, the expected
win, and the validation steps for each. Step through in order; stop when the
gain no longer justifies the risk. Everything here was measured on 2026-08-06
after the chunked GDN correctness fix (ef41a940a) and phase B rewrite
(56e7cb0c9). Read `PERF_HANDOVER.md` for history and `HANDOVER.md` for the
project as a whole.

Box: 1x AMD Radeon AI PRO R9700 (gfx1201, 64 CUs, 2 SIMDs/CU, wave32,
64 KB LDS/CU, 8 MB L2), 2350 MHz. Model: `/llm/models/Qwen3.5/4B/StewFP8/
stewfp8-ow.gguf` (24 delta-net layers, H=32 heads, K=V=128, neqk1=16).

---

## 0. Current state (verified 2026-08-06)

| metric | value | notes |
|---|---|---|
| pp512 (chunked) | 6068-6072 t/s | sequential path: 5878 |
| tg64 | 89.8-90.0 | decode untouched, sequential GDN |
| PPL | 6.2426 | sequential 6.2572, on /tmp/corpus_pride.txt |
| backend tests | 47/47 GATED_DELTA_NET pass | incl. multi-seq cases |
| GDN chunked per pp512 | 13.5 ms = ~16% of pp512 | phase A 7.1 + phase B 6.4 |
| GDN phase B | 267 us/launch, 24 launches/pp512 | 1 wave, 2 CTAs/CU, no spills |
| GDN phase A | ~280-300 us/launch, 24 launches/pp512 | 2 waves, 1 CTA/CU (50 KB LDS) |

Bench geometry (IMPORTANT, do not rediscover): `llama-bench -p 512` runs the
512-token prompt as ONE batch (n_seqs=1, n_chunks=8) and llama-bench performs
a warmup pass first. The kernel trace therefore shows 2x the per-pp512
launch counts (24 layers x 2) and 2x the per-pp512 kernel times. Per-pp512
numbers in this file are the trace numbers divided by 2 unless stated.

Profile command: `rocprofv3 -r -d /tmp/rocp_x -f csv -- llama-bench -m <model>
-p 512 -n 8 -r 1 -dev ROCm0`, then parse `soar/*kernel_trace.csv`
(Duration = End_Timestamp - Start_Timestamp; Grid_Size_X = work-items,
divide by Workgroup_Size_X for CTA count; VGPR/LDS/Scratch columns).

## 1. Where pp512 time goes (per-pp512, warmup removed)

| kernel | ms/pp512 | share | note |
|---|---|---|---|
| mul_mat_fp8_wmma | ~41 | ~48% | prefill GEMMs, ~77 TFLOP/s |
| gdn_chunk_prepare + state | 13.5 | ~16% | A 7.1 + B 6.4 |
| quantize_fp8 | ~6.3 | ~7.5% | activation staging pass |
| Cijk (CK BF16 GEMM) | ~5.3 | ~6% | delta-net in/out projections |
| concat_non_cont | ~4.7 | ~5.5% | ssm conv input assembly, 24 launches |
| flash_attn_tile | ~3.0 | ~3.5% | 8 attention layers |
| silu | ~2.9 | ~3.5% | |
| rms_norm (both sizes) | ~4.4 | ~5% | |
| k_bin_bcast / cpy / get_rows / l2_norm / rope / ssm_conv | ~3.3 | ~4% | small |
| (decode kernels: gemv, sequential GDN, etc.) | - | - | not part of pp512 |

pp512 wall = 512 / 6068 = 84.4 ms. The shares above sum to ~90% of that.

## 2. Ranked levers

### L1. mul_mat_fp8_wmma: 2 CTAs/CU via 24 KB smem tiles (biggest pp512 lever)

- Kernel: mul_mat_fp8_wmma, ~41 ms/pp512 (~48%), ~77 TFLOP/s.
- Why slow: CTA tile 128x64 with 8 warps (4m x 2n, 2x2 register tiles),
  single-buffered smem 26.4 KB, register-staged k-block pipelining,
  grouped-M swizzle. 8 warps/CU = 58 TFLOP/s, 16 warps/CU = 77. The aiter
  triton kernel hits 121 TFLOP/s on gfx1201 via a 24 KB smem tile running
  2 CTAs/CU (16 warps) - NOT via double-buffering (61 KB = 1 CTA/CU = 55).
- Approach: shrink the smem tile so 2 CTAs fit (16 warps/CU) while keeping
  the fragment-reuse ratio (1.0 LDS/wmma). 64x64 tile with 8 warps forces
  2 accs/warp (regression risk, see section 4). 128x64 with 16 warps and
  2x1 tiles per warp also regressed (fragment reuse 1.5 LDS/wmma).
  Re-measure the fragment-reuse cost before committing to a shape.
- Expected: +8-11% pp512 if 2 CTAs/CU is reached without fragment-reuse loss.
- Validation: llama-bench pp512, tg64 must stay ~90, generation parity
  check (llama-cli --single-turn, see section 3), PPL gate.
- Depends on: nothing. Independent of L2.

### L2. wmma weight repack at load time (second GEMM lever)

- Why slow: block_f8_e4m3 rows are 132 B (f32 d + 128 fp8), so staging loads
  are 4-B only (misaligned for 16-B). The wmma kernel spends a large share of
  its time staging (weights 16 KB + activations 8 KB per k-block per CTA).
- Approach: repack weights once at model load into a wmma-friendly layout:
  fp8 bytes contiguous [m][k] (16-B aligned rows), scales as a separate
  [m/128][k/128] f32 array. Do it in the backend when the tensor is first
  used (ggml_cuda_mul_mat_fp8) into a cached buffer so BOTH the safetensors
  loader and GGUF path get it. Changes Abase addressing only.
- Expected: +10-20% on the wmma kernel if staging is the limiter (was ~23% of
  kernel time at 55 TFLOP/s; re-profile the staging share first by removing
  the staging loop).
- Validation: same as L1. Note test-backend-ops has NO F8_E4M3 mul_mat cases
  (CPU can't do fp8) - generation diff and PPL are the gates.
- Depends on: profile first (staging share at current 77 TFLOP/s).

### L3. gdn_chunk_prepare (phase A): closure barriers + 1 CTA/CU (GDN priority)

- Kernel: gdn_chunk_prepare, ~7.1 ms/pp512, ~280-300 us/launch, grid 256 CTAs
  (32 heads x 8 chunks), 256 threads, 50 KB LDS (1 CTA/CU, 2 waves), 128 VGPR,
  no spills. ~8x its FMA floor (~18.5 us/chunk theoretical).
- Why slow: two things.
  1. The (I-A)^-1 closure runs as 16x16 block forward substitution with
     ~154 dynamic __syncthreads per CTA (4 diag blocks x 16 rows x 2 syncs +
     6 off-diag blocks x 3 syncs + stage syncs). Each barrier with 8 warps
     and 1 CTA/CU exposes the full L2 latency of the next stage.
  2. The matmul stages (gram, M2 k_cumsum/k_cumdecay, M3 attn_causal) are
     serial: each reads s_attn/s_k from smem after a barrier, no overlap.
- Approach (in order of risk):
  a. Reduce closure barrier count. The diag-block pass does 2 syncs per row;
     the row substitution for row i only needs rows < i - restructure so a
     full 16-row diag block is done with 2 syncs total (one per row of the
     block, or batch the substitution per 4-row group). Target: 154 -> ~30.
  b. Overlap stage c+1's global loads (k, v from L2) with stage c's compute:
     the phases read disjoint inputs; issue the float4 loads for the next
     phase before the current phase's FMA chain.
  c. Cut smem so 2 CTAs/CU: s_attn 16.6 KB + s_k 33 KB + s_bp/s_decay are the
     consumers. If the closure and M2 can share s_attn's space (they already
     reuse it), the fixed cost is s_k 33 KB - get the total under 32 KB and
     the kernel becomes 1 wave like phase B.
- Expected: 7.1 ms -> 3.5-5 ms (phase A halves), pp512 +2-4%.
- Validation: 47/47 tests, PPL 6.24, dump check if touched the closure math
  (numpy reference in /tmp/validate_gdn_dump.py; GDN_DUMP infra was removed
  from the code - re-add if needed, see GDN_DEBUG_HANDOVER.md for the
  stream-ordered dump pattern).
- Depends on: nothing. Do before L4.

### L4. gdn_chunk_state (phase B): chunk-loop latency (optional, lower value)

- Kernel: gdn_chunk_state, ~6.4 ms/pp512, 267 us/launch, 1 wave, 2 CTAs/CU,
  no spills, ~10x its FMA floor.
- Why slow: the 8-chunk serial loop; kd/q/k/ac are re-read from L2 per chunk
  (~120 MB/launch = ~40 us of L2 bandwidth at 3 TB/s - NOT bandwidth-bound,
  it is latency-bound on the load chains with only 16 warps/CU).
- Approach: prefetch chunk c+1's kd/q/ac/k float4 loads during chunk c's
  compute (registers or a small smem staging buffer - smem budget is 24.6 KB,
  room for ~8 KB), or try 8 slices of 16 (more CTAs, 2x L2 amplification -
  probably a wash).
- Expected: 6.4 -> 4-5 ms if prefetch works, pp512 +1.5-2.5%.
- Validation: same as L3.
- Depends on: after L3 (phase A is bigger).

### L5. concat_non_cont: delta-net ssm conv plumbing (4.7 ms/pp512)

- Kernel: concat_non_cont, 24 launches/pp512, 4.7 ms. It assembles the ssm
  conv input: `conv_input = ggml_concat(conv_states, qkv_mixed, 0)` in
  src/models/delta-net-base.cpp (line ~472), feeding ssm_conv_long_token_f32.
- Why slow: it moves qkv_mixed (activation) + conv_states into a fresh
  contiguous buffer every layer; the data is L2/DRAM round-tripped.
- Approach: check if ssm_conv can read the pieces in place (the conv kernel
  is strided anyway - conv_states and qkv_mixed are already contiguous
  individually; conv over a non-contiguous input needs a per-row offset
  table, which the kernel signature does not have). If not feasible, this is
  the lowest-effort: nothing.
- Expected: 4.7 -> ~0 if fusable, pp512 +5%.
- Validation: generation parity + PPL.
- Depends on: reading ssm_conv_long_token_f32 first.

### L6. quantize_fp8 fusion (small)

- Kernel: quantize_fp8, ~6.3 ms/pp512. Staging f32 activations -> fp8 for the
  wmma GEMMs; adds ~300 MB DRAM round-trip per pp512.
- Approach: fuse the quantization into the GEMM's activation staging (read
  f32 and quantize in-kernel). 4x the read bytes but removes the round-trip.
- Expected: ~1 ms + staging simplification; only after L1/L2.
- Depends on: L1/L2 (the GEMM kernel is the consumer).

### L7. Cijk BF16 GEMMs (delta-net in/out projections, 5.3 ms/pp512)

- The qwen35 delta-net projections are BF16 in the GGUF and run on rocBLAS/
  CK. Moving them to fp8 (they are linear projections like the rest) would
  put them on the wmma path; the L1/L2 gains then apply. The conversion
  script (convert_hf_to_gguf.py) already has the fp8 machinery.
- Expected: cuts ~5.3 ms by the wmma-vs-CK ratio; medium effort (conversion +
  PPL re-gate).
- Depends on: decision to quantize more tensors (PPL risk; the handover's L5
  lists the quantizable set: norms, conv1d, in_proj_a/b, A_log).

### L8. Small kernels (silu 2.9, rms_norm 4.4, k_bin_bcast 1.9) - skip

- Not worth the risk/effort individually. Revisit only if everything else is
  done and a specific one shows up hot in a fresh profile.

## 3. Session protocol (commands)

```bash
# build (after any kernel change)
cmake --build /tmp/llama-hip-full -j16 --target llama-bench llama-perplexity test-backend-ops

# bench (pp512 + tg64, 5 reps)
/tmp/llama-hip-full/bin/llama-bench -m /llm/models/Qwen3.5/4B/StewFP8/stewfp8-ow.gguf -p 512 -n 64 -r 5 -dev ROCm0

# correctness gate (always)
/tmp/llama-hip-full/bin/test-backend-ops test -b ROCm0 -o GATED_DELTA_NET   # 47/47
/tmp/llama-hip-full/bin/llama-perplexity -m /llm/models/Qwen3.5/4B/StewFP8/stewfp8-ow.gguf \
    -f /tmp/corpus_pride.txt -c 512 -b 2048 -n 4 -dev ROCm0 2>&1 | tr -d '\0' | grep "Final estimate"  # 6.24

# generation parity (PPL is not enough for fp8 changes)
/tmp/llama-hip-full/bin/llama-cli -m /llm/models/Qwen3.5/4B/StewFP8/stewfp8-ow.gguf -c 512 -n 24 \
    -dev ROCm0 -p "The capital of France is" --single-turn --no-conversation   # needs both flags in this fork

# profile
rocprofv3 -r -d /tmp/rocp_x -f csv -- /tmp/llama-hip-full/bin/llama-bench \
    -m /llm/models/Qwen3.5/4B/StewFP8/stewfp8-ow.gguf -p 512 -n 8 -r 1 -dev ROCm0
# parse soar/*kernel_trace.csv; REMEMBER the warmup doubles every kernel count/time

# sequential-vs-chunked A/B
GGML_CUDA_GDN_CHUNKED=0 <same bench>   # sequential GDN for comparison
```

## 4. Pitfalls and failed experiments (do not rediscover)

- Kernel traces include llama-bench's warmup pass: counts and times are 2x
  the per-pp512 values. Divide by 2 for per-pp512 numbers.
- GDN phase B register-tiled 16x16 grid (4x2 tiles/thread) with smem-staged
  kd/q/ac/k blocks: 192 VGPRs + spills (764-1028 B scratch; ISA showed 274
  scratch instructions vs 308 FMAs) -> 675 us/launch. The tile layout spreads
  the V dimension across threads, amplifying shared-input L2 reads 16x and
  blowing the register budget. Thread-per-column (32 cols per warp) is the
  right layout for these 64x128x32 matmuls: every kd/q/k/ac load becomes a
  warp broadcast and live registers stay ~40. Do not go back to register
  tiling here without a concrete reason.
- `__launch_bounds__(256, 2)` forcing 128 VGPRs on a 192-VGPR kernel made it
  worse (more spills). Fix spills by restructuring, not by capping.
- GGML_HIP_GRAPHS=0 is a CMAKE option, not a runtime env var. The runtime
  disable is GGML_CUDA_DISABLE_GRAPHS=1. CUDA graphs are NOT the cause of
  any correctness issue.
- cudaMemcpy/cudaStreamSynchronize inside the launch during graph capture is
  illegal/unreliable. If re-adding the GDN_DUMP debug, all copies must be
  stream-ordered (cudaMemcpyAsync on the compute stream) with one sync after
  the phase B launch.
- wmma experiments that failed: 16 warps/CTA with 2x1 tiles (fragment reuse
  1.5 LDS/wmma, 4936 t/s), double-buffered smem 61 KB (1 CTA/CU, 5110 t/s),
  burst fragment arrays (LLVM miscompile). GROUP_SIZE_M=8 was noise.
- The chunked GDN closure bug history is in GDN_DEBUG_HANDOVER.md; the
  current closure (16x16 block forward substitution) is verified exact
  (numpy diff ~1e-7). Do not "simplify" it back to I+A+A^2+A^3.

## 5. Suggested order for the next session

1. Re-bench to confirm the working tree matches section 0 (6068 pp512).
2. L1 (wmma 2 CTAs/CU) or L3 (phase A) - both independent, similar expected
   gain. L3 is lower-risk (no fp8 numerics involved) and keeps the GDN
   momentum; L1 has the bigger ceiling if the fragment-reuse cost is tamed.
3. L2 (repack) after L1's staging share is re-measured.
4. L5 (concat) only if L3 leaves the profile where expected.
5. Re-evaluate L4/L6/L7 with fresh numbers.
