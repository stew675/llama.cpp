# Handoff — FP8 E4M3 + Direct Safetensors Loading for llama.cpp

Last updated: 2026-08-05, end of session. Working tree state captured below.

## 1. Project overview

Extending the `stew675/llama.cpp` fork (branch `cllm`, commit `6ea215d17` + local changes) to:

1. **Run FP8_E4M3 weights natively** (`GGML_TYPE_F8_E4M3`, QK=128, f32 block scale), hardware-gated: **only** native-FP8 devices (RDNA4 gfx1201, CUDA sm_89+). NO CPU kernels, NO software emulation — by explicit user decision (FP8 has zero advantage without native HW; users without it should use an integer GGUF).
2. **Load vanilla HuggingFace safetensors directly** (no GGUF conversion), using the existing `llama_model_init_from_user()` API + synthesized in-memory gguf_context.
3. Target model: `/llm/models/Qwen3.5/4B/StewFP8/` (Qwen3.5-4B, multimodal, FP8 E4M3, official Qwen 128x128-block convention). Format spec in `/llm/models/Qwen3.5/4B/StewFP8/FORMAT.md`.
4. Multimodal via existing mmproj.gguf + mtmd flow (converter already registers Qwen3_5ForConditionalGeneration for mmproj).
5. Full plan: `implementation-plan.md` in repo root (read it first; M1 marked DONE).

Personal fork, no upstream submission. Follow AGENTS.md style (ASCII only, concise comments).

## 2. Environment (all verified)

- **CPU**: AMD Ryzen 9 9950X3D (16 cores, Zen 5).
- **GPUs**: 3x AMD Radeon AI PRO R9700, **gfx1201** (RDNA4), wave32, 32GB VRAM each (total ~95GB), + 1x gfx1036 (RDNA2, no FP8). `rocm_agent_enumerator` confirms.
- **ROCm**: System ROCm 7.1.1 (Fedora 44): hipcc/clang at `/usr/lib64/rocm/llvm/bin`, hipblas/rocblas installed as packages (`/usr/lib64/cmake/hipblas`). Also `/opt/rocm-7.14` (ROCm 7.15, clang 23) — both work for FP8 WMMA. **The build script uses the system one.**
- **Build script**: `/home/stew675/bin/build-llama-rocm` — uses `hipconfig -l` (system ROCm), `-DGGML_HIP=ON -DGPU_TARGETS="gfx1200;gfx1201" -DGGML_HIP_GRAPHS=ON -DGGML_HIP_RCCL=1 -DGGML_CUDA_NO_PEER_COPY=1 -DGGML_RPC=1`, clang/clang++.
- `/home/stew675/rocm-libraries/` — ROCm math libs source (hipblas/rocblas/rocwmma) — NOT needed to build (system packages suffice).
- Python: torch 2.12.0 (CPU build), safetensors available.

## 3. Completed milestones

### M0 (environment) — verified
- HIP build works; 3562 MUL_MAT tests pass on the 3 R9700 GPUs (baseline).
- Both clang 20 (system) and clang 23 (/opt/rocm-7.14) emit native FP8 instructions for gfx1201:
  `v_wmma_f32_16x16x16_fp8_fp8` and `v_dot4_f32_fp8_fp8`.
- Builtin spellings (wave32): `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12` with A/B fragments `v2i32` (8 fp8/lane = 8 bytes), acc `v8f32`; `__builtin_amdgcn_dot4_f32_fp8_fp8(i32, i32, float)`. w64 variants need `wavefrontsize64` — unused (llama.cpp uses wave32).

### M1 (FP8 type registration) — DONE and COMMITTED (commit `d56412c8c`)
- `ggml/include/ggml.h`: `GGML_TYPE_F8_E4M3 = 43`, `GGML_TYPE_COUNT = 44`.
- `ggml/src/ggml-common.h`: `block_f8_e4m3 { float d; uint8_t qs[128]; }` (132B, static_assert).
- `ggml/src/ggml.c`: type_traits (name "f8_e4m3", blck 128, size 132, is_quantized, NO to_float/from_float/vec_dot) + `ggml_quantize_chunk` case calling `quantize_f8_e4m3`.
- `ggml/src/ggml-quants.c/.h`: fp8 e4m3fn encode (RNE, saturating to ±448) + decode helpers, `quantize_row_f8_e4m3_ref`, `dequantize_row_f8_e4m3`, `quantize_f8_e4m3`, `ggml_validate_row_data` case (rejects NaN bytes 0x7F/0xFF + non-finite scales).
- `ggml/src/ggml-cpu/ggml-cpu.cpp`: `ggml_backend_cpu_device_supports_op` returns false for any op with an F8_E4M3 operand.
- `tests/test-backend-ops.cpp`: F8_E4M3 added to `all_types`.
- Verified: quantize-fns passes; CPU rejects fp8 mul_mat; all 207 StewFP8 fp8 tensors (3.7B values) decode **bit-exact vs torch.float8_e4m3fn** (script pattern in session: numpy decode formula + `torch` cross-check; dequant rule `w = fp8_to_f32(q) * scale_inv[m/128,k/128]`).

## 4. M2 (GPU kernels) — IN PROGRESS, current state

### Files (uncommitted)
- `ggml/src/ggml-cuda/fp8.cuh` (new): device fp8 encode/decode, `quantize_fp8` kernel, `mul_mat_fp8_wmma` kernel (RDNA4), `mul_mat_fp8_scalar` fallback (CUDA).
- `ggml/src/ggml-cuda/fp8.cu` (new): host launcher `ggml_cuda_mul_mat_fp8` — pre-quantizes activations via pool alloc, launches wmma kernel via `ggml_cuda_kernel_launch`.
- `ggml/src/ggml-cuda/ggml-cuda.cu`: dispatch branch in `ggml_cuda_mul_mat` (after mmf check) + `supports_op` MUL_MAT case for F8_E4M3 (gated: `GGML_CUDA_CC_IS_RDNA4(cc) || (NVIDIA && cc >= GGML_CUDA_CC_ADA_LOVELACE)`, F32 src1, k%128==0, non-batched) + extern decl.
- `ggml/src/ggml-quants.c/.h` + `ggml.c` also have uncommitted M2 additions (host quantize/dequantize — needed for test data init and llama-quantize later).

### Design (follows llama.cpp launcher patterns — CRITICAL per user advice)
Read `ggml/src/ggml-cuda/mmq.cu` + `mmq.cuh` + `quantize.cu` for the pattern:
1. **Separate pre-quantization kernel** (`quantize_fp8`): F32 activations [k,n] → fp8 staging `[k][n_pad]` (k-major, token-minor, transposed!) + scales `[n][k/128]` f32. One CTA of 128 threads per (token, 128-col block), block-reduce amax → scale = amax/448, inverse for encode. **Verified byte-exact vs host quantization.**
2. **Hot wmma kernel** (`mul_mat_fp8_wmma`): weights = `block_f8_e4m3` rows [m][k/128]; activations from staging. 128 threads (4 wave32 warps), each warp = 16x16 output tile, grid (ceil(n/64), ceil(m/16)). Per 128-col block (cb): 8 wmma k-steps of 16 accumulate into acc_cb (unscaled), then scale-and-accumulate: `acc[s] += wd * ad * acc_cb[s]` where wd = weight block d (per m-row), ad = staging scale (per token). Uses `__launch_bounds__(128,1)` and `ggml_cuda_kernel_launch` (PDL-aware).
3. Launcher pads tokens to multiple of 64 (zeroed) so OOB wmma reads are safe; guards m-boundary via warp-uniform `row_valid`.

### THE gfx12 fp8 wmma fragment layout (DEFINITIVELY VERIFIED on gfx1201, 2026-08-05)
**The original probe-derived layout was WRONG.** The earlier probes used identity B, which is symmetric under transpose and CANNOT distinguish the B fragment's k-major vs n-major layout; and the probe4 matcher reported HOST-array byte positions instead of true fragment positions. The empirical probes on real data (single wmma with random fp8 data, checked 256/256 slots) plus TileLang/composablekernel documentation converge on:

- **A fragment** (weights): lane l byte e = `A[l%16][(l//16)*8 + e]` — lane l holds weight row (l%16); lanes 0-15 carry k=0..7 of each row, lanes 16-31 carry k=8..15.
- **B fragment** (activations): lane l byte e = `B[(l//16)*8 + e][l%16]` — lane l holds the 8 k-values of token (l%16), k-contiguous. Hence staging MUST be `[n][k]` (token-major, k-minor) so each lane's fragment is one contiguous 8-byte load.
- **C fragment** (output store): slot s of lane l = `C[(l//16)*8 + s][l%16]` — lane l owns column (l%16), 8 rows (l//16)*8..+7.

Row/col helpers in fp8.cuh: `fp8_wmma_row(l,s) = (l/16)*8 + s`, `fp8_wmma_col(l,s) = l%16`. Kernel locals: `row_lane = lane%16`, `k_half = (lane/16)*8`. A load: `wblk->qs + kk*16 + k_half` (8 contiguous bytes) of block `(m0+row_lane, cb)`. B load: `src1_q + (n0+row_lane)*k + cb*128 + kk*16 + k_half` (8 contiguous bytes; note the `cb*128` k-offset — a missing cb offset was a second bug).

### Bugs fixed this session (all in fp8.cuh)
1. **B fragment transposed layout**: staging was [k][n]; changed to [n][k] (token-major) + quantize write `y_q[token*k + col0 + tid]`.
2. **A fragment row mapping**: old code used `lane/2` rows / `(lane&1)*8` cols; true mapping is `lane%16` row, k half `(lane/16)*8`.
3. **C store/scale mapping**: old `fp8_wmma_row/col` (odd-lane formula) was wrong for 56/128 odd-lane slots; replaced by the true (l/16)*8+s, l%16.
4. **B fragment cb offset**: kk loop must add `cb*128` to the staging k index.
5. `mul_mat_fp8_scalar` (CUDA fallback) updated for the [n][k] staging (`src1_q[ni*k + cb*128 + c]`).

### Verification status
- Standalone HIP test `/tmp/fp8_correctness.cpp` (k=256 m=64 n=32 random data): **PASS** (0/2048 bad, max rel err ~7e-6).
- Stress test `/tmp/fp8_stress.cpp`: 12 configs, k=128..1024 / m=16..128 / n=16..256, all PASS (max rel err <= 3.4e-3, pure fp8 accumulation-order noise).
- Full HIP build + test-backend-ops rebuild OK. **NOTE: must re-run `cmake .` in the build dir to re-evaluate `file(GLOB "*.cu")` now that fp8.cu exists** (the dir was configured before fp8.cu was added). test-backend-ops: f8_e4m3 MUL_MAT shows "not supported [CPU]" (correct skip; CPU reference can't compute fp8) and runs on ROCm0/1/2; 0 FAILs.
- Probe/verification files in /tmp: `wmma_probe4.cpp` (definitive bijective-A map), `verify_layout.cpp` (real-data 256/256 slot check), `verify_chain.cpp` (full 8-kk chain), `fp8_stress.cpp`, `fp8_correctness.cpp`. Analysis: `convert_probe4.py` (host→true position conversion), `brute*.py`.

### Remaining M2 work
- Perf check (t/s vs f16 baseline, memory ~2.6GB) — needs a runnable model; blocked until M3 (safetensors loader) or a quick synthetic benchmark.
- Optional: CUDA sm_89/sm_90 path is compile-guarded + scalar fallback only (untested on NVIDIA).

## 5. Next steps (in order)

1. ~~Fix the C-fragment mapping~~ **DONE (2026-08-05)**: root cause was the gfx12 fp8 wmma fragment layout itself (A and B fragment loads wrong; staging needed the [n][k] transpose; see section 4). `fp8_correctness` + stress suite PASS. If test-backend-ops was built before fp8.cu existed, re-run `cmake .` first (GLOB re-eval).
2. ~~Perf check~~ **DONE (2026-08-06)**: see section 4. Generation on 1x R9700: Q8_0 GGUF 91.7 t/s, BF16 safetensors 59-72 t/s, **FP8 safetensors 72.5 t/s** (was 3.2 t/s after the fixes below). Prompt: FP8 143 t/s vs Q8_0 348 t/s. M2 kernel work is DONE except the CUDA sm_89/sm_90 mma path (compile-guarded, untested).
3. **M3 — direct safetensors loader: DONE (2026-08-06)** for the qwen35 text path. See section 9 for the full writeup. Remaining M3 items: mmproj/vision companion verification (mtmd-cli + server with --mmproj), edge cases (single-file no-index, non-128 dims, E5M2 rejection, corrupt headers).
4. **M4**: parity vs HF greedy (parity_llama.cpp in /tmp + parity_check.py in model dir) — mostly done: llama.cpp-BF16-via-loader == HF-BF16 byte-exact over 20 tokens; llama.cpp-FP8 matches BF16/HF for the first 9 tokens then diverges on a True/False near-tie (expected activation-quantization divergence; fp8 matches HF farther than Q8_0 does). Remaining: PPL sanity, multi-GPU row-split parity, memory footprint check (~5.3GB expected on 1 GPU: 1319+1319+2092 MiB split across 3 in the default split).
5. **M5 (optional)**: gguf-py F8_E4M3 constants + `--preserve-fp8` in conversion/base.py.

## 6. Key references
- `implementation-plan.md` (repo root) — full plan, M1 marked done, M2 spike results documented.
- `/llm/models/Qwen3.5/4B/StewFP8/FORMAT.md` — byte-level format spec (authoritative).
- StewFP8 dir: 945 tensors, 207 fp8+scale_inv pairs, layers-0..31.safetensors + outside.safetensors (299: embeds/norms/vision) + mtp.safetensors (22). All dims multiples of 128.
- llama.cpp patterns: `ggml/src/ggml-cuda/mmq.cu` (launcher), `mmq.cuh`, `quantize.cu` (`quantize_mmq_q8_1_cuda`), `common.cuh` (`ggml_cuda_kernel_launch`, `ggml_cuda_pool_alloc`, arch macros `GGML_CUDA_CC_IS_RDNA4`), `vendors/hip.h` (RDNA4 define).
- Original probe files in /tmp: `wmma_probe*.cpp`, `fp8_correctness.cpp` (may be cleaned; recreate from session history if needed).

## 7. Gotchas
- HIP `__shfl_xor_sync` macro requires a 4th `width` argument (unlike CUDA).
- `__global__` kernel definitions must be visible to the HOST pass (for device-stub generation) — put RDNA4-only types/builtins inside `#if defined(GGML_USE_HIP) && defined(RDNA4)` WITHIN the kernel body, and give the `#else` branch a trivial `(void)` body. Forward declarations alone do NOT generate the stub (link error `__device_stub__`).
- `ext_vector_type` is clang-only (fine for HIP; nvcc CUDA would need different types — CUDA path is compile-verified only, scalar fallback kernel).
- ggml-cuda CMakeLists uses `file(GLOB "*.cu")` — new .cu files are picked up automatically, but only after re-running `cmake .` in the build dir (the GLOB is evaluated at configure time; a build dir configured before fp8.cu existed will silently miss it).
- `RDNA4` macro comes from `vendors/hip.h` when `__GFX12__` defined (device pass only).
- supports_op gets `dev_ctx->device` for the per-device cc — needed to exclude the gfx1036 from fp8.
- The `ggml_cuda_mul_mat` dispatch requires src1 F32 + dst F32 for custom kernels (else cublas → dequant → fails for fp8); supports_op gates this.

## 8. Session state at handoff
- Working tree: M1 changes committed (`d56412c8c`). Uncommitted: fp8.cuh, fp8.cu (new, kernel CORRECTED — fragment layout fixed), ggml-cuda.cu (dispatch+supports_op), ggml-quants.c/.h + ggml.c (host quantize additions).
- `/tmp/llama-hip-build` — configured HIP build (targets: ggml-hip, test-backend-ops); re-ran `cmake .` so fp8.cu is included; both targets build.
- The earlier `wmma_probe3` "hang" was a host-side infinite loop in its byte-assignment `while (used[b])` (all 256 bytes exhausted) — NOT a GPU wedge; the GPUs were never stuck. probe4 replaced probe3 as the definitive mapping probe.
- **Do NOT commit/push without explicit user approval.**

## 9. M3 — direct safetensors loader (DONE 2026-08-06, qwen35 text path)

### Files
- `src/llama-safetensors.h/.cpp` (new): path detection, config.json + index.json + per-file header parsing (nlohmann), mmap via llama_mmap, HF->GGUF tensor mapping with per-tensor transforms, gguf metadata synthesis, tensor data callback.
- `src/llama.cpp`: `llama_model_load_from_file` auto-detects safetensors (dir with config.json + *.safetensors, or a *.safetensors file), builds metadata, runs the D3 hardware gates, calls `llama_model_load_from_file_impl(metadata, set_tensor_data, &loader, ...)` with load_mode NONE.
- `src/llama-model-loader.h/.cpp`: 3 small fixes to the metadata-only (files.empty) path: (a) `get_weight` reports tensors present in the gguf metadata (dummy_weight) so the mtp_only checks work; (b) `create_tensor` returns nullptr for TENSOR_SKIP and for NOT_REQUIRED tensors missing from metadata (matches the file-backed semantics — output/nextn.embed_tokens fall back to duplicated/absent); (c) buft_for_tensor nullptr handled instead of asserted.
- `src/CMakeLists.txt`: added llama-safetensors.cpp + `../vendor` include dir.
- `ggml/src/ggml-cuda/fp8.cu/.cuh`, `ggml-cuda.cu`: see perf fixes in section 4.

### Transforms (all validated byte-exact vs an independent Python reference, 14/14)
- fp8 re-blocking: per output row r and 128-col block cb, write [f32 scale][128 fp8 bytes], scale = bf16(scale_inv[src_r/128][src_cb]) promoted to f32. Row/column V-head reorders applied in the same pass (permutation is block-preserving for head_dim 128).
- V-head reorder (16 K-heads != 32 V-heads, num_v_per_k=2): applied to in_proj_qkv rows, in_proj_z rows, in_proj_a/b rows, A_log/dt_bias elements, conv1d V channels, out_proj columns — and to the fp8 scale tensors in lockstep. Permutations verified against conversion/qwen.py `_reorder_v_heads`.
- `A_log` -> `-exp` (rounded through bf16 when the HF source is bf16; direct f32 for F32 sources), `dt_bias` -> `ssm_dt.bias`, conv1d squeeze [C,1,K]->[K,C], norms `+1` (bf16-rounded) except linear_attn.norm, MTP mapping mtp.* -> blk.32.* (fc->nextn.eh_proj, pre_fc_norm_embedding->enorm, pre_fc_norm_hidden->hnorm, norm->shared_head_norm).
- Output types mirror conversion/base.py: 1D tensors, `*_norm.weight` and ssm_conv1d are F32; 2D .weight are BF16 (or F8_E4M3). CRITICAL: the GPU binary ops (mul) do not support BF16 operands, so norms MUST be F32.
- Source dtype awareness: the ORIGINAL model stores A_log and linear_attn.norm as F32 (mamba_ssm_dtype float32); StewFP8 converted them to BF16. The loader reads the per-tensor dtype from the safetensors header.
- tie_word_embeddings: output.weight is absent; the loader fix makes the model fall back to the duplicated token_embd (same as the GGUF flow).
- Tokenizer: `convert_hf_to_gguf.py <dir> --vocab-only` -> `<dir>/tokenizer.gguf` (already generated for StewFP8 and SafeTensors); the loader copies all `tokenizer.*` KV pairs into the synthesized context; clear one-liner error if missing.

### D3 hardware gate
- `has_fp8_device()`: probes every registered device with a dummy F8 mul_mat op (ggml_backend_dev_supports_op). Fires when the model has fp8 tensors and no fp8-capable device: "FP8_E4M3 weights require a device with native FP8 support (RDNA4, or NVIDIA Ada/Hopper+); this system has none. Use an integer GGUF (e.g. Q8_0) instead."
- Partial offload: fp8 tensors live in every decoder block, so --n-gpu-layers must cover all of them (n_layer_all + 1 = 34): "FP8_E4M3 weights must be fully offloaded to the GPU - increase --n-gpu-layers to at least 34". Verified on CPU-only build and with -ngl 10.
- The natural buffer-selection error ("failed to find a compatible buffer type") is a further backstop.

### Perf fixes discovered while measuring (M2)
- **CRITICAL launcher bug**: the RDNA4 dispatch block in ggml_cuda_mul_mat_fp8 was wrapped in `#if defined(GGML_USE_HIP) && defined(RDNA4)` — RDNA4 is a DEVICE-pass macro, so the host build compiled the whole dispatch out and silently used the scalar fallback (3.2 t/s!). The dispatch is now an unguarded runtime `if (GGML_CUDA_CC_IS_RDNA4(cc))` (kernel bodies keep their #if guard for the device pass).
- **dot4 GEMV path** (`mul_mat_fp8_gemv`): for n <= 16 the wmma kernel wasted the 16-token tiles (batch 1 = 1/64 useful). One warp per output row, 32 lanes read the 128 weight bytes coalesced + dot4 with the shared activation block, warp-reduce, per-block scale. Verified correct (max rel err <= 2.2e-3) and it fixed the batch-1 efficiency.
- **GLU fusion gate**: `ggml_cuda_should_fuse_mul_mat_vec_q` now excludes F8_E4M3 (the fused-mmvq path has no F8 case and aborted at generation).
- Result: 3.2 -> 72.5 t/s generation (vs 91.7 Q8_0 baseline, 59-72 BF16).

### Verification status
- Loader transform validation: 14/14 byte-exact vs Python reference (fp8 re-blocks, V-reorders, +1, -exp, conv1d, F32 types, MTP, token_embd). Also validated the BF16 model's reorders.
- End-to-end: `llama-cli -m <StewFP8 dir>` generates coherent text on 1 GPU (72 t/s) and 3 GPUs (58 t/s); --spec-type draft-mtp works (38 t/s); BF16 SafeTensors dir loads and runs (59-72 t/s).
- Parity: llama.cpp-BF16-via-loader == HF-BF16 greedy, byte-exact over 20 tokens ("Paris.\nA. True\nB. False\nAnswer:\nA\n\n"); llama.cpp-FP8 matches through 9 tokens then flips a True/False near-tie; Q8_0 diverges at token 10. HF-side parity_check.py (BF16 vs FP8-dequant in torch): PASS, byte-identical.
- Negative tests: CPU-only build -> exact D3 error; -ngl 10 -> exact partial-offload error.

### Known gaps / next
- mmproj/vision companion not verified (text loader skips model.visual.*; need convert_hf_to_gguf.py --mmproj + mtmd-cli/server run).
- --vocab-file override not plumbed (loader auto-detects <dir>/tokenizer.gguf only).
- Edge cases untested: single-file no-index safetensors, non-128 dims, E5M2 rejection, corrupt headers.
- CUDA sm_89/sm_90 fp8 mma path still compile-guarded/untested (scalar fallback).
- ~1-ulp expf differences vs numpy for F32-source A_log (numerically irrelevant; the bf16-sourced StewFP8 path rounds through bf16 and matches byte-exact).
