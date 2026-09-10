#include "gated_delta_net.cuh"
#include "gated_delta_net_chunked.cuh"
#include "ggml-cuda/common.cuh"

template <int S_v, bool KDA, bool keep_rs_t>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K,
                                     int           beta_sigmoid) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        // beta_sigmoid: beta is the pre-sigmoid value (the UNARY(sigmoid) node was fused in); same expression as
        // unary.cu op_sigmoid
        const float beta_val = beta_sigmoid ? 1.0f / (1.0f + expf(-*beta_t)) : *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, int beta_sigmoid, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, beta_sigmoid);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, beta_sigmoid);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, beta_sigmoid);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, beta_sigmoid);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache,
        const ggml_tensor * beta_presig = nullptr) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    // beta_presig: the input of a fused UNARY(sigmoid) beta node (contiguous, same element order as the contiguous beta,
    // so beta's strides index it); the sequential kernel applies the
    // sigmoid.  Only the decode/verify band uses it (n_seqs == 1, n_tokens <= 16 never takes the chunked path below).
    const int     beta_sigmoid = beta_presig != nullptr;
    const float * b_d = (const float *) (beta_presig ? beta_presig->data : src_beta->data);
    GGML_ASSERT(!beta_sigmoid || (n_seqs == 1 && n_tokens <= 16 && ggml_is_contiguous(beta_presig) &&
                                  ggml_nelements(beta_presig) == ggml_nelements(src_beta)));

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // op param 1 is the largest per-seq batch that can be rolled back into (0 = unset). The
    // rollback snapshots only cover K tokens and only the sequential kernel writes them, so a
    // batch longer than this bound must not take the chunked path. K alone is NOT a safe bound:
    // the snapshot depth comes from the draft-model n_max, while ngram-style speculators can draft
    // far longer (ngram-mod n_max = 64 by default). A verify batch then holds 1 + n_max tokens and
    // is exactly the batch that gets rolled back into, so the bound must cover the longest draft.
    const int64_t n_rs_batch = keep_rs ? std::max<int64_t>(1, ggml_get_op_params_i32(dst, 1)) : 0;

    // K-independent chunked prefill -- no K-dependent boundary and no sequential tail.  The
    // chunked kernel is not bit-exact with the sequential kernel (the fp32 path differs only in
    // rounding order; the opt-in bf16/WMMA path is materially lossy -- issue #113), so a
    // K-dependent boundary makes the
    // post-prefill state depend on n_rs_seq: plain decode (K == 1) chunks the whole prompt while
    // the MTP path (K == n_max + 1) chunks n_tokens - K + a K-token tail, and the two disagree, so
    // greedy near-ties flip between --spec-type none and draft-mtp.
    //
    // The alignment is done by giving BOTH paths the SAME call -- chunk the whole batch, exactly
    // what K == 1 does -- rather than by sharing a sequential tail.  A batch with more tokens than
    // max(K, 16, n_rs_batch) cannot be a speculative verify batch (a verify batch decodes the
    // sampled token plus the longest draft any enabled speculator can produce)
    // and is therefore never rolled back into, so the K rollback snapshots are useless to it and
    // the sequential kernel can be skipped entirely: zero cost, unbounded n_max, no KTAIL.  Every
    // batch at or below the threshold -- in particular every verify batch -- falls through to the
    // sequential kernel below, which writes the K snapshots the spec rollback actually reads.
    // (Verified: over llama-cli draft-mtp and llama-server with --cache-reuse, every partial
    // rollback was preceded by a batch of exactly K or fewer tokens.)
    //
    // The threshold must be a constant for K <= 16 or the two paths diverge again on short prompts;
    // 16 covers K <= 16 (n_max <= 15, including adaptive MTP's recommended n_max = 12).  The floor
    // at K keeps verify batches sequential for deeper drafts, so those keep the K-dependent
    // boundary and stay correct-but-not-bit-identical, exactly as before.
    //
    // The fused GDN->cpy (cache != nullptr) is still honoured: the chunked kernel writes the new
    // state (slot 0) straight into the cache slot, so the cpy node can be skipped.  n_seqs > 1
    // keeps the whole-ubatch chunked path for K == 1 -- the chunked n_tokens override is also the
    // sequence stride, so only n_seqs == 1 may choose per-batch.  GGML_CUDA_GDN_CHUNKED=0 forces
    // the sequential kernel everywhere (the A/B switch, and the fully-snapshot-safe fallback).
    // The chunked ops report a synchronous launch rejection (ROCm: AQL dispatch refused, e.g. a
    // code object carrying a hidden hostcall buffer on a platform without PCIe atomics ->
    // hipErrorIllegalState, llama-cpp-rdna-boosts#2); a rejected launch has no side effects, so we
    // recompute with the sequential kernel below, bit-identical to GGML_CUDA_GDN_CHUNKED=0.  An
    // async fault inside a launched kernel is NOT caught here and surfaces at the next sync point.
    if (!kda && n_tokens > 1 &&
        (S_v == 16 || S_v == 32 || S_v == 64 || S_v == 128)) {
        static const char * env = getenv("GGML_CUDA_GDN_CHUNKED");
        if (env == nullptr || strcmp(env, "0") != 0) {
            // Minimum batch size that may take the whole-batch chunked path.  A constant for
            // K <= 16 (see above); the floors at K and n_rs_batch keep deeper drafts and every
            // possible verify batch on the sequential kernel, which writes the rollback snapshots.
            const int64_t GDN_CHUNKED_MIN_TOKENS = std::max<int64_t>(K > 16 ? (int64_t) K : 16, n_rs_batch);
            // GGML_CUDA_GDN_CHUNKED_BF16 / arch selection: the bf16/WMMA chunked kernel is the
            // DEFAULT for S_v == 128 prefill on RDNA3/RDNA4 (opt out with `=0`).  It is
            // near-lossless, NOT bit-exact (bf16 operands, fp32 accumulation, ~1e-5 op NMSE);
            // the r29 default-off was a workaround for the n_seqs > 1 `A_sc` stride aliasing
            // bug (llama-cpp-rdna-boosts#113), fixed in r30.  The fp32 chunked kernel is the
            // opt-out and the non-RDNA fallback.  n_tokens_limit > 0 truncates to a single-seq
            // prefix.
            auto launch_chunked = [&](float * state_d_ext, int64_t n_tokens_limit) -> bool {
#if defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
                const int cc_c = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
                const bool bf16_rdna_c = GGML_CUDA_CC_IS_RDNA4(cc_c) || GGML_CUDA_CC_IS_RDNA3(cc_c);
                static const char * envb_c = getenv("GGML_CUDA_GDN_CHUNKED_BF16");
                const bool want_bf16_c = bf16_rdna_c && S_v == 128 &&
                    (envb_c == nullptr || strcmp(envb_c, "0") != 0);
                if (want_bf16_c) {
                    if (GGML_CUDA_CC_IS_RDNA4(cc_c)) {
                        return ggml_cuda_op_gated_delta_net_chunked_bf16(ctx, dst, state_d_ext, n_tokens_limit);
                    }
                    return ggml_cuda_op_gated_delta_net_chunked_bf16_gfx11(ctx, dst, state_d_ext, n_tokens_limit);
                }
#endif
                return ggml_cuda_op_gated_delta_net_chunked(ctx, dst, state_d_ext, n_tokens_limit);
            };

            if (n_seqs == 1) {
                if (n_tokens > GDN_CHUNKED_MIN_TOKENS) {
                    if (launch_chunked(cache ? cache->data : nullptr, 0)) {
                        static bool logged_free = false;
                        if (!logged_free) {
                            GGML_LOG_INFO("%s: whole-batch chunked GDN prefill n=%ld K=%d n_seqs=%ld "
                                          "(> %ld: no snapshot tail needed)\n",
                                          __func__, (long) n_tokens, K, (long) n_seqs,
                                          (long) GDN_CHUNKED_MIN_TOKENS);
                            logged_free = true;
                        }
                        return;
                    }
                    GGML_LOG_WARN("%s: chunked GDN launch rejected by the driver; falling back to the sequential kernel (GGML_CUDA_GDN_CHUNKED=0 equivalent)\n", __func__);
                }
            } else if (K == 1) {
                if (launch_chunked(cache ? cache->data : nullptr, 0)) {
                    return;
                }
                GGML_LOG_WARN("%s: chunked GDN launch rejected by the driver; falling back to the sequential kernel (GGML_CUDA_GDN_CHUNKED=0 equivalent)\n", __func__);
            }
        }
    }

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, beta_sigmoid, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, beta_sigmoid, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, beta_sigmoid, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, beta_sigmoid, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}

void ggml_cuda_op_gated_delta_net_beta_sigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * beta_presig,
                                               const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, cache, beta_presig);
}
