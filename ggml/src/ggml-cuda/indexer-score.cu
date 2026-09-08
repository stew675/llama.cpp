// Fused indexer block-pool -> rms-norm -> rope -> score for the qwen4exp QSA
// sparse-attention decode path (ONE kernel replaces ~12 per-op kernels/layer).
//
// Replaces the per-op chain, replicating its F32 arithmetic EXACTLY so the output
// (the per-block score vector consumed by ggml_indexer_top_k) is byte-identical:
//   members  = get_rows(K, blk_cells)                       (half->f32 upcast gather)
//   pooled   = (sum over r block members in order) * (1/r)
//   normed   = rms_norm(pooled, W, eps)                     (256-thread block reduce)
//   rot      = rope_multi(normed, blk_pos, ...)             (IMROPE half-pair rotate)
//   dot[b][h] = sum_d rot[d][b]*q[d][h]                     (mmvf F32 vec kernel order)
//   score[b]  = bias[b] + sum_h relu(dot[b][h])             (head sum in h order)
//
// src0 = K    raw indexer cache [idx_dim, n_kv, n_stream] F32/BF16/F16 (view of the cache)
// src1 = BLK  block->cell map [r*n_blocks, n_stream] I32 (blk_cells)
// src2 = POS  mrope positions [4*n_blocks*n_stream] I32 (blk_pos, [k][row] stride)
// src3 = Q    rotated/normed indexer query [idx_dim, n_idx_h*n_tps, n_stream] F32
// src4 = W    rms-norm weights [idx_dim] F32 (index_k_norm)
// src5 = B    per-block bias [n_blocks, n_tps, n_stream] F32 (applied when blk_bias)
// op_params : 0=r(i32) 1=n_dims(i32) 2=mode(i32) 3=n_ctx_orig(i32)
//             4=freq_base(f32) 5=freq_scale(f32) 6=ext_factor(f32) 7=attn_factor(f32)
//             8=beta_fast(f32) 9=beta_slow(f32) 10..13=sections(i32 x4) 14=eps(f32)
// dst      : [n_blocks, n_tps, n_stream] F32 (per-op path's post-bias score)
//            row r0 = b + s*n_blocks.

#include "indexer-score.cuh"
#include "common.cuh"
#include <algorithm>

// --- rope math copied verbatim from ggml-cuda/rope.cu (rope_yarn) so the cos/sin
// --- values are bit-identical to the per-op ggml rope kernel ---------------------
static __device__ float indexer_score_rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

static __device__ void indexer_score_rope_yarn(
        const float theta_extrap, const float freq_scale, const float corr_v0, const float corr_v1,
        const int64_t i0, const float ext_factor, const float mscale,
        float & cos_theta, float & sin_theta) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        const float ramp_mix = indexer_score_rope_yarn_ramp(corr_v0, corr_v1, i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
    }
    cos_theta = cosf(theta) * mscale;
    sin_theta = sinf(theta) * mscale;
}

// --- pool + norm device helpers -----------------------------------------------
// cache element ktype: 0 = F32, 1 = BF16 (ggml bits<<16 upcast), 2 = F16 (exact half->f32)
static __device__ __forceinline__ float indexer_score_ld_val(const void * p, const int ktype) {
    if (ktype == 1) {
        return __uint_as_float(((uint32_t) *(const uint16_t *) p) << 16);
    }
    if (ktype == 2) {
        // f16 -> f32 is exact (bijection); matches get_rows' native half upcast
        return __half2float(*(const __half *) p);
    }
    return *(const float *) p;
}

static __device__ __forceinline__ float indexer_score_norm_col(
        const char * K, const int * blk_row, const int c, const int idx_dim,
        const int r, const float rinv, const int64_t nb0K, const int64_t nb1K, const int ktype) {
    float acc = 0.0f;
    const char * Kcol = K + (int64_t) c * nb0K;
    for (int i = 0; i < r; ++i) {
        acc += indexer_score_ld_val(Kcol + (int64_t) blk_row[i] * nb1K, ktype);
    }
    return acc * rinv;
}

static constexpr int IDX_SCORE_BLOCK = 256;   // rms_norm geometry for ncols < 1024
static constexpr int IDX_SCORE_WARP  = 32;    // wave32 (RDNA): mmvf's F32 vec reduce width

static __global__ void indexer_score_kernel(
        const char   * __restrict__ K,       // raw cache [idx_dim, n_kv, n_stream]
        const int    * __restrict__ BLK,     // blk_cells [r*n_blocks, n_stream]
        const int32_t* __restrict__ POS,     // blk_pos [4*nrows]
        const float  * __restrict__ Q,       // q_rot [idx_dim, n_idx_h*n_tps, n_stream]
        const float  * __restrict__ W,       // norm weights [idx_dim]
        const float  * __restrict__ B,       // per-block bias [n_blocks, n_tps, n_stream]
        float        * __restrict__ dst,     // [n_blocks, n_tps, n_stream]
        const int    idx_dim,                // 128
        const int    r,
        const float  rinv,
        const int    n_blocks,               // blocks per stream
        const int    n_stream,
        const int    nrows,                  // n_blocks*n_stream
        const int    n_idx_h,                // query heads (<= 8)
        const int    n_dims,                 // rope n_dims (n_rot)
        const int    is_imrope,              // mode == GGML_ROPE_TYPE_IMROPE
        const int    sect_v0, const int sect_v1, const int sect_v2, const int sect_v3,
        const int64_t nb0K, const int64_t nb1K, const int64_t nb2K, const int64_t nb1B,
        const char  * __restrict__ P,      // derived pool [idx_dim, n_blocks, n_stream] (nullable)
        const int   * __restrict__ LIM,    // derived limit leaf [2*n_stream] (nullable)
        const int64_t nb0P, const int64_t nb1P, const int64_t nb2P,
        const int    ktype,
        const float  eps,
        const float  freq_scale, const float ext_factor, const float attn_factor,
        const float  theta_scale, const float corr_v0, const float corr_v1) {
    const int r0    = blockIdx.x;            // merged (block, stream) row
    const int s     = r0 / n_blocks;
    const int b     = r0 - s * n_blocks;
    const int tid   = threadIdx.x;

    const char  * Ks = K + (int64_t) s * nb2K;
    const int   * blk_row = (const int *) ((const char *) BLK + (int64_t) s * nb1B) + (int64_t) r * b;
    const int32_t p0 = POS[r0];
    const int32_t p1 = POS[r0 + nrows];
    const int32_t p2 = POS[r0 + 2 * nrows];
    const int32_t p3 = POS[r0 + 3 * nrows];

    extern __shared__ float smem[];
    float * vec = smem;                            // pooled then normed/rotated [idx_dim]
    float * s_vals = smem + idx_dim;               // block_reduce warp leaders
    float * w1     = smem + idx_dim + 32;          // dot warp-1 totals bridge [n_idx_h]

    // --- derived rows: rows below the per-stream limit read the precomputed pool vector
    // --- (already pooled + normed + rotated by the fill op); rows at/above it run the
    // --- raw pool -> norm -> rope passes (dead/spare + not-yet-filled rows, as before) --
    const bool derived_row = P != nullptr && b < LIM[n_stream + s];
    if (derived_row) {
        // idx_dim (128) < BLOCK (256): threads [0, idx_dim) load one element each
        if (tid < idx_dim) {
            vec[tid] = *(const float *) (P + (int64_t) s * nb2P + (int64_t) b * nb1P + (int64_t) tid * nb0P);
        }
    } else {
        // --- pass 1: pool columns (increment-1 arithmetic, strided like rms_norm_f32) ---
        float tmp = 0.0f;
        for (int c = tid; c < idx_dim; c += IDX_SCORE_BLOCK) {
            const float pv = indexer_score_norm_col(Ks, blk_row, c, idx_dim, r, rinv, nb0K, nb1K, ktype);
            vec[c] = pv;
            tmp += pv * pv;
        }
        __syncthreads();

        const float mean  = block_reduce<block_reduce_method::SUM, IDX_SCORE_BLOCK>(tmp, s_vals) / (float) idx_dim;
        const float scale = rsqrtf(mean + eps);

        // --- pass 2: normed[c] = scale * pooled[c] * W[c] (rms_norm_f32 mul order) -----
        for (int c = tid; c < idx_dim; c += IDX_SCORE_BLOCK) {
            vec[c] = scale * vec[c] * W[c];
        }
        __syncthreads();

        // --- pass 3: rope the first n_dims channels (half-pair (j, j+n_dims/2)) --------
        const int n_pairs = n_dims / 2;
        if (tid < n_pairs) {
            const int sect_dims = sect_v0 + sect_v1 + sect_v2 + sect_v3;
            const int sector    = tid % sect_dims;    // iw/2 = j = tid (n_offs = 0)
            const int iw        = 2 * tid;

            float theta_base;
            if (is_imrope) {
                if (sector % 3 == 1 && sector < 3 * sect_v1) {                       // h
                    theta_base = (float) p1 * powf(theta_scale, iw / 2.0f);
                } else if (sector % 3 == 2 && sector < 3 * sect_v2) {                // w
                    theta_base = (float) p2 * powf(theta_scale, iw / 2.0f);
                } else if (sector % 3 == 0 && sector < 3 * sect_v0) {                // t
                    theta_base = (float) p0 * powf(theta_scale, iw / 2.0f);
                } else {
                    theta_base = (float) p3 * powf(theta_scale, iw / 2.0f);
                }
            } else {
                if (sector < sect_v0) {
                    theta_base = (float) p0 * powf(theta_scale, iw / 2.0f);
                } else if (sector < sect_v0 + sect_v1) {
                    theta_base = (float) p1 * powf(theta_scale, iw / 2.0f);
                } else if (sector < sect_v0 + sect_v1 + sect_v2) {
                    theta_base = (float) p2 * powf(theta_scale, iw / 2.0f);
                } else {
                    theta_base = (float) p3 * powf(theta_scale, iw / 2.0f);
                }
            }

            float cos_theta;
            float sin_theta;
            indexer_score_rope_yarn(theta_base, freq_scale, corr_v0, corr_v1, iw, ext_factor, attn_factor,
                                    cos_theta, sin_theta);

            const float x0 = vec[tid];
            const float x1 = vec[tid + n_pairs];
            vec[tid]                = x0 * cos_theta - x1 * sin_theta;
            vec[tid + n_pairs]      = x0 * sin_theta + x1 * cos_theta;
        }
        __syncthreads();
    }
    // the derived row fill is visible to every thread of the block before the dot pass
    __syncthreads();

    // --- pass 4: per-block dot vs each query head (mmvf F32 vec kernel order) -------
    // mmvf for ne00 == 128 on wave32: block of 64 threads, lane t owns float2 (2t,2t+1)
    // of the src0 row, accumulating `acc += v*u` twice; then per-warp xor-tree reduce
    // and the two 32-lane warp totals added.  Replicate exactly (idx_dim == 128).
    float psum[8] = {0.0f};
    if (tid < 64) {
        // scalar loads (values identical to mmvf's float2 path; no alignment constraint)
        const float xa = vec[2 * tid];
        const float xb = vec[2 * tid + 1];
        for (int h = 0; h < n_idx_h; ++h) {
            const int64_t qo = (int64_t) h * idx_dim + 2 * tid;
            float acc = 0.0f;
            acc += xa * Q[qo];
            acc += xb * Q[qo + 1];
            psum[h] = acc;
        }
    }
    __syncthreads();

    // per-warp xor-tree reduce over the 32 lanes holding partials (warps 0 and 1)
    if (tid < 64) {
        const int wid = tid / IDX_SCORE_WARP;
        for (int h = 0; h < n_idx_h; ++h) {
            float v = psum[h];
#pragma unroll
            for (int off = IDX_SCORE_WARP / 2; off > 0; off >>= 1) {
                v += __shfl_xor_sync(0xffffffff, v, off, IDX_SCORE_WARP);
            }
            if (tid % IDX_SCORE_WARP == 0) {
                if (wid == 1) {
                    w1[h] = v;                 // warp-1 total -> smem bridge
                } else {
                    psum[h] = v;               // warp-0 total stays in lane 0
                }
            }
        }
    }
    __syncthreads();

    // --- epilogue on thread 0: total = w0 + w1, relu, head sum in h order, + bias ---
    if (tid == 0) {
        float acc = 0.0f;
        for (int h = 0; h < n_idx_h; ++h) {
            const float dot = psum[h] + w1[h];
            const float rv  = fmaxf(dot, 0.0f);
            acc = (h == 0) ? rv : acc + rv;
        }
        acc += B[r0];
        dst[r0] = acc;
    }
}

static void ggml_cuda_indexer_score_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * K   = dst->src[0];
    const ggml_tensor * BLK = dst->src[1];
    const ggml_tensor * POS = dst->src[2];
    const ggml_tensor * Q   = dst->src[3];
    const ggml_tensor * W   = dst->src[4];
    const ggml_tensor * B   = dst->src[5];

    const int   r     = ggml_get_op_params_i32(dst, 0);
    const int   n_dims  = ggml_get_op_params_i32(dst, 1);
    const int   mode    = ggml_get_op_params_i32(dst, 2);
    const int   n_ctx_orig = ggml_get_op_params_i32(dst, 3);
    float freq_base;   memcpy(&freq_base,   (int32_t *) dst->op_params +  4, sizeof(float));
    float freq_scale;  memcpy(&freq_scale,  (int32_t *) dst->op_params +  5, sizeof(float));
    float ext_factor;  memcpy(&ext_factor,  (int32_t *) dst->op_params +  6, sizeof(float));
    float attn_factor; memcpy(&attn_factor, (int32_t *) dst->op_params +  7, sizeof(float));
    float beta_fast;   memcpy(&beta_fast,   (int32_t *) dst->op_params +  8, sizeof(float));
    float beta_slow;   memcpy(&beta_slow,   (int32_t *) dst->op_params +  9, sizeof(float));
    const int sect_v[4] = {
        ggml_get_op_params_i32(dst, 10),
        ggml_get_op_params_i32(dst, 11),
        ggml_get_op_params_i32(dst, 12),
        ggml_get_op_params_i32(dst, 13),
    };
    const float eps = ggml_get_op_params_f32(dst, 14);

    const int idx_dim  = (int) K->ne[0];
    const int n_kv     = (int) K->ne[1];
    const int n_stream = (int) K->ne[2];
    const int n_idx_h  = (int) Q->ne[1];

    GGML_ASSERT(idx_dim == 128);                       // dot geometry assumes 64 float2 pairs
    GGML_ASSERT(idx_dim < 1024);                       // 256-thread norm block
    GGML_ASSERT(r > 0);
    GGML_ASSERT(n_dims <= idx_dim && n_dims % 2 == 0);
    GGML_ASSERT(B->ne[0] == dst->ne[0]);
    GGML_ASSERT(mode == GGML_ROPE_TYPE_IMROPE);        // fused path targets the IMROPE model
    GGML_ASSERT(n_idx_h <= 8);                         // the mmvf F32 oracle's bound

    const int64_t n_cells = BLK->ne[0];
    GGML_ASSERT(n_cells % r == 0);
    const int n_blocks = (int) (n_cells / r);
    GGML_ASSERT(n_blocks == dst->ne[0]);
    const int nrows = n_blocks * n_stream;

    const int ktype = K->type == GGML_TYPE_F32 ? 0 : (K->type == GGML_TYPE_BF16 ? 1 : 2);
    GGML_ASSERT((K->type == GGML_TYPE_F32 || K->type == GGML_TYPE_BF16 || K->type == GGML_TYPE_F16) && BLK->type == GGML_TYPE_I32 &&
                POS->type == GGML_TYPE_I32 && Q->type == GGML_TYPE_F32 && W->type == GGML_TYPE_F32);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);
    float corr_v[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_v);

    // optional derived-cache path: src6 = pool view, src7 = per-stream limit leaf
    const ggml_tensor * P = dst->src[6];
    const ggml_tensor * LIM = dst->src[7];
    if (P != nullptr) {
        GGML_ASSERT(LIM != nullptr && LIM->type == GGML_TYPE_I32);
        GGML_ASSERT(P->type == GGML_TYPE_F32);
        GGML_ASSERT(P->ne[0] == idx_dim && P->ne[1] == n_blocks && P->ne[2] == n_stream);
        GGML_ASSERT(LIM->ne[0] == 2 * n_stream);
    }

    cudaStream_t stream = ctx.stream();
    const dim3 blocks_num(nrows);
    const size_t smem = (idx_dim + 32 + 32) * sizeof(float);
    const ggml_cuda_kernel_launch_params launch_params = {blocks_num, dim3(IDX_SCORE_BLOCK, 1, 1), smem, stream};
    ggml_cuda_kernel_launch(indexer_score_kernel, launch_params,
        (const char *) K->data, (const int *) BLK->data, (const int32_t *) POS->data,
        (const float *) Q->data, (const float *) W->data, (const float *) B->data,
        (float *) dst->data,
        idx_dim, r, 1.0f / (float) r, n_blocks, n_stream, nrows, n_idx_h, n_dims,
        mode == GGML_ROPE_TYPE_IMROPE ? 1 : 0,
        sect_v[0], sect_v[1], sect_v[2], sect_v[3],
        K->nb[0], K->nb[1], K->nb[2], BLK->nb[1],
        P != nullptr ? (const char *) P->data : nullptr,
        P != nullptr ? (const int *) LIM->data : nullptr,
        P != nullptr ? P->nb[0] : 4, P != nullptr ? P->nb[1] : (int64_t) idx_dim * 4,
        P != nullptr ? P->nb[2] : 0,
        ktype, eps, freq_scale, ext_factor, attn_factor, theta_scale, corr_v[0], corr_v[1]);
}

void ggml_cuda_indexer_score(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_indexer_score_impl(ctx, dst);
}

bool ggml_cuda_indexer_score_supported(int device, const ggml_tensor * dst) {
    GGML_ASSERT(dst->op == GGML_OP_INDEXER_SCORE);
    GGML_UNUSED(device);
    const ggml_tensor * K = dst->src[0];
    const ggml_tensor * BLK = dst->src[1];
    const ggml_tensor * POS = dst->src[2];
    const ggml_tensor * Q = dst->src[3];
    const ggml_tensor * W = dst->src[4];
    const ggml_tensor * B = dst->src[5];
    if (!((K->type == GGML_TYPE_F32 || K->type == GGML_TYPE_BF16 || K->type == GGML_TYPE_F16) && BLK->type == GGML_TYPE_I32 &&
           POS->type == GGML_TYPE_I32 && Q->type == GGML_TYPE_F32 && W->type == GGML_TYPE_F32 &&
           B->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32)) {
        return false;
    }
    if (dst->src[6] != nullptr) {
        const ggml_tensor * P = dst->src[6];
        const ggml_tensor * LIM = dst->src[7];
        if (LIM == nullptr || LIM->type != GGML_TYPE_I32 || P->type != GGML_TYPE_F32) {
            return false;
        }
    }
    return true;
}

// --------------------------------------------------------------------------------
// Derived-cache fill: pool + norm + rope ONE kernel over the per-step range of full
// blocks [from_s, lim_s) and write the normed+rotated vectors into the pool rows.
// The arithmetic replicates the score kernel's raw pass 1-3 byte-exactly, so a pool
// row equals what the score would compute from the raw cache (the toggle contract).
// --------------------------------------------------------------------------------

static __global__ void indexer_fill_kernel(
        const char   * __restrict__ K,       // raw cache [idx_dim, n_kv, n_stream]
        const int    * __restrict__ BLK,     // blk_cells [r*n_blocks, n_stream]
        const int32_t* __restrict__ POS,     // blk_pos [4*nrows]
        const float  * __restrict__ W,       // norm weights [idx_dim]
        const int    * __restrict__ RNG,     // range leaf [2*n_stream]: from in [0,ns), lim in [ns,2ns)
        float        * __restrict__ dst,     // pool view [idx_dim, n_blocks, n_stream]
        const int    idx_dim,
        const int    r,
        const float  rinv,
        const float  eps,
        const int    n_blocks,
        const int    n_stream,
        const int    nrows,
        const int    n_dims,
        const int    is_imrope,
        const int    sect_v0, const int sect_v1, const int sect_v2, const int sect_v3,
        const int64_t nb0K, const int64_t nb1K, const int64_t nb2K, const int64_t nb1B,
        const int64_t nb0D, const int64_t nb1D, const int64_t nb2D,
        const int    ktype,
        const float  freq_scale, const float ext_factor, const float attn_factor,
        const float  theta_scale, const float corr_v0, const float corr_v1) {
    const int tid = threadIdx.x;

    // grid-stride over the merged (block, stream) rows: the launch grid is capped (see the
    // impl), so a steady-state decode step (0-1 rows in the range) pays a tiny grid, while a
    // post-prefill backfill (up to n_kv/r rows) strides through the range in a few iterations
    for (int r0 = blockIdx.x; r0 < nrows; r0 += gridDim.x) {
        const int s   = r0 / n_blocks;
        const int b   = r0 - s * n_blocks;

        const int from = RNG[s];
        const int lim  = RNG[n_stream + s];
        if (b < from || b >= lim) {
            continue;                        // not part of this step's completed range
        }

    const char  * Ks = K + (int64_t) s * nb2K;
    const int   * blk_row = (const int *) ((const char *) BLK + (int64_t) s * nb1B) + (int64_t) r * b;
    const int32_t p0 = POS[r0];
    const int32_t p1 = POS[r0 + nrows];
    const int32_t p2 = POS[r0 + 2 * nrows];
    const int32_t p3 = POS[r0 + 3 * nrows];

    extern __shared__ float smem[];
    float * vec = smem;                            // pooled then normed/rotated [idx_dim]
    float * s_vals = smem + idx_dim;               // block_reduce warp leaders

    // pass 1: pool columns (score kernel's raw pass, same strided walk + reduction)
    float tmp = 0.0f;
    for (int c = tid; c < idx_dim; c += IDX_SCORE_BLOCK) {
        const float pv = indexer_score_norm_col(Ks, blk_row, c, idx_dim, r, rinv, nb0K, nb1K, ktype);
        vec[c] = pv;
        tmp += pv * pv;
    }
    __syncthreads();

    const float mean  = block_reduce<block_reduce_method::SUM, IDX_SCORE_BLOCK>(tmp, s_vals) / (float) idx_dim;
    const float scale = rsqrtf(mean + eps);

    // pass 2: normed[c] = scale * pooled[c] * W[c]
    for (int c = tid; c < idx_dim; c += IDX_SCORE_BLOCK) {
        vec[c] = scale * vec[c] * W[c];
    }
    __syncthreads();

    // pass 3: rope the first n_dims channels (score kernel's pass 3, verbatim)
    const int n_pairs = n_dims / 2;
    if (tid < n_pairs) {
        const int sect_dims = sect_v0 + sect_v1 + sect_v2 + sect_v3;
        const int sector    = tid % sect_dims;
        const int iw        = 2 * tid;

        float theta_base;
        if (is_imrope) {
            if (sector % 3 == 1 && sector < 3 * sect_v1) {                       // h
                theta_base = (float) p1 * powf(theta_scale, iw / 2.0f);
            } else if (sector % 3 == 2 && sector < 3 * sect_v2) {                // w
                theta_base = (float) p2 * powf(theta_scale, iw / 2.0f);
            } else if (sector % 3 == 0 && sector < 3 * sect_v0) {                // t
                theta_base = (float) p0 * powf(theta_scale, iw / 2.0f);
            } else {
                theta_base = (float) p3 * powf(theta_scale, iw / 2.0f);
            }
        } else {
            if (sector < sect_v0) {
                theta_base = (float) p0 * powf(theta_scale, iw / 2.0f);
            } else if (sector < sect_v0 + sect_v1) {
                theta_base = (float) p1 * powf(theta_scale, iw / 2.0f);
            } else if (sector < sect_v0 + sect_v1 + sect_v2) {
                theta_base = (float) p2 * powf(theta_scale, iw / 2.0f);
            } else {
                theta_base = (float) p3 * powf(theta_scale, iw / 2.0f);
            }
        }

        float cos_theta;
        float sin_theta;
        indexer_score_rope_yarn(theta_base, freq_scale, corr_v0, corr_v1, iw, ext_factor, attn_factor,
                                cos_theta, sin_theta);

        const float x0 = vec[tid];
        const float x1 = vec[tid + n_pairs];
        vec[tid]                = x0 * cos_theta - x1 * sin_theta;
        vec[tid + n_pairs]      = x0 * sin_theta + x1 * cos_theta;
    }
    __syncthreads();

    // write the normed+rotated row into the pool buffer (view strides from the caller)
    float * prow = (float *) ((char *) dst + (int64_t) s * nb2D + (int64_t) b * nb1D);
    for (int c = tid; c < idx_dim; c += IDX_SCORE_BLOCK) {
        *(float *) ((char *) prow + (int64_t) c * nb0D) = vec[c];
    }
    // the next grid-stride iteration reuses the shared vec: make the writes visible first
    __syncthreads();
    }   // end grid-stride loop
}

static void ggml_cuda_indexer_fill_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * K   = dst->src[0];
    const ggml_tensor * BLK = dst->src[1];
    const ggml_tensor * POS = dst->src[2];
    const ggml_tensor * W   = dst->src[3];
    const ggml_tensor * RNG = dst->src[4];

    const int   r     = ggml_get_op_params_i32(dst, 0);
    const int   n_dims  = ggml_get_op_params_i32(dst, 1);
    const int   mode    = ggml_get_op_params_i32(dst, 2);
    const int   n_ctx_orig = ggml_get_op_params_i32(dst, 3);
    float freq_base;   memcpy(&freq_base,   (int32_t *) dst->op_params +  4, sizeof(float));
    float freq_scale;  memcpy(&freq_scale,  (int32_t *) dst->op_params +  5, sizeof(float));
    float ext_factor;  memcpy(&ext_factor,  (int32_t *) dst->op_params +  6, sizeof(float));
    float attn_factor; memcpy(&attn_factor, (int32_t *) dst->op_params +  7, sizeof(float));
    float beta_fast;   memcpy(&beta_fast,   (int32_t *) dst->op_params +  8, sizeof(float));
    float beta_slow;   memcpy(&beta_slow,   (int32_t *) dst->op_params +  9, sizeof(float));
    const int sect_v[4] = {
        ggml_get_op_params_i32(dst, 10),
        ggml_get_op_params_i32(dst, 11),
        ggml_get_op_params_i32(dst, 12),
        ggml_get_op_params_i32(dst, 13),
    };
    const float eps = ggml_get_op_params_f32(dst, 14);

    const int idx_dim  = (int) K->ne[0];
    const int n_stream = (int) K->ne[2];

    GGML_ASSERT(idx_dim < 1024);                       // 256-thread norm block
    GGML_ASSERT(r > 0);
    GGML_ASSERT(n_dims <= idx_dim && n_dims % 2 == 0);
    GGML_ASSERT(mode == GGML_ROPE_TYPE_IMROPE);        // fused rope path targets the IMROPE model
    GGML_ASSERT(RNG->type == GGML_TYPE_I32 && RNG->ne[0] == 2 * n_stream);

    const int64_t n_cells = BLK->ne[0];
    GGML_ASSERT(n_cells % r == 0);
    const int n_blocks = (int) (n_cells / r);
    GGML_ASSERT(n_blocks == dst->ne[1]);
    const int nrows = n_blocks * n_stream;

    const int ktype = K->type == GGML_TYPE_F32 ? 0 : (K->type == GGML_TYPE_BF16 ? 1 : 2);
    GGML_ASSERT((K->type == GGML_TYPE_F32 || K->type == GGML_TYPE_BF16 || K->type == GGML_TYPE_F16) &&
                BLK->type == GGML_TYPE_I32 && POS->type == GGML_TYPE_I32 && W->type == GGML_TYPE_F32);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);
    float corr_v[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_v);

    cudaStream_t stream = ctx.stream();
    // cap the fill grid: steady decode completes <=1 block per step, so the launch must be cheap;
    // a post-prefill backfill strides through the whole range with the same capped grid
    const int grid = std::min(nrows, 512);
    const dim3 blocks_num(grid);
    const size_t smem = (idx_dim + 32) * sizeof(float);
    const ggml_cuda_kernel_launch_params launch_params = {blocks_num, dim3(IDX_SCORE_BLOCK, 1, 1), smem, stream};
    ggml_cuda_kernel_launch(indexer_fill_kernel, launch_params,
        (const char *) K->data, (const int *) BLK->data, (const int32_t *) POS->data,
        (const float *) W->data, (const int *) RNG->data, (float *) dst->data,
        idx_dim, r, 1.0f / (float) r, eps, n_blocks, n_stream, nrows, n_dims,
        mode == GGML_ROPE_TYPE_IMROPE ? 1 : 0,
        sect_v[0], sect_v[1], sect_v[2], sect_v[3],
        K->nb[0], K->nb[1], K->nb[2], BLK->nb[1],
        dst->nb[0], dst->nb[1], dst->nb[2],
        ktype, freq_scale, ext_factor, attn_factor, theta_scale, corr_v[0], corr_v[1]);
}

void ggml_cuda_indexer_fill(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_indexer_fill_impl(ctx, dst);
}

bool ggml_cuda_indexer_fill_supported(int device, const ggml_tensor * dst) {
    GGML_ASSERT(dst->op == GGML_OP_INDEXER_FILL);
    GGML_UNUSED(device);
    const ggml_tensor * K   = dst->src[0];
    const ggml_tensor * BLK = dst->src[1];
    const ggml_tensor * POS = dst->src[2];
    const ggml_tensor * W   = dst->src[3];
    const ggml_tensor * RNG = dst->src[4];
    return (K->type == GGML_TYPE_F32 || K->type == GGML_TYPE_BF16 || K->type == GGML_TYPE_F16) && BLK->type == GGML_TYPE_I32 &&
           POS->type == GGML_TYPE_I32 && W->type == GGML_TYPE_F32 && RNG->type == GGML_TYPE_I32 &&
           dst->type == GGML_TYPE_F32;
}
