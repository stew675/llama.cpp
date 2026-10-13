#include "hyperconn.cuh"
#include "ggml-backend-impl.h"
#include "mmb.cuh"

// Explicitly rounded mul/add: keeps the fused kernels from being contracted into FMAs so that
// the results match the separate MUL and ADD kernels of the unfused graph bit for bit.
#if defined(__HIP_PLATFORM_AMD__)
static __device__ __forceinline__ float hc_mul_rn(const float a, const float b) {
    float result;
    asm("v_mul_f32_e32 %0, %1, %2" : "=v"(result) : "v"(a), "v"(b));
    return result;
}

static __device__ __forceinline__ float hc_add_rn(const float a, const float b) {
    float result;
    asm("v_add_f32_e32 %0, %1, %2" : "=v"(result) : "v"(a), "v"(b));
    return result;
}
#else
static __device__ __forceinline__ float hc_mul_rn(const float a, const float b) {
    return __fmul_rn(a, b);
}

static __device__ __forceinline__ float hc_add_rn(const float a, const float b) {
    return __fadd_rn(a, b);
}
#endif

// same expression as op_sigmoid in unary.cu
static __device__ __forceinline__ float hc_sigmoid(const float x) {
    return 1.0f / (1.0f + expf(-x));
}

// BF16 <-> F32 with round-to-nearest-even, identical to mmb_cvt_f32_bf16 and the HC16 producers.
static __device__ __forceinline__ float hc_bf2f32(const uint16_t h) { return __uint_as_float(((uint32_t) h) << 16); }
static __device__ __forceinline__ uint16_t hc_f2bf32(const float f) {
    uint32_t u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}

// mixed[e,t] = scale * ( ((xn*sig)[0] + (xn*sig)[1]) + ... ) + bias, streams summed in order 0..hc-1
static __global__ void hc_mix_reduce_f32(
        const float * __restrict__ xn, const float * __restrict__ gate, float * __restrict__ dst,
        const int64_t n_embd, const int64_t n_tokens, const int hc, const float scale, const float bias) {
    const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= n_embd * n_tokens) {
        return;
    }

    const int64_t t = index / n_embd;
    const int64_t e = index - t * n_embd;
    const int64_t hc_dim = n_embd * hc;
    const int64_t base = t * hc_dim + e;

    float acc = hc_mul_rn(xn[base], hc_sigmoid(gate[base]));
    for (int c = 1; c < hc; ++c) {
        const int64_t i = base + int64_t(c) * n_embd;
        acc = hc_add_rn(acc, hc_mul_rn(xn[i], hc_sigmoid(gate[i])));
    }

    // identical expression to scale_f32 in scale.cu (RN: x*scale+bias, no fma)
    dst[index] = hc_add_rn(hc_mul_rn(scale, acc), bias);
}

// out[e,c,t] = residual[e,c,t] + block_out[e,t] * ( s2 * sigmoid(s1 * inject[c,t] + b1) + b2 )
static __global__ void hc_combine_f32(
        const float * __restrict__ residual, const float * __restrict__ block_out, const float * __restrict__ inject,
        float * __restrict__ dst, const int64_t n_embd, const int hc, const int64_t n_tokens,
        const float s1, const float b1, const float s2, const float b2) {
    const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t hc_dim = n_embd * hc;
    if (index >= hc_dim * n_tokens) {
        return;
    }

    const int64_t t   = index / hc_dim;
    const int64_t rem = index - t * hc_dim;
    const int64_t c   = rem / n_embd;
    const int64_t e   = rem - c * n_embd;

    // identical expressions to scale_f32 (scale.cu) and op_sigmoid (unary.cu)
    const float x1 = hc_add_rn(hc_mul_rn(s1, inject[t * hc + c]), b1);
    const float x2 = hc_sigmoid(x1);
    const float w  = hc_add_rn(hc_mul_rn(s2, x2), b2);

    const float m = hc_mul_rn(block_out[t * n_embd + e], w);
    dst[index] = hc_add_rn(residual[index], m);
}

static bool hc_ranges_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const char * a0 = (const char *) a->data;
    const char * a1 = a0 + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);
    const char * b0 = (const char *) b->data;
    const char * b1 = b0 + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);
    return a0 < b1 && b0 < a1;
}

void ggml_cuda_op_hc_mix_reduce(ggml_backend_cuda_context & ctx, const ggml_cuda_hc_mix_args & args) {
    const ggml_tensor * xn   = args.xn;
    const ggml_tensor * gate = args.gate;
    ggml_tensor *       dst  = args.dst;

    GGML_ASSERT(xn->type == GGML_TYPE_F32 && gate->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(xn) && ggml_is_contiguous(gate) && ggml_is_contiguous(dst));
    GGML_ASSERT(args.hc > 0);

    const int64_t n_embd   = dst->ne[0];
    const int64_t n_tokens = ggml_nrows(dst);
    GGML_ASSERT(xn->ne[0] == n_embd * args.hc && ggml_nrows(xn) == n_tokens);
    GGML_ASSERT(ggml_are_same_shape(xn, gate));

    // The graph allocator may reuse the buffer of the (elided) sigmoid/mul intermediates, which can
    // coincide with the gate matmul output. Every thread reads hc strided elements, so an aliased
    // output has to be staged -- unless (single token) the output lands on a whole stream of the
    // aliased input: element e of the output then overwrites element e of that stream, which only
    // the same thread reads (after it has finished reading).
    auto stream_aligned_alias = [&](const ggml_tensor * in) {
        if (!hc_ranges_overlap(dst, in)) {
            return true;
        }
        if (n_tokens != 1) {
            return false;
        }
        const ptrdiff_t off = (const char *) dst->data - (const char *) in->data;
        return off >= 0 && off % (n_embd * (ptrdiff_t) sizeof(float)) == 0 &&
            off + (ptrdiff_t) ggml_nbytes(dst) <= (ptrdiff_t) ggml_nbytes(in);
    };
    const bool stage = !stream_aligned_alias(xn) || !stream_aligned_alias(gate);

    ggml_cuda_pool_alloc<float> staged(ctx.pool());
    float * out = stage ? staged.alloc(n_embd * n_tokens) : (float *) dst->data;

    const int64_t n_items = n_embd * n_tokens;
    const int     threads = 256;
    const int     blocks  = (int) ((n_items + threads - 1) / threads);
    const ggml_cuda_kernel_launch_params launch_params(blocks, threads, 0, ctx.stream());
    ggml_cuda_kernel_launch(hc_mix_reduce_f32, launch_params,
        (const float *) xn->data, (const float *) gate->data, out, n_embd, n_tokens, args.hc, args.scale, args.bias);

    if (stage) {
        CUDA_CHECK(cudaMemcpyAsync(dst->data, out, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, ctx.stream()));
    }
}

void ggml_cuda_op_hc_combine(ggml_backend_cuda_context & ctx, const ggml_cuda_hc_combine_args & args) {
    const ggml_tensor * residual  = args.residual;
    const ggml_tensor * block_out = args.block_out;
    const ggml_tensor * inject    = args.inject;
    ggml_tensor *       dst       = args.dst;

    GGML_ASSERT(residual->type == GGML_TYPE_F32 && block_out->type == GGML_TYPE_F32 &&
                inject->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(residual) && ggml_is_contiguous(block_out) &&
                ggml_is_contiguous(inject) && ggml_is_contiguous(dst));

    const int64_t n_embd   = dst->ne[0];
    const int64_t hc       = dst->ne[1];
    const int64_t n_tokens = dst->ne[2] * dst->ne[3];
    GGML_ASSERT(ggml_are_same_shape(residual, dst));
    GGML_ASSERT(ggml_nelements(block_out) == n_embd * n_tokens);
    GGML_ASSERT(ggml_nelements(inject) == hc * n_tokens);

    // residual may be the in-place source of the add (same address, same layout): each thread then
    // reads and writes its own element only. Any other overlap needs staging.
    const bool residual_inplace = residual->data == dst->data && ggml_are_same_layout(residual, dst);
    const bool stage = (!residual_inplace && hc_ranges_overlap(dst, residual)) ||
                       hc_ranges_overlap(dst, block_out) || hc_ranges_overlap(dst, inject);

    ggml_cuda_pool_alloc<float> staged(ctx.pool());
    float * out = stage ? staged.alloc(ggml_nelements(dst)) : (float *) dst->data;

    const int64_t n_items = ggml_nelements(dst);
    const int     threads = 256;
    const int     blocks  = (int) ((n_items + threads - 1) / threads);
    const ggml_cuda_kernel_launch_params launch_params(blocks, threads, 0, ctx.stream());
    ggml_cuda_kernel_launch(hc_combine_f32, launch_params,
        (const float *) residual->data, (const float *) block_out->data, (const float *) inject->data, out,
        n_embd, (int) hc, n_tokens, args.s1, args.b1, args.s2, args.b2);

    if (stage) {
        CUDA_CHECK(cudaMemcpyAsync(dst->data, out, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, ctx.stream()));
    }
}

// ---------------------------------------------------------------------------------------------------------
// combine + grouped rms norm, one token: block per stream, 1024 threads (the rms_norm_f32<1024> layout)
//   w        = s2 * sigmoid(s1 * inject[c] + b1) + b2
//   res[e,c] = residual[e,c] + block_out[e] * w
//   xn[e,c]  = rms_norm(res[:,c]) * gamma[e,c]

#define HC_CN_BLOCK   1024
#define HC_CN_MAX_EMB 3072

static __global__ void __launch_bounds__(HC_CN_BLOCK, 1) hc_combine_norm_f32(
        const float * __restrict__ inject, const float * __restrict__ residual,
        const float * __restrict__ block_out, const float * __restrict__ gamma,
        float * __restrict__ out_res, float * __restrict__ out_xn,
        uint16_t * __restrict__ out_xn_bf16, const bool store_xn_f32,
        const uint16_t * __restrict__ res_in_bf16, uint16_t * __restrict__ res_out_bf16,
        const uint16_t * __restrict__ blk_in_bf16,
        const int n_embd, const bool bo_hc,
        const float s1, const float b1, const float s2, const float b2, const float eps) {
    __shared__ float s_sum[32];

    const int c   = blockIdx.x;   // stream
    const int t   = blockIdx.y;   // token
    const int hc  = gridDim.x;
    const int tid = threadIdx.x;

    // identical expressions to scale_f32 / op_sigmoid / scale_f32
    const float x1 = hc_add_rn(hc_mul_rn(s1, inject[(int64_t) t * hc + c]), b1);
    const float x2 = hc_sigmoid(x1);
    const float w  = hc_add_rn(hc_mul_rn(s2, x2), b2);

    const int64_t row = (int64_t) t * hc + c;
    const float * res = residual + row * n_embd;
    const uint16_t * res16 = res_in_bf16 ? res_in_bf16 + row * n_embd : nullptr;
    float *       dst = out_res  + row * n_embd;
    uint16_t *    dst16 = res_out_bf16 ? res_out_bf16 + row * n_embd : nullptr;
    block_out += (bo_hc ? row : (int64_t) t) * n_embd;
    const uint16_t * blk16 = blk_in_bf16 ? blk_in_bf16 + (bo_hc ? row : (int64_t) t) * n_embd : nullptr;
    float xs[3];
    float tmp = 0.0f;
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const int col = tid + k * HC_CN_BLOCK;
        xs[k] = 0.0f;
        if (col < n_embd) {
            const float m  = hc_mul_rn(blk16 ? hc_bf2f32(blk16[col]) : block_out[col], w);
            const float xi = hc_add_rn(res16 ? hc_bf2f32(res16[col]) : res[col], m);
            if (dst16) { dst16[col] = hc_f2bf32(xi); } else { dst[col] = xi; }
            xs[k]    = xi;
            tmp += xi * xi;
        }
    }

    tmp = block_reduce<block_reduce_method::SUM, HC_CN_BLOCK>(tmp, s_sum);

    const float mean  = tmp / n_embd;
    const float scale = rsqrtf(mean + eps);

    const float * g  = gamma  + (int64_t) c * n_embd;
    float *       xn = out_xn + row * n_embd;
    uint16_t *    xh = out_xn_bf16 ? out_xn_bf16 + row * n_embd : nullptr;
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const int col = tid + k * HC_CN_BLOCK;
        if (col < n_embd) {
            const float v = hc_mul_rn(hc_mul_rn(scale, xs[k]), g[col]);
            if (store_xn_f32) xn[col] = v;
            if (xh) xh[col] = hc_f2bf32(v);
        }
    }
}

// Same computation for one token with all streams in a single block, streams processed one after the other
// with exactly the per-stream expressions and block reduction above. Every input element is loaded into
// registers before the first store, so the outputs may alias any of the inputs (the graph allocator reuses
// the block_out buffer for xn when block_out dies at the combine).
#define HC_CN_SINGLE_MAX_HC 4

static __global__ void __launch_bounds__(HC_CN_BLOCK, 1) hc_combine_norm_single_f32(
        const float * inject, const float * residual, const float * block_out, const float * __restrict__ gamma,
        float * out_res, float * out_xn,
        uint16_t * __restrict__ out_xn_bf16, const bool store_xn_f32,
        const uint16_t * __restrict__ res_in_bf16, uint16_t * __restrict__ res_out_bf16,
        const uint16_t * __restrict__ blk_in_bf16,
        const int n_embd, const int hc, const bool bo_hc,
        const float s1, const float b1, const float s2, const float b2, const float eps) {
    __shared__ float s_sum[HC_CN_SINGLE_MAX_HC][32];

    const int t       = blockIdx.x;   // token
    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // 1. every input element of the token into registers
    float w[HC_CN_SINGLE_MAX_HC];
#pragma unroll
    for (int c = 0; c < HC_CN_SINGLE_MAX_HC; ++c) {
        w[c] = 0.0f;
        if (c < hc) {
            // identical expressions to scale_f32 / op_sigmoid / scale_f32
            const float x1 = s1 * inject[(int64_t) t * hc + c] + b1;
            const float x2 = hc_sigmoid(x1);
            w[c] = s2 * x2 + b2;
        }
    }

    const int64_t bo_row_base = bo_hc ? (int64_t) t * hc : (int64_t) t;
    float xs[HC_CN_SINGLE_MAX_HC][3];
    float tmp[HC_CN_SINGLE_MAX_HC];
#pragma unroll
    for (int c = 0; c < HC_CN_SINGLE_MAX_HC; ++c) {
        tmp[c] = 0.0f;
    }
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const int col = tid + k * HC_CN_BLOCK;
#pragma unroll
        for (int c = 0; c < HC_CN_SINGLE_MAX_HC; ++c) {
            xs[c][k] = 0.0f;
            if (c < hc && col < n_embd) {
                // single-copy block_out broadcasts over c: its row is t regardless of the stream
                const int64_t bo_off = (bo_row_base + (bo_hc ? (int64_t) c : 0)) * n_embd + col;
                const int64_t roff   = ((int64_t) t * hc + c) * n_embd + col;
                const float rv = res_in_bf16 ? hc_bf2f32(res_in_bf16[roff]) : residual[roff];
                const float bv = blk_in_bf16 ? hc_bf2f32(blk_in_bf16[bo_off]) : block_out[bo_off];
                const float m  = hc_mul_rn(bv, w[c]);
                const float xi = hc_add_rn(rv, m);
                xs[c][k] = xi;
                tmp[c] += xi * xi;
            }
        }
    }
    __syncthreads(); // all loads of the block have completed before the first store

    // 2. per-stream block sums, same reduction tree as block_reduce<SUM, 1024> for each stream
#pragma unroll
    for (int c = 0; c < HC_CN_SINGLE_MAX_HC; ++c) {
        tmp[c] = warp_reduce_sum(tmp[c]);
    }
    if (lane_id == 0) {
#pragma unroll
        for (int c = 0; c < HC_CN_SINGLE_MAX_HC; ++c) {
            s_sum[c][warp_id] = tmp[c];
        }
    }
    __syncthreads();
#pragma unroll
    for (int c = 0; c < HC_CN_SINGLE_MAX_HC; ++c) {
        tmp[c] = lane_id < HC_CN_BLOCK / WARP_SIZE ? s_sum[c][lane_id] : 0.0f;
        tmp[c] = warp_reduce_sum(tmp[c]);
    }

    // 3. stores
#pragma unroll
    for (int c = 0; c < HC_CN_SINGLE_MAX_HC; ++c) {
        if (c < hc) {
            const float mean  = tmp[c] / n_embd;
            const float scale = rsqrtf(mean + eps);
            const int64_t row = (int64_t) t * hc + c;
            float *       dst = out_res + row * n_embd;
            float *       xn  = out_xn  + row * n_embd;
            const float * g   = gamma   + (int64_t) c * n_embd;
#pragma unroll
            for (int k = 0; k < 3; ++k) {
                const int col = tid + k * HC_CN_BLOCK;
                if (col < n_embd) {
                    if (res_out_bf16) res_out_bf16[row * n_embd + col] = hc_f2bf32(xs[c][k]);
                    else              dst[col] = xs[c][k];
                    const float v = hc_mul_rn(hc_mul_rn(scale, xs[c][k]), g[col]);
                    if (store_xn_f32) xn[col] = v;
                    if (out_xn_bf16)  out_xn_bf16[row * n_embd + col] = hc_f2bf32(v);
                }
            }
        }
    }
}

bool ggml_cuda_hc_combine_norm_supported(const ggml_cuda_hc_combine_norm_args & a, const int warp_size) {
    const int64_t n_embd = a.out_res->ne[0];
    const int64_t hc     = a.out_res->ne[1];
    return warp_size == 32 && n_embd >= 1024 && n_embd <= HC_CN_MAX_EMB && hc >= 1 && hc <= 16 &&
        ggml_nelements(a.gamma) == n_embd * hc;
}

void ggml_cuda_op_hc_combine_norm(ggml_backend_cuda_context & ctx, const ggml_cuda_hc_combine_norm_args & a) {
    const int64_t n_embd   = a.out_res->ne[0];
    const int64_t hc       = a.out_res->ne[1];
    const int64_t n_tokens = a.out_res->ne[2] * a.out_res->ne[3];

    GGML_ASSERT(a.out_res->ne[3] == 1);
    // out_xn is the gamma-MUL result: [n_embd*hc, T] (2D), out_res is [n_embd, hc, T] (3D);
    // both are contiguous F32 with one shared row-major layout, so the kernel addresses them
    // identically via row = (t*hc + c).
    GGML_ASSERT(a.out_xn->type == GGML_TYPE_F32 && ggml_is_contiguous(a.out_xn));
    GGML_ASSERT(ggml_nelements(a.out_xn) == ggml_nelements(a.out_res));
    // out_xn is the gamma-MUL result: either [n_embd*hc, T] (2D, pre-rebase graphs) or
    // [n_embd, hc, T] (3D, the current qwen4exp graph). Both are contiguous F32 with the same
    // row-major layout, which is all the kernel addresses (row = t*hc + c).
    GGML_ASSERT((a.out_xn->ne[0] == n_embd * hc) ||
                (a.out_xn->ne[0] == n_embd && a.out_xn->ne[1] == hc));
    GGML_ASSERT(ggml_are_same_shape(a.out_res, a.residual));
    GGML_ASSERT(ggml_nelements(a.block_out) == n_embd * n_tokens || ggml_nelements(a.block_out) == n_embd * hc * n_tokens);
    GGML_ASSERT(ggml_nelements(a.gamma) == n_embd * hc && ggml_nelements(a.inject) == hc * n_tokens);

    if (a.single_block) {
        GGML_ASSERT(hc <= HC_CN_SINGLE_MAX_HC);
        const ggml_cuda_kernel_launch_params launch_params(dim3((int) n_tokens, 1, 1), HC_CN_BLOCK, 0, ctx.stream());
        ggml_cuda_kernel_launch(hc_combine_norm_single_f32, launch_params,
            (const float *) a.inject->data, (const float *) a.residual->data,
            (const float *) a.block_out->data, (const float *) a.gamma->data,
            (float *) a.out_res->data, (float *) a.out_xn->data,
            a.out_xn_bf16, a.store_xn_f32, a.res_in_bf16, a.res_out_bf16, a.blk_in_bf16,
            (int) n_embd, (int) hc, a.block_out_hc, a.s1, a.b1, a.s2, a.b2, a.eps);
        return;
    }

    const ggml_cuda_kernel_launch_params launch_params(dim3((int) hc, (int) n_tokens, 1), HC_CN_BLOCK, 0, ctx.stream());
    ggml_cuda_kernel_launch(hc_combine_norm_f32, launch_params,
        (const float *) a.inject->data, (const float *) a.residual->data,
        (const float *) a.block_out->data, (const float *) a.gamma->data,
        (float *) a.out_res->data, (float *) a.out_xn->data,
        a.out_xn_bf16, a.store_xn_f32, a.res_in_bf16, a.res_out_bf16, a.blk_in_bf16,
        (int) n_embd, a.block_out_hc, a.s1, a.b1, a.s2, a.b2, a.eps);
}
// try and fuse nodes and return the number of nodes to skip
int ggml_cuda_match_hc_mix(const ggml_cgraph * cgraph, const int i, ggml_cuda_hc_mix_args & args, enum ggml_op * ops) {
    const ggml_tensor * node = cgraph->nodes[i];
    if (!(node->op == GGML_OP_UNARY && ggml_get_unary_op(node) == GGML_UNARY_OP_SIGMOID && node->type == GGML_TYPE_F32 &&
        i + 5 < cgraph->n_nodes &&
        cgraph->nodes[i + 1]->op == GGML_OP_MUL && cgraph->nodes[i + 2]->op == GGML_OP_RESHAPE &&
        cgraph->nodes[i + 3]->op == GGML_OP_VIEW && cgraph->nodes[i + 4]->op == GGML_OP_CONT)) {
        return 0;
    }
    const ggml_tensor * gate  = node->src[0];
    const ggml_tensor * mul   = cgraph->nodes[i + 1];
    const ggml_tensor * view0 = cgraph->nodes[i + 3];
    const ggml_tensor * cont  = cgraph->nodes[i + 4];
    const ggml_tensor * xn    = mul->src[0] == node ? mul->src[1] : mul->src[0];

    const int64_t n_embd   = cont->ne[0];
    const int64_t n_tokens = cont->ne[1];
    const int64_t hc_dim   = mul->ne[0];
    const int     hc       = n_embd > 0 ? (int) (hc_dim / n_embd) : 0;
    const size_t  row_size = hc_dim * sizeof(float);

    auto stream_view_ok = [&](const ggml_tensor * v, int c) {
        return v->op == GGML_OP_VIEW && v->view_src == mul && v->type == GGML_TYPE_F32 &&
            v->view_offs == (size_t) c * n_embd * sizeof(float) &&
            v->ne[0] == n_embd && v->ne[1] == n_tokens && v->ne[2] == 1 && v->ne[3] == 1 &&
            v->nb[0] == sizeof(float) && v->nb[1] == row_size;
    };

    bool ok = (mul->src[0] == node || mul->src[1] == node) && xn != node &&
        mul->type == GGML_TYPE_F32 && xn->type == GGML_TYPE_F32 && gate->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(xn) && ggml_is_contiguous(gate) && ggml_are_same_shape(xn, gate) &&
        ggml_are_same_shape(mul, xn) && ggml_are_same_shape(mul, node) &&
        mul->ne[2] == 1 && mul->ne[3] == 1 && ggml_nrows(mul) == n_tokens &&
        hc >= 2 && hc <= 16 && hc_dim == (int64_t) hc * n_embd &&
        cgraph->nodes[i + 2]->view_src == mul && cont->src[0] == view0 && stream_view_ok(view0, 0) &&
        cont->type == GGML_TYPE_F32 && ggml_is_contiguous(cont) && cont->ne[2] == 1 && cont->ne[3] == 1;

    const int n_ops = ok ? 5 + 2 * (hc - 1) + 1 : 0;
    ok = ok && i + n_ops <= cgraph->n_nodes;
    if (!ok) {
        return 0;
    }

    ops[0] = GGML_OP_UNARY; ops[1] = GGML_OP_MUL; ops[2] = GGML_OP_RESHAPE; ops[3] = GGML_OP_VIEW; ops[4] = GGML_OP_CONT;
    const ggml_tensor * prev = cont;
    for (int c = 1; ok && c < hc; ++c) {
        const ggml_tensor * v   = cgraph->nodes[i + 5 + 2 * (c - 1)];
        const ggml_tensor * add = cgraph->nodes[i + 6 + 2 * (c - 1)];
        ok = stream_view_ok(v, c) && add->op == GGML_OP_ADD && add->type == GGML_TYPE_F32 &&
            add->src[0] == prev && add->src[1] == v && ggml_are_same_shape(add, cont);
        ops[5 + 2 * (c - 1)] = GGML_OP_VIEW;
        ops[6 + 2 * (c - 1)] = GGML_OP_ADD;
        prev = add;
    }
    ggml_tensor * scale = cgraph->nodes[i + n_ops - 1];
    ops[n_ops - 1] = GGML_OP_SCALE;
    ok = ok && scale->op == GGML_OP_SCALE && scale->src[0] == prev && scale->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(scale) && ggml_are_same_shape(scale, cont);
    if (!ok) {
        return 0;
    }

    args.xn    = xn;
    args.gate  = gate;
    args.dst   = scale;
    args.hc    = hc;
    args.scale = ggml_get_op_params_f32(scale, 0);
    args.bias  = ggml_get_op_params_f32(scale, 1);
    return n_ops;
}

// the hc_mix window must be closed (its only external consumer is the fused output) before the
// gate GEMM can be folded into it: otherwise the gate tensor the GEMM no longer materializes could
// still be read by another node.
int ggml_cuda_hc_mix_closed(const ggml_cgraph * cgraph, const int i, ggml_cuda_hc_mix_args & args) {
    enum ggml_op ops[5 + 2 * 15 + 1];
    const int n_ops = ggml_cuda_match_hc_mix(cgraph, i, args, ops);
    if (n_ops == 0) {
        return 0;
    }
    int node_idxs[5 + 2 * 15 + 1];
    for (int j = 0; j < n_ops; ++j) {
        node_idxs[j] = i + j;
    }
    const int out_nodes[] = { i + n_ops - 1 };
    return ggml_can_fuse_subgraph_ext(cgraph, node_idxs, n_ops, ops, out_nodes, 1) ? n_ops : 0;
}

// qwen4exp hyper-connection stream mix (HC_MIX) prefill fusion: returns the number of graph
// nodes consumed (n_ops - 1) when fused, 0 otherwise.  Decode (dst->ne[1] == 1) keeps its
// bit-identical routing and is handled by the standalone HC_MIX op dispatch.
int ggml_cuda_hc_mix_try_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    static const bool disabled = [] {
        const char * f = getenv("GGML_CUDA_DISABLE_HC_FUSION");
        if (f != nullptr && atoi(f) != 0) {
            return true;
        }
        const char * m = getenv("GGML_CUDA_DISABLE_HC_MIX");
        return m != nullptr && atoi(m) != 0;
    }();
    if (disabled) {
        return 0;
    }
    ggml_cuda_hc_mix_args args;
    enum ggml_op ops[5 + 2 * 15 + 1];
    int n_ops = ggml_cuda_match_hc_mix(cgraph, i, args, ops);
    if (n_ops > 0 && args.dst->ne[1] == 1) {
        n_ops = 0; // Decode preserves bit-identical routing. Prefill uses the fused reduction.
    }
    if (n_ops <= 0) {
        return 0;
    }
    int node_idxs[5 + 2 * 15 + 1];
    for (int j = 0; j < n_ops; ++j) {
        node_idxs[j] = i + j;
    }
    const int out_nodes[] = { i + n_ops - 1 };
    if (!ggml_can_fuse_subgraph_ext(cgraph, node_idxs, n_ops, ops, out_nodes, 1)) {
        return 0;
    }
    ggml_cuda_op_hc_mix_reduce(ctx, args);
    return n_ops - 1;
}


// moved out of ggml-cuda.cu (Phase-2 self-contained extraction)
bool ggml_cuda_hc_combine_norm_identify(const ggml_cgraph * cgraph, const int i, ggml_cuda_hc_combine_norm_args & args) {
    const ggml_tensor * rep = cgraph->nodes[i];
    if (rep->op != GGML_OP_REPEAT || rep->type != GGML_TYPE_F32 || rep->ne[3] != 1) {
        return false;
    }
    const int64_t n_embd = rep->ne[0], hc = rep->ne[1], n_tok = rep->ne[2];
    if (n_embd <= 0 || hc < 2 || hc > 8 || n_tok < 1) {
        return false;
    }
    const ggml_tensor * base_root = rep->src[0];
    while (base_root != nullptr && base_root->view_src != nullptr) {
        base_root = base_root->view_src;
    }
    if (base_root == nullptr || base_root->type != GGML_TYPE_F32 || !(base_root->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }
    const int limit = std::min(cgraph->n_nodes, i + 24);
    int k = i + 1;
    while (k < limit && !(cgraph->nodes[k]->op == GGML_OP_MUL &&
            (cgraph->nodes[k]->src[0] == rep || cgraph->nodes[k]->src[1] == rep))) {
        ++k;
    }
    if (k >= limit) {
        return false;
    }
    const ggml_tensor * mul = cgraph->nodes[k];
    int m = k + 1;
    while (m < limit && ggml_cuda_is_view_or_noop(cgraph->nodes[m]) && !ggml_is_empty(cgraph->nodes[m])) {
        ++m;
    }
    if (m >= limit || cgraph->nodes[m]->op != GGML_OP_ADD) {
        return false;
    }
    const ggml_tensor * add = cgraph->nodes[m];
    int q = m + 1;
    while (q < limit && ggml_cuda_is_view_or_noop(cgraph->nodes[q]) && !ggml_is_empty(cgraph->nodes[q])) {
        ++q;
    }
    if (q >= limit || cgraph->nodes[q]->op != GGML_OP_RMS_NORM || cgraph->nodes[q]->src[0] == nullptr) {
        return false;
    }
    const ggml_tensor * rms = cgraph->nodes[q];
    const ggml_tensor * rms_src = rms->src[0];
    while (rms_src->view_src != nullptr) {
        rms_src = rms_src->view_src;
    }
    if (rms_src != add) {
        return false;
    }
    int g = q + 1;
    while (g < limit && ggml_cuda_is_view_or_noop(cgraph->nodes[g]) && !ggml_is_empty(cgraph->nodes[g])) {
        ++g;
    }
    if (g >= limit || cgraph->nodes[g]->op != GGML_OP_MUL) {
        return false;
    }
    const ggml_tensor * mulg = cgraph->nodes[g];
    const ggml_tensor * res = add->src[0] == mul ? add->src[1] : add->src[0];
    if (res == nullptr || res == mul ||
            !ggml_are_same_shape(add, mul) || !ggml_are_same_shape(add, res) || !ggml_are_same_shape(add, rms)) {
        return false;
    }
    args.inject       = nullptr;
    args.residual     = res;
    args.block_out    = base_root;
    args.block_out_hc = false;
    args.gamma        = nullptr;
    args.out_res      = const_cast<ggml_tensor *>(add);
    args.out_xn       = const_cast<ggml_tensor *>(mulg);
    return true;
}

// Fill the BF16 HC-stream pointers for a matched combine+norm window.  Each pointer is left null when
// the graph did not mark the matching tensor, so an unmarked stream keeps its F32 read/write.
void ggml_cuda_hc_combine_norm_set_bf16(ggml_backend_cuda_context & ctx, ggml_cuda_hc_combine_norm_args & args) {
    // HC16 completion: the graph optimizer already marks out_xn BF16-only (all its consumers read a
    // BF16 copy: the MMB GEMMs through the activation cache, dsv4_hc_pre through its x16 arm).  The
    // fused combine is the one producer that did not emit the copy, so every consumer reconverted the
    // F32 (the hc_norm half of the mmb_cvt_f32_bf16 traffic).  Writing it here is bit-identical to
    // the on-the-fly conversion (same RNE rounding) and lets the F32 store be dropped.
    if (ggml_cuda_mmb_active() && ggml_cuda_mmb_is_bf16_only(args.out_xn)) {
        args.out_xn_bf16  = ggml_cuda_mmb_reserve_auto(ctx, args.out_xn, (size_t) ggml_nelements(args.out_xn));
        args.store_xn_f32 = args.out_xn_bf16 == nullptr;
    }
    // BF16 HC streams (LLAMA_HC_BLK16 / LLAMA_HC_RES16, both default OFF).
    if (!ggml_cuda_mmb_blk16() && !ggml_cuda_mmb_res16()) {
        return;
    }
    const ggml_tensor * rblk = args.block_out->view_src ? args.block_out->view_src : args.block_out;
    if (ggml_cuda_mmb_blk16() && !args.block_out_hc && args.block_out->view_offs == 0 &&
            ggml_cuda_mmb_is_bf16_only(rblk)) {
        args.blk_in_bf16 = (const uint16_t *) args.block_out->data;
    }
    if (ggml_cuda_mmb_res16()) {
        const ggml_tensor * rin  = args.residual->view_src ? args.residual->view_src : args.residual;
        const ggml_tensor * rout = args.out_res->view_src  ? args.out_res->view_src  : args.out_res;
        const bool in16  = ggml_cuda_mmb_is_bf16_only(rin)  && args.residual->view_offs == 0;
        const bool out16 = ggml_cuda_mmb_is_bf16_only(rout) && args.out_res->view_offs == 0;
        if (args.out_res->data == args.residual->data && in16 != out16) {
            GGML_ABORT("hc_combine_norm: in-place residual with mismatched BF16 marks (%s)", args.out_res->name);
        }
        args.res_in_bf16  = in16  ? (const uint16_t *) args.residual->data : nullptr;
        args.res_out_bf16 = out16 ? (uint16_t *) args.out_res->data : nullptr;
    }
}

// Multi-token combine+norm whose xn feeds the cuBLAS BF16 hc GEMMs (MMB does not take BF16 weights on every arch):
// write the BF16 copy of xn in the same pass, as the cached BF16 src1 of the GEMMs that follow (bf16_src1_cache).
// hc_f2bf32 rounds to nearest even like the conversion it replaces, and the F32 xn is still stored for dsv4_hc_pre.
void ggml_cuda_hc_combine_norm_set_bf16_src1(ggml_backend_cuda_context & ctx, ggml_cuda_hc_combine_norm_args & args) {
    const ggml_tensor * xn = args.out_xn;
    if (args.out_xn_bf16 != nullptr || !args.store_xn_f32 || !ggml_cuda_bf16_src1_cache_on() ||
            xn->view_src != nullptr || !ggml_is_contiguous(xn) || xn->ne[2]*xn->ne[3] < 64) {
        return;
    }
    args.out_xn_bf16 = (uint16_t *) ctx.bf16_src1_cache_put(xn, ggml_nelements(xn));
}

// qwen4exp hyper-connection combine + next grouped RMS norm fusion (repeat-anchored and A-chain).
// Returns the number of graph nodes to skip, 0 otherwise.  Opt-outs:
// GGML_CUDA_DISABLE_HC_FUSION=1 (all), GGML_CUDA_DISABLE_HC_COMB=1.
static bool hc_comb_enabled() {
    static const bool disabled = [] {
        const char * f = getenv("GGML_CUDA_DISABLE_HC_FUSION");
        if (f != nullptr && atoi(f) != 0) {
            return true;
        }
        const char * c = getenv("GGML_CUDA_DISABLE_HC_COMB");
        return c != nullptr && atoi(c) != 0;
    }();
    return !disabled;
}

int ggml_cuda_hc_combine_norm_try_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    if (!hc_comb_enabled()) {
        return 0;
    }
    static const bool hc_cn_dbg = getenv("LLAMA_HC_CN_DEBUG") != nullptr;
    ggml_tensor * node = cgraph->nodes[i];
    // halo-box merge (repeat-anchored combine+norm fusion): the model pre-expands block_out/w so
    // the combine run is scale -> sigmoid -> scale -> [reshape] -> repeat -> mul -> add, with the
    // hc-wide block_out REPEAT between the scale chain and the mul. A scale-anchored window would
    // have to include the [n_embd,1,T] reshape view, whose external (non-constant) view_src fails
    // the fuse check, so anchor the window AT the repeat: [repeat, mul, add, rms, (view,) mul(gamma)].
    // The SCALE/SIGMOID/SCALE nodes dispatch standalone (tiny) and their op params feed the kernel.
    // The kernel broadcasts the NARROW base (block_out_hc=false), so the ~80MB materialization never
    // dispatches. Reading the base at the window is safe without pinning it: the nodes between the
    // repeat and mul(gamma) are skipped when fused, so only add and mulg are written, and when
    // either of them reuses the freed base the kernel reads the base from a pool copy (below).
    if (node->op == GGML_OP_REPEAT && node->type == GGML_TYPE_F32 && i + 1 < cgraph->n_nodes) {
        const ggml_tensor * rep = node;
        const int64_t n_embd = rep->ne[0];
        const int64_t hc     = rep->ne[1];
        const int64_t n_tok  = rep->ne[2];
        const auto root_of = [](const ggml_tensor * t) -> const ggml_tensor * {
            while (t != nullptr && t->view_src != nullptr) {
                t = t->view_src;
            }
            return t;
        };
        if (n_embd > 0 && hc >= 2 && hc <= 8 && n_tok >= 1 && rep->ne[3] == 1) {
            const ggml_tensor * base = rep->src[0];
            const ggml_tensor * base_root = base;
            while (base_root != nullptr && base_root->view_src != nullptr) {
                base_root = base_root->view_src;
            }
            if (hc_cn_dbg) {
                static unsigned d = 0;
                if (d++ < 24) fprintf(stderr, "HC_CN[repeat] anchor n_embd=%lld hc=%lld n_tok=%lld root=%s out=%d\n",
                    (long long) n_embd, (long long) hc, (long long) n_tok,
                    base_root ? ggml_type_name(base_root->type) : "null",
                    base_root ? (int) (base_root->flags & GGML_TENSOR_FLAG_OUTPUT) : -1);
            }
            if (base_root != nullptr && base_root->type == GGML_TYPE_F32) {
                const int limit = std::min(cgraph->n_nodes, i + 24);
                // the consuming mul, then the chain: add(residual, mul) -> rms -> (view) -> mul(gamma)
                int k = i + 1;
                while (k < limit && !(cgraph->nodes[k]->op == GGML_OP_MUL &&
                        (cgraph->nodes[k]->src[0] == rep || cgraph->nodes[k]->src[1] == rep))) {
                    ++k;
                }
                if (k + 1 < limit && cgraph->nodes[k]->op == GGML_OP_MUL) {
                    const ggml_tensor * mul = cgraph->nodes[k];
                    int m = k + 1;
                    while (m < limit && ggml_cuda_is_view_or_noop(cgraph->nodes[m]) && !ggml_is_empty(cgraph->nodes[m])) {
                        ++m;
                    }
                    if (m < limit && cgraph->nodes[m]->op == GGML_OP_ADD) {
                        const ggml_tensor * add = cgraph->nodes[m];
                        int q = m + 1;
                        while (q < limit && ggml_cuda_is_view_or_noop(cgraph->nodes[q]) && !ggml_is_empty(cgraph->nodes[q])) {
                            ++q;
                        }
                        if (q < limit && cgraph->nodes[q]->op == GGML_OP_RMS_NORM) {
                            const ggml_tensor * rms = cgraph->nodes[q];
                            int g = q + 1;
                            while (g < limit && ggml_cuda_is_view_or_noop(cgraph->nodes[g]) && !ggml_is_empty(cgraph->nodes[g])) {
                                ++g;
                            }
                            if (g < limit && cgraph->nodes[g]->op == GGML_OP_MUL) {
                                const ggml_tensor * mulg = cgraph->nodes[g];
                                const int64_t hc_dim = n_embd * hc;
                                // the other mul operand roots at the inject scale chain
                                const ggml_tensor * w = nullptr;
                                for (int s2i = 0; s2i < 2; ++s2i) {
                                    const ggml_tensor * src = s2i == 0 ? mul->src[0] : mul->src[1];
                                    if (src != nullptr && src != rep && root_of(src) != root_of(rep)) {
                                        w = src;
                                    }
                                }
                                const ggml_tensor * scale2 = w != nullptr ? root_of(w) : nullptr;
                                // chain: scale1(inject) -> sigmoid -> scale2  (scale2 is the w root)
                                const ggml_tensor * sigm  = scale2 != nullptr ? scale2->src[0] : nullptr;
                                const ggml_tensor * scale1 = sigm   != nullptr ? sigm->src[0]   : nullptr;
                                const ggml_tensor * inject = scale1 != nullptr ? scale1->src[0] : nullptr;
                                const ggml_tensor * res = add->src[0] == mul ? add->src[1] : add->src[0];
                                const ggml_tensor * gamma = nullptr;
                                if (root_of(mulg->src[0]) == (const ggml_tensor *) rms) {
                                    gamma = mulg->src[1];
                                } else if (root_of(mulg->src[1]) == (const ggml_tensor *) rms) {
                                    gamma = mulg->src[0];
                                }
                                const bool w_ok = w != nullptr && scale2 != nullptr && scale1 != nullptr && sigm != nullptr &&
                                    scale2->op == GGML_OP_SCALE && scale1->op == GGML_OP_SCALE &&
                                    sigm->op == GGML_OP_UNARY && ggml_get_unary_op(sigm) == GGML_UNARY_OP_SIGMOID &&
                                    sigm->src[0] == scale1 && scale1->src[0] == inject &&
                                    w->type == GGML_TYPE_F32 && w->ne[0] == 1 && w->ne[1] == hc &&
                                    ggml_nelements(w) == hc * n_tok && ggml_is_contiguous(w);
                                const bool b_ok = base_root->ne[0] == n_embd && ggml_is_contiguous(base_root) &&
                                    ggml_nelements(base_root) == n_embd * n_tok;
                                const bool ok_a = add->ne[3] == 1 && n_tok <= 65535 &&
                                    inject != nullptr && inject->type == GGML_TYPE_F32 &&
                                    ggml_is_contiguous(inject) && ggml_nelements(inject) == hc * n_tok;
                                const bool ok_b = (add->src[0] == mul || add->src[1] == mul) && res != nullptr && res != mul &&
                                    root_of((ggml_tensor *) rms->src[0]) == (const ggml_tensor *) add && rms->src[0] != nullptr;
                                const bool ok_c = mul->type == GGML_TYPE_F32 && add->type == GGML_TYPE_F32 && res->type == GGML_TYPE_F32 &&
                                    rms->type == GGML_TYPE_F32 && mulg->type == GGML_TYPE_F32 && gamma != nullptr && gamma->type == GGML_TYPE_F32;
                                const bool ok_d = ggml_are_same_shape(mul, add) && ggml_are_same_shape(res, add) && ggml_are_same_shape(rms, add);
                                const bool ok_e = ggml_is_contiguous(res) && ggml_is_contiguous(add) && ggml_is_contiguous(rms) &&
                                    ggml_is_contiguous(mulg) && ggml_is_contiguous(gamma);
                                const bool shape3 = mulg->ne[0] == n_embd && mulg->ne[1] == hc && mulg->ne[2] == n_tok && mulg->ne[3] == 1 &&
                                    gamma->ne[0] == n_embd && gamma->ne[1] == hc;
                                const bool shape2 = mulg->ne[0] == hc_dim && mulg->ne[1] == n_tok && mulg->ne[2] == 1 && mulg->ne[3] == 1 &&
                                    gamma->ne[0] == hc_dim;
                                const bool ok_f = (shape3 || shape2) && ggml_nelements(gamma) == hc_dim &&
                                    ggml_nelements(mulg) == hc_dim * n_tok && w_ok && b_ok;
                                const bool ok = ok_a && ok_b && ok_c && ok_d && ok_e && ok_f;
                                if (!ok && hc_cn_dbg) {
                                    static unsigned dbg = 0;
                                    if (dbg++ < 40) fprintf(stderr, "HC_CN[repeat] ok=%d abcdef=%d%d%d%d%d%d n_embd=%lld hc=%lld n_tok=%lld mulg=[%lld,%lld,%lld,%lld] gamma=[%lld,%lld] out=%d\n",
                                        (int) ok, ok_a, ok_b, ok_c, ok_d, ok_e, ok_f,
                                        (long long) n_embd, (long long) hc, (long long) n_tok,
                                        (long long) mulg->ne[0], (long long) mulg->ne[1], (long long) mulg->ne[2], (long long) mulg->ne[3],
                                        (long long) gamma->ne[0], (long long) gamma->ne[1], (int) (base_root->flags & GGML_TENSOR_FLAG_OUTPUT));
                                }
                                if (ok) {
                                    ggml_cuda_hc_combine_norm_args args;
                                    args.inject       = inject;
                                    args.residual     = res;
                                    args.block_out    = base_root;
                                    args.block_out_hc = false;
                                    args.gamma        = gamma;
                                    args.out_res      = (ggml_tensor *) add;
                                    args.out_xn       = (ggml_tensor *) mulg;
                                    args.s1           = ggml_get_op_params_f32(scale1, 0);
                                    args.b1           = ggml_get_op_params_f32(scale1, 1);
                                    args.s2           = ggml_get_op_params_f32(scale2, 0);
                                    args.b2           = ggml_get_op_params_f32(scale2, 1);
                                    memcpy(&args.eps, rms->op_params, sizeof(float));
                                    auto overlap = [](const ggml_tensor * p, const ggml_tensor * q) {
                                        const uintptr_t p0 = (uintptr_t) p->data, p1 = p0 + ggml_backend_buft_get_alloc_size(p->buffer->buft, p);
                                        const uintptr_t q0 = (uintptr_t) q->data, q1 = q0 + ggml_backend_buft_get_alloc_size(q->buffer->buft, q);
                                        return p0 < q1 && q0 < p1;
                                    };
                                    const bool res_inplace = res->data == add->data && ggml_are_same_layout(res, add);
                                    const bool alias_ok_but_base = !overlap(add, inject) && (res_inplace || !overlap(add, res)) &&
                                        !overlap(mulg, inject) && !overlap(mulg, res) && !overlap(mulg, add) &&
                                        !overlap(add, gamma) && !overlap(mulg, gamma);
                                    bool alias_ok = alias_ok_but_base && !overlap(add, base_root) && !overlap(mulg, base_root);
                                    // block_out is not pinned: it dies at the repeat, so add or mulg may reuse its buffer.
                                    // Nothing in the window runs before the fused kernel, so the base is still intact here:
                                    // read it from a pool copy instead of falling back to the unfused nodes.
                                    ggml_cuda_pool_alloc<char> base_copy_buf(ctx.pool());
                                    ggml_tensor base_copy;
                                    if (!alias_ok && alias_ok_but_base && n_tok > 1 && ggml_is_contiguous(base_root)) {
                                        base_copy = *base_root;
                                        base_copy.data = base_copy_buf.alloc(ggml_nbytes(base_root));
                                        CUDA_CHECK(cudaMemcpyAsync(base_copy.data, base_root->data, ggml_nbytes(base_root),
                                            cudaMemcpyDeviceToDevice, ctx.stream()));
                                        args.block_out = &base_copy;
                                        alias_ok = true;
                                        if (hc_cn_dbg) {
                                            static unsigned d3 = 0;
                                            if (d3++ < 24) fprintf(stderr, "HC_CN[repeat] base copy n_tok=%lld\n", (long long) n_tok);
                                        }
                                    }
                                    if (!alias_ok && n_tok == 1 && hc <= 4) {
                                        args.single_block = true;
                                        alias_ok = true;
                                    }
                                    int          idxs[32];
                                    enum ggml_op ops[32];
                                    int          count = 0;
                                    for (int kk = i; kk <= g; ++kk) {
                                        idxs[count] = kk;
                                        ops[count]  = cgraph->nodes[kk]->op;
                                        ++count;
                                    }
                                    const int out_nodes[] = { m, g };
                                    const bool supp = ggml_cuda_hc_combine_norm_supported(args, ggml_cuda_info().devices[ctx.device].warp_size);
                                    const bool canf = ggml_can_fuse_subgraph_ext(cgraph, idxs, count, ops, out_nodes, 2);
                                    if (!(alias_ok && supp && canf) && hc_cn_dbg) {
                                        static unsigned d2 = 0;
                                        if (d2++ < 24) fprintf(stderr, "HC_CN[repeat] dispatch alias=%d supp=%d canf=%d n_tok=%lld hc=%lld %s\n",
                                            (int) alias_ok, (int) supp, (int) canf, (long long) n_tok, (long long) hc,
                                            ggml_can_fuse_subgraph_ext(cgraph, idxs, count, ops, out_nodes, 2) ? "" : "(canf)");
                                    }
                                    if (alias_ok && supp && canf) {
                                        ggml_cuda_hc_combine_norm_set_bf16(ctx, args);
                                        ggml_cuda_hc_combine_norm_set_bf16_src1(ctx, args);
                                        ggml_cuda_op_hc_combine_norm(ctx, args);
                                        return g - i;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // qwen4exp hyper-connection combine + next grouped rms norm (A-chain):
    //   scale(inject) -> sigmoid -> scale -> [view w] -> mul(block_out x w) -> add(residual)
    //     -> [view] -> rms_norm -> [view] -> mul(xn, gamma)
    // A's graph has no REPEAT node: block_out is already [n_embd, 1, T] and the mul broadcasts.
    // One kernel (block per stream+token) writes out_res = the add ([n_embd, hc, T]) and
    // out_xn = the gamma mul ([n_embd*hc, T]); the two share one row-major layout.
    if (node->op == GGML_OP_SCALE && node->type == GGML_TYPE_F32 && i + 3 < cgraph->n_nodes &&
        cgraph->nodes[i + 1]->op == GGML_OP_UNARY && ggml_get_unary_op(cgraph->nodes[i + 1]) == GGML_UNARY_OP_SIGMOID &&
        cgraph->nodes[i + 1]->src[0] == node &&
        cgraph->nodes[i + 2]->op == GGML_OP_SCALE && cgraph->nodes[i + 2]->src[0] == cgraph->nodes[i + 1]) {
        const ggml_tensor * scale1 = node;
        const ggml_tensor * scale2 = cgraph->nodes[i + 2];
        const ggml_tensor * inject = scale1->src[0];
        const auto root_of = [](const ggml_tensor * t) -> const ggml_tensor * {
            while (t != nullptr && t->view_src != nullptr) {
                t = t->view_src;
            }
            return t;
        };

        const int limit = std::min(cgraph->n_nodes, i + 24);
        int k = i + 3;
        while (k < limit && ggml_cuda_is_view_or_noop(cgraph->nodes[k]) && !ggml_is_empty(cgraph->nodes[k])) {
            ++k;
        }
        if (k + 1 < limit && cgraph->nodes[k]->op == GGML_OP_MUL) {
            const ggml_tensor * mul = cgraph->nodes[k];
            int m = k + 1;
            while (m < limit && ggml_cuda_is_view_or_noop(cgraph->nodes[m]) && !ggml_is_empty(cgraph->nodes[m])) {
                ++m;
            }
            if (m < limit && cgraph->nodes[m]->op == GGML_OP_ADD) {
                const ggml_tensor * add = cgraph->nodes[m];
                int q = m + 1;
                while (q < limit && ggml_cuda_is_view_or_noop(cgraph->nodes[q]) && !ggml_is_empty(cgraph->nodes[q])) {
                    ++q;
                }
                if (q < limit && cgraph->nodes[q]->op == GGML_OP_RMS_NORM) {
                    const ggml_tensor * rms = cgraph->nodes[q];
                    int g = q + 1;
                    while (g < limit && ggml_cuda_is_view_or_noop(cgraph->nodes[g]) && !ggml_is_empty(cgraph->nodes[g])) {
                        ++g;
                    }
                    if (g < limit && cgraph->nodes[g]->op == GGML_OP_MUL) {
                        const ggml_tensor * mulg = cgraph->nodes[g];
                        const int64_t n_embd = add->ne[0];
                        const int64_t hc     = add->ne[1];
                        const int64_t n_tok  = add->ne[2];
                        const int64_t hc_dim = n_embd * hc;

                        const ggml_tensor * w = nullptr;
                        const ggml_tensor * b = nullptr;
                        for (int s = 0; s < 2; ++s) {
                            const ggml_tensor * src = s == 0 ? mul->src[0] : mul->src[1];
                            if (src == nullptr) { continue; }
                            if (root_of(src) == (const ggml_tensor *) scale2) {
                                w = src;
                            } else {
                                b = src;
                            }
                        }
                        // block_out reaches the mul either as a REPEAT output ([n_embd, hc, T], copies of
                        // the broadcast base) or directly ([n_embd, 1, T]-style, ggml broadcast). Pass the
                        // mul's actual operand: it is consumed inside the fused window, so its buffer is
                        // live there. NEVER unwrap to the repeat INPUT - the standalone REPEAT may run
                        // before the window (its input then dies and the allocator reuses the buffer,
                        // e.g. for hc_inject), leaving the fused kernel reading the wrong data.
                        const ggml_tensor * bo = b;
                        // if bo is itself a REPEAT op's output, that op must have run already (producer
                        // index < i); a repeat inside the window cannot be materialized by this fusion
                        if (bo != nullptr && bo->op == GGML_OP_REPEAT) {
                            bool producer_before = false;
                            for (int j = 0; j < i && !producer_before; ++j) {
                                producer_before = cgraph->nodes[j] == bo;
                            }
                            if (!producer_before) {
                                bo = nullptr;
                            }
                        }
                        const ggml_tensor * res = add->src[0] == mul ? add->src[1] : add->src[0];
                        // the rms output may reach the gamma MUL through a reshape view: identify the
                        // operand whose root is rms and take the other side as gamma
                        const ggml_tensor * gamma = nullptr;
                        if (root_of(mulg->src[0]) == (const ggml_tensor *) rms) {
                            gamma = mulg->src[1];
                        } else if (root_of(mulg->src[1]) == (const ggml_tensor *) rms) {
                            gamma = mulg->src[0];
                        }

                        const bool w_ok = w != nullptr && w->type == GGML_TYPE_F32 && w->ne[0] == 1 && w->ne[1] == hc &&
                            ggml_nelements(w) == hc * n_tok && ggml_is_contiguous(w);
                        // the mul operand has either hc stacked copies (a REPEAT output, rows of n_embd
                        // per (c, t)) or a single copy broadcast over c (rows of n_embd per token)
                        const bool bo_hc = bo != nullptr && ggml_nelements(bo) == n_embd * hc * n_tok;
                        const bool b_ok = bo != nullptr && bo->type == GGML_TYPE_F32 && bo->ne[0] == n_embd &&
                            ggml_is_contiguous(bo) &&
                            (bo_hc || ggml_nelements(bo) == n_embd * n_tok);
                        const bool ok_a = add->ne[3] == 1 && n_tok >= 1 && n_tok <= 65535 &&
                            inject->type == GGML_TYPE_F32 && ggml_is_contiguous(inject) &&
                            ggml_nelements(inject) == hc * n_tok && ggml_nrows(inject) == n_tok &&
                            ggml_are_same_shape(scale1, inject) && ggml_are_same_shape(scale2, inject);
                        const bool ok_b = (add->src[0] == mul || add->src[1] == mul) && res != nullptr && res != mul &&
                            root_of((ggml_tensor *) rms->src[0]) == (const ggml_tensor *) add && rms->src[0] != nullptr;
                        const bool ok_c = mul->type == GGML_TYPE_F32 && add->type == GGML_TYPE_F32 && res->type == GGML_TYPE_F32 &&
                            rms->type == GGML_TYPE_F32 && mulg->type == GGML_TYPE_F32 && gamma != nullptr && gamma->type == GGML_TYPE_F32;
                        const bool ok_d = ggml_are_same_shape(mul, add) && ggml_are_same_shape(res, add) && ggml_are_same_shape(rms, add);
                        const bool ok_e = ggml_is_contiguous(res) && ggml_is_contiguous(add) && ggml_is_contiguous(rms) &&
                            ggml_is_contiguous(mulg) && ggml_is_contiguous(gamma);
                        const bool shape3 = mulg->ne[0] == n_embd && mulg->ne[1] == hc && mulg->ne[2] == n_tok && mulg->ne[3] == 1 &&
                            gamma->ne[0] == n_embd && gamma->ne[1] == hc;
                        const bool shape2 = mulg->ne[0] == hc_dim && mulg->ne[1] == n_tok && mulg->ne[2] == 1 && mulg->ne[3] == 1 &&
                            gamma->ne[0] == hc_dim;
                        const bool ok_f = (shape3 || shape2) && ggml_nelements(gamma) == hc_dim &&
                            ggml_nelements(mulg) == hc_dim * n_tok && w_ok && b_ok;
                        const bool ok = ok_a && ok_b && ok_c && ok_d && ok_e && ok_f;
                        if (!ok && hc_cn_dbg) {
                            static unsigned dbg = 0;
                            if (dbg++ < 40) fprintf(stderr, "HC_CN[chain] ok=%d abcdef=%d%d%d%d%d%d n_embd=%lld hc=%lld n_tok=%lld mulg=[%lld,%lld,%lld,%lld] gamma=[%lld,%lld]\n",
                                (int) ok, ok_a, ok_b, ok_c, ok_d, ok_e, ok_f,
                                (long long) n_embd, (long long) hc, (long long) n_tok,
                                (long long) mulg->ne[0], (long long) mulg->ne[1], (long long) mulg->ne[2], (long long) mulg->ne[3],
                                (long long) gamma->ne[0], (long long) gamma->ne[1]);
                        }
                        if (ok) {
                            ggml_cuda_hc_combine_norm_args args;
                            args.inject    = inject;
                            args.residual  = res;
                            args.block_out = bo;
                            args.block_out_hc = bo_hc;
                            args.gamma     = gamma;
                            args.out_res   = (ggml_tensor *) add;
                            args.out_xn    = (ggml_tensor *) mulg;
                            args.s1        = ggml_get_op_params_f32(scale1, 0);
                            args.b1        = ggml_get_op_params_f32(scale1, 1);
                            args.s2        = ggml_get_op_params_f32(scale2, 0);
                            args.b2        = ggml_get_op_params_f32(scale2, 1);
                            memcpy(&args.eps, rms->op_params, sizeof(float));

                            // every block reads block_out and inject and writes its own stream of both outputs:
                            // the outputs must not overlap those, and the residual may only alias out_res in place
                            auto overlap = [](const ggml_tensor * p, const ggml_tensor * q) {
                                const uintptr_t p0 = (uintptr_t) p->data, p1 = p0 + ggml_backend_buft_get_alloc_size(p->buffer->buft, p);
                                const uintptr_t q0 = (uintptr_t) q->data, q1 = q0 + ggml_backend_buft_get_alloc_size(q->buffer->buft, q);
                                return p0 < q1 && q0 < p1;
                            };
                            const bool res_inplace = res->data == add->data && ggml_are_same_layout(res, add);
                            // the kernel reads bo (block_out = the repeat input), inject, gamma and (via add)
                            // res; it writes add and mulg. All checks must use bo, NOT the repeat output.
                            bool alias_ok = !overlap(add, bo) && !overlap(add, inject) && (res_inplace || !overlap(add, res)) &&
                                !overlap(mulg, bo) && !overlap(mulg, inject) && !overlap(mulg, res) && !overlap(mulg, add) &&
                                !overlap(add, gamma) && !overlap(mulg, gamma);
                            // outputs overlapping the inputs (typically xn reusing the block_out buffer, which dies
                            // at the combine): single-block variant that reads everything before it writes
                            if (!alias_ok && n_tok == 1 && hc <= 4) {
                                args.single_block = true;
                                alias_ok = true;
                            }

                            int          idxs[32];
                            enum ggml_op ops[32];
                            int          count = 0;
                            for (int kk = i; kk <= g; ++kk) {
                                idxs[count] = kk;
                                ops[count]  = cgraph->nodes[kk]->op;
                                ++count;
                            }
                            const int out_nodes[] = { m, g };
                            const bool supp = ggml_cuda_hc_combine_norm_supported(args, ggml_cuda_info().devices[ctx.device].warp_size);
                            const bool canf = ggml_can_fuse_subgraph_ext(cgraph, idxs, count, ops, out_nodes, 2);
                            if (alias_ok && supp && canf) {
                                ggml_cuda_hc_combine_norm_set_bf16(ctx, args);
                                ggml_cuda_hc_combine_norm_set_bf16_src1(ctx, args);
                                ggml_cuda_op_hc_combine_norm(ctx, args);
                                return g - i;
                            }
                        }
                    }
                }
            }
        }
    }

    return 0;
}

// HC mix enable predicate, shared by the mix and gate-mix fusion sites.
static bool hc_mix_enabled() {
    static const bool disabled = [] {
        const char * f = getenv("GGML_CUDA_DISABLE_HC_FUSION");
        if (f != nullptr && atoi(f) != 0) {
            return true;
        }
        const char * m = getenv("GGML_CUDA_DISABLE_HC_MIX");
        return m != nullptr && atoi(m) != 0;
    }();
    return !disabled;
}

// qwen4exp hyper-connection gate GEMM + sigmoid + stream mix: when the HC gate MUL_MAT's only
// consumer is the sigmoid that opens an hc_mix window, fold the GEMM into the mix kernel.  Returns
// the number of nodes to skip, 0 otherwise.  A/B: LLAMA_HC_GATEMIX=0.
int ggml_cuda_hc_gate_mix_try_fuse(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    ggml_tensor * node = cgraph->nodes[i];
    if (!hc_mix_enabled() || !ggml_cuda_mmb_gatemix() || !(GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) ||
            node->op != GGML_OP_MUL_MAT || i + 1 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * w  = node->src[0];
    const ggml_tensor * lo = node->src[1];
    if (w != nullptr && lo != nullptr && ggml_is_quantized(w->type) && ggml_node_has_n_uses(cgraph, i, 1)) {
        // (a) unfused mix chain: sigmoid(gate) -> mul -> collapse -> scale
        ggml_cuda_hc_mix_args ma;
        const int count = ggml_cuda_hc_mix_closed(cgraph, i + 1, ma);
        if (count > 0 && ma.gate == node &&
                ggml_cuda_hc_gate_mix(ctx, w, lo, ma.xn, ma.dst, ma.hc, ma.scale, ma.bias)) {
            return count;
        }
        // (b) the explicit ggml_dsv4_hc_pre op (the qwen4exp default): MUL_MAT -> RESHAPE(view)
        //     -> DSV4_HC_PRE.  Lower it to the same kernel so the GEMM is folded either way.
        if (i + 2 < cgraph->n_nodes) {
            const ggml_tensor * gate3 = cgraph->nodes[i + 1];
            const ggml_tensor * pre   = cgraph->nodes[i + 2];
            if (gate3->op == GGML_OP_RESHAPE && gate3->view_src == node &&
                    ggml_node_get_use_count(cgraph, i + 1) == 1 &&
                    pre->op == GGML_OP_DSV4_HC_PRE && pre->type == GGML_TYPE_F32 &&
                    pre->src[1] == gate3 && pre->src[0] != nullptr &&
                    ggml_get_op_params_i32(pre, 1) != 0) {
                const int hc = (int) pre->src[0]->ne[1];
                const int64_t M = (int64_t) hc * pre->src[0]->ne[0];
                const int64_t T = pre->src[0]->ne[2];
                const ggml_tensor * xn = pre->src[0];
                while (xn->view_src != nullptr && (xn->ne[0] != M || ggml_nrows(xn) != T)) {
                    xn = xn->view_src;
                }
                if (xn->ne[0] != M || ggml_nrows(xn) != T) {
                    const ggml_tensor * a = node->src[1];
                    while (a != nullptr && a->op != GGML_OP_MUL_MAT) {
                        a = a->src[0];
                    }
                    if (a != nullptr && a->src[1] != nullptr && a->src[1]->ne[0] == M && ggml_nrows(a->src[1]) == T) {
                        xn = a->src[1];
                    }
                }
                if (ggml_cuda_hc_gate_mix(ctx, w, lo, xn, (ggml_tensor *) pre, hc,
                        ggml_get_op_params_f32(pre, 0), 0.0f)) {
                    return 2;
                }
            }
        }
    }
    return 0;
}
