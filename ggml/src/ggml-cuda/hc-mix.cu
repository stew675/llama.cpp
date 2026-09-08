// Fused hyper-connection mixer tail for qwen4exp (decode/verify band).
//
// Replaces the unfused decode chain (SCALE, SILU, MUL_MAT up, SIGMOID, MUL,
// collapse ADD/SCALE) with one op dispatch. The numerics replicate the unfused
// chain bit for bit:
//   lo_raw = w_down^T xn                       (mmvq Q8_0 dot, M = 1)
//   v      = silu(lo_raw / hc)                 (SCALE then SILU)
//   gate   = sigmoid(w_up^T v)                 (mmvq Q8_0 dot, M = 1)
//   mixed  = (1/hc) * sum_c xn * gate          (collapse of the hc streams)
// The xn -> Q8_1 quantization mirrors quantize_row_q8_1_cuda, and the per-row
// dots replicate mul_mat_vec_q<Q8_0, 1> (block (32, 8), rpb = 1) so the sums
// are bit-identical to the unfused mmvq path.
//
// Token dimension: the ops serve the whole decode/verify band, nt <=
// HC_FUSED_MAX_TOKENS, by mapping the token index onto blockIdx.y (the graph
// gates in src/models/qwen4exp.cpp use the same bound). Every token therefore
// runs exactly the per-token kernel sequence a single-token decode runs, which
// is what makes the band width-invariant; nt == 1 selects t == 0 only and its
// arithmetic is unchanged.

#include "hc-mix.cuh"

#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

// Upper bound of the decode/verify band these fused ops serve (see the header
// comment).  The graphs in src/models/qwen4exp.cpp gate the fused path on the
// same bound; a verify batch is --spec-draft-n-max + 1 tokens and the
// greedy-purity guarantee is defined over that band (GREEDY-PURITY.md).
static constexpr int64_t HC_FUSED_MAX_TOKENS = 8;


// Grouped RMSNorm over the hc streams plus the gamma scale (w_norm). One
// block of 1024 threads per stream row; the reduction mirrors rms_norm_f32
// (blockDim 1024, warp shuffles then one final warp pass over the 32 warp
// sums) and the gamma multiply keeps the rms output's separate rounding, so
// xn is bit-identical to the unfused RMS + MUL chain.
// grouped RMSNorm + gamma (the xn stream) with the Q8_1 quantize of xn merged
// in: one block per stream computes xn then quantizes its own q8_1 groups (the
// quantize pattern mirrors quantize_q8_1: one warp per 32-value group, amax
// reduction, d = amax/127, roundf(x/d), sum), so the op runs one fewer kernel.
// n_embd must be divisible by 32.
// Token dimension: `t` (blockIdx.y) selects the token; every per-token pointer is offset by
// that token's row in the packed scratch buffers (strides from the launcher).  At nt == 1 only
// t == 0 exists, so the arithmetic is unchanged.
static __global__ void hc_mix_rms_gamma_quant(
        const float * x, const float * w_norm, float * xn, block_q8_1 * y,
        const int n_embd, const float eps,
        const int64_t x_stride_t, const int64_t xn_stride_t, const int64_t y_stride_t) {
    const int c   = blockIdx.x;
    const int t   = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const float * xs = x + (int64_t) t*x_stride_t + (int64_t) c*n_embd;

    float tmp = 0.0f;
    for (int col = tid; col < n_embd; col += blockDim.x) {
        const float xi = xs[col];
        tmp += xi*xi;
    }
    __shared__ float s_sum[32];
    tmp = block_reduce<block_reduce_method::SUM>(tmp, s_sum);

    const float scale = rsqrtf(tmp / (float) n_embd + eps);
    float * xo = xn + (int64_t) t*xn_stride_t + (int64_t) c*n_embd;
    for (int col = tid; col < n_embd; col += blockDim.x) {
        const float r = scale * xs[col];
        xo[col] = r * w_norm[(int64_t) c*n_embd + col];
    }

    // quantize this stream's q8_1 groups (n_embd/32 groups, one warp each)
    __syncthreads();
    const int n_groups = n_embd / 32;
    for (int g = tid / 32; g < n_groups; g += blockDim.x / 32) {
        const float xv = xo[g*32 + lane];
        float amax = fabsf(xv);
        float sum  = xv;
        amax = warp_reduce_max<32>(amax);
        sum  = warp_reduce_sum<32>(sum);
        const float  d = amax / 127.0f;
        const int8_t q = amax == 0.0f ? 0 : (int8_t) roundf(xv / d);
        y[(int64_t) t*y_stride_t + (int64_t) c*n_groups + g].qs[lane] = q;
        if (lane == 0) {
            y[(int64_t) t*y_stride_t + (int64_t) c*n_groups + g].ds = make_half2(d, sum);
        }
    }
}

// One Q8_0 matrix-vector product against the pre-quantized Q8_1 input, rpb =
// 1 (one row per block, K fills the thread groups). The grid covers the lo
// rows (w_down, K = hc_dim) and then the inject rows (w_inject, K = hc_dim);
// the accumulation is the mul_mat_vec_q rpb=1 clone, bit-identical to the
// unfused mmvq path.  `t` (blockIdx.y) is the token; the weights are shared, the
// per-token q8_1 input and the output rows are offset by that token's scratch rows
// (`lo` is packed at hc_lr, `inject` lives in the dst tail so it carries the dst row
// stride).
static __global__ void hc_mix_down_dots(
        const block_q8_0 * w_down, float * lo, const int nrows_down,
        const block_q8_0 * w_inject, float * inject, const int nrows_inject,
        const block_q8_1 * y, const int blocks_per_row,
        const int64_t y_stride_t, const int64_t lo_stride_t, const int64_t inject_stride_t) {
    const int row = blockIdx.x;
    const int t   = blockIdx.y;
    const bool is_inject = row >= nrows_down;
    const int r = is_inject ? row - nrows_down : row;
    const block_q8_0 * w = is_inject ? w_inject : w_down;
    float * dst = is_inject ? inject + (int64_t) t*inject_stride_t : lo + (int64_t) t*lo_stride_t;
    const block_q8_1 * y_t = y + (int64_t) t*y_stride_t;

    constexpr int qi  = QI8_0;             // 8
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    const int tid      = 32*threadIdx.y + threadIdx.x;
    const int n_groups = 8*32 / (qi/vdr);
    const int n_items  = blocks_per_row;

    float acc = 0.0f;
    const int kqs = vdr * (tid % (qi/vdr));
    for (int it = tid / (qi/vdr); it < n_items; it += n_groups) {
        acc += vec_dot_q8_0_q8_1(w + (int64_t) r*blocks_per_row, &y_t[it], it, kqs);
    }

    acc = warp_reduce_sum<32>(acc);
    __shared__ float tmp_shared[7];
    if (threadIdx.y > 0) {
        tmp_shared[threadIdx.y-1] = acc;
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }
#pragma unroll
    for (int l = 0; l < 7; ++l) {
        acc += tmp_shared[l];
    }
    if (threadIdx.x == 0) {
        dst[r] = acc;
    }
}

// silu(lo/hc) and Q8_1 quantize of the low-rank vector, then the Q8_0 matrix-
// vector product w^T v (the "up" dot). The quantize kernel is merged into the
// dot kernel: every block quantizes all of v in its prologue (the writes are
// identical across blocks, so the global y buffer stays valid) and then dots,
// so the op runs one fewer kernel. The per-kblock reduction mirrors
// quantize_q8_1 (one warp over 32 consecutive values) and silu mirrors the
// standalone op, so the values are bit-identical to the unfused chain.
// `t` (blockIdx.y) is the token: the weights are shared, `lo`/`y`/`dst` are the
// packed per-token scratch rows.
template <int nwarps, int RPB>
static __global__ void hc_mix_up_silu_dot(
        const float * lo, const block_q8_0 * w, block_q8_1 * y,
        float * dst, const int nrows, const int blocks_per_row,
        const float inv_hc,
        const int64_t lo_stride_t, const int64_t y_stride_t, const int64_t dst_stride_t) {
    const int lane = threadIdx.x;
    const int t    = blockIdx.y;
    const float * lo_t = lo + (int64_t) t*lo_stride_t;
    block_q8_1 * y_t = y + (int64_t) t*y_stride_t;
    float * dst_t = dst + (int64_t) t*dst_stride_t;
    // prologue: v = silu(lo/hc) quantized to Q8_1, warps split the kblocks
    for (int kb = threadIdx.y; kb < blocks_per_row; kb += nwarps) {
        const int col = kb*32 + lane;
        const float x = ggml_cuda_op_silu_single(lo_t[col] * inv_hc);
        float amax = fabsf(x);
        float sum  = x;
        amax = warp_reduce_max<32>(amax);
        sum  = warp_reduce_sum<32>(sum);
        const float  d = amax / 127.0f;
        const int8_t q = amax == 0.0f ? 0 : (int8_t) roundf(x / d);
        y_t[kb].qs[lane] = q;
        if (lane == 0) {
            y_t[kb].ds = make_half2(d, sum);
        }
    }
    __syncthreads();

    // the dot body below is hc_mix_row_dot<8, RPB> unchanged
    const int row0 = RPB*blockIdx.x;
    constexpr int qi  = QI8_0;             // 8
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    const int tid      = 32*threadIdx.y + threadIdx.x;
    const int n_groups = nwarps*32 / (qi/vdr);
    const int n_items  = RPB * blocks_per_row;

    float tmp[RPB] = {0.0f};
    const int kqs = vdr * (tid % (qi/vdr));
    for (int it = tid / (qi/vdr); it < n_items; it += n_groups) {
        const int i   = it / blocks_per_row;
        const int kbx = it % blocks_per_row;
        if (row0 + i < nrows) {
            tmp[i] += vec_dot_q8_0_q8_1(w + (int64_t) (row0 + i) * blocks_per_row, &y_t[kbx], kbx, kqs);
        }
    }

    __shared__ float tmp_shared[nwarps > 1 ? nwarps-1 : 1][RPB];
    for (int i = 0; i < RPB; ++i) {
        tmp[i] = warp_reduce_sum<32>(tmp[i]);
        if (threadIdx.y > 0) {
            tmp_shared[threadIdx.y-1][i] = tmp[i];
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }
    for (int i = 0; i < RPB; ++i) {
#pragma unroll
        for (int l = 0; l < nwarps-1; ++l) {
            tmp[i] += tmp_shared[l][i];
        }
        if (threadIdx.x == 0 && row0 + i < nrows) {
            dst_t[row0 + i] = tmp[i];
        }
    }
}

// Collapse the gated streams to their mean: mixed[j] = (1/hc) * sum_c xn*c*gate.
// gate holds the raw up projection; sigmoid is applied inline (same formula as
// the standalone op). The products are stored to an array before summing so the
// compiler cannot contract them into FMAs: the reference rounds each xn*gate
// product (a separate MUL op) and then adds the rounded values. One thread per
// output element in (256)-thread blocks so the stream reads coalesce; the adds
// follow the graph order (left-to-right ADD chain), then SCALE.
// collapse + F32 inject merged into one dispatch: grid = collapse blocks
// (n_embd/256) + hc inject blocks. Each path is unchanged from its separate
// kernel (the collapse products are stored before summing - no FMA - and the
// inject replicates the mmvf float2 accumulation), so the op runs one fewer
// kernel per call.  `t` (blockIdx.y) is the token: `xn`/`gate_raw`/`dst` and the dst
// inject tail are the packed per-token rows, while w_inject is shared.
static __global__ void hc_mix_collapse_inject(
        const float * xn, const float * gate_raw, float * dst,
        const int n_embd, const int hc, const float inv_hc,
        const float * w_inject, float * inject, const int ncols2,
        const int n_collapse_blocks,
        const int64_t xn_stride_t, const int64_t gate_stride_t, const int64_t dst_stride_t) {
    const int t = blockIdx.y;
    const float * xn_t = xn + (int64_t) t*xn_stride_t;
    const float * gate_t = gate_raw + (int64_t) t*gate_stride_t;
    float * dst_t = dst + (int64_t) t*dst_stride_t;
    float * inject_t = inject + (int64_t) t*dst_stride_t;
    if (blockIdx.x >= n_collapse_blocks) {
        // inject rows: one 256-thread block per inject row (mmvf numerics)
        const int r   = blockIdx.x - n_collapse_blocks;
        const int tid = threadIdx.x;
        const float2 * x2 = (const float2 *) xn_t;
        const float2 * w2 = (const float2 *) (w_inject + (int64_t) r*2*ncols2);
        float sumf = 0.0f;
        for (int col2 = tid; col2 < ncols2; col2 += blockDim.x) {
            const float2 tmpx = x2[col2];
            const float2 tmpw = w2[col2];
            sumf += tmpx.x*tmpw.x;
            sumf += tmpx.y*tmpw.y;
        }
        sumf = warp_reduce_sum<32>(sumf);
        __shared__ float buf[32];
        if (tid < 32) {
            buf[tid] = 0.0f;
        }
        __syncthreads();
        buf[tid/32] = sumf;
        __syncthreads();
        if (tid < 32) {
            sumf = buf[tid];
            sumf = warp_reduce_sum<32>(sumf);
            if (tid == 0) {
                inject_t[r] = sumf;
            }
        }
        return;
    }
    const int j = blockIdx.x*blockDim.x + threadIdx.x;
    if (j >= n_embd) {
        return;
    }
    float pp[8];
    for (int c = 0; c < hc; ++c) {
        const float g = 1.0f / (1.0f + expf(-gate_t[c*n_embd + j]));
        pp[c] = xn_t[c*n_embd + j] * g;
    }
    float sum = pp[0];
    for (int c = 1; c < hc; ++c) {
        sum = sum + pp[c];
    }
    dst_t[j] = sum * inv_hc;
}

void ggml_cuda_op_hc_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x         = dst->src[0];
    const ggml_tensor * w_norm    = dst->src[1];
    const ggml_tensor * w_down    = dst->src[2];
    const ggml_tensor * w_up      = dst->src[3];
    const ggml_tensor * w_inject  = dst->src[4];

    GGML_ASSERT(x->type        == GGML_TYPE_F32);
    GGML_ASSERT(w_norm->type   == GGML_TYPE_F32);
    GGML_ASSERT(w_down->type   == GGML_TYPE_Q8_0);
    GGML_ASSERT(w_up->type     == GGML_TYPE_Q8_0);
    GGML_ASSERT(w_inject == nullptr || w_inject->type == GGML_TYPE_F32 || w_inject->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(dst->type      == GGML_TYPE_F32);

    const int   hc  = ggml_get_op_params_i32(dst, 0);
    const float eps = ggml_get_op_params_f32(dst, 1);
    GGML_ASSERT(hc > 0 && hc <= 8);

    const int64_t n_embd   = x->ne[0];
    const int64_t hc_dim   = n_embd * hc;
    const int64_t n_tokens = x->ne[2];
    const int64_t hc_lr    = w_down->ne[1];

    GGML_ASSERT(n_tokens >= 1 && n_tokens <= HC_FUSED_MAX_TOKENS);  // decode/verify band
    GGML_ASSERT(hc_dim % 32 == 0 && hc_lr % 32 == 0);
    GGML_ASSERT(x->ne[1] == hc);
    GGML_ASSERT(x->nb[2] == hc_dim*sizeof(float));  // streams contiguous
    GGML_ASSERT(ggml_nelements(w_norm) == hc_dim);
    GGML_ASSERT(w_up->ne[0] == hc_lr && w_up->ne[1] == hc_dim);
    GGML_ASSERT(w_inject == nullptr || (w_inject->ne[0] == hc_dim && w_inject->ne[1] == hc));
    GGML_ASSERT(dst->ne[0] == n_embd + (w_inject ? hc : 0));
    // dst is [out_n, n_tokens]; its row is the per-token stride for every packed
    // scratch row this op writes (mixed head + F32 inject tail)
    const int64_t out_n = n_embd + (w_inject ? hc : 0);
    GGML_ASSERT(dst->ne[1] == n_tokens);
    GGML_ASSERT(dst->nb[1] == out_n*sizeof(float));

    // F32 inject: the mmvf float2 tail of the collapse launch; Q8_0 inject:
    // extra rows on the down-dots grid (mmvq, bit-identical to the unfused path)
    const bool inject_f32  = w_inject == nullptr || w_inject->type == GGML_TYPE_F32;
    const int  n_inject_q8 = w_inject && !inject_f32 ? (int) hc : 0;

    const float * x_d   = (const float *) x->data;
    const float * wn_d  = (const float *) w_norm->data;
    float * dst_d       = (float *) dst->data;
    float * inject      = w_inject ? dst_d + n_embd : nullptr;  // the inject tail

    // per-token element strides: x streams are contiguous within a token (asserted
    // above), the packed scratch rows are contiguous within their own arrays, and
    // both the mixed head and the inject tail live in a dst row of `out_n` floats
    const int64_t x_stride_t   = hc_dim;
    const int64_t dst_stride_t = out_n;

    cudaStream_t stream = ctx.stream();

    const int blocks_down = hc_dim / 32; // xn Q8_1 blocks for the down dots
    const int blocks_up   = hc_lr  / 32; // v  Q8_1 blocks for the up dots

    ggml_cuda_pool & pool = ctx.pool();

    ggml_cuda_pool_alloc<float>      xn_alloc(pool, hc_dim*n_tokens);
    ggml_cuda_pool_alloc<block_q8_1> y_xn_alloc(pool, blocks_down*n_tokens);
    ggml_cuda_pool_alloc<float>      lo_alloc(pool, hc_lr*n_tokens);
    ggml_cuda_pool_alloc<block_q8_1> y_v_alloc(pool, blocks_up*n_tokens);
    ggml_cuda_pool_alloc<float>      gate_alloc(pool, hc_dim*n_tokens);

    float      * xn   = xn_alloc.get();
    block_q8_1 * y_xn = y_xn_alloc.get();
    float      * lo   = lo_alloc.get();
    block_q8_1 * y_v  = y_v_alloc.get();
    float      * gate = gate_alloc.get();

    // rows-per-block as the mmvq dispatch chooses: 1 when the K-blocks fill the
    // thread groups, or the short-K override (RDNA2+) that packs RPB rows per
    // block so the item loop is bit-identical to the unfused path.
    constexpr int warp_size = 32;
    constexpr int qi  = QI8_0;
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    const auto calc_rpb = [&](int blocks_per_row) {
        const int n_groups = 8 * warp_size * vdr / qi;   // nwarps = 8
        int rpb = 1;
        if (blocks_per_row > 0 && blocks_per_row < n_groups) {
            int fill = (n_groups + blocks_per_row - 1) / blocks_per_row;
            int a = blocks_per_row, b = n_groups;
            while (b) { int t = a % b; a = b; b = t; }
            rpb = std::max(fill, n_groups / a);
            int pp = 1;
            while (pp < rpb) { pp <<= 1; }
            rpb = std::min(pp, 16);
        }
        return rpb;
    };
    const int rpb_down = calc_rpb(blocks_down);
    const int rpb_up   = calc_rpb(blocks_up);

    // xn = rms(x) * w_norm with the xn Q8_1 quantize merged in (grid hc x 1024 x nt)
    {
        const dim3 block_nums(hc, n_tokens);
        const dim3 block_dims(1024);
        const ggml_cuda_kernel_launch_params launch_params = {block_nums, block_dims, 0, stream};
        ggml_cuda_kernel_launch(hc_mix_rms_gamma_quant, launch_params,
                x_d, wn_d, xn, y_xn, (int) n_embd, eps,
                x_stride_t, hc_dim, blocks_down);
    }

    // lo_raw = w_down^T xn: 320 rows x 10240 dots (rpb is always 1 here);
    // a Q8_0 inject appends its hc rows to the same grid
    if (rpb_down != 1) {
        GGML_ABORT("hc_mix: unexpected down rpb %d\n", rpb_down);
    }
    {
        const dim3 block_nums(hc_lr + n_inject_q8, n_tokens);
        const dim3 block_dims(32, 8);
        const ggml_cuda_kernel_launch_params launch_params = {block_nums, block_dims, 0, stream};
        ggml_cuda_kernel_launch(hc_mix_down_dots, launch_params,
                (const block_q8_0 *) w_down->data, lo, hc_lr,
                n_inject_q8 ? (const block_q8_0 *) w_inject->data : nullptr, inject, n_inject_q8,
                y_xn, blocks_down,
                blocks_down, hc_lr, dst_stride_t);
    }

    // gate_raw = w_up^T v: 10240 rows x 320 dots (short K -> rpb override);
    // the v = silu(lo/hc) Q8_1 quantize is the kernel prologue
    {
        const dim3 block_nums((hc_dim + rpb_up - 1) / rpb_up, n_tokens);
        const dim3 block_dims(32, 8);
        const ggml_cuda_kernel_launch_params launch_params = {block_nums, block_dims, 0, stream};
        switch (rpb_up) {
            case 1:  ggml_cuda_kernel_launch(hc_mix_up_silu_dot<8, 1>,  launch_params, lo, (const block_q8_0 *) w_up->data, y_v, gate, hc_dim, blocks_up, 1.0f / (float) hc, hc_lr, blocks_up, hc_dim); break;
            case 2:  ggml_cuda_kernel_launch(hc_mix_up_silu_dot<8, 2>,  launch_params, lo, (const block_q8_0 *) w_up->data, y_v, gate, hc_dim, blocks_up, 1.0f / (float) hc, hc_lr, blocks_up, hc_dim); break;
            case 4:  ggml_cuda_kernel_launch(hc_mix_up_silu_dot<8, 4>,  launch_params, lo, (const block_q8_0 *) w_up->data, y_v, gate, hc_dim, blocks_up, 1.0f / (float) hc, hc_lr, blocks_up, hc_dim); break;
            case 8:  ggml_cuda_kernel_launch(hc_mix_up_silu_dot<8, 8>,  launch_params, lo, (const block_q8_0 *) w_up->data, y_v, gate, hc_dim, blocks_up, 1.0f / (float) hc, hc_lr, blocks_up, hc_dim); break;
            case 16: ggml_cuda_kernel_launch(hc_mix_up_silu_dot<8, 16>, launch_params, lo, (const block_q8_0 *) w_up->data, y_v, gate, hc_dim, blocks_up, 1.0f / (float) hc, hc_lr, blocks_up, hc_dim); break;
            default: GGML_ABORT("hc_mix: unexpected up rpb %d\n", rpb_up); break;
        }
    }

    // mixed at the dst head + the F32 inject at the dst tail in one dispatch:
    // the collapse blocks (n_embd/256) and the hc inject rows share the grid.
    // A Q8_0 inject is already in the tail from the down-dots launch above
    {
        const int n_collapse_blocks = (int) ((n_embd + 255) / 256);
        const int n_inject_blocks   = w_inject && inject_f32 ? hc : 0;
        const dim3 block_nums(n_collapse_blocks + n_inject_blocks, n_tokens);
        const dim3 block_dims(256);
        const ggml_cuda_kernel_launch_params launch_params = {block_nums, block_dims, 0, stream};
        ggml_cuda_kernel_launch(hc_mix_collapse_inject, launch_params,
                xn, gate, dst_d, n_embd, hc, 1.0f / (float) hc,
                w_inject && inject_f32 ? (const float *) w_inject->data : nullptr, inject,
                (int) (hc_dim / 2), n_collapse_blocks,
                hc_dim, hc_dim, dst_stride_t);
    }
}

// Fused hyper-connection residual combine for qwen4exp (decode/verify band).
// out[r, c, t] = residual[r, c, t] + block_out[r, t] * w[c, t], with
// w[c, t] = 2 * sigmoid(inject[c, t] / hc) (the SCALE+SIGMOID+SCALE chain of
// build_hc_combine; the 1/hc and 2.0 scalars are exact in f32 so only the
// sigmoid rounds). The product block_out[r]*w[c] and the residual add keep
// their own roundings because the reference runs separate MUL and ADD ops:
// the products go to an array first so the compiler cannot contract them
// into FMAs. One thread per row handles the hc columns, mirroring the
// collapse kernel. The token index is blockIdx.y and w is recomputed per token
// in registers (identical expression, so the nt == 1 result is unchanged);
// `inject` is often a view into the mix output, so it is read with its own
// per-token stride.
// Token dimension: the op serves the whole decode/verify band, nt <=
// HC_FUSED_MAX_TOKENS (the graph gates in src/models/qwen4exp.cpp use the same
// bound), which is what makes the band width-invariant.
static __global__ void hc_combine_kernel(
        const float * residual, const float * block_out, const float * inject,
        float * dst, const int n_embd, const int hc,
        const int64_t res_stride_t, const int64_t bo_stride_t,
        const int64_t inject_stride_t, const int64_t dst_stride_t) {
    const float inv_hc = 1.0f / (float) hc;
    const int t = blockIdx.y;

    // w[c] = 2*sigmoid(inject[c]/hc); per-thread registers rather than shared
    // memory so the early return below stays barrier-free
    const float * inj = inject + (int64_t) t*inject_stride_t;
    float w[8];
    for (int c = 0; c < hc; ++c) {
        const float s = inj[c] * inv_hc;
        w[c] = (1.0f / (1.0f + expf(-s))) * 2.0f;
    }

    const int r = blockIdx.x*blockDim.x + threadIdx.x;
    if (r >= n_embd) {
        return;
    }
    const float * res = residual + (int64_t) t*res_stride_t + r;
    const float   bo  = block_out[(int64_t) t*bo_stride_t + r];
    float       * dt  = dst + (int64_t) t*dst_stride_t + r;
    float pp[8];
    for (int c = 0; c < hc; ++c) {
        pp[c] = bo * w[c];
    }
    for (int c = 0; c < hc; ++c) {
        dt[c*n_embd] = res[c*n_embd] + pp[c];
    }
}

void ggml_cuda_op_hc_combine(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * residual  = dst->src[0];
    const ggml_tensor * block_out = dst->src[1];
    const ggml_tensor * inject    = dst->src[2];

    GGML_ASSERT(residual->type  == GGML_TYPE_F32);
    GGML_ASSERT(block_out->type == GGML_TYPE_F32);
    GGML_ASSERT(inject->type    == GGML_TYPE_F32);
    GGML_ASSERT(dst->type       == GGML_TYPE_F32);

    const int hc = ggml_get_op_params_i32(dst, 0);
    GGML_ASSERT(hc > 0 && hc <= 8);

    const int64_t n_embd   = residual->ne[0];
    const int64_t n_tokens = residual->ne[2];

    GGML_ASSERT(n_tokens >= 1 && n_tokens <= HC_FUSED_MAX_TOKENS);  // decode/verify band
    GGML_ASSERT(residual->ne[1] == hc);
    GGML_ASSERT(block_out->ne[0] == n_embd);
    GGML_ASSERT(inject->ne[0] == hc);
    GGML_ASSERT(residual->nb[1] == n_embd*sizeof(float));   // contiguous rows
    GGML_ASSERT(dst->nb[1]      == n_embd*sizeof(float));
    GGML_ASSERT(residual->nb[2] == n_embd*hc*sizeof(float));  // contiguous tokens
    GGML_ASSERT(dst->nb[2]      == n_embd*hc*sizeof(float));
    GGML_ASSERT(block_out->ne[1] == n_tokens || block_out->ne[1] == 1);  // broadcast allowed
    GGML_ASSERT(inject->ne[1]    == n_tokens || inject->ne[1]    == 1);
    // block_out and inject may be broadcast (ne[1] == 1) and inject is often a
    // view into the mix output, whose row stride is the mix dst stride rather
    // than hc*4: a zero per-token stride expresses the broadcast, otherwise the
    // tensor's own nb[1] is the per-token step
    const int64_t res_stride_t    = residual->nb[2] / sizeof(float);
    const int64_t dst_stride_t    = dst->nb[2]      / sizeof(float);
    const int64_t bo_stride_t     = block_out->ne[1] == 1 ? 0 : block_out->nb[1] / sizeof(float);
    const int64_t inject_stride_t = inject->ne[1]    == 1 ? 0 : inject->nb[1]    / sizeof(float);

    cudaStream_t stream = ctx.stream();

    const dim3 block_nums((n_embd + 255) / 256, n_tokens);
    const dim3 block_dims(256);
    const ggml_cuda_kernel_launch_params launch_params = {block_nums, block_dims, 0, stream};
    ggml_cuda_kernel_launch(hc_combine_kernel, launch_params,
            (const float *) residual->data, (const float *) block_out->data,
            (const float *) inject->data, (float *) dst->data,
            (int) n_embd, hc,
            res_stride_t, bo_stride_t, inject_stride_t, dst_stride_t);
}
