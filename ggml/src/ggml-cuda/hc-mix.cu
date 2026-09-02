// Fused hyper-connection mixer tail for qwen4exp decode (nt == 1).
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

#include "hc-mix.cuh"

#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"


// One Q8_0 matrix-vector product against a pre-quantized Q8_1 input. Grid:
// (ceil(nrows / RPB), 1), block: (32, nwarps), RPB rows per block. This is a
// faithful clone of mul_mat_vec_q (ncols_dst = 1): the (row, kblock) items are
// split across the (32, nwarps) threads and reduced with a per-warp butterfly
// followed by a serial add of the warp results. RPB must match the dispatch's
// rows-per-block (1 for K >= the group count, or the short-K override for
// small K) so the sums are bit-identical to the unfused mmvq path.
template <int nwarps, int RPB>
static __global__ void hc_mix_row_dot(
        const block_q8_0 * w, const block_q8_1 * y, float * dst,
        const int nrows, const int blocks_per_row) {
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
            // the vec_dot reads the Q8_1 block from the pointer argument as-is
            // and offsets only the weight by kbx, so pass the per-kblock y pointer
            tmp[i] += vec_dot_q8_0_q8_1(w + (int64_t) (row0 + i) * blocks_per_row, &y[kbx], kbx, kqs);
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
            dst[row0 + i] = tmp[i];
        }
    }
}

// silu(lo/hc) and Q8_1 quantize of the low-rank vector. One block of 32
// threads (one warp) loops over the Q8_1 blocks; each block's 32 lanes are the
// 32 consecutive values, so the warp reduction matches quantize_q8_1.
static __global__ void hc_mix_silu_quant(
        const float * lo, block_q8_1 * y,
        const int n_blocks, const float inv_hc) {
    const int lane = threadIdx.x;
    for (int ib = 0; ib < n_blocks; ++ib) {
        const int col = ib*32 + lane;
        const float v = col < 320 ? ggml_cuda_op_silu_single(lo[col] * inv_hc) : 0.0f;
        float amax = fabsf(v);
        float sum  = v;
        amax = warp_reduce_max<32>(amax);
        sum  = warp_reduce_sum<32>(sum);
        const float  d = amax / 127.0f;
        const int8_t q = amax == 0.0f ? 0 : (int8_t) roundf(v / d);
        y[ib].qs[lane] = q;
        if (lane == 0) {
            y[ib].ds = make_half2(d, sum);
        }
    }
}

// Collapse the gated streams to their mean: mixed[j] = (1/hc) * sum_c xn*c*gate.
// gate holds the raw up projection; sigmoid is applied inline (same formula as
// the standalone op). The products are stored to an array before summing so the
// compiler cannot contract them into FMAs: the reference rounds each xn*gate
// product (a separate MUL op) and then adds the rounded values. One thread per
// output element; the adds follow the graph order (left-to-right ADD chain),
// then SCALE.
static __global__ void hc_mix_collapse(
        const float * xn, const float * gate_raw, float * dst,
        const int n_embd, const int hc, const float inv_hc) {
    const int j = blockIdx.x;
    if (j >= n_embd) {
        return;
    }
    float pp[8];
    for (int c = 0; c < hc; ++c) {
        const float g = 1.0f / (1.0f + expf(-gate_raw[c*n_embd + j]));
        pp[c] = xn[c*n_embd + j] * g;
    }
    float sum = pp[0];
    for (int c = 1; c < hc; ++c) {
        sum = sum + pp[c];
    }
    dst[j] = sum * inv_hc;
}

void ggml_cuda_op_hc_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * xn     = dst->src[0];
    const ggml_tensor * w_down = dst->src[1];
    const ggml_tensor * w_up   = dst->src[2];

    GGML_ASSERT(xn->type     == GGML_TYPE_F32);
    GGML_ASSERT(w_down->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(w_up->type   == GGML_TYPE_Q8_0);
    GGML_ASSERT(dst->type    == GGML_TYPE_F32);

    const int hc = ggml_get_op_params_i32(dst, 0);
    GGML_ASSERT(hc > 0 && hc <= 8);

    const int64_t hc_dim   = xn->ne[0];
    const int64_t n_tokens = xn->ne[1];
    const int64_t n_embd   = hc_dim / hc;
    const int64_t hc_lr    = w_down->ne[1];

    GGML_ASSERT(n_tokens == 1);             // decode-only fused op
    GGML_ASSERT(hc_dim % 32 == 0 && hc_lr % 32 == 0);
    GGML_ASSERT(xn->nb[1] == hc_dim*sizeof(float)); // contiguous
    GGML_ASSERT(w_down->ne[1] == hc_lr && w_up->ne[0] == hc_lr && w_up->ne[1] == hc_dim);


    const float * xn_d = (const float *) xn->data;
    float * dst_d      = (float *) dst->data;

    cudaStream_t stream = ctx.stream();

    const int blocks_down = hc_dim / 32; // xn Q8_1 blocks for the down dots
    const int blocks_up   = hc_lr  / 32; // v  Q8_1 blocks for the up dots

    ggml_cuda_pool & pool = ctx.pool();

    ggml_cuda_pool_alloc<block_q8_1> y_xn_alloc(pool, blocks_down);
    ggml_cuda_pool_alloc<float>      lo_alloc(pool, hc_lr);
    ggml_cuda_pool_alloc<block_q8_1> y_v_alloc(pool, blocks_up);
    ggml_cuda_pool_alloc<float>      gate_alloc(pool, hc_dim);

    block_q8_1 * y_xn = y_xn_alloc.get();
    float      * lo   = lo_alloc.get();
    block_q8_1 * y_v  = y_v_alloc.get();
    float      * gate = gate_alloc.get();

    // xn -> Q8_1 (same kernel + semantics as the mmvq quantize path)
    quantize_row_q8_1_cuda(xn_d, nullptr, y_xn, GGML_TYPE_Q8_0,
            hc_dim, hc_dim, 0, 0, hc_dim, 1, 1, 1, stream);

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

    // lo_raw = w_down^T xn: 320 rows x 10240 dots
    {
        const dim3 block_nums((hc_lr + rpb_down - 1) / rpb_down);
        const dim3 block_dims(32, 8);
        const ggml_cuda_kernel_launch_params launch_params = {block_nums, block_dims, 0, stream};
        if (rpb_down == 1) {
            ggml_cuda_kernel_launch(hc_mix_row_dot<8, 1>, launch_params,
                    (const block_q8_0 *) w_down->data, y_xn, lo, hc_lr, blocks_down);
        } else {
            GGML_ABORT("hc_mix: unexpected down rpb %d\n", rpb_down);
        }
    }

    // v = silu(lo/hc), then Q8_1
    {
        const dim3 block_nums(1);
        const dim3 block_dims(32);
        const ggml_cuda_kernel_launch_params launch_params = {block_nums, block_dims, 0, stream};
        ggml_cuda_kernel_launch(hc_mix_silu_quant, launch_params,
                lo, y_v, blocks_up, 1.0f / (float) hc);
    }

    // gate_raw = w_up^T v: 10240 rows x 320 dots (short K -> rpb override)
    {
        const dim3 block_nums((hc_dim + rpb_up - 1) / rpb_up);
        const dim3 block_dims(32, 8);
        const ggml_cuda_kernel_launch_params launch_params = {block_nums, block_dims, 0, stream};
        switch (rpb_up) {
            case 1:  ggml_cuda_kernel_launch(hc_mix_row_dot<8, 1>,  launch_params, (const block_q8_0 *) w_up->data, y_v, gate, hc_dim, blocks_up); break;
            case 2:  ggml_cuda_kernel_launch(hc_mix_row_dot<8, 2>,  launch_params, (const block_q8_0 *) w_up->data, y_v, gate, hc_dim, blocks_up); break;
            case 4:  ggml_cuda_kernel_launch(hc_mix_row_dot<8, 4>,  launch_params, (const block_q8_0 *) w_up->data, y_v, gate, hc_dim, blocks_up); break;
            case 8:  ggml_cuda_kernel_launch(hc_mix_row_dot<8, 8>,  launch_params, (const block_q8_0 *) w_up->data, y_v, gate, hc_dim, blocks_up); break;
            case 16: ggml_cuda_kernel_launch(hc_mix_row_dot<8, 16>, launch_params, (const block_q8_0 *) w_up->data, y_v, gate, hc_dim, blocks_up); break;
            default: GGML_ABORT("hc_mix: unexpected up rpb %d\n", rpb_up); break;
        }
    }

    // mixed = (1/hc) * sum_c xn * sigmoid(gate)
    {
        const dim3 block_nums(n_embd);
        const dim3 block_dims(1);
        const ggml_cuda_kernel_launch_params launch_params = {block_nums, block_dims, 0, stream};
        ggml_cuda_kernel_launch(hc_mix_collapse, launch_params,
                xn_d, gate, dst_d, n_embd, hc, 1.0f / (float) hc);
    }

}

// Fused hyper-connection residual combine for qwen4exp decode (nt == 1).
// out[r, c] = residual[r, c] + block_out[r] * w[c], with
// w[c] = 2 * sigmoid(inject[c] / hc) (the SCALE+SIGMOID+SCALE chain of
// build_hc_combine; the 1/hc and 2.0 scalars are exact in f32 so only the
// sigmoid rounds). The product block_out[r]*w[c] and the residual add keep
// their own roundings because the reference runs separate MUL and ADD ops:
// the products go to an array first so the compiler cannot contract them
// into FMAs. One thread per row handles the hc columns, mirroring the
// collapse kernel; w is computed once per block into smem.
static __global__ void hc_combine_kernel(
        const float * residual, const float * block_out, const float * inject,
        float * dst, const int n_embd, const int hc, const int n_tokens) {
    const float inv_hc = 1.0f / (float) hc;

    __shared__ float w_s[8];
    if (threadIdx.x < hc) {
        const float s = inject[threadIdx.x] * inv_hc;
        w_s[threadIdx.x] = (1.0f / (1.0f + expf(-s))) * 2.0f;
    }
    __syncthreads();

    for (int t = 0; t < n_tokens; ++t) {
        const int r = blockIdx.x*blockDim.x + threadIdx.x;
        if (r >= n_embd) {
            return;
        }
        const float * res = residual + (int64_t) t*n_embd*hc + r;
        const float   bo  = block_out[(int64_t) t*n_embd + r];
        float       * dt  = dst + (int64_t) t*n_embd*hc + r;
        float pp[8];
        for (int c = 0; c < hc; ++c) {
            pp[c] = bo * w_s[c];
        }
        for (int c = 0; c < hc; ++c) {
            dt[c*n_embd] = res[c*n_embd] + pp[c];
        }
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

    GGML_ASSERT(n_tokens == 1);                 // decode-only fused op
    GGML_ASSERT(residual->ne[1] == hc);
    GGML_ASSERT(block_out->ne[0] == n_embd);
    GGML_ASSERT(inject->ne[0] == hc);
    GGML_ASSERT(residual->nb[1] == n_embd*sizeof(float));  // contiguous rows
    GGML_ASSERT(inject->nb[1]   == hc*sizeof(float));

    cudaStream_t stream = ctx.stream();

    const dim3 block_nums((n_embd + 255) / 256);
    const dim3 block_dims(256);
    const ggml_cuda_kernel_launch_params launch_params = {block_nums, block_dims, 0, stream};
    ggml_cuda_kernel_launch(hc_combine_kernel, launch_params,
            (const float *) residual->data, (const float *) block_out->data,
            (const float *) inject->data, (float *) dst->data,
            (int) n_embd, hc, (int) n_tokens);
}
