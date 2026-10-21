// Matrix x 2..16 vectors (the speculative VERIFY band) in WMMA iu8 16x16x16, RDNA4.
//
// WHY (r290, contributor patch burnoutIA, PR #317): a verify batch re-runs the dequantization of every
// weight once PER COLUMN.  At 1-4 columns the K-split mmvq kernel is bandwidth bound, but at 7-8 columns it
// is instruction bound, and the target's matvec time grows 25.9 -> 34.3 ms per pass (27B UD-Q4_K_M, 8
// verified tokens).  Here each weight is dequantized ONCE for all the columns and the products go through
// the tensor cores, so the cost is nearly flat in the column count.
//
// SCOPE: this is an OPT-IN kernel family (`--wmma-verify off|auto|N`, alias GGML_CUDA_WMMA_MMVQ, default
// OFF) and it is WIDTH-SPLIT BY DESIGN: 1 column keeps the K-split kernel, N..8 columns take this one. That
// breaks GREEDY-PURITY.md invariant 1 (decode and verify of the same state must take the same kernel) and the
// delivery's byte-identical greedy contract, so it is off by default and NOTHING of the default path
// changes while it is off; see GREEDY-PURITY.md and ENVIRONMENT.md for the exception and its numbers.
//
// MEASURED (r290, 27B UD-Q4_K_M and Q8_0, finite work: llama-batched-bench -npl 1..8, and DFlash2 n=7):
//   TG t/s by verify width, ON vs OFF -- w2 -5..-7 %, w3 -3.5 %, w4 -2..-5 %, w5 -1..-5 %, w6 +1..-4 %,
//   w7 +2..+7 %, w8 +4 % (Q8_0) .. +10.6 % (K-quants, 2 GPU tensor split).  End to end the CYCLE time
//   drops 8-10 % at every acceptance level and the tokens/s follow acceptance.  Hence the floor defaults to
//   7 (`auto`): below it the kernel is SLOWER than the K-split one.
//
// r312 (TODO #329): the low band was measured per type and `auto` is now PER TYPE.  Why the floor cannot simply
// drop to 2: at 2-4 columns the K-split kernel is already at 92-97 % of this box's DRAM roof (590-630 GB/s of
// 644 on a 253-479 MB matrix), so no kernel can win there -- and this one still computes a full 16x16 tile
// whatever the column count (a 16x16x16 iu8 WMMA cannot be narrowed), so it lands 3-8 % behind.  What the
// measurement DID find is that the crossover is type-shaped: Q4_K/Q5_K win from 5 columns where Q6_K, IQ4_XS
// and Q8_0 only win at 8 and LOSE up to 8 % (Q6_K) below it, which the blanket floor of 7 still paid for.
// See wmmv_auto_floor().
//
// DESIGN:
// - one workgroup = one tile of 16 rows; its waves split K into interleaved 256-element super-blocks,
//   the next super-block is loaded before the current one is processed, final reduction in LDS;
// - computes C^T = X^T . W^T: A = activations (16 columns x 16 K, signed), B = weights (16 K x 16 rows);
//   lane (r = lane % 16, g = lane / 16) keeps its row r, so its block scales stay local;
// - each weight is dequantized once for all columns: that is what makes the cost flat;
// - the order of the K positions inside a 32-block is free as long as the activations follow:
//   XL = 1: WMMA 0 on k = g*8 + i, WMMA 1 on 16 + g*8 + i (IQ4_XS, Q6_K, Q8_0);
//   XL = 2: WMMA 0 on k = g*16 + i, WMMA 1 on g*16 + 8 + i (Q4_K, Q5_K: 16 contiguous bytes per lane).
// - activation scale d (block_q8_1.ds): each lane reads one, then a DPP row_share broadcast to the 8 columns
//   the lane accumulates. The min term (Q4_K, Q5_K) uses d.SUM(xq), the sum of the QUANTIZED activations,
//   through one more WMMA against a ones matrix: as vec_dot_q4_K_q8_1 does, and consistent with the main term
//   (ds.y = the float SUM(x) moves the outputs by ~1 % and drops the DFlash2 acceptance).
// - Q8_0 (r290, ours): no dequantization, no min term, no table -- the stored int8 IS the B operand. It has
//   the smallest win (+3.7..+4.2 % at 8 columns) because its K-split kernel is already 74 % of the perfect
//   weight-reuse ideal where a K-quant's is 59 %.  Its blocks are 32 elements, not 256, so each format
//   carries its own per-stride unit (RB) instead of reusing BB.
// - returns false for every case it does not cover, and the caller takes its usual path.
#include "mmvq-wmma.cuh"
#include <cstdlib>

#if defined(GGML_USE_HIP) && (defined(__gfx1200__) || defined(__gfx1201__))
#define WMMV_DEVICE 1
#endif

namespace wmmv {

typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));

#ifdef WMMV_DEVICE
#define WMMA(sa, a, sb, b, c) __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(sa, a, sb, b, c, false)
#else
#define WMMA(sa, a, sb, b, c) (c)
#endif

static __device__ __forceinline__ float h_lo(uint32_t v) { return __half2float(__ushort_as_half((unsigned short)(v & 0xFFFF))); }

struct Act { v2i a0, a1; float dx[8]; };
static constexpr v2i ONES = {0x01010101, 0x01010101};

template <int XL>
static __device__ __forceinline__ void load_act(Act & A, const block_q8_1 * __restrict__ y, const int stride_col_y,
                                                const int kb, const int lane, const int N) {
    const int c = lane & 15, g = lane >> 4;
    A.a0 = v2i{0, 0}; A.a1 = v2i{0, 0};
    if (c < N) {
        const uint8_t * qs = (const uint8_t *) y[(int64_t) c*stride_col_y + kb].qs;
        if constexpr (XL == 2) {
            const int * p = (const int *) (qs + g*16);
            A.a0 = v2i{p[0], p[1]}; A.a1 = v2i{p[2], p[3]};
        } else {
            const int * p0 = (const int *) (qs + g*8);
            const int * p1 = (const int *) (qs + 16 + g*8);
            A.a0 = v2i{p0[0], p0[1]}; A.a1 = v2i{p1[0], p1[1]};
        }
    }
    // Lane i of row g reads the scale of column 8g + i%8; row_share:e broadcasts lane e of each row.
    const int cc = (lane & 7) + 8*g;
    int ds = 0;
    if (cc < N) {
        ds = *(const int *) &y[(int64_t) cc*stride_col_y + kb].ds;
    }
#ifdef WMMV_DEVICE
#define WMMV_SHARE(e) { const uint32_t v = (uint32_t) __builtin_amdgcn_update_dpp(0, ds, 0x150 + e, 0xF, 0xF, false); \
                        A.dx[e] = h_lo(v); }
#else
#define WMMV_SHARE(e) { A.dx[e] = h_lo((uint32_t) ds); }
#endif
    WMMV_SHARE(0) WMMV_SHARE(1) WMMV_SHARE(2) WMMV_SHARE(3) WMMV_SHARE(4) WMMV_SHARE(5) WMMV_SHARE(6) WMMV_SHARE(7)
#undef WMMV_SHARE
}

static __device__ __forceinline__ void scale_min(const int j, const uint32_t s0, const uint32_t s1, const uint32_t s2, int & sc, int & mn) {
    auto byte = [&](int i) -> uint32_t { const uint32_t w = i < 4 ? s0 : (i < 8 ? s1 : s2); return (w >> (8*(i & 3))) & 0xFF; };
    if (j < 4) { sc = byte(j) & 63; mn = byte(j + 4) & 63; }
    else { sc = (byte(j + 4) & 0xF) | ((byte(j - 4) >> 6) << 4); mn = (byte(j + 4) >> 4) | ((byte(j) >> 6) << 4); }
}
// bit j of each byte -> bit 4
static __device__ __forceinline__ uint32_t hb4(const uint32_t h, const int j) {
    return j <= 4 ? (h << (4 - j)) & 0x10101010u : (h >> (j - 4)) & 0x10101010u;
}
static __device__ __forceinline__ uint32_t sext6(const uint32_t v) {
    const uint32_t t = v & 0x20202020u;
    return v | (t << 1) | (t << 2);
}

// ---------------------------------------------------------------- Q4_K: d, dmin, scales[12] (16 B), qs[128] at 16
struct Q4K {
    static constexpr int BB = 144, RB = BB, XL = 2;
    struct Regs { uint4 q[4]; uint4 hdr; };
    static __device__ __forceinline__ void load(Regs & R, const uint8_t * b, const int g) {
        R.hdr = *(const uint4 *) b;
#pragma unroll
        for (int p = 0; p < 4; ++p) R.q[p] = *(const uint4 *) (b + 16 + p*32 + g*16);
    }
    static __device__ __forceinline__ void compute(const Regs & R, const int s, const int lane, const block_q8_1 * y,
                                                   const int stride_col_y, const int N, float acc[8]) {
        const float d = h_lo(R.hdr.x), dmin = h_lo(R.hdr.x >> 16);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            int sc, mn; scale_min(j, R.hdr.y, R.hdr.z, R.hdr.w, sc, mn);
            const float fs = d*(float) sc, fm = dmin*(float) mn;
            const uint4 q = R.q[j >> 1]; const int sh = 4*(j & 1);
            const v2i w0 = {(int) ((q.x >> sh) & 0x0F0F0F0Fu), (int) ((q.y >> sh) & 0x0F0F0F0Fu)};
            const v2i w1 = {(int) ((q.z >> sh) & 0x0F0F0F0Fu), (int) ((q.w >> sh) & 0x0F0F0F0Fu)};
            Act A; load_act<XL>(A, y, stride_col_y, s*8 + j, lane, N);
            v8i c = {}, cs = {};
            c  = WMMA(true, A.a0, false, w0, c);
            c  = WMMA(true, A.a1, false, w1, c);
            // the column's Σxq over the block (as d8·dot2 in vec_dot_q4_K_q8_1)
            cs = WMMA(true, A.a0, false, ONES, cs);
            cs = WMMA(true, A.a1, false, ONES, cs);
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[e] += A.dx[e]*(fs*(float) c[e] - fm*(float) cs[e]);
        }
    }
};

// ---------------------------------------------------------------- Q5_K: 16 B header, qh[32] at 16, qs[128] at 48
struct Q5K {
    static constexpr int BB = 176, RB = BB, XL = 2;
    struct Regs { uint4 q[4]; uint4 h; uint4 hdr; };
    static __device__ __forceinline__ void load(Regs & R, const uint8_t * b, const int g) {
        R.hdr = *(const uint4 *) b;
        R.h   = *(const uint4 *) (b + 16 + g*16);
#pragma unroll
        for (int p = 0; p < 4; ++p) R.q[p] = *(const uint4 *) (b + 48 + p*32 + g*16);
    }
    static __device__ __forceinline__ void compute(const Regs & R, const int s, const int lane, const block_q8_1 * y,
                                                   const int stride_col_y, const int N, float acc[8]) {
        const float d = h_lo(R.hdr.x), dmin = h_lo(R.hdr.x >> 16);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            int sc, mn; scale_min(j, R.hdr.y, R.hdr.z, R.hdr.w, sc, mn);
            const float fs = d*(float) sc, fm = dmin*(float) mn;
            const uint4 q = R.q[j >> 1]; const int sh = 4*(j & 1);
            const v2i w0 = {(int) (((q.x >> sh) & 0x0F0F0F0Fu) | hb4(R.h.x, j)), (int) (((q.y >> sh) & 0x0F0F0F0Fu) | hb4(R.h.y, j))};
            const v2i w1 = {(int) (((q.z >> sh) & 0x0F0F0F0Fu) | hb4(R.h.z, j)), (int) (((q.w >> sh) & 0x0F0F0F0Fu) | hb4(R.h.w, j))};
            Act A; load_act<XL>(A, y, stride_col_y, s*8 + j, lane, N);
            v8i c = {}, cs = {};
            c  = WMMA(true, A.a0, false, w0, c);
            c  = WMMA(true, A.a1, false, w1, c);
            // the column's Σxq over the block (as d8·dot2 in vec_dot_q4_K_q8_1)
            cs = WMMA(true, A.a0, false, ONES, cs);
            cs = WMMA(true, A.a1, false, ONES, cs);
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[e] += A.dx[e]*(fs*(float) c[e] - fm*(float) cs[e]);
        }
    }
};

// ---------------------------------------------------------------- Q6_K: ql[128], qh[64] at 128, scales[16] at 192, d at 208
// 210-byte blocks: unaligned loads (the hardware accepts them).
struct Q6K {
    static constexpr int BB = 210, RB = BB, XL = 1;
    struct Regs { uint2 la[4], lb[4], ha[2], hb[2]; uint4 sc; uint32_t dd; };
    static __device__ __forceinline__ void load(Regs & R, const uint8_t * b, const int g) {
#pragma unroll
        for (int n = 0; n < 2; ++n) {
#pragma unroll
            for (int o = 0; o < 2; ++o) {
                R.la[n*2 + o] = *(const uint2 *) (b + n*64 + o*32 + g*8);
                R.lb[n*2 + o] = *(const uint2 *) (b + n*64 + o*32 + 16 + g*8);
            }
            R.ha[n] = *(const uint2 *) (b + 128 + n*32 + g*8);
            R.hb[n] = *(const uint2 *) (b + 128 + n*32 + 16 + g*8);
        }
        R.sc = *(const uint4 *) (b + 192);
        R.dd = *(const uint16_t *) (b + 208);
    }
    static __device__ __forceinline__ uint32_t q6(const uint32_t lo, const int sl, const uint32_t h, const int qt) {
        const uint32_t v = ((lo >> sl) & 0x0F0F0F0Fu) | ((qt <= 1 ? (h << (4 - 2*qt)) : (h >> (2*qt - 4))) & 0x30303030u);
        return sext6(v ^ 0x20202020u);
    }
    static __device__ __forceinline__ void compute(const Regs & R, const int s, const int lane, const block_q8_1 * y,
                                                   const int stride_col_y, const int N, float acc[8]) {
        const float d = h_lo(R.dd);
        const uint32_t sw[4] = {R.sc.x, R.sc.y, R.sc.z, R.sc.w};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int n = j >> 2, qt = j & 3, o = qt & 1, sl = 4*(qt >> 1);
            const uint2 la = R.la[n*2 + o], lb = R.lb[n*2 + o], ha = R.ha[n], hb = R.hb[n];
            const v2i w0 = {(int) q6(la.x, sl, ha.x, qt), (int) q6(la.y, sl, ha.y, qt)};
            const v2i w1 = {(int) q6(lb.x, sl, hb.x, qt), (int) q6(lb.y, sl, hb.y, qt)};
            const int sc0 = (int) (int8_t) (sw[j >> 1] >> (16*(j & 1)));
            const int sc1 = (int) (int8_t) (sw[j >> 1] >> (16*(j & 1) + 8));
            Act A; load_act<XL>(A, y, stride_col_y, s*8 + j, lane, N);
            const v8i z = {};
            const v8i c0 = WMMA(true, A.a0, true, w0, z);
            const v8i c1 = WMMA(true, A.a1, true, w1, z);
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[e] += (d*A.dx[e])*(float) (c0[e]*sc0 + c1[e]*sc1);
        }
    }
};

// ---------------------------------------------------------------- IQ4_XS: d, scales_h, scales_l[4], qs[128] at 8
struct IQ4XS {
    static constexpr int BB = 136, RB = BB, XL = 1;
    struct Regs { uint2 q[8]; uint2 hdr; };
    static __device__ __forceinline__ void load(Regs & R, const uint8_t * b, const int g) {
        R.hdr = *(const uint2 *) b;
#pragma unroll
        for (int j = 0; j < 8; ++j) R.q[j] = *(const uint2 *) (b + 8 + j*16 + g*8);
    }
    // 8 nibbles -> 8 signed kvalues_iq4nl bytes (low -> lo, high -> hi), as get_int_from_table_16.
    static __device__ __forceinline__ void lut(const uint32_t q, uint32_t & lo, uint32_t & hi) {
#ifdef WMMV_DEVICE
        constexpr uint32_t t0 = 0xBFAD9881u, t1 = 0xF6EADDCFu, t2 = 0x26190D01u, t3 = 0x71594535u;
        const uint32_t qe = q & 0x07070707u, qo = (q >> 4) & 0x07070707u;
        const uint32_t el = __builtin_amdgcn_perm(t1, t0, qe), eh = __builtin_amdgcn_perm(t3, t2, qe);
        const uint32_t ol = __builtin_amdgcn_perm(t1, t0, qo), oh = __builtin_amdgcn_perm(t3, t2, qo);
        lo = __builtin_amdgcn_perm(eh, el, 0x03020100u | ((q & 0x08080808u) >> 1));
        hi = __builtin_amdgcn_perm(oh, ol, 0x03020100u | ((q & 0x80808080u) >> 5));
#else
        lo = q; hi = q;
#endif
    }
    static __device__ __forceinline__ void compute(const Regs & R, const int s, const int lane, const block_q8_1 * y,
                                                   const int stride_col_y, const int N, float acc[8]) {
        const float d = h_lo(R.hdr.x);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int ls = (int) (((R.hdr.y >> (4*j)) & 0xF) | (((R.hdr.x >> (16 + 2*j)) & 3) << 4));
            const float dl = d*(float) (ls - 32);
            uint32_t a, b, c2, d2;
            lut(R.q[j].x, a, b); lut(R.q[j].y, c2, d2);
            const v2i w0 = {(int) a, (int) c2}, w1 = {(int) b, (int) d2};
            Act A; load_act<XL>(A, y, stride_col_y, s*8 + j, lane, N);
            v8i c = {};
            c = WMMA(true, A.a0, true, w0, c);
            c = WMMA(true, A.a1, true, w1, c);
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[e] += (dl*A.dx[e])*(float) c[e];
        }
    }
};

// ---------------------------------------------------------------- Q8_0: d, qs[32] at 2 (34-byte blocks, unaligned)
// EXPERIMENT (not part of PR #317): no dequant, no min term -- the weight bytes ARE the int8 operand.
struct Q80 {
    static constexpr int BB = 272, RB = 34, XL = 1;
    struct Regs { uint2 qa[8], qb[8]; uint32_t dd[8]; };
    static __device__ __forceinline__ void load(Regs & R, const uint8_t * b, const int g) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const uint8_t * p = b + j*34 + 2;
            R.qa[j] = *(const uint2 *) (p + g*8);
            R.qb[j] = *(const uint2 *) (p + 16 + g*8);
            R.dd[j] = *(const uint16_t *) (b + j*34);
        }
    }
    static __device__ __forceinline__ void compute(const Regs & R, const int s, const int lane, const block_q8_1 * y,
                                                   const int stride_col_y, const int N, float acc[8]) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float d = h_lo(R.dd[j]);
            const v2i w0 = {(int) R.qa[j].x, (int) R.qa[j].y};
            const v2i w1 = {(int) R.qb[j].x, (int) R.qb[j].y};
            Act A; load_act<XL>(A, y, stride_col_y, s*8 + j, lane, N);
            v8i c = {};
            c = WMMA(true, A.a0, true, w0, c);
            c = WMMA(true, A.a1, true, w1, c);
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[e] += (d*A.dx[e])*(float) c[e];
        }
    }
};

template <class F, int WAVES>
__launch_bounds__(32*WAVES, 1)
static __global__ void mmvq_wmma(const uint8_t * __restrict__ vx, const block_q8_1 * __restrict__ y, float * __restrict__ dst,
                                 const int nsb, const int64_t row_bytes, const int stride_col_y, const int64_t stride_col_dst, const int N) {
#ifdef WMMV_DEVICE
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, tile = blockIdx.x;
    const int r = lane & 15, g = lane >> 4;
    const uint8_t * row = vx + (int64_t) (tile*16 + r)*row_bytes;
    float acc[8] = {0.0f};
    int s = wave;
    typename F::Regs cur, nxt;
    if (s < nsb) F::load(cur, row + (int64_t) s*F::BB, g);
    while (s < nsb) {
        const int sn = s + WAVES;
        if (sn < nsb) F::load(nxt, row + (int64_t) sn*F::BB, g);
        F::compute(cur, s, lane, y, stride_col_y, N, acc);
        cur = nxt; s = sn;
    }
    __shared__ float red[WAVES][8][32];
#pragma unroll
    for (int e = 0; e < 8; ++e) red[wave][e][lane] = acc[e];
    __syncthreads();
    for (int o = threadIdx.x; o < 256; o += 32*WAVES) {
        const int e = o >> 5, l = o & 31;
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < WAVES; ++w) v += red[w][e][l];
        const int col = (l >> 4)*8 + e;
        if (col < N) dst[(int64_t) col*stride_col_dst + tile*16 + (l & 15)] = v;
    }
#else
    GGML_UNUSED_VARS(vx, y, dst, nsb, row_bytes, stride_col_y, stride_col_dst, N);
    NO_DEVICE_CODE;
#endif
}

template <class F>
static bool launch(const void * vx, const void * vy, float * dst, const int64_t ncols_x, const int64_t nrows_x,
                   const int64_t stride_row_x, const int N, const int64_t stride_col_y, const int64_t stride_col_dst, cudaStream_t stream) {
    const int nsb = (int) (ncols_x / 256);
    const dim3 grid((unsigned) (nrows_x / 16));
    const auto * x = (const uint8_t *) vx;
    const auto * y = (const block_q8_1 *) vy;
    const int64_t rb = stride_row_x*F::RB;
    const int scy = (int) stride_col_y;
    // Wave count: it must divide the super-block count (otherwise part of the workgroup idles).  A wave count
    // that does not divide it was measured instead of one that does (k = 3840, nsb = 15: 10 or 8 waves at
    // n = 8 is 1.1 % slower for Q4_K and 2.4 % faster for Q8_0 than the single wave this falls through to), so
    // the exact-divisor rule stays.
    if (nsb % 10 == 0) { mmvq_wmma<F, 10><<<grid, 320, 0, stream>>>(x, y, dst, nsb, rb, scy, stride_col_dst, N); }
    else if (nsb % 8 == 0) { mmvq_wmma<F, 8><<<grid, 256, 0, stream>>>(x, y, dst, nsb, rb, scy, stride_col_dst, N); }
    else if (nsb % 4 == 0) { mmvq_wmma<F, 4><<<grid, 128, 0, stream>>>(x, y, dst, nsb, rb, scy, stride_col_dst, N); }
    else if (nsb % 2 == 0) { mmvq_wmma<F, 2><<<grid, 64, 0, stream>>>(x, y, dst, nsb, rb, scy, stride_col_dst, N); }
    else { mmvq_wmma<F, 1><<<grid, 32, 0, stream>>>(x, y, dst, nsb, rb, scy, stride_col_dst, N); }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

} // namespace wmmv

// The floor: the FIRST column count at which this family replaces the K-split kernel (0 = off).
// `--wmma-verify N` sets it through the backend registry's proc-address table before the first graph is built;
// the GGML_CUDA_WMMA_MMVQ alias from the contributor's patch keeps working, and when it is set it WINS so a
// gate can force the arms (`GGML_CUDA_WMMA_MMVQ=0` disables a run that `--wmma-verify` enabled).
// Latched once (ENVIRONMENT.md: a behaviour variable is resolved once, not per op).
static constexpr int WMMV_AUTO = -1;   // `auto`: the per-type floor below (r312)

static int g_cli_min_cols = 0;
static int g_min_cols     = -2;   // -2 = not resolved yet

// r312: `auto` is PER TYPE, from the measured crossover rather than the blanket 7 it used to resolve to.
// The family's cost is nearly flat in the column count while the K-split kernel's grows, so the crossover is
// where the K-split kernel leaves the DRAM roof -- and that point is type-shaped: the K-quants are still at
// 608-620 GB/s at 5 columns (so they win as soon as the K-split one bends) where Q6_K and IQ4_XS, whose own
// K-split kernels are the fastest per byte, do not win until 8 (and lose 7-8 % at 2..6, which a blanket
// floor of 7 still paid for).  Measured, ENVIRONMENT.md §7.6 / GREEDY-PURITY.md §91: see the per-type table.
static constexpr int wmmv_auto_floor(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_K:   return 5;
        case GGML_TYPE_Q5_K:   return 5;
        case GGML_TYPE_Q6_K:   return 8;
        case GGML_TYPE_IQ4_XS: return 8;
        case GGML_TYPE_Q8_0:   return 8;
        default:               return 0;
    }
}

static int wmmv_clamp(const int n) {
    if (n < 0)  return WMMV_AUTO;   // an explicit `auto`
    if (n == 0) return 0;
    // 1 column is decode, which always keeps the K-split kernel: it is faster there and a decode that took a
    // different kernel from the verify would be a purity break at the widest possible band.
    if (n < 2)  return 2;
    if (n > 16) return 16;
    return n;
}

static int wmmv_min_cols(const ggml_type type) {
    if (g_min_cols == -2) {
        const char * e = getenv("GGML_CUDA_WMMA_MMVQ");
        // `GGML_CUDA_WMMA_MMVQ=auto` is the env spelling of `--wmma-verify auto` (atoi() would read it as 0)
        g_min_cols = e != nullptr ? ((e[0] == 'a' || e[0] == 'A') ? WMMV_AUTO : wmmv_clamp(atoi(e))) : g_cli_min_cols;
    }
    return g_min_cols == WMMV_AUTO ? wmmv_auto_floor(type) : g_min_cols;
}

void ggml_cuda_mmvq_wmma_set_min_cols(const int n) {
    g_cli_min_cols = wmmv_clamp(n);
    if (getenv("GGML_CUDA_WMMA_MMVQ") == nullptr) {
        g_min_cols = g_cli_min_cols;
    }
}

bool ggml_cuda_mmvq_wmma(ggml_type type, const void * vx, const void * vy, float * dst,
                         int64_t ncols_x, int64_t nrows_x, int64_t stride_row_x, int ncols_dst,
                         int64_t stride_col_y, int64_t stride_col_dst, cudaStream_t stream) {
    const int min_cols = wmmv_min_cols(type);
    if (min_cols == 0 || ncols_dst < min_cols || ncols_dst > 16) {
        return false;
    }
    if (!GGML_CUDA_CC_IS_RDNA4(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) {
        return false;
    }
    if (ncols_x % 256 != 0 || nrows_x % 16 != 0) {
        return false;
    }
    switch (type) {
        case GGML_TYPE_Q4_K:   return wmmv::launch<wmmv::Q4K>  (vx, vy, dst, ncols_x, nrows_x, stride_row_x, ncols_dst, stride_col_y, stride_col_dst, stream);
        case GGML_TYPE_Q5_K:   return wmmv::launch<wmmv::Q5K>  (vx, vy, dst, ncols_x, nrows_x, stride_row_x, ncols_dst, stride_col_y, stride_col_dst, stream);
        case GGML_TYPE_Q6_K:   return wmmv::launch<wmmv::Q6K>  (vx, vy, dst, ncols_x, nrows_x, stride_row_x, ncols_dst, stride_col_y, stride_col_dst, stream);
        case GGML_TYPE_Q8_0:   return wmmv::launch<wmmv::Q80>  (vx, vy, dst, ncols_x, nrows_x, stride_row_x, ncols_dst, stride_col_y, stride_col_dst, stream);
        case GGML_TYPE_IQ4_XS: return wmmv::launch<wmmv::IQ4XS>(vx, vy, dst, ncols_x, nrows_x, stride_row_x, ncols_dst, stride_col_y, stride_col_dst, stream);
        default: return false;
    }
}
