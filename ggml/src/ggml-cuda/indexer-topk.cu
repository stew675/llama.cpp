#include "indexer-topk.cuh"
#include "common.cuh"

// Fused indexer expand + mask + top-k.
//
// The qwen4exp indexer scores blocks, then every cell of a block carries the
// block score, the attention mask is added per cell, and the top-k cells are
// selected.  The plain graph materializes the full [n_kv, n_tps] F32 expanded
// tensor (512 MB at 64K, mirrored on every GPU) only to feed a radix top-k
// that re-reads it 5 times.  This op computes
//
//     value(c) = score[cell_blk(c)] + additive(c)
//
// on the fly in each radix pass, so the expanded tensor never exists.
//
// Layouts:
//   score     [n_blocks, n_tps, n_stream] F32
//   cell_blk  [n_kv, n_stream] I32
//   additive  [n_kv, n_tps, n_stream] F16 or F32 (attention mask or bias)
//   dst       [k, n_tps, 1, n_stream] I32
//
// Each row (tps, stream) is an independent top-k over n_kv cells.  The
// score gather goes through cell_blk, which has only n_blocks distinct
// values (r cells per block), so the score column stays in L2.

static __device__ __forceinline__ uint32_t indexer_topk_float_to_ordered(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}

struct indexer_topk_radix_state {
    uint32_t prefix;
    uint32_t prefix_mask;
    int rank;
    int greater_count;
    int equal_count;
};

static __global__ void indexer_topk_radix_init(indexer_topk_radix_state * states, int nrows, int k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row] = {0, 0, k, 0, 0};
    }
}

// value(c) for row r: score[cell_blk(c, s), t, s] + bias(b, t, s) + additive(c, t, s)
// the per-block bias is already folded into score unless extra.blk_idx is set, and the
// per-cell additive is the mask (or the bias when blk_bias is off) unless extra.cell_pos is
struct indexer_topk_extra {
    const int * cell_pos;   // I32 [n_kv, n_stream], -1 for an empty or foreign cell
    const int * q_pos;      // I32 [n_tps, n_stream]
    const int * blk_idx;    // I32 [n_blocks, n_stream], -1 incomplete, INT32_MAX for the spare
    const int * blk_tail;   // I32 [n_tps, n_stream]
};

// ordered key of a whole block when the additive is null: value = score[block] + derived bias.
// every cell of the block shares it, so the histogram/gather evaluate it once per block.
static __device__ __forceinline__ uint32_t indexer_topk_block_key(
        const float * __restrict__ score,
        const indexer_topk_extra extra,
        int b, int t, int s, int n_blocks, int n_tps) {
    float sc = score[b + t*n_blocks + s*n_blocks*n_tps];
    if (extra.blk_idx != nullptr) {
        const int bi = extra.blk_idx[b + s*n_blocks];
        const int tail = extra.blk_tail[t + s*n_tps];
        sc += bi < 0 ? -INFINITY : (bi >= tail ? 1e9f : 0.0f);
    }
    return indexer_topk_float_to_ordered(sc);
}

template<typename kv_t>
static __device__ __forceinline__ float indexer_topk_value(
        const float * __restrict__ score,
        const int   * __restrict__ cell_blk,
        const kv_t  * __restrict__ additive,
        const indexer_topk_extra extra,
        int c, int t, int s,
        int n_blocks, int n_tps, int n_kv) {
    const int b = cell_blk[c + s*n_kv];
    float sc = score[b + t*n_blocks + s*n_blocks*n_tps];

    if (extra.blk_idx != nullptr) {
        // the block's first-cell position against this token's tail start: the tail is the
        // incomplete block and is always visible.  a foreign block needs no -inf here - the
        // visibility below drops every one of its cells, and -inf + -inf is still -inf, so
        // the values (and the selection) stay identical.
        const int bi = extra.blk_idx[b + s*n_blocks];
        sc += bi < 0 ? -INFINITY : (bi >= extra.blk_tail[t + s*n_tps] ? 1e9f : 0.0f);
    }

    if (extra.cell_pos != nullptr) {
        // same predicate as set_input_kq_mask_impl: empty, foreign and future cells are -inf
        const int cp = extra.cell_pos[c + s*n_kv];
        return sc + (cp >= 0 && cp <= extra.q_pos[t + s*n_tps] ? 0.0f : -INFINITY);
    }

    return sc + (float) additive[c + t*n_kv + s*n_kv*n_tps];
}

template<int BLOCK_SIZE, int RADIX_BITS, typename kv_t>
static __global__ void indexer_topk_radix_histogram(
        const float * __restrict__ score,
        const int   * __restrict__ cell_blk,
        const kv_t  * __restrict__ additive,
        const indexer_topk_extra extra,
        const indexer_topk_radix_state * __restrict__ states,
        int * __restrict__ block_histograms,
        int ncols, int n_tps, int n_blocks, int n_kv,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const int t = row % n_tps;
    const int s = row / n_tps;
    __shared__ int histogram[NBINS];

    for (int i = tid; i < NBINS; i += BLOCK_SIZE) {
        histogram[i] = 0;
    }
    __syncthreads();

    const indexer_topk_radix_state state = states[row];
    // contiguous column range per block (instead of the old interleaved stride): this lets the
    // per-block histograms yield the gather's per-block greater/equal counts directly, so the
    // old full-width count pass is gone (see indexer_topk_hist_accum)
    const int chunk = (ncols + blocks_per_row - 1) / blocks_per_row;
    const int col0 = row_block * chunk;
    const int col1 = min(col0 + chunk, ncols);
    for (int col = col0 + tid; col < col1; col += BLOCK_SIZE) {
        const uint32_t key = indexer_topk_float_to_ordered(
                indexer_topk_value(score, cell_blk, additive, extra, col, t, s, n_blocks, n_tps, n_kv));
        if ((key & state.prefix_mask) == state.prefix) {
            atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
        }
    }
    __syncthreads();

    const size_t histogram_offset =
        ((size_t) row * blocks_per_row + row_block) * NBINS;
    block_histograms[histogram_offset + tid] = histogram[tid];
}

// block-granularity histogram: with no additive the per-cell value is
// `score[cell_blk[c]] + derived-bias` and the visibility is a per-cell predicate, so every cell of
// one block shares one ordered key.  each thread walks a contiguous run of VEC cells and recomputes
// the block key only when the block changes (the common `ratio == 4` case: once per thread step),
// which removes the per-cell score gather.  bit-identical to the cell-level kernel -- only the
// integer bin counts matter, and they are the same.
template<int BLOCK_SIZE, int RADIX_BITS, int VEC>
static __global__ void indexer_topk_radix_histogram_grouped(
        const float * __restrict__ score,
        const int   * __restrict__ cell_blk,
        const indexer_topk_extra extra,
        const indexer_topk_radix_state * __restrict__ states,
        int * __restrict__ block_histograms,
        int ncols, int n_tps, int n_blocks, int n_kv,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const int t = row % n_tps;
    const int s = row / n_tps;
    __shared__ int histogram[NBINS];

    for (int i = tid; i < NBINS; i += BLOCK_SIZE) {
        histogram[i] = 0;
    }
    __syncthreads();

    const indexer_topk_radix_state state = states[row];
    const uint32_t key_inf = indexer_topk_float_to_ordered(-INFINITY);
    const int q    = extra.q_pos    != nullptr ? extra.q_pos[t + s*n_tps]    : 0;
    const int nskv = s*n_kv;

    const int chunk = (ncols + blocks_per_row - 1) / blocks_per_row;
    const int col0 = row_block * chunk;
    const int col1 = min(col0 + chunk, ncols);
    for (int base = col0 + tid*VEC; base < col1; base += BLOCK_SIZE*VEC) {
        int      cur_b   = -1;
        uint32_t cur_key = key_inf;
        #pragma unroll
        for (int v = 0; v < VEC; ++v) {
            const int c = base + v;
            if (c >= col1) {
                break;
            }
            const int b = cell_blk[c + nskv];
            if (b != cur_b) {
                cur_b = b;
                cur_key = indexer_topk_block_key(score, extra, b, t, s, n_blocks, n_tps);
            }
            uint32_t key = cur_key;
            if (extra.cell_pos != nullptr) {
                const int cp = extra.cell_pos[c + nskv];
                if (cp < 0 || cp > q) {
                    key = key_inf;
                }
            }
            if ((key & state.prefix_mask) == state.prefix) {
                atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
            }
        }
    }
    __syncthreads();

    const size_t histogram_offset =
        ((size_t) row * blocks_per_row + row_block) * NBINS;
    block_histograms[histogram_offset + tid] = histogram[tid];
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void indexer_topk_radix_select(
        const int * __restrict__ block_histograms,
        indexer_topk_radix_state * __restrict__ states,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int histogram[NBINS];

    int count = 0;
    for (int i = tid; i < NBINS; i += BLOCK_SIZE) {
        count = 0;
        for (int row_block = 0; row_block < blocks_per_row; ++row_block) {
            const size_t offset = ((size_t) row * blocks_per_row + row_block) * NBINS;
            count += block_histograms[offset + i];
        }
        histogram[i] = count;
    }
    __syncthreads();

    if (tid == 0) {
        indexer_topk_radix_state state = states[row];
        int bin = NBINS - 1;
        while (bin > 0 && histogram[bin] < state.rank) {
            state.rank -= histogram[bin--];
        }
        state.prefix |= (uint32_t) bin << shift;
        state.prefix_mask |= (uint32_t) (NBINS - 1) << shift;
        states[row] = state;
    }
}

// ---------------------------------------------------------------------------
// deterministic gather.  the old implementation placed the selected cells with
// atomicAdd counters, so with more than one block per row the output LIST ORDER
// (and, at the rank boundary, which tied cells made it) varied run to run.  the
// QSA kernel's online softmax is order-sensitive at the ulp level, and this
// model's f16-state recurrences amplify that into llama-cli-visible
// nondeterminism.  Instead, cells are written in ascending column order.
//
// The gather's per-block greater/equal counts no longer need a second full-width
// key-evaluation pass: a cell with key > the final prefix is greater exactly at
// the first byte where it differs, so it lands in a bin above the selected one in
// exactly one radix pass, and the equal cells are the selected bin of the last
// pass.  Accumulating each pass's suffix counts into g_cnt (and the last pass's
// selected bin into e_cnt) removes the old `indexer_topk_count` pass entirely.

template<int RADIX_BITS>
static __global__ void indexer_topk_hist_accum(
        const int * __restrict__ block_histograms,
        const indexer_topk_radix_state * __restrict__ states,
        int * __restrict__ g_cnt, int * __restrict__ e_cnt,
        int nrows, int blocks_per_row, int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;   // == blockDim.x
    const int idx = blockIdx.x;
    const int row = idx / blocks_per_row;
    const int bin = threadIdx.x;
    const int sel = (int) ((states[row].prefix >> shift) & (NBINS - 1));
    const int * hist = block_histograms + (size_t) idx * NBINS;

    // coalesced read of the block's bins, summed with a warp shuffle reduction
    const int v = (bin > sel) ? hist[bin] : 0;
    __shared__ int warpsum[NBINS / 32];
    int s = v;
    for (int off = 16; off > 0; off >>= 1) {
        s += __shfl_down(s, off);
    }
    if ((bin & 31) == 0) {
        warpsum[bin >> 5] = s;
    }
    __syncthreads();
    if (bin == 0) {
        int total = 0;
        for (int w = 0; w < NBINS / 32; ++w) {
            total += warpsum[w];
        }
        g_cnt[idx] += total;
        if (shift == 0) {
            e_cnt[idx] = hist[sel];
        }
    }
}

static __global__ void indexer_topk_base_scan(
        const int * __restrict__ g_cnt, const int * __restrict__ e_cnt,
        int * __restrict__ g_base, int * __restrict__ e_base,
        int nrows, int blocks_per_row) {
    constexpr int BLOCK_SIZE = 256;
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int sg[BLOCK_SIZE], se[BLOCK_SIZE];
    __shared__ int scg, sce;
    if (tid == 0) { scg = 0; sce = 0; }
    __syncthreads();
    for (int chunk = 0; chunk*BLOCK_SIZE < blocks_per_row; ++chunk) {
        const int i = chunk*BLOCK_SIZE + tid;
        const int own_g = (i < blocks_per_row) ? g_cnt[row*blocks_per_row + i] : 0;
        const int own_e = (i < blocks_per_row) ? e_cnt[row*blocks_per_row + i] : 0;
        sg[tid] = own_g;
        se[tid] = own_e;
        // Hillis-Steele inclusive scan in shared
        for (int off = 1; off < BLOCK_SIZE; off <<= 1) {
            __syncthreads();
            const int vg = (tid >= off) ? sg[tid - off] : 0;
            const int ve = (tid >= off) ? se[tid - off] : 0;
            __syncthreads();
            if (tid >= off) { sg[tid] += vg; se[tid] += ve; }
        }
        __syncthreads();
        if (i < blocks_per_row) {
            // exclusive prefix = inclusive minus this element's own count, plus the carry
            g_base[row*blocks_per_row + i] = sg[tid] - own_g + scg;
            e_base[row*blocks_per_row + i] = se[tid] - own_e + sce;
        }
        __syncthreads();
        if (tid == 0) {
            scg += sg[BLOCK_SIZE - 1];
            sce += se[BLOCK_SIZE - 1];
        }
        __syncthreads();
    }
}

// one block per (row, coarse column range).  the per-block greater/equal counts
// (from indexer_topk_hist_accum) give the base, and a running carry over the
// range's 256-column tiles keeps the ascending-column placement exact.
template<typename kv_t>
static __global__ void indexer_topk_write_blocks(
        const float * __restrict__ score,
        const int   * __restrict__ cell_blk,
        const kv_t  * __restrict__ additive,
        const indexer_topk_extra extra,
        const indexer_topk_radix_state * __restrict__ states,
        const int * __restrict__ g_base, const int * __restrict__ e_base,
        int * __restrict__ dst,
        int ncols, int n_tps, int n_blocks, int n_kv, int k,
        int blocks_per_row) {
    constexpr int BLOCK_SIZE = 256;
    const int row = blockIdx.x / blocks_per_row;
    const int b   = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const int t = row % n_tps;
    const int s = row / n_tps;

    const int chunk = (ncols + blocks_per_row - 1) / blocks_per_row;
    const int col0  = b * chunk;
    const int col1  = min(col0 + chunk, ncols);

    int * row_dst = dst + (size_t) row*k;
    const int g_base_b = g_base[row*blocks_per_row + b];
    const int e_base_b = e_base[row*blocks_per_row + b];
    const int rank = states[row].rank;

    __shared__ int sg[BLOCK_SIZE], se[BLOCK_SIZE];
    __shared__ int carry_g, carry_e;
    if (tid == 0) { carry_g = 0; carry_e = 0; }
    __syncthreads();
    for (int c0 = col0; c0 < col1; c0 += BLOCK_SIZE) {
        // read the carry through shared memory: a per-thread register would not be visible
        // to the other threads after the previous tile's update
        const int running_g = carry_g;
        const int running_e = carry_e;
        const int col = c0 + tid;
        int g = 0, e = 0;
        if (col < col1) {
            const uint32_t key = indexer_topk_float_to_ordered(
                    indexer_topk_value(score, cell_blk, additive, extra, col, t, s, n_blocks, n_tps, n_kv));
            g = (key >  states[row].prefix) ? 1 : 0;
            e = (key == states[row].prefix) ? 1 : 0;
        }
        sg[tid] = g;
        se[tid] = e;
        for (int off = 1; off < BLOCK_SIZE; off <<= 1) {
            __syncthreads();
            const int vg = (tid >= off) ? sg[tid - off] : 0;
            const int ve = (tid >= off) ? se[tid - off] : 0;
            __syncthreads();
            if (tid >= off) { sg[tid] += vg; se[tid] += ve; }
        }
        __syncthreads();
        if (g) {
            row_dst[g_base_b + running_g + sg[tid] - 1] = col;
        } else if (e) {
            const int pos = se[tid] - 1;
            if (e_base_b + running_e + pos < rank) {
                row_dst[k - rank + e_base_b + running_e + pos] = col;
            }
        }
        __syncthreads();
        if (tid == BLOCK_SIZE - 1) {
            carry_g += sg[BLOCK_SIZE - 1];
            carry_e += se[BLOCK_SIZE - 1];
        }
        __syncthreads();
    }
}

// block-key-sharing gather for the additive==null case.  each thread walks VEC contiguous cells
// and caches the block key; the block scan then runs over the per-thread totals (the cells stay in
// ascending column order across threads) so the placement is identical to indexer_topk_write_blocks.
template<int VEC>
static __global__ void indexer_topk_write_blocks_grouped(
        const float * __restrict__ score,
        const int   * __restrict__ cell_blk,
        const indexer_topk_extra extra,
        const indexer_topk_radix_state * __restrict__ states,
        const int * __restrict__ g_base, const int * __restrict__ e_base,
        int * __restrict__ dst,
        int ncols, int n_tps, int n_blocks, int n_kv, int k,
        int blocks_per_row, int chunk_in) {
    constexpr int BLOCK_SIZE = 256;
    const int row = blockIdx.x / blocks_per_row;
    const int b   = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const int t = row % n_tps;
    const int s = row / n_tps;

    const int chunk = chunk_in > 0 ? chunk_in : (ncols + blocks_per_row - 1) / blocks_per_row;
    const int col0  = b * chunk;
    const int col1  = min(col0 + chunk, ncols);

    int * row_dst = dst + (size_t) row*k;
    const int g_base_b = g_base[row*blocks_per_row + b];
    const int e_base_b = e_base[row*blocks_per_row + b];
    const int rank = states[row].rank;
    const uint32_t prefix  = states[row].prefix;
    const uint32_t key_inf = indexer_topk_float_to_ordered(-INFINITY);
    const int q    = extra.q_pos != nullptr ? extra.q_pos[t + s*n_tps] : 0;
    const int nskv = s*n_kv;

    __shared__ int sg[BLOCK_SIZE], se[BLOCK_SIZE];
    __shared__ int carry_g, carry_e;
    if (tid == 0) { carry_g = 0; carry_e = 0; }
    __syncthreads();
    for (int t0 = col0; t0 < col1; t0 += BLOCK_SIZE*VEC) {
        const int running_g = carry_g;
        const int running_e = carry_e;
        const int c0 = t0 + tid*VEC;
        int lg = 0, le = 0;
        int gv[VEC], ev[VEC], lgpre[VEC], lepre[VEC];
        int cur_b = -1;
        uint32_t cur_key = key_inf;
        #pragma unroll
        for (int v = 0; v < VEC; ++v) {
            const int c = c0 + v;
            int g = 0, e = 0;
            if (c < col1) {
                const int bb = cell_blk[c + nskv];
                if (bb != cur_b) {
                    cur_b = bb;
                    cur_key = indexer_topk_block_key(score, extra, bb, t, s, n_blocks, n_tps);
                }
                uint32_t key = cur_key;
                if (extra.cell_pos != nullptr) {
                    const int cp = extra.cell_pos[c + nskv];
                    if (cp < 0 || cp > q) {
                        key = key_inf;
                    }
                }
                g = (key >  prefix) ? 1 : 0;
                e = (key == prefix) ? 1 : 0;
            }
            gv[v] = g; ev[v] = e;
            lgpre[v] = lg; lepre[v] = le;
            lg += g; le += e;
        }
        // inclusive block scan of (g_count, e_count).  The Hillis-Steele form this replaces ran
        // 8 steps x 2 __syncthreads (~19 barriers per 1024-cell tile) and dominated the kernel --
        // an unselected-block skip in the cell loop measured NO gain, which is what pointed here.
        // Warp-shuffle scan + a 2-barrier offset pass; integer sums are associative so the
        // prefix values (and therefore the placement) are identical.
        int wg = lg, we = le;
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const int vg = __shfl_up(wg, off);
            const int ve = __shfl_up(we, off);
            if ((tid & 31) >= off) { wg += vg; we += ve; }
        }
        __shared__ int swg[BLOCK_SIZE / 32], swe[BLOCK_SIZE / 32];
        if ((tid & 31) == 31) { swg[tid >> 5] = wg; swe[tid >> 5] = we; }
        __syncthreads();
        if (tid < BLOCK_SIZE / 32) {
            int ag = 0, ae = 0;
            for (int i = 0; i < tid; ++i) { ag += swg[i]; ae += swe[i]; }
            swg[tid] = ag; swe[tid] = ae;
        }
        __syncthreads();
        sg[tid] = wg + swg[tid >> 5];
        se[tid] = we + swe[tid >> 5];
        const int tex_g = sg[tid] - lg;
        const int tex_e = se[tid] - le;
        #pragma unroll
        for (int v = 0; v < VEC; ++v) {
            const int c = c0 + v;
            if (c >= col1) {
                continue;
            }
            if (gv[v]) {
                row_dst[g_base_b + running_g + tex_g + lgpre[v]] = c;
            } else if (ev[v]) {
                const int pos = tex_e + lepre[v];
                if (e_base_b + running_e + pos < rank) {
                    row_dst[k - rank + e_base_b + running_e + pos] = c;
                }
            }
        }
        __syncthreads();
        if (tid == BLOCK_SIZE - 1) {
            carry_g += sg[BLOCK_SIZE - 1];
            carry_e += se[BLOCK_SIZE - 1];
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// block-level histogram path (additive == nullptr, derived default).
//
// A block-aligned partition: hist-block h covers blocks [h*bchunk, (h+1)*bchunk) and the cells
// [h*bchunk*r, min((h+1)*bchunk*r, n_kv)).  Pass 1 is cell-level (it must see every cell to count
// per-block visibility) and accumulates wvis[row*n_blocks + b] = the block's visible-cell count.
// Passes 2..4 are block-level: they walk blocks, read wvis, and bin the count at the block key, so
// three of the four radix passes do ~n_blocks work instead of ~n_kv.  The gather stays cell-level
// (so its ascending-column output order is unchanged); its partition uses the same chunk.

template<int BLOCK_SIZE, int RADIX_BITS, int VEC>
static __global__ void indexer_topk_histogram_pass1(
        const float * __restrict__ score,
        const int   * __restrict__ cell_blk,
        const indexer_topk_extra extra,
        const indexer_topk_radix_state * __restrict__ states,
        int * __restrict__ block_histograms,
        int * __restrict__ wvis,
        int ncols, int n_tps, int n_blocks, int n_kv,
        int blocks_per_row, int bchunk, int r,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;
    const int row = blockIdx.x / blocks_per_row;
    const int hb  = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const int t = row % n_tps;
    const int s = row / n_tps;
    __shared__ int histogram[NBINS];
    for (int i = tid; i < NBINS; i += BLOCK_SIZE) {
        histogram[i] = 0;
    }
    __syncthreads();

    const indexer_topk_radix_state state = states[row];
    const uint32_t key_inf = indexer_topk_float_to_ordered(-INFINITY);
    const int q    = extra.q_pos != nullptr ? extra.q_pos[t + s*n_tps] : 0;
    const int nskv = s*n_kv;
    int * row_wvis = wvis + (size_t) row * n_blocks;

    // every cell of a block shares the block key, and every invisible cell shares key_inf, so a
    // whole run bins into at most two bins.  accumulate per block (and one per-thread pending
    // counter for the invisible key_inf bin) and flush with one atomic per block instead of one
    // atomic per cell.  the integer bin counts are identical -- only the atomic traffic shrinks.
    const int inf_bin = ((key_inf & state.prefix_mask) == state.prefix)
            ? (int) ((key_inf >> shift) & (NBINS - 1)) : -1;
    int inf_pending = 0;

    const int col0 = hb * bchunk * r;
    const int col1 = min(col0 + bchunk * r, ncols);
    for (int base = col0 + tid*VEC; base < col1; base += BLOCK_SIZE*VEC) {
        int      cur_b   = -1;
        int      cur_w   = 0;
        int      cur_h   = 0;
        int      cur_bin = 0;
        bool     cur_match = false;
        #pragma unroll
        for (int v = 0; v < VEC; ++v) {
            const int c = base + v;
            if (c >= col1) {
                break;
            }
            const int b = cell_blk[c + nskv];
            if (b != cur_b) {
                if (cur_b >= 0) {
                    atomicAdd(&row_wvis[cur_b], cur_w);
                    if (cur_match && cur_h > 0) {
                        atomicAdd(&histogram[cur_bin], cur_h);
                    }
                }
                cur_b     = b;
                cur_w     = 0;
                cur_h     = 0;
                const uint32_t cur_key = indexer_topk_block_key(score, extra, b, t, s, n_blocks, n_tps);
                cur_match = (cur_key & state.prefix_mask) == state.prefix;
                cur_bin   = (int) ((cur_key >> shift) & (NBINS - 1));
            }
            bool vis = true;
            if (extra.cell_pos != nullptr) {
                const int cp = extra.cell_pos[c + nskv];
                vis = (cp >= 0 && cp <= q);
            }
            if (vis) {
                cur_w += 1;
                if (cur_match) {
                    cur_h += 1;
                }
            } else if (inf_bin >= 0) {
                inf_pending += 1;
            }
        }
        if (cur_b >= 0) {
            atomicAdd(&row_wvis[cur_b], cur_w);
            if (cur_match && cur_h > 0) {
                atomicAdd(&histogram[cur_bin], cur_h);
            }
        }
    }
    if (inf_pending > 0) {
        atomicAdd(&histogram[inf_bin], inf_pending);
    }
    __syncthreads();
    const size_t off = ((size_t) row * blocks_per_row + hb) * NBINS;
    block_histograms[off + tid] = histogram[tid];
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void indexer_topk_histogram_blocks(
        const float * __restrict__ score,
        const indexer_topk_extra extra,
        const indexer_topk_radix_state * __restrict__ states,
        const int * __restrict__ wvis,
        int * __restrict__ block_histograms,
        int ncols, int n_tps, int n_blocks, int n_kv,
        int blocks_per_row, int bchunk, int r,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;
    const int row = blockIdx.x / blocks_per_row;
    const int hb  = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const int t = row % n_tps;
    const int s = row / n_tps;
    __shared__ int histogram[NBINS];
    __shared__ int s_sumw;
    for (int i = tid; i < NBINS; i += BLOCK_SIZE) {
        histogram[i] = 0;
    }
    if (tid == 0) {
        s_sumw = 0;
    }
    __syncthreads();

    const indexer_topk_radix_state state = states[row];
    const uint32_t key_inf = indexer_topk_float_to_ordered(-INFINITY);
    const int nsblk = s*n_blocks;
    const int * row_wvis = wvis + (size_t) row * n_blocks;

    const int B0 = hb * bchunk;
    const int B1 = min(B0 + bchunk, n_blocks);
    int mysum = 0;
    for (int b = B0 + tid; b < B1; b += BLOCK_SIZE) {
        if (extra.blk_idx[b + nsblk] == -1) {
            continue;   // incomplete block: no cells
        }
        const int w = row_wvis[b];
        mysum += w;
        if (w > 0) {
            const uint32_t key = indexer_topk_block_key(score, extra, b, t, s, n_blocks, n_tps);
            if ((key & state.prefix_mask) == state.prefix) {
                atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], w);
            }
        }
    }
    // warp-reduce mysum, then one atomic per warp: a same-address atomicAdd from every thread
    // serializes (256 per CU per pass) and dominated this kernel.  integer sum is order-free.
    int ws = mysum;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        ws += __shfl_down(ws, off);
    }
    if ((tid & 31) == 0) {
        atomicAdd(&s_sumw, ws);
    }
    __syncthreads();
    if (tid == 0) {
        // every cell in [B0*r, min(B1*r, n_kv)) belongs to one of this range's blocks; those not
        // in `wvis` are invisible and carry the -inf key
        const int range_cells = max(0, min(B1 * r, n_kv) - B0 * r);
        const int inv = range_cells - s_sumw;
        if (inv > 0 && (key_inf & state.prefix_mask) == state.prefix) {
            atomicAdd(&histogram[(key_inf >> shift) & (NBINS - 1)], inv);
        }
    }
    __syncthreads();
    const size_t off = ((size_t) row * blocks_per_row + hb) * NBINS;
    block_histograms[off + tid] = histogram[tid];
}

template<typename kv_t>
static void indexer_topk_radix_cuda(
        ggml_cuda_pool & pool,
        const float * score, const int * cell_blk, const kv_t * additive,
        const indexer_topk_extra extra,
        int * dst, int ncols, int nrows, int n_tps, int n_blocks, int n_kv, int k,
        bool grouped,
        cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS = 1 << RADIX_BITS;
    const int blocks_per_row = std::min((ncols + 1023) / 1024, 8);

    ggml_cuda_pool_alloc<indexer_topk_radix_state> states_alloc(pool, nrows);
    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) nrows * blocks_per_row * NBINS);
    ggml_cuda_pool_alloc<int> gc_alloc(pool, (size_t) nrows * blocks_per_row);
    ggml_cuda_pool_alloc<int> ec_alloc(pool, (size_t) nrows * blocks_per_row);
    ggml_cuda_pool_alloc<int> gb_alloc(pool, (size_t) nrows * blocks_per_row);
    ggml_cuda_pool_alloc<int> eb_alloc(pool, (size_t) nrows * blocks_per_row);
    indexer_topk_radix_state * states = states_alloc.get();
    int * histograms = histograms_alloc.get();
    int * g_cnt  = gc_alloc.get();
    int * e_cnt  = ec_alloc.get();
    int * g_base = gb_alloc.get();
    int * e_base = eb_alloc.get();

    indexer_topk_radix_init<<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows, k);
    (void) cudaMemsetAsync(g_cnt, 0, (size_t) nrows * blocks_per_row * sizeof(int), stream);

    const dim3 row_grid(blocks_per_row * nrows);
    for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
        if (grouped) {
            indexer_topk_radix_histogram_grouped<BLOCK_SIZE, RADIX_BITS, 4>
                <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                    score, cell_blk, extra, states, histograms,
                    ncols, n_tps, n_blocks, n_kv, blocks_per_row, shift);
        } else {
            indexer_topk_radix_histogram<BLOCK_SIZE, RADIX_BITS, kv_t>
                <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                    score, cell_blk, additive, extra, states, histograms,
                    ncols, n_tps, n_blocks, n_kv, blocks_per_row, shift);
        }
        indexer_topk_radix_select<BLOCK_SIZE, RADIX_BITS>
            <<<nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift);
        indexer_topk_hist_accum<RADIX_BITS>
            <<<(size_t) nrows * blocks_per_row, NBINS, 0, stream>>>(
                histograms, states, g_cnt, e_cnt, nrows, blocks_per_row, shift);
    }

    // deterministic gather: ascending column order, no atomics (see kernels above)
    if (k <= 0 || nrows == 0) {
        return;
    }
    indexer_topk_base_scan<<<nrows, BLOCK_SIZE, 0, stream>>>(
            g_cnt, e_cnt, g_base, e_base, nrows, blocks_per_row);
    if (grouped) {
        indexer_topk_write_blocks_grouped<4><<<(size_t) blocks_per_row * nrows, BLOCK_SIZE, 0, stream>>>(
                score, cell_blk, extra, states, g_base, e_base, dst,
                ncols, n_tps, n_blocks, n_kv, k, blocks_per_row, 0);
    } else {
        indexer_topk_write_blocks<kv_t><<<(size_t) blocks_per_row * nrows, BLOCK_SIZE, 0, stream>>>(
                score, cell_blk, additive, extra, states, g_base, e_base, dst,
                ncols, n_tps, n_blocks, n_kv, k, blocks_per_row);
    }
}

// Exact per-range greater/equal counts for the block path's gather (issue #89).  The block-level radix
// passes bin a block into the hist-block that owns its BLOCK range, but the gather partitions by CELL range;
// the two agree only when block b's cells are [b*r, b*r+r).  A unified KV holding several sequences keys
// blocks by (sequence, position bucket), so a block's cells can sit anywhere, and the histogram-derived
// g_cnt/e_cnt then give the gather wrong bases (unwritten and overwritten output entries).  The per-row
// radix totals do not depend on the partition and stay correct; this recounts the per-range counts over
// exactly the cells the gather walks, with the same key logic.
template<int VEC>
static __global__ void indexer_topk_count_cells_grouped(
        const float * __restrict__ score,
        const int   * __restrict__ cell_blk,
        const indexer_topk_extra extra,
        const indexer_topk_radix_state * __restrict__ states,
        int * __restrict__ g_cnt, int * __restrict__ e_cnt,
        int ncols, int n_tps, int n_blocks, int n_kv,
        int blocks_per_row, int chunk) {
    constexpr int BLOCK_SIZE = 256;
    const int row = blockIdx.x / blocks_per_row;
    const int b   = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const int t = row % n_tps;
    const int s = row / n_tps;

    const int col0 = b * chunk;
    const int col1 = min(col0 + chunk, ncols);
    const uint32_t prefix  = states[row].prefix;
    const uint32_t key_inf = indexer_topk_float_to_ordered(-INFINITY);
    const int q    = extra.q_pos != nullptr ? extra.q_pos[t + s*n_tps] : 0;
    const int nskv = s*n_kv;

    int lg = 0, le = 0;
    for (int c0 = col0 + tid*VEC; c0 < col1; c0 += BLOCK_SIZE*VEC) {
        int cur_b = -1;
        uint32_t cur_key = key_inf;
        #pragma unroll
        for (int v = 0; v < VEC; ++v) {
            const int c = c0 + v;
            if (c >= col1) {
                break;
            }
            const int bb = cell_blk[c + nskv];
            if (bb != cur_b) {
                cur_b = bb;
                cur_key = indexer_topk_block_key(score, extra, bb, t, s, n_blocks, n_tps);
            }
            uint32_t key = cur_key;
            if (extra.cell_pos != nullptr) {
                const int cp = extra.cell_pos[c + nskv];
                if (cp < 0 || cp > q) {
                    key = key_inf;
                }
            }
            lg += key >  prefix ? 1 : 0;
            le += key == prefix ? 1 : 0;
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        lg += __shfl_down(lg, off);
        le += __shfl_down(le, off);
    }
    __shared__ int swg[BLOCK_SIZE / 32], swe[BLOCK_SIZE / 32];
    if ((tid & 31) == 0) {
        swg[tid >> 5] = lg;
        swe[tid >> 5] = le;
    }
    __syncthreads();
    if (tid == 0) {
        int tg = 0, te = 0;
        for (int w = 0; w < BLOCK_SIZE / 32; ++w) {
            tg += swg[w];
            te += swe[w];
        }
        g_cnt[blockIdx.x] = tg;
        e_cnt[blockIdx.x] = te;
    }
}

static void indexer_topk_radix_cuda_blocks(
        ggml_cuda_pool & pool,
        const float * score, const int * cell_blk,
        const indexer_topk_extra extra,
        int * dst, int ncols, int nrows, int n_tps, int n_blocks, int n_kv, int k,
        int r, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS = 1 << RADIX_BITS;
    const int blocks_per_row = std::min((n_blocks + 1023) / 1024, 8);
    const int bchunk = (n_blocks + blocks_per_row - 1) / blocks_per_row;

    // [QSA_SCORE_BOUNDS] a trimmed score holds only the visible prefix, so cells of trimmed blocks
    // (cell_blk[c] >= n_blocks) must not be read.  Under the bound's single-sequence precondition
    // the cache is contiguous, so those cells are exactly c >= n_blocks*r; clamping the cell range
    // keeps pass1, the block passes and the gather inside the valid prefix.  When nothing is
    // trimmed n_blocks*r >= n_kv, so this is the identity.
    const int n_cells = std::min(n_kv, n_blocks*r);

    ggml_cuda_pool_alloc<indexer_topk_radix_state> states_alloc(pool, nrows);
    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) nrows * blocks_per_row * NBINS);
    ggml_cuda_pool_alloc<int> wvis_alloc(pool, (size_t) nrows * n_blocks);
    ggml_cuda_pool_alloc<int> gc_alloc(pool, (size_t) nrows * blocks_per_row);
    ggml_cuda_pool_alloc<int> ec_alloc(pool, (size_t) nrows * blocks_per_row);
    ggml_cuda_pool_alloc<int> gb_alloc(pool, (size_t) nrows * blocks_per_row);
    ggml_cuda_pool_alloc<int> eb_alloc(pool, (size_t) nrows * blocks_per_row);
    indexer_topk_radix_state * states = states_alloc.get();
    int * histograms = histograms_alloc.get();
    int * wvis = wvis_alloc.get();
    int * g_cnt  = gc_alloc.get();
    int * e_cnt  = ec_alloc.get();
    int * g_base = gb_alloc.get();
    int * e_base = eb_alloc.get();

    indexer_topk_radix_init<<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows, k);
    (void) cudaMemsetAsync(wvis,  0, (size_t) nrows * n_blocks * sizeof(int), stream);
    (void) cudaMemsetAsync(g_cnt, 0, (size_t) nrows * blocks_per_row * sizeof(int), stream);

    const dim3 row_grid((size_t) blocks_per_row * nrows);
    for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
        if (shift == 32 - RADIX_BITS) {
            indexer_topk_histogram_pass1<BLOCK_SIZE, RADIX_BITS, 4>
                <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                    score, cell_blk, extra, states, histograms, wvis,
                    n_cells, n_tps, n_blocks, n_cells, blocks_per_row, bchunk, r, shift);
        } else {
            indexer_topk_histogram_blocks<BLOCK_SIZE, RADIX_BITS>
                <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                    score, extra, states, wvis, histograms,
                    n_cells, n_tps, n_blocks, n_cells, blocks_per_row, bchunk, r, shift);
        }
        indexer_topk_radix_select<BLOCK_SIZE, RADIX_BITS>
            <<<nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift);
        indexer_topk_hist_accum<RADIX_BITS>
            <<<(size_t) nrows * blocks_per_row, NBINS, 0, stream>>>(
                histograms, states, g_cnt, e_cnt, nrows, blocks_per_row, shift);
    }

    if (k <= 0 || nrows == 0) {
        return;
    }
    // the histogram-derived per-range counts follow block ranges; recount them over the gather's cell ranges
    indexer_topk_count_cells_grouped<4><<<(size_t) blocks_per_row * nrows, BLOCK_SIZE, 0, stream>>>(
            score, cell_blk, extra, states, g_cnt, e_cnt,
            n_cells, n_tps, n_blocks, n_cells, blocks_per_row, bchunk * r);
    indexer_topk_base_scan<<<nrows, BLOCK_SIZE, 0, stream>>>(
            g_cnt, e_cnt, g_base, e_base, nrows, blocks_per_row);
    indexer_topk_write_blocks_grouped<4><<<(size_t) blocks_per_row * nrows, BLOCK_SIZE, 0, stream>>>(
            score, cell_blk, extra, states, g_base, e_base, dst,
            n_cells, n_tps, n_blocks, n_cells, k, blocks_per_row, bchunk * r);
}

void ggml_cuda_indexer_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * score    = dst->src[0];
    const ggml_tensor * cell_blk = dst->src[1];
    const ggml_tensor * additive = dst->src[2];
    const ggml_tensor * cell_pos = dst->src[3];
    const ggml_tensor * q_pos    = dst->src[4];
    const ggml_tensor * blk_idx  = dst->src[5];
    const ggml_tensor * blk_tail = dst->src[6];
    const ggml_tensor * blk_cells = dst->src[7];
    const float * score_d    = (const float *) score->data;
    const int   * cell_blk_d = (const int  *) cell_blk->data;
    int *         dst_d      = (int *) dst->data;
    cudaStream_t  stream     = ctx.stream();

    const indexer_topk_extra extra = {
        cell_pos != nullptr ? (const int *) cell_pos->data : nullptr,
        q_pos    != nullptr ? (const int *) q_pos->data    : nullptr,
        blk_idx  != nullptr ? (const int *) blk_idx->data  : nullptr,
        blk_tail != nullptr ? (const int *) blk_tail->data : nullptr,
    };

    GGML_ASSERT(score->type == GGML_TYPE_F32);
    GGML_ASSERT(cell_blk->type == GGML_TYPE_I32);
    GGML_ASSERT(additive == nullptr || additive->type == GGML_TYPE_F16 || additive->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(score));
    GGML_ASSERT(ggml_is_contiguous(cell_blk));
    GGML_ASSERT(additive == nullptr || ggml_is_contiguous(additive));
    GGML_ASSERT(cell_pos == nullptr || (ggml_is_contiguous(cell_pos) && ggml_is_contiguous(q_pos)));
    GGML_ASSERT(blk_idx  == nullptr || (ggml_is_contiguous(blk_idx)  && ggml_is_contiguous(blk_tail)));
    GGML_ASSERT(blk_cells == nullptr || blk_cells->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int n_blocks = score->ne[0];
    const int n_tps    = score->ne[1];
    const int n_stream = score->ne[2];
    const int n_kv     = cell_blk->ne[0];
    const int nrows    = n_tps * n_stream;
    const int k        = dst->ne[0];
    ggml_cuda_pool & pool = ctx.pool();

    if (additive == nullptr || additive->type == GGML_TYPE_F16) {
        // the additive is null on the default derived path: the value is per-block, so the
        // grouped histogram can share the key across a block's cells (A/B: LLAMA_INDEXER_NOGROUP)
        static const bool nogroup = []() { return getenv("LLAMA_INDEXER_NOGROUP") != nullptr; }();
        static const bool noblock = []() { return getenv("LLAMA_INDEXER_NOBLOCK") != nullptr; }();
        const int r = blk_cells != nullptr ? (int) (blk_cells->ne[0] / n_blocks) : 0;
        const bool block_path = additive == nullptr && !noblock && !nogroup &&
                blk_cells != nullptr && blk_tail != nullptr && blk_idx != nullptr &&
                cell_pos != nullptr && r > 0 && r * n_blocks == (int) blk_cells->ne[0];
        if (block_path) {
            indexer_topk_radix_cuda_blocks(pool, score_d, cell_blk_d, extra,
                    dst_d, n_kv, nrows, n_tps, n_blocks, n_kv, k, r, stream);
        } else {
            const bool grouped = additive == nullptr && !nogroup;
            indexer_topk_radix_cuda(pool, score_d, cell_blk_d,
                    additive != nullptr ? (const half *) additive->data : nullptr, extra,
                    dst_d, n_kv, nrows, n_tps, n_blocks, n_kv, k, grouped, stream);
        }
    } else {
        indexer_topk_radix_cuda(pool, score_d, cell_blk_d, (const float *) additive->data, extra,
                dst_d, n_kv, nrows, n_tps, n_blocks, n_kv, k, false, stream);
    }
}

bool ggml_cuda_indexer_top_k_supported(int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);

    const ggml_tensor * score    = dst->src[0];
    const ggml_tensor * cell_blk = dst->src[1];
    const ggml_tensor * additive = dst->src[2];

    // the additive is optional: the compact position/bias srcs derive it in-kernel
    return score->type == GGML_TYPE_F32 &&
        cell_blk->type == GGML_TYPE_I32 &&
        (additive == nullptr || additive->type == GGML_TYPE_F16 || additive->type == GGML_TYPE_F32) &&
        (additive != nullptr || dst->src[3] != nullptr || dst->src[5] != nullptr) &&
        dst->type == GGML_TYPE_I32;
}
