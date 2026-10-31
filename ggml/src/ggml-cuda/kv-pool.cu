#include "kv-pool.cuh"
#include "common.cuh"

// wip/kv-residency (TODO #118) B-A step 2 -- the attendance pool for the attention KV.
//
// WHY THIS OP EXISTS
//   The QSA sparse decode attends ~2051 cells per layer per step, scattered over the whole prefix
//   (`wip/kv-residency/BA-POOL-DESIGN.md` §1).  Block 15's window can only keep a positional tail, and
//   its 4 MiB map unit is 3855 contiguous cells -- a policy at that granularity hits 6 % and churns
//   601 units/step.  What the attended set needs is a COMPACT pool of pages with a page table, i.e.
//   Strata's shape: `table[block] -> slot`, the resident set chosen by attendance.
//
// THE WRITER RULE (this is what makes it exact with no invalidation)
//   The MIRROR is authoritative: every KV write lands there (the graph's `set_rows` targets the
//   mirror), and the pool only ever holds copies.  So:
//     * an eviction costs nothing (the mirror still has the page),
//     * a page a batch just WROTE is re-filled from the mirror here (a resident page would otherwise
//       hold the previous cell's bytes -- `wids`),
//     * a page that is read but not resident is claimed and filled from the mirror BEFORE the
//       attention runs (this op is ordered before it on the same stream).
//   A page therefore never needs to be invalidated -- not after a rejected draft, not on a resume,
//   not on `seq_rm`/`seq_cp`.  The pool can only change WHERE the bytes are, never WHICH.
//
// THE PHASES, and why they are separate kernels
//   1 MARK   every page this launch's selection already holds resident (`use[slot] = epoch`) and list the
//            pages the batch just wrote (they are marked too, so no claim can take their slot).
//   2 CLAIM  one slot per MISSING BLOCK -- deduplicated here, because two properties have to hold at once
//            and a single pass over the ids cannot give both: a slot may only be taken from a page this
//            launch does not use, and the number of slots handed out in one launch must stay under
//            `n_pages` or the round-robin cursor wraps and two blocks end up in one slot.  Only the
//            thread that wins the table CAS appends to the block list, so the claims count DISTINCT
//            blocks (the first cut consumed a cursor value per id attempt, wrapped on a 4-row batch, and
//            left a just-resolved block unresident -- which is what the text divergence was).
//   3 MAP    write each selected cell's pool position into the reader's K/V index list.
//   4 FILL   mirror page -> pool page for every claimed and every dirty page, K and V (they share slots).
//   The epoch is a device counter incremented once per launch (in FILL), so nothing on the host is
//   involved and a captured graph replays correctly.

static __global__ void kv_pool_mark_kernel(
        const int     * __restrict__ ids,
        const int       n_ids,
        const int64_t * __restrict__ wids,
        const int       n_wids,
        int           * __restrict__ tbl,
        int           * __restrict__ ctr,
        int           * __restrict__ use,
        int           * __restrict__ miss,
        const int       miss_cap,
        const int       page_cells,
        const int       n_pages,
        const int       n_cells) {
    const int epoch = ctr[6];

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        ctr[1] = 0;
        ctr[8] = 0;
    }

    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n_ids) {
        const int c = ids[i];
        if (c < 0 || c >= n_cells) {
            atomicAdd(&ctr[7], 1);
        } else {
            const int s = tbl[c / page_cells];
            if (s >= 0 && s < n_pages) {
                use[s] = epoch;
            }
        }
    }

    for (int j = i; j < n_wids; j += blockDim.x*gridDim.x) {
        const int c = (int) wids[j];
        if (c < 0 || c >= n_cells) {
            continue;
        }
        const int b = c / page_cells;
        const int s = tbl[b];
        if (s >= 0 && s < n_pages) {
            use[s] = epoch;                       // protected from eviction this launch
            const int m = atomicAdd(&ctr[1], 1);
            if (m < miss_cap) {
                miss[m] = b;                      // and re-filled from the mirror (the new cells' bytes)
                atomicAdd(&ctr[5], 1);
            }
        }
    }
}

// Phase 2: one claim per missing BLOCK.  The table CAS is the dedupe: a thread that finds -1 becomes the
// block's leader (and only leaders consume a cursor value); the others see -2 (or the winner's slot).
static __global__ void kv_pool_claim_kernel(
        const int     * __restrict__ ids,
        const int       n_ids,
        int           * __restrict__ tbl,
        int           * __restrict__ inv,
        int           * __restrict__ ctr,
        int           * __restrict__ use,
        int           * __restrict__ miss,
        const int       miss_cap,
        const int       n_pages,
        const int       page_cells,
        const int       n_cells) {
    const int epoch = ctr[6];

    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_ids) {
        return;
    }
    const int cell = ids[i];
    if (cell < 0 || cell >= n_cells) {
        return;
    }
    const int blk = cell / page_cells;

    if (tbl[blk] >= 0) {
        atomicAdd(&ctr[4], 1);                    // already resident: a per-id hit
        return;
    }

    const int old = atomicCAS(&tbl[blk], -1, -2);
    if (old == -1 && tbl[blk] == -2) {
        // this thread is the block's leader: take a slot.  A slot phase 1 marked is a page THIS launch
        // reads, so it must not be evicted; the cursor is monotone, so the claims of one launch take
        // distinct slots as long as it hands out fewer than n_pages of them (the caller sizes the pool
        // for one launch's DISTINCT block count, and this is where that count is realised).
        int cand = 0;
        for (int tries = 0; ; ++tries) {
            cand = atomicAdd(&ctr[0], 1) % n_pages;
            if (use[cand] != epoch || tries >= n_pages) {
                break;
            }
        }
        const int prev = atomicExch(&inv[cand], blk);
        if (prev >= 0 && prev != blk) {
            atomicCAS(&tbl[prev], cand, -1);
        }
        tbl[blk] = cand;
        const int m = atomicAdd(&ctr[1], 1);
        if (m < miss_cap) {
            miss[m] = blk;
            atomicAdd(&ctr[2], 1);
        }
    }
}

// Phase 3 + 4: map every selected cell to its slot, then fill what phase 1/2 listed (mirror page -> pool
// page, K and V).
static __global__ void kv_pool_map_fill_kernel(
        const int      * __restrict__ ids,
        const int        n_ids,
        int            * __restrict__ ctr,
        int            * __restrict__ kv_idx,
        const int      * __restrict__ tbl,
        const int      * __restrict__ miss,
        const int        miss_cap,
        const char     * __restrict__ mk, const size_t mk_row, char * __restrict__ pk, const size_t pk_row,
        const char     * __restrict__ mv, const size_t mv_row, char * __restrict__ pv, const size_t pv_row,
        const int        page_cells,
        const int        n_blocks,
        const int        n_pages,
        const int        n_cells) {
    const size_t tid     = (size_t) blockIdx.x*blockDim.x + threadIdx.x;
    const size_t nthread = (size_t) gridDim.x*blockDim.x;

    // map
    for (int i = (int) tid; i < n_ids; i += (int) nthread) {
        const int cell = ids[i];
        if (cell < 0 || cell >= n_cells) {
            kv_idx[i] = 0;
            continue;
        }
        const int blk = cell / page_cells;
        const int s   = tbl[blk];
        kv_idx[i] = (s >= 0 && s < n_pages) ? s*page_cells + (cell - blk*page_cells) : 0;
        atomicAdd(&ctr[3], 1);
    }

    // fill
    const int n_miss = min(ctr[1], miss_cap);
    if (n_miss > 0) {
        const size_t k_words = (size_t) page_cells*mk_row / 4;
        const size_t v_words = (size_t) page_cells*mv_row / 4;

        for (size_t t = tid; t < (size_t) n_miss*(k_words + v_words); t += nthread) {
            const bool   is_v = t >= (size_t) n_miss*k_words;
            const size_t u    = is_v ? t - (size_t) n_miss*k_words : t;
            const size_t wrow = is_v ? v_words : k_words;
            const int    p    = (int) (u / wrow);
            const size_t o    = u % wrow;
            const int    b    = miss[p];
            if (b < 0 || b >= n_blocks) {
                continue;
            }
            const int s = tbl[b];
            if (s < 0 || s >= n_pages) {
                continue;
            }
            if (is_v) {
                const uint32_t * src = (const uint32_t *) (mv + (size_t) b*page_cells*mv_row);
                uint32_t       * dst = (uint32_t       *) (pv + (size_t) s*page_cells*pv_row);
                dst[o] = src[o];
            } else {
                const uint32_t * src = (const uint32_t *) (mk + (size_t) b*page_cells*mk_row);
                uint32_t       * dst = (uint32_t       *) (pk + (size_t) s*page_cells*pk_row);
                dst[o] = src[o];
            }
        }
    }

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        ctr[6] += 1;
    }
}

// DIAGNOSTIC (LLAMA_KV_POOL_VERIFY=1): every selected cell's POOL page must equal its MIRROR page, byte for
// byte, at the moment the attention is about to read it.  Mismatches land in ctr[7] (reported by the
// cache's stats line).  It separates "the pool holds the wrong bytes" from "the reader looks in the wrong
// place"; never a timing arm.
static __global__ void kv_pool_verify_kernel(
        const int      * __restrict__ ids,
        const int        n_ids,
        int            * __restrict__ ctr,
        const int      * __restrict__ tbl,
        const char     * __restrict__ mk, const size_t mk_row, const char * __restrict__ pk, const size_t pk_row,
        const char     * __restrict__ mv, const size_t mv_row, const char * __restrict__ pv, const size_t pv_row,
        const int        page_cells,
        const int        n_pages,
        const int        n_cells) {
    const size_t k_words = (size_t) page_cells*mk_row / 4;
    const size_t v_words = (size_t) page_cells*mv_row / 4;
    const size_t nthread = (size_t) gridDim.x*blockDim.x;
    const size_t tid     = (size_t) blockIdx.x*blockDim.x + threadIdx.x;

    for (size_t t = tid; t < (size_t) n_ids*(k_words + v_words); t += nthread) {
        const bool   is_v = t >= (size_t) n_ids*k_words;
        const size_t u    = is_v ? t - (size_t) n_ids*k_words : t;
        const size_t wrow = is_v ? v_words : k_words;
        const int    i    = (int) (u / wrow);
        const size_t o    = u % wrow;

        const int c = ids[i];
        if (c < 0 || c >= n_cells) {
            continue;
        }
        const int b = c / page_cells;
        const int s = tbl[b];
        if (s < 0 || s >= n_pages) {
            atomicAdd(&ctr[7], 1);
            continue;
        }
        uint32_t av = 0, dv = 0;
        if (is_v) {
            av = ((const uint32_t *) (mv + (size_t) b*page_cells*mv_row))[o];
            dv = ((const uint32_t *) (pv + (size_t) s*page_cells*pv_row))[o];
        } else {
            av = ((const uint32_t *) (mk + (size_t) b*page_cells*mk_row))[o];
            dv = ((const uint32_t *) (pk + (size_t) s*page_cells*pk_row))[o];
        }
        if (av != dv) {
            const int n = atomicAdd(&ctr[7], 1);
            if (n < 8) {
                printf("KVPOOL_BAD n=%d %s cell=%d blk=%d slot=%d word=%zu mirror=%08x pool=%08x i=%d\n",
                        n, is_v ? "V" : "K", c, b, s, o, av, dv, i);
            }
        }
    }
}

void ggml_cuda_kv_pool(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * ids      = dst->src[0];
    const ggml_tensor * tbl      = dst->src[1];
    const ggml_tensor * inv      = dst->src[2];
    const ggml_tensor * ctr      = dst->src[3];
    const ggml_tensor * mirror_k = dst->src[4];
    const ggml_tensor * mirror_v = dst->src[5];
    const ggml_tensor * pool_k   = dst->src[6];
    const ggml_tensor * pool_v   = dst->src[7];
    const ggml_tensor * wids     = dst->src[8];
    const ggml_tensor * use_t    = dst->src[9];

    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ggml_is_contiguous(ids));
    GGML_ASSERT(dst->type == GGML_TYPE_I32 && ggml_is_contiguous(dst));
    GGML_ASSERT(tbl->type == GGML_TYPE_I32 && inv->type == GGML_TYPE_I32 && ctr->type == GGML_TYPE_I32);
    GGML_ASSERT(mirror_k->type == mirror_v->type && pool_k->type == pool_v->type);
    GGML_ASSERT(mirror_k->type == pool_k->type);
    GGML_ASSERT(wids == nullptr || wids->type == GGML_TYPE_I64);
    GGML_ASSERT(ctr->ne[0] >= 12);

    const int page_cells = ggml_get_op_params_i32(dst, 0);
    GGML_ASSERT(page_cells > 0);

    const int n_ids    = (int) ggml_nelements(ids);
    const int n_pages  = (int) inv->ne[0];
    const int n_blocks = (int) tbl->ne[0];
    const int n_wids   = wids != nullptr ? (int) ggml_nelements(wids) : 0;
    // the cell range the mirror actually holds (the cache's used cells are always inside it)
    const int n_cells  = (int) std::min<int64_t>((int64_t) n_blocks*page_cells, mirror_k->ne[1]);

    GGML_ASSERT(n_ids > 0 && n_pages > 0 && n_blocks > 0);
    GGML_ASSERT((int64_t) pool_k->ne[1] >= (int64_t) n_pages*page_cells);
    GGML_ASSERT((int64_t) mirror_k->ne[1] >= (int64_t) n_blocks*page_cells);
    // the fill copies 4-byte words
    GGML_ASSERT(mirror_k->nb[1] % 4 == 0 && pool_k->nb[1] % 4 == 0);
    GGML_ASSERT(mirror_v->nb[1] % 4 == 0 && pool_v->nb[1] % 4 == 0);

    cudaStream_t stream = ctx.stream();

    // NO transient scratch allocation here: a captured graph bakes the kernel arguments, so anything this op
    // allocates from the device pool and frees at its end can be handed to another op and then written by a
    // REPLAY of this graph.  That use-after-free produced exactly the "values verify clean, text differs run
    // to run" symptom.  The miss list therefore lives at the end of the counters tensor (ctr[12..]).
    const int miss_cap = n_pages + 256;
    GGML_ASSERT(ctr->ne[0] >= 12 + miss_cap);

    int * ctr_d = (int *) ctr->data;
    int * use   = (int *) use_t->data;
    int * miss  = ctr_d + 12;

    const int nthread = 256;
    const int nblock  = std::max(1, (n_ids + nthread - 1)/nthread);

    kv_pool_mark_kernel<<<nblock, nthread, 0, stream>>>(
            (const int *) ids->data, n_ids,
            wids != nullptr ? (const int64_t *) wids->data : nullptr, n_wids,
            (int *) tbl->data, ctr_d, use, miss, miss_cap, page_cells, n_pages, n_cells);

    kv_pool_claim_kernel<<<nblock, nthread, 0, stream>>>(
            (const int *) ids->data, n_ids,
            (int *) tbl->data, (int *) inv->data, ctr_d, use, miss, miss_cap,
            n_pages, page_cells, n_cells);

    kv_pool_map_fill_kernel<<<64, nthread, 0, stream>>>(
            (const int *) ids->data, n_ids, ctr_d, (int *) dst->data, (const int *) tbl->data,
            miss, miss_cap,
            (const char *) mirror_k->data, mirror_k->nb[1], (char *) pool_k->data, pool_k->nb[1],
            (const char *) mirror_v->data, mirror_v->nb[1], (char *) pool_v->data, pool_v->nb[1],
            page_cells, n_blocks, n_pages, n_cells);

    static const bool kv_pool_verify = [] {
        const char * e = getenv("LLAMA_KV_POOL_VERIFY");
        return e != nullptr && e[0] != '\0';
    }();
    if (kv_pool_verify) {
        kv_pool_verify_kernel<<<64, nthread, 0, stream>>>(
                (const int *) ids->data, n_ids, ctr_d, (const int *) tbl->data,
                (const char *) mirror_k->data, mirror_k->nb[1], (const char *) pool_k->data, pool_k->nb[1],
                (const char *) mirror_v->data, mirror_v->nb[1], (const char *) pool_v->data, pool_v->nb[1],
                page_cells, n_pages, n_cells);
    }

    CUDA_CHECK(cudaGetLastError());
}
