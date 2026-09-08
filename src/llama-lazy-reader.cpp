#include "llama-lazy-reader.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

#if LLAMA_LAZY_READER_POSIX
#include <cerrno>
#include <cstring>
#include <thread>
#include <unistd.h>
#endif

#if defined(__linux__) && LLAMA_LAZY_READER_POSIX
#include <fcntl.h>
#endif

namespace {

size_t lazy_page_size(size_t row_bytes, size_t cfg_page_size) {
    if (cfg_page_size == 0) {
        cfg_page_size = 4096;
    }
    if (cfg_page_size / row_bytes == 0) {
        // a row does not fit in the requested page size: bump to a multiple of
        // 4096 that holds a whole row
        return ((row_bytes + 4095) / 4096) * 4096;
    }
    return cfg_page_size;
}

size_t lazy_rows_per_page(size_t page_size, size_t row_bytes) {
    const size_t rpp = page_size / row_bytes;
    return rpp > 0 ? rpp : 1;
}

size_t lazy_n_pages(uint64_t n_rows, size_t rows_per_page) {
    return size_t((n_rows + rows_per_page - 1) / rows_per_page);
}

} // namespace

llama_lazy_reader::llama_lazy_reader(const config & cfg) :
    fd(cfg.fd),
    data_offs(cfg.data_offs),
    row_bytes(cfg.row_bytes),
    n_rows(cfg.n_rows),
    to_float(ggml_get_type_traits(cfg.type) ? ggml_get_type_traits(cfg.type)->to_float : nullptr),
    row_nelems(cfg.row_nelems),
    page_size(lazy_page_size(cfg.row_bytes, cfg.page_size)),
    n_pages(lazy_n_pages(cfg.n_rows, lazy_rows_per_page(page_size, cfg.row_bytes))),
    rows_per_page(lazy_rows_per_page(page_size, cfg.row_bytes)),
    n_slots(cfg.budget / page_size > 0 ? cfg.budget / page_size : 1),
    budget_bytes(n_slots * page_size),
    page_to_slot(n_pages, -1),
    slot_ref(n_slots, 0),
    slot_page(n_slots, -1),
    arena(nullptr) {
    if (fd < 0) {
        GGML_ABORT("%s: invalid fd", __func__);
    }
    if (row_bytes == 0 || n_rows == 0 || row_nelems <= 0 || cfg.budget == 0) {
        GGML_ABORT("%s: invalid config (row_bytes=%zu, n_rows=%llu, row_nelems=%lld, budget=%zu)",
                __func__, row_bytes, (unsigned long long) n_rows, (long long) row_nelems, cfg.budget);
    }
    if (row_nelems % ggml_blck_size(cfg.type) != 0) {
        GGML_ABORT("%s: row_nelems %lld not a multiple of block size %lld",
                __func__, (long long) row_nelems, (long long) ggml_blck_size(cfg.type));
    }
    if (n_slots > INT32_MAX) {
        GGML_ABORT("%s: too many slots (%zu), page_to_slot is int32", __func__, n_slots);
    }
#if !LLAMA_LAZY_READER_POSIX
    GGML_ABORT("%s: pread not available on this platform (managed lazy loading is POSIX-only for now)", __func__);
#endif
    arena = (uint8_t *) malloc(budget_bytes);
    if (arena == nullptr) {
        GGML_ABORT("%s: failed to allocate %zu bytes for the managed buffer", __func__, budget_bytes);
    }

    int thr = cfg.n_threads > 0 ? cfg.n_threads : 1;
    if (const char * env = getenv("LLAMA_LAZY_IO_THREADS")) {
        thr = atoi(env);
    }
    n_threads = thr > 0 ? thr : 1;
}

llama_lazy_reader::~llama_lazy_reader() {
    free(arena);
#if LLAMA_LAZY_READER_POSIX
    close(fd);
#else
    GGML_UNUSED(fd);
#endif
}

size_t llama_lazy_reader::evict_one() {
    // second-chance clock: sweep slots, clear the ref bit on first pass, evict
    // on the second
    for (;;) {
        const size_t slot = clock % n_slots;
        clock++;
        if (slot_ref[slot] == 0) {
            if (slot_page[slot] != -1) {
                page_to_slot[slot_page[slot]] = -1;
            }
            return slot;
        }
        slot_ref[slot] = 0;
    }
}

void llama_lazy_reader::ensure_page(uint32_t page) {
    GGML_ASSERT(page < n_pages);
    int32_t slot = page_to_slot[page];
    if (slot != -1) {
        // already resident: refresh the clock bit
        slot_ref[slot] = 1;
        n_hits++;
        return;
    }
    n_misses++;

    slot = (int32_t) evict_one();
    GGML_ASSERT(slot >= 0 && (size_t) slot < n_slots);

    // clamp the read to the table extent: the last page is partial (the byte
    // offset of the last page can exceed the table end, so compute the range
    // from the row extent, not the byte offset)
    const uint64_t row0 = (uint64_t) page * rows_per_page;
    const uint64_t row1 = std::min(row0 + rows_per_page, n_rows);
    const size_t off = data_offs + (size_t) row0 * row_bytes;
    const size_t want = (size_t) (row1 - row0) * row_bytes; // <= page_size

#if LLAMA_LAZY_READER_POSIX
    uint8_t * dst = arena + (size_t) slot * page_size;
    size_t got = 0;
    while (got < want) {
        const ssize_t r = pread(fd, dst + got, want - got, off + got);
        if (r < 0) {
            GGML_ABORT("%s: pread failed: %s", __func__, strerror(errno));
        }
        if (r == 0) {
            GGML_ABORT("%s: unexpected EOF at offset %zu", __func__, off + got);
        }
        got += (size_t) r;
    }
#else
    GGML_ABORT("%s: pread not available on this platform", __func__);
#endif

    page_to_slot[page] = slot;
    slot_page[slot] = (int32_t) page;
    slot_ref[slot] = 1;
    n_bytes_read += want;
}

void llama_lazy_reader::gather(const int32_t * rows, size_t n, float * dst) {
    std::lock_guard<std::mutex> lock(mtx);

    // pass 1: split the distinct pages this gather needs into resident hits and
    // cold misses. Sort + unique keeps the hit / miss counters per distinct
    // page and avoids re-reading a page that a previous row already needed.
    pages_scratch.clear();
    pages_scratch.reserve(n);
    for (size_t j = 0; j < n; j++) {
        GGML_ASSERT(rows[j] >= 0 && (uint64_t) rows[j] < n_rows);
        pages_scratch.push_back((uint32_t) (rows[j] / rows_per_page));
    }
    std::sort(pages_scratch.begin(), pages_scratch.end());
    pages_scratch.erase(std::unique(pages_scratch.begin(), pages_scratch.end()), pages_scratch.end());

    std::vector<uint32_t> misses;
    misses.reserve(pages_scratch.size());
    for (const uint32_t page : pages_scratch) {
        const int32_t slot = page_to_slot[page];
        if (slot != -1) {
            slot_ref[slot] = 1; // refresh the clock bit
            n_hits++;
        } else {
            misses.push_back(page);
        }
    }

    if (!misses.empty()) {
#if defined(__linux__) && LLAMA_LAZY_READER_POSIX
        // Option A: queue the cold pages with the kernel readahead before any
        // pread runs. The device-level reads then overlap instead of one pread
        // stalling on a 4 KB fault at a time; the preads below join the reads
        // already in flight (no double I/O). Adjacent pages coalesce into one hint.
        {
            bool have = false;
            off_t cur = 0;
            size_t cur_len = 0;
            for (const uint32_t page : misses) {
                const uint64_t row0 = (uint64_t) page * rows_per_page;
                const uint64_t row1 = std::min(row0 + rows_per_page, n_rows);
                const off_t off = (off_t) (data_offs + (size_t) row0 * row_bytes);
                const size_t len = (size_t) (row1 - row0) * row_bytes;
                if (have && off == cur + (off_t) cur_len) {
                    cur_len += len;
                } else {
                    if (have) {
                        posix_fadvise(fd, cur, (off_t) cur_len, POSIX_FADV_WILLNEED);
                    }
                    cur = off;
                    cur_len = len;
                    have = true;
                }
            }
            if (have) {
                posix_fadvise(fd, cur, (off_t) cur_len, POSIX_FADV_WILLNEED);
            }
        }
#endif
        fetch_misses(misses);
    }

    // pass 2: dequantize. Normally all needed pages are resident and cannot
    // be evicted because we still hold the lock (eviction only happens inside
    // the fetch paths above). If the working set of this gather exceeds the
    // budget, a page fetched during pass 1 can be evicted by a later fetch;
    // re-fault it on demand here (identical thrash behavior to the kernel
    // page cache, just bounded by our budget).
    for (size_t j = 0; j < n; j++) {
        const uint32_t row = (uint32_t) rows[j];
        const size_t page = row / rows_per_page;
        if (page_to_slot[page] == -1) {
            ensure_page(page);
        }
        const size_t off  = (row % rows_per_page) * row_bytes;
        const int32_t slot = page_to_slot[page];
        GGML_ASSERT(slot != -1);
        float * dst_row = dst + j * row_nelems;
        if (to_float != nullptr) {
            to_float(arena + (size_t) slot * page_size + off, dst_row, row_nelems);
        } else {
            // non-quantized type (F32): copy the row as-is
            memcpy(dst_row, arena + (size_t) slot * page_size + off, (size_t) row_nelems * sizeof(float));
        }
    }
}

void llama_lazy_reader::fetch_misses(const std::vector<uint32_t> & misses) {
    const size_t M = misses.size();
    if (M == 0) {
        return;
    }

    // small fetches (decode, shallow prefills): keep the direct serial path
    if (n_threads <= 1 || M <= 16) {
        for (const uint32_t page : misses) {
            ensure_page(page);
        }
        return;
    }

    // page extent helper (shared with ensure_page semantics): the last page is
    // partial, so the range is computed from the row extent, not the byte offset
    struct extent { off_t off; size_t len; };
    auto page_extent = [&](uint32_t page) -> extent {
        const uint64_t row0 = (uint64_t) page * rows_per_page;
        const uint64_t row1 = std::min(row0 + rows_per_page, n_rows);
        return { (off_t) (data_offs + (size_t) row0 * row_bytes),
                 (size_t) (row1 - row0) * row_bytes };
    };

    // Option B: read every cold page into a per-gather buffer with a small I/O
    // pool (writes are disjoint, so no locking is needed), then write the pages
    // back into the arena on this thread. The write-back is serial, so the clock
    // LRU can never hand the same slot to two pages mid-flight.
    miss_buf.resize(M * page_size);

    struct item { off_t off; size_t len; uint8_t * dst; };
    std::vector<item> items;
    items.reserve(M);
    for (size_t i = 0; i < M; i++) {
        const extent e = page_extent(misses[i]);
        items.push_back({ e.off, e.len, miss_buf.data() + i * page_size });
    }

    auto read_all = [&](size_t b, size_t e) {
        for (size_t i = b; i < e; i++) {
#if LLAMA_LAZY_READER_POSIX
            size_t got = 0;
            while (got < items[i].len) {
                const ssize_t r = pread(fd, items[i].dst + got, items[i].len - got,
                                        items[i].off + (off_t) got);
                if (r < 0) {
                    GGML_ABORT("%s: pread failed: %s", __func__, strerror(errno));
                }
                if (r == 0) {
                    GGML_ABORT("%s: unexpected EOF at offset %lld", __func__, (long long) (items[i].off + got));
                }
                got += (size_t) r;
            }
#else
            GGML_ABORT("%s: pread not available on this platform", __func__);
#endif
        }
    };

    const size_t n_use  = std::min<size_t>(n_threads, M);
    const size_t chunk  = (M + n_use - 1) / n_use;
    std::vector<std::thread> pool;
    pool.reserve(n_use - 1);
    size_t b = 0;
    for (size_t t = 1; t < n_use && b < M; t++) {
        const size_t e = std::min(b + chunk, M);
        pool.emplace_back([&read_all, b, e] { read_all(b, e); });
        b = e;
    }
    read_all(b, M);
    for (auto & th : pool) {
        th.join();
    }

    n_misses += M;
    for (size_t i = 0; i < M; i++) {
        const uint32_t page = misses[i];
        const extent e = page_extent(page);
        const size_t slot = evict_one();
        GGML_ASSERT(slot >= 0 && slot < n_slots);
        memcpy(arena + slot * page_size, miss_buf.data() + i * page_size, e.len);
        page_to_slot[page] = (int32_t) slot;
        slot_page[slot] = (int32_t) page;
        slot_ref[slot] = 1;
        n_bytes_read += e.len;
    }
}
