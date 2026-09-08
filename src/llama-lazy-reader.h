#pragma once

#include "ggml.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

#if defined(__unix__) || defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__) || defined(__HAIKU__)
#define LLAMA_LAZY_READER_POSIX 1
#else
#define LLAMA_LAZY_READER_POSIX 0
#endif

// Managed on-demand reader for a single "lazy" tensor: rows live on disk in a
// file, a fixed-size host arena caches a subset, and eviction is a clock LRU.
// All access goes through gather(), which is internally synchronized so that
// several llama_contexts sharing one model can call it concurrently.
//
// The cache stores the on-disk quantized bytes (not dequantized values), so
// the arena holds N * row_bytes worth of rows for any supported type. Rows are
// contiguous in the file: row i starts at data_offs + i * row_bytes (GGUF data
// is tight-packed, no padding). Bookkeeping is page-granular: pages of
// page_size bytes are the unit of I/O and eviction.
class llama_lazy_reader {
public:
    struct config {
        int          fd;         // owned fd (dup of the model file), closed in dtor
        size_t       data_offs;  // absolute file offset of the tensor data
        size_t       row_bytes;  // tight-packed on-disk bytes per row (NOT ggml_row_size: no padding)
        uint64_t     n_rows;     // number of rows in the table
        enum ggml_type type;     // source quantization (e.g. GGML_TYPE_Q8_0)
        int64_t      row_nelems; // elements per row (ne[0])
        size_t       budget;     // managed buffer size in bytes (> 0)
        size_t       page_size = 4096;
        int          n_threads = 4; // I/O threads for cold-page fetches (env LLAMA_LAZY_IO_THREADS overrides)
    };

    llama_lazy_reader(const config & cfg);
    ~llama_lazy_reader();

    llama_lazy_reader(const llama_lazy_reader &) = delete;
    llama_lazy_reader & operator=(const llama_lazy_reader &) = delete;

    // Dequantize rows[] (n entries) into dst, laid out as dst[j*row_nelems + d].
    // Loads missing pages with pread(), evicts with the clock LRU, all under
    // one lock hold so no page can disappear mid-gather.
    void gather(const int32_t * rows, size_t n, float * dst);

    uint64_t n_pages_total() const { return n_pages; }
    size_t   budget()         const { return budget_bytes; }

    // counters
    uint64_t hits()        const { return n_hits.load(); }
    uint64_t misses()      const { return n_misses.load(); }
    uint64_t bytes_read()  const { return n_bytes_read.load(); }

private:
    void ensure_page(uint32_t page);                 // called with mtx held
    size_t evict_one();                              // called with mtx held (clock scan)
    void fetch_misses(const std::vector<uint32_t> & misses); // called with mtx held

    // config
    const int               fd;
    const size_t            data_offs;
    const size_t            row_bytes;
    const uint64_t          n_rows;
    const ggml_to_float_t   to_float;
    const int64_t           row_nelems;
    const size_t            page_size;
    const size_t            n_pages;       // ceil(n_rows / rows_per_page)
    const size_t            rows_per_page; // floor(page_size / row_bytes), >= 1
    const size_t            n_slots;       // max(1, budget / page_size)
    const size_t            budget_bytes;
    int                     n_threads;     // I/O threads, >= 1 (env LLAMA_LAZY_IO_THREADS overrides)

    // state (guarded by mtx)
    std::mutex             mtx;
    std::vector<int32_t>   page_to_slot;   // n_pages, -1 = not resident
    std::vector<uint8_t>   slot_ref;       // n_slots, second-chance bit
    std::vector<int32_t>   slot_page;      // n_slots, page id in the slot
    std::vector<uint32_t>  pages_scratch;  // per-gather dedup buffer
    std::vector<uint8_t>   miss_buf;       // per-gather cold-page read buffer (n_misses * page_size)
    uint8_t *              arena;          // n_slots * page_size
    size_t                 clock = 0;

    // stats
    std::atomic<uint64_t>  n_hits{0}, n_misses{0}, n_bytes_read{0};
};
