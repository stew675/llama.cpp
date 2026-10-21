#include "llama-memory.h"

#include "llama-ext.h"
#include "llama-io.h"
#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-idx.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-recurrent.h"

#include <set>
#include <vector>

llama_memory_status llama_memory_status_combine(llama_memory_status s0, llama_memory_status s1) {
    bool has_update = false;

    switch (s0) {
        case LLAMA_MEMORY_STATUS_SUCCESS:
            {
                has_update = true;
                break;
            }
        case LLAMA_MEMORY_STATUS_NO_UPDATE:
            {
                break;
            }
        case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
        case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return s0;
            }
    }

    switch (s1) {
        case LLAMA_MEMORY_STATUS_SUCCESS:
            {
                has_update = true;
                break;
            }
        case LLAMA_MEMORY_STATUS_NO_UPDATE:
            {
                break;
            }
        case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
        case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return s1;
            }
    }

    // if either status has an update, then the combined status has an update
    return has_update ? LLAMA_MEMORY_STATUS_SUCCESS : LLAMA_MEMORY_STATUS_NO_UPDATE;
}

bool llama_memory_status_is_fail(llama_memory_status status) {
    switch (status) {
        case LLAMA_MEMORY_STATUS_SUCCESS:
        case LLAMA_MEMORY_STATUS_NO_UPDATE:
            {
                return false;
            }
        case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
        case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return true;
            }
    }

    return false;
}

// collects the buffers of the tensors that a state write visits, without keeping the data
struct llama_memory_buffers_writer : public llama_io_write_i {
    std::set<ggml_backend_buffer_t> buffers;
    size_t size = 0;

    void write(const void *, size_t n) override {
        size += n;
    }

    void write_tensor(ggml_tensor * tensor, size_t, size_t n) override {
        buffers.insert(tensor->buffer);
        size += n;
    }

    size_t n_bytes() override {
        return size;
    }
};

std::vector<ggml_backend_buffer_t> llama_memory_get_buffers(const llama_memory_t mem) {
    llama_memory_buffers_writer writer;

    mem->state_write(writer);

    return std::vector<ggml_backend_buffer_t>(writer.buffers.begin(), writer.buffers.end());
}

// the attention caches of a memory: a KV cache, both caches of an iswa cache, or the attention part of a hybrid;
// false for a memory type that parking does not know (a recurrent memory has none: true with an empty list)
// note: a hybrid with a second cache that mirrors the attention cells (the qwen4exp/glm5-next indexer) decides for
//       itself, see the llama_memory_hybrid_idx branch of llama_memory_can_park()/llama_memory_seq_park()
static bool llama_memory_attn_caches(llama_memory_t mem, std::vector<llama_kv_cache *> & kvs) {
    kvs.clear();
    if (auto * kv = dynamic_cast<llama_kv_cache *>(mem)) {
        kvs = { kv };
    } else if (auto * iswa = dynamic_cast<llama_kv_cache_iswa *>(mem)) {
        kvs = { iswa->get_base(), iswa->get_swa() };
    } else if (auto * hyb = dynamic_cast<llama_memory_hybrid *>(mem)) {
        kvs = { hyb->get_mem_attn() };
    } else if (auto * hyb_iswa = dynamic_cast<llama_memory_hybrid_iswa *>(mem)) {
        kvs = { hyb_iswa->get_mem_attn()->get_base(), hyb_iswa->get_mem_attn()->get_swa() };
    } else if (!dynamic_cast<llama_memory_recurrent *>(mem)) {
        return false;
    }
    return true;
}

bool llama_memory_can_park(llama_memory_t mem) {
    // a hybrid whose second cache mirrors the attention cells parks both or not at all
    if (auto * hyb_idx = dynamic_cast<llama_memory_hybrid_idx *>(mem)) {
        return hyb_idx->can_park();
    }

    std::vector<llama_kv_cache *> kvs;
    if (!llama_memory_attn_caches(mem, kvs)) {
        return false;
    }
    for (auto * kv : kvs) {
        if (kv != nullptr && !kv->can_park()) {
            return false;
        }
    }
    return true;
}

bool llama_memory_seq_park(llama_memory_t mem, llama_seq_id seq_id, bool park) {
    // a hybrid whose second cache mirrors the attention cells (qwen4exp/glm5-next's indexer) moves both together
    if (auto * hyb_idx = dynamic_cast<llama_memory_hybrid_idx *>(mem)) {
        return hyb_idx->seq_park(seq_id, park);
    }

    std::vector<llama_kv_cache *> kvs;
    if (!llama_memory_attn_caches(mem, kvs)) {
        return false;
    }

    // all or nothing: undo the caches already moved when a later one cannot move
    std::vector<bool> moved(kvs.size(), false);
    for (size_t i = 0; i < kvs.size(); ++i) {
        if (kvs[i] == nullptr) {
            continue;
        }
        const bool was = kvs[i]->seq_parked(seq_id);
        if (!(park ? kvs[i]->seq_park(seq_id) : kvs[i]->seq_unpark(seq_id))) {
            for (size_t j = 0; j < i; ++j) {
                if (moved[j]) {
                    park ? kvs[j]->seq_unpark(seq_id) : kvs[j]->seq_park(seq_id);
                }
            }
            return false;
        }
        moved[i] = kvs[i]->seq_parked(seq_id) != was;
    }

    return true;
}
