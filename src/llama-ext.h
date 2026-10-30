#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "llama.h"

#include <cstdint>
#include <map>
#include <vector>

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(quantize_state_impl * qs);

// Descriptor for constructing a mock model for quantization testing.
struct llama_quant_model_desc {
    const char * architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with llama_model_free().
LLAMA_API llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
LLAMA_API bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use llama_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
LLAMA_API void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers

    size_t total() const {
        return model + context + compute;
    }
};

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using llama_memory_breakdown = std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>;

LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);
LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

// wip/fit-slab-accounting (Phase 1): enumerate the REAL devices that own host-resident MoE expert
// weights (the `moe_host_expert_bytes` map).  Under `-sm tensor` the model's device list is the
// scheduler's Meta wrapper, so `llama_model_moe_host_expert_bytes()` sees 0 on every device; `--fit`
// needs the per-real-device bytes to reserve the expert-cache arena and the slab headroom.
LLAMA_API int32_t llama_model_moe_host_expert_dev_count(const struct llama_model * model);

// Fill *dev/*bytes with the i-th host-expert device (map iteration order).  Returns false when i is out
// of range.  Either out pointer may be null.
LLAMA_API bool llama_model_moe_host_expert_dev(const struct llama_model * model, int32_t i,
                                               ggml_backend_dev_t * dev, size_t * bytes);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);

// the backend buffers that a state write of this memory would touch - empty caches report none
LLAMA_API std::vector<ggml_backend_buffer_t> llama_memory_get_buffers(const llama_memory_t mem);

// Parking (unified KV cache).  park = true moves the cells of an idle sequence to the top of the attention cache(s) of
// mem: the sequences that are decoded keep the low cells, and attend over the same range at the same cell indices as
// in a cache that holds nothing else, so their results no longer depend on what is parked.  park = false moves a
// parked sequence back to the lowest free cells.  Returns false, changing nothing, when the move is not possible
// (a split cache, shared or fragmented cells, no room); a memory without an attention cache returns true.  The K/V data
// are copied on the device, so synchronize the context first.  A memory whose second cache mirrors the attention cells
// (the qwen4exp/glm5-next indexer) moves both, or neither.
LLAMA_API bool llama_memory_seq_park(llama_memory_t mem, llama_seq_id seq_id, bool park);

// Whether the attention cache(s) of mem support parking at all (see llama_memory_seq_park): a unified cache whose cells
// no other cache shares, with V not transposed and no sliding window, plus, for a memory whose second cache mirrors the
// attention cells, that cache as well.  A memory without an attention cache: true.
LLAMA_API bool llama_memory_can_park(llama_memory_t mem);

// Turn the KV cache attention rotation on or off for the contexts created from this model.
// Useful for testing purposes.
LLAMA_API void llama_model_set_attn_rot(struct llama_model * model, bool value);

// Set whether the context outputs nextn embeddings or not
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits
LLAMA_API void llama_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);

// Unmasked nextn rows of `seq_id` at positions < `pos` are not needed by the caller: a ubatch whose every row is
// such a row skips the device->host copy of the nextn embeddings (and the readback wait).  Those host rows then hold
// stale data.  0 (the default) = every row is needed.  Used by the MTP prompt-tail ingest (LLAMA_MTP_INGEST_TAIL);
// the skip it enables is a bisect kill-switch away (LLAMA_MTP_TAIL_READBACK=0 reads every row again).
LLAMA_API void llama_set_embeddings_nextn_from(struct llama_context * ctx, llama_seq_id seq_id, llama_pos pos);

// Select which appended NextN block the DECODER_MTP graph runs (offset past
// the trunk: il = n_layer() + offset). Used by the speculative NextN driver to
// chain multiple trained NextN heads. Default 0 (first head).
LLAMA_API void llama_set_nextn_layer_offset(struct llama_context * ctx, int32_t offset);

// Marks the entries that a decision head (clef, vllm-decision2) reads, the default is 0
// See https://github.com/ggml-org/llama.cpp/pull/29831 for details
// A run of entries with the same value is one span, spans must be separated by entries with value 0
// An option belongs to the last question before it
enum llama_decision_order {
    LLAMA_DECISION_ORDER_NONE            = 0, // not read by the head
    LLAMA_DECISION_ORDER_QUESTION_NOUL   = 1, // text of a question
    LLAMA_DECISION_ORDER_QUESTION_CHOICE = 2,
    LLAMA_DECISION_ORDER_QUESTION_SCORE  = 3,
    LLAMA_DECISION_ORDER_OPTION          = 4, // text of an option
};
// The embeddings output has one value per entry: row i is the score of option i
LLAMA_API bool llama_batch_ext_set_decision_order(struct llama_batch_ext * batch, int32_t idx, enum llama_decision_order order);

// Label of option i of a decision model ("<arch>.decision.labels"), nullptr past the last one
LLAMA_API const char * llama_model_decision_label(const struct llama_model * model, uint32_t i);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_nextn(struct llama_context * ctx);

// LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i);

// Set whether the context outputs the input embeddings of a specific layer
LLAMA_API void llama_set_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid, bool value);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid);

// Llama-Frankenstein F1: device-resident layer inputs for draft feature injection
LLAMA_API void                 llama_lf_set_layer_inp_dev(struct llama_context * ctx, bool enable);
LLAMA_API struct ggml_tensor * llama_lf_get_layer_inp_dev(struct llama_context * ctx, uint32_t lid);
// target context: the device buffer row of each row of its last batch, or nullptr when they are the same (the ubatches
// kept the batch order: always for one sequence); *n_rows = the batch size when not nullptr
LLAMA_API const int32_t *      llama_lf_get_dev_rows     (struct llama_context * ctx, int32_t * n_rows);
// draft context: while enabled, a token batch is a KV injection whose token ids are the rows of the target's
// last batch (its ctx_other) to read from the device layer inputs
LLAMA_API void                 llama_lf_set_dev_inject   (struct llama_context * ctx, bool enable);
// draft context: after the injections of a batch were enqueued; the target's next extraction waits for them on the GPU
LLAMA_API void                 llama_lf_signal_features_consumed(struct llama_context * ctx_dft);

LLAMA_API llama_context * llama_get_ctx_other(struct llama_context * ctx);

//
// model/context data extraction
//

LLAMA_API int32_t llama_model_dflash_selector_top_k(const struct llama_model * model);

// number of layers that attend over a sliding window of llama_model_n_swa() positions
LLAMA_API int32_t llama_model_n_layer_swa(const struct llama_model * model);

// returns pointer to the target-model layer indices
LLAMA_API const int32_t * llama_model_target_layer_ids  (const struct llama_model * model);
// returns the number of extracted layers from target model
LLAMA_API uint32_t        llama_model_target_layer_ids_n(const struct llama_model * model);

// layers whose input embeddings contexts created from this model extract from the start (as if
// llama_set_embeddings_layer_inp(ctx, lid, true) had been called before the context's first graph reserve).
// For a target whose draft will turn the extraction on (DFlash): the compute buffers are then sized for it once,
// instead of being freed and allocated again, larger, at the first decode after the draft attaches.
LLAMA_API void llama_model_set_embeddings_layer_inp_preset(struct llama_model * model, const int32_t * lids, uint32_t n);

// retrieves the whole token embedding matrix in F32 format (n_embd * n_vocab)
// returns total number of elements or 0 on error
// if out is nullptr, returns the number of tokens without writing to out
// caller must allocate enough memory for out before calling
LLAMA_API uint32_t llama_model_get_tok_embd(const struct llama_model * model, float * out);

// same as above, but only the rows of the given tokens, in order (n_tokens * n_embd elements)
// returns the number of elements written or 0 on error
LLAMA_API uint32_t llama_model_get_tok_embd_rows(const struct llama_model * model, const llama_token * tokens, uint32_t n_tokens, float * out);

// realtime models: number of tokens the output lags behind the audio input, used by mtmd; 0 if not applicable
LLAMA_API uint32_t llama_model_n_tokens_delay(const struct llama_model * model);

//
// wip/kv-residency (TODO #118): the windowed KV cache.  Nothing here is in the delivery.
//
// `LLAMA_KV_RESIDENT_CELLS=<cells>` keeps the newest <cells> cells of the KV in VRAM and puts everything
// older in pinned system RAM; `=auto` leaves the number to `--fit`, which sizes it against the device memory
// it has left after the expert arena's reserve and the pinned-RAM ceiling on the spill.
//
// Is the KV region provider available (`LLAMA_KV_REGION=1` and a device that offers VMM)?
LLAMA_API bool llama_kv_region_is_enabled(void);

// The window the cache would use: 0 = none, UINT32_MAX = `auto` (fit decides), else the cell count.
// Reads the fit decision first, then the environment.
LLAMA_API uint32_t llama_kv_region_resident_cells_requested(void);

// Hand a window (in cells) down to every context created afterwards; 0 clears it.  `common_params_fit` calls
// this before the real contexts exist.  One W covers every cache in the process (the A2 rule).
LLAMA_API void llama_kv_region_set_resident_cells(uint32_t cells);
