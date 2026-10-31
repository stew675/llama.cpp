#pragma once

struct ggml_cuda_mm_id_tile_spec {
    int32_t expert;
    int32_t col_low;
    int32_t col_high;

    int32_t padding;
};
static_assert(sizeof(ggml_cuda_mm_id_tile_spec) == 16, "bad ggml_cuda_mm_id_tile_spec size");

// Re-emit a tile_spec's seams at a different tile width from the per-expert bounds the compaction wrote.
// The ids mapping is tile-width independent, so a caller that needs two layouts (the sparse-MoE gate+up pair,
// whose two weights may land in different J bands) compacts ONCE and retiles the second arm (TODO #301).
void ggml_cuda_launch_mm_id_retile(
        ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec, const int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int tile_width, cudaStream_t stream);

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * __restrict__ ids,
        int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec,
        int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int tile_width,
        const int si1, const int sis1, const bool write_inverse, cudaStream_t stream);
