#pragma once
#include "common.cuh"
// MMB: dequant-to-BF16 WMMA GEMM path for IQ4_NL weights on gfx1151 (RDNA3.5). Env-gated: LLAMA_MMB=1, LLAMA_MMB_MIN_T (default 512).
bool ggml_cuda_mmb_supported_mm  (const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);
bool ggml_cuda_mmb_supported_mmid(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst);
// MMB takes this dense GEMM and reads its activation via the bf16 activation cache (so the
// activation's F32 output is optional).
bool ggml_cuda_mmb_reads_bf16_act(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);
void ggml_cuda_mul_mat_mmb   (ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cuda_mul_mat_id_mmb(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst);
void ggml_cuda_mmb_begin_graph();
// producers that can emit a BF16 copy of an F32 output register it here; returns the BF16 buffer to fill (n elements)
uint16_t * ggml_cuda_mmb_cache_reserve(ggml_backend_cuda_context & ctx, const ggml_tensor * t, size_t n);
// BF16 copy of tensor t if one is cached for the current graph (consumers may read it instead of the F32 data)
const uint16_t * ggml_cuda_mmb_cache_lookup(const ggml_tensor * t);
// producer slots (pinned until the next producer of the same kind): 0 = HC normalized stream xn, 1 = HC gate
uint16_t * ggml_cuda_mmb_slot_reserve(ggml_backend_cuda_context & ctx, int slot, const ggml_tensor * t, size_t n);
// Producer-side BF16-copy reserve that honours a graph-assigned dedicated slot (xn).
uint16_t * ggml_cuda_mmb_reserve_auto(ggml_backend_cuda_context & ctx, const ggml_tensor * t, size_t n);
void ggml_cuda_mmb_mark_bf16_slot(const ggml_tensor * t, int slot);
int  ggml_cuda_mmb_bf16_slot(const ggml_tensor * t);
void ggml_cuda_mmb_marks_clear();
// select the mark set used by the mark/lookup helpers below (per backend context; set at the start
// of ggml_backend_cuda_graph_optimize and _graph_compute before any op runs)
void ggml_cuda_mmb_set_active_ctx(const ggml_backend_cuda_context * ctx);
// per-context mark lifecycle: call optimize_begin(key) at the start of graph_optimize (it clears the
// marks when a new graph starts) and compute_done() at the end of graph_compute
bool ggml_cuda_mmb_optimize_begin(const void * graph_key);
void ggml_cuda_mmb_compute_done();
size_t ggml_cuda_mmb_marks_count();
void ggml_cuda_mmb_mark_bf16_only(const ggml_tensor * t);
bool ggml_cuda_mmb_is_bf16_only(const ggml_tensor * t);
// "also emit a BF16 copy" mark (the F32 output stays valid); set by the graph optimizer for MMB
// dense GEMM activations, honoured by the fused rms_norm+mul, fused sigmoid+mul and dsv4_hc_pre.
void ggml_cuda_mmb_mark_bf16_copy(const ggml_tensor * t);
bool ggml_cuda_mmb_wants_bf16_copy(const ggml_tensor * t);
bool ggml_cuda_mmb_gatemix();
bool ggml_cuda_mmb_active();
// True iff MMB will actually take a dense MUL_MAT / a routed MUL_MAT_ID or its SWIGLU pair for this
// weight type. The graph optimizer stands its own fusions down on this predicate rather than on the
// global gate, so enabling MMB on a model whose expert type it does not support keeps the fusions.
bool ggml_cuda_mmb_dense_will_take(const ggml_tensor * w);
bool ggml_cuda_mmb_routed_will_take(const ggml_tensor * w);
bool ggml_cuda_mmb_down16();
bool ggml_cuda_mmb_res16();
bool ggml_cuda_mmb_blk16();
bool ggml_cuda_hc_gate_mix(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * lo, const ggml_tensor * xn, ggml_tensor * dst, int hc, float scale, float bias);
bool ggml_cuda_mmb_supported_glu(const ggml_tensor * gw, const ggml_tensor * uw, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * glu);
void ggml_cuda_mul_mat_id_mmb_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * gw, const ggml_tensor * uw, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * glu);
void ggml_cuda_mmb_shadow_prepare(ggml_backend_cuda_context & ctx, const ggml_tensor * w);
void ggml_cuda_mmb_release_all();
