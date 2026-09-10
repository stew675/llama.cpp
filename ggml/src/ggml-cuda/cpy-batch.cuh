#pragma once
#include "common.cuh"
// Batched copy (RDNA4; GGML_CUDA_FUSE_CPY_BATCH=0 turns it off): runs a run of consecutive f32 GGML_OP_CPY nodes that
// share the source shape / strides and the destination shape / strides (only the data addresses differ) as one launch, e.g. the
// per-position DeltaNet conv-state snapshots of a speculative verify batch (one copy per drafted position and layer).  Element
// copies in ggml_cpy's flat order: bit-identical.
// Returns the number of graph nodes it consumed (the copies plus the no-op view nodes between them, >= 2), or 0 when the node
// at i does not start such a run.  GGML_CUDA_FUSE_CPY_BATCH_DEBUG=1 prints every run.
int ggml_cuda_cpy_batch(ggml_backend_cuda_context & ctx, struct ggml_cgraph * cgraph, int i);
