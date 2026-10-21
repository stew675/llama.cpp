#pragma once

// wip/slab-transient-absorption (step 1): DERIVE the slab's auxiliary ("draft") reserve from the draft
// model's own GGUF instead of the flat `LLAMA_MOE_CACHE_AUX_RESERVE_MIB_DEFAULT`.
//
// Background: the movable-boundary slab holds back `headroom + aux` at creation, and the MTP / speculative
// draft context is created AFTER the slab, so its VRAM comes out of that reserve.  The reserve is a one-shot
// decision (the slab's mapping cannot be reduced later), and getting it wrong is asymmetric:
//   - too SMALL: the draft eats the headroom, the outside cushion collapses and an outside `cudaMalloc`
//     aborts the run (the slab cannot unmap sub-ranges to help);
//   - too LARGE: the surplus is reclaimed into the arena by `ggml_cuda_slab_extend` when the draft left more
//     than `headroom` outside, so it mostly costs nothing.
// Measured on the field config (2026-10-17): the draft's real footprint is 3 584 MiB at
// `-ub 4096 -c 131072` and 4 000 MiB at `-ub 6144 -c 204800` -- the flat 3 584 constant is exactly the
// first and ~416 MiB short for the second, which is why a long-context run needs a bigger reserve than the
// constant implies.  The weights are 2 647 MiB of it (the draft GGUF's tensor bytes, exact); the rest is KV
// + compute + load transients.
//
// This estimate is deliberately CONSERVATIVE where the metadata is ambiguous (it returns `ok = false`
// rather than guessing when a key it needs is missing, and the caller then keeps the constant).

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <string>

struct common_draft_reserve {
    size_t weights = 0;   // draft weights that must be resident on the device (sum of GGUF tensor bytes)
    size_t kv      = 0;   // KV cache for `n_ctx`: attention k+v per layer + any declared indexer cache
    size_t compute = 0;   // the draft context's compute buffer (a measured tier keyed on the ubatch)
    size_t slack   = 0;   // the draft's own load / steady-state transients
    size_t total   = 0;   // weights + kv + compute + slack  ->  MOE_EXPERT_CACHE_AUX_RESERVE_MIB

    bool        ok   = false;   // false: the GGUF did not declare what we need -> keep the constant
    std::string note;           // why it failed, for the log
};

// Estimate the post-slab VRAM `path` (a draft GGUF) will need for `n_ctx` with the given KV types.
common_draft_reserve common_draft_reserve_estimate(const std::string & path,
                                                   uint32_t n_ctx,
                                                   uint32_t n_ubatch,
                                                   ggml_type type_k,
                                                   ggml_type type_v);
