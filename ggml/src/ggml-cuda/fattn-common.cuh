#pragma once

#include "common.cuh"
#include "convert.cuh"
#include "vecdotq.cuh"

#include <cstdint>

#define FATTN_KQ_STRIDE       256
#define HALF_MAX_HALF         __float2half(65504.0f/2) // Use neg. of this instead of -INFINITY to initialize KQ max vals to avoid NaN upon subtraction.

// Fully-masked kq mask block skip (issue #48 / upstream #28495).  A unified KV cache keeps the other
// slots' cells inside the attention range; they are fully masked (-INF) for the current sequence but
// the kernels used to process every one of their FATTN_KQ_STRIDE-sized KV groups, which made a
// concurrent prefill pay up to ~2x.  The launcher classifies the groups once -- from the derived
// per-cell state (flash_attn_kq_derived_blocks, single-sequence prefill) or from the packed mask
// (flash_attn_mask_to_KV_blocks, multi-sequence prefill) -- and the kernel skips the fully-masked
// runs.  A skipped group only ever contained maximum-subtracted -INF logits, so the softmax/PV
// accumulators are unchanged and the result is bit-identical.  GGML_CUDA_FA_MASK_SKIP=0 is the
// A/B kill-switch.
static inline bool ggml_cuda_fattn_kq_block_skip_enabled() {
    static const bool enabled = []() {
        const char * e = getenv("GGML_CUDA_FA_MASK_SKIP");
        return e == nullptr || atoi(e) != 0;
    }();
    return enabled;
}

// Runtime test of the group bitmap (either producer): true when the KV block starting at cell offset
// kb0*nbatch_fa is entirely masked for every query row the bitmap was classified against.  On the
// derived path the bitmap is batch-wide (one word run); on the packed path it is already offset to
// the (stream, query tile) by the caller.  nbatch_fa must divide FATTN_KQ_STRIDE (the launcher gates
// on that).
static __device__ __forceinline__ bool fattn_kq_group_masked(const uint32_t * kq_blocks, const int kb0, const int nbatch_fa) {
    const int group = (kb0*nbatch_fa) / FATTN_KQ_STRIDE;
    return (kq_blocks[group >> 5] >> (group & 31)) & 1u;
}
#define SOFTMAX_FTZ_THRESHOLD -20.0f                   // Softmax exp. of values smaller than this are flushed to zero to avoid NaNs.

// log(2) = 0.6931, by adding this to the KQ maximum used for the softmax the numerical range representable
//     by the VKQ accumulators is effectively being shifted up by a factor of 2.
// This reduces issues with numerical overflow but also causes larger values to be flushed to zero.
// However, as the output from FlashAttention will usually be used as an input for a matrix multiplication this should be negligible.
// Still, the value range should be shifted as much as necessary but as little as possible.
// The macro on the following line shifts it by a factor of 2**3=8, as was needed to fix https://github.com/ggml-org/llama.cpp/issues/18606 .
#define FATTN_KQ_MAX_OFFSET (3.0f*0.6931f)

// V3 derived kq mask (ggml_flash_attn_ext_add_kq_derived).  When cell_pos is not nullptr the mask
// tensor is absent and each cell's value (0 or -INFINITY) is derived from the cell's position and the
// query token's visibility window, exactly as the packed mask would hold it.  Implemented by the MMA
// and the tile kernels; the vec kernel is decode/verify-only (n_tps <= 2) and never sees a derived op
// (kq_mask_derivable() rejects n_tokens <= 8).
struct kq_derived_t {
    const int * cell_pos; // per KV cell: INT32_MIN = always dropped (empty cell, or another sequence's)
    const int * tok_lo;   // per query token: inclusive lower bound of the visible position range
    const int * tok_hi;   // per query token: inclusive upper bound
};

typedef void (* fattn_kernel_t)(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
        // Fully-masked KV-group bitmap (issue #48; see ggml_cuda_fattn_kq_block_skip_enabled):
        // batch-wide (one word run, derived path) or per (stream, query tile) (packed path).
        const uint32_t * __restrict__ kq_blocks,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33,
        // V3 derived kq mask (ggml_flash_attn_ext_add_kq_derived): all three are nullptr unless the
        // mask tensor is absent and each cell's value is derived from its position instead.
        const int  * __restrict__ cell_pos, const int  * __restrict__ tok_lo, const int  * __restrict__ tok_hi,
        // Native (non-F16-staged) K/V operands: the fattn_kv_native_type of each, i.e.
        // FATTN_KV_NATIVE_NONE when the launcher staged an F16 copy and FATTN_KV_NATIVE_Q8_0/BF16 when
        // the kernel reads the raw cache itself (V4 / the block-15 amendment).  Always NONE for the
        // tile and vec kernels, which stage whichever type they were compiled for.
        const int kv_native_K, const int kv_native_V);

typedef float (*vec_dot_KQ_t)(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds);

struct ggml_cuda_flash_attn_ext_f16_extra_data {
    uintptr_t K;
    uintptr_t V;
    uintptr_t end;
};

static inline ggml_cuda_flash_attn_ext_f16_extra_data ggml_cuda_flash_attn_ext_get_f16_extra_data(
        const ggml_tensor * dst, const bool need_f16_K, const bool need_f16_V) {
    GGML_ASSERT(dst->op == GGML_OP_FLASH_ATTN_EXT);

    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    GGML_ASSERT(K != nullptr);
    GGML_ASSERT(V != nullptr);

    const bool V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    ggml_cuda_flash_attn_ext_f16_extra_data data = {};
    data.end = (uintptr_t) dst->data + ggml_nbytes(dst);

    if (need_f16_K && K->type != GGML_TYPE_F16) {
        data.end = GGML_PAD(data.end, 128);
        data.K   = data.end;
        data.end += ggml_nelements(K)*ggml_type_size(GGML_TYPE_F16);
    }

    if (need_f16_V && V->type != GGML_TYPE_F16) {
        if (V_is_K_view) {
            data.V = data.K;
        } else {
            data.end = GGML_PAD(data.end, 128);
            data.V   = data.end;
            data.end += ggml_nelements(V)*ggml_type_size(GGML_TYPE_F16);
        }
    }

    return data;
}

// -------------------------------------------------------------------------------------------------
// V4: native q8_0 K/V in the flash-attention kernels.
//
// With a quantized KV cache the MMA/TILE kernels used to read an F16 copy of the *whole* cache that
// the launcher staged into a scratch region appended to the FA node (see get_f16_extra_data above),
// i.e. up to ~800 MiB per GPU at a 200k context.  The kernels can instead dequantize while staging
// K/V into their shared-memory tiles: a 16-byte half2 chunk covers exactly GGML_CUDA_FA_Q8_CHUNK (8)
// elements, a quarter of a q8_0 block.  The staged values are bit-identical to the ones the F16
// scratch holds: convert.cu's dequantize_block_q8_0_f16 (the contiguous conversion the launcher
// uses) computes a single F16 rounding of the exact int8 * F16-scale product, which is exactly what
// ggml_cuda_fattn_dequantize_q8_0_chunk() below computes.
//
// The F16 scratch (and its per-ubatch global conversion pass) is only skipped when the launcher and
// ggml_cuda_flash_attn_ext_get_alloc_size agree, and both ask the predicates below, so they cannot
// disagree.  Everything else (other quantized types, mixed K/V types, tile layouts the kernels
// cannot chunk, builds without fast FP16) keeps the existing conversion.

static constexpr int GGML_CUDA_FA_Q8_CHUNK = 8; // elements staged per 16-byte shared-memory chunk

// ACTIVATION POLICY (amended 2026-09-14, issue #30 — refinement uncovered by the reporter; bf16
// flipped to default-ON 2026-09-25, the r6 block-15 amendment): native staging is the DEFAULT for the
// sub-F16 quantized K/V types (q8_0): the F16 scratch's
// per-ubatch conversion pass is proportional to n_kv and runs on every decode step, so removing it is
// worth +23 % decode at d = 65536 on gfx1201 (18.92 -> 23.29 tg64, ahead of stock's 22.43) while
// costing only ~1.2 % prefill, and it drops ~744 MiB of scratch at a 200k context.  A q8_0 cache is
// itself the memory-constrained configuration, so the scratch it removes matters there too.  bf16 (V5,
// below) is now default-ON for the same class of reason: it removes the MMA scratch, and since r5's
// band coverage the native arm is what puts bf16 on the RDNA4 GQA-6 decode/verify band (+14 %
// `draft-mtp n3` at ~30k on gfx1201), so it has the equivalent decode win the original opt-in note
// was waiting for.  The prefill cost it pays (0.2-2.4 % on the MMA path) is the trade.
//   GGML_CUDA_FA_KV_NATIVE unset -> auto: q8_0/q4_0/q4_1/q5_0/q5_1/iq4_nl native ON, bf16 native ON
//   GGML_CUDA_FA_KV_NATIVE=1     -> force all ON
//   GGML_CUDA_FA_KV_NATIVE=0     -> force all OFF (the pre-2026-09-14 behavior; the F16 scratch path)
enum fattn_kv_native_policy : int {
    FATTN_KV_NATIVE_AUTO = -1,
    FATTN_KV_NATIVE_OFF  =  0,
    FATTN_KV_NATIVE_ON   =  1,
};

static inline int ggml_cuda_fattn_kv_native_policy() {
    static const int policy = []() {
        const char * env = getenv("GGML_CUDA_FA_KV_NATIVE");
        if (env == nullptr) {
            return (int) FATTN_KV_NATIVE_AUTO;
        }
        return atoi(env) != 0 ? (int) FATTN_KV_NATIVE_ON : (int) FATTN_KV_NATIVE_OFF;
    }();
    return policy;
}

// q8_0 native staging (V4): default on in auto mode.
static inline bool ggml_cuda_fattn_kv_native_q8_enabled() {
    return ggml_cuda_fattn_kv_native_policy() != (int) FATTN_KV_NATIVE_OFF;
}

// q4_0 native staging (2026-09-14, issue #30): same policy as q8_0.  q4_0 is the other sub-F16 quant
// the reporter hit; without a native arm it pays the same whole-cache F16 staging pass as q8_0 used,
// which is a depth-proportional decode cost (and its staging conversion is the more expensive one).
static inline bool ggml_cuda_fattn_kv_native_q4_enabled() {
    return ggml_cuda_fattn_kv_native_policy() != (int) FATTN_KV_NATIVE_OFF;
}

// bf16 native staging (V5): default on in auto mode (r6, 2026-09-25), same policy as q8_0/q4_0.
static inline bool ggml_cuda_fattn_kv_native_bf16_enabled() {
    return ggml_cuda_fattn_kv_native_policy() != (int) FATTN_KV_NATIVE_OFF;
}

// `t` is the K or V operand of a FLASH_ATTN_EXT node.
static inline bool ggml_cuda_fattn_kv_native_supported(const ggml_tensor * t) {
#ifdef FAST_FP16_AVAILABLE
    if (!ggml_cuda_fattn_kv_native_q8_enabled()) {
        return false;
    }
    if (t == nullptr || t->type != GGML_TYPE_Q8_0) {
        return false;
    }
    // Rows are block-contiguous and every staged chunk starts at a multiple of GGML_CUDA_FA_Q8_CHUNK
    // elements, so a chunk never straddles two q8_0 blocks (the kernel-side chunking is checked by
    // the static_asserts in the loaders).
    if (t->ne[0] % GGML_CUDA_FA_Q8_CHUNK != 0) {
        return false;
    }
    if (t->nb[0] != (size_t) ggml_type_size(GGML_TYPE_Q8_0)) {
        return false;
    }
    return true;
#else
    GGML_UNUSED(t);
    return false;
#endif // FAST_FP16_AVAILABLE
}

// The tile kernel has a single K/V type parameter, so it needs both operands to qualify.
static inline bool ggml_cuda_fattn_tile_kv_native(const ggml_tensor * K, const ggml_tensor * V) {
    return ggml_cuda_fattn_kv_native_supported(K) && ggml_cuda_fattn_kv_native_supported(V);
}

// `t` is the K or V operand of a FLASH_ATTN_EXT node.
static inline bool ggml_cuda_fattn_kv_q4_0_supported(const ggml_tensor * t) {
#ifdef FAST_FP16_AVAILABLE
    if (!ggml_cuda_fattn_kv_native_q4_enabled()) {
        return false;
    }
    if (t == nullptr || t->type != GGML_TYPE_Q4_0) {
        return false;
    }
    // Same constraints as q8_0: block-contiguous rows and 8-element (16-byte staged) chunks that
    // never straddle a q4_0 block.  A q4_0 block is 32 elements (16 low nibbles then 16 high), and
    // the chunk grid advances by 8, so an 8-element chunk is entirely low- or entirely high-nibble.
    if (t->ne[0] % GGML_CUDA_FA_Q8_CHUNK != 0) {
        return false;
    }
    if (t->nb[0] != (size_t) ggml_type_size(GGML_TYPE_Q4_0)) {
        return false;
    }
    return true;
#else
    GGML_UNUSED(t);
    return false;
#endif // FAST_FP16_AVAILABLE
}

static inline bool ggml_cuda_fattn_tile_kv_native_q4_0(const ggml_tensor * K, const ggml_tensor * V) {
    return ggml_cuda_fattn_kv_q4_0_supported(K) && ggml_cuda_fattn_kv_q4_0_supported(V);
}

// Issue #30 item 2: native arms for the remaining quantized K/V block types.  They share the sub-F16
// policy switch with q8_0/q4_0 (GGML_CUDA_FA_KV_NATIVE=0 disables them all) and the same layout
// requirement: an 8-element staged chunk must not straddle a block, and every one of them has a
// 32-element block, so 8 divides it and the chunk grid (which advances by 8) always lands inside one
// low- or one high-nibble half.
static inline bool ggml_cuda_fattn_kv_layout_ok(const ggml_tensor * t, const ggml_type type) {
    return t != nullptr && t->type == type &&
        t->ne[0] % GGML_CUDA_FA_Q8_CHUNK == 0 &&
        t->nb[0] == (size_t) ggml_type_size(type);
}

#define GGML_CUDA_FATTN_KV_NATIVE_ARM(NAME, TYPE)                                                       \
    static inline bool ggml_cuda_fattn_kv_##NAME##_supported(const ggml_tensor * t) {                    \
        if (!ggml_cuda_fattn_kv_native_q4_enabled()) { return false; }                                   \
        return ggml_cuda_fattn_kv_layout_ok(t, TYPE);                                                     \
    }                                                                                                    \
    static inline bool ggml_cuda_fattn_tile_kv_native_##NAME(const ggml_tensor * K, const ggml_tensor * V) { \
        return ggml_cuda_fattn_kv_##NAME##_supported(K) && ggml_cuda_fattn_kv_##NAME##_supported(V);      \
    }

GGML_CUDA_FATTN_KV_NATIVE_ARM(q4_1,   GGML_TYPE_Q4_1)
GGML_CUDA_FATTN_KV_NATIVE_ARM(q5_0,   GGML_TYPE_Q5_0)
GGML_CUDA_FATTN_KV_NATIVE_ARM(q5_1,   GGML_TYPE_Q5_1)
GGML_CUDA_FATTN_KV_NATIVE_ARM(iq4_nl, GGML_TYPE_IQ4_NL)

#undef GGML_CUDA_FATTN_KV_NATIVE_ARM

// -------------------------------------------------------------------------------------------------
// Block-15 amendment: native bf16 K/V in the MMA flash-attention kernel.
//
// bf16 was the one KV type that still paid the whole F16 staging cost (~712 MiB per GPU at a 200k
// context, ub 2048): the tile and vec kernels read bf16 natively (block 03) but the MMA kernel did
// not, so the launcher staged an F16 copy of the whole cache.  Unlike q8_0 (V4) there is nothing to
// dequantize: a bf16 row and the F16 tile it feeds have exactly the same byte layout (2 bytes per
// element, 16-byte chunks), so the kernel only has to convert each staged chunk in registers
// (bf16 -> f32 -> f16).  That is bit-identical to the launcher's own conversion
// (ggml_get_to_fp16_cuda(GGML_TYPE_BF16) is ggml_cuda_cast<half>(nv_bfloat16), the same rounding), so
// the arithmetic does not change and only the node's scratch disappears.
//
// OPT-IN (default off, enabled by the same GGML_CUDA_FA_KV_NATIVE=1 switch as V4's q8_0 arm - one
// switch for the whole "stage K/V natively instead of through the F16 scratch" idea): a bf16 K/V
// cache no longer needs the F16 staging scratch, so it costs exactly what an F16 cache costs
// (measured: 4B 2048 968.9 -> 256.9 MiB, 27B 1072.9 -> 488.9, gemma-4-E4B 1062.9 -> 404.9,
// gemma-4-31B 2068.9 -> 716.9 at ctx 204800).  The conversion itself is free - native bf16 staging
// measures within 0.2 % of an F16 cache - but dropping the scratch costs ~0.8-2.4 % prefill, growing
// with the prompt length: the launcher's F16 staging copy is a *dense, normalized* copy of the cache
// view (nb[1] is 4x the row size for a 4-KV-head model, because the GQA heads are interleaved), and
// the FA nodes then stage from it, whereas the native path re-reads the interleaved view on every
// staging pass.  Same trade-off and same decision as V4 (see the block-15 notes in patches/README.md);
// decode is unaffected (~0.1 %).
//
// `t` is the K or V operand of a FLASH_ATTN_EXT node.
static inline bool ggml_cuda_fattn_kv_bf16_supported(const ggml_tensor * t) {
#ifdef FAST_FP16_AVAILABLE
    if (!ggml_cuda_fattn_kv_native_bf16_enabled()) {
        return false;
    }
    if (t == nullptr || t->type != GGML_TYPE_BF16) {
        return false;
    }
    // The staged chunk is 16 bytes, i.e. GGML_CUDA_FA_Q8_CHUNK 2-byte elements (the same chunk unit
    // as the q8_0 arm: both K/V element types are 2 bytes wide), and rows must be contiguous.
    if (t->ne[0] % GGML_CUDA_FA_Q8_CHUNK != 0) {
        return false;
    }
    if (t->nb[0] != (size_t) ggml_type_size(GGML_TYPE_BF16)) {
        return false;
    }
    return true;
#else
    GGML_UNUSED(t);
    return false;
#endif // FAST_FP16_AVAILABLE
}

// The staging source of one K/V operand, as the launcher and the kernel agree on it: either the
// launcher staged an F16 copy in the node's scratch, or the kernel reads the raw cache itself.
enum fattn_kv_native_type : int {
    FATTN_KV_NATIVE_NONE = 0, // F16-staged data (the usual path)
    FATTN_KV_NATIVE_Q8_0 = 1, // raw q8_0 rows, dequantized while staging the tiles (V4, default)
    FATTN_KV_NATIVE_BF16 = 2, // raw bf16 rows, converted to F16 while staging the tiles (block 15, opt-in)
    FATTN_KV_NATIVE_Q4_0 = 3, // raw q4_0 rows, dequantized while staging the tiles (V4 analogue, default)
    FATTN_KV_NATIVE_Q4_1 = 4, // raw q4_1 rows (issue #30 item 2, default)
    FATTN_KV_NATIVE_Q5_0 = 5, // raw q5_0 rows (issue #30 item 2, default)
    FATTN_KV_NATIVE_Q5_1 = 6, // raw q5_1 rows (issue #30 item 2, default)
    FATTN_KV_NATIVE_IQ4_NL = 7, // raw iq4_nl rows (issue #30 item 2, default)
};

// The MMA kernel reads each operand with its own native type; the tile and vec kernels have a single
// K/V type parameter and therefore one type for BOTH operands.  A caller passes this sentinel to
// launch_fattn when the kernel derives the type per operand.
static constexpr int FATTN_KV_NATIVE_PER_OPERAND = -1;

// The native type the MMA kernel may use for `t` (FATTN_KV_NATIVE_NONE means the F16 staging stays).
// The launcher, ggml_cuda_flash_attn_ext_get_alloc_size and ggml_cuda_flash_attn_ext_get_f16_extra_data
// all ask this, so they cannot disagree on whether the scratch exists.
static inline int ggml_cuda_fattn_kv_native_type(const ggml_tensor * t) {
    if (ggml_cuda_fattn_kv_native_supported(t)) {
        return FATTN_KV_NATIVE_Q8_0;
    }
    if (ggml_cuda_fattn_kv_bf16_supported(t)) {
        return FATTN_KV_NATIVE_BF16;
    }
    if (ggml_cuda_fattn_kv_q4_0_supported(t)) {
        return FATTN_KV_NATIVE_Q4_0;
    }
    if (ggml_cuda_fattn_kv_q4_1_supported(t)) {
        return FATTN_KV_NATIVE_Q4_1;
    }
    if (ggml_cuda_fattn_kv_q5_0_supported(t)) {
        return FATTN_KV_NATIVE_Q5_0;
    }
    if (ggml_cuda_fattn_kv_q5_1_supported(t)) {
        return FATTN_KV_NATIVE_Q5_1;
    }
    if (ggml_cuda_fattn_kv_iq4_nl_supported(t)) {
        return FATTN_KV_NATIVE_IQ4_NL;
    }
    return FATTN_KV_NATIVE_NONE;
}

// The tile kernel has a single K/V type parameter, so it needs both operands to qualify for the SAME
// native type (issue #30: this is also why the launcher must be told the tile's choice explicitly).
static inline int ggml_cuda_fattn_tile_kv_native_type(const ggml_tensor * K, const ggml_tensor * V) {
    const int tk = ggml_cuda_fattn_kv_native_type(K);
    return tk != FATTN_KV_NATIVE_NONE && tk == ggml_cuda_fattn_kv_native_type(V) ? tk : FATTN_KV_NATIVE_NONE;
}

// The native K/V type a kernel instantiated with `type_KV` reads through the *dequant loaders*
// (constexpr, because the tile and vec kernels fix their K/V type at compile time).  BF16 and F16 are
// read through the kernel's own T_KV, so they map to NONE here.
template <ggml_type type_KV>
constexpr int ggml_cuda_fattn_native_type_from_kernel() {
    if constexpr (type_KV == GGML_TYPE_Q8_0)   { return FATTN_KV_NATIVE_Q8_0; }
    if constexpr (type_KV == GGML_TYPE_Q4_0)   { return FATTN_KV_NATIVE_Q4_0; }
    if constexpr (type_KV == GGML_TYPE_Q4_1)   { return FATTN_KV_NATIVE_Q4_1; }
    if constexpr (type_KV == GGML_TYPE_Q5_0)   { return FATTN_KV_NATIVE_Q5_0; }
    if constexpr (type_KV == GGML_TYPE_Q5_1)   { return FATTN_KV_NATIVE_Q5_1; }
    if constexpr (type_KV == GGML_TYPE_IQ4_NL) { return FATTN_KV_NATIVE_IQ4_NL; }
    return FATTN_KV_NATIVE_NONE;
}

// The ggml_type a native code stands for (FATTN_KV_NATIVE_NONE -> F16, i.e. the staged type).
static inline ggml_type ggml_cuda_fattn_native_ggml_type(const int native_type) {
    switch (native_type) {
        case FATTN_KV_NATIVE_Q8_0:   return GGML_TYPE_Q8_0;
        case FATTN_KV_NATIVE_BF16:   return GGML_TYPE_BF16;
        case FATTN_KV_NATIVE_Q4_0:   return GGML_TYPE_Q4_0;
        case FATTN_KV_NATIVE_Q4_1:   return GGML_TYPE_Q4_1;
        case FATTN_KV_NATIVE_Q5_0:   return GGML_TYPE_Q5_0;
        case FATTN_KV_NATIVE_Q5_1:   return GGML_TYPE_Q5_1;
        case FATTN_KV_NATIVE_IQ4_NL: return GGML_TYPE_IQ4_NL;
        default:                     return GGML_TYPE_F16;
    }
}

// Byte-addressed K/V rows for a native (non-F16-staged) operand.  The tile staging is unchanged
// (F16 half2 tiles); only the source of the staged values differs.  `type_K`/`type_V` is
// FATTN_KV_NATIVE_NONE when the operand is not native, i.e. the usual F16 conversion/scratch path.
struct fattn_kv_native_t {
    const char * K; // row-0 base of the K/V head (sequence/head offsets already applied)
    const char * V;
    int stride_K;   // bytes per KV cell
    int stride_V;
    int type_K;     // fattn_kv_native_type
    int type_V;
};

// Dequantize the GGML_CUDA_FA_Q8_CHUNK elements starting at element `el` of a q8_0 row into
// GGML_CUDA_FA_Q8_CHUNK/2 half2.  `el` must be a multiple of GGML_CUDA_FA_Q8_CHUNK so the chunk
// never straddles two blocks.  Same arithmetic as convert.cu's dequantize_block_q8_0_f16.
static __device__ __forceinline__ void ggml_cuda_fattn_dequantize_q8_0_chunk(
        const char * const __restrict__ row, const int el, half2 * const __restrict__ dst) {
    static_assert(sizeof(block_q8_0) == QK8_0 + 2, "bad block_q8_0");

    const int blk = el / QK8_0;
    const int off = el % QK8_0;
    const char * bp = row + (size_t) blk*sizeof(block_q8_0);

    // The block base is 2-byte aligned (the q8_0 block is 34 bytes and the rows are 16-byte
    // aligned), so the scale and the 8 quants can be fetched with 2-byte accesses.
    half d;
    ggml_cuda_memcpy_1<sizeof(half), 2>(&d, bp);
    const half2 d2 = __half2half2(d);

    int8_t q[GGML_CUDA_FA_Q8_CHUNK];
    ggml_cuda_memcpy_1<GGML_CUDA_FA_Q8_CHUNK, 2>(q, bp + sizeof(half) + off);

#pragma unroll
    for (int l = 0; l < GGML_CUDA_FA_Q8_CHUNK/2; ++l) {
        dst[l] = d2 * make_half2(q[2*l + 0], q[2*l + 1]);
    }
}

// Dequantize the GGML_CUDA_FA_Q8_CHUNK elements starting at element `el` of a q4_0 row into
// GGML_CUDA_FA_Q8_CHUNK/2 half2.  `el` must be a multiple of GGML_CUDA_FA_Q8_CHUNK.  Same arithmetic
// as convert.cu's dequantize_block_q4_0: d and dm are FP32 and each value is a single F16 rounding of
// `d * nibble + (-8*d)` (so the staged values are bit-identical to the F16 scratch the launcher
// would have built).
static __device__ __forceinline__ void ggml_cuda_fattn_dequantize_q4_0_chunk(
        const char * const __restrict__ row, const int el, half2 * const __restrict__ dst) {
    static_assert(sizeof(block_q4_0) == QK4_0/2 + 2, "bad block_q4_0");

    const int blk = el / QK4_0;
    const int off = el % QK4_0;
    const char * bp = row + (size_t) blk*sizeof(block_q4_0);

    half d_h;
    ggml_cuda_memcpy_1<sizeof(half), 2>(&d_h, bp);
    const float d  = __half2float(d_h);
    const float dm = -8.0f*d;

    // An 8-element chunk is entirely in the low half (elements 0..15) or the high half (16..31);
    // within a half, element (base + j) takes nibble j of byte j.
    const int lo   = off < QK4_0/2;
    const int base = lo ? off : off - QK4_0/2;
    const uint8_t * qs = (const uint8_t *) (bp + sizeof(half));

#pragma unroll
    for (int l = 0; l < GGML_CUDA_FA_Q8_CHUNK/2; ++l) {
        const uint8_t b0 = qs[base + 2*l + 0];
        const uint8_t b1 = qs[base + 2*l + 1];
        const float v0 = d * (lo ? (b0 & 0x0F) : (b0 >> 4)) + dm;
        const float v1 = d * (lo ? (b1 & 0x0F) : (b1 >> 4)) + dm;
        dst[l] = make_half2(__float2half(v0), __float2half(v1));
    }
}

// Dequantize the GGML_CUDA_FA_Q8_CHUNK elements starting at element `el` of a q4_1 row into
// GGML_CUDA_FA_Q8_CHUNK/2 half2.  `el` must be a multiple of GGML_CUDA_FA_Q8_CHUNK (so the chunk never
// straddles a block, and 8 divides the 32-element block).  Same arithmetic as convert.cu's
// dequantize_block_q4_1: (d, m) are FP32 and each value is a single F16 rounding of `d * nibble + m`.
static __device__ __forceinline__ void ggml_cuda_fattn_dequantize_q4_1_chunk(
        const char * const __restrict__ row, const int el, half2 * const __restrict__ dst) {
    static_assert(sizeof(block_q4_1) == 2*sizeof(half) + QK4_1/2, "bad block_q4_1");

    const int blk = el / QK4_1;
    const int off = el % QK4_1;
    const char * bp = row + (size_t) blk*sizeof(block_q4_1);

    half d_h, m_h;
    ggml_cuda_memcpy_1<sizeof(half), 2>(&d_h, bp);
    ggml_cuda_memcpy_1<sizeof(half), 2>(&m_h, bp + sizeof(half));
    const float d = __half2float(d_h);
    const float m = __half2float(m_h);

    // Elements 0..15 take the low nibble of qs[j], elements 16..31 the high nibble (interleaved).
    const int lo   = off < QK4_1/2;
    const int base = lo ? off : off - QK4_1/2;
    const uint8_t * qs = (const uint8_t *) (bp + 2*sizeof(half));

#pragma unroll
    for (int l = 0; l < GGML_CUDA_FA_Q8_CHUNK/2; ++l) {
        const uint8_t b0 = qs[base + 2*l + 0];
        const uint8_t b1 = qs[base + 2*l + 1];
        const float v0 = d * (lo ? (b0 & 0x0F) : (b0 >> 4)) + m;
        const float v1 = d * (lo ? (b1 & 0x0F) : (b1 >> 4)) + m;
        dst[l] = make_half2(__float2half(v0), __float2half(v1));
    }
}

// q5_0: as q4_1 minus the `m` term, but each element carries a 5th bit from `qh`.  The chunk at `off`
// covers elements off..off+7, and for BOTH halves the 5th bit of element e is bit e of qh (element
// e >= 16 takes bit e of qh, because its iqs = e - 16 indexes qh at iqs + 16).  Same arithmetic as
// dequantize.cuh's dequantize_q5_0 + convert.cu's cast: FP32 `(v - 16) * d`, one F16 rounding.
static __device__ __forceinline__ void ggml_cuda_fattn_dequantize_q5_0_chunk(
        const char * const __restrict__ row, const int el, half2 * const __restrict__ dst) {
    static_assert(sizeof(block_q5_0) == sizeof(half) + sizeof(uint32_t) + QK5_0/2, "bad block_q5_0");

    const int blk = el / QK5_0;
    const int off = el % QK5_0;
    const char * bp = row + (size_t) blk*sizeof(block_q5_0);

    half d_h;
    ggml_cuda_memcpy_1<sizeof(half), 2>(&d_h, bp);
    const float d = __half2float(d_h);

    uint32_t qh;
    ggml_cuda_memcpy_1<sizeof(uint32_t), 2>(&qh, bp + sizeof(half));

    const int lo   = off < QK5_0/2;
    const int base = lo ? off : off - QK5_0/2;
    // The 5th bit of element e is qh bit e in BOTH halves: the reference's low half is
    // `(qh >> j) << 4 & 0x10` (bit j) and its high half is `(qh >> (j + 12)) & 0x10`, which masks
    // bit 4 of the *shifted* value, i.e. qh bit j + 16 -- and the high half's element is j + 16.  So
    // `off + 2*l + i` (the element index within the block) is the bit index for either half; `base`
    // only selects the qs byte and the nibble.
    const uint8_t * qs = (const uint8_t *) (bp + sizeof(half) + sizeof(uint32_t));

#pragma unroll
    for (int l = 0; l < GGML_CUDA_FA_Q8_CHUNK/2; ++l) {
        const uint8_t b0 = qs[base + 2*l + 0];
        const uint8_t b1 = qs[base + 2*l + 1];
        const float f0 = (float) ((lo ? (b0 & 0x0F) : (b0 >> 4)) | (((qh >> (off + 2*l + 0)) & 1) << 4));
        const float f1 = (float) ((lo ? (b1 & 0x0F) : (b1 >> 4)) | (((qh >> (off + 2*l + 1)) & 1) << 4));
        dst[l] = make_half2(__float2half((f0 - 16.0f) * d), __float2half((f1 - 16.0f) * d));
    }
}

// q5_1: q5_0's 5th bit with q4_1's `(d, m)`: FP32 `f * d + m`, one F16 rounding (dequantize_q5_1).
static __device__ __forceinline__ void ggml_cuda_fattn_dequantize_q5_1_chunk(
        const char * const __restrict__ row, const int el, half2 * const __restrict__ dst) {
    static_assert(sizeof(block_q5_1) == 2*sizeof(half) + sizeof(uint32_t) + QK5_1/2, "bad block_q5_1");

    const int blk = el / QK5_1;
    const int off = el % QK5_1;
    const char * bp = row + (size_t) blk*sizeof(block_q5_1);

    half d_h, m_h;
    ggml_cuda_memcpy_1<sizeof(half), 2>(&d_h, bp);
    ggml_cuda_memcpy_1<sizeof(half), 2>(&m_h, bp + sizeof(half));
    const float d = __half2float(d_h);
    const float m = __half2float(m_h);

    uint32_t qh;
    ggml_cuda_memcpy_1<sizeof(uint32_t), 2>(&qh, bp + 2*sizeof(half));

    const int lo   = off < QK5_1/2;
    const int base = lo ? off : off - QK5_1/2;
    // See the q5_0 note: the 5th bit of element e is qh bit e in both halves.
    const uint8_t * qs = (const uint8_t *) (bp + 2*sizeof(half) + sizeof(uint32_t));

#pragma unroll
    for (int l = 0; l < GGML_CUDA_FA_Q8_CHUNK/2; ++l) {
        const uint8_t b0 = qs[base + 2*l + 0];
        const uint8_t b1 = qs[base + 2*l + 1];
        const float f0 = (float) ((lo ? (b0 & 0x0F) : (b0 >> 4)) | (((qh >> (off + 2*l + 0)) & 1) << 4));
        const float f1 = (float) ((lo ? (b1 & 0x0F) : (b1 >> 4)) | (((qh >> (off + 2*l + 1)) & 1) << 4));
        dst[l] = make_half2(__float2half(f0 * d + m), __float2half(f1 * d + m));
    }
}

// iq4_nl: a non-linear 16-entry codebook over the same interleaved nibble layout, `d * kvalues[i]`
// (dequantize.cuh's dequantize_iq4_nl).  Note the block is QK4_NL = 32 like the legacy types.
static __device__ __forceinline__ void ggml_cuda_fattn_dequantize_iq4_nl_chunk(
        const char * const __restrict__ row, const int el, half2 * const __restrict__ dst) {
    static_assert(sizeof(block_iq4_nl) == sizeof(half) + QK4_NL/2, "bad block_iq4_nl");

    const int blk = el / QK4_NL;
    const int off = el % QK4_NL;
    const char * bp = row + (size_t) blk*sizeof(block_iq4_nl);

    half d_h;
    ggml_cuda_memcpy_1<sizeof(half), 2>(&d_h, bp);
    const float d = __half2float(d_h);

    const int lo   = off < QK4_NL/2;
    const int base = lo ? off : off - QK4_NL/2;
    const uint8_t * qs = (const uint8_t *) (bp + sizeof(half));

#pragma unroll
    for (int l = 0; l < GGML_CUDA_FA_Q8_CHUNK/2; ++l) {
        const uint8_t b0 = qs[base + 2*l + 0];
        const uint8_t b1 = qs[base + 2*l + 1];
        const float v0 = d * (float) kvalues_iq4nl[lo ? (b0 & 0x0F) : (b0 >> 4)];
        const float v1 = d * (float) kvalues_iq4nl[lo ? (b1 & 0x0F) : (b1 >> 4)];
        dst[l] = make_half2(__float2half(v0), __float2half(v1));
    }
}

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_f16(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds_v) {

    const half2 * K_h2 = (const half2 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        __align__(16) half2 tmp[cpy_ne];
        ggml_cuda_memcpy_1<sizeof(tmp)>(tmp, K_h2 + k_KQ_0 + (threadIdx.x % nthreads)*cpy_ne);
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
#ifdef V_DOT2_F32_F16_AVAILABLE
            ggml_cuda_mad(sum,                tmp[k_KQ_1] , ((const half2  *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#else
            ggml_cuda_mad(sum, __half22float2(tmp[k_KQ_1]), ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    return sum;
}

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_bf16(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds_v) {

    const nv_bfloat162 * K_bf16 = (const nv_bfloat162 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        __align__(16) nv_bfloat162 tmp[cpy_ne];
        ggml_cuda_memcpy_1<sizeof(tmp)>(tmp, K_bf16 + k_KQ_0 + (threadIdx.x % nthreads)*cpy_ne);
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
#ifdef V_DOT2_F32_F16_AVAILABLE
            // FIXME replace macros in vector FA kernel with templating and use FP32 for BF16
            ggml_cuda_mad(sum, ggml_cuda_cast<float2>(tmp[k_KQ_1]), __half22float2(((const half2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]));
#else
            ggml_cuda_mad(sum, ggml_cuda_cast<float2>(tmp[k_KQ_1]), ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q4_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q4_0 * K_q4_0 = (const block_q4_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI4_0;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&v, K_q4_0[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];
        sum += __half2float(K_q4_0[ib].d) * (sumi*Q_ds.x - (8/QI8_1)*Q_ds.y);
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q4_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q4_1 * K_q4_1 = (const block_q4_1 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI4_1;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int)>(&v, K_q4_1[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 K_dm = __half22float2(K_q4_1[ib].dm);
        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += K_dm.x*Q_ds.x*sumi + K_dm.y*Q_ds.y/QI8_1;
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q5_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q5_0 * K_q5_0 = (const block_q5_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI5_0;
        const int iqs8  = k_KQ %  QI8_1;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&v, K_q5_0[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            int vh;
            ggml_cuda_memcpy_1<sizeof(int), 2>(&vh, K_q5_0[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += __half2float(K_q5_0[ib].d) * (sumi*Q_ds.x - (16/QI8_1)*Q_ds.y);
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q5_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q5_1 * K_q5_1 = (const block_q5_1 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI5_1;
        const int iqs8  = k_KQ %  QI8_1;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int)>(&v, K_q5_1[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            int vh;
            ggml_cuda_memcpy_1<sizeof(int)>(&vh, K_q5_1[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 K_dm = __half22float2(K_q5_1[ib].dm);
        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += K_dm.x*Q_ds.x*sumi + K_dm.y*Q_ds.y/QI8_1;
    }

    return sum;
}

// K side of the vector (per-(K,V)-pair) flash-attention kernel for iq4_nl.  Same index
//   arithmetic and scale handling as q4_0, but the 4-bit codes are values in kvalues_iq4nl
//   rather than q-8, so they must be expanded through the table before the dp4a (the same
//   idiom the mmvq iq4_nl kernel uses); there is no bias term to correct for.
template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_iq4_nl(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_iq4_nl * K_iq4_nl = (const block_iq4_nl *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI4_0;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&v, K_iq4_nl[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        // .x holds the looked-up values of the low nibbles (the four elements this
        //   thread's int covers), .y the high ones (the +QK4_NL/2 half), which `shift`
        //   has already zeroed here - so only .x is needed, exactly as many ints as the
        //   q4_0 vector dot feeds to dp4a.
        const int vq = get_int_from_table_16(v, kvalues_iq4nl).x;

        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(vq, u, 0);

        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];
        sum += __half2float(K_iq4_nl[ib].d) * sumi * Q_ds.x;
    }

    return sum;
}

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q8_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q8_0 * K_q8_0 = (const block_q8_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib  = k_KQ / QI8_0;
        const int iqs = k_KQ % QI8_0;

        int v;
        ggml_cuda_memcpy_1<sizeof(v), 2>(&v, K_q8_0[ib].qs + 4*iqs);

        const float2 * Q_ds = (const float2 *) Q_ds_v;
        const float Q_d = Q_ds[k_KQ_0/nthreads].x;

        sum += vec_dot_q8_0_q8_1_impl<float, 1>(&v, &Q_q8[k_KQ_0/nthreads], K_q8_0[ib].d, Q_d);
    }

    return sum;
}

template <typename Tds, int ni>
static __device__ __forceinline__ void quantize_q8_1_to_shared(
    const float * __restrict__ x, const float scale, int * __restrict__ yq32, void * __restrict__ yds) {

    float vals[sizeof(int)] = {0.0f};
#pragma unroll
    for (int l = 0; l < int(sizeof(int)); ++l) {
        vals[l] = (ni == WARP_SIZE || threadIdx.x < ni) ? scale * x[4*threadIdx.x + l] : 0.0f;
    }

    float amax = fabsf(vals[0]);
    float sum  = vals[0];
#pragma unroll
    for (int l = 1; l < int(sizeof(int)); ++l) {
        amax = fmaxf(amax, fabsf(vals[l]));
        sum += vals[l];
    }
#pragma unroll
    for (int mask = QI8_1/2; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, 32));
        sum +=             __shfl_xor_sync(0xFFFFFFFF, sum,  mask, 32);
    }

    const float d = amax / 127;
    int q32 = 0;
    int8_t * q8 = (int8_t *) &q32;

    if (d != 0.0f) {
#pragma unroll
        for (int l = 0; l < int(sizeof(int)); ++l) {
            q8[l] = roundf(vals[l] / d);
        }
    }

    yq32[threadIdx.x] = q32;
    if (threadIdx.x % QI8_1 == 0 && (ni == WARP_SIZE || threadIdx.x < ni)) {
        if (std::is_same<Tds, half2>::value) {
            ((half2  *) yds)[threadIdx.x/QI8_1] =  make_half2(d, sum);
        } else {
            ((float2 *) yds)[threadIdx.x/QI8_1] = make_float2(d, sum);
        }
    }
}

typedef void (*dequantize_V_t)(const void *, void *, const int64_t);

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_f16(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    if constexpr (std::is_same_v<T, half>) {
        ggml_cuda_memcpy_1<ne*sizeof(half)>(dst, (const half *) vx + i0);
    } else if constexpr (std::is_same_v<T, float>) {
        static_assert(ne % 2 == 0, "bad ne");
        __align__(16) half2 tmp[ne/2];
        ggml_cuda_memcpy_1<ne*sizeof(half)>(tmp, (const half *) vx + i0);
        float2 * dst_f2 = (float2 *) dst;
#pragma unroll
        for (int l = 0; l < ne/2; ++l) {
            dst_f2[l] = __half22float2(tmp[l]);
        }
    } else {
        static_assert(std::is_same_v<T, void>, "unsupported type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_bf16(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    static_assert(std::is_same_v<T, float>, "BF16 V dequantization only supports float output");
    static_assert(ne % 2 == 0, "bad ne");
    __align__(16) nv_bfloat162 tmp[ne/2];
    ggml_cuda_memcpy_1<ne*sizeof(nv_bfloat16)>(tmp, (const nv_bfloat16 *) vx + i0);
    float2 * dst_f2 = (float2 *) dst;
#pragma unroll
    for (int l = 0; l < ne/2; ++l) {
        dst_f2[l] = ggml_cuda_cast<float2>(tmp[l]);
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q4_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const int64_t ib    =  i0          /  QK4_0;
    const int     iqs   =  i0          % (QK4_0/2);
    const int     shift = (i0 % QK4_0) / (QK4_0/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
#if defined(GGML_USE_HIP)
    // Keep this VMEM read close to its packed-byte dequantization. Hoisting it too far
    // increases VGPR pressure substantially in some FlashAttention vector kernels.
    __builtin_amdgcn_sched_group_barrier(0x20, 1, 0);
#endif // defined(GGML_USE_HIP)
    q >>= 4*shift;
    q &= 0x0F0F0F0F;
    q = __vsub4(q, 0x08080808);

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * q8[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q4_1(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const int64_t ib    =  i0          /  QK4_1;
    const int     iqs   =  i0          % (QK4_1/2);
    const int     shift = (i0 % QK4_1) / (QK4_1/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 dm = x[ib].dm;
        const half2 d  = __half2half2( __low2half(dm));
        const half2 m  = __half2half2(__high2half(dm));

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float2 dm = __half22float2(x[ib].dm);

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = dm.x * q8[l] + dm.y;
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q5_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const int64_t ib    =  i0          /  QK5_0;
    const int     idq   =  i0          %  QK5_0;
    const int     iqs   =  i0          % (QK5_0/2);
    const int     shift = (i0 % QK5_0) / (QK5_0/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        int qh;
        ggml_cuda_memcpy_1<ne, 2>(&qh, x[ib].qh);
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    q = __vsub4(q, 0x10101010);

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * q8[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q5_1(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const int64_t ib    =  i0          /  QK5_1;
    const int     idq   =  i0          %  QK5_1;
    const int     iqs   =  i0          % (QK5_1/2);
    const int     shift = (i0 % QK5_1) / (QK5_1/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        int qh;
        ggml_cuda_memcpy_1<ne>(&qh, x[ib].qh);
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 dm = x[ib].dm;
        const half2 d  = __half2half2( __low2half(dm));
        const half2 m  = __half2half2(__high2half(dm));

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float2 dm = __half22float2(x[ib].dm);

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = dm.x * q8[l] + dm.y;
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

// iq4_nl keeps the q4_0/q5_0 nibble layout (nibble j of the block carries elements j and
//   j + QK4_NL/2) but maps each 4-bit code through kvalues_iq4nl instead of the q-8 offset
//   and stores no bias, so the index arithmetic below is q4_0's and only the value map
//   differs.  kvalues_iq4nl is a __device__ table (ggml-common.h), already read by the mmvq
//   and mmq iq4_nl kernels.
template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_iq4_nl(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_iq4_nl * x = (const block_iq4_nl *) vx;

    const int64_t ib    =  i0            /  QK4_NL;
    const int     iqs   =  i0            % (QK4_NL/2);
    const int     shift = (i0 % QK4_NL) / (QK4_NL/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    const uint8_t * q8 = (const uint8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(kvalues_iq4nl[q8[l0 + 0]], kvalues_iq4nl[q8[l0 + 1]]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * kvalues_iq4nl[q8[l]];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q8_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const int64_t ib  = i0 / QK8_0;
    const int     iqs = i0 % QK8_0;

    static_assert(ne % 2 == 0, "bad ne");
    int8_t qs[ne];
    ggml_cuda_memcpy_1<ne, 2>(qs, x[ib].qs + iqs);

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same<T, half>::value) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(qs[l0 + 0], qs[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same<T, float>::value) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * qs[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "unsupported type");
    }
}

template <ggml_type type_K, int D, int nthreads>
constexpr __device__ vec_dot_KQ_t get_vec_dot_KQ() {
    if constexpr (type_K == GGML_TYPE_F16) {
        return vec_dot_fattn_vec_KQ_f16<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q4_0) {
        return vec_dot_fattn_vec_KQ_q4_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q4_1) {
        return vec_dot_fattn_vec_KQ_q4_1<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q5_0) {
        return vec_dot_fattn_vec_KQ_q5_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q5_1) {
        return vec_dot_fattn_vec_KQ_q5_1<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q8_0) {
        return vec_dot_fattn_vec_KQ_q8_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_IQ4_NL) {
        return vec_dot_fattn_vec_KQ_iq4_nl<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_BF16) {
        return vec_dot_fattn_vec_KQ_bf16<D, nthreads>;
    } else {
        static_assert(type_K == -1, "bad type");
        return nullptr;
    }
}

template <ggml_type type_V, typename T, int ne>
constexpr __device__ dequantize_V_t get_dequantize_V() {
    if constexpr (type_V == GGML_TYPE_F16) {
        return dequantize_V_f16<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q4_0) {
        return dequantize_V_q4_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q4_1) {
        return dequantize_V_q4_1<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q5_0) {
        return dequantize_V_q5_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q5_1) {
        return dequantize_V_q5_1<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q8_0) {
        return dequantize_V_q8_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_IQ4_NL) {
        return dequantize_V_iq4_nl<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_BF16) {
        return dequantize_V_bf16<float, ne>;
    } else {
        static_assert(type_V == -1, "bad type");
        return nullptr;
    }
}

template <int ncols1>
__launch_bounds__(FATTN_KQ_STRIDE/2, 1)
static __global__ void flash_attn_mask_to_KV_max(
        const half2 * mask_ptr, int * KV_max_ptr, const int ne30, const int64_t s31, const int64_t s33) {
    const half2 * GGML_CUDA_RESTRICT mask   = mask_ptr;
    int         * GGML_CUDA_RESTRICT KV_max = KV_max_ptr;

    const int ne31     = gridDim.x;
    const int tid      = threadIdx.x;
    const int sequence = blockIdx.y;
    const int jt       = blockIdx.x;

    mask += sequence*s33 + jt*ncols1*s31;

    __shared__ int buf_iw[WARP_SIZE];
    if (tid < WARP_SIZE) {
        buf_iw[tid] = 1;
    }
    ggml_cuda_pdl_sync();
    __syncthreads();

    int KV_max_sj = (ne30 - 1) * FATTN_KQ_STRIDE;
    for (; KV_max_sj >= 0; KV_max_sj -= FATTN_KQ_STRIDE) {
        int all_inf = 1;

#pragma unroll
        for (int j = 0; j < ncols1; ++j) {
            const float2 tmp = __half22float2(mask[j*s31 + KV_max_sj/2 + tid]);
            all_inf = all_inf && int(isinf(tmp.x)) && int(isinf(tmp.y));
        }

        all_inf = warp_reduce_all(all_inf);
        if (tid % WARP_SIZE == 0) {
            buf_iw[tid / WARP_SIZE] = all_inf;
        }
        __syncthreads();
        all_inf = buf_iw[tid % WARP_SIZE];
        __syncthreads();
        all_inf = warp_reduce_all(all_inf);

        if (!all_inf) {
            break;
        }
    }

    // If the break in the loop was not triggered, KV_max_sj is now -FATTN_KQ_STRIDE.
    // If the break was triggered it's the lower edge of the tile with the first non-masked values.
    // In either case, walk back the decrementation by FATTN_KQ_STRIDE.
    KV_max_sj += FATTN_KQ_STRIDE;

    if (threadIdx.x != 0) {
        return;
    }

    KV_max[sequence*ne31 + jt] = KV_max_sj;
}

// Packed-mask counterpart of flash_attn_kq_derived_blocks (issue #48, multi-sequence prefill): a
// 1 bit per (Q stream, query tile of ncols1 rows, FATTN_KQ_STRIDE-cell group) classification of
// whether every mask entry of that tile/group is -INF.  The packed path cannot use the derived form
// because a continuous-batching ubatch can hold more than one sequence; a batch-wide test would not
// help there (each sequence's cells are visible to its own rows), so the classification is per tile.
// One warp per (stream, tile, group): all rows of the tile are scanned and `all_inf` is only kept
// when every value is -INF, so a marked group contributes exact zeros to the online softmax.
template <int ncols1>
static __global__ void flash_attn_mask_to_KV_blocks(
        const half * __restrict__ mask, uint32_t * __restrict__ blocks,
        const int n_kv, const int n_tps, const int64_t s31, const int64_t s33,
        const int n_tiles, const int n_groups, const int n_streams, const int n_mask_streams) {
    const int warp = (blockIdx.x*blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;

    const int total = n_streams*n_tiles*n_groups; // one warp per (stream, query tile, group)
    if (warp >= total) {
        return;
    }

    const int group  = warp % n_groups;
    const int tile   = (warp / n_groups) % n_tiles;
    const int stream = warp / (n_groups*n_tiles);

    const int j0 = tile*ncols1;
    const int j1 = min(n_tps, j0 + ncols1);
    const int k0 = group*FATTN_KQ_STRIDE;
    const int k1 = min(n_kv, k0 + FATTN_KQ_STRIDE);

    const half * m = mask + (int64_t) (stream % n_mask_streams)*s33;

    int all_inf = 1;
    for (int j = j0; j < j1; ++j) {
        for (int k = k0 + lane; k < k1; k += WARP_SIZE) {
            all_inf = all_inf && (__half2float(m[(int64_t) j*s31 + k]) == -INFINITY);
        }
    }
    all_inf = warp_reduce_all(all_inf);

    // the launcher zeroes the bitmap; distinct (tile, group) pairs can share a word, so OR the bit in
    if (lane == 0 && all_inf) {
        const int words_per_tile = (n_groups + 31)/32;
        atomicOr(blocks + (stream*n_tiles + tile)*words_per_tile + (group >> 5), 1u << (group & 31));
    }
}

// One thread per 32-bit group word: bits set in the word mark the FATTN_KQ_STRIDE-cell groups whose
// cells are all invisible to the whole batch.  A cell is invisible when it is INT32_MIN (empty or
// another sequence's) or its position is outside the batch's [lo_min, hi_max] window; that is
// exactly the derived predicate set_input_kq_derived publishes, so a marked group is a sequence of
// exact -INF logits.  The window is the batch-wide union, which is deliberately conservative for a
// multi-tile batch: a group a single query tile cannot see is not marked unless no tile can.
//
// The window is derived here, on the device, rather than on the host: the launcher must not
// dereference these tensors from the CPU.  The derived inputs are host graph inputs that the backend
// scheduler copies to the device, so `tok_lo`/`tok_hi` seen by the launcher are device pointers; a
// host read of that copy is an access violation wherever the allocation is not CPU-mapped
// (Windows/WDDM, issue #53) and a race against the in-flight copy everywhere else.
//
// The window reduction is cooperative (all threads of the block, then a shared tree), not a serial
// scan by the word-owning thread: the token count is the ubatch size and a single-thread scan would
// add up to O(n_words*n_tps) serial global loads to a prefill-path kernel that already scans only
// n_kv cells.  The reduction has to run before the early return, so it is issued first and every
// thread reaches the block barrier.
constexpr int FATTN_KQ_BLOCK_THREADS = 256;

static __global__ void flash_attn_kq_derived_blocks(
        const int * __restrict__ cell_pos,
        const int * __restrict__ tok_lo, const int * __restrict__ tok_hi, const int n_tps,
        uint32_t * __restrict__ blocks,
        const int n_kv, const int n_groups) {
    __shared__ int s_lo[FATTN_KQ_BLOCK_THREADS];
    __shared__ int s_hi[FATTN_KQ_BLOCK_THREADS];
    const int tid = threadIdx.x;

    int lo_min = INT32_MAX;
    int hi_max = INT32_MIN;
    for (int i = tid; i < n_tps; i += blockDim.x) {
        lo_min = min(lo_min, tok_lo[i]);
        hi_max = max(hi_max, tok_hi[i]);
    }
    s_lo[tid] = lo_min;
    s_hi[tid] = hi_max;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            s_lo[tid] = min(s_lo[tid], s_lo[tid + s]);
            s_hi[tid] = max(s_hi[tid], s_hi[tid + s]);
        }
        __syncthreads();
    }
    lo_min = s_lo[0];
    hi_max = s_hi[0];

    const int word = blockIdx.x*blockDim.x + tid;
    if (word >= (n_groups + 31)/32) {
        return;
    }

    uint32_t bits = 0;
#pragma unroll 1
    for (int b = 0; b < 32; ++b) {
        const int g = word*32 + b;
        if (g >= n_groups) {
            break;
        }

        const int j0 = g*FATTN_KQ_STRIDE;
        const int j1 = min(n_kv, j0 + FATTN_KQ_STRIDE);

        bool skip = true;
        for (int j = j0; j < j1; ++j) {
            const int p = cell_pos[j];
            if (p != INT32_MIN && p >= lo_min && p <= hi_max) {
                skip = false;
                break;
            }
        }
        if (skip) {
            bits |= 1u << b;
        }
    }

    blocks[word] = bits;
}

void ggml_cuda_flash_attn_ext_compact_mask(
        const ggml_tensor * mask, int32_t * indices, int32_t * counts, int32_t n_queries, int32_t ncols1, int32_t n_kv_max, cudaStream_t stream);

template<int D, int ncols1, int ncols2> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_stream_k_fixup_uniform(
        float * dst_ptr,
        const float2 * dst_fixup_ptr,
        const int ne01, const int ne02,
        const int ne12, const int nblocks_stream_k,
        const int gqa_ratio,
        const int blocks_per_tile,
        const uint3 fd_iter_j_z_ne12,
        const uint3 fd_iter_j_z,
        const uint3 fd_iter_j) {
    constexpr int ncols = ncols1*ncols2;
    ggml_cuda_pdl_lc();
    float        * GGML_CUDA_RESTRICT dst       = dst_ptr;
    const float2 * GGML_CUDA_RESTRICT dst_fixup = dst_fixup_ptr;

    const int tile_idx = blockIdx.x; // One block per output tile.
    const int j        = blockIdx.y;
    const int c        = blockIdx.z;
    const int jc       = j*ncols2 + c;
    const int tid      = threadIdx.x;

    // nblocks_stream_k is a multiple of ntiles_dst (== gridDim.x), so each tile gets the same number of blocks.
    const int b_first = tile_idx * blocks_per_tile;
    const int b_last  = b_first + blocks_per_tile - 1;

    const float * dst_fixup_data = ((const float *) dst_fixup) + nblocks_stream_k*(2*2*ncols);

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
    const uint2 dm0 = fast_div_modulo(tile_idx, fd_iter_j_z_ne12);
    const uint2 dm1 = fast_div_modulo(dm0.y,    fd_iter_j_z);
    const uint2 dm2 = fast_div_modulo(dm1.y,    fd_iter_j);

    const int sequence = dm0.x;
    const int z_KV     = dm1.x;
    const int zt_gqa   = dm2.x;
    const int jt       = dm2.y;

    const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    if (jt*ncols1 + j >= ne01 || zt_gqa*ncols2 + c >= gqa_ratio) {
        return;
    }

    dst += sequence*ne02*ne01*D + jt*ne02*(ncols1*D) + zt_Q*D + (j*ne02 + c)*D + tid;

    ggml_cuda_pdl_sync();
    // Load the partial result that needs a fixup
    float dst_val = *dst;
    float max_val;
    float rowsum;
    {
        const float2 tmp = dst_fixup[b_last*ncols + jc];
        max_val = tmp.x;
        rowsum  = tmp.y;
    }

    // Combine with all previous blocks in this tile.
    for (int bidx = b_last - 1; bidx >= b_first; --bidx) {
        const float dst_add = dst_fixup_data[bidx*ncols*D + jc*D + tid];

        const float2 tmp = dst_fixup[(nblocks_stream_k + bidx)*ncols + jc];

        const float max_val_new = fmaxf(max_val, tmp.x);

        const float diff_val = max_val - max_val_new;
        const float diff_add = tmp.x   - max_val_new;

        const float scale_val = diff_val >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_val) : 0.0f;
        const float scale_add = diff_add >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_add) : 0.0f;

        dst_val = scale_val*dst_val + scale_add*dst_add;
        rowsum  = scale_val*rowsum  + scale_add*tmp.y;

        max_val = max_val_new;
    }

    // Write back final result:
    *dst = dst_val / rowsum;
}

// General fixup kernel for the case where the number of blocks per tile is not uniform across tiles
// (blocks_num.x not a multiple of ntiles_dst)
template <int D, int ncols1, int ncols2> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_stream_k_fixup_general(
        float * dst_ptr,
        const float2 * dst_fixup_ptr,
        const int ne01, const int ne02,
        const int gqa_ratio,
        const int total_work,
        const uint3 fd_iter_k_j_z_ne12,
        const uint3 fd_iter_k_j_z,
        const uint3 fd_iter_k_j,
        const uint3 fd_iter_k) {
    float        * GGML_CUDA_RESTRICT dst       = dst_ptr;
    const float2 * GGML_CUDA_RESTRICT dst_fixup = dst_fixup_ptr;
    constexpr int ncols = ncols1*ncols2;

    const int bidx0 = blockIdx.x;
    const int j     = blockIdx.y;
    const int c     = blockIdx.z;
    const int jc    = j*ncols2 + c;
    const int tid   = threadIdx.x;

    const float * dst_fixup_data = ((const float *) dst_fixup) + gridDim.x*(2*2*ncols);

    const int kbc0      = int64_t(bidx0 + 0)*total_work / gridDim.x;
    const int kbc0_stop = int64_t(bidx0 + 1)*total_work / gridDim.x;

    const bool did_not_have_any_data   = kbc0 == kbc0_stop;
    const bool wrote_beginning_of_tile = fastmodulo(kbc0, fd_iter_k) == 0;
    const bool did_not_write_last      = fastdiv(kbc0, fd_iter_k) == fastdiv(kbc0_stop, fd_iter_k) && fastmodulo(kbc0_stop, fd_iter_k) != 0;
    if (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
        return;
    }

    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
    const uint2 dm0 = fast_div_modulo(kbc0, fd_iter_k_j_z_ne12);
    const uint2 dm1 = fast_div_modulo(dm0.y, fd_iter_k_j_z);
    const uint2 dm2 = fast_div_modulo(dm1.y, fd_iter_k_j);
    const uint2 dm3 = fast_div_modulo(dm2.y, fd_iter_k);

    const int sequence = dm0.x;
    const int z_KV     = dm1.x;
    const int zt_gqa   = dm2.x;
    const int jt       = dm3.x;

    const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2; // Global Q head start index.

    if (jt*ncols1 + j >= ne01 || zt_gqa*ncols2 + c >= gqa_ratio) {
        return;
    }

    dst += sequence*ne02*ne01*D + jt*ne02*(ncols1*D) + zt_Q*D + (j*ne02 + c)*D + tid;

    // Load the partial result that needs a fixup:
    float dst_val = 0.0f;
    float max_val = 0.0f;
    float rowsum  = 0.0f;
    ggml_cuda_pdl_sync();
    {
        dst_val = *dst;

        const float2 tmp = dst_fixup[bidx0*ncols + jc];
        max_val = tmp.x;
        rowsum  = tmp.y;
    }

    // Iterate over previous blocks and compute the combined results.
    // All CUDA blocks that get here must have a previous block that needs a fixup.
    const int tile_kbc0 = fastdiv(kbc0, fd_iter_k);
    int bidx = bidx0 - 1;
    int kbc_stop = kbc0;
    while(true) {
        const int kbc = int64_t(bidx)*total_work / gridDim.x;
        if (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            continue;
        }

        const float dst_add = dst_fixup_data[bidx*ncols*D + jc*D + tid];

        const float2 tmp = dst_fixup[(gridDim.x + bidx)*ncols + jc];

        // Scale the current and new value accumulators depending on the max. values.
        const float max_val_new = fmaxf(max_val, tmp.x);

        const float diff_val = max_val - max_val_new;
        const float diff_add = tmp.x   - max_val_new;

        const float scale_val = diff_val >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_val) : 0.0f;
        const float scale_add = diff_add >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_add) : 0.0f;

        dst_val = scale_val*dst_val + scale_add*dst_add;
        rowsum  = scale_val*rowsum  + scale_add*tmp.y;

        max_val = max_val_new;

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        if (fastmodulo(kbc, fd_iter_k) == 0 || fastdiv(kbc, fd_iter_k) < tile_kbc0) {
            break;
        }
        bidx--;
        kbc_stop = kbc;
    }

    // Write back final result:
    *dst = dst_val / rowsum;
}

template<int D> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_combine_results(
        const float  * VKQ_parts_ptr,
        const float2 * VKQ_meta_ptr,
        float * dst_ptr,
        const int parallel_blocks) {
    ggml_cuda_pdl_lc();
    const float  * GGML_CUDA_RESTRICT VKQ_parts = VKQ_parts_ptr;
    const float2 * GGML_CUDA_RESTRICT VKQ_meta  = VKQ_meta_ptr;
    float        * GGML_CUDA_RESTRICT dst       = dst_ptr;
    // Dimension 0: threadIdx.x
    // Dimension 1: blockIdx.x
    // Dimension 2: blockIdx.y
    // Dimension 3: blockIdx.z
    // Memory layout is permuted with [0, 2, 1, 3]

    const int ne01 = gridDim.x;
    const int ne02 = gridDim.y;

    const int col      = blockIdx.x;
    const int head     = blockIdx.y;
    const int sequence = blockIdx.z;

    const int j_dst_unrolled = (sequence*ne01 + col)*ne02 + head;

    VKQ_parts += j_dst_unrolled * parallel_blocks*D;
    VKQ_meta  += j_dst_unrolled * parallel_blocks;
    dst       += j_dst_unrolled *                 D;

    const int tid = threadIdx.x;
    __builtin_assume(tid < D);

    extern __shared__ float2 meta[];
    ggml_cuda_pdl_sync();
    for (int i = tid; i < 2*parallel_blocks; i += D) {
        ((float *) meta)[i] = ((const float *)VKQ_meta) [i];
    }

    __syncthreads();

    float kqmax = meta[0].x;
    for (int l = 1; l < parallel_blocks; ++l) {
        kqmax = max(kqmax, meta[l].x);
    }

    float VKQ_numerator   = 0.0f;
    float VKQ_denominator = 0.0f;
    for (int l = 0; l < parallel_blocks; ++l) {
        const float KQ_max_scale = expf(meta[l].x - kqmax);

        VKQ_numerator   += KQ_max_scale * VKQ_parts[l*D + tid];
        VKQ_denominator += KQ_max_scale * meta[l].y;
    }

    dst[tid] = VKQ_numerator / VKQ_denominator;
}

// Decode/verify band (n_q <= 8) on RDNA4: the whole GQA group is folded into one block
// (ncols2 = 8) via the WMMA kernel instead of the tile kernel's ncols2 = 2, which fetches and
// dequantizes every K/V element once per head pair (3x for GQA 6).  This is the default; the env var
// is the opt-out (maintainer policy: a beneficial feature is on by default).  One ncols1 for the whole
// band keeps a single kernel config, and launch_fattn fixes the round-robin KV split per output tile
// (see there), so n_q = 1 and every verify width reduce identically (GREEDY-PURITY band invariant).
// GGML_HIP_FA_BAND_WMMA=0 disables the band, 2/4 force ncols1 (any other non-zero value means 4).
//
// The band's ncols1: the wide 4-column tile is the verify-width optimum for the native-quantized
// K/V types, whose dequantization dominates and hides the columns an n_q = 1 decode leaves unused.
// The 2-byte types (f16, and bf16 when its native arm is on) have no dequantization, so those
// unused columns are a much larger fraction of the n_q = 1 cost: measured on gfx1201 (head 256,
// GQA 6, kv 102400) the 2-column tile is 747 us at n_q = 1 vs 967 for 4 columns (and 861 vs 970 at
// n_q = 4) while the verify widths stay within noise.
static inline int ggml_cuda_fattn_band_wmma_ncols1_env() {
    static const int v = []() {
        const char * env = getenv("GGML_HIP_FA_BAND_WMMA");
        return env != nullptr ? atoi(env) : -1;
    }();
    return v; // -1 = unset, 0 = disabled, otherwise the forced value
}

static inline bool ggml_cuda_fattn_band_wmma_enabled() {
    return ggml_cuda_fattn_band_wmma_ncols1_env() != 0;
}

// The 2-byte K/V types share the band's light per-iteration variant (see the ncols1 and split
// helpers).  bf16 only reaches the band through its opt-in native arm, where K->type is BF16.
static inline bool ggml_cuda_fattn_band_wmma_two_byte(const ggml_tensor * dst) {
    const ggml_tensor * K = dst != nullptr ? dst->src[1] : nullptr;
    return K != nullptr && (K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_BF16);
}

static inline int ggml_cuda_fattn_band_wmma_ncols1(const ggml_tensor * dst) {
    const int v = ggml_cuda_fattn_band_wmma_ncols1_env();
    if (v > 0) {
        return v == 2 ? 2 : 4;
    }
    return ggml_cuda_fattn_band_wmma_two_byte(dst) ? 2 : 4;
}

// Blocks per output tile in the band (the interleave period P, see launch_fattn).  0 = one block per
// CU (nsm); GGML_HIP_FA_BAND_WMMA_SPLIT overrides it for tuning.  Whatever the source, it must not
// depend on n_q or on the KV length, or decode and verify would split differently.  The 2-byte types
// do less work per KV iteration, so the P partials' fixup is a larger fraction of their cost; a
// smaller P measured best for them on gfx1201 (nsm 32) across 16k..200k, while the quantized types
// keep nsm.  Expressed as a fraction of nsm so a smaller RDNA4 part scales down with it.
static inline int ggml_cuda_fattn_band_wmma_split(const ggml_tensor * dst) {
    static const int env_split = []() {
        const char * env = getenv("GGML_HIP_FA_BAND_WMMA_SPLIT");
        const int v = env ? atoi(env) : 0;
        return v > 0 ? v : 0;
    }();
    if (env_split > 0) {
        return env_split;
    }
    if (!ggml_cuda_fattn_band_wmma_two_byte(dst)) {
        return 0;
    }
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    return std::max(2, (3*nsm)/4);
}

// Self-contained band predicate shared by the kernel chooser, the ncols dispatcher and launch_fattn,
// so the three can never disagree (a disagreement would launch the WMMA kernel with ncols2 != 8,
// whose band fast path is compiled out and whose stream-k split is query-width dependent).
static inline bool ggml_cuda_fattn_band_wmma_applies(const int cc, const ggml_tensor * dst) {
    if (!ggml_cuda_fattn_band_wmma_enabled()) {
        return false;
    }
    if (!GGML_CUDA_CC_IS_RDNA4(cc) || !amd_wmma_available(cc)) {
        return false;
    }

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    float logit_softcap = 0.0f;
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    // The same GQA-optimization / alignment conditions the GQA kernels require: a mask is present,
    // no ALiBi, the KV is FATTN_KQ_STRIDE-aligned, and every non-quantized operand is 16-byte aligned.
    bool gqa_opt = (mask != nullptr || dst->src[5] != nullptr) && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                gqa_opt = false;
                break;
            }
        }
    }

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    // Every K/V type the MMA kernel can read natively (q8_0, bf16, q4_0/q4_1/q5_0/q5_1/iq4_nl) is
    // instruction-issue bound in the tile kernel, so the 3x re-fetch matters for all of them.  F16
    // has no native read (NONE), but it is only DRAM bound at n_q = 1: at verify widths the pinned
    // tile config (ncols2 = 2 for GQA 6) re-reads and re-stages every K/V element three times per
    // query row, so the band's single fold is the width-pure answer there too (issue #45 follow-up,
    // reported by @DanoPTT).  bf16 reaches the band through its native arm (kv_native = BF16), which
    // is the default since the r6 flip; it is deliberately NOT banded while it is staged, where the
    // whole-cache conversion dominates (that staged case only appears under GGML_CUDA_FA_KV_NATIVE=0).
    const int  kv_native = ggml_cuda_fattn_kv_native_type(K);
    const bool kv_quant  = kv_native != FATTN_KV_NATIVE_NONE && kv_native == ggml_cuda_fattn_kv_native_type(V);
    const bool kv_f16    = K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16;
    const bool kv_ok     = kv_quant || kv_f16;

    return gqa_opt && Q->ne[1] <= 8 && Q->ne[3] == 1 && Q->ne[0] == 256 && V->ne[0] == 256 &&
        gqa_ratio > 4 && gqa_ratio <= 8 && logit_softcap == 0.0f && kv_ok;
}

template <int DV, int ncols1, int ncols2>
void launch_fattn(
    ggml_backend_cuda_context & ctx, ggml_tensor * dst, fattn_kernel_t fattn_kernel, const int nwarps, const size_t nbytes_shared,
    const int nbatch_fa, const bool need_f16_K, const bool need_f16_V, const bool stream_k, const bool use_sparse,
    // The native K/V type the kernel will actually read for BOTH operands when it has a single K/V
    // type (the tile and vec kernels), or FATTN_KV_NATIVE_PER_OPERAND for the MMA kernel, which reads
    // each operand natively on its own.  The launcher must not skip the F16 staging of an operand the
    // kernel will not read natively: the tile kernel's type is fixed by its instantiation, so a mixed
    // K/V pair (e.g. K=q4_0, V=f16 -- reachable through test-backend-ops even though llama.cpp
    // rejects mixed caches) fell back to the F16 tile while the launcher still skipped K's staging,
    // and the kernel read raw q4_0 bytes as F16 (NaN).  FATTN_KV_NATIVE_NONE forces staging.
    const int kv_native_kernel = FATTN_KV_NATIVE_PER_OPERAND,
    const int warp_size = WARP_SIZE
) {
    constexpr int ncols = ncols1 * ncols2;

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const bool V_is_K_view = V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    // V3 derived kq mask: present instead of the mask tensor above.
    const ggml_tensor * cell_pos = dst->src[5];
    const ggml_tensor * tok_lo   = dst->src[6];
    const ggml_tensor * tok_hi   = dst->src[7];

    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    GGML_ASSERT(Q->nb[0] == ggml_element_size(Q));
    GGML_ASSERT(K->nb[0] == ggml_element_size(K));
    GGML_ASSERT(V->nb[0] == ggml_element_size(V));

    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t main_stream = ctx.stream();
    const int id  = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[id].cc;
    const int nsm = ggml_cuda_info().devices[id].nsm;

    // V4 / block 15: the MMA kernel can read a q8_0 (dequantize) or bf16 (convert) K/V cache while
    // staging its tiles, in which case the F16 staging copy (and the whole-cache conversion pass) is
    // skipped for that operand.  The predicates are shared with
    // ggml_cuda_flash_attn_ext_get_alloc_size, which sizes the node's scratch, so the two cannot
    // disagree on whether the scratch exists.
    const int  kv_native_K  = !need_f16_K ? FATTN_KV_NATIVE_NONE :
        (kv_native_kernel != FATTN_KV_NATIVE_PER_OPERAND ? kv_native_kernel : ggml_cuda_fattn_kv_native_type(K));
    const int  kv_native_V  = !need_f16_V ? FATTN_KV_NATIVE_NONE :
        (kv_native_kernel != FATTN_KV_NATIVE_PER_OPERAND ? kv_native_kernel :
         (V_is_K_view ? kv_native_K : ggml_cuda_fattn_kv_native_type(V)));

    // The native read is a decode/verify win: it removes the whole-cache F16 conversion that a
    // quantized source otherwise pays on *every* step (cost ~ n_kv, so it grows with depth) at the
    // price of dequantizing each tile in-kernel (cost ~ n_q * n_kv / ncols, amortised at prefill but
    // not at decode).  A prefill therefore stages instead -- the conversion is paid once for many
    // query rows and the tiles feed the cp_async pipeline -- and only the <= 8-row band reads the
    // raw cache.  The staging scratch for a native-capable operand is deliberately NOT part of the
    // node allocation (see ggml_cuda_flash_attn_ext_get_alloc_size, whose scratch is sized for the
    // reserve graph's n_kv = n_ctx), so it comes from the per-context arena instead; that is safe
    // because a multi-token graph is never CUDA-graph captured (see the prefill skip in
    // ggml_backend_cuda_graph_compute), while the captured decode graph is native and needs no
    // scratch at all.  A very deep prefill can bound the transient with GGML_CUDA_FA_STAGE_MAX_MB
    // (MiB per operand, 0 = unbounded); above it the native read is used.
    const size_t stage_max_bytes = ggml_cuda_fattn_stage_max_bytes();
    // Per-operand static cap: a transient larger than the cap is not staged at all (native read).
    const size_t stage_bytes_K = (size_t) ggml_nelements(K)*sizeof(half);
    const size_t stage_bytes_V = (size_t) ggml_nelements(V)*sizeof(half);
    const bool stage_cap_K = stage_max_bytes == 0 || stage_bytes_K <= stage_max_bytes;
    const bool stage_cap_V = stage_max_bytes == 0 || stage_bytes_V <= stage_max_bytes;
    // Prefill stages only where that actually wins.  On RDNA4/RDNA3_0 the whole-prefix F16 conversion
    // is paid once per ubatch and the tiles then feed the cp_async pipeline, which beats
    // re-dequantizing each tile once per query block (gfx1201 q8_0: pp150k 691 vs 661 native, pp32k
    // 1087 vs 1073).  RDNA3_5 (Strix Halo, unified LPDDR5) is the other way round at every depth, the
    // gap growing with it (gfx1151 9B q8_0 pp16k 1405 vs 1410, pp20k 1359 vs 1369, pp32k 1253 vs
    // 1266, pp65k 1043 vs 1059), so it keeps the native read at prefill too.
    const bool prefill_stages = !GGML_CUDA_CC_IS_RDNA3_5(cc);
    const bool native_width = Q->ne[1] <= 8 || !prefill_stages;

    // A native-capable operand that stages takes its scratch from the per-context, per-stream arena
    // rather than the generic CUDA pool.  The pool grows by exact fit and *retains* every distinct
    // size (the leg pool caches up to 256 buffers), and this request grows with the prefix, so a
    // long prefill would leave ~150 distinct buffers cached; the arena bounds the retained memory to
    // ~1.25x the largest requested size.  (The growth policy itself is not a performance lever --
    // measured pp150k 691.0 with an exact-fit realloc vs 691.4 with 25% growth.)
    //
    // The arena is grown on demand and its growth can fail when the device is nearly full -- e.g. a
    // llama-server --fit run whose target left less free memory than the deep-prefill transient needs
    // (the scratch is deliberately outside the compute-graph reserve, so the fit does not count it).
    // Rather than abort on that OOM, fall back to the native read for the operand(s) that would have
    // been staged: it is the same arithmetic the decode/verify band already uses, so this is a
    // prefill slowdown, never a correctness change, and it keeps the run inside the memory the fit
    // reserved.  The request covers both operands because they share the one arena buffer.
    const bool stage_wants_K = !native_width && stage_cap_K && kv_native_K != FATTN_KV_NATIVE_NONE;
    const bool stage_wants_V = !native_width && stage_cap_V && kv_native_V != FATTN_KV_NATIVE_NONE && !V_is_K_view;
    const size_t stage_req = (stage_wants_K ? stage_bytes_K : 0) + (stage_wants_V ? stage_bytes_V : 0);
    char * stage_buf = stage_req != 0 ? (char *) ctx.fattn_stage_try_get(ctx.curr_stream_no, stage_req) : nullptr;
    const bool stage_ok = stage_buf != nullptr;

    const bool use_native_K = (native_width || !stage_ok || !stage_cap_K) && kv_native_K != FATTN_KV_NATIVE_NONE;
    // A V that is a view of K is the same cache data: it must follow K's staging decision rather than
    // take its own (stage_V below is then false and K_f16_stage is reused).
    const bool use_native_V = V_is_K_view ? use_native_K :
        ((native_width || !stage_ok || !stage_cap_V) && kv_native_V != FATTN_KV_NATIVE_NONE);
    const bool stage_K = kv_native_K != FATTN_KV_NATIVE_NONE && !use_native_K;
    const bool stage_V = kv_native_V != FATTN_KV_NATIVE_NONE && !use_native_V && !V_is_K_view;

    // The kernel reads what the launcher staged: when an operand is staged (a native-capable type
    // above the decode/verify band), it must be told FATTN_KV_NATIVE_NONE so it reads the F16 copy
    // instead of the raw cache (otherwise the staging pass is paid *and* the tiles are dequantized).
    const int kv_native_kernel_K = use_native_K ? kv_native_K : FATTN_KV_NATIVE_NONE;
    const int kv_native_kernel_V = use_native_V ? kv_native_V : FATTN_KV_NATIVE_NONE;

    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra =
        ggml_cuda_flash_attn_ext_get_f16_extra_data(KQV,
            need_f16_K && !use_native_K && !stage_K,
            need_f16_V && !use_native_V && !stage_V);

    half * K_f16_stage = nullptr;
    half * V_f16_stage = nullptr;
    if (stage_K || stage_V) {
        GGML_ASSERT(stage_buf != nullptr);
        const size_t nK = stage_K ? stage_bytes_K : 0;
        K_f16_stage = (half *) stage_buf;
        V_f16_stage = (half *) (stage_buf + nK);
    }

    // The pool buffers are freed in reverse declaration order, and the VMM pool (GGML_HIP_NO_VMM=OFF / CUDA)
    // requires that to be the reverse of the allocation order: declare them in the order they are allocated
    // (KV_max, kq_blocks, then dst_tmp / dst_tmp_meta).  kq_blocks used to be declared last although it is
    // allocated before dst_tmp_meta, which asserted in ggml_cuda_pool_vmm::free (issue #76).
    ggml_cuda_pool_alloc<int>      KV_max(pool);
    ggml_cuda_pool_alloc<uint32_t> kq_blocks(pool);
    ggml_cuda_pool_alloc<float>    dst_tmp(pool);
    ggml_cuda_pool_alloc<float2>   dst_tmp_meta(pool);

    const char * K_data = (const char *) K->data;
    size_t nb11 = K->nb[1];
    size_t nb12 = K->nb[2];
    size_t nb13 = K->nb[3];

    const char * V_data = (const char *) V->data;
    size_t nb21 = V->nb[1];
    size_t nb22 = V->nb[2];
    size_t nb23 = V->nb[3];

    if (need_f16_K && K->type != GGML_TYPE_F16 && !use_native_K) {
        const size_t bs = ggml_blck_size(K->type);
        const size_t ts = ggml_type_size(K->type);

        half * K_f16;
        if (stage_K) {
            K_f16 = K_f16_stage;
        } else {
            GGML_ASSERT(f16_extra.K != 0);
            K_f16 = (half *) f16_extra.K;
        }
        if (ggml_is_contiguously_allocated(K)) {
            to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(K->type);
            to_fp16(K_data, K_f16, ggml_nelements(K), main_stream);

            nb11 = nb11*bs*sizeof(half)/ts;
            nb12 = nb12*bs*sizeof(half)/ts;
            nb13 = nb13*bs*sizeof(half)/ts;
        } else {
            GGML_ASSERT(K->nb[0] == ts);
            to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(K->type);
            const int64_t s01 = nb11 / ts;
            const int64_t s02 = nb12 / ts;
            const int64_t s03 = nb13 / ts;
            to_fp16(K_data, K_f16, K->ne[0], K->ne[1], K->ne[2], K->ne[3], s01, s02, s03, main_stream);

            nb11 = K->ne[0] * sizeof(half);
            nb12 = K->ne[1] * nb11;
            nb13 = K->ne[2] * nb12;
        }
        K_data = (char *) K_f16;
    }

    if (need_f16_V && V->type != GGML_TYPE_F16 && !use_native_V) {
        if (V_is_K_view) {
            V_data = K_data;
            nb21   = nb11;
            nb22   = nb12;
            nb23   = nb13;
        } else {
            const size_t bs = ggml_blck_size(V->type);
            const size_t ts = ggml_type_size(V->type);

            half * V_f16;
            if (stage_V) {
                V_f16 = V_f16_stage;
            } else {
                GGML_ASSERT(f16_extra.V != 0);
                V_f16 = (half *) f16_extra.V;
            }
            if (ggml_is_contiguously_allocated(V)) {
                to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(V->type);
                to_fp16(V_data, V_f16, ggml_nelements(V), main_stream);
                V_data = (char *) V_f16;

                nb21 = nb21*bs*sizeof(half)/ts;
                nb22 = nb22*bs*sizeof(half)/ts;
                nb23 = nb23*bs*sizeof(half)/ts;
            } else {
                GGML_ASSERT(V->nb[0] == ts);
                to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(V->type);
                const int64_t s01 = nb21 / ts;
                const int64_t s02 = nb22 / ts;
                const int64_t s03 = nb23 / ts;
                to_fp16(V_data, V_f16, V->ne[0], V->ne[1], V->ne[2], V->ne[3], s01, s02, s03, main_stream);

                nb21 = V->ne[0] * sizeof(half);
                nb22 = V->ne[1] * nb21;
                nb23 = V->ne[2] * nb22;
            }
            V_data = (char *) V_f16;
        }
    }

    const int ntiles_x     = ((Q->ne[1] + ncols1 - 1) / ncols1);
    const int gqa_ratio    = Q->ne[2] / K->ne[2];
    const int ntiles_z_gqa = ((gqa_ratio + ncols2 - 1) / ncols2);
    const int ntiles_dst   = ntiles_x * ntiles_z_gqa * K->ne[2] * Q->ne[3];

    // sparse: a query tile of ncols1 queries shares one index list, the union of the queries' visible columns
    int32_t n_kv_max = 0;
    if (use_sparse) {
        GGML_ASSERT(mask != nullptr);
        const int32_t n_kv_max_query = ggml_get_op_params_i32(KQV, 4);
        GGML_ASSERT(n_kv_max_query > 0);
        n_kv_max = std::min<int64_t>(K->ne[1], int64_t(ncols1)*n_kv_max_query);

        const size_t n_lists = size_t(ntiles_x) * mask->ne[3];

        KV_max.alloc(size_t(n_kv_max)*n_lists + n_lists);
        ggml_cuda_flash_attn_ext_compact_mask(mask, KV_max.ptr, KV_max.ptr + size_t(n_kv_max)*n_lists, Q->ne[1], ncols1, n_kv_max, main_stream);
    }

    // Optional optimization where the mask is scanned to determine whether part of the calculation can be skipped.
    // Only worth the overhead if there is at lease one FATTN_KQ_STRIDE x FATTN_KQ_STRIDE square to be skipped or
    //     multiple sequences of possibly different lengths.
    if (!use_sparse && mask && K->ne[1] % FATTN_KQ_STRIDE == 0 && (Q->ne[1] >= 1024 || Q->ne[3] > 1)) {
        const int64_t s31 = mask->nb[1] / sizeof(half2);
        const int64_t s33 = mask->nb[3] / sizeof(half2);

        const dim3 blocks_num_KV_max(ntiles_x, Q->ne[3], 1);
        const dim3 block_dim_KV_max(FATTN_KQ_STRIDE/2, 1, 1);

        const int ne_KV_max = blocks_num_KV_max.x*blocks_num_KV_max.y;
        const int iter_k = K->ne[1] / FATTN_KQ_STRIDE;

        KV_max.alloc(ne_KV_max);
        ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num_KV_max, block_dim_KV_max, 0, main_stream);
        ggml_cuda_kernel_launch(flash_attn_mask_to_KV_max<ncols1>, launch_params,
            (const half2 *) mask->data, KV_max.ptr, iter_k, s31, s33);
        CUDA_CHECK(cudaGetLastError());
    }

    // Fully-masked KV-group skip (issue #48 / upstream #28495).  A unified KV cache keeps the other
    // slots' cells inside the attention range; they are all -INF for this sequence but the kernels
    // used to process every one of them, which made a concurrent prefill pay up to ~2x.  Skipping a
    // group that is fully masked for every query row of a tile is exact (it only contributes zeros to
    // the online softmax), so this is a bit-identical optimization.  There are two producers:
    //
    //  - derived mask (single-sequence prefill): the per-cell visibility is known on the host, so the
    //    launcher classifies groups into a batch-wide bitmap (flash_attn_kq_derived_blocks).
    //  - packed mask (multi-sequence prefill, e.g. continuous batching): different rows of a tile can
    //    belong to different sequences, so a batch-wide test cannot work; a GPU prepass scans the
    //    mask and classifies every (stream, query tile, group) pair (flash_attn_mask_to_KV_blocks).
    // Both layouts are consumed by the kernel through the one `kq_blocks` argument; the kernel tells
    // them apart by `mask == nullptr` (derived is batch-wide, packed is per tile).
    const bool kq_skip_ok =
        !use_sparse &&
        GGML_CUDA_CC_IS_AMD(cc) &&
        (FATTN_KQ_STRIDE % nbatch_fa) == 0 &&
        ggml_cuda_fattn_kq_block_skip_enabled();
    const bool kq_block_skip_derived =
        kq_skip_ok && cell_pos != nullptr && tok_lo != nullptr && tok_hi != nullptr;
    // prefill only: the decode/verify band keeps its tuned round-robin KV split untouched
    const bool kq_block_skip_packed =
        kq_skip_ok && cell_pos == nullptr && mask != nullptr && Q->ne[1] > 8 &&
        K->ne[1] % FATTN_KQ_STRIDE == 0;

    const uint32_t * kq_blocks_arg = nullptr;

    if (kq_block_skip_derived) {
        const int64_t n_kv = K->ne[1];
        GGML_ASSERT(cell_pos->ne[0] == n_kv);
        GGML_ASSERT(tok_lo->ne[0] == Q->ne[1]);
        GGML_ASSERT(tok_hi->ne[0] == Q->ne[1]);

        const int n_groups = (int) ((n_kv + FATTN_KQ_STRIDE - 1) / FATTN_KQ_STRIDE);
        const int n_words  = (n_groups + 31) / 32;

        const int n_blocks = (n_words + FATTN_KQ_BLOCK_THREADS - 1) / FATTN_KQ_BLOCK_THREADS;

        // The batch-wide visibility window is derived inside the kernel: `tok_lo`/`tok_hi` here are
        // device tensors (the scheduler copies the host graph inputs), so they must not be
        // dereferenced on the host (issue #53).
        kq_blocks.alloc(n_words);
        ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(
            dim3(n_blocks, 1, 1), dim3(FATTN_KQ_BLOCK_THREADS, 1, 1), 0, main_stream);
        ggml_cuda_kernel_launch(flash_attn_kq_derived_blocks, launch_params,
            (const int *) cell_pos->data,
            (const int *) tok_lo->data, (const int *) tok_hi->data, (int) Q->ne[1],
            kq_blocks.ptr, (int) n_kv, n_groups);
        CUDA_CHECK(cudaGetLastError());

        kq_blocks_arg = kq_blocks.ptr;
    } else if (kq_block_skip_packed) {
        GGML_ASSERT(mask->ne[0] == K->ne[1]);

        const int n_kv           = (int) K->ne[1];
        const int n_groups       = n_kv / FATTN_KQ_STRIDE;
        const int words_per_tile = (n_groups + 31) / 32;
        const int n_streams      = (int) Q->ne[3];
        const int n_mask_streams = (int) mask->ne[3];
        const int64_t s31        = mask->nb[1] / sizeof(half);
        const int64_t s33        = mask->nb[3] / sizeof(half);

        const int64_t n_words_total = int64_t(n_streams)*ntiles_x*words_per_tile;
        kq_blocks.alloc(n_words_total);
        CUDA_CHECK(cudaMemsetAsync(kq_blocks.ptr, 0, n_words_total*sizeof(uint32_t), main_stream));

        constexpr int n_threads = 256;
        const int64_t n_warps = int64_t(n_streams)*ntiles_x*n_groups;
        const int n_blocks = (int) ((n_warps*WARP_SIZE + n_threads - 1) / n_threads);

        ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(
            dim3(n_blocks, 1, 1), dim3(n_threads, 1, 1), 0, main_stream);
        ggml_cuda_kernel_launch(flash_attn_mask_to_KV_blocks<ncols1>, launch_params,
            (const half *) mask->data, kq_blocks.ptr, n_kv, (int) Q->ne[1], s31, s33,
            ntiles_x, n_groups, n_streams, n_mask_streams);
        CUDA_CHECK(cudaGetLastError());

        kq_blocks_arg = kq_blocks.ptr;
    }

    const int * kv_max_arg = KV_max.ptr;

    const dim3 block_dim(warp_size, nwarps, 1);
    int max_blocks_per_sm = 1; // Max. number of active blocks limited by occupancy.
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, fattn_kernel, block_dim.x * block_dim.y * block_dim.z, nbytes_shared));
    GGML_ASSERT(max_blocks_per_sm > 0);
    int parallel_blocks = max_blocks_per_sm;

    const int64_t n_kv = use_sparse ? n_kv_max : K->ne[1];
    const int ntiles_KV = (n_kv + nbatch_fa - 1) / nbatch_fa; // Max. number of parallel blocks limited by KV cache length.

    dim3 blocks_num;
    if (stream_k) {
        auto should_use_stream_k = [](const int cc, const int ntiles_dst, const int max_blocks, const int DKQ) {
            const int tiles_nwaves             = (ntiles_dst + max_blocks - 1) / max_blocks;
            const int tiles_efficiency_percent = 100 * ntiles_dst / (max_blocks*tiles_nwaves);

            if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                return true;
            }
            if (amd_wmma_available(cc) && DKQ == 64) {
                return true; // TODO better configuration
            }
            return tiles_efficiency_percent < 75;
        };

        const int  max_blocks   = max_blocks_per_sm*nsm;
        const bool use_stream_k = should_use_stream_k(cc, ntiles_dst, max_blocks, Q->ne[0]);

        blocks_num.x = ntiles_dst;
        blocks_num.y = 1;
        blocks_num.z = 1;

        // In the decode/verify band the KV of every output tile is split over a fixed number P of
        // blocks that take the nbatch_fa-row KV iterations round-robin (block i processes iterations
        // i, i+P, i+2P, ...; launched as gridDim = (ntiles_dst, P)), like the tile kernel's
        // parallel_blocks.  P depends only on the device (nsm) or an explicit override, never on n_q
        // or on the KV length, so a given KV iteration always lands in the same block at the same
        // position of its accumulation order: a longer KV (a verify batch that crossed a 256-row
        // padding boundary) only appends iterations that are fully masked for the earlier query rows,
        // which are exact no-ops, and the uniform fixup combines the P partials in a fixed order.
        // The band is a 2-D launch whose fast path only exists for ncols2 == 8, so the template
        // parameter must be part of the gate: if the dispatcher ever picked a different ncols2 this
        // would fall back to the ordinary stream-k path instead of launching a mismatched grid.
        const bool band_wmma = ncols2 == 8 && ggml_cuda_fattn_band_wmma_applies(cc, dst);

        if (band_wmma) {
            // P = nsm for the native-quantized types: measured on gfx1201 (head 256, GQA 6, 4 KV heads)
            // the verify cost is flat for P = 48..96 at kv 20K..200K and degrades below 32 or above 128.
            // The 2-byte types do less work per KV iteration, so the P partials' fixup is a larger
            // fraction of their cost; the split helper returns a smaller fixed P for them.
            const int split_env = ggml_cuda_fattn_band_wmma_split(dst);
            const int P         = split_env > 0 ? split_env : std::max(2, nsm);
            blocks_num.x = ntiles_dst;
            blocks_num.y = P;
        } else if(use_stream_k) {
            const int nblocks_stream_k_raw = std::min(max_blocks, ntiles_KV*ntiles_dst);
            // Round down to a multiple of ntiles_dst so that each output tile gets the same number of blocks (avoids fixup).
            // Only do this if the occupancy loss from rounding is acceptable.
            const int nblocks_stream_k_rounded = (nblocks_stream_k_raw / ntiles_dst) * ntiles_dst;
            const int max_efficiency_loss_percent = 5;
            const int efficiency_loss_percent = nblocks_stream_k_rounded > 0
                ? 100 * (nblocks_stream_k_raw - nblocks_stream_k_rounded) / nblocks_stream_k_raw
                : 100;
            const int nblocks_stream_k = efficiency_loss_percent <= max_efficiency_loss_percent
                ? nblocks_stream_k_rounded
                : nblocks_stream_k_raw;

            blocks_num.x = nblocks_stream_k;
        }

        const int nblocks_total = blocks_num.x * blocks_num.y; // blocks_num.y > 1 only in the band
        if (ntiles_dst % nblocks_total != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            dst_tmp_meta.alloc((size_t(nblocks_total) * ncols * (2 + DV/2)));
        }
    } else {
        // parallel_blocks must not be larger than what the tensor size allows:
        parallel_blocks = std::min(parallel_blocks, ntiles_KV);

        // Decode and speculative verify batches (n_q <= 8) must use a query-width-
        // independent KV split: ntiles_dst is a function of Q->ne[1]
        // (ntiles_x = ceil(Q->ne[1]/ncols1)), so n_q=3 and n_q=5 would otherwise
        // feed different partial sums into the online-softmax/PV combine, drift the
        // logits in the last bits and flip greedy near-ties (issue #25).  Evaluate
        // the heuristic as if n_q == 1 so every small batch agrees.
        const int ntiles_dst_eff = Q->ne[1] <= 8 ? (ntiles_z_gqa * K->ne[2] * Q->ne[3]) : ntiles_dst;

        // If ntiles_total % blocks_per_wave != 0 then some efficiency is lost due to tail effects.
        // Test whether parallel_blocks can be set to a higher value for better efficiency.
        const int blocks_per_wave = nsm * max_blocks_per_sm;
        int nwaves_best = 0;
        int efficiency_percent_best = 0;
        for (int parallel_blocks_test = parallel_blocks; parallel_blocks_test <= ntiles_KV; ++parallel_blocks_test) {
            const int nblocks_total = ntiles_dst_eff * parallel_blocks_test;
            const int nwaves = (nblocks_total + blocks_per_wave - 1) / blocks_per_wave;
            const int efficiency_percent = 100 * nblocks_total / (nwaves*blocks_per_wave);

            // Stop trying configurations with more waves if we already have good efficiency to avoid excessive overhead.
            if (efficiency_percent_best >= 95 && nwaves > nwaves_best) {
                break;
            }

            if (efficiency_percent > efficiency_percent_best) {
                nwaves_best = nwaves;
                efficiency_percent_best = efficiency_percent;
                parallel_blocks = parallel_blocks_test;
            }
        }

        blocks_num.x = ntiles_x;
        blocks_num.y = parallel_blocks;
        blocks_num.z = ntiles_z_gqa*K->ne[2]*Q->ne[3];

        if (parallel_blocks > 1) {
            dst_tmp.alloc(parallel_blocks*ggml_nelements(KQV));
            dst_tmp_meta.alloc(parallel_blocks*ggml_nrows(KQV));
        }
    }

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale,         (const float *) KQV->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) KQV->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const uint32_t n_head      = Q->ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // TODO other tensor dimensions after removal of WMMA kernel:
    const uint3 ne01 = init_fastdiv_values(Q->ne[1]);

    GGML_ASSERT(block_dim.x % warp_size == 0);

    ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num, block_dim, nbytes_shared, main_stream);
    ggml_cuda_kernel_launch(fattn_kernel, launch_params,
        (const char *) Q->data,
        K_data,
        V_data,
        mask ? ((const char *) mask->data) : nullptr,
        sinks ? ((const char *) sinks->data) : nullptr,
        kv_max_arg,
        kq_blocks_arg,
        !stream_k && parallel_blocks > 1 ? dst_tmp.ptr : (float *) KQV->data, dst_tmp_meta.ptr,
        scale, max_bias, m0, m1, n_head_log2, logit_softcap,
        Q->ne[0], ne01,     Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
        K->ne[0], n_kv, K->ne[2], K->ne[3], nb11, nb12, nb13,
        nb21, nb22, nb23,
        mask ? mask->ne[1] : 0, mask ? mask->ne[2] : 0, mask ? mask->ne[3] : 0,
        mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0, mask ? mask->nb[3] : 0,
        cell_pos ? (const int *) cell_pos->data : nullptr,
        tok_lo   ? (const int *) tok_lo  ->data : nullptr,
        tok_hi   ? (const int *) tok_hi  ->data : nullptr,
        kv_native_kernel_K, kv_native_kernel_V
    );
    CUDA_CHECK(cudaGetLastError());

    if (stream_k) {
        const int nblocks_launched = (int)(blocks_num.x * blocks_num.y); // 2-D only in the band
        if (nblocks_launched % ntiles_dst == 0 && nblocks_launched > ntiles_dst) {
            // Optimized fixup: nblocks_stream_k is a multiple of ntiles_dst, launch one block per tile.
            const int nblocks_sk  = nblocks_launched;
            const int bpt         = nblocks_sk / ntiles_dst;

            const uint3 fd0 = init_fastdiv_values(ntiles_x * ntiles_z_gqa * K->ne[2]);
            const uint3 fd1 = init_fastdiv_values(ntiles_x * ntiles_z_gqa);
            const uint3 fd2 = init_fastdiv_values(ntiles_x);

            const dim3 block_dim_combine(DV, 1, 1);
            const dim3 blocks_num_combine = {(unsigned)ntiles_dst, ncols1, ncols2};

            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num_combine, block_dim_combine, 0, main_stream);
            ggml_cuda_kernel_launch(flash_attn_stream_k_fixup_uniform<DV, ncols1, ncols2>, launch_params,
                (float *) KQV->data, dst_tmp_meta.ptr,
                 Q->ne[1], Q->ne[2], K->ne[2], nblocks_sk,
                 gqa_ratio, bpt, fd0, fd1, fd2);
        } else if (ntiles_dst % nblocks_launched != 0) {
            // General fixup for the cases where nblocks_stream_k < ntiles_dst.
            const int total_work = ntiles_KV * ntiles_dst;

            const uint3 fd_k_j_z_ne12 = init_fastdiv_values(ntiles_KV * ntiles_x * ntiles_z_gqa * K->ne[2]);
            const uint3 fd_k_j_z      = init_fastdiv_values(ntiles_KV * ntiles_x * ntiles_z_gqa);
            const uint3 fd_k_j        = init_fastdiv_values(ntiles_KV * ntiles_x);
            const uint3 fd_k          = init_fastdiv_values(ntiles_KV);

            const dim3 block_dim_combine(DV, 1, 1);
            const dim3 blocks_num_combine = {blocks_num.x, ncols1, ncols2};

            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num_combine, block_dim_combine, 0, main_stream);
            ggml_cuda_kernel_launch(flash_attn_stream_k_fixup_general<DV, ncols1, ncols2>, launch_params,
                (float *) KQV->data, dst_tmp_meta.ptr,
                 Q->ne[1], Q->ne[2], gqa_ratio, total_work,
                 fd_k_j_z_ne12, fd_k_j_z, fd_k_j, fd_k);
        }
    } else if (parallel_blocks > 1) {
        const dim3 block_dim_combine(DV, 1, 1);
        const dim3 blocks_num_combine(Q->ne[1], Q->ne[2], Q->ne[3]);
        const size_t nbytes_shared_combine = parallel_blocks*sizeof(float2);

        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num_combine, block_dim_combine, nbytes_shared_combine, main_stream);
        ggml_cuda_kernel_launch(flash_attn_combine_results<DV>, launch_params,
            dst_tmp.ptr, dst_tmp_meta.ptr, (float *) KQV->data, parallel_blocks);
    }
    CUDA_CHECK(cudaGetLastError());
}
