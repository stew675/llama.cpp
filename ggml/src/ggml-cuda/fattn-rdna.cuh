// RDNA/ROCm flash-attention predicates: the native-KV staging policy and selectors, and
// the RDNA4 decode/verify band (WMMA) helpers.  Split out of fattn-common.cuh (an upstream
// hub) so an upstream re-base collides with the predicate call points, not their bodies.
//
// Include-order requirement: this header must be included AFTER `FATTN_KQ_STRIDE` is defined
// (fattn-common.cuh line 9) -- it relies on the includer's definition.
#pragma once

#include "common.cuh"

#include <algorithm>
#include <cstdint>

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

// TODO 71: the tile kernel has ONE K/V type parameter, and its selection used to be kept by hand in
// two places -- the instantiation (ggml_cuda_flash_attn_ext_tile_case_type, fattn-tile.cu) and the
// F16 scratch sizing (ggml_cuda_flash_attn_ext_get_alloc_size, fattn.cu).  They drifted three times
// (the missing q4_0 arm, the mixed-K/V NaNs, the missing RDNA2 bf16 scratch), so the selection is
// this one function now: "the allocator and the launcher agree" is a property of the code, not of a
// comment.  bf16 only where the hardware reads it natively; otherwise the native dequant arm (the
// per-operand predicates are the same ones the launcher asks, and they fall back to NONE, i.e. F16,
// outside FAST_FP16 / outside the activation policy); BF16 without native support is staged to F16
// by the launcher, so it maps to the F16 instantiation.
static inline ggml_type ggml_cuda_fattn_tile_type(const ggml_tensor * K, const ggml_tensor * V, const int cc) {
    if (K->type == GGML_TYPE_BF16 && V->type == GGML_TYPE_BF16 && bf16_mma_hardware_available(cc)) {
        return GGML_TYPE_BF16;
    }
    const int tile_native = ggml_cuda_fattn_tile_kv_native_type(K, V);
    return (tile_native == FATTN_KV_NATIVE_BF16) ? GGML_TYPE_F16
                                                 : ggml_cuda_fattn_native_ggml_type(tile_native);
}

// Decode/verify band (n_q <= 8) on RDNA4: the whole GQA group is folded into one block
// (ncols2 = 8) via the WMMA kernel instead of the tile kernel's ncols2 = 2, which fetches and
// dequantizes every K/V element once per head pair (3x for GQA 6).  This is the default; the env var
// is the opt-out (maintainer policy: a beneficial feature is on by default).  One ncols1 for the whole
// band keeps a single kernel config, and launch_fattn fixes the round-robin KV split per output tile
// (see there), so n_q = 1 and every verify width reduce identically (GREEDY-PURITY band invariant).
// GGML_HIP_FA_BAND_WMMA=0 disables the band, 2/4 force ncols1 (any other non-zero value means 4).
//
// The band's ncols1 is 4 for every K/V type and every query width: a single 32-column block per
// output tile.  It is not a per-width choice, because ncols1 is part of the width-purity contract
// (one kernel config for n_q = 1..8, GREEDY-PURITY.md §36), and with the band's 64-half2 K/V row the
// narrower tile only wins where a wider block is mostly empty.  Measured on gfx1201 (head 256,
// GQA 6, kv 98304, f16 KV, us per call, band split nsm/2): ncols1 2 / 4 gives 683 / 686 at n_q = 1
// and 979 / 824 at n_q = 8, while at kv 16384 it is 98 / 102 at n_q = 1 and 204 / 175 at n_q = 8.
// The 2-column tile was the right default while the 2-byte arm still selected the old 128-half2
// config row (747 vs 967 us at n_q = 1, kv 102400); TODO #27 gave that arm the band's 64-half2 row,
// after which the extra output tiles (ceil(n_q/2) instead of ceil(n_q/4), each streaming the whole
// KV cache) dominate at every verify width.
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

// The band folds the GQA group into ncols2 = 8.  A ratio above 8 needs ceil(gqa_ratio/8) KV passes
// (the kernel loads a partial final group and masks its unused columns); that is still fewer passes
// than the tile kernel's ceil(gqa_ratio/ncols2_tile), where ncols2_tile is the largest power of two
// dividing the ratio (capped at 8) -- 1 for an odd ratio, and 3 for the qwen4exp ratio 12 (issue
// #133; the tile kernel picks ncols2 = 4 there).  The default cap is 12, the ratio measured on
// gfx1201 (qwen4exp GSQ-IQ3_XXS, q8_0 KV): +27 % on the deep-context MTP verify and ~-2 % on the
// short-context n_q = 1 (the wider tile pays for the query columns a decode leaves unused).
// GGML_HIP_FA_BAND_WMMA_GQA_MAX raises the cap for other ratios; a value <= 8 keeps the old
// whole-group fold (and the band is disabled wholesale with GGML_HIP_FA_BAND_WMMA=0).
static inline int ggml_cuda_fattn_band_wmma_gqa_max() {
    static const int v = []() {
        const char * env = getenv("GGML_HIP_FA_BAND_WMMA_GQA_MAX");
        const int x = env != nullptr ? atoi(env) : 12;
        return x > 8 ? x : 8;
    }();
    return v;
}

// The 2-byte K/V types share the band's light per-iteration variant (see the ncols1 and split
// helpers).  bf16 only reaches the band through its opt-in native arm, where K->type is BF16.
static inline bool ggml_cuda_fattn_band_wmma_two_byte(const ggml_tensor * dst) {
    const ggml_tensor * K = dst != nullptr ? dst->src[1] : nullptr;
    return K != nullptr && (K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_BF16);
}

static inline int ggml_cuda_fattn_band_wmma_ncols1(const ggml_tensor * dst) {
    GGML_UNUSED(dst);
    const int v = ggml_cuda_fattn_band_wmma_ncols1_env();
    if (v > 0) {
        return v == 2 ? 2 : 4;
    }
    return 4;
}

// Blocks per output tile in the band (the interleave period P, see launch_fattn).  0 = one block per
// CU (nsm); GGML_HIP_FA_BAND_WMMA_SPLIT overrides it for tuning.  Whatever the source, it must not
// depend on n_q or on the KV length, or decode and verify would split differently.  The 2-byte types
// do less work per KV iteration, so the P partials' fixup is a larger fraction of their cost; half of
// nsm measured best for them on gfx1201 (nsm 32: P 16 beats 24/32 by 6-8 % at kv 16384-51200 and is
// within noise at 98304, for n_q = 1..8), while the quantized types keep nsm.  Expressed as a
// fraction of nsm so a smaller RDNA4 part scales down with it.
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
    return std::max(2, nsm/2);
}

// With q8_0 K and V the band launches a kernel variant that fetches the raw K/V bytes ahead into registers
// (flash_attn_ext_f16_band_depth in fattn-mma-f16.cuh; bit-identical).  GGML_HIP_FA_BAND_FETCH_AHEAD=0 launches the
// kernel without it, as for the other K/V types.
static inline bool ggml_cuda_fattn_band_fetch_ahead_enabled() {
    static const bool v = []() {
        const char * env = getenv("GGML_HIP_FA_BAND_FETCH_AHEAD");
        return env == nullptr || atoi(env) != 0;
    }();
    return v;
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
        gqa_ratio > 4 && gqa_ratio <= ggml_cuda_fattn_band_wmma_gqa_max() && logit_softcap == 0.0f && kv_ok;
}
