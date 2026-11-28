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

static inline int ggml_cuda_fattn_band_wmma_ncols1(const ggml_tensor * dst) {
    GGML_UNUSED(dst);
    const int v = ggml_cuda_fattn_band_wmma_ncols1_env();
    if (v > 0) {
        return v == 2 ? 2 : 4;
    }
    return 4;
}

// Blocks per output tile in the band (the interleave period P, see launch_fattn).
// GGML_HIP_FA_BAND_WMMA_SPLIT overrides it for tuning.  Whatever the source, it must not depend on n_q
// or on the KV length, or decode and verify would split differently (GREEDY-PURITY.md §98).
//
// The default keeps ~16 warps in flight per SM, which is where the launch saturates: a 256-dim band
// block is 8 warps (2 blocks/SM) and the 128-dim row's is 4 (4 blocks/SM).  `tiles` is the number of
// OUTPUT TILES the device's own slice gets (ntiles_z_gqa x K/V heads x Q streams: a tensor split cuts
// the K/V head count, so this is per device), and the grid is tiles x P -- so the tile count decides
// how many blocks per tile are needed to fill the machine.  nsm/2 (16 on gfx1201) is the floor, which
// is what a device with 4 tiles gets (4 x 16 = 64 blocks = the saturation point) and therefore keeps
// r316's measured value for the shape it was measured on; a device that only gets 1 or 2 tiles needs
// 64 or 32 blocks per tile instead, or most of the SMs idle (issue #333: a 2-device tensor split of
// qwen4exp gives 2 tiles/device and P 16 cost 4-9 % against P 32).  The tile count is constant inside
// the band (ntiles_x is 1 for n_q = 1..8), so P is still width- and extent-independent.
// Records: benchmarks/2026-11-18-band-fixup-period.md (`r316`),
// benchmarks/2026-11-19-band-split-blocks-per-sm.md (`r322`).
static inline int ggml_cuda_fattn_band_wmma_split(const ggml_tensor * dst, const int tiles, const int nwarps) {
    GGML_UNUSED(dst);
    static const int env_split = []() {
        const char * env = getenv("GGML_HIP_FA_BAND_WMMA_SPLIT");
        const int v = env ? atoi(env) : 0;
        return v > 0 ? v : 0;
    }();
    if (env_split > 0) {
        return env_split;
    }
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    // blocks per SM that keep ~16 warps resident: 2 for the 8-warp 256-dim row, 4 for the 4-warp
    // 128-dim one, 1 for the 16-warp wide row (which therefore keeps nsm/2 unless a device gets < 2
    // tiles) -- only two of those are measured, so the unmeasured one stays at r316's value.
    const int bps     = std::max(1, 16 / std::max(1, nwarps));
    const int ntiles  = std::max(1, tiles);
    const int need    = (nsm*bps + ntiles - 1) / ntiles;
    return std::max(std::max(2, nsm/2), need);
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

// 128-dim heads in the band.  Outside the band an odd GQA ratio folds no heads at all (the generic dispatch
// folds the largest power of two dividing the ratio, and the tile kernel needs every column of a block on one
// K/V head), so a GQA 7 decode reads each K/V head once per query head.  The band folds the group into
// ncols2 = 8 and masks the unused columns.  Ratios that are a multiple of 8 already fold whole in the generic
// dispatch and stay there.  The band's numbers differ from the tile kernel's (another summation order); decode
// and verify (1..8 rows) stay bit-identical to each other.
//
// It is on where it pays, never unconditionally: the band adds a fixed per-layer cost (the packed-mask prepass
// and the P = nsm fixup combine) and its round-robin split leaves most blocks idle on a short cache, so it only
// wins once folding ceil(ratio/ncols2_tile) tile passes into ceil(ratio/8) band passes amortizes that cost over
// enough KV cells.  GGML_HIP_FA_BAND_WMMA_H128: unset = auto (the work gate below), 0 = off (kill switch),
// 1 = on for every 128-dim shape the predicate reaches (the tuning/measurement arm).
static inline int ggml_cuda_fattn_band_wmma_h128_env() {
    static const int v = []() {
        const char * env = getenv("GGML_HIP_FA_BAND_WMMA_H128");
        return env != nullptr ? atoi(env) : -1;
    }();
    return v; // -1 = auto, 0 = off, >0 = forced on
}

static inline bool ggml_cuda_fattn_band_wmma_h128_enabled() {
    return ggml_cuda_fattn_band_wmma_h128_env() != 0;
}

// cells x passes the band has to save before its per-layer cost is paid back.  Measured on gfx1201 (q8_0 K/V,
// decode width, band time / tile-kernel time), the crossover sits at 16-25k in every row of the class:
//   ratio 5 (5 -> 1 passes): 1.26 at 2048 cells, 0.94 at 4096, 0.65 at 8192, 0.51 at 16384
//   ratio 7 (7 -> 1):        1.05 at 2048,       0.72 at 4096, 0.48 at 8192, 0.37 at 16384
//   ratio 10 (5 -> 2):       1.70 at 2048,       1.24 at 4096, 0.96 at 8192, 0.81 at 16384
//   ratio 12 (3 -> 2):       1.83 at 2048,       1.45 at 4096, 1.14 at 8192, 0.95 at 16384
// 20000 is the middle of that band.  GGML_HIP_FA_BAND_H128_MIN_WORK overrides it for tuning.
static inline int64_t ggml_cuda_fattn_band_h128_min_work() {
    static const int64_t v = []() {
        const char * env = getenv("GGML_HIP_FA_BAND_H128_MIN_WORK");
        const long long x = env != nullptr ? atoll(env) : 20000;
        return x >= 0 ? (int64_t) x : (int64_t) 20000;
    }();
    return v;
}

// Passes over the layer's KV cache that each candidate kernel makes for a GQA ratio: the tile kernel folds the
// largest power of two dividing the ratio (capped at 8 -- launch_fattn_tile_switch_ncols2, fattn-tile.cuh), the
// band always folds into its ncols2 = 8 columns.  Both work gates (128- and 256-dim) price their decision from
// these two numbers, so they cannot disagree about the saving a band takes.
static inline int ggml_cuda_fattn_band_passes_tile(const int gqa_ratio) {
    int ncols2_tile = 1;
    while (ncols2_tile < 8 && gqa_ratio % (2*ncols2_tile) == 0) {
        ncols2_tile *= 2;
    }
    return (gqa_ratio + ncols2_tile - 1) / ncols2_tile;
}

static inline int ggml_cuda_fattn_band_passes_band(const int gqa_ratio) {
    return (gqa_ratio + 7) / 8;
}

// Does a 128-dim layer with this ratio take the band?  The work test reads K->ne[1], which is the number of KV
// iterations the kernel will execute: K is a permuted view of the KV cache whose extent is the PADDED USED
// length (r315 corrected r314's "cache size" here -- measured on a 27B, `-c 8192`: 256 cells at d0, 5376 ->
// 5632 while generating, 8192 only during a prompt ubatch or in the graph-RESERVATION pass, where the
// reservation context sets n_kv = kv->get_size()).  It is the right attribute for the crossover -- the cost the
// band adds is per launch and per KV iteration -- and it is never a function of the query width.  A verify
// batch can straddle a 256-cell padding boundary and then select a different kernel family for that ONE step
// (GREEDY-PURITY.md 96/97); the alternative, keying on the cache size, would band a shallow context on a large
// `-c` and reintroduce the losses this gate exists to remove.  A sliding-window / hybrid model's windowed
// layers hold n_swa + n_ubatch cells for the whole run, so they are declined at every depth: measured on
// ricursive 65B (64 of 80 layers windowed at 1024) the unconditional band cost 4.5-5.8 % of decode at d0..d32K.
static inline bool ggml_cuda_fattn_band_wmma_h128_applies(const ggml_tensor * K, const int gqa_ratio) {
    if (!ggml_cuda_fattn_band_wmma_h128_enabled()) {
        return false;
    }
    if (ggml_cuda_fattn_band_wmma_h128_env() > 0) {
        return true; // forced on: no work gate (the arm that measures the band itself)
    }
    const int64_t passes_tile = ggml_cuda_fattn_band_passes_tile(gqa_ratio);
    const int64_t passes_band = ggml_cuda_fattn_band_passes_band(gqa_ratio);
    const int64_t saving      = passes_tile - passes_band;
    return saving > 0 && K != nullptr && K->ne[1]*saving >= ggml_cuda_fattn_band_h128_min_work();
}

// 256-dim heads (27B / 35B-A3B / qwen4exp, the published default since r6) take the band only where its
// per-layer cost is paid back -- the same work test as the 128-dim arm, on the same attribute.
//
// Per op (gfx1201, one R9700, head 256, 4 KV heads, band / tile time, the best P of 4/8/16/32/64, q8_0 / f16):
//   GQA 6 decode width: 2.64 / 1.96 at 512 cells, 2.13 / 1.59 at 1024, 1.30 / 1.89 at 2048, 1.26 / 1.48 at
//                       4096, 0.96 / 1.09 at 8192, 0.66 / 1.10 at 32768
//   GQA 6 verify width 8: 1.14 / 1.24 at 512, 0.76 / 0.71 at 1024, 0.39 / 0.47 at 2048, 0.31 / 0.40 at 4096
// Below ~8192 cells the band loses at the decode width at EVERY P and every K/V type: what it costs is its
// fixed per-launch work (the packed-mask prepass and the P-partial fixup combine), not the idle blocks -- at
// 512 cells P = 8 (one iteration per block) measures 52.6 us against 31.1 for P = 16, because the blocks that
// own no iteration write a neutral partial and exit while a smaller P serialises the KV iterations
// (TODO #345 route (b), refuted).  At verify widths the band wins from 1024 cells, and an adaptive P cannot be
// keyed on the used length anyway: it moves with the context (see above).
//
// End to end, 3 GPUs `-sm tensor`, `ROCBLAS_USE_HIPBLASLT=0`, band against GGML_HIP_FA_BAND_WMMA=0:
//   27B Q8_0 (GQA 6):      -4.6 % at d0, -1.6/-1.5/-1.6/-1.7 % at d4K/8K/16K/32K, -4.7 % with draft-mtp n3
//   35B-A3B Q8_0 (GQA 8):  -9.8 % at d0, -2.7 % at d8192, -1.8 % with draft-mtp n3
//   Flash-Next (GQA 12):   -4.0 % at d0, -2.7 % at d8192
// and the arm that keeps the band above the gate: 27B, MTP n7 at a 20.5K-token prompt, +6.7 % (79.4 vs 74.4
// t/s), where the same config's plain decode is -1.6 % -- the band is a verify-width kernel that a weight-bound
// decode cannot pay for.  A ratio that already folds whole outside the band (GQA 8: the tile kernel's ncols2 is
// 8 there too, so the band saves no passes) still has to clear the fixed threshold as ONE pass; that is what
// removes the 35B-A3B loss, and it keeps the band for the deep verify widths where it measures 0.45 at 8192
// cells instead of disabling it for the whole class.
//
// Work: cells x max(passes_tile - passes_band, 1) >= GGML_HIP_FA_BAND_H256_MIN_WORK (default 20000), i.e. band
// from 10000 cells for GQA 6 (3 -> 1 passes), 20000 for GQA 8 (no saving) and GQA 12 (3 -> 2), 5000 for GQA 5.
// 20000 is the 128-dim arm's measured crossover and every 256-dim row above crosses below it, so the gate is on
// the conservative side: it removes the shallow regressions and leaves the depth residual (the 27B's -1.6 % at
// d16K+) in exchange for the verify-width wins.  GGML_HIP_FA_BAND_WMMA=0 remains the escape hatch, and
// GGML_HIP_FA_BAND_WMMA_H256 is the class switch: unset = auto (this gate), 0 = off, 1 = on for every 256-dim
// shape (the measurement arm, and what the band's own test-backend-ops oracle is run with -- with the gate on,
// a short-cache case takes the tile kernel).
static inline int ggml_cuda_fattn_band_wmma_h256_env() {
    static const int v = []() {
        const char * env = getenv("GGML_HIP_FA_BAND_WMMA_H256");
        return env != nullptr ? atoi(env) : -1;
    }();
    return v; // -1 = auto, 0 = off, >0 = forced on
}

static inline int64_t ggml_cuda_fattn_band_h256_min_work() {
    static const int64_t v = []() {
        const char * env = getenv("GGML_HIP_FA_BAND_H256_MIN_WORK");
        const long long x = env != nullptr ? atoll(env) : 20000;
        return x >= 0 ? (int64_t) x : (int64_t) 20000;
    }();
    return v;
}

static inline bool ggml_cuda_fattn_band_wmma_h256_applies(const ggml_tensor * K, const int gqa_ratio) {
    const int forced = ggml_cuda_fattn_band_wmma_h256_env();
    if (forced == 0) {
        return false;
    }
    if (forced > 0) {
        return true; // forced on: no work gate (the measurement arm / the band's own oracle)
    }
    const int64_t passes_tile = ggml_cuda_fattn_band_passes_tile(gqa_ratio);
    const int64_t passes_band = ggml_cuda_fattn_band_passes_band(gqa_ratio);
    const int64_t saving      = std::max<int64_t>(1, passes_tile - passes_band);
    return K != nullptr && K->ne[1]*saving >= ggml_cuda_fattn_band_h256_min_work();
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
    // The band's 2-D launch (gridDim.y = P) has no kv_rows arm: its process_tile calls pass no index
    // list, so the round-robin blocks would stage the wrong K/V rows.  It is RDNA4-only, so this costs
    // nothing anywhere the form was ever validated (NVIDIA, where kv_rows is upstream's); the rows form
    // falls through to the ordinary stream-k MMA path instead.  See ggml_cuda_get_best_fattn_kernel.
    if (dst->src[5] != nullptr) {
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

    // The same GQA-optimization / alignment conditions the GQA kernels require: a mask (either form,
    // see ggml_cuda_flash_attn_ext_has_mask - a derived op has none at src[3]), no ALiBi, the KV is
    // FATTN_KQ_STRIDE-aligned, and every non-quantized operand is 16-byte aligned.
    bool gqa_opt = ggml_cuda_flash_attn_ext_has_mask(dst) && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
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

    const bool head_ok =
        (Q->ne[0] == 256 && V->ne[0] == 256 && ggml_cuda_fattn_band_wmma_h256_applies(K, gqa_ratio)) ||
        (Q->ne[0] == 128 && V->ne[0] == 128 && gqa_ratio % 8 != 0 &&
         ggml_cuda_fattn_band_wmma_h128_applies(K, gqa_ratio));

    return gqa_opt && Q->ne[1] <= 8 && Q->ne[3] == 1 && head_ok &&
        gqa_ratio > 4 && gqa_ratio <= ggml_cuda_fattn_band_wmma_gqa_max() && logit_softcap == 0.0f && kv_ok;
}
