#include "mmvq.cuh"
#include "moe-expert-cache.h"   // Phase 1b: moe_cache_enabled/moe_cache_get_cold (the UVA cold seam)
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

#include <cstdint>
#include <type_traits>

// only enabled on DGX Spark, where it is a gain on every type below. On the higher-bandwidth parts the kernel
// has little exposed latency left to hide and the extra requests cost more than they save.
// For perf data, see https://github.com/ggml-org/llama.cpp/pull/26705#issuecomment-5569335031
#if __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
// returns true only for those quants that benefit from prefetch and false otherwise
static constexpr __host__ __device__ bool mmvq_should_prefetch(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}

static __device__ __forceinline__ void mmvq_prefetch_l2(const void * p) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}
#endif

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

// The wide-VDR MoE expert entry points (mul_mat_vec_q_moe) are a RDNA4/RDNA3_0
// (gfx12xx / gfx1100) optimisation.  Every other target -- notably RDNA3_5
// (gfx115x), where the wide chunk never verified -- uses the dense VDR, so the
// MoE expert matmul reduces exactly like the rest of that arch.  Scope it here,
// once, for both selectors below: a per-quant gate is how the Q4_K/Q6_K arms
// (the Q4_K_M expert types) once leaked onto gfx1151.
#if !(defined(RDNA4) || defined(RDNA3_0))
#define GGML_CUDA_MMVQ_MOE_DENSE 1
#endif

static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda(ggml_type type, bool moe = false) {
#ifdef GGML_CUDA_MMVQ_MOE_DENSE
    moe = false;
#endif
    switch (type) {
        case GGML_TYPE_Q1_0:    return vec_dot_q1_0_q8_1;
        case GGML_TYPE_Q2_0:    return vec_dot_q2_0_q8_1;
        case GGML_TYPE_Q4_0:    return vec_dot_q4_0_q8_1;
        case GGML_TYPE_Q4_1:    return vec_dot_q4_1_q8_1;
        case GGML_TYPE_Q5_0:    return vec_dot_q5_0_q8_1;
        case GGML_TYPE_Q5_1:    return vec_dot_q5_1_q8_1;
        case GGML_TYPE_Q8_0:    return moe ? vec_dot_q8_0_q8_1_moe : vec_dot_q8_0_q8_1;
        case GGML_TYPE_MXFP4:   return vec_dot_mxfp4_q8_1;
        case GGML_TYPE_NVFP4:   return vec_dot_nvfp4_q8_1;
        case GGML_TYPE_Q2_K:    return vec_dot_q2_K_q8_1;
        case GGML_TYPE_Q3_K:    return vec_dot_q3_K_q8_1;
        case GGML_TYPE_Q4_K:    return moe ? vec_dot_q4_K_q8_1_vdr4 : vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q5_K:    return moe ? vec_dot_q5_K_q8_1_vdr4 : vec_dot_q5_K_q8_1;
#if defined(RDNA4) || defined(RDNA3_0)
        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1_vdr2;
#else
        case GGML_TYPE_Q6_K:    return moe ? vec_dot_q6_K_q8_1_vdr2 : vec_dot_q6_K_q8_1;
#endif
        case GGML_TYPE_IQ2_XXS: return vec_dot_iq2_xxs_q8_1;
        case GGML_TYPE_IQ2_XS:  return vec_dot_iq2_xs_q8_1;
        case GGML_TYPE_IQ2_S:   return vec_dot_iq2_s_q8_1;
        case GGML_TYPE_IQ3_XXS: return vec_dot_iq3_xxs_q8_1;
        case GGML_TYPE_IQ1_S:   return vec_dot_iq1_s_q8_1;
        case GGML_TYPE_IQ1_M:   return vec_dot_iq1_m_q8_1;
        case GGML_TYPE_IQ4_NL:  return vec_dot_iq4_nl_q8_1;
        case GGML_TYPE_IQ4_XS:  return vec_dot_iq4_xs_q8_1;
        case GGML_TYPE_IQ3_S:   return vec_dot_iq3_s_q8_1;
        default:                return nullptr;
    }
}

static constexpr __host__ __device__ int get_vdr_mmvq(ggml_type type, bool moe = false) {
#ifdef GGML_CUDA_MMVQ_MOE_DENSE
    moe = false;
#endif
    switch (type) {
        case GGML_TYPE_Q1_0:    return VDR_Q1_0_Q8_1_MMVQ;
        case GGML_TYPE_Q2_0:    return VDR_Q2_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_0:    return VDR_Q4_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_1:    return VDR_Q4_1_Q8_1_MMVQ;
        case GGML_TYPE_Q5_0:    return VDR_Q5_0_Q8_1_MMVQ;
        case GGML_TYPE_Q5_1:    return VDR_Q5_1_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0:    return moe ? VDR_Q8_0_Q8_1_MMVQ_MOE : VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_MXFP4:   return VDR_MXFP4_Q8_1_MMVQ;
        case GGML_TYPE_NVFP4:   return VDR_NVFP4_Q8_1_MMVQ;
        case GGML_TYPE_Q2_K:    return VDR_Q2_K_Q8_1_MMVQ;
        case GGML_TYPE_Q3_K:    return VDR_Q3_K_Q8_1_MMVQ;
        case GGML_TYPE_Q4_K:    return moe ? VDR_Q4_K_Q8_1_MMVQ_MOE : VDR_Q4_K_Q8_1_MMVQ;
        case GGML_TYPE_Q5_K:    return moe ? VDR_Q5_K_Q8_1_MMVQ_MOE : VDR_Q5_K_Q8_1_MMVQ;
        case GGML_TYPE_Q6_K:    return moe ? VDR_Q6_K_Q8_1_MMVQ_MOE : VDR_Q6_K_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XXS: return VDR_IQ2_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XS:  return VDR_IQ2_XS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_S:   return VDR_IQ2_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_XXS: return VDR_IQ3_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_S:   return VDR_IQ3_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_NL:  return VDR_IQ4_NL_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_XS:  return VDR_IQ4_XS_Q8_1_MMVQ;
        default:                return 1;
    }
}

enum mmvq_parameter_table_id {
    MMVQ_PARAMETERS_GENERIC = 0,
    MMVQ_PARAMETERS_TURING,
    MMVQ_PARAMETERS_GCN,
    MMVQ_PARAMETERS_RDNA2,
    MMVQ_PARAMETERS_RDNA3_0,
    MMVQ_PARAMETERS_RDNA3_5,
    MMVQ_PARAMETERS_RDNA4,
    MMVQ_PARAMETERS_GB10
};

static constexpr __device__ mmvq_parameter_table_id get_device_table_id() {
#if defined(RDNA4)
    return MMVQ_PARAMETERS_RDNA4;
#elif defined(RDNA3_0)
    return MMVQ_PARAMETERS_RDNA3_0;
#elif defined(RDNA3_5)
    return MMVQ_PARAMETERS_RDNA3_5;
#elif defined(RDNA2)
    return MMVQ_PARAMETERS_RDNA2;
#elif defined(GCN) || defined(CDNA)
    return MMVQ_PARAMETERS_GCN;
#elif __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA && __CUDA_ARCH__ < GGML_CUDA_CC_AMPERE
    return MMVQ_PARAMETERS_TURING;
#elif __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
    return MMVQ_PARAMETERS_GB10;
#else
    return MMVQ_PARAMETERS_GENERIC;
#endif
}

static __host__ mmvq_parameter_table_id get_device_table_id(int cc) {
    if (GGML_CUDA_CC_IS_RDNA4(cc)) {
        return MMVQ_PARAMETERS_RDNA4;
    }
    if (GGML_CUDA_CC_IS_RDNA3_0(cc)) {
        return MMVQ_PARAMETERS_RDNA3_0;
    }
    if (GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        return MMVQ_PARAMETERS_RDNA3_5;
    }
    if (GGML_CUDA_CC_IS_RDNA2(cc)) {
        return MMVQ_PARAMETERS_RDNA2;
    }
    if (GGML_CUDA_CC_IS_GCN(cc) || GGML_CUDA_CC_IS_CDNA(cc)) {
        return MMVQ_PARAMETERS_GCN;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA && ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_AMPERE) {
        return MMVQ_PARAMETERS_TURING;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_DGX_SPARK) {
        return MMVQ_PARAMETERS_GB10;
    }
    return MMVQ_PARAMETERS_GENERIC;
}

// Per-architecture maximum batch size for which MMVQ should be used for MUL_MAT_ID.
// Returns a value <= MMVQ_MAX_BATCH_SIZE. Default is MMVQ_MAX_BATCH_SIZE.
// Check https://github.com/ggml-org/llama.cpp/pull/20905#issuecomment-4145835627 for details

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_pascal_older(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 4;
        case GGML_TYPE_NVFP4:   return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 6;
        case GGML_TYPE_Q4_1:    return 6;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_0:    return 6;
        case GGML_TYPE_Q5_1:    return 6;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_turing_plus(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 7;
        case GGML_TYPE_IQ3_S:   return 6;
        case GGML_TYPE_IQ3_XXS: return 7;
        case GGML_TYPE_MXFP4:   return 7;
        case GGML_TYPE_NVFP4:   return 8;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_gcn(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 5;
        case GGML_TYPE_IQ1_M:   return 5;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 5;
        case GGML_TYPE_Q4_1:    return 5;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_cdna(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 5;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna1_rdna2(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_K:    return 6;
        case GGML_TYPE_Q6_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna3(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 6;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna4(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 7;
        case GGML_TYPE_IQ1_M:   return 7;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 7;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 5;
        case GGML_TYPE_NVFP4:   return 5;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 7;
        case GGML_TYPE_Q4_1:    return 7;
        case GGML_TYPE_Q4_K:    return 7;
        case GGML_TYPE_Q5_0:    return 7;
        case GGML_TYPE_Q5_1:    return 7;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 5;
        case GGML_TYPE_Q8_0:    return 7;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

// rdna-boosts: floor the per-type mmvq cap at the decode/verify band.
//
// The caps above are a throughput heuristic for the generic MUL_MAT_ID mmvq-vs-MMQ choice, but
// they subdivide the decode/verify band (ncols_dst <= MMVQ_MAX_BATCH_SIZE).  Greedy MTP requires a
// 1-token decode and an (n_draft + 1)-token speculative verify batch of the same MoE matmul to be
// bit-identical, and MUL_MAT_ID mmvq (mul_mat_vec_q_moe, one warp per token) and MMQ reduce in
// different orders, so no cap may split the band.  The cap still applies above the band.
static constexpr __host__ __device__ int mmvq_mmid_max_batch_band(int cap) {
    return cap > MMVQ_MAX_BATCH_SIZE ? cap : MMVQ_MAX_BATCH_SIZE;
}

// Host function: returns the max batch size for the current arch+type at runtime.
int get_mmvq_mmid_max_batch(ggml_type type, int cc) {
    // NVIDIA: Volta, Ada Lovelace, and Blackwell always use MMVQ for MUL_MAT_ID.
    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        if (cc == GGML_CUDA_CC_VOLTA || cc >= GGML_CUDA_CC_ADA_LOVELACE) {
            return MMVQ_MAX_BATCH_SIZE;
        }
        if (cc >= GGML_CUDA_CC_TURING) {
            return get_mmvq_mmid_max_batch_turing_plus(type);
        }
        return get_mmvq_mmid_max_batch_pascal_older(type);
    }

    // AMD
    if (GGML_CUDA_CC_IS_AMD(cc)) {
        if (GGML_CUDA_CC_IS_RDNA4(cc)) {
            return mmvq_mmid_max_batch_band(get_mmvq_mmid_max_batch_rdna4(type));
        }
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            return mmvq_mmid_max_batch_band(get_mmvq_mmid_max_batch_rdna3(type));
        }
        if (GGML_CUDA_CC_IS_RDNA1(cc) || GGML_CUDA_CC_IS_RDNA2(cc)) {
            return mmvq_mmid_max_batch_band(get_mmvq_mmid_max_batch_rdna1_rdna2(type));
        }
        if (GGML_CUDA_CC_IS_CDNA(cc)) {
            return mmvq_mmid_max_batch_band(get_mmvq_mmid_max_batch_cdna(type));
        }
        if (GGML_CUDA_CC_IS_GCN(cc)) {
            return mmvq_mmid_max_batch_band(get_mmvq_mmid_max_batch_gcn(type));
        }
    }
    return MMVQ_MAX_BATCH_SIZE;
}

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11) {
    if (!ggml_is_quantized(type)) {
        return false;
    }
    // k-quants cost more to decode and mvq redoes that per column, so MMQ wins sooner.
    // Only list quant-types MMQ supports, others would fall back to cuBLAS.
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_ADA_LOVELACE) {
        switch (type) { // tuned on RTX 4090
            case GGML_TYPE_Q2_K:
                return ne11 <= 4;
            case GGML_TYPE_Q3_K:
                return ne11 <= 6;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_BLACKWELL) {
        switch (type) { // tuned on RTX 5090
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
                return ne11 <= 5;
            case GGML_TYPE_Q5_K:
                return ne11 <= 6;
            case GGML_TYPE_Q6_K:
                return ne11 <= 7;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_DGX_SPARK) {
        switch (type) { // tuned on DGX Spark GB10
            case GGML_TYPE_Q2_K:
                return ne11 <= 6;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_ORIN) {
        switch (type) { // tuned for Jetson Orin
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q6_K:
                return ne11 <= 1;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_VOLTA) {
        switch (type) {
            case GGML_TYPE_Q2_K:
                return ne11 <= 4;
            case GGML_TYPE_Q3_K:
                return ne11 <= 6;
            case GGML_TYPE_Q4_K:
                return ne11 <= 5;
            case GGML_TYPE_Q5_K:
                return ne11 <= 6;
            case GGML_TYPE_Q6_K:
                return ne11 <= 7;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_CDNA(cc)) {
        if (GGML_CUDA_CC_IS_CDNA1(cc)) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q5_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q8_0:
                    return ne11 <= 6;
                case GGML_TYPE_Q2_K:
                    return ne11 <= 4;
                case GGML_TYPE_Q3_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q4_K:
                    return ne11 <= 2;
                case GGML_TYPE_Q5_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q6_K:
                    return ne11 <= 4;
                case GGML_TYPE_IQ1_S:
                    return ne11 <= 5;
                case GGML_TYPE_IQ2_XXS:
                case GGML_TYPE_IQ3_S:
                case GGML_TYPE_IQ4_XS:
                    return ne11 <= 6;
                default:
                    return ne11 <= MMVQ_MAX_BATCH_SIZE;
            }
        }
        switch (type) { // tuned for CDNA2
            case GGML_TYPE_Q2_K:
                return ne11 <= 5;
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
                return ne11 <= 3;
            case GGML_TYPE_Q6_K:
                return ne11 <= 5;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    return ne11 <= MMVQ_MAX_BATCH_SIZE;
}

// Device constexpr: returns the max batch size for the current arch+type at compile time.
template <ggml_type type>
static constexpr __device__ int get_mmvq_mmid_max_batch_for_device() {
#if defined(RDNA4)
    return mmvq_mmid_max_batch_band(get_mmvq_mmid_max_batch_rdna4(type));
#elif defined(RDNA3)
    return mmvq_mmid_max_batch_band(get_mmvq_mmid_max_batch_rdna3(type));
#elif defined(RDNA2) || defined(RDNA1)
    return mmvq_mmid_max_batch_band(get_mmvq_mmid_max_batch_rdna1_rdna2(type));
#elif defined(CDNA)
    return mmvq_mmid_max_batch_band(get_mmvq_mmid_max_batch_cdna(type));
#elif defined(GCN)
    return mmvq_mmid_max_batch_band(get_mmvq_mmid_max_batch_gcn(type));
#elif !defined(GGML_USE_MUSA) && (__CUDA_ARCH__ == GGML_CUDA_CC_VOLTA || __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE)
    return MMVQ_MAX_BATCH_SIZE;
#elif !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
    return get_mmvq_mmid_max_batch_turing_plus(type);
#else
    return get_mmvq_mmid_max_batch_pascal_older(type);
#endif
}

static constexpr __host__ __device__ int calc_nwarps(ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id, bool small_k = false, bool halve_iters = false) {
    if (table_id == MMVQ_PARAMETERS_GENERIC) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    } else if (table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 2;
            case 5:
            case 6:
            case 7:
            case 8:
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_RDNA4) {
        // Band-uniform nwarps=1 for the whole mmvq band (ncols_dst 1..8).
        // nwarps participates in the K-split accumulation order, so decode and the
        // speculative verify batch must use the same value.  The Q8_0 weight kernel
        // gets a per-(type,K) override from calc_nwarps_weight() below; every other
        // launch (including the GDN/SSM and shared-expert fusions) stays at 1.
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA3_0) {
        // RDNA3 (W7900): stricter whitelist than RDNA4.
        // Q2_K / Q5_K / IQ4_XS regress in full quant sweeps.
        // Apply to the whole mmvq range (ncols_dst 1..8), not just decode: the
        // speculative verify batch (n_draft+1 tokens) must use the same nwarps
        // as decode so its per-row dot-product accumulation is bit-identical.
        if (ncols_dst <= MMVQ_MAX_BATCH_SIZE) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                    return 8;
                case GGML_TYPE_Q6_K:
                    // gfx1100 sweep 2026-08-28 (rdna3-boosts R5): nwarps=8 beats
                    // the old 2 by +0.8-1.0% decode; the other widened types
                    // (Q2_K/Q4_K/Q5_K/IQ4_XS at 8) regressed -4..-7% and stay 1.
                    return 8;
                case GGML_TYPE_IQ4_NL:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA3_5) {
        // gfx1151 (Strix Halo iGPU): nwarps=1 (the RDNA2 table) underutilizes the
        // wave32 datapath on the large-K decode matmuls; nwarps=8 (the RDNA3_0
        // table) over-parallelizes the small ones. Swept 2025-08: nwarps=2 wins
        // (~+0.6% decode on Qwen3.6-35B-A3B Q8_0), nwarps=4 regresses.
        // Apply to the whole mmvq range (ncols_dst 1..8), not just decode: the
        // speculative verify batch (n_draft+1 tokens) must use the same nwarps
        // as decode so its per-row dot-product accumulation is bit-identical.
        if (ncols_dst <= MMVQ_MAX_BATCH_SIZE) {
            switch (type) {
                case GGML_TYPE_Q8_0:
                    return 2;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_TURING) {
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q3_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                    return 2;
                default:
                    return 4;
            }
        }
        switch (ncols_dst) {
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_GB10) {
        const int generic = calc_nwarps(type, ncols_dst, MMVQ_PARAMETERS_GENERIC);
        // Only worth the wider block when it actually retires the K loop in half the trips (Observation)
        if (ncols_dst == 1 && !small_k && halve_iters) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                    return 2 * generic;
                default:
                    break;
            }
        }
        return generic;
    }
    return 1;
}

// nwarps for a dense mmvq *weight* launch (the ksplit kernel).  A Q8_0 weight with a
// short K (the MoE attention qkv/gate and the lm_head, K < 4096) wins from the
// pre-2026-09-11 wider block; the long-K Q8_0 hybrid projections (dense 27B
// attn_v/ssm_out, K >= 5120) lose from it and shift the MTP arithmetic, so they stay
// at 1.  The choice is per tensor shape (K is fixed for a weight), so W = 1..8 of the
// same tensor still agree.  Fusion ops (GDN/SSM, shared-expert, the gate fusions)
// MUST keep plain calc_nwarps -- their nwarps is a single-token reduction-order anchor.
static constexpr __host__ __device__ int calc_nwarps_weight(ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id, bool long_k) {
    if (table_id == MMVQ_PARAMETERS_RDNA4 && type == GGML_TYPE_Q8_0 && !long_k && ncols_dst <= MMVQ_MAX_BATCH_SIZE) {
        return 8;
    }
    return calc_nwarps(type, ncols_dst, table_id);
}

static constexpr __host__ __device__ int calc_rows_per_block(int ncols_dst, int table_id, bool small_k = false, int nwarps = 1) {
    if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_GCN || table_id == MMVQ_PARAMETERS_TURING || table_id == MMVQ_PARAMETERS_GB10) {
        switch (ncols_dst) {
            case 1:
                return small_k ? nwarps : 1;
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    return 1;
}

// rows_per_block override for dispatch-selected kernels (e.g. the small-K MoE
// down projection on RDNA, where a single 544 B row per block underutilizes the
// memory system). 0 = use calc_rows_per_block().
static constexpr __host__ __device__ int calc_rows_per_block_override(int rows_per_block, int ncols_dst, int table_id, bool small_k, int nwarps) {
    return rows_per_block > 0 ? rows_per_block : calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
}

// rows per block for a dense mmvq *weight* launch (the ksplit kernel).  On RDNA4 a
// multi-token (verify) batch computes several rows per block so the ncols_dst q8_1
// activation columns are loaded once per block instead of once per row.  Each row's
// accumulation (thread mapping, K order, warp reduction) is unchanged, so W = 1..8
// stay bit-identical.
#ifndef GGML_MMVQ_RDNA4_WEIGHT_RPB
#define GGML_MMVQ_RDNA4_WEIGHT_RPB 4
#endif
// Multi-row blocks only pay off while the launch still has enough blocks to fill the GPU:
// with fewer than this many blocks (e.g. 1024-row K/V or 48-row SSM projections at 4 rows)
// the one-row launch is faster.
#ifndef GGML_MMVQ_RDNA4_WEIGHT_MIN_BLOCKS
#define GGML_MMVQ_RDNA4_WEIGHT_MIN_BLOCKS 512
#endif
static constexpr __host__ __device__ int calc_rows_per_block_weight(ggml_type type, int ncols_dst, int table_id, bool small_k, int nwarps) {
    if (table_id == MMVQ_PARAMETERS_RDNA4 && ncols_dst >= 2 && ncols_dst <= MMVQ_MAX_BATCH_SIZE) {
        constexpr int rpb_max = GGML_MMVQ_RDNA4_WEIGHT_RPB;
        switch (type) {
            // gfx1201 sweep (rows 1/2/4 x every type x n = 2..8 x 4 shapes): these gain
            // nothing from more rows (NVFP4 loses 40-60 %, Q1_0 ~10 %) and keep one row.
            case GGML_TYPE_NVFP4:
            case GGML_TYPE_Q1_0:
                return 1;
            // these peak at 2 rows
            case GGML_TYPE_IQ2_XXS:
            case GGML_TYPE_IQ2_XS:
            case GGML_TYPE_IQ3_XXS:
            case GGML_TYPE_Q2_0:
            case GGML_TYPE_Q5_1:
                return rpb_max < 2 ? rpb_max : 2;
            // Q2_K's dot product exceeds the 256-VGPR budget at 4 rows x 7..8 columns and
            // spills to scratch; it keeps 2 rows there.
            case GGML_TYPE_Q2_K:
                return ncols_dst > 6 && rpb_max > 2 ? 2 : rpb_max;
            default:
                return rpb_max;
        }
    }
    return calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
}

template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool halve_iters = false, int rows_per_block = 0>
__launch_bounds__(calc_nwarps(type, ncols_dst, get_device_table_id(), small_k, halve_iters)*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion, float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t ids_stride) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int nwarps = calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);
    constexpr int rows_per_cuda_block = calc_rows_per_block_override(rows_per_block, ncols_dst, table_id, small_k, nwarps);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const     int tid = warp_size*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;

    const uint32_t channel_dst = blockIdx.y;

    uint32_t channel_x;
    uint32_t channel_y;
    uint32_t sample_dst;

    ggml_cuda_pdl_sync();
    channel_x  = ncols_dst == 1 && ids ? ids[channel_dst]                     : fastdiv(channel_dst, channel_ratio);
    channel_y  = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    sample_dst = blockIdx.z;

    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    bool use_scale = false;
    bool use_gate_scale = false;
    bool use_dst_gate = false;
    bool use_conv_input = false;
    [[maybe_unused]] const void * vgate = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    const float * x_scale = nullptr;
    const float * gate_scale = nullptr;
    [[maybe_unused]] const void * dst_gate = nullptr;
    [[maybe_unused]] float * conv_input = nullptr;
    [[maybe_unused]] const float * conv_states = nullptr;
    [[maybe_unused]] int conv_kernel_size = 0;
    ggml_glu_op active_glu;
    float glu_limit = 0.0f;

    if constexpr (has_fusion) {
        use_gate      = fusion.gate      != nullptr;
        use_bias      = fusion.x_bias    != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr && use_gate;
        vgate         = fusion.gate;
        x_bias        = (const float *) fusion.x_bias;
        gate_bias     = (const float *) fusion.gate_bias;
        active_glu    = fusion.glu_op;
        glu_limit     = fusion.glu_limit;
        use_dst_gate  = fusion.dst_gate != nullptr && use_gate;
        if (use_dst_gate) {
            dst_gate = fusion.dst_gate;
        }
        use_conv_input = fusion.conv_input != nullptr && fusion.conv_states != nullptr;
        if (use_conv_input) {
            conv_input       = (float *) fusion.conv_input;
            conv_states      = (const float *) fusion.conv_states;
            conv_kernel_size = fusion.conv_kernel_size;
        }
        if constexpr (type == GGML_TYPE_NVFP4) {
            use_scale      = fusion.x_scale    != nullptr;
            use_gate_scale = fusion.gate_scale != nullptr && use_gate;
            x_scale        = (const float *) fusion.x_scale;
            gate_scale     = (const float *) fusion.gate_scale;
        }
        // Per-token scale (MoE down x topk weights). Indexed by channel_dst.
        if (fusion.x_scale_channel_dst) {
            use_scale = true;
            x_scale   = (const float *) fusion.x_scale;
        }
    }


    [[maybe_unused]] float x_biases[ncols_dst]    = { 0.0f };
    [[maybe_unused]] float gate_biases[ncols_dst] = { 0.0f };
    [[maybe_unused]] float x_scales = 1.0f;
    [[maybe_unused]] float gate_scales = 1.0f;
    if constexpr (has_fusion) {
        // 1. Hide latency by prefetching bias, gates and scales here
        // 2. load only on threads that won't die after partial sum calculation
        const uint32_t channel_bias = ids ? channel_x : channel_dst;
        if (threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
            (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
            if (use_bias) {
                x_bias = x_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    x_biases[j] = x_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if (use_gate_bias) {
                gate_bias = gate_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    gate_biases[j] = gate_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if (use_scale) {
                x_scales = fusion.x_scale_channel_dst ? x_scale[channel_dst] : x_scale[ids ? channel_x : 0];
            }
            if (use_gate_scale) {
                gate_scales = gate_scale[ids ? channel_x : 0];
            }
        }
    }

    // partial sum for each thread
    float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};
    float tmp_gate[ncols_dst][rows_per_cuda_block] = {{0.0f}};

    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;

    // split (token, row, kblock) items across the thread groups: the plain K-split
    // loop leaves most groups idle on short-K decode GEMMs (MoE down K=512, router
    // K=2048, qkv K=8192), wasting ~3/4 of the issue slots on RDNA4
    const int n_items  = ncols_dst * rows_per_cuda_block * blocks_per_row_x;
    const int n_groups = nwarps * warp_size / (qi/vdr);
    const int kqs      = vdr * (tid % (qi/vdr)); // x block quant index when casting the quants to int
    for (int it = tid / (qi/vdr); it < n_items; it += n_groups) {
        const int j   = it / (rows_per_cuda_block * blocks_per_row_x);
        const int rem = it % (rows_per_cuda_block * blocks_per_row_x);
        const int i   = rem / blocks_per_row_x;
        const int kbx = rem % blocks_per_row_x;

        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        tmp[j][i] += vec_dot_q_cuda(
            vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
        if constexpr (has_fusion) {
            if (use_gate) {
                tmp_gate[j][i] += vec_dot_q_cuda(
                    vgate, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
            }
        }
    }

    // cross-warp combine: each warp reduces its own lanes first (deterministic
    // tree), then one value per warp is summed serially by warp 0. The shared
    // footprint no longer scales with ncols_dst x rows_per_block, which lets the
    // short-K rows_per_block override fill the thread groups at any batch size.
    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block];
    [[maybe_unused]] __shared__ float tmp_shared_gate[(has_fusion && (nwarps-1 > 0)) ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block];

    {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] = warp_reduce_sum<warp_size>(tmp_gate[j][i]);
                    }
                }
                if (threadIdx.y > 0) {
                    tmp_shared[threadIdx.y-1][j][i] = tmp[j][i];
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            tmp_shared_gate[threadIdx.y-1][j][i] = tmp_gate[j][i];
                        }
                    }
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += tmp_shared_gate[l][j][i];
                    }
                }
            }

            float result_val = 0.0f;
            if (threadIdx.x == i && (rows_per_cuda_block == 1 || uint32_t(row0 + i) < stride_col_dst)) {
                result_val = tmp[j][i];
                if constexpr (has_fusion) {
                    if (use_scale) {
                        result_val *= x_scales;
                    }
                    result_val += x_biases[j];
                    if (use_gate) {
                        float gate_value = tmp_gate[j][i];
                        if constexpr (type == GGML_TYPE_NVFP4) {
                            gate_value *= gate_scales;
                        }
                        gate_value += gate_biases[j];
                        if (use_dst_gate) {
                            // separate output: write the gate result to its own destination
                            // (dst was already offset by sample/channel/row; apply the same to dst_gate)
                            float * dst_gate_row = (float *) dst_gate + sample_dst*stride_sample_dst +
                                                   channel_dst*stride_channel_dst + row0 + j*stride_col_dst;
                            dst_gate_row[i] = gate_value;
                        } else {
                            switch (active_glu) {
                                case GGML_GLU_OP_SWIGLU:
                                    result_val *= ggml_cuda_op_silu_single(gate_value);
                                    break;
                                case GGML_GLU_OP_GEGLU:
                                    result_val *= ggml_cuda_op_gelu_single(gate_value);
                                    break;
                                case GGML_GLU_OP_SWIGLU_OAI:
                                    result_val = ggml_cuda_op_swiglu_oai_single(gate_value, result_val);
                                    break;
                                case GGML_GLU_OP_SWIGLU_CLAMP:
                                    result_val = ggml_cuda_op_swiglu_clamp_single(gate_value, result_val, glu_limit);
                                    break;
                                default:
                                    result_val = result_val * gate_value;
                                    break;
                            }
                        }
                    }
                }
            }
            if (use_conv_input) {
                // interleaved conv input: [state0, state1, ..., state_{cs-2}, result] per
                // channel. conv_input [cs, C] has nb1 = cs floats, conv_states [(cs-1), C]
                // has nb1 = cs-1 floats. Spread the 4 writes over threads 0..cs-1.
                const float r_bcast = __shfl_sync(0xffffffff, result_val, i, warp_size);
                const int c = row0 + i;
                const int cs = conv_kernel_size;
                if (threadIdx.x < cs) {
                    const int k = threadIdx.x;
                    const float v = k < cs - 1 ? conv_states[(cs-1)*c + k] : r_bcast;
                    conv_input[cs*c + k] = v;
                }
            } else if (threadIdx.x == i && (rows_per_cuda_block == 1 || uint32_t(row0 + i) < stride_col_dst)) {
                dst[j*stride_col_dst + i] = result_val;
            }
        }
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, use_scale, use_gate_scale, use_dst_gate, use_conv_input, active_glu, glu_limit, gate_bias, x_bias, x_scale, gate_scale, tmp_gate, dst_gate, conv_input, conv_states, conv_kernel_size);
    }
    if constexpr (type != GGML_TYPE_NVFP4) {
        GGML_UNUSED_VARS(use_scale, use_gate_scale, x_scale, gate_scale, x_scales, gate_scales);
    }
}

// ---------------------------------------------------------------------------
// ksplit variant of mul_mat_vec_q: the pre-block-13 K-split accumulation.
// Block 13's item-split kernel wins for ncols_dst == 1 short-K rows (MoE
// down/router) but its per-thread accumulator fan-out (tmp[ncols_dst][rpb])
// collapses multi-token decode batches (ncols_dst 2..8, e.g. the speculative
// verify step). Those batches dispatch here instead. Kept bit-identical to the
// 12-block release kernel (same per-row chunk->group mapping as the old
// ncols_dst == 1 path).
// ---------------------------------------------------------------------------
template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool halve_iters = false, int rows_per_block = 0, bool long_k = false>
__launch_bounds__(calc_nwarps_weight(type, ncols_dst, get_device_table_id(), long_k)*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_ksplit(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion, float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t ids_stride) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int nwarps = calc_nwarps_weight(type, ncols_dst, table_id, long_k);
    constexpr int rows_per_cuda_block = rows_per_block > 0 ? rows_per_block : calc_rows_per_block_weight(type, ncols_dst, table_id, small_k, nwarps);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const     int tid = warp_size*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;

    const bool shared_expert = has_fusion && fusion.shared_up && blockIdx.y == gridDim.y - 1;
    const uint32_t channel_dst = shared_expert ? 0 : blockIdx.y;
    if (shared_expert) {
        vx = fusion.shared_up;
        dst = fusion.shared_dst;
        stride_col_dst = fusion.shared_stride_col_dst;
    }

    uint32_t channel_x;
    uint32_t channel_y;
    uint32_t sample_dst;

    ggml_cuda_pdl_sync();
    channel_x  = shared_expert ? 0 : ncols_dst == 1 && ids ? ids[channel_dst] : fastdiv(channel_dst, channel_ratio);
    channel_y  = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    sample_dst = blockIdx.z;

    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    bool use_scale = false;
    bool use_gate_scale = false;
    bool use_dst_gate = false;
    bool use_conv_input = false;
    [[maybe_unused]] const void * vgate = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    const float * x_scale = nullptr;
    const float * gate_scale = nullptr;
    [[maybe_unused]] const void * dst_gate = nullptr;
    [[maybe_unused]] float * conv_input = nullptr;
    [[maybe_unused]] const float * conv_states = nullptr;
    [[maybe_unused]] int conv_kernel_size = 0;
    ggml_glu_op active_glu;
    float glu_limit = 0.0f;

    if constexpr (has_fusion) {
        use_gate      = fusion.gate      != nullptr;
        use_bias      = fusion.x_bias    != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr && use_gate;
        vgate         = shared_expert ? fusion.shared_gate : fusion.gate;
        x_bias        = (const float *) fusion.x_bias;
        gate_bias     = (const float *) fusion.gate_bias;
        active_glu    = fusion.glu_op;
        glu_limit     = fusion.glu_limit;
        use_dst_gate  = fusion.dst_gate != nullptr && use_gate;
        if (use_dst_gate) {
            dst_gate = fusion.dst_gate;
        }
        use_conv_input = fusion.conv_input != nullptr && fusion.conv_states != nullptr;
        if (use_conv_input) {
            conv_input       = (float *) fusion.conv_input;
            conv_states      = (const float *) fusion.conv_states;
            conv_kernel_size = fusion.conv_kernel_size;
        }
        if constexpr (type == GGML_TYPE_NVFP4) {
            use_scale      = fusion.x_scale    != nullptr;
            use_gate_scale = fusion.gate_scale != nullptr && use_gate;
            x_scale        = (const float *) fusion.x_scale;
            gate_scale     = (const float *) fusion.gate_scale;
        }
        // Per-token scale (MoE down x topk weights). Indexed by channel_dst.
        if (fusion.x_scale_channel_dst) {
            use_scale = true;
            x_scale   = (const float *) fusion.x_scale;
        }
    }


    [[maybe_unused]] float x_biases[ncols_dst]    = { 0.0f };
    [[maybe_unused]] float gate_biases[ncols_dst] = { 0.0f };
    [[maybe_unused]] float x_scales = 1.0f;
    [[maybe_unused]] float gate_scales = 1.0f;
    if constexpr (has_fusion) {
        // 1. Hide latency by prefetching bias, gates and scales here
        // 2. load only on threads that won't die after partial sum calculation
        const uint32_t channel_bias = ids ? channel_x : channel_dst;
        if (threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
            (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
            if (use_bias) {
                x_bias = x_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    x_biases[j] = x_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if (use_gate_bias) {
                gate_bias = gate_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    gate_biases[j] = gate_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if (use_scale) {
                x_scales = fusion.x_scale_channel_dst ? x_scale[channel_dst] : x_scale[ids ? channel_x : 0];
            }
            if (use_gate_scale) {
                gate_scales = gate_scale[ids ? channel_x : 0];
            }
        }
    }

    // partial sum for each thread
    float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};
    float tmp_gate[ncols_dst][rows_per_cuda_block] = {{0.0f}};

    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

#if __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
        // start the next iterations' weight loads early
        if constexpr (mmvq_should_prefetch(type)) {
            constexpr int pf_dist = 2; // loop iterations, not blocks
            const int kbx_pf = kbx + pf_dist*blocks_per_iter;
            if (kbx_pf < blocks_per_row_x) {
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    const size_t off = (size_t)(kbx_offset + i*stride_row_x + kbx_pf) * ggml_cuda_type_traits<type>::bs;
                    mmvq_prefetch_l2((const char *) vx + off);
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            mmvq_prefetch_l2((const char *) vgate + off);
                        }
                    }
                }
            }
        }
#endif

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp[j][i] += vec_dot_q_cuda(
                    vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += vec_dot_q_cuda(
                            vgate, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                    }
                }
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
    [[maybe_unused]] __shared__ float tmp_shared_gate[(has_fusion && (nwarps-1 > 0)) ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];

    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_shared_gate[threadIdx.y-1][j][i][threadIdx.x] = tmp_gate[j][i];
                    }
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += tmp_shared_gate[l][j][i][threadIdx.x];
                    }
                }
            }
            tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[j][i] = warp_reduce_sum<warp_size>(tmp_gate[j][i]);
                }
            }

            float result_val = 0.0f;
            if (threadIdx.x == i && (rows_per_cuda_block == 1 || uint32_t(row0 + i) < stride_col_dst)) {
                result_val = tmp[j][i];
                if constexpr (has_fusion) {
                    if (use_scale) {
                        result_val *= x_scales;
                    }
                    result_val += x_biases[j];
                    if (use_gate) {
                        float gate_value = tmp_gate[j][i];
                        if constexpr (type == GGML_TYPE_NVFP4) {
                            gate_value *= gate_scales;
                        }
                        gate_value += gate_biases[j];
                        if (use_dst_gate) {
                            // separate output: write the gate result to its own destination
                            // (dst was already offset by sample/channel/row; apply the same to dst_gate)
                            float * dst_gate_row = (float *) dst_gate + sample_dst*stride_sample_dst +
                                                   channel_dst*stride_channel_dst + row0 + j*stride_col_dst;
                            dst_gate_row[i] = gate_value;
                        } else {
                            switch (active_glu) {
                                case GGML_GLU_OP_SWIGLU:
                                    result_val *= ggml_cuda_op_silu_single(gate_value);
                                    break;
                                case GGML_GLU_OP_GEGLU:
                                    result_val *= ggml_cuda_op_gelu_single(gate_value);
                                    break;
                                case GGML_GLU_OP_SWIGLU_OAI:
                                    result_val = ggml_cuda_op_swiglu_oai_single(gate_value, result_val);
                                    break;
                                case GGML_GLU_OP_SWIGLU_CLAMP:
                                    result_val = ggml_cuda_op_swiglu_clamp_single(gate_value, result_val, glu_limit);
                                    break;
                                default:
                                    result_val = result_val * gate_value;
                                    break;
                            }
                        }
                    }
                }
            }
            if (use_conv_input) {
                // interleaved conv input: [state0, state1, ..., state_{cs-2}, result] per
                // channel. conv_input [cs, C] has nb1 = cs floats, conv_states [(cs-1), C]
                // has nb1 = cs-1 floats. Spread the 4 writes over threads 0..cs-1.
                const float r_bcast = __shfl_sync(0xffffffff, result_val, i, warp_size);
                const int c = row0 + i;
                const int cs = conv_kernel_size;
                if (threadIdx.x < cs) {
                    const int k = threadIdx.x;
                    const float v = k < cs - 1 ? conv_states[(cs-1)*c + k] : r_bcast;
                    conv_input[cs*c + k] = v;
                }
            } else if (threadIdx.x == i && (rows_per_cuda_block == 1 || uint32_t(row0 + i) < stride_col_dst)) {
                dst[j*stride_col_dst + i] = result_val;
            }
        }
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, use_scale, use_gate_scale, use_dst_gate, use_conv_input, active_glu, glu_limit, gate_bias, x_bias, x_scale, gate_scale, tmp_gate, dst_gate, conv_input, conv_states, conv_kernel_size);
    }
    if constexpr (type != GGML_TYPE_NVFP4) {
        GGML_UNUSED_VARS(use_scale, use_gate_scale, x_scale, gate_scale, x_scales, gate_scales);
    }
}

// Dedicated MoE multi-token kernel.
// Grid: (ceil(nrows_x / c_rows_per_block), nchannels_dst)
// Block: (warp_size, ncols_dst) - each warp handles one token independently.
// No shared memory reduction needed since each warp works alone.
// The block is (warp_size, ncols_dst) and ncols_dst spans the whole decode/verify band, and this
// kernel is the one that keeps the band on a single accumulation order (see the has_ids branch in
// mul_mat_vec_q_switch_ncols_dst), so its launch bound must cover the band rather than the per-type
// cap -- otherwise a launch with ncols_dst > cap fails outright.
template <ggml_type type, int c_rows_per_block, bool has_fusion = false>
__launch_bounds__(MMVQ_MAX_BATCH_SIZE*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_moe(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion,
        float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride, const uint32_t nchannels_dst,
        const void * cold_base, const uint32_t n_res_cold,
        const uint32_t cold_channel_stride, const uint32_t cold_row_stride,
        const void * cold_gate_base, const uint32_t cold_gate_n_res,
        const uint32_t cold_gate_channel_stride, const uint32_t cold_gate_row_stride) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type, true);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type, true);

    const bool shared_expert = has_fusion && fusion.shared_up && blockIdx.y == gridDim.y - 1;
    if (shared_expert) {
        vx = fusion.shared_up;
        dst = fusion.shared_dst;
        stride_col_dst = fusion.shared_stride_col_dst;
    }

    // fuse gate, bias, scales, and glu_op into the up projection
    bool use_gate = false;
    const void  * vgate      = nullptr;
    const float * x_bias     = nullptr;
    const float * gate_bias  = nullptr;
    const float * x_scale    = nullptr;
    const float * gate_scale = nullptr;
    ggml_glu_op   active_glu = GGML_GLU_OP_SWIGLU;
    float         glu_limit  = 0.0f;

    if constexpr (has_fusion) {
        use_gate   = fusion.gate != nullptr;
        vgate      = shared_expert ? fusion.shared_gate : fusion.gate;
        x_bias     = (const float *) fusion.x_bias;
        gate_bias  = (const float *) fusion.gate_bias;
        active_glu = fusion.glu_op;
        glu_limit  = fusion.glu_limit;
        if constexpr (type == GGML_TYPE_NVFP4) {
            x_scale    = (const float *) fusion.x_scale;
            gate_scale = (const float *) fusion.gate_scale;
        }
        // Per-(expert, token) MoE scale (down x topk weights). Indexed by
        // channel_dst + token_idx*nchannels_dst, one scalar per (expert, token).
        if (fusion.x_scale_channel_dst) {
            x_scale = (const float *) fusion.x_scale;
        }
    }

    const uint32_t token_idx   = threadIdx.y;
    const int      row0        = c_rows_per_block*blockIdx.x;
    const int      blocks_per_row_x = ncols_x / qk;

    const uint32_t channel_dst = shared_expert ? 0 : blockIdx.y;

    if (token_idx >= ncols_dst) {
        return;
    }

    ggml_cuda_pdl_sync();
    // The shared-expert lane reads its own table at channel 0; its routing id is not in `ids`,
    // so the cache slot lookup and the UVA cold seam apply to the routed lanes only.
    const int32_t raw_id = shared_expert ? 0 : ids[channel_dst + token_idx * ids_stride];
    const uint32_t channel_x_raw = (uint32_t) raw_id;
    // Phase 1b/3 UVA cold read.  An id >= n_res_cold is not a VRAM slot but host expert `id - n_res_cold`,
    // read in place from the pinned host alias.  The host geometry (per-expert stride `cold_channel_stride`,
    // per-row stride `cold_row_stride`) is returned by `moe_cache_get_cold`: for an unsplit or contiguous
    // axis-1 slice the row stride matches the arena (`cold_row_stride == 0`), while a strided axis-0 slice
    // uses the host pitch.  The default (cold_base == nullptr, n_res_cold == 0) leaves the resident-only
    // 1a contract untouched.
    const bool     cold      = !shared_expert && cold_base != nullptr && channel_x_raw >= n_res_cold;
    const void *   vx_use    = cold ? cold_base : vx;
    const uint32_t channel_x = shared_expert ? 0u : (cold ? (channel_x_raw - n_res_cold) : channel_x_raw);
    const uint32_t stride_channel_x_use = cold ? cold_channel_stride : stride_channel_x;
    const uint32_t stride_row_x_use     = (cold && cold_row_stride != 0) ? cold_row_stride : stride_row_x;
    // The fused gate lane is a SEPARATE expert table with its own host geometry (same raw routing id,
    // but its own slot count and pinned host master), so resolve it independently of the up lane.
    const bool     gate_cold        = !shared_expert && cold_gate_base != nullptr && channel_x_raw >= cold_gate_n_res;
    const void *   vgate_use        = gate_cold ? cold_gate_base : vgate;
    const uint32_t gate_channel_x   = shared_expert ? 0u : (gate_cold ? (channel_x_raw - cold_gate_n_res) : channel_x_raw);
    const uint32_t gate_channel_x_st = gate_cold ? cold_gate_channel_stride : stride_channel_x;
    const uint32_t gate_row_x_st     = (gate_cold && cold_gate_row_stride != 0) ? cold_gate_row_stride : stride_row_x;
    const uint32_t channel_y = fastmodulo(channel_dst, nchannels_y);

    const block_q8_1 * y = ((const block_q8_1 *) vy) + channel_y*stride_channel_y + token_idx*stride_col_y;
    const int kbx_offset      = channel_x*stride_channel_x_use + row0*stride_row_x_use;
    const int kbx_offset_gate = gate_channel_x*gate_channel_x_st + row0*gate_row_x_st;

    // partial sum for each thread
    float tmp[c_rows_per_block] = {0.0f};
    float tmp_gate[c_rows_per_block] = {0.0f};

    // split (row, kblock) items across the 8 thread groups: the K-split loop
    // leaves most groups idle on short-K MoE GEMMs (down K=512 -> 2 K-blocks)
    //
    // Row-loop clamp: the last row-block may hold fewer than `c_rows_per_block` valid rows, but the loop
    // below used to read all of them (only the write was guarded).  The dead loads walk `rpb - rem` rows
    // past the expert's last row -- into the next expert, and off the end of the tensor for the last
    // expert of the last host tensor (a cold read sources the pinned master in place).  Clamping the item
    // count skips them and leaves the valid-row accumulation order unchanged.
    const int rows_left  = row0 < (int) nrows_x ? (int) nrows_x - row0 : 0;
    const int rows_valid = rows_left < c_rows_per_block ? rows_left : c_rows_per_block;
    const int n_items    = rows_valid * blocks_per_row_x;
    const int n_groups = warp_size / (qi/vdr);
    const int kqs      = vdr * (threadIdx.x % (qi/vdr));
    for (int it = threadIdx.x / (qi/vdr); it < n_items; it += n_groups) {
        const int i   = it / blocks_per_row_x;
        const int kbx = it % blocks_per_row_x;
        const int kby = kbx * (qk/QK8_1);
        tmp[i] += vec_dot_q_cuda(vx_use, &y[kby], kbx_offset + i*stride_row_x_use + kbx, kqs);
        if constexpr (has_fusion) {
            if (use_gate) {
                tmp_gate[i] += vec_dot_q_cuda(vgate_use, &y[kby], kbx_offset_gate + i*gate_row_x_st + kbx, kqs);
            }
        }
    }

    ggml_cuda_pdl_lc();

    // Warp-level reduction only - no shared memory needed
#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
        if constexpr (has_fusion) {
            if (use_gate) {
                tmp_gate[i] = warp_reduce_sum<warp_size>(tmp_gate[i]);
            }
        }
    }

    // Write results
    if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
        float result = tmp[threadIdx.x];
        if constexpr (has_fusion) {
            const uint32_t bias_idx = channel_x*stride_channel_dst + row0 + threadIdx.x;

            if constexpr (type == GGML_TYPE_NVFP4) {
                if (x_scale) {
                    result *= x_scale[channel_x];
                }
            }
            if (fusion.x_scale_channel_dst) {
                result *= x_scale[channel_dst + token_idx * nchannels_dst];
            }
            if (x_bias) {
                result += x_bias[bias_idx];
            }
            if (use_gate) {
                float gate_value = tmp_gate[threadIdx.x];
                if constexpr (type == GGML_TYPE_NVFP4) {
                    if (gate_scale) {
                        gate_value *= gate_scale[channel_x];
                    }
                }
                if (gate_bias) {
                    gate_value += gate_bias[bias_idx];
                }
                switch (active_glu) {
                    case GGML_GLU_OP_SWIGLU:
                        result *= ggml_cuda_op_silu_single(gate_value);
                        break;
                    case GGML_GLU_OP_GEGLU:
                        result *= ggml_cuda_op_gelu_single(gate_value);
                        break;
                    case GGML_GLU_OP_SWIGLU_OAI:
                        result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                        break;
                    case GGML_GLU_OP_SWIGLU_CLAMP:
                        result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, glu_limit);
                        break;
                    default:
                        result = result * gate_value;
                        break;
                }
            }
        }
        dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = result;
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, tmp_gate, vgate, x_bias, gate_bias, active_glu, glu_limit, x_scale, gate_scale);
    } else if constexpr (type != GGML_TYPE_NVFP4) {
        GGML_UNUSED_VARS(x_scale, gate_scale);
    }
}

template<ggml_type type, bool weight_rule = false>
static std::pair<dim3, dim3> calc_launch_params(
        const int ncols_dst, const int nrows_x, const int nchannels_dst, const int nsamples_or_ntokens,
        const int warp_size, const mmvq_parameter_table_id table_id, const bool small_k = false, const bool halve_iters = false, const bool long_k = false) {
    const int nwarps = weight_rule ? calc_nwarps_weight(type, ncols_dst, table_id, long_k) : calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);
    const int rpb = weight_rule ? calc_rows_per_block_weight(type, ncols_dst, table_id, small_k, nwarps) : calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
    const int64_t nblocks = (nrows_x + rpb - 1) / rpb;
    const dim3 block_nums(nblocks, nchannels_dst, nsamples_or_ntokens);
    const dim3 block_dims(warp_size, nwarps, 1);
    return {block_nums, block_dims};
}

template<ggml_type type, int c_ncols_dst, bool small_k = false, bool halve_iters = false, int rows_per_block = 0>
static void mul_mat_vec_q_switch_fusion(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const dim3 & block_nums, const dim3 & block_dims, const int nbytes_shared,
        const uint32_t ids_stride, cudaStream_t stream) {

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr || fusion.x_scale_channel_dst ||
                            fusion.conv_input != nullptr;
    if constexpr (c_ncols_dst == 1) {
        if (has_fusion) {
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
            ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, true, small_k, halve_iters, rows_per_block>, launch_params,
                 vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
            return;
        }
    }

    GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1");

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
    ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, small_k, halve_iters, rows_per_block>, launch_params,
        vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
}

template<ggml_type type, int c_ncols_dst, bool small_k = false, bool halve_iters = false, int rows_per_block = 0, bool long_k = false>
static void mul_mat_vec_q_switch_fusion_ksplit(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const dim3 & block_nums, const dim3 & block_dims, const int nbytes_shared,
        const uint32_t ids_stride, cudaStream_t stream) {

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr || fusion.x_scale_channel_dst ||
                            fusion.conv_input != nullptr;
    if constexpr (c_ncols_dst == 1) {
        if (has_fusion) {
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
            ggml_cuda_kernel_launch(mul_mat_vec_q_ksplit<type, c_ncols_dst, true, small_k, halve_iters, rows_per_block, long_k>, launch_params,
                 vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
            return;
        }
    }

    GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1");

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
    ggml_cuda_kernel_launch(mul_mat_vec_q_ksplit<type, c_ncols_dst, false, small_k, halve_iters, rows_per_block, long_k>, launch_params,
        vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
}

template <ggml_type type>
static void mul_mat_vec_q_moe_launch(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride,
        const int warp_size, const int nchannels_dst, cudaStream_t stream) {

    // Phase 1b/3 UVA cold seam.  The slot-remap consumer redirects the op's `src0` to the cache arena, so
    // `vx` IS the arena base here; look the table up by it and, when it serves cold experts
    // (MOE_EXPERT_CACHE_COLD=uva), pass the pinned host alias + resident slot count + host geometry to the
    // kernel.  The default lookup misses and the kernel gets (nullptr, 0, 0, 0): the resident-only 1a
    // contract, unchanged.  The fused gate lane is a different table/arena, so resolve its cold geometry
    // too (without it the fused kernel would index the compact gate arena with a cold (host) expert id).
    void *  cold_base  = nullptr;
    int64_t n_res_cold = 0;
    size_t  cold_channel_bytes = 0;
    size_t  cold_row_bytes     = 0;
    void *  cold_gate_base     = nullptr;
    int64_t cold_gate_n_res    = 0;
    size_t  cold_gate_channel_bytes = 0;
    size_t  cold_gate_row_bytes     = 0;
    if (moe_cache_enabled()) {
        (void) moe_cache_get_cold(vx, &cold_base, &n_res_cold, &cold_channel_bytes, &cold_row_bytes);
        if (cold_base != nullptr && fusion.gate != nullptr) {
            (void) moe_cache_get_cold(fusion.gate, &cold_gate_base, &cold_gate_n_res,
                                      &cold_gate_channel_bytes, &cold_gate_row_bytes);
        }
    }
    // The kernel indexes in type-block units; the cold geometry is reported in bytes.
    const size_t   ts = ggml_type_size(type);
    const uint32_t cold_channel_stride      = cold_channel_bytes      > 0 ? (uint32_t) (cold_channel_bytes      / ts) : 0;
    const uint32_t cold_row_stride          = cold_row_bytes          > 0 ? (uint32_t) (cold_row_bytes          / ts) : 0;
    const uint32_t cold_gate_channel_stride = cold_gate_channel_bytes > 0 ? (uint32_t) (cold_gate_channel_bytes / ts) : 0;
    const uint32_t cold_gate_row_stride     = cold_gate_row_bytes     > 0 ? (uint32_t) (cold_gate_row_bytes     / ts) : 0;

    // rows_per_block scaled for short-K expert GEMMs: the item-split loop in
    // mul_mat_vec_q_moe spreads (row, kblock) items over the 8 thread groups,
    // so pick rows_per_block to fill them (e.g. the down K=512 -> rpb=4).
    const int blocks_per_row_x = ncols_x / ggml_cuda_type_traits<type>::qk;
    int rpb = 2;
    if (blocks_per_row_x > 0 && blocks_per_row_x < 8) {
        rpb = std::min((8 + blocks_per_row_x - 1) / blocks_per_row_x, 8);
    }
    // mul_mat_vec_q_moe is only instantiated for RPB 2/4/8, and nblocks_rows below must be sized with
    // the value the kernel actually launches with.  ceil(8/blocks_per_row_x) is 3 for
    // blocks_per_row_x == 3 (k == 3*qk), which armed the grid for RPB 3 and then fell through the
    // switch to the RPB 2 default: only the first 2/3 of the rows were computed and the rest of the
    // output was left unwritten (test-backend-ops MUL_MAT_ID, m=64/n=16/k=96, ERR ~0.5).  Snap to a
    // supported value before sizing the grid.
    if (rpb != 4 && rpb != 8) {
        rpb = 2;
    }
    const int64_t nblocks_rows = (nrows_x + rpb - 1) / rpb;
    // Upstream's fused shared-expert lane is the last blockIdx.y column.
    const dim3 block_nums(nblocks_rows, nchannels_dst + (fusion.shared_up != nullptr));
    const dim3 block_dims(warp_size, ncols_dst);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr || fusion.x_scale_channel_dst;

    const auto launch = [&](auto rpb_c, auto fusion_c) {
        constexpr int  RPB   = decltype(rpb_c)::value;
        constexpr bool CFUSE = decltype(fusion_c)::value;
        ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, RPB, CFUSE>, launch_params,
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride, nchannels_dst,
            cold_base, (uint32_t) n_res_cold, cold_channel_stride, cold_row_stride,
            cold_gate_base, (uint32_t) cold_gate_n_res, cold_gate_channel_stride, cold_gate_row_stride);
    };
    const auto launch_rpb = [&](auto rpb_c) {
        if (has_fusion) {
            launch(rpb_c, std::true_type{});
        } else {
            launch(rpb_c, std::false_type{});
        }
    };
    switch (rpb) {
        case 2:  launch_rpb(std::integral_constant<int, 2>{});  break;
        case 4:  launch_rpb(std::integral_constant<int, 4>{});  break;
        case 8:  launch_rpb(std::integral_constant<int, 8>{});  break;
        default: launch_rpb(std::integral_constant<int, 2>{});  break;
    }
}

template <ggml_type type>
static void mul_mat_vec_q_switch_ncols_dst(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream) {

    GGML_ASSERT(ncols_x % ggml_blck_size(type) == 0);
    GGML_ASSERT(ncols_dst <= MMVQ_MAX_BATCH_SIZE);

    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0)              : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst  / nsamples_x);

    const int device = ggml_cuda_get_device();
    const int                     cc        = ggml_cuda_info().devices[device].cc;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const mmvq_parameter_table_id table_id  = get_device_table_id(cc);

    const bool has_ids = ids != nullptr;

    // How the K loop divides up at the baseline block width, both decisions below use these.
    constexpr int qk                    = ggml_cuda_type_traits<type>::qk;
    constexpr int qi                    = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr                   = get_vdr_mmvq(type);
    const int     blocks_per_row_x      = ncols_x / qk;
    const int     blocks_per_iter_1warp = vdr * warp_size / qi;

    const auto should_use_small_k = [&](int c_ncols_dst) {
        // When K is small, increase rows_per_block to match nwarps so each warp has more work to do
        // Trigger when the full thread block covers all K blocks in a single loop iteration and few threads remain idle.
        const int  nwarps = calc_nwarps(type, c_ncols_dst, table_id);
        bool       use    = nwarps > 1 && blocks_per_row_x < nwarps * blocks_per_iter_1warp;

        constexpr std::array<ggml_type, 2> iq_slow_turing = {
            GGML_TYPE_IQ3_XXS,
            GGML_TYPE_IQ3_S,
        };
        constexpr std::array<ggml_type, 8> iq_slow_other = {
            GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,   GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS,
            GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,   GGML_TYPE_IQ4_XS,
        };
        constexpr std::array<ggml_type, 3> slow_pascal = {
            GGML_TYPE_IQ3_S,
            GGML_TYPE_Q2_K,
            GGML_TYPE_Q3_K,
        };

        const bool is_nvidia_turing_plus  = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING;
        const bool is_nvidia_pascal_older = GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_VOLTA;

        if (is_nvidia_turing_plus) {
            if (ncols_dst == 1 &&
                    std::find(iq_slow_turing.begin(), iq_slow_turing.end(), type) != iq_slow_turing.end()) {
                use = false;
            }
        } else if ((ncols_dst == 1 && std::find(iq_slow_other.begin(), iq_slow_other.end(), type) != iq_slow_other.end()) ||
                (is_nvidia_pascal_older && std::find(slow_pascal.begin(), slow_pascal.end(), type) != slow_pascal.end()) ||
                GGML_CUDA_CC_IS_RDNA(cc)) {
            use = false;
        }

        return use;
    };

    // Whether doubling nwarps pays off on the ncols_dst == 1 path, where K sets the K loop trip count.
    const auto should_halve_iters = [&] {
        if (table_id != MMVQ_PARAMETERS_GB10) {
            return false;
        }

        // Expert rows are gathered per token, so a wider block adds reduction work without reuse.
        if (has_ids) {
            return false;
        }

        const int blocks_per_iter = calc_nwarps(type, 1, table_id) * blocks_per_iter_1warp;
        const int iters           = (blocks_per_row_x + blocks_per_iter - 1) /  blocks_per_iter;
        const int iters_wide      = (blocks_per_row_x + blocks_per_iter * 2 - 1) / (blocks_per_iter * 2);

        // An odd trip count leaves half the wider block idle for its last iteration, that tail is
        // only affordable once the loop is long enough to dilute it to an eighth of the work (observation).
        const int idle = iters_wide * 2 - iters;

        return idle * 8 <= iters_wide * 2;
    };

    if (has_ids) {
        // MUL_MAT_ID path - dedicated MoE kernel.  This must cover ncols_dst == 1
        // as well: the dense ksplit path below computes the same rows with a
        // different accumulation order, so a 1-token decode and a multi-token
        // verify batch of the same MoE matmul would not be bit-identical (the
        // decode == verify invariant).  The kernel is column-generic -- each
        // block column is one token (`if (token_idx >= ncols_dst) return;`,
        // `ids[channel_dst + token_idx*ids_stride]`) -- so ncols_dst == 1 is a
        // plain single-column launch.
        mul_mat_vec_q_moe_launch<type>(
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride, warp_size, nchannels_dst, stream);
        return;
    }

    switch (ncols_dst) {
        case 1: case 2: case 3: case 4: case 5: case 6: case 7: case 8: {
            // dispatch per ncols with a compile-time tag
            const auto dispatch = [&](auto ncols_tag) {
                constexpr int c_ncols_dst = decltype(ncols_tag)::value;

                // Tag types keep the flags compile-time, so __launch_bounds__ matches what is launched.
                const auto launch = [&](auto small_k_tag, auto halve_iters_tag, auto rpb_tag) {
                    constexpr bool c_small_k = decltype(small_k_tag)::value;
                    // Types the table does not promote would compile a second, identical kernel.
                    constexpr bool c_promoted =
                        calc_nwarps(type, c_ncols_dst, MMVQ_PARAMETERS_GB10, false, true) !=
                        calc_nwarps(type, c_ncols_dst, MMVQ_PARAMETERS_GB10, false, false);

                    constexpr bool c_halve_iters = decltype(halve_iters_tag)::value && c_promoted;
                    constexpr int  c_rpb         = decltype(rpb_tag)::value;

                    // rows_per_block > 0 (the short-K RDNA override) computes the grid
                    // from rpb directly; 0 keeps the table config via calc_launch_params.
                    std::pair<dim3, dim3> dims;
                    if constexpr (c_rpb > 0) {
                        dims = { dim3((nrows_x + c_rpb - 1) / c_rpb, nchannels_dst + (fusion.shared_up != nullptr), nsamples_dst),
                                 dim3(warp_size, calc_nwarps(type, c_ncols_dst, table_id), 1) };
                    } else {
                        dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst + (fusion.shared_up != nullptr),
                                                        nsamples_dst, warp_size, table_id, c_small_k, c_halve_iters);
                    }
                    mul_mat_vec_q_switch_fusion<type, c_ncols_dst, c_small_k, c_halve_iters, c_rpb>(
                        vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                        channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                        stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, 0, ids_stride,
                        stream);
                };

                // rows_per_block override for short-K GEMMs: the item-split loop in
                // mul_mat_vec_q spreads (row, kblock) items over the thread groups, so
                // pick rows_per_block to fill them on RDNA (the table returns rpb=1
                // there). The MoE down (K=512) and router (K=2048) previously left
                // ~3/4 of the issue slots idle on RDNA4.
                //
                // The j (token) offset of the item index is j*rpb*blocks; choose rpb
                // so that rpb*blocks is a multiple of the group count. That kills the
                // j-dependence and keeps the per-row chunk->group mapping identical
                // for every ncols_dst - the speculative verify batch (ncols 2..8)
                // must stay per-row bit-identical to single-token decode
                // (GREEDY-PURITY.md section 6, invariant 2).
                if (table_id >= MMVQ_PARAMETERS_RDNA2) {
                    const int blocks_per_row_x = ncols_x / ggml_cuda_type_traits<type>::qk;
                    const int qi_type = ggml_cuda_type_traits<type>::qi;
                    const int n_groups = qi_type > 0
                        ? calc_nwarps(type, c_ncols_dst, table_id) * warp_size * get_vdr_mmvq(type) / qi_type
                        : 0;
                    int rpb = 1;
                    if (blocks_per_row_x > 0 && n_groups > 0 && blocks_per_row_x < n_groups) {
                        // fill the groups (ceil) and kill the j-dependence (n_groups/gcd)
                        int fill = (n_groups + blocks_per_row_x - 1) / blocks_per_row_x;
                        int a = blocks_per_row_x, b = n_groups;
                        while (b) {
                            int t = a % b; a = b; b = t;
                        }
                        int inv = n_groups / a;
                        rpb = std::max(fill, inv);
                        int p = 1;
                        while (p < rpb) {
                            p <<= 1;
                        }
                        rpb = std::min(p, 16);
                    }
                    switch (rpb) {
                        case 1:  launch(std::false_type{}, std::false_type{}, std::integral_constant<int, 1>{});  break;
                        case 2:  launch(std::false_type{}, std::false_type{}, std::integral_constant<int, 2>{});  break;
                        case 4:  launch(std::false_type{}, std::false_type{}, std::integral_constant<int, 4>{});  break;
                        case 8:  launch(std::false_type{}, std::false_type{}, std::integral_constant<int, 8>{});  break;
                        case 16: launch(std::false_type{}, std::false_type{}, std::integral_constant<int, 16>{}); break;
                        default: GGML_ABORT("unreachable"); break;
                    }
                } else if (should_use_small_k(c_ncols_dst)) {
                    launch(std::true_type{},  std::false_type{}, std::integral_constant<int, 0>{});
                } else if (should_halve_iters()) {
                    launch(std::false_type{}, std::true_type{}, std::integral_constant<int, 0>{});
                } else {
                    launch(std::false_type{}, std::false_type{}, std::integral_constant<int, 0>{});
                }
            };

            // Dense rows always use the ksplit kernel.  Its nwarps is chosen per
            // (type, K): a Q8_0 weight with K < 4096 (the MoE attention qkv/gate and
            // the lm_head) takes the wider block, K >= 4096 (the dense hybrid
            // attn_v/ssm_out) stays at 1 -- band-uniform within each shape.
            const bool long_k = ncols_x >= 4096;
            const auto launch_ksplit_rpb = [&](auto ncols_tag, auto rpb_tag, auto long_k_tag) {
                constexpr int  c_ncols_dst = decltype(ncols_tag)::value;
                constexpr int  c_rpb       = decltype(rpb_tag)::value;
                constexpr bool c_long_k    = decltype(long_k_tag)::value;
                std::pair<dim3, dim3> dims = calc_launch_params<type, true>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id, false, false, c_long_k);
                if constexpr (c_rpb > 0) {
                    dims.first.x = nrows_x / c_rpb;
                }
                mul_mat_vec_q_switch_fusion_ksplit<type, c_ncols_dst, false, false, c_rpb, c_long_k>(
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                    sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                    dims.first, dims.second, 0, ids_stride, stream);
            };
            const auto launch_ksplit = [&](auto ncols_tag) {
                constexpr int c_ncols_dst = decltype(ncols_tag)::value;
                // A multi-row block reads every row it owns, so a row count that is not a
                // multiple of the block height would read past the end of the weight; those
                // shapes keep one row per block, as do weights too short to fill the GPU.
                // (Only instantiated where a multi-row block exists: HIP builds, 2..8 columns.)
#if defined(GGML_USE_HIP)
                constexpr int c_rpb_weight = calc_rows_per_block_weight(type, c_ncols_dst, MMVQ_PARAMETERS_RDNA4, false, 1);
                if constexpr (c_rpb_weight > 1) {
                    if (table_id == MMVQ_PARAMETERS_RDNA4 &&
                        (nrows_x % c_rpb_weight != 0 || nrows_x < c_rpb_weight * GGML_MMVQ_RDNA4_WEIGHT_MIN_BLOCKS)) {
                        if (long_k) {
                            launch_ksplit_rpb(ncols_tag, std::integral_constant<int, 1>{}, std::true_type{});
                        } else {
                            launch_ksplit_rpb(ncols_tag, std::integral_constant<int, 1>{}, std::false_type{});
                        }
                        return;
                    }
                }
#endif // defined(GGML_USE_HIP)
                if (long_k) {
                    launch_ksplit_rpb(ncols_tag, std::integral_constant<int, 0>{}, std::true_type{});
                } else {
                    launch_ksplit_rpb(ncols_tag, std::integral_constant<int, 0>{}, std::false_type{});
                }
            };

            switch (ncols_dst) {
                case 1: {
                    // Single-token decode. Block 13's item-split kernel + rpb
                    // override wins on short-K MoE rows (down/router, K < 4096),
                    // but its rpb=16 register fan-out loses on long-K dense
                    // projections (qkv/FFN K >= 4096); those use the ksplit kernel
                    // (pre-block-13 geometry) instead.
                    //
                    // Dense (non-MUL_MAT_ID) rows are ALWAYS ksplit, for every K:
                    // the multi-token dispatch below (ncols 2..8) uses ksplit
                    // unconditionally, so a dense ncols==1 row that stayed on the
                    // item-split kernel accumulated K in a different order than the
                    // verify batch and stopped being bit-identical row by row.  A
                    // ~1e-6 per-row difference is amplified by the recurrent GDN
                    // into greedy flips between --spec-type none and draft-mtp
                    // (and under --split-mode tensor, where the per-GPU K shard
                    // drops below 4096).  MoE rows keep the item-split: their
                    // multi-token path is the dedicated mul_mat_vec_q_moe kernel.
                    if (table_id >= MMVQ_PARAMETERS_RDNA2) {
                        if (!has_ids || ncols_x >= 4096) {
                            launch_ksplit(std::integral_constant<int, 1>{});
                            break;
                        }
                    }
                    dispatch(std::integral_constant<int, 1>{});
                } break;
        case 2: {
            // multi-token decode (ncols_dst = 2): MTP/NextN verify batches and
            // batched dense decode. Uses the ksplit kernel (see above): block 13's
            // item-split accumulator layout is register-bound at ncols_dst > 1.
            launch_ksplit(std::integral_constant<int, 2>{});
        } break;
        case 3: {
            // multi-token decode (ncols_dst = 3): MTP/NextN verify batches and
            // batched dense decode. Uses the ksplit kernel (see above): block 13's
            // item-split accumulator layout is register-bound at ncols_dst > 1.
            launch_ksplit(std::integral_constant<int, 3>{});
        } break;
        case 4: {
            // multi-token decode (ncols_dst = 4): MTP/NextN verify batches and
            // batched dense decode. Uses the ksplit kernel (see above): block 13's
            // item-split accumulator layout is register-bound at ncols_dst > 1.
            launch_ksplit(std::integral_constant<int, 4>{});
        } break;
        case 5: {
            // multi-token decode (ncols_dst = 5): MTP/NextN verify batches and
            // batched dense decode. Uses the ksplit kernel (see above): block 13's
            // item-split accumulator layout is register-bound at ncols_dst > 1.
            launch_ksplit(std::integral_constant<int, 5>{});
        } break;
        case 6: {
            // multi-token decode (ncols_dst = 6): MTP/NextN verify batches and
            // batched dense decode. Uses the ksplit kernel (see above): block 13's
            // item-split accumulator layout is register-bound at ncols_dst > 1.
            launch_ksplit(std::integral_constant<int, 6>{});
        } break;
        case 7: {
            // multi-token decode (ncols_dst = 7): MTP/NextN verify batches and
            // batched dense decode. Uses the ksplit kernel (see above): block 13's
            // item-split accumulator layout is register-bound at ncols_dst > 1.
            launch_ksplit(std::integral_constant<int, 7>{});
        } break;
        case 8: {
            // multi-token decode (ncols_dst = 8): MTP/NextN verify batches and
            // batched dense decode. Uses the ksplit kernel (see above): block 13's
            // item-split accumulator layout is register-bound at ncols_dst > 1.
            launch_ksplit(std::integral_constant<int, 8>{});
        } break;
                default: GGML_ABORT("fatal error"); break;
            }
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}
static void mul_mat_vec_q_switch_type(
        const void * vx, const ggml_type type_x, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream) {
    switch (type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q1_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q2_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q8_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_MXFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_NVFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q3_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q6_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ1_M:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_M>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_NL>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void ggml_cuda_mul_mat_vec_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    GGML_ASSERT(!ids || ne12 <= MMVQ_MAX_BATCH_SIZE);

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    ggml_cuda_mm_fusion_args_device fusion_local{};

    if (fusion) {
        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        GGML_ASSERT( !ids || dst->ne[2] <= get_mmvq_mmid_max_batch(src0->type, cc));
        GGML_ASSERT(  ids || dst->ne[1] == 1);
        // Scale fusion is only allowed for NVFP4 currently as the cost of checking this at run-time in the prologue is
        // non-negligible for some models such as gpt-oss-20b. The per-token MoE scale is a second exception.
        GGML_ASSERT((fusion->x_scale == nullptr && fusion->gate_scale == nullptr) || src0->type == GGML_TYPE_NVFP4 || fusion->x_scale_channel_dst);

        if (fusion->shared_up) {
            GGML_ASSERT(ids && fusion->gate && fusion->shared_gate && fusion->shared_dst);
            GGML_ASSERT(!fusion->x_bias && !fusion->gate_bias && !fusion->x_scale && !fusion->gate_scale);
            GGML_ASSERT(ne11 == 1 && ne03 == 1 && ne13 == 1);
            GGML_ASSERT(fusion->shared_up->type == src0->type && fusion->shared_gate->type == src0->type);
            GGML_ASSERT(ggml_are_same_shape(fusion->shared_up, fusion->shared_gate));
            GGML_ASSERT(ggml_is_contiguous(fusion->shared_up) && ggml_is_contiguous(fusion->shared_gate));
            GGML_ASSERT(fusion->shared_up->ne[0] == ne00 && fusion->shared_up->ne[1] == ne01);
            GGML_ASSERT(fusion->shared_up->nb[1] == nb01 && ggml_is_matrix(fusion->shared_up));
            GGML_ASSERT(fusion->shared_dst->type == GGML_TYPE_F32 && ggml_is_contiguous(fusion->shared_dst));
            GGML_ASSERT(fusion->shared_dst->ne[0] == ne0 && fusion->shared_dst->ne[1] == ne2);
            fusion_local.shared_up   = fusion->shared_up->data;
            fusion_local.shared_gate = fusion->shared_gate->data;
            fusion_local.shared_dst  = (float *) fusion->shared_dst->data;
            fusion_local.shared_stride_col_dst = fusion->shared_dst->nb[1] / ts_dst;
        }

        if (fusion->x_bias) {
            GGML_ASSERT(fusion->x_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->x_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->x_bias->ne[1] == src0->ne[2]);
            fusion_local.x_bias = fusion->x_bias->data;
        }
        if (fusion->gate) {
            GGML_ASSERT(fusion->gate->type == src0->type && ggml_are_same_stride(fusion->gate, src0));
            fusion_local.gate = fusion->gate->data;
        }
        if (fusion->gate_bias) {
            GGML_ASSERT(fusion->gate_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->gate_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->gate_bias->ne[1] == src0->ne[2]);
            fusion_local.gate_bias = fusion->gate_bias->data;
        }
        if (fusion->x_scale) {
            GGML_ASSERT(fusion->x_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->x_scale));
            if (fusion->x_scale_channel_dst) {
                GGML_ASSERT(ids);
                GGML_ASSERT(ggml_nelements(fusion->x_scale) == dst->ne[1] * dst->ne[2]);
            } else {
                GGML_ASSERT(ggml_nelements(fusion->x_scale) == (ids ? src0->ne[2] : 1));
            }
            fusion_local.x_scale = fusion->x_scale->data;
        }
        fusion_local.x_scale_channel_dst = fusion->x_scale_channel_dst;
        if (fusion->conv_input) {
            GGML_ASSERT(fusion->conv_input->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->conv_states && fusion->conv_states->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->conv_kernel_size >= 2);
            fusion_local.conv_input       = fusion->conv_input->data;
            fusion_local.conv_states      = fusion->conv_states->data;
            fusion_local.conv_kernel_size = fusion->conv_kernel_size;
        }
        if (fusion->gate_scale) {
            GGML_ASSERT(fusion->gate_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->gate_scale));
            GGML_ASSERT(ggml_nelements(fusion->gate_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.gate_scale = fusion->gate_scale->data;
        }
        fusion_local.glu_op = fusion->glu_op;
        fusion_local.glu_limit = fusion->glu_limit;
        if (fusion->dst_gate) {
            GGML_ASSERT(fusion->dst_gate->type == GGML_TYPE_F32);
            fusion_local.dst_gate = fusion->dst_gate->data;
        }
    }

    // If src0 is a temporary compute buffer, clear any potential padding.  A redirected expert
    // table (wip/moe-expert-cache) carries no buffer; its compact arena has no padding to clear,
    // and `ggml_nbytes(src0)` would describe the full expert count, so the memset below would run
    // far past the arena.
    if (src0->buffer != nullptr && ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const size_t src1_q8_1_size = ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1;

    // Quantize src1 to Q8_1 once per graph; matmuls sharing the same src1 data
    // (views of the same tensor included) reuse the buffer.
    const ggml_tensor * src1_key = src1;
    while (src1_key->view_src != nullptr) {
        src1_key = src1_key->view_src;
    }
    const int64_t src1_s11 = src1->nb[1] / ts_src1;
    const int64_t src1_s12 = src1->nb[2] / ts_src1;
    const int64_t src1_s13 = src1->nb[3] / ts_src1;
    bool src1_q8_1_cached = false;
    void * src1_q8_1 = ctx.q8_1_cache_get(src1_key, ctx.curr_stream_no, src1_q8_1_size,
                                          ne10, ne11, ne12, ne13, src1_s11, src1_s12, src1_s13, src1_q8_1_cached);
    if (!src1_q8_1_cached) {
        quantize_row_q8_1_cuda(src1_d, nullptr, src1_q8_1, src0->type, ne10, src1_s11, src1_s12, src1_s13, ne10_padded, ne11, ne12, ne13, stream);
    }

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = ne10_padded / QK8_1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const int64_t s12 = ne11*s11;
    const int64_t s13 = ne12*s12;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_col_y       = ids ? s12  : s11;
    const int64_t stride_channel_dst = ids ? s1   : s2;
    const int64_t stride_channel_y   = ids ? s11  : s12;

    const int64_t ids_stride = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;

    mul_mat_vec_q_switch_type(
        src0->data, src0->type, src1_q8_1, ids_d, fusion_local, dst_d, ne00,
        ne01,              ncols_dst,     s01, stride_col_y,     stride_col_dst,
        ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
        ne03,              ne3,           s03, s13,              s3,               ids_stride, stream);
}

static __device__ __forceinline__ float ssm_op_softplus(float x) {
    return (x > 20.0f) ? x : logf(1.0f + expf(x));
}

static __device__ __forceinline__ float ssm_op_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Fused SSM gate/beta projections: two Q8_0 matmuls over the same input, plus
// the gating chain. gate = softplus(alpha*x + dt) * a; beta = sigmoid(beta*x).
// Grid: (nrows, 1), block: (warp_size, nwarps), nwarps per row. Each workgroup
// computes one row of both outputs.
template <int nwarps>
static __global__ void ssm_gate_beta_fused_q8_0(
        const void * vx_alpha, const void * vx_beta, const block_q8_1 * y,
        const float * dt, const float * ssm_a,
        float * dst_gate, float * dst_beta,
        const int nrows, const int ncols_x) {
    const int row = blockIdx.x;
    constexpr int warp_size = 32;
    const int tid = warp_size*threadIdx.y + threadIdx.x;
    constexpr int qk = 32; // Q8_0 block size
    constexpr int qi = QI8_0; // int32 per Q8_0 block
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    const int blocks_per_row_x = ncols_x / qk;

    const block_q8_0 * w_alpha = (const block_q8_0 *) vx_alpha + (int64_t) row * blocks_per_row_x;
    const block_q8_0 * w_beta  = (const block_q8_0 *) vx_beta  + (int64_t) row * blocks_per_row_x;

    float sum_alpha = 0.0f;
    float sum_beta  = 0.0f;
    const int blocks_per_iter = vdr * nwarps*warp_size / qi;
    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx; // qk == QK8_1
        const int kqs = vdr * (tid % (qi/vdr));
        sum_alpha += vec_dot_q8_0_q8_1(w_alpha, &y[kby], kbx, kqs);
        sum_beta  += vec_dot_q8_0_q8_1(w_beta,  &y[kby], kbx, kqs);
    }

    // cross-warp reduction, same order as mul_mat_vec_q
    __shared__ float sh_alpha[nwarps > 1 ? nwarps-1 : 1][warp_size];
    __shared__ float sh_beta[nwarps > 1 ? nwarps-1 : 1][warp_size];
    if (threadIdx.y > 0) {
        sh_alpha[threadIdx.y-1][threadIdx.x] = sum_alpha;
        sh_beta[threadIdx.y-1][threadIdx.x] = sum_beta;
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }
    for (int l = 0; l < nwarps-1; ++l) {
        sum_alpha += sh_alpha[l][threadIdx.x];
        sum_beta  += sh_beta[l][threadIdx.x];
    }
    sum_alpha = warp_reduce_sum<warp_size>(sum_alpha);
    sum_beta  = warp_reduce_sum<warp_size>(sum_beta);

    if (threadIdx.x == 0) {
        dst_gate[row] = ssm_op_softplus(sum_alpha + dt[row]) * ssm_a[row];
        dst_beta[row] = ssm_op_sigmoid(sum_beta);
    }

    GGML_UNUSED(nrows);
}

void ggml_cuda_op_ssm_gate_beta(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0_alpha, const ggml_tensor * src0_beta, const ggml_tensor * src1,
        const ggml_tensor * dt, const ggml_tensor * ssm_a,
        ggml_tensor * dst_gate, ggml_tensor * dst_beta) {
    GGML_ASSERT(src0_alpha->type == GGML_TYPE_Q8_0 && src0_beta->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst_gate->type == GGML_TYPE_F32 && dst_beta->type == GGML_TYPE_F32);
    GGML_ASSERT(src0_alpha->ne[0] == src0_beta->ne[0] && src0_alpha->ne[1] == src0_beta->ne[1]);
    GGML_ASSERT(ggml_nelements(dst_gate) == ggml_nelements(dst_beta));

    cudaStream_t stream = ctx.stream();
    const size_t ts_src1 = ggml_type_size(src1->type);

    // Quantize src1 to Q8_1 once per graph; reuse the same buffer as the other
    // matmuls over this input.
    const int64_t ne10_padded = GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING);
    const size_t src1_q8_1_size = src1->ne[3]*src1->ne[2] * src1->ne[1]*ne10_padded * sizeof(block_q8_1)/QK8_1;
    const ggml_tensor * src1_key = src1;
    while (src1_key->view_src != nullptr) {
        src1_key = src1_key->view_src;
    }
    const int64_t src1_s11 = src1->nb[1] / ts_src1;
    const int64_t src1_s12 = src1->nb[2] / ts_src1;
    const int64_t src1_s13 = src1->nb[3] / ts_src1;
    bool src1_q8_1_cached = false;
    void * src1_q8_1 = ctx.q8_1_cache_get(src1_key, ctx.curr_stream_no, src1_q8_1_size,
                                          src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                                          src1_s11, src1_s12, src1_s13, src1_q8_1_cached);
    if (!src1_q8_1_cached) {
        quantize_row_q8_1_cuda((const float *) src1->data, nullptr, src1_q8_1, GGML_TYPE_Q8_0,
                               src1->ne[0], src1_s11, src1_s12, src1_s13, ne10_padded,
                               src1->ne[1], src1->ne[2], src1->ne[3], stream);
    }

    const int nrows = dst_gate->ne[0];
    const int device = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[device].cc;
    // This fusion is DECODE-ONLY (the matcher requires alpha_w->src[1]->ne[1] == 1), so it must
    // reproduce the standalone dense mmvq *weight* launch it replaces.  calc_nwarps_weight() is that
    // launch's selector (Q8_0 with K < 4096 takes the wide block on RDNA4); pinning plain
    // calc_nwarps() here reduced K with a different warp count than the unfused chain the verify
    // batch runs, so W = 1 and W >= 2 disagreed by ULPs and a near-tie flipped after a few hundred
    // tokens -- a width-impurity that confounds every plain-vs-draft-mtp coherence check.  It is a
    // no-op for K >= 4096 (long_k), so the long-K models keep the pinned single-token order.
    const bool long_k = src1->ne[0] >= 4096;
    const int nwarps = calc_nwarps_weight(GGML_TYPE_Q8_0, 1, get_device_table_id(cc), long_k);
    const dim3 block_nums(nrows);
    const dim3 block_dims(32, nwarps);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
    const auto launch = [&](auto nwarps_c) {
        constexpr int NW = decltype(nwarps_c)::value;
        ggml_cuda_kernel_launch(ssm_gate_beta_fused_q8_0<NW>, launch_params,
            src0_alpha->data, src0_beta->data, (const block_q8_1 *) src1_q8_1, (const float *) dt->data, (const float *) ssm_a->data,
            (float *) dst_gate->data, (float *) dst_beta->data, nrows, src1->ne[0]);
    };
    switch (nwarps) {
        case 1:  launch(std::integral_constant<int, 1>{}); break;
        case 2:  launch(std::integral_constant<int, 2>{}); break;
        case 4:  launch(std::integral_constant<int, 4>{}); break;
        case 8:  launch(std::integral_constant<int, 8>{}); break;
        default: GGML_ASSERT(false); break;
    }

}

// Fused SSM pre-scan chain: conv + silu, l2_norm(q, k) and the gate/beta
// projections in one kernel. Replaces ssm_conv (+silu), l2_norm_pair and
// ssm_gate_beta. Grid: (n_qk_heads + n_v_heads + ceil(n_v_heads/4), 1),
// block: (128, 1).
//   [0, n_qk_heads): q head b (threads 0..31) + k head b (threads 32..63);
//                    threads 0..63 each handle 4 channels, accumulating the L2
//                    sum in the same order as l2_norm_pair (cols tid, tid+32,
//                    tid+64, tid+96). Threads 64..127 idle.
//   [n_qk_heads, +n_v_heads): v head, one channel per thread.
//   remainder: gate/beta, one head per warp.
template <int nwarps>
static __global__ void ssm_conv_l2_gatebeta_fused(
        const float * conv_input, const float * conv_w,
        const block_q8_0 * w_alpha, const block_q8_0 * w_beta, const block_q8_1 * y,
        const float * dt, const float * ssm_a,
        float * q_norm, float * k_norm, float * v_raw,
        float * dst_gate, float * dst_beta,
        const int d_conv, const int head_k_dim, const int n_qk_heads,
        const int head_v_dim, const int n_v_heads,
        const float eps, const int ncols_x) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int n_qk_ch = head_k_dim * n_qk_heads;

    if (bidx >= n_qk_heads + n_v_heads) {
        // gate/beta: nwarps per head, heads_per_block heads per block. The
        // per-lane loop and the cross-warp reduction replicate mul_mat_vec_q
        // (nwarps) so the sums are bit-identical to the unfused mmvq path used
        // by verify batches.
        constexpr int warp_size = 32;
        constexpr int threads_per_head = nwarps * warp_size;
        constexpr int heads_per_block = (128 / threads_per_head) > 0 ? 128 / threads_per_head : 1;
        const int head = (bidx - n_qk_heads - n_v_heads) * heads_per_block + tid / threads_per_head;
        const bool valid = head < n_v_heads;
        const int tid_in_head = tid % threads_per_head;
        const int lane = tid_in_head % 32;
        const int warp_in_head = tid_in_head / 32;
        constexpr int qk = 32;
        constexpr int qi = QI8_0;
        constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
        const int blocks_per_row_x = ncols_x / qk;
        const block_q8_0 * w_a = w_alpha + (int64_t) head * blocks_per_row_x;
        const block_q8_0 * w_b = w_beta  + (int64_t) head * blocks_per_row_x;
        float sum_alpha = 0.0f;
        float sum_beta  = 0.0f;
        const int blocks_per_iter = vdr * nwarps * warp_size / qi;
        if (valid) {
            for (int kbx = tid_in_head / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
                const int kby = kbx; // qk == QK8_1
                const int kqs = vdr * (tid_in_head % (qi/vdr));
                sum_alpha += vec_dot_q8_0_q8_1(w_a, &y[kby], kbx, kqs);
                sum_beta  += vec_dot_q8_0_q8_1(w_b, &y[kby], kbx, kqs);
            }
        }
        __shared__ float sh_alpha[heads_per_block][nwarps > 1 ? nwarps-1 : 1][warp_size];
        __shared__ float sh_beta[heads_per_block][nwarps > 1 ? nwarps-1 : 1][warp_size];
        if (valid && warp_in_head > 0) {
            sh_alpha[head % heads_per_block][warp_in_head-1][lane] = sum_alpha;
            sh_beta[head % heads_per_block][warp_in_head-1][lane] = sum_beta;
        }
        __syncthreads();
        if (warp_in_head > 0 || !valid) {
            return;
        }
        for (int l = 0; l < nwarps-1; ++l) {
            sum_alpha += sh_alpha[head % heads_per_block][l][lane];
            sum_beta  += sh_beta[head % heads_per_block][l][lane];
        }
        sum_alpha = warp_reduce_sum<warp_size>(sum_alpha);
        sum_beta  = warp_reduce_sum<warp_size>(sum_beta);
        if (lane == 0) {
            dst_gate[head] = ssm_op_softplus(sum_alpha + dt[head]) * ssm_a[head];
            dst_beta[head] = ssm_op_sigmoid(sum_beta);
        }
        return;
    }

    if (bidx >= n_qk_heads) {
        // v head: one channel per thread
        if (tid >= head_v_dim) {
            return;
        }
        const int c = 2*n_qk_ch + (bidx - n_qk_heads)*head_v_dim + tid;
        float sumf = 0.0f;
#pragma unroll
        for (int j = 0; j < d_conv; ++j) {
            sumf += conv_input[c*d_conv + j] * conv_w[c*d_conv + j];
        }
        v_raw[c] = ggml_cuda_op_silu_single(sumf);
        return;
    }

    // q head (warp 0) / k head (warp 1): conv + silu, 4 channels per thread
    if (tid >= 64) {
        return;
    }
    const bool is_k = tid >= 32;
    const int  lane = tid % 32;
    const int  base = is_k ? n_qk_ch : 0;
    float *    dst_qk = is_k ? k_norm : q_norm;
    float      xv[4];
    float      tmp = 0.0f;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int col = lane + 32*r;
        const int c   = base + bidx*head_k_dim + col;
        // same arithmetic as ssm_conv_f32: load into registers, then
        // accumulate the weighted sum (identical rounding)
        float sumf = 0.0f;
        float x4[15] = { 0.0f };
        float w4[15] = { 0.0f };
#pragma unroll
        for (int j = 0; j < d_conv; ++j) {
            x4[j] = conv_input[c*d_conv + j];
            w4[j] = conv_w[c*d_conv + j];
        }
#pragma unroll
        for (int j = 0; j < d_conv; ++j) {
            sumf += x4[j] * w4[j];
        }
        xv[r] = ggml_cuda_op_silu_single(sumf);
        tmp  = __fmaf_rn(xv[r], xv[r], tmp);
    }
    tmp = warp_reduce_sum<32>(tmp);
    const float scale = rsqrtf(fmaxf(tmp, eps * eps));
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        dst_qk[bidx*head_k_dim + lane + 32*r] = scale * xv[r];
    }
}

void ggml_cuda_op_ssm_conv_l2_gatebeta(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * conv_input, const ggml_tensor * conv_w,
        const ggml_tensor * src0_alpha, const ggml_tensor * src0_beta, const ggml_tensor * src1,
        const ggml_tensor * dt, const ggml_tensor * ssm_a,
        ggml_tensor * q_norm, ggml_tensor * k_norm, ggml_tensor * v_raw,
        ggml_tensor * dst_gate, ggml_tensor * dst_beta,
        const int head_k_dim, const int n_qk_heads,
        const int head_v_dim, const int n_v_heads, const float eps) {
    GGML_ASSERT(conv_input->type == GGML_TYPE_F32 && conv_w->type == GGML_TYPE_F32);
    GGML_ASSERT(src0_alpha->type == GGML_TYPE_Q8_0 && src0_beta->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && q_norm->type == GGML_TYPE_F32);
    GGML_ASSERT(k_norm->type == GGML_TYPE_F32 && v_raw->type == GGML_TYPE_F32);
    GGML_ASSERT(dst_gate->type == GGML_TYPE_F32 && dst_beta->type == GGML_TYPE_F32);
    GGML_ASSERT(src0_alpha->ne[0] == src0_beta->ne[0] && src0_alpha->ne[1] == src0_beta->ne[1]);
    GGML_ASSERT(ggml_nelements(dst_gate) == ggml_nelements(dst_beta));

    cudaStream_t stream = ctx.stream();
    const size_t ts_src1 = ggml_type_size(src1->type);

    // Quantize src1 to Q8_1 once per graph; reuse the same buffer as the other
    // matmuls over this input.
    const int64_t ne10_padded = GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING);
    const size_t src1_q8_1_size = src1->ne[3]*src1->ne[2] * src1->ne[1]*ne10_padded * sizeof(block_q8_1)/QK8_1;
    const ggml_tensor * src1_key = src1;
    while (src1_key->view_src != nullptr) {
        src1_key = src1_key->view_src;
    }
    const int64_t src1_s11 = src1->nb[1] / ts_src1;
    const int64_t src1_s12 = src1->nb[2] / ts_src1;
    const int64_t src1_s13 = src1->nb[3] / ts_src1;
    bool src1_q8_1_cached = false;
    void * src1_q8_1 = ctx.q8_1_cache_get(src1_key, ctx.curr_stream_no, src1_q8_1_size,
                                          src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                                          src1_s11, src1_s12, src1_s13, src1_q8_1_cached);
    if (!src1_q8_1_cached) {
        quantize_row_q8_1_cuda((const float *) src1->data, nullptr, src1_q8_1, GGML_TYPE_Q8_0,
                               src1->ne[0], src1_s11, src1_s12, src1_s13, ne10_padded,
                               src1->ne[1], src1->ne[2], src1->ne[3], stream);
    }

    const int d_conv  = conv_w->ne[0];
    const int ncols_x = src1->ne[0];

    const int device = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[device].cc;
    const int nwarps = calc_nwarps(GGML_TYPE_Q8_0, 1, get_device_table_id(cc));
    const int threads_per_head = nwarps * 32;
    const int heads_per_block = (128 / threads_per_head) > 0 ? 128 / threads_per_head : 1;
    const dim3 block_nums(n_qk_heads + n_v_heads + (n_v_heads + heads_per_block - 1) / heads_per_block);
    const dim3 block_dims(heads_per_block * threads_per_head);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
    const auto launch = [&](auto nwarps_c) {
        constexpr int NW = decltype(nwarps_c)::value;
        ggml_cuda_kernel_launch(ssm_conv_l2_gatebeta_fused<NW>, launch_params,
            (const float *) conv_input->data, (const float *) conv_w->data,
            (const block_q8_0 *) src0_alpha->data, (const block_q8_0 *) src0_beta->data, (const block_q8_1 *) src1_q8_1,
            (const float *) dt->data, (const float *) ssm_a->data,
            (float *) q_norm->data, (float *) k_norm->data, (float *) v_raw->data,
            (float *) dst_gate->data, (float *) dst_beta->data,
            d_conv, head_k_dim, n_qk_heads, head_v_dim, n_v_heads, eps, ncols_x);
    };
    switch (nwarps) {
        case 1:  launch(std::integral_constant<int, 1>{}); break;
        case 2:  launch(std::integral_constant<int, 2>{}); break;
        case 4:  launch(std::integral_constant<int, 4>{}); break;
        case 8:  launch(std::integral_constant<int, 8>{}); break;
        default: GGML_ASSERT(false); break;
    }
}


// Fused shared-expert output: down projection plus the gate and residual adds.
// dst = down(swiglu) * sigmoid(gate(x)) + moe_out + ffn_residual.
// Two kernels: the gate is a per-token scalar dot over the shared input, computed by a tiny
// kernel (grid: (ncols), block: (warp_size)); the down projection then reads it (grid:
// (nrows, ncols), block: (warp_size, nwarps), nwarps per row).  Both serve the whole
// decode/verify band and pin their reduction orders to the single-token ones, so every width
// computes each (row, token) bit-identically.
static __global__ void shexp_gate_sigmoid(
        const float * vx_gate, const float * x_gate, float * dst, const int k_gate, const int64_t sx_gate) {
    const int tid = threadIdx.x;
    constexpr int warp_size = 32;
    const int t = blockIdx.x;
    const float * x_gate_t = x_gate + (int64_t) t * sx_gate;

    float sum_gate = 0.0f;
#pragma unroll 8
    for (int i = tid; i < k_gate; i += warp_size) {
        sum_gate += vx_gate[i] * x_gate_t[i];
    }
    sum_gate = warp_reduce_sum<warp_size>(sum_gate);
    if (tid == 0) {
        dst[t] = ssm_op_sigmoid(sum_gate);
    }
}

// One block per output row, the whole decode/verify band in registers: the weight row is read once
// per (row, k-block) and reused for every token, instead of once per (row, token).  The token only
// selects the input/output columns, so each token keeps the single-token kernel's per-thread
// accumulation order and the same cross-warp reduction order -- every (row, token) result is
// bit-identical at every width (the decode == verify invariant, GREEDY-PURITY.md §19).
template <int nwarps, int ncols_dst>
static __global__ void shexp_down_gated_q8_0(
        const void * vx_down, const block_q8_1 * y_swiglu, const float * gate_sig,
        const float * moe_out, const float * ffn_residual,
        float * dst, const int k_down, const int stride_col_y, const int nrows_dst) {
    const int row = blockIdx.x;
    constexpr int warp_size = 32;
    const int tid = warp_size*threadIdx.y + threadIdx.x;
    constexpr int qk = 32; // Q8_0 block size
    constexpr int qi = QI8_0; // int32 per Q8_0 block
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;

    const int blocks_per_row_x = k_down / qk;
    const block_q8_0 * wd = (const block_q8_0 *) vx_down + (int64_t) row * blocks_per_row_x;

    float sum_down[ncols_dst];
#pragma unroll
    for (int t = 0; t < ncols_dst; ++t) {
        sum_down[t] = 0.0f;
    }

    const int blocks_per_iter = vdr * nwarps*warp_size / qi;
    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kqs = vdr * (tid % (qi/vdr));
        // the weight block is read once and reused by every token of the band
#pragma unroll
        for (int t = 0; t < ncols_dst; ++t) {
            sum_down[t] += vec_dot_q8_0_q8_1(wd, &y_swiglu[(int64_t) t*stride_col_y + kbx], kbx, kqs);
        }
    }

    // cross-warp reduction, same order as mul_mat_vec_q (and per token, as the single-token kernel)
    __shared__ float sh_down[nwarps > 1 ? nwarps-1 : 1][ncols_dst][warp_size];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int t = 0; t < ncols_dst; ++t) {
            sh_down[threadIdx.y-1][t][threadIdx.x] = sum_down[t];
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }
#pragma unroll
    for (int t = 0; t < ncols_dst; ++t) {
#pragma unroll
        for (int l = 0; l < nwarps-1; ++l) {
            sum_down[t] += sh_down[l][t][threadIdx.x];
        }
        sum_down[t] = warp_reduce_sum<warp_size>(sum_down[t]);
    }

    if (threadIdx.x == 0) {
        // Bit-exactness with the unfused chain: the graph computes
        //   gated = down * gate_sig   (separate MUL kernel, rounded)
        //   ffn_out = gated + moe_out (separate ADD kernel, rounded)
        //   dst = ffn_out + ffn_residual
        // so the multiply must NOT be contracted into an FMA here (an FMA would
        // round once instead of twice and break decode == verify).
#pragma unroll
        for (int t = 0; t < ncols_dst; ++t) {
            const int64_t o = (int64_t) t * nrows_dst + row;
            dst[o] = __fmul_rn(sum_down[t], gate_sig[t]) + moe_out[o] + ffn_residual[o];
        }
    }
}

void ggml_cuda_op_shexp_down_gate(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * w_down, const ggml_tensor * swiglu,
        const ggml_tensor * w_gate, const ggml_tensor * x_gate,
        const ggml_tensor * moe_out, const ggml_tensor * ffn_residual,
        ggml_tensor * dst) {
    GGML_ASSERT(w_down->type == GGML_TYPE_Q8_0 && w_gate->type == GGML_TYPE_F32);
    GGML_ASSERT(swiglu->type == GGML_TYPE_F32 && x_gate->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    cudaStream_t stream = ctx.stream();
    const size_t ts_src1 = ggml_type_size(swiglu->type);

    // Quantize the swiglu to Q8_1 once per graph.
    const int64_t ne10_padded = GGML_PAD(swiglu->ne[0], MATRIX_ROW_PADDING);
    const size_t q8_1_size = swiglu->ne[3]*swiglu->ne[2] * swiglu->ne[1]*ne10_padded * sizeof(block_q8_1)/QK8_1;    const ggml_tensor * src1_key = swiglu;
    while (src1_key->view_src != nullptr) {
        src1_key = src1_key->view_src;
    }
    const int64_t s11 = swiglu->nb[1] / ts_src1;
    const int64_t s12 = swiglu->nb[2] / ts_src1;
    const int64_t s13 = swiglu->nb[3] / ts_src1;
    bool cached = false;
    void * y_swiglu = ctx.q8_1_cache_get(src1_key, ctx.curr_stream_no, q8_1_size,
                                         swiglu->ne[0], swiglu->ne[1], swiglu->ne[2], swiglu->ne[3],
                                         s11, s12, s13, cached);
    if (!cached) {
        quantize_row_q8_1_cuda((const float *) swiglu->data, nullptr, y_swiglu, GGML_TYPE_Q8_0,
                               swiglu->ne[0], s11, s12, s13, ne10_padded,
                               swiglu->ne[1], swiglu->ne[2], swiglu->ne[3], stream);
    }

    // The gate is a per-token scalar dot over the shared input (ncols = the decode/verify band).
    // Compute it in a tiny kernel, then the down projection reads it.  The scratch is a pool
    // allocation: stream ordering keeps it valid for both launches.
    const int64_t ncols = dst->ne[1];
    GGML_ASSERT(ncols >= 1 && ncols <= MMVQ_MAX_BATCH_SIZE && swiglu->ne[1] == ncols && x_gate->ne[1] == ncols &&
                moe_out->ne[1] == ncols && ffn_residual->ne[1] == ncols);
    GGML_ASSERT(moe_out->nb[1] == moe_out->nb[0]*moe_out->ne[0] &&
                ffn_residual->nb[1] == ffn_residual->nb[0]*ffn_residual->ne[0] &&
                dst->nb[1] == dst->nb[0]*dst->ne[0]);
    ggml_cuda_pool_alloc<float> gate_scratch(ctx.pool(), ncols);
    {
        const dim3 block_nums(ncols);
        const dim3 block_dims(32);
        const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        ggml_cuda_kernel_launch(shexp_gate_sigmoid, lp,
            (const float *) w_gate->data, (const float *) x_gate->data, gate_scratch.get(), w_gate->ne[0],
            x_gate->nb[1] / (int64_t) sizeof(float));
    }

    const int nrows = dst->ne[0];
    const int device = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[device].cc;
    // nwarps is pinned to the single-token value: it sets blocks_per_iter and hence the
    // reduction order of the down projection, so a width-dependent nwarps would make the
    // verify batch arithmetically different from the decode (the decode == verify invariant).
    const int nwarps = calc_nwarps(GGML_TYPE_Q8_0, 1, get_device_table_id(cc));
    const int stride_col_y = (int) (ne10_padded / QK8_1);
    // One block per row: the band is block-internal (ncols_dst), so the weight row is read once for
    // all tokens.  This keeps grid.x at nrows regardless of the batch width (the per-token grid
    // duplicated the whole block -- its two barriers, the reduction and the epilogue -- per token).
    const dim3 block_nums(nrows);
    const dim3 block_dims(32, nwarps);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
    const auto launch = [&](auto nwarps_c, auto ncols_c) {
        constexpr int NW = decltype(nwarps_c)::value;
        constexpr int NC = decltype(ncols_c)::value;
        ggml_cuda_kernel_launch(shexp_down_gated_q8_0<NW, NC>, launch_params,
            w_down->data, (const block_q8_1 *) y_swiglu, gate_scratch.get(),
            (const float *) moe_out->data, (const float *) ffn_residual->data,
            (float *) dst->data, w_down->ne[0], stride_col_y, nrows);
    };
    const auto launch_ncols = [&](auto nwarps_c) {
        switch (ncols) {
            case 1:  launch(nwarps_c, std::integral_constant<int, 1>{}); break;
            case 2:  launch(nwarps_c, std::integral_constant<int, 2>{}); break;
            case 3:  launch(nwarps_c, std::integral_constant<int, 3>{}); break;
            case 4:  launch(nwarps_c, std::integral_constant<int, 4>{}); break;
            case 5:  launch(nwarps_c, std::integral_constant<int, 5>{}); break;
            case 6:  launch(nwarps_c, std::integral_constant<int, 6>{}); break;
            case 7:  launch(nwarps_c, std::integral_constant<int, 7>{}); break;
            case 8:  launch(nwarps_c, std::integral_constant<int, 8>{}); break;
            default: GGML_ASSERT(false); break;
        }
    };
    switch (nwarps) {
        case 1:  launch_ncols(std::integral_constant<int, 1>{}); break;
        case 2:  launch_ncols(std::integral_constant<int, 2>{}); break;
        case 4:  launch_ncols(std::integral_constant<int, 4>{}); break;
        case 8:  launch_ncols(std::integral_constant<int, 8>{}); break;
        default: GGML_ASSERT(false); break;
    }
}

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    const int stride_row_x = ne00 / ggml_blck_size(src0->type);
    const int stride_col_y = src1_padded_row_size / QK8_1;

    ggml_cuda_mm_fusion_args_device fusion_local{};
    mul_mat_vec_q_switch_type(
        src0_dd_i, src0->type, src1_ddq_i, nullptr, fusion_local, dst_dd_i, ne00, row_diff, src1_ncols, stride_row_x, stride_col_y, nrows_dst,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, stream);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_ncols, src1_padded_row_size);
}

// qwen4exp IQ4_NL/Q8_0 routed down projection fused with the 10-expert weighted sum (ported from
// halo-box/strix-llama.cpp, ggml_cuda_mul_mat_id_weighted_rdna3_5): compute all selected experts
// for one output row in a wave and apply their routing weights immediately, avoiding the
// [n_embd, n_used] intermediate and its separate reduction launch. Single-token (n_tokens == 1)
// only - gated by the _ok() shape checks. RDNA3.5 only (B parity).
#if defined(__HIP_PLATFORM_AMD__)
static __device__ __forceinline__ float mmvq_hc_mul_rn(const float a, const float b) {
    float result;
    asm("v_mul_f32_e32 %0, %1, %2" : "=v"(result) : "v"(a), "v"(b));
    return result;
}

static __device__ __forceinline__ float mmvq_hc_add_rn(const float a, const float b) {
    float result;
    asm("v_add_f32_e32 %0, %1, %2" : "=v"(result) : "v"(a), "v"(b));
    return result;
}
#else
static __device__ __forceinline__ float mmvq_hc_mul_rn(const float a, const float b) { return __fmul_rn(a, b); }
static __device__ __forceinline__ float mmvq_hc_add_rn(const float a, const float b) { return __fadd_rn(a, b); }
#endif

template <int n_expert_used>
__launch_bounds__(32, 8)
static __global__ void mul_mat_id_iq4_nl_weighted_rdna3_5(
        const void * vx_ptr, const block_q8_1 * y, const int32_t * ids, const float * weights, float * dst,
        const int nrows, const int blocks_per_row, const int stride_row_x, const int stride_channel_x,
        const int stride_y) {
    constexpr int qi        = ggml_cuda_type_traits<GGML_TYPE_IQ4_NL>::qi;
    constexpr int vdr       = VDR_Q4_0_Q8_1_MMVQ;
    constexpr int warp_size = 32;
    constexpr int blocks_per_iter = vdr * warp_size / qi;

    const int lane = threadIdx.x;
    const int row  = blockIdx.x;
    if (row >= nrows) {
        return;
    }

    float result = 0.0f;
#pragma unroll
    for (int ex = 0; ex < n_expert_used; ++ex) {
        const int channel = ids[ex];
        const int x_off = channel * stride_channel_x + row * stride_row_x;
        float sum = 0.0f;
        for (int kbx = lane / (qi/vdr); kbx < blocks_per_row; kbx += blocks_per_iter) {
            const int kqs = vdr * (lane % (qi/vdr));
            sum += vec_dot_iq4_nl_q8_1(vx_ptr, y + ex * stride_y + kbx, x_off + kbx, kqs);
        }
        sum = warp_reduce_sum<warp_size>(sum);
        const float term = mmvq_hc_mul_rn(sum, weights[ex]);
        result = ex == 0 ? term : mmvq_hc_add_rn(result, term);
    }

    if (lane == 0) {
        dst[row] = result;
    }
}

template <int n_expert_used>
__launch_bounds__(32, 8)
static __global__ void mul_mat_id_q8_0_weighted_rdna3_5(
        const void * vx_ptr, const block_q8_1 * y, const int32_t * ids, const float * weights, float * dst,
        const int nrows, const int blocks_per_row, const int stride_row_x, const int stride_channel_x,
        const int stride_y) {
    constexpr int qi        = QI8_0;
    constexpr int vdr       = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int warp_size = 32;
    constexpr int blocks_per_iter = vdr * warp_size / qi;

    const int lane = threadIdx.x;
    const int row  = blockIdx.x;
    if (row >= nrows) {
        return;
    }

    float result = 0.0f;
#pragma unroll
    for (int ex = 0; ex < n_expert_used; ++ex) {
        const int channel = ids[ex];
        const int x_off = channel * stride_channel_x + row * stride_row_x;
        float sum = 0.0f;
        for (int kbx = lane / (qi/vdr); kbx < blocks_per_row; kbx += blocks_per_iter) {
            const int kqs = vdr * (lane % (qi/vdr));
            sum += vec_dot_q8_0_q8_1(vx_ptr, y + ex * stride_y + kbx, x_off + kbx, kqs);
        }
        sum = warp_reduce_sum<warp_size>(sum);
        const float term = mmvq_hc_mul_rn(sum, weights[ex]);
        result = ex == 0 ? term : mmvq_hc_add_rn(result, term);
    }

    if (lane == 0) {
        dst[row] = result;
    }
}

bool ggml_cuda_mul_mat_id_weighted_rdna3_5_ok(
        const ggml_tensor * experts, const ggml_tensor * weights, const ggml_tensor * dst) {
    if (experts->op != GGML_OP_MUL_MAT_ID ||
            (experts->src[0]->type != GGML_TYPE_IQ4_NL && experts->src[0]->type != GGML_TYPE_Q8_0) ||
            experts->src[1]->type != GGML_TYPE_F32 || experts->src[2]->type != GGML_TYPE_I32 ||
            experts->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    const ggml_tensor * w   = experts->src[0];
    const ggml_tensor * y   = experts->src[1];
    const ggml_tensor * ids = experts->src[2];
    const int64_t n_used = ids->ne[0];
    // RDNA3_5: see ggml_cuda_rdna3_5_dense_glu_disabled (ggml-cuda.cu).  This weighted-down
    // tail is single-token-only too, so keep it opt-in on that arch as well - otherwise a
    // 1-token decode and an n-token verify of the same MoE layer are not bit-identical.
    static const bool enable = [] {
        const char * env = getenv("GGML_CUDA_ENABLE_RDNA3_5_SINGLE_TOKEN_FUSIONS");
        return env != nullptr && std::atoi(env) != 0;
    }();
    return enable && GGML_CUDA_CC_IS_RDNA3_5(ggml_cuda_info().devices[ggml_cuda_get_device()].cc) &&
        n_used == 10 && w->ne[0] == 640 && w->ne[1] == 2560 && w->ne[2] == 512 && w->ne[3] == 1 &&
        y->ne[0] == 640 && ggml_nelements(y) == 640 * n_used && ggml_is_contiguous(y) &&
        ggml_nelements(ids) == n_used && ggml_is_contiguous(ids) &&
        weights->ne[0] == 1 && weights->ne[1] == n_used && ggml_nelements(weights) == n_used && ggml_is_contiguous(weights) &&
        ggml_nelements(experts) == 2560 * n_used && ggml_is_contiguous(experts) &&
        ggml_nelements(dst) == 2560 && ggml_is_contiguous(dst);
}

void ggml_cuda_mul_mat_id_weighted_rdna3_5(
        ggml_backend_cuda_context & ctx, const ggml_tensor * experts, const ggml_tensor * weights, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_mul_mat_id_weighted_rdna3_5_ok(experts, weights, dst));
    const ggml_tensor * w   = experts->src[0];
    const ggml_tensor * y   = experts->src[1];
    const ggml_tensor * ids = experts->src[2];
    constexpr int n_used = 10;
    constexpr int ncols  = 640;
    constexpr int nrows  = 2560;
    constexpr int nblocks = ncols / QK8_1;

    ggml_cuda_pool_alloc<block_q8_1> y_q8(ctx.pool(), n_used * nblocks);
    quantize_row_q8_1_cuda(
        (const float *) y->data, nullptr, y_q8.get(), w->type,
        ncols,
        y->nb[1] / sizeof(float), y->nb[2] / sizeof(float), y->nb[3] / sizeof(float),
        y->ne[0], y->ne[1], y->ne[2], y->ne[3], ctx.stream());

    const ggml_cuda_kernel_launch_params params(nrows, 32, 0, ctx.stream());
    if (w->type == GGML_TYPE_IQ4_NL) {
        ggml_cuda_kernel_launch(mul_mat_id_iq4_nl_weighted_rdna3_5<n_used>, params,
            w->data, y_q8.get(), (const int32_t *) ids->data, (const float *) weights->data, (float *) dst->data,
            nrows, nblocks, (int) (w->nb[1] / ggml_type_size(w->type)),
            (int) (w->nb[2] / ggml_type_size(w->type)), nblocks);
    } else {
        ggml_cuda_kernel_launch(mul_mat_id_q8_0_weighted_rdna3_5<n_used>, params,
            w->data, y_q8.get(), (const int32_t *) ids->data, (const float *) weights->data, (float *) dst->data,
            nrows, nblocks, (int) (w->nb[1] / ggml_type_size(w->type)),
            (int) (w->nb[2] / ggml_type_size(w->type)), nblocks);
    }
}
