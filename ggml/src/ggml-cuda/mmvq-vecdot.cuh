// Shared vec-dot / VDR selectors for the MMVQ kernels (immovable module-local
// declarations split out of mmvq.cu so the MoE kernel (mmvq-moe.cu) and the dense
// dispatch (mmvq.cu) can share them).
#pragma once

#include "common.cuh"
#include "vecdotq.cuh"

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
#if defined(GGML_CUDA_Q4K_DENSE_VDR4) && defined(GGML_CUDA_Q4K_MOE_VDR4)
        // dense and MoE both: same 32-element layout and VDR as vec_dot_q4_K_q8_1_vdr4, one integer chain per sub-block
        case GGML_TYPE_Q4_K:    return vec_dot_q4_K_q8_1_vdr4_dense;
#elif defined(GGML_CUDA_Q4K_DENSE_VDR4)
        case GGML_TYPE_Q4_K:    return moe ? vec_dot_q4_K_q8_1_vdr4 : vec_dot_q4_K_q8_1_vdr4_dense;
#elif defined(GGML_CUDA_Q4K_MOE_VDR4)
        case GGML_TYPE_Q4_K:    return moe ? vec_dot_q4_K_q8_1_vdr4_dense : vec_dot_q4_K_q8_1;
#else
        case GGML_TYPE_Q4_K:    return moe ? vec_dot_q4_K_q8_1_vdr4 : vec_dot_q4_K_q8_1;
#endif
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
