#include "convert.cuh"
#include "fwht.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml.h"
#include "rope.cuh"
#include "set-rows.cuh"

// The ROPE -> VIEW -> SET_ROWS fusion (ggml_cuda_should_fuse_rope_set_rows, upstream #16884) is
// selected by ggml_cuda_check_fusion_memory_ranges(), i.e. by buffer addresses, so its fused
// (D=half/bf16) kernel must round identically to the unfused chain (D=float/float + k_set_rows).
// It does not: clang contracts the multiply-adds of the two template instantiations differently,
// and a single f16 cache element of a 27B IMROPE prefill then differs by 1 ULP, which amplifies
// into different greedy text across process starts (issue #67).  Disable FP contraction for this
// file so every rope instantiation uses the same rounding; the fused kernels then reproduce the
// chain they replace.  The unfused chain's rounding moves to the contracted-off form as well
// (both agree on the new value); the 4B same-seed gate and the width probe are unchanged.
#pragma clang fp contract(off)

struct rope_corr_dims {
    float v[2];
};


struct mrope_sections {
    int v[4];
};

static __device__ float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
template<bool forward>
static __device__ void rope_yarn(
        const float theta_extrap, const float freq_scale, const rope_corr_dims corr_dims, const int64_t i0, const float ext_factor,
        float mscale, float & cos_theta, float & sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    cos_theta = cosf(theta) * mscale;
    sin_theta = sinf(theta) * mscale;
    if (!forward) {
        sin_theta *= -1.0f;
    }
}

template <bool forward, bool has_ff, typename T, typename D>
static __global__ void rope_norm(const T *            x,
                                 D *                  dst,
                                 const int            ne00,
                                 const int            ne01,
                                 const int            ne02,
                                 const int            s01,
                                 const int            s02,
                                 const int            s03,
                                 const int            s1,
                                 const int            s2,
                                 const int            s3,
                                 const int            n_dims,
                                 const int            n_offs,
                                 const int32_t *      pos,
                                 const float          freq_scale,
                                 const float          ext_factor,
                                 const float          attn_factor,
                                 const rope_corr_dims corr_dims,
                                 const float          theta_scale,
                                 const float *        freq_factors,
                                 const int64_t *      row_indices,
                                 const int            set_rows_stride,
                                 const bool           inplace) {
    const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int       idst = i0 + i1 * s1  + i2 * s2  + i3 * s3;
    const int ix   = i0 + i1 * s01 + i2 * s02 + i3 * s03;
    // Fusion optimization: ROPE + VIEW + SET_ROWS.
    // The rope output is viewed as a 1D tensor and offset based on a row index in row_indices.
    if (set_rows_stride != 0) {
        idst = i1 * s1 + i0;
        idst += row_indices[i2] * set_rows_stride;
    }

    const auto & store_coaelsced = [&](float x0, float x1) {
        if constexpr (std::is_same_v<float, D>) {
            float2 v = make_float2(x0, x1);
            ggml_cuda_memcpy_1<8>(dst + idst, &v);
        } else if constexpr (std::is_same_v<half, D>) {
            half2 v = make_half2(x0, x1);
            ggml_cuda_memcpy_1<4>(dst + idst, &v);
        }
    };
    if (i0 < n_offs || i0 >= n_offs + n_dims) {
        if (inplace) {
            return;
        }
        store_coaelsced(x[ix + 0], x[ix + 1]);
        return;
    }

    const int iw = i0 - n_offs; // relative idx

    const float theta_base = pos[i2]*powf(theta_scale, iw/2.0f);

    const float freq_factor = has_ff ? freq_factors[iw/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, iw, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + 1];

    store_coaelsced(x0 * cos_theta - x1 * sin_theta, x0 * sin_theta + x1 * cos_theta);
}

template <bool forward, bool has_ff, typename T, typename D>
static __global__ void rope_neox(const T *            x,
                                 D *                  dst,
                                 const int            ne00,
                                 const int            ne01,
                                 const int            ne02,
                                 const int            s01,
                                 const int            s02,
                                 const int            s03,
                                 const int            s1,
                                 const int            s2,
                                 const int            s3,
                                 const int            n_dims,
                                 const int            n_offs,
                                 const int32_t *      pos,
                                 const float          freq_scale,
                                 const float          ext_factor,
                                 const float          attn_factor,
                                 const rope_corr_dims corr_dims,
                                 const float          theta_scale,
                                 const float *        freq_factors,
                                 const int64_t *      row_indices,
                                 const int            set_rows_stride,
                                 const bool           inplace) {
    ggml_cuda_pdl_lc();
    const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int       idst = i0 / 2 + i1 * s1  + i2 * s2  + i3 * s3;
    const int ix   = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;
    ggml_cuda_pdl_sync();

    // Fusion optimization: ROPE + VIEW + SET_ROWS.
    // The rope output is viewed as a 1D tensor and offset based on a row index in row_indices.
    if (set_rows_stride != 0) {
        idst = i1 * s1 + i0 / 2;
        idst += row_indices[i2] * set_rows_stride;
    }

    if (i0 < n_offs || i0 >= n_offs + n_dims) {
        if (inplace) {
            return;
        }
        dst[idst + i0 / 2 + 0] = ggml_cuda_cast<D>(x[ix + i0 / 2 + 0]);
        dst[idst + i0 / 2 + 1] = ggml_cuda_cast<D>(x[ix + i0 / 2 + 1]);

        return;
    }

    const int iw = i0 - n_offs; // relative idx

    const float theta_base = pos[i2]*powf(theta_scale, iw/2.0f);

    const float freq_factor = has_ff ? freq_factors[iw/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, iw, ext_factor, attn_factor, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    const float x0 = x[ix + n_offs/2 + 0];
    const float x1 = x[ix + n_offs/2 + n_dims/2];

    dst[idst + n_offs/2 + 0]          = ggml_cuda_cast<D>(x0 * cos_theta - x1 * sin_theta);
    dst[idst + n_offs/2 + n_dims / 2] = ggml_cuda_cast<D>(x0 * sin_theta + x1 * cos_theta);
}

template <bool forward, bool has_ff, typename T, typename D>
static __global__ void rope_multi(const T *            x,
                                  D *                  dst,
                                  const int            ne00,
                                  const int            ne01,
                                  const int            ne02,
                                  const int            s01,
                                  const int            s02,
                                  const int            s03,
                                  const int            s1,
                                  const int            s2,
                                  const int            s3,
                                  const int            n_dims,
                                  const int            n_offs,
                                  const int32_t *      pos,
                                  const float          freq_scale,
                                  const float          ext_factor,
                                  const float          attn_factor,
                                  const rope_corr_dims corr_dims,
                                  const float          theta_scale,
                                  const float *        freq_factors,
                                  const mrope_sections sections,
                                  const bool           is_imrope,
                                  const int64_t *      row_indices,
                                  const int            set_rows_stride,
                                  const bool           inplace) {

    const int i0 = 2 * (blockDim.y * blockIdx.y + threadIdx.y);

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int       idst = i0 / 2 + i1 * s1  + i2 * s2  + i3 * s3;
    const int ix   = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    // Fusion optimization: ROPE + VIEW + SET_ROWS.
    // The rope output is viewed as a 1D tensor and offset based on a row index in row_indices.
    if (set_rows_stride != 0) {
        idst = i1 * s1 + i0 / 2;
        idst += row_indices[i2] * set_rows_stride;
    }

    ggml_cuda_pdl_sync();

    // Fusion optimization: ROPE + VIEW + SET_ROWS.
    // The rope output is viewed as a 1D tensor and offset based on a row index in row_indices.
    if (set_rows_stride != 0) {
        idst = i1 * s1 + i0 / 2;
        idst += row_indices[i2] * set_rows_stride;
    }

    if (i0 < n_offs || i0 >= n_offs + n_dims) {
        if (inplace) {
            return;
        }
        dst[idst + i0/2 + 0] = ggml_cuda_cast<D>(x[ix + i0/2 + 0]);
        dst[idst + i0/2 + 1] = ggml_cuda_cast<D>(x[ix + i0/2 + 1]);

        return;
    }

    const int iw = i0 - n_offs; // relative idx

    const int sect_dims = sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
    const int sec_w = sections.v[1] + sections.v[0];
    const int sector = (iw / 2) % sect_dims;

    // the same power in every section arm: computed once so a divergent wave does not run powf per arm
    const float theta_pow = powf(theta_scale, iw / 2.0f);

    float theta_base = 0.0;
    if (is_imrope) {
        if (sector % 3 == 1 && sector < 3 * sections.v[1]) {         // h
            theta_base = pos[i2 + ne02 * 1] * theta_pow;
        } else if (sector % 3 == 2 && sector < 3 * sections.v[2]) {  // w
            theta_base = pos[i2 + ne02 * 2] * theta_pow;
        } else if (sector % 3 == 0 && sector < 3 * sections.v[0]) {  // t
            theta_base = pos[i2] * theta_pow;
        } else {
            theta_base = pos[i2 + ne02 * 3] * theta_pow;
        }
    } else {
        if (sector < sections.v[0]) {
            theta_base = pos[i2] * theta_pow;
        } else if (sector >= sections.v[0] && sector < sec_w) {
            theta_base = pos[i2 + ne02 * 1] * theta_pow;
        } else if (sector >= sec_w && sector < sec_w + sections.v[2]) {
            theta_base = pos[i2 + ne02 * 2] * theta_pow;
        } else if (sector >= sec_w + sections.v[2]) {
            theta_base = pos[i2 + ne02 * 3] * theta_pow;
        }
    }

    const float freq_factor = has_ff ? freq_factors[iw/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, iw, ext_factor, attn_factor, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    const float x0 = x[ix + n_offs/2 + 0];
    const float x1 = x[ix + n_offs/2 + n_dims/2];

    dst[idst + n_offs/2 + 0]        = ggml_cuda_cast<D>(x0*cos_theta - x1*sin_theta);
    dst[idst + n_offs/2 + n_dims/2] = ggml_cuda_cast<D>(x0*sin_theta + x1*cos_theta);
}

template <bool forward, bool has_ff, typename T>
static __global__ void rope_vision(const T *            x,
                                   T *                  dst,
                                   const int            ne00,
                                   const int            ne01,
                                   const int            ne02,
                                   const int            s01,
                                   const int            s02,
                                   const int            s03,
                                   const int            s1,
                                   const int            s2,
                                   const int            s3,
                                   const int            n_dims,
                                   const int32_t *      pos,
                                   const float          freq_scale,
                                   const float          ext_factor,
                                   const float          attn_factor,
                                   const rope_corr_dims corr_dims,
                                   const float          theta_scale,
                                   const float *        freq_factors,
                                   const mrope_sections sections) {
    const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne00) {
        return;
    }

    const int row_dst = blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = row_dst / (ne01 * ne02);
    const uint32_t i2 = (row_dst - i3 * ne01 * ne02) / ne01;
    const uint32_t i1 = row_dst - i3 * ne01 * ne02 - i2 * ne01;

    int       idst = i0 / 2 + i1 * s1  + i2 * s2  + i3 * s3;
    const int ix   = i0 / 2 + i1 * s01 + i2 * s02 + i3 * s03;

    ggml_cuda_pdl_sync();
    const int sect_dims = sections.v[0] + sections.v[1];
    const int sec_w     = sections.v[1] + sections.v[0];
    const int sector    = (i0 / 2) % sect_dims;

    float theta_base = 0.0;
    if (sector < sections.v[0]) {
        const int p = sector;
        theta_base  = pos[i2] * powf(theta_scale, p);
    } else if (sector >= sections.v[0] && sector < sec_w) {
        const int p = sector - sections.v[0];
        theta_base  = pos[i2 + ne02] * powf(theta_scale, p);
    }

    const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + n_dims];

    dst[idst + 0]      = x0*cos_theta - x1*sin_theta;
    dst[idst + n_dims] = x0*sin_theta + x1*cos_theta;
}

template <bool forward, typename T, typename D>
static void rope_norm_cuda(const T *            x,
                           D *                  dst,
                           const int            ne00,
                           const int            ne01,
                           const int            ne02,
                           const int            s01,
                           const int            s02,
                           const int            s03,
                           const int            s1,
                           const int            s2,
                           const int            s3,
                           const int            n_dims,
                           const int            n_offs,
                           const int            nr,
                           const int32_t *      pos,
                           const float          freq_scale,
                           const float          freq_base,
                           const float          ext_factor,
                           const float          attn_factor,
                           const rope_corr_dims corr_dims,
                           const float *        freq_factors,
                           const int64_t *      row_indices,
                           const int            set_rows_stride,
                           const bool           inplace,
                           cudaStream_t         stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int  n_blocks_x = (ne00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    if (freq_factors == nullptr) {
        rope_norm<forward, false><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride, inplace);
    } else {
        rope_norm<forward, true><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride, inplace);
    }
}

template <bool forward, typename T, typename D>
static void rope_neox_cuda(const T *            x,
                           D *                  dst,
                           const int            ne00,
                           const int            ne01,
                           const int            ne02,
                           const int            s01,
                           const int            s02,
                           const int            s03,
                           const int            s1,
                           const int            s2,
                           const int            s3,
                           const int            n_dims,
                           const int            n_offs,
                           const int            nr,
                           const int32_t *      pos,
                           const float          freq_scale,
                           const float          freq_base,
                           const float          ext_factor,
                           const float          attn_factor,
                           const rope_corr_dims corr_dims,
                           const float *        freq_factors,
                           const int64_t *      row_indices,
                           const int            set_rows_stride,
                           const bool           inplace,
                           cudaStream_t         stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int  n_blocks_x = (ne00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);
    const ggml_cuda_kernel_launch_params launch_params = {block_nums, block_dims, 0, stream};

    if (freq_factors == nullptr) {
        ggml_cuda_kernel_launch(rope_neox<forward, false, T, D>, launch_params,
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride, inplace);
    } else {
        ggml_cuda_kernel_launch(rope_neox<forward, true, T, D>, launch_params,
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride, inplace);
    }
}

template <bool forward, typename T, typename D>
static void rope_multi_cuda(const T *            x,
                            D *                  dst,
                            const int            ne00,
                            const int            ne01,
                            const int            ne02,
                            const int            s01,
                            const int            s02,
                            const int            s03,
                            const int            s1,
                            const int            s2,
                            const int            s3,
                            const int            n_dims,
                            const int            n_offs,
                            const int            nr,
                            const int32_t *      pos,
                            const float          freq_scale,
                            const float          freq_base,
                            const float          ext_factor,
                            const float          attn_factor,
                            const rope_corr_dims corr_dims,
                            const float *        freq_factors,
                            const mrope_sections sections,
                            const bool           is_imrope,
                            const int64_t *      row_indices,
                            const int            set_rows_stride,
                            const bool           inplace,
                            cudaStream_t         stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int  n_blocks_x = (ne00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    if (freq_factors == nullptr) {
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        ggml_cuda_kernel_launch(rope_multi<forward, false, T, D>, launch_params,
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections, is_imrope, row_indices, set_rows_stride, inplace);

    } else {
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        ggml_cuda_kernel_launch(rope_multi<forward, true, T, D>, launch_params,
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, n_offs, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections, is_imrope, row_indices, set_rows_stride, inplace);

    }
}

template <bool forward, typename T>
static void rope_vision_cuda(const T *            x,
                             T *                  dst,
                             const int            ne00,
                             const int            ne01,
                             const int            ne02,
                             const int            s01,
                             const int            s02,
                             const int            s03,
                             const int            s1,
                             const int            s2,
                             const int            s3,
                             const int            n_dims,
                             const int            nr,
                             const int32_t *      pos,
                             const float          freq_scale,
                             const float          freq_base,
                             const float          ext_factor,
                             const float          attn_factor,
                             const rope_corr_dims corr_dims,
                             const float *        freq_factors,
                             const mrope_sections sections,
                             cudaStream_t         stream) {
    GGML_ASSERT(ne00 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int  n_blocks_x = (ne00 + 2 * CUDA_ROPE_BLOCK_SIZE - 1) / (2 * CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);
    // break down (head_dim, heads, seq) into (CUDA_ROPE_BLOCK_SIZE, x, heads * seq)
    // where x ~= ceil(head_dim / CUDA_ROPE_BLOCK_SIZE);

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    if (freq_factors == nullptr) {
        rope_vision<forward, false, T><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections);
    } else {
        rope_vision<forward, true, T><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections);
    }
}

template <bool forward>
void ggml_cuda_op_rope_impl(ggml_backend_cuda_context & ctx,
                            ggml_tensor *               dst,
                            const ggml_tensor *         set_rows = nullptr) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    const float * src0_d = (const float *)src0->data;
    const float * src1_d = (const float *)src1->data;

    void *          dst_d           = dst->data;
    const int64_t * row_indices     = nullptr;
    ggml_type       dst_type        = dst->type;
    int             set_rows_stride = 0;

    if (set_rows != nullptr) {
        GGML_ASSERT(forward);
        dst_d           = set_rows->data;
        row_indices     = (const int64_t *) set_rows->src[1]->data;
        dst_type        = set_rows->type;
        set_rows_stride = set_rows->nb[1] / ggml_type_size(set_rows->type);
    }
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    // When not fused, src0 and dst types must match
    // When fused (ROPE+VIEW+SET_ROWS), src0 may be F32 and dst may be F16 or BF16
    GGML_ASSERT(src0->type == dst->type || (src0->type == GGML_TYPE_F32 && (dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_BF16)));

    const int64_t ne00 = src0->ne[0]; // head dims
    const int64_t ne01 = src0->ne[1]; // num heads
    const int64_t ne02 = src0->ne[2]; // num heads
    const int64_t nr = ggml_nrows(src0);

    const size_t s01 = src0->nb[1] / ggml_type_size(src0->type);
    const size_t s02 = src0->nb[2] / ggml_type_size(src0->type);
    const size_t s03 = src0->nb[3] / ggml_type_size(src0->type);

    const size_t s1 = dst->nb[1] / ggml_type_size(dst->type);
    const size_t s2 = dst->nb[2] / ggml_type_size(dst->type);
    const size_t s3 = dst->nb[3] / ggml_type_size(dst->type);

    //const int n_past     = ((int32_t *) dst->op_params)[0];
    const int n_dims     = ((int32_t *) dst->op_params)[1];
    const int mode       = ((int32_t *) dst->op_params)[2];
    //const int n_ctx      = ((int32_t *) dst->op_params)[3];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];
    const int n_offs     = ((int32_t *) dst->op_params)[15];
    mrope_sections sections;

    // when dst aliases src0, the channels outside the rotated window already hold the correct data
    const bool inplace = dst_d == src0->data;

    // RoPE alteration for extended context
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;

    memcpy(&freq_base,   (int32_t *) dst->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));
    memcpy(&sections.v,  (int32_t *) dst->op_params + 11, sizeof(int)*4);

    const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;
    const bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;
    const bool is_imrope = mode == GGML_ROPE_TYPE_IMROPE;
    const bool is_vision = mode == GGML_ROPE_TYPE_VISION;

    if (is_mrope) {
        GGML_ASSERT(sections.v[0] > 0 || sections.v[1] > 0 || sections.v[2] > 0);
    }

    if (is_vision) {
        GGML_ASSERT(n_dims == ne00/2);
        GGML_ASSERT(n_offs == 0); // offset not supported for vision, as the rotated pairs span the whole row
    }

    const int32_t * pos = (const int32_t *) src1_d;

    const float * freq_factors = nullptr;
    if (src2 != nullptr) {
        freq_factors = (const float *) src2->data;
    }

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims.v);

    // compute
    if (is_neox) {
        if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_neox_cuda<forward, float, float>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02,
                                                  s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                  ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                  set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_neox_cuda<forward, float, half>((const float *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                 s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                 ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                 set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_neox_cuda<forward, half, half>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_BF16) {
            rope_neox_cuda<forward, float, nv_bfloat16>((const float *) src0_d, (nv_bfloat16 *) dst_d, ne00, ne01, ne02, s01,
                                                        s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                        ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                        set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_BF16) {
            rope_neox_cuda<forward, half, nv_bfloat16>((const half *) src0_d, (nv_bfloat16 *) dst_d, ne00, ne01, ne02, s01,
                                                       s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                       ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                       set_rows_stride, inplace, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    } else if (is_mrope && !is_vision) {
        if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_multi_cuda<forward, float, float>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02,
                                                   s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                   ext_factor, attn_factor, corr_dims, freq_factors, sections, is_imrope,
                                                   row_indices, set_rows_stride, inplace, stream);

        } else if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_multi_cuda<forward, float, half>((const float *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                  s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                  ext_factor, attn_factor, corr_dims, freq_factors, sections, is_imrope,
                                                  row_indices, set_rows_stride, inplace, stream);

        } else if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_BF16) {
            rope_multi_cuda<forward, float, nv_bfloat16>((const float *) src0_d, (nv_bfloat16 *) dst_d, ne00, ne01, ne02, s01,
                                                         s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                                                         freq_base, ext_factor, attn_factor, corr_dims, freq_factors, sections,
                                                         is_imrope, row_indices, set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_multi_cuda<forward, half, half>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                 s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                 ext_factor, attn_factor, corr_dims, freq_factors, sections, is_imrope,
                                                 row_indices, set_rows_stride, inplace, stream);

        } else if (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_BF16) {
            rope_multi_cuda<forward, half, nv_bfloat16>((const half *) src0_d, (nv_bfloat16 *) dst_d, ne00, ne01, ne02, s01,
                                                        s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale,
                                                        freq_base, ext_factor, attn_factor, corr_dims, freq_factors, sections,
                                                        is_imrope, row_indices, set_rows_stride, inplace, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    } else if (is_vision) {
        if (src0->type == GGML_TYPE_F32) {
            rope_vision_cuda<forward>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02, s03, s1,
                                      s2, s3, n_dims, nr, pos, freq_scale, freq_base, ext_factor, attn_factor,
                                      corr_dims, freq_factors, sections, stream);
        } else if (src0->type == GGML_TYPE_F16) {
            rope_vision_cuda<forward>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02, s03, s1,
                                      s2, s3, n_dims, nr, pos, freq_scale, freq_base, ext_factor, attn_factor,
                                      corr_dims, freq_factors, sections, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    } else {
        if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
            rope_norm_cuda<forward, float, float>((const float *) src0_d, (float *) dst_d, ne00, ne01, ne02, s01, s02,
                                                  s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                  ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                  set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
            rope_norm_cuda<forward, float, half>((const float *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                 s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                 ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                 set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
            rope_norm_cuda<forward, half, half>((const half *) src0_d, (half *) dst_d, ne00, ne01, ne02, s01, s02,
                                                s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_F32 && dst_type == GGML_TYPE_BF16) {
            rope_norm_cuda<forward, float, nv_bfloat16>((const float *) src0_d, (nv_bfloat16 *) dst_d, ne00, ne01, ne02, s01,
                                                        s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                        ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                        set_rows_stride, inplace, stream);
        } else if (src0->type == GGML_TYPE_F16 && dst_type == GGML_TYPE_BF16) {
            rope_norm_cuda<forward, half, nv_bfloat16>((const half *) src0_d, (nv_bfloat16 *) dst_d, ne00, ne01, ne02, s01,
                                                       s02, s03, s1, s2, s3, n_dims, n_offs, nr, pos, freq_scale, freq_base,
                                                       ext_factor, attn_factor, corr_dims, freq_factors, row_indices,
                                                       set_rows_stride, inplace, stream);
        } else {
            GGML_ABORT("fatal error");
        }
    }
}

void ggml_cuda_op_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_rope_impl<true>(ctx, dst);
}

void ggml_cuda_op_rope_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_rope_impl<false>(ctx, dst);
}

void ggml_cuda_op_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * rope, ggml_tensor * set_rows) {
    ggml_cuda_op_rope_impl<true>(ctx, rope, set_rows);
}

// KV-cache writes folded into the fwht_n kernel: SET_ROWS (q8_0) of the rotated K row and, optionally, of the V row
// (already rotated, F32), for one token per channel
struct rope_kv_q8_0 {
    block_q8_0    * k_dst;    // K cache view; the rotated row goes here instead of dst
    const int64_t * k_idx;    // destination cache row per token
    int64_t         k_nb1;    // bytes per cache row
    const float   * v_src;    // nullptr: no V write
    int64_t         v_s1;     // floats per token row of v_src
    int             v_fwht_n; // 0: v_src is the rotated V row; 64: v_src is the unrotated row, rotated here (fwht_cuda<64>)
    float           v_fwht_scale;
    block_q8_0    * v_dst;
    const int64_t * v_idx;
    int64_t         v_nb1;
    // plain (F32 / F16, the K cache's type) V row write next to the non-FWHT K SET_ROWS; vp_src == nullptr: none
    const float   * vp_src;
    int64_t         vp_s1;    // floats per token row of vp_src
    void          * vp_dst;
    const int64_t * vp_idx;
    int64_t         vp_nb1;   // bytes per V cache row
};

// fused RMS_NORM + MUL + ROPE (+ VIEW + SET_ROWS)
// one block per row: block_reduce gives the norm scale, then each thread applies mul and rope to the elements it owns
// fwht_n > 0 (ncols == fwht_n): the KV rotation's Hadamard MUL_MAT follows; the roped row goes through shared
// memory into fwht_cuda's in-warp FWHT (scaled by fwht_scale) and only the rotated row is written to dst
// kv (with fwht_n): the rotated row is written as q8_0 to the K cache instead (and the V row to the V cache), with
// k_set_rows_q8_0_group's quantization; row r of the block covers elements [r*ncols, (r+1)*ncols) of the token's row
// no_norm: ROPE alone (no RMS_NORM / MUL in front, x read as is), for the rope + FWHT forms
template <int block_size, bool has_ff, typename D, int fwht_n = 0, bool kv = false, bool no_norm = false>
static __device__ __forceinline__ void rms_norm_mul_rope_f32_dev(
        const float * x, D * dst, const int ncols, const int nchannels, const int nsamples,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s1, const int64_t s2, const int64_t s3,
        const float eps,
        const float * mul,
        const int64_t mul_s01, const int64_t mul_s02, const int64_t mul_s03,
        const uint3 mul_ncols_packed, const uint3 mul_nrows_packed,
        const uint3 mul_nchannels_packed, const uint3 mul_nsamples_packed,
        const int n_dims, const int32_t * pos,
        const float freq_scale, const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims, const float theta_scale,
        const float * freq_factors,
        const int64_t * row_indices, const int set_rows_stride,
        const bool is_neox, const float fwht_scale, const rope_kv_q8_0 kv_args,
        const mrope_sections sections, const int mrope,
        const int row_) {
    static_assert(!kv || fwht_n > 0, "the KV-cache writes need the FWHT path");
    static_assert(!no_norm || fwht_n > 0, "the rope-only form is an FWHT form");
    ggml_cuda_pdl_lc();
    const int row = row_;
    const int tid = threadIdx.x;

    extern __shared__ float s_sum[];
    __shared__ float s_row[fwht_n > 0 ? fwht_n : 1];

    ggml_cuda_pdl_sync();

    // grid.y and grid.z are clamped to the CUDA limit, iterate over the excess channels/samples
    for (int sample = blockIdx.z; sample < nsamples; sample += gridDim.z) {
        for (int channel = blockIdx.y; channel < nchannels; channel += gridDim.y) {
            const float * xc = x + sample*s03 + channel*s02 + row*s01;

            const uint32_t mul_row     = fastmodulo(row,     mul_nrows_packed);
            const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
            const uint32_t mul_sample  = fastmodulo(sample,  mul_nsamples_packed);
            const float * mulc = mul + mul_sample*mul_s03 + mul_channel*mul_s02 + mul_row*mul_s01;

            if constexpr (kv) {
                // V row slice of this block: whole 32-lane groups since ncols % 32 == 0 and ncols <= block_size
                constexpr int warp_size = ggml_cuda_get_physical_warp_size();
                if constexpr (warp_size == 32) {
                    if (kv_args.v_src != nullptr && kv_args.v_fwht_n == 64) {
                        // V's rotation folded in: warp w rotates the 64-element V row slice [row*ncols + 64*w, +64) with
                        // fwht_cuda<64>'s in-warp FWHT (lane l holds elements l and 32 + l), then quantizes both 32-element
                        // groups with the same lane -> element map as below
                        const int w    = tid / warp_size;
                        const int lane = tid % warp_size;
                        if (w < ncols/64) {
                            const int col0 = row*ncols + 64*w;
                            float reg[64/warp_size];
                            ggml_cuda_fwht_warp<64, warp_size>(kv_args.v_src + channel*kv_args.v_s1 + col0, kv_args.v_fwht_scale, reg, lane);
                            block_q8_0 * vrow = (block_q8_0 *) ((char *) kv_args.v_dst + kv_args.v_idx[channel]*kv_args.v_nb1);
#pragma unroll
                            for (int e = 0; e < 64/warp_size; ++e) {
                                const int col = col0 + e*warp_size + lane;
                                ggml_cuda_quantize_q8_0_group(reg[e], vrow + col/QK8_0, col % QK8_0);
                            }
                        }
                    }
                }
                if (kv_args.v_src != nullptr && kv_args.v_fwht_n == 0 && tid < ncols) {
                    const int col = row*ncols + tid;
                    block_q8_0 * vrow = (block_q8_0 *) ((char *) kv_args.v_dst + kv_args.v_idx[channel]*kv_args.v_nb1);
                    ggml_cuda_quantize_q8_0_group(kv_args.v_src[channel*kv_args.v_s1 + col], vrow + col/QK8_0, col % QK8_0);
                }
            }

            if constexpr (!kv) {
                // V row slice of this head into the V cache, converted as k_set_rows does (same cast)
                if (kv_args.vp_src != nullptr) {
                    D * vrow = (D *) ((char *) kv_args.vp_dst + kv_args.vp_idx[channel]*kv_args.vp_nb1);
                    for (int c = tid; c < ncols; c += block_size) {
                        vrow[row*ncols + c] = ggml_cuda_cast<D>(kv_args.vp_src[channel*kv_args.vp_s1 + row*ncols + c]);
                    }
                }
            }

            float scale = 1.0f;
            if constexpr (!no_norm) {
                float tmp = 0.0f;

                for (int col = tid; col < ncols; col += block_size) {
                    const float xi = xc[col];
                    tmp += xi * xi;
                }

                tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

                scale = rsqrtf(tmp/ncols + eps);
            }

            int64_t idst = sample*s3 + channel*s2 + row*s1;
            if (set_rows_stride != 0) {
                idst = row*s1 + row_indices[channel]*set_rows_stride;
            }
            D * dstc = dst + idst;

            const auto store = [&](const int i, const float v) {
                if constexpr (fwht_n > 0) {
                    s_row[i] = v;
                } else {
                    dstc[i] = ggml_cuda_cast<D>(v);
                }
            };

            for (int i0 = 2*tid; i0 < ncols; i0 += 2*block_size) {
                int ix0;
                int ix1;
                if ((is_neox || mrope != 0) && i0 < n_dims) {
                    ix0 = i0/2;
                    ix1 = i0/2 + n_dims/2;
                } else {
                    ix0 = i0 + 0;
                    ix1 = i0 + 1;
                }

                const float x0 = no_norm ? xc[ix0] : scale * xc[ix0] * mulc[fastmodulo(ix0, mul_ncols_packed)];
                const float x1 = no_norm ? xc[ix1] : scale * xc[ix1] * mulc[fastmodulo(ix1, mul_ncols_packed)];

                if (i0 >= n_dims) {
                    store(ix0, x0);
                    store(ix1, x1);
                    continue;
                }

                float theta_base;
                if (mrope == 0) {
                    theta_base = pos[channel]*powf(theta_scale, i0/2.0f);
                } else {
                    // as rope_multi with n_offs == 0 (iw == i0, ne02 == nchannels); mrope 2 = IMROPE
                    const int sect_dims = sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
                    const int sec_w     = sections.v[1] + sections.v[0];
                    const int sector    = (i0 / 2) % sect_dims;
                    // the same power in every section arm (see rope_multi)
                    const float theta_pow = powf(theta_scale, i0 / 2.0f);
                    theta_base = 0.0;
                    if (mrope == 2) {
                        if (sector % 3 == 1 && sector < 3 * sections.v[1]) {         // h
                            theta_base = pos[channel + nchannels * 1] * theta_pow;
                        } else if (sector % 3 == 2 && sector < 3 * sections.v[2]) {  // w
                            theta_base = pos[channel + nchannels * 2] * theta_pow;
                        } else if (sector % 3 == 0 && sector < 3 * sections.v[0]) {  // t
                            theta_base = pos[channel] * theta_pow;
                        } else {
                            theta_base = pos[channel + nchannels * 3] * theta_pow;
                        }
                    } else {
                        if (sector < sections.v[0]) {
                            theta_base = pos[channel] * theta_pow;
                        } else if (sector >= sections.v[0] && sector < sec_w) {
                            theta_base = pos[channel + nchannels * 1] * theta_pow;
                        } else if (sector >= sec_w && sector < sec_w + sections.v[2]) {
                            theta_base = pos[channel + nchannels * 2] * theta_pow;
                        } else if (sector >= sec_w + sections.v[2]) {
                            theta_base = pos[channel + nchannels * 3] * theta_pow;
                        }
                    }
                }
                const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;

                float cos_theta;
                float sin_theta;
                rope_yarn<true>(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

                store(ix0, x0*cos_theta - x1*sin_theta);
                store(ix1, x0*sin_theta + x1*cos_theta);
            }

            if constexpr (fwht_n > 0) {
                constexpr int warp_size = ggml_cuda_get_physical_warp_size();
                __syncthreads();
                if (tid < warp_size) {
                    float reg[fwht_n / warp_size];
                    ggml_cuda_fwht_warp<fwht_n, warp_size>(s_row, fwht_scale, reg, tid);
                    if constexpr (kv) {
                        block_q8_0 * krow = (block_q8_0 *) ((char *) kv_args.k_dst + kv_args.k_idx[channel]*kv_args.k_nb1);
#pragma unroll
                        for (int i = 0; i < fwht_n / warp_size; ++i) {
                            const int col = row*ncols + i*warp_size + tid;
                            ggml_cuda_quantize_q8_0_group(reg[i], krow + col/QK8_0, col % QK8_0);
                        }
                    } else {
#pragma unroll
                        for (int i = 0; i < fwht_n / warp_size; ++i) {
                            dstc[i * warp_size + tid] = reg[i];
                        }
                    }
                }
            }

            if constexpr (block_size > WARP_SIZE) {
                // sync is needed as we reuse s_sum across block_reduce invocations, see #26385
                __syncthreads();
            }
        }
    }
}

template <int block_size, bool has_ff, typename D, int fwht_n = 0, bool kv = false, bool no_norm = false>
static __global__ void rms_norm_mul_rope_f32(
        const float * x, D * dst, const int ncols, const int nchannels, const int nsamples,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s1, const int64_t s2, const int64_t s3,
        const float eps,
        const float * mul,
        const int64_t mul_s01, const int64_t mul_s02, const int64_t mul_s03,
        const uint3 mul_ncols_packed, const uint3 mul_nrows_packed,
        const uint3 mul_nchannels_packed, const uint3 mul_nsamples_packed,
        const int n_dims, const int32_t * pos,
        const float freq_scale, const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims, const float theta_scale,
        const float * freq_factors,
        const int64_t * row_indices, const int set_rows_stride,
        const bool is_neox, const float fwht_scale, const rope_kv_q8_0 kv_args,
        const mrope_sections sections, const int mrope) {
    rms_norm_mul_rope_f32_dev<block_size, has_ff, D, fwht_n, kv, no_norm>(x, dst, ncols, nchannels, nsamples, s01, s02, s03, s1, s2, s3, eps, mul, mul_s01, mul_s02, mul_s03, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale, freq_factors, row_indices, set_rows_stride, is_neox, fwht_scale, kv_args, sections, mrope, blockIdx.x);
}

// The launch arguments of one rms_norm_mul_rope_f32<256, has_ff, D> launch (rms_norm_mul_rope_cuda's capture mode).
template <typename D>
struct rmr_args {
    const float * x; D * dst; int ncols; int nrows; int nchannels; int nsamples;
    int64_t s01, s02, s03, s1, s2, s3; float eps;
    const float * mul; int64_t mul_s01, mul_s02, mul_s03;
    uint3 mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed;
    int n_dims; const int32_t * pos; float freq_scale, ext_factor, attn_factor; rope_corr_dims corr_dims; float theta_scale;
    const float * freq_factors; const int64_t * row_indices; int set_rows_stride; bool is_neox; float fwht_scale;
    rope_kv_q8_0 kv; mrope_sections sections; int mrope;
    int fwht_n; // 0, or the FWHT form's rotation width (rms_norm_mul_rope_f32_qk_pair's fwht_n)
};

// Q and K norm+rope of one attention layer in one launch: blocks [0, q.nrows) are the Q heads, the rest the K heads; each
// block runs exactly the single launch's body with its own arguments, so the results are those of the two launches.
// fwht_n > 0: the quantized-KV form (Q: rms_norm_mul_rope_f32<256, has_ff, float, fwht_n>, K: the same with kv).
template <bool has_ff, typename DQ, typename DK, int fwht_n = 0, bool kv = false>
static __global__ void rms_norm_mul_rope_f32_qk_pair(const rmr_args<DQ> q, const rmr_args<DK> k) {
    if ((int) blockIdx.x < q.nrows) {
        rms_norm_mul_rope_f32_dev<256, has_ff, DQ, fwht_n, false>(q.x, q.dst, q.ncols, q.nchannels, q.nsamples, q.s01, q.s02, q.s03, q.s1, q.s2, q.s3,
            q.eps, q.mul, q.mul_s01, q.mul_s02, q.mul_s03, q.mul_ncols_packed, q.mul_nrows_packed, q.mul_nchannels_packed,
            q.mul_nsamples_packed, q.n_dims, q.pos, q.freq_scale, q.ext_factor, q.attn_factor, q.corr_dims, q.theta_scale,
            q.freq_factors, q.row_indices, q.set_rows_stride, q.is_neox, q.fwht_scale, q.kv, q.sections, q.mrope, blockIdx.x);
    } else {
        rms_norm_mul_rope_f32_dev<256, has_ff, DK, fwht_n, kv>(k.x, k.dst, k.ncols, k.nchannels, k.nsamples, k.s01, k.s02, k.s03, k.s1, k.s2, k.s3,
            k.eps, k.mul, k.mul_s01, k.mul_s02, k.mul_s03, k.mul_ncols_packed, k.mul_nrows_packed, k.mul_nchannels_packed,
            k.mul_nsamples_packed, k.n_dims, k.pos, k.freq_scale, k.ext_factor, k.attn_factor, k.corr_dims, k.theta_scale,
            k.freq_factors, k.row_indices, k.set_rows_stride, k.is_neox, k.fwht_scale, k.kv, k.sections, k.mrope,
            (int) blockIdx.x - q.nrows);
    }
}

template <typename D>
static void rms_norm_mul_rope_cuda(
        const float * x, D * dst,
        const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s1, const int64_t s2, const int64_t s3,
        const float eps,
        const float * mul,
        const int64_t mul_s01, const int64_t mul_s02, const int64_t mul_s03,
        const uint32_t mul_ncols, const uint32_t mul_nrows,
        const uint32_t mul_nchannels, const uint32_t mul_nsamples,
        const int n_dims, const int32_t * pos,
        const float freq_scale, const float freq_base, const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims,
        const float * freq_factors,
        const int64_t * row_indices, const int set_rows_stride,
        const bool is_neox, const int fwht_n, const float fwht_scale, const rope_kv_q8_0 & kv,
        const mrope_sections sections, const int mrope, cudaStream_t stream, rmr_args<D> * capture = nullptr,
        const bool no_norm = false) {
    GGML_ASSERT(ncols % 2 == 0);

    const dim3 blocks_num(nrows, MIN(nchannels, UINT16_MAX), MIN(nsamples, UINT16_MAX));

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
    const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
    const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
    const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);

    if (capture != nullptr) {
        // record the plain 256-thread launch instead of making it (rms_norm_mul_rope_f32_qk_pair)
        // the FWHT form (F32 rows, ncols == fwht_n) records its width for the pair kernel's template
        GGML_ASSERT(ncols < 1024 && (fwht_n == 0 ? kv.k_dst == nullptr : (std::is_same_v<D, float> && ncols == fwht_n)));
        *capture = { x, dst, ncols, nrows, nchannels, nsamples, s01, s02, s03, s1, s2, s3, eps,
                     mul, mul_s01, mul_s02, mul_s03, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                     n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale,
                     freq_factors, row_indices, set_rows_stride, is_neox, fwht_scale, kv, sections, mrope, fwht_n };
        return;
    }

    if constexpr (std::is_same_v<D, float>) {
        if (fwht_n != 0) {
            // one head per block, rotated by the in-block FWHT (see rms_norm_mul_rope_f32)
            GGML_ASSERT(ncols == fwht_n);
            const dim3 block_dims(256, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = {blocks_num, block_dims, 32*sizeof(float), stream};
            const auto launch = [&](auto kernel) {
                ggml_cuda_kernel_launch(kernel, launch_params,
                    x, dst, ncols, nchannels, nsamples, s01, s02, s03, s1, s2, s3, eps, mul, mul_s01, mul_s02, mul_s03,
                    mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                    n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale,
                    freq_factors, row_indices, set_rows_stride, is_neox, fwht_scale, kv, sections, mrope);
            };
            const bool has_ff = freq_factors != nullptr;
            if (no_norm) {
                GGML_ASSERT(capture == nullptr);
                if (kv.k_dst != nullptr) {
                    switch (fwht_n) {
                        case  64: has_ff ? launch(rms_norm_mul_rope_f32<256, true, D,  64, true, true>) : launch(rms_norm_mul_rope_f32<256, false, D,  64, true, true>); break;
                        case 128: has_ff ? launch(rms_norm_mul_rope_f32<256, true, D, 128, true, true>) : launch(rms_norm_mul_rope_f32<256, false, D, 128, true, true>); break;
                        case 256: has_ff ? launch(rms_norm_mul_rope_f32<256, true, D, 256, true, true>) : launch(rms_norm_mul_rope_f32<256, false, D, 256, true, true>); break;
                        default: GGML_ABORT("fatal error");
                    }
                } else {
                    switch (fwht_n) {
                        case  64: has_ff ? launch(rms_norm_mul_rope_f32<256, true, D,  64, false, true>) : launch(rms_norm_mul_rope_f32<256, false, D,  64, false, true>); break;
                        case 128: has_ff ? launch(rms_norm_mul_rope_f32<256, true, D, 128, false, true>) : launch(rms_norm_mul_rope_f32<256, false, D, 128, false, true>); break;
                        case 256: has_ff ? launch(rms_norm_mul_rope_f32<256, true, D, 256, false, true>) : launch(rms_norm_mul_rope_f32<256, false, D, 256, false, true>); break;
                        default: GGML_ABORT("fatal error");
                    }
                }
                return;
            }
            if (kv.k_dst != nullptr) {
                switch (fwht_n) {
                    case  64: has_ff ? launch(rms_norm_mul_rope_f32<256, true, D,  64, true>) : launch(rms_norm_mul_rope_f32<256, false, D,  64, true>); break;
                    case 128: has_ff ? launch(rms_norm_mul_rope_f32<256, true, D, 128, true>) : launch(rms_norm_mul_rope_f32<256, false, D, 128, true>); break;
                    case 256: has_ff ? launch(rms_norm_mul_rope_f32<256, true, D, 256, true>) : launch(rms_norm_mul_rope_f32<256, false, D, 256, true>); break;
                    default: GGML_ABORT("fatal error");
                }
            } else {
                switch (fwht_n) {
                    case  64: has_ff ? launch(rms_norm_mul_rope_f32<256, true, D,  64>) : launch(rms_norm_mul_rope_f32<256, false, D,  64>); break;
                    case 128: has_ff ? launch(rms_norm_mul_rope_f32<256, true, D, 128>) : launch(rms_norm_mul_rope_f32<256, false, D, 128>); break;
                    case 256: has_ff ? launch(rms_norm_mul_rope_f32<256, true, D, 256>) : launch(rms_norm_mul_rope_f32<256, false, D, 256>); break;
                    default: GGML_ABORT("fatal error");
                }
            }
            return;
        }
    } else {
        GGML_ASSERT(fwht_n == 0 && kv.k_dst == nullptr);
    }
    GGML_ASSERT(!no_norm);

    if (ncols < 1024) {
        const dim3 block_dims(256, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = {blocks_num, block_dims, 32*sizeof(float), stream};
        if (freq_factors == nullptr) {
            ggml_cuda_kernel_launch(rms_norm_mul_rope_f32<256, false, D>, launch_params,
                x, dst, ncols, nchannels, nsamples, s01, s02, s03, s1, s2, s3, eps, mul, mul_s01, mul_s02, mul_s03,
                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale,
                freq_factors, row_indices, set_rows_stride, is_neox, fwht_scale, kv, sections, mrope);
        } else {
            ggml_cuda_kernel_launch(rms_norm_mul_rope_f32<256, true, D>, launch_params,
                x, dst, ncols, nchannels, nsamples, s01, s02, s03, s1, s2, s3, eps, mul, mul_s01, mul_s02, mul_s03,
                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale,
                freq_factors, row_indices, set_rows_stride, is_neox, fwht_scale, kv, sections, mrope);
        }
    } else {
        const dim3 block_dims(1024, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = {blocks_num, block_dims, 32*sizeof(float), stream};
        if (freq_factors == nullptr) {
            ggml_cuda_kernel_launch(rms_norm_mul_rope_f32<1024, false, D>, launch_params,
                x, dst, ncols, nchannels, nsamples, s01, s02, s03, s1, s2, s3, eps, mul, mul_s01, mul_s02, mul_s03,
                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale,
                freq_factors, row_indices, set_rows_stride, is_neox, fwht_scale, kv, sections, mrope);
        } else {
            ggml_cuda_kernel_launch(rms_norm_mul_rope_f32<1024, true, D>, launch_params,
                x, dst, ncols, nchannels, nsamples, s01, s02, s03, s1, s2, s3, eps, mul, mul_s01, mul_s02, mul_s03,
                mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims, theta_scale,
                freq_factors, row_indices, set_rows_stride, is_neox, fwht_scale, kv, sections, mrope);
        }
    }
}

void ggml_cuda_op_rms_norm_mul_rope_fused(ggml_backend_cuda_context & ctx,
        ggml_tensor * rms_norm, ggml_tensor * mul, ggml_tensor * rope, ggml_tensor * set_rows, ggml_tensor * fwht,
        ggml_tensor * k_set_rows, ggml_tensor * v_set_rows, void * capture, ggml_tensor * v_fwht) {
    // rms_norm == nullptr: ROPE alone (the rope + FWHT forms); x is the ROPE's input, the MUL operand a dummy
    const bool no_norm = rms_norm == nullptr;
    GGML_ASSERT(!no_norm || (fwht != nullptr && capture == nullptr));
    const ggml_tensor * x = no_norm ? rope->src[0] : rms_norm->src[0];
    const ggml_tensor * mul_src = no_norm ? x : (mul->src[0] == rms_norm ? mul->src[1] : mul->src[0]);

    float eps = 0.0f;
    if (!no_norm) {
        memcpy(&eps, rms_norm->op_params, sizeof(float));
    }
    GGML_ASSERT(eps >= 0.0f);

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_src->type == GGML_TYPE_F32);
    GGML_ASSERT(rope->type == GGML_TYPE_F32);

    void *          dst_d           = rope->data;
    ggml_type       dst_type        = rope->type;
    const int64_t * row_indices     = nullptr;
    int             set_rows_stride = 0;

    int   fwht_n     = 0;
    float fwht_scale = 0.0f;

    if (fwht != nullptr) {
        // the KV rotation's Hadamard MUL_MAT (same shape, contiguous): write its rotated rows instead
        GGML_ASSERT(set_rows == nullptr && fwht->type == GGML_TYPE_F32 && ggml_is_contiguous(rope) && ggml_is_contiguous(fwht));
        dst_d      = fwht->data;
        fwht_n     = fwht->src[0]->ne[0];
        fwht_scale = 1 / sqrtf(fwht_n);   // as in ggml_cuda_op_fwht_impl
    }

    rope_kv_q8_0 kv = {};
    if (k_set_rows != nullptr) {
        // q8_0 SET_ROWS of the rotated K row (and of the V row) instead of the F32 rotation output; the matcher
        // checked the shapes (one contiguous [nrows*ncols] row per token, I64 indices, q8_0 cache views)
        GGML_ASSERT(fwht != nullptr && k_set_rows->type == GGML_TYPE_Q8_0);
        kv.k_dst = (block_q8_0 *) k_set_rows->data;
        kv.k_idx = (const int64_t *) k_set_rows->src[1]->data;
        kv.k_nb1 = k_set_rows->nb[1];
        if (v_set_rows != nullptr) {
            GGML_ASSERT(v_set_rows->type == GGML_TYPE_Q8_0 && v_set_rows->src[0]->type == GGML_TYPE_F32);
            kv.v_src = (const float *) v_set_rows->src[0]->data;
            kv.v_s1  = v_set_rows->src[0]->nb[1] / sizeof(float);
            kv.v_dst = (block_q8_0 *) v_set_rows->data;
            kv.v_idx = (const int64_t *) v_set_rows->src[1]->data;
            kv.v_nb1 = v_set_rows->nb[1];
            if (v_fwht != nullptr) {
                // V's Hadamard MUL_MAT folded in: read its unrotated input (same contiguous layout as the rotated rows
                // the SET_ROWS view reads) and rotate in the kernel with fwht_cuda's scale
                GGML_ASSERT(v_set_rows->src[0]->view_src == v_fwht && v_fwht->src[0]->ne[0] == 64 && v_fwht->src[1]->type == GGML_TYPE_F32 &&
                            ggml_is_contiguous(v_fwht->src[1]) && ggml_are_same_shape(v_fwht->src[1], v_fwht));
                kv.v_src        = (const float *) ((const char *) v_fwht->src[1]->data + v_set_rows->src[0]->view_offs); // the VIEW's offset into the rotated rows (its view_src is v_fwht)
                kv.v_fwht_n     = 64;
                kv.v_fwht_scale = 1 / sqrtf(64);   // as in ggml_cuda_op_fwht_impl
            }
        }
    }

    if (k_set_rows == nullptr && v_set_rows != nullptr) {
        // the V write next to the plain K SET_ROWS (ggml_cuda_rope_set_rows_v_count checked the shapes and types)
        GGML_ASSERT(fwht == nullptr && set_rows != nullptr && v_set_rows->type == set_rows->type &&
                    v_set_rows->src[0]->type == GGML_TYPE_F32);
        kv.vp_src = (const float *) v_set_rows->src[0]->data;
        kv.vp_s1  = v_set_rows->src[0]->nb[1] / sizeof(float);
        kv.vp_dst = v_set_rows->data;
        kv.vp_idx = (const int64_t *) v_set_rows->src[1]->data;
        kv.vp_nb1 = v_set_rows->nb[1];
    }
    if (set_rows != nullptr) {
        dst_d           = set_rows->data;
        dst_type        = set_rows->type;
        row_indices     = (const int64_t *) set_rows->src[1]->data;
        set_rows_stride = set_rows->nb[1] / ggml_type_size(set_rows->type);
    }

    const int n_dims     = ((const int32_t *) rope->op_params)[1];
    const int mode       = ((const int32_t *) rope->op_params)[2];
    const int n_ctx_orig = ((const int32_t *) rope->op_params)[4];

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;

    memcpy(&freq_base,   (const int32_t *) rope->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (const int32_t *) rope->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (const int32_t *) rope->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (const int32_t *) rope->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (const int32_t *) rope->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (const int32_t *) rope->op_params + 10, sizeof(float));

    const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;

    const int32_t * pos = (const int32_t *) rope->src[1]->data;

    const float * freq_factors = rope->src[2] != nullptr ? (const float *) rope->src[2]->data : nullptr;

    // M-RoPE / IMROPE (the matcher admits them, never VISION): rope_multi's section-selected positions
    mrope_sections sections;
    memcpy(&sections.v, (const int32_t *) rope->op_params + 11, sizeof(int)*4);
    const int mrope = mode == GGML_ROPE_TYPE_IMROPE ? 2 : mode == GGML_ROPE_TYPE_MROPE ? 1 : 0;

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims.v);

    const size_t ts0 = ggml_type_size(x->type);
    GGML_ASSERT(x->nb[0] == ts0);
    const int64_t s01 = x->nb[1] / ts0;
    const int64_t s02 = x->nb[2] / ts0;
    const int64_t s03 = x->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const size_t ts_dst = ggml_type_size(rope->type);
    const int64_t s1 = rope->nb[1] / ts_dst;
    const int64_t s2 = rope->nb[2] / ts_dst;
    const int64_t s3 = rope->nb[3] / ts_dst;

    cudaStream_t stream = ctx.stream();

    if (dst_type == GGML_TYPE_F32) {
        rms_norm_mul_rope_cuda<float>((const float *) x->data, (float *) dst_d,
            x->ne[0], x->ne[1], x->ne[2], x->ne[3], s01, s02, s03, s1, s2, s3, eps,
            (const float *) mul_src->data, mul_s01, mul_s02, mul_s03,
            mul_src->ne[0], mul_src->ne[1], mul_src->ne[2], mul_src->ne[3],
            n_dims, pos, freq_scale, freq_base, ext_factor, attn_factor, corr_dims,
            freq_factors, row_indices, set_rows_stride, is_neox, fwht_n, fwht_scale, kv, sections, mrope, stream, (rmr_args<float> *) capture,
            no_norm);
    } else if (dst_type == GGML_TYPE_F16) {
        GGML_ASSERT(!no_norm);
        rms_norm_mul_rope_cuda<half>((const float *) x->data, (half *) dst_d,
            x->ne[0], x->ne[1], x->ne[2], x->ne[3], s01, s02, s03, s1, s2, s3, eps,
            (const float *) mul_src->data, mul_s01, mul_s02, mul_s03,
            mul_src->ne[0], mul_src->ne[1], mul_src->ne[2], mul_src->ne[3],
            n_dims, pos, freq_scale, freq_base, ext_factor, attn_factor, corr_dims,
            freq_factors, row_indices, set_rows_stride, is_neox, fwht_n, fwht_scale, kv, sections, mrope, stream, (rmr_args<half> *) capture);
    } else {
        GGML_ABORT("fatal error");
    }
}

void ggml_cuda_op_rms_norm_mul_rope_qk_pair_fwht(ggml_backend_cuda_context & ctx,
        ggml_tensor * q_norm, ggml_tensor * q_mul, ggml_tensor * q_rope, ggml_tensor * q_had,
        ggml_tensor * k_norm, ggml_tensor * k_mul, ggml_tensor * k_rope, ggml_tensor * k_had,
        ggml_tensor * k_set_rows, ggml_tensor * v_set_rows, ggml_tensor * v_had) {
    rmr_args<float> q;
    rmr_args<float> k;
    ggml_cuda_op_rms_norm_mul_rope_fused(ctx, q_norm, q_mul, q_rope, nullptr, q_had, nullptr, nullptr, &q);
    ggml_cuda_op_rms_norm_mul_rope_fused(ctx, k_norm, k_mul, k_rope, nullptr, k_had, k_set_rows, v_set_rows, &k, v_had);
    const bool has_ff = q.freq_factors != nullptr;
    GGML_ASSERT(q.fwht_n == k.fwht_n && q.kv.k_dst == nullptr && k.kv.k_dst != nullptr &&
                k.nchannels == q.nchannels && k.nsamples == q.nsamples && (k.freq_factors != nullptr) == has_ff);
    const dim3 blocks_num(q.nrows + k.nrows, MIN(q.nchannels, UINT16_MAX), MIN(q.nsamples, UINT16_MAX));
    const ggml_cuda_kernel_launch_params launch_params = {blocks_num, dim3(256, 1, 1), 32*sizeof(float), ctx.stream()};
    const auto launch = [&](auto kernel) {
        ggml_cuda_kernel_launch(kernel, launch_params, q, k);
    };
    switch (q.fwht_n) {
        case  64: has_ff ? launch(rms_norm_mul_rope_f32_qk_pair<true, float, float,  64, true>) : launch(rms_norm_mul_rope_f32_qk_pair<false, float, float,  64, true>); break;
        case 128: has_ff ? launch(rms_norm_mul_rope_f32_qk_pair<true, float, float, 128, true>) : launch(rms_norm_mul_rope_f32_qk_pair<false, float, float, 128, true>); break;
        case 256: has_ff ? launch(rms_norm_mul_rope_f32_qk_pair<true, float, float, 256, true>) : launch(rms_norm_mul_rope_f32_qk_pair<false, float, float, 256, true>); break;
        default: GGML_ABORT("fatal error");
    }
}

// Q norm+rope (F32 output) and the K norm+rope + cache SET_ROWS (+ V row) of the same layer in one launch.
void ggml_cuda_op_rms_norm_mul_rope_qk_pair(ggml_backend_cuda_context & ctx,
        ggml_tensor * q_norm, ggml_tensor * q_mul, ggml_tensor * q_rope,
        ggml_tensor * k_norm, ggml_tensor * k_mul, ggml_tensor * k_rope, ggml_tensor * k_set_rows, ggml_tensor * v_set_rows) {
    GGML_ASSERT(q_rope->type == GGML_TYPE_F32 && (k_set_rows->type == GGML_TYPE_F16 || k_set_rows->type == GGML_TYPE_F32));
    rmr_args<float> q;
    ggml_cuda_op_rms_norm_mul_rope_fused(ctx, q_norm, q_mul, q_rope, nullptr, nullptr, nullptr, nullptr, &q);
    const bool has_ff = q.freq_factors != nullptr;
    const auto launch = [&](auto k_arg) {
        using DK = std::remove_pointer_t<decltype(k_arg.dst)>;
        GGML_ASSERT(k_arg.nchannels == q.nchannels && k_arg.nsamples == q.nsamples && (k_arg.freq_factors != nullptr) == has_ff);
        const dim3 blocks_num(q.nrows + k_arg.nrows, MIN(q.nchannels, UINT16_MAX), MIN(q.nsamples, UINT16_MAX));
        const ggml_cuda_kernel_launch_params launch_params = {blocks_num, dim3(256, 1, 1), 32*sizeof(float), ctx.stream()};
        if (has_ff) {
            ggml_cuda_kernel_launch(rms_norm_mul_rope_f32_qk_pair<true, float, DK>, launch_params, q, k_arg);
        } else {
            ggml_cuda_kernel_launch(rms_norm_mul_rope_f32_qk_pair<false, float, DK>, launch_params, q, k_arg);
        }
    };
    if (k_set_rows->type == GGML_TYPE_F16) {
        rmr_args<half> k;
        ggml_cuda_op_rms_norm_mul_rope_fused(ctx, k_norm, k_mul, k_rope, k_set_rows, nullptr, nullptr, v_set_rows, &k);
        launch(k);
    } else {
        rmr_args<float> k;
        ggml_cuda_op_rms_norm_mul_rope_fused(ctx, k_norm, k_mul, k_rope, k_set_rows, nullptr, nullptr, v_set_rows, &k);
        launch(k);
    }
}
