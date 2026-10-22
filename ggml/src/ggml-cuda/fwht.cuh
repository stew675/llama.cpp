#include "common.cuh"
#include "convert.cuh"

// In-warp FWHT of one N-element row: loads reg[i] = src[i*warp_size + lane] * scale and transforms it in
// place.  Shared by fwht_cuda and the fused rms_norm_mul_rope + FWHT kernel, and kept in this header so
// both compile it under the same FP-contraction mode (rope.cu turns contraction off after its includes);
// the two then produce the same bits.
template <int N, int warp_size, typename T>
static __device__ __forceinline__ void ggml_cuda_fwht_warp(const T * src, const float scale, float (&reg)[N / warp_size], const int lane) {
    static constexpr int el_w = N / warp_size;

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        reg[i] = ggml_cuda_cast<float>(src[i * warp_size + lane]) * scale;
    }

#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; j++) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

#pragma unroll
    for (int h = warp_size; h < N; h *= 2) {
        const int step = h / warp_size;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }
}

bool ggml_cuda_op_mul_mat_use_fwht(const struct ggml_tensor * op);

// Returns whether the Fast Walsh-Hadamard transform could be used.  q8_1: also write dst as Q8_1 blocks there
// (rows of up to 512 elements; dst rows padded to MATRIX_ROW_PADDING are not handled).
bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst, void * q8_1 = nullptr);
