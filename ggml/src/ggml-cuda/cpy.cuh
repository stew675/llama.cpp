#include "common.cuh"

#define CUDA_CPY_BLOCK_SIZE 64

void ggml_cuda_cpy(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, ggml_tensor * src1);

void ggml_cuda_dup(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// A strided row block copy between two DEVICE-side ranges: `height` rows of `width` bytes from
// `src + r*spitch` to `dst + r*dpitch`.  Vectorised (one thread per 16-byte unit, the row from the grid)
// when alignment allows, a byte-wise kernel otherwise.  It exists because a 2-D copy that touches a
// region-backed VA (the windowed KV) is refused by the driver on some stacks (issue #313: `-sm tensor`
// prompt-cache save, `ROCm error: invalid argument`) while 1-D copies and kernels are accepted, so a
// region's row block is gathered/scattered on the device instead of issued as one host-side 1-D copy per
// row (measured: 5 246 rows of 544 B that way = 1.48 s for one prompt-cache save).
void ggml_cuda_cpy_2d_rows(char * dst, const char * src, size_t width, size_t height,
                           size_t spitch, size_t dpitch, cudaStream_t stream);
