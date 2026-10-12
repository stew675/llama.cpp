#include "common.cuh"
#include "fattn-tile.cuh"

template <int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_tile_case_type(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    // The tile kernel reads BF16 K/V natively only on hardware with native BF16 support.
    if (bf16_mma_hardware_available(ggml_cuda_info().devices[ggml_cuda_get_device()].cc) &&
            K->type == GGML_TYPE_BF16 && V->type == GGML_TYPE_BF16) {
        // BF16 K/V is read natively and accumulated in FP32.
        ggml_cuda_flash_attn_ext_tile_case<DKQ, DV, GGML_TYPE_BF16>(ctx, dst);
        return;
    }

#ifdef FAST_FP16_AVAILABLE
    // V4 / issue #30 item 2: q8_0, q4_0, q4_1, q5_0, q5_1 and iq4_nl K/V are dequantized while staging
    // the tiles, so no F16 copy of the cache is needed.  The predicate is shared with
    // ggml_cuda_flash_attn_ext_get_alloc_size (which sizes the node's staging scratch) and with the
    // launcher's kv_native_kernel argument, so the three cannot disagree.
    switch (ggml_cuda_fattn_tile_kv_native_type(K, V)) {
        case FATTN_KV_NATIVE_Q8_0:
            ggml_cuda_flash_attn_ext_tile_case<DKQ, DV, GGML_TYPE_Q8_0>(ctx, dst);
            return;
        case FATTN_KV_NATIVE_Q4_0:
            ggml_cuda_flash_attn_ext_tile_case<DKQ, DV, GGML_TYPE_Q4_0>(ctx, dst);
            return;
        case FATTN_KV_NATIVE_Q4_1:
            ggml_cuda_flash_attn_ext_tile_case<DKQ, DV, GGML_TYPE_Q4_1>(ctx, dst);
            return;
        case FATTN_KV_NATIVE_Q5_0:
            ggml_cuda_flash_attn_ext_tile_case<DKQ, DV, GGML_TYPE_Q5_0>(ctx, dst);
            return;
        case FATTN_KV_NATIVE_Q5_1:
            ggml_cuda_flash_attn_ext_tile_case<DKQ, DV, GGML_TYPE_Q5_1>(ctx, dst);
            return;
        case FATTN_KV_NATIVE_IQ4_NL:
            ggml_cuda_flash_attn_ext_tile_case<DKQ, DV, GGML_TYPE_IQ4_NL>(ctx, dst);
            return;
        default:
            break;
    }
#endif // FAST_FP16_AVAILABLE

    // F16, F32, quantized K/V are read as F16; BF16 is converted to F16 by the launcher.
    ggml_cuda_flash_attn_ext_tile_case<DKQ, DV, GGML_TYPE_F16>(ctx, dst);
}

void ggml_cuda_flash_attn_ext_tile(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    switch (K->ne[0]) {
        case  40: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case_type< 40,  40>(ctx, dst);
        } break;
        case  64: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case_type< 64,  64>(ctx, dst);
        } break;
        case  72: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case_type< 72,  72>(ctx, dst);
        } break;
        case  80: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case_type< 80,  80>(ctx, dst);
        } break;
        case  96: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case_type< 96,  96>(ctx, dst);
        } break;
        case 112: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case_type<112, 112>(ctx, dst);
        } break;
        case 128: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case_type<128, 128>(ctx, dst);
        } break;
        case 192: {
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_tile_case_type<192, 128>(ctx, dst);
        } break;
        case 256: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case_type<256, 256>(ctx, dst);
        } break;
        case 320: {
            GGML_ASSERT(V->ne[0] == 256);
            ggml_cuda_flash_attn_ext_tile_case_type<320, 256>(ctx, dst);
        } break;
        case 512: {
            GGML_ASSERT(V->ne[0] == K->ne[0]);
            ggml_cuda_flash_attn_ext_tile_case_type<512, 512>(ctx, dst);
        } break;
        case 576: {
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_tile_case_type<576, 512>(ctx, dst);
        } break;
        default: {
            GGML_ABORT("Unsupported head size");
        } break;
    }
}
