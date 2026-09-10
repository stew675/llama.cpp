#include "cpy-batch.cuh"

namespace {

constexpr int CB_MAX = 16;

bool cb_on() {
    static const bool on = getenv("GGML_CUDA_FUSE_CPY_BATCH") == nullptr || atoi(getenv("GGML_CUDA_FUSE_CPY_BATCH")) != 0; // default on
    return on;
}

struct cb_args {
    const char * src[CB_MAX];
    char *       dst[CB_MAX];
    int64_t      sne[4], snb[4];   // source shape / byte strides (shared by the run)
    int64_t      dne[4], dnb[4];   // destination shape / byte strides (shared by the run)
};

// blockIdx.y = copy; element i in ggml_cpy's flat order, located separately in the source and the destination layouts
__global__ void cb_kernel(const cb_args a, const int64_t n) {
    const int c = blockIdx.y;
    const char * __restrict__ s = a.src[c];
    char * __restrict__ d = a.dst[c];
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        int64_t r = i;
        const int64_t s0 = r % a.sne[0]; r /= a.sne[0];
        const int64_t s1 = r % a.sne[1]; r /= a.sne[1];
        const int64_t s2 = r % a.sne[2]; const int64_t s3 = r / a.sne[2];
        r = i;
        const int64_t d0 = r % a.dne[0]; r /= a.dne[0];
        const int64_t d1 = r % a.dne[1]; r /= a.dne[1];
        const int64_t d2 = r % a.dne[2]; const int64_t d3 = r / a.dne[2];
        *(float *) (d + d0 * a.dnb[0] + d1 * a.dnb[1] + d2 * a.dnb[2] + d3 * a.dnb[3]) =
            *(const float *) (s + s0 * a.snb[0] + s1 * a.snb[1] + s2 * a.snb[2] + s3 * a.snb[3]);
    }
}

bool cb_member(const ggml_tensor * n) {
    if (n->op != GGML_OP_CPY) return false;
    const ggml_tensor * s = n->src[0], * d = n->src[1];
    return s && d && s->type == GGML_TYPE_F32 && d->type == GGML_TYPE_F32 && ggml_nelements(s) == ggml_nelements(d);
}

// nodes that compute nothing (the views a copy reads / writes through sit between the copies in the node list)
bool cb_noop(const ggml_tensor * n) {
    return n->op == GGML_OP_VIEW || n->op == GGML_OP_RESHAPE || n->op == GGML_OP_PERMUTE || n->op == GGML_OP_TRANSPOSE ||
           n->op == GGML_OP_NONE;
}

bool cb_same(const ggml_tensor * a, const ggml_tensor * b) {
    for (int k = 0; k < 4; ++k) if (a->ne[k] != b->ne[k] || a->nb[k] != b->nb[k]) return false;
    return true;
}

// byte span touched by a (possibly strided) view
void cb_span(const ggml_tensor * t, uintptr_t & lo, uintptr_t & hi) {
    lo = (uintptr_t) t->data;
    size_t last = 0;
    for (int k = 0; k < 4; ++k) last += (size_t) (t->ne[k] - 1) * t->nb[k];
    hi = lo + last + ggml_type_size(t->type);
}

} // namespace

int ggml_cuda_cpy_batch(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, const int i) {
    if (!cb_on() || !GGML_CUDA_CC_IS_RDNA4(ggml_cuda_info().devices[ctx.device].cc)) return 0;
    ggml_tensor * first = cgraph->nodes[i];
    if (!cb_member(first)) return 0;
    // the run: copies with the first one's layouts, possibly separated by no-op view nodes (consumed with the run)
    int idx[CB_MAX] = {i}, n = 1, last = i;
    for (int j = i + 1; j < cgraph->n_nodes && n < CB_MAX; ++j) {
        const ggml_tensor * b = cgraph->nodes[j];
        if (cb_noop(b)) continue;
        if (!cb_member(b) || !cb_same(b->src[0], first->src[0]) || !cb_same(b->src[1], first->src[1])) break;
        idx[n++] = j; last = j;
    }
    if (n < 2) return 0;
    // the copies of a run must be independent: no destination may touch another copy's source or destination
    for (int a = 0; a < n; ++a) {
        uintptr_t dlo, dhi;
        cb_span(cgraph->nodes[idx[a]]->src[1], dlo, dhi);
        for (int b = 0; b < n; ++b) {
            uintptr_t lo, hi;
            cb_span(cgraph->nodes[idx[b]]->src[0], lo, hi);
            if (dlo < hi && lo < dhi) return 0;
            if (b != a) {
                cb_span(cgraph->nodes[idx[b]]->src[1], lo, hi);
                if (dlo < hi && lo < dhi) return 0;
            }
        }
    }
    cb_args args{};
    for (int k = 0; k < 4; ++k) {
        args.sne[k] = first->src[0]->ne[k]; args.snb[k] = first->src[0]->nb[k];
        args.dne[k] = first->src[1]->ne[k]; args.dnb[k] = first->src[1]->nb[k];
    }
    for (int c = 0; c < n; ++c) {
        args.src[c] = (const char *) cgraph->nodes[idx[c]]->src[0]->data;
        args.dst[c] = (char *) cgraph->nodes[idx[c]]->src[1]->data;
    }
    const int64_t ne = ggml_nelements(first->src[0]);
    const int blocks = (int) std::min<int64_t>(256, (ne + 255) / 256);
    cb_kernel<<<dim3(blocks, n), 256, 0, ctx.stream()>>>(args, ne);
    CUDA_CHECK(cudaGetLastError());
    static const bool dbg = getenv("GGML_CUDA_FUSE_CPY_BATCH_DEBUG") != nullptr;
    if (dbg) fprintf(stderr, "cpy_batch: %d copies over nodes %d..%d (%lld elements each)\n", n, i, last, (long long) ne);
    return last - i + 1;
}
