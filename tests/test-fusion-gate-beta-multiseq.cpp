// Multi-sequence regression test for the GDN gate/beta decode fusions (issue #204).
//
// The GPU gate/beta fusions (GGML_CUDA_FUSE_GATE_BETA_BF16, mmvf.cu, and its Q8_0 sibling
// GGML_CUDA_FUSE_GATE_BETA_VERIFY, mmvq.cu) replace the unfused chain
//
//     alpha = MUL_MAT(alpha_w, y);  a3 = RESHAPE3D(alpha, nr, T, S);  bias = a3 + dt;
//     sp = SOFTPLUS(bias);  gate = sp * ssm_a;  gate4 = RESHAPE4D(gate, 1, nr, T, S)
//     beta = MUL_MAT(beta_w, y);    beta4 = RESHAPE4D(beta, 1, nr, T, S)
//
// with one mmvf/mmvq launch that writes the alpha rows (through the gate chain) and the beta rows.
// The launch derives its geometry from the dst tensor, so it is only correct if the shapes it is
// given describe the *flat* projection: y is [k, n_tokens] with n_tokens = T*S (one channel), the
// gate MUL is [nr, T, S].  Passing the gate MUL as dst (as the r102..r107 code did) gave the launch
// n_seq_tokens columns and n_seqs channels, so every channel past the first was computed from y's
// first n_seq_tokens columns - i.e. every sequence but the first silently took sequence 0's
// alpha/beta gates.  Single-sequence decode/verify was unaffected, which is why the same-seed text
// and MTP gates never saw it.
//
// What this test checks, per (T, S) shape, with no model and no ubatch scheduler involved:
//
//   1. the fused chain and the unfused chain produce bit-identical gate and beta values
//      (the unfused arm is the same graph plus one extra consumer of the ADD output, which makes
//      ggml_can_fuse_subgraph reject the window - the intermediate is used outside it - so the two
//      arms differ *only* in whether the fusion fires);
//   2. the fusion really did fire in the fused arm: the fused launch skips the ADD/UNARY/MUL
//      nodes, so the softplus output is never written.  It is pre-filled with NaN and must still
//      be NaN afterwards.  Without this check a regression that made the fusion silently stop
//      matching would make the test pass vacuously.
//
// At least one shape in the matrix must fire on the running device, otherwise the test fails
// ("nothing exercised the fusion"): the fusions are band-limited (BF16: n_tokens <= 3 on a device
// with bf16 MMA where MMF is applicable, <= 8 otherwise), so on some devices only the small shapes
// fire.  (T, S) = (1, 2) and (1, 3) are the multi-sequence shapes that regressed in #204.
//
// usage: test-fusion-gate-beta-multiseq [device-index]
// exit:  0 all cases bit-identical and at least one fused; 2 otherwise; 0 with a SKIP line when
//        the build has no CUDA/ROCm/MUSA backend device (the fusions do not exist elsewhere).
//
// The sibling Q8_0 fusion (GGML_CUDA_FUSE_GATE_BETA_VERIFY) is covered through its own switch; the
// gate/beta arms in gdn-conv.* / unary.* are unaffected by the dst-shape rule (they write their own
// outputs) and are covered by the same-seed text gates.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static bool is_cuda_family(const char * name) {
    return strncmp(name, "CUDA", 4) == 0 || strncmp(name, "ROCm", 4) == 0 || strncmp(name, "MUSA", 4) == 0;
}

struct case_t {
    int64_t T;   // n_seq_tokens
    int64_t S;   // n_seqs
};

struct graph_t {
    ggml_context * ctx = nullptr;
    ggml_cgraph  * gf  = nullptr;
    ggml_tensor  * gate    = nullptr;   // [nr, T, S]  -- what the fused launch writes
    ggml_tensor  * beta    = nullptr;   // [nr, T*S]   -- what the fused launch's beta rows write
    ggml_tensor  * bias    = nullptr;
    ggml_tensor  * softplus= nullptr;   // fused arm: never written
    ggml_tensor  * alpha_w = nullptr;
    ggml_tensor  * beta_w  = nullptr;
    ggml_tensor  * y       = nullptr;
    ggml_tensor  * dt = nullptr;
    ggml_tensor  * ssm_a   = nullptr;
};

// build the chain; `unfused` adds one extra consumer of the ADD output, which makes the fusion
// window ineligible (the intermediate would be read outside the window) while leaving the
// arithmetic identical.
static graph_t build_graph(int64_t nr, int64_t k, int64_t T, int64_t S, bool unfused) {
    graph_t g;
    // no_alloc: the tensors are allocated below, into one backend buffer
ggml_init_params ip = { /*.mem_size =*/ 16 * 1024 * 1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
    g.ctx = ggml_init(ip);
    if (!g.ctx) { fprintf(stderr, "ggml_init failed\n"); exit(1); }

    g.alpha_w = ggml_new_tensor_2d(g.ctx, GGML_TYPE_BF16, k, nr);
    g.beta_w  = ggml_new_tensor_2d(g.ctx, GGML_TYPE_BF16, k, nr);
    g.y       = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32,  k, T * S);
    g.ssm_a   = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32,  nr);

    g.dt = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, nr);

    ggml_tensor * alpha   = ggml_mul_mat(g.ctx, g.alpha_w, g.y);        // [nr, T*S]
    ggml_tensor * alpha3  = ggml_reshape_3d(g.ctx, alpha, nr, T, S);
    ggml_tensor * bias    = ggml_add(g.ctx, alpha3, g.dt);
    g.bias = bias;
    g.softplus            = ggml_softplus(g.ctx, bias);
    g.gate                = ggml_mul(g.ctx, g.softplus, g.ssm_a);       // [nr, T, S]
    ggml_tensor * gate4   = ggml_reshape_4d(g.ctx, g.gate, 1, nr, T, S);
    g.beta                = ggml_mul_mat(g.ctx, g.beta_w, g.y);         // [nr, T*S]
    ggml_tensor * beta4   = ggml_reshape_4d(g.ctx, g.beta, 1, nr, T, S);

    g.gf = ggml_new_graph(g.ctx);
    // this order is what the matcher sees: alpha chain, then the beta projection (the real graphs
    // create beta first but the executed order is the dependency order below)
    ggml_build_forward_expand(g.gf, gate4);
    ggml_build_forward_expand(g.gf, beta4);
    if (unfused) {
        // one extra consumer of the ADD output, outside the fusion window: subgraph_uses (1) no
        // longer matches the graph-wide use count (2), so the matcher declines and the plain
        // chain runs
        ggml_build_forward_expand(g.gf, ggml_sum(g.ctx, bias));
    }

    return g;
}

struct result_t {
    bool            fused = false;
    std::vector<float> gate;
    std::vector<float> beta;
};

static result_t run_graph(graph_t & g, ggml_backend_t backend,
                          const std::vector<ggml_bf16_t> & wa,
                          const std::vector<ggml_bf16_t> & wb,
                          const std::vector<float> & y,
                          const std::vector<float> & dt,
                          const std::vector<float> & ssm_a) {
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(g.ctx, backend);
    if (!buf) { fprintf(stderr, "alloc_ctx_tensors failed\n"); exit(1); }

    ggml_backend_tensor_set(g.alpha_w, wa.data(), 0, wa.size() * sizeof(ggml_bf16_t));
    ggml_backend_tensor_set(g.beta_w,  wb.data(), 0, wb.size() * sizeof(ggml_bf16_t));
    ggml_backend_tensor_set(g.y,       y.data(),  0, y.size()  * sizeof(float));
    ggml_backend_tensor_set(g.ssm_a, ssm_a.data(), 0, ssm_a.size() * sizeof(float));
    ggml_backend_tensor_set(g.dt, dt.data(), 0, dt.size() * sizeof(float));

    // NaN sentinel: the fused launch skips ADD/UNARY/MUL, so softplus stays untouched
    {
        std::vector<float> nan(ggml_nelements(g.softplus), NAN);
        ggml_backend_tensor_set(g.bias,     nan.data(), 0, nan.size() * sizeof(float));
        ggml_backend_tensor_set(g.softplus, nan.data(), 0, nan.size() * sizeof(float));
    }

    if (ggml_backend_graph_compute(backend, g.gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed\n");
        exit(1);
    }

    result_t r;
    std::vector<float> sp(ggml_nelements(g.softplus));
    std::vector<float> bi(ggml_nelements(g.bias));
    ggml_backend_tensor_get(g.softplus, sp.data(), 0, sp.size() * sizeof(float));
    ggml_backend_tensor_get(g.bias,     bi.data(), 0, bi.size() * sizeof(float));
    r.fused = true;
    for (float v : sp) {
        if (!std::isnan(v)) { r.fused = false; break; }
    }

    r.gate.resize(ggml_nelements(g.gate));
    r.beta.resize(ggml_nelements(g.beta));
    ggml_backend_tensor_get(g.gate, r.gate.data(), 0, r.gate.size() * sizeof(float));
    ggml_backend_tensor_get(g.beta, r.beta.data(), 0, r.beta.size() * sizeof(float));

    ggml_backend_buffer_free(buf);
    return r;
}

int main(int argc, char ** argv) {
    const int want_dev = argc > 1 ? atoi(argv[1]) : -1;

    ggml_backend_load_all();

    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_GPU) { continue; }
        if (!is_cuda_family(ggml_backend_dev_name(d))) { continue; }
        if (want_dev >= 0 && (int) i != want_dev)      { continue; }
        dev = d;
        break;
    }
    if (!dev) {
        printf("SKIP: no CUDA/ROCm/MUSA backend device (the gate/beta fusions only exist there)\n");
        return 0;
    }
    const char * dev_name = ggml_backend_dev_name(dev);
    printf("device: %s\n", dev_name);

    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) { fprintf(stderr, "backend init failed\n"); return 1; }

    const int64_t nr = 48;      // n_v_heads, as in the GDN models the fusion fires on
    const int64_t k  = 1024;    // contraction length (n_embd of the smallest such models is larger)
    const case_t cases[] = {
        { 1, 1 }, { 2, 1 }, { 3, 1 }, { 4, 1 },
        { 1, 2 }, { 1, 3 }, { 1, 4 },            // the multi-sequence shapes (#204)
        { 2, 2 }, { 3, 2 }, { 2, 3 }, { 2, 4 },
    };

    std::mt19937 gen(42);
    std::normal_distribution<float> dis(0.0f, 0.1f);

    std::vector<ggml_bf16_t> wa(nr * k), wb(nr * k);
    for (int64_t i = 0; i < nr * k; ++i) {
        wa[i] = ggml_fp32_to_bf16(dis(gen));
        wb[i] = ggml_fp32_to_bf16(dis(gen));
    }
    std::vector<float> ssm_a(nr);
    for (int64_t i = 0; i < nr; ++i) {
        ssm_a[i] = -fabsf(dis(gen));   // -A_log.exp(), negative as in the models
    }

    int n_fired = 0;
    int n_failed = 0;

    for (const case_t & c : cases) {
        std::vector<float> y(k * c.T * c.S);
        std::vector<float> dt(nr);
        for (auto & v : y)  { v = dis(gen); }
        for (auto & v : dt) { v = dis(gen); }

        graph_t ga = build_graph(nr, k, c.T, c.S, /*unfused =*/ false);
        graph_t gb = build_graph(nr, k, c.T, c.S, /*unfused =*/ true);

        const result_t a = run_graph(ga, backend, wa, wb, y, dt, ssm_a);
        const result_t b = run_graph(gb, backend, wa, wb, y, dt, ssm_a);

        double max_diff = 0.0;
        size_t n_bad = 0;
        auto compare = [&](const std::vector<float> & x, const std::vector<float> & z) {
            for (size_t i = 0; i < x.size(); ++i) {
                if (memcmp(&x[i], &z[i], sizeof(float)) != 0) {
                    ++n_bad;
                    max_diff = std::max(max_diff, (double) fabsf(x[i] - z[i]));
                }
            }
        };
        compare(a.gate, b.gate);
        compare(a.beta, b.beta);

        // at n_tokens == 1 the ADD->SOFTPLUS->MUL chain also has its own fold (GGML_CUDA_FUSE_GDN_GATE,
        // unary.cu), which skips the softplus node as well, so the sentinel cannot tell the two apart
        // there and the reference arm is not guaranteed to be unfused.  From n_tokens >= 2 that fold is
        // out of band and "softplus untouched" is specific to the gate/beta fusion.
        const bool strict_ref = c.T * c.S > 1;
        const bool ok = (n_bad == 0) && (!strict_ref || !b.fused);
        if (!ok) { ++n_failed; }
        if (a.fused) { ++n_fired; }

        printf("nr=%2lld k=%4lld T=%lld S=%lld  fused=%s  reference_fused=%s  "
               "maxdiff=%.9g  mismatches=%zu  %s\n",
               (long long) nr, (long long) k, (long long) c.T, (long long) c.S,
               a.fused ? "yes" : "no ", b.fused ? "YES" : "no ", max_diff, n_bad,
               ok ? "OK" : "FAIL");

        ggml_free(ga.ctx);
        ggml_free(gb.ctx);
    }

    ggml_backend_free(backend);

    printf("\n%d/%zu shapes fused; %d failed\n", n_fired, sizeof(cases) / sizeof(cases[0]), n_failed);
    if (n_fired == 0) {
        printf("FAIL: no shape exercised the fusion - the test verified nothing on %s\n", dev_name);
        return 2;
    }
    if (n_failed != 0) {
        printf("FAIL: the fused chain does not match the unfused chain (issue #204 class)\n");
        return 2;
    }
    printf("PASS\n");
    return 0;
}
