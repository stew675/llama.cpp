// Deterministic sweep of the recurrent rollback snapshot machinery for the
// speculative verify band.  For every (n_rs_seq, batch shape, rollback) it:
//   ctx_roll: decode the full batch, partial-rollback r tokens through the
//             snapshot path, then replay the removed tokens one at a time;
//   ctx_ref : decode only the accepted prefix and then the same replay tokens.
// A correct snapshot restore makes the replayed logits match bit-for-bit
// (eps allows backend scheduling noise).
//
// This is the Phase 0 gate for the issue-#30 clamp policy: it must be green for
// every allowed depth (n_rs_seq up to 15) before purity above 7 can be traded.

#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

static llama_context * make_ctx(
        const common_params & params, llama_model * model,
        uint32_t n_rs_seq, uint32_t n_rs_batch, uint32_t n_ubatch,
        uint32_t n_seq_max, uint32_t n_ctx) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max  = n_seq_max;
    cparams.n_rs_seq   = n_rs_seq;
    cparams.n_rs_batch = n_rs_batch;
    cparams.n_ctx      = n_ctx;
    cparams.n_batch    = n_ctx;
    cparams.n_ubatch   = n_ubatch;
    cparams.kv_unified = false;
    return llama_init_from_model(model, cparams);
}

static bool decode_batch(llama_context * ctx, const std::vector<llama_token> & toks,
                         llama_pos pos0, uint32_t n, llama_seq_id seq, bool last_logits) {
    common_batch batch(ctx);
    for (uint32_t i = 0; i < n; ++i) {
        batch.add(toks[pos0 + i], pos0 + i, seq, last_logits && i + 1 == n);
    }
    return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
}

static bool decode_one(llama_context * ctx, llama_token tok, llama_pos pos, llama_seq_id seq) {
    common_batch batch(ctx);
    batch.add(tok, pos, seq, true);
    return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
}

static std::vector<llama_token> make_tokens(
        llama_context * ctx, const llama_vocab * vocab, uint32_t n_tokens, uint32_t n_vocab) {
    std::vector<llama_token> toks;
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
        for (uint32_t i = 0; i < n_tokens; ++i) {
            toks.push_back((llama_token) ((7*i + 3) % n_vocab));
        }
    } else {
        auto toks0 = common_tokenize(ctx, "The quick brown fox jumps over the lazy dog", true);
        if (toks0.empty()) {
            toks0 = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        }
        for (uint32_t i = 0; i < n_tokens; ++i) {
            toks.push_back(toks0[i % toks0.size()] + (llama_token) ((i / toks0.size()) % 3));
        }
    }
    return toks;
}

// returns max logit diff, or -1 on hard error
static double run_case(
        const common_params & params, llama_model * model, llama_context * ctx_tok,
        const llama_vocab * vocab,
        uint32_t n_rs_seq, uint32_t n_rs_batch, uint32_t n_ubatch,
        uint32_t n_tokens, uint32_t n_rollback, bool verbose) {
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> toks = make_tokens(ctx_tok, vocab, n_tokens, (uint32_t) n_vocab);

    llama_context * ctx_roll = make_ctx(params, model, n_rs_seq, n_rs_batch, n_ubatch, 1, 512);
    llama_context * ctx_ref  = make_ctx(params, model, n_rs_seq, n_rs_batch, n_ubatch, 1, 512);
    if (ctx_roll == nullptr || ctx_ref == nullptr) {
        fprintf(stderr, "  [%s] failed to init contexts\n", __func__);
        if (ctx_roll) llama_free(ctx_roll);
        if (ctx_ref)  llama_free(ctx_ref);
        return -1;
    }

    const uint32_t p0 = n_tokens - n_rollback;

    bool ok = true;
    ok = ok && decode_batch(ctx_roll, toks, 0, n_tokens, 0, true);
    ok = ok && decode_batch(ctx_ref,  toks, 0, p0,      0, true);
    if (!ok) {
        fprintf(stderr, "  [prefill] decode failed\n");
        llama_free(ctx_roll); llama_free(ctx_ref);
        return -1;
    }

    const bool rm = llama_memory_seq_rm(llama_get_memory(ctx_roll), 0, (llama_pos) p0, -1);
    if (!rm) {
        fprintf(stderr, "  [rollback] seq_rm refused (n_tokens=%u rollback=%u)\n", n_tokens, n_rollback);
        llama_free(ctx_roll); llama_free(ctx_ref);
        return -1;
    }

    double diff_max = 0.0;
    for (uint32_t i = p0; i < n_tokens; ++i) {
        if (!decode_one(ctx_roll, toks[i], (llama_pos) i, 0) ||
            !decode_one(ctx_ref,  toks[i], (llama_pos) i, 0)) {
            fprintf(stderr, "  [replay] decode failed at pos %u\n", i);
            llama_free(ctx_roll); llama_free(ctx_ref);
            return -1;
        }
        const float * l_roll = llama_get_logits_ith(ctx_roll, 0);
        const float * l_ref  = llama_get_logits_ith(ctx_ref,  0);
        if (l_roll == nullptr || l_ref == nullptr) {
            fprintf(stderr, "  [replay] missing logits at pos %u\n", i);
            llama_free(ctx_roll); llama_free(ctx_ref);
            return -1;
        }
        for (int t = 0; t < n_vocab; ++t) {
            const double d = std::fabs((double) l_roll[t] - (double) l_ref[t]);
            if (d > diff_max) diff_max = d;
        }
    }

    if (verbose) {
        fprintf(stderr, "  n_rs_seq=%2u n_rs_batch=%2u n_ub=%4u n_tokens=%3u rollback=%2u -> max diff %g\n",
                n_rs_seq, n_rs_batch, n_ubatch, n_tokens, n_rollback, diff_max);
    }

    llama_free(ctx_roll);
    llama_free(ctx_ref);
    return diff_max;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.sampling.seed = 1234;
    params.n_predict     = 1;

    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        fprintf(stderr, "failed to init model\n");
        return 1;
    }
    if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
        fprintf(stderr, "skipping for non-recurrent model\n");
        return 0;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    constexpr double eps = 1e-5;
    int n_fail = 0;

    // Phase A: verify-batch shape.  n_tokens = K = n_rs_seq + 1, rollback the
    // whole allowed range.  This is exactly the speculative verify + partial
    // accept path for depth n_max = n_rs_seq.
    fprintf(stderr, "=== Phase A: verify-batch shape (n_tokens = n_rs_seq+1) ===\n");
    for (uint32_t n_rs_seq = 1; n_rs_seq <= 15; ++n_rs_seq) {
        const uint32_t n_rs_batch = n_rs_seq + 1;
        const uint32_t n_tokens   = n_rs_seq + 1;
        int fail_a = 0;
        for (uint32_t r = 1; r <= n_rs_seq; ++r) {
            const double d = run_case(params, model, llama_init->context(), vocab, n_rs_seq, n_rs_batch, 32, n_tokens, r, false);
            if (d < 0 || d > eps) {
                fprintf(stderr, "FAIL A: n_rs_seq=%u rollback=%u max diff=%g\n", n_rs_seq, r, d);
                ++fail_a;
            }
        }
        n_fail += fail_a;
        fprintf(stderr, "  n_rs_seq=%2u verify shape: %s\n", n_rs_seq, fail_a ? "FAIL" : "PASS");
    }

    // Phase B: deeper-than-K batch still at/below the draft bound.  A batch with
    // n_tokens > K can be rolled back into when the speculator drafts deeply
    // (n_rs_batch > K).  n_tokens = n_rs_batch, rollback the full draft range.
    fprintf(stderr, "=== Phase B: deep draft (n_tokens = n_rs_batch > K) ===\n");
    for (uint32_t n_rs_seq = 7; n_rs_seq <= 15; n_rs_seq += 4) {
        const uint32_t K = n_rs_seq + 1;
        for (uint32_t extra = 1; extra <= 16; extra += 5) {
            const uint32_t n_rs_batch = K + extra;
            const uint32_t n_tokens   = n_rs_batch;
            int fail_b = 0;
            for (uint32_t r = 1; r <= n_rs_seq; ++r) {
                const double d = run_case(params, model, llama_init->context(), vocab, n_rs_seq, n_rs_batch, 32, n_tokens, r, false);
                if (d < 0 || d > eps) {
                    fprintf(stderr, "FAIL B: n_rs_seq=%u n_rs_batch=%u n_tokens=%u rollback=%u max diff=%g\n",
                            n_rs_seq, n_rs_batch, n_tokens, r, d);
                    ++fail_b;
                }
            }
            n_fail += fail_b;
            fprintf(stderr, "  n_rs_seq=%2u n_rs_batch=%2u n_tokens=%2u: %s\n",
                    n_rs_seq, n_rs_batch, n_tokens, fail_b ? "FAIL" : "PASS");
        }
    }

    fprintf(stderr, "\n%s: total failures = %d\n", n_fail ? "FAIL" : "PASS", n_fail);
    return n_fail ? 2 : 0;
}
