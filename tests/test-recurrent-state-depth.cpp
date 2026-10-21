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

#include <algorithm>
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
        uint32_t n_tokens, uint32_t n_rollback, bool verbose, bool * refused_out = nullptr) {
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
        if (refused_out) {
            *refused_out = true;
        } else {
            fprintf(stderr, "  [rollback] seq_rm refused (n_tokens=%u rollback=%u)\n", n_tokens, n_rollback);
        }
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


// ---------------------------------------------------------------------------
// Phase C: a cell that find_slot MOVES between a verify batch and that
// sequence's rollback must keep its snapshot planes.  build_rs relocates the
// extra states (the cells inside [head, head+n_rs) that no sequence of this
// ubatch uses) by copying plane 0 only, so a moved cell's planes 1..n_rs_seq
// stay at the old index and are overwritten by the cells that take their
// place.  With n_seq_max = 3 and 3 resident sequences, a batch of
// [seq 0, seq 2] spans cells 0..2, and find_slot swaps seq 1 (cell 1) with
// seq 2 (cell 2): seq 1 becomes the extra state, its snapshots are read from
// its OLD cell, and its new cell keeps whatever the old cell 2 held.
//
//   moved = false : the two tokens are decoded one at a time, so no cell moves
//   moved = true  : the two tokens are decoded in ONE batch of [seq 0, seq 2]
//
// Both arms must produce identical logits for the rolled-back sequence.
static llama_context * make_ctx_seq( const common_params & params, llama_model * model,
        uint32_t n_rs_seq, uint32_t n_rs_batch, uint32_t n_ubatch) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max  = 3;
    cparams.n_rs_seq   = n_rs_seq;
    cparams.n_rs_batch = n_rs_batch;
    cparams.n_ctx      = 512;
    cparams.n_batch    = 512;
    cparams.n_ubatch   = n_ubatch;
    cparams.kv_unified = true; // a non-sequential split is what puts two seqs in one ubatch
    return llama_init_from_model(model, cparams);
}

static bool decode_pair(llama_context * ctx, llama_token t0, llama_token t1,
                        llama_pos pos, llama_seq_id s0, llama_seq_id s1, bool separate,
                        bool last_logits) {
    if (separate) {
        common_batch b0(ctx);
        b0.add(t0, pos, s0, last_logits);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, b0.get()) != 0) {
            return false;
        }
        common_batch b1(ctx);
        b1.add(t1, pos, s1, last_logits);
        return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, b1.get()) == 0;
    }
    common_batch b(ctx);
    b.add(t0, pos, s0, last_logits);
    b.add(t1, pos, s1, last_logits);
    return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, b.get()) == 0;
}

static std::vector<float> run_moved_cell(
        const common_params & params, llama_model * model, llama_context * ctx_tok,
        const llama_vocab * vocab, uint32_t n_rs_seq, uint32_t n_rollback,
        bool moved, bool verbose, bool * ok_out) {
    const int      n_vocab = llama_vocab_n_tokens(vocab);
    const uint32_t n_pre   = 6;
    const uint32_t K       = n_rs_seq + 1;

    *ok_out = false;
    std::vector<float> none;

    // distinct token streams per sequence, drawn from the shared pool
    std::vector<llama_token> pool = make_tokens(ctx_tok, vocab, 64, (uint32_t) n_vocab);
    auto tok = [&](uint32_t i, uint32_t s) { return pool[(i + 7u*s) % pool.size()]; };

    llama_context * ctx = make_ctx_seq(params, model, n_rs_seq, n_rs_seq + 1, 32);
    if (ctx == nullptr) {
        return none;
    }

    bool ok = true;
    // 3 prefills -> cells 0, 1, 2 in sequence-id order
    for (llama_seq_id s = 0; s < 3; ++s) {
        common_batch b(ctx);
        for (uint32_t i = 0; i < n_pre; ++i) {
            b.add(tok(i, (uint32_t) s), (llama_pos) i, s, i + 1 == n_pre);
        }
        ok = ok && llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, b.get()) == 0;
    }
    // verify batch for seq 1: K tokens, so K-1 = n_rs_seq snapshot planes
    {
        common_batch b(ctx);
        for (uint32_t i = 0; i < K; ++i) {
            b.add(tok(n_pre + i, 1), (llama_pos) (n_pre + i), 1, i + 1 == K);
        }
        ok = ok && llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, b.get()) == 0;
    }
    // the move: seq 0 and seq 2 decode one token each
    ok = ok && decode_pair(ctx, tok(n_pre, 0), tok(n_pre, 2), (llama_pos) n_pre, 0, 2, !moved, false);
    if (!ok) {
        llama_free(ctx);
        return none;
    }
    // partial rollback of seq 1 into its verify batch, then replay
    const uint32_t p0 = n_pre + K - n_rollback;
    if (!llama_memory_seq_rm(llama_get_memory(ctx), 1, (llama_pos) p0, -1)) {
        if (verbose) {
            fprintf(stderr, "  [moved-cell] seq_rm refused (n_rs_seq=%u rollback=%u)\n", n_rs_seq, n_rollback);
        }
        llama_free(ctx);
        return none;
    }
    std::vector<float> last;
    for (uint32_t i = p0; i < n_pre + K; ++i) {
        if (!decode_one(ctx, tok(i, 1), (llama_pos) i, 1)) {
            llama_free(ctx);
            return none;
        }
        const float * l = llama_get_logits_ith(ctx, 0);
        if (l == nullptr) {
            llama_free(ctx);
            return none;
        }
        last.assign(l, l + n_vocab);
    }
    llama_free(ctx);
    *ok_out = true;
    return last;
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

    // Phase B: a batch deeper than K = n_rs_seq + 1.  The rollback bound is upstream's
    // (#31368): rs_max = min(n_seq_tokens, n_rs_seq) for a ubatch of n_seq_tokens tokens, and
    // 0 when the ubatch is longer than n_rs_seq + 1 -- a batch that long only guarantees its final
    // state, so every rollback into it must be REFUSED.  (The delivery's own ubatch tracker used to
    // allow a rollback up to n_rs_batch tokens, which is where the deep-draft verify batch lived;
    // that relaxation went with the tracker -- see wip/upstream-rebase-261108, rung R04, D4.)
    fprintf(stderr, "=== Phase B: deeper than K (n_tokens > n_rs_seq+1) refuses every rollback ===\n");
    for (uint32_t n_rs_seq = 7; n_rs_seq <= 15; n_rs_seq += 4) {
        const uint32_t K = n_rs_seq + 1;
        for (uint32_t extra = 1; extra <= 16; extra += 5) {
            const uint32_t n_rs_batch = K + extra;
            const uint32_t n_tokens   = n_rs_batch;
            int fail_b = 0;
            for (uint32_t r = 1; r <= n_rs_seq; ++r) {
                bool refused = false;
                const double d = run_case(params, model, llama_init->context(), vocab, n_rs_seq, n_rs_batch, 32,
                                          n_tokens, r, false, &refused);
                if (d >= 0 || !refused) {
                    fprintf(stderr, "FAIL B: n_rs_seq=%u n_rs_batch=%u n_tokens=%u rollback=%u was NOT refused (d=%g)\n",
                            n_rs_seq, n_rs_batch, n_tokens, r, d);
                    ++fail_b;
                }
            }
            n_fail += fail_b;
            fprintf(stderr, "  n_rs_seq=%2u n_rs_batch=%2u n_tokens=%2u: %s\n",
                    n_rs_seq, n_rs_batch, n_tokens, fail_b ? "FAIL" : "PASS");
        }
    }

    // Phase C: a find_slot-moved cell keeps its rollback snapshots.  The two arms
    // differ only in whether seq 1's cell is relocated between its verify batch and
    // the rollback (one batch of [seq 0, seq 2] vs the same two tokens decoded one
    // at a time), so their logits must match.
    {
        fprintf(stderr, "=== Phase C: snapshots of a moved cell (3 sequences) ===\n");
        for (uint32_t n_rs_seq = 1; n_rs_seq <= 8; n_rs_seq *= 2) {
            int fail_c = 0;
            for (uint32_t r = 1; r <= n_rs_seq; ++r) {
                bool ok_a = false, ok_b = false;
                std::vector<float> a = run_moved_cell(params, model, llama_init->context(), vocab, n_rs_seq, r, false, false, &ok_a);
                std::vector<float> b = run_moved_cell(params, model, llama_init->context(), vocab, n_rs_seq, r, true,  false, &ok_b);
                if (!ok_a || !ok_b || a.empty() || a.size() != b.size()) {
                    fprintf(stderr, "FAIL C: n_rs_seq=%u rollback=%u: arm did not run (ok_plain=%d ok_moved=%d)\n",
                            n_rs_seq, r, (int) ok_a, (int) ok_b);
                    ++fail_c;
                    continue;
                }
                double d = 0.0;
                for (size_t t = 0; t < a.size(); ++t) {
                    d = std::max(d, std::fabs((double) a[t] - (double) b[t]));
                }
                if (d > eps) {
                    fprintf(stderr, "FAIL C: n_rs_seq=%u rollback=%u max diff=%g\n", n_rs_seq, r, d);
                    ++fail_c;
                }
            }
            n_fail += fail_c;
            fprintf(stderr, "  n_rs_seq=%2u moved cell: %s\n", n_rs_seq, fail_c ? "FAIL" : "PASS");
        }
    }

    fprintf(stderr, "\n%s: total failures = %d\n", n_fail ? "FAIL" : "PASS", n_fail);
    return n_fail ? 2 : 0;
}
