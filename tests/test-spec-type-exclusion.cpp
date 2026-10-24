// The speculative types that share ONE draft context are rejected in combination, never merged.
//
// `draft-mtp` and `draft-mtp-adaptive` are one implementation class
// (`common_speculative_impl_draft_mtp`) over one `ctx_dft`, so enabling both builds two impls that
// each run process() on every batch and advance the same draft context twice.  The check that
// rejected the pair arrived with block 02 (adaptive MTP) and was LOST in the r328 re-base: it was a
// `throw` inside a hunk the re-expression no longer carried, and it is not a file, a define, an env
// var or an option, so nothing in the release flow failed for twelve releases (issue #348).  This is
// the instrument that would have: the combination is a behaviour, so it is tested, not grepped.
//
// It needs no model and no draft context -- the rejection happens while the enabled-type bitmask is
// inspected, before any context is touched, so every arm runs with `ctx_dft == nullptr` (which also
// means the accepted arms build no implementation at all).

#include "speculative.h"

#undef NDEBUG

#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <vector>

static bool init_throws(std::vector<enum common_speculative_type> types) {
    common_params_speculative params;
    params.types = std::move(types);
    try {
        common_speculative * spec = common_speculative_init(params, /*n_seq=*/ 1);
        common_speculative_free(spec);
    } catch (const std::exception &) {
        return true;
    }
    return false;
}

int main(void) {
    // the MTP pair shares the draft context: rejected
    assert(init_throws({ COMMON_SPECULATIVE_TYPE_DRAFT_MTP, COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE }));
    // each MTP type alone is a valid run
    assert(!init_throws({ COMMON_SPECULATIVE_TYPE_DRAFT_MTP }));
    assert(!init_throws({ COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE }));
    // the DFlash family is the same rule on the same shared context: the control that the pair check
    // itself is still there (a fix to one pair that drops the other would fail here)
    assert(init_throws({ COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH, COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK }));
    assert(!init_throws({ COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH }));

    printf("OK: the shared-draft-context type pairs are rejected and single types are accepted\n");
    return 0;
}
