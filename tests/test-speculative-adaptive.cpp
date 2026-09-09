#include "speculative-adaptive.h"

#undef NDEBUG

#include <cassert>
#include <cstdio>

// The controller's constants (see speculative-adaptive.h):
//   drop_pressure(d)  = max(60, 10*d)      -> the reset point of the bucket
//   climb_budget(d)   = 20 + 6*(d - 1)     -> the distance from there to the cap
//   full accept       = max(1, n_accepted - 1)
//   miss              = n_accepted - d
//   cold start        = min(cap, max(floor, cap - 3))
// Every case below is written against those numbers so a retune has to update
// this file deliberately rather than silently.

static void test_reset(void) {
    common_speculative_adaptive ctrl;

    // the cold start sits three steps below the ceiling, with the bucket at the
    // drop pressure of that depth
    ctrl.reset(12, 3);
    assert(ctrl.n_cur == 9);
    assert(ctrl.n_bucket == 90);

    ctrl.reset(8, 1);
    assert(ctrl.n_cur == 5);
    assert(ctrl.n_bucket == 60);

    // the floor bounds the cold start from below
    ctrl.reset(4, 3);
    assert(ctrl.n_cur == 3);
    assert(ctrl.n_bucket == 60);

    ctrl.reset(3, 3);
    assert(ctrl.n_cur == 3);
    assert(ctrl.n_bucket == 60);

    // ... and the ceiling from above (a cap below the floor wins)
    ctrl.reset(1, 3);
    assert(ctrl.n_cur == 1);
    assert(ctrl.n_bucket == 60);
}

static void test_reset_start(void) {
    common_speculative_adaptive ctrl;

    // --spec-draft-n-start overrides the default cold start; the chosen depth is
    // clamped to [floor, cap] and the bucket is that depth's drop pressure
    ctrl.reset(12, 3, 7);
    assert(ctrl.n_cur == 7);
    assert(ctrl.n_bucket == 70);

    ctrl.reset(12, 3, 3);
    assert(ctrl.n_cur == 3);
    assert(ctrl.n_bucket == 60);

    // below the floor clamps up ...
    ctrl.reset(12, 3, 1);
    assert(ctrl.n_cur == 3);
    assert(ctrl.n_bucket == 60);

    // ... and above the cap clamps down
    ctrl.reset(12, 4, 15);
    assert(ctrl.n_cur == 12);
    assert(ctrl.n_bucket == 120);

    // n_start == 0 (unset) keeps the default cold start (cap - 3)
    ctrl.reset(12, 3, 0);
    assert(ctrl.n_cur == 9);
    assert(ctrl.n_bucket == 90);
}

static void test_climb(void) {
    common_speculative_adaptive ctrl;

    // from the cold start at 9 a step needs climb_budget(9) = 68, and a full
    // accept is worth n_accepted - 1 = 8, so nine of them
    ctrl.reset(12, 3);
    for (int i = 0; i < 8; ++i) {
        ctrl.update(9, 12, 3);
        assert(ctrl.n_cur == 9);
    }
    assert(ctrl.n_bucket == 154);
    ctrl.update(9, 12, 3);
    assert(ctrl.n_cur == 10);
    // the 4-token surplus over the cap carries, and the new depth adds its own
    // drop pressure: 0 + 4 carry + drop_pressure(10)
    assert(ctrl.n_bucket == 104);

    // depth 10: budget 74, a full accept worth 9
    for (int i = 0; i < 7; ++i) {
        ctrl.update(10, 12, 3);
        assert(ctrl.n_cur == 10);
    }
    assert(ctrl.n_bucket == 167);
    ctrl.update(10, 12, 3);
    assert(ctrl.n_cur == 11);
    assert(ctrl.n_bucket == 112);

    // off the floor: at depth 3 the budget is 32 and a full accept is worth 2
    ctrl.reset(4, 3);
    assert(ctrl.n_cur == 3);
    assert(ctrl.n_bucket == 60);
    for (int i = 0; i < 15; ++i) {
        ctrl.update(3, 4, 3);
        assert(ctrl.n_cur == 3);
    }
    assert(ctrl.n_bucket == 90);
    ctrl.update(3, 4, 3);
    assert(ctrl.n_cur == 4);
    assert(ctrl.n_bucket == 60);

    // the ceiling blocks further climbs and the bucket parks at the cap
    for (int i = 0; i < 100; ++i) {
        ctrl.update(4, 4, 3);
    }
    assert(ctrl.n_cur == 4);
    assert(ctrl.n_bucket == 98);  // drop_pressure(4) + climb_budget(4)

    // depth 1: a full accept is worth max(1, 1 - 1) = 1 and the budget is 20
    ctrl.reset(8, 1);
    ctrl.n_cur = 1;
    ctrl.n_bucket = 60;
    for (int i = 0; i < 19; ++i) {
        ctrl.update(1, 8, 1);
        assert(ctrl.n_cur == 1);
    }
    assert(ctrl.n_bucket == 79);
    ctrl.update(1, 8, 1);
    assert(ctrl.n_cur == 2);
    assert(ctrl.n_bucket == 60);
}

static void test_drop(void) {
    common_speculative_adaptive ctrl;

    // from 9, drop_pressure(9) = 90 and a total miss drains 9, so ten of them
    ctrl.reset(12, 3);
    for (int i = 0; i < 9; ++i) {
        ctrl.update(0, 12, 3);
        assert(ctrl.n_cur == 9);
    }
    assert(ctrl.n_bucket == 9);
    ctrl.update(0, 12, 3);
    assert(ctrl.n_cur == 8);
    // 0 - drop_pressure(9) + bucket_cap(8)
    assert(ctrl.n_bucket == 52);

    // the floor clamps: no drop below it, and the bucket parks at 0
    ctrl.reset(8, 2);
    ctrl.n_cur = 2;
    ctrl.n_bucket = 60;
    for (int i = 0; i < 29; ++i) {
        ctrl.update(0, 8, 2);
        assert(ctrl.n_cur == 2);
    }
    assert(ctrl.n_bucket == 2);
    ctrl.update(0, 8, 2);
    assert(ctrl.n_cur == 2);
    assert(ctrl.n_bucket == 0);

    // no matter how bad it gets, the depth never leaves the floor
    for (int i = 0; i < 1000; ++i) {
        ctrl.update(0, 8, 2);
    }
    assert(ctrl.n_cur == 2);
    assert(ctrl.n_bucket == 0);
}

static void test_delta(void) {
    common_speculative_adaptive ctrl;
    ctrl.reset(8, 3);
    ctrl.n_cur = 3;
    ctrl.n_bucket = 60;

    // a full accept is worth n_accepted - 1
    ctrl.update(3, 8, 3);
    assert(ctrl.n_bucket == 62);

    // an accept from another speculator can exceed the depth: still max(1, n - 1)
    ctrl.update(6, 8, 3);
    assert(ctrl.n_bucket == 67);

    // a draft that came up one token short is a real miss, not neutral
    ctrl.update(2, 8, 3);
    assert(ctrl.n_bucket == 66);

    // ... and a total miss drains the whole depth
    ctrl.update(0, 8, 3);
    assert(ctrl.n_bucket == 63);
}

static void test_mixed_workload_settles(void) {
    common_speculative_adaptive ctrl;

    // A workload whose acceptance is good enough to hold a high depth must not
    // be walked up to the ceiling by the surplus carry: after the cold start the
    // controller may climb while the acceptance is there, but once the full
    // accepts stop it has to give the depth back. Alternating a full accept with
    // a half miss at depth 9 (drift 8 and -4.5, so positive) climbs; alternating
    // it with a total miss (drift 8 and -9, negative) drains and drops.
    ctrl.reset(12, 3);
    for (int i = 0; i < 400; ++i) {
        ctrl.update(i % 2 ? 0 : 9, 12, 3);
    }
    assert(ctrl.n_cur < 12);

    // a workload that never accepts anything must end at the floor with an
    // empty bucket, whatever depth it started from
    ctrl.reset(12, 3);
    for (int i = 0; i < 400; ++i) {
        ctrl.update(0, 12, 3);
    }
    assert(ctrl.n_cur == 3);
    assert(ctrl.n_bucket == 0);

    // and one that always accepts must end at the ceiling
    ctrl.reset(12, 3);
    for (int i = 0; i < 400; ++i) {
        ctrl.update(ctrl.n_cur + 1, 12, 3);
    }
    assert(ctrl.n_cur == 12);
}

int main(void) {
    test_reset();
    test_reset_start();
    test_climb();
    test_drop();
    test_delta();
    test_mixed_workload_settles();

    printf("test-speculative-adaptive: all tests OK\n\n");

    return 0;
}
