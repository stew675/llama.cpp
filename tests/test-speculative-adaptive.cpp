#include "speculative-adaptive.h"

#undef NDEBUG

#include <cassert>
#include <cstdio>

static void test_reset(void) {
    common_speculative_adaptive ctrl;

    // cold start at the floor max(1, n_min_adaptive), bucket at the drop weight
    ctrl.reset(8, 1);
    assert(ctrl.n_cur == 1);
    assert(ctrl.n_bucket == 20);

    // the default adaptive floor of 3 starts the controller at depth 3
    ctrl.reset(8, 3);
    assert(ctrl.n_cur == 3);
    assert(ctrl.n_bucket == 20);

    // the ceiling clamps the cold start to n_max
    ctrl.reset(1, 3);
    assert(ctrl.n_cur == 1);
    assert(ctrl.n_bucket == 20);
}

static void test_climb(void) {
    common_speculative_adaptive ctrl;
    ctrl.reset(8, 1); // ceiling 8, cold start at the floor 1

    // a full accept at depth 1 adds +1; the cap is 40, so 20 full accepts
    // climb from the reset point of 20
    ctrl.update(1, 8, 1);
    assert(ctrl.n_cur == 1);
    assert(ctrl.n_bucket == 21);

    for (int i = 0; i < 18; ++i) {
        ctrl.update(1, 8, 1);
        assert(ctrl.n_cur == 1);
    }
    assert(ctrl.n_bucket == 39);
    ctrl.update(1, 8, 1);
    assert(ctrl.n_cur == 2);
    assert(ctrl.n_bucket == 20);

    // a near miss at depth 2 drains 1: it is a real miss, not neutral
    ctrl.update(1, 8, 1);
    assert(ctrl.n_cur == 2);
    assert(ctrl.n_bucket == 19);

    // depth 2 full accepts add +1 each, so 21 of them reach the cap from 19
    for (int i = 0; i < 20; ++i) {
        ctrl.update(2, 8, 1);
        assert(ctrl.n_cur == 2);
    }
    assert(ctrl.n_bucket == 39);
    ctrl.update(2, 8, 1);
    assert(ctrl.n_cur == 3);
    assert(ctrl.n_bucket == 20);

    // depth 3 full accepts add +2 each, so 10 of them reach the cap
    for (int i = 0; i < 9; ++i) {
        ctrl.update(3, 8, 1);
        assert(ctrl.n_cur == 3);
    }
    assert(ctrl.n_bucket == 38);
    ctrl.update(3, 8, 1);
    assert(ctrl.n_cur == 4);
    assert(ctrl.n_bucket == 20);

    // a draft truncated one token short of the depth is a small loss: the
    // controller has no truncation signal, it just sees a 1-token miss
    ctrl.update(3, 8, 1); // depth 4, only 3 tokens drafted, all accepted
    assert(ctrl.n_cur == 4);
    assert(ctrl.n_bucket == 19);

    // depth 4 full accepts add +3 each; the truncated draft above drained 1,
    // so it takes 7 full accepts to climb from here
    for (int i = 0; i < 6; ++i) {
        ctrl.update(4, 8, 1);
        assert(ctrl.n_cur == 4);
    }
    assert(ctrl.n_bucket == 37);
    ctrl.update(4, 8, 1);
    assert(ctrl.n_cur == 5);
    assert(ctrl.n_bucket == 20);

    // depth 5 full accepts add +4 each, so 5 of them climb from the reset
    for (int i = 0; i < 4; ++i) {
        ctrl.update(5, 8, 1);
        assert(ctrl.n_cur == 5);
    }
    assert(ctrl.n_bucket == 36);
    ctrl.update(5, 8, 1);
    assert(ctrl.n_cur == 6);
    assert(ctrl.n_bucket == 24); // drop pressure grows with depth: max(20, 6*4)

    // depth 6 full accepts add +5 each; the cap is 44 (drop 24 + climb 20),
    // so 4 of them climb from the reset of 24
    for (int i = 0; i < 3; ++i) {
        ctrl.update(6, 8, 1);
        assert(ctrl.n_cur == 6);
    }
    assert(ctrl.n_bucket == 39);
    ctrl.update(6, 8, 1);
    assert(ctrl.n_cur == 7);
    assert(ctrl.n_bucket == 28);

    // depth 7 full accepts add +6 each; cap 48 (drop 28 + climb 20), so 4
    // of them climb from the reset of 28
    for (int i = 0; i < 3; ++i) {
        ctrl.update(7, 8, 1);
        assert(ctrl.n_cur == 7);
    }
    assert(ctrl.n_bucket == 46);
    ctrl.update(7, 8, 1);
    assert(ctrl.n_cur == 8);
    assert(ctrl.n_bucket == 32);

    // at the ceiling the bucket saturates at the cap (52) instead of growing
    // unbounded; the first full accept clamps it down to the cap
    for (int i = 0; i < 8; ++i) {
        ctrl.update(8, 8, 1);
    }
    assert(ctrl.n_cur == 8);
    assert(ctrl.n_bucket == 52);

    // a round with zero accepted tokens is a total miss: it drains the bucket
    // (8 at depth 8) but the depth stays at the ceiling
    ctrl.update(0, 8, 1);
    assert(ctrl.n_cur == 8);
    assert(ctrl.n_bucket == 44);

    // deep climbs are fast: at depth 10 a full accept adds +9 and the drop
    // pressure is 40, so 3 of them climb; at depth 11 two of them climb
    ctrl.reset(12, 1);
    ctrl.n_cur    = 10;
    ctrl.n_bucket = 40;
    for (int i = 0; i < 2; ++i) {
        ctrl.update(10, 12, 1);
        assert(ctrl.n_cur == 10);
    }
    assert(ctrl.n_bucket == 58);
    ctrl.update(10, 12, 1);
    assert(ctrl.n_cur == 11);
    assert(ctrl.n_bucket == 44);

    // at depth 11 a full accept adds +10: two of them climb
    ctrl.update(11, 12, 1);
    assert(ctrl.n_cur == 11);
    assert(ctrl.n_bucket == 54);
    ctrl.update(11, 12, 1);
    assert(ctrl.n_cur == 12);
    assert(ctrl.n_bucket == 48);
}

static void test_drop(void) {
    common_speculative_adaptive ctrl;
    ctrl.reset(8, 1); // cold start at the floor

    // at the floor a total miss drains 1, but the depth cannot drop below
    // the floor so the bucket clamps at 0
    for (int i = 0; i < 20; ++i) {
        ctrl.update(0, 8, 1);
    }
    assert(ctrl.n_cur == 1);
    assert(ctrl.n_bucket == 0);
    for (int i = 0; i < 80; ++i) {
        ctrl.update(0, 8, 1);
    }
    assert(ctrl.n_cur == 1);
    assert(ctrl.n_bucket == 0);

    // at depth 3 a total miss adds -3, so 7 misses drain the bucket of 20
    ctrl.reset(8, 1);
    ctrl.n_cur    = 3;
    ctrl.n_bucket = 20;
    for (int i = 0; i < 6; ++i) {
        ctrl.update(0, 8, 1);
        assert(ctrl.n_cur == 3);
    }
    assert(ctrl.n_bucket == 2);
    ctrl.update(0, 8, 1);
    assert(ctrl.n_cur == 2);
    assert(ctrl.n_bucket == 20);

    // at depth 2 a near miss drains 1: 20 near misses drop one step
    for (int i = 0; i < 19; ++i) {
        ctrl.update(1, 8, 1);
        assert(ctrl.n_cur == 2);
    }
    assert(ctrl.n_bucket == 1);
    ctrl.update(1, 8, 1);
    assert(ctrl.n_cur == 1);
    assert(ctrl.n_bucket == 20);

    // at depth 2 a total miss adds -2, so 10 misses drop one step
    ctrl.reset(8, 1);
    ctrl.n_cur    = 2;
    ctrl.n_bucket = 20;
    for (int i = 0; i < 9; ++i) {
        ctrl.update(0, 8, 1);
        assert(ctrl.n_cur == 2);
    }
    assert(ctrl.n_bucket == 2);
    ctrl.update(0, 8, 1);
    assert(ctrl.n_cur == 1);
    assert(ctrl.n_bucket == 20);

    // back at the floor, misses drain and clamp at 0
    for (int i = 0; i < 100; ++i) {
        ctrl.update(0, 8, 1);
    }
    assert(ctrl.n_cur == 1);
    assert(ctrl.n_bucket == 0);

    // deep depths fall quickly: at depth 5 a total miss adds -5, so 5 misses
    // drain the bucket of 25
    ctrl.reset(8, 1);
    ctrl.n_cur    = 5; // simulate a controller that already climbed to 5
    ctrl.n_bucket = 25;
    for (int i = 0; i < 4; ++i) {
        ctrl.update(0, 8, 1);
        assert(ctrl.n_cur == 5);
    }
    assert(ctrl.n_bucket == 5);
    ctrl.update(0, 8, 1);
    assert(ctrl.n_cur == 4);
    assert(ctrl.n_bucket == 20);

    // at depth 9 a total miss adds -9: two misses drain 18 and drop, and the
    // reset honors the depth-scaled drop pressure of 32 at depth 8
    ctrl.reset(8, 1);
    ctrl.n_cur    = 9;
    ctrl.n_bucket = 18;
    ctrl.update(0, 8, 1);
    assert(ctrl.n_cur == 9);
    assert(ctrl.n_bucket == 9);
    ctrl.update(0, 8, 1);
    assert(ctrl.n_cur == 8);
    assert(ctrl.n_bucket == 32);
}

static void test_momentum(void) {
    common_speculative_adaptive ctrl;
    ctrl.reset(8, 1);
    ctrl.n_cur    = 3;
    ctrl.n_bucket = 20;

    // a bad patch drains the bucket: 5 total misses at depth 3 add -3 each
    for (int i = 0; i < 5; ++i) {
        ctrl.update(0, 8, 1); // total miss
    }
    assert(ctrl.n_bucket == 5);
    assert(ctrl.n_cur == 3);

    // full accepts do not wipe the damage: they recover the bucket 2 at a
    // time, and 10 of them only get it back to 25
    for (int i = 0; i < 10; ++i) {
        ctrl.update(3, 8, 1);
    }
    assert(ctrl.n_bucket == 25);
    assert(ctrl.n_cur == 3);

    // 7 more full accepts reach 39; one more reaches the cap of 40 and climbs
    for (int i = 0; i < 7; ++i) {
        ctrl.update(3, 8, 1);
    }
    assert(ctrl.n_bucket == 39);
    assert(ctrl.n_cur == 3);
    ctrl.update(3, 8, 1);
    assert(ctrl.n_cur == 4);
    assert(ctrl.n_bucket == 20);
}

static void test_misses(void) {
    common_speculative_adaptive ctrl;
    ctrl.reset(8, 1);
    ctrl.n_cur    = 5;
    ctrl.n_bucket = 25;

    // there is no special case for truncated drafts: a draft that stops one
    // token short of the depth is a 1-token miss, whatever the reason
    ctrl.update(4, 8, 1);
    assert(ctrl.n_bucket == 24);

    // two tokens short: -2
    ctrl.update(3, 8, 1);
    assert(ctrl.n_bucket == 22);

    // three tokens short: -3
    ctrl.update(2, 8, 1);
    assert(ctrl.n_bucket == 19);

    // a rejected token drains in full: accepting 1 of 2 drafted at depth 5
    // is a 4-token miss
    ctrl.update(1, 8, 1);
    assert(ctrl.n_bucket == 15);

    // a full-length near miss is a 1-token miss, a full-length total miss -5
    ctrl.update(4, 8, 1);
    assert(ctrl.n_bucket == 14);
    ctrl.update(0, 8, 1);
    assert(ctrl.n_bucket == 9);
}

static void test_floor(void) {
    common_speculative_adaptive ctrl;

    // with the floor at 2 the depth never drops below 2, no matter how bad
    // the content gets; the bucket clamps at 0 instead
    ctrl.reset(8, 2);
    for (int i = 0; i < 1000; ++i) {
        ctrl.update(0, 8, 2);
    }
    assert(ctrl.n_cur == 2);
    assert(ctrl.n_bucket == 0);

    // climbs still work from the floor: from a clamped 0 it takes the full
    // cap, i.e. 40 full accepts at depth 2 (+1 each)
    for (int i = 0; i < 39; ++i) {
        ctrl.update(2, 8, 2);
        assert(ctrl.n_cur == 2);
    }
    assert(ctrl.n_bucket == 39);
    ctrl.update(2, 8, 2);
    assert(ctrl.n_cur == 3);
    assert(ctrl.n_bucket == 20);

    // drops stop at the floor, not below it
    for (int i = 0; i < 6; ++i) {
        ctrl.update(0, 8, 2); // -3 each: 6 misses leave 2
    }
    assert(ctrl.n_cur == 3);
    assert(ctrl.n_bucket == 2);
    ctrl.update(0, 8, 2); // the 7th drops to the floor
    assert(ctrl.n_cur == 2);
    assert(ctrl.n_bucket == 20);
    for (int i = 0; i < 20; ++i) {
        ctrl.update(0, 8, 2); // 10 misses drain the 20-token bucket
    }
    assert(ctrl.n_cur == 2);
    assert(ctrl.n_bucket == 0); // clamped at 0, not dropped
}

static void test_pinned(void) {
    common_speculative_adaptive ctrl;

    // with the floor at the ceiling the depth is pinned: it cannot climb
    // above n_max or drop below n_min_adaptive, and the bucket clamps at
    // both ends instead of changing the depth
    ctrl.reset(3, 3);
    assert(ctrl.n_cur == 3);
    assert(ctrl.n_bucket == 20);

    // full accepts saturate the bucket at the cap (40), the depth stays 3
    for (int i = 0; i < 100; ++i) {
        ctrl.update(3, 3, 3);
    }
    assert(ctrl.n_cur == 3);
    assert(ctrl.n_bucket == 40);

    // total misses drain the bucket and clamp at 0, the depth stays 3
    for (int i = 0; i < 100; ++i) {
        ctrl.update(0, 3, 3);
    }
    assert(ctrl.n_cur == 3);
    assert(ctrl.n_bucket == 0);

    // it recovers from the clamped 0: 20 full accepts refill the bucket
    for (int i = 0; i < 19; ++i) {
        ctrl.update(3, 3, 3);
        assert(ctrl.n_cur == 3);
    }
    assert(ctrl.n_bucket == 38);
    ctrl.update(3, 3, 3);
    assert(ctrl.n_cur == 3);
    assert(ctrl.n_bucket == 40); // saturated at the cap again
}

static void test_other_feed(void) {
    common_speculative_adaptive ctrl;

    // another speculator (ngram-mod) that accepted far more than the current
    // depth feeds update(n, n): the full-accept credit is max(1, n-1), so one
    // strong round climbs one full depth step
    ctrl.reset(12, 3); // floor 3, ceiling 12
    assert(ctrl.n_cur == 3);
    assert(ctrl.n_bucket == 20);

    // ngram accepts 40 tokens in one round: +39, past the cap of 40, one climb
    ctrl.update(40, 12, 3);
    assert(ctrl.n_cur == 4);
    assert(ctrl.n_bucket == 20);

    // the next strong round climbs again: one step per round while the run lasts
    ctrl.update(40, 12, 3);
    assert(ctrl.n_cur == 5);
    assert(ctrl.n_bucket == 20);

    // a modest over-full accept (>= depth but short of the cap) gives partial
    // credit: at depth 5 the cap is 40, so accepting 10 adds +9, no climb yet
    ctrl.update(10, 12, 3);
    assert(ctrl.n_cur == 5);
    assert(ctrl.n_bucket == 29);

    // a weak ngram round (accepting fewer than the depth) drains like a miss
    ctrl.update(2, 12, 3);
    assert(ctrl.n_cur == 5);
    assert(ctrl.n_bucket == 26);

    // at the ceiling the depth is clamped: strong rounds saturate the bucket
    // at the cap (drop 48 + climb 20 = 68 at depth 12)
    ctrl.reset(12, 3);
    ctrl.n_cur    = 12;
    ctrl.n_bucket = 48;
    ctrl.update(40, 12, 3);
    assert(ctrl.n_cur == 12);
    assert(ctrl.n_bucket == 68);
}

int main(void) {
    test_reset();
    test_climb();
    test_drop();
    test_momentum();
    test_misses();
    test_floor();
    test_pinned();
    test_other_feed();

    printf("test-speculative-adaptive: all tests OK\n\n");

    return 0;
}
