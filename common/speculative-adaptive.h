#pragma once

#include <algorithm>

// Adaptive draft depth controller for MTP speculative decoding (draft-mtp-adaptive).
//
// Single-bucket state machine. Every verification result moves a bucket B by
// (n_accepted - depth), except that a full accept (n_accepted >= depth) is
// flipped to add max(1, n_accepted - 1) instead: a deeper or longer accept is
// worth more, so climbing accelerates with depth, and a strong run by another
// speculator (which can accept far more than the current depth in one round)
// climbs the depth one step per round while it lasts. Any shortfall drains
// depth - n_accepted, i.e. a miss of k tokens subtracts k. There is no special
// case for truncated drafts: a p_min-truncated draft is just a miss of the
// tokens it came up short, which keeps the controller honest about the depth
// it actually holds.
//
// B starts at the drop pressure D = max(20, 4*depth) on every depth change.
// When B reaches the cap T = D + 20 the depth climbs one step and B resets to
// D; when B falls to 0 the depth drops one step and B resets to D. The climb
// budget is a flat 20, so from the reset point a climb costs
// ceil(20 / max(1, depth - 1)) full accepts: slow off the floor (20 full
// accepts at depth 1-2), moderate in the middle (10 at depth 3, 5 at depth
// 5), quicker at depth (3 at depth 8+, 2 at 11+). The drop pressure is flat
// 20 up to depth 5 and then grows 4 per step (32 at depth 8, 40 at 10, 48 at
// 12), so a total miss drains depth and a drop costs max(ceil(20/depth), 4)
// misses: shallow depths fall slowly, deep depths fall in ~4 misses regardless of how deep
// they are. That keeps prose/reasoning pinned at the floor, lets code climb
// only while it earns it, and lets verbatim recall ride on to the ceiling. At
// the floor B clamps at 0 instead of dropping the depth; at the ceiling it
// clamps at the cap instead of growing unbounded.
struct common_speculative_adaptive {
    int n_cur    = 0;  // current adaptive draft depth N
    int n_bucket = 0;  // accumulated bucket: net accepted surplus over the depth

    // accumulated (n_cur - n_accepted) needed to drop one step from depth N;
    // flat 20 at the floor and mid depths, then growing 4 per step so a deep
    // depth still falls in ~4 total misses instead of collapsing instantly
    int drop_pressure(void) const {
        return std::max(20, n_cur * 4);
    }

    // bucket cap: drop pressure plus the climb budget, so climbing from the
    // reset point costs 20 net full-accept-equivalents
    int bucket_cap(void) const {
        return drop_pressure() + 20;
    }

    // reset to the floor max(1, n_min_adaptive), bounded by the ceiling n_max;
    // the controller climbs from there once acceptance feedback arrives
    void reset(int n_max, int n_min_adaptive) {
        const int cap   = std::max(1, n_max);
        const int floor = std::max(1, n_min_adaptive);

        n_cur    = std::min(floor, cap);
        n_bucket = drop_pressure();
    }

    // feed one verification result: n_accepted is the number of tokens the
    // target accepted this round. When another speculator (ngram-mod) produced
    // the accepted draft, callers pass its accepted count (which can exceed
    // the current depth), so strong runs by the other speculator are credited
    // like full accepts
    void update(int n_accepted, int n_max, int n_min_adaptive) {
        const int cap   = std::max(1, n_max);
        const int floor = std::max(1, n_min_adaptive);

        int delta;

        if (n_accepted >= n_cur) {
            delta = std::max(1, n_accepted - 1);
        } else {
            delta = n_accepted - n_cur;
        }

        n_bucket += delta;

        // At the bucket boundaries adjust the current depth if possible. Otherwise
        // both n_cur and n_bucket always get clamped to their minimum/maximum values
        if (n_bucket >= bucket_cap()) {
            // At or above the high water mark
            if (n_cur < cap) {
                n_cur++;
                n_bucket = drop_pressure();
            } else {
                n_cur = cap;
                n_bucket = bucket_cap();
            }
        } else if (n_bucket <= 0) {
            // At or below low water mark
            if (n_cur > floor) {
                n_cur--;
                n_bucket = drop_pressure();
            } else {
                n_cur = floor;
                n_bucket = 0;
            }
        }
    }
};
