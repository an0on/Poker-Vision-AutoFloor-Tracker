"""REQ-23: `optimal_assignment` (max-cardinality, then min-distance matching)."""

from __future__ import annotations

from poker_vision.tracking.matching import optimal_assignment


def test_no_rows_or_columns_matches_nothing():
    assert optimal_assignment(row_count=0, col_count=0, valid_cost={}, max_valid_cost=1.0) == {}


def test_single_valid_pair_matches():
    result = optimal_assignment(
        row_count=1, col_count=1, valid_cost={(0, 0): 0.2}, max_valid_cost=1.0
    )
    assert result == {0: 0}


def test_pair_absent_from_valid_cost_is_never_matched():
    # Only row 0 -> col 1 is offered as valid; row 0 -> col 0 doesn't exist.
    result = optimal_assignment(
        row_count=1, col_count=2, valid_cost={(0, 1): 0.2}, max_valid_cost=1.0
    )
    assert result == {0: 1}


# The exact scenario Codex flagged in a greedy matcher: tracks (columns) at
# 0.00/0.04, detections (rows) at 0.03/0.08, threshold 0.05. Greedy grabs
# the globally closest pair (col 1 <-> row 0, distance 0.01) first, which
# strands row 1 unmatched (row 1 <-> col 0 is 0.08, over threshold) even
# though row0<->col0 (0.03) and row1<->col1 (0.04) are both valid and keep
# every track. Maximizing matched pairs must win over the single cheapest
# pair.
def test_maximizes_matched_pairs_over_the_single_cheapest_pair():
    # row 0 = det 0.03, row 1 = det 0.08; col 0 = track 0.00, col 1 = track
    # 0.04. (row 1, col 0) = |0.08-0.00| = 0.08 is over the 0.05 threshold,
    # so it's simply absent here, exactly as the tracker would filter it.
    valid_cost = {
        (0, 0): 0.03,  # det 0.03 <-> track 0.00
        (0, 1): 0.01,  # det 0.03 <-> track 0.04 -- the single cheapest pair
        (1, 1): 0.04,  # det 0.08 <-> track 0.04
    }
    result = optimal_assignment(
        row_count=2, col_count=2, valid_cost=valid_cost, max_valid_cost=0.05
    )
    # Taking (0, 1) alone (cheapest) would strand row 1 unmatched. The
    # 2-pair matching (0, 0) + (1, 1) keeps both rows matched instead.
    assert result == {0: 0, 1: 1}


def test_among_max_cardinality_matchings_picks_the_cheaper_one():
    # Two disjoint 1-1 matchings are both size 1 (row 1 has no valid pair at
    # all); the cheaper of the two available options for row 0 must win.
    result = optimal_assignment(
        row_count=1,
        col_count=2,
        valid_cost={(0, 0): 0.5, (0, 1): 0.1},
        max_valid_cost=1.0,
    )
    assert result == {0: 1}


def test_more_rows_than_columns_leaves_extra_rows_unmatched():
    result = optimal_assignment(
        row_count=2, col_count=1, valid_cost={(0, 0): 0.1, (1, 0): 0.2}, max_valid_cost=1.0
    )
    assert result == {0: 0}


# Codex finding: with 5+ nodes, reaching maximum cardinality can require
# reassigning a whole chain of existing pairs (an augmenting path), not
# just adding one edge -- so the "stay unmatched" penalty must dominate the
# cost of an entire chain, not a single edge. Here, (i, i) for i=0..3 are
# free (cost 0) direct matches; the only way to also match row 4 is to
# shift the whole chain 4->0->1->1->2->2->3->3->4 (row4->col0, row0->col1,
# row1->col2, row2->col3, row3->col4), each edge costing max_valid_cost.
# A matcher that only weighs single edges against the unmatched penalty
# settles for the 4 free matches (row 4 and col 4 stranded); the correct
# answer keeps every row and column matched, even at higher total cost.
def test_reaching_max_cardinality_can_require_a_whole_chain_of_reassignments():
    max_valid_cost = 0.05
    valid_cost = {
        (0, 0): 0.0,
        (1, 1): 0.0,
        (2, 2): 0.0,
        (3, 3): 0.0,
        (4, 0): max_valid_cost,
        (0, 1): max_valid_cost,
        (1, 2): max_valid_cost,
        (2, 3): max_valid_cost,
        (3, 4): max_valid_cost,
    }
    result = optimal_assignment(
        row_count=5, col_count=5, valid_cost=valid_cost, max_valid_cost=max_valid_cost
    )
    assert result == {0: 1, 1: 2, 2: 3, 3: 4, 4: 0}


def test_more_columns_than_rows_leaves_extra_columns_unmatched():
    result = optimal_assignment(
        row_count=1, col_count=2, valid_cost={(0, 0): 0.1, (0, 1): 0.2}, max_valid_cost=1.0
    )
    assert result == {0: 0}
