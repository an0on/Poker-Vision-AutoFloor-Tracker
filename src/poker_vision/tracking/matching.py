"""Optimal bipartite assignment for nearest-match tracking (REQ-23).

`NearestMatchTracker` needs, per class, the pairing between existing tracks
and this frame's detections that keeps as many valid (within-threshold)
pairs as possible, and among those maximum-cardinality pairings, the one
with the smallest total distance. A greedy "take the globally closest pair
first" strategy does not guarantee this: e.g. tracks at 0.00/0.04 and
detections at 0.03/0.08 with threshold 0.05 has greedy grab 0.04->0.03
(distance 0.01) first, stranding 0.08 as a new track, even though
0.00->0.03 and 0.04->0.08 both stay under threshold and would keep both
IDs. `optimal_assignment` solves the actual assignment problem instead.

No `scipy` dependency exists in this project (see `pyproject.toml`), so
this is a small pure-Python O(n^3) Kuhn-Munkres (Hungarian algorithm) --
fast enough at the scale this project ever needs (REQ-42 caps at 50
detections/frame across all classes combined).
"""

from __future__ import annotations

_INF = float("inf")


def _min_cost_perfect_matching(cost: list[list[float]]) -> list[int]:
    """Kuhn-Munkres on a square cost matrix. Returns `assignment` where
    `assignment[i]` is the column matched to row `i`, minimizing total cost.
    """
    n = len(cost)
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)  # p[j] = row currently matched to column j (1-indexed); 0 = none
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        min_to = [_INF] * (n + 1)
        visited = [False] * (n + 1)
        while True:
            visited[j0] = True
            i0 = p[j0]
            delta = _INF
            j1 = -1
            for j in range(1, n + 1):
                if visited[j]:
                    continue
                reduced_cost = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if reduced_cost < min_to[j]:
                    min_to[j] = reduced_cost
                    way[j] = j0
                if min_to[j] < delta:
                    delta = min_to[j]
                    j1 = j
            for j in range(n + 1):
                if visited[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    min_to[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    assignment = [0] * n
    for j in range(1, n + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    return assignment


def optimal_assignment(
    row_count: int, col_count: int, valid_cost: dict[tuple[int, int], float], max_valid_cost: float
) -> dict[int, int]:
    """Match rows to columns, maximizing the number of valid pairs used and,
    among matchings of that size, minimizing the summed cost.

    `valid_cost[(row, col)]` gives the cost of a pair that is allowed to
    match; any `(row, col)` absent from it is never used. `max_valid_cost`
    must be >= every value in `valid_cost` (the caller's distance
    threshold): it sizes the padding that lets a row or column stay
    unmatched instead of being forced into an invalid pair.

    Returns `{row: col}` for exactly the matched pairs; unmatched rows are
    simply absent from the result.
    """
    # Reduction to a square min-cost perfect matching: pad with `col_count`
    # dummy rows and `row_count` dummy columns so every real row/column has
    # a "stay unmatched" escape (dummy-real cost = `unmatched_cost`) instead
    # of ever being forced into an invalid pair (cost = `forbidden_cost`).
    # `unmatched_cost` exceeds every real valid cost and `forbidden_cost`
    # exceeds the total any combination of escapes could cost, so the
    # minimum-cost solution never uses a forbidden pair, and strictly
    # prefers one more valid match over leaving two nodes unmatched.
    size = row_count + col_count
    unmatched_cost = max_valid_cost * 2.0
    forbidden_cost = unmatched_cost * (size + 1) + 1.0

    cost = [[0.0] * size for _ in range(size)]
    for row in range(row_count):
        for col in range(col_count):
            cost[row][col] = valid_cost.get((row, col), forbidden_cost)
        for dummy_col in range(col_count, size):
            cost[row][dummy_col] = unmatched_cost
    for dummy_row in range(row_count, size):
        for col in range(col_count):
            cost[dummy_row][col] = unmatched_cost
        for dummy_col in range(col_count, size):
            cost[dummy_row][dummy_col] = 0.0

    assignment = _min_cost_perfect_matching(cost)
    return {
        row: assignment[row]
        for row in range(row_count)
        if assignment[row] < col_count and (row, assignment[row]) in valid_cost
    }
