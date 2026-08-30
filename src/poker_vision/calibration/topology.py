"""Polygon containment and overlap for table-plane zones (REQ-11).

Pure-Python 2D polygon geometry (no numpy/shapely dependency): the two
predicates `polygon_contains`/`polygons_overlap` that `zones.py` uses to
enforce zone topology. Zones are hand-authored (REQ-10), typically small
quadrilaterals, possibly concave seat wedges around a round table — these
functions make no convexity assumption, but do rely on polygons being
simple (non-self-intersecting), which `TablePolygon`'s own validator
enforces (REQ-11, see `geometry.py`).

Both predicates decompose their polygons into triangles (ear clipping,
which always succeeds for a simple polygon) rather than testing sampled
points (vertices, edge midpoints, ...): for a concave polygon there is no
finite set of "representative" sample points that's guaranteed to catch
every violation — see this module's git history for two sampling
heuristics that each looked reasonable and each had a concrete
counterexample. Triangles are always convex, which makes both operations
exact instead: `polygons_overlap` tests every triangle pair for
positive-area overlap via the separating-axis theorem, and
`polygon_contains` clips every inner triangle against every outer triangle
(Sutherland-Hodgman, exact for convex-convex) and checks the clipped area
sums back up to the inner triangle's own area.
"""

from __future__ import annotations

import math

from poker_vision.calibration.geometry import TablePoint, TablePolygon, polygon_signed_area

# Absolute tolerance for "is this value zero" — both in cross-product/
# orientation tests, and as an area (in `polygon_contains`) or projected-
# length (in `polygons_overlap`'s SAT) difference. Table coordinates are
# authored, not measured, so points meant to be collinear/coincident land
# there exactly or with float noise many orders of magnitude smaller than
# any real zone dimension (and hence its area).
_EPSILON = 1e-9


def _cross(o: TablePoint, a: TablePoint, b: TablePoint) -> float:
    """Cross product of (a - o) and (b - o); sign gives turn direction at o."""
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)


def _sign(value: float) -> int:
    if value > _EPSILON:
        return 1
    if value < -_EPSILON:
        return -1
    return 0


def _on_segment(p: TablePoint, a: TablePoint, b: TablePoint) -> bool:
    """True if p is collinear with, and between, a and b (inclusive)."""
    if _sign(_cross(a, b, p)) != 0:
        return False
    return (
        min(a.x, b.x) - _EPSILON <= p.x <= max(a.x, b.x) + _EPSILON
        and min(a.y, b.y) - _EPSILON <= p.y <= max(a.y, b.y) + _EPSILON
    )


def _triangle_strictly_contains(
    a: TablePoint, b: TablePoint, c: TablePoint, p: TablePoint
) -> bool:
    """True if p is strictly inside triangle a-b-c (works for either winding)."""
    d1 = _sign(_cross(a, b, p))
    d2 = _sign(_cross(b, c, p))
    d3 = _sign(_cross(c, a, p))
    if d1 == 0 or d2 == 0 or d3 == 0:
        return False
    return d1 == d2 == d3


def _is_ear(polygon: list[TablePoint], index: int, orientation: int) -> bool:
    """True if polygon[index] is currently a clippable "ear" tip.

    An ear tip must turn the same way as the polygon's overall winding (not
    reflex, not collinear) and its closing triangle must not contain any
    other vertex of the polygon — otherwise "clipping" it would cut off
    part of the polygon that isn't actually this triangle.
    """
    n = len(polygon)
    prev_p = polygon[(index - 1) % n]
    curr_p = polygon[index]
    next_p = polygon[(index + 1) % n]
    if _sign(_cross(prev_p, curr_p, next_p)) != orientation:
        return False
    skip = {(index - 1) % n, index, (index + 1) % n}
    for j, point in enumerate(polygon):
        if j in skip:
            continue
        if _triangle_strictly_contains(prev_p, curr_p, next_p, point) or _on_segment(
            point, prev_p, next_p
        ):
            return False
    return True


def _triangulate(points: list[TablePoint]) -> list[tuple[TablePoint, TablePoint, TablePoint]]:
    """Ear-clipping triangulation. Works for any simple polygon, convex or concave.

    Every simple polygon with n >= 4 vertices has at least one ear (a
    classical result), so repeatedly clipping one always terminates in
    exactly n - 2 triangles that exactly tile the polygon. Raises rather
    than returning a partial (or, worst case, empty) result if that
    theorem ever fails to hold in practice (e.g. floating-point noise on
    a pathological input): `polygon_contains`/`polygons_overlap` treat "no
    triangles to check" as vacuously true/false respectively, so silently
    returning less than a full triangulation would let a genuinely invalid
    zone topology pass REQ-11's supposedly hard validation.
    """
    remaining = list(points)
    orientation = _sign(polygon_signed_area(points))
    triangles: list[tuple[TablePoint, TablePoint, TablePoint]] = []
    while len(remaining) > 3:
        n = len(remaining)
        for i in range(n):
            if _is_ear(remaining, i, orientation):
                prev_p = remaining[(i - 1) % n]
                curr_p = remaining[i]
                next_p = remaining[(i + 1) % n]
                triangles.append((prev_p, curr_p, next_p))
                del remaining[i]
                break
        else:
            raise ValueError(
                "polygon triangulation failed: no clippable ear found for a "
                "polygon that should be simple"
            )
    triangles.append((remaining[0], remaining[1], remaining[2]))
    return triangles


def _unit_edge_normals(
    triangle: tuple[TablePoint, TablePoint, TablePoint],
) -> list[tuple[float, float]]:
    """Unit-length outward-ish normal of each edge — the SAT candidate axes for a triangle."""
    normals = []
    n = len(triangle)
    for i in range(n):
        a, b = triangle[i], triangle[(i + 1) % n]
        dx, dy = b.x - a.x, b.y - a.y
        length = math.hypot(dx, dy)
        if length < _EPSILON:
            continue  # zero-length edge; TablePolygon's own checks rule this out already
        normals.append((-dy / length, dx / length))
    return normals


def _project_onto_axis(
    points: tuple[TablePoint, ...], axis: tuple[float, float]
) -> tuple[float, float]:
    ax, ay = axis
    values = [p.x * ax + p.y * ay for p in points]
    return min(values), max(values)


def _triangles_share_positive_area(
    t1: tuple[TablePoint, TablePoint, TablePoint], t2: tuple[TablePoint, TablePoint, TablePoint]
) -> bool:
    """Separating-axis test, but for *positive-area* overlap rather than mere touching.

    Two convex shapes are disjoint (SAT) iff some edge-normal axis has their
    projections cleanly separated. Using `<=` here (instead of the usual
    strict `<`) additionally rules out projections that only *touch* along
    an axis — if that happens for any axis, the shapes can share at most a
    lower-dimensional boundary (an edge or a point) there, not a 2D region,
    so the intersection has zero area even though the shapes do meet.
    """
    for axis in _unit_edge_normals(t1) + _unit_edge_normals(t2):
        min1, max1 = _project_onto_axis(t1, axis)
        min2, max2 = _project_onto_axis(t2, axis)
        overlap_length = min(max1, max2) - max(min1, min2)
        if overlap_length <= _EPSILON:
            return False
    return True


def _line_intersection(
    p1: TablePoint, p2: TablePoint, p3: TablePoint, p4: TablePoint
) -> TablePoint:
    """Intersection of infinite line p1-p2 with infinite line p3-p4.

    Only ever called from `_clip_convex_by_convex` on a pair known to cross
    (one endpoint of p1-p2 is on each side of p3-p4), so the lines are
    never parallel here and the division is safe.
    """
    x1, y1, x2, y2 = p1.x, p1.y, p2.x, p2.y
    x3, y3, x4, y4 = p3.x, p3.y, p4.x, p4.y
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
    return TablePoint(x=x1 + t * (x2 - x1), y=y1 + t * (y2 - y1))


def _clip_convex_by_convex(
    subject: list[TablePoint], clip: list[TablePoint]
) -> list[TablePoint]:
    """Sutherland-Hodgman: the exact intersection of `subject` with convex `clip`.

    Correct regardless of `subject`'s own winding or convexity (each of
    `clip`'s edges cuts `subject` down to its half-plane in turn); here
    both arguments happen to be triangles, so `subject` is convex too.
    """
    orientation = _sign(polygon_signed_area(clip))
    output = list(subject)
    n = len(clip)
    for i in range(n):
        if not output:
            break
        edge_a, edge_b = clip[i], clip[(i + 1) % n]
        input_points = output
        output = []
        m = len(input_points)
        for j in range(m):
            curr = input_points[j]
            prev = input_points[j - 1]
            curr_inside = _sign(_cross(edge_a, edge_b, curr)) in (0, orientation)
            prev_inside = _sign(_cross(edge_a, edge_b, prev)) in (0, orientation)
            if curr_inside:
                if not prev_inside:
                    output.append(_line_intersection(prev, curr, edge_a, edge_b))
                output.append(curr)
            elif prev_inside:
                output.append(_line_intersection(prev, curr, edge_a, edge_b))
    return output


def polygon_contains(outer: TablePolygon, inner: TablePolygon) -> bool:
    """True if `inner` lies entirely within `outer` (touching `outer`'s boundary is fine)."""
    outer_triangles = _triangulate(outer.points)
    inner_triangles = _triangulate(inner.points)
    for inner_triangle in inner_triangles:
        inner_area = abs(polygon_signed_area(list(inner_triangle)))
        covered_area = 0.0
        for outer_triangle in outer_triangles:
            clipped = _clip_convex_by_convex(list(inner_triangle), list(outer_triangle))
            covered_area += abs(polygon_signed_area(clipped))
        # outer's own triangles never overlap each other (a valid
        # triangulation tiles outer exactly), so summing the clipped area
        # against each of them can't double-count part of inner_triangle.
        if inner_area - covered_area > _EPSILON:
            return False
    return True


def polygons_overlap(a: TablePolygon, b: TablePolygon) -> bool:
    """True if `a` and `b` share a positive-area region.

    Merely touching (a shared edge or vertex, no interior overlap) is not
    considered an overlap. Exact (triangulation + separating-axis test),
    not a sampling heuristic — see the module docstring for why that
    distinction matters for concave zones.
    """
    triangles_a = _triangulate(a.points)
    triangles_b = _triangulate(b.points)
    return any(
        _triangles_share_positive_area(ta, tb) for ta in triangles_a for tb in triangles_b
    )
