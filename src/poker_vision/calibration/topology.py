"""Polygon containment and overlap for table-plane zones (REQ-11).

Pure-Python 2D polygon geometry (no numpy/shapely dependency): point-in-
polygon via ray casting, segment intersection, and the two predicates
`polygon_contains`/`polygons_overlap` that `zones.py` uses to enforce zone
topology. Zones are hand-authored (REQ-10), typically small quadrilaterals,
possibly concave seat wedges around a round table — these functions make no
convexity assumption, but do rely on polygons being simple
(non-self-intersecting), which `TablePolygon`'s own validator enforces
(REQ-11, see `geometry.py`).

`polygons_overlap` decomposes both polygons into triangles (ear clipping,
which always succeeds for a simple polygon) and tests every triangle pair
for positive-area overlap via the separating-axis theorem. This isn't the
simplest possible approach, but a sampling heuristic (test vertices/edge-
midpoints for strict containment, plus edge-crossing) provably cannot work
in general: for two coincident (or merely similar) *concave* polygons,
every sampled point can land exactly on the other polygon's boundary and
every edge pair can be collinear rather than crossing, even though the
polygons fully overlap. Triangulation sidesteps this because triangles are
always convex, and SAT is exact (not sample-based) for convex shapes.
"""

from __future__ import annotations

import math

from poker_vision.calibration.geometry import TablePoint, TablePolygon, polygon_signed_area

# Absolute tolerance for "is this value zero" in cross-product/orientation
# tests below. Table coordinates are authored, not measured, so points
# meant to be collinear/coincident land there exactly or with float noise
# many orders of magnitude smaller than any real zone dimension.
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


def _segments_properly_intersect(
    p1: TablePoint, p2: TablePoint, p3: TablePoint, p4: TablePoint
) -> bool:
    """True if segment p1-p2 crosses segment p3-p4 at an interior point of both.

    Shared endpoints or collinear touching are deliberately excluded (that's
    "adjacent", not "crossing") — those are handled by the caller as
    boundary contact, not as an overlap-causing intersection.
    """
    d1 = _sign(_cross(p3, p4, p1))
    d2 = _sign(_cross(p3, p4, p2))
    d3 = _sign(_cross(p1, p2, p3))
    d4 = _sign(_cross(p1, p2, p4))
    if d1 == 0 or d2 == 0 or d3 == 0 or d4 == 0:
        return False
    return d1 != d2 and d3 != d4


def _point_on_boundary(point: TablePoint, polygon_points: list[TablePoint]) -> bool:
    n = len(polygon_points)
    return any(
        _on_segment(point, polygon_points[i], polygon_points[(i + 1) % n]) for i in range(n)
    )


def _ray_cast_inside(point: TablePoint, polygon_points: list[TablePoint]) -> bool:
    """Even-odd ray-casting test. Unreliable exactly on an edge by design —
    callers must check `_point_on_boundary` separately when that matters."""
    n = len(polygon_points)
    inside = False
    for i in range(n):
        a, b = polygon_points[i], polygon_points[(i + 1) % n]
        if (a.y > point.y) != (b.y > point.y):
            x_at_y = a.x + (point.y - a.y) * (b.x - a.x) / (b.y - a.y)
            if point.x < x_at_y:
                inside = not inside
    return inside


def _point_inside_or_on(point: TablePoint, polygon_points: list[TablePoint]) -> bool:
    return _point_on_boundary(point, polygon_points) or _ray_cast_inside(point, polygon_points)


def _edge_midpoints(points: list[TablePoint]) -> list[TablePoint]:
    """Midpoint of every edge, in addition to the vertices themselves.

    Two convex quads that share a flush edge (e.g. same y-range, overlapping
    x-range) can overlap with positive area while every *vertex* of each
    lands exactly on the other's boundary (a T-junction, not a crossing) —
    vertices alone would then miss the overlap entirely. An edge's midpoint
    can't coincide with the other polygon's boundary in that same way, so
    sampling it too closes the gap without needing full polygon clipping.
    """
    n = len(points)
    midpoints = []
    for i in range(n):
        a, b = points[i], points[(i + 1) % n]
        midpoints.append(TablePoint(x=(a.x + b.x) / 2, y=(a.y + b.y) / 2))
    return midpoints


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
    exactly n - 2 triangles that exactly tile the polygon.
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
            # A genuinely simple polygon always has a clippable ear; bail
            # out rather than loop forever if float noise ever prevents
            # finding one (leaves this triangulation incomplete, which
            # only makes `polygons_overlap` under-detect, never over-).
            break
    if len(remaining) == 3:
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


def polygon_contains(outer: TablePolygon, inner: TablePolygon) -> bool:
    """True if `inner` lies entirely within `outer` (touching `outer`'s boundary is fine)."""
    outer_points, inner_points = outer.points, inner.points
    inner_samples = inner_points + _edge_midpoints(inner_points)
    if not all(_point_inside_or_on(p, outer_points) for p in inner_samples):
        return False
    n_outer, n_inner = len(outer_points), len(inner_points)
    for i in range(n_inner):
        a, b = inner_points[i], inner_points[(i + 1) % n_inner]
        for j in range(n_outer):
            c, d = outer_points[j], outer_points[(j + 1) % n_outer]
            if _segments_properly_intersect(a, b, c, d):
                # All sampled inner points are inside/on outer, yet an inner
                # edge crosses an outer edge: inner pokes out through a
                # concave part of outer, so it isn't fully contained after all.
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
