"""Polygon containment and overlap for table-plane zones (REQ-11).

Pure-Python 2D polygon geometry (no numpy/shapely dependency): point-in-
polygon via ray casting, segment intersection, and the two predicates
`polygon_contains`/`polygons_overlap` that `zones.py` uses to enforce zone
topology. Zones are hand-authored (REQ-10), typically small quadrilaterals,
possibly concave seat wedges around a round table — these functions make no
convexity assumption, but do rely on polygons being simple
(non-self-intersecting), which `TablePolygon`'s own validator enforces
(REQ-11, see `geometry.py`).
"""

from __future__ import annotations

from poker_vision.calibration.geometry import TablePoint, TablePolygon

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


def _point_strictly_inside(point: TablePoint, polygon_points: list[TablePoint]) -> bool:
    if _point_on_boundary(point, polygon_points):
        return False
    return _ray_cast_inside(point, polygon_points)


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
    considered an overlap.
    """
    a_points, b_points = a.points, b.points
    a_samples = a_points + _edge_midpoints(a_points)
    b_samples = b_points + _edge_midpoints(b_points)
    if any(_point_strictly_inside(p, b_points) for p in a_samples):
        return True
    if any(_point_strictly_inside(p, a_points) for p in b_samples):
        return True
    n_a, n_b = len(a_points), len(b_points)
    for i in range(n_a):
        p1, p2 = a_points[i], a_points[(i + 1) % n_a]
        for j in range(n_b):
            p3, p4 = b_points[j], b_points[(j + 1) % n_b]
            if _segments_properly_intersect(p1, p2, p3, p4):
                return True
    return False
