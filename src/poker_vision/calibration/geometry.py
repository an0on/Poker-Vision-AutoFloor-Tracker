"""Geometry primitives shared by the calibration schemas (REQ-4, REQ-7).

Points and polygons come in two flavors: `PixelPoint` (raw image pixel
space, used only for homography source correspondences) and `TablePoint`
(the table plane's own coordinate system, used everywhere geometry is
actually decided — REQ-5). Nothing below `detection` may accept pixel
coordinates; that boundary is enforced by which type a field declares.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated

from pydantic import AfterValidator, Field, model_validator

from poker_vision.schema_base import StrictModel

# Below this, a polygon/matrix is treated as degenerate/singular rather than
# merely numerically noisy (REQ-11).
_AREA_EPSILON = 1e-9


class PixelPoint(StrictModel):
    """A point in raw image pixel space (origin top-left, x right, y down)."""

    x: float = Field(allow_inf_nan=False)
    y: float = Field(allow_inf_nan=False)


class TablePoint(StrictModel):
    """A point in table-plane coordinates, in the table's own unit (see `TableDimensions.unit`)."""

    x: float = Field(allow_inf_nan=False)
    y: float = Field(allow_inf_nan=False)


def polygon_signed_area(points: list[TablePoint]) -> float:
    """Shoelace signed area. Near zero means collinear/duplicate points (degenerate)."""
    total = 0.0
    n = len(points)
    for i in range(n):
        a, b = points[i], points[(i + 1) % n]
        total += a.x * b.y - b.x * a.y
    return total / 2.0


def _cross(o: TablePoint, a: TablePoint, b: TablePoint) -> float:
    """Cross product of (a - o) and (b - o); sign gives turn direction at o."""
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)


def _sign(value: float) -> int:
    if value > _AREA_EPSILON:
        return 1
    if value < -_AREA_EPSILON:
        return -1
    return 0


def _on_segment(p: TablePoint, a: TablePoint, b: TablePoint) -> bool:
    """True if p is collinear with, and between, a and b (inclusive)."""
    if _sign(_cross(a, b, p)) != 0:
        return False
    return (
        min(a.x, b.x) - _AREA_EPSILON <= p.x <= max(a.x, b.x) + _AREA_EPSILON
        and min(a.y, b.y) - _AREA_EPSILON <= p.y <= max(a.y, b.y) + _AREA_EPSILON
    )


def _segments_intersect(p1: TablePoint, p2: TablePoint, p3: TablePoint, p4: TablePoint) -> bool:
    """True if segment p1-p2 and segment p3-p4 share any point (crossing or touching)."""
    d1 = _sign(_cross(p3, p4, p1))
    d2 = _sign(_cross(p3, p4, p2))
    d3 = _sign(_cross(p1, p2, p3))
    d4 = _sign(_cross(p1, p2, p4))
    if d1 != d2 and d3 != d4 and d1 != 0 and d2 != 0 and d3 != 0 and d4 != 0:
        return True
    if d1 == 0 and _on_segment(p1, p3, p4):
        return True
    if d2 == 0 and _on_segment(p2, p3, p4):
        return True
    if d3 == 0 and _on_segment(p3, p1, p2):
        return True
    return d4 == 0 and _on_segment(p4, p1, p2)


def _is_simple_polygon(points: list[TablePoint]) -> bool:
    """False if any two non-adjacent edges touch or cross.

    Point-in-polygon, containment and overlap (see `topology.py`) are only
    well-defined for simple polygons; a self-intersecting "bowtie" can have
    nonzero net (shoelace) area, so it isn't caught by the area check alone.
    """
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue  # adjacent edges legitimately share one endpoint
            if _segments_intersect(points[i], points[(i + 1) % n], points[j], points[(j + 1) % n]):
                return False
    return True


def _points_coincide(a: TablePoint, b: TablePoint) -> bool:
    return abs(a.x - b.x) < _AREA_EPSILON and abs(a.y - b.y) < _AREA_EPSILON


def _adjacent_edges_are_valid(prev: TablePoint, curr: TablePoint, next_: TablePoint) -> bool:
    """False if edge prev->curr and edge curr->next overlap along more than their shared point.

    `_is_simple_polygon` deliberately skips adjacent edge pairs (sharing one
    endpoint is normal), so a duplicate vertex (curr == prev or curr ==
    next_, a zero-length edge) or a backtrack (prev, curr, next_ collinear
    with next_ heading back towards prev) needs its own direct check.
    """
    if _points_coincide(prev, curr) or _points_coincide(curr, next_):
        return False
    if _sign(_cross(prev, curr, next_)) != 0:
        return True  # not collinear, so can't overlap beyond the shared point
    incoming = (curr.x - prev.x, curr.y - prev.y)
    outgoing = (next_.x - curr.x, next_.y - curr.y)
    return incoming[0] * outgoing[0] + incoming[1] * outgoing[1] > 0


def _has_valid_adjacent_edges(points: list[TablePoint]) -> bool:
    n = len(points)
    return all(
        _adjacent_edges_are_valid(points[(i - 1) % n], points[i], points[(i + 1) % n])
        for i in range(n)
    )


class TablePolygon(StrictModel):
    """A closed polygon in table coordinates.

    Points are listed in order; the closing edge (last point back to the
    first) is implicit and must not be repeated.
    """

    points: list[TablePoint] = Field(min_length=3)

    @model_validator(mode="after")
    def _check_valid_polygon(self) -> TablePolygon:
        # REQ-11: zero (or near-zero) area means the points are collinear or
        # duplicated, i.e. the polygon doesn't enclose any actual area.
        if abs(polygon_signed_area(self.points)) < _AREA_EPSILON:
            raise ValueError("polygon is degenerate: zero area (collinear or duplicate points)")
        # REQ-11: a duplicate consecutive vertex or a backtracking edge pair
        # (three consecutive collinear points doubling back on themselves)
        # is already rejected indirectly by the checks below — it forces
        # some other, non-adjacent edge pair to share an exact point, which
        # `_is_simple_polygon` catches — but checking it directly here is a
        # much clearer, more specific error for what's a common authoring
        # typo, rather than relying on that indirect interaction.
        if not _has_valid_adjacent_edges(self.points):
            raise ValueError(
                "polygon is invalid: adjacent edges overlap (duplicate vertex or "
                "backtracking edge)"
            )
        # REQ-11: a self-intersecting polygon isn't "closed" in any usable
        # sense — topology.py's containment/overlap checks assume simple
        # polygons, and ray casting has no well-defined answer otherwise.
        if not _is_simple_polygon(self.points):
            raise ValueError("polygon is invalid: edges self-intersect (not a simple polygon)")
        return self


class TableUnit(StrEnum):
    MM = "mm"
    CM = "cm"
    M = "m"


class TableDimensions(StrictModel):
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    unit: TableUnit


def _check_3x3(value: list[list[float]]) -> list[list[float]]:
    if len(value) != 3 or any(len(row) != 3 for row in value):
        raise ValueError("expected a 3x3 matrix (3 rows of 3 floats)")
    # REQ-11: NaN/inf would silently defeat HomographyMatrix's invertibility
    # check downstream (e.g. `abs(nan - expected) > epsilon` is always
    # False), so reject them here at the type boundary instead.
    if any(not math.isfinite(entry) for row in value for entry in row):
        raise ValueError("matrix entries must be finite (no NaN/inf)")
    return value


Matrix3x3 = Annotated[list[list[float]], AfterValidator(_check_3x3)]


def matrix3x3_multiply(left: Matrix3x3, right: Matrix3x3) -> list[list[float]]:
    return [
        [sum(left[row][k] * right[k][col] for k in range(3)) for col in range(3)]
        for row in range(3)
    ]
