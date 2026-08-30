"""`calibration.geometry` helpers shared across modules (REQ-27, REQ-28, REQ-37)."""

from __future__ import annotations

import pytest

from poker_vision.calibration.geometry import TablePoint, TablePolygon, polygon_centroid


def _polygon(*coords: tuple[float, float]) -> TablePolygon:
    return TablePolygon(points=[TablePoint(x=x, y=y) for x, y in coords])


# --- polygon_centroid: area-weighted, not a vertex average ------------------


def test_centroid_of_rectangle_matches_vertex_average():
    # For a rectangle, the area-weighted centroid and the plain vertex
    # average coincide -- this is the case a naive vertex-average
    # implementation would also get right.
    rectangle = _polygon((0, 0), (10, 0), (10, 4), (0, 4))
    centroid = polygon_centroid(rectangle)
    assert centroid.x == pytest.approx(5.0)
    assert centroid.y == pytest.approx(2.0)


def test_centroid_uses_area_weighting_not_vertex_average():
    # Triangle (0,0)-(0,12)-(4,0) with an extra vertex (2,6) added exactly
    # on the hypotenuse (collinear, no effect on the actual shape). Its true
    # (area-weighted) centroid is the average of the *triangle's own three*
    # vertices, (4/3, 4) -- but naively averaging all four listed points
    # gives (1.5, 4.5) instead. A centroid that shifts just from adding a
    # collinear vertex to an unchanged shape could flip REQ-28's nearest-seat
    # tie-break, or REQ-37's rubber-band anchor, for reasons that have
    # nothing to do with the polygon's actual geometry.
    triangle_with_collinear_vertex = _polygon((0, 0), (0, 12), (2, 6), (4, 0))
    centroid = polygon_centroid(triangle_with_collinear_vertex)
    assert centroid.x == pytest.approx(4 / 3)
    assert centroid.y == pytest.approx(4.0)
