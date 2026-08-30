import pytest

from poker_vision.calibration import topology as topology_module
from poker_vision.calibration.geometry import TablePoint, TablePolygon
from poker_vision.calibration.topology import point_in_polygon, polygon_contains, polygons_overlap


def _polygon(*coords: tuple[float, float]) -> TablePolygon:
    return TablePolygon(points=[{"x": x, "y": y} for x, y in coords])


def _point(x: float, y: float) -> TablePoint:
    return TablePoint(x=x, y=y)


SQUARE_0_100 = _polygon((0, 0), (100, 0), (100, 100), (0, 100))


# --- polygon_contains --------------------------------------------------------


def test_contains_polygon_fully_inside():
    inner = _polygon((10, 10), (50, 10), (50, 50), (10, 50))
    assert polygon_contains(SQUARE_0_100, inner) is True


def test_contains_identical_polygon():
    assert polygon_contains(SQUARE_0_100, SQUARE_0_100) is True


def test_contains_polygon_touching_outer_boundary():
    # chip_zone flush against the near edge of player_area is a normal layout.
    inner = _polygon((0, 10), (50, 10), (50, 50), (0, 50))
    assert polygon_contains(SQUARE_0_100, inner) is True


def test_contains_polygon_partially_outside_rejected():
    inner = _polygon((80, 10), (120, 10), (120, 50), (80, 50))
    assert polygon_contains(SQUARE_0_100, inner) is False


def test_contains_polygon_fully_outside_rejected():
    inner = _polygon((200, 200), (250, 200), (250, 250), (200, 250))
    assert polygon_contains(SQUARE_0_100, inner) is False


def test_contains_polygon_pokes_out_through_concave_outer_rejected():
    # "L"-shaped outer (concave at (50, 50)); inner's vertices all sit inside
    # or on the outer, but its edge cuts across the missing corner.
    outer = _polygon((0, 0), (100, 0), (100, 50), (50, 50), (50, 100), (0, 100))
    inner = _polygon((40, 40), (90, 40), (90, 90), (40, 90))
    assert polygon_contains(outer, inner) is False


# Outer with a rectangular notch cut from the top, mouth exactly at y=15;
# a sampling-based check missed cases where an inner edge enters and exits
# through the notch's reflex corner *vertices* rather than a mid-edge point
# (a stale `_segments_properly_intersect`-based check treats a vertex-exact
# touch as "adjacent", not "crossing"). polygon_contains is now exact
# (triangulate + Sutherland-Hodgman clip + area comparison), so it doesn't
# depend on where along an edge the notch happens to be entered.
NOTCH_OUTER = _polygon((0, 0), (20, 0), (20, 20), (12, 20), (12, 15), (8, 15), (8, 20), (0, 20))


def test_contains_inner_vertex_at_both_reflex_corners_is_valid():
    # Inner stays entirely below the notch (y <= 15); touching both reflex
    # corners exactly is a legitimate flush layout, not a violation.
    inner = _polygon((2, 10), (18, 10), (12, 15), (8, 15))
    assert polygon_contains(NOTCH_OUTER, inner) is True


def test_contains_inner_pokes_through_notch_via_reflex_vertex_rejected():
    # One inner vertex sits exactly on the notch's left reflex corner
    # (8, 15); from there the inner boundary pokes up into the excluded
    # notch area before coming back down.
    inner = _polygon((2, 10), (18, 10), (18, 17), (8, 15), (2, 17))
    assert polygon_contains(NOTCH_OUTER, inner) is False


def test_contains_edge_dips_into_off_center_notch_rejected():
    # This edge's own midpoint (7, 17) sits outside the notch's x-range
    # ([8, 12]), so a midpoint-only sample would miss the violation — the
    # edge still dips into the excluded notch area near its other end.
    inner = _polygon((1, 17), (13, 17), (13, 10), (1, 10))
    assert polygon_contains(NOTCH_OUTER, inner) is False


# --- point_in_polygon (REQ-26) -------------------------------------------------


def test_point_in_polygon_strictly_inside():
    assert point_in_polygon(SQUARE_0_100, _point(50, 50)) is True


def test_point_in_polygon_strictly_outside():
    assert point_in_polygon(SQUARE_0_100, _point(150, 50)) is False


def test_point_in_polygon_on_boundary_edge_counts_as_inside():
    assert point_in_polygon(SQUARE_0_100, _point(0, 50)) is True


def test_point_in_polygon_on_vertex_counts_as_inside():
    assert point_in_polygon(SQUARE_0_100, _point(0, 0)) is True


def test_point_in_polygon_just_outside_boundary_is_outside():
    assert point_in_polygon(SQUARE_0_100, _point(-0.5, 50)) is False


def test_point_in_polygon_in_concave_notch_is_outside():
    # U_SHAPE's notch (x in [1, 3], y in [1, 4]) isn't part of the polygon;
    # a point there must not register as inside even though it sits within
    # the polygon's bounding box.
    assert point_in_polygon(U_SHAPE, _point(2, 2)) is False


def test_point_in_polygon_on_solid_leg_of_concave_shape_is_inside():
    assert point_in_polygon(U_SHAPE, _point(0.5, 2)) is True


def test_point_in_polygon_propagates_triangulation_failure(monkeypatch):
    monkeypatch.setattr(topology_module, "_is_ear", lambda polygon, index, orientation: False)
    with pytest.raises(ValueError, match="triangulation failed"):
        point_in_polygon(SQUARE_0_100, _point(50, 50))


# --- polygons_overlap ---------------------------------------------------------


def test_overlap_disjoint_polygons():
    a = _polygon((0, 0), (10, 0), (10, 10), (0, 10))
    b = _polygon((200, 200), (210, 200), (210, 210), (200, 210))
    assert polygons_overlap(a, b) is False


def test_overlap_partial_with_vertex_inside():
    a = _polygon((0, 0), (60, 0), (60, 60), (0, 60))
    b = _polygon((30, 30), (90, 30), (90, 90), (30, 90))
    assert polygons_overlap(a, b) is True


def test_overlap_coincident_polygons():
    # E.g. a chip_zone accidentally copy-pasted for two different seats:
    # every vertex/edge-midpoint of each lands exactly on the other's
    # boundary and every edge pair is collinear rather than crossing, so a
    # sampling heuristic would see nothing but touching.
    a = _polygon((10, 10), (50, 10), (50, 50), (10, 50))
    b = _polygon((10, 10), (50, 10), (50, 50), (10, 50))
    assert polygons_overlap(a, b) is True


U_SHAPE = _polygon((0, 0), (4, 0), (4, 4), (3, 4), (3, 1), (1, 1), (1, 4), (0, 4))


def test_overlap_coincident_concave_polygons():
    # A concave "U"/comb shape whose own vertex-average centroid falls
    # *outside* it (in the notch) — any sampling heuristic keyed on a
    # vertex/midpoint/centroid "representative point" can miss this exact
    # case; the triangulation-based implementation doesn't rely on picking
    # a representative point at all.
    assert polygons_overlap(U_SHAPE, U_SHAPE) is True


def test_overlap_rectangle_in_concave_notch_is_not_overlap():
    # U_SHAPE's notch (the area cut out of the top, x in [1, 3] / y in
    # [1, 4]) isn't part of the polygon at all — a rectangle sitting
    # entirely in that notch must not register as overlapping it.
    in_notch = _polygon((1.5, 2), (2.5, 2), (2.5, 3), (1.5, 3))
    assert polygons_overlap(U_SHAPE, in_notch) is False


def test_overlap_rectangle_straddling_concave_leg_overlaps():
    # Half of this rectangle sits on U_SHAPE's solid left leg (x in [0, 1]),
    # half sits in the notch — genuine partial overlap with a concave shape.
    on_leg = _polygon((0.5, 2), (1.5, 2), (1.5, 3), (0.5, 3))
    assert polygons_overlap(U_SHAPE, on_leg) is True


def test_overlap_one_fully_inside_other():
    inner = _polygon((10, 10), (50, 10), (50, 50), (10, 50))
    assert polygons_overlap(SQUARE_0_100, inner) is True
    assert polygons_overlap(inner, SQUARE_0_100) is True


def test_overlap_touching_edge_only_is_not_overlap():
    a = _polygon((0, 0), (10, 0), (10, 10), (0, 10))
    b = _polygon((10, 0), (20, 0), (20, 10), (10, 10))
    assert polygons_overlap(a, b) is False


def test_overlap_touching_corner_only_is_not_overlap():
    a = _polygon((0, 0), (10, 0), (10, 10), (0, 10))
    b = _polygon((10, 10), (20, 10), (20, 20), (10, 20))
    assert polygons_overlap(a, b) is False


def test_overlap_flush_shared_edge_range_with_partial_x_overlap():
    # Same y-range on both rectangles: every vertex of each lands exactly on
    # the other's boundary (T-junctions, not crossings) even though a real
    # x in [60, 90] / y in [10, 50] region is shared by both.
    a = _polygon((40, 10), (90, 10), (90, 50), (40, 50))
    b = _polygon((60, 10), (110, 10), (110, 50), (60, 50))
    assert polygons_overlap(a, b) is True


def test_overlap_crossing_without_contained_vertices():
    # Plus-sign arrangement: a wide-short rect crossing a narrow-tall rect;
    # neither has a vertex inside the other, only edges cross.
    a = _polygon((0, 4), (10, 4), (10, 6), (0, 6))
    b = _polygon((4, 0), (6, 0), (6, 10), (4, 10))
    assert polygons_overlap(a, b) is True


# --- _triangulate failure handling --------------------------------------------


def test_triangulate_raises_instead_of_returning_incomplete_result(monkeypatch):
    # If ear-clipping ever fails to find a clippable ear (float noise on a
    # pathological input, or a latent bug in _is_ear), the old behavior was
    # to silently return whatever triangles had been found so far — up to
    # and including an empty list. Force that path and confirm it now
    # raises instead of returning a partial/empty triangulation.
    monkeypatch.setattr(topology_module, "_is_ear", lambda polygon, index, orientation: False)
    with pytest.raises(ValueError, match="triangulation failed"):
        topology_module._triangulate(SQUARE_0_100.points)


def test_polygon_contains_propagates_triangulation_failure(monkeypatch):
    # Before the fix: an empty inner triangulation meant the "does every
    # inner triangle fit inside outer" loop never ran, so polygon_contains
    # returned True unconditionally — even for an inner polygon nowhere
    # near outer. Force _triangulate's real ear-search into its failure
    # branch (rather than replacing _triangulate itself, which would just
    # bypass the fix) and confirm the failure propagates instead of
    # silently passing.
    monkeypatch.setattr(topology_module, "_is_ear", lambda polygon, index, orientation: False)
    far_away = _polygon((200, 200), (210, 200), (210, 210), (200, 210))
    with pytest.raises(ValueError, match="triangulation failed"):
        polygon_contains(SQUARE_0_100, far_away)


def test_polygons_overlap_propagates_triangulation_failure(monkeypatch):
    # Before the fix: an empty triangulation on either side meant the "do
    # any triangle pair overlap" check ran over zero pairs, so
    # polygons_overlap returned False unconditionally — even for two
    # identical polygons. Confirm the failure propagates instead of
    # silently under-reporting.
    monkeypatch.setattr(topology_module, "_is_ear", lambda polygon, index, orientation: False)
    with pytest.raises(ValueError, match="triangulation failed"):
        polygons_overlap(SQUARE_0_100, SQUARE_0_100)
