from poker_vision.calibration.geometry import TablePolygon
from poker_vision.calibration.topology import polygon_contains, polygons_overlap


def _polygon(*coords: tuple[float, float]) -> TablePolygon:
    return TablePolygon(points=[{"x": x, "y": y} for x, y in coords])


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
