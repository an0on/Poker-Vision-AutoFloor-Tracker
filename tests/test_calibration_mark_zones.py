"""REQ-10a: `calib mark-zones`' pure click-to-authoring geometry."""

from __future__ import annotations

import math

import pytest

from poker_vision.calibration.geometry import polygon_signed_area
from poker_vision.calibration.mark_zones import (
    MarkedZones,
    build_authoring_from_marked_zones,
    number_seats_clockwise,
)
from poker_vision.calibration.topology import polygon_contains

# --- number_seats_clockwise --------------------------------------------------

# Four seats at the compass points around (0, 0): a small square polygon at
# each so every one is a valid, distinct player_area. In y-down pixel space
# (see PixelPoint's docstring), N -> E -> S -> W is the visually clockwise
# order a viewer sees -- verified independently by hand for this fixture in
# the PR that added it (atan2 offsets from "N": 0, pi/2, pi, 3pi/2).
def _square(cx: float, cy: float, half: float = 1.0) -> list[tuple[float, float]]:
    return [
        (cx - half, cy - half),
        (cx + half, cy - half),
        (cx + half, cy + half),
        (cx - half, cy + half),
    ]


COMPASS_SEATS = {
    "north": _square(0, -10),
    "east": _square(10, 0),
    "south": _square(0, 10),
    "west": _square(-10, 0),
}


def test_number_seats_clockwise_from_dealer_north():
    seat_ids = number_seats_clockwise(COMPASS_SEATS, dealer_seat_key="north")
    assert seat_ids == {"east": "seat_1", "south": "seat_2", "west": "seat_3", "north": "seat_4"}


def test_number_seats_clockwise_dealer_always_last():
    # Same four seats, different dealer -- the dealer key must still land on
    # the highest seat number (REQ-7: "seat_10" in the real 10-seat case).
    seat_ids = number_seats_clockwise(COMPASS_SEATS, dealer_seat_key="south")
    assert seat_ids["south"] == "seat_4"
    assert seat_ids == {"west": "seat_1", "north": "seat_2", "east": "seat_3", "south": "seat_4"}


def test_number_seats_clockwise_unknown_dealer_key_rejected():
    with pytest.raises(ValueError, match="not one of the marked seats"):
        number_seats_clockwise(COMPASS_SEATS, dealer_seat_key="does_not_exist")


def test_number_seats_clockwise_too_few_seats_rejected():
    with pytest.raises(ValueError, match="at least"):
        number_seats_clockwise({"a": _square(0, 0), "b": _square(5, 5)}, dealer_seat_key="a")


def test_number_seats_clockwise_independent_of_dict_order():
    # Scrambled insertion order must not change the result -- ordering comes
    # from each polygon's own centroid, not click/dict order.
    scrambled = {
        "west": COMPASS_SEATS["west"],
        "north": COMPASS_SEATS["north"],
        "east": COMPASS_SEATS["east"],
        "south": COMPASS_SEATS["south"],
    }
    assert number_seats_clockwise(scrambled, "north") == number_seats_clockwise(
        COMPASS_SEATS, "north"
    )


# --- build_authoring_from_marked_zones --------------------------------------


def _small_marked_zones(inner_oval_points: list[tuple[float, float]] | None = None) -> MarkedZones:
    # A minimal, plausible 4-seat "table": seats far enough apart and small
    # enough that even a generous inset can never collide with a neighbor's
    # chip_zone, and the board_zone/inner_oval sit well clear of all four.
    seats = {
        "north": _square(0, -100, half=20),
        "east": _square(100, 0, half=20),
        "south": _square(0, 100, half=20),
        "west": _square(-100, 0, half=20),
    }
    board_zone = [(-10, -10), (10, -10), (10, 10), (-10, 10)]
    if inner_oval_points is None:
        inner_oval_points = [(-50, -50), (50, -50), (50, 50), (-50, 50)]
    return MarkedZones(
        seat_polygons=seats,
        dealer_seat_key="north",
        inner_oval_points=inner_oval_points,
        board_zone_points=board_zone,
        image_size=(2000, 2000),
    )


def test_build_authoring_produces_valid_calibration():
    authoring = build_authoring_from_marked_zones(_small_marked_zones(), table_id="test_table")
    assert {s.seat_id for s in authoring.seats} == {"seat_1", "seat_2", "seat_3", "seat_4"}
    assert authoring.inference_resolution.width == 2000
    assert authoring.table.width == 2000.0


def test_build_authoring_card_dealer_seat_id_is_the_marked_seat():
    authoring = build_authoring_from_marked_zones(_small_marked_zones(), table_id="test_table")
    # "north" was marked as dealer -> lands on seat_4 (highest number, see
    # number_seats_clockwise's docstring), and card_dealer_seat_id must
    # point at exactly that seat.
    assert authoring.card_dealer_seat_id == "seat_4"
    north_seat = next(s for s in authoring.seats if s.seat_id == "seat_4")
    assert polygon_signed_area(north_seat.zones.player_area.points) != 0


def test_build_authoring_dealer_area_is_the_clicked_inner_oval_trace():
    inner_oval = [(-50, -50), (50, -50), (50, 50), (-50, 50)]
    authoring = build_authoring_from_marked_zones(
        _small_marked_zones(inner_oval), table_id="t"
    )
    assert [(p.x, p.y) for p in authoring.zones.dealer_area.points] == inner_oval


def test_build_authoring_chip_zone_inset_pixels_is_configurable():
    default = build_authoring_from_marked_zones(_small_marked_zones(), table_id="t")
    inset_more = build_authoring_from_marked_zones(
        _small_marked_zones(), table_id="t", chip_zone_inset_pixels=15.0
    )

    def chip_zone_area(authoring, seat_id):
        seat = next(s for s in authoring.seats if s.seat_id == seat_id)
        return abs(polygon_signed_area(seat.zones.chip_zone.points))

    assert chip_zone_area(inset_more, "seat_1") < chip_zone_area(default, "seat_1")


def test_build_authoring_chip_zone_stays_flush_with_rail_edge():
    # "north"'s rail edge is its y=-120 top edge (facing away from the table
    # centroid at (0, 0)) -- the inset must never pull chip_zone away from
    # it (zero margin toward the rail, since players stack chips right up
    # against it); only the side/inner edges get inset.
    authoring = build_authoring_from_marked_zones(
        _small_marked_zones(), table_id="t", chip_zone_inset_pixels=5.0
    )
    north_seat = next(s for s in authoring.seats if s.seat_id == "seat_4")
    chip_min_y = min(p.y for p in north_seat.zones.chip_zone.points)
    player_min_y = min(p.y for p in north_seat.zones.player_area.points)
    assert chip_min_y == pytest.approx(player_min_y)


def test_build_authoring_is_deterministic():
    a = build_authoring_from_marked_zones(_small_marked_zones(), table_id="t")
    b = build_authoring_from_marked_zones(_small_marked_zones(), table_id="t")
    assert a.model_dump() == b.model_dump()


def test_build_authoring_homography_is_identity_from_image_corners():
    authoring = build_authoring_from_marked_zones(_small_marked_zones(), table_id="t")
    width, height = 2000.0, 2000.0
    expected = {(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)}
    actual = {(c.image_point.x, c.image_point.y) for c in authoring.homography.points}
    assert actual == expected
    for correspondence in authoring.homography.points:
        assert correspondence.image_point.x == correspondence.table_point.x
        assert correspondence.image_point.y == correspondence.table_point.y


def test_build_authoring_rejects_negative_chip_zone_inset_pixels():
    with pytest.raises(ValueError, match="chip_zone_inset_pixels"):
        build_authoring_from_marked_zones(
            _small_marked_zones(), table_id="t", chip_zone_inset_pixels=-1.0
        )


def test_build_authoring_accepts_chip_zone_inset_pixels_of_zero():
    # chip_zone == player_area is unusual but valid (REQ-11 allows touching
    # boundaries) -- 0 is the lower edge of the accepted range, not past it.
    authoring = build_authoring_from_marked_zones(
        _small_marked_zones(), table_id="t", chip_zone_inset_pixels=0.0
    )
    seat = next(s for s in authoring.seats if s.seat_id == "seat_1")
    player_points = {(p.x, p.y) for p in seat.zones.player_area.points}
    chip_points = {(p.x, p.y) for p in seat.zones.chip_zone.points}
    assert chip_points == player_points


def test_build_authoring_chip_zone_offsets_a_collinear_vertex_in_place():
    # Three consecutive collinear clicks along a locally-straight stretch
    # of a freehand trace are normal input (not a special case the operator
    # has to avoid): the two edges meeting at the middle point are then
    # parallel, and `_line_intersection` can't find a crossing for them.
    # An earlier version's fallback answered with one of the *lines'* own
    # anchor points instead of the actual vertex, silently folding the
    # middle point back towards its neighbour instead of offsetting it in
    # place. The middle point here (0, 50) is on the seat's inner edge (a
    # non-rail edge, inset by 5px) -- its chip_zone counterpart must move
    # straight along that edge's own normal, landing at x == 0, not drift
    # toward the x == 50 or x == -50 of its neighbours.
    collinear_seat = [(-50.0, -50.0), (50.0, -50.0), (50.0, 50.0), (0.0, 50.0), (-50.0, 50.0)]
    # "north"/"east" are placed only to make the table centroid land at
    # (0, 10030): far enough below "collinear" (centroid (0, 30)) that its
    # top edge (y=-50) unambiguously classifies as the rail edge.
    seats = {
        "north": _square(-2000, 15030, half=20),
        "east": _square(2000, 15030, half=20),
        "collinear": collinear_seat,
    }
    marked = MarkedZones(
        seat_polygons=seats,
        dealer_seat_key="north",
        inner_oval_points=[(-100, -100), (100, -100), (100, 100), (-100, 100)],
        board_zone_points=[(3000, -3000), (3020, -3000), (3020, -2980), (3000, -2980)],
        image_size=(20000, 20000),
    )
    authoring = build_authoring_from_marked_zones(marked, table_id="t", chip_zone_inset_pixels=5.0)
    collinear_zone = next(s for s in authoring.seats if len(s.zones.player_area.points) == 5)
    # Point order is preserved end to end (no stage in this pipeline
    # reorders a polygon's points), so index 3 is still the collinear
    # seat's own click index 3 -- (0, 50) in player_area.
    middle_point = collinear_zone.zones.chip_zone.points[3]
    assert middle_point.x == pytest.approx(0.0, abs=0.5)


def test_build_authoring_chip_zone_handles_a_concave_seat():
    # A concave (non-star-shaped) player_area with a notch cut into one
    # side -- REQ-11 explicitly allows this, but an earlier version of
    # `_derive_chip_zone` picked each edge's outward direction by comparing
    # it against the seat's own (vertex-average) centroid, which isn't
    # guaranteed to sit on the interior side of every edge of a concave
    # shape. That reversed the notch edges' normals, offsetting them
    # *outward* instead of inward and producing a self-intersecting
    # chip_zone. Edge orientation must come from the whole polygon's
    # winding instead (a global, shape-independent fact), not a per-edge
    # comparison against a single reference point.
    notched = [(0, 0), (10, 0), (10, 10), (6, 10), (6, 3), (4, 3), (4, 10), (0, 10)]
    seats = {
        "north": _square(0, -500, half=20),
        "east": _square(500, 0, half=20),
        "notched": notched,
    }
    marked = MarkedZones(
        seat_polygons=seats,
        dealer_seat_key="north",
        inner_oval_points=[(-100, -100), (100, -100), (100, 100), (-100, 100)],
        board_zone_points=[(3000, -3000), (3020, -3000), (3020, -2980), (3000, -2980)],
        image_size=(4000, 4000),
    )
    authoring = build_authoring_from_marked_zones(marked, table_id="t", chip_zone_inset_pixels=1.0)
    notched_seat = next(s for s in authoring.seats if len(s.zones.player_area.points) == 8)
    player_area = notched_seat.zones.player_area
    chip_zone = notched_seat.zones.chip_zone
    assert polygon_contains(player_area, chip_zone)


def test_build_authoring_chip_zone_bounds_a_sharp_corner_via_miter_limit():
    # "spike"'s apex is a ~7-degree needle tip between its two (non-rail)
    # side edges. A naive per-vertex offset -- intersecting both edges'
    # own inset lines -- blows up for a sharp angle like this (miter
    # length = offset / sin(half-angle), here ~10 / sin(3.6 deg) =~ 160px
    # for a nominal 10px inset): the same failure mode confirmed on the
    # real reference table, where one seat's apex moved 82px off a 5px
    # inset before the miter-limit bevel fallback was added. With the
    # fallback, the apex must stay close to its original position.
    seats = {
        "north": _square(0, -500, half=20),
        "east": _square(500, 0, half=20),
        "spike": [(-50.0, 600.0), (50.0, 600.0), (0.0, -200.0)],
    }
    marked = MarkedZones(
        seat_polygons=seats,
        dealer_seat_key="north",
        inner_oval_points=[(-100, -100), (100, -100), (100, 100), (-100, 100)],
        board_zone_points=[(3000, -3000), (3020, -3000), (3020, -2980), (3000, -2980)],
        image_size=(4000, 4000),
    )
    authoring = build_authoring_from_marked_zones(
        marked, table_id="t", chip_zone_inset_pixels=10.0
    )
    spike_seat = next(s for s in authoring.seats if len(s.zones.player_area.points) == 3)
    apex = next(p for p in spike_seat.zones.chip_zone.points if p.y < 0)
    assert math.hypot(apex.x - 0.0, apex.y - (-200.0)) < 20.0


# Real player_area click data from the actual DOPO POKER reference table
# (calib mark-zones session), reused verbatim for the regression test
# below -- only the real table's own point layout (and the real
# table-wide centroid it implies) reproduces the exact failure being
# guarded against; a small synthetic approximation didn't.
_REAL_TABLE_SEAT_POLYGONS: dict[str, list[tuple[float, float]]] = {
    "seat_5": [
        (1586.88, 662.4), (1788.48, 1036.8), (2350.08, 1028.16), (2540.16, 630.72),
        (1589.76, 659.52),
    ],
    "seat_6": [
        (2545.92, 639.36), (2352.96, 1025.28), (3084.48, 1008.0), (3340.8, 682.56),
        (3263.04, 648.0), (3205.44, 630.72), (3127.68, 624.96), (3061.44, 624.96),
        (2557.44, 633.6),
    ],
    "seat_7": [
        (3335.04, 676.8), (3084.48, 1002.24), (3188.16, 1028.16), (3245.76, 1059.84),
        (3317.76, 1140.48), (3363.84, 1241.28), (3375.36, 1324.8), (3752.64, 1319.04),
        (3755.52, 1238.4), (3738.24, 1137.6), (3712.32, 1062.72), (3677.76, 984.96),
        (3640.32, 924.48), (3591.36, 852.48), (3510.72, 789.12), (3456.0, 740.16),
        (3392.64, 702.72), (3343.68, 679.68),
    ],
    "seat_8": [
        (3749.76, 1321.92), (3375.36, 1327.68), (3360.96, 1425.6), (3320.64, 1500.48),
        (3257.28, 1572.48), (3176.64, 1627.2), (3127.68, 1641.6), (3075.84, 1650.24),
        (3317.76, 1955.52), (3438.72, 1897.92), (3533.76, 1814.4), (3631.68, 1707.84),
        (3695.04, 1578.24), (3729.6, 1480.32), (3752.64, 1376.64), (3755.52, 1327.68),
    ],
    "seat_9": [
        (3320.64, 1952.64), (3075.84, 1650.24), (2367.36, 1664.64), (2545.92, 2016.0),
        (3052.8, 2021.76), (3182.4, 1998.72), (3326.4, 1958.4),
    ],
    "seat_10": [
        (2548.8, 2018.88), (2358.72, 1661.76), (1794.24, 1670.4), (1618.56, 2024.64),
        (2540.16, 2021.76),
    ],
    "seat_1": [
        (1618.56, 2021.76), (1791.36, 1673.28), (1126.08, 1676.16), (910.08, 1972.8),
        (1013.76, 2013.12), (1114.56, 2021.76), (1618.56, 2024.64),
    ],
    "seat_2": [
        (910.08, 1967.04), (1123.2, 1676.16), (1054.08, 1664.64), (984.96, 1638.72),
        (927.36, 1595.52), (884.16, 1546.56), (855.36, 1483.2), (835.2, 1422.72),
        (829.44, 1373.76), (492.48, 1376.64), (501.12, 1494.72), (541.44, 1612.8),
        (587.52, 1702.08), (650.88, 1785.6), (717.12, 1848.96), (777.6, 1906.56),
        (904.32, 1975.68),
    ],
    "seat_3": [
        (498.24, 1379.52), (832.32, 1368.0), (838.08, 1296.0), (869.76, 1215.36),
        (915.84, 1143.36), (959.04, 1105.92), (1025.28, 1065.6), (1100.16, 1054.08),
        (843.84, 737.28), (748.8, 797.76), (668.16, 875.52), (601.92, 950.4),
        (552.96, 1033.92), (529.92, 1105.92), (504.0, 1212.48), (492.48, 1310.4),
        (495.36, 1382.4),
    ],
    "seat_4": [
        (849.6, 745.92), (1100.16, 1051.2), (1785.6, 1039.68), (1586.88, 659.52),
        (1126.08, 668.16), (1025.28, 682.56), (950.4, 702.72), (887.04, 722.88),
        (855.36, 743.04),
    ],
}


def test_build_authoring_rejects_near_duplicate_click_with_actionable_message():
    # Two of "seat_5"'s player_area points (its own last two, closing the
    # polygon back to the start) are only ~4px apart, an accidental
    # double-click while tracing the seat. A default-sized inset (10px)
    # then swings the derived corner past unrelated, far-away geometry
    # elsewhere in the same polygon -- a genuine invalid chip_zone, not
    # something a smaller inset alone would fix (contrast the miter-limit
    # test above: that one *is* just a sharp corner). The tool must raise
    # a clear, actionable error naming the seat instead of a bare pydantic
    # validation traceback. (The real table's own click data has this same
    # problem in more than one seat -- seat_5's is simply first in
    # dict/click order, so it's the one that surfaces.)
    marked = MarkedZones(
        seat_polygons=_REAL_TABLE_SEAT_POLYGONS,
        dealer_seat_key="seat_10",
        inner_oval_points=[(1025.28, 1065.6), (1100.16, 1054.08), (1785.6, 1039.68)],
        board_zone_points=[
            (1702.08, 1166.4), (2488.32, 1154.88), (2491.2, 1419.84), (1710.72, 1431.36),
        ],
        image_size=(4032, 3024),
    )
    with pytest.raises(ValueError, match=r"(?s)seat 'seat_5'.*double-click"):
        build_authoring_from_marked_zones(marked, table_id="t", chip_zone_inset_pixels=10.0)
