"""REQ-10a: `calib mark-zones`' pure click-to-authoring geometry."""

from __future__ import annotations

import pytest

from poker_vision.calibration.geometry import polygon_signed_area
from poker_vision.calibration.mark_zones import (
    ArcClick,
    MarkedZones,
    build_authoring_from_marked_zones,
    build_oval_polygon,
    number_seats_clockwise,
)

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


# --- build_oval_polygon ------------------------------------------------------

# A simple stadium: two same-radius circles centered at (-100, 0) and
# (100, 0), radius 50, straight sides at y = -50 and y = +50 (tangent
# points already fall exactly on those lines for a symmetric stadium).
LEFT_END = ArcClick(start=(-100, -50), center=(-100, 0), end=(-100, 50))
RIGHT_END = ArcClick(start=(100, 50), center=(100, 0), end=(100, -50))


def test_build_oval_polygon_is_closed_simple_and_nondegenerate():
    points = build_oval_polygon(LEFT_END, RIGHT_END, arc_samples=16)
    assert len(points) == 34  # 17 + 17 points, one shared pair of tangent points each end
    # Non-zero area, i.e. not degenerate/collinear (mirrors TablePolygon's own check).
    area = abs(
        sum(
            points[i][0] * points[(i + 1) % len(points)][1]
            - points[(i + 1) % len(points)][0] * points[i][1]
            for i in range(len(points))
        )
        / 2.0
    )
    assert area > 0


def test_build_oval_polygon_arc_bulges_outward_not_inward():
    points = build_oval_polygon(LEFT_END, RIGHT_END, arc_samples=16)
    # The left arc must bulge further left (more negative x) than either of
    # its own tangent points -- confirms "the long way round", not the short
    # arc that would cut through the table's interior between the two ends.
    left_arc_xs = [x for x, y in points[:17]]
    assert min(left_arc_xs) < -100 - 1e-6


def test_build_oval_polygon_degenerate_center_on_point_raises():
    bad = ArcClick(start=(0, 0), center=(0, 0), end=(0, 0))
    with pytest.raises(ValueError, match="degenerate"):
        build_oval_polygon(bad, RIGHT_END)


# --- build_authoring_from_marked_zones --------------------------------------


def _small_marked_zones() -> MarkedZones:
    # A minimal, plausible 4-seat "table": seats far enough apart and small
    # enough that a 50%-toward-centroid chip_zone can never collide with a
    # neighbor's, and the board_zone/dealer_area sit well clear of all four.
    seats = {
        "north": _square(0, -100, half=20),
        "east": _square(100, 0, half=20),
        "south": _square(0, 100, half=20),
        "west": _square(-100, 0, half=20),
    }
    board_zone = [(-10, -10), (10, -10), (10, 10), (-10, 10)]
    inner_oval = (
        ArcClick(start=(-40, -40), center=(-40, 0), end=(-40, 40)),
        ArcClick(start=(40, 40), center=(40, 0), end=(40, -40)),
    )
    outer_oval = (
        ArcClick(start=(-140, -140), center=(-140, 0), end=(-140, 140)),
        ArcClick(start=(140, 140), center=(140, 0), end=(140, -140)),
    )
    return MarkedZones(
        seat_polygons=seats,
        dealer_seat_key="north",
        board_zone_points=board_zone,
        inner_oval=inner_oval,
        outer_oval=outer_oval,
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


def test_build_authoring_chip_zone_shrink_factor_is_configurable():
    default = build_authoring_from_marked_zones(_small_marked_zones(), table_id="t")
    shrunk_more = build_authoring_from_marked_zones(
        _small_marked_zones(), table_id="t", chip_zone_shrink_factor=0.1
    )

    def chip_zone_area(authoring, seat_id):
        seat = next(s for s in authoring.seats if s.seat_id == seat_id)
        return abs(polygon_signed_area(seat.zones.chip_zone.points))

    assert chip_zone_area(shrunk_more, "seat_1") < chip_zone_area(default, "seat_1")


def test_build_authoring_dealer_area_is_derived_from_inner_oval():
    authoring = build_authoring_from_marked_zones(_small_marked_zones(), table_id="t")
    # Roughly the inner oval's extent (radius 40 circles at x=-40/+40) --
    # confirms dealer_area came from the inner, not outer, oval.
    xs = [p.x for p in authoring.zones.dealer_area.points]
    assert max(xs) < 100
    assert min(xs) > -100


def test_build_authoring_is_deterministic():
    a = build_authoring_from_marked_zones(_small_marked_zones(), table_id="t")
    b = build_authoring_from_marked_zones(_small_marked_zones(), table_id="t")
    assert a.model_dump() == b.model_dump()


# Codex review finding (P2): the outer oval's clicked radius/curvature must
# actually influence the output -- not be collected and then discarded,
# leaving only its 4 tangent points to matter.


def test_build_authoring_homography_uses_full_outer_oval_curve():
    base = _small_marked_zones()
    wider = MarkedZones(
        seat_polygons=base.seat_polygons,
        dealer_seat_key=base.dealer_seat_key,
        board_zone_points=base.board_zone_points,
        inner_oval=base.inner_oval,
        # Same 4 tangent points as base.outer_oval, but a much larger
        # radius/center for each end -- if only the tangent points were
        # used, this would be indistinguishable from `base`.
        outer_oval=(
            ArcClick(start=(-140, -140), center=(-500, 0), end=(-140, 140)),
            ArcClick(start=(140, 140), center=(500, 0), end=(140, -140)),
        ),
        image_size=base.image_size,
    )

    authoring_base = build_authoring_from_marked_zones(base, table_id="t")
    authoring_wider = build_authoring_from_marked_zones(wider, table_id="t")

    assert authoring_base.homography.points != authoring_wider.homography.points
    # Every correspondence is still image_point == table_point (identity).
    for correspondence in authoring_wider.homography.points:
        assert correspondence.image_point.x == correspondence.table_point.x
        assert correspondence.image_point.y == correspondence.table_point.y


@pytest.mark.parametrize("bad_factor", [0.0, -0.5, 1.5])
def test_build_authoring_rejects_out_of_range_chip_zone_shrink_factor(bad_factor):
    with pytest.raises(ValueError, match="chip_zone_shrink_factor"):
        build_authoring_from_marked_zones(
            _small_marked_zones(), table_id="t", chip_zone_shrink_factor=bad_factor
        )


def test_build_authoring_accepts_shrink_factor_of_exactly_one():
    # chip_zone == player_area is unusual but valid (REQ-11 allows touching
    # boundaries) -- 1.0 is the upper edge of the accepted range, not past it.
    authoring = build_authoring_from_marked_zones(
        _small_marked_zones(), table_id="t", chip_zone_shrink_factor=1.0
    )
    seat = authoring.seats[0]
    for chip_point, player_point in zip(
        seat.zones.chip_zone.points, seat.zones.player_area.points, strict=True
    ):
        assert chip_point.x == pytest.approx(player_point.x)
        assert chip_point.y == pytest.approx(player_point.y)
