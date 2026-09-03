"""REQ-10a: `calib mark-zones`' pure click-to-authoring geometry."""

from __future__ import annotations

import pytest

from poker_vision.calibration.geometry import polygon_signed_area
from poker_vision.calibration.mark_zones import (
    MarkedZones,
    build_authoring_from_marked_zones,
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
