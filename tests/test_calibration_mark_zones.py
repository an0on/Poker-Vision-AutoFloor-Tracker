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


def _small_marked_zones() -> MarkedZones:
    # A minimal, plausible 4-seat "table": seats far enough apart that
    # DEFAULT_CHIP_ZONE_INSET_PIXELS (a fixed pixel distance, meaningful at
    # real reference-photo scale) can never collide with a neighbor's chip
    # zone or escape player_area, and the board_zone sits well clear of all
    # four.
    seats = {
        "north": _square(0, -1000, half=200),
        "east": _square(1000, 0, half=200),
        "south": _square(0, 1000, half=200),
        "west": _square(-1000, 0, half=200),
    }
    board_zone = [(-100, -100), (100, -100), (100, 100), (-100, 100)]
    inner_oval = _square(0, 0, half=150)
    return MarkedZones(
        seat_polygons=seats,
        dealer_seat_key="north",
        inner_oval_points=inner_oval,
        board_zone_points=board_zone,
        image_size=(2000, 2000),
    )


# Codex review finding (P1): anchoring a neighbour clip's cut line only on
# a seat's own inner-side points fixes where the line sits, but not which
# vertices fall on which side of it -- an outer, rail-facing corner can
# still measure closer to a given neighbour's clip direction than the
# seat's own inner points do, for a sufficiently tapered wedge, and get
# clipped away regardless of the anchor restriction. Reproduced directly
# from the repo owner's real 10-seat click session (one wedge near the
# table's curved end) rather than a constructed worst case -- table_centroid
# and the neighbour's centroid below are that real session's actual values.
_TAPERED_SEAT = [
    (1586.88, 662.4),
    (1788.48, 1036.8),
    (2350.08, 1028.16),
    (2540.16, 630.72),
    (1589.76, 659.52),
]
_TAPERED_SEAT_TABLE_CENTROID = (2097.634968067227, 1333.7083966386556)
_TAPERED_SEAT_NEIGHBOR_CENTROID = (1129.6, 779.52)


def test_safe_chip_zone_never_removes_a_tapered_seats_outer_points():
    from poker_vision.calibration.mark_zones import _outer_points, _safe_chip_zone

    zone = _safe_chip_zone(
        _TAPERED_SEAT,
        _TAPERED_SEAT_TABLE_CENTROID,
        [(_TAPERED_SEAT_NEIGHBOR_CENTROID, 10.0)],
        10.0,
    )
    outer = _outer_points(_TAPERED_SEAT, _TAPERED_SEAT_TABLE_CENTROID)
    assert all(point in zone for point in outer)


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


def test_build_authoring_chip_zone_keeps_outer_edge_at_full_extent():
    # Players stack chips right up against the rail -- the two corners of
    # each seat wedge farther from the table centroid (its outer, rail-
    # facing edge) must be untouched, not pulled inward like the old
    # uniform shrink-toward-centroid did.
    authoring = build_authoring_from_marked_zones(_small_marked_zones(), table_id="t")
    seat = next(s for s in authoring.seats if s.seat_id == "seat_4")  # "north"
    outer_chip_points = {
        (p.x, p.y) for p in seat.zones.chip_zone.points if max(abs(p.x), abs(p.y)) > 1000
    }
    assert outer_chip_points == {(-200.0, -1200.0), (200.0, -1200.0)}


def test_build_authoring_chip_zone_inset_pixels_is_configurable():
    default = build_authoring_from_marked_zones(_small_marked_zones(), table_id="t")
    inset_more = build_authoring_from_marked_zones(
        _small_marked_zones(), table_id="t", chip_zone_inset_pixels=80.0
    )

    def chip_zone_area(authoring, seat_id):
        seat = next(s for s in authoring.seats if s.seat_id == seat_id)
        return abs(polygon_signed_area(seat.zones.chip_zone.points))

    assert chip_zone_area(inset_more, "seat_4") < chip_zone_area(default, "seat_4")


def test_build_authoring_chip_zone_inset_of_zero_matches_player_area_area():
    # inset_pixels=0.0 cuts exactly at the innermost point's own distance
    # to the table centroid, clipping away nothing -- the clip can still
    # restructure the point list at that exact boundary case (e.g. an
    # inserted point coincident with an existing one), so this checks area
    # rather than an exact point-for-point match.
    authoring = build_authoring_from_marked_zones(
        _small_marked_zones(), table_id="t", chip_zone_inset_pixels=0.0
    )
    seat = authoring.seats[0]
    chip_area = abs(polygon_signed_area(seat.zones.chip_zone.points))
    player_area = abs(polygon_signed_area(seat.zones.player_area.points))
    assert chip_area == pytest.approx(player_area)


def test_build_authoring_dealer_area_is_the_clicked_inner_oval_trace():
    marked = _small_marked_zones()
    authoring = build_authoring_from_marked_zones(marked, table_id="t")
    # dealer_area is exactly the operator's freehand inner-oval trace, not
    # derived from the seat polygons -- REQ-10a's manual-click design.
    assert [(p.x, p.y) for p in authoring.zones.dealer_area.points] == marked.inner_oval_points


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
