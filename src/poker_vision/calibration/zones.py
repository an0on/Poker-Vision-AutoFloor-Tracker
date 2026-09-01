"""Seat and table zones (REQ-4, REQ-7).

Zones are authored directly in table-plane coordinates: the physical table
layout is camera-independent, only the homography (see `homography.py`)
ties a given camera view back into this fixed plane. `CalibrationRuntime`
carries the same resolved zones through unchanged.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from poker_vision.calibration.geometry import TablePolygon
from poker_vision.calibration.topology import polygon_contains, polygons_overlap
from poker_vision.schema_base import StrictModel


class SeatZones(StrictModel):
    player_area: TablePolygon
    chip_zone: TablePolygon


class CalibrationSeat(StrictModel):
    seat_id: str = Field(min_length=1)
    zones: SeatZones


class GlobalZones(StrictModel):
    board_zone: TablePolygon
    dealer_area: TablePolygon


def require_unique_seat_ids(seats: list[CalibrationSeat]) -> None:
    ids = [seat.seat_id for seat in seats]
    if len(ids) != len(set(ids)):
        raise ValueError("seat_id values must be unique")


class SeatListModel(StrictModel):
    """Mixin providing the `seats` field plus its uniqueness check.

    Shared by `CalibrationAuthoring` and `CalibrationRuntime` so the two
    schemas can't drift on how seat-id uniqueness (REQ-7: "stabil und
    eindeutig") is enforced.
    """

    seats: list[CalibrationSeat] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_unique_seat_ids(self) -> SeatListModel:
        require_unique_seat_ids(self.seats)
        return self


class CalibrationGeometryModel(SeatListModel):
    """Mixin adding `zones` plus the cross-zone topology checks (REQ-11).

    Shared by `CalibrationAuthoring` and `CalibrationRuntime` for the same
    reason as `SeatListModel`: the two schemas carry identical seat/zone
    geometry and must not drift on how it's validated.
    """

    zones: GlobalZones
    # REQ-7: the fixed physical "Kartengeber" (card dealer) position -- one
    # specific seat, unrelated to the *rotating* dealer button tracked at
    # runtime (`state.dealer.DealerSeatTracker`'s `dealer_seat`, resolved
    # fresh every hand via `dealer_area`/nearest-seat fallback, REQ-27/30).
    # This field is the opposite: a fixed calibration fact about the table
    # itself. Named `card_dealer_seat_id` specifically to avoid colliding
    # with that unrelated runtime concept. Whether a 10th player currently
    # plays from this seat is tournament/game state, out of scope here.
    card_dealer_seat_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_card_dealer_seat_id_exists(self) -> CalibrationGeometryModel:
        if self.card_dealer_seat_id not in {seat.seat_id for seat in self.seats}:
            raise ValueError(
                f"card_dealer_seat_id '{self.card_dealer_seat_id}' does not reference "
                "an existing seat"
            )
        return self

    @model_validator(mode="after")
    def _check_zone_topology(self) -> CalibrationGeometryModel:
        for seat in self.seats:
            if not polygon_contains(seat.zones.player_area, seat.zones.chip_zone):
                raise ValueError(
                    f"zone topology violation: seat '{seat.seat_id}' chip_zone "
                    "is not fully contained in its player_area"
                )

        for index, seat_a in enumerate(self.seats):
            for seat_b in self.seats[index + 1 :]:
                if polygons_overlap(seat_a.zones.chip_zone, seat_b.zones.chip_zone):
                    raise ValueError(
                        "zone topology violation: chip_zone overlap between seats "
                        f"'{seat_a.seat_id}' and '{seat_b.seat_id}'"
                    )

        for seat in self.seats:
            if polygons_overlap(self.zones.board_zone, seat.zones.chip_zone):
                raise ValueError(
                    "zone topology violation: board_zone overlaps chip_zone of seat "
                    f"'{seat.seat_id}'"
                )
        return self
