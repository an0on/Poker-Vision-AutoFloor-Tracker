"""Seat and table zones (REQ-4, REQ-7).

Zones are authored directly in table-plane coordinates: the physical table
layout is camera-independent, only the homography (see `homography.py`)
ties a given camera view back into this fixed plane. `CalibrationRuntime`
carries the same resolved zones through unchanged.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from poker_vision.calibration.geometry import TablePolygon
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
