"""State snapshot schema (REQ-4, REQ-33).

The full pipeline state, queryable at any time and sent as the first
WebSocket message on connect (REQ-35) so a client can catch up before
consuming the live event stream.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from poker_vision.schema_base import StrictModel
from poker_vision.state.events import Street

STATE_SNAPSHOT_SCHEMA_VERSION: Literal["1.0"] = "1.0"


class SeatOccupancy(StrictModel):
    seat: str = Field(min_length=1)
    occupied: bool


class StateSnapshot(StrictModel):
    schema_version: Literal["1.0"]
    sequence: int = Field(ge=0)
    timestamp: datetime
    frame_index: int = Field(ge=0)
    seats: list[SeatOccupancy]
    dealer_seat: str | None = None
    hand_id: int | None = Field(default=None, ge=1)
    street: Street | None = None
    hand_active: bool

    @model_validator(mode="after")
    def _check_unique_seats(self) -> StateSnapshot:
        ids = [seat.seat for seat in self.seats]
        if len(ids) != len(set(ids)):
            raise ValueError("seat values must be unique")
        return self
