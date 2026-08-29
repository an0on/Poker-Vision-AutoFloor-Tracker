"""Typed pipeline events emitted by the state machine (REQ-4, REQ-33).

Each event type only carries the fields relevant to it (`hand_id`/`seat`
"falls zutreffend" per REQ-33) via a discriminated union on `event_type`,
rather than one flat model with everything optional — so e.g. attaching a
`hand_id` to a `seat_occupied` event is a validation error, not a silently
accepted stray field.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter

from poker_vision.schema_base import StrictModel

EVENT_SCHEMA_VERSION: Literal["1.0"] = "1.0"


class Street(StrEnum):
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"


class EventBase(StrictModel):
    schema_version: Literal["1.0"]
    sequence: int = Field(ge=0)
    timestamp: datetime
    frame_index: int = Field(ge=0)


class SeatOccupiedEvent(EventBase):
    event_type: Literal["seat_occupied"] = "seat_occupied"
    seat: str = Field(min_length=1)


class SeatVacatedEvent(EventBase):
    event_type: Literal["seat_vacated"] = "seat_vacated"
    seat: str = Field(min_length=1)


class DealerMovedEvent(EventBase):
    event_type: Literal["dealer_moved"] = "dealer_moved"
    from_seat: str | None = Field(default=None, min_length=1)
    to_seat: str = Field(min_length=1)


class StreetChangedEvent(EventBase):
    event_type: Literal["street_changed"] = "street_changed"
    hand_id: int = Field(ge=1)
    street: Street


class HandStartedEvent(EventBase):
    event_type: Literal["hand_started"] = "hand_started"
    hand_id: int = Field(ge=1)


class HandEndedEvent(EventBase):
    event_type: Literal["hand_ended"] = "hand_ended"
    hand_id: int = Field(ge=1)


Event = Annotated[
    (
        SeatOccupiedEvent
        | SeatVacatedEvent
        | DealerMovedEvent
        | StreetChangedEvent
        | HandStartedEvent
        | HandEndedEvent
    ),
    Field(discriminator="event_type"),
]

EventAdapter: TypeAdapter[Event] = TypeAdapter(Event)
