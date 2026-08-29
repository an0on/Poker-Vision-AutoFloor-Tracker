import json

import pytest
from pydantic import ValidationError

from poker_vision.state.events import (
    DealerMovedEvent,
    EventAdapter,
    HandEndedEvent,
    HandStartedEvent,
    SeatOccupiedEvent,
    SeatVacatedEvent,
    StreetChangedEvent,
)

BASE: dict = {
    "schema_version": "1.0",
    "sequence": 0,
    "timestamp": "2026-08-29T12:00:00Z",
    "frame_index": 42,
}


def _payload(base: dict, **overrides: object) -> dict:
    merged = json.loads(json.dumps(base))
    for key, value in overrides.items():
        merged[key] = value
    return merged


def test_seat_occupied_event_round_trips_through_the_union():
    payload = _payload(BASE, event_type="seat_occupied", seat="seat_3")
    event = EventAdapter.validate_python(payload)
    assert isinstance(event, SeatOccupiedEvent)
    assert event.seat == "seat_3"


def test_seat_vacated_event_round_trips_through_the_union():
    payload = _payload(BASE, event_type="seat_vacated", seat="seat_3")
    event = EventAdapter.validate_python(payload)
    assert isinstance(event, SeatVacatedEvent)


def test_dealer_moved_event_round_trips_through_the_union():
    payload = _payload(BASE, event_type="dealer_moved", from_seat="seat_1", to_seat="seat_2")
    event = EventAdapter.validate_python(payload)
    assert isinstance(event, DealerMovedEvent)
    assert event.from_seat == "seat_1"
    assert event.to_seat == "seat_2"


def test_dealer_moved_event_from_seat_optional():
    payload = _payload(BASE, event_type="dealer_moved", to_seat="seat_2")
    event = DealerMovedEvent.model_validate(payload)
    assert event.from_seat is None


def test_street_changed_event_round_trips_through_the_union():
    payload = _payload(BASE, event_type="street_changed", hand_id=1, street="flop")
    event = EventAdapter.validate_python(payload)
    assert isinstance(event, StreetChangedEvent)
    assert event.street.value == "flop"


def test_hand_started_and_ended_events_round_trip():
    started = EventAdapter.validate_python(_payload(BASE, event_type="hand_started", hand_id=1))
    ended = EventAdapter.validate_python(_payload(BASE, event_type="hand_ended", hand_id=1))
    assert isinstance(started, HandStartedEvent)
    assert isinstance(ended, HandEndedEvent)


def test_unknown_event_type_rejected():
    payload = _payload(BASE, event_type="pot_won", seat="seat_1")
    with pytest.raises(ValidationError):
        EventAdapter.validate_python(payload)


# AC-3-equivalent: wrong schema_version fails
def test_event_wrong_schema_version_rejected():
    payload = _payload(BASE, event_type="hand_started", hand_id=1, schema_version="2.0")
    with pytest.raises(ValidationError):
        EventAdapter.validate_python(payload)


# hand_id / seat are only valid on the event types that define them
def test_hand_id_on_seat_occupied_event_rejected():
    payload = _payload(BASE, event_type="seat_occupied", seat="seat_1", hand_id=1)
    with pytest.raises(ValidationError):
        SeatOccupiedEvent.model_validate(payload)


def test_seat_on_hand_started_event_rejected():
    payload = _payload(BASE, event_type="hand_started", hand_id=1, seat="seat_1")
    with pytest.raises(ValidationError):
        HandStartedEvent.model_validate(payload)


def test_invalid_street_value_rejected():
    payload = _payload(BASE, event_type="street_changed", hand_id=1, street="preflop")
    with pytest.raises(ValidationError):
        StreetChangedEvent.model_validate(payload)


def test_negative_sequence_rejected():
    payload = _payload(BASE, event_type="hand_started", hand_id=1, sequence=-1)
    with pytest.raises(ValidationError):
        HandStartedEvent.model_validate(payload)
