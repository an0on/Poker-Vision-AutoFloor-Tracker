import json

import pytest
from pydantic import ValidationError

from poker_vision.state.snapshot import StateSnapshot

VALID_SNAPSHOT: dict = {
    "schema_version": "1.0",
    "sequence": 12,
    "timestamp": "2026-08-29T12:00:00Z",
    "frame_index": 120,
    "seats": [
        {"seat": "seat_1", "occupied": True},
        {"seat": "seat_2", "occupied": False},
    ],
    "dealer_seat": "seat_1",
    "hand_id": 3,
    "street": "flop",
    "hand_active": True,
}


def _payload(base: dict, **overrides: object) -> dict:
    merged = json.loads(json.dumps(base))
    for key, value in overrides.items():
        merged[key] = value
    return merged


def test_valid_snapshot_loads():
    snapshot = StateSnapshot.model_validate(VALID_SNAPSHOT)
    assert snapshot.dealer_seat == "seat_1"
    assert snapshot.street.value == "flop"
    assert snapshot.seats[0].occupied is True


def test_snapshot_defaults_for_no_hand_in_progress():
    payload = _payload(
        VALID_SNAPSHOT,
        dealer_seat=None,
        hand_id=None,
        street=None,
        hand_active=False,
    )
    snapshot = StateSnapshot.model_validate(payload)
    assert snapshot.hand_id is None
    assert snapshot.street is None


# AC-3: wrong schema_version fails
def test_snapshot_wrong_schema_version_rejected():
    with pytest.raises(ValidationError):
        StateSnapshot.model_validate(_payload(VALID_SNAPSHOT, schema_version="2.0"))


def test_snapshot_missing_schema_version_rejected():
    payload = json.loads(json.dumps(VALID_SNAPSHOT))
    del payload["schema_version"]
    with pytest.raises(ValidationError):
        StateSnapshot.model_validate(payload)


# AC-3: unknown top-level field fails
def test_snapshot_unknown_top_level_field_rejected():
    with pytest.raises(ValidationError):
        StateSnapshot.model_validate(_payload(VALID_SNAPSHOT, unexpected_field="nope"))


def test_snapshot_unknown_nested_field_rejected():
    payload = _payload(VALID_SNAPSHOT)
    payload["seats"][0] = {"seat": "seat_1", "occupied": True, "typo_field": 1}
    with pytest.raises(ValidationError):
        StateSnapshot.model_validate(payload)


def test_snapshot_duplicate_seat_rejected():
    payload = _payload(VALID_SNAPSHOT)
    payload["seats"][1]["seat"] = "seat_1"
    with pytest.raises(ValidationError, match="unique"):
        StateSnapshot.model_validate(payload)


def test_snapshot_invalid_street_rejected():
    with pytest.raises(ValidationError):
        StateSnapshot.model_validate(_payload(VALID_SNAPSHOT, street="preflop"))


def test_snapshot_negative_sequence_rejected():
    with pytest.raises(ValidationError):
        StateSnapshot.model_validate(_payload(VALID_SNAPSHOT, sequence=-1))
