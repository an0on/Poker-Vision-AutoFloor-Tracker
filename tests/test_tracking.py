"""REQ-23: nearest-match tracker, per-class stable track IDs."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from poker_vision.calibration.geometry import TablePoint
from poker_vision.detection.models import Detection, DetectionClass, FrameDetections
from poker_vision.tracking.models import TrackedFrame
from poker_vision.tracking.tracker import NearestMatchTracker

MAX_DISTANCE = 0.05


def _frame(frame_index: int, detections: list[Detection]) -> FrameDetections:
    return FrameDetections(schema_version="1.0", frame_index=frame_index, detections=detections)


def _chip(x: float, y: float, confidence: float = 0.9) -> Detection:
    return Detection(
        object_class=DetectionClass.CHIP,
        confidence=confidence,
        center=TablePoint(x=x, y=y),
    )


def _card(x: float, y: float, confidence: float = 0.9) -> Detection:
    return Detection(
        object_class=DetectionClass.CARD,
        confidence=confidence,
        center=TablePoint(x=x, y=y),
    )


def test_first_sighting_gets_a_new_track_id():
    tracker = NearestMatchTracker(max_distance=MAX_DISTANCE)
    tracked = tracker.update(_frame(0, [_chip(1.0, 1.0)]))
    assert len(tracked.tracks) == 1
    assert tracked.tracks[0].track_id == 1
    assert tracked.tracks[0].object_class == DetectionClass.CHIP


# AC-14: track ID is preserved across a replay while per-frame movement
# stays under the configured distance threshold.
def test_track_id_preserved_while_movement_stays_under_threshold():
    tracker = NearestMatchTracker(max_distance=MAX_DISTANCE)
    step = MAX_DISTANCE * 0.6  # under threshold each frame
    x = 0.0
    first = tracker.update(_frame(0, [_chip(x, 0.0)]))
    track_id = first.tracks[0].track_id

    for frame_index in range(1, 10):
        x += step
        tracked = tracker.update(_frame(frame_index, [_chip(x, 0.0)]))
        assert len(tracked.tracks) == 1
        assert tracked.tracks[0].track_id == track_id


def test_movement_over_threshold_starts_a_new_track():
    tracker = NearestMatchTracker(max_distance=MAX_DISTANCE)
    first = tracker.update(_frame(0, [_chip(0.0, 0.0)]))
    first_id = first.tracks[0].track_id

    jump = tracker.update(_frame(1, [_chip(0.0, MAX_DISTANCE * 2)]))
    assert jump.tracks[0].track_id != first_id


def test_matching_is_scoped_per_class():
    tracker = NearestMatchTracker(max_distance=MAX_DISTANCE)
    tracked = tracker.update(_frame(0, [_chip(1.0, 1.0), _card(1.0, 1.0)]))
    chip_track = next(t for t in tracked.tracks if t.object_class == DetectionClass.CHIP)
    card_track = next(t for t in tracked.tracks if t.object_class == DetectionClass.CARD)
    assert chip_track.track_id != card_track.track_id

    # Same table position, next frame: each class keeps its own track ID.
    tracked_2 = tracker.update(_frame(1, [_chip(1.0, 1.0), _card(1.0, 1.0)]))
    chip_track_2 = next(t for t in tracked_2.tracks if t.object_class == DetectionClass.CHIP)
    card_track_2 = next(t for t in tracked_2.tracks if t.object_class == DetectionClass.CARD)
    assert chip_track_2.track_id == chip_track.track_id
    assert card_track_2.track_id == card_track.track_id


def test_two_same_class_detections_match_the_nearer_track():
    tracker = NearestMatchTracker(max_distance=MAX_DISTANCE)
    first = tracker.update(_frame(0, [_chip(0.0, 0.0), _chip(1.0, 0.0)]))
    left_id = next(t.track_id for t in first.tracks if t.center.x == 0.0)
    right_id = next(t.track_id for t in first.tracks if t.center.x == 1.0)

    # Both move slightly toward each other but stay clearly closer to their
    # own previous position than to the other track's.
    step = MAX_DISTANCE * 0.5
    second = tracker.update(_frame(1, [_chip(step, 0.0), _chip(1.0 - step, 0.0)]))
    left_track = next(t for t in second.tracks if t.center.x == pytest.approx(step))
    right_track = next(t for t in second.tracks if t.center.x == pytest.approx(1.0 - step))
    assert left_track.track_id == left_id
    assert right_track.track_id == right_id


def test_track_recovers_its_id_after_a_frame_with_no_detection_of_its_class():
    tracker = NearestMatchTracker(max_distance=MAX_DISTANCE)
    first = tracker.update(_frame(0, [_chip(1.0, 1.0)]))
    track_id = first.tracks[0].track_id

    dropout = tracker.update(_frame(1, []))
    assert dropout.tracks == []

    reappeared = tracker.update(_frame(2, [_chip(1.0, 1.0 + MAX_DISTANCE * 0.5)]))
    assert reappeared.tracks[0].track_id == track_id


def test_unmatched_track_position_is_kept_when_a_sibling_of_the_same_class_is_detected():
    # Two chips tracked; one goes briefly undetected (e.g. occluded) while
    # the other keeps being seen every frame. The occluded one must still
    # recover its own track ID, not lose its remembered position just
    # because its class wasn't entirely absent that frame.
    tracker = NearestMatchTracker(max_distance=MAX_DISTANCE)
    first = tracker.update(_frame(0, [_chip(0.0, 0.0), _chip(5.0, 5.0)]))
    occluded_id = next(t.track_id for t in first.tracks if t.center.x == 0.0)

    only_visible = tracker.update(_frame(1, [_chip(5.0, 5.0)]))
    assert len(only_visible.tracks) == 1

    reappeared = tracker.update(_frame(2, [_chip(0.0, MAX_DISTANCE * 0.5), _chip(5.0, 5.0)]))
    reappeared_id = next(t.track_id for t in reappeared.tracks if t.center.x == 0.0)
    assert reappeared_id == occluded_id


def test_frame_index_is_carried_through():
    tracker = NearestMatchTracker(max_distance=MAX_DISTANCE)
    tracked = tracker.update(_frame(7, [_chip(1.0, 1.0)]))
    assert tracked.frame_index == 7


VALID_TRACKED_FRAME: dict = {
    "schema_version": "1.0",
    "frame_index": 0,
    "tracks": [
        {
            "track_id": 1,
            "object_class": "chip",
            "confidence": 0.9,
            "center": {"x": 1.0, "y": 1.0},
        }
    ],
}


def _payload(base: dict, **overrides: object) -> dict:
    merged = json.loads(json.dumps(base))
    for key, value in overrides.items():
        merged[key] = value
    return merged


def test_valid_tracked_frame_loads():
    frame = TrackedFrame.model_validate(VALID_TRACKED_FRAME)
    assert frame.tracks[0].track_id == 1


def test_tracked_frame_wrong_schema_version_rejected():
    with pytest.raises(ValidationError):
        TrackedFrame.model_validate(_payload(VALID_TRACKED_FRAME, schema_version="2.0"))


def test_tracked_frame_unknown_field_rejected():
    with pytest.raises(ValidationError):
        TrackedFrame.model_validate(_payload(VALID_TRACKED_FRAME, unexpected_field="nope"))


def test_track_id_must_be_positive():
    invalid_track = {**VALID_TRACKED_FRAME["tracks"][0], "track_id": 0}
    with pytest.raises(ValidationError):
        TrackedFrame.model_validate(_payload(VALID_TRACKED_FRAME, tracks=[invalid_track]))
