"""REQ-23: nearest-match tracker, per-class stable track IDs."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from poker_vision.calibration.geometry import TableDimensions, TablePoint, TableUnit
from poker_vision.detection.models import (
    Detection,
    DetectionClass,
    FrameDetections,
    TableBoundingBox,
)
from poker_vision.tracking.models import TrackedFrame
from poker_vision.tracking.tracker import (
    _MAX_KNOWN_TRACKS_PER_CLASS,
    _STALE_TRACK_TTL_CALLS,
    NearestMatchTracker,
)

MAX_DISTANCE = 0.05
TABLE = TableDimensions(width=100.0, height=100.0, unit=TableUnit.CM)


def _tracker(max_distance: float = MAX_DISTANCE) -> NearestMatchTracker:
    return NearestMatchTracker(max_distance=max_distance, table=TABLE)


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
    tracker = _tracker()
    tracked = tracker.update(_frame(0, [_chip(1.0, 1.0)]))
    assert len(tracked.tracks) == 1
    assert tracked.tracks[0].track_id == 1
    assert tracked.tracks[0].object_class == DetectionClass.CHIP


# AC-14: track ID is preserved across a replay while per-frame movement
# stays under the configured distance threshold.
def test_track_id_preserved_while_movement_stays_under_threshold():
    tracker = _tracker()
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
    tracker = _tracker()
    first = tracker.update(_frame(0, [_chip(0.0, 0.0)]))
    first_id = first.tracks[0].track_id

    jump = tracker.update(_frame(1, [_chip(0.0, MAX_DISTANCE * 2)]))
    assert jump.tracks[0].track_id != first_id


def test_matching_is_scoped_per_class():
    tracker = _tracker()
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
    tracker = _tracker()
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


# Codex finding: a greedy "closest pair first" matcher can strand a
# detection as a spurious new track even though a pairing exists that keeps
# both tracks under threshold. Tracks at 0.00/0.04, detections at
# 0.03/0.08, threshold 0.05: greedy grabs 0.04<->0.03 (distance 0.01)
# first, leaving 0.08 unmatched (0.00<->0.08 is 0.08, over threshold) --
# even though 0.00<->0.03 (0.03) and 0.04<->0.08 (0.04) both stay valid and
# keep every ID. The optimal assignment must prefer that full pairing.
def test_matching_maximizes_kept_tracks_over_the_single_closest_pair():
    tracker = _tracker()
    first = tracker.update(_frame(0, [_chip(0.00, 0.0), _chip(0.04, 0.0)]))
    id_at_000 = next(t.track_id for t in first.tracks if t.center.x == 0.00)
    id_at_004 = next(t.track_id for t in first.tracks if t.center.x == 0.04)

    second = tracker.update(_frame(1, [_chip(0.03, 0.0), _chip(0.08, 0.0)]))
    assert len(second.tracks) == 2
    id_at_003 = next(t.track_id for t in second.tracks if t.center.x == pytest.approx(0.03))
    id_at_008 = next(t.track_id for t in second.tracks if t.center.x == pytest.approx(0.08))
    assert id_at_003 == id_at_000
    assert id_at_008 == id_at_004


def test_track_recovers_its_id_after_a_frame_with_no_detection_of_its_class():
    tracker = _tracker()
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
    tracker = _tracker()
    first = tracker.update(_frame(0, [_chip(0.0, 0.0), _chip(5.0, 5.0)]))
    occluded_id = next(t.track_id for t in first.tracks if t.center.x == 0.0)

    only_visible = tracker.update(_frame(1, [_chip(5.0, 5.0)]))
    assert len(only_visible.tracks) == 1

    reappeared = tracker.update(_frame(2, [_chip(0.0, MAX_DISTANCE * 0.5), _chip(5.0, 5.0)]))
    reappeared_id = next(t.track_id for t in reappeared.tracks if t.center.x == 0.0)
    assert reappeared_id == occluded_id


def test_frame_index_is_carried_through():
    tracker = _tracker()
    tracked = tracker.update(_frame(7, [_chip(1.0, 1.0)]))
    assert tracked.frame_index == 7


# Codex finding: a track that never reappears (e.g. a one-off ghost
# detection) must not be kept in memory forever -- that both grows without
# bound and lets an unrelated later object at the same spot inherit a
# stale ID. This TTL is a plain safety net, not REQ-24's hysteresis (which
# is frame-counted, configurable, and per-class).
def test_stale_track_is_forgotten_after_ttl_and_reappearance_gets_a_new_id():
    tracker = _tracker()
    first = tracker.update(_frame(0, [_chip(1.0, 1.0)]))
    first_id = first.tracks[0].track_id

    # Advance the tracker's internal call clock well past the TTL without
    # ever matching the chip class again (a different class's detections,
    # or none at all, both count as calls).
    for frame_index in range(1, _STALE_TRACK_TTL_CALLS + 2):
        tracker.update(_frame(frame_index, []))

    reappeared = tracker.update(
        _frame(_STALE_TRACK_TTL_CALLS + 2, [_chip(1.0, 1.0)])
    )
    assert reappeared.tracks[0].track_id != first_id


# Codex finding: eviction must run before matching, not after -- otherwise
# a detection landing on a track's old position in the exact call where it
# crosses the TTL would refresh `_last_matched_call` first, and the
# eviction sweep that call would then find nothing stale to remove.
def test_track_evicted_before_matching_in_the_same_call_it_goes_stale():
    tracker = _tracker()
    first = tracker.update(_frame(0, [_chip(1.0, 1.0)]))
    first_id = first.tracks[0].track_id

    for frame_index in range(1, _STALE_TRACK_TTL_CALLS + 1):
        tracker.update(_frame(frame_index, []))

    # This call is exactly where the chip's age first exceeds the TTL. A
    # detection right on its old position must get a new ID, not revive it.
    boundary = tracker.update(
        _frame(_STALE_TRACK_TTL_CALLS + 1, [_chip(1.0, 1.0)])
    )
    assert boundary.tracks[0].track_id != first_id


# Codex finding: a rejected update() (invalid detection) must be atomic --
# no call-count bump, no eviction -- otherwise a failed call can silently
# push an unrelated, still-valid track over the staleness TTL.
def test_rejected_update_has_no_side_effects_on_the_stale_ttl_clock():
    tracker = _tracker()
    first = tracker.update(_frame(0, [_chip(1.0, 1.0)]))
    first_id = first.tracks[0].track_id

    # One call short of the point where the *next* successful call would
    # land exactly on the staleness boundary (age == TTL, not yet stale).
    for frame_index in range(1, _STALE_TRACK_TTL_CALLS):
        tracker.update(_frame(frame_index, []))

    with pytest.raises(ValueError, match="outside the calibrated table"):
        tracker.update(_frame(_STALE_TRACK_TTL_CALLS, [_chip(-1.0, 1.0)]))

    # If the rejected call above had still bumped the call counter and run
    # eviction, this call would land one step past the boundary and lose
    # the ID. It must not: the failed call left no trace.
    boundary = tracker.update(
        _frame(_STALE_TRACK_TTL_CALLS + 1, [_chip(1.0, 1.0)])
    )
    assert boundary.tracks[0].track_id == first_id


# Codex finding: without a hard cap, a burst of never-matching detections
# within a *single* frame (e.g. 50+ ghosts, all equally "fresh" so the
# age-based TTL can't distinguish any of them yet) can grow one class's
# remembered-track count far beyond what any later frame could plausibly
# match against, making `optimal_assignment`'s O(n^3) cost explode. The
# per-class cap must bound this immediately, not just eventually via TTL.
def test_known_tracks_per_class_are_capped_even_within_a_single_frame():
    tracker = _tracker()
    count = _MAX_KNOWN_TRACKS_PER_CLASS + 10
    # Spaced 1.0 apart, i.e. 20x MAX_DISTANCE: no two ever accidentally
    # match each other.
    xs = [float(i) for i in range(count)]

    first = tracker.update(_frame(0, [_chip(x, 0.0) for x in xs]))
    # Capping only bounds what's *remembered*, not what's reported: every
    # detection in a frame always gets some track_id, this frame.
    assert len(first.tracks) == count
    first_ids = {track.center.x: track.track_id for track in first.tracks}

    # Same positions again: anything evicted by the cap after frame 0 must
    # mint a new ID instead of resuming the old one.
    second = tracker.update(_frame(1, [_chip(x, 0.0) for x in xs]))
    second_ids = {track.center.x: track.track_id for track in second.tracks}

    preserved = sum(1 for x in xs if first_ids[x] == second_ids[x])
    assert preserved <= _MAX_KNOWN_TRACKS_PER_CLASS
    assert preserved < count


def test_track_survives_well_under_the_stale_ttl():
    tracker = _tracker()
    first = tracker.update(_frame(0, [_chip(1.0, 1.0)]))
    first_id = first.tracks[0].track_id

    for frame_index in range(1, _STALE_TRACK_TTL_CALLS - 1):
        tracker.update(_frame(frame_index, []))

    reappeared = tracker.update(
        _frame(_STALE_TRACK_TTL_CALLS - 1, [_chip(1.0, 1.0)])
    )
    assert reappeared.tracks[0].track_id == first_id


# Codex finding: a detection whose table-plane coordinates fall outside the
# calibrated table must be rejected before matching, not silently tracked.
def test_detection_center_outside_table_bounds_is_rejected():
    tracker = _tracker()
    with pytest.raises(ValueError, match="outside the calibrated table"):
        tracker.update(_frame(0, [_chip(-1.0, 1.0)]))

    tracker_2 = _tracker()
    with pytest.raises(ValueError, match="outside the calibrated table"):
        tracker_2.update(_frame(0, [_chip(1.0, TABLE.height + 1.0)]))


# Codex finding: an optional box that straddles the table edge (e.g. an
# object detected right at the rim) is normal detector output (REQ-17 only
# requires table coordinates, not containment) and must not be rejected --
# only the center (what matching actually uses) is checked.
def test_detection_box_straddling_table_boundary_is_accepted():
    tracker = _tracker()
    detection = Detection(
        object_class=DetectionClass.CHIP,
        confidence=0.9,
        center=TablePoint(x=1.0, y=1.0),
        box=TableBoundingBox(
            min=TablePoint(x=0.5, y=0.5),
            max=TablePoint(x=TABLE.width + 5.0, y=1.5),
        ),
    )
    tracked = tracker.update(_frame(0, [detection]))
    assert len(tracked.tracks) == 1
    assert tracked.tracks[0].box == detection.box


def test_detection_exactly_on_table_boundary_is_accepted():
    tracker = _tracker()
    tracked = tracker.update(_frame(0, [_chip(0.0, 0.0), _chip(TABLE.width, TABLE.height)]))
    assert len(tracked.tracks) == 2


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
