"""REQ-24/REQ-25: presence hysteresis on top of per-frame tracking (AC-12)."""

from __future__ import annotations

import pytest

from poker_vision.calibration.geometry import TablePoint
from poker_vision.config import HysteresisConfig, HysteresisOverride
from poker_vision.detection.models import DetectionClass
from poker_vision.tracking.hysteresis import HysteresisFilter
from poker_vision.tracking.models import TrackedFrame, TrackedObject


def _track(
    track_id: int,
    object_class: DetectionClass = DetectionClass.CHIP,
    x: float = 1.0,
    y: float = 1.0,
    confidence: float = 0.9,
) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        object_class=object_class,
        confidence=confidence,
        center=TablePoint(x=x, y=y),
    )


def _frame(frame_index: int, tracks: list[TrackedObject]) -> TrackedFrame:
    return TrackedFrame(schema_version="1.0", frame_index=frame_index, tracks=tracks)


def _stable_ids(frame: TrackedFrame) -> set[int]:
    return {track.track_id for track in frame.tracks}


def test_track_not_reported_before_n_on_consecutive_frames():
    hysteresis = HysteresisFilter(HysteresisConfig(n_on=3, n_off=2))

    first = hysteresis.update(_frame(0, [_track(5)]))
    assert _stable_ids(first) == set()

    second = hysteresis.update(_frame(1, [_track(5)]))
    assert _stable_ids(second) == set()

    third = hysteresis.update(_frame(2, [_track(5)]))
    assert _stable_ids(third) == {5}


# AC-12: a ghost detection seen fewer than n_on consecutive times must never
# be confirmed -- a gap before reaching n_on resets its run, it does not
# accumulate credit across the gap.
def test_gap_before_n_on_resets_the_consecutive_count():
    hysteresis = HysteresisFilter(HysteresisConfig(n_on=3, n_off=2))

    hysteresis.update(_frame(0, [_track(5)]))  # count 1
    dropout = hysteresis.update(_frame(1, []))  # miss resets pending count
    assert _stable_ids(dropout) == set()

    reappear_1 = hysteresis.update(_frame(2, [_track(5)]))  # count 1 again, not 3
    assert _stable_ids(reappear_1) == set()

    reappear_2 = hysteresis.update(_frame(3, [_track(5)]))  # count 2
    assert _stable_ids(reappear_2) == set()

    reappear_3 = hysteresis.update(_frame(4, [_track(5)]))  # count 3 -> confirmed
    assert _stable_ids(reappear_3) == {5}


def test_n_on_of_one_confirms_on_first_sighting():
    hysteresis = HysteresisFilter(HysteresisConfig(n_on=1, n_off=2))
    first = hysteresis.update(_frame(0, [_track(5)]))
    assert _stable_ids(first) == {5}


# AC-12: dropout shorter than n_off must not drop the track -- it stays
# present, reported with its last known state, until n_off is reached.
def test_confirmed_track_survives_dropout_under_n_off_with_carried_state():
    hysteresis = HysteresisFilter(HysteresisConfig(n_on=1, n_off=2))
    hysteresis.update(_frame(0, [_track(5, x=1.0, y=1.0)]))

    dropout = hysteresis.update(_frame(1, []))
    assert _stable_ids(dropout) == {5}
    carried = next(t for t in dropout.tracks if t.track_id == 5)
    assert carried.center == TablePoint(x=1.0, y=1.0)


# AC-12: dropout reaching n_off consecutive missed frames removes the track
# exactly then, not before.
def test_confirmed_track_removed_after_n_off_consecutive_misses():
    hysteresis = HysteresisFilter(HysteresisConfig(n_on=1, n_off=2))
    hysteresis.update(_frame(0, [_track(5)]))

    still_there = hysteresis.update(_frame(1, []))  # miss 1 of 2
    assert _stable_ids(still_there) == {5}

    gone = hysteresis.update(_frame(2, []))  # miss 2 of 2
    assert _stable_ids(gone) == set()


def test_reappearance_before_n_off_resets_the_miss_count_and_updates_state():
    hysteresis = HysteresisFilter(HysteresisConfig(n_on=1, n_off=2))
    hysteresis.update(_frame(0, [_track(5, x=1.0, y=1.0)]))
    hysteresis.update(_frame(1, []))  # miss 1 of 2

    reappeared = hysteresis.update(_frame(2, [_track(5, x=2.0, y=2.0)]))
    assert _stable_ids(reappeared) == {5}
    assert reappeared.tracks[0].center == TablePoint(x=2.0, y=2.0)

    # Miss count was reset by the reappearance: one more miss alone must not
    # remove the track yet (that would only be miss 1 of 2 again).
    still_there = hysteresis.update(_frame(3, []))
    assert _stable_ids(still_there) == {5}


def test_track_id_removed_by_n_off_needs_a_fresh_n_on_run_to_return():
    hysteresis = HysteresisFilter(HysteresisConfig(n_on=2, n_off=1))
    hysteresis.update(_frame(0, [_track(5)]))  # on 1
    hysteresis.update(_frame(1, [_track(5)]))  # on 2 -> confirmed
    hysteresis.update(_frame(2, []))  # off 1 -> removed (n_off=1)

    # Same track_id resurfacing (e.g. tracker's own dropout recovery) starts
    # a brand-new onset count, not an instant reconfirmation.
    first_again = hysteresis.update(_frame(3, [_track(5)]))
    assert _stable_ids(first_again) == set()

    second_again = hysteresis.update(_frame(4, [_track(5)]))
    assert _stable_ids(second_again) == {5}


def test_hysteresis_state_is_scoped_per_class():
    hysteresis = HysteresisFilter(HysteresisConfig(n_on=2, n_off=2))
    hysteresis.update(
        _frame(
            0,
            [
                _track(1, object_class=DetectionClass.CHIP),
                _track(1, object_class=DetectionClass.CARD),
            ],
        )
    )
    confirmed = hysteresis.update(
        _frame(1, [_track(1, object_class=DetectionClass.CHIP)])
    )
    # Chip's track_id 1 reaches n_on=2; card's track_id 1 (unrelated track,
    # same numeric ID in a different class) was only seen once and must not
    # be confirmed just because the chip was.
    stable_classes = {t.object_class for t in confirmed.tracks}
    assert stable_classes == {DetectionClass.CHIP}


def test_per_class_override_shortens_n_on_for_one_class_only():
    config = HysteresisConfig(
        n_on=3, n_off=3, per_class={"chip": HysteresisOverride(n_on=1)}
    )
    hysteresis = HysteresisFilter(config)

    chip_confirmed = hysteresis.update(_frame(0, [_track(1, object_class=DetectionClass.CHIP)]))
    assert _stable_ids(chip_confirmed) == {1}

    # Same call, card class: global n_on=3 still applies, one sighting is
    # not enough. Filter card out of the (carried-forward chip) result.
    card_pending = hysteresis.update(_frame(1, [_track(2, object_class=DetectionClass.CARD)]))
    card_ids = {t.track_id for t in card_pending.tracks if t.object_class == DetectionClass.CARD}
    assert card_ids == set()


def test_per_class_override_can_set_only_n_off_leaving_n_on_at_global_default():
    config = HysteresisConfig(
        n_on=1, n_off=3, per_class={"card": HysteresisOverride(n_off=1)}
    )
    hysteresis = HysteresisFilter(config)
    # n_on=1 global -> confirmed on first sighting.
    hysteresis.update(_frame(0, [_track(1, object_class=DetectionClass.CARD)]))

    # Card's override sets n_off=1: a single miss must remove it, unlike the
    # global default of 3.
    gone = hysteresis.update(_frame(1, []))
    assert _stable_ids(gone) == set()


def test_frame_index_is_carried_through():
    hysteresis = HysteresisFilter(HysteresisConfig())
    result = hysteresis.update(_frame(7, []))
    assert result.frame_index == 7


def test_output_schema_version_matches_tracked_frame():
    hysteresis = HysteresisFilter(HysteresisConfig(n_on=1))
    result = hysteresis.update(_frame(0, [_track(1)]))
    assert result.schema_version == "1.0"


# Codex finding: n_on/n_off are defined in actual (`frame_index`) frames,
# not update() calls. A jump in frame_index between two calls must count as
# that many frames having silently elapsed with no data, not as a single
# consecutive step.
def test_frame_index_gap_counts_as_multiple_missed_frames_for_onset():
    hysteresis = HysteresisFilter(HysteresisConfig(n_on=3, n_off=2))
    hysteresis.update(_frame(0, [_track(5)]))  # on 1

    # Jump straight to frame 100: two frames (1, 2) elapsed unseen in
    # between, breaking the pending run just as two explicit empty calls
    # would -- this call's sighting must only be on-count 1, not 2.
    almost = hysteresis.update(_frame(100, [_track(5)]))
    assert _stable_ids(almost) == set()

    confirmed = hysteresis.update(_frame(101, [_track(5)]))
    assert _stable_ids(confirmed) == set()  # on-count 2, n_on=3

    confirmed = hysteresis.update(_frame(102, [_track(5)]))
    assert _stable_ids(confirmed) == {5}  # on-count 3


def test_frame_index_gap_counts_as_multiple_missed_frames_for_offset():
    hysteresis = HysteresisFilter(HysteresisConfig(n_on=1, n_off=3))
    hysteresis.update(_frame(0, [_track(5)]))  # confirmed immediately

    # Frames 1 and 2 elapsed with no call at all -- a gap of 2 misses. One
    # more explicit miss at frame 3 must reach n_off=3 and drop the track.
    still_there = hysteresis.update(_frame(3, []))
    assert _stable_ids(still_there) == set()


def test_frame_index_gap_shorter_than_n_off_keeps_track_confirmed():
    hysteresis = HysteresisFilter(HysteresisConfig(n_on=1, n_off=5))
    hysteresis.update(_frame(0, [_track(5)]))  # confirmed immediately

    # Only 2 frames elapsed unseen (indices 1, 2); n_off=5 not reached yet.
    still_there = hysteresis.update(_frame(3, []))
    assert _stable_ids(still_there) == {5}


def test_non_increasing_frame_index_is_rejected():
    hysteresis = HysteresisFilter(HysteresisConfig())
    hysteresis.update(_frame(5, []))

    with pytest.raises(ValueError, match="frame_index"):
        hysteresis.update(_frame(5, []))

    with pytest.raises(ValueError, match="frame_index"):
        hysteresis.update(_frame(4, []))
