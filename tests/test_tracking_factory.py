"""REQ-45: `tracking.create_tracker` -- config-driven `NearestMatchTracker` construction."""

from __future__ import annotations

from poker_vision.calibration.geometry import TableDimensions, TableUnit
from poker_vision.config import (
    Config,
    HysteresisConfig,
    HysteresisOverride,
    PathsConfig,
    SourceConfig,
    SourceType,
    ThresholdsConfig,
)
from poker_vision.detection.models import DetectionClass
from poker_vision.tracking import create_tracker
from poker_vision.tracking.tracker import NearestMatchTracker

TABLE = TableDimensions(width=100.0, height=100.0, unit=TableUnit.CM)


def _config(
    hysteresis: HysteresisConfig | None = None, thresholds: ThresholdsConfig | None = None
) -> Config:
    return Config(
        schema_version="1.0",
        device="cpu",
        source=SourceConfig(type=SourceType.IMAGE_DIR, path="images"),
        paths=PathsConfig(
            calibration_authoring="calib_authoring.json",
            calibration_runtime="calib_runtime.json",
            jsonl_export_dir="events",
            mock_script="script.jsonl",
        ),
        hysteresis=hysteresis or HysteresisConfig(),
        thresholds=thresholds or ThresholdsConfig(),
    )


def test_create_tracker_uses_tracking_max_distance_threshold():
    config = _config(thresholds=ThresholdsConfig(tracking_max_distance=7.5))
    tracker = create_tracker(config, TABLE)
    assert isinstance(tracker, NearestMatchTracker)
    assert tracker._max_distance == 7.5


def test_create_tracker_stale_track_ttl_matches_global_n_off():
    config = _config(hysteresis=HysteresisConfig(n_off=12))
    tracker = create_tracker(config, TABLE)
    assert tracker._stale_track_ttl == 12


def test_create_tracker_stale_track_ttl_uses_largest_per_class_override():
    config = _config(
        hysteresis=HysteresisConfig(
            n_off=3,
            per_class={
                DetectionClass.CHIP: HysteresisOverride(n_off=9),
                DetectionClass.CARD: HysteresisOverride(n_off=5),
            },
        )
    )
    tracker = create_tracker(config, TABLE)
    # The tracker's own stale-eviction safety net must never outrank the
    # largest configured n_off (global or per-class) -- see tracker.py's
    # module docstring -- so it must be at least the max of all of them.
    assert tracker._stale_track_ttl == 9


def test_create_tracker_stale_track_ttl_ignores_unset_per_class_n_off():
    config = _config(
        hysteresis=HysteresisConfig(
            n_off=20, per_class={DetectionClass.CHIP: HysteresisOverride(n_on=1)}
        )
    )
    tracker = create_tracker(config, TABLE)
    assert tracker._stale_track_ttl == 20
