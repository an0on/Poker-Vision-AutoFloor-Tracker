"""Tracking stage: builds `NearestMatchTracker` from config (REQ-23, REQ-24).

Mirrors `capture.create_capture` / `detection.create_detector`: this is the
stage's own config-driven construction entry point.
"""

from __future__ import annotations

from poker_vision.calibration.geometry import TableDimensions
from poker_vision.config import Config
from poker_vision.tracking.tracker import NearestMatchTracker

__all__ = ["create_tracker"]


def create_tracker(config: Config, table: TableDimensions) -> NearestMatchTracker:
    """Build a `NearestMatchTracker` sized against the configured hysteresis.

    `NearestMatchTracker`'s own `stale_track_ttl` safety net must not
    silently outrank real hysteresis (`HysteresisConfig.n_off`, global or
    per-class -- see `tracker.py`'s module docstring): passing the
    default here would let a track's ID be evicted before hysteresis ever
    gets to decide it's actually gone, for any class whose configured
    `n_off` exceeds that default. Using the largest configured `n_off`
    (global or any per-class override) guarantees the opposite -- hysteresis
    always gets first say -- for every class in play.
    """
    n_off_values = [config.hysteresis.n_off] + [
        override.n_off
        for override in config.hysteresis.per_class.values()
        if override.n_off is not None
    ]
    return NearestMatchTracker(
        max_distance=config.thresholds.tracking_max_distance,
        table=table,
        stale_track_ttl=max(n_off_values),
    )
