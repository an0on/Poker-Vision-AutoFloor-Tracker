"""Tracking output schema (REQ-4, REQ-23).

`NearestMatchTracker` (see `tracker.py`) turns one frame's `Detection`s into
`TrackedObject`s: the same detection fields, plus a `track_id` that stays
stable across frames for as long as nearest-matching keeps re-linking it to
the same underlying object. Only objects actually observed in this frame
are reported here (REQ-25's "nur bestätigte Tracks" hysteresis filter is a
separate, later stage on top of this).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from poker_vision.calibration.geometry import TablePoint
from poker_vision.detection.models import DetectionClass, TableBoundingBox
from poker_vision.schema_base import StrictModel

TRACKING_SCHEMA_VERSION: Literal["1.0"] = "1.0"


class TrackedObject(StrictModel):
    track_id: int = Field(ge=1)
    object_class: DetectionClass
    confidence: float = Field(ge=0.0, le=1.0)
    center: TablePoint
    box: TableBoundingBox | None = None


class TrackedFrame(StrictModel):
    schema_version: Literal["1.0"]
    frame_index: int = Field(ge=0)
    tracks: list[TrackedObject] = Field(default_factory=list)
