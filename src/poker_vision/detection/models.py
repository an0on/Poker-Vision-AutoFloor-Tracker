"""Detector output schema (REQ-4, REQ-17).

Every `Detector` implementation (`mock`, later `yolo`) emits `FrameDetections`
for a frame. The pixel -> table-plane transform happens inside the detection
stage, before results leave it, so `center`/`box` are already table
coordinates (REQ-5): nothing downstream accepts pixel coordinates.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from poker_vision.calibration.geometry import TablePoint
from poker_vision.schema_base import StrictModel

DETECTION_SCHEMA_VERSION: Literal["1.0"] = "1.0"


class DetectionClass(StrEnum):
    CHIP = "chip"
    CARD = "card"
    DEALER_BUTTON = "dealer_button"


class TableBoundingBox(StrictModel):
    """Axis-aligned bounding box in table coordinates."""

    min: TablePoint
    max: TablePoint

    @model_validator(mode="after")
    def _check_ordered(self) -> TableBoundingBox:
        if self.min.x > self.max.x or self.min.y > self.max.y:
            raise ValueError("box min must be <= max on both axes")
        return self


class Detection(StrictModel):
    object_class: DetectionClass
    confidence: float = Field(ge=0.0, le=1.0)
    center: TablePoint
    box: TableBoundingBox | None = None


class FrameDetections(StrictModel):
    schema_version: Literal["1.0"]
    frame_index: int = Field(ge=0)
    detections: list[Detection] = Field(default_factory=list)
