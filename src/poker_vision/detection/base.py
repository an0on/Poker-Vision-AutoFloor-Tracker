"""`Detector` interface (REQ-17).

Concrete detectors (`mock` modes A/B/C - REQ-18/19/20, `yolo` - REQ-22) only
ever implement `_detect_raw`, returning `RawDetection`s in pixel space.
`Detector.detect` is the one path from a raw detection to the stage's public
output (`FrameDetections`), and it is the only place that calls the
pixel -> table transform: no subclass has a way to construct a `Detection`
directly, so a detector cannot leave the stage without going through the
homography (REQ-5, REQ-17).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from poker_vision.calibration.geometry import PixelPoint
from poker_vision.calibration.runtime import CalibrationRuntime
from poker_vision.capture.frame import Frame
from poker_vision.detection.geometry import (
    PixelBox,
    apply_homography_to_point,
    transform_box_to_table,
)
from poker_vision.detection.models import (
    DETECTION_SCHEMA_VERSION,
    Detection,
    DetectionClass,
    FrameDetections,
)


@dataclass(frozen=True, slots=True)
class RawDetection:
    """One detection in pixel space, before the detection-stage transform.

    `center` is required and already computed by the concrete detector (via
    `geometry.box_center` for box-based sources, or however a source without
    a box, e.g. an ArUco marker centroid - REQ-19, derives its own centre).
    `box` is optional, matching the optional box on the final `Detection`.
    """

    object_class: DetectionClass
    confidence: float
    center: PixelPoint
    box: PixelBox | None = None


class Detector(ABC):
    """Produces `FrameDetections` for one frame, in table coordinates."""

    def __init__(self, calibration: CalibrationRuntime) -> None:
        self._calibration = calibration

    @abstractmethod
    def _detect_raw(self, frame: Frame) -> list[RawDetection]:
        """Return this frame's detections in pixel space."""

    def detect(self, frame: Frame) -> FrameDetections:
        raw_detections = self._detect_raw(frame)
        homography = self._calibration.homography
        detections = [
            Detection(
                object_class=raw.object_class,
                confidence=raw.confidence,
                center=apply_homography_to_point(raw.center, homography),
                box=transform_box_to_table(raw.box, homography) if raw.box is not None else None,
            )
            for raw in raw_detections
        ]
        return FrameDetections(
            schema_version=DETECTION_SCHEMA_VERSION,
            frame_index=frame.frame_index,
            detections=detections,
        )
