"""`Detector` interface (REQ-17).

Concrete detectors (`mock` modes A/B/C - REQ-18/19/20, `yolo` - REQ-22) only
ever implement `_detect_raw`, returning `RawDetection`s in pixel space.
`Detector.detect` is the one path from a raw detection to the stage's public
output (`FrameDetections`), and it is the only place that calls the
pixel -> table transform (undistort, then homography - see detection/
geometry.py): no subclass has a way to construct a `Detection` directly, so
a detector cannot leave the stage without going through it (REQ-5, REQ-17).

`detect` also rejects a frame whose pixel size doesn't match the
calibration's `inference_resolution`: the homography/camera intrinsics are
only valid for the pixel grid they were solved against, and a
smaller-than-cap frame (REQ-14) can otherwise diverge from it silently.
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
    box_center,
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

    `center` is only authoritative when `box` is `None` (e.g. an ArUco
    marker centroid - REQ-19, which has no box at all). Whenever `box` is
    present, `Detector.detect` recomputes the centre itself via
    `geometry.box_center` and ignores this field: a subclass cannot supply
    a centre that disagrees with Phase 0's verified bounding-box-centre
    method for any detection that has a box (REQ-17).
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
        self._check_resolution(frame)
        raw_detections = self._detect_raw(frame)
        homography = self._calibration.homography
        camera = self._calibration.camera
        distortion = self._calibration.distortion
        detections = [
            Detection(
                object_class=raw.object_class,
                confidence=raw.confidence,
                center=apply_homography_to_point(
                    # A box's centre is always Phase 0's method (REQ-17), never
                    # whatever the subclass put in raw.center: only a boxless
                    # source's own centre is trusted as-is.
                    box_center(raw.box) if raw.box is not None else raw.center,
                    homography,
                    camera,
                    distortion,
                ),
                box=transform_box_to_table(raw.box, homography, camera, distortion)
                if raw.box is not None
                else None,
            )
            for raw in raw_detections
        ]
        return FrameDetections(
            schema_version=DETECTION_SCHEMA_VERSION,
            frame_index=frame.frame_index,
            detections=detections,
        )

    def _check_resolution(self, frame: Frame) -> None:
        # The homography/camera intrinsics are only valid for the pixel grid
        # they were solved against; `apply_resolution_cap` leaves a
        # smaller-than-cap frame unchanged, so its size can silently diverge
        # from that (REQ-14) without this check.
        expected = self._calibration.inference_resolution
        height, width = frame.image.shape[:2]
        if (width, height) != (expected.width, expected.height):
            raise ValueError(
                f"frame resolution {width}x{height} does not match calibration's "
                f"inference_resolution {expected.width}x{expected.height} "
                f"(source: {frame.source_id})"
            )
