"""`mock` detector, Modus B: ArUco markers in the image (REQ-19).

Each detected marker's ID is looked up in `ArucoDetectionConfig.marker_class_map`
(REQ-2 config, not a hard-coded mapping) to get its `DetectionClass`; a marker
whose ID has no entry is not one of this project's objects and is skipped
(see `ArucoDetectionConfig`'s docstring). The marker's centre is the mean of
its four corners -- correct for a marker at any rotation, unlike a
bounding-box centre -- and is emitted as a boxless `RawDetection`
(`RawDetection.center` is only authoritative when `box` is `None`, which is
exactly this case; see `detection/base.py`).

ArUco detection is binary (a marker is either found or not); there is no
underlying confidence score the way a trained model would have one, so every
detection is emitted with `confidence=1.0`.
"""

from __future__ import annotations

import cv2

from poker_vision.calibration.geometry import PixelPoint
from poker_vision.calibration.runtime import CalibrationRuntime
from poker_vision.capture.frame import Frame
from poker_vision.config import ArucoDetectionConfig
from poker_vision.detection.base import Detector, RawDetection

# ArUco detection confidence is not a graded score (found or not found), so
# every marker-based detection is reported at full confidence.
_ARUCO_CONFIDENCE = 1.0


class MockArucoDetector(Detector):
    """`mock` detector, Modus B (REQ-19): detections from ArUco markers."""

    def __init__(self, calibration: CalibrationRuntime, config: ArucoDetectionConfig) -> None:
        super().__init__(calibration)
        self._marker_class_map = config.marker_class_map
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.dictionary.value))
        self._aruco_detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

    def _detect_raw(self, frame: Frame) -> list[RawDetection]:
        corners, ids, _rejected = self._aruco_detector.detectMarkers(frame.image)
        if ids is None:
            return []
        raw_detections: list[RawDetection] = []
        for marker_corners, marker_id in zip(corners, ids.flatten(), strict=True):
            object_class = self._marker_class_map.get(int(marker_id))
            if object_class is None:
                continue
            center = marker_corners.reshape(4, 2).mean(axis=0)
            raw_detections.append(
                RawDetection(
                    object_class=object_class,
                    confidence=_ARUCO_CONFIDENCE,
                    center=PixelPoint(x=float(center[0]), y=float(center[1])),
                    box=None,
                )
            )
        return raw_detections
