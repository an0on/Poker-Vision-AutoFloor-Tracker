"""`mock` detector perturbation wrapper (REQ-21).

Not a fourth standalone mock mode: `PerturbedDetector` wraps any other
`Detector` instance (Modus A/B/C from `mock.py`/`mock_aruco.py`/
`mock_coco.py`, or later `yolo`) and injects configurable, seeded
perturbations into its output, so tracking/hysteresis (REQ-23/REQ-24) can
be tested against reproducible stress scenarios instead of only clean
detections:

- Positions-Jitter: each wrapped detection's table-plane centre gets
  independent Gaussian noise (`PerturbationConfig.position_jitter_std`,
  in the table's own unit -- the same unit `ThresholdsConfig.
  tracking_max_distance` uses, so a scenario can be sized directly against
  that threshold).
- Frame-Dropout: with `dropout_probability`, an entire frame's detections
  are suppressed, as if the camera missed the frame or everything in it
  was occluded.
- Geister-Detections: with `ghost_probability`, one extra detection at a
  uniformly random table position (within the calibration's table bounds)
  is added, with a class drawn from `ghost_classes` -- a detection with no
  counterpart in the wrapped detector's real output.

The wrapper only ever adds/removes detections; it never fabricates pixel
geometry that bypasses the shared pixel -> table transform (REQ-5, REQ-17).
Both the jittered table point and the ghost's table point are round-tripped
through `geometry.apply_inverse_homography_to_point` -- the same mechanism
`mock.py`'s Modus A already uses for its `"table"`-coordinate-space script
entries -- so they re-enter `Detector.detect()`'s ordinary pixel ->
table pipeline like any other raw detection, recovering the intended table
point to well under a pixel (see that function's docstring).
"""

from __future__ import annotations

import random

from poker_vision.calibration.geometry import PixelPoint, TablePoint
from poker_vision.calibration.runtime import CalibrationRuntime
from poker_vision.capture.frame import Frame
from poker_vision.config import PerturbationConfig
from poker_vision.detection.base import Detector, RawDetection
from poker_vision.detection.geometry import apply_inverse_homography_to_point
from poker_vision.detection.models import Detection


class PerturbedDetector(Detector):
    """Wraps another `Detector`, injecting seeded perturbations (REQ-21)."""

    def __init__(
        self,
        calibration: CalibrationRuntime,
        inner: Detector,
        config: PerturbationConfig,
    ) -> None:
        super().__init__(calibration)
        self._inner = inner
        self._config = config
        self._rng = random.Random(config.seed)

    def _detect_raw(self, frame: Frame) -> list[RawDetection]:
        inner_result = self._inner.detect(frame)

        if self._rng.random() < self._config.dropout_probability:
            return []

        raw_detections = [self._jittered(detection) for detection in inner_result.detections]

        if self._rng.random() < self._config.ghost_probability:
            raw_detections.append(self._ghost())

        return raw_detections

    def _jittered(self, detection: Detection) -> RawDetection:
        std = self._config.position_jitter_std
        jittered_center = TablePoint(
            x=detection.center.x + self._rng.gauss(0.0, std),
            y=detection.center.y + self._rng.gauss(0.0, std),
        )
        return RawDetection(
            object_class=detection.object_class,
            confidence=detection.confidence,
            center=self._to_pixel(jittered_center),
            box=None,
        )

    def _ghost(self) -> RawDetection:
        table = self._calibration.table
        ghost_center = TablePoint(
            x=self._rng.uniform(0.0, table.width),
            y=self._rng.uniform(0.0, table.height),
        )
        return RawDetection(
            object_class=self._rng.choice(self._config.ghost_classes),
            confidence=self._config.ghost_confidence,
            center=self._to_pixel(ghost_center),
            box=None,
        )

    def _to_pixel(self, point: TablePoint) -> PixelPoint:
        return apply_inverse_homography_to_point(
            point,
            self._calibration.homography,
            self._calibration.camera,
            self._calibration.distortion,
        )
