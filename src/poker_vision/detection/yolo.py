"""`yolo` detector (REQ-22): registered `Detector` implementation, no model yet.

v0.1 has no project-trained model (own chips, cards, dealer button) to back
this class -- that is the eventual v0.2 deliverable (dataset pipeline,
training, CoreML export; see `tools/`). This class exists only to reserve
the `yolo` slot on the `Detector` interface so the later real implementation
drops in without changing callers.

`Config` already rejects `detector: yolo` at load time (see
`Config._reject_yolo_detector`), so in practice nothing in v0.1 reaches this
class. It raises the same explicit, v0.2-pointing error itself so that
constructing it directly -- bypassing `Config` -- fails just as clearly
instead of silently yielding a `Detector` that can't detect anything.
"""

from __future__ import annotations

from poker_vision.calibration.runtime import CalibrationRuntime
from poker_vision.capture.frame import Frame
from poker_vision.detection.base import Detector, RawDetection

_NOT_AVAILABLE_MESSAGE = (
    "yolo detector is not available in v0.1 (no trained model yet, "
    "planned for v0.2); use 'mock'"
)


class YoloDetector(Detector):
    """Placeholder for the project-trained YOLO model (REQ-22). Unusable in v0.1."""

    def __init__(self, calibration: CalibrationRuntime) -> None:
        raise NotImplementedError(_NOT_AVAILABLE_MESSAGE)

    def _detect_raw(self, frame: Frame) -> list[RawDetection]:
        raise NotImplementedError(_NOT_AVAILABLE_MESSAGE)
