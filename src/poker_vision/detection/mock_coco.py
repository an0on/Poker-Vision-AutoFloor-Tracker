"""`mock` detector, Modus C: pretrained COCO model (REQ-20).

Runs an off-the-shelf COCO model (default `yolov8n.pt`, no training, no
project-specific weights) and maps each detected COCO class name through
`CocoDetectionConfig.class_map` (REQ-2 config, not a hard-coded mapping) to
this project's `DetectionClass` -- e.g. `mouse` -> `dealer_button`,
`cell phone` -> `chip`, continuing Phase 0's placeholder mapping
(`phase0_poc.py`). A COCO class with no entry in `class_map` is not one of
this project's objects and is skipped, the same way Modus B
(`mock_aruco.py`) skips an unmapped marker ID.

Device comes from `Config.device` (REQ-3: `cpu`/`mps` only, the other
reserved value already rejected by `Config` itself), never resolved here.
The confidence threshold is `Config.thresholds.detection_confidence`, the
same field Phase 0's `--conf` defaulted to (0.25).

Every raw box's centre is computed via `geometry.box_center` -- Phase 0's
verified exact-bounding-box-centre method -- before being handed to
`Detector.detect()`, which recomputes it identically and applies the pixel
-> table transform (REQ-17). This is what AC-11 checks against
`phase0_poc.py` on the Phase 0 test image (Toleranz 1 px im Pixelraum).
"""

from __future__ import annotations

from ultralytics import YOLO

from poker_vision.calibration.runtime import CalibrationRuntime
from poker_vision.capture.frame import Frame
from poker_vision.config import CocoDetectionConfig, DeviceType
from poker_vision.detection.base import Detector, RawDetection
from poker_vision.detection.geometry import PixelBox, box_center


class CocoMockDetector(Detector):
    """`mock` detector, Modus C (REQ-20): a pretrained COCO model + class mapping."""

    def __init__(
        self,
        calibration: CalibrationRuntime,
        config: CocoDetectionConfig,
        device: DeviceType,
        confidence_threshold: float,
    ) -> None:
        super().__init__(calibration)
        self._class_map = config.class_map
        self._device = device.value
        self._confidence_threshold = confidence_threshold
        self._model = YOLO(str(config.model_path))

    def _detect_raw(self, frame: Frame) -> list[RawDetection]:
        results = self._model.predict(
            source=frame.image,
            device=self._device,
            conf=self._confidence_threshold,
            verbose=False,
        )
        names = self._model.names
        raw_detections: list[RawDetection] = []
        for box in results[0].boxes:
            coco_class = names[int(box.cls[0])]
            object_class = self._class_map.get(coco_class)
            if object_class is None:
                continue
            pixel_box: PixelBox = tuple(float(v) for v in box.xyxy[0].tolist())
            raw_detections.append(
                RawDetection(
                    object_class=object_class,
                    confidence=float(box.conf[0]),
                    center=box_center(pixel_box),
                    box=pixel_box,
                )
            )
        return raw_detections
