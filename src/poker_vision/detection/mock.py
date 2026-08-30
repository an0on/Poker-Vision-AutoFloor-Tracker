"""`mock` detector, Modus A: detections from a JSONL script (REQ-18).

The script is a JSON-Lines file, one line per frame: `{"frame_index": int,
"detections": [...]}`. Each detection carries an explicit
`coordinate_space` marker, `"pixel"` or `"table"` (REQ-18's "wahlweise in
Pixel- oder Tischkoordinaten mit Kennzeichnung"):

- `"pixel"` entries are raw image-pixel coordinates, exactly like a real
  detector's output before `Detector.detect()`'s pixel -> table transform;
  an optional `box` is also pixel-space.
- `"table"` entries are already table-plane coordinates (e.g. hand-picked
  against a calibration's zones). Since every `_detect_raw` implementation
  must still return pixel-space `RawDetection`s (REQ-5, REQ-17), such an
  entry is converted back to pixel space with
  `geometry.apply_inverse_homography_to_point` -- the exact inverse of what
  `Detector.detect()` will apply to it -- so it round-trips to the original
  table point (well under a pixel of numerical error; see AC-11). A `box`
  is not supported for `"table"` entries.

A frame index with no line in the script yields no detections; the script
only needs a line for a frame index where something is present.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, ValidationError, model_validator

from poker_vision.calibration.geometry import PixelPoint, TablePoint
from poker_vision.calibration.runtime import CalibrationRuntime
from poker_vision.capture.frame import Frame
from poker_vision.detection.base import Detector, RawDetection
from poker_vision.detection.geometry import PixelBox, apply_inverse_homography_to_point
from poker_vision.detection.models import DetectionClass
from poker_vision.schema_base import StrictModel


class MockPixelBox(StrictModel):
    """A pixel-space box in the script format, as (x1, y1) top-left / (x2, y2) bottom-right."""

    x1: float = Field(allow_inf_nan=False)
    y1: float = Field(allow_inf_nan=False)
    x2: float = Field(allow_inf_nan=False)
    y2: float = Field(allow_inf_nan=False)

    @model_validator(mode="after")
    def _check_ordered(self) -> MockPixelBox:
        if self.x1 > self.x2 or self.y1 > self.y2:
            raise ValueError("box x1/y1 must be <= x2/y2")
        return self


class MockPixelDetection(StrictModel):
    coordinate_space: Literal["pixel"] = "pixel"
    object_class: DetectionClass
    confidence: float = Field(ge=0.0, le=1.0)
    center: PixelPoint
    box: MockPixelBox | None = None


class MockTableDetection(StrictModel):
    coordinate_space: Literal["table"] = "table"
    object_class: DetectionClass
    confidence: float = Field(ge=0.0, le=1.0)
    center: TablePoint


MockScriptDetection = Annotated[
    MockPixelDetection | MockTableDetection,
    Field(discriminator="coordinate_space"),
]


class MockScriptFrame(StrictModel):
    """One line of the mock script: a frame index and its detections."""

    frame_index: int = Field(ge=0)
    detections: list[MockScriptDetection] = Field(default_factory=list)


def _load_mock_script(path: Path) -> dict[int, list[MockPixelDetection | MockTableDetection]]:
    frames: dict[int, list[MockPixelDetection | MockTableDetection]] = {}
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: not valid JSON ({exc})") from exc
        try:
            frame = MockScriptFrame.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(f"{path}:{line_number}: invalid mock script entry ({exc})") from exc
        if frame.frame_index in frames:
            raise ValueError(
                f"{path}:{line_number}: duplicate frame_index {frame.frame_index} "
                "(each frame index may appear at most once in the script)"
            )
        frames[frame.frame_index] = frame.detections
    return frames


class MockDetector(Detector):
    """`mock` detector, Modus A (REQ-18): detections read from a JSONL script."""

    def __init__(self, calibration: CalibrationRuntime, script_path: str | Path) -> None:
        super().__init__(calibration)
        self._frames = _load_mock_script(Path(script_path))

    def _detect_raw(self, frame: Frame) -> list[RawDetection]:
        entries = self._frames.get(frame.frame_index, [])
        return [self._to_raw_detection(entry) for entry in entries]

    def _to_raw_detection(
        self, entry: MockPixelDetection | MockTableDetection
    ) -> RawDetection:
        if isinstance(entry, MockPixelDetection):
            box: PixelBox | None = (
                (entry.box.x1, entry.box.y1, entry.box.x2, entry.box.y2)
                if entry.box is not None
                else None
            )
            return RawDetection(
                object_class=entry.object_class,
                confidence=entry.confidence,
                center=entry.center,
                box=box,
            )
        pixel_center = apply_inverse_homography_to_point(
            entry.center,
            self._calibration.homography,
            self._calibration.camera,
            self._calibration.distortion,
        )
        return RawDetection(
            object_class=entry.object_class,
            confidence=entry.confidence,
            center=pixel_center,
            box=None,
        )
