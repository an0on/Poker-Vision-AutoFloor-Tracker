"""Central runtime configuration (REQ-2).

Every module reads its settings from a `Config` instance instead of
environment variables or hard-coded constants. `Config` and all its nested
models are Pydantic v2 models with `schema_version` and reject unknown
fields (REQ-4).
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, field_validator, model_validator

from poker_vision.detection.models import DetectionClass
from poker_vision.schema_base import StrictModel

CONFIG_SCHEMA_VERSION: Literal["1.0"] = "1.0"


class DeviceType(StrEnum):
    CPU = "cpu"
    MPS = "mps"
    # Reserved for the later Windows/TD phase. Rejected in v0.1 by
    # Config._reject_cuda_device below — this is the only place in `src/`
    # where the string "cuda" may legitimately appear (REQ-3, AC-2).
    CUDA = "cuda"


class SourceType(StrEnum):
    CONTINUITY = "continuity"
    VIDEO_FILE = "video_file"
    IMAGE_DIR = "image_dir"


class DetectorType(StrEnum):
    MOCK = "mock"
    # Registered interface slot for the project-trained model (REQ-22).
    # Rejected in v0.1 by Config._reject_yolo_detector below, the same way
    # DeviceType.CUDA is reserved-but-rejected above (AC-13).
    YOLO = "yolo"


class Resolution(StrictModel):
    """A pixel width/height pair.

    Used both as `SourceConfig.resolution_cap` (REQ-14: the inference
    resolution `capture` scales every frame to) and, reusing the same type,
    as `CalibrationAuthoring`/`CalibrationRuntime`'s `inference_resolution`
    field — so a calibration explicitly references the resolution its pixel
    geometry (homography, zones) was authored against.
    """

    width: int = Field(gt=0)
    height: int = Field(gt=0)


class SourceConfig(StrictModel):
    type: SourceType
    device_index: int | None = Field(default=None, ge=0)
    path: Path | None = None
    resolution_cap: Resolution = Field(default_factory=lambda: Resolution(width=1920, height=1080))

    @model_validator(mode="after")
    def _check_required_field_for_type(self) -> SourceConfig:
        if self.type is SourceType.CONTINUITY and self.device_index is None:
            raise ValueError("source.device_index is required when source.type is 'continuity'")
        if self.type is not SourceType.CONTINUITY and self.path is None:
            raise ValueError(f"source.path is required when source.type is '{self.type.value}'")
        return self


class HysteresisOverride(StrictModel):
    n_on: int | None = Field(default=None, ge=1)
    n_off: int | None = Field(default=None, ge=1)


class HysteresisConfig(StrictModel):
    n_on: int = Field(default=3, ge=1)
    n_off: int = Field(default=3, ge=1)
    per_class: dict[str, HysteresisOverride] = Field(default_factory=dict)


class ThresholdsConfig(StrictModel):
    detection_confidence: float = Field(default=0.25, ge=0.0, le=1.0)
    tracking_max_distance: float = Field(default=0.05, gt=0.0)
    dealer_nearest_seat_max_distance: float = Field(default=0.1, gt=0.0)


class PortsConfig(StrictModel):
    websocket: int = Field(default=8765, ge=1, le=65535)
    rest: int = Field(default=8000, ge=1, le=65535)
    mjpeg: int = Field(default=8001, ge=1, le=65535)

    @model_validator(mode="after")
    def _check_ports_distinct(self) -> PortsConfig:
        values = [self.websocket, self.rest, self.mjpeg]
        if len(set(values)) != len(values):
            raise ValueError("ports.websocket, ports.rest and ports.mjpeg must be distinct")
        return self


class PathsConfig(StrictModel):
    calibration_authoring: Path
    calibration_runtime: Path
    jsonl_export_dir: Path
    mock_script: Path | None = None


class ArucoDictionary(StrEnum):
    """Mirrors OpenCV's predefined ArUco dictionaries by name (REQ-19).

    `mock`'s Modus B (`detection/mock_aruco.py`) looks the matching
    `cv2.aruco.DICT_*` constant up by this member's value, so names here must
    stay identical to OpenCV's own.
    """

    DICT_4X4_50 = "DICT_4X4_50"
    DICT_4X4_100 = "DICT_4X4_100"
    DICT_4X4_250 = "DICT_4X4_250"
    DICT_4X4_1000 = "DICT_4X4_1000"
    DICT_5X5_50 = "DICT_5X5_50"
    DICT_5X5_100 = "DICT_5X5_100"
    DICT_5X5_250 = "DICT_5X5_250"
    DICT_5X5_1000 = "DICT_5X5_1000"
    DICT_6X6_50 = "DICT_6X6_50"
    DICT_6X6_100 = "DICT_6X6_100"
    DICT_6X6_250 = "DICT_6X6_250"
    DICT_6X6_1000 = "DICT_6X6_1000"
    DICT_7X7_50 = "DICT_7X7_50"
    DICT_7X7_100 = "DICT_7X7_100"
    DICT_7X7_250 = "DICT_7X7_250"
    DICT_7X7_1000 = "DICT_7X7_1000"
    DICT_ARUCO_ORIGINAL = "DICT_ARUCO_ORIGINAL"


class ArucoDetectionConfig(StrictModel):
    """`mock` detector Modus B config (REQ-19): marker-ID -> class mapping.

    A marker whose ID has no entry here is not one of this project's
    objects (e.g. a calibration reference marker sharing the same physical
    frame) and is ignored by the detector, the same way a real detector
    simply doesn't report classes outside its trained set.
    """

    dictionary: ArucoDictionary = ArucoDictionary.DICT_4X4_50
    marker_class_map: dict[int, DetectionClass]

    @field_validator("marker_class_map")
    @classmethod
    def _check_nonempty(cls, value: dict[int, DetectionClass]) -> dict[int, DetectionClass]:
        if not value:
            raise ValueError("aruco.marker_class_map must not be empty")
        return value


class CocoDetectionConfig(StrictModel):
    """`mock` detector Modus C config (REQ-20): pretrained COCO model + class mapping.

    A COCO class with no entry here is not one of this project's objects
    (e.g. COCO's `person`) and is ignored by the detector, the same way
    `ArucoDetectionConfig.marker_class_map` ignores an unmapped marker ID.
    """

    model_path: Path = Path("yolov8n.pt")
    class_map: dict[str, DetectionClass]

    @field_validator("class_map")
    @classmethod
    def _check_nonempty(cls, value: dict[str, DetectionClass]) -> dict[str, DetectionClass]:
        if not value:
            raise ValueError("coco.class_map must not be empty")
        return value


class PerturbationConfig(StrictModel):
    """`mock` detector perturbation wrapper config (REQ-21).

    Not a fourth mock mode: wraps any other `Detector` (Modus A/B/C, or
    later `yolo`) to inject reproducible stress into its otherwise-clean
    output, so tracking/hysteresis (REQ-23/REQ-24) can be exercised against
    dropout and noise, not just ideal detections. All perturbations are
    drawn from one `random.Random(seed)`, so the same seed against the same
    wrapped detector reproduces the exact same sequence (REQ-21's "fester
    Seed").
    """

    seed: int
    position_jitter_std: float = Field(default=0.0, ge=0.0)
    dropout_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    ghost_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    ghost_classes: list[DetectionClass] = Field(default_factory=lambda: list(DetectionClass))
    ghost_confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_ghost_classes_present_if_needed(self) -> PerturbationConfig:
        if self.ghost_probability > 0.0 and not self.ghost_classes:
            raise ValueError(
                "perturbation.ghost_classes must not be empty when ghost_probability > 0"
            )
        return self


class Config(StrictModel):
    schema_version: Literal["1.0"]
    device: DeviceType
    detector: DetectorType = DetectorType.MOCK
    source: SourceConfig
    paths: PathsConfig
    thresholds: ThresholdsConfig = Field(default_factory=ThresholdsConfig)
    hysteresis: HysteresisConfig = Field(default_factory=HysteresisConfig)
    ports: PortsConfig = Field(default_factory=PortsConfig)
    aruco: ArucoDetectionConfig | None = None
    coco: CocoDetectionConfig | None = None
    perturbation: PerturbationConfig | None = None

    @field_validator("device")
    @classmethod
    def _reject_cuda_device(cls, value: DeviceType) -> DeviceType:
        if value is DeviceType.CUDA:
            raise ValueError(
                "device 'cuda' is reserved and not supported in v0.1; use 'cpu' or 'mps'"
            )
        return value

    @field_validator("detector")
    @classmethod
    def _reject_yolo_detector(cls, value: DetectorType) -> DetectorType:
        if value is DetectorType.YOLO:
            raise ValueError(
                "detector 'yolo' is not available in v0.1 (no trained model yet, "
                "planned for v0.2); use 'mock'"
            )
        return value


def load_config(path: str | Path) -> Config:
    """Load and validate a Config from a JSON file. Raises on any schema violation."""
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{config_path}: not valid JSON ({exc})") from exc
    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"{config_path}: invalid config ({exc})") from exc
