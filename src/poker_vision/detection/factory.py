"""Detection stage: builds the configured `Detector` implementation (REQ-17).

Deliberately its own module, not `detection/__init__.py`: `Config` (REQ-2)
itself imports `poker_vision.detection.models` (for `DetectionClass`), and
importing *any* submodule of `poker_vision.detection` -- `models`
included -- always runs `detection/__init__.py` first (parent-package
import). Putting this factory's `poker_vision.config`/`poker_vision.
calibration.runtime` imports there would make importing `detection.models`
transitively import `Config` back, a circular import
(`config -> detection.models -> detection (__init__) -> config`) that
only "works" by accident of which module happens to import first. Kept
here instead, `detection/__init__.py` stays empty and this module is
never imported as a side effect of importing `detection.models` alone --
only `runner.lifecycle` imports it, explicitly.
"""

from __future__ import annotations

from poker_vision.calibration.runtime import CalibrationRuntime
from poker_vision.config import Config
from poker_vision.detection.base import Detector
from poker_vision.detection.mock import MockDetector
from poker_vision.detection.mock_aruco import MockArucoDetector
from poker_vision.detection.mock_perturbation import PerturbedDetector

__all__ = ["create_detector"]


def create_detector(config: Config, calibration: CalibrationRuntime) -> Detector:
    """Build `config.detector`'s implementation -- in v0.1 always `mock`,
    in one of its three modes (A/B/C -- REQ-18/19/20), optionally wrapped
    in `PerturbedDetector` (REQ-21).

    `Config.detector == DetectorType.YOLO` never reaches here: `Config`
    itself already rejects that value at load time (REQ-22). Which of the
    three mock modes applies is selected by *which* of `paths.mock_script`
    / `aruco` / `coco` is set -- `Config` leaves all three optional and
    independent (so e.g. `aruco` and `perturbation` can both be exercised
    in isolation by other stages' own tests), so exactly-one-of-three isn't
    a schema-level constraint; an ambiguous or empty selection is reported
    here instead, as part of "Stufen konstruieren", still strictly before
    the frame loop ever starts.
    """
    modes = [
        config.paths.mock_script is not None,
        config.aruco is not None,
        config.coco is not None,
    ]
    if sum(modes) != 1:
        raise ValueError(
            "exactly one of paths.mock_script, aruco, coco must be set to select "
            "the mock detector's mode (A/B/C)"
        )

    detector: Detector
    if config.paths.mock_script is not None:
        detector = MockDetector(calibration, config.paths.mock_script)
    elif config.aruco is not None:
        detector = MockArucoDetector(calibration, config.aruco)
    else:
        assert config.coco is not None  # narrowed by the exactly-one check above
        # Imported lazily: `mock_coco.py` pulls in `ultralytics` at module
        # scope, a multi-second import cost that every other mode (A/B) --
        # and every CLI invocation that doesn't select Modus C at all --
        # would otherwise always pay just for importing this module.
        from poker_vision.detection.mock_coco import CocoMockDetector

        detector = CocoMockDetector(
            calibration, config.coco, config.device, config.thresholds.detection_confidence
        )

    if config.perturbation is not None:
        detector = PerturbedDetector(calibration, detector, config.perturbation)
    return detector
