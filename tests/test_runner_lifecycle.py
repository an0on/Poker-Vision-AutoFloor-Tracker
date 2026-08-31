"""REQ-45: CLI-facing lifecycle -- stage construction, exit codes,
SIGINT/SIGTERM shutdown, and continuity retry-with-backoff.
"""

from __future__ import annotations

import json
import signal
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from poker_vision.calibration.camera import CameraIntrinsics, DistortionCoefficients
from poker_vision.calibration.geometry import TableDimensions, TablePoint, TablePolygon, TableUnit
from poker_vision.calibration.homography import HomographyMatrix
from poker_vision.calibration.runtime import CalibrationRuntime
from poker_vision.calibration.zones import CalibrationSeat, GlobalZones, SeatZones
from poker_vision.capture.frame import Frame
from poker_vision.capture.image_dir import ImageDirCapture
from poker_vision.config import (
    Config,
    ContinuityRetryConfig,
    HysteresisConfig,
    PathsConfig,
    Resolution,
    SourceConfig,
    SourceType,
)
from poker_vision.detection.mock import MockDetector
from poker_vision.export.jsonl import JsonlEventExporter
from poker_vision.export.manager import ExportManager
from poker_vision.runner.lifecycle import (
    EXIT_CALIBRATION_ERROR,
    EXIT_CONFIG_ERROR,
    EXIT_FORCED_ABORT,
    EXIT_OK,
    EXIT_PIPELINE_ERROR,
    ContinuityRetryExhausted,
    ShutdownController,
    _build_stages,
    _handle_continuity_failure,
    _RetryWindow,
    _run_capture_with_retry,
    run_command,
    validate_command,
)
from poker_vision.runner.loop import FatalPipelineError, FrameLoop, LoopExitReason
from poker_vision.state.machine import PipelineStateMachine
from poker_vision.tracking.hysteresis import HysteresisFilter
from poker_vision.tracking.tracker import NearestMatchTracker

_IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
_RESOLUTION = Resolution(width=100, height=100)
_DEALER_MAX_DISTANCE = 5.0


def _polygon(*coords: tuple[float, float]) -> TablePolygon:
    return TablePolygon(points=[TablePoint(x=x, y=y) for x, y in coords])


_SEAT_1 = CalibrationSeat(
    seat_id="seat_1",
    zones=SeatZones(
        player_area=_polygon((0, 0), (50, 0), (50, 50), (0, 50)),
        chip_zone=_polygon((10, 10), (30, 10), (30, 30), (10, 30)),
    ),
)
_BOARD_ZONE = _polygon((60, 60), (90, 60), (90, 90), (60, 90))
_DEALER_AREA = _polygon((0, 60), (20, 60), (20, 80), (0, 80))


def _calibration() -> CalibrationRuntime:
    return CalibrationRuntime(
        schema_version="1.0",
        table_id="test_table",
        based_on="test",
        inference_resolution=_RESOLUTION,
        camera=CameraIntrinsics(fx=1000.0, fy=1000.0, cx=50.0, cy=50.0),
        distortion=DistortionCoefficients(),
        homography=HomographyMatrix(forward=_IDENTITY, inverse=_IDENTITY),
        table=TableDimensions(width=100.0, height=100.0, unit=TableUnit.CM),
        seats=[_SEAT_1],
        zones=GlobalZones(board_zone=_BOARD_ZONE, dealer_area=_DEALER_AREA),
    )


def _chip_entry(frame_index: int, x: float, y: float) -> dict:
    return {
        "frame_index": frame_index,
        "detections": [
            {
                "coordinate_space": "table",
                "object_class": "chip",
                "confidence": 0.9,
                "center": {"x": x, "y": y},
            }
        ],
    }


def _write_script(path: Path, lines: list[dict]) -> Path:
    script_path = path / "script.jsonl"
    with script_path.open("w") as handle:
        for line in lines:
            handle.write(json.dumps(line))
            handle.write("\n")
    return script_path


def _make_image_dir(tmp_path: Path, count: int) -> Path:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for i in range(count):
        image = np.full((_RESOLUTION.height, _RESOLUTION.width, 3), i, dtype=np.uint8)
        cv2.imwrite(str(image_dir / f"frame_{i:03d}.png"), image)
    return image_dir


def _frame(frame_index: int) -> Frame:
    image = np.zeros((_RESOLUTION.height, _RESOLUTION.width, 3), dtype=np.uint8)
    return Frame(
        image=image, timestamp=datetime.now(UTC), frame_index=frame_index, source_id="test"
    )


def _stages(
    tmp_path: Path, script_lines: list[dict], *, n_on: int = 1, n_off: int = 1
) -> tuple[
    CalibrationRuntime,
    MockDetector,
    NearestMatchTracker,
    HysteresisFilter,
    PipelineStateMachine,
    ExportManager,
]:
    calibration = _calibration()
    script = _write_script(tmp_path, script_lines)
    detector = MockDetector(calibration, script)
    tracker = NearestMatchTracker(max_distance=5.0, table=calibration.table)
    hysteresis = HysteresisFilter(HysteresisConfig(n_on=n_on, n_off=n_off))
    state_machine = PipelineStateMachine(["seat_1"])
    export_manager = ExportManager([])
    return calibration, detector, tracker, hysteresis, state_machine, export_manager


def _continuity_config(**retry_kwargs: object) -> Config:
    return Config(
        schema_version="1.0",
        device="cpu",
        source=SourceConfig(
            type=SourceType.CONTINUITY,
            device_index=0,
            continuity_retry=ContinuityRetryConfig(**retry_kwargs),
        ),
        paths=PathsConfig(
            calibration_authoring="a.json", calibration_runtime="r.json", jsonl_export_dir="e"
        ),
    )


def _image_dir_config(path: Path) -> Config:
    return Config(
        schema_version="1.0",
        device="cpu",
        source=SourceConfig(type=SourceType.IMAGE_DIR, path=path),
        paths=PathsConfig(
            calibration_authoring="a.json",
            calibration_runtime="r.json",
            jsonl_export_dir="e",
            mock_script="s.jsonl",
        ),
    )


class _ScriptedCapture:
    """A `Capture` test double driven by a fixed script of frames/exceptions
    (`StopIteration` implicitly once the script runs out)."""

    source_id = "scripted"

    def __init__(self, script: list[Frame | Exception]) -> None:
        self._script = list(script)
        self.closed = False

    def __iter__(self) -> _ScriptedCapture:
        return self

    def __next__(self) -> Frame:
        if not self._script:
            raise StopIteration
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self) -> None:
        self.closed = True


# --- _RetryWindow / _handle_continuity_failure -------------------------------


def test_retry_window_exhausts_after_continuous_failure_exceeds_timeout():
    retry_config = ContinuityRetryConfig(backoff_seconds=0.01, timeout_seconds=0.05)
    shutdown = ShutdownController()
    window = _RetryWindow()

    results = []
    for _ in range(30):
        ok = _handle_continuity_failure(RuntimeError("x"), window, retry_config, shutdown, "read")
        results.append(ok)
        if not ok:
            break

    assert results[-1] is False
    assert all(results[:-1])


def test_retry_window_reset_extends_the_effective_budget():
    retry_config = ContinuityRetryConfig(backoff_seconds=0.01, timeout_seconds=0.05)
    shutdown = ShutdownController()
    window = _RetryWindow()

    assert _handle_continuity_failure(RuntimeError("x"), window, retry_config, shutdown, "read")
    window.reset()
    time.sleep(0.06)  # would have exhausted the original (un-reset) window
    assert _handle_continuity_failure(RuntimeError("x"), window, retry_config, shutdown, "read")


def test_handle_continuity_failure_gives_up_immediately_when_shutdown_requested():
    retry_config = ContinuityRetryConfig(backoff_seconds=10.0, timeout_seconds=10.0)
    shutdown = ShutdownController()
    shutdown._event.set()
    window = _RetryWindow()

    assert not _handle_continuity_failure(RuntimeError("x"), window, retry_config, shutdown, "read")


# --- _run_capture_with_retry ---------------------------------------------------


def test_run_capture_with_retry_reopens_after_a_read_failure(tmp_path, monkeypatch):
    calibration, detector, tracker, hysteresis, state_machine, export_manager = _stages(
        tmp_path, [_chip_entry(0, 20.0, 20.0)]
    )
    attempts: list[object] = []

    def fake_create_capture(source):
        attempts.append(source)
        if len(attempts) == 1:
            return _ScriptedCapture([_frame(0), RuntimeError("read failed")])
        return _ScriptedCapture([])

    monkeypatch.setattr("poker_vision.runner.lifecycle.create_capture", fake_create_capture)
    config = _continuity_config(backoff_seconds=0.01, timeout_seconds=2.0)
    shutdown = ShutdownController()

    reason = _run_capture_with_retry(
        config,
        shutdown,
        calibration,
        detector,
        tracker,
        hysteresis,
        state_machine,
        export_manager,
        None,
    )

    assert reason == LoopExitReason.EOF
    assert len(attempts) == 2


def test_run_capture_with_retry_never_retries_the_very_first_open_failure(tmp_path, monkeypatch):
    calibration, detector, tracker, hysteresis, state_machine, export_manager = _stages(
        tmp_path, []
    )

    def fake_create_capture(source):
        raise RuntimeError("camera not available")

    monkeypatch.setattr("poker_vision.runner.lifecycle.create_capture", fake_create_capture)
    config = _continuity_config(backoff_seconds=0.01, timeout_seconds=5.0)
    shutdown = ShutdownController()

    with pytest.raises(RuntimeError, match="camera not available"):
        _run_capture_with_retry(
            config,
            shutdown,
            calibration,
            detector,
            tracker,
            hysteresis,
            state_machine,
            export_manager,
            None,
        )


def test_run_capture_with_retry_gives_up_once_timeout_exceeded(tmp_path, monkeypatch):
    calibration, detector, tracker, hysteresis, state_machine, export_manager = _stages(
        tmp_path, []
    )
    attempts: list[object] = []

    def fake_create_capture(source):
        attempts.append(source)
        if len(attempts) == 1:
            return _ScriptedCapture([RuntimeError("boom")])
        raise RuntimeError("still broken")

    monkeypatch.setattr("poker_vision.runner.lifecycle.create_capture", fake_create_capture)
    config = _continuity_config(backoff_seconds=0.02, timeout_seconds=0.08)
    shutdown = ShutdownController()

    with pytest.raises(ContinuityRetryExhausted):
        _run_capture_with_retry(
            config,
            shutdown,
            calibration,
            detector,
            tracker,
            hysteresis,
            state_machine,
            export_manager,
            None,
        )
    assert len(attempts) >= 2


def test_run_capture_with_retry_never_retries_non_continuity_sources(tmp_path, monkeypatch):
    calibration, detector, tracker, hysteresis, state_machine, export_manager = _stages(
        tmp_path, []
    )

    def fake_create_capture(source):
        return _ScriptedCapture([RuntimeError("boom")])

    monkeypatch.setattr("poker_vision.runner.lifecycle.create_capture", fake_create_capture)
    config = _image_dir_config(tmp_path / "images")
    shutdown = ShutdownController()

    with pytest.raises(RuntimeError, match="boom"):
        _run_capture_with_retry(
            config,
            shutdown,
            calibration,
            detector,
            tracker,
            hysteresis,
            state_machine,
            export_manager,
            None,
        )


def test_run_capture_with_retry_propagates_fatal_pipeline_error_without_retrying(
    tmp_path, monkeypatch
):
    # Every scripted detection lies outside the calibrated table, so every
    # frame fails in the tracking stage -- a core-chain problem, not a
    # capture problem, so it must never be treated as a continuity outage.
    calibration, detector, tracker, hysteresis, state_machine, export_manager = _stages(
        tmp_path,
        [_chip_entry(i, -50.0, -50.0) for i in range(5)],
    )

    def fake_create_capture(source):
        return _ScriptedCapture([_frame(i) for i in range(5)])

    monkeypatch.setattr("poker_vision.runner.lifecycle.create_capture", fake_create_capture)
    config = _continuity_config(backoff_seconds=0.01, timeout_seconds=5.0)
    config.runner.max_consecutive_core_errors = 3
    shutdown = ShutdownController()

    with pytest.raises(FatalPipelineError):
        _run_capture_with_retry(
            config,
            shutdown,
            calibration,
            detector,
            tracker,
            hysteresis,
            state_machine,
            export_manager,
            None,
        )


# --- ShutdownController -------------------------------------------------------


def test_shutdown_controller_first_sigint_sets_event_without_forcing_exit(monkeypatch):
    controller = ShutdownController()
    exits: list[int] = []
    monkeypatch.setattr("poker_vision.runner.lifecycle.os._exit", exits.append)

    controller._handle(signal.SIGINT, None)

    assert controller.requested() is True
    assert exits == []


def test_shutdown_controller_second_sigint_forces_immediate_exit(monkeypatch):
    controller = ShutdownController()
    exits: list[int] = []
    monkeypatch.setattr("poker_vision.runner.lifecycle.os._exit", exits.append)

    controller._handle(signal.SIGINT, None)
    controller._handle(signal.SIGINT, None)

    assert exits == [EXIT_FORCED_ABORT]


def test_shutdown_controller_sigterm_never_forces_exit(monkeypatch):
    controller = ShutdownController()
    exits: list[int] = []
    monkeypatch.setattr("poker_vision.runner.lifecycle.os._exit", exits.append)

    controller._handle(signal.SIGTERM, None)
    controller._handle(signal.SIGTERM, None)
    controller._handle(signal.SIGTERM, None)

    assert controller.requested() is True
    assert exits == []


def test_shutdown_controller_install_and_restore_round_trip_handlers():
    original = signal.getsignal(signal.SIGINT)
    controller = ShutdownController()
    try:
        controller.install()
        # Bound methods aren't singletons (`x.m is x.m` is False even for
        # the same underlying method), so compare by equality instead.
        assert signal.getsignal(signal.SIGINT) == controller._handle
    finally:
        controller.restore()
    assert signal.getsignal(signal.SIGINT) is original


# --- shutdown mid-run: current frame completes, export is flushed & valid ----


def test_shutdown_mid_run_flushes_a_complete_valid_jsonl_file(tmp_path):
    calibration = _calibration()
    script_lines = [_chip_entry(i, 20.0, 20.0) for i in range(5)]
    script = _write_script(tmp_path, script_lines)
    image_dir = _make_image_dir(tmp_path, 5)
    export_dir = tmp_path / "exports"

    shutdown = ShutdownController()

    class _StopAfterTwoCapture:
        source_id = "test"

        def __init__(self, inner) -> None:
            self._inner = inner
            self._count = 0

        def __iter__(self):
            return self

        def __next__(self) -> Frame:
            frame = next(self._inner)
            self._count += 1
            if self._count == 2:
                # Simulate a SIGINT landing right after this frame was
                # captured -- the loop must still finish processing it.
                shutdown._event.set()
            return frame

        def close(self) -> None:
            self._inner.close()

    capture = _StopAfterTwoCapture(ImageDirCapture(image_dir, _RESOLUTION))
    detector = MockDetector(calibration, script)
    tracker = NearestMatchTracker(max_distance=5.0, table=calibration.table)
    hysteresis = HysteresisFilter(HysteresisConfig(n_on=1, n_off=1))
    state_machine = PipelineStateMachine(["seat_1"])
    jsonl_exporter = JsonlEventExporter(export_dir)
    export_manager = ExportManager([jsonl_exporter])

    loop = FrameLoop(
        capture=capture,
        detector=detector,
        tracker=tracker,
        hysteresis=hysteresis,
        calibration=calibration,
        dealer_nearest_seat_max_distance=_DEALER_MAX_DISTANCE,
        state_machine=state_machine,
        export_manager=export_manager,
    )

    reason = loop.run(should_stop=shutdown.requested)
    export_manager.close()

    assert reason == LoopExitReason.SHUTDOWN_REQUESTED
    files = list(export_dir.glob("*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text().splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["event_type"] == "seat_occupied"
    assert event["seat"] == "seat_1"


# --- run_command / validate_command exit codes --------------------------------


def _valid_setup(tmp_path: Path) -> tuple[Path, Path]:
    calibration = _calibration()
    calib_path = tmp_path / "calibration.json"
    calib_path.write_text(calibration.model_dump_json())
    script = _write_script(tmp_path, [_chip_entry(i, 20.0, 20.0) for i in range(3)])
    image_dir = _make_image_dir(tmp_path, 3)
    export_dir = tmp_path / "exports"
    config = {
        "schema_version": "1.0",
        "device": "cpu",
        "source": {"type": "image_dir", "path": str(image_dir)},
        "paths": {
            "calibration_authoring": str(calib_path),
            "calibration_runtime": str(calib_path),
            "jsonl_export_dir": str(export_dir),
            "mock_script": str(script),
        },
        "debug": {"enabled": False},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    return config_path, export_dir


def test_run_command_full_pipeline_succeeds_and_exports(tmp_path):
    config_path, export_dir = _valid_setup(tmp_path)

    exit_code = run_command(config_path)

    assert exit_code == EXIT_OK
    files = list(export_dir.glob("*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["event_type"] == "seat_occupied"


def test_run_command_invalid_config_returns_config_error(tmp_path):
    bad_config = tmp_path / "config.json"
    bad_config.write_text("{not valid json")
    assert run_command(bad_config) == EXIT_CONFIG_ERROR


def test_run_command_missing_config_file_returns_config_error(tmp_path):
    assert run_command(tmp_path / "does_not_exist.json") == EXIT_CONFIG_ERROR


def test_run_command_invalid_calibration_returns_calibration_error(tmp_path):
    config_path, _ = _valid_setup(tmp_path)
    config = json.loads(config_path.read_text())
    Path(config["paths"]["calibration_runtime"]).write_text("{not valid json")

    assert run_command(config_path) == EXIT_CALIBRATION_ERROR


def test_run_command_ambiguous_detector_mode_returns_config_error(tmp_path):
    config_path, _ = _valid_setup(tmp_path)
    config = json.loads(config_path.read_text())
    del config["paths"]["mock_script"]
    config_path.write_text(json.dumps(config))

    assert run_command(config_path) == EXIT_CONFIG_ERROR


def test_run_command_missing_capture_source_returns_pipeline_error(tmp_path):
    config_path, _ = _valid_setup(tmp_path)
    config = json.loads(config_path.read_text())
    config["source"]["path"] = str(tmp_path / "no_such_directory")
    config_path.write_text(json.dumps(config))

    assert run_command(config_path) == EXIT_PIPELINE_ERROR


def test_validate_command_valid_setup_returns_ok(tmp_path):
    config_path, _ = _valid_setup(tmp_path)
    assert validate_command(config_path) == EXIT_OK


def test_validate_command_invalid_config_returns_config_error(tmp_path):
    bad_config = tmp_path / "config.json"
    bad_config.write_text("not json")
    assert validate_command(bad_config) == EXIT_CONFIG_ERROR


def test_validate_command_invalid_calibration_returns_calibration_error(tmp_path):
    config_path, _ = _valid_setup(tmp_path)
    config = json.loads(config_path.read_text())
    Path(config["paths"]["calibration_runtime"]).write_text("not json")

    assert validate_command(config_path) == EXIT_CALIBRATION_ERROR


def test_validate_command_never_constructs_stages(tmp_path):
    # An ambiguous detector mode only breaks *stage construction*
    # (`_build_stages`, used by `run`) -- `validate` only checks config +
    # calibration (REQ-45's own AC), so it must still report success.
    config_path, _ = _valid_setup(tmp_path)
    config = json.loads(config_path.read_text())
    del config["paths"]["mock_script"]
    config_path.write_text(json.dumps(config))

    assert validate_command(config_path) == EXIT_OK


def test_build_stages_raises_value_error_for_unbuildable_detector(tmp_path):
    calibration = _calibration()
    config = _image_dir_config(tmp_path / "images")
    config.paths.mock_script = None

    with pytest.raises(ValueError, match="exactly one of"):
        _build_stages(config, calibration)
