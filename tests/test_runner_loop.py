"""REQ-44: frame-loop orchestration.

Exercises the full core chain (detection -> tracking -> assignment ->
state) headlessly, with `mock` detection, `image_dir` capture and `jsonl`
export -- no camera, GUI or network (REQ-44's own AC).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from poker_vision.calibration.camera import CameraIntrinsics, DistortionCoefficients
from poker_vision.calibration.geometry import TableDimensions, TablePoint, TablePolygon, TableUnit
from poker_vision.calibration.homography import HomographyMatrix
from poker_vision.calibration.runtime import CalibrationRuntime
from poker_vision.calibration.zones import CalibrationSeat, GlobalZones, SeatZones
from poker_vision.capture.image_dir import ImageDirCapture
from poker_vision.config import HysteresisConfig, Resolution
from poker_vision.detection.mock import MockDetector
from poker_vision.detection.models import Detection, DetectionClass, FrameDetections
from poker_vision.export.jsonl import JsonlEventExporter
from poker_vision.export.manager import ExportManager
from poker_vision.runner.context import FrameContext
from poker_vision.runner.loop import FatalPipelineError, FrameLoop, LoopExitReason
from poker_vision.state.machine import PipelineStateMachine
from poker_vision.tracking.hysteresis import HysteresisFilter
from poker_vision.tracking.tracker import NearestMatchTracker

_IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
_RESOLUTION = Resolution(width=100, height=100)
_TABLE = TableDimensions(width=100.0, height=100.0, unit=TableUnit.CM)
_DEALER_MAX_DISTANCE = 5.0


def _polygon(*coords: tuple[float, float]) -> TablePolygon:
    return TablePolygon(points=[TablePoint(x=x, y=y) for x, y in coords])


# seat_1's chip_zone is (10,10)-(30,30); a chip at (20, 20) lands inside it.
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
        table=_TABLE,
        seats=[_SEAT_1],
        zones=GlobalZones(board_zone=_BOARD_ZONE, dealer_area=_DEALER_AREA),
    )


def _write_script(path: Path, lines: list[dict]) -> Path:
    script_path = path / "script.jsonl"
    with script_path.open("w") as handle:
        for line in lines:
            handle.write(json.dumps(line))
            handle.write("\n")
    return script_path


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


def _make_image_dir(tmp_path: Path, count: int) -> Path:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for i in range(count):
        image = np.full((_RESOLUTION.height, _RESOLUTION.width, 3), i, dtype=np.uint8)
        cv2.imwrite(str(image_dir / f"frame_{i:03d}.png"), image)
    return image_dir


def _build_loop(
    tmp_path: Path,
    script_lines: list[dict],
    image_count: int,
    *,
    n_on: int = 1,
    n_off: int = 1,
    max_consecutive_core_errors: int = 30,
    export_manager: ExportManager | None = None,
    on_frame_processed=None,
) -> tuple[FrameLoop, PipelineStateMachine, NearestMatchTracker, CalibrationRuntime]:
    calibration = _calibration()
    script = _write_script(tmp_path, script_lines)
    detector = MockDetector(calibration, script)
    tracker = NearestMatchTracker(max_distance=5.0, table=calibration.table)
    hysteresis = HysteresisFilter(HysteresisConfig(n_on=n_on, n_off=n_off))
    state_machine = PipelineStateMachine(["seat_1"])
    if export_manager is None:
        export_manager = ExportManager([])
    capture = ImageDirCapture(_make_image_dir(tmp_path, image_count), _RESOLUTION)

    loop = FrameLoop(
        capture=capture,
        detector=detector,
        tracker=tracker,
        hysteresis=hysteresis,
        calibration=calibration,
        dealer_nearest_seat_max_distance=_DEALER_MAX_DISTANCE,
        state_machine=state_machine,
        export_manager=export_manager,
        max_consecutive_core_errors=max_consecutive_core_errors,
        on_frame_processed=on_frame_processed,
    )
    return loop, state_machine, tracker, calibration


# --- headless end-to-end run (mock detection + image_dir + jsonl export) ---


def test_run_processes_frames_headlessly_and_exports_events(tmp_path):
    script_lines = [_chip_entry(i, 20.0, 20.0) for i in range(3)]
    jsonl_exporter = JsonlEventExporter(tmp_path / "exports")
    export_manager = ExportManager([jsonl_exporter])

    loop, state_machine, _tracker, _calib = _build_loop(
        tmp_path, script_lines, image_count=3, export_manager=export_manager
    )

    reason = loop.run()
    jsonl_exporter.close()

    assert reason == LoopExitReason.EOF

    lines = jsonl_exporter.path.read_text().splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["event_type"] == "seat_occupied"
    assert event["seat"] == "seat_1"

    snapshot = state_machine.snapshot()
    assert any(s.seat == "seat_1" and s.occupied for s in snapshot.seats)


def test_process_frame_fills_context_progressively_on_success(tmp_path):
    script_lines = [_chip_entry(0, 20.0, 20.0)]
    loop, _machine, _tracker, _calib = _build_loop(tmp_path, script_lines, image_count=1)

    frame = next(loop._capture)
    context = loop.process_frame(frame)

    assert isinstance(context, FrameContext)
    assert context.succeeded
    assert context.frame_id == 0
    assert context.detections is not None and len(context.detections.detections) == 1
    assert context.tracks is not None and len(context.tracks.tracks) == 1
    assert context.assignments is not None
    assert context.state_snapshot is not None
    assert any(e.event_type == "seat_occupied" for e in context.events)
    assert context.errors == []


# --- EOF on video_file/image_dir ends the loop regularly --------------------


def test_run_returns_eof_when_image_dir_is_exhausted(tmp_path):
    loop, _machine, _tracker, _calib = _build_loop(tmp_path, [], image_count=2)
    assert loop.run() == LoopExitReason.EOF


# --- core-chain exception discards the whole frame, no partial update ------


class _RaisingStateMachine:
    """Test double: `compute_update()` always raises, standing in for a
    core-chain stage failing *after* tracking has already computed (but
    not committed) its own update."""

    def compute_update(self, frame_assignments):
        raise ValueError("synthetic state-stage failure")


def test_a_later_stage_failing_leaves_tracker_state_uncommitted(tmp_path):
    script_lines = [_chip_entry(0, 20.0, 20.0)]
    calibration = _calibration()
    script = _write_script(tmp_path, script_lines)
    detector = MockDetector(calibration, script)
    tracker = NearestMatchTracker(max_distance=5.0, table=calibration.table)
    hysteresis = HysteresisFilter(HysteresisConfig(n_on=1, n_off=1))
    capture = ImageDirCapture(_make_image_dir(tmp_path, 1), _RESOLUTION)

    loop = FrameLoop(
        capture=capture,
        detector=detector,
        tracker=tracker,
        hysteresis=hysteresis,
        calibration=calibration,
        dealer_nearest_seat_max_distance=_DEALER_MAX_DISTANCE,
        state_machine=_RaisingStateMachine(),
        export_manager=ExportManager([]),
    )

    frame = next(loop._capture)
    context = loop.process_frame(frame)

    assert not context.succeeded
    assert "synthetic state-stage failure" in context.errors[0]

    # The tracker's own persistent state was never committed (tracking
    # succeeded and computed a valid update before the state stage raised,
    # but that update must not have been applied) -- a fresh, standalone
    # call with the same detection must mint track_id 1 again, not 2.
    fresh = tracker.update(
        FrameDetections(
            schema_version="1.0",
            frame_index=0,
            detections=[
                Detection(
                    object_class=DetectionClass.CHIP,
                    confidence=0.9,
                    center=TablePoint(x=20.0, y=20.0),
                )
            ],
        )
    )
    assert fresh.tracks[0].track_id == 1


def test_failing_frame_leaves_state_snapshot_and_sequence_unchanged(tmp_path):
    script_lines = [
        _chip_entry(0, 20.0, 20.0),
        # frame 1: a chip far outside the calibrated table -> tracking
        # rejects it before matching runs at all (core-chain failure).
        _chip_entry(1, -50.0, -50.0),
    ]
    loop, state_machine, _tracker, _calib = _build_loop(tmp_path, script_lines, image_count=2)

    frame_0 = next(loop._capture)
    ok_context = loop.process_frame(frame_0)
    assert ok_context.succeeded
    snapshot_before = state_machine.snapshot()

    frame_1 = next(loop._capture)
    bad_context = loop.process_frame(frame_1)
    assert not bad_context.succeeded
    snapshot_after = state_machine.snapshot()

    assert snapshot_before == snapshot_after
    assert snapshot_after.sequence == 1  # only frame 0's seat_occupied consumed a sequence slot


# --- N consecutive core failures abort the loop; a success resets the count


def test_run_raises_fatal_after_max_consecutive_core_errors(tmp_path):
    # Every scripted detection lies outside the calibrated table, so every
    # frame fails in the tracking stage.
    script_lines = [_chip_entry(i, -50.0, -50.0) for i in range(5)]
    loop, _machine, _tracker, _calib = _build_loop(
        tmp_path, script_lines, image_count=5, max_consecutive_core_errors=3
    )

    with pytest.raises(FatalPipelineError, match="3 consecutive"):
        loop.run()


def test_run_resets_the_consecutive_error_count_after_a_success(tmp_path):
    # Fails, fails, succeeds, fails, fails -- never 3 in a row, so the
    # threshold (3) is never crossed and the loop runs to EOF.
    script_lines = [
        _chip_entry(0, -50.0, -50.0),
        _chip_entry(1, -50.0, -50.0),
        _chip_entry(2, 20.0, 20.0),
        _chip_entry(3, -50.0, -50.0),
        _chip_entry(4, -50.0, -50.0),
    ]
    outcomes: list[FrameContext] = []
    loop, _machine, _tracker, _calib = _build_loop(
        tmp_path,
        script_lines,
        image_count=5,
        max_consecutive_core_errors=3,
        on_frame_processed=outcomes.append,
    )

    assert loop.run() == LoopExitReason.EOF
    assert [c.succeeded for c in outcomes] == [False, False, True, False, False]


# --- export failures never stop the loop ------------------------------------


class _AlwaysFailingExporter:
    def export(self, events):
        raise RuntimeError("synthetic export failure")


def test_run_completes_despite_a_failing_export_adapter(tmp_path):
    script_lines = [_chip_entry(0, 20.0, 20.0)]
    export_manager = ExportManager([_AlwaysFailingExporter()])
    loop, _machine, _tracker, _calib = _build_loop(
        tmp_path, script_lines, image_count=1, export_manager=export_manager
    )

    assert loop.run() == LoopExitReason.EOF


# --- debug failures never stop the loop --------------------------------------


class _AlwaysFailingDebugServer:
    def update_frame(self, frame, tracked_frame, frame_assignments):
        raise RuntimeError("synthetic debug rendering failure")


def test_run_completes_despite_a_failing_debug_server(tmp_path):
    script_lines = [_chip_entry(0, 20.0, 20.0)]
    calibration = _calibration()
    script = _write_script(tmp_path, script_lines)
    detector = MockDetector(calibration, script)
    tracker = NearestMatchTracker(max_distance=5.0, table=calibration.table)
    hysteresis = HysteresisFilter(HysteresisConfig(n_on=1, n_off=1))
    capture = ImageDirCapture(_make_image_dir(tmp_path, 1), _RESOLUTION)

    loop = FrameLoop(
        capture=capture,
        detector=detector,
        tracker=tracker,
        hysteresis=hysteresis,
        calibration=calibration,
        dealer_nearest_seat_max_distance=_DEALER_MAX_DISTANCE,
        state_machine=PipelineStateMachine(["seat_1"]),
        export_manager=ExportManager([]),
        debug_server=_AlwaysFailingDebugServer(),
    )

    assert loop.run() == LoopExitReason.EOF
