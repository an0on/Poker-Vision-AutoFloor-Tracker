"""REQ-37: MJPEG debug endpoint (AC-24), rendered on demand from a
`LatestFrameHub` per REQ-46.

The `/mjpeg` route streams forever by design (a live debug feed) and only
stops once the client actually disconnects (`_stream`'s own `request.
is_disconnected()` check) -- which a real browser tab closing triggers, but
`fastapi.testclient.TestClient`'s in-process ASGI transport does not
reliably surface for a stream whose body is never read to completion. So
rather than reading the endpoint through a live HTTP round-trip (which
would hang waiting for a disconnect signal that never arrives), these
tests drive `MjpegDebugServer._stream()` directly against a hand-built
`Request` -- one whose `receive` callable reports "still connected" or
"already disconnected" on demand -- exercising the exact same code path
uvicorn would run, without going through the test client's transport at
all.
"""

from __future__ import annotations

from datetime import UTC, datetime

import anyio
import cv2
import numpy as np
import pytest
from fastapi import Request

from poker_vision.assignment.models import FrameAssignments
from poker_vision.calibration.camera import CameraIntrinsics, DistortionCoefficients
from poker_vision.calibration.geometry import TableDimensions, TablePoint, TablePolygon, TableUnit
from poker_vision.calibration.homography import HomographyMatrix
from poker_vision.calibration.runtime import CalibrationRuntime
from poker_vision.calibration.zones import CalibrationSeat, GlobalZones, SeatZones
from poker_vision.capture.frame import Frame
from poker_vision.config import Config
from poker_vision.debug.frame_hub import DebugSnapshot, LatestFrameHub
from poker_vision.debug.mjpeg import MjpegDebugServer, build_debug_server
from poker_vision.state.machine import PipelineStateMachine
from poker_vision.tracking.models import TrackedFrame

_IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def _polygon(*coords: tuple[float, float]) -> TablePolygon:
    return TablePolygon(points=[TablePoint(x=x, y=y) for x, y in coords])


def _calibration() -> CalibrationRuntime:
    from poker_vision.config import Resolution

    seat = CalibrationSeat(
        seat_id="seat_1",
        zones=SeatZones(
            player_area=_polygon((0, 150), (100, 150), (100, 250), (0, 250)),
            chip_zone=_polygon((10, 160), (50, 160), (50, 200), (10, 200)),
        ),
    )
    return CalibrationRuntime(
        schema_version="1.0",
        table_id="test_table",
        based_on="test",
        inference_resolution=Resolution(width=1920, height=1080),
        camera=CameraIntrinsics(fx=1400.0, fy=1400.0, cx=960.0, cy=540.0),
        distortion=DistortionCoefficients(),
        homography=HomographyMatrix(forward=_IDENTITY, inverse=_IDENTITY),
        table=TableDimensions(width=1200.0, height=900.0, unit=TableUnit.MM),
        seats=[seat],
        zones=GlobalZones(
            board_zone=_polygon((150, 300), (250, 300), (250, 350), (150, 350)),
            dealer_area=_polygon((300, 300), (350, 300), (350, 350), (300, 350)),
        ),
    )


def _frame(frame_index: int = 0) -> Frame:
    return Frame(
        image=np.zeros((400, 400, 3), dtype=np.uint8),
        timestamp=datetime.now(UTC),
        frame_index=frame_index,
        source_id="test",
    )


def _tracked_frame(frame_index: int = 0) -> TrackedFrame:
    return TrackedFrame(schema_version="1.0", frame_index=frame_index, tracks=[])


def _frame_assignments(frame_index: int = 0) -> FrameAssignments:
    return FrameAssignments(schema_version="1.0", frame_index=frame_index, assignments=[])


def _snapshot(frame_index: int = 0) -> DebugSnapshot:
    machine = PipelineStateMachine(["seat_1"])
    return DebugSnapshot(
        tracked_frame=_tracked_frame(frame_index),
        frame_assignments=_frame_assignments(frame_index),
        state_snapshot=machine.snapshot(),
    )


def _server(hub: LatestFrameHub | None = None) -> MjpegDebugServer:
    return MjpegDebugServer(_calibration(), hub if hub is not None else LatestFrameHub())


def _request(*, connected: bool) -> Request:
    """A `Request` whose `receive()` reports "still connected" or "already
    disconnected" on demand, without any real ASGI transport behind it --
    see the module docstring for why `_stream()` is driven directly against
    this instead of through `TestClient`.
    """

    async def receive():
        if connected:
            return {"type": "http.request", "body": b"", "more_body": False}
        return {"type": "http.disconnect"}

    scope = {"type": "http", "method": "GET", "path": "/mjpeg", "headers": [], "query_string": b""}
    return Request(scope, receive)


def _first_part(server: MjpegDebugServer) -> bytes:
    async def run() -> bytes:
        generator = server._stream(_request(connected=True))
        try:
            return await generator.__anext__()
        finally:
            await generator.aclose()

    return anyio.run(run)


# --- REQ-37: FastAPI route wiring --------------------------------------------


def test_mjpeg_route_declares_multipart_x_mixed_replace():
    server = _server()
    route = next(route for route in server.app.routes if route.path == "/mjpeg")
    response = route.endpoint(_request(connected=True))
    assert response.media_type == "multipart/x-mixed-replace; boundary=frame"


# --- AC-24: MJPEG stream carries the rendered overlay ------------------------


def test_stream_yields_nothing_for_an_already_disconnected_request():
    hub = LatestFrameHub()
    server = _server(hub)
    hub.publish(_frame(), _snapshot())

    async def run() -> None:
        generator = server._stream(_request(connected=False))
        with pytest.raises(StopAsyncIteration):
            await generator.__anext__()

    anyio.run(run)


def test_stream_carries_a_decodable_jpeg_frame():
    hub = LatestFrameHub()
    server = _server(hub)
    hub.publish(_frame(), _snapshot())

    part = _first_part(server)
    assert part.startswith(b"--frame\r\n")
    assert b"Content-Type: image/jpeg" in part
    jpeg_bytes = part.split(b"\r\n\r\n", 1)[1].rstrip(b"\r\n")
    decoded = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == (400, 400, 3)


def test_stream_reflects_the_published_state_snapshot():
    hub = LatestFrameHub()
    server = _server(hub)
    hub.publish(_frame(), _snapshot())

    part = _first_part(server)
    jpeg_bytes = part.split(b"\r\n\r\n", 1)[1].rstrip(b"\r\n")
    decoded = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    # The state panel is drawn opaque-black across the top rows regardless
    # of snapshot contents; its white text glyphs prove the state was
    # actually rendered rather than the frame passing through untouched.
    assert decoded[0:100, :].any()


# --- AC-24: `debug.enabled: false` starts no MJPEG endpoint -------------------


def _config(debug_enabled: bool) -> Config:
    payload = {
        "schema_version": "1.0",
        "device": "cpu",
        "source": {"type": "image_dir", "path": "data/raw/images"},
        "paths": {
            "calibration_authoring": "calibration/instance.json",
            "calibration_runtime": "calibration/runtime.json",
            "jsonl_export_dir": "data/events",
        },
        "debug": {"enabled": debug_enabled},
    }
    return Config.model_validate(payload)


def test_build_debug_server_returns_none_when_disabled():
    server = build_debug_server(_config(debug_enabled=False), _calibration())
    assert server is None


def test_build_debug_server_returns_server_when_enabled():
    server = build_debug_server(_config(debug_enabled=True), _calibration())
    assert isinstance(server, MjpegDebugServer)


def test_debug_enabled_defaults_to_true():
    payload = {
        "schema_version": "1.0",
        "device": "cpu",
        "source": {"type": "image_dir", "path": "data/raw/images"},
        "paths": {
            "calibration_authoring": "calibration/instance.json",
            "calibration_runtime": "calibration/runtime.json",
            "jsonl_export_dir": "data/events",
        },
    }
    assert Config.model_validate(payload).debug.enabled is True


def test_debug_config_rejects_unknown_field():
    payload = {
        "schema_version": "1.0",
        "device": "cpu",
        "source": {"type": "image_dir", "path": "data/raw/images"},
        "paths": {
            "calibration_authoring": "calibration/instance.json",
            "calibration_runtime": "calibration/runtime.json",
            "jsonl_export_dir": "data/events",
        },
        "debug": {"enabled": True, "typo_field": True},
    }
    with pytest.raises(Exception):
        Config.model_validate(payload)
