"""REQ-38: static HTML debug page (MJPEG + WebSocket event list) (AC-24)."""

from __future__ import annotations

import re

from fastapi.testclient import TestClient

from poker_vision.calibration.camera import CameraIntrinsics, DistortionCoefficients
from poker_vision.calibration.geometry import TableDimensions, TablePoint, TablePolygon, TableUnit
from poker_vision.calibration.homography import HomographyMatrix
from poker_vision.calibration.runtime import CalibrationRuntime
from poker_vision.calibration.zones import CalibrationSeat, GlobalZones, SeatZones
from poker_vision.config import Config, Resolution
from poker_vision.debug.frame_hub import LatestFrameHub
from poker_vision.debug.mjpeg import MjpegDebugServer, build_debug_server

_IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def _polygon(*coords: tuple[float, float]) -> TablePolygon:
    return TablePolygon(points=[TablePoint(x=x, y=y) for x, y in coords])


def _calibration() -> CalibrationRuntime:
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
        card_dealer_seat_id="seat_1",
    )


def _server(websocket_port: int = 8765) -> MjpegDebugServer:
    return MjpegDebugServer(_calibration(), LatestFrameHub(), websocket_port=websocket_port)


def _config(*, debug_enabled: bool = True, websocket_port: int = 8765) -> Config:
    payload = {
        "schema_version": "1.0",
        "device": "cpu",
        "source": {"type": "image_dir", "path": "data/raw/images"},
        "paths": {
            "calibration_authoring": "calibration/instance.json",
            "calibration_runtime": "calibration/runtime.json",
            "jsonl_export_dir": "data/events",
        },
        "ports": {"websocket": websocket_port, "rest": 8000, "mjpeg": 8001},
        "debug": {"enabled": debug_enabled},
    }
    return Config.model_validate(payload)


# --- REQ-38: GET / serves the debug page -------------------------------------


def test_debug_page_is_served_at_root():
    server = _server()
    client = TestClient(server.app)

    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


def test_debug_page_references_the_mjpeg_stream():
    server = _server()
    client = TestClient(server.app)

    body = client.get("/").text

    assert 'src="/mjpeg"' in body


def test_debug_page_bakes_in_the_configured_websocket_port():
    server = _server(websocket_port=9999)
    client = TestClient(server.app)

    body = client.get("/").text

    assert "const WS_PORT = 9999;" in body


def test_different_servers_bake_in_their_own_websocket_port():
    body_a = _server(websocket_port=1111).app
    body_b = _server(websocket_port=2222).app

    text_a = TestClient(body_a).get("/").text
    text_b = TestClient(body_b).get("/").text

    assert "const WS_PORT = 1111;" in text_a
    assert "const WS_PORT = 2222;" in text_b


def test_build_debug_server_wires_config_websocket_port_into_the_page():
    server = build_debug_server(_config(websocket_port=4242), _calibration())
    assert server is not None
    client = TestClient(server.app)

    body = client.get("/").text

    assert "const WS_PORT = 4242;" in body


def test_build_debug_server_returns_none_when_disabled_no_page_either():
    server = build_debug_server(_config(debug_enabled=False), _calibration())
    assert server is None


# --- AC-24: no external dependencies, no frontend framework, no build step ---


_EXTERNAL_SRC_OR_HREF = re.compile(r'(?:src|href)="(https?:)?//')


def test_debug_page_has_no_external_resource_references():
    body = _server().app
    text = TestClient(body).get("/").text

    assert not _EXTERNAL_SRC_OR_HREF.search(text)
    assert "<script src=" not in text
    assert "<link" not in text
    assert "cdn." not in text


def test_debug_page_is_a_single_self_contained_file_on_disk():
    from poker_vision.debug.mjpeg import _DEBUG_PAGE_TEMPLATE

    assert _DEBUG_PAGE_TEMPLATE.exists()
    assert _DEBUG_PAGE_TEMPLATE.suffix == ".html"


# --- REQ-38: page also mentions the events list / snapshot panel -------------


def test_debug_page_has_an_events_container_and_a_snapshot_panel():
    body = TestClient(_server().app).get("/").text

    assert 'id="events"' in body
    assert 'id="snapshot"' in body


def test_debug_page_escapes_wire_values_before_inserting_them_as_html():
    # seat_id is only constrained to be non-empty (CalibrationSeat.seat_id),
    # so a seat/event value can legally contain HTML metacharacters; the
    # page must escape it rather than pass it through innerHTML verbatim.
    body = TestClient(_server().app).get("/").text

    assert "function escapeHtml(value)" in body
    # Every value taken from the wire payload and interpolated into a
    # template string must be wrapped in escapeHtml(...), not inserted raw.
    for raw_interpolation in (
        "${seat.seat}",
        "${snapshot.dealer_seat",
        "${snapshot.street",
        "${event_type}",
        "${sequence}",
        "${key}=${value}",
    ):
        assert raw_interpolation not in body
    assert "escapeHtml(seat.seat)" in body
    assert "escapeHtml(event_type)" in body
    assert "escapeHtml(key)" in body and "escapeHtml(value)" in body
