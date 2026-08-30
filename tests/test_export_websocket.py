"""REQ-35: websocket export adapter + REST status/health (AC-22)."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from poker_vision.assignment.models import FrameAssignments, ZoneAssignment, ZoneKind
from poker_vision.detection.models import DetectionClass
from poker_vision.export.jsonl import JsonlEventExporter
from poker_vision.export.websocket import WebSocketEventExporter
from poker_vision.state.events import EventAdapter
from poker_vision.state.machine import PipelineStateMachine
from poker_vision.state.snapshot import StateSnapshot


def _assignment(
    track_id: int, object_class: DetectionClass, zone: ZoneKind, seat_id: str | None
) -> ZoneAssignment:
    return ZoneAssignment(
        schema_version="1.0",
        track_id=track_id,
        object_class=object_class,
        zone=zone,
        seat_id=seat_id,
    )


def _frame(*assignments: ZoneAssignment, frame_index: int) -> FrameAssignments:
    return FrameAssignments(
        schema_version="1.0", frame_index=frame_index, assignments=list(assignments)
    )


def _chip(track_id: int, seat_id: str) -> ZoneAssignment:
    return _assignment(track_id, DetectionClass.CHIP, ZoneKind.CHIP_ZONE, seat_id)


def _dealer_button(track_id: int, seat_id: str) -> ZoneAssignment:
    return _assignment(track_id, DetectionClass.DEALER_BUTTON, ZoneKind.PLAYER_AREA, seat_id)


def _card(track_id: int) -> ZoneAssignment:
    return _assignment(track_id, DetectionClass.CARD, ZoneKind.BOARD_ZONE, None)


# --- AC-22: first message is a schema-valid snapshot ------------------------


def test_first_websocket_message_is_a_valid_snapshot():
    machine = PipelineStateMachine(["seat_1"])
    machine.update(_frame(_chip(1, "seat_1"), frame_index=0))
    exporter = WebSocketEventExporter(machine)
    client = TestClient(exporter.app)

    with client.websocket_connect("/ws") as ws:
        first_message = ws.receive_text()

    snapshot = StateSnapshot.model_validate_json(first_message)
    assert snapshot == machine.snapshot()


def test_snapshot_on_connect_reflects_state_before_any_new_events():
    machine = PipelineStateMachine(["seat_1"])
    exporter = WebSocketEventExporter(machine)
    client = TestClient(exporter.app)

    with client.websocket_connect("/ws") as ws:
        snapshot = StateSnapshot.model_validate_json(ws.receive_text())

    assert snapshot.hand_active is False
    assert all(not seat.occupied for seat in snapshot.seats)


# --- AC-22: subsequent events match the JSONL file of the same session -----


def test_subsequent_events_match_jsonl_file_of_same_session(tmp_path):
    machine = PipelineStateMachine(["seat_1", "seat_2"])
    ws_exporter = WebSocketEventExporter(machine)
    jsonl_exporter = JsonlEventExporter(tmp_path, session_id="session_a")
    client = TestClient(ws_exporter.app)

    with client.websocket_connect("/ws") as ws:
        ws.receive_text()  # initial snapshot, covered above

        frames = [
            _frame(_chip(1, "seat_1"), _dealer_button(2, "seat_1"), frame_index=0),
            _frame(
                _chip(1, "seat_1"),
                _dealer_button(2, "seat_2"),
                _card(3),
                _card(4),
                _card(5),
                frame_index=1,
            ),
            _frame(frame_index=2),  # empties the board -> hand_ended
        ]

        received: list[str] = []
        for frame in frames:
            events = machine.update(frame)
            ws_exporter.export(events)
            jsonl_exporter.export(events)
            for _ in events:
                received.append(ws.receive_text())

    jsonl_exporter.close()

    jsonl_lines = tmp_path.joinpath("session_a.jsonl").read_text(encoding="utf-8").splitlines()
    assert received == jsonl_lines
    assert len(received) > 0
    for line in received:
        EventAdapter.validate_json(line)


def test_export_with_no_connected_clients_does_not_raise():
    machine = PipelineStateMachine(["seat_1"])
    exporter = WebSocketEventExporter(machine)

    events = machine.update(_frame(_chip(1, "seat_1"), frame_index=0))
    exporter.export(events)  # no client ever connected


def test_export_with_no_events_sends_nothing_extra():
    machine = PipelineStateMachine(["seat_1"])
    exporter = WebSocketEventExporter(machine)
    client = TestClient(exporter.app)

    with client.websocket_connect("/ws") as ws:
        ws.receive_text()  # snapshot
        exporter.export([])
        # A follow-up real event must be the very next message -- proves the
        # empty export() above produced no phantom message ahead of it.
        events = machine.update(_frame(_chip(1, "seat_1"), frame_index=0))
        exporter.export(events)
        message = ws.receive_text()

    assert json.loads(message) == json.loads(events[0].model_dump_json())


# --- REST: GET /status ------------------------------------------------------


def test_status_endpoint_returns_current_snapshot():
    machine = PipelineStateMachine(["seat_1"])
    machine.update(_frame(_chip(1, "seat_1"), frame_index=0))
    exporter = WebSocketEventExporter(machine)
    client = TestClient(exporter.app)

    response = client.get("/status")

    assert response.status_code == 200
    assert StateSnapshot.model_validate(response.json()) == machine.snapshot()


def test_status_endpoint_reflects_updates_between_calls():
    machine = PipelineStateMachine(["seat_1"])
    exporter = WebSocketEventExporter(machine)
    client = TestClient(exporter.app)

    before = client.get("/status").json()
    machine.update(_frame(_chip(1, "seat_1"), frame_index=0))
    after = client.get("/status").json()

    assert before != after
    assert StateSnapshot.model_validate(after) == machine.snapshot()


# --- REST: GET /health -------------------------------------------------------


def test_health_endpoint_is_ok():
    machine = PipelineStateMachine(["seat_1"])
    exporter = WebSocketEventExporter(machine)
    client = TestClient(exporter.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
