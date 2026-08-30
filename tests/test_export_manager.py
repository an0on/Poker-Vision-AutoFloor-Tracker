"""REQ-37a: per-adapter Config enablement + failure isolation (AC-23)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from poker_vision.assignment.models import FrameAssignments, ZoneAssignment, ZoneKind
from poker_vision.config import Config
from poker_vision.detection.models import DetectionClass
from poker_vision.export.jsonl import JsonlEventExporter
from poker_vision.export.manager import ExportManager, build_exporters
from poker_vision.export.tournament_director import TournamentDirectorExporter
from poker_vision.export.websocket import WebSocketEventExporter
from poker_vision.state.events import Event
from poker_vision.state.machine import PipelineStateMachine

VALID_CONFIG: dict = {
    "schema_version": "1.0",
    "device": "cpu",
    "source": {"type": "image_dir", "path": "data/raw/images"},
    "paths": {
        "calibration_authoring": "calibration/instance.json",
        "calibration_runtime": "calibration/runtime.json",
        "jsonl_export_dir": "data/events",
    },
}


def _config(export_overrides: dict | None = None, jsonl_export_dir=None) -> Config:
    payload = json.loads(json.dumps(VALID_CONFIG))
    if export_overrides is not None:
        payload["export"] = export_overrides
    if jsonl_export_dir is not None:
        payload["paths"]["jsonl_export_dir"] = str(jsonl_export_dir)
    return Config.model_validate(payload)


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


class _RecordingExporter:
    def __init__(self) -> None:
        self.received: list[list[Event]] = []

    def export(self, events) -> None:
        self.received.append(list(events))


class _FailingExporter:
    def export(self, events) -> None:
        raise RuntimeError("deliberately failing adapter")


# --- REQ-37a: adapters are individually enableable via Config --------------


def test_all_adapters_enabled_by_default_except_tournament_director(tmp_path):
    machine = PipelineStateMachine(["seat_1"])
    config = _config(jsonl_export_dir=tmp_path)

    exporters = build_exporters(config, machine)

    assert [type(e) for e in exporters] == [JsonlEventExporter, WebSocketEventExporter]


def test_all_three_adapters_enabled(tmp_path):
    machine = PipelineStateMachine(["seat_1"])
    config = _config(
        export_overrides={"jsonl": True, "websocket": True, "tournament_director": True},
        jsonl_export_dir=tmp_path,
    )

    exporters = build_exporters(config, machine)

    assert [type(e) for e in exporters] == [
        JsonlEventExporter,
        WebSocketEventExporter,
        TournamentDirectorExporter,
    ]


def test_all_adapters_disabled_yields_empty_list(tmp_path):
    machine = PipelineStateMachine(["seat_1"])
    config = _config(
        export_overrides={"jsonl": False, "websocket": False, "tournament_director": False},
        jsonl_export_dir=tmp_path,
    )

    assert build_exporters(config, machine) == []


@pytest.mark.parametrize(
    "enabled_key,expected_type",
    [
        ("jsonl", JsonlEventExporter),
        ("websocket", WebSocketEventExporter),
        ("tournament_director", TournamentDirectorExporter),
    ],
)
def test_only_one_adapter_enabled_at_a_time(tmp_path, enabled_key, expected_type):
    machine = PipelineStateMachine(["seat_1"])
    overrides = {"jsonl": False, "websocket": False, "tournament_director": False}
    overrides[enabled_key] = True
    config = _config(export_overrides=overrides, jsonl_export_dir=tmp_path)

    exporters = build_exporters(config, machine)

    assert [type(e) for e in exporters] == [expected_type]


# --- ExportManager: fans events out to every configured adapter ------------


def test_export_manager_dispatches_to_every_adapter():
    a, b = _RecordingExporter(), _RecordingExporter()
    manager = ExportManager([a, b])
    machine = PipelineStateMachine(["seat_1"])

    events = machine.update(_frame(_chip(1, "seat_1"), frame_index=0))
    manager.export(events)

    assert a.received == [events]
    assert b.received == [events]


def test_export_manager_with_no_adapters_does_not_raise():
    manager = ExportManager([])
    machine = PipelineStateMachine(["seat_1"])

    manager.export(machine.update(_frame(_chip(1, "seat_1"), frame_index=0)))


# --- REQ-37a: a failing adapter does not stop the others (AC-23) -----------


@pytest.mark.parametrize("failing_position", [0, 1, 2])
def test_a_failing_adapter_does_not_stop_the_others(failing_position):
    recorders = [_RecordingExporter(), _RecordingExporter()]
    adapters = list(recorders)
    adapters.insert(failing_position, _FailingExporter())
    manager = ExportManager(adapters)
    machine = PipelineStateMachine(["seat_1"])

    events = machine.update(_frame(_chip(1, "seat_1"), frame_index=0))
    manager.export(events)  # must not raise

    for recorder in recorders:
        assert recorder.received == [events]


def test_failing_adapter_does_not_interrupt_jsonl_or_websocket(tmp_path):
    # AC-23: "absichtlich fehlschlagender Adapter unterbricht weder JSONL
    # noch WebSocket" -- exercised with the real adapters, not fakes.
    machine = PipelineStateMachine(["seat_1"])
    jsonl_exporter = JsonlEventExporter(tmp_path, session_id="session_a")
    websocket_exporter = WebSocketEventExporter(machine)
    manager = ExportManager([_FailingExporter(), jsonl_exporter, websocket_exporter])
    client = TestClient(websocket_exporter.app)

    with client.websocket_connect("/ws") as ws:
        ws.receive_text()  # initial snapshot

        events = machine.update(_frame(_chip(1, "seat_1"), frame_index=0))
        manager.export(events)  # must not raise despite the failing adapter

        received = ws.receive_text()

    jsonl_exporter.close()

    assert json.loads(received) == json.loads(events[0].model_dump_json())
    lines = tmp_path.joinpath("session_a.jsonl").read_text(encoding="utf-8").splitlines()
    assert lines == [events[0].model_dump_json()]
