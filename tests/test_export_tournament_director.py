"""REQ-36: tournament_director export adapter stub."""

from __future__ import annotations

import logging

from poker_vision.assignment.models import FrameAssignments, ZoneAssignment, ZoneKind
from poker_vision.detection.models import DetectionClass
from poker_vision.export.tournament_director import TournamentDirectorExporter
from poker_vision.state.machine import PipelineStateMachine


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


def test_export_logs_each_event(caplog):
    machine = PipelineStateMachine(["seat_1"])
    exporter = TournamentDirectorExporter()
    events = machine.update(_frame(_chip(1, "seat_1"), frame_index=0))
    assert len(events) == 1

    with caplog.at_level(logging.INFO, logger="poker_vision.export.tournament_director"):
        exporter.export(events)

    assert len(caplog.records) == 1
    assert events[0].model_dump_json() in caplog.records[0].message


def test_export_logs_one_line_per_event_in_order(caplog):
    machine = PipelineStateMachine(["seat_1", "seat_2"])
    exporter = TournamentDirectorExporter()
    events = machine.update(
        _frame(_chip(1, "seat_1"), _chip(2, "seat_2"), frame_index=0)
    )
    assert len(events) == 2

    with caplog.at_level(logging.INFO, logger="poker_vision.export.tournament_director"):
        exporter.export(events)

    assert len(caplog.records) == 2
    for record, event in zip(caplog.records, events, strict=True):
        assert event.model_dump_json() in record.message


def test_export_with_no_events_logs_nothing(caplog):
    exporter = TournamentDirectorExporter()

    with caplog.at_level(logging.INFO, logger="poker_vision.export.tournament_director"):
        exporter.export([])

    assert caplog.records == []


def test_module_imports_no_network_or_windows_dependency():
    # REQ-36: "keine Netzwerk- oder Windows-Abhängigkeit" -- inspect the
    # actual `import` statements (not prose, e.g. this module's own
    # docstring mentions "socket" while explaining why there isn't one)
    # so the adapter's module only ever imports stdlib logging plus the
    # project's own typed event schema.
    import ast

    import poker_vision.export.tournament_director as module

    with open(module.__file__, encoding="utf-8") as f:
        tree = ast.parse(f.read())

    imported_modules = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }

    forbidden = {
        "socket",
        "requests",
        "httpx",
        "urllib",
        "asyncio",
        "ssl",
        "win32",
        "win32com",
        "win32api",
        "pywin32",
        "ctypes",
    }
    assert imported_modules.isdisjoint(forbidden)
    assert imported_modules == {"__future__", "logging", "collections", "poker_vision"}
