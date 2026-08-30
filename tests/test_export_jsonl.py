"""REQ-34: jsonl export adapter (AC-21)."""

from __future__ import annotations

import json

import pytest

from poker_vision.assignment.models import FrameAssignments, ZoneAssignment, ZoneKind
from poker_vision.detection.models import DetectionClass
from poker_vision.export.jsonl import JsonlEventExporter
from poker_vision.state.events import EventAdapter
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


def _dealer_button(track_id: int, seat_id: str) -> ZoneAssignment:
    return _assignment(track_id, DetectionClass.DEALER_BUTTON, ZoneKind.PLAYER_AREA, seat_id)


def _card(track_id: int) -> ZoneAssignment:
    return _assignment(track_id, DetectionClass.CARD, ZoneKind.BOARD_ZONE, None)


def _read_lines(path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


# --- one event per line, in call order --------------------------------------


def test_export_writes_one_json_line_per_event(tmp_path):
    machine = PipelineStateMachine(["seat_1"])
    exporter = JsonlEventExporter(tmp_path, session_id="session_a")

    events = machine.update(_frame(_chip(1, "seat_1"), _card(2), _card(3), _card(4), frame_index=0))
    exporter.export(events)
    exporter.close()

    lines = _read_lines(exporter.path)
    assert len(lines) == len(events) == 3
    for line, event in zip(lines, events, strict=True):
        assert json.loads(line) == json.loads(event.model_dump_json())


# --- append-only: repeated export() calls never overwrite prior lines ------


def test_export_appends_across_multiple_calls_and_is_never_truncated(tmp_path):
    machine = PipelineStateMachine(["seat_1"])
    exporter = JsonlEventExporter(tmp_path, session_id="session_a")

    first = machine.update(_frame(_chip(1, "seat_1"), frame_index=0))
    exporter.export(first)
    second = machine.update(_frame(frame_index=1))
    exporter.export(second)

    exporter.close()

    lines = _read_lines(exporter.path)
    assert len(lines) == 2

    # A second exporter instance for the same still-live machine and
    # session_id (e.g. a second export sink, not a process restart -- state
    # is in-memory only per the architecture doc, so a real restart starts a
    # new session/file rather than resuming this machine's sequence) appends
    # rather than truncating what's already on disk.
    second_exporter = JsonlEventExporter(tmp_path, session_id="session_a")
    third = machine.update(_frame(_chip(2, "seat_1"), frame_index=2))
    second_exporter.export(third)
    second_exporter.close()

    assert _read_lines(exporter.path) == [*lines, third[0].model_dump_json()]


def test_export_with_no_events_writes_nothing(tmp_path):
    exporter = JsonlEventExporter(tmp_path, session_id="session_a")
    exporter.export([])
    exporter.close()

    assert exporter.path.read_text(encoding="utf-8") == ""


# --- one file per session ----------------------------------------------------


def test_distinct_sessions_get_distinct_files(tmp_path):
    exporter_a = JsonlEventExporter(tmp_path, session_id="session_a")
    exporter_b = JsonlEventExporter(tmp_path, session_id="session_b")

    assert exporter_a.path != exporter_b.path
    assert exporter_a.path.parent == exporter_b.path.parent == tmp_path

    exporter_a.close()
    exporter_b.close()


def test_default_session_id_is_generated_when_omitted(tmp_path):
    exporter = JsonlEventExporter(tmp_path)

    assert exporter.session_id
    assert exporter.path.exists()

    exporter.close()


@pytest.mark.parametrize("session_id", ["../escaped", "sub/session", "/absolute", ".", "..", ""])
def test_rejects_session_id_that_would_escape_export_dir(tmp_path, session_id):
    with pytest.raises(ValueError, match="session_id"):
        JsonlEventExporter(tmp_path, session_id=session_id)

    # Nothing outside export_dir was created by the rejected attempt.
    assert list(tmp_path.iterdir()) == []


def test_export_dir_is_created_if_missing(tmp_path):
    export_dir = tmp_path / "does" / "not" / "exist" / "yet"

    exporter = JsonlEventExporter(export_dir, session_id="session_a")
    exporter.close()

    assert exporter.path.parent == export_dir


# --- AC-21: gapless ascending sequence, schema-valid lines, no image data ---


def test_sequence_values_in_file_are_gapless_and_ascending(tmp_path):
    machine = PipelineStateMachine(["seat_1", "seat_2"])
    exporter = JsonlEventExporter(tmp_path, session_id="session_a")

    exporter.export(
        machine.update(
            _frame(
                _chip(1, "seat_1"),
                _dealer_button(2, "seat_1"),
                _card(3),
                _card(4),
                _card(5),
                frame_index=0,
            )
        )
    )
    # A quiet frame emits no events -- must not create a gap or a blank line.
    exporter.export(machine.update(_frame(_chip(1, "seat_1"), frame_index=1)))
    exporter.export(
        machine.update(
            _frame(
                _chip(1, "seat_1"),
                _dealer_button(2, "seat_2"),
                _card(3),
                _card(4),
                _card(5),
                _card(6),
                frame_index=2,
            )
        )
    )
    exporter.close()

    sequences = [json.loads(line)["sequence"] for line in _read_lines(exporter.path)]
    assert sequences == list(range(len(sequences)))


def test_every_line_validates_against_the_event_schema(tmp_path):
    machine = PipelineStateMachine(["seat_1"])
    exporter = JsonlEventExporter(tmp_path, session_id="session_a")

    exporter.export(
        machine.update(
            _frame(
                _chip(1, "seat_1"),
                _dealer_button(2, "seat_1"),
                _card(3),
                _card(4),
                _card(5),
                frame_index=0,
            )
        )
    )
    exporter.close()

    lines = _read_lines(exporter.path)
    assert lines  # sanity: this frame does emit events
    for line in lines:
        EventAdapter.validate_json(line)


def test_file_contains_no_image_data(tmp_path):
    machine = PipelineStateMachine(["seat_1"])
    exporter = JsonlEventExporter(tmp_path, session_id="session_a")

    exporter.export(machine.update(_frame(_chip(1, "seat_1"), frame_index=0)))
    exporter.close()

    lines = _read_lines(exporter.path)
    assert lines
    allowed_keys = {
        "schema_version",
        "sequence",
        "timestamp",
        "frame_index",
        "event_type",
        "seat",
        "from_seat",
        "to_seat",
        "hand_id",
        "street",
    }
    for line in lines:
        assert set(json.loads(line).keys()) <= allowed_keys
