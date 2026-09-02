"""REQ-9 / REQ-10: the `calib` CLI (`compile`, `validate`, `create`, `edit`)."""

from __future__ import annotations

import json

import pytest

from poker_vision.calibration.cli import main
from poker_vision.calibration.runtime import load_calibration_runtime


def _create(tmp_path, seats=6, table_id="t"):
    out = tmp_path / "authoring.json"
    exit_code = main(
        ["create", "--out", str(out), "--table-id", table_id, "--seats", str(seats)]
    )
    assert exit_code == 0
    return out


def _seat_player_area_points(authoring_path, seat_id) -> list[dict]:
    document = json.loads(authoring_path.read_text())
    seat = next(s for s in document["seats"] if s["seat_id"] == seat_id)
    return seat["zones"]["player_area"]["points"]


def _shrink_toward_centroid(points: list[dict], factor: float) -> list[dict]:
    cx = sum(p["x"] for p in points) / len(points)
    cy = sum(p["y"] for p in points) / len(points)
    return [
        {"x": cx + factor * (p["x"] - cx), "y": cy + factor * (p["y"] - cy)} for p in points
    ]


def _points_args(points: list[dict]) -> list[str]:
    return [f"{p['x']},{p['y']}" for p in points]


# --- create ------------------------------------------------------------------


def test_create_writes_valid_authoring(tmp_path):
    out = _create(tmp_path, seats=8, table_id="my_table")
    document = json.loads(out.read_text())
    assert document["table_id"] == "my_table"
    assert len(document["seats"]) == 8


# Codex review: `--out` pointing at an unwritable path (missing parent
# directory here) must be a clean CLI error, not an unhandled OSError
# traceback -- covers `compile`/`create`/`edit`'s shared write path.
def test_create_unwritable_out_path_returns_clean_error(tmp_path, capsys):
    out = tmp_path / "no_such_dir" / "out.json"
    exit_code = main(["create", "--out", str(out), "--table-id", "t", "--seats", "6"])
    assert exit_code == 1
    assert capsys.readouterr().err


def test_compile_unwritable_out_path_returns_clean_error(tmp_path, capsys):
    authoring_path = _create(tmp_path)
    out = tmp_path / "no_such_dir" / "runtime.json"
    exit_code = main(["compile", "--authoring", str(authoring_path), "--out", str(out)])
    assert exit_code == 1
    assert capsys.readouterr().err


def test_edit_unwritable_out_path_returns_clean_error(tmp_path, capsys):
    authoring_path = _create(tmp_path)
    out = tmp_path / "no_such_dir" / "edited.json"
    exit_code = main(
        [
            "edit",
            "move-zone",
            "--authoring",
            str(authoring_path),
            "--out",
            str(out),
            "--global-zone",
            "board_zone",
            "--dx",
            "1",
            "--dy",
            "1",
        ]
    )
    assert exit_code == 1
    assert capsys.readouterr().err


def test_create_below_min_seat_count_fails_without_writing_file(tmp_path, capsys):
    out = tmp_path / "authoring.json"
    exit_code = main(["create", "--out", str(out), "--table-id", "t", "--seats", "1"])
    assert exit_code == 1
    assert not out.exists()
    assert "seat_count must be >=" in capsys.readouterr().err


# --- validate ------------------------------------------------------------------


def test_validate_valid_authoring_returns_ok(tmp_path):
    out = _create(tmp_path)
    assert main(["validate", "--authoring", str(out)]) == 0


def test_validate_invalid_authoring_returns_error(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": "9.9"}))
    assert main(["validate", "--authoring", str(bad)]) == 1
    assert "invalid calibration" in capsys.readouterr().err


def test_validate_missing_file_returns_error(tmp_path, capsys):
    missing = tmp_path / "does_not_exist.json"
    assert main(["validate", "--authoring", str(missing)]) == 1
    assert capsys.readouterr().err


# --- compile (REQ-9) ----------------------------------------------------------


def test_compile_writes_loadable_runtime(tmp_path):
    authoring_path = _create(tmp_path)
    runtime_path = tmp_path / "runtime.json"
    exit_code = main(["compile", "--authoring", str(authoring_path), "--out", str(runtime_path)])
    assert exit_code == 0
    runtime = load_calibration_runtime(runtime_path)
    assert runtime.based_on == str(authoring_path)
    assert len(runtime.seats) == 6


def test_compile_is_byte_identical_across_two_runs(tmp_path):
    authoring_path = _create(tmp_path)
    out_a = tmp_path / "a.json"
    out_b = tmp_path / "b.json"
    assert main(["compile", "--authoring", str(authoring_path), "--out", str(out_a)]) == 0
    assert main(["compile", "--authoring", str(authoring_path), "--out", str(out_b)]) == 0
    assert out_a.read_bytes() == out_b.read_bytes()


def test_compile_invalid_authoring_returns_error(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": "9.9"}))
    out = tmp_path / "runtime.json"
    exit_code = main(["compile", "--authoring", str(bad), "--out", str(out)])
    assert exit_code == 1
    assert not out.exists()
    assert capsys.readouterr().err


# --- edit add-seat / remove-seat -----------------------------------------------


def test_edit_add_seat_appends_new_seat(tmp_path):
    authoring_path = _create(tmp_path, seats=3)
    exit_code = main(
        [
            "edit",
            "add-seat",
            "--authoring",
            str(authoring_path),
            "--seat-id",
            "seat_extra",
            "--player-area",
            "1000,1000",
            "1100,1000",
            "1100,1100",
            "1000,1100",
            "--chip-zone",
            "1020,1020",
            "1060,1020",
            "1060,1060",
            "1020,1060",
        ]
    )
    assert exit_code == 0
    document = json.loads(authoring_path.read_text())
    assert "seat_extra" in {s["seat_id"] for s in document["seats"]}
    assert len(document["seats"]) == 4


def test_edit_add_seat_duplicate_seat_id_rejected(tmp_path, capsys):
    authoring_path = _create(tmp_path, seats=3)
    original = authoring_path.read_text()
    exit_code = main(
        [
            "edit",
            "add-seat",
            "--authoring",
            str(authoring_path),
            "--seat-id",
            "seat_1",
            "--player-area",
            "1000,1000",
            "1100,1000",
            "1100,1100",
            "--chip-zone",
            "1020,1020",
            "1040,1020",
            "1040,1040",
        ]
    )
    assert exit_code == 1
    assert "already exists" in capsys.readouterr().err
    assert authoring_path.read_text() == original  # no partial write on failure


def test_edit_remove_seat_removes_it(tmp_path):
    authoring_path = _create(tmp_path, seats=4)
    exit_code = main(
        ["edit", "remove-seat", "--authoring", str(authoring_path), "--seat-id", "seat_2"]
    )
    assert exit_code == 0
    document = json.loads(authoring_path.read_text())
    assert "seat_2" not in {s["seat_id"] for s in document["seats"]}
    assert len(document["seats"]) == 3


def test_edit_remove_seat_unknown_seat_id_rejected(tmp_path, capsys):
    authoring_path = _create(tmp_path, seats=3)
    exit_code = main(
        ["edit", "remove-seat", "--authoring", str(authoring_path), "--seat-id", "does_not_exist"]
    )
    assert exit_code == 1
    assert "no seat with seat_id" in capsys.readouterr().err


# --- edit set-zone / move-zone --------------------------------------------------


def test_edit_set_zone_seat_scoped_updates_points(tmp_path):
    authoring_path = _create(tmp_path, seats=6)
    player_area = _seat_player_area_points(authoring_path, "seat_2")
    new_chip_zone = _shrink_toward_centroid(player_area, 0.2)
    exit_code = main(
        [
            "edit",
            "set-zone",
            "--authoring",
            str(authoring_path),
            "--seat",
            "seat_2",
            "--zone",
            "chip_zone",
            "--points",
            *_points_args(new_chip_zone),
        ]
    )
    assert exit_code == 0
    updated = json.loads(authoring_path.read_text())
    seat = next(s for s in updated["seats"] if s["seat_id"] == "seat_2")
    assert seat["zones"]["chip_zone"]["points"][0]["x"] == pytest.approx(new_chip_zone[0]["x"])


def test_edit_set_zone_rejecting_invalid_geometry_does_not_write(tmp_path, capsys):
    authoring_path = _create(tmp_path, seats=6)
    original = authoring_path.read_text()
    # Far outside seat_2's player_area (which sits within ~1150 of the
    # table origin for this skeleton) -- must violate REQ-11 containment.
    # Large positive coordinates, not negative ones: argparse's nargs='+'
    # treats a leading '-' token as looking like another option rather than
    # a value unless it matches its plain-negative-number regex, which a
    # comma-bearing "x,y" token never does.
    exit_code = main(
        [
            "edit",
            "set-zone",
            "--authoring",
            str(authoring_path),
            "--seat",
            "seat_2",
            "--zone",
            "chip_zone",
            "--points",
            "90000,90000",
            "90100,90000",
            "90100,90100",
        ]
    )
    assert exit_code == 1
    assert "not fully contained" in capsys.readouterr().err
    assert authoring_path.read_text() == original


def test_edit_set_zone_global_zone_updates_points(tmp_path):
    authoring_path = _create(tmp_path, seats=6)
    exit_code = main(
        [
            "edit",
            "set-zone",
            "--authoring",
            str(authoring_path),
            "--global-zone",
            "dealer_area",
            "--points",
            "5,5",
            "15,5",
            "15,15",
            "5,15",
        ]
    )
    assert exit_code == 0
    document = json.loads(authoring_path.read_text())
    assert document["zones"]["dealer_area"]["points"][0] == {"x": 5.0, "y": 5.0}


def test_edit_move_zone_translates_global_zone(tmp_path):
    authoring_path = _create(tmp_path, seats=6)
    before = json.loads(authoring_path.read_text())["zones"]["dealer_area"]["points"]
    exit_code = main(
        [
            "edit",
            "move-zone",
            "--authoring",
            str(authoring_path),
            "--global-zone",
            "dealer_area",
            "--dx",
            "3.5",
            "--dy",
            "-2.0",
        ]
    )
    assert exit_code == 0
    after = json.loads(authoring_path.read_text())["zones"]["dealer_area"]["points"]
    for b, a in zip(before, after, strict=True):
        assert a["x"] == pytest.approx(b["x"] + 3.5)
        assert a["y"] == pytest.approx(b["y"] - 2.0)


def test_edit_zone_target_requires_exactly_one_form(tmp_path, capsys):
    authoring_path = _create(tmp_path, seats=6)
    # Neither --seat/--zone nor --global-zone given.
    exit_code = main(
        ["edit", "move-zone", "--authoring", str(authoring_path), "--dx", "1", "--dy", "1"]
    )
    assert exit_code == 1
    assert "specify either" in capsys.readouterr().err


def test_edit_zone_target_rejects_both_forms_combined(tmp_path, capsys):
    authoring_path = _create(tmp_path, seats=6)
    exit_code = main(
        [
            "edit",
            "move-zone",
            "--authoring",
            str(authoring_path),
            "--seat",
            "seat_1",
            "--zone",
            "chip_zone",
            "--global-zone",
            "board_zone",
            "--dx",
            "1",
            "--dy",
            "1",
        ]
    )
    assert exit_code == 1
    assert "cannot be combined" in capsys.readouterr().err


def test_edit_out_writes_to_separate_file_leaving_original_untouched(tmp_path):
    authoring_path = _create(tmp_path, seats=6)
    original = authoring_path.read_text()
    out_path = tmp_path / "edited.json"
    exit_code = main(
        [
            "edit",
            "move-zone",
            "--authoring",
            str(authoring_path),
            "--out",
            str(out_path),
            "--global-zone",
            "board_zone",
            "--dx",
            "1.0",
            "--dy",
            "1.0",
        ]
    )
    assert exit_code == 0
    assert authoring_path.read_text() == original
    assert out_path.exists()


# --- end-to-end: create -> edit -> validate -> compile -------------------------


def test_full_workflow_create_edit_validate_compile(tmp_path):
    authoring_path = _create(tmp_path, seats=6, table_id="workflow_table")
    assert (
        main(
            [
                "edit",
                "move-zone",
                "--authoring",
                str(authoring_path),
                "--global-zone",
                "board_zone",
                "--dx",
                "2.0",
                "--dy",
                "-1.0",
            ]
        )
        == 0
    )
    assert main(["validate", "--authoring", str(authoring_path)]) == 0
    runtime_path = tmp_path / "runtime.json"
    assert (
        main(["compile", "--authoring", str(authoring_path), "--out", str(runtime_path)]) == 0
    )
    runtime = load_calibration_runtime(runtime_path)
    assert runtime.table_id == "workflow_table"


# --- mark-zones (REQ-10a) -----------------------------------------------------
#
# The actual interactive window/mouse loop needs a display this project has
# no headless-CI equivalent for (see mark_zones_interactive.py's docstring);
# what's testable here is that the CLI parses its arguments and forwards them
# correctly, via monkeypatching the one function `_cmd_mark_zones` calls.


def test_mark_zones_forwards_parsed_arguments(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    def fake_run_interactive_mark_zones(*, image_path, out_path, table_id, chip_zone_inset_pixels):
        captured["image_path"] = image_path
        captured["out_path"] = out_path
        captured["table_id"] = table_id
        captured["chip_zone_inset_pixels"] = chip_zone_inset_pixels
        return 0

    monkeypatch.setattr(
        "poker_vision.calibration.cli.run_interactive_mark_zones",
        fake_run_interactive_mark_zones,
    )
    image_path = tmp_path / "reference.jpg"
    out_path = tmp_path / "authoring.json"
    exit_code = main(
        [
            "mark-zones",
            "--image",
            str(image_path),
            "--out",
            str(out_path),
            "--table-id",
            "dopo_table",
            "--chip-zone-inset-pixels",
            "25",
        ]
    )
    assert exit_code == 0
    assert captured == {
        "image_path": image_path,
        "out_path": out_path,
        "table_id": "dopo_table",
        "chip_zone_inset_pixels": 25.0,
    }


def test_mark_zones_default_chip_zone_inset_pixels(tmp_path, monkeypatch):
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "poker_vision.calibration.cli.run_interactive_mark_zones",
        lambda **kwargs: captured.update(kwargs) or 0,
    )
    main(
        [
            "mark-zones",
            "--image",
            str(tmp_path / "reference.jpg"),
            "--out",
            str(tmp_path / "authoring.json"),
            "--table-id",
            "dopo_table",
        ]
    )
    assert captured["chip_zone_inset_pixels"] == pytest.approx(10.0)
