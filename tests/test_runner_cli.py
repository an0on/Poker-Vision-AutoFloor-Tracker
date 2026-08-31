"""REQ-45: `poker-vision run`/`poker-vision validate` CLI argument parsing
and dispatch. `run_command`/`validate_command` themselves are exercised in
`tests/test_runner_lifecycle.py` -- this file only covers `cli.main()`'s
own responsibility: parsing argv and dispatching to them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from poker_vision.runner.cli import EXIT_UNEXPECTED_ERROR, entry_point, main


def test_main_dispatches_run_with_the_parsed_config_path(monkeypatch):
    calls: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        "poker_vision.runner.cli.run_command", lambda path: calls.append(("run", path)) or 0
    )

    exit_code = main(["run", "--config", "cfg.json"])

    assert exit_code == 0
    assert calls == [("run", Path("cfg.json"))]


def test_main_dispatches_validate_with_the_parsed_config_path(monkeypatch):
    calls: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        "poker_vision.runner.cli.validate_command",
        lambda path: calls.append(("validate", path)) or 0,
    )

    exit_code = main(["validate", "--config", "cfg.json"])

    assert exit_code == 0
    assert calls == [("validate", Path("cfg.json"))]


def test_main_returns_the_command_functions_exit_code(monkeypatch):
    monkeypatch.setattr("poker_vision.runner.cli.run_command", lambda path: 4)
    assert main(["run", "--config", "cfg.json"]) == 4


def test_main_requires_a_command():
    with pytest.raises(SystemExit):
        main([])


def test_main_requires_the_config_argument():
    with pytest.raises(SystemExit):
        main(["run"])


def test_main_rejects_an_unknown_command():
    with pytest.raises(SystemExit):
        main(["frobnicate", "--config", "cfg.json"])


def test_main_catches_unexpected_exceptions_as_exit_code_1(monkeypatch):
    def boom(path):
        raise RuntimeError("kaboom")

    monkeypatch.setattr("poker_vision.runner.cli.run_command", boom)

    assert main(["run", "--config", "cfg.json"]) == EXIT_UNEXPECTED_ERROR


def test_entry_point_exits_with_mains_return_code(monkeypatch):
    monkeypatch.setattr("poker_vision.runner.cli.main", lambda: 7)

    with pytest.raises(SystemExit) as exc_info:
        entry_point()

    assert exc_info.value.code == 7
