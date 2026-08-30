"""`jsonl` export adapter (REQ-34): append-only event log, one file per session.

`JsonlEventExporter` only ever accepts `Event` instances (state/events.py's
typed, schema-validated union), so writing `event.model_dump_json()` is by
construction the same JSON the event's own schema accepts back -- there is
no path through this adapter for a frame, an image crop, or any other
payload to reach disk (AC-21's "keine Bilddaten").

The file is opened in append mode and never truncated, rewritten, or
reordered: every `export()` call writes its events in the order given, one
JSON object per line, immediately flushed so a crash mid-session loses at
most the write in flight, not prior lines. Preserving call order is what
keeps the file's `sequence` values gapless and ascending (AC-21) --
`PipelineStateMachine` (REQ-33) is the one that guarantees gaplessness in
the first place, this adapter just never reorders or drops what it's given.

That guarantee assumes one file maps to one `PipelineStateMachine`'s
lifetime: the architecture keeps all pipeline state in-memory only (no
persisted/resumed state machine), so a process restart is expected to start
a new session (new `session_id`/file), not reopen an old one's sequence
mid-stream. Reopening an existing `session_id` against a *fresh* state
machine is out of scope here -- it would need sequence-recovery logic this
adapter deliberately doesn't have.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType

from poker_vision.state.events import Event


def _default_session_id() -> str:
    # Colon-free and Windows-safe (see CLAUDE.md: capture must stay portable
    # to a later Windows/TD phase) while still sorting chronologically.
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _validate_session_id(session_id: str) -> str:
    # An explicit session_id may eventually come from runtime config
    # (REQ-37a); rejecting anything but a plain filename component keeps it
    # from ever escaping export_dir via "../" or an absolute path.
    if not session_id or session_id in {".", ".."} or Path(session_id).name != session_id:
        raise ValueError(
            f"invalid session_id: {session_id!r} (must be a plain filename, no path separators)"
        )
    return session_id


class JsonlEventExporter:
    """Writes `Event`s to `<export_dir>/<session_id>.jsonl`, one per line."""

    def __init__(self, export_dir: Path, session_id: str | None = None) -> None:
        export_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = (
            _validate_session_id(session_id) if session_id is not None else _default_session_id()
        )
        self.path = export_dir / f"{self.session_id}.jsonl"
        self._file = self.path.open("a", encoding="utf-8")

    def export(self, events: Iterable[Event]) -> None:
        for event in events:
            # One write() call per line: a crash between writing the JSON
            # payload and its trailing newline would otherwise leave a
            # malformed trailing line that fails schema validation on
            # re-read. Flushing per event bounds any crash to at most the
            # one event in flight, matching this module's docstring.
            self._file.write(event.model_dump_json() + "\n")
            self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> JsonlEventExporter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
