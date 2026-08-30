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


class JsonlEventExporter:
    """Writes `Event`s to `<export_dir>/<session_id>.jsonl`, one per line."""

    def __init__(self, export_dir: Path, session_id: str | None = None) -> None:
        export_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or _default_session_id()
        self.path = export_dir / f"{self.session_id}.jsonl"
        self._file = self.path.open("a", encoding="utf-8")

    def export(self, events: Iterable[Event]) -> None:
        wrote = False
        for event in events:
            self._file.write(event.model_dump_json())
            self._file.write("\n")
            wrote = True
        if wrote:
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
