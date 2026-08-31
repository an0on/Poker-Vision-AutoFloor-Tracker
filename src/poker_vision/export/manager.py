"""Config-driven adapter composition with per-adapter failure isolation (REQ-37a).

"Ausfall eines Adapters stoppt die Pipeline nicht": `ExportManager.export()`
calls every constructed adapter's `export()` in turn, each wrapped in its
own try/except, so one adapter raising -- deliberately, as in AC-23's
"absichtlich fehlschlagender Adapter" -- doesn't stop the remaining
adapters in the same call from receiving the same events, and doesn't
propagate out to whatever pipeline code called `export()` in the first
place.

`build_exporters()` is kept separate from `ExportManager` itself: which
adapters exist for a given `Config` (REQ-37a's "einzeln aktivierbar") is a
one-shot construction decision, while fanning events out to whatever list
resulted is `ExportManager`'s only job -- the same split `load_config()`
(construction) and `Config` (validation) already use elsewhere in this
project.

"Ausfall eines Adapters" isn't only a failure during `export()`: an
adapter can just as well fail to come up at all (e.g. `jsonl_export_dir`
sits under a read-only or otherwise unwritable path), and that too must
not stop the pipeline -- an unrelated, enabled adapter that *would* have
constructed fine shouldn't be denied a chance to run because a different
adapter's constructor raised first. `build_exporters()` therefore
constructs each enabled adapter under its own try/except, exactly
mirroring `ExportManager.export()`'s per-adapter isolation, and simply
omits any adapter whose constructor failed rather than aborting the whole
batch or unwinding adapters that already succeeded.

`build_exporters()` is also what opens `JsonlEventExporter`'s file handle,
so `ExportManager` -- the only thing holding a reference to that adapter
afterwards -- is the one place left that can release it. `close()` closes
whichever constructed adapters expose a `close()` of their own (currently
just `JsonlEventExporter`; `WebSocketEventExporter` and
`TournamentDirectorExporter` own no such resource and are skipped) via
`getattr` rather than an `EventExporter.close()` method, since `close()`
isn't part of every adapter's contract -- only of the ones that need it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from types import TracebackType

from poker_vision.config import Config
from poker_vision.export.base import EventExporter
from poker_vision.export.jsonl import JsonlEventExporter
from poker_vision.export.tournament_director import TournamentDirectorExporter
from poker_vision.export.websocket import WebSocketEventExporter
from poker_vision.state.events import Event
from poker_vision.state.machine import PipelineStateMachine

logger = logging.getLogger(__name__)


def build_exporters(
    config: Config, state_machine: PipelineStateMachine
) -> list[EventExporter]:
    """Construct one adapter instance per REQ-37a-enabled entry in `config.export`.

    One enabled adapter failing to construct is skipped, logged, and does
    not prevent the other enabled adapters from being built.
    """
    factories: list[tuple[bool, str, Callable[[], EventExporter]]] = [
        (
            config.export.jsonl,
            "jsonl",
            lambda: JsonlEventExporter(config.paths.jsonl_export_dir),
        ),
        (
            config.export.websocket,
            "websocket",
            lambda: WebSocketEventExporter(state_machine),
        ),
        (
            config.export.tournament_director,
            "tournament_director",
            TournamentDirectorExporter,
        ),
    ]
    exporters: list[EventExporter] = []
    for enabled, name, factory in factories:
        if not enabled:
            continue
        try:
            exporters.append(factory())
        except Exception:
            logger.exception("failed to construct %s export adapter; skipping it", name)
    return exporters


class ExportManager:
    """Fans `export()` out to every adapter, isolating each one's failures."""

    def __init__(self, exporters: Iterable[EventExporter]) -> None:
        self._exporters = list(exporters)

    @property
    def exporters(self) -> list[EventExporter]:
        """The constructed adapters, e.g. so a caller can find the one
        `WebSocketEventExporter` instance (if `export.websocket` is
        enabled) to actually serve its FastAPI app -- REQ-45's lifecycle,
        not this class, owns running a real server for it.
        """
        return list(self._exporters)

    def export(self, events: Iterable[Event]) -> None:
        # Materialized once so a one-shot iterable (e.g. a generator) isn't
        # exhausted by the first adapter and left empty for the rest.
        events = list(events)
        for exporter in self._exporters:
            try:
                exporter.export(events)
            except Exception:
                logger.exception(
                    "export adapter %s failed; remaining adapters still ran",
                    type(exporter).__name__,
                )

    def close(self) -> None:
        """Close every constructed adapter that owns a closeable resource.

        One adapter's `close()` raising must not skip the rest, for the same
        reason one adapter's `export()` raising must not skip the rest.
        """
        for exporter in self._exporters:
            close = getattr(exporter, "close", None)
            if close is None:
                continue
            try:
                close()
            except Exception:
                logger.exception(
                    "closing export adapter %s failed", type(exporter).__name__
                )

    def __enter__(self) -> ExportManager:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
