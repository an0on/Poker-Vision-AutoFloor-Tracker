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
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

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
    """Construct one adapter instance per REQ-37a-enabled entry in `config.export`."""
    exporters: list[EventExporter] = []
    if config.export.jsonl:
        exporters.append(JsonlEventExporter(config.paths.jsonl_export_dir))
    if config.export.websocket:
        exporters.append(WebSocketEventExporter(state_machine))
    if config.export.tournament_director:
        exporters.append(TournamentDirectorExporter())
    return exporters


class ExportManager:
    """Fans `export()` out to every adapter, isolating each one's failures."""

    def __init__(self, exporters: Iterable[EventExporter]) -> None:
        self._exporters = list(exporters)

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
