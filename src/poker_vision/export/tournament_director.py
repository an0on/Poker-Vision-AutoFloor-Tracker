"""`tournament_director` export adapter stub (REQ-36).

The real Tournament Director integration belongs to a later, Windows-phase
milestone (see CLAUDE.md's "Windows/TD-Phase" risk entry) whose wire
protocol isn't known yet. This stub exists so the rest of the export layer
-- Config's per-adapter enable flags and `ExportManager`'s failure
isolation (REQ-37a) -- can already be built and tested against the
adapter's eventual shape (`export(events)`, the same `EventExporter`
contract every other adapter satisfies) without waiting on that protocol.

It performs no network I/O and references no Windows-specific API: enabling
it only logs. Anything more (a real socket, a real message format) is out
of scope until the actual TD interface is known.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from poker_vision.state.events import Event

logger = logging.getLogger(__name__)


class TournamentDirectorExporter:
    """Stub `tournament_director` adapter: logs every event, does nothing else."""

    def export(self, events: Iterable[Event]) -> None:
        for event in events:
            logger.info("tournament_director stub received event: %s", event.model_dump_json())
