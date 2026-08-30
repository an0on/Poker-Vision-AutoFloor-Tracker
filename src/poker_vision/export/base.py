"""Shared export-adapter interface (REQ-36, REQ-37a).

`JsonlEventExporter` (REQ-34) and `WebSocketEventExporter` (REQ-35) already
agree, purely by construction, on the one method `ExportManager` (REQ-37a)
needs to treat every adapter uniformly: `export(events) -> None`.
`EventExporter` names that agreement as a `Protocol` instead of retrofitting
either existing class onto a shared base class -- neither adapter gains any
behavior from being "an EventExporter", and a Protocol lets
`TournamentDirectorExporter` (REQ-36) satisfy the same contract by
structure, without inheriting from anything. Three adapters agreeing on one
method is what makes this worth naming now; it is not a bet on adapters
that don't exist yet.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from poker_vision.state.events import Event


@runtime_checkable
class EventExporter(Protocol):
    def export(self, events: Iterable[Event]) -> None: ...
