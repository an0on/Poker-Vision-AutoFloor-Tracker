"""Pipeline orchestration: frame loop, lifecycle, CLI entry point (REQ-44/45/46).

The only module that depends on every pipeline stage -- no stage imports
`runner` (see `runner/loop.py`'s module docstring). REQ-44's frame loop is
implemented here now; `lifecycle.py`/`cli.py` (REQ-45) follow separately.
"""

from __future__ import annotations

from poker_vision.runner.context import FrameContext
from poker_vision.runner.loop import FatalPipelineError, FrameLoop, LoopExitReason

__all__ = [
    "FatalPipelineError",
    "FrameContext",
    "FrameLoop",
    "LoopExitReason",
]
