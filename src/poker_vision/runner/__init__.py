"""Pipeline orchestration: frame loop, lifecycle, CLI entry point (REQ-44/45/46).

The only module that depends on every pipeline stage -- no stage imports
`runner` (see `runner/loop.py`'s module docstring).
"""

from __future__ import annotations

from poker_vision.runner.context import FrameContext
from poker_vision.runner.lifecycle import (
    ContinuityRetryExhausted,
    ShutdownController,
    run_command,
    validate_command,
)
from poker_vision.runner.loop import FatalPipelineError, FrameLoop, LoopExitReason

__all__ = [
    "ContinuityRetryExhausted",
    "FatalPipelineError",
    "FrameContext",
    "FrameLoop",
    "LoopExitReason",
    "ShutdownController",
    "run_command",
    "validate_command",
]
