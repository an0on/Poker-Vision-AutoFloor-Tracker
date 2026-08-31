"""CLI entry point (REQ-45): `poker-vision run --config <path>` /
`poker-vision validate --config <path>`.

Thin argument parsing only -- every actual behavior (loading config,
loading calibration, constructing stages, the signal-driven shutdown, exit
code mapping) lives in `runner.lifecycle`, which is also what tests drive
directly rather than always going through a subprocess.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from poker_vision.runner.lifecycle import EXIT_UNEXPECTED_ERROR, run_command, validate_command


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="poker-vision")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Start the pipeline")
    run_parser.add_argument("--config", required=True, type=Path)

    validate_parser = subparsers.add_parser(
        "validate", help="Validate config + calibration without starting the pipeline"
    )
    validate_parser.add_argument("--config", required=True, type=Path)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _build_parser().parse_args(argv)

    try:
        if args.command == "run":
            return run_command(args.config)
        return validate_command(args.config)
    except Exception:
        # A safety net, not part of REQ-45's documented exit-code scheme
        # (`lifecycle`'s own module docstring): every condition that
        # scheme covers is already caught inside `run_command`/
        # `validate_command` themselves. Anything that still reaches here
        # is a genuine bug, not an expected abort -- logged with its full
        # traceback (unlike every other exit path here) so it's still
        # diagnosable, instead of an unhandled traceback with no exit-code
        # contract at all.
        logging.getLogger(__name__).exception("unexpected error")
        return EXIT_UNEXPECTED_ERROR


def entry_point() -> None:
    sys.exit(main())


if __name__ == "__main__":
    entry_point()
