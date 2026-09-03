"""`calib` CLI: authoring-JSON tooling (REQ-9, REQ-10).

Two families of subcommand:

- `calib compile`: `CalibrationAuthoring` -> `CalibrationRuntime` (REQ-9),
  via `calibration.compile.compile_calibration`.
- `calib validate` / `calib create` / `calib edit`: create and edit the
  authoring JSON itself (REQ-10), replacing every
  `build_landscape_calibration_instance*.py` variant with this one tool.
  `edit`'s subcommands (`add-seat`, `remove-seat`, `set-zone`, `move-zone`)
  always round-trip through `CalibrationAuthoring.model_validate` on the
  full, modified document before writing anything back, so every edit is
  checked against REQ-11's zone/topology validation the same way loading
  the file always is -- there is no separate "skip validation" path.
- `calib mark-zones`: interactive click-based authoring against a
  reference photo (REQ-10a) -- the actual geometry work lives in
  `mark_zones.py`/`mark_zones_session.py`, this subcommand just delegates
  to `mark_zones_interactive.run_interactive_mark_zones`.
- `calib learn-table`: automatic runtime calibration for a new photo of the
  same physical table design (REQ-10b), via feature matching against the
  reference photo instead of re-running `mark-zones` -- delegates to
  `learn_table.learn_table_calibration`.

Thin argument parsing + JSON/dict plumbing only; the actual geometry (zone
topology, homography solving) lives in `calibration/zones.py`,
`calibration/topology.py` and `calibration/compile.py`, exactly like
`runner/cli.py` delegates its real behavior to `runner/lifecycle.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from poker_vision.calibration.authoring import (
    CalibrationAuthoring,
    load_calibration_authoring,
    write_calibration_authoring,
)
from poker_vision.calibration.compile import compile_calibration
from poker_vision.calibration.geometry import TableUnit
from poker_vision.calibration.learn_table import (
    DEFAULT_CENTER_STRIP_MARGIN_RATIO,
    DEFAULT_MIN_INLIER_RATIO,
    DEFAULT_MIN_MATCH_COUNT,
    DEFAULT_RANSAC_REPROJ_THRESHOLD_PIXELS,
    LearnTableConfig,
    learn_table_calibration,
)
from poker_vision.calibration.mark_zones import DEFAULT_CHIP_ZONE_INSET_PIXELS
from poker_vision.calibration.mark_zones_interactive import run_interactive_mark_zones
from poker_vision.calibration.runtime import load_calibration_runtime, write_calibration_runtime
from poker_vision.calibration.skeleton import MIN_SEAT_COUNT, build_authoring_skeleton
from poker_vision.config import Resolution

EXIT_OK = 0
EXIT_ERROR = 1

_GLOBAL_ZONE_NAMES = {"board_zone", "dealer_area"}
_SEAT_ZONE_NAMES = {"player_area", "chip_zone"}


def _parse_point(raw: str) -> dict[str, float]:
    parts = raw.split(",")
    if len(parts) != 2:
        raise ValueError(f"invalid point '{raw}': expected 'x,y'")
    try:
        x, y = float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise ValueError(f"invalid point '{raw}': expected numeric 'x,y'") from exc
    return {"x": x, "y": y}


def _find_seat(document: dict, seat_id: str) -> dict:
    for seat in document["seats"]:
        if seat["seat_id"] == seat_id:
            return seat
    raise ValueError(f"no seat with seat_id '{seat_id}'")


def _resolve_zone_polygon_dict(
    document: dict, seat_id: str | None, zone: str | None, global_zone: str | None
) -> dict:
    """Navigate `document` to the specific zone's `{"points": [...]}` dict
    an edit subcommand targets, per its `--seat`/`--zone` vs `--global-zone`
    arguments (see `_add_zone_target_arguments`'s docstring for why exactly
    one of the two forms is required).
    """
    if global_zone is not None:
        return document["zones"][global_zone]
    seat = _find_seat(document, seat_id)  # type: ignore[arg-type]
    return seat["zones"][zone]  # type: ignore[index]


def _add_zone_target_arguments(parser: argparse.ArgumentParser) -> None:
    """Shared `--seat`/`--zone` vs `--global-zone` arguments for `set-zone`
    and `move-zone`: a zone edit targets exactly one polygon, either one
    seat's `player_area`/`chip_zone` (`--seat` + `--zone`) or one of the two
    table-wide zones (`--global-zone`) -- never both, never neither.
    """
    group = parser.add_argument_group("zone target (exactly one form)")
    group.add_argument("--seat", metavar="SEAT_ID", help="target this seat's zone")
    group.add_argument(
        "--zone", choices=sorted(_SEAT_ZONE_NAMES), help="which of the seat's zones"
    )
    group.add_argument(
        "--global-zone",
        choices=sorted(_GLOBAL_ZONE_NAMES),
        help="target this table-wide zone instead",
    )


def _validate_zone_target_arguments(
    args: argparse.Namespace,
) -> tuple[str | None, str | None, str | None]:
    seat_given = args.seat is not None
    zone_given = args.zone is not None
    global_given = args.global_zone is not None
    if global_given:
        if seat_given or zone_given:
            raise ValueError("--global-zone cannot be combined with --seat/--zone")
        return None, None, args.global_zone
    if not (seat_given and zone_given):
        raise ValueError("specify either --global-zone, or both --seat and --zone")
    return args.seat, args.zone, None


def _write_authoring_document(document: dict, out_path: Path) -> CalibrationAuthoring:
    """Re-validate the full document (REQ-11 included) before writing it anywhere."""
    authoring = CalibrationAuthoring.model_validate(document)
    write_calibration_authoring(authoring, out_path)
    return authoring


# --- compile (REQ-9) --------------------------------------------------------


def _cmd_compile(args: argparse.Namespace) -> int:
    try:
        authoring = load_calibration_authoring(args.authoring)
        runtime = compile_calibration(authoring, based_on=str(args.authoring))
        write_calibration_runtime(runtime, args.out)
    except (ValueError, ValidationError, OSError) as exc:
        print(f"calib compile: {exc}", file=sys.stderr)
        return EXIT_ERROR
    print(f"compiled '{args.authoring}' -> '{args.out}'")
    return EXIT_OK


# --- validate (REQ-10) ------------------------------------------------------


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        load_calibration_authoring(args.authoring)
    except ValueError as exc:
        print(f"calib validate: {exc}", file=sys.stderr)
        return EXIT_ERROR
    print(f"'{args.authoring}' is a valid calibration authoring")
    return EXIT_OK


# --- create (REQ-10) --------------------------------------------------------


def _cmd_create(args: argparse.Namespace) -> int:
    try:
        authoring = build_authoring_skeleton(
            table_id=args.table_id,
            seat_count=args.seats,
            table_width=args.width,
            table_height=args.height,
            table_unit=TableUnit(args.unit),
            inference_resolution=Resolution(
                width=args.inference_width, height=args.inference_height
            ),
        )
        write_calibration_authoring(authoring, args.out)
    except (ValueError, ValidationError, OSError) as exc:
        print(f"calib create: {exc}", file=sys.stderr)
        return EXIT_ERROR
    print(f"created skeleton calibration authoring with {args.seats} seats -> '{args.out}'")
    return EXIT_OK


# --- mark-zones (REQ-10a) ----------------------------------------------------


def _cmd_mark_zones(args: argparse.Namespace) -> int:
    return run_interactive_mark_zones(
        image_path=args.image,
        out_path=args.out,
        table_id=args.table_id,
        chip_zone_inset_pixels=args.chip_zone_inset_pixels,
    )


# --- learn-table (REQ-10b) ---------------------------------------------------


def _cmd_learn_table(args: argparse.Namespace) -> int:
    try:
        reference = load_calibration_runtime(args.reference_runtime)
        config = LearnTableConfig(
            min_match_count=args.min_match_count,
            min_inlier_ratio=args.min_inlier_ratio,
            ransac_reproj_threshold=args.ransac_reproj_threshold,
            center_strip_margin_ratio=args.center_strip_margin_ratio,
        )
        runtime = learn_table_calibration(
            reference,
            reference_image_path=args.reference_image,
            live_image_path=args.live_image,
            based_on=(
                f"calib learn-table: reference_runtime={args.reference_runtime}, "
                f"reference_image={args.reference_image}, live_image={args.live_image}"
            ),
            table_id=args.table_id,
            config=config,
        )
        write_calibration_runtime(runtime, args.out)
    except (ValueError, ValidationError, OSError) as exc:
        print(f"calib learn-table: {exc}", file=sys.stderr)
        return EXIT_ERROR
    print(f"learned table calibration from '{args.live_image}' -> '{args.out}'")
    return EXIT_OK


# --- edit (REQ-10) -----------------------------------------------------------


def _cmd_edit_add_seat(args: argparse.Namespace) -> int:
    try:
        authoring = load_calibration_authoring(args.authoring)
        document = authoring.model_dump(mode="json")
        if any(seat["seat_id"] == args.seat_id for seat in document["seats"]):
            raise ValueError(f"seat_id '{args.seat_id}' already exists")
        document["seats"].append(
            {
                "seat_id": args.seat_id,
                "zones": {
                    "player_area": {"points": [_parse_point(p) for p in args.player_area]},
                    "chip_zone": {"points": [_parse_point(p) for p in args.chip_zone]},
                },
            }
        )
        out_path = args.out if args.out is not None else args.authoring
        _write_authoring_document(document, out_path)
    except (ValueError, ValidationError, OSError) as exc:
        print(f"calib edit add-seat: {exc}", file=sys.stderr)
        return EXIT_ERROR
    print(f"added seat '{args.seat_id}' -> '{out_path}'")
    return EXIT_OK


def _cmd_edit_remove_seat(args: argparse.Namespace) -> int:
    try:
        authoring = load_calibration_authoring(args.authoring)
        document = authoring.model_dump(mode="json")
        remaining = [s for s in document["seats"] if s["seat_id"] != args.seat_id]
        if len(remaining) == len(document["seats"]):
            raise ValueError(f"no seat with seat_id '{args.seat_id}'")
        document["seats"] = remaining
        out_path = args.out if args.out is not None else args.authoring
        _write_authoring_document(document, out_path)
    except (ValueError, ValidationError, OSError) as exc:
        print(f"calib edit remove-seat: {exc}", file=sys.stderr)
        return EXIT_ERROR
    print(f"removed seat '{args.seat_id}' -> '{out_path}'")
    return EXIT_OK


def _cmd_edit_set_zone(args: argparse.Namespace) -> int:
    try:
        seat_id, zone, global_zone = _validate_zone_target_arguments(args)
        authoring = load_calibration_authoring(args.authoring)
        document = authoring.model_dump(mode="json")
        polygon = _resolve_zone_polygon_dict(document, seat_id, zone, global_zone)
        polygon["points"] = [_parse_point(p) for p in args.points]
        out_path = args.out if args.out is not None else args.authoring
        _write_authoring_document(document, out_path)
    except (ValueError, ValidationError, OSError) as exc:
        print(f"calib edit set-zone: {exc}", file=sys.stderr)
        return EXIT_ERROR
    print(f"updated zone -> '{out_path}'")
    return EXIT_OK


def _cmd_edit_move_zone(args: argparse.Namespace) -> int:
    try:
        seat_id, zone, global_zone = _validate_zone_target_arguments(args)
        authoring = load_calibration_authoring(args.authoring)
        document = authoring.model_dump(mode="json")
        polygon = _resolve_zone_polygon_dict(document, seat_id, zone, global_zone)
        polygon["points"] = [
            {"x": p["x"] + args.dx, "y": p["y"] + args.dy} for p in polygon["points"]
        ]
        out_path = args.out if args.out is not None else args.authoring
        _write_authoring_document(document, out_path)
    except (ValueError, ValidationError, OSError) as exc:
        print(f"calib edit move-zone: {exc}", file=sys.stderr)
        return EXIT_ERROR
    print(f"moved zone by ({args.dx}, {args.dy}) -> '{out_path}'")
    return EXIT_OK


# --- argument parsing --------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="calib")
    subparsers = parser.add_subparsers(dest="command", required=True)

    compile_parser = subparsers.add_parser("compile", help="Authoring JSON -> runtime JSON (REQ-9)")
    compile_parser.add_argument("--authoring", required=True, type=Path)
    compile_parser.add_argument("--out", required=True, type=Path)
    compile_parser.set_defaults(func=_cmd_compile)

    validate_parser = subparsers.add_parser(
        "validate", help="Validate an authoring JSON (schema + REQ-11 zone topology)"
    )
    validate_parser.add_argument("--authoring", required=True, type=Path)
    validate_parser.set_defaults(func=_cmd_validate)

    create_parser = subparsers.add_parser(
        "create", help="Create a new, REQ-11-valid authoring JSON skeleton"
    )
    create_parser.add_argument("--out", required=True, type=Path)
    create_parser.add_argument("--table-id", required=True)
    create_parser.add_argument(
        "--seats", type=int, default=6, help=f"default 6, minimum {MIN_SEAT_COUNT}"
    )
    create_parser.add_argument(
        "--width", type=float, default=1200.0, help="table width, default 1200"
    )
    create_parser.add_argument(
        "--height", type=float, default=900.0, help="table height, default 900"
    )
    create_parser.add_argument(
        "--unit", choices=[u.value for u in TableUnit], default=TableUnit.MM.value
    )
    create_parser.add_argument("--inference-width", type=int, default=1920)
    create_parser.add_argument("--inference-height", type=int, default=1080)
    create_parser.set_defaults(func=_cmd_create)

    mark_zones_parser = subparsers.add_parser(
        "mark-zones", help="Interactively click a reference photo's geometry (REQ-10a)"
    )
    mark_zones_parser.add_argument("--image", required=True, type=Path)
    mark_zones_parser.add_argument("--out", required=True, type=Path)
    mark_zones_parser.add_argument("--table-id", required=True)
    mark_zones_parser.add_argument(
        "--chip-zone-inset-pixels", type=float, default=DEFAULT_CHIP_ZONE_INSET_PIXELS
    )
    mark_zones_parser.set_defaults(func=_cmd_mark_zones)

    learn_table_parser = subparsers.add_parser(
        "learn-table",
        help="Derive a runtime calibration for a new photo of the same table design (REQ-10b)",
    )
    learn_table_parser.add_argument("--reference-runtime", required=True, type=Path)
    learn_table_parser.add_argument("--reference-image", required=True, type=Path)
    learn_table_parser.add_argument("--live-image", required=True, type=Path)
    learn_table_parser.add_argument("--out", required=True, type=Path)
    learn_table_parser.add_argument(
        "--table-id", help="default: reuse the reference calibration's table_id"
    )
    learn_table_parser.add_argument(
        "--min-match-count", type=int, default=DEFAULT_MIN_MATCH_COUNT
    )
    learn_table_parser.add_argument(
        "--min-inlier-ratio", type=float, default=DEFAULT_MIN_INLIER_RATIO
    )
    learn_table_parser.add_argument(
        "--ransac-reproj-threshold", type=float, default=DEFAULT_RANSAC_REPROJ_THRESHOLD_PIXELS
    )
    learn_table_parser.add_argument(
        "--center-strip-margin-ratio", type=float, default=DEFAULT_CENTER_STRIP_MARGIN_RATIO
    )
    learn_table_parser.set_defaults(func=_cmd_learn_table)

    edit_parser = subparsers.add_parser("edit", help="Edit an existing authoring JSON in place")
    edit_subparsers = edit_parser.add_subparsers(dest="edit_command", required=True)

    add_seat_parser = edit_subparsers.add_parser(
        "add-seat", help="Add a new seat with explicit zone polygons"
    )
    add_seat_parser.add_argument("--authoring", required=True, type=Path)
    add_seat_parser.add_argument("--out", type=Path, help="default: overwrite --authoring")
    add_seat_parser.add_argument("--seat-id", required=True)
    add_seat_parser.add_argument(
        "--player-area",
        nargs="+",
        metavar="X,Y",
        required=True,
        help="polygon points, e.g. 0,0 100,0 100,100",
    )
    add_seat_parser.add_argument("--chip-zone", nargs="+", metavar="X,Y", required=True)
    add_seat_parser.set_defaults(func=_cmd_edit_add_seat)

    remove_seat_parser = edit_subparsers.add_parser("remove-seat", help="Remove a seat by seat_id")
    remove_seat_parser.add_argument("--authoring", required=True, type=Path)
    remove_seat_parser.add_argument("--out", type=Path, help="default: overwrite --authoring")
    remove_seat_parser.add_argument("--seat-id", required=True)
    remove_seat_parser.set_defaults(func=_cmd_edit_remove_seat)

    set_zone_parser = edit_subparsers.add_parser("set-zone", help="Replace a zone's polygon points")
    set_zone_parser.add_argument("--authoring", required=True, type=Path)
    set_zone_parser.add_argument("--out", type=Path, help="default: overwrite --authoring")
    _add_zone_target_arguments(set_zone_parser)
    set_zone_parser.add_argument("--points", nargs="+", metavar="X,Y", required=True)
    set_zone_parser.set_defaults(func=_cmd_edit_set_zone)

    move_zone_parser = edit_subparsers.add_parser(
        "move-zone", help="Translate a zone's polygon by (dx, dy)"
    )
    move_zone_parser.add_argument("--authoring", required=True, type=Path)
    move_zone_parser.add_argument("--out", type=Path, help="default: overwrite --authoring")
    _add_zone_target_arguments(move_zone_parser)
    move_zone_parser.add_argument("--dx", type=float, required=True)
    move_zone_parser.add_argument("--dy", type=float, required=True)
    move_zone_parser.set_defaults(func=_cmd_edit_move_zone)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.func(args)


def entry_point() -> None:
    sys.exit(main())


if __name__ == "__main__":
    entry_point()
