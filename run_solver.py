"""Command-line entry point for the local level constraint solver.

Subcommands:
    solve   Run (or resume) a solve against a JSON specification.
    verify  Validate digests of every snapshot in a snapshot directory.

All failures are emitted as JSON documents carrying an ``E_xxx`` business code.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

from level_solver.errors import SolverError
from level_solver.parser import parse_spec
from level_solver.snapshot import INTERRUPT_EXIT_CODE, SnapshotStore, build_version_id
from level_solver.solver import LevelSolver


def _load_spec(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise SolverError("E_IO", f"specification file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SolverError("E_SCHEMA", f"invalid JSON: {exc}") from exc


def _emit(error: SolverError) -> int:
    sys.stderr.write(json.dumps(error.to_dict(), ensure_ascii=False, indent=2) + "\n")
    return {
        "E_CONFLICT": 2,
        "E_DIGEST": 3,
        "E_VERSION": 4,
        "E_INTERRUPTED": INTERRUPT_EXIT_CODE,
    }.get(error.code, 1)


def cmd_solve(args: argparse.Namespace) -> int:
    # Parse and validate fully before touching the snapshot directory.
    spec = parse_spec(_load_spec(args.spec))

    solver = LevelSolver(spec, args.snapshot_dir)
    solver.prepare(allow_resume=not args.no_resume)

    def handle_signal(signum, frame):  # pragma: no cover - exercised via runner
        solver.request_interrupt(f"signal-{signum}")

    previous = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, handle_signal)

    try:
        result = solver.solve(
            interrupt_after=args.interrupt_after,
        )
    except SolverError as exc:
        return _emit(exc)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)

    if result.interrupted:
        payload = {
            "ok": True,
            "interrupted": True,
            "exit_code": INTERRUPT_EXIT_CODE,
            "sequence": result.sequence,
            "next_index": result.sequence,
            "version_id": solver.version_id,
            "digest_warnings": result.digest_warnings,
            "resumed": result.resumed,
        }
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        return INTERRUPT_EXIT_CODE

    payload = {
        "ok": True,
        "interrupted": False,
        "version_id": solver.version_id,
        "sequence": result.sequence,
        "grid": result.grid.to_list(),
        "rooms": result.rooms,
        "constraint_counts": result.constraint_counts,
        "constraints_applied": [frame.__dict__ for frame in result.applied],
        "conflicts": result.conflicts,
        "backtracks": solver.backtracks,
        "resumed": result.resumed,
        "digest_warnings": result.digest_warnings,
    }
    if args.output:
        Path(args.output).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        payload["output_file"] = args.output
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    spec = parse_spec(_load_spec(args.spec))
    version_id = build_version_id(spec)
    store = SnapshotStore(args.snapshot_dir, version_id)
    reports = store.verify_all()
    payload = {"ok": all(report["ok"] for report in reports), "snapshots": reports}
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    if not payload["ok"]:
        return 3
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local deterministic level constraint solver")
    sub = parser.add_subparsers(dest="command", required=True)

    solve = sub.add_parser("solve", help="run or resume a solve")
    solve.add_argument("--spec", required=True, help="path to the level specification JSON")
    solve.add_argument("--snapshot-dir", required=True, help="directory for durable snapshots")
    solve.add_argument("--output", help="optional path for the final result JSON")
    solve.add_argument(
        "--interrupt-after",
        type=int,
        default=None,
        help="cooperatively suspend after N committed constraints (deterministic test hook)",
    )
    solve.add_argument("--no-resume", action="store_true", help="ignore any existing snapshots")
    solve.set_defaults(func=cmd_solve)

    verify = sub.add_parser("verify", help="verify snapshot digests")
    verify.add_argument("--spec", required=True)
    verify.add_argument("--snapshot-dir", required=True)
    verify.set_defaults(func=cmd_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except SolverError as exc:
        return _emit(exc)


if __name__ == "__main__":
    raise SystemExit(main())
