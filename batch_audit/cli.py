"""Command-line entry point for the local deterministic batch audit workbench."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .audit import AuditMismatch, run_audit
from .constants import DEFAULT_OUTPUT_DIM, DEFAULT_SUPPORTED_SLOTS, NUMBERING_VERSION
from .demo import create_demo_workspace
from .engine import RunOptions, Scheduler, WorkspacePaths
from .feed import FeedError
from .ledger import Ledger, LedgerError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="batch-audit",
        description=(
            "Deterministic local batch ownership auditor: shuffled/replayed feeds, "
            "SQLite slot commits, reproducible ownership IDs, and independent replay."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="create a deterministic demo workspace")
    _add_workspace_option(init)
    init.add_argument("--seed", type=int, default=20261120)
    init.add_argument("--samples", type=int, default=12)
    init.add_argument("--input-dim", type=int, default=6)
    init.add_argument("--hidden-dim", type=int, default=8)
    init.add_argument("--output-dim", type=int, default=DEFAULT_OUTPUT_DIM)
    init.add_argument("--slots", default="0,1,2,3,4")

    run = sub.add_parser("run", help="execute or resume slot processing and write report.json")
    _add_workspace_option(run)
    _add_path_override_options(run)
    _add_run_options(run)

    audit = sub.add_parser("audit", help="independently recompute and verify the ledger")
    _add_workspace_option(audit)
    audit.add_argument("--ledger", type=Path, default=None)
    audit.add_argument("--feed", type=Path, default=None)
    audit.add_argument("--debug", action="store_true")

    status = sub.add_parser("status", help="show active generation and slot progress")
    _add_workspace_option(status)
    status.add_argument("--ledger", type=Path, default=None)

    gui = sub.add_parser("gui", help="open the local Tkinter workbench")
    _add_workspace_option(gui)
    return parser


def _add_workspace_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("workspace"),
        help="workspace directory (default: ./workspace)",
    )


def _add_path_override_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--feed", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--ledger", type=Path, default=None)
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=None)


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--slots",
        default=None,
        help="supported slots; defaults to workspace/manifest.json, then 0,1,2,3,4",
    )
    parser.add_argument("--numbering-version", default=NUMBERING_VERSION)
    parser.add_argument(
        "--slot-order",
        default=None,
        help="testing aid: process a permutation of slots; identities must not change",
    )
    parser.add_argument(
        "--crash-after-commits",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--debug", action="store_true")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            slots = _parse_slots(args.slots)
            created = create_demo_workspace(
                args.workspace,
                seed=args.seed,
                sample_count=args.samples,
                input_dim=args.input_dim,
                hidden_dim=args.hidden_dim,
                output_dim=args.output_dim,
                supported_slots=slots,
            )
            print(
                json.dumps(
                    {
                        "code": "OK",
                        "workspace": str(Path(args.workspace).resolve()),
                        "seed": args.seed,
                        "sample_count": args.samples,
                        "input_dim": args.input_dim,
                        "hidden_dim": args.hidden_dim,
                        "output_dim": args.output_dim,
                        "supported_slots": list(slots),
                        "shuffled_file_order": True,
                        "feed": str((Path(args.workspace) / "feed.json").resolve()),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
            )
            return 0

        if args.command == "run":
            paths = _paths_from_args(args)
            slots = _slots_for_run(args, paths.workspace)
            options = RunOptions(
                supported_slots=slots,
                numbering_version=args.numbering_version,
                slot_order=_parse_slots(args.slot_order) if args.slot_order else None,
                crash_after_commits=args.crash_after_commits,
            )
            report = Scheduler(paths, options).run()
            print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
            return 0

        if args.command == "audit":
            paths = WorkspacePaths.under(args.workspace)
            result = run_audit(
                ledger_path=args.ledger or paths.ledger,
                feed_path=args.feed or paths.feed,
            )
            print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
            return 0

        if args.command == "status":
            paths = WorkspacePaths.under(args.workspace)
            ledger_path = args.ledger or paths.ledger
            with Ledger(ledger_path) as ledger:
                generation = ledger.latest_generation()
                if generation is None:
                    print(json.dumps({"code": "NO_GENERATION"}, sort_keys=True, indent=2))
                    return 0
                slots = ledger.slots(generation["generation_id"])
                print(
                    json.dumps(
                        {
                            "code": "OK",
                            "generation": generation["generation_id"],
                            "version": generation["generation_id"],
                            "status": generation["status"],
                            "progress_cursor": generation["progress_cursor"],
                            "completed_slot_count": generation["completed_slot_count"],
                            "slots": [
                                {
                                    "batch_slot": row["batch_slot"],
                                    "status": row["status"],
                                    "error_code": row["error_code"],
                                }
                                for row in slots
                            ],
                            "ledger_content_digest": ledger.content_digest,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        indent=2,
                    )
                )
            return 0

        if args.command == "gui":
            from .gui import main as gui_main

            return gui_main(args.workspace)
    except FeedError as exc:
        print(json.dumps(exc.as_dict(), ensure_ascii=False, sort_keys=True, indent=2))
        return 2
    except AuditMismatch as exc:
        print(json.dumps(exc.as_dict(), ensure_ascii=False, sort_keys=True, indent=2))
        return 3
    except (LedgerError, sqlite3.DatabaseError, OSError, ValueError) as exc:
        payload: dict[str, Any] = {
            "code": getattr(exc, "code", "E_STATE"),
            "message": str(exc),
        }
        if isinstance(exc, LedgerError):
            payload.update(exc.details)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
        if getattr(args, "debug", False):
            raise
        return 4
    return 0


def _parse_slots(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if not parts:
            raise ValueError("slots must not be empty")
        try:
            return tuple(int(part) for part in parts)
        except ValueError as exc:
            raise ValueError(f"invalid slot list: {value}") from exc
    return tuple(int(item) for item in value)


def _paths_from_args(args: argparse.Namespace) -> WorkspacePaths:
    base = WorkspacePaths.under(args.workspace)
    return WorkspacePaths(
        workspace=base.workspace,
        feed=args.feed or base.feed,
        dataset=args.dataset or base.dataset,
        model=args.model or base.model,
        ledger=args.ledger or base.ledger,
        snapshot=args.snapshot or base.snapshot,
        report=args.report or base.report,
        structured_log=args.log or base.structured_log,
    )


def _slots_for_run(args: argparse.Namespace, workspace: Path) -> tuple[int, ...]:
    if args.slots is not None:
        return _parse_slots(args.slots)
    manifest_path = Path(workspace) / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            slots = manifest.get("supported_slots")
            if isinstance(slots, list):
                return _parse_slots(slots)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"cannot read supported_slots from {manifest_path}: {exc}") from exc
    return tuple(DEFAULT_SUPPORTED_SLOTS)


if __name__ == "__main__":
    raise SystemExit(main())
