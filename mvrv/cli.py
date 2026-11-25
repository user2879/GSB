from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .acceptance import run_acceptance
from .editor import apply_command
from .engine import build_reconstruction, canonical_state_blob
from .snapshot import SnapshotStore
from .synthetic import generate_project
from .validation import load_project


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str))


def cmd_generate(args: argparse.Namespace) -> int:
    result = generate_project(args.root)
    print(f"generated deterministic sample at {Path(args.root).resolve()}")
    print(f"views={len(result['cameras']['views'])} landmarks={len(result['scene']['landmarks'])}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    loaded = load_project(args.root)
    _print_json(loaded.report())
    if loaded.failures and args.verbose:
        _print_json([item.to_dict() for item in loaded.failures])
    return 0 if loaded.ok else 2


def cmd_rebuild(args: argparse.Namespace) -> int:
    loaded = load_project(args.root)
    if not loaded.ok:
        _print_json(loaded.report())
        if args.verbose:
            _print_json([item.to_dict() for item in loaded.failures])
        return 2

    store = SnapshotStore(args.root)
    previous = None if args.full else store.load_latest_state()
    result = build_reconstruction(
        loaded,
        previous_state=previous,
        workers=args.workers,
        global_budget=args.global_budget,
        view_quota=args.view_quota,
    )
    state = result["state"]
    print(f"state_hash={canonical_state_blob(state)}")
    _print_json({"metrics": state["metrics"], "ledger_counts": result["ledger"]["counts"]})
    if not args.no_snapshot:
        committed = store.commit(
            validation_report=loaded.report(), state=state, ledger=result["ledger"]
        )
        print(f"snapshot=v{committed['manifest']['global_version']:06d} {committed['snapshot']}")
    return 0


def cmd_acceptance(args: argparse.Namespace) -> int:
    report = run_acceptance(args.root)
    summary = Path(args.root) / "results" / "acceptance_summary.txt"
    print(summary.read_text(encoding="utf-8"))
    print(f"report={Path(args.root) / 'results' / 'acceptance_report.json'}")
    return 0 if report["ok"] else 1


def cmd_ui(args: argparse.Namespace) -> int:
    try:
        from .ui import launch
    except ImportError as exc:
        print(f"Tkinter is unavailable: {exc}", file=sys.stderr)
        print("On Ubuntu/WSL install it with: sudo apt-get update && sudo apt-get install -y python3-tk", file=sys.stderr)
        return 3
    launch(Path(args.root))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mvrv", description="Deterministic multi-view voxel occupancy diagnostics")
    parser.add_argument("--root", default=".", help="project root containing config/ and data/")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default=".", help="project root containing config/ and data/")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("generate", parents=[common], help="generate deterministic local sample").set_defaults(func=cmd_generate)

    validate = sub.add_parser("validate", parents=[common], help="validate images, cameras, config and commands")
    validate.add_argument("--verbose", action="store_true")
    validate.set_defaults(func=cmd_validate)

    rebuild = sub.add_parser("rebuild", parents=[common], help="run full or incremental reconstruction")
    rebuild.add_argument("--full", action="store_true", help="ignore previous snapshot/cache")
    rebuild.add_argument("--workers", type=int, default=None)
    rebuild.add_argument("--global-budget", type=int, default=None)
    rebuild.add_argument("--view-quota", type=int, default=None)
    rebuild.add_argument("--no-snapshot", action="store_true")
    rebuild.add_argument("--verbose", action="store_true")
    rebuild.set_defaults(func=cmd_rebuild)

    acceptance = sub.add_parser("acceptance", parents=[common], help="run repeatable acceptance checks and write results/")
    acceptance.set_defaults(func=cmd_acceptance)

    ui = sub.add_parser("ui", parents=[common], help="launch Tkinter diagnostic interface")
    ui.set_defaults(func=cmd_ui)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
