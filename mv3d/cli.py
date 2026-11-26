"""Command line interface for sample generation, runs, and acceptance checks."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from .commands import apply_commands
from .constants import EXIT_INTERNAL_ERROR, EXIT_OK, EXIT_WORKSPACE_LOCKED, ENGINE_VERSION
from .errors import WorkspaceLockedError
from .io import read_jsonl
from .pipeline import ReconstructionRunner
from .snapshot import SnapshotStore
from .synth import generate_demo, load_project
from .verification import run_acceptance


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _run_command(args: argparse.Namespace) -> int:
    project = load_project(Path(args.project))
    commands = read_jsonl(Path(args.commands)) if args.commands else []
    if args.reset_snapshots:
        snapshots = Path(args.workspace_root).resolve() / ".mv3d" / "snapshots"
        if snapshots.exists():
            shutil.rmtree(snapshots)
    runner = ReconstructionRunner(Path(args.workspace_root))
    report_path = Path(args.report) if args.report else None
    if args.after_each and commands:
        state = project
        collected_failures = []
        result = None
        for command in commands:
            state, failures = apply_commands(state, [command])
            collected_failures.extend(failures)
            result = runner.run(
                state,
                external_failures=list(collected_failures),
                workers_override=args.workers,
            )
            collected_failures = []
        assert result is not None
        if report_path is not None:
            result.persist_report(report_path)
    else:
        state, failures = apply_commands(project, commands)
        result = runner.run(
            state,
            external_failures=failures,
            workers_override=args.workers,
            sigkill_after_stage=args.sigkill_after_stage,
            write_report=report_path,
        )
    _print(
        {
            "status": result.status,
            "exit_code": result.exit_code,
            "canonical_digest": result.canonical_digest,
            "failure_counts": result.report["failure_counts"],
            "pending_queue": result.pending_queue,
            "stage_summaries": result.report["stage_summaries"],
            "ledger_summary": result.report["ledger"]["summary"],
            "snapshot": result.report["snapshot"],
            "report": str(report_path) if report_path else None,
        }
    )
    return result.exit_code


def _init_demo(args: argparse.Namespace) -> int:
    root = Path(args.root)
    try:
        generate_demo(root, seed=args.seed, force=args.force)
    except FileExistsError as error:
        print(str(error), file=sys.stderr)
        return EXIT_INTERNAL_ERROR
    _print(
        {
            "status": "generated",
            "root": str(root.resolve()),
            "project": str((root / "project.json").resolve()),
            "commands": str((root / "commands.jsonl").resolve()),
            "seed": args.seed,
        }
    )
    return EXIT_OK


def _snapshot_info(args: argparse.Namespace) -> int:
    store = SnapshotStore(Path(args.workspace_root))
    snapshot = store.load_latest(ENGINE_VERSION)
    if snapshot is None:
        _print({"status": "no_snapshot", "engine_version": ENGINE_VERSION})
        return EXIT_OK
    _print(
        {
            "status": "ok",
            "generation": snapshot.generation,
            "path": str(snapshot.path),
            "engine_version": snapshot.engine_version,
            "config_hash": snapshot.config_hash,
            "completed_stages": snapshot.completed_stages,
            "run_complete": snapshot.run_complete,
            "stages_present": sorted(snapshot.stages),
        }
    )
    return EXIT_OK


def _verify(args: argparse.Namespace) -> int:
    project = load_project(Path(args.project))
    commands = read_jsonl(Path(args.commands)) if args.commands else []
    report = run_acceptance(
        project=project,
        commands=commands,
        seed=args.seed,
        workers=(1, 4),
        run_crash_check=not args.skip_crash,
        repo_root=Path(__file__).resolve().parent.parent,
    )
    if args.report:
        Path(args.report).write_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    _print(report)
    return EXIT_OK if report["passed"] else EXIT_INTERNAL_ERROR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mv3d",
        description="Deterministic fixed-point multi-view 3D reconstruction diagnostics",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init-demo", help="generate the local deterministic sample")
    init_parser.add_argument("--root", default="demo")
    init_parser.add_argument("--seed", type=int, default=7)
    init_parser.add_argument("--force", action="store_true")
    init_parser.set_defaults(handler=_init_demo)

    run_parser = subparsers.add_parser("run", help="run or incrementally continue reconstruction")
    run_parser.add_argument("--project", required=True)
    run_parser.add_argument("--commands")
    run_parser.add_argument("--workspace-root", default=".")
    run_parser.add_argument("--report")
    run_parser.add_argument("--workers", type=int, default=None)
    run_parser.add_argument("--after-each", action="store_true")
    run_parser.add_argument("--sigkill-after-stage", choices=("load", "features", "matches", "triangulation", "depth", "voxels"))
    run_parser.add_argument("--reset-snapshots", action="store_true")
    run_parser.set_defaults(handler=_run_command)

    snapshot_parser = subparsers.add_parser("snapshot-info", help="inspect the latest complete snapshot")
    snapshot_parser.add_argument("--workspace-root", default=".")
    snapshot_parser.set_defaults(handler=_snapshot_info)

    verify_parser = subparsers.add_parser("verify", help="run incremental/full, concurrency, budget, and crash checks")
    verify_parser.add_argument("--project", required=True)
    verify_parser.add_argument("--commands")
    verify_parser.add_argument("--seed", type=int, default=7)
    verify_parser.add_argument("--skip-crash", action="store_true")
    verify_parser.add_argument("--report")
    verify_parser.set_defaults(handler=_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except WorkspaceLockedError as error:
        print(f"workspace is locked by another process: {error}", file=sys.stderr)
        return EXIT_WORKSPACE_LOCKED
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return EXIT_INTERNAL_ERROR


if __name__ == "__main__":
    raise SystemExit(main())

