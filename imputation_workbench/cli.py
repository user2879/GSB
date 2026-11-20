"""Command-line entry point for the imputation workbench."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import tempfile
from typing import Any

from .engine import (
    WorkbenchEngine,
    WorkspacePaths,
    compare_semantic_states,
)
from .errors import WorkbenchError
from .samplegen import generate_sample
from .snapshot import SnapshotStore, clear_state
from .util import write_json_atomic


class StopController:
    def __init__(self) -> None:
        self.requested = False

    def request(self, signum: int, frame: Any) -> None:  # noqa: ARG002
        self.requested = True

    def __call__(self) -> bool:
        return self.requested

    def install(self) -> None:
        for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            candidate = getattr(signal, name, None)
            if candidate is not None:
                try:
                    signal.signal(candidate, self.request)
                except (OSError, ValueError):
                    pass


def _paths(args: argparse.Namespace, state: Path | None = None) -> WorkspacePaths:
    injections = None
    if getattr(args, "no_injections", False):
        injections = Path(args.workspace) / "__no_injections__.json"
    elif getattr(args, "injections", None):
        injections = Path(args.injections)
    return WorkspacePaths.from_workspace(
        Path(args.workspace),
        state=state or Path(args.state),
        injections=injections,
    )


def _print_summary(result: dict[str, Any], report_path: Path) -> None:
    report = result["report"]
    summary = {
        "status": result["status"],
        "business_codes": report.get("business_codes", []),
        "completed_batches": report.get("completed_batches", []),
        "in_progress_batch": report.get("in_progress_batch"),
        "pending_count": report.get("imputation", {}).get("pending_count", 0),
        "snapshot_version": report.get("snapshot_version"),
        "model_accuracy": (report.get("model") or {}).get("accuracy"),
        "model_log_loss": (report.get("model") or {}).get("log_loss"),
        "report_path": str(report_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def cmd_generate(args: argparse.Namespace) -> int:
    generate_sample(Path(args.out), seed=args.seed, rows=args.rows)
    print(
        json.dumps(
            {
                "status": "generated",
                "workspace": str(Path(args.out).resolve()),
                "seed": args.seed,
                "rows": args.rows,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    controller = StopController()
    controller.install()
    paths = _paths(args)
    engine = WorkbenchEngine(paths)
    try:
        result = engine.run(
            fresh=args.fresh,
            max_slices=args.max_slices,
            max_batches=args.max_batches,
            workers_override=args.workers,
            budget_override=args.cell_budget,
            stop_requested=controller,
            hard_kill_after_commits=args.simulate_hard_kill_after_commit,
        )
    except WorkbenchError as exc:
        error_report = {
            "format": 1,
            "status": "error",
            "business_codes": [exc.code],
            **exc.as_dict(),
        }
        try:
            write_json_atomic(Path(args.state) / "report.json", error_report)
            write_json_atomic(Path(args.state) / "error_report.json", error_report)
        except OSError:
            pass
        print(json.dumps(error_report, ensure_ascii=False, indent=2, sort_keys=True))
        return 2
    _print_summary(result, Path(args.state) / "report.json")
    return 0


def cmd_rebuild(args: argparse.Namespace) -> int:
    output = Path(args.output)
    clear_state(output)
    paths = WorkspacePaths(
        workspace=Path(args.workspace),
        state=output,
        config=Path(args.workspace) / "imputation.yaml",
        data=Path(args.workspace) / "raw_data.jsonl",
        group_map=Path(args.workspace) / "group_map.json",
        plan=Path(args.workspace) / "edit_plan.json",
        injections=(
            Path(args.workspace) / "__no_injections__.json"
            if args.no_injections
            else (Path(args.injections) if args.injections else Path(args.workspace) / "injections.json")
        ),
    )
    engine = WorkbenchEngine(paths)
    try:
        result = engine.run(fresh=True, workers_override=args.workers)
    except WorkbenchError as exc:
        print(json.dumps({"status": "error", **exc.as_dict()}, ensure_ascii=False, indent=2, sort_keys=True))
        return 2
    _print_summary(result, output / "report.json")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    state_root = Path(args.state)
    store = SnapshotStore(state_root)
    loaded = store.load_state()
    if loaded is None:
        print(json.dumps({"status": "error", "message": "no incremental snapshot to verify"}))
        return 2
    incremental_state, _manifest = loaded
    paths = _paths(args)
    with tempfile.TemporaryDirectory(prefix="iw-rebuild-") as tmp:
        rebuild_root = Path(tmp) / "work"
        rebuild_paths = WorkspacePaths(
            workspace=paths.workspace,
            state=rebuild_root,
            config=paths.config,
            data=paths.data,
            group_map=paths.group_map,
            plan=paths.plan,
            injections=paths.injections,
        )
        result = WorkbenchEngine(rebuild_paths).run(
            fresh=True,
            workers_override=args.workers,
        )
        rebuilt_loaded = SnapshotStore(rebuild_root).load_state()
        if rebuilt_loaded is None:
            print(json.dumps({"status": "error", "message": "rebuild produced no snapshot"}))
            return 2
        rebuilt_state, rebuilt_manifest = rebuilt_loaded
        comparison = compare_semantic_states(incremental_state, rebuilt_state)
        verification = {
            "format": 1,
            "status": "verified" if comparison["equal"] else "mismatch",
            "comparison": comparison,
            "incremental_snapshot_version": _manifest["global_version"],
            "rebuild_snapshot_version": rebuilt_manifest["global_version"],
            "incremental_report": str(state_root / "report.json"),
            "rebuild_report": str(rebuild_root / "report.json"),
            "rebuild_status": result["status"],
        }
        write_json_atomic(state_root / "verification.json", verification)
        print(json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if comparison["equal"] else 3


def cmd_compare(args: argparse.Namespace) -> int:
    left = SnapshotStore(Path(args.left)).load_state()
    right = SnapshotStore(Path(args.right)).load_state()
    if left is None or right is None:
        print(json.dumps({"status": "error", "message": "one or both snapshots are missing"}))
        return 2
    comparison = compare_semantic_states(left[0], right[0])
    print(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if comparison["equal"] else 3


def cmd_report(args: argparse.Namespace) -> int:
    report = SnapshotStore(Path(args.state)).read_report()
    if report is None and not Path(args.state).exists():
        print(json.dumps({"status": "error", "message": "state directory does not exist"}))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report is not None else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="imputation-workbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate a deterministic sample workspace")
    generate.add_argument("--out", default="samples/demo")
    generate.add_argument("--seed", type=int, default=20261121)
    generate.add_argument("--rows", type=int, default=24)
    generate.set_defaults(func=cmd_generate)

    run = subparsers.add_parser("run", help="execute or resume incremental batches")
    run.add_argument("--workspace", default="samples/demo")
    run.add_argument("--state", default="samples/demo/work")
    run.add_argument("--fresh", action="store_true")
    run.add_argument("--max-slices", type=int, default=None)
    run.add_argument("--max-batches", type=int, default=None)
    run.add_argument("--workers", type=int, default=None)
    run.add_argument("--cell-budget", type=int, default=None)
    run.add_argument("--injections", default=None)
    run.add_argument("--no-injections", action="store_true")
    run.add_argument("--simulate-hard-kill-after-commit", type=int, default=None, help=argparse.SUPPRESS)
    run.set_defaults(func=cmd_run)

    rebuild = subparsers.add_parser("rebuild", help="build a fresh full snapshot")
    rebuild.add_argument("--workspace", default="samples/demo")
    rebuild.add_argument("--output", default="samples/demo/rebuild")
    rebuild.add_argument("--workers", type=int, default=None)
    rebuild.add_argument("--injections", default=None)
    rebuild.add_argument("--no-injections", action="store_true")
    rebuild.set_defaults(func=cmd_rebuild)

    verify = subparsers.add_parser("verify", help="compare current incremental state with a fresh rebuild")
    verify.add_argument("--workspace", default="samples/demo")
    verify.add_argument("--state", default="samples/demo/work")
    verify.add_argument("--workers", type=int, default=None)
    verify.add_argument("--injections", default=None)
    verify.add_argument("--no-injections", action="store_true")
    verify.set_defaults(func=cmd_verify)

    compare = subparsers.add_parser("compare", help="compare two snapshot states field by field")
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)
    compare.set_defaults(func=cmd_compare)

    report = subparsers.add_parser("report", help="print the latest JSON report")
    report.add_argument("--state", default="samples/demo/work")
    report.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except WorkbenchError as exc:
        error_report = {
            "format": 1,
            "status": "error",
            "business_codes": [exc.code],
            **exc.as_dict(),
        }
        state = getattr(args, "state", None)
        if state:
            try:
                write_json_atomic(Path(state) / "report.json", error_report)
            except OSError:
                pass
        print(json.dumps(error_report, ensure_ascii=False, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
