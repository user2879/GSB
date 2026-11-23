from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback
from typing import Any

from .errors import LabError
from .ledger import Ledger
from .loader import load_workload
from .session import ExperimentSession


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="joinlab",
        description="Deterministic join-order, cost-model and constrained scheduler laboratory",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="run the complete end-to-end demonstration")
    _add_common_paths(demo)
    demo.add_argument("--reset", action="store_true", help="delete the existing ledger before running")
    demo.add_argument("--interrupt-after-execution", type=int, default=None)
    demo.add_argument("--interrupt-before-commit", type=int, default=None)
    demo.add_argument(
        "--fail-baseline",
        action="store_true",
        help="inject a deterministic baseline row drop to demonstrate E_DIVERGE",
    )
    demo.add_argument(
        "--tamper-ledger",
        choices=("estimate_cost", "batch_digest", "replay_operator"),
        default=None,
    )
    demo.add_argument("--no-reexecute-replay", action="store_true")
    demo.add_argument(
        "--changes",
        default=None,
        help="JSON file containing a change set; applies it and runs the replacement generation",
    )

    replay = subparsers.add_parser("replay", help="restore a ledger snapshot and verify replay")
    _add_common_paths(replay)
    replay.add_argument("--snapshot-version", type=int, default=None)
    replay.add_argument("--no-reexecute", action="store_true")

    verify = subparsers.add_parser("verify", help="verify ledger content digests and normalized rows")
    _add_common_paths(verify)

    gui = subparsers.add_parser("gui", help="open the optional Tkinter desktop panel")
    gui.add_argument("--workload", default="samples/demo_workload.json")
    gui.add_argument("--db", default="artifacts/joinlab_gui.sqlite")
    return parser


def _add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workload", default="samples/demo_workload.json")
    parser.add_argument("--db", default="artifacts/joinlab.sqlite")
    parser.add_argument("--out", default=None)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "gui":
        from .gui import launch

        launch(args.workload, args.db)
        return 0
    try:
        if args.command == "demo":
            return _demo(args)
        if args.command == "replay":
            return _replay(args)
        if args.command == "verify":
            return _verify(args)
        raise LabError("E_CLI", f"未知命令 {args.command}", field="command")
    except LabError as exc:
        _emit_error(exc, getattr(args, "out", None))
        return 75 if exc.code == "E_INTERRUPTED" else 2
    except BaseException as exc:
        error = LabError(
            "E_INTERNAL",
            f"未处理异常: {type(exc).__name__}: {exc}",
            context={"traceback": traceback.format_exc(limit=8)},
        )
        _emit_error(error, getattr(args, "out", None))
        return 3


def _demo(args: argparse.Namespace) -> int:
    db_path = Path(args.db)
    if args.reset:
        for suffix in ("", "-shm", "-wal"):
            candidate = Path(str(db_path) + suffix)
            if candidate.exists():
                candidate.unlink()
    workload = load_workload(args.workload)
    session = ExperimentSession(workload)
    interrupted = False
    with Ledger(db_path) as ledger:
        ledger.verify_integrity(session.run_id)
        latest = ledger.latest_snapshot(session.run_id)
        if latest is not None:
            session, restored_snapshot = session.restore_from_ledger(ledger)
        else:
            restored_snapshot = None
        initial_analysis = session.analysis_report()
        try:
            execution_report = session.run(
                ledger,
                interrupt_after_execution=args.interrupt_after_execution,
                interrupt_before_commit=args.interrupt_before_commit,
                execution_faults={"baseline": "drop_last_row"} if args.fail_baseline else {},
            )
        except LabError as exc:
            if exc.code != "E_INTERRUPTED":
                raise
            raise
        if args.tamper_ledger:
            ledger.tamper_for_test(args.tamper_ledger)
            ledger.verify_integrity(session.run_id)
        restored, snapshot = session.restore_from_ledger(ledger)
        replay_report = restored.replay_check(
            ledger,
            snapshot_version=snapshot["version"],
            reexecute=not args.no_reexecute_replay,
        )
        change_report = None
        replacement_execution = None
        current_analysis = initial_analysis
        if args.changes:
            changes = json.loads(Path(args.changes).read_text(encoding="utf-8"))
            change_report = session.apply_change_set(changes, ledger=ledger)
            replacement_execution = session.run(ledger)
            current_analysis = session.analysis_report()
        report = {
            "ok": True,
            "workload": str(Path(args.workload).resolve()),
            "ledger": str(Path(args.db).resolve()),
            "analysis": initial_analysis,
            "current_analysis": current_analysis,
            "execution": execution_report,
            "snapshot_restore": {
                "version": snapshot["version"],
                "content_digest": snapshot["content_digest"],
                "format_version": snapshot["format_version"],
                "progress": restored.progress,
                "candidate_ids": restored.current_view["candidate_ids"],
            },
            "replay": replay_report,
            "invalidation": change_report,
            "replacement_execution": replacement_execution,
            "report_out": str(Path(args.out).resolve()) if args.out else None,
            "resources": {
                "process": "local Python threads only; no container",
                "numpy_version": _numpy_version(),
                "sqlite_version": _sqlite_version(),
            },
        }
        _write_json(args.out, report)
        _print_summary(report)
    return 0


def _replay(args: argparse.Namespace) -> int:
    workload = load_workload(args.workload)
    session = ExperimentSession(workload)
    with Ledger(args.db) as ledger:
        restored, snapshot = session.restore_from_ledger(
            ledger, snapshot_version=args.snapshot_version
        )
        report = restored.replay_check(
            ledger,
            snapshot_version=snapshot["version"],
            reexecute=not args.no_reexecute,
        )
    _write_json(args.out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _verify(args: argparse.Namespace) -> int:
    workload = load_workload(args.workload)
    session = ExperimentSession(workload)
    with Ledger(args.db) as ledger:
        report = ledger.verify_integrity(session.run_id)
    _write_json(args.out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _emit_error(error: LabError, out: str | None) -> None:
    payload = {"ok": False, "error": error.to_dict()}
    _write_json(out, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)


def _write_json(path: str | None, value: Any) -> None:
    if not path:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _print_summary(report: dict[str, Any]) -> None:
    analysis = report["analysis"]
    execution = report["execution"]
    replay = report["replay"]
    print(
        json.dumps(
            {
                "ok": report["ok"],
                "run_id": analysis["run_id"],
                "graph_digest": analysis["graph"]["digest"],
                "candidate_count": analysis["candidate_enumeration"]["candidate_count"],
                "total_before_limit": analysis["candidate_enumeration"]["total_before_limit"],
                "best_candidate_id": analysis["selection"]["best_candidate_id"],
                "best_plan": next(
                    (
                        candidate["plan_signature"]
                        for candidate in analysis.get("candidates", [])
                        if candidate["candidate_id"] == analysis["selection"]["best_candidate_id"]
                    ),
                    None,
                ),
                "batches_committed": len(execution["commits"]),
                "consistency": execution["consistency"],
                "bias": execution["bias"],
                "snapshot_version": report["snapshot_restore"]["version"],
                "replay_ok": replay["ok"],
                "report_out": report.get("report_out"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _numpy_version() -> str | None:
    try:
        import numpy

        return numpy.__version__
    except ImportError:
        return None


def _sqlite_version() -> str:
    import sqlite3

    return sqlite3.sqlite_version


if __name__ == "__main__":
    raise SystemExit(main())
