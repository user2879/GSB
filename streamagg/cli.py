from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .demo import run_demo
from .diagnostics import run_diagnostics
from .engine import Engine
from .errors import BusinessError, exit_code_for
from .generator import generate_session
from .ledger import Ledger
from .logging_utils import JsonlLogger
from .replay import validate_replay
from .scenarios import (
    DemoPaths,
    inject_grid,
    inject_order,
    inject_replay,
    inject_snapshot,
    prepare_scenario3_crash,
    run_scenario,
)
from .snapshot import SnapshotStore, restore_engine
from .spec import load_spec


def _paths(args: argparse.Namespace) -> DemoPaths:
    return DemoPaths.from_values(
        args.workspace,
        args.demo_root,
        args.ledger,
        args.snapshots,
        args.logs,
        args.reports,
    )


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--demo-root", default="artifacts/demo")
    parser.add_argument("--ledger", default=None)
    parser.add_argument("--snapshots", default=None)
    parser.add_argument("--logs", default=None)
    parser.add_argument("--reports", default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="streamagg")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="validate a local JSON specification")
    validate_parser.add_argument("--spec", required=True)
    validate_parser.add_argument("--log-dir", default="artifacts/logs")

    run_parser = subparsers.add_parser("run", help="run one session from a JSON specification")
    run_parser.add_argument("--spec", required=True)
    run_parser.add_argument("--session-id", default=None)
    _common(run_parser)
    run_parser.add_argument("--save-snapshot", action="store_true")

    scenario_parser = subparsers.add_parser("scenario", help="run one of the three demonstration scenarios")
    scenario_parser.add_argument("--scenario", type=int, choices=(1, 2, 3), required=True)
    _common(scenario_parser)

    crash_parser = subparsers.add_parser("crash-writer", help=argparse.SUPPRESS)
    crash_parser.add_argument("--metadata", required=True)
    _common(crash_parser)

    demo_parser = subparsers.add_parser("demo", help="run all three scenarios and anomaly injections")
    demo_parser.add_argument("--workspace", default=".")
    demo_parser.add_argument("--no-clean", action="store_true")

    inject_parser = subparsers.add_parser("inject", help="inject a fault and emit a structured error code")
    inject_parser.add_argument(
        "--kind", choices=("order", "grid", "snapshot", "replay"), required=True
    )
    _common(inject_parser)

    ui_parser = subparsers.add_parser("ui", help="open the read-only Tkinter diagnostics window")
    ui_parser.add_argument("--ledger", required=True)
    ui_parser.add_argument("--session-id", default=None)
    ui_parser.add_argument("--view-id", default=None)

    restore_parser = subparsers.add_parser("restore", help="restore a snapshot and validate replay")
    restore_parser.add_argument("--snapshot", required=True)
    _common(restore_parser)

    replay_parser = subparsers.add_parser("replay", help="replay the current ledger cursor and compare")
    replay_parser.add_argument("--spec", required=True)
    replay_parser.add_argument("--session-id", required=True)
    replay_parser.add_argument("--view-id", default=None)
    _common(replay_parser)

    return parser


def dispatch(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "validate":
        spec = load_spec(args.spec)
        events_a, generations_a = generate_session(spec)
        events_b, generations_b = generate_session(spec)
        if [event.to_dict() for event in events_a] != [event.to_dict() for event in events_b]:
            raise BusinessError(
                code="E_SPEC", message="source generation is not deterministic", field="sources"
            )
        return {
            "business_code": "OK_VALIDATE",
            "spec_digest": spec.spec_digest,
            "source_digests": {
                key: value.digest for key, value in sorted(generations_a.items())
            },
            "event_count": len(events_a),
            "partition_keys": list(spec.partition_keys),
            "rules_version": spec.rules_version,
            "fixed_scale": spec.fixed_scale,
        }
    if args.command == "run":
        paths = _paths(args)
        paths.prepare()
        logger = JsonlLogger(paths.logs, f"run-{args.session_id or 'session'}")
        spec = load_spec(args.spec)
        with Ledger(paths.ledger) as ledger:
            engine = Engine(
                spec,
                session_id=args.session_id,
                ledger=ledger,
                logger=logger,
            )
            summary = engine.run_all()
            snapshot_path = None
            if args.save_snapshot:
                snapshot = SnapshotStore(paths.snapshots).save(engine)
                snapshot_path = str(snapshot.path)
        return {
            "business_code": "OK_RUN",
            "session_id": summary["session_id"],
            "view_id": summary["view_id"],
            "event_cursor": summary["event_cursor"],
            "finalized_window_count": len(summary["finalized_windows"]),
            "drop_count": summary["drop_count"],
            "late_accept_count": summary["late_accept_count"],
            "snapshot": snapshot_path,
            "ledger": str(paths.ledger),
        }
    if args.command == "scenario":
        paths = _paths(args)
        result = run_scenario(args.scenario, paths)
        return {
            "business_code": result["business_code"],
            "scenario": args.scenario,
            "report_path": str(paths.reports / f"scenario{args.scenario}.json"),
        }
    if args.command == "crash-writer":
        paths = _paths(args)
        paths.prepare()
        prepare_scenario3_crash(paths, Path(args.metadata))
        os._exit(17)
    if args.command == "demo":
        report = run_demo(args.workspace, clean=not args.no_clean)
        return {
            "business_code": report["business_code"],
            "all_passed": report["all_passed"],
            "report_path": str(Path(args.workspace).resolve() / "artifacts" / "demo" / "reports" / "demo_report.json"),
            "scenarios": report["scenarios"],
            "injections": report["injections"],
        }
    if args.command == "inject":
        paths = _paths(args)
        paths.prepare()
        if args.kind == "order":
            return inject_order(paths)
        if args.kind == "grid":
            return inject_grid(paths)
        if args.kind == "snapshot":
            return inject_snapshot(paths)
        return inject_replay(paths)
    if args.command == "ui":
        run_diagnostics(args.ledger, session_id=args.session_id, view_id=args.view_id)
        return {"business_code": "OK_UI"}
    if args.command == "restore":
        paths = _paths(args)
        paths.prepare()
        logger = JsonlLogger(paths.logs, "restore")
        with Ledger(paths.ledger) as ledger:
            engine, loaded, unknown = restore_engine(
                args.snapshot,
                store=SnapshotStore(paths.snapshots),
                ledger=ledger,
                logger=logger,
            )
            replay_report = validate_replay(engine, ledger) if engine.ledger is not None else None
        return {
            "business_code": "OK_RESTORE",
            "session_id": engine.session_id,
            "view_id": engine.view_id,
            "rules_version": engine.spec.rules_version,
            "unknown_windows": unknown,
            "comparison": loaded.comparison.to_dict() if loaded.comparison else None,
            "replay": replay_report,
        }
    if args.command == "replay":
        paths = _paths(args)
        logger = JsonlLogger(paths.logs, "replay")
        spec = load_spec(args.spec)
        with Ledger(paths.ledger) as ledger:
            runtime = ledger.get_runtime(args.session_id)
            view_id = args.view_id or runtime["current_view_id"]
            engine = Engine(
                spec,
                session_id=args.session_id,
                ledger=None,
                logger=logger,
                view_id=view_id,
            )
            report = validate_replay(engine, ledger, session_id=args.session_id, view_id=view_id)
        return {"business_code": "OK_REPLAY", **report}
    raise BusinessError(code="E_SPEC", message="unknown command", field="command")


def _logger_for_errors(args: argparse.Namespace) -> JsonlLogger | None:
    try:
        log_dir = getattr(args, "logs", None) or getattr(args, "log_dir", None)
        if not log_dir:
            workspace = Path(getattr(args, "workspace", ".")).resolve()
            log_dir = workspace / "artifacts" / "demo" / "logs"
        name = "errors"
        if getattr(args, "command", None) == "inject":
            name = f"injection_{args.kind}"
        return JsonlLogger(log_dir, name)
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = dispatch(args)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except BusinessError as error:
        logger = _logger_for_errors(args)
        if logger is not None:
            logger.error(error, event=f"{getattr(args, 'command', 'command')}_failed")
        print(json.dumps(error.as_dict(), ensure_ascii=False, sort_keys=True))
        return exit_code_for(error.code)
    except RuntimeError as error:
        logger = _logger_for_errors(args)
        if logger is not None:
            logger.emit(
                level="ERROR",
                code="E_INTERNAL",
                event="runtime_failure",
                message=str(error),
                details={"type": type(error).__name__},
            )
        print(json.dumps({"code": "E_INTERNAL", "message": str(error)}, ensure_ascii=False))
        return exit_code_for("E_INTERNAL")
    except Exception as error:
        logger = _logger_for_errors(args)
        if logger is not None:
            logger.error(error, event="unhandled_exception")
        print(
            json.dumps(
                {"code": "E_INTERNAL", "message": f"{type(error).__name__}: {error}"},
                ensure_ascii=False,
            )
        )
        return exit_code_for("E_INTERNAL")


if __name__ == "__main__":
    sys.exit(main())
