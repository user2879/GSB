"""Command-line entry points, including genuine fault-injection scenarios."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

from .errors import WorkbenchError
from .logging_utils import JsonlLogger, utc_now
from .samples import DEFAULT_SAMPLES_DIR
from .workbench import Workbench


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _print_payload(payload: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))


def _create_demo_workbench(work_dir: Path, log_path: Path) -> tuple[Workbench, Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    session_path = work_dir / "demo.sqlite"
    if session_path.exists():
        session_path.unlink()
    workbench = Workbench.create(
        session_path,
        name="slow-query-demo",
        overwrite=True,
        logger_path=log_path,
    )
    workbench.initialize_samples(DEFAULT_SAMPLES_DIR)
    return workbench, session_path


def run_demo(args: argparse.Namespace) -> int:
    work_dir = Path(args.work_dir).resolve()
    log_path = Path(args.log).resolve() if args.log else work_dir / "workbench.jsonl"
    logger = JsonlLogger(log_path)
    logger.info("cli_started", "demo command started", inject=args.inject, work_dir=str(work_dir))

    if args.inject == "data":
        session_path = work_dir / "inject-data.sqlite"
        work_dir.mkdir(parents=True, exist_ok=True)
        workbench = Workbench.create(
            session_path,
            name="inject-data",
            overwrite=True,
            logger_path=log_path,
        )
        try:
            workbench.initialize_samples(DEFAULT_SAMPLES_DIR, corrupt_input=True)
        except WorkbenchError as exc:
            logger.error(
                "cli_failure",
                exc.message,
                code=exc.code,
                details=exc.details,
                command="demo",
                injection="data",
                log=str(log_path),
            )
            _print_payload(
                {"ok": False, "injection": "data", **exc.to_dict(), "log": str(log_path)},
                args.json,
            )
            return 2
        finally:
            workbench.close()
        logger.error("cli_failure", "data corruption injection did not fail", code="E_DATA", command="demo")
        _print_payload({"ok": False, "injection": "data", "code": "E_DATA", "message": "injection did not fail", "log": str(log_path)}, args.json)
        return 2

    workbench, session_path = _create_demo_workbench(work_dir, log_path)
    try:
        query_catalog = workbench.sample_queries(DEFAULT_SAMPLES_DIR)
        query_entry = next(item for item in query_catalog if item["name"] == "slow_join_city_status")
        query_sql = str(query_entry["sql"])

        if args.inject == "index-version":
            versions = workbench.versions()
            workbench.close()
            try:
                Workbench.open(
                    session_path,
                    expected_versions={"index_version": int(versions["index_version"]) + 1},
                    logger_path=log_path,
                )
            except WorkbenchError as exc:
                logger.error(
                    "cli_failure",
                    exc.message,
                    code=exc.code,
                    details=exc.details,
                    command="demo",
                    injection="index-version",
                    log=str(log_path),
                )
                _print_payload(
                    {"ok": False, "injection": "index-version", **exc.to_dict(), "log": str(log_path)},
                    args.json,
                )
                return 2
            logger.error(
                "cli_failure",
                "index version injection did not fail",
                code="E_VERSION",
                command="demo",
            )
            return 2

        if args.inject == "timeout":
            timeout_query = next(item for item in query_catalog if item["name"] == "slow_customer_segment")
            try:
                workbench.execute_select(str(timeout_query["sql"]), timeout_ms=args.inject_timeout_ms)
            except WorkbenchError as exc:
                logger.error(
                    "cli_failure",
                    exc.message,
                    code=exc.code,
                    details=exc.details,
                    command="demo",
                    injection="timeout",
                    log=str(log_path),
                )
                _print_payload(
                    {"ok": False, "injection": "timeout", **exc.to_dict(), "log": str(log_path)},
                    args.json,
                )
                return 2
            logger.error(
                "cli_failure",
                "timeout injection did not fail",
                code="E_TIMEOUT",
                command="demo",
                timeout_ms=args.inject_timeout_ms,
            )
            return 2

        original = workbench.execute_select(query_sql, timeout_ms=500)
        slow = workbench.slow_queries()
        diagnosis = workbench.diagnose(
            query_sql,
            timeout_ms=500,
            budget_ms=4000,
            threshold_ms=0,
            threshold_rows=0,
        )
        query_id = diagnosis["query_id"]

        index_result = workbench.add_index("orders", ["amount"], name="idx_orders_amount")
        stale_after_index = workbench.current_best(query_id) is None
        invalidation_event = workbench.change_events()[-1]

        fresh_diagnosis = workbench.diagnose(
            query_sql,
            timeout_ms=500,
            budget_ms=4000,
            threshold_ms=0,
            threshold_rows=0,
        )
        snapshot_path = workbench.save_snapshot(work_dir / "demo-snapshot.sqlite")

        workbench.append_rows(
            "orders",
            ["id", "customer_id", "status", "amount", "created_at"],
            [(999999, 1, "paid", 123.45, "2024-12-31")],
        )
        restore_conflict: dict[str, Any] | None = None
        try:
            workbench.restore_snapshot(snapshot_path)
        except WorkbenchError as exc:
            if exc.code != "E_VERSION":
                raise
            restore_conflict = exc.to_dict()

        replay = workbench.replay_conclusions(repeats=2, timeout_ms=500, budget_ms=8000)
        payload = {
            "ok": True,
            "session": str(session_path),
            "snapshot": str(snapshot_path),
            "log": str(log_path),
            "query_id": query_id,
            "slow_query_count": len(slow),
            "original_elapsed_ms": original["elapsed_ms"],
            "candidate_kinds": [item["kind"] for item in diagnosis["candidates"]],
            "candidate_statuses": {item["kind"]: item["status"] for item in diagnosis["candidates"]},
            "optimal_candidate_id": fresh_diagnosis["conclusion"]["optimal_candidate_id"],
            "index_version_after_add": index_result["versions"]["index_version"],
            "stale_after_index_change": stale_after_index,
            "invalidation_event": invalidation_event,
            "snapshot_restore_version_error": restore_conflict,
            "restored_versions": workbench.versions(),
            "replay_matched": replay["matched"],
            "replay_validation_id": replay["validation_id"],
        }
        logger.info("cli_completed", "demo command completed", **payload)
        _print_payload(payload, args.json)
        return 0
    finally:
        workbench.close()


def run_check(args: argparse.Namespace) -> int:
    tkinter_status = "ok"
    try:
        import tkinter  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        tkinter_status = f"unavailable: {exc}"
    payload = {
        "python": sys.version.split()[0],
        "sqlite": sqlite3.sqlite_version,
        "sqlite_stdlib": "ok",
        "tkinter": tkinter_status,
        "clock": "time.perf_counter_ns monotonic milliseconds",
    }
    _print_payload(payload, args.json)
    return 0


def run_gui(args: argparse.Namespace) -> int:
    try:
        from .gui import run_gui as launch_gui
    except Exception as exc:
        print(f"Tkinter GUI unavailable: {exc}", file=sys.stderr)
        return 2
    return launch_gui(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m slowquery_workbench")
    subparsers = parser.add_subparsers(dest="command")

    demo = subparsers.add_parser("demo", help="run the deterministic end-to-end scenario")
    demo.add_argument("--work-dir", default=".workbench-demo")
    demo.add_argument("--log", default=None)
    demo.add_argument("--json", action="store_true")
    demo.add_argument(
        "--inject",
        choices=("none", "data", "index-version", "timeout"),
        default="none",
        help="inject an input-side fault (never a fabricated exception or sleep)",
    )
    demo.add_argument("--inject-timeout-ms", type=int, default=1)
    demo.set_defaults(func=run_demo)

    check = subparsers.add_parser("check", help="check Python, SQLite and Tkinter")
    check.add_argument("--json", action="store_true")
    check.set_defaults(func=run_check)

    gui = subparsers.add_parser("gui", help="launch the Tkinter workbench")
    gui.add_argument("--session", default=None)
    gui.add_argument("--samples", default=str(DEFAULT_SAMPLES_DIR))
    gui.set_defaults(func=run_gui)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        args = parser.parse_args(["gui"])
    try:
        return int(args.func(args))
    except WorkbenchError as exc:
        logger = JsonlLogger(Path(args.log).resolve() if getattr(args, "log", None) else Path("workbench.jsonl"))
        logger.error("cli_failure", exc.message, code=exc.code, details=exc.details, command=args.command)
        _print_payload({"ok": False, **exc.to_dict(), "log": str(logger.path)}, getattr(args, "json", False))
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
