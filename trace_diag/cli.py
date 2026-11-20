from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .canonical import write_json_atomic
from .errors import AppError, SnapshotError
from .ledger import Ledger
from .logging_utils import JsonlLogger, print_json
from .replay import verify_ledger_run
from .samplegen import generate_sample_files
from .spec import load_policies_file, load_services_file
from .workflow import analyze_and_persist, recover_latest

DEFAULT_SERVICES = "data/services.json"
DEFAULT_POLICIES = "data/policies.json"
DEFAULT_DB = "data/ledger/trace_diag.sqlite"
DEFAULT_LOGS = "data/logs"


def _logger(args: argparse.Namespace) -> JsonlLogger:
    return JsonlLogger(args.log_dir)


def _common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--services", default=DEFAULT_SERVICES, help="service declaration JSON")
    parser.add_argument("--policies", default=DEFAULT_POLICIES, help="sampling policy JSON")
    parser.add_argument("--db", default=DEFAULT_DB, help="SQLite ledger path")
    parser.add_argument("--log-dir", default=DEFAULT_LOGS, help="structured JSONL log directory")


def _summary(result: Any) -> dict[str, Any]:
    bundle = result.bundle
    diagnostics = bundle.diagnostics
    return {
        "ok": True,
        "run_id": result.run_id,
        "mode": result.mode,
        "service_version": bundle.services_doc.version,
        "strategy_version": bundle.policies_doc.version,
        "snapshot_digest": result.view["run"]["snapshot_digest"],
        "services": [row["service_id"] for row in diagnostics["service_ranking"]],
        "service_ranking": diagnostics["service_ranking"],
        "selected_policies": {
            service_id: row["winner"]
            for service_id, row in diagnostics["policy_choices"].items()
        },
        "boundary_decision_count": len(diagnostics["boundary_decisions"]),
        "flip_event_count": len(bundle.flip_events),
        "flip_events": list(bundle.flip_events),
        "newly_committed": list(result.newly_committed),
        "skipped_committed": list(result.skipped_committed),
        "batches": [
            {
                "ordinal": service["ordinal"],
                "service_id": service["service_id"],
                "status": service["status"],
                "progress_cursor": service["progress_cursor"],
            }
            for service in result.view["services"]
        ],
    }


def command_generate(args: argparse.Namespace) -> int:
    paths = generate_sample_files(args.data_dir, record_count=args.records)
    logger = _logger(args)
    logger.event(
        "samples_generated",
        context={
            "services": str(paths["services"]),
            "policies": str(paths["policies"]),
            "records_per_service": args.records,
        },
    )
    print_json(
        {
            "ok": True,
            "services": str(paths["services"]),
            "policies": str(paths["policies"]),
            "records_per_service": args.records,
        }
    )
    return 0


def _inject_snapshot(args: argparse.Namespace, logger: JsonlLogger) -> int:
    services_doc = load_services_file(args.services)
    policies_doc = load_policies_file(args.policies)
    result = analyze_and_persist(
        services_doc, policies_doc, args.db, logger=logger
    )
    with Ledger(args.db, mark_pending_unknown=False) as ledger:
        original = ledger.corrupt_snapshot_digest(result.run_id)
        try:
            ledger.get_run_view(result.run_id, verify=True)
        except SnapshotError as exc:
            logger.event(
                "injection_detected",
                level="ERROR",
                code=exc.code,
                path=exc.path,
                message=exc.message,
                context={"injection": "snapshot", **exc.context},
            )
            return _error(exc)
        finally:
            ledger.restore_snapshot_digest(result.run_id, original)
    raise SnapshotError("$.snapshot", "snapshot injection did not trigger a mismatch")


def command_analyze(args: argparse.Namespace) -> int:
    logger = _logger(args)
    if args.inject == "sample":
        try:
            load_services_file(args.services, inject_missing_id=True)
        except AppError as exc:
            logger.event(
                "injection_detected",
                level="ERROR",
                code=exc.code,
                path=exc.path,
                message=exc.message,
                context={"injection": "sample"},
            )
            return _error(exc)
    if args.inject == "policy-ratio":
        try:
            load_policies_file(args.policies, inject_ratio_bucket_error=True)
        except AppError as exc:
            logger.event(
                "injection_detected",
                level="ERROR",
                code=exc.code,
                path=exc.path,
                message=exc.message,
                context={"injection": "policy-ratio"},
            )
            return _error(exc)
    if args.inject == "snapshot":
        return _inject_snapshot(args, logger)

    services_doc = load_services_file(args.services)
    policies_doc = load_policies_file(args.policies)
    result = analyze_and_persist(
        services_doc,
        policies_doc,
        args.db,
        logger=logger,
        include_sensitivity=not args.no_sensitivity,
    )
    report = result.bundle.to_report()
    report["ledger_run"] = result.view["run"]
    report["ledger_batches"] = [
        {
            "ordinal": service["ordinal"],
            "service_id": service["service_id"],
            "status": service["status"],
            "progress_cursor": service["progress_cursor"],
            "content_digest": service["content_digest"],
        }
        for service in result.view["services"]
    ]
    if args.report_out:
        write_json_atomic(Path(args.report_out), report)
    print_json(_summary(result))
    return 0


def command_status(args: argparse.Namespace) -> int:
    logger = _logger(args)
    with Ledger(args.db) as ledger:
        runs = ledger.list_runs()
        current_row = ledger.get_current_run_row()
        current = None
        if current_row:
            current = ledger.get_run_view(current_row["run_id"], verify=True)
        recoverable = ledger.recoverable_runs()
        payload: dict[str, Any] = {
            "ok": True,
            "database": str(Path(args.db)),
            "runs": runs,
            "current_run_id": current_row["run_id"] if current_row else None,
            "recoverable_runs": [
                {
                    "run_id": run["run_id"],
                    "status": run["status"],
                    "committed_batches": run["committed_batches"],
                    "expected_batches": run["expected_batches"],
                    "unknown": [
                        batch["service_id"]
                        for batch in ledger.unknown_batches(run["run_id"])
                    ],
                }
                for run in recoverable
            ],
            "batches": ledger.list_batches(current_row["run_id"]) if current_row else [],
        }
        if current:
            payload["current"] = {
                "run": current["run"],
                "services": [
                    {
                        "service_id": service["service_id"],
                        "status": service["status"],
                        "record_count": (
                            service["summary"]["stream_summary"]["record_count"]
                            if service["summary"]
                            else None
                        ),
                    }
                    for service in current["services"]
                ],
                "flip_count": len(current["flips"]),
            }
    logger.event("status_viewed", context={"current_run_id": payload.get("current_run_id")})
    print_json(payload)
    return 0


def command_verify(args: argparse.Namespace) -> int:
    logger = _logger(args)
    result = verify_ledger_run(args.db, run_id=args.run_id)
    logger.event(
        "replay_verified",
        context={
            "run_id": result["run_id"],
            "snapshot_digest": result["snapshot_digest"],
            "services_checked": result["services_checked"],
        },
    )
    print_json(result)
    return 0


def command_recover(args: argparse.Namespace) -> int:
    logger = _logger(args)
    result = recover_latest(args.db, logger=logger)
    payload = _summary(result)
    payload["recovered"] = True
    print_json(payload)
    return 0


def command_demo(args: argparse.Namespace) -> int:
    generate_sample_files(args.data_dir, record_count=args.records)
    logger = _logger(args)
    services_doc = load_services_file(Path(args.data_dir) / "services.json")
    policies_doc = load_policies_file(Path(args.data_dir) / "policies.json")
    result = analyze_and_persist(
        services_doc,
        policies_doc,
        args.db,
        logger=logger,
    )
    verification = verify_ledger_run(args.db, run_id=result.run_id)
    payload = _summary(result)
    payload["verification"] = verification
    report_path = Path(args.data_dir) / "diagnostic_report.json"
    report = result.bundle.to_report()
    report["verification"] = verification
    report["ledger_run"] = result.view["run"]
    write_json_atomic(report_path, report)
    payload["report"] = str(report_path)
    print_json(payload)
    return 0


def command_gui(args: argparse.Namespace) -> int:
    from .gui import launch_gui

    launch_gui(
        services_path=Path(args.services),
        policies_path=Path(args.policies),
        db_path=Path(args.db),
        log_dir=Path(args.log_dir),
    )
    return 0


def _error(exc: AppError) -> int:
    print_json({"ok": False, "error": exc.as_dict()})
    if exc.code == "E_REPLAY":
        return 3
    if exc.code == "E_SNAPSHOT":
        return 4
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trace-diag")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-samples", help="write deterministic local sample JSON")
    generate.add_argument("--data-dir", default="data")
    generate.add_argument("--records", type=int, default=600)
    generate.add_argument("--log-dir", default=DEFAULT_LOGS)
    generate.set_defaults(handler=command_generate)

    analyze = subparsers.add_parser("analyze", help="analyze, commit batches, and activate a replacement run")
    _common_paths(analyze)
    analyze.add_argument("--report-out", default=None)
    analyze.add_argument("--no-sensitivity", action="store_true")
    analyze.add_argument(
        "--inject",
        choices=("sample", "policy-ratio", "snapshot"),
        default=None,
        help="exercise a controlled error path",
    )
    analyze.set_defaults(handler=command_analyze)

    status = subparsers.add_parser("status", help="show active run and recoverable batches")
    status.add_argument("--db", default=DEFAULT_DB)
    status.add_argument("--log-dir", default=DEFAULT_LOGS)
    status.set_defaults(handler=command_status)

    verify = subparsers.add_parser("verify", help="independently replay ledger statistics and attribution")
    verify.add_argument("--db", default=DEFAULT_DB)
    verify.add_argument("--run-id", default=None)
    verify.add_argument("--log-dir", default=DEFAULT_LOGS)
    verify.set_defaults(handler=command_verify)

    recover = subparsers.add_parser("recover", help="resume committed batches and rerun unknown batches")
    recover.add_argument("--db", default=DEFAULT_DB)
    recover.add_argument("--log-dir", default=DEFAULT_LOGS)
    recover.set_defaults(handler=command_recover)

    demo = subparsers.add_parser("demo", help="generate samples, analyze, verify, and write a report")
    demo.add_argument("--data-dir", default="data")
    demo.add_argument("--db", default=DEFAULT_DB)
    demo.add_argument("--records", type=int, default=600)
    demo.add_argument("--log-dir", default=DEFAULT_LOGS)
    demo.set_defaults(handler=command_demo)

    gui = subparsers.add_parser("gui", help="open the Tkinter workbench")
    _common_paths(gui)
    gui.set_defaults(handler=command_gui)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (AppError, RuntimeError) as exc:
        logger = JsonlLogger(getattr(args, "log_dir", DEFAULT_LOGS))
        if isinstance(exc, AppError):
            logger.event(
                "error",
                level="ERROR",
                code=exc.code,
                path=exc.path,
                message=exc.message,
                context=exc.context,
            )
            return _error(exc)
        logger.event("error", level="ERROR", code="E_ENV", message=str(exc))
        print_json({"ok": False, "error": {"code": "E_ENV", "message": str(exc)}})
        return 1


if __name__ == "__main__":
    sys.exit(main())
