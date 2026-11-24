from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .convergence import check_restart_convergence
from .engine import RunOptions, run_engine
from .errors import LeaseError
from .spec import load_spec
from .util import atomic_write_json
from .verifier import verify_ledger


EXIT_CODES = {
    "E_SPEC": 2,
    "E_PORT": 3,
    "E_HANDSHAKE": 4,
    "E_ACK": 5,
    "E_VERIFY": 6,
    "E_LEDGER": 7,
    "E_QUEUE": 8,
    "E_EXPIRY": 8,
    "E_LEASE_LIMIT": 9,
    "E_CAPACITY": 9,
}


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _summary(report: dict[str, Any], report_path: Path) -> dict[str, Any]:
    scheduling = report.get("scheduling") or {}
    return {
        "ok": report.get("status") == "ok",
        "status": report.get("status"),
        "report_path": str(report_path),
        "ledger_path": report.get("ledger", {}).get("path"),
        "ledger_content_digest": report.get("ledger", {}).get("content_digest"),
        "reused_keys": (report.get("incremental") or {}).get("reusable_keys", []),
        "recomputed_keys": (report.get("incremental") or {}).get("recomputed_keys", []),
        "request_count": len(scheduling.get("requests", [])),
        "queue_wait_count": len(scheduling.get("queue_waits", [])),
        "peak_global_inflight": scheduling.get("peak_global_inflight"),
        "all_artifacts_reusable": all(
            artifact.get("reusable") for artifact in report.get("artifacts", [])
        ),
    }


def _run_command(args: argparse.Namespace) -> int:
    db_path = Path(args.db).resolve()
    report_path = Path(
        args.report or (str(db_path.with_suffix("")) + ".report.json")
    ).resolve()
    actual_db = db_path
    engine_report_path = report_path
    if args.replace:
        actual_db = db_path.with_name(db_path.name + ".rebuild.tmp")
        engine_report_path = report_path.with_name(report_path.name + ".rebuild.tmp")
        try:
            actual_db.unlink(missing_ok=True)
            engine_report_path.unlink(missing_ok=True)
            side = actual_db.with_name(actual_db.name + "-wal")
            side.unlink(missing_ok=True)
            side = actual_db.with_name(actual_db.name + "-shm")
            side.unlink(missing_ok=True)
        except OSError as exc:
            raise LeaseError(
                "E_LEDGER",
                f"cannot prepare temporary replacement ledger: {exc}",
                field="ledger.path",
                expected="writable temporary ledger path",
                actual=str(actual_db),
            ) from exc
    options = RunOptions(
        lease_ms=args.lease_ms,
        settle=args.settle,
        release_active_on_close=args.release_active,
        reuse_only=args.reuse_only,
        wait_timeout_ms=args.wait_timeout_ms,
        node_start_timeout_ms=args.node_start_timeout_ms,
        request_timeout_ms=args.request_timeout_ms,
        service_delay_ms=args.service_delay_ms,
        interrupt_after_grants=args.interrupt_after_grants,
        report_path=str(engine_report_path),
    )
    report = run_engine(args.spec, actual_db, args.workspace, options=options)
    if args.replace:
        backup = db_path.with_name(db_path.name + ".previous")
        if db_path.exists():
            os.replace(db_path, backup)
        for suffix in ("-wal", "-shm"):
            db_path.with_name(db_path.name + suffix).unlink(missing_ok=True)
        os.replace(actual_db, db_path)
        report["ledger"]["path"] = str(db_path)
        report["ledger"]["replaced_previous"] = backup.exists()
        report["full_rebuild_replacement"] = True
        atomic_write_json(report_path, report)
        engine_report_path.unlink(missing_ok=True)
    _print_json(_summary(report, report_path))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="leasecache",
        description=(
            "Deterministic cross-node pipeline artifact-cache lease arbitration"
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate a UTF-8 JSON specification")
    validate.add_argument("--spec", required=True)

    run = subparsers.add_parser("run", help="start nodes, schedule leases, persist a run")
    run.add_argument("--spec", required=True)
    run.add_argument("--db", required=True)
    run.add_argument("--workspace", required=True)
    run.add_argument("--report")
    run.add_argument("--lease-ms", type=int)
    run.add_argument("--wait-timeout-ms", type=int, default=20000)
    run.add_argument("--node-start-timeout-ms", type=int, default=6000)
    run.add_argument("--request-timeout-ms", type=int, default=6000)
    run.add_argument("--service-delay-ms", type=int, default=0, help=argparse.SUPPRESS)
    run.add_argument("--interrupt-after-grants", type=int, default=0, help=argparse.SUPPRESS)
    run.add_argument("--reuse-only", action="store_true")
    run.add_argument("--settle", dest="settle", action="store_true", default=True)
    run.add_argument("--no-settle", dest="settle", action="store_false")
    run.add_argument(
        "--release-active",
        dest="release_active",
        action="store_true",
        default=True,
    )
    run.add_argument(
        "--no-release-active",
        dest="release_active",
        action="store_false",
    )
    run.add_argument(
        "--replace",
        action="store_true",
        help="build a fresh ledger and atomically replace the target database",
    )

    verify = subparsers.add_parser("verify", help="independently replay and verify a ledger")
    verify.add_argument("--spec", required=True)
    verify.add_argument("--db", required=True)
    verify.add_argument("--out")

    converge = subparsers.add_parser(
        "converge", help="regenerate copies and restart one or more cache nodes"
    )
    converge.add_argument("--spec", required=True)
    converge.add_argument("--db", required=True)
    converge.add_argument("--workspace", required=True)
    converge.add_argument("--node", action="append", dest="nodes")
    converge.add_argument("--out")

    show = subparsers.add_parser("show", help="print ledger projections and recent events")
    show.add_argument("--db", required=True)
    show.add_argument("--events", type=int, default=20)

    demo = subparsers.add_parser("demo", help="run the complete reproducible acceptance scenario")
    demo.add_argument("--root", default="reports/acceptance")

    gui = subparsers.add_parser("gui", help="open the Tkinter desktop console")
    gui.add_argument("--spec", default="samples/demo_spec.json")
    gui.add_argument("--db", default="reports/gui.sqlite")
    gui.add_argument("--workspace", default="reports/gui-workspace")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            spec = load_spec(args.spec)
            _print_json(
                {
                    "ok": True,
                    "spec": str(spec.path),
                    "spec_sha256": spec.content_sha256,
                    "pipeline_id": spec.pipeline_id,
                    "nodes": [node.node_id for node in spec.nodes],
                    "artifact_layers": spec.topological_layers(),
                }
            )
            return 0
        if args.command == "run":
            return _run_command(args)
        if args.command == "verify":
            result = verify_ledger(args.spec, args.db)
            if args.out:
                atomic_write_json(Path(args.out), result)
            _print_json(result)
            return 0
        if args.command == "converge":
            result = check_restart_convergence(
                args.spec,
                args.db,
                args.workspace,
                node_ids=args.nodes,
                report_path=args.out,
            )
            _print_json(result)
            return 0
        if args.command == "show":
            from .ledger import Ledger

            with Ledger(args.db, create=False) as ledger:
                _print_json(
                    {
                        "nodes": ledger.nodes(),
                        "decisions": ledger.decisions(),
                        "leases": ledger.list_leases(),
                        "progress": ledger.progress(),
                        "events": ledger.events(limit=args.events),
                    }
                )
            return 0
        if args.command == "demo":
            from .demo import run_acceptance

            result = run_acceptance(Path(args.root))
            _print_json(result)
            return 0 if result.get("ok") else 1
        if args.command == "gui":
            from .gui import main as gui_main

            return gui_main(args.spec, args.db, args.workspace)
        parser.error("unknown command")
        return 2
    except LeaseError as exc:
        print(
            json.dumps(exc.to_dict(), ensure_ascii=False, sort_keys=True),
            file=sys.stderr,
        )
        return EXIT_CODES.get(exc.code, 1)
