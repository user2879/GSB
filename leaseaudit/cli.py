from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .clock import RealMonotonicClock
from .crash_recovery import run_crash_child, run_crash_recovery_demo
from .demo import DEFAULT_TEMPLATE, run_demo
from .engine import LeaseEngine
from .errors import BusinessError
from .faults import run_fault_injection
from .gui import run_gui
from .ledger import AuditLedger
from .logging_utils import StructuredLogger
from .replay import replay_ledger
from .scenario import load_scenario
from .view import build_audit_view, view_as_text


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _logger(args: argparse.Namespace) -> StructuredLogger:
    path = getattr(args, "log", None)
    if path is None:
        workdir = getattr(args, "workdir", None)
        if workdir:
            path = str(Path(workdir) / "structured.jsonl")
    return StructuredLogger(path)


def _cmd_validate(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.scenario)
    result = {
        "status": "OK",
        "business_code": "OK",
        "scenario_version": scenario.scenario_version,
        "scenario_digest": scenario.scenario_digest,
        "params_version": scenario.lease_params_version,
        "params_digest": scenario.params_digest,
        "external_digest": scenario.external_digest,
        "initial_global_allocatable": scenario.initial_global_allocatable,
        "sources": [source.source_id for source in scenario.ordered_sources()],
    }
    _print_json(result)
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.scenario) if args.scenario else None
    result = replay_ledger(args.ledger, scenario)
    if args.json:
        _print_json(result)
    else:
        print(
            f"AUDIT status={result['status']} code={result['business_code']} "
            f"events={result['event_count']} digest={result['content_digest']}"
        )
        if result["external_drift"]:
            print(
                "EXTERNAL_DRIFT "
                f"ledger_version={result['external_drift']['ledger_scenario_version']} "
                f"current_version={result['external_drift']['current_scenario_version']}"
            )
            for diff in result["external_drift"]["field_differences"]:
                print(
                    f"  field={diff['field_path']} ledger={diff['ledger']} current={diff['current']}"
                )
    return 0 if result["status"] == "OK" else 3


def _cmd_view(args: argparse.Namespace) -> int:
    view = build_audit_view(args.ledger)
    if args.json:
        _print_json(view)
    else:
        print(view_as_text(view))
    return 0


def _cmd_gui(args: argparse.Namespace) -> int:
    run_gui(args.ledger)
    return 0


def _cmd_snapshot(args: argparse.Namespace) -> int:
    ledger = AuditLedger(args.ledger, readonly=False, verify=True)
    try:
        manifest = ledger.create_snapshot(args.out)
    finally:
        ledger.close()
    _print_json(manifest)
    return 0


def _cmd_restore(args: argparse.Namespace) -> int:
    result = AuditLedger.restore_snapshot(args.snapshot, args.ledger)
    _print_json(result)
    return 0


def _cmd_recover(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.scenario)
    ledger = AuditLedger(args.ledger, scenario=scenario, verify=True)
    clock = RealMonotonicClock()
    engine = LeaseEngine(scenario, ledger, clock=clock)
    try:
        expired = engine.process_expirations(now_ms=clock.now_ms())
        state = engine.snapshot()
    finally:
        engine.close()
    result = {
        "status": "RECOVERED",
        "business_code": "OK",
        "expired_during_recovery": [
            {"seq": event.seq, "source_id": event.source_id, "lease_id": event.lease_id}
            for event in expired
        ],
        "state": state,
        "exit_report_available": state["exit_report_available"],
    }
    _print_json(result)
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    summary = run_demo(
        args.workdir,
        template_path=args.template,
        base_port=args.base_port,
        fresh=not args.keep_existing,
    )
    if args.json:
        _print_json(summary)
    else:
        report = Path(args.workdir) / "report.txt"
        print(report.read_text(encoding="utf-8"), end="")
    return 0 if summary["overall_code"] == "OK" else 1


def _cmd_fault(args: argparse.Namespace) -> int:
    result = run_fault_injection(args.workdir, args.kind)
    _print_json(result)
    return 0 if result["status"] == "EXPECTED_FAILURE" else 1


def _cmd_crash_demo(args: argparse.Namespace) -> int:
    result = run_crash_recovery_demo(args.workdir)
    _print_json(result)
    return 0 if result["status"] == "OK" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="leaseaudit",
        description="Monotonic concurrency lease, quota reclaim, and SQLite audit toolkit",
    )
    parser.add_argument("--log", help="optional JSONL structured log path")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="strictly validate a scenario file")
    validate.add_argument("--scenario", required=True)
    validate.set_defaults(handler=_cmd_validate)

    audit = subparsers.add_parser("audit", help="replay and audit a ledger independently")
    audit.add_argument("--ledger", required=True)
    audit.add_argument("--scenario", required=True)
    audit.add_argument("--json", action="store_true")
    audit.set_defaults(handler=_cmd_audit)

    view = subparsers.add_parser("view", help="print the stable read-only audit view")
    view.add_argument("--ledger", required=True)
    view.add_argument("--json", action="store_true")
    view.set_defaults(handler=_cmd_view)

    gui = subparsers.add_parser("gui", help="open the Tkinter read-only audit UI")
    gui.add_argument("--ledger", required=True)
    gui.set_defaults(handler=_cmd_gui)

    snapshot = subparsers.add_parser("snapshot", help="export an integrity-verified ledger snapshot")
    snapshot.add_argument("--ledger", required=True)
    snapshot.add_argument("--out", required=True)
    snapshot.set_defaults(handler=_cmd_snapshot)

    restore = subparsers.add_parser("restore", help="verify and atomically replace a ledger from a snapshot")
    restore.add_argument("--snapshot", required=True)
    restore.add_argument("--ledger", required=True)
    restore.set_defaults(handler=_cmd_restore)

    recover = subparsers.add_parser("recover", help="recover committed state after a forced termination")
    recover.add_argument("--ledger", required=True)
    recover.add_argument("--scenario", required=True)
    recover.set_defaults(handler=_cmd_recover)

    demo = subparsers.add_parser("demo", help="run the three-phase deterministic loopback demo")
    demo.add_argument("--workdir", default="artifacts/demo")
    demo.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    demo.add_argument("--base-port", type=int, default=55441)
    demo.add_argument("--keep-existing", action="store_true")
    demo.add_argument("--json", action="store_true")
    demo.set_defaults(handler=_cmd_demo)

    fault = subparsers.add_parser("fault-demo", help="inject one of the three required fault classes")
    fault.add_argument("--workdir", default="artifacts/faults")
    fault.add_argument(
        "--kind",
        required=True,
        choices=("scene_corrupt", "quota_overflow", "ledger_digest"),
    )
    fault.set_defaults(handler=_cmd_fault)

    crash = subparsers.add_parser("crash-demo", help="simulate forced termination and committed-event recovery")
    crash.add_argument("--workdir", default="artifacts/crash-recovery")
    crash.set_defaults(handler=_cmd_crash_demo)

    child = subparsers.add_parser("_crash-child", help=argparse.SUPPRESS)
    child.add_argument("--scenario", required=True)
    child.add_argument("--ledger", required=True)
    child.set_defaults(handler=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "_crash-child":
        return run_crash_child(args.scenario, args.ledger)
    logger = _logger(args)
    try:
        return int(args.handler(args))
    except BusinessError as exc:
        logger.error(exc)
        _print_json(exc.to_dict())
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
