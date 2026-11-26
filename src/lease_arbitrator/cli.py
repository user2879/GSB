from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .cluster import NodeCluster
from .engine import ArbitrationEngine
from .envcheck import collect_environment
from .errors import LeaseArbitratorError, SpecError, VerificationError
from .hashing import sha256_digest
from .invalidation import plan_edit_invalidation
from .ledger import Ledger
from .replay import DeterministicReplay, verify_ledger
from .spec import load_bundle, load_events, load_json_file, load_spec, validate_spec


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str))


def cmd_validate(args: argparse.Namespace) -> int:
    raw = load_json_file(args.spec)
    spec = validate_spec(raw)
    event_count = None
    if args.events:
        events = load_events(args.events, spec)
        event_count = len(events)
    print_json(
        {
            "ok": True,
            "code": "OK",
            "spec": args.spec,
            "events": args.events,
            "node_count": len(spec["nodes"]),
            "lease_count": len(spec["leases"]),
            "event_count": event_count,
            "first_error_path": None,
        }
    )
    return 0


def cmd_env_check(_args: argparse.Namespace) -> int:
    info = collect_environment()
    print_json(info)
    return 0 if info["sqlite3"]["available"] and info["tkinter"]["available"] else 1


def _reset_ledger(path: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        target = Path(str(path) + suffix)
        target.unlink(missing_ok=True)


def cmd_run(args: argparse.Namespace) -> int:
    spec, events = load_bundle(args.spec, args.events)
    ledger_path = Path(args.ledger)
    if args.reset_ledger:
        _reset_ledger(ledger_path)
    ledger = Ledger(ledger_path)
    ledger.conn.execute(
        "INSERT OR REPLACE INTO meta(key,value) VALUES('spec_digest',?)",
        (sha256_digest(spec),),
    )
    ledger.conn.execute(
        "INSERT OR REPLACE INTO meta(key,value) VALUES('events_digest',?)",
        (sha256_digest(events),),
    )
    ledger.conn.commit()

    cluster = NodeCluster(
        spec["nodes"],
        timeout=spec["node_timeout_seconds"],
        log_dir=args.node_log_dir,
    )
    verification = None
    run_report: dict[str, Any] = {}
    engine: ArbitrationEngine | None = None
    try:
        cluster.start()
        engine = ArbitrationEngine(spec, events, cluster, ledger)
        run_report = engine.run(settle_seconds=args.settle_seconds)
        try:
            verification = verify_ledger(spec, events, ledger_path)
        except VerificationError as exc:
            verification = {"ok": False, **exc.to_dict()}
            run_report["ok"] = False
        run_report["verification"] = verification
        run_report["ledger"] = {
            "path": str(ledger_path),
            "entries": len(ledger.entries()),
            "cursor": ledger.cursor(),
        }
        run_report["spec_path"] = str(args.spec)
        run_report["events_path"] = str(args.events)
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(run_report, ensure_ascii=False, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
    finally:
        if engine is not None:
            engine.shutdown()
        cluster.stop()
        ledger.close()

    summary = {
        "ok": run_report.get("ok", False),
        "report": args.report,
        "ledger": args.ledger,
        "verification": verification,
        "current_lease_states": [
            {"lease_id": lease["lease_id"], "state": lease["state"], "generation": lease["generation_current"]}
            for lease in run_report.get("current_view", {}).get("leases", [])
        ],
    }
    print_json(summary)
    return 0 if run_report.get("ok") and verification and verification.get("ok") else 4


def cmd_verify(args: argparse.Namespace) -> int:
    spec, events = load_bundle(args.spec, args.events)
    result = verify_ledger(spec, events, args.ledger)
    print_json(result)
    return 0


def cmd_invalidate(args: argparse.Namespace) -> int:
    spec = validate_spec(load_json_file(args.spec))
    events = load_events(args.events, spec) if args.events else None
    edit = load_json_file(args.edit)
    result = plan_edit_invalidation(spec, edit, events=events)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print_json(result)
    return 0


def cmd_water_report(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    events = load_events(args.events, spec) if args.events else []
    replay = DeterministicReplay(spec, events)
    for event in events:
        if event["type"] == "apply" and event["arrival_ms"] <= args.at_ms:
            replay.domain.register_apply(event["lease_id"], args.at_ms, event["seq"])
    replay.admit_pending(args.at_ms)
    result = {
        "ok": True,
        "at_ms": args.at_ms,
        "water": replay.domain.water_report(),
        "decisions": replay.domain.decisions,
        "leases": replay.domain.lease_view(),
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
    print_json({"ok": True, "out": args.out, "at_ms": args.at_ms})
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    from .ui import launch_report_viewer

    return launch_report_viewer(args.report)


def cmd_demo(args: argparse.Namespace) -> int:
    root = PROJECT_ROOT
    demo_args = argparse.Namespace(
        spec=str(root / "samples" / "spec.json"),
        events=str(root / "samples" / "events.ndjson"),
        ledger=str(root / "ledger" / "demo.sqlite3"),
        report=str(root / "reports" / "demo-report.json"),
        node_log_dir=str(root / "reports" / "node-logs"),
        reset_ledger=True,
        settle_seconds=0.8,
    )
    return cmd_run(demo_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lease-arbitrator")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate JSON declaration and NDJSON events")
    validate.add_argument("--spec", required=True)
    validate.add_argument("--events")
    validate.set_defaults(func=cmd_validate)

    env_check = sub.add_parser("env-check", help="check Python, sqlite3, tkinter, numpy and pytest")
    env_check.set_defaults(func=cmd_env_check)

    run = sub.add_parser("run", help="start node subprocesses and execute event stream")
    run.add_argument("--spec", required=True)
    run.add_argument("--events", required=True)
    run.add_argument("--ledger", required=True)
    run.add_argument("--report", required=True)
    run.add_argument("--node-log-dir", default="reports/node-logs")
    run.add_argument("--reset-ledger", action="store_true")
    run.add_argument("--settle-seconds", type=float, default=0.8)
    run.set_defaults(func=cmd_run)

    verify = sub.add_parser("verify", help="independently replay declaration and verify ledger")
    verify.add_argument("--spec", required=True)
    verify.add_argument("--events", required=True)
    verify.add_argument("--ledger", required=True)
    verify.set_defaults(func=cmd_verify)

    invalidate = sub.add_parser("invalidate", help="plan stable invalidation propagation")
    invalidate.add_argument("--spec", required=True)
    invalidate.add_argument("--edit", required=True)
    invalidate.add_argument("--events")
    invalidate.add_argument("--out")
    invalidate.set_defaults(func=cmd_invalidate)

    water = sub.add_parser("water-report", help="generate deterministic fixed-point water/adjudication report")
    water.add_argument("--spec", required=True)
    water.add_argument("--events")
    water.add_argument("--at-ms", type=int, default=0)
    water.add_argument("--out")
    water.set_defaults(func=cmd_water_report)

    ui = sub.add_parser("ui", help="open a read-only Tk report viewer")
    ui.add_argument("--report", required=True)
    ui.set_defaults(func=cmd_ui)

    demo = sub.add_parser("demo", help="run the bundled deterministic sample")
    demo.set_defaults(func=cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except VerificationError as exc:
        print_json({"ok": False, **exc.to_dict()})
        return 3
    except SpecError as exc:
        print_json({"ok": False, **exc.to_dict()})
        return 2
    except LeaseArbitratorError as exc:
        print_json({"ok": False, **exc.to_dict()})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
