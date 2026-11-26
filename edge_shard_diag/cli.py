from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .acceptance import crash_recovery_demo, incremental_demo, normalize_demo_receipts, run_acceptance
from .demo import bootstrap_repository_demo
from .errors import DiagnosticError
from .ledger import Ledger
from .nodepool import NodePool
from .receipts import adopt_receipts, normalize_receipt_file
from .scheduler import Scheduler
from .spec import load_spec_bundle


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str))


def cmd_init_demo(args: argparse.Namespace) -> int:
    paths = bootstrap_repository_demo(args.root)
    _print({"ok": True, "paths": {key: str(value) for key, value in paths.items()}})
    return 0


def cmd_validate_spec(args: argparse.Namespace) -> int:
    bundle = load_spec_bundle(args.spec)
    _print(
        {
            "ok": True,
            "name": bundle.name,
            "spec_format_version": bundle.spec_format_version,
            "node_set_version": bundle.node_set_version,
            "content_digest": bundle.content_digest,
            "counts": {
                "nodes": len(bundle.nodes),
                "shards": len(bundle.shards),
                "samples": len(bundle.samples),
                "capacity_versions": len(bundle.capacity_versions),
                "fixed_revisions": len(bundle.fixed_revisions),
                "events": len(bundle.events),
            },
            "active_capacity_version": bundle.active_capacity_version,
            "active_fixed_revision": bundle.active_fixed_revision,
        }
    )
    return 0


def cmd_receipts(args: argparse.Namespace) -> int:
    bundle = load_spec_bundle(args.spec)
    entries = [("d1", args.d1, "d1"), ("d2", args.d2, "d2"), ("d3", args.d3, "d3")]
    all_records: list[dict[str, Any]] = []
    summary: dict[str, int] = {}
    for label, path, fmt in entries:
        if path is None:
            continue
        records = normalize_receipt_file(path, fmt, bundle=bundle)
        summary[label] = len(records)
        all_records.extend(records)
    all_records = adopt_receipts(all_records)
    result = {"ok": True, "counts": summary, "adopted": [r for r in all_records if r["adopted"]], "records": all_records}
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        result["out"] = str(out)
    _print(result if not args.quiet else {"ok": True, "counts": summary, "out": args.out})
    return 0


def cmd_ports(args: argparse.Namespace) -> int:
    bundle = load_spec_bundle(args.spec)
    _print({"ok": True, "ports": [{"node_id": n.id, "host": n.host, "port": n.port} for n in bundle.nodes]})
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    bundle = load_spec_bundle(args.spec)
    ledger = Ledger(args.db)
    recovery = ledger.recover_orphans()
    snapshot, replaced, reasons = ledger.open_or_replace_snapshot(
        bundle,
        bundle.active_fixed().scale_factor,
        {"cursor_key": "current"},
        {"command": "run"},
    )
    pool = NodePool(bundle, args.logs)
    pool.start()
    try:
        report = Scheduler(bundle, ledger, pool).run(lease_ttl_ms=args.ttl_ms)
    finally:
        pool.stop()
    result = {
        "ok": True,
        "report": report,
        "recovery_on_open": recovery,
        "snapshot": {"id": snapshot["snapshot_id"], "replaced": replaced, "reasons": reasons},
        "outcomes": ledger.list_outcomes(bundle.active_capacity_version),
    }
    ledger.close()
    _print(result)
    return 0


def cmd_crash_child(args: argparse.Namespace) -> int:
    bundle = load_spec_bundle(args.spec)
    ledger = Ledger(args.db)
    ledger.open_or_replace_snapshot(bundle, bundle.active_fixed().scale_factor, {"crash": True}, {"child": True})
    pool = NodePool(bundle, Path(args.db).parent / "crash_child_logs")
    pool.start()
    scheduler = Scheduler(bundle, ledger, pool)
    print(f"CRASH_CHILD_READY pid={os.getpid()}", flush=True)
    scheduler.run(crash_before_commit=args.crash_shard, lease_ttl_ms=10_000)
    pool.stop()
    ledger.close()
    return 0


def cmd_acceptance(args: argparse.Namespace) -> int:
    report = run_acceptance(args.workdir)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    summary = {
        "ok": True,
        "report": str(report_path),
        "workdir": report["workdir"],
        "receipt_counts": report["steps"]["receipts"]["counts"],
        "lease_wrong_version_code": report["steps"]["leases"]["wrong_version_error"]["code"],
        "incremental_invalid": report["steps"]["incremental"]["propagation"]["invalid_shards"],
        "byte_preservation": report["steps"]["incremental"]["byte_preservation"],
        "crash_sigkilled": report["steps"]["crash_recovery"]["child_was_sigkilled"],
        "recovery_unknown": report["steps"]["crash_recovery"]["recovery"]["unknown_shards"],
    }
    _print(summary)
    return 0


def cmd_verify_incremental(args: argparse.Namespace) -> int:
    result = incremental_demo(Path(args.workdir))
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    _print(
        {
            "ok": True,
            "report": str(report_path),
            "invalid_shards": result["propagation"]["invalid_shards"],
            "byte_preservation": result["byte_preservation"],
            "field_compare": result["field_compare"],
        }
    )
    return 0


def cmd_verify_recovery(args: argparse.Namespace) -> int:
    result = crash_recovery_demo(Path(args.workdir))
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    _print(
        {
            "ok": True,
            "report": str(report_path),
            "child_was_sigkilled": result["child_was_sigkilled"],
            "unknown_shards": result["recovery"]["unknown_shards"],
            "replay_compare": result["replay_compare"],
        }
    )
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    from .ui import run_tk_ui

    return run_tk_ui(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="edge-shard-diag", description="Edge inference shard scheduling diagnostics")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init-demo")
    init.add_argument("--root", default=".")
    init.set_defaults(func=cmd_init_demo)

    validate = sub.add_parser("validate-spec")
    validate.add_argument("--spec", default="config/demo")
    validate.set_defaults(func=cmd_validate_spec)

    receipts = sub.add_parser("receipts")
    receipts.add_argument("--spec", default="config/demo")
    receipts.add_argument("--d1", default="samples/receipts/receipts.d1")
    receipts.add_argument("--d2", default="samples/receipts/receipts.d2.jsonl")
    receipts.add_argument("--d3", default="samples/receipts/receipts.d3.jsonl")
    receipts.add_argument("--out", default="artifacts/receipts-normalized.json")
    receipts.add_argument("--quiet", action="store_true")
    receipts.set_defaults(func=cmd_receipts)

    ports = sub.add_parser("ports")
    ports.add_argument("--spec", default="config/demo")
    ports.set_defaults(func=cmd_ports)

    run = sub.add_parser("run")
    run.add_argument("--spec", default="config/demo")
    run.add_argument("--db", default="artifacts/runtime/ledger.sqlite")
    run.add_argument("--logs", default="artifacts/runtime/logs")
    run.add_argument("--ttl-ms", type=int, default=30_000)
    run.set_defaults(func=cmd_run)

    crash = sub.add_parser("crash-child")
    crash.add_argument("--spec", required=True)
    crash.add_argument("--db", required=True)
    crash.add_argument("--crash-shard", required=True)
    crash.set_defaults(func=cmd_crash_child)

    acceptance = sub.add_parser("acceptance")
    acceptance.add_argument("--workdir", default="artifacts/acceptance")
    acceptance.add_argument("--report", default="artifacts/validation-report.json")
    acceptance.set_defaults(func=cmd_acceptance)

    verify = sub.add_parser("verify-incremental")
    verify.add_argument("--workdir", default="artifacts/verify-incremental")
    verify.add_argument("--report", default="artifacts/incremental-verification.json")
    verify.set_defaults(func=cmd_verify_incremental)

    recover = sub.add_parser("verify-recovery")
    recover.add_argument("--workdir", default="artifacts/verify-recovery")
    recover.add_argument("--report", default="artifacts/recovery-verification.json")
    recover.set_defaults(func=cmd_verify_recovery)

    ui = sub.add_parser("ui")
    ui.add_argument("--spec", default="config/demo")
    ui.add_argument("--db", default="artifacts/runtime/ui-ledger.sqlite")
    ui.add_argument("--check", action="store_true", help="only report Tk availability")
    ui.set_defaults(func=cmd_ui)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except DiagnosticError as exc:
        print(json.dumps({"ok": False, "error": exc.to_dict()}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
