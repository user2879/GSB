"""Command line entry points for the batch attribution diagnostic component."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace

from .engine import DiagnosticEngine
from .errors import DiagnosticError, SpecError
from .invalidation import propagate
from .ledger import SnapshotLedger
from .logging_utils import StructuredLogger
from .network import check_port_bindable
from .planning import build_plan
from .replay import replay_ledger
from .spec import load_spec, load_json_file
from .verifier import verify_ledger


def _emit(payload: dict) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


def _default_logger(args: argparse.Namespace) -> StructuredLogger:
    log_path = getattr(args, "log_path", None) or os.path.join(
        os.path.dirname(os.path.abspath(args.ledger)), "events.jsonl"
    )
    return StructuredLogger(log_path)


def _check_all_ports(spec) -> None:
    seen: set[int] = set()
    for worker in spec.workers:
        if worker.port in seen:  # spec validation already guards, kept defensive
            raise SpecError(
                "duplicate loopback port",
                entry=f"worker:{worker.worker_id}",
                field="port",
                value=worker.port,
            )
        check_port_bindable(worker.host, worker.port)
        seen.add(worker.port)


def cmd_check_ports(args: argparse.Namespace) -> int:
    spec = load_spec(args.requests, args.workers, args.budget)
    _check_all_ports(spec)
    _emit(
        {
            "code": "OK",
            "message": "all loopback ports are free and distinct",
            "ports": [[worker.host, worker.port] for worker in spec.workers],
        }
    )
    return 0


def _run_engine(spec, ledger, *, only_batch_ids=None, invalidation=None) -> dict:
    engine = DiagnosticEngine(spec, ledger)
    engine.start_workers()
    try:
        summary = engine.run(only_batch_ids=only_batch_ids, invalidation=invalidation)
    finally:
        engine.stop_workers()
    return {
        "processed_batches": list(summary.processed_batches),
        "skipped_batches": list(summary.skipped_batches),
        "peak_global": summary.peak_global,
        "peak_worker": summary.peak_worker,
        "total_budget_ms": summary.total_budget_ms,
    }


def cmd_run(args: argparse.Namespace) -> int:
    logger = _default_logger(args)
    spec = load_spec(args.requests, args.workers, args.budget)
    _check_all_ports(spec)
    ledger = SnapshotLedger.create(args.ledger, spec)
    result = _run_engine(spec, ledger)
    ledger.close()
    logger.log("run_completed", code="OK", **result)
    _emit({"code": "OK", **result, "ledger": os.path.abspath(args.ledger)})
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    spec = load_spec(args.requests, args.workers, args.budget)
    plan = build_plan(spec)
    _emit(
        {
            "code": "OK",
            "remainder_rule": plan.remainder_rule,
            "batches": [batch.to_dict() for batch in plan.batches],
        }
    )
    return 0


def cmd_rerun(args: argparse.Namespace) -> int:
    logger = _default_logger(args)
    old_spec = load_spec(args.requests, args.workers, args.budget)
    new_workers_raw = load_json_file(args.new_workers)
    from .spec import _validate_workers

    new_workers = _validate_workers(new_workers_raw, old_spec.input_dim())
    new_spec = replace(old_spec, workers=new_workers)

    plan = build_plan(new_spec)
    report = propagate(old_spec, new_spec, plan)
    invalid_set = set(report.invalid_batch_ids)

    ledger = SnapshotLedger.open(args.ledger, old_spec, on_mismatch="replace")
    backup = getattr(ledger, "backup_path", None)
    before = {batch_id: ledger.load_batch(batch_id) for batch_id in ledger.committed_batch_ids()}

    if invalid_set:
        result = _run_engine(new_spec, ledger, only_batch_ids=invalid_set, invalidation=report)
    else:
        result = {"processed_batches": [], "skipped_batches": list(ledger.committed_batch_ids())}
    ledger.update_header(new_spec)

    preserved = {}
    for batch_id, stored_before in before.items():
        if batch_id not in invalid_set:
            stored_after = ledger.load_batch(batch_id)
            preserved[batch_id] = stored_before == stored_after
    ledger.close()
    logger.log(
        "rerun_completed",
        code="OK",
        invalid_batches=list(invalid_set),
        preserved_identical=preserved,
        steps=len(report.steps),
    )
    _emit(
        {
            "code": "OK",
            "invalidation": report.to_dict(),
            "rerun": result,
            "preserved_identical": preserved,
            "readonly_backup": backup,
        }
    )
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    logger = _default_logger(args)
    spec = load_spec(args.requests, args.workers, args.budget)
    ledger = SnapshotLedger.open(args.ledger, spec, on_mismatch="raise")
    unknown = ledger.unknown_batches(build_plan(spec))
    report = replay_ledger(spec, ledger)
    ledger.close()
    logger.log("replay_completed", code="OK", unknown_batches=list(unknown), **report)
    _emit({"code": "OK", **report, "unknown_batches": list(unknown)})
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    logger = _default_logger(args)
    spec = load_spec(args.requests, args.workers, args.budget)
    ledger = SnapshotLedger.open(args.ledger, spec, on_mismatch="raise")
    report = verify_ledger(args.requests, args.workers, args.budget, ledger)
    ledger.close()
    logger.log("verify_completed", code="OK", **report)
    _emit({"code": "OK", **report})
    return 0


def cmd_inject(args: argparse.Namespace) -> int:
    logger = _default_logger(args)
    kind = args.kind
    try:
        if kind == "request_corrupt":
            raw = load_json_file(args.requests)
            if raw:
                raw[0]["arrival_seq"] = -1
            tmp = args.requests + ".corrupt.json"
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(raw, handle)
            try:
                load_spec(tmp, args.workers, args.budget)
            finally:
                os.remove(tmp)
        elif kind == "dangling_ref":
            spec = load_spec(args.requests, args.workers, args.budget)
            ledger = SnapshotLedger.open(args.ledger, spec, on_mismatch="raise")
            row = ledger.conn.execute(
                "SELECT batch_id, normalized_json FROM receipts ORDER BY batch_id, request_id LIMIT 1"
            ).fetchone()
            if row is None:
                raise DiagnosticError("ledger has no receipts to tamper")
            batch_id, normalized_json = row
            receipt = json.loads(normalized_json)
            receipt["duplicate_ref"] = "req-does-not-exist"
            ledger.conn.execute(
                "UPDATE receipts SET normalized_json = ? WHERE batch_id = ? AND request_id = ?",
                (json.dumps(receipt, sort_keys=True), batch_id, receipt["request_id"]),
            )
            ledger.conn.commit()
            replay_ledger(spec, ledger)
            ledger.close()
        elif kind == "snapshot_digest":
            spec = load_spec(args.requests, args.workers, args.budget)
            ledger = SnapshotLedger.open(args.ledger, spec)
            ledger.tamper_header_digest()
            ledger.close()
            SnapshotLedger.open(args.ledger, spec, on_mismatch="raise")
        else:  # pragma: no cover - argparse choices guard
            raise DiagnosticError(f"unknown injection {kind}")
    except DiagnosticError as exc:
        record = logger.log("injection_detected", **exc.to_dict())
        _emit(record)
        return 1
    raise RuntimeError("injection failed to trigger the expected error")


def cmd_dashboard(args: argparse.Namespace) -> int:
    from .gui import launch_gui, render_overview_json

    spec = load_spec(args.requests, args.workers, args.budget)
    if args.text:
        print(render_overview_json(spec))
        return 0
    return launch_gui(
        {"requests": args.requests, "workers": args.workers, "budget": args.budget}
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="batch-diagnostic",
        description="Local inference batch attribution diagnostic component",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(sub_parser: argparse.ArgumentParser) -> None:
        sub_parser.add_argument("--requests", required=True)
        sub_parser.add_argument("--workers", required=True)
        sub_parser.add_argument("--budget", required=True)
        sub_parser.add_argument("--ledger", required=True)
        sub_parser.add_argument("--log-path")

    check_ports = sub.add_parser("check-ports", help="verify loopback ports are bindable")
    add_common(check_ports)
    check_ports.set_defaults(func=cmd_check_ports)

    run = sub.add_parser("run", help="run the full deterministic pipeline")
    add_common(run)
    run.set_defaults(func=cmd_run)

    plan = sub.add_parser("plan", help="print the deterministic batch plan")
    plan.add_argument("--requests", required=True)
    plan.add_argument("--workers", required=True)
    plan.add_argument("--budget", required=True)
    plan.set_defaults(func=cmd_plan)

    rerun = sub.add_parser("rerun", help="invalidating rerun against a new worker spec")
    add_common(rerun)
    rerun.add_argument("--new-workers", required=True)
    rerun.set_defaults(func=cmd_rerun)

    replay = sub.add_parser("replay", help="replay committed batches and compare")
    add_common(replay)
    replay.set_defaults(func=cmd_replay)

    verify = sub.add_parser("verify", help="independent ledger verification")
    add_common(verify)
    verify.set_defaults(func=cmd_verify)

    inject = sub.add_parser("inject", help="inject one of the three fault classes")
    add_common(inject)
    inject.add_argument(
        "--kind",
        required=True,
        choices=("request_corrupt", "dangling_ref", "snapshot_digest"),
    )
    inject.set_defaults(func=cmd_inject)

    dashboard = sub.add_parser("dashboard", help="Tkinter diagnostic dashboard")
    dashboard.add_argument("--requests", required=True)
    dashboard.add_argument("--workers", required=True)
    dashboard.add_argument("--budget", required=True)
    dashboard.add_argument("--ledger", default=":memory:")
    dashboard.add_argument("--log-path")
    dashboard.add_argument("--text", action="store_true", help="headless JSON rendering")
    dashboard.set_defaults(func=cmd_dashboard)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except DiagnosticError as exc:
        payload = exc.to_dict()
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
        return 1


if __name__ == "__main__":
    sys.exit(main())
