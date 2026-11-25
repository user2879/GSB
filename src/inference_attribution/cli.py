"""Command line interface for validation, planning, execution, and recovery."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from typing import Any

from .engine import execute_plan
from .errors import DiagnosticError
from .invalidation import diagnose_invalidation
from .jsonio import atomic_write_json, load_json_file
from .ledger import Ledger
from .planner import build_plan
from .receipts import normalize_receipts, read_receipt_file
from .snapshot import create_snapshot, restore_snapshot
from .spec import load_merge_rules, load_requests, load_workers


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _load_world(args: argparse.Namespace):
    workers = load_workers(args.workers)
    requests = load_requests(args.requests, workers)
    rules = load_merge_rules(args.rules)
    plan = build_plan(workers, requests)
    return workers, requests, rules, plan


def command_validate(args: argparse.Namespace) -> int:
    workers, requests, rules, plan = _load_world(args)
    _print(
        {
            "ok": True,
            "workers": len(workers.workers),
            "requests": len(requests.requests),
            "global_concurrency": rules.global_concurrency,
            "batches": len(plan.batches),
            "plan_digest": plan.content_digest,
        }
    )
    return 0


def command_plan(args: argparse.Namespace) -> int:
    _, _, _, plan = _load_world(args)
    atomic_write_json(args.output, plan.to_dict())
    _print(
        {
            "ok": True,
            "output": str(args.output),
            "content_digest": plan.content_digest,
            "batches": [
                {
                    "batch_id": batch.batch_id,
                    "worker_id": batch.worker_id,
                    "batch_seq": batch.batch_seq,
                    "request_ids": list(batch.request_ids),
                    "param_version_seq": batch.param_version_seq,
                    "param_digest": batch.param_digest,
                }
                for batch in plan.batches
            ],
        }
    )
    return 0


def command_normalize(args: argparse.Namespace) -> int:
    workers, requests, rules, plan = _load_world(args)
    raw = {}
    for format_version in ("r1", "r2", "r3"):
        path = getattr(args, format_version)
        if path is not None:
            raw[format_version] = read_receipt_file(path, format_version)
    ledger = normalize_receipts(
        raw,
        request_set=requests,
        worker_set=workers,
        plan=plan,
        rules=rules,
    )
    atomic_write_json(args.output, ledger.to_dict())
    _print(
        {
            "ok": True,
            "output": str(args.output),
            "batches": len(ledger.batches),
            "content_digest": ledger.content_digest,
            "source_formats": sorted(raw),
            "filled_fields": sorted(
                {
                    field
                    for batch in ledger.batches
                    for receipt in batch["receipts"]
                    for field in receipt["filled_fields"]
                }
            ),
        }
    )
    return 0


def command_invalidate(args: argparse.Namespace) -> int:
    old_workers = load_workers(args.old_workers)
    old_requests = load_requests(args.old_requests, old_workers)
    new_workers = load_workers(args.new_workers)
    new_requests = load_requests(args.new_requests, new_workers)
    rules = load_merge_rules(args.rules)
    old_plan = build_plan(old_workers, old_requests)
    new_plan = build_plan(new_workers, new_requests)
    report = diagnose_invalidation(
        old_workers, old_plan, new_workers, new_plan
    )
    output = report.to_dict()
    output["old_plan_digest"] = old_plan.content_digest
    output["new_plan_digest"] = new_plan.content_digest
    atomic_write_json(args.output, output)
    _print(output)
    return 0


def command_run(args: argparse.Namespace) -> int:
    workers, requests, rules, plan = _load_world(args)
    with Ledger(args.ledger) as ledger:
        outcome = execute_plan(
            plan=plan,
            request_set=requests,
            worker_set=workers,
            rules=rules,
            ledger=ledger,
            crash_after_commits=args.crash_after,
        )
        result = outcome.to_dict()
        result["recovery"] = ledger.recovery_report().to_dict()
    _print({"ok": True, **result})
    return 0


def command_recover(args: argparse.Namespace) -> int:
    with Ledger(args.ledger) as ledger:
        _print(
            {
                "ok": True,
                "ledger": str(args.ledger),
                "format_version": ledger.format_version,
                "content_digest": ledger.content_digest,
                "recovery": ledger.recovery_report().to_dict(),
            }
        )
    return 0


def command_snapshot(args: argparse.Namespace) -> int:
    workers = load_workers(args.workers)
    with Ledger(args.ledger) as ledger:
        snapshot = create_snapshot(workers, ledger, args.output)
    _print(
        {
            "ok": True,
            "output": str(args.output),
            "worker_set_version": snapshot["worker_set_version"],
            "ledger_digest": snapshot["ledger_digest"],
            "cursor": snapshot["cursor"],
            "content_digest": snapshot["content_digest"],
        }
    )
    return 0


def command_restore(args: argparse.Namespace) -> int:
    workers = load_workers(args.workers)
    outcome = restore_snapshot(args.snapshot, args.ledger, workers)
    _print({"ok": True, **outcome.to_dict()})
    return 0


def command_compare(args: argparse.Namespace) -> int:
    from .invalidation import assert_replay_consistent

    with Ledger(args.expected_ledger) as expected:
        expected_rows = expected.rows()
    with Ledger(args.actual_ledger) as actual:
        actual_rows = actual.rows()

    assert_replay_consistent(None, expected_rows, actual_rows)
    _print(
        {
            "ok": True,
            "batches": len(expected_rows),
            "note": "worker-observed latency is treated as volatile and excluded",
        }
    )
    return 0


def _add_world_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--requests", required=True)
    parser.add_argument("--workers", required=True)
    parser.add_argument("--rules", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inference-attribution")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate")
    _add_world_arguments(validate)
    validate.set_defaults(func=command_validate)

    plan = subparsers.add_parser("plan")
    _add_world_arguments(plan)
    plan.add_argument("--output", required=True)
    plan.set_defaults(func=command_plan)

    normalize = subparsers.add_parser("normalize")
    _add_world_arguments(normalize)
    normalize.add_argument("--r1")
    normalize.add_argument("--r2")
    normalize.add_argument("--r3")
    normalize.add_argument("--output", required=True)
    normalize.set_defaults(func=command_normalize)

    invalidate = subparsers.add_parser("invalidate")
    invalidate.add_argument("--old-requests", required=True)
    invalidate.add_argument("--old-workers", required=True)
    invalidate.add_argument("--new-requests", required=True)
    invalidate.add_argument("--new-workers", required=True)
    invalidate.add_argument("--rules", required=True)
    invalidate.add_argument("--output", required=True)
    invalidate.set_defaults(func=command_invalidate)

    run = subparsers.add_parser("run")
    _add_world_arguments(run)
    run.add_argument("--ledger", required=True)
    run.add_argument("--crash-after", type=int)
    run.set_defaults(func=command_run)

    recover = subparsers.add_parser("recover")
    recover.add_argument("--ledger", required=True)
    recover.set_defaults(func=command_recover)

    snapshot = subparsers.add_parser("snapshot")
    snapshot.add_argument("--workers", required=True)
    snapshot.add_argument("--ledger", required=True)
    snapshot.add_argument("--output", required=True)
    snapshot.set_defaults(func=command_snapshot)

    restore = subparsers.add_parser("restore")
    restore.add_argument("--workers", required=True)
    restore.add_argument("--snapshot", required=True)
    restore.add_argument("--ledger", required=True)
    restore.set_defaults(func=command_restore)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--expected-ledger", required=True)
    compare.add_argument("--actual-ledger", required=True)
    compare.set_defaults(func=command_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except DiagnosticError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2
    except sqlite3.Error as exc:
        error = DiagnosticError(
            code="E_LEDGER", message=str(exc), object=args.command, field="<ledger>"
        )
        print(json.dumps(error.to_dict(), ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError) as exc:
        error = DiagnosticError(
            code="E_SPEC", message=str(exc), object=args.command, field="<command>"
        )
        print(json.dumps(error.to_dict(), ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
