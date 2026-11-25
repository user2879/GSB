"""Command-line entry point for the gray-route receipt gateway."""
from __future__ import annotations

import argparse
import json
import os
import sys

from .backend_proc import BackendPool
from .batching import plan_batches
from .engine import GatewayEngine
from .ledger import Ledger
from .recovery import classify_batches, replay_unfinished
from .sharding import attribute_all
from .snapshot import build_snapshot, restore_or_replace
from .spec import load_spec
from .viewmodel import DashboardModel


def _load(args):
    request_path = args.requests or os.path.join(args.data_dir, "requests.ndjson")
    return load_spec(args.config_dir, request_path)


def cmd_validate(args) -> int:
    spec = _load(args)
    print(json.dumps({
        "ok": True,
        "backends": len(spec.backends),
        "rules": len(spec.rules),
        "requests": len(spec.requests),
        "spec_summary": spec.spec_summary(),
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_plan(args) -> int:
    spec = _load(args)
    hits = attribute_all(spec)
    batches = plan_batches(spec, hits)
    print(json.dumps({
        "hits": {rid: hit.to_dict() for rid, hit in sorted(hits.items())},
        "batches": [b.to_dict() for b in batches],
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_run(args) -> int:
    spec = _load(args)
    hits = attribute_all(spec)
    batches = plan_batches(spec, hits)
    invalid_versions = _load_invalid_versions(args)
    ledger_path = os.path.join(args.data_dir, "ledger.sqlite")
    with Ledger(ledger_path) as ledger, BackendPool(spec.backends) as pool:
        engine = GatewayEngine(
            spec, ledger, pool,
            max_global=args.max_global,
            per_worker_quota=args.worker_quota,
            emit_format=args.emit_format)
        result = engine.run(batches, hits, invalid_versions)
        snapshot = build_snapshot(
            rule_set_version=args.rule_set_version,
            worker_set_version=args.worker_set_version,
            param_version_seq=max(h.param_version_seq for h in hits.values()),
            ledger_summary=result.ledger_summary,
            cursor=ledger.cursor() or {},
        )
        snap_path = os.path.join(args.data_dir, "session.snapshot.json")
        _, action = restore_or_replace(snap_path, snapshot)
        model = DashboardModel(spec, hits, batches, result, invalid_versions)
        consistency = model.consistency_check(ledger, args.budget_ms)
    print(json.dumps({
        "committed_batches": result.committed_batches,
        "invalidated_requests": sorted(result.invalidated_requests),
        "ledger_summary": result.ledger_summary,
        "snapshot_action": action,
        "consistency": consistency,
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_recover(args) -> int:
    spec = _load(args)
    hits = attribute_all(spec)
    batches = plan_batches(spec, hits)
    ledger_path = os.path.join(args.data_dir, "ledger.sqlite")
    with Ledger(ledger_path) as ledger:
        committed, unknown = classify_batches(ledger, batches)
    print(json.dumps({
        "committed": [b.batch_id for b in committed],
        "unknown_requires_rerun": [b.batch_id for b in unknown],
        "exit_report_generated": False,
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_replay(args) -> int:
    spec = _load(args)
    hits = attribute_all(spec)
    batches = plan_batches(spec, hits)
    receipts_by_batch = _load_replay_receipts(args.receipts)
    ledger_path = os.path.join(args.data_dir, "ledger.sqlite")
    with Ledger(ledger_path) as ledger:
        report = replay_unfinished(spec, ledger, batches, receipts_by_batch)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _load_invalid_versions(args) -> set[tuple[str, int]]:
    if not args.invalidate:
        return set()
    with open(args.invalidate, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {(row["worker_id"], int(row["param_version_seq"])) for row in raw}


def _load_replay_receipts(path: str) -> dict[str, list[dict]]:
    if not path or not os.path.exists(path):
        return {}
    grouped: dict[str, list[dict]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                grouped.setdefault(row["batch_id"], []).append(row)
    return grouped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="grayroute")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--requests", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("validate").set_defaults(func=cmd_validate)
    sub.add_parser("plan").set_defaults(func=cmd_plan)

    run = sub.add_parser("run")
    run.add_argument("--max-global", type=int, default=4)
    run.add_argument("--worker-quota", type=int, default=2)
    run.add_argument("--emit-format", type=int, default=3, choices=(1, 2, 3))
    run.add_argument("--budget-ms", type=int, default=5000)
    run.add_argument("--rule-set-version", type=int, default=1)
    run.add_argument("--worker-set-version", type=int, default=1)
    run.add_argument("--invalidate", default=None)
    run.set_defaults(func=cmd_run)

    sub.add_parser("recover").set_defaults(func=cmd_recover)
    replay = sub.add_parser("replay")
    replay.add_argument("--receipts", default=None)
    replay.set_defaults(func=cmd_replay)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
