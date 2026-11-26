from __future__ import annotations

import argparse
import json
import platform
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .canonical import canonical_json
from .determinism import compute_shard_summary
from .diagnostics import verify_ledger_rebuild
from .errors import DiagnosticError
from .invalidation import plan_change
from .ledger import Ledger
from .node_protocol import NodeCluster
from .scheduler import DispatchScheduler
from .spec import check_declared_ports, is_port_available, load_spec
from .topology import stable_topological_order


def _print(value: Any) -> None:
    print(canonical_json(value) if not isinstance(value, str) else value)


def check_env(spec_path: str | None) -> int:
    report: dict[str, Any] = {
        "linux_wsl": platform.system() == "Linux",
        "python": platform.python_version(),
        "python_supported": sys.version_info >= (3, 11),
        "numpy": np.__version__,
        "sqlite": sqlite3.sqlite_version,
        "external_network_required": False,
    }
    if spec_path:
        spec = load_spec(spec_path)
        report["ports"] = {
            f"{node.host}:{node.port}": is_port_available(node.host, node.port)
            for node in spec.nodes
        }
        report["all_ports_available"] = all(report["ports"].values())
    _print(report)
    return 0 if report["python_supported"] else 1


def validate(spec_path: str) -> int:
    spec = load_spec(spec_path)
    check_declared_ports(spec)
    _print(
        {
            "ok": True,
            "session_id": spec.session_id,
            "topological_order": stable_topological_order(spec.shards),
            "nodes": len(spec.nodes),
            "shards": len(spec.shards),
            "delay_batches": len(spec.delay_batches),
        }
    )
    return 0


def materialize_samples(spec_path: str, out_path: str) -> int:
    spec = load_spec(spec_path)
    output = {
        "session_id": spec.session_id,
        "summaries": [
            compute_shard_summary(spec.node(shard.node_id), shard).to_dict()
            for shard in spec.shards
        ],
    }
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print({"ok": True, "output": str(path), "shards": len(spec.shards)})
    return 0


def run_dispatch(spec_path: str, db_path: str, reset: bool, report_path: str | None) -> int:
    spec = load_spec(spec_path)
    check_declared_ports(spec)
    db = Path(db_path)
    if reset and db.exists():
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(db) + suffix)
            if candidate.exists():
                candidate.unlink()

    ledger = Ledger(db, spec)
    cluster: NodeCluster | None = None
    try:
        orphaned = ledger.recover_orphaned_occupancy()
        cluster = NodeCluster(spec.nodes).start()
        scheduler = DispatchScheduler(spec, clients=cluster.clients, ledger=ledger)
        scheduler.preload_committed()
        missing = set(stable_topological_order(spec.shards)) - ledger.committed_ids()
        scheduler.reset_for_rerun(missing)
        run_summary = scheduler.run(target_ids=set(scheduler.states))
        rebuild = verify_ledger_rebuild(spec, ledger)
        report = {
            "ok": True,
            "spec": str(spec_path),
            "database": str(db),
            "recovered_orphaned_occupancy": len(orphaned),
            "rerun_shards": sorted(missing),
            "committed_shards": sorted(ledger.committed_ids()),
            "replay_verified_shards": len(rebuild),
            "cursor": ledger.latest_cursor(),
            "run": run_summary,
        }
        if report_path:
            path = Path(report_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _print({key: value for key, value in report.items() if key != "run"})
        return 0
    finally:
        if cluster is not None:
            cluster.stop()
        ledger.close()


def invalidation_plan(spec_path: str, change_json: str) -> int:
    spec = load_spec(spec_path)
    change = json.loads(change_json)
    _print(plan_change(spec, change).to_dict())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shard-diag")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check-env")
    check.add_argument("--spec")

    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("--spec", default="config/spec.json")

    samples = sub.add_parser("materialize-samples")
    samples.add_argument("--spec", default="config/spec.json")
    samples.add_argument("--out", default="samples/expected_summaries.json")

    run = sub.add_parser("run")
    run.add_argument("--spec", default="config/spec.json")
    run.add_argument("--db", default="artifacts/demo-ledger.sqlite3")
    run.add_argument("--reset", action="store_true")
    run.add_argument("--report", default="artifacts/run-report.json")

    plan = sub.add_parser("invalidation-plan")
    plan.add_argument("--spec", default="config/spec.json")
    plan.add_argument("--change", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "check-env":
            return check_env(args.spec)
        if args.command == "validate":
            return validate(args.spec)
        if args.command == "materialize-samples":
            return materialize_samples(args.spec, args.out)
        if args.command == "run":
            return run_dispatch(args.spec, args.db, args.reset, args.report)
        if args.command == "invalidation-plan":
            return invalidation_plan(args.spec, args.change)
        return 2
    except DiagnosticError as exc:
        _print({"ok": False, **exc.to_dict()})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
