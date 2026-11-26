from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

from .errors import QuotaLabError
from .fixtures import write_demo_files, write_evolution_files
from .invalidation import compare_specs
from .ledger import Ledger, replace_json_snapshot_if_stale
from .processes import NodeProcess, bind_release_probe, hold_port
from .samples import load_event_stream
from .scheduler import Scheduler
from .spec import NodeSpec, parse_session, load_session_files
from .verify import byte_preservation_check, rebuild_consistency_check


def write_json(path: str | Path | None, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(text, end="")


def write_combined_spec(spec: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def cmd_init_demo(args: argparse.Namespace) -> int:
    paths = write_demo_files(args.root)
    write_json(args.result, {"ok": True, "files": paths})
    return 0


def cmd_make_evolution_spec(args: argparse.Namespace) -> int:
    paths = write_evolution_files(args.root)
    write_json(args.result, {"ok": True, "files": paths})
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    spec = load_session_files(args.declaration, args.shards, capacity_path=args.capacity)
    event_count = None
    if args.events:
        samples = load_event_stream(args.events, spec)
        event_count = sum(len(rows) for rows in samples.values())
    write_json(
        args.result,
        {
            "ok": True,
            "session_id": spec.session_id,
            "spec_digest": spec.digest(),
            "topological_order": list(spec.topo_order),
            "node_count": len(spec.nodes),
            "shard_count": len(spec.shards),
            "event_count": event_count,
            "q": spec.q,
            "budget_q": spec.budget_q,
        },
    )
    return 0


def _minimal_network_spec(path: Path, port: int) -> Any:
    declaration = {
        "schema_version": 1,
        "session_id": "network-check",
        "scale_power": 0,
        "global_budget_units": 1,
        "lease_ttl_ms": 1000,
        "lease_max_renewals": 0,
        "nodes": [
            {
                "node_id": "net",
                "host": "127.0.0.1",
                "port": port,
                "concurrency_quota": 1,
                "lease_limit": 1,
                "seed": 7,
                "capacity_version": 1,
            }
        ],
    }
    shards = {
        "shards": [
            {
                "shard_id": "timeout-shard",
                "node_id": "net",
                "depends_on": [],
                "sample_rows": 4,
                "resource_units": 1,
                "priority": 0,
            }
        ]
    }
    spec = parse_session(declaration, shards)
    write_combined_spec(spec, path)
    return spec


def cmd_verify_network(args: argparse.Namespace) -> int:
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    bind = bind_release_probe()

    conflict_sock, conflict_port = hold_port(0)
    conflict_spec_path = work / "conflict-spec.json"
    conflict_spec = _minimal_network_spec(conflict_spec_path, conflict_port)
    runner = NodeProcess(
        node=conflict_spec.node("net"),
        spec_path=conflict_spec_path,
        log_path=work / "conflict-node.log",
    )
    port_result: dict[str, Any] | None = None
    try:
        try:
            runner.start()
        except QuotaLabError as exc:
            port_result = exc.to_dict()["error"]
        if port_result is None:
            raise QuotaLabError("occupied port did not produce E_PORT", code="E_PORT")
    finally:
        runner.stop()
        conflict_sock.close()

    timeout_sock, timeout_port = hold_port(0)
    timeout_sock.close()
    timeout_spec_path = work / "timeout-spec.json"
    timeout_spec = _minimal_network_spec(timeout_spec_path, timeout_port)
    timeout_runner = NodeProcess(
        node=timeout_spec.node("net"),
        spec_path=timeout_spec_path,
        log_path=work / "timeout-node.log",
    )
    timeout_result: dict[str, Any] | None = None
    try:
        timeout_runner.start()
        try:
            timeout_runner.request(
                {
                    "op": "aggregate",
                    "node_id": "net",
                    "shard_id": "timeout-shard",
                    "capacity_version": 1,
                    "samples": [[index % 2, index + 1] for index in range(4)],
                    "fault_delay_ms": 500,
                },
                timeout_ms=100,
            )
        except QuotaLabError as exc:
            timeout_result = exc.to_dict()["error"]
        if timeout_result is None or timeout_result.get("code") != "E_TIMEOUT":
            raise QuotaLabError("slow node did not produce E_TIMEOUT", code="E_TIMEOUT")
    finally:
        timeout_runner.stop()

    result = {
        "ok": True,
        "bind_release": bind,
        "occupied_port": conflict_port,
        "port_conflict_error": port_result,
        "timeout_error": timeout_result,
    }
    write_json(args.result, result)
    return 0


def _open_ledger(args: argparse.Namespace, spec_digest: str) -> Ledger:
    ledger_path = Path(args.ledger)
    if getattr(args, "reset_ledger", False) and ledger_path.exists():
        for suffix in ("", "-wal", "-shm"):
            stale = Path(str(ledger_path) + suffix)
            if stale.exists():
                stale.unlink()
    return Ledger.open_for_session(
        ledger_path,
        spec_digest=spec_digest,
        replace_on_mismatch=not getattr(args, "no_replace", False),
    )


def cmd_run(args: argparse.Namespace) -> int:
    spec = load_session_files(args.declaration, args.shards, capacity_path=args.capacity)
    samples = load_event_stream(args.events, spec) if args.events else {
        shard.shard_id: __import__("quota_lab.samples", fromlist=["generate_samples_for_shard"]).generate_samples_for_shard(
            spec, shard.shard_id
        )
        for shard in spec.shards
    }
    ledger = _open_ledger(args, spec.digest()) if args.ledger else None
    recovered: list[dict[str, Any]] = []
    if ledger is not None and args.resume:
        recovered = ledger.recover_incomplete()

    combined_path = Path(args.combined_spec) if args.combined_spec else Path(args.ledger or "artifacts/run") .parent / "runtime-spec.json"
    write_combined_spec(spec, combined_path)
    log_dir = Path(args.log_dir)
    from .processes import Cluster

    try:
        with Cluster(spec, combined_path, log_dir) as cluster:
            scheduler = Scheduler(
                spec,
                samples,
                cluster,
                ledger=ledger,
                recovered=recovered,
                enable_kill_switch=not args.resume,
            )
            result = scheduler.run()
    finally:
        if ledger is not None:
            ledger.close()

    if args.snapshot:
        replace_json_snapshot_if_stale(
            args.snapshot,
            expected_version=1,
            spec_digest=spec.digest(),
            fresh_payload={
                "session_id": spec.session_id,
                "ok": result["ok"],
                "completed_shard_ids": result["completed_shard_ids"],
            },
        )
    write_json(args.result, result)
    return 0 if result["ok"] else 1


def cmd_rebuild_check(args: argparse.Namespace) -> int:
    spec = load_session_files(args.declaration, args.shards, capacity_path=args.capacity)
    samples = load_event_stream(args.events, spec)
    conn = sqlite3.connect(args.ledger)
    conn.row_factory = sqlite3.Row
    ledger = Ledger(args.ledger, conn)
    try:
        result = rebuild_consistency_check(spec, samples, ledger)
        write_json(args.result, result)
        return 0 if result["ok"] else 1
    finally:
        ledger.close()


def cmd_evolve(args: argparse.Namespace) -> int:
    old_spec = load_session_files(args.old_declaration, args.old_shards, capacity_path=args.old_capacity)
    new_spec = load_session_files(args.new_declaration, args.new_shards, capacity_path=args.new_capacity)
    samples = load_event_stream(args.events, new_spec)
    plan = compare_specs(old_spec, new_spec)

    old_ledger = Ledger.open_for_session(args.old_ledger, spec_digest=old_spec.digest())
    new_path = Path(args.new_ledger)
    if args.reset_new_ledger and new_path.exists():
        for suffix in ("", "-wal", "-shm"):
            stale = Path(str(new_path) + suffix)
            if stale.exists():
                stale.unlink()
    new_ledger = Ledger.open_for_session(new_path, spec_digest=new_spec.digest())
    copy_report = new_ledger.copy_completed_records_from(old_ledger, plan.unchanged_shard_ids)
    preservation = byte_preservation_check(args.old_ledger, new_ledger, plan.unchanged_shard_ids)

    combined_path = Path(args.combined_spec or new_path.parent / "evolved-runtime-spec.json")
    write_combined_spec(new_spec, combined_path)
    from .processes import Cluster

    try:
        with Cluster(new_spec, combined_path, Path(args.log_dir)) as cluster:
            scheduler = Scheduler(new_spec, samples, cluster, ledger=new_ledger)
            run_result = scheduler.run()
    finally:
        old_ledger.close()
        new_ledger.close()

    verify_conn = sqlite3.connect(args.new_ledger)
    verify_conn.row_factory = sqlite3.Row
    verify_ledger = Ledger(args.new_ledger, verify_conn)
    try:
        rebuild = rebuild_consistency_check(new_spec, samples, verify_ledger)
        post_preservation = byte_preservation_check(args.old_ledger, verify_ledger, plan.unchanged_shard_ids)
    finally:
        verify_ledger.close()

    result = {
        "ok": run_result["ok"] and preservation["ok"] and post_preservation["ok"] and rebuild["ok"],
        "invalidation": plan.to_dict(),
        "copy_report": copy_report,
        "byte_preservation_before_run": preservation,
        "byte_preservation_after_run": post_preservation,
        "rebuild_consistency": rebuild,
        "run": run_result,
    }
    write_json(args.result, result)
    return 0 if result["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quota_lab", description="Local quota watermark/lease propagation lab")
    sub = parser.add_subparsers(dest="command", required=True)

    init_parser = sub.add_parser("init-demo")
    init_parser.add_argument("--root", default=".")
    init_parser.add_argument("--result", default="artifacts/init-demo.json")
    init_parser.set_defaults(func=cmd_init_demo)

    evolution_fixture_parser = sub.add_parser("make-evolution-spec")
    evolution_fixture_parser.add_argument("--root", default=".")
    evolution_fixture_parser.add_argument("--result", default="artifacts/evolution-fixture.json")
    evolution_fixture_parser.set_defaults(func=cmd_make_evolution_spec)

    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("--declaration", required=True)
    validate_parser.add_argument("--shards", required=True)
    validate_parser.add_argument("--capacity")
    validate_parser.add_argument("--events")
    validate_parser.add_argument("--result")
    validate_parser.set_defaults(func=cmd_validate)

    network_parser = sub.add_parser("verify-network")
    network_parser.add_argument("--work-dir", default="artifacts/network-check")
    network_parser.add_argument("--result", default="artifacts/network-result.json")
    network_parser.set_defaults(func=cmd_verify_network)

    run_parser = sub.add_parser("run")
    run_parser.add_argument("--declaration", required=True)
    run_parser.add_argument("--shards", required=True)
    run_parser.add_argument("--capacity")
    run_parser.add_argument("--events")
    run_parser.add_argument("--ledger", required=True)
    run_parser.add_argument("--result", required=True)
    run_parser.add_argument("--snapshot")
    run_parser.add_argument("--combined-spec")
    run_parser.add_argument("--log-dir", default="artifacts/logs")
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument("--reset-ledger", action="store_true")
    run_parser.add_argument("--no-replace", action="store_true")
    run_parser.set_defaults(func=cmd_run)

    rebuild_parser = sub.add_parser("rebuild-check")
    rebuild_parser.add_argument("--declaration", required=True)
    rebuild_parser.add_argument("--shards", required=True)
    rebuild_parser.add_argument("--capacity")
    rebuild_parser.add_argument("--events", required=True)
    rebuild_parser.add_argument("--ledger", required=True)
    rebuild_parser.add_argument("--result")
    rebuild_parser.set_defaults(func=cmd_rebuild_check)

    evolve_parser = sub.add_parser("evolve")
    evolve_parser.add_argument("--old-declaration", required=True)
    evolve_parser.add_argument("--old-shards", required=True)
    evolve_parser.add_argument("--old-capacity")
    evolve_parser.add_argument("--new-declaration", required=True)
    evolve_parser.add_argument("--new-shards", required=True)
    evolve_parser.add_argument("--new-capacity")
    evolve_parser.add_argument("--events", required=True)
    evolve_parser.add_argument("--old-ledger", required=True)
    evolve_parser.add_argument("--new-ledger", required=True)
    evolve_parser.add_argument("--combined-spec")
    evolve_parser.add_argument("--log-dir", default="artifacts/logs-evolve")
    evolve_parser.add_argument("--result", required=True)
    evolve_parser.add_argument("--reset-new-ledger", action="store_true")
    evolve_parser.set_defaults(func=cmd_evolve)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except QuotaLabError as exc:
        write_json(None, exc.to_dict())
        return 2
    except Exception as exc:
        payload = {"ok": False, "error": {"code": "E_INTERNAL", "message": str(exc), "type": type(exc).__name__}}
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
