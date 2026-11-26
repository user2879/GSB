"""Command line entry point for local shard diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .canonical import atomic_write_json, canonical_digest
from .cluster import NodeCluster, check_declared_ports
from .engine import Engine, recompute_invalid_subset_local
from .errors import DiagnosticError
from .fixedpoint import quantize_shard_demand
from .invalidation import (
    changed_nodes_between_versions,
    derive_invalidation,
    hash_records,
    verify_byte_unchanged,
)
from .ledger import Ledger, replace_sqlite_with_snapshot, verify_snapshot_file
from .spec import load_spec
from .scenarios import run_and_write_demo


EXIT_CODES = {
    "E_SPEC": 2,
    "E_PORT": 3,
    "E_TIMEOUT": 4,
    "E_PROTO": 5,
    "E_LEASE": 6,
    "E_LEDGER": 7,
}


def print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def cmd_validate(args: argparse.Namespace) -> dict[str, Any]:
    spec = load_spec(args.spec)
    result = {
        "ok": True,
        "spec": str(args.spec),
        "spec_digest": spec.spec_digest,
        "node_count": len(spec.nodes),
        "shard_count": len(spec.shards),
        "topological_order": spec.topo_order,
        "capacity_versions": spec.version_chain(),
        "event_count": len(spec.event_list),
    }
    print_json(result)
    return result


def cmd_ports(args: argparse.Namespace) -> dict[str, Any]:
    spec = load_spec(args.spec)
    results = check_declared_ports(spec)
    payload = {"ok": all(item["available_before_start"] for item in results), "ports": results}
    print_json(payload)
    return payload


def _unlink_sidecars(path: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        target = Path(str(path) + suffix)
        if target.exists():
            target.unlink()


def cmd_run(args: argparse.Namespace) -> dict[str, Any]:
    spec = load_spec(args.spec)
    db_path = Path(args.db)
    if args.fresh:
        _unlink_sidecars(db_path)
    ledger = Ledger.open(db_path, expected_spec_digest=spec.spec_digest)
    cluster = NodeCluster(spec, log_dir=args.log_dir)
    try:
        cluster.start()
        engine = Engine(
            spec,
            ledger,
            cluster,
            batch_size=args.batch_size,
            kill_after_terminal_commits=args.kill_after_terminal,
        )
        report = engine.run()
        if args.report:
            atomic_write_json(args.report, report)
        if args.basis_out:
            basis = {
                shard_id: {"basis": state.get("basis"), "summary": state.get("summary")}
                for shard_id, state in engine.states.items()
            }
            atomic_write_json(args.basis_out, basis)
        if args.snapshot:
            ledger.export_snapshot(args.snapshot)
        print_json(
            {
                "ok": report["ok"],
                "report": args.report,
                "basis_out": args.basis_out,
                "snapshot": args.snapshot,
                "ledger_summary_digest": report["ledger_summary"]["ledger_summary_digest"],
                "ledger_batch_count": report["ledger_batch_count"],
            }
        )
        return report
    finally:
        cluster.stop()
        ledger.close()


def cmd_explain(args: argparse.Namespace) -> dict[str, Any]:
    spec = load_spec(args.spec)
    if args.shard not in spec.shards:
        raise DiagnosticError("E_SPEC", f"unknown shard {args.shard}", field="shard", location=args.shard)
    records = recompute_invalid_subset_local(
        spec, [args.shard], target_capacity_version=args.capacity_version
    )
    payload = records[args.shard]
    if args.out:
        atomic_write_json(args.out, payload)
    print_json({"ok": True, "shard_id": args.shard, "summary_digest": payload["summary"]["summary_digest"], "out": args.out})
    return payload


def cmd_invalidate(args: argparse.Namespace) -> dict[str, Any]:
    spec = load_spec(args.spec)
    old_version = args.from_version or spec.initial_capacity_version
    new_version = args.to_version or spec.current_capacity_version
    changed_nodes = changed_nodes_between_versions(spec, old_version, new_version)
    propagation = derive_invalidation(
        spec,
        changed_node_ids=changed_nodes,
        reason=f"capacity_{old_version}_to_{new_version}",
    )
    before_records = recompute_invalid_subset_local(spec, list(spec.shards), target_capacity_version=old_version)
    before_hashes = hash_records(before_records)
    after_records = dict(before_records)
    recomputed = recompute_invalid_subset_local(
        spec, propagation["invalid_shard_ids"], target_capacity_version=new_version
    )
    after_records.update(recomputed)
    after_hashes = hash_records(after_records)
    verification = verify_byte_unchanged(
        before_hashes, after_hashes, allowed_changed=propagation["invalid_shard_ids"]
    )
    old_capacity = spec.capacity_versions[old_version]
    new_capacity = spec.capacity_versions[new_version]
    water_before = {
        shard_id: quantize_shard_demand(
            shard_id=shard_id,
            resource_units=int(shard["resource_units"]),
            budget_units=int(old_capacity["global_budget_units"]),
            scale_power=int(old_capacity["session_scale"]),
        )
        for shard_id, shard in spec.shards.items()
    }
    water_after = dict(water_before)
    for shard_id in propagation["invalid_shard_ids"]:
        shard = spec.shards[shard_id]
        water_after[shard_id] = quantize_shard_demand(
            shard_id=shard_id,
            resource_units=int(shard["resource_units"]),
            budget_units=int(new_capacity["global_budget_units"]),
            scale_power=int(new_capacity["session_scale"]),
        )
    water_verification = verify_byte_unchanged(
        hash_records(water_before),
        hash_records(water_after),
        allowed_changed=propagation["invalid_shard_ids"],
    )
    payload = {
        "ok": verification["ok"] and water_verification["ok"],
        "old_version": old_version,
        "new_version": new_version,
        "changed_node_ids": changed_nodes,
        "propagation": propagation,
        "verification": verification,
        "water_verification": water_verification,
        "lease_attribution_policy": (
            "Lease lifecycle records are append-only; capacity advance appends revocation and "
            "preserves lease_id/shard_id/node_id ownership identity. See evolution.json."
        ),
        "recomputed_digests": {
            shard_id: recomputed[shard_id]["summary"]["summary_digest"]
            for shard_id in propagation["invalid_shard_ids"]
        },
        "unaffected_digests": {
            shard_id: before_records[shard_id]["summary"]["summary_digest"]
            for shard_id in propagation["unaffected_shard_ids"]
        },
    }
    if args.out:
        atomic_write_json(args.out, payload)
    print_json({"ok": payload["ok"], "invalid": propagation["invalid_shard_ids"], "verification": verification})
    if not payload["ok"]:
        raise DiagnosticError("E_PROTO", "unexpected unchanged-record verification failure")
    return payload


def cmd_gui(args: argparse.Namespace) -> int:
    from .gui import launch  # imported lazily so headless tests do not require Tk

    return launch(args.spec, args.report)


def cmd_evolve_demo(args: argparse.Namespace) -> dict[str, Any]:
    spec = load_spec(args.spec)
    result = run_and_write_demo(
        spec,
        db_path=args.db,
        out_path=args.out,
        log_dir=args.log_dir,
        fresh=args.fresh,
    )
    print_json(
        {
            "ok": result["ok"],
            "out": args.out,
            "invalid": result["invalidation"]["invalid_shard_ids"],
            "final_active_used_units": result["final_active_used_units"],
            "cross_generation_code": (result["cross_generation_renewal_error"] or {}).get("code"),
        }
    )
    if not result["ok"]:
        raise DiagnosticError("E_LEASE", "evolution demo assertions failed")
    return result


def cmd_snapshot_verify(args: argparse.Namespace) -> dict[str, Any]:
    snapshot = verify_snapshot_file(args.snapshot)
    result = {
        "ok": True,
        "snapshot": args.snapshot,
        "format_version": snapshot["format_version"],
        "content_digest": snapshot["content_digest"],
        "batch_count": snapshot.get("batch_count"),
    }
    print_json(result)
    return result


def cmd_snapshot_restore(args: argparse.Namespace) -> dict[str, Any]:
    snapshot = verify_snapshot_file(args.snapshot)
    ledger = replace_sqlite_with_snapshot(args.db, snapshot)
    digest = ledger.canonical_accounting_summary()["ledger_summary_digest"]
    count = ledger.batch_count()
    ledger.close()
    result = {
        "ok": True,
        "db": args.db,
        "restored_from": args.snapshot,
        "replacement_policy": "whole_database_replacement_not_incremental_merge",
        "ledger_batch_count": count,
        "ledger_summary_digest": digest,
    }
    print_json(result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sharddiag", description="Local shard dispatch diagnostics")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate")
    validate.add_argument("--spec", required=True)
    validate.set_defaults(func=cmd_validate)

    ports = sub.add_parser("ports")
    ports.add_argument("--spec", required=True)
    ports.set_defaults(func=cmd_ports)

    run = sub.add_parser("run")
    run.add_argument("--spec", required=True)
    run.add_argument("--db", required=True)
    run.add_argument("--report")
    run.add_argument("--basis-out")
    run.add_argument("--snapshot")
    run.add_argument("--log-dir", required=True)
    run.add_argument("--batch-size", type=int)
    run.add_argument("--fresh", action="store_true")
    run.add_argument("--kill-after-terminal", type=int)
    run.set_defaults(func=cmd_run)

    explain = sub.add_parser("explain")
    explain.add_argument("--spec", required=True)
    explain.add_argument("--shard", required=True)
    explain.add_argument("--capacity-version", type=int)
    explain.add_argument("--out")
    explain.set_defaults(func=cmd_explain)

    invalidate = sub.add_parser("invalidate")
    invalidate.add_argument("--spec", required=True)
    invalidate.add_argument("--from-version", type=int)
    invalidate.add_argument("--to-version", type=int)
    invalidate.add_argument("--out")
    invalidate.set_defaults(func=cmd_invalidate)

    gui = sub.add_parser("gui")
    gui.add_argument("--spec")
    gui.add_argument("--report")
    gui.set_defaults(func=cmd_gui)

    evolve = sub.add_parser("evolve-demo")
    evolve.add_argument("--spec", required=True)
    evolve.add_argument("--db", required=True)
    evolve.add_argument("--out", required=True)
    evolve.add_argument("--log-dir", required=True)
    evolve.add_argument("--fresh", action="store_true")
    evolve.set_defaults(func=cmd_evolve_demo)

    snapshot_verify = sub.add_parser("snapshot-verify")
    snapshot_verify.add_argument("--snapshot", required=True)
    snapshot_verify.set_defaults(func=cmd_snapshot_verify)

    snapshot_restore = sub.add_parser("snapshot-restore")
    snapshot_restore.add_argument("--snapshot", required=True)
    snapshot_restore.add_argument("--db", required=True)
    snapshot_restore.set_defaults(func=cmd_snapshot_restore)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.func(args)
        if isinstance(result, dict) and result.get("ok") is False:
            return 10
        return 0
    except DiagnosticError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False, indent=2), file=sys.stderr)
        return EXIT_CODES.get(exc.code, 1)
    except KeyboardInterrupt:
        print(json.dumps({"ok": False, "code": "E_INTERRUPTED"}), file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
