from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .demo import write_demo_spec, write_receipt_fixtures
from .errors import Codes, DiagnosticError
from .fixedpoint import attribute_decision, fixed_ratio
from .invalidation import compare_semantic_views, hash_file, propagate_invalidation, write_shard_record
from .ledger import Ledger
from .nodepool import REPO_ROOT, NodePool
from .receipts import adopt_receipts, normalize_receipt_file
from .scheduler import Scheduler
from .spec import load_spec_bundle, stable_topological_order
from .tcpio import renew_lease_request, request_response


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def normalize_demo_receipts(spec_dir: Path, receipt_dir: Path) -> dict[str, Any]:
    bundle = load_spec_bundle(spec_dir)
    all_receipts: list[dict[str, Any]] = []
    per_format: dict[str, int] = {}
    for fmt, name in [("d1", "receipts.d1"), ("d2", "receipts.d2.jsonl"), ("d3", "receipts.d3.jsonl")]:
        records = normalize_receipt_file(receipt_dir / name, fmt, bundle=bundle)
        per_format[fmt] = len(records)
        all_receipts.extend(records)
    adopted = adopt_receipts(all_receipts)
    return {
        "counts": per_format,
        "total": len(adopted),
        "adopted": [r for r in adopted if r["adopted"]],
        "records": adopted,
    }


def _run_all(spec_dir: Path, db_path: Path, record_dir: Path, *, ttl_ms: int = 30_000) -> tuple[dict[str, Any], Ledger, NodePool, Scheduler]:
    bundle = load_spec_bundle(spec_dir)
    ledger = Ledger(db_path)
    pool = NodePool(bundle, spec_dir / "logs")
    pool.start()
    scheduler = Scheduler(bundle, ledger, pool)
    report = scheduler.run(lease_ttl_ms=ttl_ms)
    view = ledger.semantic_view(bundle.active_capacity_version)
    for record in view.values():
        write_shard_record(record_dir, record)
    return report, ledger, pool, scheduler


def fixed_point_demo() -> dict[str, Any]:
    low_a = fixed_ratio(3, 6, 1)
    low_b = fixed_ratio(4, 6, 1)
    medium_a = fixed_ratio(3, 6, 10)
    medium_b = fixed_ratio(4, 6, 10)
    high_a = fixed_ratio(3, 6, 100)
    high_b = fixed_ratio(4, 6, 100)
    low_decision = attribute_decision(
        candidate_ids=["shard-0002", "shard-0003"],
        candidate_levels_q=[low_a["q"], low_b["q"]],
        slack_q=0,
        scale=1,
        capacity_changed_dimensions=1,
        dependency_propagation_hops=0,
        renewal_count=0,
    )
    high_decision = attribute_decision(
        candidate_ids=["shard-0002", "shard-0003"],
        candidate_levels_q=[high_a["q"], high_b["q"]],
        slack_q=100,
        scale=100,
        capacity_changed_dimensions=1,
        dependency_propagation_hops=0,
        renewal_count=0,
    )
    return {
        "comparison": {
            "scale_1": [low_a["q"], low_b["q"], low_decision],
            "scale_10": [medium_a["q"], medium_b["q"]],
            "scale_100": [high_a["q"], high_b["q"], high_decision],
        },
        "conclusion": "scale=1 quantizes 3/6 and 4/6 both to 0 and is level-close; scale=100 separates 50 vs 66 beyond bound 4 and, with slack_q=100, is decisive",
        "tie_break": low_decision["tie_break_rule"],
    }


def lease_demo(work: Path) -> dict[str, Any]:
    spec_dir = write_demo_spec(work / "lease_spec", allocate_ports=True)
    capacity_path = spec_dir / "capacity_versions.json"
    capacity_payload = json.loads(capacity_path.read_text(encoding="utf-8"))
    capacity_payload["versions"][0]["budget_units"] = 6
    capacity_path.write_text(json.dumps(capacity_payload, indent=2) + "\n", encoding="utf-8")
    bundle = load_spec_bundle(spec_dir)
    ledger = Ledger(work / "lease.db")
    pool = NodePool(bundle, spec_dir / "logs")
    pool.start()
    try:
        cap = bundle.active_capacity()
        ledger.initialize_capacity_slots(cap)
        grant = ledger.grant_lease(
            lease_id="lease-demo-renew",
            shard_id="shard-0001",
            node_id="node-a",
            capacity_version=1,
            resource_units=2,
            ttl_ms=1_000,
            budget_units=cap.budget_units,
            concurrency_quota=2,
            lease_limit=2,
        )
        scheduler = Scheduler(bundle, ledger, pool)
        node_a = bundle.nodes_by_id["node-a"]
        request_response(
            node_a.host,
            node_a.port,
            {"type": "lease_grant", "node_id": "node-a", "lease_id": grant["lease_id"], "lease_version": 1},
            handshake_node_id="node-a",
        )
        renewed = scheduler.renew_lease(grant["lease_id"], ttl_ms=1_000)
        before_wrong = ledger.get_lease(grant["lease_id"])
        wrong_error: dict[str, Any] | None = None
        try:
            renew_lease_request(bundle.nodes_by_id["node-a"].host, bundle.nodes_by_id["node-a"].port, node_id="node-a", lease_id=grant["lease_id"], expected_version=999)
        except DiagnosticError as exc:
            wrong_error = exc.to_dict()
        after_wrong = ledger.get_lease(grant["lease_id"])

        short = ledger.grant_lease(
            lease_id="lease-demo-expire",
            shard_id="shard-0003",
            node_id="node-c",
            capacity_version=1,
            resource_units=2,
            ttl_ms=40,
            budget_units=cap.budget_units,
            concurrency_quota=2,
            lease_limit=2,
        )
        time.sleep(0.09)
        expired = ledger.expire_due_leases()
        second_expire = ledger.expire_due_leases()
        return {
            "renewed_version": renewed["lease_version"],
            "renew_count": renewed["renew_count"],
            "wrong_version_error": wrong_error,
            "wrong_version_did_not_extend": before_wrong["expires_ms"] == after_wrong["expires_ms"] and after_wrong["lease_version"] == 2,
            "expired_lease_ids": [row["lease_id"] for row in expired],
            "second_expire_count": len(second_expire),
            "real_sleep_ms": 90,
        }
    finally:
        pool.stop()
        ledger.close()


def incremental_demo(work: Path) -> dict[str, Any]:
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    spec_dir = write_demo_spec(work / "incremental_spec", allocate_ports=True)
    receipt_dir = work / "receipts"
    write_receipt_fixtures(spec_dir, receipt_dir)
    initial_report, ledger, pool, scheduler = _run_all(spec_dir, work / "incremental.db", work / "records_before")
    before_hashes = {path.name: hash_file(path) for path in (work / "records_before").glob("*.json")}
    pool.stop()

    bundle = load_spec_bundle(spec_dir)
    changed_sample = bundle.shards_by_id["shard-0002"].sample_ids[0]
    samples_path = spec_dir / "samples.json"
    samples_payload = json.loads(samples_path.read_text(encoding="utf-8"))
    for sample in samples_payload["samples"]:
        if sample["id"] == changed_sample:
            sample["values"][0] = int(sample["values"][0] + 1) % 97
    samples_path.write_text(json.dumps(samples_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    edited_bundle = load_spec_bundle(spec_dir)
    propagation = propagate_invalidation(edited_bundle, changed_samples=[changed_sample])
    invalid_ids = set(propagation["invalid_shards"])

    pool = NodePool(edited_bundle, spec_dir / "logs_incremental")
    pool.start()
    try:
        scheduler = Scheduler(edited_bundle, ledger, pool)
        incremental_run = scheduler.run(target_ids=invalid_ids, rerun_ids=invalid_ids)
        incremental_view = ledger.semantic_view(1)
        for shard_id in incremental_run["committed"]:
            write_shard_record(work / "records_before", incremental_view[shard_id])
    finally:
        pool.stop()

    after_hashes = {path.name: hash_file(path) for path in (work / "records_before").glob("*.json")}
    unchanged = propagation["unchanged_shards"]
    byte_preservation = {
        shard_id: before_hashes[f"{shard_id}.json"] == after_hashes[f"{shard_id}.json"]
        for shard_id in unchanged
    }

    full_spec = write_demo_spec(work / "full_spec", allocate_ports=True)
    full_samples_path = full_spec / "samples.json"
    full_samples = json.loads(full_samples_path.read_text(encoding="utf-8"))
    for sample in full_samples["samples"]:
        if sample["id"] == changed_sample:
            sample["values"][0] = int(sample["values"][0] + 1) % 97
    full_samples_path.write_text(json.dumps(full_samples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    full_report, full_ledger, full_pool, _ = _run_all(full_spec, work / "full.db", work / "records_full")
    try:
        edited_bundle = load_spec_bundle(full_spec)
        full_view = full_ledger.semantic_view(1)
        compare_semantic_views(incremental_view, full_view, code=Codes.DIVERGE, order=stable_topological_order(edited_bundle.shards))
    finally:
        full_pool.stop()
        full_ledger.close()
        ledger.close()

    return {
        "changed_sample": changed_sample,
        "initial_committed": initial_report["committed"],
        "initial_wait_records": initial_report["wait_records"],
        "propagation": propagation,
        "incremental_committed": incremental_run["committed"],
        "byte_preservation": byte_preservation,
        "all_unchanged_bytes_identical": all(byte_preservation.values()),
        "full_committed": full_report["committed"],
        "field_compare": "passed: incremental semantic view equals blank full rebuild",
    }


def capacity_demo(work: Path) -> dict[str, Any]:
    spec_dir = write_demo_spec(work / "capacity_spec", allocate_ports=True)
    capacity_path = spec_dir / "capacity_versions.json"
    capacity_payload = json.loads(capacity_path.read_text(encoding="utf-8"))
    capacity_payload["versions"][0]["budget_units"] = 6
    capacity_path.write_text(json.dumps(capacity_payload, indent=2) + "\n", encoding="utf-8")
    bundle = load_spec_bundle(spec_dir)
    ledger = Ledger(work / "capacity.db")
    cap1 = bundle.capacity_by_version[1]
    ledger.initialize_capacity_slots(cap1)
    ledger.grant_lease(
        lease_id="lease-old-long",
        shard_id="shard-0001",
        node_id="node-a",
        capacity_version=1,
        resource_units=2,
        ttl_ms=10_000,
        budget_units=cap1.budget_units,
        concurrency_quota=2,
        lease_limit=2,
    )
    ledger.grant_lease(
        lease_id="lease-old-short",
        shard_id="shard-0003",
        node_id="node-c",
        capacity_version=1,
        resource_units=2,
        ttl_ms=30,
        budget_units=cap1.budget_units,
        concurrency_quota=2,
        lease_limit=2,
    )
    time.sleep(0.08)
    expired = [row["lease_id"] for row in ledger.expire_due_leases()]
    replacement = ledger.revoke_capacity_leases(2)
    second_expired = ledger.expire_due_leases()
    owner_rows = ledger.conn.execute("SELECT COUNT(*) count FROM resource_owners WHERE status='HELD'").fetchone()["count"]
    slot_rows = ledger.conn.execute("SELECT COUNT(*) count FROM lease_slots WHERE status='HELD'").fetchone()["count"]

    manifest_path = spec_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["active_capacity_version"] = 2
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    bundle2 = load_spec_bundle(spec_dir)
    snapshot, replaced, reasons = ledger.open_or_replace_snapshot(bundle2, 1, {"cursor": 0}, {"view": "capacity-v2"})
    pool = NodePool(bundle2, spec_dir / "logs")
    pool.start()
    try:
        report = Scheduler(bundle2, ledger, pool).run()
    finally:
        pool.stop()
    current_view = ledger.semantic_view(2)
    snapshots = ledger.list_snapshots()
    ledger.close()
    return {
        "expired_before_upgrade": expired,
        "revoked": replacement["revoked_lease_ids"],
        "second_expired_count": len(second_expired),
        "held_resource_owner_rows": owner_rows,
        "held_lease_slot_rows": slot_rows,
        "snapshot_replaced": replaced,
        "snapshot_replace_reasons": reasons,
        "snapshot_versions": snapshots,
        "v2_committed": report["committed"],
        "current_view_shards": sorted(current_view),
        "stale_guard": "v1 rows remain marked stale=1 and semantic_view(2) is the wholesale replacement",
    }


def crash_recovery_demo(work: Path) -> dict[str, Any]:
    spec_dir = write_demo_spec(work / "crash_spec", allocate_ports=True)
    db_path = work / "crash.db"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        sys.executable,
        "-m",
        "edge_shard_diag.cli",
        "crash-child",
        "--spec",
        str(spec_dir),
        "--db",
        str(db_path),
        "--crash-shard",
        "shard-0002",
    ]
    child = subprocess.Popen(command, cwd=str(REPO_ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        output, _ = child.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        output, _ = child.communicate()
        raise
    exit_code = child.returncode
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    time.sleep(0.2)

    ledger = Ledger(db_path)
    recovery = ledger.recover_orphans()
    bundle = load_spec_bundle(spec_dir)
    snapshot, snapshot_replaced, reasons = ledger.open_or_replace_snapshot(bundle, 1, {"cursor": "recovery"}, {"recovered": True})
    pool = NodePool(bundle, spec_dir / "recovery_logs")
    pool.start()
    try:
        rerun = Scheduler(bundle, ledger, pool).run()
        recovered_view = ledger.semantic_view(1)
    finally:
        pool.stop()

    full_spec = write_demo_spec(work / "replay_full_spec", allocate_ports=True)
    full_ledger = Ledger(work / "replay_full.db")
    full_pool = NodePool(load_spec_bundle(full_spec), full_spec / "logs")
    full_pool.start()
    try:
        Scheduler(load_spec_bundle(full_spec), full_ledger, full_pool).run()
        full_view = full_ledger.semantic_view(1)
        order = stable_topological_order(load_spec_bundle(full_spec).shards)
        compare_semantic_views(recovered_view, full_view, code=Codes.REPLAY, order=order)
    finally:
        full_pool.stop()
        full_ledger.close()
        ledger.close()

    return {
        "child_exit_code": exit_code,
        "child_was_sigkilled": exit_code == -9,
        "child_tail": output[-1200:],
        "crash_exit_report": "unavailable_by_design: SIGKILL permits only committed SQLite records",
        "recovery": recovery,
        "snapshot_replaced_on_reopen": snapshot_replaced,
        "snapshot_reasons": reasons,
        "rerun_committed": rerun["committed"],
        "replay_compare": "passed: recovered ledger replay equals blank full rebuild",
    }


def run_acceptance(workdir: str | Path = "artifacts/acceptance") -> dict[str, Any]:
    work = Path(workdir)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"workdir": str(work.resolve()), "steps": {}}

    spec_dir = write_demo_spec(work / "spec", allocate_ports=True)
    receipt_dir = work / "receipts"
    write_receipt_fixtures(spec_dir, receipt_dir)
    report["steps"]["spec_load"] = {"spec": str(spec_dir), "ok": True}
    report["steps"]["receipts"] = normalize_demo_receipts(spec_dir, receipt_dir)
    report["steps"]["fixed_point"] = fixed_point_demo()
    report["steps"]["leases"] = lease_demo(work)
    report["steps"]["incremental"] = incremental_demo(work)
    report["steps"]["capacity"] = capacity_demo(work)
    report["steps"]["crash_recovery"] = crash_recovery_demo(work)
    report["ok"] = True
    _json_dump(work / "acceptance-report.json", report)
    return report
