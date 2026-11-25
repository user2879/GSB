from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

from .canonical import canonical_bytes, write_json_atomic
from .errors import DivergenceError, LeaseError, ReplayError
from .http_health import make_health_server
from .ledger import Ledger
from .leases import LeaseManager, replay_quota_events
from .portcheck import check_loopback_port
from .propagation import VersionedPlan
from .replay import recover_and_replay, verify_ledger_replay
from .sample_data import write_sample_bundle
from .spec import load_spec_file
from .workbench import RoundRunner


def _service_order_proof(ttl_ms: int = 60) -> tuple[bytes, bytes]:
    def run(order: list[str]) -> dict[str, Any]:
        manager = LeaseManager(3, {"alpha": 1, "beta": 1, "gamma": 1}, ttl_ms)
        for index, service_id in enumerate(order):
            manager.grant(
                f"order-lease-{index}",
                service_id,
                batch_id="order-batch",
                task_id=f"task-{index}",
            )
        time.sleep(ttl_ms / 1000.0 + 0.04)
        for service_id in order:
            manager.revoke_batch("order-batch", service_id=service_id)
            manager.reap_expired(service_id=service_id)
        snapshot = manager.snapshot()
        return {
            "global_remaining": snapshot["global_remaining"],
            "owner_set": snapshot["owner_set"],
            "services": {
                service_id: {
                    "held": snapshot["services"][service_id]["held"],
                    "remaining": snapshot["services"][service_id]["remaining"],
                }
                for service_id in sorted(snapshot["services"])
            },
        }

    ascending = run(["alpha", "beta", "gamma"])
    descending = run(["gamma", "beta", "alpha"])
    return canonical_bytes(ascending), canonical_bytes(descending)


def _tamper_allocation(db_path: Path, spec_service: str = "alpha", minute: int = 4) -> dict[str, Any]:
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT id, canonical_json FROM minute_results "
        "WHERE service_id=? AND minute=? AND record_type='allocation'",
        (spec_service, minute),
    ).fetchone()
    if row is None:
        conn.close()
        raise RuntimeError("allocation row not found for tamper demo")
    payload = json.loads(row[1])
    payload["allocated_replicas"] = int(payload["allocated_replicas"]) + 1
    conn.execute(
        "UPDATE minute_results SET canonical_json=? WHERE id=?",
        (json.dumps(payload, ensure_ascii=False, sort_keys=True), row[0]),
    )
    conn.commit()
    conn.close()
    return payload


def run_acceptance(root: str | Path = ".") -> dict[str, Any]:
    root = Path(root).resolve()
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    paths = write_sample_bundle(root)
    checks: dict[str, bool] = {}
    evidence: dict[str, Any] = {}

    def check(name: str, ok: bool, detail: Any = None) -> None:
        checks[name] = bool(ok)
        evidence[name] = detail

    port_info = check_loopback_port(port=0)
    server = make_health_server("127.0.0.1", 0)
    health_host, health_port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(
            f"http://{health_host}:{health_port}/health", timeout=3
        ) as response:
            health = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    check("loopback_port_and_health", port_info["ok"] and health["ok"], {
        "port_check": port_info,
        "health_url": f"http://{health_host}:{health_port}/health",
        "health": health,
    })

    spec = load_spec_file(paths["base_spec"])
    spec_check = None
    try:
        load_spec_file(paths["invalid_spec"])
    except Exception as exc:
        spec_check = exc.to_dict() if hasattr(exc, "to_dict") else {"message": str(exc)}
    check("e_spec_first_field", spec_check and spec_check["error"]["code"] == "E_SPEC", spec_check)

    plan = VersionedPlan(spec)
    audit = plan.audit_algorithm_paths()
    check("two_algorithm_paths_match", audit["equal"], audit)

    divergence_demo: dict[str, Any] | None = None
    try:
        from dataclasses import replace as dc_replace

        from .algorithms import run_incremental_path
        from .consistency import verify_paths

        full = plan.full_result
        bad_incremental = run_incremental_path(spec)
        alpha_minutes = list(bad_incremental.minutes["alpha"])
        alpha_minutes[2] = dc_replace(alpha_minutes[2], clamped_replicas=999)
        bad_incremental = dc_replace(
            bad_incremental,
            minutes={**bad_incremental.minutes, "alpha": tuple(alpha_minutes)},
        )
        verify_paths(
            spec,
            full,
            bad_incremental,
            artifact_dir=artifacts / "acceptance-divergence",
        )
    except DivergenceError as exc:
        divergence_demo = exc.to_dict()
    check(
        "e_diverge_locates_first_difference",
        divergence_demo is not None
        and divergence_demo["error"]["details"]["service_id"] == "alpha"
        and divergence_demo["error"]["details"]["minute"] == 2,
        divergence_demo,
    )

    ledger_path = artifacts / "acceptance-ledger.sqlite3"
    if ledger_path.exists():
        ledger_path.unlink()
    ledger = Ledger(ledger_path)
    runner = RoundRunner(spec, ledger, plan)
    round_result = runner.run_round(7)
    check(
        "parallel_limit_and_wait_recorded",
        round_result["max_observed_inflight"] <= spec.lease_policy.max_inflight_tasks
        and len(round_result["wait_spans"]) > 0,
        {
            "max_observed_inflight": round_result["max_observed_inflight"],
            "max_allowed": spec.lease_policy.max_inflight_tasks,
            "task_count": len(round_result["task_results"]),
            "wait_spans": round_result["wait_spans"],
            "round_commit": round_result["commit"],
        },
    )

    first_lease = round_result["leases"][0]
    stale_lease_error: dict[str, Any] | None = None
    try:
        runner.lease_manager.renew(first_lease.lease_id, 999_999)
    except LeaseError as exc:
        stale_lease_error = exc.to_dict()
    renewed = runner.lease_manager.renew(first_lease.lease_id, first_lease.version)
    runner.persist_new_lease_events()
    check(
        "e_lease_optimistic_version",
        stale_lease_error is not None
        and stale_lease_error["error"]["code"] == "E_LEASE"
        and renewed.version == first_lease.version + 1,
        {
            "stale_rejection": stale_lease_error,
            "renewed_lease": renewed.to_dict(),
            "renew_interval_ms": spec.lease_policy.renew_interval_ms,
        },
    )

    time.sleep(spec.lease_policy.lease_ttl_ms / 1000.0 + 0.08)
    revoke_events = runner.lease_manager.revoke_batch(round_result["payload"]["batch_id"])
    expiry_events = runner.lease_manager.reap_expired()
    runner.persist_new_lease_events()
    quota_snapshot = runner.lease_manager.snapshot()
    effective_releases = [
        event
        for event in runner.lease_manager.events()
        if event["event_type"] == "quota_released" and event["effective"]
    ]
    service_replica_quotas = {
        service_id: (
            spec.budget.service_concurrency_quotas[service_id]
            // spec.service(service_id).concurrency_per_replica
        )
        for service_id in spec.service_ids
    }
    quota_replay = replay_quota_events(
        spec.budget.global_replica_budget,
        service_replica_quotas,
        runner.lease_manager.events(),
    )
    check(
        "lease_attribution_unique",
        quota_snapshot["global_remaining"] == spec.budget.global_replica_budget
        and quota_snapshot["owner_set"] == []
        and len(effective_releases) == len(round_result["leases"])
        and all(event["source"] == "expiry" for event in effective_releases)
        and quota_replay["conflicts"] == [],
        {
            "revoke_call_events": revoke_events,
            "subsequent_expiry_events": expiry_events,
            "effective_release_count": len(effective_releases),
            "effective_releases": effective_releases,
            "quota_snapshot": {
                "global_remaining": quota_snapshot["global_remaining"],
                "owner_set": quota_snapshot["owner_set"],
            },
            "quota_replay": quota_replay,
        },
    )

    order_left, order_right = _service_order_proof()
    check(
        "service_order_replay_same_quota_state",
        order_left == order_right,
        {"ascending_proof": json.loads(order_left), "descending_proof": json.loads(order_right)},
    )

    old_demand = dict(plan.demand_bytes)
    old_allocations = dict(plan.allocation_bytes)
    batch1 = json.loads(paths["batch_001"].read_text(encoding="utf-8"))["requests"]
    propagation1 = plan.append_minutes(batch1)
    append1_bytes_ok = all(plan.demand_bytes[k] == v for k, v in old_demand.items()) and all(
        plan.allocation_bytes[k] == v for k, v in old_allocations.items()
    )
    before_batch2 = dict(plan.demand_bytes)
    before_batch2_allocations = dict(plan.allocation_bytes)
    batch2 = json.loads(paths["batch_002"].read_text(encoding="utf-8"))["requests"]
    propagation2 = plan.append_minutes(batch2)
    append2_bytes_ok = all(
        plan.demand_bytes[k] == v for k, v in before_batch2.items()
    ) and all(
        plan.allocation_bytes[k] == v for k, v in before_batch2_allocations.items()
    )
    budget_propagation = plan.update_budget(global_replica_budget=7)
    factor_propagation = plan.update_scaling_factor(1005)
    final_audit = plan.audit_algorithm_paths()
    ledger.save_plan_version(plan)
    check(
        "rolling_append_invalidation_and_byte_reuse",
        plan.version == 5
        and propagation1["affected_minutes"] == [18, 19, 20]
        and propagation2["affected_minutes"] == [21, 22, 23]
        and append1_bytes_ok
        and append2_bytes_ok
        and final_audit["equal"],
        {
            "version": plan.version,
            "append_001": propagation1,
            "append_002": propagation2,
            "budget_update": budget_propagation,
            "factor_update": factor_propagation,
            "final_algorithm_audit": final_audit,
        },
    )

    snapshot_path = artifacts / "acceptance-snapshot.json"
    ledger.write_snapshot(snapshot_path)
    tampered_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    tampered_snapshot["content_digest"] = "f" * 64
    snapshot_path.write_text(json.dumps(tampered_snapshot, sort_keys=True), encoding="utf-8")
    replaced_snapshot, did_replace, snapshot_reason = ledger.open_or_replace_snapshot(
        snapshot_path
    )
    check(
        "snapshot_digest_mismatch_overall_replace",
        did_replace and snapshot_reason == "snapshot_digest_mismatch",
        {"reason": snapshot_reason, "replaced_digest": replaced_snapshot["content_digest"]},
    )

    live_recovery = recover_and_replay(ledger, spec)
    check(
        "committed_ledger_replay",
        live_recovery["ok"] and len(live_recovery["replay"]["rounds"]) == 1,
        {
            "replayed_rounds": len(live_recovery["replay"]["rounds"]),
            "digest": live_recovery["replay"]["digest"],
            "next_minute": live_recovery["next_minute"],
        },
    )
    ledger.close()

    crash_db = artifacts / "acceptance-crash.sqlite3"
    for suffix in ("", "-wal", "-shm", "-journal"):
        candidate = Path(str(crash_db) + suffix)
        if candidate.exists():
            candidate.unlink()
    command = [
        sys.executable,
        "-m",
        "scale_review.cli",
        "crash-child",
        "--spec",
        str(paths["base_spec"]),
        "--db",
        str(crash_db),
        "--minute",
        "4",
    ]
    crashed = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=20,
    )
    crash_ledger = Ledger(crash_db)
    crash_recovery = recover_and_replay(crash_ledger, spec)
    check(
        "forced_kill_recovery_replays_committed_round",
        crashed.returncode != 0
        and len(crash_recovery["recovered_attempts"]) == 1
        and len(crash_recovery["replay"]["rounds"]) == 1
        and crash_recovery["recovered_attempts"][0]["status"] == "unknown",
        {
            "command": command,
            "returncode": crashed.returncode,
            "stdout": crashed.stdout,
            "stderr": crashed.stderr[-1000:],
            "recovery": crash_recovery,
        },
    )

    tampered_db = artifacts / "acceptance-e-replay.sqlite3"
    shutil.copy2(crash_db, tampered_db)
    crash_ledger.close()
    _tamper_allocation(tampered_db)
    replay_error: dict[str, Any] | None = None
    tampered_ledger = Ledger(tampered_db)
    try:
        verify_ledger_replay(tampered_ledger, spec)
    except ReplayError as exc:
        replay_error = exc.to_dict()
    finally:
        tampered_ledger.close()
    check(
        "e_replay_locates_first_difference",
        replay_error is not None
        and replay_error["error"]["code"] == "E_REPLAY"
        and "allocated_replicas" in replay_error["error"]["field"],
        replay_error,
    )

    ok = all(checks.values())
    report = {
        "ok": ok,
        "generated_at_monotonic_note": "all timing evidence uses time.monotonic_ns",
        "checks": checks,
        "evidence": evidence,
        "report_path": str(artifacts / "acceptance-report.json"),
    }
    write_json_atomic(artifacts / "acceptance-report.json", report)
    return report
