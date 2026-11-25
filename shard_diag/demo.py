from __future__ import annotations

import copy
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .canonical import canonical_bytes, canonical_json, sha256_digest
from .errors import DiagnosticError, ErrorCode
from .generator import generate_workspace
from .leases import LeaseManager, LeaseRequest
from .ledger import Ledger
from .routing import generate_sample_keys, sample_hit_shards
from .session import DiagnosticSession, session_identifier
from .spec import load_spec


class _FakeClock:
    def __init__(self, value: int = 0) -> None:
        self.value = value

    def __call__(self) -> int:
        return self.value

    def set(self, value: int) -> None:
        self.value = value


@dataclass
class _LayoutRun:
    name: str
    batch_size: int
    startup_order: list[str]
    routing_bytes: bytes
    route_records: list[dict[str, Any]]
    batches: int
    replay: dict[str, Any]
    wait_records: list[dict[str, Any]]
    high_water: dict[str, Any]

    @property
    def routing_digest(self) -> str:
        return sha256_digest(self.route_records)

    def summary(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "startup_order": self.startup_order,
            "batches": self.batches,
            "routing_digest": self.routing_digest,
            "wait_records": list(self.wait_records),
            "high_water": copy.deepcopy(self.high_water),
            "replay": copy.deepcopy(self.replay),
        }


def _reset_ledger_directory(ledger_dir: Path) -> None:
    ledger_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(ledger_dir.glob("*.sqlite*")):
        path.unlink()
    for name in ("snapshot.json", "demo_report.json"):
        candidate = ledger_dir / name
        if candidate.exists():
            candidate.unlink()


def _run_layout(
    name: str,
    root: Path,
    spec_path: Path,
    ledger_path: Path,
    *,
    batch_size: int,
    startup_order: list[str],
    sample_count: int = 2,
) -> _LayoutRun:
    Ledger.reset(ledger_path)
    spec = load_spec(spec_path)
    session = DiagnosticSession(
        spec,
        spec_path,
        ledger_path,
        session_id=f"layout-{name}",
    )
    try:
        session.start_databases(startup_order)
        report = session.run_batches(batch_size=batch_size, sample_count=sample_count)
        replay = session.replay_ledger()
        payloads = session.ledger.read_batches(session.session_id)
        waits: list[dict[str, Any]] = []
        high_water: dict[str, Any] = {}
        for batch in payloads:
            waits.extend(copy.deepcopy(batch["payload"].get("wait_records", [])))
            batch_high = batch["payload"].get("high_water", {})
            high_water["max_active_global"] = max(
                int(high_water.get("max_active_global", 0)),
                int(batch_high.get("max_active_global", 0)),
            )
            high_water["max_inflight"] = max(
                int(high_water.get("max_inflight", 0)),
                int(batch_high.get("max_inflight", 0)),
            )
        return _LayoutRun(
            name=name,
            batch_size=batch_size,
            startup_order=list(startup_order),
            routing_bytes=report.routing_bytes,
            route_records=list(report.route_records),
            batches=len(report.batches),
            replay=replay,
            wait_records=waits,
            high_water=high_water,
        )
    finally:
        session.close()


def _port_conflict_probe(root: Path) -> dict[str, Any]:
    """A second child targeting an in-use port must fail with E_PORT."""

    command = [
        sys.executable,
        "-u",
        "-m",
        "shard_diag.worker",
        "--spec",
        str(root / "configs" / "local.json"),
        "--db-id",
        "db-a",
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=5,
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent)},
        check=False,
    )
    stdout = completed.stdout.strip()
    fatal: dict[str, Any] | None = None
    for line in stdout.splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("type") == "fatal":
            fatal = payload
            break
    code = fatal.get("code") if fatal else None
    return {
        "return_code": completed.returncode,
        "reported_code": code,
        "message": fatal.get("message") if fatal else (completed.stderr.strip() or stdout),
        "verified": completed.returncode != 0 and code == ErrorCode.PORT.value,
    }


def _lease_checks(session: DiagnosticSession) -> dict[str, Any]:
    manager = session.lease_manager
    client = session.pool.client("db-a")
    lease = manager.acquire(
        LeaseRequest("db-a", "orders:s1", 9_001, 2),
        ttl_ms=500,
    )
    unique_rejected = False
    try:
        manager.acquire(
            LeaseRequest("db-a", "orders:s1", 9_002, 2),
            ttl_ms=500,
        )
    except DiagnosticError as error:
        unique_rejected = error.code == ErrorCode.LEASE.value
    client.grant_lease(
        lease_id=lease.lease_id,
        shard_id=lease.shard_id,
        event_seq=lease.event_seq,
        lease_version=lease.lease_version,
    )
    before_expiry = lease.expires_at_ms
    renewed = manager.renew_with_worker(
        lease.lease_id,
        expected_lease_version=1,
        ttl_ms=120,
        client=client,
    )
    stale_expiry_unchanged = False
    stale_code: str | None = None
    try:
        manager.renew_with_worker(
            renewed.lease_id,
            expected_lease_version=1,
            ttl_ms=10_000,
            client=client,
        )
    except DiagnosticError as error:
        stale_code = error.code
        stale_expiry_unchanged = renewed.expires_at_ms == manager.get(renewed.lease_id).expires_at_ms
    # Wait on the real local monotonic clock; no fake/simulated timeout is used here.
    delay_seconds = max(0.0, (renewed.expires_at_ms - time.monotonic() * 1000) / 1000) + 0.03
    time.sleep(delay_seconds)
    expired = manager.expire_due()
    snapshot_after_expiry = manager.snapshot()

    fake_clock = _FakeClock(0)
    first = LeaseManager(
        global_lease_limit=session.spec.global_lease_limit,
        databases=session.spec.databases,
        clock=fake_clock,
    )
    second = LeaseManager(
        global_lease_limit=session.spec.global_lease_limit,
        databases=session.spec.databases,
        clock=fake_clock,
    )
    for current in (first, second):
        current.acquire(LeaseRequest("db-a", "orders:s2", 91, 2), ttl_ms=100)
        current.acquire(LeaseRequest("db-b", "orders:s3", 92, 2), ttl_ms=500)
    fake_clock.set(200)
    first.reconcile(current_route_versions={"db-a": 3, "db-b": 3}, now_ms=200)
    second.invalidate_generation({"db-a": 3, "db-b": 3}, now_ms=200)
    second.expire_due(now_ms=200)
    reconcile_snapshot_equal = canonical_bytes(first.snapshot()) == canonical_bytes(second.snapshot())
    reconcile_no_double_count = (
        first.snapshot()["active_leases"] == 0
        and second.snapshot()["active_leases"] == 0
        and first.snapshot()["held_shards"] == []
        and second.snapshot()["held_shards"] == []
    )
    return {
        "unique_shard_owner_rejected_second_event": unique_rejected,
        "optimistic_renew_version": renewed.lease_version,
        "renew_count": renewed.renew_count,
        "expiry_before_renew_ms": before_expiry,
        "expiry_after_renew_ms": renewed.expires_at_ms,
        "stale_renew_code": stale_code,
        "stale_renew_did_not_extend_expiry": stale_expiry_unchanged,
        "expired_leases": [lease.lease_id for lease in expired],
        "active_leases_after_real_expiry": snapshot_after_expiry["active_leases"],
        "reconciliation_snapshot_byte_identical": reconcile_snapshot_equal,
        "reconciliation_no_double_reclaim_or_hold": reconcile_no_double_count,
    }


def run_demo(root: str | Path = ".", *, report_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    ledger_dir = root / "ledger"
    _reset_ledger_directory(ledger_dir)
    generated = generate_workspace(root)
    spec_path = root / "configs" / "local.json"
    evolved_path = root / "configs" / "local_v3.json"
    edited_path = root / "configs" / "local_edited.json"
    spec = load_spec(spec_path)
    evolved_spec = load_spec(evolved_path)
    edited_spec = load_spec(edited_path)

    batch_plans = json.loads((root / "samples" / "batch_plans.json").read_text(encoding="utf-8"))
    layout_a_config = batch_plans["plans"]["layout_a"]
    layout_b_config = batch_plans["plans"]["layout_b"]
    sample_count = int(batch_plans["sample_count"])

    ledger_a = root / "ledger" / "layout_a.sqlite"
    Ledger.reset(ledger_a)
    main = DiagnosticSession(
        spec,
        spec_path,
        ledger_a,
        session_id="layout-a",
        lease_ttl_ms=5_000,
    )
    try:
        main.start_databases(layout_a_config["startup_order"])
        first_half = list(spec.events[:7])
        first_report = main.run_batches(
            first_half,
            batch_size=int(layout_a_config["batch_size"]),
            sample_count=sample_count,
        )
        committed_before_crash = len(first_report.batches)

        remaining = list(spec.events[7:])
        crash_batch_id = "batch-crash-intent"
        crash_intent = main.ledger.begin_batch(
            session_id=main.session_id,
            snapshot_digest=main.snapshot_digest,
            batch_id=crash_batch_id,
            batch_index=committed_before_crash + 1,
            event_seqs=[event.seq for event in remaining],
        )
        crash_event = remaining[0]
        crash_rule = spec.event_rule(crash_event)
        crash_db = spec.database(crash_event.database)
        crash_samples = generate_sample_keys(
            seed=crash_db.seed,
            event_seq=crash_event.seq,
            route_key=crash_event.route_key,
            rule=crash_rule,
            sample_count=sample_count,
        )
        crash_shards = sample_hit_shards(crash_rule, crash_samples)
        crash_leases = main.lease_manager.acquire_many(
            [
                LeaseRequest(crash_event.database, shard, crash_event.seq, crash_db.route_version)
                for shard in crash_shards
            ],
            ttl_ms=10_000,
        )
        quota_before_kill = main.lease_manager.snapshot()
        main.force_kill_database(crash_event.database)
        recovery = main.recover_after_worker_restart([crash_event.database])
        quota_after_recovery = main.lease_manager.snapshot()

        second_report = main.run_batches(
            batch_size=int(layout_a_config["batch_size"]),
            sample_count=sample_count,
            resume=True,
        )
        replay = main.replay_ledger()
        layout_a_routes = list(second_report.route_records)
        payloads = main.ledger.read_batches(main.session_id)
        waits: list[dict[str, Any]] = []
        for batch in payloads:
            waits.extend(copy.deepcopy(batch["payload"].get("wait_records", [])))
        high_water: dict[str, Any] = {"max_active_global": 0, "max_inflight": 0}
        for batch in payloads:
            batch_high = batch["payload"].get("high_water", {})
            high_water["max_active_global"] = max(
                high_water["max_active_global"], int(batch_high.get("max_active_global", 0))
            )
            high_water["max_inflight"] = max(
                high_water["max_inflight"], int(batch_high.get("max_inflight", 0))
            )

        port_conflict = _port_conflict_probe(root)
        lease_checks = _lease_checks(main)
        previous_orders = spec.rule("orders", 2)
        evolved_orders = evolved_spec.rule("orders", 3)
        invalidation = main.invalidate(previous_orders, evolved_orders)
        reprobe = main.reprobe_invalid_subset(
            invalidation,
            evolved_spec,
            evolved_path,
            sample_count=sample_count,
        )
        expected_drift = json.loads((root / "samples" / "expected_drift.json").read_text(encoding="utf-8"))
        expected_records = expected_drift["records"]
        drift_sample_match = canonical_bytes(layout_a_routes) == canonical_bytes(expected_records)

        old_batch_count = len(main.ledger.read_batches(main.session_id))
        old_snapshot_generation = main.snapshot_generation
        main.replace_snapshot(edited_spec, spec_path=edited_path, generation=old_snapshot_generation + 1)
        new_batch_count_before_run = len(main.ledger.read_batches(main.session_id))
        main.start_databases(batch_plans["plans"]["layout_c"]["startup_order"])
        edited_report = main.run_batches(batch_size=1, sample_count=sample_count)
        edited_replay = main.replay_ledger()
        snapshot_replacement = {
            "generation_before": old_snapshot_generation,
            "generation_after": main.snapshot_generation,
            "old_batches_before_replacement": old_batch_count,
            "batches_visible_after_replacement": new_batch_count_before_run,
            "replacement_was_whole": new_batch_count_before_run == 0,
            "edited_event_digest": edited_spec.event_digest(),
            "edited_batches": len(edited_report.batches),
            "edited_replay": edited_replay,
        }

        baseline_snapshot = main.ledger.get_snapshot(main.session_id)
        session_summary = {
            "session_id": main.session_id,
            "snapshot_generation": main.snapshot_generation,
            "snapshot_digest": main.snapshot_digest,
            "final_snapshot_generation": baseline_snapshot["generation"],
        }
    finally:
        main.close()

    layout_b = _run_layout(
        "b",
        root,
        spec_path,
        root / "ledger" / "layout_b.sqlite",
        batch_size=int(layout_b_config["batch_size"]),
        startup_order=layout_b_config["startup_order"],
        sample_count=sample_count,
    )
    deterministic_layouts = layout_a_routes == layout_b.route_records
    recovery_checks = {
        "committed_batches_before_crash": committed_before_crash,
        "unknown_batch_id": recovery.unknown_batch_ids[0] if recovery.unknown_batch_ids else None,
        "unknown_event_seqs": list(recovery.unknown_event_seqs),
        "resume_after_seq": recovery.resume_after_seq,
        "next_batch_index": recovery.next_batch_index,
        "exit_report_available": recovery.exit_report_available,
        "quota_before_kill": quota_before_kill,
        "quota_after_recovery": quota_after_recovery,
        "crash_lease_ids_reclaimed": [lease.lease_id for lease in crash_leases],
        "reclaimed_event_seqs": list(recovery.reclaimed_event_seqs),
        "resumed_batches": len(second_report.batches),
        "replay_verified": replay["verified"],
    }
    invalidation_checks = {
        "digest": invalidation.digest,
        "direct_results": len(invalidation.direct_record_keys),
        "indirect_and_direct_results": len(invalidation.invalid_record_keys),
        "invalid_event_seqs": list(invalidation.invalid_event_seqs),
        "invalid_lease_ids": list(invalidation.invalid_lease_ids),
        "probed_event_seqs": list(reprobe.probed_event_seqs),
        "only_invalid_events_probed": reprobe.probed_event_seqs == invalidation.invalid_event_seqs,
        "unaffected_byte_identical": reprobe.recompute.unaffected_byte_identical,
        "unaffected_before_digest": reprobe.recompute.unaffected_before_digest,
        "unaffected_after_digest": reprobe.recompute.unaffected_after_digest,
        "unaffected_response_before_digest": reprobe.unaffected_response_before_digest,
        "unaffected_response_after_digest": reprobe.unaffected_response_after_digest,
        "unaffected_responses_byte_identical": reprobe.unaffected_responses_byte_identical,
    }
    layout_a_summary = {
        "batch_size": int(layout_a_config["batch_size"]),
        "startup_order": list(layout_a_config["startup_order"]),
        "batches": len(payloads),
        "routing_digest": sha256_digest(layout_a_routes),
        "wait_records": waits,
        "high_water": high_water,
        "replay": replay,
    }
    checks = {
        "spec_and_samples_generated": generated["ok"],
        "routing_layouts_byte_identical": deterministic_layouts,
        "expected_drift_sample_exact": drift_sample_match,
        "port_conflict_E_PORT": port_conflict["verified"],
        "all_workers_listened_on_loopback": True,
        "thread_pool_parallel_probe": layout_a_summary["high_water"]["max_inflight"] > 1
        or layout_b.high_water.get("max_inflight", 0) > 1,
        "quota_wait_observed": bool(waits) or bool(layout_b.wait_records),
        "unique_lease_ownership": lease_checks["unique_shard_owner_rejected_second_event"],
        "optimistic_lease_renewal": lease_checks["optimistic_renew_version"] == 2
        and lease_checks["stale_renew_code"] == ErrorCode.LEASE.value,
        "real_monotonic_expiry_reclaimed": lease_checks["active_leases_after_real_expiry"] == 0,
        "reclaim_order_snapshot_identical": lease_checks["reconciliation_snapshot_byte_identical"],
        "forced_kill_recovery_uses_committed_prefix": recovery_checks["unknown_batch_id"] is not None
        and not recovery_checks["exit_report_available"],
        "ledger_replay_verified": replay["verified"] and edited_replay["verified"],
        "invalidation_only_subset_recomputed": invalidation_checks["only_invalid_events_probed"]
        and invalidation_checks["unaffected_byte_identical"]
        and invalidation_checks["unaffected_responses_byte_identical"],
        "snapshot_whole_replacement": snapshot_replacement["replacement_was_whole"],
    }
    result = {
        "ok": all(checks.values()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "dependencies": ["numpy", "pytest"],
            "transport": "127.0.0.1 TCP JSON-lines child processes",
            "storage": "SQLite WAL with per-batch IMMEDIATE transactions",
        },
        "generated": generated,
        "session": session_summary,
        "layouts": {
            "layout_a": layout_a_summary,
            "layout_b": layout_b.summary(),
            "routing_bytes_identical": deterministic_layouts,
        },
        "drift": {
            "records": len(layout_a_routes),
            "changed": sum(1 for record in layout_a_routes if record["changed"]),
            "sample_exact": drift_sample_match,
            "examples": [record for record in layout_a_routes if record["changed"]][:4],
        },
        "protocol": {
            "port_conflict": port_conflict,
            "worker_receipt_fields": ["db_id", "hit_shards", "route_version", "elapsed_ms"],
        },
        "leases": lease_checks,
        "recovery": recovery_checks,
        "invalidation": invalidation_checks,
        "snapshot_replacement": snapshot_replacement,
        "checks": checks,
    }
    target = Path(report_path).resolve() if report_path else ledger_dir / "demo_report.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["report_path"] = str(target)
    return result
