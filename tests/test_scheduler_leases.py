from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from workbench.engine import WorkbenchEngine
from workbench.errors import LeaseError
from workbench.timeutil import MonotonicClock, ScriptedClock
from workbench.testkit import write_test_spec


def test_parallel_schedule_completes_and_releases_all(tmp_path):
    spec_path = tmp_path / "spec.json"
    spec = write_test_spec(spec_path)
    engine = WorkbenchEngine(
        spec,
        tmp_path / "work",
        clock=ScriptedClock(start_ms=1000),
        parallel=True,
        max_workers=3,
    )
    try:
        result = engine.run_until_complete()
        assert len(result["completed"]) == 6
        assert engine.ledger.active_resource_sum() == 0
        assert engine.ledger.list_leases(active_only=True) == []
        shards = engine.ledger.list_shards()
        assert all(row["status"] == "completed" and row["summary_digest"] for row in shards)
        water_keys = {row["record_key"] for row in engine.ledger.export_view()["water_records"]}
        assert "global:global_budget" in water_keys
        assert "node:n1" in water_keys
    finally:
        engine.close()


def test_global_budget_queues_overshoot_with_wait_fields(tmp_path):
    raw = write_test_spec(tmp_path / "base.json").raw
    raw["global_budget_units"] = 4
    spec = write_test_spec(tmp_path / "budget.json", raw)
    engine = WorkbenchEngine(
        spec, tmp_path / "work", clock=ScriptedClock(start_ms=5000), parallel=False
    )
    try:
        result = engine.run_until_complete()
        assert set(result["completed"]) == {"s1", "s2", "s3", "s4", "s5", "s6"}
        blocked = [d for d in engine.decisions if not d["admitted"] and d["reason"] == "global_budget"]
        assert blocked
        water_affected = [
            candidate
            for decision in engine.water_decisions
            for candidate in decision["candidates"]
            if candidate["shard_id"] == "s2" and candidate["error_affected"]
        ]
        assert water_affected
        s2 = engine.ledger.get_shard("s2")
        assert s2["wait_started_ms"] is not None
        assert s2["wait_ended_ms"] is not None
        assert s2["wait_ms"] >= 0
    finally:
        engine.close()


def test_lease_optimistic_version_rejects_stale_renew(tmp_path):
    spec = write_test_spec(tmp_path / "spec.json")
    engine = WorkbenchEngine(spec, tmp_path / "work", clock=ScriptedClock())
    try:
        engine.start_nodes()
        context = engine.admit_shard("s1")
        assert context is not None
        renewed = engine.renew_lease("lease-s1", 1)
        assert renewed["lease_version"] == 2
        assert renewed["renew_count"] == 1
        expires = renewed["expires_at_ms"]
        with pytest.raises(LeaseError) as exc:
            engine.renew_lease("lease-s1", 1)
        assert exc.value.code == "E_LEASE"
        unchanged = engine.ledger.get_lease("lease-s1")
        assert unchanged["lease_version"] == 2
        assert unchanged["expires_at_ms"] == expires
    finally:
        engine.close()


def test_real_monotonic_expiry_reaps_once(tmp_path):
    raw = write_test_spec(tmp_path / "base.json").raw
    raw["session"]["lease_ttl_ms"] = 120
    spec = write_test_spec(tmp_path / "ttl.json", raw)
    engine = WorkbenchEngine(spec, tmp_path / "work", clock=MonotonicClock())
    try:
        engine.start_nodes()
        assert engine.admit_shard("s1") is not None
        time.sleep(0.18)
        first = engine.reap_expired_leases()
        second = engine.reap_expired_leases()
        assert first == ["lease-s1"]
        assert second == []
        lease = engine.ledger.get_lease("lease-s1")
        assert lease["state"] == "expired"
        assert lease["reap_reason"] == "monotonic_ttl"
        assert engine.ledger.active_resource_sum() == 0
        assert engine.reap_capacity_generation("n1", 2) == []
    finally:
        engine.close()


def test_version_generation_reaps_same_quota_only_once(tmp_path):
    spec = write_test_spec(tmp_path / "spec.json")
    engine = WorkbenchEngine(spec, tmp_path / "work", clock=ScriptedClock())
    try:
        engine.start_nodes()
        assert engine.admit_shard("s1") is not None
        reaped = engine.reap_capacity_generation("n1", 2)
        assert reaped == ["lease-s1"]
        lease = engine.ledger.get_lease("lease-s1")
        assert lease["state"] == "superseded"
        assert lease["reap_reason"] == "capacity_version_generation"
        assert engine.reap_expired_leases() == []
        assert engine.ledger.active_resource_sum() == 0
    finally:
        engine.close()


def test_concurrent_expiry_and_version_reclaim_only_one_wins(tmp_path):
    spec = write_test_spec(tmp_path / "spec.json")
    engine = WorkbenchEngine(spec, tmp_path / "work", clock=ScriptedClock())
    try:
        engine.start_nodes()
        assert engine.admit_shard("s1") is not None

        with ThreadPoolExecutor(max_workers=2) as pool:
            expired_future = pool.submit(engine.ledger.reap_expired, 10**18)
            version_future = pool.submit(
                engine.ledger.reap_version_generation, "n1", 2, 10**18
            )
            expired = expired_future.result(timeout=5)
            versioned = version_future.result(timeout=5)

        assert len(expired) + len(versioned) == 1
        lease = engine.ledger.get_lease("lease-s1")
        assert lease["state"] in {"expired", "superseded"}
        assert engine.ledger.active_resource_sum() == 0
        assert engine.ledger.list_leases(active_only=True) == []
    finally:
        engine.close()


def test_concurrency_quota_zero_blocks_even_with_lease_headroom(tmp_path):
    raw = write_test_spec(tmp_path / "base.json").raw
    n1 = next(node for node in raw["nodes"] if node["node_id"] == "n1")
    n1["concurrency_quota"] = 0
    n1["lease_limit"] = 10
    spec = write_test_spec(tmp_path / "quota-zero.json", raw)
    engine = WorkbenchEngine(spec, tmp_path / "work", clock=ScriptedClock())
    try:
        assert engine.admission_block_reason("s1") == "concurrency_quota"
    finally:
        engine.close()
