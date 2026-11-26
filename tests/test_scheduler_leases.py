from __future__ import annotations

import hashlib
import time
from typing import Any

from gsb.scheduler import LeaseManager, Scheduler
from gsb.spec import parse_spec
from helpers import base_raw_spec


def _aggregate(shard_id: str) -> dict[str, Any]:
    return {
        "aggregate_digest": hashlib.sha256(shard_id.encode()).hexdigest(),
        "sample_digest": hashlib.sha256(b"sample" + shard_id.encode()).hexdigest(),
        "participating_rows": 10,
        "node_elapsed_ms": 1,
        "network_elapsed_ms": 2,
    }


def test_scheduler_enforces_all_capacity_gates_and_records_waits() -> None:
    spec = parse_spec(base_raw_spec(global_budget=6, max_workers=3))
    holder: dict[str, Scheduler] = {}
    snapshots: list[dict[str, Any]] = []

    def task(shard, handle) -> dict[str, Any]:
        handle.renew()
        stats = holder["scheduler"].leases.stats()
        snapshots.append(stats)
        assert stats["global_units"] <= spec.global_budget
        for node in spec.nodes:
            assert stats["node_units"][node.id] <= node.quota
            assert stats["node_lease_counts"][node.id] <= node.lease_limit
            assert stats["node_shard_counts"][node.id] <= node.concurrency
        time.sleep(0.03)
        return _aggregate(shard.id)

    scheduler = Scheduler(spec, task=task, session_version=1)
    holder["scheduler"] = scheduler
    result = scheduler.run()
    assert len(result.records) == len(spec.shards)
    assert any("global_budget" in wait["reasons"] for wait in result.waits)
    assert all(wait["end_ms"] >= wait["start_ms"] for wait in result.waits)
    assert any(event["event"] == "renewed" for event in result.lease_events)
    assert result.lease_stats["global_units"] == 0
    assert snapshots


def test_expiry_and_quota_reclaim_release_capacity_once() -> None:
    spec = parse_spec(base_raw_spec())
    manager = LeaseManager(spec)
    shard = spec.shard_map()["raw-a"]
    manager.issue(lease_id="lease-expiry", shard=shard, ttl_ms=10)
    assert manager.stats()["global_units"] == shard.demand
    time.sleep(0.03)
    reclaimed = manager.reclaim_expired()
    assert len(reclaimed) == 1
    assert reclaimed[0]["lease_id"] == "lease-expiry"
    assert manager.stats()["global_units"] == 0
    assert manager.release("lease-expiry", reason="late") is False
    assert manager.stats()["global_units"] == 0
    assert manager.reclaim_for_quota_change("node-a") == []


def test_quota_reclaim_cannot_double_release() -> None:
    spec = parse_spec(base_raw_spec())
    manager = LeaseManager(spec)
    shard = spec.shard_map()["raw-a"]
    manager.issue(lease_id="lease-quota", shard=shard, ttl_ms=1000)
    assert manager.stats()["global_units"] == shard.demand
    assert len(manager.reclaim_for_quota_change("node-a")) == 1
    assert manager.stats()["global_units"] == 0
    assert manager.release("lease-quota", reason="completed") is False
    assert manager.reclaim_for_quota_change("node-a") == []
    assert manager.stats()["global_units"] == 0


def test_expired_running_lease_stays_quarantined_until_worker_exit() -> None:
    spec = parse_spec(base_raw_spec())
    manager = LeaseManager(spec)
    shard = spec.shard_map()["raw-a"]
    manager.issue(lease_id="lease-protected", shard=shard, ttl_ms=10)
    time.sleep(0.03)
    reclaimed = manager.reclaim_expired(protected_lease_ids=["lease-protected"])
    assert reclaimed[0]["quarantined_until_worker_exit"] is True
    assert manager.stats()["global_units"] == shard.demand
    assert manager.lease("lease-protected").status == "expired"
    assert manager.release("lease-protected", reason="worker_exit") is True
    assert manager.stats()["global_units"] == 0
