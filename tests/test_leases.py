from __future__ import annotations

import time

import pytest

from shard_diag.canonical import canonical_bytes, monotonic_ms
from shard_diag.errors import DiagnosticError, ErrorCode
from shard_diag.leases import LeaseManager, LeaseRequest


class FakeClock:
    def __init__(self, value=0):
        self.value = value

    def __call__(self):
        return self.value


def _manager(spec, clock=None, global_limit=None):
    kwargs = {}
    if clock is not None:
        kwargs["clock"] = clock
    return LeaseManager(
        global_lease_limit=global_limit or spec.global_lease_limit,
        databases=spec.databases,
        **kwargs,
    )


def test_global_shard_quota_cannot_be_held_by_two_databases(spec_bundle):
    manager = _manager(spec_bundle["spec"])
    manager.acquire(LeaseRequest("db-a", "orders:s1", 1, 2), ttl_ms=100)
    with pytest.raises(DiagnosticError) as caught:
        manager.acquire(LeaseRequest("db-b", "orders:s1", 2, 2), ttl_ms=100)
    assert caught.value.code == ErrorCode.LEASE.value
    assert "already held" in caught.value.message


def test_global_and_per_database_lease_limits(spec_bundle):
    spec = spec_bundle["spec"]
    manager = _manager(spec, global_limit=2)
    manager.acquire(LeaseRequest("db-a", "orders:s0", 1, 2), ttl_ms=100)
    manager.acquire(LeaseRequest("db-a", "orders:s1", 2, 2), ttl_ms=100)
    with pytest.raises(DiagnosticError) as caught:
        manager.acquire(LeaseRequest("db-b", "orders:s2", 3, 2), ttl_ms=100)
    assert caught.value.code == ErrorCode.QUOTA.value

    per_db = _manager(spec)
    for seq, shard in enumerate(["orders:s0", "orders:s1", "orders:s2"], start=10):
        per_db.acquire(LeaseRequest("db-a", shard, seq, 2), ttl_ms=100)
    with pytest.raises(DiagnosticError) as caught:
        per_db.acquire(LeaseRequest("db-a", "orders:s3", 20, 2), ttl_ms=100)
    assert caught.value.code == ErrorCode.QUOTA.value


def test_expiry_uses_monotonic_clock_semantics(spec_bundle):
    clock = FakeClock(100)
    manager = _manager(spec_bundle["spec"], clock=clock)
    lease = manager.acquire(LeaseRequest("db-a", "orders:s1", 1, 2), ttl_ms=20)
    assert lease.granted_at_ms == 100
    assert lease.expires_at_ms == 120
    clock.value = 119
    assert manager.expire_due() == []
    clock.value = 120
    expired = manager.expire_due()
    assert [item.lease_id for item in expired] == [lease.lease_id]
    assert manager.snapshot()["active_leases"] == 0


def test_expiry_with_real_local_monotonic_clock(spec_bundle):
    manager = _manager(spec_bundle["spec"])
    lease = manager.acquire(LeaseRequest("db-a", "orders:s2", 2, 2), ttl_ms=20)
    started = monotonic_ms()
    time.sleep(0.04)
    expired = manager.expire_due()
    assert expired and expired[0].lease_id == lease.lease_id
    assert monotonic_ms() - started >= 20
    assert manager.snapshot()["active_leases"] == 0


def test_worker_optimistic_renewal_and_stale_rejection(running_pool, spec_bundle):
    manager = _manager(spec_bundle["spec"])
    lease = manager.acquire(LeaseRequest("db-a", "orders:s1", 101, 2), ttl_ms=5_000)
    client = running_pool.client("db-a")
    client.grant_lease(
        lease_id=lease.lease_id,
        shard_id=lease.shard_id,
        event_seq=lease.event_seq,
        lease_version=lease.lease_version,
    )
    renewed = manager.renew_with_worker(
        lease.lease_id,
        expected_lease_version=1,
        ttl_ms=6_000,
        client=client,
    )
    assert renewed.lease_version == 2
    assert renewed.renew_count == 1
    expiry_before_stale = renewed.expires_at_ms
    with pytest.raises(DiagnosticError) as caught:
        manager.renew_with_worker(
            renewed.lease_id,
            expected_lease_version=1,
            ttl_ms=20_000,
            client=client,
        )
    assert caught.value.code == ErrorCode.LEASE.value
    assert manager.get(renewed.lease_id).expires_at_ms == expiry_before_stale


def test_expiry_and_generation_reclaim_are_single_owner_transitions(spec_bundle):
    clock_a = FakeClock(0)
    clock_b = FakeClock(0)
    first = _manager(spec_bundle["spec"], clock=clock_a)
    second = _manager(spec_bundle["spec"], clock=clock_b)
    for manager in (first, second):
        manager.acquire(LeaseRequest("db-a", "orders:s2", 1, 2), ttl_ms=100)
        manager.acquire(LeaseRequest("db-b", "orders:s3", 2, 2), ttl_ms=500)
    clock_a.value = 200
    clock_b.value = 200
    first.reconcile(current_route_versions={"db-a": 3, "db-b": 3}, now_ms=200)
    second.invalidate_generation({"db-a": 3, "db-b": 3}, now_ms=200)
    second.expire_due(now_ms=200)
    assert canonical_bytes(first.snapshot()) == canonical_bytes(second.snapshot())
    assert first.snapshot()["active_leases"] == 0
    assert second.snapshot()["active_leases"] == 0
    assert first.snapshot()["held_shards"] == []
    assert len(first.transitions()) == 4
    assert len(second.transitions()) == 4
    granted = [
        transition
        for transition in first.transitions()
        if transition["to_status"] == "active"
    ]
    terminal = [
        transition
        for transition in first.transitions()
        if transition["to_status"] in {"expired", "superseded"}
    ]
    assert len(granted) == 2
    assert len(terminal) == 2
    assert {transition["reason"] for transition in terminal} == {"lease_expired", "route_generation_superseded"}
