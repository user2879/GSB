from __future__ import annotations

import threading
import time

import pytest

from ckptsim.errors import CkptError
from ckptsim.lease import LeaseScheduler


def test_scheduler_enforces_three_limits_and_records_wait() -> None:
    gate = threading.Event()
    scheduler = LeaseScheduler(
        global_lease_limit=2,
        shard_lease_limits={"a": 1, "b": 2},
        shard_concurrency={"a": 2, "b": 1},
        ttl_ms=2_000,
    )
    try:
        tasks = []

        def work(lease):
            gate.wait(2)
            return lease.lease_id

        for index in range(6):
            shard = "a" if index % 2 == 0 else "b"
            tasks.append(
                scheduler.submit(
                    seq=index,
                    shard_id=shard,
                    task_id=f"task-{index}",
                    fn=work,
                )
            )
        time.sleep(0.1)
        snapshot = scheduler.snapshot()
        assert snapshot["max_observed_active_leases"] <= 2
        assert snapshot["max_observed_active_leases_by_shard"]["a"] <= 1
        assert snapshot["max_observed_active_leases_by_shard"]["b"] <= 2
        assert snapshot["max_observed_inflight_by_shard"]["a"] <= 2
        assert snapshot["max_observed_inflight_by_shard"]["b"] <= 1
        gate.set()
        scheduler.wait_all(timeout=5)
        snapshot = scheduler.snapshot()
        assert len(snapshot["wait_records"]) == 6
        assert any(record["wait_ms"] > 0 for record in snapshot["wait_records"])
        assert all(record["wait_start_ms"] <= record["wait_end_ms"] for record in snapshot["wait_records"])
    finally:
        scheduler.close()


def test_lease_expiry_uses_monotonic_time_and_releases_quota() -> None:
    gate = threading.Event()
    scheduler = LeaseScheduler(
        global_lease_limit=1,
        shard_lease_limits={"a": 1, "b": 1},
        shard_concurrency={"a": 1, "b": 1},
        ttl_ms=50,
    )
    try:
        first = scheduler.submit(seq=1, shard_id="a", task_id="first", fn=lambda lease: gate.wait(2))
        lease = first.wait_for_lease(1)
        time.sleep(0.12)
        scheduler.expire_now()
        assert lease.status == "expired"
        second = scheduler.submit(seq=2, shard_id="b", task_id="second", fn=lambda lease: gate.wait(2))
        second.wait_for_lease(1)
        gate.set()
        scheduler.wait_all(timeout=5)
    finally:
        gate.set()
        scheduler.close()


def test_renewal_is_validated_remotely_and_mismatch_does_not_extend() -> None:
    gate = threading.Event()
    scheduler = LeaseScheduler(
        global_lease_limit=1,
        shard_lease_limits={"a": 1},
        shard_concurrency={"a": 1},
        ttl_ms=2_000,
    )
    calls: list[int] = []
    try:
        task = scheduler.submit(
            seq=1,
            shard_id="a",
            task_id="task",
            fn=lambda lease: gate.wait(2),
        )
        lease = task.wait_for_lease(1)
        original_expiry = lease.expires_ms

        def validate(current_lease):
            calls.append(current_lease.version)
            if current_lease.version != 1:
                raise CkptError("E_LEASE", "version mismatch")
            return {"ok": True}

        with pytest.raises(CkptError) as caught:
            scheduler.renew(task, expected_version=9, validate=validate)
        assert caught.value.code == "E_LEASE"
        assert calls == [1]
        assert lease.expires_ms == original_expiry
        scheduler.renew(task, expected_version=1, validate=validate)
        assert lease.version == 2
        assert lease.expires_ms >= original_expiry
    finally:
        gate.set()
        scheduler.close()
