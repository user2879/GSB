"""Dual-quota scheduling and lease lifecycle."""
from __future__ import annotations

import threading

import pytest

from obsflow.clock import ManualClock
from obsflow.errors import LeaseError
from obsflow.lease import LeaseManager
from obsflow.scheduler import AggregationTask, BoundedScheduler


def test_scheduler_respects_global_and_per_source_limits():
    clock = ManualClock(0)
    scheduler = BoundedScheduler(global_concurrency=4,
                                 per_source_concurrency=2, clock=clock)
    tasks = [AggregationTask(f"t{i}", "m", i, f"s{i % 3}", i)
             for i in range(12)]
    gate = threading.Event()
    lock = threading.Lock()
    inflight = 0
    current_source: dict[str, int] = {}
    peak_source: dict[str, int] = {}

    def worker(task):
        nonlocal inflight
        with lock:
            inflight += 1
            current_source[task.source_id] = current_source.get(
                task.source_id, 0) + 1
            peak_source[task.source_id] = max(
                peak_source.get(task.source_id, 0),
                current_source[task.source_id])
        gate.wait(1)
        with lock:
            inflight -= 1
            current_source[task.source_id] -= 1
        return task.payload

    results, timings = scheduler.run(tasks, worker)
    assert results == list(range(12))
    assert inflight == 0
    assert scheduler.observed_max_global() <= 4
    assert scheduler.observed_max_global() >= 1
    assert all(peak_source[s] <= 2 for s in peak_source)
    assert all(scheduler.observed_max_source(s) <= 2
               for s in ("s0", "s1", "s2"))
    assert all(t.wait_end_ms >= t.wait_start_ms for t in timings)
    assert all(t.run_end_ms >= t.run_start_ms for t in timings)


def test_scheduler_records_queue_waiting_on_manual_clock():
    clock = ManualClock(100)
    scheduler = BoundedScheduler(1, 1, clock=clock)
    tasks = [AggregationTask(f"t{i}", "m", i, "s", i) for i in range(3)]

    def worker(task):
        clock.advance(10)
        return task.payload

    _, timings = scheduler.run(tasks, worker)
    assert [t.enqueued_ms for t in timings] == [100, 100, 100]
    # Serial execution moves the manual clock forward.
    assert timings[-1].run_end_ms > timings[0].run_start_ms


def test_lease_renew_optimistic_version_and_real_expiry():
    clock = ManualClock(0)
    manager = LeaseManager(clock=clock, ttl_ms=25)
    lease = manager.grant("m", 0, 1, 5)
    renewed = manager.renew(lease.lease_id, lease.version)
    assert renewed.version == 2
    with pytest.raises(LeaseError) as info:
        manager.renew(lease.lease_id, lease.version)
    assert info.value.code == "E_LEASE"
    assert manager.active_lease("m", 0).version == 2

    clock.advance(26)
    assert manager.reap_expired()[0].state == "expired"
    assert manager.active_lease("m", 0) is None
    with pytest.raises(LeaseError) as info:
        manager.renew(renewed.lease_id, 2)
    assert info.value.code == "E_LEASE"


def test_revoked_lease_does_not_block_regrant():
    clock = ManualClock(0)
    manager = LeaseManager(clock=clock, ttl_ms=25)
    lease = manager.grant("m", 0, 1, 5)
    manager.release(lease.lease_id)
    new_lease = manager.grant("m", 0, 1, 5)
    assert new_lease.lease_id != lease.lease_id
