import threading
import time

from ckpt_diag.clock import VirtualClock
from ckpt_diag.scheduler import BatchTicket, Scheduler


def test_global_budget_blocks_then_admits():
    clock = VirtualClock(0)
    scheduler = Scheduler(
        global_budget=2,
        shard_limits={"a": (5, 5)},
        max_workers=4, clock=clock.now_ms)
    gate = threading.Event()
    started = threading.Event()

    def slow_job(ticket):
        started.set()
        gate.wait(timeout=2)
        return ticket.batch_id

    def quick_job(ticket):
        return ticket.batch_id

    first = scheduler.submit(BatchTicket("b1", "a", 2), slow_job)
    started.wait(timeout=2)
    # second job needs 1 unit but budget is fully consumed by b1
    second = scheduler.submit(BatchTicket("b2", "a", 1), quick_job)
    time.sleep(0.1)
    assert not second.done()
    ticket2 = scheduler.tickets["b2"]
    assert ticket2.status == "queued"
    gate.set()
    assert first.result(timeout=2) == "b1"
    assert second.result(timeout=2) == "b2"
    report = {row["batch_id"]: row for row in scheduler.wait_report()}
    assert report["b2"]["wait_ms"] >= 0
    scheduler.shutdown()


def test_shard_concurrency_quota_serializes():
    scheduler = Scheduler(
        global_budget=10,
        shard_limits={"a": (10, 1)},
        max_workers=4, clock=VirtualClock(0).now_ms)
    active = {"count": 0, "peak": 0}
    lock = threading.Lock()

    def job(ticket):
        with lock:
            active["count"] += 1
            active["peak"] = max(active["peak"], active["count"])
        time.sleep(0.05)
        with lock:
            active["count"] -= 1
        return ticket.batch_id

    futures = [scheduler.submit(BatchTicket(f"b{i}", "a", 1), job)
               for i in range(4)]
    results = [future.result(timeout=5) for future in futures]
    assert results == [f"b{i}" for i in range(4)]
    assert active["peak"] == 1
    scheduler.shutdown()


def test_lease_cap_blocks_on_units():
    scheduler = Scheduler(
        global_budget=10,
        shard_limits={"a": (1, 5)},
        max_workers=4, clock=VirtualClock(0).now_ms)
    gate = threading.Event()

    def slow(ticket):
        gate.wait(timeout=2)
        return ticket.batch_id

    first = scheduler.submit(BatchTicket("b1", "a", 1), slow)
    second = scheduler.submit(BatchTicket("b2", "a", 1), lambda t: "b2")
    time.sleep(0.1)
    assert not second.done()
    gate.set()
    assert first.result(2) == "b1" and second.result(2) == "b2"
    scheduler.shutdown()


def test_quota_update_unblocks_waiting_batch():
    scheduler = Scheduler(
        global_budget=10,
        shard_limits={"a": (1, 1)},
        max_workers=4, clock=VirtualClock(0).now_ms)
    gate = threading.Event()
    first = scheduler.submit(BatchTicket("b1", "a", 1),
                             lambda t: gate.wait(2) or "b1")
    second = scheduler.submit(BatchTicket("b2", "a", 1), lambda t: "b2")
    time.sleep(0.1)
    assert not second.done()
    scheduler.update_quota("a", lease_cap=5, concurrency_quota=5)
    gate.set()
    assert second.result(2) == "b2"
    scheduler.shutdown()
