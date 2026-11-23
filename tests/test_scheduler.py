import threading
import time

import pytest

from gsb.scheduler import BatchJob, RetryableError, Scheduler


def test_global_limit_respected_and_waits_recorded():
    scheduler = Scheduler(2, 10, ["b1"], max_workers=8)
    running = 0
    peak = 0
    lock = threading.Lock()

    def execute(payload, timeout_ms):
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        time.sleep(0.05)
        with lock:
            running -= 1
        return {"ok": True}

    futures = [
        scheduler.submit(BatchJob(batch_id=f"ba-{i}", backend_id="b1"), execute)
        for i in range(6)
    ]
    outcomes = [f.result() for f in futures]
    scheduler.shutdown()
    assert peak <= 2
    # 配额耗尽时后续批次必须排队并记录等待
    waited = [w for o in outcomes for w in o.waits if w.waited_ms > 0]
    assert waited, "应存在真实等待观测"
    for wait in (w for o in outcomes for w in o.waits):
        assert wait.wait_end_ms >= wait.wait_start_ms


def test_per_backend_quota():
    scheduler = Scheduler(10, 1, ["b1", "b2"], max_workers=8)
    per_backend_running = {"b1": 0, "b2": 0}
    peak = {"b1": 0, "b2": 0}
    lock = threading.Lock()

    def make_execute(backend_id):
        def execute(payload, timeout_ms):
            with lock:
                per_backend_running[backend_id] += 1
                peak[backend_id] = max(peak[backend_id], per_backend_running[backend_id])
            time.sleep(0.05)
            with lock:
                per_backend_running[backend_id] -= 1
            return {"ok": True}

        return execute

    futures = []
    for i in range(4):
        futures.append(
            scheduler.submit(
                BatchJob(batch_id=f"ba-{i}", backend_id="b1"), make_execute("b1")
            )
        )
        futures.append(
            scheduler.submit(
                BatchJob(batch_id=f"bb-{i}", backend_id="b2"), make_execute("b2")
            )
        )
    for f in futures:
        f.result()
    scheduler.shutdown()
    assert peak["b1"] <= 1
    assert peak["b2"] <= 1


def test_retry_does_not_bypass_quota():
    scheduler = Scheduler(1, 1, ["b1"], max_workers=4)
    attempts = {"slow": 0}
    running = 0
    peak = 0
    lock = threading.Lock()

    def execute(payload, timeout_ms):
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        try:
            if payload.get("flaky") and attempts["slow"] < 2:
                attempts["slow"] += 1
                time.sleep(0.02)
                raise RetryableError("模拟超时")
            time.sleep(0.03)
            return {"ok": True}
        finally:
            with lock:
                running -= 1

    flaky = scheduler.submit(
        BatchJob(batch_id="flaky", backend_id="b1", payload={"flaky": True},
                 max_retries=3),
        execute,
    )
    other = scheduler.submit(BatchJob(batch_id="other", backend_id="b1", payload={}), execute)
    outcome = flaky.result()
    other.result()
    scheduler.shutdown()
    assert outcome.attempts == 3  # 两次超时重试后成功
    assert len(outcome.waits) == 3  # 每次尝试都重新记录等待（重新排队获取配额）
    assert peak <= 1


def test_retry_exhaustion_raises():
    scheduler = Scheduler(1, 1, ["b1"], max_workers=2)

    def always_timeout(payload, timeout_ms):
        raise RetryableError("持续超时")

    future = scheduler.submit(
        BatchJob(batch_id="x", backend_id="b1", max_retries=1), always_timeout
    )
    with pytest.raises(RetryableError):
        future.result()
    scheduler.shutdown()
