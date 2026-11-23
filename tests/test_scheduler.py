import threading
import time

from kvdiag.scheduler import Scheduler


def test_global_and_per_shard_limits_respected():
    sched = Scheduler(max_workers=2, per_shard_quota=1)
    running_global = 0
    max_global = 0
    running_shard = {}
    max_shard = {}
    lock = threading.Lock()

    def task(sid):
        nonlocal running_global, max_global
        with lock:
            running_global += 1
            max_global = max(max_global, running_global)
            running_shard[sid] = running_shard.get(sid, 0) + 1
            max_shard[sid] = max(max_shard.get(sid, 0), running_shard[sid])
        time.sleep(0.05)
        with lock:
            running_global -= 1
            running_shard[sid] -= 1

    futures = [sched.submit("shard-%d" % (i % 2), "replay", task,
                            "shard-%d" % (i % 2)) for i in range(8)]
    for f in futures:
        f.result()
    stats = sched.stats()
    sched.shutdown()
    assert max_global <= 2
    assert stats["max_running_global"] <= 2
    assert all(v <= 1 for v in max_shard.values())
    assert all(v <= 1 for v in stats["max_running_shard"].values())
    assert len(stats["wait_events"]) == 8
    for event in stats["wait_events"]:
        assert event["wait_end_mono_ms"] >= event["wait_start_mono_ms"]


def test_queued_tasks_record_wait_window():
    sched = Scheduler(max_workers=1, per_shard_quota=1)
    blocker = threading.Event()

    def slow():
        blocker.wait(timeout=5)

    first = sched.submit("shard-0", "replay", slow)
    time.sleep(0.05)
    second = sched.submit("shard-0", "replay", lambda: None)
    time.sleep(0.05)  # second task stays queued behind the blocker
    blocker.set()
    first.result()
    second.result()
    stats = sched.stats()
    sched.shutdown()
    waits = sorted(stats["wait_events"],
                   key=lambda e: e["wait_end_mono_ms"] - e["wait_start_mono_ms"])
    longest = waits[-1]
    assert longest["wait_end_mono_ms"] - longest["wait_start_mono_ms"] >= 30
