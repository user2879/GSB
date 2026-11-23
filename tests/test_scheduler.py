import threading

from shardbench.clock import ManualClock
from shardbench.leases import LeaseManager
from shardbench.scheduler import ProbeScheduler, ProbeTask


def make_scheduler(global_limit=2, per_db_limit=1, ttl=60_000):
    clock = ManualClock(1_000_000)
    leases = LeaseManager(ttl_ms=ttl, clock=clock)
    sched = ProbeScheduler(global_limit, per_db_limit, leases, clock=clock)
    return sched, leases


def test_dual_quotas_never_exceeded_under_contention():
    sched, _ = make_scheduler(global_limit=3, per_db_limit=1)
    gate = threading.Event()
    started = []

    def work():
        started.append(threading.current_thread().name)
        gate.wait(timeout=5)
        return {"ok": True}

    tasks = [ProbeTask(f"t{i}", "u1", db=i % 2, table=0, work=work)
             for i in range(8)]
    import concurrent.futures
    futures = [sched.submit(t) for t in tasks]
    # let some tasks pile up behind the quotas, then release
    threading.Timer(0.2, gate.set).start()
    records = [f.result(timeout=10) for f in futures]
    sched.shutdown()
    stats = sched.stats()
    assert stats["max_observed_global"] <= 3
    assert all(v <= 1 for v in stats["max_observed_per_db"].values())
    assert all(r.status == "ok" for r in records)
    assert any(r.wait_ms >= 0 for r in records)
    # every probe held a lease on its target db while running
    for r in records:
        assert r.lease["db"] == r.db
        assert r.lease["state"] == "released"


def test_wait_window_recorded_in_monotonic_ms():
    sched, _ = make_scheduler(global_limit=1, per_db_limit=1)
    blocker = threading.Event()

    def slow():
        blocker.wait(timeout=5)

    def fast():
        return 1

    t1 = ProbeTask("a", "u", 0, 0, slow)
    t2 = ProbeTask("b", "u", 0, 0, fast)
    f1 = sched.submit(t1)
    f2 = sched.submit(t2)
    threading.Timer(0.1, blocker.set).start()
    r1, r2 = f1.result(timeout=5), f2.result(timeout=5)
    sched.shutdown()
    assert r2.wait_start_ms >= r1.wait_start_ms
    assert r2.wait_end_ms >= r2.wait_start_ms
    assert r2.exec_end_ms >= r2.exec_start_ms


def test_errors_are_recorded_not_raised():
    sched, _ = make_scheduler()

    def boom():
        raise ValueError("injected probe failure")

    rec = sched.run_all([ProbeTask("x", "u", 0, 0, boom)])[0]
    sched.shutdown()
    assert rec.status == "error"
    assert "injected probe failure" in rec.error
