"""引擎三态判决、拒绝三分类与稳定唤醒。"""
from quota_arbiter import (
    REASON_BURST,
    REASON_POOL_CAPACITY,
    REASON_TENANT_QUOTA,
    VERDICT_GRANT,
    VERDICT_REJECT,
)
from quota_arbiter.engine import ArbiterEngine
from quota_arbiter.models import Pool, Request, Tenant


def _engine():
    tenants = [
        Tenant("A", quota=10, concurrency_cap=1, burst=4),
        Tenant("B", quota=100, concurrency_cap=2, burst=0),
    ]
    pools = [Pool("P1", capacity=6, priority_order=("A", "B"))]
    return ArbiterEngine(tenants, pools), tenants, pools


def _run(requests, tenants, pools):
    engine = ArbiterEngine(tenants, pools)
    for req in sorted(requests, key=lambda r: (r.arrival, r.rid)):
        engine.process_request(req)
    engine.finalize()
    return engine


def test_tenant_quota_exhausted():
    # 租户 A 配额仅 10：第一条授予后余量 2，第二条需求 3 触发配额耗尽拒绝
    tenants = [Tenant("A", quota=10, concurrency_cap=5, burst=0)]
    pools = [Pool("P1", capacity=100, priority_order=("A",))]
    reqs = [
        Request("r1", "A", "P1", demand=8, arrival=0, priority=0),
        Request("r2", "A", "P1", demand=3, arrival=1, priority=0),
    ]
    engine = _run(reqs, tenants, pools)
    assert engine.outcomes["r1"].verdict == VERDICT_GRANT
    assert engine.outcomes["r2"].verdict == VERDICT_REJECT
    assert engine.outcomes["r2"].reason == REASON_TENANT_QUOTA


def test_pool_capacity_exhausted_no_burst():
    tenants = [Tenant("B", quota=100, concurrency_cap=5, burst=0)]
    pools = [Pool("P1", capacity=5, priority_order=("B",))]
    reqs = [
        Request("r1", "B", "P1", demand=5, arrival=0, priority=0),
        Request("r2", "B", "P1", demand=1, arrival=1, priority=0),
    ]
    engine = _run(reqs, tenants, pools)
    assert engine.outcomes["r1"].verdict == VERDICT_GRANT
    assert engine.outcomes["r2"].verdict == VERDICT_REJECT
    assert engine.outcomes["r2"].reason == REASON_POOL_CAPACITY


def test_burst_insufficient():
    tenants = [Tenant("A", quota=100, concurrency_cap=5, burst=2)]
    pools = [Pool("P1", capacity=5, priority_order=("A",))]
    reqs = [
        Request("r1", "A", "P1", demand=5, arrival=0, priority=0),
        Request("r2", "A", "P1", demand=4, arrival=1, priority=0),
    ]
    engine = _run(reqs, tenants, pools)
    assert engine.outcomes["r1"].verdict == VERDICT_GRANT
    assert engine.outcomes["r2"].verdict == VERDICT_REJECT
    assert engine.outcomes["r2"].reason == REASON_BURST


def test_burst_grant_when_enough():
    tenants = [Tenant("A", quota=100, concurrency_cap=5, burst=5)]
    pools = [Pool("P1", capacity=5, priority_order=("A",))]
    reqs = [
        Request("r1", "A", "P1", demand=5, arrival=0, priority=0),
        Request("r2", "A", "P1", demand=5, arrival=1, priority=0),
    ]
    engine = _run(reqs, tenants, pools)
    assert engine.outcomes["r2"].verdict == VERDICT_GRANT
    assert engine.outcomes["r2"].basis.endswith("|BURST")
    assert engine.burst_remaining["A"] == 0


def test_concurrency_cap_enqueues_then_wakes():
    tenants = [Tenant("A", quota=100, concurrency_cap=1, burst=0)]
    pools = [Pool("P1", capacity=100, priority_order=("A",))]
    reqs = [
        Request("r1", "A", "P1", demand=5, arrival=0, priority=0),
        Request("r2", "A", "P1", demand=5, arrival=1, priority=0),
    ]
    engine = _run(reqs, tenants, pools)
    # r2 先排队，r1 完成释放并发位后被稳定唤醒为授予
    assert any(e.rid == "r2" and e.kind == "ENQUEUE" for e in engine.queue_events)
    assert any(e.rid == "r2" and e.kind == "WAKE" for e in engine.queue_events)
    assert engine.outcomes["r2"].verdict == VERDICT_GRANT
    assert engine.outcomes["r2"].enqueued is True
    assert engine.outcomes["r2"].wait_ms > 0


def test_wake_uses_stable_priority_rule():
    # 两租户并发上限均为 1，先各占一个在途事务；后续请求因并发上限排队，
    # 待释放后按资源池优先级序（hi 高于 lo）稳定唤醒。
    tenants = [
        Tenant("hi", quota=100, concurrency_cap=1, burst=0),
        Tenant("lo", quota=100, concurrency_cap=1, burst=0),
    ]
    pools = [Pool("P1", capacity=100, priority_order=("hi", "lo"))]
    reqs = [
        Request("occ_hi", "hi", "P1", 1, 0, 0),
        Request("occ_lo", "lo", "P1", 1, 1, 0),
        # lo 先排队、hi 后排队；唤醒时 hi 池优先级更高，应先出队
        Request("q_lo", "lo", "P1", 1, 2, 0),
        Request("q_hi", "hi", "P1", 1, 3, 0),
    ]
    engine = _run(reqs, tenants, pools)
    wakes = [e.rid for e in engine.queue_events if e.kind == "WAKE"]
    assert "q_hi" in wakes and "q_lo" in wakes
    assert wakes.index("q_hi") < wakes.index("q_lo")


def test_one_outcome_per_request():
    engine, tenants, pools = _engine()
    reqs = [Request(f"r{i}", "A", "P1", 1, i, 0) for i in range(6)]
    engine = _run(reqs, tenants, pools)
    assert len(engine.outcomes) == len(reqs)


def test_wake_independent_of_internal_queue_order():
    tenants = [Tenant(f"T{i}", quota=1000, concurrency_cap=1, burst=0) for i in range(5)]
    pools = [Pool("P", capacity=1, priority_order=tuple(f"T{i}" for i in range(5)))]

    def wake_sequence(reverse_internal):
        engine = ArbiterEngine(tenants, pools)
        holder = Request("holder", "T0", "P", 1, 0, 0)
        engine.process_request(holder)
        queued = [
            Request(f"q{i}", f"T{i}", "P", 1, 1 + i, priority=i % 3) for i in range(5)
        ]
        for req in queued:
            engine.requests_by_id[req.rid] = req
            engine.quota_remaining[req.tid] -= 1
            engine._enqueue(req, req.arrival, "TEST")
        if reverse_internal:
            engine.queues["P"].reverse()
        engine.inflight["T0"] -= 1
        engine.pool_remaining["P"] += 1
        while engine.queues["P"]:
            engine._wake_pool("P", engine._now + 1)
            if engine.active:
                complete_at, rid = min(engine.active)
                req = engine.requests_by_id[rid]
                if rid in {r.rid for r in queued}:
                    engine.active.remove((complete_at, rid))
                    engine.inflight[req.tid] -= 1
                    engine.pool_remaining["P"] += 1
                    engine._now += 1
                else:
                    break
            else:
                break
        return [e.rid for e in engine.queue_events if e.kind == "WAKE"]

    assert wake_sequence(False) == wake_sequence(True)
