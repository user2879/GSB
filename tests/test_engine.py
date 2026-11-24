from __future__ import annotations

from faultisolation.clock import LogicalClock
from faultisolation.engine import Engine, canonical_event_order
from faultisolation.errors import LeaseError
from faultisolation.replay import replay_path, verify_paths, verify_slot_uniqueness
from faultisolation.spec import DomainSpec, Event, InstanceSpec, Session, SpecBundle


def _bundle(tmp_path, global_cap=2):
    domains = [
        DomainSpec("d1", "127.0.0.1", 19001, slots=3, lease_cap=2,
                   lease_ms=100, renew_ms=30, seed=1),
    ]
    instances = [
        InstanceSpec("i1", "d1", 1, 2),
        InstanceSpec("i2", "d1", 2, 1),
        InstanceSpec("i3", "d1", 3, 1),
    ]
    events = [
        Event(1, 0, "i1", "acquire"),
        Event(2, 10, "i2", "acquire"),
        Event(3, 20, "i3", "acquire"),  # 配额/上限受限，排队
        Event(4, 120, "i1", "release"),  # i1 到期/释放后泵取 i3
        Event(5, 130, "i3", "release"),
    ]
    session = Session(global_lease_cap=global_cap, scale_power=2, pacing=0.0, batch_size=2)
    return SpecBundle(domains, instances, events, session,
                      tmp_path / "s.json", tmp_path / "e.ndjson", tmp_path / "session.json")


def test_queue_fifo_no_priority_jump(tmp_path):
    bundle = _bundle(tmp_path)
    engine = Engine(bundle, LogicalClock(0))
    outcomes = engine.run_event_path()
    by_seq = {o.seq: o for o in outcomes}
    # i3 优先级最低且容量不足，必须等待，不能靠优先级插队
    assert by_seq[3].status == "granted"
    assert by_seq[3].wait_start_ms == 20 and by_seq[3].wait_end_ms >= 120
    # 等待记录存在
    assert any(w.seq == 3 for w in engine.waits)


def test_slot_never_aliased(tmp_path):
    bundle = _bundle(tmp_path)
    result = verify_slot_uniqueness(bundle)
    assert result["ok"] is True


def test_three_paths_byte_identical(small_bundle):
    result = verify_paths(small_bundle)
    assert result["byte_identical"] is True
    assert len(set(result["byte_hashes"].values())) == 1


def test_optimistic_version_mismatch_rejects_without_extension(tmp_path):
    bundle = _bundle(tmp_path)
    engine = Engine(bundle, LogicalClock(0))
    engine.process_event(bundle.events[0])  # i1 granted v1
    lease = engine._find_active_lease("i1", "d1")
    arbiter = engine.arbiters["d1"]
    # 错误版本号续约 -> E_LEASE，且到期时刻不变
    with __import__("pytest").raises(LeaseError) as exc:
        arbiter.renew(lease.lease_id, version=99, now=50)
    assert exc.value.code == "E_LEASE"
    assert arbiter.active[lease.lease_id].expiry_ms == lease.expiry_ms
    assert arbiter.active[lease.lease_id].version == 1


def test_expiry_uses_clock_not_sleep(tmp_path):
    bundle = _bundle(tmp_path)
    engine = Engine(bundle, LogicalClock(0))
    engine.process_event(bundle.events[0])
    lease = engine._find_active_lease("i1", "d1")
    assert lease.state == "granted"
    # 推进逻辑时钟超过到期，下一个事件触发回收
    engine.process_event(Event(99, 101, "i2", "acquire"))
    assert lease.state == "expired"


def test_canonical_order_uses_priority_then_id(tmp_path):
    bundle = _bundle(tmp_path)
    events = [
        Event(2, 5, "i2", "acquire"),
        Event(1, 5, "i1", "acquire"),
    ]
    ordered = canonical_event_order(events, {"i1": 1, "i2": 2})
    assert [e.instance_id for e in ordered] == ["i1", "i2"]


def test_denied_lease_for_unknown_renew_release(small_bundle):
    outcomes = replay_path(small_bundle, "event")
    denied = [o for o in outcomes if o.status == "denied_lease"]
    assert denied, "样本应包含对无活跃租约实例的续约/释放拒绝"


def test_alias_error_on_double_slot_holding(tmp_path):
    from faultisolation.errors import AliasError
    from faultisolation.spec import DomainSpec, InstanceSpec
    from faultisolation.clock import LogicalClock
    from faultisolation.engine import DomainArbiter
    dom = DomainSpec("d1", "127.0.0.1", 19001, 2, 2, 100, 30, 1)
    arb = DomainArbiter(dom, LogicalClock(0))
    inst = InstanceSpec("i1", "d1", 1, 1)
    arb.acquire("L1", inst, 0)
    # 人为破坏槽位归属表，模拟同一槽位被两个租约声明，必须 E_ALIAS
    arb.slot_owner[0] = "L9"
    with __import__("pytest").raises(AliasError) as exc:
        arb.release("L1", 10)
    assert exc.value.code == "E_ALIAS"
    assert exc.value.lease_a == "L9" and exc.value.lease_b == "L1"
    assert exc.value.slot == 0
