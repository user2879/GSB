import pytest

from dispatch_tokens.errors import LeaseError
from dispatch_tokens.lease import LeaseRegistry


def make_registry():
    return LeaseRegistry()


def test_issue_is_exclusive_per_node_and_shard():
    registry = make_registry()
    registry.issue("t1", "s1", "n1", 0, demand=2, param_version=1,
                   ttl_ms=100, event_seq=0, basis="test")
    with pytest.raises(LeaseError) as exc:
        registry.issue("t2", "s2", "n1", 10, demand=1, param_version=1,
                       ttl_ms=100, event_seq=1, basis="test")
    assert exc.value.code == "E_LEASE"
    with pytest.raises(LeaseError):
        registry.issue("t3", "s1", "n2", 10, demand=1, param_version=1,
                       ttl_ms=100, event_seq=2, basis="test")


def test_renew_optimistic_version_rejects_stale_and_keeps_expiry():
    registry = make_registry()
    token = registry.issue("t1", "s1", "n1", 0, demand=2, param_version=1,
                           ttl_ms=100, event_seq=0, basis="test")
    before = token.expires_ms
    with pytest.raises(LeaseError) as exc:
        registry.renew("t1", expected_version=99, now_ms=20, ttl_ms=100,
                       event_seq=1, param_version=1, node_check_ok=True)
    assert exc.value.code == "E_LEASE"
    assert token.expires_ms == before
    assert token.renew_count == 0
    renewed = registry.renew("t1", expected_version=1, now_ms=20,
                             ttl_ms=100, event_seq=2, param_version=1,
                             node_check_ok=True)
    assert renewed.token_version == 2
    assert renewed.renew_count == 1
    assert renewed.expires_ms == 120


def test_node_check_failure_rejects_without_extension():
    registry = make_registry()
    token = registry.issue("t1", "s1", "n1", 0, 2, 1, 100, 0, "test")
    before = token.expires_ms
    with pytest.raises(LeaseError):
        registry.renew("t1", 1, 20, 100, 1, 1, node_check_ok=False)
    assert token.expires_ms == before


def test_expiry_uses_monotonic_clock():
    registry = make_registry()
    token = registry.issue("t1", "s1", "n1", 0, 2, 1, 100, 0, "test")
    assert registry.expire_if_due(50, 1, 1) == []
    expired = registry.expire_if_due(100, 2, 1)
    assert len(expired) == 1
    assert token.active is False
    # 释放后节点可再被占用。
    registry.issue("t2", "s2", "n1", 100, 1, 1, 100, 3, "reuse")


def test_cancel_reclaims_once_and_records_decision():
    registry = make_registry()
    registry.issue("t1", "s1", "n1", 0, 2, 1, 100, 0, "test")
    token = registry.cancel("s1", 1, 1, "cancel_reclaims_held_quota_once")
    assert token.active is False
    decisions = [d for d in registry.decisions if d.decision == "CANCEL"]
    assert len(decisions) == 1
    # 再次取消不会产生第二次回收。
    with pytest.raises(LeaseError):
        registry.cancel("s1", 2, 1, "again")


def test_stable_event_order_uses_shard_id_then_type():
    class E:
        def __init__(self, seq, shard_id, kind):
            self.seq = seq
            self.shard_id = shard_id
            self.kind = kind
            self.node_id = None

    events = [E(1, "s2", "revise"), E(1, "s1", "dispatch"),
              E(1, "s1", "cancel"), E(0, "s9", "dispatch")]
    ordered = LeaseRegistry.stable_event_order(events)
    assert [(e.seq, e.shard_id, e.kind) for e in ordered] == [
        (0, "s9", "dispatch"),
        (1, "s1", "dispatch"),
        (1, "s1", "cancel"),
        (1, "s2", "revise"),
    ]
