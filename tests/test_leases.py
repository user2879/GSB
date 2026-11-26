from edge_cache.clock import DeterministicClock
from edge_cache.config import load_config_dir
from edge_cache.errors import (
    E_BUDGET_EXHAUSTED,
    E_HARD_WATERMARK,
    E_LEASE_EXPIRED,
    E_LEASE_LIMIT,
)
from edge_cache.leases import Lease, LeaseManager


def _manager(sample_config_dir):
    config = load_config_dir(sample_config_dir)
    clock = DeterministicClock(list(range(1, 1000)))
    manager = LeaseManager(
        nodes={node.node_id: node for node in config.nodes},
        global_concurrency=config.global_concurrency,
        clock=clock,
    )
    return config, manager, clock


def _lease(lease_id, node_id, now_ms, ttl=100, slots=1):
    return Lease(
        lease_id=lease_id,
        node_id=node_id,
        shard_id="s1",
        tag=0,
        issued_ms=now_ms,
        expires_ms=now_ms + ttl,
        slots=slots,
    )


def test_expired_lease_is_really_reclaimed(sample_config_dir):
    config, manager, clock = _manager(sample_config_dir)
    manager.register(_lease("l1", "node-a", now_ms=10, ttl=50))
    assert manager.active_lease_count() == 1
    reclaimed = manager.reclaim_expired(now_ms=49)
    assert reclaimed == []
    reclaimed = manager.reclaim_expired(now_ms=60)
    assert len(reclaimed) == 1
    lease, failure = reclaimed[0]
    assert lease.lease_id == "l1"
    assert failure.code == E_LEASE_EXPIRED
    assert manager.active_lease_count() == 0


def test_reclamation_invokes_callback_for_node_release(sample_config_dir):
    config, manager, clock = _manager(sample_config_dir)
    calls = []
    manager.reclamation_callback = lambda lease, now_ms: calls.append((lease.lease_id, now_ms))
    manager.register(_lease("l1", "node-a", now_ms=10, ttl=10))
    manager.reclaim_expired(now_ms=50)
    assert calls == [("l1", 50)]


def test_global_and_node_limits_enforced(sample_config_dir):
    config, manager, clock = _manager(sample_config_dir)
    node = config.node("node-a")
    manager.register(_lease("a1", node.node_id, 10))
    manager.register(_lease("a2", node.node_id, 11))
    rejection = manager.check_admission(node, 1, 12, budget_remaining=10)
    assert rejection is not None and rejection.code == E_LEASE_LIMIT


def test_hard_watermark_rejects_and_soft_watermark_warns(sample_config_dir):
    config, manager, clock = _manager(sample_config_dir)
    node = config.node("node-c")  # capacity 10, soft 500, hard 800
    rejection = manager.check_admission(node, 9, 10, budget_remaining=100)
    assert rejection is not None and rejection.code == E_HARD_WATERMARK
    assert manager.watermarks == []
    rejection = manager.check_admission(node, 6, 11, budget_remaining=100)
    assert rejection is None
    assert len(manager.watermarks) == 1
    assert manager.watermarks[0].level == "SOFT_WATERMARK"


def test_budget_exhaustion_reported(sample_config_dir):
    config, manager, clock = _manager(sample_config_dir)
    node = config.node("node-a")
    rejection = manager.check_admission(node, 1, 10, budget_remaining=0)
    assert rejection is not None and rejection.code == E_BUDGET_EXHAUSTED
