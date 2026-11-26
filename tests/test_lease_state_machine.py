from __future__ import annotations

from reco_shard_diag.models import NodeSpec
from reco_shard_diag.node import NodeRuntime


def test_stale_generation_rejected_without_generation_change():
    runtime = NodeRuntime(spec=NodeSpec("n", "127.0.0.1", 1, 2, 2))
    lease = runtime.acquire({"shard_id": "s", "ttl_ms": 100})
    stale = runtime.renew(
        {
            "lease_id": lease["lease_id"],
            "expected_generation": lease["lease_generation"] - 1,
            "ttl_ms": 100,
        }
    )
    assert stale["code"] == "E_LEASE_GENERATION_MISMATCH"
    assert runtime.lease_generation == lease["lease_generation"]
    release = runtime.release(
        {
            "lease_id": lease["lease_id"],
            "expected_generation": lease["lease_generation"],
        }
    )
    assert release["ok"]


def test_expired_lease_is_swept_and_slot_reused():
    now = [1000]
    runtime = NodeRuntime(
        spec=NodeSpec("n", "127.0.0.1", 1, 1, 1), clock_ms=lambda: now[0]
    )
    lease = runtime.acquire({"shard_id": "s", "ttl_ms": 5})
    now[0] += 10
    sweep = runtime.sweep()
    assert sweep["expired"][0]["code"] == "E_LEASE_EXPIRED"
    assert sweep["active_work_units"] == 0
    renew = runtime.renew(
        {
            "lease_id": lease["lease_id"],
            "expected_generation": lease["lease_generation"],
            "ttl_ms": 5,
        }
    )
    assert renew["code"] == "E_LEASE_EXPIRED"
    reused = runtime.acquire({"shard_id": "s", "ttl_ms": 10})
    assert reused["ok"]
    assert reused["lease_generation"] == lease["lease_generation"] + 1


def test_capacity_and_lease_limit_block_extra_work_units():
    runtime = NodeRuntime(spec=NodeSpec("n", "127.0.0.1", 1, 3, 1))
    assert runtime.acquire({"shard_id": "s1", "ttl_ms": 20})["ok"]
    blocked = runtime.acquire({"shard_id": "s2", "ttl_ms": 20})
    assert blocked["code"] == "E_NODE_LEASE_LIMIT"
