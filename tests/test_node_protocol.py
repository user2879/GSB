import json
import os

import pytest

from edge_cache.config import load_config_dir
from edge_cache.errors import E_LEASE_TAG_MISMATCH
from edge_cache.node_pool import NodePool
from edge_cache.protocol import NodeClient


@pytest.fixture()
def pool(sample_config_dir, tmp_workdir):
    config = load_config_dir(sample_config_dir)
    pool = NodePool(config, os.path.join(tmp_workdir, "specs"))
    pool.start_all()
    try:
        yield config, pool
    finally:
        pool.close()


def test_ping_acquire_compute_release_chain(pool):
    config, running_pool = pool
    node = config.node("node-a")
    client = NodeClient(node.port)
    ping = client.request({"type": "ping"})
    assert ping["ok"] and ping["node_id"] == "node-a"

    acquired = client.request({
        "type": "acquire", "lease_id": "lease-1", "shard_id": "s1", "at_ms": 5
    })
    assert acquired["ok"] and acquired["tag"] == 0

    computed = client.request({
        "type": "compute", "lease_id": "lease-1", "tag": 0,
        "shard_id": "s1", "at_ms": 6
    })
    assert computed["ok"]
    assert len(computed["digest"]) == 64
    assert computed["quota_version"] == node.quota_version
    assert computed["duration_ms"] >= 0

    released = client.request({
        "type": "release", "lease_id": "lease-1", "tag": 0, "at_ms": 7
    })
    assert released["ok"] and released["tag"] == 1


def test_tag_mismatch_rejects_without_advancing_generation(pool):
    config, running_pool = pool
    node = config.node("node-b")
    client = NodeClient(node.port)
    client.request({"type": "acquire", "lease_id": "lease-2",
                    "shard_id": "s2", "at_ms": 1})
    bad_renew = client.request({
        "type": "renew", "lease_id": "lease-2", "tag": 99, "at_ms": 2
    })
    assert not bad_renew["ok"]
    assert bad_renew["error"]["code"] == E_LEASE_TAG_MISMATCH
    assert bad_renew["error"]["context"]["expected"] == 0

    good_renew = client.request({
        "type": "renew", "lease_id": "lease-2", "tag": 0, "at_ms": 3
    })
    assert good_renew["ok"] and good_renew["tag"] == 1

    bad_release = client.request({
        "type": "release", "lease_id": "lease-2", "tag": 0, "at_ms": 4
    })
    assert bad_release["error"]["code"] == E_LEASE_TAG_MISMATCH
    good_release = client.request({
        "type": "release", "lease_id": "lease-2", "tag": 1, "at_ms": 5
    })
    assert good_release["ok"] and good_release["tag"] == 2


def test_lease_limit_enforced_by_node(pool):
    config, running_pool = pool
    node = config.node("node-c")
    client = NodeClient(node.port)
    first = client.request({"type": "acquire", "lease_id": "x1",
                            "shard_id": "s1", "at_ms": 1})
    second = client.request({"type": "acquire", "lease_id": "x2",
                             "shard_id": "s2", "at_ms": 2})
    third = client.request({"type": "acquire", "lease_id": "x3",
                            "shard_id": "s3", "at_ms": 3})
    assert first["ok"] and second["ok"]
    assert not third["ok"] and third["error"]["code"] == "E_LEASE_LIMIT"


def test_snapshot_restore_rebuilds_algebra(pool):
    config, running_pool = pool
    node = config.node("node-a")
    client = NodeClient(node.port)
    client.request({"type": "acquire", "lease_id": "r1",
                    "shard_id": "s1", "at_ms": 1})
    client.request({"type": "renew", "lease_id": "r1", "tag": 0, "at_ms": 2})
    snapshot = client.request({"type": "snapshot"})
    assert snapshot["ok"]
    assert snapshot["leases"][0]["tag"] == 1

    restored = client.request({
        "type": "restore",
        "leases": snapshot["leases"],
        "algebra_log": snapshot["algebra_log"],
    })
    assert restored["ok"] and restored["events"] == 2
    # After restore the same current tag must still be required.
    mismatched = client.request({
        "type": "renew", "lease_id": "r1", "tag": 0, "at_ms": 3
    })
    assert mismatched["error"]["code"] == E_LEASE_TAG_MISMATCH
