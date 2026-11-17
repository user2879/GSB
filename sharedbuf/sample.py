"""Deterministic local sample configuration: 3 nodes, 2 buffers x 4 sectors."""
from __future__ import annotations


def sample_config():
    buffers = {"b0": 4, "b1": 4}
    nodes = {"n0": {"port": 25101, "spawn": True},
             "n1": {"port": 25102, "spawn": True},
             "n2": {"port": 25103, "spawn": True}}
    sectors = []
    for bname, count in buffers.items():
        for i in range(count):
            sectors.append(f"{bname}#{i}")
    node_ids = list(nodes)
    owners = {sid: node_ids[i % len(node_ids)] for i, sid in enumerate(sectors)}
    return {
        "seed": 20241118,
        "buffers": buffers,
        "sectors": sectors,
        "sector_index": {sid: i for i, sid in enumerate(sectors)},
        "owners": owners,
        "nodes": nodes,
        "lease_ttl_ms": 5000,
        "global_limit": 4,
        "node_quota": 2,
        "adjudication_timeout_s": 2.0,
        "adjudication_retries": 1,
    }
