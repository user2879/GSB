from __future__ import annotations

from typing import Any


def build_example_spec(ports: list[int] | tuple[int, ...] | None = None) -> dict[str, Any]:
    if ports is None:
        from .orchestrator import allocate_free_port

        ports = [allocate_free_port() for _ in range(3)]
    if len(ports) != 3:
        raise ValueError("example requires three ports")
    return {
        "spec_format_version": 1,
        "session": {
            "scale_exponent": 2,
            "lease_ttl_ms": 10_000,
            "handshake_timeout_ms": 1_000,
            "socket_timeout_ms": 5_000,
        },
        "global_budget_units": 6,
        "capacity_chain": [
            {"version": 1, "parent": None},
            {"version": 2, "parent": 1},
            {"version": 3, "parent": 2},
        ],
        "capacity_version": 3,
        "nodes": [
            {
                "node_id": "n1",
                "host": "127.0.0.1",
                "port": int(ports[0]),
                "concurrency_quota": 2,
                "lease_limit": 3,
                "seed": 10101,
                "capacity_version": 2,
                "fixed_point_exponent": 1,
            },
            {
                "node_id": "n2",
                "host": "127.0.0.1",
                "port": int(ports[1]),
                "concurrency_quota": 2,
                "lease_limit": 2,
                "seed": 20202,
                "capacity_version": 3,
                "fixed_point_exponent": 2,
            },
            {
                "node_id": "n3",
                "host": "127.0.0.1",
                "port": int(ports[2]),
                "concurrency_quota": 1,
                "lease_limit": 1,
                "seed": 30303,
                "capacity_version": 1,
                "fixed_point_exponent": 0,
            },
        ],
        "shards": [
            {
                "shard_id": "s1",
                "node_id": "n1",
                "upstreams": [],
                "sample_rows": 24,
                "resource_units": 2,
                "priority": 10,
            },
            {
                "shard_id": "s2",
                "node_id": "n2",
                "upstreams": [],
                "sample_rows": 18,
                "resource_units": 3,
                "priority": 9,
            },
            {
                "shard_id": "s3",
                "node_id": "n1",
                "upstreams": ["s1"],
                "sample_rows": 31,
                "resource_units": 2,
                "priority": 5,
            },
            {
                "shard_id": "s4",
                "node_id": "n3",
                "upstreams": ["s2"],
                "sample_rows": 16,
                "resource_units": 1,
                "priority": 4,
            },
            {
                "shard_id": "s5",
                "node_id": "n2",
                "upstreams": ["s3", "s4"],
                "sample_rows": 27,
                "resource_units": 3,
                "priority": 2,
            },
            {
                "shard_id": "s6",
                "node_id": "n1",
                "upstreams": ["s5"],
                "sample_rows": 12,
                "resource_units": 1,
                "priority": 1,
            },
        ],
        "events": [
            {"seq": 1, "type": "arrival", "shard_id": "s1"},
            {"seq": 2, "type": "arrival", "shard_id": "s2"},
            {"seq": 3, "type": "schedule_tick"},
            {"seq": 4, "type": "arrival", "shard_id": "s3"},
            {"seq": 5, "type": "arrival", "shard_id": "s4"},
            {"seq": 6, "type": "arrival", "shard_id": "s5"},
            {"seq": 7, "type": "arrival", "shard_id": "s6"},
        ],
    }
