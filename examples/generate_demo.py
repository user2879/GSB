"""Generate the deterministic demo declaration and NDJSON event stream."""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO = os.path.join(ROOT, "demo")


def build_spec():
    return {
        "global_budget": 6,
        "lease_ttl_ms": 180,
        "handshake_timeout_ms": 3000,
        "io_timeout_ms": 5000,
        "nodes": [
            {
                "node_id": "node-a",
                "host": "127.0.0.1",
                "port": 49101,
                "concurrency": 2,
                "lease_limit": 2,
                "max_renewals": 5,
                "seed": 1001,
                "capacity_version": 1,
                "fixed_scale": 4,
                "compute_iters": 120000,
            },
            {
                "node_id": "node-b",
                "host": "127.0.0.1",
                "port": 49102,
                "concurrency": 1,
                "lease_limit": 2,
                "max_renewals": 0,
                "seed": 2002,
                "capacity_version": 1,
                "fixed_scale": 8,
                "compute_iters": 90000,
            },
            {
                "node_id": "node-c",
                "host": "127.0.0.1",
                "port": 49103,
                "concurrency": 2,
                "lease_limit": 3,
                "max_renewals": 2,
                "seed": 3003,
                "capacity_version": 1,
                "fixed_scale": 2,
                "compute_iters": 45000,
            },
        ],
        "shards": [
            {"shard_id": "s1", "node_id": "node-a", "depends_on": [],
             "resource_units": 2, "sample_count": 120,
             "config_keys": ["alpha"]},
            {"shard_id": "s2", "node_id": "node-a", "depends_on": [],
             "resource_units": 2, "sample_count": 96,
             "config_keys": ["beta"]},
            {"shard_id": "s3", "node_id": "node-b", "depends_on": ["s1"],
             "resource_units": 2, "sample_count": 80,
             "config_keys": ["alpha", "gamma"]},
            {"shard_id": "s4", "node_id": "node-b", "depends_on": ["s2"],
             "resource_units": 2, "sample_count": 72,
             "config_keys": ["beta"]},
            {"shard_id": "s5", "node_id": "node-c", "depends_on": ["s1", "s2"],
             "resource_units": 1, "sample_count": 64,
             "config_keys": ["alpha"]},
            {"shard_id": "s6", "node_id": "node-c", "depends_on": ["s3"],
             "resource_units": 1, "sample_count": 88,
             "config_keys": ["gamma"]},
            {"shard_id": "s7", "node_id": "node-a", "depends_on": ["s4", "s5"],
             "resource_units": 2, "sample_count": 100,
             "config_keys": ["alpha", "beta"]},
            {"shard_id": "s8", "node_id": "node-c", "depends_on": ["s6", "s7"],
             "resource_units": 1, "sample_count": 56,
             "config_keys": ["gamma"]},
        ],
        "revisions": [
            {"version": 1, "parent": None,
             "values": {"alpha": 2, "beta": 3, "gamma": 5}},
            {"version": 2, "parent": 1, "values": {"alpha": 4}},
            {"version": 3, "parent": 2, "values": {"gamma": 7}},
        ],
    }


def build_events():
    return [
        {"type": "dispatch", "shard_id": "s1"},
        {"type": "dispatch", "shard_id": "s2"},
        {"type": "dispatch", "shard_id": "s3"},
        {"type": "dispatch", "shard_id": "s4"},
        {"type": "dispatch", "shard_id": "s5"},
        {"type": "dispatch", "shard_id": "s6"},
        {"type": "dispatch", "shard_id": "s7"},
        {"type": "cancel", "shard_id": "s7"},
        {"type": "dispatch", "shard_id": "s8"},
        {"type": "quota_change", "node_id": "node-b", "concurrency": 2},
        {"type": "config_apply", "revision": 2},
        {"type": "config_apply", "revision": 3},
    ]


def main():
    os.makedirs(DEMO, exist_ok=True)
    spec = build_spec()
    spec_path = os.path.join(DEMO, "demo.spec.json")
    events_path = os.path.join(DEMO, "demo.events.ndjson")
    with open(spec_path, "w", encoding="utf-8") as handle:
        json.dump(spec, handle, ensure_ascii=False, indent=2)
    with open(events_path, "w", encoding="utf-8") as handle:
        for event in build_events():
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(spec_path)
    print(events_path)


if __name__ == "__main__":
    main()
