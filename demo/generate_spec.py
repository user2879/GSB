"""Generate the deterministic demo declaration used for acceptance runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_spec() -> dict:
    nodes = [
        {
            "node_id": "node-a",
            "host": "127.0.0.1",
            "port": 19201,
            "seed": 101,
            "initial_capacity_version": 1,
            "capacities": [
                {"version": 1, "concurrency": 2, "lease_limit": 2},
                {"version": 2, "concurrency": 1, "lease_limit": 1},
            ],
        },
        {
            "node_id": "node-b",
            "host": "127.0.0.1",
            "port": 19202,
            "seed": 202,
            "initial_capacity_version": 1,
            "capacities": [
                {"version": 1, "concurrency": 2, "lease_limit": 3},
                {"version": 2, "concurrency": 2, "lease_limit": 2},
            ],
        },
        {
            "node_id": "node-c",
            "host": "127.0.0.1",
            "port": 19203,
            "seed": 303,
            "initial_capacity_version": 1,
            "capacities": [
                {"version": 1, "concurrency": 1, "lease_limit": 2},
                {"version": 2, "concurrency": 3, "lease_limit": 3},
            ],
        },
    ]
    shards = [
        {"shard_id": "s-root", "depends_on": [], "sample_rows": 120,
         "resource_units": 2, "priority": 0},
        {"shard_id": "s-alpha", "depends_on": ["s-root"], "sample_rows": 96,
         "resource_units": 3, "priority": 1},
        {"shard_id": "s-beta", "depends_on": ["s-root"], "sample_rows": 80,
         "resource_units": 4, "priority": 2},
        {"shard_id": "s-gamma", "depends_on": ["s-alpha"], "sample_rows": 64,
         "resource_units": 2, "priority": 1},
        {"shard_id": "s-delta", "depends_on": ["s-alpha", "s-beta"], "sample_rows": 72,
         "resource_units": 3, "priority": 3},
        {"shard_id": "s-epsilon", "depends_on": ["s-gamma", "s-delta"],
         "sample_rows": 48, "resource_units": 2, "priority": 2},
    ]
    events = [
        {"kind": "arrive", "shard_id": "s-root"},
        {"kind": "arrive", "shard_id": "s-alpha"},
        {"kind": "arrive", "shard_id": "s-beta"},
        {"kind": "capacity", "capacity_version": 2},
        {"kind": "arrive", "shard_id": "s-gamma"},
        {"kind": "freeze", "shard_id": "s-delta"},
        {"kind": "arrive", "shard_id": "s-delta"},
        {"kind": "unfreeze", "shard_id": "s-delta"},
        {"kind": "arrive", "shard_id": "s-epsilon"},
    ]
    return {
        "format_version": 1,
        "lease_ttl_ms": 5000,
        "global_budget": {1: 6, 2: 5},
        "nodes": nodes,
        "shards": shards,
        "events": events,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="demo/out/spec.json")
    args = parser.parse_args()
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_spec(), indent=2), encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
