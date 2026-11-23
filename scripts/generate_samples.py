#!/usr/bin/env python3
"""Generate deterministic local JSON/NDJSON sample files for this workspace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_manifest(port_a: int, port_b: int) -> dict:
    return {
        "global_resource_budget": 4,
        "clusters": [
            {
                "id": "alpha",
                "host": "127.0.0.1",
                "port": port_a,
                "nodes": ["alpha-1", "alpha-2"],
                "concurrency": 2,
                "seed": 101,
            },
            {
                "id": "beta",
                "host": "127.0.0.1",
                "port": port_b,
                "nodes": ["beta-1"],
                "concurrency": 1,
                "seed": 202,
            },
        ],
        "tasks": [
            {
                "id": "rollout",
                "config_keys": ["capacity", "scaled"],
                "target_clusters": ["alpha", "beta"],
                "shards": 4,
                "resource_units": 1,
                "concurrency_limit": 2,
            },
            {
                "id": "audit",
                "config_keys": ["threshold"],
                "target_clusters": ["alpha"],
                "shards": 2,
                "resource_units": 1,
                "concurrency_limit": 1,
            },
        ],
        "revisions": [
            {
                "version": 1,
                "parent": None,
                "values": {"capacity": 10, "threshold": 3, "unused": 1},
                "derived": {
                    "scaled": {"op": "sum", "inputs": ["capacity", "threshold"]}
                },
            }
        ],
    }


def build_evolved(manifest: dict) -> dict:
    evolved = json.loads(json.dumps(manifest))
    evolved["revisions"].append(
        {
            "version": 2,
            "parent": 1,
            "values": {"capacity": 14},
        }
    )
    evolved["revisions"].append(
        {
            "version": 3,
            "parent": 2,
            "values": {"unused": 2},
        }
    )
    return evolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="samples")
    parser.add_argument("--alpha-port", type=int, default=43101)
    parser.add_argument("--beta-port", type=int, default=43102)
    args = parser.parse_args()
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(args.alpha_port, args.beta_port)
    evolved = build_evolved(manifest)
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (destination / "manifest.evolved.json").write_text(
        json.dumps(evolved, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    shard = {
        "type": "dispatch",
        "shard_id": "alpha::rollout::s000000",
        "cluster_id": "alpha",
        "task_id": "rollout",
        "shard_index": 0,
        "revision_version": 1,
        "config_keys": ["capacity", "scaled"],
        "config_values": {"capacity": 10, "scaled": 13},
        "seed": 101,
        "sample_count": 64,
        "group_count": 8,
    }
    ndjson = [
        {"type": "hello", "protocol": 1},
        shard,
        {"type": "shutdown"},
    ]
    (destination / "requests.ndjson").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in ndjson),
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "manifest": str(destination / "manifest.json"),
                "evolved_manifest": str(destination / "manifest.evolved.json"),
                "ndjson": str(destination / "requests.ndjson"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
