"""Generate a self-contained deterministic sample without external downloads."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from .util import atomic_write_json, digest_obj


def build_sample(seed: int = 20261125, base_port: int | None = None) -> dict[str, dict[str, Any]]:
    rng = random.Random(seed)
    if base_port is None:
        base_port = 41000 + (seed % 1000)
    node_seeds = [seed + 101, seed + 202, seed + 303]
    nodes = {
        "format_version": 1,
        "global_budget": 4,
        "default_lease_ttl_ms": 8000,
        "nodes": [
            {
                "id": "node-a",
                "host": "127.0.0.1",
                "port": base_port,
                "concurrency": 2,
                "lease_limit": 2,
                "seed": node_seeds[0],
                "capacity_version": 1,
            },
            {
                "id": "node-b",
                "host": "127.0.0.1",
                "port": base_port + 1,
                "concurrency": 2,
                "lease_limit": 1,
                "seed": node_seeds[1],
                "capacity_version": 1,
            },
            {
                "id": "node-c",
                "host": "127.0.0.1",
                "port": base_port + 2,
                "concurrency": 1,
                "lease_limit": 1,
                "seed": node_seeds[2],
                "capacity_version": 1,
            },
        ],
    }
    shard_specs = [
        ("s01", "node-a", [], 2, 64),
        ("s02", "node-b", [], 2, 64),
        ("s03", "node-c", [], 1, 48),
        ("s04", "node-a", ["s01"], 2, 56),
        ("s05", "node-b", ["s01", "s02"], 2, 72),
        ("s06", "node-c", ["s02"], 1, 48),
        ("s07", "node-a", ["s04", "s05"], 1, 64),
        ("s08", "node-b", ["s05", "s06"], 1, 56),
        ("s09", "node-c", ["s07", "s08"], 1, 80),
        ("s10", "node-a", [], 1, 32),
    ]
    offsets: dict[str, int] = {}
    shards: list[dict[str, Any]] = []
    for shard_id, node_id, deps, units, sample_count in shard_specs:
        offset = rng.randrange(1, 1_000_000)
        offsets[shard_id] = offset
        shards.append(
            {
                "id": shard_id,
                "node_id": node_id,
                "dependencies": deps,
                "units": units,
                "sample_count": sample_count,
                "seed_offset": offset,
            }
        )
    shard_doc = {"format_version": 1, "shards": shards}
    revisions = {
        "format_version": 1,
        "revisions": [
            {
                "version": 1,
                "parent": None,
                "global_budget": 4,
                "aggregation": {"scale": 1000, "group_modulus": 7},
                "nodes": {
                    "node-a": {"concurrency": 2, "lease_limit": 2, "capacity_version": 1},
                    "node-b": {"concurrency": 2, "lease_limit": 1, "capacity_version": 1},
                    "node-c": {"concurrency": 1, "lease_limit": 1, "capacity_version": 1},
                },
            },
            {
                "version": 2,
                "parent": 1,
                "nodes": {
                    "node-b": {"concurrency": 2, "lease_limit": 2, "capacity_version": 2},
                },
            },
        ],
    }
    events: list[dict[str, Any]] = []
    seq = 1
    for shard in shards:
        events.append(
            {
                "seq": seq,
                "type": "dispatch",
                "shard_id": shard["id"],
                "ttl_ms": 8000,
                "arrival_rule": "stable-shard-id-order",
            }
        )
        seq += 1
    events.append({"seq": seq, "type": "cancel", "shard_id": "s10", "reason": "sample_cancel_leaf"})
    seq += 1
    events.append({"seq": seq, "type": "revision", "version": 2})
    seq += 1
    events.append(
        {
            "seq": seq,
            "type": "quota",
            "node_id": "node-c",
            "concurrency": 2,
            "lease_limit": 2,
        }
    )
    event_doc = {"format_version": 1, "events": events}
    manifest = {
        "format_version": 1,
        "seed": seed,
        "base_port": base_port,
        "rules": {
            "node_seeds": "seed + 101/202/303",
            "ports": "base_port + node index",
            "sample_count": "per-shard fixed table",
            "seed_offset": "Python random.Random(seed).randrange(1, 1000000) in shard order",
            "dependency_order": "stable shard ids",
        },
        "shards": {shard_id: offsets[shard_id] for shard_id in offsets},
    }
    return {
        "nodes": nodes,
        "shards": shard_doc,
        "revisions": revisions,
        "events": event_doc,
        "manifest": manifest,
    }


def write_sample(out_dir: str | Path, seed: int = 20261125, base_port: int | None = None) -> dict[str, Any]:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    docs = build_sample(seed, base_port)
    for name, filename in (
        ("nodes", "nodes.json"),
        ("shards", "shards.json"),
        ("revisions", "revisions.json"),
        ("events", "events.json"),
        ("manifest", "manifest.json"),
    ):
        atomic_write_json(target / filename, docs[name])
    combined = {name: docs[name] for name in ("nodes", "shards", "revisions", "events")}
    atomic_write_json(target / "spec.json", combined)
    digest = digest_obj(combined)
    (target / "sample_digest.txt").write_text(digest + "\n", encoding="utf-8")
    return {"out_dir": str(target), "sample_digest": digest, "base_port": docs["manifest"]["base_port"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="generate deterministic quota-ledger sample")
    parser.add_argument("--out", required=True, help="output sample directory")
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--base-port", type=int, default=None, help="first high loopback port")
    args = parser.parse_args(argv)
    result = write_sample(args.out, args.seed, args.base_port)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
