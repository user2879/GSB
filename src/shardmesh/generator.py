"""Generate the deterministic local demo declaration and samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .samples import batch_seed, generate_batch


def demo_spec_dict(*, ports: list[int] | None = None, lease_ttl_ms: int = 2000) -> dict[str, Any]:
    ports = ports or [43181, 43182, 43183]
    return {
        "format_version": 1,
        "seed": 365,
        "host": "127.0.0.1",
        "global_budget": 7,
        "scale": 1000,
        "max_renewals": 1,
        "lease_ttl_ms": lease_ttl_ms,
        "request_timeout_ms": 3000,
        "nodes": [
            {
                "id": "n1",
                "host": "127.0.0.1",
                "port": ports[0],
                "concurrency": 2,
                "lease_limit": 3,
                "seed": 1101,
                "capacity_version": 1,
            },
            {
                "id": "n2",
                "host": "127.0.0.1",
                "port": ports[1],
                "concurrency": 2,
                "lease_limit": 2,
                "seed": 2202,
                "capacity_version": 1,
            },
            {
                "id": "n3",
                "host": "127.0.0.1",
                "port": ports[2],
                "concurrency": 1,
                "lease_limit": 2,
                "seed": 3303,
                "capacity_version": 1,
            },
        ],
        "shards": [
            {"id": "s01", "node_id": "n1", "deps": [], "rows": 512, "units": 2, "priority": 10},
            {"id": "s05", "node_id": "n2", "deps": [], "rows": 512, "units": 2, "priority": 7},
            {"id": "s02", "node_id": "n2", "deps": ["s01"], "rows": 768, "units": 3, "priority": 5},
            {"id": "s03", "node_id": "n1", "deps": ["s01"], "rows": 640, "units": 2, "priority": 8},
            {"id": "s06", "node_id": "n1", "deps": ["s03"], "rows": 512, "units": 1, "priority": 6},
            {"id": "s08", "node_id": "n2", "deps": ["s06"], "rows": 768, "units": 2, "priority": 9},
            {"id": "s04", "node_id": "n3", "deps": ["s02", "s03"], "rows": 1024, "units": 2, "priority": 1},
            {"id": "s07", "node_id": "n3", "deps": ["s04", "s05"], "rows": 1536, "units": 3, "priority": 4},
        ],
        "events": [
            {"seq": 1, "type": "arrive", "shard_id": "s01"},
            {"seq": 2, "type": "arrive", "shard_id": "s05"},
            {"seq": 3, "type": "arrive", "shard_id": "s02"},
            {"seq": 4, "type": "arrive", "shard_id": "s03"},
            {"seq": 5, "type": "arrive", "shard_id": "s06"},
            {"seq": 6, "type": "arrive", "shard_id": "s08"},
            {"seq": 7, "type": "arrive", "shard_id": "s04"},
            {"seq": 8, "type": "arrive", "shard_id": "s07"},
            {"seq": 9, "type": "set_node_quota", "node_id": "n2", "concurrency": 1},
            {"seq": 10, "type": "bump_capacity_version", "node_id": "n3", "version": 2},
        ],
    }


def generate_samples(spec: dict[str, Any]) -> dict[str, Any]:
    shard_index = {shard["id"]: index for index, shard in enumerate(spec["shards"])}
    node_ids = [node["id"] for node in spec["nodes"]]
    node_seed = {node_id: node["seed"] for node_id, node in zip(node_ids, spec["nodes"])}
    samples: dict[str, Any] = {"format_version": 1, "seed": spec.get("seed", 365), "batches": {}}
    for index, shard in enumerate(spec["shards"]):
        node_id = shard.get("node_id") or node_ids[index % len(node_ids)]
        seed = batch_seed(node_seed[node_id], shard_index[shard["id"]], spec.get("seed", 365))
        samples["batches"][shard["id"]] = generate_batch(seed, shard["rows"])
        samples["batches"][shard["id"]]["derived_seed"] = seed
    return samples


def write_demo_files(root: str | Path) -> dict[str, Path]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    spec_path = root / "spec.json"
    samples_path = root / "samples.json"
    spec = demo_spec_dict()
    samples = generate_samples(spec)
    spec_path.write_text(json.dumps(spec, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    samples_path.write_text(json.dumps(samples, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return {"spec": spec_path, "samples": samples_path}
