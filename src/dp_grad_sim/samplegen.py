from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def _digest(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def generate_sample(
    out_dir: str | Path,
    *,
    seed: int = 20261119,
    workers: int = 3,
    steps: int = 4,
    layers: int = 2,
    hidden_dim: int = 8,
    topology_type: str = "ring",
    link_bandwidth_bytes: int = 4096,
    window_budget_bytes: int = 65536,
) -> dict[str, Path]:
    if workers < 1 or steps < 1 or layers < 1 or hidden_dim < 1:
        raise ValueError("workers, steps, layers, and hidden_dim must be positive")
    if topology_type not in {"ring", "full"}:
        raise ValueError("topology_type must be ring or full")
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {
        "version": 1,
        "seed": seed,
        "model": {
            "layers": layers,
            "hidden_dim": hidden_dim,
            "layer_tensors": [
                {"name": "weight", "shape": ["hidden_dim", "hidden_dim"]},
                {"name": "bias", "shape": ["hidden_dim"]},
            ],
        },
        "steps": steps,
        "data_parallel_workers": workers,
        "compression": {
            "candidates": [
                {"scheme": "symmetric", "bits": [8, 6, 4]},
                {"scheme": "asymmetric", "bits": [8, 6, 4]},
            ],
            "window_steps": 1,
            "max_bytes_per_window": window_budget_bytes,
        },
        "aggregation": {"mode": "mean", "order": "auto"},
        "fidelity_generations": 2,
        "concurrency": 4,
        "snapshot_interval_steps": 1,
        "communication_error_tolerance": 0.001,
        "snapshot_dir": "runtime/snapshots",
    }
    config_path = output / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")

    shard_lines: list[str] = []
    for worker_id in range(workers):
        for shard_index in range(1):
            shard_id = f"w{worker_id}-s{shard_index}"
            shard = {
                "worker_id": worker_id,
                "shard_id": shard_id,
                "samples": 8 + worker_id * 2 + shard_index,
                "feature_digest": _digest("features", seed, worker_id, shard_index),
                "label_digest": _digest("labels", seed, worker_id, shard_index),
            }
            shard_lines.append(json.dumps(shard, sort_keys=True, separators=(",", ":")))
    shards_path = output / "shards.jsonl"
    shards_path.write_text("\n".join(shard_lines) + "\n", encoding="utf-8")

    links: list[dict[str, Any]] = []
    if topology_type == "ring":
        for src in range(workers):
            for dst in ((src + 1) % workers, (src - 1) % workers):
                if src != dst:
                    links.append({"src": src, "dst": dst, "bandwidth_bytes": link_bandwidth_bytes})
    else:
        for src in range(workers):
            for dst in range(workers):
                if src != dst:
                    links.append({"src": src, "dst": dst, "bandwidth_bytes": link_bandwidth_bytes})
    topology = {
        "type": topology_type,
        "workers": workers,
        "link_bandwidth_bytes": link_bandwidth_bytes,
        "links": links,
    }
    topology_path = output / "topology.json"
    topology_path.write_text(json.dumps(topology, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    injection_text = """# This file is inactive unless passed explicitly with --inject.
# Examples:
# nonfinite:
#   - {window: 1, worker_id: 1, tensor: layer_0.weight, index: [0, 0], value: .nan}
# communication_failures:
#   - {window: 1, origin_worker: 0, dst_worker: 1, tensor: layer_0.bias, error_type: InjectedSocketError}
# link_budget_overrides:
#   "0->1": 1
# crash_after_phase: communicated
# send_delay_ms: 25
version: 1
nonfinite: []
communication_failures: []
link_budget_overrides: {}
crash_after_phase: null
send_delay_ms: 0
"""
    injection_path = output / "injection.yaml"
    injection_path.write_text(injection_text, encoding="utf-8")
    return {
        "config": config_path,
        "shards": shards_path,
        "topology": topology_path,
        "injection": injection_path,
    }
