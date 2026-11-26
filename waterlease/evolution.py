from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .spec import load_spec


def write_spec_directory(raw: dict[str, Any], output_dir: str | Path) -> Path:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    files = {
        "session.json": raw["session"],
        "nodes.json": {"nodes": raw["nodes"]},
        "shards.json": {"shards": raw["shards"]},
        "capacity_versions.json": {"capacity_versions": raw["capacity_versions"]},
        "events.json": {"events": raw["events"]},
    }
    for name, document in files.items():
        (target / name).write_text(
            json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return target


def build_v2_spec(v1_dir: str | Path, output_dir: str | Path) -> tuple[Path, Any]:
    v1 = load_spec(v1_dir)
    raw = copy.deepcopy(v1.raw)
    raw["session"]["spec_version"] = "demo-v2"

    raw["capacity_versions"].append(
        {
            "version": "cap-v2",
            "parent": "cap-v1",
            "budget_units": raw["session"]["global_resource_budget"],
            "revision": 2,
            "note": "chain revision used by replacement shard s2_raw_b",
        }
    )
    for shard in raw["shards"]:
        if shard["shard_id"] == "s2_raw_b":
            shard["sample_rows"] = 128
            shard["capacity_version"] = "cap-v2"
    for node in raw["nodes"]:
        if node["node_id"] == "node-c":
            node["concurrency_quota"] = 1
    next_sequence = max((int(event["sequence"]) for event in raw["events"]), default=0) + 1
    raw["events"].extend(
        [
            {
                "event_id": "evt-v2-capacity",
                "event_type": "capacity_revised",
                "sequence": next_sequence,
                "at_monotonic_ms": 10,
                "capacity_version": "cap-v2",
                "detail": {"parent": "cap-v1"},
            },
            {
                "event_id": "evt-v2-shard",
                "event_type": "shard_replaced",
                "sequence": next_sequence + 1,
                "at_monotonic_ms": 11,
                "shard_id": "s2_raw_b",
                "detail": {"sample_rows": 128},
            },
            {
                "event_id": "evt-v2-quota",
                "event_type": "quota_changed",
                "sequence": next_sequence + 2,
                "at_monotonic_ms": 12,
                "node_id": "node-c",
                "detail": {"concurrency_quota": 1},
            },
        ]
    )
    path = write_spec_directory(raw, output_dir)
    return path, load_spec(path)


def build_crash_spec(v1_dir: str | Path, output_dir: str | Path, *, delay_ms: int = 500) -> tuple[Path, Any]:
    v1 = load_spec(v1_dir)
    raw = copy.deepcopy(v1.raw)
    raw["session"]["spec_version"] = "demo-crash"
    raw["session"]["lease_ttl_ms"] = 5000
    for node in raw["nodes"]:
        node["processing_delay_ms"] = delay_ms
    path = write_spec_directory(raw, output_dir)
    return path, load_spec(path)
