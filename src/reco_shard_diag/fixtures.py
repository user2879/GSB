from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .client import reserve_ports


def demo_config_dict(ports: list[int], *, include_events: bool = True) -> dict[str, Any]:
    if len(ports) != 3:
        raise ValueError("demo config requires three ports")
    nodes = [
        {
            "node_id": "node-a",
            "host": "127.0.0.1",
            "port": ports[0],
            "capacity_slots": 4,
            "lease_limit": 4,
        },
        {
            "node_id": "node-b",
            "host": "127.0.0.1",
            "port": ports[1],
            "capacity_slots": 3,
            "lease_limit": 3,
        },
        {
            "node_id": "node-c",
            "host": "127.0.0.1",
            "port": ports[2],
            "capacity_slots": 2,
            "lease_limit": 2,
        },
    ]
    quotas = [
        {
            "version": 1,
            "parent_version": None,
            "capacity_slots": {"node-a": 3, "node-b": 2, "node-c": 2},
            "lease_limits": {"node-a": 3, "node-b": 2, "node-c": 2},
            "soft_watermark_bp": 7000,
            "hard_watermark_bp": 10000,
            "width": 16,
            "points_per_resource_unit": 60,
        },
        {
            "version": 2,
            "parent_version": 1,
            "capacity_slots": {"node-a": 4, "node-b": 3, "node-c": 2},
            "lease_limits": {"node-a": 4, "node-b": 3, "node-c": 2},
            "soft_watermark_bp": 7000,
            "hard_watermark_bp": 10000,
            "width": 16,
            "points_per_resource_unit": 60,
        },
        {
            "version": 3,
            "parent_version": 2,
            "capacity_slots": {"node-a": 5, "node-b": 4, "node-c": 3},
            "lease_limits": {"node-a": 5, "node-b": 4, "node-c": 3},
            "soft_watermark_bp": 8000,
            "hard_watermark_bp": 10000,
            "width": 18,
            "points_per_resource_unit": 55,
        },
    ]
    shard_a = {
        "shard_id": "shard-a",
        "node_id": "node-a",
        "upstream": [],
        "dimensions": 3,
        "groups": ["g0", "g1", "g2"],
        "batches": [
            {
                "batch_id": "a1",
                "sample_count": 30,
                "value_bound": 900,
                "group_keys": ["g0", "g1", "g2"],
            }
        ],
    }
    shard_b = {
        "shard_id": "shard-b",
        "node_id": "node-a",
        "upstream": ["shard-a"],
        "dimensions": 3,
        "groups": ["g0", "g1", "g2"],
        "batches": [
            {
                "batch_id": "b1",
                "sample_count": 24,
                "value_bound": 800,
                "group_keys": ["g0", "g1", "g2"],
            }
        ],
    }
    shard_c = {
        "shard_id": "shard-c",
        "node_id": "node-b",
        "upstream": ["shard-b"],
        "dimensions": 3,
        "groups": ["g0", "g1", "g2"],
        "batches": [
            {
                "batch_id": "c1",
                "sample_count": 18,
                "value_bound": 700,
                "group_keys": ["g0", "g1", "g2"],
            }
        ],
    }
    shard_d = {
        "shard_id": "shard-d",
        "node_id": "node-c",
        "upstream": ["shard-c"],
        "dimensions": 2,
        "groups": ["g0", "g1"],
        "batches": [
            {
                "batch_id": "d1",
                "sample_count": 12,
                "value_bound": 600,
                "group_keys": ["g0", "g1"],
            }
        ],
    }
    shard_e = {
        "shard_id": "shard-e",
        "node_id": "node-b",
        "upstream": [],
        "dimensions": 2,
        "groups": ["g0", "g1"],
        "batches": [
            {
                "batch_id": "e1",
                "sample_count": 10,
                "value_bound": 500,
                "group_keys": ["g0", "g1"],
            }
        ],
    }
    events: list[dict[str, Any]] = [
        {
            "seq": 1,
            "type": "batch_append",
            "payload": {
                "shard_id": "shard-a",
                "batch": {
                    "batch_id": "a2",
                    "sample_count": 18,
                    "value_bound": 700,
                    "group_keys": ["g0", "g1", "g2"],
                },
            },
        },
        {
            "seq": 2,
            "type": "shard_replace",
            "payload": {
                "shard_id": "shard-c",
                "shard": {
                    "shard_id": "shard-c",
                    "node_id": "node-b",
                    "upstream": ["shard-b"],
                    "dimensions": 2,
                    "groups": ["g0", "g1", "g2"],
                    "batches": [
                        {
                            "batch_id": "c1",
                            "sample_count": 20,
                            "value_bound": 750,
                            "group_keys": ["g0", "g1", "g2"],
                        }
                    ],
                },
            },
        },
        {
            "seq": 3,
            "type": "dependency_update",
            "payload": {"shard_id": "shard-d", "upstream": ["shard-a", "shard-c"]},
        },
        {
            "seq": 4,
            "type": "quota_version_add",
            "payload": {
                "activate": False,
                "quota": {
                    "version": 4,
                    "parent_version": 3,
                    "capacity_slots": {"node-a": 6, "node-b": 5, "node-c": 4},
                    "lease_limits": {"node-a": 5, "node-b": 4, "node-c": 3},
                    "soft_watermark_bp": 8500,
                    "hard_watermark_bp": 10000,
                    "width": 8,
                    "points_per_resource_unit": 50,
                },
            },
        },
        {
            "seq": 5,
            "type": "quota_rollback",
            "payload": {"target_version": 2},
        },
        {
            "seq": 6,
            "type": "node_adjust",
            "payload": {"node_id": "node-c", "capacity_slots": 3, "lease_limit": 3},
        },
        {"seq": 7, "type": "width_switch", "payload": {"width": 12}},
        {"seq": 8, "type": "shard_delete", "payload": {"shard_id": "shard-e"}},
        {
            "seq": 9,
            "type": "batch_append",
            "payload": {
                "shard_id": "shard-b",
                "batch": {
                    "batch_id": "b2",
                    "sample_count": 12,
                    "value_bound": 650,
                    "group_keys": ["g0", "g1", "g2"],
                },
            },
        },
    ]
    return {
        "global_version": 1,
        "seed": 20261127,
        "resource_budget": 30,
        "max_batch_points": 500,
        "reference_width": 17,
        "active_quota_version": 2,
        "lease_ttl_ms": 3000,
        "nodes": nodes,
        "quota_versions": quotas,
        "shards": [shard_a, shard_b, shard_c, shard_d, shard_e],
        "events": events if include_events else [],
        "injections": [
            {"injection_id": "I_DEPENDENCY_CYCLE", "kind": "dependency_cycle"},
            {"injection_id": "I_QUOTA_PARENT_MISSING", "kind": "quota_missing_parent"},
            {"injection_id": "I_SAMPLE_DIMENSION", "kind": "sample_dimension"},
            {"injection_id": "I_STALE_LEASE", "kind": "stale_lease"},
            {"injection_id": "I_EXPIRED_RENEW", "kind": "expired_renew"},
        ],
    }


def build_demo_config(*, include_events: bool = True) -> dict[str, Any]:
    return demo_config_dict(reserve_ports(3), include_events=include_events)


def write_workspace(path: Path, *, include_events: bool = True, force: bool = False) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    config_path = path / "config.json"
    if config_path.exists() and not force:
        return config_path
    raw = build_demo_config(include_events=include_events)
    config_path.write_text(
        json.dumps(raw, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    events_path = path / "events.jsonl"
    events_path.write_text(
        "".join(
            json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
            for event in raw["events"]
        ),
        encoding="utf-8",
    )
    injections_path = path / "injections.json"
    injections_path.write_text(
        json.dumps(raw["injections"], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    from .models import ProjectConfig
    from .state import materialize

    state = materialize(ProjectConfig.from_raw(raw))
    samples_path = path / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as handle:
        for shard_id in sorted(state.shards):
            for sample in state.samples(shard_id):
                handle.write(
                    json.dumps(
                        {"shard_id": shard_id, **sample},
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
    return config_path
