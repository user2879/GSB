from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .samples import write_event_stream
from .spec import parse_session


def demo_declaration(*, crash: bool = False) -> dict[str, Any]:
    fault: dict[str, Any] = {}
    if crash:
        fault = {
            "delay_ms_by_shard": {"s1": 300},
            "kill_after_commits": 1,
        }
    return {
        "schema_version": 1,
        "session_id": "quotalab-demo",
        "scale_power": 1,
        "global_budget_units": 6,
        "lease_ttl_ms": 5000,
        "lease_max_renewals": 1,
        "socket_timeout_ms": 1000,
        "ready_timeout_ms": 5000,
        "worker_count": 8,
        "tolerances_q": {
            "shard_q": 3,
            "node_watermark_q": 1,
            "global_budget_q": 1,
            "dominance_threshold_bp": 5000,
        },
        "accounting": {"pressure_den": 8, "group_count": 8},
        "fault_injection": fault,
        "nodes": [
            {
                "node_id": "n1",
                "host": "127.0.0.1",
                "port": 19161,
                "concurrency_quota": 2,
                "lease_limit": 2,
                "seed": 1001001,
                "capacity_version": 1,
            },
            {
                "node_id": "n2",
                "host": "127.0.0.1",
                "port": 19162,
                "concurrency_quota": 2,
                "lease_limit": 2,
                "seed": 2002002,
                "capacity_version": 1,
            },
        ],
    }


def demo_shards() -> dict[str, Any]:
    return {
        "shards": [
            {
                "shard_id": "s1",
                "node_id": "n1",
                "depends_on": [],
                "sample_rows": 48,
                "resource_units": 2,
                "priority": 100,
                "planned_renewals": 1,
            },
            {
                "shard_id": "s2",
                "node_id": "n2",
                "depends_on": [],
                "sample_rows": 44,
                "resource_units": 2,
                "priority": 90,
            },
            {
                "shard_id": "s3",
                "node_id": "n1",
                "depends_on": [],
                "sample_rows": 36,
                "resource_units": 2,
                "priority": 80,
            },
            {
                "shard_id": "s4",
                "node_id": "n2",
                "depends_on": [],
                "sample_rows": 32,
                "resource_units": 2,
                "priority": 70,
            },
            {
                "shard_id": "s5",
                "node_id": "n1",
                "depends_on": ["s3"],
                "sample_rows": 24,
                "resource_units": 1,
                "priority": 60,
            },
            {
                "shard_id": "s6",
                "node_id": "n2",
                "depends_on": ["s2", "s5"],
                "sample_rows": 20,
                "resource_units": 2,
                "priority": 50,
            },
        ]
    }


def capacity_document(declaration: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "capacity_versions": [
            {"node_id": node["node_id"], "capacity_version": node["capacity_version"]}
            for node in declaration["nodes"]
        ],
    }


def write_demo_files(root: str | Path) -> dict[str, str]:
    root = Path(root)
    config_dir = root / "configs"
    sample_dir = root / "samples"
    config_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    declaration = demo_declaration()
    crash_declaration = demo_declaration(crash=True)
    shards = demo_shards()
    capacity = capacity_document(declaration)

    paths = {
        "declaration": config_dir / "declaration.json",
        "crash_declaration": config_dir / "declaration.crash.json",
        "shards": config_dir / "shards.json",
        "capacity": config_dir / "capacity.json",
        "events": sample_dir / "event_stream.jsonl",
    }
    paths["declaration"].write_text(json.dumps(declaration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    paths["crash_declaration"].write_text(
        json.dumps(crash_declaration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    paths["shards"].write_text(json.dumps(shards, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    paths["capacity"].write_text(json.dumps(capacity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    spec = parse_session(declaration, shards, capacity_doc=capacity)
    event_count = write_event_stream(spec, paths["events"])
    return {key: str(path) for key, path in paths.items()} | {"event_count": str(event_count)}


def write_evolution_files(root: str | Path) -> dict[str, str]:
    root = Path(root)
    config_dir = root / "configs"
    declaration_v2 = demo_declaration()
    declaration_v2["nodes"][0]["capacity_version"] = 2
    capacity_v2 = capacity_document(declaration_v2)

    declaration_v2_path = config_dir / "declaration.v2.json"
    capacity_v2_path = config_dir / "capacity.v2.json"
    declaration_v2_path.write_text(
        json.dumps(declaration_v2, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    capacity_v2_path.write_text(
        json.dumps(capacity_v2, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "declaration_v2": str(declaration_v2_path),
        "capacity_v2": str(capacity_v2_path),
    }
