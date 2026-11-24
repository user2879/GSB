from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

from .hashing import canonical_dumps


def find_free_ports(count: int) -> list[int]:
    sockets: list[socket.socket] = []
    ports: list[int] = []
    for _ in range(count):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        sockets.append(sock)
        ports.append(int(sock.getsockname()[1]))
    for sock in sockets:
        sock.close()
    return ports


def build_sample(ports: list[int] | None = None) -> dict[str, Any]:
    if ports is None:
        ports = find_free_ports(4)
    if len(ports) != 4:
        raise ValueError("sample requires four ports")

    config: dict[str, Any] = {
        "schema_version": "workbench.config.v1",
        "clock_mode": "seeded",
        "capacity_policies": [
            {"capacity_version": 1, "soft_permille": 600, "hard_permille": 800},
            {"capacity_version": 2, "soft_permille": 300, "hard_permille": 500},
        ],
        "nodes": [
            {
                "node_id": "n-core-a",
                "port": ports[0],
                "capacity_slots": 10,
                "lease_limit": 4,
                "seed": 1101,
                "capacity_version": 1,
                "quota_version": 1,
            },
            {
                "node_id": "n-core-b",
                "port": ports[1],
                "capacity_slots": 10,
                "lease_limit": 4,
                "seed": 2202,
                "capacity_version": 2,
                "quota_version": 1,
            },
            {
                "node_id": "n-water",
                "port": ports[2],
                "capacity_slots": 10,
                "lease_limit": 8,
                "seed": 3303,
                "capacity_version": 1,
                "quota_version": 1,
            },
            {
                "node_id": "n-lease",
                "port": ports[3],
                "capacity_slots": 5,
                "lease_limit": 5,
                "seed": 4404,
                "capacity_version": 1,
                "quota_version": 1,
            },
        ],
        "manifest": {
            "schema_version": "feature-manifest.v1",
            "stages": [
                {
                    "stage_id": "source_profile",
                    "artifact_key": "feat:user_profile",
                    "needs": [],
                    "concurrency_quota": 2,
                },
                {
                    "stage_id": "window_builder",
                    "artifact_key": "feat:window_7d",
                    "needs": ["feat:user_profile"],
                    "concurrency_quota": 3,
                },
                {
                    "stage_id": "rank_builder",
                    "artifact_key": "feat:rank_score",
                    "needs": ["feat:user_profile", "feat:window_7d"],
                    "concurrency_quota": 10,
                },
            ],
        },
    }

    events: list[dict[str, Any]] = [
        {
            "seq": 1,
            "type": "WRITE",
            "node_ids": ["n-core-a", "n-core-b"],
            "key": "feat:user_profile",
            "stage_id": "source_profile",
            "content": {"user_id": "u-1001", "embedding": [1, 2, 3], "version": "v1"},
            "priority": 2,
            "ttl_ms": 120000,
            "now_ms": 1000,
        },
        {
            "seq": 2,
            "type": "WRITE",
            "node_id": "n-core-a",
            "key": "feat:user_profile",
            "stage_id": "source_profile",
            "content": {"user_id": "u-1001", "embedding": [1, 2, 4], "version": "v2-overwrite"},
            "priority": 2,
            "ttl_ms": 120000,
            "now_ms": 1001,
        },
        {
            "seq": 3,
            "type": "WRITE",
            "node_id": "n-core-a",
            "key": "feat:window_7d",
            "stage_id": "window_builder",
            "content": {"windows": [7, 30], "clicks": 42},
            "priority": 1,
            "ttl_ms": 120000,
            "now_ms": 1002,
        },
        {
            "seq": 4,
            "type": "READ",
            "node_id": "n-core-b",
            "key": "feat:user_profile",
            "now_ms": 1003,
        },
        {"seq": 10, "type": "WRITE", "node_id": "n-lease", "key": "lease:high", "content": "high-priority", "priority": 5, "ttl_ms": 200, "now_ms": 2000},
        {"seq": 11, "type": "WRITE", "node_id": "n-lease", "key": "lease:middle", "content": "middle-priority", "priority": 1, "ttl_ms": 150, "now_ms": 2000},
        {"seq": 12, "type": "WRITE", "node_id": "n-lease", "key": "lease:renewed", "content": "will-renew", "priority": 1, "ttl_ms": 100, "now_ms": 2000},
        {"seq": 13, "type": "WRITE", "node_id": "n-lease", "key": "lease:late", "content": "late-renew", "priority": 0, "ttl_ms": 100, "now_ms": 2000},
        {"seq": 14, "type": "LEASE_RENEW", "node_id": "n-lease", "key": "lease:renewed", "ttl_ms": 500, "now_ms": 2050},
        {"seq": 15, "type": "LEASE_RENEW", "node_id": "n-lease", "key": "lease:late", "ttl_ms": 500, "now_ms": 2150},
        {"seq": 16, "type": "TICK", "node_id": "n-lease", "now_ms": 2300},
    ]

    water_specs = [
        ("block:01", 1, 3000),
        ("block:02", 5, 4000),
        ("block:03", 1, 3500),
        ("block:04", 9, 5000),
        ("block:05", 0, 6000),
        ("block:06", 0, 6100),
        ("block:07", 0, 6200),
        ("block:08", 0, 6300),
    ]
    seq = 20
    for key, priority, expires_base in water_specs:
        events.append({
            "seq": seq,
            "type": "WRITE",
            "node_id": "n-water",
            "key": key,
            "stage_id": "rank_builder",
            "content": {"block": key, "weights": [seq, seq + 1]},
            "priority": priority,
            "ttl_ms": 100000,
            "now_ms": 3000,
        })
        seq += 1

    events.extend([
        {
            "seq": 30,
            "type": "APPLY",
            "application_id": "app-pending-a",
            "node_id": "n-water",
            "key": "af:pending:a",
            "stage_id": "rank_builder",
            "content": {"pending": "a"},
            "slots": 1,
            "priority": 2,
            "ttl_ms": 50000,
            "now_ms": 3100,
        },
        {
            "seq": 31,
            "type": "APPLY",
            "application_id": "app-pending-b",
            "node_id": "n-water",
            "key": "af:pending:b",
            "stage_id": "rank_builder",
            "content": {"pending": "b"},
            "slots": 1,
            "priority": 1,
            "ttl_ms": 50000,
            "now_ms": 3101,
        },
        {
            "seq": 32,
            "type": "POLICY_CHANGED",
            "node_id": "n-water",
            "capacity_version": 2,
            "lease_limit": 6,
            "now_ms": 3200,
        },
        {
            "seq": 33,
            "type": "RUN_EVICTION",
            "node_id": "n-water",
            "now_ms": 3201,
        },
        {
            "seq": 34,
            "type": "WRITE",
            "node_id": "n-water",
            "key": "block:overflow",
            "stage_id": "rank_builder",
            "content": "should-be-rejected-at-hard-watermark",
            "ttl_ms": 10000,
            "now_ms": 3202,
        },
        {
            "seq": 40,
            "type": "QUOTA_CHANGED",
            "node_id": "n-core-a",
            "quota_version": 2,
            "concurrency_quotas": {"source_profile": 3},
            "now_ms": 4000,
        },
    ])
    events.sort(key=lambda item: item["seq"])

    invalid = {
        "schema_version": "workbench.config.v1",
        "nodes": [
            {"node_id": "dup", "port": ports[0] + 100, "capacity_slots": 4, "lease_limit": 2, "seed": 1, "capacity_version": 1},
            {"node_id": "dup", "port": ports[0] + 101, "capacity_slots": 0, "lease_limit": -1, "seed": "bad", "capacity_version": 99},
            {"node_id": "valid-ish", "port": ports[0] + 102, "capacity_slots": 2, "lease_limit": 2, "seed": 3, "capacity_version": 1},
            {"node_id": "port-clash", "port": ports[0] + 100, "capacity_slots": 2, "lease_limit": 2, "seed": 4, "capacity_version": 1},
        ],
        "capacity_policies": [
            {"capacity_version": 1, "soft_permille": 500, "hard_permille": 800},
            {"capacity_version": 1, "soft_permille": 400, "hard_permille": 700},
        ],
        "manifest": {
            "stages": [
                {"stage_id": "a", "artifact_key": "art:a", "needs": ["art:c"], "concurrency_quota": 1},
                {"stage_id": "b", "artifact_key": "art:b", "needs": ["art:a"], "concurrency_quota": 1},
                {"stage_id": "c", "artifact_key": "art:c", "needs": ["art:b"], "concurrency_quota": 1},
                {"stage_id": "d", "artifact_key": "art:d", "needs": ["art:missing"], "concurrency_quota": 1},
                {"stage_id": "e", "artifact_key": "art:e", "needs": [], "concurrency_quota": 0},
            ]
        },
        "events": [
            {"seq": 1, "type": "WRITE", "node_id": "valid-ish", "key": "ok", "content": "x", "ttl_ms": 10, "now_ms": 1},
            {"seq": 1, "type": "NOT_ALLOWED"},
            {"seq": 0, "type": "TICK", "now_ms": 2},
            "not-an-object",
        ],
    }
    return {"config": config, "events": events, "invalid": invalid}


def write_sample_files(target_dir: str | Path, *, ports: list[int] | None = None) -> dict[str, Path]:
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    sample = build_sample(ports)
    config_path = target / "sample-config.json"
    events_path = target / "events.jsonl"
    invalid_path = target / "invalid-input.json"
    config_path.write_text(json.dumps(sample["config"], ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    events_text = "\n".join(json.dumps(event, ensure_ascii=False, sort_keys=True) for event in sample["events"]) + "\n"
    events_path.write_text(events_text, encoding="utf-8")
    invalid_path.write_text(canonical_dumps(sample["invalid"]) + "\n", encoding="utf-8")
    return {"config": config_path, "events": events_path, "invalid": invalid_path}
