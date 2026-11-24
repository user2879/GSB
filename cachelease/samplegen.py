from __future__ import annotations

import json
import random
import socket
from pathlib import Path
from typing import Any

from .canonical import canonical_json
from .content import generate_content


def _available_port(rng: random.Random, used: set[int]) -> int:
    for _ in range(2000):
        port = rng.randint(41_000, 48_999)
        if port in used:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
        used.add(port)
        return port
    raise RuntimeError("unable to find free high loopback ports")


def build_sample(seed: int = 243) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    rng = random.Random(seed)
    used: set[int] = set()
    ports = [_available_port(rng, used) for _ in range(3)]
    config: dict[str, Any] = {
        "format_version": 1,
        "session": {"scale": 1000},
        "runtime": {
            "global_concurrency": 4,
            "handshake_timeout_ms": 4000,
            "reply_timeout_ms": 4000,
            "grace_ms": 160,
        },
        "nodes": [
            {
                "node_id": "node-a",
                "host": "127.0.0.1",
                "port": ports[0],
                "capacity_slots": 5,
                "lease_limit": 6,
                "seed": 101,
                "quota_version": 1,
                "request_concurrency": 2,
            },
            {
                "node_id": "node-b",
                "host": "127.0.0.1",
                "port": ports[1],
                "capacity_slots": 4,
                "lease_limit": 4,
                "seed": 202,
                "quota_version": 1,
                "request_concurrency": 2,
            },
            {
                "node_id": "node-c",
                "host": "127.0.0.1",
                "port": ports[2],
                "capacity_slots": 3,
                "lease_limit": 3,
                "seed": 303,
                "quota_version": 1,
                "request_concurrency": 1,
            },
        ],
        "pipeline": {
            "stages": [
                {"stage": "build", "concurrency": 3},
                {"stage": "test", "concurrency": 2},
            ],
            "artifacts": [
                {"key": "core", "stage": "build", "refs": [], "shards": {"count": 3, "bytes": 96}},
                {"key": "api", "stage": "build", "refs": ["core"], "shards": {"count": 2, "bytes": 64}},
                {
                    "key": "suite",
                    "stage": "test",
                    "refs": ["core", "api"],
                    "shards": {"count": 2, "bytes": 80},
                },
                {"key": "docs", "stage": "test", "refs": ["core"], "shards": {"count": 1, "bytes": 48}},
            ],
        },
    }

    events: list[dict[str, Any]] = [
        {"seq": 1, "batch": 1, "type": "write", "artifact": "core", "lease_ms": 60_000},
        {"seq": 2, "batch": 1, "type": "write", "artifact": "api", "lease_ms": 60_000},
        {"seq": 3, "batch": 2, "type": "write", "artifact": "suite", "lease_ms": 60_000},
        {"seq": 4, "batch": 2, "type": "write", "artifact": "docs", "lease_ms": 60_000},
        {"seq": 5, "batch": 3, "type": "write", "artifact": "core", "lease_ms": 60_000},
        {"seq": 6, "batch": 3, "type": "read", "artifact": "api", "lease_ms": 60_000},
        {"seq": 7, "batch": 4, "type": "release", "artifact": "core"},
        {"seq": 8, "batch": 5, "type": "write", "artifact": "core", "lease_ms": 60_000},
        {
            "seq": 9,
            "batch": 6,
            "type": "quota",
            "node": "node-a",
            "quota_version": 2,
            "capacity_slots": 5,
            "lease_limit": 6,
            "policy": "renew",
        },
        {"seq": 10, "batch": 7, "type": "read", "artifact": "core", "lease_ms": 60_000},
        {"seq": 11, "batch": 7, "type": "write", "artifact": "api", "lease_ms": 60_000},
        {"seq": 12, "batch": 8, "type": "release", "artifact": "api"},
        {
            "seq": 13,
            "batch": 9,
            "type": "quota",
            "node": "node-b",
            "quota_version": 2,
            "capacity_slots": 4,
            "lease_limit": 4,
            "policy": "renew",
        },
        {
            "seq": 14,
            "batch": 10,
            "type": "write",
            "artifact": "docs",
            "lease_ms": 60_000,
            "hold_ms": 260,
        },
        {"seq": 15, "batch": 11, "type": "write", "artifact": "docs", "lease_ms": 60_000},
        {"seq": 16, "batch": 12, "type": "read", "artifact": "suite", "lease_ms": 70_000},
        {"seq": 17, "batch": 12, "type": "read", "artifact": "docs", "lease_ms": 65_000},
        {"seq": 18, "batch": 13, "type": "release", "artifact": "docs"},
        {"seq": 19, "batch": 14, "type": "write", "artifact": "api", "lease_ms": 58_000},
        {"seq": 20, "batch": 14, "type": "write", "artifact": "suite", "lease_ms": 62_000},
        {"seq": 21, "batch": 15, "type": "read", "artifact": "core", "lease_ms": 55_000},
        {"seq": 22, "batch": 15, "type": "read", "artifact": "suite", "lease_ms": 55_000},
        {
            "seq": 23,
            "batch": 15,
            "type": "write",
            "artifact": "docs",
            "lease_ms": 55_000,
            "hold_ms": 90,
        },
        {"seq": 24, "batch": 15, "type": "write", "artifact": "api", "lease_ms": 55_000},
        {"seq": 25, "batch": 15, "type": "write", "artifact": "suite", "lease_ms": 55_000},
    ]
    return config, events, 15


def generate_sample(out_dir: str | Path, seed: int = 243) -> dict[str, Any]:
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    config, events, checkpoint_seq = build_sample(seed)
    config_path = destination / "sample.config.json"
    events_path = destination / "events.ndjson"
    prefix_path = destination / "events_prefix.ndjson"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    events_path.write_text("".join(canonical_json(event) + "\n" for event in events), encoding="utf-8")
    prefix = [event for event in events if event["seq"] <= checkpoint_seq]
    prefix_path.write_text("".join(canonical_json(event) + "\n" for event in prefix), encoding="utf-8")

    content_manifest: dict[str, Any] = {}
    for node in config["nodes"]:
        node_entries: dict[str, Any] = {}
        for artifact in config["pipeline"]["artifacts"]:
            content = generate_content(
                node["node_id"],
                artifact["key"],
                node["seed"],
                artifact["shards"]["count"],
                artifact["shards"]["bytes"],
            )
            node_entries[artifact["key"]] = {
                "digest": content.digest,
                "shards": len(content.shards),
                "bytes": sum(len(shard) for shard in content.shards),
            }
        content_manifest[node["node_id"]] = node_entries
    manifest = {
        "sample_seed": seed,
        "checkpoint_seq": checkpoint_seq,
        "config": config_path.name,
        "events": events_path.name,
        "events_prefix": prefix_path.name,
        "content": content_manifest,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Generate deterministic sample declarations and events")
    parser.add_argument("--out", default="samples/sample")
    parser.add_argument("--seed", type=int, default=243)
    args = parser.parse_args(argv)
    manifest = generate_sample(args.out, args.seed)
    print(canonical_json(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

