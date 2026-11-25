from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .content import build_shard


def demo_documents() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    nodes_document = {
        "schema_version": 1,
        "global_concurrency": 3,
        "initial_quota_version": 1,
        "shard_bytes": 96,
        "handshake_timeout_ms": 3000,
        "request_timeout_ms": 3000,
        "node_concurrency": {
            "node-a": 2,
            "node-b": 2,
            "node-c": 2,
        },
        "nodes": [
            {
                "node_id": "node-a",
                "host": "127.0.0.1",
                "port": 48101,
                "capacity_slots": 2,
                "lease_limit": 2,
                "seed": 1101,
            },
            {
                "node_id": "node-b",
                "host": "127.0.0.1",
                "port": 48102,
                "capacity_slots": 2,
                "lease_limit": 2,
                "seed": 2202,
            },
            {
                "node_id": "node-c",
                "host": "127.0.0.1",
                "port": 48103,
                "capacity_slots": 4,
                "lease_limit": 4,
                "seed": 3303,
            },
        ],
    }

    pipeline_document = {
        "schema_version": 1,
        "pipeline_id": "cross-node-cache-demo",
        "watermark_scale": 1000,
        "high_watermark_fixed": 900,
        "stages": [
            {"stage_id": "build", "concurrency_quota": 2},
            {"stage_id": "test", "concurrency_quota": 1},
            {"stage_id": "package", "concurrency_quota": 1},
        ],
        "artifacts": [
            {"key": "core", "stage": "build", "references": []},
            {"key": "alpha-lib", "stage": "test", "references": ["core"]},
            {"key": "beta-lib", "stage": "test", "references": ["core"]},
            {"key": "package-bundle", "stage": "package", "references": ["alpha-lib", "beta-lib"]},
            {"key": "release-manifest", "stage": "package", "references": ["package-bundle"]},
        ],
    }

    events = [
        {"seq": 1, "type": "write", "artifact_key": "core", "shard_index": 0, "lease_ref": "core-lease", "lease_ms": 60000},
        {"seq": 2, "type": "write", "artifact_key": "alpha-lib", "shard_index": 0, "candidate_nodes": ["node-a", "node-b"], "lease_ref": "alpha-lease", "lease_ms": 120},
        {"seq": 3, "type": "write", "artifact_key": "beta-lib", "shard_index": 0, "candidate_nodes": ["node-a", "node-b"], "lease_ref": "beta-lease", "lease_ms": 60000},
        {"seq": 4, "type": "write", "artifact_key": "package-bundle", "shard_index": 0, "candidate_nodes": ["node-a", "node-b"], "lease_ref": "package-failed", "lease_ms": 60000, "fail_write": True},
        {"seq": 5, "type": "write", "artifact_key": "release-manifest", "shard_index": 0, "lease_ref": "manifest-lease", "lease_ms": 60000},
        {"seq": 6, "type": "renew", "lease_ref": "beta-lease", "expected_version": 1, "lease_ms": 60000},
        {"seq": 7, "type": "renew", "lease_ref": "beta-lease", "expected_version": 1, "lease_ms": 60000},
        {"seq": 8, "type": "release", "lease_ref": "manifest-lease"},
        {"seq": 9, "type": "wait", "duration_ms": 240},
        {"seq": 10, "type": "read", "artifact_key": "core", "shard_index": 0},
        {"seq": 11, "type": "quota_change", "quota_version": 2, "global_concurrency": 2, "reason": "sample quota rotation"},
        {"seq": 12, "type": "write", "artifact_key": "core", "shard_index": 0, "candidate_nodes": ["node-a", "node-b"], "lease_ref": "core-v2-lease", "lease_ms": 60000},
    ]
    return nodes_document, pipeline_document, events


def write_sample(target: str | Path, *, force: bool = False) -> Path:
    directory = Path(target)
    directory.mkdir(parents=True, exist_ok=True)
    expected = [directory / "nodes.json", directory / "pipeline.json", directory / "events.jsonl"]
    if any(path.exists() for path in expected) and not force:
        existing = [str(path) for path in expected if path.exists()]
        raise FileExistsError(f"sample files already exist: {existing}; pass force=True/--force to overwrite")

    nodes_document, pipeline_document, events = demo_documents()
    (directory / "nodes.json").write_text(json.dumps(nodes_document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (directory / "pipeline.json").write_text(json.dumps(pipeline_document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (directory / "events.jsonl").open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")

    digests = {}
    for node in nodes_document["nodes"]:
        shard = build_shard(
            node_id=node["node_id"],
            node_seed=node["seed"],
            artifact_key="core",
            shard_index=0,
            byte_count=nodes_document["shard_bytes"],
        )
        digests[node["node_id"]] = {
            "artifact_key": "core",
            "shard_index": 0,
            "digest": shard["digest"],
            "shard_bytes": shard["shard_bytes"],
            "source_version": shard["source_version"],
        }
    (directory / "expected-core-digests.json").write_text(
        json.dumps(digests, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return directory
