from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

from .aggregate import aggregate_shard, generate_integer_samples
from .receipts import encode_d1, encode_json_receipt
from .spec import load_spec_bundle


def free_loopback_ports(count: int) -> list[int]:
    sockets: list[socket.socket] = []
    ports: list[int] = []
    for _ in range(count):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        ports.append(int(sock.getsockname()[1]))
        sockets.append(sock)
    for sock in sockets:
        sock.close()
    return ports


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_demo_spec(target: str | Path, *, allocate_ports: bool = False, active_capacity_version: int = 1) -> Path:
    root = Path(target)
    root.mkdir(parents=True, exist_ok=True)
    ports = free_loopback_ports(3) if allocate_ports else [19101, 19102, 19103]
    samples = generate_integer_samples(407, count=6, value_count=8)

    manifest = {
        "spec_format_version": 1,
        "node_set_version": 1,
        "name": "edge-shard-diagnostic-demo",
        "active_capacity_version": active_capacity_version,
        "active_fixed_revision": 1,
    }
    _write_json(root / "manifest.json", manifest)
    _write_json(
        root / "nodes.json",
        {
            "nodes": [
                {"id": "node-a", "seq": 1, "host": "127.0.0.1", "port": ports[0], "concurrency_quota": 2, "lease_limit": 2, "fixed_scale": 10},
                {"id": "node-b", "seq": 2, "host": "127.0.0.1", "port": ports[1], "concurrency_quota": 1, "lease_limit": 1, "fixed_scale": 100},
                {"id": "node-c", "seq": 3, "host": "127.0.0.1", "port": ports[2], "concurrency_quota": 2, "lease_limit": 2, "fixed_scale": 1},
            ]
        },
    )
    _write_json(root / "samples.json", {"samples": samples})
    sample_ids = [item["id"] for item in samples]
    _write_json(
        root / "shards.json",
        {
            "shards": [
                {"id": "shard-0001", "node_id": "node-a", "batch_id": "batch-407", "seq_start": 1001, "seq_end": 1002, "resource_units": 2, "dependencies": [], "sample_ids": sample_ids[0:2]},
                {"id": "shard-0002", "node_id": "node-b", "batch_id": "batch-407", "seq_start": 2001, "seq_end": 2002, "resource_units": 2, "dependencies": ["shard-0001"], "sample_ids": [sample_ids[2]]},
                {"id": "shard-0003", "node_id": "node-c", "batch_id": "batch-407", "seq_start": 3001, "seq_end": 3002, "resource_units": 2, "dependencies": ["shard-0001"], "sample_ids": [sample_ids[3]]},
                {"id": "shard-0004", "node_id": "node-a", "batch_id": "batch-407", "seq_start": 4001, "seq_end": 4003, "resource_units": 3, "dependencies": ["shard-0002", "shard-0003"], "sample_ids": sample_ids[4:6]},
            ]
        },
    )
    _write_json(
        root / "capacity_versions.json",
        {
            "versions": [
                {
                    "version": 1,
                    "parent_version": None,
                    "budget_units": 3,
                    "node_limits": {
                        "node-a": {"concurrency_quota": 2, "lease_limit": 2},
                        "node-b": {"concurrency_quota": 1, "lease_limit": 1},
                        "node-c": {"concurrency_quota": 2, "lease_limit": 2},
                    },
                },
                {
                    "version": 2,
                    "parent_version": 1,
                    "budget_units": 12,
                    "node_limits": {
                        "node-a": {"concurrency_quota": 3, "lease_limit": 3},
                        "node-b": {"concurrency_quota": 2, "lease_limit": 2},
                        "node-c": {"concurrency_quota": 1, "lease_limit": 1},
                    },
                },
            ]
        },
    )
    _write_json(
        root / "fixed_revisions.json",
        {
            "revisions": [
                {"revision": 1, "parent_revision": None, "scale_factor": 1},
                {"revision": 2, "parent_revision": 1, "scale_factor": 10},
                {"revision": 3, "parent_revision": 2, "scale_factor": 100},
            ]
        },
    )
    _write_json(
        root / "event_stream.json",
        {
            "events": [
                {"seq": 1, "type": "SPEC_LOADED", "detail": {"source": "local-deterministic-demo"}},
                {"seq": 2, "type": "CAPACITY_CHAIN_VALIDATED", "detail": {"latest": 2}},
                {"seq": 3, "type": "FIXED_REVISION_CHAIN_VALIDATED", "detail": {"active": 1}},
                {"seq": 4, "type": "RECEIPT_FIXTURES_READY", "detail": {"formats": ["d1", "d2", "d3"]}},
            ]
        },
    )
    _write_json(root / "formats.json", {"d1": {"fields": ["node_seq", "seq_start", "seq_end", "summary_prefix"], "widths": [4, 5, 5, 16], "prefix_chars": 16}})
    _write_json(root / "delivery_refs.json", {"known_refs": ["dup-delivery-407-001", "dup-delivery-407-002", "dup-delivery-407-003"]})
    return root


def write_receipt_fixtures(spec_dir: str | Path, receipt_dir: str | Path) -> dict[str, Path]:
    bundle = load_spec_bundle(spec_dir)
    out = Path(receipt_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"d1": out / "receipts.d1", "d2": out / "receipts.d2.jsonl", "d3": out / "receipts.d3.jsonl"}

    summaries_scale1 = {
        shard.id: aggregate_shard(shard, bundle.samples_by_id, capacity_version=1, scale_factor=1, node_fixed_scale=bundle.nodes_by_id[shard.node_id].fixed_scale)
        for shard in bundle.shards
    }
    d1_lines: list[str] = []
    for shard in bundle.shards:
        node = bundle.nodes_by_id[shard.node_id]
        result = summaries_scale1[shard.id]
        d1_lines.append(
            encode_d1(
                {
                    "node_seq": node.seq,
                    "seq_start": shard.seq_start,
                    "seq_end": shard.seq_end,
                    "summary_prefix": result["summary_prefix"],
                },
                bundle.format_spec,
            )
        )
    paths["d1"].write_text("\n".join(d1_lines) + "\n", encoding="utf-8")

    d2_records: list[str] = []
    for index, shard in enumerate(bundle.shards[:3], start=1):
        result = aggregate_shard(shard, bundle.samples_by_id, capacity_version=1, scale_factor=10, node_fixed_scale=bundle.nodes_by_id[shard.node_id].fixed_scale)
        d2_records.append(
            encode_json_receipt(
                {
                    "format": "d2",
                    "receipt_id": f"r2-{index:04d}",
                    "shard_id": shard.id,
                    "node_seq": bundle.nodes_by_id[shard.node_id].seq,
                    "seq_start": shard.seq_start,
                    "seq_end": shard.seq_end,
                    "summary_prefix": result["summary_prefix"],
                    "capacity_version": 1,
                    "scale_factor": 10,
                    "duration_ms": 8.0 + index,
                }
            )
        )
    paths["d2"].write_text("\n".join(d2_records) + "\n", encoding="utf-8")

    d3_records: list[str] = []
    duplicate_map = {
        "shard-0002": ["dup-delivery-407-002"],
        "shard-0004": ["dup-delivery-407-001", "dup-delivery-407-003"],
    }
    for rank, shard in enumerate(bundle.shards, start=1):
        result = aggregate_shard(shard, bundle.samples_by_id, capacity_version=1, scale_factor=100, node_fixed_scale=bundle.nodes_by_id[shard.node_id].fixed_scale)
        d3_records.append(
            encode_json_receipt(
                {
                    "format": "d3",
                    "receipt_id": f"r3-{rank:04d}",
                    "shard_id": shard.id,
                    "node_seq": bundle.nodes_by_id[shard.node_id].seq,
                    "seq_start": shard.seq_start,
                    "seq_end": shard.seq_end,
                    "summary_prefix": result["summary_prefix"],
                    "capacity_version": 1,
                    "scale_factor": 100,
                    "duration_ms": 11.5 + rank,
                    "stable_rank": rank,
                    "duplicate_refs": duplicate_map.get(shard.id, []),
                    "sample_snapshot": result["sample_snapshot"],
                }
            )
        )
    paths["d3"].write_text("\n".join(d3_records) + "\n", encoding="utf-8")
    return paths


def bootstrap_repository_demo(root: str | Path = ".") -> dict[str, Path]:
    root = Path(root)
    spec = write_demo_spec(root / "config" / "demo", allocate_ports=False)
    receipts = write_receipt_fixtures(spec, root / "samples" / "receipts")
    return {"spec": spec, **receipts}
