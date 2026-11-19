"""Deterministic, dependency-free local classroom sample generator."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .canonical import atomic_write_bytes, canonical_json_text
from .content import content_digest, generate_shard_bytes
from .spec import compute_block_checksum

DEMO_SEED = 20261120
DEMO_GROUP_ID = "classroom-repair"
DEMO_NODES: tuple[tuple[str, int], ...] = (
    ("node-a", 2),
    ("node-b", 1),
    ("node-c", 1),
)
DEMO_LENGTHS: tuple[int, ...] = (1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920)
DEMO_RELATIONS: tuple[dict[str, Any], ...] = (
    {"kind": "mirror", "source": "s1", "target": "s2"},
    {"kind": "mirror", "source": "s2", "target": "s4"},
    {"kind": "join", "sources": ["s3", "s4"], "target": "s5"},
    {"kind": "join", "sources": ["s5", "s6"], "target": "s7"},
)
DEMO_BLOCKS: tuple[tuple[str, int, int], ...] = (
    ("block-alpha", 0, 2),
    ("block-beta", 3, 5),
    ("block-gamma", 6, 7),
)


def demoSamplePath(root: str | Path) -> Path:
    return Path(root).resolve() / "samples" / "spec.json"


def generate_sample(root: str | Path, *, force: bool = True) -> Path:
    """Create the complete local sample under ``<root>/samples``.

    Every shard replica with the same logical id has byte-identical content. The
    mirror graph is metadata used for invalidation and repacking, not a shortcut
    for file traversal. Physical shard ids are globally unique.
    """

    root_path = Path(root).resolve()
    samples_dir = root_path / "samples"
    spec_path = samples_dir / "spec.json"
    if spec_path.exists() and not force:
        return spec_path
    if samples_dir.exists():
        shutil.rmtree(samples_dir)
    nodes_dir = samples_dir / "nodes"
    nodes_dir.mkdir(parents=True, exist_ok=True)

    logical_meta: dict[str, tuple[int, int, str]] = {}
    seq_to_logical: dict[int, str] = {}
    for seq, byte_count in enumerate(DEMO_LENGTHS):
        logical_id = f"s{seq}"
        digest = content_digest(generate_shard_bytes(DEMO_SEED, logical_id, byte_count))
        logical_meta[logical_id] = (seq, byte_count, digest)
        seq_to_logical[seq] = logical_id

    nodes_json: list[dict[str, Any]] = []
    for node_id, quota in DEMO_NODES:
        node_dir = nodes_dir / node_id
        node_dir.mkdir(parents=True, exist_ok=True)
        shards_json: list[dict[str, Any]] = []
        for seq, byte_count in enumerate(DEMO_LENGTHS):
            logical_id = f"s{seq}"
            file_name = f"s{seq:02d}.bin"
            payload = generate_shard_bytes(DEMO_SEED, logical_id, byte_count)
            digest = logical_meta[logical_id][2]
            if content_digest(payload) != digest:
                raise RuntimeError("deterministic content generator is inconsistent")
            atomic_write_bytes(node_dir / file_name, payload)
            shards_json.append(
                {
                    "shard_id": f"{node_id}:{logical_id}",
                    "logical_id": logical_id,
                    "seq": seq,
                    "bytes": byte_count,
                    "digest": digest,
                    "file": file_name,
                }
            )
        nodes_json.append(
            {
                "node_id": node_id,
                "directory": f"nodes/{node_id}",
                "pack_quota": quota,
                "repair_enabled": True,
                "shards": shards_json,
            }
        )

    blocks_json: list[dict[str, Any]] = []
    for block_id, start, end in DEMO_BLOCKS:
        checksum = compute_block_checksum(block_id, start, end, logical_meta, seq_to_logical)
        blocks_json.append(
            {
                "block_id": block_id,
                "seq_start": start,
                "seq_end": end,
                "checksum": checksum,
            }
        )

    spec = {
        "format": "shardrep-spec",
        "format_version": 1,
        "group_id": DEMO_GROUP_ID,
        "group_version": 1,
        "mirror_graph_version": 1,
        "content": {"engine": "shake256-v1", "seed": DEMO_SEED},
        "nodes": nodes_json,
        "blocks": blocks_json,
        "relations": list(DEMO_RELATIONS),
    }
    spec_path.write_text(canonical_json_text(spec), encoding="utf-8")
    return spec_path


def sample_manifest(root: str | Path) -> dict[str, Any]:
    spec_path = demoSamplePath(root)
    return json.loads(spec_path.read_text(encoding="utf-8"))

