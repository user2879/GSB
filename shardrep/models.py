"""Immutable declaration and runtime models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ShardDecl:
    node_id: str
    shard_id: str
    logical_id: str
    seq: int
    byte_count: int
    digest: str
    file_name: str
    path: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "shard_id": self.shard_id,
            "logical_id": self.logical_id,
            "seq": self.seq,
            "bytes": self.byte_count,
            "digest": self.digest,
            "file": self.file_name,
        }


@dataclass(frozen=True)
class NodeDecl:
    node_id: str
    directory: str
    path: Path
    pack_quota: int
    repair_enabled: bool
    shards: tuple[ShardDecl, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "directory": self.directory,
            "pack_quota": self.pack_quota,
            "repair_enabled": self.repair_enabled,
            "shards": [shard.as_dict() for shard in self.shards],
        }


@dataclass(frozen=True)
class BlockDecl:
    block_id: str
    seq_start: int
    seq_end: int
    checksum: str

    @property
    def seqs(self) -> range:
        return range(self.seq_start, self.seq_end + 1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "seq_start": self.seq_start,
            "seq_end": self.seq_end,
            "checksum": self.checksum,
        }


@dataclass(frozen=True)
class RelationDecl:
    kind: str
    sources: tuple[str, ...]
    target: str

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "sources": list(self.sources), "target": self.target}


@dataclass(frozen=True)
class LogicalShard:
    logical_id: str
    seq: int
    byte_count: int
    digest: str
    block_id: str
    replicas: tuple[ShardDecl, ...]


@dataclass(frozen=True)
class Spec:
    raw: dict[str, Any]
    source_path: Path
    spec_dir: Path
    format_version: int
    group_id: str
    group_version: int
    mirror_graph_version: int
    seed: int
    content_engine: str
    nodes: tuple[NodeDecl, ...]
    blocks: tuple[BlockDecl, ...]
    relations: tuple[RelationDecl, ...]
    logical_shards: dict[str, LogicalShard]
    seq_to_logical: dict[int, str]
    block_by_id: dict[str, BlockDecl]
    node_by_id: dict[str, NodeDecl]
    shard_by_id: dict[str, ShardDecl]
    topo_order: tuple[str, ...]
    adjacency: dict[str, tuple[str, ...]]
    predecessors: dict[str, tuple[str, ...]]

    def logicals_for_block(self, block: BlockDecl) -> tuple[str, ...]:
        return tuple(self.seq_to_logical[seq] for seq in block.seqs)

    def block_for_logical(self, logical_id: str) -> BlockDecl:
        return self.block_by_id[self.logical_shards[logical_id].block_id]

    def mirror_graph(self) -> dict[str, Any]:
        return {
            "mirror_graph_version": self.mirror_graph_version,
            "nodes": sorted(self.logical_shards),
            "relations": [relation.as_dict() for relation in self.relations],
        }

