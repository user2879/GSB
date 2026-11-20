from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .canonical import canonical_bytes, sha256_bytes, write_json
from .errors import ErrorCode, fail
from .heightfield import HEIGHT_SCALE


CHUNK_SEGMENTS = 32


@dataclass(frozen=True)
class Bounds:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float

    def expanded(self, world_margin: float) -> "Bounds":
        margin = max(0.0, float(world_margin))
        return Bounds(
            self.xmin - margin,
            self.xmax + margin,
            self.ymin - margin,
            self.ymax + margin,
            self.zmin - margin,
            self.zmax + margin,
        )

    def to_list(self) -> list[float]:
        return [self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax]


@dataclass(frozen=True)
class QuadtreeNode:
    node_id: str
    block_id: str | None
    level: int
    grid_x: int
    grid_z: int
    span_samples: int
    sample_x0: int
    sample_x1: int
    sample_z0: int
    sample_z1: int
    height_min_raw: int
    height_max_raw: int
    bounds: Bounds
    parent_id: str | None
    child_ids: tuple[str, ...]
    height_bytes: int
    memory_bytes: int
    vram_bytes: int
    payload_sha256: str | None

    @property
    def is_leaf(self) -> bool:
        return self.block_id is not None

    @property
    def height_range_raw(self) -> int:
        return self.height_max_raw - self.height_min_raw

    @property
    def height_range_world(self) -> float:
        return self.height_range_raw * HEIGHT_SCALE

    @property
    def sample_count(self) -> int:
        return (self.sample_x1 - self.sample_x0 + 1) * (self.sample_z1 - self.sample_z0 + 1)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "block_id": self.block_id,
            "level": self.level,
            "grid": [self.grid_x, self.grid_z],
            "span_samples": self.span_samples,
            "sample_range": [
                self.sample_x0,
                self.sample_x1,
                self.sample_z0,
                self.sample_z1,
            ],
            "sample_count": self.sample_count,
            "height_min_raw": self.height_min_raw,
            "height_max_raw": self.height_max_raw,
            "height_range_raw": self.height_range_raw,
            "height_range_world": self.height_range_world,
            "bounds": self.bounds.to_list(),
            "parent_id": self.parent_id,
            "child_ids": list(self.child_ids),
            "height_bytes": self.height_bytes,
            "memory_bytes": self.memory_bytes,
            "vram_bytes": self.vram_bytes,
            "payload_sha256": self.payload_sha256,
        }


@dataclass(frozen=True)
class QuadtreeManifest:
    size: int
    seed: int
    terrain_sha256: str
    chunk_segments: int
    tree_depth: int
    nodes: tuple[QuadtreeNode, ...]
    root_id: str
    content_sha256: str

    @property
    def leaves(self) -> tuple[QuadtreeNode, ...]:
        return tuple(node for node in self.nodes if node.is_leaf)

    @property
    def node_by_id(self) -> dict[str, QuadtreeNode]:
        return {node.node_id: node for node in self.nodes}

    @property
    def leaf_by_id(self) -> dict[str, QuadtreeNode]:
        return {node.block_id: node for node in self.leaves if node.block_id is not None}

    def get_block(self, block_id: str) -> QuadtreeNode:
        node = self.leaf_by_id.get(block_id)
        if node is None:
            fail(
                ErrorCode.E_BLOCK_MISSING,
                "requested terrain block is not present in the quadtree manifest",
                block_id=block_id,
            )
        return node

    def neighbors(self, block_id: str, radius: int = 1) -> tuple[str, ...]:
        node = self.get_block(block_id)
        max_index = (1 << self.tree_depth) - 1
        result: list[str] = []
        for dz in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == 0 and dz == 0:
                    continue
                gx = node.grid_x + dx
                gz = node.grid_z + dz
                if 0 <= gx <= max_index and 0 <= gz <= max_index:
                    result.append(f"B{self.tree_depth:03d}_{gx:05d}_{gz:05d}")
        return tuple(sorted(result))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "size": self.size,
            "seed": self.seed,
            "terrain_sha256": self.terrain_sha256,
            "chunk_segments": self.chunk_segments,
            "tree_depth": self.tree_depth,
            "height_scale": HEIGHT_SCALE,
            "root_id": self.root_id,
            "leaf_count": len(self.leaves),
            "node_count": len(self.nodes),
            "content_sha256": self.content_sha256,
            "nodes": [node.to_manifest() for node in self.nodes],
        }


def _node_id(level: int, grid_x: int, grid_z: int) -> str:
    return f"N{level:03d}_{grid_x:05d}_{grid_z:05d}"


def _block_id(tree_depth: int, grid_x: int, grid_z: int) -> str:
    return f"B{tree_depth:03d}_{grid_x:05d}_{grid_z:05d}"


def build_quadtree(heights: np.ndarray, seed: int, terrain_sha256: str) -> QuadtreeManifest:
    size = int(heights.shape[0])
    if heights.shape != (size, size):
        fail(ErrorCode.E_HEIGHT_RANGE, "terrain height field must be square", shape=list(heights.shape))
    root_segments = size - 1
    if root_segments % CHUNK_SEGMENTS != 0:
        raise ValueError("terrain size minus one must be divisible by chunk size")
    tree_depth_float = math.log2(root_segments / CHUNK_SEGMENTS)
    if tree_depth_float != int(tree_depth_float):
        raise ValueError("chunk configuration must form an exact quadtree")
    tree_depth = int(tree_depth_float)

    nodes: dict[str, QuadtreeNode] = {}

    def build_node(level: int, grid_x: int, grid_z: int, parent_id: str | None) -> QuadtreeNode:
        span = root_segments >> level
        x0 = grid_x * span
        z0 = grid_z * span
        x1 = x0 + span
        z1 = z0 + span
        node_id = _node_id(level, grid_x, grid_z)

        if level == tree_depth:
            block = _block_id(tree_depth, grid_x, grid_z)
            payload = np.ascontiguousarray(heights[z0 : z1 + 1, x0 : x1 + 1], dtype="<u2")
            height_min = int(payload.min())
            height_max = int(payload.max())
            height_bytes = int(payload.nbytes)
            memory_bytes = height_bytes + 192
            vram_bytes = height_bytes + 128
            payload_sha256 = sha256_bytes(payload.tobytes(order="C"))
            node = QuadtreeNode(
                node_id=node_id,
                block_id=block,
                level=level,
                grid_x=grid_x,
                grid_z=grid_z,
                span_samples=span,
                sample_x0=x0,
                sample_x1=x1,
                sample_z0=z0,
                sample_z1=z1,
                height_min_raw=height_min,
                height_max_raw=height_max,
                bounds=Bounds(
                    float(x0),
                    float(x1),
                    height_min * HEIGHT_SCALE,
                    height_max * HEIGHT_SCALE,
                    float(z0),
                    float(z1),
                ),
                parent_id=parent_id,
                child_ids=tuple(),
                height_bytes=height_bytes,
                memory_bytes=memory_bytes,
                vram_bytes=vram_bytes,
                payload_sha256=payload_sha256,
            )
            nodes[node_id] = node
            return node

        half = span // 2
        children = [
            build_node(level + 1, grid_x * 2 + dx, grid_z * 2 + dz, node_id)
            for dz in (0, 1)
            for dx in (0, 1)
        ]
        height_min = min(child.height_min_raw for child in children)
        height_max = max(child.height_max_raw for child in children)
        node = QuadtreeNode(
            node_id=node_id,
            block_id=None,
            level=level,
            grid_x=grid_x,
            grid_z=grid_z,
            span_samples=span,
            sample_x0=x0,
            sample_x1=x1,
            sample_z0=z0,
            sample_z1=z1,
            height_min_raw=height_min,
            height_max_raw=height_max,
            bounds=Bounds(
                float(x0),
                float(x1),
                height_min * HEIGHT_SCALE,
                height_max * HEIGHT_SCALE,
                float(z0),
                float(z1),
            ),
            parent_id=parent_id,
            child_ids=tuple(child.node_id for child in children),
            height_bytes=0,
            memory_bytes=0,
            vram_bytes=0,
            payload_sha256=None,
        )
        nodes[node_id] = node
        return node

    root = build_node(0, 0, 0, None)
    ordered_nodes = tuple(nodes[node_id] for node_id in sorted(nodes))
    manifest_without_hash = {
        "schema_version": 1,
        "size": size,
        "seed": int(seed),
        "terrain_sha256": terrain_sha256,
        "chunk_segments": CHUNK_SEGMENTS,
        "tree_depth": tree_depth,
        "height_scale": HEIGHT_SCALE,
        "root_id": root.node_id,
        "leaf_count": sum(1 for node in ordered_nodes if node.is_leaf),
        "node_count": len(ordered_nodes),
        "nodes": [node.to_manifest() for node in ordered_nodes],
    }
    content_sha256 = sha256_bytes(canonical_bytes(manifest_without_hash, newline=False))
    return QuadtreeManifest(
        size=size,
        seed=int(seed),
        terrain_sha256=terrain_sha256,
        chunk_segments=CHUNK_SEGMENTS,
        tree_depth=tree_depth,
        nodes=ordered_nodes,
        root_id=root.node_id,
        content_sha256=content_sha256,
    )


def manifest_path(root: str | Path, size: int, seed: int) -> Path:
    return Path(root) / "manifests" / f"terrain_{size}_{int(seed):016x}.quadtree.json"


def save_quadtree_manifest(root: str | Path, manifest: QuadtreeManifest) -> tuple[Path, str]:
    target = manifest_path(root, manifest.size, manifest.seed)
    digest = write_json(target, manifest.to_dict())
    return target, digest


def load_quadtree_manifest(root: str | Path, size: int, seed: int) -> dict[str, Any]:
    import json

    target = manifest_path(root, size, seed)
    if not target.is_file():
        fail(ErrorCode.E_BLOCK_MISSING, "quadtree manifest is missing", size=size, seed=int(seed))
    return json.loads(target.read_text(encoding="utf-8"))


def block_ids(manifest: QuadtreeManifest) -> tuple[str, ...]:
    return tuple(sorted(node.block_id for node in manifest.leaves if node.block_id is not None))


def total_leaf_bytes(manifest: QuadtreeManifest) -> tuple[int, int]:
    memory = sum(node.memory_bytes for node in manifest.leaves)
    vram = sum(node.vram_bytes for node in manifest.leaves)
    return memory, vram
