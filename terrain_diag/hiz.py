from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .canonical import canonical_bytes, numpy_sha256, sha256_bytes, write_json
from .camera import CameraPose, ProjectionResult
from .errors import ErrorCode, fail
from .heightfield import HEIGHT_SCALE
from .quadtree import QuadtreeManifest


HIZ_BASE_RESOLUTION = 64
HIZ_SPLAT_RADIUS = 1
SPLAT_DEPTH_SAFETY = 2.0e-7
FLOAT32_EPSILON = float(np.finfo(np.float32).eps)


@dataclass(frozen=True)
class DepthRender:
    depth: np.ndarray
    owner: np.ndarray
    block_ordinals: tuple[str, ...]
    coverage_pixels: dict[str, int]
    base_sha256: str
    projected_samples: int
    accepted_samples: int

    @property
    def width(self) -> int:
        return int(self.depth.shape[1])

    @property
    def height(self) -> int:
        return int(self.depth.shape[0])


@dataclass(frozen=True)
class HiZLevel:
    level: int
    width: int
    height: int
    depth: np.ndarray
    depth_error_upper_bound: float
    content_sha256: str

    def to_metadata(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "width": self.width,
            "height": self.height,
            "depth_error_upper_bound": self.depth_error_upper_bound,
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True)
class HiZPyramid:
    base_width: int
    base_height: int
    levels: tuple[HiZLevel, ...]
    base_depth_sha256: str
    content_sha256: str

    @property
    def max_level(self) -> int:
        return len(self.levels) - 1

    def get_level(self, level: int) -> HiZLevel:
        if not isinstance(level, int) or level < 0 or level >= len(self.levels):
            fail(
                ErrorCode.E_HIZ_LEVEL,
                "Hi-Z level is outside the built pyramid",
                requested_level=level,
                max_level=len(self.levels) - 1,
            )
        return self.levels[level]

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "base_width": self.base_width,
            "base_height": self.base_height,
            "max_level": self.max_level,
            "base_depth_sha256": self.base_depth_sha256,
            "content_sha256": self.content_sha256,
            "levels": [level.to_metadata() for level in self.levels],
            "level_error_model": {
                "downsample": "max over 2x2 footprint",
                "formula": "base_quant + (level+1)*f32_eps*max_abs_depth + (2^level-1)*base_quant",
                "base_quant": 2.0**-24,
            },
        }


def validate_depth_bias(value: float) -> float:
    try:
        bias = float(value)
    except (TypeError, ValueError):
        fail(ErrorCode.E_DEPTH_BIAS, "depth bias is not numeric", value=repr(value))
    if not math.isfinite(bias) or bias < 0.0 or bias > 0.25:
        fail(
            ErrorCode.E_DEPTH_BIAS,
            "depth bias must be finite and within [0, 0.25]",
            value=value,
            allowed_min=0.0,
            allowed_max=0.25,
        )
    return bias


def _classify_depth(depth: np.ndarray, level: int) -> tuple[np.ndarray, float]:
    source = np.asarray(depth)
    if source.ndim != 2 or source.shape[0] != source.shape[1]:
        fail(ErrorCode.E_HIZ_LEVEL, "Hi-Z input must be a square 2D depth map", shape=list(source.shape))
    if source.shape[0] < 1 or source.shape[0] & (source.shape[0] - 1):
        fail(ErrorCode.E_HIZ_LEVEL, "Hi-Z base resolution must be a power of two", shape=list(source.shape))
    values = source.astype(np.float32, copy=False)
    if not np.all(np.isfinite(values)):
        fail(ErrorCode.E_HIZ_LEVEL, "Hi-Z input contains non-finite depth values")
    if float(values.min()) < 0.0 or float(values.max()) > 1.0:
        fail(ErrorCode.E_HIZ_LEVEL, "Hi-Z input depth must lie in [0,1]")
    base_quant = 2.0 ** -24
    max_abs = float(np.max(np.abs(values))) if values.size else 0.0
    error = base_quant + (level + 1) * FLOAT32_EPSILON * max_abs + ((2**level) - 1) * base_quant
    return np.ascontiguousarray(values, dtype="<f4"), float(error)


def build_hiz(depth: np.ndarray) -> HiZPyramid:
    base, base_error = _classify_depth(depth, 0)
    levels_data: list[np.ndarray] = [base]
    level_errors: list[float] = [base_error]
    current = base
    level = 0
    while current.shape[0] > 1 and current.shape[1] > 1:
        level += 1
        rows, columns = current.shape
        if rows % 2 or columns % 2:
            fail(ErrorCode.E_HIZ_LEVEL, "cannot downsample an odd-sized Hi-Z level", level=level)
        blocks = current.reshape(rows // 2, 2, columns // 2, 2)
        current = np.max(blocks, axis=(1, 3)).astype(np.float32, copy=False)
        current, error = _classify_depth(current.astype(np.float32, copy=False), level)
        levels_data.append(current)
        level_errors.append(error)

    level_objects: list[HiZLevel] = []
    for level_index, level_depth in enumerate(levels_data):
        level_objects.append(
            HiZLevel(
                level=level_index,
                width=int(level_depth.shape[1]),
                height=int(level_depth.shape[0]),
                depth=level_depth,
                depth_error_upper_bound=level_errors[level_index],
                content_sha256=numpy_sha256(level_depth, "<f4"),
            )
        )

    base_sha = level_objects[0].content_sha256
    header = {
        "schema_version": 1,
        "base_width": int(base.shape[1]),
        "base_height": int(base.shape[0]),
        "base_depth_sha256": base_sha,
        "levels": [level.to_metadata() for level in level_objects],
    }
    raw = b"".join(level.depth.astype("<f4", copy=False).tobytes(order="C") for level in level_objects)
    content_sha = sha256_bytes(canonical_bytes(header, newline=False) + raw)
    return HiZPyramid(
        base_width=int(base.shape[1]),
        base_height=int(base.shape[0]),
        levels=tuple(level_objects),
        base_depth_sha256=base_sha,
        content_sha256=content_sha,
    )


def save_hiz(prefix: str | Path, pyramid: HiZPyramid, extra: dict[str, Any] | None = None) -> tuple[Path, Path, str]:
    prefix_path = Path(prefix)
    prefix_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = prefix_path.with_suffix(".bin")
    metadata_path = prefix_path.with_suffix(".json")
    raw = b"".join(level.depth.astype("<f4", copy=False).tobytes(order="C") for level in pyramid.levels)
    raw_path.write_bytes(raw)
    metadata = pyramid.metadata()
    if extra:
        metadata.update(extra)
    metadata["raw_file"] = raw_path.name
    metadata["raw_sha256"] = sha256_bytes(raw)
    write_json(metadata_path, metadata)
    return raw_path, metadata_path, pyramid.content_sha256


def load_hiz(prefix: str | Path) -> HiZPyramid:
    prefix_path = Path(prefix)
    raw_path = prefix_path.with_suffix(".bin")
    metadata_path = prefix_path.with_suffix(".json")
    if not raw_path.is_file() or not metadata_path.is_file():
        fail(ErrorCode.E_HIZ_LEVEL, "persisted Hi-Z pyramid is incomplete", prefix=str(prefix_path))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    raw = raw_path.read_bytes()
    levels: list[HiZLevel] = []
    offset = 0
    for item in metadata["levels"]:
        width = int(item["width"])
        height = int(item["height"])
        byte_count = width * height * 4
        chunk = raw[offset : offset + byte_count]
        if len(chunk) != byte_count:
            fail(ErrorCode.E_HIZ_LEVEL, "persisted Hi-Z level is truncated", level=item["level"])
        values = np.frombuffer(chunk, dtype="<f4").reshape(height, width).copy()
        digest = numpy_sha256(values, "<f4")
        if digest != item["content_sha256"]:
            fail(ErrorCode.E_HIZ_LEVEL, "persisted Hi-Z level hash mismatch", level=item["level"])
        levels.append(
            HiZLevel(
                level=int(item["level"]),
                width=width,
                height=height,
                depth=values,
                depth_error_upper_bound=float(item["depth_error_upper_bound"]),
                content_sha256=digest,
            )
        )
        offset += byte_count
    if offset != len(raw):
        fail(ErrorCode.E_HIZ_LEVEL, "persisted Hi-Z raw file has trailing bytes")
    return HiZPyramid(
        base_width=int(metadata["base_width"]),
        base_height=int(metadata["base_height"]),
        levels=tuple(levels),
        base_depth_sha256=str(metadata["base_depth_sha256"]),
        content_sha256=str(metadata["content_sha256"]),
    )


def render_depth_and_owner(
    heights: np.ndarray,
    manifest: QuadtreeManifest,
    pose: CameraPose,
    projections: dict[str, ProjectionResult],
    resolution: int = HIZ_BASE_RESOLUTION,
) -> DepthRender:
    depth = np.ones((resolution, resolution), dtype=np.float32)
    owner_init = np.iinfo(np.int32).max
    owner = np.full((resolution, resolution), owner_init, dtype=np.int32)
    blocks = sorted(manifest.leaves, key=lambda node: node.block_id or "")
    block_ordinals = tuple(node.block_id for node in blocks if node.block_id is not None)
    ordinal_for_block = {block_id: index for index, block_id in enumerate(block_ordinals)}
    view_projection = np.asarray(pose.view_projection_matrix, dtype=np.float64).reshape(4, 4)
    projected_samples = 0
    accepted_samples = 0
    candidate_blocks: list[tuple[int, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    for node in blocks:
        if node.block_id is None or not projections[node.block_id].intersects_frustum:
            continue
        xs = np.arange(node.sample_x0, node.sample_x1 + 1, dtype=np.float64)
        zs = np.arange(node.sample_z0, node.sample_z1 + 1, dtype=np.float64)
        tile = heights[node.sample_z0 : node.sample_z1 + 1, node.sample_x0 : node.sample_x1 + 1]
        grid_x, grid_z = np.meshgrid(xs, zs)
        world = np.empty((grid_x.size, 4), dtype=np.float64)
        world[:, 0] = grid_x.reshape(-1)
        world[:, 1] = tile.reshape(-1).astype(np.float64) * HEIGHT_SCALE
        world[:, 2] = grid_z.reshape(-1)
        world[:, 3] = 1.0
        clip = world @ view_projection.T
        w = clip[:, 3]
        front = w > 1e-12
        projected_samples += int(world.shape[0])
        if not np.any(front):
            continue
        ndc_x = clip[:, 0] / np.where(front, w, 1.0)
        ndc_y = clip[:, 1] / np.where(front, w, 1.0)
        depth_value = clip[:, 2] / np.where(front, w, 1.0) * 0.5 + 0.5
        pixel_x = np.floor((ndc_x * 0.5 + 0.5) * resolution)
        pixel_y = np.floor((0.5 - ndc_y * 0.5) * resolution)
        valid = (
            front
            & (depth_value >= 0.0)
            & (depth_value <= 1.0)
            & (pixel_x >= 0)
            & (pixel_x < resolution)
            & (pixel_y >= 0)
            & (pixel_y < resolution)
        )
        if not np.any(valid):
            continue
        accepted_samples += int(valid.sum())
        candidate_blocks.append(
            (
                ordinal_for_block[node.block_id],
                node.block_id,
                pixel_x[valid].astype(np.int32),
                pixel_y[valid].astype(np.int32),
                depth_value[valid].astype(np.float32),
                w[valid],
            )
        )

    for _, _, pixel_x, pixel_y, sample_depth, _ in candidate_blocks:
        for offset_y in range(-HIZ_SPLAT_RADIUS, HIZ_SPLAT_RADIUS + 1):
            for offset_x in range(-HIZ_SPLAT_RADIUS, HIZ_SPLAT_RADIUS + 1):
                splat_x = pixel_x + offset_x
                splat_y = pixel_y + offset_y
                splat_valid = (
                    (splat_x >= 0)
                    & (splat_x < resolution)
                    & (splat_y >= 0)
                    & (splat_y < resolution)
                )
                if not np.any(splat_valid):
                    continue
                flat = splat_y[splat_valid] * resolution + splat_x[splat_valid]
                values = np.minimum(sample_depth[splat_valid] + SPLAT_DEPTH_SAFETY, np.float32(1.0))
                np.minimum.at(depth.reshape(-1), flat, values)

    for ordinal, _, pixel_x, pixel_y, sample_depth, _ in candidate_blocks:
        for offset_y in range(-HIZ_SPLAT_RADIUS, HIZ_SPLAT_RADIUS + 1):
            for offset_x in range(-HIZ_SPLAT_RADIUS, HIZ_SPLAT_RADIUS + 1):
                splat_x = pixel_x + offset_x
                splat_y = pixel_y + offset_y
                splat_valid = (
                    (splat_x >= 0)
                    & (splat_x < resolution)
                    & (splat_y >= 0)
                    & (splat_y < resolution)
                )
                if not np.any(splat_valid):
                    continue
                flat = splat_y[splat_valid] * resolution + splat_x[splat_valid]
                values = np.minimum(sample_depth[splat_valid] + SPLAT_DEPTH_SAFETY, np.float32(1.0))
                equal = depth.reshape(-1)[flat] == values
                if np.any(equal):
                    np.minimum.at(owner.reshape(-1), flat[equal], np.int32(ordinal))

    owner = np.where(owner == owner_init, -1, owner).astype(np.int32)
    unique, counts = np.unique(owner, return_counts=True)
    coverage: dict[str, int] = {}
    for ordinal, count in zip(unique.tolist(), counts.tolist()):
        if ordinal >= 0:
            coverage[block_ordinals[ordinal]] = int(count)
    return DepthRender(
        depth=np.ascontiguousarray(depth, dtype=np.float32),
        owner=np.ascontiguousarray(owner, dtype=np.int32),
        block_ordinals=block_ordinals,
        coverage_pixels=coverage,
        base_sha256=numpy_sha256(depth, "<f4"),
        projected_samples=projected_samples,
        accepted_samples=accepted_samples,
    )
