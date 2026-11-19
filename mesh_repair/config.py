"""Configuration loading for deterministic benchmark runs."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import BusinessError, ErrorCode


def _as_int(data: dict[str, Any], key: str, default: int | None = None) -> int:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise BusinessError(ErrorCode.E_INTERNAL, f"config value {key!r} must be an integer")
    return value


def _as_float(data: dict[str, Any], key: str, default: float) -> float:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BusinessError(ErrorCode.E_INTERNAL, f"config value {key!r} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise BusinessError(ErrorCode.E_INTERNAL, f"config value {key!r} must be finite")
    return result


@dataclass(frozen=True)
class ShapeSpec:
    kind: str
    translate: tuple[float, float, float]
    radius: float = 0.72
    subdivisions: int = 2
    major_radius: float = 1.0
    minor_radius: float = 0.32
    tube_segments: int = 32
    ring_segments: int = 12
    half_height: float = 0.55

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "translate": list(self.translate),
            "radius": self.radius,
            "subdivisions": self.subdivisions,
            "major_radius": self.major_radius,
            "minor_radius": self.minor_radius,
            "tube_segments": self.tube_segments,
            "ring_segments": self.ring_segments,
            "half_height": self.half_height,
        }


@dataclass(frozen=True)
class DefectCounts:
    non_manifold_edge: int
    duplicate_vertex: int
    zero_area_face: int
    isolated_component: int
    self_intersection: int

    @property
    def injection_count(self) -> int:
        return (
            self.non_manifold_edge
            + self.duplicate_vertex
            + self.zero_area_face
            + self.isolated_component
            + self.self_intersection
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "non_manifold_edge": self.non_manifold_edge,
            "duplicate_vertex": self.duplicate_vertex,
            "zero_area_face": self.zero_area_face,
            "isolated_component": self.isolated_component,
            "self_intersection": self.self_intersection,
        }


@dataclass(frozen=True)
class PipelineConfig:
    seed: int
    shapes: tuple[ShapeSpec, ...]
    defects: DefectCounts
    weld_tolerance: float
    boundary_tolerance: float
    self_intersection_epsilon: float
    max_faces_per_block: int
    isolated_component_face_threshold: int
    threads: int
    memory_budget_bytes: int
    block_time_budget_ms: float
    volume_relative_tolerance: float
    area_relative_tolerance: float
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "shapes": [shape.to_dict() for shape in self.shapes],
            "defects": self.defects.to_dict(),
            "weld_tolerance": self.weld_tolerance,
            "boundary_tolerance": self.boundary_tolerance,
            "self_intersection_epsilon": self.self_intersection_epsilon,
            "max_faces_per_block": self.max_faces_per_block,
            "isolated_component_face_threshold": self.isolated_component_face_threshold,
            "threads": self.threads,
            "memory_budget_bytes": self.memory_budget_bytes,
            "block_time_budget_ms": self.block_time_budget_ms,
            "volume_relative_tolerance": self.volume_relative_tolerance,
            "area_relative_tolerance": self.area_relative_tolerance,
        }

    def with_overrides(
        self,
        *,
        threads: int | None = None,
        memory_budget_bytes: int | None = None,
        block_time_budget_ms: float | None = None,
    ) -> "PipelineConfig":
        return PipelineConfig(
            seed=self.seed,
            shapes=self.shapes,
            defects=self.defects,
            weld_tolerance=self.weld_tolerance,
            boundary_tolerance=self.boundary_tolerance,
            self_intersection_epsilon=self.self_intersection_epsilon,
            max_faces_per_block=self.max_faces_per_block,
            isolated_component_face_threshold=self.isolated_component_face_threshold,
            threads=self.threads if threads is None else threads,
            memory_budget_bytes=(
                self.memory_budget_bytes if memory_budget_bytes is None else memory_budget_bytes
            ),
            block_time_budget_ms=(
                self.block_time_budget_ms
                if block_time_budget_ms is None
                else block_time_budget_ms
            ),
            volume_relative_tolerance=self.volume_relative_tolerance,
            area_relative_tolerance=self.area_relative_tolerance,
            schema_version=self.schema_version,
        )


def _parse_shape(raw: Any, index: int) -> ShapeSpec:
    if not isinstance(raw, dict):
        raise BusinessError(ErrorCode.E_INTERNAL, "shape entry must be an object", location={"shape": index})
    kind = raw.get("kind")
    if kind not in {"icosphere", "torus", "grooved_cube"}:
        raise BusinessError(
            ErrorCode.E_INTERNAL,
            "unsupported base shape",
            location={"shape": index},
            details={"kind": kind},
        )
    translate_raw = raw.get("translate", [0.0, 0.0, 0.0])
    if not isinstance(translate_raw, list) or len(translate_raw) != 3:
        raise BusinessError(ErrorCode.E_INTERNAL, "shape translate must be a length-3 array")
    translate = tuple(_as_float({"v": value}, "v", 0.0) for value in translate_raw)
    shape = ShapeSpec(
        kind=str(kind),
        translate=(translate[0], translate[1], translate[2]),
        radius=_as_float(raw, "radius", 0.72),
        subdivisions=_as_int(raw, "subdivisions", 2),
        major_radius=_as_float(raw, "major_radius", 1.0),
        minor_radius=_as_float(raw, "minor_radius", 0.32),
        tube_segments=_as_int(raw, "tube_segments", 32),
        ring_segments=_as_int(raw, "ring_segments", 12),
        half_height=_as_float(raw, "half_height", 0.55),
    )
    if shape.radius <= 0 or shape.major_radius <= 0 or shape.minor_radius <= 0:
        raise BusinessError(ErrorCode.E_INTERNAL, "shape radii must be positive")
    if shape.subdivisions < 0 or shape.tube_segments < 3 or shape.ring_segments < 3:
        raise BusinessError(ErrorCode.E_INTERNAL, "shape segment counts are too small")
    if shape.half_height <= 0:
        raise BusinessError(ErrorCode.E_INTERNAL, "shape half_height must be positive")
    return shape


def config_from_dict(raw: dict[str, Any]) -> PipelineConfig:
    try:
        defects_raw = raw["defects"]
        if not isinstance(defects_raw, dict):
            raise KeyError("defects")
        defects = DefectCounts(
            non_manifold_edge=_as_int(defects_raw, "non_manifold_edge", 0),
            duplicate_vertex=_as_int(defects_raw, "duplicate_vertex", 0),
            zero_area_face=_as_int(defects_raw, "zero_area_face", 0),
            isolated_component=_as_int(defects_raw, "isolated_component", 0),
            self_intersection=_as_int(defects_raw, "self_intersection", 0),
        )
        shapes_raw = raw["shapes"]
        if not isinstance(shapes_raw, list) or not shapes_raw:
            raise BusinessError(ErrorCode.E_INTERNAL, "config must declare at least one shape")
        config = PipelineConfig(
            schema_version=_as_int(raw, "schema_version", 1),
            seed=_as_int(raw, "seed", 0),
            shapes=tuple(_parse_shape(shape, index) for index, shape in enumerate(shapes_raw)),
            defects=defects,
            weld_tolerance=_as_float(raw, "weld_tolerance", 1.0e-9),
            boundary_tolerance=_as_float(raw, "boundary_tolerance", 1.0e-9),
            self_intersection_epsilon=_as_float(raw, "self_intersection_epsilon", 1.0e-10),
            max_faces_per_block=_as_int(raw, "max_faces_per_block", 96),
            isolated_component_face_threshold=_as_int(
                raw, "isolated_component_face_threshold", 12
            ),
            threads=_as_int(raw, "threads", 4),
            memory_budget_bytes=_as_int(raw, "memory_budget_bytes", 64 * 1024 * 1024),
            block_time_budget_ms=_as_float(raw, "block_time_budget_ms", 5.0),
            volume_relative_tolerance=_as_float(raw, "volume_relative_tolerance", 1.0e-9),
            area_relative_tolerance=_as_float(raw, "area_relative_tolerance", 1.0e-9),
        )
    except KeyError as exc:
        raise BusinessError(
            ErrorCode.E_INTERNAL, "missing required config value", details={"key": str(exc)}
        ) from exc

    if config.schema_version != 1:
        raise BusinessError(ErrorCode.E_INTERNAL, "unsupported config schema version")
    if config.seed < 0:
        raise BusinessError(ErrorCode.E_INTERNAL, "seed must be non-negative")
    if any(
        count < 0
        for count in (
            defects.non_manifold_edge,
            defects.duplicate_vertex,
            defects.zero_area_face,
            defects.isolated_component,
            defects.self_intersection,
        )
    ):
        raise BusinessError(ErrorCode.E_INTERNAL, "defect counts must be non-negative")
    if config.weld_tolerance < 0 or config.boundary_tolerance < 0:
        raise BusinessError(ErrorCode.E_INTERNAL, "weld tolerances must be non-negative")
    if config.self_intersection_epsilon <= 0:
        raise BusinessError(ErrorCode.E_INTERNAL, "self-intersection epsilon must be positive")
    if config.max_faces_per_block < 1:
        raise BusinessError(ErrorCode.E_INTERNAL, "max_faces_per_block must be positive")
    if config.isolated_component_face_threshold < 1:
        raise BusinessError(
            ErrorCode.E_INTERNAL, "isolated_component_face_threshold must be positive"
        )
    if config.threads < 1:
        raise BusinessError(ErrorCode.E_INTERNAL, "threads must be at least one")
    if config.memory_budget_bytes < 1:
        raise BusinessError(ErrorCode.E_BUDGET_DENIED, "memory budget must be positive")
    if config.block_time_budget_ms <= 0:
        raise BusinessError(ErrorCode.E_BUDGET_DENIED, "block time budget must be positive")
    if config.volume_relative_tolerance < 0 or config.area_relative_tolerance < 0:
        raise BusinessError(ErrorCode.E_INTERNAL, "quality tolerances must be non-negative")
    return config


def load_config(path: str | Path) -> PipelineConfig:
    target = Path(path)
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BusinessError(
            ErrorCode.E_INTERNAL,
            "cannot read config file",
            details={"path": str(target), "error": str(exc)},
        ) from exc
    except json.JSONDecodeError as exc:
        raise BusinessError(
            ErrorCode.E_INTERNAL,
            "config file is not valid JSON",
            details={"path": str(target), "error": str(exc)},
        ) from exc
    if not isinstance(raw, dict):
        raise BusinessError(ErrorCode.E_INTERNAL, "config root must be an object")
    return config_from_dict(raw)
