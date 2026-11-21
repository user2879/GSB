"""Alignment configuration parsing, validation and fingerprints."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import E_CONFIG_INVALID, E_CONFIG_THRESHOLD, WorkbenchError
from .hashing import hash_json


def _pick(raw: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in raw:
            return raw[name]
    return default


def _as_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkbenchError(E_CONFIG_INVALID, f"{name} must be an integer")
    return value


def _as_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkbenchError(E_CONFIG_INVALID, f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise WorkbenchError(E_CONFIG_INVALID, f"{name} must be finite")
    return result


@dataclass(frozen=True)
class AlignmentConfig:
    schema_version: int
    chunk_size: int
    quantization_level: int
    quantization_levels: tuple[int, ...]
    visual_dim: int
    text_dim: int
    visual_weight: float
    text_weight: float
    similarity_threshold: float
    top_k: int
    fixed_decimal_bits: int
    thread_pool_size: int
    feature_point_budget: int
    snapshot_commit_interval: int
    required_attributes: tuple[str, ...]
    edge_threshold: float
    temperature: float
    boundary_neighborhood: float
    diagnostic_tolerance: float
    projection_salt: str
    raw: dict[str, Any]

    @property
    def effective_decimal_bits(self) -> int:
        return self.fixed_decimal_bits - self.quantization_level

    @property
    def fixed_scale(self) -> int:
        return 1 << self.effective_decimal_bits

    @property
    def weight_sum(self) -> float:
        return self.visual_weight + self.text_weight

    @property
    def visual_weight_normalized(self) -> float:
        return self.visual_weight / self.weight_sum

    @property
    def text_weight_normalized(self) -> float:
        return self.text_weight / self.weight_sum

    @property
    def config_hash(self) -> str:
        return hash_json(self.raw)

    def stage_visual_params(self) -> dict[str, Any]:
        return {
            "chunk_size": self.chunk_size,
            "visual_dim": self.visual_dim,
            "fixed_decimal_bits": self.fixed_decimal_bits,
            "quantization_level": self.quantization_level,
            "effective_decimal_bits": self.effective_decimal_bits,
            "edge_threshold": self.edge_threshold,
            "projection_salt": self.projection_salt,
        }

    def stage_text_params(self) -> dict[str, Any]:
        return {
            "text_dim": self.text_dim,
            "fixed_decimal_bits": self.fixed_decimal_bits,
            "quantization_level": self.quantization_level,
            "effective_decimal_bits": self.effective_decimal_bits,
            "required_attributes": list(self.required_attributes),
            "projection_salt": self.projection_salt,
        }

    def stage_alignment_params(self) -> dict[str, Any]:
        return {
            "visual_weight": self.visual_weight,
            "text_weight": self.text_weight,
            "visual_dim": self.visual_dim,
            "text_dim": self.text_dim,
            "temperature": self.temperature,
            "fixed_decimal_bits": self.fixed_decimal_bits,
            "quantization_level": self.quantization_level,
            "projection_salt": self.projection_salt,
        }

    def stage_retrieval_params(self) -> dict[str, Any]:
        return {
            "similarity_threshold": self.similarity_threshold,
            "top_k": self.top_k,
            "tie_breaker": "score_fixed_desc_then_sample_asc",
            "boundary_rule": "include_when_score_fixed_ge_threshold_fixed",
            "fixed_decimal_bits": self.fixed_decimal_bits,
            "quantization_level": self.quantization_level,
        }

    def with_quantization_level(self, level: int) -> "AlignmentConfig":
        raw = dict(self.raw)
        raw["quantization_level"] = level
        if "quantization_levels" not in raw:
            raw["quantization_levels"] = list(self.quantization_levels)
        return AlignmentConfig.from_mapping(raw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "chunk_size": self.chunk_size,
            "quantization_level": self.quantization_level,
            "quantization_levels": list(self.quantization_levels),
            "visual_dim": self.visual_dim,
            "text_dim": self.text_dim,
            "visual_weight": self.visual_weight,
            "text_weight": self.text_weight,
            "similarity_threshold": self.similarity_threshold,
            "top_k": self.top_k,
            "fixed_decimal_bits": self.fixed_decimal_bits,
            "thread_pool_size": self.thread_pool_size,
            "feature_point_budget": self.feature_point_budget,
            "snapshot_commit_interval": self.snapshot_commit_interval,
            "required_attributes": list(self.required_attributes),
            "edge_threshold": self.edge_threshold,
            "temperature": self.temperature,
            "boundary_neighborhood": self.boundary_neighborhood,
            "diagnostic_tolerance": self.diagnostic_tolerance,
            "projection_salt": self.projection_salt,
        }

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "AlignmentConfig":
        if not isinstance(raw, dict):
            raise WorkbenchError(E_CONFIG_INVALID, "configuration must be an object")
        schema_version = _as_int(raw.get("schema_version", 1), "schema_version")
        if schema_version != 1:
            raise WorkbenchError(E_CONFIG_INVALID, "unsupported schema_version")

        chunk_size = _as_int(
            _pick(raw, "chunk_size", "image_chunk_size", "patch_size"), "chunk_size"
        )
        quantization_level = _as_int(
            _pick(raw, "quantization_level", "quant_level", default=0),
            "quantization_level",
        )
        levels_raw = _pick(raw, "quantization_levels", "quant_levels")
        if levels_raw is None:
            levels = (quantization_level,)
        else:
            if not isinstance(levels_raw, list) or not levels_raw:
                raise WorkbenchError(E_CONFIG_INVALID, "quantization_levels must be a non-empty list")
            levels = tuple(_as_int(item, "quantization_levels[]") for item in levels_raw)
        levels = tuple(sorted(set(levels)))
        if quantization_level not in levels:
            levels = tuple(sorted(set(levels) | {quantization_level}))

        visual_dim = _as_int(
            _pick(raw, "visual_dim", "visual_feature_dim", "image_feature_dim"),
            "visual_dim",
        )
        text_dim = _as_int(
            _pick(raw, "text_dim", "text_feature_dim", "language_feature_dim"),
            "text_dim",
        )
        visual_weight = _as_float(
            _pick(raw, "visual_weight", "align_weight_visual", "image_weight"),
            "visual_weight",
        )
        text_weight = _as_float(
            _pick(raw, "text_weight", "align_weight_text", "language_weight"),
            "text_weight",
        )
        threshold_raw = _pick(
            raw, "similarity_threshold", "retrieval_threshold", "threshold", default=0.0
        )
        try:
            similarity_threshold = _as_float(threshold_raw, "similarity_threshold")
        except WorkbenchError as exc:
            raise WorkbenchError(E_CONFIG_THRESHOLD, str(exc)) from exc
        top_k = _as_int(_pick(raw, "top_k", "retrieval_top_k", "return_k"), "top_k")
        fixed_decimal_bits = _as_int(
            _pick(raw, "fixed_decimal_bits", "fixed_bits", "decimal_bits"),
            "fixed_decimal_bits",
        )
        thread_pool_size = _as_int(
            _pick(raw, "thread_pool_size", "thread_pool_concurrency", "max_workers"),
            "thread_pool_size",
        )
        feature_point_budget = _as_int(
            _pick(raw, "feature_point_budget", "batch_feature_point_budget", "point_budget"),
            "feature_point_budget",
        )
        snapshot_commit_interval = _as_int(
            _pick(raw, "snapshot_commit_interval", "commit_interval"),
            "snapshot_commit_interval",
        )
        required_attributes_raw = _pick(
            raw, "required_attributes", default=["color", "shape", "position"]
        )
        if (
            not isinstance(required_attributes_raw, list)
            or not required_attributes_raw
            or any(not isinstance(item, str) or not item for item in required_attributes_raw)
        ):
            raise WorkbenchError(E_CONFIG_INVALID, "required_attributes must be non-empty strings")
        edge_threshold = _as_float(_pick(raw, "edge_threshold", default=0.08), "edge_threshold")
        temperature = _as_float(_pick(raw, "temperature", default=0.25), "temperature")
        boundary_neighborhood = _as_float(
            _pick(raw, "boundary_neighborhood", default=0.02), "boundary_neighborhood"
        )
        diagnostic_tolerance = _as_float(
            _pick(raw, "diagnostic_tolerance", default=1e-6), "diagnostic_tolerance"
        )
        projection_salt = _pick(raw, "projection_salt", default="cm-align-v1")
        if not isinstance(projection_salt, str) or not projection_salt:
            raise WorkbenchError(E_CONFIG_INVALID, "projection_salt must be a non-empty string")

        if chunk_size <= 0:
            raise WorkbenchError(E_CONFIG_INVALID, "chunk_size must be positive")
        if visual_dim < 4 or text_dim < 4 or visual_dim > 512 or text_dim > 512:
            raise WorkbenchError(E_CONFIG_INVALID, "feature dimensions must be in [4, 512]")
        if visual_weight < 0 or text_weight < 0 or visual_weight + text_weight <= 0:
            raise WorkbenchError(E_CONFIG_INVALID, "alignment weights must be non-negative with a positive sum")
        if not 0.0 <= similarity_threshold <= 1.0:
            raise WorkbenchError(
                E_CONFIG_THRESHOLD,
                "similarity_threshold must be in [0, 1]",
                {"value": similarity_threshold},
            )
        if top_k <= 0:
            raise WorkbenchError(E_CONFIG_INVALID, "top_k must be positive")
        if not 1 <= fixed_decimal_bits <= 24:
            raise WorkbenchError(E_CONFIG_INVALID, "fixed_decimal_bits must be in [1, 24]")
        if quantization_level < 0 or any(level < 0 for level in levels):
            raise WorkbenchError(E_CONFIG_INVALID, "quantization levels must be non-negative")
        if quantization_level >= fixed_decimal_bits:
            raise WorkbenchError(
                E_CONFIG_INVALID, "quantization_level must be smaller than fixed_decimal_bits"
            )
        if thread_pool_size <= 0 or thread_pool_size > 64:
            raise WorkbenchError(E_CONFIG_INVALID, "thread_pool_size must be in [1, 64]")
        if feature_point_budget < 0:
            raise WorkbenchError(E_CONFIG_INVALID, "feature_point_budget must be non-negative")
        if snapshot_commit_interval <= 0:
            raise WorkbenchError(E_CONFIG_INVALID, "snapshot_commit_interval must be positive")
        if not 0.0 <= edge_threshold <= 1.0:
            raise WorkbenchError(E_CONFIG_INVALID, "edge_threshold must be in [0, 1]")
        if temperature <= 0:
            raise WorkbenchError(E_CONFIG_INVALID, "temperature must be positive")
        if boundary_neighborhood < 0:
            raise WorkbenchError(E_CONFIG_INVALID, "boundary_neighborhood must be non-negative")
        if diagnostic_tolerance <= 0:
            raise WorkbenchError(E_CONFIG_INVALID, "diagnostic_tolerance must be positive")

        return cls(
            schema_version=schema_version,
            chunk_size=chunk_size,
            quantization_level=quantization_level,
            quantization_levels=levels,
            visual_dim=visual_dim,
            text_dim=text_dim,
            visual_weight=visual_weight,
            text_weight=text_weight,
            similarity_threshold=similarity_threshold,
            top_k=top_k,
            fixed_decimal_bits=fixed_decimal_bits,
            thread_pool_size=thread_pool_size,
            feature_point_budget=feature_point_budget,
            snapshot_commit_interval=snapshot_commit_interval,
            required_attributes=tuple(required_attributes_raw),
            edge_threshold=edge_threshold,
            temperature=temperature,
            boundary_neighborhood=boundary_neighborhood,
            diagnostic_tolerance=diagnostic_tolerance,
            projection_salt=projection_salt,
            raw=json.loads(json.dumps(raw)),
        )

    @classmethod
    def load(cls, path: Path | str) -> "AlignmentConfig":
        path = Path(path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise WorkbenchError(E_CONFIG_INVALID, f"configuration not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise WorkbenchError(E_CONFIG_INVALID, f"invalid configuration JSON: {exc}") from exc
        return cls.from_mapping(raw)

    def adjacent_quantization_levels(self) -> tuple[int, ...]:
        lower = [level for level in self.quantization_levels if level < self.quantization_level]
        higher = [level for level in self.quantization_levels if level > self.quantization_level]
        adjacent: list[int] = []
        if lower:
            adjacent.append(max(lower))
        if higher:
            adjacent.append(min(higher))
        return tuple(
            level
            for level in adjacent
            if 0 <= self.fixed_decimal_bits - level
        )
