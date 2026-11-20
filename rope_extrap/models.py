from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable

from .canonical import canonical_json, sha256_json


SCHEME_STANDARD = "standard_rope"
SCHEME_LINEAR = "linear_interpolation"
SCHEME_NTK = "ntk_aware"
SUPPORTED_SCHEMES = (SCHEME_STANDARD, SCHEME_LINEAR, SCHEME_NTK)

PRECISION_F64 = "float64"
PRECISION_F32 = "float32"
SUPPORTED_PRECISIONS = (PRECISION_F64, PRECISION_F32)

PRUNING_NONE = "none"
PRUNING_ENERGY = "energy"
PRUNING_FREQUENCY = "frequency"
SUPPORTED_PRUNING_POLICIES = (PRUNING_NONE, PRUNING_ENERGY, PRUNING_FREQUENCY)

STAGES = ("position_encoding", "attention_score", "error_trace")
STAGE_CODES = {
    "position_encoding": 10,
    "attention_score": 20,
    "error_trace": 30,
}


@dataclass(frozen=True)
class PositionEncodingConfig:
    scheme: str
    base: float
    training_context_length: int
    extrapolation_target_length: int
    dimension: int
    scale_factor: float
    precision: str
    dimension_chunk_size: int
    interaction_budget_ms: int
    concurrency: int
    snapshot_commit_interval: int
    pruning_tolerance: float
    pruning_policy: str = PRUNING_ENERGY
    error_tolerance: float = 0.02

    def changed(self, **changes: Any) -> "PositionEncodingConfig":
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scheme": self.scheme,
            "base": self.base,
            "training_context_length": self.training_context_length,
            "extrapolation_target_length": self.extrapolation_target_length,
            "dimension": self.dimension,
            "scale_factor": self.scale_factor,
            "precision": self.precision,
            "dimension_chunk_size": self.dimension_chunk_size,
            "interaction_budget_ms": self.interaction_budget_ms,
            "concurrency": self.concurrency,
            "snapshot_commit_interval": self.snapshot_commit_interval,
            "pruning_tolerance": self.pruning_tolerance,
            "pruning_policy": self.pruning_policy,
            "error_tolerance": self.error_tolerance,
        }

    def config_fingerprint(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True)
class AttentionSample:
    sample_id: str
    query_position: int
    key_positions: tuple[int, ...]
    query_features: tuple[float, ...]
    key_features: tuple[tuple[float, ...], ...]
    feature_line: int
    source_ref: str = "file"

    def content_fingerprint(self) -> str:
        # feature_line/source_ref are provenance only, not content.
        return sha256_json(
            {
                "sample_id": self.sample_id,
                "query_position": self.query_position,
                "key_positions": list(self.key_positions),
                "query_features": list(self.query_features),
                "key_features": [list(row) for row in self.key_features],
            }
        )

    def to_input_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "query_position": self.query_position,
            "key_positions": list(self.key_positions),
            "query_features": list(self.query_features),
            "key_features": [list(row) for row in self.key_features],
        }

    def with_source(self, *, feature_line: int, source_ref: str) -> "AttentionSample":
        return replace(self, feature_line=feature_line, source_ref=source_ref)


@dataclass(frozen=True)
class InteractionCommand:
    index: int
    command_id: str
    command_type: str
    changes: dict[str, Any]
    sample: AttentionSample | None = None
    budget_ms: int | None = None
    raw: dict[str, Any] | None = None


@dataclass(frozen=True)
class EncodedSample:
    query_position: int
    query_vector: tuple[float, ...]
    key_positions: tuple[int, ...]
    key_vectors: tuple[tuple[float, ...], ...]
    kept_chunks: tuple[int, ...]
    kept_dimensions: tuple[int, ...]
    pruned_dimensions: tuple[int, ...]
    effective_base: float
    position_scale: float
    scheme: str
    position_fingerprint: str


@dataclass(frozen=True)
class KeyScore:
    key_position: int
    relative_distance: int
    interval: str
    score: float
    attention_weight: float
    dominant_dimension: int
    dominant_contribution: float
    contributions: tuple[float, ...]


@dataclass(frozen=True)
class ScoreResult:
    entries: tuple[KeyScore, ...]
    has_extrapolation: bool
    sample_interval: str
    dominant_dimension: int
    dominant_contribution_abs: float
    raw_scores: tuple[float, ...]
    attention_weights: tuple[float, ...]
    score_fingerprint: str


@dataclass(frozen=True)
class KeyTrace:
    key_position: int
    relative_distance: int
    interval: str
    reference_query_position: int
    reference_key_position: int
    stage_deviations: dict[str, float]
    first_exceeded_stage: str | None
    max_deviation: float


@dataclass(frozen=True)
class TraceResult:
    reference_query_position: int
    key_traces: tuple[KeyTrace, ...]
    stage_max_deviations: dict[str, float]
    first_exceeded_stage: str | None
    violation: dict[str, Any] | None
    trace_fingerprint: str


@dataclass(frozen=True)
class CacheEntry:
    stage: str
    cache_key: str
    sample_id: str
    scheme: str
    feature_line: int
    source_ref: str
    result: Any

    def stable_sort_key(self) -> str:
        return f"{self.scheme}/{self.sample_id}/{self.stage}"

    def audit_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "stage_order": STAGE_CODES[self.stage],
            "cache_key": self.cache_key,
            "sample_id": self.sample_id,
            "scheme": self.scheme,
            "feature_line": self.feature_line,
            "source_ref": self.source_ref,
            "stable_sort_key": self.stable_sort_key(),
        }


def sorted_sample_ids(samples: Iterable[AttentionSample]) -> list[str]:
    return sorted(sample.sample_id for sample in samples)


def stage_sort_key(scheme: str, sample_id: str, stage: str) -> str:
    return f"{scheme}/{sample_id}/{STAGE_CODES[stage]:02d}_{stage}"


def canonical_feature_value(value: float) -> str:
    return float(value).hex()


def config_summary(config: PositionEncodingConfig) -> dict[str, Any]:
    return {
        "scheme": config.scheme,
        "base": config.base,
        "training_context_length": config.training_context_length,
        "extrapolation_target_length": config.extrapolation_target_length,
        "dimension": config.dimension,
        "scale_factor": config.scale_factor,
        "precision": config.precision,
        "dimension_chunk_size": config.dimension_chunk_size,
        "pruning_policy": config.pruning_policy,
        "pruning_tolerance": config.pruning_tolerance,
        "error_tolerance": config.error_tolerance,
        "concurrency": config.concurrency,
        "interaction_budget_ms": config.interaction_budget_ms,
        "snapshot_commit_interval": config.snapshot_commit_interval,
        "config_fingerprint": config.config_fingerprint(),
    }


def deterministic_payload(value: Any) -> str:
    """A debug helper that is also useful in tests and audit logs."""

    return canonical_json(value)
