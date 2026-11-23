"""Immutable data models shared by the loader, scheduler and engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from .canonical import canonical_data, digest
from .errors import Failure

MAX_CONCURRENCY = 32
VALID_NORMALIZATIONS = ("minmax", "zscore")


@dataclass(frozen=True)
class FeatureGroup:
    name: str
    features: tuple[str, ...]
    conflict: bool = False


@dataclass(frozen=True)
class FeatureCatalog:
    feature_order: tuple[str, ...]
    groups: tuple[FeatureGroup, ...]
    similarity_features: tuple[str, ...]
    similarity_threshold: Fraction
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def dimension(self) -> int:
        return len(self.feature_order)

    def group_for(self, feature: str) -> str | None:
        for group in self.groups:
            if feature in group.features:
                return group.name
        return None

    @property
    def similarity_indices(self) -> tuple[int, ...]:
        index = {name: idx for idx, name in enumerate(self.feature_order)}
        return tuple(index[name] for name in self.similarity_features)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_order": list(self.feature_order),
            "similarity_features": list(self.similarity_features),
            "similarity_threshold": str(self.similarity_threshold),
            "groups": [
                {
                    "name": group.name,
                    "features": list(group.features),
                    "conflict": group.conflict,
                }
                for group in self.groups
            ],
            "hash": digest(self.raw),
        }


@dataclass(frozen=True)
class LabConfig:
    feature_weights: tuple[Fraction, ...]
    normalization: str
    fusion_temperature: Fraction
    score_min: Fraction
    score_max: Fraction
    fixed_decimal_places: int
    quantization_step: int
    rerank_window_size: int
    diversity_penalty: Fraction
    concurrency: int
    global_mac_budget: int
    per_request_quota: int
    cache_residency_budget: int
    snapshot_commit_interval_commands: int
    interaction_latency_budget_ms: Fraction = Fraction(50)
    min_command_batch: int = 1
    latency_units_per_ms: Fraction = Fraction(1000)
    similarity_threshold: Fraction = Fraction(8, 10)
    raw_extra: dict[str, Any] = field(default_factory=dict)

    def with_weights(self, weights: tuple[Fraction, ...]) -> "LabConfig":
        return LabConfig(**{**self.__dict__, "feature_weights": weights})

    def with_values(self, **changes: Any) -> "LabConfig":
        return LabConfig(**{**self.__dict__, **changes})

    def quantisation_dict(self) -> dict[str, Any]:
        return {
            "fixed_decimal_places": self.fixed_decimal_places,
            "quantization_step": self.quantization_step,
        }

    def fusion_dict(self) -> dict[str, Any]:
        return {
            "feature_weights": [str(weight) for weight in self.feature_weights],
            "fusion_temperature": str(self.fusion_temperature),
            "score_min": str(self.score_min),
            "score_max": str(self.score_max),
        }

    def rerank_dict(self) -> dict[str, Any]:
        return {
            "rerank_window_size": self.rerank_window_size,
            "diversity_penalty": str(self.diversity_penalty),
            "similarity_threshold": str(self.similarity_threshold),
        }

    def to_dict(self) -> dict[str, Any]:
        out = {
            "feature_weights": [str(weight) for weight in self.feature_weights],
            "normalization": self.normalization,
            "fusion_temperature": str(self.fusion_temperature),
            "score_min": str(self.score_min),
            "score_max": str(self.score_max),
            "fixed_decimal_places": self.fixed_decimal_places,
            "quantization_step": self.quantization_step,
            "rerank_window_size": self.rerank_window_size,
            "diversity_penalty": str(self.diversity_penalty),
            "concurrency": self.concurrency,
            "global_mac_budget": self.global_mac_budget,
            "per_request_quota": self.per_request_quota,
            "cache_residency_budget": self.cache_residency_budget,
            "snapshot_commit_interval_commands": self.snapshot_commit_interval_commands,
            "interaction_latency_budget_ms": str(self.interaction_latency_budget_ms),
            "min_command_batch": self.min_command_batch,
            "latency_units_per_ms": str(self.latency_units_per_ms),
            "similarity_threshold": str(self.similarity_threshold),
        }
        if self.raw_extra:
            out["extra"] = canonical_data(self.raw_extra)
        return out

    @property
    def hash(self) -> str:
        return digest({"config": self.to_dict()})

    @property
    def fixed_scale(self) -> int:
        return 10**self.fixed_decimal_places


@dataclass(frozen=True)
class RawCandidate:
    request_id: str
    item_id: str
    features: dict[str, Fraction]
    recall_scores: dict[str, Fraction]
    dense_features: tuple[Fraction, ...]
    sparse_features: dict[str, int]
    line: int
    source_record: dict[str, Any]

    def feature_vector(self, catalog: FeatureCatalog) -> tuple[int | Fraction, ...]:
        return tuple(self.features.get(name, Fraction(0)) for name in catalog.feature_order)

    def content_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "item_id": self.item_id,
            "features": {name: str(value) for name, value in sorted(self.features.items())},
        }

    @property
    def content_fingerprint(self) -> str:
        return digest(self.content_dict())


@dataclass
class LoadedSample:
    candidates: list[RawCandidate]
    config: LabConfig | None
    catalog: FeatureCatalog
    commands: list[dict[str, Any]]
    injections: list[dict[str, Any]]
    failures: list[Failure]
    config_valid: bool
    raw_config: dict[str, Any]
    raw_groups: dict[str, Any]
    input_files: dict[str, str]
    input_hashes: dict[str, str]
    command_failures: list[Failure] = field(default_factory=list)

    def grouped_candidates(self) -> dict[str, dict[str, RawCandidate]]:
        grouped: dict[str, dict[str, RawCandidate]] = {}
        for candidate in self.candidates:
            grouped.setdefault(candidate.request_id, {})[candidate.item_id] = candidate
        return {
            request_id: {
                item_id: grouped[request_id][item_id]
                for item_id in sorted(grouped[request_id])
            }
            for request_id in sorted(grouped)
        }

    @property
    def candidates_hash(self) -> str:
        return digest(
            [
                candidate.content_dict()
                for candidate in sorted(
                    self.candidates, key=lambda c: (c.request_id, c.item_id)
                )
            ]
        )


@dataclass
class RequestPlan:
    request_id: str
    candidates: tuple[RawCandidate, ...]
    request_digest: str
    stats_key: str
    stats_entry: dict[str, Any] | None
    fusion_keys: dict[str, str]
    fusion_entries: dict[str, dict[str, Any]]
    rerank_key: str
    rerank_entry: dict[str, Any] | None
    final_key: str
    final_entry: dict[str, Any] | None
    missing_fusion: list[str]
    need_stats: bool
    need_rerank: bool
    cost: int
    admitted: bool = False
    budget_status: str = "new"
    failure: Failure | None = None


@dataclass
class RequestExecution:
    plan: RequestPlan
    pure_result: dict[str, Any] | None
    telemetry: dict[str, Any]
    failure: Failure | None = None


@dataclass
class Evaluation:
    version: int
    requests: dict[str, dict[str, Any]]
    plans: dict[str, RequestPlan]
    failures: list[Failure]
    pending: list[dict[str, Any]]
    admitted: list[str]
    completed: list[str]
    failed: list[str]
    mac_units_charged: int
    mac_units_remaining: int
    evictions: list[dict[str, Any]]
    reuse_requests: list[str]
    recompute_requests: list[str]
    affected_requests: list[str] = field(default_factory=list)
    timed_out: bool = False
    logical_latency_ms: Fraction = Fraction(0)

    def canonical_result_dicts(self) -> dict[str, dict[str, Any]]:
        return {request_id: self.requests[request_id] for request_id in sorted(self.requests)}
