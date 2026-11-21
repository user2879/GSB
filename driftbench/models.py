from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from .errors import Failure
from .fingerprint import canonical_json, digest


SOURCE_ORDER = (
    "truncation_rank",
    "normalization",
    "dedup_count",
    "sample_filter",
)

EVENT_FIELDS = (
    "request_id",
    "experiment_group",
    "user_id",
    "item_id",
    "rank",
    "model_score",
    "clicked",
    "converted",
    "event_ms",
)

METRIC_ALIASES = {
    "ctr": "ctr",
    "click_through_rate": "ctr",
    "click_rate": "ctr",
    "cvr": "cvr",
    "conversion_rate": "cvr",
}

NORMALIZATION_ALIASES = {
    "none": "none",
    "raw": "none",
    "rank_inverse": "rank_inverse",
    "reciprocal_rank": "rank_inverse",
    "position_decay": "position_decay",
    "log_decay": "position_decay",
}

FILTER_OPS = {"eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in"}


@dataclass(frozen=True)
class FilterCondition:
    field: str
    op: str
    value: Any

    def to_dict(self) -> dict[str, Any]:
        return {"field": self.field, "op": self.op, "value": self.value}


@dataclass(frozen=True)
class Spec:
    version: str
    metric_name: str
    truncation_rank: int
    normalization: str
    dedup_count: bool
    sample_filter: tuple[FilterCondition, ...] = ()

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "metric_name": self.metric_name,
            "truncation_rank": self.truncation_rank,
            "normalization": self.normalization,
            "dedup_count": self.dedup_count,
            "sample_filter": [condition.to_dict() for condition in self.sample_filter],
        }

    def fingerprint(self) -> str:
        return digest(self.canonical_dict())

    def to_dict(self) -> dict[str, Any]:
        return self.canonical_dict()


@dataclass(frozen=True)
class Exposure:
    request_id: str
    experiment_group: str
    user_id: str
    item_id: str
    rank: int
    model_score: float
    clicked: int
    converted: int
    event_ms: int

    @property
    def stable_id(self) -> str:
        return f"{self.request_id}|{self.item_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "experiment_group": self.experiment_group,
            "user_id": self.user_id,
            "item_id": self.item_id,
            "rank": self.rank,
            "model_score": self.model_score,
            "clicked": self.clicked,
            "converted": self.converted,
            "event_ms": self.event_ms,
        }

    def content_dict(self) -> dict[str, Any]:
        return self.to_dict()


@dataclass(frozen=True)
class Config:
    control_group: str
    experiment_group: str
    bucket_hash_seed: str
    bucket_count: int
    active_spec_version: str
    fixed_decimal_places: int
    quantization_step: int
    concurrency: int
    batch_size: int
    snapshot_commit_interval_batches: int
    bucket_property_key: str
    fallback_policy: str
    missing_attribute_group: str | None
    drift_tolerance: Decimal
    min_affected_ratio: Decimal
    attribution_order: tuple[str, ...]
    snapshot_version: int

    @property
    def scale(self) -> int:
        return 10**self.fixed_decimal_places

    @property
    def group_order(self) -> tuple[str, str]:
        return self.control_group, self.experiment_group

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "control_group": self.control_group,
            "experiment_group": self.experiment_group,
            "bucket_hash_seed": self.bucket_hash_seed,
            "bucket_count": self.bucket_count,
            "active_spec_version": self.active_spec_version,
            "fixed_decimal_places": self.fixed_decimal_places,
            "quantization_step": self.quantization_step,
            "concurrency": self.concurrency,
            "batch_size": self.batch_size,
            "snapshot_commit_interval_batches": self.snapshot_commit_interval_batches,
            "bucket_property_key": self.bucket_property_key,
            "fallback_policy": self.fallback_policy,
            "missing_attribute_group": self.missing_attribute_group,
            "drift_tolerance": str(self.drift_tolerance),
            "min_affected_ratio": str(self.min_affected_ratio),
            "attribution_order": list(self.attribution_order),
            "snapshot_version": self.snapshot_version,
        }

    def fingerprint(self) -> str:
        return digest(self.canonical_dict())

    def to_dict(self) -> dict[str, Any]:
        return self.canonical_dict()


@dataclass
class WorkspaceState:
    config: Config
    specs: dict[str, Spec]
    exposures: dict[str, Exposure]
    properties: dict[str, dict[str, Any]]
    plan: dict[str, Any] = field(default_factory=dict)
    failures: list[Failure] = field(default_factory=list)
    input_paths: dict[str, str] = field(default_factory=dict)

    def clone(self) -> "WorkspaceState":
        return WorkspaceState(
            config=self.config,
            specs=dict(self.specs),
            exposures=dict(self.exposures),
            properties={user: dict(values) for user, values in self.properties.items()},
            plan=self.plan,
            failures=list(self.failures),
            input_paths=dict(self.input_paths),
        )

    def sorted_exposures(self) -> list[Exposure]:
        return sorted(self.exposures.values(), key=lambda event: (event.request_id, event.item_id))

    def spec_versions(self) -> list[str]:
        return sorted(self.specs, key=natural_version_key)

    def state_fingerprint(self) -> str:
        return digest(
            {
                "config": self.config.canonical_dict(),
                "specs": [self.specs[key].canonical_dict() for key in self.spec_versions()],
                "events": [event.content_dict() for event in self.sorted_exposures()],
                "properties": self.properties,
            }
        )


def natural_version_key(version: str) -> tuple:
    chunks: list[Any] = []
    current = ""
    for char in version:
        if char.isdigit():
            current += char
        else:
            if current:
                chunks.append(int(current))
                current = ""
            chunks.append(char)
    if current:
        chunks.append(int(current))
    return tuple(chunks)


def changed_spec_sources(before: Spec, after: Spec) -> tuple[str, ...]:
    changed: list[str] = []
    if before.truncation_rank != after.truncation_rank:
        changed.append("truncation_rank")
    if before.normalization != after.normalization:
        changed.append("normalization")
    if before.dedup_count != after.dedup_count:
        changed.append("dedup_count")
    if canonical_json([condition.to_dict() for condition in before.sample_filter]) != canonical_json(
        [condition.to_dict() for condition in after.sample_filter]
    ):
        changed.append("sample_filter")
    return tuple(changed)
