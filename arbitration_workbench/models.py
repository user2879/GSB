"""Validated input models."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from .canonical import decimal_text, sha256_obj


@dataclass(frozen=True)
class Evidence:
    workorder_id: str
    evidence_seq: int
    evidence_type: str
    source_role: str
    event_ms: Any
    terms: Any
    confidence: Decimal
    confidence_text: str
    line_no: int

    def canonical(self) -> dict[str, Any]:
        return {
            "workorder_id": self.workorder_id,
            "evidence_seq": self.evidence_seq,
            "evidence_type": self.evidence_type,
            "source_role": self.source_role,
            "event_ms": self.event_ms,
            "terms": list(self.terms) if isinstance(self.terms, list) else self.terms,
            "confidence": self.confidence_text,
        }

    def fingerprint(self) -> str:
        return sha256_obj(self.canonical())


@dataclass(frozen=True)
class Relation:
    relation_id: str
    kind: str
    left_type: str
    right_type: str
    strength: Decimal | None = None

    def canonical(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.relation_id,
            "kind": self.kind,
            "left_type": self.left_type,
            "right_type": self.right_type,
        }
        if self.strength is not None:
            result["strength"] = decimal_text(self.strength)
        return result


@dataclass(frozen=True)
class Threshold:
    priority: str
    min_score: Decimal

    def canonical(self) -> dict[str, str]:
        return {"priority": self.priority, "min_score": decimal_text(self.min_score)}


@dataclass(frozen=True)
class RulePackage:
    weights: dict[str, Decimal]
    role_credibility: dict[str, Decimal]
    relations: tuple[Relation, ...]
    half_life_ms: int
    thresholds: tuple[Threshold, ...]
    normalized: dict[str, Any]
    fingerprint: str


@dataclass(frozen=True)
class ArbitrationConfig:
    concurrency: int
    batch_evidence_budget: int
    workorder_evidence_quota: int
    fixed_decimal_width: int
    quantization_step: int
    snapshot_commit_interval: int
    exclusion_strength: Decimal
    support_strength: Decimal
    zero_lower_bound: Decimal
    fusion_mode: str
    normalized: dict[str, Any]
    config_hash: str

    @property
    def scale(self) -> int:
        return 10**self.fixed_decimal_width


@dataclass(frozen=True)
class LoadedInputs:
    config: ArbitrationConfig
    rules: RulePackage
    evidence: tuple[Evidence, ...]
    evidence_failures: tuple[dict[str, Any], ...]
    input_version: int


@dataclass(frozen=True)
class Injection:
    injection_id: str
    workorder_id: str
    stage: str
    fault: str
    evidence_seq: int | None = None


@dataclass
class WorkOrderContext:
    workorder_id: str
    evidence: list[Evidence]
    load_failures: list[dict[str, Any]]
    injection_hits: list[dict[str, Any]]
    extra_relations: list[Relation]
    effective_half_life_ms: int
    evidence_fingerprint: str
    previous_record: dict[str, Any] | None = None
    decay_key: str | None = None
    decay_payload: dict[str, Any] | None = None
    relation_key: str | None = None
    relation_payload: dict[str, Any] | None = None
    aggregate_key: str | None = None
    aggregate_payload: dict[str, Any] | None = None
    references: dict[str, Any] | None = None
    failure: dict[str, Any] | None = None
    stage_computed: dict[str, bool] | None = None

    def __post_init__(self) -> None:
        if self.stage_computed is None:
            self.stage_computed = {"decay": False, "relations": False, "aggregation": False}
