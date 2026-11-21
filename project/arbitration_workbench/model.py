"""Dataclasses for rules, evidence and configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from .numeric import decimal_text


@dataclass(frozen=True)
class Evidence:
    ticket_id: str
    evidence_seq: int
    evidence_type: str
    source_role: str
    timestamp: int
    terms: tuple[str, ...]
    line_no: int = 0

    def business_dict(self) -> dict:
        return {
            "ticket_id": self.ticket_id,
            "evidence_seq": self.evidence_seq,
            "evidence_type": self.evidence_type,
            "source_role": self.source_role,
            "timestamp": self.timestamp,
            "terms": list(self.terms),
        }

    def jsonl_dict(self) -> dict:
        data = self.business_dict()
        return data


@dataclass(frozen=True)
class Selector:
    evidence_type: str
    source_roles: tuple[str, ...] = ()

    def matches(self, evidence: Evidence) -> bool:
        if evidence.evidence_type != self.evidence_type:
            return False
        return not self.source_roles or evidence.source_role in self.source_roles

    def as_dict(self) -> dict:
        return {
            "evidence_type": self.evidence_type,
            "source_roles": list(self.source_roles),
        }


@dataclass(frozen=True)
class Relation:
    relation_id: str
    kind: str
    left: Selector
    right: Selector
    bonus: Decimal | None = None

    def as_dict(self) -> dict:
        data = {
            "id": self.relation_id,
            "kind": self.kind,
            "left": self.left.as_dict(),
            "right": self.right.as_dict(),
        }
        if self.bonus is not None:
            data["bonus"] = decimal_text(self.bonus)
        return data


@dataclass(frozen=True)
class PriorityBand:
    level: int
    name: str
    min_score: Decimal

    def as_dict(self) -> dict:
        return {
            "level": self.level,
            "name": self.name,
            "min_score": decimal_text(self.min_score),
        }


@dataclass(frozen=True)
class Rules:
    schema: str
    version: str
    weights: dict[str, Decimal]
    trusts: dict[str, Decimal]
    relations: tuple[Relation, ...]
    priority_bands: tuple[PriorityBand, ...]

    def as_dict(self) -> dict:
        return {
            "schema": self.schema,
            "version": self.version,
            "evidence_type_weights": {k: decimal_text(v) for k, v in sorted(self.weights.items())},
            "role_trust": {k: decimal_text(v) for k, v in sorted(self.trusts.items())},
            "relations": [relation.as_dict() for relation in self.relations],
            "priority_bands": [band.as_dict() for band in self.priority_bands],
        }


@dataclass(frozen=True)
class ArbitrationConfig:
    fixed_decimal_places: int
    quantization_units: int
    decay_half_life: int
    exclusion_strength: Decimal
    support_bonus: Decimal
    fusion: str
    decision_threshold: Decimal
    interaction_budget_ms: int
    worker_threads: int
    snapshot_commit_interval: int
    snapshot_dir: str
    temp_dir: str
    target_version: str = "v2"
    threshold_equality_upgrades: bool = True

    @property
    def scale(self) -> int:
        return 10 ** self.fixed_decimal_places

    def as_dict(self) -> dict:
        return {
            "fixed_decimal_places": self.fixed_decimal_places,
            "quantization_units": self.quantization_units,
            "decay_half_life": self.decay_half_life,
            "exclusion_strength": decimal_text(self.exclusion_strength),
            "support_bonus": decimal_text(self.support_bonus),
            "fusion": self.fusion,
            "decision_threshold": decimal_text(self.decision_threshold),
            "interaction_budget_ms": self.interaction_budget_ms,
            "worker_threads": self.worker_threads,
            "snapshot_commit_interval": self.snapshot_commit_interval,
            "snapshot_dir": self.snapshot_dir,
            "temp_dir": self.temp_dir,
            "target_version": self.target_version,
            "threshold_equality_upgrades": self.threshold_equality_upgrades,
        }


@dataclass
class EvidenceLoadResult:
    evidence_by_ticket: dict[str, tuple[Evidence, ...]] = field(default_factory=dict)
    failures: list[dict] = field(default_factory=list)
    ticket_ids: list[str] = field(default_factory=list)

