"""Immutable domain models shared by all pipeline stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class Slot:
    name: str
    start: int
    end: int

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "start": self.start, "end": self.end}


@dataclass(frozen=True)
class Query:
    query_id: str
    terms: tuple[str, ...]
    session_id: str
    turn: int
    source: str
    ground_truth_intent: str
    candidates: tuple[str, ...]
    slots: tuple[Slot, ...]
    priority: int = 0
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "text": self.text,
            "terms": list(self.terms),
            "session_id": self.session_id,
            "turn": self.turn,
            "source": self.source,
            "ground_truth_intent": self.ground_truth_intent,
            "candidates": list(self.candidates),
            "slots": [slot.to_dict() for slot in self.slots],
            "priority": self.priority,
        }


@dataclass(frozen=True)
class RevisionOp:
    op: str
    trigger_id: str
    terms: tuple[str, ...] | None = None
    intent_id: str | None = None
    weight: int | None = None
    slot_pattern: tuple[str, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"op": self.op, "trigger_id": self.trigger_id}
        if self.terms is not None:
            result["terms"] = list(self.terms)
        if self.intent_id is not None:
            result["intent_id"] = self.intent_id
        if self.weight is not None:
            result["weight"] = self.weight
        if self.slot_pattern is not None:
            result["slot_pattern"] = list(self.slot_pattern)
        return result


@dataclass(frozen=True)
class Revision:
    revision_id: str
    parent_id: str | None
    seq: int
    label: str
    ops: tuple[RevisionOp, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision_id": self.revision_id,
            "parent_id": self.parent_id,
            "seq": self.seq,
            "label": self.label,
            "ops": [op.to_dict() for op in self.ops],
        }


@dataclass(frozen=True)
class TriggerEntry:
    trigger_id: str
    terms: tuple[str, ...]
    intent_id: str
    weight: int
    slot_pattern: tuple[str, ...]
    source_revision: str
    entry_order: int
    provenance: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "trigger_id": self.trigger_id,
            "terms": list(self.terms),
            "intent_id": self.intent_id,
            "weight": self.weight,
            "slot_pattern": list(self.slot_pattern),
            "source_revision": self.source_revision,
            "entry_order": self.entry_order,
            "provenance": list(self.provenance),
        }


@dataclass(frozen=True)
class RuleView:
    entries: tuple[TriggerEntry, ...]
    active_revision_ids: tuple[str, ...]
    lineage_revision_ids: tuple[str, ...]
    chain_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_revision_ids": list(self.active_revision_ids),
            "lineage_revision_ids": list(self.lineage_revision_ids),
            "chain_version": self.chain_version,
            "entries": [entry.to_dict() for entry in self.entries],
        }


@dataclass(frozen=True)
class MatchSpan:
    trigger_id: str
    terms: tuple[str, ...]
    intent_id: str
    start: int
    end: int
    char_start: int
    char_end: int
    entry_order: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "trigger_id": self.trigger_id,
            "terms": list(self.terms),
            "intent_id": self.intent_id,
            "start": self.start,
            "end": self.end,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "entry_order": self.entry_order,
        }


@dataclass(frozen=True)
class MatchResult:
    query_id: str
    selected: tuple[MatchSpan, ...]
    suppressed: tuple[dict[str, Any], ...]
    considered_trigger_ids: tuple[str, ...]
    char_length: int
    term_count: int
    cache_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "selected": [span.to_dict() for span in self.selected],
            "suppressed": list(self.suppressed),
            "considered_trigger_ids": list(self.considered_trigger_ids),
            "char_length": self.char_length,
            "term_count": self.term_count,
            "cache_key": self.cache_key,
        }


@dataclass(frozen=True)
class Contribution:
    trigger_id: str
    intent_id: str
    weight: int
    hit_count: int
    numerator: int
    contribution: int
    entry_order: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IntentScore:
    intent_id: str
    raw_score: int
    score: int
    threshold: int
    decision: bool
    boundary: bool
    margin: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScoreResult:
    query_id: str
    predicted_intent: str
    scores: tuple[IntentScore, ...]
    contributions: tuple[Contribution, ...]
    referenced_trigger_ids: tuple[str, ...]
    boundary: bool
    margin: int
    fraction_bits: int
    quantum: int
    cache_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "predicted_intent": self.predicted_intent,
            "scores": [score.to_dict() for score in self.scores],
            "contributions": [item.to_dict() for item in self.contributions],
            "referenced_trigger_ids": list(self.referenced_trigger_ids),
            "boundary": self.boundary,
            "margin": self.margin,
            "fraction_bits": self.fraction_bits,
            "quantum": self.quantum,
            "cache_key": self.cache_key,
        }


@dataclass(frozen=True)
class AlignedSlot:
    name: str
    query_id: str
    local_start: int
    local_end: int
    start: int
    end: int
    status: str
    conflict_with: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Segment:
    segment_id: str
    session_id: str
    intent: str
    query_ids: tuple[str, ...]
    start_turn: int
    end_turn: int
    terms: tuple[str, ...]
    aligned_slots: tuple[AlignedSlot, ...]
    conflicts: tuple[dict[str, Any], ...]
    cache_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "session_id": self.session_id,
            "intent": self.intent,
            "query_ids": list(self.query_ids),
            "start_turn": self.start_turn,
            "end_turn": self.end_turn,
            "terms": list(self.terms),
            "aligned_slots": [slot.to_dict() for slot in self.aligned_slots],
            "conflicts": list(self.conflicts),
            "cache_key": self.cache_key,
        }


@dataclass(frozen=True)
class RuntimeConfig:
    fraction_bits: int
    quantum: int
    allowed_fraction_bits: tuple[int, ...]
    allowed_quantums: tuple[int, ...]
    allowed_weights: tuple[int, ...]
    thresholds: dict[str, int]
    default_threshold: int
    tie_rule: str
    fallback_intent: str
    tolerance: float
    reference_extra_bits: int
    workers: int
    match_budget_points: int
    source_quotas: dict[str, int]
    batch_query_limit: int
    max_turn_gap: int
    term_gap: int
    global_version: int
    raw: dict[str, Any] = field(default_factory=dict)

    def threshold_for(self, intent_id: str) -> int:
        return self.thresholds.get(intent_id, self.default_threshold)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fraction_bits": self.fraction_bits,
            "quantum": self.quantum,
            "allowed_fraction_bits": list(self.allowed_fraction_bits),
            "allowed_quantums": list(self.allowed_quantums),
            "allowed_weights": list(self.allowed_weights),
            "thresholds": dict(self.thresholds),
            "default_threshold": self.default_threshold,
            "tie_rule": self.tie_rule,
            "fallback_intent": self.fallback_intent,
            "tolerance": self.tolerance,
            "reference_extra_bits": self.reference_extra_bits,
            "workers": self.workers,
            "match_budget_points": self.match_budget_points,
            "source_quotas": dict(self.source_quotas),
            "batch_query_limit": self.batch_query_limit,
            "max_turn_gap": self.max_turn_gap,
            "term_gap": self.term_gap,
            "global_version": self.global_version,
        }


@dataclass(frozen=True)
class QueryConclusion:
    query: Query
    match: MatchResult
    score: ScoreResult
    match_reused: bool
    score_reused: bool
    segment_id: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query.to_dict(),
            "match": self.match.to_dict(),
            "score": self.score.to_dict(),
            "match_reused": self.match_reused,
            "score_reused": self.score_reused,
            "segment_id": self.segment_id,
            "diagnostics": self.diagnostics,
        }

