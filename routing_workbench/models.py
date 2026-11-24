"""Immutable domain models for utterances, routing configuration and results."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any


@dataclass(frozen=True)
class Utterance:
    utterance_id: str
    session_id: str
    turn: int
    tokens: tuple[str, ...]
    domain_truth: str | None = None
    intent_truth: str | None = None
    slots_truth: dict[str, str] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    source_line: int = 0

    def canonical_content(self) -> dict[str, Any]:
        return {
            "utterance_id": self.utterance_id,
            "session_id": self.session_id,
            "turn": self.turn,
            "tokens": list(self.tokens),
            "domain_truth": self.domain_truth,
            "intent_truth": self.intent_truth,
            "slots_truth": dict(sorted(self.slots_truth.items())),
        }


@dataclass(frozen=True)
class TriggerSpec:
    entry_id: str
    owner_kind: str
    owner_name: str
    term: str
    tokens: tuple[str, ...]
    weight: Fraction
    ordinal: int

    def reference(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "owner_kind": self.owner_kind,
            "owner_name": self.owner_name,
            "term": self.term,
            "tokens": list(self.tokens),
            "weight": f"{self.weight.numerator}/{self.weight.denominator}",
        }


@dataclass(frozen=True)
class NegationRule:
    negators: tuple[str, ...]
    scope: int
    mode: str
    penalty: Fraction
    direction: str


@dataclass(frozen=True)
class SlotPatternSpec:
    name: str
    intent: str
    slot: str
    tokens: tuple[str, ...]
    placeholder_index: int | None
    window: int
    direction: str
    match_weight: Fraction = Fraction(1, 1)


@dataclass(frozen=True)
class IntentSpec:
    name: str
    domain: str
    triggers: tuple[TriggerSpec, ...]
    slot_patterns: tuple[SlotPatternSpec, ...]
    required_slots: tuple[str, ...]


@dataclass(frozen=True)
class DomainSpec:
    name: str
    triggers: tuple[TriggerSpec, ...]
    intents: dict[str, IntentSpec]


@dataclass(frozen=True)
class RecalibrationSpec:
    completeness_weight: Fraction
    suppression_penalty: Fraction
    minimum: Fraction
    maximum: Fraction
    suppression_cap: int | None


@dataclass(frozen=True)
class RoutingConfig:
    raw: dict[str, Any]
    vocabulary: frozenset[str]
    domains: dict[str, DomainSpec]
    slot_declarations: dict[str, dict[str, Any]]
    negation_rules: tuple[NegationRule, ...]
    domain_threshold: Fraction
    intent_threshold: Fraction
    fixed_decimals: int
    quantization_levels: int
    quantization_scale: int
    thread_pool_size: int
    max_thread_pool_size: int
    batch_point_budget: int
    snapshot_interval_commands: int
    recalibration: RecalibrationSpec
    config_hash: str = ""

    def intents_in(self, domain: str) -> dict[str, IntentSpec]:
        domain_spec = self.domains.get(domain)
        return domain_spec.intents if domain_spec else {}

    def selected_intent_spec(self, domain: str | None, intent: str | None) -> IntentSpec | None:
        if domain is None or intent is None:
            return None
        return self.domains.get(domain, DomainSpec(domain, (), {})).intents.get(intent)

    def to_snapshot_dict(self) -> dict[str, Any]:
        return self.raw

