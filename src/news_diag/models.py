from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any


@dataclass(frozen=True)
class WorkbenchConfig:
    max_summary_sentences: int
    word_frequency_weight: Decimal
    entity_coverage_weight: Decimal
    position_decay_coefficient: Decimal
    entity_coverage_min: Decimal
    style_template_id: str
    synonym_table_version: str
    fixed_decimals: int
    quantization_levels: int
    max_workers: int
    synonym_table_path: str
    source_name: str = "config.yaml"

    @property
    def score_scale(self) -> int:
        return 10**self.fixed_decimals

    @property
    def quantization_unit(self) -> int:
        return self.score_scale // self.quantization_levels

    def to_report_dict(self) -> dict[str, Any]:
        return {
            "max_summary_sentences": self.max_summary_sentences,
            "word_frequency_weight": decimal_string(self.word_frequency_weight),
            "entity_coverage_weight": decimal_string(self.entity_coverage_weight),
            "position_decay_coefficient": decimal_string(self.position_decay_coefficient),
            "entity_coverage_min": decimal_string(self.entity_coverage_min),
            "style_template_id": self.style_template_id,
            "synonym_table_version": self.synonym_table_version,
            "fixed_decimals": self.fixed_decimals,
            "quantization_levels": self.quantization_levels,
            "fixed_scale": self.score_scale,
            "quantization_unit": self.quantization_unit,
            "max_workers": self.max_workers,
            "synonym_table_path": self.synonym_table_path,
        }


@dataclass(frozen=True)
class Entity:
    entity_id: str
    name: str
    aliases: tuple[str, ...]
    entity_type: str

    def surface_terms(self) -> tuple[str, ...]:
        terms = {self.name, *self.aliases}
        return tuple(sorted((term for term in terms if term), key=lambda item: (-len(item), item)))


@dataclass(frozen=True)
class Event:
    event_id: str
    subject: str
    predicate: str
    object: str

    def to_dict(self) -> dict[str, str]:
        return {
            "event_id": self.event_id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
        }


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    paragraphs: tuple[str, ...]
    entities: tuple[Entity, ...]
    events: tuple[Event, ...]
    source_name: str
    source_line: int
    source_index: int
    text: str
    paragraph_spans: tuple[tuple[int, int], ...]
    sort_key: str


@dataclass(frozen=True)
class Sentence:
    sentence_index: int
    paragraph_index: int
    start: int
    end: int
    text: str
    sort_key: str
    entity_ids: tuple[str, ...]
    frequency_hits: Decimal


@dataclass(frozen=True)
class SynonymRule:
    order: int
    rule_id: str
    template_id: str
    source: str
    target: str


@dataclass(frozen=True)
class SynonymTable:
    version: str
    templates: dict[str, dict[str, Any]]
    rules: tuple[SynonymRule, ...]
    source_name: str

    def with_rule_template(self, rule_id: str, template_id: str) -> "SynonymTable":
        replaced = False
        new_rules: list[SynonymRule] = []
        for rule in self.rules:
            if rule.rule_id == rule_id:
                new_rules.append(
                    SynonymRule(
                        order=rule.order,
                        rule_id=rule.rule_id,
                        template_id=template_id,
                        source=rule.source,
                        target=rule.target,
                    )
                )
                replaced = True
            else:
                new_rules.append(rule)
        if not replaced:
            raise KeyError(rule_id)
        return SynonymTable(
            version=self.version,
            templates=dict(self.templates),
            rules=tuple(new_rules),
            source_name=self.source_name,
        )


@dataclass
class InjectionSpec:
    injection_id: str
    kind: str
    ordinal: int
    doc_id: str | None = None
    sentence_index: int | None = None
    paragraph_index: int | None = None
    delta: Decimal | None = None
    entity: dict[str, Any] | None = None
    rule_id: str | None = None
    missing_template_id: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    reason: str | None = None
    effect: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "injection_id": self.injection_id,
            "type": self.kind,
            "ordinal": self.ordinal,
            "status": self.status,
        }
        if self.reason:
            result["reason"] = self.reason
        if self.doc_id is not None:
            result["doc_id"] = self.doc_id
        if self.sentence_index is not None:
            result["sentence_index"] = self.sentence_index
        if self.paragraph_index is not None:
            result["paragraph_index"] = self.paragraph_index
        if self.delta is not None:
            result["delta"] = decimal_string(self.delta)
        if self.rule_id is not None:
            result["rule_id"] = self.rule_id
        if self.missing_template_id is not None:
            result["missing_template_id"] = self.missing_template_id
        if self.effect:
            result["effect"] = self.effect
        return result


@dataclass(frozen=True)
class TermScore:
    name: str
    normalized: Decimal
    weight: Decimal
    weighted: Decimal
    fixed_integer: int
    fixed_value: Decimal
    rounding_residual: Decimal
    reference: float


@dataclass(frozen=True)
class SentenceScore:
    fixed_integer: int
    fixed_value: Decimal
    ideal_total: Decimal
    reference: float
    residual: Decimal
    terms: tuple[TermScore, ...]
    frequency_hits: Decimal
    frequency_total: Decimal
    entity_hits: int
    entity_total: int
    entity_delta: Decimal

    def term(self, name: str) -> TermScore:
        for term in self.terms:
            if term.name == name:
                return term
        raise KeyError(name)


@dataclass(frozen=True)
class CoverageEvent:
    sentence_index: int
    covered_entities: int
    total_entities: int
    coverage: Decimal
    threshold: Decimal
    rule: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sentence_index": self.sentence_index,
            "covered_entities": self.covered_entities,
            "total_entities": self.total_entities,
            "coverage": decimal_string(self.coverage),
            "threshold": decimal_string(self.threshold),
            "rule": self.rule,
        }


@dataclass(frozen=True)
class SelectionResult:
    selected_indices: tuple[int, ...]
    ordered_indices: tuple[int, ...]
    covered_entity_ids: tuple[str, ...]
    coverage_events: tuple[CoverageEvent, ...]
    coverage_satisfied: bool


def decimal_string(value: Decimal, *, places: int | None = None) -> str:
    if places is None:
        text = format(value.normalize(), "f")
        if text == "-0":
            return "0"
        return text
    quantum = Decimal(1).scaleb(-places)
    return format(value.quantize(quantum), "f")
