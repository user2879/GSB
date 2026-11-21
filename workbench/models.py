"""In-memory data models for vocabulary, hierarchy config and samples."""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from .errors import (
    E_CONFIG_INVALID,
    E_DOMAIN_UNKNOWN,
    E_INTENT_OUT_OF_DOMAIN,
    E_SLOT_SCHEMA_INVALID,
    E_THRESHOLD_NOT_FINITE,
)
from .fixedpoint import DEFAULT_SCALE_BITS, decimal_to_fixed, is_finite_number
from .io_utils import fingerprint


@dataclass(frozen=True)
class Sample:
    utterance_id: str
    tokens: tuple
    domain: str
    intent: str
    gold_slots: tuple
    source_line: int = -1

    def content_fingerprint(self) -> str:
        return fingerprint({"id": self.utterance_id, "tokens": list(self.tokens)})


@dataclass
class Vocabulary:
    tokens: frozenset
    domain_bags: dict  # domain -> {token: integer weight}

    @classmethod
    def from_dict(cls, raw: dict) -> "Vocabulary":
        tokens = frozenset(raw["tokens"])
        bags = {}
        for domain, bag in raw.get("domain_word_bags", {}).items():
            bags[domain] = {tok: int(weight) for tok, weight in bag.items()}
        return cls(tokens=tokens, domain_bags=bags)


@dataclass
class HierarchyConfig:
    domains: list
    domain_bags: dict            # domain -> {token: weight}
    domain_intents: dict         # domain -> [intent]
    templates: dict              # intent -> {triggers, negations, negation_window}
    slot_defs: dict              # slot -> {values: [[token,...], ...]}
    intent_slots: dict           # intent -> [slot]
    domain_threshold: str
    intent_threshold: str
    slot_window: int
    fixed_point_bits: int
    quant_levels: dict           # name -> {"grid": decimal str, "bits": int}
    active_quant_level: str
    recal: dict                  # fixed-point knobs as decimal strings
    max_workers: int
    slot_budget_per_batch: int
    snapshot_commit_interval: int
    raw: dict = field(repr=False, default_factory=dict)

    # ------------------------------------------------------------------ load
    @classmethod
    def from_dict(cls, raw: dict, vocab: Vocabulary) -> "HierarchyConfig":
        errors = list(cls._validate(raw, vocab))
        if errors:
            raise ConfigError(errors)
        thresholds = raw["thresholds"]
        return cls(
            domains=list(raw["domains"]),
            domain_bags={
                d: {t: int(w) for t, w in bag.items()}
                for d, bag in raw["domain_word_bags"].items()
            },
            domain_intents={d: list(names) for d, names in raw["domain_intents"].items()},
            templates={
                name: {
                    "triggers": list(spec["triggers"]),
                    "negations": list(spec.get("negations", [])),
                    "negation_window": int(spec.get("negation_window", 1)),
                }
                for name, spec in raw["intent_templates"].items()
            },
            slot_defs={
                name: {"values": [list(v) for v in spec["values"]]}
                for name, spec in raw["slot_definitions"].items()
            },
            intent_slots={k: list(v) for k, v in raw["intent_required_slots"].items()},
            domain_threshold=str(thresholds["domain"]),
            intent_threshold=str(thresholds["intent"]),
            slot_window=int(raw["slot_match_window"]),
            fixed_point_bits=int(raw.get("fixed_point_bits", DEFAULT_SCALE_BITS)),
            quant_levels={
                name: {"grid": str(spec["grid"]), "bits": int(spec.get("bits", 2))}
                for name, spec in raw["quantization_levels"].items()
            },
            active_quant_level=raw["active_quantization_level"],
            recal={k: str(v) for k, v in raw["recalibration"].items()},
            max_workers=int(raw["max_workers"]),
            slot_budget_per_batch=int(raw["slot_budget_per_batch"]),
            snapshot_commit_interval=int(raw["snapshot_commit_interval"]),
            raw=raw,
        )

    # -------------------------------------------------------------- validation
    @staticmethod
    def _validate(raw: dict, vocab: Vocabulary):
        required = [
            "domains", "domain_word_bags", "domain_intents", "intent_templates",
            "slot_definitions", "intent_required_slots", "thresholds",
            "slot_match_window", "quantization_levels", "active_quantization_level",
            "recalibration", "max_workers", "slot_budget_per_batch",
            "snapshot_commit_interval",
        ]
        for key in required:
            if key not in raw:
                yield {"code": E_CONFIG_INVALID, "message": f"missing config key: {key}"}
        domains = set(raw.get("domains", []))
        for domain in raw.get("domain_intents", {}):
            if domain not in domains:
                yield {"code": E_DOMAIN_UNKNOWN, "message": f"domain mapping unknown: {domain}"}
        declared_intents = set()
        for domain, names in raw.get("domain_intents", {}).items():
            for intent in names:
                declared_intents.add(intent)
        for intent, spec in raw.get("intent_templates", {}).items():
            if intent not in declared_intents:
                yield {
                    "code": E_INTENT_OUT_OF_DOMAIN,
                    "message": f"template intent not mapped to any domain: {intent}",
                }
            for token in spec.get("triggers", []) + spec.get("negations", []):
                if token not in vocab.tokens:
                    yield {
                        "code": E_CONFIG_INVALID,
                        "message": f"template token outside vocabulary: {token}",
                    }
        for domain, bag in raw.get("domain_word_bags", {}).items():
            if domain not in domains:
                yield {"code": E_DOMAIN_UNKNOWN, "message": f"bag domain unknown: {domain}"}
            for token in bag:
                if token not in vocab.tokens:
                    yield {
                        "code": E_CONFIG_INVALID,
                        "message": f"bag token outside vocabulary: {token}",
                    }
        slot_names = set(raw.get("slot_definitions", {}).keys())
        for intent, slots in raw.get("intent_required_slots", {}).items():
            if intent not in declared_intents:
                yield {
                    "code": E_INTENT_OUT_OF_DOMAIN,
                    "message": f"slot pattern on unmapped intent: {intent}",
                }
            for slot in slots:
                if slot not in slot_names:
                    yield {
                        "code": E_SLOT_SCHEMA_INVALID,
                        "message": f"intent {intent} references undeclared slot {slot}",
                    }
        for slot, spec in raw.get("slot_definitions", {}).items():
            for value in spec.get("values", []):
                for token in value:
                    if token not in vocab.tokens:
                        yield {
                            "code": E_SLOT_SCHEMA_INVALID,
                            "message": f"slot {slot} value token outside vocabulary: {token}",
                        }
        thresholds = raw.get("thresholds", {})
        for level in ("domain", "intent"):
            value = thresholds.get(level)
            if not is_finite_number(value):
                yield {
                    "code": E_THRESHOLD_NOT_FINITE,
                    "message": f"{level} threshold is not finite: {value!r}",
                }
            elif not (Decimal("0") <= Decimal(str(value)) <= Decimal("1")):
                yield {
                    "code": E_CONFIG_INVALID,
                    "message": f"{level} threshold outside [0,1]: {value}",
                }
        for knob in ("completeness_weight", "suppression_weight", "floor"):
            if knob in raw.get("recalibration", {}):
                value = raw["recalibration"][knob]
                if not is_finite_number(value):
                    yield {
                        "code": E_THRESHOLD_NOT_FINITE,
                        "message": f"recalibration {knob} not finite: {value!r}",
                    }
        active = raw.get("active_quantization_level")
        if active and active not in raw.get("quantization_levels", {}):
            yield {
                "code": E_CONFIG_INVALID,
                "message": f"active quantization level not declared: {active}",
            }
        for int_key in ("slot_match_window", "max_workers",
                        "slot_budget_per_batch", "snapshot_commit_interval"):
            if int_key in raw and int(raw[int_key]) < 0:
                yield {
                    "code": E_CONFIG_INVALID,
                    "message": f"{int_key} must be non-negative",
                }

    # ------------------------------------------------------------- derivatives
    def scale_bits(self) -> int:
        return self.fixed_point_bits

    def domain_threshold_fixed(self) -> int:
        return decimal_to_fixed(self.domain_threshold, self.scale_bits())

    def intent_threshold_fixed(self) -> int:
        return decimal_to_fixed(self.intent_threshold, self.scale_bits())

    def recal_fixed(self) -> dict:
        return {k: decimal_to_fixed(v, self.scale_bits()) for k, v in self.recal.items()}

    def quant_grid(self, level: str | None = None) -> str:
        level = level or self.active_quant_level
        return self.quant_levels[level]["grid"]

    def intent_to_domain(self) -> dict:
        mapping = {}
        for domain, names in self.domain_intents.items():
            for intent in names:
                mapping[intent] = domain
        return mapping

    # ------------------------------------------------------------- fingerprints
    # All fingerprints read the mutable ``raw`` dict live so an in-place edit
    # (applied by EditApplier on the shared raw config) changes the hash even
    # when a long-lived HierarchyConfig instance is reused.
    def full_fingerprint(self) -> str:
        return fingerprint(self.raw)

    def routing_fingerprint(self) -> str:
        return fingerprint({
            "domain_word_bags": self.raw["domain_word_bags"],
            "domain_threshold": self.raw["thresholds"]["domain"],
        })

    def intent_mapping_fingerprint(self) -> str:
        return fingerprint({
            "domain_intents": self.raw["domain_intents"],
            "intent_threshold": self.raw["thresholds"]["intent"],
        })

    def template_fingerprint(self, intent: str) -> str:
        return fingerprint(self.raw["intent_templates"][intent])

    def all_templates_fingerprint(self) -> str:
        return fingerprint(self.raw["intent_templates"])

    def slots_fingerprint(self) -> str:
        return fingerprint({
            "slot_definitions": self.raw["slot_definitions"],
            "intent_required_slots": self.raw["intent_required_slots"],
            "slot_match_window": self.raw["slot_match_window"],
        })

    def recal_fingerprint(self) -> str:
        return fingerprint({
            "recalibration": self.raw["recalibration"],
            "active_quantization_level": self.raw["active_quantization_level"],
            "quantization_levels": self.raw["quantization_levels"],
        })


class ConfigError(Exception):
    def __init__(self, errors: list):
        super().__init__(f"config validation failed with {len(errors)} issue(s)")
        self.errors = errors
