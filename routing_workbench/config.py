"""YAML routing configuration parsing and strict validation."""

from __future__ import annotations

import math
from copy import deepcopy
from fractions import Fraction
from typing import Any, Iterable

from .errors import ErrorCode, Failure
from .models import (
    DomainSpec,
    IntentSpec,
    NegationRule,
    RecalibrationSpec,
    RoutingConfig,
    SlotPatternSpec,
    TriggerSpec,
)
from .util import as_fraction, fingerprint, jsonable


class ConfigValidationError(ValueError):
    def __init__(self, field: str, message: str) -> None:
        self.field = field
        self.message = message
        super().__init__(message)

    def as_failure(self) -> Failure:
        return Failure(
            business_code=ErrorCode.ROUTE_INVALID.value,
            utterance_id="__config__",
            stage="validation",
            field=self.field,
            message=self.message,
        )


def _first(mapping: dict[str, Any], names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def _as_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ConfigValidationError(field, f"{field} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("+-").isdigit():
        return int(value.strip())
    raise ConfigValidationError(field, f"{field} must be an integer")


def _as_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.casefold() in {"true", "false"}:
        return value.casefold() == "true"
    raise ConfigValidationError(field, f"{field} must be a boolean")


def _as_trigger_tokens(value: Any, field: str) -> tuple[str, ...]:
    if isinstance(value, str):
        tokens = (value,)
    elif isinstance(value, (list, tuple)):
        if not value:
            raise ConfigValidationError(field, f"{field} must not be empty")
        tokens = tuple(str(item) for item in value)
    else:
        raise ConfigValidationError(field, f"{field} must be a string or token list")
    if any(not token for token in tokens):
        raise ConfigValidationError(field, f"{field} must not contain empty tokens")
    return tokens


def _parse_trigger_entry(raw: Any, owner_kind: str, owner_name: str, ordinal: int, field: str) -> TriggerSpec:
    entry_id: str | None = None
    weight_raw: Any = 1
    if isinstance(raw, str):
        term = raw
        tokens = _as_trigger_tokens(raw, field)
    elif isinstance(raw, (list, tuple)):
        tokens = _as_trigger_tokens(raw, field)
        term = "".join(tokens)
    elif isinstance(raw, dict):
        term_raw = _first(raw, ("term", "trigger", "text", "phrase", "pattern", "value"))
        tokens_raw = _first(raw, ("tokens", "words", "items", "sequence"))
        if term_raw is None and tokens_raw is None:
            raise ConfigValidationError(field, f"{field} must declare a trigger term")
        if tokens_raw is not None:
            tokens = _as_trigger_tokens(tokens_raw, f"{field}.tokens")
        else:
            tokens = _as_trigger_tokens(term_raw, f"{field}.term")
        term = str(term_raw) if term_raw is not None else "".join(tokens)
        if not term:
            raise ConfigValidationError(field, f"{field}.term must not be empty")
        weight_raw = _first(raw, ("weight", "w", "score"), 1)
        raw_id = _first(raw, ("entry_id", "id", "trigger_id"))
        if raw_id is not None:
            entry_id = str(raw_id)
            if not entry_id:
                raise ConfigValidationError(field, f"{field}.id must not be empty")
    else:
        raise ConfigValidationError(field, f"{field} must be a string, token list, or object")
    try:
        weight = as_fraction(weight_raw, f"{field}.weight")
    except ValueError as exc:
        raise ConfigValidationError(f"{field}.weight", str(exc)) from exc
    if not math.isfinite(float(weight)):
        raise ConfigValidationError(f"{field}.weight", "trigger weight must be finite")
    if weight < 0:
        raise ConfigValidationError(f"{field}.weight", "trigger weight must be non-negative")
    if entry_id is None:
        entry_id = f"{owner_kind}:{owner_name}:{ordinal:04d}"
    return TriggerSpec(entry_id, owner_kind, owner_name, term, tokens, weight, ordinal)


def _parse_trigger_table(raw: Any, owner_kind: str, owner_name: str, field: str) -> tuple[TriggerSpec, ...]:
    if raw is None:
        return ()
    entries: list[Any]
    if isinstance(raw, dict):
        entries = [{"term": key, "weight": value} for key, value in raw.items()]
    elif isinstance(raw, (list, tuple)):
        entries = list(raw)
    else:
        raise ConfigValidationError(field, f"{field} must be a list or mapping")
    specs: list[TriggerSpec] = []
    seen: dict[tuple[str, ...], str] = {}
    seen_ids: set[str] = set()
    for ordinal, entry in enumerate(entries, start=1):
        spec = _parse_trigger_entry(entry, owner_kind, owner_name, ordinal, f"{field}[{ordinal - 1}]")
        marker = spec.tokens
        if marker in seen:
            raise ConfigValidationError(field, f"duplicate trigger {spec.term!r} in {owner_name}")
        if spec.entry_id in seen_ids:
            raise ConfigValidationError(field, f"duplicate trigger entry id {spec.entry_id!r} in {owner_name}")
        seen[marker] = spec.term
        seen_ids.add(spec.entry_id)
        specs.append(spec)
    return tuple(specs)


def _parse_negation_rules(raw: Any) -> tuple[NegationRule, ...]:
    if raw is None:
        return ()
    entries: list[Any]
    if isinstance(raw, list):
        entries = raw
    elif isinstance(raw, dict):
        if any(key in raw for key in ("negators", "words", "terms")):
            entries = [raw]
        else:
            entries = [raw]
    else:
        raise ConfigValidationError("negation_rules", "negation rules must be a list or mapping")
    rules: list[NegationRule] = []
    all_negators: set[str] = set()
    for index, item in enumerate(entries):
        field = f"negation_rules[{index}]"
        if isinstance(item, str):
            negators = (item,)
            scope = 3
            mode = "cancel"
            penalty = Fraction(0)
            direction = "before"
        elif isinstance(item, (list, tuple)):
            negators = tuple(str(value) for value in item)
            scope = 3
            mode = "cancel"
            penalty = Fraction(0)
            direction = "before"
        elif isinstance(item, dict):
            negator_raw = _first(item, ("negators", "words", "terms", "negator", "word", "term"))
            if isinstance(negator_raw, str):
                negators = (negator_raw,)
            elif isinstance(negator_raw, (list, tuple)):
                negators = tuple(str(value) for value in negator_raw)
            else:
                raise ConfigValidationError(field, f"{field} must declare negators")
            scope = _as_int(_first(item, ("scope", "window", "distance"), 3), f"{field}.scope")
            mode = str(_first(item, ("mode", "action", "strategy"), "cancel")).casefold()
            direction = str(_first(item, ("direction", "side"), "before")).casefold()
            try:
                penalty = as_fraction(_first(item, ("penalty", "penalty_weight", "factor"), 0), f"{field}.penalty")
            except ValueError as exc:
                raise ConfigValidationError(f"{field}.penalty", str(exc)) from exc
        else:
            raise ConfigValidationError(field, f"{field} must be a string, list, or object")
        if not negators or any(not value for value in negators):
            raise ConfigValidationError(field, "negation words must be non-empty")
        if scope <= 0:
            raise ConfigValidationError(f"{field}.scope", "negation scope must be positive")
        if mode not in {"cancel", "penalty", "penalize"}:
            raise ConfigValidationError(f"{field}.mode", "negation mode must be cancel or penalty")
        if direction not in {"before", "after", "both"}:
            raise ConfigValidationError(f"{field}.direction", "negation direction must be before, after, or both")
        if penalty < 0 or penalty > 1:
            raise ConfigValidationError(f"{field}.penalty", "negation penalty must be within 0..1")
        for negator in negators:
            if not negator.strip():
                raise ConfigValidationError(field, "negation words must be non-empty")
            if negator in all_negators:
                raise ConfigValidationError(field, f"duplicate negation word {negator!r}")
            all_negators.add(negator)
        rules.append(NegationRule(negators, scope, "penalty" if mode == "penalize" else mode, penalty, direction))
    return tuple(rules)


def _parse_slot_declarations(raw: Any) -> dict[str, dict[str, Any]]:
    if raw is None:
        return {}
    result: dict[str, dict[str, Any]] = {}
    if isinstance(raw, dict):
        for name, decl in raw.items():
            key = str(name)
            if not key:
                raise ConfigValidationError("slots", "slot name must not be empty")
            if decl is None:
                result[key] = {}
            elif isinstance(decl, dict):
                result[key] = deepcopy(decl)
            elif isinstance(decl, str):
                result[key] = {"type": decl}
            else:
                result[key] = {"declaration": decl}
    elif isinstance(raw, (list, tuple)):
        for index, decl in enumerate(raw):
            if isinstance(decl, str):
                if not decl:
                    raise ConfigValidationError(f"slots[{index}]", "slot name must not be empty")
                result[decl] = {}
            elif isinstance(decl, dict):
                name = _first(decl, ("name", "slot", "id"))
                if not name:
                    raise ConfigValidationError(f"slots[{index}]", "slot declaration must have a name")
                result[str(name)] = deepcopy(decl)
            else:
                raise ConfigValidationError(f"slots[{index}]", "invalid slot declaration")
    else:
        raise ConfigValidationError("slots", "slots must be a mapping or list")
    return result


def _parse_slot_pattern(raw: Any, default_intent: str | None, index: int) -> SlotPatternSpec:
    field = f"slot_patterns[{index}]"
    if isinstance(raw, str):
        raise ConfigValidationError(field, "a named slot pattern must be declared at top level")
    if not isinstance(raw, dict):
        raise ConfigValidationError(field, f"{field} must be an object")
    name = _first(raw, ("name", "pattern_id", "id"))
    if not name:
        raise ConfigValidationError(field, f"{field}.name is required")
    intent = _first(raw, ("intent", "intent_name"), default_intent)
    slot = _first(raw, ("slot", "slot_name", "target"))
    if not slot:
        raise ConfigValidationError(field, f"{field}.slot is required")
    pattern_raw = _first(raw, ("tokens", "pattern", "sequence", "words", "literals", "anchor"))
    if pattern_raw is None:
        raise ConfigValidationError(field, f"{field} must declare tokens or pattern")
    tokens = _as_trigger_tokens(pattern_raw, f"{field}.tokens")
    placeholder_index = next((i for i, token in enumerate(tokens) if token in {"{}", "{slot}"} or token == f"{{{slot}}}"), None)
    window = _as_int(_first(raw, ("window", "max_window", "distance"), len(tokens) + 1), f"{field}.window")
    if window <= 0:
        raise ConfigValidationError(f"{field}.window", "slot window must be positive")
    if placeholder_index is not None and len(tokens) > window:
        raise ConfigValidationError(f"{field}.window", "slot pattern span must not exceed its window")
    direction = str(_first(raw, ("direction", "capture", "side"), "after")).casefold()
    if direction not in {"after", "before", "both"}:
        raise ConfigValidationError(f"{field}.direction", "slot direction must be after, before, or both")
    try:
        match_weight = as_fraction(_first(raw, ("weight", "match_weight"), 1), f"{field}.weight")
    except ValueError as exc:
        raise ConfigValidationError(f"{field}.weight", str(exc)) from exc
    if match_weight < 0:
        raise ConfigValidationError(f"{field}.weight", "slot pattern weight must be non-negative")
    return SlotPatternSpec(
        name=str(name),
        intent=str(intent) if intent is not None else "",
        slot=str(slot),
        tokens=tokens,
        placeholder_index=placeholder_index,
        window=window,
        direction=direction,
        match_weight=match_weight,
    )


def _parse_intent(raw: Any, name: str, domain: str, top_patterns: dict[str, dict[str, Any]], top_required: dict[str, Any], field: str) -> IntentSpec:
    if not isinstance(raw, dict):
        raise ConfigValidationError(field, f"{field} must be an object")
    triggers = _parse_trigger_table(_first(raw, ("triggers", "trigger_words", "keywords")), "intent", name, f"{field}.triggers")
    pattern_refs = _first(raw, ("slot_patterns", "patterns", "slots"), [])
    inline_patterns: list[dict[str, Any]] = []
    patterns: list[SlotPatternSpec] = []
    if isinstance(pattern_refs, str):
        pattern_refs = [pattern_refs]
    if isinstance(pattern_refs, dict):
        # A single inline object is also accepted.
        if any(key in pattern_refs for key in ("tokens", "pattern", "slot", "slot_name")):
            pattern_refs = [pattern_refs]
        else:
            pattern_refs = list(pattern_refs.values())
    if not isinstance(pattern_refs, (list, tuple)):
        raise ConfigValidationError(f"{field}.slot_patterns", "slot patterns must be a list, string, or mapping")
    for index, ref in enumerate(pattern_refs):
        if isinstance(ref, str):
            if ref not in top_patterns:
                raise ConfigValidationError(f"{field}.slot_patterns", f"unknown slot pattern {ref!r}")
            base = deepcopy(top_patterns[ref])
            base.setdefault("name", ref)
            # A top-level pattern is a reusable declaration. Its intent is the intent that
            # references it, not a globally exclusive owner.
            base["intent"] = name
            patterns.append(_parse_slot_pattern(base, name, index))
        else:
            inline_patterns.append(ref)
            patterns.append(_parse_slot_pattern(ref, name, index))

    required_raw = _first(raw, ("required_slots", "required", "must_have_slots"), top_required.get(name, []))
    if isinstance(required_raw, str):
        required = (required_raw,)
    elif isinstance(required_raw, (list, tuple)):
        required = tuple(str(value) for value in required_raw)
    elif required_raw is None:
        required = ()
    else:
        raise ConfigValidationError(f"{field}.required_slots", "required slots must be a list")
    if any(not slot for slot in required):
        raise ConfigValidationError(f"{field}.required_slots", "required slot names must not be empty")
    return IntentSpec(name=name, domain=domain, triggers=triggers, slot_patterns=tuple(patterns), required_slots=required)


def parse_config(raw: Any, fallback_vocabulary: Iterable[str] | None = None) -> RoutingConfig:
    if not isinstance(raw, dict):
        raise ConfigValidationError("config", "routing configuration must be a mapping")
    raw = deepcopy(raw)
    vocabulary_raw = _first(raw, ("vocabulary", "vocab", "controlled_vocabulary", "terms"))
    vocabulary: set[str] = set(fallback_vocabulary or ())
    if vocabulary_raw is not None:
        if not isinstance(vocabulary_raw, (list, tuple, set)):
            raise ConfigValidationError("vocabulary", "vocabulary must be a list of strings")
        for index, token in enumerate(vocabulary_raw):
            if not isinstance(token, str) or not token:
                raise ConfigValidationError(f"vocabulary[{index}]", "vocabulary terms must be non-empty strings")
            vocabulary.add(token)

    domains_raw = _first(raw, ("domains", "domain_set", "routes"))
    if not isinstance(domains_raw, (dict, list)):
        raise ConfigValidationError("domains", "domains must be a mapping or list")
    domain_entries: list[tuple[str, Any]] = []
    if isinstance(domains_raw, dict):
        for name, value in domains_raw.items():
            domain_entries.append((str(name), value))
    else:
        for index, value in enumerate(domains_raw):
            if not isinstance(value, dict):
                raise ConfigValidationError(f"domains[{index}]", "domain entry must be an object")
            name = _first(value, ("name", "domain", "id"))
            if not name:
                raise ConfigValidationError(f"domains[{index}]", "domain name is required")
            domain_entries.append((str(name), value))
    if not domain_entries:
        raise ConfigValidationError("domains", "at least one domain is required")

    slot_declarations = _parse_slot_declarations(_first(raw, ("slots", "slot_table", "slot_declarations")))

    top_patterns_raw = _first(raw, ("slot_patterns", "patterns"), {})
    top_patterns: dict[str, dict[str, Any]] = {}
    if isinstance(top_patterns_raw, dict):
        for name, value in top_patterns_raw.items():
            if not isinstance(value, dict):
                raise ConfigValidationError(f"slot_patterns.{name}", "top-level slot pattern must be an object")
            item = deepcopy(value)
            item.setdefault("name", str(name))
            top_patterns[str(name)] = item
    elif isinstance(top_patterns_raw, (list, tuple)):
        for index, value in enumerate(top_patterns_raw):
            if not isinstance(value, dict) or not value.get("name"):
                raise ConfigValidationError(f"slot_patterns[{index}]", "top-level slot pattern must declare a name")
            top_patterns[str(value["name"])] = deepcopy(value)
    else:
        raise ConfigValidationError("slot_patterns", "slot patterns must be a mapping or list")

    required_raw = _first(raw, ("required_slots", "required_slot_table", "required"), {})
    top_required: dict[str, Any] = {}
    if isinstance(required_raw, dict):
        top_required = {str(key): value for key, value in required_raw.items()}
    elif required_raw is not None:
        raise ConfigValidationError("required_slots", "required slots table must be a mapping")

    domains: dict[str, DomainSpec] = {}
    for name, value in domain_entries:
        if not name:
            raise ConfigValidationError("domains", "domain name must not be empty")
        if name in domains:
            raise ConfigValidationError("domains", f"duplicate domain {name!r}")
        if not isinstance(value, dict):
            raise ConfigValidationError(f"domains.{name}", "domain entry must be an object")
        triggers = _parse_trigger_table(_first(value, ("triggers", "trigger_words", "keywords")), "domain", name, f"domains.{name}.triggers")
        intents_raw = _first(value, ("intents", "intent_set", "routes"), {})
        intent_entries: list[tuple[str, Any]] = []
        if isinstance(intents_raw, dict):
            intent_entries = [(str(key), item) for key, item in intents_raw.items()]
        elif isinstance(intents_raw, list):
            for index, item in enumerate(intents_raw):
                if not isinstance(item, dict):
                    raise ConfigValidationError(f"domains.{name}.intents[{index}]", "intent must be an object")
                intent_name = _first(item, ("name", "intent", "id"))
                if not intent_name:
                    raise ConfigValidationError(f"domains.{name}.intents[{index}]", "intent name is required")
                intent_entries.append((str(intent_name), item))
        else:
            raise ConfigValidationError(f"domains.{name}.intents", "intents must be a mapping or list")
        if not intent_entries:
            raise ConfigValidationError(f"domains.{name}.intents", "each domain must declare at least one intent")
        intents: dict[str, IntentSpec] = {}
        for intent_name, intent_raw in intent_entries:
            if not intent_name:
                raise ConfigValidationError(f"domains.{name}.intents", "intent name must not be empty")
            if intent_name in intents:
                raise ConfigValidationError(f"domains.{name}.intents", f"duplicate intent {intent_name!r}")
            intent = _parse_intent(intent_raw, intent_name, name, top_patterns, top_required, f"domains.{name}.intents.{intent_name}")
            intents[intent_name] = intent
        domains[name] = DomainSpec(name=name, triggers=triggers, intents=intents)

    # Patterns declared only at the top level are attached through intent references. If a pattern has an
    # explicit intent and is not referenced, attach it for ergonomic configuration.
    for pattern_name, pattern_raw in top_patterns.items():
        intent_name = pattern_raw.get("intent")
        if not intent_name:
            continue
        domain_name = pattern_raw.get("domain")
        candidates = [domains[domain_name]] if domain_name in domains else list(domains.values())
        for domain in candidates:
            if intent_name in domain.intents:
                intent = domain.intents[intent_name]
                if not any(pattern.name == pattern_name for pattern in intent.slot_patterns):
                    parsed = _parse_slot_pattern({**pattern_raw, "name": pattern_name}, intent_name, len(intent.slot_patterns))
                    domains[domain.name] = DomainSpec(
                        domain.name,
                        domain.triggers,
                        {
                            **domain.intents,
                            intent_name: IntentSpec(
                                intent.name,
                                intent.domain,
                                intent.triggers,
                                intent.slot_patterns + (parsed,),
                                intent.required_slots,
                            ),
                        },
                    )
                break

    negation_rules = _parse_negation_rules(_first(raw, ("negation_rules", "negations", "negation", "negation_words")))

    thresholds_raw = _first(raw, ("thresholds", "threshold"), {})
    if thresholds_raw is None:
        thresholds_raw = {}
    if not isinstance(thresholds_raw, dict):
        raise ConfigValidationError("thresholds", "thresholds must be a mapping")
    domain_threshold_raw = _first(thresholds_raw, ("domain", "domain_threshold", "level1"), _first(raw, ("domain_threshold", "level1_threshold"), "0.5"))
    intent_threshold_raw = _first(thresholds_raw, ("intent", "intent_threshold", "level2"), _first(raw, ("intent_threshold", "level2_threshold"), "0.5"))
    try:
        domain_threshold = as_fraction(domain_threshold_raw, "thresholds.domain")
        intent_threshold = as_fraction(intent_threshold_raw, "thresholds.intent")
    except ValueError as exc:
        raise ConfigValidationError("thresholds", str(exc)) from exc
    for label, threshold in (("domain", domain_threshold), ("intent", intent_threshold)):
        if threshold < 0 or threshold > 1:
            raise ConfigValidationError(f"thresholds.{label}", f"{label} threshold must be within 0..1")

    fixed_decimals = _as_int(_first(raw, ("fixed_decimals", "decimal_places", "scale_decimals", "decimals"), 5), "fixed_decimals")
    if fixed_decimals < 0 or fixed_decimals > 9:
        raise ConfigValidationError("fixed_decimals", "fixed decimal width must be within 0..9")
    quantization_levels = _as_int(_first(raw, ("quantization_levels", "quant_levels", "quantization_scale", "quant_scale"), 10**fixed_decimals), "quantization_levels")
    if quantization_levels <= 0 or quantization_levels > 10**9:
        raise ConfigValidationError("quantization_levels", "quantization levels must be a positive integer not greater than 1e9")
    quantization_scale = _as_int(_first(raw, ("quantization_scale", "score_scale"), quantization_levels), "quantization_scale")
    if quantization_scale <= 0 or quantization_scale > 10**9:
        raise ConfigValidationError("quantization_scale", "quantization scale must be a positive integer not greater than 1e9")

    thread_pool_size = _as_int(_first(raw, ("thread_pool_size", "max_workers", "workers", "thread_count", "threads"), 1), "thread_pool_size")
    max_thread_pool_size = _as_int(_first(raw, ("max_thread_pool_size", "worker_limit", "max_threads"), 32), "max_thread_pool_size")
    if max_thread_pool_size <= 0:
        raise ConfigValidationError("max_thread_pool_size", "maximum thread count must be positive")
    if thread_pool_size <= 0:
        raise ConfigValidationError("thread_pool_size", "thread pool size must be positive")
    if thread_pool_size > max_thread_pool_size:
        raise ConfigValidationError("thread_pool_size", "thread pool size must not exceed the configured maximum")
    batch_point_budget = _as_int(_first(raw, ("batch_point_budget", "route_point_budget", "point_budget", "budget"), 1000000), "batch_point_budget")
    if batch_point_budget <= 0:
        raise ConfigValidationError("batch_point_budget", "batch point budget must be a positive integer")
    snapshot_interval = _as_int(_first(raw, ("snapshot_interval_commands", "snapshot_interval", "checkpoint_interval"), 1), "snapshot_interval_commands")
    if snapshot_interval <= 0:
        raise ConfigValidationError("snapshot_interval_commands", "snapshot interval must be a positive integer")

    recal_raw = _first(raw, ("recalibration", "recalibration_weights", "confidence_recalibration"), {})
    if recal_raw is None:
        recal_raw = {}
    if not isinstance(recal_raw, dict):
        raise ConfigValidationError("recalibration", "recalibration settings must be a mapping")
    try:
        completeness_weight = as_fraction(_first(recal_raw, ("completeness_weight", "completeness_bonus", "up_weight"), "0.15"), "recalibration.completeness_weight")
        suppression_penalty = as_fraction(_first(recal_raw, ("suppression_penalty", "suppression_down", "down_weight"), "0.05"), "recalibration.suppression_penalty")
        minimum = as_fraction(_first(recal_raw, ("min", "minimum", "lower"), "0"), "recalibration.min")
        maximum = as_fraction(_first(recal_raw, ("max", "maximum", "upper"), "1"), "recalibration.max")
    except ValueError as exc:
        raise ConfigValidationError("recalibration", str(exc)) from exc
    if completeness_weight < 0:
        raise ConfigValidationError("recalibration.completeness_weight", "completeness weight must be non-negative")
    if suppression_penalty < 0:
        raise ConfigValidationError("recalibration.suppression_penalty", "suppression penalty must be non-negative")
    if minimum < 0 or maximum > 1 or minimum > maximum:
        raise ConfigValidationError("recalibration", "recalibration bounds must satisfy 0 <= min <= max <= 1")
    suppression_cap_raw = _first(recal_raw, ("suppression_cap", "max_suppressed", "suppressed_limit"))
    suppression_cap: int | None = None
    if suppression_cap_raw is not None:
        suppression_cap = _as_int(suppression_cap_raw, "recalibration.suppression_cap")
        if suppression_cap < 0:
            raise ConfigValidationError("recalibration.suppression_cap", "suppression cap must be non-negative")

    declared_slots = set(slot_declarations)
    for domain in domains.values():
        for intent in domain.intents.values():
            for pattern in intent.slot_patterns:
                if not pattern.intent:
                    raise ConfigValidationError(f"slot_patterns.{pattern.name}", "slot pattern intent is required")
                if pattern.intent != intent.name:
                    raise ConfigValidationError(f"slot_patterns.{pattern.name}.intent", "slot pattern intent does not match its owner")
                if pattern.slot not in declared_slots:
                    raise ConfigValidationError(f"slot_patterns.{pattern.name}.slot", f"slot {pattern.slot!r} is not declared")
            for required_slot in intent.required_slots:
                if required_slot not in declared_slots:
                    raise ConfigValidationError(f"domains.{domain.name}.intents.{intent.name}.required_slots", f"required slot {required_slot!r} is not declared")

    semantic = {
        "vocabulary": sorted(vocabulary),
        "domains": {
            domain.name: {
                "triggers": [trigger.reference() for trigger in domain.triggers],
                "intents": {
                    intent.name: {
                        "triggers": [trigger.reference() for trigger in intent.triggers],
                        "slot_patterns": [
                            {
                                "name": pattern.name,
                                "intent": pattern.intent,
                                "slot": pattern.slot,
                                "tokens": list(pattern.tokens),
                                "placeholder_index": pattern.placeholder_index,
                                "window": pattern.window,
                                "direction": pattern.direction,
                                "weight": f"{pattern.match_weight.numerator}/{pattern.match_weight.denominator}",
                            }
                            for pattern in intent.slot_patterns
                        ],
                        "required_slots": list(intent.required_slots),
                    }
                    for intent in domain.intents.values()
                },
            }
            for domain in domains.values()
        },
        "slot_declarations": slot_declarations,
        "negation_rules": [
            {
                "negators": list(rule.negators),
                "scope": rule.scope,
                "mode": rule.mode,
                "penalty": f"{rule.penalty.numerator}/{rule.penalty.denominator}",
                "direction": rule.direction,
            }
            for rule in negation_rules
        ],
        "domain_threshold": f"{domain_threshold.numerator}/{domain_threshold.denominator}",
        "intent_threshold": f"{intent_threshold.numerator}/{intent_threshold.denominator}",
        "fixed_decimals": fixed_decimals,
        "quantization_levels": quantization_levels,
        "quantization_scale": quantization_scale,
        "thread_pool_size": thread_pool_size,
        "max_thread_pool_size": max_thread_pool_size,
        "batch_point_budget": batch_point_budget,
        "snapshot_interval_commands": snapshot_interval,
        "recalibration": {
            "completeness_weight": f"{completeness_weight.numerator}/{completeness_weight.denominator}",
            "suppression_penalty": f"{suppression_penalty.numerator}/{suppression_penalty.denominator}",
            "minimum": f"{minimum.numerator}/{minimum.denominator}",
            "maximum": f"{maximum.numerator}/{maximum.denominator}",
            "suppression_cap": suppression_cap,
        },
    }
    config_hash = fingerprint(semantic)
    return RoutingConfig(
        raw=raw,
        vocabulary=frozenset(vocabulary),
        domains=domains,
        slot_declarations=slot_declarations,
        negation_rules=negation_rules,
        domain_threshold=domain_threshold,
        intent_threshold=intent_threshold,
        fixed_decimals=fixed_decimals,
        quantization_levels=quantization_levels,
        quantization_scale=quantization_scale,
        thread_pool_size=thread_pool_size,
        max_thread_pool_size=max_thread_pool_size,
        batch_point_budget=batch_point_budget,
        snapshot_interval_commands=snapshot_interval,
        recalibration=RecalibrationSpec(completeness_weight, suppression_penalty, minimum, maximum, suppression_cap),
        config_hash=config_hash,
    )


def parse_vocabulary_file(path: Any) -> tuple[str, ...]:
    from pathlib import Path

    vocabulary_path = Path(path)
    terms: list[str] = []
    for line in vocabulary_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        terms.extend(part for part in text.replace(",", " ").split() if part)
    return tuple(terms)


def config_failure(code: str, field: str, message: str) -> Failure:
    return Failure(code, "__config__", "validation", message, field=field)
