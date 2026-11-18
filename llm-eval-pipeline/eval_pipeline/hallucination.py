"""Entity, numeric and citation consistency checks with weighted risk scoring."""

from __future__ import annotations

import re

_ENTITY_RE = re.compile(r"\bENT_[A-Za-z0-9]+\b")
_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_QUOTE_RE = re.compile(r"\[REF:(.*?)\]")


def _entities(text: str) -> set:
    return set(_ENTITY_RE.findall(text))


def _numbers(text: str) -> list:
    return [float(token) for token in _NUMBER_RE.findall(text)]


def _quotes(text: str) -> list:
    return _QUOTE_RE.findall(text)


def entity_risk(sample: dict, params: dict) -> dict:
    """Risk = 1 - overlap ratio of answer entities with context entities."""
    context_entities = _entities(sample["context"])
    answer_entities = _entities(sample["model_answer"])
    if not answer_entities:
        risk = 0.0
        overlap_ratio = 1.0
    else:
        overlap = len(answer_entities & context_entities)
        overlap_ratio = overlap / len(answer_entities)
        risk = 1.0 - overlap_ratio
    return {"risk": risk, "overlap_ratio": overlap_ratio, "answer_entities": sorted(answer_entities)}


def numeric_risk(sample: dict, params: dict) -> dict:
    """Mean relative deviation of answer numbers versus reference numbers."""
    tolerance = float(params.get("relative_tolerance", 0.05))
    answer_numbers = _numbers(sample["model_answer"])
    reference_numbers = _numbers(sample["reference_answer"])
    if not answer_numbers:
        risk = 0.0
    elif not reference_numbers:
        risk = 1.0
    else:
        deviations = []
        for index, value in enumerate(answer_numbers):
            reference = reference_numbers[index % len(reference_numbers)]
            denom = max(abs(reference), 1e-9)
            deviations.append(min(1.0, abs(value - reference) / denom))
        risk = min(1.0, sum(deviations) / len(deviations))
    return {
        "risk": risk,
        "answer_numbers": answer_numbers,
        "reference_numbers": reference_numbers,
        "tolerance": tolerance,
    }


def citation_risk(sample: dict, params: dict) -> dict:
    """Risk = fraction of quoted snippets that do not occur in the context."""
    quotes = _quotes(sample["model_answer"])
    context = sample["context"]
    if not quotes:
        risk = 0.0
        missing = []
    else:
        missing = [quote for quote in quotes if quote not in context]
        risk = len(missing) / len(quotes)
    return {"risk": risk, "quotes": quotes, "missing_quotes": missing}


RULE_FUNCTIONS = {
    "entity_consistency": entity_risk,
    "numeric_consistency": numeric_risk,
    "citation_consistency": citation_risk,
}


def detect(sample: dict, rules: tuple, adjusted_threshold: float) -> dict:
    """Run every rule, combine weighted risk and apply the fixed priority."""
    details = {}
    weighted_sum = 0.0
    weight_total = 0.0
    triggering = []
    for rule in rules:
        function = RULE_FUNCTIONS[rule.rule_id]
        outcome = function(sample, rule.params)
        details[rule.rule_id] = outcome
        weighted_sum += rule.weight * outcome["risk"]
        weight_total += rule.weight
        if outcome["risk"] > float(rule.params.get("trigger_threshold", adjusted_threshold)):
            triggering.append((rule.priority, rule.rule_id))

    composite = weighted_sum / weight_total if weight_total > 0 else 0.0
    suspected = composite > adjusted_threshold
    triggering.sort(key=lambda item: item[0])
    deciding_rule = triggering[0][1] if triggering else None
    return {
        "sample_id": sample["sample_id"],
        "rule_risks": {rule_id: outcome["risk"] for rule_id, outcome in details.items()},
        "weighted_risk": composite,
        "threshold_applied": adjusted_threshold,
        "suspected_hallucination": suspected,
        "deciding_rule": deciding_rule,
        "details": details,
    }


def fixed_tie_order(results: list) -> list:
    """Deterministic order: risk desc, deciding-rule priority, then sample id."""
    def sort_key(result):
        deciding = result["deciding_rule"]
        return (
            -result["weighted_risk"],
            deciding if deciding is not None else "~",
            result["sample_id"],
        )

    return sorted(results, key=sort_key)


def attach_priority_rank(rules: tuple) -> dict:
    return {rule.rule_id: rule.priority for rule in rules}
