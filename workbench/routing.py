"""Two-level deterministic routing: domain bag-of-words then intent templates."""
from __future__ import annotations

import numpy as np

from .errors import E_DOMAIN_BELOW_THRESHOLD, E_INTENT_BELOW_THRESHOLD
from .fixedpoint import fixed_to_str, scale_factor


def route_domain(tokens: tuple, config) -> dict:
    bits = config.scale_bits()
    factor = scale_factor(bits)
    unique = set(tokens)
    scores = {}
    score_values = []
    for domain in config.domains:
        bag = config.domain_bags.get(domain, {})
        bag_weight = sum(bag.values())
        hits = sum(weight for token, weight in bag.items() if token in unique)
        score = (hits * factor) // bag_weight if bag_weight else 0
        scores[domain] = score
        score_values.append(score)
    vector = np.asarray(score_values, dtype=np.int64)
    best_index = int(np.argmax(vector))
    best_score = int(vector[best_index])
    tied = [domain for domain, score in scores.items() if score == best_score]
    winner = min(tied)
    tie = len(tied) > 1
    accepted = best_score >= config.domain_threshold_fixed()
    criterion = "domain_tie_lexicographic" if tie else "domain_unique_max"
    return {
        "stage": "domain_routing",
        "scores": {d: fixed_to_str(s, bits) for d, s in sorted(scores.items())},
        "winner": winner if accepted else None,
        "score": fixed_to_str(best_score, bits),
        "tie": tie,
        "tied_domains": sorted(tied),
        "threshold": config.domain_threshold,
        "criterion": criterion,
        "accepted": accepted,
        "error": None if accepted else {
            "code": E_DOMAIN_BELOW_THRESHOLD,
            "message": f"domain score {fixed_to_str(best_score, bits)} "
                       f"< threshold {config.domain_threshold}",
        },
    }


def _resolve_triggers(tokens: tuple, template: dict) -> list:
    positions = {}
    for index, token in enumerate(tokens):
        positions.setdefault(token, []).append(index)
    window = int(template["negation_window"])
    entries = []
    for trigger in template["triggers"]:
        hit_positions = positions.get(trigger, [])
        if not hit_positions:
            continue
        first = hit_positions[0]
        negator = None
        negator_pos = None
        for neg in template["negations"]:
            for pos in positions.get(neg, []):
                if first - window <= pos < first:
                    negator = neg
                    negator_pos = pos
                    break
            if negator is not None:
                break
        entries.append({
            "trigger": trigger,
            "position": first,
            "negated": negator is not None,
            "negator": negator,
            "negator_position": negator_pos,
        })
    entries.sort(key=lambda item: (item["position"], item["trigger"]))
    return entries


def route_intent(tokens: tuple, domain: str, config) -> dict:
    bits = config.scale_bits()
    factor = scale_factor(bits)
    intents = sorted(config.domain_intents.get(domain, []))
    per_intent = {}
    score_values = []
    for intent in intents:
        template = config.templates[intent]
        entries = _resolve_triggers(tokens, template)
        total = len(template["triggers"])
        effective = sum(0 if entry["negated"] else 1 for entry in entries)
        score = (effective * factor) // total if total else 0
        per_intent[intent] = {"score": score, "entries": entries,
                              "effective": effective, "total": total}
        score_values.append(score)
    if not intents:
        return {
            "stage": "intent_routing", "domain": domain, "scores": {},
            "winner": None, "score": "0", "tie": False, "tied_intents": [],
            "threshold": config.intent_threshold, "criterion": "no_intent_in_domain",
            "accepted": False, "trigger_entries": [],
            "error": {"code": E_INTENT_BELOW_THRESHOLD,
                      "message": f"domain {domain} declares no intents"},
        }
    vector = np.asarray(score_values, dtype=np.int64)
    best_index = int(np.argmax(vector))
    best_score = int(vector[best_index])
    tied = [intent for intent in intents if per_intent[intent]["score"] == best_score]
    winner = min(tied)
    tie = len(tied) > 1
    at_threshold = best_score == config.intent_threshold_fixed()
    accepted = best_score >= config.intent_threshold_fixed()
    if tie:
        criterion = "intent_tie_lexicographic"
    elif at_threshold:
        criterion = "intent_score_equal_threshold_lexicographic"
    else:
        criterion = "intent_unique_max"
    return {
        "stage": "intent_routing",
        "domain": domain,
        "scores": {intent: fixed_to_str(per_intent[intent]["score"], bits)
                   for intent in intents},
        "winner": winner if accepted else None,
        "score": fixed_to_str(best_score, bits),
        "tie": tie,
        "tied_intents": sorted(tied),
        "threshold": config.intent_threshold,
        "criterion": criterion,
        "accepted": accepted,
        "trigger_entries": per_intent[winner]["entries"],
        "error": None if accepted else {
            "code": E_INTENT_BELOW_THRESHOLD,
            "message": f"intent score {fixed_to_str(best_score, bits)} "
                       f"< threshold {config.intent_threshold}",
        },
    }
