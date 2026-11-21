"""Windowed longest-match slot filling with suppression bookkeeping."""
from __future__ import annotations

from .errors import E_SLOT_BUDGET_PENDING, E_SLOT_MISSING


def _find_candidates(tokens: tuple, slot: str, config) -> list:
    window = config.slot_window
    values = config.slot_defs[slot]["values"]
    candidates = []
    length = len(tokens)
    for value in values:
        value_len = len(value)
        max_span = max(value_len, value_len + max(0, window - 1))
        for start in range(length):
            limit = min(length, start + max_span + 1)
            span_index = start
            matched = 0
            gap = 0
            while span_index < limit and matched < value_len:
                if tokens[span_index] == value[matched]:
                    matched += 1
                    gap = 0
                else:
                    gap += 1
                    if gap >= window:
                        break
                span_index += 1
            if matched == value_len:
                end = span_index
                candidates.append({
                    "slot": slot,
                    "value": list(value),
                    "start": start,
                    "end": end,
                    "match_length": end - start,
                    "token_length": value_len,
                })
    return candidates


def fill_slots(tokens: tuple, intent: str, config) -> dict:
    required = list(config.intent_slots.get(intent, []))
    filled = []
    missing = []
    suppressed = []
    points = 0
    for slot in required:
        candidates = _find_candidates(tokens, slot, config)
        if not candidates:
            missing.append(slot)
            continue
        ordered = sorted(
            candidates,
            key=lambda item: (-item["match_length"], item["start"], item["value"]),
        )
        winner = ordered[0]
        filled.append({
            "slot": slot,
            "value": list(winner["value"]),
            "start": winner["start"],
            "end": winner["end"],
            "match_length": winner["match_length"],
        })
        points += winner["token_length"]
        for loser in ordered[1:]:
            suppressed.append({
                "slot": slot,
                "value": list(loser["value"]),
                "start": loser["start"],
                "end": loser["end"],
                "match_length": loser["match_length"],
                "reason": "lost_to_longer_or_earlier_or_lexicographic_candidate",
            })
    suppressed.sort(key=lambda item: (
        item["slot"], -item["match_length"], item["start"], item["value"]))
    errors = []
    if missing:
        errors.append({
            "code": E_SLOT_MISSING,
            "message": f"required slot(s) with no candidate: {missing}",
            "missing_slots": missing,
        })
    return {
        "stage": "slot_filling",
        "intent": intent,
        "required_slots": required,
        "filled_slots": filled,
        "slot_pairs": [[item["slot"], item["value"]] for item in filled],
        "suppressed": suppressed,
        "missing_slots": missing,
        "points_consumed": points,
        "errors": errors,
    }


def budget_guard(slot_result: dict, remaining_budget: int) -> dict | None:
    cost = slot_result["points_consumed"]
    if cost > remaining_budget:
        return {
            "code": E_SLOT_BUDGET_PENDING,
            "message": (f"slot points {cost} exceed remaining batch budget "
                        f"{remaining_budget}; retained in todo queue"),
            "required_points": cost,
            "remaining_budget": remaining_budget,
        }
    return None
