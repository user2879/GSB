"""Slot pattern matching, deterministic candidate selection and recalibration."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from .models import IntentSpec, RoutingConfig, SlotPatternSpec, Utterance
from .util import clamp_int, format_fixed, quantize


@dataclass(frozen=True)
class SlotCandidate:
    slot: str
    value: str
    pattern: str
    token_start: int
    token_end: int
    match_length: int

    def sort_key(self) -> tuple[Any, ...]:
        return (-self.match_length, self.token_start, self.value, self.pattern)

    def to_dict(self, status: str, reason: str | None = None) -> dict[str, Any]:
        result = {
            "slot": self.slot,
            "value": self.value,
            "pattern": self.pattern,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "match_length": self.match_length,
            "status": status,
        }
        if reason is not None:
            result["reason"] = reason
        return result


@dataclass
class SlotResult:
    slots: dict[str, str]
    required_slots: tuple[str, ...]
    missing_slots: tuple[str, ...]
    completeness: Fraction
    candidates: list[dict[str, Any]] = field(default_factory=list)
    suppression: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self, config: RoutingConfig) -> dict[str, Any]:
        scale = config.quantization_scale
        decimals = config.fixed_decimals
        return {
            "values": dict(sorted(self.slots.items())),
            "required_slots": list(self.required_slots),
            "missing_slots": list(self.missing_slots),
            "completeness": format_fixed(self.completeness, scale, decimals),
            "completeness_int": quantize(self.completeness, scale),
            "candidates": self.candidates,
            "suppression": self.suppression,
            "suppressed_count": len(self.suppression),
        }


def _pattern_candidates(pattern: SlotPatternSpec, tokens: tuple[str, ...]) -> list[SlotCandidate]:
    results: list[SlotCandidate] = []
    if pattern.placeholder_index is not None:
        index = pattern.placeholder_index
        length = len(pattern.tokens)
        if length == 0 or length > max(pattern.window, length):
            return results
        for start in range(0, len(tokens) - length + 1):
            matched = True
            for offset, expected in enumerate(pattern.tokens):
                if offset == index:
                    continue
                if tokens[start + offset] != expected:
                    matched = False
                    break
            if not matched:
                continue
            value = tokens[start + index]
            if not value:
                continue
            results.append(
                SlotCandidate(
                    slot=pattern.slot,
                    value=value,
                    pattern=pattern.name,
                    token_start=start,
                    token_end=start + length - 1,
                    match_length=length,
                )
            )
        return results

    # Anchor-only patterns capture one adjacent token within the declared window.
    anchor = pattern.tokens
    anchor_length = len(anchor)
    if anchor_length == 0:
        return results
    for start in range(0, len(tokens) - anchor_length + 1):
        if tuple(tokens[start : start + anchor_length]) != anchor:
            continue
        anchor_end = start + anchor_length - 1
        directions: tuple[str, ...]
        if pattern.direction == "after":
            directions = ("after",)
        elif pattern.direction == "before":
            directions = ("before",)
        else:
            directions = ("before", "after")
        for direction in directions:
            if direction == "after":
                candidates = range(anchor_end + 1, min(len(tokens), anchor_end + 1 + pattern.window))
            else:
                candidates = range(max(0, start - pattern.window), start)
            for capture_index in candidates:
                value = tokens[capture_index]
                if not value:
                    continue
                span_start = min(start, capture_index)
                span_end = max(anchor_end, capture_index)
                results.append(
                    SlotCandidate(
                        slot=pattern.slot,
                        value=value,
                        pattern=pattern.name,
                        token_start=span_start,
                        token_end=span_end,
                        match_length=span_end - span_start + 1,
                    )
                )
    return results


def fill_slots(utterance: Utterance, intent: IntentSpec | None, config: RoutingConfig) -> SlotResult:
    if intent is None:
        return SlotResult({}, (), (), Fraction(0))
    by_slot: dict[str, list[SlotCandidate]] = {}
    for pattern in intent.slot_patterns:
        for candidate in _pattern_candidates(pattern, utterance.tokens):
            by_slot.setdefault(candidate.slot, []).append(candidate)
    selected: dict[str, str] = {}
    contributions: list[dict[str, Any]] = []
    suppression: list[dict[str, Any]] = []
    for slot in sorted(by_slot):
        ordered = sorted(by_slot[slot], key=lambda candidate: candidate.sort_key())
        winner = ordered[0]
        selected[slot] = winner.value
        contributions.append(winner.to_dict("selected"))
        for candidate in ordered[1:]:
            reason = "lower_rank"
            if candidate.match_length < winner.match_length:
                reason = "shorter_match"
            elif candidate.token_start > winner.token_start:
                reason = "later_start"
            elif candidate.value != winner.value:
                reason = "lexicographic_value"
            suppression.append(candidate.to_dict("suppressed", reason))
    required = tuple(intent.required_slots)
    missing = tuple(slot for slot in required if slot not in selected)
    completeness = Fraction(len(required) - len(missing), len(required)) if required else Fraction(1)
    contributions.sort(key=lambda item: (item["slot"], item["token_start"], -item["match_length"], item["value"], item["pattern"]))
    suppression.sort(key=lambda item: (item["slot"], item["token_start"], -item["match_length"], item["value"], item["pattern"]))
    return SlotResult(selected, required, missing, completeness, contributions, suppression)


@dataclass(frozen=True)
class RecalibrationResult:
    before_q: int
    after_q: int
    scale: int
    decimals: int
    completeness_q: int
    completeness_bonus_q: int
    suppression_count: int
    suppression_penalty_q: int
    raw_after_q: int
    clipped: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "before_int": self.before_q,
            "after_int": self.after_q,
            "scale": self.scale,
            "decimals": self.decimals,
            "before": format_fixed(self.before_q, self.scale, self.decimals),
            "after": format_fixed(self.after_q, self.scale, self.decimals),
            "completeness_int": self.completeness_q,
            "completeness_bonus_int": self.completeness_bonus_q,
            "suppression_count": self.suppression_count,
            "suppression_penalty_int": self.suppression_penalty_q,
            "raw_after_int": self.raw_after_q,
            "clipped": self.clipped,
        }


def recalibrate_confidence(
    base_score: Fraction,
    slot_result: SlotResult,
    config: RoutingConfig,
) -> RecalibrationResult:
    scale = config.quantization_scale
    decimals = config.fixed_decimals
    recal = config.recalibration
    before_q = clamp_int(quantize(base_score, scale), 0, scale)
    completeness_q = clamp_int(quantize(slot_result.completeness, scale), 0, scale)
    bonus_q = quantize(recal.completeness_weight * slot_result.completeness, scale)
    count = len(slot_result.suppression)
    if recal.suppression_cap is not None:
        count = min(count, recal.suppression_cap)
    penalty_q = quantize(recal.suppression_penalty * count, scale)
    raw_after = before_q + bonus_q - penalty_q
    low = clamp_int(quantize(recal.minimum, scale), 0, scale)
    high = clamp_int(quantize(recal.maximum, scale), 0, scale)
    after_q = clamp_int(raw_after, low, high)
    after_q = clamp_int(after_q, 0, scale)
    return RecalibrationResult(
        before_q=before_q,
        after_q=after_q,
        scale=scale,
        decimals=decimals,
        completeness_q=completeness_q,
        completeness_bonus_q=bonus_q,
        suppression_count=count,
        suppression_penalty_q=penalty_q,
        raw_after_q=raw_after,
        clipped=after_q != raw_after,
    )


def slots_fingerprint(intent: IntentSpec | None, config: RoutingConfig) -> str:
    from .util import fingerprint

    if intent is None:
        return fingerprint({"intent": None, "slots": config.slot_declarations})
    return fingerprint(
        {
            "intent": intent.name,
            "patterns": [
                {
                    "name": pattern.name,
                    "slot": pattern.slot,
                    "tokens": list(pattern.tokens),
                    "placeholder_index": pattern.placeholder_index,
                    "window": pattern.window,
                    "direction": pattern.direction,
                    "weight": f"{pattern.match_weight.numerator}/{pattern.match_weight.denominator}",
                }
                for pattern in intent.slot_patterns
            ],
            "declarations": config.slot_declarations,
            "required": list(intent.required_slots),
        }
    )
