"""Fixed-point per-text intent scoring with unique boundary decisions."""

from __future__ import annotations

from dataclasses import dataclass, field

from .fixedpoint import QuantGear
from .library import MatchResult
from .models import RoutingConfig


@dataclass
class IntentPath:
    """One intent's accumulated fixed-point path score for a text."""

    intent: str
    reference_raw: int
    quantized_raw: int
    hit_count: int
    entry_ids: list[str]
    threshold_raw: int

    def to_dict(self, gear: QuantGear) -> dict:
        return {
            "intent": self.intent,
            "hit_count": self.hit_count,
            "entry_ids": list(self.entry_ids),
            "reference_raw": self.reference_raw,
            "reference_value": gear.to_float(self.reference_raw),
            "quantized_raw": self.quantized_raw,
            "quantized_value": gear.to_float(self.quantized_raw),
            "threshold_raw": self.threshold_raw,
        }


@dataclass
class ScoreResult:
    text_id: str
    decided_intent: str
    paths: dict[str, IntentPath] = field(default_factory=dict)
    boundary: bool = False
    boundary_reasons: list[str] = field(default_factory=list)
    eligible: list[str] = field(default_factory=list)

    def to_dict(self, gear: QuantGear) -> dict:
        return {
            "text_id": self.text_id,
            "decided_intent": self.decided_intent,
            "boundary": self.boundary,
            "boundary_reasons": list(self.boundary_reasons),
            "eligible": list(self.eligible),
            "paths": [
                self.paths[intent].to_dict(gear)
                for intent in sorted(self.paths)
            ],
        }


def score_text(
    text_id: str,
    text_length: int,
    match: MatchResult,
    config: RoutingConfig,
    gear: QuantGear,
) -> ScoreResult:
    """Accumulate scores entirely in fixed-point integers.

    Raw reference score per intent, accumulated in a fixed order (intent id
    ascending, then accepted entry order which is deterministic)::

        raw = SUM(weight * hits * SCALE) - length * LEN_PENALTY

    where LEN_PENALTY = SCALE / 8 + 1 (a fixed odd integer constant).  The odd
    offset guarantees quantization half-steps remain reachable in the integer
    domain.  The raw value is then quantized by the gear using round-half-up;
    the quantized integer is the only value used for the authoritative decision.
    """
    scale = gear.scale
    length_penalty = scale // 8 + 1

    # Group accepted hits by intent in deterministic order.
    hits_by_intent: dict[str, list] = {intent: [] for intent in config.intents}
    for candidate in match.accepted:
        hits_by_intent.setdefault(candidate.intent, []).append(candidate)

    paths: dict[str, IntentPath] = {}
    result = ScoreResult(text_id=text_id, decided_intent=config.fallback_intent)

    for intent in sorted(hits_by_intent):
        candidates = sorted(hits_by_intent[intent], key=lambda c: c.entry_id)
        accum = 0
        entry_ids: list[str] = []
        hit_count = 0
        # Entry order ascending; repeated hits each add weight * SCALE.
        for candidate in candidates:
            accum += candidate.weight * scale
            entry_ids.append(candidate.entry_id)
            hit_count += 1
        accum -= text_length * length_penalty
        quantized = gear.quantize(accum)

        threshold_int = config.intent_thresholds.get(intent, 0)
        threshold_raw = gear.from_int(threshold_int)
        path = IntentPath(
            intent=intent,
            reference_raw=accum,
            quantized_raw=quantized,
            hit_count=hit_count,
            entry_ids=entry_ids,
            threshold_raw=threshold_raw,
        )
        paths[intent] = path

        # Boundary flags based on full-resolution raw value.
        if accum == threshold_raw:
            result.boundary = True
            result.boundary_reasons.append(f"{intent}:exact_threshold")
        if gear.at_half_step(accum):
            result.boundary = True
            result.boundary_reasons.append(f"{intent}:quant_half_step")

    result.paths = paths

    # Eligibility uses quantized score vs threshold (authoritative integer math).
    eligible = [
        intent
        for intent in sorted(paths)
        if paths[intent].quantized_raw >= paths[intent].threshold_raw
        and paths[intent].hit_count > 0
    ]
    result.eligible = eligible

    if eligible:
        # Highest quantized score; tie -> intent id ascending.
        chosen = min(
            eligible,
            key=lambda intent: (
                -paths[intent].quantized_raw,
                intent,
            ),
        )
        result.decided_intent = chosen
        # Record an exact-score tie for audit when two intents share top score.
        top = paths[chosen].quantized_raw
        tied = sorted(
            i for i in eligible if paths[i].quantized_raw == top
        )
        if len(tied) > 1:
            result.boundary = True
            result.boundary_reasons.append(
                "score_tie:" + "=".join(tied)
            )
    else:
        result.decided_intent = config.fallback_intent

    return result
