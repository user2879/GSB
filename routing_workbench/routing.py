"""Deterministic two-level trigger matching and routing."""

from __future__ import annotations

import bisect
import hashlib
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Iterable

from .models import IntentSpec, NegationRule, RoutingConfig, TriggerSpec, Utterance
from .util import quantize


@dataclass(frozen=True)
class TriggerMatch:
    spec: TriggerSpec
    char_start: int
    char_end: int
    token_start: int
    token_end: int
    text: str

    @property
    def length(self) -> int:
        return self.char_end - self.char_start


@dataclass(frozen=True)
class NegationMatch:
    rule_index: int
    negator: str
    char_start: int
    char_end: int
    token_start: int
    token_end: int
    distance: int


@dataclass
class TierResult:
    stage: str
    selected: str | None
    score: Fraction
    top_score: Fraction
    threshold: Fraction
    candidates: list[dict[str, Any]] = field(default_factory=list)
    matched: list[dict[str, Any]] = field(default_factory=list)
    suppressed: list[dict[str, Any]] = field(default_factory=list)
    boundary: bool = False
    boundary_reasons: list[str] = field(default_factory=list)
    tied_names: list[str] = field(default_factory=list)

    def to_dict(self, scale: int, decimals: int) -> dict[str, Any]:
        from .util import format_fixed

        return {
            "stage": self.stage,
            "selected": self.selected,
            "score": format_fixed(self.score, scale, decimals),
            "score_int": quantize(self.score, scale),
            "top_score": format_fixed(self.top_score, scale, decimals),
            "top_score_int": quantize(self.top_score, scale),
            "threshold": format_fixed(self.threshold, scale, decimals),
            "threshold_int": quantize(self.threshold, scale),
            "score_scale": scale,
            "decimals": decimals,
            "candidates": self.candidates,
            "matched": self.matched,
            "suppressed": self.suppressed,
            "boundary": self.boundary,
            "boundary_reasons": list(self.boundary_reasons),
            "tied_names": list(self.tied_names),
        }


def _token_offsets(tokens: tuple[str, ...]) -> tuple[str, list[int], list[int]]:
    pieces: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    cursor = 0
    for token in tokens:
        starts.append(cursor)
        pieces.append(token)
        cursor += len(token)
        ends.append(cursor)
    return "".join(pieces), starts, ends


def _char_to_token(offset: int, starts: list[int], ends: list[int], is_end: bool = False) -> int:
    if not starts:
        return 0
    if is_end:
        # End offsets are exclusive. A match ending exactly at a token boundary belongs to the
        # preceding token for overlap tests.
        index = bisect.bisect_left(starts, offset) - 1
    else:
        index = bisect.bisect_right(starts, offset) - 1
    return max(0, min(len(starts) - 1, index))


def _find_all(haystack: str, needle: str) -> Iterable[int]:
    if not needle:
        return ()
    positions: list[int] = []
    cursor = 0
    while True:
        found = haystack.find(needle, cursor)
        if found < 0:
            break
        positions.append(found)
        cursor = found + 1
    return positions


def match_trigger(spec: TriggerSpec, tokens: tuple[str, ...]) -> list[TriggerMatch]:
    haystack, starts, ends = _token_offsets(tokens)
    needle = "".join(spec.tokens)
    matches: list[TriggerMatch] = []
    for position in _find_all(haystack, needle):
        char_end = position + len(needle)
        token_start = _char_to_token(position, starts, ends)
        token_end = _char_to_token(char_end, starts, ends, is_end=True)
        matches.append(TriggerMatch(spec, position, char_end, token_start, token_end, needle))
    return matches


def _overlaps(left: TriggerMatch, right: TriggerMatch) -> bool:
    return left.char_start < right.char_end and right.char_start < left.char_end


def _match_sort_key(match: TriggerMatch) -> tuple[Any, ...]:
    return (
        -match.length,
        match.spec.term,
        match.char_start,
        match.char_end,
        match.spec.entry_id,
    )


def resolve_overlapping(matches: Iterable[TriggerMatch]) -> tuple[list[TriggerMatch], list[dict[str, Any]]]:
    ordered = sorted(matches, key=_match_sort_key)
    accepted: list[TriggerMatch] = []
    suppressed: list[dict[str, Any]] = []
    for candidate in ordered:
        winner = next((item for item in accepted if _overlaps(item, candidate)), None)
        if winner is None:
            accepted.append(candidate)
            continue
        suppressed.append(
            {
                "entry_id": candidate.spec.entry_id,
                "term": candidate.spec.term,
                "reason": "overlap_longest_match",
                "start": candidate.char_start,
                "end": candidate.char_end,
                "token_start": candidate.token_start,
                "token_end": candidate.token_end,
                "winner_entry_id": winner.spec.entry_id,
                "winner_term": winner.spec.term,
            }
        )
    accepted.sort(key=lambda item: (item.char_start, item.char_end, item.spec.term, item.spec.entry_id))
    suppressed.sort(key=lambda item: (item["start"], item["end"], item["term"], item["entry_id"]))
    return accepted, suppressed


def _trigger_contribution(match: TriggerMatch, effective_weight: Fraction, status: str = "accepted") -> dict[str, Any]:
    return {
        "entry_id": match.spec.entry_id,
        "term": match.spec.term,
        "weight": f"{match.spec.weight.numerator}/{match.spec.weight.denominator}",
        "effective_weight": f"{effective_weight.numerator}/{effective_weight.denominator}",
        "start": match.char_start,
        "end": match.char_end,
        "token_start": match.token_start,
        "token_end": match.token_end,
        "status": status,
    }


def _select_tier(
    stage: str,
    scores: dict[str, Fraction],
    threshold: Fraction,
    scale: int,
    decimals: int,
    matched: list[dict[str, Any]],
    suppressed: list[dict[str, Any]],
    evidence: dict[str, tuple[Fraction, Fraction]] | None = None,
) -> TierResult:
    from .util import format_fixed

    candidates = [
        {
            "name": name,
            "score": format_fixed(score, scale, decimals),
            "score_int": quantize(score, scale),
            "passes_threshold": score >= threshold,
            "matched_weight": f"{evidence[name][0].numerator}/{evidence[name][0].denominator}" if evidence and name in evidence else None,
            "total_weight": f"{evidence[name][1].numerator}/{evidence[name][1].denominator}" if evidence and name in evidence else None,
        }
        for name, score in sorted(scores.items())
    ]
    if not scores:
        return TierResult(stage, None, Fraction(0), Fraction(0), threshold, candidates, matched, suppressed)
    top_score = max(scores.values())
    top_names = sorted(name for name, score in scores.items() if score == top_score)
    qualifying = [name for name, score in scores.items() if score >= threshold and score == top_score]
    selected: str | None = None
    boundary_reasons: list[str] = []
    tied_names: list[str] = []
    if qualifying:
        selected = sorted(qualifying)[0]
        if top_score == threshold:
            boundary_reasons.append("score_equals_threshold")
        if len(top_names) > 1:
            tied_names = top_names
            boundary_reasons.append("score_tie_lexicographic")
    return TierResult(
        stage=stage,
        selected=selected,
        score=top_score,
        top_score=top_score,
        threshold=threshold,
        candidates=candidates,
        matched=matched,
        suppressed=suppressed,
        boundary=bool(boundary_reasons),
        boundary_reasons=boundary_reasons,
        tied_names=tied_names,
    )


def route_domain(utterance: Utterance, config: RoutingConfig) -> TierResult:
    scores: dict[str, Fraction] = {}
    evidence: dict[str, tuple[Fraction, Fraction]] = {}
    all_matched: list[dict[str, Any]] = []
    all_suppressed: list[dict[str, Any]] = []
    for domain_name in sorted(config.domains):
        domain = config.domains[domain_name]
        matches: list[TriggerMatch] = []
        for trigger in domain.triggers:
            matches.extend(match_trigger(trigger, utterance.tokens))
        accepted, suppressed = resolve_overlapping(matches)
        numerator = Fraction(0)
        counted_entries: set[str] = set()
        for match in accepted:
            contribution = _trigger_contribution(match, match.spec.weight)
            contribution["domain"] = domain_name
            all_matched.append(contribution)
            if match.spec.entry_id not in counted_entries:
                counted_entries.add(match.spec.entry_id)
                numerator += match.spec.weight
        for item in suppressed:
            item["domain"] = domain_name
            all_suppressed.append(item)
        denominator = sum((trigger.weight for trigger in domain.triggers), Fraction(0))
        scores[domain_name] = numerator / denominator if denominator > 0 else Fraction(0)
        evidence[domain_name] = (numerator, denominator)
    return _select_tier(
        "domain",
        scores,
        config.domain_threshold,
        config.quantization_scale,
        config.fixed_decimals,
        sorted(all_matched, key=lambda item: (item["domain"], item["start"], item["term"], item["entry_id"])),
        sorted(all_suppressed, key=lambda item: (item["domain"], item["start"], item["term"], item["entry_id"])),
        evidence,
    )


def _negation_matches(
    utterance: Utterance,
    trigger_match: TriggerMatch,
    config: RoutingConfig,
) -> list[NegationMatch]:
    haystack, starts, ends = _token_offsets(utterance.tokens)
    found: list[NegationMatch] = []
    for rule_index, rule in enumerate(config.negation_rules):
        for negator in rule.negators:
            for position in _find_all(haystack, negator):
                negator_end = position + len(negator)
                negator_token_start = _char_to_token(position, starts, ends)
                negator_token_end = _char_to_token(negator_end, starts, ends, is_end=True)
                before = negator_end <= trigger_match.char_start
                after = position >= trigger_match.char_end
                direction_ok = (
                    (rule.direction == "before" and before)
                    or (rule.direction == "after" and after)
                    or (rule.direction == "both" and (before or after))
                )
                if not direction_ok:
                    continue
                if before:
                    distance = max(0, trigger_match.token_start - negator_token_end)
                else:
                    distance = max(0, negator_token_start - trigger_match.token_end)
                if distance > rule.scope:
                    continue
                found.append(NegationMatch(rule_index, negator, position, negator_end, negator_token_start, negator_token_end, distance))
    found.sort(key=lambda item: (item.distance, item.char_start, item.rule_index, item.negator))
    return found


def _negation_for(
    utterance: Utterance,
    trigger_match: TriggerMatch,
    config: RoutingConfig,
) -> tuple[NegationMatch | None, NegationRule | None]:
    matches = _negation_matches(utterance, trigger_match, config)
    if not matches:
        return None, None
    match = matches[0]
    return match, config.negation_rules[match.rule_index]


def route_intent(utterance: Utterance, domain: str | None, config: RoutingConfig) -> TierResult:
    if domain is None or domain not in config.domains:
        return TierResult("intent", None, Fraction(0), Fraction(0), config.intent_threshold)
    intent_scores: dict[str, Fraction] = {}
    intent_evidence: dict[str, tuple[Fraction, Fraction]] = {}
    all_matched: list[dict[str, Any]] = []
    all_suppressed: list[dict[str, Any]] = []
    for intent_name, intent in sorted(config.domains[domain].intents.items()):
        matches: list[TriggerMatch] = []
        for trigger in intent.triggers:
            matches.extend(match_trigger(trigger, utterance.tokens))
        accepted, suppressed = resolve_overlapping(matches)
        for item in suppressed:
            item["intent"] = intent_name
            all_suppressed.append(item)
        best_effective: dict[str, Fraction] = {}
        for match in accepted:
            negator_match, rule = _negation_for(utterance, match, config)
            if rule is None:
                effective = match.spec.weight
                contribution = _trigger_contribution(match, effective)
            elif rule.mode == "cancel":
                effective = Fraction(0)
                contribution = _trigger_contribution(match, effective, status="negated")
                contribution["negator"] = negator_match.negator if negator_match else ""
                contribution["negation_rule"] = rule.negators[0] if rule.negators else ""
                all_suppressed.append(
                    {
                        "entry_id": match.spec.entry_id,
                        "term": match.spec.term,
                        "reason": "negation_cancel",
                        "start": match.char_start,
                        "end": match.char_end,
                        "token_start": match.token_start,
                        "token_end": match.token_end,
                        "negator": negator_match.negator if negator_match else "",
                        "intent": intent_name,
                    }
                )
            else:
                effective = match.spec.weight * (1 - rule.penalty)
                contribution = _trigger_contribution(match, effective, status="negation_penalty")
                contribution["negator"] = negator_match.negator if negator_match else ""
                contribution["negation_penalty"] = f"{rule.penalty.numerator}/{rule.penalty.denominator}"
            contribution["intent"] = intent_name
            all_matched.append(contribution)
            best_effective[match.spec.entry_id] = max(
                best_effective.get(match.spec.entry_id, Fraction(0)),
                effective,
            )
        numerator = sum(best_effective.values(), Fraction(0))
        denominator = sum((trigger.weight for trigger in intent.triggers), Fraction(0))
        intent_scores[intent_name] = numerator / denominator if denominator > 0 else Fraction(0)
        intent_evidence[intent_name] = (numerator, denominator)
    all_suppressed.sort(key=lambda item: (item["intent"], item["start"], item["term"], item["entry_id"], item["reason"]))
    return _select_tier(
        "intent",
        intent_scores,
        config.intent_threshold,
        config.quantization_scale,
        config.fixed_decimals,
        sorted(all_matched, key=lambda item: (item["intent"], item["start"], item["term"], item["entry_id"])),
        all_suppressed,
        intent_evidence,
    )


def trigger_hit_count(utterance: Utterance, config: RoutingConfig) -> int:
    """Number of distinct non-empty trigger entries with at least one textual hit.

    The count is deliberately independent of thresholds, thread scheduling, and partial cache
    state, which makes the routing-point budget reproducible.
    """
    seen: set[str] = set()
    for domain in config.domains.values():
        for trigger in domain.triggers:
            if trigger.term and trigger.term in "".join(utterance.tokens):
                seen.add(trigger.entry_id)
        for intent in domain.intents.values():
            for trigger in intent.triggers:
                if trigger.term and trigger.term in "".join(utterance.tokens):
                    seen.add(trigger.entry_id)
    return len(seen)


def routing_point_cost(utterance: Utterance, config: RoutingConfig) -> int:
    return len(utterance.tokens) * trigger_hit_count(utterance, config)


def stable_task_key(utterance_id: str, stage: str) -> str:
    return hashlib.sha256(f"{utterance_id}\x1f{stage}".encode("utf-8")).hexdigest()
