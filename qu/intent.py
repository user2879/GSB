"""Deterministic vocabulary-based intent scoring.

Score of an intent for query ``q``::

    score(intent) =
        sum over every occurrence of every feature term t of the intent
            weight_t * len(t) * decay^position
        + rule_bonus_weight * sum(rule.weight for rules boosting the intent)

Normalization subtracts the maximum score, so the winning intent is exactly
0.0 and all other scores are <= 0.0. Ties are broken by intent name
lexicographic order. Two hit scopes are offered; ``first_char_pruned`` only
iterates terms whose first character occurs in the query, yet performs the
same comparisons, so the result is bit-identical to ``full_scan``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from .config import PipelineConfig
from .loader import RewriteRule, VocabEntry
from .normalization import normalize_query


def _norm(term: str) -> str:
    try:
        return normalize_query(term).normalized
    except Exception:
        return term.casefold()


@dataclass
class IntentResult:
    primary_intent: str = ""
    raw_scores: Dict[str, float] = field(default_factory=dict)
    normalized_scores: Dict[str, float] = field(default_factory=dict)
    contributions: List[dict] = field(default_factory=list)
    rule_bonuses: List[dict] = field(default_factory=list)
    hit_checks: int = 0
    mode: str = "full_scan"


class IntentScorer:
    def __init__(self, vocab: Sequence[VocabEntry], rules: Sequence[RewriteRule],
                 config: PipelineConfig):
        self.config = config
        self.terms: List[Tuple[str, str, float]] = []
        self.by_first: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)
        intents = set()
        for entry in vocab:
            normed = _norm(entry.term)
            if not normed:
                continue
            triple = (entry.intent, normed, float(entry.weight))
            self.terms.append(triple)
            self.by_first[normed[0]].append(triple)
            intents.add(entry.intent)
        self.intents = sorted(intents)
        self._rule_bonus: Dict[str, float] = defaultdict(float)
        self._rule_detail: Dict[str, List[dict]] = defaultdict(list)
        for rule in rules:
            if rule.rule_type != "category_expand":
                continue
            # Expansion rules reference an intent-bearing category term;
            # credit the bonus to every intent that owns the trigger term.
            trigger = _norm(rule.trigger_term)
            owners = sorted({intent for intent, term, _ in self.terms if term == trigger})
            for intent in owners:
                bonus = self.config.rule_bonus_weight * rule.weight
                self._rule_bonus[intent] += bonus
                self._rule_detail[intent].append({
                    "rule_id": rule.rule_id,
                    "bonus": round(bonus, 10),
                })

    def score(self, normalized_query: str, mode: str) -> IntentResult:
        raw_scores = {intent: 0.0 for intent in self.intents}
        contributions: List[dict] = []
        checks = 0

        if mode == "full_scan":
            iterable = self.terms
        elif mode == "first_char_pruned":
            iterable: List[Tuple[str, str, float]] = []
            for char in set(normalized_query):
                iterable.extend(self.by_first.get(char, ()))
            # Preserve vocabulary order even when the index is consulted.
            iterable = sorted(set(iterable), key=self.terms.index)
        else:
            raise ValueError(f"unknown intent hit mode: {mode}")

        for intent, term, weight in iterable:
            checks += 1
            start = normalized_query.find(term)
            while start != -1:
                decay = self.config.position_decay ** start
                contribution = weight * len(term) * decay
                raw_scores[intent] += contribution
                contributions.append({
                    "intent": intent,
                    "term": term,
                    "start": start,
                    "length": len(term),
                    "weight": round(weight, 10),
                    "position_decay": round(decay, 10),
                    "contribution": round(contribution, 10),
                })
                start = normalized_query.find(term, start + 1)

        rule_bonuses: List[dict] = []
        for intent in sorted(self._rule_bonus):
            bonus = self._rule_bonus[intent]
            raw_scores[intent] += bonus
            for detail in self._rule_detail[intent]:
                rule_bonuses.append({"intent": intent, **detail})

        raw_scores = {intent: round(score, 10)
                      for intent, score in raw_scores.items()}
        max_score = max(raw_scores.values(), default=0.0)
        normalized_scores = {
            intent: round(score - max_score, 10)
            for intent, score in raw_scores.items()
        }
        # Deterministic winner: highest score, then lexicographic intent name.
        primary = min(
            self.intents,
            key=lambda intent: (-raw_scores[intent], intent),
        ) if self.intents else ""

        contributions.sort(key=lambda item: (item["start"], item["intent"],
                                             item["term"]))
        rule_bonuses.sort(key=lambda item: (item["intent"], item["rule_id"]))
        return IntentResult(
            primary_intent=primary,
            raw_scores=dict(sorted(raw_scores.items())),
            normalized_scores=dict(sorted(normalized_scores.items())),
            contributions=contributions,
            rule_bonuses=rule_bonuses,
            hit_checks=checks,
            mode=mode,
        )
