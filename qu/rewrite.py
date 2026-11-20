"""Query rewrite suggestion engine.

Three rule families are supported:

* ``synonym``        - replace a matched term with a synonym;
* ``category_expand``- expand a category term into a broader category;
* ``drop_recover``   - recover/rewrite when a required term is missing.

The engine always computes the unpruned reference list, then optionally
applies the active strategy's rule pruning. A per-query diagnosis states
whether pruning lost any suggestion and diffs the two lists rule by rule.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from .loader import RewriteRule


@dataclass
class RewriteOutcome:
    suggestions: List[dict] = field(default_factory=list)
    full_candidates: List[dict] = field(default_factory=list)
    pruned_candidates: List[dict] = field(default_factory=list)
    pruning_diagnosis: dict = field(default_factory=dict)
    full_rule_checks: int = 0
    pruned_rule_checks: int = 0


class RewriteEngine:
    def __init__(self, rules: Sequence[RewriteRule], max_suggestions: int):
        self.rules = list(rules)
        self.max_suggestions = max_suggestions
        self._by_trigger: Dict[str, List[RewriteRule]] = {}
        self._by_first_char: Dict[str, List[RewriteRule]] = defaultdict(list)
        self._drop_rules: List[RewriteRule] = []
        for rule in self.rules:
            self._by_trigger.setdefault(rule.trigger_term, []).append(rule)
            if rule.rule_type == "drop_recover":
                self._drop_rules.append(rule)
            elif rule.trigger_term:
                self._by_first_char[rule.trigger_term[0]].append(rule)

    def _candidate(self, rule: RewriteRule, query: str, trigger: str,
                   mode: str) -> dict:
        basis = ("missing_term" if rule.rule_type == "drop_recover"
                 else "term_hit")
        if rule.rule_type == "drop_recover":
            benefit = round(rule.weight * 0.5 + len(rule.replacement) * 0.1, 10)
            rewritten = f"{query} {rule.replacement}".strip()
        elif rule.rule_type == "category_expand":
            benefit = round(rule.weight * 1.2 + len(rule.replacement) * 0.1, 10)
            rewritten = query.replace(trigger, rule.replacement, 1)
        else:
            benefit = round(rule.weight + len(rule.replacement) * 0.1, 10)
            rewritten = query.replace(trigger, rule.replacement, 1)
        return {
            "rule_id": rule.rule_id,
            "type": rule.rule_type,
            "trigger_term": trigger,
            "suggestion": rewritten,
            "estimated_benefit": benefit,
            "trigger_basis": basis,
        }

    def _full_scan(self, query: str) -> List[dict]:
        candidates: List[dict] = []
        for rule in self.rules:
            if rule.rule_type == "drop_recover":
                if rule.trigger_term not in query:
                    candidates.append(
                        self._candidate(rule, query, rule.trigger_term,
                                        "missing_term_full_scan"))
            elif rule.trigger_term and rule.trigger_term in query:
                candidates.append(
                    self._candidate(rule, query, rule.trigger_term,
                                    "term_hit_full_scan"))
        return self._sort(candidates), len(self.rules)

    def _indexed_scan(self, query: str) -> Tuple[List[dict], int]:
        candidates: List[dict] = []
        checks = 0
        # First-character inverted index: only rules whose trigger starts on
        # a character present in the query are probed.
        for char in set(query):
            for rule in self._by_first_char.get(char, ()):
                checks += 1
                if rule.trigger_term in query:
                    candidates.append(
                        self._candidate(rule, query, rule.trigger_term,
                                        "term_hit_indexed"))
        # Missing-term rules are always few; check them exhaustively so the
        # pruned path cannot lose drop_recover suggestions.
        for rule in self._drop_rules:
            checks += 1
            if rule.trigger_term not in query:
                candidates.append(
                    self._candidate(rule, query, rule.trigger_term,
                                    "missing_term_indexed"))
        return self._sort(candidates), checks

    @staticmethod
    def _sort(items: List[dict]) -> List[dict]:
        return sorted(items, key=lambda item: (
            -item["estimated_benefit"], item["rule_id"], item["trigger_term"]))

    def generate(self, query: str, prune: bool) -> RewriteOutcome:
        full, full_checks = self._full_scan(query)
        pruned, pruned_checks = self._indexed_scan(query) if prune else (full, full_checks)

        full_keys = [item["rule_id"] for item in full]
        pruned_keys = [item["rule_id"] for item in pruned]
        missing = [rule_id for rule_id in full_keys if rule_id not in pruned_keys]
        # The cap itself may truncate even the unpruned list; keep the two
        # effects distinguishable in the diagnosis.
        capped_full = full[: self.max_suggestions]
        capped_pruned = pruned[: self.max_suggestions]
        cap_dropped = full_keys[self.max_suggestions:]

        comparison = []
        for index, item in enumerate(full):
            comparison.append({
                "rank_full": index,
                "rule_id": item["rule_id"],
                "in_pruned": item["rule_id"] in pruned_keys,
                "survives_cap_full": index < self.max_suggestions,
                "survives_cap_pruned": item["rule_id"] in
                    [rule["rule_id"] for rule in capped_pruned],
            })

        diagnosis = {
            "pruning_enabled": prune,
            "full_candidate_count": len(full),
            "pruned_candidate_count": len(pruned),
            "suggestions_cap": self.max_suggestions,
            "dropped_by_cap_rule_ids": cap_dropped,
            "missing_due_to_pruning_rule_ids": missing,
            "pruning_lost_suggestion": bool(missing),
            "full_rule_checks": full_checks,
            "pruned_rule_checks": pruned_checks,
            "item_by_item_comparison": comparison,
            "conclusion": (
                "pruning omitted at least one suggestion"
                if missing else
                "pruned path reproduces every unpruned suggestion"
            ),
        }
        return RewriteOutcome(
            suggestions=capped_pruned if prune else capped_full,
            full_candidates=full,
            pruned_candidates=pruned,
            pruning_diagnosis=diagnosis,
            full_rule_checks=full_checks,
            pruned_rule_checks=pruned_checks,
        )
