"""Content-fingerprint dependency evaluation and precise cache reuse."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .fingerprints import (
    match_cache_key,
    query_content_fingerprint,
    score_cache_key,
    trigger_entry_fingerprint,
    trigger_match_fingerprint,
)
from .matching import match_result_from_dict, refresh_match_metadata
from .models import MatchResult, Query, RuleView, RuntimeConfig, ScoreResult, TriggerEntry
from .scoring import score_result_from_dict


def _contains_terms(haystack: Sequence[str], needle: Sequence[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    width = len(needle)
    for start in range(0, len(haystack) - width + 1):
        if haystack[start : start + width] == needle:
            return True
    return False


@dataclass
class QueryCacheDecision:
    match: MatchResult | None
    score: ScoreResult | None
    match_basis: str
    score_basis: str
    match_invalidated_by: tuple[str, ...] = ()
    score_invalidated_by: tuple[str, ...] = ()


@dataclass
class CachePlan:
    decisions: dict[str, QueryCacheDecision]
    ledger: list[dict[str, Any]]
    invalidation_counts: dict[str, dict[str, int]]
    changed_trigger_ids: tuple[str, ...]
    match_changed_trigger_ids: tuple[str, ...]


@dataclass
class PreviousArtifacts:
    generation: str
    queries: Mapping[str, dict[str, Any]] = field(default_factory=dict)
    matches: Mapping[str, dict[str, Any]] = field(default_factory=dict)
    scores: Mapping[str, dict[str, Any]] = field(default_factory=dict)
    segments: Mapping[str, dict[str, Any]] = field(default_factory=dict)
    rule_entries: tuple[TriggerEntry, ...] = ()


class CacheManager:
    def __init__(
        self,
        *,
        previous: PreviousArtifacts | None,
        current_view: RuleView,
        queries: Iterable[Query],
        config: RuntimeConfig,
    ) -> None:
        self.previous = previous
        self.current_view = current_view
        self.queries = {query.query_id: query for query in queries}
        self.config = config
        self.old_entries = {entry.trigger_id: entry for entry in (previous.rule_entries if previous else ())}
        self.new_entries = {entry.trigger_id: entry for entry in current_view.entries}
        self.trigger_revision: dict[str, str] = {}
        self.changed_ids: set[str] = set()
        self.match_changed_ids: set[str] = set()
        self.score_changed_ids: set[str] = set()
        self.added_ids: set[str] = set()
        self.removed_ids: set[str] = set()
        for trigger_id in sorted(set(self.old_entries) | set(self.new_entries)):
            old = self.old_entries.get(trigger_id)
            new = self.new_entries.get(trigger_id)
            source = new.source_revision if new is not None else old.source_revision if old is not None else ""
            self.trigger_revision[trigger_id] = source
            if old is None:
                self.added_ids.add(trigger_id)
                self.changed_ids.add(trigger_id)
                self.match_changed_ids.add(trigger_id)
                self.score_changed_ids.add(trigger_id)
                continue
            if new is None:
                self.removed_ids.add(trigger_id)
                self.changed_ids.add(trigger_id)
                self.match_changed_ids.add(trigger_id)
                self.score_changed_ids.add(trigger_id)
                continue
            if trigger_match_fingerprint(old) != trigger_match_fingerprint(new):
                self.match_changed_ids.add(trigger_id)
            if trigger_entry_fingerprint(old) != trigger_entry_fingerprint(new):
                self.score_changed_ids.add(trigger_id)
            if trigger_id in self.match_changed_ids or trigger_id in self.score_changed_ids:
                self.changed_ids.add(trigger_id)

    def _empty_counts(self) -> dict[str, dict[str, int]]:
        return {}

    def _count_invalidation(self, counts: dict[str, dict[str, int]], revision_id: str, stage: str) -> None:
        bucket = counts.setdefault(revision_id or "unknown-revision", {"match": 0, "score": 0, "merge": 0})
        bucket[stage] = bucket.get(stage, 0) + 1

    def _match_dependencies(self, trigger_ids: Iterable[str]) -> set[str]:
        return {trigger_id for trigger_id in trigger_ids if trigger_id in self.changed_ids}

    def plan(self) -> CachePlan:
        decisions: dict[str, QueryCacheDecision] = {}
        ledger: list[dict[str, Any]] = []
        invalidation_counts: dict[str, dict[str, int]] = self._empty_counts()
        for query_id in sorted(self.queries):
            query = self.queries[query_id]
            previous = self.previous
            old_query = previous.queries.get(query_id) if previous else None
            old_match_raw = previous.matches.get(query_id) if previous else None
            old_score_raw = previous.scores.get(query_id) if previous else None
            if previous is None or old_query is None:
                decisions[query_id] = QueryCacheDecision(None, None, "no_previous_snapshot", "no_previous_snapshot")
                continue
            query_same = old_query.get("fingerprint") == query_content_fingerprint(query)
            if not query_same:
                self._count_invalidation(invalidation_counts, "query-content", "match")
                self._count_invalidation(invalidation_counts, "query-content", "score")
                decisions[query_id] = QueryCacheDecision(
                    None,
                    None,
                    "query_content_changed",
                    "query_content_changed",
                    ("query-content",),
                    ("query-content",),
                )
                continue

            old_match: MatchResult | None = None
            if old_match_raw is not None:
                try:
                    old_match = match_result_from_dict(old_match_raw)
                except (KeyError, TypeError, ValueError):
                    old_match = None

            match_causes: list[str] = []
            if old_match is not None:
                considered = set(old_match.considered_trigger_ids)
                match_causes = sorted(considered & self.match_changed_ids)
                for trigger_id in sorted(self.match_changed_ids):
                    entry = self.new_entries.get(trigger_id)
                    if entry is not None and _contains_terms(query.terms, entry.terms) and trigger_id not in match_causes:
                        match_causes.append(trigger_id)
                    old_entry = self.old_entries.get(trigger_id)
                    if old_entry is not None and _contains_terms(query.terms, old_entry.terms) and trigger_id not in match_causes:
                        match_causes.append(trigger_id)
                if match_causes:
                    old_match = None
                else:
                    missing_considered = [trigger_id for trigger_id in considered if trigger_id not in self.new_entries]
                    if missing_considered:
                        old_match = None
                        match_causes.extend(sorted(missing_considered))

            match = None
            match_basis = "match_cache_miss"
            if old_match is not None:
                computed_entries = [
                    self.new_entries[trigger_id]
                    for trigger_id in sorted(old_match.considered_trigger_ids)
                    if trigger_id in self.new_entries
                ]
                expected_key = match_cache_key(
                    query,
                    computed_entries,
                    match_rule_version="longest-char-span-v1",
                )
                if expected_key == old_match.cache_key:
                    match = refresh_match_metadata(old_match, query, self.current_view)
                    match_basis = "query_fingerprint_same;match_dependencies_unchanged"
                    ledger.append(
                        {
                            "stage": "match",
                            "query_id": query_id,
                            "cache_key": match.cache_key,
                            "basis": match_basis,
                            "source_generation": previous.generation if previous else None,
                            "dependencies": list(match.considered_trigger_ids),
                        }
                    )
                else:
                    match_causes.extend(["match-rule-key"])

            if match is None and match_causes:
                for trigger_id in match_causes:
                    revision_id = self.trigger_revision.get(trigger_id, "match-rule-key")
                    self._count_invalidation(invalidation_counts, revision_id, "match")

            score = None
            score_basis = "score_cache_miss"
            score_causes: list[str] = []
            if old_score_raw is not None and match is not None:
                try:
                    old_score = score_result_from_dict(old_score_raw)
                except (KeyError, TypeError, ValueError):
                    old_score = None
                if old_score is not None:
                    referenced_entries = [
                        self.new_entries[trigger_id]
                        for trigger_id in old_score.referenced_trigger_ids
                        if trigger_id in self.new_entries
                    ]
                    missing_refs = [
                        trigger_id
                        for trigger_id in old_score.referenced_trigger_ids
                        if trigger_id not in self.new_entries
                    ]
                    expected_score_key = score_cache_key(query, match.cache_key, referenced_entries, self.config)
                    if not missing_refs and expected_score_key == old_score.cache_key:
                        score = old_score
                        score_basis = "query_and_match_keys_same;referenced_trigger_entries_same;quantum_and_threshold_same"
                        ledger.append(
                            {
                                "stage": "score",
                                "query_id": query_id,
                                "cache_key": score.cache_key,
                                "basis": score_basis,
                                "source_generation": previous.generation if previous else None,
                                "dependencies": list(score.referenced_trigger_ids),
                            }
                        )
                    else:
                        score_causes.extend(missing_refs)
                        score_causes.extend(
                            sorted(set(old_score.referenced_trigger_ids) & self.score_changed_ids)
                        )
                        if expected_score_key != old_score.cache_key:
                            score_causes.append("score-config-key")
            if match is not None and score is None:
                for cause in sorted(set(score_causes) or {"score-config-key"}):
                    revision_id = self.trigger_revision.get(cause, cause)
                    self._count_invalidation(invalidation_counts, revision_id, "score")

            decisions[query_id] = QueryCacheDecision(
                match=match,
                score=score,
                match_basis=match_basis,
                score_basis=score_basis,
                match_invalidated_by=tuple(sorted(set(match_causes))),
                score_invalidated_by=tuple(sorted(set(score_causes))),
            )
        return CachePlan(
            decisions=decisions,
            ledger=ledger,
            invalidation_counts=invalidation_counts,
            changed_trigger_ids=tuple(sorted(self.changed_ids)),
            match_changed_trigger_ids=tuple(sorted(self.match_changed_ids)),
        )
