"""Deterministic longest-span trigger matching with suppression ledger."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .fingerprints import match_cache_key
from .models import MatchResult, MatchSpan, Query, RuleView, TriggerEntry


MATCH_RULE_VERSION = "longest-char-span-v1"


@dataclass(frozen=True)
class _Candidate:
    entry: TriggerEntry
    start: int
    end: int
    char_start: int
    char_end: int

    @property
    def char_length(self) -> int:
        return self.char_end - self.char_start

    def sort_key(self) -> tuple[object, ...]:
        return (
            -self.char_length,
            self.entry.terms,
            self.start,
            self.entry.trigger_id,
            self.end,
            self.entry.entry_order,
        )


def _char_offsets(terms: tuple[str, ...]) -> tuple[int, ...]:
    offsets = [0]
    for term in terms:
        offsets.append(offsets[-1] + len(term) + 1)
    return tuple(offsets)


def _occurrences(query: Query, entry: TriggerEntry, offsets: tuple[int, ...]) -> list[_Candidate]:
    needle = entry.terms
    haystack = query.terms
    if not needle or len(needle) > len(haystack):
        return []
    width = len(needle)
    candidates: list[_Candidate] = []
    for start in range(0, len(haystack) - width + 1):
        if haystack[start : start + width] != needle:
            continue
        end = start + width
        char_start = offsets[start]
        char_end = offsets[end - 1] + len(needle[-1])
        candidates.append(
            _Candidate(
                entry=entry,
                start=start,
                end=end,
                char_start=char_start,
                char_end=char_end,
            )
        )
    return candidates


def match_query(
    query: Query,
    view: RuleView,
    *,
    match_rule_version: str = MATCH_RULE_VERSION,
) -> MatchResult:
    offsets = _char_offsets(query.terms)
    candidates: list[_Candidate] = []
    for entry in view.entries:
        candidates.extend(_occurrences(query, entry, offsets))
    candidates.sort(key=lambda candidate: candidate.sort_key())

    selected: list[_Candidate] = []
    suppressed: list[dict[str, object]] = []
    for candidate in candidates:
        swallowed_by: _Candidate | None = None
        for accepted in selected:
            if candidate.start < accepted.end and candidate.end > accepted.start:
                swallowed_by = accepted
                break
        if swallowed_by is not None:
            suppressed.append(
                {
                    "trigger_id": candidate.entry.trigger_id,
                    "terms": list(candidate.entry.terms),
                    "start": candidate.start,
                    "end": candidate.end,
                    "char_start": candidate.char_start,
                    "char_end": candidate.char_end,
                    "swallowed_by": swallowed_by.entry.trigger_id,
                    "swallowed_by_start": swallowed_by.start,
                    "swallowed_by_end": swallowed_by.end,
                }
            )
            continue
        selected.append(candidate)

    selected.sort(key=lambda candidate: (candidate.start, candidate.end, candidate.entry.trigger_id))
    considered_ids = tuple(sorted({candidate.entry.trigger_id for candidate in candidates}))
    considered_entries = [entry for entry in view.entries if entry.trigger_id in set(considered_ids)]
    cache_key = match_cache_key(
        query,
        considered_entries,
        match_rule_version=match_rule_version,
    )
    char_length = max((len(query.text), 0))
    return MatchResult(
        query_id=query.query_id,
        selected=tuple(
            MatchSpan(
                trigger_id=candidate.entry.trigger_id,
                terms=candidate.entry.terms,
                intent_id=candidate.entry.intent_id,
                start=candidate.start,
                end=candidate.end,
                char_start=candidate.char_start,
                char_end=candidate.char_end,
                entry_order=candidate.entry.entry_order,
            )
            for candidate in selected
        ),
        suppressed=tuple(suppressed),
        considered_trigger_ids=considered_ids,
        char_length=char_length,
        term_count=len(query.terms),
        cache_key=cache_key,
    )


def refresh_match_metadata(match: MatchResult, query: Query, view: RuleView) -> MatchResult:
    """Keep discrete spans while refreshing intent/order metadata from a reused match."""

    entries: Mapping[str, TriggerEntry] = {entry.trigger_id: entry for entry in view.entries}
    selected: list[MatchSpan] = []
    for span in match.selected:
        entry = entries.get(span.trigger_id)
        if entry is None or entry.terms != span.terms:
            return match_query(query, view)
        selected.append(
            MatchSpan(
                trigger_id=span.trigger_id,
                terms=span.terms,
                intent_id=entry.intent_id,
                start=span.start,
                end=span.end,
                char_start=span.char_start,
                char_end=span.char_end,
                entry_order=entry.entry_order,
            )
        )
    return MatchResult(
        query_id=match.query_id,
        selected=tuple(selected),
        suppressed=match.suppressed,
        considered_trigger_ids=match.considered_trigger_ids,
        char_length=match.char_length,
        term_count=match.term_count,
        cache_key=match.cache_key,
    )


def match_point_cost(query: Query, view: RuleView) -> int:
    """The scheduler reserves all term x trigger anchor checks before dispatch."""

    return len(query.terms) * max(1, len(view.entries))


def match_result_from_dict(data: Mapping[str, object]) -> MatchResult:
    selected = tuple(
        MatchSpan(
            trigger_id=str(item["trigger_id"]),
            terms=tuple(str(term) for term in item["terms"]),
            intent_id=str(item["intent_id"]),
            start=int(item["start"]),
            end=int(item["end"]),
            char_start=int(item["char_start"]),
            char_end=int(item["char_end"]),
            entry_order=int(item["entry_order"]),
        )
        for item in data.get("selected", [])
    )
    return MatchResult(
        query_id=str(data["query_id"]),
        selected=selected,
        suppressed=tuple(dict(item) for item in data.get("suppressed", [])),
        considered_trigger_ids=tuple(str(item) for item in data.get("considered_trigger_ids", [])),
        char_length=int(data["char_length"]),
        term_count=int(data["term_count"]),
        cache_key=str(data.get("cache_key", "")),
    )
