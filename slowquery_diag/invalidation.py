"""Invalidation propagation and whole-view replacement on scheme edits."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .models import canonical_json


def source_fingerprint(source) -> str:
    payload = {
        "source_id": source.source_id,
        "source_type": source.source_type,
        "format_version": source.format_version,
        "sample_rows": source.sample_rows,
        "latency_caliber": source.latency_caliber,
        "sample_period_ms": source.sample_period_ms,
        "latency_algebra": source.latency_algebra,
        "frame_file": source.frame_file,
    }
    return hashlib.sha256(canonical_json(payload)).hexdigest()[:16]


def candidate_fingerprint(candidate) -> str:
    payload = {
        "candidate_id": candidate.candidate_id,
        "original_query": candidate.original_query,
        "rewritten_sql": candidate.rewritten_sql,
        "target_sources": list(candidate.target_sources),
        "expected_gain_bps": candidate.expected_gain_bps,
    }
    return hashlib.sha256(canonical_json(payload)).hexdigest()[:16]


@dataclass
class SchemeDiff:
    added_sources: tuple[str, ...]
    replaced_sources: tuple[str, ...]
    removed_sources: tuple[str, ...]
    added_candidates: tuple[str, ...]
    replaced_candidates: tuple[str, ...]
    removed_candidates: tuple[str, ...]

    @property
    def changed_source_ids(self) -> set[str]:
        return set(self.added_sources) | set(self.replaced_sources)

    @property
    def changed_candidate_ids(self) -> set[str]:
        return set(self.added_candidates) | set(self.replaced_candidates)


def diff_scheme(previous_sources, current_sources,
                previous_candidates, current_candidates) -> SchemeDiff:
    old_sources = {s.source_id: source_fingerprint(s) for s in previous_sources}
    new_sources = {s.source_id: source_fingerprint(s) for s in current_sources}
    old_candidates = {c.candidate_id: candidate_fingerprint(c)
                      for c in previous_candidates}
    new_candidates = {c.candidate_id: candidate_fingerprint(c)
                      for c in current_candidates}

    added_sources = sorted(set(new_sources) - set(old_sources))
    removed_sources = sorted(set(old_sources) - set(new_sources))
    replaced_sources = sorted(
        sid for sid in set(old_sources) & set(new_sources)
        if old_sources[sid] != new_sources[sid]
    )
    added_candidates = sorted(set(new_candidates) - set(old_candidates))
    removed_candidates = sorted(set(old_candidates) - set(new_candidates))
    replaced_candidates = sorted(
        cid for cid in set(old_candidates) & set(new_candidates)
        if old_candidates[cid] != new_candidates[cid]
    )
    return SchemeDiff(
        added_sources=tuple(added_sources),
        replaced_sources=tuple(replaced_sources),
        removed_sources=tuple(removed_sources),
        added_candidates=tuple(added_candidates),
        replaced_candidates=tuple(replaced_candidates),
        removed_candidates=tuple(removed_candidates),
    )


def affected_queries(diff: SchemeDiff, source_to_queries: dict[str, set[int]],
                     candidate_to_query: dict[str, int]) -> set[int]:
    queries: set[int] = set()
    for source_id in diff.changed_source_ids:
        queries.update(source_to_queries.get(source_id, set()))
    for candidate_id in diff.changed_candidate_ids | set(diff.removed_candidates):
        if candidate_id in candidate_to_query:
            queries.add(candidate_to_query[candidate_id])
    return queries


@dataclass
class ViewReplacement:
    added_entries: tuple[tuple[int, str], ...]
    replaced_entries: tuple[tuple[int, str], ...]
    removed_entries: tuple[tuple[int, str], ...]
    basis_scheme_version: int


def replace_view(previous_keys: set[tuple[int, str]],
                 current_entries: list[tuple[int, str]],
                 scheme_version: int) -> ViewReplacement:
    current_keys = set(current_entries)
    return ViewReplacement(
        added_entries=tuple(sorted(current_keys - previous_keys)),
        replaced_entries=tuple(sorted(current_keys & previous_keys)),
        removed_entries=tuple(sorted(previous_keys - current_keys)),
        basis_scheme_version=scheme_version,
    )


@dataclass
class StaleEntry:
    """A conclusion/replay verdict excluded from the current view."""

    entry_key: tuple[int, str]
    stale_scheme_version: int
    reason: str
    related_sources: tuple[str, ...]
    related_candidates: tuple[str, ...]


def mark_stale_conclusions(
    diff: SchemeDiff,
    source_to_queries: dict[str, set[int]],
    candidate_to_sources: dict[str, tuple[str, ...]],
    previous_scheme_version: int,
) -> list[StaleEntry]:
    """Propagate edits along query -> metric -> evidence -> replay refs.

    Every metric entry of an affected query is marked stale with the
    explicit triggering source/candidate references.  Stale entries are
    retained for audit but never presented in the current view.
    """
    query_to_sources: dict[int, set[str]] = {}
    query_to_candidates: dict[int, set[str]] = {}
    for source_id, query_ids in source_to_queries.items():
        for query_id in query_ids:
            query_to_sources.setdefault(query_id, set()).add(source_id)
    for candidate_id, sources in candidate_to_sources.items():
        match = re.search(r"\d+", candidate_id)
        if match:
            query_id = int(match.group(0))
            query_to_candidates.setdefault(query_id, set()).add(candidate_id)

    affected: set[int] = set()
    for source_id in diff.changed_source_ids:
        affected.update(source_to_queries.get(source_id, set()))
    for candidate_id in diff.changed_candidate_ids | set(diff.removed_candidates):
        match = re.search(r"\d+", candidate_id)
        if match:
            affected.add(int(match.group(0)))

    stale: list[StaleEntry] = []
    changed_sources = tuple(sorted(diff.changed_source_ids))
    changed_candidates = tuple(
        sorted(diff.changed_candidate_ids | set(diff.removed_candidates))
    )
    for query_id in sorted(affected):
        for metric in ("latency_us", "rows_scanned"):
            stale.append(
                StaleEntry(
                    entry_key=(query_id, metric),
                    stale_scheme_version=previous_scheme_version,
                    reason="source_or_candidate_edit",
                    related_sources=tuple(sorted(
                        set(changed_sources)
                        & query_to_sources.get(query_id, set())
                    )),
                    related_candidates=tuple(sorted(
                        set(changed_candidates)
                        & query_to_candidates.get(query_id, set())
                    )),
                )
            )
    stale.sort(key=lambda item: item.entry_key)
    return stale
