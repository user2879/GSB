"""Entity slot extraction via maximum matching.

Three interchangeable candidate-generation paths produce *identical* slot
results; they differ only in work performed:

* ``exhaustive``   - one ``str.find`` per dictionary term (reference path);
* ``inverted``     - first-character inverted index, only terms whose first
  character appears in the query are probed;
* ``prefix_group`` - terms grouped by their first two characters; only groups
  for prefixes observed in the query are probed.

Overlap resolution is always: longest span first, then dictionary-type
priority, then earliest start position. Swallowed candidates are reported as
suppressed items together with an audit record.
"""

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from .loader import EntityEntry


@dataclass
class Candidate:
    entity_type: str
    term: str
    entity_id: str
    start: int
    end: int
    weight: float
    priority: int


@dataclass
class EntityMatchResult:
    slots: List[dict] = field(default_factory=list)
    suppressed: List[dict] = field(default_factory=list)
    resolutions: List[dict] = field(default_factory=list)
    match_checks: int = 0
    candidate_count: int = 0
    mode: str = "exhaustive"


class EntityMatcher:
    def __init__(self, entries: Sequence[EntityEntry], type_priority: Sequence[str]):
        self.entries = list(entries)
        self.type_priority = {name: idx for idx, name in enumerate(type_priority)}
        self.by_first: Dict[str, List[EntityEntry]] = defaultdict(list)
        self.by_prefix: Dict[str, List[EntityEntry]] = defaultdict(list)
        for entry in self.entries:
            self.by_first[entry.term[0]].append(entry)
            self.by_prefix[entry.term[:2]].append(entry)

    # -- candidate generation paths -----------------------------------------
    def _exhaustive_candidates(self, text: str) -> Tuple[List[Candidate], int]:
        candidates: List[Candidate] = []
        checks = 0
        for entry in self.entries:
            checks += 1
            start = text.find(entry.term)
            while start != -1:
                candidates.append(self._to_candidate(entry, start))
                start = text.find(entry.term, start + 1)
        return candidates, checks

    def _inverted_candidates(self, text: str) -> Tuple[List[Candidate], int]:
        candidates: List[Candidate] = []
        checks = 0
        for char in set(text):
            for entry in self.by_first.get(char, ()):
                checks += 1
                start = text.find(entry.term)
                while start != -1:
                    candidates.append(self._to_candidate(entry, start))
                    start = text.find(entry.term, start + 1)
        return candidates, checks

    def _prefix_candidates(self, text: str) -> Tuple[List[Candidate], int]:
        candidates: List[Candidate] = []
        checks = 0
        observed = {text[i:i + 2] for i in range(len(text) - 1)}
        if len(text) == 1:
            observed.add(text)
        for prefix in observed:
            for entry in self.by_prefix.get(prefix, ()):
                checks += 1
                start = text.find(entry.term)
                while start != -1:
                    candidates.append(self._to_candidate(entry, start))
                    start = text.find(entry.term, start + 1)
        return candidates, checks

    def _to_candidate(self, entry: EntityEntry, start: int) -> Candidate:
        return Candidate(
            entity_type=entry.entity_type,
            term=entry.term,
            entity_id=entry.entity_id,
            start=start,
            end=start + len(entry.term),
            weight=entry.weight,
            priority=self.type_priority[entry.entity_type],
        )

    # -- shared deterministic resolution ------------------------------------
    def match(self, text: str, mode: str) -> EntityMatchResult:
        if mode == "exhaustive":
            candidates, checks = self._exhaustive_candidates(text)
        elif mode == "inverted":
            candidates, checks = self._inverted_candidates(text)
        elif mode == "prefix_group":
            candidates, checks = self._prefix_candidates(text)
        else:
            raise ValueError(f"unknown entity match mode: {mode}")

        # Deduplicate identical spans+terms; determinism comes from loader
        # order (line numbers), which is stable across runs.
        unique: Dict[tuple, Candidate] = {}
        for cand in candidates:
            key = (cand.start, cand.end, cand.entity_type, cand.term)
            unique.setdefault(key, cand)
        ordered = sorted(
            unique.values(),
            key=lambda c: (- (c.end - c.start), c.priority, c.start,
                           c.entity_id, c.term),
        )

        result = EntityMatchResult(mode=mode, match_checks=checks,
                                   candidate_count=len(ordered))
        occupied: List[Tuple[int, int]] = []
        chosen: List[Candidate] = []
        for cand in ordered:
            overlap = next(
                (slot for slot in chosen
                 if cand.start < slot.end and slot.start < cand.end),
                None,
            )
            payload = {
                "entity_type": cand.entity_type,
                "term": cand.term,
                "entity_id": cand.entity_id,
                "start": cand.start,
                "end": cand.end,
                "weight": cand.weight,
            }
            if overlap is None:
                result.slots.append(payload)
                chosen.append(cand)
                occupied.append((cand.start, cand.end))
            else:
                suppressed = dict(payload)
                suppressed["suppressed_by"] = {
                    "entity_type": overlap.entity_type,
                    "term": overlap.term,
                    "entity_id": overlap.entity_id,
                    "start": overlap.start,
                    "end": overlap.end,
                }
                result.suppressed.append(suppressed)
                result.resolutions.append({
                    "winner": {
                        "entity_type": overlap.entity_type,
                        "term": overlap.term,
                        "start": overlap.start,
                        "end": overlap.end,
                    },
                    "loser": {
                        "entity_type": cand.entity_type,
                        "term": cand.term,
                        "start": cand.start,
                        "end": cand.end,
                    },
                    "rule": self._resolution_rule(cand, overlap),
                })

        result.slots.sort(key=lambda item: (item["start"], item["end"],
                                            self.type_priority[item["entity_type"]],
                                            item["entity_id"]))
        result.suppressed.sort(key=lambda item: (item["start"], item["end"],
                                                 item["entity_type"], item["term"]))
        result.resolutions.sort(key=lambda item: (
            item["loser"]["start"], item["loser"]["end"], item["loser"]["term"]))
        return result

    @staticmethod
    def _resolution_rule(loser: Candidate, winner: Candidate) -> str:
        loser_len = loser.end - loser.start
        winner_len = winner.end - winner.start
        if winner_len > loser_len:
            return "longest_match"
        if winner_len == loser_len and winner.priority < loser.priority:
            return "dict_priority"
        if winner_len == loser_len and winner.priority == loser.priority:
            return "start_position"
        return "longest_match"
