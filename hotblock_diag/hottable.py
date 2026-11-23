"""Deterministic hot-key-block table with eviction and suppression records.

The table is fully reconstructible by replaying the append-only ledger against
the same capacity. Once a key block is evicted it loses all counters; a later
selection starts accumulating from zero again.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HotEntry:
    kb: int
    selected_count: int
    accumulated_score: float
    born_round: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "kb": self.kb,
            "selected_count": self.selected_count,
            "accumulated_score": self.accumulated_score,
            "born_round": self.born_round,
        }


@dataclass
class HotTable:
    capacity: int
    keep: int
    entries: dict[int, HotEntry] = field(default_factory=dict)
    evictions: list[dict[str, Any]] = field(default_factory=list)
    suppressions: list[dict[str, Any]] = field(default_factory=list)
    rounds: list[dict[str, Any]] = field(default_factory=list)

    def apply_round(self, round_idx: int, selections: dict[int, list[int]],
                    scores: dict[tuple[int, int], float]) -> dict[str, Any]:
        """selections: qb -> ordered selected kb list for this round."""
        candidates: dict[int, dict[str, Any]] = {}
        for qb, kb_list in selections.items():
            for kb in kb_list:
                bucket = candidates.setdefault(
                    kb, {"count": 0, "score": 0.0})
                bucket["count"] += 1
                bucket["score"] += scores.get((qb, kb), 0.0)

        round_evictions: list[dict[str, Any]] = []
        round_suppressions: list[dict[str, Any]] = []
        for kb in sorted(candidates):
            count = candidates[kb]["count"]
            gained_score = candidates[kb]["score"]
            if kb in self.entries:
                entry = self.entries[kb]
                entry.selected_count += count
                entry.accumulated_score += gained_score
            else:
                self.entries[kb] = HotEntry(
                    kb=kb, selected_count=count,
                    accumulated_score=gained_score, born_round=round_idx,
                )
            while len(self.entries) > self.capacity:
                victim = self._eviction_victim(exclude=kb)
                victim_entry = self.entries.pop(victim)
                record = {
                    "round": round_idx,
                    "kb": victim,
                    "selected_count": victim_entry.selected_count,
                    "accumulated_score": victim_entry.accumulated_score,
                    "reason": "capacity_full",
                }
                self.evictions.append(record)
                round_evictions.append(record)

        ordered = self._ordered_entries()
        effective_keep = min(self.keep, self.capacity)
        kept = ordered[:effective_keep]
        if self.keep > self.capacity:
            for entry in ordered[self.capacity:]:
                record = {
                    "round": round_idx,
                    "kb": entry.kb,
                    "reason": "keep_exceeds_capacity",
                }
                self.suppressions.append(record)
                round_suppressions.append(record)
        existing = {(item["round"], item["kb"]) for item in self.suppressions}
        for entry in ordered[effective_keep:]:
            record = {
                "round": round_idx,
                "kb": entry.kb,
                "reason": "hot_keep_limit",
            }
            key = (round_idx, entry.kb)
            if key not in existing:
                self.suppressions.append(record)
                existing.add(key)
            if record not in round_suppressions:
                round_suppressions.append(record)
        snapshot = {
            "round": round_idx,
            "entries": [entry.to_dict() for entry in kept],
            "evictions": round_evictions,
            "suppressions": round_suppressions,
            "capacity": self.capacity,
            "keep": self.keep,
        }
        self.rounds.append(snapshot)
        return snapshot

    def _eviction_victim(self, exclude: int | None = None) -> int:
        candidates = [entry for entry in self.entries.values()
                      if entry.kb != exclude]
        entries = sorted(candidates,
                         key=lambda item: (item.selected_count, item.kb))
        return entries[0].kb

    def _ordered_entries(self) -> list[HotEntry]:
        entries = list(self.entries.values())
        entries.sort(key=lambda item: (-item.selected_count,
                                       -item.accumulated_score, item.kb))
        return entries

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "keep": self.keep,
            "entries": [entry.to_dict() for entry in self._ordered_entries()],
            "evictions": self.evictions,
            "suppressions": self.suppressions,
            "rounds": self.rounds,
        }

    @classmethod
    def rebuild(cls, capacity: int, keep: int,
                rounds: list[tuple[int, dict[int, list[int]],
                                    dict[tuple[int, int], float]]]) -> "HotTable":
        table = cls(capacity=capacity, keep=keep)
        for round_idx, selections, scores in rounds:
            table.apply_round(round_idx, selections, scores)
        return table
