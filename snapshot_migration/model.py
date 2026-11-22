"""In-memory domain model for a snapshot generation chain.

The model is deliberately immutable once validated; :mod:`snapshot_migration.chain`
turns these declarations into persistent fragments.  Key ranges use inclusive
string bounds so that two adjacent fragments can be proven contiguous by
comparing the previous upper bound with the next lower bound.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class KeyRange:
    range_id: str
    lower: str
    upper: str
    keys: Tuple[str, ...]
    action: str  # inherit | rewrite | discard | new
    tombstone: bool

    @property
    def is_active(self):
        return not self.tombstone


@dataclass(frozen=True)
class Generation:
    generation_id: str
    seq: int
    parent_id: Optional[str]
    ranges: Tuple[KeyRange, ...]
    rules: Tuple[Tuple[str, str], ...]  # (range_id, action)

    def range(self, range_id):
        for item in self.ranges:
            if item.range_id == range_id:
                return item
        return None

    @property
    def active_ranges(self):
        return tuple(item for item in self.ranges if item.is_active)


@dataclass(frozen=True)
class Limits:
    lower: int
    upper: int


@dataclass(frozen=True)
class SnapshotChain:
    title: str
    limits: Limits
    generations: Tuple[Generation, ...]
    range_universe: Tuple[Tuple[str, str, str], ...]  # (range_id, lower, upper)

    def generation(self, generation_id):
        for item in self.generations:
            if item.generation_id == generation_id:
                return item
        return None

    def ancestors(self, generation_id):
        """Return [self, parent, grandparent, ...] following parent links."""
        chain = []
        current = self.generation(generation_id)
        seen = set()
        while current is not None:
            if current.generation_id in seen:
                break
            seen.add(current.generation_id)
            chain.append(current.generation_id)
            current = self.generation(current.parent_id) if current.parent_id else None
        return chain
