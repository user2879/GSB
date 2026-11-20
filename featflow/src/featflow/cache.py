"""LRU block cache for feature values.

A block key is ``feature|side|entity``.  Tie-breaking for equal LRU access
sequence numbers falls back to block-key lexicographic order, which makes
eviction fully deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass

from .compute import FeatResult


def block_key(feature: str, side: str, entity_id: str) -> str:
    return f"{feature}|{side}|{entity_id}"


@dataclass
class CacheEvent:
    op: str                 # put | hit | evict | recompute
    block_key: str
    point_seq: int
    access_seq: int


class BlockCache:
    def __init__(self, block_limit: int) -> None:
        self.block_limit = block_limit
        self._blocks: dict[str, FeatResult] = {}
        self._access: dict[str, int] = {}
        self._next_seq = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.recomputes = 0
        self.events: list[CacheEvent] = []
        # canonical registry: point_seq -> sorted block keys created in the point
        self.registry: dict[int, list[str]] = {}

    @property
    def enabled(self) -> bool:
        return self.block_limit > 0

    def _touch(self, key: str, point_seq: int) -> int:
        self._next_seq += 1
        self._access[key] = self._next_seq
        return self._next_seq

    def get(self, key: str, point_seq: int) -> FeatResult | None:
        if not self.enabled or key not in self._blocks:
            self.misses += 1
            return None
        seq = self._touch(key, point_seq)
        self.hits += 1
        self.events.append(CacheEvent("hit", key, point_seq, seq))
        return self._blocks[key]

    def put(self, key: str, result: FeatResult, point_seq: int) -> list[str]:
        """Insert/refresh a block, evicting LRU blocks while over the limit."""
        evicted: list[str] = []
        if not self.enabled:
            self.registry.setdefault(point_seq, [])
            return evicted
        bucket = self.registry.setdefault(point_seq, [])
        if key not in bucket:
            bucket.append(key)
            bucket.sort()
        self._blocks[key] = result
        seq = self._touch(key, point_seq)
        self.events.append(CacheEvent("put", key, point_seq, seq))
        while len(self._blocks) > self.block_limit:
            victim = min(self._blocks, key=lambda k: (self._access[k], k))
            del self._blocks[victim]
            del self._access[victim]
            self.evictions += 1
            evicted.append(victim)
            self.events.append(CacheEvent("evict", victim, point_seq, seq))
        return evicted

    def reference(self, key: str, point_seq: int,
                  point_created: set[str]) -> FeatResult | None:
        """Downstream reference within one point.

        A miss on a block already created during *this* point means the block
        was LRU-evicted and therefore its recomputation is counted.  A first
        reference to a not-yet-created block is an ordinary miss.
        """
        result = self.get(key, point_seq)
        if result is None and key in point_created:
            self.recomputes += 1
            self.events.append(CacheEvent("recompute", key, point_seq, self._next_seq))
        return result

    def resident_keys(self) -> list[str]:
        return sorted(self._blocks)

    def contains(self, key: str) -> bool:
        return key in self._blocks

    def restore(self, registry: dict[int, list[str]], next_seq: int) -> None:
        """Used by snapshot recovery: registry persists even when blocks do not."""
        self.registry = {k: list(v) for k, v in registry.items()}
        self._next_seq = next_seq

    def clear_residency(self) -> None:
        """Point boundary: blocks and the per-point registry reset.

        Recompute/hit/evict counters are run-level telemetry and persist.
        """
        self._blocks.clear()
        self._access.clear()
        self._next_seq = 0
