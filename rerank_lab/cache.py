"""A small deterministic LRU block cache.

Blocks are deliberately small stage artifacts.  Cache accesses happen on the
engine's coordinating thread, never inside worker threads, so LRU order cannot
depend on thread completion order.  Eviction uses (last_access, block_key) as
its total order, making equal access ticks stable by key.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any


@dataclass
class CacheBlock:
    key: str
    stage: str
    value: dict[str, Any]
    last_access: int
    created_access: int


class DeterministicLRUCache:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("cache capacity must be positive")
        self.capacity = capacity
        self._blocks: dict[str, CacheBlock] = {}
        self._access = 0
        self.evictions: list[dict[str, Any]] = []

    def _tick(self) -> int:
        self._access += 1
        return self._access

    def peek(self, key: str) -> dict[str, Any] | None:
        block = self._blocks.get(key)
        return copy.deepcopy(block.value) if block is not None else None

    def get(self, key: str) -> dict[str, Any] | None:
        block = self._blocks.get(key)
        if block is None:
            return None
        block.last_access = self._tick()
        return copy.deepcopy(block.value)

    def put(self, key: str, stage: str, value: dict[str, Any]) -> list[dict[str, Any]]:
        evicted: list[dict[str, Any]] = []
        tick = self._tick()
        if key in self._blocks:
            block = self._blocks[key]
            block.stage = stage
            block.value = copy.deepcopy(value)
            block.last_access = tick
            return evicted
        self._blocks[key] = CacheBlock(
            key=key,
            stage=stage,
            value=copy.deepcopy(value),
            last_access=tick,
            created_access=tick,
        )
        while len(self._blocks) > self.capacity:
            victim = min(
                self._blocks.values(), key=lambda block: (block.last_access, block.key)
            )
            del self._blocks[victim.key]
            event = {
                "block_key": victim.key,
                "stage": victim.stage,
                "last_access": victim.last_access,
                "created_access": victim.created_access,
                "evicted_at_access": tick,
                "tie_break": "latest_access_then_lexicographic_block_key",
            }
            self.evictions.append(event)
            evicted.append(event)
        return evicted

    def keys(self) -> list[str]:
        return sorted(self._blocks)

    def snapshot(self) -> list[dict[str, Any]]:
        return [
            {
                "block_key": key,
                "stage": self._blocks[key].stage,
                "last_access": self._blocks[key].last_access,
                "created_access": self._blocks[key].created_access,
            }
            for key in sorted(self._blocks)
        ]

    def clear(self) -> None:
        self._blocks.clear()
        self._access = 0

    def __len__(self) -> int:
        return len(self._blocks)
