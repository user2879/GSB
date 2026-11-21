"""Bounded, deterministic LRU block cache used by visual extraction."""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

ValueT = TypeVar("ValueT")


@dataclass(frozen=True)
class CacheStats:
    hits: int
    misses: int
    evictions: int
    recomputations_after_eviction: int
    resident_entries: int
    capacity: int

    def as_dict(self) -> dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "recomputations_after_eviction": self.recomputations_after_eviction,
            "resident_entries": self.resident_entries,
            "capacity": self.capacity,
        }


class BoundedBlockCache(Generic[ValueT]):
    """Thread-safe LRU cache with a capacity measured in feature blocks."""

    def __init__(self, capacity: int) -> None:
        if capacity < 0:
            raise ValueError("cache capacity must be non-negative")
        self.capacity = capacity
        self._entries: OrderedDict[str, ValueT] = OrderedDict()
        self._inflight: set[str] = set()
        self._evicted_keys: set[str] = set()
        self._condition = threading.Condition()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._recomputations = 0

    def get_or_compute(self, key: str, factory: Callable[[], ValueT]) -> ValueT:
        with self._condition:
            while key in self._inflight:
                self._condition.wait()
                if key in self._entries:
                    self._touch_locked(key)
                    self._hits += 1
                    return self._entries[key]
            if key in self._entries:
                self._touch_locked(key)
                self._hits += 1
                return self._entries[key]
            self._inflight.add(key)
            self._misses += 1
            if key in self._evicted_keys:
                self._recomputations += 1

        try:
            value = factory()
        except BaseException:
            with self._condition:
                self._inflight.discard(key)
                self._condition.notify_all()
            raise

        with self._condition:
            self._inflight.discard(key)
            if self.capacity > 0:
                self._entries[key] = value
                self._entries.move_to_end(key)
                while len(self._entries) > self.capacity:
                    evicted_key, _ = self._entries.popitem(last=False)
                    self._evicted_keys.add(evicted_key)
                    self._evictions += 1
            self._condition.notify_all()
            return value

    def _touch_locked(self, key: str) -> None:
        self._entries.move_to_end(key)

    def stats(self) -> CacheStats:
        with self._condition:
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                recomputations_after_eviction=self._recomputations,
                resident_entries=len(self._entries),
                capacity=self.capacity,
            )
