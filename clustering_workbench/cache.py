"""Thread-safe LRU block residency with deterministic eviction tie breaks."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class CacheEntry:
    key: str
    value: Any
    last_access: int
    compute_count: int = 1


class ResidentBlockCache:
    def __init__(self, max_blocks: int):
        if int(max_blocks) <= 0:
            raise ValueError("max_blocks must be positive")
        self.max_blocks = int(max_blocks)
        self._entries: dict[str, CacheEntry] = {}
        self._compute_counts: dict[str, int] = {}
        self._sequence = 0
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.recomputes = 0
        self.events: list[dict[str, Any]] = []

    @property
    def keys(self) -> list[str]:
        with self._lock:
            return sorted(self._entries)

    def _record(self, event: str, key: str, **extra: Any) -> None:
        if len(self.events) < 400:
            self.events.append({"sequence": self._sequence, "event": event, "key": key, **extra})

    def get(self, key: str, compute: Callable[[], Any]) -> Any:
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
            entry = self._entries.get(key)
            if entry is not None:
                self.hits += 1
                entry.last_access = sequence
                self._record("hit", key)
                return entry.value

            self.misses += 1
            value = compute()
            previous_computes = self._compute_counts.get(key, 0)
            self._compute_counts[key] = previous_computes + 1
            if previous_computes:
                self.recomputes += 1
            entry = CacheEntry(key=key, value=value, last_access=sequence)
            entry.compute_count = previous_computes + 1
            self._entries[key] = entry
            self._record("miss_compute", key)
            while len(self._entries) > self.max_blocks:
                eviction_key, eviction_entry = min(
                    self._entries.items(),
                    key=lambda item: (item[1].last_access, item[0]),
                )
                del self._entries[eviction_key]
                self.evictions += 1
                self._record("evict", eviction_key, access_sequence=eviction_entry.last_access)
            return value

    def invalidate(self, key: str) -> bool:
        with self._lock:
            removed = self._entries.pop(key, None)
            if removed is not None:
                self._record("invalidate", key)
            return removed is not None

    def summary(self) -> dict[str, Any]:
        with self._lock:
            compute_counts = {key: self._compute_counts[key] for key in sorted(self._compute_counts)}
            return {
                "max_blocks": self.max_blocks,
                "resident_blocks": len(self._entries),
                "resident_keys": sorted(self._entries),
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "recomputes": self.recomputes,
                "compute_counts": compute_counts,
                "events": list(self.events),
                "policy": "least recently used; tie by block key ascending",
            }


def deterministic_recompute_probe(cache: ResidentBlockCache, rounds: int = 4) -> dict[str, Any]:
    """Exercise eviction and reprovision deterministically for diagnostics/tests."""
    values: dict[str, int] = {}
    for index in range(rounds):
        key = f"probe:{index:02d}"
        value = cache.get(key, lambda index=index: index * index)
        values[key] = int(value)
    first_key = "probe:00"
    values[first_key] = int(cache.get(first_key, lambda: 0))
    return {"values": values, "summary": cache.summary()}
