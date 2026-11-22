"""Thread-safe LRU block cache with deterministic eviction ties."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any

from .errors import AppError, E_CONFIG_INVALID


@dataclass
class _Entry:
    value: Any
    last_access: int
    sort_key: str
    inserts: int


class LRUBlockCache:
    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise AppError(
                E_CONFIG_INVALID,
                "cache_blocks must be a positive integer (less than one block is invalid)",
                stage="cache",
                details={"cache_blocks": capacity},
            )
        self.capacity = capacity
        self._entries: dict[str, _Entry] = {}
        self._lock = RLock()
        self._sequence = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.inserts = 0
        self.events: list[dict[str, Any]] = []

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            entry.last_access = self._next_sequence()
            self.hits += 1
            self.events.append(
                {
                    "kind": "cache_hit",
                    "key": key,
                    "sort_key": entry.sort_key,
                    "access_sequence": entry.last_access,
                    "insert_sequence": entry.inserts,
                }
            )
            return entry.value

    def put(self, key: str, value: Any, *, sort_key: str) -> None:
        with self._lock:
            existing = self._entries.get(key)
            if existing is not None:
                existing.value = value
                existing.last_access = self._next_sequence()
                existing.sort_key = sort_key
                return
            while len(self._entries) >= self.capacity:
                victim_key, victim = min(
                    self._entries.items(),
                    key=lambda item: (item[1].last_access, item[0]),
                )
                del self._entries[victim_key]
                self.evictions += 1
                self.events.append(
                    {
                        "kind": "cache_evict",
                        "key": victim_key,
                        "sort_key": victim.sort_key,
                        "evicted_access_sequence": victim.last_access,
                        "insert_sequence": victim.inserts,
                        "tie_break": "block_key_lexicographic",
                    }
                )
            insert_sequence = self._next_sequence()
            self._entries[key] = _Entry(
                value=value,
                last_access=insert_sequence,
                sort_key=sort_key,
                inserts=insert_sequence,
            )
            self.inserts += 1
            self.events.append(
                {
                    "kind": "cache_insert",
                    "key": key,
                    "sort_key": sort_key,
                    "access_sequence": insert_sequence,
                }
            )

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "capacity": self.capacity,
                "resident_blocks": len(self._entries),
                "hits": self.hits,
                "misses": self.misses,
                "inserts": self.inserts,
                "evictions": self.evictions,
                "access_sequence": self._sequence,
            }

    def sorted_events(self) -> list[dict[str, Any]]:
        with self._lock:
            return sorted(
                (dict(event) for event in self.events),
                key=lambda event: (
                    event.get("sort_key", ""),
                    event.get("kind", ""),
                    event.get("access_sequence", 0),
                    event.get("key", ""),
                ),
            )
