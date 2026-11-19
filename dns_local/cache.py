"""Thread-safe TTL cache with single-flight fills and deterministic eviction."""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Literal

from .clock import Clock
from .names import normalize_name
from .wire import CLASS_IN, ResourceRecord, type_to_int


CacheSource = Literal["ZONE", "NEGATIVE", "UPSTREAM"]
CACHE_SOURCES: tuple[CacheSource, ...] = ("ZONE", "NEGATIVE", "UPSTREAM")


@dataclass(frozen=True, order=True)
class CacheKey:
    name: str
    qtype: int
    qclass: int = CLASS_IN

    @classmethod
    def create(cls, name: str, qtype: str | int, qclass: int = CLASS_IN) -> "CacheKey":
        return cls(normalize_name(name), type_to_int(qtype), qclass)

    @property
    def display(self) -> str:
        type_name = {
            1: "A",
            2: "NS",
            5: "CNAME",
            6: "SOA",
            28: "AAAA",
        }.get(self.qtype, f"TYPE{self.qtype}")
        return f"{self.name}|{type_name}|CLASS{self.qclass}"


@dataclass(frozen=True)
class CacheEntry:
    key: CacheKey
    rcode: int
    answers: tuple[ResourceRecord, ...]
    authority: tuple[ResourceRecord, ...]
    ttl: int
    stored_ns: int
    source: CacheSource
    source_serial: int
    attribution: tuple[str, ...]
    negative: bool = False
    metadata: tuple[str, ...] = ()

    def age_seconds(self, now_ns: int) -> int:
        return max(0, (int(now_ns) - self.stored_ns) // 1_000_000_000)

    def remaining_ttl(self, now_ns: int) -> int:
        return max(0, self.ttl - self.age_seconds(now_ns))

    def expired(self, now_ns: int) -> bool:
        return self.remaining_ttl(now_ns) <= 0

    def aged_records(self, now_ns: int) -> tuple[tuple[ResourceRecord, ...], tuple[ResourceRecord, ...]]:
        age = self.age_seconds(now_ns)
        answers = tuple(record.with_ttl(record.ttl - age) for record in self.answers)
        authority = tuple(record.with_ttl(record.ttl - age) for record in self.authority)
        return answers, authority


@dataclass(frozen=True)
class CacheHit:
    entry: CacheEntry
    remaining_ttl: int
    age_seconds: int
    filled: bool = False


class FillDiscarded(Exception):
    """The fill finished after its state generation became stale."""


class Cache:
    """A complete-or-absent cache guarded by one condition variable.

    Eviction policy: expired entries first, then least-recently-used, with the
    canonical key string as a stable tie-breaker.
    """

    def __init__(self, clock: Clock, *, capacity: int = 128) -> None:
        if capacity <= 0:
            raise ValueError("cache capacity must be positive")
        self.clock = clock
        self._capacity = capacity
        self._entries: dict[CacheKey, CacheEntry] = {}
        self._last_access: dict[CacheKey, int] = {}
        self._access_counter = 0
        self._fill_inflight: set[CacheKey] = set()
        self._generation = 0
        self._lock = threading.Condition(threading.RLock())

    @property
    def capacity(self) -> int:
        with self._lock:
            return self._capacity

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def __len__(self) -> int:
        with self._lock:
            now = self.clock.monotonic_ns()
            self._reap_locked(now)
            return len(self._entries)

    def _touch_locked(self, key: CacheKey) -> None:
        self._access_counter += 1
        self._last_access[key] = self._access_counter

    def _reap_locked(self, now: int) -> list[CacheKey]:
        expired = [key for key, entry in self._entries.items() if entry.expired(now)]
        for key in expired:
            self._entries.pop(key, None)
            self._last_access.pop(key, None)
        if expired:
            self._generation += 1
        return expired

    def _get_locked(self, key: CacheKey, now: int) -> CacheHit | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        remaining = entry.remaining_ttl(now)
        if remaining <= 0:
            self._entries.pop(key, None)
            self._last_access.pop(key, None)
            self._generation += 1
            return None
        self._touch_locked(key)
        return CacheHit(entry, remaining, entry.age_seconds(now), filled=False)

    def _evict_locked(self, now: int) -> CacheKey | None:
        self._reap_locked(now)
        if len(self._entries) < self._capacity:
            return None
        victim = min(
            self._entries,
            key=lambda key: (self._last_access.get(key, -1), key.display),
        )
        self._entries.pop(victim, None)
        self._last_access.pop(victim, None)
        self._generation += 1
        return victim

    def _put_locked(self, entry: CacheEntry, now: int) -> None:
        if entry.expired(now):
            self._entries.pop(entry.key, None)
            self._last_access.pop(entry.key, None)
            self._generation += 1
            return
        if entry.key not in self._entries:
            self._evict_locked(now)
        self._entries[entry.key] = entry
        self._touch_locked(entry.key)
        self._generation += 1

    def get(self, key: CacheKey) -> CacheHit | None:
        with self._lock:
            return self._get_locked(key, self.clock.monotonic_ns())

    def put(self, entry: CacheEntry) -> None:
        with self._lock:
            self._put_locked(entry, self.clock.monotonic_ns())

    def get_or_fill(
        self,
        key: CacheKey,
        loader: Callable[[], CacheEntry],
        *,
        commit_guard: Callable[[], bool] | None = None,
    ) -> CacheHit:
        """Return a hit or let exactly one thread fill the key.

        The loader runs outside the cache lock. The guard is evaluated under the
        lock immediately before insertion and is used to reject fills whose
        zone/session generation changed while the upstream request was in flight.
        """

        while True:
            with self._lock:
                now = self.clock.monotonic_ns()
                hit = self._get_locked(key, now)
                if hit is not None:
                    return hit
                if key not in self._fill_inflight:
                    self._fill_inflight.add(key)
                    break
                self._lock.wait(timeout=0.25)

        try:
            entry = loader()
            with self._lock:
                if commit_guard is not None and not commit_guard():
                    raise FillDiscarded("cache fill became stale before commit")
                now = self.clock.monotonic_ns()
                self._put_locked(entry, now)
                hit = self._get_locked(key, now)
                if hit is None:
                    hit = CacheHit(entry, entry.remaining_ttl(now), entry.age_seconds(now), filled=True)
                else:
                    hit = replace(hit, filled=True)
                return hit
        finally:
            with self._lock:
                self._fill_inflight.discard(key)
                self._lock.notify_all()

    def invalidate_sources(self, sources: Iterable[CacheSource]) -> list[CacheKey]:
        wanted = set(sources)
        with self._lock:
            keys = [key for key, entry in self._entries.items() if entry.source in wanted]
            for key in keys:
                self._entries.pop(key, None)
                self._last_access.pop(key, None)
            if keys:
                self._generation += 1
            return keys

    def invalidate_keys(self, keys: Iterable[CacheKey]) -> list[CacheKey]:
        with self._lock:
            removed: list[CacheKey] = []
            for key in keys:
                if key in self._entries:
                    self._entries.pop(key, None)
                    self._last_access.pop(key, None)
                    removed.append(key)
            if removed:
                self._generation += 1
            return removed

    def clear(self) -> int:
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            self._last_access.clear()
            if count:
                self._generation += 1
            return count

    def replace_all(self, entries: Iterable[CacheEntry]) -> int:
        """Atomically swap the complete entry set; no reader sees a mixed table."""

        with self._lock:
            replacement = {entry.key: entry for entry in entries if not entry.expired(self.clock.monotonic_ns())}
            count = len(replacement)
            self._entries = replacement
            self._last_access = {}
            for key in sorted(replacement, key=lambda item: item.display):
                self._touch_locked(key)
            self._generation += 1
            self._lock.notify_all()
            return count

    def entries(self) -> tuple[CacheEntry, ...]:
        with self._lock:
            now = self.clock.monotonic_ns()
            self._reap_locked(now)
            return tuple(self._entries[key] for key in sorted(self._entries, key=lambda item: item.display))

    def snapshot_view(self) -> tuple[tuple[CacheEntry, int], ...]:
        with self._lock:
            now = self.clock.monotonic_ns()
            self._reap_locked(now)
            return tuple(
                (self._entries[key], self._entries[key].remaining_ttl(now))
                for key in sorted(self._entries, key=lambda item: item.display)
            )
