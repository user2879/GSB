"""带 TTL 的缓存：单调时钟毫秒递减、过期回收、负应答缓存与稳定 LRU 逐出。"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

from .records import CacheEntry


class Clock(Protocol):
    def now_ms(self) -> int: ...


class MonotonicClock:
    def now_ms(self) -> int:
        return time.monotonic_ns() // 1_000_000


@dataclass
class StoredEntry:
    entry: CacheEntry
    stored_at_ms: int
    expire_at_ms: int
    last_access_ms: int
    access_order: int

    def remaining_ttl_ms(self, now_ms: int) -> int:
        return max(0, self.expire_at_ms - now_ms)

    def remaining_ttl_seconds(self, now_ms: int) -> int:
        return (self.remaining_ttl_ms(now_ms) + 999) // 1000

    def is_expired(self, now_ms: int) -> bool:
        return now_ms >= self.expire_at_ms


@dataclass
class Eviction:
    key: tuple[str, int]
    reason: str
    source_version: str
    content_digest: str
    ttl_remaining_s: int


@dataclass
class HitResult:
    entry: CacheEntry
    remaining_ttl_s: int
    remaining_ttl_ms: int
    hits: int


@dataclass
class CacheSnapshot:
    """某一时刻缓存账本的确定性视图（用于快照与界面）。"""

    taken_at_ms: int
    items: list[dict] = field(default_factory=list)


class TtlCache:
    """显式稳定规则：

    - 过期优先：任何写入/读取先回收已过期条目；
    - 容量满时按 (最早 access_order) 逐出最近最少使用，键顺序作为最终稳定兜底；
    - 负应答 TTL = min(SOA minimum, negative_ttl_cap)。
    """

    def __init__(
        self,
        *,
        capacity: int = 256,
        negative_ttl_cap: int = 300,
        clock: Clock | None = None,
    ) -> None:
        self.capacity = capacity
        self.negative_ttl_cap = negative_ttl_cap
        self.clock = clock or MonotonicClock()
        self._store: dict[tuple[str, int], StoredEntry] = {}
        self._access_counter = 0
        self.evictions: list[Eviction] = []

    def _now(self) -> int:
        return self.clock.now_ms()

    def negative_ttl(self, soa_minimum: int) -> int:
        return max(0, min(soa_minimum, self.negative_ttl_cap))

    def put(self, entry: CacheEntry, *, ttl_seconds: int | None = None) -> list[Eviction]:
        now = self._now()
        self.reap_expired(now)
        ttl = entry.ttl if ttl_seconds is None else ttl_seconds
        ttl_ms = max(0, ttl) * 1000
        previous = self._store.get(entry.key())
        order = previous.access_order if previous is not None else self._next_order()
        stored = StoredEntry(
            entry=entry,
            stored_at_ms=now,
            expire_at_ms=now + ttl_ms,
            last_access_ms=now,
            access_order=order,
        )
        self._store[entry.key()] = stored
        evictions: list[Eviction] = []
        while len(self._store) > self.capacity:
            evictions.append(self._evict_lru(now))
        self.evictions.extend(evictions)
        return evictions

    def get(self, key: tuple[str, int]) -> HitResult | None:
        now = self._now()
        self.reap_expired(now)
        stored = self._store.get(key)
        if stored is None:
            return None
        stored.last_access_ms = now
        stored.access_order = self._next_order()
        stored.entry.hits += 1
        return HitResult(
            entry=stored.entry,
            remaining_ttl_s=stored.remaining_ttl_seconds(now),
            remaining_ttl_ms=stored.remaining_ttl_ms(now),
            hits=stored.entry.hits,
        )

    def reap_expired(self, now_ms: int | None = None) -> list[tuple[str, int]]:
        now = self._now() if now_ms is None else now_ms
        expired = [key for key, stored in self._store.items() if stored.is_expired(now)]
        for key in expired:
            del self._store[key]
        return expired

    def _evict_lru(self, now: int) -> Eviction:
        key = min(self._store, key=lambda item: (self._store[item].access_order, item))
        stored = self._store.pop(key)
        return Eviction(
            key=key,
            reason="lru",
            source_version=stored.entry.source_version,
            content_digest=stored.entry.content_digest,
            ttl_remaining_s=stored.remaining_ttl_seconds(now),
        )

    def _next_order(self) -> int:
        self._access_counter += 1
        return self._access_counter

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: tuple[str, int]) -> bool:
        return key in self._store

    def view(self) -> list[StoredEntry]:
        """按 (名称, 类型) 稳定排序的当前条目视图。"""
        now = self._now()
        return [self._store[key] for key in sorted(self._store)]

    def export_snapshot(self) -> CacheSnapshot:
        now = self._now()
        items: list[dict] = []
        for key in sorted(self._store):
            stored = self._store[key]
            items.append(
                {
                    "key": list(key),
                    "entry": stored.entry.normalized_payload(),
                    "source_version": stored.entry.source_version,
                    "content_digest": stored.entry.content_digest,
                    "stored_at_ms": stored.stored_at_ms,
                    "expire_at_ms": stored.expire_at_ms,
                    "last_access_ms": stored.last_access_ms,
                    "access_order": stored.access_order,
                    "remaining_ttl_s": stored.remaining_ttl_seconds(now),
                }
            )
        return CacheSnapshot(taken_at_ms=now, items=items)
