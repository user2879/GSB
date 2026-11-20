"""Deterministic capacity-bounded edge cache."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Callable

from .errors import EdgeError
from .model import EVICTION_RULES, Policy, cache_key


def monotonic_ms() -> int:
    return time.monotonic_ns() // 1_000_000


@dataclass(frozen=True)
class CacheEntry:
    key: str
    path: str
    etag: str
    content_version: str
    written_mono_ms: int
    ttl_ms: int
    remaining_ttl_ms: int
    entry_version: int
    content_size: int
    last_access_mono_ms: int

    @property
    def expires_mono_ms(self) -> int:
        return self.written_mono_ms + self.ttl_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_key": self.key,
            "path": self.path,
            "etag": self.etag,
            "content_version": self.content_version,
            "written_mono_ms": self.written_mono_ms,
            "ttl_ms": self.ttl_ms,
            "remaining_ttl_ms": self.remaining_ttl_ms,
            "entry_version": self.entry_version,
            "content_size": self.content_size,
        }


class EdgeCache:
    """A node-local cache.

    Reads may refresh TTL accounting, but victim selection deliberately uses
    only write-time and key metadata.  Consequently, the set evicted for a
    fixed entry population is independent of lookup/access order.
    """

    def __init__(
        self,
        *,
        node_id: str,
        capacity: int,
        eviction_rule: str,
        policy: Policy,
        ledger_version: int,
        clock: Callable[[], int] = monotonic_ms,
    ) -> None:
        if eviction_rule not in EVICTION_RULES:
            raise EdgeError(
                code="E_SPEC",
                message="unsupported eviction rule",
                field_path="eviction_rule",
                details={"rule": eviction_rule},
            )
        if capacity <= 0:
            raise EdgeError(
                code="E_SPEC",
                message="capacity must be positive",
                field_path="capacity",
            )
        self.node_id = node_id
        self.capacity = capacity
        self.eviction_rule = eviction_rule
        self.policy = policy
        self.ledger_version = ledger_version
        self.clock = clock
        self._entries: dict[str, CacheEntry] = {}
        self._events: list[dict[str, Any]] = []
        self._event_seq = 0
        self._entry_version = 0

    @property
    def entries(self) -> dict[str, CacheEntry]:
        return self._entries

    def bind(self, ledger_version: int) -> None:
        if ledger_version < self.ledger_version:
            raise EdgeError(
                code="E_STALE",
                message="cache write is bound to a stale ledger version",
                details={
                    "requested_ledger_version": ledger_version,
                    "active_ledger_version": self.ledger_version,
                },
            )
        if ledger_version != self.ledger_version:
            self._entries.clear()
            self._events.clear()
            self._event_seq = 0
            self.ledger_version = ledger_version

    def advance(self, ledger_version: int) -> None:
        """Move to a newer ledger round while preserving cached entries.

        Content evolution needs the old entry ETags available long enough to
        emit explicit invalidation events, so this is distinct from ``bind``.
        """

        if ledger_version < self.ledger_version:
            raise EdgeError(
                code="E_STALE",
                message="cache cannot advance to an older ledger version",
                details={
                    "requested_ledger_version": ledger_version,
                    "active_ledger_version": self.ledger_version,
                },
            )
        self.ledger_version = ledger_version
        self._events.clear()
        self._event_seq = 0

    def configure(self, policy: Policy, *, ledger_version: int) -> None:
        if ledger_version < self.ledger_version:
            raise EdgeError(
                code="E_STALE",
                message="cannot configure a node with an older ledger version",
                details={
                    "requested_ledger_version": ledger_version,
                    "active_ledger_version": self.ledger_version,
                },
            )
        node = policy.node(self.node_id)
        self.policy = policy
        self.capacity = node.capacity
        self.eviction_rule = node.eviction_rule
        if ledger_version != self.ledger_version:
            self.ledger_version = ledger_version
            self._entries.clear()
            self._events.clear()
            self._event_seq = 0
            self._entry_version = 0

    def _key(self, path: str, etag: str) -> str:
        return cache_key(self.policy, node_id=self.node_id, path=path, etag=etag)

    def _record_event(self, entry: CacheEntry, reason: str, **details: Any) -> None:
        self._event_seq += 1
        event = {
            "ledger_version": self.ledger_version,
            "node_id": self.node_id,
            "event_seq": self._event_seq,
            "cache_key": entry.key,
            "path": entry.path,
            "etag": entry.etag,
            "entry_version": entry.entry_version,
            "reason": reason,
            "rule": self.eviction_rule,
        }
        event.update(details)
        self._events.append(event)

    def _rank(self, entry: CacheEntry) -> tuple[Any, ...]:
        if self.eviction_rule == "stable_fifo":
            return (entry.written_mono_ms, entry.entry_version, entry.key)
        if self.eviction_rule == "smallest_key":
            return (entry.key, entry.written_mono_ms, entry.entry_version)
        if self.eviction_rule == "nearest_expiry":
            return (entry.expires_mono_ms, entry.key, entry.entry_version)
        raise AssertionError(self.eviction_rule)

    def _evict_one(self, reason: str, **details: Any) -> CacheEntry:
        victim = min(self._entries.values(), key=self._rank)
        rank = list(self._rank(victim))
        del self._entries[victim.key]
        self._record_event(victim, reason, victim_rank=rank, **details)
        return victim

    def _reconcile_expired(self, now: int) -> None:
        expired = [
            entry
            for entry in self._entries.values()
            if entry.remaining_ttl_ms - max(0, now - entry.last_access_mono_ms) <= 0
        ]
        for entry in sorted(expired, key=lambda item: item.key):
            current = self._entries.get(entry.key)
            if current is None:
                continue
            updated = self._decrement(current, now)
            if updated.remaining_ttl_ms <= 0:
                del self._entries[updated.key]
                self._record_event(updated, "expired")

    def _decrement(self, entry: CacheEntry, now: int) -> CacheEntry:
        elapsed = max(0, now - entry.last_access_mono_ms)
        remaining = max(0, entry.remaining_ttl_ms - elapsed)
        return replace(entry, remaining_ttl_ms=remaining, last_access_mono_ms=now)

    def get(self, *, ledger_version: int, path: str, etag: str) -> CacheEntry | None:
        self._require_ledger(ledger_version)
        key = self._key(path, etag)
        entry = self._entries.get(key)
        if entry is None:
            return None
        now = self.clock()
        updated = self._decrement(entry, now)
        if updated.remaining_ttl_ms <= 0:
            del self._entries[key]
            self._record_event(updated, "expired")
            return None
        self._entries[key] = updated
        return updated

    def put(
        self,
        *,
        ledger_version: int,
        path: str,
        etag: str,
        content_version: str,
        ttl_ms: int,
        content_size: int,
        written_mono_ms: int | None = None,
        remaining_ttl_ms: int | None = None,
        entry_version: int | None = None,
    ) -> CacheEntry:
        self._require_ledger(ledger_version)
        if ttl_ms <= 0:
            raise EdgeError(
                code="E_STATE",
                message="ttl_ms must be positive",
                path=path,
                field_path="ttl_ms",
            )
        now = self.clock()
        self._reconcile_expired(now)
        key = self._key(path, etag)
        if entry_version is None:
            self._entry_version += 1
            sequence = self._entry_version
        else:
            sequence = int(entry_version)
            self._entry_version = max(self._entry_version, sequence)
        written = int(written_mono_ms if written_mono_ms is not None else now)
        if remaining_ttl_ms is None:
            initial_remaining = max(0, ttl_ms - max(0, now - written))
        else:
            initial_remaining = max(0, min(ttl_ms, int(remaining_ttl_ms)))
        entry = CacheEntry(
            key=key,
            path=path,
            etag=etag,
            content_version=content_version,
            written_mono_ms=written,
            ttl_ms=ttl_ms,
            remaining_ttl_ms=initial_remaining,
            entry_version=sequence,
            content_size=content_size,
            last_access_mono_ms=now,
        )
        if initial_remaining <= 0:
            self._record_event(entry, "expired")
            return entry
        self._entries[key] = entry
        while len(self._entries) > self.capacity:
            self._evict_one("capacity")
        return entry

    def invalidate_path(self, *, ledger_version: int, path: str, reason: str = "invalidate") -> list[dict[str, Any]]:
        self._require_ledger(ledger_version)
        removed: list[dict[str, Any]] = []
        for key in sorted(self._entries):
            entry = self._entries[key]
            if entry.path != path:
                continue
            del self._entries[key]
            self._record_event(entry, reason)
            removed.append(entry.to_dict())
        return removed

    def clear(self, *, ledger_version: int, reason: str = "clear") -> None:
        self._require_ledger(ledger_version)
        for key in sorted(self._entries):
            entry = self._entries[key]
            del self._entries[key]
            self._record_event(entry, reason)

    def rebase_content_version(
        self,
        *,
        ledger_version: int,
        content_version: str,
        valid_etags: dict[str, str],
    ) -> None:
        """Bind unchanged entries to the round's single content version."""

        self._require_ledger(ledger_version)
        for key in sorted(self._entries):
            entry = self._entries[key]
            if valid_etags.get(entry.path) != entry.etag:
                del self._entries[key]
                self._record_event(entry, "rebase_stale")
                continue
            if entry.content_version != content_version:
                self._entries[key] = replace(entry, content_version=content_version)

    def state(self, *, ledger_version: int, expire: bool = True) -> list[CacheEntry]:
        self._require_ledger(ledger_version)
        if expire:
            self._reconcile_expired(self.clock())
        return [self._entries[key] for key in sorted(self._entries)]

    def entry_state(self, *, ledger_version: int, expire: bool = True) -> list[dict[str, Any]]:
        return [entry.to_dict() for entry in self.state(ledger_version=ledger_version, expire=expire)]

    def events(self, *, ledger_version: int) -> list[dict[str, Any]]:
        self._require_ledger(ledger_version)
        return [dict(event) for event in self._events]

    def restore(
        self,
        *,
        ledger_version: int,
        entries: list[dict[str, Any]],
        normalize_snapshot_ttl: bool = False,
    ) -> None:
        """Wipe and restore a node from persisted state.

        When ``normalize_snapshot_ttl`` is true the saved remaining TTL is
        restarted at restore time.  Ledger-backed crash recovery keeps the
        original monotonic baseline, so elapsed downtime still counts.
        """

        self.bind(ledger_version)
        self._entries.clear()
        self._events.clear()
        self._event_seq = 0
        now = self.clock()
        for raw in sorted(entries, key=lambda item: (str(item.get("node_id", "")), str(item.get("cache_key", "")))):
            if str(raw.get("node_id", self.node_id)) != self.node_id:
                raise EdgeError(
                    code="E_STATE",
                    message="entry belongs to a different node",
                    details={"node_id": raw.get("node_id"), "expected": self.node_id},
                )
            path = str(raw["path"])
            etag = str(raw["etag"])
            ttl_ms = int(raw["ttl_ms"])
            written = int(raw.get("written_mono_ms", now))
            remaining_raw = int(raw.get("remaining_ttl_ms", ttl_ms))
            if normalize_snapshot_ttl:
                written = now
                remaining = max(0, min(ttl_ms, remaining_raw))
                ttl_ms = max(1, remaining)
            else:
                remaining = max(0, min(ttl_ms, ttl_ms - max(0, now - written)))
            entry = CacheEntry(
                key=str(raw.get("cache_key") or self._key(path, etag)),
                path=path,
                etag=etag,
                content_version=str(raw["content_version"]),
                written_mono_ms=written,
                ttl_ms=ttl_ms,
                remaining_ttl_ms=remaining,
                entry_version=int(raw.get("entry_version", 0)),
                content_size=int(raw.get("content_size", 0)),
                last_access_mono_ms=now,
            )
            if entry.remaining_ttl_ms <= 0:
                self._record_event(entry, "expired", during="restore")
                continue
            self._entries[entry.key] = entry
            self._entry_version = max(self._entry_version, entry.entry_version)
        while len(self._entries) > self.capacity:
            self._evict_one("capacity", during="restore")

    def _require_ledger(self, ledger_version: int) -> None:
        if ledger_version != self.ledger_version:
            raise EdgeError(
                code="E_STALE",
                message="cache operation does not match the node's active ledger version",
                details={
                    "requested_ledger_version": ledger_version,
                    "active_ledger_version": self.ledger_version,
                },
            )
