from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .fingerprint import digest_json


@dataclass
class HotEntry:
    block_id: int
    invalidations: int = 0
    recompute_nnz: int = 0
    last_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "invalidations": self.invalidations,
            "recompute_nnz": self.recompute_nnz,
            "last_reason": self.last_reason,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "HotEntry":
        return cls(
            block_id=int(value["block_id"]),
            invalidations=int(value["invalidations"]),
            recompute_nnz=int(value["recompute_nnz"]),
            last_reason=str(value.get("last_reason", "")),
        )


class HotBlockTable:
    """Capacity-bounded invalidation statistics with deterministic eviction."""

    def __init__(self, capacity: int, entries: Iterable[HotEntry] = ()) -> None:
        if capacity < 0:
            raise ValueError("hot capacity must not be negative")
        self.capacity = int(capacity)
        self._entries: dict[int, HotEntry] = {}
        for entry in entries:
            if self.capacity == 0:
                break
            self._entries[entry.block_id] = HotEntry.from_dict(entry.to_dict())
        self._trim_to_capacity()

    def _eviction_key(self, entry: HotEntry) -> tuple[int, int]:
        return (entry.invalidations, entry.block_id)

    def _trim_to_capacity(self) -> None:
        while len(self._entries) > self.capacity:
            victim = min(self._entries.values(), key=self._eviction_key)
            del self._entries[victim.block_id]

    def invalidate(self, block_id: int, reason: str) -> None:
        if self.capacity == 0:
            return
        entry = self._entries.get(block_id)
        if entry is None:
            entry = HotEntry(block_id=block_id, invalidations=1, last_reason=reason)
            self._entries[block_id] = entry
            if len(self._entries) > self.capacity:
                victim = min(self._entries.values(), key=self._eviction_key)
                del self._entries[victim.block_id]
        else:
            entry.invalidations += 1
            entry.last_reason = reason

    def add_recompute(self, block_id: int, nnz: int) -> None:
        entry = self._entries.get(block_id)
        if entry is None:
            if self.capacity == 0:
                return
            entry = HotEntry(block_id=block_id, invalidations=0)
            self._entries[block_id] = entry
        entry.recompute_nnz += int(nnz)
        self._trim_to_capacity()

    def clear(self) -> None:
        self._entries.clear()

    def prune(self, valid_block_ids: Iterable[int]) -> None:
        valid = set(valid_block_ids)
        self._entries = {
            block_id: entry
            for block_id, entry in self._entries.items()
            if block_id in valid
        }

    def entries(self) -> list[HotEntry]:
        return sorted(
            (HotEntry.from_dict(entry.to_dict()) for entry in self._entries.values()),
            key=lambda entry: (-entry.invalidations, -entry.recompute_nnz, entry.block_id),
        )

    def block_ids(self) -> set[int]:
        return set(self._entries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "entries": [entry.to_dict() for entry in self.entries()],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "HotBlockTable":
        return cls(
            capacity=int(value["capacity"]),
            entries=[HotEntry.from_dict(item) for item in value.get("entries", [])],
        )

    def fingerprint(self) -> str:
        return digest_json(self.to_dict())
