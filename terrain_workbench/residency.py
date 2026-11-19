"""Texture residency management with LRU eviction and an auditable ledger.

Every elevation/normal tile upload is accounted in bytes. When the residency
budget would be exceeded, least-recently-used tiles are evicted. Each event
(upload / evict / hit) is appended to the ledger with the running totals, so
the ledger can be replayed and audited. A tile larger than the whole budget
raises E_TEXTURE_RESIDENT. Re-upload after eviction must produce identical
content - callers pass a content hash and the manager verifies consistency.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from .errors import E_TEXTURE_RESIDENT, TerrainError


@dataclass
class LedgerEvent:
    seq: int
    frame: int
    op: str          # upload | evict | hit
    key: str
    bytes: int
    resident_after: int
    count_after: int
    content_sha: str

    def to_dict(self):
        return {
            "seq": self.seq,
            "frame": self.frame,
            "op": self.op,
            "key": self.key,
            "bytes": self.bytes,
            "resident_after": self.resident_after,
            "count_after": self.count_after,
            "content_sha": self.content_sha,
        }


class ResidencyManager:
    def __init__(self, capacity_bytes: int):
        if capacity_bytes <= 0:
            raise TerrainError(
                E_TEXTURE_RESIDENT, "memory budget must be positive",
                capacity=capacity_bytes,
            )
        self.capacity = int(capacity_bytes)
        self._resident: OrderedDict[str, tuple[int, str]] = OrderedDict()
        self._bytes = 0
        self.ledger: list[LedgerEvent] = []
        self._seq = 0

    @property
    def resident_bytes(self) -> int:
        return self._bytes

    @property
    def resident_count(self) -> int:
        return len(self._resident)

    def _record(self, frame, op, key, nbytes, sha):
        self._seq += 1
        self.ledger.append(
            LedgerEvent(self._seq, frame, op, key, nbytes, self._bytes,
                        len(self._resident), sha)
        )

    def require(self, key: str, nbytes: int, content_sha: str, frame: int) -> None:
        if nbytes > self.capacity:
            raise TerrainError(
                E_TEXTURE_RESIDENT,
                f"tile {key} needs {nbytes} bytes > budget {self.capacity}",
                block=key,
                bytes=nbytes,
                capacity=self.capacity,
            )
        if key in self._resident:
            old_bytes, old_sha = self._resident[key]
            if old_sha != content_sha:
                raise TerrainError(
                    E_TEXTURE_RESIDENT,
                    f"content hash changed for resident tile {key}",
                    block=key,
                )
            self._resident.move_to_end(key)
            self._record(frame, "hit", key, nbytes, content_sha)
            return
        while self._bytes + nbytes > self.capacity and self._resident:
            evict_key, (evict_bytes, evict_sha) = next(iter(self._resident.items()))
            del self._resident[evict_key]
            self._bytes -= evict_bytes
            self._record(frame, "evict", evict_key, evict_bytes, evict_sha)
        self._resident[key] = (nbytes, content_sha)
        self._bytes += nbytes
        self._record(frame, "upload", key, nbytes, content_sha)

    def audit(self) -> dict:
        """Replay the ledger and verify the running totals."""
        resident: OrderedDict[str, int] = OrderedDict()
        total = 0
        uploads = evictions = hits = 0
        for ev in self.ledger:
            if ev.op == "upload":
                resident[ev.key] = ev.bytes
                total += ev.bytes
                uploads += 1
            elif ev.op == "evict":
                total -= resident.pop(ev.key)
                evictions += 1
            elif ev.op == "hit":
                resident.move_to_end(ev.key)
                hits += 1
            if total != ev.resident_after or len(resident) != ev.count_after:
                raise TerrainError(
                    E_TEXTURE_RESIDENT,
                    f"ledger audit failed at seq {ev.seq}",
                    block=ev.key,
                )
        if total > self.capacity:
            raise TerrainError(
                E_TEXTURE_RESIDENT, "resident bytes exceed capacity after replay"
            )
        return {
            "uploads": uploads,
            "evictions": evictions,
            "hits": hits,
            "resident_bytes": total,
            "resident_count": len(resident),
            "capacity": self.capacity,
        }
