"""Auditable reuse ledger with content-fingerprint cache keys."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from .fingerprints import cache_key, composite_key, digest


@dataclass
class InvalidationRecord:
    product: str
    product_key: str
    text_id: str
    reason: str
    changed_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReuseRecord:
    product: str
    product_key: str
    text_id: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CacheEntry:
    """One cached product with the fingerprints it depends on."""

    key: str
    kind: str
    text_id: str
    referenced_entry_fp: dict[str, str]
    text_fp: str
    chain_fp: str
    gear_key: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CacheEntry":
        return cls(
            key=data["key"],
            kind=data["kind"],
            text_id=data["text_id"],
            referenced_entry_fp=dict(data["referenced_entry_fp"]),
            text_fp=data["text_fp"],
            chain_fp=data["chain_fp"],
            gear_key=data["gear_key"],
            payload=dict(data["payload"]),
        )


class Ledger:
    """Persistent product cache plus invalidation/reuse audit trails."""

    def __init__(self) -> None:
        self.entries: dict[str, CacheEntry] = {}
        self.invalidations: list[InvalidationRecord] = []
        self.reuses: list[ReuseRecord] = []

    # ---- key construction -------------------------------------------------
    @staticmethod
    def library_key(chain_fp: str, entry_fps: dict[str, str], gear_key: str) -> str:
        return cache_key(
            "library",
            chain_fp,
            composite_key(entry_fps[k] for k in sorted(entry_fps)),
            gear_key,
        )

    @staticmethod
    def match_key(text_fp: str, library_key: str) -> str:
        return cache_key("match", text_fp, library_key)

    @staticmethod
    def score_key(match_key: str, gear_key: str, config_fp: str) -> str:
        return cache_key("score", match_key, gear_key, config_fp)

    @staticmethod
    def score_key_for(match_key: str, gear_key: str, config_fp: str) -> str:
        # Explicit alias kept so callers cannot accidentally drop gear/width.
        return cache_key("score", match_key, gear_key, config_fp)

    @staticmethod
    def dedup_key(
        text_fp: str, threshold_permille: int, gear_key: str, config_fp: str
    ) -> str:
        return cache_key(
            "dedup", text_fp, str(threshold_permille), gear_key, config_fp
        )

    # ---- cache operations -------------------------------------------------
    def get(self, key: str) -> CacheEntry | None:
        return self.entries.get(key)

    def put(self, entry: CacheEntry) -> None:
        self.entries[entry.key] = entry

    def invalidate(
        self,
        product: str,
        product_key: str,
        text_id: str,
        reason: str,
        changed_fingerprint: str,
    ) -> None:
        self.entries.pop(product_key, None)
        self.invalidations.append(
            InvalidationRecord(
                product=product,
                product_key=product_key,
                text_id=text_id,
                reason=reason,
                changed_fingerprint=changed_fingerprint,
            )
        )

    def record_reuse(
        self, product: str, product_key: str, text_id: str, reason: str
    ) -> None:
        self.reuses.append(
            ReuseRecord(
                product=product,
                product_key=product_key,
                text_id=text_id,
                reason=reason,
            )
        )

    def prune_text(self, text_id: str) -> None:
        """Drop every cached product belonging to a deleted/replaced text."""
        doomed = [k for k, e in self.entries.items() if e.text_id == text_id]
        for key in doomed:
            self.entries.pop(key, None)

    def reset_trails(self) -> None:
        self.invalidations.clear()
        self.reuses.clear()

    # ---- (de)serialisation ------------------------------------------------
    def to_snapshot(self) -> dict[str, Any]:
        return {
            "entries": [entry.to_dict() for entry in self.entries.values()],
            "invalidations": [r.to_dict() for r in self.invalidations],
            "reuses": [r.to_dict() for r in self.reuses],
        }

    @classmethod
    def from_snapshot(cls, data: dict[str, Any]) -> "Ledger":
        ledger = cls()
        ledger.entries = {
            entry["key"]: CacheEntry.from_dict(entry)
            for entry in data.get("entries", [])
        }
        ledger.invalidations = [
            InvalidationRecord(**rec) for rec in data.get("invalidations", [])
        ]
        ledger.reuses = [
            ReuseRecord(**rec) for rec in data.get("reuses", [])
        ]
        return ledger

    def fingerprint(self) -> str:
        return digest(self.to_snapshot())
