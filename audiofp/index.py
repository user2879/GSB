"""Stable in-memory reference fingerprint index with deterministic buckets."""

from __future__ import annotations

from collections import defaultdict

from .models import ReferenceEntry, TrackFingerprint


class ReferenceIndex:
    """Hash index keyed by (anchor band, target band, frame delta)."""

    def __init__(self, bucket_capacity: int) -> None:
        if bucket_capacity <= 0:
            raise ValueError("bucket_capacity must be positive")
        self.bucket_capacity = bucket_capacity
        self._buckets: dict[tuple[int, int, int, int, int, int, int], list[ReferenceEntry]] = defaultdict(list)
        self._fingerprints: dict[str, TrackFingerprint] = {}
        self._reference_order: list[str] = []
        self.dropped_entries = 0
        self.entries_added = 0

    @property
    def reference_ids(self) -> tuple[str, ...]:
        return tuple(self._reference_order)

    @property
    def bucket_count(self) -> int:
        return len(self._buckets)

    @property
    def entry_count(self) -> int:
        return sum(len(bucket) for bucket in self._buckets.values())

    def add_fingerprint(self, fingerprint: TrackFingerprint, *, reference_order: int | None = None) -> None:
        if fingerprint.role != "reference":
            raise ValueError("only reference fingerprints can be indexed")
        order = len(self._reference_order) if reference_order is None else reference_order
        self._reference_order.append(fingerprint.track_id)
        self._fingerprints[fingerprint.track_id] = fingerprint
        for item in fingerprint.items:
            bucket = self._buckets[item.key]
            if len(bucket) >= self.bucket_capacity:
                self.dropped_entries += 1
                continue
            bucket.append(ReferenceEntry(fingerprint.track_id, order, item))
            self.entries_added += 1

    def lookup(self, key: tuple[int, int, int, int, int, int, int]) -> tuple[ReferenceEntry, ...]:
        return tuple(self._buckets.get(key, ()))

    def get(self, reference_id: str) -> TrackFingerprint:
        return self._fingerprints[reference_id]

    def stats(self) -> dict[str, int]:
        return {
            "bucket_count": self.bucket_count,
            "bucket_capacity": self.bucket_capacity,
            "entry_count": self.entry_count,
            "entries_added": self.entries_added,
            "dropped_entries": self.dropped_entries,
        }
