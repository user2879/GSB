"""Deterministic sharding algorithms: mod, range, consistent-hash ring.

All routing derives from sha256-based stable hashing of canonical key tuples,
so results never depend on Python hash randomization or dict ordering.
"""

from __future__ import annotations

import bisect
import hashlib
from dataclasses import dataclass

from .canon import canonical_bytes


def stable_hash_int(parts) -> int:
    """Map a list of JSON-safe parts to a non-negative int (deterministic)."""
    raw = hashlib.sha256(canonical_bytes(list(parts))).digest()
    return int.from_bytes(raw[:8], "big")


def key_digest(key_tuple) -> int:
    return stable_hash_int(["key", list(key_tuple)])


@dataclass(frozen=True)
class Hit:
    db: int
    table: int
    reason: str

    def as_dict(self):
        return {"db": self.db, "table": self.table, "reason": self.reason}


def slot_to_db_table(slot: int, tables_per_db: int):
    return divmod(slot, tables_per_db)


def _hit_for_slot(unit, slot, reason):
    db, table = slot_to_db_table(slot, unit.tables_per_db)
    return Hit(db=db, table=table, reason=reason)


def _merge_hits(hits):
    """Dedup by (db, table, reason) and order deterministically."""
    uniq = {}
    for h in hits:
        uniq[(h.db, h.table, h.reason)] = h
    return [uniq[k] for k in sorted(uniq)]


def broadcast_hits(unit, why):
    hits = []
    for slot in range(unit.total_slots):
        hits.append(_hit_for_slot(unit, slot, f"full_scan: {why}"))
    return hits


def route_mod(unit, key_tuples):
    modulus = unit.param("modulus")
    salt = unit.param("salt", "")
    hits = []
    for tup in key_tuples:
        slot = stable_hash_int(["mod", salt, list(tup)]) % modulus
        hits.append(
            _hit_for_slot(unit, slot, f"mod: hash{list(tup)} % {modulus} = slot {slot}")
        )
    return _merge_hits(hits)


def _range_slot(unit, value):
    """Locate the bucket for a point value honoring the boundary policy."""
    boundaries = unit.param("boundaries")
    policy = unit.param("boundary_policy", "left_closed")
    if policy == "left_closed":
        # buckets: (-inf, b0), [b0, b1), ..., [b_{n-1}, +inf)
        return bisect.bisect_right(boundaries, value)
    # right_closed: (-inf, b0], (b0, b1], ...
    return bisect.bisect_left(boundaries, value)


def route_range(unit, key_tuples, intervals):
    boundaries = unit.param("boundaries")
    policy = unit.param("boundary_policy", "left_closed")
    hits = []
    col = unit.shard_keys[0]
    if col in intervals:
        low, high = intervals[col]
        lo_slot = 0 if low is None else _range_slot(unit, low)
        hi_slot = unit.total_slots - 1 if high is None else _range_slot(unit, high)
        for slot in range(lo_slot, hi_slot + 1):
            hits.append(
                _hit_for_slot(
                    unit,
                    slot,
                    f"range: [{low}, {high}] overlaps bucket {slot} ({policy})",
                )
            )
        return _merge_hits(hits)
    for tup in key_tuples:
        value = tup[0]
        slot = _range_slot(unit, value)
        hits.append(
            _hit_for_slot(
                unit, slot, f"range: {value} -> bucket {slot} ({policy}, boundaries={boundaries})"
            )
        )
    return _merge_hits(hits)


class HashRing:
    """Consistent-hash ring over physical slots with virtual nodes."""

    def __init__(self, unit, ring_version: int):
        self.unit_id = unit.unit_id
        self.ring_version = ring_version
        vnodes = unit.param("virtual_nodes")
        points = []
        for slot in range(unit.total_slots):
            for v in range(vnodes):
                point = stable_hash_int(["ring", ring_version, unit.unit_id, slot, v])
                points.append((point, slot, v))
        points.sort()
        self._points = [p[0] for p in points]
        self._slots = [p[1] for p in points]
        self.vnodes = vnodes

    def lookup_slot(self, key_tuple) -> int:
        point = stable_hash_int(["key", self.ring_version, list(key_tuple)])
        idx = bisect.bisect_left(self._points, point)
        if idx == len(self._points):
            idx = 0  # wrap around the ring
        return self._slots[idx]

    def node_count(self) -> int:
        return len(self._points)


def route_consistent_hash(unit, ring: HashRing, key_tuples):
    hits = []
    for tup in key_tuples:
        slot = ring.lookup_slot(tup)
        hits.append(
            _hit_for_slot(
                unit,
                slot,
                f"hash_ring: key{list(tup)} -> slot {slot} "
                f"(ring_v{ring.ring_version}, vnodes={ring.vnodes})",
            )
        )
    return _merge_hits(hits)
