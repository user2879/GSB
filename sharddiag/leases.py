"""Deterministic lease identities and monotonic-clock lease state."""

from __future__ import annotations

import threading
import time
from typing import Any

from .errors import LeaseError


def monotonic_ms() -> int:
    return time.monotonic_ns() // 1_000_000


def lease_id_for(shard_id: str, capacity_version: int) -> str:
    return f"lease-{shard_id}-gen{int(capacity_version)}"


class LeaseManager:
    """In-process authority for active resource equity.

    Status transitions are compare-and-set-like under one lock.  Expiry and
    capacity revocation are mutually exclusive: the first transition moves a
    lease out of ``active`` and the other path can no longer reclaim it.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.leases: dict[str, dict[str, Any]] = {}
        self.active_by_shard: dict[str, str] = {}

    def grant(
        self,
        *,
        shard_id: str,
        node_id: str,
        capacity_version: int,
        resource_units: int,
        lease_limit: int,
        ttl_ms: int,
        now_ms: int | None = None,
    ) -> dict[str, Any]:
        now = monotonic_ms() if now_ms is None else int(now_ms)
        lease_id = lease_id_for(shard_id, capacity_version)
        with self._lock:
            if shard_id in self.active_by_shard:
                raise LeaseError(
                    "shard already holds an active lease",
                    shard_id=shard_id,
                    lease_id=self.active_by_shard[shard_id],
                )
            active_on_node = sum(
                1
                for lease in self.leases.values()
                if lease["status"] == "active" and lease["node_id"] == node_id
            )
            if active_on_node >= lease_limit:
                raise LeaseError(
                    "node active lease count exceeds lease_limit",
                    node_id=node_id,
                    lease_limit=lease_limit,
                    active_leases=active_on_node,
                )
            lease = {
                "lease_id": lease_id,
                "shard_id": shard_id,
                "node_id": node_id,
                "capacity_version": int(capacity_version),
                "resource_units": int(resource_units),
                "granted_monotonic_ms": now,
                "expires_monotonic_ms": now + int(ttl_ms),
                "renew_count": 0,
                "lease_version": 1,
                "status": "active",
                "reclaim_reason": None,
            }
            self.leases[lease_id] = lease
            self.active_by_shard[shard_id] = lease_id
            return dict(lease)

    def get(self, lease_id: str) -> dict[str, Any] | None:
        with self._lock:
            lease = self.leases.get(lease_id)
            return dict(lease) if lease else None

    def active_leases_on_node(self, node_id: str) -> int:
        with self._lock:
            return sum(
                1
                for lease in self.leases.values()
                if lease["status"] == "active" and lease["node_id"] == node_id
            )

    def used_units_on_node(self, node_id: str) -> int:
        with self._lock:
            return sum(
                int(lease["resource_units"])
                for lease in self.leases.values()
                if lease["status"] == "active" and lease["node_id"] == node_id
            )

    def used_units(self) -> int:
        with self._lock:
            return sum(int(lease["resource_units"]) for lease in self.leases.values() if lease["status"] == "active")

    def reconcile_expired(self, *, now_ms: int | None = None) -> list[dict[str, Any]]:
        now = monotonic_ms() if now_ms is None else int(now_ms)
        changed: list[dict[str, Any]] = []
        with self._lock:
            for lease in self.leases.values():
                if lease["status"] == "active" and now >= int(lease["expires_monotonic_ms"]):
                    lease["status"] = "expired"
                    lease["reclaim_reason"] = "monotonic_ttl_expiry"
                    self.active_by_shard.pop(lease["shard_id"], None)
                    changed.append(dict(lease))
        return changed

    def revoke_for_capacity(self, new_capacity_version: int) -> list[dict[str, Any]]:
        changed: list[dict[str, Any]] = []
        with self._lock:
            for lease in self.leases.values():
                if (
                    lease["status"] == "active"
                    and int(lease["capacity_version"]) < int(new_capacity_version)
                ):
                    lease["status"] = "revoked_capacity_generation"
                    lease["reclaim_reason"] = "capacity_generation_advance"
                    lease["revoked_to_version"] = int(new_capacity_version)
                    self.active_by_shard.pop(lease["shard_id"], None)
                    changed.append(dict(lease))
        return changed

    def release(self, lease_id: str, *, reason: str = "completed") -> dict[str, Any] | None:
        with self._lock:
            lease = self.leases.get(lease_id)
            if lease is None:
                return None
            if lease["status"] == "active":
                lease["status"] = "released"
                lease["reclaim_reason"] = reason
            self.active_by_shard.pop(lease["shard_id"], None)
            return dict(lease)

    def renew_local(
        self,
        lease_id: str,
        expected_version: int,
        *,
        max_renewals: int,
        ttl_ms: int,
        now_ms: int | None = None,
    ) -> dict[str, Any]:
        now = monotonic_ms() if now_ms is None else int(now_ms)
        with self._lock:
            lease = self.leases.get(lease_id)
            if lease is None or lease["status"] != "active":
                raise LeaseError(
                    "lease is not active and cannot be renewed",
                    lease_id=lease_id,
                    basis_generation=int((lease or {}).get("lease_version", 0)),
                )
            if int(lease["lease_version"]) != int(expected_version):
                raise LeaseError(
                    "optimistic lease version mismatch",
                    lease_id=lease_id,
                    expected_version=int(expected_version),
                    actual_version=int(lease["lease_version"]),
                    basis_generation=int(lease["lease_version"]),
                )
            if int(lease["renew_count"]) >= int(max_renewals):
                raise LeaseError(
                    "renewal count exceeds declared max_renewals",
                    lease_id=lease_id,
                    max_renewals=int(max_renewals),
                    renew_count=int(lease["renew_count"]),
                    basis_generation=int(lease["lease_version"]),
                )
            if now >= int(lease["expires_monotonic_ms"]):
                lease["status"] = "expired"
                lease["reclaim_reason"] = "renew_after_expiry"
                self.active_by_shard.pop(lease["shard_id"], None)
                raise LeaseError(
                    "lease expired before renewal",
                    lease_id=lease_id,
                    basis_generation=int(lease["lease_version"]),
                    now_monotonic_ms=now,
                    expires_monotonic_ms=lease["expires_monotonic_ms"],
                )
            lease["renew_count"] += 1
            lease["lease_version"] += 1
            lease["expires_monotonic_ms"] = now + int(ttl_ms)
            return dict(lease)

    def replace_recovered_leases(self, leases: list[dict[str, Any]]) -> None:
        with self._lock:
            self.leases.clear()
            self.active_by_shard.clear()
            for lease in leases:
                record = dict(lease)
                self.leases[record["lease_id"]] = record
                if record.get("status") == "active":
                    self.active_by_shard[record["shard_id"]] = record["lease_id"]

    def assert_single_holder(self) -> None:
        with self._lock:
            shard_holders: dict[str, str] = {}
            node_shard_pairs: set[tuple[str, str]] = set()
            for lease in self.leases.values():
                if lease["status"] != "active":
                    continue
                shard_id = lease["shard_id"]
                holder = lease["lease_id"]
                if shard_id in shard_holders and shard_holders[shard_id] != holder:
                    raise LeaseError("shard held by two active leases", shard_id=shard_id)
                shard_holders[shard_id] = holder
                pair = (lease["node_id"], shard_id)
                if pair in node_shard_pairs:
                    raise LeaseError("same node/shard equity held twice", pair=pair)
                node_shard_pairs.add(pair)
