from __future__ import annotations

import threading
import time
from typing import Any

from .accounting import deterministic_lease_id
from .errors import LeaseError
from .spec import SessionSpec


def monotonic_ms() -> int:
    return time.monotonic_ns() // 1_000_000


def new_lease(spec: SessionSpec, shard_id: str) -> dict[str, Any]:
    shard = spec.shard(shard_id)
    now = monotonic_ms()
    lease_id = deterministic_lease_id(spec, shard)
    return {
        "lease_id": lease_id,
        "shard_id": shard_id,
        "node_id": shard.node_id,
        "granted_monotonic_ms": now,
        "expires_monotonic_ms": now + spec.lease_ttl_ms,
        "ttl_ms": spec.lease_ttl_ms,
        "renew_count": 0,
        "lease_version": 1,
        "capacity_version": spec.node(shard.node_id).capacity_version,
        "state": "active",
    }


def ownership_view(lease: dict[str, Any]) -> dict[str, Any]:
    return {
        "lease_id": lease["lease_id"],
        "shard_id": lease["shard_id"],
        "node_id": lease["node_id"],
        "capacity_version": lease["capacity_version"],
        "lease_version": lease["lease_version"],
        "renew_count": lease["renew_count"],
        "owner": lease["node_id"],
    }


class LeaseRegistry:
    def __init__(self, spec: SessionSpec) -> None:
        self.spec = spec
        self.leases: dict[str, dict[str, Any]] = {}
        self.lock = threading.RLock()

    def add(self, lease: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self.leases[lease["lease_id"]] = lease
            return dict(lease)

    def active_count(self, node_id: str) -> int:
        with self.lock:
            return sum(1 for lease in self.leases.values() if lease["node_id"] == node_id and lease["state"] == "active")

    def terminalize(self, lease_id: str, state: str, reason: str) -> dict[str, Any] | None:
        with self.lock:
            lease = self.leases.get(lease_id)
            if lease is None or lease["state"] != "active":
                return None
            lease["state"] = state
            lease["terminal_reason"] = reason
            lease["terminal_monotonic_ms"] = monotonic_ms()
            return dict(lease)

    def reap_expired(self) -> list[dict[str, Any]]:
        now = monotonic_ms()
        terminalized: list[dict[str, Any]] = []
        with self.lock:
            for lease in list(self.leases.values()):
                if lease["state"] == "active" and now >= int(lease["expires_monotonic_ms"]):
                    changed = self.terminalize(
                        lease["lease_id"],
                        "expired",
                        "monotonic_ttl_reached",
                    )
                    if changed is not None:
                        terminalized.append(changed)
        return terminalized

    def void_capacity_version(self, node_id: str, new_version: int | None = None) -> list[dict[str, Any]]:
        with self.lock:
            node = self.spec.node(node_id)
            target_version = int(node.capacity_version if new_version is None else new_version)
            voided: list[dict[str, Any]] = []
            for lease in list(self.leases.values()):
                if (
                    lease["node_id"] == node_id
                    and lease["state"] == "active"
                    and int(lease["capacity_version"]) < target_version
                ):
                    changed = self.terminalize(
                        lease["lease_id"],
                        "version_void",
                        f"capacity_version_advanced_to_{target_version}",
                    )
                    if changed is not None:
                        voided.append(changed)
            return voided

    def renew_optimistic(self, cluster: Any, lease: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if lease["state"] != "active":
                raise LeaseError(
                    "cannot renew a non-active lease",
                    field="lease.state",
                    details={"state": lease["state"], "lease_id": lease["lease_id"]},
                )
            if int(lease["renew_count"]) >= self.spec.lease_max_renewals:
                raise LeaseError(
                    "lease renewal limit reached",
                    field="lease.renew_count",
                    details={
                        "lease_id": lease["lease_id"],
                        "renew_count": lease["renew_count"],
                        "max_renewals": self.spec.lease_max_renewals,
                    },
                )
            expected_version = int(lease["lease_version"])
            response = cluster.request(
                lease["node_id"],
                {
                    "op": "lease_renew",
                    "node_id": lease["node_id"],
                    "lease_id": lease["lease_id"],
                    "expected_version": expected_version,
                },
                timeout_ms=self.spec.socket_timeout_ms,
            )
            node_lease = response["lease"]
            lease["lease_version"] = int(node_lease["lease_version"])
            lease["renew_count"] = int(node_lease["renew_count"])
            now = monotonic_ms()
            lease["granted_monotonic_ms"] = now
            lease["expires_monotonic_ms"] = now + self.spec.lease_ttl_ms
            return dict(lease)
