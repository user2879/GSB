from __future__ import annotations

import threading
import uuid
from dataclasses import asdict, dataclass
from typing import Any

from .clock import monotonic_ms
from .errors import LeaseError
from .node_protocol import NodeClient
from .spec import NodeSpec


@dataclass(slots=True)
class Lease:
    lease_id: str
    shard_id: str
    node_id: str
    granted_at_ms: int
    expires_at_ms: int
    renewal_count: int
    lease_version: int
    capacity_version: int
    status: str = "ACTIVE"
    released_at_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LeaseManager:
    def __init__(
        self,
        nodes: dict[str, NodeSpec],
        clients: dict[str, NodeClient] | None = None,
        *,
        default_ttl_ms: int = 5_000,
    ) -> None:
        self.nodes = dict(nodes)
        self.clients = dict(clients or {})
        self.default_ttl_ms = default_ttl_ms
        self._leases: dict[str, Lease] = {}
        self._rearrangements: dict[str, int] = {node_id: 0 for node_id in self.nodes}
        self._lock = threading.RLock()

    def _bump_rearrangement(self, node_id: str) -> None:
        self._rearrangements[node_id] = self._rearrangements.get(node_id, 0) + 1

    def rearrangements(self, node_id: str) -> int:
        with self._lock:
            return self._rearrangements.get(node_id, 0)

    def _expire_due_locked(self) -> list[Lease]:
        now = monotonic_ms()
        expired: list[Lease] = []
        for lease in self._leases.values():
            if lease.status == "ACTIVE" and now >= lease.expires_at_ms:
                lease.status = "EXPIRED"
                self._bump_rearrangement(lease.node_id)
                expired.append(lease)
        return expired

    def expire_due(self) -> list[Lease]:
        with self._lock:
            return self._expire_due_locked()

    def acquire(
        self,
        shard_id: str,
        node_id: str,
        capacity_version: int,
        *,
        ttl_ms: int | None = None,
    ) -> Lease:
        if node_id not in self.nodes:
            raise LeaseError(f"unknown node {node_id!r}", object=node_id, field="node_id")
        ttl = int(ttl_ms or self.default_ttl_ms)
        if ttl <= 0:
            raise LeaseError("ttl_ms must be positive", object=shard_id, field="ttl_ms")

        with self._lock:
            self._expire_due_locked()
            node = self.nodes[node_id]
            active = self.active_count(node_id)
            if active >= node.lease_limit:
                raise LeaseError(
                    f"node {node_id} lease limit {node.lease_limit} reached",
                    object=node_id,
                    field="lease_limit",
                    details={"active_leases": active, "lease_limit": node.lease_limit},
                )

            now = monotonic_ms()
            lease = Lease(
                lease_id=f"lease-{uuid.uuid4().hex}",
                shard_id=shard_id,
                node_id=node_id,
                granted_at_ms=now,
                expires_at_ms=now + ttl,
                renewal_count=0,
                lease_version=1,
                capacity_version=capacity_version,
            )
            client = self.clients.get(node_id)
            if client is not None:
                response = client.grant_lease(lease.lease_id, 1, ttl, capacity_version)
                lease.lease_version = int(response.get("lease_version", 1))
            self._leases[lease.lease_id] = lease
            self._bump_rearrangement(node_id)
            return lease

    def get(self, lease_id: str) -> Lease:
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None:
                raise LeaseError("lease not found", object=lease_id, field="lease_id")
            self._expire_due_locked()
            return lease

    def active_count(self, node_id: str) -> int:
        with self._lock:
            self._expire_due_locked()
            return sum(
                1
                for lease in self._leases.values()
                if lease.node_id == node_id and lease.status == "ACTIVE"
            )

    def active_leases(self, node_id: str | None = None) -> list[Lease]:
        with self._lock:
            self._expire_due_locked()
            return [
                lease
                for lease in self._leases.values()
                if lease.status == "ACTIVE" and (node_id is None or lease.node_id == node_id)
            ]

    def renew(self, lease_id: str, expected_version: int, ttl_ms: int | None = None) -> Lease:
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None:
                raise LeaseError("lease not found", object=lease_id, field="lease_id")
            self._expire_due_locked()
            if lease.status != "ACTIVE":
                raise LeaseError(
                    f"cannot renew {lease.status.lower()} lease",
                    object=lease_id,
                    field="expires_at_ms",
                    details={"status": lease.status},
                )
            client = self.clients.get(lease.node_id)
            if client is None and expected_version != lease.lease_version:
                raise LeaseError(
                    "optimistic lease version mismatch before node call",
                    object=lease_id,
                    field="lease_version",
                    details={
                        "expected_version": expected_version,
                        "current_version": lease.lease_version,
                    },
                )

            if client is not None:
                response = client.renew_lease(lease_id, expected_version)
                new_version = int(response["lease_version"])
            else:
                if expected_version != lease.lease_version:
                    raise LeaseError(
                        "optimistic lease version mismatch",
                        object=lease_id,
                        field="lease_version",
                        details={
                            "expected_version": expected_version,
                            "current_version": lease.lease_version,
                        },
                    )
                new_version = lease.lease_version + 1

            ttl = int(ttl_ms or self.default_ttl_ms)
            now = monotonic_ms()
            lease.lease_version = new_version
            lease.renewal_count += 1
            lease.expires_at_ms = now + ttl
            self._bump_rearrangement(lease.node_id)
            return lease

    def release(self, lease_id: str) -> Lease:
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None:
                raise LeaseError("lease not found", object=lease_id, field="lease_id")
            if lease.status == "ACTIVE":
                lease.status = "RELEASED"
                lease.released_at_ms = monotonic_ms()
                self._bump_rearrangement(lease.node_id)
            client = self.clients.get(lease.node_id)
            if client is not None:
                try:
                    client.release_lease(lease_id)
                except Exception:
                    # Local scheduling state remains authoritative if the node
                    # was killed; restart/fence clears node-side state.
                    pass
            return lease

    def revoke_for_capacity(self, node_id: str, new_capacity_version: int) -> list[Lease]:
        with self._lock:
            revoked: list[Lease] = []
            now = monotonic_ms()
            for lease in self._leases.values():
                if lease.node_id == node_id and lease.status == "ACTIVE":
                    lease.status = "REVOKED"
                    lease.released_at_ms = now
                    revoked.append(lease)
                    self._bump_rearrangement(node_id)
            client = self.clients.get(node_id)
            if client is not None:
                try:
                    client.fence_capacity(new_capacity_version)
                except Exception:
                    pass
            return revoked

    def ownership_snapshot(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {lease_id: lease.to_dict() for lease_id, lease in sorted(self._leases.items())}
