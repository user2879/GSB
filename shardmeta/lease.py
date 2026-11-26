"""Shard lease lifecycle: grant, renew, abort and real monotonic expiry.

Expiry is a pure monotonic-timestamp comparison (no sleeps and no artificial
delays). Renewal uses an optimistic lease version number: a stale version is
rejected with E_LEASE and never extends the deadline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .clock import Clock
from .content import mix64
from .errors import LeaseError
from .models import Lease


def make_lease_id(node_id: str, key: str, shard_index: int, granted_ms: int) -> str:
    raw = mix64(
        mix64(granted_ms)
        ^ mix64(shard_index * 7919 + 17)
        ^ int.from_bytes(node_id.encode(), "big")
        ^ int.from_bytes(key.encode(), "big")
    )
    return f"L-{node_id}-{shard_index:04d}-{raw:016x}"


@dataclass
class NodeLeaseState:
    capacity_slots: int
    lease_limit: int
    active_count: int = 0

    def has_capacity(self) -> bool:
        return self.active_count < self.capacity_slots and self.active_count < self.lease_limit


class LeaseManager:
    """In-process lease authority used by both node servers and replay."""

    def __init__(self, node_id: str, capacity_slots: int, lease_limit: int,
                 clock: Optional[Clock] = None) -> None:
        self.node_id = node_id
        self.state = NodeLeaseState(capacity_slots=capacity_slots, lease_limit=lease_limit)
        self._clock = clock or Clock()
        self._leases: dict[str, Lease] = {}

    def now_ms(self) -> int:
        return self._clock.now_ms()

    def _sweep_expired(self, now_ms: int) -> list[Lease]:
        expired: list[Lease] = []
        for lease in self._leases.values():
            if lease.state == "active" and now_ms >= lease.expires_ms:
                self._leases[lease.lease_id] = Lease(
                    lease_id=lease.lease_id, object_key=lease.object_key,
                    shard_index=lease.shard_index, node_id=lease.node_id,
                    granted_ms=lease.granted_ms, expires_ms=lease.expires_ms,
                    renew_count=lease.renew_count, lease_version=lease.lease_version,
                    state="expired",
                )
                self.state.active_count -= 1
                expired.append(self._leases[lease.lease_id])
        return expired

    def sweep(self) -> list[Lease]:
        return self._sweep_expired(self.now_ms())

    def grant(self, key: str, shard_index: int, ttl_ms: int) -> Lease:
        if ttl_ms <= 0:
            raise LeaseError("ttl_ms must be positive", node=self.node_id, key=key,
                             field="ttl_ms")
        now_ms = self.now_ms()
        self._sweep_expired(now_ms)
        if not self.state.has_capacity():
            raise LeaseError(
                "no capacity slot or lease budget available",
                node=self.node_id, key=key, field="capacity_slots",
            )
        granted = now_ms
        lease_id = make_lease_id(self.node_id, key, shard_index, granted)
        lease = Lease(
            lease_id=lease_id, object_key=key, shard_index=shard_index,
            node_id=self.node_id, granted_ms=granted, expires_ms=granted + ttl_ms,
            renew_count=0, lease_version=1, state="active",
        )
        self._leases[lease_id] = lease
        self.state.active_count += 1
        return lease

    def renew(self, lease_id: str, expected_version: int, ttl_ms: int) -> Lease:
        now_ms = self.now_ms()
        self._sweep_expired(now_ms)
        lease = self._leases.get(lease_id)
        if lease is None:
            raise LeaseError("unknown lease_id", node=self.node_id, field="lease_id")
        if lease.state != "active":
            raise LeaseError(f"lease is {lease.state}", node=self.node_id, key=lease.object_key,
                             field="state")
        # Optimistic version check: mismatch rejects and does NOT extend expiry.
        if lease.lease_version != expected_version:
            raise LeaseError(
                f"lease version conflict expected {expected_version} "
                f"actual {lease.lease_version}",
                node=self.node_id, key=lease.object_key, field="lease_version",
            )
        renewed = Lease(
            lease_id=lease.lease_id, object_key=lease.object_key,
            shard_index=lease.shard_index, node_id=lease.node_id,
            granted_ms=lease.granted_ms, expires_ms=now_ms + ttl_ms,
            renew_count=lease.renew_count + 1, lease_version=lease.lease_version + 1,
            state="active",
        )
        self._leases[lease_id] = renewed
        return renewed

    def abort(self, lease_id: str) -> Lease:
        now_ms = self.now_ms()
        self._sweep_expired(now_ms)
        lease = self._leases.get(lease_id)
        if lease is None:
            raise LeaseError("unknown lease_id", node=self.node_id, field="lease_id")
        if lease.state == "active":
            aborted = Lease(
                lease_id=lease.lease_id, object_key=lease.object_key,
                shard_index=lease.shard_index, node_id=lease.node_id,
                granted_ms=lease.granted_ms, expires_ms=lease.expires_ms,
                renew_count=lease.renew_count, lease_version=lease.lease_version,
                state="aborted",
            )
            self._leases[lease_id] = aborted
            self.state.active_count -= 1
            return aborted
        return lease

    def get(self, lease_id: str) -> Optional[Lease]:
        return self._leases.get(lease_id)

    def snapshot(self) -> dict[str, Lease]:
        self._sweep_expired(self.now_ms())
        return dict(self._leases)

    def restore(self, lease: Lease) -> None:
        """Rebuild manager state from committed ledger rows after a crash."""
        self._leases[lease.lease_id] = lease
        if lease.state == "active":
            now_ms = self.now_ms()
            if now_ms >= lease.expires_ms:
                # Real time has passed: a restored active lease is already expired.
                self._leases[lease.lease_id] = Lease(
                    lease_id=lease.lease_id, object_key=lease.object_key,
                    shard_index=lease.shard_index, node_id=lease.node_id,
                    granted_ms=lease.granted_ms, expires_ms=lease.expires_ms,
                    renew_count=lease.renew_count, lease_version=lease.lease_version,
                    state="expired",
                )
            else:
                self.state.active_count += 1


__all__ = ["LeaseManager", "NodeLeaseState", "make_lease_id"]
