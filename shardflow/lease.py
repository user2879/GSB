"""Lease value object and deterministic lease id construction."""

from dataclasses import dataclass

from .errors import LeaseError
from .util import sha256_hex


@dataclass
class Lease:
    lease_id: str
    shard_id: str
    node_id: str
    granted_ms: int
    expires_ms: int
    renewals: int
    lease_version: int
    attempt: int
    renewed_ms: int = 0

    def to_dict(self):
        return {
            "lease_id": self.lease_id,
            "shard_id": self.shard_id,
            "node_id": self.node_id,
            "granted_ms": self.granted_ms,
            "expires_ms": self.expires_ms,
            "renewals": self.renewals,
            "lease_version": self.lease_version,
            "attempt": self.attempt,
            "renewed_ms": self.renewed_ms,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            lease_id=data["lease_id"],
            shard_id=data["shard_id"],
            node_id=data["node_id"],
            granted_ms=data["granted_ms"],
            expires_ms=data["expires_ms"],
            renewals=data["renewals"],
            lease_version=data["lease_version"],
            attempt=data["attempt"],
            renewed_ms=data.get("renewed_ms", 0),
        )

    def is_expired(self, monotonic_ms):
        return monotonic_ms >= self.expires_ms


def make_lease_id(shard_id, attempt, granted_ms):
    digest = sha256_hex(
        {"shard_id": shard_id, "attempt": attempt, "granted_ms": granted_ms}
    )[:16]
    return "lease-%s-a%d-%s" % (shard_id, attempt, digest)


def grant_lease(shard_id, node_id, attempt, granted_ms, ttl_ms):
    return Lease(
        lease_id=make_lease_id(shard_id, attempt, granted_ms),
        shard_id=shard_id,
        node_id=node_id,
        granted_ms=granted_ms,
        expires_ms=granted_ms + ttl_ms,
        renewals=0,
        lease_version=1,
        attempt=attempt,
        renewed_ms=granted_ms,
    )


def check_renew(lease, expected_version, max_renewals, now, ttl_ms):
    """Pure optimistic renewal rule used by engine and replay alike."""
    if lease.lease_version != expected_version:
        raise LeaseError(
            "lease version mismatch: expected %d, got %d"
            % (lease.lease_version, expected_version),
            object_id=lease.shard_id,
            field="lease_version",
        )
    if lease.renewals >= max_renewals:
        raise LeaseError(
            "renewal limit %d reached" % max_renewals,
            object_id=lease.shard_id,
            field="renewals",
        )
    lease.renewals += 1
    lease.lease_version += 1
    lease.expires_ms = now + ttl_ms
    lease.renewed_ms = now
    return lease
