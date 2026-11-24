from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import socketserver
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .canonical import canonical_bytes, canonical_json
from .content import generate_content
from .errors import CacheLeaseError
from .spec import Config, load_config


PROTOCOL_VERSION = 1


def mono_ms() -> int:
    return time.monotonic_ns() // 1_000_000


def deterministic_id(prefix: str, *values: Any) -> str:
    material = canonical_bytes([prefix, *values])
    return f"{prefix}-{hashlib.sha256(material).hexdigest()[:28]}"


def watermark_payload(
    used: int,
    capacity: int,
    active_leases: int,
    lease_limit: int,
    scale: int,
    reserved_leases: int = 0,
) -> dict[str, int]:
    used = max(0, min(used, capacity))
    free = max(0, capacity - used)
    ratio_units = (used * scale + capacity // 2) // capacity
    error_upper = (scale + 2 * capacity - 1) // (2 * capacity)
    return {
        "used_slots": used,
        "free_slots": free,
        "active_leases": active_leases,
        "lease_remaining": max(0, lease_limit - active_leases - reserved_leases),
        "watermark_units": ratio_units,
        "quantization_error_upper": error_upper,
    }


@dataclass
class LeaseRecord:
    lease_id: str
    artifact_key: str
    digest: str
    quota_version: int
    apply_seq: int
    apply_ordinal: int
    created_ms: int
    expires_at_ms: int


@dataclass
class EntryRecord:
    artifact_key: str
    digest: str
    quota_version: int
    shard_count: int
    shard_bytes: int
    shards: tuple[bytes, ...]
    lease_ids: set[str] = field(default_factory=set)


@dataclass
class ReservationRecord:
    reservation_id: str
    artifact_key: str
    lease_ms: int
    apply_seq: int
    apply_ordinal: int
    quota_version: int
    created_ms: int
    grace_deadline_ms: int
    slot_held: bool = True


@dataclass
class Disposition:
    lease_id: str
    artifact_key: str
    status: str
    reason: str
    new_lease_id: str | None = None


class NodeState:
    def __init__(self, config: Config, node_id: str) -> None:
        self.config = config
        self.spec = config.node_by_id[node_id]
        self.node_id = node_id
        self.capacity_slots = self.spec.capacity_slots
        self.lease_limit = self.spec.lease_limit
        self.quota_version = self.spec.quota_version
        self.scale = config.scale
        self.lock = threading.RLock()
        self.entries: dict[str, EntryRecord] = {}
        self.leases: dict[str, LeaseRecord] = {}
        self.reservations: dict[str, ReservationRecord] = {}
        self.expired_count = 0
        self.auto_rollback_count = 0
        self.revoked_count = 0
        self.renewed_count = 0
        self.in_flight = 0
        self.max_in_flight = 0
        self.request_count = 0
        self.started_ms = mono_ms()

    def _entry_holds(self, artifact_key: str) -> bool:
        if artifact_key in self.entries:
            return True
        return any(res.artifact_key == artifact_key for res in self.reservations.values())

    def _cleanup_entry_locked(self, artifact_key: str) -> None:
        entry = self.entries.get(artifact_key)
        if entry is None:
            return
        if entry.lease_ids:
            return
        if any(res.artifact_key == artifact_key for res in self.reservations.values()):
            return
        del self.entries[artifact_key]

    def expire_locked(self, now: int | None = None) -> list[dict[str, Any]]:
        now = mono_ms() if now is None else now
        actions: list[dict[str, Any]] = []
        expired_ids = [lease_id for lease_id, lease in self.leases.items() if lease.expires_at_ms <= now]
        for lease_id in sorted(expired_ids):
            lease = self.leases.pop(lease_id)
            entry = self.entries.get(lease.artifact_key)
            if entry is not None:
                entry.lease_ids.discard(lease_id)
                self._cleanup_entry_locked(lease.artifact_key)
            self.expired_count += 1
            actions.append(
                {
                    "kind": "lease_expired",
                    "lease_id": lease_id,
                    "artifact_key": lease.artifact_key,
                    "apply_seq": lease.apply_seq,
                    "apply_ordinal": lease.apply_ordinal,
                    "at_ms": now,
                }
            )

        expired_reservations = [
            reservation
            for reservation in self.reservations.values()
            if reservation.grace_deadline_ms <= now
        ]
        for reservation in sorted(expired_reservations, key=lambda item: item.reservation_id):
            del self.reservations[reservation.reservation_id]
            self.auto_rollback_count += 1
            self._cleanup_entry_locked(reservation.artifact_key)
            actions.append(
                {
                    "kind": "reservation_auto_rollback",
                    "reservation_id": reservation.reservation_id,
                    "artifact_key": reservation.artifact_key,
                    "apply_seq": reservation.apply_seq,
                    "apply_ordinal": reservation.apply_ordinal,
                    "reason": "grace_expired",
                    "at_ms": now,
                }
            )
        return actions

    def watermark_locked(self) -> dict[str, int]:
        used = len(self.entries) + sum(1 for reservation in self.reservations.values() if reservation.slot_held)
        active = len(self.leases)
        return watermark_payload(
            used,
            self.capacity_slots,
            active,
            self.lease_limit,
            self.scale,
            len(self.reservations),
        )

    def status_locked(self) -> dict[str, Any]:
        now = mono_ms()
        return {
            "node_id": self.node_id,
            "quota_version": self.quota_version,
            "capacity_slots": self.capacity_slots,
            "lease_limit": self.lease_limit,
            "scale": self.scale,
            "watermark": self.watermark_locked(),
            "entries": [
                {
                    "artifact_key": key,
                    "digest": entry.digest,
                    "quota_version": entry.quota_version,
                    "lease_count": len(entry.lease_ids),
                    "shards": entry.shard_count,
                    "shard_bytes": entry.shard_bytes,
                    "content_bytes": sum(len(shard) for shard in entry.shards),
                }
                for key, entry in sorted(self.entries.items())
            ],
            "leases": [
                {
                    "lease_id": lease.lease_id,
                    "artifact_key": lease.artifact_key,
                    "digest": lease.digest,
                    "quota_version": lease.quota_version,
                    "apply_seq": lease.apply_seq,
                    "remaining_ms": max(0, lease.expires_at_ms - now),
                }
                for lease in sorted(self.leases.values(), key=lambda item: item.lease_id)
            ],
            "reservations": [
                {
                    "reservation_id": reservation.reservation_id,
                    "artifact_key": reservation.artifact_key,
                    "apply_seq": reservation.apply_seq,
                    "quota_version": reservation.quota_version,
                    "remaining_grace_ms": max(0, reservation.grace_deadline_ms - now),
                }
                for reservation in sorted(self.reservations.values(), key=lambda item: item.reservation_id)
            ],
            "counters": {
                "expired": self.expired_count,
                "auto_rollback": self.auto_rollback_count,
                "revoked": self.revoked_count,
                "renewed": self.renewed_count,
                "requests": self.request_count,
                "in_flight": self.in_flight,
                "max_in_flight": self.max_in_flight,
            },
            "monotonic_ms": now,
        }

    def status(self) -> dict[str, Any]:
        with self.lock:
            self.expire_locked()
            return self.status_locked()

    def _reservation_id(
        self, artifact_key: str, apply_seq: int, apply_ordinal: int, lease_ms: int
    ) -> str:
        return deterministic_id(
            "resv",
            self.node_id,
            artifact_key,
            apply_seq,
            apply_ordinal,
            lease_ms,
            self.quota_version,
        )

    def _lease_id(self, reservation: ReservationRecord) -> str:
        return deterministic_id(
            "lease",
            self.node_id,
            reservation.artifact_key,
            reservation.apply_seq,
            reservation.apply_ordinal,
            reservation.reservation_id,
            self.quota_version,
        )

    def reserve(
        self,
        artifact_key: str,
        lease_ms: int,
        apply_seq: int,
        grace_ms: int,
        expected_quota_version: int | None,
        apply_ordinal: int | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            self.expire_locked()
            if expected_quota_version is not None and expected_quota_version != self.quota_version:
                return self.error("E_QUOTA_VERSION", "quota version changed", {
                    "expected": expected_quota_version,
                    "actual": self.quota_version,
                })
            if artifact_key not in self.config.artifact_by_key:
                return self.error("E_ARTIFACT", "unknown artifact key")
            if not isinstance(lease_ms, int) or isinstance(lease_ms, bool) or lease_ms <= 0:
                return self.error("E_ARGUMENT", "lease_ms must be a positive integer")
            if not isinstance(apply_seq, int) or isinstance(apply_seq, bool) or apply_seq <= 0:
                return self.error("E_ARGUMENT", "apply_seq must be a positive integer")
            if apply_ordinal is None:
                apply_ordinal = apply_seq
            if not isinstance(apply_ordinal, int) or isinstance(apply_ordinal, bool) or apply_ordinal <= 0:
                return self.error("E_ARGUMENT", "apply_ordinal must be a positive integer")
            if not isinstance(grace_ms, int) or isinstance(grace_ms, bool) or grace_ms <= 0:
                return self.error("E_ARGUMENT", "grace_ms must be a positive integer")

            active_plus_reserved = len(self.leases) + len(self.reservations)
            if active_plus_reserved >= self.lease_limit:
                return self.error("E_LEASE_LIMIT", "per-node lease limit reached", {
                    "active_leases": len(self.leases),
                    "reservations": len(self.reservations),
                    "lease_limit": self.lease_limit,
                })
            used = len(self.entries) + sum(1 for item in self.reservations.values() if item.slot_held)
            if used >= self.capacity_slots:
                return self.error("E_CAPACITY", "capacity watermark reached", {
                    "used_slots": used,
                    "capacity_slots": self.capacity_slots,
                })

            reservation_id = self._reservation_id(artifact_key, apply_seq, apply_ordinal, lease_ms)
            existing = self.reservations.get(reservation_id)
            if existing is not None:
                return self.ok({
                    "reservation_id": reservation_id,
                    "slot_held": existing.slot_held,
                    "grace_deadline_ms": existing.grace_deadline_ms,
                    "quota_version": self.quota_version,
                    "watermark": self.watermark_locked(),
                    "idempotent": True,
                })

            now = mono_ms()
            reservation = ReservationRecord(
                reservation_id=reservation_id,
                artifact_key=artifact_key,
                lease_ms=lease_ms,
                apply_seq=apply_seq,
                apply_ordinal=apply_ordinal,
                quota_version=self.quota_version,
                created_ms=now,
                grace_deadline_ms=now + grace_ms,
            )
            self.reservations[reservation_id] = reservation
            return self.ok({
                "reservation_id": reservation_id,
                "slot_held": True,
                "grace_deadline_ms": reservation.grace_deadline_ms,
                "quota_version": self.quota_version,
                "watermark": self.watermark_locked(),
                "idempotent": False,
            })

    def commit(self, reservation_id: str) -> dict[str, Any]:
        with self.lock:
            now = mono_ms()
            self.expire_locked(now)
            reservation = self.reservations.get(reservation_id)
            if reservation is None:
                return self.error("E_EXPIRED", "reservation was rolled back, committed or expired", {
                    "reservation_id": reservation_id,
                })
            if reservation.quota_version != self.quota_version:
                del self.reservations[reservation_id]
                self._cleanup_entry_locked(reservation.artifact_key)
                return self.error("E_STALE", "reservation belongs to a stale quota version")

            artifact_spec = self.config.artifact_by_key[reservation.artifact_key]
            content = generate_content(
                self.node_id,
                reservation.artifact_key,
                self.spec.seed,
                artifact_spec.shards,
                artifact_spec.shard_bytes,
            )
            entry = self.entries.get(reservation.artifact_key)
            if entry is None:
                entry = EntryRecord(
                    artifact_key=reservation.artifact_key,
                    digest=content.digest,
                    quota_version=self.quota_version,
                    shard_count=artifact_spec.shards,
                    shard_bytes=artifact_spec.shard_bytes,
                    shards=content.shards,
                )
                self.entries[reservation.artifact_key] = entry
            elif entry.digest != content.digest:
                del self.reservations[reservation_id]
                self._cleanup_entry_locked(reservation.artifact_key)
                return self.error("E_DIGEST", "deterministic digest changed unexpectedly")

            del self.reservations[reservation_id]
            lease_id = self._lease_id(reservation)
            lease = LeaseRecord(
                lease_id=lease_id,
                artifact_key=reservation.artifact_key,
                digest=entry.digest,
                quota_version=self.quota_version,
                apply_seq=reservation.apply_seq,
                apply_ordinal=reservation.apply_ordinal,
                created_ms=now,
                expires_at_ms=now + reservation.lease_ms,
            )
            self.leases[lease_id] = lease
            entry.lease_ids.add(lease_id)
            return self.ok({
                "reservation_id": reservation_id,
                "lease_id": lease_id,
                "artifact_key": lease.artifact_key,
                "digest": lease.digest,
                "quota_version": lease.quota_version,
                "lease_expires_mono_ms": lease.expires_at_ms,
                "shards": artifact_spec.shards,
                "shard_bytes": artifact_spec.shard_bytes,
                "content_bytes": sum(len(shard) for shard in entry.shards),
                "watermark": self.watermark_locked(),
            })

    def rollback(self, reservation_id: str, reason: str) -> dict[str, Any]:
        with self.lock:
            self.expire_locked()
            reservation = self.reservations.pop(reservation_id, None)
            if reservation is None:
                return self.ok({
                    "reservation_id": reservation_id,
                    "rolled_back": False,
                    "reason": "not_active",
                    "watermark": self.watermark_locked(),
                })
            self._cleanup_entry_locked(reservation.artifact_key)
            return self.ok({
                "reservation_id": reservation_id,
                "rolled_back": True,
                "reason": reason,
                "artifact_key": reservation.artifact_key,
                "apply_seq": reservation.apply_seq,
                "watermark": self.watermark_locked(),
            })

    def read(
        self,
        artifact_key: str,
        lease_ms: int,
        apply_seq: int,
        apply_ordinal: int | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            now = mono_ms()
            self.expire_locked(now)
            entry = self.entries.get(artifact_key)
            if entry is None:
                return self.error("E_NOT_FOUND", "artifact is not present on this node")
            if len(self.leases) + len(self.reservations) >= self.lease_limit:
                return self.error("E_LEASE_LIMIT", "per-node lease limit reached")
            if not isinstance(lease_ms, int) or isinstance(lease_ms, bool) or lease_ms <= 0:
                return self.error("E_ARGUMENT", "lease_ms must be a positive integer")
            if not isinstance(apply_seq, int) or isinstance(apply_seq, bool) or apply_seq <= 0:
                return self.error("E_ARGUMENT", "apply_seq must be a positive integer")
            if apply_ordinal is None:
                apply_ordinal = apply_seq
            if not isinstance(apply_ordinal, int) or isinstance(apply_ordinal, bool) or apply_ordinal <= 0:
                return self.error("E_ARGUMENT", "apply_ordinal must be a positive integer")
            lease_id = deterministic_id(
                "read",
                self.node_id,
                artifact_key,
                apply_seq,
                apply_ordinal,
                self.quota_version,
            )
            if lease_id in self.leases:
                return self.error("E_DUPLICATE", "read lease already exists for this application sequence")
            lease = LeaseRecord(
                lease_id=lease_id,
                artifact_key=artifact_key,
                digest=entry.digest,
                quota_version=self.quota_version,
                apply_seq=apply_seq,
                apply_ordinal=apply_ordinal,
                created_ms=now,
                expires_at_ms=now + lease_ms,
            )
            self.leases[lease_id] = lease
            entry.lease_ids.add(lease_id)
            return self.ok({
                "lease_id": lease_id,
                "artifact_key": artifact_key,
                "digest": lease.digest,
                "quota_version": lease.quota_version,
                "lease_expires_mono_ms": lease.expires_at_ms,
                "watermark": self.watermark_locked(),
                "shards": entry.shard_count,
                "content_bytes": sum(len(shard) for shard in entry.shards),
            })

    def release(self, lease_id: str) -> dict[str, Any]:
        with self.lock:
            self.expire_locked()
            lease = self.leases.pop(lease_id, None)
            if lease is None:
                return self.error("E_NOT_FOUND", "lease is not active")
            entry = self.entries.get(lease.artifact_key)
            if entry is not None:
                entry.lease_ids.discard(lease_id)
                self._cleanup_entry_locked(lease.artifact_key)
            return self.ok({
                "lease_id": lease_id,
                "artifact_key": lease.artifact_key,
                "released": True,
                "freed_slot": lease.artifact_key not in self.entries,
                "watermark": self.watermark_locked(),
            })

    def release_artifact(self, artifact_key: str) -> dict[str, Any]:
        with self.lock:
            self.expire_locked()
            lease_ids = sorted(
                lease_id for lease_id, lease in self.leases.items() if lease.artifact_key == artifact_key
            )
            for lease_id in lease_ids:
                self.leases.pop(lease_id, None)
            entry = self.entries.get(artifact_key)
            if entry is not None:
                entry.lease_ids.clear()
                self._cleanup_entry_locked(artifact_key)
            return self.ok({
                "artifact_key": artifact_key,
                "released_lease_ids": lease_ids,
                "freed_slot": artifact_key not in self.entries,
                "watermark": self.watermark_locked(),
            })

    def set_quota(
        self,
        quota_version: int,
        capacity_slots: int | None,
        lease_limit: int | None,
        policy: str,
    ) -> dict[str, Any]:
        with self.lock:
            self.expire_locked()
            if not isinstance(quota_version, int) or isinstance(quota_version, bool) or quota_version <= self.quota_version:
                return self.error("E_QUOTA_VERSION", "quota_version must increase")
            if policy not in {"renew", "revoke"}:
                return self.error("E_ARGUMENT", "policy must be renew or revoke")
            new_capacity = self.capacity_slots if capacity_slots is None else capacity_slots
            new_limit = self.lease_limit if lease_limit is None else lease_limit
            if not isinstance(new_capacity, int) or isinstance(new_capacity, bool) or new_capacity <= 0:
                return self.error("E_ARGUMENT", "capacity_slots must be positive")
            if not isinstance(new_limit, int) or isinstance(new_limit, bool) or new_limit <= 0:
                return self.error("E_ARGUMENT", "lease_limit must be positive")

            old_version = self.quota_version
            old_capacity = self.capacity_slots
            old_limit = self.lease_limit
            reservation_dispositions: list[dict[str, Any]] = []
            for reservation_id in sorted(list(self.reservations)):
                reservation = self.reservations.pop(reservation_id)
                self.auto_rollback_count += 1
                self._cleanup_entry_locked(reservation.artifact_key)
                reservation_dispositions.append({
                    "reservation_id": reservation_id,
                    "artifact_key": reservation.artifact_key,
                    "apply_seq": reservation.apply_seq,
                    "status": "rolled_back",
                    "reason": f"quota_version_{old_version}_became_stale",
                })

            lease_dispositions: list[Disposition] = []
            for lease_id in sorted(list(self.leases)):
                lease = self.leases[lease_id]
                if policy == "revoke":
                    old_lease_id = lease.lease_id
                    entry = self.entries.get(lease.artifact_key)
                    if entry is not None:
                        entry.lease_ids.discard(lease_id)
                    del self.leases[lease_id]
                    self.revoked_count += 1
                    self._cleanup_entry_locked(lease.artifact_key)
                    lease_dispositions.append(Disposition(old_lease_id, lease.artifact_key, "revoked", "policy_revoke"))
                else:
                    old_lease_id = lease.lease_id
                    new_lease_id = deterministic_id(
                        "lease", self.node_id, lease.artifact_key, lease.apply_seq, old_lease_id, quota_version
                    )
                    entry = self.entries.get(lease.artifact_key)
                    if entry is None:
                        # A lease should always have an entry; defensive handling records the anomaly.
                        del self.leases[lease_id]
                        lease_dispositions.append(Disposition(old_lease_id, lease.artifact_key, "revoked", "missing_entry"))
                        continue
                    entry.lease_ids.discard(lease_id)
                    del self.leases[lease_id]
                    renewed = LeaseRecord(
                        lease_id=new_lease_id,
                        artifact_key=lease.artifact_key,
                        digest=lease.digest,
                        quota_version=quota_version,
                        apply_seq=lease.apply_seq,
                        apply_ordinal=lease.apply_ordinal,
                        created_ms=lease.created_ms,
                        expires_at_ms=lease.expires_at_ms,
                    )
                    self.leases[new_lease_id] = renewed
                    entry.lease_ids.add(new_lease_id)
                    entry.quota_version = quota_version
                    self.renewed_count += 1
                    lease_dispositions.append(
                        Disposition(old_lease_id, lease.artifact_key, "renewed", "policy_renew", new_lease_id)
                    )

            self.capacity_slots = new_capacity
            self.lease_limit = new_limit
            self.quota_version = quota_version

            overflow = len(self.entries) > self.capacity_slots
            if overflow:
                for artifact_key in sorted(self.entries):
                    entry = self.entries[artifact_key]
                    for lease_id in sorted(entry.lease_ids):
                        lease = self.leases.pop(lease_id, None)
                        if lease is None:
                            continue
                        self.revoked_count += 1
                        lease_dispositions.append(
                            Disposition(lease_id, artifact_key, "revoked", "capacity_shrink_after_renew")
                        )
                    entry.lease_ids.clear()
                    self._cleanup_entry_locked(artifact_key)
                    if len(self.entries) <= self.capacity_slots:
                        break

            lease_overflow = len(self.leases) > self.lease_limit
            if lease_overflow:
                for lease_id in sorted(list(self.leases)):
                    if len(self.leases) <= self.lease_limit:
                        break
                    lease = self.leases.pop(lease_id)
                    self.revoked_count += 1
                    if lease is not None:
                        entry = self.entries.get(lease.artifact_key)
                        if entry is not None:
                            entry.lease_ids.discard(lease_id)
                            self._cleanup_entry_locked(lease.artifact_key)
                        lease_dispositions.append(
                            Disposition(lease_id, lease.artifact_key, "revoked", "lease_limit_shrink_after_renew")
                        )

            return self.ok({
                "old_quota_version": old_version,
                "quota_version": self.quota_version,
                "old_capacity_slots": old_capacity,
                "capacity_slots": self.capacity_slots,
                "old_lease_limit": old_limit,
                "lease_limit": self.lease_limit,
                "policy": policy,
                "lease_dispositions": [disposition.__dict__ for disposition in lease_dispositions],
                "reservation_dispositions": reservation_dispositions,
                "watermark": self.watermark_locked(),
            })

    def reconcile(self, max_committed_seq: int, max_committed_ordinal: int | None = None) -> dict[str, Any]:
        with self.lock:
            self.expire_locked()
            if max_committed_ordinal is None:
                max_committed_ordinal = max_committed_seq
            removed_reservations: list[dict[str, Any]] = []
            for reservation_id in sorted(list(self.reservations)):
                reservation = self.reservations[reservation_id]
                if reservation.apply_ordinal > max_committed_ordinal or (
                    reservation.apply_ordinal == 0 and reservation.apply_seq > max_committed_seq
                ):
                    del self.reservations[reservation_id]
                    self._cleanup_entry_locked(reservation.artifact_key)
                    removed_reservations.append({
                        "reservation_id": reservation_id,
                        "artifact_key": reservation.artifact_key,
                        "apply_seq": reservation.apply_seq,
                        "apply_ordinal": reservation.apply_ordinal,
                        "status": "unknown",
                        "reason": "uncommitted_batch_after_committed_progress",
                    })
            removed_leases: list[dict[str, Any]] = []
            for lease_id in sorted(list(self.leases)):
                lease = self.leases[lease_id]
                if lease.apply_ordinal > max_committed_ordinal or (
                    lease.apply_ordinal == 0 and lease.apply_seq > max_committed_seq
                ):
                    del self.leases[lease_id]
                    if lease.artifact_key in self.entries:
                        self.entries[lease.artifact_key].lease_ids.discard(lease_id)
                    self._cleanup_entry_locked(lease.artifact_key)
                    removed_leases.append({
                        "lease_id": lease_id,
                        "artifact_key": lease.artifact_key,
                        "apply_seq": lease.apply_seq,
                        "apply_ordinal": lease.apply_ordinal,
                        "status": "unknown",
                        "reason": "uncommitted_batch_after_committed_progress",
                    })
            return self.ok({
                "max_committed_seq": max_committed_seq,
                "max_committed_ordinal": max_committed_ordinal,
                "unknown_reservations": removed_reservations,
                "unknown_leases": removed_leases,
                "watermark": self.watermark_locked(),
            })

    def restore(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        """Restore committed entry/lease state for an incremental verification run."""
        with self.lock:
            self.expire_locked()
            if snapshot.get("node_id") != self.node_id:
                return self.error("E_RESTORE", "snapshot node mismatch")
            self.capacity_slots = int(snapshot["capacity_slots"])
            self.lease_limit = int(snapshot["lease_limit"])
            self.quota_version = int(snapshot["quota_version"])
            self.entries.clear()
            self.leases.clear()
            self.reservations.clear()
            now = mono_ms()
            for item in snapshot.get("entries", []):
                key = str(item["artifact_key"])
                digest = str(item["digest"])
                shard_count = int(item.get("shards", self.config.artifact_by_key[key].shards))
                shard_bytes = int(
                    item.get("shard_bytes", self.config.artifact_by_key[key].shard_bytes)
                )
                generated = generate_content(
                    self.node_id,
                    key,
                    self.spec.seed,
                    shard_count,
                    shard_bytes,
                )
                if generated.digest != digest:
                    return self.error(
                        "E_RESTORE",
                        "snapshot content digest does not match deterministic node content",
                        {"artifact_key": key},
                    )
                self.entries[key] = EntryRecord(
                    artifact_key=key,
                    digest=digest,
                    quota_version=int(item.get("quota_version", self.quota_version)),
                    shard_count=shard_count,
                    shard_bytes=shard_bytes,
                    shards=generated.shards,
                )
            for item in snapshot.get("leases", []):
                key = str(item["artifact_key"])
                entry = self.entries.get(key)
                if entry is None:
                    continue
                lease_id = str(item["lease_id"])
                remaining = max(1, int(item.get("remaining_ms", 1)))
                self.leases[lease_id] = LeaseRecord(
                    lease_id=lease_id,
                    artifact_key=key,
                    digest=str(item.get("digest", entry.digest)),
                    quota_version=int(item.get("quota_version", self.quota_version)),
                    apply_seq=int(item.get("apply_seq", 1)),
                    apply_ordinal=int(item.get("apply_ordinal", item.get("apply_seq", 1))),
                    created_ms=now,
                    expires_at_ms=now + remaining,
                )
                entry.lease_ids.add(lease_id)
            return self.ok({"restored": True, "watermark": self.watermark_locked()})

    @staticmethod
    def ok(payload: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True, **payload}

    @staticmethod
    def error(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {"ok": False, "error": code, "message": message}
        if details:
            result["details"] = details
        return result


class _Reaper(threading.Thread):
    def __init__(self, state: NodeState, stop_event: threading.Event) -> None:
        super().__init__(name="cachelease-reaper", daemon=True)
        self.state = state
        self.stop_event = stop_event

    def run(self) -> None:
        while not self.stop_event.wait(0.005):
            try:
                with self.state.lock:
                    self.state.expire_locked()
            except Exception:
                traceback.print_exc()


class _ParentWatchdog(threading.Thread):
    def __init__(self, parent_pid: int, server: "_ThreadingLineServer", stop_event: threading.Event) -> None:
        super().__init__(name="cachelease-parent-watchdog", daemon=True)
        self.parent_pid = parent_pid
        self.server = server
        self.stop_event = stop_event

    def _parent_alive(self) -> bool:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            synchronize = 0x00100000
            wait_timeout = 0x00000102
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.argtypes = [
                wintypes.DWORD,
                wintypes.BOOL,
                wintypes.DWORD,
            ]
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            kernel32.WaitForSingleObject.restype = wintypes.DWORD
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel32.CloseHandle.restype = wintypes.BOOL
            handle = kernel32.OpenProcess(synchronize, False, self.parent_pid)
            if not handle:
                return False
            try:
                result = kernel32.WaitForSingleObject(handle, 0)
                return result == wait_timeout
            finally:
                kernel32.CloseHandle(handle)
        return os.getppid() == self.parent_pid

    def run(self) -> None:
        while not self.stop_event.wait(0.1):
            if not self._parent_alive():
                self.server.shutdown_requested.set()
                return


class _ThreadingLineServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 128

    def __init__(self, address: tuple[str, int], handler: type[socketserver.BaseRequestHandler], state: NodeState):
        self.state = state
        self.shutdown_requested = threading.Event()
        super().__init__(address, handler)


class _LineHandler(socketserver.StreamRequestHandler):
    server: _ThreadingLineServer

    def setup(self) -> None:
        super().setup()
        self.request.settimeout(10.0)
        state = self.server.state
        with state.lock:
            state.request_count += 1
            state.in_flight += 1
            state.max_in_flight = max(state.max_in_flight, state.in_flight)

    def finish(self) -> None:
        state = self.server.state
        try:
            with state.lock:
                state.in_flight -= 1
        finally:
            super().finish()

    def handle(self) -> None:
        try:
            while not self.server.shutdown_requested.is_set():
                raw = self.rfile.readline(1024 * 1024 + 1)
                if not raw:
                    return
                if len(raw) > 1024 * 1024:
                    self._send({"ok": False, "error": "E_PROTOCOL", "message": "request line too long"})
                    return
                started = mono_ms()
                request: dict[str, Any] = {}
                try:
                    request = json.loads(raw.decode("utf-8"))
                    if not isinstance(request, dict):
                        raise ValueError("request must be an object")
                    response = self._dispatch(request)
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                    response = {"ok": False, "error": "E_PROTOCOL", "message": f"invalid JSON request: {exc}"}
                except Exception as exc:
                    response = {"ok": False, "error": "E_INTERNAL", "message": str(exc)}
                    traceback.print_exc()
                response.setdefault("elapsed_ms", max(0, mono_ms() - started))
                self._send(response)
                if request.get("op") == "shutdown" and response.get("ok"):
                    self.server.shutdown_requested.set()
                    return
        except (ConnectionError, OSError, socket.timeout):
            return

    def _send(self, payload: dict[str, Any]) -> None:
        data = (canonical_json(payload) + "\n").encode("utf-8")
        self.wfile.write(data)
        self.wfile.flush()

    def _dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        state = self.server.state
        op = request.get("op")
        if op == "hello":
            return state.ok({
                "protocol_version": PROTOCOL_VERSION,
                "node_id": state.node_id,
                "quota_version": state.quota_version,
                "status": state.status(),
                "content_rule": "CACHELEASE-ARTIFACT-SHARD-v1",
            })
        if op == "status":
            return state.ok({"status": state.status()})
        if op == "reserve":
            return state.reserve(
                request.get("artifact_key"),
                request.get("lease_ms"),
                request.get("apply_seq"),
                request.get("grace_ms", 1000),
                request.get("expected_quota_version"),
                request.get("apply_ordinal"),
            )
        if op == "commit":
            return state.commit(request.get("reservation_id"))
        if op == "rollback":
            return state.rollback(request.get("reservation_id"), str(request.get("reason", "client_rollback")))
        if op == "read":
            return state.read(
                request.get("artifact_key"),
                request.get("lease_ms"),
                request.get("apply_seq"),
                request.get("apply_ordinal"),
            )
        if op == "release":
            return state.release(request.get("lease_id"))
        if op == "release_artifact":
            return state.release_artifact(request.get("artifact_key"))
        if op == "set_quota":
            return state.set_quota(
                request.get("quota_version"),
                request.get("capacity_slots"),
                request.get("lease_limit"),
                request.get("policy"),
            )
        if op == "reconcile":
            return state.reconcile(
                int(request.get("max_committed_seq", 0)),
                int(request.get("max_committed_ordinal", request.get("max_committed_seq", 0))),
            )
        if op == "restore":
            snapshot = request.get("snapshot")
            if not isinstance(snapshot, dict):
                return state.error("E_ARGUMENT", "snapshot must be an object")
            return state.restore(snapshot)
        if op == "shutdown":
            return state.ok({"shutdown": True})
        return state.error("E_PROTOCOL", f"unknown op {op!r}")


def serve(config_path: str, node_id: str, parent_pid: int, ready_file: str | None) -> int:
    config = load_config(config_path)
    if node_id not in config.node_by_id:
        print(f"unknown node {node_id}", file=sys.stderr)
        return 2
    spec = config.node_by_id[node_id]
    state = NodeState(config, node_id)
    server = _ThreadingLineServer((spec.host, spec.port), _LineHandler, state)
    stop_event = threading.Event()
    reaper = _Reaper(state, stop_event)
    reaper.start()
    watchdog: _ParentWatchdog | None = None
    if parent_pid > 0:
        watchdog = _ParentWatchdog(parent_pid, server, stop_event)
        watchdog.start()

    if ready_file:
        Path(ready_file).write_text(
            canonical_json({"ready": True, "node_id": node_id, "pid": os.getpid(), "port": spec.port}),
            encoding="utf-8",
        )

    def request_shutdown(_signum: int, _frame: Any) -> None:
        server.shutdown_requested.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    for sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGINT", None)):
        if sig is not None:
            try:
                signal.signal(sig, request_shutdown)
            except (OSError, ValueError):
                pass

    thread = threading.Thread(target=server.serve_forever, name="cachelease-server", daemon=True)
    thread.start()
    try:
        while not server.shutdown_requested.wait(0.05):
            pass
    finally:
        stop_event.set()
        server.shutdown()
        server.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one deterministic cache lease node")
    parser.add_argument("--config", required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--parent-pid", type=int, default=0)
    parser.add_argument("--ready-file")
    args = parser.parse_args(argv)
    try:
        return serve(args.config, args.node_id, args.parent_pid, args.ready_file)
    except CacheLeaseError as exc:
        print(canonical_json(exc.to_dict()), file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"node startup failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
