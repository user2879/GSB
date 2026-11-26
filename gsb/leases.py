"""Monotonic-clock leases with optimistic versions checked by the worker."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Callable

from .canonical import sha256_hex
from .errors import DiagnosticError

RemoteRenew = Callable[[str, str, int, int], dict[str, Any]]
RemoteRelease = Callable[[str, str], dict[str, Any]]


def monotonic_ms() -> int:
    return time.monotonic_ns() // 1_000_000


@dataclass(frozen=True)
class LeaseRecord:
    lease_id: str
    shard_id: str
    granted_at_ms: int
    expires_at_ms: int
    renew_count: int
    lease_version: int
    state: str
    owner: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "shard_id": self.shard_id,
            "granted_at_ms": self.granted_at_ms,
            "expires_at_ms": self.expires_at_ms,
            "renew_count": self.renew_count,
            "lease_version": self.lease_version,
            "state": self.state,
            "owner": self.owner,
        }

    @property
    def digest(self) -> str:
        return sha256_hex(self.to_dict())


class LeaseManager:
    def __init__(
        self,
        *,
        clock: Callable[[], int] = monotonic_ms,
        remote_renew: RemoteRenew | None = None,
        remote_release: RemoteRelease | None = None,
    ) -> None:
        self._clock = clock
        self._remote_renew = remote_renew
        self._remote_release = remote_release
        self._records: dict[str, LeaseRecord] = {}
        self._next_id: dict[str, int] = {}
        self._default_ttl_ms: dict[str, int] = {}

    @property
    def now_ms(self) -> int:
        return int(self._clock())

    def _expire_locked(self) -> list[LeaseRecord]:
        now = self.now_ms
        expired: list[LeaseRecord] = []
        for lease_id, record in list(self._records.items()):
            if record.state == "active" and now >= record.expires_at_ms:
                updated = replace(record, state="expired")
                self._records[lease_id] = updated
                expired.append(updated)
        return expired

    def issue(
        self,
        shard_id: str,
        ttl_ms: int,
        *,
        owner: str = "scheduler",
        lease_id: str | None = None,
        remote_issue: Callable[[str, str, int], dict[str, Any]] | None = None,
    ) -> LeaseRecord:
        if ttl_ms <= 0:
            raise DiagnosticError("E_LEASE", "lease ttl must be positive", field="ttl_ms")
        self._expire_locked()
        if lease_id is None:
            next_number = self._next_id.get(shard_id, 0) + 1
            self._next_id[shard_id] = next_number
            lease_id = f"lease:{shard_id}:{next_number}"
        if lease_id in self._records:
            raise DiagnosticError("E_LEASE", f"duplicate lease id {lease_id}", field="lease_id")
        if remote_issue is not None:
            response = remote_issue(shard_id, lease_id, ttl_ms)
            if not response.get("ok", False):
                raise DiagnosticError(
                    response.get("code", "E_LEASE"),
                    response.get("message", "worker rejected lease issue"),
                    field="lease_id",
                )
            if response.get("lease_version") != 1:
                raise DiagnosticError(
                    "E_FIELD",
                    "worker lease issue response has an invalid lease_version",
                    field="lease_version",
                )
        now = self.now_ms
        record = LeaseRecord(
            lease_id=lease_id,
            shard_id=shard_id,
            granted_at_ms=now,
            expires_at_ms=now + ttl_ms,
            renew_count=0,
            lease_version=1,
            state="active",
            owner=owner,
        )
        self._records[lease_id] = record
        self._default_ttl_ms[lease_id] = int(ttl_ms)
        return record

    def get(self, lease_id: str) -> LeaseRecord:
        self._expire_locked()
        try:
            return self._records[lease_id]
        except KeyError as exc:
            raise DiagnosticError("E_LEASE", f"unknown lease {lease_id}", field="lease_id") from exc

    def renew(
        self, lease_id: str, lease_version: int, *, ttl_ms: int | None = None
    ) -> LeaseRecord:
        self._expire_locked()
        try:
            record = self._records[lease_id]
        except KeyError as exc:
            raise DiagnosticError("E_LEASE", f"unknown lease {lease_id}", field="lease_id") from exc
        if record.state != "active":
            raise DiagnosticError(
                "E_LEASE", f"lease {lease_id} is {record.state}", field="lease_id"
            )
        if record.lease_version != lease_version:
            # Deliberately leave expires_at_ms unchanged.
            raise DiagnosticError(
                "E_LEASE",
                f"lease version mismatch: expected {record.lease_version}, got {lease_version}",
                field="lease_version",
            )
        effective_ttl = (
            ttl_ms if ttl_ms is not None else self._default_ttl_ms.get(lease_id, 0)
        )
        if effective_ttl <= 0:
            raise DiagnosticError("E_LEASE", "lease ttl must be positive", field="ttl_ms")
        if self._remote_renew is not None:
            response = self._remote_renew(
                record.shard_id, lease_id, lease_version, int(effective_ttl)
            )
            if not response.get("ok", False):
                raise DiagnosticError(
                    response.get("code", "E_LEASE"),
                    response.get("message", "worker rejected lease renewal"),
                    field="lease_version",
                )
            remote_version = response.get("lease_version")
            if remote_version != lease_version + 1:
                raise DiagnosticError(
                    "E_FIELD",
                    "worker lease renewal response has an invalid lease_version",
                    field="lease_version",
                )
        now = self.now_ms
        updated = replace(
            record,
            expires_at_ms=now + int(effective_ttl),
            renew_count=record.renew_count + 1,
            lease_version=record.lease_version + 1,
        )
        self._records[lease_id] = updated
        return updated

    def release(self, lease_id: str) -> LeaseRecord:
        self._expire_locked()
        try:
            record = self._records[lease_id]
        except KeyError as exc:
            raise DiagnosticError("E_LEASE", f"unknown lease {lease_id}", field="lease_id") from exc
        if self._remote_release is not None and record.state == "active":
            try:
                self._remote_release(record.shard_id, lease_id)
            except DiagnosticError:
                # Local quota is still reclaimed if the child has already exited.
                pass
        if record.state == "active":
            record = replace(record, state="released")
            self._records[lease_id] = record
        return record

    def cancel(self, lease_id: str) -> LeaseRecord:
        return self.release(lease_id)

    def prune(self) -> tuple[LeaseRecord, ...]:
        return tuple(self._expire_locked())

    def active_for_shard(self, shard_id: str) -> tuple[LeaseRecord, ...]:
        self._expire_locked()
        return tuple(
            record
            for record in self._records.values()
            if record.shard_id == shard_id and record.state == "active"
        )

    def active_count(self, shard_id: str) -> int:
        return len(self.active_for_shard(shard_id))

    def snapshot(self) -> dict[str, Any]:
        self._expire_locked()
        records = {key: self._records[key].to_dict() for key in sorted(self._records)}
        return {"records": records, "clock_ms": self.now_ms}
