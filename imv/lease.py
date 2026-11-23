from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from typing import Any, Callable

from .errors import BusinessError, fail
from .util import monotonic_ms, wall_ms


LEASE_STATES = {"active", "completed", "expired", "failed"}


@dataclass(frozen=True)
class LeaseRecord:
    view_id: str
    param_version: int
    refresh_version: int
    state: str
    granted_mono_ms: int
    expires_mono_ms: int
    granted_wall_ms: int
    expires_wall_ms: int
    ttl_ms: int
    renew_ms: int
    renew_count: int = 0
    last_renew_mono_ms: int | None = None
    job_id: str | None = None
    stale: bool = False
    note: str | None = None

    def payload(self) -> dict[str, Any]:
        return {
            "view_id": self.view_id,
            "param_version": self.param_version,
            "refresh_version": self.refresh_version,
            "state": self.state,
            "granted_mono_ms": self.granted_mono_ms,
            "expires_mono_ms": self.expires_mono_ms,
            "granted_wall_ms": self.granted_wall_ms,
            "expires_wall_ms": self.expires_wall_ms,
            "ttl_ms": self.ttl_ms,
            "renew_ms": self.renew_ms,
            "renew_count": self.renew_count,
            "last_renew_mono_ms": self.last_renew_mono_ms,
            "job_id": self.job_id,
            "stale": self.stale,
            "note": self.note,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "LeaseRecord":
        return cls(
            view_id=payload["view_id"],
            param_version=int(payload["param_version"]),
            refresh_version=int(payload["refresh_version"]),
            state=payload.get("state", "expired"),
            granted_mono_ms=int(payload.get("granted_mono_ms", 0)),
            expires_mono_ms=int(payload.get("expires_mono_ms", 0)),
            granted_wall_ms=int(payload.get("granted_wall_ms", 0)),
            expires_wall_ms=int(payload.get("expires_wall_ms", 0)),
            ttl_ms=int(payload.get("ttl_ms", 0)),
            renew_ms=int(payload.get("renew_ms", 0)),
            renew_count=int(payload.get("renew_count", 0)),
            last_renew_mono_ms=payload.get("last_renew_mono_ms"),
            job_id=payload.get("job_id"),
            stale=bool(payload.get("stale", False)),
            note=payload.get("note"),
        )


class LeaseManager:
    """Lease issuance and optimistic renewal using a real monotonic clock."""

    def __init__(
        self,
        *,
        monotonic_clock: Callable[[], int] = monotonic_ms,
        wall_clock: Callable[[], int] = wall_ms,
    ):
        self._lock = threading.RLock()
        self._leases: dict[str, LeaseRecord] = {}
        self._monotonic_clock = monotonic_clock
        self._wall_clock = wall_clock

    def load(self, records: dict[str, LeaseRecord] | list[LeaseRecord]) -> None:
        with self._lock:
            items = records.values() if isinstance(records, dict) else records
            self._leases = {record.view_id: replace(record) for record in items}

    def issue(
        self,
        view_id: str,
        param_version: int,
        refresh_version: int,
        ttl_ms: int,
        renew_ms: int,
        *,
        job_id: str | None = None,
    ) -> LeaseRecord:
        if ttl_ms <= 0 or renew_ms <= 0 or renew_ms >= ttl_ms:
            fail("E_LEASE", "invalid lease timing", location={"view_id": view_id}, details={"ttl_ms": ttl_ms, "renew_ms": renew_ms})
        with self._lock:
            current = self._leases.get(view_id)
            now_mono = self._monotonic_clock()
            now_wall = self._wall_clock()
            if current is not None:
                if current.state == "active" and now_mono < current.expires_mono_ms:
                    fail(
                        "E_LEASE",
                        "view already holds an unexpired active lease",
                        location={"view_id": view_id},
                        details={"active_refresh_version": current.refresh_version},
                    )
                if refresh_version <= current.refresh_version:
                    fail(
                        "E_LEASE",
                        "refresh version must increase monotonically",
                        location={"view_id": view_id},
                        details={"current": current.refresh_version, "requested": refresh_version},
                    )
            record = LeaseRecord(
                view_id=view_id,
                param_version=param_version,
                refresh_version=refresh_version,
                state="active",
                granted_mono_ms=now_mono,
                expires_mono_ms=now_mono + ttl_ms,
                granted_wall_ms=now_wall,
                expires_wall_ms=now_wall + ttl_ms,
                ttl_ms=ttl_ms,
                renew_ms=renew_ms,
                job_id=job_id,
            )
            self._leases[view_id] = record
            return record

    def renew(self, view_id: str, expected_version: int, ttl_ms: int | None = None) -> LeaseRecord:
        with self._lock:
            record = self._leases.get(view_id)
            if record is None:
                fail("E_LEASE", "view has no lease", location={"view_id": view_id})
            if record.refresh_version != expected_version:
                fail(
                    "E_LEASE",
                    "lease refresh version conflict",
                    location={"view_id": view_id, "field": "refresh_version"},
                    details={"expected": expected_version, "actual": record.refresh_version},
                )
            if record.state != "active":
                fail("E_LEASE", "lease is not active", location={"view_id": view_id}, details={"state": record.state})
            now_mono = self._monotonic_clock()
            if now_mono >= record.expires_mono_ms:
                expired = replace(record, state="expired", note="renewal arrived after expiry")
                self._leases[view_id] = expired
                fail(
                    "E_LEASE",
                    "lease expired before renewal",
                    location={"view_id": view_id, "field": "expires_mono_ms"},
                    details={"now_mono_ms": now_mono, "expires_mono_ms": record.expires_mono_ms},
                )
            ttl = record.ttl_ms if ttl_ms is None else ttl_ms
            now_wall = self._wall_clock()
            renewed = replace(
                record,
                expires_mono_ms=now_mono + ttl,
                expires_wall_ms=now_wall + ttl,
                ttl_ms=ttl,
                renew_count=record.renew_count + 1,
                last_renew_mono_ms=now_mono,
            )
            self._leases[view_id] = renewed
            return renewed

    def complete(self, view_id: str, expected_version: int) -> LeaseRecord:
        with self._lock:
            record = self._leases.get(view_id)
            if record is None:
                fail("E_LEASE", "view has no lease", location={"view_id": view_id})
            if record.refresh_version != expected_version:
                fail(
                    "E_LEASE",
                    "lease refresh version conflict on completion",
                    location={"view_id": view_id, "field": "refresh_version"},
                    details={"expected": expected_version, "actual": record.refresh_version},
                )
            if record.state != "active":
                fail("E_LEASE", "lease is not active", location={"view_id": view_id}, details={"state": record.state})
            now_mono = self._monotonic_clock()
            if now_mono >= record.expires_mono_ms:
                expired = replace(record, state="expired", note="job finished after lease expiry")
                self._leases[view_id] = expired
                fail(
                    "E_LEASE",
                    "lease expired before job completion",
                    location={"view_id": view_id, "field": "expires_mono_ms"},
                    details={"now_mono_ms": now_mono, "expires_mono_ms": record.expires_mono_ms},
                )
            completed = replace(record, state="completed")
            self._leases[view_id] = completed
            return completed

    def fail(self, view_id: str, expected_version: int, note: str) -> LeaseRecord | None:
        with self._lock:
            record = self._leases.get(view_id)
            if record is None or record.refresh_version != expected_version:
                return record
            failed = replace(record, state="failed", note=note)
            self._leases[view_id] = failed
            return failed

    def expire_due(self) -> list[LeaseRecord]:
        expired: list[LeaseRecord] = []
        with self._lock:
            now_mono = self._monotonic_clock()
            for view_id, record in list(self._leases.items()):
                if record.state == "active" and now_mono >= record.expires_mono_ms:
                    expired_record = replace(record, state="expired", note="lease TTL elapsed without renewal")
                    self._leases[view_id] = expired_record
                    expired.append(expired_record)
        return expired

    def check_view(self, view_id: str) -> LeaseRecord | None:
        with self._lock:
            record = self._leases.get(view_id)
            if record is None:
                return None
            if record.state == "active" and self._monotonic_clock() >= record.expires_mono_ms:
                expired = replace(record, state="expired", note="lease TTL elapsed without renewal")
                self._leases[view_id] = expired
                return expired
            return record

    def get(self, view_id: str) -> LeaseRecord | None:
        with self._lock:
            return self._leases.get(view_id)

    def snapshot(self, view_id: str) -> dict[str, Any] | None:
        record = self.check_view(view_id)
        return record.payload() if record is not None else None

    def all_payload(self) -> dict[str, dict[str, Any]]:
        self.expire_due()
        with self._lock:
            return {view_id: record.payload() for view_id, record in sorted(self._leases.items())}

    def expire_active_after_restore(self) -> list[str]:
        """A process restart cannot prove that an old in-process renewer is alive."""

        expired_views: list[str] = []
        with self._lock:
            for view_id, record in list(self._leases.items()):
                if record.state == "active":
                    self._leases[view_id] = replace(record, state="expired", note="process restart ended lease renewal")
                    expired_views.append(view_id)
        return sorted(expired_views)

    def mark_inconsistent(self, view_id: str, note: str) -> None:
        with self._lock:
            record = self._leases.get(view_id)
            if record is not None and record.state == "active":
                self._leases[view_id] = replace(record, state="failed", note=note)

    def usable(self, view_id: str) -> bool:
        record = self.check_view(view_id)
        if record is None:
            return False
        return record.state in {"active", "completed"}

    def remaining_ms(self, view_id: str) -> int | None:
        record = self.check_view(view_id)
        if record is None or record.state != "active":
            return None
        return max(0, record.expires_mono_ms - self._monotonic_clock())

    def assert_version(self, view_id: str, expected_version: int) -> None:
        record = self.get(view_id)
        if record is None or record.refresh_version != expected_version:
            actual = None if record is None else record.refresh_version
            raise BusinessError(
                "E_LEASE",
                "lease refresh version conflict",
                {"view_id": view_id, "field": "refresh_version"},
                {"expected": expected_version, "actual": actual},
            )
