from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from .config import LeaseDeclaration
from .errors import (
    BusinessError,
    LeaseError,
    LeaseExpiredError,
    LeaseGenerationRollbackError,
)
from .util import (
    atomic_write_json,
    process_is_same_instance,
    process_start_token,
    read_json,
)


class Clock(Protocol):
    def now_ms(self) -> int: ...


class MonotonicClock:
    def now_ms(self) -> int:
        return time.monotonic_ns() // 1_000_000


@dataclass
class LeaseRecord:
    lease_id: str
    worker_id: str
    cluster_id: int
    generation: int
    issued_ms: int
    expires_ms: int
    ttl_ms: int
    owner_pid: int
    owner_start_token: str | None
    state: str = "active"
    released_ms: int | None = None
    supersedes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LeaseManager:
    def __init__(
        self,
        declaration: LeaseDeclaration,
        state_dir: Path,
        clock: Clock | None = None,
        *,
        owner_pid: int | None = None,
        save_on_init: bool = True,
    ):
        self.declaration = declaration
        self.state_dir = Path(state_dir)
        self.state_path = self.state_dir / "leases.json"
        self.clock = clock or MonotonicClock()
        self.owner_pid = int(owner_pid if owner_pid is not None else os.getpid())
        self.owner_start_token = process_start_token(self.owner_pid)
        self.high_water_generation = 0
        self.records: dict[str, LeaseRecord] = {}
        self.events: list[dict[str, Any]] = []
        self.reclaimed: list[dict[str, Any]] = []
        self.rollback_error: BusinessError | None = None
        self._load()
        if (
            self.declaration.restore_generation > 0
            and self.declaration.restore_generation < self.high_water_generation
        ):
            self.rollback_error = BusinessError(
                code="E_LEASE_GENERATION_ROLLBACK",
                path="lease.restore_generation",
                message=(
                    "lease declaration generation floor is behind persisted high-water "
                    f"generation ({self.declaration.restore_generation} < "
                    f"{self.high_water_generation})"
                ),
                stage="lease",
            )
            self._event(
                "generation_rollback_detected",
                None,
                self.declaration.restore_generation,
                self.rollback_error.message,
            )
        elif self.declaration.restore_generation > self.high_water_generation:
            self.high_water_generation = self.declaration.restore_generation
            self._event(
                "generation_floor_restored",
                None,
                self.high_water_generation,
                "initialized high-water generation from lease declaration",
            )
        self.recover_orphans()
        if save_on_init:
            self.save()

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            data = read_json(self.state_path)
        except (OSError, ValueError) as exc:
            raise LeaseError(f"cannot read lease state: {exc}", path="state/leases.json") from exc
        if not isinstance(data, dict):
            raise LeaseError("lease state must be a mapping", path="state/leases.json")
        high_water = data.get("high_water_generation", 0)
        if not isinstance(high_water, int) or isinstance(high_water, bool) or high_water < 0:
            raise LeaseError(
                "lease state high_water_generation is invalid",
                path="state/leases.json.high_water_generation",
            )
        self.high_water_generation = high_water
        records = data.get("leases", {})
        if not isinstance(records, dict):
            raise LeaseError("lease state leases must be a mapping", path="state/leases.json.leases")
        for lease_id, record in records.items():
            if not isinstance(record, dict):
                continue
            try:
                self.records[str(lease_id)] = LeaseRecord(**record)
            except TypeError as exc:
                raise LeaseError(
                    f"invalid persisted lease record: {lease_id}",
                    path=f"state/leases.json.leases.{lease_id}",
                ) from exc
        events = data.get("events", [])
        if isinstance(events, list):
            self.events = events

    def save(self) -> None:
        payload = {
            "schema_version": 1,
            "high_water_generation": self.high_water_generation,
            "worker_id": self.declaration.worker_id,
            "leases": {
                lease_id: record.to_dict()
                for lease_id, record in sorted(self.records.items())
            },
            "events": self.events,
        }
        atomic_write_json(self.state_path, payload)

    def _event(
        self, event: str, lease_id: str | None, generation: int, message: str
    ) -> None:
        self.events.append(
            {
                "event": event,
                "lease_id": lease_id,
                "generation": generation,
                "monotonic_ms": self.clock.now_ms(),
                "message": message,
            }
        )

    def recover_orphans(self) -> list[dict[str, Any]]:
        reclaimed: list[dict[str, Any]] = []
        now = self.clock.now_ms()
        for lease_id, record in sorted(self.records.items()):
            if record.state != "active":
                continue
            alive = process_is_same_instance(record.owner_pid, record.owner_start_token)
            if alive:
                continue
            record.state = "reclaimed"
            record.released_ms = now
            item = {
                "lease_id": lease_id,
                "generation": record.generation,
                "owner_pid": record.owner_pid,
                "reason": "owner_process_not_alive",
            }
            reclaimed.append(item)
            self.reclaimed.append(item)
            self._event("lease_reclaimed", lease_id, record.generation, item["reason"])
        return reclaimed

    def issue(self, cluster_id: int) -> LeaseRecord:
        if self.rollback_error is not None:
            raise LeaseGenerationRollbackError(
                self.rollback_error.message,
                path=self.rollback_error.path,
            )
        now = self.clock.now_ms()
        generation = self.high_water_generation + 1
        self.high_water_generation = generation
        lease_id = f"{self.declaration.worker_id}:cluster-{cluster_id}:gen-{generation}"
        record = LeaseRecord(
            lease_id=lease_id,
            worker_id=self.declaration.worker_id,
            cluster_id=int(cluster_id),
            generation=generation,
            issued_ms=now,
            expires_ms=now + self.declaration.ttl_ms,
            ttl_ms=self.declaration.ttl_ms,
            owner_pid=self.owner_pid,
            owner_start_token=self.owner_start_token,
        )
        self.records[lease_id] = record
        self._event("lease_issued", lease_id, generation, "cluster compute lease issued")
        self.save()
        return record

    def issue_for_clusters(self, families: int) -> list[LeaseRecord]:
        return [self.issue(cluster_id) for cluster_id in range(families)]

    def get(self, lease_id: str) -> LeaseRecord:
        try:
            return self.records[lease_id]
        except KeyError as exc:
            raise LeaseError(
                f"lease does not exist: {lease_id}", path=f"leases.{lease_id}"
            ) from exc

    def renew(self, lease_id: str) -> LeaseRecord:
        record = self.get(lease_id)
        now = self.clock.now_ms()
        if record.state == "expired":
            raise LeaseExpiredError(
                f"lease already expired at monotonic millisecond {record.expires_ms}",
                path=f"leases.{lease_id}.expires_ms",
            )
        if record.state != "active":
            raise LeaseError(
                f"lease is not active: {record.state}", path=f"leases.{lease_id}.state"
            )
        if now >= record.expires_ms:
            record.state = "expired"
            record.released_ms = now
            self._event("lease_expired", lease_id, record.generation, "renew attempted after expiry")
            self.save()
            raise LeaseExpiredError(
                f"lease expired at monotonic millisecond {record.expires_ms}",
                path=f"leases.{lease_id}.expires_ms",
            )
        record.state = "superseded"
        record.released_ms = now
        generation = self.high_water_generation + 1
        self.high_water_generation = generation
        renewed_id = f"{self.declaration.worker_id}:cluster-{record.cluster_id}:gen-{generation}"
        renewed = LeaseRecord(
            lease_id=renewed_id,
            worker_id=self.declaration.worker_id,
            cluster_id=record.cluster_id,
            generation=generation,
            issued_ms=now,
            expires_ms=now + record.ttl_ms,
            ttl_ms=record.ttl_ms,
            owner_pid=self.owner_pid,
            owner_start_token=self.owner_start_token,
            supersedes=lease_id,
        )
        self.records[renewed_id] = renewed
        self._event(
            "lease_renewed",
            renewed_id,
            generation,
            f"renewed before expiry; superseded {lease_id}",
        )
        self.save()
        return renewed

    def expire_due(self) -> list[dict[str, Any]]:
        now = self.clock.now_ms()
        expired: list[dict[str, Any]] = []
        for lease_id, record in sorted(self.records.items()):
            if record.state != "active" or now < record.expires_ms:
                continue
            record.state = "expired"
            record.released_ms = now
            item = {
                "lease_id": lease_id,
                "cluster_id": record.cluster_id,
                "generation": record.generation,
                "expires_ms": record.expires_ms,
                "released_ms": now,
            }
            expired.append(item)
            self._event("lease_expired", lease_id, record.generation, "expiry reclaimed")
        if expired:
            self.save()
        return expired

    def commit(self, lease_ids: list[str]) -> None:
        now = self.clock.now_ms()
        for lease_id in sorted(lease_ids):
            record = self.get(lease_id)
            if record.state == "expired":
                raise LeaseExpiredError(
                    f"cannot commit expired lease: {lease_id}",
                    path=f"leases.{lease_id}.state",
                )
            if record.state == "active":
                if now >= record.expires_ms:
                    record.state = "expired"
                    record.released_ms = now
                    self._event(
                        "lease_expired",
                        lease_id,
                        record.generation,
                        "commit attempted after expiry",
                    )
                    self.save()
                    raise LeaseExpiredError(
                        "cannot commit an active lease after its expiry",
                        path=f"leases.{lease_id}.expires_ms",
                    )
                record.state = "committed"
                record.released_ms = now
                self._event(
                    "lease_committed",
                    lease_id,
                    record.generation,
                    "artifact committed and compute resource released",
                )
        self.save()

    def artifact_is_valid(
        self, lease_id: str, generation: int, *, allow_committed: bool = True
    ) -> bool:
        record = self.records.get(lease_id)
        if record is None or record.generation != generation:
            return False
        if record.state == "committed":
            return allow_committed
        if record.state != "active":
            return False
        return self.clock.now_ms() < record.expires_ms

    def lease_map(self, records: list[LeaseRecord]) -> dict[str, int]:
        return {record.lease_id: record.generation for record in records}

    def snapshot_lease_state(
        self, records: list[LeaseRecord]
    ) -> tuple[int, dict[str, int]]:
        if not records:
            return self.high_water_generation, {}
        return max(record.generation for record in records), self.lease_map(records)

    def summary(self) -> dict[str, Any]:
        return {
            "state_path": str(self.state_path),
            "worker_id": self.declaration.worker_id,
            "ttl_ms": self.declaration.ttl_ms,
            "restore_generation": self.declaration.restore_generation,
            "high_water_generation": self.high_water_generation,
            "rollback_error": self.rollback_error.to_dict() if self.rollback_error else None,
            "reclaimed": self.reclaimed,
            "leases": [
                record.to_dict() for _, record in sorted(self.records.items())
            ],
            "events": self.events,
        }
