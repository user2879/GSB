from __future__ import annotations

import threading
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any

from .errors import LeaseError, QuotaError
from .ledger import Ledger
from .spec import NodeSpec, SpecBundle
from .timeutil import monotonic_ms


@dataclass(frozen=True)
class Lease:
    lease_id: str
    quota_version: int
    lease_version: int
    node_id: str
    shard_id: str
    shard_seq: int
    slot_index: int
    byte_count: int
    granted_ms: int
    expires_ms: int
    renewal_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Acquisition:
    lease: Lease
    wait_started_ms: int
    wait_ended_ms: int

    @property
    def waited_ms(self) -> int:
        return max(0, self.wait_ended_ms - self.wait_started_ms)


class LeaseService:
    """Thread-safe lease/adjudication authority backed by SQLite."""

    def __init__(self, spec: SpecBundle, ledger: Ledger):
        self.spec = spec
        self.ledger = ledger
        self.condition = threading.Condition(threading.RLock())
        self._queue_starts: dict[tuple[str, str], int] = {}
        self.last_stale_counts: dict[str, int] = {}

    def _node_for_margin(self, node_id: str) -> NodeSpec:
        try:
            return self.spec.node(node_id)
        except Exception:
            return NodeSpec(
                node_id=node_id,
                host="127.0.0.1",
                port=0,
                capacity_slots=0,
                max_node_leases=0,
                seed=0,
            )

    def _active_rows_locked(self) -> list[Any]:
        return list(self.ledger.conn.execute("SELECT * FROM leases WHERE state='active'"))

    def reap_expired(self, now_ms: int | None = None) -> list[int]:
        now = monotonic_ms() if now_ms is None else now_ms
        rows = self.ledger.active_lease_rows()
        won: list[int] = []
        for row in rows:
            if int(row["expires_ms"]) > now:
                continue
            seq = self.ledger.terminalize_lease(
                row["lease_id"],
                new_state="expired",
                event_type="lease_expired",
                winner="expiry_reaper",
                basis="monotonic_now_ge_expires_at",
                occurred_ms=now,
                policy=self.spec.policy,
                node=self._node_for_margin(row["node_id"]),
                extra_payload={"observed_now_ms": now},
            )
            if seq is not None:
                won.append(seq)
                with self.condition:
                    self.condition.notify_all()
        return won

    def _can_fit_locked(self, node: NodeSpec, byte_count: int) -> int | None:
        active = self._active_rows_locked()
        global_active = len(active)
        if global_active >= self.spec.policy.global_max_active_leases:
            return None
        node_rows = [row for row in active if row["node_id"] == node.node_id]
        if len(node_rows) >= node.effective_lease_cap:
            return None
        shard_active = [row for row in active if row["shard_id"] == self._pending_shard]
        if shard_active:
            return None
        budget = self.ledger.conn.execute(
            "SELECT bytes_reserved,bytes_verified FROM budget_state WHERE budget_id=1"
        ).fetchone()
        if int(budget["bytes_verified"]) + int(budget["bytes_reserved"]) + byte_count > self.spec.policy.global_byte_budget:
            return None
        occupied = {int(row["slot_index"]) for row in node_rows}
        for slot in range(node.capacity_slots):
            if slot not in occupied:
                return slot
        return None

    _pending_shard: str = ""

    def acquire(
        self,
        node_id: str,
        shard_id: str,
        shard_seq: int,
        byte_count: int,
        *,
        ttl_ms: int | None = None,
        now_ms: int | None = None,
        queue_timeout_ms: int | None = None,
    ) -> Acquisition:
        node = self.spec.node(node_id)
        timeout = self.spec.policy.queue_timeout_ms if queue_timeout_ms is None else queue_timeout_ms
        deadline = time.monotonic() + max(0.001, timeout / 1000.0)
        wait_started = 0
        queue_key = (node_id, shard_id)

        with self.condition:
            while True:
                now = monotonic_ms() if now_ms is None else now_ms
                self.reap_expired(now)
                self._pending_shard = shard_id
                slot = self._can_fit_locked(node, byte_count)
                self._pending_shard = ""
                if slot is not None:
                    ttl = self.spec.policy.ttl_ms if ttl_ms is None else ttl_ms
                    lease = Lease(
                        lease_id="lease-" + uuid.uuid4().hex[:16],
                        quota_version=self.spec.quota_version,
                        lease_version=1,
                        node_id=node_id,
                        shard_id=shard_id,
                        shard_seq=shard_seq,
                        slot_index=slot,
                        byte_count=byte_count,
                        granted_ms=now,
                        expires_ms=now + ttl,
                        renewal_count=0,
                    )
                    self.ledger.record_grant(lease, node, self.spec.policy)
                    wait_ended = monotonic_ms()
                    if wait_started:
                        self.ledger.append_event(
                            "queue_wait_ended",
                            quota_version=lease.quota_version,
                            occurred_ms=wait_ended,
                            node_id=node_id,
                            shard_id=shard_id,
                            lease_id=lease.lease_id,
                            basis="quota_slot_available",
                            payload={
                                "wait_started_ms": wait_started,
                                "wait_ended_ms": wait_ended,
                                "waited_ms": max(0, wait_ended - wait_started),
                            },
                        )
                    self.condition.notify_all()
                    return Acquisition(lease, wait_started or wait_ended, wait_ended)

                if wait_started == 0:
                    wait_started = now
                    self._queue_starts[queue_key] = wait_started
                    self.ledger.append_event(
                        "queue_wait_started",
                        quota_version=self.spec.quota_version,
                        occurred_ms=wait_started,
                        node_id=node_id,
                        shard_id=shard_id,
                        basis="global_node_or_byte_quota_unavailable",
                        payload={"byte_count": byte_count},
                    )

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise QuotaError(
                        f"queue timeout waiting for lease {node_id}/{shard_id}",
                        field="quota",
                        shard_id=shard_id,
                    )
                self.condition.wait(timeout=min(0.02, remaining))

    def commit_renewal_result(
        self,
        lease: Lease,
        expected_version: int,
        *,
        now_ms: int | None = None,
    ) -> Lease:
        now = monotonic_ms() if now_ms is None else now_ms
        with self.condition:
            row = self.ledger.conn.execute(
                "SELECT * FROM leases WHERE lease_id=?", (lease.lease_id,)
            ).fetchone()
            if row is None or row["state"] != "active":
                raise LeaseError("lease is no longer active", field="state")
            if int(row["lease_version"]) != expected_version:
                raise LeaseError(
                    f"lease version mismatch: expected {expected_version}, actual {row['lease_version']}",
                    field="lease_version",
                )
            if int(row["expires_ms"]) <= now:
                self.reap_expired(now)
                raise LeaseError("lease has expired and cannot be renewed", field="expires_ms")
            if int(row["renewal_count"]) >= self.spec.policy.max_renewals:
                raise LeaseError("lease renewal limit reached", field="renewal_count")
            new_version = expected_version + 1
            renewal_count = int(row["renewal_count"]) + 1
            new_expires = now + self.spec.policy.ttl_ms
            node = self.spec.node(lease.node_id)
            self.ledger.record_renewal(
                lease_id=lease.lease_id,
                expected_version=expected_version,
                new_version=new_version,
                renewal_count=renewal_count,
                new_expires_ms=new_expires,
                occurred_ms=now,
                policy=self.spec.policy,
                node=node,
            )
            updated = Lease(
                **{
                    **lease.to_dict(),
                    "lease_version": new_version,
                    "renewal_count": renewal_count,
                    "granted_ms": int(row["granted_ms"]),
                    "expires_ms": new_expires,
                }
            )
            self.condition.notify_all()
            return updated

    def renew(
        self,
        lease: Lease,
        expected_version: int,
        node_client: Any | None = None,
        *,
        now_ms: int | None = None,
    ) -> Lease:
        if node_client is not None:
            node_client.renew_lease(lease.lease_id, expected_version, self.spec.policy.ttl_ms)
        return self.commit_renewal_result(lease, expected_version, now_ms=now_ms)

    def complete(self, lease: Lease, proof: Any, *, source_format: str = "v3", now_ms: int | None = None) -> int:
        now = monotonic_ms() if now_ms is None else now_ms
        node = self.spec.node(lease.node_id)
        seq = self.ledger.commit_result(
            lease,
            proof_json=proof if isinstance(proof, str) else _proof_json(proof),
            digest=proof["digest"] if isinstance(proof, dict) else proof.digest,
            content_digest=(
                proof["content_bytes_digest"] if isinstance(proof, dict) else proof.content_bytes_digest
            ),
            source_format=source_format,
            occurred_ms=now,
            policy=self.spec.policy,
            node=node,
        )
        if seq is None:
            raise LeaseError("lease completion lost CAS or version mismatch", field="state")
        with self.condition:
            self.condition.notify_all()
        return seq

    def recover_active_leases(self, *, now_ms: int | None = None) -> list[str]:
        now = monotonic_ms() if now_ms is None else now_ms
        recovered: list[str] = []
        for row in self.ledger.active_lease_rows():
            seq = self.ledger.terminalize_lease(
                row["lease_id"],
                new_state="recovery_reclaimed",
                event_type="recovery_reclaimed",
                winner="startup_recovery",
                basis="committed_grant_without_committed_result:capacity_reclaimed_shard_unknown",
                occurred_ms=now,
                policy=self.spec.policy,
                node=self._node_for_margin(row["node_id"]),
                mark_unknown=True,
            )
            if seq is not None:
                recovered.append(row["shard_id"])
        with self.condition:
            self.condition.notify_all()
        return recovered

    def revoke_for_quota_revision(
        self,
        new_spec: SpecBundle,
        *,
        reason: str = "quota_version_or_node_set_evolved",
        now_ms: int | None = None,
    ) -> list[int]:
        now = monotonic_ms() if now_ms is None else now_ms
        old_version = self.spec.quota_version
        new_version = new_spec.quota_version
        new_node_ids = {node.node_id for node in new_spec.nodes}
        self.ledger.record_quota_revision(
            old_version,
            new_version,
            now,
            reason,
            sorted(new_node_ids),
        )
        seqs: list[int] = []
        for row in self.ledger.active_lease_rows():
            lease_quota = int(row["quota_version"])
            node_removed = row["node_id"] not in new_node_ids
            old_epoch = lease_quota < new_version or reason == "node_set_evolved"
            if not (old_epoch or node_removed):
                continue
            basis = (
                "node_absent_from_new_declaration"
                if node_removed
                else f"lease_quota_version {lease_quota}<{new_version}"
            )
            seq = self.ledger.terminalize_lease(
                row["lease_id"],
                new_state="quota_revoked",
                event_type="lease_quota_revoked",
                winner="quota_revision",
                basis=basis,
                occurred_ms=now,
                policy=new_spec.policy,
                node=self._node_for_margin(row["node_id"]),
                event_quota_version=new_version,
                old_quota_version=lease_quota,
                competing_action="expiry_reaper",
            )
            if seq is not None:
                seqs.append(seq)
        with self.ledger.lock, self.ledger.conn:
            self.ledger.conn.execute(
                "UPDATE budget_state SET quota_version=?, bytes_reserved=0, bytes_verified=0 WHERE budget_id=1",
                (new_version,),
            )
        stale_counts = self.ledger.mark_quota_stale(old_version)
        self.spec = new_spec
        self.last_stale_counts = stale_counts
        with self.condition:
            self.condition.notify_all()
        return seqs


class ExpiryReaper(threading.Thread):
    def __init__(self, service: LeaseService, interval_ms: int | None = None):
        super().__init__(name="edge-expiry-reaper", daemon=True)
        self.service = service
        self.interval_s = (interval_ms or service.spec.policy.reaper_interval_ms) / 1000.0
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            self.service.reap_expired()
            self._stop.wait(self.interval_s)

    def stop(self) -> None:
        self._stop.set()


def _proof_json(proof: Any) -> str:
    from .canonical import canonical_dumps

    if hasattr(proof, "to_canonical"):
        return canonical_dumps(proof.to_canonical())
    return canonical_dumps(proof)
