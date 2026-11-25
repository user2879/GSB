from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

from .canonical import canonical_json, monotonic_ms, sha256_digest
from .errors import DiagnosticError, ErrorCode
from .leases import Lease, LeaseManager, LeaseRequest
from .protocol import WorkerClient
from .routing import compare_route, generate_sample_keys, sample_hit_shards
from .spec import Event, Spec, TableRule


@dataclass(frozen=True, slots=True)
class EventProbeOutcome:
    event_seq: int
    db_id: str
    table: str
    route_key: int
    route_record: dict[str, Any]
    hit_shards: tuple[str, ...]
    sample_keys: tuple[int, ...]
    worker_route_version: int
    worker_elapsed_ms: float
    leases: tuple[dict[str, Any], ...]
    thread_name: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_seq": self.event_seq,
            "db_id": self.db_id,
            "table": self.table,
            "route_key": self.route_key,
            "route_record": self.route_record,
            "hit_shards": list(self.hit_shards),
            "sample_keys": list(self.sample_keys),
            "worker_route_version": self.worker_route_version,
            "worker_elapsed_ms": self.worker_elapsed_ms,
            "leases": list(self.leases),
            "thread_name": self.thread_name,
        }


@dataclass(frozen=True, slots=True)
class SchedulerOutcome:
    batch_id: str
    event_outcomes: tuple[EventProbeOutcome, ...]
    wait_records: tuple[dict[str, Any], ...]
    lease_transitions: tuple[dict[str, Any], ...]
    quota_snapshot: dict[str, Any]
    high_water: dict[str, Any]

    @property
    def route_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(outcome.route_record for outcome in self.event_outcomes)

    def canonical_routing_bytes(self) -> bytes:
        return canonical_json(list(self.route_records)).encode("utf-8")

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "events": [outcome.as_dict() for outcome in self.event_outcomes],
            "wait_records": list(self.wait_records),
            "lease_transitions": list(self.lease_transitions),
            "quota_snapshot": self.quota_snapshot,
            "high_water": self.high_water,
        }


class ProbeScheduler:
    """Parallel scheduler constrained by global, per-db, and in-flight limits."""

    def __init__(
        self,
        spec: Spec,
        lease_manager: LeaseManager,
        client_for_db: Callable[[str], WorkerClient],
        *,
        lease_ttl_ms: int = 5_000,
        clock: Callable[[], int] = monotonic_ms,
    ) -> None:
        self.spec = spec
        self.lease_manager = lease_manager
        self.client_for_db = client_for_db
        self.lease_ttl_ms = lease_ttl_ms
        self.clock = clock
        self._condition = threading.Condition()
        self._inflight_by_db = {database.id: 0 for database in spec.databases}
        self._inflight = 0
        self.last_wait_records: list[dict[str, Any]] = []
        self._high_water = {
            "max_active_global": 0,
            "max_inflight": 0,
            "max_inflight_by_db": {database.id: 0 for database in spec.databases},
            "max_active_by_db": {database.id: 0 for database in spec.databases},
        }

    def _reserve_one(
        self,
        *,
        event: Event,
        requests: list[LeaseRequest],
        db_concurrency: int,
    ) -> tuple[list[Lease], dict[str, Any] | None]:
        wait_started: int | None = None
        reasons: list[str] = []
        with self._condition:
            while True:
                self.lease_manager.expire_due()
                reason: str | None = None
                if self._inflight_by_db[event.database] >= db_concurrency:
                    reason = (
                        f"database {event.database!r} in-flight concurrency "
                        f"{self._inflight_by_db[event.database]}/{db_concurrency} is exhausted"
                    )
                else:
                    available, lease_reason = self.lease_manager.can_acquire_many(requests)
                    if not available:
                        reason = lease_reason
                if reason is None:
                    self._inflight_by_db[event.database] += 1
                    self._inflight += 1
                    try:
                        leases = self.lease_manager.acquire_many(requests, ttl_ms=self.lease_ttl_ms)
                    except DiagnosticError:
                        self._inflight_by_db[event.database] -= 1
                        self._inflight -= 1
                        raise
                    snapshot = self.lease_manager.snapshot()
                    self._high_water["max_active_global"] = max(
                        self._high_water["max_active_global"], snapshot["active_leases"]
                    )
                    self._high_water["max_inflight"] = max(
                        self._high_water["max_inflight"], self._inflight
                    )
                    for db_id in self._high_water["max_inflight_by_db"]:
                        self._high_water["max_inflight_by_db"][db_id] = max(
                            self._high_water["max_inflight_by_db"][db_id],
                            self._inflight_by_db[db_id],
                        )
                        self._high_water["max_active_by_db"][db_id] = max(
                            self._high_water["max_active_by_db"][db_id],
                            snapshot["by_db"][db_id]["active_leases"],
                        )
                    if wait_started is not None:
                        wait_ended = self.clock()
                        wait_record = {
                            "event_seq": event.seq,
                            "db_id": event.database,
                            "wait_started_ms": wait_started,
                            "wait_ended_ms": wait_ended,
                            "wait_ms": wait_ended - wait_started,
                            "reasons": sorted(set(reasons)),
                        }
                    else:
                        wait_record = None
                    return leases, wait_record

                if wait_started is None:
                    wait_started = self.clock()
                if reason not in reasons:
                    reasons.append(reason)
                timeout = 0.05
                earliest = self.lease_manager.earliest_expiry_ms()
                if earliest is not None:
                    timeout = min(timeout, max(0.001, (earliest - self.clock()) / 1000.0))
                self._condition.wait(timeout=timeout)

    def _finish_inflight(self, db_id: str) -> None:
        with self._condition:
            self._inflight_by_db[db_id] -= 1
            self._inflight -= 1
            self._condition.notify_all()

    def _run_one(
        self,
        event: Event,
        rule: TableRule,
        previous_rule: TableRule | None,
        *,
        batch_id: str,
        sample_count: int,
    ) -> EventProbeOutcome:
        database = self.spec.database(event.database)
        sample_keys = generate_sample_keys(
            seed=database.seed,
            event_seq=event.seq,
            route_key=event.route_key,
            rule=rule,
            sample_count=sample_count,
        )
        hit_shards = sample_hit_shards(rule, sample_keys)
        requests = [
            LeaseRequest(event.database, shard_id, event.seq, database.route_version)
            for shard_id in hit_shards
        ]
        leases, wait_record = self._reserve_one(
            event=event,
            requests=requests,
            db_concurrency=database.concurrency,
        )
        client = self.client_for_db(event.database)
        probe_succeeded = False
        common: dict[str, Any] | None = None
        thread_name = threading.current_thread().name
        try:
            for lease in leases:
                client.grant_lease(
                    lease_id=lease.lease_id,
                    shard_id=lease.shard_id,
                    event_seq=lease.event_seq,
                    lease_version=lease.lease_version,
                )
            result = client.probe_batch(
                [
                    {
                        "event_seq": event.seq,
                        "route_key": event.route_key,
                        "table": event.table,
                        "route_version": database.route_version,
                        "sample_count": sample_count,
                    }
                ],
                batch_id=batch_id,
            )[0]
            returned_hits = tuple(sorted(result["hit_shards"]))
            if returned_hits != hit_shards:
                raise DiagnosticError(
                    ErrorCode.PROTO.value,
                    f"worker hit shards {returned_hits} do not match reserved shards {hit_shards}",
                    db_id=event.database,
                    field="hit_shards",
                )
            if result.get("sample_keys") and tuple(result["sample_keys"]) != sample_keys:
                raise DiagnosticError(
                    ErrorCode.PROTO.value,
                    "worker sample keys do not match deterministic parent prediction",
                    db_id=event.database,
                    field="sample_keys",
                )
            route_record = compare_route(
                event_seq=event.seq,
                db_id=event.database,
                table=event.table,
                key_column=rule.key_column,
                route_key=event.route_key,
                previous_rule=previous_rule,
                current_rule=rule,
            )
            if route_record["new_shard"] not in returned_hits:
                raise DiagnosticError(
                    ErrorCode.PROTO.value,
                    "worker response does not include the event key's current shard",
                    db_id=event.database,
                    field="hit_shards",
                )
            common = {
                "event_seq": event.seq,
                "db_id": event.database,
                "table": event.table,
                "route_key": event.route_key,
                "route_record": route_record,
                "hit_shards": returned_hits,
                "sample_keys": sample_keys,
                "worker_route_version": int(result["route_version"]),
                "worker_elapsed_ms": float(result["elapsed_ms"]),
            }
            probe_succeeded = True
        finally:
            for lease in leases:
                try:
                    self.lease_manager.release(lease.lease_id)
                except DiagnosticError:
                    pass
            if probe_succeeded:
                for lease in leases:
                    try:
                        client.revoke_lease(
                            lease_id=lease.lease_id,
                            shard_id=lease.shard_id,
                            event_seq=lease.event_seq,
                        )
                    except DiagnosticError:
                        pass
            self._finish_inflight(event.database)
            if wait_record is not None:
                self.last_wait_records.append(wait_record)
        if common is None:  # pragma: no cover - defensive invariant
            raise DiagnosticError(ErrorCode.STATE.value, "probe completed without an outcome")
        return EventProbeOutcome(
            **common,
            leases=tuple(lease.as_dict() for lease in leases),
            thread_name=thread_name,
        )

    def run_batch(
        self,
        events: list[Event],
        *,
        batch_id: str,
        sample_count: int = 1,
    ) -> SchedulerOutcome:
        if not events:
            raise DiagnosticError(ErrorCode.STATE.value, "cannot probe an empty batch", field="events")
        if sample_count < 1:
            raise DiagnosticError(ErrorCode.PROTO.value, "sample_count must be >= 1", field="sample_count")
        ordered = sorted(events, key=lambda event: (event.seq, event.database, event.table))
        self.last_wait_records = []
        starting_transition_count = len(self.lease_manager.transitions())
        self._high_water = {
            "max_active_global": 0,
            "max_inflight": 0,
            "max_inflight_by_db": {database.id: 0 for database in self.spec.databases},
            "max_active_by_db": {database.id: 0 for database in self.spec.databases},
        }
        max_workers = max(
            1,
            min(
                self.spec.global_lease_limit,
                sum(database.concurrency for database in self.spec.databases),
                len(ordered),
            ),
        )
        outcomes: list[EventProbeOutcome | None] = [None] * len(ordered)
        errors: list[tuple[int, DiagnosticError]] = []
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="shard-probe") as executor:
            futures = [
                executor.submit(
                    self._run_one,
                    event,
                    self.spec.event_rule(event),
                    self.spec.previous_rule(event.table, self.spec.database(event.database).route_version),
                    batch_id=batch_id,
                    sample_count=sample_count,
                )
                for event in ordered
            ]
            for index, future in enumerate(futures):
                try:
                    outcomes[index] = future.result()
                except DiagnosticError as error:
                    errors.append((ordered[index].seq, error))
        if errors:
            errors.sort(key=lambda item: item[0])
            raise errors[0][1]
        finished_outcomes = tuple(outcome for outcome in outcomes if outcome is not None)
        transitions = self.lease_manager.transitions()
        quota_snapshot = self.lease_manager.snapshot()
        return SchedulerOutcome(
            batch_id=batch_id,
            event_outcomes=finished_outcomes,
            wait_records=tuple(sorted(self.last_wait_records, key=lambda item: (item["event_seq"], item["db_id"]))),
            lease_transitions=tuple(transitions[starting_transition_count:]),
            quota_snapshot=quota_snapshot,
            high_water={
                "max_active_global": self._high_water["max_active_global"],
                "max_inflight": self._high_water["max_inflight"],
                "max_inflight_by_db": dict(self._high_water["max_inflight_by_db"]),
                "max_active_by_db": dict(self._high_water["max_active_by_db"]),
            },
        )

    def renew_lease(
        self,
        lease_id: str,
        *,
        expected_lease_version: int,
        client: WorkerClient,
        ttl_ms: int | None = None,
    ) -> Lease:
        return self.lease_manager.renew_with_worker(
            lease_id,
            expected_lease_version=expected_lease_version,
            ttl_ms=self.lease_ttl_ms if ttl_ms is None else ttl_ms,
            client=client,
        )


def routing_records_digest(records: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> str:
    ordered = sorted(records, key=lambda record: (record["event_seq"], record["db_id"], record["table"]))
    return sha256_digest(ordered)
