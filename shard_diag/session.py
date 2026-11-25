from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .canonical import canonical_json, sha256_digest
from .errors import DiagnosticError, ErrorCode
from .invalidation import InvalidationEngine, InvalidationReport, RecomputeResult
from .ledger import Ledger, RecoveryReport
from .leases import LeaseManager
from .routing import compare_route
from .runtime import DatabasePool, force_kill_process_group
from .scheduler import ProbeScheduler, SchedulerOutcome
from .spec import Event, Spec


def session_identifier(spec: Spec) -> str:
    material = sha256_digest({"spec_digest": spec.spec_digest(), "event_digest": spec.event_digest()})
    return "session-" + material.removeprefix("sha256:")[:20]


@dataclass(frozen=True, slots=True)
class BatchCommit:
    batch_id: str
    batch_index: int
    content_digest: str
    route_records: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "batch_index": self.batch_index,
            "content_digest": self.content_digest,
            "route_records": list(self.route_records),
        }


@dataclass(frozen=True, slots=True)
class SessionRunReport:
    session_id: str
    snapshot_generation: int
    snapshot_digest: str
    batches: tuple[BatchCommit, ...]
    route_records: tuple[dict[str, Any], ...]
    routing_bytes: bytes

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "snapshot_generation": self.snapshot_generation,
            "snapshot_digest": self.snapshot_digest,
            "batches": [batch.as_dict() for batch in self.batches],
            "routing_digest": sha256_digest(list(self.route_records)),
            "routing_bytes_sha256": "sha256:" + hashlib.sha256(self.routing_bytes).hexdigest(),
        }


@dataclass(frozen=True, slots=True)
class InvalidSubsetReprobe:
    report: InvalidationReport
    probed_event_seqs: tuple[int, ...]
    recompute: RecomputeResult
    route_records: tuple[dict[str, Any], ...]
    response_records: tuple[dict[str, Any], ...]
    unaffected_response_before_digest: str
    unaffected_response_after_digest: str
    unaffected_responses_byte_identical: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "invalidation": self.report.as_dict(),
            "probed_event_seqs": list(self.probed_event_seqs),
            "recomputed_record_keys": [list(key) for key in self.recompute.recomputed_record_keys],
            "unaffected_byte_identical": self.recompute.unaffected_byte_identical,
            "unaffected_before_digest": self.recompute.unaffected_before_digest,
            "unaffected_after_digest": self.recompute.unaffected_after_digest,
            "unaffected_response_before_digest": self.unaffected_response_before_digest,
            "unaffected_response_after_digest": self.unaffected_response_after_digest,
            "unaffected_responses_byte_identical": self.unaffected_responses_byte_identical,
            "route_records": list(self.route_records),
        }


class DiagnosticSession:
    """Integrated demo/API session spanning children, leases, and ledger."""

    def __init__(
        self,
        spec: Spec,
        spec_path: str | Path,
        ledger_path: str | Path,
        *,
        session_id: str | None = None,
        lease_ttl_ms: int = 5_000,
        socket_timeout_seconds: float = 2.0,
        startup_timeout_seconds: float = 3.0,
        allow_existing_snapshot: bool = True,
        replace_existing_snapshot: bool = False,
    ) -> None:
        self.spec = spec
        self.spec_path = str(Path(spec_path).resolve())
        self.ledger = Ledger(ledger_path)
        self.session_id = session_id or session_identifier(spec)
        self.lease_ttl_ms = lease_ttl_ms
        self.socket_timeout_seconds = socket_timeout_seconds
        self.startup_timeout_seconds = startup_timeout_seconds
        self._configure_runtime()
        self._install_or_validate_snapshot(
            allow_existing=allow_existing_snapshot,
            replace_existing=replace_existing_snapshot,
        )

    def _configure_runtime(self) -> None:
        self.pool = DatabasePool(
            self.spec,
            self.spec_path,
            startup_timeout_seconds=self.startup_timeout_seconds,
            socket_timeout_seconds=self.socket_timeout_seconds,
        )
        self.lease_manager = LeaseManager(
            global_lease_limit=self.spec.global_lease_limit,
            databases=self.spec.databases,
        )
        self.scheduler = ProbeScheduler(
            self.spec,
            self.lease_manager,
            self.pool.client,
            lease_ttl_ms=self.lease_ttl_ms,
        )

    def _snapshot_payload(self, generation: int) -> dict[str, Any]:
        return self.spec.snapshot(generation=generation)

    def _install_or_validate_snapshot(
        self,
        *,
        allow_existing: bool,
        replace_existing: bool,
    ) -> None:
        try:
            existing = self.ledger.get_snapshot(self.session_id)
        except DiagnosticError as error:
            if error.code != ErrorCode.STATE.value:
                raise
            existing = None
        if existing is None:
            installed = self.ledger.replace_snapshot(
                self.session_id,
                self._snapshot_payload(1),
                generation=1,
            )
            self.snapshot_generation = installed["generation"]
            self.snapshot_digest = installed["snapshot_digest"]
            return
        existing_payload = existing["payload"]
        same_snapshot = (
            existing_payload.get("spec_digest") == self.spec.spec_digest()
            and existing_payload.get("event_digest") == self.spec.event_digest()
        )
        if same_snapshot:
            self.snapshot_generation = int(existing["generation"])
            self.snapshot_digest = existing["snapshot_digest"]
            return
        if not allow_existing and not replace_existing:
            raise DiagnosticError(
                ErrorCode.STATE.value,
                "session id already has a different snapshot",
                field="snapshot_digest",
            )
        if replace_existing:
            self.replace_snapshot(self.spec, generation=int(existing["generation"]) + 1)
            return
        raise DiagnosticError(
            ErrorCode.STATE.value,
            "existing ledger snapshot differs from the supplied spec/event stream; use replace_snapshot",
            field="snapshot_digest",
            details={"session_id": self.session_id, "existing_generation": existing["generation"]},
        )

    @property
    def databases_started(self) -> bool:
        return all(runtime.running for runtime in self.pool._runtimes.values())

    def start_databases(self, order: list[str] | None = None) -> None:
        self.pool.start_all(order)

    def stop_databases(self, *, force: bool = False) -> None:
        self.pool.stop_all(force=force)

    def __enter__(self) -> "DiagnosticSession":
        self.start_databases()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        self.stop_databases()
        self.ledger.close()

    def _ensure_started(self) -> None:
        if not self.databases_started:
            self.start_databases()

    def _event_map(self) -> dict[int, Event]:
        return {event.seq: event for event in self.spec.events}

    def _build_payload(
        self,
        *,
        batch_id: str,
        batch_index: int,
        outcome: SchedulerOutcome,
        sample_count: int,
    ) -> dict[str, Any]:
        event_dicts = sorted(
            (copy.deepcopy(item.as_dict()) for item in outcome.event_outcomes),
            key=lambda item: (item["event_seq"], item["db_id"], item["table"]),
        )
        route_records = sorted(
            (copy.deepcopy(record) for record in outcome.route_records),
            key=lambda record: (record["event_seq"], record["db_id"], record["table"]),
        )
        leases = sorted(
            (
                copy.deepcopy(lease)
                for event in event_dicts
                for lease in event.get("leases", [])
            ),
            key=lambda lease: (
                lease["event_seq"],
                lease["db_id"],
                lease["shard_id"],
                lease["lease_id"],
            ),
        )
        transitions = sorted(
            (copy.deepcopy(transition) for transition in outcome.lease_transitions),
            key=lambda transition: (
                transition["event_seq"],
                transition["db_id"],
                transition["shard_id"],
                transition["lease_id"],
                transition["to_status"],
                transition["reason"],
            ),
        )
        quota_snapshot = copy.deepcopy(outcome.quota_snapshot)
        quota_snapshot["high_water"] = copy.deepcopy(outcome.high_water)
        quota_snapshot["wait_records"] = list(outcome.wait_records)
        event_seqs = [event["event_seq"] for event in event_dicts]
        return {
            "payload_format_version": 1,
            "session_id": self.session_id,
            "snapshot_digest": self.snapshot_digest,
            "batch_id": batch_id,
            "batch_index": batch_index,
            "start_seq": event_seqs[0],
            "end_seq": event_seqs[-1],
            "events": event_dicts,
            "route_records": route_records,
            "drift_records": copy.deepcopy(route_records),
            "leases": leases,
            "lease_transitions": transitions,
            "quota_snapshot": quota_snapshot,
            "wait_records": [copy.deepcopy(record) for record in outcome.wait_records],
            "high_water": copy.deepcopy(outcome.high_water),
            "probe_sample_count": sample_count,
            "progress": {"last_batch_index": batch_index, "last_seq": event_seqs[-1]},
        }

    def run_batches(
        self,
        events: Iterable[Event] | None = None,
        *,
        batch_size: int = 4,
        sample_count: int = 1,
        resume: bool = True,
        batch_index_start: int | None = None,
    ) -> SessionRunReport:
        if batch_size <= 0:
            raise DiagnosticError(ErrorCode.STATE.value, "batch_size must be positive", field="batch_size")
        self._ensure_started()
        selected = list(events) if events is not None else list(self.spec.events)
        event_map = self._event_map()
        for event in selected:
            if event.seq not in event_map or event_map[event.seq] != event:
                raise DiagnosticError(
                    ErrorCode.STATE.value,
                    "event does not belong to the installed snapshot",
                    field="event_seq",
                )
        selected.sort(key=lambda event: (event.seq, event.database, event.table))
        existing_records_before = self.read_route_records()
        progress = self.ledger.get_progress(self.session_id)
        if resume and progress is not None:
            selected = [event for event in selected if event.seq > int(progress["last_seq"])]
        next_index = (
            int(progress["last_batch_index"]) + 1
            if progress is not None and batch_index_start is None
            else (batch_index_start or 1)
        )
        commits: list[BatchCommit] = []
        all_records: list[dict[str, Any]] = []
        for offset in range(0, len(selected), batch_size):
            batch_events = selected[offset : offset + batch_size]
            if not batch_events:
                continue
            batch_index = next_index + len(commits)
            batch_id = self._batch_identifier(batch_index)
            intent = self.ledger.begin_batch(
                session_id=self.session_id,
                snapshot_digest=self.snapshot_digest,
                batch_id=batch_id,
                batch_index=batch_index,
                event_seqs=[event.seq for event in batch_events],
            )
            try:
                outcome = self.scheduler.run_batch(
                    batch_events,
                    batch_id=batch_id,
                    sample_count=sample_count,
                )
                payload = self._build_payload(
                    batch_id=batch_id,
                    batch_index=batch_index,
                    outcome=outcome,
                    sample_count=sample_count,
                )
                committed = self.ledger.commit_batch(payload)
            except DiagnosticError as error:
                self.ledger.abandon_batch(
                    batch_id,
                    reason=f"{error.code}: {error.message}",
                )
                self.lease_manager.release_events(
                    set(intent["event_seqs"]),
                    reason="uncommitted_batch_recovery",
                )
                raise
            except Exception as error:  # pragma: no cover - defensive process boundary
                self.ledger.abandon_batch(batch_id, reason=f"unexpected: {type(error).__name__}: {error}")
                self.lease_manager.release_events(
                    set(intent["event_seqs"]),
                    reason="uncommitted_batch_recovery",
                )
                raise
            records = tuple(copy.deepcopy(record) for record in payload["route_records"])
            all_records.extend(records)
            commits.append(
                BatchCommit(
                    batch_id=batch_id,
                    batch_index=batch_index,
                    content_digest=committed["content_digest"],
                    route_records=records,
                )
            )
        combined = sorted(
            [*existing_records_before, *all_records],
            key=lambda record: (record["event_seq"], record["db_id"], record["table"]),
        )
        return SessionRunReport(
            session_id=self.session_id,
            snapshot_generation=self.snapshot_generation,
            snapshot_digest=self.snapshot_digest,
            batches=tuple(commits),
            route_records=tuple(combined),
            routing_bytes=canonical_json(combined).encode("utf-8"),
        )

    def _batch_identifier(self, batch_index: int) -> str:
        digest_hex = self.snapshot_digest.removeprefix("sha256:")
        return f"batch-{digest_hex[:12]}-{batch_index:03d}"

    def force_kill_database(self, db_id: str) -> None:
        runtime = self.pool.runtime(db_id)
        force_kill_process_group(runtime)

    def recover_after_worker_restart(
        self,
        db_ids: Iterable[str] | None = None,
    ) -> RecoveryReport:
        targets = sorted(list(db_ids)) if db_ids is not None else sorted(self.pool._runtimes)
        for db_id in targets:
            runtime = self.pool.runtime(db_id)
            runtime.stop(force=True)
            runtime.start()
        report = self.ledger.recover_session(self.session_id)
        self.lease_manager.release_events(
            set(report.unknown_event_seqs),
            reason="uncommitted_batch_recovery",
        )
        return report

    def read_route_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for batch in self.ledger.read_batches(self.session_id):
            records.extend(copy.deepcopy(batch["payload"].get("route_records", [])))
        return sorted(records, key=lambda record: (record["event_seq"], record["db_id"], record["table"]))

    def read_lease_records(self) -> list[dict[str, Any]]:
        leases: list[dict[str, Any]] = []
        for batch in self.ledger.read_batches(self.session_id):
            leases.extend(copy.deepcopy(batch["payload"].get("leases", [])))
        return sorted(
            leases,
            key=lambda lease: (lease["event_seq"], lease["db_id"], lease["shard_id"], lease["lease_id"]),
        )

    def read_response_records(self) -> list[dict[str, Any]]:
        responses: list[dict[str, Any]] = []
        for batch in self.ledger.read_batches(self.session_id):
            responses.extend(copy.deepcopy(batch["payload"].get("events", [])))
        return sorted(
            responses,
            key=lambda response: (
                response["event_seq"],
                response["db_id"],
                response["table"],
            ),
        )

    def replay_ledger(self, *, require_complete: bool = True, sample_count: int | None = None) -> dict[str, Any]:
        self._ensure_started()
        stored_batches = self.ledger.read_batches(self.session_id)
        if not stored_batches:
            return self.ledger.replay_against(self.session_id, [], require_complete=require_complete)
        events_by_seq = self._event_map()
        replay_manager = LeaseManager(
            global_lease_limit=self.spec.global_lease_limit,
            databases=self.spec.databases,
        )
        replay_scheduler = ProbeScheduler(
            self.spec,
            replay_manager,
            self.pool.client,
            lease_ttl_ms=self.lease_ttl_ms,
        )
        recomputed: list[dict[str, Any]] = []
        for stored in stored_batches:
            payload = stored["payload"]
            event_seqs = [event["event_seq"] for event in payload["events"]]
            batch_events = [events_by_seq[seq] for seq in sorted(event_seqs)]
            detected_sample_count = sample_count or int(payload.get("probe_sample_count", 1))
            outcome = replay_scheduler.run_batch(
                batch_events,
                batch_id=stored["batch_id"],
                sample_count=detected_sample_count,
            )
            recomputed_payload = self._build_payload(
                batch_id=stored["batch_id"],
                batch_index=stored["batch_index"],
                outcome=outcome,
                sample_count=detected_sample_count,
            )
            recomputed_payload["snapshot_digest"] = payload.get("snapshot_digest", self.snapshot_digest)
            recomputed.append(recomputed_payload)
        return self.ledger.replay_against(
            self.session_id,
            recomputed,
            require_complete=require_complete,
        )

    def replace_snapshot(
        self,
        spec: Spec,
        *,
        spec_path: str | Path | None = None,
        generation: int | None = None,
    ) -> dict[str, Any]:
        self.stop_databases(force=True)
        self.spec = spec
        if spec_path is not None:
            self.spec_path = str(Path(spec_path).resolve())
        existing = self.ledger.get_snapshot(self.session_id)
        next_generation = generation if generation is not None else int(existing["generation"]) + 1
        installed = self.ledger.replace_snapshot(
            self.session_id,
            spec.snapshot(generation=next_generation),
            generation=next_generation,
        )
        self.snapshot_generation = installed["generation"]
        self.snapshot_digest = installed["snapshot_digest"]
        self._configure_runtime()
        return installed

    def invalidate(
        self,
        previous_rule,
        current_rule,
    ) -> InvalidationReport:
        engine = InvalidationEngine(
            self.read_route_records(),
            self.read_lease_records(),
            self.read_response_records(),
        )
        return engine.invalidate_table_version(previous_rule, current_rule)

    def reprobe_invalid_subset(
        self,
        report: InvalidationReport,
        new_spec: Spec,
        new_spec_path: str | Path,
        *,
        sample_count: int = 1,
    ) -> InvalidSubsetReprobe:
        """Stop this runtime, launch workers from the evolved spec, and probe only invalid events."""

        self.stop_databases(force=True)
        invalid_seqs = set(report.invalid_event_seqs)
        new_events = sorted(
            (event for event in new_spec.events if event.seq in invalid_seqs),
            key=lambda event: (event.seq, event.database, event.table),
        )
        probe_pool = DatabasePool(
            new_spec,
            new_spec_path,
            startup_timeout_seconds=self.startup_timeout_seconds,
            socket_timeout_seconds=self.socket_timeout_seconds,
        )
        probe_manager = LeaseManager(
            global_lease_limit=new_spec.global_lease_limit,
            databases=new_spec.databases,
        )
        probe_scheduler = ProbeScheduler(
            new_spec,
            probe_manager,
            probe_pool.client,
            lease_ttl_ms=self.lease_ttl_ms,
        )
        try:
            probe_pool.start_all()
            outcome = probe_scheduler.run_batch(
                new_events,
                batch_id="invalid-subset-reprobe",
                sample_count=sample_count,
            )
        finally:
            probe_pool.stop_all(force=True)
        old_responses = self.read_response_records()
        invalid_response_keys = set(report.invalid_response_keys)
        new_responses = [item.as_dict() for item in outcome.event_outcomes]
        new_responses_by_key = {
            (response["event_seq"], response["db_id"], response["table"]): response
            for response in new_responses
        }
        unaffected_before = [
            response
            for response in old_responses
            if (response["event_seq"], response["db_id"], response["table"]) not in invalid_response_keys
        ]
        final_responses = [
            (
                copy.deepcopy(new_responses_by_key[(response["event_seq"], response["db_id"], response["table"])])
                if (response["event_seq"], response["db_id"], response["table"]) in invalid_response_keys
                else copy.deepcopy(response)
            )
            for response in old_responses
        ]
        final_responses.sort(
            key=lambda response: (response["event_seq"], response["db_id"], response["table"])
        )
        unaffected_after = [
            response
            for response in final_responses
            if (response["event_seq"], response["db_id"], response["table"]) not in invalid_response_keys
        ]
        unaffected_response_before_digest = sha256_digest(unaffected_before)
        unaffected_response_after_digest = sha256_digest(unaffected_after)
        unaffected_responses_identical = (
            canonical_json(unaffected_before) == canonical_json(unaffected_after)
        )
        new_records_by_key = {
            (record["event_seq"], record["db_id"], record["table"], record["route_key"]): record
            for record in outcome.route_records
        }
        engine = InvalidationEngine(
            self.read_route_records(),
            self.read_lease_records(),
            old_responses,
        )

        def replacement(record: dict[str, Any]) -> dict[str, Any]:
            key = (record["event_seq"], record["db_id"], record["table"], record["route_key"])
            if key not in new_records_by_key:
                raise DiagnosticError(
                    ErrorCode.STATE.value,
                    "invalid-subset re-probe did not return a required record",
                    field="event_seq",
                    details={"record_key": list(key)},
                )
            return copy.deepcopy(new_records_by_key[key])

        recompute = engine.recompute_invalid_subset(report, replacement)
        return InvalidSubsetReprobe(
            report=report,
            probed_event_seqs=tuple(sorted(event.seq for event in new_events)),
            recompute=recompute,
            route_records=recompute.records,
            response_records=tuple(final_responses),
            unaffected_response_before_digest=unaffected_response_before_digest,
            unaffected_response_after_digest=unaffected_response_after_digest,
            unaffected_responses_byte_identical=unaffected_responses_identical,
        )
