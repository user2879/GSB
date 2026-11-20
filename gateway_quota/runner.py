from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .engine import QuotaEngine
from .errors import ErrorRecord, ReplayMismatchError, VersionMismatchError
from .logging import StructuredLogger
from .models import ParameterSnapshot, TraceEvent
from .replay import verify_replay
from .storage import SQLiteLedger


@dataclass
class RunnerConfig:
    ledger_path: str | Path
    initial_snapshot: ParameterSnapshot
    events: list[TraceEvent]
    reconfigurations: list[dict[str, Any]]
    batch_size: int = 5
    crash_after_seq: int | None = None


class ExperimentRunner:
    """Batch-oriented orchestration with recovery and replay verification."""

    def __init__(self, config: RunnerConfig, *, signal_broker: Any, logger: StructuredLogger):
        if config.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.config = config
        self.signal_broker = signal_broker
        self.logger = logger
        self.events_by_seq = {event.seq: event for event in config.events}
        if len(self.events_by_seq) != len(config.events):
            raise ValueError("trace contains duplicate event sequences")
        self.reconfigs_by_after = {
            int(item["after_event_seq"]): item["snapshot"] for item in config.reconfigurations
        }
        for after_seq, snapshot in sorted(self.reconfigs_by_after.items()):
            if snapshot.version <= config.initial_snapshot.version:
                raise VersionMismatchError(
                    "reconfiguration version must be greater than the initial parameter version",
                    path=f"reconfigurations[after_event_seq={after_seq}].version",
                    details={
                        "initial_version": config.initial_snapshot.version,
                        "reconfiguration_version": snapshot.version,
                    },
                )

    def run(self) -> dict[str, Any]:
        with SQLiteLedger(self.config.ledger_path) as store:
            store.initialize(self.config.initial_snapshot)
            latest = store.latest_batch()
            pending = store.pending_batch()
            replay_report: dict[str, Any] | None = None
            recovery_report: dict[str, Any] | None = None
            current_snapshot = self.config.initial_snapshot

            if latest is not None:
                replay_report = verify_replay(store)
                self.logger.emit(
                    "REPLAY_OK",
                    "ledger replay verification passed",
                    events_checked=replay_report["events_checked"],
                )
                recovery = store.recovery_state()
                assert recovery is not None
                current_snapshot = recovery["snapshot"]
                engine = QuotaEngine.from_restored(
                    snapshot=current_snapshot,
                    connections=recovery["connections"],
                    loans=recovery["loans"],
                    warnings=recovery["warnings"],
                    runtime=recovery["runtime"],
                    signal_broker=self.signal_broker,
                    parameter_history=recovery["parameter_history"],
                )
                active_connection_ids = [
                    connection.connection_id
                    for connection in recovery["connections"]
                    if connection.state in {"established", "idle"}
                ]
                if hasattr(self.signal_broker, "resync"):
                    reopened = self.signal_broker.resync(active_connection_ids)
                    if reopened:
                        self.logger.emit(
                            "LOOPBACK_RESYNC",
                            "loopback signal transports reopened for committed active connections",
                            connection_ids=reopened,
                        )
                self._validate_recovered_pool_rows(engine, recovery["pool_rows"], latest)
                start_seq = int(recovery["runtime"].get("last_event_seq", 0)) + 1
                recovery_report = {
                    "recovered_batch_id": int(latest["batch_id"]),
                    "recovered_through_seq": int(recovery["runtime"].get("last_event_seq", 0)),
                    "parameter_version": current_snapshot.version,
                    "pending_batch": None,
                    "unknown_connections": [],
                }
                if pending is not None:
                    start_seq = int(pending["start_seq"])
                    candidates = sorted(set(pending.get("candidate_connections", [])))
                    engine.set_unknown_connections(candidates)
                    recovery_report["pending_batch"] = {
                        "batch_id": int(pending["batch_id"]),
                        "start_seq": int(pending["start_seq"]),
                        "end_seq": int(pending["end_seq"]),
                    }
                    recovery_report["unknown_connections"] = candidates
                    self.logger.emit(
                        "E_RECOVERY_UNKNOWN",
                        "incomplete batch connections are unknown and must be established again",
                        batch_id=int(pending["batch_id"]),
                        start_seq=int(pending["start_seq"]),
                        end_seq=int(pending["end_seq"]),
                        connection_ids=candidates,
                        exit_report_available=False,
                    )
                self.logger.emit(
                    "RECOVERY_COMPLETE",
                    "restored from the latest complete batch",
                    **recovery_report,
                )
                batch_id = int(pending["batch_id"]) if pending else store.next_batch_id()
            else:
                if pending is not None:
                    raise VersionMismatchError(
                        "pending batch exists without any completed batch",
                        path="ledger.pending_batches",
                        details={"pending": pending},
                    )
                engine = QuotaEngine(current_snapshot, signal_broker=self.signal_broker)
                start_seq = 1
                batch_id = 1

            max_seq = max(self.events_by_seq) if self.events_by_seq else 0
            run_start_seq = start_seq
            committed_batches = 0
            committed_events = 0
            for after_seq, schedule in sorted(self.reconfigs_by_after.items()):
                if schedule.version > engine.parameter_version and after_seq < start_seq - 1:
                    raise VersionMismatchError(
                        "a required reconfiguration was skipped before recovery point",
                        path=f"reconfigurations[after_event_seq={after_seq}].version",
                        details={
                            "after_event_seq": after_seq,
                            "recovery_start_seq": start_seq,
                            "current_parameter_version": engine.parameter_version,
                            "reconfiguration_version": schedule.version,
                        },
                    )
            while start_seq <= max_seq:
                next_transition_seq = self._next_transition_after(start_seq - 1, engine.parameter_version)
                end_seq = min(
                    start_seq + self.config.batch_size - 1,
                    max_seq,
                    next_transition_seq if next_transition_seq is not None else max_seq,
                )
                batch_events = self._events_between(start_seq, end_seq)
                if not batch_events:
                    break
                start_version = engine.parameter_version
                candidate_connections = [
                    event.connection for event in batch_events if event.event_type == "establish"
                ]
                store.begin_batch(
                    batch_id=batch_id,
                    start_seq=start_seq,
                    end_seq=end_seq,
                    parameter_version=start_version,
                    candidate_connections=candidate_connections,
                )
                self.logger.emit(
                    "BATCH_STARTED",
                    "complete batch processing started",
                    batch_id=batch_id,
                    start_seq=start_seq,
                    end_seq=end_seq,
                    parameter_version=start_version,
                )
                results = []
                for event in batch_events:
                    result = engine.process_event(event)
                    results.append(result)
                    if self.config.crash_after_seq is not None and event.seq == self.config.crash_after_seq:
                        self.logger.emit(
                            "E_SIMULATED_KILL",
                            "process forced to terminate before batch commit; no exit report is produced",
                            batch_id=batch_id,
                            event_seq=event.seq,
                            pending_batch_persisted=True,
                            exit_report_available=False,
                        )
                        os._exit(137)
                transition_list: list[tuple[ParameterSnapshot, dict[str, Any]]] = []
                schedule = self.reconfigs_by_after.get(end_seq)
                if schedule is not None and schedule.version > engine.parameter_version:
                    transition = engine.apply_parameters(schedule, after_event_seq=end_seq)
                    transition_list.append((schedule, transition))
                store.commit_batch(
                    batch_id=batch_id,
                    start_seq=start_seq,
                    end_seq=end_seq,
                    start_parameter_version=start_version,
                    engine=engine,
                    results=results,
                    transitions=transition_list,
                )
                self._log_committed_results(results)
                for snapshot, transition in transition_list:
                    self.logger.emit(
                        "PARAMETER_VERSION_CHANGED",
                        "parameter view replaced and stale diagnostics detached",
                        version=snapshot.version,
                        previous_version=transition["previous_version"],
                        after_event_seq=transition["after_event_seq"],
                        affected_pools=transition["affected_pools"],
                        cleared_alert_records=transition["cleared_alert_records"],
                        invalidated_borrows=transition["invalidated_borrows"],
                        stale_legacy_borrows=transition["stale_legacy_borrows"],
                        invalidated_borrow_relations=transition["invalidated_borrow_relations"],
                        added_borrow_relations=transition["added_borrow_relations"],
                    )
                committed_batches += 1
                committed_events += len(results)
                batch_id += 1
                start_seq = end_seq + 1

            transitions_after_last = [
                (after_seq, schedule)
                for after_seq, schedule in sorted(self.reconfigs_by_after.items())
                if after_seq >= max(0, start_seq - 1) and schedule.version > engine.parameter_version
            ]
            for after_seq, schedule in transitions_after_last:
                if after_seq != max_seq:
                    raise VersionMismatchError(
                        "a reconfiguration after the trace cannot be positioned deterministically",
                        path="reconfigurations.after_event_seq",
                        details={"after_event_seq": after_seq, "last_event_seq": max_seq},
                    )
                start_version = engine.parameter_version
                store.begin_batch(
                    batch_id=batch_id,
                    start_seq=max_seq,
                    end_seq=max_seq,
                    parameter_version=start_version,
                    candidate_connections=[],
                )
                transition = engine.apply_parameters(schedule, after_event_seq=max_seq)
                store.commit_batch(
                    batch_id=batch_id,
                    start_seq=max_seq,
                    end_seq=max_seq,
                    start_parameter_version=start_version,
                    engine=engine,
                    results=[],
                    transitions=[(schedule, transition)],
                )
                self.logger.emit(
                    "PARAMETER_VERSION_CHANGED",
                    "trailing parameter view replaced without additional events",
                    version=schedule.version,
                    previous_version=transition["previous_version"],
                    after_event_seq=after_seq,
                    affected_pools=transition["affected_pools"],
                    cleared_alert_records=transition["cleared_alert_records"],
                    invalidated_borrows=transition["invalidated_borrows"],
                    stale_legacy_borrows=transition["stale_legacy_borrows"],
                    invalidated_borrow_relations=transition["invalidated_borrow_relations"],
                    added_borrow_relations=transition["added_borrow_relations"],
                )
                committed_batches += 1
                batch_id += 1

            final_state = {
                "parameter_version": engine.parameter_version,
                "pools": engine.pool_rows(),
                "borrows": engine.loan_rows(),
                "current_warnings": engine.current_warnings(),
                "last_event_seq": engine.last_event_seq,
            }
            final_replay_report = verify_replay(store)
            self.logger.emit(
                "REPLAY_FINAL_OK",
                "final full-ledger replay verification passed",
                events_checked=final_replay_report["events_checked"],
            )
            self.logger.emit(
                "RUN_COMPLETE",
                "all requested events are committed",
                committed_batches=committed_batches,
                committed_events=committed_events,
                last_event_seq=engine.last_event_seq,
                parameter_version=engine.parameter_version,
            )
            return {
                "code": "OK",
                "started_at_seq": run_start_seq,
                "committed_batches": committed_batches,
                "committed_events": committed_events,
                "last_event_seq": engine.last_event_seq,
                "replay": replay_report,
                "final_replay": final_replay_report,
                "recovery": recovery_report,
                "final_state": {
                    "parameter_version": engine.parameter_version,
                    "pools": engine.pool_rows(),
                    "loans": engine.loan_rows(),
                    "current_warnings": engine.current_warnings(),
                },
                "ledger_counts": store.counts(),
            }

    def _events_between(self, start_seq: int, end_seq: int) -> list[TraceEvent]:
        events: list[TraceEvent] = []
        for seq in range(start_seq, end_seq + 1):
            event = self.events_by_seq.get(seq)
            if event is None:
                raise VersionMismatchError(
                    "trace is missing a sequence required for recovery",
                    path=f"events[seq={seq}]",
                    details={"expected_seq": seq},
                )
            events.append(event)
        return events

    def _next_transition_after(self, seq: int, current_version: int) -> int | None:
        candidates = [
            after_seq
            for after_seq, snapshot in self.reconfigs_by_after.items()
            if after_seq >= seq and snapshot.version > current_version
        ]
        return min(candidates) if candidates else None

    def _log_committed_results(self, results: Iterable[Any]) -> None:
        for result in results:
            for issue in result.issues:
                self.logger.emit(
                    issue["code"],
                    issue["message"],
                    event_seq=issue.get("event_seq"),
                    pool_id=issue.get("pool_id"),
                    connection_id=issue.get("connection_id"),
                    severity=issue.get("severity"),
                    details=issue.get("details", {}),
                    committed=True,
                )
            for warning in result.warnings:
                payload = warning.to_dict()
                self.logger.emit(
                    payload["type"],
                    "waterline is at or below the declared alert threshold",
                    event_seq=payload["event_seq"],
                    pool_id=payload["pool_id"],
                    parameter_version=payload["parameter_version"],
                    remaining_connections=payload["remaining_connections"],
                    capacity_in_use=payload["capacity_in_use"],
                    active_connections=payload["active_connections"],
                    waterline_percentage_fixed=payload["waterline_percentage_fixed"],
                    percentage_scale=payload["percentage_scale"],
                    rounding=payload["rounding"],
                    arrival_rate=payload["arrival_rate"],
                    exhaustion_eta=payload["exhaustion_eta"],
                    committed=True,
                )

    @staticmethod
    def _validate_recovered_pool_rows(
        engine: QuotaEngine,
        persisted_rows: list[dict[str, Any]],
        latest_batch: dict[str, Any],
    ) -> None:
        expected = {row["pool_id"]: row for row in engine.pool_rows()}
        actual = {row["pool_id"]: row for row in persisted_rows}
        fields = (
            "own_active",
            "borrowed_in",
            "lent_out",
            "active_connections",
            "capacity_in_use",
            "remaining_connections",
            "warning",
        )
        for pool_id in sorted(set(expected) | set(actual)):
            if pool_id not in expected or pool_id not in actual:
                raise ReplayMismatchError(
                    ErrorRecord(
                        code="E_REPLAY",
                        message=f"recovered pool presence mismatch for {pool_id}",
                        path=f"ledger.batches[{latest_batch['batch_id']}].pool_state[{pool_id}]",
                        business_code="E_REPLAY",
                        details={"pool_id": pool_id, "field": "presence"},
                    )
                )
            for field in fields:
                if expected[pool_id][field] != actual[pool_id][field]:
                    raise ReplayMismatchError(
                        ErrorRecord(
                            code="E_REPLAY",
                            message=f"recovered state mismatch at batch {latest_batch['batch_id']}, pool {pool_id}, field {field}",
                            path=f"ledger.batches[{latest_batch['batch_id']}].pool_state[{pool_id}].{field}",
                            business_code="E_REPLAY",
                            details={
                                "batch_id": int(latest_batch["batch_id"]),
                                "pool_id": pool_id,
                                "field": field,
                                "ledger_value": actual[pool_id][field],
                                "recovered_value": expected[pool_id][field],
                                "basis": "connection and borrow ledger reconstruction",
                            },
                        )
                    )
