"""End-to-end diagnostic session with resumable, batch-atomic progress."""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import load_json_strict
from .config import EvaluationConfig
from .declarations import load_declaration, normalize_declaration
from .drift import aggregate_all
from .errors import replay_error
from .events import apply_event, load_events
from .ledger import LedgerStore, make_batch_key
from .models import ObservationEvent, TargetSet
from .scheduler import schedule_decisions


@dataclass
class SessionResult:
    target: TargetSet
    events: list[ObservationEvent]
    snapshot_install: dict[str, Any]
    recovered_unknown: list[dict[str, Any]]
    batches: list[dict[str, Any]]
    latest_batch: dict[str, Any] | None
    final_actual_state: dict[str, dict[str, Any]]
    overall_converged: bool
    replay_verified: bool
    replay_report: dict[str, Any]
    ledger_summary: dict[str, Any]

    def to_dict(self, *, include_trace: bool = True) -> dict[str, Any]:
        latest = self.latest_batch
        result = {
            "ok": True,
            "target": self.target.to_dict(),
            "normalization": {
                "format_versions": list(self.target.declaration_format_versions),
                "defaults_version": self.target.defaults_version,
                "declaration_digest": self.target.declaration_digest,
                "target_digest": self.target.target_digest,
                "filled_fields": {
                    workload.name: {
                        "fields": list(workload.filled_fields),
                        "basis": workload.fill_basis,
                    }
                    for workload in self.target.ordered_workloads()
                },
            },
            "snapshot_install": self.snapshot_install,
            "recovered_unknown_workloads": self.recovered_unknown,
            "overall_converged": self.overall_converged,
            "replay_verified": self.replay_verified,
            "replay_report": self.replay_report,
            "final_actual_state": {
                name: {field: self.final_actual_state[name][field] for field in sorted(self.final_actual_state[name])}
                for name in sorted(self.final_actual_state)
            },
            "batch_count": len(self.batches),
            "batches": [
                {
                    "batch_id": batch["batch_id"],
                    "batch_key": batch["batch_key"],
                    "event_cursor": batch["event_cursor"],
                    "first_event_seq": batch["first_event_seq"],
                    "batch_kind": batch["batch_kind"],
                    "content_digest": batch["content_digest"],
                    "overall_converged": batch["workload_metrics"]["overall_converged"],
                }
                for batch in self.batches
            ],
            "batch_schedules": [
                {
                    "batch_id": batch["batch_id"],
                    "event_cursor": batch["event_cursor"],
                    "schedule_trace": batch["trace"],
                }
                for batch in self.batches
            ]
            if include_trace
            else [],
            "latest_event_cursor": latest["event_cursor"] if latest else None,
            "ledger_summary": self.ledger_summary,
        }
        if latest is not None:
            result["decisions"] = latest["decisions"]
            result["drift_decisions"] = [decision for decision in latest["decisions"] if decision["drift"]]
            result["workload_metrics"] = latest["workload_metrics"]
            if include_trace:
                result["schedule_trace"] = latest["trace"]
        return result


def run_session(
    *,
    declaration_path: str | Path,
    events_path: str | Path,
    db_path: str | Path,
    defaults_version: str = "defaults-v1",
    config: EvaluationConfig | None = None,
    crash_after_attempt: bool = False,
) -> SessionResult:
    config = (config or EvaluationConfig()).validate()
    raw_declaration = load_declaration(declaration_path)
    target = normalize_declaration(raw_declaration.raw, defaults_version=defaults_version)
    events = load_events(events_path, target)

    with LedgerStore(db_path) as ledger:
        recovered_unknown = ledger.recover_inflight()
        install = ledger.install_snapshot(
            target,
            raw_declaration.raw,
            scale_version=config.scale_version,
            scale_exp=config.scale_exp,
            tolerance_num=config.tolerance_num,
            tolerance_den=config.tolerance_den,
        )
        ledger.verify_integrity()
        batches = ledger.read_batches()
        latest = batches[-1] if batches else None
        cursor = latest["event_cursor"] if latest else 0
        if cursor < 0 or cursor > len(events):
            raise replay_error(
                "ledger event cursor is outside the supplied event stream",
                item=cursor,
                field="event_cursor",
                details={"event_count": len(events)},
            )

        replay_report: dict[str, Any] | None = None
        if batches:
            from .replay import verify_replay

            replay_report = verify_replay(ledger, target, events, config).to_dict()

        actual_state: dict[str, dict[str, Any]] = {}
        for event in events[:cursor]:
            apply_event(actual_state, event)

        if latest is None:
            _run_batch(
                ledger=ledger,
                target=target,
                actual_state=actual_state,
                config=config,
                event_cursor=0,
                first_event_seq=0,
                batch_kind="initial",
                crash_after_attempt=crash_after_attempt,
            )
            batches = ledger.read_batches()
            latest = batches[-1]
            cursor = 0

        while cursor < len(events):
            chunk = events[cursor : cursor + config.batch_events]
            for event in chunk:
                apply_event(actual_state, event)
            cursor = chunk[-1].seq
            _run_batch(
                ledger=ledger,
                target=target,
                actual_state=actual_state,
                config=config,
                event_cursor=cursor,
                first_event_seq=chunk[0].seq,
                batch_kind="events",
                crash_after_attempt=crash_after_attempt,
            )
            batches = ledger.read_batches()
            latest = batches[-1]

        ledger.verify_integrity()
        batches = ledger.read_batches()
        latest = batches[-1] if batches else None
        if latest is None:
            raise AssertionError("session produced no diagnostic batch")
        overall = bool(latest["workload_metrics"]["overall_converged"])
        from .replay import verify_replay

        replay_report = verify_replay(ledger, target, events, config).to_dict()
        return SessionResult(
            target=target,
            events=events,
            snapshot_install=install,
            recovered_unknown=recovered_unknown,
            batches=batches,
            latest_batch=latest,
            final_actual_state=deepcopy(actual_state),
            overall_converged=overall,
            replay_verified=True,
            replay_report=replay_report,
            ledger_summary=ledger.ledger_summary(),
        )


def _run_batch(
    *,
    ledger: LedgerStore,
    target: TargetSet,
    actual_state: dict[str, dict[str, Any]],
    config: EvaluationConfig,
    event_cursor: int,
    first_event_seq: int,
    batch_kind: str,
    crash_after_attempt: bool,
) -> dict[str, Any]:
    snapshot = ledger.read_snapshot()
    if snapshot is None:
        raise AssertionError("snapshot must be installed before running a batch")
    batch_key = make_batch_key(snapshot["snapshot_digest"], event_cursor)
    attempt_id = ledger.begin_attempt(
        batch_key=batch_key,
        workloads=sorted(target.workloads),
        plan={
            "event_cursor": event_cursor,
            "first_event_seq": first_event_seq,
            "batch_kind": batch_kind,
            "global_limit": config.global_limit,
            "per_workload_quota": config.per_workload_quota,
            "scale_exp": config.scale_exp,
        },
    )
    if crash_after_attempt:
        # Deliberately skip cleanup and the SQLite batch transaction. A real
        # forced termination would have the same externally visible effect.
        os._exit(91)

    decisions, trace = schedule_decisions(target, actual_state, config)
    workload_metrics = aggregate_all(target, decisions, config)
    return ledger.commit_batch(
        event_cursor=event_cursor,
        first_event_seq=first_event_seq,
        batch_kind=batch_kind,
        decisions=decisions,
        workload_metrics=workload_metrics,
        trace=trace,
        attempt_id=attempt_id,
    )


def load_change_set_file(path: str | Path) -> dict[str, Any]:
    raw = load_json_strict(path)
    if not isinstance(raw, dict):
        from .errors import format_error

        raise format_error("change set root must be an object", field="<root>")
    return raw
