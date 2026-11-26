"""End-to-end orchestration of workers, leases, checkpoints, scheduling and ledger."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .checkpoint import CheckpointManager
from .errors import DiagnosticError
from .leases import LeaseManager, LeaseRecord, monotonic_ms
from .ledger import BatchLedgerRecord, Ledger
from .model import BatchSpec, EventSpec, SessionSpec
from .params import compute_batch_summary, get_parameter_set
from .process import WorkerPool
from .scheduler import (
    QuotaScheduler,
    ScheduledOutcome,
    SchedulerJob,
)


@dataclass(frozen=True)
class _WorkItem:
    sequence: int
    kind: str
    payload: BatchSpec | EventSpec


def _ordered_work(session: SessionSpec) -> list[_WorkItem]:
    items = [
        _WorkItem(batch.order, "batch", batch) for batch in session.batches
    ] + [_WorkItem(event.seq, "event", event) for event in session.events]
    # At an equal sequence, a control event is observed before the batch.
    return sorted(items, key=lambda item: (item.sequence, 0 if item.kind == "event" else 1))


class DiagnosticEngine:
    def __init__(
        self,
        session: SessionSpec,
        ledger: Ledger,
        pool: WorkerPool,
        *,
        lease_manager: LeaseManager | None = None,
        scheduler: QuotaScheduler | None = None,
        max_workers: int | None = None,
        clock=monotonic_ms,
    ) -> None:
        self.session = session
        self.ledger = ledger
        self.pool = pool
        self.clock = clock
        self.checkpoints = CheckpointManager(
            session,
            lambda shard_id, version: get_parameter_set(
                session.shard(shard_id), version
            ).digest,
        )
        self.lease_manager = lease_manager or self._build_lease_manager()
        self.scheduler = scheduler or QuotaScheduler(
            session, self.lease_manager, clock=clock, max_workers=max_workers
        )
        self.event_results: list[dict[str, Any]] = []

    def _build_lease_manager(self) -> LeaseManager:
        def remote_renew(shard_id: str, lease_id: str, version: int, ttl_ms: int) -> dict[str, Any]:
            return self.pool.client(shard_id).lease_renew(
                shard_id, lease_id, version, ttl_ms
            )

        def remote_release(shard_id: str, lease_id: str) -> dict[str, Any]:
            return self.pool.client(shard_id).lease_release(shard_id, lease_id)

        return LeaseManager(
            clock=self.clock,
            remote_renew=remote_renew,
            remote_release=remote_release,
        )

    def _issue_lease(self, job: SchedulerJob) -> LeaseRecord:
        shard = job.shard
        # LeaseManager assigns a unique deterministic id; remote issue is done by it.
        return self.lease_manager.issue(
            shard.shard_id,
            shard.lease_ttl_ms,
            owner=f"batch:{job.batch.batch_id}",
            remote_issue=lambda shard_id, lease_id, ttl: self.pool.client(
                shard_id
            ).lease_issue(shard_id, lease_id, ttl),
        )

    def _execute(
        self, job: SchedulerJob, lease: LeaseRecord
    ) -> dict[str, Any]:
        response = self.pool.client(job.shard.shard_id).run_batch(
            shard_id=job.shard.shard_id,
            batch_id=job.batch.batch_id,
            sample_ids=job.batch.sample_ids,
            parameter_version=job.parameter_version,
            checkpoint_id=job.batch.checkpoint_id,
        )
        expected = compute_batch_summary(
            job.shard,
            job.batch.batch_id,
            job.batch.sample_ids,
            job.parameter_version,
        )
        for field in ("parameter_digest", "loss", "loss_digest", "sample_ids"):
            actual = response.get(field)
            wanted = expected[field]
            if field == "sample_ids":
                actual = sorted(int(value) for value in actual)
                wanted = sorted(int(value) for value in wanted)
            if actual != wanted:
                raise DiagnosticError(
                    "E_REPLAY",
                    f"worker result differs from deterministic local result for field {field}",
                    details={
                        "batch_id": job.batch.batch_id,
                        "shard_id": job.shard.shard_id,
                        "field": field,
                        "expected": wanted,
                        "actual": actual,
                    },
                )
        elapsed = response.get("elapsed_ms")
        if isinstance(elapsed, bool) or not isinstance(elapsed, int) or elapsed < 0:
            raise DiagnosticError(
                "E_FIELD", "worker elapsed_ms must be a non-negative integer", field="elapsed_ms"
            )
        return response

    def _on_start(self, outcome: ScheduledOutcome) -> None:
        assert outcome.lease is not None
        self.ledger.start_batch(
            outcome.job, outcome.lease.to_dict(), now_ms=self.clock()
        )

    def _on_success(self, outcome: ScheduledOutcome) -> None:
        assert outcome.response is not None and outcome.lease is not None
        job = outcome.job
        response = outcome.response
        self.checkpoints.record_batch(
            sequence=job.batch.order,
            shard_id=job.shard.shard_id,
            batch_id=job.batch.batch_id,
            loss_digest=str(response["loss_digest"]),
        )
        checkpoint_state = self.checkpoints.state_for_shard(job.shard.shard_id)
        now = self.clock()
        record = BatchLedgerRecord(
            sequence=job.batch.order,
            batch_id=job.batch.batch_id,
            shard_id=job.shard.shard_id,
            batch_order=job.batch.order,
            parameter_version=int(response["parameter_version"]),
            parameter_digest=str(response["parameter_digest"]),
            loss_value=str(response["loss"]),
            loss_digest=str(response["loss_digest"]),
            sample_ids=tuple(sorted(int(value) for value in response["sample_ids"])),
            lease_id=outcome.lease.lease_id,
            lease_version=outcome.lease.lease_version,
            lease_owner=outcome.lease.owner,
            checkpoint_state=checkpoint_state,
            progress_cursor=job.batch.order,
            committed_at_ms=now,
        )
        outcome.thread_name = response.get("_thread_name")
        self.ledger.commit_batch(record)

    def _on_error(self, outcome: ScheduledOutcome) -> None:
        error = outcome.error or DiagnosticError("E_PROTOCOL", "batch failed")
        self.ledger.mark_unknown(
            batch_id=outcome.job.batch.batch_id,
            shard_id=outcome.job.shard.shard_id,
            batch_order=outcome.job.batch.order,
            code=error.code,
            message=error.message,
            now_ms=self.clock(),
        )

    def _already_committed(self) -> dict[str, BatchLedgerRecord]:
        return {record.batch_id: record for record in self.ledger.committed_records()}

    def _apply_event(self, event: EventSpec) -> None:
        payload: dict[str, Any] = {"event": event.to_dict()}
        try:
            if event.type in {"commit", "rollback"}:
                state = self.checkpoints.apply_event(event)
                payload["checkpoint"] = state.to_dict() if state is not None else None
            elif event.type == "renew":
                assert event.lease_id is not None and event.lease_version is not None
                lease = self.lease_manager.renew(
                    event.lease_id, event.lease_version, ttl_ms=event.ttl_ms
                )
                payload["lease"] = lease.to_dict()
            elif event.type == "cancel":
                assert event.lease_id is not None
                lease = self.lease_manager.cancel(event.lease_id)
                payload["lease"] = lease.to_dict()
            else:  # pragma: no cover - validation prevents this
                raise DiagnosticError("E_SPEC", f"unsupported event type {event.type}")
            payload["ok"] = True
        except DiagnosticError as exc:
            payload["ok"] = False
            payload["error"] = exc.to_dict()
        self.event_results.append(payload)
        manager_state = self.checkpoints.snapshot()
        self.ledger.record_checkpoint_state(
            self.checkpoints.state_for_shard(event.shard_id), now_ms=self.clock()
        )
        self.ledger.record_event(
            seq=event.seq,
            event_type=event.type,
            payload=payload,
            state_digest=manager_state["state_digest"],
            now_ms=self.clock(),
        )
        self.ledger.save_runtime_state(manager_state, now_ms=self.clock())

    def run(self) -> dict[str, Any]:
        recovery = self.ledger.recover(now_ms=self.clock())
        for shard in self.session.shards:
            self.ledger.ensure_checkpoint_state(
                self.checkpoints.state_for_shard(shard.shard_id), now_ms=self.clock()
            )
        committed = self._already_committed()
        reports: list[dict[str, Any]] = []
        segment: list[SchedulerJob] = []
        items = _ordered_work(self.session)

        def flush_segment() -> None:
            nonlocal segment
            if not segment:
                return
            jobs = sorted(segment, key=lambda job: job.sort_key)
            runnable: list[SchedulerJob] = []
            for job in jobs:
                previous = committed.get(job.batch.batch_id)
                if previous is not None:
                    if previous.shard_id != job.shard.shard_id:
                        raise DiagnosticError(
                            "E_REPLAY",
                            "committed batch ownership differs from declaration",
                            details={
                                "batch_id": job.batch.batch_id,
                                "shard_id": job.shard.shard_id,
                                "field": "shard_id",
                                "expected": job.shard.shard_id,
                                "actual": previous.shard_id,
                            },
                        )
                    self.checkpoints.record_batch(
                        sequence=job.batch.order,
                        shard_id=job.shard.shard_id,
                        batch_id=job.batch.batch_id,
                        loss_digest=previous.loss_digest,
                    )
                    reports.append(
                        {
                            "batch_id": job.batch.batch_id,
                            "shard_id": job.shard.shard_id,
                            "status": "already_committed",
                            "loss_digest": previous.loss_digest,
                        }
                    )
                    continue
                runnable.append(job)
            segment = []
            if not runnable:
                return
            outcomes = self.scheduler.run(
                runnable,
                self._execute,
                self._issue_lease,
                on_start=self._on_start,
                on_success=self._on_success,
                on_error=self._on_error,
            )
            reports.extend(outcome.to_dict() for outcome in outcomes)
            errors = [outcome for outcome in outcomes if outcome.error is not None]
            if errors:
                first = errors[0]
                assert first.error is not None
                raise DiagnosticError(
                    first.error.code,
                    first.error.message,
                    field=first.error.field,
                    details={
                        **first.error.details,
                        "batch_id": first.job.batch.batch_id,
                        "shard_id": first.job.shard.shard_id,
                    },
                )

        for item in items:
            if item.kind == "batch":
                batch = item.payload
                assert isinstance(batch, BatchSpec)
                shard = self.session.shard(batch.shard_id)
                version = self.checkpoints.current_version(batch.shard_id)
                parameters = get_parameter_set(shard, version)
                # Lazy import avoids pulling scheduler details into validation.
                from .params import quantization_error_bound

                bound = quantization_error_bound(
                    shard, batch.sample_ids, version
                ).loss_bound
                segment.append(
                    SchedulerJob(
                        batch=batch,
                        shard=shard,
                        parameter_version=version,
                        parameter_digest=parameters.digest,
                        error_bound=bound,
                    )
                )
            else:
                flush_segment()
                event = item.payload
                assert isinstance(event, EventSpec)
                self._apply_event(event)
                committed = self._already_committed()
        flush_segment()
        runtime_state = self.checkpoints.snapshot()
        self.ledger.save_runtime_state(runtime_state, now_ms=self.clock())
        replay = self.ledger.replay()
        return {
            "ok": True,
            "format_version": self.session.format_version,
            "recovery": recovery,
            "batches": reports,
            "events": self.event_results,
            "checkpoint_state": runtime_state,
            "quota_high_watermarks": self.scheduler.quota_high_watermarks,
            "ledger": {
                "path": self.ledger.path,
                "snapshot": self.ledger.snapshot_info(),
                "replay": replay,
            },
        }


def run_session(
    session: SessionSpec,
    *,
    ledger_path: str | Path,
    startup_timeout_s: float = 8.0,
    handshake_timeout_s: float = 3.0,
    connect_timeout_s: float = 2.0,
    read_timeout_s: float = 30.0,
    max_workers: int | None = None,
) -> dict[str, Any]:
    ledger = Ledger.open(ledger_path, session)
    try:
        with WorkerPool(
            session,
            startup_timeout_s=startup_timeout_s,
            handshake_timeout_s=handshake_timeout_s,
            connect_timeout_s=connect_timeout_s,
            read_timeout_s=read_timeout_s,
        ) as pool:
            engine = DiagnosticEngine(
                session,
                ledger,
                pool,
                max_workers=max_workers,
            )
            return engine.run()
    finally:
        ledger.close()
