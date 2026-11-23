"""End-to-end orchestration for deterministic dispatch and recovery."""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait as wait_for_futures
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from .batching import SubscriptionPlan, build_plan
from .errors import (
    BusDiagnosticError,
    LeaseError,
    ReplayError,
)
from .ledger import (
    CONFIRMED,
    FAILED,
    Ledger,
    PreparedRun,
    VerificationResult,
)
from .leases import LeaseManager
from .logging_utils import StructuredLogger
from .models import Acquisition, Declaration, PlanBatch, Subscriber
from .protocol import SubscriberProcess
from .spec import Dataset


@dataclass(frozen=True)
class FaultInjection:
    processing_delay_ms: int = 0
    omit_ack_field: str | None = None


@dataclass
class RunReport:
    run_id: str
    declaration_version: int
    resumed: bool
    status: str
    batch_count: int
    confirmed_batches: int
    failed_batches: int
    unknown_batches: int
    verified_batches: int
    recovery_unknown_batches: int
    errors: list[dict[str, Any]] = field(default_factory=list)
    elapsed_ms: float = 0.0
    plan_digest: str = ""
    ledger_digest: str = ""
    replay_mismatch: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DeclarationDiff:
    old_version: int | None
    new_version: int
    added_subscribers: tuple[str, ...]
    removed_subscribers: tuple[str, ...]
    upgraded_subscribers: tuple[dict[str, Any], ...]
    changed_channels: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "old_version": self.old_version,
            "new_version": self.new_version,
            "added_subscribers": list(self.added_subscribers),
            "removed_subscribers": list(self.removed_subscribers),
            "upgraded_subscribers": list(self.upgraded_subscribers),
            "changed_channels": list(self.changed_channels),
        }


def declaration_diff(old: Declaration | None, new: Declaration) -> DeclarationDiff:
    old_subscribers = old.subscriber_map() if old else {}
    new_subscribers = new.subscriber_map()
    old_channels = old.channel_map() if old else {}
    new_channels = new.channel_map()
    added = tuple(sorted(set(new_subscribers) - set(old_subscribers)))
    removed = tuple(sorted(set(old_subscribers) - set(new_subscribers)))
    upgraded = tuple(
        {
            "subscriber_id": subscriber_id,
            "old_declaration_version": old.declaration_version if old else None,
            "new_declaration_version": new.declaration_version,
            "old_digest": old_subscribers[subscriber_id].digest(),
            "new_digest": new_subscribers[subscriber_id].digest(),
        }
        for subscriber_id in sorted(set(old_subscribers) & set(new_subscribers))
        if old_subscribers[subscriber_id].digest() != new_subscribers[subscriber_id].digest()
    )
    changed_channels = tuple(
        sorted(
            channel_id
            for channel_id in set(old_channels) | set(new_channels)
            if channel_id not in old_channels
            or channel_id not in new_channels
            or old_channels[channel_id].declaration_content()
            != new_channels[channel_id].declaration_content()
        )
    )
    return DeclarationDiff(
        old_version=old.declaration_version if old else None,
        new_version=new.declaration_version,
        added_subscribers=added,
        removed_subscribers=removed,
        upgraded_subscribers=upgraded,
        changed_channels=changed_channels,
    )


class DiagnosticEngine:
    def __init__(
        self,
        dataset: Dataset,
        *,
        db_path: str | Path,
        log_path: str | Path | None = None,
        worker_log_path: str | Path | None = None,
        crash_after: int | None = None,
        fault_injections: dict[str, FaultInjection] | None = None,
        logger: StructuredLogger | None = None,
    ) -> None:
        self.dataset = dataset
        self.declaration = dataset.declaration
        self.plan: SubscriptionPlan = build_plan(self.declaration, dataset.events)
        self.db_path = str(db_path)
        self.log_path = str(log_path) if log_path else None
        self.worker_log_path = (
            str(worker_log_path)
            if worker_log_path
            else (str(Path(self.log_path).with_suffix(".workers.jsonl")) if self.log_path else None)
        )
        self.logger = logger or StructuredLogger(self.log_path, echo=False)
        self.ledger = Ledger(self.db_path, self.logger)
        self.crash_after = crash_after
        self.fault_injections = dict(fault_injections or {})
        self._lock = threading.RLock()
        self._running = False
        self._pool_processes: dict[str, SubscriberProcess] = {}
        self._lease_manager: LeaseManager | None = None
        self._monitor_stop = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        self._prepared: PreparedRun | None = None
        self._replacement_pending: dict[str, Any] | None = None

    def close(self) -> None:
        self._stop_services()
        self.ledger.close()

    def replace_declaration(self, new_dataset: Dataset) -> DeclarationDiff:
        """Atomically fence the old declaration and install a new one.

        No old dispatch result is merged into the new view.  The SQLite run
        boundary marks all old rows stale, and the returned diff is sorted.
        """

        with self._lock:
            if self._running:
                raise BusDiagnosticError(
                    "E_STATE", "cannot replace declaration while a run is active"
                )
            old = self.declaration
            if new_dataset.declaration.declaration_version <= old.declaration_version:
                raise BusDiagnosticError(
                    "E_SPEC",
                    "replacement declaration_version must be strictly greater",
                    field="declaration_version",
                    details={
                        "old_version": old.declaration_version,
                        "new_version": new_dataset.declaration.declaration_version,
                    },
                )
            if self._lease_manager is not None:
                self._lease_manager.revoke_all("declaration_replaced")
            self._stop_services()
            diff = declaration_diff(old, new_dataset.declaration)
            self.dataset = new_dataset
            self.declaration = new_dataset.declaration
            self.plan = build_plan(self.declaration, new_dataset.events)
            self._prepared = self.ledger.prepare_run(
                self.declaration,
                spec_digest=self.declaration.spec_digest(),
                events_digest=new_dataset.events_digest,
                content_digest=new_dataset.content_digest,
                replacement=diff.as_dict(),
            )
            self._replacement_pending = diff.as_dict()
            self.logger.info(
                "DECLARATION_REPLACED",
                "declaration replaced as a whole",
                **diff.as_dict(),
            )
            return diff

    def run(self, *, resume: bool = True, verify: bool = True) -> RunReport:
        with self._lock:
            if self._running:
                raise BusDiagnosticError("E_STATE", "a run is already active")
            self._running = True
        started = time.monotonic()
        prepared: PreparedRun | None = None
        pool: dict[str, SubscriberProcess] = {}
        lease_manager: LeaseManager | None = None
        executor: ThreadPoolExecutor | None = None
        recovery_unknown = 0
        verified = 0
        replay_mismatch: dict[str, Any] | None = None
        runtime_errors: list[BusDiagnosticError] = []
        try:
            self._stop_services()
            self._completed_for_crash = 0
            prepared = self._prepared or self.ledger.prepare_run(
                self.declaration,
                spec_digest=self.declaration.spec_digest(),
                events_digest=self.dataset.events_digest,
                content_digest=self.dataset.content_digest,
                replacement=self._replacement_pending,
            )
            self._prepared = prepared
            if prepared.resumed and resume:
                validation = self._verify_for_recovery(prepared.run_id)
                verified = len(validation.verified_batches)
                recovery_unknown = len(validation.unknown_batches)

            pending = self.ledger.pending_batches(prepared.run_id, self.plan.batches)
            self.logger.info(
                "RUN_START",
                "diagnostic run started",
                run_id=prepared.run_id,
                declaration_version=self.declaration.declaration_version,
                resumed=prepared.resumed,
                pending_batches=len(pending),
                plan_batches=len(self.plan.batches),
                plan_digest=self.plan.digest(),
            )
            if not pending:
                status = self.ledger.finalize_run(prepared.run_id)
                snapshot = self.ledger.snapshot(prepared.run_id)
                return self._build_report(
                    prepared,
                    status=status,
                    started=started,
                    verified=verified,
                    recovery_unknown=recovery_unknown,
                    runtime_errors=runtime_errors,
                    snapshot=snapshot,
                )

            processes: dict[str, SubscriberProcess] = {}
            port_release_grace_ms = (
                min(self.declaration.handshake_timeout_ms, 2000)
                if prepared.resumed
                else 0
            )
            for subscriber in self.declaration.subscribers:
                fault = self.fault_injections.get(subscriber.subscriber_id, FaultInjection())
                processes[subscriber.subscriber_id] = SubscriberProcess(
                    subscriber,
                    handshake_timeout_ms=self.declaration.handshake_timeout_ms,
                    logger=self.logger,
                    worker_log_path=self.worker_log_path,
                    inject_processing_delay_ms=fault.processing_delay_ms,
                    omit_ack_field=fault.omit_ack_field,
                    port_release_grace_ms=port_release_grace_ms,
                )
            pool = processes
            startup_errors: dict[str, BusDiagnosticError] = {}
            for subscriber_id in sorted(processes):
                try:
                    processes[subscriber_id].start()
                except BusDiagnosticError as exc:
                    startup_errors[subscriber_id] = exc
                    runtime_errors.append(exc)
                    self.ledger.record_error(exc, run_id=prepared.run_id)
            self._pool_processes = processes

            lease_manager = LeaseManager(
                global_limit=self.declaration.global_lease_limit,
                subscriber_limits={
                    subscriber.subscriber_id: subscriber.lease_limit
                    for subscriber in self.declaration.subscribers
                },
                channel_rate_limits={
                    channel.channel_id: channel.send_rate_limit
                    for channel in self.declaration.channels
                },
                logger=self.logger,
            )
            self._lease_manager = lease_manager
            self._start_expiry_monitor(lease_manager, prepared.run_id)

            remaining: list[PlanBatch] = []
            for batch in pending:
                if batch.subscriber_id in startup_errors:
                    self.ledger.record_unsent_failure(
                        prepared.run_id, batch, startup_errors[batch.subscriber_id]
                    )
                else:
                    remaining.append(batch)

            if remaining:
                executor = ThreadPoolExecutor(
                    max_workers=self.declaration.global_lease_limit,
                    thread_name_prefix="dispatch",
                )
                queued_mono_ms = lease_manager.clock.milliseconds()
                pending_entries: list[tuple[PlanBatch, float, str]] = [
                    (batch, queued_mono_ms, "none") for batch in remaining
                ]
                futures: dict[Future[None], PlanBatch] = {}

                while pending_entries or futures:
                    still_pending: list[tuple[PlanBatch, float, str]] = []
                    admitted_any = False
                    for batch, wait_start, waited_reason in pending_entries:
                        subscriber = self._subscriber(batch.subscriber_id)
                        try:
                            acquisition, blocking_reason = lease_manager.try_acquire(
                                batch_id=batch.batch_id,
                                subscriber_id=batch.subscriber_id,
                                channel_id=batch.channel_id,
                                ttl_ms=subscriber.processing_budget_ms,
                                wait_start_mono_ms=wait_start,
                            )
                        except BusDiagnosticError as exc:
                            runtime_errors.append(exc)
                            self.ledger.record_unsent_failure(
                                prepared.run_id, batch, exc
                            )
                            continue
                        if acquisition is None:
                            if waited_reason == "none":
                                waited_reason = blocking_reason
                            still_pending.append((batch, wait_start, waited_reason))
                            continue
                        if waited_reason != "none":
                            acquisition = replace(
                                acquisition,
                                wait=replace(acquisition.wait, reason=waited_reason),
                            )
                        future = executor.submit(
                            self._dispatch_one,
                            prepared.run_id,
                            batch,
                            processes[batch.subscriber_id],
                            lease_manager,
                            acquisition,
                        )
                        futures[future] = batch
                        admitted_any = True

                    pending_entries = still_pending
                    done: set[Future[None]] = set()
                    if futures:
                        done, _ = wait_for_futures(
                            futures,
                            timeout=0.02 if pending_entries else 0.1,
                            return_when=FIRST_COMPLETED,
                        )
                    for future in done:
                        batch = futures.pop(future)
                        try:
                            future.result()
                        except BusDiagnosticError as exc:
                            runtime_errors.append(exc)
                        except Exception as exc:  # defensive: one worker must not hide the run
                            error = BusDiagnosticError(
                                "E_INTERNAL",
                                f"unexpected dispatch worker failure: {exc}",
                                details={"error_type": type(exc).__name__},
                            )
                            runtime_errors.append(error)
                            self.ledger.record_error(
                                error,
                                run_id=prepared.run_id,
                                batch_id=batch.batch_id,
                            )
                    if not admitted_any and not done and pending_entries:
                        lease_manager.wait_for_change(0.01)

            if verify:
                try:
                    final_verification = self.ledger.verify_committed(
                        prepared.run_id, self.plan.batches
                    )
                    verified = max(verified, len(final_verification.verified_batches))
                    self.ledger.record_replay_check(
                        prepared.run_id,
                        batch_id=None,
                        mismatch_field=None,
                        expected={"plan_digest": self.plan.digest()},
                        actual={
                            "verified_batches": len(final_verification.verified_batches),
                            "remaining_batches": len(final_verification.remaining_batches),
                        },
                        matched=not final_verification.remaining_batches,
                    )
                except ReplayError as exc:
                    replay_mismatch = exc.as_dict()
                    runtime_errors.append(exc)
                    self.ledger.record_error(exc, run_id=prepared.run_id)
                    self.logger.diagnostic(exc, run_id=prepared.run_id)

            status = self.ledger.finalize_run(prepared.run_id)
            snapshot = self.ledger.snapshot(prepared.run_id)
            report = self._build_report(
                prepared,
                status=status,
                started=started,
                verified=verified,
                recovery_unknown=recovery_unknown,
                runtime_errors=runtime_errors,
                snapshot=snapshot,
            )
            report.replay_mismatch = replay_mismatch
            self.logger.info(
                "RUN_FINISH",
                "diagnostic run finished",
                run_id=prepared.run_id,
                status=status,
                confirmed_batches=report.confirmed_batches,
                failed_batches=report.failed_batches,
                unknown_batches=report.unknown_batches,
                elapsed_ms=round(report.elapsed_ms, 3),
            )
            return report
        except ReplayError as exc:
            runtime_errors.append(exc)
            if prepared is not None:
                self.ledger.record_error(exc, run_id=prepared.run_id)
            self.logger.diagnostic(exc)
            raise
        finally:
            self._monitor_stop.set()
            if self._monitor_thread is not None:
                self._monitor_thread.join(timeout=1.0)
                self._monitor_thread = None
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=False)
            if pool:
                for subscriber_id in sorted(pool, reverse=True):
                    try:
                        pool[subscriber_id].stop()
                    except Exception:
                        pass
            self._pool_processes = {}
            self._lease_manager = lease_manager
            self._prepared = None
            self._replacement_pending = None
            with self._lock:
                self._running = False

    def verify_replay(self, run_id: str | None = None) -> VerificationResult:
        current = self.ledger.current_run()
        if current is None:
            raise ReplayError("no current run is available for replay verification")
        selected_run_id = run_id or str(current["run_id"])
        if int(current["declaration_version"]) != self.declaration.declaration_version:
            raise ReplayError(
                "current ledger declaration version differs from the loaded declaration",
                mismatch_field="declaration_version",
                details={
                    "ledger_version": int(current["declaration_version"]),
                    "loaded_version": self.declaration.declaration_version,
                },
            )
        try:
            result = self.ledger.verify_committed(selected_run_id, self.plan.batches)
        except ReplayError as exc:
            self.ledger.record_replay_check(
                selected_run_id,
                batch_id=exc.details.get("batch_id"),
                mismatch_field=exc.details.get("mismatch_field"),
                expected={"mismatch_field": exc.details.get("mismatch_field")},
                actual={"message": exc.message},
                matched=False,
            )
            self.ledger.record_error(exc, run_id=selected_run_id)
            raise
        return result

    def snapshot(self) -> dict[str, Any]:
        snapshot = self.ledger.snapshot()
        if self._lease_manager is not None:
            snapshot["live_leases"] = self._lease_manager.limits_snapshot()
            snapshot["active_lease_rows"] = [
                asdict(lease) for lease in self._lease_manager.snapshot()
            ]
        else:
            snapshot["live_leases"] = None
            snapshot["active_lease_rows"] = []
        snapshot["plan_digest"] = self.plan.digest()
        return snapshot

    def _dispatch_one(
        self,
        run_id: str,
        batch: PlanBatch,
        process: SubscriberProcess,
        lease_manager: LeaseManager,
        acquisition: Acquisition,
    ) -> None:
        subscriber = self._subscriber(batch.subscriber_id)
        try:
            self.ledger.begin_batch(run_id, batch, acquisition, reason="dispatch_grant")
            request_timeout_ms = max(
                self.declaration.handshake_timeout_ms,
                subscriber.processing_budget_ms * 2,
                1000,
            )
            ack = process.dispatch(batch, request_timeout_ms=request_timeout_ms)
            if not lease_manager.validate(
                acquisition.lease.lease_id, acquisition.lease.version
            ):
                raise LeaseError(
                    "lease expired before confirmation and cannot confirm the batch",
                    lease_id=acquisition.lease.lease_id,
                    batch_id=batch.batch_id,
                    subscriber_id=batch.subscriber_id,
                )
            self.ledger.complete_batch(
                run_id,
                batch,
                state=CONFIRMED,
                lease=acquisition.lease,
                acquisition=acquisition,
                ack=ack,
            )
            self.logger.info(
                "BATCH_CONFIRMED",
                "batch confirmed",
                run_id=run_id,
                batch_id=batch.batch_id,
                subscriber_id=batch.subscriber_id,
                channel_id=batch.channel_id,
                wait_ms=round(acquisition.wait.wait_ms, 3),
                processing_ms=ack.processing_ms,
                lease_version=acquisition.lease.version,
                thread=threading.current_thread().name,
            )
        except Exception as exc:
            error = exc if isinstance(exc, BusDiagnosticError) else BusDiagnosticError(
                "E_INTERNAL",
                f"unexpected dispatch failure: {exc}",
                details={"error_type": type(exc).__name__},
            )
            try:
                self.ledger.complete_batch(
                    run_id,
                    batch,
                    state=FAILED,
                    lease=acquisition.lease,
                    acquisition=acquisition,
                    error=error,
                )
            except Exception as ledger_exc:
                self.logger.error(
                    "E_LEDGER",
                    f"could not record failed batch: {ledger_exc}",
                    run_id=run_id,
                    batch_id=batch.batch_id,
                )
            if error is exc:
                raise
            raise error from exc
        finally:
            lease_manager.release(
                acquisition.lease.lease_id, acquisition.lease.version
            )
            self._maybe_crash(run_id)

    def _maybe_crash(self, run_id: str) -> None:
        if self.crash_after is None:
            return
        with self._lock:
            if not hasattr(self, "_completed_for_crash"):
                self._completed_for_crash = 0
            self._completed_for_crash += 1
            if self._completed_for_crash != self.crash_after:
                return
        self.logger.error(
            "E_TEST_FORCED_TERMINATION",
            "forced termination requested after committed batches",
            run_id=run_id,
            crash_after=self.crash_after,
        )
        os._exit(75)

    def _start_expiry_monitor(self, manager: LeaseManager, run_id: str) -> None:
        self._monitor_stop = threading.Event()

        def monitor() -> None:
            while not self._monitor_stop.wait(0.01):
                expired = manager.expire()
                for expiry in expired:
                    try:
                        self.ledger.mark_lease_expired(run_id, expiry)
                    except Exception as exc:
                        self.logger.error(
                            "E_LEDGER",
                            f"could not persist lease expiry: {exc}",
                            run_id=run_id,
                            lease_id=expiry.lease_id,
                        )

        self._monitor_thread = threading.Thread(
            target=monitor, name="lease-expiry-monitor", daemon=True
        )
        self._monitor_thread.start()

    def _verify_for_recovery(self, run_id: str) -> VerificationResult:
        try:
            result = self.ledger.verify_and_mark_unknown(run_id, self.plan.batches)
        except ReplayError as exc:
            self.ledger.record_error(exc, run_id=run_id)
            self.logger.diagnostic(exc, run_id=run_id)
            raise
        self.logger.info(
            "RECOVERY_VERIFIED",
            "committed ledger entries verified before resume",
            run_id=run_id,
            verified_batches=len(result.verified_batches),
            unknown_batches=len(result.unknown_batches),
            remaining_batches=len(result.remaining_batches),
        )
        return result

    def _subscriber(self, subscriber_id: str) -> Subscriber:
        for subscriber in self.declaration.subscribers:
            if subscriber.subscriber_id == subscriber_id:
                return subscriber
        raise LeaseError(f"unknown subscriber {subscriber_id!r}", subscriber_id=subscriber_id)

    def _stop_services(self) -> None:
        self._monitor_stop.set()
        if self._lease_manager is not None:
            self._lease_manager.revoke_all("runtime_stopped")
        for subscriber_id in sorted(self._pool_processes, reverse=True):
            try:
                self._pool_processes[subscriber_id].stop()
            except Exception:
                pass
        self._pool_processes = {}

    def _build_report(
        self,
        prepared: PreparedRun,
        *,
        status: str,
        started: float,
        verified: int,
        recovery_unknown: int,
        runtime_errors: list[BusDiagnosticError],
        snapshot: dict[str, Any],
    ) -> RunReport:
        stats = snapshot.get("stats", {})
        errors = [error.as_dict() for error in runtime_errors]
        if not errors:
            errors = [
                {
                    "code": error["code"],
                    "message": error["message"],
                    "field": error.get("field"),
                    "line": error.get("source_line"),
                    "details": error.get("payload"),
                }
                for error in snapshot.get("errors", [])
            ]
        ledger_info = snapshot.get("ledger_info", {})
        return RunReport(
            run_id=prepared.run_id,
            declaration_version=prepared.declaration_version,
            resumed=prepared.resumed,
            status=status,
            batch_count=int(stats.get("batch_count", 0)),
            confirmed_batches=int(stats.get("confirmed_batches", 0)),
            failed_batches=int(stats.get("failed_batches", 0)),
            unknown_batches=int(stats.get("unknown_batches", 0)),
            verified_batches=verified,
            recovery_unknown_batches=recovery_unknown,
            errors=errors,
            elapsed_ms=(time.monotonic() - started) * 1000.0,
            plan_digest=self.plan.digest(),
            ledger_digest=str(ledger_info.get("content_digest", "")),
        )
