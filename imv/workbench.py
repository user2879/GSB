from __future__ import annotations

import copy
import json
import shutil
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .engine import (
    MaintenanceOutcome,
    Transition,
    authoritative_results,
    build_view_result,
    find_first_difference,
    maintain_view_incrementally,
    replay_base_rows,
)
from .errors import BusinessError, as_business_error, fail
from .lease import LeaseManager, LeaseRecord
from .logging_utils import StructuredLogger
from .model import (
    BaseTable,
    ChangeStream,
    Definitions,
    ViewDef,
    canonical_rows_payload,
    load_dataset,
    load_definitions,
    rows_from_payload,
    validate_definitions_against_base,
)
from .scheduler import JobHandle, JobSpec, RefreshScheduler
from .store import LedgerStore, PersistedState, SnapshotAudit
from .util import (
    digest_bytes,
    digest_obj,
    ensure_directory,
    monotonic_ms,
    wall_ms,
)


@dataclass
class RuntimeView:
    view: ViewDef
    result_json: str
    result_digest: str
    result_version: int
    cursor: int
    stale: bool = False
    inconsistent: bool = False
    inconsistent_reason: str | None = None
    updated_wall_ms: int = 0

    def to_store(self) -> dict[str, Any]:
        return {
            "result_version": self.result_version,
            "cursor": self.cursor,
            "result_json": self.result_json,
            "result_digest": self.result_digest,
            "stale": self.stale,
            "inconsistent": self.inconsistent,
            "updated_wall_ms": self.updated_wall_ms or wall_ms(),
        }


class Workbench:
    """Application service shared by the Tk UI, CLI and tests."""

    def __init__(
        self,
        base_path: str | Path,
        views_path: str | Path,
        changes_path: str | Path,
        data_dir: str | Path,
        *,
        reset: bool = False,
        global_concurrency: int | None = None,
        auto_rebuild_on_expiry: bool = True,
        start_watchdog: bool = True,
        watchdog_interval_ms: int = 25,
    ):
        dataset = load_dataset(base_path, views_path, changes_path)
        self.base_path = str(Path(base_path))
        self.views_path = str(Path(views_path))
        self.changes_path = str(Path(changes_path))
        self.data_dir = Path(data_dir)
        if reset and self.data_dir.exists():
            resolved = self.data_dir.resolve()
            if len(resolved.parts) < 2:
                raise ValueError("refusing to reset a filesystem root")
            shutil.rmtree(resolved)
        ensure_directory(self.data_dir)
        (self.data_dir / "logs").mkdir(parents=True, exist_ok=True)
        self.logger = StructuredLogger(self.data_dir / "logs" / "structured.ndjson")
        self.base_initial: BaseTable = dataset.base
        self.changes: ChangeStream = dataset.changes
        self.definitions: Definitions = dataset.definitions
        self.global_concurrency = global_concurrency or self.definitions.session.global_concurrency
        if self.global_concurrency <= 0:
            fail("E_SCHEMA", "global concurrency must be positive", location={"path": "$.session.global_concurrency"})
        self.auto_rebuild_on_expiry = auto_rebuild_on_expiry
        self.watchdog_interval_ms = max(10, watchdog_interval_ms)
        self.startup_events: list[dict[str, Any]] = []
        self.definition_replacement_summary: dict[str, Any] | None = None
        self._state_lock = threading.RLock()
        self._batch_lock = threading.Lock()
        self._recovery_guard = threading.Lock()
        self._recovering: set[str] = set()
        self._watchdog_stop = threading.Event()
        self._watchdog_thread: threading.Thread | None = None
        self._closed = False
        self.store = LedgerStore(self.data_dir / "ledger.sqlite3")
        self.lease_manager = LeaseManager()
        self._refresh_versions: dict[str, int] = {}
        self.scheduler: RefreshScheduler | None = None
        self.param_version = 1
        self.cursor = 0
        self.batch_counter = 0
        self.base_rows: dict[Any, dict[str, Any]] = {}
        self.runtime: dict[str, RuntimeView] = {}
        self._last_snapshot_audit: SnapshotAudit | None = None
        try:
            self._initialize()
            if start_watchdog:
                self._start_watchdog()
        except BaseException:
            self._shutdown_resources()
            raise

    # ------------------------------------------------------------------
    # Initialization and recovery
    # ------------------------------------------------------------------
    def _initialize(self) -> None:
        if not self.store.is_initialized():
            self.param_version = 1
            self.cursor = 0
            self.batch_counter = 0
            self.base_rows = {row_id: dict(row) for row_id, row in self.base_initial.rows.items()}
            self._create_scheduler()
            self._initial_materialize()
            return

        audit = self.store.validate_and_restore_snapshot()
        self._last_snapshot_audit = audit
        if audit.error is not None:
            self._record_event(audit.error, "snapshot_validation")
        if audit.error is not None and not audit.recovered:
            raise audit.error

        stored_initial_digest = self.store.get_meta("base_initial_digest")
        if stored_initial_digest and stored_initial_digest != self.base_initial.digest:
            error = BusinessError(
                "E_DATA",
                "base table digest differs from the ledger's initial base snapshot",
                {"path": "$.base", "field": "digest"},
                {"expected": stored_initial_digest, "actual": self.base_initial.digest},
            )
            self._record_event(error, "source_validation")
            raise error
        stored_changes_digest = self.store.get_meta("changes_digest")
        if stored_changes_digest and stored_changes_digest != self.changes.digest:
            error = BusinessError(
                "E_DATA",
                "change stream digest differs from the ledger snapshot",
                {"path": "$.changes", "field": "digest"},
                {"expected": stored_changes_digest, "actual": self.changes.digest},
            )
            self._record_event(error, "source_validation")
            raise error

        state = self.store.load_state()
        self.param_version = state.param_version
        self.cursor = state.cursor
        self.batch_counter = state.batch_counter
        self.store.verify_base_payload(state.base_rows_payload)
        try:
            _table, _row_id_column, reconstructed_columns, initial_rows = rows_from_payload(
                self.base_initial.canonical_payload()
            )
        except (KeyError, TypeError, ValueError) as exc:
            fail("E_DATA", "base table cannot be reconstructed", {"path": "$.base"}, {"error": str(exc)})
        if reconstructed_columns != self.base_initial.columns or initial_rows != self.base_initial.rows:
            fail("E_DATA", "base table row reconstruction changed values", {"path": "$.base.rows"})
        _, _, _, current_rows = rows_from_payload(state.base_rows_payload)
        self.base_rows = current_rows
        current_payload = canonical_rows_payload(
            self.base_initial.table_name,
            self.base_initial.row_id_column,
            self.base_initial.columns,
            self.base_rows,
        )
        if digest_obj(current_payload) != state.base_digest:
            error = BusinessError("E_DATA", "persisted base row digest does not match its content", {"path": "$.base_current", "field": "digest"})
            self._record_event(error, "base_validation")
            raise error
        replayed_rows = replay_base_rows(self.base_initial.rows, self.changes.changes, cursor=self.cursor)
        replay_payload = canonical_rows_payload(
            self.base_initial.table_name,
            self.base_initial.row_id_column,
            self.base_initial.columns,
            replayed_rows,
        )
        if digest_obj(replay_payload) != state.base_digest:
            error = BusinessError(
                "E_DATA",
                "base rows do not reproduce from the initial table and recorded change cursor",
                {"path": "$.base_current", "field": "digest"},
                {"replayed": digest_obj(replay_payload), "stored": state.base_digest},
            )
            self._record_event(error, "base_validation")
            raise error

        lease_records = [LeaseRecord.from_payload(payload) for payload in state.leases.values()]
        self.lease_manager.load(lease_records)
        self._refresh_versions = {
            record.view_id: record.refresh_version for record in lease_records
        }
        self._create_scheduler()

        if self.store.current_fingerprint != self.definitions.fingerprint:
            self._replace_definitions_locked(self.definitions, startup=True)
        else:
            self._load_runtime_from_state(state)

        expired_after_restart = self.lease_manager.expire_active_after_restore()
        for view_id in expired_after_restart:
            if view_id in self.runtime:
                with self._state_lock:
                    runtime = self.runtime[view_id]
                    runtime.inconsistent = True
                    runtime.inconsistent_reason = "process restart ended the active lease renewal"
                self.logger.emit(
                    "WARN",
                    "lease_expired_after_restart",
                    "active lease was expired because no in-process renewer survived restart",
                    code="E_LEASE",
                    location={"view_id": view_id},
                )

        self._validate_loaded_runtime()
        self._start_recovery_for_inconsistent()

    def _load_runtime_from_state(self, state: PersistedState) -> None:
        runtime: dict[str, RuntimeView] = {}
        for view_id in sorted(self.definitions.views):
            result = state.results.get(view_id)
            if result is None:
                error = BusinessError("E_SNAPSHOT", "current view result is missing", {"view_id": view_id, "path": "$.view_results"})
                self._record_event(error, "snapshot_load")
                raise error
            result_json = result["result_json"]
            computed_digest = digest_bytes(result_json.encode("utf-8"))
            if computed_digest != result["result_digest"]:
                error = BusinessError(
                    "E_DATA",
                    "incremental result digest does not match result bytes",
                    {"view_id": view_id, "field": "result_digest"},
                    {"stored": result["result_digest"], "computed": computed_digest},
                )
                self._record_event(error, "result_validation")
                raise error
            lease = self.lease_manager.get(view_id)
            runtime[view_id] = RuntimeView(
                view=self.definitions.views[view_id],
                result_json=result_json,
                result_digest=result["result_digest"],
                result_version=int(result["result_version"]),
                cursor=int(result["cursor"]),
                stale=bool(result.get("stale", False)),
                inconsistent=bool(result.get("inconsistent", False)),
                inconsistent_reason="restored inconsistent result" if result.get("inconsistent") else None,
                updated_wall_ms=int(result.get("updated_wall_ms", 0)),
            )
            if lease is not None:
                self._refresh_versions[view_id] = max(self._refresh_versions.get(view_id, 0), lease.refresh_version)
        self.runtime = runtime

    def _validate_loaded_runtime(self) -> None:
        with self._state_lock:
            for view_id, runtime in self.runtime.items():
                if runtime.cursor != self.cursor:
                    runtime.inconsistent = True
                    runtime.inconsistent_reason = (
                        f"result cursor {runtime.cursor} does not match ledger cursor {self.cursor}"
                    )
                if runtime.stale:
                    runtime.inconsistent = True
                    runtime.inconsistent_reason = "result belongs to a stale parameter version"
                lease = self.lease_manager.check_view(view_id)
                if lease is None or lease.state in {"expired", "failed"} or lease.stale:
                    runtime.inconsistent = True
                    runtime.inconsistent_reason = "lease is not usable"

    def _start_recovery_for_inconsistent(self) -> None:
        inconsistent = [view_id for view_id, runtime in self.runtime.items() if runtime.inconsistent]
        for view_id in inconsistent:
            self._start_recovery(view_id, reason=runtime_reason(self.runtime[view_id]), do_rebuild=self.auto_rebuild_on_expiry)

    def _create_scheduler(self) -> None:
        quotas = {view_id: view.per_view_quota for view_id, view in self.definitions.views.items()}
        if self.scheduler is not None:
            self.scheduler.per_view_quotas = quotas
            return
        self.scheduler = RefreshScheduler(
            self.lease_manager,
            self.logger,
            global_limit=self.global_concurrency,
            per_view_quotas=quotas,
            on_lease_lost=self._on_lease_lost,
        )

    def _initial_materialize(self) -> None:
        assert self.scheduler is not None
        handles: list[JobHandle] = []
        outcomes: dict[str, MaintenanceOutcome] = {}
        for view_id in sorted(self.definitions.views):
            view = self.definitions.views[view_id]
            refresh_version = self._next_refresh_version(view_id)

            def job_fn(view=view) -> MaintenanceOutcome:
                return build_view_result(
                    view,
                    self.definitions.fixed,
                    self.base_rows,
                    cursor=0,
                    param_version=1,
                )

            spec = JobSpec(
                view_id=view_id,
                kind="initial_materialize",
                param_version=1,
                refresh_version=refresh_version,
                ttl_ms=view.lease_ttl_ms,
                renew_ms=view.lease_renew_ms,
                fn=job_fn,
                batch_id="initial",
                per_view_quota=view.per_view_quota,
            )
            handles.append(self.scheduler.submit(spec))
        values, errors, pending = self.scheduler.wait_all(handles)
        if pending:
            fail("E_SCHEDULE", "initial materialization job did not finish", location={"pending": len(pending)})
        if errors:
            raise errors[0]
        for handle, value in zip(handles, values):
            outcomes[handle.spec.view_id] = value
        results = self._results_from_outcomes(outcomes, definitions=self.definitions)
        leases = self._current_lease_payloads(self.definitions.views.keys())
        jobs = self.scheduler.traces_for([handle.job_id for handle in handles])
        summary = {
            "event": "initial_materialize",
            "param_version": 1,
            "cursor": 0,
            "views": {view_id: outcomes[view_id].trace_payload() for view_id in sorted(outcomes)},
        }
        self.store.initialize_session(
            self.definitions,
            self.base_initial.canonical_payload(),
            self.base_initial.digest,
            self.changes.digest,
            results,
            leases,
            jobs,
            summary,
        )
        self.param_version = 1
        self.cursor = 0
        self.batch_counter = 0
        self.base_rows = {row_id: dict(row) for row_id, row in self.base_initial.rows.items()}
        self.runtime = self._runtime_from_outcomes(outcomes)
        self.logger.emit(
            "INFO",
            "session_initialized",
            "initial materialized views committed with leases and snapshot",
            details={"views": sorted(self.runtime), "snapshot_version": self.store.snapshot_notice()},
        )

    # ------------------------------------------------------------------
    # Batch maintenance
    # ------------------------------------------------------------------
    def advance_batch(self, batch_size: int | None = None) -> dict[str, Any]:
        started_mono = monotonic_ms()
        started_wall = wall_ms()
        with self._batch_lock:
            assert self.scheduler is not None
            self._raise_if_inconsistent()
            size = batch_size or self.definitions.session.batch_size
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                fail("E_SCHEMA", "batch size must be a positive integer", location={"path": "$.batch_size"})
            if self.cursor >= len(self.changes.changes):
                return {
                    "status": "no_changes",
                    "cursor": self.cursor,
                    "total_changes": len(self.changes.changes),
                    "latency_ms": monotonic_ms() - started_mono,
                }
            batch = self.changes.changes[self.cursor : self.cursor + size]
            rows_working = {row_id: dict(row) for row_id, row in self.base_rows.items()}
            transitions: list[Transition] = []
            for change in batch:
                before = rows_working.get(change.row_id)
                if change.op == "delete":
                    after = None
                    rows_working.pop(change.row_id, None)
                else:
                    assert change.row is not None
                    after = dict(change.row)
                    rows_working[change.row_id] = after
                transitions.append(
                    Transition(change.seq, change.op, change.row_id, copy.deepcopy(before), copy.deepcopy(after))
                )
            cursor_before = self.cursor
            cursor_after = cursor_before + len(batch)
            batch_no = self.batch_counter + 1
            handles: list[JobHandle] = []
            with self._state_lock:
                previous_runtime = {view_id: replace(runtime) for view_id, runtime in self.runtime.items()}
            for view_id in sorted(self.definitions.views):
                view = self.definitions.views[view_id]
                previous = json.loads(previous_runtime[view_id].result_json)
                refresh_version = self._next_refresh_version(view_id)

                def job_fn(
                    view=view,
                    previous=previous,
                    transitions=tuple(transitions),
                    cursor_before=cursor_before,
                    cursor_after=cursor_after,
                    rows_after=rows_working,
                ) -> MaintenanceOutcome:
                    if view.refresh_mode == "full":
                        return build_view_result(
                            view,
                            self.definitions.fixed,
                            rows_after,
                            cursor=cursor_after,
                            param_version=self.param_version,
                        )
                    return maintain_view_incrementally(
                        view,
                        self.definitions.fixed,
                        previous,
                        rows_after,
                        list(transitions),
                        cursor_before=cursor_before,
                        cursor_after=cursor_after,
                        param_version=self.param_version,
                    )

                handles.append(
                    self.scheduler.submit(
                        JobSpec(
                            view_id=view_id,
                            kind="incremental_maintenance" if view.refresh_mode == "incremental" else "full_refresh",
                            param_version=self.param_version,
                            refresh_version=refresh_version,
                            ttl_ms=view.lease_ttl_ms,
                            renew_ms=view.lease_renew_ms,
                            fn=job_fn,
                            batch_id=f"batch-{batch_no}",
                            per_view_quota=view.per_view_quota,
                        )
                    )
                )
            values, errors, pending = self.scheduler.wait_all(handles)
            if pending:
                fail("E_SCHEDULE", "refresh jobs are still running; they were not marked terminated", location={"pending": len(pending)})
            if errors:
                first_error = errors[0]
                failed_jobs: list[tuple[JobHandle, BusinessError]] = []
                for handle in handles:
                    error = handle.future.exception()
                    if isinstance(error, BusinessError):
                        failed_jobs.append((handle, error))
                for handle, error in failed_jobs:
                    if error.code == "E_LEASE" and handle.spec.view_id in self.runtime:
                        self._persist_inconsistency_locked(handle.spec.view_id, error.message)
                self.logger.emit(
                    "ERROR",
                    "batch_rejected",
                    "one or more refresh jobs failed; cursor was not advanced",
                    code=first_error.code,
                    location=first_error.location,
                    details={
                        "batch_no": batch_no,
                        "failed_views": [handle.spec.view_id for handle, _error in failed_jobs],
                    },
                )
                raise first_error

            outcomes: dict[str, MaintenanceOutcome] = {}
            for handle, value in zip(handles, values):
                outcomes[handle.spec.view_id] = value
            jobs = self.scheduler.traces_for([handle.job_id for handle in handles])
            leases = self._current_lease_payloads(self.definitions.views.keys())
            results = self._results_from_outcomes(outcomes, definitions=self.definitions)
            base_payload = canonical_rows_payload(
                self.base_initial.table_name,
                self.base_initial.row_id_column,
                self.base_initial.columns,
                rows_working,
            )
            summary = {
                "event": "change_batch",
                "batch_no": batch_no,
                "param_version": self.param_version,
                "cursor_before": cursor_before,
                "cursor_after": cursor_after,
                "first_seq": batch[0].seq,
                "last_seq": batch[-1].seq,
                "change_ops": {change.op: sum(1 for item in batch if item.op == change.op) for change in batch},
                "views": {
                    view_id: {
                        **outcomes[view_id].trace_payload(),
                        "refresh_mode": self.definitions.views[view_id].refresh_mode,
                    }
                    for view_id in sorted(outcomes)
                },
            }
            receipt = self.store.commit_batch(
                batch_no=batch_no,
                first_seq=batch[0].seq,
                last_seq=batch[-1].seq,
                cursor=cursor_after,
                param_version=self.param_version,
                base_payload=base_payload,
                results=results,
                leases=leases,
                jobs=jobs,
                summary=summary,
            )
            with self._state_lock:
                self.base_rows = rows_working
                self.cursor = cursor_after
                self.batch_counter = batch_no
                self.runtime = self._runtime_from_outcomes(outcomes)
            latency = monotonic_ms() - started_mono
            report = {
                "status": "committed",
                "batch_no": batch_no,
                "cursor_before": cursor_before,
                "cursor_after": cursor_after,
                "first_seq": batch[0].seq,
                "last_seq": batch[-1].seq,
                "latency_ms": latency,
                "started_wall_ms": started_wall,
                "receipt": {
                    "ledger_format_version": receipt.ledger_format_version,
                    "content_digest": receipt.content_digest,
                    "snapshot_version": receipt.snapshot_version,
                    "snapshot_digest": receipt.snapshot_digest,
                },
                "views": [
                    {
                        "view_id": view_id,
                        "digest": outcomes[view_id].digest,
                        "group_count": outcomes[view_id].group_count,
                        "impacted_groups": [list(key) for key in outcomes[view_id].impacted_groups],
                        "recomputed_groups": [list(key) for key in outcomes[view_id].recomputed_groups],
                        "lease": leases.get(view_id),
                    }
                    for view_id in sorted(outcomes)
                ],
            }
            self.logger.emit(
                "INFO",
                "batch_committed",
                "change batch committed atomically with results, leases, scheduler trace and cursor",
                details={
                    "batch_no": batch_no,
                    "cursor": cursor_after,
                    "latency_ms": latency,
                    "content_digest": receipt.content_digest,
                    "snapshot_version": receipt.snapshot_version,
                },
            )
            return report

    def advance_all(self, *, max_batches: int = 10_000) -> list[dict[str, Any]]:
        reports: list[dict[str, Any]] = []
        for _ in range(max_batches):
            if self.cursor >= len(self.changes.changes):
                break
            report = self.advance_batch()
            reports.append(report)
            if report.get("status") != "committed":
                break
        return reports

    def _raise_if_inconsistent(self) -> None:
        with self._state_lock:
            for view_id in sorted(self.runtime):
                if self.runtime[view_id].inconsistent:
                    fail(
                        "E_LEASE",
                        "view is inconsistent and must complete a full rebuild before a new batch",
                        location={"view_id": view_id},
                        details={"reason": self.runtime[view_id].inconsistent_reason},
                    )

    # ------------------------------------------------------------------
    # Authoritative replay and cache checks
    # ------------------------------------------------------------------
    def authoritative_recompute(self) -> dict[str, Any]:
        with self._batch_lock:
            with self._state_lock:
                runtime_snapshot = {view_id: replace(runtime) for view_id, runtime in self.runtime.items()}
                cursor = self.cursor
                param_version = self.param_version
            authoritative = authoritative_results(
                self.base_initial,
                self.definitions,
                self.changes.changes,
                cursor=cursor,
                param_version=param_version,
            )
            differences = []
            view_reports: dict[str, Any] = {}
            for view_id in sorted(self.definitions.views):
                runtime = runtime_snapshot.get(view_id)
                if runtime is None:
                    differences.append({"view_id": view_id, "group_key": [], "field": "__view__", "cached": None, "authoritative": "present"})
                    continue
                outcome = authoritative[view_id]
                difference = find_first_difference(view_id, runtime.result_json, outcome.payload)
                view_reports[view_id] = {
                    "cached_digest": runtime.result_digest,
                    "authoritative_digest": outcome.digest,
                    "group_count": outcome.group_count,
                    "byte_identical": difference is None,
                    "cache_visible": not runtime.inconsistent and not runtime.stale,
                }
                if difference is not None:
                    differences.append(
                        {
                            "view_id": difference.view_id,
                            "group_key": difference.group_key,
                            "field": difference.field,
                            "cached": difference.cached,
                            "authoritative": difference.authoritative,
                        }
                    )
            report = {
                "status": "matched" if not differences else "mismatch",
                "code": None if not differences else "E_PREVI",
                "param_version": param_version,
                "cursor": cursor,
                "views": view_reports,
                "first_difference": differences[0] if differences else None,
                "differences": differences,
                "byte_identical": not differences,
            }
            digest = self.store.record_authority_run(
                param_version=param_version,
                cursor=cursor,
                status=report["status"],
                report=report,
            )
            report["content_digest"] = digest
            if differences:
                first = differences[0]
                error = BusinessError(
                    "E_PREVI",
                    f"incremental cache differs from authoritative replay at {first['field']}",
                    {"view_id": first["view_id"], "group_key": first["group_key"], "field": first["field"]},
                    {"first_difference": first, "report_digest": digest},
                )
                self._record_event(error, "authority_compare")
                raise error
            self.logger.emit(
                "INFO",
                "authority_matched",
                "fast cache and authoritative replay are byte-identical",
                details={"cursor": cursor, "views": sorted(view_reports), "report_digest": digest},
            )
            return report

    def fast_snapshot(self) -> dict[str, Any]:
        started = monotonic_ms()
        with self._state_lock:
            views: list[dict[str, Any]] = []
            for view_id in sorted(self.runtime):
                runtime = self.runtime[view_id]
                lease = self.lease_manager.check_view(view_id)
                lease_state = None if lease is None else lease.state
                if lease is not None and lease.state == "expired" and not runtime.inconsistent:
                    runtime.inconsistent = True
                    runtime.inconsistent_reason = lease.note or "lease expired"
                usable_lease = lease_state in {"active", "completed"} and not (lease and lease.stale)
                visible = usable_lease and not runtime.stale and not runtime.inconsistent
                entry: dict[str, Any] = {
                    "view_id": view_id,
                    "group_by": list(runtime.view.group_by),
                    "aggregates": [spec.payload() for spec in runtime.view.aggregates],
                    "refresh_mode": runtime.view.refresh_mode,
                    "result_version": runtime.result_version,
                    "cursor": runtime.cursor,
                    "result_digest": runtime.result_digest,
                    "stale": runtime.stale,
                    "inconsistent": runtime.inconsistent,
                    "visible": visible,
                    "hide_reason": None
                    if visible
                    else runtime.inconsistent_reason
                    or ("stale parameter version" if runtime.stale else "lease unavailable or expired"),
                    "lease": None
                    if lease is None
                    else {
                        "state": lease.state,
                        "refresh_version": lease.refresh_version,
                        "granted_mono_ms": lease.granted_mono_ms,
                        "expires_mono_ms": lease.expires_mono_ms,
                        "renew_count": lease.renew_count,
                        "remaining_ms": self.lease_manager.remaining_ms(view_id),
                    },
                    "group_count": 0,
                    "groups": [],
                }
                if visible:
                    try:
                        payload = json.loads(runtime.result_json)
                        entry["groups"] = payload.get("groups", [])
                        entry["group_count"] = len(entry["groups"])
                    except json.JSONDecodeError:
                        entry["visible"] = False
                        entry["hide_reason"] = "cached result is not valid JSON"
                views.append(entry)
            scheduler_payload = self.scheduler.payload() if self.scheduler is not None else {}
            scheduler_payload["recent_jobs"] = [
                job
                for job in scheduler_payload.get("recent_jobs", [])
                if int(job.get("param_version", -1)) == self.param_version
            ]
            snapshot = {
                "param_version": self.param_version,
                "cursor": self.cursor,
                "total_changes": len(self.changes.changes),
                "batch_counter": self.batch_counter,
                "views": views,
                "scheduler": scheduler_payload,
                "recovering": sorted(self._recovering),
                "snapshot": self.store.snapshot_notice(),
                "fixed_point": {
                    "scale_factor": self.definitions.session.scale_factor,
                    "truncation": self.definitions.session.truncation,
                    "overflow": self.definitions.session.overflow,
                },
            }
        snapshot["render_latency_ms"] = monotonic_ms() - started
        snapshot["source"] = "incremental_cache"
        return snapshot

    # ------------------------------------------------------------------
    # Leases and rebuild
    # ------------------------------------------------------------------
    def _on_lease_lost(self, lease: LeaseRecord, error: BusinessError) -> None:
        with self._state_lock:
            runtime = self.runtime.get(lease.view_id)
            if runtime is not None:
                runtime.inconsistent = True
                runtime.inconsistent_reason = error.message
        self._record_event(error, "lease_lost")
        self._start_recovery(lease.view_id, reason=error.message, do_rebuild=True)

    def _start_watchdog(self) -> None:
        if self._watchdog_thread is not None:
            return
        self._watchdog_stop.clear()
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, name="imv-lease-watchdog", daemon=True)
        self._watchdog_thread.start()

    def _watchdog_loop(self) -> None:
        while not self._watchdog_stop.wait(self.watchdog_interval_ms / 1000.0):
            for record in self.lease_manager.expire_due():
                with self._state_lock:
                    runtime = self.runtime.get(record.view_id)
                    if runtime is not None and not runtime.inconsistent:
                        runtime.inconsistent = True
                        runtime.inconsistent_reason = record.note or "lease expired"
                self.logger.emit(
                    "WARN",
                    "lease_expired",
                    "lease expired without renewal; view hidden from fast cache and scheduled for rebuild",
                    code="E_LEASE",
                    location={"view_id": record.view_id},
                    details={"refresh_version": record.refresh_version, "expires_mono_ms": record.expires_mono_ms},
                )
                self._start_recovery(
                    record.view_id,
                    reason=record.note or "lease expired",
                    do_rebuild=self.auto_rebuild_on_expiry,
                )

    def _start_recovery(self, view_id: str, *, reason: str, do_rebuild: bool) -> None:
        with self._recovery_guard:
            if view_id in self._recovering:
                return
            self._recovering.add(view_id)
        thread = threading.Thread(
            target=self._recovery_worker,
            args=(view_id, reason, do_rebuild),
            name=f"imv-recovery-{view_id}",
            daemon=True,
        )
        thread.start()

    def _recovery_worker(self, view_id: str, reason: str, do_rebuild: bool) -> None:
        try:
            with self._batch_lock:
                self._persist_inconsistency_locked(view_id, reason)
                if do_rebuild:
                    self._rebuild_view_locked(view_id, reason)
        except BaseException as exc:
            error = as_business_error(exc)
            self._record_event(error, "rebuild_failure")
        finally:
            with self._recovery_guard:
                self._recovering.discard(view_id)

    def _persist_inconsistency_locked(self, view_id: str, reason: str) -> None:
        with self._state_lock:
            runtime = self.runtime.get(view_id)
            if runtime is None:
                return
            runtime.inconsistent = True
            runtime.inconsistent_reason = reason
            result = runtime.to_store()
            lease = self.lease_manager.snapshot(view_id)
            if lease is None:
                lease = {
                    "view_id": view_id,
                    "param_version": self.param_version,
                    "refresh_version": runtime.result_version,
                    "state": "expired",
                    "granted_mono_ms": 0,
                    "expires_mono_ms": 0,
                    "granted_wall_ms": 0,
                    "expires_wall_ms": 0,
                    "ttl_ms": runtime.view.lease_ttl_ms,
                    "renew_ms": runtime.view.lease_renew_ms,
                    "renew_count": 0,
                    "last_renew_mono_ms": None,
                    "job_id": None,
                    "stale": False,
                    "note": reason,
                }
        self.store.persist_inconsistency(
            param_version=self.param_version,
            view_id=view_id,
            result=result,
            lease=lease,
        )

    def _rebuild_view_locked(self, view_id: str, reason: str) -> dict[str, Any]:
        assert self.scheduler is not None
        with self._state_lock:
            runtime = self.runtime.get(view_id)
            if runtime is None:
                fail("E_STATE", "cannot rebuild an unknown view", {"view_id": view_id})
            view = runtime.view
            old_digest = runtime.result_digest
            cursor = self.cursor
            param_version = self.param_version
        refresh_version = self._next_refresh_version(view_id)
        started_mono = monotonic_ms()

        def job_fn() -> MaintenanceOutcome:
            rows = replay_base_rows(self.base_initial.rows, self.changes.changes, cursor=cursor)
            return build_view_result(view, self.definitions.fixed, rows, cursor=cursor, param_version=param_version)

        handle = self.scheduler.submit(
            JobSpec(
                view_id=view_id,
                kind="full_rebuild",
                param_version=param_version,
                refresh_version=refresh_version,
                ttl_ms=view.lease_ttl_ms,
                renew_ms=view.lease_renew_ms,
                fn=job_fn,
                batch_id=f"rebuild-{view_id}-{refresh_version}",
                per_view_quota=view.per_view_quota,
            )
        )
        values, errors, pending = self.scheduler.wait_all([handle])
        if pending or errors:
            error = errors[0] if errors else BusinessError("E_SCHEDULE", "rebuild still running", {"view_id": view_id})
            self._record_event(error, "rebuild_failure")
            raise error
        outcome = values[0]
        finished_mono = monotonic_ms()
        lease = self.lease_manager.snapshot(view_id)
        if lease is None:
            fail("E_LEASE", "rebuild completed without a lease", {"view_id": view_id})
        result = {
            "result_version": refresh_version,
            "cursor": cursor,
            "result_json": outcome.result_json,
            "result_digest": outcome.digest,
            "stale": False,
            "inconsistent": False,
            "updated_wall_ms": wall_ms(),
        }
        trace = {
            "view_id": view_id,
            "param_version": param_version,
            "refresh_version": refresh_version,
            "reason": reason,
            "status": "rebuilt",
            "old_digest": old_digest,
            "new_digest": outcome.digest,
            "started_mono_ms": handle.trace.started_mono_ms or started_mono,
            "finished_mono_ms": finished_mono,
            "lease": lease,
            "cursor": cursor,
            "group_count": outcome.group_count,
        }
        receipt = self.store.commit_rebuild(
            param_version=param_version,
            view_id=view_id,
            result=result,
            lease=lease,
            job=handle.trace.payload(),
            trace=trace,
        )
        with self._state_lock:
            self.runtime[view_id] = RuntimeView(
                view=view,
                result_json=outcome.result_json,
                result_digest=outcome.digest,
                result_version=refresh_version,
                cursor=cursor,
                stale=False,
                inconsistent=False,
                inconsistent_reason=None,
                updated_wall_ms=result["updated_wall_ms"],
            )
        self.logger.emit(
            "INFO",
            "view_rebuilt",
            "lease-expired view rebuilt from base table and change stream under a new lease",
            details={
                "view_id": view_id,
                "refresh_version": refresh_version,
                "old_digest": old_digest,
                "new_digest": outcome.digest,
                "trace_digest": digest_obj(trace),
                "snapshot_version": receipt.snapshot_version,
            },
        )
        return trace

    def rebuild_view(self, view_id: str, reason: str = "manual rebuild") -> dict[str, Any]:
        with self._batch_lock:
            self._persist_inconsistency_locked(view_id, reason)
            return self._rebuild_view_locked(view_id, reason)

    def simulate_lease_expiry(self, view_id: str, *, ttl_ms: int = 140, rebuild: bool | None = None) -> dict[str, Any]:
        with self._batch_lock:
            with self._state_lock:
                if view_id not in self.runtime:
                    fail("E_LEASE", "unknown view", {"view_id": view_id})
            version = self._next_refresh_version(view_id)
            renew = max(1, min(ttl_ms - 1, max(20, ttl_ms // 3)))
            if renew >= ttl_ms:
                renew = max(1, ttl_ms - 1)
            lease = self.lease_manager.issue(
                view_id,
                self.param_version,
                version,
                ttl_ms,
                renew,
                job_id="lease-expiry-probe",
            )
            time.sleep((ttl_ms + 25) / 1000.0)
            self.lease_manager.expire_due()
            self._persist_inconsistency_locked(view_id, "simulated lease expiry")
            current_lease = self.lease_manager.get(view_id)
            should_rebuild = self.auto_rebuild_on_expiry if rebuild is None else rebuild
        if should_rebuild:
            self._start_recovery(view_id, reason="simulated lease expiry", do_rebuild=True)
        return {
            "view_id": view_id,
            "lease": lease.payload(),
            "expired": current_lease is not None and current_lease.state == "expired",
            "inconsistent": True,
            "auto_rebuild_scheduled": should_rebuild,
        }

    def wait_for_recovery(self, view_id: str, timeout_ms: int = 5000) -> bool:
        deadline = monotonic_ms() + timeout_ms
        while monotonic_ms() < deadline:
            with self._recovery_guard:
                recovering = view_id in self._recovering
            with self._state_lock:
                runtime = self.runtime.get(view_id)
                clear = runtime is not None and not runtime.inconsistent
            if clear and not recovering:
                return True
            time.sleep(0.01)
        return False

    # ------------------------------------------------------------------
    # Definition replacement
    # ------------------------------------------------------------------
    def apply_views_file(self, path: str | Path) -> dict[str, Any]:
        definitions = load_definitions(path)
        validate_definitions_against_base(definitions, self.base_initial)
        with self._batch_lock:
            if definitions.fingerprint == self.definitions.fingerprint:
                return {"status": "unchanged", "param_version": self.param_version, "summary": None}
            self._replace_definitions_locked(definitions, startup=False)
            return {
                "status": "replaced",
                "param_version": self.param_version,
                "summary": self.definition_replacement_summary,
            }

    def _replace_definitions_locked(self, new_definitions: Definitions, *, startup: bool) -> None:
        assert self.scheduler is not None
        old_param_version = self.param_version
        new_param_version = old_param_version + 1
        old_summary = self.store.current_param_summary()
        old_view_defs = self.store.load_current_view_defs()
        old_view_ids = set(old_view_defs)
        new_view_ids = set(new_definitions.views)
        added = sorted(new_view_ids - old_view_ids)
        removed = sorted(old_view_ids - new_view_ids)
        changed = sorted(
            view_id
            for view_id in old_view_ids & new_view_ids
            if old_view_defs[view_id]["digest"] != new_definitions.views[view_id].digest
        )
        unchanged = sorted((old_view_ids & new_view_ids) - set(changed))
        before_views = {
            view_id: {
                "digest": payload["digest"],
                "group_by": payload["payload"].get("group_by", []),
            }
            for view_id, payload in sorted(old_view_defs.items())
        }
        after_views = {
            view_id: {
                "digest": new_definitions.views[view_id].digest,
                "group_by": list(new_definitions.views[view_id].group_by),
            }
            for view_id in sorted(new_definitions.views)
        }
        old_definition_summary = None if old_summary is None else old_summary.get("summary", {})
        replacement_summary = {
            "old_param_version": old_param_version,
            "new_param_version": new_param_version,
            "added": added,
            "removed": removed,
            "changed": changed,
            "unchanged": unchanged,
            "before": {
                "fingerprint": None if old_summary is None else old_summary["fingerprint"],
                "view_count": len(old_view_ids),
                "views": before_views,
                "scale_factor": None if old_definition_summary is None else old_definition_summary.get("scale_factor"),
                "truncation": None if old_definition_summary is None else old_definition_summary.get("truncation"),
                "overflow": None if old_definition_summary is None else old_definition_summary.get("overflow"),
            },
            "after": {
                "fingerprint": new_definitions.fingerprint,
                "view_count": len(new_view_ids),
                "views": after_views,
                "scale_factor": new_definitions.session.scale_factor,
                "truncation": new_definitions.session.truncation,
                "overflow": new_definitions.session.overflow,
            },
            "startup": startup,
        }
        self.scheduler.per_view_quotas = {
            view_id: view.per_view_quota for view_id, view in new_definitions.views.items()
        }
        rows = replay_base_rows(self.base_initial.rows, self.changes.changes, cursor=self.cursor)
        handles: list[JobHandle] = []
        for view_id in sorted(new_definitions.views):
            view = new_definitions.views[view_id]
            refresh_version = self._next_refresh_version(view_id)

            def job_fn(view=view, rows=rows) -> MaintenanceOutcome:
                return build_view_result(
                    view,
                    new_definitions.fixed,
                    rows,
                    cursor=self.cursor,
                    param_version=new_param_version,
                )

            handles.append(
                self.scheduler.submit(
                    JobSpec(
                        view_id=view_id,
                        kind="definition_rebuild",
                        param_version=new_param_version,
                        refresh_version=refresh_version,
                        ttl_ms=view.lease_ttl_ms,
                        renew_ms=view.lease_renew_ms,
                        fn=job_fn,
                        batch_id=f"definition-{old_param_version}-{new_param_version}",
                        per_view_quota=view.per_view_quota,
                    )
                )
            )
        values, errors, pending = self.scheduler.wait_all(handles)
        if pending or errors:
            raise errors[0] if errors else BusinessError("E_SCHEDULE", "definition rebuild did not finish", {"pending": len(pending)})
        outcomes = {handle.spec.view_id: value for handle, value in zip(handles, values)}
        results = self._results_from_outcomes(
            outcomes,
            definitions=new_definitions,
            param_version=new_param_version,
        )
        leases = self._current_lease_payloads(new_definitions.views.keys())
        jobs = self.scheduler.traces_for([handle.job_id for handle in handles])
        base_payload = canonical_rows_payload(
            self.base_initial.table_name,
            self.base_initial.row_id_column,
            self.base_initial.columns,
            rows,
        )
        receipt = self.store.replace_definitions(
            old_param_version=old_param_version,
            new_param_version=new_param_version,
            definitions=new_definitions,
            cursor=self.cursor,
            base_payload=base_payload,
            results=results,
            leases=leases,
            jobs=jobs,
            replacement_summary=replacement_summary,
        )
        with self._state_lock:
            self.definitions = new_definitions
            self.param_version = new_param_version
            self.base_rows = rows
            self.runtime = self._runtime_from_outcomes(
                outcomes,
                definitions=new_definitions,
                param_version=new_param_version,
            )
        self.definition_replacement_summary = replacement_summary
        self.logger.emit(
            "INFO",
            "definitions_replaced",
            "view definitions and scaling parameters replaced as a whole; old results and leases marked stale",
            details={
                "old_param_version": old_param_version,
                "new_param_version": new_param_version,
                "added": added,
                "removed": removed,
                "changed": changed,
                "replacement_digest": digest_obj(replacement_summary),
                "snapshot_version": receipt.snapshot_version,
            },
        )

    # ------------------------------------------------------------------
    # Replay verification
    # ------------------------------------------------------------------
    def replay_verify(self) -> dict[str, Any]:
        with self._batch_lock:
            snapshot = self.store.latest_snapshot_payload()
            if int(snapshot.get("param_version", -1)) != self.param_version:
                self._raise_replay(
                    "param_version",
                    [],
                    "__meta__.param_version",
                    snapshot.get("param_version"),
                    self.param_version,
                )
            if snapshot.get("definitions_fingerprint") != self.definitions.fingerprint:
                self._raise_replay("__session__", [], "__meta__.definitions_fingerprint", snapshot.get("definitions_fingerprint"), self.definitions.fingerprint)
            cursor = int(snapshot.get("cursor", -1))
            replay_rows = {row_id: dict(row) for row_id, row in self.base_initial.rows.items()}
            per_view_previous: dict[str, MaintenanceOutcome] = {}
            for view_id in sorted(self.definitions.views):
                per_view_previous[view_id] = build_view_result(
                    self.definitions.views[view_id],
                    self.definitions.fixed,
                    replay_rows,
                    cursor=0,
                    param_version=self.param_version,
                )
            for change in self.changes.changes[:cursor]:
                before = replay_rows.get(change.row_id)
                if change.op == "delete":
                    after = None
                    replay_rows.pop(change.row_id, None)
                else:
                    after = dict(change.row)
                    replay_rows[change.row_id] = after
                transition = Transition(change.seq, change.op, change.row_id, copy.deepcopy(before), copy.deepcopy(after))
                for view_id in sorted(self.definitions.views):
                    per_view_previous[view_id] = maintain_view_incrementally(
                        self.definitions.views[view_id],
                        self.definitions.fixed,
                        per_view_previous[view_id],
                        replay_rows,
                        [transition],
                        cursor_before=change.seq - 1,
                        cursor_after=change.seq,
                        param_version=self.param_version,
                    )
            replay_base_payload = canonical_rows_payload(
                self.base_initial.table_name,
                self.base_initial.row_id_column,
                self.base_initial.columns,
                replay_rows,
            )
            if replay_base_payload != snapshot.get("base_current"):
                self._raise_replay("__base__", [], "base_current", snapshot.get("base_current"), replay_base_payload)
            ledger_results = snapshot.get("results", {})
            current_leases = self.lease_manager.all_payload()
            ledger_leases = snapshot.get("leases", {})
            report_views: dict[str, Any] = {}
            for view_id in sorted(self.definitions.views):
                replay = per_view_previous[view_id]
                ledger_result = ledger_results.get(view_id)
                if ledger_result is None:
                    self._raise_replay(view_id, [], "__view__", None, replay.digest)
                difference = find_first_difference(view_id, ledger_result["result_json"], replay.payload)
                if difference is not None:
                    self._raise_replay(
                        view_id,
                        difference.group_key,
                        difference.field,
                        difference.cached,
                        difference.authoritative,
                    )
                ledger_lease = ledger_leases.get(view_id)
                current_lease = current_leases.get(view_id)
                if ledger_lease is None or current_lease is None:
                    self._raise_replay(view_id, [], "lease.state", ledger_lease, current_lease)
                for field in ("state", "refresh_version", "stale"):
                    if ledger_lease.get(field) != current_lease.get(field):
                        self._raise_replay(view_id, [], f"lease.{field}", ledger_lease.get(field), current_lease.get(field))
                with self._state_lock:
                    runtime = self.runtime.get(view_id)
                    if runtime is None:
                        self._raise_replay(view_id, [], "__runtime__", None, "present")
                    expected_inconsistent = bool(ledger_result.get("inconsistent", False))
                    if runtime.inconsistent != expected_inconsistent:
                        self._raise_replay(view_id, [], "inconsistent", expected_inconsistent, runtime.inconsistent)
                    if runtime.result_json != ledger_result["result_json"]:
                        difference = find_first_difference(view_id, runtime.result_json, replay.payload)
                        self._raise_replay(
                            view_id,
                            [] if difference is None else difference.group_key,
                            "live_cache.result_json" if difference is None else difference.field,
                            runtime.result_json if difference is None else difference.cached,
                            replay.payload if difference is None else difference.authoritative,
                        )
                report_views[view_id] = {
                    "digest": replay.digest,
                    "group_count": replay.group_count,
                    "lease_state": current_lease.get("state"),
                    "inconsistent": bool(ledger_result.get("inconsistent", False)),
                    "byte_identical": True,
                }
            ledger_traces = snapshot.get("rebuild_traces", [])
            current_traces = [
                trace
                for trace in reversed(self.store.list_rebuild_traces(limit=10_000))
                if int(trace.get("param_version", -1)) == self.param_version
            ]
            if ledger_traces != current_traces:
                for index in range(max(len(ledger_traces), len(current_traces))):
                    expected_trace = ledger_traces[index] if index < len(ledger_traces) else None
                    actual_trace = current_traces[index] if index < len(current_traces) else None
                    if expected_trace == actual_trace:
                        continue
                    trace_view = (expected_trace or actual_trace or {}).get("view_id", "__session__")
                    differing_field = "rebuild_traces"
                    if expected_trace is not None and actual_trace is not None:
                        differing_field = next(
                            (
                                key
                                for key in sorted(set(expected_trace) | set(actual_trace))
                                if expected_trace.get(key) != actual_trace.get(key)
                            ),
                            "rebuild_traces",
                        )
                    self._raise_replay(
                        trace_view,
                        [],
                        f"rebuild_trace.{differing_field}",
                        expected_trace,
                        actual_trace,
                    )
            batches = self.store.list_batches(limit=1)
            if batches and batches[0]["cursor"] != cursor:
                self._raise_replay("__session__", [], "ledger.cursor", batches[0]["cursor"], cursor)
            report = {
                "status": "matched",
                "code": None,
                "param_version": self.param_version,
                "cursor": cursor,
                "snapshot_version": snapshot.get("snapshot_version"),
                "views": report_views,
                "rebuild_trace_count": len(current_traces),
                "byte_identical": True,
            }
            self.logger.emit(
                "INFO",
                "replay_verified",
                "ledger parameter version, cursor, results, lease states, inconsistency flags and rebuild traces matched replay",
                details={"param_version": self.param_version, "cursor": cursor, "views": sorted(report_views)},
            )
            return report

    def _raise_replay(self, view_id: str, key: list[Any], field: str, expected: Any, actual: Any) -> None:
        error = BusinessError(
            "E_REPLAY",
            f"replay verification failed at field {field}",
            {"view_id": view_id, "group_key": key, "field": field},
            {"ledger_or_replay": expected, "current": actual},
        )
        self._record_event(error, "replay_verification")
        raise error

    # ------------------------------------------------------------------
    # Anomaly injection
    # ------------------------------------------------------------------
    def inject_lease_conflict(self) -> dict[str, Any]:
        manager = LeaseManager()
        view_id = sorted(self.definitions.views)[0]
        record = manager.issue(view_id, self.param_version, 1, 5_000, 1_000)
        try:
            manager.renew(view_id, record.refresh_version + 1)
        except BusinessError as exc:
            self._record_event(exc, "anomaly_injection")
            return {"status": "reported", "code": exc.code, "location": exc.location, "details": exc.details}
        fail("E_LEASE", "lease conflict injection did not produce E_LEASE")

    def inject_base_corruption(self) -> dict[str, Any]:
        state = self.store.load_state()
        payload = copy.deepcopy(state.base_rows_payload)
        rows = payload.get("rows", [])
        if not rows:
            fail("E_DATA", "no base row is available for corruption injection")
        row = rows[0]
        numeric_column = next(
            (
                column["name"]
                for column in payload.get("columns", [])
                if column["type"] in {"integer", "float"} and column["name"] != payload.get("row_id_column")
            ),
            None,
        )
        if numeric_column is None:
            fail("E_DATA", "no numeric base column is available for corruption injection")
        row.setdefault("values", {})[numeric_column] = {"corrupt": True}
        try:
            self.store.verify_base_payload(payload)
        except BusinessError as exc:
            self._record_event(exc, "anomaly_injection")
            return {
                "status": "reported",
                "code": exc.code,
                "location": exc.location,
                "details": exc.details,
                "injected_row_id": row.get("row_id"),
                "injected_field": numeric_column,
            }
        fail("E_DATA", "base corruption injection was not detected")

    def inject_snapshot_tamper(self) -> dict[str, Any]:
        tamper = self.store.tamper_result_digest()
        audit = self.store.validate_and_restore_snapshot()
        if audit.error is not None:
            self._record_event(audit.error, "anomaly_injection")
        if not audit.recovered:
            fail("E_SNAPSHOT", "snapshot tamper was not recovered", details={"tamper": tamper})
        return {
            "status": "recovered",
            "code": audit.error.code if audit.error is not None else "E_SNAPSHOT",
            "tamper": tamper,
            "recovered": audit.recovered,
            "snapshot_version": audit.snapshot_version,
        }

    # ------------------------------------------------------------------
    # Presentation and lifecycle helpers
    # ------------------------------------------------------------------
    def event_log(self, limit: int = 200) -> list[dict[str, Any]]:
        return self.logger.read_tail(limit)

    def ledger_batches(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.store.list_batches(limit)

    def scheduler_jobs(self, limit: int = 200) -> list[dict[str, Any]]:
        return self.store.list_scheduler_jobs(limit)

    def rebuild_traces(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.store.list_rebuild_traces(limit)

    def replacements(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.store.list_replacements(limit)

    def status(self) -> dict[str, Any]:
        snapshot = self.fast_snapshot()
        return {
            "param_version": snapshot["param_version"],
            "cursor": snapshot["cursor"],
            "total_changes": snapshot["total_changes"],
            "batch_counter": snapshot["batch_counter"],
            "visible_views": sum(1 for view in snapshot["views"] if view["visible"]),
            "inconsistent_views": sum(1 for view in snapshot["views"] if view["inconsistent"]),
            "recovering": snapshot["recovering"],
            "scheduler": snapshot["scheduler"],
            "snapshot": snapshot["snapshot"],
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._shutdown_resources()

    def _shutdown_resources(self) -> None:
        self._watchdog_stop.set()
        if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
            self._watchdog_thread.join(timeout=2.0)
        if self.scheduler is not None:
            self.scheduler.shutdown(wait=True)
            self.scheduler = None
        try:
            self.store.close()
        except Exception:
            pass

    def __enter__(self) -> "Workbench":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _next_refresh_version(self, view_id: str) -> int:
        with self._state_lock:
            lease = self.lease_manager.get(view_id)
            current = max(self._refresh_versions.get(view_id, 0), 0 if lease is None else lease.refresh_version)
            next_version = current + 1
            self._refresh_versions[view_id] = next_version
            return next_version

    def _current_lease_payloads(self, view_ids: Any | None = None) -> dict[str, dict[str, Any]]:
        payloads: dict[str, dict[str, Any]] = {}
        source_ids = self.definitions.views.keys() if view_ids is None else view_ids
        for view_id in sorted(source_ids):
            lease = self.lease_manager.snapshot(view_id)
            if lease is not None:
                payloads[view_id] = lease
        return payloads

    def _results_from_outcomes(
        self,
        outcomes: dict[str, MaintenanceOutcome],
        *,
        definitions: Definitions | None = None,
        param_version: int | None = None,
    ) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        for view_id in sorted(outcomes):
            outcome = outcomes[view_id]
            lease = self.lease_manager.snapshot(view_id)
            if lease is None:
                fail("E_LEASE", "refresh result has no lease", {"view_id": view_id})
            results[view_id] = {
                "result_version": lease["refresh_version"],
                "cursor": outcome.payload["cursor"],
                "result_json": outcome.result_json,
                "result_digest": outcome.digest,
                "stale": False,
                "inconsistent": False,
                "updated_wall_ms": wall_ms(),
            }
        return results

    def _runtime_from_outcomes(
        self,
        outcomes: dict[str, MaintenanceOutcome],
        *,
        definitions: Definitions | None = None,
        param_version: int | None = None,
    ) -> dict[str, RuntimeView]:
        runtime: dict[str, RuntimeView] = {}
        for view_id in sorted(outcomes):
            outcome = outcomes[view_id]
            lease = self.lease_manager.snapshot(view_id)
            if lease is None:
                fail("E_LEASE", "refresh result has no lease", {"view_id": view_id})
            active_definitions = definitions or self.definitions
            view = active_definitions.views[view_id]
            runtime[view_id] = RuntimeView(
                view=view,
                result_json=outcome.result_json,
                result_digest=outcome.digest,
                result_version=int(lease["refresh_version"]),
                cursor=int(outcome.payload["cursor"]),
                stale=False,
                inconsistent=False,
                inconsistent_reason=None,
                updated_wall_ms=wall_ms(),
            )
        return runtime

    def _record_event(self, error: BusinessError, event: str) -> None:
        record = self.logger.emit(
            "ERROR",
            event,
            error.message,
            code=error.code,
            location=error.location,
            details=error.details,
        )
        self.startup_events.append(record)


def runtime_reason(runtime: RuntimeView) -> str:
    return runtime.inconsistent_reason or "inconsistent result"
