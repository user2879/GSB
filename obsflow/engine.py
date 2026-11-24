"""Orchestrating engine tying together every obsflow subsystem.

Public high-level operations (also re-exported from the package root):

* :meth:`ObsFlowEngine.load_source_set` / load declarations
* :meth:`ObsFlowEngine.read_records` - decode & normalise record streams
* :meth:`ObsFlowEngine.advance` - concurrent window aggregation
* :meth:`ObsFlowEngine.query_windows` - read window conclusions
* :meth:`ObsFlowEngine.save_snapshot` / :meth:`ObsFlowEngine.recover`
* :meth:`ObsFlowEngine.replay_verify`
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from typing import Optional

from .aggregation import (
    aggregate_all,
    aggregate_window,
    group_by_metric,
    ordered_records,
    window_slices,
)
from .canonical import content_digest
from .clock import MonotonicClock, SystemClock
from .codec import load_stream, resolve_stream_path
from .config import EngineConfig, SourceSetConfig, load_engine_config, load_source_set
from .errors import ReplayError
from .lease import LeaseManager
from .ledger import LEDGER_FORMAT_VERSION, Ledger, SnapshotIdentity
from .models import (
    NormalizedRecord,
    PropagationEvent,
    WindowResult,
)
from .propagation import (
    PropagationPlan,
    SourceChange,
    derive_propagation,
    diff_source_sets,
)
from .scheduler import AggregationTask, BoundedScheduler


@dataclass(frozen=True, slots=True)
class AdvanceReport:
    batch_id: int
    windows: tuple[WindowResult, ...]
    recomputed: tuple[tuple[str, int], ...]
    reused: tuple[tuple[str, int], ...]
    propagation: tuple[PropagationEvent, ...]
    unknown: tuple[tuple[str, int], ...]


class ObsFlowEngine:
    def __init__(self, engine_config: EngineConfig,
                 source_set: SourceSetConfig,
                 ledger_path: Optional[str] = None,
                 clock: Optional[MonotonicClock] = None,
                 backup_dir: Optional[str] = None):
        self.config = engine_config
        self.source_set = source_set
        self.clock = clock or SystemClock()
        self.lease_manager = LeaseManager(self.clock,
                                          engine_config.lease_ttl_ms)
        self._records: list[NormalizedRecord] = []
        self._windows: dict[tuple[str, int], WindowResult] = {}
        self._window_versions: dict[tuple[str, int], int] = {}
        self._propagation: list[PropagationEvent] = []
        self._cursors: dict[str, tuple[int, int]] = {}
        self._batch_seq = 0
        self._last_identity: Optional[SnapshotIdentity] = None
        self.backup_dir = backup_dir or os.path.join(
            os.path.dirname(os.path.abspath(ledger_path)) if ledger_path
            else ".", "backups")
        self.ledger: Optional[Ledger] = None
        if ledger_path:
            self.ledger = Ledger(ledger_path)

    # ------------------------------------------------------------- factories
    @classmethod
    def from_files(cls, declaration_path: str, window_path: str,
                   ledger_path: Optional[str] = None,
                   clock: Optional[MonotonicClock] = None
                   ) -> "ObsFlowEngine":
        source_set = load_source_set(declaration_path)
        engine_config = load_engine_config(window_path)
        return cls(engine_config, source_set, ledger_path, clock)

    # ------------------------------------------------------------ ingestion
    def read_records(self) -> list[NormalizedRecord]:
        """Load, validate and normalise every declared source stream."""
        records: list[NormalizedRecord] = []
        for declaration in self.source_set.sources:
            path = resolve_stream_path(self.source_set.base_dir,
                                       declaration.file_path)
            records.extend(load_stream(path, declaration))
        self._records = ordered_records(records)
        return list(self._records)

    def add_records(self, records: list[NormalizedRecord]) -> None:
        self._records = ordered_records(self._records + records)

    def _group(self) -> dict[str, list[NormalizedRecord]]:
        return group_by_metric(self._records)

    # ------------------------------------------------------------- identity
    def current_identity(self,
                         windows: list[WindowResult]) -> SnapshotIdentity:
        return Ledger.build_identity(
            source_set_version=self.source_set.source_set_version,
            format_versions={s.source_id: s.format_version
                             for s in self.source_set.sources},
            scale_convention_version=self.config.scale_convention_version,
            session_scale=self.config.windows.scale,
            window_payload=Ledger.canonical_window_payload(windows),
        )

    def save_snapshot(self) -> SnapshotIdentity:
        windows = sorted(self._windows.values(),
                         key=lambda w: (w.metric_key, w.window_index))
        identity = self.current_identity(windows)
        self._last_identity = identity
        if self.ledger is not None:
            self.ledger.write_identity(identity)
            self.ledger.set_meta("ledger_format_version",
                                 LEDGER_FORMAT_VERSION)
            self.ledger._conn.commit()
        return identity

    # ------------------------------------------------------- full aggregation
    def _compute_all(self,
                     force_refs: Optional[set[tuple[str, int]]] = None
                     ) -> tuple[list[WindowResult], set[tuple[str, int]]]:
        """Recompute windows; only bump versions when inputs changed."""
        grouped = self._group()
        fresh = aggregate_all(grouped, self.config.windows,
                              versions=self._window_versions)
        results: list[WindowResult] = []
        recomputed: set[tuple[str, int]] = set()
        for result in fresh:
            key = (result.metric_key, result.window_index)
            previous = self._windows.get(key)
            forced = force_refs is not None and key in force_refs
            if previous is None:
                version = self._window_versions.get(key, 1)
                self._window_versions[key] = version
                result = self._with_version(result, version)
                recomputed.add(key)
            elif previous.input_digest != result.input_digest or forced:
                version = previous.window_version + 1
                self._window_versions[key] = version
                result = self._with_version(result, version)
                recomputed.add(key)
            else:
                # Reuse byte-for-byte: keep prior result incl. version/lease.
                result = previous
            results.append(result)
        results.sort(key=lambda w: (w.metric_key, w.window_index))
        self._windows = {
            (w.metric_key, w.window_index): w for w in results
        }
        return results, recomputed

    @staticmethod
    def _with_version(result: WindowResult, version: int) -> WindowResult:
        payload = {
            "metric_key": result.metric_key,
            "window_index": result.window_index,
            "start_source_id": result.start_source_id,
            "end_source_id": result.end_source_id,
            "start_seq": result.start_seq,
            "end_seq": result.end_seq,
            "count": result.count,
            "sum_q": result.sum_q,
            "mean_q": result.mean_q,
            "value_error_bound_q": result.value_error_bound_q,
            "sum_error_bound_q": result.sum_error_bound_q,
            "scale": result.scale,
            "truncation": result.truncation,
            "saturation": result.saturation,
            "window_version": version,
            "status": result.status,
            "input_digest": result.input_digest,
        }
        return WindowResult(
            metric_key=result.metric_key,
            window_index=result.window_index,
            start_source_id=result.start_source_id,
            end_source_id=result.end_source_id,
            start_seq=result.start_seq,
            end_seq=result.end_seq,
            count=result.count,
            sum_q=result.sum_q,
            mean_q=result.mean_q,
            value_error_bound_q=result.value_error_bound_q,
            sum_error_bound_q=result.sum_error_bound_q,
            scale=result.scale,
            truncation=result.truncation,
            saturation=result.saturation,
            window_version=version,
            status=result.status,
            input_digest=result.input_digest,
            result_digest=content_digest(payload),
            lease_id=result.lease_id,
        )

    # ------------------------------------------------------------ advancing
    def advance(self,
                plan: Optional[PropagationPlan] = None,
                *,
                use_scheduler: bool = True) -> AdvanceReport:
        """Aggregate the current records into windows.

        When ``plan`` is supplied, only windows in its affected set are
        recomputed (forced invalidation); every other window stays byte
        identical including its window version.
        """
        force_refs: Optional[set[tuple[str, int]]] = None
        if plan is not None:
            force_refs = set(plan.affected)
            self._propagation.extend(plan.events)

        grouped = self._group()
        fresh = aggregate_all(grouped, self.config.windows,
                              versions=self._window_versions)
        candidate = {
            (w.metric_key, w.window_index): w for w in fresh
        }

        recomputed: set[tuple[str, int]] = set()
        final: list[WindowResult] = []
        tasks: list[AggregationTask[WindowResult]] = []

        for key in sorted(candidate):
            result = candidate[key]
            previous = self._windows.get(key)
            forced = force_refs is not None and key in force_refs
            unknown = previous is not None and previous.status == "unknown"
            if previous is None:
                version = self._window_versions.get(key, 1)
                self._window_versions[key] = version
                result = self._with_version(result, version)
                recomputed.add(key)
            elif previous.input_digest != result.input_digest or forced \
                    or unknown:
                # Unknown windows come from an uncommitted batch and must be
                # recomputed even when their inputs are unchanged.
                version = previous.window_version + 1 if not unknown \
                    else previous.window_version
                self._window_versions[key] = version
                result = self._with_version(result, version)
                recomputed.add(key)
            else:
                result = previous  # byte-for-byte reuse
            final.append(result)
            if key in recomputed:
                owner = result.end_source_id
                tasks.append(AggregationTask(
                    task_id=f"task-{key[0]}-{key[1]}",
                    metric_key=key[0],
                    window_index=key[1],
                    source_id=owner,
                    payload=result,
                ))

        timings = self._run_tasks(tasks, use_scheduler)
        leased: dict[tuple[str, int], str] = {}
        for task, timing in zip(tasks, timings):
            result = task.payload
            key = (task.metric_key, task.window_index)
            previous = self._windows.get(key)
            if previous is not None and previous.lease_id:
                self.lease_manager.release(previous.lease_id)
            lease = self.lease_manager.grant(
                result.metric_key, result.window_index, result.start_seq,
                result.end_seq)
            leased[key] = lease.lease_id
        final = [
            replace(w, lease_id=leased[(w.metric_key, w.window_index)])
            if (w.metric_key, w.window_index) in leased else w
            for w in final
        ]
        final.sort(key=lambda w: (w.metric_key, w.window_index))
        self._windows = {(w.metric_key, w.window_index): w for w in final}

        unknown = tuple(sorted(
            key for key, w in self._windows.items() if w.status == "unknown"))
        reused = tuple(sorted(
            key for key in self._windows if key not in recomputed))
        report = AdvanceReport(
            batch_id=self._batch_seq + 1,
            windows=tuple(final),
            recomputed=tuple(sorted(recomputed)),
            reused=reused,
            propagation=tuple(self._propagation),
            unknown=unknown,
        )
        if self.ledger is not None:
            self._commit_batch(report)
        self._batch_seq += 1
        return report

    def _run_tasks(self, tasks, use_scheduler: bool):
        if not tasks:
            return []
        if not use_scheduler:
            now = self.clock.now_ms()
            from .models import TaskTiming
            return [
                TaskTiming(metric_key=t.metric_key,
                           window_index=t.window_index,
                           source_id=t.source_id,
                           enqueued_ms=now, wait_start_ms=now,
                           wait_end_ms=now, run_start_ms=now,
                           run_end_ms=now)
                for t in tasks
            ]
        scheduler = BoundedScheduler(
            global_concurrency=self.config.global_concurrency,
            per_source_concurrency=self.config.per_source_concurrency,
            clock=self.clock)

        def worker(task: AggregationTask[WindowResult]):
            # Recompute on the worker thread to project work onto real threads.
            return aggregate_window(
                self._window_records(task.metric_key, task.window_index),
                task.metric_key, task.window_index, self.config.windows,
                task.payload.window_version)

        _, timings = scheduler.run(tasks, worker)
        self._last_scheduler = scheduler
        return timings

    def _window_records(self, metric_key: str,
                        window_index: int) -> list[NormalizedRecord]:
        ordered = ordered_records(self._group()[metric_key])
        start = window_index * self.config.windows.slide_step
        end = start + self.config.windows.window_size
        return ordered[start:end]

    # ------------------------------------------------------------ queries
    def query_windows(self,
                      metric_key: Optional[str] = None) -> list[WindowResult]:
        windows = list(self._windows.values())
        if metric_key is not None:
            windows = [w for w in windows if w.metric_key == metric_key]
        return sorted(windows, key=lambda w: (w.metric_key, w.window_index))

    def propagation_trail(self) -> list[PropagationEvent]:
        return list(self._propagation)

    # ------------------------------------------------------------ changes
    def apply_source_set(self, new_set: SourceSetConfig
                         ) -> PropagationPlan:
        old = self.source_set.by_id()
        new = new_set.by_id()
        changes = diff_source_sets(old, new)
        plan = derive_propagation(
            old, new, self.query_windows(), changes,
            window_size_changed=False, slide_step_changed=False)
        self.source_set = new_set
        return plan

    def apply_window_params(self, window_size: int,
                            slide_step: int) -> PropagationPlan:
        old_params = self.config.windows
        new_params = old_params.__class__(
            window_size=window_size, slide_step=slide_step,
            scale=old_params.scale)
        new_params.validate()
        plan = derive_propagation(
            self.source_set.by_id(), self.source_set.by_id(),
            self.query_windows(), [],
            window_size_changed=(window_size != old_params.window_size),
            slide_step_changed=(slide_step != old_params.slide_step))
        self.config = EngineConfig(
            windows=new_params,
            global_concurrency=self.config.global_concurrency,
            per_source_concurrency=self.config.per_source_concurrency,
            lease_ttl_ms=self.config.lease_ttl_ms,
            scale_convention_version=self.config.scale_convention_version)
        # Grid geometry changes reset the version map.
        self._window_versions = {}
        return plan

    # ------------------------------------------------------------ ledger
    def _commit_batch(self, report: AdvanceReport) -> None:
        assert self.ledger is not None
        self._batch_seq = self.ledger.latest_batch_id()
        batch_id = self._batch_seq + 1
        now = self.clock.now_ms()
        digest = content_digest([w.result_digest for w in report.windows])
        self.ledger.begin_batch(
            batch_id,
            sorted({s.source_id for s in self.source_set.sources}),
            len(self._records),
            max((r.seq for r in self._records), default=0),
            digest, now)
        leases = self.lease_manager.all_leases()
        for source_id in sorted({r.source_id for r in self._records}):
            source_records = [r for r in self._records
                              if r.source_id == source_id]
            if source_records:
                self._cursors[source_id] = (
                    max(r.seq for r in source_records),
                    max(r.line_no for r in source_records))
        self.ledger.commit_batch(
            batch_id, list(report.windows), leases,
            list(self._propagation), self._cursors, now)
        self.save_snapshot()

    def recover(self) -> tuple[list[WindowResult], list[tuple[str, int]]]:
        """Recover from a possibly force-killed process.

        Only fully committed batches are honoured. Windows belonging to an
        ``in_progress`` batch are marked unknown and must be rerun; no exit
        report is fabricated.
        """
        if self.ledger is None:
            raise ValueError("engine has no ledger; cannot recover")
        stored_identity = self.ledger.snapshot_identity()
        backup_path: Optional[str] = None

        for lease in self.ledger.all_leases():
            # Leases restored from a ledger are ownership records; windows
            # that later require recomputation explicitly release them first.
            self.lease_manager.restore(lease)
        windows = self.ledger.all_windows()
        self._propagation = self.ledger.propagation_trail()
        self._cursors = self.ledger.cursors()
        self._batch_seq = self.ledger.latest_batch_id()

        # Only fully committed batches are trusted. Windows touched by an
        # uncommitted batch are marked unknown and must be rerun; no exit
        # report is fabricated from them.
        unknown_refs: list[tuple[str, int]] = []
        if self.ledger.in_progress_batches():
            unknown_refs = sorted(
                (w.metric_key, w.window_index) for w in windows)
        if unknown_refs:
            self.ledger.mark_windows_unknown(unknown_refs)
            # Leases guarding uncommitted windows cannot participate in later
            # aggregation; revoke them from the in-memory lease manager.
            unknown_set = set(unknown_refs)
            for lease in self.lease_manager.all_leases():
                if (lease.metric_key, lease.window_index) in unknown_set \
                        and lease.state == "active":
                    self.lease_manager.release(lease.lease_id)
            windows = self.ledger.all_windows()
        self._windows = {(w.metric_key, w.window_index): w for w in windows}
        self._window_versions = {
            (w.metric_key, w.window_index): w.window_version for w in windows
        }

        # Version-set comparison decides whether this is the same session.
        # Version evolution -> whole-state replacement (never incremental
        # merge). Within the same session the stored content digest is
        # verified separately as a corruption check (E_SNAPSHOT path).
        if stored_identity is not None:
            current_identity = self.current_identity(self.query_windows())
            version_mismatch = (
                stored_identity.source_set_version
                != current_identity.source_set_version
                or stored_identity.format_versions
                != current_identity.format_versions
                or stored_identity.scale_convention_version
                != current_identity.scale_convention_version)
            if version_mismatch:
                backup_path = self.ledger.replace_state(self.backup_dir)
                self.lease_manager.clear()
                self._windows = {}
                self._window_versions = {}
                self._propagation = []
                self._cursors = {}
                self._last_backup = backup_path
                return [], []
        self._last_backup = backup_path
        return windows, unknown_refs

    def verify_snapshot(self) -> None:
        """Explicit E_SNAPSHOT content-digest verification on reopen."""
        if self.ledger is None:
            raise ValueError("engine has no ledger; cannot verify snapshot")
        stored = self.ledger.snapshot_identity()
        if stored is None:
            return
        current = self.current_identity(self.query_windows())
        self.ledger.verify_content_digest(current)

    @staticmethod
    def _identity_mismatch(stored: SnapshotIdentity,
                           current: SnapshotIdentity) -> bool:
        return (
            stored.source_set_version != current.source_set_version
            or stored.format_versions != current.format_versions
            or stored.scale_convention_version !=
            current.scale_convention_version
            or stored.content_digest != current.content_digest
        )

    @staticmethod
    def _identity_without_digest(identity: SnapshotIdentity
                                 ) -> SnapshotIdentity:
        return SnapshotIdentity(
            source_set_version=identity.source_set_version,
            format_versions=identity.format_versions,
            scale_convention_version=identity.scale_convention_version,
            session_scale=identity.session_scale,
            content_digest="",
        )

    def replay_verify(self) -> list[WindowResult]:
        """Replay remaining records and compare against the ledger field by
        field. The first mismatch is reported with E_REPLAY."""
        if self.ledger is None:
            raise ValueError("engine has no ledger; cannot replay")
        ledger_windows = {
            (w.metric_key, w.window_index): w
            for w in self.ledger.all_windows()
        }
        replayed = aggregate_all(self._group(), self.config.windows,
                                 versions=self._window_versions)
        compare_fields = ("count", "sum_q", "mean_q", "value_error_bound_q",
                          "sum_error_bound_q", "window_version", "status",
                          "result_digest")
        for replayed_window in sorted(replayed,
                                      key=lambda w: (w.metric_key,
                                                     w.window_index)):
            key = (replayed_window.metric_key, replayed_window.window_index)
            stored = ledger_windows.get(key)
            if stored is None:
                raise ReplayError(
                    f"window {key[1]} of metric {key[0]!r} missing in ledger",
                    field="window",
                    detail={"metric_key": key[0], "window_index": key[1]})
            for field_name in compare_fields:
                if getattr(stored, field_name) != \
                        getattr(replayed_window, field_name):
                    raise ReplayError(
                        f"replay mismatch at metric {key[0]!r} window "
                        f"{key[1]} field {field_name!r}: "
                        f"ledger={getattr(stored, field_name)!r} "
                        f"replay={getattr(replayed_window, field_name)!r}",
                        field=field_name,
                        detail={"metric_key": key[0],
                                "window_index": key[1],
                                "ledger_value": getattr(stored, field_name),
                                "replay_value": getattr(
                                    replayed_window, field_name)})
        return replayed

    def close(self) -> None:
        if self.ledger is not None:
            self.ledger.close()
