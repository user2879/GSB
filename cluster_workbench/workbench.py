from __future__ import annotations

import copy
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

from .cache import CacheStore
from .commands import apply_command, command_type, compute_state_fingerprint
from .config import load_config, normalize_config
from .data import generate_table, load_data_description, normalize_data_description
from .errors import (
    E_CONFIG_INVALID,
    E_PIPELINE_ERROR,
    E_RESULT_STALE,
    Issue,
    issue_dicts,
)
from .pipeline import execute_pipeline
from .snapshot import SnapshotStore
from .util import canonical_json, deep_copy, read_yaml_mapping, to_jsonable


class Workbench:
    def __init__(
        self,
        config: dict[str, Any],
        data_description: dict[str, Any],
        *,
        workspace: str | Path = "workbench_state",
        cache_dir: str | Path | None = None,
        injections: list[dict[str, Any]] | None = None,
        base_issues: Iterable[Issue] | None = None,
        restore_snapshot: bool = False,
    ):
        normalized_config, extra_config_issues = normalize_config(config)
        normalized_data, extra_data_issues = normalize_data_description(data_description)
        config = normalized_config
        data_description = normalized_data
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.cache = CacheStore(cache_dir or (self.workspace / "cache"))
        self.snapshots = SnapshotStore(self.workspace / "snapshots")
        merged_injections = _merge_injections(
            list(config.get("injections", []) or []), list(injections or [])
        )
        self.state: dict[str, Any] = {
            "config": deep_copy(config),
            "raw_config": deep_copy(config),
            "data_description": deep_copy(data_description),
            "candidates": deep_copy(config.get("candidates", [])),
            "data_overrides": {},
            "injections": merged_injections,
            "command_seq": 0,
        }
        self.base_issues: list[Issue] = [
            *extra_config_issues,
            *extra_data_issues,
            *list(base_issues or []),
        ]
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="workbench-pipeline")
        self._snapshot_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="workbench-snapshot")
        self._futures: list[Future[Any]] = []
        self._snapshot_futures: list[Future[Any]] = []
        self._latest_report: dict[str, Any] | None = None
        self._latest_report_state_fingerprint: str | None = None
        self._latest_report_command_seq = 0
        self._latest_report_generation = 0
        self._latest_error: dict[str, Any] | None = None
        self._generation = 0
        self._current_job_event: threading.Event | None = None
        self._progress: dict[int, dict[str, Any]] = {}
        self._snapshot_requests: dict[int, dict[str, Any]] = {}
        self._snapshot_inflight: set[int] = set()
        self._snapshot_completed_seq = 0
        self._snapshot_commit_lock = threading.Lock()
        self._latest_snapshot: dict[str, Any] | None = None
        self._closed = False
        self._recovery_issues: list[Issue] = []
        if restore_snapshot:
            self._restore_snapshot()

    @classmethod
    def from_files(
        cls,
        config_path: str | Path,
        data_path: str | Path,
        *,
        injections_path: str | Path | None = None,
        workspace: str | Path = "workbench_state",
        cache_dir: str | Path | None = None,
        restore_snapshot: bool = False,
    ) -> "Workbench":
        config, config_issues = load_config(config_path)
        data_description, data_issues = load_data_description(data_path)
        injections: list[dict[str, Any]] = []
        injection_issues: list[Issue] = []
        if injections_path is not None:
            payload, error = read_yaml_mapping(injections_path)
            if error is not None or payload is None:
                injection_issues.append(
                    Issue(
                        E_CONFIG_INVALID,
                        error or "injection manifest must be a YAML mapping",
                        context={"scope": "injection", "path": str(injections_path)},
                    )
                )
            else:
                supplied = payload.get("injections", [])
                if isinstance(supplied, list):
                    injections = supplied
                else:
                    injection_issues.append(
                        Issue(
                            E_CONFIG_INVALID,
                            "injections must be a list",
                            context={"scope": "injection", "path": str(injections_path)},
                        )
                    )
        return cls(
            config,
            data_description,
            workspace=workspace,
            cache_dir=cache_dir,
            injections=injections,
            base_issues=[*config_issues, *data_issues, *injection_issues],
            restore_snapshot=restore_snapshot,
        )

    def _restore_snapshot(self) -> None:
        payload, issues = self.snapshots.recover()
        self._recovery_issues.extend(issues)
        if payload is None:
            return
        restored_state = payload.get("state")
        restored_report = payload.get("report")
        if not isinstance(restored_state, dict) or not isinstance(restored_report, dict):
            self._recovery_issues.append(
                Issue(
                    E_PIPELINE_ERROR,
                    "snapshot payload does not contain state and report mappings",
                    severity="warning",
                    context={"scope": "snapshot"},
                )
            )
            return
        self.state = restored_state
        self._latest_report = restored_report
        self._latest_report_state_fingerprint = restored_report.get("state_fingerprint")
        self._latest_report_command_seq = int(restored_report.get("command_seq", 0))
        self._progress[self._latest_report_command_seq] = {
            "stage": "restored",
            "completed": int(restored_report.get("counts", {}).get("completed", 0)),
            "total": int(restored_report.get("counts", {}).get("candidates", 0)),
            "ratio": 1.0,
        }

    def submit_command(self, command: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        if self._closed:
            return {
                "command_seq": self.state.get("command_seq", 0),
                "accepted": False,
                "issues": issue_dicts(
                    [
                        Issue(
                            E_PIPELINE_ERROR,
                            "workbench is closed",
                            context={"scope": "workbench"},
                        )
                    ]
                ),
                "stale": True,
                "summary": self._summary(self._latest_report),
            }
        with self._lock:
            old_state = self.state
            new_state, accepted, command_issues, changed = apply_command(old_state, command)
            command_seq = int(old_state.get("command_seq", 0)) + 1
            new_state["command_seq"] = command_seq
            old_fingerprint = compute_state_fingerprint(old_state)
            new_fingerprint = compute_state_fingerprint(new_state)
            state_changed = accepted and changed and new_fingerprint != old_fingerprint
            affected = _estimate_affected(old_state, new_state) if new_state is not old_state else []
            if accepted:
                self.state = new_state
            else:
                new_state = old_state
            event = threading.Event()
            generation: int | None = None
            if state_changed:
                self._generation += 1
                generation = self._generation
                self._progress[generation] = {
                    "stage": "queued",
                    "completed": 0,
                    "total": len(affected),
                    "ratio": 0.0,
                    "command_seq": command_seq,
                }
                future = self._executor.submit(
                    self._run_job,
                    deep_copy(self.state),
                    generation,
                    event,
                )
                self._futures.append(future)
                self._current_job_event = event
            else:
                event.set()
            current_fingerprint = compute_state_fingerprint(self.state)
            latest_report = self._latest_report
            latest_fingerprint = self._latest_report_state_fingerprint
            wait_event: threading.Event | None = None
            if state_changed:
                wait_event = event
            elif self._current_job_event is not None and latest_fingerprint != current_fingerprint:
                wait_event = self._current_job_event
            timeout = float(
                self.state.get("config", {}).get("scheduler", {}).get(
                    "interaction_latency_budget_ms", 0.0
                )
            ) / 1000.0
        if wait_event is not None:
            deadline_remaining = timeout - (time.perf_counter() - started)
            # Leave a proportional tail allowance for response assembly and
            # scheduler wake-up latency.  Waiting until the absolute deadline
            # would otherwise turn a valid stale summary into a late response.
            overhead_allowance = min(timeout * 0.5, max(0.010, timeout * 0.25))
            wait_event.wait(timeout=max(0.0, deadline_remaining - overhead_allowance))
            completed_in_time = bool(wait_event.is_set())
        else:
            completed_in_time = True
        with self._lock:
            fresh = (
                completed_in_time
                and latest_fingerprint == current_fingerprint
                and latest_report is not None
            )
            if not fresh:
                fresh = (
                    completed_in_time
                    and
                    self._latest_report is not None
                    and self._latest_report_state_fingerprint == current_fingerprint
                )
            report = self._latest_report if self._latest_report is not None else latest_report
            stale = not fresh
            if stale and report is None:
                report = None
            response_issues = issue_dicts(command_issues)
            if stale:
                response_issues.append(
                    Issue(
                        E_RESULT_STALE,
                        "the interaction deadline elapsed before the current state completed",
                        severity="warning",
                        context={
                            "scope": "interaction",
                            "command_seq": self.state.get("command_seq", command_seq),
                            "result_command_seq": report.get("command_seq") if report else None,
                        },
                    ).to_dict()
                )
            if self._latest_error is not None and state_changed:
                response_issues.append(self._latest_error)
            summary = self._summary(report)
            progress = self._response_progress(
                generation or self._generation,
                report,
                affected,
            )
            result_command_seq = int(report.get("command_seq", 0)) if report else None
            partition = self._partition_for_command(command, report) if command_type(command) == "view" else None
            latency_ms = (time.perf_counter() - started) * 1000.0
            budget_ms = float(
                self.state.get("config", {}).get("scheduler", {}).get(
                    "interaction_latency_budget_ms", 0.0
                )
            )
            interval = int(
                self.state.get("config", {}).get("scheduler", {}).get(
                    "snapshot_commit_interval", 1
                )
            )
            if command_type(command) == "snapshot" or (
                interval > 0 and command_seq % interval == 0
            ):
                self._snapshot_requests[command_seq] = {"state": deep_copy(self.state)}
                self._maybe_schedule_snapshot_locked()
            return {
                "command_seq": command_seq,
                "accepted": bool(accepted),
                "issues": response_issues,
                "stale": bool(stale),
                "code": E_RESULT_STALE if stale else None,
                "result_command_seq": result_command_seq,
                "latency_ms": round(latency_ms, 6),
                "interaction_latency_budget_ms": budget_ms,
                "within_budget": latency_ms <= budget_ms + 1.0,
                "affected_candidates": affected,
                "summary": summary,
                "progress": progress,
                "partition": partition,
            }

    def _run_job(self, state: dict[str, Any], generation: int, event: threading.Event) -> None:
        command_seq = int(state.get("command_seq", 0))

        def progress_callback(payload: dict[str, Any]) -> None:
            with self._lock:
                if generation < self._generation - 1:
                    return
                progress = dict(payload)
                progress["command_seq"] = command_seq
                progress["updated_monotonic"] = time.monotonic()
                self._progress[generation] = progress

        try:
            report = execute_pipeline(
                state,
                self.cache,
                base_issues=self.base_issues + self._recovery_issues,
                progress=progress_callback,
            )
            with self._lock:
                if generation >= self._latest_report_generation or self._latest_report is None:
                    self._latest_report = report
                    self._latest_report_state_fingerprint = report.get("state_fingerprint")
                    self._latest_report_command_seq = command_seq
                    self._latest_report_generation = generation
                self._latest_error = None
            with self._lock:
                self._maybe_schedule_snapshot_locked()
        except Exception as exc:
            error = Issue(
                E_PIPELINE_ERROR,
                f"pipeline execution failed: {exc}",
                severity="error",
                context={"scope": "pipeline", "command_seq": command_seq},
            ).to_dict()
            with self._lock:
                self._latest_error = error
                self._progress[generation] = {
                    "stage": "failed",
                    "completed": 0,
                    "total": 0,
                    "ratio": 0.0,
                    "command_seq": command_seq,
                    "error": error,
                }
        finally:
            with self._lock:
                if self._current_job_event is event:
                    self._current_job_event = None
            event.set()

    def drain(self, timeout: float | None = None) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._lock:
            futures = list(self._futures)
        for future in futures:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            try:
                future.result(timeout=remaining)
            except Exception:
                pass

    def _maybe_schedule_snapshot_locked(self) -> None:
        report = self._latest_report
        if report is None:
            return
        for sequence in sorted(self._snapshot_requests):
            if sequence in self._snapshot_inflight or sequence <= self._snapshot_completed_seq:
                continue
            request = self._snapshot_requests[sequence]
            state = request["state"]
            if report.get("state_fingerprint") != compute_state_fingerprint(state):
                continue
            snapshot_report = deep_copy(report)
            snapshot_report["command_seq"] = sequence
            self._snapshot_inflight.add(sequence)
            future = self._snapshot_executor.submit(
                self._commit_snapshot_job,
                sequence,
                deep_copy(state),
                snapshot_report,
            )
            self._snapshot_futures.append(future)

    def _commit_snapshot_job(self, sequence: int, state: dict[str, Any], report: dict[str, Any]) -> None:
        try:
            with self._snapshot_commit_lock:
                pointer = self.snapshots.commit(state, report)
            with self._lock:
                self._latest_snapshot = pointer
        finally:
            with self._lock:
                self._snapshot_inflight.discard(sequence)
                self._snapshot_completed_seq = max(self._snapshot_completed_seq, sequence)
                self._maybe_schedule_snapshot_locked()

    def close(self) -> None:
        if self._closed:
            return
        self.drain()
        with self._lock:
            current_fingerprint = compute_state_fingerprint(self.state)
            if (
                self._latest_report is None
                or self._latest_report_state_fingerprint != current_fingerprint
            ):
                state = deep_copy(self.state)
                report = execute_pipeline(
                    state,
                    self.cache,
                    base_issues=self.base_issues + self._recovery_issues,
                )
                self._latest_report = report
                self._latest_report_state_fingerprint = report.get("state_fingerprint")
                self._latest_report_command_seq = int(state.get("command_seq", 0))
            elif int(self._latest_report.get("command_seq", 0)) != int(
                self.state.get("command_seq", 0)
            ):
                self._latest_report = deep_copy(self._latest_report)
                self._latest_report["command_seq"] = int(self.state.get("command_seq", 0))
            self._closed = True
            self._snapshot_requests[int(self.state.get("command_seq", 0))] = {
                "state": deep_copy(self.state)
            }
            self._maybe_schedule_snapshot_locked()
        with self._lock:
            snapshot_futures = list(self._snapshot_futures)
        for future in snapshot_futures:
            try:
                future.result()
            except Exception:
                pass
        current_sequence = int(self.state.get("command_seq", 0))
        with self._lock:
            snapshot_complete = self._snapshot_completed_seq >= current_sequence
        if not snapshot_complete and self._latest_report is not None:
            snapshot_report = deep_copy(self._latest_report)
            snapshot_report["command_seq"] = current_sequence
            with self._snapshot_commit_lock:
                pointer = self.snapshots.commit(deep_copy(self.state), snapshot_report)
            with self._lock:
                self._latest_snapshot = pointer
                self._snapshot_completed_seq = max(self._snapshot_completed_seq, current_sequence)
        self._executor.shutdown(wait=True)
        self._snapshot_executor.shutdown(wait=True)

    def snapshot_now(self) -> dict[str, Any]:
        self.drain()
        with self._lock:
            report = self._latest_report
            state = deep_copy(self.state)
        if report is None:
            report = execute_pipeline(state, self.cache, base_issues=self.base_issues + self._recovery_issues)
            with self._lock:
                self._latest_report = report
                self._latest_report_state_fingerprint = report.get("state_fingerprint")
        if report.get("state_fingerprint") != compute_state_fingerprint(state):
            report = execute_pipeline(state, self.cache, base_issues=self.base_issues + self._recovery_issues)
            with self._lock:
                self._latest_report = report
                self._latest_report_state_fingerprint = report.get("state_fingerprint")
        snapshot_report = deep_copy(report)
        snapshot_report["command_seq"] = int(state.get("command_seq", 0))
        with self._snapshot_commit_lock:
            pointer = self.snapshots.commit(state, snapshot_report)
        with self._lock:
            self._latest_snapshot = pointer
            self._snapshot_completed_seq = max(
                self._snapshot_completed_seq, int(state.get("command_seq", 0))
            )
        return pointer

    @property
    def latest_report(self) -> dict[str, Any] | None:
        with self._lock:
            return self._latest_report

    @property
    def latest_snapshot(self) -> dict[str, Any] | None:
        with self._lock:
            return self._latest_snapshot

    def full_rebuild_report(self, cache_dir: str | Path) -> dict[str, Any]:
        self.drain()
        with self._lock:
            state = deep_copy(self.state)
        return execute_pipeline(
            state,
            CacheStore(cache_dir),
            base_issues=self.base_issues + self._recovery_issues,
        )

    def _summary(self, report: dict[str, Any] | None) -> dict[str, Any]:
        if report is None:
            return {
                "command_seq": None,
                "complete": False,
                "completed_candidates": 0,
                "total_candidates": 0,
                "best_candidate": None,
                "counts": {"completed": 0, "failed": 0, "pending": 0},
                "failure_count": 0,
                "warning_count": 0,
                "diagnostics_key": None,
            }
        counts = report.get("counts", {})
        return {
            "command_seq": report.get("command_seq"),
            "state_fingerprint": report.get("state_fingerprint"),
            "complete": bool(report.get("complete", False)),
            "completed_candidates": int(counts.get("completed", 0)),
            "total_candidates": int(counts.get("candidates", 0)),
            "failed_candidates": int(counts.get("failed", 0)),
            "pending_candidates": int(counts.get("pending", 0)),
            "best_candidate": report.get("best_candidate"),
            "counts": counts,
            "failure_count": len(report.get("failures", [])),
            "warning_count": len(report.get("warnings", [])),
            "diagnostics_key": report.get("diagnostics", {}).get("cache_key"),
        }

    def _response_progress(
        self,
        generation: int,
        report: dict[str, Any] | None,
        affected: list[str],
    ) -> dict[str, Any]:
        live = self._progress.get(generation, {})
        counts = report.get("counts", {}) if report else {}
        return {
            "stage": live.get("stage", "unknown"),
            "candidate": live.get("candidate"),
            "completed": int(live.get("completed", 0)),
            "total": int(live.get("total", 0)),
            "ratio": float(live.get("ratio", 0.0)),
            "command_seq": live.get("command_seq", self.state.get("command_seq", 0)),
            "completed_candidates": int(counts.get("completed", 0)),
            "total_candidates": int(counts.get("candidates", 0)),
            "affected_candidates": affected,
            "has_current_result": bool(
                report is not None
                and report.get("state_fingerprint") == compute_state_fingerprint(self.state)
            ),
        }

    def _partition_for_command(
        self, command: dict[str, Any], report: dict[str, Any] | None
    ) -> dict[str, Any]:
        part = str(command.get("part", command.get("partition", "candidates")))
        offset = int(command.get("offset", 0) or 0)
        limit = int(command.get("limit", 20) or 20)
        offset = max(0, offset)
        limit = max(1, min(limit, 500))
        if part in {"candidates", "results", "ranking"}:
            candidates = report.get("candidates", []) if report else []
            selected = [
                {
                    "id": item["id"],
                    "method": item["method"],
                    "params": item["params"],
                    "status": item["status"],
                    "rank": item.get("rank"),
                    "metrics": item.get("metrics"),
                    "metrics_comparable": item.get("metrics_comparable"),
                }
                for item in candidates
            ]
            return {
                "part": "candidates",
                "offset": offset,
                "limit": limit,
                "total": len(selected),
                "items": selected[offset : offset + limit],
                "result_command_seq": report.get("command_seq") if report else None,
            }
        if part in {"labels", "candidate_labels", "assignments"}:
            identifier = command.get("candidate", command.get("id"))
            candidates = report.get("candidates", []) if report else []
            candidate = next((item for item in candidates if item.get("id") == identifier), None)
            if candidate is None:
                return {
                    "part": "labels",
                    "candidate": identifier,
                    "offset": offset,
                    "limit": limit,
                    "total": 0,
                    "items": [],
                    "error": "candidate_not_found",
                    "result_command_seq": report.get("command_seq") if report else None,
                }
            labels = candidate.get("labels") or []
            items = [
                {"row": index, "label": int(labels[index])}
                for index in range(offset, min(len(labels), offset + limit))
            ]
            return {
                "part": "labels",
                "candidate": identifier,
                "offset": offset,
                "limit": limit,
                "total": len(labels),
                "items": items,
                "result_command_seq": report.get("command_seq") if report else None,
            }
        if part in {"data", "rows"}:
            with self._lock:
                state = deep_copy(self.state)
            table = generate_table(
                state["data_description"],
                overrides=state.get("data_overrides", {}),
                injections=state.get("injections", []),
            )
            columns = command.get("columns")
            if not isinstance(columns, list) or not columns:
                columns = sorted(table.values)
            else:
                columns = [str(column) for column in columns]
            items = [
                table.as_object_row(index, columns)
                for index in range(offset, min(table.rows, offset + limit))
            ]
            return {
                "part": "data",
                "offset": offset,
                "limit": limit,
                "total": table.rows,
                "columns": columns,
                "items": items,
                "command_seq": state.get("command_seq"),
            }
        if part in {"failures", "issues"}:
            issues = report.get("failures", []) if report else []
            return {
                "part": "failures",
                "offset": offset,
                "limit": limit,
                "total": len(issues),
                "items": issues[offset : offset + limit],
                "result_command_seq": report.get("command_seq") if report else None,
            }
        return {
            "part": part,
            "offset": offset,
            "limit": limit,
            "total": 0,
            "items": [],
            "error": "unknown_partition",
        }


def _estimate_affected(old_state: dict[str, Any], new_state: dict[str, Any]) -> list[str]:
    old_candidates = {item.get("id"): item for item in old_state.get("candidates", []) if isinstance(item, dict)}
    new_candidates = {item.get("id"): item for item in new_state.get("candidates", []) if isinstance(item, dict)}
    affected: set[str] = set()
    for identifier in new_candidates.keys() - old_candidates.keys():
        affected.add(str(identifier))
    for identifier in old_candidates.keys() - new_candidates.keys():
        affected.add(str(identifier))
    for identifier in old_candidates.keys() & new_candidates.keys():
        if canonical_json(old_candidates[identifier]) != canonical_json(new_candidates[identifier]):
            affected.add(str(identifier))

    old_config = old_state.get("config", {})
    new_config = new_state.get("config", {})
    if canonical_json(old_config.get("preprocessing", {})) != canonical_json(new_config.get("preprocessing", {})):
        affected.update(str(key) for key in new_candidates)
    if canonical_json(old_config.get("evaluation", {})) != canonical_json(new_config.get("evaluation", {})):
        affected.update(str(key) for key in new_candidates)
    old_scheduler = old_config.get("scheduler", {})
    new_scheduler = new_config.get("scheduler", {})
    for field in ("global_iteration_budget", "per_candidate_iteration_quota"):
        if old_scheduler.get(field) != new_scheduler.get(field):
            affected.update(str(key) for key in new_candidates)
    old_methods = old_config.get("methods", {})
    new_methods = new_config.get("methods", {})
    for method in sorted(set(old_methods) | set(new_methods)):
        if canonical_json(old_methods.get(method, {})) != canonical_json(new_methods.get(method, {})):
            for identifier, candidate in new_candidates.items():
                if candidate.get("method") == method:
                    affected.add(str(identifier))

    if canonical_json(old_state.get("data_description", {})) != canonical_json(new_state.get("data_description", {})):
        affected.update(str(key) for key in new_candidates)
    old_overrides = old_state.get("data_overrides", {})
    new_overrides = new_state.get("data_overrides", {})
    changed_columns = {
        column
        for column in set(old_overrides) | set(new_overrides)
        if canonical_json(old_overrides.get(column)) != canonical_json(new_overrides.get(column))
    }
    if changed_columns:
        all_columns = sorted(
            str(column.get("name"))
            for column in new_state.get("data_description", {}).get("columns", [])
            if isinstance(column, dict) and column.get("name")
        )
        affected_columns = set(changed_columns)
        column_specs = [
            column
            for column in new_state.get("data_description", {}).get("columns", [])
            if isinstance(column, dict) and column.get("name") and column.get("type") == "numeric"
        ]
        changed_dependency = True
        while changed_dependency:
            changed_dependency = False
            for column in column_specs:
                name = str(column["name"])
                source = column.get("source")
                if source in affected_columns and name not in affected_columns:
                    affected_columns.add(name)
                    changed_dependency = True
        for identifier, candidate in new_candidates.items():
            selected = candidate.get("feature_columns") or all_columns
            if isinstance(selected, list) and set(selected) & affected_columns:
                affected.add(str(identifier))
    if canonical_json(old_state.get("injections", [])) != canonical_json(new_state.get("injections", [])):
        affected.update(str(key) for key in new_candidates)
    return sorted(affected)


def _merge_injections(
    first: list[dict[str, Any]], second: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for injection in [*first, *second]:
        if not isinstance(injection, dict):
            result.append(copy.deepcopy(injection))
            continue
        identifier = injection.get("id")
        if isinstance(identifier, str):
            if identifier in seen:
                continue
            seen.add(identifier)
        result.append(copy.deepcopy(injection))
    return result
