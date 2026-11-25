"""End-to-end deterministic fitting engine."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .cache import CacheStore
from .errors import E_INTERNAL, FitError, Issue
from .fingerprints import (
    CACHE_STAGES,
    all_stage_keys,
    config_computation_fingerprint,
    config_full_fingerprint,
    library_fingerprint,
    series_library_fingerprint,
)
from .jsonutil import fingerprint
from .model import CacheAction, FitConfig, SeriesRecord
from .scheduler import BudgetPlan, PlanEntry, build_budget_plan
from .series_fit import CoreSeriesResult, fit_series, load_cached_core
from .snapshot import SNAPSHOT_STAGES, SnapshotManager, SnapshotView


def _issue_from_dict(data: Mapping[str, Any]) -> Issue:
    return Issue(
        code=str(data["code"]),
        message=str(data["message"]),
        source=str(data.get("source", "")),
        line=int(data["line"]) if data.get("line") is not None else None,
        series_id=(
            str(data["series_id"]) if data.get("series_id") is not None else None
        ),
        stage=str(data.get("stage", "")),
    )


def _issue_sort_key(issue: Issue) -> tuple[Any, ...]:
    return (
        issue.source,
        issue.line if issue.line is not None else 0,
        issue.series_id or "",
        issue.code,
        issue.message,
    )


def _fit_failure_sort_key(issue: Issue) -> tuple[Any, ...]:
    return (issue.series_id or "", issue.stage, issue.code, issue.message)


def _plan_to_manifest(plan: BudgetPlan) -> dict[str, Any]:
    return {
        "entries": [entry.as_manifest_dict() for entry in plan.entries],
        "batches": [list(batch) for batch in plan.batches],
        "consumed_points": plan.consumed_points,
        "remaining_points": plan.remaining_points,
        "pending_queue": [issue.as_dict() for issue in plan.pending_queue],
        "failures": [issue.as_dict() for issue in plan.failures],
        "cache_hits": list(plan.cache_hits),
    }


def _plan_from_manifest(data: Mapping[str, Any]) -> BudgetPlan:
    entries: list[PlanEntry] = []
    for entry in data.get("entries", []):
        issue_data = entry.get("issue")
        entries.append(
            PlanEntry(
                series_id=str(entry["series_id"]),
                observation_points=int(entry["observation_points"]),
                kind=str(entry["kind"]),
                required_points=int(entry["required_points"]),
                deducted_points=int(entry["deducted_points"]),
                remaining_before=int(entry["remaining_before"]),
                remaining_after=int(entry["remaining_after"]),
                issue=_issue_from_dict(issue_data) if issue_data else None,
                batch_index=(
                    int(entry["batch_index"])
                    if entry.get("batch_index") is not None
                    else None
                ),
            )
        )
    return BudgetPlan(
        entries=tuple(entries),
        batches=tuple(
            tuple(str(series_id) for series_id in batch)
            for batch in data.get("batches", [])
        ),
        consumed_points=int(data.get("consumed_points", 0)),
        remaining_points=int(data.get("remaining_points", 0)),
        pending_queue=tuple(
            _issue_from_dict(issue) for issue in data.get("pending_queue", [])
        ),
        failures=tuple(_issue_from_dict(issue) for issue in data.get("failures", [])),
        cache_hits=tuple(str(value) for value in data.get("cache_hits", [])),
    )


def _actions_payload(actions: Iterable[CacheAction]) -> list[dict[str, Any]]:
    return [action.as_dict() for action in actions]


def _actions_from_payload(data: Iterable[Mapping[str, Any]]) -> tuple[CacheAction, ...]:
    return tuple(
        CacheAction(
            stage=str(item["stage"]),
            key=str(item["key"]),
            action=str(item["action"]),
            reason=str(item["reason"]),
            resumed=bool(item.get("resumed", False)),
        )
        for item in data
    )


@dataclass
class ExecutionState:
    results: dict[str, CoreSeriesResult]
    failures: dict[str, Issue]
    executed_this_run: int = 0


class FitEngine:
    """Deterministic series fitting engine bound to one workspace."""

    def __init__(
        self,
        workspace: str | Path,
        config: FitConfig,
        series: Iterable[SeriesRecord],
        *,
        validation_issues: Iterable[Issue] = (),
        mode: str = "concurrent",
    ) -> None:
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.series = tuple(sorted(series, key=lambda item: item.series_id))
        self.series_map = {record.series_id: record for record in self.series}
        if len(self.series_map) != len(self.series):
            raise ValueError("FitEngine received duplicate series ids")
        self.validation_issues = tuple(validation_issues)
        self.mode = mode
        if mode not in ("concurrent", "serial"):
            raise ValueError("mode must be 'concurrent' or 'serial'")
        self.cache = CacheStore(self.workspace / "cache")
        self.snapshots = SnapshotManager(self.workspace)

    def _artifacts(
        self,
        results: Mapping[str, CoreSeriesResult],
        validation_issues: Iterable[Issue],
    ) -> dict[str, dict[str, Any]]:
        artifacts: dict[str, dict[str, Any]] = {
            stage: {} for stage in SNAPSHOT_STAGES
        }
        artifacts["loading"] = {
            "series_ids": [record.series_id for record in self.series],
            "series_content_fingerprints": series_library_fingerprint(self.series),
            "validation_failures": [
                issue.as_dict() for issue in sorted(validation_issues, key=_issue_sort_key)
            ],
        }
        for series_id in sorted(results):
            core = results[series_id].artifact
            artifacts["interpolation"][series_id] = core["interpolation"]
            artifacts["basis"][series_id] = core["basis"]
            artifacts["normal_equations"][series_id] = core["normal_equations"]
            artifacts["elimination"][series_id] = core["elimination"]
            artifacts["back_substitution"][series_id] = {
                "artifact": core,
                "actions": _actions_payload(results[series_id].actions),
            }
        return artifacts

    def _load_resumed_results(
        self, snapshot: SnapshotView
    ) -> tuple[dict[str, CoreSeriesResult], dict[str, Issue]]:
        results: dict[str, CoreSeriesResult] = {}
        failures: dict[str, Issue] = {}
        back_stage = snapshot.artifacts.get("back_substitution", {})
        for series_id, item in back_stage.items():
            if not isinstance(item, dict) or not isinstance(item.get("artifact"), dict):
                continue
            results[series_id] = CoreSeriesResult(
                artifact=item["artifact"],
                actions=_actions_from_payload(item.get("actions", [])),
            )
        manifest = snapshot.manifest or {}
        for issue_data in manifest.get("execution_failures", []):
            issue = _issue_from_dict(issue_data)
            if issue.series_id:
                failures[issue.series_id] = issue
        return results, failures

    def _execute_batches(
        self,
        plan: BudgetPlan,
        state: ExecutionState,
        *,
        stop_after: int | None,
    ) -> bool:
        """Run planned batches.  Return True when interrupted."""

        workers = 1 if self.mode == "serial" else max(1, self.config.thread_pool_workers)
        interrupted = False
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="fixedseason") as pool:
            for batch in plan.batches:
                remaining_before_submit = any(
                    series_id not in state.results and series_id not in state.failures
                    for series_id in plan.fit_series_ids
                )
                if not remaining_before_submit:
                    break
                if stop_after is not None:
                    allowance = stop_after - state.executed_this_run
                    if allowance <= 0:
                        interrupted = True
                        break
                else:
                    allowance = None
                pending_ids = [
                    series_id
                    for series_id in batch
                    if series_id not in state.results and series_id not in state.failures
                ]
                if allowance is not None:
                    pending_ids = pending_ids[:allowance]
                if not pending_ids:
                    continue
                futures = {
                    pool.submit(
                        fit_series,
                        self.series_map[series_id],
                        self.config,
                        self.cache,
                        planned_action="miss",
                    ): series_id
                    for series_id in pending_ids
                }
                for future in as_completed(futures):
                    series_id = futures[future]
                    try:
                        state.results[series_id] = future.result()
                    except FitError as exc:
                        state.failures[series_id] = Issue(
                            code=exc.code,
                            message=exc.message,
                            series_id=series_id,
                            stage="fit",
                        )
                    except Exception as exc:  # pragma: no cover - defensive boundary
                        state.failures[series_id] = Issue(
                            code=E_INTERNAL,
                            message=f"{type(exc).__name__}: {exc}",
                            series_id=series_id,
                            stage="fit",
                        )
                    state.executed_this_run += 1
                if stop_after is not None and state.executed_this_run >= stop_after:
                    remaining = any(
                        series_id not in state.results and series_id not in state.failures
                        for series_id in plan.fit_series_ids
                    )
                    if remaining:
                        interrupted = True
                        break
        return interrupted

    def _report(
        self,
        *,
        version: int,
        plan: BudgetPlan,
        state: ExecutionState,
        snapshot_status: str,
        interrupted: bool,
        resumed: bool,
        archive_path: str | None,
    ) -> dict[str, Any]:
        config_fp = config_full_fingerprint(self.config)
        compute_fp = config_computation_fingerprint(self.config)
        library_fp = library_fingerprint(self.series)
        series_fps = series_library_fingerprint(self.series)
        execution_outcomes: dict[str, str] = {}
        for entry in plan.entries:
            if entry.series_id in state.results:
                if entry.kind == "cache_hit":
                    execution_outcomes[entry.series_id] = "cache_hit"
                else:
                    execution_outcomes[entry.series_id] = "fit_completed"
            elif entry.series_id in state.failures:
                execution_outcomes[entry.series_id] = "fit_failed"
            elif entry.kind == "budget_exceeded":
                execution_outcomes[entry.series_id] = "budget_exceeded"
            elif entry.kind == "pending":
                execution_outcomes[entry.series_id] = "pending"
            else:
                execution_outcomes[entry.series_id] = "unfinished"

        cache_ledger: list[dict[str, Any]] = []
        reuse_ledger: list[dict[str, Any]] = []
        for record in self.series:
            series_id = record.series_id
            series_fp = series_fps[series_id]
            keys = all_stage_keys(
                series_fingerprint=series_fp,
                config=self.config,
                config_fingerprint=compute_fp,
            )
            result = state.results.get(series_id)
            actions = (
                {action.stage: action for action in result.actions}
                if result is not None
                else {}
            )
            for stage in CACHE_STAGES:
                action = actions.get(stage)
                cache_ledger.append(
                    {
                        "series_id": series_id,
                        "stage": stage,
                        "key": keys[stage],
                        "action": action.action if action else "not_attempted",
                        "reason": action.reason if action else "no_fit_result",
                    }
                )
            reuse_ledger.append(
                {
                    "series_id": series_id,
                    "content_fingerprint": series_fp,
                    "stage_keys": keys,
                    "referenced_observations": {
                        "observed_indices": list(record.observed_indices),
                        "missing_indices": list(record.missing_indices),
                        "window": "entire_series",
                    },
                }
            )

        validation_failures = sorted(self.validation_issues, key=_issue_sort_key)
        fit_failures = list(plan.failures)
        fit_failures.extend(state.failures.values())
        fit_failures = sorted(fit_failures, key=_fit_failure_sort_key)
        all_failures = validation_failures + fit_failures
        unfinished = [
            series_id
            for series_id in plan.fit_series_ids
            if series_id not in state.results and series_id not in state.failures
        ]
        complete = not interrupted
        series_reports = [state.results[series_id].artifact for series_id in sorted(state.results)]
        report = {
            "schema_version": 1,
            "run": {
                "global_version": version,
                "config_hash": config_fp,
                "compute_config_hash": compute_fp,
                "series_library_fingerprint": library_fp,
                "series_content_fingerprints": series_fps,
                "bit_width_tier": self.config.bit_tier,
                "mode": self.mode,
                "workers": 1 if self.mode == "serial" else self.config.thread_pool_workers,
                "snapshot_status": snapshot_status,
                "resumed": resumed,
                "interrupted": interrupted,
                "complete": complete,
                "archive_path": archive_path,
                "run_key": fingerprint(
                    {
                        "version": version,
                        "config_hash": config_fp,
                        "series_library_fingerprint": library_fp,
                        "bit_width_tier": self.config.bit_tier,
                    }
                ),
            },
            "config": self.config.full_dict(),
            "validation_failures": [issue.as_dict() for issue in validation_failures],
            "fit_failures": [issue.as_dict() for issue in fit_failures],
            "failures": [issue.as_dict() for issue in all_failures],
            "pending": [issue.as_dict() for issue in plan.pending_queue],
            "unfinished": unfinished,
            "series": series_reports,
            "cache_ledger": cache_ledger,
            "reuse_ledger": reuse_ledger,
            "budget": {
                "settings": {
                    "global_fit_points": self.config.global_fit_points,
                    "per_series_quota": self.config.per_series_quota,
                    "batch_observation_points": self.config.batch_observation_points,
                    "fit_cost_unit": "one_observation_slot",
                },
                "consumed_fit_points": plan.consumed_points,
                "remaining_fit_points": plan.remaining_points,
                "ledger": plan.ledger(execution_outcomes),
                "batches": [list(batch) for batch in plan.batches],
                "pending_queue": [issue.as_dict() for issue in plan.pending_queue],
                "cache_hit_series": list(plan.cache_hits),
            },
            "summary": {
                "input_series_count": len(self.series),
                "completed_series_count": len(series_reports),
                "cache_hit_series_count": sum(
                    1 for entry in plan.entries if entry.kind == "cache_hit"
                ),
                "budget_exceeded_count": sum(
                    1 for issue in plan.failures if issue.code == "E_FIT_BUDGET_EXCEEDED"
                ),
                "pending_count": len(plan.pending_queue),
                "fit_failure_count": len(state.failures),
                "validation_failure_count": len(validation_failures),
                "unfinished_count": len(unfinished),
            },
            "stage_contract": {
                "stages": list(SNAPSHOT_STAGES),
                "cache_stages": list(CACHE_STAGES),
                "truth_path": "float64_observed_only_no_imputation",
            },
        }
        return report

    def run(
        self,
        *,
        version: int | None = None,
        stop_after: int | None = None,
        force_rerun: bool = False,
    ) -> dict[str, Any]:
        version = self.config.version if version is None else int(version)
        config_fp = config_full_fingerprint(self.config)
        compute_fp = config_computation_fingerprint(self.config)
        library_fp = library_fingerprint(self.series)
        series_fps = series_library_fingerprint(self.series)
        snapshot = self.snapshots.begin(
            global_version=version,
            config_hash=config_fp,
            compute_config_hash=compute_fp,
            series_data_fingerprint=series_fps,
            series_library_fingerprint=library_fp,
            bit_width_tier=self.config.bit_tier,
        )
        if snapshot.status == "already_complete" and not force_rerun:
            assert snapshot.final_report is not None
            return snapshot.final_report
        resumed = snapshot.status == "resume"
        prior_plan = None
        if resumed and snapshot.manifest and isinstance(snapshot.manifest.get("budget_plan"), dict):
            try:
                prior_plan = _plan_from_manifest(snapshot.manifest["budget_plan"])
            except (KeyError, TypeError, ValueError):
                prior_plan = None
                resumed = False

        results: dict[str, CoreSeriesResult] = {}
        failures: dict[str, Issue] = {}
        if resumed and prior_plan is not None:
            results, failures = self._load_resumed_results(snapshot)
            cached_ids = set(prior_plan.cache_hits)
            for series_id in prior_plan.cache_hits:
                if series_id in results:
                    continue
                cached = load_cached_core(
                    self.series_map[series_id],
                    self.config,
                    self.cache,
                    config_fp=compute_fp,
                    series_fp=series_fps[series_id],
                )
                if cached is not None:
                    results[series_id] = cached
            plan = prior_plan
        else:
            cached_results: dict[str, CoreSeriesResult] = {}
            for record in self.series:
                cached = load_cached_core(
                    record,
                    self.config,
                    self.cache,
                    config_fp=compute_fp,
                    series_fp=series_fps[record.series_id],
                )
                if cached is not None:
                    cached_results[record.series_id] = cached
            cached_ids = set(cached_results)
            plan = build_budget_plan(self.series, self.config, cached_ids)
            results.update(cached_results)
        state = ExecutionState(results=results, failures=failures)

        # Commit the deterministic plan and read-only cache hits before any
        # worker starts.  This makes the budget plan durable for recovery.
        initial_manifest = self._manifest(
            version=version,
            config_fp=config_fp,
            compute_fp=compute_fp,
            library_fp=library_fp,
            series_fps=series_fps,
            plan=plan,
            state=state,
            interrupted=False,
            complete=False,
        )
        self.snapshots.commit(
            initial_manifest,
            self._artifacts(state.results, self.validation_issues),
            partial_report=None,
        )

        interrupted = self._execute_batches(plan, state, stop_after=stop_after)
        report = self._report(
            version=version,
            plan=plan,
            state=state,
            snapshot_status=snapshot.status,
            interrupted=interrupted,
            resumed=resumed,
            archive_path=snapshot.archive_path,
        )
        final_manifest = self._manifest(
            version=version,
            config_fp=config_fp,
            compute_fp=compute_fp,
            library_fp=library_fp,
            series_fps=series_fps,
            plan=plan,
            state=state,
            interrupted=interrupted,
            complete=not interrupted,
        )
        self.snapshots.commit(
            final_manifest,
            self._artifacts(state.results, self.validation_issues),
            partial_report=None if not interrupted else report,
            final_report=report if not interrupted else None,
        )
        return report

    def _manifest(
        self,
        *,
        version: int,
        config_fp: str,
        compute_fp: str,
        library_fp: str,
        series_fps: Mapping[str, str],
        plan: BudgetPlan,
        state: ExecutionState,
        interrupted: bool,
        complete: bool,
    ) -> dict[str, Any]:
        completed = sorted(state.results)
        return {
            "snapshot_schema_version": 1,
            "global_version": version,
            "config_hash": config_fp,
            "compute_config_hash": compute_fp,
            "series_data_fingerprint": dict(series_fps),
            "series_library_fingerprint": library_fp,
            "bit_width_tier": self.config.bit_tier,
            "completed_series_cursor": len(completed),
            "completed_series": completed,
            "completed_series_count": len(completed),
            "interrupted": interrupted,
            "complete": complete,
            "stage_names": list(SNAPSHOT_STAGES),
            "bit_width": self.config.fixed_point_bits,
            "quantization_shift": self.config.quantization_shift,
            "pending_queue": [issue.as_dict() for issue in plan.pending_queue],
            "budget_plan": _plan_to_manifest(plan),
            "execution_failures": [
                issue.as_dict() for issue in sorted(state.failures.values(), key=_fit_failure_sort_key)
            ],
            "remaining_fit_series": [
                series_id
                for series_id in plan.fit_series_ids
                if series_id not in state.results and series_id not in state.failures
            ],
            "mode": self.mode,
            "workers": 1 if self.mode == "serial" else self.config.thread_pool_workers,
            "pid": os.getpid(),
        }


def run_once(
    workspace: str | Path,
    config: FitConfig,
    series: Iterable[SeriesRecord],
    *,
    validation_issues: Iterable[Issue] = (),
    mode: str = "concurrent",
    version: int | None = None,
    stop_after: int | None = None,
    force_rerun: bool = False,
) -> dict[str, Any]:
    engine = FitEngine(
        workspace,
        config,
        series,
        validation_issues=validation_issues,
        mode=mode,
    )
    return engine.run(version=version, stop_after=stop_after, force_rerun=force_rerun)
