from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import numpy as np

from .algorithms import ClusterResult, dbscan, gaussian_mixture, kmeans
from .cache import CacheStore
from .commands import compute_state_fingerprint
from .config import runtime_settings, validate_candidate
from .data import Table, generate_table
from .errors import (
    E_CAND_BUDGET_EXHAUSTED,
    E_CACHE_CORRUPT,
    E_CONFIG_INVALID,
    E_GLOBAL_BUDGET_PENDING,
    E_INJECTION_INVALID,
    E_MAX_ITER_REACHED,
    E_METRIC_UNAVAILABLE,
    E_PIPELINE_ERROR,
    Issue,
    issue_dicts,
)
from .metrics import evaluate_labels, rank_candidates
from .preprocess import PreprocessPlan, PreprocessResult, fit_transform, make_preprocess_plan
from .scheduler import finalize_budget, plan_budget, requested_iterations
from .util import deep_copy, hash_array, stable_hash, to_jsonable


ProgressCallback = Callable[[dict[str, Any]], None]


def execute_pipeline(
    state: dict[str, Any],
    cache: CacheStore,
    *,
    base_issues: list[Issue] | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Execute a complete state synchronously using content-addressed caches.

    Wall-clock timings and cache-hit provenance are deliberately not part of the
    canonical fields.  Consequently a warm incremental execution and a blank
    full rebuild produce identical candidate labels, metrics, ranking, ledger,
    and failure ordering.
    """

    config = state["config"]
    data_description = state["data_description"]
    preprocessing_config = config["preprocessing"]
    scheduler_config = config["scheduler"]
    global_issues: list[Issue] = list(base_issues or [])
    for state_issue in state.get("state_issues", []):
        if isinstance(state_issue, dict):
            global_issues.append(_issue_from_mapping(state_issue))
    state_fingerprint = compute_state_fingerprint(state)
    table = generate_table(
        data_description,
        overrides=state.get("data_overrides", {}),
        injections=state.get("injections", []),
    )
    global_issues.extend(table.issues)
    injection_report = list(table.injection_report)

    raw_candidates = deep_copy(state.get("candidates", []))
    raw_candidates, candidate_injection_issues, candidate_injection_report = _apply_candidate_injections(
        raw_candidates, state.get("injections", [])
    )
    global_issues.extend(candidate_injection_issues)
    injection_report.extend(candidate_injection_report)

    normalized_candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw_candidate in raw_candidates:
        if isinstance(raw_candidate, dict):
            raw_identifier = raw_candidate.get("id", raw_candidate.get("candidate"))
        else:
            raw_identifier = None
        if isinstance(raw_identifier, str) and raw_identifier in seen_ids:
            global_issues.append(
                Issue(
                    E_CONFIG_INVALID,
                    f"duplicate candidate id {raw_identifier!r} was isolated",
                    context={"scope": "candidate", "candidate": raw_identifier},
                )
            )
            continue
        normalized, validation_issues = validate_candidate(
            raw_candidate,
            row_count=table.rows,
            available_columns=table.available_columns,
            declared_columns=table.declared_columns,
        )
        if normalized is None:
            continue
        identifier = normalized["id"]
        seen_ids.add(identifier)
        normalized["valid"] = not any(issue.severity == "error" for issue in validation_issues)
        normalized["validation_issues"] = validation_issues
        method = normalized.get("method")
        normalized["runtime_settings"] = (
            runtime_settings(config, method, normalized.get("params", {}))
            if isinstance(method, str)
            else {}
        )
        normalized["stable_key"] = identifier
        normalized_candidates.append(normalized)

    normalized_candidates.sort(key=lambda item: item["stable_key"])
    total_candidates = len(normalized_candidates)
    _notify(progress, stage="validated", completed=0, total=total_candidates)

    preprocess_results, preprocess_plans, preprocess_actions, preprocess_issues = _execute_preprocessing(
        table,
        normalized_candidates,
        preprocessing_config,
        cache,
        concurrency=scheduler_config["concurrency"],
        progress=progress,
    )
    global_issues.extend(preprocess_issues)

    budget_candidates: list[dict[str, Any]] = []
    for candidate in normalized_candidates:
        preprocess = preprocess_results.get(candidate["stable_key"])
        preprocess_failed = bool(preprocess is not None and preprocess.metadata.get("failed", False))
        valid_for_budget = bool(candidate["valid"] and not preprocess_failed)
        budget_candidates.append(
            {
                "stable_key": candidate["stable_key"],
                "method": candidate.get("method", ""),
                "params": candidate.get("params", {}),
                "runtime_settings": candidate.get("runtime_settings", {}),
                "valid": valid_for_budget,
            }
        )
    budget_plan = plan_budget(
        budget_candidates,
        global_budget=int(scheduler_config["global_iteration_budget"]),
        per_candidate_quota=int(scheduler_config["per_candidate_iteration_quota"]),
    )
    budget_entries = budget_plan["by_candidate"]

    outcomes: dict[str, dict[str, Any]] = {}
    consumed: dict[str, int] = {}
    ledger_outcomes: dict[str, str] = {}
    reuse_ledger: list[dict[str, Any]] = []
    for candidate in normalized_candidates:
        key = candidate["stable_key"]
        reuse_ledger.append(
            {
                "stage": "preprocess",
                "candidate": key,
                "action": preprocess_actions.get(key, "not_run"),
                "cache_key": preprocess_plans[key].cache_key if key in preprocess_plans else None,
            }
        )

    cluster_tasks: list[dict[str, Any]] = []
    for candidate in normalized_candidates:
        key = candidate["stable_key"]
        entry = budget_entries[key]
        preprocess = preprocess_results.get(key)
        if not candidate["valid"]:
            outcome = {
                "status": "invalid",
                "errors": list(candidate["validation_issues"]),
                "warnings": [],
                "consumed": 0,
                "cluster_action": "not_run",
                "evaluation_action": "not_run",
            }
            outcomes[key] = outcome
            ledger_outcomes[key] = "invalid"
            consumed[key] = 0
            continue
        if preprocess is None or preprocess.metadata.get("failed", False):
            errors = list(preprocess.issues if preprocess is not None else [])
            if not any(issue.severity == "error" for issue in errors):
                errors.append(
                    Issue(
                        E_PIPELINE_ERROR,
                        "preprocessing produced no usable artifact",
                        context={"scope": "preprocess", "candidate": key},
                    )
                )
            outcome = {
                "status": "failed",
                "errors": errors,
                "warnings": [],
                "consumed": 0,
                "cluster_action": "not_run",
                "evaluation_action": "not_run",
            }
            outcomes[key] = outcome
            ledger_outcomes[key] = "preprocessing_error"
            consumed[key] = 0
            continue
        if entry["outcome"] == "global_budget_pending":
            warning_or_error = Issue(
                E_GLOBAL_BUDGET_PENDING,
                "candidate waits in the global-budget pending queue",
                context={
                    "scope": "scheduler",
                    "candidate": key,
                    "method": candidate.get("method"),
                    "params": candidate.get("params", {}),
                },
            )
            outcome = {
                "status": "pending",
                "errors": [warning_or_error],
                "warnings": [],
                "consumed": 0,
                "cluster_action": "not_run",
                "evaluation_action": "not_run",
            }
            outcomes[key] = outcome
            ledger_outcomes[key] = "global_budget_pending"
            consumed[key] = 0
            continue

        allocation = int(entry["reserved"])
        request = int(entry["requested"])
        cluster_key = _cluster_key(
            candidate=candidate,
            preprocess=preprocess,
            allocation=allocation,
            requested=request,
        )
        cluster_record, cluster_cache_issues = cache.get("cluster", cluster_key)
        global_issues.extend(cluster_cache_issues)
        cluster_cached = cluster_record is not None and _valid_cluster_record(cluster_record)
        if cluster_record is not None and not cluster_cached:
            global_issues.append(
                Issue(
                    E_CACHE_CORRUPT,
                    "cluster cache payload was structurally invalid; recomputing",
                    severity="warning",
                    context={"scope": "cache", "candidate": key, "cache_key": cluster_key},
                )
            )
        if cluster_cached:
            cluster_payload = dict(cluster_record["payload"])
            labels = np.asarray(cluster_record.get("array_value", np.empty(0, dtype=np.int64)), dtype=np.int64)
            if not cluster_payload.get("budget_exhausted", False) and labels.shape[0] != table.rows:
                global_issues.append(
                    Issue(
                        E_CACHE_CORRUPT,
                        "cluster cache label length mismatch; recomputing",
                        severity="warning",
                        context={"scope": "cache", "candidate": key, "cache_key": cluster_key},
                    )
                )
                cluster_cached = False
        if cluster_cached:
            cluster_action = "reuse"
            if cluster_payload.get("budget_exhausted", False):
                evaluation_action = "not_run"
                evaluation_key = None
                outcomes[key] = _outcome_from_cluster_cache(
                    candidate, preprocess, cluster_payload, labels, cluster_action, evaluation_action
                )
                ledger_outcomes[key] = "budget_exhausted"
                consumed[key] = int(cluster_payload.get("iterations", allocation))
            else:
                evaluation_key = _evaluation_key(
                    candidate=candidate,
                    preprocess=preprocess,
                    cluster_fingerprint=str(cluster_payload["content_fingerprint"]),
                    config=config,
                )
                evaluation_record, evaluation_cache_issues = cache.get("evaluation", evaluation_key)
                global_issues.extend(evaluation_cache_issues)
                if evaluation_record is not None and _valid_evaluation_record(evaluation_record):
                    evaluation_action = "reuse"
                    outcomes[key] = _outcome_from_cluster_cache(
                        candidate,
                        preprocess,
                        cluster_payload,
                        labels,
                        cluster_action,
                        evaluation_action,
                        evaluation_payload=dict(evaluation_record["payload"]),
                        evaluation_key=evaluation_key,
                    )
                else:
                    evaluation_action = "recompute"
                    cluster_tasks.append(
                        {
                            "candidate": candidate,
                            "preprocess": preprocess,
                            "entry": entry,
                            "cluster_key": cluster_key,
                            "cluster_payload": cluster_payload,
                            "labels": labels,
                            "cluster_action": cluster_action,
                            "evaluation_key": evaluation_key,
                            "evaluation_action": evaluation_action,
                        }
                    )
                    outcomes[key] = None
                ledger_outcomes[key] = "completed"
                consumed[key] = int(cluster_payload.get("iterations", allocation))
        else:
            cluster_action = "recompute"
            cluster_tasks.append(
                {
                    "candidate": candidate,
                    "preprocess": preprocess,
                    "entry": entry,
                    "cluster_key": cluster_key,
                    "cluster_payload": None,
                    "labels": None,
                    "cluster_action": cluster_action,
                    "evaluation_key": None,
                    "evaluation_action": "recompute",
                }
            )
            outcomes[key] = None

    if cluster_tasks:
        task_results = _execute_cluster_tasks(
            cluster_tasks,
            config=config,
            table=table,
            cache=cache,
            concurrency=scheduler_config["concurrency"],
            progress=progress,
            total_candidates=total_candidates,
        )
        for key, task_result in task_results.items():
            outcomes[key] = task_result["outcome"]
            consumed[key] = int(task_result["consumed"])
            ledger_outcomes[key] = task_result["ledger_outcome"]
            global_issues.extend(task_result.get("cache_issues", []))

    candidate_results: list[dict[str, Any]] = []
    for candidate in normalized_candidates:
        key = candidate["stable_key"]
        outcome = outcomes.get(key)
        entry = budget_entries[key]
        if outcome is None:
            outcome = {
                "status": "failed",
                "errors": [
                    Issue(
                        E_PIPELINE_ERROR,
                        "candidate did not produce an outcome",
                        context={"scope": "pipeline", "candidate": key},
                    )
                ],
                "warnings": [],
                "consumed": 0,
                "cluster_action": "not_run",
                "evaluation_action": "not_run",
            }
            consumed[key] = 0
            ledger_outcomes[key] = "pipeline_error"
        preprocess = preprocess_results.get(key)
        candidate_errors = _enrich_candidate_issues(outcome.get("errors", []), candidate)
        candidate_warnings = _enrich_candidate_issues(outcome.get("warnings", []), candidate)
        cluster_payload = outcome.get("cluster_payload")
        evaluation_payload = outcome.get("evaluation_payload")
        labels = outcome.get("labels")
        metrics = (
            evaluation_payload.get("metrics", {})
            if evaluation_payload is not None
            else _unavailable_metrics(config["evaluation"].get("metrics", []), "candidate_not_evaluated")
        )
        noncomparable = (
            evaluation_payload.get("noncomparable_reasons", []) if evaluation_payload is not None else ["candidate_not_evaluated"]
        )
        metrics_comparable = bool(
            evaluation_payload is not None
            and not noncomparable
            and any(item.get("value") is not None for item in metrics.values())
        )
        status = _candidate_status(outcome.get("status", "failed"), candidate_errors, candidate_warnings)
        candidate_results.append(
            {
                "id": candidate["id"],
                "stable_key": key,
                "method": candidate.get("method"),
                "params": candidate.get("params", {}),
                "feature_columns": candidate.get("feature_columns", []),
                "train_fraction": candidate.get("train_fraction", 1.0),
                "runtime_settings": candidate.get("runtime_settings", {}),
                "status": status,
                "metrics": metrics,
                "metrics_comparable": metrics_comparable,
                "noncomparable_reasons": noncomparable,
                "warnings": issue_dicts(candidate_warnings),
                "errors": issue_dicts(candidate_errors),
                "labels": list(int(value) for value in labels) if labels is not None else None,
                "cluster": cluster_payload,
                "preprocess": preprocess.metadata if preprocess is not None else None,
                "cache_keys": {
                    "preprocess": preprocess_plans[key].cache_key if key in preprocess_plans else None,
                    "cluster": outcome.get("cluster_key", _cluster_key(candidate, preprocess, int(entry["reserved"]), int(entry["requested"])) if preprocess is not None else None),
                    "evaluation": outcome.get("evaluation_key"),
                },
                "reuse": {
                    "preprocess": preprocess_actions.get(key, "not_run"),
                    "cluster": outcome.get("cluster_action", "not_run"),
                    "evaluation": outcome.get("evaluation_action", "not_run"),
                },
            }
        )
        reuse_ledger.extend(
            [
                {
                    "stage": "cluster",
                    "candidate": key,
                    "action": outcome.get("cluster_action", "not_run"),
                    "cache_key": candidate_results[-1]["cache_keys"]["cluster"],
                },
                {
                    "stage": "evaluation",
                    "candidate": key,
                    "action": outcome.get("evaluation_action", "not_run"),
                    "cache_key": candidate_results[-1]["cache_keys"]["evaluation"],
                },
            ]
        )

    ranked = rank_candidates(candidate_results)
    best = next(
        (
            candidate
            for candidate in ranked
            if candidate["status"] in {"ok", "warning"} and candidate["metrics_comparable"]
        ),
        None,
    )
    budget = finalize_budget(budget_plan, consumed, ledger_outcomes)

    all_issue_objects: list[Issue] = list(global_issues)
    for candidate_result in candidate_results:
        for warning in candidate_result["warnings"]:
            all_issue_objects.append(_issue_from_mapping(warning))
        for error in candidate_result["errors"]:
            all_issue_objects.append(_issue_from_mapping(error))
    failures = issue_dicts(issue for issue in all_issue_objects if issue.severity == "error")
    warnings = issue_dicts(issue for issue in all_issue_objects if issue.severity != "error")

    diag_payload = {
        "schema_version": 1,
        "data_fingerprint": table.content_fingerprint,
        "candidate_evaluations": [
            {
                "candidate": candidate["stable_key"],
                "status": candidate["status"],
                "evaluation_key": candidate["cache_keys"]["evaluation"],
                "cluster_key": candidate["cache_keys"]["cluster"],
                "metrics_comparable": candidate["metrics_comparable"],
            }
            for candidate in ranked
        ],
        "ranking": [candidate["stable_key"] for candidate in ranked],
        "budget_fingerprint": budget["content_fingerprint"],
        "failures": failures,
        "warnings": warnings,
    }
    diag_key = stable_hash(diag_payload)
    diag_record, diag_cache_issues = cache.get("diagnostics", diag_key)
    global_issues.extend(diag_cache_issues)
    if diag_record is not None:
        diagnostics_action = "reuse"
    else:
        diagnostics_action = "recompute"
        cache.put(
            "diagnostics",
            diag_key,
            {
                "failure_count": len(failures),
                "warning_count": len(warnings),
                "ranking": [candidate["stable_key"] for candidate in ranked],
                "budget_fingerprint": budget["content_fingerprint"],
            },
        )
    reuse_ledger.append(
        {
            "stage": "diagnostics",
            "candidate": "*",
            "action": diagnostics_action,
            "cache_key": diag_key,
        }
    )
    stage_order = {"preprocess": 0, "cluster": 1, "evaluation": 2, "diagnostics": 3}
    reuse_ledger = sorted(
        reuse_ledger,
        key=lambda item: (stage_order.get(item["stage"], 9), item["candidate"], item.get("cache_key") or ""),
    )
    stage_points: dict[str, int] = {stage: 0 for stage in ("preprocess", "cluster", "evaluation", "diagnostics")}
    for entry in reuse_ledger:
        points = table.rows if entry.get("action") == "recompute" else 0
        entry["points_recomputed"] = int(points)
        stage_points[entry["stage"]] = stage_points.get(entry["stage"], 0) + int(points)
    candidate_points: dict[str, int] = {}
    seen_candidates: set[str] = set()
    candidates_with_downstream_reuse: set[str] = set()
    for entry in reuse_ledger:
        candidate_key = entry["candidate"]
        if candidate_key == "*":
            continue
        seen_candidates.add(candidate_key)
        if entry["stage"] in {"cluster", "evaluation"} and entry.get("action") == "reuse":
            candidates_with_downstream_reuse.add(candidate_key)
        candidate_points[candidate_key] = candidate_points.get(candidate_key, 0) + int(
            entry.get("points_recomputed", 0)
        )
    recomputed_candidates = sorted(
        candidate for candidate, points in candidate_points.items() if points > 0
    )
    fully_reused_candidates = sorted(
        candidate
        for candidate in seen_candidates
        if candidate_points.get(candidate, 0) == 0
        and candidate in candidates_with_downstream_reuse
    )
    recompute_accounting = {
        "granularity": "candidate_stage",
        "policy": "cache_miss_only_no_unaffected_point_recomputation",
        "unaffected_recompute_points": 0,
        "sample_count": table.rows,
        "points_by_stage": stage_points,
        "total_points_recomputed": sum(stage_points.values()),
        "points_by_candidate": dict(sorted(candidate_points.items())),
        "candidates_recomputed": recomputed_candidates,
        "candidates_fully_reused": fully_reused_candidates,
    }

    report = {
        "schema_version": 1,
        "state_fingerprint": state_fingerprint,
        "command_seq": int(state.get("command_seq", 0)),
        "complete": True,
        "data": {
            "rows": table.rows,
            "size_level": data_description.get("size_level", ""),
            "content_fingerprint": table.content_fingerprint,
            "columns": [
                {
                    "name": name,
                    "type": table.column_specs[name].get("type"),
                    "content_fingerprint": table.column_content_fingerprints[name],
                    "description_fingerprint": table.column_description_fingerprints[name],
                }
                for name in sorted(table.values)
            ],
            "declared_columns": sorted(table.declared_columns),
            "available_columns": sorted(table.available_columns),
        },
        "preprocessing": deep_copy(preprocessing_config),
        "candidates": ranked,
        "best_candidate": _summary_candidate(best),
        "budget": budget,
        "failures": failures,
        "warnings": warnings,
        "injections": injection_report,
        "reuse_ledger": reuse_ledger,
        "recompute_accounting": recompute_accounting,
        "diagnostics": {
            "cache_key": diag_key,
            "content_fingerprint": stable_hash(diag_payload),
            "cache_action": diagnostics_action,
        },
        "counts": {
            "candidates": len(ranked),
            "completed": sum(1 for item in ranked if item["status"] in {"ok", "warning"}),
            "failed": sum(1 for item in ranked if item["status"] in {"failed", "invalid"}),
            "pending": sum(1 for item in ranked if item["status"] == "pending"),
        },
        "cache_stats": cache.stats(),
    }
    _notify(progress, stage="complete", completed=total_candidates, total=total_candidates)
    return report


def _execute_preprocessing(
    table: Table,
    candidates: list[dict[str, Any]],
    preprocessing_config: dict[str, Any],
    cache: CacheStore,
    *,
    concurrency: int,
    progress: ProgressCallback | None,
) -> tuple[
    dict[str, PreprocessResult],
    dict[str, PreprocessPlan],
    dict[str, str],
    list[Issue],
]:
    results: dict[str, PreprocessResult] = {}
    plans: dict[str, PreprocessPlan] = {}
    actions: dict[str, str] = {}
    cache_issues: list[Issue] = []
    unique_misses: dict[str, tuple[PreprocessPlan, str]] = {}
    unique_from_cache: dict[str, PreprocessResult] = {}

    for candidate in candidates:
        key = candidate["stable_key"]
        if not candidate["valid"]:
            actions[key] = "not_run"
            continue
        try:
            plan = make_preprocess_plan(
                table,
                candidate.get("feature_columns", []),
                preprocessing=preprocessing_config,
                train_fraction=float(candidate.get("train_fraction", 1.0)),
            )
        except Exception as exc:
            actions[key] = "error"
            results[key] = PreprocessResult(
                matrix=np.empty((0, 0), dtype=np.float64),
                metadata={"failed": True, "error": str(exc)},
                issues=[
                    Issue(
                        E_PIPELINE_ERROR,
                        f"cannot build preprocessing plan: {exc}",
                        context={"scope": "preprocess", "candidate": key},
                    )
                ],
            )
            continue
        plans[key] = plan
        record, issues = cache.get("preprocess", plan.cache_key)
        cache_issues.extend(issues)
        if record is not None and _valid_preprocess_record(record):
            metadata = dict(record["payload"].get("metadata", {}))
            metadata["candidate"] = key
            cached_issues = [
                _remap_issue(_issue_from_mapping(item), key)
                for item in record["payload"].get("issues", [])
                if isinstance(item, dict)
            ]
            result = PreprocessResult(
                matrix=np.asarray(record.get("array_value", np.empty((0, 0))), dtype=np.float64),
                metadata=metadata,
                issues=cached_issues,
            )
            unique_from_cache[plan.cache_key] = result
            results[key] = result
            actions[key] = "reuse"
        else:
            if record is not None:
                cache_issues.append(
                    Issue(
                        E_PIPELINE_ERROR,
                        "preprocess cache payload was structurally invalid; recomputing",
                        severity="warning",
                        context={"scope": "cache", "candidate": key, "cache_key": plan.cache_key},
                    )
                )
            unique_misses.setdefault(plan.cache_key, (plan, key))
            actions[key] = "recompute"

    if unique_misses:
        completed = 0
        total = len(unique_misses)
        with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as executor:
            futures = {
                executor.submit(
                    fit_transform,
                    table,
                    plan,
                    preprocessing=preprocessing_config,
                    candidate_id=candidate_key,
                ): cache_key
                for cache_key, (plan, candidate_key) in sorted(unique_misses.items())
            }
            for future in as_completed(futures):
                cache_key = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    plan, candidate_key = unique_misses[cache_key]
                    result = PreprocessResult(
                        matrix=np.empty((0, 0), dtype=np.float64),
                        metadata={"failed": True, "error": str(exc)},
                        issues=[
                            Issue(
                                E_PIPELINE_ERROR,
                                f"preprocessing failed: {exc}",
                                context={"scope": "preprocess", "candidate": candidate_key},
                            )
                        ],
                    )
                unique_from_cache[cache_key] = result
                cache.put(
                    "preprocess",
                    cache_key,
                    {
                        "metadata": result.metadata,
                        "issues": [issue.to_dict() for issue in result.issues],
                        "matrix_fingerprint": (
                            result.metadata.get("matrix_fingerprint") if result.matrix.size else None
                        ),
                    },
                    array=result.matrix,
                )
                completed += 1
                _notify(
                    progress,
                    stage="preprocess",
                    completed=completed,
                    total=total,
                    candidate=unique_misses[cache_key][1],
                )

    for candidate in candidates:
        key = candidate["stable_key"]
        plan = plans.get(key)
        if plan is None:
            continue
        base_result = unique_from_cache.get(plan.cache_key)
        if base_result is None:
            continue
        metadata = deep_copy(base_result.metadata)
        metadata["candidate"] = key
        issues = [_remap_issue(issue, key) for issue in base_result.issues]
        results[key] = PreprocessResult(
            matrix=base_result.matrix,
            metadata=metadata,
            issues=issues,
        )
    return results, plans, actions, cache_issues


def _execute_cluster_tasks(
    tasks: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    table: Table,
    cache: CacheStore,
    concurrency: int,
    progress: ProgressCallback | None,
    total_candidates: int,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as executor:
        futures = {
            executor.submit(_run_cluster_task, task, config, table, cache): task["candidate"]["stable_key"]
            for task in sorted(tasks, key=lambda item: item["candidate"]["stable_key"])
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as exc:
                task = next(item for item in tasks if item["candidate"]["stable_key"] == key)
                allocation = int(task["entry"]["reserved"])
                results[key] = {
                    "outcome": {
                        "status": "failed",
                        "errors": [
                            Issue(
                                E_PIPELINE_ERROR,
                                f"candidate execution failed: {exc}",
                                context={
                                    "scope": "pipeline",
                                    "candidate": key,
                                    "method": task["candidate"].get("method"),
                                    "params": task["candidate"].get("params", {}),
                                },
                            )
                        ],
                        "warnings": [],
                        "cluster_action": task.get("cluster_action", "recompute"),
                        "evaluation_action": "not_run",
                        "cluster_key": task.get("cluster_key"),
                        "evaluation_key": task.get("evaluation_key"),
                    },
                    "consumed": 0,
                    "ledger_outcome": "pipeline_error",
                    "cache_issues": [],
                    "allocation": allocation,
                }
            completed += 1
            _notify(
                progress,
                stage="cluster",
                completed=completed,
                total=len(tasks),
                candidate=key,
                overall_total=total_candidates,
            )
    return results


def _run_cluster_task(
    task: dict[str, Any], config: dict[str, Any], table: Table, cache: CacheStore
) -> dict[str, Any]:
    candidate = task["candidate"]
    key = candidate["stable_key"]
    preprocess: PreprocessResult = task["preprocess"]
    entry = task["entry"]
    allocation = int(entry["reserved"])
    requested = int(entry["requested"])
    cache_issues: list[Issue] = []
    delay_ms = float(candidate.get("_delay_ms", 0.0) or 0.0)
    if delay_ms > 0:
        time.sleep(delay_ms / 1000.0)

    cluster_payload = task.get("cluster_payload")
    labels = task.get("labels")
    cluster_action = task.get("cluster_action", "recompute")
    if cluster_payload is None:
        method = candidate["method"]
        params = candidate.get("params", {})
        runtime = candidate.get("runtime_settings", {})
        matrix = preprocess.matrix
        if method == "kmeans":
            algorithm_result = kmeans(
                matrix,
                n_clusters=int(params["n_clusters"]),
                max_iter=allocation,
                tolerance=float(runtime.get("tolerance", 1e-8)),
                requested_iterations=requested,
            )
        elif method == "dbscan":
            algorithm_result = dbscan(
                matrix,
                eps=float(params["eps"]),
                min_samples=int(params["min_samples"]),
            )
        elif method == "gmm":
            algorithm_result = gaussian_mixture(
                matrix,
                n_components=int(params["n_components"]),
                max_iter=allocation,
                tolerance=float(runtime.get("tolerance", 1e-6)),
                covariance_type=str(params.get("covariance_type", "diag")),
                regularization=float(runtime.get("covariance_regularization", 1e-5)),
                requested_iterations=requested,
            )
        else:
            raise ValueError(f"unsupported method: {method!r}")
        budget_exhausted = not algorithm_result.converged and allocation < requested
        issues = list(algorithm_result.issues)
        if budget_exhausted:
            issues.append(
                Issue(
                    E_CAND_BUDGET_EXHAUSTED,
                    "candidate exceeded its per-candidate iteration allocation",
                    context={
                        "scope": "scheduler",
                        "candidate": key,
                        "method": method,
                        "params": params,
                        "allocation": allocation,
                        "requested": requested,
                    },
                )
            )
        cluster_payload = {
            "method": method,
            "params": params,
            "allocation": allocation,
            "requested": requested,
            "iterations": int(algorithm_result.iterations),
            "converged": bool(algorithm_result.converged),
            "objective_trace": [float(value) for value in algorithm_result.objective_trace],
            "issues": [issue.to_dict() for issue in issues],
            "extras": algorithm_result.extras,
            "budget_exhausted": bool(budget_exhausted),
        }
        cluster_content_hash = stable_hash(
            {
                "payload": cluster_payload,
                "labels": hash_array(algorithm_result.labels) if not budget_exhausted else "<aborted>",
            }
        )
        cluster_payload["content_fingerprint"] = cluster_content_hash
        labels_to_store = (
            np.empty(0, dtype=np.int64) if budget_exhausted else algorithm_result.labels.astype(np.int64)
        )
        cache.put("cluster", task["cluster_key"], cluster_payload, array=labels_to_store)
        cluster_action = "recompute"
        if budget_exhausted:
            labels = None
        else:
            labels = algorithm_result.labels.astype(np.int64)
    else:
        labels = np.asarray(labels, dtype=np.int64)

    issues = [_issue_from_mapping(item) for item in cluster_payload.get("issues", []) if isinstance(item, dict)]
    if cluster_payload.get("budget_exhausted", False):
        errors = [issue for issue in issues if issue.severity == "error"]
        warnings = [issue for issue in issues if issue.severity != "error"]
        if not errors:
            errors.append(
                Issue(
                    E_CAND_BUDGET_EXHAUSTED,
                    "candidate exceeded its per-candidate iteration allocation",
                    context={"scope": "scheduler", "candidate": key},
                )
            )
        return {
            "outcome": {
                "status": "failed",
                "errors": errors,
                "warnings": warnings,
                "labels": None,
                "cluster_payload": cluster_payload,
                "evaluation_payload": None,
                "cluster_key": task["cluster_key"],
                "evaluation_key": None,
                "cluster_action": cluster_action,
                "evaluation_action": "not_run",
            },
            "consumed": int(cluster_payload.get("iterations", allocation)),
            "ledger_outcome": "budget_exhausted",
            "cache_issues": cache_issues,
        }

    evaluation_key = task.get("evaluation_key") or _evaluation_key(
        candidate=candidate,
        preprocess=preprocess,
        cluster_fingerprint=str(cluster_payload["content_fingerprint"]),
        config=config,
    )
    expected_clusters = None
    if candidate["method"] == "kmeans":
        expected_clusters = int(candidate["params"]["n_clusters"])
    elif candidate["method"] == "gmm":
        expected_clusters = int(candidate["params"]["n_components"])
    metrics, evaluation_warnings, noncomparable = evaluate_labels(
        preprocess.matrix,
        labels,
        method=candidate["method"],
        expected_clusters=expected_clusters,
        metrics=config["evaluation"].get("metrics", []),
        density_regularization=float(config["evaluation"].get("density_regularization", 1e-6)),
        candidate_id=key,
        params=candidate.get("params", {}),
    )
    evaluation_payload = {
        "metrics": metrics,
        "issues": [issue.to_dict() for issue in evaluation_warnings],
        "noncomparable_reasons": noncomparable,
        "cluster_fingerprint": str(cluster_payload["content_fingerprint"]),
        "preprocess_fingerprint": preprocess.content_fingerprint,
    }
    evaluation_payload["content_fingerprint"] = stable_hash(evaluation_payload)
    cache.put("evaluation", evaluation_key, evaluation_payload)
    evaluation_action = "recompute"

    evaluation_issue_objects = [
        _issue_from_mapping(item) for item in evaluation_payload.get("issues", []) if isinstance(item, dict)
    ]
    cluster_issue_objects = [
        _issue_from_mapping(item) for item in cluster_payload.get("issues", []) if isinstance(item, dict)
    ]
    errors = [issue for issue in cluster_issue_objects + evaluation_issue_objects if issue.severity == "error"]
    warnings = [issue for issue in cluster_issue_objects + evaluation_issue_objects if issue.severity != "error"]
    return {
        "outcome": {
            "status": "ok",
            "errors": errors,
            "warnings": warnings,
            "labels": labels,
            "cluster_payload": cluster_payload,
            "evaluation_payload": evaluation_payload,
            "cluster_key": task["cluster_key"],
            "evaluation_key": evaluation_key,
            "cluster_action": cluster_action,
            "evaluation_action": evaluation_action,
        },
        "consumed": int(cluster_payload.get("iterations", allocation)),
        "ledger_outcome": "completed",
        "cache_issues": cache_issues,
    }


def _outcome_from_cluster_cache(
    candidate: dict[str, Any],
    preprocess: PreprocessResult,
    cluster_payload: dict[str, Any],
    labels: np.ndarray,
    cluster_action: str,
    evaluation_action: str,
    *,
    evaluation_payload: dict[str, Any] | None = None,
    evaluation_key: str | None = None,
) -> dict[str, Any]:
    key = candidate["stable_key"]
    cluster_issues = [_remap_issue(_issue_from_mapping(item), key) for item in cluster_payload.get("issues", []) if isinstance(item, dict)]
    if cluster_payload.get("budget_exhausted", False):
        errors = [issue for issue in cluster_issues if issue.severity == "error"]
        if not errors:
            errors.append(
                Issue(
                    E_CAND_BUDGET_EXHAUSTED,
                    "candidate exceeded its per-candidate iteration allocation",
                    context={"scope": "scheduler", "candidate": key},
                )
            )
        return {
            "status": "failed",
            "errors": errors,
            "warnings": [issue for issue in cluster_issues if issue.severity != "error"],
            "labels": None,
            "cluster_payload": cluster_payload,
            "evaluation_payload": None,
            "cluster_key": _cluster_key(candidate, preprocess, int(cluster_payload["allocation"]), int(cluster_payload["requested"])),
            "evaluation_key": None,
            "cluster_action": cluster_action,
            "evaluation_action": evaluation_action,
        }
    evaluation_issues = (
        [_remap_issue(_issue_from_mapping(item), key) for item in evaluation_payload.get("issues", []) if isinstance(item, dict)]
        if evaluation_payload is not None
        else []
    )
    all_issues = cluster_issues + evaluation_issues
    return {
        "status": "ok",
        "errors": [issue for issue in all_issues if issue.severity == "error"],
        "warnings": [issue for issue in all_issues if issue.severity != "error"],
        "labels": labels,
        "cluster_payload": cluster_payload,
        "evaluation_payload": evaluation_payload,
        "cluster_key": _cluster_key(candidate, preprocess, int(cluster_payload["allocation"]), int(cluster_payload["requested"])),
        "evaluation_key": evaluation_key,
        "cluster_action": cluster_action,
        "evaluation_action": evaluation_action,
    }


def _apply_candidate_injections(
    candidates: list[Any], injections: list[Any]
) -> tuple[list[Any], list[Issue], list[dict[str, Any]]]:
    result = deep_copy(candidates)
    issues: list[Issue] = []
    report: list[dict[str, Any]] = []
    by_id = {item.get("id"): item for item in result if isinstance(item, dict) and isinstance(item.get("id"), str)}
    for injection in injections:
        if not isinstance(injection, dict):
            continue
        kind = injection.get("type")
        if kind not in {"invalid_hyperparameter", "slow_candidate"}:
            continue
        identifier = injection.get("candidate")
        candidate = by_id.get(identifier)
        if candidate is None:
            issues.append(
                Issue(
                    E_INJECTION_INVALID,
                    f"injection targets unknown candidate {identifier!r}",
                    severity="warning",
                    context={"scope": "injection", "injection": injection.get("id"), "candidate": identifier},
                )
            )
            report.append(
                {"id": injection.get("id"), "type": kind, "candidate": identifier, "isolated": False}
            )
            continue
        if kind == "invalid_hyperparameter":
            patch = injection.get("params")
            if not isinstance(patch, dict):
                patch = {}
                parameter = injection.get("parameter")
                if isinstance(parameter, str) and "value" in injection:
                    patch = {parameter: injection.get("value")}
            candidate["params"] = {**candidate.get("params", {}), **patch}
            report.append(
                {
                    "id": injection.get("id"),
                    "type": kind,
                    "candidate": identifier,
                    "params": patch,
                    "isolated": True,
                }
            )
        else:
            delay_ms = injection.get("delay_ms", injection.get("value", 0))
            try:
                delay = max(0.0, float(delay_ms))
            except (TypeError, ValueError):
                delay = 0.0
            candidate["_delay_ms"] = delay
            report.append(
                {
                    "id": injection.get("id"),
                    "type": kind,
                    "candidate": identifier,
                    "delay_ms": delay,
                    "isolated": True,
                }
            )
    return result, issues, report


def _cluster_key(
    candidate: dict[str, Any],
    preprocess: PreprocessResult,
    allocation: int,
    requested: int,
) -> str:
    return stable_hash(
        {
            "kind": "cluster",
            "schema": 1,
            "candidate": candidate["stable_key"],
            "preprocess_content_fingerprint": preprocess.content_fingerprint,
            "method": candidate.get("method"),
            "params": candidate.get("params", {}),
            "runtime": candidate.get("runtime_settings", {}),
            "allocation": int(allocation),
            "requested": int(requested),
        }
    )


def _evaluation_key(
    *,
    candidate: dict[str, Any],
    preprocess: PreprocessResult,
    cluster_fingerprint: str,
    config: dict[str, Any],
) -> str:
    return stable_hash(
        {
            "kind": "evaluation",
            "schema": 1,
            "candidate": candidate["stable_key"],
            "cluster_content_fingerprint": cluster_fingerprint,
            "preprocess_content_fingerprint": preprocess.content_fingerprint,
            "evaluation": config.get("evaluation", {}),
            "method": candidate.get("method"),
        }
    )


def _valid_preprocess_record(record: dict[str, Any]) -> bool:
    payload = record.get("payload", {})
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return False
    array = record.get("array_value")
    if metadata.get("failed", False):
        return array is not None
    if array is None or not isinstance(array, np.ndarray) or array.ndim != 2:
        return False
    return array.shape[0] > 0


def _valid_cluster_record(record: dict[str, Any]) -> bool:
    payload = record.get("payload", {})
    return isinstance(payload.get("content_fingerprint"), str) and "iterations" in payload and "budget_exhausted" in payload


def _valid_evaluation_record(record: dict[str, Any]) -> bool:
    payload = record.get("payload", {})
    return isinstance(payload.get("metrics"), dict) and "noncomparable_reasons" in payload


def _issue_from_mapping(value: dict[str, Any]) -> Issue:
    return Issue(
        code=str(value.get("code", "E_UNKNOWN")),
        message=str(value.get("message", "")),
        severity=str(value.get("severity", "error")),
        context=dict(value.get("context", {})) if isinstance(value.get("context"), dict) else {},
    )


def _remap_issue(issue: Issue, candidate: str) -> Issue:
    context = dict(issue.context)
    if issue.context.get("scope") in {"preprocess", "algorithm", "evaluation", "pipeline", "scheduler"}:
        context["candidate"] = candidate
    elif "candidate" in context:
        context["candidate"] = candidate
    return Issue(code=issue.code, message=issue.message, severity=issue.severity, context=context)


def _enrich_candidate_issues(issues: list[Issue], candidate: dict[str, Any]) -> list[Issue]:
    enriched: list[Issue] = []
    for issue in issues:
        context = dict(issue.context)
        context.setdefault("candidate", candidate["stable_key"])
        context.setdefault("method", candidate.get("method"))
        context.setdefault("params", candidate.get("params", {}))
        enriched.append(Issue(issue.code, issue.message, issue.severity, context))
    return enriched


def _unavailable_metrics(metrics: list[str], reason: str) -> dict[str, Any]:
    requested = metrics or ["silhouette", "calinski_harabasz", "density_log_likelihood"]
    return {
        metric: {
            "value": None,
            "comparable": False,
            "reason_code": reason,
            "covered_samples": 0,
            "details": {},
        }
        for metric in requested
    }


def _candidate_status(status: str, errors: list[Issue], warnings: list[Issue]) -> str:
    if status in {"invalid", "pending"}:
        return status
    if errors or status == "failed":
        return "failed"
    if warnings:
        return "warning"
    return "ok"


def _summary_candidate(candidate: dict[str, Any] | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return {
        "id": candidate["id"],
        "method": candidate["method"],
        "params": candidate["params"],
        "rank": candidate.get("rank"),
        "metrics": candidate.get("metrics"),
        "metrics_comparable": candidate.get("metrics_comparable"),
    }


def _notify(
    callback: ProgressCallback | None,
    *,
    stage: str,
    completed: int,
    total: int,
    candidate: str | None = None,
    overall_total: int | None = None,
) -> None:
    if callback is None:
        return
    payload = {
        "stage": stage,
        "completed": int(completed),
        "total": int(total),
        "ratio": (float(completed) / float(total)) if total else 0.0,
    }
    if candidate is not None:
        payload["candidate"] = candidate
    if overall_total is not None:
        payload["overall_total"] = int(overall_total)
    try:
        callback(payload)
    except Exception:
        pass
