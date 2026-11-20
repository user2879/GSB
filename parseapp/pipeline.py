"""End-to-end deterministic parser orchestration."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .charts import chart_cache_key, structure_charts
from .config import ParseConfig, load_config
from .detection import classify_regions, detect_lines, enumerate_candidates, line_cache_key
from .grid import grid_cache_key, rebuild_grid
from .loader import PageInput, load_pages
from .scheduler import plan_budget
from .snapshot import Snapshot, SnapshotStore, write_json_atomic
from .validation import PageInvalid


class PipelineError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class SimulatedTermination(RuntimeError):
    """Deterministic failpoint used to test recovery between commits."""


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _input_signature(sample_dir: Path) -> str:
    entries: list[dict[str, Any]] = []
    if sample_dir.is_dir():
        for path in sorted(sample_dir.iterdir(), key=lambda item: item.name):
            if not path.is_file() or path.suffix.lower() not in {".png", ".json"}:
                continue
            try:
                data = path.read_bytes()
            except OSError as exc:
                entries.append({"name": path.name, "error": str(exc)})
                continue
            entries.append(
                {
                    "name": path.name,
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
    return _sha256_text(_canonical_json(entries))


def _prior_page_artifacts(snapshot: Snapshot | None) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if snapshot is None:
        return result
    for relative, payload in snapshot.artifacts.items():
        if not relative.startswith("pages/") or not relative.endswith(".json"):
            continue
        if not isinstance(payload, Mapping):
            continue
        file_name = payload.get("file_name")
        if isinstance(file_name, str):
            result[file_name] = dict(payload)
    return result


def _prior_stage(
    prior_page: Mapping[str, Any] | None, stage: str, key: str
) -> Mapping[str, Any] | None:
    if not prior_page:
        return None
    entry = prior_page.get(stage)
    if (
        isinstance(entry, Mapping)
        and entry.get("key") == key
        and isinstance(entry.get("artifact"), Mapping)
    ):
        return entry
    return None


def _stage_reason(
    prior_page: Mapping[str, Any] | None, stage: str, key: str, reused: bool
) -> str:
    if reused:
        return "CACHE_KEY_MATCH"
    if not prior_page or not isinstance(prior_page.get(stage), Mapping):
        return "NO_PRIOR_ARTIFACT"
    return "CACHE_KEY_CHANGED"


def _process_valid_page(
    page: PageInput,
    config: ParseConfig,
    previous_page: Mapping[str, Any] | None,
    line_key: str,
    reuse_line: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prior_line = _prior_stage(previous_page, "line", line_key)
    if reuse_line and prior_line is None:
        raise PipelineError("E_INTERNAL_CACHE", f"missing reusable line artifact for {page.file_name}")

    if reuse_line:
        line_artifact = dict(prior_line["artifact"])  # type: ignore[index]
        line_status = "reused"
        line_reason = "CACHE_KEY_MATCH"
        points_consumed = 0
    else:
        binary = page.gray < config.binary_threshold
        line_artifact = detect_lines(binary, config, page.candidates)
        line_status = "recomputed"
        line_reason = _stage_reason(previous_page, "line", line_key, False)
        points_consumed = int(line_artifact["points_consumed"])

    decisions = classify_regions(
        line_artifact["segments"], page.truth, config.chart_decision_threshold
    )

    grid_key = grid_cache_key(
        line_artifact["segments"], page.truth, config, line_key, decisions["tables"]
    )
    prior_grid = _prior_stage(previous_page, "grid", grid_key)
    if prior_grid is not None:
        grid_artifact = dict(prior_grid["artifact"])
        grid_status = "reused"
    else:
        grid_artifact = rebuild_grid(
            line_artifact["segments"],
            page.truth,
            page.gray,
            config,
            line_key,
            decisions["tables"],
            previous_grid=(previous_page or {}).get("grid", {}).get("artifact")
            if isinstance((previous_page or {}).get("grid"), Mapping)
            else None,
        )
        grid_status = "recomputed"
    if grid_status == "reused":
        grid_reason = "CACHE_KEY_MATCH"
    elif line_status == "recomputed":
        grid_reason = "UPSTREAM_LINE_RECOMPUTED"
    else:
        grid_reason = _stage_reason(previous_page, "grid", grid_key, False)

    chart_key = chart_cache_key(
        page.content_fingerprint,
        line_key,
        page.truth,
        config,
        decisions["charts"],
    )
    prior_chart = _prior_stage(previous_page, "chart", chart_key)
    if prior_chart is not None:
        chart_artifact = dict(prior_chart["artifact"])
        chart_status = "reused"
        chart_reason = "CACHE_KEY_MATCH"
    else:
        previous_chart_artifact = None
        if isinstance((previous_page or {}).get("chart"), Mapping):
            previous_chart_artifact = (previous_page or {}).get("chart", {}).get("artifact")
        chart_artifact = structure_charts(
            page.gray,
            page.truth,
            line_artifact["segments"],
            decisions["charts"],
            config,
            page.content_fingerprint,
            line_key,
            previous_chart_artifact,
        )
        chart_status = "recomputed"
        chart_reason = _stage_reason(previous_page, "chart", chart_key, False)

    tables = copy.deepcopy(grid_artifact.get("tables", []))
    charts = copy.deepcopy(chart_artifact.get("charts", []))
    if grid_status == "reused":
        for table in tables:
            for cell in table.get("cells", []):
                cell["cache_status"] = "reused"
        cell_cache = {
            "reused": sum(len(table.get("cells", [])) for table in tables),
            "recomputed": 0,
        }
    else:
        cell_cache = dict(grid_artifact.get("cell_cache", {"reused": 0, "recomputed": 0}))
    if chart_status == "reused":
        for chart in charts:
            for point in chart.get("points", []):
                point["cache_status"] = "reused"
        point_cache = {
            "reused": sum(len(chart.get("points", [])) for chart in charts),
            "recomputed": 0,
        }
    else:
        point_cache = dict(chart_artifact.get("point_cache", {"reused": 0, "recomputed": 0}))

    artifact = {
        "schema_version": "1.0",
        "file_name": page.file_name,
        "page_id": page.page_id,
        "content_fingerprint": page.content_fingerprint,
        "sort_key": page.sort_key,
        "detection_point_cost": page.detection_point_cost,
        "line": {"key": line_key, "artifact": line_artifact},
        "grid": {"key": grid_key, "artifact": grid_artifact},
        "chart": {"key": chart_key, "artifact": chart_artifact},
    }

    cache = {
        "line": {
            "key": line_key,
            "status": line_status,
            "reason": line_reason,
            "sort_key": f"{page.file_name}|line",
        },
        "grid": {
            "key": grid_key,
            "status": grid_status,
            "reason": grid_reason,
            "sort_key": f"{page.file_name}|grid",
        },
        "chart": {
            "key": chart_key,
            "status": chart_status,
            "reason": chart_reason,
            "sort_key": f"{page.file_name}|chart",
        },
        "cells": cell_cache,
        "points": point_cache,
    }
    invalidation_units: list[dict[str, Any]] = [
        {"unit": "line_segments", "status": line_status, "reason": line_reason, "cache_key": line_key},
        {"unit": "cell_grid", "status": grid_status, "reason": grid_reason, "cache_key": grid_key},
        {"unit": "chart_points", "status": chart_status, "reason": chart_reason, "cache_key": chart_key},
    ]
    for table in tables:
        for cell in table.get("cells", []):
            invalidation_units.append(
                {
                    "unit": "cell",
                    "id": cell.get("cell_id"),
                    "status": cell.get("cache_status"),
                    "cache_key": cell.get("cache_key"),
                    "row_line_ids": cell.get("row_line_ids"),
                    "col_line_ids": cell.get("col_line_ids"),
                }
            )
    for chart in charts:
        for point in chart.get("points", []):
            invalidation_units.append(
                {
                    "unit": "data_point",
                    "id": point.get("point_id"),
                    "status": point.get("cache_status"),
                    "cache_key": point.get("cache_key"),
                    "axis_source": point.get("axis_source"),
                }
            )

    report = {
        "file": page.file_name,
        "page_id": page.page_id,
        "sort_key": page.sort_key,
        "status": "ok",
        "business_code": None,
        "detail_code": None,
        "message": None,
        "content_fingerprint": page.content_fingerprint,
        "detection_point_cost": page.detection_point_cost,
        "detection_points_consumed": points_consumed,
        "line_detection": {
            "segments": copy.deepcopy(line_artifact.get("segments", [])),
            "dropped_candidates": copy.deepcopy(line_artifact.get("dropped_candidates", [])),
            "drop_counts": copy.deepcopy(line_artifact.get("drop_counts", {})),
            "candidate_count": line_artifact.get("candidate_count"),
            "points_consumed": line_artifact.get("points_consumed"),
            "merge_rule": copy.deepcopy(line_artifact.get("merge_rule", {})),
            "param_hash": line_artifact.get("param_hash"),
        },
        "tables": tables,
        "charts": charts,
        "warnings": copy.deepcopy(chart_artifact.get("warnings", [])),
        "cache": cache,
        "invalidations": {
            "page_any_recomputed": any(
                status == "recomputed" for status in (line_status, grid_status, chart_status)
            ),
            "units": invalidation_units,
        },
    }
    return report, artifact


def _failure_report(
    file_name: str,
    *,
    stem: str,
    detail_code: str,
    message: str,
    content_fingerprint: str | None = None,
    points_consumed: int = 0,
    point_cost: int = 0,
) -> dict[str, Any]:
    return {
        "file": file_name,
        "page_id": stem,
        "sort_key": f"{file_name}|page",
        "status": "failed",
        "business_code": "E_PAGE_INVALID",
        "detail_code": detail_code,
        "message": message,
        "content_fingerprint": content_fingerprint,
        "detection_point_cost": point_cost,
        "detection_points_consumed": points_consumed,
        "tables": [],
        "charts": [],
        "line_detection": None,
        "warnings": [],
        "cache": {},
        "invalidations": {
            "page_any_recomputed": False,
            "units": [{"unit": "page", "status": "invalid"}],
        },
    }


def _pending_report(
    page: PageInput,
    line_key: str,
    *,
    points_required: int,
    reason: str,
) -> dict[str, Any]:
    return {
        "file": page.file_name,
        "page_id": page.page_id,
        "sort_key": page.sort_key,
        "status": "pending",
        "business_code": "E_POINT_BUDGET_PENDING",
        "detail_code": "E_POINT_BUDGET_PENDING",
        "message": "line detection deferred until a later batch",
        "content_fingerprint": page.content_fingerprint,
        "detection_point_cost": page.detection_point_cost,
        "detection_points_consumed": 0,
        "tables": [],
        "charts": [],
        "line_detection": None,
        "warnings": [],
        "cache": {
            "line": {
                "key": line_key,
                "status": "pending",
                "reason": reason,
                "points_required": points_required,
                "sort_key": f"{page.file_name}|line",
            },
            "grid": {
                "status": "upstream_pending",
                "reason": "LINE_DETECTION_PENDING",
                "sort_key": f"{page.file_name}|grid",
            },
            "chart": {
                "status": "upstream_pending",
                "reason": "LINE_DETECTION_PENDING",
                "sort_key": f"{page.file_name}|chart",
            },
            "cells": {"reused": 0, "recomputed": 0},
            "points": {"reused": 0, "recomputed": 0},
        },
        "invalidations": {
            "page_any_recomputed": False,
            "units": [
                {
                    "unit": "line_segments",
                    "status": "pending",
                    "reason": reason,
                    "cache_key": line_key,
                }
            ],
        },
    }


def _artifact_versions(component_artifacts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    versions: dict[str, Any] = {}
    for file_name, artifact in sorted(component_artifacts.items()):
        versions[f"pages/{Path(file_name).stem}.json"] = {
            "line": artifact.get("line", {}).get("key"),
            "grid": artifact.get("grid", {}).get("key"),
            "chart": artifact.get("chart", {}).get("key"),
        }
    return versions


def _failure_entry(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "file": report["file"],
        "sort_key": report["sort_key"],
        "code": report["business_code"],
        "detail_code": report["detail_code"],
        "message": report["message"],
    }


def run_pipeline(
    sample_dir: str | Path,
    config_path: str | Path | Mapping[str, Any] | ParseConfig,
    state_dir: str | Path,
    *,
    serial: bool = False,
    fail_after_commits: int | None = None,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    sample_root = Path(sample_dir)
    state_root = Path(state_dir)
    config = config_path if isinstance(config_path, ParseConfig) else load_config(config_path)
    if not sample_root.is_dir():
        raise PipelineError("E_INPUT_DIRECTORY", f"sample directory not found: {sample_root}")

    store = SnapshotStore(state_root)
    prior_load = store.load_current()
    prior_snapshot = prior_load.snapshot
    prior_manifest = dict(prior_snapshot.manifest) if prior_snapshot else {}
    prior_pages = _prior_page_artifacts(prior_snapshot)
    snapshot_warnings = list(prior_load.warnings)

    input_signature = _input_signature(sample_root)
    config_hash = config.config_hash
    same_config = bool(prior_snapshot and prior_manifest.get("config_hash") == config_hash)
    same_input = bool(prior_snapshot and prior_manifest.get("input_signature") == input_signature)
    resume_active = bool(
        same_config
        and same_input
        and prior_manifest.get("status") == "active"
        and isinstance(prior_manifest.get("budget_plan"), Mapping)
    )
    prior_recovery_version = prior_snapshot.version if prior_snapshot else 0
    prior_completed_pages = (
        [str(value) for value in prior_manifest.get("completed_pages", [])]
        if same_config and same_input
        else []
    )

    concurrency = 1 if serial else int(config.thread_pool_concurrency)
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="page-load") as loader:
        valid_pages, invalid_pages = load_pages(sample_root, config, loader)

    all_files = sorted(
        set([page.file_name for page in valid_pages] + [page.file_name for page in invalid_pages])
    )
    valid_by_file = {page.file_name: page for page in valid_pages}
    invalid_by_file = {page.file_name: page for page in invalid_pages}
    present_files = set(all_files)
    baseline_completed = sorted(set(prior_completed_pages).intersection(present_files))
    baseline_artifacts = {
        file_name: artifact
        for file_name, artifact in prior_pages.items()
        if file_name in baseline_completed
    }
    baseline_failures = {
        entry.get("file"): dict(entry)
        for entry in prior_manifest.get("failure_entries", [])
        if isinstance(entry, Mapping) and entry.get("file") in present_files
    }

    line_keys: dict[str, str] = {}
    line_costs: dict[str, int] = {}
    needs_line: dict[str, bool] = {}
    for page in valid_pages:
        key = line_cache_key(page.content_fingerprint, config)
        line_keys[page.file_name] = key
        prior_page = prior_pages.get(page.file_name)
        reuse = _prior_stage(prior_page, "line", key) is not None
        needs_line[page.file_name] = not reuse
        if not reuse:
            page.candidates = enumerate_candidates(page.gray < config.binary_threshold)
            page.detection_point_cost = len(page.candidates)
            line_costs[page.file_name] = page.detection_point_cost
        else:
            prior_line_artifact = (prior_page or {}).get("line", {}).get("artifact", {})
            prior_artifact_cost = (prior_page or {}).get("detection_point_cost")
            if prior_artifact_cost is None and isinstance(prior_line_artifact, Mapping):
                prior_artifact_cost = prior_line_artifact.get("candidate_count", 0)
            page.detection_point_cost = int(prior_artifact_cost or 0)

    if resume_active:
        plan = plan_budget(
            all_files,
            line_costs,
            int(prior_manifest.get("batch_budget_limit", config.batch_detection_point_budget)),
            prior_plan=prior_manifest.get("budget_plan"),
        )
        batch_id = int(prior_manifest.get("batch_id", 1))
    else:
        plan = plan_budget(
            all_files,
            line_costs,
            int(config.batch_detection_point_budget),
        )
        previous_batch_id = int(prior_manifest.get("batch_id", 0)) if prior_snapshot else 0
        batch_id = previous_batch_id + 1

    admitted_set = set(plan.admitted_pages)
    pending_set = {entry["file"] for entry in plan.pending}
    future_by_file: dict[str, Future[Any]] = {}
    executor = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="page-parse")
    try:
        for page in valid_pages:
            reuse_line = not needs_line[page.file_name]
            if needs_line[page.file_name] and page.file_name not in admitted_set:
                continue
            future_by_file[page.file_name] = executor.submit(
                _process_valid_page,
                page,
                config,
                prior_pages.get(page.file_name),
                line_keys[page.file_name],
                reuse_line,
            )

        results: list[dict[str, Any]] = []
        current_artifacts: dict[str, dict[str, Any]] = dict(baseline_artifacts)
        baseline_failure_files = set(baseline_failures)
        completed_status_map: dict[str, str] = {
            file_name: "failed" if file_name in baseline_failure_files else "ok"
            for file_name in baseline_completed
        }
        current_failures: dict[str, dict[str, Any]] = dict(baseline_failures)
        current_version = prior_recovery_version
        processed_completed = 0
        commit_count = 0

        def commit(status: str) -> Snapshot:
            nonlocal current_version, commit_count
            completed_pages = sorted(completed_status_map)
            failures = sorted(current_failures.values(), key=lambda item: str(item.get("sort_key", "")))
            manifest = {
                "config_hash": config_hash,
                "input_signature": input_signature,
                "status": status,
                "batch_id": batch_id,
                "batch_budget_limit": plan.limit,
                "consumed_budget": plan.consumed,
                "current_batch_consumed_budget": plan.consumed,
                "current_batch_remaining_budget": plan.remaining,
                "completed_pages": completed_pages,
                "pending_queue": plan.pending,
                "failure_entries": failures,
                "budget_plan": plan.as_dict(),
                "artifact_versions": _artifact_versions(current_artifacts),
                "recovery_source_version": prior_recovery_version,
            }
            artifact_payloads = {
                f"pages/{Path(file_name).stem}.json": artifact
                for file_name, artifact in sorted(current_artifacts.items())
            }
            snapshot = store.commit(
                manifest,
                artifact_payloads,
                expected_parent_version=current_version,
            )
            current_version = snapshot.version
            commit_count += 1
            if fail_after_commits is not None and commit_count >= fail_after_commits:
                raise SimulatedTermination(
                    f"simulated termination after commit {commit_count} at version {snapshot.version}"
                )
            return snapshot

        for file_name in all_files:
            if file_name in invalid_by_file:
                invalid = invalid_by_file[file_name]
                report = _failure_report(
                    file_name,
                    stem=invalid.stem,
                    detail_code=invalid.detail_code,
                    message=invalid.message,
                )
                current_failures[file_name] = _failure_entry(report)
                completed_status_map[file_name] = "failed"
                processed_completed += 1
            elif file_name in pending_set:
                pending_entry = next(
                    entry for entry in plan.pending if entry["file"] == file_name
                )
                pending_reason = "RESUMED_PLAN_NOT_ADMITTED" if resume_active else "BATCH_BUDGET_EXHAUSTED"
                report = _pending_report(
                    valid_by_file[file_name],
                    line_keys[file_name],
                    points_required=int(pending_entry["points_required"]),
                    reason=pending_reason,
                )
            else:
                page = valid_by_file[file_name]
                try:
                    report, artifact = future_by_file[file_name].result()
                except PageInvalid as exc:
                    consumed = (
                        page.detection_point_cost if needs_line[page.file_name] else 0
                    )
                    report = _failure_report(
                        file_name,
                        stem=page.stem,
                        detail_code=exc.detail_code,
                        message=exc.message,
                        content_fingerprint=page.content_fingerprint,
                        points_consumed=consumed,
                        point_cost=page.detection_point_cost,
                    )
                    current_artifacts.pop(file_name, None)
                    current_failures[file_name] = _failure_entry(report)
                    completed_status_map[file_name] = "failed"
                    processed_completed += 1
                except PipelineError as exc:
                    report = _failure_report(
                        file_name,
                        stem=page.stem,
                        detail_code=exc.code,
                        message=exc.message,
                        content_fingerprint=page.content_fingerprint,
                        point_cost=page.detection_point_cost,
                    )
                    current_artifacts.pop(file_name, None)
                    current_failures[file_name] = _failure_entry(report)
                    completed_status_map[file_name] = "failed"
                    processed_completed += 1
                except Exception as exc:
                    report = _failure_report(
                        file_name,
                        stem=page.stem,
                        detail_code="E_PAGE_PROCESS",
                        message=f"{type(exc).__name__}: {exc}",
                        content_fingerprint=page.content_fingerprint,
                        point_cost=page.detection_point_cost,
                    )
                    current_artifacts.pop(file_name, None)
                    current_failures[file_name] = _failure_entry(report)
                    completed_status_map[file_name] = "failed"
                    processed_completed += 1
                else:
                    current_artifacts[file_name] = artifact
                    current_failures.pop(file_name, None)
                    completed_status_map[file_name] = "ok"
                    processed_completed += 1
            results.append(report)
            if (
                processed_completed > 0
                and processed_completed % config.snapshot_commit_interval == 0
            ):
                commit("active")
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    run_status = "budget_incomplete" if plan.pending else "completed"
    final_snapshot = commit(run_status)

    failure_reports = sorted(
        [report for report in results if report["status"] == "failed"],
        key=lambda report: report["sort_key"],
    )
    pending_reports = sorted(
        [report for report in results if report["status"] == "pending"],
        key=lambda report: report["sort_key"],
    )
    failure_entries = [_failure_entry(report) for report in failure_reports]
    pending_entries = list(plan.pending)

    unit_counts = {
        "cells_reused": sum(
            int(report.get("cache", {}).get("cells", {}).get("reused", 0))
            for report in results
        ),
        "cells_recomputed": sum(
            int(report.get("cache", {}).get("cells", {}).get("recomputed", 0))
            for report in results
        ),
        "points_reused": sum(
            int(report.get("cache", {}).get("points", {}).get("reused", 0))
            for report in results
        ),
        "points_recomputed": sum(
            int(report.get("cache", {}).get("points", {}).get("recomputed", 0))
            for report in results
        ),
        "pages_recomputed": sum(
            1
            for report in results
            if report.get("invalidations", {}).get("page_any_recomputed")
        ),
        "pages_reused": sum(
            1
            for report in results
            if report.get("status") == "ok"
            and not report.get("invalidations", {}).get("page_any_recomputed")
        ),
    }

    restored_pending = prior_manifest.get("pending_queue", []) if same_config else []
    recovery = {
        "used_snapshot": prior_snapshot is not None,
        "resumed_active": resume_active,
        "snapshot_version": prior_recovery_version,
        "recovery_point_pages": sorted(prior_completed_pages),
        "restored_completed_pages": baseline_completed,
        "restored_consumed_budget": int(prior_manifest.get("consumed_budget", 0))
        if same_config
        else 0,
        "restored_pending_queue": restored_pending,
        "config_matches": same_config,
        "input_signature_matches": same_input,
        "warnings": snapshot_warnings,
    }

    report = {
        "schema_version": "1.0",
        "status": run_status,
        "config_hash": config_hash,
        "input_signature": input_signature,
        "batch_id": batch_id,
        "page_order": all_files,
        "pages": results,
        "budget_ledger": {
            "batch_id": batch_id,
            **plan.as_dict(),
        },
        "pending_queue": pending_entries,
        "chart_quota": {
            "per_page_limit": config.page_chart_point_quota,
            "assignment_order": [
                {
                    "file": file_name,
                    "sort_key": f"{file_name}|chart",
                    "limit": config.page_chart_point_quota,
                }
                for file_name in all_files
            ],
            "truncated_pages": sorted(
                report_item["file"]
                for report_item in results
                if any(
                    warning.get("code") == "E_CHART_POINT_QUOTA_TRUNCATED"
                    for warning in report_item.get("warnings", [])
                )
            ),
        },
        "failures": failure_entries,
        "invalidation": {
            "summary": unit_counts,
            "entries": [
                {
                    "file": report_item["file"],
                    "sort_key": report_item["sort_key"],
                    "units": report_item["invalidations"]["units"],
                }
                for report_item in results
            ],
        },
        "recovery": recovery,
        "snapshot": {
            "generation": final_snapshot.generation,
            "version": final_snapshot.version,
            "status": run_status,
            "config_hash": config_hash,
            "completed_pages": sorted(completed_status_map),
            "consumed_budget": plan.consumed,
            "artifact_versions": _artifact_versions(current_artifacts),
        },
        "counts": {
            "pages": len(all_files),
            "ok": sum(1 for item in results if item["status"] == "ok"),
            "failed": len(failure_reports),
            "pending": len(pending_reports),
        },
    }

    destination = Path(report_path) if report_path is not None else state_root / "report.json"
    write_json_atomic(destination, report)
    return report
