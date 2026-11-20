"""Incremental versus empty-state full-rebuild equivalence verification."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from .config import ExperimentConfig, apply_config_edits, validate_experiment
from .engine import Engine, RunResult
from .errors import BusinessValidationError
from .report import write_json_report


def equivalence_projection(result: RunResult) -> dict[str, Any]:
    artifacts = result.artifacts
    effective_config = result.effective_config.as_dict() if result.effective_config is not None else None
    if effective_config is not None:
        # Worker count is an execution control, not a model artifact. Concurrency
        # consistency therefore compares the remaining effective configuration and
        # every computed field exactly.
        effective_config = dict(effective_config)
        effective_config["thread_pool_size"] = None
    return {
        "status": result.status,
        "effective_config": effective_config,
        "data": result.data_summary,
        "binning": artifacts.get("binning"),
        "trees": artifacts.get("trees"),
        "ensemble": artifacts.get("ensemble"),
        "contributions": artifacts.get("contributions"),
        "failures": result.failures,
    }


def compare_values(left: Any, right: Any, path: str = "$", limit: int = 50) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []

    def visit(first: Any, second: Any, current: str) -> None:
        if len(differences) >= limit:
            return
        if type(first) is not type(second):
            differences.append({"path": current, "left": first, "right": second, "kind": "type"})
            return
        if isinstance(first, dict):
            first_keys = set(first)
            second_keys = set(second)
            for key in sorted(first_keys - second_keys):
                differences.append({"path": f"{current}.{key}", "left": first[key], "right": "<missing>", "kind": "missing-right"})
            for key in sorted(second_keys - first_keys):
                differences.append({"path": f"{current}.{key}", "left": "<missing>", "right": second[key], "kind": "missing-left"})
            for key in sorted(first_keys & second_keys):
                visit(first[key], second[key], f"{current}.{key}")
            return
        if isinstance(first, list):
            if len(first) != len(second):
                differences.append(
                    {"path": current, "left": f"length={len(first)}", "right": f"length={len(second)}", "kind": "length"}
                )
            for index, (left_item, right_item) in enumerate(zip(first, second)):
                visit(left_item, right_item, f"{current}[{index}]")
            return
        if first != second:
            differences.append({"path": current, "left": first, "right": second, "kind": "value"})

    visit(left, right, path)
    return differences


def scenario_batches(
    batches: list[dict[str, Any]],
    through_batch: int | None = None,
) -> list[tuple[str, list[dict[str, Any]]]]:
    selected = list(enumerate(batches))
    if through_batch is not None:
        selected = [(index, batch) for index, batch in selected if index < through_batch]
    scenarios: list[tuple[str, list[dict[str, Any]]]] = [("initial", [])]
    cumulative: list[dict[str, Any]] = []
    for index, batch in selected:
        cumulative = cumulative + [batch]
        name = str(batch.get("name") or f"batch-{index + 1}")
        scenarios.append((name, list(cumulative)))
    return scenarios


def _safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return normalized or "scenario"


def run_verification(
    *,
    config_data: dict[str, Any],
    spec_data: dict[str, Any],
    batches: list[dict[str, Any]],
    work_dir: str | Path,
    through_batch: int | None = None,
    parallel_workers: int = 4,
    faults: dict[int, str] | None = None,
    tree_delay_ms: float = 0.0,
) -> tuple[dict[str, Any], int]:
    root = Path(work_dir) / f"verification-{uuid.uuid4().hex[:10]}"
    incremental_reports = root / "reports" / "incremental"
    full_reports = root / "reports" / "full"
    concurrency_reports = root / "reports" / "concurrency"
    tmp_dir = root / "tmp"
    root.mkdir(parents=True, exist_ok=False)

    incremental_engine = Engine(root / "incremental-state", tmp_dir=tmp_dir, report_dir=incremental_reports)
    scenarios = scenario_batches(batches, through_batch)
    scenario_results: list[dict[str, Any]] = []
    all_equal = True

    for index, (name, cumulative) in enumerate(scenarios):
        safe = _safe_name(name)
        incremental = incremental_engine.run(
            config_data,
            spec_data,
            cumulative,
            label=f"incremental:{name}",
            report_name=f"incremental-{index:02d}-{safe}.json",
            faults=faults,
            tree_delay_ms=tree_delay_ms,
        )
        if incremental.status == "failed":
            all_equal = False
            scenario_results.append(
                {
                    "index": index,
                    "name": name,
                    "equal": False,
                    "incremental_status": incremental.status,
                    "incremental_failures": incremental.failures,
                    "incremental_report": incremental.report_path,
                }
            )
            break
        if incremental.status == "interrupted":
            all_equal = False
            scenario_results.append(
                {
                    "index": index,
                    "name": name,
                    "equal": False,
                    "incremental_status": "interrupted",
                    "incremental_report": incremental.report_path,
                }
            )
            break

        full_engine = Engine(
            root / f"full-state-{index:02d}",
            tmp_dir=tmp_dir,
            report_dir=full_reports,
        )
        full = full_engine.run(
            config_data,
            spec_data,
            cumulative,
            label=f"full:{name}",
            report_name=f"full-{index:02d}-{safe}.json",
            faults=faults,
            tree_delay_ms=tree_delay_ms,
        )
        left = equivalence_projection(incremental)
        right = equivalence_projection(full)
        differences = compare_values(left, right)
        equal = full.status == "completed" and not differences
        all_equal = all_equal and equal
        scenario_results.append(
            {
                "index": index,
                "name": name,
                "equal": equal,
                "incremental_status": incremental.status,
                "full_status": full.status,
                "difference_count": len(differences),
                "differences": differences,
                "incremental_report": incremental.report_path,
                "full_report": full.report_path,
                "incremental_cache_audit": incremental.cache_audit,
                "full_cache_audit": full.cache_audit,
            }
        )
        if full.status != "completed":
            break

    final_cumulative = scenarios[-1][1] if scenarios else []
    config, failures = validate_experiment(config_data, spec_data, final_cumulative)
    concurrency_result: dict[str, Any] | None = None
    if failures:
        all_equal = False
        concurrency_result = {"equal": False, "failures": [failure.as_dict() for failure in failures]}
    elif config is not None:
        try:
            effective = apply_config_edits(config, final_cumulative)
        except BusinessValidationError as exception:
            all_equal = False
            concurrency_result = {"equal": False, "failures": exception.failures}
        else:
            workers_b = max(2, int(parallel_workers))
            engine_a = Engine(root / "concurrency-state-1", tmp_dir=tmp_dir, report_dir=concurrency_reports)
            engine_b = Engine(root / "concurrency-state-b", tmp_dir=tmp_dir, report_dir=concurrency_reports)
            run_a = engine_a.run(
                config_data,
                spec_data,
                final_cumulative,
                label="concurrency:1",
                report_name="concurrency-1.json",
                concurrency_override=1,
                faults=faults,
            )
            run_b = engine_b.run(
                config_data,
                spec_data,
                final_cumulative,
                label=f"concurrency:{workers_b}",
                report_name=f"concurrency-{workers_b}.json",
                concurrency_override=workers_b,
                faults=faults,
            )
            differences = compare_values(equivalence_projection(run_a), equivalence_projection(run_b))
            concurrency_equal = (
                run_a.status == "completed"
                and run_b.status == "completed"
                and not differences
            )
            all_equal = all_equal and concurrency_equal
            concurrency_result = {
                "equal": concurrency_equal,
                "workers_a": 1,
                "workers_b": workers_b,
                "difference_count": len(differences),
                "differences": differences,
                "report_a": run_a.report_path,
                "report_b": run_b.report_path,
                "metrics": run_a.artifacts.get("ensemble", {}).get("metrics"),
            }

    verification = {
        "schema": "equivalence-verification/v1",
        "root": str(root),
        "all_equal": all_equal,
        "scenario_count": len(scenario_results),
        "scenarios": scenario_results,
        "concurrency": concurrency_result,
    }
    write_json_report(root / "verification.json", verification)
    return verification, 0 if all_equal else 1
