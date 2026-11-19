"""Concurrent, deterministic orchestration of the complete diagnostic pipeline."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .boundary import diagnose_boundaries
from .cache import ContentCache
from .config import REDUCTION_PATHS, ExperimentConfig, ScenarioConfig
from .crossing import build_design, cross_parameter_fingerprint
from .data import generate_table
from .errors import E_SNAPSHOT_STALE, Failure
from .fingerprint import cache_key, canonical_json, fingerprint, sha256_text
from .metrics import compare_precision, compute_metric
from .reductions import reduce_path, reduction_parameter_fingerprint, standardize
from .report import build_report, write_reports
from .serialization import atomic_write_text, dump_json
from .snapshot import SNAPSHOT_STAGES, SnapshotManager


@dataclass
class RunResult:
    report: dict[str, Any] | None
    snapshot_info: dict[str, Any]
    report_paths: dict[str, str]
    partial: bool = False
    stopped_after_stage: str | None = None


def _dtype(precision: str) -> np.dtype:
    return np.dtype(precision)


def _payload_hash(payload: dict[str, Any]) -> str:
    return sha256_text(dump_json(payload))


def _effective_workers(experiment: ExperimentConfig, requested: int | None) -> int:
    budget = experiment.max_workers
    if requested is not None:
        if requested <= 0:
            raise ValueError("max_workers override must be positive")
        budget = min(budget, int(requested))
    for scenario in experiment.scenarios:
        budget = min(budget, scenario.max_workers)
    return max(1, int(budget))


def _run_jobs(
    jobs: Iterable[dict[str, Any]],
    worker: Callable[[dict[str, Any]], Any],
    workers: int,
) -> list[Any]:
    ordered_jobs = list(jobs)
    if not ordered_jobs:
        return []
    effective = max(1, min(int(workers), len(ordered_jobs)))
    if effective == 1:
        return [worker(job) for job in ordered_jobs]
    with ThreadPoolExecutor(max_workers=effective) as executor:
        futures = [executor.submit(worker, job) for job in ordered_jobs]
        return [future.result() for future in futures]


def _artifact_record(
    *,
    key: str,
    kind: str,
    lineage: str,
    sort_key: tuple[Any, ...],
    upstream_fingerprint: str,
    parameter_fingerprint: str,
    precision: str,
    scenario_id: str,
    sample_size: int,
    order_limit: int,
    path: str | None,
    payload: dict[str, Any],
    consumer: str,
) -> dict[str, Any]:
    return {
        "cache_key": key,
        "artifact_kind": kind,
        "lineage": lineage,
        "sort_key": list(sort_key),
        "upstream_fingerprint": upstream_fingerprint,
        "parameter_fingerprint": parameter_fingerprint,
        "precision": precision,
        "scenario_id": scenario_id,
        "sample_size": int(sample_size),
        "order_limit": int(order_limit),
        "path": path,
        "payload_fingerprint": _payload_hash(payload),
        "consumers": [consumer],
    }


def _merge_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for record in records:
        key = record["cache_key"]
        if key not in merged:
            merged[key] = {**record, "consumers": sorted(set(record.get("consumers", [])))}
        else:
            merged[key]["consumers"] = sorted(
                set(merged[key].get("consumers", [])) | set(record.get("consumers", []))
            )
    return [merged[key] for key in sorted(merged)]


def _register_snapshot_records(cache: ContentCache, records: Iterable[dict[str, Any]]) -> None:
    for record in records:
        consumers = record.get("consumers") or ["snapshot"]
        for consumer in consumers:
            cache.register_snapshot_reuse(
                key=record["cache_key"],
                kind=record["artifact_kind"],
                lineage=record["lineage"],
                sort_key=tuple(record["sort_key"]),
                upstream_fingerprint=record["upstream_fingerprint"],
                parameter_fingerprint=record["parameter_fingerprint"],
                precision=record["precision"],
                scenario_id=record["scenario_id"],
                sample_size=record["sample_size"],
                order_limit=record["order_limit"],
                path=record.get("path"),
                payload_fingerprint=record["payload_fingerprint"],
                consumer=consumer,
            )


def _build_plan(
    experiment: ExperimentConfig,
    scenario_data: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for scenario in experiment.scenarios:
        for sample_size in scenario.sample_sizes:
            data = scenario_data.get(f"{scenario.scenario_id}|n{sample_size}")
            if data is None:
                continue
            ids = np.asarray(data["sample_ids"], dtype=object)
            ids_hash = fingerprint([str(value) for value in ids])
            for order_limit in scenario.cross_orders:
                for path in REDUCTION_PATHS:
                    stable_sort_key = f"{order_limit:04d}|{path}|{ids_hash}"
                    task_id = f"{scenario.scenario_id}|n{sample_size}|o{order_limit}|{path}"
                    tasks.append(
                        {
                            "task_id": task_id,
                            "scenario": scenario,
                            "scenario_id": scenario.scenario_id,
                            "sample_size": int(sample_size),
                            "order_limit": int(order_limit),
                            "path": path,
                            "sample_ids_fingerprint": ids_hash,
                            "stable_sort_key": stable_sort_key,
                        }
                    )
    tasks.sort(
        key=lambda item: (
            item["scenario_id"],
            item["sample_size"],
            item["order_limit"],
            REDUCTION_PATHS.index(item["path"]),
        )
    )
    return tasks


def _cross_dependencies(experiment: ExperimentConfig) -> dict[str, Any]:
    return {
        "schema": 1,
        "stage": "crossing",
        "config_hash": experiment.config_hash,
        "data_spec_hash": experiment.data_spec.content_hash,
    }


def _reduction_dependencies(
    experiment: ExperimentConfig,
    crossing_fingerprint: str,
) -> dict[str, Any]:
    return {
        "schema": 1,
        "stage": "reductions",
        "config_hash": experiment.config_hash,
        "crossing_payload_fingerprint": crossing_fingerprint,
    }


def _metrics_dependencies(
    experiment: ExperimentConfig,
    reductions_fingerprint: str,
) -> dict[str, Any]:
    return {
        "schema": 1,
        "stage": "metrics",
        "config_hash": experiment.config_hash,
        "reductions_payload_fingerprint": reductions_fingerprint,
    }


def _diagnostics_dependencies(
    experiment: ExperimentConfig,
    metrics_fingerprint: str,
) -> dict[str, Any]:
    return {
        "schema": 1,
        "stage": "diagnostics",
        "config_hash": experiment.config_hash,
        "metrics_payload_fingerprint": metrics_fingerprint,
    }


def _crossing_stage(
    *,
    experiment: ExperimentConfig,
    snapshot: SnapshotManager,
    cache: ContentCache,
    workers: int,
) -> tuple[dict[str, Any], str]:
    dependencies = _cross_dependencies(experiment)
    loaded, load_info = snapshot.load_stage("crossing", dependencies)
    if loaded is not None:
        _register_snapshot_records(cache, loaded.get("artifact_records", []))
        return loaded, load_info["payload_fingerprint"]

    data_jobs = [
        {
            "scenario": scenario,
            "sample_size": sample_size,
        }
        for scenario in experiment.scenarios
        for sample_size in scenario.sample_sizes
    ]
    data_jobs.sort(key=lambda item: (item["scenario"].scenario_id, item["sample_size"]))

    def generate_job(job: dict[str, Any]) -> tuple[str, dict[str, Any], Any]:
        scenario: ScenarioConfig = job["scenario"]
        sample_size = int(job["sample_size"])
        table = generate_table(experiment.data_spec, scenario, sample_size)
        key = f"{scenario.scenario_id}|n{sample_size}"
        data = {
            "scenario_id": scenario.scenario_id,
            "sample_size": sample_size,
            "seed": scenario.seed,
            "target_column": scenario.target_column,
            "sample_ids": table.sample_ids.copy(),
            "target": table.target.copy(),
            "table_fingerprint": table.fingerprint,
            "missing_counts": table.missing_counts,
            "category_maps": table.category_maps,
        }
        return key, data, table

    generated = _run_jobs(data_jobs, generate_job, workers)
    scenario_data: dict[str, dict[str, Any]] = {}
    tables: dict[str, Any] = {}
    for key, data, table in generated:
        scenario_data[key] = data
        tables[key] = table

    design_jobs: list[dict[str, Any]] = []
    for scenario in experiment.scenarios:
        for sample_size in scenario.sample_sizes:
            table = tables[f"{scenario.scenario_id}|n{sample_size}"]
            for order_limit in scenario.cross_orders:
                for precision in scenario.precision_tiers:
                    design_jobs.append(
                        {
                            "scenario": scenario,
                            "sample_size": int(sample_size),
                            "order_limit": int(order_limit),
                            "precision": precision,
                            "table": table,
                        }
                    )
    design_jobs.sort(
        key=lambda item: (
            item["scenario"].scenario_id,
            item["sample_size"],
            item["order_limit"],
            item["precision"],
        )
    )

    def design_job(job: dict[str, Any]) -> dict[str, Any]:
        scenario: ScenarioConfig = job["scenario"]
        sample_size = int(job["sample_size"])
        order_limit = int(job["order_limit"])
        precision = job["precision"]
        table = job["table"]
        parameter_fp = cross_parameter_fingerprint(table, scenario, order_limit, precision)
        key = cache_key(
            kind="crossing",
            upstream_fingerprint=table.fingerprint,
            parameter_fingerprint=parameter_fp,
            precision=precision,
        )
        lineage = f"crossing|{scenario.scenario_id}|n{sample_size}|o{order_limit}|{precision}"
        sort_key = (0, scenario.scenario_id, sample_size, order_limit, "", precision, lineage)
        consumer = f"crossing:{scenario.scenario_id}|n{sample_size}|o{order_limit}|{precision}"

        def compute() -> dict[str, Any]:
            return build_design(table, scenario, order_limit, precision)

        design = cache.get_or_compute(
            key=key,
            kind="crossing",
            lineage=lineage,
            sort_key=sort_key,
            upstream_fingerprint=table.fingerprint,
            parameter_fingerprint=parameter_fp,
            precision=precision,
            scenario_id=scenario.scenario_id,
            sample_size=sample_size,
            order_limit=order_limit,
            path=None,
            consumer=consumer,
            compute=compute,
        )
        record = _artifact_record(
            key=key,
            kind="crossing",
            lineage=lineage,
            sort_key=sort_key,
            upstream_fingerprint=table.fingerprint,
            parameter_fingerprint=parameter_fp,
            precision=precision,
            scenario_id=scenario.scenario_id,
            sample_size=sample_size,
            order_limit=order_limit,
            path=None,
            payload=design,
            consumer=consumer,
        )
        return {
            "base_key": f"{scenario.scenario_id}|n{sample_size}|o{order_limit}|{precision}",
            "design": design,
            "record": record,
        }

    design_results = _run_jobs(design_jobs, design_job, workers)
    designs = {result["base_key"]: result["design"] for result in design_results}
    records = _merge_records(result["record"] for result in design_results)
    payload = {
        "schema_version": 1,
        "scenario_data": scenario_data,
        "designs": designs,
        "artifact_records": records,
    }
    pointer = snapshot.commit_stage("crossing", payload, dependencies)
    return payload, pointer["payload_fingerprint"]


def _reductions_stage(
    *,
    experiment: ExperimentConfig,
    snapshot: SnapshotManager,
    cache: ContentCache,
    workers: int,
    plan: list[dict[str, Any]],
    designs: dict[str, dict[str, Any]],
    crossing_fingerprint: str,
) -> tuple[dict[str, Any], str]:
    dependencies = _reduction_dependencies(experiment, crossing_fingerprint)
    loaded, load_info = snapshot.load_stage("reductions", dependencies)
    if loaded is not None:
        _register_snapshot_records(cache, loaded.get("artifact_records", []))
        return loaded, load_info["payload_fingerprint"]

    jobs: list[dict[str, Any]] = []
    for task in plan:
        for precision in task["scenario"].precision_tiers:
            jobs.append({**task, "precision": precision})
    jobs.sort(
        key=lambda item: (
            item["scenario_id"],
            item["sample_size"],
            item["order_limit"],
            REDUCTION_PATHS.index(item["path"]),
            item["precision"],
        )
    )

    def reduction_job(job: dict[str, Any]) -> dict[str, Any]:
        scenario: ScenarioConfig = job["scenario"]
        precision = job["precision"]
        sample_size = int(job["sample_size"])
        order_limit = int(job["order_limit"])
        path = job["path"]
        design_key = f"{scenario.scenario_id}|n{sample_size}|o{order_limit}|{precision}"
        design = designs[design_key]
        standard_lineage = f"standardization|{scenario.scenario_id}|n{sample_size}|o{order_limit}|{precision}"
        standard_parameter = fingerprint(
            {
                "kind": "standardization",
                "precision": precision,
                "ddof": 0,
                "upstream": design["fingerprint"],
            }
        )
        standard_key = cache_key(
            kind="standardization",
            upstream_fingerprint=design["fingerprint"],
            parameter_fingerprint=standard_parameter,
            precision=precision,
        )
        standard_sort = (1, scenario.scenario_id, sample_size, order_limit, "", precision, standard_lineage)
        standard = cache.get_or_compute(
            key=standard_key,
            kind="standardization",
            lineage=standard_lineage,
            sort_key=standard_sort,
            upstream_fingerprint=design["fingerprint"],
            parameter_fingerprint=standard_parameter,
            precision=precision,
            scenario_id=scenario.scenario_id,
            sample_size=sample_size,
            order_limit=order_limit,
            path=None,
            consumer=job["task_id"],
            compute=lambda: standardize(design, precision),
        )
        standard_record = _artifact_record(
            key=standard_key,
            kind="standardization",
            lineage=standard_lineage,
            sort_key=standard_sort,
            upstream_fingerprint=design["fingerprint"],
            parameter_fingerprint=standard_parameter,
            precision=precision,
            scenario_id=scenario.scenario_id,
            sample_size=sample_size,
            order_limit=order_limit,
            path=None,
            payload=standard,
            consumer=job["task_id"],
        )

        projection_scope = fingerprint(
            {
                "scenario_id": scenario.scenario_id,
                "data_spec_hash": experiment.data_spec.content_hash,
                "seed": scenario.seed,
            }
        )
        reduction_parameter = reduction_parameter_fingerprint(
            standard["fingerprint"],
            path,
            scenario.reductions[path],
            precision,
            projection_scope,
        )
        reduction_key = cache_key(
            kind="reduction",
            upstream_fingerprint=standard["fingerprint"],
            parameter_fingerprint=reduction_parameter,
            precision=precision,
        )
        reduction_lineage = (
            f"reduction|{scenario.scenario_id}|n{sample_size}|o{order_limit}|{path}|{precision}"
        )
        reduction_sort = (
            2,
            scenario.scenario_id,
            sample_size,
            order_limit,
            path,
            precision,
            reduction_lineage,
        )
        reduction = cache.get_or_compute(
            key=reduction_key,
            kind="reduction",
            lineage=reduction_lineage,
            sort_key=reduction_sort,
            upstream_fingerprint=standard["fingerprint"],
            parameter_fingerprint=reduction_parameter,
            precision=precision,
            scenario_id=scenario.scenario_id,
            sample_size=sample_size,
            order_limit=order_limit,
            path=path,
            consumer=job["task_id"],
            compute=lambda: reduce_path(
                standard,
                path,
                scenario.reductions[path],
                precision,
                projection_scope,
            ),
        )
        reduction_record = _artifact_record(
            key=reduction_key,
            kind="reduction",
            lineage=reduction_lineage,
            sort_key=reduction_sort,
            upstream_fingerprint=standard["fingerprint"],
            parameter_fingerprint=reduction_parameter,
            precision=precision,
            scenario_id=scenario.scenario_id,
            sample_size=sample_size,
            order_limit=order_limit,
            path=path,
            payload=reduction,
            consumer=job["task_id"],
        )
        return {
            "standard_key": f"{scenario.scenario_id}|n{sample_size}|o{order_limit}|{precision}",
            "reduction_key": (
                f"{scenario.scenario_id}|n{sample_size}|o{order_limit}|{path}|{precision}"
            ),
            "standard": standard,
            "reduction": reduction,
            "records": [standard_record, reduction_record],
        }

    results = _run_jobs(jobs, reduction_job, workers)
    standards = {result["standard_key"]: result["standard"] for result in results}
    reductions = {result["reduction_key"]: result["reduction"] for result in results}
    records = _merge_records(record for result in results for record in result["records"])
    payload = {
        "schema_version": 1,
        "standards": standards,
        "reductions": reductions,
        "artifact_records": records,
    }
    pointer = snapshot.commit_stage("reductions", payload, dependencies)
    return payload, pointer["payload_fingerprint"]


def _metrics_stage(
    *,
    experiment: ExperimentConfig,
    snapshot: SnapshotManager,
    cache: ContentCache,
    workers: int,
    plan: list[dict[str, Any]],
    designs: dict[str, dict[str, Any]],
    reductions_payload: dict[str, Any],
    reductions_fingerprint: str,
) -> tuple[dict[str, Any], str]:
    dependencies = _metrics_dependencies(experiment, reductions_fingerprint)
    loaded, load_info = snapshot.load_stage("metrics", dependencies)
    if loaded is not None:
        _register_snapshot_records(cache, loaded.get("artifact_records", []))
        return loaded, load_info["payload_fingerprint"]

    standards = reductions_payload["standards"]
    reductions = reductions_payload["reductions"]

    def metric_job(task: dict[str, Any]) -> dict[str, Any]:
        scenario: ScenarioConfig = task["scenario"]
        sample_size = int(task["sample_size"])
        order_limit = int(task["order_limit"])
        path = task["path"]
        metric_name = str(scenario.decision.get("metric", "first_coordinate"))
        metric_payloads: dict[str, dict[str, Any]] = {}
        metric_records: list[dict[str, Any]] = []
        for precision in scenario.precision_tiers:
            reduction_key = (
                f"{scenario.scenario_id}|n{sample_size}|o{order_limit}|{path}|{precision}"
            )
            reduction = reductions[reduction_key]
            parameter_fp = fingerprint(
                {
                    "kind": "metrics",
                    "metric": metric_name,
                    "precision": precision,
                    "upstream": reduction["fingerprint"],
                }
            )
            key = cache_key(
                kind="metrics",
                upstream_fingerprint=reduction["fingerprint"],
                parameter_fingerprint=parameter_fp,
                precision=precision,
            )
            lineage = (
                f"metrics|{scenario.scenario_id}|n{sample_size}|o{order_limit}|{path}|{precision}"
            )
            sort_key = (
                3,
                scenario.scenario_id,
                sample_size,
                order_limit,
                path,
                precision,
                lineage,
            )
            metric = cache.get_or_compute(
                key=key,
                kind="metrics",
                lineage=lineage,
                sort_key=sort_key,
                upstream_fingerprint=reduction["fingerprint"],
                parameter_fingerprint=parameter_fp,
                precision=precision,
                scenario_id=scenario.scenario_id,
                sample_size=sample_size,
                order_limit=order_limit,
                path=path,
                consumer=task["task_id"],
                compute=lambda reduction=reduction, precision=precision: compute_metric(
                    reduction,
                    metric_name,
                    precision,
                ),
            )
            metric_payloads[precision] = metric
            metric_records.append(
                _artifact_record(
                    key=key,
                    kind="metrics",
                    lineage=lineage,
                    sort_key=sort_key,
                    upstream_fingerprint=reduction["fingerprint"],
                    parameter_fingerprint=parameter_fp,
                    precision=precision,
                    scenario_id=scenario.scenario_id,
                    sample_size=sample_size,
                    order_limit=order_limit,
                    path=path,
                    payload=metric,
                    consumer=task["task_id"],
                )
            )

        design32 = designs[f"{scenario.scenario_id}|n{sample_size}|o{order_limit}|float32"]
        design64 = designs[f"{scenario.scenario_id}|n{sample_size}|o{order_limit}|float64"]
        standard32 = standards[f"{scenario.scenario_id}|n{sample_size}|o{order_limit}|float32"]
        standard64 = standards[f"{scenario.scenario_id}|n{sample_size}|o{order_limit}|float64"]
        reduction32 = reductions[
            f"{scenario.scenario_id}|n{sample_size}|o{order_limit}|{path}|float32"
        ]
        reduction64 = reductions[
            f"{scenario.scenario_id}|n{sample_size}|o{order_limit}|{path}|float64"
        ]
        comparison = compare_precision(
            scenario_id=scenario.scenario_id,
            sample_size=sample_size,
            order_limit=order_limit,
            path=path,
            sample_ids=np.asarray(design64["sample_ids"], dtype=object),
            design32=design32,
            design64=design64,
            standard32=standard32,
            standard64=standard64,
            reduction32=reduction32,
            reduction64=reduction64,
            metric32=metric_payloads["float32"],
            metric64=metric_payloads["float64"],
            decision_config=scenario.decision,
            tolerances=scenario.tolerances,
        )
        return {
            "task_id": task["task_id"],
            "metrics": metric_payloads,
            "comparison": comparison,
            "records": metric_records,
        }

    results = _run_jobs(plan, metric_job, workers)
    metrics: dict[str, dict[str, Any]] = {}
    comparisons: dict[str, dict[str, Any]] = {}
    precision_sensitive: list[dict[str, Any]] = []
    for result in results:
        for precision, metric in result["metrics"].items():
            metrics[f"{result['task_id']}|{precision}"] = metric
        comparisons[result["task_id"]] = result["comparison"]
        precision_sensitive.extend(result["comparison"]["precision_sensitive"])
    precision_sensitive.sort(
        key=lambda item: (
            item["scenario_id"],
            item["sample_size"],
            item["order_limit"],
            item["path"],
            item["sample_id"],
        )
    )
    records = _merge_records(record for result in results for record in result["records"])
    payload = {
        "schema_version": 1,
        "metrics": metrics,
        "comparisons": comparisons,
        "precision_sensitive": precision_sensitive,
        "artifact_records": records,
    }
    pointer = snapshot.commit_stage("metrics", payload, dependencies)
    return payload, pointer["payload_fingerprint"]


def _snapshot_failure(snapshot: SnapshotManager) -> list[Failure]:
    if snapshot.stale_event is None:
        return []
    event = snapshot.stale_event
    return [
        Failure(
            code=E_SNAPSHOT_STALE,
            scope="__snapshot__",
            field="snapshot",
            reason=str(event.get("reason")),
            message=str(event.get("message")),
            details={key: value for key, value in event.items() if key not in {"code", "reason", "message"}},
        )
    ]


def _write_partial_recovery(
    output_dir: str | Path,
    *,
    experiment: ExperimentConfig,
    snapshot: SnapshotManager,
    stage: str,
) -> str:
    path = Path(output_dir) / "recovery.json"
    payload = {
        "schema_version": 1,
        "status": "partial",
        "stopped_after_stage": stage,
        "config_hash": experiment.config_hash,
        "global_version": snapshot.global_version,
        "snapshot": snapshot.recovery_info(),
    }
    atomic_write_text(path, dump_json(payload, pretty=True))
    return str(path)


def run_experiment(
    experiment: ExperimentConfig,
    *,
    output_dir: str | Path,
    cache_dir: str | Path | None = None,
    snapshot_dir: str | Path | None = None,
    max_workers: int | None = None,
    stop_after_stage: str | None = None,
    cache_enabled: bool = True,
) -> RunResult:
    if stop_after_stage is not None and stop_after_stage not in SNAPSHOT_STAGES:
        raise ValueError(f"stop_after_stage must be one of {SNAPSHOT_STAGES}")
    workers = _effective_workers(experiment, max_workers)
    cache = ContentCache(cache_dir or experiment.cache_dir, enabled=cache_enabled)
    snapshot = SnapshotManager(
        snapshot_dir or experiment.snapshot_dir,
        global_version=f"{experiment.schema_version}:{experiment.version}",
        config_hash=experiment.config_hash,
        precision_fingerprints=experiment.precision_fingerprints,
        schema_version=experiment.schema_version,
    )

    crossing_payload, crossing_fingerprint = _crossing_stage(
        experiment=experiment,
        snapshot=snapshot,
        cache=cache,
        workers=workers,
    )
    if stop_after_stage == "crossing":
        recovery_path = _write_partial_recovery(output_dir, experiment=experiment, snapshot=snapshot, stage="crossing")
        return RunResult(
            report=None,
            snapshot_info=snapshot.recovery_info(),
            report_paths={"recovery": recovery_path},
            partial=True,
            stopped_after_stage="crossing",
        )

    scenario_data = crossing_payload["scenario_data"]
    designs = crossing_payload["designs"]
    plan = _build_plan(experiment, scenario_data)
    reductions_payload, reductions_fingerprint = _reductions_stage(
        experiment=experiment,
        snapshot=snapshot,
        cache=cache,
        workers=workers,
        plan=plan,
        designs=designs,
        crossing_fingerprint=crossing_fingerprint,
    )
    if stop_after_stage == "reductions":
        recovery_path = _write_partial_recovery(output_dir, experiment=experiment, snapshot=snapshot, stage="reductions")
        return RunResult(
            report=None,
            snapshot_info=snapshot.recovery_info(),
            report_paths={"recovery": recovery_path},
            partial=True,
            stopped_after_stage="reductions",
        )

    metrics_payload, metrics_fingerprint = _metrics_stage(
        experiment=experiment,
        snapshot=snapshot,
        cache=cache,
        workers=workers,
        plan=plan,
        designs=designs,
        reductions_payload=reductions_payload,
        reductions_fingerprint=reductions_fingerprint,
    )
    if stop_after_stage == "metrics":
        cache_ledger = cache.finalize()
        recovery_path = _write_partial_recovery(output_dir, experiment=experiment, snapshot=snapshot, stage="metrics")
        return RunResult(
            report=None,
            snapshot_info={**snapshot.recovery_info(), "partial_cache_ledger_summary": cache_ledger["summary"]},
            report_paths={"recovery": recovery_path},
            partial=True,
            stopped_after_stage="metrics",
        )

    diagnostics_dependencies = _diagnostics_dependencies(experiment, metrics_fingerprint)
    diagnostics_loaded, diagnostics_load_info = snapshot.load_stage("diagnostics", diagnostics_dependencies)
    if diagnostics_loaded is not None:
        _register_snapshot_records(cache, metrics_payload.get("artifact_records", []))
        boundaries = diagnostics_loaded["boundaries"]
        cache_ledger = cache.finalize(persist_state=False)
        failure_objects = [failure.as_dict() for failure in experiment.failures]
        failure_objects.extend(failure.as_dict() for failure in _snapshot_failure(snapshot))
        failure_objects.sort(
            key=lambda item: (item["scope"], item["field"], item["reason"], item["code"])
        )
        diagnostics_fingerprint = diagnostics_load_info["payload_fingerprint"]
    else:
        boundaries = diagnose_boundaries(
            designs=designs,
            reductions=reductions_payload["reductions"],
            comparisons=metrics_payload["comparisons"],
            neighborhood=experiment.boundary["neighborhood"],
            tie_tolerance=experiment.boundary["tie_tolerance"],
        )
        cache_ledger = cache.finalize()
        failure_objects = [failure.as_dict() for failure in experiment.failures]
        failure_objects.extend(failure.as_dict() for failure in _snapshot_failure(snapshot))
        failure_objects.sort(
            key=lambda item: (item["scope"], item["field"], item["reason"], item["code"])
        )
        diagnostics_payload = {
            "schema_version": 1,
            "boundaries": boundaries,
            "cache_ledger": cache_ledger,
            "failures": failure_objects,
        }
        pointer = snapshot.commit_stage("diagnostics", diagnostics_payload, diagnostics_dependencies)
        diagnostics_fingerprint = pointer["payload_fingerprint"]

    if stop_after_stage == "diagnostics":
        recovery_path = _write_partial_recovery(output_dir, experiment=experiment, snapshot=snapshot, stage="diagnostics")
        return RunResult(
            report=None,
            snapshot_info=snapshot.recovery_info(),
            report_paths={"recovery": recovery_path},
            partial=True,
            stopped_after_stage="diagnostics",
        )

    execution_info = {
        "max_workers_requested": experiment.max_workers if max_workers is None else min(experiment.max_workers, max_workers),
        "max_workers_effective": workers,
        "task_count": len(plan),
        "tasks": [
            {
                "task_id": task["task_id"],
                "order_limit": task["order_limit"],
                "path": task["path"],
                "sample_ids_fingerprint": task["sample_ids_fingerprint"],
                "stable_sort_key": task["stable_sort_key"],
            }
            for task in plan
        ],
    }
    report = build_report(
        experiment=experiment,
        plan=plan,
        designs=designs,
        reductions=reductions_payload["reductions"],
        metrics=metrics_payload["metrics"],
        comparisons=metrics_payload["comparisons"],
        precision_sensitive=metrics_payload["precision_sensitive"],
        boundaries=boundaries,
        cache_ledger=cache_ledger,
        failures=failure_objects,
        snapshot_info=snapshot.recovery_info(),
        execution_info=execution_info,
    )
    report_paths = write_reports(report, output_dir)
    return RunResult(
        report=report,
        snapshot_info=snapshot.recovery_info(),
        report_paths=report_paths,
    )
