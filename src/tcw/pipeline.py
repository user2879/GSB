"""End-to-end workbench orchestration."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import Candidate, ExperimentConfig, load_experiment_config
from .data import PreparedData, generate_dataset, prepare_dataset
from .errors import (
    E_CONFIG_INVALID,
    E_CONFIG_MISMATCH,
    E_EVALUATION_FAILED,
    E_MODEL_FAILED,
    E_NONDETERMINISTIC,
    BusinessError,
    Failure,
    sorted_failures,
)
from .evaluation import (
    Evaluation,
    attach_reproducibility,
    build_ranking,
    evaluate_fit,
    metrics_table,
)
from .injection import (
    InjectionConfig,
    InjectionContext,
    load_injection_config,
    run_injections,
)
from .models import FitResult, fit_candidate
from .preprocessing import PreprocessResult, fit_transform, verify_preprocess_result
from .snapshot import SnapshotStore
from .util import Span, atomic_write_json, max_abs_diff, sha256_value


@dataclass
class RunOutcome:
    report: dict[str, Any]
    exit_code: int
    failures: list[dict[str, Any]]


def _run_candidate_twice(
    matrix: np.ndarray,
    candidate: Candidate,
    seed: int,
) -> tuple[FitResult, FitResult]:
    first = fit_candidate(matrix, candidate, seed)
    second = fit_candidate(matrix, candidate, seed)
    return first, second


def _model_failure(candidate: Candidate, exc: Exception) -> Failure:
    if isinstance(exc, BusinessError):
        return Failure(
            code=exc.code,
            message=exc.failure.message,
            stage="clustering",
            candidate_key=candidate.sort_key,
            path=exc.failure.path,
            extra=exc.failure.extra,
        )
    return Failure(
        code=E_MODEL_FAILED,
        message=f"{type(exc).__name__}: {exc}",
        stage="clustering",
        candidate_key=candidate.sort_key,
    )


def _clustering_payload(
    preprocess_result: PreprocessResult,
    runs: list[dict[str, Any]],
    failures: list[Failure],
) -> dict[str, Any]:
    return {
        "preprocess_fingerprint": preprocess_result.fingerprint,
        "runs": sorted(runs, key=lambda entry: entry["candidate_sort_key"]),
        "failures": [failure.to_dict() for failure in sorted(failures, key=Failure.sort_key)],
    }


def _execute_clustering(
    config: ExperimentConfig,
    preprocess_result: PreprocessResult,
    snapshot: SnapshotStore,
    restored_payload: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[Failure], bool]:
    if restored_payload is not None:
        if restored_payload.get("preprocess_fingerprint") != preprocess_result.fingerprint:
            raise BusinessError(
                E_CONFIG_MISMATCH,
                "restored clustering snapshot does not match preprocessing fingerprint",
                stage="snapshot",
                fatal=True,
            )
        runs = restored_payload.get("runs", [])
        expected = [candidate.sort_key for candidate in config.candidates]
        restored_failures = [
            _failure_from_dict(entry)
            for entry in restored_payload.get("failures", [])
        ]
        present = {entry.get("candidate_sort_key") for entry in runs}
        present.update(
            failure.candidate_key
            for failure in restored_failures
            if failure.candidate_key is not None
        )
        if sorted(present) != expected:
            raise BusinessError(
                E_CONFIG_MISMATCH,
                "restored clustering candidate set does not match configuration",
                stage="snapshot",
                fatal=True,
            )
        return runs, restored_failures, True

    matrix = np.asarray(preprocess_result.matrix, dtype=float)
    interval = config.execution["snapshot_commit_interval"]
    concurrency = config.execution["concurrency"]
    runs: list[dict[str, Any]] = []
    failures: list[Failure] = []
    groups = [
        config.candidates[index : index + interval]
        for index in range(0, len(config.candidates), interval)
    ]
    for sequence, group in enumerate(groups, start=1):
        group_results: list[dict[str, Any]] = []
        group_failures: list[Failure] = []
        workers = max(1, min(concurrency, len(group)))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="candidate") as executor:
            futures = {
                executor.submit(_run_candidate_twice, matrix, candidate, config.seed): candidate
                for candidate in group
            }
            for future, candidate in futures.items():
                try:
                    first, second = future.result()
                except Exception as exc:
                    group_failures.append(_model_failure(candidate, exc))
                    continue
                group_results.append(
                    {
                        "candidate_sort_key": candidate.sort_key,
                        "method": candidate.method,
                        "params": candidate.params,
                        "first": first.to_dict(),
                        "second": second.to_dict(),
                    }
                )
        group_results.sort(key=lambda entry: entry["candidate_sort_key"])
        runs.extend(group_results)
        failures.extend(group_failures)
        checkpoint_payload = _clustering_payload(preprocess_result, runs, failures)
        snapshot.commit_checkpoint("clustering", sequence, checkpoint_payload)
    return sorted(runs, key=lambda entry: entry["candidate_sort_key"]), failures, False


def _execute_evaluation(
    config: ExperimentConfig,
    preprocess_result: PreprocessResult,
    runs: list[dict[str, Any]],
    snapshot: SnapshotStore,
    restored_payload: dict[str, Any] | None,
) -> tuple[list[Evaluation], list[Failure], bool]:
    if restored_payload is not None:
        if restored_payload.get("preprocess_fingerprint") != preprocess_result.fingerprint:
            raise BusinessError(
                E_CONFIG_MISMATCH,
                "restored evaluation snapshot does not match preprocessing fingerprint",
                stage="snapshot",
                fatal=True,
            )
        evaluations = [
            Evaluation.from_dict(entry)
            for entry in restored_payload.get("evaluations", [])
        ]
        expected = [candidate.sort_key for candidate in config.candidates]
        restored_failures = [
            _failure_from_dict(entry)
            for entry in restored_payload.get("failures", [])
        ]
        present = {evaluation.candidate.sort_key for evaluation in evaluations}
        present.update(
            failure.candidate_key
            for failure in restored_failures
            if failure.candidate_key is not None
        )
        if sorted(present) != expected:
            raise BusinessError(
                E_CONFIG_MISMATCH,
                "restored evaluation candidate set does not match configuration",
                stage="snapshot",
                fatal=True,
            )
        return evaluations, restored_failures, True

    matrix = np.asarray(preprocess_result.matrix, dtype=float)
    candidate_map = config.candidate_by_key()
    evaluations: list[Evaluation] = []
    failures: list[Failure] = []
    for entry in sorted(runs, key=lambda item: item["candidate_sort_key"]):
        candidate = candidate_map[entry["candidate_sort_key"]]
        try:
            first_fit = FitResult.from_dict(entry["first"])
            second_fit = FitResult.from_dict(entry["second"])
            first_evaluation = evaluate_fit(candidate, first_fit, matrix)
            evaluation = evaluate_fit(candidate, second_fit, matrix)
            attach_reproducibility(evaluation, first_evaluation)
            if not evaluation.reproducibility["deterministic"]:
                failures.append(
                    Failure(
                        code=E_NONDETERMINISTIC,
                        message="independent candidate runs produced different assignments or metrics",
                        stage="evaluation",
                        candidate_key=candidate.sort_key,
                        extra={"reproducibility": evaluation.reproducibility},
                    )
                )
            evaluations.append(evaluation)
        except BusinessError as exc:
            failures.append(
                Failure(
                    code=E_EVALUATION_FAILED,
                    message=exc.failure.message,
                    stage="evaluation",
                    candidate_key=candidate.sort_key,
                    path=exc.failure.path,
                    extra=exc.failure.extra,
                )
            )
        except Exception as exc:
            failures.append(
                Failure(
                    code=E_EVALUATION_FAILED,
                    message=f"{type(exc).__name__}: {exc}",
                    stage="evaluation",
                    candidate_key=candidate.sort_key,
                )
            )
    build_ranking(evaluations, config.evaluation)
    return evaluations, failures, False


def _failure_from_dict(entry: dict[str, Any]) -> Failure:
    return Failure(
        code=entry["code"],
        message=entry["message"],
        stage=entry.get("stage", "unknown"),
        injection_id=entry.get("injection_id"),
        candidate_key=entry.get("candidate_key"),
        path=entry.get("path"),
        fatal=bool(entry.get("fatal", False)),
        extra=entry.get("extra", {}),
    )


def execute_experiment(
    config: ExperimentConfig,
    injection_config: InjectionConfig,
    snapshot_dir: Path,
    report_path: Path,
    *,
    resume: bool = False,
) -> RunOutcome:
    failures: list[Failure] = list(config.failures)
    timings: dict[str, float] = {}
    restored_stages: list[str] = []
    computed_stages: list[str] = []
    preprocess_reproducibility: dict[str, Any] = {}
    total_span = Span()

    with total_span:
        with Span() as data_span:
            if not config.candidates:
                raise BusinessError(
                    E_CONFIG_INVALID,
                    "no valid clustering candidates remain after configuration validation",
                    stage="config",
                    fatal=True,
                )
            dataset = generate_dataset(config.description, seed=config.seed)
            prepared = prepare_dataset(dataset, config.preprocessing["fit_fraction"])
        timings["data"] = data_span.elapsed_ms

        snapshot = SnapshotStore(
            snapshot_dir,
            config.execution["snapshot_version"],
            config.config_hash,
            injection_config.hash,
            prepared.data_hash,
        )
        with Span() as snapshot_span:
            if resume:
                snapshot.verify_manifest()
            else:
                snapshot.initialize()
        timings["snapshot"] = snapshot_span.elapsed_ms

        with Span() as preprocess_span:
            restored_preprocess = snapshot.restore_stage("preprocess") if resume else None
            if restored_preprocess is not None:
                preprocess_result = PreprocessResult.from_dict(restored_preprocess["preprocess"])
                preprocess_reproducibility = dict(restored_preprocess.get("reproducibility", {}))
                if preprocess_result.source_data_hash != prepared.data_hash:
                    raise BusinessError(
                        E_CONFIG_MISMATCH,
                        "restored preprocessing snapshot does not match the current dataset",
                        stage="snapshot",
                        fatal=True,
                    )
                if not verify_preprocess_result(preprocess_result):
                    raise BusinessError(
                        E_CONFIG_MISMATCH,
                        "restored preprocessing payload failed content verification",
                        stage="snapshot",
                        fatal=True,
                    )
                if not preprocess_reproducibility.get("deterministic", False):
                    failures.append(
                        Failure(
                            code=E_NONDETERMINISTIC,
                            message="restored preprocessing snapshot records a non-deterministic run",
                            stage="preprocess",
                            extra={"reproducibility": preprocess_reproducibility},
                        )
                    )
                restored_stages.append("preprocess")
            else:
                first = fit_transform(prepared, config.preprocessing)
                second = fit_transform(prepared, config.preprocessing)
                flattened_first = [value for row in first.matrix for value in row]
                flattened_second = [value for row in second.matrix for value in row]
                matrix_diff = max_abs_diff(flattened_first, flattened_second)
                preprocess_reproducibility = {
                    "second_run_is_independent": True,
                    "matrix_max_abs_diff": matrix_diff,
                    "fingerprints_equal": first.fingerprint == second.fingerprint,
                    "output_hashes_equal": first.output_hash == second.output_hash,
                    "deterministic": matrix_diff <= 0.0
                    and first.fingerprint == second.fingerprint,
                }
                if not preprocess_reproducibility["deterministic"]:
                    failures.append(
                        Failure(
                            code=E_NONDETERMINISTIC,
                            message="two preprocessing runs were not elementwise identical",
                            stage="preprocess",
                            extra={"reproducibility": preprocess_reproducibility},
                        )
                    )
                preprocess_result = first
                snapshot_payload = {
                    "preprocess": preprocess_result.to_dict(),
                    "reproducibility": preprocess_reproducibility,
                }
                snapshot.commit_stage("preprocess", snapshot_payload)
                computed_stages.append("preprocess")
        timings["preprocess"] = preprocess_span.elapsed_ms

        with Span() as clustering_span:
            restored_clustering = snapshot.restore_stage("clustering") if resume else None
            runs, model_failures, clustering_restored = _execute_clustering(
                config, preprocess_result, snapshot, restored_clustering
            )
            failures.extend(model_failures)
            if clustering_restored:
                restored_stages.append("clustering")
            else:
                clustering_payload = _clustering_payload(preprocess_result, runs, model_failures)
                snapshot.commit_stage("clustering", clustering_payload)
                computed_stages.append("clustering")
        timings["clustering"] = clustering_span.elapsed_ms

        with Span() as evaluation_span:
            restored_evaluation = snapshot.restore_stage("evaluation") if resume else None
            evaluations, evaluation_failures, evaluation_restored = _execute_evaluation(
                config, preprocess_result, runs, snapshot, restored_evaluation
            )
            failures.extend(evaluation_failures)
            if evaluation_restored:
                restored_stages.append("evaluation")
            else:
                evaluation_payload = {
                    "preprocess_fingerprint": preprocess_result.fingerprint,
                    "evaluations": [evaluation.to_dict() for evaluation in evaluations],
                    "metrics_table": metrics_table(evaluations),
                    "failures": [failure.to_dict() for failure in sorted(evaluation_failures, key=Failure.sort_key)],
                }
                snapshot.commit_stage("evaluation", evaluation_payload)
                computed_stages.append("evaluation")
        timings["evaluation"] = evaluation_span.elapsed_ms

        with Span() as injection_span:
            context = InjectionContext(
                dataset=dataset,
                prepared=prepared,
                matrix=np.asarray(preprocess_result.matrix, dtype=float),
                candidates=config.candidates,
            )
            injection_hits, injection_failures = run_injections(injection_config, context)
            failures.extend(injection_failures)
        timings["injection"] = injection_span.elapsed_ms

    timings["total"] = total_span.elapsed_ms
    manifest_summary = snapshot.manifest_summary()
    ranked = sorted(evaluations, key=lambda evaluation: (evaluation.rank or 0))

    candidate_ranking = [
        {
            "rank": evaluation.rank,
            "sort_key": evaluation.candidate.sort_key,
            "method": evaluation.candidate.method,
            "params": evaluation.candidate.params,
            "rank_score": evaluation.rank_score,
            "metrics": evaluation.metrics,
            "diagnostics": evaluation.diagnostics,
            "reproducibility": evaluation.reproducibility,
            "assignments": {
                "sample_ids": prepared.row_ids,
                "labels": evaluation.fit.labels,
                "responsibilities": evaluation.fit.responsibilities,
            },
            "model": {
                "n_iter": evaluation.fit.n_iter,
                "converged": evaluation.fit.converged,
                "centers": evaluation.fit.centers,
                "seed": evaluation.fit.seed,
            },
        }
        for evaluation in ranked
    ]
    diagnostics = {
        "candidates_with_empty_clusters": [
            {
                "sort_key": evaluation.candidate.sort_key,
                "method": evaluation.candidate.method,
                "params": evaluation.candidate.params,
                "empty_clusters": evaluation.diagnostics.get("empty_clusters", []),
            }
            for evaluation in ranked
            if evaluation.diagnostics.get("empty_clusters")
        ],
        "candidates_with_all_noise": [
            {
                "sort_key": evaluation.candidate.sort_key,
                "method": evaluation.candidate.method,
                "params": evaluation.candidate.params,
                "noise_count": evaluation.diagnostics.get("noise_count", 0),
            }
            for evaluation in ranked
            if evaluation.diagnostics.get("all_noise")
        ],
        "candidate_diagnostic_count": len(ranked),
        "metrics_unavailable": [
            {
                "sort_key": evaluation.candidate.sort_key,
                "metrics": {
                    metric: detail
                    for metric, detail in evaluation.metric_diagnostics.items()
                    if detail.get("status") != "ok"
                },
            }
            for evaluation in ranked
            if any(
                detail.get("status") != "ok"
                for detail in evaluation.metric_diagnostics.values()
            )
        ],
    }

    failure_records = sorted_failures(failures)
    exit_code = 0 if not failure_records else 1
    budget = config.execution["interaction_latency_budget_ms"]
    budget_check = {
        "budget_ms": budget,
        "stages": {
            stage: {
                "elapsed_ms": timings.get(stage),
                "within_budget": timings.get(stage, math.inf) <= budget,
            }
            for stage in ("preprocess", "clustering", "evaluation")
        },
        "exceeded_stages": [
            stage
            for stage in ("preprocess", "clustering", "evaluation")
            if timings.get(stage, math.inf) > budget
        ],
    }
    report = {
        "schema_version": 1,
        "experiment": config.name,
        "status": "ok" if exit_code == 0 else "completed_with_failures",
        "exit_code": exit_code,
        "config_hash": config.config_hash,
        "config_path": str(config.path),
        "data_description_path": str(config.description_path),
        "data_description_hash": config.description.content_hash,
        "injection_hash": injection_config.hash,
        "injection_path": str(injection_config.path) if injection_config.path else None,
        "snapshot_version": config.execution["snapshot_version"],
        "snapshot": manifest_summary,
        "execution": {
            "seed": config.seed,
            "seed_source": config.seed_source,
            "concurrency": config.execution["concurrency"],
            "candidate_sort_order": [candidate.sort_key for candidate in config.candidates],
            "snapshot_commit_interval": config.execution["snapshot_commit_interval"],
            "restored_stages": restored_stages,
            "computed_stages": computed_stages,
        },
        "dataset": dataset.to_summary(),
        "preprocessing": {
            "method": preprocess_result.method,
            "fingerprint": preprocess_result.fingerprint,
            "input_hash": preprocess_result.input_hash,
            "output_hash": preprocess_result.output_hash,
            "rows": preprocess_result.rows,
            "columns": preprocess_result.columns,
            "train_count": preprocess_result.train_count,
            "fill_statistics": {
                "numeric": prepared.numeric_fill_values,
                "categorical": prepared.categorical_fill_values,
            },
            "statistics": preprocess_result.stats,
            "reproducibility": preprocess_reproducibility,
        },
        "candidate_ranking": candidate_ranking,
        "metrics_table": metrics_table(evaluations),
        "diagnostics": diagnostics,
        "injection_hits": injection_hits,
        "failure_list": failure_records,
        "business_codes": sorted({entry["code"] for entry in failure_records}),
        "stage_timings_ms": timings,
        "interaction_latency": budget_check,
    }
    atomic_write_json(report_path, report)
    return RunOutcome(report=report, exit_code=exit_code, failures=failure_records)


def load_and_execute(
    config_path: Path,
    *,
    data_description: Path | None = None,
    injection_path: Path | None = None,
    snapshot_dir: Path = Path("artifacts/snapshots"),
    report_path: Path = Path("artifacts/runs/diagnostic_report.json"),
    resume: bool = False,
    snapshot_version: int | None = None,
) -> RunOutcome:
    config = load_experiment_config(config_path, data_description)
    if snapshot_version is not None:
        config.execution["snapshot_version"] = int(snapshot_version)
    injection = load_injection_config(injection_path)
    return execute_experiment(
        config,
        injection,
        snapshot_dir,
        report_path,
        resume=resume,
    )


def build_fatal_report(
    error: BusinessError,
    report_path: Path,
    *,
    config_path: Path | None = None,
    injection_path: Path | None = None,
) -> RunOutcome:
    failure_records = sorted_failures([error])
    report = {
        "schema_version": 1,
        "status": "failed",
        "exit_code": 2,
        "config_path": str(config_path) if config_path else None,
        "injection_path": str(injection_path) if injection_path else None,
        "failure_list": failure_records,
        "business_codes": [error.code],
    }
    atomic_write_json(report_path, report)
    return RunOutcome(report=report, exit_code=2, failures=failure_records)
