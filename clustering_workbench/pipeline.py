"""End-to-end deterministic clustering diagnostics pipeline."""

from __future__ import annotations

import io
import math
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .artifacts import ArtifactStore, preprocess_from_payload, preprocess_payload
from .cache import ResidentBlockCache
from .codes import Failure, code_catalog, sort_failures
from .config import config_fingerprint, load_config
from .data import GeneratedDataset, generate_dataset
from .injections import InjectionManager
from .methods import (
    CandidateSpec,
    MethodExecution,
    build_candidate_specs,
    execute_method,
    validate_candidate_params,
)
from .metrics import (
    EvaluationResult,
    evaluate_candidate,
    evaluation_identity,
    rank_candidates,
    ranking_table,
)
from .preprocess import (
    PreprocessResult,
    preprocess_dataset,
    preprocess_identity,
    preprocess_artifact_fingerprint,
    validate_preprocess_columns,
)
from .snapshots import SnapshotStore
from .utils import (
    atomic_write_json,
    canonical_json,
    hash_array,
    hash_object,
    json_safe,
    sha256_text,
    truncate_text,
)


def _npz_bytes(**arrays: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def _execution_to_record(
    identity_fingerprint: str,
    execution: MethodExecution,
) -> dict[str, Any]:
    artifact_fingerprint = hash_object(
        "clusterart",
        {
            "identity": identity_fingerprint,
            "label_hash": execution.label_fingerprint,
            "status": execution.status,
            "iterations_executed": execution.iterations_executed,
            "trajectory": execution.trajectory,
            "diagnostics": execution.diagnostics,
            "flags": execution.diagnostic_flags,
            "error_code": execution.error_code,
        },
    )
    return json_safe(
        {
            "identity_fingerprint": identity_fingerprint,
            "artifact_fingerprint": artifact_fingerprint,
            "labels": execution.labels,
            "status": execution.status,
            "iterations_executed": execution.iterations_executed,
            "iterations_requested": execution.iterations_requested,
            "diagnostics": execution.diagnostics,
            "trajectory": execution.trajectory,
            "loglik_per_sample": execution.loglik_per_sample,
            "diagnostic_flags": execution.diagnostic_flags,
            "label_fingerprint": execution.label_fingerprint,
            "error": execution.error,
            "error_code": execution.error_code,
        }
    )


def _execution_from_record(record: dict[str, Any]) -> MethodExecution:
    loglik = record.get("loglik_per_sample")
    return MethodExecution(
        labels=np.asarray(record.get("labels", []), dtype=np.int64),
        status=str(record.get("status", "error")),
        iterations_executed=int(record.get("iterations_executed", 0)),
        iterations_requested=int(record.get("iterations_requested", 0)),
        diagnostics=dict(record.get("diagnostics", {})),
        trajectory=[float(value) for value in record.get("trajectory", [])],
        loglik_per_sample=None if loglik is None else np.asarray(loglik, dtype=np.float64),
        diagnostic_flags=[str(value) for value in record.get("diagnostic_flags", [])],
        error=record.get("error"),
        error_code=record.get("error_code"),
    )


def _evaluation_record(
    identity_fingerprint: str,
    result: EvaluationResult,
) -> dict[str, Any]:
    return {
        "identity_fingerprint": identity_fingerprint,
        "artifact_fingerprint": result.artifact_fingerprint,
        "label_hash": result.label_fingerprint,
        "evaluation": result.to_dict(),
    }


def _evaluation_from_record(record: dict[str, Any]) -> dict[str, Any]:
    return dict(record.get("evaluation", {}))


class WorkbenchPipeline:
    def __init__(
        self,
        config_path: str | Path,
        *,
        overrides: dict[str, Any] | None = None,
        output_dir: str | Path | None = None,
        algorithm_runner: Callable[[str, np.ndarray, dict[str, Any], int], MethodExecution] | None = None,
    ):
        self.config_path = Path(config_path)
        self.overrides = overrides or {}
        self.output_override = Path(output_dir) if output_dir else None
        self.algorithm_runner = algorithm_runner

    def run(self) -> dict[str, Any]:
        started = time.perf_counter()
        started_at = datetime.now(timezone.utc).isoformat()
        config, resolved_config_path, config_failures = load_config(self.config_path, self.overrides)
        output_root = self.output_override or Path(config["paths"]["outputs"])
        output_root = output_root.expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        failures: list[Failure] = list(config_failures)
        injection_manager = InjectionManager(config.get("injections", []))
        config_hash = config_fingerprint(config)
        injection_hash = injection_manager.fingerprint
        snapshot_version = int(config.get("snapshot", {}).get("version", 1))
        snapshot = SnapshotStore(
            output_root,
            snapshot_version,
            config_hash,
            injection_hash,
            enabled=bool(config.get("snapshot", {}).get("enabled", True)),
        )
        snapshot_status = snapshot.validate()
        if snapshot.invalidated_manifest is not None:
            failures.append(
                Failure(
                    code="SNAPSHOT_VERSION_INVALID",
                    stage="snapshot",
                    item="active",
                    message="Snapshot identity changed; old state was archived and not resumed.",
                    details={
                        "reason": snapshot.reason,
                        "previous": {
                            "global_version": snapshot.invalidated_manifest.get("global_version"),
                            "config_hash": snapshot.invalidated_manifest.get("config_hash"),
                            "injection_hash": snapshot.invalidated_manifest.get("injection_hash"),
                        },
                        "current": {
                            "global_version": snapshot_version,
                            "config_hash": config_hash,
                            "injection_hash": injection_hash,
                        },
                    },
                )
            )

        dataset, dataset_failures = generate_dataset(config, resolved_config_path, injection_manager)
        failures.extend(dataset_failures)
        artifact_store = ArtifactStore(output_root)
        previous_ledger = artifact_store.load_ledger()
        ledger_entries: dict[str, dict[str, Any]] = {}
        run_counters = {
            "preprocess_compute_calls": 0,
            "cluster_compute_calls": 0,
            "evaluation_compute_calls": 0,
            "snapshot_resume_calls": 0,
            "artifact_reuse_calls": 0,
        }

        report: dict[str, Any] = {
            "schema_version": "1.0",
            "run": {
                "started_at": started_at,
                "config_path": str(resolved_config_path),
                "output_dir": str(output_root),
            },
            "config_hash": config_hash,
            "injection_hash": injection_hash,
            "dataset": {},
            "preprocess": {},
            "candidates": [],
            "ranking": [],
            "scheduler": {},
            "reuse": {},
            "snapshots": snapshot.status(),
            "injections": injection_manager.summary(),
            "diagnostics": {},
            "failures": [],
            "business_codes": code_catalog(),
        }
        if dataset is None:
            failures.append(
                Failure(
                    code="PREPROCESS_NO_FEATURES",
                    stage="load",
                    item="dataset",
                    message="Dataset generation did not produce any usable table.",
                )
            )
            return self._finalize_early(
                report,
                failures,
                snapshot,
                injection_manager,
                output_root,
                started,
                ledger_entries,
                run_counters,
                previous_ledger,
            )

        report["dataset"] = {
            "name": dataset.descriptor.get("name", ""),
            "seed": dataset.descriptor.get("seed", config.get("seed")),
            "sample_scale": float(config.get("data", {}).get("sample_scale", 1.0) or 1.0),
            "rows_requested": dataset.rows_requested,
            "rows_generated": dataset.rows_generated,
            "max_rows": dataset.max_rows,
            "content_fingerprint": dataset.content_fingerprint,
            "descriptor_fingerprint": dataset.descriptor_fingerprint,
            "column_types": dataset.column_types,
            "target_columns": dataset.target_columns,
            "non_finite_locations": dataset.non_finite_locations,
            "preview": dataset.preview(),
        }

        preprocess_result, preprocess_source = self._obtain_preprocess(
            dataset,
            config,
            failures,
            artifact_store,
            snapshot,
            run_counters,
        )
        if preprocess_result is None:
            return self._finalize_early(
                report,
                failures,
                snapshot,
                injection_manager,
                output_root,
                started,
                ledger_entries,
                run_counters,
                previous_ledger,
            )

        report["preprocess"] = {
            **preprocess_result.to_metadata(),
            "preview": preprocess_result.preview(),
            "source": preprocess_source,
        }
        ledger_entries["preprocess"] = self._ledger_entry(
            phase="preprocess",
            logical_key="preprocess",
            status="reused" if preprocess_source != "rebuilt" else "rebuilt",
            identity_fingerprint=preprocess_result.content_fingerprint,
            artifact_fingerprint=preprocess_result.artifact_fingerprint,
            source=preprocess_source,
            details={"method": preprocess_result.method, "shape": list(preprocess_result.shape)},
        )
        snapshot.commit_stage(
            "preprocess",
            {
                "metadata": preprocess_payload(preprocess_result),
                "source": preprocess_source,
            },
            fingerprint=preprocess_result.artifact_fingerprint,
            binary_files={"preprocess.npz": _npz_bytes(X=preprocess_result.X)},
            extra_manifest={
                "status": "preprocess_committed",
                "rows_generated": dataset.rows_generated,
                "features": list(preprocess_result.feature_names),
            },
        )

        specs = build_candidate_specs(config)
        candidate_rows, valid_jobs = self._validate_candidates(
            specs,
            dataset.rows_generated,
            injection_manager,
            failures,
        )
        budget_plan, budget_failures = self._plan_budget(valid_jobs, config)
        failures.extend(budget_failures)
        budget_by_key = {entry["stable_key"]: entry for entry in budget_plan}
        for row in candidate_rows:
            budget = budget_by_key.get(row["stable_key"])
            if budget is not None and not budget.get("started", False):
                row["status"] = "budget_aborted"
                row["budget"] = budget

        scheduler = {
            "mode": "thread_pool" if int(config["scheduler"]["max_workers"]) > 1 else "serial",
            "max_workers": int(config["scheduler"]["max_workers"]),
            "global_iteration_budget": int(config["scheduler"]["global_iteration_budget"]),
            "per_candidate_iteration_quota": int(config["scheduler"]["per_candidate_iteration_quota"]),
            "budget_log": budget_plan,
            "budget_reserved": int(sum(entry["grant"] for entry in budget_plan)),
            "budget_remaining": int(
                max(
                    0,
                    int(config["scheduler"]["global_iteration_budget"])
                    - sum(entry["grant"] for entry in budget_plan),
                )
            ),
            "iterations_executed_total": 0,
            "completion_order": [],
            "policy": (
                "Deterministic commitments in candidate stable-key order; grants are deducted "
                "before thread execution and results are assembled in stable-key order."
            ),
        }

        cluster_records, cluster_sources = self._obtain_clusters(
            specs,
            valid_jobs,
            budget_by_key,
            preprocess_result,
            config,
            artifact_store,
            snapshot,
            injection_manager,
            failures,
            scheduler,
            run_counters,
        )

        for row in candidate_rows:
            stable_key = row["stable_key"]
            record = cluster_records.get(stable_key)
            budget = budget_by_key.get(stable_key)
            if record is None:
                row["budget"] = budget
                continue
            execution = _execution_from_record(record)
            row.update(
                {
                    "status": execution.status,
                    "labels": [int(value) for value in execution.labels.tolist()],
                    "label_fingerprint": record.get("label_fingerprint"),
                    "trajectory": execution.trajectory,
                    "loglik_per_sample": record.get("loglik_per_sample"),
                    "diagnostics": execution.diagnostics,
                    "diagnostic_flags": list(execution.diagnostic_flags),
                    "cluster_identity_fingerprint": record["identity_fingerprint"],
                    "cluster_artifact_fingerprint": record["artifact_fingerprint"],
                    "source": cluster_sources.get(stable_key, "missing"),
                    "budget": {
                        **(budget or {}),
                        "iterations_executed": execution.iterations_executed,
                    },
                }
            )
            ledger_entries[f"cluster:{stable_key}"] = self._ledger_entry(
                phase="cluster",
                logical_key=f"cluster:{stable_key}",
                status="reused" if cluster_sources.get(stable_key) != "rebuilt" else "rebuilt",
                identity_fingerprint=record["identity_fingerprint"],
                artifact_fingerprint=record["artifact_fingerprint"],
                source=cluster_sources.get(stable_key, "missing"),
                details={
                    "method": row["method"],
                    "hyperparameters": row.get("params", {}),
                    "parameter_fingerprint": hash_object("candparam", row.get("params", {})),
                    "preprocess_reference": {
                        "content_fingerprint": preprocess_result.content_fingerprint,
                        "artifact_fingerprint": preprocess_result.artifact_fingerprint,
                    },
                    "status": execution.status,
                    "iterations_executed": execution.iterations_executed,
                },
            )
            if execution.status == "truncated" and budget is not None:
                if budget.get("limit_reason") == "candidate_quota":
                    failures.append(
                        Failure(
                            code="SCHED_CANDIDATE_QUOTA_HIT",
                            stage="scheduler",
                            item=stable_key,
                            message="Candidate reached its per-candidate iteration quota before convergence.",
                            details={"budget": budget, "trajectory_length": len(execution.trajectory)},
                        )
                    )
                elif budget.get("limit_reason") == "global_budget":
                    failures.append(
                        Failure(
                            code="SCHED_GLOBAL_BUDGET_TRUNCATED",
                            stage="scheduler",
                            item=stable_key,
                            message="Candidate was truncated by the final global budget grant.",
                            details={"budget": budget, "trajectory_length": len(execution.trajectory)},
                        )
                    )
            if execution.status == "error":
                failures.append(
                    Failure(
                        code="CANDIDATE_RUNTIME_ERROR",
                        stage="cluster",
                        item=stable_key,
                        message=execution.error or "Candidate failed in isolation.",
                        details={"method": row["method"], "error_code": execution.error_code},
                    )
                )

        scheduler["iterations_executed_total"] = int(
            sum(
                int((row.get("budget") or {}).get("iterations_executed", 0))
                for row in candidate_rows
            )
        )
        scheduler["budget_consumed"] = scheduler["iterations_executed_total"]
        report["scheduler"] = scheduler

        evaluation_config = dict(config.get("evaluation", {}))
        cache = ResidentBlockCache(int(config["scheduler"]["cache_resident_blocks"]))
        evaluation_records, evaluation_sources = self._obtain_evaluations(
            candidate_rows,
            preprocess_result,
            evaluation_config,
            artifact_store,
            snapshot,
            cache,
            run_counters,
            int(config["scheduler"].get("block_size", 64)),
        )
        boundary_failures = self._apply_boundary_injections(
            candidate_rows,
            evaluation_records,
            specs,
            injection_manager,
        )
        failures.extend(boundary_failures)

        for row in candidate_rows:
            stable_key = row["stable_key"]
            source = evaluation_sources.get(stable_key)
            record = evaluation_records.get(stable_key)
            if record is None:
                continue
            evaluation = _evaluation_from_record(record)
            row["metrics"] = evaluation.get("metrics", {})
            row["metric_flags"] = evaluation.get("metric_flags", [])
            row["diagnostic_flags"] = evaluation.get("diagnostic_flags", row.get("diagnostic_flags", []))
            row["boundary_check"] = evaluation.get("boundary_check")
            row["evaluation_identity_fingerprint"] = record["identity_fingerprint"]
            row["evaluation_artifact_fingerprint"] = record["artifact_fingerprint"]
            row["evaluation_source"] = source
            ledger_entries[f"evaluation:{stable_key}"] = self._ledger_entry(
                phase="evaluation",
                logical_key=f"evaluation:{stable_key}",
                status="reused" if source != "rebuilt" else "rebuilt",
                identity_fingerprint=record["identity_fingerprint"],
                artifact_fingerprint=record["artifact_fingerprint"],
                source=source,
                details={"metrics": row["metrics"], "flags": row["diagnostic_flags"]},
            )
            self._record_evaluation_flags(row, failures)

        ranked = rank_candidates(candidate_rows, str(evaluation_config.get("primary_metric", "silhouette")))
        ranking = ranking_table(ranked, str(evaluation_config.get("primary_metric", "silhouette")))
        report["ranking"] = ranking
        report["candidates"] = sorted(
            candidate_rows,
            key=lambda row: (
                row.get("rank") is None,
                row.get("rank") or 10**9,
                row["stable_key"],
            ),
        )
        report["scheduler"]["cache"] = cache.summary()
        self._commit_metrics_stage(snapshot, evaluation_records, ranking)

        diagnosis = self._build_diagnostics(
            dataset,
            preprocess_result,
            candidate_rows,
            budget_plan,
            failures,
            injection_manager,
            cache,
        )
        failure_list = sort_failures(failures)
        report["diagnostics"] = diagnosis
        report["failures"] = [failure.to_dict() for failure in failure_list]
        report["injections"] = injection_manager.summary()
        report["snapshots"] = {
            **snapshot.status(),
            "commits": list(snapshot.manifest().get("stage_versions", {}).keys()),
        }

        diagnosis_identity = hash_object(
            "diag",
            {
                "evaluation_artifacts": {
                    key: record["artifact_fingerprint"]
                    for key, record in sorted(evaluation_records.items())
                },
                "injection_hash": injection_hash,
                "failure_codes": [failure.code for failure in failure_list],
                "flags": diagnosis["flag_counts"],
            },
        )
        diagnosis_artifact = hash_object(
            "diagart",
            {
                "identity": diagnosis_identity,
                "summary": diagnosis,
                "injections": injection_manager.summary(),
            },
        )
        ledger_entries["diagnostics"] = self._ledger_entry(
            phase="diagnostics",
            logical_key="diagnostics",
            status="rebuilt",
            identity_fingerprint=diagnosis_identity,
            artifact_fingerprint=diagnosis_artifact,
            source="rebuilt",
            details={"failure_count": len(failure_list), "flags": diagnosis["flag_counts"]},
        )
        report["diagnostics"]["identity_fingerprint"] = diagnosis_identity
        report["diagnostics"]["artifact_fingerprint"] = diagnosis_artifact

        snapshot.commit_stage(
            "diagnostics",
            {
                "diagnosis": diagnosis,
                "failures": report["failures"],
                "injections": report["injections"],
            },
            fingerprint=diagnosis_artifact,
            extra_manifest={
                "status": "completed",
                "completed_candidates": sorted(cluster_records),
                "budget_consumed": scheduler["iterations_executed_total"],
                "budget_reserved": scheduler["budget_reserved"],
                "iterations_executed": scheduler["iterations_executed_total"],
            },
        )

        reuse_report = self._build_reuse_report(
            ledger_entries,
            previous_ledger,
            run_counters,
            config_hash,
            injection_hash,
        )
        report["reuse"] = reuse_report
        artifact_store.save_ledger(
            {
                "config_hash": config_hash,
                "injection_hash": injection_hash,
                "entries": ledger_entries,
                "counters": run_counters,
            }
        )
        run_status = "completed_with_failures" if failure_list else "completed"
        report["run"]["status"] = run_status
        report["run"]["duration_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        report["run"]["run_id"] = hash_object(
            "run", {"config": config_hash, "injections": injection_hash}
        )
        report["determinism_digest"] = self._determinism_digest(report)
        report["snapshots"] = {
            **snapshot.status(),
            "manifest": snapshot.manifest(),
        }
        self._persist_report(report, output_root)
        return report

    def _obtain_preprocess(
        self,
        dataset: GeneratedDataset,
        config: dict[str, Any],
        failures: list[Failure],
        artifact_store: ArtifactStore,
        snapshot: SnapshotStore,
        counters: dict[str, int],
    ) -> tuple[PreprocessResult | None, str]:
        identity = preprocess_identity(dataset, config)
        content_fingerprint = str(identity["content_fingerprint"])
        if snapshot.valid:
            stage = snapshot.load_stage("preprocess")
            binary = snapshot.load_stage_npz("preprocess", "preprocess.npz")
            if stage is not None and binary is not None and "X" in binary:
                metadata = stage.get("payload", {}).get("metadata", {})
                if metadata.get("content_fingerprint") == content_fingerprint:
                    candidate = preprocess_from_payload(metadata, binary["X"])
                    expected_artifact = preprocess_artifact_fingerprint(
                        candidate.content_fingerprint,
                        candidate.X,
                        candidate.feature_names,
                        candidate.statistics,
                        candidate.dropped_columns,
                        candidate.train_mask,
                    )
                    if (
                        candidate.sample_ids == dataset.sample_ids
                        and candidate.shape == (dataset.rows_generated, candidate.shape[1])
                        and candidate.artifact_fingerprint == expected_artifact
                    ):
                        counters["snapshot_resume_calls"] += 1
                        validate_preprocess_columns(dataset, config, failures)
                        return candidate, "snapshot"
        cached = artifact_store.load_preprocess(content_fingerprint)
        if cached is not None and cached.sample_ids == dataset.sample_ids:
            counters["artifact_reuse_calls"] += 1
            validate_preprocess_columns(dataset, config, failures)
            return cached, "artifact"

        counters["preprocess_compute_calls"] += 1
        result = preprocess_dataset(dataset, config, failures)
        if result is not None:
            artifact_store.save_preprocess(result)
        return result, "rebuilt"

    def _validate_candidates(
        self,
        specs: list[CandidateSpec],
        n_samples: int,
        injection_manager: InjectionManager,
        failures: list[Failure],
    ) -> tuple[list[dict[str, Any]], list[CandidateSpec]]:
        rows: list[dict[str, Any]] = []
        valid_jobs: list[CandidateSpec] = []
        for spec in sorted(specs, key=lambda item: item.stable_key):
            injected_params, injection_hits = injection_manager.inject_candidate(
                spec.method,
                spec.configured_index,
                spec.params,
            )
            validation_spec = CandidateSpec(
                method=spec.method,
                params=injected_params,
                stable_key=spec.stable_key,
                lexical_key=spec.lexical_key,
                configured_index=spec.configured_index,
                display_id=spec.display_id,
            )
            validation_code, validation_message = validate_candidate_params(validation_spec, n_samples)
            invalid_reason = None
            if injection_hits:
                invalid_reason = "INJ_HYPERPARAMETER_OUT_OF_RANGE"
                for injection_id, evidence in injection_hits:
                    failure = injection_manager.record_hit(injection_id, evidence=evidence)
                    if failure:
                        failures.append(failure)
            elif validation_code:
                invalid_reason = validation_code
                failures.append(
                    Failure(
                        code=validation_code,
                        stage="config",
                        item=spec.stable_key,
                        message=validation_message or "Candidate rejected by parameter validation.",
                        details={"method": spec.method, "params": injected_params},
                    )
                )
            row = {
                "stable_key": spec.stable_key,
                "display_id": spec.display_id,
                "method": spec.method,
                "params": injected_params,
                "original_params": spec.params,
                "canonical_params": canonical_json(injected_params),
                "occurrence": spec.lexical_key[2],
                "status": "invalid" if invalid_reason else "planned",
                "invalid_code": invalid_reason,
                "metrics": {},
                "diagnostic_flags": ["CANDIDATE_REJECTED"] if invalid_reason else [],
                "rank": None,
            }
            rows.append(row)
            if invalid_reason is None:
                valid_jobs.append(spec)
        return rows, valid_jobs

    def _requested_iterations(self, spec: CandidateSpec) -> int:
        if spec.method == "DBSCAN":
            return 1
        return max(1, int(spec.params.get("max_iter", 1)))

    def _plan_budget(
        self,
        valid_jobs: list[CandidateSpec],
        config: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[Failure]]:
        failures: list[Failure] = []
        global_budget = int(config["scheduler"].get("global_iteration_budget", 0))
        quota = int(config["scheduler"].get("per_candidate_iteration_quota", 1))
        remaining = max(0, global_budget)
        plan: list[dict[str, Any]] = []
        for sequence, spec in enumerate(sorted(valid_jobs, key=lambda item: item.stable_key), start=1):
            requested = self._requested_iterations(spec)
            grant = min(quota, remaining)
            if grant <= 0:
                failures.append(
                    Failure(
                        code="SCHED_GLOBAL_BUDGET_EXHAUSTED",
                        stage="scheduler",
                        item=spec.stable_key,
                        message="No global iteration budget remained; candidate was not started.",
                        details={"requested": requested, "remaining_before": remaining},
                    )
                )
                plan.append(
                    {
                        "sequence": sequence,
                        "stable_key": spec.stable_key,
                        "method": spec.method,
                        "requested": requested,
                        "grant": 0,
                        "budget_before": remaining,
                        "budget_after": remaining,
                        "limit_reason": "global_budget",
                        "started": False,
                    }
                )
                continue
            if grant < requested:
                reason = "candidate_quota" if grant == quota else "global_budget"
            else:
                reason = "within_request"
            budget_before = remaining
            remaining -= grant
            plan.append(
                {
                    "sequence": sequence,
                    "stable_key": spec.stable_key,
                    "method": spec.method,
                    "requested": requested,
                    "grant": grant,
                    "budget_before": budget_before,
                    "budget_after": remaining,
                    "limit_reason": reason,
                    "started": True,
                }
            )
        return plan, failures

    def _cluster_identity(
        self,
        spec: CandidateSpec,
        budget: dict[str, Any],
        preprocess_result: PreprocessResult,
        config: dict[str, Any],
    ) -> str:
        return hash_object(
            "clid",
            {
                "preprocess_artifact": preprocess_result.artifact_fingerprint,
                "method": spec.method,
                "params": spec.params,
                "iteration_grant": int(budget.get("grant", 0)),
                "algorithm_version": config["scheduler"].get("algorithm_version", "det-v1"),
            },
        )

    def _obtain_clusters(
        self,
        specs: list[CandidateSpec],
        valid_jobs: list[CandidateSpec],
        budget_by_key: dict[str, dict[str, Any]],
        preprocess_result: PreprocessResult,
        config: dict[str, Any],
        artifact_store: ArtifactStore,
        snapshot: SnapshotStore,
        injection_manager: InjectionManager,
        failures: list[Failure],
        scheduler: dict[str, Any],
        counters: dict[str, int],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
        records: dict[str, dict[str, Any]] = {}
        sources: dict[str, str] = {}
        pending: list[tuple[CandidateSpec, dict[str, Any], str]] = []
        snapshot_payload: dict[str, Any] = {}
        if snapshot.valid:
            stage = snapshot.load_stage("clusters")
            if stage is not None:
                snapshot_payload = stage.get("payload", {}).get("records", {})

        for spec in sorted(valid_jobs, key=lambda item: item.stable_key):
            budget = budget_by_key[spec.stable_key]
            if not budget.get("started", False):
                continue
            identity = self._cluster_identity(spec, budget, preprocess_result, config)
            snapshot_record = snapshot_payload.get(spec.stable_key) if isinstance(snapshot_payload, dict) else None
            if isinstance(snapshot_record, dict) and snapshot_record.get("identity_fingerprint") == identity:
                records[spec.stable_key] = snapshot_record
                sources[spec.stable_key] = "snapshot"
                counters["snapshot_resume_calls"] += 1
                continue
            cached = artifact_store.load_cluster(identity)
            if cached is not None:
                records[spec.stable_key] = cached
                sources[spec.stable_key] = "artifact"
                counters["artifact_reuse_calls"] += 1
                continue
            pending.append((spec, budget, identity))

        def execute_one(spec: CandidateSpec, budget: dict[str, Any], identity: str) -> tuple[str, dict[str, Any]]:
            try:
                execution = execute_method(
                    spec,
                    preprocess_result.X,
                    int(budget["grant"]),
                    self.algorithm_runner,
                )
            except Exception as error:  # candidate isolation boundary
                execution = MethodExecution(
                    labels=np.full(preprocess_result.X.shape[0], -1, dtype=np.int64),
                    status="error",
                    iterations_executed=0,
                    iterations_requested=self._requested_iterations(spec),
                    diagnostics={"algorithm": spec.method},
                    trajectory=[],
                    error=truncate_text(f"{type(error).__name__}: {error}"),
                    error_code="CANDIDATE_RUNTIME_ERROR",
                )
            return spec.stable_key, _execution_to_record(identity, execution)

        completion_order: list[str] = []
        if pending:
            counters["cluster_compute_calls"] += len(pending)
            max_workers = max(1, int(config["scheduler"]["max_workers"]))
            commit_interval = max(1, int(config["scheduler"]["snapshot_commit_interval"]))
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="cluster") as executor:
                future_map = {
                    executor.submit(execute_one, spec, budget, identity): spec.stable_key
                    for spec, budget, identity in pending
                }
                completed_since_commit = 0
                for future in as_completed(future_map):
                    stable_key = future_map[future]
                    try:
                        key, record = future.result()
                    except Exception as error:
                        key = stable_key
                        record = _execution_to_record(
                            next(identity for spec, _, identity in pending if spec.stable_key == stable_key),
                            MethodExecution(
                                labels=np.full(preprocess_result.X.shape[0], -1, dtype=np.int64),
                                status="error",
                                iterations_executed=0,
                                iterations_requested=1,
                                diagnostics={},
                                trajectory=[],
                                error=truncate_text(f"{type(error).__name__}: {error}"),
                                error_code="CANDIDATE_RUNTIME_ERROR",
                            ),
                        )
                    records[key] = record
                    sources[key] = "rebuilt"
                    completion_order.append(key)
                    artifact_store.save_cluster(record["identity_fingerprint"], record)
                    completed_since_commit += 1
                    if completed_since_commit % commit_interval == 0:
                        self._commit_cluster_stage(
                            snapshot,
                            records,
                            budget_by_key,
                            scheduler,
                        )
        self._commit_cluster_stage(snapshot, records, budget_by_key, scheduler)
        scheduler["completion_order"] = completion_order
        return records, sources

    def _commit_cluster_stage(
        self,
        snapshot: SnapshotStore,
        records: dict[str, dict[str, Any]],
        budget_by_key: dict[str, dict[str, Any]],
        scheduler: dict[str, Any],
    ) -> None:
        ordered_records = {key: records[key] for key in sorted(records)}
        completed = sorted(records)
        budget_consumed = 0
        iterations_executed = 0
        for key in completed:
            budget = budget_by_key.get(key, {})
            budget_consumed += int(budget.get("grant", 0))
            iterations_executed += int(records[key].get("iterations_executed", 0))
        fingerprint = hash_object(
            "clusterstage",
            {
                "records": {
                    key: record.get("artifact_fingerprint") for key, record in ordered_records.items()
                },
                "budget": scheduler.get("budget_log", []),
            },
        )
        snapshot.commit_stage(
            "clusters",
            {
                "records": ordered_records,
                "completed_candidates": completed,
                "budget_log": scheduler.get("budget_log", []),
                "budget_policy": scheduler.get("policy", ""),
            },
            fingerprint=fingerprint,
            extra_manifest={
                "status": "clusters_committed",
                "completed_candidates": completed,
                "budget_consumed": iterations_executed,
                "budget_reserved": budget_consumed,
                "iterations_executed": iterations_executed,
            },
        )

    def _obtain_evaluations(
        self,
        candidate_rows: list[dict[str, Any]],
        preprocess_result: PreprocessResult,
        evaluation_config: dict[str, Any],
        artifact_store: ArtifactStore,
        snapshot: SnapshotStore,
        cache: ResidentBlockCache,
        counters: dict[str, int],
        block_size: int,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
        records: dict[str, dict[str, Any]] = {}
        sources: dict[str, str] = {}
        snapshot_payload: dict[str, Any] = {}
        if snapshot.valid:
            stage = snapshot.load_stage("metrics")
            if stage is not None:
                snapshot_payload = stage.get("payload", {}).get("records", {})
        specs_by_key = {row["stable_key"]: row for row in candidate_rows}
        for row in sorted(candidate_rows, key=lambda item: item["stable_key"]):
            if row.get("status") not in {"completed", "converged", "truncated"}:
                continue
            cluster_identity = row.get("cluster_identity_fingerprint")
            if not cluster_identity:
                continue
            identity = evaluation_identity(str(cluster_identity), evaluation_config)
            snapshot_record = snapshot_payload.get(row["stable_key"]) if isinstance(snapshot_payload, dict) else None
            if isinstance(snapshot_record, dict) and snapshot_record.get("identity_fingerprint") == identity:
                records[row["stable_key"]] = snapshot_record
                sources[row["stable_key"]] = "snapshot"
                counters["snapshot_resume_calls"] += 1
                continue
            cached = artifact_store.load_evaluation(identity)
            if cached is not None:
                records[row["stable_key"]] = cached
                sources[row["stable_key"]] = "artifact"
                counters["artifact_reuse_calls"] += 1
                continue
            counters["evaluation_compute_calls"] += 1
            spec = CandidateSpec(
                method=row["method"],
                params=dict(row["params"]),
                stable_key=row["stable_key"],
                lexical_key=(row["method"], row["canonical_params"], row.get("occurrence", "000")),
                configured_index=0,
                display_id=row["display_id"],
            )
            execution = _execution_from_record(
                {
                    "labels": row["labels"],
                    "status": row["status"],
                    "iterations_executed": (row.get("budget") or {}).get("iterations_executed", 0),
                    "iterations_requested": (row.get("budget") or {}).get("requested", 0),
                    "diagnostics": row.get("diagnostics", {}),
                    "trajectory": row.get("trajectory", []),
                    "loglik_per_sample": row.get("loglik_per_sample"),
                    "diagnostic_flags": row.get("diagnostic_flags", []),
                }
            )
            result = evaluate_candidate(
                preprocess_result.X,
                spec,
                execution,
                str(cluster_identity),
                evaluation_config,
                cache,
                block_size,
            )
            record = _evaluation_record(identity, result)
            records[row["stable_key"]] = record
            sources[row["stable_key"]] = "rebuilt"
            artifact_store.save_evaluation(identity, record)
        return records, sources

    def _apply_boundary_injections(
        self,
        candidate_rows: list[dict[str, Any]],
        evaluation_records: dict[str, dict[str, Any]],
        specs: list[CandidateSpec],
        injection_manager: InjectionManager,
    ) -> list[Failure]:
        failures: list[Failure] = []
        items = injection_manager.boundary_targets()
        if not items:
            return failures
        spec_by_key = {spec.stable_key: spec for spec in specs}
        for row in sorted(candidate_rows, key=lambda item: item["stable_key"]):
            record = evaluation_records.get(row["stable_key"])
            spec = spec_by_key.get(row["stable_key"])
            if record is None or spec is None:
                continue
            evaluation = _evaluation_from_record(record)
            score = (evaluation.get("metrics") or {}).get("silhouette")
            if score is None:
                continue
            for item in items:
                injection_id = str(item.get("id"))
                report = injection_manager.reports.get(injection_id)
                if report is not None and report.hit:
                    continue
                if not injection_manager.boundary_matches(item, spec.method, spec.configured_index, spec.params):
                    continue
                threshold = float(score)
                boundary = {
                    "metric": "silhouette",
                    "threshold": threshold,
                    "score": float(score),
                    "operator": ">=",
                    "decision": "accept" if float(score) >= threshold else "reject",
                    "boundary_hit": True,
                    "injected": True,
                    "delta": float(score) - threshold,
                }
                evaluation["boundary_check"] = boundary
                artifact_fingerprint = hash_object(
                    "evalart",
                    {
                        "identity": record["identity_fingerprint"],
                        "labels": record.get("label_hash"),
                        "metrics": evaluation.get("metrics"),
                        "flags": evaluation.get("diagnostic_flags"),
                        "boundary_check": boundary,
                    },
                )
                evaluation["artifact_fingerprint"] = artifact_fingerprint
                record["artifact_fingerprint"] = artifact_fingerprint
                record["evaluation"] = evaluation
                failure = injection_manager.record_hit(
                    injection_id,
                    evidence={
                        "candidate": row["stable_key"],
                        "metric": "silhouette",
                        "score": float(score),
                        "threshold": threshold,
                        "delta": 0.0,
                        "decision": boundary["decision"],
                        "comparison": "score >= threshold is inclusive",
                    },
                )
                if failure:
                    failures.append(failure)
        return failures

    def _record_evaluation_flags(self, row: dict[str, Any], failures: list[Failure]) -> None:
        flags = row.get("diagnostic_flags", [])
        if "EMPTY_CLUSTER" in flags:
            failures.append(
                Failure(
                    code="EVAL_EMPTY_CLUSTER",
                    stage="evaluation",
                    item=row["stable_key"],
                    message="Candidate produced an empty cluster/component and remains in the ranking.",
                    details={"flags": flags, "metrics": row.get("metrics")},
                )
            )
        if "ALL_NOISE" in flags:
            failures.append(
                Failure(
                    code="EVAL_ALL_NOISE",
                    stage="evaluation",
                    item=row["stable_key"],
                    message="DBSCAN classified all samples as noise; candidate remains in the ranking.",
                    details={"flags": flags, "metrics": row.get("metrics")},
                )
            )
        if row.get("metric_flags"):
            failures.append(
                Failure(
                    code="METRIC_UNDEFINED",
                    stage="evaluation",
                    item=row["stable_key"],
                    message="One or more metrics are undefined for this label structure; candidate was retained.",
                    details={"metrics": row["metric_flags"]},
                )
            )

    def _commit_metrics_stage(
        self,
        snapshot: SnapshotStore,
        evaluation_records: dict[str, dict[str, Any]],
        ranking: list[dict[str, Any]],
    ) -> None:
        fingerprint = hash_object(
            "metricstage",
            {
                "records": {
                    key: record["artifact_fingerprint"]
                    for key, record in sorted(evaluation_records.items())
                },
                "ranking": ranking,
            },
        )
        snapshot.commit_stage(
            "metrics",
            {
                "records": {key: evaluation_records[key] for key in sorted(evaluation_records)},
                "ranking": ranking,
            },
            fingerprint=fingerprint,
            extra_manifest={"status": "metrics_committed"},
        )

    def _build_diagnostics(
        self,
        dataset: GeneratedDataset,
        preprocess_result: PreprocessResult,
        candidate_rows: list[dict[str, Any]],
        budget_plan: list[dict[str, Any]],
        failures: list[Failure],
        injection_manager: InjectionManager,
        cache: ResidentBlockCache,
    ) -> dict[str, Any]:
        flag_counts: dict[str, int] = {}
        for row in candidate_rows:
            for flag in row.get("diagnostic_flags", []):
                flag_counts[flag] = flag_counts.get(flag, 0) + 1
        ranked = [row for row in candidate_rows if row.get("rank") is not None]
        injection_summary = injection_manager.summary()
        failure_codes: dict[str, int] = {}
        for failure in failures:
            failure_codes[failure.code] = failure_codes.get(failure.code, 0) + 1
        return {
            "ranked_candidates": len(ranked),
            "rejected_candidates": sum(1 for row in candidate_rows if row.get("status") == "invalid"),
            "budget_granted": int(sum(item.get("grant", 0) for item in budget_plan)),
            "budget_remaining": int(budget_plan[-1]["budget_after"]) if budget_plan else 0,
            "flag_counts": {key: flag_counts[key] for key in sorted(flag_counts)},
            "failure_code_counts": {key: failure_codes[key] for key in sorted(failure_codes)},
            "injection_hit_count": injection_summary["hit"],
            "injection_missed_count": injection_summary["missed"],
            "cache": cache.summary(),
            "preprocess_statistics_summary": {
                name: {
                    key: value
                    for key, value in stats.items()
                    if key in {"type", "fill_value", "center", "scale", "variance", "output_domain"}
                }
                for name, stats in preprocess_result.statistics.items()
                if name != "_pipeline"
            },
            "dataset_rows": dataset.rows_generated,
        }

    def _ledger_entry(
        self,
        *,
        phase: str,
        logical_key: str,
        status: str,
        identity_fingerprint: str,
        artifact_fingerprint: str,
        source: str,
        details: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "phase": phase,
            "logical_key": logical_key,
            "status": status,
            "identity_fingerprint": identity_fingerprint,
            "artifact_fingerprint": artifact_fingerprint,
            "source": source,
            "details": json_safe(details),
        }

    def _build_reuse_report(
        self,
        entries: dict[str, dict[str, Any]],
        previous_ledger: dict[str, Any],
        counters: dict[str, int],
        config_hash: str,
        injection_hash: str,
    ) -> dict[str, Any]:
        previous_entries = previous_ledger.get("entries", {}) if isinstance(previous_ledger, dict) else {}
        if not isinstance(previous_entries, dict):
            previous_entries = {}
        current_entries = []
        for key in sorted(entries):
            entry = dict(entries[key])
            previous = previous_entries.get(key)
            if isinstance(previous, dict):
                entry["previous_identity_fingerprint"] = previous.get("identity_fingerprint")
                entry["identity_changed"] = previous.get("identity_fingerprint") != entry["identity_fingerprint"]
                entry["recomputed_despite_same_identity"] = (
                    not entry["identity_changed"]
                    and entry["status"] == "rebuilt"
                    and previous.get("artifact_fingerprint") == entry["artifact_fingerprint"]
                )
            else:
                entry["previous_identity_fingerprint"] = None
                entry["identity_changed"] = True
                entry["recomputed_despite_same_identity"] = False
            current_entries.append(entry)
        retired = []
        for key in sorted(set(previous_entries) - set(entries)):
            previous = previous_entries[key]
            if isinstance(previous, dict):
                retired.append(
                    {
                        "logical_key": key,
                        "phase": previous.get("phase"),
                        "identity_fingerprint": previous.get("identity_fingerprint"),
                        "reason": "logical_key_no_longer_present",
                    }
                )
        status_counts: dict[str, int] = {}
        for entry in current_entries:
            status_counts[entry["status"]] = status_counts.get(entry["status"], 0) + 1
        return {
            "config_hash": config_hash,
            "injection_hash": injection_hash,
            "entries": current_entries,
            "retired": retired,
            "status_counts": {key: status_counts[key] for key in sorted(status_counts)},
            "counters": counters,
            "reused_total": sum(1 for entry in current_entries if entry["status"] == "reused"),
            "rebuilt_total": sum(1 for entry in current_entries if entry["status"] == "rebuilt"),
            "policy": (
                "Identity fingerprints are derived from upstream content, normalized parameters, "
                "and cache keys. Reused entries are loaded verbatim; only identity mismatches rebuild."
            ),
        }

    def _determinism_digest(self, report: dict[str, Any]) -> str:
        core = {
            "dataset": {
                key: report["dataset"].get(key)
                for key in ("rows_generated", "content_fingerprint", "descriptor_fingerprint")
            },
            "preprocess": {
                key: report["preprocess"].get(key)
                for key in ("method", "shape", "content_fingerprint", "artifact_fingerprint", "feature_names")
            },
            "candidates": [
                {
                    "stable_key": row.get("stable_key"),
                    "status": row.get("status"),
                    "labels": row.get("labels"),
                    "metrics": row.get("metrics"),
                    "diagnostic_flags": row.get("diagnostic_flags"),
                    "rank": row.get("rank"),
                    "budget": row.get("budget"),
                    "cluster_artifact_fingerprint": row.get("cluster_artifact_fingerprint"),
                    "evaluation_artifact_fingerprint": row.get("evaluation_artifact_fingerprint"),
                }
                for row in report.get("candidates", [])
            ],
            "ranking": report.get("ranking", []),
            "budget_log": report.get("scheduler", {}).get("budget_log", []),
            # Snapshot bookkeeping failures describe where a run resumed from, not
            # the computed result. They are retained in the report but excluded
            # from the cross-run computational digest.
            "failures": [
                failure
                for failure in report.get("failures", [])
                if failure.get("stage") != "snapshot"
            ],
            "injections": report.get("injections", {}).get("items", []),
        }
        return hash_object("det", json_safe(core))

    def _finalize_early(
        self,
        report: dict[str, Any],
        failures: list[Failure],
        snapshot: SnapshotStore,
        injection_manager: InjectionManager,
        output_root: Path,
        started: float,
        ledger_entries: dict[str, dict[str, Any]],
        counters: dict[str, int],
        previous_ledger: dict[str, Any],
    ) -> dict[str, Any]:
        failure_list = sort_failures(failures)
        report["failures"] = [failure.to_dict() for failure in failure_list]
        report["injections"] = injection_manager.summary()
        report["reuse"] = self._build_reuse_report(
            ledger_entries,
            previous_ledger,
            counters,
            report["config_hash"],
            report["injection_hash"],
        )
        report["run"]["status"] = "failed"
        report["run"]["duration_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        report["run"]["run_id"] = hash_object(
            "run", {"config": report["config_hash"], "injections": report["injection_hash"]}
        )
        report["determinism_digest"] = self._determinism_digest(report)
        report["snapshots"] = {**snapshot.status(), "manifest": snapshot.manifest()}
        self._persist_report(report, output_root)
        return report

    def _persist_report(self, report: dict[str, Any], output_root: Path) -> None:
        atomic_write_json(output_root / "latest_report.json", report)
        core = {
            "schema_version": report["schema_version"],
            "config_hash": report["config_hash"],
            "injection_hash": report["injection_hash"],
            "dataset": report.get("dataset"),
            "preprocess": report.get("preprocess"),
            "candidates": report.get("candidates"),
            "ranking": report.get("ranking"),
            "scheduler": {
                key: value
                for key, value in report.get("scheduler", {}).items()
                if key in {"budget_log", "budget_reserved", "budget_remaining", "iterations_executed_total", "max_workers"}
            },
            "reuse": report.get("reuse"),
            "snapshots": report.get("snapshots"),
            "injections": report.get("injections"),
            "diagnostics": report.get("diagnostics"),
            "failures": report.get("failures"),
            "determinism_digest": report.get("determinism_digest"),
        }
        atomic_write_json(output_root / "latest_core.json", json_safe(core))
