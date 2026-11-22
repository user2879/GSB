from __future__ import annotations

import copy
import json
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .canonical import (
    SCHEMA_VERSION,
    fingerprint,
    normalized_artifact,
    sorted_failures,
)
from .config import PipelineConfig, validate_config
from .edits import EditPlan, apply_batch
from .features import compute_features, feature_cache_key
from .inference import (
    build_summary_artifact,
    infer_and_summarize,
    inference_cache_key,
    summary_cache_key,
    validation_dataset_fingerprint,
)
from .loader import (
    SampleStore,
    feature_input_fingerprint,
    full_data_fingerprint,
    parse_samples,
    state_failures,
)
from .snapshot import SnapshotData, SnapshotStore
from .training import (
    train_model,
    training_cache_key,
    training_dataset_fingerprint,
)


STAGE_ORDER = ("features", "training", "inference", "summary")


@dataclass
class BatchOutcome:
    report: dict[str, Any]
    stages: dict[str, dict[str, Any]]
    ledger: list[dict[str, Any]]
    complete: bool
    pending: list[str]
    data_failures: list[dict[str, Any]]
    all_failures: list[dict[str, Any]]
    config: PipelineConfig
    parsed_samples: Any


@dataclass
class RunResult:
    reports: list[dict[str, Any]]
    manifest: dict[str, Any] | None
    applied_batches: list[str]
    unprocessed_batches: list[str] = field(default_factory=list)


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 3)


def _canonical_dependency(ledger: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dependency: list[dict[str, Any]] = []
    for entry in ledger:
        dependency.append(
            {
                "stage": entry["stage"],
                "cache_key": entry.get("cache_key"),
                "sample_ids": entry.get("sample_ids", []),
                "upstreams": entry.get("upstreams", []),
            }
        )
    return dependency


def _diff_paths(left: Any, right: Any, path: str = "$") -> list[str]:
    if type(left) is not type(right):
        return [f"{path}:type"]
    if isinstance(left, dict):
        differences: list[str] = []
        for key in sorted(set(left) | set(right)):
            if key not in left:
                differences.append(f"{path}.{key}:missing-left")
            elif key not in right:
                differences.append(f"{path}.{key}:missing-right")
            else:
                differences.extend(_diff_paths(left[key], right[key], f"{path}.{key}"))
        return differences
    if isinstance(left, list):
        if len(left) != len(right):
            return [f"{path}:length {len(left)} != {len(right)}"]
        differences = []
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            differences.extend(
                _diff_paths(left_item, right_item, f"{path}[{index}]")
            )
        return differences
    if left != right:
        return [f"{path}:value"]
    return []


def canonical_comparison_values(
    stages: dict[str, dict[str, Any]],
    data_failures: list[dict[str, Any]],
    ledger: list[dict[str, Any]],
    *,
    pipeline_complete: bool,
) -> dict[str, Any]:
    return {
        "pipeline_complete": pipeline_complete,
        "stages": {
            stage: normalized_artifact(stages.get(stage))
            for stage in STAGE_ORDER
        },
        "data_failures": sorted_failures(data_failures),
        "dependency": _canonical_dependency(ledger),
    }


class PipelineRunner:
    def __init__(
        self,
        *,
        snapshot_store: SnapshotStore,
        sample_store: SampleStore,
        raw_config: dict[str, Any],
        applied_batches: list[str] | None = None,
        feature_row_cache: dict[str, dict[str, Any]] | None = None,
        inference_row_cache: dict[str, dict[str, Any]] | None = None,
        stages: dict[str, dict[str, Any]] | None = None,
        ledger: list[dict[str, Any]] | None = None,
        pipeline_complete: bool = True,
        pending: list[str] | None = None,
        plan_fingerprint: str | None = None,
        last_manifest: dict[str, Any] | None = None,
    ):
        self.snapshot_store = snapshot_store
        self.sample_store = sample_store
        self.raw_config = copy.deepcopy(raw_config)
        self.applied_batches = list(applied_batches or [])
        self.feature_row_cache = dict(feature_row_cache or {})
        self.inference_row_cache = dict(inference_row_cache or {})
        self.stages = dict(stages or {})
        self.ledger = list(ledger or [])
        self.pipeline_complete = pipeline_complete
        self.pending = list(pending or [])
        self.plan_fingerprint = plan_fingerprint
        self.last_manifest = last_manifest
        self._last_outcome: BatchOutcome | None = None
        self._batches_since_commit = 0

    @classmethod
    def from_snapshot(cls, snapshot_store: SnapshotStore) -> "PipelineRunner | None":
        data: SnapshotData | None = snapshot_store.load_current()
        if data is None:
            return None
        return cls(
            snapshot_store=snapshot_store,
            sample_store=SampleStore.from_lines(data.sample_lines),
            raw_config=data.raw_config,
            applied_batches=data.applied_batches,
            feature_row_cache=data.feature_row_cache,
            inference_row_cache=data.inference_row_cache,
            stages=data.stages,
            ledger=data.ledger,
            pipeline_complete=data.pipeline_complete,
            pending=data.pending,
            plan_fingerprint=data.plan_fingerprint,
            last_manifest=data.manifest,
        )

    def _target_keys(
        self,
        config: PipelineConfig,
        parsed_samples: Any,
    ) -> dict[str, str | None]:
        keys: dict[str, str | None] = {
            "features": feature_cache_key(config, parsed_samples.samples),
            "training": None,
            "inference": None,
            "summary": None,
        }
        return keys

    def process_current(
        self,
        *,
        batch_id: str,
        edit_failures: list[dict[str, Any]] | None = None,
        plan_failures: list[dict[str, Any]] | None = None,
        mode: str = "incremental",
    ) -> BatchOutcome:
        edit_failures = list(edit_failures or [])
        plan_failures = list(plan_failures or [])
        config = validate_config(self.raw_config)
        parsed = parse_samples(self.sample_store.lines, config)
        data_failures = state_failures(config.failures, parsed.failures)
        data_fp = full_data_fingerprint(parsed.samples)
        raw_fp = self.sample_store.raw_fingerprint()
        previous_stages = self.stages
        target_keys = self._target_keys(config, parsed)
        stages: dict[str, dict[str, Any]] = {}
        ledger: list[dict[str, Any]] = []
        stage_timings: dict[str, float] = {}
        stage_calls: dict[str, int] = {}
        fragment_reuse: dict[str, Any] = {}

        feature_key = target_keys["features"]
        previous_features = previous_stages.get("features")
        feature_status: str
        feature_reason: str
        if previous_features is not None and previous_features.get("cache_key") == feature_key:
            feature_status = "reused"
            feature_reason = "cache_key_match"
            stage_timings["features"] = 0.0
            stage_calls["features"] = 0
            stages["features"] = previous_features
            budget_report = {
                "name": "feature_rows",
                "limit": int(config.concurrency["batch_budget"]),
                "consumed": 0,
                "remaining": int(config.concurrency["batch_budget"]),
                "sequence": [],
                "pending": [],
            }
            feature_pending: list[str] = []
            fragment_reuse["features"] = {
                "rows_total": len(parsed.samples),
                "rows_reused": len(parsed.samples),
                "rows_computed": 0,
                "pending": 0,
            }
        else:
            feature_status = "recomputed"
            feature_reason = (
                "cold_start"
                if previous_features is None
                else "feature_input_or_params_changed"
            )
            started = time.perf_counter()
            feature_run = compute_features(
                config,
                parsed.samples,
                row_cache=self.feature_row_cache,
            )
            stage_timings["features"] = _elapsed_ms(started)
            stage_calls["features"] = 1
            self.feature_row_cache = feature_run.row_cache
            budget_report = feature_run.budget_report
            feature_pending = feature_run.pending
            fragment_reuse["features"] = {
                "rows_total": feature_run.rows_total,
                "rows_reused": feature_run.rows_reused,
                "rows_computed": feature_run.rows_computed,
                "pending": len(feature_run.pending),
            }
            if feature_run.complete and feature_run.artifact is not None:
                stages["features"] = feature_run.artifact
            else:
                feature_status = "pending"
                feature_reason = "E_BATCH_BUDGET_PENDING"

        ledger.append(
            {
                "stage": "features",
                "status": feature_status,
                "reason": feature_reason,
                "cache_key": feature_key if feature_status != "pending" else None,
                "target_cache_key": feature_key,
                "previous_cache_key": (
                    previous_features.get("cache_key")
                    if previous_features is not None
                    else None
                ),
                "sample_ids": [sample.sample_id for sample in parsed.samples],
                "upstreams": [
                    {
                        "name": "raw_samples",
                        "raw_data_fingerprint": raw_fp,
                        "feature_input_fingerprint": feature_input_fingerprint(
                            parsed.samples
                        ),
                    }
                ],
                "fragments": fragment_reuse["features"],
            }
        )

        training_pending = feature_status == "pending"
        inference_pending = feature_status == "pending"
        summary_pending = feature_status == "pending"
        train_evaluations = 0
        inference_evaluations = 0

        if not training_pending:
            feature_artifact = stages["features"]
            training_key = training_cache_key(
                config, parsed.train_samples, feature_artifact
            )
            target_keys["training"] = training_key
            previous_training = previous_stages.get("training")
            if (
                previous_training is not None
                and previous_training.get("cache_key") == training_key
            ):
                training_status = "reused"
                training_reason = "cache_key_match"
                stage_timings["training"] = 0.0
                stage_calls["training"] = 0
                stages["training"] = previous_training
            else:
                training_status = "recomputed"
                training_reason = (
                    "cold_start"
                    if previous_training is None
                    else "training_data_or_params_changed"
                )
                started = time.perf_counter()
                training_run = train_model(
                    config, parsed.train_samples, feature_artifact
                )
                stage_timings["training"] = _elapsed_ms(started)
                stage_calls["training"] = 1
                train_evaluations = training_run.gradient_evaluations
                stages["training"] = training_run.artifact
            fragment_reuse["training"] = {
                "gradient_evaluations": train_evaluations,
            }
            ledger.append(
                {
                    "stage": "training",
                    "status": training_status,
                    "reason": training_reason,
                    "cache_key": training_key,
                    "target_cache_key": training_key,
                    "previous_cache_key": (
                        previous_training.get("cache_key")
                        if previous_training is not None
                        else None
                    ),
                    "sample_ids": [
                        sample.sample_id for sample in parsed.train_samples
                    ],
                    "upstreams": [
                        {
                            "name": "features",
                            "cache_key": stages["features"]["cache_key"],
                            "training_dataset_fingerprint": training_dataset_fingerprint(
                                parsed.train_samples, stages["features"]
                            ),
                        }
                    ],
                    "fragments": fragment_reuse["training"],
                }
            )
        else:
            training_status = "pending"
            training_reason = "upstream_features_pending"
            stage_timings["training"] = 0.0
            stage_calls["training"] = 0
            ledger.append(
                {
                    "stage": "training",
                    "status": training_status,
                    "reason": training_reason,
                    "cache_key": None,
                    "target_cache_key": None,
                    "previous_cache_key": (
                        previous_stages.get("training", {}).get("cache_key")
                        if previous_stages.get("training")
                        else None
                    ),
                    "sample_ids": [
                        sample.sample_id for sample in parsed.train_samples
                    ],
                    "upstreams": [],
                    "fragments": {"gradient_evaluations": 0},
                }
            )

        if not inference_pending and "training" in stages and "features" in stages:
            inference_key = inference_cache_key(
                config,
                parsed.validation_samples,
                stages["features"],
                stages["training"],
            )
            target_keys["inference"] = inference_key
            previous_inference = previous_stages.get("inference")
            if (
                previous_inference is not None
                and previous_inference.get("cache_key") == inference_key
            ):
                inference_status = "reused"
                inference_reason = "cache_key_match"
                stage_timings["inference"] = 0.0
                stage_calls["inference"] = 0
                stages["inference"] = previous_inference
                fragment_reuse["inference"] = {
                    "rows_total": len(parsed.validation_samples),
                    "rows_reused": len(parsed.validation_samples),
                    "rows_computed": 0,
                }
            else:
                inference_status = "recomputed"
                inference_reason = (
                    "cold_start"
                    if previous_inference is None
                    else "model_validation_or_quantization_changed"
                )
                started = time.perf_counter()
                inference_run = infer_and_summarize(
                    config,
                    parsed.validation_samples,
                    stages["features"],
                    stages["training"],
                    row_cache=self.inference_row_cache,
                )
                stage_timings["inference"] = _elapsed_ms(started)
                stage_calls["inference"] = 1
                self.inference_row_cache = inference_run.row_cache
                inference_evaluations = inference_run.inference_evaluations
                stages["inference"] = inference_run.inference_artifact
                stages["summary"] = inference_run.summary_artifact
                fragment_reuse["inference"] = {
                    "rows_total": inference_run.rows_total,
                    "rows_reused": inference_run.rows_reused,
                    "rows_computed": inference_run.rows_computed,
                }
            ledger.append(
                {
                    "stage": "inference",
                    "status": inference_status,
                    "reason": inference_reason,
                    "cache_key": inference_key,
                    "target_cache_key": inference_key,
                    "previous_cache_key": (
                        previous_inference.get("cache_key")
                        if previous_inference is not None
                        else None
                    ),
                    "sample_ids": [
                        sample.sample_id for sample in parsed.validation_samples
                    ],
                    "upstreams": [
                        {
                            "name": "training",
                            "cache_key": stages["training"]["cache_key"],
                            "validation_dataset_fingerprint": validation_dataset_fingerprint(
                                parsed.validation_samples, stages["features"]
                            ),
                            "quantization": config.quantization_params(),
                        }
                    ],
                    "fragments": fragment_reuse["inference"],
                }
            )
            inference_pending = False
        else:
            inference_status = "pending" if inference_pending else "pending"
            inference_reason = (
                "upstream_features_pending"
                if inference_pending
                else "upstream_training_pending"
            )
            stage_timings["inference"] = 0.0
            stage_calls["inference"] = 0
            ledger.append(
                {
                    "stage": "inference",
                    "status": "pending",
                    "reason": inference_reason,
                    "cache_key": None,
                    "target_cache_key": None,
                    "previous_cache_key": (
                        previous_stages.get("inference", {}).get("cache_key")
                        if previous_stages.get("inference")
                        else None
                    ),
                    "sample_ids": [
                        sample.sample_id for sample in parsed.validation_samples
                    ],
                    "upstreams": [],
                    "fragments": {
                        "rows_total": len(parsed.validation_samples),
                        "rows_reused": 0,
                        "rows_computed": 0,
                    },
                }
            )

        if "inference" in stages and "summary" not in stages:
            target_summary_key = summary_cache_key(config, stages["inference"])
            previous_summary = previous_stages.get("summary")
            if (
                previous_summary is not None
                and previous_summary.get("cache_key") == target_summary_key
            ):
                stages["summary"] = previous_summary
                summary_status = "reused"
                summary_reason = "cache_key_match"
                stage_calls["summary"] = 0
                stage_timings["summary"] = 0.0
            else:
                started = time.perf_counter()
                stages["summary"] = build_summary_artifact(
                    config,
                    stages["inference"]["predictions"],
                    stages["inference"],
                )
                summary_status = "recomputed"
                summary_reason = "summary_key_changed"
                stage_calls["summary"] = 1
                stage_timings["summary"] = _elapsed_ms(started)
        elif "summary" in stages:
            target_summary_key = stages["summary"]["cache_key"]
            previous_summary = previous_stages.get("summary")
            if (
                previous_summary is not None
                and previous_summary.get("cache_key") == target_summary_key
            ):
                summary_status = "reused"
                summary_reason = "cache_key_match"
                stage_calls["summary"] = 0
                stage_timings["summary"] = 0.0
            else:
                summary_status = "recomputed"
                summary_reason = "inference_recomputed"
                stage_calls["summary"] = 1
                stage_timings["summary"] = 0.0
        else:
            target_summary_key = None
            previous_summary = previous_stages.get("summary")
            summary_status = "pending"
            summary_reason = "upstream_inference_pending"
            stage_calls["summary"] = 0
            stage_timings["summary"] = 0.0

        target_keys["summary"] = target_summary_key
        ledger.append(
            {
                "stage": "summary",
                "status": summary_status,
                "reason": summary_reason,
                "cache_key": (
                    target_summary_key if summary_status != "pending" else None
                ),
                "target_cache_key": target_summary_key,
                "previous_cache_key": (
                    previous_summary.get("cache_key")
                    if previous_summary is not None
                    else None
                ),
                "sample_ids": [
                    sample.sample_id for sample in parsed.validation_samples
                ],
                "upstreams": (
                    [
                        {
                            "name": "inference",
                            "cache_key": stages["inference"]["cache_key"],
                        }
                    ]
                    if "inference" in stages
                    else []
                ),
                "fragments": {"rows_total": len(parsed.validation_samples)},
            }
        )

        if feature_status != "pending" and training_status != "pending" and inference_status != "pending":
            complete = summary_status in ("reused", "recomputed")
        else:
            complete = False
        if not complete:
            stages = {stage: artifact for stage, artifact in stages.items() if stage in ("features",) and stage == "features" and feature_status != "pending"}
            if feature_status == "pending":
                stages = {}

        pending = list(feature_pending)
        budget_failures = [
            {
                "scope": "budget",
                "path": task_key,
                "sample_id": task_key.split(":", 1)[-1],
                "codes": ["E_BATCH_BUDGET_PENDING"],
                "message": "single-batch sample processing budget exhausted",
            }
            for task_key in pending
        ]
        all_failures = sorted_failures(
            data_failures + edit_failures + plan_failures + budget_failures
        )
        status_by_stage = {entry["stage"]: entry["status"] for entry in ledger}
        invalidations = [
            stage for stage in STAGE_ORDER if status_by_stage[stage] != "reused"
        ]
        reuses = [
            stage for stage in STAGE_ORDER if status_by_stage[stage] == "reused"
        ]
        report: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "mode": mode,
            "batch_id": batch_id,
            "config_hash": config.config_hash,
            "data_fingerprint": data_fp,
            "raw_data_fingerprint": raw_fp,
            "sample_count": len(parsed.samples),
            "pipeline_complete": complete,
            "stage_timings_ms": stage_timings,
            "stage_calls": stage_calls,
            "stages": ledger,
            "invalidations": invalidations,
            "reuses": reuses,
            "fragment_reuse": fragment_reuse,
            "budget": budget_report,
            "pending": pending,
            "failures": all_failures,
            "data_failures": data_failures,
            "edit_failures": sorted_failures(edit_failures),
            "plan_failures": sorted_failures(plan_failures),
            "stage_versions": {
                stage: (
                    stages[stage].get("cache_key") if stage in stages else None
                )
                for stage in STAGE_ORDER
            },
        }
        if not stages and not complete:
            report["stage_versions"] = {stage: None for stage in STAGE_ORDER}

        self.ledger = ledger
        self.stages = stages
        self.pipeline_complete = complete
        self.pending = pending
        self._last_outcome = BatchOutcome(
            report=report,
            stages=stages,
            ledger=ledger,
            complete=complete,
            pending=pending,
            data_failures=data_failures,
            all_failures=all_failures,
            config=config,
            parsed_samples=parsed,
        )
        self._last_data_failure = data_failures
        self._last_data_fp = data_fp
        self._last_raw_fp = raw_fp
        return self._last_outcome

    def _commit(self) -> dict[str, Any]:
        if self._last_outcome is None:
            raise RuntimeError("no completed batch is available to commit")
        manifest = self.snapshot_store.commit(
            sample_lines=self.sample_store.lines,
            raw_config=self.raw_config,
            applied_batches=self.applied_batches,
            feature_row_cache=self.feature_row_cache,
            inference_row_cache=self.inference_row_cache,
            stages=self.stages if self.pipeline_complete else {},
            ledger=self.ledger,
            pipeline_complete=self.pipeline_complete,
            pending=self.pending,
            config_hash=self._last_outcome.config.config_hash,
            data_fingerprint=self._last_data_fp,
            raw_data_fingerprint=self._last_raw_fp,
            sample_count=len(self._last_outcome.parsed_samples.samples),
            plan_fingerprint=self.plan_fingerprint,
        )
        self.last_manifest = manifest
        return manifest

    def _attach_snapshot(self, report: dict[str, Any], manifest: dict[str, Any]) -> None:
        report["snapshot"] = {
            "global_version": manifest.get("global_version"),
            "version_id": manifest.get("version_id"),
            "config_hash": manifest.get("config_hash"),
            "data_fingerprint": manifest.get("data_fingerprint"),
            "stage_versions": manifest.get("stage_versions"),
            "pipeline_complete": manifest.get("pipeline_complete"),
        }

    def _verify_against_full_rebuild(self, outcome: BatchOutcome) -> dict[str, Any]:
        if not outcome.complete:
            return {
                "equal": None,
                "skipped": True,
                "reason": "incremental_run_has_pending_stages",
                "diffs": [],
            }
        temp_root = Path(tempfile.mkdtemp(prefix="domain-pipeline-full-"))
        try:
            full_store = SnapshotStore(temp_root / "snapshots")
            full_runner = PipelineRunner(
                snapshot_store=full_store,
                sample_store=SampleStore.from_lines(self.sample_store.lines),
                raw_config=copy.deepcopy(self.raw_config),
                pipeline_complete=False,
            )
            full_outcome = full_runner.process_current(
                batch_id=f"full-{outcome.report['batch_id']}",
                mode="full_rebuild",
            )
            if not full_outcome.complete:
                return {
                    "equal": None,
                    "skipped": True,
                    "reason": "full_rebuild_has_pending_stages",
                    "diffs": [],
                }
            left = canonical_comparison_values(
                outcome.stages,
                outcome.data_failures,
                outcome.ledger,
                pipeline_complete=outcome.complete,
            )
            right = canonical_comparison_values(
                full_outcome.stages,
                full_outcome.data_failures,
                full_outcome.ledger,
                pipeline_complete=full_outcome.complete,
            )
            diffs = _diff_paths(left, right)
            return {
                "equal": not diffs,
                "skipped": False,
                "reason": "exact_artifact_dependency_and_data_failure_match"
                if not diffs
                else "canonical_mismatch",
                "diffs": diffs[:50],
            }
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def run_plan(self, plan: EditPlan, *, verify_full: bool = False) -> RunResult:
        self.plan_fingerprint = plan.fingerprint
        reports: list[dict[str, Any]] = []
        unprocessed: list[str] = []
        batches = plan.batches
        if not batches:
            outcome = self.process_current(
                batch_id="run-0001",
                plan_failures=plan.failures,
                mode="incremental",
            )
            if verify_full:
                outcome.report["full_rebuild"] = self._verify_against_full_rebuild(
                    outcome
                )
            manifest = self._commit()
            self._attach_snapshot(outcome.report, manifest)
            reports.append(outcome.report)
            return RunResult(
                reports=reports,
                manifest=manifest,
                applied_batches=list(self.applied_batches),
            )

        stopped_for_pending = False
        last_commit_index = -1
        for index, batch in enumerate(batches):
            if stopped_for_pending:
                unprocessed.append(batch.batch_id)
                continue
            edit_failures: list[dict[str, Any]] = []
            if batch.batch_id not in self.applied_batches:
                current_config = validate_config(self.raw_config)
                updated_config, edit_failures = apply_batch(
                    batch,
                    sample_store=self.sample_store,
                    raw_config=self.raw_config,
                    config_id_field=current_config.data["id_field"],
                )
                self.raw_config = updated_config
                self.applied_batches.append(batch.batch_id)
            outcome = self.process_current(
                batch_id=batch.batch_id,
                edit_failures=edit_failures,
                plan_failures=plan.failures,
                mode="incremental",
            )
            if verify_full:
                outcome.report["full_rebuild"] = self._verify_against_full_rebuild(
                    outcome
                )
            self._batches_since_commit += 1
            interval = int(outcome.config.snapshot["commit_interval_batches"])
            should_commit = (
                self._batches_since_commit >= interval
                or index == len(batches) - 1
                or not outcome.complete
            )
            if should_commit:
                manifest = self._commit()
                self._batches_since_commit = 0
                last_commit_index = index
                self._attach_snapshot(outcome.report, manifest)
            reports.append(outcome.report)
            if not outcome.complete:
                stopped_for_pending = True

        if last_commit_index != len(batches) - 1:
            manifest = self._commit()
            if reports:
                self._attach_snapshot(reports[-1], manifest)
        else:
            manifest = self.last_manifest
        return RunResult(
            reports=reports,
            manifest=manifest,
            applied_batches=list(self.applied_batches),
            unprocessed_batches=unprocessed,
        )
