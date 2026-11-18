from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from .calibration import build_calibration_stage
from .config import fixed_point, stage_inputs
from .data import build_data_stage
from .metrics import build_metrics_stage
from .models import build_model_stage
from .snapshot import SnapshotStore
from .util import (
    STAGE_ORDER,
    canonical_json,
    fixed_to_decimal,
    fingerprint,
    scheme_rank,
    sha256_text,
    sort_failures,
)


class PipelineError(Exception):
    def __init__(self, code: str, message: str, failures: list[dict[str, str]] | None = None):
        super().__init__(message)
        self.code = code
        self.failures = failures or []


INPUT_LABELS = {
    "data": {
        "spec_hash": "data description",
        "overrides_hash": "single-sample replacement file",
        "seed": "generation seed",
        "n_samples": "sample count",
        "features": "feature type declarations",
        "target": "target declaration",
        "split": "train/validation split",
    },
    "model": {
        "data_key": "upstream data-encoding artifact",
        "models": "model candidate configuration",
        "fixed_point": "fixed-point probability settings",
    },
    "calibration": {
        "model_key": "upstream model artifact",
        "calibrators": "calibration scheme configuration",
        "fixed_point": "fixed-point probability settings",
    },
    "metrics": {
        "calibration_key": "upstream calibration artifact",
        "data_key": "validation labels from data artifact",
        "calibration_bins": "calibration-error bin count",
        "threshold_step": "threshold grid step",
        "fixed_point": "fixed-point metric settings",
    },
}


def execute(
    bundle: dict[str, Any],
    state_dir: Path,
    stop_event: threading.Event | None = None,
    expect_version: int | None = None,
    skip_config_hash_check: bool = False,
) -> dict[str, Any]:
    stop_event = stop_event or threading.Event()
    store = SnapshotStore(state_dir)
    previous = store.load_manifest()
    if expect_version is not None:
        found_version = previous.get("global_version") if previous is not None else None
        if found_version is None or int(found_version) != int(expect_version):
            raise PipelineError(
                "E_CONFIG_MISMATCH",
                "snapshot version does not match --expect-version",
                [
                    {
                        "code": "E_CONFIG_MISMATCH",
                        "stage": "snapshot",
                        "field": "global_version",
                        "message": f"expected {expect_version}, found {found_version}",
                    }
                ],
            )
    if (
        previous is not None
        and not skip_config_hash_check
        and previous.get("config_hash") != bundle["config_hash"]
    ):
        raise PipelineError(
            "E_CONFIG_MISMATCH",
            "configuration changed outside the command interface",
            [
                {
                    "code": "E_CONFIG_MISMATCH",
                    "stage": "snapshot",
                    "field": "config_hash",
                    "message": "manifest config hash differs from the current YAML configuration",
                }
            ],
        )

    keys = stage_inputs(bundle)
    previous_version = int(previous["global_version"]) if previous else 0
    version = previous_version + 1
    snapshot_id = f"v{version:09d}"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "global_version": version,
        "snapshot_id": snapshot_id,
        "config_hash": bundle["config_hash"],
        "data_fingerprint": None,
        "config_path": str(bundle["config_path"]),
        "stages": {},
        "snapshot_policy": bundle["config"]["snapshot"],
        "previous_snapshot": previous.get("snapshot_id") if previous else None,
    }
    config = bundle["config"]
    fixed = fixed_point(config)
    workers = int(config["concurrency"]["max_workers"])
    interval = int(config["snapshot"]["commit_interval"])
    artifacts: dict[str, dict[str, Any]] = {}
    statuses: dict[str, str] = {}
    reads: dict[str, list[str]] = {}
    commits = 0
    interrupted = False

    def commit_manifest() -> None:
        manifest["data_fingerprint"] = (
            artifacts["data"].get("data_fingerprint") if "data" in artifacts else manifest.get("data_fingerprint")
        )
        store.commit_manifest(manifest)

    def run_stage(stage: str) -> dict[str, Any]:
        cache_key = keys[stage]["key"]
        previous_record = (previous or {}).get("stages", {}).get(stage)
        reused = bool(
            previous_record
            and previous_record.get("cache_key") == cache_key
            and store.load_artifact(stage, cache_key) is not None
        )
        if reused:
            payload = store.load_artifact(stage, cache_key)
            statuses[stage] = "reused"
            reads[stage] = [store.artifact_relative_path(stage, cache_key)]
            return payload
        if stage == "data":
            payload = build_data_stage(
                bundle["spec"], config, bundle["overrides"], cache_key
            )
            reads[stage] = [
                str(bundle["spec_path"]),
                str(bundle["overrides_path"]),
                "configuration: data/features/target/split",
            ]
        elif stage == "model":
            payload = build_model_stage(
                artifacts["data"], config["models"], fixed, cache_key, workers
            )
            reads[stage] = [
                store.artifact_relative_path("data", artifacts["data"]["data_key"]),
                "configuration: models",
                "artifact slice: training features and labels",
                "artifact slice: validation features only",
            ]
        elif stage == "calibration":
            payload = build_calibration_stage(
                artifacts["data"],
                artifacts["model"],
                config["calibrators"],
                fixed,
                cache_key,
                workers,
            )
            reads[stage] = [
                store.artifact_relative_path(
                    "model", artifacts["model"]["model_key"]
                ),
                "configuration: calibrators",
                "artifact slice: training labels only",
            ]
        elif stage == "metrics":
            payload = build_metrics_stage(
                artifacts["data"],
                artifacts["calibration"],
                config["evaluation"],
                fixed,
                cache_key,
            )
            reads[stage] = [
                store.artifact_relative_path(
                    "calibration", artifacts["calibration"]["calibration_key"]
                ),
                "artifact slice: validation labels",
                "configuration: evaluation",
            ]
        else:
            raise PipelineError("E_RUNTIME", f"unknown stage: {stage}")
        store.commit_artifact(stage, cache_key, payload)
        statuses[stage] = "recomputed"
        return payload

    for stage in STAGE_ORDER:
        if stop_event.is_set():
            interrupted = True
            break
        payload = run_stage(stage)
        artifacts[stage] = payload
        cache_key = keys[stage]["key"]
        manifest["stages"][stage] = {
            "cache_key": cache_key,
            "version": sha256_text(cache_key)[:16],
            "artifact": store.artifact_relative_path(stage, cache_key),
            "inputs": keys[stage]["inputs"],
            "status": statuses[stage],
        }
        if stage == "data":
            manifest["data_fingerprint"] = payload["data_fingerprint"]
        commits += 1
        if commits % interval == 0:
            commit_manifest()
        if stop_event.is_set():
            interrupted = True
            break

    if interrupted and "data" in artifacts:
        manifest["data_fingerprint"] = artifacts["data"]["data_fingerprint"]
    commit_manifest()
    audit = _build_reuse_audit(previous, keys, statuses, reads, artifacts)
    report = _build_report(
        manifest=manifest,
        audit=audit,
        artifacts=artifacts,
        statuses=statuses,
        interrupted=interrupted,
        store=store,
    )
    committed_id = store.commit_snapshot(manifest, report)
    report["snapshot"]["id"] = committed_id
    report["snapshot"]["version"] = version
    # Update the two convenience copies with the final report object.
    from .util import atomic_write_json

    atomic_write_json(store.reports_dir / f"{committed_id}.json", report)
    atomic_write_json(store.state_dir / "latest_report.json", report)
    return report


def _build_reuse_audit(
    previous: dict[str, Any] | None,
    keys: dict[str, dict[str, Any]],
    statuses: dict[str, str],
    reads: dict[str, list[str]],
    artifacts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    previous_stages = (previous or {}).get("stages", {})
    reused = [stage for stage in STAGE_ORDER if statuses.get(stage) == "reused"]
    recomputed = [stage for stage in STAGE_ORDER if statuses.get(stage) == "recomputed"]
    pending = [stage for stage in STAGE_ORDER if stage not in statuses]
    invalidated_inputs: list[dict[str, Any]] = []
    invalidated_stages: list[str] = []
    for stage in STAGE_ORDER:
        previous_record = previous_stages.get(stage)
        current_key = keys[stage]["key"]
        if previous_record and previous_record.get("cache_key") == current_key:
            continue
        invalidated_stages.append(stage)
        if not previous_record:
            for input_name in keys[stage]["inputs"]:
                invalidated_inputs.append(
                    {
                        "stage": stage,
                        "input": INPUT_LABELS[stage].get(input_name, input_name),
                        "input_key": input_name,
                        "reason": "initial_run_or_missing_stage",
                        "previous_version": None,
                        "current_version": _short(keys[stage]["inputs"][input_name]),
                    }
                )
            continue
        old_inputs = previous_record.get("inputs", {})
        for input_name, current_value in keys[stage]["inputs"].items():
            old_value = old_inputs.get(input_name)
            if canonical_json(old_value) == canonical_json(current_value):
                continue
            invalidated_inputs.append(
                {
                    "stage": stage,
                    "input": INPUT_LABELS[stage].get(input_name, input_name),
                    "input_key": input_name,
                    "reason": "content_after_upstream_dependency",
                    "previous_version": _short(old_value),
                    "current_version": _short(current_value),
                }
            )
        if (
            stage != "data"
            and not any(item["stage"] == stage for item in invalidated_inputs)
        ):
            invalidated_inputs.append(
                {
                    "stage": stage,
                    "input": "upstream_stage_artifact",
                    "input_key": "upstream_cache_key",
                    "reason": "upstream_stage_changed",
                    "previous_version": previous_record.get("version"),
                    "current_version": manifest_stage_version(keys, stage),
                }
            )
    invalidated_inputs.sort(
        key=lambda item: (
            STAGE_ORDER.index(item["stage"]),
            item["input_key"],
            item["reason"],
        )
    )
    return {
        "reused_stages": reused,
        "recomputed_stages": recomputed,
        "pending_stages": pending,
        "invalidated_stages": invalidated_stages,
        "invalidated_inputs": invalidated_inputs,
        "reads_by_stage": {stage: reads.get(stage, []) for stage in STAGE_ORDER},
        "statistics": {
            "reused_count": len(reused),
            "recomputed_count": len(recomputed),
            "pending_count": len(pending),
            "invalidated_stage_count": len(invalidated_stages),
            "invalidated_input_count": len(invalidated_inputs),
        },
        "policy": "a stage is reused only when its content-derived cache key equals the previous committed key; downstream stages never read an invalidated upstream input while skipped",
    }


def manifest_stage_version(keys: dict[str, dict[str, Any]], stage: str) -> str:
    return sha256_text(keys[stage]["key"])[:16]


def _short(value: Any) -> str | None:
    if value is None:
        return None
    return fingerprint(value)[:16]


def _build_report(
    manifest: dict[str, Any],
    audit: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    statuses: dict[str, str],
    interrupted: bool,
    store: SnapshotStore,
) -> dict[str, Any]:
    config = manifest
    data = artifacts.get("data")
    model = artifacts.get("model")
    calibration = artifacts.get("calibration")
    metrics = artifacts.get("metrics")
    models_report = []
    if model:
        for entry in model["models"]:
            raw = entry["raw_probabilities"]["validation"]
            models_report.append(
                {
                    "model_name": entry["model_name"],
                    "model_id": entry["model_id"],
                    "model_type": entry["model_type"],
                    "fold_id": entry["fold_id"],
                    "params": entry["params"],
                    "stable_key": entry["stable_key"],
                    "loss_trajectory": entry["loss_trajectory"],
                    "raw_validation_probabilities": [
                        {
                            "sample_id": sample_id,
                            "probability_fixed": probability,
                        }
                        for sample_id, probability in zip(
                            raw["sample_ids"], raw["probabilities_fixed"], strict=True
                        )
                    ],
                    "raw_training_probabilities": [
                        {
                            "sample_id": sample_id,
                            "probability_fixed": probability,
                        }
                        for sample_id, probability in zip(
                            entry["raw_probabilities"]["train"]["sample_ids"],
                            entry["raw_probabilities"]["train"]["probabilities_fixed"],
                            strict=True,
                        )
                    ],
                    "prediction_audit": entry["prediction_audit"],
                }
            )
    calibrations_report = []
    reverse_pairs = []
    decimal_places = _decimal_places_from_manifest_artifacts(artifacts)
    scale = 10**decimal_places
    if calibration:
        for result in calibration["results"]:
            records = []
            for record in result["probabilities"]:
                enriched = dict(record)
                enriched["raw_probability"] = fixed_to_decimal(
                    record["raw_probability_fixed"], scale, decimal_places
                )
                enriched["calibrated_probability"] = fixed_to_decimal(
                    record["calibrated_probability_fixed"], scale, decimal_places
                )
                records.append(enriched)
            calibrations_report.append(
                {
                    "model_name": result["model_name"],
                    "model_id": result["model_id"],
                    "model_type": result["model_type"],
                    "fold_id": result["fold_id"],
                    "scheme": result["scheme"],
                    "stable_key": result["stable_key"],
                    "fitted_parameters": result["fitted_parameters"],
                    "bin_details": result["bin_details"],
                    "probabilities": records,
                    "inversions": result["inversions"],
                }
            )
            for inversion in result["inversions"]:
                enriched = dict(inversion)
                enriched.update(
                    {"model_name": result["model_name"], "model_id": result["model_id"]}
                )
                reverse_pairs.append(enriched)
    reverse_pairs.sort(
        key=lambda item: (
            item["model_name"],
            scheme_rank(item["scheme"]),
            item["left_sample_id"],
            item["right_sample_id"],
        )
    )
    stage_versions = {
        stage: {
            "cache_key": audit_stage.get("cache_key"),
            "version": audit_stage.get("version"),
            "status": statuses.get(stage, "pending"),
        }
        for stage, audit_stage in manifest["stages"].items()
    }
    return {
        "schema_version": 1,
        "status": "interrupted" if interrupted else "ok",
        "business_code": None,
        "failures": [],
        "global_version": manifest["global_version"],
        "snapshot": {
            "id": manifest["snapshot_id"],
            "version": manifest["global_version"],
            "config_hash": manifest["config_hash"],
            "data_fingerprint": manifest["data_fingerprint"],
            "stage_versions": stage_versions,
        },
        "execution": audit,
        "models": models_report,
        "calibrations": calibrations_report,
        "scheme_metrics": metrics["scheme_metrics"] if metrics else [],
        "threshold_grid": metrics["threshold_grid"] if metrics else [],
        "reverse_pairs": reverse_pairs,
        "evaluation": metrics["evaluation"] if metrics else None,
        "snapshot_layout": store.verify_layout(),
    }


def _decimal_places_from_manifest_artifacts(artifacts: dict[str, dict[str, Any]]) -> int:
    calibration = artifacts.get("calibration")
    if calibration:
        if "fixed_point" in calibration:
            return int(calibration["fixed_point"]["decimal_places"])
    metrics = artifacts.get("metrics")
    if metrics:
        return int(metrics["evaluation"]["fixed_point"]["decimal_places"])
    return 6


def report_core(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "models": report.get("models", []),
        "calibrations": report.get("calibrations", []),
        "scheme_metrics": report.get("scheme_metrics", []),
        "threshold_grid": report.get("threshold_grid", []),
        "reverse_pairs": report.get("reverse_pairs", []),
        "failures": report.get("failures", []),
        "evaluation": report.get("evaluation"),
    }


def core_digest(report: dict[str, Any]) -> str:
    return fingerprint(report_core(report))


def failed_report(code: str, failures: list[dict[str, str]], message: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "failed",
        "business_code": code,
        "message": message,
        "failures": sort_failures(failures),
        "execution": {
            "reused_stages": [],
            "recomputed_stages": [],
            "invalidated_inputs": [],
            "statistics": {
                "reused_count": 0,
                "recomputed_count": 0,
                "invalidated_input_count": len(failures),
            },
        },
    }
