"""End-to-end deterministic pipeline orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    ENGINE_REVISION,
    PipelineConfig,
    config_for_public_report,
    failure,
    load_pipeline_config,
    normalize_failures,
)
from .cross import (
    build_cross_tasks,
    encode_combination,
)
from .metrics import attribute_boundary_cases, compute_metrics, metrics_cache_key
from .reduce import reduce_path, reduction_cache_key
from .sample import LoadedSample, generate_sample, load_sample
from .snapshot import STAGE_ORDER, SnapshotStore, StageCommit
from .standardize import dtype_for_tier, standard_cache_key, standardize_values
from .util import atomic_write_json, fingerprint

DIFF_STAGE_ORDER = ("cross_encoding", "standardization", "projection", "final_metrics")
PATH_ORDER = ("pca", "random_projection", "variance")


def _display_path(config: PipelineConfig, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(config.root))
    except ValueError:
        return str(path)


def _previous_stage_versions(store: SnapshotStore) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    manifest = store.load_global_manifest()
    if manifest is None:
        return None, {}
    stages = manifest.get("stages", {})
    versions = {
        stage: stages.get(stage, {}).get("version")
        for stage in STAGE_ORDER
        if isinstance(stages, dict)
    }
    return manifest, versions


def _stage_version_changes(
    previous: dict[str, Any], current: dict[str, Any]
) -> list[dict[str, Any]]:
    changes = []
    for stage in STAGE_ORDER:
        old = previous.get(stage)
        new = current.get(stage)
        changes.append(
            {
                "stage": stage,
                "previous_version": old,
                "current_version": new,
                "changed": old != new,
                "sort_key": f"stage_change/{stage}",
            }
        )
    return changes


def _existing_stage_state(
    config: PipelineConfig,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        store = SnapshotStore(config.paths["cache"], config.temp_dir)
        recovery_actions = store.repair()
        _, previous = _previous_stage_versions(store)
        current: dict[str, Any] = {}
        for stage in STAGE_ORDER:
            manifest = store.stage_manifest(stage)
            if manifest is not None:
                current[stage] = manifest["version"]
        return current, previous, _stage_version_changes(previous, current), recovery_actions
    except Exception:
        return {}, {}, [], []


def _failure_report(
    config: PipelineConfig,
    failures: list[dict[str, Any]],
    *,
    loaded_columns: list[str] | None = None,
    rows: int = 0,
    sample_checksum: str = "",
    effective_data_hash: str = "",
    recovery_actions: list[dict[str, Any]] | None = None,
    stage_versions: dict[str, Any] | None = None,
    previous_stage_versions: dict[str, Any] | None = None,
    stage_version_changes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    report_path = config.paths["report"]
    report = {
        "schema_version": "1.0",
        "status": "failed",
        "engine_revision": ENGINE_REVISION,
        "config_hash": config.config_hash,
        "sample_checksum": sample_checksum,
        "effective_data_hash": effective_data_hash,
        "loaded_columns": loaded_columns or [],
        "rows_loaded": rows,
        "failures": normalize_failures(failures),
        "diagnostics": {
            "stage_diffs": [],
            "stage_summary": [],
            "first_exceeded_stage": None,
            "first_exceeded_item": None,
            "boundary_cases": [],
            "reuse_ledger": [],
            "invalidation_ledger": [],
            "dependency_ledger": [],
        },
        "stage_versions": stage_versions or {},
        "previous_stage_versions": previous_stage_versions or {},
        "stage_version_changes": stage_version_changes or [],
        "recovery_actions": recovery_actions or [],
        "runtime": {
            "max_workers": config.max_workers,
            "max_observed_workers": 0,
        },
        "artifacts": {"report": _display_path(config, report_path)},
    }
    report["semantic_fingerprint"] = fingerprint(
        {
            "status": report["status"],
            "config_hash": report["config_hash"],
            "sample_checksum": report["sample_checksum"],
            "effective_data_hash": report["effective_data_hash"],
            "failures": report["failures"],
            "loaded_columns": report["loaded_columns"],
        }
    )
    atomic_write_json(report_path, report)
    return report


def _array_difference(first: Any, second: Any) -> tuple[float, float]:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"cannot compare arrays with shapes {left.shape} and {right.shape}")
    if left.size == 0:
        return 0.0, 0.0
    deltas = np.abs(left - right)
    return float(np.max(deltas)), float(np.mean(deltas))


def _diff_item(stage: str, item: str, first: Any, second: Any, tolerance: float) -> dict[str, Any]:
    maximum, mean = _array_difference(first, second)
    return {
        "stage": stage,
        "item": item,
        "max_abs_diff": maximum,
        "mean_abs_diff": mean,
        "tolerance": tolerance,
        "exceeded": maximum > tolerance,
        "sort_key": f"stage_diff/{DIFF_STAGE_ORDER.index(stage):02d}/{item}",
    }


def _stage_summaries(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for stage in DIFF_STAGE_ORDER:
        selected = [item for item in items if item["stage"] == stage]
        maximum = max((item["max_abs_diff"] for item in selected), default=0.0)
        exceeded_items = [item for item in selected if item["exceeded"]]
        summaries.append(
            {
                "stage": stage,
                "item_count": len(selected),
                "max_abs_diff": maximum,
                "exceeded": bool(exceeded_items),
                "first_exceeded_item": exceeded_items[0]["item"] if exceeded_items else None,
                "sort_key": f"stage_summary/{DIFF_STAGE_ORDER.index(stage):02d}/{stage}",
            }
        )
    return summaries


def _metric_payload(store: SnapshotStore, tier: str, path: str) -> dict[str, Any]:
    _, artifact = store.load_entry("metrics", f"metrics/{path}/tier={tier}")
    return artifact["content"]


def _build_dependency_ledger(
    cross_entries: dict[str, dict[str, Any]],
    standard_entries: dict[str, dict[str, Any]],
    reduction_entries: dict[str, dict[str, Any]],
    metric_entries: dict[str, dict[str, Any]],
    active_features: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ledger: list[dict[str, Any]] = []
    for sort_key, entry in sorted(cross_entries.items()):
        combination = sort_key.split("/")[1]
        ledger.append(
            {
                "stage": "cross_encoding",
                "sort_key": sort_key,
                "cache_key": entry["cache_key"],
                "content_fingerprint": entry["content_fingerprint"],
                "referenced_columns": combination.split("+"),
            }
        )
    for sort_key, entry in sorted(standard_entries.items()):
        ledger.append(
            {
                "stage": "standardization",
                "sort_key": sort_key,
                "cache_key": entry["cache_key"],
                "content_fingerprint": entry["content_fingerprint"],
                "referenced_cross_features": [sort_key.split("/")[1]],
            }
        )
    feature_keys = [item["feature_key"] for item in active_features]
    for sort_key, entry in sorted(reduction_entries.items()):
        ledger.append(
            {
                "stage": "dimensionality_reduction",
                "sort_key": sort_key,
                "cache_key": entry["cache_key"],
                "content_fingerprint": entry["content_fingerprint"],
                "referenced_cross_features": list(feature_keys),
            }
        )
    for sort_key, entry in sorted(metric_entries.items()):
        ledger.append(
            {
                "stage": "metrics",
                "sort_key": sort_key,
                "cache_key": entry["cache_key"],
                "content_fingerprint": entry["content_fingerprint"],
                "referenced_cross_features": list(feature_keys),
            }
        )
    return sorted(ledger, key=lambda item: (STAGE_ORDER.index(item["stage"]), item["sort_key"]))


def run_pipeline(
    config_or_path: str | Path | PipelineConfig,
    *,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Run the complete pipeline and return the persisted diagnostic report."""

    config = (
        config_or_path
        if isinstance(config_or_path, PipelineConfig)
        else load_pipeline_config(config_or_path, root=root)
    )
    if config.failures:
        preview_columns = [descriptor["name"] for descriptor in config.descriptors]
        preview_rows = config.rows
        preview_checksum = ""
        preview_data_hash = ""
        if config.rows > 0 and config.descriptors:
            preview = load_sample(config)
            if not preview.failures:
                preview_columns = sorted(preview.values)
                preview_rows = preview.rows
                preview_checksum = preview.base_checksum
                preview_data_hash = preview.effective_checksum
            else:
                try:
                    in_memory = generate_sample(
                        config,
                        target_column=str(config.raw["metrics"]["target_column"]),
                    )
                    preview_columns = sorted(in_memory["values"])
                    preview_rows = len(in_memory["target"])
                except Exception:
                    pass
        versions, previous, changes, recovery = _existing_stage_state(config)
        report = _failure_report(
            config,
            config.failures,
            loaded_columns=preview_columns,
            rows=preview_rows,
            sample_checksum=preview_checksum,
            effective_data_hash=preview_data_hash,
            recovery_actions=recovery,
            stage_versions=versions,
            previous_stage_versions=previous,
            stage_version_changes=changes,
        )
        return report

    sample: LoadedSample = load_sample(config)
    if sample.failures:
        versions, previous, changes, recovery = _existing_stage_state(config)
        return _failure_report(
            config,
            sample.failures,
            loaded_columns=sorted(sample.values),
            rows=sample.rows,
            sample_checksum=sample.base_checksum,
            effective_data_hash=sample.effective_checksum,
            recovery_actions=recovery,
            stage_versions=versions,
            previous_stage_versions=previous,
            stage_version_changes=changes,
        )

    combinations = build_cross_tasks(config, sample)
    if not combinations:
        return _failure_report(
            config,
            [failure("E_CROSS_ORDER", "crossing", "configuration produced no crossing combinations")],
            loaded_columns=sorted(sample.values),
            rows=sample.rows,
            sample_checksum=sample.base_checksum,
            effective_data_hash=sample.effective_checksum,
        )

    store = SnapshotStore(config.paths["cache"], config.temp_dir)
    recovery_actions = store.repair()
    _, previous_versions = _previous_stage_versions(store)
    max_workers = config.max_workers
    stage_commits: dict[str, StageCommit] = {}
    all_events: list[dict[str, Any]] = []
    max_observed = 0

    cross_descriptors: list[dict[str, Any]] = []
    for task in combinations:
        def cross_builder(task: dict[str, Any] = task) -> dict[str, Any]:
            result = encode_combination(
                task["columns"],
                sample.values,
                crossing=task["crossing"],
            )
            return {
                "content": result["content"],
                "metadata": {"degradation": result["degradation"]},
            }

        cross_descriptors.append(
            {
                "sort_key": task["sort_key"],
                "item": task["combination"],
                "cache_key": task["cache_key"],
                "builder": cross_builder,
            }
        )
    precision_fingerprints = {
        tier: config.precision_fingerprint(tier) for tier in ("single", "double")
    }
    cross_commit = store.commit_stage(
        "cross_encoding",
        descriptors=cross_descriptors,
        metadata={
            "engine_revision": ENGINE_REVISION,
            "crossing": config.raw["crossing"],
            "precision_fingerprints": precision_fingerprints,
        },
        max_workers=max_workers,
    )
    stage_commits["cross_encoding"] = cross_commit
    all_events.extend(cross_commit.events)
    max_observed = max(max_observed, cross_commit.max_observed_workers)

    cross_entries = cross_commit.entries
    feature_records: list[dict[str, Any]] = []
    degradations: list[dict[str, Any]] = []
    active_features: list[dict[str, Any]] = []
    cross_by_feature_tier: dict[str, dict[str, dict[str, Any]]] = {"single": {}, "double": {}}
    for sort_key in sorted(
        (key for key in cross_entries if key.endswith("tier=double")),
        key=lambda key: key.split("/")[1],
    ):
        double_entry = cross_entries[sort_key]
        double_artifact = store.load_entry("cross_encoding", sort_key)[1]
        single_key = sort_key.replace("tier=double", "tier=single")
        single_entry = cross_entries[single_key]
        single_artifact = store.load_entry("cross_encoding", single_key)[1]
        double_content = double_artifact["content"]
        single_content = single_artifact["content"]
        if double_content["encoded"] != single_content["encoded"]:
            return _failure_report(
                config,
                [failure("E_DATA", sort_key, "cross encoding differs between precision tiers")],
                loaded_columns=sorted(sample.values),
                rows=sample.rows,
                sample_checksum=sample.base_checksum,
                effective_data_hash=sample.effective_checksum,
                recovery_actions=recovery_actions,
            )
        feature_key = double_content["feature_key"]
        cross_by_feature_tier["single"][feature_key] = single_artifact
        cross_by_feature_tier["double"][feature_key] = double_artifact
        feature_record = {
            **double_content,
            "tier_content_fingerprints": {
                "single": single_entry["content_fingerprint"],
                "double": double_entry["content_fingerprint"],
            },
            "sort_key": sort_key,
        }
        if double_content["encoding_mode"] != "skipped":
            active_features.append(feature_record)
        feature_records.append(feature_record)
        degradation = double_artifact.get("metadata", {}).get("degradation")
        if degradation is not None:
            degradations.append(degradation)
    feature_records.sort(key=lambda item: item["feature_key"])
    degradations.sort(key=lambda item: item["sort_key"])
    active_features.sort(key=lambda item: item["feature_key"])
    if not active_features:
        return _failure_report(
            config,
            [failure("E_CROSS_ORDER", "crossing", "all crossing combinations were skipped")],
            loaded_columns=sorted(sample.values),
            rows=sample.rows,
            sample_checksum=sample.base_checksum,
            effective_data_hash=sample.effective_checksum,
            recovery_actions=recovery_actions,
        )

    standard_descriptors: list[dict[str, Any]] = []
    for tier in ("single", "double"):
        for feature in active_features:
            feature_key = feature["feature_key"]
            artifact = cross_by_feature_tier[tier][feature_key]
            upstream_fp = fingerprint(artifact["content"])
            cache_key = standard_cache_key(config, tier, feature_key, upstream_fp)

            def standard_builder(
                feature_key: str = feature_key,
                artifact: dict[str, Any] = artifact,
                tier: str = tier,
            ) -> dict[str, Any]:
                return standardize_values(
                    feature_key,
                    artifact["content"]["encoded"],
                    tier=tier,
                    method=str(config.raw["standardization"]["method"]),
                    epsilon=float(config.raw["standardization"]["epsilon"]),
                )

            standard_descriptors.append(
                {
                    "sort_key": f"standardization/{feature_key}/tier={tier}",
                    "item": feature_key,
                    "cache_key": cache_key,
                    "builder": standard_builder,
                }
            )
    standard_commit = store.commit_stage(
        "standardization",
        descriptors=standard_descriptors,
        metadata={
            "engine_revision": ENGINE_REVISION,
            "standardization": config.raw["standardization"],
            "precision_fingerprints": precision_fingerprints,
        },
        max_workers=max_workers,
    )
    stage_commits["standardization"] = standard_commit
    all_events.extend(standard_commit.events)
    max_observed = max(max_observed, standard_commit.max_observed_workers)
    standard_entries = standard_commit.entries

    standard_payloads: dict[str, dict[str, dict[str, Any]]] = {"single": {}, "double": {}}
    standard_stats: list[dict[str, Any]] = []
    matrices: dict[str, np.ndarray] = {}
    feature_keys = [feature["feature_key"] for feature in active_features]
    for tier in ("single", "double"):
        columns = []
        for feature_key in feature_keys:
            sort_key = f"standardization/{feature_key}/tier={tier}"
            entry, artifact = store.load_entry("standardization", sort_key)
            standard_payloads[tier][feature_key] = artifact["content"]
            columns.append(np.asarray(artifact["content"]["values"], dtype=dtype_for_tier(tier)))
            if tier == "double":
                standard_stats.append(
                    {
                        "feature_key": feature_key,
                        "mean": artifact["content"]["mean"],
                        "std": artifact["content"]["std"],
                        "constant": artifact["content"]["constant"],
                        "content_fingerprints": {
                            "single": standard_entries[f"standardization/{feature_key}/tier=single"][
                                "content_fingerprint"
                            ],
                            "double": entry["content_fingerprint"],
                        },
                    }
                )
        matrices[tier] = np.column_stack(columns)

    reduction_settings = config.raw["reduction"]
    reduction_descriptors: list[dict[str, Any]] = []
    upstream_standard = [
        {
            "feature_key": feature_key,
            "single": standard_entries[f"standardization/{feature_key}/tier=single"]["content_fingerprint"],
            "double": standard_entries[f"standardization/{feature_key}/tier=double"]["content_fingerprint"],
        }
        for feature_key in feature_keys
    ]
    for tier in ("single", "double"):
        tier_upstream = [
            {"feature_key": item["feature_key"], "content_fingerprint": item[tier]}
            for item in upstream_standard
        ]
        for path in PATH_ORDER:
            settings = dict(reduction_settings[path])
            cache_key = reduction_cache_key(config, tier, path, tier_upstream, settings)

            def reduction_builder(
                path: str = path,
                tier: str = tier,
                settings: dict[str, Any] = settings,
            ) -> dict[str, Any]:
                return reduce_path(
                    path,
                    matrices[tier].copy(),
                    list(feature_keys),
                    settings,
                    config,
                )

            reduction_descriptors.append(
                {
                    "sort_key": f"dimensionality_reduction/{path}/tier={tier}",
                    "item": path,
                    "cache_key": cache_key,
                    "builder": reduction_builder,
                }
            )
    reduction_commit = store.commit_stage(
        "dimensionality_reduction",
        descriptors=reduction_descriptors,
        metadata={
            "engine_revision": ENGINE_REVISION,
            "precision_fingerprints": precision_fingerprints,
        },
        max_workers=max_workers,
    )
    stage_commits["dimensionality_reduction"] = reduction_commit
    all_events.extend(reduction_commit.events)
    max_observed = max(max_observed, reduction_commit.max_observed_workers)
    reduction_entries = reduction_commit.entries

    reduction_payloads: dict[str, dict[str, dict[str, Any]]] = {"single": {}, "double": {}}
    for tier in ("single", "double"):
        for path in PATH_ORDER:
            _, artifact = store.load_entry("dimensionality_reduction", f"dimensionality_reduction/{path}/tier={tier}")
            reduction_payloads[tier][path] = artifact["content"]

    target_fingerprint = fingerprint(sample.target)
    metric_descriptors: list[dict[str, Any]] = []
    for tier in ("single", "double"):
        for path in PATH_ORDER:
            sort_key = f"metrics/{path}/tier={tier}"
            reduction_entry = reduction_entries[f"dimensionality_reduction/{path}/tier={tier}"]
            cache_key = metrics_cache_key(
                config,
                tier,
                path,
                reduction_entry["content_fingerprint"],
                target_fingerprint,
            )

            def metrics_builder(
                tier: str = tier,
                path: str = path,
            ) -> dict[str, Any]:
                content = reduction_payloads[tier][path]
                return compute_metrics(
                    content["projected"],
                    content["projected_raw"],
                    sample.target,
                    tier=tier,
                    path=path,
                    threshold=float(config.raw["metrics"]["decision_threshold"]),
                )

            metric_descriptors.append(
                {
                    "sort_key": sort_key,
                    "item": path,
                    "cache_key": cache_key,
                    "builder": metrics_builder,
                }
            )
    metrics_commit = store.commit_stage(
        "metrics",
        descriptors=metric_descriptors,
        metadata={
            "engine_revision": ENGINE_REVISION,
            "metrics": config.raw["metrics"],
            "target_fingerprint": target_fingerprint,
            "precision_fingerprints": precision_fingerprints,
        },
        max_workers=max_workers,
    )
    stage_commits["metrics"] = metrics_commit
    all_events.extend(metrics_commit.events)
    max_observed = max(max_observed, metrics_commit.max_observed_workers)
    metric_entries = metrics_commit.entries

    stage_versions = {
        stage: stage_commits[stage].version for stage in STAGE_ORDER
    }
    stage_version_changes = _stage_version_changes(previous_versions, stage_versions)
    global_version = fingerprint(
        {
            "engine_revision": ENGINE_REVISION,
            "config_hash": config.config_hash,
            "sample_checksum": sample.base_checksum,
            "effective_data_hash": sample.effective_checksum,
            "target_fingerprint": target_fingerprint,
            "precision_fingerprints": precision_fingerprints,
            "stages": stage_versions,
        }
    )

    tolerance = config.stage_tolerance()
    stage_diffs: list[dict[str, Any]] = []
    for feature in feature_records:
        encrypted_single = cross_by_feature_tier["single"][feature["feature_key"]]["content"]["encoded"]
        encrypted_double = cross_by_feature_tier["double"][feature["feature_key"]]["content"]["encoded"]
        stage_diffs.append(
            _diff_item("cross_encoding", feature["feature_key"], encrypted_single, encrypted_double, tolerance)
        )
    for feature_key in feature_keys:
        stage_diffs.append(
            _diff_item(
                "standardization",
                feature_key,
                standard_payloads["single"][feature_key]["values"],
                standard_payloads["double"][feature_key]["values"],
                tolerance,
            )
        )
    for path in PATH_ORDER:
        stage_diffs.append(
            _diff_item(
                "projection",
                f"{path}/raw",
                reduction_payloads["single"][path]["projected_raw"],
                reduction_payloads["double"][path]["projected_raw"],
                tolerance,
            )
        )
        stage_diffs.append(
            _diff_item(
                "projection",
                f"{path}/quantized",
                reduction_payloads["single"][path]["projected"],
                reduction_payloads["double"][path]["projected"],
                tolerance,
            )
        )
    for path in PATH_ORDER:
        single_metrics = _metric_payload(store, "single", path)
        double_metrics = _metric_payload(store, "double", path)
        for metric_name in sorted(single_metrics["metrics"]):
            stage_diffs.append(
                _diff_item(
                    "final_metrics",
                    f"{path}/{metric_name}",
                    [single_metrics["metrics"][metric_name]],
                    [double_metrics["metrics"][metric_name]],
                    tolerance,
                )
            )
        stage_diffs.append(
            _diff_item(
                "final_metrics",
                f"{path}/decisions",
                single_metrics["decisions"],
                double_metrics["decisions"],
                tolerance,
            )
        )
    stage_diffs.sort(key=lambda item: item["sort_key"])
    stage_summary = _stage_summaries(stage_diffs)
    first_stage = next((item for item in stage_summary if item["exceeded"]), None)

    boundary_cases: list[dict[str, Any]] = []
    for path in PATH_ORDER:
        single_metrics = _metric_payload(store, "single", path)
        double_metrics = _metric_payload(store, "double", path)
        boundary_cases.extend(
            attribute_boundary_cases(path=path, single=single_metrics, double=double_metrics)
        )
    boundary_cases.sort(key=lambda item: item["sort_key"])

    reduction_report: list[dict[str, Any]] = []
    for path in PATH_ORDER:
        double_content = reduction_payloads["double"][path]
        reduction_report.append(
            {
                "path": path,
                "retained_dimensions": double_content["retained_dimensions"],
                "output_feature_names": double_content["output_feature_names"],
                "input_feature_keys": double_content["input_feature_keys"],
                "included_cross_feature_keys": list(feature_keys),
                "summary": double_content["summary"],
                "quantization": double_content["quantization"],
                "cache_keys": {
                    tier: reduction_entries[f"dimensionality_reduction/{path}/tier={tier}"][
                        "cache_key"
                    ]
                    for tier in ("single", "double")
                },
                "content_fingerprints": {
                    tier: reduction_entries[f"dimensionality_reduction/{path}/tier={tier}"][
                        "content_fingerprint"
                    ]
                    for tier in ("single", "double")
                },
                "tiers": {
                    tier: {
                        "projected": reduction_payloads[tier][path]["projected"],
                        "projected_raw": reduction_payloads[tier][path]["projected_raw"],
                        "metrics": _metric_payload(store, tier, path)["metrics"],
                        "decisions": _metric_payload(store, tier, path)["decisions"],
                        "scores": _metric_payload(store, tier, path)["scores"],
                    }
                    for tier in ("single", "double")
                },
                "sort_key": f"reduction/{path}",
            }
        )

    reuse_ledger = [
        event for event in all_events if event["action"] == "reused"
    ]
    invalidation_ledger = [
        event for event in all_events if event["action"] != "reused"
    ]
    reuse_ledger.sort(key=lambda item: (STAGE_ORDER.index(item["stage"]), item["sort_key"]))
    invalidation_ledger.sort(
        key=lambda item: (STAGE_ORDER.index(item["stage"]), item["sort_key"], item["action"])
    )
    dependency_ledger = _build_dependency_ledger(
        cross_entries,
        standard_entries,
        reduction_entries,
        metric_entries,
        active_features,
    )

    report_path = config.paths["report"]
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "ok",
        "engine_revision": ENGINE_REVISION,
        "config": config_for_public_report(config),
        "config_hash": config.config_hash,
        "sample_checksum": sample.base_checksum,
        "effective_data_hash": sample.effective_checksum,
        "target_fingerprint": target_fingerprint,
        "global_version": global_version,
        "precision_fingerprints": precision_fingerprints,
        "stage_versions": stage_versions,
        "previous_stage_versions": previous_versions,
        "stage_version_changes": stage_version_changes,
        "snapshots": {
            stage: {
                "version": stage_commits[stage].version,
                "replaced": stage_commits[stage].replaced,
                "item_count": len(stage_commits[stage].manifest["items"]),
            }
            for stage in STAGE_ORDER
        },
        "cross_encoding": {
            "order": int(config.raw["crossing"]["order"]),
            "whitelist": list(config.valid_whitelist),
            "cardinality_budget": int(config.raw["crossing"]["cardinality_budget"]),
            "overflow_strategy": config.raw["crossing"]["overflow_strategy"],
            "feature_count": len(feature_records),
            "active_feature_count": len(active_features),
            "features": feature_records,
            "degradations": degradations,
        },
        "standardization": {
            "method": config.raw["standardization"]["method"],
            "features": standard_stats,
        },
        "reduction": {
            "paths": reduction_report,
            "path_order": list(PATH_ORDER),
        },
        "diagnostics": {
            "stage_diffs": stage_diffs,
            "stage_summary": stage_summary,
            "first_exceeded_stage": first_stage["stage"] if first_stage else None,
            "first_exceeded_item": first_stage["first_exceeded_item"] if first_stage else None,
            "boundary_cases": boundary_cases,
            "reuse_ledger": reuse_ledger,
            "invalidation_ledger": invalidation_ledger,
            "dependency_ledger": dependency_ledger,
            "cache_summary": {
                "reused": len(reuse_ledger),
                "invalidated_or_recomputed": len(invalidation_ledger),
                "removed": int(sum(item["action"] == "removed" for item in invalidation_ledger)),
            },
        },
        "failures": [],
        "recovery_actions": recovery_actions,
        "runtime": {
            "max_workers": max_workers,
            "max_observed_workers": max_observed,
        },
        "artifacts": {
            "report": _display_path(config, report_path),
            "cache_manifest": _display_path(config, store.manifest_path),
            "cache_root": _display_path(config, config.paths["cache"]),
        },
    }
    semantic_payload = {
        "engine_revision": ENGINE_REVISION,
        "config_hash": report["config_hash"],
        "sample_checksum": report["sample_checksum"],
        "effective_data_hash": report["effective_data_hash"],
        "target_fingerprint": target_fingerprint,
        "global_version": global_version,
        "stage_versions": stage_versions,
        "cross_encoding": report["cross_encoding"],
        "standardization": report["standardization"],
        "reduction": report["reduction"],
        "stage_diffs": stage_diffs,
        "stage_summary": stage_summary,
        "first_exceeded_stage": report["diagnostics"]["first_exceeded_stage"],
        "first_exceeded_item": report["diagnostics"]["first_exceeded_item"],
        "boundary_cases": boundary_cases,
        "failures": [],
    }
    report["semantic_fingerprint"] = fingerprint(semantic_payload)
    report["event_order_fingerprint"] = fingerprint(
        [[event["stage"], event["sort_key"]] for event in all_events]
    )

    global_manifest = {
        "schema_version": 1,
        "engine_revision": ENGINE_REVISION,
        "global_version": global_version,
        "config_hash": config.config_hash,
        "sample_checksum": sample.base_checksum,
        "effective_data_hash": sample.effective_checksum,
        "target_fingerprint": target_fingerprint,
        "precision_fingerprints": precision_fingerprints,
        "stages": {
            stage: {
                "version": stage_commits[stage].version,
                "replaced": stage_commits[stage].replaced,
                "item_count": len(stage_commits[stage].manifest["items"]),
            }
            for stage in STAGE_ORDER
        },
    }
    store.write_global_manifest(global_manifest)
    atomic_write_json(report_path, report)
    return report
