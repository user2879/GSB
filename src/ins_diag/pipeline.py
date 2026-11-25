from __future__ import annotations

import os
import signal
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .cache import PersistentCache, content_cache_key
from .config import ConfigValidationError, EffectiveConfig, load_config
from .errors import BusinessError, FailureCollector, LeaseExpiredError
from .features import (
    compute_standardization,
    extract_window_features,
    failure_to_error,
    window_set_fingerprint_ids,
)
from .fixedpoint import FixedPoint
from .kde import run_kde
from .kmeans import run_fixed_kmeans
from .lease import Clock, LeaseManager, LeaseRecord
from .snapshot import SnapshotStore
from .util import atomic_write_json, fingerprint
from .waveform import apply_edit_commands, synthesize_waveform
from .windows import WindowRef, build_windows


CRASH_STAGES = {"after_lease", "after_cluster", "before_snapshot"}


def _base_report(config: EffectiveConfig | None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "running",
        "config": (
            {
                "path": str(config.config_path),
                "hash": config.config_hash,
            }
            if config
            else None
        ),
        "data": {},
        "edits": {"applied": [], "failures": []},
        "recovery": {},
        "windows": [],
        "failed_windows": [],
        "standardization": None,
        "clusters": None,
        "density": None,
        "boundary_windows": [],
        "lease": None,
        "snapshot": None,
        "cache": None,
        "failures": [],
        "diagnostics": [],
        "business_errors": [],
    }


def config_error_report(error: ConfigValidationError) -> dict[str, Any]:
    report = _base_report(None)
    report["status"] = "error"
    report["failures"] = [issue.to_dict() for issue in error.issues]
    report["business_errors"] = [issue.to_dict() for issue in error.issues]
    report["error_codes"] = sorted({issue.code for issue in error.issues})
    return report


def _maybe_crash(stage: str | None, current_stage: str) -> None:
    if stage == current_stage:
        os.kill(os.getpid(), signal.SIGKILL)


def _config_payload(config: EffectiveConfig) -> dict[str, Any]:
    return {
        "schema_version": config.schema_version,
        "seed": config.seed,
        "window": {"length": config.window_length, "shift": config.window_shift},
        "features": {
            "bands_hz": [[band.low_hz, band.high_hz] for band in config.bands]
        },
        "standardization": {"dispersion": config.dispersion},
        "clustering": {
            "families": config.families,
            "family_limit": config.family_limit,
            "max_iterations": config.max_iterations,
            "fixed_point": asdict(config.fixed_point),
        },
        "kde": {
            "bandwidth_label": config.bandwidth_label,
            "bandwidth": config.bandwidth,
            "threshold_permille": config.threshold_permille,
        },
        "runtime": {"max_workers": config.max_workers},
        "lease": asdict(config.lease),
    }


def _stable_error_dedupe(errors: list[BusinessError]) -> list[BusinessError]:
    collector = FailureCollector()
    collector.extend(errors)
    return collector.sorted()


def _window_common(window: WindowRef) -> dict[str, Any]:
    return window.to_dict()


def _lease_signature(records: list[LeaseRecord]) -> dict[str, int]:
    return {record.lease_id: record.generation for record in records}


def _lease_failure(record: LeaseRecord, reason: str) -> BusinessError:
    return BusinessError(
        code="E_LEASE_EXPIRED",
        path=f"leases.{record.lease_id}",
        message=reason,
        stage="lease",
    )


def _validate_leases(
    manager: LeaseManager,
    records: list[LeaseRecord],
    failures: FailureCollector,
) -> tuple[bool, list[dict[str, Any]]]:
    expired = manager.expire_due()
    for item in expired:
        failures.add(
            BusinessError(
                code="E_LEASE_EXPIRED",
                path=f"leases.{item['lease_id']}.expires_ms",
                message="lease expired and was reclaimed before artifact commit",
                stage="lease",
            )
        )
    valid = True
    for record in records:
        if not manager.artifact_is_valid(record.lease_id, record.generation):
            valid = False
            failures.add(
                _lease_failure(
                    record,
                    "cluster/density artifacts produced under this lease are invalid",
                )
            )
    return valid, expired


def run_pipeline(
    config_path: str | Path,
    *,
    state_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    crash_after: str | None = None,
    edits_limit: int | None = None,
    clock: Clock | None = None,
) -> dict[str, Any]:
    if crash_after is not None and crash_after not in CRASH_STAGES:
        raise ValueError(f"unsupported crash stage: {crash_after}")
    try:
        config = load_config(config_path)
    except ConfigValidationError as error:
        report = config_error_report(error)
        if output_path is not None:
            atomic_write_json(Path(output_path), report)
        return report

    if state_dir is not None:
        config = replace(config, state_dir=Path(state_dir).resolve())
    config.state_dir.mkdir(parents=True, exist_ok=True)

    selected_edits = config.edits if edits_limit is None else config.edits[:edits_limit]
    dataset = synthesize_waveform(config.waveform, config.seed)
    try:
        dataset, effective_config, edit_failures, applied_edits = apply_edit_commands(
            dataset, config, selected_edits
        )
    except ConfigValidationError as error:
        report = config_error_report(error)
        report["config"] = {
            "path": str(config.config_path),
            "hash": config.config_hash,
        }
        if output_path is not None:
            atomic_write_json(Path(output_path), report)
        return report
    effective_config = replace(effective_config, state_dir=config.state_dir)
    data_fingerprint = dataset.content_fingerprint()
    windows = build_windows(dataset, effective_config)
    window_set_fp = window_set_fingerprint_ids(
        [
            {
                "window_index": window.window_index,
                "window_id": window.window_id,
                "content_fingerprint": window.content_fingerprint(),
            }
            for window in windows
        ],
        effective_config,
    )

    report = _base_report(effective_config)
    report["config"] = {
        "path": str(effective_config.config_path),
        "hash": effective_config.config_hash,
        "effective": _config_payload(effective_config),
        "edits_limit": edits_limit,
    }
    report["data"] = {
        "fingerprint": data_fingerprint,
        "seed": dataset.seed,
        "sample_rate_hz": effective_config.waveform.sample_rate_hz,
        "channel_count": len(dataset.channels),
        "channels": [
            {
                "device_id": channel.device_id,
                "channel_id": channel.channel_id,
                "samples": int(channel.samples.size),
                "present_samples": int(channel.present.sum()),
            }
            for channel in dataset.channels
        ],
        "window_count": len(windows),
        "window_set_fingerprint": window_set_fp,
    }
    report["edits"] = {"applied": applied_edits, "failures": []}

    failures = FailureCollector()
    failures.extend(edit_failures)
    diagnostics: list[BusinessError] = []
    fatal = False

    lease_manager = LeaseManager(
        effective_config.lease,
        effective_config.state_dir,
        clock=clock,
    )
    snapshot_store = SnapshotStore(effective_config.state_dir, clock=clock)
    prior_version = snapshot_store.current_global_version()
    persisted_lease_high_water_before = lease_manager.high_water_generation
    restore_validation = snapshot_store.evaluate_restore(
        expected_global_version=prior_version,
        expected_config_hash=effective_config.config_hash,
        expected_data_fingerprint=data_fingerprint,
        expected_lease_generation=lease_manager.high_water_generation,
    )
    diagnostics.extend(restore_validation.errors)

    snapshot_generation = None
    existing_snapshot = snapshot_store.read()
    if existing_snapshot is not None:
        value = existing_snapshot.get("lease_generation")
        if isinstance(value, int) and not isinstance(value, bool):
            snapshot_generation = value
    if (
        snapshot_generation is not None
        and snapshot_generation > lease_manager.high_water_generation
    ):
        rollback = BusinessError(
            code="E_LEASE_GENERATION_ROLLBACK",
            path="snapshot.lease_generation",
            message=(
                "snapshot references a lease generation ahead of persisted high-water "
                f"state ({snapshot_generation} > {lease_manager.high_water_generation})"
            ),
            stage="lease",
        )
        diagnostics.append(rollback)
        fatal = True
    if lease_manager.rollback_error is not None:
        failures.add(lease_manager.rollback_error)
        fatal = True

    recovery_scope = "empty"
    if restore_validation.snapshot_present:
        recovery_scope = "snapshot_exact" if restore_validation.restoreable else "cache_only_superseded"
    if lease_manager.rollback_error is not None or fatal:
        recovery_scope = "blocked_lease_generation_rollback"

    cache = PersistentCache(effective_config.state_dir)
    raw_by_index: dict[int, dict[str, Any]] = {}
    raw_keys_by_index: dict[int, str] = {}
    feature_misses: list[WindowRef] = []
    for window in windows:
        raw_key = content_cache_key(
            "raw_window_feature",
            {
                "window_content": window.content_fingerprint(),
                "feature_parameter_fingerprint": effective_config.feature_parameter_fingerprint(),
            },
        )
        raw_keys_by_index[window.window_index] = raw_key
        cached = cache.lookup(
            kind="raw_window_feature",
            key=raw_key,
            dependencies=[],
            miss_reason="raw_window_content_or_feature_parameters_changed",
            sort_key=(window.window_index, "10-raw-window-feature"),
            window_index=window.window_index,
            path=window.window_id,
        )
        if cached is not None:
            normalized = dict(cached)
            normalized["window_index"] = window.window_index
            normalized["window_id"] = window.window_id
            raw_by_index[window.window_index] = normalized
        else:
            feature_misses.append(window)

    with ThreadPoolExecutor(
        max_workers=effective_config.max_workers,
        thread_name_prefix="ins-diag",
    ) as executor:
        futures = []
        for window in feature_misses:
            key = (window.window_index, "10-raw-window-feature")
            future = executor.submit(extract_window_features, window, effective_config)
            futures.append((key, window, future))
        futures.sort(key=lambda item: item[0])
        for _, window, future in futures:
            try:
                payload = future.result()
            except Exception as exc:
                if hasattr(exc, "code") and hasattr(exc, "path") and hasattr(exc, "message"):
                    failures.add(failure_to_error(exc, window))
                else:
                    failures.add(
                        BusinessError(
                            code="E_WINDOW_PROCESSING",
                            path=f"windows[{window.window_index}]",
                            message=str(exc),
                            stage="window",
                            window_id=window.window_id,
                        )
                    )
                continue
            raw_by_index[window.window_index] = payload
            cache.put(
                kind="raw_window_feature",
                key=raw_keys_by_index[window.window_index],
                payload=payload,
                dependencies=[],
                window_index=window.window_index,
            )
        cache.save()

        ordered_raw = [raw_by_index[index] for index in sorted(raw_by_index)]
        successful_raw_keys = [
            raw_keys_by_index[int(item["window_index"])] for item in ordered_raw
        ]
        stats: dict[str, Any] | None = None
        stats_key: str | None = None
        observations: dict[int, list[float]] = {}
        obs_keys_by_index: dict[int, str] = {}

        if ordered_raw:
            stats_key = content_cache_key(
                "standardization_stats",
                {
                    "window_set": [
                        {
                            "window_index": item["window_index"],
                            "raw_cache_key": raw_keys_by_index[int(item["window_index"])],
                            "content_fingerprint": item["content_fingerprint"],
                        }
                        for item in ordered_raw
                    ],
                    "standardization_parameter_fingerprint": effective_config.standardization_fingerprint(),
                },
            )
            cached_stats = cache.lookup(
                kind="standardization_stats",
                key=stats_key,
                dependencies=successful_raw_keys,
                miss_reason="standardization_window_set_or_parameters_changed",
                sort_key=(-1, "20-standardization-stats"),
                window_index=None,
                path="standardization",
            )
            if cached_stats is not None:
                stats = cached_stats
            else:
                stats = compute_standardization(ordered_raw, effective_config)
                cache.put(
                    kind="standardization_stats",
                    key=stats_key,
                    payload=stats,
                    dependencies=successful_raw_keys,
                )

            for item in ordered_raw:
                window_index = int(item["window_index"])
                raw_key = raw_keys_by_index[window_index]
                obs_key = content_cache_key(
                    "standardized_observation",
                    {
                        "raw_cache_key": raw_key,
                        "statistics_cache_key": stats_key,
                        "quantization_parameter_fingerprint": effective_config.quantization_fingerprint(),
                    },
                )
                obs_keys_by_index[window_index] = obs_key
                cached_obs = cache.lookup(
                    kind="standardized_observation",
                    key=obs_key,
                    dependencies=[raw_key, stats_key],
                    miss_reason="upstream_feature_statistics_or_quantization_changed",
                    sort_key=(window_index, "30-standardized-observation"),
                    window_index=window_index,
                    path=item["window_id"],
                )
                if cached_obs is not None:
                    observations[window_index] = list(cached_obs["vector"])
                else:
                    vector = [
                        (float(value) - float(stats["means"][dimension_index]))
                        / float(stats["dispersions"][dimension_index])
                        for dimension_index, value in enumerate(item["values"])
                    ]
                    observations[window_index] = vector
                    cache.put(
                        kind="standardized_observation",
                        key=obs_key,
                        payload={"window_index": window_index, "vector": vector},
                        dependencies=[raw_key, stats_key],
                        window_index=window_index,
                    )
            cache.save()

        if not windows:
            failures.add(
                BusinessError(
                    code="E_NO_WINDOWS",
                    path="data.windows",
                    message="no analysis windows were produced",
                    stage="window",
                )
            )
            fatal = True
        elif not ordered_raw:
            failures.add(
                BusinessError(
                    code="E_NO_VALID_WINDOWS",
                    path="data.valid_windows",
                    message="all analysis windows failed feature extraction",
                    stage="window",
                )
            )
            fatal = True
        if effective_config.families > len(ordered_raw):
            failures.add(
                BusinessError(
                    code="E_CLUSTER_COUNT_EXCEEDS_WINDOWS",
                    path="config.clustering.families",
                    message=(
                        f"family count {effective_config.families} exceeds valid windows "
                        f"{len(ordered_raw)}"
                    ),
                    stage="cluster",
                )
            )
            fatal = True

        cluster_payload: dict[str, Any] | None = None
        density_payload: dict[str, Any] | None = None
        lease_records: list[LeaseRecord] = []
        lease_valid = False
        lease_map: dict[str, int] = {}
        lease_generation = lease_manager.high_water_generation
        expired_leases: list[dict[str, Any]] = []

        if not fatal and ordered_raw:
            lease_records = lease_manager.issue_for_clusters(effective_config.families)
            lease_map = _lease_signature(lease_records)
            lease_generation = max(record.generation for record in lease_records)
            _maybe_crash(crash_after, "after_lease")
            obs_keys = [obs_keys_by_index[index] for index in sorted(obs_keys_by_index)]
            cluster_key = content_cache_key(
                "cluster_model",
                {
                    "observation_cache_keys": obs_keys,
                    "model_parameter_fingerprint": effective_config.model_fingerprint(),
                    "lease_generations": lease_map,
                },
            )
            cached_cluster = cache.lookup(
                kind="cluster_model",
                key=cluster_key,
                dependencies=obs_keys,
                miss_reason="observation_set_model_or_lease_generation_changed",
                sort_key=(-1, "40-cluster-model"),
                window_index=None,
                path="clusters",
            )
            if cached_cluster is not None:
                cluster_payload = cached_cluster
            else:
                try:
                    cluster_payload = run_fixed_kmeans(
                        observations, effective_config, executor
                    )
                    cache.put(
                        kind="cluster_model",
                        key=cluster_key,
                        payload=cluster_payload,
                        dependencies=obs_keys,
                        lease_generation=lease_generation,
                    )
                except ValueError as exc:
                    failures.add(
                        BusinessError(
                            code="E_CLUSTERING_FAILED",
                            path="clusters",
                            message=str(exc),
                            stage="cluster",
                        )
                    )
                    cluster_payload = None
                    fatal = True
            cache.save()
            _maybe_crash(crash_after, "after_cluster")

            lease_valid, expired_leases = _validate_leases(
                lease_manager, lease_records, failures
            )
            if not lease_valid:
                fatal = True

            if lease_valid and cluster_payload is not None:
                density_key = content_cache_key(
                    "density_model",
                    {
                        "observation_cache_keys": obs_keys,
                        "kde_parameter_fingerprint": effective_config.kde_fingerprint(),
                        "lease_generations": lease_map,
                    },
                )
                cached_density = cache.lookup(
                    kind="density_model",
                    key=density_key,
                    dependencies=obs_keys,
                    miss_reason="observation_set_kde_parameters_or_lease_generation_changed",
                    sort_key=(-1, "50-density-model"),
                    window_index=None,
                    path="density",
                )
                if cached_density is not None:
                    density_payload = cached_density
                else:
                    density_payload = run_kde(observations, effective_config, executor)
                    cache.put(
                        kind="density_model",
                        key=density_key,
                        payload=density_payload,
                        dependencies=obs_keys,
                        lease_generation=lease_generation,
                    )
                cache.save()
                lease_valid, expired_leases_after = _validate_leases(
                    lease_manager, lease_records, failures
                )
                expired_leases.extend(expired_leases_after)
                if not lease_valid:
                    fatal = True

            if lease_valid and density_payload is not None:
                try:
                    lease_manager.commit([record.lease_id for record in lease_records])
                    lease_valid = all(
                        lease_manager.artifact_is_valid(record.lease_id, record.generation)
                        for record in lease_records
                    )
                except LeaseExpiredError as exc:
                    failures.add(exc.business_error())
                    lease_valid = False
                if not lease_valid:
                    fatal = True
                    failures.add(
                        BusinessError(
                            code="E_LEASE_EXPIRED",
                            path="leases",
                            message="committed lease validation failed",
                            stage="lease",
                        )
                    )
                _maybe_crash(crash_after, "before_snapshot")

        if clusters_invalid := (fatal or cluster_payload is None or not lease_valid):
            if cluster_payload is not None and clusters_invalid:
                cluster_payload = dict(cluster_payload)
                cluster_payload["invalidated"] = True
                cluster_payload["invalidation_reason"] = "lease_or_runtime_failure"
            if density_payload is not None and clusters_invalid:
                density_payload = dict(density_payload)
                density_payload["invalidated"] = True
                density_payload["invalidation_reason"] = "lease_or_runtime_failure"
            elif density_payload is None and fatal:
                density_payload = {
                    "invalidated": True,
                    "invalidation_reason": "lease_or_runtime_failure",
                    "window_scores": [],
                    "boundary_windows": [],
                    "below_threshold_windows": [],
                }

        snapshot_commit: dict[str, Any] | None = None
        supersession_validation = None
        if not fatal and cluster_payload is not None and density_payload is not None:
            new_version = snapshot_store.current_global_version() + 1
            supersession_validation = snapshot_store.audit_supersession(
                new_global_version=new_version,
                new_config_hash=effective_config.config_hash,
                new_data_fingerprint=data_fingerprint,
                new_lease_generation=lease_generation,
                new_lease_map=lease_map,
            )
            diagnostics.extend(supersession_validation.errors)
            snapshot_key = content_cache_key(
                "snapshot",
                {
                    "global_version": new_version,
                    "config_hash": effective_config.config_hash,
                    "data_fingerprint": data_fingerprint,
                    "lease_generation": lease_generation,
                },
            )
            cache.lookup(
                kind="snapshot",
                key=snapshot_key,
                dependencies=[],
                miss_reason="global_version_or_config_data_lease_changed",
                sort_key=(-1, "60-snapshot"),
                window_index=None,
                path="snapshot",
            )
            snapshot_assignment = {
                int(item["window_index"]): item
                for item in cluster_payload.get("assignments", [])
            }
            snapshot_density = {
                int(item["window_index"]): item
                for item in density_payload.get("window_scores", [])
            }
            snapshot_payload = {
                "config": {
                    "hash": effective_config.config_hash,
                    "effective": _config_payload(effective_config),
                },
                "data": report["data"],
                "standardization": stats,
                "clusters": cluster_payload,
                "density": density_payload,
                "edits": report["edits"],
                "window_index": [
                    {
                        "window_index": window.window_index,
                        "window_id": window.window_id,
                        "device_id": window.device_id,
                        "channel_id": window.channel_id,
                        "start": window.start,
                        "end": window.end,
                    }
                    for window in windows
                ],
                "windows": [
                    {
                        "window_index": window.window_index,
                        "window_id": window.window_id,
                        "raw_features": raw_by_index.get(window.window_index, {}).get(
                            "values"
                        ),
                        "fill_source": raw_by_index.get(window.window_index, {}).get(
                            "fill_source"
                        ),
                        "standardized_vector": observations.get(window.window_index),
                        "standardized_fixed_units": (
                            FixedPoint.from_config(
                                effective_config.fixed_point
                            ).quantize_vector(observations[window.window_index])
                            if window.window_index in observations
                            else None
                        ),
                        "cluster_id": snapshot_assignment.get(
                            window.window_index, {}
                        ).get("cluster_id"),
                        "density": snapshot_density.get(window.window_index),
                    }
                    for window in windows
                ],
                "window_failures": [
                    error.to_dict()
                    for error in failures.sorted()
                    if error.window_id is not None
                ],
            }
            cache.put(
                kind="snapshot",
                key=snapshot_key,
                payload=snapshot_payload,
                dependencies=[],
                lease_generation=lease_generation,
            )
            cache.save()
            snapshot = snapshot_store.commit(
                global_version=new_version,
                config_hash=effective_config.config_hash,
                data_fingerprint=data_fingerprint,
                lease_generation=lease_generation,
                lease_map=lease_map,
                payload=snapshot_payload,
                cache_digest=cache.digest(),
            )
            snapshot_commit = {
                "committed": True,
                "version": new_version,
                "checksum": snapshot["checksum"],
                "lease_generation": lease_generation,
                "lease_map": lease_map,
            }
        else:
            snapshot_commit = {"committed": False}

    raw_order = sorted(raw_by_index)
    assignment_by_index: dict[int, dict[str, Any]] = {}
    if cluster_payload is not None and not bool(cluster_payload.get("invalidated")):
        assignment_by_index = {
            int(item["window_index"]): item for item in cluster_payload["assignments"]
        }
    density_by_index: dict[int, dict[str, Any]] = {}
    if density_payload is not None and not bool(density_payload.get("invalidated")):
        density_by_index = {
            int(item["window_index"]): item for item in density_payload["window_scores"]
        }
    failure_by_window: dict[str, list[dict[str, Any]]] = {}
    for error in failures.sorted():
        if error.window_id:
            failure_by_window.setdefault(error.window_id, []).append(error.to_dict())

    fixed = FixedPoint.from_config(effective_config.fixed_point)
    windows_report: list[dict[str, Any]] = []
    for window in windows:
        item = _window_common(window)
        raw = raw_by_index.get(window.window_index)
        if raw is None:
            item.update(
                {
                    "status": "failed",
                    "processing_errors": failure_by_window.get(window.window_id, []),
                }
            )
            windows_report.append(item)
            continue
        standardized = observations.get(window.window_index)
        quantized = fixed.quantize_vector(standardized) if standardized is not None else None
        item.update(
            {
                "status": "ok",
                "feature_names": raw["feature_names"],
                "raw_features": raw["values"],
                "fill_source": raw["fill_source"],
                "standardized_vector": standardized,
                "standardized_fixed_units": quantized,
                "cluster_id": assignment_by_index.get(window.window_index, {}).get("cluster_id"),
                "cluster_distance_units": assignment_by_index.get(window.window_index, {}).get(
                    "distance_units"
                ),
                "density": density_by_index.get(window.window_index),
            }
        )
        windows_report.append(item)

    boundary_window_ids: list[str] = []
    if density_payload is not None and not bool(density_payload.get("invalidated")):
        index_to_id = {window.window_index: window.window_id for window in windows}
        boundary_window_ids = [
            index_to_id[index] for index in density_payload["boundary_windows"]
        ]

    report["windows"] = windows_report
    report["failed_windows"] = [
        error.to_dict() for error in failures.sorted() if error.window_id is not None
    ]
    report["standardization"] = stats
    report["clusters"] = cluster_payload
    report["density"] = density_payload
    report["boundary_windows"] = boundary_window_ids
    report["lease"] = {
        **lease_manager.summary(),
        "run_lease_generation": lease_generation,
        "run_lease_map": lease_map,
        "expired_during_run": expired_leases,
        "valid_at_commit": lease_valid,
    }
    report["snapshot"] = {
        "restore_validation": restore_validation.to_dict(),
        "supersession_validation": (
            supersession_validation.to_dict() if supersession_validation else None
        ),
        "commit": snapshot_commit,
        "used_old_snapshot_for_analysis": False,
    }
    report["recovery"] = {
        "snapshot_present_before_run": restore_validation.snapshot_present,
        "snapshot_checksum_valid": restore_validation.checksum_valid,
        "restoreable_four_way": restore_validation.restoreable,
        "resume_scope": recovery_scope,
        "snapshot_lease_generation": snapshot_generation,
        "persisted_lease_high_water_before_run": persisted_lease_high_water_before,
        "persisted_lease_high_water_after_run": lease_manager.high_water_generation,
        "reclaimed_orphan_leases": lease_manager.reclaimed,
        "snapshot_referenced_generation_ahead": (
            snapshot_generation is not None
            and snapshot_generation > persisted_lease_high_water_before
        ),
    }
    cache_summary = cache.summary()
    report["cache"] = cache_summary
    report["edits"]["failures"] = [error.to_dict() for error in edit_failures]
    sorted_failures = failures.sorted()
    sorted_diagnostics = _stable_error_dedupe(diagnostics)
    report["failures"] = [error.to_dict() for error in sorted_failures]
    report["diagnostics"] = [error.to_dict() for error in sorted_diagnostics]
    all_errors = _stable_error_dedupe(sorted_failures + sorted_diagnostics)
    report["business_errors"] = [error.to_dict() for error in all_errors]
    report["error_codes"] = sorted({error.code for error in all_errors})
    report["status"] = "failed" if fatal else ("partial" if sorted_failures else "ok")
    report["workers"] = {
        "max_workers": effective_config.max_workers,
        "thread_pool": "ThreadPoolExecutor",
        "stable_task_sort_keys": True,
    }

    if output_path is not None:
        atomic_write_json(Path(output_path), report)
    return report
