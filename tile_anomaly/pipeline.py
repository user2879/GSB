"""End-to-end incremental tile anomaly detection pipeline."""

from __future__ import annotations

import copy
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cache import SnapshotStore, previous_by_key, previous_by_window
from .config import config_hash, validate_config
from .detection import (
    build_boundary_diagnostics,
    compute_robust_statistics,
    prepare_score_context,
    score_from_context,
    score_key_for_feature,
    statistics_key_for_features,
    summarize_tiles,
    threshold_decisions,
)
from .errors import (
    E_BUDGET_PENDING,
    E_FUSION_PENDING,
    E_WINDOW_FEATURE,
    BusinessEvent,
)
from .features import compute_window_feature, feature_key_for_window
from .fusion import (
    apply_fused_bands,
    build_tile_pixel_maps,
    compute_fusion_band,
    pixel_raster_bytes,
    plan_fusion_bands,
)
from .manifest import load_manifest
from .scheduler import allocate_windows, run_threaded, stable_grouped_windows
from .tiles import build_mosaic, load_tiles
from .util import atomic_write_json, load_structured, resolve_path, sha256_json
from .windows import WindowSpec, generate_windows, windows_by_tile


CACHE_POLICY_VERSION = "content-fingerprint-stage-cache-v1"


@dataclass
class PipelineResult:
    complete: bool
    snapshot_sequence: int
    snapshot_dir: Path
    global_version: str
    output_png: Path | None
    events: list[dict[str, Any]] = field(default_factory=list)
    audit_summary: dict[str, Any] = field(default_factory=dict)
    run_manifest: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "complete": self.complete,
            "snapshot_sequence": self.snapshot_sequence,
            "snapshot_dir": str(self.snapshot_dir),
            "global_version": self.global_version,
            "output_png": str(self.output_png) if self.output_png else None,
            "events": self.events,
            "audit_summary": self.audit_summary,
            "run_manifest": self.run_manifest,
        }


def _load_injection(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    value = load_structured(path)
    if not isinstance(value, dict):
        raise ValueError("fault injection configuration must be an object")
    return value


def _feature_stage(
    feature_by_key: dict[str, dict[str, Any]], features: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "count": len(features),
        "by_key": feature_by_key,
        "by_window": {
            record["window_id"]: record["feature_key"] for record in features
        },
    }


def _score_stage(
    score_by_key: dict[str, dict[str, Any]], scores: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "count": len(scores),
        "by_key": score_by_key,
        "by_window": {record["window_id"]: record["score_key"] for record in scores},
    }


def _changed_tile_ids(
    old_record: dict[str, Any] | None, new_record: dict[str, Any]
) -> list[str]:
    if not old_record:
        return sorted(new_record.get("upstream_tile_keys", {}).keys())
    old_keys = old_record.get("upstream_tile_keys", {})
    new_keys = new_record.get("upstream_tile_keys", {})
    changed = []
    for tile_id in sorted(set(old_keys) | set(new_keys)):
        if old_keys.get(tile_id) != new_keys.get(tile_id):
            changed.append(tile_id)
    return changed


def run_pipeline(
    config_path: Path,
    fault_config_path: Path | None = None,
    reset_state: bool = False,
) -> PipelineResult:
    config_path = config_path.resolve()
    raw_config = load_structured(config_path)
    if not isinstance(raw_config, dict):
        raise ValueError("configuration root must be an object")
    config = validate_config(raw_config)
    config_root = config_path.parent
    manifest_path = resolve_path(config_root, config["paths"]["manifest"])
    tiles_dir = resolve_path(config_root, config["paths"]["tiles_dir"])
    state_dir = resolve_path(config_root, config["paths"]["state_dir"])
    output_dir = resolve_path(config_root, config["paths"]["output_dir"])
    if reset_state and state_dir.exists():
        shutil.rmtree(state_dir)
    store = SnapshotStore(state_dir)
    previous = store.load_current()
    injections = _load_injection(fault_config_path)
    injection_hash = sha256_json(injections) if injections else None

    events: list[BusinessEvent] = []
    audit_entries: list[dict[str, Any]] = []
    cache_stats: dict[str, dict[str, int]] = {}

    def record_audit(
        stage: str,
        key: str,
        status: str,
        reason: str,
        *,
        window_id: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        stats = cache_stats.setdefault(stage, {"reuse": 0, "invalidate": 0})
        stats[status] = stats.get(status, 0) + 1
        audit_entries.append(
            {
                "stage": stage,
                "key": key,
                "status": status,
                "reason": reason,
                "window_id": window_id,
                "detail": detail or {},
            }
        )

    manifest, manifest_events, structurally_invalid = load_manifest(
        manifest_path, config, injections
    )
    events.extend(manifest_events)
    valid_mask = set(manifest.by_id()) - structurally_invalid
    tiles, tile_events = load_tiles(
        manifest,
        tiles_dir,
        config,
        injections,
        skip_tile_ids=structurally_invalid,
    )
    events.extend(tile_events)
    valid_mask.intersection_update(tiles.keys())
    mosaic = build_mosaic(tiles, config, manifest)
    windows, window_events = generate_windows(manifest, mosaic, set(tiles), config)
    events.extend(window_events)
    owned_windows = windows_by_tile(windows)
    tile_order = [
        tile.spec.tile_id
        for tile in sorted(tiles.values(), key=lambda item: (item.spec.row, item.spec.col))
    ]
    window_by_id = {window.window_id: window for window in windows}

    budget_limit = int(config["scheduler"]["global_window_budget"])
    consumed_budget = 0
    cumulative_budget = 0
    if previous is not None:
        previous_manifest = previous.stage("run_manifest", {}) or {}
        cumulative_budget = int(
            previous_manifest.get("scheduler_budget", {}).get(
                "cumulative_consumed",
                previous_manifest.get("scheduler_budget", {}).get("consumed", 0),
            )
        )

    prev_feature_by_key = previous_by_key(previous, "features")
    prev_feature_by_window = previous_by_window(previous, "features")
    current_feature_by_key: dict[str, dict[str, Any]] = {}
    current_features: list[dict[str, Any]] = []
    needed_feature_windows: list[WindowSpec] = []
    feature_meta: dict[str, tuple[WindowSpec, str]] = {}
    for window in windows:
        feature_key, _, _, _, _ = feature_key_for_window(window, tiles, config)
        feature_meta[window.window_id] = (window, feature_key)
        previous_record = prev_feature_by_key.get(feature_key)
        if previous_record is not None:
            current_feature_by_key[feature_key] = copy.deepcopy(previous_record)
            current_features.append(current_feature_by_key[feature_key])
            record_audit(
                "window_features",
                feature_key,
                "reuse",
                "cache_key_match",
                window_id=window.window_id,
            )
            continue
        old_window_key = prev_feature_by_window.get(window.window_id)
        old_record = prev_feature_by_key.get(old_window_key)
        detail: dict[str, Any] = {}
        if previous is None:
            reason = "no_previous_snapshot"
        elif not old_record:
            reason = "no_previous_window_entry"
        elif old_record.get("window_parameter_key") != feature_key_for_window(
            window, tiles, config
        )[1]:
            reason = "window_parameters_changed"
            detail["old_window_parameter_key"] = old_record.get("window_parameter_key")
        elif old_record.get("quantization_key") != feature_key_for_window(
            window, tiles, config
        )[2]:
            reason = "quantization_parameters_changed"
            detail["old_quantization_key"] = old_record.get("quantization_key")
        else:
            preview_record = {
                "upstream_tile_keys": {
                    tile_id: tiles[tile_id].content_key
                    for tile_id in window.source_tile_ids
                }
            }
            detail["changed_tiles"] = _changed_tile_ids(old_record, preview_record)
            reason = "upstream_tile_content_changed"
        record_audit(
            "window_features",
            feature_key,
            "invalidate",
            reason,
            window_id=window.window_id,
            detail=detail,
        )
        needed_feature_windows.append(window)

    needed_feature_ids = {window.window_id for window in needed_feature_windows}
    grouped_features = stable_grouped_windows(owned_windows, tile_order, needed_feature_ids)
    feature_allocation = allocate_windows(
        grouped_features,
        stage="window_features",
        budget_limit=budget_limit,
        consumed_budget=consumed_budget,
        tile_quota=int(config["scheduler"]["tile_window_quota"]),
    )
    events.extend(feature_allocation.events)
    consumed_budget += feature_allocation.consumed_increment
    feature_failures: list[str] = []
    if feature_allocation.selected:
        selected_windows = [window for _, window in feature_allocation.selected]

        def feature_worker(window: WindowSpec) -> dict[str, Any]:
            try:
                return compute_window_feature(window, mosaic, tiles, config)
            except BaseException as exc:
                return {
                    "__error__": str(exc),
                    "__window_id__": window.window_id,
                    "__tile_id__": window.owner_tile_id,
                }

        computed_features = run_threaded(
            selected_windows, feature_worker, int(config["_derived"]["max_workers"])
        )
        for record in computed_features:
            if "__error__" in record:
                feature_failures.append(record["__window_id__"])
                events.append(
                    BusinessEvent(
                        tile_id=record["__tile_id__"],
                        stage="window_features",
                        code=E_WINDOW_FEATURE,
                        message="window feature computation failed",
                        detail={
                            "window_id": record["__window_id__"],
                            "error": record["__error__"],
                        },
                    )
                )
                continue
            current_feature_by_key[record["feature_key"]] = record
            current_features.append(record)

    current_feature_by_key = {
        record["feature_key"]: record for record in current_features
    }
    current_features.sort(key=lambda record: tuple(record["core_rect"]))
    expected_feature_ids = set(feature_meta) - set(feature_failures)
    missing_feature_ids = {
        window_id
        for window_id in expected_feature_ids
        if feature_meta[window_id][1] not in current_feature_by_key
    }
    pending_queue: list[dict[str, Any]] = list(feature_allocation.pending)
    if missing_feature_ids:
        known_pending = {entry["window_id"] for entry in pending_queue}
        for window_id in sorted(missing_feature_ids - known_pending):
            window = window_by_id[window_id]
            pending_queue.append(
                {
                    "tile_id": window.owner_tile_id,
                    "window_id": window_id,
                    "stage": "window_features",
                    "code": (
                        E_BUDGET_PENDING
                        if consumed_budget >= budget_limit
                        else "E_BATCH_ABORTED"
                    ),
                }
            )

    global_version = sha256_json(
        {
            "cache_policy_version": CACHE_POLICY_VERSION,
            "config_hash": config_hash(config),
            "manifest_content_key": manifest.content_key(),
            "tile_content_keys": {
                tile_id: tile.content_key
                for tile_id, tile in sorted(tiles.items())
            },
            "band_set_versions": sorted(
                {tile.spec.band_set_version for tile in tiles.values()}
            ),
            "fusion_strategy_version": config["fusion"]["strategy_version"],
            "fault_injection_hash": injection_hash,
        }
    )

    statistics: dict[str, Any] | None = None
    statistics_key: str | None = None
    statistics_by_key: dict[str, Any] = {}
    current_scores: list[dict[str, Any]] = []
    score_by_key: dict[str, dict[str, Any]] = {}
    decisions: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    summary_by_key: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, Any] = {}
    diagnostic_by_key: dict[str, Any] = {}
    diagnostic_key: str | None = None
    fusion_results: list[dict[str, Any]] = []
    fusion_by_key: dict[str, dict[str, Any]] = {}
    pixel_stage: dict[str, Any] = {}
    raster_bytes: bytes | None = None
    score_pending: list[dict[str, Any]] = []
    fusion_pending: list[dict[str, Any]] = []
    feature_complete = not missing_feature_ids and bool(current_features)

    if feature_complete:
        statistics_key = statistics_key_for_features(current_features, config)
        prev_statistics_by_key = previous_by_key(previous, "statistics")
        statistics = prev_statistics_by_key.get(statistics_key)
        if statistics is not None:
            statistics = copy.deepcopy(statistics)
            record_audit(
                "robust_statistics",
                statistics_key,
                "reuse",
                "cache_key_match",
            )
        else:
            statistics = compute_robust_statistics(current_features, config)
            previous_statistics = (previous.stage("statistics", {}) or {}) if previous else {}
            previous_current_key = previous_statistics.get("current_key")
            previous_stats = (
                previous_statistics.get("by_key", {}).get(previous_current_key)
                if previous_current_key
                else None
            )
            old_feature_keys = set(
                (previous_stats or {}).get("dependency", {}).get("feature_keys", [])
            )
            new_feature_keys = {record["feature_key"] for record in current_features}
            detail = {
                "added_or_changed_feature_keys": sorted(
                    new_feature_keys - old_feature_keys
                ),
                "removed_feature_keys": sorted(old_feature_keys - new_feature_keys),
            }
            record_audit(
                "robust_statistics",
                statistics_key,
                "invalidate",
                (
                    "global_feature_set_changed"
                    if previous_current_key
                    else "no_previous_statistics"
                ),
                detail=detail,
            )
        statistics_by_key[statistics_key] = statistics
        feature_set_key = sha256_json(
            [feature["feature_key"] for feature in current_features]
        )

        score_keys_by_window: dict[str, str] = {}
        missing_score_windows: list[WindowSpec] = []
        prev_score_by_key = previous_by_key(previous, "scores")
        prev_score_by_window = previous_by_window(previous, "scores")
        score_record_by_window = {
            record["window_id"]: record for record in current_scores
        }
        for feature in current_features:
            key = score_key_for_feature(
                feature, statistics_key, feature_set_key, config
            )
            score_keys_by_window[feature["window_id"]] = key
            cached = prev_score_by_key.get(key)
            if cached is not None:
                score_record_by_window[feature["window_id"]] = copy.deepcopy(cached)
                record_audit(
                    "window_anomaly_scores",
                    key,
                    "reuse",
                    "cache_key_match",
                    window_id=feature["window_id"],
                )
            else:
                old_key = prev_score_by_window.get(feature["window_id"])
                old_record = prev_score_by_key.get(old_key)
                if previous is None:
                    reason = "no_previous_snapshot"
                elif old_record is None:
                    reason = "no_previous_window_entry"
                elif old_record.get("statistics_key") != statistics_key:
                    reason = "global_statistics_key_changed"
                elif old_record.get("feature_key") != feature["feature_key"]:
                    reason = "upstream_feature_key_changed"
                else:
                    reason = "score_parameters_changed"
                record_audit(
                    "window_anomaly_scores",
                    key,
                    "invalidate",
                    reason,
                    window_id=feature["window_id"],
                    detail={
                        "old_statistics_key": (old_record or {}).get("statistics_key"),
                        "new_statistics_key": statistics_key,
                    },
                )
                missing_score_windows.append(window_by_id[feature["window_id"]])

        needed_score_ids = {window.window_id for window in missing_score_windows}
        grouped_scores = stable_grouped_windows(owned_windows, tile_order, needed_score_ids)
        score_allocation = allocate_windows(
            grouped_scores,
            stage="window_anomaly_scores",
            budget_limit=budget_limit,
            consumed_budget=consumed_budget,
            tile_quota=int(config["scheduler"]["tile_window_quota"]),
            quota_used=feature_allocation.quota_used,
        )
        events.extend(score_allocation.events)
        consumed_budget += score_allocation.consumed_increment
        score_failures: list[str] = []
        if score_allocation.selected:
            order_index = {
                feature["window_id"]: index
                for index, feature in enumerate(current_features)
            }
            feature_by_window = {
                feature["window_id"]: feature for feature in current_features
            }
            score_context = prepare_score_context(current_features, statistics, config)

            def score_worker(window: WindowSpec) -> dict[str, Any]:
                feature = feature_by_window[window.window_id]
                index = order_index[window.window_id]
                try:
                    return score_from_context(
                        index, feature, score_context, statistics, config
                    )
                except BaseException as exc:
                    return {
                        "__error__": str(exc),
                        "__window_id__": window.window_id,
                        "__tile_id__": window.owner_tile_id,
                    }

            computed_scores = run_threaded(
                [window for _, window in score_allocation.selected],
                score_worker,
                int(config["_derived"]["max_workers"]),
            )
            for record in computed_scores:
                if "__error__" in record:
                    score_failures.append(record["__window_id__"])
                    events.append(
                        BusinessEvent(
                            tile_id=record["__tile_id__"],
                            stage="window_anomaly_scores",
                            code=E_WINDOW_FEATURE,
                            message="window anomaly scoring failed",
                            detail={
                                "window_id": record["__window_id__"],
                                "error": record["__error__"],
                            },
                        )
                    )
                    continue
                score_record_by_window[record["window_id"]] = record

        active_score_ids = {
            feature["window_id"] for feature in current_features
        } - set(score_failures)
        missing_score_ids = {
            window_id
            for window_id in active_score_ids
            if window_id not in score_record_by_window
        }
        score_pending = list(score_allocation.pending)
        known_score_pending = {entry["window_id"] for entry in score_pending}
        for window_id in sorted(missing_score_ids - known_score_pending):
            window = window_by_id[window_id]
            score_pending.append(
                {
                    "tile_id": window.owner_tile_id,
                    "window_id": window_id,
                    "stage": "window_anomaly_scores",
                    "code": E_BUDGET_PENDING,
                }
            )
        if score_failures:
            active_score_ids -= set(score_failures)
        current_scores = [
            score_record_by_window[feature["window_id"]]
            for feature in current_features
            if feature["window_id"] in active_score_ids
            and feature["window_id"] in score_record_by_window
        ]
        score_by_key = {record["score_key"]: record for record in current_scores}
        if not missing_score_ids and not score_failures and len(current_scores) == len(
            current_features
        ):
            decisions = threshold_decisions(current_scores, config)
            summaries = summarize_tiles(tiles, current_features, current_scores, decisions, config)
            prev_summary_by_key = previous_by_key(previous, "summaries")
            for tile_id, summary in summaries.items():
                cached = prev_summary_by_key.get(summary["summary_key"])
                if cached is not None:
                    summaries[tile_id] = copy.deepcopy(cached)
                    record_audit(
                        "tile_decision_summaries",
                        summary["summary_key"],
                        "reuse",
                        "cache_key_match",
                        detail={"tile_id": tile_id},
                    )
                else:
                    record_audit(
                        "tile_decision_summaries",
                        summary["summary_key"],
                        "invalidate",
                        "score_or_threshold_changed",
                        detail={"tile_id": tile_id},
                    )
            summary_by_key = {
                summary["summary_key"]: summary for summary in summaries.values()
            }

            diagnostic_key = sha256_json(
                {
                    "diagnostic_version": "boundary-diagnostic-v1",
                    "score_keys": [record["score_key"] for record in current_scores],
                    "threshold_permille": config["detector"]["threshold_permille"],
                    "reference_fraction_bits": config["quantization"]["fraction_bits"] + 1,
                }
            )
            prev_diagnostic_by_key = previous_by_key(previous, "diagnostics")
            cached_diagnostics = prev_diagnostic_by_key.get(diagnostic_key)
            if cached_diagnostics is not None:
                diagnostics = copy.deepcopy(cached_diagnostics)
                record_audit(
                    "boundary_diagnostics",
                    diagnostic_key,
                    "reuse",
                    "cache_key_match",
                )
            else:
                diagnostics = build_boundary_diagnostics(
                    windows,
                    current_features,
                    decisions,
                    current_scores,
                    statistics,
                    mosaic,
                    tiles,
                    config,
                )
                record_audit(
                    "boundary_diagnostics",
                    diagnostic_key,
                    "invalidate",
                    "score_or_threshold_changed",
                )
            diagnostic_by_key[diagnostic_key] = diagnostics

            pixel_maps = build_tile_pixel_maps(
                tiles, windows, current_scores, decisions, config
            )
            fusion_plans = plan_fusion_bands(
                tiles, windows, current_scores, summaries, config
            )
            prev_fusion_by_key = previous_by_key(previous, "fusion")
            missing_fusion_plans = []
            for plan in fusion_plans:
                cached = prev_fusion_by_key.get(plan["fusion_key"])
                if cached is not None:
                    fusion_by_key[plan["fusion_key"]] = copy.deepcopy(cached)
                    record_audit(
                        "fusion_conclusions",
                        plan["fusion_key"],
                        "reuse",
                        "cache_key_match",
                        detail={"band_id": plan["band_id"]},
                    )
                else:
                    missing_fusion_plans.append(plan)
                    record_audit(
                        "fusion_conclusions",
                        plan["fusion_key"],
                        "invalidate",
                        "tile_summary_or_overlap_scores_changed",
                        detail={"band_id": plan["band_id"]},
                    )
            fusion_limit = int(config["scheduler"]["fusion_recompute_limit"])
            allowed_fusion_plans = missing_fusion_plans[:fusion_limit]
            postponed_fusion_plans = missing_fusion_plans[fusion_limit:]
            for plan in allowed_fusion_plans:
                result = compute_fusion_band(plan, tiles, pixel_maps, config)
                fusion_by_key[result["fusion_key"]] = result
            if postponed_fusion_plans:
                events.append(
                    BusinessEvent(
                        tile_id=None,
                        stage="fusion_scheduler",
                        code=E_FUSION_PENDING,
                        message="fusion recomputation limit reached; bands remain pending",
                        detail={
                            "fusion_recompute_limit": fusion_limit,
                            "pending_bands": [
                                plan["band_id"] for plan in postponed_fusion_plans
                            ],
                        },
                    )
                )
                fusion_pending = [
                    {
                        "tile_id": None,
                        "window_id": None,
                        "band_id": plan["band_id"],
                        "stage": "fusion_conclusions",
                        "code": E_FUSION_PENDING,
                    }
                    for plan in postponed_fusion_plans
                ]
            if not fusion_pending:
                fusion_results = [
                    fusion_by_key[plan["fusion_key"]] for plan in fusion_plans
                ]
                fused = apply_fused_bands(
                    pixel_maps,
                    fusion_results,
                    tiles,
                    config,
                    grid_shape=(mosaic.height, mosaic.width),
                )
                anomaly = fused["anomaly"]
                raster_bytes = pixel_raster_bytes(anomaly)
                pixel_stage = {
                    "pixel_key": fused["pixel_key"],
                    "shape": fused["shape"],
                    "anomaly_pixel_count": int((anomaly == 1).sum()),
                    "normal_pixel_count": int((anomaly == 0).sum()),
                    "unknown_pixel_count": int((anomaly < 0).sum()),
                    "fusion_band_ids": [result["band_id"] for result in fusion_results],
                    "fusion_keys": [result["fusion_key"] for result in fusion_results],
                    "postprocess": fused["postprocess"],
                    "pre_post_rows_encoded": [
                        "".join(
                            "1" if value == 1 else "0" if value == 0 else "u"
                            for value in row
                        )
                        for row in fused["pre_post"].tolist()
                    ],
                    "post_rows_encoded": [
                        "".join(
                            "1" if value == 1 else "0" if value == 0 else "u"
                            for value in row
                        )
                        for row in anomaly.tolist()
                    ],
                    "raster_file": "pixel_anomaly.png",
                }

    pending_queue.extend(score_pending)
    pending_queue.extend(fusion_pending)
    pending_queue.sort(
        key=lambda entry: (
            entry.get("stage", ""),
            entry.get("tile_id") or "",
            entry.get("window_id") or entry.get("band_id") or "",
        )
    )
    score_complete = feature_complete and not score_pending and len(current_scores) == len(
        current_features
    )
    fusion_complete = score_complete and not fusion_pending and bool(pixel_stage)
    complete = feature_complete and score_complete and fusion_complete and not feature_failures

    scores_by_tile: dict[str, set[str]] = {}
    for score in current_scores:
        scores_by_tile.setdefault(score["owner_tile_id"], set()).add(score["window_id"])
    completed_tiles = []
    cursor = None
    for tile_id in tile_order:
        expected = {window.window_id for window in owned_windows.get(tile_id, [])}
        if expected and expected.issubset(scores_by_tile.get(tile_id, set())):
            completed_tiles.append(tile_id)
            tile = tiles[tile_id]
            cursor = {"tile_id": tile_id, "row": tile.spec.row, "col": tile.spec.col}
        else:
            break

    sequence = store.next_sequence()
    audit_summary = {
        "cache_policy_version": CACHE_POLICY_VERSION,
        "stages": {stage: dict(values) for stage, values in sorted(cache_stats.items())},
        "total_reuse": sum(values.get("reuse", 0) for values in cache_stats.values()),
        "total_invalidate": sum(
            values.get("invalidate", 0) for values in cache_stats.values()
        ),
    }
    stage_content_keys = {
        "feature_set": sha256_json(
            [record["feature_key"] for record in current_features]
        ) if current_features else None,
        "statistics": statistics_key,
        "score_set": sha256_json([record["score_key"] for record in current_scores])
        if current_scores
        else None,
        "summary_set": sha256_json(
            [summary["summary_key"] for summary in summaries.values()]
        )
        if summaries
        else None,
        "fusion_set": sha256_json(
            [result["fusion_key"] for result in fusion_results]
        )
        if fusion_results
        else None,
        "pixel": pixel_stage.get("pixel_key"),
    }
    run_manifest = {
        "snapshot_sequence": sequence,
        "global_version": global_version,
        "cache_policy_version": CACHE_POLICY_VERSION,
        "config_hash": config_hash(config),
        "manifest_hash": manifest.content_key(),
        "band_set_versions": sorted(
            {tile.spec.band_set_version for tile in tiles.values()}
        ),
        "fusion_strategy_version": config["fusion"]["strategy_version"],
        "completed_tile_cursor": cursor,
        "completed_tiles": completed_tiles,
        "pending_queue": pending_queue,
        "scheduler_budget": {
            "limit": budget_limit,
            "consumed": consumed_budget,
            "remaining": max(0, budget_limit - consumed_budget),
            "cumulative_consumed": cumulative_budget + consumed_budget,
            "scope": "per_batch",
        },
        "tile_window_quota": int(config["scheduler"]["tile_window_quota"]),
        "fusion_recompute_limit": int(config["scheduler"]["fusion_recompute_limit"]),
        "max_workers": int(config["_derived"]["max_workers"]),
        "requested_max_workers": int(config["_derived"]["requested_max_workers"]),
        "complete": complete,
        "stage_content_keys": stage_content_keys,
        "stage_cache": audit_summary,
        "events": [event.to_dict() for event in events],
        "fault_injection_hash": injection_hash,
        "construction_order": "tile_row_col_window_row_col_dimension_ascending",
        "threshold_permille": config["detector"]["threshold_permille"],
        "window": {
            "size": config["window"]["size"],
            "step": config["window"]["step"],
            "context_overlap": config["window"]["context_overlap"],
        },
        "quantization": {
            "fraction_bits": config["quantization"]["fraction_bits"],
            "gear": config["quantization"]["gear"],
        },
    }
    stages: dict[str, Any] = {
        "features": _feature_stage(current_feature_by_key, current_features),
        "statistics": {
            "current_key": statistics_key,
            "by_key": statistics_by_key,
        },
        "scores": _score_stage(score_by_key, current_scores),
        "decisions": {"count": len(decisions), "records": decisions},
        "summaries": {
            "by_key": summary_by_key,
            "by_tile": {
                tile_id: summary["summary_key"] for tile_id, summary in summaries.items()
            },
        },
        "diagnostics": {
            "current_key": diagnostic_key,
            "by_key": diagnostic_by_key,
        },
        "fusion": {
            "by_key": fusion_by_key,
            "by_band": {
                plan["band_id"]: plan["fusion_key"]
                for plan in plan_fusion_bands(tiles, windows, current_scores, summaries, config)
                if plan["fusion_key"] in fusion_by_key
            }
            if summaries and current_scores
            else {},
        },
        "pixel_map": pixel_stage,
        "audit": {
            "summary": audit_summary,
            "entries": audit_entries,
        },
        "run_manifest": run_manifest,
    }
    binary_files = {"pixel_anomaly.png": raster_bytes} if raster_bytes is not None else {}
    snapshot_dir = store.commit(stages, binary_files)
    output_png = None
    if raster_bytes is not None:
        output_png = output_dir / "pixel_anomaly.png"
        store.publish_file(snapshot_dir, "pixel_anomaly.png", output_png)
    else:
        stale_output = output_dir / "pixel_anomaly.png"
        if stale_output.exists():
            stale_output.unlink()
    atomic_write_json(
        output_dir / "run_result.json",
        {
            "complete": complete,
            "snapshot_sequence": sequence,
            "snapshot_dir": str(snapshot_dir),
            "global_version": global_version,
            "pixel_key": pixel_stage.get("pixel_key"),
            "output_png": str(output_png) if output_png is not None else None,
            "pending_queue": pending_queue,
            "events": [event.to_dict() for event in events],
            "audit_summary": audit_summary,
        },
    )
    return PipelineResult(
        complete=complete,
        snapshot_sequence=sequence,
        snapshot_dir=snapshot_dir,
        global_version=global_version,
        output_png=output_png,
        events=[event.to_dict() for event in events],
        audit_summary=audit_summary,
        run_manifest=run_manifest,
    )
