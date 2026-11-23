"""Parallel, cache-aware computation of the complete stable artifact graph."""

from __future__ import annotations

import copy
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from .aggregate import aggregate_speaker, aggregation_cache_key
from .assign import fold_assignment, pair_decision, similarity
from .config import stage_fingerprint
from .features import extract_features
from .quantize import compute_quantization_plan, description_cache_key, quantize_segment
from .segment import annotation_for_interval, segment_content_fp
from .util import cache_key, sha256_obj


class ProgressTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {
            "stage": "idle",
            "phase": "",
            "phase_completed": 0,
            "phase_total": 0,
            "completed_tasks": 0,
            "total_tasks": 0,
            "fraction": 0.0,
            "active_workers": 0,
            "peak_workers": 0,
            "last_sort_key": "",
        }
        self.task_order: list[str] = []

    def begin_phase(self, phase: str, total: int) -> None:
        with self._lock:
            self._state.update(
                {
                    "stage": phase,
                    "phase": phase,
                    "phase_completed": 0,
                    "phase_total": int(total),
                    "total_tasks": int(self._state["total_tasks"]) + int(total),
                }
            )
            self._update_fraction_locked()

    def task_started(self, sort_key: str) -> None:
        with self._lock:
            self._state["active_workers"] = int(self._state["active_workers"]) + 1
            self._state["peak_workers"] = max(
                int(self._state["peak_workers"]), int(self._state["active_workers"])
            )
            self._state["last_sort_key"] = sort_key
            self.task_order.append(sort_key)

    def task_finished(self) -> None:
        with self._lock:
            self._state["active_workers"] = max(0, int(self._state["active_workers"]) - 1)
            self._state["phase_completed"] = int(self._state["phase_completed"]) + 1
            self._state["completed_tasks"] = int(self._state["completed_tasks"]) + 1
            self._update_fraction_locked()

    def _update_fraction_locked(self) -> None:
        total = int(self._state["total_tasks"])
        completed = int(self._state["completed_tasks"])
        self._state["fraction"] = round(completed / total, 9) if total else 0.0

    def snapshot(self, *, complete: bool = False) -> dict[str, Any]:
        with self._lock:
            result = copy.deepcopy(self._state)
        if complete:
            result["stage"] = "complete"
            result["phase"] = "complete"
            result["fraction"] = 1.0
        result["task_order_stable"] = sorted(self.task_order)
        return result


def _run_parallel(
    tasks: dict[str, tuple[str, Callable[[], Any]]],
    *,
    max_workers: int,
    tracker: ProgressTracker,
    phase: str,
) -> tuple[dict[str, Any], list[str]]:
    ordered = sorted(tasks.items(), key=lambda item: (item[1][0], item[0]))
    tracker.begin_phase(phase, len(ordered))
    if not ordered:
        return {}, []
    results: dict[str, Any] = {}
    errors: list[tuple[str, BaseException]] = []

    def wrapped(sort_key: str, function: Callable[[], Any]) -> Any:
        tracker.task_started(sort_key)
        try:
            return function()
        finally:
            tracker.task_finished()

    with ThreadPoolExecutor(max_workers=max(1, int(max_workers)), thread_name_prefix=f"timbre-{phase}") as executor:
        future_to_key = {
            executor.submit(wrapped, sort_key, function): key for key, (sort_key, function) in ordered
        }
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                results[key] = future.result()
            except BaseException as exc:  # captured and re-raised in deterministic key order
                errors.append((key, exc))
    if errors:
        errors.sort(key=lambda item: item[0])
        raise errors[0][1]
    return results, [sort_key for _key, (sort_key, _function) in ordered]


def _ordered_defs(segments_by_file: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for file_name in sorted(segments_by_file, key=lambda value: (value.casefold(), value)):
        segments = list(segments_by_file[file_name])
        segments.sort(key=lambda row: (int(row["start_sample"]), int(row["end_sample"])))
        for ordinal, segment in enumerate(segments):
            rows.append(
                {
                    "file": file_name,
                    "start_sample": int(segment["start_sample"]),
                    "end_sample": int(segment["end_sample"]),
                    "ordinal_in_file": ordinal,
                }
            )
    rows.sort(
        key=lambda row: (
            row["file"].casefold(),
            row["file"],
            row["start_sample"],
            row["end_sample"],
        )
    )
    for index, row in enumerate(rows):
        row["segment_index"] = index
        row["segment_id"] = f"{row['file']}::seg_{row['start_sample']:08d}_{row['end_sample']:08d}"
        row["segment_uid"] = sha256_obj(
            {"file": row["file"], "start_sample": row["start_sample"], "end_sample": row["end_sample"]}
        )
    return rows


def recompute_pipeline(
    *,
    assets: dict[str, Any],
    segments_by_file: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
    labels: dict[str, Any],
    tier_overrides: dict[str, int | None],
    caches: dict[str, dict[str, Any]] | None,
    progress: ProgressTracker,
    force_recompute: bool = False,
    affected_segment_ids: list[str] | None = None,
    reason: str = "incremental",
) -> dict[str, Any]:
    old_caches = caches or {}
    old_feature = old_caches.get("feature", {})
    old_description = old_caches.get("description", {})
    old_assignment = old_caches.get("assignment", {})
    old_aggregation = old_caches.get("aggregation", {})

    feature_config_fp = stage_fingerprint(
        config,
        "audio",
        "segmentation",
        "features",
        extra={"algorithm": "features-v1"},
    )
    assignment_config_fp = stage_fingerprint(config, "assignment", extra={"algorithm": "assignment-v1"})
    aggregation_config_fp = stage_fingerprint(config, "aggregation", extra={"algorithm": "aggregation-v1"})
    max_workers = int(config["runtime"]["max_workers"])

    rows = _ordered_defs(segments_by_file)
    for row in rows:
        if row["file"] not in assets:
            raise KeyError(f"audio asset is missing for segment: {row['file']}")
        asset = assets[row["file"]]
        row["annotation"] = annotation_for_interval(
            labels, row["file"], row["start_sample"], row["end_sample"]
        )
        row["content_fingerprint"] = segment_content_fp(
            asset.samples,
            row["start_sample"],
            row["end_sample"],
            config,
            asset.sample_rate,
        )
        row["feature_key"] = cache_key(
            "feature_product",
            row["content_fingerprint"],
            feature_config_fp,
            asset.sample_rate,
        )

    feature_records: dict[str, dict[str, Any]] = {}
    feature_tasks: dict[str, tuple[str, Callable[[], Any]]] = {}
    reused_keys: dict[str, set[str]] = {name: set() for name in ("feature", "description", "assignment", "aggregation")}
    recomputed_keys: dict[str, set[str]] = {name: set() for name in ("feature", "description", "assignment", "aggregation")}

    for row in rows:
        key = row["feature_key"]
        if key in feature_records:
            continue
        if not force_recompute and key in old_feature:
            feature_records[key] = old_feature[key]
            reused_keys["feature"].add(key)
            continue
        asset = assets[row["file"]]
        start_sample = row["start_sample"]
        end_sample = row["end_sample"]
        content_fingerprint = row["content_fingerprint"]
        sort_key = f"{row['file'].casefold()}|{row['file']}|{row['segment_index']:08d}|feature"

        def feature_job(
            asset=asset,
            start_sample=start_sample,
            end_sample=end_sample,
            key=key,
            content_fingerprint=content_fingerprint,
        ) -> dict[str, Any]:
            return {
                "key": key,
                "content_fingerprint": content_fingerprint,
                "feature_config_fingerprint": feature_config_fp,
                "features": extract_features(
                    asset.samples,
                    asset.sample_rate,
                    start_sample,
                    end_sample,
                    config,
                ),
            }

        existing = feature_tasks.get(key)
        if existing is None or sort_key < existing[0]:
            feature_tasks[key] = (sort_key, feature_job)

    computed, feature_order = _run_parallel(
        feature_tasks,
        max_workers=max_workers,
        tracker=progress,
        phase="feature_concurrency",
    )
    for key, record in computed.items():
        feature_records[key] = record
        recomputed_keys["feature"].add(key)
    for row in rows:
        row["features"] = feature_records[row["feature_key"]]["features"]

    plan = compute_quantization_plan(rows, config, tier_overrides)
    quant_config_fp = sha256_obj(config["quantization"])
    description_records: dict[str, dict[str, Any]] = {}
    description_tasks: dict[str, tuple[str, Callable[[], Any]]] = {}
    for row in rows:
        key = description_cache_key(row["feature_key"], plan, config["quantization"])
        row["description_key"] = key
        if key in description_records:
            continue
        if not force_recompute and key in old_description:
            description_records[key] = old_description[key]
            reused_keys["description"].add(key)
            continue
        feature_key = row["feature_key"]
        sort_key = f"{row['file'].casefold()}|{row['file']}|{row['segment_index']:08d}|description"

        def description_job(feature_key=feature_key, key=key) -> dict[str, Any]:
            return {
                "key": key,
                "feature_key": feature_key,
                "plan_signature": plan["signature"],
                "quantization_config_fingerprint": quant_config_fp,
                "description": quantize_segment(feature_records[feature_key]["features"], plan),
            }

        existing = description_tasks.get(key)
        if existing is None or sort_key < existing[0]:
            description_tasks[key] = (sort_key, description_job)

    computed, description_order = _run_parallel(
        description_tasks,
        max_workers=max_workers,
        tracker=progress,
        phase="description_concurrency",
    )
    for key, record in computed.items():
        description_records[key] = record
        recomputed_keys["description"].add(key)
    for row in rows:
        row["description"] = description_records[row["description_key"]]["description"]

    previous_row_by_file: dict[str, dict[str, Any]] = {}
    pair_decisions: dict[str, dict[str, Any]] = {}
    assignment_tasks: dict[str, tuple[str, Callable[[], Any]]] = {}
    for row in rows:
        previous = previous_row_by_file.get(row["file"])
        if previous is None:
            row["pair_decision_key"] = None
            row["assignment_key"] = cache_key(
                "assignment_root", row["description_key"], assignment_config_fp
            )
            row["assignment_dependencies"] = [row["description_key"], assignment_config_fp]
        else:
            key = cache_key(
                "assignment_pair",
                previous["description_key"],
                row["description_key"],
                assignment_config_fp,
            )
            row["pair_decision_key"] = key
            row["assignment_key"] = key
            row["assignment_dependencies"] = [
                previous["description_key"],
                row["description_key"],
                assignment_config_fp,
            ]
            if key not in pair_decisions:
                if not force_recompute and key in old_assignment:
                    pair_decisions[key] = old_assignment[key]
                    reused_keys["assignment"].add(key)
                else:
                    left = list(previous["description"]["vector"])
                    right = list(row["description"]["vector"])
                    threshold = float(config["assignment"]["similarity_threshold"])
                    metric_config = copy.deepcopy(config["assignment"])
                    sort_key = (
                        f"{row['file'].casefold()}|{row['file']}|"
                        f"{previous['segment_index']:08d}|{row['segment_index']:08d}|assignment"
                    )

                    def assignment_job(
                        key=key,
                        left=left,
                        right=right,
                        threshold=threshold,
                        metric_config=metric_config,
                    ) -> dict[str, Any]:
                        return {
                            "key": key,
                            "metric": metric_config["metric"],
                            "threshold": threshold,
                            **pair_decision(similarity(left, right, metric_config), threshold),
                        }

                    existing = assignment_tasks.get(key)
                    if existing is None or sort_key < existing[0]:
                        assignment_tasks[key] = (sort_key, assignment_job)
        previous_row_by_file[row["file"]] = row

    computed, assignment_order = _run_parallel(
        assignment_tasks,
        max_workers=max_workers,
        tracker=progress,
        phase="assignment_concurrency",
    )
    for key, record in computed.items():
        pair_decisions[key] = record
        recomputed_keys["assignment"].add(key)

    assignments, conflicts = fold_assignment(rows, pair_decisions, config["assignment"])
    for row, assignment in zip(rows, assignments):
        row["assignment"] = assignment

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["assignment"]["speaker_id"], []).append(row)
    aggregation_records: dict[str, dict[str, Any]] = {}
    aggregation_tasks: dict[str, tuple[str, Callable[[], Any]]] = {}
    aggregation_keys_by_speaker: dict[str, str] = {}
    for speaker_id in sorted(grouped, key=lambda value: (value.casefold(), value)):
        speaker_rows = grouped[speaker_id]
        key = aggregation_cache_key(speaker_id, speaker_rows, config["aggregation"])
        aggregation_keys_by_speaker[speaker_id] = key
        if key in aggregation_records:
            continue
        if not force_recompute and key in old_aggregation:
            aggregation_records[key] = old_aggregation[key]
            reused_keys["aggregation"].add(key)
            continue
        sort_key = f"{speaker_id.casefold()}|{speaker_id}|speaker_aggregation"
        frozen_rows = copy.deepcopy(speaker_rows)

        def aggregation_job(
            speaker_id=speaker_id,
            key=key,
            frozen_rows=frozen_rows,
        ) -> dict[str, Any]:
            return {
                "key": key,
                "speaker_id": speaker_id,
                "aggregation": aggregate_speaker(speaker_id, frozen_rows, config["aggregation"]),
            }

        existing = aggregation_tasks.get(key)
        if existing is None or sort_key < existing[0]:
            aggregation_tasks[key] = (sort_key, aggregation_job)

    computed, aggregation_order = _run_parallel(
        aggregation_tasks,
        max_workers=max_workers,
        tracker=progress,
        phase="aggregation_concurrency",
    )
    for key, record in computed.items():
        aggregation_records[key] = record
        recomputed_keys["aggregation"].add(key)

    segment_outputs: list[dict[str, Any]] = []
    for row in rows:
        previous_description_key = None
        if row["ordinal_in_file"] > 0:
            prior = [
                candidate
                for candidate in rows
                if candidate["file"] == row["file"] and candidate["ordinal_in_file"] == row["ordinal_in_file"] - 1
            ]
            if prior:
                previous_description_key = prior[0]["description_key"]
        segment_outputs.append(
            {
                "segment_index": row["segment_index"],
                "segment_id": row["segment_id"],
                "segment_uid": row["segment_uid"],
                "file": row["file"],
                "ordinal_in_file": row["ordinal_in_file"],
                "start_sample": row["start_sample"],
                "end_sample": row["end_sample"],
                "sample_count": row["end_sample"] - row["start_sample"],
                "annotation_speaker_id": row["annotation"],
                "content_fingerprint": row["content_fingerprint"],
                "feature_key": row["feature_key"],
                "features": row["features"],
                "description_key": row["description_key"],
                "description": row["description"],
                "assignment_key": row["assignment_key"],
                "assignment": row["assignment"],
                "dependencies": {
                    "content_fingerprint": row["content_fingerprint"],
                    "feature_key": row["feature_key"],
                    "feature_depends_on": [row["content_fingerprint"], feature_config_fp],
                    "description_key": row["description_key"],
                    "description_depends_on": [
                        row["feature_key"],
                        plan["signature"],
                        quant_config_fp,
                    ],
                    "assignment_key": row["assignment_key"],
                    "assignment_depends_on": [
                        previous_description_key,
                        row["description_key"],
                        assignment_config_fp,
                    ],
                },
            }
        )

    speaker_outputs: list[dict[str, Any]] = []
    for speaker_id in sorted(grouped, key=lambda value: (value.casefold(), value)):
        key = aggregation_keys_by_speaker[speaker_id]
        record = aggregation_records[key]["aggregation"]
        speaker_outputs.append(
            {
                **copy.deepcopy(record),
                "aggregation_key": key,
                "dependencies": {
                    "aggregation_key": key,
                    "depends_on_description_keys": [
                        participant["description_key"] for participant in record["participants"]
                    ],
                    "aggregation_config_fingerprint": aggregation_config_fp,
                },
            }
        )

    old_by_stage = {
        "feature": set(old_feature),
        "description": set(old_description),
        "assignment": set(old_assignment),
        "aggregation": set(old_aggregation),
    }
    current_by_stage = {
        "feature": set(feature_records),
        "description": set(description_records),
        "assignment": set(pair_decisions),
        "aggregation": set(aggregation_records),
    }
    invalidated_keys = {
        stage: sorted(old_by_stage[stage] - current_by_stage[stage]) for stage in old_by_stage
    }

    product_versions = {
        "feature": sha256_obj(sorted(current_by_stage["feature"]) + [feature_config_fp]),
        "description": sha256_obj(sorted(current_by_stage["description"]) + [plan["signature"]]),
        "assignment": sha256_obj(sorted(current_by_stage["assignment"]) + [assignment_config_fp]),
        "aggregation": sha256_obj(sorted(current_by_stage["aggregation"]) + [aggregation_config_fp]),
        "segments": sha256_obj([row["segment_uid"] for row in rows]),
        "speakers": sha256_obj([speaker["aggregation_key"] for speaker in speaker_outputs]),
    }
    dependency_manifest = {
        "order": "file_casefold_then_file_then_start_sample",
        "segments": [
            {"segment_id": segment["segment_id"], **segment["dependencies"]}
            for segment in segment_outputs
        ],
        "speakers": [
            {"speaker_id": speaker["speaker_id"], **speaker["dependencies"]}
            for speaker in speaker_outputs
        ],
        "product_versions": product_versions,
    }
    cache_manifest = {
        stage: sorted(current_by_stage[stage]) for stage in ("feature", "description", "assignment", "aggregation")
    }
    invalidation_reuse_manifest = {
        "policy": "exact_product_keys_retained_and_eligible_for_future_reuse",
        "current_product_keys": cache_manifest,
        "canonical_reuse_eligibility": {
            stage: {"key_count": len(keys), "all_current_keys_eligible": True}
            for stage, keys in cache_manifest.items()
        },
    }
    cache_events = {
        "forced_rebuild": bool(force_recompute),
        "reason": reason,
        "affected_segment_ids": sorted(set(affected_segment_ids or [])),
        "reused_keys": {stage: sorted(values) for stage, values in reused_keys.items()},
        "recomputed_keys": {stage: sorted(values) for stage, values in recomputed_keys.items()},
        "invalidated_keys": invalidated_keys,
        "reused_count": sum(len(values) for values in reused_keys.values()),
        "recomputed_count": sum(len(values) for values in recomputed_keys.values()),
        "invalidated_count": sum(len(values) for values in invalidated_keys.values()),
        "precise_scope": {
            "feature_reused": len(reused_keys["feature"]),
            "description_reused": len(reused_keys["description"]),
            "assignment_reused": len(reused_keys["assignment"]),
            "aggregation_reused": len(reused_keys["aggregation"]),
        },
    }
    task_orders = {
        "feature": feature_order,
        "description": description_order,
        "assignment": assignment_order,
        "aggregation": aggregation_order,
    }
    speaker_count = len(speaker_outputs)
    audio_files = [
        {
            "path": name,
            "sha256": assets[name].sha256,
            "sample_count": assets[name].sample_count,
            "sample_rate": assets[name].sample_rate,
            "channels": assets[name].channels,
            "bits_per_sample": assets[name].bits_per_sample,
        }
        for name in sorted(assets, key=lambda value: (value.casefold(), value))
    ]
    summary = {
        "completed_segments": len(segment_outputs),
        "speaker_count": speaker_count,
        "vector_dimension_count": int(plan["dimension_count"]),
        "affected_segment_ids": sorted(set(affected_segment_ids or [])),
        "affected_segment_count": len(set(affected_segment_ids or [])),
        "progress": progress.snapshot(complete=True),
        "valid_audio_file_count": len(audio_files),
        "conflict_count": len(conflicts),
        "reason": reason,
    }
    caches_out = {
        "feature": feature_records,
        "description": description_records,
        "assignment": pair_decisions,
        "aggregation": aggregation_records,
    }
    return {
        "status": "complete",
        "segments": segment_outputs,
        "speakers": speaker_outputs,
        "conflicts": conflicts,
        "quantization": {
            **plan,
            "tier_overrides": dict(sorted(tier_overrides.items())),
            "quantization_config_fingerprint": quant_config_fp,
        },
        "summary": summary,
        "audio_files": audio_files,
        "dependencies": dependency_manifest,
        "cache_manifest": cache_manifest,
        "invalidation_reuse_manifest": invalidation_reuse_manifest,
        "cache_events": cache_events,
        "task_orders": task_orders,
        "product_versions": product_versions,
        "threading": {
            "max_workers_configured": max_workers,
            "peak_workers_observed": int(progress.snapshot().get("peak_workers", 0)),
            "stable_sort_keys": True,
            "result_order": "sorted_by_stable_key",
        },
        "_caches": caches_out,
    }
