"""Stable incremental reconstruction pipeline with stage snapshots."""

from __future__ import annotations

import os
import signal
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cache import CacheIndex, empty_cache
from .constants import (
    ENGINE_VERSION,
    EXIT_OK,
    EXIT_POINT_BUDGET_PENDING,
    EXIT_VALIDATION_FAILURE,
    STAGE_ORDER,
)
from .depth import collect_candidates, compute_pixel, depth_pixel_key, stage_view_record
from .errors import Failure, sorted_failures
from .features import extract_features, feature_key
from .fixed import canonical_json, stable_hash
from .io import ensure_writable, write_json_atomic
from .ledger import Ledger
from .matching import match_key, match_pair, pair_identifier, unavailable_pair
from .models import ProjectState
from .selector import resolution_specific_strategy
from .snapshot import Snapshot, SnapshotStore, WorkspaceLock, read_pending
from .scheduler import BudgetAllocation, allocate_feature_points
from .triangulation import triangle_identifier, triangle_key, triangulate_inlier
from .validation import ValidationResult, validate_project
from .voxels import build_voxels


@dataclass
class RunResult:
    stages: dict[str, Any]
    failures: list[Failure]
    pending_queue: list[dict[str, Any]]
    ledger: Ledger
    report: dict[str, Any]
    canonical_digest: str
    snapshot: Snapshot | None
    exit_code: int

    @property
    def status(self) -> str:
        return str(self.report.get("status", "unknown"))

    def persist_report(self, path: Path) -> None:
        write_json_atomic(path, self.report)

    def compare(self, other: "RunResult") -> dict[str, Any]:
        return {
            "equal": self.canonical_digest == other.canonical_digest,
            "left_digest": self.canonical_digest,
            "right_digest": other.canonical_digest,
            "failures_equal": [failure.as_dict() for failure in self.failures]
            == [failure.as_dict() for failure in other.failures],
            "pending_equal": self.pending_queue == other.pending_queue,
            "stages_equal": self.stages == other.stages,
        }


def _failure_dicts(failures: list[Failure]) -> list[dict[str, Any]]:
    return [failure.as_dict() for failure in sorted_failures(failures)]


def _exit_code(failures: list[Failure]) -> int:
    codes = {failure.code for failure in failures}
    if "E_POINT_BUDGET_PENDING" in codes or "E_VIEW_QUOTA_EXCEEDED" in codes:
        return EXIT_POINT_BUDGET_PENDING
    if failures:
        return EXIT_VALIDATION_FAILURE
    return EXIT_OK


def _classify(
    ledger: Ledger,
    stage: str,
    artifact: str,
    sort_key: str,
    key: str,
    previous: dict[str, Any] | None,
) -> None:
    if previous is None:
        ledger.add(stage, artifact, sort_key, "computed", "no_prior_artifact", key)
    elif previous.get("key") == key:
        ledger.add(stage, artifact, sort_key, "reused", "cache_key_match", key)
    else:
        ledger.add(
            stage,
            artifact,
            sort_key,
            "invalidated",
            "upstream_or_parameter_cache_key_changed",
            key,
        )


def _load_stage_payload(
    validation: ValidationResult,
    strategy: dict[str, Any] | None,
    config_hash: str,
    project_fingerprint: str,
) -> dict[str, Any]:
    return {
        "engine_version": ENGINE_VERSION,
        "config_hash": config_hash,
        "project_fingerprint": project_fingerprint,
        "image": validation.image_spec.canonical(),
        "config": validation.config.canonical() if validation.config is not None else None,
        "views": [
            {
                "view_id": view_id,
                "image_sha256": validation.views[view_id].image_sha256,
                "input_key": validation.views[view_id].input_key,
                "camera": validation.views[view_id].camera.canonical(),
            }
            for view_id in validation.valid_view_ids
        ],
        "pairs": [list(pair) for pair in validation.pairs],
        "strategy": strategy,
        "failures": _failure_dicts(validation.failures),
        "runnable": validation.runnable,
    }


def _parallel_features(
    validation: ValidationResult,
    cache: CacheIndex,
    ledger: Ledger,
    strategy: dict[str, Any],
    workers: int,
) -> tuple[dict[str, Any], int]:
    suites: dict[str, dict[str, Any]] = {}
    missing: list[int] = []
    for view_id in validation.valid_view_ids:
        view = validation.views[view_id]
        key = feature_key(view, validation.config, strategy)
        cached = cache.feature_by_key.get(key)
        if cached is not None:
            suites[str(view_id)] = cached
            ledger.add(
                "features",
                f"v:{view_id:04d}",
                f"v:{view_id:04d}",
                "reused",
                "cache_key_match",
                key,
            )
        else:
            missing.append(view_id)
    math = 0
    if missing:
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(missing)))) as executor:
            futures = {
                view_id: executor.submit(
                    extract_features,
                    validation.views[view_id],
                    validation.config,
                    strategy,
                )
                for view_id in missing
            }
            for view_id in missing:
                suite = futures[view_id].result()
                suites[str(view_id)] = suite
                previous = cache.feature_by_id.get(view_id)
                _classify(
                    ledger,
                    "features",
                    f"v:{view_id:04d}",
                    f"v:{view_id:04d}",
                    suite["key"],
                    previous,
                )
                math += suite["multiply_adds"]
    for view_id in validation.valid_view_ids:
        if str(view_id) in suites and view_id not in missing:
            math += suites[str(view_id)]["multiply_adds"]
    stage = {
        "views": {key: suites[key] for key in sorted(suites, key=int)},
        "order": sorted((int(key) for key in suites), key=int),
        "total_points": sum(len(suite["points"]) for suite in suites.values()),
        "total_candidate_blocks": sum(suite["candidate_blocks"] for suite in suites.values()),
        "multiply_adds": math,
    }
    return stage, math


def _match_stage(
    feature_stage: dict[str, Any],
    validation: ValidationResult,
    config_hash: str,
    cache: CacheIndex,
    ledger: Ledger,
    strategy: dict[str, Any],
    selected_view_ids: set[int],
) -> dict[str, Any]:
    pairs: dict[str, dict[str, Any]] = {}
    math = 0
    for first, second in validation.pairs:
        pair_id = pair_identifier(first, second)
        first_key = str(first)
        second_key = str(second)
        if (
            first_key not in feature_stage["views"]
            or second_key not in feature_stage["views"]
            or first not in selected_view_ids
            or second not in selected_view_ids
        ):
            pairs[pair_id] = unavailable_pair(
                first,
                second,
                "view_not_available_after_validation_or_budget",
            )
            ledger.add(
                "matches",
                pair_id,
                pair_id,
                "computed",
                "pair_unavailable_without_feature_inputs",
                pairs[pair_id]["key"],
            )
            continue
        first_suite = feature_stage["views"][first_key]
        second_suite = feature_stage["views"][second_key]
        key = match_key(first_suite, second_suite, validation.config, strategy)
        cached = cache.match_by_key.get(key)
        if cached is not None:
            pairs[pair_id] = cached
            ledger.add("matches", pair_id, pair_id, "reused", "cache_key_match", key)
            math += cached.get("multiply_adds", 0)
            continue
        record = match_pair(first_suite, second_suite, validation.config, strategy)
        pairs[pair_id] = record
        previous = cache.match_by_id.get(pair_id)
        _classify(ledger, "matches", pair_id, pair_id, key, previous)
        math += record["multiply_adds"]
    return {
        "pairs": pairs,
        "order": sorted(pairs, key=lambda item: tuple(int(part) for part in item[2:].split(":"))),
        "declared_pairs": [list(pair) for pair in validation.pairs],
        "total_inliers": sum(pair.get("inlier_count", 0) for pair in pairs.values()),
        "multiply_adds": math,
        "config_hash": config_hash,
    }


def _triangulation_stage(
    match_stage: dict[str, Any],
    validation: ValidationResult,
    cache: CacheIndex,
    ledger: Ledger,
) -> tuple[dict[str, Any], list[Failure]]:
    cameras = {
        view_id: validation.views[view_id].camera for view_id in validation.views
    }
    points: dict[str, dict[str, Any]] = {}
    failures: list[Failure] = []
    for pair_id in match_stage["order"]:
        pair = match_stage["pairs"][pair_id]
        if pair.get("status") == "unavailable":
            continue
        view_ids = (pair["view_ids"][0], pair["view_ids"][1])
        for inlier in sorted(
            pair["inliers"],
            key=lambda item: (
                item["left"][0],
                item["left"][1],
                item["right"][0],
                item["right"][1],
                item["left_id"],
                item["right_id"],
            ),
        ):
            identifier = triangle_identifier(pair, inlier)
            key = triangle_key(pair, inlier, validation.config)
            cached = cache.triangle_by_key.get(key)
            if cached is not None:
                points[identifier] = cached
                ledger.add(
                    "triangulation",
                    identifier,
                    identifier,
                    "reused",
                    "cache_key_match",
                    key,
                )
                continue
            record, error = triangulate_inlier(
                view_ids,
                cameras,
                pair,
                inlier,
                validation.config,
            )
            previous = cache.triangle_by_id.get(identifier)
            if record is None:
                failures.append(
                    Failure(
                        "E_TRIANGULATION_DEGENERATE",
                        view_id=view_ids[0],
                        artifact=identifier,
                        detail=error,
                    )
                )
                if previous is not None:
                    ledger.add(
                        "triangulation",
                        identifier,
                        identifier,
                        "invalidated",
                        "triangulation_degenerate",
                        previous["key"],
                    )
                continue
            points[identifier] = record
            _classify(ledger, "triangulation", identifier, identifier, key, previous)
    return (
        {
            "points": points,
            "order": sorted(points),
            "triangulated_points": len(points),
            "degenerate_points": len(failures),
            "failures": _failure_dicts(failures),
        },
        failures,
    )


def _depth_stage(
    triangles: dict[str, dict[str, Any]],
    validation: ValidationResult,
    cache: CacheIndex,
    ledger: Ledger,
) -> dict[str, Any]:
    candidates = collect_candidates(
        triangles,
        validation.views,
        validation.image_spec.width,
        validation.image_spec.height,
        validation.config,
    )
    views: dict[str, dict[str, Any]] = {}
    total_pixels = 0
    conflict_pixels = 0
    occupied_pixels = 0
    for view_id in validation.valid_view_ids:
        view = validation.views[view_id]
        pixels: list[dict[str, Any]] = []
        for y in range(validation.image_spec.height):
            for x in range(validation.image_spec.width):
                bucket = candidates[view_id].get((x, y), [])
                key = depth_pixel_key(view, x, y, bucket, validation.config)
                cached = cache.depth_pixel_by_key.get(key)
                identifier = f"d:{view_id:04d}:{y:04d}:{x:04d}"
                if cached is not None:
                    pixel = cached
                    ledger.add(
                        "depth",
                        identifier,
                        identifier,
                        "reused",
                        "cache_key_match",
                        key,
                    )
                else:
                    pixel = compute_pixel(
                        view,
                        x,
                        y,
                        bucket,
                        validation.config,
                    )
                    previous = cache.depth_pixel_by_id.get(identifier)
                    _classify(
                        ledger,
                        "depth",
                        identifier,
                        identifier,
                        key,
                        previous,
                    )
                pixels.append(pixel)
                total_pixels += 1
                conflict_pixels += int(pixel["conflict"])
                occupied_pixels += int(not pixel["empty"])
        views[str(view_id)] = stage_view_record(
            view,
            validation.image_spec.width,
            validation.image_spec.height,
            candidates[view_id],
            pixels,
        )
    return {
        "views": views,
        "order": sorted(views, key=int),
        "pixel_count": total_pixels,
        "occupied_pixels": occupied_pixels,
        "conflict_pixels": conflict_pixels,
    }


def _canonical_payload(
    stages: dict[str, Any],
    failures: list[Failure],
    pending_queue: list[dict[str, Any]],
    budget: dict[str, Any],
) -> dict[str, Any]:
    included_stages = {
        stage: stages[stage]
        for stage in STAGE_ORDER
        if stage in stages
    }
    return {
        "stages": included_stages,
        "failures": _failure_dicts(failures),
        "pending_queue": pending_queue,
        "budget": budget,
    }


class ReconstructionRunner:
    def __init__(
        self,
        workspace_root: Path,
        *,
        acquire_lock: bool = True,
    ) -> None:
        self.workspace_root = workspace_root.resolve()
        self.acquire_lock = acquire_lock
        self.store = SnapshotStore(self.workspace_root)
        self.last_snapshot: Snapshot | None = None

    def run(
        self,
        state: ProjectState,
        *,
        external_failures: list[Failure] | None = None,
        workers_override: int | None = None,
        sigkill_after_stage: str | None = None,
        write_report: Path | None = None,
    ) -> RunResult:
        ensure_writable(self.workspace_root)
        if self.acquire_lock:
            with WorkspaceLock(self.workspace_root):
                return self._run_locked(
                    state,
                    external_failures=external_failures,
                    workers_override=workers_override,
                    sigkill_after_stage=sigkill_after_stage,
                    write_report=write_report,
                )
        return self._run_locked(
            state,
            external_failures=external_failures,
            workers_override=workers_override,
            sigkill_after_stage=sigkill_after_stage,
            write_report=write_report,
        )

    def _run_locked(
        self,
        state: ProjectState,
        *,
        external_failures: list[Failure] | None,
        workers_override: int | None,
        sigkill_after_stage: str | None,
        write_report: Path | None,
    ) -> RunResult:
        target_config_hash = state.config_hash()
        target_project_fingerprint = state.project_fingerprint()
        loaded = self.store.load_latest(ENGINE_VERSION)
        config_same = loaded is not None and loaded.config_hash == target_config_hash
        cache_compatible = loaded is not None and loaded.engine_version == ENGINE_VERSION
        cache = CacheIndex.from_stages(loaded.stages) if cache_compatible else empty_cache()
        carried_stages = dict(loaded.stages) if cache_compatible else {}
        completed: set[str] = set()
        resume_from = ""
        if loaded is not None and config_same:
            completed = set(loaded.completed_stages)
            if loaded.completed_stages:
                resume_from = loaded.completed_stages[-1]
        resume_used = bool(config_same and completed)
        resume_reason = (
            "resume_after_interrupted_run"
            if resume_used and not loaded.run_complete
            else ("warm_complete_snapshot" if resume_used else "no_compatible_snapshot")
        )
        ledger = Ledger()
        validation = validate_project(state, external_failures)
        if validation.config is None:
            strategy: dict[str, Any] | None = None
        else:
            strategy = resolution_specific_strategy(
                validation.config,
                len(validation.valid_view_ids),
                len(validation.pairs),
                validation.image_spec.width,
                validation.image_spec.height,
            )
        load_stage = _load_stage_payload(
            validation,
            strategy,
            target_config_hash,
            target_project_fingerprint,
        )
        working_stages = dict(carried_stages)
        working_stages["load"] = load_stage
        completed.add("load")
        self._commit(
            working_stages,
            [],
            state,
            target_config_hash,
            target_project_fingerprint,
            completed,
            False,
        )
        self._maybe_sigkill("load", sigkill_after_stage)
        failures = list(validation.failures)
        pending_queue: list[dict[str, Any]] = []
        allocation = BudgetAllocation()
        feature_stage: dict[str, Any] = {
            "views": {},
            "order": [],
            "total_points": 0,
            "total_candidate_blocks": 0,
            "multiply_adds": 0,
        }
        match_stage: dict[str, Any] = {
            "pairs": {},
            "order": [],
            "declared_pairs": [list(pair) for pair in validation.pairs],
            "total_inliers": 0,
            "multiply_adds": 0,
            "config_hash": target_config_hash,
        }
        triangulation_stage: dict[str, Any] = {
            "points": {},
            "order": [],
            "triangulated_points": 0,
            "degenerate_points": 0,
            "failures": [],
        }
        depth_stage: dict[str, Any] = {
            "views": {},
            "order": [],
            "pixel_count": 0,
            "occupied_pixels": 0,
            "conflict_pixels": 0,
        }
        voxel_stage: dict[str, Any] = {
            "nodes": {},
            "order": [],
            "roots": [],
            "leaf_count": 0,
            "occupied_count": 0,
            "parent_count": 0,
            "node_count": 0,
        }
        voxel_math = {
            "triangle_intersection_checks": 0,
            "depth_projection_checks": 0,
            "multiply_adds": 0,
        }
        effective_workers = (
            workers_override
            if workers_override is not None
            else (validation.config.workers if validation.config is not None else 1)
        )
        if validation.config is not None and validation.runnable and strategy is not None:
            feature_stage, feature_math = _parallel_features(
                validation,
                cache,
                ledger,
                strategy,
                effective_workers,
            )
            working_stages["features"] = feature_stage
            completed.add("features")
            self._commit(
                working_stages,
                pending_queue,
                state,
                target_config_hash,
                target_project_fingerprint,
                completed,
                False,
            )
            self._maybe_sigkill("features", sigkill_after_stage)
            allocation = allocate_feature_points(feature_stage, validation.config)
            failures.extend(allocation.failures)
            pending_queue = allocation.pending_queue
            selected = set(allocation.accepted_view_ids)
            match_stage = _match_stage(
                feature_stage,
                validation,
                target_config_hash,
                cache,
                ledger,
                strategy,
                selected,
            )
            working_stages["matches"] = match_stage
            completed.add("matches")
            self._commit(
                working_stages,
                pending_queue,
                state,
                target_config_hash,
                target_project_fingerprint,
                completed,
                False,
            )
            self._maybe_sigkill("matches", sigkill_after_stage)
            triangulation_stage, triangulation_failures = _triangulation_stage(
                match_stage,
                validation,
                cache,
                ledger,
            )
            failures.extend(triangulation_failures)
            working_stages["triangulation"] = triangulation_stage
            completed.add("triangulation")
            self._commit(
                working_stages,
                pending_queue,
                state,
                target_config_hash,
                target_project_fingerprint,
                completed,
                False,
            )
            self._maybe_sigkill("triangulation", sigkill_after_stage)
            depth_stage = _depth_stage(
                triangulation_stage["points"],
                validation,
                cache,
                ledger,
            )
            working_stages["depth"] = depth_stage
            completed.add("depth")
            self._commit(
                working_stages,
                pending_queue,
                state,
                target_config_hash,
                target_project_fingerprint,
                completed,
                False,
            )
            self._maybe_sigkill("depth", sigkill_after_stage)
            voxel_build = build_voxels(
                triangulation_stage["points"],
                validation.views,
                depth_stage["views"],
                validation.config,
                validation.image_spec.width,
                validation.image_spec.height,
                strategy,
                cache.voxel_by_key,
                cache.voxel_by_path,
            )
            voxel_stage = voxel_build.stage
            voxel_math = voxel_build.math
            ledger.extend(voxel_build.events)
            working_stages["voxels"] = voxel_stage
            completed.add("voxels")
            self._commit(
                working_stages,
                pending_queue,
                state,
                target_config_hash,
                target_project_fingerprint,
                completed,
                True,
            )
            self._maybe_sigkill("voxels", sigkill_after_stage)
        else:
            for stage in ("features", "matches", "triangulation", "depth", "voxels"):
                completed.add(stage)
            snapshot = self._commit(
                working_stages,
                pending_queue,
                state,
                target_config_hash,
                target_project_fingerprint,
                completed,
                True,
            )
            self.last_snapshot = snapshot

        failures = sorted_failures(failures)
        budget = {
            "requested": allocation.requested_points,
            "consumed": allocation.consumed_points,
            "remaining": allocation.remaining_points,
            "accepted_view_ids": allocation.accepted_view_ids,
            "pending_view_count": len(pending_queue),
        }
        canonical_payload = _canonical_payload(
            working_stages,
            failures,
            pending_queue,
            budget,
        )
        canonical_digest = stable_hash(canonical_payload)
        exit_code = _exit_code(failures)
        status = "ok"
        if any(failure.code == "E_POINT_BUDGET_PENDING" for failure in failures):
            status = "partial_budget_pending"
        elif failures:
            status = "partial_with_failures"
        failure_counts: dict[str, int] = {}
        for failure in failures:
            failure_counts[failure.code] = failure_counts.get(failure.code, 0) + 1
        stages_summary = {
            "load": {
                "valid_views": len(validation.valid_view_ids),
                "declared_pairs": len(validation.pairs),
                "runnable": validation.runnable,
            },
            "features": {
                "views": len(feature_stage["views"]),
                "points": feature_stage["total_points"],
                "candidate_blocks": feature_stage["total_candidate_blocks"],
                "multiply_adds": feature_stage["multiply_adds"],
            },
            "matches": {
                "pairs": len(match_stage["pairs"]),
                "inliers": match_stage["total_inliers"],
                "multiply_adds": match_stage["multiply_adds"],
            },
            "triangulation": {
                "points": triangulation_stage["triangulated_points"],
                "degenerate": triangulation_stage["degenerate_points"],
            },
            "depth": {
                "views": len(depth_stage["views"]),
                "pixels": depth_stage["pixel_count"],
                "occupied_pixels": depth_stage["occupied_pixels"],
                "conflict_pixels": depth_stage["conflict_pixels"],
            },
            "voxels": {
                "nodes": voxel_stage["node_count"],
                "roots": len(voxel_stage["roots"]),
                "parents": voxel_stage["parent_count"],
                "leaves": voxel_stage["leaf_count"],
                "occupied": voxel_stage["occupied_count"],
                "math": voxel_math,
            },
        }
        report = {
            "engine_version": ENGINE_VERSION,
            "config_hash": target_config_hash,
            "project_fingerprint": target_project_fingerprint,
            "status": status,
            "exit_code": exit_code,
            "strategy": strategy,
            "stage_summaries": stages_summary,
            "budget": budget,
            "failures": _failure_dicts(failures),
            "failure_counts": failure_counts,
            "pending_queue": pending_queue,
            "ledger": {
                "entries": ledger.normalized(),
                "summary": ledger.summary(),
            },
            "snapshot": {
                "loaded": loaded is not None,
                "config_same": config_same,
                "version_change_detected": loaded is not None and not config_same,
                "cache_carried": cache_compatible,
                "resume_used": resume_used,
                "resume_reason": resume_reason,
                "resume_from_stage": resume_from,
                "final_generation": self.last_snapshot.generation if self.last_snapshot else None,
                "run_complete": True,
            },
            "canonical_digest": canonical_digest,
            "effective_workers": effective_workers,
        }
        report["ledger"]["summary"] = ledger.summary()
        result = RunResult(
            stages={key: working_stages[key] for key in working_stages},
            failures=failures,
            pending_queue=pending_queue,
            ledger=ledger,
            report=report,
            canonical_digest=canonical_digest,
            snapshot=self.last_snapshot,
            exit_code=exit_code,
        )
        if write_report is not None:
            result.persist_report(write_report)
        return result

    def _commit(
        self,
        working_stages: dict[str, Any],
        pending_queue: list[dict[str, Any]],
        state: ProjectState,
        config_hash: str,
        project_fingerprint: str,
        completed: set[str],
        run_complete: bool,
    ) -> Snapshot:
        completed_ordered = [
            stage for stage in STAGE_ORDER if stage in completed and stage in working_stages
        ]
        snapshot = self.store.commit(
            working_stages,
            pending_queue,
            config_hash,
            project_fingerprint,
            str(state.data.get("recipe_version", "")),
            completed_ordered,
            run_complete,
            previous=self.store.latest_loaded,
        )
        self.last_snapshot = snapshot
        return snapshot

    def _maybe_sigkill(self, stage: str, selected_stage: str | None) -> None:
        if selected_stage == stage:
            os.kill(os.getpid(), signal.SIGKILL)


def canonical_reconstruction(result: RunResult) -> dict[str, Any]:
    return {
        "stages": {
            stage: result.stages[stage]
            for stage in STAGE_ORDER
            if stage in result.stages
        },
        "failures": [failure.as_dict() for failure in sorted_failures(result.failures)],
        "pending_queue": result.pending_queue,
    }

