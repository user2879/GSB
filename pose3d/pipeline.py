"""Deterministic parallel pipeline, budget arbitration, and reporting."""

from __future__ import annotations

import copy
import json
import math
import os
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .budget import BudgetLedger
from .cache import StageCache
from .config import ENGINE_VERSION, REPORT_SCHEMA_VERSION, SNAPSHOT_SCHEMA_VERSION, SolverConfig, load_config
from .errors import (
    E_IO,
    E_RESIDUAL_BUDGET_EXHAUSTED,
    Failure,
)
from .hashing import sha256_json
from .injections import InjectionSet, load_injections
from .loader import load_inputs
from .models import InputBundle, PointSource
from .snapshot import SnapshotStore
from .solver import PointSolveResult, solve_point


@dataclass
class _AcceptedPoint:
    result: PointSolveResult
    cost: int
    ledger_entry: dict[str, Any]


def _failure_dict(
    code: str,
    keypoint_id: str | None,
    stage: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return Failure(
        code=code,
        keypoint_id=keypoint_id,
        stage=stage,
        message=message,
        details=details or {},
    ).to_dict()


def _sort_failure_dicts(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(item: dict[str, Any]) -> tuple[Any, ...]:
        return (
            item.get("keypoint_id") or "",
            item.get("code") or "",
            item.get("stage") or "",
            item.get("view_index", -1),
            item.get("observation_index", -1),
            item.get("message") or "",
            sha256_json(item),
        )

    unique: dict[str, dict[str, Any]] = {}
    for item in items:
        unique[sha256_json(item)] = item
    return sorted(unique.values(), key=key)


def _unexpected_point_result(
    point: PointSource, cfg: SolverConfig, exc: BaseException
) -> PointSolveResult:
    failure = _failure_dict(
        E_IO,
        point.keypoint_id,
        "solve",
        "unexpected per-keypoint solver exception was isolated",
        {
            "exception_type": type(exc).__name__,
            "message": str(exc),
        },
    )
    record = {
        "keypoint_id": point.keypoint_id,
        "line_number": point.line_number,
        "base_name": point.base_name,
        "instance_suffix": point.instance_suffix,
        "status": "error",
        "error_codes": [E_IO],
        "failures": [failure],
        "observation_count_loaded": len(point.raw_observations),
        "observation_count_valid": len(point.observations),
        "observation_count_final": 0,
        "observations": [obs.to_dict() for obs in point.observations],
        "triangulation": None,
        "disparity": None,
        "outlier_rejection": None,
        "final": None,
        "quantization": None,
        "cache": {
            "stage_keys": {},
            "reused_stages": [],
            "recomputed_stages": [],
            "fully_reused": False,
        },
        "residual_evaluations": 0,
        "injection_hits": dict(point.injection_hits),
    }
    return PointSolveResult(
        keypoint_id=point.keypoint_id,
        record=record,
        residual_evaluations=0,
        status="error",
        injection_hits=dict(point.injection_hits),
    )


def _input_version(
    cfg: SolverConfig,
    bundle: InputBundle,
    injections: InjectionSet,
) -> str:
    return sha256_json(
        {
            "engine_version": ENGINE_VERSION,
            "manifest_version": bundle.input_versions.get("manifest_version"),
            "config_version": bundle.input_versions.get("config_version"),
            "injection_version": bundle.input_versions.get("injection_version")
            or injections.version,
            "observations_sha256": bundle.observations_sha256,
            "version_file_sha256": bundle.version_file_sha256,
            "config_file_sha256": bundle.config_sha256,
            "config_fingerprint": cfg.fingerprint,
            "camera_hash": bundle.camera_hash,
            "injection_fingerprint": injections.fingerprint,
            "image_resolution": list(bundle.image_resolution or []),
        }
    )


def _merge_cache_entries(
    cache: StageCache, results: list[PointSolveResult]
) -> dict[str, dict[str, Any]]:
    entries = {key: copy.deepcopy(value) for key, value in cache.entries.items()}
    for result in results:
        for key, value in result.cache_additions.items():
            if key not in entries:
                entries[key] = copy.deepcopy(value)
    return entries


def _stage_data_for_prefix(
    accepted: list[_AcceptedPoint],
) -> dict[str, dict[str, Any]]:
    data: dict[str, dict[str, Any]] = {
        "load": {},
        "triangulation": {},
        "rejection": {},
        "finalization": {},
    }
    for item in accepted:
        for stage in data:
            payload = item.result.stage_payloads.get(stage)
            if payload is not None:
                data[stage][item.result.keypoint_id] = copy.deepcopy(payload)
    return data


def _cache_entries_for_prefix(
    accepted: list[_AcceptedPoint],
    merged_entries: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    keys: set[str] = set()
    for item in accepted:
        keys.update(item.result.record.get("cache", {}).get("stage_keys", {}).values())
    return {key: copy.deepcopy(merged_entries[key]) for key in sorted(keys) if key in merged_entries}


def _update_injection_report(
    initial_report: list[dict[str, Any]],
    results: list[PointSolveResult],
    accepted_ids: set[str],
) -> list[dict[str, Any]]:
    hits: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        if result.keypoint_id not in accepted_ids:
            continue
        for injection_id, business_code in result.injection_hits.items():
            hits.setdefault(injection_id, []).append(
                {
                    "keypoint_id": result.keypoint_id,
                    "business_code": business_code,
                    "point_status": result.record.get("status"),
                    "point_error_codes": result.record.get("error_codes", []),
                }
            )
    report: list[dict[str, Any]] = []
    for item in initial_report:
        record = copy.deepcopy(item)
        observations = sorted(
            hits.get(str(item.get("id")), []), key=lambda value: value["keypoint_id"]
        )
        record["hit"] = bool(observations)
        if observations:
            record["affected_keypoints"] = [value["keypoint_id"] for value in observations]
            record["observed_business_codes"] = sorted(
                {value["business_code"] for value in observations}
            )
            record["matched"] = observations
        else:
            record["observed_business_codes"] = []
        report.append(record)
    return sorted(report, key=lambda value: str(value.get("id", "")))


def _precision_report(points: list[dict[str, Any]]) -> dict[str, Any]:
    final_residuals: list[float] = []
    low_point_deltas: list[float] = []
    low_residual_deltas: list[float] = []
    high_point_deltas: list[float] = []
    high_residual_deltas: list[float] = []
    static_point_deltas: list[float] = []
    static_residual_deltas: list[float] = []
    first_exceeded: dict[str, int] = {}
    flip_count = 0
    successful = 0
    for point in points:
        if point.get("status") != "ok" or not point.get("final"):
            continue
        successful += 1
        final_residuals.extend(
            float(item["residual_px"]) for item in point["final"].get("residuals", [])
        )
        quantization = point.get("quantization") or {}
        if quantization.get("status") == "ok":
            static_point_deltas.append(
                float(
                    math.sqrt(
                        sum(
                            (float(a) - float(b)) ** 2
                            for a, b in zip(
                                quantization["static_primary_point"]["quantized"],
                                point["final"]["point"],
                                strict=True,
                            )
                        )
                    )
                )
            )
            static_quant = quantization["static_primary_residuals"]["quantized"]
            base_residual_values = [
                float(item["residual_px"])
                for item in point["final"].get("residuals", [])
            ]
            if static_quant and base_residual_values:
                static_residual_deltas.append(
                    max(
                        abs(float(a) - float(b))
                        for a, b in zip(static_quant, base_residual_values, strict=True)
                    )
                )
            for key, point_target, residual_target in (
                ("low", low_point_deltas, low_residual_deltas),
                ("high", high_point_deltas, high_residual_deltas),
            ):
                propagation = quantization.get(f"error_propagation_{key}") or {}
                for stage in propagation.get("stages", []):
                    if stage.get("point_delta_l2") is not None:
                        point_target.append(float(stage["point_delta_l2"]))
                    if stage.get("residual_delta_max_abs") is not None:
                        residual_target.append(float(stage["residual_delta_max_abs"]))
                exceeded = propagation.get("first_tolerance_exceeded_stage")
                if exceeded:
                    first_exceeded[exceeded] = first_exceeded.get(exceeded, 0) + 1
            flip = quantization.get("removal_flip") or {}
            if flip.get("detected"):
                flip_count += 1
    rmse = (
        math.sqrt(sum(value * value for value in final_residuals) / len(final_residuals))
        if final_residuals
        else None
    )
    return {
        "float_format": "IEEE-754 binary64; JSON emits shortest round-trip decimal",
        "minimum_significant_digits": 6,
        "successful_points": successful,
        "final_residual_count": len(final_residuals),
        "final_residual_max_px": max(final_residuals) if final_residuals else None,
        "final_residual_rmse_px": rmse,
        "static_primary_point_delta_l2_max": max(static_point_deltas)
        if static_point_deltas
        else None,
        "static_primary_residual_delta_max_px": max(static_residual_deltas)
        if static_residual_deltas
        else None,
        "low_chain_point_delta_l2_max": max(low_point_deltas)
        if low_point_deltas
        else None,
        "low_chain_residual_delta_max_px": max(low_residual_deltas)
        if low_residual_deltas
        else None,
        "high_chain_point_delta_l2_max": max(high_point_deltas)
        if high_point_deltas
        else None,
        "high_chain_residual_delta_max_px": max(high_residual_deltas)
        if high_residual_deltas
        else None,
        "first_tolerance_exceeded_stage_counts": dict(sorted(first_exceeded.items())),
        "removal_flip_count": flip_count,
    }


def _write_report(path: str | Path, report: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temp_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temp_path, output)


class SolverPipeline:
    def __init__(
        self,
        *,
        config_path: str | Path,
        snapshot_dir: str | Path,
        injections_path: str | Path | None = None,
        input_dir: str | Path | None = None,
        fresh: bool = False,
    ):
        self.config_path = Path(config_path)
        self.snapshot_dir = Path(snapshot_dir)
        self.input_dir = Path(input_dir) if input_dir is not None else self.config_path.parent
        self.injections_path = Path(injections_path) if injections_path else None
        self.fresh = bool(fresh)
        self.cfg: SolverConfig | None = None
        self.injections = InjectionSet()
        self.bundle: InputBundle | None = None
        self.input_version = ""
        self.cache = StageCache()
        self.recovery_decision = "no_snapshot"

    def prepare(self) -> None:
        self.cfg = load_config(self.config_path)
        self.injections = load_injections(self.injections_path)
        self.bundle = load_inputs(
            self.input_dir,
            self.cfg,
            config_path=self.config_path,
            injections=self.injections,
        )
        self.input_version = _input_version(self.cfg, self.bundle, self.injections)
        store = SnapshotStore(self.snapshot_dir)
        manifest, cache_payload, recovery = store.load()
        if self.fresh:
            self.recovery_decision = "fresh_run_cache_ignored"
            self.cache = StageCache()
        elif (
            manifest is None
            or manifest.get("engine_version") != ENGINE_VERSION
            or manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION
        ):
            self.cache = StageCache()
            self.recovery_decision = (
                recovery
                if manifest is None
                else "incompatible_snapshot_cache_ignored"
            )
        else:
            self.cache = StageCache.from_serialized(cache_payload)
            if manifest.get("input_version") != self.input_version:
                self.recovery_decision = "input_version_changed_cache_partial"
            else:
                completed = set(manifest.get("completed_keypoints", []))
                all_points = set(self.bundle.sorted_keypoints)
                if completed == all_points:
                    self.recovery_decision = "exact_snapshot_available"
                else:
                    self.recovery_decision = "partial_checkpoint_same_input_version"

    def _run_workers(self) -> dict[str, PointSolveResult]:
        assert self.cfg is not None and self.bundle is not None
        points = self.bundle.points
        results: dict[str, PointSolveResult] = {}
        with ThreadPoolExecutor(max_workers=self.cfg.max_workers) as executor:
            future_map: dict[Future[PointSolveResult], str] = {}
            for keypoint_id in self.bundle.sorted_keypoints:
                point = points[keypoint_id]
                future = executor.submit(
                    solve_point,
                    point,
                    self.bundle,
                    self.cfg,
                    self.injections,
                    self.cache,
                )
                future_map[future] = keypoint_id
            for future, keypoint_id in future_map.items():
                try:
                    results[keypoint_id] = future.result()
                except BaseException as exc:  # isolate one point from the rest
                    results[keypoint_id] = _unexpected_point_result(
                        points[keypoint_id], self.cfg, exc
                    )
        return {key: results[key] for key in sorted(results)}

    def run(self) -> dict[str, Any]:
        self.prepare()
        assert self.cfg is not None and self.bundle is not None
        cfg = self.cfg
        bundle = self.bundle
        results = self._run_workers()
        ordered_results = [results[key] for key in bundle.sorted_keypoints]
        merged_cache_entries = _merge_cache_entries(self.cache, ordered_results)

        ledger = BudgetLedger.create(cfg.global_residual_budget)
        accepted: list[_AcceptedPoint] = []
        pending_queue: list[str] = []
        budget_blocked = False
        final_records: dict[str, dict[str, Any]] = {}
        accepted_failures: list[dict[str, Any]] = []
        pending_failures: list[dict[str, Any]] = []
        snapshot_store = SnapshotStore(self.snapshot_dir)
        checkpoint_reports: list[dict[str, Any]] = []
        last_commit: dict[str, Any] | None = None
        checkpoint_consumed = 0
        run_has_recomputed = False

        def commit_prefix(pending: list[str]) -> dict[str, Any]:
            nonlocal last_commit
            stage_data = _stage_data_for_prefix(accepted)
            cache_entries = _cache_entries_for_prefix(accepted, merged_cache_entries)
            commit = snapshot_store.commit(
                input_version=self.input_version,
                config_hash=cfg.fingerprint,
                camera_hash=bundle.camera_hash,
                completed_keypoints=[item.result.keypoint_id for item in accepted],
                consumed_budget=checkpoint_consumed,
                pending_queue=pending,
                stage_payloads=stage_data,
                cache_entries=cache_entries,
                snapshot_commit_interval=cfg.snapshot_commit_interval,
            )
            checkpoint_reports.append(
                {
                    "sequence": len(checkpoint_reports) + 1,
                    "version": commit["version"],
                    "decision": commit["decision"],
                    "completed_count": len(accepted),
                    "consumed_budget": checkpoint_consumed,
                    "pending_count": len(pending),
                }
            )
            last_commit = commit
            return commit

        for result in ordered_results:
            keypoint_id = result.keypoint_id
            if budget_blocked:
                pending_queue.append(keypoint_id)
                record = copy.deepcopy(result.record)
                failure = _failure_dict(
                    E_RESIDUAL_BUDGET_EXHAUSTED,
                    keypoint_id,
                    "scheduler",
                    "global residual budget is exhausted; keypoint remains queued",
                    {
                        "required_evaluations": result.residual_evaluations,
                        "remaining_budget": ledger.remaining,
                        "queue_position": len(pending_queue),
                    },
                )
                record["budget_pending"] = True
                record["status"] = "error"
                record["final"] = None
                record["quantization"] = None
                record["observation_count_final"] = 0
                failures = record.get("failures", []) + [failure]
                record["failures"] = _sort_failure_dicts(failures)
                record["error_codes"] = sorted(
                    {item["code"] for item in record["failures"]}
                )
                final_records[keypoint_id] = record
                pending_failures.append(failure)
                continue

            cost = int(result.residual_evaluations)
            if cost > ledger.remaining:
                budget_blocked = True
                ledger.exhausted = True
                pending_queue.append(keypoint_id)
                failure = _failure_dict(
                    E_RESIDUAL_BUDGET_EXHAUSTED,
                    keypoint_id,
                    "scheduler",
                    "global residual budget cannot cover the next keypoint",
                    {
                        "required_evaluations": cost,
                        "remaining_budget": ledger.remaining,
                        "queue_position": len(pending_queue),
                    },
                )
                record = copy.deepcopy(result.record)
                record["budget_pending"] = True
                record["status"] = "error"
                record["final"] = None
                record["quantization"] = None
                record["observation_count_final"] = 0
                failures = record.get("failures", []) + [failure]
                record["failures"] = _sort_failure_dicts(failures)
                record["error_codes"] = sorted(
                    {item["code"] for item in record["failures"]}
                )
                final_records[keypoint_id] = record
                pending_failures.append(failure)
                continue

            ledger_entry = ledger.charge(keypoint_id, cost)
            checkpoint_consumed += cost
            accepted_result = _AcceptedPoint(
                result=result, cost=cost, ledger_entry=ledger_entry
            )
            accepted.append(accepted_result)
            final_records[keypoint_id] = copy.deepcopy(result.record)
            accepted_failures.extend(result.record.get("failures", []))
            if result.record.get("cache", {}).get("recomputed_stages"):
                run_has_recomputed = True
            if (
                run_has_recomputed
                and len(accepted) % cfg.snapshot_commit_interval == 0
            ):
                commit_prefix([])

        final_commit = commit_prefix(pending_queue)
        if (
            checkpoint_reports
            and checkpoint_reports[-1]["version"] != final_commit["version"]
        ):
            pass
        deduplicated_checkpoints: dict[tuple[int, str], dict[str, Any]] = {}
        for checkpoint in checkpoint_reports:
            deduplicated_checkpoints[
                (int(checkpoint["completed_count"]), str(checkpoint["version"]))
            ] = checkpoint
        checkpoints = sorted(
            deduplicated_checkpoints.values(),
            key=lambda item: int(item["sequence"]),
        )
        decisions = {item["decision"] for item in checkpoints}
        if "created" in decisions:
            overall_snapshot_decision = "created"
        elif "replaced" in decisions:
            overall_snapshot_decision = "replaced"
        else:
            overall_snapshot_decision = final_commit["decision"]
        snapshot_report = {
            "input_version": self.input_version,
            "global_version": final_commit["version"],
            "decision": overall_snapshot_decision,
            "recovery_decision": self.recovery_decision,
            "manifest_path": str(
                Path(self.snapshot_dir) / "current" / "manifest.json"
            ),
            "completed_keypoints": final_commit["completed_keypoints"],
            "consumed_budget": final_commit["consumed_budget"],
            "pending_queue": sorted(pending_queue),
            "checkpoints": checkpoints,
        }

        points_report = [final_records[key] for key in sorted(final_records)]
        failures = (
            bundle.sorted_global_failures
            + accepted_failures
            + pending_failures
        )
        # Point records already contain per-observation failures; the global list
        # intentionally mirrors them once for report consumers.
        failures = _sort_failure_dicts(failures)
        accepted_ids = {item.result.keypoint_id for item in accepted}
        injection_report = _update_injection_report(
            bundle.injection_report, ordered_results, accepted_ids
        )
        success_count = sum(
            1 for point in points_report if point.get("status") == "ok"
        )
        error_count = sum(
            1 for point in points_report if point.get("status") == "error"
        )
        invalid_observations = sum(
            1 for failure in failures if failure.get("code") == "E_OBS_INVALID"
        )
        removed_total = sum(
            int((point.get("outlier_rejection") or {}).get("rounds_executed", 0) or 0)
            for point in points_report
            if point.get("status") == "ok"
        )
        reused_points = sorted(
            point["keypoint_id"]
            for point in points_report
            if point.get("cache", {}).get("fully_reused") and point["keypoint_id"] in accepted_ids
        )
        recomputed_points = sorted(
            point["keypoint_id"]
            for point in points_report
            if not point.get("cache", {}).get("fully_reused")
            and point["keypoint_id"] in accepted_ids
        )
        stage_reused_counts = {
            stage: sum(
                1
                for point in points_report
                if stage in point.get("cache", {}).get("reused_stages", [])
                and point["keypoint_id"] in accepted_ids
            )
            for stage in ("load", "triangulation", "rejection", "finalization")
        }
        stage_recomputed_counts = {
            stage: sum(
                1
                for point in points_report
                if stage in point.get("cache", {}).get("recomputed_stages", [])
                and point["keypoint_id"] in accepted_ids
            )
            for stage in ("load", "triangulation", "rejection", "finalization")
        }
        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "engine_version": ENGINE_VERSION,
            "run_mode": "fresh" if self.fresh else "incremental",
            "input": {
                "input_dir": str(bundle.input_dir),
                "observations_sha256": bundle.observations_sha256,
                "version_file_sha256": bundle.version_file_sha256,
                "versions": bundle.input_versions,
                "manifest_version": bundle.input_versions.get("manifest_version"),
                "config_hash": cfg.fingerprint,
                "config_file_sha256": bundle.config_sha256,
                "camera_hash": bundle.camera_hash,
                "injection_version": self.injections.version,
                "injection_fingerprint": self.injections.fingerprint,
                "image_resolution": list(bundle.image_resolution or []),
                "input_version": self.input_version,
            },
            "config": cfg.describe(),
            "summary": {
                "keypoints_total": len(points_report),
                "succeeded": success_count,
                "failed": error_count,
                "pending": len(pending_queue),
                "observation_valid_total": sum(
                    int(point.get("observation_count_valid", 0))
                    for point in points_report
                ),
                "observation_final_total": sum(
                    int(point.get("observation_count_final", 0))
                    for point in points_report
                ),
                "invalid_observation_failures": invalid_observations,
                "removed_observations_in_successful_points": removed_total,
                "residual_budget_consumed": ledger.initial - ledger.remaining,
                "residual_budget_remaining": ledger.remaining,
            },
            "points": points_report,
            "failures": failures,
            "budget": ledger.to_dict(),
            "pending_queue": sorted(pending_queue),
            "injections": injection_report,
            "cache": {
                "reused_points": reused_points,
                "recomputed_points": recomputed_points,
                "stage_reused_counts": stage_reused_counts,
                "stage_recomputed_counts": stage_recomputed_counts,
            },
            "snapshot": snapshot_report,
            "precision": _precision_report(points_report),
        }
        return report


def run_pipeline(
    *,
    input_dir: str | Path,
    config_path: str | Path,
    snapshot_dir: str | Path,
    injections_path: str | Path | None = None,
    output_path: str | Path | None = None,
    fresh: bool = False,
) -> dict[str, Any]:
    """Run the solver and optionally write a deterministic JSON report."""

    pipeline = SolverPipeline(
        config_path=Path(config_path),
        input_dir=Path(input_dir),
        snapshot_dir=snapshot_dir,
        injections_path=injections_path,
        fresh=fresh,
    )
    report = pipeline.run()
    if output_path is not None:
        _write_report(output_path, report)
    return report
