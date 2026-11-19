"""End-to-end diagnostic pipeline with content-addressed stage reuse."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .calibration import CalibrationResult, calibrate_frame, calibration_key
from .cache import (
    load_calibration,
    load_metrics,
    load_unmix,
    prune_stage,
    save_calibration,
    save_metrics,
    save_unmix,
    stage_dir,
)
from .config import CubeConfig, PipelineConfig
from .drift import DriftSequenceResult, estimate_drift_sequence, shift_stack_to_reference
from .errors import ErrorCode, WorkbenchError
from .generator import load_endmembers, load_frame_array, load_manifest, load_truth
from .io_utils import canonical_json, key_for, read_json, sha256_text, write_json
from .metrics import FrameMetrics, aggregate_frame_metrics, compute_frame_metrics
from .unmix import EndmemberModel, UnmixResult, build_endmember_model, unmix_frame, unmix_key


@dataclass(frozen=True)
class FrameStageRecord:
    frame: int
    calibration_source: str
    calibration_version: str
    drift_source: str
    drift_version: str
    unmix_source: str
    unmix_version: str
    metrics_source: str
    metrics_version: str
    pre_metrics_source: Optional[str]
    pre_metrics_version: Optional[str]
    compensated: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame": self.frame,
            "calibration": {"source": self.calibration_source, "version": self.calibration_version},
            "drift": {"source": self.drift_source, "version": self.drift_version},
            "unmix": {"source": self.unmix_source, "version": self.unmix_version},
            "metrics": {"source": self.metrics_source, "version": self.metrics_version},
            "pre_metrics": (
                {"source": self.pre_metrics_source, "version": self.pre_metrics_version}
                if self.pre_metrics_source is not None
                else None
            ),
            "compensated": self.compensated,
        }


@dataclass(frozen=True)
class PipelineResult:
    cube_root: Path
    output_dir: Path
    config: PipelineConfig
    model: EndmemberModel
    drift: DriftSequenceResult
    post_unmix: UnmixResult
    pre_unmix: UnmixResult
    post_frame_metrics: Tuple[FrameMetrics, ...]
    pre_frame_metrics: Tuple[FrameMetrics, ...]
    post_metrics: Dict[str, Any]
    pre_metrics: Dict[str, Any]
    frames: Tuple[FrameStageRecord, ...]
    report: Dict[str, Any]
    report_path: Path

    @property
    def closure_verified(self) -> bool:
        return bool(self.report.get("closure", {}).get("verified", False))


def _model_from_inputs(
    cube_root: Path,
    manifest: Dict[str, Any],
    config: PipelineConfig,
) -> EndmemberModel:
    if config.unmix.endmember_file:
        path = Path(config.unmix.endmember_file)
        if not path.is_file():
            raise WorkbenchError(ErrorCode.E_CUBE_FORMAT, "endmember override file is missing", {"path": str(path)})
        try:
            endmembers = np.load(path, allow_pickle=False)
        except Exception as exc:
            raise WorkbenchError(ErrorCode.E_CUBE_FORMAT, "cannot load endmember override", {"path": str(path)}) from exc
        names = tuple(str(value) for value in manifest.get("fluorophore_names", []))
    else:
        endmembers = load_endmembers(cube_root, manifest)
        names = tuple(str(value) for value in manifest["fluorophore_names"])
    return build_endmember_model(endmembers, names, config.unmix)


def _drift_cache_key(
    calibrated_hashes: Sequence[str],
    reference_centers: Sequence[Sequence[float]],
    truth_shifts: Sequence[Sequence[float]],
    config: PipelineConfig,
) -> str:
    return key_for(
        {
            "stage": "drift-estimate-v3",
            "calibrated_hashes": list(calibrated_hashes),
            "reference_centers": [list(value) for value in reference_centers],
            "truth_shifts": [list(value) for value in truth_shifts],
            "config": {
                "max_shift_px": config.drift.max_shift_px,
                "window_radius_px": config.drift.window_radius_px,
                "background_percentile": config.drift.background_percentile,
                "min_center_signal_dn": config.drift.min_center_signal_dn,
            },
        }
    )


def _unmix_input_version(
    calibration_results: Sequence[CalibrationResult],
    drift: DriftSequenceResult,
    compensated: bool,
) -> str:
    return key_for(
        {
            "stage": "unmix-input-v1",
            "compensated": bool(compensated),
            "bands": [
                {
                    "corrected_sha256": result.corrected_sha256,
                    "invalid_sha256": result.invalid_sha256,
                    "shift_px": (
                        [drift.frames[index].estimated_dx, drift.frames[index].estimated_dy]
                        if compensated
                        else [0.0, 0.0]
                    ),
                }
                for index, result in enumerate(calibration_results)
            ],
        }
    )


def _drift_from_dict(value: Dict[str, Any]) -> DriftSequenceResult:
    from .drift import DriftFrameResult

    frames = []
    for item in value["frames"]:
        frames.append(
            DriftFrameResult(
                frame=int(item["frame"]),
                estimated_dx=float(item["estimated_shift_px"][0]),
                estimated_dy=float(item["estimated_shift_px"][1]),
                truth_dx=float(item["truth_shift_px"][0]),
                truth_dy=float(item["truth_shift_px"][1]),
                residual_dx=float(item["estimate_residual_px"][0]),
                residual_dy=float(item["estimate_residual_px"][1]),
                residual_norm=float(item["estimate_residual_norm_px"]),
                max_estimated_norm=float(item["estimated_shift_norm_px"]),
                centers_observed=tuple(tuple(float(v) for v in pair) for pair in item["centers_observed_xy"]),
            )
        )
    return DriftSequenceResult(
        frames=tuple(frames),
        max_residual_px=float(value["max_estimate_residual_px"]),
        rms_residual_px=float(value["rms_estimate_residual_px"]),
        max_shift_px=float(value["max_estimated_shift_px"]),
        compensation_applied=bool(value["compensation_enabled"]),
    )


def _metrics_cache_key(
    unmix_version: str,
    truth: Dict[str, Any],
    tolerance: float,
    compensated: bool,
) -> str:
    return key_for(
        {
            "stage": "concentration-metrics-v2",
            "unmix_version": unmix_version,
            "truth_sha256": sha256_text(canonical_json(truth)),
            "tolerance": float(tolerance),
            "compensated": bool(compensated),
        }
    )


def _stage_summary(records: Sequence[FrameStageRecord], field: str) -> Dict[str, Any]:
    source_field = f"{field}_source"
    version_field = f"{field}_version"
    reused = sum(1 for record in records if getattr(record, source_field) == "reused")
    computed = sum(1 for record in records if getattr(record, source_field) == "computed")
    versions = sorted({getattr(record, version_field) for record in records if getattr(record, version_field)})
    return {"reused_frames": reused, "computed_frames": computed, "versions": versions}


def _single_frame_divergence(
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> Optional[Dict[str, Any]]:
    left = np.asarray(expected)
    right = np.asarray(actual)
    if left.shape != right.shape:
        return {"kind": "shape", "expected": list(left.shape), "actual": list(right.shape)}
    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
    bad = delta > (atol + rtol * np.abs(left.astype(np.float64)))
    if not np.any(bad):
        return None
    flat_index = int(np.flatnonzero(bad.ravel(order="C"))[0])
    index = tuple(int(v) for v in np.unravel_index(flat_index, bad.shape))
    return {
        "kind": "value",
        "array_index": list(index),
        "pixel": list(index[-2:]),
        "expected": float(left[index]),
        "actual": float(right[index]),
        "absolute_delta": float(delta[index]),
    }


class PipelineRunner:
    def __init__(self, cube_root: str | Path, config: PipelineConfig):
        self.cube_root = Path(cube_root)
        self.config = config

    def run(
        self,
        output_dir: str | Path,
        *,
        verify_closure: bool = False,
        keep_cache: bool = True,
    ) -> PipelineResult:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        cache_root = output / "cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        manifest = load_manifest(self.cube_root, verify_hashes=True)
        cube_config = CubeConfig.from_dict(manifest["cube"])
        height, width = cube_config.height, cube_config.width
        frame_count = cube_config.frames
        model = _model_from_inputs(self.cube_root, manifest, self.config)
        truths = [load_truth(self.cube_root, frame) for frame in range(frame_count)]
        truth_shifts = [truth["drift_xy"] for truth in truths]

        calibration_results: List[CalibrationResult] = []
        calibration_lookups: List[Tuple[str, str]] = []
        active_calibration: set[str] = set()
        for frame, record in enumerate(manifest["frames"]):
            raw = load_frame_array(self.cube_root, record, "raw", (height, width))
            dark = load_frame_array(self.cube_root, record, "dark", (height, width))
            flat = load_frame_array(self.cube_root, record, "flat", (height, width))
            key = calibration_key(raw, dark, flat, self.config.calibration)
            active_calibration.add(key)
            hit = load_calibration(cache_root, key)
            if hit is None:
                result = calibrate_frame(raw, dark, flat, self.config.calibration, frame_index=frame)
                save_calibration(cache_root, key, result)
                source = "computed"
            else:
                result = hit[0]
                source = "reused"
            calibration_results.append(result)
            calibration_lookups.append((source, key))

        calibrated_stack = np.stack([result.corrected for result in calibration_results], axis=0)
        invalid_stack = np.stack([result.invalid_mask for result in calibration_results], axis=0)
        drift_key = _drift_cache_key(
            [result.corrected_sha256 for result in calibration_results],
            manifest["reference_centers"],
            truth_shifts,
            self.config,
        )
        drift_cache_path = stage_dir(cache_root, "drift", drift_key) / "result.json"
        if drift_cache_path.is_file():
            drift = _drift_from_dict(read_json(drift_cache_path))
            drift_source = "reused"
        else:
            drift = estimate_drift_sequence(
                calibrated_stack,
                invalid_stack,
                manifest["reference_centers"],
                truth_shifts,
                self.config.drift,
            )
            drift_cache_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(drift_cache_path, drift.to_dict())
            drift_source = "computed"

        if self.config.drift.compensate:
            compensated_bands = []
            compensated_invalid = []
            for frame in range(frame_count):
                shifted, shifted_invalid = shift_stack_to_reference(
                    calibrated_stack[frame],
                    invalid_stack[frame],
                    drift.frames[frame].estimated_dx,
                    drift.frames[frame].estimated_dy,
                )
                compensated_bands.append(shifted)
                compensated_invalid.append(shifted_invalid)
            final_stack = np.stack(compensated_bands, axis=0)
            final_invalid = np.stack(compensated_invalid, axis=0)
        else:
            final_stack = calibrated_stack
            final_invalid = invalid_stack
        pre_stack = calibrated_stack
        pre_invalid_stack = invalid_stack
        quantization_upper_bound = max(result.quantization_upper_bound for result in calibration_results)

        post_input_version = _unmix_input_version(calibration_results, drift, self.config.drift.compensate)
        post_unmix_key = unmix_key(model, post_input_version, quantization_upper_bound, self.config.unmix)
        post_hit = load_unmix(cache_root, post_unmix_key)
        if post_hit is None:
            full_invalid = np.any(final_invalid != 0, axis=0).astype(np.uint8)
            post_unmix = unmix_frame(
                final_stack,
                full_invalid,
                model,
                self.config.unmix,
                quantization_upper_bound=quantization_upper_bound,
                frame_index=-1,
            )
            save_unmix(cache_root, post_unmix_key, post_unmix)
            post_unmix_source = "computed"
        else:
            post_unmix = post_hit[0]
            post_unmix_source = "reused"

        pre_input_version = _unmix_input_version(calibration_results, drift, False)
        pre_unmix_key = unmix_key(model, pre_input_version, quantization_upper_bound, self.config.unmix)
        if self.config.drift.compensate:
            pre_hit = load_unmix(cache_root, pre_unmix_key)
            if pre_hit is None:
                full_pre_invalid = np.any(pre_invalid_stack != 0, axis=0).astype(np.uint8)
                pre_unmix = unmix_frame(
                    pre_stack,
                    full_pre_invalid,
                    model,
                    self.config.unmix,
                    quantization_upper_bound=quantization_upper_bound,
                    frame_index=-1,
                )
                save_unmix(cache_root, pre_unmix_key, pre_unmix)
                pre_unmix_source = "computed"
            else:
                pre_unmix = pre_hit[0]
                pre_unmix_source = "reused"
        else:
            pre_unmix = post_unmix
            pre_unmix_key = post_unmix_key
            pre_unmix_source = post_unmix_source

        post_metrics_frames: List[FrameMetrics] = []
        pre_metrics_frames: List[FrameMetrics] = []
        frame_records: List[FrameStageRecord] = []
        active_metrics: set[str] = set()
        for frame in range(frame_count):
            post_metrics_key = _metrics_cache_key(
                post_unmix_key,
                truths[frame],
                self.config.concentration_tolerance,
                self.config.drift.compensate,
            )
            active_metrics.add(post_metrics_key)
            post_hit_metrics = load_metrics(cache_root, post_metrics_key)
            if post_hit_metrics is None:
                post_metric = compute_frame_metrics(
                    cube_config,
                    truths[frame],
                    model,
                    post_unmix,
                    frame_index=frame,
                    compensated=self.config.drift.compensate,
                    tolerance=self.config.concentration_tolerance,
                )
                save_metrics(cache_root, post_metrics_key, post_metric)
                post_metrics_source = "computed"
            else:
                post_metric = post_hit_metrics[0]
                post_metrics_source = "reused"
            post_metrics_frames.append(post_metric)

            if self.config.drift.compensate:
                pre_metrics_key = _metrics_cache_key(
                    pre_unmix_key,
                    truths[frame],
                    self.config.concentration_tolerance,
                    False,
                )
                active_metrics.add(pre_metrics_key)
                pre_hit_metrics = load_metrics(cache_root, pre_metrics_key)
                if pre_hit_metrics is None:
                    pre_metric = compute_frame_metrics(
                        cube_config,
                        truths[frame],
                        model,
                        pre_unmix,
                        frame_index=frame,
                        compensated=False,
                        tolerance=self.config.concentration_tolerance,
                    )
                    save_metrics(cache_root, pre_metrics_key, pre_metric)
                    pre_metrics_source = "computed"
                else:
                    pre_metric = pre_hit_metrics[0]
                    pre_metrics_source = "reused"
                pre_metrics_version = pre_metrics_key
            else:
                pre_metric = post_metric
                pre_metrics_source = post_metrics_source
                pre_metrics_version = post_metrics_key
            pre_metrics_frames.append(pre_metric)

            frame_records.append(
                FrameStageRecord(
                    frame=frame,
                    calibration_source=calibration_lookups[frame][0],
                    calibration_version=calibration_lookups[frame][1],
                    drift_source=drift_source,
                    drift_version=drift_key,
                    unmix_source=post_unmix_source,
                    unmix_version=post_unmix_key,
                    metrics_source=post_metrics_source,
                    metrics_version=post_metrics_key,
                    pre_metrics_source=pre_metrics_source,
                    pre_metrics_version=pre_metrics_version,
                    compensated=self.config.drift.compensate,
                )
            )

        post_aggregate = aggregate_frame_metrics(model, post_metrics_frames, self.config.concentration_tolerance)
        pre_aggregate = aggregate_frame_metrics(model, pre_metrics_frames, self.config.concentration_tolerance)
        compensation_evaluation = self._compensation_evaluation(pre_aggregate, post_aggregate)
        warnings = []
        if model.ill_conditioned:
            warnings.append(
                {
                    "code": "ILL_CONDITIONED_ENDMEMBERS",
                    "message": "endmember condition number exceeds warning threshold",
                    "condition_number": model.condition_number,
                    "threshold": model.warn_threshold,
                }
            )
        if self.config.drift.compensate and not compensation_evaluation["improved"]:
            warnings.append(
                {
                    "code": "COMPENSATION_NOT_IMPROVED",
                    "message": "drift compensation did not reduce the declared aggregate error metric",
                    "pre": compensation_evaluation["pre_absolute_error_mean"],
                    "post": compensation_evaluation["post_absolute_error_mean"],
                }
            )
        calibration_stats = {
            "saturated_pixels": int(sum(result.saturated_pixels for result in calibration_results)),
            "zero_flat_pixels": int(sum(result.zero_flat_pixels for result in calibration_results)),
            "invalid_pixels": int(sum(result.invalid_pixels for result in calibration_results)),
            "quantization_upper_bound_max_dn": float(quantization_upper_bound),
            "quantization_upper_bound_mean_dn": float(
                np.mean([result.quantization_upper_bound for result in calibration_results])
            ),
        }
        unmix_stats = {
            "condition_number": model.condition_number,
            "condition_warn_threshold": model.warn_threshold,
            "condition_fatal_threshold": model.fatal_threshold,
            "matrix_ill_conditioned": model.ill_conditioned,
            "ill_conditioned_pixels": post_unmix.ill_conditioned_pixels,
            "invalid_pixels_in_final_map": int(np.count_nonzero(post_unmix.invalid_mask)),
            "residual_mean_dn": float(post_unmix.record["residual_mean_dn"]),
            "residual_p95_dn": float(post_unmix.record["residual_p95_dn"]),
            "global_cube_stage": True,
        }
        stage_reuse = {
            "calibration": _stage_summary(frame_records, "calibration"),
            "drift": {
                "source": drift_source,
                "version": drift_key,
                "reused_frames": frame_count if drift_source == "reused" else 0,
                "computed_frames": frame_count if drift_source == "computed" else 0,
            },
            "unmix": _stage_summary(frame_records, "unmix"),
            "pre_unmix": {
                "source": pre_unmix_source,
                "version": pre_unmix_key,
                "reused_frames": frame_count if pre_unmix_source == "reused" else 0,
                "computed_frames": frame_count if pre_unmix_source == "computed" else 0,
            }
            if self.config.drift.compensate
            else {"source": "reused", "version": pre_unmix_key, "reused_frames": frame_count, "computed_frames": 0},
            "metrics": _stage_summary(frame_records, "metrics"),
            "pre_metrics": _stage_summary(
                [record for record in frame_records if record.pre_metrics_source is not None], "pre_metrics"
            )
            if self.config.drift.compensate
            else {"reused_frames": 0, "computed_frames": 0, "versions": []},
        }
        model_record = {
            "order": model.order,
            "names": list(model.names),
            "endmember_sha256": model.endmember_sha256,
            "condition_number": model.condition_number,
            "warn_threshold": model.warn_threshold,
            "fatal_threshold": model.fatal_threshold,
            "ill_conditioned": model.ill_conditioned,
        }
        report: Dict[str, Any] = {
            "format_version": 1,
            "cube": {
                "content_hash": manifest["content_hash"],
                "shape": manifest["shape"],
                "injections": manifest["injections"],
            },
            "config": self.config.to_dict(),
            "model": model_record,
            "calibration": calibration_stats,
            "drift": {**drift.to_dict(), "source": drift_source, "version": drift_key},
            "unmix": unmix_stats,
            "metrics": {
                "pre_compensation": pre_aggregate,
                "post_compensation": post_aggregate,
                "compensation_evaluation": compensation_evaluation,
                "interval_definition": {
                    "formula_version": "concentration-interval-v2",
                    "residual_expansion": {
                        "band_count_factor": "sqrt(band_count)",
                        "confidence_factor": 2.0,
                    },
                    "components": [
                        "calibration_quantization",
                        "unmix_residual",
                        "condition_amplification",
                        "float_roundoff",
                    ],
                },
            },
            "incremental": {
                "deduplication_scope": "calibration, drift, cube-level unmix, and per-band metrics",
                "stage_reuse": stage_reuse,
                "frames": [record.to_dict() for record in frame_records],
            },
            "warnings": warnings,
            "closure": {"verified": False, "reason": "not requested"},
        }
        report["run_hash"] = key_for(report)
        report_path = output / "diagnostic_report.json"
        write_json(report_path, report)
        write_json(
            output / "frame_versions.json",
            {"run_hash": report["run_hash"], "frames": [record.to_dict() for record in frame_records]},
        )
        result = PipelineResult(
            cube_root=self.cube_root,
            output_dir=output,
            config=self.config,
            model=model,
            drift=drift,
            post_unmix=post_unmix,
            pre_unmix=pre_unmix,
            post_frame_metrics=tuple(post_metrics_frames),
            pre_frame_metrics=tuple(pre_metrics_frames),
            post_metrics=post_aggregate,
            pre_metrics=pre_aggregate,
            frames=tuple(frame_records),
            report=report,
            report_path=report_path,
        )
        if verify_closure:
            report["closure"] = self.verify_closure(result)
            report["run_hash"] = key_for({key: value for key, value in report.items() if key != "run_hash"})
            write_json(report_path, report)
            result = PipelineResult(
                cube_root=result.cube_root,
                output_dir=result.output_dir,
                config=result.config,
                model=result.model,
                drift=result.drift,
                post_unmix=result.post_unmix,
                pre_unmix=result.pre_unmix,
                post_frame_metrics=result.post_frame_metrics,
                pre_frame_metrics=result.pre_frame_metrics,
                post_metrics=result.post_metrics,
                pre_metrics=result.pre_metrics,
                frames=result.frames,
                report=report,
                report_path=report_path,
            )
        active_unmix = {post_unmix_key}
        if self.config.drift.compensate:
            active_unmix.add(pre_unmix_key)
        if not keep_cache:
            for stage, keys in (
                ("calibration", active_calibration),
                ("drift", {drift_key}),
                ("unmix", active_unmix),
                ("metrics", active_metrics),
            ):
                prune_stage(cache_root, stage, set())
        else:
            report["incremental"]["prune"] = [
                prune_stage(cache_root, "calibration", active_calibration),
                prune_stage(cache_root, "drift", {drift_key}),
                prune_stage(cache_root, "unmix", active_unmix),
                prune_stage(cache_root, "metrics", active_metrics),
            ]
            report["run_hash"] = key_for({key: value for key, value in report.items() if key != "run_hash"})
            write_json(report_path, report)
        return result

    def _compensation_evaluation(self, pre: Dict[str, Any], post: Dict[str, Any]) -> Dict[str, Any]:
        pre_value = float(pre["overall_absolute_error_mean"])
        post_value = float(post["overall_absolute_error_mean"])
        per_fluorophore = []
        for before, after in zip(pre["per_fluorophore"], post["per_fluorophore"]):
            before_value = before["absolute_error_mean"]
            after_value = after["absolute_error_mean"]
            per_fluorophore.append(
                {
                    "fluorophore": before["fluorophore"],
                    "name": before["name"],
                    "pre_absolute_error_mean": before_value,
                    "post_absolute_error_mean": after_value,
                    "absolute_error_delta": (
                        float(after_value - before_value)
                        if before_value is not None and after_value is not None
                        else None
                    ),
                }
            )
        return {
            "enabled": self.config.drift.compensate,
            "pre_absolute_error_mean": pre_value,
            "post_absolute_error_mean": post_value,
            "absolute_error_reduction": float(pre_value - post_value),
            "improved": bool(self.config.drift.compensate and post_value < pre_value),
            "declared_metric": "mean absolute concentration error over truth-mask pixels across all spectral bands",
            "per_fluorophore": per_fluorophore,
        }

    def verify_closure(self, result: PipelineResult) -> Dict[str, Any]:
        """Recompute all stages without reuse and locate the first divergence."""
        manifest = load_manifest(self.cube_root, verify_hashes=False)
        cube_config = CubeConfig.from_dict(manifest["cube"])
        height, width = cube_config.height, cube_config.width
        truths = [load_truth(self.cube_root, frame) for frame in range(cube_config.frames)]
        fresh_calibrated: List[np.ndarray] = []
        fresh_invalid: List[np.ndarray] = []
        for frame, record in enumerate(manifest["frames"]):
            raw = load_frame_array(self.cube_root, record, "raw", (height, width))
            dark = load_frame_array(self.cube_root, record, "dark", (height, width))
            flat = load_frame_array(self.cube_root, record, "flat", (height, width))
            fresh = calibrate_frame(raw, dark, flat, self.config.calibration, frame_index=frame)
            cached = load_calibration(result.output_dir / "cache", result.frames[frame].calibration_version)
            if cached is None:
                raise WorkbenchError(ErrorCode.E_CLOSURE_MISMATCH, "cached calibration is missing during closure verification")
            divergence = _single_frame_divergence(
                cached[0].corrected,
                fresh.corrected,
                rtol=self.config.closure_rtol,
                atol=self.config.closure_atol,
            )
            if divergence is not None:
                raise WorkbenchError(
                    ErrorCode.E_CLOSURE_MISMATCH,
                    "full rebuild diverged at calibration stage",
                    {"stage": "calibration", "frame": frame, "divergence": divergence},
                )
            if not np.array_equal(cached[0].invalid_mask, fresh.invalid_mask):
                index = int(np.flatnonzero(cached[0].invalid_mask.ravel() != fresh.invalid_mask.ravel())[0])
                pixel = [int(v) for v in np.unravel_index(index, fresh.invalid_mask.shape)]
                raise WorkbenchError(
                    ErrorCode.E_CLOSURE_MISMATCH,
                    "full rebuild diverged at calibration validity mask",
                    {"stage": "calibration", "frame": frame, "pixel": pixel},
                )
            fresh_calibrated.append(fresh.corrected)
            fresh_invalid.append(fresh.invalid_mask)
        fresh_stack = np.stack(fresh_calibrated, axis=0)
        fresh_invalid_stack = np.stack(fresh_invalid, axis=0)
        fresh_drift = estimate_drift_sequence(
            fresh_stack,
            fresh_invalid_stack,
            manifest["reference_centers"],
            [truth["drift_xy"] for truth in truths],
            self.config.drift,
        )
        for frame, (cached, fresh) in enumerate(zip(result.drift.frames, fresh_drift.frames)):
            shift_delta = float(
                np.hypot(cached.estimated_dx - fresh.estimated_dx, cached.estimated_dy - fresh.estimated_dy)
            )
            if shift_delta > self.config.closure_atol:
                raise WorkbenchError(
                    ErrorCode.E_CLOSURE_MISMATCH,
                    "full rebuild diverged at drift stage",
                    {"stage": "drift", "frame": frame, "shift_delta_px": shift_delta},
                )
        if self.config.drift.compensate:
            shifted_bands = []
            shifted_invalid = []
            for frame in range(cube_config.frames):
                image, mask = shift_stack_to_reference(
                    fresh_stack[frame],
                    fresh_invalid_stack[frame],
                    fresh_drift.frames[frame].estimated_dx,
                    fresh_drift.frames[frame].estimated_dy,
                )
                shifted_bands.append(image)
                shifted_invalid.append(mask)
            final_stack = np.stack(shifted_bands, axis=0)
            final_invalid = np.stack(shifted_invalid, axis=0)
        else:
            final_stack = fresh_stack
            final_invalid = fresh_invalid_stack
        quantization_upper_bound = max(
            float(
                read_json(
                    stage_dir(
                        result.output_dir / "cache",
                        "calibration",
                        result.frames[frame].calibration_version,
                    )
                    / "record.json"
                )["quantization_upper_bound_dn"]
            )
            for frame in range(cube_config.frames)
        )
        fresh_unmix = unmix_frame(
            final_stack,
            np.any(final_invalid != 0, axis=0).astype(np.uint8),
            result.model,
            self.config.unmix,
            quantization_upper_bound=quantization_upper_bound,
            frame_index=-1,
        )
        cached_unmix = load_unmix(result.output_dir / "cache", result.frames[0].unmix_version)
        if cached_unmix is None:
            raise WorkbenchError(ErrorCode.E_CLOSURE_MISMATCH, "cached unmix output is missing during closure verification")
        for field in ("concentrations", "residual"):
            divergence = _single_frame_divergence(
                getattr(cached_unmix[0], field),
                getattr(fresh_unmix, field),
                rtol=self.config.closure_rtol,
                atol=self.config.closure_atol,
            )
            if divergence is not None:
                raise WorkbenchError(
                    ErrorCode.E_CLOSURE_MISMATCH,
                    f"full rebuild diverged at unmix {field}",
                    {"stage": "unmix", "array": field, "frame": result.frames[0].frame, "divergence": divergence},
                )
        if not np.array_equal(cached_unmix[0].active_subset, fresh_unmix.active_subset):
            index = int(np.flatnonzero(cached_unmix[0].active_subset.ravel() != fresh_unmix.active_subset.ravel())[0])
            pixel = [int(v) for v in np.unravel_index(index, fresh_unmix.active_subset.shape)]
            raise WorkbenchError(
                ErrorCode.E_CLOSURE_MISMATCH,
                "full rebuild diverged at NNLS active subset",
                {"stage": "unmix", "array": "active_subset", "frame": result.frames[0].frame, "pixel": pixel},
            )
        return {
            "verified": True,
            "mode": "full_rebuild_without_cache",
            "rtol": self.config.closure_rtol,
            "atol": self.config.closure_atol,
            "bands_compared": cube_config.frames,
            "first_divergence": None,
            "scope_note": "the NNLS solve is cube-level because every pixel spectrum spans all spectral bands",
        }
