"""Incremental orchestration with content-addressed stage caching."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .analysis import DelayAnalysis, MismatchAnalysis, estimate_delays, analyze_mismatch
from .calibration import (
    CalibrationDesign,
    CalibrationVerification,
    design_calibration,
    simulate_calibrated_measurement,
    verify_calibration,
)
from .deconvolution import DeconvolutionResult, estimate_ir_set
from .hashing import content_hash
from .models import MeasurementRun, WorkbenchConfig, as_serializable, validate_config
from .simulation import generate_measurement, measurement_layout, validate_sync_marks
from .snapshot import ArtifactCache, Snapshot, SnapshotStore, build_snapshot


@dataclass(frozen=True)
class StageRecord:
    stage: str
    key: str
    content_hash: str
    compute_source: str
    reused: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "key": self.key,
            "content_hash": self.content_hash,
            "compute_source": self.compute_source,
            "reused": self.reused,
        }


@dataclass
class PipelineRun:
    config: WorkbenchConfig
    snapshot: Snapshot
    measurement: MeasurementRun
    detected_sync: tuple[dict[str, Any], ...]
    deconvolution: DeconvolutionResult
    delays: DelayAnalysis
    mismatch: MismatchAnalysis
    calibration: CalibrationDesign
    calibrated_measurement: MeasurementRun
    verification: CalibrationVerification
    stages: tuple[StageRecord, ...]


def _round(value: float, digits: int = 12) -> float:
    value = float(value)
    if value == -0.0:
        return 0.0
    return round(value, digits)


def _floats(values, digits: int = 12) -> list[float]:
    return [_round(float(value), digits) for value in values]


def _ints(values) -> list[int]:
    return [int(value) for value in values]


def _delay_payload(delays: DelayAnalysis) -> dict[str, Any]:
    return {
        "reference_channel": delays.reference_channel,
        "formula": delays.formula,
        "errors": [dict(event) for event in delays.errors],
        "channels": [
            {
                "channel_index": item.channel_index,
                "channel": item.channel,
                "truth_relative_samples": _round(item.truth_relative_samples),
                "estimated_relative_samples": _round(item.estimated_relative_samples),
                "bias_samples": _round(item.bias_samples),
                "error_bound_samples": _round(item.error_bound_samples),
                "snr_db": _round(item.snr_db),
                "crb_samples": _round(item.crb_samples),
                "curvature_relative": _round(item.curvature_relative),
                "interpolation_model_error_samples": _round(item.interpolation_model_error_samples),
                "correlation_band_hz": _floats(item.correlation_band_hz),
                "status": item.status,
                "error_code": item.error_code,
                "details": dict(item.details),
            }
            for item in delays.channels
        ],
    }


def _mismatch_payload(mismatch: MismatchAnalysis) -> dict[str, Any]:
    channels: list[dict[str, Any]] = []
    for channel in mismatch.channels:
        channels.append(
            {
                "channel_index": channel.channel_index,
                "channel": channel.channel,
                "amplitude_ratio_db": _floats(channel.amplitude_ratio_db),
                "phase_difference_deg": _floats(channel.phase_difference_deg),
                "amplitude_error_halfwidth_db": _floats(channel.amplitude_error_halfwidth_db),
                "phase_error_halfwidth_deg": _floats(channel.phase_error_halfwidth_deg),
                "noise_contribution_db": _floats(channel.noise_contribution_db),
                "leakage_contribution_db": _floats(channel.leakage_contribution_db),
                "band_limit_contribution_db": _floats(channel.band_limit_contribution_db),
                "weighted_gain_db": _round(channel.weighted_gain_db),
                "truth_gain_db": _round(channel.truth_gain_db),
                "gain_bias_db": _round(channel.gain_bias_db),
                "interval_low_db": _round(channel.interval_low_db),
                "interval_high_db": _round(channel.interval_high_db),
                "noise_bound_db": _round(channel.noise_bound_db),
                "leakage_bound_db": _round(channel.leakage_bound_db),
                "band_limit_bound_db": _round(channel.band_limit_bound_db),
                "phase_interval_halfwidth_deg": _round(channel.phase_interval_halfwidth_deg),
                "truth_within_interval": bool(channel.truth_within_interval),
                "details": dict(channel.details),
                "subbands": [
                    {
                        "index": item.index,
                        "low_hz": _round(item.low_hz),
                        "high_hz": _round(item.high_hz),
                        "weight": _round(item.weight),
                        "estimated_gain_db": _round(item.estimated_gain_db),
                        "truth_gain_db": _round(item.truth_gain_db),
                        "bias_db": _round(item.bias_db),
                        "noise_contribution_db": _round(item.noise_contribution_db),
                        "leakage_contribution_db": _round(item.leakage_contribution_db),
                        "band_limit_contribution_db": _round(item.band_limit_contribution_db),
                        "interval_low_db": _round(item.interval_low_db),
                        "interval_high_db": _round(item.interval_high_db),
                        "phase_interval_halfwidth_deg": _round(item.phase_interval_halfwidth_deg),
                    }
                    for item in channel.subbands
                ],
            }
        )
    return {
        "reference_channel": mismatch.reference_channel,
        "band_low_hz": _round(mismatch.band_low_hz),
        "band_high_hz": _round(mismatch.band_high_hz),
        "frequencies_hz": _floats(mismatch.frequencies_hz),
        "point_weights": _floats(mismatch.point_weights),
        "error_model": dict(mismatch.error_model),
        "channels": channels,
    }


def _calibration_payload(design: CalibrationDesign) -> dict[str, Any]:
    return {
        "filter_length": design.filter_length,
        "window": design.window,
        "frequencies_hz": _floats(design.frequencies_hz),
        "filters": [
            {
                "channel_index": item.channel_index,
                "channel": item.channel,
                "delay_samples": item.delay_samples,
                "coefficients": _floats(item.coefficients),
                "response_magnitude_db": _floats(item.response_magnitude_db),
                "response_phase_deg": _floats(item.response_phase_deg),
                "maximum_abs_correction_db": _round(item.maximum_abs_correction_db),
                "design_status": item.design_status,
                "details": dict(item.details),
            }
            for item in design.filters
        ],
    }


def _verification_payload(verification: CalibrationVerification) -> dict[str, Any]:
    return {
        "status": verification.status,
        "remeasured_hash": verification.remeasured_hash,
        "errors": [dict(event) for event in verification.errors],
        "channels": [
            {
                "channel_index": item.channel_index,
                "channel": item.channel,
                "residual_gain_db": _round(item.residual_gain_db),
                "residual_amplitude_db": _floats(item.residual_amplitude_db),
                "residual_phase_deg": _floats(item.residual_phase_deg),
                "allowed_amplitude_halfwidth_db": _floats(item.allowed_amplitude_halfwidth_db),
                "allowed_phase_halfwidth_deg": _floats(item.allowed_phase_halfwidth_deg),
                "max_amplitude_excess_db": _round(item.max_amplitude_excess_db),
                "max_phase_excess_deg": _round(item.max_phase_excess_deg),
                "compatible": bool(item.compatible),
                "violation_bands": [dict(event) for event in item.violation_bands],
            }
            for item in verification.channels
        ],
    }


def _frequency_view(mismatch: MismatchAnalysis, verification: CalibrationVerification) -> dict[str, Any]:
    residual_by_channel = {item.channel_index: item for item in verification.channels}
    channels: list[dict[str, Any]] = []
    for channel, residual in zip(mismatch.channels, verification.channels):
        amplitude = np.asarray(channel.amplitude_ratio_db, dtype=np.float64)
        phase = np.asarray(channel.phase_difference_deg, dtype=np.float64)
        amp_half = np.asarray(channel.amplitude_error_halfwidth_db, dtype=np.float64)
        phase_half = np.asarray(channel.phase_error_halfwidth_deg, dtype=np.float64)
        channels.append(
            {
                "channel_index": channel.channel_index,
                "channel": channel.channel,
                "amplitude_ratio_db": _floats(amplitude),
                "amplitude_low_db": _floats(amplitude - amp_half),
                "amplitude_high_db": _floats(amplitude + amp_half),
                "phase_difference_deg": _floats(phase),
                "phase_low_deg": _floats(phase - phase_half),
                "phase_high_deg": _floats(phase + phase_half),
                "residual_amplitude_db": _floats(residual.residual_amplitude_db),
                "residual_phase_deg": _floats(residual.residual_phase_deg),
                "residual_amplitude_allowed_db": _floats(residual.allowed_amplitude_halfwidth_db),
                "residual_phase_allowed_deg": _floats(residual.allowed_phase_halfwidth_deg),
            }
        )
    return {
        "frequencies_hz": _floats(mismatch.frequencies_hz),
        "point_weights": _floats(mismatch.point_weights),
        "channels": channels,
        "snapshot_semantics": "render-only numeric payload; no DSP recomputation in the GUI",
    }


def _time_view(run: MeasurementRun, deconvolution: DeconvolutionResult) -> dict[str, Any]:
    layout = measurement_layout(run.config)
    times = np.asarray(deconvolution.channels[0].time_axis_samples, dtype=np.int64)
    view_start = int(min(times[0], -layout.n_guard_pre - 16))
    view_end = int(times[-1] + 1)
    view_times = np.arange(view_start, view_end, dtype=np.int64)
    channels: list[dict[str, Any]] = []
    for channel in deconvolution.channels:
        values = np.zeros(view_times.size, dtype=np.float64)
        source_indices = np.searchsorted(view_times, channel.time_axis_samples)
        valid = (source_indices >= 0) & (source_indices < view_times.size)
        values[source_indices[valid]] = channel.ir[valid]
        sync_relative: list[float] = []
        for marker in run.sync_markers:
            truth_row = marker["channels"][channel.channel_index]
            sync_relative.append(
                _round(float(truth_row["truth_sample_fractional"]) - float(marker["sweep_start_sample"]))
            )
        channels.append(
            {
                "channel_index": channel.channel_index,
                "channel": run.config.channels[channel.channel_index].name,
                "ir": _floats(values),
                "peak_relative_sample": _round(channel.arrival_sample),
                "peak_level_db": _round(channel.peak_level_db),
                "sync_marker_relative_samples": sync_relative,
                "sync_marker_relative_ms": [_round(value * 1000.0 / run.config.sample_rate) for value in sync_relative],
            }
        )
    return {
        "sample_axis": _ints(view_times),
        "time_ms": _floats(view_times * 1000.0 / run.config.sample_rate),
        "guard_pre_samples": layout.n_guard_pre,
        "view_note": "zero-extended around the deconvolved IR so sync-marker truth is visible on the time axis",
        "channels": channels,
    }


class WorkbenchPipeline:
    """Pipeline with content-addressed stage reuse and immutable snapshots."""

    def __init__(self, cache_dir: str | Path, snapshot_path: str | Path) -> None:
        self.cache = ArtifactCache(cache_dir)
        self.snapshot_store = SnapshotStore(snapshot_path)
        if self.snapshot_store.exists():
            current = self.snapshot_store.load()
            self._last_snapshot_hash: str | None = current.content_hash
            self._revision = current.revision
        else:
            self._last_snapshot_hash = None
            self._revision = 0
        self._stage_records: dict[str, StageRecord] = {}

    def _stage(self, stage: str, key: str, compute: Callable[[], Any]) -> tuple[Any, StageRecord]:
        payload, source, digest = self.cache.get_or_compute(stage, key, compute)
        record = StageRecord(
            stage=stage,
            key=key,
            content_hash=digest,
            compute_source=source,
            reused=source != "computed",
        )
        self._stage_records[stage] = record
        return payload, record

    def run(self, config: WorkbenchConfig) -> PipelineRun:
        validate_config(config)
        self._stage_records = {}
        measurement_key = content_hash({"schema": "measurement/1", "config": config.measurement})
        measurement, measurement_stage = self._stage(
            "measurement",
            measurement_key,
            lambda: generate_measurement(config.measurement),
        )
        sync_key = content_hash(
            {
                "schema": "sync/1",
                "recorded_hash": measurement.recorded_hash,
                "markers": measurement.sync_markers,
            }
        )
        detected_sync, sync_stage = self._stage(
            "sync",
            sync_key,
            lambda: validate_sync_marks(measurement),
        )

        deconv_key = content_hash(
            {
                "schema": "deconvolution/1",
                "measurement": [measurement.source_hash, measurement.recorded_hash],
                "deconvolution": config.deconvolution,
                "reference_channel": config.analysis.reference_channel,
            }
        )
        deconvolution, deconv_stage = self._stage(
            "deconvolution",
            deconv_key,
            lambda: estimate_ir_set(measurement, config),
        )

        delay_key = content_hash(
            {
                "schema": "delay/1",
                "deconvolution_hash": deconvolution.result_hash,
                "measurement_hashes": [measurement.recorded_hash, measurement.source_hash],
                "noise_sigma": measurement.noise_sigma_estimate,
                "parameters": {
                    "reference_channel": config.analysis.reference_channel,
                    "coverage_sigma": config.analysis.coverage_sigma,
                    "delay_tolerance_samples": config.analysis.delay_tolerance_samples,
                    "delay_bias_tolerance_samples": config.analysis.delay_bias_tolerance_samples,
                },
            }
        )
        delays, delay_stage = self._stage(
            "delay",
            delay_key,
            lambda: estimate_delays(measurement, deconvolution, config, raise_on_bias=True),
        )

        mismatch_key = content_hash(
            {
                "schema": "mismatch/1",
                "deconvolution_hash": deconvolution.result_hash,
                "delay_hash": delays.result_hash,
                "analysis": config.analysis,
            }
        )
        mismatch, mismatch_stage = self._stage(
            "mismatch",
            mismatch_key,
            lambda: analyze_mismatch(measurement, deconvolution, delays, config),
        )

        calibration_key = content_hash(
            {
                "schema": "calibration/1",
                "deconvolution_hash": deconvolution.result_hash,
                "mismatch_hash": mismatch.result_hash,
                "analysis_band": [config.analysis.band_low_hz, config.analysis.band_high_hz],
                "reference_channel": config.analysis.reference_channel,
                "calibration": config.calibration,
            }
        )
        calibration, calibration_stage = self._stage(
            "calibration",
            calibration_key,
            lambda: design_calibration(measurement, deconvolution, mismatch, config),
        )

        remeasure_key = content_hash(
            {
                "schema": "remeasurement/1",
                "measurement_hashes": [measurement.source_hash, measurement.recorded_hash],
                "measurement_config": config.measurement,
                "calibration_hash": calibration.result_hash,
            }
        )
        calibrated_measurement, remeasure_stage = self._stage(
            "remeasurement",
            remeasure_key,
            lambda: simulate_calibrated_measurement(measurement, calibration),
        )

        verification_key = content_hash(
            {
                "schema": "verification/1",
                "calibration_hash": calibration.result_hash,
                "remeasured_hash": calibrated_measurement.recorded_hash,
                "mismatch_hash": mismatch.result_hash,
                "analysis": config.analysis,
                "calibration": config.calibration,
                "deconvolution": config.deconvolution,
            }
        )
        verification, verification_stage = self._stage(
            "verification",
            verification_key,
            lambda: verify_calibration(
                measurement,
                calibrated_measurement,
                deconvolution,
                mismatch,
                delays,
                calibration,
                config,
                raise_on_mismatch=False,
            ),
        )

        stages = tuple(self._stage_records[name] for name in (
            "measurement",
            "sync",
            "deconvolution",
            "delay",
            "mismatch",
            "calibration",
            "remeasurement",
            "verification",
        ))
        step_hashes = {stage.stage: stage.content_hash for stage in stages}
        reuse = {stage.stage: stage.to_dict() for stage in stages}
        layout = measurement_layout(measurement.config)
        errors = [dict(event) for event in delays.errors] + [dict(event) for event in verification.errors]
        snapshot_payload = {
            "pipeline_schema": "acoustic-workbench.pipeline/1.0",
            "config": as_serializable(config),
            "config_hash": content_hash(config),
            "input_hashes": {
                "source": measurement.source_hash,
                "recorded_samples": measurement.recorded_hash,
                "wav": measurement.wav_hash,
                "calibrated_remeasurement_wav": calibrated_measurement.wav_hash,
            },
            "noise_sigma_estimate": _floats(measurement.noise_sigma_estimate),
            "audio": {
                "format": "PCM",
                "sample_rate": measurement.config.sample_rate,
                "bit_depth": measurement.config.bit_depth,
                "channels": len(measurement.config.channels),
                "frames": layout.n_frames,
                "duration_s": _round(layout.n_frames / measurement.config.sample_rate),
            },
            "step_hashes": step_hashes,
            "reuse": reuse,
            "sync": {
                "truth": [dict(item) for item in measurement.sync_markers],
                "detected": [dict(item) for item in detected_sync],
                "status": "OK",
            },
            "delay": _delay_payload(delays),
            "deconvolution": {
                "nfft": int(deconvolution.nfft),
                "regularization": _round(config.deconvolution.regularization),
                "effective_band_hz": [
                    _round(config.deconvolution.effective_low_hz),
                    _round(config.deconvolution.effective_high_hz),
                ],
                "window": config.deconvolution.window,
                "channels": [
                    {
                        "channel_index": channel.channel_index,
                        "channel": run_name,
                        "nfft": int(channel.nfft),
                        "window": channel.window,
                        "truncation_samples": int(channel.truncation_samples),
                        "origin_shift_samples": int(channel.origin_shift_samples),
                        "segment_offsets": _ints(channel.segment_offsets),
                        "window_starts": _ints(channel.window_starts),
                        "peak_index": int(channel.peak_index),
                        "peak_sample": int(channel.peak_sample),
                        "arrival_sample": _round(channel.arrival_sample),
                        "peak_level_db": _round(channel.peak_level_db),
                        "snr_db": _round(channel.snr_db),
                        "normalization_factor": _round(channel.normalization_factor),
                        "truncation_energy_fraction": _round(channel.truncation_energy_fraction),
                        "window_leakage_db": _round(channel.window_leakage_db),
                        "ir_hash": channel.ir_hash,
                    }
                    for channel, run_name in zip(
                        deconvolution.channels,
                        [item.name for item in measurement.config.channels],
                    )
                ],
            },
            "mismatch": _mismatch_payload(mismatch),
            "calibration": _calibration_payload(calibration),
            "verification": _verification_payload(verification),
            "frequency_view": _frequency_view(mismatch, verification),
            "time_view": _time_view(measurement, deconvolution),
            "errors": errors,
            "acceptance": {
                "sync_ok": True,
                "delay_ok": not delays.errors,
                "calibration_ok": verification.status == "OK",
                "mismatch_truth_within_interval": all(item.truth_within_interval for item in mismatch.channels),
            },
        }
        snapshot = build_snapshot(snapshot_payload, revision=self._revision + 1)
        self.snapshot_store.commit(snapshot, expected_previous_hash=self._last_snapshot_hash)
        self._last_snapshot_hash = snapshot.content_hash
        self._revision = snapshot.revision
        return PipelineRun(
            config=config,
            snapshot=snapshot,
            measurement=measurement,
            detected_sync=detected_sync,
            deconvolution=deconvolution,
            delays=delays,
            mismatch=mismatch,
            calibration=calibration,
            calibrated_measurement=calibrated_measurement,
            verification=verification,
            stages=stages,
        )
