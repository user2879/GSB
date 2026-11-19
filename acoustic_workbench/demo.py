"""Repeatable demonstration sequence and error-code coverage."""

from __future__ import annotations

import filecmp
import io
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .analysis import estimate_delays
from .audio import decode_wav_24
from .calibration import verify_calibration
from .deconvolution import estimate_ir_set
from .errors import ErrorCode, WorkbenchError
from .gui import render_snapshot_png
from .hashing import canonical_json, content_hash, write_bytes_atomic, write_text_atomic
from .models import MeasurementRun, WorkbenchConfig, default_config, validate_config
from .pipeline import PipelineRun, WorkbenchPipeline
from .report import build_report, render_markdown
from .simulation import manifest_dict, measurement_layout, validate_sync_marks
from .snapshot import SnapshotStore


def _write_json(path: Path, value: Any) -> None:
    write_text_atomic(path, canonical_json(value, pretty=True))


def demo_config() -> WorkbenchConfig:
    base = default_config()
    return replace(
        base,
        measurement=replace(
            base.measurement,
            sweep_duration_s=1.5,
            repeats=2,
        ),
    )


def _measurement_bundle(output: Path, run: PipelineRun) -> None:
    root = output / "measurement"
    root.mkdir(parents=True, exist_ok=True)
    write_bytes_atomic(root / "measurement.wav", run.measurement.wav_bytes)
    _write_json(
        root / "channel_config.json",
        [
            {
                "index": index,
                "name": channel.name,
                "delay_samples": channel.delay_samples,
                "gain": channel.gain,
                "band_low_hz": channel.band_low_hz,
                "band_high_hz": channel.band_high_hz,
                "noise_seed": channel.noise_seed,
                "noise_rms": channel.noise_rms,
            }
            for index, channel in enumerate(run.measurement.config.channels)
        ],
    )
    _write_json(
        root / "sync_markers.json",
        {
            "truth": [dict(item) for item in run.measurement.sync_markers],
            "detected": [dict(item) for item in run.detected_sync],
        },
    )
    _write_json(root / "manifest.json", manifest_dict(run.measurement, measurement_layout(run.measurement.config)))
    source_buffer = io.BytesIO()
    np.save(source_buffer, np.asarray(run.measurement.source, dtype="<f8"), allow_pickle=False)
    write_bytes_atomic(root / "source_reference.npy", source_buffer.getvalue())
    _write_json(
        root / "deconvolution_recipe.json",
        {
            "config": run.config.deconvolution,
            "reference_channel": run.config.analysis.reference_channel,
            "sweep_starts": run.measurement.sweep_starts,
            "channels": [
                {
                    "channel_index": channel.channel_index,
                    "nfft": channel.nfft,
                    "window": channel.window,
                    "truncation_samples": channel.truncation_samples,
                    "origin_shift_samples": channel.origin_shift_samples,
                    "window_starts": channel.window_starts,
                    "normalization_factor": channel.normalization_factor,
                    "arrival_sample": channel.arrival_sample,
                    "peak_level_db": channel.peak_level_db,
                }
                for channel in run.deconvolution.channels
            ],
        },
    )
    filter_stack = np.stack(
        [np.asarray(item.coefficients, dtype="<f8") for item in run.calibration.filters],
        axis=0,
    )
    filter_buffer = io.BytesIO()
    np.save(filter_buffer, filter_stack, allow_pickle=False)
    write_bytes_atomic(root / "calibration_filters.npy", filter_buffer.getvalue())
    _write_json(
        root / "calibration_filter_meta.json",
        {
            "filter_length": run.calibration.filter_length,
            "window": run.calibration.window,
            "channels": [item.channel for item in run.calibration.filters],
            "delay_samples": [item.delay_samples for item in run.calibration.filters],
            "calibration_hash": run.calibration.result_hash,
        },
    )


def _write_step(output: Path, index: int, name: str, run: PipelineRun) -> dict[str, Any]:
    step_dir = output / "runs" / f"{index:02d}_{name}"
    step_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(run.snapshot)
    _write_json(step_dir / "snapshot.json", run.snapshot.to_dict())
    _write_json(step_dir / "report.json", report)
    write_text_atomic(step_dir / "report.md", render_markdown(report))
    _write_json(
        step_dir / "reuse.json",
        {
            "revision": run.snapshot.revision,
            "version_stamp": run.snapshot.version_stamp,
            "snapshot_hash": run.snapshot.content_hash,
            "stages": [stage.to_dict() for stage in run.stages],
        },
    )
    return {
        "index": index,
        "name": name,
        "directory": f"runs/{index:02d}_{name}",
        "snapshot_hash": run.snapshot.content_hash,
        "version_stamp": run.snapshot.version_stamp,
        "report_hash": report["report_hash"],
        "stage_hashes": {stage.stage: stage.content_hash for stage in run.stages},
        "reuse": {stage.stage: stage.compute_source for stage in run.stages},
        "acceptance": dict(run.snapshot.data["acceptance"]),
    }


def _capture_error(events: list[dict[str, Any]], label: str, action) -> None:
    try:
        action()
    except WorkbenchError as exc:
        events.append({"scenario": label, **exc.to_dict()})
    else:
        events.append(
            {
                "scenario": label,
                "code": "UNEXPECTED_SUCCESS",
                "stage": "error-chain",
                "message": "expected a business error",
                "details": {},
            }
        )


def error_chain(
    run: PipelineRun,
    snapshot_path: Path,
    *,
    config: WorkbenchConfig,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    _capture_error(
        events,
        "audio-format",
        lambda: decode_wav_24(run.measurement.wav_bytes, expected_sample_rate=44_100),
    )

    def missing_sync() -> None:
        broken = replace(run.measurement, recorded=np.zeros_like(run.measurement.recorded))
        validate_sync_marks(broken)

    _capture_error(events, "sync-mark-missing", missing_sync)
    _capture_error(
        events,
        "delay-bias",
        lambda: estimate_delays(
            run.measurement,
            run.deconvolution,
            replace(config, analysis=replace(config.analysis, delay_tolerance_samples=0.10)),
            raise_on_bias=True,
        ),
    )

    def calibration_mismatch() -> None:
        tight = replace(
            config,
            calibration=replace(config.calibration, residual_tolerance_factor=0.0),
        )
        verify_calibration(
            run.measurement,
            run.calibrated_measurement,
            run.deconvolution,
            run.mismatch,
            run.delays,
            run.calibration,
            tight,
            raise_on_mismatch=True,
        )

    _capture_error(events, "calibration-mismatch", calibration_mismatch)
    _capture_error(
        events,
        "snapshot-stale",
        lambda: SnapshotStore(snapshot_path).load(expected_hash="0" * 64),
    )
    _capture_error(
        events,
        "band-invalid",
        lambda: validate_config(
            replace(config, analysis=replace(config.analysis, band_low_hz=1_000.0, band_high_hz=100.0))
        ),
    )
    _capture_error(
        events,
        "filter-length",
        lambda: validate_config(
            replace(config, calibration=replace(config.calibration, filter_length=32))
        ),
    )
    return events


def _clear_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def run_demo_sequence(output: str | Path) -> dict[str, Any]:
    output_path = Path(output)
    _clear_directory(output_path)
    pipeline = WorkbenchPipeline(output_path / "state" / "cache", output_path / "state" / "current_snapshot.json")
    config = demo_config()
    steps: list[dict[str, Any]] = []

    first = pipeline.run(config)
    steps.append(_write_step(output_path, 1, "initial", first))

    config_regularization = config.with_deconvolution(regularization=2.0e-6)
    second = pipeline.run(config_regularization)
    steps.append(_write_step(output_path, 2, "regularization", second))

    config_band = config_regularization.with_analysis(band_low_hz=300.0, band_high_hz=8_000.0)
    third = pipeline.run(config_band)
    steps.append(_write_step(output_path, 3, "analysis_band", third))

    config_filter = config_band.with_calibration(filter_length=513)
    fourth = pipeline.run(config_filter)
    steps.append(_write_step(output_path, 4, "filter_length", fourth))

    config_sweep = config_filter.with_measurement(sweep_duration_s=1.2)
    fifth = pipeline.run(config_sweep)
    steps.append(_write_step(output_path, 5, "sweep_length", fifth))
    _measurement_bundle(output_path, fifth)

    error_events = error_chain(fifth, output_path / "state" / "current_snapshot.json", config=config_sweep)
    _write_json(output_path / "error_chain.json", {"events": error_events})

    filter_stack = np.stack(
        [np.asarray(item.coefficients, dtype="<f8") for item in fifth.calibration.filters], axis=0
    )
    filter_buffer = io.BytesIO()
    np.save(filter_buffer, filter_stack, allow_pickle=False)
    write_bytes_atomic(output_path / "final_calibration_filters.npy", filter_buffer.getvalue())
    render_snapshot_png(fifth.snapshot, output_path / "diagnostic_view.png", width=1440, height=900)

    manifest = {
        "schema": "acoustic-workbench.demo/1.0",
        "steps": steps,
        "error_chain": error_events,
        "final_snapshot_hash": fifth.snapshot.content_hash,
        "final_version_stamp": fifth.snapshot.version_stamp,
        "final_step_hashes": {stage.stage: stage.content_hash for stage in fifth.stages},
        "final_acceptance": dict(fifth.snapshot.data["acceptance"]),
        "output_files": sorted(
            str(path.relative_to(output_path)).replace("\\", "/")
            for path in output_path.rglob("*")
            if path.is_file()
        ),
    }
    _write_json(output_path / "demo_manifest.json", manifest)
    return manifest


def compare_directories(first: Path, second: Path, *, ignore: set[str] | None = None) -> dict[str, Any]:
    ignored = ignore or set()
    first_files = {
        str(path.relative_to(first)).replace("\\", "/")
        for path in first.rglob("*")
        if path.is_file() and str(path.relative_to(first)).replace("\\", "/") not in ignored
    }
    second_files = {
        str(path.relative_to(second)).replace("\\", "/")
        for path in second.rglob("*")
        if path.is_file() and str(path.relative_to(second)).replace("\\", "/") not in ignored
    }
    only_first = sorted(first_files - second_files)
    only_second = sorted(second_files - first_files)
    mismatches = sorted(
        name
        for name in (first_files & second_files)
        if not filecmp.cmp(first / name, second / name, shallow=False)
    )
    return {
        "identical": not only_first and not only_second and not mismatches,
        "only_first": only_first,
        "only_second": only_second,
        "content_mismatches": mismatches,
        "compared_files": len(first_files & second_files),
    }


def run_demo(output: str | Path, *, verify_determinism: bool = False) -> dict[str, Any]:
    output_path = Path(output)
    manifest = run_demo_sequence(output_path)
    if verify_determinism:
        with tempfile.TemporaryDirectory(prefix="acoustic-workbench-demo-") as temporary:
            second = Path(temporary) / "demo"
            run_demo_sequence(second)
            comparison = compare_directories(
                output_path,
                second,
                ignore={"determinism_check.json"},
            )
        manifest["determinism_check"] = comparison
        _write_json(output_path / "determinism_check.json", comparison)
        _write_json(output_path / "demo_manifest.json", manifest)
    return manifest
