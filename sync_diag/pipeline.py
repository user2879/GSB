from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .config import config_content_hash, load_config
from .detect import detect_and_pair, pair_pulses
from .errors import SyncError, require
from .estimate import drift_estimate_report, estimate_drift
from .hashing import sha256_file, write_json
from .manifest import load_manifest
from .resample import compensate_wav, detect_aligned_pulse_positions
from .snapshot import publish_snapshot, render_diagnostic_view
from .strategy import choose_strategy
from .verify import verify_alignment


def _comparison_block_frames(config: dict[str, Any], current_block_frames: int) -> int:
    rate = int(config["sample_rate_hz"])
    candidates = [int(round(float(value) * rate)) for value in config["resample"]["comparison_block_s"]]
    different = [value for value in candidates if value != current_block_frames]
    return different[0] if different else max(64, current_block_frames // 2)


def run_analysis(
    input_dir: Path,
    output_dir: Path,
    config_path: Path | None = None,
    snapshot_version: int = 1,
    consistency: bool = True,
    keep_aligned: bool | None = None,
) -> dict[str, Any]:
    if config_path is None:
        config_path = input_dir / "config.resolved.json"
    config = load_config(config_path)
    manifest_path = input_dir / "manifest.json"
    manifest, manifest_file_sha = load_manifest(manifest_path, config)
    wav_path = input_dir / "capture.wav"
    require(wav_path.exists(), "E_MANIFEST_HASH", "capture WAV is missing", path=str(wav_path))
    actual_wav_sha = sha256_file(wav_path)
    require(
        manifest.get("wav_sha256") == actual_wav_sha,
        "E_MANIFEST_HASH",
        "capture WAV hash does not match manifest",
        expected=manifest.get("wav_sha256"),
        actual=actual_wav_sha,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    detection_pairs = detect_and_pair(wav_path, manifest, config, frames=int(manifest["frames"]))
    decision = choose_strategy(
        float(manifest.get("duration_s", manifest["analysis_frames"] / config["sample_rate_hz"])),
        len(manifest["reference_pulse_samples"]),
        config,
    )
    estimate = estimate_drift(detection_pairs, manifest, config, decision)
    duration_s = float(manifest.get("duration_s", int(manifest["analysis_frames"]) / int(config["sample_rate_hz"])))
    if keep_aligned is None:
        threshold = float(config.get("pipeline", {}).get("auto_drop_aligned_after_s", 120.0))
        keep_aligned = duration_s <= threshold
    aligned_path = output_dir / "aligned.wav" if keep_aligned else None
    compensation = compensate_wav(
        wav_path,
        aligned_path,
        manifest,
        config,
        estimate["curves_by_channel"],
    )
    if aligned_path is not None:
        aligned_pairs = detect_and_pair(aligned_path, manifest, config, frames=int(manifest["analysis_frames"]))
    else:
        aligned_detected = detect_aligned_pulse_positions(wav_path, manifest, config, estimate["curves_by_channel"])
        aligned_pairs = pair_pulses(
            aligned_detected,
            manifest["reference_pulse_samples"],
            config["detector"],
            int(config["sample_rate_hz"]),
            int(config["reference_channel"]),
        )
    verification = verify_alignment(detection_pairs, aligned_pairs, manifest, estimate, config)
    comparison: dict[str, Any] | None = None
    if consistency:
        comparison_path = output_dir / "_comparison_aligned.wav" if keep_aligned else None
        comparison_block = _comparison_block_frames(config, int(compensation["block_frames"]))
        comparison_compensation = compensate_wav(
            wav_path,
            comparison_path,
            manifest,
            config,
            estimate["curves_by_channel"],
            block_frames=comparison_block,
        )
        if comparison_path is not None:
            comparison_pairs = detect_and_pair(comparison_path, manifest, config, frames=int(manifest["analysis_frames"]))
        else:
            comparison_detected = detect_aligned_pulse_positions(
                wav_path,
                manifest,
                config,
                estimate["curves_by_channel"],
            )
            comparison_pairs = pair_pulses(
                comparison_detected,
                manifest["reference_pulse_samples"],
                config["detector"],
                int(config["sample_rate_hz"]),
                int(config["reference_channel"]),
            )
        comparison_verification = verify_alignment(detection_pairs, comparison_pairs, manifest, estimate, config)
        comparison_hash = comparison_compensation["output_sha256"]
        main_hash = compensation["output_sha256"]
        residual_delta = abs(
            float(comparison_verification["summary"]["after_rmse_samples_max"])
            - float(verification["summary"]["after_rmse_samples_max"])
        )
        tolerance = float(config["resample"].get("consistency_tolerance_samples", 0.05))
        require(
            residual_delta <= tolerance,
            "E_RESAMPLE_PHASE",
            "two block lengths produced inconsistent alignment residuals",
            residual_delta_samples=residual_delta,
            tolerance_samples=tolerance,
            block_a=compensation["block_frames"],
            block_b=comparison_block,
        )
        comparison = {
            "block_frames": comparison_block,
            "aligned_sha256": comparison_hash,
            "main_aligned_sha256": main_hash,
            "byte_identical": comparison_hash == main_hash,
            "after_rmse_samples_max": comparison_verification["summary"]["after_rmse_samples_max"],
            "residual_delta_samples": residual_delta,
            "tolerance_samples": tolerance,
            "continuity": comparison_compensation["continuity"],
            "consistent": True,
        }
        if comparison_path is not None:
            comparison_path.unlink()
    aligned_sha = compensation["output_sha256"]
    if aligned_path is not None:
        require(sha256_file(aligned_path) == aligned_sha, "E_IO", "streamed aligned hash differs from file hash")
    estimate_report = drift_estimate_report(estimate, decision)
    drift_report_path = output_dir / "drift_report.json"
    decision_path = output_dir / "decision.json"
    residual_path = output_dir / "residual_report.json"
    pairing_path = output_dir / "pulse_pairing.json"
    drift_report_sha = write_json(drift_report_path, estimate_report)
    decision_sha = write_json(decision_path, decision)
    residual_sha = write_json(residual_path, verification)
    pairing_sha = write_json(pairing_path, detection_pairs)
    payload = {
        "schema_version": 1,
        "capture": {
            "sample_rate_hz": int(config["sample_rate_hz"]),
            "channel_count": int(config["channel_count"]),
            "bit_depth": int(config["bit_depth"]),
            "frames": int(manifest["frames"]),
            "analysis_frames": int(manifest["analysis_frames"]),
            "pulse_count": len(manifest["reference_pulse_samples"]),
            "manifest_sha256": manifest_file_sha,
            "wav_sha256": actual_wav_sha,
        },
        "decision": decision,
        "estimate": estimate_report,
        "resample": {**compensation, "comparison": comparison},
        "verification": verification,
    }
    snapshot_info = publish_snapshot(output_dir / "snapshots", snapshot_version, payload)
    view_path = output_dir / "diagnostic_view.svg"
    view_sha = render_diagnostic_view(output_dir / "snapshots", view_path)
    step_hashes = {
        "config_sha256": config_content_hash(config),
        "manifest_file_sha256": manifest_file_sha,
        "capture_wav_sha256": actual_wav_sha,
        "pulse_pairing_sha256": pairing_sha,
        "decision_sha256": decision_sha,
        "drift_report_sha256": drift_report_sha,
        "residual_report_sha256": residual_sha,
        "aligned_wav_sha256": aligned_sha,
        "snapshot_content_sha256": snapshot_info["content_hash"],
        "snapshot_file_sha256": snapshot_info["file_sha256"],
        "diagnostic_view_sha256": view_sha,
    }
    hashes_path = output_dir / "hashes.json"
    hashes_file_sha = write_json(hashes_path, {"schema_version": 1, "step_hashes": step_hashes})
    run_report = {
        "schema_version": 1,
        "ok": True,
        "capture": payload["capture"],
        "strategy": decision["selected"],
        "accuracy_target_samples": float(config["accuracy_target_samples"]),
        "drift_tolerance_ppm": float(config["drift_tolerance_ppm"]),
        "summary": verification["summary"],
        "resample": {
            "engine": config["resample"]["engine"],
            "block_frames": compensation["block_frames"],
            "sinc_half_width": compensation["sinc_half_width"],
            "comparison": comparison,
            "aligned_wav_retained": keep_aligned,
        },
        "aligned_wav_sha256": aligned_sha,
        "snapshot": {
            "version": snapshot_info["version"],
            "content_hash": snapshot_info["content_hash"],
            "file_sha256": snapshot_info["file_sha256"],
        },
        "hashes_sha256": hashes_file_sha,
        "deterministic_outputs": True,
    }
    write_json(output_dir / "run_report.json", run_report)
    return run_report


def _remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def run_analysis_atomic(
    input_dir: Path,
    output_dir: Path,
    config_path: Path | None = None,
    snapshot_version: int = 1,
    consistency: bool = True,
    keep_aligned: bool | None = None,
) -> dict[str, Any]:
    stage = output_dir.parent / (output_dir.name + ".staging")
    _remove_tree(stage)
    stage.mkdir(parents=True)
    try:
        report = run_analysis(input_dir, stage, config_path, snapshot_version, consistency, keep_aligned)
    except Exception:
        _remove_tree(stage)
        raise
    _remove_tree(output_dir)
    stage.replace(output_dir)
    return report


def write_error_report(path: Path, error: SyncError) -> None:
    write_json(path, error.to_dict())
