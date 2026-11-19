from __future__ import annotations

import argparse
import json
import math
import sys
import wave
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sync_diag.hashing import sha256_file, sha256_json


EXPECTED_VARIANTS = ("low_drift", "high_drift", "temp_drift")
EXPECTED_SCALES = (30, 1800)


class Recorder:
    def __init__(self) -> None:
        self.checks: dict[str, bool] = {}
        self.failures: list[dict[str, Any]] = []

    def check(self, name: str, condition: bool, **context: Any) -> None:
        passed = bool(condition)
        self.checks[name] = passed
        if not passed:
            self.failures.append({"check": name, "context": context})

    def equal(self, name: str, actual: Any, expected: Any) -> None:
        self.check(name, actual == expected, actual=actual, expected=expected)

    def close(self, name: str, actual: float, expected: float, tolerance: float) -> None:
        self.check(name, abs(float(actual) - float(expected)) <= float(tolerance), actual=actual, expected=expected, tolerance=tolerance)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def wav_info(path: Path) -> dict[str, int]:
    with wave.open(str(path), "rb") as handle:
        return {
            "channels": handle.getnchannels(),
            "bit_depth": handle.getsampwidth() * 8,
            "sample_rate_hz": handle.getframerate(),
            "frames": handle.getnframes(),
        }


def rms(values: list[float]) -> float:
    return math.sqrt(math.fsum(float(value) ** 2 for value in values) / len(values))


def audit_residuals(
    recorder: Recorder,
    residual: dict[str, Any],
    manifest: dict[str, Any],
    accuracy_target: float,
) -> dict[str, Any]:
    reference = [float(value) for value in manifest["reference_pulse_samples"]]
    slaves = [channel for channel in residual["channels"] if channel["role"] == "slave"]
    recorder.equal("residual_channel_count", len(residual["channels"]), 8)
    recorder.equal("residual_slave_count", len(slaves), 7)
    recorder.equal("residual_reference_pulse_count", len(reference), int(manifest.get("pulse_count", len(reference))))

    after_rmse: list[float] = []
    before_rmse: list[float] = []
    max_after = 0.0
    for channel in slaves:
        pulses = channel["pulses"]
        channel_id = int(channel["channel"])
        recorder.equal(f"residual_pulse_count_ch{channel_id}", len(pulses), len(reference))
        recorder.equal(f"residual_pulse_indices_ch{channel_id}", [int(pulse["pulse_index"]) for pulse in pulses], list(range(len(reference))))
        reference_matches = all(
            float(pulse["reference_sample"]) == reference[int(pulse["pulse_index"])]
            for pulse in pulses
        )
        recorder.check(f"residual_reference_truth_ch{channel_id}", reference_matches)

        before_values = [float(pulse["detected_before_sample"]) - float(pulse["reference_sample"]) for pulse in pulses]
        after_values = [float(pulse["detected_aligned_sample"]) - float(pulse["reference_sample"]) for pulse in pulses]
        reported_before = [float(pulse["residual_before_samples"]) for pulse in pulses]
        reported_after = [float(pulse["residual_after_samples"]) for pulse in pulses]
        recorder.close(f"residual_before_equation_ch{channel_id}", max(abs(a - b) for a, b in zip(before_values, reported_before)), 0.0, 1.0e-9)
        recorder.close(f"residual_after_equation_ch{channel_id}", max(abs(a - b) for a, b in zip(after_values, reported_after)), 0.0, 1.0e-9)

        channel_before = rms(reported_before)
        channel_after = rms(reported_after)
        before_rmse.append(channel_before)
        after_rmse.append(channel_after)
        max_after = max(max_after, channel_after)
        recorder.close(f"residual_before_rmse_recompute_ch{channel_id}", channel["before_rmse_samples"], channel_before, 1.0e-12)
        recorder.close(f"residual_after_rmse_recompute_ch{channel_id}", channel["after_rmse_samples"], channel_after, 1.0e-12)
        recorder.close(f"residual_after_max_recompute_ch{channel_id}", channel["after_max_abs_samples"], max(abs(value) for value in reported_after), 1.0e-12)
        recorder.equal(f"residual_channel_status_ch{channel_id}", channel["status"], "IMPROVED")
        recorder.check(f"residual_channel_improved_ch{channel_id}", channel_after < channel_before)

    summary = residual["summary"]
    recorder.close("residual_summary_before_mean_recompute", summary["before_rmse_samples_mean"], math.fsum(before_rmse) / len(before_rmse), 1.0e-12)
    recorder.close("residual_summary_after_mean_recompute", summary["after_rmse_samples_mean"], math.fsum(after_rmse) / len(after_rmse), 1.0e-12)
    recorder.close("residual_summary_after_max_recompute", summary["after_rmse_samples_max"], max_after, 1.0e-12)
    recorder.close("residual_accuracy_target", summary["accuracy_target_samples"], accuracy_target, 0.0)
    recorder.check("residual_target_met", summary["target_met"] and max_after <= accuracy_target)
    recorder.equal("residual_improvement_status", summary["status"], "IMPROVED")
    recorder.check("residual_all_channels_improved", summary["all_channels_improved"])
    return {
        "before_rmse_samples_mean": summary["before_rmse_samples_mean"],
        "after_rmse_samples_mean": summary["after_rmse_samples_mean"],
        "after_rmse_samples_max": summary["after_rmse_samples_max"],
        "target_met": bool(summary["target_met"]),
    }


def audit_drift(recorder: Recorder, drift: dict[str, Any], decision: dict[str, Any], tolerance_ppm: float) -> None:
    slaves = [channel for channel in drift["channels"] if channel["role"] == "slave"]
    recorder.equal("drift_channel_count", len(drift["channels"]), 8)
    recorder.equal("drift_slave_count", len(slaves), 7)
    recorder.equal("drift_selected_strategy", drift["selected_strategy"], decision["selected"])
    recorder.close("drift_tolerance_reported", drift["tolerance_ppm"], tolerance_ppm, 0.0)
    for channel in slaves:
        channel_id = int(channel["channel"])
        segments = channel["segments"]
        recorder.check(f"drift_segments_present_ch{channel_id}", bool(segments))
        recorder.check(f"drift_rate_within_tolerance_ch{channel_id}", all(abs(float(segment["rate_ppm"])) <= tolerance_ppm for segment in segments))
        recorder.check(f"drift_rate_ci_ordered_ch{channel_id}", all(float(segment["rate_ppm_ci95_low"]) <= float(segment["rate_ppm"]) <= float(segment["rate_ppm_ci95_high"]) for segment in segments))
        recorder.check(f"drift_offset_ci_nonnegative_ch{channel_id}", float(channel["fixed_delay_ci95_samples"]) >= 0.0)
        recorder.check(f"drift_fit_residual_finite_ch{channel_id}", math.isfinite(float(channel["fit_residual_rmse_samples"])))
        curve_x = [float(value) for value in channel["curve"]["x_samples"]]
        curve_y = [float(value) for value in channel["curve"]["offset_samples"]]
        recorder.equal(f"drift_curve_length_ch{channel_id}", len(curve_x), len(curve_y))
        recorder.check(f"drift_curve_sorted_ch{channel_id}", curve_x == sorted(curve_x))
        recorder.check(f"drift_curve_finite_ch{channel_id}", all(math.isfinite(value) for value in curve_x + curve_y))


def audit_run(
    demo_root: Path,
    summary_record: dict[str, Any],
    expected_variant: str,
    expected_scale: int,
) -> dict[str, Any]:
    recorder = Recorder()
    run_name = str(summary_record["run"])
    dataset_dir = demo_root / "datasets" / run_name
    run_dir = demo_root / "runs" / run_name
    recorder.equal("summary_variant", summary_record["variant"], expected_variant)
    recorder.equal("summary_scale_s", summary_record["scale_s"], expected_scale)
    recorder.check("summary_ok", bool(summary_record["ok"]))

    required_dataset = ("capture.wav", "config.resolved.json", "manifest.json", "manifest.json.sha256")
    required_run = (
        "aligned.wav",
        "decision.json",
        "diagnostic_view.svg",
        "drift_report.json",
        "hashes.json",
        "pulse_pairing.json",
        "residual_report.json",
        "run_report.json",
        "snapshots/current_snapshot.json",
    )
    for name in required_dataset:
        recorder.check(f"file_exists_dataset_{name}", (dataset_dir / name).is_file())
    for name in required_run:
        expected = name != "aligned.wav" or bool(summary_record["aligned_wav_retained"])
        recorder.equal(f"file_existence_run_{name}", (run_dir / name).is_file(), expected)

    manifest_path = dataset_dir / "manifest.json"
    sidecar_path = dataset_dir / "manifest.json.sha256"
    wav_path = dataset_dir / "capture.wav"
    config_path = dataset_dir / "config.resolved.json"
    manifest = read_json(manifest_path)
    config = read_json(config_path)
    run_report = read_json(run_dir / "run_report.json")
    hashes = read_json(run_dir / "hashes.json")["step_hashes"]
    decision = read_json(run_dir / "decision.json")
    drift = read_json(run_dir / "drift_report.json")
    residual = read_json(run_dir / "residual_report.json")

    manifest_file_sha = sha256_file(manifest_path)
    manifest_sidecar_sha = sidecar_path.read_text(encoding="ascii").strip()
    capture_sha = sha256_file(wav_path)
    config_content_sha = sha256_json({key: value for key, value in config.items() if key != "config_sha256"})
    recorder.equal("manifest_sidecar", manifest_sidecar_sha, manifest_file_sha)
    recorder.equal("manifest_wav_hash", manifest["wav_sha256"], capture_sha)
    recorder.equal("manifest_frame_count", manifest["frames"], int(run_report["capture"]["frames"]))
    recorder.equal("manifest_analysis_frames", manifest["analysis_frames"], int(run_report["capture"]["analysis_frames"]))
    recorder.equal("manifest_channel_count", manifest["channel_count"], 8)
    recorder.equal("manifest_bit_depth", manifest["bit_depth"], 24)
    recorder.equal("manifest_sample_rate_hz", manifest["sample_rate_hz"], 48000)
    recorder.equal("manifest_reference_pulse_count", len(manifest["reference_pulse_samples"]), int(run_report["capture"]["pulse_count"]))
    recorder.equal("config_channel_count", config["channel_count"], 8)
    recorder.equal("config_bit_depth", config["bit_depth"], 24)
    recorder.equal("config_sample_rate_hz", config["sample_rate_hz"], 48000)
    recorder.equal("config_content_hash", config["config_sha256"], config_content_sha)

    actual_wav = wav_info(wav_path)
    recorder.equal("wav_channel_count", actual_wav["channels"], 8)
    recorder.equal("wav_bit_depth", actual_wav["bit_depth"], 24)
    recorder.equal("wav_sample_rate_hz", actual_wav["sample_rate_hz"], 48000)
    recorder.equal("wav_frame_count", actual_wav["frames"], int(manifest["frames"]))
    recorder.equal("wav_file_size", wav_path.stat().st_size, 44 + int(actual_wav["frames"]) * 8 * 3)

    step_files = {
        "decision_sha256": run_dir / "decision.json",
        "diagnostic_view_sha256": run_dir / "diagnostic_view.svg",
        "drift_report_sha256": run_dir / "drift_report.json",
        "pulse_pairing_sha256": run_dir / "pulse_pairing.json",
        "residual_report_sha256": run_dir / "residual_report.json",
    }
    for hash_name, path in step_files.items():
        recorder.equal(f"step_hash_{hash_name}", hashes[hash_name], sha256_file(path))
    recorder.equal("step_hash_config_sha256", hashes["config_sha256"], config_content_sha)
    recorder.equal("step_hash_manifest_file_sha256", hashes["manifest_file_sha256"], manifest_file_sha)
    recorder.equal("step_hash_capture_wav_sha256", hashes["capture_wav_sha256"], capture_sha)

    aligned_path = run_dir / "aligned.wav"
    if aligned_path.is_file():
        recorder.equal("step_hash_aligned_wav_sha256", hashes["aligned_wav_sha256"], sha256_file(aligned_path))
    else:
        recorder.equal("aligned_hash_without_file", hashes["aligned_wav_sha256"], run_report["aligned_wav_sha256"])

    snapshot_store = run_dir / "snapshots"
    pointer = read_json(snapshot_store / "current_snapshot.json")
    snapshot_path = snapshot_store / str(pointer["file"])
    recorder.check("snapshot_file_exists", snapshot_path.is_file())
    snapshot_file_sha = sha256_file(snapshot_path)
    snapshot = read_json(snapshot_path)
    snapshot_content_sha = sha256_json(snapshot["payload"])
    recorder.equal("snapshot_pointer_file_hash", pointer["file_sha256"], snapshot_file_sha)
    recorder.equal("snapshot_pointer_content_hash", pointer["content_hash"], snapshot_content_sha)
    recorder.equal("snapshot_document_version", snapshot["version"], pointer["version"])
    recorder.equal("snapshot_document_content_hash", snapshot["content_hash"], snapshot_content_sha)
    recorder.equal("step_hash_snapshot_content_sha256", hashes["snapshot_content_sha256"], snapshot_content_sha)
    recorder.equal("step_hash_snapshot_file_sha256", hashes["snapshot_file_sha256"], snapshot_file_sha)
    recorder.equal("snapshot_matches_run_report_content", run_report["snapshot"]["content_hash"], snapshot_content_sha)
    recorder.equal("snapshot_matches_run_report_file", run_report["snapshot"]["file_sha256"], snapshot_file_sha)
    recorder.equal("snapshot_payload_decision", snapshot["payload"]["decision"], decision)
    recorder.equal("snapshot_payload_estimate", snapshot["payload"]["estimate"], drift)
    recorder.equal("snapshot_payload_verification", snapshot["payload"]["verification"], residual)
    recorder.equal("snapshot_payload_capture", snapshot["payload"]["capture"], run_report["capture"])
    svg_text = (run_dir / "diagnostic_view.svg").read_text(encoding="utf-8")
    recorder.check("view_references_snapshot_hash", snapshot_file_sha in svg_text)

    recorder.equal("run_report_hashes_hash", run_report["hashes_sha256"], sha256_file(run_dir / "hashes.json"))
    recorder.equal("summary_run_report_hash", summary_record["run_report_sha256"], sha256_file(run_dir / "run_report.json"))
    recorder.equal("summary_strategy", summary_record["strategy"], run_report["strategy"])
    recorder.equal("summary_aligned_hash", summary_record["aligned_wav_sha256"], run_report["aligned_wav_sha256"])
    recorder.equal("summary_aligned_retained", summary_record["aligned_wav_retained"], run_report["resample"]["aligned_wav_retained"])
    recorder.equal("summary_snapshot_hash", summary_record["snapshot_content_hash"], snapshot_content_sha)
    recorder.equal("run_strategy_decision", run_report["strategy"], decision["selected"])
    recorder.equal("run_accuracy_target", run_report["accuracy_target_samples"], config["accuracy_target_samples"])
    recorder.check("run_ok", run_report["ok"] and run_report["deterministic_outputs"])

    selection = decision["selected"]
    recorder.check("strategy_selected_feasible", selection in decision["feasible"])
    recorder.check("strategy_error_bound_within_target", float(decision["all_error_bounds_samples"][selection]) <= float(decision["accuracy_target_samples"]))
    recorder.check("strategy_memory_within_budget", float(decision["resource_model"]["memory_mb"]) <= float(decision["memory_budget_mb"]))
    recorder.check("strategy_time_within_budget", float(decision["resource_model"]["time_s"]) <= float(decision["time_budget_s"]))
    recorder.equal("strategy_accuracy_target_preserved", decision["accuracy_target_samples"], config["accuracy_target_samples"])

    comparison = run_report["resample"]["comparison"]
    recorder.check("block_comparison_present", isinstance(comparison, dict))
    recorder.check("block_comparison_consistent", bool(comparison["consistent"]))
    recorder.check("block_comparison_byte_identical", bool(comparison["byte_identical"]))
    recorder.close("block_comparison_residual_delta", float(comparison["residual_delta_samples"]), 0.0, float(comparison["tolerance_samples"]))
    recorder.check("block_comparison_phase_continuity", float(comparison["continuity"]["max_coordinate_step_deviation_samples"]) <= float(comparison["continuity"]["tolerance_samples"]))

    audit_drift(recorder, drift, decision, float(config["drift_tolerance_ppm"]))
    metrics = audit_residuals(recorder, residual, manifest, float(config["accuracy_target_samples"]))
    recorder.check("run_target_met", bool(run_report["summary"]["target_met"]))
    recorder.equal("summary_target_met", summary_record["target_met"], run_report["summary"]["target_met"])
    recorder.close("summary_after_max", summary_record["after_rmse_samples_max"], metrics["after_rmse_samples_max"], 0.0)

    return {
        "run": run_name,
        "variant": expected_variant,
        "scale_s": expected_scale,
        "ok": all(recorder.checks.values()),
        "strategy": run_report["strategy"],
        "target_met": bool(run_report["summary"]["target_met"]),
        "metrics": metrics,
        "capture_sha256": capture_sha,
        "manifest_file_sha256": manifest_file_sha,
        "config_content_sha256": config_content_sha,
        "run_report_sha256": sha256_file(run_dir / "run_report.json"),
        "hashes_file_sha256": sha256_file(run_dir / "hashes.json"),
        "snapshot_content_sha256": snapshot_content_sha,
        "snapshot_file_sha256": snapshot_file_sha,
        "diagnostic_view_sha256": sha256_file(run_dir / "diagnostic_view.svg"),
        "aligned_wav_sha256": run_report["aligned_wav_sha256"],
        "aligned_wav_retained": bool(run_report["resample"]["aligned_wav_retained"]),
        "pulse_count": len(manifest["reference_pulse_samples"]),
        "frames": int(manifest["frames"]),
        "analysis_frames": int(manifest["analysis_frames"]),
        "capture_bytes": wav_path.stat().st_size,
        "run_artifact_bytes": sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file()),
        "checks": recorder.checks,
        "failures": recorder.failures,
    }


def build_validation(demo_root: Path) -> dict[str, Any]:
    global_recorder = Recorder()
    summary_path = demo_root / "demo_summary.json"
    summary_sidecar_path = demo_root / "demo_summary.sha256"
    summary = read_json(summary_path)
    summary_sha = sha256_file(summary_path)
    global_recorder.check("demo_summary_ok", bool(summary["ok"]))
    global_recorder.equal("demo_summary_sidecar", summary_sidecar_path.read_text(encoding="ascii").strip(), summary_sha)
    global_recorder.equal("demo_summary_schema", summary["schema_version"], 1)
    global_recorder.equal("demo_summary_variants", summary["variants"], list(EXPECTED_VARIANTS))
    global_recorder.equal("demo_summary_scales", summary["scales_s"], list(EXPECTED_SCALES))
    expected_runs = [f"{variant}-{scale}s" for variant in EXPECTED_VARIANTS for scale in EXPECTED_SCALES]
    actual_runs = [record["run"] for record in summary["runs"]]
    global_recorder.equal("demo_run_names", actual_runs, expected_runs)
    global_recorder.equal("demo_run_count", len(summary["runs"]), 6)
    global_recorder.check("no_staging_directories", not (demo_root / "datasets.staging").exists() and not (demo_root / "runs.staging").exists())

    run_results = []
    variants_by_scale: dict[int, list[str]] = {}
    for record in summary["runs"]:
        variant, scale_text = str(record["run"]).rsplit("-", 1)
        scale = int(scale_text.removesuffix("s"))
        result = audit_run(demo_root, record, variant, scale)
        run_results.append(result)
        variants_by_scale.setdefault(scale, []).append(result["strategy"])

    for scale, strategies in sorted(variants_by_scale.items()):
        expected = "full_exact" if scale == 30 else "block_accumulation"
        global_recorder.check(f"strategy_for_scale_{scale}", all(value == expected for value in strategies), actual=strategies, expected=expected)
    all_runs_ok = all(run["ok"] for run in run_results)
    global_recorder.check("all_runs_ok", all_runs_ok)
    global_recorder.equal("all_run_failures_empty", sum(len(run["failures"]) for run in run_results), 0)

    demo_relative = str(demo_root.resolve().relative_to(ROOT)) if demo_root.resolve().is_relative_to(ROOT) else demo_root.name
    validation = {
        "schema_version": 1,
        "ok": all(global_recorder.checks.values()) and all_runs_ok,
        "audit": "read-only independent hash and residual recomputation",
        "demo_root": demo_relative,
        "demo_summary": {
            "sha256": summary_sha,
            "declared_sha256": summary_sidecar_path.read_text(encoding="ascii").strip(),
            "run_count": len(run_results),
        },
        "checks": global_recorder.checks,
        "failures": global_recorder.failures,
        "totals": {
            "capture_bytes": sum(run["capture_bytes"] for run in run_results),
            "run_artifact_bytes": sum(run["run_artifact_bytes"] for run in run_results),
            "runs_ok": sum(1 for run in run_results if run["ok"]),
            "runs_target_met": sum(1 for run in run_results if run["target_met"]),
        },
        "runs": run_results,
    }
    return validation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit deterministic demo artifacts without modifying synchronization state")
    parser.add_argument("--demo", default="artifacts/demo", help="demo output directory")
    parser.add_argument("--out", default=None, help="validation JSON path (default: <demo>/validation.json)")
    args = parser.parse_args(argv)

    demo_root = Path(args.demo)
    if not demo_root.is_absolute():
        demo_root = ROOT / demo_root
    output_path = Path(args.out) if args.out else demo_root / "validation.json"
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    validation = build_validation(demo_root)
    text = json.dumps(validation, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    output_path.write_bytes(text.encode("utf-8"))
    output_path.with_suffix(".sha256").write_text(sha256_file(output_path) + "\n", encoding="ascii")
    print(json.dumps({
        "ok": validation["ok"],
        "runs": validation["demo_summary"]["run_count"],
        "runs_target_met": validation["totals"]["runs_target_met"],
        "validation": str(output_path),
        "validation_sha256": sha256_file(output_path),
    }, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if validation["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
