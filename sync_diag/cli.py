from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from .config import load_config
from .errors import SyncError
from .hashing import sha256_file, write_json
from .pipeline import run_analysis_atomic, write_error_report
from .synth import synthesize_dataset


def _remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _synth_atomic(config_path: Path, duration: float | None, output_dir: Path) -> dict[str, Any]:
    stage = output_dir.parent / (output_dir.name + ".staging")
    _remove_tree(stage)
    stage.mkdir(parents=True)
    try:
        config = load_config(config_path, duration_s=duration)
        result = synthesize_dataset(config, stage)
    except Exception:
        _remove_tree(stage)
        raise
    _remove_tree(output_dir)
    stage.replace(output_dir)
    result["wav_path"] = str(output_dir / "capture.wav")
    result["manifest_path"] = str(output_dir / "manifest.json")
    return result


def _print(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))


def command_synth(args: argparse.Namespace) -> int:
    output_dir = Path(args.out).resolve()
    try:
        result = _synth_atomic(Path(args.config).resolve(), args.duration, output_dir)
    except SyncError as exc:
        error_path = output_dir.parent / (output_dir.name + ".error.json")
        write_error_report(error_path, exc)
        _print({**exc.to_dict(), "error_report": str(error_path)})
        return 2
    result["output"] = str(output_dir)
    _print(result)
    return 0


def command_analyze(args: argparse.Namespace) -> int:
    input_dir = Path(args.input).resolve()
    output_dir = Path(args.out).resolve()
    config_path = Path(args.config).resolve() if args.config else None
    try:
        report = run_analysis_atomic(
            input_dir,
            output_dir,
            config_path,
            snapshot_version=args.snapshot_version,
            consistency=not args.skip_consistency,
            keep_aligned=args.keep_aligned,
        )
    except SyncError as exc:
        error_path = output_dir.parent / (output_dir.name + ".error.json")
        write_error_report(error_path, exc)
        _print({**exc.to_dict(), "error_report": str(error_path)})
        return 2
    _print({"ok": True, "output": str(output_dir), "report": report})
    return 0


def _scale_label(seconds: int) -> str:
    return f"{seconds}s"


def command_demo(args: argparse.Namespace) -> int:
    root = Path(args.out).resolve()
    stage = root.parent / (root.name + ".staging")
    _remove_tree(stage)
    stage.mkdir(parents=True)
    config_dir = Path(args.configs).resolve()
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    scales = [int(value.strip()) for value in args.scales.split(",") if value.strip()]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "ok": True,
        "variants": variants,
        "scales_s": scales,
        "sequence": ["synth", "detect_pair", "strategy", "estimate", "resample", "verify", "snapshot", "diagnostic_view", "hashes"],
        "runs": [],
    }
    for variant in variants:
        config_path = config_dir / f"{variant}.json"
        if not config_path.exists():
            summary["ok"] = False
            summary["runs"].append({"variant": variant, "ok": False, "error": {"code": "E_CONFIG", "message": "variant config missing"}})
            continue
        for scale in scales:
            run_name = f"{variant}-{_scale_label(scale)}"
            dataset_dir = stage / "datasets" / run_name
            analysis_dir = stage / "runs" / run_name
            record: dict[str, Any] = {"variant": variant, "scale_s": scale, "run": run_name}
            try:
                synth = _synth_atomic(config_path, float(scale), dataset_dir)
                record["dataset"] = {
                    "frames": synth["frames"],
                    "analysis_frames": synth["analysis_frames"],
                    "capture_guard_frames": synth["capture_guard_frames"],
                    "pulse_count": synth["pulse_count"],
                    "wav_sha256": synth["wav_sha256"],
                    "manifest_sha256": synth["manifest_sha256"],
                }
                report = run_analysis_atomic(
                    dataset_dir,
                    analysis_dir,
                    None,
                    snapshot_version=1,
                    consistency=not args.skip_consistency,
                )
                record.update(
                    {
                        "ok": True,
                        "strategy": report["strategy"],
                        "before_rmse_samples_mean": report["summary"]["before_rmse_samples_mean"],
                        "after_rmse_samples_mean": report["summary"]["after_rmse_samples_mean"],
                        "after_rmse_samples_max": report["summary"]["after_rmse_samples_max"],
                        "accuracy_target_samples": report["accuracy_target_samples"],
                        "target_met": report["summary"]["target_met"],
                        "aligned_wav_sha256": report["aligned_wav_sha256"],
                        "aligned_wav_retained": report["resample"]["aligned_wav_retained"],
                        "snapshot_content_hash": report["snapshot"]["content_hash"],
                        "run_report_sha256": sha256_file(analysis_dir / "run_report.json"),
                    }
                )
            except SyncError as exc:
                record.update({"ok": False, "error": exc.to_dict()["error"]})
                summary["ok"] = False
                write_error_report(stage / "runs" / f"{run_name}.error.json", exc)
            summary["runs"].append(record)
    summary_sha = write_json(stage / "demo_summary.json", summary)
    (stage / "demo_summary.sha256").write_text(summary_sha + "\n", encoding="ascii")
    _remove_tree(root)
    stage.replace(root)
    _print(summary)
    return 0 if summary["ok"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sync-diag", description="Deterministic multichannel clock-sync and drift-compensation diagnostic workbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    synth = subparsers.add_parser("synth", help="generate deterministic 8-channel PCM24 capture and truth manifest")
    synth.add_argument("--config", required=True)
    synth.add_argument("--duration", type=float, default=None)
    synth.add_argument("--out", required=True)
    synth.set_defaults(handler=command_synth)

    analyze = subparsers.add_parser("analyze", help="estimate, compensate, verify, snapshot, and report one capture")
    analyze.add_argument("--input", required=True)
    analyze.add_argument("--out", required=True)
    analyze.add_argument("--config", default=None)
    analyze.add_argument("--snapshot-version", type=int, default=1)
    analyze.add_argument("--skip-consistency", action="store_true")
    analyze.add_argument("--keep-aligned", action=argparse.BooleanOptionalAction, default=None)
    analyze.set_defaults(handler=command_analyze)

    demo = subparsers.add_parser("demo", help="run the deterministic low/high/temperature matrix at 30 s and 30 min")
    demo.add_argument("--configs", default="configs")
    demo.add_argument("--out", default="artifacts/demo")
    demo.add_argument("--variants", default="low_drift,high_drift,temp_drift")
    demo.add_argument("--scales", default="30,1800")
    demo.add_argument("--skip-consistency", action="store_true")
    demo.set_defaults(handler=command_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
