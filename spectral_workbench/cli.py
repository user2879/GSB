"""Offline command-line entry points for the diagnostic workbench."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

from .config import CubeConfig, InjectionConfig, PipelineConfig
from .errors import ErrorCode, WorkbenchError
from .generator import generate_cube, load_endmembers, load_manifest, load_truth
from .io_utils import read_json
from .pipeline import PipelineRunner
from .service import DiagnosticService
from .unmix import build_endmember_model, concentration_error_components


_EXIT_CODES = {
    ErrorCode.E_CUBE_FORMAT: 10,
    ErrorCode.E_DARK_GAIN: 11,
    ErrorCode.E_UNMIX_COND: 12,
    ErrorCode.E_DRIFT_BIAS: 13,
    ErrorCode.E_CLOSURE_MISMATCH: 14,
    ErrorCode.E_SNAPSHOT_STALE: 15,
    ErrorCode.E_INJECT_CONFIG: 16,
}


def _emit(value: Any) -> None:
    print(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False))


def _load_injection(value: Optional[str]) -> InjectionConfig:
    if not value:
        return InjectionConfig()
    return InjectionConfig.from_dict(read_json(value))


def _load_config(value: Optional[str]) -> PipelineConfig:
    if not value:
        return PipelineConfig()
    return PipelineConfig.from_dict(read_json(value))


def _apply_overrides(config: PipelineConfig, args: argparse.Namespace) -> PipelineConfig:
    from dataclasses import replace

    calibration = config.calibration
    unmix = config.unmix
    drift = config.drift
    if getattr(args, "calibration_epsilon", None) is not None:
        calibration = replace(calibration, flat_epsilon_dn=float(args.calibration_epsilon))
    if getattr(args, "unmix_order", None) is not None:
        unmix = replace(unmix, order=int(args.unmix_order))
    if getattr(args, "cond_warn", None) is not None:
        unmix = replace(unmix, cond_warn=float(args.cond_warn))
    if getattr(args, "endmember_file", None):
        unmix = replace(unmix, endmember_file=str(args.endmember_file))
    if getattr(args, "no_compensate", False):
        drift = replace(drift, compensate=False)
    if getattr(args, "max_shift", None) is not None:
        drift = replace(drift, max_shift_px=float(args.max_shift))
    tolerance = float(getattr(args, "tolerance", config.concentration_tolerance))
    return replace(config, calibration=calibration, unmix=unmix, drift=drift, concentration_tolerance=tolerance)


def command_generate(args: argparse.Namespace) -> int:
    cube_config = CubeConfig(
        seed=int(args.seed),
        frames=int(args.frames),
        height=int(args.height),
        width=int(args.width),
        fluorophores=int(args.fluorophores),
    )
    injection = _load_injection(args.injections)
    generated = generate_cube(cube_config, injection, args.out)
    manifest = load_manifest(generated.root, verify_hashes=True)
    _emit(
        {
            "status": "ok",
            "root": str(generated.root.resolve()),
            "manifest": str(generated.manifest_path.resolve()),
            "content_hash": generated.content_hash,
            "shape": manifest["shape"],
            "frames": manifest["frame_stats"],
        }
    )
    return 0


def command_run(args: argparse.Namespace) -> int:
    config = _apply_overrides(_load_config(args.config), args)
    runner = PipelineRunner(args.cube, config)
    service = DiagnosticService(runner)
    result = service.recompute(1, args.out, verify_closure=bool(args.verify_closure))
    snapshot = None
    if args.snapshot:
        snapshot = service.build_and_publish_snapshot(int(args.version), args.snapshot)
    _emit(
        {
            "status": "ok",
            "run_hash": result.report["run_hash"],
            "report": str(result.report_path.resolve()),
            "snapshot": str(Path(args.snapshot).resolve()) if snapshot is not None else None,
            "snapshot_hash": snapshot.content_hash if snapshot is not None else None,
            "condition_number": result.report["unmix"]["condition_number"],
            "compensation": result.report["metrics"]["compensation_evaluation"],
            "post_metrics": result.report["metrics"]["post_compensation"],
            "reuse": result.report["incremental"]["stage_reuse"],
            "closure": result.report["closure"],
            "warnings": result.report["warnings"],
        }
    )
    return 0


def command_ui(args: argparse.Namespace) -> int:
    from .ui import show_snapshot_directory

    return show_snapshot_directory(args.snapshot)


def command_inspect_pixel(args: argparse.Namespace) -> int:
    output = Path(args.run)
    report = read_json(output / "diagnostic_report.json")
    config = PipelineConfig.from_dict(report["config"])
    manifest = load_manifest(args.cube, verify_hashes=False)
    names = manifest["fluorophore_names"]
    endmembers = load_endmembers(args.cube, manifest)
    model = build_endmember_model(endmembers, names, config.unmix)
    frame_record = report["incremental"]["frames"][int(args.band)]
    calibration_version = frame_record["calibration"]["version"]
    unmix_version = frame_record["unmix"]["version"]
    from .cache import load_unmix
    from .io_utils import read_json as json_read
    from .cache import stage_dir

    calibration_record = json_read(stage_dir(output / "cache", "calibration", calibration_version) / "record.json")
    hit = load_unmix(output / "cache", unmix_version)
    if hit is None:
        raise WorkbenchError(ErrorCode.E_CUBE_FORMAT, "cached unmix result is unavailable", {"version": unmix_version})
    result = hit[0]
    components = concentration_error_components(
        model,
        result,
        frame_index=int(args.band),
        y=int(args.y),
        x=int(args.x),
    )
    truth = load_truth(args.cube, int(args.band))
    components["truth"] = {
        "drift_xy": truth["drift_xy"],
        "blobs": truth["blobs"],
    }
    components["calibration_record"] = {
        "quantization_upper_bound_dn": calibration_record["quantization_upper_bound_dn"],
        "saturated_pixels": calibration_record["saturated_pixels"],
        "invalid_pixels": calibration_record["invalid_pixels"],
    }
    _emit(components)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spectral-workbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate a deterministic cube and truth manifest")
    generate.add_argument("--out", required=True)
    generate.add_argument("--seed", type=int, default=20261119)
    generate.add_argument("--frames", type=int, default=64)
    generate.add_argument("--height", type=int, default=256)
    generate.add_argument("--width", type=int, default=256)
    generate.add_argument("--fluorophores", type=int, default=4)
    generate.add_argument("--injections", help="path to canonical injection JSON")
    generate.set_defaults(func=command_generate)

    run = subparsers.add_parser("run", help="calibrate, estimate drift, unmix, and report")
    run.add_argument("--cube", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--config", help="path to pipeline config JSON")
    run.add_argument("--snapshot", help="optional snapshot directory to persist")
    run.add_argument("--version", type=int, default=1)
    run.add_argument("--verify-closure", action="store_true")
    run.add_argument("--calibration-epsilon", type=float)
    run.add_argument("--unmix-order", type=int)
    run.add_argument("--cond-warn", type=float)
    run.add_argument("--endmember-file")
    run.add_argument("--no-compensate", action="store_true")
    run.add_argument("--max-shift", type=float)
    run.add_argument("--tolerance", type=float, default=0.08)
    run.set_defaults(func=command_run)

    ui = subparsers.add_parser("ui", help="render one persisted snapshot")
    ui.add_argument("--snapshot", required=True)
    ui.set_defaults(func=command_ui)

    inspect = subparsers.add_parser("inspect-pixel", help="recompute one pixel's uncertainty decomposition")
    inspect.add_argument("--cube", required=True)
    inspect.add_argument("--run", required=True)
    inspect.add_argument("--band", type=int, default=0)
    inspect.add_argument("--y", type=int, required=True)
    inspect.add_argument("--x", type=int, required=True)
    inspect.set_defaults(func=command_inspect_pixel)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except WorkbenchError as error:
        print(json.dumps(error.to_dict(), sort_keys=True, ensure_ascii=False), file=sys.stderr)
        return _EXIT_CODES.get(error.code, 1)
    except Exception as error:  # pragma: no cover - last-resort CLI boundary
        print(json.dumps({"code": "E_UNEXPECTED", "message": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1
