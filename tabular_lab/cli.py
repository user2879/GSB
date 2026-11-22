"""Command-line interface for the tabular regularization lab."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

from .config import load_config
from .data import dataset_summary, generate_dataset
from .errors import (
    E_CONFIG_INVALID,
    E_INTERNAL,
    E_SNAPSHOT_STALE,
    LabError,
    fail,
)
from .model_io import (
    compare_records,
    read_model_file,
    write_v1,
    write_v2,
)
from .runner import run_experiment
from .util import to_jsonable


EXIT_CONFIG = 2
EXIT_STALE = 3
EXIT_PARTIAL = 4
EXIT_INTERRUPTED = 130


def _configure_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8")
            except (OSError, ValueError):
                pass


def _print_json(payload: Any) -> None:
    print(json.dumps(to_jsonable(payload), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))


def _install_stop_handlers(stop_event: threading.Event) -> None:
    def handler(signum: int, frame: Any) -> None:
        stop_event.set()

    for signal_name in ("SIGINT", "SIGTERM"):
        if hasattr(signal, signal_name):
            try:
                signal.signal(getattr(signal, signal_name), handler)
            except (ValueError, OSError):
                pass


def _command_run(args: argparse.Namespace) -> int:
    stop_event = threading.Event()
    _install_stop_handlers(stop_event)
    state_dir = Path(args.state_dir).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve() if args.report else state_dir / "last_run_report.json"
    report = run_experiment(
        args.config,
        state_dir=state_dir,
        rebuild=args.rebuild,
        resume=args.resume,
        exchange_version=args.exchange_version,
        report_path=report_path,
        stop_event=stop_event,
    )
    _print_json(report)
    return int(report.get("exit_code", 0))


def _command_validate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    dataset = generate_dataset(config.data, config.seed)
    model_features = {}
    for model in config.models:
        indices = dataset.resolve_features(model.features)
        selected = dataset.selected(indices)
        model_features[model.name] = {
            "scheme": model.scheme,
            "features": selected.feature_names,
            "feature_indices": list(selected.indices),
            "data_slice_fingerprint": selected.data_slice_fingerprint(),
            "standardization_fingerprint": selected.standardization_fingerprint(),
        }
    _print_json(
        {
            "status": "ok",
            "experiment": config.name,
            "config_hash": config.config_hash,
            "exchange_version": config.exchange_version,
            "data": dataset_summary(dataset),
            "models": model_features,
        }
    )
    return 0


def _input_version(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        fail(E_CONFIG_INVALID, f"cannot read model input: {exc}", path=str(path))
    stripped = text.lstrip()
    return "v2" if stripped.startswith("{") or stripped.startswith("[") else "v1"


def _roundtrip_check(records: Any, target_version: str, tolerance: float) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="tabular-lab-roundtrip-") as temporary:
        root = Path(temporary)
        if target_version == "v1":
            output = root / "models.v1.txt"
            sidecars = root / "sidecars"
            write_v1(records, output)
            from .model_io import write_downgrade_sidecar

            for record in records:
                write_downgrade_sidecar(record, sidecars)
            parsed = read_model_file(output, sidecar_dir=sidecars)
        else:
            output = root / "models.v2.json"
            write_v2(records, output)
            parsed = read_model_file(output)
        comparison = compare_records(records, parsed)
        max_delta = max(
            float(comparison.get("max_abs_coefficient_delta", 0.0)),
            float(comparison.get("max_abs_intercept_delta", 0.0)),
        )
        return {
            "max_abs_delta": max_delta,
            "tolerance": float(tolerance),
            "passed": bool(comparison.get("equal_names")) and max_delta <= tolerance,
            "comparison": comparison,
        }


def _command_convert(args: argparse.Namespace) -> int:
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    sidecar_input = Path(args.sidecar_input).expanduser().resolve() if args.sidecar_input else None
    source_version = _input_version(input_path)
    records = read_model_file(input_path, sidecar_dir=sidecar_input)
    if args.to == "v1":
        sidecar_dir = Path(args.sidecar_dir).expanduser().resolve() if args.sidecar_dir else output_path.parent / "sidecars"
        write_v1(records, output_path)
        from .model_io import write_downgrade_sidecar

        sidecars = [str(write_downgrade_sidecar(record, sidecar_dir)) for record in records]
        loaded = read_model_file(output_path, sidecar_dir=sidecar_dir)
        report = {
            "status": "ok",
            "source_version": source_version,
            "target_version": "v1",
            "input": str(input_path),
            "output": str(output_path),
            "sidecar_dir": str(sidecar_dir),
            "sidecars": sidecars,
            "models": [record.name for record in records],
            "roundtrip": _roundtrip_check(records, "v1", args.tolerance),
            "loaded_model_count": len(loaded),
        }
    else:
        write_v2(records, output_path)
        report = {
            "status": "ok",
            "source_version": source_version,
            "target_version": "v2",
            "input": str(input_path),
            "output": str(output_path),
            "models": [record.name for record in records],
            "roundtrip": _roundtrip_check(records, "v2", args.tolerance),
        }
    _print_json(report)
    return 0 if report["roundtrip"].get("passed") else EXIT_PARTIAL


def _command_inspect(args: argparse.Namespace) -> int:
    sidecar_dir = Path(args.sidecar_dir).expanduser().resolve() if args.sidecar_dir else None
    known_features = args.known_features.split(",") if args.known_features else None
    records = read_model_file(args.input, sidecar_dir=sidecar_dir, known_feature_names=known_features)
    _print_json(
        {
            "status": "ok",
            "input": str(Path(args.input).expanduser().resolve()),
            "models": [record.canonical() for record in records],
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tabular-lab", description="Deterministic regularized tabular-model lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run an incremental experiment")
    run_parser.add_argument("--config", required=True, help="experiment YAML file")
    run_parser.add_argument("--state-dir", default=".lab-state", help="snapshot/cache state directory")
    mode = run_parser.add_mutually_exclusive_group()
    mode.add_argument("--rebuild", action="store_true", help="ignore previous cache and rebuild from blank")
    mode.add_argument("--resume", action="store_true", help="require config/data-compatible snapshot recovery")
    run_parser.add_argument("--exchange-version", choices=("v1", "v2"), help="override the exchange version")
    run_parser.add_argument("--report", help="write the JSON report to this path")
    run_parser.set_defaults(function=_command_run)

    validate_parser = subparsers.add_parser("validate", help="validate config/data and print fingerprints")
    validate_parser.add_argument("--config", required=True)
    validate_parser.set_defaults(function=_command_validate)

    convert_parser = subparsers.add_parser("convert", help="convert model files between v1 and v2")
    convert_parser.add_argument("--input", required=True)
    convert_parser.add_argument("--to", required=True, choices=("v1", "v2"))
    convert_parser.add_argument("--output", required=True)
    convert_parser.add_argument("--sidecar-dir", help="output downgrade sidecar directory (v2 -> v1)")
    convert_parser.add_argument("--sidecar-input", help="input downgrade sidecar directory (v1 load)")
    convert_parser.add_argument("--tolerance", type=float, default=0.001)
    convert_parser.set_defaults(function=_command_convert)

    inspect_parser = subparsers.add_parser("inspect", help="parse and inspect a v1/v2 model file")
    inspect_parser.add_argument("--input", required=True)
    inspect_parser.add_argument("--sidecar-dir")
    inspect_parser.add_argument("--known-features", help="comma-separated feature schema for column validation")
    inspect_parser.set_defaults(function=_command_inspect)
    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.function(args))
    except LabError as exc:
        exit_code = EXIT_STALE if exc.code == E_SNAPSHOT_STALE else EXIT_CONFIG
        _print_json({"status": "error", "error": exc.to_dict(), "exit_code": exit_code})
        return exit_code
    except KeyboardInterrupt:
        _print_json(
            {
                "status": "error",
                "error": {"code": "E_INTERRUPTED", "message": "interrupted before a snapshot could be committed"},
                "exit_code": EXIT_INTERRUPTED,
            }
        )
        return EXIT_INTERRUPTED
    except Exception as exc:
        _print_json(
            {
                "status": "error",
                "error": {"code": E_INTERNAL, "message": str(exc), "exception_type": type(exc).__name__},
                "exit_code": 1,
            }
        )
        return 1
