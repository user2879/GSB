from __future__ import annotations

import argparse
import json
import platform
import sqlite3
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .errors import PipelineError
from .logging_utils import StructuredLogger
from .pipeline import execute_pipeline, recover_pipeline
from .samplegen import generate_samples
from .spec import (
    MAX_SAFE_TIMESTAMP_MS,
    SUPPORTED_FORMAT_VERSIONS,
    load_dataset,
)
from .util import load_json_file, pretty_json
from .verifier import verify_pipeline


EXIT_CODES = {
    "OK": 0,
    "E_SPEC": 2,
    "E_FIELD": 3,
    "E_WINDOW": 4,
    "E_VERSION": 5,
    "E_REPLAY": 6,
    "E_VERIFY": 7,
    "E_STATE": 8,
    "E_IO": 9,
}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", help="JSON manifest with paths and pipeline config")
    parser.add_argument("--services", help="service dependency graph JSON")
    parser.add_argument("--rules", help="root-cause rule JSON")
    parser.add_argument("--alerts", help="mixed-version NDJSON alert stream")
    parser.add_argument("--database", help="SQLite ledger path")
    parser.add_argument("--log", help="structured JSONL log path")
    parser.add_argument("--output-dir", help="report and canonical conclusion directory")
    parser.add_argument("--declaration-version", type=_positive_int)
    parser.add_argument("--format-version", type=int)
    parser.add_argument("--silence-window-ms", type=_nonnegative_int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch-size", type=_positive_int)
    parser.add_argument("--global-concurrency", type=_positive_int)
    parser.add_argument("--per-service-concurrency", type=_positive_int)


def _manifest_values(path: str | None) -> tuple[dict[str, Any], Path]:
    if not path:
        return {}, Path.cwd()
    manifest_path = Path(path).resolve()
    try:
        value = load_json_file(manifest_path)
    except FileNotFoundError as exc:
        raise PipelineError("E_IO", f"manifest does not exist: {manifest_path}") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise PipelineError(
            "E_SPEC",
            f"manifest JSON is invalid: {exc}",
            line=getattr(exc, "lineno", 1),
            field="$",
        ) from exc
    if not isinstance(value, dict):
        raise PipelineError("E_SPEC", "manifest root must be an object", field="$")
    return value, manifest_path.parent


def _path_value(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    base: Path,
    argument: str,
    manifest_key: str,
    *,
    required: bool,
    default: str | None = None,
) -> Path | None:
    cli_value = getattr(args, argument)
    if cli_value is not None:
        value = cli_value
        value_base = Path.cwd()
    else:
        value = manifest.get(manifest_key)
        value_base = base
    if value is None:
        if default is not None:
            value = default
        elif required:
            raise PipelineError(
                "E_STATE",
                f"missing required path: --{argument.replace('_', '-')} or manifest.{manifest_key}",
                field=manifest_key,
            )
        else:
            return None
    path = Path(value)
    if not path.is_absolute():
        path = value_base / path
    return path.resolve()


def _config_value(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    argument: str,
    manifest_key: str,
    default: Any,
) -> Any:
    cli_value = getattr(args, argument)
    if cli_value is not None:
        return cli_value
    return manifest.get(manifest_key, default)


def build_config(args: argparse.Namespace) -> PipelineConfig:
    manifest, base = _manifest_values(args.manifest)
    services = _path_value(
        args, manifest, base, "services", "services_path", required=True
    )
    rules = _path_value(args, manifest, base, "rules", "rules_path", required=True)
    alerts = _path_value(args, manifest, base, "alerts", "alerts_path", required=True)
    database = _path_value(
        args,
        manifest,
        base,
        "database",
        "database_path",
        required=False,
        default="data/pipeline.db",
    )
    log = _path_value(
        args,
        manifest,
        base,
        "log",
        "log_path",
        required=False,
        default="logs/pipeline.jsonl",
    )
    output_dir = _path_value(
        args,
        manifest,
        base,
        "output_dir",
        "output_dir",
        required=False,
        default="artifacts/run",
    )
    assert services and rules and alerts and database and log and output_dir
    declaration_version = _config_value(
        args, manifest, "declaration_version", "declaration_version", 1
    )
    format_version = _config_value(
        args, manifest, "format_version", "format_version", 3
    )
    silence_window_ms = _config_value(
        args,
        manifest,
        "silence_window_ms",
        "default_silence_window_ms",
        30000,
    )
    seed = _config_value(args, manifest, "seed", "seed", 20261122)
    batch_size = _config_value(args, manifest, "batch_size", "batch_size", 8)
    global_concurrency = _config_value(
        args, manifest, "global_concurrency", "global_concurrency", 4
    )
    per_service_concurrency = _config_value(
        args,
        manifest,
        "per_service_concurrency",
        "per_service_concurrency",
        1,
    )
    for name, value in (
        ("declaration_version", declaration_version),
        ("batch_size", batch_size),
        ("global_concurrency", global_concurrency),
        ("per_service_concurrency", per_service_concurrency),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise PipelineError(
                "E_STATE", f"{name} must be a positive integer", field=name
            )
    if format_version not in SUPPORTED_FORMAT_VERSIONS:
        raise PipelineError(
            "E_FIELD",
            f"unsupported pipeline format version: {format_version}",
            field="format_version",
            details={"supported_versions": list(SUPPORTED_FORMAT_VERSIONS)},
        )
    if not isinstance(silence_window_ms, int) or silence_window_ms < 0:
        raise PipelineError(
            "E_FIELD",
            "default_silence_window_ms must be a non-negative integer",
            field="default_silence_window_ms",
        )
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise PipelineError("E_FIELD", "seed must be an integer", field="seed")
    if per_service_concurrency > global_concurrency:
        raise PipelineError(
            "E_STATE",
            "per-service concurrency cannot exceed global concurrency",
            field="per_service_concurrency",
        )
    return PipelineConfig(
        declaration_version=declaration_version,
        format_version=format_version,
        default_silence_window_ms=silence_window_ms,
        seed=seed,
        batch_size=batch_size,
        global_concurrency=global_concurrency,
        per_service_concurrency=per_service_concurrency,
        services_path=services,
        rules_path=rules,
        alerts_path=alerts,
        database_path=database,
        log_path=log,
        output_dir=output_dir,
    )


def _inject_alert_file(config: PipelineConfig, mode: str) -> PipelineConfig:
    if mode == "none":
        return config
    source = config.alerts_path
    injection_dir = config.output_dir / "injected"
    injection_dir.mkdir(parents=True, exist_ok=True)
    target = injection_dir / f"alerts-{mode}.ndjson"
    lines = source.read_text(encoding="utf-8-sig").splitlines(keepends=True)
    if not lines:
        raise PipelineError("E_STATE", "cannot inject into an empty alert stream")
    if mode == "corrupt":
        lines.append('{"version":3,"timestamp_ms":')
    elif mode == "unsupported-version":
        record = json.loads(lines[0])
        record["version"] = 99
        record.pop("format_version", None)
        record.pop("schema_version", None)
        lines[0] = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    elif mode == "window-conflict":
        record = json.loads(lines[0])
        record["timestamp_ms"] = MAX_SAFE_TIMESTAMP_MS
        record["silence_window_ms"] = 1
        for key in ("timestamp",):
            record.pop(key, None)
        lines[0] = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    else:
        raise PipelineError(
            "E_STATE",
            f"unknown injection mode: {mode}",
            field="inject",
        )
    target.write_text("".join(lines), encoding="utf-8", newline="\n")
    return replace(config, alerts_path=target.resolve())


def _print(value: Any) -> None:
    print(pretty_json(value))


def _error_exit(error: PipelineError, logger: StructuredLogger | None, **fields: Any) -> int:
    if logger is not None:
        logger.error(error, **fields)
    _print(error.to_dict())
    return EXIT_CODES.get(error.code, 8)


def command_generate(args: argparse.Namespace) -> int:
    paths = generate_samples(args.output, seed=args.seed)
    _print({"business_code": "OK", "files": paths})
    return 0


def command_run(args: argparse.Namespace) -> int:
    config = build_config(args)
    logger = StructuredLogger(config.log_path)
    try:
        config = _inject_alert_file(config, args.inject)
        dataset = load_dataset(config)
        outcome = execute_pipeline(
            config,
            dataset,
            mode=args.mode,
            global_limit=config.global_concurrency,
            per_service_limit=config.per_service_concurrency,
            crash_before_commit_on_batch=args.crash_before_commit_batch,
            logger=logger,
        )
        report = outcome.report
        summary = {
            "business_code": "OK",
            "generation_id": report["generation_id"],
            "declaration_version": report["declaration_version"],
            "status": report["status"],
            "already_complete": report["already_complete"],
            "alert_count": report["alert_count"],
            "aggregate_count": report["aggregate_count"],
            "ledger_count": report["ledger_count"],
            "batch_count": len(report["batches"]),
            "batches": [
                {
                    "batch_index": batch["batch_index"],
                    "start_cursor": batch["start_cursor"],
                    "end_cursor": batch["end_cursor"],
                    "business_code": batch["business_code"],
                    "peak_global": batch["peak_global"],
                    "peak_by_service": batch["peak_by_service"],
                    "summary_digest": batch["summary_digest"],
                }
                for batch in report["batches"]
            ],
            "conclusion_digest": report["conclusion_digest"],
            "replacement_affected_count": (
                report.get("replacement", {}).get("affected_count", 0)
                if report.get("replacement")
                else 0
            ),
            "artifacts": report.get("artifacts"),
        }
        if args.inject != "none":
            raise PipelineError(
                "E_STATE",
                "injected anomaly was not rejected by the loader",
                field="inject",
            )
        _print(summary)
        return 0
    except PipelineError as error:
        return _error_exit(error, logger, command="run", inject=args.inject)


def command_verify(args: argparse.Namespace) -> int:
    config = build_config(args)
    logger = StructuredLogger(config.log_path)
    try:
        dataset = load_dataset(config)
        report = verify_pipeline(config, dataset, logger=logger)
        _print(report)
        return 0
    except PipelineError as error:
        return _error_exit(error, logger, command="verify")


def command_recover(args: argparse.Namespace) -> int:
    config = build_config(args)
    logger = StructuredLogger(config.log_path)
    try:
        dataset = load_dataset(config)
        report = recover_pipeline(
            config,
            dataset,
            declaration_version=config.declaration_version,
            logger=logger,
        )
        _print(report)
        return 0
    except PipelineError as error:
        return _error_exit(error, logger, command="recover")


def command_diagnose(_: argparse.Namespace) -> int:
    import importlib.util

    tkinter_available = importlib.util.find_spec("tkinter") is not None
    numpy_version = None
    try:
        import numpy

        numpy_version = numpy.__version__
    except Exception:
        pass
    result = {
        "business_code": "OK",
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "sqlite3": sqlite3.sqlite_version,
        "numpy": numpy_version,
        "tkinter_available": tkinter_available,
        "requirements": {
            "python_3_11_or_newer": sys.version_info >= (3, 11),
            "sqlite3_available": True,
            "numpy_1_26_or_newer": bool(numpy_version),
            "tkinter_available": tkinter_available,
        },
    }
    _print(result)
    return 0 if all(result["requirements"].values()) else 10


def command_gui(args: argparse.Namespace) -> int:
    config = build_config(args)
    try:
        from .gui import launch_gui

        launch_gui(config)
        return 0
    except PipelineError as error:
        return _error_exit(error, StructuredLogger(config.log_path), command="gui")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m alert_pipeline",
        description="Deterministic alert denoising and root-cause correlation pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate deterministic local samples")
    generate.add_argument("--output", default="samples")
    generate.add_argument("--seed", type=int, default=20261122)
    generate.set_defaults(handler=command_generate)

    run = subparsers.add_parser("run", help="load, process, persist and export conclusions")
    _add_config_arguments(run)
    run.add_argument("--mode", choices=("parallel", "serial"), default="parallel")
    run.add_argument(
        "--inject",
        choices=("none", "corrupt", "unsupported-version", "window-conflict"),
        default="none",
        help="inject one controlled anomaly before loading",
    )
    run.add_argument(
        "--crash-before-commit-batch",
        type=_nonnegative_int,
        default=None,
        help="simulate forced termination after batch work and before commit",
    )
    run.set_defaults(handler=command_run)

    verify = subparsers.add_parser("verify", help="independently verify current ledger")
    _add_config_arguments(verify)
    verify.set_defaults(handler=command_verify)

    recover = subparsers.add_parser("recover", help="replay committed batches and report unknown work")
    _add_config_arguments(recover)
    recover.set_defaults(handler=command_recover)

    diagnose = subparsers.add_parser("diagnose", help="show local environment diagnostics")
    diagnose.set_defaults(handler=command_diagnose)

    gui = subparsers.add_parser("gui", help="open the Tkinter diagnostic interface")
    _add_config_arguments(gui)
    gui.set_defaults(handler=command_gui)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except PipelineError as error:
        logger = None
        try:
            config = build_config(args)
            logger = StructuredLogger(config.log_path)
        except Exception:
            pass
        return _error_exit(error, logger, command=args.command)
    except KeyboardInterrupt:
        _print({"business_code": "E_STATE", "message": "interrupted by user"})
        return 130
    except BaseException as error:
        wrapped = PipelineError(
            "E_STATE",
            f"unexpected failure: {type(error).__name__}: {error}",
            details={"exception_type": type(error).__name__},
        )
        return _error_exit(wrapped, None, command=args.command)


if __name__ == "__main__":
    raise SystemExit(main())
