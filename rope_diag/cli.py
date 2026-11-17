"""Command-line entry point for the RoPE extrapolation diagnostic workbench."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from pathlib import Path
from typing import Any

from .engine import DiagnosticEngine, EngineOptions
from .errors import (
    ConfigError,
    E_CONFIG_INVALID,
    E_CONFIG_MISMATCH,
    WorkbenchError,
)
from .util import atomic_write_json, hash_object, read_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deterministic RoPE extrapolation diagnostics with fault injection"
    )
    parser.add_argument("--config", required=True, help="position encoding configuration JSON")
    parser.add_argument("--samples", required=True, help="synthetic attention sample JSONL")
    parser.add_argument("--injections", required=True, help="fault injection manifest JSON")
    parser.add_argument(
        "--state-dir",
        default=".workbench_state",
        help="directory for atomic stage cache and run manifest",
    )
    parser.add_argument(
        "--report",
        default="artifacts/diagnostic_report.json",
        help="final diagnostic report path",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="require an interrupted snapshot to match version, config, injection and samples",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="disable stage cache reads/writes (diagnostic reproducibility mode)",
    )
    parser.add_argument(
        "--stop-after-stage",
        choices=("encoding", "scoring", "tracing", "sensitivity"),
        help="test hook: persist the completed stage then exit with code 0",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress the final stdout summary")
    return parser


def _failure_report(
    *,
    config_path: str,
    samples_path: str,
    injections_path: str,
    error: WorkbenchError,
) -> dict[str, Any]:
    config_hash: str | None = None
    injection_hash: str | None = None
    try:
        config_hash = hash_object(read_json(Path(config_path)))
    except (OSError, ValueError):
        pass
    try:
        injection_hash = hash_object(read_json(Path(injections_path)))
    except (OSError, ValueError):
        pass
    return {
        "schema_version": 1,
        "status": "failed",
        "config_hash": config_hash,
        "samples_hash": None,
        "injection_hash": injection_hash,
        "stratified_statistics": [],
        "error_traces": [],
        "sensitivity": {"rankings": [], "boundary_sensitive_samples": []},
        "injection_hits": [],
        "injection_hits_by_sample": [],
        "reuse_manifest": {"reused": [], "recomputed": [], "invalidated": [], "bypassed": []},
        "failures": [
            {
                "code": error.code,
                "scheme": None,
                "sample_id": None,
                "stage": error.stage or "configuration",
                "injection_id": None,
                "injection_index": None,
                "message": error.message,
                "details": error.details,
            }
        ],
        "report_contract": {
            "business_codes": [
                "E_CONFIG_INVALID",
                "E_CONFIG_MISMATCH",
                "E_NONFINITE_INPUT",
                "E_POSITION_OUT_OF_RANGE",
                "E_BASE_ZERO",
                "E_SCALE_INVALID",
            ]
        },
    }


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if sys.version_info < (3, 10):
        _print_json(
            {
                "status": "failed",
                "code": E_CONFIG_INVALID,
                "message": "Python 3.10 or newer is required",
            }
        )
        return 2

    stop_event = threading.Event()
    previous_handlers: dict[int, Any] = {}

    def handle_signal(signum: int, _frame: Any) -> None:
        stop_event.set()

    for signum in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
        if signum is None:
            continue
        try:
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, handle_signal)
        except (ValueError, OSError):
            pass

    try:
        engine = DiagnosticEngine(
            config_path=args.config,
            samples_path=args.samples,
            injections_path=args.injections,
            options=EngineOptions(
                state_dir=Path(args.state_dir),
                resume=args.resume,
                use_cache=not args.no_cache,
                stop_after_stage=args.stop_after_stage,
                stop_event=stop_event,
            ),
        )
        result = engine.run(Path(args.report))
    except WorkbenchError as exc:
        if exc.code == E_CONFIG_MISMATCH:
            _print_json(
                {
                    "status": "rejected",
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                    "exit_code": 3,
                }
            )
            return 3
        report = _failure_report(
            config_path=args.config,
            samples_path=args.samples,
            injections_path=args.injections,
            error=exc,
        )
        atomic_write_json(Path(args.report), report)
        _print_json(
            {
                "status": "failed",
                "code": exc.code,
                "message": exc.message,
                "failure_count": 1,
                "report": str(Path(args.report)),
                "exit_code": 2,
            }
        )
        return 2
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        error = ConfigError("failed to read input files", {"error": str(exc)})
        report = _failure_report(
            config_path=args.config,
            samples_path=args.samples,
            injections_path=args.injections,
            error=error,
        )
        atomic_write_json(Path(args.report), report)
        _print_json(
            {
                "status": "failed",
                "code": error.code,
                "message": error.message,
                "failure_count": 1,
                "report": str(Path(args.report)),
                "exit_code": 2,
            }
        )
        return 2
    finally:
        for signum, handler in previous_handlers.items():
            try:
                signal.signal(signum, handler)
            except (ValueError, OSError):
                pass

    if not args.quiet:
        if result.get("status") == "interrupted":
            _print_json(result)
        else:
            _print_json(
                {
                    "status": result.get("status"),
                    "report": str(Path(args.report)),
                    "failure_count": len(result.get("failures", [])),
                    "injection_hit_count": len(result.get("injection_hits", [])),
                    "diagnostic_digest": result.get("diagnostic_digest"),
                }
            )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
