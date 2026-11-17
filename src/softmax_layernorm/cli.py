from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
from typing import Any

import numpy as np

from .engine import Engine, compute_from_scratch
from .errors import BusinessError
from .jsonutil import atomic_write_json, sanitize_json
from .loader import load_commands_file, load_run


DEFAULT_COMMANDS = "samples/commands.json"
DEFAULT_CALLS = "samples/calls.json"
DEFAULT_CONFIG = "samples/config.json"
DEFAULT_TENSOR = "samples/tensor.json"


def _write_failure_diagnostic(output_dir: Path, category: str, error: BusinessError) -> Path:
    path = output_dir / "failures" / f"{category}.json"
    atomic_write_json(path, {"failures": [failure.to_dict() for failure in error.failures]})
    return path


def _run_once(
    calls_path: str,
    tensor_path: str | None,
    commands_path: str | None,
    config_path: str,
    output_dir: str | None,
    *,
    fresh: bool,
    stream_responses: bool = False,
    install_signal_handlers: bool = True,
) -> tuple[Engine, dict[str, Any], list[dict[str, Any]], int | None, str]:
    received_signal: list[int] = []
    previous_handlers: dict[int, Any] = {}
    engine_holder: dict[str, Engine] = {}
    stop_before_engine = False

    def handle_signal(signum: int, _frame: Any) -> None:
        nonlocal stop_before_engine
        received_signal.append(signum)
        if "engine" in engine_holder:
            engine_holder["engine"].request_stop("signal")
        else:
            stop_before_engine = True

    if install_signal_handlers and threading.current_thread() is threading.main_thread():
        for signum in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
            if signum is None:
                continue
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, handle_signal)

    try:
        loaded = load_run(calls_path, tensor_path, config_path)
        _, commands, digest = load_commands_file(commands_path)
        engine = Engine(loaded, commands, digest, output_dir=output_dir, fresh=fresh)
        engine_holder["engine"] = engine
        if stop_before_engine:
            engine.request_stop("signal")

        responses: list[dict[str, Any]] = []
        for command in commands[engine.next_command_index :]:
            if engine.stop_requested:
                break
            response = engine.handle_command(command)
            responses.append(response)
            if stream_responses:
                print(json.dumps(sanitize_json(response), ensure_ascii=False), flush=True)
        report = engine.finalize()
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)

    signal_name = ""
    if received_signal:
        try:
            signal_name = signal.Signals(received_signal[0]).name
        except ValueError:
            signal_name = str(received_signal[0])
    return engine, report, responses, (received_signal[0] if received_signal else None), signal_name


def _run_result_payload(
    engine: Engine,
    report: dict[str, Any],
    signal_number: int | None,
    signal_name: str,
) -> dict[str, Any]:
    return {
        "ok": True,
        "exit_code": 0,
        "signal": signal_name or None,
        "signal_number": signal_number,
        "snapshot_version": report["snapshot_version"],
        "global_version": report["global_version"],
        "completed_rows": report["summary"]["completed_rows"],
        "failure_count": len(report["failures"]),
        "business_code_counts": report["business_code_counts"],
        "report_path": report["report_path"],
        "failure_path": report["failure_path"],
        "summary": report["summary"],
    }


def _array_bytes_equal(left: np.ndarray, right: np.ndarray) -> bool:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.dtype != right_array.dtype or left_array.shape != right_array.shape:
        return False
    return left_array.tobytes(order="C") == right_array.tobytes(order="C")


def _write_full_artifacts(full_output: Path, full: Any) -> Path:
    full_output.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".full-", suffix=".npz", dir=full_output)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    arrays: dict[str, np.ndarray] = {"tensor": full.tensor}
    for index, state in enumerate(full.states):
        prefix = f"call_{index}"
        arrays[f"{prefix}_softmax"] = state.softmax_result
        arrays[f"{prefix}_layernorm"] = state.layernorm_result
        arrays[f"{prefix}_mean"] = state.layernorm_mean
        arrays[f"{prefix}_variance"] = state.layernorm_variance
        arrays[f"{prefix}_epsilon_applied"] = state.layernorm_epsilon_applied
    try:
        np.savez_compressed(temporary_path, **arrays)
        os.replace(temporary_path, full_output / "results.npz")
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    atomic_write_json(
        full_output / "summary.json",
        {
            "summary": full.summary,
            "failures": [failure.to_dict() for failure in full.failures],
            "calls": [call.to_dict() for call in full.calls],
        },
    )
    return full_output


def run_verification(
    calls_path: str,
    tensor_path: str | None,
    commands_path: str | None,
    config_path: str,
    incremental_output: Path,
    full_output: Path,
) -> tuple[bool, dict[str, Any]]:
    engine, report, _, _, _ = _run_once(
        calls_path,
        tensor_path,
        commands_path,
        config_path,
        str(incremental_output),
        fresh=True,
        install_signal_handlers=False,
    )
    final_calls, final_tensor = engine.export_final()
    full = compute_from_scratch(
        final_calls,
        final_tensor,
        engine.config,
        affected_rows=report["summary"].get("affected_rows", []),
    )
    _write_full_artifacts(full_output, full)

    inc_by_id = {state.spec.call_id: state for state in engine.states}
    checks: dict[str, Any] = {"calls": {}, "arrays_equal": True, "failures_equal": True, "summary_equal": True}
    for state in full.states:
        incremental_state = inc_by_id[state.spec.call_id]
        call_checks = {
            "softmax_equal": _array_bytes_equal(incremental_state.softmax_result, state.softmax_result),
            "layernorm_equal": _array_bytes_equal(incremental_state.layernorm_result, state.layernorm_result),
            "mean_equal": _array_bytes_equal(incremental_state.layernorm_mean, state.layernorm_mean),
            "variance_equal": _array_bytes_equal(incremental_state.layernorm_variance, state.layernorm_variance),
            "epsilon_marker_equal": _array_bytes_equal(
                incremental_state.layernorm_epsilon_applied, state.layernorm_epsilon_applied
            ),
        }
        checks["calls"][state.spec.call_id] = call_checks
        if not all(call_checks.values()):
            checks["arrays_equal"] = False

    inc_failures = sanitize_json(report.get("failures", []))
    full_failures = sanitize_json([failure.to_dict() for failure in full.failures])
    checks["failures_equal"] = inc_failures == full_failures
    checks["summary_equal"] = report.get("summary") == full.summary
    checks["active_business_codes_equal"] = report.get("active_business_code_counts", {}) == dict(
        sorted(
            {
                code: sum(1 for failure in full_failures if failure["code"] == code)
                for code in {item["code"] for item in full_failures}
            }.items()
        )
    )
    passed = all(
        bool(checks[key])
        for key in ("arrays_equal", "failures_equal", "summary_equal", "active_business_codes_equal")
    )
    verification = {
        "passed": passed,
        "incremental_report": str(incremental_output / "reports" / "latest.json"),
        "full_artifacts": str(full_output),
        "checks": checks,
        "incremental_summary": report.get("summary"),
        "full_summary": full.summary,
        "incremental_failures": inc_failures,
        "full_failures": full_failures,
        "final_snapshot_version": report.get("snapshot_version"),
    }
    return passed, verification


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="softmax-layernorm",
        description="Deterministic incremental Softmax / LayerNorm inference lab",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="execute the interactive command stream")
    run_parser.add_argument("--calls", default=DEFAULT_CALLS)
    run_parser.add_argument("--tensor", default=DEFAULT_TENSOR)
    run_parser.add_argument("--commands", default=DEFAULT_COMMANDS)
    run_parser.add_argument("--config", default=DEFAULT_CONFIG)
    run_parser.add_argument("--output", default=None)
    run_parser.add_argument("--fresh", action="store_true")
    run_parser.add_argument("--stream-responses", action="store_true")

    verify_parser = subparsers.add_parser(
        "verify", help="run incremental, then recompute the final state from scratch and compare"
    )
    verify_parser.add_argument("--calls", default=DEFAULT_CALLS)
    verify_parser.add_argument("--tensor", default=DEFAULT_TENSOR)
    verify_parser.add_argument("--commands", default=DEFAULT_COMMANDS)
    verify_parser.add_argument("--config", default=DEFAULT_CONFIG)
    verify_parser.add_argument("--incremental-output", default="output/verification/incremental")
    verify_parser.add_argument("--full-output", default="output/verification/full")
    verify_parser.add_argument("--report", default="output/verification/verification.json")

    demo_parser = subparsers.add_parser("demo", help="run verify with the bundled deterministic sample")
    demo_parser.add_argument("--output", default="output/demo")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        output_hint = Path(args.output or "output")
        try:
            engine, report, _, signal_number, signal_name = _run_once(
                args.calls,
                args.tensor,
                args.commands,
                args.config,
                args.output,
                fresh=args.fresh,
                stream_responses=args.stream_responses,
            )
        except BusinessError as exc:
            path = _write_failure_diagnostic(output_hint, "load_or_resume_failure", exc)
            print(
                json.dumps(
                    {
                        "ok": False,
                        "exit_code": 2,
                        "business_codes": sorted({failure.code for failure in exc.failures}),
                        "failure_path": str(path),
                        "failures": [failure.to_dict() for failure in exc.failures],
                    },
                    ensure_ascii=False,
                )
            )
            return 2
        payload = _run_result_payload(engine, report, signal_number, signal_name)
        print(json.dumps(sanitize_json(payload), ensure_ascii=False, indent=2))
        return 0

    if args.command in {"verify", "demo"}:
        if args.command == "demo":
            base = Path(args.output)
            incremental = base / "incremental"
            full_output = base / "full"
            report_path = base / "verification.json"
        else:
            incremental = Path(args.incremental_output)
            full_output = Path(args.full_output)
            report_path = Path(args.report)
        try:
            passed, verification = run_verification(
                getattr(args, "calls", DEFAULT_CALLS),
                getattr(args, "tensor", DEFAULT_TENSOR),
                getattr(args, "commands", DEFAULT_COMMANDS),
                getattr(args, "config", DEFAULT_CONFIG),
                incremental,
                full_output,
            )
        except BusinessError as exc:
            path = _write_failure_diagnostic(incremental.parent, "verification_failure", exc)
            print(
                json.dumps(
                    {
                        "ok": False,
                        "exit_code": 2,
                        "business_codes": sorted({failure.code for failure in exc.failures}),
                        "failure_path": str(path),
                    },
                    ensure_ascii=False,
                )
            )
            return 2
        atomic_write_json(report_path, verification)
        print(
            json.dumps(
                sanitize_json(
                    {
                        "ok": passed,
                        "exit_code": 0 if passed else 3,
                        "verification_report": str(report_path),
                        "checks": verification["checks"],
                        "incremental_report": verification["incremental_report"],
                        "full_artifacts": verification["full_artifacts"],
                    }
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if passed else 3

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
