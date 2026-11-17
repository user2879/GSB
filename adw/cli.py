"""Command-line interactive diagnostic workbench."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from .commands import HELP_TEXT, apply_command
from .config import ConfigBundle, load_config_bundle
from .data import generate_dataset
from .engine import WorkbenchEngine
from .errors import ConfigValidationError, sort_failures
from .injection import load_injection_bundle, require_valid_injection


class _ShutdownRequested(Exception):
    """Raised by the main-thread signal handler to unblock stdin/condition waits."""


def _emit(event: dict[str, Any]) -> None:
    print(json.dumps(event, ensure_ascii=False, separators=(",", ":"), allow_nan=False), flush=True)


def _failed_bundle(bundle: ConfigBundle | None, failures: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "event": "config_failed",
        "status": "failed",
        "code": "E_CONFIG_INVALID",
        "failures": sort_failures(failures),
        "config_path": str(bundle.config_path) if bundle is not None else None,
    }


def _read_commands(script: str | None) -> Iterable[str]:
    if script is not None:
        with open(script, "r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text and not text.startswith("#"):
                    yield text
        return
    if not sys.stdin.isatty():
        for line in sys.stdin:
            text = line.strip()
            if text and not text.startswith("#"):
                yield text
        return
    while True:
        print("adw> ", end="", file=sys.stderr, flush=True)
        try:
            line = sys.stdin.readline()
        except KeyboardInterrupt:
            raise
        if line == "":
            return
        text = line.strip()
        if text and not text.startswith("#"):
            yield text


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic anomaly diagnostic workbench")
    parser.add_argument("--config", default="config.yaml", help="path to workbench YAML config")
    parser.add_argument("--root", default=None, help="base directory for relative config paths")
    parser.add_argument("--injection", default=None, help="separate injection YAML file")
    parser.add_argument("--script", default=None, help="run a non-interactive command script")
    parser.add_argument("--no-start", action="store_true", help="load only; do not start the scheduler")
    parser.add_argument("--version", action="version", version="adw 1.0.0")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _make_parser().parse_args(argv)
    bundle = load_config_bundle(args.config, root=args.root)
    if bundle.failures:
        _emit(_failed_bundle(bundle, bundle.failures))
        return 2
    injection_items: list[dict[str, Any]] = []
    injection_labels: dict[str, Any] = {}
    if args.injection:
        try:
            injection_bundle = require_valid_injection(load_injection_bundle(args.injection, root=bundle.root))
            injection_items = list(injection_bundle.data.get("items", []))
            injection_labels = {
                "report_path": injection_bundle.data.get("report_path"),
                "root": str(injection_bundle.root),
                "path": str(injection_bundle.path),
            }
        except ConfigValidationError as exc:
            _emit(
                {
                    "event": "injection_config_failed",
                    "status": "failed",
                    "code": "E_CONFIG_INVALID",
                    "failures": exc.failures,
                    "injection_path": str(Path(args.injection).resolve()),
                }
            )
            return 2
    try:
        dataset = generate_dataset(bundle.description)
    except ConfigValidationError as exc:
        _emit(_failed_bundle(bundle, exc.failures))
        return 2
    engine = WorkbenchEngine(
        dataset,
        bundle.config,
        root=bundle.root,
        injection_items=injection_items,
        injection_labels=injection_labels,
    )
    stopped_by_signal = False

    def handle_signal(signum: int, frame: Any) -> None:
        nonlocal stopped_by_signal
        stopped_by_signal = True
        engine.request_shutdown()
        raise _ShutdownRequested(f"received signal {signum}")

    for signal_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        candidate = getattr(signal, signal_name, None)
        if candidate is not None:
            try:
                signal.signal(candidate, handle_signal)
            except (OSError, ValueError):
                pass

    if not args.no_start:
        engine.start()
    _emit(
        {
            "event": "ready",
            "status": "ok",
            "config_path": str(bundle.config_path),
            "data_fingerprint": dataset.fingerprint,
            "sample_count": len(dataset.sample_ids),
            "latency_budget_ms": bundle.config["interaction"]["latency_budget_ms"],
            "concurrency": bundle.config["interaction"]["concurrency"],
            "recovery": engine.recovery,
            "injection_items": len(injection_items),
            "help": HELP_TEXT.splitlines(),
        }
    )
    exit_code = 0
    try:
        for line in _read_commands(args.script):
            if stopped_by_signal or engine.graceful_stopped:
                break
            command = line.split(None, 1)[0].lower()
            if command in ("quit", "exit"):
                # Quit flushes the latest generation before closing; it is not a config edit.
                engine.wait_current()
                paths = engine.write_report() if engine.completed_generation else {}
                engine.request_shutdown()
                engine.join_worker()
                _emit({"event": "exit", "status": "ok", "code": 0, "paths": paths})
                return 0
            if command == "help":
                _emit({"status": "ok", "help": HELP_TEXT.splitlines()})
                continue
            if command == "status":
                _emit(engine.status_response())
                continue
            if command in ("result", "results"):
                _emit(engine.result_response())
                continue
            if command == "settle":
                _emit(engine.settle_response())
                continue
            if command == "report":
                _emit(engine.report_response())
                continue
            if command == "recover":
                _emit(engine.recover_response())
                continue
            if command == "compare" and line.lower().startswith("compare full"):
                comparison = engine.compare_full()
                if comparison.get("status") == "ok":
                    try:
                        engine.write_report()
                    except OSError:
                        pass
                _emit(comparison)
                continue
            try:
                result = apply_command(engine.current_config, bundle.description, line)
            except ConfigValidationError as exc:
                _emit(engine.invalid_command_response(line, exc.failures))
                continue
            if result.message in ("quit", "exit", "help"):
                continue
            response = engine.submit_config(
                result.config,
                affected_algorithms=result.affected_algorithms,
                changed_fields=result.changed_fields,
                needs_run=result.needs_run,
                source_command=line,
            )
            response["event"] = "command"
            response["message_detail"] = result.message
            _emit(response)
            if stopped_by_signal or engine.graceful_stopped:
                break
    except (KeyboardInterrupt, _ShutdownRequested):
        stopped_by_signal = True
        engine.request_shutdown()
    except BrokenPipeError:
        exit_code = 0
    try:
        engine.wait_current()
    except (KeyboardInterrupt, _ShutdownRequested):
        stopped_by_signal = True
        engine.request_shutdown()
    engine.request_shutdown()
    engine.join_worker()
    try:
        paths = engine.write_report() if engine.completed_generation else {}
    except OSError as exc:
        paths = {"error": str(exc)}
    if engine.last_error and not engine.graceful_stopped:
        exit_code = 1
    _emit(
        {
            "event": "shutdown",
            "status": "stopped" if stopped_by_signal else "ok",
            "code": exit_code,
            "graceful": engine.graceful_stopped,
            "paths": paths,
        }
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
