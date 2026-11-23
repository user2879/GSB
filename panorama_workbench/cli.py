from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .cache import StateStore, atomic_write_json
from .commands import read_command_stream
from .config import ConfigError, load_config
from .engine import Workbench
from .failures import Failure, failures_as_dicts
from .samples import generate_demo
from .snapshot import SnapshotStore
from .validation import load_inputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="panorama-workbench",
        description="Deterministic panoramic segmentation and mask post-processing workbench",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("generate-demo", help="generate deterministic local samples")
    demo.add_argument("--out", required=True, help="demo workspace directory")
    demo.add_argument("--images", type=int, default=4, help="number of generated images")

    preflight = subparsers.add_parser("preflight", help="validate environment and input files")
    _add_input_arguments(preflight)

    check = subparsers.add_parser("check", help="run the load/validation stage only")
    _add_input_arguments(check)
    check.add_argument("--report", help="optional JSON validation report")

    run = subparsers.add_parser("run", help="run one deterministic batch")
    _add_input_arguments(run)
    run.add_argument("--state-dir", required=True, help="durable state and snapshot directory")
    run.add_argument("--report", help="optional JSON diagnostic report path")
    run.add_argument("--stop-after-images", type=int, default=None)
    run.add_argument("--drain", action="store_true", help="repeat batches while work is pending")
    run.add_argument("--max-batches", type=int, default=32)

    inspect = subparsers.add_parser("inspect-state", help="inspect manifest and state metadata")
    inspect.add_argument("--state-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate-demo":
        paths = generate_demo(args.out, args.images)
        print(
            json.dumps(
                {
                    "commands": str(paths.commands),
                    "config": str(paths.config),
                    "replacements": str(paths.replacements),
                    "samples": str(paths.samples),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "preflight":
        return _preflight(args)
    if args.command == "check":
        return _check(args)
    if args.command == "run":
        return _run(args)
    if args.command == "inspect-state":
        return _inspect_state(args)
    parser.error("unknown command")
    return 2


def _add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--samples", required=True, help="sample directory containing PNG/JSON pairs")
    parser.add_argument("--config", required=True, help="segmentation configuration YAML")
    parser.add_argument("--commands", help="optional command stream JSON")


def _preflight(args: argparse.Namespace) -> int:
    checks: dict[str, Any] = {
        "cwd": str(Path.cwd()),
        "package_version": __version__,
        "python": sys.version.split()[0],
    }
    failures: list[Failure] = []
    if sys.version_info < (3, 10):
        failures.append(
            Failure(
                code="E_CONFIG_INVALID",
                item="environment",
                stage="load",
                message="Python 3.10 or newer is required",
            )
        )
    try:
        import numpy
        import PIL

        checks["numpy"] = numpy.__version__
        checks["pillow"] = PIL.__version__
    except Exception as exc:
        failures.append(
            Failure(
                code="E_CONFIG_INVALID",
                item="environment",
                stage="load",
                message="required package import failed",
                details={"error": repr(exc)},
            )
        )
    config = None
    try:
        config = load_config(args.config)
        checks["config_hash"] = config.hash
    except ConfigError as exc:
        failures.append(
            Failure(
                code="E_CONFIG_INVALID",
                item=str(Path(args.config)),
                stage="load",
                message=str(exc),
            )
        )
    samples_path = Path(args.samples)
    checks["samples_exists"] = samples_path.is_dir()
    if not samples_path.is_dir():
        failures.append(
            Failure(
                code="E_CONFIG_INVALID",
                item=str(samples_path),
                stage="load",
                message="sample directory does not exist",
            )
        )
    if args.commands:
        try:
            stream = read_command_stream(args.commands)
            checks["command_count"] = len(stream)
        except Exception as exc:
            failures.append(
                Failure(
                    code="E_CONFIG_INVALID",
                    item=str(Path(args.commands)),
                    stage="load",
                    message="command stream cannot be parsed",
                    details={"error": str(exc)},
                )
            )
    output = {
        "checks": checks,
        "failure_list": failures_as_dicts(failures),
        "ok": not failures,
        "project_root": str(Path(__file__).resolve().parent.parent),
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if not failures else 2


def _check(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(
            json.dumps(
                {
                    "failure_list": failures_as_dicts(
                        [
                            Failure(
                                code="E_CONFIG_INVALID",
                                item=str(Path(args.config)),
                                stage="load",
                                message=str(exc),
                            )
                        ]
                    ),
                    "ok": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    result = load_inputs(args.samples, config)
    payload = {
        "failure_list": failures_as_dicts(result.failures),
        "ok": not result.failures,
        "samples": result.summary,
    }
    if args.report:
        atomic_write_json(args.report, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not result.failures else 2


def _run(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        failure = Failure(
            code="E_CONFIG_INVALID",
            item=str(Path(args.config)),
            stage="load",
            message=str(exc),
        )
        print(json.dumps({"failure_list": failures_as_dicts([failure])}, indent=2))
        return 2
    command_list = None
    if args.commands:
        try:
            command_list = read_command_stream(args.commands)
        except Exception as exc:
            failure = Failure(
                code="E_CONFIG_INVALID",
                item=str(Path(args.commands)),
                stage="load",
                message="command stream cannot be parsed",
                details={"error": str(exc)},
            )
            print(json.dumps({"failure_list": failures_as_dicts([failure])}, indent=2))
            return 2
    stop_event = threading.Event()
    previous_handlers: dict[int, Any] = {}
    if threading.current_thread() is threading.main_thread():
        for signal_name in ("SIGINT", "SIGTERM"):
            signum = getattr(signal, signal_name, None)
            if signum is None:
                continue
            previous_handlers[signum] = signal.getsignal(signum)

            def handler(signum: int, frame: Any) -> None:
                stop_event.set()

            signal.signal(signum, handler)
    try:
        workbench = Workbench(args.state_dir)
        batches = 1
        result = workbench.run(
            samples_dir=args.samples,
            config=config,
            commands=command_list,
            command_base=Path(args.commands).resolve().parent if args.commands else None,
            report_path=args.report,
            stop_event=stop_event,
            stop_after_images=args.stop_after_images,
        )
        if args.drain:
            while (
                result.pending
                and not result.stopped
                and batches < max(1, args.max_batches)
            ):
                batches += 1
                result = workbench.run(
                    samples_dir=args.samples,
                    config=config,
                    commands=command_list,
                    command_base=Path(args.commands).resolve().parent if args.commands else None,
                    report_path=args.report,
                    stop_event=stop_event,
                )
        summary = {
            "batch_consumed": result.report["budget"]["batch_consumed"],
            "completed_images": result.report["state"]["completed_images"],
            "exit_code": result.exit_code,
            "failure_list": result.report["failure_list"],
            "pending_queue": list(result.pending),
            "recovery_code": result.recovery_code,
            "stopped": result.stopped,
            "tile_cache_hits": result.report["cache"]["tile_hits"],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return result.exit_code
    finally:
        if threading.current_thread() is threading.main_thread():
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)


def _inspect_state(args: argparse.Namespace) -> int:
    state, state_error = StateStore(args.state_dir).load()
    manifest, manifest_error = SnapshotStore(args.state_dir).load_manifest()
    payload = {
        "manifest": manifest,
        "manifest_error": manifest_error,
        "state": {
            "completed_images": state.get("completed_images", []) if state else [],
            "config_hash": state.get("config_hash") if state else None,
            "cumulative_consumed": state.get("cumulative_consumed", 0) if state else 0,
            "global_version": state.get("global_version", 0) if state else 0,
            "pending": state.get("pending", []) if state else [],
            "run_complete": state.get("run_complete", False) if state else False,
            "tile_cache_size": len(state.get("tile_cache", {})) if state else 0,
        },
        "state_error": state_error,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if state is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
