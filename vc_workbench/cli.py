"""Command line interface for the deterministic voice-conversion workbench."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any

from .config import ConfigError, load_config
from .engine import EngineOptions, WorkbenchEngine
from .plan import CommandError, SessionPlan
from .state import SnapshotError, inspect_current
from .util import atomic_write_json, sha256_bytes


def _write_wav_atomic(path: Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".wav.tmp", dir=path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        with wave.open(str(temp_path), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16000)
            writer.writeframes(pcm)
        with temp_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _run(args: argparse.Namespace) -> int:
    samples = Path(args.samples).resolve()
    config_path = Path(args.config).resolve()
    commands_path = Path(args.commands).resolve()
    state_root = Path(args.state).resolve()
    output_path = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    try:
        config = load_config(config_path)
        plan = SessionPlan.load(commands_path)
    except (ConfigError, CommandError) as exc:
        print(f"E_CONFIG_INVALID: {exc}", file=sys.stderr)
        return 2
    options = EngineOptions(
        fresh=bool(args.fresh),
        stop_after_command=args.stop_after_command,
        crash_after_command=args.crash_after_command,
        crash_on_checkpoint=args.crash_on_checkpoint,
        output_path=output_path,
        report_path=report_path,
    )
    try:
        engine = WorkbenchEngine(
            samples_dir=samples,
            config=config,
            plan=plan,
            state_root=state_root,
            options=options,
        )
    except Exception as exc:
        print(f"E_ENGINE_INIT: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    for signal_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, signal_name, None)
        if sig is not None:
            try:
                signal.signal(sig, engine.request_stop)
            except (OSError, ValueError):
                pass
    try:
        report = engine.run()
        _write_wav_atomic(output_path, engine.assembly_pcm)
        report["output"]["sha256"] = sha256_bytes(engine.assembly_pcm)
        report["output"]["sample_count"] = len(engine.assembly_pcm) // 2
        report["output"]["written"] = True
        atomic_write_json(report_path, report)
    except SnapshotError as exc:
        print(f"E_SNAPSHOT_STALE: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:
        print(f"E_INTERNAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    if engine.recovered:
        print("RECOVERY_POINT " + json.dumps(report["recovery"]["recovery_point"], ensure_ascii=False, sort_keys=True))
    for event in report.get("startup_events", []):
        print(f"STARTUP_EVENT {json.dumps(event, ensure_ascii=False, sort_keys=True)}")
    summary = {
        "run_status": report["run_status"],
        "command_cursor": report["command_cursor"],
        "command_count": report["command_count"],
        "global_version": report["global_version"],
        "exit_code": report["exit_code"],
        "failures": len(report["failures"]),
        "sample_count": report["output"]["sample_count"],
        "output_sha256": report["output"]["sha256"],
        "report": str(report_path),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return int(report["exit_code"])


def _inspect(args: argparse.Namespace) -> int:
    result = inspect_current(Path(args.state).resolve())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result.get("recoverable") else 4


def _generate(args: argparse.Namespace) -> int:
    try:
        from tools.generate_samples import generate_samples
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from tools.generate_samples import generate_samples

    result = generate_samples(Path(args.dir).resolve(), force=bool(args.force))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vc-workbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="execute or resume a conversion session")
    run_parser.add_argument("--samples", default="samples")
    run_parser.add_argument("--config", default="samples/config.json")
    run_parser.add_argument("--commands", default="samples/commands.json")
    run_parser.add_argument("--state", default="workbench_state")
    run_parser.add_argument("--output", default="converted.wav")
    run_parser.add_argument("--report", default="run_report.json")
    run_parser.add_argument("--fresh", action="store_true")
    run_parser.add_argument("--stop-after-command", type=int, default=None)
    run_parser.add_argument("--crash-after-command", type=int, default=None, help=argparse.SUPPRESS)
    run_parser.add_argument("--crash-on-checkpoint", type=int, default=None, help=argparse.SUPPRESS)
    run_parser.set_defaults(func=_run)

    inspect_parser = subparsers.add_parser("inspect", help="validate and print the current recovery point")
    inspect_parser.add_argument("--state", default="workbench_state")
    inspect_parser.set_defaults(func=_inspect)

    generate_parser = subparsers.add_parser("generate", help="create the fixed-seed local sample set")
    generate_parser.add_argument("--dir", default="samples")
    generate_parser.add_argument("--force", action="store_true")
    generate_parser.set_defaults(func=_generate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
