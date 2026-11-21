from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any

from .audio import scan_wavs
from .config import canonical_json, load_config
from .engine import IncrementalEngine
from .errors import E_SNAPSHOT_STALE, BusinessError
from .report import build_report, write_fatal_report, write_report
from .segmentation import build_document
from .snapshot import SnapshotManager
from .synth import generate_demo


def load_commands(path: str | Path) -> list[dict[str, Any]]:
    command_path = Path(path)
    try:
        value = json.loads(command_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BusinessError("E_COMMAND_INVALID", f"cannot read command stream: {exc}") from exc
    if isinstance(value, dict):
        value = value.get("commands")
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise BusinessError("E_COMMAND_INVALID", "command stream must be an array of objects")
    return value


def _install_signal_handlers() -> tuple[dict[str, bool], dict[int, Any]]:
    state = {"stop": False}
    previous: dict[int, Any] = {}

    def handler(signum: int, frame: Any) -> None:
        state["stop"] = True

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, handler)
    return state, previous


def _restore_signal_handlers(previous: dict[int, Any]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def run_analysis(
    *,
    samples_dir: str | Path,
    config_path: str | Path,
    commands_path: str | Path,
    output_dir: str | Path,
    fresh: bool = False,
) -> tuple[int, dict[str, Any]]:
    started = time.perf_counter()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    commands = load_commands(commands_path)
    commands_base_dir = Path(commands_path).resolve().parent
    manager = SnapshotManager(output)
    manager.cleanup_staging()
    resume_info: dict[str, Any] = {"attempted": not fresh, "restored": False, "snapshot": None}
    restored = None
    if not fresh:
        try:
            restored = manager.load(config, commands)
            if restored is not None:
                resume_info = {
                    "attempted": True,
                    "restored": True,
                    "snapshot": restored.path.name,
                    "command_cursor": restored.command_cursor,
                    "global_version": restored.manifest.get("global_version"),
                    "config_hash": restored.manifest.get("config_hash"),
                }
        except BusinessError as exc:
            if exc.code == E_SNAPSHOT_STALE:
                write_fatal_report(output, code=exc.code, message=exc.message, details=exc.details)
                resume_info.update({"restored": False, "error": exc.as_dict()})
                raise
    signal_state, previous_handlers = _install_signal_handlers()
    engine: IncrementalEngine | None = None
    command_events: list[dict[str, Any]] = []
    try:
        if restored is not None:
            engine = IncrementalEngine(
                config,
                restored.document,
                commands,
                output,
                overrides=restored.overrides,
                failures=restored.failures,
                command_cursor=restored.command_cursor,
                global_version=int(restored.manifest.get("global_version", restored.command_cursor)),
                restored=restored,
                commands_base_dir=commands_base_dir,
                auto_initialize=False,
            )
            engine.initialize(commit=False)
        else:
            scan = scan_wavs(samples_dir)
            document, segmentation_summaries, failures = build_document(scan, config)
            engine = IncrementalEngine(
                config,
                document,
                commands,
                output,
                failures=failures,
                commands_base_dir=commands_base_dir,
                auto_initialize=True,
            )
            engine.last_semantic["segmentation_summaries"] = segmentation_summaries
        remaining = commands[engine.command_cursor :]
        for command in remaining:
            if signal_state["stop"]:
                break
            preview = engine.submit(command)
            event = preview
            if preview["status"] == "pending":
                engine.wait_for_idle()
                completed = engine.jobs[-1].result
                if completed is not None:
                    event = dict(completed)
                    event["budget_response_code"] = preview["code"]
                    event["budget_preview"] = preview["preview"]
            command_events.append(event)
            if signal_state["stop"]:
                break
        engine.join()
        if signal_state["stop"]:
            engine.commit_snapshot()
        elif engine.last_commit_cursor != engine.command_cursor:
            engine.commit_snapshot()
        semantic = engine.last_semantic
        if semantic is None:
            semantic = engine.initialize(commit=False)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        report = build_report(
            semantic=semantic,
            config_hash=config.config_hash,
            command_events=command_events,
            snapshot=engine.last_snapshot,
            resume_info=resume_info,
            stop_requested=signal_state["stop"],
            elapsed_ms=elapsed_ms,
        )
        report["commands_submitted"] = engine.command_cursor
        report["commands_total"] = len(commands)
        report_path = write_report(output, report)
        summary = {
            "status": report["status"],
            "report": str(report_path),
            "snapshot": engine.last_snapshot.get("snapshot") if engine.last_snapshot else None,
            "command_cursor": engine.command_cursor,
            "segments": report["counts"]["segments"],
            "speakers": report["counts"]["speakers"],
            "incremental_full_equal": report["incremental_full_equal"],
            "failure_codes": report["failure_codes"],
            "signal_received": signal_state["stop"],
        }
        exit_code = 0 if report["incremental_full_equal"] is not False else 3
        return exit_code, summary
    finally:
        if engine is not None:
            engine.close()
        _restore_signal_handlers(previous_handlers)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="timbre-lab", description="Editable speaker-timbre analysis pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    synth = subparsers.add_parser("synth", help="generate deterministic demo WAVs, config, commands, and anomalies")
    synth.add_argument("--out", default="examples/demo")
    synth.add_argument("--seed", type=int, default=20261121)

    run = subparsers.add_parser("run", help="run or resume the analysis pipeline")
    run.add_argument("--samples", required=True)
    run.add_argument("--config", required=True)
    run.add_argument("--commands", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--fresh", action="store_true", help="ignore CURRENT snapshot and rebuild from the input audio")

    demo = subparsers.add_parser("demo", help="generate a demo workspace and run it end-to-end")
    demo.add_argument("--out", default="artifacts/demo")
    demo.add_argument("--seed", type=int, default=20261121)
    demo.add_argument("--fresh", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "synth":
            paths = generate_demo(args.out, args.seed)
            print(
                canonical_json(
                    {
                        "status": "generated",
                        **{key: str(value) for key, value in paths.items()},
                    }
                )
            )
            return 0
        if args.command == "demo":
            root = Path(args.out)
            generated = generate_demo(root / "input", args.seed)
            code, summary = run_analysis(
                samples_dir=generated["samples"],
                config_path=generated["config"],
                commands_path=generated["commands"],
                output_dir=root / "run",
                fresh=args.fresh,
            )
            print(canonical_json(summary))
            return code
        if args.command == "run":
            code, summary = run_analysis(
                samples_dir=args.samples,
                config_path=args.config,
                commands_path=args.commands,
                output_dir=args.output,
                fresh=args.fresh,
            )
            print(canonical_json(summary))
            return code
    except BusinessError as exc:
        output = args.output if getattr(args, "output", None) else (Path(args.out) / "run" if getattr(args, "out", None) else ".")
        write_fatal_report(output, code=exc.code, message=exc.message, details=exc.details)
        print(canonical_json({"status": "failed", "code": exc.code, "message": exc.message, "details": exc.details}), file=sys.stderr)
        return 2
    except Exception as exc:
        output = getattr(args, "output", None) or "."
        write_fatal_report(output, code="E_INTERNAL", message=str(exc))
        print(canonical_json({"status": "failed", "code": "E_INTERNAL", "message": str(exc)}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
