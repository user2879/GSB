"""Command-line entry points for run, verify, and snapshot inspection."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path
from typing import Any

from .config import load_command_stream, load_render_config
from .errors import E_SNAPSHOT_STALE, RenderError
from .session import SessionRunner
from .snapshot import SnapshotManager, write_top_level_diagnostics
from .util import atomic_write_json, read_json
from .verify import verify


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _exit_code_for_error(error: RenderError) -> int:
    return 3 if error.code == E_SNAPSHOT_STALE else 2


def _write_failure_report(workdir: Path, status: str, error: RenderError) -> Path:
    payload = {
        "schema_version": 1,
        "status": status,
        "business_code": error.code,
        "error": error.as_dict(),
        "workdir": str(workdir.resolve()),
    }
    path = workdir.resolve() / "reports" / "run_failure.json"
    atomic_write_json(path, payload)
    return path


def _load_inputs(config_path: str, commands_path: str) -> tuple[Any, list[dict[str, Any]], str]:
    config = load_render_config(config_path)
    commands, command_hash = load_command_stream(commands_path)
    return config, commands, command_hash


def _run_command(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir).resolve()
    try:
        config, commands, command_hash = _load_inputs(args.config, args.commands)
    except RenderError as exc:
        path = _write_failure_report(workdir, "failed", exc)
        _print({"status": "failed", "business_code": exc.code, "error": exc.as_dict(), "report": str(path)})
        return _exit_code_for_error(exc)

    stop_signal: dict[str, int | None] = {"number": None}

    def handle_signal(signum: int, frame: Any) -> None:
        if stop_signal["number"] is None:
            stop_signal["number"] = signum

    previous_handlers: dict[int, Any] = {}
    for signal_name in ("SIGINT", "SIGTERM"):
        signum = getattr(signal, signal_name, None)
        if signum is None:
            continue
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, handle_signal)

    try:
        session = SessionRunner(
            config=config,
            commands=commands,
            command_stream_hash=command_hash,
            workdir=workdir,
            resume=bool(args.resume),
            expected_version=args.expected_version,
        )
        if args.hard_crash_after is not None:
            session.run(
                stop_after=args.hard_crash_after,
                commit_periodic=True,
                commit_final=False,
                should_stop=lambda: stop_signal["number"] is not None,
            )
            sys.stderr.write(
                f"simulated hard termination after {args.hard_crash_after} commands; "
                "only committed snapshots are reusable\n"
            )
            sys.stderr.flush()
            os._exit(137)

        run_info = session.run(
            stop_after=args.stop_after,
            commit_periodic=True,
            commit_final=True,
            should_stop=lambda: stop_signal["number"] is not None,
        )
        status = "interrupted" if stop_signal["number"] is not None else "ok"
        diagnostics = session.diagnostics(status=status)
        diagnostics["run"] = run_info
        diagnostics["signal_number"] = stop_signal["number"]
        report_path = write_top_level_diagnostics(workdir, diagnostics)
        current = session.snapshot_manager.current_pointer()
        output_root = None
        if current is not None:
            generation_dir = workdir / "snapshots" / str(current["generation"])
            manifest_path = generation_dir / "manifest.json"
            if manifest_path.exists():
                manifest = read_json(manifest_path)
                output_root = str(generation_dir / "outputs")
                diagnostics["final_output_root"] = output_root
                diagnostics["snapshot_manifest"] = str(manifest_path)
                atomic_write_json(report_path, diagnostics)
        summary = {
            "status": status,
            "workdir": str(workdir),
            "diagnostics": str(report_path),
            "snapshot_pointer": current,
            "outputs": output_root,
            "processed_commands": run_info["processed"],
            "final_cursor": run_info["final_cursor"],
            "immediate_pending_responses": diagnostics["immediate_pending_responses"],
            "failures": [
                {"command_id": item.get("command_id"), "code": item.get("code")}
                for item in diagnostics["failures"]
            ],
            "final": diagnostics["final"],
        }
        _print(summary)
        if stop_signal["number"] is not None:
            return 128 + int(stop_signal["number"])
        if args.stop_after is not None:
            return 130
        return 0
    except RenderError as exc:
        path = _write_failure_report(workdir, "failed", exc)
        _print({"status": "failed", "business_code": exc.code, "error": exc.as_dict(), "report": str(path)})
        return _exit_code_for_error(exc)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def _verify_command(args: argparse.Namespace) -> int:
    try:
        report = verify(
            config_path=args.config,
            commands_path=args.commands,
            workdir=args.workdir,
            force=bool(args.force),
        )
    except RenderError as exc:
        _print({"status": "failed", "business_code": exc.code, "error": exc.as_dict()})
        return _exit_code_for_error(exc)
    summary = {
        "status": report["status"],
        "checks": report["checks"],
        "latency": {
            "budget_ms": report["latency"]["budget_ms"],
            "max_response_latency_ms": report["latency"]["max_response_latency_ms"],
            "pending_count": report["latency"]["pending_count"],
        },
        "incremental_vs_full": report["incremental_vs_full"],
        "recovery": {
            "resumed_from_cursor": report["recovery"]["resumed_from_cursor"],
            "final_state_equal": report["recovery"]["final_state_equal"],
            "output_equality": report["recovery"]["output_equality"],
        },
        "stale": report["stale"],
        "report": str(Path(args.workdir).resolve() / "reports" / "verification.json"),
    }
    _print(summary)
    return 0 if report["status"] == "ok" else 1


def _inspect_command(args: argparse.Namespace) -> int:
    manager = SnapshotManager(args.workdir)
    pointer = manager.current_pointer()
    if pointer is None:
        _print({"status": "empty", "workdir": str(Path(args.workdir).resolve())})
        return 1
    generation_dir = Path(args.workdir).resolve() / "snapshots" / str(pointer["generation"])
    manifest = read_json(generation_dir / "manifest.json")
    diagnostics_path = Path(args.workdir).resolve() / "reports" / "diagnostics.json"
    diagnostics = read_json(diagnostics_path) if diagnostics_path.exists() else None
    _print(
        {
            "status": "ok",
            "pointer": pointer,
            "manifest": manifest,
            "diagnostics_path": str(diagnostics_path) if diagnostics is not None else None,
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fixedvideo",
        description="Deterministic local text/image-to-video multi-stage renderer",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="execute a local command stream with snapshots")
    run.add_argument("--config", required=True, help="render configuration JSON")
    run.add_argument("--commands", required=True, help="interactive command stream JSON")
    run.add_argument("--workdir", required=True, help="artifact, cache, and snapshot workspace")
    run.add_argument("--resume", action="store_true", help="resume from the current committed snapshot")
    run.add_argument("--fresh", action="store_true", help="start a fresh generation (default unless --resume)")
    run.add_argument("--expected-version", type=int, default=None, help="require this committed global version")
    run.add_argument("--stop-after", type=int, default=None, help="gracefully stop after N new commands and commit")
    run.add_argument("--hard-crash-after", type=int, default=None, help="finish N commands, commit only periodic snapshots, then exit 137")
    run.set_defaults(handler=_run_command)

    verify_parser = subparsers.add_parser("verify", help="run incremental/full/recovery/stale acceptance checks")
    verify_parser.add_argument("--config", required=True, help="render configuration JSON")
    verify_parser.add_argument("--commands", required=True, help="interactive command stream JSON")
    verify_parser.add_argument("--workdir", required=True, help="verification workspace")
    verify_parser.add_argument("--force", action="store_true", help="replace an existing verification workspace")
    verify_parser.set_defaults(handler=_verify_command)

    inspect_parser = subparsers.add_parser("inspect", help="print the current committed snapshot manifest")
    inspect_parser.add_argument("--workdir", required=True, help="workspace containing snapshots/CURRENT.json")
    inspect_parser.set_defaults(handler=_inspect_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
