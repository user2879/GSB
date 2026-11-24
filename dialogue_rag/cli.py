"""Command line entry point for deterministic dialogue RAG diagnostics."""

from __future__ import annotations

import argparse
import signal
import sys

from .codes import Code
from .diagnostics import write_json_atomic
from .engine import IncrementalEngine
from .model import load_commands, load_config
from .sample_data import write_samples
from .snapshot import SnapshotStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dialogue_rag")
    subparsers = parser.add_subparsers(dest="action", required=True)

    run_parser = subparsers.add_parser("run", help="run config and command stream")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--commands", required=True)
    run_parser.add_argument("--output", default="artifacts/diagnostic_report.json")
    run_parser.add_argument("--state-dir", default="artifacts/snapshot")
    run_parser.add_argument("--workers", type=int, default=None)
    run_parser.add_argument("--no-resume", action="store_true")

    sample_parser = subparsers.add_parser("generate-samples", help="generate deterministic JSON samples")
    sample_parser.add_argument("--out", default="samples")
    sample_parser.add_argument("--seed", type=int, default=23609)
    return parser


def _install_signal_handlers(engine: IncrementalEngine) -> None:
    def handler(signum, _frame):
        engine.request_stop()
        engine.recovery["last_signal"] = signum
        engine.recovery["stop_code"] = Code.STOPPING

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(signum, handler)
        except (ValueError, OSError):
            pass


def run_command(args: argparse.Namespace) -> int:
    state, config_failures, config_hash = load_config(args.config)
    commands, command_failures = load_commands(args.commands)
    if args.workers is not None:
        state.workers = max(1, args.workers)

    store = SnapshotStore(args.state_dir or "artifacts/state")
    snapshot = None if args.no_resume else store.load()
    recovery = {"mode": "fresh", "code": "I_FRESH_RUN"}
    if snapshot is not None and snapshot.manifest:
        if snapshot.manifest.get("config_hash") == config_hash:
            engine = IncrementalEngine(
                snapshot.state,
                snapshot.failures,
                cache=snapshot.cache,
                history=snapshot.history,
                pending=snapshot.pending,
                store=store,
                recovery=snapshot.recovery,
                command_cursor=int(snapshot.manifest.get("command_cursor", 0)),
            )
            engine.manifest = snapshot.manifest
            recovery = snapshot.recovery
        else:
            recovery = {
                "mode": "rebuild",
                "code": Code.SNAPSHOT_REBUILD,
                "reason": "config_hash_changed",
                "expected_config_hash": config_hash,
                "snapshot_config_hash": snapshot.manifest.get("config_hash"),
            }
            engine = IncrementalEngine(
                state,
                config_failures + command_failures,
                store=store,
                recovery=recovery,
            )
    elif snapshot is not None:
        recovery = snapshot.recovery
        engine = IncrementalEngine(
            state,
            config_failures + command_failures,
            store=store,
            recovery=recovery,
        )
    else:
        engine = IncrementalEngine(
            state,
            config_failures + command_failures,
            store=store,
            recovery=recovery,
        )

    _install_signal_handlers(engine)
    report = engine.run_commands(commands)
    report["config_path"] = args.config
    report["commands_path"] = args.commands
    report["output_path"] = args.output
    report["state_dir"] = args.state_dir
    write_json_atomic(args.output, report)
    print(args.output)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.action == "generate-samples":
        paths = write_samples(args.out, args.seed)
        for path in paths:
            print(path)
        return 0
    return run_command(args)


if __name__ == "__main__":
    sys.exit(main())
