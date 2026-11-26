"""Offline command-line entry points for the local demo."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .blocks import parse_block_stream
from .canonical import canonical_json
from .config import ConfigError, load_config
from .engine import MonitorEngine
from .errors import ValidationError
from .ledger import LedgerWriter
from .synth import load_synth_spec, write_generated_stream


def _load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_commands(path: str | Path) -> list[dict[str, Any]]:
    data = _load_json(path)
    if isinstance(data, dict):
        commands = data.get("commands", [])
    else:
        commands = data
    if not isinstance(commands, list):
        raise ValidationError("E_CONFIG_INVALID", "commands must be an array", "commands")
    return commands


def _print(value: Any) -> None:
    print(canonical_json(value))


def _config_or_exit(path: str):
    try:
        return load_config(path)
    except ConfigError as exc:
        _print({"status": "ERROR", "failures": exc.as_list()})
        raise SystemExit(2)


def _blocks_or_exit(config, path: str):
    try:
        return parse_block_stream(_load_json(path), config)
    except ValidationError as exc:
        _print({"status": "ERROR", "failures": [exc.as_dict()]})
        raise SystemExit(2)


def action_synth(args: argparse.Namespace) -> int:
    config = _config_or_exit(args.config)
    try:
        spec = load_synth_spec(args.spec)
    except ValidationError as exc:
        _print({"status": "ERROR", "failures": [exc.as_dict()]})
        return 2
    document = write_generated_stream(config, spec, args.out)
    _print(
        {
            "status": "OK",
            "output": str(args.out),
            "block_count": len(document["blocks"]),
            "generator_fingerprint": document["generator_fingerprint"],
        }
    )
    return 0


def action_validate(args: argparse.Namespace) -> int:
    config = _config_or_exit(args.config)
    try:
        blocks = parse_block_stream(_load_json(args.blocks), config)
        commands = _load_commands(args.commands) if args.commands else []
    except ValidationError as exc:
        _print({"status": "ERROR", "failures": [exc.as_dict()]})
        return 2
    _print(
        {
            "status": "OK",
            "config_hash": config.config_hash,
            "block_count": len(blocks),
            "command_count": len(commands),
            "channel_ids": [channel.channel_id for channel in config.channels],
        }
    )
    return 0


def _summary(engine: MonitorEngine, reports: list[dict[str, Any]]) -> dict[str, Any]:
    state = engine.canonical_state()
    phases = state["phases"]
    verification = engine.verify_full_replay()
    return {
        "status": "OK" if verification["equal"] else "MISMATCH",
        "version": engine.version,
        "config_hash": engine.config.config_hash,
        "canonical_digest": state["canonical_digest"],
        "full_replay": verification,
        "blocks": len(engine.blocks),
        "retained_windows": phases["window"]["retained_window_count"],
        "evictions": len(phases["window"]["evictions"]),
        "observations": phases["observation"]["observation_count"],
        "assignments": len(phases["cluster"]["assignments"]),
        "deductions": len(phases["cluster"]["deduction_records"]),
        "merges": len(phases["cluster"]["merge_records"]),
        "annihilations": len(phases["cluster"]["annihilation_records"]),
        "segments": phases["segment"]["segment_count"],
        "scheduler_pending": len(phases["scheduler"]["pending"]),
        "scheduler_consumed_points": phases["scheduler"]["consumed_points_total"],
        "failures": state["failures"],
        "transactions": [
            {
                "command": report["command"],
                "status": report["status"],
                "failures": len(report["failures"]),
                "invalidated": len(report["invalidated_cache_keys"]),
                "reused": len(report["reused_cache_keys"]),
            }
            for report in reports
        ],
    }


def action_run(args: argparse.Namespace) -> int:
    config = _config_or_exit(args.config)
    blocks = _blocks_or_exit(config, args.blocks)
    commands = _load_commands(args.commands) if args.commands else []
    engine = MonitorEngine(
        config,
        blocks,
        snapshot_root=args.snapshot_dir,
        ledger_path=args.ledger_dir,
        auto_snapshot=args.snapshot_dir is not None,
    )
    if args.snapshot_dir:
        engine.commit_snapshot(force=True)
    reports: list[dict[str, Any]] = []
    if args.crash_after is not None:
        if args.crash_after < 0 or args.crash_after > len(commands):
            _print({"status": "ERROR", "message": "invalid crash-after index"})
            return 2
        for command in commands[: args.crash_after]:
            reports.append(engine.apply_command(command))
        engine.auto_snapshot = False
        for command in commands[args.crash_after :]:
            reports.append(engine.apply_command(command))
        if args.snapshot_dir:
            engine.snapshot_store.write_uncommitted_generation(
                version=engine.version,
                config_hash=engine.config.config_hash,
                stages=engine._snapshot_stages(),
            )
        print(
            canonical_json(
                {
                    "status": "SIMULATED_CRASH_EXIT_137",
                    "committed_pointer": engine.snapshot_store.current_pointer()
                    if engine.snapshot_store
                    else None,
                    "dirty_version": engine.version,
                }
            ),
            flush=True,
        )
        os._exit(137)

    reports = engine.apply_commands(commands)
    if args.state_out:
        Path(args.state_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.state_out).write_text(canonical_json(engine.canonical_state()) + "\n", encoding="utf-8")
    summary = _summary(engine, reports)
    _print(summary)
    return 0 if summary["status"] == "OK" else 1


def action_restore(args: argparse.Namespace) -> int:
    try:
        engine = MonitorEngine.restore(
            args.snapshot_dir,
            ledger_path=args.ledger_dir,
            auto_snapshot=False,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        _print({"status": "ERROR", "message": str(exc), "type": type(exc).__name__})
        return 2
    verification = engine.verify_full_replay()
    state = engine.canonical_state()
    _print(
        {
            "status": "OK" if verification["equal"] else "MISMATCH",
            "restored_version": engine.version,
            "config_hash": engine.config.config_hash,
            "canonical_digest": state["canonical_digest"],
            "full_replay": verification,
            "retained_windows": state["phases"]["window"]["retained_window_count"],
            "segments": state["phases"]["segment"]["segment_count"],
            "scheduler_pending": len(state["phases"]["scheduler"]["pending"]),
            "committed_pointer": engine.snapshot_store.current_pointer(),
        }
    )
    return 0 if verification["equal"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vibstream")
    sub = parser.add_subparsers(dest="action", required=True)

    synth = sub.add_parser("synth", help="generate deterministic sample blocks")
    synth.add_argument("--config", default="configs/app.json")
    synth.add_argument("--spec", default="samples/synth_spec.json")
    synth.add_argument("--out", default="samples/data_blocks.json")
    synth.set_defaults(func=action_synth)

    validate = sub.add_parser("validate", help="validate config, blocks, and command JSON")
    validate.add_argument("--config", default="configs/app.json")
    validate.add_argument("--blocks", default="samples/data_blocks.json")
    validate.add_argument("--commands", default=None)
    validate.set_defaults(func=action_validate)

    run = sub.add_parser("run", help="run commands and verify incremental/full equality")
    run.add_argument("--config", default="configs/app.json")
    run.add_argument("--blocks", default="samples/data_blocks.json")
    run.add_argument("--commands", default="samples/commands.json")
    run.add_argument("--snapshot-dir", default="ledger/snapshot")
    run.add_argument("--ledger-dir", default="ledger/demo")
    run.add_argument("--state-out", default=None)
    run.add_argument("--crash-after", type=int, default=None, help=argparse.SUPPRESS)
    run.set_defaults(func=action_run)

    restore = sub.add_parser("restore", help="restore and verify the last committed snapshot")
    restore.add_argument("--snapshot-dir", default="ledger/snapshot")
    restore.add_argument("--ledger-dir", default="ledger/restore")
    restore.set_defaults(func=action_restore)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
