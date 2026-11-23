"""Command line interface for the deterministic inpainting pyramid workbench."""

from __future__ import annotations

import argparse
import json
import shutil
import signal
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import apply_command, load_commands, load_config
from .engine import (
    ENGINE_VERSION,
    IncrementalEngine,
    compare_engines,
)
from .errors import (
    E_CONFIG_INVALID,
    E_STATE_EXISTS,
    CommandError,
    ConfigError,
    StateError,
    WorkbenchError,
)
from .model import CommandStream, ConfigBundle
from .pyramid import UPSAMPLE_METADATA
from .samplegen import generate_sample
from .snapshot import SnapshotManager
from .util import atomic_write_json


@dataclass
class StopRequest:
    requested: bool = False
    signum: int | None = None

    def install(self) -> None:
        def handler(signum: int, _frame: Any) -> None:
            self.requested = True
            self.signum = signum

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)


def _json_print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _base_report(bundle: ConfigBundle, stream: CommandStream, mode: str) -> dict[str, Any]:
    return {
        "schema": "ipw-report-v1",
        "engine_version": ENGINE_VERSION,
        "mode": mode,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_config_hash": bundle.base_config_hash,
        "commands_hash": stream.commands_hash,
        "source": bundle.source_name,
        "style": bundle.style_name,
        "source_fingerprint": bundle.source_fingerprint,
        "style_fingerprint": bundle.style_fingerprint,
        "level_sizes": [list(size) for size in bundle.spec.sizes],
        "tile_size": bundle.spec.tile_size,
        "fraction_bits": bundle.spec.fraction_bits,
        "fixed_point_rounding": "half_up",
        "upsample": UPSAMPLE_METADATA,
        "thread_limit": bundle.threads,
        "tile_budget": bundle.recompute_tile_budget,
        "commands": [],
        "failures": [],
    }


def _finalize_report(engine: IncrementalEngine, report: dict[str, Any]) -> None:
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["effective_config_hash"] = engine.bundle.effective_config_hash(engine.mutable)
    report["completed_command_cursor"] = engine.cursor
    report["layer_timings_seconds"] = {
        str(level): round(value, 9) for level, value in sorted(engine.layer_timings.items())
    }
    report["layer_summaries"] = engine.layer_summary()
    report["pending_tiles"] = [list(key) for key in sorted(engine.dirty_tiles)]
    report["failures"] = engine.failures
    report["concurrency_peak"] = engine.concurrency_peak
    report["thread_limit_respected"] = engine.concurrency_peak <= engine.bundle.threads
    report["composite_stale"] = engine.composite_stale
    if engine.composite is not None:
        report["composite"] = engine.composite.to_metadata()
    else:
        report["composite"] = None
    report["budget_ledger"] = [
        item
        for command in report.get("commands", [])
        for item in command.get("schedule", {}).get("budget", {}).get("ledger", [])
    ]
    report["budget_decisions"] = [
        {
            "index": command.get("index"),
            "id": command.get("id"),
            "status": command.get("schedule", {}).get("status"),
            "budget": command.get("schedule", {}).get("budget"),
            "pending_tiles": command.get("schedule", {}).get("pending_tiles", []),
        }
        for command in report.get("commands", [])
        if "schedule" in command
    ]
    report["diagnostic_result"] = (
        "OK" if not engine.dirty_tiles else "E_TILE_BUDGET_PENDING"
    )


def _validate_commands(bundle: ConfigBundle, stream: CommandStream) -> list[dict[str, Any]]:
    mutable = bundle.new_mutable()
    records: list[dict[str, Any]] = []
    for index, command in enumerate(stream.commands):
        try:
            normalized, change = apply_command(mutable, command, bundle.spec)
            records.append(
                {
                    "index": index,
                    "id": command.get("id") if isinstance(command, dict) else None,
                    "type": command.get("type") if isinstance(command, dict) else None,
                    "status": "valid",
                    "normalized": normalized,
                    "changed": bool(change.changed),
                }
            )
        except CommandError as exc:
            records.append(
                {
                    "index": index,
                    "id": command.get("id") if isinstance(command, dict) else None,
                    "type": command.get("type") if isinstance(command, dict) else None,
                    "status": "invalid_skipped",
                    "error": exc.as_dict(),
                }
            )
    return records


def command_validate(args: argparse.Namespace) -> int:
    bundle = load_config(args.config, args.images)
    stream = load_commands(args.commands)
    records = _validate_commands(bundle, stream)
    summary = {
        "status": "ok",
        "base_config_hash": bundle.base_config_hash,
        "commands_hash": stream.commands_hash,
        "images": bundle.image_records,
        "source": bundle.source_name,
        "style": bundle.style_name,
        "pyramid": bundle.spec.to_metadata(),
        "commands": records,
        "invalid_command_count": sum(1 for record in records if record["status"] != "valid"),
    }
    _json_print(summary)
    return 0


def _safe_reset_state(state_dir: Path) -> None:
    if not state_dir.exists():
        return
    if not (state_dir / "CURRENT").exists() and not (state_dir / "snapshots").exists():
        raise StateError(E_STATE_EXISTS, f"refusing to reset unowned directory: {state_dir}")
    shutil.rmtree(state_dir)


def _run_incremental(
    bundle: ConfigBundle,
    stream: CommandStream,
    args: argparse.Namespace,
    report: dict[str, Any],
) -> tuple[IncrementalEngine, dict[str, Any]]:
    state_dir = Path(args.state_dir).resolve()
    manager = SnapshotManager(state_dir)
    stop = StopRequest()
    stop.install()
    if args.reset:
        _safe_reset_state(state_dir)
        manager = SnapshotManager(state_dir)
    if args.resume:
        engine, manifest = manager.load(bundle, commands_hash=stream.commands_hash)
        report["resumed_from_version"] = int(manifest["global_version"])
        start_index = engine.cursor + 1
    else:
        if manager.current_info() is not None:
            raise StateError(
                E_STATE_EXISTS,
                "state directory already has a committed snapshot; use --resume or --reset",
                {"state_dir": str(state_dir)},
            )
        engine = IncrementalEngine(bundle)
        engine.commands_hash = stream.commands_hash
        bootstrap_started = time.perf_counter()
        bootstrap = engine.bootstrap()
        report["bootstrap"] = {
            "elapsed_seconds": round(time.perf_counter() - bootstrap_started, 9),
            "recomputed_tiles": [list(key) for key in bootstrap.recomputed],
            "tile_count": len(bootstrap.recomputed),
        }
        start_index = 0

    engine.commands_hash = stream.commands_hash
    if start_index > len(stream.commands):
        raise StateError(
            "E_RESUME_MISMATCH",
            "snapshot command cursor is beyond the supplied command stream",
            {"cursor": engine.cursor, "command_count": len(stream.commands)},
        )

    snapshot_result: dict[str, Any] = {}
    stopped = False
    for index in range(start_index, len(stream.commands)):
        if stop.requested:
            stopped = True
            break
        entry = engine.process_command(stream.commands[index], index)
        report["commands"].append(entry)
        report["failures"] = engine.failures
        should_commit = (index + 1) % bundle.commit_every_commands == 0
        if args.stop_after is not None and len(report["commands"]) >= args.stop_after:
            stopped = True
        if stop.requested or stopped:
            should_commit = True
        if should_commit:
            _finalize_report(engine, report)
            snapshot_result = manager.commit(engine, report)
        if stop.requested or stopped:
            break

    if stop.requested:
        stopped = True
    _finalize_report(engine, report)
    report["stop_requested"] = stopped
    report["stop_signal"] = stop.signum
    snapshot_result = manager.commit(engine, report)
    report["snapshot"] = {
        key: value
        for key, value in snapshot_result.items()
        if key not in ("report",)
    }
    return engine, report


def _run_full(
    bundle: ConfigBundle,
    stream: CommandStream,
    args: argparse.Namespace,
    report: dict[str, Any],
) -> tuple[IncrementalEngine, dict[str, Any]]:
    mutable = bundle.new_mutable()
    for index, command in enumerate(stream.commands):
        try:
            normalized, change = apply_command(mutable, command, bundle.spec)
            report["commands"].append(
                {
                    "index": index,
                    "id": command.get("id") if isinstance(command, dict) else None,
                    "type": command.get("type") if isinstance(command, dict) else None,
                    "status": "applied",
                    "command": normalized,
                    "changed": bool(change.changed),
                }
            )
        except CommandError as exc:
            report["commands"].append(
                {
                    "index": index,
                    "id": command.get("id") if isinstance(command, dict) else None,
                    "type": command.get("type") if isinstance(command, dict) else None,
                    "status": "invalid_skipped",
                    "error": exc.as_dict(),
                    "changed": False,
                }
            )
    started = time.perf_counter()
    engine = IncrementalEngine.build_full(
        bundle,
        mutable=mutable,
        cursor=len(stream.commands) - 1,
        commands_hash=stream.commands_hash,
    )
    report["full_rebuild_elapsed_seconds"] = round(time.perf_counter() - started, 9)
    report["snapshot"] = None
    _finalize_report(engine, report)
    return engine, report


def command_run(args: argparse.Namespace) -> int:
    bundle = load_config(args.config, args.images)
    stream = load_commands(args.commands)
    mode = "full" if args.mode == "full" else "incremental"
    report = _base_report(bundle, stream, mode)
    if mode == "full":
        engine, report = _run_full(bundle, stream, args, report)
    else:
        engine, report = _run_incremental(bundle, stream, args, report)
    output_summary = engine.export_outputs(args.output_dir)
    report["outputs"] = output_summary
    atomic_write_json(args.report, report)
    _json_print(
        {
            "status": report.get("diagnostic_result", "OK"),
            "mode": mode,
            "report": str(Path(args.report).resolve()),
            "outputs": output_summary,
            "snapshot": report.get("snapshot"),
            "pending_tiles": report.get("pending_tiles", []),
            "layer_fingerprints": engine.layer_fingerprints(),
            "composite_fingerprint": None
            if engine.composite is None
            else engine.composite.fingerprint,
        }
    )
    return 0


def command_recover(args: argparse.Namespace) -> int:
    manager = SnapshotManager(args.state_dir)
    _json_print(manager.recover_info())
    return 0


def command_verify(args: argparse.Namespace) -> int:
    bundle = load_config(args.config, args.images)
    stream = load_commands(args.commands)
    incremental = IncrementalEngine(bundle)
    incremental.commands_hash = stream.commands_hash
    bootstrap = incremental.bootstrap()
    full_mutable = bundle.new_mutable()
    comparisons: list[dict[str, Any]] = []
    deferred_commands: list[int] = []
    invalid_commands: list[int] = []
    for index, command in enumerate(stream.commands):
        entry = incremental.process_command(command, index)
        if entry["status"] == "invalid_skipped":
            invalid_commands.append(index)
        try:
            apply_command(full_mutable, command, bundle.spec)
        except CommandError:
            pass
        if incremental.dirty_tiles:
            deferred_commands.append(index)
            comparisons.append(
                {
                    "command_index": index,
                    "command_id": entry.get("id"),
                    "checked": False,
                    "reason": "E_TILE_BUDGET_PENDING",
                    "pending_tiles": len(incremental.dirty_tiles),
                }
            )
            continue
        full = IncrementalEngine.build_full(
            bundle,
            mutable=full_mutable.clone(),
            cursor=index,
            commands_hash=stream.commands_hash,
        )
        comparison = compare_engines(incremental, full)
        comparisons.append(
            {
                "command_index": index,
                "command_id": entry.get("id"),
                "checked": True,
                "equal": comparison["equal"],
                "differences": comparison["differences"],
            }
        )

    drain_outcomes = incremental.drain_pending()
    full_final = IncrementalEngine.build_full(
        bundle,
        mutable=full_mutable.clone(),
        cursor=len(stream.commands) - 1,
        commands_hash=stream.commands_hash,
    )
    final_comparison = compare_engines(incremental, full_final)
    all_checked_equal = all(
        item.get("equal", True) for item in comparisons if item.get("checked")
    )
    summary = {
        "status": "ok" if all_checked_equal and final_comparison["equal"] else "mismatch",
        "bootstrap_tiles": len(bootstrap.recomputed),
        "commands": len(stream.commands),
        "invalid_commands": invalid_commands,
        "deferred_commands_due_to_budget": deferred_commands,
        "comparisons": comparisons,
        "drain_batches": [
            {
                "command_id": outcome.command_id,
                "recomputed": len(outcome.recomputed),
                "pending": len(outcome.pending),
                "status": outcome.status,
            }
            for outcome in drain_outcomes
        ],
        "final_comparison": {
            "equal": final_comparison["equal"],
            "differences": final_comparison["differences"],
            "actual": final_comparison["actual"],
            "expected": final_comparison["expected"],
        },
        "final_pending_tiles": [list(key) for key in sorted(incremental.dirty_tiles)],
        "final_layer_fingerprints": incremental.layer_fingerprints(),
        "final_composite_fingerprint": None
        if incremental.composite is None
        else incremental.composite.fingerprint,
    }
    if args.report:
        atomic_write_json(args.report, summary)
    _json_print(summary)
    return 0 if summary["status"] == "ok" else 1


def command_generate(args: argparse.Namespace) -> int:
    manifest = generate_sample(
        args.output,
        seed=args.seed,
        size=args.size,
        levels=args.levels,
        tile_size=args.tile_size,
        tile_budget=args.tile_budget,
        threads=args.threads,
    )
    _json_print({"status": "ok", "output": str(Path(args.output).resolve()), "manifest": manifest})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ipw", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate deterministic local PNG samples")
    generate.add_argument("--output", required=True)
    generate.add_argument("--seed", type=int, default=20261123)
    generate.add_argument("--size", type=int, default=128)
    generate.add_argument("--levels", type=int, default=4)
    generate.add_argument("--tile-size", type=int, default=16)
    generate.add_argument("--tile-budget", type=int, default=8)
    generate.add_argument("--threads", type=int, default=4)
    generate.set_defaults(func=command_generate)

    validate = subparsers.add_parser("validate", help="validate config, images and commands")
    validate.add_argument("--config", required=True)
    validate.add_argument("--commands", required=True)
    validate.add_argument("--images", required=True)
    validate.set_defaults(func=command_validate)

    run = subparsers.add_parser("run", help="run incremental or full reconstruction")
    run.add_argument("--config", required=True)
    run.add_argument("--commands", required=True)
    run.add_argument("--images", required=True)
    run.add_argument("--state-dir", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--report", required=True)
    run.add_argument("--mode", choices=["incremental", "full"], default="incremental")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--reset", action="store_true")
    run.add_argument("--stop-after", type=int, default=None)
    run.set_defaults(func=command_run)

    recover = subparsers.add_parser("recover", help="validate and report the committed recovery point")
    recover.add_argument("--state-dir", required=True)
    recover.set_defaults(func=command_recover)

    verify = subparsers.add_parser("verify", help="compare incremental and full reconstructions")
    verify.add_argument("--config", required=True)
    verify.add_argument("--commands", required=True)
    verify.add_argument("--images", required=True)
    verify.add_argument("--report", default=None)
    verify.set_defaults(func=command_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ConfigError as exc:
        _json_print({"status": "error", **exc.as_dict()})
        return 2
    except WorkbenchError as exc:
        _json_print({"status": "error", **exc.as_dict()})
        return 2 if exc.code == E_CONFIG_INVALID else 3
    except Exception as exc:
        _json_print(
            {
                "status": "error",
                "code": "E_UNEXPECTED",
                "message": str(exc),
                "type": type(exc).__name__,
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
