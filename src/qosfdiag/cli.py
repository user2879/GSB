"""Command-line entry points for validation, run, replay and the UI demo."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .demo import format_demo_summary, run_demo
from .engine import CascadeEngine
from .error_propagation import propagate_fixed_errors
from .errors import LedgerError, PolicyError, ReplayError, SnapshotError
from .generator import DeterministicUdpGenerator
from .ledger import DecisionLedger
from .policy import load_policy
from .replay import verify_snapshot_replay
from .samples import write_sample_files
from .snapshot import SnapshotManager


def _json_dump(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _clean_ledger(path: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(path) + suffix)
        if candidate.exists():
            candidate.unlink()


def _crash_hook(batch_no: int, phase: str, target: int, target_phase: str) -> None:
    if batch_no == target and phase == target_phase:
        os._exit(137)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qos-fixed-diag")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="run the three deterministic scenarios")
    demo.add_argument("--output", default="artifacts/demo")
    demo.add_argument("--json", action="store_true")

    generate = subparsers.add_parser("generate-samples", help="write bundled JSON samples")
    generate.add_argument("--output", default="examples/data")

    validate = subparsers.add_parser("validate", help="strictly validate a policy file")
    validate.add_argument("--policy", required=True)
    validate.add_argument("--max-rate-kbps", type=int)
    validate.add_argument("--json", action="store_true")

    run = subparsers.add_parser("run", help="run/resume UDP shaping into SQLite")
    run.add_argument("--policy", required=True)
    run.add_argument("--generator", required=True)
    run.add_argument("--scale", type=int, default=3)
    run.add_argument("--ledger", default="artifacts/run/ledger.sqlite")
    run.add_argument("--base-port", type=int, default=49200)
    run.add_argument("--reset", action="store_true")
    run.add_argument("--in-memory", action="store_true")
    run.add_argument("--snapshot-dir")
    run.add_argument("--snapshot-name", default="run.json")
    run.add_argument("--crash-after-marker-batch", type=int)
    run.add_argument("--crash-before-commit-batch", type=int)
    run.add_argument("--json", action="store_true")

    replay = subparsers.add_parser("replay", help="verify a snapshot by deterministic replay")
    replay.add_argument("--snapshot", required=True)
    replay.add_argument("--current-policy-version")
    replay.add_argument("--current-scale", type=int)
    replay.add_argument("--current-generator-digest")
    replay.add_argument("--json", action="store_true")

    ui = subparsers.add_parser("ui", help="launch the Tkinter diagnostics interface")
    ui.add_argument("--policy")
    ui.add_argument("--generator")
    ui.add_argument("--workdir", default="artifacts/ui")
    return parser


def _run_command(args: argparse.Namespace) -> int:
    policy = load_policy(args.policy)
    generator = DeterministicUdpGenerator(json.loads(Path(args.generator).read_text("utf-8")))
    ledger_path = Path(args.ledger)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    if args.reset:
        _clean_ledger(ledger_path)
    crash_hook = None
    if args.crash_after_marker_batch is not None:
        crash_hook = lambda batch_no, phase: _crash_hook(
            batch_no, phase, args.crash_after_marker_batch, "after_marker"
        )
    elif args.crash_before_commit_batch is not None:
        crash_hook = lambda batch_no, phase: _crash_hook(
            batch_no, phase, args.crash_before_commit_batch, "before_commit"
        )
    with DecisionLedger(ledger_path) as ledger:
        with CascadeEngine(
            policy,
            generator,
            scale=args.scale,
            use_loopback=not args.in_memory,
            base_port=args.base_port,
            crash_hook=crash_hook,
        ) as engine:
            result = engine.run(ledger=ledger)
        errors = propagate_fixed_errors(
            policy,
            result.summary,
            result.events,
            scale=args.scale,
            generator_digest=generator.sequence_digest,
        )
        progress = ledger.progress()
        ledger_info = {
            "cursor_seq": progress.cursor_seq,
            "committed_batches": progress.committed_batches,
            "content_digest": progress.content_digest,
        }
        snapshot_path = None
        if args.snapshot_dir:
            manager = SnapshotManager(args.snapshot_dir)
            document = manager.save(
                filename=args.snapshot_name,
                policy=policy,
                scale=args.scale,
                generator=generator,
                ledger=ledger,
                errors=errors,
            )
            snapshot_path = str(document.path)
    output = {
        "summary": result.summary.as_dict(),
        "errors": errors.as_dict(),
        "recovery": None if result.recovery is None else result.recovery.as_dict(),
        "ledger": ledger_info,
        "snapshot": snapshot_path,
    }
    if args.json:
        _json_dump(output)
    else:
        summary = result.summary
        print(
            f"generated={summary.total_generated_packets} "
            f"released={summary.final_released_packets} "
            f"bytes={summary.final_released_bytes}"
        )
        print(
            f"bytes_error_bound={errors.bytes_error_bound} "
            f"avg_delay_ns={errors.average_delay_ns} "
            f"delay_error_bound_ns={errors.average_delay_error_bound_ns}"
        )
        print(f"dominant_stage={errors.dominant_stage}")
        print(
            f"ledger_cursor={ledger_info['cursor_seq']} "
            f"batches={ledger_info['committed_batches']} "
            f"digest={ledger_info['content_digest']}"
        )
        if result.recovery and result.recovery.requires_rerun:
            print(
                "recovered_unknown=" + ",".join(result.recovery.unknown_packet_ids)
            )
        if snapshot_path:
            print(f"snapshot={snapshot_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            result = run_demo(args.output)
            if args.json:
                _json_dump(result)
            else:
                print(format_demo_summary(result))
            return 0
        if args.command == "generate-samples":
            paths = write_sample_files(args.output)
            for path in paths:
                print(path)
            return 0
        if args.command == "validate":
            policy = load_policy(args.policy, max_rate_kbps=args.max_rate_kbps)
            output = {
                "ok": True,
                "policy_version": policy.policy_version,
                "shapers": len(policy.shapers),
                "edges": len(policy.edges),
                "roots": list(policy.roots),
                "topological_order": list(policy.topological_order()),
            }
            if args.json:
                _json_dump(output)
            else:
                print(
                    f"OK policy_version={policy.policy_version} "
                    f"shapers={len(policy.shapers)} edges={len(policy.edges)}"
                )
            return 0
        if args.command == "run":
            return _run_command(args)
        if args.command == "replay":
            result = verify_snapshot_replay(
                args.snapshot,
                current_policy_version=args.current_policy_version,
                current_scale=args.current_scale,
                current_generator_digest=args.current_generator_digest,
            )
            if args.json:
                _json_dump(result.as_dict())
            else:
                print(
                    f"replay_ok={str(result.ok).lower()} "
                    f"events_compared={result.events_compared} "
                    f"classification={result.classification}"
                )
            return 0
        if args.command == "ui":
            from .ui import launch_ui

            launch_ui(
                policy_path=args.policy,
                generator_path=args.generator,
                workdir=args.workdir,
            )
            return 0
    except PolicyError as exc:
        _json_dump(exc.as_dict())
        return 2
    except ReplayError as exc:
        _json_dump(exc.as_dict())
        return 3
    except (LedgerError, SnapshotError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    parser.error("unknown command")
    return 2
