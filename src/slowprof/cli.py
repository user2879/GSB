"""Command line interface for profiling, replay and injected anomalies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .aggregate import aggregate_file
from .demo import EXIT_CODES, run_demo
from .errors import E_IO, SlowProfError
from .logging import StructuredLogger
from .models import Interval, ProfilePolicy
from .ranking import build_ranking, canonical_core_bytes
from .replay import replay_snapshot
from .session import profile_file
from .snapshot import restore_snapshot, save_snapshot


def _add_interval_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--start-ms", type=int, required=True)
    parser.add_argument("--end-ms", type=int, required=True)
    parser.add_argument("--interval-version", default="obs-v1")


def _add_policy_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-depth", type=int, default=64)
    parser.add_argument("--unique-threshold", type=int, default=64)
    parser.add_argument("--weight-threshold", type=int, default=4096)
    parser.add_argument("--scale", type=int, default=1_000_000)
    parser.add_argument("--top-k", type=int, default=5)


def _policy_from_args(args: argparse.Namespace) -> ProfilePolicy:
    return ProfilePolicy(
        max_depth=args.max_depth,
        full_unique_stack_threshold=args.unique_threshold,
        full_total_weight_threshold=args.weight_threshold,
        scale=args.scale,
        top_k=args.top_k,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="slowprof")
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile = subparsers.add_parser("profile", help="profile one NDJSON stream")
    profile.add_argument("--samples", required=True)
    _add_interval_options(profile)
    _add_policy_options(profile)
    profile.add_argument("--force-path", choices=("full", "streaming"), default=None)
    profile.add_argument("--snapshot", default=None)
    profile.add_argument("--log-path", default="runtime/logs/slowprof.jsonl")

    validate = subparsers.add_parser("validate", help="validate an NDJSON stream")
    validate.add_argument("--samples", required=True)
    _add_interval_options(validate)
    _add_policy_options(validate)
    validate.add_argument("--log-path", default="runtime/logs/slowprof.jsonl")

    parity = subparsers.add_parser("parity", help="compare full and streaming paths")
    parity.add_argument("--samples", required=True)
    _add_interval_options(parity)
    _add_policy_options(parity)
    parity.add_argument("--log-path", default="runtime/logs/slowprof.jsonl")

    replay = subparsers.add_parser("replay", help="replay and verify a snapshot")
    replay.add_argument("--snapshot", required=True)
    replay.add_argument("--samples", required=True)
    replay.add_argument("--interval-version", default=None)
    replay.add_argument("--log-path", default="runtime/logs/slowprof.jsonl")

    restore = subparsers.add_parser("restore", help="restore against a current profile session")
    restore.add_argument("--snapshot", required=True)
    restore.add_argument("--samples", required=True)
    _add_interval_options(restore)
    _add_policy_options(restore)
    restore.add_argument("--replacement-snapshot", default=None)
    restore.add_argument("--log-path", default="runtime/logs/slowprof.jsonl")

    demo = subparsers.add_parser("demo", help="run the three-session end-to-end example")
    demo.add_argument("--samples-dir", default="samples")
    demo.add_argument("--runtime-dir", default="runtime")
    demo.add_argument("--log-path", default=None)
    demo.add_argument(
        "--inject",
        choices=("none", "zero-weight", "depth-exceeded", "missing-version"),
        default="none",
    )
    return parser


def _emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2))


def _run(args: argparse.Namespace, logger: StructuredLogger) -> int:
    if args.command == "demo":
        runtime_dir = Path(args.runtime_dir)
        logger = logger if args.log_path else StructuredLogger(runtime_dir / "logs" / "slowprof.jsonl")
        outcome = run_demo(
            args.samples_dir,
            args.runtime_dir,
            logger=logger,
            inject=args.inject,
        )
        _emit(outcome.summary)
        return outcome.exit_code

    if args.command == "validate":
        from .ndjson import scan_ndjson

        policy = _policy_from_args(args)
        interval = Interval(args.start_ms, args.end_ms, args.interval_version)
        metadata = scan_ndjson(args.samples, policy, interval)
        logger.event("validation_ok", samples=args.samples, **metadata.to_dict())
        _emit({"ok": True, "metadata": metadata.to_dict()})
        return 0

    if args.command == "profile":
        policy = _policy_from_args(args)
        interval = Interval(args.start_ms, args.end_ms, args.interval_version)
        session = profile_file(
            args.samples,
            policy,
            interval,
            force_path=args.force_path,
        )
        result: dict[str, Any] = {"summary": session.to_summary()}
        if args.snapshot:
            save_snapshot(session, args.snapshot)
            result["snapshot"] = args.snapshot
        logger.event(
            "profile_completed",
            samples=args.samples,
            conclusion_sha256=session.conclusion_sha256,
            aggregation_path=session.aggregation.selection.path,
        )
        _emit(result)
        return 0

    if args.command == "parity":
        policy = _policy_from_args(args)
        interval = Interval(args.start_ms, args.end_ms, args.interval_version)
        full = aggregate_file(args.samples, policy, interval, force_path="full")
        streaming = aggregate_file(args.samples, policy, interval, force_path="streaming")
        full_bytes = canonical_core_bytes(full, build_ranking(full, policy))
        streaming_bytes = canonical_core_bytes(streaming, build_ranking(streaming, policy))
        import hashlib

        result = {
            "ok": True,
            "byte_equal": full_bytes == streaming_bytes,
            "full_path": full.selection.path,
            "streaming_path": streaming.selection.path,
            "canonical_sha256": hashlib.sha256(full_bytes).hexdigest(),
            "full_peak_memory_estimate_bytes": full.memory_estimate["estimated_peak_bytes"],
            "streaming_peak_memory_estimate_bytes": streaming.memory_estimate[
                "estimated_peak_bytes"
            ],
        }
        logger.event("parity_checked", samples=args.samples, **result)
        _emit(result)
        return 0 if result["byte_equal"] else 1

    if args.command == "replay":
        outcome = replay_snapshot(
            args.snapshot,
            args.samples,
            current_interval_version=args.interval_version,
        )
        logger.event("replay_verified", snapshot=args.snapshot, **outcome.report)
        _emit(outcome.report)
        return 0

    if args.command == "restore":
        policy = _policy_from_args(args)
        interval = Interval(args.start_ms, args.end_ms, args.interval_version)
        current = profile_file(args.samples, policy, interval)
        outcome = restore_snapshot(args.snapshot, current)
        if args.replacement_snapshot:
            save_snapshot(outcome.active_session, args.replacement_snapshot)
        result = outcome.to_dict()
        if args.replacement_snapshot:
            result["replacement_snapshot"] = args.replacement_snapshot
        logger.event("snapshot_restore", snapshot=args.snapshot, **result)
        _emit(result)
        return 0

    raise SlowProfError(E_IO, "unknown command", field="command")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    log_path = getattr(args, "log_path", None)
    logger = StructuredLogger(log_path) if log_path else StructuredLogger(
        Path("runtime") / "logs" / "slowprof.jsonl"
    )
    try:
        return _run(args, logger)
    except SlowProfError as error:
        logger.error(error, command=args.command)
        _emit(error.to_dict())
        return EXIT_CODES.get(error.code, 1)
    except OSError as error:
        wrapped = SlowProfError(
            E_IO,
            "local filesystem operation failed",
            field="filesystem",
            details={"error": str(error)},
        )
        logger.error(wrapped, command=args.command)
        _emit(wrapped.to_dict())
        return EXIT_CODES[E_IO]


if __name__ == "__main__":
    raise SystemExit(main())
