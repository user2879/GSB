"""Command line interface for the routing workbench."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .engine import Workbench
from .errors import SimulatedCrash
from .injection import run_injections
from .sample_generator import generate_samples
from .util import atomic_write_json
from .verification import run_verification


def _emit(report: dict[str, Any], path: str | None) -> None:
    if path:
        atomic_write_json(Path(path), report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))


def _add_common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--samples", default=None, help="sample directory")
    parser.add_argument("--utterances", default=None, help="utterance JSONL path")
    parser.add_argument("--config", default=None, help="routing YAML path")
    parser.add_argument("--commands", default=None, help="editing command JSON path")
    parser.add_argument("--vocabulary", default=None, help="optional controlled vocabulary text file")
    parser.add_argument("--state", default="state", help="state directory")
    parser.add_argument("--report", default=None, help="write the complete report JSON here")
    parser.add_argument("--full", action="store_true", help="ignore any existing snapshot and rebuild")
    parser.add_argument("--no-resume", action="store_true", help="do not reuse an existing snapshot")
    parser.add_argument("--expected-version", type=int, default=None, help="require this snapshot global version")
    parser.add_argument("--batch-budget", type=int, default=None, help="override routing point budget for this batch")
    parser.add_argument("--workers", type=int, default=None, help="override thread pool size, capped by configuration")
    parser.add_argument("--crash-after-commands", type=int, default=None, help="simulate forced termination after N commands")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m routing_workbench", description="Deterministic hierarchical intent routing workbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-samples", help="create deterministic local samples")
    generate.add_argument("--samples", default="samples")

    run = subparsers.add_parser("run", help="apply commands and route utterances")
    _add_common_run_args(run)

    verify = subparsers.add_parser("verify", help="run the complete acceptance ladder")
    verify.add_argument("--samples", default="samples")
    verify.add_argument("--workdir", default=".verify")
    verify.add_argument("--report", default="reports/verification.json")
    verify.add_argument("--skip-pytest", action="store_true")

    inject = subparsers.add_parser("inject", help="run the independent real-fault injection harness")
    inject.add_argument("--samples", default="samples")
    inject.add_argument("--utterances", default=None)
    inject.add_argument("--config", default=None)
    inject.add_argument("--injections", default=None)
    inject.add_argument("--vocabulary", default=None)
    inject.add_argument("--report", default="reports/injection-report.json")

    inspect = subparsers.add_parser("snapshot-info", help="print an existing snapshot manifest")
    inspect.add_argument("--state", default="state")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate-samples":
        paths = generate_samples(args.samples)
        print(json.dumps(paths, ensure_ascii=False, indent=2))
        return 0
    if args.command == "run":
        try:
            report = Workbench(
                samples_dir=args.samples,
                utterances_path=args.utterances,
                config_path=args.config,
                commands_path=args.commands,
                vocabulary_path=args.vocabulary,
                state_dir=args.state,
                full=args.full,
                resume=not args.no_resume,
                expected_global_version=args.expected_version,
                batch_budget=args.batch_budget,
                workers=args.workers,
                crash_after_commands=args.crash_after_commands,
            ).run()
        except SimulatedCrash as exc:
            print(f"SIMULATED_CRASH: {exc}", file=sys.stderr)
            return 75
        _emit(report, args.report)
        return 0
    if args.command == "verify":
        report = run_verification(
            samples_dir=args.samples,
            workdir=args.workdir,
            report_path=args.report,
            run_pytest=not args.skip_pytest,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report["all_passed"] else 1
    if args.command == "inject":
        samples = Path(args.samples)
        report = run_injections(
            utterances_path=args.utterances or samples / "utterances.jsonl",
            config_path=args.config or samples / "routing.yaml",
            injections_path=args.injections or samples / "injections.yaml",
            vocabulary_path=args.vocabulary or (samples / "vocabulary.txt" if (samples / "vocabulary.txt").exists() else None),
        )
        _emit(report, args.report)
        return 0 if report["all_hit"] else 1
    if args.command == "snapshot-info":
        from .snapshot import SnapshotManager

        loaded = SnapshotManager(args.state).load()
        print(json.dumps(loaded.manifest or {}, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if loaded.status == "ok" else 1
    return 2

