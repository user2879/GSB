from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .errors import ConfigError
from .io_utils import atomic_write_json
from .runner import (
    FixturePaths,
    load_fixtures,
    replay_full,
    replay_incremental,
    verify_consistency,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLES = ROOT / "samples"


def _paths(args: argparse.Namespace) -> FixturePaths:
    base = Path(args.samples_dir) if getattr(args, "samples_dir", None) else DEFAULT_SAMPLES
    return FixturePaths(
        samples=str(Path(args.samples) if getattr(args, "samples", None) else base / "exposure_samples.jsonl"),
        feature_config=str(
            Path(args.feature_config)
            if getattr(args, "feature_config", None)
            else base / "feature_config.json"
        ),
        training_config=str(
            Path(args.train_config)
            if getattr(args, "train_config", None)
            else base / "train_config.yaml"
        ),
        commands=str(Path(args.commands) if getattr(args, "commands", None) else base / "commands.json"),
    )


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ctr_cvr_pipeline",
        description="Deterministic fixed-point CTR/CVR joint estimation diagnostics",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="write deterministic sample fixtures")
    generate_parser.add_argument("--output", default=str(DEFAULT_SAMPLES))

    for name in ("replay", "verify"):
        command_parser = subparsers.add_parser(name)
        command_parser.add_argument("--samples-dir", default=None)
        command_parser.add_argument("--samples", default=None)
        command_parser.add_argument("--feature-config", default=None)
        command_parser.add_argument("--train-config", default=None)
        command_parser.add_argument("--commands", default=None)
        command_parser.add_argument(
            "--output",
            default=None,
            help="output directory (defaults to artifacts/<mode>)",
        )
        command_parser.add_argument("--workers", type=int, default=None)
        command_parser.add_argument("--keep", action="store_true", help="do not clear the output directory")
        if name == "replay":
            command_parser.add_argument("--mode", choices=("incremental", "full"), default="incremental")

    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            from scripts.generate_samples import generate

            manifest = generate(Path(args.output))
            _print({"generated": args.output, **manifest})
            return 0

        fixtures = load_fixtures(_paths(args))
        if args.command == "replay":
            output = Path(args.output) if args.output else ROOT / "artifacts" / args.mode
            if args.mode == "incremental":
                result = replay_incremental(
                    fixtures,
                    output,
                    workers=args.workers,
                    clean=not args.keep,
                )
            else:
                result = replay_full(
                    fixtures,
                    output,
                    workers=args.workers,
                    clean=not args.keep,
                )
            _print(result["summary"])
            return 0

        output = Path(args.output) if args.output else ROOT / "artifacts" / "verification"
        summary = verify_consistency(fixtures, output, clean=not args.keep)
        _print(summary)
        return 0 if summary["consistent"] else 2
    except ConfigError as exc:
        report = {"ok": False, "error": exc.to_dict()}
        if getattr(args, "output", None):
            atomic_write_json(Path(args.output) / "config_error.json", report)
        _print(report)
        return 1


if __name__ == "__main__":
    sys.exit(main())
