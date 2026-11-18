"""Command line entry points for synthesis, pipeline runs and incremental updates."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

from .config import build_config, load_config, load_pair_manifest
from .errors import ConfigMismatchError
from .pipeline import VoiceprintPipeline
from .reporting import build_report
from .synth import generate_dataset
from .util import atomic_write_json, fingerprint, read_json


def _injection_hash(injection_path: str) -> str:
    return fingerprint(read_json(injection_path))


def cmd_synth(args: argparse.Namespace) -> int:
    result = generate_dataset(args.input_dir, args.injection)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _execute(args: argparse.Namespace, resume: bool) -> int:
    config = load_config(args.config)
    pair_manifest = load_pair_manifest(args.pairs)
    injection_hash = _injection_hash(args.injection)
    pipeline = VoiceprintPipeline(
        config=config,
        pair_manifest=pair_manifest,
        input_dir=args.input_dir,
        work_dir=args.work_dir,
        injection_hash=injection_hash,
    )
    started = time.perf_counter()
    exit_code = pipeline.run(
        resume=resume, allow_config_change=getattr(args, "allow_config_change", False)
    )
    if exit_code != 0:
        return exit_code
    if pipeline.shutdown.requested:
        return 0
    elapsed = time.perf_counter() - started
    report = build_report(pipeline, pair_manifest, elapsed)
    atomic_write_json(args.report, report)
    print(f"report written: {args.report}")
    print(f"failures: {len(report['failures'])} "
          f"reused: {report['incremental']['reused_count']} "
          f"recomputed: {report['incremental']['recomputed_count']}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    try:
        return _execute(args, resume=args.resume)
    except ConfigMismatchError as exc:
        payload = {"code": exc.code, "message": str(exc)}
        atomic_write_json(args.report, payload)
        print(json.dumps(payload), file=sys.stderr)
        return 2


def cmd_incremental(args: argparse.Namespace) -> int:
    if args.replace_audio:
        replacement, target_name = args.replace_audio
        target = os.path.join(args.input_dir, os.path.basename(target_name))
        with open(replacement, "rb") as src, open(target, "wb") as dst:
            dst.write(src.read())
    return _execute(args, resume=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voiceprint",
                                     description="Deterministic voiceprint embedding pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    synth = sub.add_parser("synth", help="generate deterministic synthetic WAV dataset")
    synth.add_argument("--input-dir", required=True)
    synth.add_argument("--injection", required=True)
    synth.set_defaults(func=cmd_synth)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--input-dir", required=True)
    common.add_argument("--config", required=True)
    common.add_argument("--pairs", required=True)
    common.add_argument("--injection", required=True)
    common.add_argument("--work-dir", required=True)
    common.add_argument("--report", required=True)

    run = sub.add_parser("run", parents=[common], help="run the full pipeline")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--allow-config-change", action="store_true",
                     help="resume while accepting config changes; only cache-key-missed "
                          "stages recompute (used for threshold/width/level adjustments)")
    run.set_defaults(func=cmd_run)

    incr = sub.add_parser("incremental", parents=[common],
                          help="replace one audio then resume with precise invalidation")
    incr.add_argument("--replace-audio", nargs=2, metavar=("REPLACEMENT_WAV", "TARGET_NAME"))
    incr.set_defaults(func=cmd_incremental)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
