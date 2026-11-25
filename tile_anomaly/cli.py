"""Command-line entry points for sample generation, edits, and pipeline runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .edits import apply_edit_commands
from .pipeline import run_pipeline
from .samplegen import generate_sample


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tile-anomaly")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="generate deterministic local sample")
    generate_parser.add_argument("--output", type=Path, default=Path("sample"))
    generate_parser.add_argument("--seed", type=int, default=20261126)
    generate_parser.add_argument("--no-overwrite", action="store_true")

    run_parser = subparsers.add_parser("run", help="run or incrementally resume pipeline")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--fault-config", type=Path)
    run_parser.add_argument("--reset", action="store_true")
    run_parser.add_argument("--json", action="store_true")

    edit_parser = subparsers.add_parser("apply-edits", help="apply deterministic edit command stream")
    edit_parser.add_argument("--config", type=Path, required=True)
    edit_parser.add_argument("--edits", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "generate":
        summary = generate_sample(
            arguments.output,
            seed=arguments.seed,
            overwrite=not arguments.no_overwrite,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if arguments.command == "apply-edits":
        result = apply_edit_commands(arguments.config, arguments.edits)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    result = run_pipeline(
        arguments.config,
        fault_config_path=arguments.fault_config,
        reset_state=arguments.reset,
    )
    payload = result.to_dict()
    if arguments.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"complete={result.complete}")
        print(f"snapshot={result.snapshot_dir}")
        print(f"global_version={result.global_version}")
        print(f"output_png={result.output_png}")
        print(f"reuse={result.audit_summary.get('total_reuse', 0)}")
        print(f"invalidate={result.audit_summary.get('total_invalidate', 0)}")
        if result.events:
            print("events:")
            for event in result.events:
                print(
                    f"  {event['code']} tile={event['tile_id']} stage={event['stage']} {event['message']}"
                )
    return 0 if result.complete else 2


if __name__ == "__main__":
    raise SystemExit(main())

