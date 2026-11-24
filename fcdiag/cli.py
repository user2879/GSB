"""Command line entry point for fixture generation, runs and injections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

from .experiment import (
    assert_run_equal,
    full_rebuild,
    incremental_runs,
    result_signature,
    run_injections,
)
from .fingerprint import canonical_json
from .loader import load_commands, load_config, load_pipeline, load_samples
from .receipts import parse_receipts
from .runner import Runner
from .snapshot import SnapshotStore
from .state import WorkState, apply_commands


def _load_context(sample_dir: Path):
    samples = load_samples((sample_dir / "samples.jsonl").read_text(encoding="utf-8"))
    pipeline = load_pipeline((sample_dir / "pipeline.yaml").read_text(encoding="utf-8"))
    config = load_config(json.loads((sample_dir / "config.json").read_text(encoding="utf-8")))
    commands = load_commands((sample_dir / "commands.jsonl").read_text(encoding="utf-8"))
    receipts = parse_receipts(
        3, (sample_dir / "receipts" / "v3.txt").read_text(encoding="utf-8")
    )
    return samples, pipeline, config, commands, receipts


def main(argv: Tuple[str, ...] = None) -> int:
    parser = argparse.ArgumentParser(description="Feature cache hit diagnostics")
    parser.add_argument("command", choices=("run", "incremental-check", "injections"))
    parser.add_argument("--samples", default="samples")
    parser.add_argument("--output", default="output/run-report.json")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    sample_dir = Path(args.samples)
    samples, pipeline, config, commands, receipts = _load_context(sample_dir)
    state = WorkState.initial(samples, pipeline, config)
    if args.command == "run":
        result = Runner(state, receipts, max_workers=args.workers).run()
        report = result_signature(result)
    elif args.command == "incremental-check":
        incremental = incremental_runs(state, commands, receipts, args.workers)
        final_state = apply_commands(state, commands)[-1]
        rebuilt = full_rebuild(final_state, receipts, args.workers)
        assert_run_equal(incremental[-1], rebuilt)
        report = {
            "command_count": len(commands),
            "equal": True,
            "final_signature": result_signature(rebuilt),
        }
    else:
        receipt_texts = {
            version: (sample_dir / "receipts" / f"v{version}.txt").read_text(encoding="utf-8")
            for version in (1, 2, 3)
        }
        reports = run_injections(
            (sample_dir / "samples.jsonl").read_text(encoding="utf-8"),
            (sample_dir / "pipeline.yaml").read_text(encoding="utf-8"),
            json.loads((sample_dir / "config.json").read_text(encoding="utf-8")),
            receipt_texts,
            json.loads((sample_dir / "injections.json").read_text(encoding="utf-8")),
        )
        report = {"reports": [item.to_dict() for item in reports]}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_json(report) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
