#!/usr/bin/env python3
"""Generate samples (unless present) and run the full acceptance suite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qcal.acceptance import run_acceptance
from generate_samples import generate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default="samples")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args()
    samples = Path(args.samples)
    if args.regenerate or not (samples / "manifest.json").exists():
        generate(samples, args.seed)
    report = run_acceptance(samples, args.artifacts)
    print(json_summary(report))


def json_summary(report: dict) -> str:
    import json

    keys = [
        "schema_version",
        "seed",
        "full_run",
        "conversion",
        "edit_sequence",
        "incremental_reuse",
        "forced_interrupt",
        "budget",
        "injection",
        "verification",
    ]
    return json.dumps({key: report[key] for key in keys}, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
