#!/usr/bin/env python3
"""Regenerate the fixed sample corpus and run the complete acceptance verifier."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_sample import generate

from fixedvideo.verify import verify


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", default="samples", help="fixed sample directory")
    parser.add_argument("--workdir", default="runs/demo-verification", help="verification workspace")
    parser.add_argument("--seed", type=int, default=20261123, help="sample generation seed")
    parser.add_argument("--force", action="store_true", help="replace an existing verification workspace")
    args = parser.parse_args()
    samples = Path(args.samples).resolve()
    generate(samples, args.seed)
    report = verify(
        config_path=samples / "render_config.json",
        commands_path=samples / "commands.json",
        workdir=Path(args.workdir).resolve(),
        force=args.force,
    )
    summary = {
        "status": report["status"],
        "checks": report["checks"],
        "latency": {
            "budget_ms": report["latency"]["budget_ms"],
            "max_response_latency_ms": report["latency"]["max_response_latency_ms"],
            "pending_count": report["latency"]["pending_count"],
        },
        "incremental_vs_full": {
            "pixels_equal": report["incremental_vs_full"]["pixels_equal"],
            "frame_fingerprints_equal": report["incremental_vs_full"]["frame_fingerprints_equal"],
            "step_trace_equal": report["incremental_vs_full"]["step_trace_equal"],
            "required_artifact_content_equal": report["incremental_vs_full"][
                "required_artifact_content_equal"
            ],
        },
        "recovery": {
            "resumed_from_cursor": report["recovery"]["resumed_from_cursor"],
            "pixels_equal": report["recovery"]["output_equality"]["pixels_equal"],
        },
        "stale": report["stale"],
        "report": str(Path(args.workdir).resolve() / "reports" / "verification.json"),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
