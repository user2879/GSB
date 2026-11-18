"""Run the complete deterministic demo with the public Python API.

Usage:
    python examples/run_demo.py --samples-dir samples --runtime-dir runtime
    python examples/run_demo.py --inject zero-weight
    python examples/run_demo.py --inject depth-exceeded
    python examples/run_demo.py --inject missing-version
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slowprof.demo import run_demo
from slowprof.logging import StructuredLogger


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-dir", default="samples")
    parser.add_argument("--runtime-dir", default="runtime")
    parser.add_argument(
        "--inject",
        choices=("none", "zero-weight", "depth-exceeded", "missing-version"),
        default="none",
    )
    args = parser.parse_args()
    runtime_dir = Path(args.runtime_dir)
    logger = StructuredLogger(runtime_dir / "logs" / "example.jsonl")
    outcome = run_demo(
        args.samples_dir,
        args.runtime_dir,
        logger=logger,
        inject=args.inject,
    )
    print(json.dumps(outcome.summary, sort_keys=True, ensure_ascii=False, indent=2))
    return outcome.exit_code


if __name__ == "__main__":
    raise SystemExit(main())

