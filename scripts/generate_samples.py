"""Convenience wrapper for generating the bundled deterministic sample set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recall_workbench.samplegen import DEFAULT_SEED, generate_samples  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "data"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    manifest = generate_samples(Path(args.output), args.seed)
    print(f"generated {Path(args.output).resolve()} digest={manifest['deterministic_digest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
