from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from posdiag.samples import generate_sample_directory


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic attention/POS-encoding samples.")
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent), help="sample output directory")
    parser.add_argument("--seed", type=int, default=20261124)
    args = parser.parse_args()
    generated = generate_sample_directory(args.output, seed=args.seed)
    for name, path in generated.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
