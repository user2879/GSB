"""Generate the deterministic local sample workspace.

This thin wrapper uses numpy to produce an auxiliary, reproducible weight grid
(a local artifact only - no network access) and then delegates JSON authoring
to the platform CLI so every run yields byte-identical input files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from edge_lease.cli import generate_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="create deterministic sample data")
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--base-port", type=int, default=47000)
    parser.add_argument("--seed", type=int, default=20261123)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    weight_grid = rng.integers(1, 100, size=16, dtype=np.int32)
    args.workdir.mkdir(parents=True, exist_ok=True)
    (args.workdir / "weight_grid.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "dtype": "int32",
                "weights": weight_grid.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    paths = generate_samples(args.workdir, args.base_port)
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
