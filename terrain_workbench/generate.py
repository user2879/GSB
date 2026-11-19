"""CLI: generate the three deterministic sample tiers + pyramids locally."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .heightfield import VALID_SIZES, generate_heightfield, heightfield_hash
from .pyramid import build_pyramid, save_pyramid


def generate_all(out_dir: str = "data", seed: int = 20261119, sizes=VALID_SIZES):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index = {}
    for size in sizes:
        tag = f"terrain_{size}"
        heights = generate_heightfield(size, seed)
        levels, meta = build_pyramid(heights)
        full_meta = save_pyramid(levels, meta, out, tag)
        index[tag] = {
            "size": size,
            "seed": seed,
            "level0_sha256": heightfield_hash(heights),
            "levels": full_meta["level_count"],
            "cumulative_dev": full_meta["cumulative_dev"],
        }
        print(f"[generate] {tag}: levels={full_meta['level_count']} "
              f"sha={index[tag]['level0_sha256'][:16]}")
    (out / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data")
    parser.add_argument("--seed", type=int, default=20261119)
    args = parser.parse_args()
    generate_all(args.out, args.seed)
