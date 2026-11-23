#!/usr/bin/env python3
"""Generate deterministic local YAML/JSONL samples for the pipeline demo."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


CONFIG_TEXT = """# Deterministic local rendering configuration\nframe_width: 32\nframe_height: 24\nprediction_steps: 2\nhistory_frames: 4\nmotion_model:\n  linear_gain: 1.0\n  damping: 0.82\n  search_radius: 2\nnoise_layer:\n  amplitude: 2.0\n  seed_salt: 97\nfixed_point_bits: 12\nquantization_levels: [1, 2, 4]\nprecision_levels: [6, 8, 10, 12]\nscale_tiers: [1, 2, 4]\nlatency_budget_ms: 250\nthread_pool_workers: 4\nsnapshot_commit_interval_batches: 1\n"""


PALETTE = [
    [220, 72, 48],
    [40, 186, 120],
    [52, 112, 230],
    [238, 196, 44],
    [178, 76, 218],
    [34, 196, 212],
    [236, 116, 166],
    [132, 214, 66],
]


def generate_sequences(count: int, frames: int, trajectories: int, seed: int) -> list[dict]:
    sequences: list[dict] = []
    width = 32
    height = 24
    for sequence_index in range(count):
        sequence_seed = seed + sequence_index * 1009
        rng = np.random.default_rng(sequence_seed)
        sequence_id = f"seq-{sequence_index + 1:04d}"
        trajectory_specs = []
        for trajectory_index in range(trajectories):
            shape = "circle" if trajectory_index % 3 else "rect"
            trajectory_specs.append(
                {
                    "trajectory_id": f"t-{trajectory_index + 1:02d}",
                    "shape": shape,
                    "x0": float(rng.uniform(4.0, width - 5.0)),
                    "y0": float(rng.uniform(4.0, height - 5.0)),
                    "vx": float(rng.uniform(-1.1, 1.1)),
                    "vy": float(rng.uniform(-0.8, 0.8)),
                    "radius": float(rng.uniform(1.4, 3.1)),
                    "width": float(rng.uniform(2.4, 5.2)),
                    "height": float(rng.uniform(2.2, 4.8)),
                    "color": PALETTE[trajectory_index % len(PALETTE)],
                    "phase": float(rng.uniform(0.0, math.tau)),
                }
            )
        frame_rows = []
        for frame_index in range(frames):
            frame_trajectories = []
            for trajectory_index, spec in enumerate(trajectory_specs):
                x = spec["x0"] + spec["vx"] * frame_index + math.sin(
                    spec["phase"] + frame_index * 0.37
                ) * 0.45
                y = spec["y0"] + spec["vy"] * frame_index + math.cos(
                    spec["phase"] * 0.7 + frame_index * 0.31
                ) * 0.35
                x = min(width - 1.0, max(0.0, x))
                y = min(height - 1.0, max(0.0, y))
                row = {
                    "trajectory_id": spec["trajectory_id"],
                    "shape": spec["shape"],
                    "x": round(x, 6),
                    "y": round(y, 6),
                    "vx": round(spec["vx"], 6),
                    "vy": round(spec["vy"], 6),
                    "color": spec["color"],
                }
                if spec["shape"] == "circle":
                    row["radius"] = round(spec["radius"], 6)
                else:
                    row["width"] = round(spec["width"], 6)
                    row["height"] = round(spec["height"], 6)
                frame_trajectories.append(row)
            moving_x = 2.0 + ((frame_index * 1.35 + sequence_index * 0.9) % (width - 10.0))
            color_blocks = [
                {
                    "x": 1.0,
                    "y": 1.0,
                    "width": 7.0,
                    "height": 4.0,
                    "color": [30, 46, 64],
                },
                {
                    "x": round(moving_x, 6),
                    "y": round(height - 6.0 - (sequence_index % 3), 6),
                    "width": 5.0,
                    "height": 4.0,
                    "color": [92, 68, 46],
                },
            ]
            frame_rows.append(
                {
                    "trajectories": frame_trajectories,
                    "color_blocks": color_blocks,
                    "noise_seed": int(sequence_seed + frame_index * 131 + 17),
                }
            )
        sequences.append(
            {
                "sequence_id": sequence_id,
                "noise_seed": sequence_seed,
                "frames": frame_rows,
            }
        )
    return sequences


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent),
        help="Directory for config.yaml and sequences.jsonl",
    )
    parser.add_argument("--sequences", type=int, default=8)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--trajectories", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261124)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(CONFIG_TEXT, encoding="utf-8")
    rows = generate_sequences(args.sequences, args.frames, args.trajectories, args.seed)
    with (out_dir / "sequences.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")
    print(f"wrote {len(rows)} sequences to {out_dir}")


if __name__ == "__main__":
    main()

