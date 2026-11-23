#!/usr/bin/env python3
"""Generate the deterministic local sample configuration and edit plan."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path


SEED = 20261124


def make_sequence(
    seq_id: str,
    length: int,
    *,
    phase: float,
    baseline: float,
    rng: random.Random,
    missing_index: int | None = None,
) -> dict:
    observations: list[float | None] = []
    for index in range(length):
        value = (
            baseline
            + 0.071 * index
            + 0.84 * math.sin(phase + 0.47 * index)
            + 0.22 * math.cos(0.19 * index)
            + rng.uniform(-0.055, 0.055)
        )
        observations.append(round(value, 5))
    if missing_index is not None:
        observations[missing_index % length] = None
    split = max(1, length // 2)
    segment_ids = [0] * split + [1] * (length - split)
    return {
        "id": seq_id,
        "observations": observations,
        "timestamp_interval": 10 + (len(seq_id) % 3),
        "segment_ids": segment_ids,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    samples = root / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    records = [
        make_sequence("S00", 16, phase=0.1, baseline=0.0, rng=rng),
        make_sequence("S01", 14, phase=0.4, baseline=0.25, rng=rng, missing_index=8),
        make_sequence("S02", 18, phase=0.7, baseline=-0.1, rng=rng),
        make_sequence("S03", 12, phase=1.1, baseline=0.12, rng=rng, missing_index=3),
        make_sequence("S04", 16, phase=1.7, baseline=-0.25, rng=rng),
        make_sequence("S05", 15, phase=2.1, baseline=0.08, rng=rng),
        make_sequence("S06", 17, phase=2.6, baseline=-0.05, rng=rng),
    ]
    with (samples / "sequences.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    config_text = """\
version: 1
reference_sequence_id: S00
step_mode: symmetric
bandwidth_permille: 260
distance_coefficient: 2
fixed_point:
  fraction_bits: 8
  quantization_step: 4
missing_marker: null
missing_penalty: 24
max_path_length: 80
path_length_policy: reject
thread_pool_workers: 4
cell_recompute_budget: 180
snapshot_interval_batches: 2
"""
    (samples / "config.yaml").write_text(config_text, encoding="utf-8")

    replacement_s04 = make_sequence(
        "S04", 13, phase=1.9, baseline=-0.18, rng=random.Random(SEED + 4)
    )
    replacement_s03 = make_sequence(
        "S03", 11, phase=1.3, baseline=0.16, rng=random.Random(SEED + 3)
    )
    replacement_s02 = make_sequence(
        "S02", 16, phase=0.65, baseline=-0.04, rng=random.Random(SEED + 2)
    )
    plan = {
        "batches": [
            {
                "batch": 1,
                "edits": [
                    {
                        "type": "point_update",
                        "sequence_id": "S02",
                        "index": 4,
                        "value": 0.42617,
                    },
                    {"type": "sequence_replace", "record": replacement_s04},
                ],
            },
            {
                "batch": 2,
                "edits": [
                    {
                        "type": "sequence_append",
                        "sequence_id": "S05",
                        "observations": [0.31, -0.22, 0.08],
                        "segment_ids": [2, 2, 2],
                    },
                    {"type": "bandwidth_update", "bandwidth_permille": 320},
                ],
            },
            {
                "batch": 3,
                "edits": [
                    {"type": "step_mode_update", "step_mode": "asymmetric"},
                    {
                        "type": "point_update",
                        "sequence_id": "S01",
                        "index": 9,
                        "value": None,
                    },
                ],
            },
            {
                "batch": 4,
                "edits": [
                    {"type": "sequence_replace", "record": replacement_s03},
                    {"type": "bandwidth_update", "bandwidth_permille": 180},
                ],
            },
            {
                "batch": 5,
                "edits": [
                    {"type": "step_mode_update", "step_mode": "symmetric"},
                    {
                        "type": "sequence_append",
                        "sequence_id": "S06",
                        "observations": [0.04, 0.19],
                        "segment_ids": [3, 3],
                    },
                ],
            },
            {
                "batch": 6,
                "edits": [
                    {
                        "type": "point_update",
                        "sequence_id": "S00",
                        "index": 5,
                        "value": 0.73331,
                    },
                    {"type": "bandwidth_update", "bandwidth_permille": 260},
                ],
            },
            {
                "batch": 7,
                "edits": [
                    {"type": "sequence_replace", "record": replacement_s02},
                    {"type": "bandwidth_update", "bandwidth_permille": 300},
                ],
            },
        ]
    }
    (samples / "edits.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
