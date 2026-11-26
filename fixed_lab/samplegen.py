"""Deterministic local synthetic regression sample generation."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


COLUMNS = [
    "base_signal",
    "strong_correlated",
    "collinear_left",
    "collinear_right",
    "independent_noise",
    "mixed_signal",
]

TRUE_COEFFICIENTS = [1.45, -0.85, 0.62, 0.55, 0.18, -0.38]


def generate_samples(count: int = 18, base_seed: int = 396) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for index in range(count):
        seed = base_seed + index * 1009
        rng = random.Random(seed)
        z0 = rng.gauss(0.0, 1.0)
        noise1 = rng.gauss(0.0, 0.22)
        z2 = rng.gauss(0.0, 1.0)
        noise_col = rng.gauss(0.0, 0.035)
        z4 = rng.gauss(0.0, 1.0)
        noise_mix = rng.gauss(0.0, 0.42)

        x0 = z0
        x1 = 0.97 * z0 + noise1
        x2 = 0.68 * z0 + 0.72 * z2
        x3 = x2 + noise_col
        x4 = z4
        x5 = 0.52 * z0 - 0.46 * z4 + noise_mix
        features = [x0, x1, x2, x3, x4, x5]
        observation_noise = rng.gauss(0.0, 0.12)
        target = sum(coef * value for coef, value in zip(TRUE_COEFFICIENTS, features)) + observation_noise
        weight = 0.75 + 0.5 * ((index * 37) % 11) / 10.0
        samples.append({
            "id": f"S{index + 1:04d}",
            "seed": seed,
            "features": [round(value, 6) for value in features],
            "target": round(target, 6),
            "weight": round(weight, 3),
        })
    return samples


def write_jsonl(samples: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
    return path
