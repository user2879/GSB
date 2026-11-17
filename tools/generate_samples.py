"""Generate the deterministic local diagnostic fixture set."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


SEED = 20261118
DIMENSION = 16

CONFIG = {
    "schemes": ["RoPE", "LinearInterpolation", "NTK-aware"],
    "base": 10000.0,
    "train_context_length": 32,
    "extrapolation_target_length": 96,
    "dimension": DIMENSION,
    "scaling_factor": 2.0,
    "precision": "float64",
    "concurrency": 4,
    "tolerances": {
        "position_vector": 1e-6,
        "dot_score": 1e-6,
        "normalized_attention_weight": 1e-6,
    },
    "sensitivity": {
        "base_values": [0.0, 1e-300, 1000.0, 5000.0, 10000.0, 20000.0, 1e300],
        "scaling_factor_values": [0.0, 0.5, 1.0, 2.0, 4.0, 1e300],
    },
}

SAMPLE_SPECS = [
    ("s-train-001", 8, [1, 4, 7]),
    ("s-boundary-001", 32, [0, 16, 32]),
    ("s-mixed-001", 48, [8, 24, 40, 47]),
    ("s-extrap-close-001", 64, [20, 36, 52, 63]),
    ("s-extrap-far-001", 95, [0, 20, 50, 80]),
    ("s-inj-nonfinite", 50, [10, 45]),
    ("s-inj-position", 60, [10, 55]),
    ("s-inj-base", 70, [20, 65]),
    ("s-inj-scale", 80, [30, 75]),
]

INJECTIONS = {
    "version": 1,
    "injections": [
        {
            "injection_id": "INJ-001",
            "sample_id": "s-inj-nonfinite",
            "stage": "encoding",
            "fault": "nonfinite_feature",
            "target": "query_features",
            "index": 2,
            "value": "nan",
        },
        {
            "injection_id": "INJ-002",
            "sample_id": "s-inj-position",
            "stage": "scoring",
            "fault": "position_out_of_range",
            "target": "key_position",
            "key_index": 0,
            "position": 96,
        },
        {
            "injection_id": "INJ-003",
            "sample_id": "s-inj-base",
            "scheme": "RoPE",
            "stage": "encoding",
            "fault": "base_zero",
        },
        {
            "injection_id": "INJ-004",
            "sample_id": "s-inj-scale",
            "scheme": "LinearInterpolation",
            "stage": "encoding",
            "fault": "scale_nonpositive",
            "value": -1.0,
        },
    ],
}


def _rng_for(token: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{SEED}:{token}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _features(token: str) -> list[float]:
    values = _rng_for(token).normal(0.0, 0.35, size=DIMENSION).round(12)
    return [float(item) for item in values]


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "position_config.json").write_text(
        json.dumps(CONFIG, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines: list[str] = []
    for sample_id, query_position, key_positions in SAMPLE_SPECS:
        sample = {
            "sample_id": sample_id,
            "query_position": query_position,
            "query_features": _features(f"{sample_id}:query"),
            "key_positions": key_positions,
            "key_features": [
                _features(f"{sample_id}:key:{index}") for index in range(len(key_positions))
            ],
        }
        lines.append(json.dumps(sample, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    (output_dir / "attention_samples.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "injections.json").write_text(
        json.dumps(INJECTIONS, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    generate(Path(__file__).resolve().parents[1] / "samples")
