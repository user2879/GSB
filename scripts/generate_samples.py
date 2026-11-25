"""Generate deterministic local samples with a fixed integer seed.

Run from the workspace root:  python scripts/generate_samples.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SEED = 20261125
ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
RECEIPTS_A = ROOT / "receipts" / "set_a"
RECEIPTS_B = ROOT / "receipts" / "set_b"

# (layer name, shape, group hint, content version)
LAYER_SPECS = [
    ("embedding_proj", (4, 8), 8, 1),
    ("retrieval_head", (4, 8), 4, 2),
    ("fusion_block", (2, 8), 2, 3),
    ("score_mixer", (8,), 8, 4),
]
SAMPLE_COUNT = 12
DIM = 8


def main() -> None:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    weight_lines = []
    layer_weights = {}
    for name, shape, group_hint, version in LAYER_SPECS:
        weights = rng.normal(0.0, 0.7, size=shape)
        # Keep every element finite and deterministic across platforms.
        weights = np.round(weights, 6)
        layer_weights[name] = weights
        weight_lines.append(json.dumps({
            "layer": name,
            "shape": list(shape),
            "weights": weights.ravel().tolist(),
            "content_version": version,
            "group_hint": group_hint,
        }, ensure_ascii=False))
    (SAMPLES / "weights.jsonl").write_text(
        "\n".join(weight_lines) + "\n", encoding="utf-8")

    inputs = np.round(rng.normal(0.0, 1.0, size=(SAMPLE_COUNT, DIM)), 5)
    # Reference outputs are deterministic projections through all layers.
    references = inputs @ layer_weights["embedding_proj"].T
    sample_ids = [f"cal-{index:03d}" for index in range(SAMPLE_COUNT)]
    calibration = {
        "inputs": inputs.tolist(),
        "references": references.tolist(),
        "sample_ids": sample_ids,
        "tier_sizes": {"small": 4, "medium": 8, "large": 12},
    }
    (SAMPLES / "calibration.json").write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8")

    # Single-layer weight replacement with a strictly newer content version.
    replacement = layer_weights["retrieval_head"].copy()
    replacement = np.round(replacement + rng.normal(0.0, 0.15,
                                                    size=replacement.shape), 6)
    (SAMPLES / "retrieval_head_replacement.json").write_text(json.dumps({
        "layer": "retrieval_head",
        "shape": list(replacement.shape),
        "weights": replacement.ravel().tolist(),
        "content_version": 5,
    }, ensure_ascii=False), encoding="utf-8")

    _write_receipts(RECEIPTS_A, variant="a")
    _write_receipts(RECEIPTS_B, variant="b")
    print(f"generated samples under {SAMPLES}")
    print(f"generated receipts under {RECEIPTS_A} and {RECEIPTS_B}")


def _write_receipts(directory: Path, variant: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    q1 = {"entries": [
        {"version": "q1", "layer": "embedding_proj",
         "coverage_permille": 950},
        {"version": "q1", "layer": "retrieval_head",
         "coverage_permille": 900},
    ]}
    q2 = {"entries": [
        {"version": "q2", "layer": "embedding_proj",
         "coverage_permille": 960, "param_version": 2, "elapsed_ms": 12},
        {"version": "q2", "layer": "fusion_block",
         "coverage_permille": 880, "param_version": 1, "elapsed_ms": 9},
    ]}
    q3 = {"entries": [
        {"version": "q3", "layer": "embedding_proj",
         "coverage_permille": 990, "param_version": 3, "elapsed_ms": 20,
         "group_scale_summary": [0.0812, 0.0756],
         "override_refs": ["receipt_q1.json#entry0"],
         "checksum": f"{variant}-emb-0001"},
        {"version": "q3", "layer": "score_mixer",
         "coverage_permille": 830, "param_version": 1, "elapsed_ms": 7,
         "group_scale_summary": [0.1203],
         "override_refs": [],
         "checksum": f"{variant}-mix-0001"},
    ]}
    (directory / "receipt_q1.json").write_text(
        json.dumps(q1, ensure_ascii=False, indent=2), encoding="utf-8")
    (directory / "receipt_q2.json").write_text(
        json.dumps(q2, ensure_ascii=False, indent=2), encoding="utf-8")
    (directory / "receipt_q3.json").write_text(
        json.dumps(q3, ensure_ascii=False, indent=2), encoding="utf-8")
    # A deliberately malformed receipt to exercise E_RECEIPT_MALFORMED.
    (directory / "receipt_bad.json").write_text(json.dumps({
        "entries": [{"version": "q1", "layer": "broken",
                     "coverage_permille": 9999}],
    }), encoding="utf-8")


if __name__ == "__main__":
    main()
