"""Generate the four deterministic input files under ``samples/``.

Everything is derived from fixed seeds so repeated regeneration produces
byte-identical files.  Three scale tiers span the requested range:

* ``small``  : hundreds of samples, tens of dims, 2x extrapolation;
* ``medium`` : ~a thousand samples, mid dims, 8x extrapolation;
* ``large``  : thousands of samples, hundreds of dims, ~32x extrapolation.

Run:  python scripts/generate_inputs.py [--out samples]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pe_diag.common import canonical_json, rng_for  # noqa: E402

MASTER_SEED = 20261121

# (tier id, sample count, dim, train len, target len, extrap ratio)
TIERS = [
    ("small", 300, 32, 256, 512, 2.0),
    ("medium", 500, 96, 512, 4096, 8.0),
    ("large", 800, 192, 512, 16384, 32.0),
]
KEYS_PER_SAMPLE = 5
# representative subset used by the (relatively expensive) sensitivity scan
SENSITIVITY_SAMPLES_PER_TIER = 12


def _features(rng, dim):
    return [round(float(x), 10) for x in rng.normal(0.0, 1.0, size=dim)]


def generate_samples(out: Path) -> list[dict]:
    rows: list[dict] = []
    for tier_index, (tier_id, count, dim, train_len, target_len,
                     ratio) in enumerate(TIERS):
        for local_idx in range(count):
            sample_id = "{}-{:05d}".format(tier_id, local_idx)
            rng = rng_for(MASTER_SEED, tier_id, local_idx)
            # query kept inside the training window
            query_pos = int(rng.integers(0, max(1, train_len // 2)))
            key_positions = []
            for k in range(KEYS_PER_SAMPLE):
                if k < 2:  # first keys inside training window
                    pos = min(target_len,
                              query_pos + int(rng.integers(0, train_len + 1)))
                else:     # remaining keys in the extrapolation region
                    low = min(target_len, query_pos + train_len + 1)
                    pos = int(rng.integers(low, target_len + 1))
                key_positions.append(int(pos))
            rows.append({
                "sample_id": sample_id,
                "tier_id": tier_id,
                "query_pos": query_pos,
                "key_positions": key_positions,
                "query_features": _features(rng, dim),
                "key_features": [_features(rng, dim)
                                 for _ in range(KEYS_PER_SAMPLE)],
            })
    with open(out / "attention_samples.jsonl", "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(canonical_json(row, indent=None))
            fh.write("\n")
    return rows


def generate_config(out: Path) -> dict:
    config = {
        "schemes": ["rope", "linear", "ntk"],
        "global": {
            "base": 10000.0,
            "scale": 1.0,
            "precision_tiers": ["float64", "float32"],
            "chunk_size": 32,
            "workers": 4,
            "snapshot_interval": 128,
            "batch_point_budget": 12000,
            "memory_limit_mb": 64,
            "tolerances": {"position": 1e-6, "score": 1e-6, "weight": 1e-8},
            "sensitivity": {
                "base_grid": [0.0, 1000.0, 5000.0, 10000.0, 20000.0],
                "scale_grid": [0.0, 0.5, 1.0, 2.0, 4.0],
            },
        },
        "scale_profiles": {
            "small": {
                "train_len": 256, "target_len": 512, "dim": 32,
                "scale": 1.0, "chunk_size": 16,
                "batch_point_budget": 12000, "memory_limit_mb": 64,
            },
            "medium": {
                "train_len": 512, "target_len": 4096, "dim": 96,
                "scale": 2.0, "chunk_size": 32,
                "batch_point_budget": 100000, "memory_limit_mb": 0.3,
            },
            "large": {
                "train_len": 512, "target_len": 16384, "dim": 192,
                "scale": 4.0, "chunk_size": 64,
                "batch_point_budget": 100000, "memory_limit_mb": 0.5,
            },
        },
        "paths": {
            "samples": "samples/attention_samples.jsonl",
            "scale": "samples/scale_descriptor.json",
            "injections": "samples/injection_manifest.json",
            "snapshot_dir": "snapshots",
            "report_dir": "reports",
        },
    }
    with open(out / "position_encoding_config.json", "w",
              encoding="utf-8") as fh:
        fh.write(canonical_json(config))
    return config


def generate_scale_descriptor(out: Path) -> dict:
    tiers = []
    for tier_id, count, dim, train_len, target_len, ratio in TIERS:
        tiers.append({
            "tier_id": tier_id,
            "sample_count": count,
            "dim": dim,
            "train_len": train_len,
            "target_len": target_len,
            "extrapolation_ratio": ratio,
            "expected_tradeoff": {
                "small": "full-dimension single-batch, high precision",
                "medium": "reduced-dimension chunked scoring",
                "large": "half-dimension streaming, float32 layering",
            }[tier_id],
        })
    doc = {"master_seed": MASTER_SEED, "tiers": tiers}
    with open(out / "scale_descriptor.json", "w", encoding="utf-8") as fh:
        fh.write(canonical_json(doc))
    return doc


def generate_injections(out: Path, rows: list) -> dict:
    def first(tier):
        return next(r for r in rows if r["tier_id"] == tier)["sample_id"]

    injections = [
        {
            "injection_id": "INJ-001",
            "sample_id": first("small"),
            "stage": "score",
            "kind": "nonfinite_feature",
            "params": {"target": "query", "component": 0, "value": "nan"},
        },
        {
            "injection_id": "INJ-002",
            "sample_id": first("medium"),
            "stage": "score",
            "kind": "key_pos_out_of_range",
            "params": {"key_index": 0, "position": 10 ** 9},
        },
        {
            "injection_id": "INJ-003",
            "sample_id": first("large"),
            "stage": "encode",
            "kind": "base_zero",
            "params": {},
        },
        {
            "injection_id": "INJ-004",
            "sample_id": "{:s}".format(first("large")),
            "stage": "score",
            "kind": "scale_invalid",
            "params": {"value": -1.0},
        },
    ]
    doc = {"injections": injections}
    with open(out / "injection_manifest.json", "w", encoding="utf-8") as fh:
        fh.write(canonical_json(doc))
    return doc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="samples")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = generate_samples(out)
    generate_config(out)
    generate_scale_descriptor(out)
    generate_injections(out, rows)
    print("generated {} samples and 3 config files in {}".format(
        len(rows), out))


if __name__ == "__main__":
    main()
