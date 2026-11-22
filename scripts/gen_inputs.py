#!/usr/bin/env python3
"""确定性生成逐列标定配置、样本 JSONL 与编辑计划 (离线, 固定种子)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cqdiag.config import validate_config
from cqdiag.samples import generate_samples, write_samples_jsonl, SAMPLE_GENERATOR_SEED


def build_default_config() -> dict:
    return {
        "tensors": [
            {"name": "weights_a", "shape": [48, 12], "column_axis": 1},
            {"name": "weights_b", "shape": [6, 8, 8], "column_axis": 2},
        ],
        "quantile_lo_permille": 20,
        "quantile_hi_permille": 980,
        "scheme": "asymmetric",
        "quant_bits_candidates": [4, 8],
        "quant_bits": 8,
        "frac_bits": 8,
        "saturation": "clamp",
        "abs_tol": 0.5,
        "rel_tol": 0.25,
        "groups": {
            "weights_a": [i % 3 for i in range(12)],
            "weights_b": [0, 0, 1, 1, 2, 2, 1, 0],
        },
        "group_oor_limit": 0.35,
        "max_workers": 4,
        "element_budget": 100000,
    }


def build_default_edits() -> dict:
    return {
        "batches": [
            {"ops": [
                {"op": "replace_column", "tensor": "weights_a", "column": 3,
                 "seed": 777, "scale": 5.0},
                {"op": "set_quantile", "lo_permille": 5, "hi_permille": 995},
            ]},
            {"ops": [
                {"op": "set_quant_bits", "bits": 4},
                {"op": "set_frac_bits", "bits": 6},
            ]},
            {"ops": [
                {"op": "set_saturation", "strategy": "mask"},
                {"op": "set_groups", "tensor": "weights_a",
                 "labels": [0, 1, 0, 1, 2, 2, 1, 0, 2, 1, 0, 2]},
                {"op": "replace_column", "tensor": "weights_b", "column": 5,
                 "seed": 4242, "scale": 9.0},
            ]},
            {"ops": [
                {"op": "set_quant_bits", "bits": 8},
                {"op": "set_saturation", "strategy": "clamp"},
                {"op": "set_tolerances", "abs_tol": 0.4, "rel_tol": 0.2},
            ]},
        ]
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seed", type=int, default=SAMPLE_GENERATOR_SEED)
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    config_raw = build_default_config()
    config = validate_config(config_raw)
    (outdir / "config.json").write_text(
        json.dumps(config_raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    records = generate_samples(config, seed=args.seed)
    write_samples_jsonl(records, str(outdir / "samples.jsonl"))
    (outdir / "edits.json").write_text(
        json.dumps(build_default_edits(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(f"generated config.json, samples.jsonl, edits.json in {outdir} (seed={args.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
