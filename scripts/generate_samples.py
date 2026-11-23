"""Generate deterministic local samples (fixed seed, no network)."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hotblock_diag.tensorpack import write_pack


def generate(sample_dir: str, seq_q: int, seq_k: int, dim: int,
             seed: int = 20261123) -> None:
    os.makedirs(sample_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    query = rng.normal(0.0, 1.0, size=(seq_q, dim)).astype(np.float32)
    key = rng.normal(0.0, 0.8, size=(seq_k, dim)).astype(np.float32)
    value = rng.normal(0.0, 0.6, size=(seq_k, dim)).astype(np.float32)
    write_pack(os.path.join(sample_dir, "tensor_pack.bin"),
               {"query": query, "key": key, "value": value},
               order=["query", "key", "value"])

    prefilter = {
        "q_block": 4,
        "k_block": 4,
        "candidates": 2,
        "score_floor_ratio": 0.55,
        "temperature": 1.25,
        "padding": "last_zero",
    }
    fixedpoint = {
        "frac_bits": 8,
        "quant_gear": "q8",
        "acc_bits": 32,
        "saturation": "saturate",
        "rounding": "nearest_even",
        "hot_capacity": 4,
        "hot_keep": 3,
        "max_rounds": 6,
        "threads": 3,
        "batch_budget": 18,
        "qblock_quota": 6,
        "snapshot_interval": 1,
        "tolerance": 1e-9,
    }
    with open(os.path.join(sample_dir, "prefilter.json"), "w",
              encoding="utf-8", newline="\n") as handle:
        json.dump(prefilter, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with open(os.path.join(sample_dir, "fixedpoint.json"), "w",
              encoding="utf-8", newline="\n") as handle:
        json.dump(fixedpoint, handle, indent=2, sort_keys=True)
        handle.write("\n")

    rng_edit = np.random.default_rng(seed + 1)
    q_replacement = rng_edit.normal(0, 1, prefilter["q_block"] * dim).astype(float).tolist()
    k_replacement = rng_edit.normal(0, 0.8, prefilter["k_block"] * dim).astype(float).tolist()
    commands = [
        {"cmd": "replace_qblock", "index": 1, "data": q_replacement},
        {"cmd": "replace_kblock", "index": 2, "data": k_replacement},
        {"cmd": "update_prefilter",
         "params": {"score_floor_ratio": 0.45, "temperature": 1.1}},
        {"cmd": "set_quant", "quant_gear": "q12"},
        {"cmd": "set_hot_capacity", "capacity": 5},
    ]
    with open(os.path.join(sample_dir, "commands.json"), "w",
              encoding="utf-8", newline="\n") as handle:
        json.dump(commands, handle, indent=2, sort_keys=True)
        handle.write("\n")

    injections = {
        "injections": [
            {"id": "inj-nonfinite-001", "kind": "nonfinite_scale",
             "target": {"qb": 0, "kb": 1, "value": "nan"}},
            {"id": "inj-overflow-002", "kind": "force_overflow",
             "target": {"qb": 2, "kb": 0}},
            {"id": "inj-truncate-003", "kind": "truncate_pack",
             "target": {"drop_bytes": 16}},
            {"id": "inj-tamper-004", "kind": "tamper_snapshot", "target": {}},
        ]
    }
    with open(os.path.join(sample_dir, "injections.json"), "w",
              encoding="utf-8", newline="\n") as handle:
        json.dump(injections, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"samples written to {sample_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--seq-q", type=int, default=16)
    parser.add_argument("--seq-k", type=int, default=16)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261123)
    args = parser.parse_args()
    generate(args.sample_dir, args.seq_q, args.seq_k, args.dim, args.seed)


if __name__ == "__main__":
    main()
