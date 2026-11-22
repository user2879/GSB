"""Generate the local deterministic sample directory.

Creates, under the target directory:
  base_pack/      layers.json + per-layer weight/bias binary files (float64 LE)
  tasks.json      task manifest (name, target layer, rank, alpha)
  samples.jsonl   instruction samples (id, key, target layer, token sequence)
  peft_config.yaml
  upgrade_plan.json
  teacher.json    synthetic teacher deltas used to build sample targets
Everything derives from a fixed seed; no network, no real model weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np
import yaml

LAYER_SHAPES = {
    "encoder.layer0.mlp": (8, 6),
    "encoder.layer1.mlp": (8, 8),
    "decoder.layer0.attn": (6, 8),
}
BASE_SEED = 20240501
TEACHER_SCALE = 2.0
VOCAB_SIZE = 64


def _seeded_rng(*parts) -> np.random.Generator:
    material = "|".join(str(p) for p in parts)
    seed = int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "little")
    return np.random.default_rng(seed)


def gen_base_pack(pack_dir: str) -> None:
    os.makedirs(pack_dir, exist_ok=True)
    entries = []
    for idx, (name, shape) in enumerate(sorted(LAYER_SHAPES.items())):
        rng = _seeded_rng("base", BASE_SEED, name)
        weight = rng.normal(0.0, 0.5, size=shape)
        bias = rng.normal(0.0, 0.1, size=(shape[0],))
        w_file, b_file = f"{name}.weight.bin", f"{name}.bias.bin"
        with open(os.path.join(pack_dir, w_file), "wb") as fh:
            fh.write(np.ascontiguousarray(weight, dtype="<f8").tobytes())
        with open(os.path.join(pack_dir, b_file), "wb") as fh:
            fh.write(np.ascontiguousarray(bias, dtype="<f8").tobytes())
        entries.append(
            {
                "name": name,
                "shape": list(shape),
                "dtype": "float64",
                "weight_file": w_file,
                "bias_file": b_file,
            }
        )
    with open(os.path.join(pack_dir, "layers.json"), "w", encoding="utf-8") as fh:
        json.dump({"layers": entries}, fh, indent=2, sort_keys=True)


def gen_teacher(path: str) -> None:
    layers = {}
    for name, shape in sorted(LAYER_SHAPES.items()):
        rng = _seeded_rng("teacher", name)
        rank = 2
        layers[name] = {
            "A": rng.normal(0.0, 0.3, size=(rank, shape[1])).tolist(),
            "B": rng.normal(0.0, 0.3, size=(shape[0], rank)).tolist(),
            "scale": TEACHER_SCALE,
        }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"layers": layers}, fh, indent=2, sort_keys=True)


def gen_tasks(path: str) -> None:
    tasks = [
        {"name": "task_alpha", "target_layer": "encoder.layer0.mlp", "rank": 2, "alpha": 1.0},
        {"name": "task_beta", "target_layer": "encoder.layer0.mlp", "rank": 3, "alpha": 2.0},
        {"name": "task_gamma", "target_layer": "encoder.layer1.mlp", "rank": 2, "alpha": 1.5},
        {"name": "task_delta", "target_layer": "decoder.layer0.attn", "rank": 2, "alpha": 1.0},
    ]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"tasks": tasks}, fh, indent=2, sort_keys=True)


def gen_samples(path: str) -> None:
    rng = _seeded_rng("samples", BASE_SEED)
    layers = sorted(LAYER_SHAPES)
    lines = []
    for i in range(12):
        layer = layers[i % len(layers)]
        n_tokens = int(rng.integers(4, 10))
        tokens = [int(t) for t in rng.integers(0, VOCAB_SIZE, size=n_tokens)]
        lines.append(
            json.dumps(
                {
                    "sample_id": f"s{i:03d}",
                    "sample_key": f"key-{i:03d}",
                    "target_layer": layer,
                    "tokens": tokens,
                },
                sort_keys=True,
            )
        )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def gen_config(path: str) -> None:
    cfg = {
        "target_layers": sorted(LAYER_SHAPES),
        "learning_rate": 3.0,
        "epochs": 8,
        "frac_bits": 12,
        "quant_tier": 1,
        "atol": 5e-3,
        "rtol": 5e-2,
        "max_workers": 4,
        "merge_budget": 100000,
        "snapshot_interval": 1,
    }
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=True)


def gen_upgrade_plan(path: str) -> None:
    plan = {
        "steps": [
            {
                "step_id": 1,
                "description": "replace encoder.layer0.mlp weights",
                "replacements": [
                    {
                        "layer": "encoder.layer0.mlp",
                        "seed": 910001,
                        "shape": list(LAYER_SHAPES["encoder.layer0.mlp"]),
                    }
                ],
            },
            {
                "step_id": 2,
                "description": "replace decoder.layer0.attn weights",
                "replacements": [
                    {
                        "layer": "decoder.layer0.attn",
                        "seed": 910002,
                        "shape": list(LAYER_SHAPES["decoder.layer0.attn"]),
                    }
                ],
            },
        ]
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(plan, fh, indent=2, sort_keys=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    gen_base_pack(os.path.join(args.out_dir, "base_pack"))
    gen_teacher(os.path.join(args.out_dir, "teacher.json"))
    gen_tasks(os.path.join(args.out_dir, "tasks.json"))
    gen_samples(os.path.join(args.out_dir, "samples.jsonl"))
    gen_config(os.path.join(args.out_dir, "peft_config.yaml"))
    gen_upgrade_plan(os.path.join(args.out_dir, "upgrade_plan.json"))
    print(f"sample directory generated at {args.out_dir}")


if __name__ == "__main__":
    main()
