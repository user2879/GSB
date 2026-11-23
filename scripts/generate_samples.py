from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def build_config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "global_version": 1,
        "schemes": ["rope", "linear_interpolation", "ntk_aware"],
        "base": 10000.0,
        "training_context_length": 16,
        "extrapolation_target_length": 96,
        "dimension": 32,
        "scaling_factor": 4.0,
        "precision": "float64",
        "quantization": "none",
        "dimension_chunk_size": 32,
        "point_budget": 100000,
        "concurrency": 4,
        "snapshot_interval": 5,
        "error_tolerances": {
            "position_vector": 0.01,
            "score": 0.1,
            "attention": 0.001,
        },
        "sensitivity": {
            "base_step": 2500.0,
            "scale_step": 0.5,
            "radius": 2,
        },
    }


def build_pruning_policies() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "policies": [
            {
                "policy_id": "full",
                "description": "Keep every RoPE pair.",
                "pair_ranges": [{"start_ratio": 0.0, "end_ratio": 1.0}],
                "pair_stride": 1,
            },
            {
                "policy_id": "balanced_75",
                "description": "Keep the first 75% of RoPE pairs.",
                "pair_ranges": [{"start_ratio": 0.0, "end_ratio": 0.75}],
                "pair_stride": 1,
            },
            {
                "policy_id": "low_rank_50",
                "description": "Keep the first half of RoPE pairs.",
                "pair_ranges": [{"start_ratio": 0.0, "end_ratio": 0.5}],
                "pair_stride": 1,
            },
            {
                "policy_id": "even_pairs",
                "description": "Keep every second RoPE pair.",
                "pair_ranges": [{"start_ratio": 0.0, "end_ratio": 1.0}],
                "pair_stride": 2,
            },
        ],
    }


def build_injections() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "injections": [
            {
                "injection_id": "INJ-SAMPLE-SCORING",
                "enabled": True,
                "scheme": "linear_interpolation",
                "sample_id": "s0006",
                "stage": "scoring",
                "action": "raise",
                "message": "deterministic scoring fault for isolation test",
            },
        ],
    }


def build_samples(*, count: int, dimension: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    samples: list[dict[str, Any]] = []
    for index in range(count):
        query_position = 22 + index * 4
        start = max(0, query_position - 40)
        positions = np.linspace(start, query_position, num=12, dtype=np.int64).tolist()
        positions = sorted(set(int(value) for value in positions))
        query_features = rng.normal(0.0, 1.0, size=dimension)
        key_features = rng.normal(0.0, 1.0, size=(len(positions), dimension))
        samples.append(
            {
                "sample_id": f"s{index + 1:04d}",
                "query_position": query_position,
                "key_positions": positions,
                "query_features": [float(value) for value in query_features.tolist()],
                "key_features": [
                    [float(value) for value in row] for row in key_features.tolist()
                ],
            }
        )
    return samples


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic local diagnostic samples.")
    parser.add_argument("--output-dir", type=Path, default=Path("sample_data"))
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261124)
    args = parser.parse_args()

    config = build_config()
    instructions = [
        "python scripts/generate_samples.py --output-dir sample_data --count 8 --seed 20261124",
        "python scripts/selfcheck.py --workspace .",
        "python -m rope_diag --workspace .",
    ]
    write_json(args.output_dir / "config.json", config)
    write_json(args.output_dir / "pruning_strategies.json", build_pruning_policies())
    write_json(args.output_dir / "injections.json", build_injections())
    write_json(args.output_dir / "generation_manifest.json", {"seed": args.seed, "count": args.count, "commands": instructions})
    samples = build_samples(
        count=args.count, dimension=int(config["dimension"]), seed=args.seed
    )
    sample_path = args.output_dir / "samples.jsonl"
    with sample_path.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
