from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from peft_lora.base import generate_base_package


SAMPLES = ROOT / "samples"
LAYER_SHAPES = {
    "layer_0": (4, 3),
    "layer_1": (3, 4),
    "layer_2": (5, 4),
}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def main() -> None:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    generate_base_package(SAMPLES / "base_v1", LAYER_SHAPES)
    generate_base_package(
        SAMPLES / "base_v2",
        LAYER_SHAPES,
        changed_layers={"layer_1"},
        variant_salt="upgrade-2026-11",
    )

    tasks = {
        "schema_version": 1,
        "tasks": [
            {
                "task_name": "alpha_classify",
                "target_layer": "layer_0",
                "rank": 2,
                "scale": 1.0,
            },
            {
                "task_name": "beta_extract",
                "target_layer": "layer_1",
                "rank": 2,
                "scale": 1.25,
            },
            {
                "task_name": "delta_guard",
                "target_layer": "layer_2",
                "rank": 2,
                "scale": 1.5,
            },
            {
                "task_name": "gamma_rerank",
                "target_layer": "layer_1",
                "rank": 3,
                "scale": 0.75,
            },
        ],
    }
    write_json(SAMPLES / "tasks.json", tasks)

    layer_tokens = {
        "layer_0": [
            [3, 8, 12, 2],
            [5, 5, 21, 34],
            [1, 13, 55, 8],
            [9, 27, 4, 44],
            [16, 2, 63, 19],
        ],
        "layer_1": [
            [7, 14, 2, 31, 6],
            [11, 11, 25, 48],
            [4, 38, 20, 9, 61],
            [0, 17, 42, 33],
            [23, 7, 58, 12, 28],
        ],
        "layer_2": [
            [2, 46, 10, 5],
            [31, 18, 7, 54, 1],
            [13, 13, 39, 22],
            [60, 4, 27, 16, 35],
            [8, 51, 3, 44],
        ],
    }
    instructions: list[dict[str, object]] = []
    for layer_name in sorted(layer_tokens):
        for index, tokens in enumerate(layer_tokens[layer_name], start=1):
            instructions.append(
                {
                    "sample_id": f"{layer_name}-sample-{index:02d}",
                    "sample_key": f"controlled-{layer_name}-{index:02d}",
                    "target_layer": layer_name,
                    "token_ids": tokens,
                }
            )
    write_jsonl(SAMPLES / "instructions.jsonl", instructions)

    config = """schema_version: 1
base_dir: base_v1
target_layer_whitelist: [layer_0, layer_1, layer_2]
adapter:
  rank_candidates: [2, 3]
  scale_candidates: [0.75, 1.0, 1.25, 1.5]
learning_rate: 0.04
epochs: 10
batch_size: 2
quantization:
  scale: 1000000
  tolerance_steps: 1
concurrency: 3
merge_budget:
  weight_elements: 1000000
  compute_units: 50000000
snapshot_commit_interval: 2
"""
    (SAMPLES / "peft_config.yaml").write_text(config, encoding="utf-8")


if __name__ == "__main__":
    main()
