from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


SEED = 20261124


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _feature_config() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "seed": SEED,
        "model": {"hidden_dim": 4},
        "quantization": {"tier": "Q16", "fractional_bits": 8},
        "features": [
            {"name": "price", "kind": "dense", "binning": "equal_width", "bins": 5},
            {"name": "dwell", "kind": "dense", "binning": "equal_frequency", "bins": 4},
            {"name": "quality", "kind": "dense", "binning": "equal_width", "bins": 4},
            {"name": "category", "kind": "sparse"},
            {"name": "brand", "kind": "sparse"},
            {"name": "device", "kind": "sparse"},
        ],
        "crosses": [
            {"name": "category_x_brand", "left": "category", "right": "brand"},
        ],
        "shared_features": ["price", "dwell", "category", "category_x_brand"],
        "tower_features": {
            "ctr": ["quality", "device", "category"],
            "cvr": ["price", "brand", "category_x_brand"],
        },
        "fusion": {"mode": "weighted_sum", "ctr_weight": 0.65, "cvr_weight": 0.35},
    }


def _training_config() -> dict[str, Any]:
    return {
        "seed": 71029,
        "learning_rate": 0.08,
        "batch_size": 4,
        "epochs": 2,
        "positive_weight": 2.0,
        "negative_weight": 1.0,
        "conversion_weight": 1.5,
        "conversion_delay_cutoff_ms": 6000,
        "global_train_sample_budget": 24,
        "per_batch_sample_quota": 4,
        "thread_pool_workers": 4,
        "snapshot_commit_interval_batches": 2,
        "initial_event_time_ms": 0,
        "tolerances": {
            "feature_quantization": 0.18,
            "shared_hidden": 0.12,
            "ctr_tower": 0.10,
            "cvr_tower": 0.10,
            "fusion": 0.08,
        },
    }


def _make_batch(rng: random.Random, start_request: int, count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dwell_values = [0.5, 1.5, 3.0, 5.0, 8.0, 13.0, 21.0, 30.0]
    for offset in range(count):
        request_number = start_request + offset // 2
        candidate = offset % 2
        request_id = f"R{request_number:04d}"
        item_id = f"I{request_number:04d}_{candidate}"
        price = round(rng.uniform(0.0, 100.0), 3)
        dwell = rng.choice(dwell_values)
        quality = round(rng.uniform(0.0, 1.0), 3)
        category = rng.randint(0, 3)
        brand = rng.randint(0, 2)
        device = rng.randint(0, 2)
        latent = 0.48 * quality + 0.22 * min(dwell / 20.0, 1.0) + 0.16 * (price / 100.0)
        latent += 0.04 * category + 0.03 * brand - 0.04 * device
        click = int(rng.random() < 1.0 / (1.0 + math.exp(-(3.2 * (latent - 0.35)))))
        conversion = int(click == 1 and rng.random() < 0.48)
        delay = rng.choice([0, 1200, 4000, 9000, None])
        rows.append(
            {
                "request_id": request_id,
                "item_id": item_id,
                "dense": {"price": price, "dwell": dwell, "quality": quality},
                "sparse_counts": {"category": category, "brand": brand, "device": device},
                "click": click,
                "conversion": conversion,
                "conversion_delay_ms": delay,
            }
        )
    return rows


def generate(output_dir: Path) -> dict[str, Any]:
    rng = random.Random(SEED)
    base = _make_batch(rng, 1, 8)
    batch_1 = _make_batch(rng, 5, 6)
    batch_2 = _make_batch(rng, 8, 6)
    batch_3 = _make_batch(rng, 11, 6)
    batch_4 = _make_batch(rng, 14, 6)
    commands = [
        {"type": "append_batch", "batch_id": "b0001", "event_time_ms": 1000, "samples": batch_1},
        {
            "type": "correct_label",
            "request_id": "R0001",
            "item_id": "I0001_0",
            "click": 1,
            "conversion": 1,
            "conversion_delay_ms": 0,
            "event_time_ms": 1000,
        },
        {"type": "append_batch", "batch_id": "b0002", "event_time_ms": 3000, "samples": batch_2},
        {
            "type": "modify_bin_edges",
            "feature": "price",
            "edges": [0, 10, 30, 55, 80, 120],
            "event_time_ms": 3000,
        },
        {
            "type": "switch_quantization",
            "tier": "Q12",
            "fractional_bits": 6,
            "event_time_ms": 3000,
        },
        {"type": "set_conversion_cutoff", "cutoff_ms": 2500, "event_time_ms": 3000},
        {"type": "append_batch", "batch_id": "b0003", "event_time_ms": 7000, "samples": batch_3},
        {"type": "switch_quantization", "tier": "Q8", "fractional_bits": 4, "event_time_ms": 7000},
        {
            "type": "correct_label",
            "request_id": "R0002",
            "item_id": "I0002_0",
            "click": 1,
            "conversion": 1,
            "conversion_delay_ms": 0,
            "event_time_ms": 7000,
        },
        {"type": "append_batch", "batch_id": "b0004", "event_time_ms": 10000, "samples": batch_4},
        {"type": "switch_quantization", "tier": "Q24", "fractional_bits": 10, "event_time_ms": 10000},
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "exposure_samples.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in base),
        encoding="utf-8",
    )
    _write_json(output_dir / "feature_config.json", _feature_config())
    train = _training_config()
    yaml_lines = [
        "seed: {seed}",
        "learning_rate: {learning_rate}",
        "batch_size: {batch_size}",
        "epochs: {epochs}",
        "positive_weight: {positive_weight}",
        "negative_weight: {negative_weight}",
        "conversion_weight: {conversion_weight}",
        "conversion_delay_cutoff_ms: {conversion_delay_cutoff_ms}",
        "global_train_sample_budget: {global_train_sample_budget}",
        "per_batch_sample_quota: {per_batch_sample_quota}",
        "thread_pool_workers: {thread_pool_workers}",
        "snapshot_commit_interval_batches: {snapshot_commit_interval_batches}",
        "initial_event_time_ms: {initial_event_time_ms}",
        "tolerances:",
    ]
    for stage, value in train["tolerances"].items():
        yaml_lines.append(f"  {stage}: {value}")
    (output_dir / "train_config.yaml").write_text(
        "\n".join(line.format(**train) for line in yaml_lines) + "\n",
        encoding="utf-8",
    )
    _write_json(output_dir / "commands.json", {"schema_version": "1.0", "commands": commands})
    manifest = {
        "seed": SEED,
        "base_samples": len(base),
        "appended_samples": len(batch_1) + len(batch_2) + len(batch_3) + len(batch_4),
        "total_samples": len(base) + len(batch_1) + len(batch_2) + len(batch_3) + len(batch_4),
        "commands": len(commands),
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic CTR/CVR sample fixtures")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "samples"),
        help="output samples directory",
    )
    args = parser.parse_args()
    manifest = generate(Path(args.output))
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
