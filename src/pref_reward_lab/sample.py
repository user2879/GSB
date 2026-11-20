from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .canonical import atomic_write_json, atomic_write_text
from .formats import V1Record, V2Record, encode_v1_record, write_v2_file

VOCABULARY = (
    "amber anchor archive atlas beacon birch bridge calm canvas cedar cipher circle "
    "clover cobalt comet coral delta drift ember falcon fern field flint forest forge "
    "glade granite harbor hazel hollow horizon indigo island ivory juniper kernel lagoon "
    "lantern maple meadow mist mosaic nectar north oak ocean onyx orbit orchard pebble "
    "pine prism quartz quiet raven reed ridge river saffron sand signal silver sky "
    "solstice spruce star stone summit thistle tide timber topaz trail valley velvet "
    "violet willow wind winter zephyr"
).split()


def _words(rng: random.Random, low: int, high: int) -> str:
    return " ".join(rng.choice(VOCABULARY) for _ in range(rng.randint(low, high)))


def default_config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "tokenizer": {"version": "subword_v1"},
        "feature_dim": 48,
        "hash_buckets": 128,
        "normalize_text": True,
        "scoring_weights": {
            "prompt": 0.15,
            "chosen": 1.0,
            "rejected": 1.0,
            "length": 0.02,
            "bias": 0.0,
        },
        "reward_delta_threshold": 0.05,
        "concurrency": 4,
        "snapshot_commit_interval": 2,
        "reward_labels": [
            "strong_preference",
            "preference",
            "neutral",
            "weak_preference",
        ],
        "precision_labels": ["milli_exact", "float32", "float64"],
        "v1_default_label": "neutral",
        "source_version": "v2",
        "strategy_tiers": [
            {
                "name": "small",
                "max_pairs": 80,
                "bucket_multiplier": 1,
                "threshold_correction": 0.0,
                "precision_tier": "compact",
                "description": "Up to 80 pairs: base buckets, compact subwords.",
            },
            {
                "name": "medium",
                "max_pairs": 300,
                "bucket_multiplier": 2,
                "threshold_correction": 0.01,
                "precision_tier": "balanced",
                "description": "Up to 300 pairs: 2x buckets, balanced subwords.",
            },
            {
                "name": "large",
                "max_pairs": 1000,
                "bucket_multiplier": 4,
                "threshold_correction": 0.02,
                "precision_tier": "fine",
                "description": "Up to 1000 pairs: 4x buckets, fine subwords.",
            },
        ],
        "forced_strategy_tier": None,
        "metadata": {
            "name": "deterministic-controlled-vocabulary-demo",
            "description": "Fixed-seed sample for interoperability and reproducibility tests.",
        },
    }


def generate_dataset(
    output_dir: str | Path,
    *,
    pair_count: int = 120,
    seed: int = 20260101,
    include_invalid_rows: bool = True,
) -> dict[str, str]:
    if pair_count <= 0:
        raise ValueError("pair_count must be positive")
    if pair_count > 9999:
        raise ValueError("sample generator caps pair_count at 9999 (use 24-character IDs)")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    raw_rows: list[dict[str, Any]] = []
    v1_records: list[V1Record] = []
    v2_records: list[V2Record] = []
    labels = ["strong_preference", "preference", "neutral", "weak_preference"]
    for index in range(1, pair_count + 1):
        pair_id = f"pair-{index:06d}"
        prompt = _words(rng, 5, 12)
        chosen = _words(rng, 9, 22)
        rejected = _words(rng, 9, 22)
        timestamp = f"2026-01-{(index - 1) % 28 + 1:02d}T{(index * 7) % 24:02d}:00:00Z"
        raw_rows.append(
            {
                "pair_id": pair_id,
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "annotator_id": f"ann-{rng.randint(1, 12):02d}",
                "timestamp": timestamp,
            }
        )
        score_milli = rng.randint(350, 900)
        if rng.random() < 0.16:
            delta_milli = rng.randint(8, 50)
        else:
            delta_milli = rng.randint(55, 260)
        v1_records.append(V1Record(pair_id, score_milli, delta_milli))
        score = score_milli / 1000.0
        delta = delta_milli / 1000.0
        if index % 7 == 0:
            score += 0.0004
            delta -= 0.0003
        v2_records.append(
            V2Record(
                pair_id=pair_id,
                reward_score=score,
                reward_delta=delta,
                reward_label=labels[index % len(labels)],
                precision_label="float64" if index % 7 == 0 else "float32",
            )
        )

    raw_text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in raw_rows
    )
    atomic_write_text(output / "preference_pairs.jsonl", raw_text)
    v1_text = "".join(encode_v1_record(record) + "\n" for record in v1_records)
    if include_invalid_rows:
        v1_text += "BROKEN-V1-ROW\n"
    atomic_write_text(output / "exchange_v1.txt", v1_text)
    v2_payload: list[dict[str, Any]] = [
        {
            "pair_id": record.pair_id,
            "reward_score": record.reward_score,
            "reward_delta": record.reward_delta,
            "reward_label": record.reward_label,
            "precision_label": record.precision_label,
        }
        for record in v2_records
    ]
    if include_invalid_rows:
        v2_payload.append(
            {
                "pair_id": "invalid-label-demo",
                "reward_score": 0.5,
                "reward_delta": 0.1,
                "reward_label": "not-in-config-enum",
                "precision_label": "float64",
            }
        )
    atomic_write_text(
        output / "exchange_v2.json",
        json.dumps(v2_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    atomic_write_json(output / "config.json", default_config())
    injections = [
        {
            "id": "INJ-001",
            "pair_id": "pair-000001",
            "stage": "features",
            "kind": "zero_hash_buckets",
        },
        {
            "id": "INJ-002",
            "pair_id": "pair-000002",
            "stage": "scoring",
            "kind": "non_finite_weight",
        },
        {
            "id": "INJ-003",
            "pair_id": "pair-000003",
            "stage": "scoring",
            "kind": "boundary_equal_threshold",
        },
        {
            "id": "INJ-004",
            "pair_id": "pair-000004",
            "stage": "conversion",
            "kind": "missing_sidecar",
        },
    ]
    atomic_write_json(output / "injections.json", {"injections": injections})
    return {
        "raw_pairs": str(output / "preference_pairs.jsonl"),
        "exchange_v1": str(output / "exchange_v1.txt"),
        "exchange_v2": str(output / "exchange_v2.json"),
        "config": str(output / "config.json"),
        "injections": str(output / "injections.json"),
    }


def write_v2_exchange(path: str | Path, records: list[V2Record]) -> None:
    write_v2_file(path, records)

