#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path


SEED = 20261121
ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


CONFIG = {
    "timeline_alignment_ms": 100,
    "feature_dim": 4,
    "normalization": "minmax",
    "channel_weights": {"lexical": 0.5, "semantic": 0.3, "trending": 0.2},
    "fusion_temperature": 1.0,
    "fusion_bias": 0.0,
    "fusion_score_min": 0.0,
    "fusion_score_max": 1.0,
    "dedup_similarity_threshold": 0.6,
    "rerank_window_size": 4,
    "diversity_discount_factor": 0.8,
    "fixed_decimal_places": 4,
    "quantization_levels": 10001,
    "recompute_budget_per_run": 100,
    "thread_pool_workers": 4,
    "snapshot_commit_interval_shards": 2,
    "similarity_weights": {"category": 0.6, "price_band": 0.4},
}

ITEMS = {
    "I001": {"category": "A", "price_band": "high"},
    "I002": {"category": "A", "price_band": "low"},
    "I003": {"category": "B", "price_band": "mid"},
    "I004": {"category": "A", "price_band": "high"},
    "I005": {"category": "A", "price_band": "high"},
    "I006": {"category": "B", "price_band": "high"},
    "I007": {"category": "A", "price_band": "low"},
    "I008": {"category": "C", "price_band": "mid"},
    "I009": {"category": "C", "price_band": "mid"},
    "I010": {"category": "D", "price_band": "high"},
    "I011": {"category": "D", "price_band": "low"},
    "I012": {"category": "E", "price_band": "mid"},
    "I013": {"category": "E", "price_band": "low"},
    "I014": {"category": "F", "price_band": "high"},
    "I015": {"category": "F", "price_band": "high"},
    "I016": {"category": "G", "price_band": "mid"},
    "I017": {"category": "H", "price_band": "high"},
    "I018": {"category": "H", "price_band": "high"},
}


def feature(item_id: str) -> list[float]:
    rng = random.Random(f"{SEED}:{item_id}")
    return [round(rng.uniform(-1.0, 1.0), 6) for _ in range(4)]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_shard(
    path: Path,
    channel: str,
    shard_seq: int,
    start_ms: int,
    end_ms: int,
    rows: list[tuple[str, str, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for request_id, item_id, score in rows:
        lines.append(
            json.dumps(
                {
                    "request_id": request_id,
                    "channel": channel,
                    "item_id": item_id,
                    "score": score,
                    "features": feature(item_id),
                    "shard_seq": shard_seq,
                    "coverage_start_ms": start_ms,
                    "coverage_end_ms": end_ms,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    for directory in ("shards", "plan/step1", "plan/step2", "injections"):
        target = SAMPLES / directory
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)

    write_json(SAMPLES / "rerank_config.json", CONFIG)
    write_json(SAMPLES / "items.json", ITEMS)

    write_shard(
        SAMPLES / "shards" / "lexical" / "seq0.jsonl",
        "lexical",
        0,
        0,
        100,
        [
            ("R001", "I001", 0.90),
            ("R001", "I002", 0.70),
            ("R001", "I002", 0.65),
            ("R002", "I006", 0.80),
            ("R003", "I010", 0.50),
            ("R003", "I011", 0.50),
            ("R004", "I018", 0.50),
        ],
    )
    write_shard(
        SAMPLES / "shards" / "lexical" / "seq1.jsonl",
        "lexical",
        1,
        100,
        200,
        [
            ("R001", "I001", 0.95),
            ("R001", "I003", 0.60),
            ("R002", "I007", 0.80),
            ("R003", "I012", 0.40),
        ],
    )
    write_shard(
        SAMPLES / "shards" / "lexical" / "seq2.jsonl",
        "lexical",
        2,
        200,
        300,
        [
            ("R001", "I005", 0.40),
            ("R002", "I008", 0.60),
            ("R003", "I013", 0.30),
            ("R004", "I017", 1.00),
            ("R004", "I018", 0.50),
        ],
    )
    write_shard(
        SAMPLES / "shards" / "semantic" / "seq0.jsonl",
        "semantic",
        0,
        0,
        100,
        [
            ("R001", "I001", 0.80),
            ("R001", "I001", 0.75),
            ("R001", "I004", 0.70),
            ("R002", "I006", 0.90),
            ("R003", "I010", 0.60),
            ("R003", "I011", 0.60),
            ("R004", "I018", 0.40),
        ],
    )
    write_shard(
        SAMPLES / "shards" / "semantic" / "seq1.jsonl",
        "semantic",
        1,
        100,
        200,
        [
            ("R001", "I002", 0.50),
            ("R001", "I005", 0.65),
            ("R002", "I009", 0.70),
            ("R003", "I014", 0.45),
        ],
    )
    write_shard(
        SAMPLES / "shards" / "trending" / "seq0.jsonl",
        "trending",
        0,
        0,
        200,
        [
            ("R001", "I004", 0.60),
            ("R001", "I006", 0.40),
            ("R002", "I007", 0.50),
            ("R003", "I012", 0.70),
            ("R004", "I018", 0.30),
        ],
    )
    write_shard(
        SAMPLES / "shards" / "trending" / "seq1.jsonl",
        "trending",
        1,
        200,
        300,
        [
            ("R001", "I005", 0.75),
            ("R002", "I009", 0.85),
            ("R003", "I013", 0.60),
            ("R004", "I017", 1.00),
        ],
    )

    write_shard(
        SAMPLES / "plan" / "step1" / "semantic_seq2.jsonl",
        "semantic",
        2,
        200,
        300,
        [
            ("R001", "I003", 0.88),
            ("R001", "I006", 0.55),
            ("R002", "I008", 0.95),
            ("R003", "I015", 0.70),
            ("R004", "I017", 1.00),
        ],
    )
    write_shard(
        SAMPLES / "plan" / "step1" / "trending_seq2.jsonl",
        "trending",
        2,
        300,
        400,
        [
            ("R001", "I007", 0.65),
            ("R002", "I010", 0.55),
            ("R003", "I016", 0.80),
        ],
    )
    write_shard(
        SAMPLES / "plan" / "step2" / "lexical_seq11.jsonl",
        "lexical",
        11,
        100,
        200,
        [
            ("R001", "I001", 0.40),
            ("R001", "I003", 0.30),
            ("R002", "I007", 0.20),
            ("R003", "I012", 0.20),
        ],
    )

    invalid_path = SAMPLES / "injections" / "invalid_shard.jsonl"
    invalid_path.write_text(
        json.dumps(
            {
                "request_id": "R001",
                "channel": "lexical",
                "item_id": "I999",
                "score": "NaN",
                "features": [0.1, 0.2, 0.3, 0.4],
                "shard_seq": 98,
                "coverage_start_ms": 100,
                "coverage_end_ms": 200,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    plan = {
        "version": 1,
        "seed": SEED,
        "steps": [
            {
                "step": 1,
                "name": "append_semantic_and_trending_tail_shards",
                "operations": [
                    {
                        "op": "append",
                        "source": "plan/step1/semantic_seq2.jsonl",
                    },
                    {
                        "op": "append",
                        "source": "plan/step1/trending_seq2.jsonl",
                    },
                ],
            },
            {
                "step": 2,
                "name": "replace_lexical_100_200_and_inject_invalid_shard",
                "operations": [
                    {
                        "op": "replace",
                        "channel": "lexical",
                        "start_ms": 100,
                        "end_ms": 200,
                        "source": "plan/step2/lexical_seq11.jsonl",
                    },
                    {
                        "op": "append",
                        "source": "injections/invalid_shard.jsonl",
                        "injection_id": "INJ-001",
                    },
                ],
            },
            {
                "step": 3,
                "name": "revoke_lexical_head_shard",
                "operations": [
                    {
                        "op": "revoke",
                        "channel": "lexical",
                        "shard_seq": 0,
                    }
                ],
            },
            {
                "step": 4,
                "name": "change_channel_weights_temperature_and_window",
                "operations": [
                    {
                        "op": "update_config",
                        "patch": {
                            "channel_weights": {
                                "lexical": 0.45,
                                "semantic": 0.35,
                                "trending": 0.20,
                            },
                            "rerank_window_size": 3,
                            "diversity_discount_factor": 0.7,
                        },
                    }
                ],
            },
        ],
    }
    write_json(SAMPLES / "roll_plan.json", plan)
    write_json(
        SAMPLES / "injections.json",
        {
            "version": 1,
            "items": [
                {
                    "id": "INJ-001",
                    "type": "invalid_shard",
                    "expected_code": "E_SHARD_INVALID",
                    "enabled": True,
                    "description": "Step 2 appends a shard with a non-finite score string.",
                },
                {
                    "id": "INJ-002",
                    "type": "fusion_clip_boundary",
                    "enabled": True,
                    "description": "R004/I017 is max in all channels and reaches fusion score 1.0 exactly.",
                },
                {
                    "id": "INJ-003",
                    "type": "fusion_tie",
                    "enabled": True,
                    "description": "R003/I010 and R003/I011 have identical channel scores.",
                },
                {
                    "id": "INJ-004",
                    "type": "budget_pending",
                    "enabled": True,
                    "description": "Use --budget 0 or a low budget to retain pending candidates.",
                },
                {
                    "id": "INJ-005",
                    "type": "similarity_threshold_equal",
                    "enabled": True,
                    "description": "Some candidates share only a category and have similarity exactly 0.6.",
                },
                {
                    "id": "INJ-006",
                    "type": "discount",
                    "enabled": True,
                    "description": "Same category and price-band candidates trigger diversity discount.",
                },
                {
                    "id": "INJ-007",
                    "type": "snapshot_stale",
                    "enabled": False,
                    "description": "Enable a dedicated stale-snapshot scenario when testing recovery guards.",
                },
            ],
        },
    )
    write_json(
        SAMPLES / "seed_manifest.json",
        {
            "seed": SEED,
            "python": ">=3.11",
            "deterministic": True,
            "channels": ["lexical", "semantic", "trending"],
            "requests": ["R001", "R002", "R003", "R004"],
            "plan_steps": [1, 2, 3, 4],
        },
    )


if __name__ == "__main__":
    main()
