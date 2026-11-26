from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


SEED = 20261127
DOMAINS = ("alpha", "beta", "gamma")


def base_config() -> dict[str, Any]:
    return {
        "config_version": "cfg-2026-11-27-v1",
        "domains": [
            {"name": "alpha", "sampling_weight": 0.2, "target_ratio": 0.40, "quota": 40},
            {"name": "beta", "sampling_weight": 0.8, "target_ratio": 0.35, "quota": 40},
            {"name": "gamma", "sampling_weight": 0.6, "target_ratio": 0.25, "quota": 40},
        ],
        "missing_bitmap_width": 6,
        "missing_rules": [
            {
                "code": "MISSING_PRIMARY_FIELD",
                "position": 0,
                "config_field": "missing_rules[primary].position",
            }
        ],
        "cleaning": {"min_total_token_count": 5},
        "dedup": {
            "similarity_threshold_permille": 800,
            "shard_size": 6,
            "score_weights_permille": {
                "token_volume": 300,
                "completeness": 500,
                "domain_weight": 200,
            },
            "score_token_cap": 16,
        },
        "sampling": {"sample_count": 12, "seed": SEED},
        "fixed_point": {"decimal_width": 6, "quantization_tier": "floor"},
        "concurrency": {"max_workers": 4, "cleaning_workers": 4, "dedup_workers": 4},
        "budgets": {
            "global_units": 100_000,
            "domain_quota": 100,
            "shard_comparison_limit": 120,
        },
    }


def _features(prefix: str, count: int) -> dict[str, int]:
    return {f"{prefix}-f{i:02d}": 2 for i in range(count)}


def _sample(
    rng: np.random.Generator,
    domain: str,
    number: int,
    token_counts: dict[str, int],
    *,
    bitmap: list[int] | None = None,
    lengths: list[int] | None = None,
) -> dict[str, Any]:
    sample_id = f"{domain}-s{number:02d}"
    batch_index = 0 if number <= 6 else 1
    return {
        "record_type": "sample",
        "sample_id": sample_id,
        "batch_id": f"{domain}-b{batch_index}",
        "domain": domain,
        "seed": SEED + number * 1009 + DOMAINS.index(domain) * 97,
        "token_counts": token_counts,
        "length_distribution": lengths if lengths is not None else [int(x) for x in rng.integers(1, 7, size=6)],
        "missing_bitmap": bitmap if bitmap is not None else [0, 0, 0, 0, 0, 0],
    }


def _header(domain: str, batch_index: int) -> dict[str, Any]:
    return {
        "record_type": "batch_header",
        "batch_id": f"{domain}-b{batch_index}",
        "generated_from_seed": SEED + DOMAINS.index(domain) * 101 + batch_index,
        "declared_fields": ["token_counts", "length_distribution", "missing_bitmap"],
        "impute_values": {"length_distribution": [3, 3, 4, 4, 5, 5]},
    }


def base_batch_files() -> dict[str, list[dict[str, Any]]]:
    rng = np.random.default_rng(SEED)
    files: dict[str, list[dict[str, Any]]] = {}
    for domain_index, domain in enumerate(DOMAINS):
        rows_by_batch: list[list[dict[str, Any]]] = [[_header(domain, 0)], [_header(domain, 1)]]
        for number in range(1, 13):
            batch_index = 0 if number <= 6 else 1
            counts: dict[str, int]
            bitmap = [0, 0, 0, 0, 0, 0]
            if number == 1:
                counts = _features(f"{domain}-missing", 6)
                bitmap[0] = 1
            elif number == 2:
                counts = {f"{domain}-low-token": 1}
            elif number == 3:
                counts = {f"{domain}-tradeoff-{i:02d}": 8 for i in range(8)}
                bitmap = [0, 1, 1, 1, 0, 0]
            elif number == 4:
                counts = {f"{domain}-tradeoff-{i:02d}": 1 for i in range(8)}
            elif number in (5, 6, 7):
                counts = _features(f"{domain}-cluster-a", 6)
                if number == 7:
                    counts[f"{domain}-cluster-a-extra"] = 1
            elif number == 8:
                counts = _features(f"{domain}-singleton", 7)
            elif number in (9, 10):
                counts = _features(f"{domain}-cluster-b", 6)
            elif number in (11, 12):
                counts = _features(f"{domain}-cluster-c", 6)
            else:
                counts = _features(f"{domain}-extra", 5)

            sample = _sample(rng, domain, number, counts, bitmap=bitmap)
            rows_by_batch[batch_index].append(sample)

        files[f"{domain}-batch-000.jsonl"] = rows_by_batch[0]
        files[f"{domain}-batch-001.jsonl"] = rows_by_batch[1]
    return dict(sorted(files.items()))


def replacement_alpha_s05() -> dict[str, Any]:
    return {
        "record_type": "sample",
        "sample_id": "alpha-s05",
        "batch_id": "alpha-b0",
        "domain": "alpha",
        "seed": SEED + 50001,
        "token_counts": {f"alpha-replacement-unique-{i:02d}": 3 for i in range(8)},
        "length_distribution": [2, 4, 1, 6, 3, 5],
        "missing_bitmap": [0, 0, 1, 0, 0, 0],
    }


def command_stream() -> list[dict[str, Any]]:
    return [
        {
            "command_id": "edit-001-weight-alpha-up",
            "op": "set_weight",
            "domain": "alpha",
            "weight": 0.9,
        },
        {
            "command_id": "edit-002-replace-alpha-s05",
            "op": "replace_sample",
            "sample": replacement_alpha_s05(),
        },
        {
            "command_id": "edit-003-threshold-850",
            "op": "set_threshold",
            "threshold_permille": 850,
        },
        {
            "command_id": "edit-004-shard-size-5",
            "op": "set_shard_size",
            "shard_size": 5,
        },
        {
            "command_id": "edit-005-min-tokens-6",
            "op": "set_min_tokens",
            "min_total_token_count": 6,
        },
    ]


def write_fixture_tree(root: str | Path) -> dict[str, Path]:
    root = Path(root)
    config_path = root / "configs" / "pipeline.json"
    sample_dir = root / "data" / "batches"
    edit_path = root / "data" / "edits" / "command_stream.jsonl"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    edit_path.parent.mkdir(parents=True, exist_ok=True)

    config_path.write_text(json.dumps(base_config(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for filename, rows in base_batch_files().items():
        (sample_dir / filename).write_text(
            "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
            encoding="utf-8",
        )
    edit_path.write_text(
        "\n".join(json.dumps(command, ensure_ascii=False, sort_keys=True) for command in command_stream()) + "\n",
        encoding="utf-8",
    )
    return {"config": config_path, "samples": sample_dir, "edits": edit_path}
