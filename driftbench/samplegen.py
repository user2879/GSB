from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np

from .bucketing import bucket_hash, group_for_bucket
from .models import Config

GENERATOR_SEED = 20261122
EVENT_BASE_MS = 1_760_000_000_000
EVENT_SPACING_MS = 8_500
TOTAL_EVENTS = 504
INITIAL_EVENTS = 480
FILTER_THRESHOLD = EVENT_BASE_MS + 252 * EVENT_SPACING_MS


def _config() -> Config:
    return Config(
        control_group="control",
        experiment_group="treatment",
        bucket_hash_seed="seed-2026-11",
        bucket_count=64,
        active_spec_version="v2",
        fixed_decimal_places=6,
        quantization_step=1,
        concurrency=4,
        batch_size=32,
        snapshot_commit_interval_batches=2,
        bucket_property_key="bucket_key",
        fallback_policy="user_id_only",
        missing_attribute_group=None,
        drift_tolerance=Decimal("0.002"),
        min_affected_ratio=Decimal("0.05"),
        attribution_order=(
            "truncation_rank",
            "normalization",
            "dedup_count",
            "sample_filter",
        ),
        snapshot_version=1,
    )


def generate_sample(root: str | Path) -> dict[str, Any]:
    out = Path(root)
    out.mkdir(parents=True, exist_ok=True)
    config = _config()
    rng = np.random.default_rng(GENERATOR_SEED)

    user_ids = [f"u{index:04d}" for index in range(1, 73)]
    properties: dict[str, dict[str, Any]] = {}
    for index, user_id in enumerate(user_ids):
        if index % 6 == 5:
            continue
        properties[user_id] = {
            "bucket_key": f"segment-{['A', 'B', 'C', 'D'][index % 4]}",
            "region": ["north", "south", "east", "west"][index % 4],
        }

    item_ids = [f"item-{index:03d}" for index in range(1, 65)]
    events: list[dict[str, Any]] = []
    conversion_candidates: list[int] = []
    for index in range(TOTAL_EVENTS):
        # The last 24 events are deliberately split into two edit batches.
        if index >= INITIAL_EVENTS:
            user_index = index % 60
        else:
            user_index = int(rng.integers(0, len(user_ids)))
        user_id = user_ids[user_index]
        item_id = item_ids[int(rng.integers(0, len(item_ids)))]
        rank = int(rng.integers(1, 21))
        score = float(np.round(rng.normal(0.0, 1.0), 6))
        attribute = properties.get(user_id, {}).get(config.bucket_property_key)
        bucket = bucket_hash(config.bucket_hash_seed, user_id, attribute, config.bucket_count)
        group = group_for_bucket(bucket, config)
        event_ms = EVENT_BASE_MS + index * EVENT_SPACING_MS + int(rng.integers(0, 1200))
        late_lift = 0.025 if event_ms >= FILTER_THRESHOLD else 0.0
        experiment_lift = 0.035 if group == config.experiment_group else 0.0
        click_probability = max(
            0.01,
            min(
                0.92,
                0.14
                + late_lift
                + experiment_lift
                + 0.045 * (score / (1.0 + abs(score)))
                - 0.004 * rank,
            ),
        )
        clicked = int(rng.random() < click_probability)
        converted = int(clicked == 1 and rng.random() < 0.28)
        request_id = f"req-{index + 1:06d}"
        events.append(
            {
                "request_id": request_id,
                "experiment_group": group,
                "user_id": user_id,
                "item_id": item_id,
                "rank": rank,
                "model_score": score,
                "clicked": clicked,
                "converted": converted,
                "event_ms": event_ms,
            }
        )
        if clicked:
            conversion_candidates.append(index)

    specs = [
        {
            "spec_version": "v1",
            "metric_name": "ctr",
            "truncation_rank": 10,
            "normalization": "none",
            "dedup_count": False,
            "sample_filter": [],
        },
        {
            "spec_version": "v2",
            "metric_name": "ctr",
            "truncation_rank": 5,
            "normalization": "rank_inverse",
            "dedup_count": False,
            "sample_filter": [],
        },
        {
            "spec_version": "v3",
            "metric_name": "ctr",
            "truncation_rank": 5,
            "normalization": "rank_inverse",
            "dedup_count": True,
            "sample_filter": [
                {"field": "event_ms", "op": "gte", "value": FILTER_THRESHOLD}
            ],
        },
    ]

    fix_index = 37
    fix_target = events[fix_index]
    while fix_target["rank"] <= 5:
        fix_index += 1
        fix_target = events[fix_index]

    edit_plan = {
        "seed": GENERATOR_SEED,
        "batches": [
            {
                "batch": 1,
                "operations": [
                    {"type": "add_exposures", "events": events[480:492]}
                ],
            },
            {
                "batch": 2,
                "operations": [
                    {
                        "type": "fix_exposure",
                        "request_id": fix_target["request_id"],
                        "patch": {"rank": 2},
                    }
                ],
            },
            {
                "batch": 3,
                "operations": [
                    {"type": "switch_spec_version", "spec_version": "v3"}
                ],
            },
            {
                "batch": 4,
                "operations": [
                    {"type": "change_bucket_seed", "seed": "seed-2026-12"}
                ],
            },
            {
                "batch": 5,
                "operations": [
                    {"type": "add_exposures", "events": events[492:504]}
                ],
            },
            {
                "batch": 6,
                "operations": [
                    {"type": "switch_spec_version", "spec_version": "v2"}
                ],
            },
            {
                "batch": 7,
                "operations": [
                    {"type": "switch_spec_version", "spec_version": "v3"}
                ],
            },
        ],
    }

    injections = {
        "injections": [
            {
                "injection_id": "I01",
                "batch": 1,
                "type": "non_finite_model_score",
                "request_id": "inj-nonfinite-001",
            },
            {
                "injection_id": "I02",
                "batch": 2,
                "type": "unknown_group",
                "request_id": "inj-group-001",
            },
            {
                "injection_id": "I03",
                "batch": 3,
                "type": "missing_spec_version",
                "value": "v-missing",
            },
            {
                "injection_id": "I04",
                "batch": 4,
                "type": "invalid_attribution_order",
                "value": ["truncation_rank", "normalization", "dedup_count", "bogus"],
            },
            {
                "injection_id": "I05",
                "batch": 5,
                "type": "zero_tolerance",
                "value": "0",
            },
        ]
    }

    (out / "exposures.jsonl").write_text(
        "".join(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n" for event in events[:INITIAL_EVENTS]),
        encoding="utf-8",
    )
    (out / "specs.jsonl").write_text(
        "".join(json.dumps(spec, ensure_ascii=False, sort_keys=True) + "\n" for spec in specs),
        encoding="utf-8",
    )
    (out / "config.json").write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out / "user_properties.json").write_text(
        json.dumps(
            {"schema_version": 1, "users": properties},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "edit_plan.json").write_text(
        json.dumps(edit_plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out / "injections.json").write_text(
        json.dumps(injections, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "generator_seed": GENERATOR_SEED,
        "initial_event_count": INITIAL_EVENTS,
        "edited_event_count": TOTAL_EVENTS,
        "spec_versions": ["v1", "v2", "v3"],
        "filter_threshold_event_ms": FILTER_THRESHOLD,
        "missing_property_users": len(user_ids) - len(properties),
        "click_count_initial": sum(event["clicked"] for event in events[:INITIAL_EVENTS]),
        "conversion_count_initial": sum(event["converted"] for event in events[:INITIAL_EVENTS]),
    }
    (out / "sample_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata
