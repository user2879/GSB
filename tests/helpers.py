from __future__ import annotations

from typing import Any

from ctr_cvr_pipeline.cache import ArtifactCache
from ctr_cvr_pipeline.config import FeatureConfig, TrainingConfig, feature_config_from_dict, training_config_from_dict
from ctr_cvr_pipeline.samples import validate_sample_record
from ctr_cvr_pipeline.state import PipelineState, initial_state


def feature_config_dict(tier: str = "Q16", bits: int = 8) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "seed": 9917,
        "model": {"hidden_dim": 3},
        "quantization": {"tier": tier, "fractional_bits": bits},
        "features": [
            {"name": "price", "kind": "dense", "binning": "equal_width", "bins": 3},
            {"name": "dwell", "kind": "dense", "binning": "equal_frequency", "bins": 2},
            {"name": "category", "kind": "sparse"},
            {"name": "brand", "kind": "sparse"},
        ],
        "crosses": [
            {"name": "category_x_brand", "left": "category", "right": "brand"},
        ],
        "shared_features": ["price", "dwell", "category_x_brand"],
        "tower_features": {
            "ctr": ["category", "brand"],
            "cvr": ["price", "category_x_brand"],
        },
        "fusion": {"mode": "weighted_sum", "ctr_weight": 0.6, "cvr_weight": 0.4},
    }


def training_config_dict(**overrides: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "seed": 771,
        "learning_rate": 0.1,
        "batch_size": 2,
        "epochs": 2,
        "positive_weight": 2.0,
        "negative_weight": 1.0,
        "conversion_weight": 1.5,
        "conversion_delay_cutoff_ms": 5000,
        "global_train_sample_budget": 8,
        "per_batch_sample_quota": 2,
        "thread_pool_workers": 2,
        "snapshot_commit_interval_batches": 1,
        "initial_event_time_ms": 0,
        "tolerances": {
            "feature_quantization": 0.2,
            "shared_hidden": 0.2,
            "ctr_tower": 0.15,
            "cvr_tower": 0.15,
            "fusion": 0.1,
        },
    }
    raw.update(overrides)
    return raw


def make_raw(
    request_number: int,
    candidate: int,
    *,
    price: float | None = None,
    dwell: float | None = None,
    category: int = 1,
    brand: int = 1,
    click: int = 0,
    conversion: int = 0,
    delay: int | None = 1000,
) -> dict[str, Any]:
    return {
        "request_id": f"R{request_number:04d}",
        "item_id": f"I{request_number:04d}_{candidate}",
        "dense": {
            "price": price if price is not None else float(10 + request_number * 7 + candidate * 3),
            "dwell": dwell if dwell is not None else float(1 + (request_number + candidate) % 7),
        },
        "sparse_counts": {"category": category, "brand": brand},
        "click": click,
        "conversion": conversion,
        "conversion_delay_ms": delay,
    }


def build_state(
    raws: list[dict[str, Any]] | None = None,
    *,
    feature_config: FeatureConfig | None = None,
    training_config: TrainingConfig | None = None,
) -> tuple[PipelineState, FeatureConfig, TrainingConfig]:
    feature_config = feature_config or feature_config_from_dict(feature_config_dict())
    training_config = training_config or training_config_from_dict(training_config_dict())
    raws = raws or [
        make_raw(1, 0, click=1, conversion=1, delay=0),
        make_raw(1, 1, click=0, conversion=0, delay=4000),
        make_raw(2, 0, click=1, conversion=1, delay=2000),
        make_raw(2, 1, click=1, conversion=0, delay=0),
        make_raw(3, 0, click=0, conversion=1, delay=9000),
        make_raw(3, 1, click=1, conversion=1, delay=1000),
    ]
    seen: set[tuple[str, str]] = set()
    samples = []
    failures = []
    for raw in raws:
        sample, problems = validate_sample_record(
            raw,
            feature_config,
            event_time_ms=training_config.initial_event_time_ms,
            batch_id="b0000",
            seen_keys=seen,
        )
        failures.extend(problems)
        if sample is not None:
            samples.append(sample)
            seen.add(sample.key)
    state = initial_state(samples, feature_config, training_config, failures)
    return state, feature_config, training_config
