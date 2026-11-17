"""Canonical deterministic sample declarations shipped with the project."""

from __future__ import annotations

from typing import Any

from .records import (
    build_current_record,
    build_v1_record,
    build_v2_record,
    parameters,
)


def legacy_library_document() -> dict[str, Any]:
    baseline = build_current_record(
        run_id="baseline",
        run_version=1,
        parameters=parameters(
            seed=11,
            model_dim=32,
            learning_rate=0.01,
            steps=12,
            hyperparameters={"dropout": 0.10},
        ),
        aggregation="last",
        timestamp=1_700_000_011,
    )
    tuned = build_current_record(
        run_id="tuned",
        run_version=1,
        parameters=parameters(
            seed=29,
            model_dim=48,
            learning_rate=0.015,
            steps=12,
            hyperparameters={"dropout": 0.05},
        ),
        aggregation="last",
        timestamp=1_700_000_029,
    )
    ensemble = build_current_record(
        run_id="ensemble",
        run_version=1,
        parameters=parameters(
            seed=7,
            model_dim=64,
            learning_rate=0.012,
            steps=12,
            hyperparameters={"momentum": 0.90},
        ),
        aggregation="mean_last_5",
        parent_run_id="baseline",
        timestamp=1_700_000_007,
    )
    return {
        "library_schema": 1,
        "library_id": "local-demo-experiment",
        "runs": [
            build_v1_record(baseline),
            build_v2_record(tuned),
            ensemble,
        ],
        "history": [],
    }


def replacement_v2_document() -> dict[str, Any]:
    return build_current_record(
        run_id="baseline",
        run_version=2,
        parameters=parameters(
            seed=11,
            model_dim=32,
            learning_rate=0.02,
            steps=12,
            hyperparameters={"dropout": 0.08},
        ),
        aggregation="last",
        timestamp=1_700_000_111,
    )


def replacement_v3_document() -> dict[str, Any]:
    return build_current_record(
        run_id="baseline",
        run_version=3,
        parameters=parameters(
            seed=11,
            model_dim=32,
            learning_rate=0.03,
            steps=12,
            hyperparameters={"dropout": 0.08},
        ),
        aggregation="mean_last_5",
        timestamp=1_700_000_211,
    )

