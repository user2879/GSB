from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from exp_tracker.records import (
    build_current_record,
    build_v1_record,
    build_v2_record,
    parameters,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "samples" / "legacy_experiment.json"


def sample_document() -> dict[str, Any]:
    return json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))


def mixed_document() -> dict[str, Any]:
    baseline = build_current_record(
        run_id="baseline",
        run_version=1,
        parameters=parameters(
            seed=11,
            model_dim=32,
            learning_rate=0.01,
            steps=8,
            hyperparameters={"dropout": 0.1},
        ),
        timestamp=1_700_000_001,
    )
    tuned = build_current_record(
        run_id="tuned",
        run_version=1,
        parameters=parameters(
            seed=29,
            model_dim=48,
            learning_rate=0.015,
            steps=8,
            hyperparameters={"dropout": 0.05},
        ),
        timestamp=1_700_000_002,
    )
    ensemble = build_current_record(
        run_id="ensemble",
        run_version=1,
        parameters=parameters(
            seed=7,
            model_dim=64,
            learning_rate=0.012,
            steps=8,
            hyperparameters={"momentum": 0.9},
        ),
        aggregation="mean_last_5",
        parent_run_id="baseline",
        timestamp=1_700_000_003,
    )
    return {
        "library_schema": 1,
        "library_id": "test-experiment",
        "runs": [
            build_v1_record(baseline),
            build_v2_record(tuned),
            ensemble,
        ],
        "history": [],
    }


def current_record(
    *,
    run_id: str = "baseline",
    run_version: int = 1,
    seed: int = 11,
    learning_rate: float = 0.01,
    aggregation: str = "last",
    parent_run_id: str | None = None,
    steps: int = 8,
    extra_hyperparameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return build_current_record(
        run_id=run_id,
        run_version=run_version,
        parameters=parameters(
            seed=seed,
            model_dim=32,
            learning_rate=learning_rate,
            steps=steps,
            hyperparameters=extra_hyperparameters,
        ),
        aggregation=aggregation,
        parent_run_id=parent_run_id,
        timestamp=1_700_000_000 + run_version,
    )


def clone(value: Any) -> Any:
    return copy.deepcopy(value)

