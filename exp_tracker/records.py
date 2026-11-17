"""Helpers for producing canonical declarations used by samples and demos."""

from __future__ import annotations

from typing import Any

from .metrics import aggregate_metrics, generate_step_metrics
from .models import AGGREGATION_VERSION, Aggregation, Parameters


def parameters(
    *,
    seed: int,
    model_dim: int,
    learning_rate: float,
    steps: int,
    hyperparameters: dict[str, Any] | None = None,
) -> Parameters:
    values = dict(hyperparameters or {})
    values["steps"] = steps
    return Parameters(
        seed=seed,
        model_dim=model_dim,
        learning_rate=learning_rate,
        hyperparameters=values,
    )


def build_current_record(
    *,
    run_id: str,
    run_version: int,
    parameters: Parameters,
    aggregation: str = "last",
    parent_run_id: str | None = None,
    timestamp: int | None = None,
) -> dict[str, Any]:
    points = generate_step_metrics(parameters)
    aggregation_value = Aggregation(aggregation, AGGREGATION_VERSION)
    final_metrics = aggregate_metrics(points, aggregation_value)
    return {
        "format_version": 3,
        "run_id": run_id,
        "run_version": run_version,
        "parameters": parameters.to_dict(),
        "step_metrics": [point.to_dict() for point in points],
        "timestamp": timestamp if timestamp is not None else 1_710_000_000 + run_version,
        "aggregation": aggregation_value.to_dict(),
        "parent_run_id": parent_run_id,
        "final_metrics": final_metrics,
    }


def build_v2_record(current_record: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "format_version",
        "run_id",
        "run_version",
        "parameters",
        "final_metrics",
        "step_metrics",
        "timestamp",
    }
    result = {key: current_record[key] for key in current_record if key in keys}
    result["format_version"] = 2
    return result


def build_v1_record(current_record: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "format_version",
        "run_id",
        "run_version",
        "parameters",
        "final_metrics",
    }
    result = {key: current_record[key] for key in current_record if key in keys}
    result["format_version"] = 1
    return result

