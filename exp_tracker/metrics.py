"""Deterministic NumPy metric generation, aggregation and replay comparisons."""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np

from .errors import MetricError, ReplayError
from .models import Aggregation, MetricPoint, Parameters, RunRecord
from .util import sha256_json

METRIC_ALGORITHM_VERSION = "metrics-v1"


def _hyperparameter_factor(hyperparameters: dict[str, Any]) -> float:
    total = 0.0
    index = 0
    for key in sorted(hyperparameters):
        value = hyperparameters[key]
        if key == "steps" or isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        index += 1
        numeric = float(value)
        total += numeric / (1.0 + abs(numeric)) / index
    return total


def generate_step_metrics(parameters: Parameters) -> tuple[MetricPoint, ...]:
    """Generate the canonical step series for a parameter set.

    NumPy's PCG64 uniform stream and vectorized float64 operations are used
    without libm transcendental functions. This keeps repeated generation
    bit-for-bit identical and avoids platform-specific exp/sin/cos rounding.
    """

    steps = parameters.steps
    if steps < 1 or steps > 100_000:
        raise ValueError("hyperparameters.steps must be between 1 and 100000")

    seed_key = int(parameters.seed) & ((1 << 64) - 1)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed_key])))
    step_numbers = np.arange(1, steps + 1, dtype=np.float64)
    progress = step_numbers / float(steps)
    integer_steps = step_numbers.astype(np.int64)

    phase = float(parameters.seed % 10007) / 10007.0
    dimension_root = math.sqrt(float(parameters.model_dim))
    dimension_scale = 0.90 + 0.18 * dimension_root / (dimension_root + 8.0)
    learning_scale = min(max(float(parameters.learning_rate), 1e-8), 1.0)
    hyper_scale = _hyperparameter_factor(parameters.hyperparameters)
    decay = 0.028 + 0.42 * learning_scale + 0.012 * hyper_scale

    loss_noise = rng.random(steps) - 0.5
    accuracy_noise = rng.random(steps) - 0.5
    integer_phase = int(parameters.seed % 10007)
    loss_wave = (
        ((integer_steps * 37 + integer_phase) % 97).astype(np.float64) / 96.0 - 0.5
    )
    accuracy_wave = (
        ((integer_steps * 19 + integer_phase * 3) % 89).astype(np.float64) / 88.0
        - 0.5
    )
    loss = (
        0.22
        + dimension_scale * (1.0 / (1.0 + decay * step_numbers))
        + 0.035 * loss_wave
        + 0.0060 * loss_noise
    )
    accuracy = (
        0.50
        + 0.42 * (1.0 - 1.0 / (1.0 + 1.15 * decay * step_numbers))
        + 0.020 * accuracy_wave
        + 0.0035 * accuracy_noise
    )
    loss = np.maximum(loss, 1e-9)
    accuracy = np.clip(accuracy, 0.0, 0.999999)

    # progress is deliberately part of the formula so optimizer-like schedules
    # remain visible in deterministic samples. phase is retained for the small
    # optional learning-rate offset below.
    loss = loss * (1.0 - 0.015 * progress * max(hyper_scale, 0.0))
    accuracy = np.clip(
        accuracy
        + 0.006 * progress * max(hyper_scale, 0.0)
        + 0.0005 * phase,
        0.0,
        0.999999,
    )

    return tuple(
        MetricPoint(
            step=index + 1,
            loss=float(loss[index]),
            accuracy=float(accuracy[index]),
        )
        for index in range(steps)
    )


def aggregate_metrics(
    points: Sequence[MetricPoint], aggregation: Aggregation | str
) -> dict[str, float]:
    name = aggregation.name if isinstance(aggregation, Aggregation) else aggregation
    if not points:
        raise ValueError("cannot aggregate an empty metric series")
    if name == "last":
        selected = points[-1:]
    elif name == "mean_last_5":
        selected = points[-5:]
    elif name == "mean":
        selected = points
    else:
        raise ValueError(f"unknown aggregation: {name}")
    count = float(len(selected))
    return {
        "loss": float(sum(point.loss for point in selected) / count),
        "accuracy": float(sum(point.accuracy for point in selected) / count),
    }


def first_metric_difference(
    actual: Sequence[MetricPoint], expected: Sequence[MetricPoint]
) -> dict[str, Any] | None:
    common = min(len(actual), len(expected))
    for index in range(common):
        left = actual[index]
        right = expected[index]
        if left.step != right.step:
            return {
                "step": right.step,
                "field": "step",
                "expected": right.step,
                "actual": left.step,
            }
        if left.loss != right.loss:
            return {
                "step": right.step,
                "field": "loss",
                "expected": right.loss,
                "actual": left.loss,
            }
        if left.accuracy != right.accuracy:
            return {
                "step": right.step,
                "field": "accuracy",
                "expected": right.accuracy,
                "actual": left.accuracy,
            }
    if len(actual) != len(expected):
        return {
            "step": common + 1,
            "field": "step_metrics",
            "expected": len(expected),
            "actual": len(actual),
        }
    return None


def first_final_difference(
    actual: dict[str, float], expected: dict[str, float], step: int
) -> dict[str, Any] | None:
    for field in ("loss", "accuracy"):
        if actual.get(field) != expected.get(field):
            return {
                "step": step,
                "field": f"final_metrics.{field}",
                "expected": expected.get(field),
                "actual": actual.get(field),
            }
    return None


def raise_for_metric_difference(
    run_id: str,
    difference: dict[str, Any],
    *,
    path: str | None = None,
) -> None:
    details: dict[str, Any] = {
        "expected": difference.get("expected"),
        "actual": difference.get("actual"),
    }
    if path is not None:
        details["path"] = path
    raise MetricError(
        run_id=run_id,
        step=int(difference["step"]),
        field=str(difference["field"]),
        reason="deterministic metric sequence mismatch",
        **details,
    )


def verify_generated_series(record: RunRecord, *, path: str | None = None) -> None:
    expected = generate_step_metrics(record.parameters)
    difference = first_metric_difference(record.step_metrics, expected)
    if difference is not None:
        raise_for_metric_difference(record.run_id, difference, path=path)
    expected_final = aggregate_metrics(expected, record.aggregation)
    final_difference = first_final_difference(
        record.final_metrics, expected_final, len(expected)
    )
    if final_difference is not None:
        raise_for_metric_difference(record.run_id, final_difference, path=path)


def metric_digest(
    points: Iterable[MetricPoint],
    aggregation: Aggregation,
    final_metrics: dict[str, float],
) -> str:
    payload = {
        "algorithm_version": METRIC_ALGORITHM_VERSION,
        "aggregation": aggregation.to_dict(),
        "step_metrics": [point.to_dict() for point in points],
        "final_metrics": {
            "loss": final_metrics["loss"],
            "accuracy": final_metrics["accuracy"],
        },
    }
    return sha256_json(payload)


def record_metric_digest(record: RunRecord) -> str:
    return metric_digest(record.step_metrics, record.aggregation, record.final_metrics)


def replay_record(record: RunRecord, expected_digest: str | None) -> dict[str, Any]:
    """Regenerate and verify one snapshot record or raise E_REPLAY."""

    version = (
        f"run_version={record.run_version},format_version={record.format_version},"
        f"aggregation_version={record.aggregation.version}"
    )
    if not expected_digest:
        raise ReplayError(
            record.run_id,
            version,
            "metric summary is missing from the snapshot",
            run_version=record.run_version,
            format_version=record.format_version,
            aggregation_version=record.aggregation.version,
        )
    if record.aggregation.version != "agg-v1":
        raise ReplayError(
            record.run_id,
            version,
            f"aggregation version {record.aggregation.version!r} cannot be reproduced by this build",
            run_version=record.run_version,
            format_version=record.format_version,
            aggregation_version=record.aggregation.version,
        )

    generated = generate_step_metrics(record.parameters)
    generated_final = aggregate_metrics(generated, record.aggregation)
    generated_digest = metric_digest(generated, record.aggregation, generated_final)
    difference = first_metric_difference(generated, record.step_metrics)
    if difference is not None:
        raise ReplayError(
            record.run_id,
            version,
            "regenerated metric series differs from the snapshot",
            run_version=record.run_version,
            format_version=record.format_version,
            aggregation_version=record.aggregation.version,
            first_offset_step=difference["step"],
            field=difference["field"],
            expected=difference["expected"],
            actual=difference["actual"],
            expected_digest=expected_digest,
            regenerated_digest=generated_digest,
        )
    if generated_digest != expected_digest:
        final_difference = first_final_difference(
            record.final_metrics, generated_final, len(generated)
        )
        details: dict[str, Any] = {
            "run_version": record.run_version,
            "format_version": record.format_version,
            "aggregation_version": record.aggregation.version,
            "expected_digest": expected_digest,
            "regenerated_digest": generated_digest,
        }
        if final_difference is not None:
            details.update(
                {
                    "first_offset_step": final_difference["step"],
                    "field": final_difference["field"],
                    "expected": final_difference["expected"],
                    "actual": final_difference["actual"],
                }
            )
        raise ReplayError(
            record.run_id,
            version,
            "regenerated metric summary does not match the snapshot summary",
            **details,
        )
    return {
        "run_id": record.run_id,
        "run_version": record.run_version,
        "format_version": record.format_version,
        "source_format_version": record.provenance.source_format_version,
        "aggregation": record.aggregation.to_dict(),
        "metric_digest": generated_digest,
        "status": "pass",
    }
