from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .fingerprint import derive_seed


@dataclass
class Dataset:
    columns: dict[str, list[Any]]
    target: list[Any]
    target_name: str
    positive_value: Any
    negative_value: Any

    @property
    def rows(self) -> int:
        return len(self.target)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -35.0, 35.0)))


def _numeric_values(rows: int, rng: np.random.Generator, spec: dict[str, Any], latent: np.ndarray) -> np.ndarray:
    distribution = str(spec.get("distribution", "normal")).lower()
    if distribution == "normal":
        mean = float(spec.get("mean", 0.0))
        std = float(spec.get("std", 1.0))
        values = mean + std * (0.58 * latent + 0.82 * rng.normal(size=rows))
    elif distribution == "lognormal":
        mean = float(spec.get("mean", 0.0))
        sigma = float(spec.get("sigma", 1.0))
        values = np.exp(mean + sigma * (0.45 * latent + 0.86 * rng.normal(size=rows)))
    elif distribution == "uniform":
        minimum = float(spec.get("min", 0.0))
        maximum = float(spec.get("max", 1.0))
        values = minimum + (maximum - minimum) * np.clip(0.5 + 0.22 * latent + 0.22 * rng.random(rows), 0.0, 1.0)
    else:
        raise ValueError(f"unsupported numeric distribution: {distribution!r}")

    outlier_rate = float(spec.get("outlier_rate", 0.0))
    if outlier_rate > 0.0:
        mask = rng.random(rows) < outlier_rate
        scale = float(spec.get("outlier_scale", 6.0))
        spread = float(np.std(values)) or 1.0
        directions = np.where(rng.random(rows) < 0.5, -1.0, 1.0)
        values = values + mask * directions * scale * spread

    return np.rint(values).astype(np.int64)


def _categorical_values(
    rows: int,
    rng: np.random.Generator,
    spec: dict[str, Any],
    latent: np.ndarray,
) -> np.ndarray:
    cardinality = int(spec.get("cardinality", 1))
    name = str(spec["name"])
    labels = np.asarray([f"{name}#{index:04d}" for index in range(cardinality)], dtype=object)
    weights = rng.dirichlet(np.full(cardinality, 0.72, dtype=np.float64))
    # A small latent shift makes the category distribution depend on the same
    # cohort signal used by the target while keeping the process deterministic.
    shifted = weights * np.exp(0.12 * latent.mean())
    shifted = shifted / shifted.sum()
    return rng.choice(labels, size=rows, replace=True, p=shifted)


def generate_dataset(schema: dict[str, Any]) -> Dataset:
    rows = int(schema.get("rows", 0))
    if rows <= 0:
        raise ValueError("synthetic.rows must be a positive integer")
    base_seed = int(schema.get("seed"))

    target_spec = schema.get("target")
    if not isinstance(target_spec, dict) or not target_spec.get("name"):
        raise ValueError("target description is missing")
    target_name = str(target_spec["name"])
    positive_value = target_spec.get("positive_value", 1)
    negative_value = target_spec.get("negative_value", 0)

    latent_rng = np.random.default_rng(derive_seed(base_seed, "latent"))
    target_rng = np.random.default_rng(derive_seed(base_seed, "target"))
    latent = latent_rng.normal(size=rows)
    logits = -0.08 + 1.25 * latent + 0.42 * target_rng.normal(size=rows)
    raw_positive = target_rng.random(rows) < _sigmoid(logits)
    target = [positive_value if bool(value) else negative_value for value in raw_positive]

    target_missing_rate = float(target_spec.get("missing_rate", 0.0))
    if target_missing_rate > 0.0:
        missing_mask = target_rng.random(rows) < target_missing_rate
        for index, missing in enumerate(missing_mask):
            if bool(missing):
                target[index] = None

    columns: dict[str, list[Any]] = {}
    for spec in schema.get("columns", []):
        name = str(spec["name"])
        column_type = str(spec["type"])
        seed_offset = int(spec.get("seed_offset", 0))
        rng = np.random.default_rng(derive_seed(base_seed, column_type, name, seed_offset))
        if column_type == "numeric":
            values = _numeric_values(rows, rng, spec, latent)
        elif column_type == "categorical":
            values = _categorical_values(rows, rng, spec, latent)
        else:
            raise ValueError(f"unsupported column type: {column_type!r}")

        missing_rate = float(spec.get("missing_rate", 0.0))
        missing_mask = rng.random(rows) < missing_rate
        if column_type == "categorical":
            cardinality = int(spec.get("cardinality", 1))
            labels = np.asarray([f"{name}#{index:04d}" for index in range(cardinality)], dtype=object)
            nonmissing_positions = np.asarray(
                [index for index, missing in enumerate(missing_mask) if not bool(missing)],
                dtype=np.int64,
            )
            if len(nonmissing_positions) >= cardinality:
                protected = rng.choice(nonmissing_positions, size=cardinality, replace=False)
            else:
                protected = rng.choice(rows, size=cardinality, replace=False)
                missing_mask[protected] = False
            # Preserve the configured support size even under missing injection.
            for position, label in zip(protected, labels):
                values[int(position)] = label
        values_out: list[Any] = []
        for index, value in enumerate(values):
            if bool(missing_mask[index]):
                values_out.append(None)
            elif column_type == "numeric":
                values_out.append(int(value))
            else:
                values_out.append(str(value))
        columns[name] = values_out

    # The public table exposes the target as a normal column too; this is what
    # downstream validation and recovery checks inspect.
    columns[target_name] = list(target)
    return Dataset(
        columns=columns,
        target=list(target),
        target_name=target_name,
        positive_value=positive_value,
        negative_value=negative_value,
    )
