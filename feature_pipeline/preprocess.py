"""Missing-value normalization, filling and dimension unification."""

from __future__ import annotations

import math
from statistics import median

import numpy as np

from .models import Cell, MergeConfig


def normalize_missing(tables: list) -> None:
    """All three parsers already emit Cell.missing; this guarantees the
    single canonical missing marker (NaN + missing=True) and nothing else."""
    for table in tables:
        for cell in table.cells:
            if cell.missing:
                cell.value = float("nan")
            elif not math.isfinite(cell.value):
                from .errors import NonFiniteInputError

                raise NonFiniteInputError(
                    f"non-finite value for {cell.entity_id}/{cell.feature}",
                    entity_id=cell.entity_id,
                    stage="missing_normalize",
                )


def collect_nonmissing(tables: list, feature: str) -> list[float]:
    values: list[float] = []
    for table in tables:
        for cell in table.cells:
            if cell.feature == feature and not cell.missing:
                values.append(float(cell.value))
    return values


def fill_missing(tables: list, config: MergeConfig) -> dict:
    """Fill missing cells before merging; statistics use present values only."""
    audit: dict[str, dict] = {}
    strategy = config.missing_fill["strategy"]
    for feature in config.feature_whitelist:
        present = collect_nonmissing(tables, feature)
        if strategy == "zero":
            fill_value = 0.0
            basis = "fixed_zero"
        elif strategy == "constant":
            fill_value = float(config.missing_fill["value"])
            basis = "configured_constant"
        elif strategy == "mean":
            fill_value = float(np.mean(present)) if present else 0.0
            basis = "mean_of_nonmissing" if present else "empty_fallback_zero"
        elif strategy == "median":
            fill_value = float(median(present)) if present else 0.0
            basis = "median_of_nonmissing" if present else "empty_fallback_zero"
        else:  # pragma: no cover - validated at config load
            raise ValueError(f"unknown fill strategy {strategy}")

        filled_cells = 0
        for table in tables:
            for cell in table.cells:
                if cell.feature == feature and cell.missing:
                    cell.value = fill_value
                    cell.missing = False
                    cell.filled = True  # type: ignore[attr-defined]
                    filled_cells += 1
        audit[feature] = {
            "strategy": strategy,
            "basis": basis,
            "fill_value": fill_value,
            "nonmissing_sample_size": len(present),
            "filled_cell_count": filled_cells,
        }
    return audit


def scaling_statistics(tables: list, config: MergeConfig) -> dict:
    """Pooled per-feature statistics computed from non-missing values only."""
    stats: dict[str, dict] = {}
    for feature in config.feature_whitelist:
        values = np.array(collect_nonmissing(tables, feature), dtype=np.float64)
        entry: dict = {
            "method": config.scaling_method,
            "sample_size": int(values.size),
            "tie_rule": config.percentile_tie_rule,
        }
        if values.size == 0:
            entry.update(
                {
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "all_equal": True,
                    "decision": "no_nonmissing_values_identity_transform",
                }
            )
            stats[feature] = entry
            continue

        mean = float(np.mean(values))
        std = float(np.std(values))  # population standard deviation
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        all_equal = vmax == vmin
        entry.update(
            {
                "mean": mean,
                "std": std,
                "min": vmin,
                "max": vmax,
                "all_equal": all_equal,
            }
        )
        if config.scaling_method == "percentile":
            if all_equal:
                entry["decision"] = (
                    "boundary_equal_all_values_get_50th_percentile"
                )
            else:
                entry["decision"] = (
                    f"empirical_cdf_{config.percentile_tie_rule}"
                )
        elif config.scaling_method == "zscore":
            entry["decision"] = (
                "zero_std_maps_to_zero" if std == 0.0 else "population_zscore"
            )
        else:
            entry["decision"] = "raw_identity"
        stats[feature] = entry
    return stats


def percentile_rank(value: float, samples: np.ndarray, tie_rule: str) -> float:
    """Empirical CDF percentile in [0, 100] with a fixed tie rule."""
    if samples.size == 0:
        return 0.0
    less = int(np.sum(samples < value))
    equal = int(np.sum(samples == value))
    total = samples.size
    if tie_rule == "min_rank":
        rank = less + 1
    elif tie_rule == "max_rank":
        rank = less + equal
    else:  # average_rank, the default fixed rule
        rank = less + (equal + 1) / 2.0
    return 100.0 * rank / total


def scale_cell(cell: Cell, stats: dict, config: MergeConfig) -> float:
    feature_stats = stats[cell.feature]
    value = float(cell.value)
    if config.scaling_method == "raw":
        return value
    if feature_stats["sample_size"] == 0:
        return value
    if config.scaling_method == "zscore":
        if feature_stats["std"] == 0.0:
            return 0.0
        return (value - feature_stats["mean"]) / feature_stats["std"]
    # percentile: precomputed pooled samples are attached by the pipeline
    samples = feature_stats["_samples"]
    if feature_stats["all_equal"]:
        return 50.0
    return percentile_rank(value, samples, config.percentile_tie_rule)


def apply_scaling(tables: list, stats: dict, config: MergeConfig) -> None:
    for feature in config.feature_whitelist:
        samples = np.array(
            collect_nonmissing(tables, feature), dtype=np.float64
        )
        stats[feature]["_samples"] = samples
    for table in tables:
        for cell in table.cells:
            if cell.feature in stats:
                cell.value = scale_cell(cell, stats, config)
    for entry in stats.values():
        entry.pop("_samples", None)
