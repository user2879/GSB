"""Deterministic equal-frequency feature binning."""

from __future__ import annotations

from bisect import bisect_right
from typing import Any

from .config import ExperimentConfig
from .data import PreparedData
from .fingerprint import fingerprint, rounded, rounded_tree


def binning_cache_key(config: ExperimentConfig, data: PreparedData) -> str:
    return fingerprint(
        {
            "stage": "binning/v1",
            "data_fingerprint": data.data_fingerprint,
            "bin_count": config.bin_count,
            "features": list(config.features),
            "feature_types": data.feature_types,
            "seed": config.seed,
            "precision_decimals": config.precision_decimals,
        }
    )


def compute_binning(
    config: ExperimentConfig,
    data: PreparedData,
    key: str | None = None,
) -> dict[str, Any]:
    decimals = config.precision_decimals
    columns: dict[str, dict[str, Any]] = {}
    for feature_index, feature in enumerate(data.feature_names):
        column_type = data.feature_types[feature]
        values = [data.feature_values[row_index][feature_index] for row_index in range(data.row_count)]
        if column_type == "numeric":
            ordered = sorted(
                (float(value), data.row_ids[row_index], row_index)
                for row_index, value in enumerate(values)
            )
            rank_by_row = {row_index: rank for rank, (_, _, row_index) in enumerate(ordered)}
            boundaries: list[int] = []
            for bin_index in range(1, config.bin_count):
                position = (bin_index * data.row_count) // config.bin_count
                if 0 < position < data.row_count and position not in boundaries:
                    boundaries.append(position)
            boundaries.sort()
            bin_indices = [bisect_right(boundaries, rank_by_row[row_index]) for row_index in range(data.row_count)]
            edges = [ordered[position][0] for position in boundaries]
            content = {
                "name": feature,
                "type": column_type,
                "boundaries": boundaries,
                "edges": [rounded(edge, decimals) for edge in edges],
                "bin_indices": bin_indices,
                "row_ids": list(data.row_ids),
                "values": [rounded(float(value), decimals) for value in values],
            }
            content_fingerprint = fingerprint({"schema": "numeric-bin-content/v1", "content": content})
            columns[feature] = {
                "name": feature,
                "type": column_type,
                "boundaries": boundaries,
                "edges": content["edges"],
                "bin_indices": bin_indices,
                "content_fingerprint": content_fingerprint,
            }
        else:
            categories = sorted({int(value) for value in values})
            content = {
                "name": feature,
                "type": column_type,
                "categories": list(data.category_codes[feature].keys()),
                "bin_indices": [int(value) for value in values],
                "row_ids": list(data.row_ids),
            }
            content_fingerprint = fingerprint({"schema": "categorical-bin-content/v1", "content": content})
            columns[feature] = {
                "name": feature,
                "type": column_type,
                "boundaries": [],
                "edges": [],
                "categories": content["categories"],
                "observed_codes": categories,
                "bin_indices": content["bin_indices"],
                "content_fingerprint": content_fingerprint,
            }

    effective_key = key or binning_cache_key(config, data)
    body = {
        "schema": "binning/v1",
        "key": effective_key,
        "parameters": {
            "bin_count": config.bin_count,
            "seed": config.seed,
            "data_fingerprint": data.data_fingerprint,
            "precision_decimals": decimals,
        },
        "row_ids": list(data.row_ids),
        "feature_names": list(data.feature_names),
        "columns": columns,
    }
    body["fingerprint"] = fingerprint({"schema": "binning-product/v1", "body": body})
    return rounded_tree(body, decimals)


def reject_binned_column_metadata_change(previous: dict[str, Any], current: dict[str, Any]) -> None:
    """Defensive assertion used by tests/entrypoints before reuse."""
    if previous.get("fingerprint") != current.get("fingerprint"):
        raise ValueError("binning product changed despite matching cache key")

