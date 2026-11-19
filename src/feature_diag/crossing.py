"""Deterministic categorical crossing, cardinality-budget degradation, and design assembly."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .config import ScenarioConfig
from .data import GeneratedTable
from .data_spec import CATEGORY_KINDS, NUMERIC_KINDS
from .fingerprint import array_fingerprint, fingerprint, sha256_bytes, sha256_text


@dataclass
class CrossFeature:
    name: str
    columns: tuple[str, ...]
    order: int
    values: np.ndarray
    unique_count: int
    effective_unique_count: int
    downgraded: bool
    policy: str
    hash_prefix: str
    fingerprint: str

    def detail(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "columns": list(self.columns),
            "order": self.order,
            "unique_count": self.unique_count,
            "effective_unique_count": self.effective_unique_count,
            "downgraded": self.downgraded,
            "policy": self.policy,
            "hash_prefix": self.hash_prefix,
            "fingerprint": self.fingerprint,
        }


@dataclass
class CrossingBundle:
    order_limit: int
    features: list[CrossFeature]
    skipped: list[dict[str, Any]]
    downgrades: list[dict[str, Any]]
    processed_order: list[dict[str, Any]]


def _combination_order(scenario: ScenarioConfig, order_limit: int) -> list[tuple[str, ...]]:
    columns = sorted(scenario.candidate_columns)
    selected_orders = sorted(order for order in scenario.cross_orders if order <= order_limit)
    combinations: list[tuple[str, ...]] = []
    for order in selected_orders:
        if order > len(columns):
            continue
        for combination in itertools.combinations(columns, order):
            combinations.append(tuple(sorted(combination)))
    return sorted(set(combinations), key=lambda item: (len(item), item))


def build_crossing_bundle(table: GeneratedTable, scenario: ScenarioConfig, order_limit: int) -> CrossingBundle:
    feature_names_seen: set[str] = set()
    features: list[CrossFeature] = []
    skipped: list[dict[str, Any]] = []
    downgrades: list[dict[str, Any]] = []
    processed_order: list[dict[str, Any]] = []

    for columns in _combination_order(scenario, order_limit):
        order = len(columns)
        raw_keys = [
            "|".join(str(table.columns[column][index]) for column in columns)
            for index in range(table.size)
        ]
        unique_keys = sorted(set(raw_keys))
        unique_count = len(unique_keys)
        digest_by_key = {key: sha256_text(key).encode("ascii") for key in unique_keys}
        hash_prefix = sha256_bytes(b"".join(sorted(digest_by_key.values())))[:16]
        name = f"x{order}__" + "__".join(columns)
        if name in feature_names_seen:
            continue
        feature_names_seen.add(name)

        processed_order.append(
            {
                "name": name,
                "columns": list(columns),
                "order": order,
                "unique_count": unique_count,
            }
        )
        if unique_count > scenario.cardinality_budget:
            if scenario.overflow_policy == "skip":
                skipped.append(
                    {
                        "name": name,
                        "columns": list(columns),
                        "order": order,
                        "unique_count": unique_count,
                        "budget": scenario.cardinality_budget,
                        "policy": "skip",
                        "reason": "cardinality_budget_exceeded",
                    }
                )
                continue
            bucket_size = scenario.cardinality_budget
            encoded = np.asarray(
                [int.from_bytes(digest_by_key[key][:8], "big") % bucket_size for key in raw_keys],
                dtype=np.int64,
            )
            effective_unique = int(np.unique(encoded).size)
            downgrade = {
                "name": name,
                "columns": list(columns),
                "order": order,
                "unique_count": unique_count,
                "effective_unique_count": effective_unique,
                "budget": bucket_size,
                "policy": "hash_bucket",
                "reason": "cardinality_budget_exceeded",
            }
            downgrades.append(downgrade)
            feature = CrossFeature(
                name=name,
                columns=columns,
                order=order,
                values=encoded,
                unique_count=unique_count,
                effective_unique_count=effective_unique,
                downgraded=True,
                policy="hash_bucket",
                hash_prefix=hash_prefix,
                fingerprint=fingerprint(
                    {
                        "name": name,
                        "columns": columns,
                        "unique_count": unique_count,
                        "effective_unique_count": effective_unique,
                        "policy": "hash_bucket",
                        "data": array_fingerprint(encoded),
                    }
                ),
            )
            features.append(feature)
            continue

        rank_by_key = {
            key: rank for rank, key in enumerate(sorted(unique_keys, key=lambda item: (digest_by_key[item], item)))
        }
        encoded = np.asarray([rank_by_key[key] for key in raw_keys], dtype=np.int64)
        feature = CrossFeature(
            name=name,
            columns=columns,
            order=order,
            values=encoded,
            unique_count=unique_count,
            effective_unique_count=unique_count,
            downgraded=False,
            policy="none",
            hash_prefix=hash_prefix,
            fingerprint=fingerprint(
                {
                    "name": name,
                    "columns": columns,
                    "unique_count": unique_count,
                    "policy": "none",
                    "data": array_fingerprint(encoded),
                }
            ),
        )
        features.append(feature)

    return CrossingBundle(
        order_limit=order_limit,
        features=features,
        skipped=skipped,
        downgrades=downgrades,
        processed_order=processed_order,
    )


def base_feature_columns(table: GeneratedTable, scenario: ScenarioConfig) -> list[tuple[str, np.ndarray]]:
    output: list[tuple[str, np.ndarray]] = []
    for column in table.spec.columns:
        if column.kind in {"id", "target"} or column.name == scenario.target_column:
            continue
        if column.kind in NUMERIC_KINDS | {"redundant"}:
            output.append((f"num__{column.name}", np.asarray(table.columns[column.name], dtype=np.float64)))
        elif column.kind in CATEGORY_KINDS:
            output.append((f"cat__{column.name}", np.asarray(table.encoded_columns[column.name], dtype=np.float64)))
    return output


def cross_parameter_fingerprint(
    table: GeneratedTable,
    scenario: ScenarioConfig,
    order_limit: int,
    precision: str,
) -> str:
    return fingerprint(
        {
            "kind": "crossing",
            "schema": 1,
            "candidate_columns": list(scenario.candidate_columns),
            "order_limit": int(order_limit),
            "configured_orders": [order for order in scenario.cross_orders if order <= order_limit],
            "cardinality_budget": scenario.cardinality_budget,
            "overflow_policy": scenario.overflow_policy,
            "base_features": [
                {"name": column.name, "kind": column.kind}
                for column in table.spec.columns
                if column.kind not in {"id", "target"} and column.name != scenario.target_column
            ],
            "category_maps": table.category_maps,
            "target_column": scenario.target_column,
            "precision": precision,
        }
    )


def build_design(
    table: GeneratedTable,
    scenario: ScenarioConfig,
    order_limit: int,
    precision: str,
) -> dict[str, Any]:
    if precision not in {"float32", "float64"}:
        raise ValueError(f"unsupported precision: {precision}")
    bundle = build_crossing_bundle(table, scenario, order_limit)
    feature_names: list[str] = []
    feature_arrays: list[np.ndarray] = []
    for name, values in base_feature_columns(table, scenario):
        feature_names.append(name)
        feature_arrays.append(values)
    for feature in bundle.features:
        feature_names.append(feature.name)
        feature_arrays.append(np.asarray(feature.values, dtype=np.float64))
    if not feature_arrays:
        raise ValueError("design matrix has no features")
    matrix = np.column_stack(feature_arrays).astype(precision, copy=False)

    parameter_fingerprint = cross_parameter_fingerprint(table, scenario, order_limit, precision)
    design_fingerprint = fingerprint(
        {
            "upstream": table.fingerprint,
            "parameters": parameter_fingerprint,
            "precision": precision,
            "feature_names": feature_names,
            "matrix": array_fingerprint(matrix),
        }
    )
    return {
        "scenario_id": scenario.scenario_id,
        "sample_size": table.size,
        "order_limit": order_limit,
        "precision": precision,
        "sample_ids": table.sample_ids.copy(),
        "target": table.target.copy(),
        "feature_names": feature_names,
        "matrix": matrix,
        "upstream_fingerprint": table.fingerprint,
        "parameter_fingerprint": parameter_fingerprint,
        "fingerprint": design_fingerprint,
        "cross_features": [
            {
                **feature.detail(),
                "values": feature.values.copy(),
            }
            for feature in bundle.features
        ],
        "cross_skipped": bundle.skipped,
        "cross_downgrades": bundle.downgrades,
        "cross_processed_order": bundle.processed_order,
        "category_maps": table.category_maps,
        "missing_counts": table.missing_counts,
        "table_fingerprint": table.fingerprint,
    }
