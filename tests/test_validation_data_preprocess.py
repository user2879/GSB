from __future__ import annotations

import math

import numpy as np

from cluster_workbench.config import normalize_config, validate_candidate
from cluster_workbench.data import build_feature_matrix, generate_table
from cluster_workbench.errors import (
    E_CLUSTER_COUNT_RANGE,
    E_CLUSTER_COUNT_ZERO,
    E_COLUMN_DESCRIPTION_MISSING,
    E_COLUMN_UNKNOWN,
    E_CONCURRENCY_NON_POSITIVE,
    E_EPS_NON_POSITIVE,
    E_GLOBAL_BUDGET_NON_POSITIVE,
    E_LATENCY_BUDGET_NON_POSITIVE,
    E_TYPE_MISMATCH,
)
from cluster_workbench.preprocess import fit_transform, make_preprocess_plan

from conftest import candidate, normalized_pair


def test_generation_and_encoding_are_deterministic(base_data):
    left = generate_table(base_data)
    right = generate_table(base_data)
    assert left.values == right.values
    assert left.column_content_fingerprints == right.column_content_fingerprints
    assert left.content_fingerprint == right.content_fingerprint
    assert any(value is None for value in left.values["n1"])
    matrix, columns, unknown = build_feature_matrix(left, ["n1", "c1", "ccos"])
    assert columns == ["n1", "c1", "ccos"]
    assert np.isnan(matrix[:, 0]).any()
    assert unknown == {}
    assert set(np.unique(matrix[~np.isnan(matrix[:, 1]), 1])).issubset({0.0, 1.0, 2.0})


def test_config_validation_codes_do_not_abort(base_config):
    raw = dict(base_config)
    raw["preprocessing"] = dict(base_config["preprocessing"], method="unknown-method")
    raw["scheduler"] = dict(
        base_config["scheduler"],
        concurrency=0,
        global_iteration_budget=0,
        interaction_latency_budget_ms=0,
    )
    normalized, issues = normalize_config(raw)
    codes = {issue.code for issue in issues}
    assert E_CONCURRENCY_NON_POSITIVE in codes
    assert E_GLOBAL_BUDGET_NON_POSITIVE in codes
    assert E_LATENCY_BUDGET_NON_POSITIVE in codes
    assert normalized["scheduler"]["concurrency"] == 1
    assert normalized["scheduler"]["global_iteration_budget"] == 0
    assert normalized["preprocessing"]["method"] == "standardize"


def test_candidate_validation_codes(base_data, base_config):
    config, data = normalized_pair(base_config, base_data)
    table = generate_table(data)
    common = {
        "row_count": table.rows,
        "available_columns": table.available_columns,
        "declared_columns": table.declared_columns,
    }
    _, issues = validate_candidate(
        candidate("zero", params={"n_clusters": 0}), **common
    )
    assert any(issue.code == E_CLUSTER_COUNT_ZERO for issue in issues)
    _, issues = validate_candidate(
        candidate("negative", params={"n_clusters": -2}), **common
    )
    assert any(issue.code == E_CLUSTER_COUNT_RANGE for issue in issues)
    _, issues = validate_candidate(
        candidate("type", params={"n_clusters": "3"}), **common
    )
    assert any(issue.code == E_TYPE_MISMATCH for issue in issues)
    _, issues = validate_candidate(
        candidate("unknown", features=["ghost"]), **common
    )
    assert any(issue.code == E_COLUMN_UNKNOWN for issue in issues)
    _, issues = validate_candidate(
        candidate("eps", method="dbscan", params={"eps": 0, "min_samples": 3}), **common
    )
    assert any(issue.code == E_EPS_NON_POSITIVE for issue in issues)

    injected = generate_table(
        data,
        overrides={},
        injections=[
            {"id": "remove", "type": "missing_column_description", "column": "c1"}
        ],
    )
    _, issues = validate_candidate(
        candidate("missing", params={"n_clusters": 2}, features=["c1"]),
        row_count=injected.rows,
        available_columns=injected.available_columns,
        declared_columns=injected.declared_columns,
    )
    assert any(issue.code == E_COLUMN_DESCRIPTION_MISSING for issue in issues)


def test_preprocess_is_deterministic_and_fills_only_from_training_part(base_data, base_config):
    config, data = normalized_pair(base_config, base_data)
    table = generate_table(data)
    columns = ["n1", "n2", "c1", "ccos"]
    plan = make_preprocess_plan(
        table,
        columns,
        preprocessing=config["preprocessing"],
        train_fraction=0.6,
    )
    left = fit_transform(table, plan, preprocessing=config["preprocessing"], candidate_id="a")
    right = fit_transform(table, plan, preprocessing=config["preprocessing"], candidate_id="a")
    assert np.array_equal(left.matrix, right.matrix)
    assert left.content_fingerprint == right.content_fingerprint
    assert left.metadata["fill_values"] == right.metadata["fill_values"]
    raw, _, _ = build_feature_matrix(table, columns)
    for column_index, column in enumerate(columns):
        observed = [
            float(raw[index, column_index])
            for index in plan.train_indices
            if not math.isnan(float(raw[index, column_index]))
        ]
        ordered = sorted(observed)
        middle = len(ordered) // 2
        expected = (
            ordered[middle]
            if len(ordered) % 2
            else (ordered[middle - 1] + ordered[middle]) / 2.0
        )
        assert left.metadata["fill_values"][column] == expected
    assert left.metadata["finite_accumulation_order"].startswith("ascending row index")
    assert left.metadata["missing_fill_policy"].startswith("median_of_candidate_training")


def test_quantile_tie_policy_is_recorded_and_equal_inputs_map_equal(base_data, base_config):
    config, data = normalized_pair(base_config, base_data)
    config["preprocessing"] = dict(
        config["preprocessing"],
        method="quantile",
        quantile_tie_policy="average_rank",
    )
    table = generate_table(
        data,
        overrides={"n3": {"kind": "constant", "value": 0.25}},
    )
    plan = make_preprocess_plan(
        table,
        ["n3"],
        preprocessing=config["preprocessing"],
        train_fraction=0.6,
    )
    result = fit_transform(table, plan, preprocessing=config["preprocessing"], candidate_id="q")
    assert result.metadata["quantile_tie_policy"] == "average_rank"
    quantile = result.metadata["statistics"]["quantiles"]["n3"]
    assert quantile["tie_policy"] == "average_rank"
    observed = result.matrix[~np.isnan(result.matrix)]
    assert np.allclose(observed, observed[0])
