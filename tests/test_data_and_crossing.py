from __future__ import annotations

from dataclasses import replace

import numpy as np

from feature_diag.config import load_experiment
from feature_diag.crossing import build_crossing_bundle, build_design
from feature_diag.data import generate_table


def test_data_generation_is_deterministic_and_fills_missing(make_experiment) -> None:
    experiment = load_experiment(make_experiment())
    scenario = experiment.scenarios[0]
    first = generate_table(experiment.data_spec, scenario, 64)
    second = generate_table(experiment.data_spec, scenario, 64)
    assert first.fingerprint == second.fingerprint
    assert np.array_equal(np.asarray(first.target), np.asarray(second.target))
    assert all(np.all(np.isfinite(np.asarray(values, dtype=np.float64))) for values in first.columns.values() if values.dtype != object)
    assert all(first.missing_counts[name] >= 0 for name in first.missing_counts)
    assert set(first.category_maps) == {"region", "channel", "user_id"}
    assert np.array_equal(first.encoded_columns["region"], second.encoded_columns["region"])


def test_combinations_use_order_then_lexicographic_and_degrade(make_experiment) -> None:
    experiment = load_experiment(make_experiment())
    scenario = experiment.scenarios[0]
    table = generate_table(experiment.data_spec, scenario, 64)
    bundle = build_crossing_bundle(table, scenario, 2)
    ordered_columns = [feature.columns for feature in bundle.features]
    assert ordered_columns == sorted(ordered_columns, key=lambda item: (len(item), item))
    high_cardinality = next(feature for feature in bundle.features if feature.columns == ("user_id",))
    assert high_cardinality.unique_count > scenario.cardinality_budget
    assert high_cardinality.downgraded is True
    assert high_cardinality.policy == "hash_bucket"
    assert high_cardinality.effective_unique_count <= scenario.cardinality_budget
    assert any(item["name"] == "x1__user_id" for item in bundle.downgrades)

    skip_scenario = replace(scenario, overflow_policy="skip")
    skip_bundle = build_crossing_bundle(table, skip_scenario, 2)
    assert any(item["name"] == "x1__user_id" for item in skip_bundle.skipped)
    assert all(feature.columns != ("user_id",) for feature in skip_bundle.features)


def test_design_contains_sample_ids_features_and_exact_hash_codes(make_experiment) -> None:
    experiment = load_experiment(make_experiment())
    scenario = experiment.scenarios[0]
    table = generate_table(experiment.data_spec, scenario, 64)
    design32 = build_design(table, scenario, 2, "float32")
    design32_again = build_design(table, scenario, 2, "float32")
    assert design32["fingerprint"] == design32_again["fingerprint"]
    assert design32["matrix"].dtype == np.dtype("float32")
    assert np.array_equal(design32["matrix"], design32_again["matrix"])
    assert design32["feature_names"] == design32_again["feature_names"]
    assert list(design32["sample_ids"][:2]) == ["row-000000", "row-000001"]
    assert len(design32["cross_processed_order"]) == len(design32["cross_features"])
