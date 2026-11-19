from __future__ import annotations

import numpy as np

from feature_diag.config import load_experiment
from feature_diag.crossing import build_design
from feature_diag.data import generate_table
from feature_diag.metrics import compare_precision, compute_metric
from feature_diag.reductions import reduce_path, standardize


def _pipeline(make_experiment, precision: str = "float64"):
    experiment = load_experiment(make_experiment())
    scenario = experiment.scenarios[0]
    table = generate_table(experiment.data_spec, scenario, 64)
    design = build_design(table, scenario, 2, precision)
    standard = standardize(design, precision)
    scope = "compact-projection-scope"
    reductions = {
        path: reduce_path(standard, path, scenario.reductions[path], precision, scope)
        for path in ("pca", "random_projection", "variance_filter")
    }
    return experiment, scenario, table, design, standard, reductions


def test_pca_sign_normalization_and_random_projection_determinism(make_experiment) -> None:
    standard = standardize(
        build_design(
            generate_table(
                (experiment := load_experiment(make_experiment())).data_spec,
                experiment.scenarios[0],
                64,
            ),
            experiment.scenarios[0],
            2,
            "float64",
        ),
        "float64",
    )
    scenario = experiment.scenarios[0]
    pca1 = reduce_path(standard, "pca", scenario.reductions["pca"], "float64", "scope")
    pca2 = reduce_path(standard, "pca", scenario.reductions["pca"], "float64", "scope")
    assert np.array_equal(pca1["scores"], pca2["scores"])
    for component in pca1["components"]:
        dominant = int(np.argmax(np.abs(component)))
        assert float(component[dominant]) >= 0.0

    rp1 = reduce_path(standard, "random_projection", scenario.reductions["random_projection"], "float64", "scope")
    rp2 = reduce_path(standard, "random_projection", scenario.reductions["random_projection"], "float64", "scope")
    assert np.array_equal(rp1["scores"], rp2["scores"])
    assert rp1["summary"]["projection_matrix_fingerprint"] == rp2["summary"]["projection_matrix_fingerprint"]


def test_variance_filter_keeps_stable_feature_order(make_experiment) -> None:
    _, scenario, _, _, _, reductions = _pipeline(make_experiment, "float32")
    selected = reductions["variance_filter"]["summary"]["selected_features"]
    assert selected[:3] == ["num__amount", "num__score", "num__tenure"]


def test_precision_comparison_reports_four_stage_errors_and_attribution(make_experiment) -> None:
    experiment = load_experiment(make_experiment())
    scenario = experiment.scenarios[0]
    table = generate_table(experiment.data_spec, scenario, 64)
    design32 = build_design(table, scenario, 2, "float32")
    design64 = build_design(table, scenario, 2, "float64")
    standard32 = standardize(design32, "float32")
    standard64 = standardize(design64, "float64")
    reduction32 = reduce_path(
        standard32,
        "pca",
        scenario.reductions["pca"],
        "float32",
        "scope",
    )
    reduction64 = reduce_path(
        standard64,
        "pca",
        scenario.reductions["pca"],
        "float64",
        "scope",
    )
    metric32 = compute_metric(reduction32, "first_coordinate", "float32")
    metric64 = compute_metric(reduction64, "first_coordinate", "float64")
    comparison = compare_precision(
        scenario_id="compact",
        sample_size=64,
        order_limit=2,
        path="pca",
        sample_ids=design64["sample_ids"],
        design32=design32,
        design64=design64,
        standard32=standard32,
        standard64=standard64,
        reduction32=reduction32,
        reduction64=reduction64,
        metric32=metric32,
        metric64=metric64,
        decision_config=scenario.decision,
        tolerances=scenario.tolerances,
    )
    assert [item["stage"] for item in comparison["stage_summary"]] == [
        "cross_encoding",
        "standardization",
        "projection",
        "final_metric",
    ]
    for sample in comparison["per_sample"]:
        assert set(sample["stage_errors"]) == {
            "cross_encoding",
            "standardization",
            "projection",
            "final_metric",
        }
        expected_first = next(
            (
                stage
                for stage in ("cross_encoding", "standardization", "projection", "final_metric")
                if sample["stage_errors"][stage] > scenario.tolerances[stage]
            ),
            None,
        )
        assert sample["first_exceeded_stage"] == expected_first
