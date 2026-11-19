from __future__ import annotations

from pathlib import Path

from feature_diag.config import load_experiment
from feature_diag.errors import E_CONFIG_INVALID


def test_all_config_error_classes_are_isolated() -> None:
    experiment = load_experiment(Path("sample/experiment_invalid.yaml"))
    assert [scenario.scenario_id for scenario in experiment.scenarios] == ["valid_after_invalid"]
    reasons = {(failure.scope, failure.reason) for failure in experiment.failures}
    assert ("unknown_column", "unknown_column") in reasons
    assert ("type_mismatch", "type_mismatch") in reasons
    assert ("missing_target", "target_missing") in reasons
    assert ("non_positive_order", "non_positive_cross_order") in reasons
    assert ("non_positive_budget", "non_positive_cardinality_budget") in reasons
    assert ("invalid_precision", "invalid_precision_tier") in reasons
    assert all(failure.code == E_CONFIG_INVALID for failure in experiment.failures)


def test_invalid_scenario_does_not_block_valid_scenario(make_experiment) -> None:
    path = make_experiment()
    experiment = load_experiment(path)
    assert experiment.failures == []
    assert experiment.scenarios[0].scenario_id == "compact"
