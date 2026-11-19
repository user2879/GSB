from __future__ import annotations

import copy

from spatial_diag.config import Config
from spatial_diag.strategy import select_and_fit


def test_strategy_ledger_covers_scale_adaptation(config, committed_workspace):
    _, report = committed_workspace
    strategies = {name: value["strategy"] for name, value in report["tiers"].items()}
    assert strategies["small"] == "coarse_to_fine"
    assert strategies["medium"] == "full_least_squares"
    assert strategies["large"] == "clustered_representatives"
    for name, value in report["tiers"].items():
        budget = config.tier(name)["memory_budget_bytes"]
        assert value["peak_memory_bytes"] <= budget


def test_budget_change_only_changes_budget_dependent_decision(config, committed_workspace):
    workspace, report = committed_workspace
    from spatial_diag.snapshot import SnapshotStore

    snapshot = SnapshotStore(workspace).load_current()
    projection = snapshot["tiers"]["medium"]["projection"]
    generous_data = copy.deepcopy(config.data)
    tight_data = copy.deepcopy(config.data)
    for tier in generous_data["tiers"]:
        if tier["name"] == "medium":
            tier["memory_budget_bytes"] = 131072
    for tier in tight_data["tiers"]:
        if tier["name"] == "medium":
            tier["memory_budget_bytes"] = 70000
    generous = Config(generous_data)
    tight = Config(tight_data)
    _, generous_decision = select_and_fit(generous, generous.tier("medium"), projection)
    _, tight_decision = select_and_fit(tight, tight.tier("medium"), projection)

    assert generous_decision["data_signature"] == tight_decision["data_signature"]
    assert generous_decision["inputs"]["high_confidence_directions"] == tight_decision[
        "inputs"
    ]["high_confidence_directions"]
    assert generous_decision["selected_strategy"] == "full_least_squares"
    assert tight_decision["selected_strategy"] == "clustered_representatives"
    assert tight_decision["peak_memory_bytes"] <= 70000
    assert tight_decision["cluster"]["all_directions_assigned_exactly_once"] is True
    assert tight_decision["cluster"]["cluster_conservation"] is True


def test_cross_tier_report_has_first_divergence_and_reason(committed_workspace):
    _, report = committed_workspace
    consistency = report["consistency"]
    assert len(consistency["pairs"]) == 3
    for pair in consistency["pairs"]:
        assert "divergence_ratio" in pair
        if not pair["consistent"]:
            divergence = pair["first_direction_divergence"]
            assert divergence is not None
            assert divergence["reason"] in {
                "sampling_density_insufficient",
                "order_mismatch",
                "strategy_choice_difference",
            }
            assert "sh_component_index" in divergence
            assert "azimuth_deg" in divergence and "elevation_deg" in divergence
