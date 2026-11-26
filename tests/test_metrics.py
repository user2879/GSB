from __future__ import annotations

from hopledger.canonical import canonical_json
from hopledger.fixedpoint import I64_MAX, quantize_component, saturate_i64
from hopledger.metrics import attribute_errors, build_conclusion
from hopledger.specs import ErrorModelConfig


NODE_IDS = ["client", "fwd1", "rev2", "origin"]


def make_receipts():
    return [
        {
            "node_id": "fwd1",
            "request_id": "req-0001",
            "kind": "hop",
            "config_version": 1,
            "scale_factor": 1,
            "queue_ms": 1,
            "process_ms": 1,
            "forward_ms": 1,
        },
        {
            "node_id": "rev2",
            "request_id": "req-0001",
            "kind": "hop",
            "config_version": 1,
            "scale_factor": 1000,
            "queue_ms": 1,
            "process_ms": 1,
            "forward_ms": 1,
        },
        {
            "node_id": "origin",
            "request_id": "req-0001",
            "kind": "origin",
            "config_version": 1,
            "scale_factor": 1000,
            "queue_ms": 0,
            "process_ms": 0,
            "forward_ms": 0,
        },
    ]


def make_conclusion():
    return build_conclusion(
        request_id="req-0001",
        node_ids=NODE_IDS,
        receipts=make_receipts(),
        session_scale=1000,
        scale_caliber_version=1,
        hop_set_version=1,
    )


def test_fixed_point_chain_total_and_order_are_deterministic():
    conclusion = make_conclusion()
    assert conclusion["e2e_total_units"] == 6000
    assert conclusion["accumulation_order"] == ["fwd1", "rev2", "origin"]
    assert [node["cumulative_units"] for node in conclusion["nodes"]] == [3000, 6000, 6000]
    assert conclusion["config_versions"] == {
        "client": 0,
        "fwd1": 1,
        "rev2": 1,
        "origin": 1,
    }

    reordered = build_conclusion(
        request_id="req-0001",
        node_ids=NODE_IDS,
        receipts=list(reversed(make_receipts())),
        session_scale=1000,
        scale_caliber_version=1,
        hop_set_version=1,
    )
    assert canonical_json(reordered) == canonical_json(conclusion)


def test_truncation_and_saturation_rules_are_integer_and_bounded():
    component = quantize_component("queue", 1, 10_000, 1_000)
    assert component.units == 1000
    assert component.remainder_numerator == 0
    assert saturate_i64(I64_MAX + 10) == I64_MAX


def test_error_attribution_dominates_at_coarse_first_hop():
    conclusion = make_conclusion()
    attribution = attribute_errors(conclusion, ErrorModelConfig())
    assert attribution["primary_cause"] == "dominant_hop_identified"
    assert attribution["dominant_hop"] == "fwd1"
    assert attribution["declared_dominant_hop"] == "fwd1"
    assert attribution["dominance_threshold_bp"] == 6000
    shares = [node["contribution_bp"] for node in attribution["nodes"]]
    assert sum(shares) == 10_000
    source_total = sum(attribution["source_totals"].values())
    assert attribution["end_to_end_error_upper_bound_units"] == source_total


def test_error_attribution_classifies_version_conflict():
    conclusion = make_conclusion()
    attribution = attribute_errors(
        conclusion,
        ErrorModelConfig(),
        expected_config_versions={"fwd1": 2, "rev2": 1, "origin": 1},
    )
    assert attribution["primary_cause"] == "config_version_conflict"
    assert attribution["version_conflicts"][0]["node_id"] == "fwd1"
    assert attribution["version_conflicts"][0]["field"] == "config_version"


def test_error_attribution_classifies_high_failure_ratio():
    conclusion = make_conclusion()
    attribution = attribute_errors(
        conclusion,
        ErrorModelConfig(),
        failure_ratio_bp_by_node={"fwd1": 8000, "rev2": 0, "origin": 0},
    )
    assert attribution["primary_cause"] == "failure_injection_ratio_high"
    assert attribution["high_failure_nodes"] == ["fwd1"]


def test_error_attribution_classifies_indistinguishable():
    conclusion = make_conclusion()
    attribution = attribute_errors(
        conclusion,
        ErrorModelConfig(dominance_threshold_bp=9999, indistinguishable_margin_bp=0),
    )
    assert attribution["primary_cause"] == "error_indistinguishable"
    assert attribution["declared_dominant_hop"] is None
