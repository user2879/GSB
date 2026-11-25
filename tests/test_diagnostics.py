from __future__ import annotations

from log_sampler.diagnostics import (
    _classify_flip,
    analyze_reservoir_admission,
    analyze_truncation_boundary,
)


def test_random_source_boundary_flip_is_classified():
    numerator = int(0.499 * (1 << 256))
    event = {
        "arrival_ordinal": 4,
        "threshold_numerator": 2,
        "threshold_denominator": 4,
        "random_numerator_hex": f"{numerator:064x}",
        "random_unit": numerator / (1 << 256),
        "random_digest": "00" * 32,
        "log_id": "log-boundary",
        "bucket_id": "edge-a|info",
    }
    result = analyze_reservoir_admission(
        event, {"decimals": 2, "tier": "coarse"}
    )
    assert result["reference_decision"] == "admit"
    assert result["fixed_decision"] == "reject"
    assert result["difference_direction"] == "fixed_rejects_reference_admits"
    assert result["flip_source"] == "RANDOM_SOURCE_DIGEST"
    assert result["business_code"] == "E_D_FLIP_RANDOM_SOURCE"


def test_flip_classifier_distinguishes_ordinal_and_mixed():
    assert (
        _classify_flip(
            exact_random=0.4998,
            fixed_random=0.4998,
            exact_threshold=0.50,
            fixed_threshold=0.499,
        )
        == "ORDINAL_CALCULATION"
    )
    assert (
        _classify_flip(
            exact_random=0.4998,
            fixed_random=0.50,
            exact_threshold=0.50,
            fixed_threshold=0.499,
        )
        == "MIXED"
    )


def test_truncation_boundary_tier_flip_uses_business_code():
    left = {
        "log_id": "left",
        "bucket_id": "b1",
        "global_rank": 3,
        "priority": 10,
        "arrival_ordinal": 2,
        "metric": 1.249,
        "quantized_weight": 1250,
    }
    right = {
        "log_id": "right",
        "bucket_id": "b2",
        "global_rank": 4,
        "priority": 10,
        "arrival_ordinal": 3,
        "metric": 1.250,
        "quantized_weight": 1250,
    }
    result = analyze_truncation_boundary(
        left, right, {"decimals": 2, "tier": "coarse"}
    )
    assert result["tier_flip"] is True
    assert result["flip_source"] == "TRUNCATION_TIE"
    assert result["business_code"] == "E_D_FLIP_TRUNCATION_TIE"
    assert result["half_quantum"] == 50

