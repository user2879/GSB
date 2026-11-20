from __future__ import annotations

from bisect import bisect_right
from trace_diag.canonical import as_fraction
from trace_diag.generator import generate_records
from trace_diag.sampling import sample_records
from trace_diag.spec import parse_policies_document
from trace_diag.stats import analyze_policy, bucket_cut_values


def _all_head_policy(scale_power: int = 1, bucket_count: int = 8):
    return parse_policies_document(
        {
            "version": 1,
            "policies": [
                {
                    "id": "all-head",
                    "head_ratio_bp": 10_000,
                    "tail_duration_threshold_us": 1,
                    "tail_ratio_bp": 0,
                    "scale_power": scale_power,
                    "bucket_count": bucket_count,
                }
            ],
        }
    ).policies[0]


def test_bucket_equality_belongs_to_right_bucket():
    cuts = bucket_cut_values([10, 20, 30, 40], 2)
    assert cuts == [20]
    assert bisect_right(cuts, 19) == 0
    assert bisect_right(cuts, 20) == 1
    assert bisect_right(cuts, 21) == 1


def test_fixed_quantiles_and_attribution_are_reproducible(service_doc):
    service = service_doc.services[0]
    records = generate_records(service).records
    policy = _all_head_policy()
    sampling = sample_records(records, policy, 1)
    first = analyze_policy(
        service_id=service.id,
        records=records,
        sampling=sampling,
        policy=policy,
        strategy_version=1,
    )
    reversed_records = list(reversed(records))
    reversed_sampling = sample_records(reversed_records, policy, 1)
    second = analyze_policy(
        service_id=service.id,
        records=reversed_records,
        sampling=reversed_sampling,
        policy=policy,
        strategy_version=1,
    )
    assert first["bucket_stats"] == second["bucket_stats"]
    assert first["attribution"] == second["attribution"]
    assert first["quantiles"] == second["quantiles"]
    assert first["fixed_error"] == second["fixed_error"]


def test_attribution_decomposition_adds_up_and_bounds_hold(service_doc):
    service = service_doc.services[0]
    records = generate_records(service).records
    policy = _all_head_policy()
    sampling = sample_records(records, policy, 1)
    result = analyze_policy(
        service_id=service.id,
        records=records,
        sampling=sampling,
        policy=policy,
        strategy_version=1,
    )
    for bucket in result["attribution"]["buckets"]:
        contributions = bucket["contributions_us"]
        signed_sum = sum(as_fraction(value) for value in contributions.values())
        assert signed_sum == as_fraction(bucket["estimated_minus_true_us"])
        bounds = bucket["contribution_bounds_us"]
        assert sum(as_fraction(value) for value in bounds.values()) == as_fraction(
            bucket["total_error_bound_us"]
        )
        assert as_fraction(bucket["error_abs_us"]) <= as_fraction(
            bucket["total_error_bound_us"]
        )
    totals = result["attribution"]["totals"]
    signed_total = sum(as_fraction(value) for value in totals["contributions_us"].values())
    assert signed_total == as_fraction(totals["estimated_minus_true_us"])
    assert as_fraction(totals["error_abs_us"]) <= as_fraction(
        totals["total_error_bound_us"]
    )


def test_fixed_point_quantization_truncates_down(service_doc):
    service = service_doc.services[0]
    records = generate_records(service).records
    policy = _all_head_policy(scale_power=2)
    sampling = sample_records(records, policy, 1)
    result = analyze_policy(
        service_id=service.id,
        records=records,
        sampling=sampling,
        policy=policy,
        strategy_version=1,
    )
    quantum = 100
    assert result["fixed_error"]["quantum_us"] == quantum
    assert result["fixed_error"]["single_record_max_us"] == 99
    for quantile in result["quantiles"].values():
        assert quantile["full_fixed_us"] % quantum == 0
        assert quantile["estimated_fixed_us"] % quantum == 0
        assert as_fraction(quantile["total_error_bound_us"]) >= abs(
            as_fraction(quantile["difference_us"])
        )


def test_dominant_rule_is_explicit(service_doc):
    service = service_doc.services[0]
    records = generate_records(service).records
    policy = _all_head_policy()
    sampling = sample_records(records, policy, 1)
    result = analyze_policy(
        service_id=service.id,
        records=records,
        sampling=sampling,
        policy=policy,
        strategy_version=1,
    )
    totals = result["attribution"]["totals"]
    assert "dominant_rule" in totals
    assert totals["dominant_segment"] in {
        "sampling_sparse",
        "fixed_truncation",
        "bucket_boundary",
        "balanced",
        "none",
    }
