from __future__ import annotations

from driftwatch.fixedpoint import judge_drift
from driftwatch.merge import merge_metric
from driftwatch.models import HostSpec, MergedResult, Observation

from .conftest import make_criteria


def _obs(agent: str, sequence: int, value: int) -> Observation:
    return Observation(agent, sequence, "host-a", "cpu_quota", value, f"agg-{agent}", 1)


def _host(expected: int, tolerance: int) -> HostSpec:
    return HostSpec(
        host_id="host-a",
        expected={"cpu_quota": expected, "memory_pages": 100, "processes": 100},
        tolerance_bps={"cpu_quota": tolerance, "memory_pages": 100, "processes": 100},
    )


def test_authority_order_and_source_deviation_are_stable():
    merged = merge_metric(
        1,
        "host-a",
        "cpu_quota",
        [_obs("a1", 9, 100), _obs("a2", 10, 120), _obs("a3", 10, 110)],
        2,
    )
    assert merged.status == "AUTHORITATIVE"
    assert merged.authority_agent == "a2"
    assert merged.authority_value == 120
    assert merged.source_count == 3
    assert merged.max_abs_deviation == 20
    assert merged.sources[0]["deviation"] == 0
    assert merged.sources[1]["deviation"] == -10
    assert merged.sources[2]["deviation"] == -20


def test_duplicate_agent_records_do_not_satisfy_min_sources():
    merged = merge_metric(
        1,
        "host-a",
        "cpu_quota",
        [_obs("a1", 9, 100), _obs("a1", 10, 110)],
        2,
    )
    assert merged.sparse is True
    assert merged.status == "SPARSE"
    assert merged.error_code == "E_SPARSE"
    assert merged.authority_value is None


def test_fixed_point_clear_accept_reject_and_ambiguous_boundary():
    accept_merged = merge_metric(1, "host-a", "cpu_quota", [_obs("a1", 1, 105), _obs("a2", 1, 105)], 2)
    accept = judge_drift(1, "host-a", "cpu_quota", _host(100, 1000), accept_merged, make_criteria(scale_exp=3))
    assert accept.decision == "ACCEPT"
    assert accept.error_affected is False
    assert accept.error_bound is not None
    assert accept.error_bound.composite == 1001
    assert accept.drift_ratio_bps == 500

    reject_merged = merge_metric(1, "host-a", "cpu_quota", [_obs("a1", 1, 200), _obs("a2", 1, 200)], 2)
    reject = judge_drift(1, "host-a", "cpu_quota", _host(100, 100), reject_merged, make_criteria(scale_exp=0))
    assert reject.decision == "REJECT"
    assert reject.error_affected is False
    assert reject.drift_excess_fixed > (reject.allowed_fixed or 0) + (reject.error_bound.composite if reject.error_bound else 0)

    ambiguous_merged = merge_metric(1, "host-a", "cpu_quota", [_obs("a1", 1, 106), _obs("a2", 1, 106)], 2)
    ambiguous = judge_drift(1, "host-a", "cpu_quota", _host(100, 500), ambiguous_merged, make_criteria(scale_exp=0))
    assert ambiguous.decision == "REJECT"
    assert ambiguous.error_affected is True
    assert "ambiguous" in ambiguous.reason

    tie_accept_merged = merge_metric(1, "host-a", "cpu_quota", [_obs("a1", 1, 105), _obs("a2", 1, 105)], 2)
    tie_accept = judge_drift(
        1,
        "host-a",
        "cpu_quota",
        _host(100, 500),
        tie_accept_merged,
        make_criteria(scale_exp=0),
    )
    assert tie_accept.decision == "ACCEPT"
    assert tie_accept.error_affected is True
    assert "tolerance-floor" in tie_accept.reason


def test_sparse_never_asserts_an_authority_or_drift_decision():
    merged = merge_metric(1, "host-a", "cpu_quota", [_obs("a1", 1, 999)], 2)
    result = judge_drift(1, "host-a", "cpu_quota", _host(100, 100), merged, make_criteria())
    assert result.decision == "UNKNOWN"
    assert result.error_code == "E_SPARSE"
    assert result.observed is None
    assert result.error_bound is None


def test_fixed_point_saturates_large_values():
    merged = MergedResult(
        snapshot_seq=1,
        host_id="host-a",
        metric_id="cpu_quota",
        status="AUTHORITATIVE",
        authority_agent="a1",
        authority_observation_seq=1,
        authority_value=10**30,
        aggregation_version="v1",
        source_count=2,
        sparse=False,
        sources=[],
        max_abs_deviation=10**30,
    )
    result = judge_drift(1, "host-a", "cpu_quota", _host(10**20, 100), merged, make_criteria(scale_exp=12))
    assert result.saturated is True
    assert result.error_bound is not None and result.error_bound.saturated is True
    assert result.drift_excess_fixed == (1 << 63) - 1
