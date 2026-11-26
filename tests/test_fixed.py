from __future__ import annotations

import pytest

from gsb.arbitration import WaterCandidate, arbitrate_budget, arbitrate_water
from gsb.fixed import MAX_I64, analyze_fixed_error, water_level
from gsb.spec import parse_spec
from helpers import base_raw_spec


def test_water_uses_integer_floor_and_exact_remainders() -> None:
    level = water_level("node-a", 1, 3, 1000, lease_limit_units=2)
    assert level.utilization_fp == 333
    assert level.utilization_remainder == 1
    assert level.headroom_fp == 666
    assert level.headroom_remainder == 2
    assert level.margin_units == 2
    assert level.quota_fp == 3000
    assert level.used_fp == 1000
    assert level.lease_limit_fp == 2000
    assert level.truncation == "floor_non_negative"
    assert level.overflow_rule == "saturate_i64"


def test_water_overflow_saturates() -> None:
    level = water_level("node-a", MAX_I64, MAX_I64, 100_000)
    assert level.utilization_fp == 100_000
    assert level.unit_overflowed is True
    assert level.ratio_overflowed is False
    assert level.overflowed is True
    saturated = water_level("node-a", MAX_I64, 1, 100_000)
    assert saturated.utilization_fp == MAX_I64
    assert saturated.ratio_overflowed is True
    assert saturated.overflowed is True


def test_error_breakdown_and_dominant_rule() -> None:
    model = analyze_fixed_error(4, 1000)
    assert model.per_shard_quantization_fp == 1
    assert model.per_node_comparison_bound_fp == 2
    assert model.global_aggregation_bound_fp == 4
    assert model.quantization_contribution_fp == 8
    assert model.comparison_contribution_fp == 2
    assert model.aggregation_contribution_fp == 4
    assert model.total_bound_fp == 14
    assert model.dominant_segment == "quantization_truncation"


def test_water_boundary_uses_stable_topology_and_id() -> None:
    spec = parse_spec(base_raw_spec())
    model = analyze_fixed_error(len(spec.nodes), spec.fixed_scale)
    first = WaterCandidate("raw-a", "node-a", 3, water_level("node-a", 1, 3, 1000))
    second = WaterCandidate("raw-b", "node-b", 3, water_level("node-b", 2, 6, 1000))
    result = arbitrate_water(first, second, topo_rank=spec.topo_rank(), error_model=model)
    assert result["decision"] == "raw-a"
    assert result["error_affected"] is True
    assert result["difference"]["utilization_fp"] == 0
    assert result["error_bound"]["per_node_comparison_bound_fp"] == 2
    assert "stable_topological_order" in result["decision_basis"]


def test_water_outside_bound_selects_lower_utilization() -> None:
    spec = parse_spec(base_raw_spec())
    model = analyze_fixed_error(len(spec.nodes), spec.fixed_scale)
    first = WaterCandidate("raw-a", "node-a", 3, water_level("node-a", 5, 10, 1000))
    second = WaterCandidate("raw-b", "node-b", 3, water_level("node-b", 1, 10, 1000))
    result = arbitrate_water(first, second, topo_rank=spec.topo_rank(), error_model=model)
    assert result["decision"] == "raw-b"
    assert result["error_affected"] is False


def test_budget_exact_tie_is_stable_and_flagged() -> None:
    spec = parse_spec(base_raw_spec())
    model = analyze_fixed_error(len(spec.nodes), spec.fixed_scale)
    level = water_level("node-a", 1, 3, 1000)
    first = WaterCandidate("raw-a", "node-a", 4, level)
    second = WaterCandidate("raw-b", "node-b", 4, level)
    result = arbitrate_budget(
        4,
        first,
        second,
        topo_rank=spec.topo_rank(),
        error_model=model,
        budget_error_bound_units=0,
    )
    assert result["decision"] == "raw-a"
    assert result["difference_units"] == 0
    assert result["admitted"] is True
    assert result["error_affected"] is True


def test_budget_infeasible_defers_stably() -> None:
    spec = parse_spec(base_raw_spec())
    model = analyze_fixed_error(len(spec.nodes), spec.fixed_scale)
    level = water_level("node-a", 1, 3, 1000)
    first = WaterCandidate("raw-a", "node-a", 4, level)
    second = WaterCandidate("raw-b", "node-b", 5, level)
    result = arbitrate_budget(3, first, second, topo_rank=spec.topo_rank(), error_model=model)
    assert result["decision"] == "raw-a"
    assert result["admitted"] is False
    assert result["defer"] is True


def test_budget_bound_cannot_exceed_total_fixed_error() -> None:
    spec = parse_spec(base_raw_spec())
    model = analyze_fixed_error(len(spec.nodes), spec.fixed_scale)
    level = water_level("node-a", 1, 3, 1000)
    first = WaterCandidate("raw-a", "node-a", 4, level)
    second = WaterCandidate("raw-b", "node-b", 4, level)
    with pytest.raises(ValueError):
        arbitrate_budget(
            4,
            first,
            second,
            topo_rank=spec.topo_rank(),
            error_model=model,
            budget_error_bound_units=1,
        )
