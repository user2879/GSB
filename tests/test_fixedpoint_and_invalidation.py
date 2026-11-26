from __future__ import annotations

from lease_arbitrator.fixedpoint import (
    attribute_error_segments,
    global_aggregation_bound_fp,
    make_scale,
    water_level_fp,
)
from lease_arbitrator.invalidation import plan_edit_invalidation
from lease_arbitrator.spec import validate_spec


def test_fixed_point_water_and_bounds() -> None:
    scale = make_scale(2)
    level = water_level_fp(3, 6, scale)
    assert level["level_fp"] == 50
    assert level["remaining_q"] == 300
    assert global_aggregation_bound_fp(3) == 4
    attribution = attribute_error_segments(
        quantization_ulp=4,
        scheduling_order_ulp=0,
        generation_reorder_ulp=1,
        scale=scale,
    )
    assert attribution["total_bound_ulp"] == 5
    assert attribution["bound_units_exact"] == "5/100"
    assert attribution["dominant_segment"] == "quantization_truncation"


def test_invalidation_order_is_stable() -> None:
    spec = validate_spec(
        {
            "scale": 0,
            "priority_range": [0, 10],
            "global_budget": 8,
            "nodes": [
                {
                    "node_id": "b",
                    "host": "127.0.0.1",
                    "port": 24002,
                    "quota": 4,
                    "max_active": 1,
                    "seed": 2,
                    "quota_version": 1,
                },
                {
                    "node_id": "a",
                    "host": "127.0.0.1",
                    "port": 24001,
                    "quota": 4,
                    "max_active": 1,
                    "seed": 1,
                    "quota_version": 1,
                },
            ],
            "leases": [
                {
                    "lease_id": "z-dependent",
                    "node_id": "a",
                    "units": 1,
                    "priority": 1,
                    "hold_ms": 10,
                    "references": ["x-base"],
                },
                {
                    "lease_id": "x-base",
                    "node_id": "b",
                    "units": 1,
                    "priority": 1,
                    "hold_ms": 10,
                },
            ],
        }
    )
    result = plan_edit_invalidation(spec, {"edit_type": "delete_lease", "lease_id": "x-base"})
    assert result["invalid_lease_ids"] == ["z-dependent", "x-base"]
    assert [item["lease_id"] for item in result["invalid_set"]] == ["z-dependent", "x-base"]
    assert any(step["reason"] == "lease_reference" for step in result["propagation_steps"])


def test_remove_node_propagates_in_node_lease_order() -> None:
    spec = validate_spec(
        {
            "scale": 0,
            "priority_range": [0, 10],
            "global_budget": 8,
            "nodes": [
                {
                    "node_id": "a",
                    "host": "127.0.0.1",
                    "port": 24011,
                    "quota": 4,
                    "max_active": 1,
                    "seed": 1,
                    "quota_version": 1,
                }
            ],
            "leases": [
                {"lease_id": "z", "node_id": "a", "units": 1, "priority": 1, "hold_ms": 1},
                {"lease_id": "a", "node_id": "a", "units": 1, "priority": 1, "hold_ms": 1},
            ],
        }
    )
    result = plan_edit_invalidation(spec, {"edit_type": "remove_node", "node_id": "a"})
    assert result["invalid_lease_ids"] == ["a", "z"]
