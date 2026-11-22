from __future__ import annotations

import hashlib

import pytest

from pageagg.catalog import load_catalog
from pageagg.engine import assert_route_equivalence, build_execution
from pageagg.errors import EBudget
from pageagg.planner import RESCAN, RESIDENT, decide_route, plan_slots
from pageagg.sample import generate_sample


def test_planner_is_deterministic_and_counts_hits() -> None:
    sequence = [1, 2, 3, 1, 4, 2, 3, 5, 6, 1, 7, 8]
    first = plan_slots(sequence, 4)
    second = plan_slots(sequence, 4)
    assert first == second
    assert first.hit_count == 3
    assert first.miss_count == 9
    assert [event.event_seq for event in first.evictions] == list(
        range(1, len(first.evictions) + 1)
    )
    assert all(event.victim_page_id != event.incoming_page_id for event in first.evictions)


def test_zero_budget_is_rejected_with_minimum() -> None:
    with pytest.raises(EBudget) as caught:
        plan_slots([1], 0)
    assert caught.value.details["minimum_feasible_slots"] == 1


def test_dual_routes_are_byte_identical(tmp_path) -> None:
    sample = generate_sample(tmp_path / "sample", force=True)
    catalog = load_catalog(sample["catalog_path"])
    resident = build_execution(catalog, 4, force_route="RESIDENT")
    rescan = build_execution(catalog, 4, force_route="RESCAN")
    equivalence = assert_route_equivalence(resident, rescan)
    assert equivalence["result_bytes_equal"]
    assert equivalence["eviction_bytes_equal"]
    assert resident.cache_hits == 3
    assert rescan.cache_hits == 0
    assert rescan.physical_scans == len(catalog.scan_sequence)
    assert resident.eviction_events == rescan.eviction_events
    assert resident.result_bytes() == rescan.result_bytes()
    assert hashlib.sha256(resident.result_bytes()).hexdigest() == equivalence["result_bytes_sha256"]


def test_route_decision_uses_boundary_and_tie_rule(tmp_path) -> None:
    sample = generate_sample(tmp_path / "sample", force=True)
    catalog = load_catalog(sample["catalog_path"])
    decision = decide_route(catalog, 4)
    assert decision.selected_route == RESIDENT
    assert decision.pressure_ppm == 375000
    assert decision.threshold_ppm == 400000
    assert decision.boundary_affected is True
    high_pressure = decide_route(catalog, 2)
    assert high_pressure.selected_route == RESCAN
    forced = decide_route(catalog, 2, force_route="resident")
    assert forced.selected_route == RESIDENT
    assert forced.forced is True


def test_error_bounds_grow_with_page_terms(tmp_path) -> None:
    sample = generate_sample(tmp_path / "sample", force=True)
    catalog = load_catalog(sample["catalog_path"])
    execution = build_execution(catalog, 4)
    group = execution.result["groups"][0]
    float_metric = next(metric for metric in group["metrics"] if metric["column"] == "value_float")
    quant = float_metric["quantization"]
    assert quant["page_term_count"] == len(quant["page_terms"])
    assert quant["float_values"] == sum(term["error_units"] for term in quant["page_terms"])
    assert quant["sum_error_bound_units"] == quant["float_values"]
    assert quant["mean_bound_rule"] == "ceil(float_values/value_count)+1"
    assert quant["page_terms"][0]["access_seq"] == 1

