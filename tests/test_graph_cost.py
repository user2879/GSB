from __future__ import annotations

from decimal import Decimal

from joinlab.cost import MAX_FIXED, quantize
from joinlab.graph import compile_join_graph, enumerate_candidates
from joinlab.stats import collect_statistics
from joinlab.session import ExperimentSession

from conftest import WHITESPACE_SAMPLE, with_tuning
from joinlab import load_workload


def test_candidate_enumeration_is_connected_and_deterministic(workload):
    graph = compile_join_graph(workload.query, workload.table_map())
    enumeration = enumerate_candidates(workload, graph)
    assert enumeration.total_before_limit == 36
    assert len(enumeration.plans) == 18
    assert len(enumeration.omitted_candidates) == 18
    assert all(item["reason"] == "CANDIDATE_LIMIT" for item in enumeration.omitted_candidates)
    assert enumeration.pruned_orders
    assert all(item["reason"] == "CARTESIAN_WITHOUT_EQUALITY" for item in enumeration.pruned_orders)
    joined = set()
    for plan in enumeration.plans:
        joined = {plan.table_order[0]}
        for step in plan.steps:
            assert step.predicates
            assert any(
                predicate.left.table == step.right_table
                or predicate.right.table == step.right_table
                for predicate in step.predicates
            )
            joined.add(step.right_table)
    assert len({plan.candidate_id for plan in enumeration.plans}) == 18


def test_candidate_limit_is_explicit_and_deterministic(workload):
    limited_workload = with_tuning(workload, max_candidates=4)
    graph = compile_join_graph(limited_workload.query, limited_workload.table_map())
    limited = enumerate_candidates(limited_workload, graph)
    assert len(limited.plans) == 4
    assert limited.total_before_limit == 36
    assert len(limited.omitted_candidates) == 32
    assert limited.applied_limit == 4


def test_query_whitespace_produces_same_graph_and_candidate_set(workload):
    other = load_workload(WHITESPACE_SAMPLE)
    first = ExperimentSession(workload)
    second = ExperimentSession(other)
    assert first.graph.digest == second.graph.digest
    assert first.enumeration.candidate_ids == second.enumeration.candidate_ids
    assert [item.total_cost_fx for item in first.estimates] == [
        item.total_cost_fx for item in second.estimates
    ]


def test_fixed_point_truncation_and_saturation():
    assert quantize(Decimal("1.9999"), 1000) == 1999
    assert quantize(Decimal("-2.5"), 1000) == 0
    assert quantize(Decimal("1e100"), 1000) == MAX_FIXED


def test_operator_estimates_expose_versions_and_error_bounds(workload):
    session = ExperimentSession(workload)
    estimate = session.best_estimate
    report = estimate.as_dict(workload.tuning.fixed_point_scale)
    assert report["fixed_point"]["truncation"] == "toward_zero"
    assert report["fixed_point"]["plan_error_bound_fx"] == len(estimate.operators)
    assert all(operator["hist_versions"] for operator in report["operators"])
    assert any(operator["stats_versions"] for operator in report["operators"])
    assert estimate.output_rows_fx > 0


def test_adaptive_statistics_keep_both_calibers(workload):
    stats = collect_statistics(workload)
    by_column = {item.column: item for item in stats.filter_stats}
    assert by_column["active"].selected.caliber == "exact"
    assert by_column["active"].exact.scan_rows == 60
    assert by_column["amount"].selected.caliber == "sample"
    assert by_column["amount"].sampled.sample_size == 128
    assert by_column["amount"].selected.scan_rows == 128
    assert by_column["amount"].exact.scan_rows == 480
    assert 0 <= by_column["amount"].sampled.confidence_low <= by_column["amount"].sampled.confidence_high <= 1
    assert by_column["amount"].difference == (
        by_column["amount"].exact.selectivity - by_column["amount"].sampled.selectivity
    )
    assert by_column["amount"].as_dict(1000)["difference_fx"] < 0


def test_selection_uses_cost_then_stable_id(workload):
    session = ExperimentSession(workload)
    ordered = sorted(
        session.estimates,
        key=lambda estimate: (estimate.total_cost_fx, estimate.plan.candidate_id),
    )
    assert session.selection["best_candidate_id"] == ordered[0].plan.candidate_id
    assert [row["candidate_id"] for row in session.selection["ranking"]] == [
        estimate.plan.candidate_id for estimate in ordered
    ]
