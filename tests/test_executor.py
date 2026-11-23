from __future__ import annotations

from dataclasses import replace

import pytest

from joinlab import LabError, load_workload
from joinlab.cost import estimate_plan
from joinlab.executor import bias_report, compare_execution_results, execute_plan
from joinlab.graph import compile_join_graph, enumerate_candidates
from joinlab.loader import rebuild_workload
from joinlab.session import ExperimentSession
from joinlab.stats import collect_statistics


def test_all_enumerated_algorithms_agree_bytewise(workload):
    session = ExperimentSession(workload)
    expected = None
    algorithm_set = set()
    for estimate in session.estimates:
        run = execute_plan(workload, estimate.plan, estimate)
        algorithm_set.update(step.algorithm for step in estimate.plan.steps)
        if expected is None:
            expected = run
        else:
            assert run.result.raw_bytes == expected.result.raw_bytes
            assert run.result.grouped_bytes == expected.result.grouped_bytes
    baseline = execute_plan(workload, session.baseline_estimate.plan, session.baseline_estimate)
    assert baseline.result.raw_digest == expected.result.raw_digest
    assert baseline.result.grouped_digest == expected.result.grouped_digest
    assert {"HASH", "MERGE", "NESTED_LOOP"}.issubset(algorithm_set)


def test_scheduler_trace_respects_all_three_limits(workload):
    session = ExperimentSession(workload)
    run = execute_plan(workload, session.best_estimate.plan, session.best_estimate)
    constraints = run.schedule.constraints
    assert constraints["global_limit_respected"] is True
    assert constraints["table_quota_respected"] is True
    assert constraints["memory_budget_respected"] is True
    assert constraints["observed_max_global_running"] <= workload.tuning.global_concurrency
    assert max(constraints["observed_max_per_table_running"].values()) <= workload.tuning.per_table_scan_quota
    assert constraints["observed_max_memory_in_use_bytes"] <= workload.tuning.memory_budget_bytes
    assert all(trace.completed_rows >= 0 for trace in run.schedule.traces)
    assert all(trace.end_ns >= trace.start_ns for trace in run.schedule.traces)


def test_diverge_reports_first_group_key_and_field(workload):
    session = ExperimentSession(workload)
    best = execute_plan(workload, session.best_estimate.plan, session.best_estimate)
    baseline = execute_plan(
        workload,
        session.baseline_estimate.plan,
        session.baseline_estimate,
        fault_mode="drop_last_row",
    )
    with pytest.raises(LabError) as caught:
        compare_execution_results(best, baseline)
    error = caught.value
    assert error.code == "E_DIVERGE"
    assert error.field in {"total_amount", "order_count", "row_presence"}
    assert error.context["group_key"] is not None


def test_bias_threshold_reports_dominant_operator(workload):
    biased = rebuild_workload(
        workload.tables,
        workload.query,
        replace(workload.tuning, bias_threshold_ratio=2.0),
    )
    session = ExperimentSession(biased)
    run = execute_plan(biased, session.best_estimate.plan, session.best_estimate)
    with pytest.raises(LabError) as caught:
        bias_report(
            session.best_estimate,
            run,
            threshold_ratio=2.0,
            scale=biased.tuning.fixed_point_scale,
        )
    assert caught.value.code == "E_BIAS"
    assert caught.value.operator_id == max(
        session.best_estimate.operators, key=lambda operator: operator.cost_fx
    ).operator_id

