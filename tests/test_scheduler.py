import dataclasses

from pathlib import Path

from mvcc_lab.executor import ReplayExecutor
from mvcc_lab.loader import load_spec
from mvcc_lab.scheduler import Scheduler

ROOT = Path(__file__).parents[1]
SPEC = ROOT / "samples" / "sample_spec.json"


def test_scheduler_respects_both_limits():
    spec = load_spec(SPEC)
    with ReplayExecutor(SPEC) as executor:
        scheduler = Scheduler(spec, batch_size=2)
        summaries = scheduler.run(executor)
    scheduler.verify_within_limits()
    peaks = scheduler.peaks()
    assert peaks["global_peak"] <= spec.global_concurrency
    for rp_id, peak in peaks["rp_peak"].items():
        assert peak <= spec.per_readpoint_quota, rp_id
    assert len(summaries) == len(spec.queries)


def test_partial_admission_never_overcommits():
    # 每读点配额 1、全局 1：任何时刻最多一个任务放行。
    spec = load_spec(SPEC)
    spec = dataclasses.replace(spec, global_concurrency=1)
    with ReplayExecutor(SPEC) as executor:
        scheduler = Scheduler(spec, batch_size=1)
        scheduler.run(executor)
    scheduler.verify_within_limits()
    peaks = scheduler.peaks()
    assert peaks["global_peak"] == 1
    assert all(peak == 1 for peak in peaks["rp_peak"].values())


def test_trace_records_monotonic_wait_windows():
    spec = load_spec(SPEC)
    with ReplayExecutor(SPEC) as executor:
        scheduler = Scheduler(spec, batch_size=2)
        scheduler.run(executor)
    traces = scheduler.occupancy_trace()
    assert len(traces) == len(spec.queries)
    for trace in traces:
        # 单调时钟毫秒：各区间非负且时间点单调不减。
        assert trace["rp_wait_end_ms"] >= trace["rp_wait_start_ms"]
        assert trace["global_wait_end_ms"] >= trace["global_wait_start_ms"]
        assert trace["run_end_ms"] >= trace["run_start_ms"]
        assert trace["run_start_ms"] >= trace["global_wait_end_ms"]
        assert trace["global_held"] <= spec.global_concurrency
        assert trace["rp_held"] <= spec.per_readpoint_quota


def test_trace_is_deterministic_across_runs():
    spec = load_spec(SPEC)
    with ReplayExecutor(SPEC) as executor:
        first = Scheduler(spec, batch_size=2)
        first.run(executor)
        first_ids = [(t["query_id"], t["read_point_id"], t["row_key"],
                      t["batch_id"]) for t in first.occupancy_trace()]
    with ReplayExecutor(SPEC) as executor:
        second = Scheduler(spec, batch_size=2)
        second.run(executor)
        second_ids = [(t["query_id"], t["read_point_id"], t["row_key"],
                       t["batch_id"]) for t in second.occupancy_trace()]
    assert first_ids == second_ids
