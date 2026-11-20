"""全局点数预算与单视图配额调度测试。"""

from __future__ import annotations

from conftest import run_pipeline, write_config
from mvrwb.budget import BudgetLedger
from mvrwb.errors import E_POINT_BUDGET_EXHAUSTED


def test_ledger_deduction_is_stable_and_auditable():
    ledger = BudgetLedger(global_budget=10, per_view_quota=5)
    for view_id in (1, 2, 3):
        ledger.reserve(view_id)
    accepted, exhausted, _ = ledger.settle(1, 4)
    assert (accepted, exhausted) == (4, False)
    accepted, exhausted, _ = ledger.settle(2, 8)
    # 只剩 6 个预算，请求 8 => 接受 6、截断 2、触发耗尽
    assert (accepted, exhausted) == (6, True)
    accepted, exhausted, _ = ledger.settle(3, 3)
    # 预算已空：接受 0 且保持队列稳定位置
    assert (accepted, exhausted) == (0, True)
    assert ledger.queue_order == [1, 2, 3]
    assert ledger.exhausted_views == [2, 3]
    report = ledger.to_dict()
    assert [entry["view_id"] for entry in report["deduction_sequence"]] == [1, 2, 3]
    assert report["consumed"] == 10 and report["remaining"] == 0


def test_pipeline_emits_budget_exhausted_and_keeps_queue(sample_dir, work_dirs, tmp_path):
    from mvrwb.editor import apply_edit_stream
    apply_edit_stream(sample_dir, work_dirs["work"], None)
    # 极小预算强制触发耗尽
    write_config(work_dirs["work"], {"global_point_budget": 12})
    report = run_pipeline(sample_dir, work_dirs)
    codes = [failure["code"] for failure in report["failures"]]
    assert E_POINT_BUDGET_EXHAUSTED in codes
    ledger = report["budget_ledger"]
    # 队列顺序始终按视图序号稳定，耗尽视图仍保留在队列中
    assert ledger["queue_order"] == sorted(ledger["queue_order"])
    assert ledger["consumed"] <= 12
    exhausted = ledger["exhausted_views"]
    assert exhausted == sorted(exhausted) and exhausted


def test_budget_outcome_is_deterministic_across_worker_counts(sample_dir, work_dirs, tmp_path):
    from mvrwb.editor import apply_edit_stream
    results = []
    for workers in (1, 8):
        dirs = {
            "work": str(tmp_path / f"w{workers}"),
            "snapshot": str(tmp_path / f"s{workers}"),
            "report": str(tmp_path / f"r{workers}.json"),
        }
        apply_edit_stream(sample_dir, dirs["work"], None)
        write_config(dirs["work"], {"global_point_budget": 12, "thread_workers": workers})
        results.append(run_pipeline(sample_dir, dirs)["budget_ledger"])
    assert results[0] == results[1]
