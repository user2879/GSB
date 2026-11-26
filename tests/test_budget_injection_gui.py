import time

import pytest

from vertical_workbench.clock import ManualClock
from vertical_workbench.config import WorkbenchConfig
from vertical_workbench.injection import run_injection_suite
from vertical_workbench.scheduler import plan_point_budget, run_ordered
from vertical_workbench.studio import Studio


def test_budget_plan_is_stable_and_marks_quota():
    units = [
        {"unit_id": "a:c0", "image_id": "a", "column_rank": 0, "planned_points": 4, "stable_order": 0},
        {"unit_id": "a:c1", "image_id": "a", "column_rank": 1, "planned_points": 9, "stable_order": 1},
        {"unit_id": "b:c0", "image_id": "b", "column_rank": 0, "planned_points": 3, "stable_order": 2},
    ]
    plan = plan_point_budget(units, [], global_point_budget=6, column_point_quota=4)
    assert plan.approved_ids == ["a:c0"]
    assert plan.quota_blocked_ids == ["a:c1"]
    assert plan.pending_ids == ["b:c0"]
    assert {event["code"] for event in plan.events} == {"E_QUOTA_EXCEEDED", "E_POINT_BUDGET_PENDING"}


def test_pending_queue_has_next_batch_priority(tmp_path):
    config = WorkbenchConfig(sample_count=2, global_point_budget=15, stride=64, max_active_leases=16)
    studio = Studio.create(tmp_path, config, ManualClock(42))
    first = studio.engine.run(1)
    assert first["pending_queue"]
    pending = first["pending_queue"][0]["unit_id"]
    raw = studio.read_config_raw()
    raw["global_point_budget"] = 500
    studio.write_config_raw(raw)
    second = studio.engine.run(2)
    assert second["budget"]["candidate_ids"][0] == pending
    assert second["budget"]["approved_ids"][0] == pending


def test_run_ordered_preserves_input_order():
    def worker(value):
        time.sleep(max(0.0, (29 - value) * 0.001))
        return value * value

    assert run_ordered(list(range(30)), worker, 4) == [value * value for value in range(30)]


def test_injection_suite_is_isolated(tmp_path):
    report = run_injection_suite(tmp_path)
    assert report["all_injections_hit"]
    assert report["good_samples_isolated"]
    assert len(report["processed_sample_ids"]) == 4


def test_gui_smoke_or_skip(tmp_path):
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except Exception as exc:
        pytest.skip(f"no display/Tk runtime: {exc}")
    config = WorkbenchConfig(sample_count=1, stride=64)
    Studio.create(tmp_path, config, ManualClock(7))
    from vertical_workbench.gui import WorkbenchGUI

    gui = WorkbenchGUI(root, tmp_path)
    root.update_idletasks()
    root.update()
    assert gui.result["samples"]
    root.destroy()
