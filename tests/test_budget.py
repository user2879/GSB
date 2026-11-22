
from mlimgedit.budget import make_feature_tasks, schedule_budget
from mlimgedit.errors import E_FEATURE_BUDGET_PENDING


def test_budget_consumes_stable_order_and_retains_pending():
    tasks = make_feature_tasks(["c", "a", "b"], cost=10)
    schedule = schedule_budget(tasks, budget=25)
    assert [task.sample for task in schedule.ready] == ["a", "b"]
    assert [task.sample for task in schedule.pending] == ["c"]
    assert schedule.pending_records[0].code == E_FEATURE_BUDGET_PENDING
    assert schedule.deductions[0].remaining_after == 15
    assert schedule.deductions[1].remaining_after == 5


def test_carried_pending_is_processed_first():
    carried = make_feature_tasks(["z"], cost=10)
    fresh = make_feature_tasks(["a", "m"], cost=10)
    schedule = schedule_budget(fresh, budget=20, carried_pending=carried)
    assert [task.sample for task in schedule.ready] == ["z", "a"]
    assert [task.sample for task in schedule.pending] == ["m"]


def test_deduction_sequence_is_replayable():
    tasks = make_feature_tasks([f"s{idx}" for idx in range(5)], cost=4)
    schedule = schedule_budget(tasks, budget=10)
    sequences = [item.to_dict() for item in schedule.deductions]
    assert [item["sequence"] for item in sequences] == [0, 1]
    assert [item["sample"] for item in sequences] == ["s0", "s1"]
    # Replaying the same inputs yields the identical deductions.
    again = schedule_budget(make_feature_tasks([f"s{idx}" for idx in range(5)], 4), 10)
    assert [item.to_dict() for item in again.deductions] == sequences


def test_zero_task_budget_pending_all():
    tasks = make_feature_tasks(["a"], cost=5)
    schedule = schedule_budget(tasks, budget=3)
    assert not schedule.ready
    assert schedule.pending[0].sample == "a"
