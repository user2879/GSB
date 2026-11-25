from qcalib.errors import E_BUDGET_PENDING, E_QUOTA_EXCEEDED, FailureList
from qcalib.scheduler import Task, batch_quant_budget, run_map, schedule


def test_stable_order_and_triple_budget_constraints():
    tasks = [
        Task("b", 1, 4, {}),
        Task("a", 1, 4, {}),
        Task("a", 0, 4, {}),
        Task("b", 0, 4, {}),
        Task("c", 0, 4, {}),
    ]
    failures = FailureList()
    result = schedule(tasks, global_budget=10,
                      per_layer_quotas={"a": 4}, failures=failures)
    assert [task.sort_key for task in result.completed] == [
        ("a", 0), ("b", 0)]
    assert result.quota_aborted == ["a"]
    assert result.budget_consumed == 8
    codes = [failure["code"] for failure in failures.to_list()]
    assert E_QUOTA_EXCEEDED in codes
    assert E_BUDGET_PENDING in codes
    pending_keys = [task.sort_key for task in result.pending]
    assert ("c", 0) in pending_keys and ("b", 1) in pending_keys
    assert ("a", 1) in pending_keys


def test_run_map_serial_parallel_identical():
    tasks = [Task("l", index, 1, index) for index in range(16)]
    worker = lambda task: (task.sort_key, task.payload * 2)
    serial = run_map(worker, tasks, 1)
    parallel = run_map(worker, tasks, 4)
    assert serial == parallel
    assert [item[0] for item in serial] == sorted(
        [item[0] for item in serial])


def test_batch_quant_budget():
    assert batch_quant_budget(32, 16) == 2
    assert batch_quant_budget(33, 16) == 3
