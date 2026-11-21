import copy

from workbench import codes, scheduler
from workbench.cache import StageCache
from workbench.pipeline import Pipeline


def test_budget_consumed_in_turn_id_order():
    in_batch, pending, deductions = scheduler.plan_budget(
        ["T003", "T001", "T002", "T004"], budget=2 * scheduler.POINTS_PER_TURN)
    assert in_batch == ["T001", "T002"]
    assert pending == ["T003", "T004"]
    assert [d["turn_id"] for d in deductions] == ["T001", "T002"]
    assert [d["spent"] for d in deductions] == [3, 6]


def test_budget_exhaustion_marks_pending(dataset):
    cfg = copy.deepcopy(dataset["config"])
    cfg["fusion_point_budget"] = scheduler.POINTS_PER_TURN  # room for one turn
    report = Pipeline(dataset["turns"], dataset["acoustics"], dataset["tools"],
                      dataset["lexicon"], cfg, cache=StageCache()).run()
    pending = report["budget"]["pending"]
    assert len(pending) == len(dataset["turns"]) - 1
    assert all(p["code"] == codes.E_FUSE_BUDGET_PENDING for p in pending)
    pending_turns = [t for t in report["turns"]
                     if codes.E_FUSE_BUDGET_PENDING in t["codes"]]
    assert len(pending_turns) == len(pending)


def test_concurrency_matches_serial(dataset):
    cfg = copy.deepcopy(dataset["config"])
    cfg["concurrency"] = 1
    serial = Pipeline(dataset["turns"], dataset["acoustics"], dataset["tools"],
                      dataset["lexicon"], cfg, cache=StageCache()).run()
    parallel = Pipeline(dataset["turns"], dataset["acoustics"], dataset["tools"],
                        dataset["lexicon"], dataset["config"], cache=StageCache()).run()
    serial.pop("config_fingerprint")
    parallel.pop("config_fingerprint")
    assert serial == parallel


def test_max_workers_never_exceeded(dataset):
    # run_tasks must never use more workers than configured
    tasks = [((f"T{i:03d}", "s"), (lambda: 1)) for i in range(10)]
    results = scheduler.run_tasks(tasks, max_workers=2)
    assert results == [1] * 10
