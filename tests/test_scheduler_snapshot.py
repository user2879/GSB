import tempfile
import unittest

from regbench.config import load_config
from regbench.data import prepare_data
from regbench.errors import (
    E_CONFIG_MISMATCH,
    E_TASK_QUOTA_EXCEEDED,
    ConfigMismatchError,
    FailureLog,
)
from regbench import grid
from regbench.runner import StopController, run_training
from regbench.session import InteractiveSession
from regbench.snapshot import SnapshotStore


class SchedulerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, _ = load_config("samples/experiment.yaml")
        cls.data = prepare_data(cls.config)
        cls.tasks = grid.expand_grid(cls.config, cls.data)

    def test_grid_expansion_and_stable_order(self):
        keys = [t.task_key for t in self.tasks]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(len(keys), 10)
        schemes = {t.scheme for t in self.tasks}
        self.assertEqual(schemes, {"ridge", "lasso", "elasticnet"})

    def test_quota_and_budget_decisions_are_deterministic(self):
        plan = grid.build_schedule_plan(self.config, self.tasks)
        self.assertEqual(plan.consumed_budget, 300)
        self.assertEqual(plan.remaining_budget, 130)
        self.assertEqual(len(plan.quota_tasks), 10)
        self.assertEqual(plan.exhausted_tasks, [])
        # 决策顺序严格等于稳定键顺序。
        self.assertEqual(
            [d["task_key"] for d in plan.decision_log],
            [t.task_key for t in self.tasks],
        )

    def test_budget_exhaustion_marks_never_started(self):
        tight = _with_budget(self.config, 60)
        plan = grid.build_schedule_plan(tight, self.tasks)
        self.assertTrue(plan.exhausted_tasks)
        started = [t for t in plan.ordered_tasks if t.starts]
        self.assertTrue(all(t.planned_epochs > 0 for t in started))
        total = sum(t.planned_epochs for t in plan.ordered_tasks)
        self.assertLessEqual(total, 60)


class SnapshotAndRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.snap = self.temp.name
        self.config, _ = load_config("samples/experiment.yaml")

    def tearDown(self):
        self.temp.cleanup()

    def test_resume_identical_to_full_run(self):
        store = SnapshotStore(f"{self.snap}/full")
        full = run_training(self.config, store, FailureLog())

        resumed_store = SnapshotStore(f"{self.snap}/resumed")
        first_task = full.plan.ordered_tasks[0].task.task_key

        def failpoint(stage, epoch, task_key):
            if stage == "train" and epoch == 3 and task_key == first_task:
                raise KeyboardInterrupt

        try:
            run_training(
                self.config,
                resumed_store,
                FailureLog(),
                StopController(),
                failpoint,
            )
        except KeyboardInterrupt:
            pass
        resumed = run_training(self.config, resumed_store, FailureLog())
        self.assertEqual(
            _comparable(full.results), _comparable(resumed.results)
        )
        self.assertEqual(
            full.plan.decision_log, resumed.plan.decision_log
        )

    def test_config_mismatch_raises_business_code(self):
        other, _ = load_config("samples/long_experiment.yaml")
        store = SnapshotStore(f"{self.snap}/mm")
        run_training(self.config, store, FailureLog())
        with self.assertRaises(ConfigMismatchError) as ctx:
            store.resume_or_raise(other.config_hash)
        self.assertEqual(ctx.exception.code, E_CONFIG_MISMATCH)

    def test_task_exception_isolation(self):
        store = SnapshotStore(f"{self.snap}/iso")
        target = "lasso|pen=0.1"

        def failpoint(stage, epoch, task_key):
            if stage == "train" and epoch == 2 and target in task_key:
                raise RuntimeError("boom")

        outcome = run_training(
            self.config, store, FailureLog(), StopController(), failpoint
        )
        codes = [f.code for f in outcome.failures.records]
        self.assertIn("E_TASK_FAILED", codes)
        self.assertEqual(outcome.summary["grid_completion"]["completed"], 9)
        summarized = {
            row["task_key"]
            for rows in outcome.summary["schemes"].values()
            for row in rows
        }
        self.assertFalse(any(target in key for key in summarized))

    def test_quota_failures_recorded(self):
        store = SnapshotStore(f"{self.snap}/quota")
        outcome = run_training(self.config, store, FailureLog())
        self.assertTrue(
            all(f.code == E_TASK_QUOTA_EXCEEDED for f in outcome.failures.records)
        )


class SessionCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.config, _ = load_config("samples/experiment.yaml")

    def tearDown(self):
        self.temp.cleanup()

    def test_precise_invalidation_and_reuse(self):
        session = InteractiveSession(self.config, f"{self.temp.name}/s")
        session.execute_line("set-workers 8")
        session.execute_line("set-penalty ridge 0.2")
        session.execute_line("set-ratio elasticnet 0.6")
        info = session.run()
        self.assertEqual(info["reused"], [])
        self.assertTrue(info["recomputed"])

        second = InteractiveSession(self.config, f"{self.temp.name}/s")
        second.execute_line("set-workers 8")
        second.execute_line("set-penalty ridge 0.2")
        second.execute_line("set-ratio elasticnet 0.6")
        info2 = second.run()
        self.assertEqual(len(info2["reused"]), len(info2["recomputed"]) + len(info2["reused"]))
        self.assertEqual(info2["recomputed"], [])

    def test_switch_does_not_invalidate(self):
        session = InteractiveSession(self.config, f"{self.temp.name}/s2")
        trace = session.execute_line("switch lasso")
        self.assertEqual(trace.invalidated, [])


def _with_budget(config, budget):
    import copy

    clone = copy.deepcopy(config)
    object.__setattr__(
        clone,
        "scheduler",
        type(config.scheduler)(
            config.scheduler.max_workers,
            budget,
            config.scheduler.per_scheme_epoch_quota,
            config.scheduler.snapshot_interval,
        ),
    )
    return clone


def _comparable(results):
    return {
        key: {
            "loss": value.loss_history,
            "weights": value.weights.tolist(),
            "validation": value.validation,
            "status": value.status,
        }
        for key, value in results.items()
    }


if __name__ == "__main__":
    unittest.main()
