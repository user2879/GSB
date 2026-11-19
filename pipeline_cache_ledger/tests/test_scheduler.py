import unittest

from pcl.scheduler import BudgetScheduler, TaskResult, TaskSpec


def make_tasks():
    return [
        TaskSpec(key="a", stage_id="s1", order=0),
        TaskSpec(key="b", stage_id="s1", order=1),
        TaskSpec(key="c", stage_id="s2", order=2),
        TaskSpec(key="d", stage_id="s2", order=3),
        TaskSpec(key="e", stage_id="s3", order=4),
    ]


def worker_factory(outcome_map):
    def worker(task, slot, start_tick):
        return TaskResult(
            key=task.key,
            stage_id=task.stage_id,
            outcome=outcome_map.get(task.key, "HIT"),
            reason="ok",
            detail={"slot": slot},
        )

    return worker


class SchedulerBudgetTests(unittest.TestCase):
    def test_dual_budget_never_exceeded(self):
        sched = BudgetScheduler(global_limit=2, stage_quotas={
            "s1": 1, "s2": 1, "s3": 1
        })
        records, _ = sched.run(make_tasks(), worker_factory({}))
        running_by_tick = {}
        stage_by_tick = {}
        for rec in records:
            for tick in range(rec.start_tick, rec.end_tick + 1):
                running_by_tick.setdefault(tick, set()).add(rec.slot)
                stage_by_tick.setdefault((tick, rec.stage_id), set()).add(
                    rec.slot
                )
        self.assertTrue(all(len(v) <= 2 for v in running_by_tick.values()))
        self.assertTrue(
            all(len(v) <= 1 for v in stage_by_tick.values())
        )
        # Queued tasks waited because each stage quota is one.
        self.assertTrue(any(rec.wait_ticks > 0 for rec in records))

    def test_slot_and_tick_records_present(self):
        sched = BudgetScheduler(global_limit=3, stage_quotas={
            "s1": 2, "s2": 2, "s3": 2
        })
        records, _ = sched.run(make_tasks(), worker_factory({}))
        for rec in records:
            self.assertGreaterEqual(rec.slot, 0)
            self.assertLess(rec.slot, 3)
            self.assertLessEqual(rec.start_tick, rec.end_tick)
            self.assertEqual(rec.wait_ticks, rec.start_tick - rec.queued_tick)
        timeline = sched.slot_timeline()
        self.assertTrue(timeline)

    def test_deterministic_outcome_set_under_interleavings(self):
        outcomes = {"a": "HIT", "b": "MISS", "c": "HIT",
                    "d": "MISS", "e": "HIT"}
        configs = [
            (1, {"s1": 1, "s2": 1, "s3": 1}),
            (4, {"s1": 2, "s2": 2, "s3": 2}),
            (2, {"s1": 1, "s2": 2, "s3": 1}),
        ]
        conclusion_sets = []
        for global_limit, quotas in configs:
            sched = BudgetScheduler(global_limit, quotas, jitter_ms=1)
            _, results = sched.run(make_tasks(), worker_factory(outcomes))
            conclusion_sets.append(
                {k: (v.outcome, v.reason) for k, v in results.items()}
            )
        for doc in conclusion_sets[1:]:
            self.assertEqual(doc, conclusion_sets[0])


if __name__ == "__main__":
    unittest.main()
