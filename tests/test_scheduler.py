import unittest

from graphidx import BudgetError, BudgetScheduler, Task


def make_tasks(n, deps=None):
    deps = deps or {}
    return [
        Task(seq=i, kind="index", target=f"idx:{i}", fn=lambda: None, deps=tuple(deps.get(i, ())))
        for i in range(n)
    ]


class TestBudgetScheduler(unittest.TestCase):
    def test_budget_limits_recompute_and_queues_rest(self):
        report = BudgetScheduler(concurrency=2, budget=3).execute(make_tasks(5))
        self.assertEqual(report["recomputed"], 3)
        self.assertEqual(report["skipped"], 2)
        done = [e for e in report["ledger"] if e["status"] == "done"]
        queued = [e for e in report["ledger"] if e["status"] == "queued"]
        self.assertEqual(len(done), 3)
        self.assertEqual(len(queued), 2)
        for entry in done:
            self.assertIn("start_ms", entry)
            self.assertIn("end_ms", entry)
            self.assertGreaterEqual(entry["end_ms"], entry["start_ms"])
            self.assertEqual(entry["budget"], 1)
        for entry in queued:
            self.assertIn("queued_at_ms", entry)  # 等待开始时刻
            self.assertEqual(entry["budget"], 0)

    def test_zero_budget_raises_e_budget(self):
        with self.assertRaises(BudgetError) as ctx:
            BudgetScheduler(concurrency=2, budget=0).execute(make_tasks(2))
        self.assertEqual(ctx.exception.code, "E_BUDGET")
        self.assertEqual(len(ctx.exception.details), 2)

    def test_sequential_when_concurrency_one(self):
        report = BudgetScheduler(concurrency=1, budget=10).execute(make_tasks(4))
        done = [e for e in report["ledger"] if e["status"] == "done"]
        for prev, nxt in zip(done, done[1:]):
            self.assertLessEqual(prev["end_ms"], nxt["start_ms"])

    def test_dependency_waves_respected(self):
        order = []

        def fn(name):
            return lambda: order.append(name)

        tasks = [
            Task(seq=0, kind="view", target="upstream", fn=fn("upstream")),
            Task(seq=1, kind="view", target="downstream", fn=fn("downstream"), deps=(0,)),
        ]
        BudgetScheduler(concurrency=4, budget=10).execute(tasks)
        self.assertEqual(order, ["upstream", "downstream"])

    def test_queued_tasks_not_executed(self):
        ran = []
        tasks = [
            Task(seq=i, kind="index", target=f"t{i}", fn=lambda i=i: ran.append(i))
            for i in range(4)
        ]
        BudgetScheduler(concurrency=2, budget=1).execute(tasks)
        self.assertEqual(ran, [0])


if __name__ == "__main__":
    unittest.main()
