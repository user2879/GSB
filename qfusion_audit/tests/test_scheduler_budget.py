import unittest

from qfusion.errors import E_RECOMPUTE_BUDGET_PENDING
from qfusion.scheduler import RecomputeScheduler, WorkUnit


def make_units(n_req=3, n_item=4):
    return [WorkUnit(f"r{r}", f"i{i}") for r in range(n_req)
            for i in range(n_item)]


class SchedulerBudgetTest(unittest.TestCase):
    def test_pending_retained_and_priority(self):
        units = make_units()
        sched = RecomputeScheduler(units, concurrency=2,
                                   worker=lambda u: (u.request_id, u.item_id))
        first = sched.run_batch(budget=5)
        self.assertEqual(first.consumed_points, 5)
        self.assertEqual(len(first.pending), len(units) - 5)
        self.assertTrue(all(
            f["code"] == E_RECOMPUTE_BUDGET_PENDING
            for f in first.pending_failures))
        second = sched.run_batch(budget=100)
        self.assertEqual(second.consumed_points, len(units) - 5)
        self.assertEqual(second.pending, [])
        self.assertTrue(sched.drained)
        self.assertEqual(len(sched.results), len(units))

    def test_small_batches_equal_full_budget_results(self):
        full = RecomputeScheduler(make_units(), 4,
                                  worker=lambda u: (u.request_id, u.item_id))
        full.run_batch(budget=1000)

        small = RecomputeScheduler(make_units(), 4,
                                   worker=lambda u: (u.request_id, u.item_id))
        ledgers = small.run_to_drain(batch_budget=3)

        self.assertEqual(small.results, full.results)
        self.assertEqual(sum(b.consumed_points for b in ledgers),
                         len(make_units()))
        processed = [k for b in ledgers for k in b.processed]
        self.assertEqual(processed, sorted(processed))

    def test_concurrency_never_exceeds_config(self):
        sched = RecomputeScheduler(make_units(2, 10), concurrency=3,
                                   worker=lambda u: u.key)
        ledger = sched.run_batch(budget=20)
        self.assertLessEqual(ledger.concurrency_used, 3)

    def test_batch_ledger_records_sources(self):
        sched = RecomputeScheduler(make_units(1, 2), 2,
                                   worker=lambda u: u.key)
        ledger = sched.run_batch(budget=2)
        self.assertTrue(all(v == "fresh" for v in ledger.sources.values()))


if __name__ == "__main__":
    unittest.main()
