import tempfile
import unittest
from pathlib import Path

from colbench.errors import DivergenceError
from colbench.ledger import Ledger
from colbench.replay import recompute, verify_ledger
from colbench.sample_data import generate
from colbench.scheduler import Scheduler
from colbench.versioning import BudgetParams
from colbench.workbench import Workbench


class LedgerReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.bench = Workbench(self.tmp)
        self.bench.build_from_specs(generate(self.tmp / "samples"))
        self.query = ("SELECT order_id, region FROM orders "
                      "WHERE amount >= 50 AND priority = 2")

    def _scan(self, memory=8192):
        params = BudgetParams(2, memory, 1)
        version = self.bench.registry.observe_budget(params)
        plan = self.bench.plan(self.query)
        result = Scheduler(params, self.bench.segments).run(
            plan.candidate_blocks)
        return plan, result, version

    def test_ledger_persists_and_replay_matches(self):
        plan, result, version = self._scan()
        db = self.tmp / "ledger.sqlite3"
        with Ledger(db) as ledger:
            written = ledger.append_batch(result, version)
            self.assertEqual(written, len(result.tasks))
            report = verify_ledger(plan, self.bench.segments, ledger,
                                   version)
        self.assertEqual(report["status"], "MATCH")
        self.assertEqual(report["blocks_compared"], len(result.tasks))

    def test_divergence_locates_first_block_column_field(self):
        plan, result, version = self._scan()
        db = self.tmp / "diverge.sqlite3"
        with Ledger(db) as ledger:
            ledger.append_batch(result, version)
            # 直接篡改第一条账本轨迹的结果摘要
            ledger.conn.execute(
                "UPDATE scan_ledger SET result_digest = ? "
                "WHERE task_id = ?",
                ("deadbeef", result.tasks[0].task_id))
            ledger.conn.commit()
            with self.assertRaises(DivergenceError) as ctx:
                verify_ledger(plan, self.bench.segments, ledger, version)
        self.assertEqual(ctx.exception.code, "E_DIVERGE")
        self.assertEqual(ctx.exception.details["field"], "result_digest")
        self.assertEqual(ctx.exception.details["column"],
                         result.tasks[0].column)
        self.assertEqual(ctx.exception.details["block_index"],
                         result.tasks[0].block_index)

    def test_recompute_is_independent_path(self):
        plan, _, _ = self._scan()
        blocks = recompute(plan, self.bench.segments)
        self.assertTrue(all(block.result_digest for block in blocks))
        self.assertEqual([b.segment_id for b in blocks],
                         sorted(b.segment_id for b in blocks))


if __name__ == "__main__":
    unittest.main()
