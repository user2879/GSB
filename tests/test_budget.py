"""全局预算、单批配额、待办保留与不重复计费。"""
import unittest
from dataclasses import replace

from reward_workbench.config import AppConfig, RewardWeights
from reward_workbench.pipeline import Pipeline
from reward_workbench.samplegen import DeterministicRng, generate_pairs
from reward_workbench.errors import E_TRAIN_BUDGET_EXHAUSTED


def cfg(**over):
    base = dict(
        tokenizer_version="v1", normalize=True, feature_dim=64, hash_buckets=32,
        weights=RewardWeights(0.25, 1.0, 0.75), boundary_threshold=0.5,
        decimal_places=6, quant_mode="half_even", batch_size=2, learning_rate=0.01,
        global_pair_budget=3, per_batch_quota=2, max_workers=2,
        snapshot_commit_pairs=2, time_partition_ms=1000,
    )
    base.update(over)
    return AppConfig(**base)


class BudgetTest(unittest.TestCase):
    def setUp(self):
        rng = DeterministicRng()
        recs = generate_pairs(rng, n_per_batch=3, batches=("b0", "b1"))
        from reward_workbench.pairs import PreferencePair
        self.pairs = {r["pair_id"]: PreferencePair(**r) for r in recs}

    def test_quota_and_global_budget(self):
        pipe = Pipeline(cfg(), dict(self.pairs))
        pipe.run_all_batches()
        ledger = pipe.report.budget_ledger
        self.assertLessEqual(ledger["consumed"], 3)
        self.assertTrue(pipe.report.failures.has(E_TRAIN_BUDGET_EXHAUSTED))
        self.assertTrue(len(ledger["todo"]) >= 1)

    def test_no_double_charge_on_rerun(self):
        c = cfg()
        pipe = Pipeline(c, dict(self.pairs))
        pipe.run_all_batches()
        consumed_first = pipe.report.budget_ledger["consumed"]
        # 再次运行（同状态），已计费对不重复扣减
        pipe.run_all_batches()
        self.assertEqual(pipe.report.budget_ledger["consumed"], consumed_first)

    def test_budget_increase_processes_todo(self):
        c = cfg()
        pipe = Pipeline(c, dict(self.pairs))
        pipe.run_all_batches()
        blocked = list(pipe.report.budget_ledger["todo"])
        self.assertTrue(blocked)
        pipe.cfg = replace(pipe.cfg, global_pair_budget=100, per_batch_quota=100)
        pipe.budget.global_budget = 100
        pipe.run_all_batches()
        self.assertEqual(pipe.report.budget_ledger["todo"], [])


if __name__ == "__main__":
    unittest.main()
