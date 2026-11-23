"""内容指纹精确失效：未受影响对必须复用、不重算。"""
import tempfile
import unittest
from dataclasses import replace

from reward_workbench.commands import Command
from reward_workbench.pipeline import Pipeline
from reward_workbench.samplegen import write_workspace
from reward_workbench.cli import _load_all
from reward_workbench.errors import FailureLog


class InvalidationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rwb-inv-")
        self.paths = write_workspace(self.tmp)
        fl = FailureLog()
        self.cfg, self.pairs = _load_all(
            self.paths["pairs"], self.paths["config"], fl)
        # 放大预算/配额，确保全部对都进入打分缓存，便于验证精确失效
        self.cfg = replace(self.cfg, global_pair_budget=1000, per_batch_quota=1000)

    def test_single_pair_replace_only_invalidates_that_pair(self):
        pipe = Pipeline(self.cfg, dict(self.pairs))
        pipe.run_all_batches()
        target = "pair-0003"
        keep_key = pipe.cache.feature.pair_index["pair-0001"]

        record = dict(self.pairs[target].to_dict())
        record["chosen"] = "accurate concise answer output response"
        cmd = Command(seq=1, op="replace_pair", payload={"record": record})
        pipe.apply_and_run(cmd)

        event = pipe.report.invalidation_events[-1]
        self.assertEqual(event["changed_pairs"], [target])
        # 未受影响对的 feature key 原样保留
        self.assertEqual(pipe.cache.feature.pair_index["pair-0001"], keep_key)
        # 受影响对被重算（新 key）
        self.assertIn(target, pipe.cache.feature.pair_index)

    def test_weight_change_invalidates_all_scores(self):
        pipe = Pipeline(self.cfg, dict(self.pairs))
        pipe.run_all_batches()
        before = dict(pipe.cache.score.pair_index)
        cmd = Command(seq=2, op="set_weight",
                      payload={"name": "chosen", "value": 2.0})
        pipe.apply_and_run(cmd)
        # 权重变化后所有对的 score key 都改变，feature key 不变
        for pid in before:
            self.assertNotEqual(pipe.cache.score.pair_index[pid], before[pid])
        self.assertEqual(
            pipe.cache.feature.pair_index.keys(), before.keys())


if __name__ == "__main__":
    unittest.main()
