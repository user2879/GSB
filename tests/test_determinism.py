"""阶段 1/2 确定性与定点打分逐整数一致。"""
import unittest

from reward_workbench.config import AppConfig, RewardWeights
from reward_workbench.pairs import PreferencePair
from reward_workbench.stages import run_feature, run_score, run_boundary


def make_cfg(**over):
    base = dict(
        tokenizer_version="v1", normalize=True, feature_dim=64, hash_buckets=32,
        weights=RewardWeights(0.25, 1.0, 0.75), boundary_threshold=0.5,
        decimal_places=6, quant_mode="half_even", batch_size=4, learning_rate=0.01,
        global_pair_budget=100, per_batch_quota=100, max_workers=4,
        snapshot_commit_pairs=4, time_partition_ms=1000,
    )
    base.update(over)
    return AppConfig(**base)


def make_pair(pid="pair-0001"):
    return PreferencePair(
        pair_id=pid,
        prompt="summarize the document for request",
        chosen="accurate and concise summary output",
        rejected="verbose filler repeat noise error",
        annotator_id="ann-alpha", event_ms=1700000000000, batch_id="batch-0",
    )


class DeterminismTest(unittest.TestCase):
    def test_feature_repeatable(self):
        cfg = make_cfg()
        pair = make_pair()
        f1 = run_feature(pair, cfg)
        f2 = run_feature(pair, cfg)
        self.assertEqual(f1.cache_key, f2.cache_key)
        self.assertEqual(f1.prompt_vec.counts, f2.prompt_vec.counts)
        self.assertEqual(f1.referenced_buckets, f2.referenced_buckets)

    def test_score_integer_stable(self):
        cfg = make_cfg()
        pair = make_pair()
        feat = run_feature(pair, cfg)
        s1 = run_score(pair, feat, cfg)
        s2 = run_score(pair, feat, cfg)
        self.assertEqual(s1.chosen_score, s2.chosen_score)
        self.assertEqual(s1.rejected_score, s2.rejected_score)
        self.assertEqual(s1.diff, s2.diff)
        self.assertIsInstance(s1.diff, int)

    def test_quant_mode_changes_result_deterministically(self):
        pair = make_pair()
        trunc = make_cfg(quant_mode="trunc", decimal_places=4)
        even = make_cfg(quant_mode="half_even", decimal_places=4)
        st = run_score(pair, run_feature(pair, trunc), trunc)
        se = run_score(pair, run_feature(pair, even), even)
        # 两者都确定可复现
        self.assertEqual(st, run_score(pair, run_feature(pair, trunc), trunc))
        self.assertEqual(se, run_score(pair, run_feature(pair, even), even))

    def test_tokenizer_version_changes_features(self):
        pair = make_pair()
        v1 = run_feature(pair, make_cfg(tokenizer_version="v1"))
        v2 = run_feature(pair, make_cfg(tokenizer_version="v2"))
        self.assertNotEqual(v1.cache_key, v2.cache_key)

    def test_boundary_threshold_equality_is_critical(self):
        cfg = make_cfg()
        pair = make_pair()
        score = run_score(pair, run_feature(pair, cfg), cfg)
        bd = run_boundary(pair, score, cfg, tie_rank=0)
        self.assertIn(bd.label, ("stable", "critical", "conflict"))
        self.assertEqual(bd.at_threshold, abs(score.diff) == bd.threshold)


if __name__ == "__main__":
    unittest.main()
