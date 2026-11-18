"""跨模态对齐打分、稳定归一化与检索排序测试。"""

import unittest

import numpy as np

from xmodal.alignment import (alignment_score, search,
                              _stable_softmax)
from xmodal.features import FeatureRecord


def record(sample, seed, dim=32):
    rng = np.random.RandomState(seed)
    return FeatureRecord(
        sample=sample,
        image_blocks=rng.rand(8, 7).astype(np.float32),
        text_vector=rng.rand(dim).astype(np.float32),
        align_weights={"mean": 0.35, "gradient": 0.25,
                       "orientation": 0.25, "text": 0.15},
        source_version=2)


class AlignmentTest(unittest.TestCase):
    def test_stable_softmax_sums_to_one(self):
        weights = _stable_softmax(np.array([0.9, -0.2, 0.4]))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=9)
        # 减去最大值不改变结果，且对大数值稳定
        big = _stable_softmax(np.array([1000.0, 1000.0, 1000.0]))
        self.assertTrue(np.allclose(big, 1 / 3))

    def test_score_bounded_and_deterministic(self):
        rec = record("a", 3)
        score_a, detail_a = alignment_score(rec, 4)
        score_b, detail_b = alignment_score(rec, 4)
        self.assertEqual(score_a, score_b)
        self.assertEqual(detail_a, detail_b)
        self.assertGreaterEqual(score_a, -1.0)
        self.assertLessEqual(score_a, 1.0)
        self.assertEqual(set(detail_a), {"mean", "gradient",
                                         "orientation", "text"})

    def test_zero_vectors_do_not_crash(self):
        rec = FeatureRecord(
            sample="z", image_blocks=np.zeros((4, 7), np.float32),
            text_vector=np.zeros(16, np.float32),
            align_weights={"mean": 1.0, "gradient": 0.0,
                           "orientation": 0.0, "text": 0.0},
            source_version=2)
        score, _ = alignment_score(rec, 4)
        self.assertEqual(score, 0.0)


class SearchTest(unittest.TestCase):
    def setUp(self):
        self.scores = {"b": 0.4, "a": 0.4, "c": 0.9, "d": 0.1}

    def test_ordering_and_tie_break(self):
        result = search(self.scores, top_k=3, threshold=0.05)
        names = [c["sample"] for c in result["candidates"]]
        self.assertEqual(names, ["c", "a", "b"])
        self.assertFalse(result["low_score"])
        self.assertIsNone(result["code"])

    def test_low_score_marks_code_keeps_audit_list(self):
        result = search(self.scores, top_k=2, threshold=0.95)
        self.assertTrue(result["low_score"])
        self.assertEqual(result["code"], "E_RETRIEVAL_LOW_SCORE")
        self.assertEqual(len(result["all_candidates"]), 4)

    def test_boundary_equal_threshold_passes(self):
        # 规则：查询目标得分 < 阈值才标记；等于阈值不算低分
        result = search({"x": 0.5}, top_k=1, threshold=0.5,
                        query_sample="x")
        self.assertFalse(result["low_score"])
        result_below = search({"x": 0.499999999}, top_k=1,
                              threshold=0.5, query_sample="x")
        self.assertTrue(result_below["low_score"])

    def test_query_target_score_distinct_from_global_top(self):
        # 全库最高分高，但查询自身图像得分低于阈值时仍应标记低分，
        # 同时保留全库候选清单供审计。
        scores = {"q": 0.2, "hi": 0.9}
        result = search(scores, top_k=2, threshold=0.5,
                        query_sample="q")
        self.assertTrue(result["low_score"])
        self.assertEqual(result["code"], "E_RETRIEVAL_LOW_SCORE")
        self.assertEqual(result["top_score"], 0.2)
        self.assertEqual([c["sample"] for c in result["all_candidates"]],
                         ["hi", "q"])


if __name__ == "__main__":
    unittest.main()
