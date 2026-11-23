import os
import random
import unittest

from repairbench.fixedpoint import FixedPoint
from repairbench.spec import load_spec
from repairbench.verdict import (
    ERROR_INDISTINGUISHABLE,
    EXACT_SEPARABLE,
    INSUFFICIENT_EVIDENCE,
    Candidate,
    UrgencyRanker,
)

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "samples", "groups.json")


class TestVerdict(unittest.TestCase):
    def setUp(self):
        self.group = load_spec(SAMPLE)[0]
        self.fp = FixedPoint(self.group.params.scale_exponent)
        self.ranker = UrgencyRanker(self.fp)
        self.cands = {
            b.block_id: Candidate(b, self.group.node(b.node_id))
            for b in self.group.blocks
        }

    def ranked(self, block_id):
        return self.ranker.evaluate(self.cands[block_id])

    def test_exact_separable(self):
        # a-b1: 800‰*w5 => 4000；a-b6: 350‰*w1 => 350，差远超误差界
        v = self.ranker.compare(self.ranked("a-b1"), self.ranked("a-b6"))
        self.assertEqual(v["verdict"], EXACT_SEPARABLE)
        self.assertEqual(v["winner"], "a-b1")
        self.assertFalse(v["error_affected"])

    def test_error_indistinguishable_with_stable_tiebreak(self):
        # a-b3: 601‰*w3 => 1803；a-b4: 600‰*w3 => 1800；差 3 <= 合成界 8
        v = self.ranker.compare(self.ranked("a-b3"), self.ranked("a-b4"))
        self.assertEqual(v["verdict"], ERROR_INDISTINGUISHABLE)
        self.assertTrue(v["error_affected"])
        self.assertEqual(v["winner"], "a-b3")  # 块标识序决胜
        self.assertEqual(v["tiebreak"], "block_id,node_id")

    def test_insufficient_evidence(self):
        # a-b5 未声明失配程度
        v = self.ranker.compare(self.ranked("a-b5"), self.ranked("a-b1"))
        self.assertEqual(v["verdict"], INSUFFICIENT_EVIDENCE)

    def test_cost_comparison(self):
        v = self.ranker.compare_cost(self.ranked("a-b1"), self.ranked("a-b2"))
        self.assertEqual(v["verdict"], EXACT_SEPARABLE)
        self.assertEqual(v["cheaper"], "a-b2")

    def test_order_independent_of_input_order(self):
        cands = list(self.cands.values())
        base = self.ranker.adjudicate_all(cands)
        for seed in range(5):
            shuffled = list(cands)
            random.Random(seed).shuffle(shuffled)
            result = self.ranker.adjudicate_all(shuffled)
            self.assertEqual(base["order"], result["order"])
            self.assertEqual(base["classes"], result["classes"])

    def test_three_classes_present(self):
        result = self.ranker.adjudicate_all(list(self.cands.values()))
        self.assertGreater(result["classes"][EXACT_SEPARABLE], 0)
        self.assertGreater(result["classes"][ERROR_INDISTINGUISHABLE], 0)
        self.assertGreater(result["classes"][INSUFFICIENT_EVIDENCE], 0)


if __name__ == "__main__":
    unittest.main()
