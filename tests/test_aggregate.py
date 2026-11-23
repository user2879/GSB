import unittest

from diagconsole import aggregate as agg
from diagconsole.errors import canonical_json


class AggregateTests(unittest.TestCase):
    def test_deterministic_across_calls(self):
        args = (20240101, 64, {"rate_limit": 100, "burst": 50},
                "rolling", 1, [1])
        first = agg.aggregate(*args)
        second = agg.aggregate(*args)
        self.assertEqual(canonical_json(first), canonical_json(second))

    def test_three_calibers_distinct_and_integer(self):
        summaries = [agg.aggregate(7, 32, {"k": 9}, "full", caliber, [2])
                     for caliber in (1, 2, 3)]
        for summary in summaries:
            self.assertTrue(all(isinstance(v, int) for v in summary["groups"]))
            self.assertEqual(summary["total"], sum(summary["groups"]))
        self.assertNotEqual(summaries[0]["groups"], summaries[1]["groups"])
        self.assertNotEqual(summaries[1]["groups"], summaries[2]["groups"])

    def test_derives_change_content(self):
        base = agg.aggregate(7, 32, {"k": 9}, "canary", 1, [])
        derived = agg.aggregate(7, 32, {"k": 9}, "canary", 1, [5])
        self.assertNotEqual(canonical_json(base), canonical_json(derived))

    def test_strategy_factor(self):
        canary = agg.aggregate(7, 32, {}, "canary", 1, [])
        full = agg.aggregate(7, 32, {}, "full", 1, [])
        self.assertEqual([v * 3 for v in canary["groups"]], full["groups"])

    def test_effective_values_fold_from_root(self):
        from diagconsole.model import Revision
        chain = [Revision(1, None, {"a": 1, "b": 2}, "canary", ()),
                 Revision(2, 1, {"a": 5}, "rolling", (1,))]
        merged = agg.effective_values(chain, ("a", "b", "c"))
        self.assertEqual(merged, {"a": 5, "b": 2, "c": 0})


if __name__ == "__main__":
    unittest.main()
