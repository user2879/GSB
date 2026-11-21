import unittest

from hashlot.aggregation import LocalAggregator, merge_aggregators
from hashlot.digest import content_digest

AGG = (
    ("count", "*", "cnt"),
    ("sum", "amount", "amount_sum"),
    ("avg", "latency", "latency_avg"),
)
KEY_A = ((0x11, (1).to_bytes(8, "little", signed=True)),)
KEY_B = ((0x11, (2).to_bytes(8, "little", signed=True)),)


class AggregationTest(unittest.TestCase):
    def test_count_sum_avg_fixed_point(self):
        agg = LocalAggregator("c1", AGG, 1000)
        agg.add_row(KEY_A, (None, 10.5, 2.0))
        agg.add_row(KEY_A, (None, None, 4.0))
        merged = merge_aggregators([agg], AGG, 1000)
        row = merged["rows"][0]
        self.assertEqual(row["aggregates"]["cnt"]["fixed_point"], 2)
        self.assertEqual(row["aggregates"]["amount_sum"]["fixed_point"], 10500)
        self.assertEqual(row["aggregates"]["latency_avg"]["fixed_point"], 3000)
        self.assertEqual(row["aggregates"]["latency_avg"]["count_inputs"], 2)

    def test_merge_order_independent(self):
        def build(prefix, swap):
            a1 = LocalAggregator(f"{prefix}1", AGG, 1000)
            a2 = LocalAggregator(f"{prefix}2", AGG, 1000)
            a1.add_row(KEY_A, (None, 1.1, 1.0))
            a2.add_row(KEY_A, (None, 2.2, 3.0))
            a2.add_row(KEY_B, (None, 5.5, 7.0))
            order = [a1, a2] if not swap else [a2, a1]
            return merge_aggregators(order, AGG, 1000)
        self.assertEqual(content_digest(build("c", False)),
                         content_digest(build("c", True)))

    def test_nulls_excluded_from_sum_avg(self):
        agg = LocalAggregator("c1", AGG, 1000)
        agg.add_row(KEY_A, (None, None, None))
        merged = merge_aggregators([agg], AGG, 1000)
        row = merged["rows"][0]
        self.assertEqual(row["aggregates"]["cnt"]["fixed_point"], 1)
        self.assertEqual(row["aggregates"]["amount_sum"]["fixed_point"], 0)
        self.assertIsNone(row["aggregates"]["latency_avg"]["fixed_point"])

    def test_saturated_sum(self):
        from hashlot.fixedpoint import INT64_MAX
        agg = LocalAggregator("c1", AGG, 1)
        huge = float(INT64_MAX)
        agg.add_row(KEY_A, (None, huge, 1.0))
        agg.add_row(KEY_A, (None, huge, 1.0))
        merged = merge_aggregators([agg], AGG, 1)
        self.assertEqual(
            merged["rows"][0]["aggregates"]["amount_sum"]["fixed_point"], INT64_MAX)


if __name__ == "__main__":
    unittest.main()
