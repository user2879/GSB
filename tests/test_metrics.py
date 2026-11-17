from __future__ import annotations

import unittest

from exp_tracker.codec import parse_record
from exp_tracker.errors import MetricError
from exp_tracker.metrics import (
    aggregate_metrics,
    generate_step_metrics,
    metric_digest,
)
from exp_tracker.models import Aggregation

from .helpers import current_record
from exp_tracker.records import build_v1_record, parameters


class MetricTests(unittest.TestCase):
    def test_repeated_generation_is_bit_identical(self) -> None:
        params = parameters(
            seed=2026,
            model_dim=64,
            learning_rate=0.017,
            steps=40,
            hyperparameters={"dropout": 0.2},
        )
        first = generate_step_metrics(params)
        second = generate_step_metrics(params)
        self.assertEqual(first, second)
        self.assertEqual(
            [point.loss.hex() for point in first],
            [point.loss.hex() for point in second],
        )
        first_final = aggregate_metrics(first, "last")
        second_final = aggregate_metrics(second, "last")
        self.assertEqual(
            metric_digest(first, Aggregation("last"), first_final),
            metric_digest(second, Aggregation("last"), second_final),
        )

    def test_all_aggregations_are_consistent(self) -> None:
        params = parameters(
            seed=5,
            model_dim=16,
            learning_rate=0.02,
            steps=10,
            hyperparameters={"decay": 0.5},
        )
        points = generate_step_metrics(params)
        last = aggregate_metrics(points, "last")
        mean_tail = aggregate_metrics(points, "mean_last_5")
        mean = aggregate_metrics(points, "mean")
        self.assertEqual(last["loss"], points[-1].loss)
        self.assertAlmostEqual(mean_tail["accuracy"], sum(p.accuracy for p in points[-5:]) / 5)
        self.assertAlmostEqual(mean["loss"], sum(p.loss for p in points) / 10)

    def test_step_mismatch_reports_first_offset_and_field(self) -> None:
        record = current_record()
        record["step_metrics"][2]["loss"] += 0.000123
        with self.assertRaises(MetricError) as caught:
            parse_record(record)
        self.assertEqual(caught.exception.code, "E_METRIC")
        self.assertEqual(caught.exception.details["step"], 3)
        self.assertEqual(caught.exception.details["field"], "loss")
        self.assertEqual(caught.exception.details["run_id"], "baseline")

    def test_final_metric_mismatch_reports_aggregation_field(self) -> None:
        record = current_record()
        record["final_metrics"]["accuracy"] += 0.01
        with self.assertRaises(MetricError) as caught:
            parse_record(record)
        self.assertEqual(caught.exception.details["field"], "final_metrics.accuracy")
        self.assertEqual(caught.exception.details["step"], len(record["step_metrics"]))

    def test_v1_final_metric_mismatch_is_metric_error(self) -> None:
        record = build_v1_record(current_record())
        record["final_metrics"]["loss"] += 0.01
        with self.assertRaises(MetricError) as caught:
            parse_record(record)
        self.assertEqual(caught.exception.details["field"], "final_metrics.loss")


if __name__ == "__main__":
    unittest.main()
