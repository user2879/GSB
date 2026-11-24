from __future__ import annotations

import random
import unittest
from types import MappingProxyType

from config_arbiter.arbitration import (
    arbitrate_key,
    summarize_view,
)
from config_arbiter.spec import KeyRule, ViewSpec

from .util import candidate


class ArbitrationTests(unittest.TestCase):
    def view(self, mode: str = "latest", tolerance: int = 0) -> ViewSpec:
        return ViewSpec(
            id="view-test",
            environments=("a", "b", "c"),
            keys=MappingProxyType({"k": KeyRule(mode=mode, tolerance=tolerance)}),
        )

    def test_latest_selects_highest_version_then_env_id(self) -> None:
        view = self.view("latest")
        result = arbitrate_key(
            view=view,
            key="k",
            rule=view.keys["k"],
            candidates=[
                candidate("c", 30, version=3),
                candidate("a", 10, version=2),
                candidate("b", 20, version=2),
            ],
            scale_exponent=3,
        )
        self.assertEqual(result.summary_units, 30)
        self.assertEqual(result.adopted, ("c",))
        self.assertEqual(result.rejected, ("a", "b"))

    def test_weighted_majority_uses_quantized_weights(self) -> None:
        view = self.view("weighted_majority")
        result = arbitrate_key(
            view=view,
            key="k",
            rule=view.keys["k"],
            candidates=[
                candidate("a", 100, weight=1000),
                candidate("b", 102, weight=1000),
                candidate("c", 200, weight=3000),
            ],
            scale_exponent=3,
        )
        self.assertEqual(result.summary_units, 200)
        self.assertEqual(result.adopted, ("c",))

    def test_weighted_majority_tolerance_cluster_weighted_mean(self) -> None:
        view = self.view("weighted_majority", tolerance=150)
        result = arbitrate_key(
            view=view,
            key="k",
            rule=view.keys["k"],
            candidates=[
                candidate("a", 100, weight=1000),
                candidate("b", 102, weight=1000),
                candidate("c", 200, weight=3000),
            ],
            scale_exponent=3,
        )
        self.assertEqual(result.summary_units, 160)
        self.assertEqual(result.adopted, ("a", "b", "c"))

    def test_interval_midpoint_finds_maximum_weight_intersection(self) -> None:
        view = self.view("interval_midpoint", tolerance=6)
        result = arbitrate_key(
            view=view,
            key="k",
            rule=view.keys["k"],
            candidates=[
                candidate("a", 0),
                candidate("b", 10),
                candidate("c", 100),
            ],
            scale_exponent=3,
        )
        self.assertEqual(result.summary_units, 5)
        self.assertEqual(result.adopted, ("a", "b"))
        self.assertEqual(result.rejected, ("c",))

    def test_result_bytes_are_independent_of_report_order(self) -> None:
        view = ViewSpec(
            id="order-view",
            environments=("a", "b", "c"),
            keys=MappingProxyType(
                {
                    "latest_k": KeyRule("latest", 0),
                    "weighted_k": KeyRule("weighted_majority", 4),
                    "interval_k": KeyRule("interval_midpoint", 8),
                }
            ),
        )
        candidates = [
            candidate("c", 1004, version=1, weight=1000, value_error="0.4"),
            candidate("a", 1000, version=2, weight=1250, value_error="0.7"),
            candidate("b", 1002, version=1, weight=1000, value_error="0.1"),
        ]
        baseline: dict[str, tuple[str, str]] = {}
        for key in sorted(view.keys):
            result = arbitrate_key(
                view=view,
                key=key,
                rule=view.keys[key],
                candidates=candidates,
                scale_exponent=3,
            )
            baseline[key] = (result.result_digest, result.result_json)
        rng = random.Random(91)
        for _ in range(20):
            shuffled = list(candidates)
            rng.shuffle(shuffled)
            for key in sorted(view.keys):
                result = arbitrate_key(
                    view=view,
                    key=key,
                    rule=view.keys[key],
                    candidates=shuffled,
                    scale_exponent=3,
                )
                self.assertEqual(result.result_digest, baseline[key][0])
                self.assertEqual(result.result_json, baseline[key][1])

    def test_near_tie_is_stable_and_attributed(self) -> None:
        view = self.view("latest")
        result = arbitrate_key(
            view=view,
            key="k",
            rule=view.keys["k"],
            candidates=[
                candidate("b", 1001, version=1, value_error="0.5"),
                candidate("a", 1000, version=2, value_error="0.5"),
            ],
            scale_exponent=3,
        )
        self.assertEqual(result.adopted, ("a",))
        self.assertTrue(result.error_bound["indistinguishable"])
        self.assertEqual(result.attribution["primary_cause"], "ERROR_INDISTINGUISHABLE")
        self.assertIn("slice_version", result.tie_break)

    def test_error_bound_decomposition_and_view_upper_bound(self) -> None:
        view = self.view("interval_midpoint", tolerance=5)
        result = arbitrate_key(
            view=view,
            key="k",
            rule=view.keys["k"],
            candidates=[
                candidate("a", 1000, weight=1000, value_error="0.5"),
                candidate("b", 1004, weight=1000, value_error="0.2"),
            ],
            scale_exponent=3,
        )
        bound = result.error_bound
        self.assertEqual(
            bound["decision_bound_units"],
            bound["quantization_truncation_units"]
            + bound["arbitration_mode_difference_units"]
            + bound["tolerance_absorption_units"],
        )
        summary = summarize_view("view-test", 1, [result])
        self.assertEqual(
            summary.error_bound["decision_bound_units"],
            summary.error_bound["quantization_truncation_units"]
            + summary.error_bound["arbitration_mode_difference_units"]
            + summary.error_bound["tolerance_absorption_units"],
        )
        shares = [float(value) for value in summary.attribution["cause_shares"].values()]
        self.assertAlmostEqual(sum(shares), 1.0, places=5)

    def test_no_candidates_produces_error_result_instead_of_crashing(self) -> None:
        view = self.view("latest")
        result = arbitrate_key(
            view=view,
            key="k",
            rule=view.keys["k"],
            candidates=[],
            scale_exponent=3,
        )
        self.assertEqual(result.status, "ERROR")
        self.assertIsNone(result.summary_units)
        self.assertEqual(result.attribution["primary_cause"], "SERVICE_UNAVAILABLE")

    def test_view_config_digest_ignores_participant_declaration_order(self) -> None:
        first = ViewSpec(
            id="same-view",
            environments=("a", "b", "c"),
            keys=MappingProxyType({"k": KeyRule("latest", 0)}),
        )
        second = ViewSpec(
            id="same-view",
            environments=("c", "a", "b"),
            keys=MappingProxyType({"k": KeyRule("latest", 0)}),
        )
        self.assertEqual(first.config_digest(), second.config_digest())


if __name__ == "__main__":
    unittest.main()
