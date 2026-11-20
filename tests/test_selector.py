from __future__ import annotations

import unittest
from collections import Counter

from meshgov.rules import parse_rules
from meshgov.selector import SelectionPlanner, SmoothWeightedRoundRobin
from tests.helpers import instance, make_requests


class SelectorTests(unittest.TestCase):
    def _rules(self, weights: tuple[int, int]):
        document = {
            "schema_version": 1,
            "limits": {"global_concurrency": 4},
            "services": [
                {
                    "name": "alpha",
                    "instances": [
                        instance("a", 28001, weights[0]),
                        instance("b", 28002, weights[1]),
                        instance("zero", 28003, 0),
                    ],
                }
            ],
            "routes": [
                {
                    "prefix": "/api/alpha",
                    "service": "alpha",
                    "timeout_ms": 100,
                    "max_retries": 2,
                }
            ],
        }
        import json

        return parse_rules(json.dumps(document).encode(), "<test>")

    def test_smooth_weighted_round_robin_is_exact_and_zero_weight_safe(self) -> None:
        rules = self._rules((3, 2))
        selector = SmoothWeightedRoundRobin(rules.service("alpha").instances)
        names = [selector.choose().target.name for _ in range(50)]
        counts = Counter(names)
        self.assertEqual(counts["a"], 30)
        self.assertEqual(counts["b"], 20)
        self.assertNotIn("zero", counts)

    def test_repeated_instances_produce_identical_sequences(self) -> None:
        rules = self._rules((5, 3))
        first = SmoothWeightedRoundRobin(rules.service("alpha").instances)
        second = SmoothWeightedRoundRobin(list(reversed(rules.service("alpha").instances)))
        self.assertEqual(
            [first.choose().target.port for _ in range(80)],
            [second.choose().target.port for _ in range(80)],
        )

    def test_planner_is_attempt_layered_and_deterministic(self) -> None:
        rules = self._rules((3, 2))
        requests = make_requests(rules, 20)
        first = SelectionPlanner(rules, requests)
        second = SelectionPlanner(rules, list(reversed(requests)))
        first_layer = Counter(first.get(request.request_id, 0).target.name for request in requests)
        self.assertEqual(first_layer["a"], 12)
        self.assertEqual(first_layer["b"], 8)
        self.assertEqual(
            {
                key: (value.target.port, value.ordinal)
                for key, value in first._plans.items()
            },
            {
                key: (value.target.port, value.ordinal)
                for key, value in second._plans.items()
            },
        )


if __name__ == "__main__":
    unittest.main()

