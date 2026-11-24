from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from receipt_diag.errors import DiagnosticError
from receipt_diag.planning import plan_batches
from receipt_diag.specs import load_specs

from tests.helpers import write_sample


class SpecValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def sample(self, mutate=None) -> Path:
        return write_sample(self.root / "sample", mutate=mutate)

    def assert_spec_error(self, directory: Path, path_fragment: str) -> None:
        with self.assertRaises(DiagnosticError) as caught:
            load_specs(directory)
        self.assertEqual(caught.exception.code, "E_SPEC")
        self.assertIn(path_fragment, caught.exception.path)

    def test_valid_sample_and_plan(self) -> None:
        spec = load_specs(self.sample())
        self.assertEqual(len(spec.requests), 10)
        self.assertEqual(len(spec.workers), 2)
        plan = plan_batches(spec)
        self.assertEqual(len(plan.batches), 5)
        self.assertEqual(
            sorted(
                request_id
                for batch in plan.batches
                for request_id in batch.request_ids
            ),
            sorted(spec.request_by_id),
        )

    def test_duplicate_request_id(self) -> None:
        def mutate(req, workers, rules):
            req["requests"][1]["request_id"] = req["requests"][0]["request_id"]

        self.assert_spec_error(self.sample(mutate), "request_id")

    def test_negative_arrival(self) -> None:
        def mutate(req, workers, rules):
            req["requests"][0]["arrival_seq"] = -1

        self.assert_spec_error(self.sample(mutate), "arrival_seq")

    def test_duplicate_arrival(self) -> None:
        def mutate(req, workers, rules):
            req["requests"][1]["arrival_seq"] = req["requests"][0]["arrival_seq"]

        self.assert_spec_error(self.sample(mutate), "arrival_seq")

    def test_dimension_mismatch(self) -> None:
        def mutate(req, workers, rules):
            req["requests"][0]["input"] = [1, 2, 3]

        self.assert_spec_error(self.sample(mutate), ".input")

    def test_hidden_dimension_positive(self) -> None:
        def mutate(req, workers, rules):
            workers["workers"][0]["hidden_dim"] = 0

        self.assert_spec_error(self.sample(mutate), "hidden_dim")

    def test_seed_requires_integer(self) -> None:
        def mutate(req, workers, rules):
            workers["workers"][0]["seed"] = True

        self.assert_spec_error(self.sample(mutate), "seed")

    def test_host_literal(self) -> None:
        def mutate(req, workers, rules):
            workers["workers"][0]["host"] = "localhost"

        self.assert_spec_error(self.sample(mutate), "host")

    def test_duplicate_ports(self) -> None:
        def mutate(req, workers, rules):
            workers["workers"][1]["port"] = workers["workers"][0]["port"]

        self.assert_spec_error(self.sample(mutate), "port")

    def test_unassigned_request(self) -> None:
        def mutate(req, workers, rules):
            rules["routes"][0]["request_ids"].pop()

        self.assert_spec_error(self.sample(mutate), "request_id")

    def test_plan_independent_of_request_and_key_order(self) -> None:
        first = write_sample(self.root / "first")

        def reorder(req, workers, rules):
            req["requests"].reverse()
            rules["routes"].reverse()
            for route in rules["routes"]:
                route["request_ids"].reverse()
            # Force non-sorted key insertion order in each object.
            for document in (req, workers, rules):
                items = list(document.items())
                document.clear()
                document.update(reversed(items))

        second = write_sample(self.root / "second", mutate=reorder, sort_keys=False)
        plan_a = plan_batches(load_specs(first))
        plan_b = plan_batches(load_specs(second))
        self.assertEqual(plan_a.fingerprint, plan_b.fingerprint)
        self.assertEqual(plan_a.to_json(), plan_b.to_json())


if __name__ == "__main__":
    unittest.main()
