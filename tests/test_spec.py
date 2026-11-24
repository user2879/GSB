from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lateness_ledger.canonical import canonical_json
from lateness_ledger.errors import SpecError
from lateness_ledger.simulate import generate_events
from lateness_ledger.spec import load_bundle
from tests.helpers import mutate_sample, sample_bundle, write_json


class SpecValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="spec-test-")
        self.directory = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def load_mutated(self, **kwargs):
        sources, rules, budgets = mutate_sample(self.directory, **kwargs)
        return load_bundle(sources, rules, budgets)

    def assert_spec_error(self, callable_, expected_field: str) -> SpecError:
        with self.assertRaises(SpecError) as caught:
            callable_()
        self.assertEqual(caught.exception.code, "E_SPEC")
        details = caught.exception.details
        self.assertIn("field", details)
        self.assertIn(expected_field, str(details["field"]))
        return caught.exception

    def test_valid_samples_and_stable_digest(self) -> None:
        bundle = sample_bundle("source_max")
        self.assertEqual(bundle.session_id, "diag-source-max")
        self.assertEqual(bundle.scale, 3)
        self.assertEqual(
            [rule.rule_id for rule in bundle.topological_rules],
            ["base", "derived", "tail"],
        )
        self.assertEqual(len(bundle.spec_digest), 64)
        self.assertEqual(canonical_json(bundle.public_spec()), canonical_json(bundle.public_spec()))

    def test_duplicate_source_is_rejected_at_first_source(self) -> None:
        def mutate(doc):
            doc["sources"][1]["source_id"] = doc["sources"][0]["source_id"]

        exc = self.assert_spec_error(
            lambda: self.load_mutated(source_mutator=mutate), "source_id"
        )
        self.assertEqual(exc.details["kind"], "source")
        self.assertEqual(exc.details["index"], 1)

    def test_out_of_order_ppm_upper_bound(self) -> None:
        def mutate(doc):
            doc["sources"][0]["out_of_order_ppm"] = 10001

        self.assert_spec_error(
            lambda: self.load_mutated(source_mutator=mutate), "out_of_order_ppm"
        )
        self.assertEqual(self.load_mutated(source_mutator=lambda doc: None).scale, 3)

    def test_seed_and_event_count_are_non_negative_integers(self) -> None:
        def mutate(doc):
            doc["sources"][0]["seed"] = -1

        self.assert_spec_error(lambda: self.load_mutated(source_mutator=mutate), "seed")

        def mutate_bool(doc):
            doc["sources"][0]["event_count"] = True

        self.assert_spec_error(
            lambda: self.load_mutated(source_mutator=mutate_bool), "event_count"
        )

    def test_window_width_and_lateness_are_positive(self) -> None:
        def mutate(doc):
            doc["rules"][0]["width_ms"] = 0

        exc = self.assert_spec_error(
            lambda: self.load_mutated(rules_mutator=mutate), "width_ms"
        )
        self.assertEqual(exc.details["kind"], "window")

        def mutate_late(doc):
            doc["rules"][1]["allowed_lateness_ms"] = 0

        self.assert_spec_error(
            lambda: self.load_mutated(rules_mutator=mutate_late),
            "allowed_lateness_ms",
        )

    def test_partition_key_must_exist_in_every_source(self) -> None:
        def mutate(doc):
            doc["sources"][2]["partition_values"] = {"other": ["x"]}

        self.assert_spec_error(
            lambda: self.load_mutated(source_mutator=mutate),
            "partition_values.region",
        )

    def test_derivation_target_must_exist(self) -> None:
        def mutate(doc):
            doc["derivations"][0]["child_rule_id"] = "missing"

        self.assert_spec_error(
            lambda: self.load_mutated(budget_mutator=mutate), "child_rule_id"
        )

    def test_derivation_cycle_is_rejected(self) -> None:
        def mutate(doc):
            doc["derivations"][1]["child_rule_id"] = "base"

        exc = self.assert_spec_error(
            lambda: self.load_mutated(budget_mutator=mutate), "derivations"
        )
        self.assertEqual(exc.details["kind"], "window")
        self.assertIn(exc.details["rule_id"], exc.details["cycle"])

    def test_rule_override_must_exist(self) -> None:
        def mutate(doc):
            doc["late_budgets"]["rule_overrides"]["missing"] = {
                "event_count_limit": 1,
                "absolute_measure_limit_fixed": 1,
            }

        self.assert_spec_error(
            lambda: self.load_mutated(budget_mutator=mutate),
            "rule_overrides",
        )

    def test_duplicate_json_key_is_rejected(self) -> None:
        path = self.directory / "duplicate.json"
        path.write_text('{"format_version":1,"format_version":1}', encoding="utf-8")
        with self.assertRaises(SpecError) as caught:
            from lateness_ledger.spec import load_json_utf8

            load_json_utf8(path)
        self.assertIn("duplicate JSON key", caught.exception.message)


class DeterministicSimulationTests(unittest.TestCase):
    def test_three_strategies_share_one_event_stream(self) -> None:
        streams = [
            generate_events(sample_bundle(strategy))
            for strategy in ("source_max", "partition_min", "fixed_step")
        ]
        comparable = [
            [event.to_public_dict() for event in stream] for stream in streams
        ]
        self.assertEqual(comparable[0], comparable[1])
        self.assertEqual(comparable[1], comparable[2])

    def test_zero_and_near_limit_out_of_order_sources(self) -> None:
        bundle = sample_bundle("source_max")
        events = generate_events(bundle)
        by_source = {
            source.source_id: sorted(
                [event for event in events if event.source_id == source.source_id],
                key=lambda event: event.source_event_index,
            )
            for source in bundle.sources
        }
        alpha = by_source["alpha_clean"]
        self.assertFalse(
            any(current.event_time_ms < previous.event_time_ms for previous, current in zip(alpha, alpha[1:]))
        )
        beta = by_source["beta_bursty"]
        regressions = sum(
            1
            for previous, current in zip(beta, beta[1:])
            if current.event_time_ms < previous.event_time_ms
        )
        self.assertGreater(regressions, 0)
        self.assertLessEqual(regressions * 10000 / max(1, len(beta) - 1), 9500)


if __name__ == "__main__":
    unittest.main()
