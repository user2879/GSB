from __future__ import annotations

import unittest

from feature_backfill.config_loader import (
    load_feature_file,
    load_plan,
    parse_config,
)
from feature_backfill.data_loader import load_samples, rows_by_entity
from feature_backfill.errors import E_CONFIG_INVALID, E_FEATURE_DAG_INVALID, E_ROW_INVALID
from feature_backfill.plan import materialize_plan_state

from tests.helpers import DATA, load_fixture


class ValidationTests(unittest.TestCase):
    def test_valid_fixture_loads_and_topo_order_is_deterministic(self) -> None:
        loaded = load_fixture()
        self.assertEqual(len(loaded["rows"]), 32)
        self.assertEqual(len(rows_by_entity(loaded["rows"])), 8)
        self.assertEqual(loaded["raw_fields"], ["amount", "qty", "score"])
        from feature_backfill.config_loader import validate_feature_graph

        order = validate_feature_graph(loaded["features"], loaded["raw_fields"])
        self.assertLess(
            order.index("amount_mean_5000"), order.index("amount_chain_5000")
        )
        self.assertEqual(order, validate_feature_graph(loaded["features"], loaded["raw_fields"]))

    def test_bad_rows_are_isolated_with_e_row_invalid(self) -> None:
        directory = DATA / "illegal" / "bad_rows_all"
        config = parse_config(directory / "config.yaml")
        features, raw_fields, feature_failures = load_feature_file(
            directory / "features.json"
        )
        self.assertFalse(feature_failures)
        rows, failures = load_samples(
            directory / "samples.jsonl", config=config, raw_fields=raw_fields
        )
        self.assertEqual(len(rows), 5)
        self.assertTrue(failures)
        self.assertTrue(all(failure["code"] == E_ROW_INVALID for failure in failures))
        messages = " | ".join(failure["message"] for failure in failures)
        self.assertIn("duplicate row_id", messages)
        self.assertIn("non-standard JSON number NaN", messages)
        self.assertIn("strictly increasing", messages)

    def test_feature_cycle_is_rejected(self) -> None:
        directory = DATA / "illegal" / "bad_dag"
        features, raw_fields, failures = load_feature_file(directory / "features.json")
        self.assertFalse(features.get("cycle_a") is None)
        self.assertTrue(failures)
        self.assertEqual(failures[0]["code"], E_FEATURE_DAG_INVALID)
        self.assertIn("cycle", failures[0]["message"])

    def test_zero_quota_and_lease_ttl_are_rejected(self) -> None:
        with self.assertRaises(Exception) as context:
            parse_config(DATA / "illegal" / "bad_config_quota" / "config.yaml")
        self.assertEqual(context.exception.code, E_CONFIG_INVALID)

    def test_each_illegal_row_class_can_be_triggered_alone(self) -> None:
        expected = {
            "duplicate_row": "duplicate row_id",
            "nonfinite_row": "non-standard JSON number NaN",
            "timestamp_reversal": "strictly increasing",
        }
        for name, message in expected.items():
            with self.subTest(fixture=name):
                directory = DATA / "illegal" / name
                config = parse_config(directory / "config.yaml")
                features, raw_fields, feature_failures = load_feature_file(
                    directory / "features.json"
                )
                self.assertFalse(feature_failures)
                _rows, failures = load_samples(
                    directory / "samples.jsonl",
                    config=config,
                    raw_fields=raw_fields,
                )
                self.assertEqual(len(failures), 1)
                self.assertEqual(failures[0]["code"], E_ROW_INVALID)
                self.assertIn(message, failures[0]["message"])

    def test_missing_upstream_and_missing_raw_field_are_separate(self) -> None:
        for name, fragment in {
            "missing_upstream": "missing upstream",
            "missing_raw_field": "unknown raw field",
        }.items():
            with self.subTest(fixture=name):
                directory = DATA / "illegal" / name
                _features, _raw, failures = load_feature_file(directory / "features.json")
                config = parse_config(DATA / "default" / "config.yaml")
                _rows, _row_failures = load_samples(
                    directory / "samples.jsonl",
                    config=config,
                    raw_fields=_raw,
                )
                self.assertTrue(failures)
                self.assertEqual(failures[0]["code"], E_FEATURE_DAG_INVALID)
                self.assertIn(fragment, failures[0]["message"])

    def test_zero_lease_ttl_is_triggered_alone(self) -> None:
        with self.assertRaises(Exception) as context:
            parse_config(DATA / "illegal" / "bad_config_lease_ttl" / "config.yaml")
        self.assertEqual(context.exception.code, E_CONFIG_INVALID)

    def test_bad_plan_is_rejected(self) -> None:
        plan, failures = load_plan(DATA / "illegal" / "bad_plan" / "plan.json")
        self.assertEqual(plan, [])
        self.assertTrue(failures)
        self.assertEqual(failures[0]["code"], "E_PLAN_INVALID")

    def test_plan_materialization_matches_generated_final_fixture(self) -> None:
        base = load_fixture()
        final = load_fixture(DATA / "final")
        batches, failures = load_plan(DATA / "default" / "plan.json")
        self.assertFalse(failures)
        rows, features, errors = materialize_plan_state(
            base["rows"],
            base["features"],
            batches,
            config=base["config"],
            raw_fields=base["raw_fields"],
        )
        self.assertFalse(errors)
        self.assertEqual(
            [row.to_dict() for row in sorted(rows, key=lambda row: row.row_id)],
            [row.to_dict() for row in sorted(final["rows"], key=lambda row: row.row_id)],
        )
        self.assertEqual(
            {name: feature.to_dict() for name, feature in features.items()},
            {name: feature.to_dict() for name, feature in final["features"].items()},
        )

    def test_rollback_discards_uncommitted_append(self) -> None:
        base = load_fixture()
        batches, _ = load_plan(DATA / "default" / "plan.json")
        target_ids = [batch.batch_id for batch in batches[:2]]
        rows, features, errors = materialize_plan_state(
            base["rows"],
            base["features"],
            batches,
            completed_batches=target_ids,
            config=base["config"],
            raw_fields=base["raw_fields"],
        )
        self.assertFalse(errors)
        self.assertNotIn("E009", rows_by_entity(rows))
        self.assertEqual(features["amount_sum_5000"].window_ms, 5000)


if __name__ == "__main__":
    unittest.main()
