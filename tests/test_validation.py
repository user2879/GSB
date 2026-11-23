from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from drift_diag.declarations import load_declaration, normalize_declaration, validate_declaration
from drift_diag.errors import DiagnosticError
from drift_diag.events import load_events, validate_events
from drift_diag.models import FIELD_ORDER

from .helpers import sample_path, target_v3, write_ndjson


class DeclarationValidationTests(unittest.TestCase):
    def assert_format_error(self, raw, field, item=0):
        with self.assertRaises(DiagnosticError) as context:
            validate_declaration(raw)
        self.assertEqual(context.exception.code, "E_FORMAT")
        self.assertEqual(context.exception.field, field)
        if item is not None:
            self.assertEqual(context.exception.item, item)

    def test_unknown_version_and_missing_root_fields(self):
        self.assert_format_error({"format_version": 9, "workloads": []}, "format_version")
        self.assert_format_error({"workloads": []}, "format_version")
        self.assert_format_error({"format_version": 1}, "workloads")
        self.assert_format_error([], "<root>")

    def test_v2_missing_required_field(self):
        self.assert_format_error(
            {
                "format_version": 2,
                "workloads": [
                    {"name": "api", "replicas": 1, "resource_limit": 10}
                ],
            },
            "image",
        )

    def test_first_invalid_workload_is_reported(self):
        self.assert_format_error(
            {
                "format_version": 1,
                "workloads": [
                    {"name": "ok", "replicas": 1},
                    {"name": "bad", "replicas": -1},
                    {"name": "also-bad", "replicas": 2},
                ],
            },
            "replicas",
            item=1,
        )

    def test_integer_rules_reject_bool_and_nonpositive_resource(self):
        self.assert_format_error(
            {"format_version": 1, "workloads": [{"name": "a", "replicas": True}]},
            "replicas",
        )
        self.assert_format_error(
            {
                "format_version": 2,
                "workloads": [
                    {
                        "name": "a",
                        "replicas": 1,
                        "image": "x:1",
                        "resource_limit": 0,
                    }
                ],
            },
            "resource_limit",
        )

    def test_invalid_label_key(self):
        self.assert_format_error(
            {
                "format_version": 3,
                "workloads": [
                    {
                        "name": "a",
                        "replicas": 1,
                        "image": "x:1",
                        "resource_limit": 1,
                        "probe_ms": 1,
                        "labels": {"Bad Key": "x"},
                        "parent_ref": None,
                    }
                ],
            },
            "labels",
        )

    def test_dangling_parent_reference(self):
        self.assert_format_error(
            {
                "format_version": 3,
                "workloads": [
                    {
                        "name": "a",
                        "replicas": 1,
                        "image": "x:1",
                        "resource_limit": 1,
                        "probe_ms": 1,
                        "labels": {},
                        "parent_ref": "missing",
                    }
                ],
            },
            "parent_ref",
        )

    def test_duplicate_workload_name(self):
        self.assert_format_error(
            {
                "format_version": 1,
                "workloads": [
                    {"name": "a", "replicas": 1},
                    {"name": "a", "replicas": 2},
                ],
            },
            "name",
            item=1,
        )

    def test_v1_rejects_future_fields(self):
        self.assert_format_error(
            {
                "format_version": 1,
                "workloads": [
                    {"name": "a", "replicas": 1, "image": "x:1"}
                ],
            },
            "image",
        )

    def test_strict_json_rejects_duplicate_object_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(
                '{"format_version":1,"workloads":[{"name":"a","replicas":1,"replicas":2}]}',
                encoding="utf-8",
            )
            with self.assertRaises(DiagnosticError) as context:
                load_declaration(path)
        self.assertEqual(context.exception.code, "E_FORMAT")


class NormalizationTests(unittest.TestCase):
    def test_v1_and_v2_fill_fields_with_traceable_defaults(self):
        raw_v1 = {
            "format_version": 1,
            "workloads": [{"name": "api", "replicas": 2}],
        }
        target = normalize_declaration(raw_v1, defaults_version="defaults-v1")
        workload = target.workloads["api"]
        self.assertEqual(workload.filled_fields, FIELD_ORDER[1:])
        self.assertEqual(workload.fields["resource_limit"], 1024)
        self.assertEqual(workload.provenance["image"].source, "default")
        self.assertEqual(
            workload.provenance["image"].rule_id, "defaults-v1/image/registry-baseline"
        )
        self.assertEqual(workload.fill_basis["image"]["default_value"], "registry.local/default:1.0")

        raw_v2 = json.loads(sample_path("declaration-v2.json").read_text(encoding="utf-8"))
        target_v2 = normalize_declaration(raw_v2, defaults_version="defaults-v1")
        self.assertEqual(target_v2.workloads["worker"].filled_fields, ("probe_ms", "labels", "parent_ref"))

    def test_v3_has_no_filled_fields(self):
        target = target_v3()
        for workload in target.ordered_workloads():
            self.assertEqual(workload.filled_fields, ())
            self.assertEqual(workload.fill_basis, {})

    def test_defaults_versions_are_observable(self):
        raw = {"format_version": 1, "workloads": [{"name": "api", "replicas": 1}]}
        target_v2_rules = normalize_declaration(raw, defaults_version="defaults-v2")
        workload = target_v2_rules.workloads["api"]
        self.assertEqual(workload.fields["resource_limit"], 2048)
        self.assertEqual(workload.provenance["labels"].defaults_version, "defaults-v2")


class EventValidationTests(unittest.TestCase):
    def setUp(self):
        self.target = target_v3()

    def test_out_of_order_lines_are_sorted_by_sequence(self):
        events = [
            {"seq": 2, "workload": "worker", "field": "replicas", "value": 2, "at_ms": 20},
            {"seq": 1, "workload": "api-gateway", "field": "replicas", "value": 3, "at_ms": 10},
        ]
        validated = validate_events(events, self.target)
        self.assertEqual([event.seq for event in validated], [1, 2])

    def test_duplicate_and_missing_sequences(self):
        with self.assertRaises(DiagnosticError) as context:
            validate_events(
                [
                    {"seq": 1, "workload": "worker", "field": "replicas", "value": 2, "at_ms": 1},
                    {"seq": 1, "workload": "cache", "field": "replicas", "value": 1, "at_ms": 2},
                ],
                self.target,
            )
        self.assertEqual(context.exception.code, "E_EVENT")
        with self.assertRaises(DiagnosticError) as context:
            validate_events(
                [{"seq": 2, "workload": "worker", "field": "replicas", "value": 2, "at_ms": 1}],
                self.target,
            )
        self.assertEqual(context.exception.code, "E_EVENT")

    def test_unknown_workload_path_and_type_conflict(self):
        cases = [
            {"seq": 1, "workload": "missing", "field": "replicas", "value": 1, "at_ms": 1},
            {"seq": 1, "workload": "worker", "field": "unknown", "value": 1, "at_ms": 1},
            {"seq": 1, "workload": "worker", "field": "replicas", "value": "two", "at_ms": 1},
            {"seq": 1, "workload": "worker", "field": "labels", "value": {"Bad Key": "x"}, "at_ms": 1},
        ]
        for event in cases:
            with self.subTest(event=event), self.assertRaises(DiagnosticError) as context:
                validate_events([event], self.target)
            self.assertEqual(context.exception.code, "E_EVENT")

    def test_event_timestamps_are_monotonic_after_sequence_sort(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.ndjson"
            write_ndjson(
                path,
                [
                    {"seq": 1, "workload": "worker", "field": "replicas", "value": 2, "at_ms": 20},
                    {"seq": 2, "workload": "worker", "field": "image", "value": "x:1", "at_ms": 10},
                ],
            )
            with self.assertRaises(DiagnosticError) as context:
                load_events(path, self.target)
        self.assertEqual(context.exception.code, "E_EVENT")
        self.assertEqual(context.exception.field, "at_ms")


if __name__ == "__main__":
    unittest.main()
