from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from alert_pipeline.errors import PipelineError
from alert_pipeline.spec import (
    decode_alert,
    encode_alert,
    load_dataset,
    load_root_rules,
    load_service_graph,
)

from .helpers import make_config, write_graph, write_rules


class SpecTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _minimal(self, records=None):
        return make_config(self.root, records=records)

    def test_v1_v3_normalize_to_identical_core_and_trace_defaults(self) -> None:
        records = [
            {
                "version": 1,
                "timestamp_ms": 1000,
                "service_id": "svc-a",
                "level": "critical",
            },
            {
                "version": 3,
                "timestamp_ms": 1000,
                "service_id": "svc-a",
                "level": "critical",
                "fingerprint": "derived:svc-a:critical",
                "silence_window_ms": 30000,
                "root_candidates": ["svc-b"],
                "strategy_id": "s1",
            },
        ]
        config = self._minimal(records)
        dataset = load_dataset(config)
        first, second = dataset.alerts
        self.assertEqual(first.core_view(), second.core_view())
        self.assertEqual(first.format_version, 1)
        self.assertEqual(second.format_version, 3)
        self.assertEqual(first.provenance["fingerprint"].source, "default")
        self.assertEqual(first.provenance["silence_window_ms"].source, "default")
        self.assertEqual(second.provenance["fingerprint"].source, "record")
        round_trip = decode_alert(
            encode_alert(first),
            first.line_number,
            dataset.graph,
            config,
            None,
        )
        self.assertEqual(round_trip.core_view(), first.core_view())

    def test_unknown_version_is_e_field(self) -> None:
        config = self._minimal(
            [{"version": 9, "timestamp_ms": 1, "service_id": "svc-a", "level": "info"}]
        )
        with self.assertRaises(PipelineError) as caught:
            load_dataset(config)
        self.assertEqual(caught.exception.code, "E_FIELD")
        self.assertEqual(caught.exception.line, 1)
        self.assertEqual(caught.exception.field, "version")

    def test_missing_v2_field_is_e_field(self) -> None:
        config = self._minimal(
            [
                {
                    "version": 2,
                    "timestamp_ms": 1,
                    "service_id": "svc-a",
                    "level": "info",
                    "fingerprint": "a:b",
                }
            ]
        )
        with self.assertRaises(PipelineError) as caught:
            load_dataset(config)
        self.assertEqual(caught.exception.code, "E_FIELD")
        self.assertEqual(caught.exception.field, "silence_window_ms")

    def test_negative_window_is_e_field(self) -> None:
        config = self._minimal(
            [
                {
                    "version": 2,
                    "timestamp_ms": 1,
                    "service_id": "svc-a",
                    "level": "info",
                    "fingerprint": "f1",
                    "silence_window_ms": -1,
                }
            ]
        )
        with self.assertRaises(PipelineError) as caught:
            load_dataset(config)
        self.assertEqual(caught.exception.code, "E_FIELD")
        self.assertEqual(caught.exception.field, "silence_window_ms")

    def test_dangling_v3_root_reference_is_e_field(self) -> None:
        config = self._minimal(
            [
                {
                    "version": 3,
                    "timestamp_ms": 1,
                    "service_id": "svc-a",
                    "level": "info",
                    "fingerprint": "f1",
                    "silence_window_ms": 1,
                    "root_candidates": ["missing"],
                    "strategy_id": "s1",
                }
            ]
        )
        with self.assertRaises(PipelineError) as caught:
            load_dataset(config)
        self.assertEqual(caught.exception.code, "E_FIELD")
        self.assertEqual(caught.exception.field, "root_candidates[0]")

    def test_invalid_level_and_fingerprint_are_e_spec(self) -> None:
        config = self._minimal(
            [{"version": 1, "timestamp_ms": 1, "service_id": "svc-a", "level": "fatal"}]
        )
        with self.assertRaises(PipelineError) as caught:
            load_dataset(config)
        self.assertEqual(caught.exception.code, "E_SPEC")
        self.assertEqual(caught.exception.field, "level")

        config = self._minimal(
            [
                {
                    "version": 2,
                    "timestamp_ms": 1,
                    "service_id": "svc-a",
                    "level": "info",
                    "fingerprint": "bad fingerprint",
                    "silence_window_ms": 1,
                }
            ]
        )
        with self.assertRaises(PipelineError) as caught:
            load_dataset(config)
        self.assertEqual(caught.exception.code, "E_SPEC")
        self.assertEqual(caught.exception.field, "fingerprint")

    def test_nonmonotonic_timestamp_is_e_spec_at_first_bad_line(self) -> None:
        config = self._minimal(
            [
                {"version": 1, "timestamp_ms": 100, "service_id": "svc-a", "level": "info"},
                {"version": 1, "timestamp_ms": 99, "service_id": "svc-b", "level": "info"},
            ]
        )
        with self.assertRaises(PipelineError) as caught:
            load_dataset(config)
        self.assertEqual(caught.exception.code, "E_SPEC")
        self.assertEqual(caught.exception.line, 2)
        self.assertEqual(caught.exception.field, "timestamp_ms")

    def test_graph_cycle_and_duplicate_are_e_spec(self) -> None:
        graph_path = self.root / "cycle.json"
        write_graph(
            graph_path,
            ["a", "b"],
            [("a", "b"), ("b", "a")],
        )
        with self.assertRaises(PipelineError) as caught:
            load_service_graph(graph_path)
        self.assertEqual(caught.exception.code, "E_SPEC")
        self.assertEqual(caught.exception.field, "dependencies")

        duplicate_path = self.root / "duplicate.json"
        duplicate_path.write_text(
            json.dumps({"version": 1, "services": ["a", "a"], "dependencies": []}),
            encoding="utf-8",
        )
        with self.assertRaises(PipelineError) as caught:
            load_service_graph(duplicate_path)
        self.assertEqual(caught.exception.code, "E_SPEC")
        self.assertIn("services[1]", caught.exception.field)

    def test_rule_unknown_service_is_e_spec(self) -> None:
        graph_path = self.root / "services.json"
        write_graph(graph_path, ["a"], [])
        rules_path = self.root / "rules.json"
        write_rules(
            rules_path,
            rules=[
                {
                    "rule_id": "r1",
                    "services": ["missing"],
                    "directions": ["upstream"],
                }
            ],
        )
        graph = load_service_graph(graph_path)
        with self.assertRaises(PipelineError) as caught:
            load_root_rules(rules_path, graph)
        self.assertEqual(caught.exception.code, "E_SPEC")
        self.assertIn("services[0]", caught.exception.field)

    def test_context_and_nonexistent_graph_reference_are_e_spec(self) -> None:
        config = self._minimal(
            [{"version": 1, "timestamp_ms": 1, "service_id": "unknown", "level": "info"}]
        )
        with self.assertRaises(PipelineError) as caught:
            load_dataset(config)
        self.assertEqual(caught.exception.code, "E_SPEC")
        self.assertEqual(caught.exception.field, "service_id")

        graph_path = self.root / "services.json"
        write_graph(graph_path, ["a"], [("a", "missing")])
        with self.assertRaises(PipelineError) as caught:
            load_service_graph(graph_path)
        self.assertEqual(caught.exception.code, "E_SPEC")

    def test_corrupt_ndjson_is_e_spec_with_line(self) -> None:
        config = self._minimal(
            [{"version": 1, "timestamp_ms": 1, "service_id": "svc-a", "level": "info"}]
        )
        with config.alerts_path.open("a", encoding="utf-8") as handle:
            handle.write('{"version":')
        with self.assertRaises(PipelineError) as caught:
            load_dataset(config)
        self.assertEqual(caught.exception.code, "E_SPEC")
        self.assertEqual(caught.exception.line, 2)

    def test_conflicting_aliases_are_e_field(self) -> None:
        config = self._minimal(
            [
                {
                    "version": 2,
                    "format_version": 1,
                    "timestamp_ms": 1,
                    "service_id": "svc-a",
                    "level": "info",
                    "fingerprint": "f1",
                    "silence_window_ms": 1,
                }
            ]
        )
        with self.assertRaises(PipelineError) as caught:
            load_dataset(config)
        self.assertEqual(caught.exception.code, "E_FIELD")
        self.assertEqual(caught.exception.field, "version")


if __name__ == "__main__":
    unittest.main()
