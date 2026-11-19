from __future__ import annotations

import json
import unittest

from srebudget.errors import SreError
from srebudget.loader import load_chain_data, load_trace_lines

from .support import call, chain, service, trace


class ChainValidationTests(unittest.TestCase):
    def assert_chain_error(self, data, field_suffix: str) -> SreError:
        with self.assertRaises(SreError) as raised:
            load_chain_data(data)
        self.assertEqual(raised.exception.code, "E_CHAIN")
        self.assertTrue(raised.exception.details["field_path"].endswith(field_suffix))
        return raised.exception

    def test_duplicate_service_and_bad_reference(self):
        duplicate = {
            "version": "v1",
            "services": [service("a", []), service("a", [])],
        }
        self.assert_chain_error(duplicate, ".id")

        bad_ref = {
            "version": "v1",
            "services": [service("a", ["missing"])],
        }
        self.assert_chain_error(bad_ref, ".upstream[0]")

    def test_cycle_is_rejected(self):
        data = {
            "version": "v1",
            "services": [service("a", ["b"]), service("b", ["a"])],
        }
        error = self.assert_chain_error(data, "upstream")
        self.assertEqual(set(error.details["cycle"]), {"a", "b"})
        self.assertEqual(error.details["cycle"][0], error.details["cycle"][-1])

    def test_numeric_constraints_and_field_paths(self):
        cases = [
            (service("a", [], timeout_ms=0), ".timeout_ms"),
            (service("a", [], ratio_bp=10001), ".ratio_bp"),
            (service("a", [], min_samples=0), ".min_samples"),
            (service("a", [], cooldown_ms=0), ".cooldown_ms"),
            (service("a", [], failure_threshold=0), ".failure_threshold"),
            (service("a", [], half_open_probes=0), ".half_open_probes"),
        ]
        for node, suffix in cases:
            with self.subTest(suffix=suffix):
                self.assert_chain_error(
                    {"version": "v1", "services": [node]}, suffix
                )

        boolean_timeout = service("a", [], timeout_ms=True)
        self.assert_chain_error(
            {"version": "v1", "services": [boolean_timeout]}, ".timeout_ms"
        )


class TraceValidationTests(unittest.TestCase):
    def setUp(self):
        self.chain = chain([service("leaf", ["root"]), service("root", [])])

    def assert_trace_error(self, lines, line_number: int, field_suffix: str):
        with self.assertRaises(SreError) as raised:
            load_trace_lines(lines, self.chain)
        self.assertEqual(raised.exception.code, "E_TRACE")
        self.assertEqual(raised.exception.details["line"], line_number)
        self.assertTrue(raised.exception.details["field_path"].endswith(field_suffix))

    def test_timestamp_regression_reports_second_line(self):
        lines = [
            json.dumps(call(100, ["leaf", "root"])) + "\n",
            json.dumps(call(99, ["leaf", "root"])) + "\n",
        ]
        self.assert_trace_error(lines, 2, ".timestamp_ms")

    def test_unknown_service_and_bad_outcome(self):
        self.assert_trace_error(
            [json.dumps(call(1, ["ghost"])) + "\n"], 1, ".path[0]"
        )
        bad = call(1, ["leaf"])
        bad["outcome"] = "exploded"
        self.assert_trace_error([json.dumps(bad) + "\n"], 1, ".outcome")

    def test_valid_trace_is_numbered_and_monotonic(self):
        stream = trace(
            [call(10, ["leaf"]), call(10, ["leaf"], "timeout")],
            self.chain,
        )
        self.assertEqual([item.index for item in stream.calls], [1, 2])
        self.assertEqual(stream.calls[1].source_line, 2)


if __name__ == "__main__":
    unittest.main()
