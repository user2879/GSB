"""Requirement 1: strong JSON/NDJSON validation with E_SPEC."""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from grayroute.errors import SpecError
from grayroute.spec import (load_backends, load_merge_rule, load_requests,
                            load_rules)

CONFIG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


class SpecValidationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, name, obj):
        with open(os.path.join(self.dir, name), "w", encoding="utf-8") as fh:
            json.dump(obj, fh)

    def test_sample_spec_loads(self):
        backends = load_backends(os.path.join(CONFIG, "backends.json"))
        rules = load_rules(os.path.join(CONFIG, "rules.json"), backends)
        merge = load_merge_rule(os.path.join(CONFIG, "merge_rules.json"))
        requests = load_requests(os.path.join(DATA, "requests.ndjson"),
                                 rules, 8)
        self.assertEqual(len(backends), 4)
        self.assertEqual(len(requests), 12)
        self.assertEqual(merge.order[0], "param_version_seq")

    def test_reject_non_loopback_host(self):
        backends = [{
            "backend_id": "b1", "host": "10.0.0.1", "port": 19000,
            "model_dim": 2, "hidden_dim": 3, "quantization": "int8",
            "param_format": "r2", "param_summary": "x", "role": "stable"}]
        self._write("b.json", backends)
        with self.assertRaises(SpecError) as ctx:
            load_backends(os.path.join(self.dir, "b.json"))
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertEqual(ctx.exception.field, "host")

    def test_reject_duplicate_port(self):
        row = {"host": "127.0.0.1", "model_dim": 2, "hidden_dim": 3,
               "quantization": "int8", "param_format": "r2",
               "param_summary": "x"}
        backends = [
            dict(row, backend_id="b1", port=19001, role="stable"),
            dict(row, backend_id="b2", port=19001, role="gray")]
        self._write("b.json", backends)
        with self.assertRaises(SpecError) as ctx:
            load_backends(os.path.join(self.dir, "b.json"))
        self.assertEqual(ctx.exception.field, "port")

    def test_reject_bad_dimension_and_enum(self):
        for bad, field in [
            ({"model_dim": 0}, "model_dim"),
            ({"quantization": "fp8"}, "quantization"),
            ({"param_format": "r9"}, "param_format"),
        ]:
            row = {"backend_id": "b1", "host": "127.0.0.1", "port": 19002,
                   "model_dim": 2, "hidden_dim": 3, "quantization": "int8",
                   "param_format": "r2", "param_summary": "x",
                   "role": "stable", **bad}
            self._write("b.json", [row])
            with self.assertRaises(SpecError) as ctx:
                load_backends(os.path.join(self.dir, "b.json"))
            self.assertEqual(ctx.exception.field, field)

    def test_reject_weight_out_of_range(self):
        backends = load_backends(os.path.join(CONFIG, "backends.json"))
        rules = [{"rule_id": "r", "revisions": [
            {"revision": 1, "weight_permyriad": 10001, "batch_size": 2,
             "param_version_seq": 1, "param_summary": "s",
             "backend_ids": ["stable-a", "gray-a"]}]}]
        self._write("r.json", rules)
        with self.assertRaises(SpecError) as ctx:
            load_rules(os.path.join(self.dir, "r.json"), backends)
        self.assertEqual(ctx.exception.field, "weight_permyriad")

    def test_reject_non_monotonic_revision(self):
        backends = load_backends(os.path.join(CONFIG, "backends.json"))
        rules = [{"rule_id": "r", "revisions": [
            {"revision": 2, "weight_permyriad": 10, "batch_size": 2,
             "param_version_seq": 1, "param_summary": "s",
             "backend_ids": ["stable-a", "gray-a"]},
            {"revision": 2, "weight_permyriad": 10, "batch_size": 2,
             "param_version_seq": 2, "param_summary": "t",
             "backend_ids": ["stable-a", "gray-a"]}]}]
        self._write("r.json", rules)
        with self.assertRaises(SpecError) as ctx:
            load_rules(os.path.join(self.dir, "r.json"), backends)
        self.assertEqual(ctx.exception.field, "revision")

    def test_reject_duplicate_arrival_and_vector_length(self):
        backends = load_backends(os.path.join(CONFIG, "backends.json"))
        rules = load_rules(os.path.join(CONFIG, "rules.json"), backends)
        rows = [
            {"request_id": "a", "arrival_seq": 0, "client_key": "c",
             "rule_id": "route-main", "rule_revision": 1,
             "vector": [0.1] * 8},
            {"request_id": "b", "arrival_seq": 0, "client_key": "c",
             "rule_id": "route-main", "rule_revision": 1,
             "vector": [0.1] * 8},
        ]
        path = os.path.join(self.dir, "req.ndjson")
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        with self.assertRaises(SpecError) as ctx:
            load_requests(path, rules, 8)
        self.assertEqual(ctx.exception.field, "arrival_seq")

        rows[1]["arrival_seq"] = 1
        rows[1]["vector"] = [0.1] * 3
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        with self.assertRaises(SpecError) as ctx:
            load_requests(path, rules, 8)
        self.assertEqual(ctx.exception.field, "vector")


if __name__ == "__main__":
    unittest.main()
