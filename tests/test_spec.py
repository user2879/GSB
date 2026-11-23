import copy
import json
import os
import tempfile
import unittest

from repairbench.errors import E_SPEC, WorkbenchError
from repairbench.spec import load_spec

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "samples", "groups.json")


def base_group():
    with open(SAMPLE, encoding="utf-8") as fh:
        return json.load(fh)["groups"][0]


def write_tmp(obj):
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)
    return path


class TestSpec(unittest.TestCase):
    def test_sample_loads(self):
        groups = load_spec(SAMPLE)
        self.assertEqual([g.group_id for g in groups], ["alpha", "beta"])
        self.assertEqual(groups[0].params.segment_bytes, 4096)

    def expect_spec(self, group, field_part):
        path = write_tmp(group)
        try:
            with self.assertRaises(WorkbenchError) as cm:
                load_spec(path)
            self.assertEqual(cm.exception.code, E_SPEC)
            self.assertIn(field_part, cm.exception.context["field"])
            self.assertIn("entry", cm.exception.context)
        finally:
            os.remove(path)

    def test_duplicate_node_id(self):
        g = base_group()
        g["nodes"].append(copy.deepcopy(g["nodes"][0]))
        self.expect_spec(g, "node_id")

    def test_bad_host(self):
        g = base_group()
        g["nodes"][0]["host"] = "localhost"
        self.expect_spec(g, "host")

    def test_duplicate_port(self):
        g = base_group()
        g["nodes"][1]["port"] = g["nodes"][0]["port"]
        self.expect_spec(g, "port")

    def test_negative_bandwidth(self):
        g = base_group()
        g["nodes"][0]["bandwidth_kbps"] = -1
        self.expect_spec(g, "bandwidth_kbps")

    def test_negative_weight(self):
        g = base_group()
        g["nodes"][0]["priority_weight"] = -2
        self.expect_spec(g, "priority_weight")

    def test_zero_block_bytes(self):
        g = base_group()
        g["blocks"][0]["bytes"] = 0
        self.expect_spec(g, "bytes")

    def test_duplicate_block_id(self):
        g = base_group()
        g["blocks"].append(copy.deepcopy(g["blocks"][0]))
        self.expect_spec(g, "block_id")

    def test_unsupported_segment(self):
        g = base_group()
        g["session"]["segment_bytes"] = 3000
        self.expect_spec(g, "segment_bytes")

    def test_negative_budget(self):
        g = base_group()
        g["session"]["global_budget_bytes"] = -1
        self.expect_spec(g, "global_budget_bytes")

    def test_negative_concurrency(self):
        g = base_group()
        g["session"]["concurrency_limit"] = -1
        self.expect_spec(g, "concurrency_limit")

    def test_bad_scale_exponent(self):
        g = base_group()
        g["session"]["scale_exponent"] = 7
        self.expect_spec(g, "scale_exponent")

    def test_block_unknown_node(self):
        g = base_group()
        g["blocks"][0]["node_id"] = "ghost"
        self.expect_spec(g, "node_id")

    def test_first_invalid_entry_reported(self):
        g = base_group()
        g["nodes"][0]["bandwidth_kbps"] = -1
        g["nodes"][1]["priority_weight"] = -1
        path = write_tmp(g)
        try:
            with self.assertRaises(WorkbenchError) as cm:
                load_spec(path)
            self.assertEqual(cm.exception.context["entry"], "group.nodes[0]")
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
