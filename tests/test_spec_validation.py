"""Requirement 1: strict local declaration validation (E_SPEC)."""
import json
import os
import tempfile
import unittest

from shardmeta.errors import SpecError
from shardmeta.spec import (
    load_node_specs,
    load_object_manifests,
    load_request_stream,
)


def write_tmp(payload):
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(payload, handle)
    handle.close()
    return handle.name


class SpecValidationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.nodes_doc = {"nodes": [
            {"node_id": "n1", "host": "127.0.0.1", "port": 42001,
             "capacity_slots": 2, "lease_limit": 2, "seed": 1, "meta_version": 1},
            {"node_id": "n2", "host": "127.0.0.1", "port": 42002,
             "capacity_slots": 2, "lease_limit": 2, "seed": 2, "meta_version": 1},
        ]}
        self.nodes_path = write_tmp(self.nodes_doc)
        self.manifest_doc = {"objects": [{
            "object_key": "k1", "total_bytes": 16, "shard_size": 8,
            "content_seed": 9,
            "placements": [
                {"shard_index": 0, "node_id": "n1", "size": 8},
                {"shard_index": 1, "node_id": "n2", "size": 8},
            ],
        }]}
        self.manifest_path = write_tmp(self.manifest_doc)

    def test_valid_bundle_loads(self):
        nodes = load_node_specs(self.nodes_path)
        manifests = load_object_manifests(self.manifest_path, nodes)
        self.assertEqual(set(nodes), {"n1", "n2"})
        self.assertEqual(manifests["k1"].shard_count, 2)

    def test_duplicate_node_id(self):
        self.nodes_doc["nodes"][1]["node_id"] = "n1"
        path = write_tmp(self.nodes_doc)
        with self.assertRaises(SpecError) as ctx:
            load_node_specs(path)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertEqual(ctx.exception.field, "node_id")

    def test_host_must_be_loopback_literal(self):
        self.nodes_doc["nodes"][0]["host"] = "localhost"
        path = write_tmp(self.nodes_doc)
        with self.assertRaises(SpecError) as ctx:
            load_node_specs(path)
        self.assertEqual(ctx.exception.field, "host")

    def test_duplicate_port(self):
        self.nodes_doc["nodes"][1]["port"] = 42001
        path = write_tmp(self.nodes_doc)
        with self.assertRaises(SpecError) as ctx:
            load_node_specs(path)
        self.assertEqual(ctx.exception.field, "port")

    def test_non_positive_capacity(self):
        self.nodes_doc["nodes"][0]["capacity_slots"] = 0
        path = write_tmp(self.nodes_doc)
        with self.assertRaises(SpecError) as ctx:
            load_node_specs(path)
        self.assertEqual(ctx.exception.field, "capacity_slots")

    def test_shard_size_must_divide_total(self):
        self.manifest_doc["objects"][0]["shard_size"] = 7
        path = write_tmp(self.manifest_doc)
        nodes = load_node_specs(self.nodes_path)
        with self.assertRaises(SpecError) as ctx:
            load_object_manifests(path, nodes)
        self.assertEqual(ctx.exception.field, "shard_size")

    def test_unknown_placement_node(self):
        self.manifest_doc["objects"][0]["placements"][0]["node_id"] = "ghost"
        path = write_tmp(self.manifest_doc)
        nodes = load_node_specs(self.nodes_path)
        with self.assertRaises(SpecError) as ctx:
            load_object_manifests(path, nodes)
        self.assertEqual(ctx.exception.field, "placements.node_id")

    def test_request_seq_monotonic_and_unknown_key(self):
        nodes = load_node_specs(self.nodes_path)
        manifests = load_object_manifests(self.manifest_path, nodes)
        good = os.path.join(self.tmp, "good.jsonl")
        with open(good, "w") as handle:
            handle.write('{"seq":1,"op":"read","object_key":"k1"}\n')
            handle.write('{"seq":2,"op":"read","object_key":"k1"}\n')
        requests = load_request_stream(good, manifests)
        self.assertEqual([r.seq for r in requests], [1, 2])

        bad = os.path.join(self.tmp, "bad.jsonl")
        with open(bad, "w") as handle:
            handle.write('{"seq":2,"op":"read","object_key":"k1"}\n')
            handle.write('{"seq":2,"op":"read","object_key":"k1"}\n')
        with self.assertRaises(SpecError) as ctx:
            load_request_stream(bad, manifests)
        self.assertEqual(ctx.exception.field, "seq")

        unknown = os.path.join(self.tmp, "unknown.jsonl")
        with open(unknown, "w") as handle:
            handle.write('{"seq":1,"op":"read","object_key":"nope"}\n')
        with self.assertRaises(SpecError) as ctx:
            load_request_stream(unknown, manifests)
        self.assertEqual(ctx.exception.field, "object_key")


if __name__ == "__main__":
    unittest.main()
