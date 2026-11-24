from __future__ import annotations

import json
import unittest

from leasecache.arbiter import arbitrate
from leasecache.digest import ArtifactStore, generate_artifact
from leasecache.errors import LeaseError
from leasecache.spec import load_spec

from .helpers import free_ports, make_spec, test_tempdir, write_spec


class SpecValidationTests(unittest.TestCase):
    def test_valid_spec_and_topological_layers(self) -> None:
        directory = test_tempdir()
        spec = load_spec(write_spec(directory, make_spec(free_ports(2))))
        self.assertEqual(spec.pipeline_id, "test-pipeline")
        self.assertEqual(spec.topological_layers(), [["src/a", "src/b"], ["pkg/all"]])
        self.assertEqual(spec.sorted_node_ids, ("node-a", "node-b"))

    def test_first_duplicate_port_has_e_spec_field(self) -> None:
        directory = test_tempdir()
        raw = make_spec(free_ports(2))
        raw["nodes"][1]["port"] = raw["nodes"][0]["port"]
        path = write_spec(directory, raw)
        with self.assertRaises(LeaseError) as caught:
            load_spec(path)
        error = caught.exception
        self.assertEqual(error.code, "E_SPEC")
        self.assertEqual(error.field, "nodes[1].port")
        self.assertIn("globally unique", str(error.expected))

    def test_non_loopback_and_boolean_integer_rejected(self) -> None:
        directory = test_tempdir()
        raw = make_spec(free_ports(1))
        raw["nodes"][0]["host"] = "0.0.0.0"
        with self.assertRaises(LeaseError) as caught:
            load_spec(write_spec(directory, raw, "host.json"))
        self.assertEqual(caught.exception.field, "nodes[0].host")
        raw = make_spec(free_ports(1))
        raw["nodes"][0]["capacity_slots"] = True
        with self.assertRaises(LeaseError) as caught:
            load_spec(write_spec(directory, raw, "bool.json"))
        self.assertEqual(caught.exception.field, "nodes[0].capacity_slots")

    def test_reference_cycle_locates_artifact_field(self) -> None:
        directory = test_tempdir()
        stages = [
            {
                "id": "one",
                "concurrency": 1,
                "artifacts": [
                    {"key": "a", "references": ["b"]},
                    {"key": "b", "references": ["a"]},
                ],
            }
        ]
        raw = make_spec(free_ports(1), stages=stages)
        with self.assertRaises(LeaseError) as caught:
            load_spec(write_spec(directory, raw))
        error = caught.exception
        self.assertEqual(error.code, "E_SPEC")
        self.assertEqual(error.field, "pipeline.stages[0].artifacts[0].references")
        self.assertIn("acyclic", str(error.expected))

    def test_missing_reference_rejected(self) -> None:
        directory = test_tempdir()
        stages = [
            {
                "id": "one",
                "concurrency": 1,
                "artifacts": [{"key": "a", "references": ["missing"]}],
            }
        ]
        raw = make_spec(free_ports(1), stages=stages)
        with self.assertRaises(LeaseError) as caught:
            load_spec(write_spec(directory, raw))
        self.assertEqual(
            caught.exception.field, "pipeline.stages[0].artifacts[0].references"
        )


class DeterministicDigestTests(unittest.TestCase):
    def test_same_node_and_key_are_bit_stable(self) -> None:
        directory = test_tempdir()
        raw = make_spec(free_ports(2), seeds=[7, 7])
        spec = load_spec(write_spec(directory, raw))
        first = generate_artifact(spec.nodes[0], "src/a")
        second = generate_artifact(spec.nodes[0], "src/a")
        self.assertEqual(first.digest, second.digest)
        self.assertEqual(first.content, second.content)
        other_key = generate_artifact(spec.nodes[0], "src/b")
        self.assertNotEqual(first.digest, other_key.digest)
        other_seed = generate_artifact(spec.nodes[1], "src/a")
        self.assertEqual(first.digest, other_seed.digest)

    def test_workspace_files_and_manifest_match_digest(self) -> None:
        directory = test_tempdir()
        spec = load_spec(write_spec(directory, make_spec(free_ports(1))))
        store = ArtifactStore(directory / "workspace", spec.nodes[0])
        digests, refs, manifest = store.materialize(spec.artifacts)
        self.assertEqual(sorted(digests), ["pkg/all", "src/a", "src/b"])
        self.assertEqual(refs["pkg/all"], ["src/a", "src/b"])
        for entry in manifest:
            self.assertEqual(
                (directory / "workspace" / entry["file"]).read_bytes(),
                generate_artifact(spec.nodes[0], entry["key"]).content,
            )


class ArbitrationTests(unittest.TestCase):
    def test_digest_capacity_and_node_id_ordering_is_stable(self) -> None:
        nodes = [
            {
                "node_id": "node-a",
                "free_slots": 4,
                "active_leases": 0,
                "lease_limit": 2,
            },
            {
                "node_id": "node-b",
                "free_slots": 9,
                "active_leases": 0,
                "lease_limit": 2,
            },
            {
                "node_id": "node-c",
                "free_slots": 6,
                "active_leases": 0,
                "lease_limit": 2,
            },
        ]
        digests = {"node-a": "same", "node-b": "different", "node-c": "same"}
        result = arbitrate("src/a", "same", list(reversed(nodes)), digests)
        self.assertEqual(result.selected_node, "node-c")
        verdicts = {verdict.node_id: verdict for verdict in result.verdicts}
        self.assertEqual(verdicts["node-b"].veto_reason, "DIGEST_MISMATCH")
        self.assertEqual(verdicts["node-a"].veto_reason, "FEWER_FREE_SLOTS")
        reversed_result = arbitrate("src/a", "same", nodes, digests)
        self.assertEqual(
            [item.to_dict() for item in result.verdicts],
            [item.to_dict() for item in reversed_result.verdicts],
        )

    def test_full_nodes_require_queue_not_bypass(self) -> None:
        nodes = [
            {
                "node_id": "node-a",
                "free_slots": 4,
                "active_leases": 2,
                "lease_limit": 2,
            }
        ]
        result = arbitrate("src/a", "same", nodes, {"node-a": "same"})
        self.assertIsNone(result.selected_node)
        self.assertTrue(result.queue_required)
        self.assertEqual(result.verdicts[0].veto_reason, "E_LEASE_LIMIT")


if __name__ == "__main__":
    unittest.main()
