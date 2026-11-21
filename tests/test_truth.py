from __future__ import annotations

import unittest
from replica_repair.model import Manifest
from replica_repair.truth import CopyState, TruthEngine
from tests.helpers import stable_json


DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64


def manifest_payload(priorities: dict[str, int], reverse: bool = False) -> dict:
    nodes = [
        {
            "node_id": node_id,
            "priority": priority,
            "blocks": [
                {
                    "block_id": "blk-0001",
                    "size_bytes": 1024,
                    "digest": DIGEST_A,
                    "repair_weight": 1,
                }
            ],
        }
        for node_id, priority in priorities.items()
    ]
    if reverse:
        nodes.reverse()
    return {
        "format_version": 2,
        "group_id": "truth-test",
        "manifest_version": 1,
        "nodes": nodes,
    }


def state(node_id: str, digest: str, priority: int) -> CopyState:
    return CopyState(
        node_id=node_id,
        block_id="blk-0001",
        size_bytes=1024,
        digest=digest,
        expected_size_bytes=1024,
        expected_digest=digest,
        priority=priority,
        status="ok",
    )


class TruthTests(unittest.TestCase):
    def test_priority_weight_breaks_frequency_tie(self) -> None:
        manifest = Manifest.from_dict(
            manifest_payload({"n-a": 5, "n-b": 3, "n-c": 2, "n-d": 1})
        )
        states = [
            state("n-d", DIGEST_B, 1),
            state("n-c", DIGEST_B, 2),
            state("n-b", DIGEST_A, 3),
            state("n-a", DIGEST_A, 5),
        ]
        report = TruthEngine(manifest).determine(states)
        block = report.blocks[0]
        self.assertEqual(block.truth_digest, DIGEST_A)
        self.assertEqual(block.truth_nodes, ("n-a", "n-b"))
        self.assertEqual([item.node_id for item in block.mismatches], ["n-c", "n-d"])
        self.assertEqual(block.ranking[0].priority_weight, 8)
        self.assertEqual(block.ranking[1].priority_weight, 3)

    def test_node_identifier_order_breaks_equal_weight_tie(self) -> None:
        manifest = Manifest.from_dict(
            manifest_payload({"n-a": 1, "n-b": 1, "n-c": 1, "n-d": 1})
        )
        states = [
            state("n-c", DIGEST_B, 1),
            state("n-b", DIGEST_B, 1),
            state("n-d", DIGEST_A, 1),
            state("n-a", DIGEST_A, 1),
        ]
        report = TruthEngine(manifest).determine(states)
        block = report.blocks[0]
        self.assertEqual(block.truth_digest, DIGEST_A)
        self.assertEqual(block.truth_nodes, ("n-a", "n-d"))
        self.assertEqual(block.ranking[0].node_ids, ("n-a", "n-d"))

    def test_output_is_byte_identical_and_independent_of_input_order(self) -> None:
        manifest = Manifest.from_dict(
            manifest_payload({"n-a": 5, "n-b": 3, "n-c": 2, "n-d": 1})
        )
        states = [
            state("n-a", DIGEST_A, 5),
            state("n-b", DIGEST_A, 3),
            state("n-c", DIGEST_B, 2),
            state("n-d", DIGEST_B, 1),
        ]
        first = TruthEngine(manifest).determine(states).to_dict()
        second = TruthEngine(manifest).determine(reversed(states)).to_dict()
        self.assertEqual(stable_json(first), stable_json(second))
        self.assertEqual(stable_json(first), stable_json(TruthEngine(manifest).determine(states).to_dict()))

        reversed_manifest = Manifest.from_dict(
            manifest_payload({"n-a": 5, "n-b": 3, "n-c": 2, "n-d": 1}, reverse=True)
        )
        self.assertEqual(
            stable_json(first),
            stable_json(TruthEngine(reversed_manifest).determine(states).to_dict()),
        )


if __name__ == "__main__":
    unittest.main()
