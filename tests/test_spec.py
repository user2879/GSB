from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from gsbcache.errors import SpecError
from gsbcache.spec import parse_spec

from tests.helpers import make_sample, mutate_spec


class SpecValidationTests(unittest.TestCase):
    def test_valid_sample_and_topological_stages(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _path, spec, _report = make_sample(Path(td))
            self.assertEqual(spec.pipeline.pipeline_id, "windows-local-cache-demo")
            self.assertEqual([s.stage_id for s in spec.topological_stages()], ["prepare", "compile", "test"])
            self.assertEqual(len(spec.artifacts), 4)
            self.assertEqual(len(spec.node_by_id), 3)

    def test_duplicate_node_id_is_first_violation(self) -> None:
        def mutate(raw):
            raw["nodes"][1]["node_id"] = "node-a"

        with tempfile.TemporaryDirectory() as td:
            path, _ = mutate_spec(Path(td), mutate)
            with self.assertRaises(SpecError) as ctx:
                parse_spec(__import__("json").loads(path.read_text()))
            self.assertEqual(ctx.exception.code, "E_SPEC")
            self.assertEqual(ctx.exception.details["node_id"], "node-a")
            self.assertEqual(ctx.exception.details["field"], "node_id")

    def test_non_loopback_literal_rejected(self) -> None:
        def mutate(raw):
            raw["nodes"][0]["host"] = "localhost"

        with tempfile.TemporaryDirectory() as td:
            path, _ = mutate_spec(Path(td), mutate)
            with self.assertRaises(SpecError) as ctx:
                parse_spec(__import__("json").loads(path.read_text()))
            self.assertEqual(ctx.exception.details["node_id"], "node-a")
            self.assertEqual(ctx.exception.details["field"], "host")

    def test_duplicate_port_rejected(self) -> None:
        def mutate(raw):
            raw["nodes"][1]["port"] = raw["nodes"][0]["port"]

        with tempfile.TemporaryDirectory() as td:
            path, _ = mutate_spec(Path(td), mutate)
            with self.assertRaises(SpecError) as ctx:
                parse_spec(__import__("json").loads(path.read_text()))
            self.assertEqual(ctx.exception.details["node_id"], "node-b")
            self.assertEqual(ctx.exception.details["field"], "port")

    def test_nonpositive_capacity_rejected(self) -> None:
        def mutate(raw):
            raw["nodes"][0]["capacity_slots"] = 0

        with tempfile.TemporaryDirectory() as td:
            path, _ = mutate_spec(Path(td), mutate)
            with self.assertRaises(SpecError) as ctx:
                parse_spec(__import__("json").loads(path.read_text()))
            self.assertEqual(ctx.exception.details["node_id"], "node-a")
            self.assertEqual(ctx.exception.details["field"], "capacity_slots")

    def test_nonpositive_lease_limit_rejected(self) -> None:
        def mutate(raw):
            raw["nodes"][1]["lease_limit"] = -1

        with tempfile.TemporaryDirectory() as td:
            path, _ = mutate_spec(Path(td), mutate)
            with self.assertRaises(SpecError) as ctx:
                parse_spec(__import__("json").loads(path.read_text()))
            self.assertEqual(ctx.exception.details["node_id"], "node-b")
            self.assertEqual(ctx.exception.details["field"], "lease_limit")

    def test_duplicate_artifact_key(self) -> None:
        def mutate(raw):
            stage = raw["pipeline"]["stages"][0]
            stage["artifacts"].append(dict(stage["artifacts"][0]))

        with tempfile.TemporaryDirectory() as td:
            path, _ = mutate_spec(Path(td), mutate)
            with self.assertRaises(SpecError) as ctx:
                parse_spec(__import__("json").loads(path.read_text()))
            self.assertEqual(ctx.exception.details["artifact_key"], "src:main")
            self.assertEqual(ctx.exception.details["field"], "artifact_key")

    def test_missing_reference_target(self) -> None:
        def mutate(raw):
            raw["pipeline"]["stages"][1]["artifacts"][0]["refs"].append("missing:artifact")

        with tempfile.TemporaryDirectory() as td:
            path, _ = mutate_spec(Path(td), mutate)
            with self.assertRaises(SpecError) as ctx:
                parse_spec(__import__("json").loads(path.read_text()))
            self.assertEqual(ctx.exception.details["artifact_key"], "app:bin")
            self.assertEqual(ctx.exception.details["missing_reference"], "missing:artifact")

    def test_reference_cycle(self) -> None:
        def mutate(raw):
            raw["pipeline"]["stages"][0]["artifacts"][0]["refs"] = ["test:report"]

        with tempfile.TemporaryDirectory() as td:
            path, _ = mutate_spec(Path(td), mutate)
            with self.assertRaises(SpecError) as ctx:
                parse_spec(__import__("json").loads(path.read_text()))
            self.assertEqual(ctx.exception.code, "E_SPEC")
            self.assertIn(ctx.exception.details["artifact_key"], {"src:main", "app:bin", "test:report"})
            self.assertEqual(ctx.exception.details["field"], "refs")

    def test_node_error_precedes_later_artifact_error(self) -> None:
        def mutate(raw):
            raw["nodes"][1]["port"] = raw["nodes"][0]["port"]
            stage = raw["pipeline"]["stages"][0]
            stage["artifacts"].append(dict(stage["artifacts"][0]))

        with tempfile.TemporaryDirectory() as td:
            path, _ = mutate_spec(Path(td), mutate)
            with self.assertRaises(SpecError) as ctx:
                parse_spec(__import__("json").loads(path.read_text()))
            self.assertEqual(ctx.exception.details["node_id"], "node-b")
            self.assertEqual(ctx.exception.details["field"], "port")


if __name__ == "__main__":
    unittest.main()
