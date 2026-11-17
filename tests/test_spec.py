from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from cluster_diag.errors import DiagError, E_REV, E_SPEC
from cluster_diag.spec import load_spec

from .support import base_payload, free_ports, write_spec


class SpecificationValidationTests(unittest.TestCase):
    def test_valid_spec_is_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = write_spec(Path(temp) / "spec.json", base_payload())
            spec = load_spec(path)
            self.assertEqual(spec.cluster_ids, ("edge-a", "edge-b"))
            self.assertEqual([revision.version for revision in spec.revisions], [1, 2, 3])

    def test_invalid_entries_reject_whole_spec_with_paths(self) -> None:
        payload = base_payload()
        payload["clusters"][1]["id"] = payload["clusters"][0]["id"]
        payload["clusters"][1]["port"] = payload["clusters"][0]["port"]
        payload["tasks"][1]["config_keys"].append("missing.key")
        payload["tasks"][0]["targets"] = []
        with tempfile.TemporaryDirectory() as temp:
            path = write_spec(Path(temp) / "spec.json", payload)
            with self.assertRaises(DiagError) as raised:
                load_spec(path)
        self.assertEqual(raised.exception.code, E_SPEC)
        paths = {issue.field_path for issue in raised.exception.issues}
        self.assertTrue(any("clusters[1].id" in path for path in paths))
        self.assertTrue(any("clusters[1].port" in path for path in paths))
        self.assertTrue(any("config.missing.key" in path for path in paths))
        self.assertTrue(any("targets" in path for path in paths))

    def test_non_loopback_and_missing_parent_are_rejected(self) -> None:
        payload = base_payload()
        payload["clusters"][0]["host"] = "192.0.2.10"
        payload["revisions"][2]["parent"] = 99
        with tempfile.TemporaryDirectory() as temp:
            path = write_spec(Path(temp) / "spec.json", payload)
            with self.assertRaises(DiagError) as raised:
                load_spec(path)
        self.assertEqual(raised.exception.code, E_SPEC)
        rendered = str(raised.exception.as_dict())
        self.assertIn("host", rendered)
        self.assertIn("parent", rendered)

    def test_cycle_is_e_spec_normally_and_e_rev_when_injected(self) -> None:
        payload = base_payload()
        payload["revisions"][2]["parent"] = 3
        with tempfile.TemporaryDirectory() as temp:
            path = write_spec(Path(temp) / "spec.json", payload)
            with self.assertRaises(DiagError) as normal:
                load_spec(path)
            with self.assertRaises(DiagError) as injected:
                load_spec(path, cycle_code=E_REV, inject_cycle=True)
        self.assertEqual(normal.exception.code, E_SPEC)
        self.assertEqual(injected.exception.code, E_REV)

    def test_overlapping_tasks_in_one_revision_are_rejected(self) -> None:
        payload = base_payload()
        payload["tasks"][0]["targets"] = ["edge-a", "edge-b"]
        payload["tasks"][1]["targets"] = ["edge-a"]
        payload["revisions"][0]["task_ids"] = ["core-network", "edge-cache"]
        with tempfile.TemporaryDirectory() as temp:
            path = write_spec(Path(temp) / "spec.json", payload)
            with self.assertRaises(DiagError) as raised:
                load_spec(path)
        self.assertEqual(raised.exception.code, E_SPEC)
        self.assertTrue(any("multiple tasks" in issue.message for issue in raised.exception.issues))

    def test_non_finite_json_numbers_are_rejected(self) -> None:
        payload = base_payload()
        payload["revisions"][0]["config"]["net.timeout_ms"] = float("nan")
        with tempfile.TemporaryDirectory() as temp:
            path = write_spec(Path(temp) / "spec.json", payload)
            with self.assertRaises(DiagError) as raised:
                load_spec(path)
        self.assertEqual(raised.exception.code, E_SPEC)


if __name__ == "__main__":
    unittest.main()
