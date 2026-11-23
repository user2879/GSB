from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from config_pipeline.errors import E_MANIFEST, PipelineError
from config_pipeline.manifest import load_manifest, manifest_from_dict
from tests.helpers import manifest


class ManifestValidationTests(unittest.TestCase):
    def assert_manifest_error(self, data: dict, field: str) -> PipelineError:
        with self.assertRaises(PipelineError) as caught:
            manifest_from_dict(data)
        self.assertEqual(caught.exception.code, E_MANIFEST)
        self.assertEqual(caught.exception.field, field)
        return caught.exception

    def test_duplicate_port_reports_first_duplicate_field(self) -> None:
        data = manifest(ports=(41001, 41002))
        data["clusters"][1]["port"] = data["clusters"][0]["port"]
        self.assert_manifest_error(data, "clusters[1].port")

    def test_non_loopback_literal_is_rejected(self) -> None:
        data = manifest()
        data["clusters"][0]["host"] = "localhost"
        self.assert_manifest_error(data, "clusters[0].host")

    def test_negative_shard_count_is_rejected(self) -> None:
        data = manifest()
        data["tasks"][0]["shards"] = -1
        self.assert_manifest_error(data, "tasks[0].shards")

    def test_boolean_is_not_an_integer(self) -> None:
        data = manifest()
        data["tasks"][0]["resource_units"] = True
        self.assert_manifest_error(data, "tasks[0].resource_units")

    def test_parent_must_exist(self) -> None:
        data = manifest()
        data["revisions"].append(
            {"version": 2, "parent": 99, "values": {"base": 11}}
        )
        self.assert_manifest_error(data, "revisions[1].parent")

    def test_versions_must_be_strictly_increasing(self) -> None:
        data = manifest()
        data["revisions"].append(
            {"version": 1, "parent": None, "values": {"base": 11}}
        )
        self.assert_manifest_error(data, "revisions[1].version")

    def test_derived_cycle_is_rejected(self) -> None:
        data = manifest(config_keys=("a",))
        data["revisions"][0] = {
            "version": 1,
            "parent": None,
            "values": {},
            "derived": {
                "a": {"op": "sum", "inputs": ["b"]},
                "b": {"op": "sum", "inputs": ["a"]},
            },
        }
        with self.assertRaises(PipelineError) as caught:
            manifest_from_dict(data)
        self.assertEqual(caught.exception.code, E_MANIFEST)
        self.assertTrue(caught.exception.field.startswith("revisions[0].derived."))

    def test_unknown_target_and_latest_key_are_rejected(self) -> None:
        data = manifest()
        data["tasks"][0]["target_clusters"] = ["missing"]
        self.assert_manifest_error(data, "tasks[0].target_clusters[0]")

        data = manifest(config_keys=("missing",))
        self.assert_manifest_error(data, "tasks[0].config_keys[0]")

    def test_derived_values_inherit_and_recalculate(self) -> None:
        data = manifest()
        data["revisions"].append(
            {"version": 2, "parent": 1, "values": {"base": 15}}
        )
        parsed = manifest_from_dict(data)
        self.assertEqual(parsed.latest_revision.effective_values["scaled"], 15)
        self.assertEqual(parsed.latest_revision.parent, 1)

    def test_duplicate_json_object_key_is_rejected_before_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text('{"global_resource_budget":1,"clusters":[],"clusters":[]}', encoding="utf-8")
            with self.assertRaises(PipelineError) as caught:
                load_manifest(path)
            self.assertEqual(caught.exception.code, E_MANIFEST)
            self.assertEqual(caught.exception.field, "$")

    def test_reordered_normalised_lists_have_same_digest(self) -> None:
        data = manifest(ports=(41011, 41012))
        data["tasks"][0]["config_keys"] = ["scaled", "base"]
        data["tasks"][0]["target_clusters"] = ["c1", "c0"]
        reordered = json.loads(json.dumps(data))
        reordered["clusters"].reverse()
        reordered["clusters"][0]["nodes"] = list(reversed(reordered["clusters"][0]["nodes"]))
        reordered["tasks"][0]["config_keys"].reverse()
        reordered["tasks"][0]["target_clusters"].reverse()
        self.assertEqual(
            manifest_from_dict(data).manifest_digest,
            manifest_from_dict(reordered).manifest_digest,
        )


if __name__ == "__main__":
    unittest.main()
