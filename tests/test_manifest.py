from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from replica_repair.errors import FormatError
from replica_repair.manifest_store import ManifestStore
from replica_repair.model import CURRENT_MANIFEST_VERSION, Manifest
from replica_repair.sample import init_sample, sample_manifest_path


class ManifestFormatTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        init_sample(self.root)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_all_three_versions_decode_and_round_trip(self) -> None:
        for group_id, version in (("interop-v1", 1), ("interop-v2", 2), ("interop-v3", 3)):
            path = sample_manifest_path(self.root, group_id)
            manifest = Manifest.from_file(path)
            self.assertEqual(manifest.format_version, version)
            self.assertEqual(manifest.to_dict(version), json.loads(path.read_text()))
            self.assertEqual(Manifest.from_dict(manifest.to_dict(version)), manifest)
        upgraded = ManifestStore(self.root).load_current(sample_manifest_path(self.root, "interop-v1")).manifest
        self.assertEqual(upgraded.format_version, CURRENT_MANIFEST_VERSION)
        self.assertTrue(all(block.digest for node in upgraded.nodes for block in node.blocks))

    def test_v1_only_encodes_identifier_and_size(self) -> None:
        manifest = Manifest.from_file(sample_manifest_path(self.root, "interop-v1"))
        payload = manifest.to_dict(1)
        for node in payload["nodes"]:
            for block in node["blocks"]:
                self.assertEqual(set(block), {"block_id", "size_bytes"})

    def test_unknown_version_is_format_error_with_path(self) -> None:
        payload = json.loads(sample_manifest_path(self.root, "interop-v3").read_text())
        payload["format_version"] = 99
        with self.assertRaises(FormatError) as caught:
            Manifest.from_dict(payload)
        self.assertEqual(caught.exception.code, "E_FORMAT")
        self.assertEqual(caught.exception.path, "$.format_version")

    def test_missing_required_field_is_rejected(self) -> None:
        payload = json.loads(sample_manifest_path(self.root, "interop-v2").read_text())
        del payload["nodes"][0]["blocks"][0]["digest"]
        with self.assertRaises(FormatError) as caught:
            Manifest.from_dict(payload)
        self.assertEqual(caught.exception.code, "E_FORMAT")
        self.assertTrue(caught.exception.path.endswith(".digest"))

    def test_invalid_digest_is_rejected(self) -> None:
        payload = json.loads(sample_manifest_path(self.root, "interop-v2").read_text())
        payload["nodes"][0]["blocks"][0]["digest"] = "sha256:not-hex"
        with self.assertRaises(FormatError) as caught:
            Manifest.from_dict(payload)
        self.assertEqual(caught.exception.path, "$.nodes[0].blocks[0].digest")

    def test_non_positive_size_is_rejected(self) -> None:
        payload = json.loads(sample_manifest_path(self.root, "interop-v1").read_text())
        payload["nodes"][0]["blocks"][0]["size_bytes"] = 0
        with self.assertRaises(FormatError) as caught:
            Manifest.from_dict(payload)
        self.assertEqual(caught.exception.path, "$.nodes[0].blocks[0].size_bytes")

    def test_unknown_field_and_bad_derivation_are_rejected(self) -> None:
        payload = json.loads(sample_manifest_path(self.root, "interop-v3").read_text())
        bad = copy.deepcopy(payload)
        bad["nodes"][0]["blocks"][0]["surprise"] = True
        with self.assertRaises(FormatError) as caught:
            Manifest.from_dict(bad)
        self.assertEqual(caught.exception.code, "E_FORMAT")
        self.assertIn("surprise", caught.exception.path)

        bad = copy.deepcopy(payload)
        for node in bad["nodes"]:
            for block in node["blocks"]:
                if block["checksum_derivation"] is not None:
                    block["checksum_derivation"]["digest"] = "sha256:" + "0" * 64
        with self.assertRaises(FormatError) as caught:
            Manifest.from_dict(bad)
        self.assertEqual(caught.exception.code, "E_FORMAT")
        self.assertTrue(caught.exception.path.endswith(".checksum_derivation.digest"))


if __name__ == "__main__":
    unittest.main()
