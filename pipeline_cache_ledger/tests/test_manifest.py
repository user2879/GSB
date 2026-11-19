import json
import tempfile
import unittest
from pathlib import Path

from pcl.errors import SpecError
from pcl.samples import write_sample_workspace


class ManifestValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.manifest_path, _ = write_sample_workspace(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_valid_manifest_loads(self):
        from pcl.manifest import load_manifest

        manifest = load_manifest(self.manifest_path, self.root)
        self.assertEqual(len(manifest.artifacts), 7)
        self.assertEqual(
            manifest.artifacts["codegen"].refs[0].key, "base-lib"
        )

    def test_duplicate_stage_id_rejected(self):
        doc = json.loads(self.manifest_path.read_text())
        doc["stages"][1]["id"] = doc["stages"][0]["id"]
        self.manifest_path.write_text(json.dumps(doc))
        self._assert_path_contains("stages[1].id")

    def test_bad_key_and_version_rejected(self):
        doc = json.loads(self.manifest_path.read_text())
        doc["stages"][0]["artifacts"][0]["key"] = "BAD KEY"
        doc["stages"][0]["artifacts"][1]["version"] = -2
        self.manifest_path.write_text(json.dumps(doc))
        self._assert_path_contains("stages[0].artifacts[0].key")
        self._assert_path_contains("stages[0].artifacts[1].version")

    def test_unsupported_format_version(self):
        doc = json.loads(self.manifest_path.read_text())
        doc["format_version"] = 99
        self.manifest_path.write_text(json.dumps(doc))
        self._assert_path_contains("format_version")

    def test_missing_reference_target(self):
        doc = json.loads(self.manifest_path.read_text())
        doc["stages"][1]["artifacts"][0]["refs"][0]["key"] = "ghost"
        self.manifest_path.write_text(json.dumps(doc))
        self._assert_path_contains(
            "stages[1].artifacts[0].refs[0].key"
        )

    def test_reference_version_not_comparable(self):
        doc = json.loads(self.manifest_path.read_text())
        doc["stages"][1]["artifacts"][0]["refs"][0]["version"] = 999
        self.manifest_path.write_text(json.dumps(doc))
        self._assert_path_contains(
            "stages[1].artifacts[0].refs[0].version"
        )

    def test_reference_digest_mismatch(self):
        doc = json.loads(self.manifest_path.read_text())
        doc["stages"][1]["artifacts"][0]["refs"][0]["digest"] = "sha256:deadbeef"
        self.manifest_path.write_text(json.dumps(doc))
        self._assert_path_contains(
            "stages[1].artifacts[0].refs[0].digest"
        )

    def test_path_escape_rejected(self):
        doc = json.loads(self.manifest_path.read_text())
        doc["stages"][0]["artifacts"][0]["file"] = "../../../etc/passwd"
        self.manifest_path.write_text(json.dumps(doc))
        self._assert_path_contains("stages[0].artifacts[0].file")

    def test_missing_file_rejected(self):
        doc = json.loads(self.manifest_path.read_text())
        doc["stages"][0]["artifacts"][0]["file"] = "artifacts/nope.txt"
        self.manifest_path.write_text(json.dumps(doc))
        self._assert_path_contains("stages[0].artifacts[0].file")

    def test_all_or_nothing_no_partial_load(self):
        doc = json.loads(self.manifest_path.read_text())
        doc["stages"][0]["artifacts"][0]["key"] = "bad key"
        doc["stages"][2]["artifacts"][0]["version"] = "x"
        self.manifest_path.write_text(json.dumps(doc))
        with self.assertRaises(SpecError) as ctx:
            from pcl.manifest import load_manifest

            load_manifest(self.manifest_path, self.root)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertGreaterEqual(len(ctx.exception.details["paths"]), 2)

    def _assert_path_contains(self, expected):
        with self.assertRaises(SpecError) as ctx:
            from pcl.manifest import load_manifest

            load_manifest(self.manifest_path, self.root)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertIn(expected, ctx.exception.details["paths"])


if __name__ == "__main__":
    unittest.main()
