import os
import unittest

from layerbench.errors import E_MANIFEST, ManifestError
from layerbench.manifest import load_manifest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "samples", "workspace")
MANIFESTS = os.path.join(ROOT, "samples", "manifests")


class ManifestValidationTests(unittest.TestCase):
    def test_valid_manifest_accepted(self):
        manifest = load_manifest(
            os.path.join(MANIFESTS, "sample_manifest.json"), workspace=SRC
        )
        self.assertEqual(manifest.topo_order, ["base", "builder", "packager"])
        self.assertEqual(manifest.global_slots, 2)
        self.assertEqual(len(manifest.all_instructions()), 10)

    def test_bad_manifest_rejected_with_all_problems(self):
        with self.assertRaises(ManifestError) as ctx:
            load_manifest(os.path.join(MANIFESTS, "bad_manifest.json"),
                          workspace=SRC)
        self.assertEqual(ctx.exception.code, E_MANIFEST)
        problems = ctx.exception.problems
        self.assertTrue(any("global_slots" in p for p in problems))
        self.assertTrue(any("阶段名重复" in p for p in problems))
        self.assertTrue(any("ghost" in p for p in problems))
        self.assertTrue(any("missing.txt" in p for p in problems))

    def test_path_escape_rejected(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            bad = os.path.join(tmp, "escape.json")
            with open(bad, "w", encoding="utf-8") as handle:
                json.dump({
                    "version": "v", "global_slots": 1,
                    "default_timeout_ms": 1000, "cache_capacity": 1,
                    "stages": [{"name": "s", "depends_on": [], "quota": 1,
                                "instructions": [
                                    {"id": "c", "type": "copy",
                                     "params": {"src": "../escape.txt",
                                                "dst": "/x"}}]}],
                }, handle)
            with self.assertRaises(ManifestError) as ctx:
                load_manifest(bad, workspace=SRC)
            self.assertEqual(ctx.exception.code, E_MANIFEST)


if __name__ == "__main__":
    unittest.main()
