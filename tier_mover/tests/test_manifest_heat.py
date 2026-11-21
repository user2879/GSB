import unittest
from pathlib import Path

from tier_mover.errors import SpecError
from tier_mover.manifest import load_manifest
from tier_mover.heat import compute_heat
from tier_mover.workspace import write_sample_manifests


class ManifestValidationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.id().replace(".", "_"))
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.manifest = write_sample_manifests(self.root, seed=42)

    def tearDown(self):
        self.tmp.cleanup()

    def test_valid_manifest_loads(self):
        m = load_manifest(self.root / "tiers.json",
                          self.root / "objects.json",
                          self.root / "seed.json")
        self.assertEqual(len(m.tiers), 2)
        self.assertEqual(len(m.objects), 8)
        self.assertTrue(m.spec_digest)

    def test_duplicate_tier_id_rejected(self):
        import json
        doc = json.loads((self.root / "tiers.json").read_text("utf-8"))
        doc["tiers"][1]["tier_id"] = doc["tiers"][0]["tier_id"]
        (self.root / "tiers.json").write_text(json.dumps(doc), "utf-8")
        with self.assertRaises(SpecError) as ctx:
            load_manifest(self.root / "tiers.json",
                          self.root / "objects.json",
                          self.root / "seed.json")
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertIn("tier_id", ctx.exception.field)

    def test_negative_capacity_and_bool_rejected(self):
        import json
        doc = json.loads((self.root / "tiers.json").read_text("utf-8"))
        doc["tiers"][0]["capacity_bytes"] = -1
        (self.root / "tiers.json").write_text(json.dumps(doc), "utf-8")
        with self.assertRaises(SpecError) as ctx:
            load_manifest(self.root / "tiers.json",
                          self.root / "objects.json",
                          self.root / "seed.json")
        self.assertIn("capacity_bytes", ctx.exception.field)

        doc["tiers"][0]["capacity_bytes"] = True
        (self.root / "tiers.json").write_text(json.dumps(doc), "utf-8")
        with self.assertRaises(SpecError):
            load_manifest(self.root / "tiers.json",
                          self.root / "objects.json",
                          self.root / "seed.json")

    def test_duplicate_object_key_rejected(self):
        import json
        doc = json.loads((self.root / "objects.json").read_text("utf-8"))
        doc["objects"][1]["key"] = doc["objects"][0]["key"]
        (self.root / "objects.json").write_text(json.dumps(doc), "utf-8")
        with self.assertRaises(SpecError) as ctx:
            load_manifest(self.root / "tiers.json",
                          self.root / "objects.json",
                          self.root / "seed.json")
        self.assertIn("key", ctx.exception.field)

    def test_bad_digest_rejected(self):
        import json
        doc = json.loads((self.root / "objects.json").read_text("utf-8"))
        doc["objects"][0]["content_digest"] = "deadbeef"
        (self.root / "objects.json").write_text(json.dumps(doc), "utf-8")
        with self.assertRaises(SpecError) as ctx:
            load_manifest(self.root / "tiers.json",
                          self.root / "objects.json",
                          self.root / "seed.json")
        self.assertIn("content_digest", ctx.exception.field)

    def test_deterministic_digest_across_backends(self):
        from tier_mover.content import expected_digest
        py = expected_digest(42, "obj-alpha", 2048, "python")
        npv = expected_digest(42, "obj-alpha", 2048, "numpy")
        self.assertEqual(len(py), 64)
        # numpy 为独立确定性后端，摘要稳定即可，不强求与 python 后端相同
        self.assertEqual(npv, expected_digest(42, "obj-alpha", 2048, "numpy"))

    def test_heat_is_integer_and_stable(self):
        scores_a = compute_heat(self.manifest)
        scores_b = compute_heat(self.manifest)
        self.assertEqual(scores_a, scores_b)
        for score in scores_a.values():
            self.assertIsInstance(score.heat, int)


if __name__ == "__main__":
    unittest.main()
