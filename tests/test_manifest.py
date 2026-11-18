import json
import tempfile
import unittest
from pathlib import Path

from tier_mover.errors import SpecError
from tier_mover.manifest import load_manifest
from tier_mover.samplegen import build_sample


def _write(workdir: Path, payload: dict) -> Path:
    path = workdir / "manifest_bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class ManifestValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.workdir = Path(self.tmp.name)
        build_sample(self.workdir)
        self.good = json.loads((self.workdir / "manifest.json").read_text())

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_valid_manifest_loads(self) -> None:
        manifest = load_manifest(self.workdir / "manifest.json")
        self.assertEqual(len(manifest.tiers), 2)
        self.assertEqual(len(manifest.objects), 8)

    def test_duplicate_tier_id_rejected(self) -> None:
        bad = json.loads(json.dumps(self.good))
        bad["tiers"][1]["tier_id"] = bad["tiers"][0]["tier_id"]
        with self.assertRaises(SpecError) as ctx:
            load_manifest(_write(self.workdir, bad))
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertTrue(any("tier_id" in i for i in ctx.exception.context["issues"]))

    def test_duplicate_object_key_rejected(self) -> None:
        bad = json.loads(json.dumps(self.good))
        bad["objects"][1]["key"] = bad["objects"][0]["key"]
        with self.assertRaises(SpecError) as ctx:
            load_manifest(_write(self.workdir, bad))
        self.assertTrue(any(".key" in i for i in ctx.exception.context["issues"]))

    def test_negative_capacity_and_zero_size_rejected(self) -> None:
        bad = json.loads(json.dumps(self.good))
        bad["tiers"][0]["capacity_bytes"] = -1
        bad["objects"][0]["size_bytes"] = 0
        with self.assertRaises(SpecError) as ctx:
            load_manifest(_write(self.workdir, bad))
        joined = " ".join(ctx.exception.context["issues"])
        self.assertIn("capacity_bytes", joined)
        self.assertIn("size_bytes", joined)

    def test_boolean_is_not_integer(self) -> None:
        bad = json.loads(json.dumps(self.good))
        bad["rule"]["batch_size"] = True
        with self.assertRaises(SpecError):
            load_manifest(_write(self.workdir, bad))

    def test_watermark_ordering_rejected(self) -> None:
        bad = json.loads(json.dumps(self.good))
        bad["rule"]["hot_watermark_bytes"] = bad["rule"]["cold_reclaim_bytes"] + 1
        with self.assertRaises(SpecError) as ctx:
            load_manifest(_write(self.workdir, bad))
        self.assertIn("hot_watermark_bytes", " ".join(ctx.exception.context["issues"]))

    def test_watermark_above_capacity_rejected(self) -> None:
        bad = json.loads(json.dumps(self.good))
        bad["rule"]["hot_watermark_bytes"] = bad["tiers"][0]["capacity_bytes"] + 1
        with self.assertRaises(SpecError):
            load_manifest(_write(self.workdir, bad))


if __name__ == "__main__":
    unittest.main()
