import json
import unittest
from pathlib import Path

from cfgdist.config.manifest import load_manifest, load_registry, topo_order
from cfgdist.errors import PlatformError

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


class ManifestValidationTests(unittest.TestCase):
    def setUp(self):
        self.registry = load_registry(SAMPLES / "registry.json")

    def test_valid_manifest_loads(self):
        manifest = load_manifest(SAMPLES / "manifest.json", self.registry)
        self.assertEqual([v.version for v in manifest.versions], [1, 2, 3])

    def test_corrupt_file_is_e_manifest(self):
        tmp = Path("data/test-work")
        tmp.mkdir(parents=True, exist_ok=True)
        target = tmp / "corrupt.json"
        text = (SAMPLES / "manifest.json").read_text()
        target.write_text(text[: len(text) // 2])
        with self.assertRaises(PlatformError) as ctx:
            load_manifest(target, self.registry)
        self.assertEqual(ctx.exception.code, "E_MANIFEST")
        self.assertTrue(ctx.exception.details[0].source.endswith("corrupt.json"))

    def test_unknown_target_node_rejected(self):
        data = json.loads((SAMPLES / "manifest.json").read_text())
        data["versions"][0]["items"][0]["targets"] = ["ghost-node"]
        target = Path("data/test-work/unknown-node.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data))
        with self.assertRaises(PlatformError) as ctx:
            load_manifest(target, self.registry)
        self.assertEqual(ctx.exception.code, "E_MANIFEST")

    def test_version_must_increase(self):
        data = json.loads((SAMPLES / "manifest.json").read_text())
        data["versions"][1]["version"] = 1  # duplicate / non-increasing
        target = Path("data/test-work/bad-version.json")
        target.write_text(json.dumps(data))
        with self.assertRaises(PlatformError) as ctx:
            load_manifest(target, self.registry)
        self.assertEqual(ctx.exception.code, "E_MANIFEST")

    def test_cycle_is_e_cycle(self):
        data = json.loads((SAMPLES / "manifest.json").read_text())
        items = data["versions"][0]["items"]
        first, second = items[0]["key"], items[1]["key"]
        items[0].pop("value", None)
        items[0]["derivation"] = {"offset": 0, "coefficients": {second: 1}}
        items[1].pop("value", None)
        items[1]["derivation"] = {"offset": 0, "coefficients": {first: 1}}
        target = Path("data/test-work/cycle.json")
        target.write_text(json.dumps(data))
        with self.assertRaises(PlatformError) as ctx:
            load_manifest(target, self.registry)
        self.assertEqual(ctx.exception.code, "E_CYCLE")

    def test_topo_order_leaves_first(self):
        manifest = load_manifest(SAMPLES / "manifest.json", self.registry)
        order = topo_order(manifest.versions[0])
        self.assertLess(order.index("base_x"), order.index("derived_u"))
        self.assertLess(order.index("derived_u"), order.index("derived_w"))


if __name__ == "__main__":
    unittest.main()
