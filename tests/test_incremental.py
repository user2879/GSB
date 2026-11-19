import unittest
from pathlib import Path

from cfgdist.config.grid import Grid
from cfgdist.config.manifest import load_manifest, load_registry, ManifestVersion
from cfgdist.config.compiler import compile_version
from cfgdist.config.incremental import incremental_compile, derive_invalidation
from cfgdist.errors import PlatformError

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


class IncrementalTests(unittest.TestCase):
    def setUp(self):
        registry = load_registry(SAMPLES / "registry.json")
        self.manifest = load_manifest(SAMPLES / "manifest.json", registry)
        self.grid = Grid.from_q(2)

    def test_only_changed_subtree_invalidated_v2(self):
        v1, v2 = self.manifest.versions[0], self.manifest.versions[1]
        invalidated, steps = derive_invalidation(v1, v2)
        self.assertEqual(invalidated, {"base_x", "derived_u", "derived_w"})
        triggers = {(s.key, s.trigger) for s in steps}
        self.assertIn(("derived_w", "derivation-downstream"), triggers)

    def test_untouched_artifacts_byte_identical(self):
        v1, v2 = self.manifest.versions[0], self.manifest.versions[1]
        first = compile_version(v1, self.grid)
        compiled, invalidated, _, byte_check = incremental_compile(v1, v2, self.grid, first)
        self.assertTrue(all(byte_check.values()))
        for key, identical in byte_check.items():
            self.assertTrue(identical, f"{key} artifact changed despite no input change")

    def test_delete_referenced_item_is_e_ref(self):
        v1 = self.manifest.versions[0]
        kept = {k: item for k, item in v1.items.items() if k != "base_y"}
        new_order = tuple(k for k in v1.order if k != "base_y")
        v_new = ManifestVersion(version=9, strategy=v1.strategy, items=kept, order=new_order)
        with self.assertRaises(PlatformError) as ctx:
            derive_invalidation(v1, v_new)
        self.assertEqual(ctx.exception.code, "E_REF")
        self.assertIn("derived_u", ctx.exception.details[0].reason)


if __name__ == "__main__":
    unittest.main()
