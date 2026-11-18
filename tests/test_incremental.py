import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.generate_samples import write_bundle
from texmat.config import load_generation_config, load_materials
from texmat.cache import build_graph, final_node_id, layer_node_id, texture_node_id
from texmat.loader import scan_textures
from texmat.testkit import (apply_edit, materialize_full_reference,
                            run_incremental, snapshot_arrays)


class InvalidationTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.sample = self.tmp / "sample"
        write_bundle(self.sample, size=18, workers=2, budget=100000)

    def tearDown(self):
        self._tmp.cleanup()

    def _graph(self):
        config = load_generation_config(self.sample / "generation_config.json")
        materials = load_materials(self.sample / "materials.json", config)
        textures, _ = scan_textures(self.sample, config.active_tier)
        graph = build_graph(config, materials, {n: t.content_fingerprint for n, t in textures.items()})
        return graph, config, materials, textures

    def test_texture_replacement_only_invalidates_downstream(self):
        graph, _, _, _ = self._graph()
        plan = graph.plan([texture_node_id("tex_a")])
        # layer00 and layer01 consume tex_a; layer02 consumes tex_b and must survive
        self.assertIn(layer_node_id(0), plan.recompute)
        self.assertIn(layer_node_id(1), plan.recompute)
        self.assertNotIn(layer_node_id(2), plan.recompute)
        self.assertIn(layer_node_id(2), plan.reuse)
        # channels/finals depending on affected layers are downstream
        self.assertTrue(any(node.startswith("channel:") for node in plan.recompute))
        self.assertIn(final_node_id("mat_stone"), plan.recompute)

    def test_layer_frequency_change_is_precise(self):
        graph, _, _, _ = self._graph()
        plan = graph.plan([layer_node_id(1)])
        self.assertIn(layer_node_id(1), plan.recompute)
        self.assertNotIn(layer_node_id(0), plan.recompute)
        self.assertIn(layer_node_id(0), plan.reuse)
        self.assertNotIn(layer_node_id(2), plan.recompute)

    def test_blend_and_bits_invalidate_channels_and_finals_only(self):
        graph, _, _, _ = self._graph()
        # Simulate a blend-mode change by rebuilding graph and diffing keys.
        before = {node_id: record.cache_key for node_id, record in graph.nodes.items()}
        config = load_generation_config(self.sample / "generation_config.json")
        raw = json.loads((self.sample / "generation_config.json").read_text())
        raw["blend_mode"] = "additive"
        (self.sample / "generation_config.json").write_text(json.dumps(raw, sort_keys=True))
        graph2, _, _, _ = self._graph()
        after = {node_id: record.cache_key for node_id, record in graph2.nodes.items()}
        changed = {node_id for node_id in before if before[node_id] != after[node_id]}
        self.assertTrue(all(node.startswith(("channel:", "final:")) for node in changed))
        self.assertFalse(any(node.startswith(("layer:", "texture:")) for node in changed))


class IncrementalConsistencyTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _copy_sample(self, name: str) -> Path:
        source = self.tmp / "source"
        write_bundle(source, size=16, workers=3, budget=100000)
        target = self.tmp / name
        shutil.copytree(source, target)
        return target

    def test_edit_sequence_matches_from_scratch(self):
        sample = self._copy_sample("inc")
        snapshot = self.tmp / "snap_inc"
        edits = json.loads((sample / "edit_sequence.json").read_text())
        for edit in edits:
            apply_edit(sample, edit)
            run_incremental(sample, snapshot)
        incremental = snapshot_arrays(snapshot)

        full_snapshot = self.tmp / "snap_full"
        materialize_full_reference(sample, full_snapshot)
        full = snapshot_arrays(full_snapshot)
        self.assertEqual(set(incremental), set(full))
        for key in incremental:
            np.testing.assert_array_equal(incremental[key], full[key], err_msg=key)

    def test_blend_and_bits_combinations(self):
        for blend in ("weighted", "additive", "multiply"):
            for bits in (6, 10):
                sample = self._copy_sample(f"combo_{blend}_{bits}")
                raw = json.loads((sample / "generation_config.json").read_text())
                raw["blend_mode"] = blend
                raw["fractional_bits"] = bits
                (sample / "generation_config.json").write_text(json.dumps(raw, sort_keys=True))

                inc_snap = self.tmp / f"inc_{blend}_{bits}"
                run_incremental(sample, inc_snap)
                full_snap = self.tmp / f"full_{blend}_{bits}"
                materialize_full_reference(sample, full_snap)
                inc_arrays = snapshot_arrays(inc_snap)
                full_arrays = snapshot_arrays(full_snap)
                self.assertEqual(set(inc_arrays), set(full_arrays))
                for key in inc_arrays:
                    np.testing.assert_array_equal(inc_arrays[key], full_arrays[key],
                                                  err_msg=f"{blend}/{bits}/{key}")

    def test_reused_nodes_are_not_recomputed(self):
        sample = self._copy_sample("reuse")
        snapshot = self.tmp / "snap_reuse"
        first = run_incremental(sample, snapshot)
        self.assertEqual(first["result"].reused, [])
        second = run_incremental(sample, snapshot)
        self.assertEqual(second["result"].recomputed, [])
        self.assertTrue(second["result"].reused)


if __name__ == "__main__":
    unittest.main()
