from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from matgen.cache import CacheStore
from matgen.commands import apply_command
from matgen.engine import MaterialEngine
from matgen.injection import InjectionRegistry
from matgen.loader import load_input
from matgen.mixing import mix_channel
from matgen.models import Material, MaterialLayer
from matgen.trace import compare_full_rebuild

from tests.helpers import BASIC, layer_artifact


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.loaded = load_input(BASIC)

    def test_deterministic_build_matches_blank_rebuild(self) -> None:
        cache = CacheStore()
        engine = MaterialEngine(cache, InjectionRegistry())
        first = engine.build(self.loaded.config, self.loaded.materials, self.loaded.textures)
        second = engine.build(self.loaded.config, self.loaded.materials, self.loaded.textures)
        self.assertEqual(first.final_digest, second.final_digest)
        self.assertEqual(second.recomputed, [])
        self.assertTrue(second.final_reused)
        comparison = compare_full_rebuild(
            incremental=first,
            config=self.loaded.config,
            materials=self.loaded.materials,
            textures=self.loaded.textures,
        )
        self.assertTrue(comparison["exact_match"])

    def test_layer_frequency_invalidation_reuses_only_upstream_layers(self) -> None:
        cache = CacheStore()
        engine = MaterialEngine(cache, InjectionRegistry())
        initial = engine.build(self.loaded.config, self.loaded.materials, self.loaded.textures)
        config, materials, affected, _, issues = apply_command(
            self.loaded.config, self.loaded.materials, self.loaded.commands[0]
        )
        self.assertEqual(issues, [])
        self.assertEqual(affected, {1})
        updated = engine.build(config, materials, self.loaded.textures)
        statuses = {
            entry["layer_index"]: entry["status"]
            for entry in updated.ledger_entries
            if entry["kind"] == "noise_layer"
        }
        self.assertEqual(statuses[0], "reused")
        self.assertEqual(statuses[1], "recomputed")
        self.assertEqual(statuses[2], "recomputed")
        self.assertEqual(statuses[3], "recomputed")
        self.assertEqual(initial.layers[0].key, updated.layers[0].key)

    def test_weight_change_invalidates_only_one_channel(self) -> None:
        cache = CacheStore()
        engine = MaterialEngine(cache, InjectionRegistry())
        engine.build(self.loaded.config, self.loaded.materials, self.loaded.textures)
        config, materials, _, _, issues = apply_command(
            self.loaded.config, self.loaded.materials, self.loaded.commands[1]
        )
        self.assertEqual(issues, [])
        updated = engine.build(config, materials, self.loaded.textures)
        self.assertEqual(updated.affected_layers, [])
        channel_status = {
            (entry["material_id"], entry["channel"]): entry["status"]
            for entry in updated.ledger_entries
            if entry["kind"] == "material_channel"
        }
        self.assertEqual(channel_status[("brushed_steel", "roughness")], "recomputed")
        self.assertEqual(channel_status[("brushed_steel", "diffuse")], "reused")
        self.assertEqual(channel_status[("painted_plastic", "roughness")], "reused")

    def test_texture_replacement_invalidates_downstream_chain(self) -> None:
        cache = CacheStore()
        engine = MaterialEngine(cache, InjectionRegistry())
        engine.build(self.loaded.config, self.loaded.materials, self.loaded.textures)
        texture = self.loaded.textures["textures/base_01.png"]
        changed_luma = texture.luma.copy()
        changed_luma[:, :] = (changed_luma.astype(np.int16) + 1).clip(0, 255).astype(np.uint8)
        replacement = replace(
            texture,
            luma=changed_luma,
            content_hash="replacement-content-hash",
            cache_key="replacement-cache-key",
        )
        textures = dict(self.loaded.textures)
        textures[texture.name] = replacement
        updated = engine.build(self.loaded.config, self.loaded.materials, textures)
        self.assertEqual(updated.affected_layers, [0, 1, 2, 3])

    def test_equal_weights_and_half_boundary_are_flagged(self) -> None:
        zeros = np.zeros((2, 2), dtype=np.int64)
        ones = np.ones((2, 2), dtype=np.int64)
        layers = {0: layer_artifact(0, zeros), 1: layer_artifact(1, ones)}
        material = Material(
            material_id="tie",
            label="tie",
            layers=(
                MaterialLayer(layer_index=0, weights={c: 1.0 for c in ("diffuse", "roughness", "metallic", "normal_perturbation")}),
                MaterialLayer(layer_index=1, weights={c: 1.0 for c in ("diffuse", "roughness", "metallic", "normal_perturbation")}),
            ),
        )
        artifact = mix_channel(
            material=material,
            channel="diffuse",
            layers=layers,
            key="tie-key",
            scale=256,
            weight_fraction_bits=12,
            mix_mode="weighted",
            rounding="half_even",
            max_boundary_samples=100,
        )
        kinds = {sample["kind"] for sample in artifact.boundary_samples}
        self.assertIn("equal_weight_tiebreak", kinds)
        self.assertIn("rounding_half_boundary", kinds)
        self.assertEqual(set(artifact.contributions), {0, 1} | set())
        tie = next(sample for sample in artifact.boundary_samples if sample["kind"] == "equal_weight_tiebreak")
        self.assertEqual(tie["chosen_layer"], 0)


if __name__ == "__main__":
    unittest.main()
