from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.helpers import clone_bundle, make_sample

from inpaint_pyramid.config import apply_command
from inpaint_pyramid.engine import IncrementalEngine, compare_engines
from inpaint_pyramid.errors import E_TILE_BUDGET_PENDING
from inpaint_pyramid.style import StyleMapper


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.bundle, cls.stream, _ = make_sample(Path(cls.temp.name) / "samples", budget=3)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_layer_scoped_edit_does_not_invalidate_lower_layer(self) -> None:
        engine = IncrementalEngine(clone_bundle(self.bundle, budget=999))
        engine.bootstrap()
        lower_before = engine.layer_outputs[0].copy()
        lower_keys_before = {
            key: engine.tile_meta[key].cache_key
            for key in engine.valid_tiles
            if key[0] == 0
        }
        entry = engine.process_command(
            {"id": "layer-one-iterations", "type": "set_iterations", "layer": 1, "value": 6},
            0,
        )
        invalidated = [tuple(key) for key in entry["schedule"]["invalidated_tiles"]]
        self.assertTrue(invalidated)
        self.assertTrue(all(key[0] >= 1 for key in invalidated))
        self.assertTrue(np.array_equal(lower_before, engine.layer_outputs[0]))
        lower_keys_after = {
            key: engine.tile_meta[key].cache_key
            for key in engine.valid_tiles
            if key[0] == 0
        }
        self.assertEqual(lower_keys_before, lower_keys_after)
        self.assertEqual(entry["schedule"]["status"], "OK")

    def test_mask_edit_invalidates_only_intersecting_tiles_and_upper_refs(self) -> None:
        engine = IncrementalEngine(clone_bundle(self.bundle, budget=999))
        engine.bootstrap()
        entry = engine.process_command(self.stream.commands[0], 0)
        invalidated = [tuple(key) for key in entry["schedule"]["invalidated_tiles"]]
        self.assertTrue(invalidated)
        self.assertTrue(all(key[0] >= 1 for key in invalidated))
        self.assertEqual(entry["schedule"]["status"], "OK")
        self.assertEqual(len(invalidated), 3)

    def test_replacing_mask_invalidates_removed_old_region(self) -> None:
        engine = IncrementalEngine(clone_bundle(self.bundle, budget=999))
        engine.bootstrap()
        entry = engine.process_command(
            {
                "id": "remove-old-mask-region",
                "type": "set_mask_rect",
                "layer": 1,
                "rect": [0, 0, 2, 2],
            },
            0,
        )
        invalidated = {tuple(key) for key in entry["schedule"]["invalidated_tiles"]}
        self.assertTrue(
            {
                (1, 0, 0),
                (1, 0, 1),
            }.issubset(invalidated)
        )

    def test_quantization_changes_lookup_entries(self) -> None:
        mapper = StyleMapper(
            self.bundle.style_rgb,
            self.bundle.spec.fraction_bits,
            self.bundle.style_fingerprint,
        )
        lut4, fp4 = mapper.lookup_table(4)
        lut8, fp8 = mapper.lookup_table(8)
        self.assertNotEqual(fp4, fp8)
        self.assertFalse(np.array_equal(lut4, lut8))
        self.assertLessEqual(len(np.unique(lut4, axis=0)), 4)
        self.assertLessEqual(len(np.unique(lut8, axis=0)), 8)

        height = width = self.bundle.spec.tile_size
        repaired = np.tile(
            np.arange(256, dtype=np.int64).reshape(16, 16, 1), (1, 1, 3)
        ) << self.bundle.spec.fraction_bits
        self.assertEqual(repaired.shape, (height, width, 3))
        mask = np.zeros((height, width), dtype=bool)
        weight = self.bundle.spec.scale // 2
        weights = (weight, weight, weight)
        mapped4 = mapper.map_patch(
            repaired, mask, lut4, fp4, weights, "mapped", "mapped", mask
        ).image
        mapped8 = mapper.map_patch(
            repaired, mask, lut8, fp8, weights, "mapped", "mapped", mask
        ).image
        self.assertFalse(np.array_equal(mapped4, mapped8))

    def test_every_command_matches_full_rebuild_when_budget_is_sufficient(self) -> None:
        bundle = clone_bundle(self.bundle, budget=999)
        incremental = IncrementalEngine(bundle)
        incremental.commands_hash = self.stream.commands_hash
        incremental.bootstrap()
        full_mutable = bundle.new_mutable()
        for index, command in enumerate(self.stream.commands):
            incremental.process_command(command, index)
            try:
                apply_command(full_mutable, command, bundle.spec)
            except Exception:
                pass
            self.assertFalse(incremental.dirty_tiles)
            full = IncrementalEngine.build_full(
                bundle,
                mutable=full_mutable.clone(),
                cursor=index,
                commands_hash=self.stream.commands_hash,
            )
            comparison = compare_engines(incremental, full)
            self.assertTrue(comparison["equal"], comparison["differences"])

    def test_budget_pending_and_stable_ledger_then_drain_matches_full(self) -> None:
        bundle = clone_bundle(self.bundle, budget=3)
        incremental = IncrementalEngine(bundle)
        incremental.bootstrap()
        full_mutable = bundle.new_mutable()
        for index in range(2):
            command = self.stream.commands[index]
            entry = incremental.process_command(command, index)
            try:
                apply_command(full_mutable, command, bundle.spec)
            except Exception:
                pass
        self.assertEqual(entry["schedule"]["status"], E_TILE_BUDGET_PENDING)
        budget = entry["schedule"]["budget"]
        self.assertEqual(budget["consumed"], 3)
        self.assertEqual(
            [(item["remaining_before"], item["remaining_after"]) for item in budget["ledger"]],
            [(3, 2), (2, 1), (1, 0)],
        )
        self.assertTrue(entry["schedule"]["pending_tiles"])
        outcomes = incremental.drain_pending()
        self.assertTrue(outcomes)
        self.assertFalse(incremental.dirty_tiles)
        full = IncrementalEngine.build_full(
            bundle, mutable=full_mutable.clone(), cursor=1, commands_hash=self.stream.commands_hash
        )
        self.assertTrue(compare_engines(incremental, full)["equal"])

    def test_thread_count_does_not_change_order_budget_or_pixels(self) -> None:
        single = IncrementalEngine(clone_bundle(self.bundle, budget=2, threads=1))
        multi = IncrementalEngine(clone_bundle(self.bundle, budget=2, threads=4))
        single.bootstrap()
        multi.bootstrap()
        entry_single = single.process_command(self.stream.commands[1], 1)
        entry_multi = multi.process_command(self.stream.commands[1], 1)
        self.assertEqual(entry_single["schedule"]["budget"]["ledger"], entry_multi["schedule"]["budget"]["ledger"])
        self.assertEqual(entry_single["schedule"]["recomputed_tiles"], entry_multi["schedule"]["recomputed_tiles"])
        self.assertEqual(entry_single["schedule"]["pending_tiles"], entry_multi["schedule"]["pending_tiles"])
        single.drain_pending()
        multi.drain_pending()
        self.assertTrue(compare_engines(single, multi)["equal"])
        self.assertLessEqual(multi.concurrency_peak, 4)


if __name__ == "__main__":
    unittest.main()
