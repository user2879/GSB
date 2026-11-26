import filecmp
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fxrender.app import build_engine, load_state
from fxrender.config import RenderConfig
from fxrender.pipeline import (PipelineState, RenderEngine, apply_command,
                               render_all, render_incremental)
from fxrender.samplegen import generate_sample_bundle


def fresh_bundle(work: Path, frames: int = 6,
                 cfg: RenderConfig | None = None) -> Path:
    return generate_sample_bundle(work / "samples", cfg=cfg,
                                  frame_count=frames)


def frame_digests(engine: RenderEngine) -> dict:
    result = {}
    for entry in engine.cache.entries.values():
        if entry.stage == "frame" and entry.status in ("computed", "reused"):
            payload = engine.cache.load_artifact(entry.key)
            result[entry.frame_id] = {
                "pixel_digest": payload["pixel_digest"],
                "tile_digests": payload["tile_digests"],
                "latent_fp": payload["latent_fp"],
            }
    return dict(sorted(result.items()))


def canonical_ledger(engine: RenderEngine) -> list:
    return engine.cache.canonical_ledger()


class SampleReproTests(unittest.TestCase):
    def test_samples_byte_reproducible(self):
        work_a = Path(tempfile.mkdtemp())
        work_b = Path(tempfile.mkdtemp())
        generate_sample_bundle(work_a / "bundle", frame_count=4)
        generate_sample_bundle(work_b / "bundle", frame_count=4)
        cmp = filecmp.dircmp(work_a / "bundle", work_b / "bundle")
        self.assertEqual(cmp.diff_files, [])
        self.assertEqual(sorted(cmp.common_files),
                         sorted(p.name for p in
                                (work_a / "bundle").iterdir()
                                if p.is_file()))


class IncrementalEquivalenceTests(unittest.TestCase):
    def setUp(self):
        self.work = Path(tempfile.mkdtemp())
        self.cfg = RenderConfig(max_workers=2)
        self.bundle = fresh_bundle(self.work, frames=6, cfg=self.cfg)

    def _full_render(self, state: PipelineState, cache_dir: Path,
                     workers: int) -> RenderEngine:
        engine = RenderEngine(state, cache_dir, max_workers=workers)
        render_all(engine)
        return engine

    def test_prompt_edit_only_invalidates_own_frame(self):
        engine, _ = build_engine(self.bundle, self.work / "cache_base",
                                 max_workers=1)
        render_all(engine)
        before = frame_digests(engine)
        command = {"op": "replace_prompt", "frame_id": 2,
                   "terms": engine.state.vocabulary[:2]}
        invalidated, result = render_incremental(engine, command)
        after = frame_digests(engine)
        affected_stages = {
            engine.cache.entries[k].stage for k in invalidated
            if k in engine.cache.entries}
        # Unaffected frames must keep identical pixel digests.
        for frame_id in before:
            if frame_id != 2:
                self.assertEqual(before[frame_id]["pixel_digest"],
                                 after[frame_id]["pixel_digest"])
        self.assertNotEqual(before[2]["pixel_digest"],
                            after[2]["pixel_digest"])
        self.assertTrue(any(
            item["reused"] for item in result.processed))

    def test_full_equals_incremental_after_command_sequence(self):
        commands = [
            {"op": "replace_prompt", "frame_id": 0,
             "terms": None},
            {"op": "set_weight", "condition": "image", "value": 0.42},
            {"op": "adjust_motion", "tile_row": 0, "tile_col": 0,
             "dx": 5, "dy": -3},
            {"op": "set_denoise_steps", "value": 7},
            {"op": "set_quant", "width": 16, "tier": "fine"},
            {"op": "set_exchange_version", "value": 1},
            {"op": "set_exchange_version", "value": 2},
        ]
        # Incremental path.
        inc_engine, _ = build_engine(self.bundle,
                                     self.work / "cache_inc", max_workers=2)
        for command in commands:
            command = dict(command)
            if command.get("terms") is None:
                command["terms"] = inc_engine.state.vocabulary[1:3]
            change = apply_command(inc_engine.state, command)
            self.assertNotIn("error", change)
            render_incremental(inc_engine, change)

        # Full from-scratch path built from the same final state.
        final_state = inc_engine.state.clone()
        full_engine = self._full_render(
            final_state, self.work / "cache_full", workers=1)
        self.assertEqual(frame_digests(full_engine),
                         frame_digests(inc_engine))
        # Current live cache keys must match coordinate-by-coordinate even
        # though the incremental store retains superseded history entries.
        version = inc_engine.cfg.exchange_version
        self.assertEqual(
            full_engine.cache.current_valid_view(version),
            inc_engine.cache.current_valid_view(version))

    def test_worker_count_does_not_change_pixels(self):
        base_state, _ = load_state(self.bundle, RenderConfig(max_workers=1))
        engines = {}
        for workers in (1, 2, 4):
            state = base_state.clone()
            engine = RenderEngine(state,
                                  self.work / f"cache_w{workers}",
                                  max_workers=workers)
            render_all(engine)
            engines[workers] = frame_digests(engine)
        self.assertEqual(engines[1], engines[2])
        self.assertEqual(engines[2], engines[4])

    def test_tile_rows_change_partition_not_pixels_frame_digest(self):
        # Tile partition changes while the fused latent stays equivalent.
        bundle_a = generate_sample_bundle(
            self.work / "samples_t4",
            cfg=RenderConfig(tile_rows=4), frame_count=3)
        bundle_b = generate_sample_bundle(
            self.work / "samples_t2",
            cfg=RenderConfig(tile_rows=2), frame_count=3)
        state_a, _ = load_state(bundle_a, RenderConfig(tile_rows=4))
        engine_a = RenderEngine(state_a, self.work / "tiles_a",
                                max_workers=1)
        render_all(engine_a)
        state_b, _ = load_state(bundle_b, RenderConfig(tile_rows=2))
        engine_b = RenderEngine(state_b, self.work / "tiles_b",
                                max_workers=1)
        render_all(engine_b)
        digests_a = frame_digests(engine_a)
        digests_b = frame_digests(engine_b)
        self.assertEqual(set(digests_a), set(digests_b))
        # The number of tile digests differs because the partition changed.
        self.assertNotEqual(
            digests_a[0]["tile_digests"], digests_b[0]["tile_digests"])

    def test_budget_ledger_matches_between_runs(self):
        state_a, _ = load_state(self.bundle)
        engine_a = RenderEngine(state_a, self.work / "budget_a",
                                max_workers=2)
        result_a = render_all(engine_a)
        state_b, _ = load_state(self.bundle)
        engine_b = RenderEngine(state_b, self.work / "budget_b",
                                max_workers=1)
        result_b = render_all(engine_b)
        self.assertEqual(result_a.consumed_budget, result_b.consumed_budget)
        keys_a = [tuple(d["key"]) for d in result_a.deductions]
        keys_b = [tuple(d["key"]) for d in result_b.deductions]
        self.assertEqual(keys_a, keys_b)


if __name__ == "__main__":
    unittest.main()
