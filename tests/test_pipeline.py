from __future__ import annotations

import json
import random
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from condrender.acceptance import (
    run_injection_acceptance,
    run_negative_validation,
    run_preview_acceptance,
    run_sigkill_recovery,
    run_tile_cursor_kill_recovery,
)
from condrender.bundle import initial_state, load_bundle
from condrender.engine import RenderEngine, decode_tile_slice, tile_bounds
from condrender.hashing import array_fingerprint
from condrender.samples import generate_samples
from condrender.session import InteractiveSession, field_report, render_final_state


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix="condrender-test-")
        cls.root = Path(cls.temp.name)
        generate_samples(cls.root)
        cls.bundle, cls.failures = load_bundle(cls.root)
        assert not cls.failures, [f.to_dict() for f in cls.failures]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_01_valid_bundle_and_negative_business_codes(self) -> None:
        self.assertEqual(self.failures, [])
        result = run_negative_validation(self.root, self.root / "negative-report")
        self.assertTrue(result["ok"], result["case_results"])

    def test_02_text_hits_and_deterministic_encoding(self) -> None:
        from condrender.conditions import encode_text
        state = initial_state(self.bundle)
        first = encode_text(0, state)
        second = encode_text(0, state)
        self.assertEqual(first, second)
        self.assertEqual(first["dim"], 14)
        terms = {hit["term"] for hit in first["hit_positions"]}
        self.assertEqual(terms, {"red", "bright", "cube", "grid"})
        self.assertTrue(all("char_start" in hit for hit in first["hit_positions"]))

    def test_03_quant_and_guidance_change_pixels(self) -> None:
        state = initial_state(self.bundle)
        base_engine = RenderEngine(state, workers=1)
        base = base_engine.render_all()
        base_pixels = base_engine.cache[
            base_engine.frame_results[0]["keys"]["frame_compose"]
        ]["data"]["rgb"]
        base_latent = base_engine.cache[
            base_engine.frame_results[0]["keys"]["denoise_frame"]
        ]["data"]["latent"]

        changed = initial_state(self.bundle)
        changed["config"]["fixed_point"]["quant_profile"] = "Q8_FAST"
        changed["config"]["fixed_point"]["frac_bits"] = 8
        q_engine = RenderEngine(changed, workers=1)
        q_result = q_engine.render_all()
        q_pixels = q_engine.cache[
            q_engine.frame_results[0]["keys"]["frame_compose"]
        ]["data"]["rgb"]
        q_latent = q_engine.cache[
            q_engine.frame_results[0]["keys"]["denoise_frame"]
        ]["data"]["latent"]
        self.assertNotEqual(base_pixels, q_pixels)
        self.assertFalse(np.array_equal(
            np.asarray(base_latent), np.asarray(q_latent)
        ))

        guided = initial_state(self.bundle)
        guided["config"]["guidance_scale"] = "3.0"
        g_engine = RenderEngine(guided, workers=1)
        g_engine.render_all()
        g_pixels = g_engine.cache[
            g_engine.frame_results[0]["keys"]["frame_compose"]
        ]["data"]["rgb"]
        self.assertNotEqual(base_pixels, g_pixels)
        g_latent = g_engine.cache[
            g_engine.frame_results[0]["keys"]["denoise_frame"]
        ]["data"]["latent"]
        self.assertFalse(np.array_equal(
            np.asarray(base_latent), np.asarray(g_latent)
        ))
        self.assertEqual(base["status"], "complete")
        self.assertEqual(q_result["status"], "complete")

        more_steps = initial_state(self.bundle)
        more_steps["config"]["denoise_steps"] = 6
        step_engine = RenderEngine(more_steps, workers=1)
        step_engine.render_all()
        step_pixels = step_engine.cache[
            step_engine.frame_results[0]["keys"]["frame_compose"]
        ]["data"]["rgb"]
        self.assertNotEqual(base_pixels, step_pixels)
        step_latent = step_engine.cache[
            step_engine.frame_results[0]["keys"]["denoise_frame"]
        ]["data"]["latent"]
        self.assertFalse(np.array_equal(
            np.asarray(base_latent), np.asarray(step_latent)
        ))

    def test_04_tile_slice_matches_full_frame_and_outside_does_not_change_key(self) -> None:
        state = initial_state(self.bundle)
        state["config"]["tile_rows"] = 4
        engine = RenderEngine(state, workers=2)
        result = engine.render_all()
        self.assertEqual(result["status"], "complete")
        frame_result = engine.frame_results[0]
        compose = engine.cache[frame_result["keys"]["frame_compose"]]["data"]
        full_rgb = np.asarray(compose["rgb"], dtype=np.uint8)
        latent_key = frame_result["keys"]["denoise_frame"]
        latent = np.asarray(engine.cache[latent_key]["data"]["latent"], dtype=np.int64)
        bounds = tile_bounds(8, 4)

        for index, (y0, y1) in enumerate(bounds):
            direct = decode_tile_slice(latent[y0:y1, :, :].copy(), state["config"])
            expected = full_rgb[y0:y1, :, :]
            np.testing.assert_array_equal(np.asarray(direct["rgb"], dtype=np.uint8), expected)

        y0, y1 = bounds[0]
        mutated = latent.copy()
        mutated[y1 + 1, :, :] += 12345
        self.assertEqual(
            array_fingerprint(latent[y0:y1]), array_fingerprint(mutated[y0:y1])
        )
        upstream = {
            "denoise_frame_key": latent_key,
            "slice_fp": array_fingerprint(latent[y0:y1]),
            "decoder_fp": engine.make_key("x", {}, {}) and __import__(
                "condrender.hashing", fromlist=["full_fingerprint"]
            ).full_fingerprint(state["config"]["decoder"]),
        }
        key = engine.make_key("tile_decode", upstream, {
            "tile_index": 0, "y0": y0, "y1": y1, "width": 16, "tile_rows": 4,
        })
        self.assertEqual(key, frame_result["tile_keys"][0])

    def test_05_targeted_invalidation(self) -> None:
        session = InteractiveSession(self.bundle, workers=4)
        session.finish()
        report = session.handle({
            "id": "T", "op": "replace_prompt", "frame": 0,
            "text": "green sphere dark smooth",
        })
        recompute_frames = {
            entry["frame"] for entry in report["entries"]
            if entry["status"] == "RECOMPUTE"
        }
        hit_frames = {entry["frame"] for entry in report["entries"] if entry["status"] == "HIT"}
        self.assertEqual(recompute_frames, {0})
        self.assertIn(1, hit_frames)
        self.assertIn(2, hit_frames)
        self.assertIn(3, hit_frames)

        session2 = InteractiveSession(self.bundle, workers=4)
        session2.finish()
        session2.handle({"id": "steps5", "op": "set_denoise_steps", "steps": 5})
        step_entries = [
            e for e in session2.command_reports[-1]["entries"] if e["stage"] == "denoise_step"
        ]
        hit_indices = {e["index"] for e in step_entries if e["status"] == "HIT"}
        recompute_indices = {e["index"] for e in step_entries if e["status"] == "RECOMPUTE"}
        self.assertEqual(hit_indices, {0, 1, 2})
        self.assertEqual(recompute_indices, {3, 4})

        report3 = session2.handle({"id": "tiles4", "op": "set_tile_rows", "tile_rows": 4})
        recompute_stages = {
            e["stage"] for e in report3["entries"] if e["status"] == "RECOMPUTE"
        }
        self.assertEqual(recompute_stages, {"tile_decode", "frame_compose"})

    def test_06_command_stream_matches_cold_full(self) -> None:
        commands = self.bundle["commands"]["commands"]
        for workers in (1, 4):
            session = InteractiveSession(self.bundle, workers=workers)
            session.finish()
            session.run_stream(commands, budget_ms=None)
            result = session.finish()
            incremental = field_report(session.last_engine, result)
            engine, cold_result, _ = render_final_state(
                self.bundle, commands, workers=workers
            )
            cold = field_report(engine, cold_result)
            self.assertEqual(incremental["pixels"], cold["pixels"])
            self.assertEqual(incremental["denoise_residuals"], cold["denoise_residuals"])
            self.assertEqual(incremental["tile_completion_order"], cold["tile_completion_order"])
            self.assertEqual(incremental["lineage"], cold["lineage"])

    def test_07_random_command_sequences(self) -> None:
        rng = random.Random(2901126)
        prompts = [
            "red cube bright grid", "blue sphere dark terrain",
            "green cylinder rough", "pyramid striped smooth red",
            "blue grid terrain dark", "green sphere bright smooth",
        ]
        profiles = ["Q8_FAST", "Q12_BALANCED", "Q16_DIAG"]
        for sequence_index in range(6):
            commands = []
            for command_index in range(rng.randint(5, 9)):
                op = rng.choice([
                    "prompt", "image", "delete_image", "motion", "weights",
                    "steps", "tiles", "guidance", "quant",
                ])
                command_id = f"R{sequence_index}-{command_index}"
                if op == "prompt":
                    command = {"id": command_id, "op": "replace_prompt",
                               "frame": rng.randrange(4), "text": rng.choice(prompts)}
                elif op == "image":
                    command = {"id": command_id, "op": "replace_image",
                               "frame": rng.randrange(4),
                               "image_id": rng.choice(["condition_a", "condition_b", "condition_c"])}
                elif op == "delete_image":
                    command = {"id": command_id, "op": "delete_image",
                               "frame": rng.randrange(4)}
                elif op == "motion":
                    command = {"id": command_id, "op": "adjust_motion",
                               "frame": rng.randrange(4), "row": rng.randrange(2),
                               "col": rng.randrange(4),
                               "component": rng.choice(["dx", "dy"]),
                               "value": rng.randint(-6, 6)}
                elif op == "weights":
                    command = {"id": command_id, "op": "set_weights", "weights": {
                        "text": f"{rng.randint(2, 7) / 10:.1f}",
                        "image": f"{rng.randint(1, 4) / 10:.1f}",
                        "motion": f"{rng.randint(1, 4) / 10:.1f}",
                    }}
                elif op == "steps":
                    command = {"id": command_id, "op": "set_denoise_steps",
                               "steps": rng.randint(1, 7)}
                elif op == "tiles":
                    command = {"id": command_id, "op": "set_tile_rows",
                               "tile_rows": rng.randint(1, 8)}
                elif op == "guidance":
                    command = {"id": command_id, "op": "set_guidance",
                               "guidance_scale": f"{rng.randint(5, 30) / 10:.1f}"}
                else:
                    profile = rng.choice(profiles)
                    command = {"id": command_id, "op": "set_quant_profile",
                               "quant_profile": profile}
                commands.append(command)

            workers = rng.choice([1, 2, 4])
            session = InteractiveSession(self.bundle, workers=workers)
            session.finish()
            session.run_stream(commands, budget_ms=None)
            result = session.finish()
            incremental = field_report(session.last_engine, result)
            engine, cold_result, _ = render_final_state(
                self.bundle, commands, workers=workers
            )
            cold = field_report(engine, cold_result)
            self.assertEqual(incremental["pixels"], cold["pixels"], sequence_index)
            self.assertEqual(incremental["denoise_residuals"], cold["denoise_residuals"])
            self.assertEqual(incremental["tile_completion_order"], cold["tile_completion_order"])

    def test_08_concurrency_cap(self) -> None:
        state = initial_state(self.bundle)
        engine = RenderEngine(state, workers=99, task_delay_ms=8)
        engine.render_all()
        self.assertLessEqual(engine.workers, int(state["config"]["max_workers"]))
        self.assertLessEqual(
            engine.max_observed_concurrency, int(state["config"]["max_workers"])
        )
        self.assertGreaterEqual(engine.max_observed_concurrency, 2)

    def test_09_injection_isolation(self) -> None:
        result = run_injection_acceptance(self.root, self.root / "injection-report")
        self.assertTrue(result["ok"], json.dumps(result, ensure_ascii=False)[:4000])

    def test_10_stale_preview(self) -> None:
        result = run_preview_acceptance(self.root, self.root / "preview-report")
        self.assertTrue(result["ok"], result)

    def test_11_sigkill_snapshot_recovery(self) -> None:
        result = run_sigkill_recovery(self.root, self.root / "kill-report")
        self.assertTrue(result["ok"], json.dumps(result, ensure_ascii=False, indent=2))

    def test_12_partial_tile_cursor_sigkill_recovery(self) -> None:
        result = run_tile_cursor_kill_recovery(self.root, self.root / "tile-kill-report")
        self.assertTrue(result["ok"], json.dumps(result, ensure_ascii=False, indent=2))

    def test_13_snapshot_version_and_config_hash_decisions(self) -> None:
        from condrender.snapshot import (
            SnapshotStore,
            decide_restore,
            restore_payload_into_engine,
        )

        state = initial_state(self.bundle)
        engine = RenderEngine(state, workers=1)
        engine.render_all()
        store = SnapshotStore(self.root / "decision-snapshot")
        payload = store.commit(engine)
        loaded = store.load_current()
        self.assertEqual(decide_restore(loaded, state)["decision"], "resume_pending")

        changed = initial_state(self.bundle)
        changed["config"]["seed"] = int(state["config"]["seed"]) + 1
        self.assertEqual(
            decide_restore(payload, changed)["decision"],
            "incremental_config_changed",
        )

        incompatible = dict(loaded)
        incompatible["snapshot_version"] = 999
        fresh = RenderEngine(initial_state(self.bundle), workers=1)
        decision = restore_payload_into_engine(fresh, incompatible)
        self.assertEqual(decision["decision"], "full_rebuild_snapshot_version")
        self.assertEqual(fresh.cache, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
