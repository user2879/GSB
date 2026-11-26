from __future__ import annotations

import unittest

from renderdiag.codec import encode_image, encode_motion, encode_text, fuse_conditions
from renderdiag.resources import load_image, load_motion_table, load_prompt_set
from tests.support import SampleWorkspace


class DeterminismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sample = SampleWorkspace()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.sample.close()

    def test_encoding_and_fusion_repeat_field_for_field(self) -> None:
        config = self.sample.config
        prompts = load_prompt_set(config.resources.prompts, config)
        image = load_image(config.resources.image, config)
        motion = load_motion_table(config.resources.motion, config)
        text_a = encode_text(prompts.prompts["p0"], config)
        text_b = encode_text(prompts.prompts["p0"], config)
        image_a = encode_image(image, config)
        image_b = encode_image(image, config)
        motion_a = encode_motion(motion, 0, config)
        motion_b = encode_motion(motion, 0, config)
        self.assertEqual(text_a, text_b)
        self.assertEqual(image_a, image_b)
        self.assertEqual(motion_a, motion_b)
        self.assertEqual(len(text_a["vector"]), config.condition_dim)
        fusion_a = fuse_conditions(0, text_a, image_a, motion_a, config.weights_dict(), config)
        fusion_b = fuse_conditions(0, text_b, image_b, motion_b, config.weights_dict(), config)
        self.assertEqual(fusion_a, fusion_b)
        self.assertEqual(
            [(item["quantization_step"], item["quantized_integer"], item["residual"]) for item in fusion_a["ledger"]],
            [(item["quantization_step"], item["quantized_integer"], item["residual"]) for item in fusion_b["ledger"]],
        )

    def test_guidance_seed_and_step_count_change_real_integer_outputs(self) -> None:
        engine = self.sample.engine("determinism")
        self.assertEqual(engine.drain_current()["code"], "OK")
        baseline = engine.results()
        baseline_steps = [item["summary"] for item in baseline["denoise_steps"]]
        baseline_frames = [item["content_fingerprint"] for item in baseline["frames"]]

        response = engine.run_command({"op": "set_guidance", "guidance": 1.75})
        self.assertEqual(response["code"], "OK")
        guidance_steps = [item["summary"] for item in engine.results()["denoise_steps"]]
        self.assertEqual(len(guidance_steps), len(baseline_steps))
        self.assertTrue(all(a != b for a, b in zip(guidance_steps, baseline_steps)))

        response = engine.run_command({"op": "set_seed", "seed": 20261225})
        self.assertEqual(response["code"], "OK")
        seeded = engine.results()
        self.assertNotEqual(
            [item["summary"] for item in seeded["denoise_steps"]],
            guidance_steps,
        )

        response = engine.run_command({"op": "set_denoise_steps", "steps": 6})
        self.assertEqual(response["code"], "OK")
        stepped = engine.results()
        self.assertEqual(len(stepped["denoise_steps"]), engine.config.frame_count * 6)
        self.assertNotEqual(
            [item["summary"] for item in stepped["denoise_steps"]],
            [item["summary"] for item in seeded["denoise_steps"]],
        )
        self.assertNotEqual(
            [item["content_fingerprint"] for item in stepped["frames"]],
            baseline_frames,
        )

    def test_two_cold_renders_are_identical(self) -> None:
        first = self.sample.engine("cold-a")
        second = self.sample.engine("cold-b")
        self.assertEqual(first.drain_current()["code"], "OK")
        self.assertEqual(second.drain_current()["code"], "OK")
        self.assertEqual(first.results()["frames"], second.results()["frames"])
        self.assertEqual(first.results()["denoise_steps"], second.results()["denoise_steps"])
        self.assertEqual(first.results()["fusion"], second.results()["fusion"])


if __name__ == "__main__":
    unittest.main()
