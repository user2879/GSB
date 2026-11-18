from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from renderlab.config import load_workbench
from renderlab.errors import E_CONFIG_INVALID, E_IMG2IMG_STEP_INVALID, E_NOISE_TABLE_MISMATCH
from tests.helpers import write_case


class ConfigValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_valid_case_and_generated_table_lengths(self) -> None:
        result = load_workbench(write_case(self.root / "valid"))
        self.assertTrue(result.can_render)
        self.assertEqual(len(result.noise_table), 3)
        self.assertEqual(len(result.guidance_table), 3)
        self.assertEqual(result.failures, [])

    def test_unknown_config_key(self) -> None:
        def mutate(config: dict) -> None:
            config["surprise"] = True

        result = load_workbench(write_case(self.root / "unknown", mutate))
        self.assertFalse(result.can_render)
        self.assertIn(E_CONFIG_INVALID, {failure.code for failure in result.failures})
        self.assertIn("surprise", {failure.item for failure in result.failures})

    def test_cache_budget_zero(self) -> None:
        result = load_workbench(
            write_case(self.root / "budget", lambda config: config["render"].update(cache_budget_blocks=0))
        )
        self.assertFalse(result.can_render)
        self.assertEqual(result.failures[0].code, E_CONFIG_INVALID)
        self.assertEqual(result.failures[0].item, "render.cache_budget_blocks")

    def test_resolution_tiers_must_increase(self) -> None:
        def mutate(config: dict) -> None:
            config["render"]["resolution_tiers"] = [
                {"name": "large", "width": 16, "height": 16},
                {"name": "small", "width": 8, "height": 8},
            ]

        result = load_workbench(write_case(self.root / "tiers", mutate))
        self.assertFalse(result.can_render)
        self.assertIn(E_CONFIG_INVALID, {failure.code for failure in result.failures})
        self.assertIn("render.resolution_tiers", {failure.item for failure in result.failures})

    def test_noise_table_length_mismatch(self) -> None:
        def mutate(config: dict) -> None:
            config["render"]["noise_table"] = {"values": [0.1, 0.2], "expected_length": 3}

        result = load_workbench(write_case(self.root / "noise", mutate))
        self.assertFalse(result.can_render)
        self.assertIn(E_NOISE_TABLE_MISMATCH, {failure.code for failure in result.failures})

    def test_img2img_step_overflow(self) -> None:
        def mutate(config: dict) -> None:
            config["img2img"]["replace_early_steps"] = 4

        result = load_workbench(write_case(self.root / "imgsteps", mutate))
        self.assertFalse(result.can_render)
        self.assertIn(E_IMG2IMG_STEP_INVALID, {failure.code for failure in result.failures})

    def test_embedding_dimension_mismatch(self) -> None:
        def mutate(sample: dict) -> None:
            sample["embedding_dim"] = 5

        result = load_workbench(write_case(self.root / "dim", mutate_sample=mutate))
        self.assertFalse(result.can_render)
        self.assertIn(E_CONFIG_INVALID, {failure.code for failure in result.failures})
        self.assertIn("embedding_dim", {failure.item for failure in result.failures})

    def test_bad_seed_is_excluded_but_valid_seed_renders(self) -> None:
        def config_mutate(config: dict) -> None:
            config["render"]["seed_max"] = 100

        def sample_mutate(sample: dict) -> None:
            sample["seeds"] = [11, 999]

        result = load_workbench(
            write_case(self.root / "seed", config_mutate, sample_mutate)
        )
        self.assertTrue(result.can_render)
        self.assertEqual(result.valid_seeds, [11])
        self.assertTrue(any(failure.code == E_CONFIG_INVALID for failure in result.failures))

    def test_unknown_prompt_token_is_excluded(self) -> None:
        def mutate(sample: dict) -> None:
            sample["prompts"][1]["tokens"] = ["forest", "not-in-vocab"]

        result = load_workbench(write_case(self.root / "token", mutate_sample=mutate))
        self.assertTrue(result.can_render)
        self.assertEqual([prompt.prompt_id for prompt in result.valid_prompts], ["p0"])
        self.assertIn(E_CONFIG_INVALID, {failure.code for failure in result.failures})


if __name__ == "__main__":
    unittest.main()
