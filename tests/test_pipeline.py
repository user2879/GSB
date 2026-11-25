from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from fixed_render.config import RenderConfig
from fixed_render.errors import InputInvalid
from fixed_render.inputs import load_conditions
from fixed_render.lease import LeaseClient, LeaseServer
from fixed_render.samples import generate_all_samples
from fixed_render.verification import verify_sequence_equivalence


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        generate_all_samples(cls.root)
        cls.config = RenderConfig.load(cls.root / "configs" / "render.yaml")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_valid_samples_load(self) -> None:
        bundle = load_conditions(self.config)
        self.assertEqual(len(bundle.vocab.tokens), self.config.text_dim)
        self.assertEqual(bundle.image.shape, (64, 64, 3))
        self.assertEqual(bundle.motion.shape, (8, 8, 2))

    def test_invalid_prompt_rejects_field(self) -> None:
        with self.assertRaises(InputInvalid) as context:
            load_conditions(self.config.with_changes({"prompt_tokens": ("ghost_token",)}))
        self.assertEqual(context.exception.code, "E_INPUT_INVALID")
        self.assertIn("conditions.prompt_tokens", context.exception.fields)

    def test_tile_height_must_divide_frame(self) -> None:
        data = {**self.config.to_mapping(), "tile_height": 7}
        with self.assertRaises(InputInvalid) as context:
            RenderConfig.from_mapping(data, base_dir=self.root)
        self.assertIn("tile_height", context.exception.fields)

    def test_quant_changes_integers_and_pixels(self) -> None:
        from fixed_render.workbench import CommandSession

        workspace_a = self.root / "quant_a"
        workspace_b = self.root / "quant_b"
        with CommandSession(self.root, workspace_a, self.config) as session:
            first = session.render_current(ledger_name="q8")["manifest"]
            text_q8 = session.engine.snapshots.current()["nodes"]["encode:text"]["trace"]
        config_b = self.config.with_changes({"fixed_frac_bits": 12, "quant_mode": "floor"})
        with CommandSession(self.root, workspace_b, config_b) as session:
            second = session.render_current(ledger_name="q12")["manifest"]
        self.assertNotEqual(
            first["nodes"]["encode:text"]["cache_key"],
            second["nodes"]["encode:text"]["cache_key"],
        )
        self.assertNotEqual(first["frame_fingerprints"], second["frame_fingerprints"])

    def test_lease_expiry_really_releases_slot(self) -> None:
        with LeaseServer("127.0.0.1", 0, max_slots=1) as server:
            client = LeaseClient("127.0.0.1", server.port)
            lease = client.acquire(lease_key="unit", scopes=["x"], ttl_ms=40)
            self.assertEqual(client.snapshot()["used_slots"], 1)
            deadline = time.time() + 2
            while time.time() < deadline and client.snapshot()["used_slots"]:
                time.sleep(0.01)
            self.assertEqual(client.snapshot()["used_slots"], 0)
            with self.assertRaises(Exception) as context:
                client.renew(lease["lease_id"], lease["generation"], 100)
            self.assertEqual(context.exception.code, "E_LEASE_EXPIRED")

    def test_short_edit_sequence_matches_blank_full(self) -> None:
        commands = [
            {"op": "set_prompt", "tokens": ["entity_grid", "attr_green", "quality_soft"]},
            {"op": "set_image", "path": "samples/images/condition_b.png"},
            {"op": "set_weights", "weights": {"text": 2, "image": 3, "motion": 2}},
            {"op": "set_denoise_steps", "value": 3},
            {"op": "set_tile_height", "value": 16},
        ]
        result = verify_sequence_equivalence(
            self.root,
            self.config,
            commands,
            "unit_sequence",
            self.root / "verification",
        )
        self.assertTrue(result["ok"], json.dumps(result, indent=2)[:4000])
        self.assertEqual(result["pixel_frame_count"], 2)


if __name__ == "__main__":
    unittest.main()
