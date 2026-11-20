import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from texmat_pipeline.models import CHANNELS, load_generation_config, load_materials
from texmat_pipeline.pipeline import TexturePipeline

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "tools" / "generate_samples.py"


def snapshot_state(result) -> dict:
    """Field-level state used to prove incremental == full rebuild."""
    state = {
        "noise": [
            (entry["index"], entry["fingerprint"], entry["image"].tolist())
            for entry in result.noise_result["layers"]
        ],
        "materials": [],
        "composite": dict(result.composite_keys),
    }
    for mat in result.material_result["materials"]:
        channels = {}
        for channel in CHANNELS:
            info = mat["channels"][channel]
            channels[channel] = (info["fingerprint"], info["image"].tolist(),
                                 info["boundary"].tolist())
        state["materials"].append((mat["material_id"], channels))
    return state


class IncrementalConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="texmat-incr-"))
        cls.sample = cls.tmp / "sample"
        subprocess.run(
            [sys.executable, str(GENERATOR), "--out", str(cls.sample),
             "--tiers", "16", "--layers", "4", "--point-budget", "4096"],
            check=True, capture_output=True,
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def setUp(self):
        import itertools
        self._copy_seq = itertools.count()

    def _load(self, sample_dir: Path):
        config, _ = load_generation_config(sample_dir / "generation_config.json")
        materials, _ = load_materials(sample_dir / "materials.json")
        return config, materials

    def _copy_sample(self, name: str) -> Path:
        target = self.tmp / f"{name}_{next(self._copy_seq)}"
        shutil.copytree(self.sample, target)
        return target

    def _assert_states_equal(self, incremental, fresh):
        a = snapshot_state(incremental)
        b = snapshot_state(fresh)
        self.assertEqual([x[0] for x in a["noise"]], [x[0] for x in b["noise"]])
        for (ia, fp_a, arr_a), (ib, fp_b, arr_b) in zip(a["noise"], b["noise"]):
            self.assertEqual(ia, ib)
            self.assertEqual(fp_a, fp_b)
            self.assertEqual(arr_a, arr_b)
        self.assertEqual([m[0] for m in a["materials"]], [m[0] for m in b["materials"]])
        for (mid, ca), (_, cb) in zip(a["materials"], b["materials"]):
            self.assertEqual(set(ca), set(cb))
            for channel in ca:
                self.assertEqual(ca[channel][0], cb[channel][0])
                self.assertEqual(ca[channel][1], cb[channel][1])
                self.assertEqual(ca[channel][2], cb[channel][2])
        self.assertEqual(a["composite"], b["composite"])

    def test_edit_sequence_matches_fresh_full_rebuild(self):
        # One long-lived working directory receives the full edit sequence.
        live_sample = self._copy_sample("live")
        config, materials = self._load(live_sample)
        work = self.tmp / "live_work"
        pipeline = TexturePipeline(live_sample, work, config, materials)
        first = pipeline.run(16)
        self.assertEqual(len(first.recompute_keys), len(first.reuse_keys) + len(first.recompute_keys))
        self.assertEqual(first.reuse_keys, [])

        edits = [
            ("texture_replace", self._edit_texture),
            ("layer_frequency", self._edit_layer_frequency),
            ("blend_mode", self._edit_blend_mode),
            ("frac_bits", self._edit_frac_bits),
            ("round_mode", self._edit_round_mode),
        ]
        previous_result = first
        for name, edit in edits:
            with self.subTest(edit=name):
                edit(live_sample)
                config2, materials2 = self._load(live_sample)
                pipeline = TexturePipeline(live_sample, work, config2, materials2)
                incremental = pipeline.run(16)

                # Independent fresh workspace: full rebuild from the edited
                # inputs (a copy of the live sample), never the original sample.
                fresh_sample = self.tmp / f"fresh_{name}_{next(self._copy_seq)}"
                shutil.copytree(live_sample, fresh_sample)
                fresh_work = self.tmp / f"fresh_work_{name}_{next(self._copy_seq)}"
                fresh_pipe = TexturePipeline(fresh_sample, fresh_work, config2, materials2)
                fresh = fresh_pipe.run(16, force_rebuild=True)

                self._assert_states_equal(incremental, fresh)
                self.assertGreater(len(incremental.reuse_keys), 0,
                                   "unaffected artifacts must be reused")
                # Reuse+recompute partition matches a fresh-build derivation.
                self.assertEqual(
                    sorted(incremental.reuse_keys + incremental.recompute_keys),
                    sorted(fresh.recompute_keys),
                )
                previous_result = incremental

    def test_texture_edit_invalidates_only_downstream_subgraph(self):
        live_sample = self._copy_sample("scope")
        config, materials = self._load(live_sample)
        work = self.tmp / "scope_work"
        pipeline = TexturePipeline(live_sample, work, config, materials)
        pipeline.run(16)
        # Replace stone texture; layer 0 and layer 2 reference stone (i % 3).
        self._edit_texture(live_sample)
        config2, _ = self._load(live_sample)
        pipe2 = TexturePipeline(live_sample, work, config2, materials)
        result = pipe2.run(16)
        affected_layers = result.affected_scope["noise_layers"]
        self.assertTrue(affected_layers, "texture replacement must invalidate noise layers")
        # Base texture node itself is in the base scope.
        self.assertTrue(result.affected_scope["base_textures"])
        # All downstream channels/composites eventually reference an affected layer.
        self.assertTrue(result.affected_scope["composites"])
        # Unaffected layer keys remain reusable.
        self.assertTrue(result.reuse_keys)

    # ------------------------------------------------------------------ #
    # Deterministic edit helpers
    # ------------------------------------------------------------------ #
    def _edit_texture(self, sample_dir: Path) -> None:
        target = sample_dir / "stone_16.png"
        img = Image.open(target)
        arr = np.asarray(img, dtype=np.uint8).copy()
        # Deterministic perturbation driven by pixel coordinates only.
        yy, xx = np.mgrid[0: arr.shape[0], 0: arr.shape[1]]
        delta = ((xx * 7 + yy * 13) % 11).astype(np.uint8)
        Image.fromarray(np.clip(arr.astype(np.uint16) + delta, 0, 255).astype(np.uint8), mode=img.mode).save(
            target, format="PNG", optimize=True
        )

    def _edit_layer_frequency(self, sample_dir: Path) -> None:
        path = sample_dir / "generation_config.json"
        cfg = json.loads(path.read_text())
        cfg["layers"][1]["frequency"] = round(cfg["layers"][1]["frequency"] + 2.5, 4)
        cfg["layers"][1]["amplitude"] = round(cfg["layers"][1]["amplitude"] * 0.75, 4)
        path.write_text(json.dumps(cfg, indent=2))

    def _edit_blend_mode(self, sample_dir: Path) -> None:
        path = sample_dir / "generation_config.json"
        cfg = json.loads(path.read_text())
        cfg["blend_mode"] = "weighted_sum"
        path.write_text(json.dumps(cfg, indent=2))

    def _edit_frac_bits(self, sample_dir: Path) -> None:
        path = sample_dir / "generation_config.json"
        cfg = json.loads(path.read_text())
        cfg["fixed_point"]["frac_bits"] = cfg["fixed_point"]["frac_bits"] + 2
        path.write_text(json.dumps(cfg, indent=2))

    def _edit_round_mode(self, sample_dir: Path) -> None:
        path = sample_dir / "generation_config.json"
        cfg = json.loads(path.read_text())
        cfg["fixed_point"]["round_mode"] = "half_up"
        path.write_text(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    unittest.main()
