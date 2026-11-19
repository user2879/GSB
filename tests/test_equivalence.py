"""Requirement 7: incremental edit-sequence renders equal a clean full run."""

from __future__ import annotations

import copy
import dataclasses
import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from dbench.config import load_config
from dbench.orchestrator import Orchestrator
from dbench.reporting import compare_runs
from dbench.store import SnapshotStore

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load(name: str):
    path = os.path.join(ROOT, "configs", name)
    cfg = load_config(path)
    base = os.path.dirname(path)
    return dataclasses.replace(
        cfg,
        vocab_file=os.path.normpath(os.path.join(base, cfg.vocab_file)),
        initial_image=(os.path.normpath(os.path.join(base, cfg.initial_image))
                       if cfg.initial_image else None),
    )


def _run(cfg, tmp, name):
    store = SnapshotStore(os.path.join(tmp, name))
    png = os.path.join(tmp, name + ".png")
    report = Orchestrator(cfg, store, output_png=png).run()
    arrays = {
        item["stage"]: store.load_array(item["stage"], item["version"])
        for item in report["stages"]
    }
    with open(png, "rb") as handle:
        png_bytes = handle.read()
    return report, arrays, png_bytes


def _edit_sequence(base_cfg):
    """Yield (label, mutated config) covering every editable input."""
    cfg = dataclasses.replace(base_cfg)
    yield "start", cfg

    cfg = dataclasses.replace(cfg, prompt="neon city night star blue")
    yield "prompt", cfg

    cfg = dataclasses.replace(cfg, seed=cfg.seed + 99)
    yield "seed", cfg

    cfg = dataclasses.replace(cfg, guide_scale=cfg.guide_scale + 1.25)
    yield "guide", cfg

    table = [list(row) for row in cfg.noise_table]
    table[0] = [v + 0.1 for v in table[0]]
    cfg = dataclasses.replace(cfg,
                              noise_table=tuple(tuple(r) for r in table))
    yield "noise", cfg

    # Switch into image-to-image with replacement steps.
    i2i = _load("valid_img2img.json")
    cfg = dataclasses.replace(cfg, initial_image=i2i.initial_image,
                              image_replacement_steps=3)
    yield "img2img", cfg

    # Change the replacement count.
    cfg = dataclasses.replace(cfg, image_replacement_steps=2)
    yield "replace_steps", cfg

    # Add a local latent patch at step 4.
    patch_cfg = _load("valid_patch.json")
    cfg = dataclasses.replace(cfg, patches=patch_cfg.patches)
    yield "patch", cfg

    # Change step count (grow), keeping noise table consistent.
    extra = tuple(tuple(0.5 for _ in cfg.noise_table[0])
                  for _ in range(2))
    cfg = dataclasses.replace(
        cfg, steps=cfg.steps + 2,
        noise_table=cfg.noise_table + extra,
        patches=tuple(),
    )
    yield "steps", cfg


class EquivalenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="dbench-eq-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_every_edit_matches_clean_full_run_byte_for_byte(self):
        base = _load("valid_base.json")
        incremental_store = SnapshotStore(
            os.path.join(self.tmp, "incremental")
        )
        last_label = None
        for label, cfg in _edit_sequence(base):
            last_label = label
            png_inc = os.path.join(self.tmp, f"inc_{label}.png")
            report_inc = Orchestrator(
                cfg, incremental_store, output_png=png_inc
            ).run()
            # Clean full run from an empty workspace with identical final input.
            clean_dir = os.path.join(self.tmp, f"full_{label}")
            report_full, arrays_full, png_full = _run(cfg, self.tmp,
                                                      f"full_{label}")
            arrays_inc = {
                item["stage"]: incremental_store.load_array(
                    item["stage"], item["version"]
                )
                for item in report_inc["stages"]
            }
            # Every shared intermediate latent matches elementwise.
            for stage, arr_full in arrays_full.items():
                self.assertIn(stage, arrays_inc, f"{label}: missing {stage}")
                np.testing.assert_array_equal(
                    arrays_inc[stage], arr_full,
                    err_msg=f"{label}: latent mismatch at {stage}",
                )
            with open(png_inc, "rb") as handle:
                png_inc_bytes = handle.read()
            self.assertEqual(png_inc_bytes, png_full,
                             f"{label}: PNG bytes differ")
            comparison = compare_runs(report_inc, report_full,
                                      "incremental", "full")
            self.assertTrue(comparison["equivalent"],
                            f"{label}: {comparison['first_mismatch_stage']}")
        self.assertEqual(last_label, "steps")


if __name__ == "__main__":
    unittest.main()
