"""Requirements 1, 2, 4: cache invalidation and precise patch propagation."""

from __future__ import annotations

import dataclasses
import os
import shutil
import tempfile
import unittest

import numpy as np

from dbench.config import load_config
from dbench.dag import denoise_stage
from dbench.orchestrator import Orchestrator
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


def _run(cfg, tmp, name, injections=()):
    store = SnapshotStore(os.path.join(tmp, name))
    orch = Orchestrator(cfg, store, injections=injections,
                        output_png=os.path.join(tmp, name + ".png"))
    return store, orch.run()


class CachingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="dbench-cache-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_second_run_reuses_everything(self):
        cfg = _load("valid_base.json")
        _, report1 = _run(cfg, self.tmp, "a")
        _, report2 = _run(cfg, self.tmp, "a")
        self.assertEqual(report1["recompute"], report2["reuse"])
        self.assertEqual(report2["recompute"], [])

    def test_seed_change_only_invalidates_init_and_downstream(self):
        cfg = _load("valid_base.json")
        _, report1 = _run(cfg, self.tmp, "seed")
        cfg2 = dataclasses.replace(cfg, seed=cfg.seed + 7)
        _, report2 = _run(cfg2, self.tmp, "seed")
        recompute = set(report2["recompute"])
        reuse = set(report2["reuse"])
        self.assertIn("encode_prompt", reuse)          # upstream stays hit
        self.assertIn("init_latent", recompute)
        self.assertIn(denoise_stage(1), recompute)
        self.assertIn("decode", recompute)

    def test_patch_only_invalidates_that_step_and_downstream(self):
        cfg = _load("valid_patch.json")
        _, report1 = _run(cfg, self.tmp, "patch")
        # Remove the patch -> everything upstream of step 4 must stay hit.
        cfg_no_patch = dataclasses.replace(cfg, patches=tuple())
        _, report2 = _run(cfg_no_patch, self.tmp, "patch")
        reuse = set(report2["reuse"])
        recompute = set(report2["recompute"])
        self.assertIn("encode_prompt", reuse)
        self.assertIn("init_latent", reuse)
        self.assertIn(denoise_stage(1), reuse)
        self.assertIn(denoise_stage(3), reuse)
        self.assertIn(denoise_stage(4), recompute)
        self.assertIn(denoise_stage(5), recompute)
        self.assertIn("decode", recompute)

    def test_versions_are_sequential_and_manifest_complete(self):
        cfg = _load("valid_base.json")
        store, _ = _run(cfg, self.tmp, "ver")
        cfg2 = dataclasses.replace(cfg, seed=cfg.seed + 1)
        _run(cfg2, self.tmp, "ver")
        versions = store.list_versions("init_latent")
        self.assertEqual(versions, [1, 2])
        manifest = store.read_manifest("init_latent", 2)
        for key in ("stage", "version", "content_fingerprint",
                    "dependency_fingerprints", "array_summary"):
            self.assertIn(key, manifest)

    def test_snapshot_gap_is_rejected(self):
        from dbench.errors import E_SNAPSHOT_VERSION, WorkbenchError

        cfg = _load("valid_base.json")
        store, _ = _run(cfg, self.tmp, "gap")
        cfg2 = dataclasses.replace(cfg, seed=cfg.seed + 1)
        _run(cfg2, self.tmp, "gap")  # creates v0002
        # Remove a committed middle version, keeping v0002 -> real gap.
        os.remove(store._manifest_path("init_latent", 1))
        os.remove(store._data_path("init_latent", 1))
        with self.assertRaises(WorkbenchError) as ctx:
            _run(cfg2, self.tmp, "gap")
        self.assertEqual(ctx.exception.code, E_SNAPSHOT_VERSION)

    def test_orphan_data_from_forced_kill_is_reconciled(self):
        cfg = _load("valid_base.json")
        store, report = _run(cfg, self.tmp, "orphan")
        orphan = store._data_path("encode_prompt", 2)
        with open(orphan, "wb") as handle:
            handle.write(b"interrupted")
        # A subsequent run cleans the orphan and reuses committed artifacts.
        _, report2 = _run(cfg, self.tmp, "orphan")
        self.assertFalse(os.path.exists(orphan))
        self.assertIn("encode_prompt", report2["reuse"])


if __name__ == "__main__":
    unittest.main()
