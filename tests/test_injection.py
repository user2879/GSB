"""Requirement 6: real fault injection, unique ids and failure isolation."""

from __future__ import annotations

import dataclasses
import os
import shutil
import tempfile
import unittest

from dbench.config import load_config
from dbench.dag import denoise_stage
from dbench.errors import (
    E_IMAGE_SIZE_MISMATCH,
    E_INJECTED_PATCH_OUT_OF_RANGE,
    E_NON_FINITE_LATENT,
    E_NOISE_ENTRY_TRUNCATED,
)
from dbench.injection import load_injections
from dbench.orchestrator import Orchestrator
from dbench.store import SnapshotStore

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_cfg(name):
    path = os.path.join(ROOT, "configs", name)
    cfg = load_config(path)
    base = os.path.dirname(path)
    return dataclasses.replace(
        cfg,
        vocab_file=os.path.normpath(os.path.join(base, cfg.vocab_file)),
        initial_image=(os.path.normpath(os.path.join(base, cfg.initial_image))
                       if cfg.initial_image else None),
    )


def _resolve_inj(name):
    return load_injections(os.path.join(ROOT, "samples", "injections", name))


class InjectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="dbench-inj-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, cfg_name, inj_name, dirname):
        cfg = _resolve_cfg(cfg_name)
        injections = _resolve_inj(inj_name)
        store = SnapshotStore(os.path.join(self.tmp, dirname))
        report = Orchestrator(
            cfg, store, injections=injections,
            output_png=os.path.join(self.tmp, dirname + ".png"),
        ).run()
        return store, report

    def test_single_nonfinite_is_isolated_and_mapped(self):
        _, report = self._run("valid_base.json", "single_fault.json", "single")
        codes = {f["code"] for f in report["failures"]}
        self.assertEqual(codes, {E_NON_FINITE_LATENT})
        hit_ids = {h["injection_id"] for h in report["injection_hits"]}
        self.assertEqual(hit_ids, {"INJ-001"})
        # Step 2 fails; step 1 and the encoder still succeed and are recorded.
        failed_stage = report["failures"][0]["stage"]
        self.assertEqual(failed_stage, denoise_stage(2))
        self.assertIn("encode_prompt",
                      {s["stage"] for s in report["stages"]})
        # Downstream steps get blocked, not silently recomputed.
        blocked = {b["stage"] for b in report["blocked"]}
        self.assertIn(denoise_stage(3), blocked)
        self.assertIn("decode", blocked)

    def test_each_denoise_fault_family_hits_its_unique_id(self):
        cases = [
            ("inj_001.json", "INJ-001", E_NON_FINITE_LATENT, denoise_stage(2)),
            ("inj_003.json", "INJ-003", E_NOISE_ENTRY_TRUNCATED, denoise_stage(3)),
            ("inj_004.json", "INJ-004", E_INJECTED_PATCH_OUT_OF_RANGE,
             denoise_stage(5)),
        ]
        for idx, (manifest, inj_id, code, stage) in enumerate(cases):
            with self.subTest(manifest=manifest):
                _, report = self._run("valid_base.json", manifest, f"fam{idx}")
                mapping = {h["injection_id"]: h["code"]
                           for h in report["injection_hits"]}
                self.assertEqual(mapping.get(inj_id), code)
                failure = next(f for f in report["failures"]
                               if f["code"] == code)
                self.assertEqual(failure["stage"], stage)
                self.assertEqual(failure["context"].get("injection_id"), inj_id)

    def test_multi_fault_report_keeps_stable_failure_ordering(self):
        # INJ-001 fails step 2; later chain stages are blocked, proving
        # failure isolation while independent upstream stages still complete.
        _, report = self._run("valid_base.json", "faults_t2i.json", "multi")
        self.assertTrue(
            {h["injection_id"] for h in report["injection_hits"]} >= {"INJ-001"}
        )
        stages = [f["stage"] for f in report["failures"]]
        self.assertEqual(
            stages,
            sorted(stages,
                   key=lambda x: report["dependency_graph"][x]["sort_key"]),
        )
        self.assertIn("encode_prompt",
                      {x["stage"] for x in report["stages"]})

    def test_image_size_fault_hits_in_img2img(self):
        _, report = self._run("valid_img2img.json", "faults.json", "allimg")
        mapping = {h["injection_id"]: h["code"] for h in report["injection_hits"]}
        self.assertEqual(mapping.get("INJ-002"), E_IMAGE_SIZE_MISMATCH)
        self.assertIn(E_IMAGE_SIZE_MISMATCH,
                      {f["code"] for f in report["failures"]})

    def test_injected_stage_writes_no_artifact(self):
        store, report = self._run("valid_base.json", "single_fault.json", "noart")
        self.assertEqual(store.list_versions(denoise_stage(2)), [])
        # Independent earlier stage still committed.
        self.assertEqual(store.list_versions(denoise_stage(1)), [1])


if __name__ == "__main__":
    unittest.main()
