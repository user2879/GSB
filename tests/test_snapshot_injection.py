import json
import os
import unittest

from hotblock_diag.engine import DiagnosticEngine
from hotblock_diag.errors import (E_ACC_OVERFLOW, E_NONFINITE_SCALE,
                                  E_PACK_TRUNCATED, E_SNAPSHOT_STALE,
                                  FailureList)
from hotblock_diag.injection import Injection, InjectionPlan
from tests._harness import make_fixedpoint, make_pack, make_prefilter, temp_workspace
from tests.test_engine import build_engine


class SnapshotTests(unittest.TestCase):
    def test_snapshot_manifest_and_stage_payloads_committed(self):
        ws = temp_workspace()
        pack, _ = make_pack(ws)
        run = os.path.join(ws, "run")
        engine = build_engine(run, pack)
        report = engine.run_to_completion()
        manifest_path = os.path.join(run, "snapshots", "committed", "manifest.json")
        self.assertTrue(os.path.exists(manifest_path))
        with open(manifest_path) as handle:
            manifest = json.load(handle)
        self.assertEqual(manifest["config_hash"], report["config_hash"])
        for stage in ("tensor_parse", "block_scoring", "selection", "hot_summary"):
            path = os.path.join(run, "snapshots", "committed", stage, "data.json")
            self.assertTrue(os.path.exists(path))

    def test_tampered_snapshot_forces_full_rebuild(self):
        ws = temp_workspace()
        pack, _ = make_pack(ws)
        run = os.path.join(ws, "run")
        engine = build_engine(run, pack)
        engine.run_to_completion()
        engine.snapshots.tamper()

        failures = FailureList()
        rebuilt = DiagnosticEngine(run, make_prefilter(), make_fixedpoint(),
                                   InjectionPlan(), failures, fresh=False)
        manifest = rebuilt.snapshots.resume_or_rebuild(
            config_hash=rebuilt.config_hash() if False else "x",
            tensor_fingerprint="y",
        )
        self.assertIsNone(manifest)
        self.assertIn(E_SNAPSHOT_STALE, failures.codes())


class InjectionTests(unittest.TestCase):
    def test_nonfinite_scale_is_isolated_to_target_block(self):
        ws = temp_workspace()
        pack, _ = make_pack(ws)
        plan = InjectionPlan([Injection("inj-1", "nonfinite_scale",
                                        {"qb": 0, "kb": 1, "value": "nan"})])
        failures = FailureList()
        engine = DiagnosticEngine(os.path.join(ws, "run"), make_prefilter(),
                                  make_fixedpoint(), plan, failures, fresh=True)
        assert engine.load_pack(pack)
        assert engine.build_layout()
        report = engine.run_to_completion()
        codes = {item["code"] for item in report["failures"]}
        self.assertIn(E_NONFINITE_SCALE, codes)
        self.assertEqual(report["block_pair_count"], 15)
        self.assertNotIn("0:1", report["block_pairs"])

    def test_force_overflow_records_business_code(self):
        ws = temp_workspace()
        pack, _ = make_pack(ws)
        plan = InjectionPlan([Injection("inj-2", "force_overflow",
                                        {"qb": 2, "kb": 0})])
        failures = FailureList()
        engine = DiagnosticEngine(os.path.join(ws, "run"), make_prefilter(),
                                  make_fixedpoint(), plan, failures, fresh=True)
        assert engine.load_pack(pack)
        assert engine.build_layout()
        report = engine.run_to_completion()
        overflow = [item for item in report["failures"]
                    if item["code"] == E_ACC_OVERFLOW]
        self.assertTrue(overflow)
        self.assertEqual(overflow[0]["target"], "qb=2;kb=0")
        self.assertTrue(report["block_pairs"]["2:0"]["overflows"])

    def test_truncated_pack_aborts_with_code(self):
        ws = temp_workspace()
        pack, _ = make_pack(ws)
        plan = InjectionPlan([Injection("inj-3", "truncate_pack",
                                        {"drop_bytes": 8})])
        failures = FailureList()
        engine = DiagnosticEngine(os.path.join(ws, "run"), make_prefilter(),
                                  make_fixedpoint(), plan, failures, fresh=True)
        engine._apply_truncation(pack)
        resolved = engine.resolve_pack_path(pack)
        self.assertFalse(engine.load_pack(resolved))
        self.assertIn(E_PACK_TRUNCATED, failures.codes())


if __name__ == "__main__":
    unittest.main()
