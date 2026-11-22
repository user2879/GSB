import copy
import os
import shutil
import tempfile
import unittest

import numpy as np

from egw.config import config_hash, load_config
from egw.engine import GatherEngine
from egw.errors import (
    E_AGG_MODE_UNKNOWN,
    E_CONFIG_INVALID,
    E_DIM_MISMATCH,
    E_ID_MISSING,
    E_NONFINITE_VECTOR,
    E_POINT_BUDGET_PENDING,
    E_ROW_INVALID,
)
from egw.layout_probe import run_probe
from egw.requests import load_requests
from egw.samplegen import generate_samples
from egw.scenario import (
    apply_agg_overrides,
    apply_state_edits,
    load_edits,
    run_full,
    run_incremental,
)
from egw.storage import apply_edit, block_layout_fingerprint, load_rows

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DIR = os.path.join(ROOT, "samples")


class SampleFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="egw-test-")
        cls.info = generate_samples(cls.tmp)
        cls.cfg = load_config(cls.info["config_path"])
        cls.table, cls.row_failures = load_rows(cls.info["rows_path"], cls.cfg["dim"])
        cls.requests, cls.request_failures = load_requests(
            cls.info["requests_path"], cls.cfg["agg_modes"]
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)


class ValidationTests(SampleFixture):
    def test_injected_row_failures(self):
        codes = [item["code"] for item in self.row_failures]
        self.assertIn(E_DIM_MISMATCH, codes)
        self.assertIn(E_NONFINITE_VECTOR, codes)
        self.assertEqual(codes.count(E_ROW_INVALID), 4)

    def test_injected_request_failures(self):
        codes = [item["code"] for item in self.request_failures]
        self.assertIn(E_AGG_MODE_UNKNOWN, codes)
        self.assertEqual(len(self.requests), 8)

    def test_zero_budget_config_rejected(self):
        from egw.errors import EGWError

        with self.assertRaises(EGWError) as ctx:
            load_config(self.info["zero_budget_config_path"])
        self.assertEqual(ctx.exception.code, E_CONFIG_INVALID)

    def test_valid_rows_loaded(self):
        self.assertEqual(len(self.table.rows), 32)
        self.assertEqual(set(self.table.block_offsets), {0, 1, 2, 3})


class GatherTests(SampleFixture):
    def _engine(self):
        return GatherEngine(self.cfg, snapshot_dir=None)

    def _prepare(self, engine):
        engine.prepare_quantization(
            self.table,
            {
                "quant_level": self.cfg["quant_level"],
                "fractional_bits": self.cfg["fractional_bits"],
            },
        )

    def test_sum_and_mean_share_gather(self):
        engine = self._engine()
        engine.prepare_quantization(self.table, {
            "quant_level": self.cfg["quant_level"],
            "fractional_bits": self.cfg["fractional_bits"],
        })
        req_sum = next(req for req in self.requests if req.request_id == "rq_sum_01")
        req_mean = copy.copy(req_sum)
        req_mean.agg_mode = "MEAN"
        sum_result = engine.run_gather(self.table, req_sum)
        mean_result = engine.run_gather(self.table, req_mean)
        self.assertEqual(sum_result["gathered_int"], mean_result["gathered_int"])
        self.assertNotEqual(sum_result["result_int"], mean_result["result_int"])
        self.assertEqual(len(mean_result["remainder_int"]), self.cfg["dim"])
        # 余数恒满足 total = quotient*count + remainder（FLOOR 语义下可为负）
        rebuilt = [
            q * len(req_mean.id_sequence) + r
            for q, r in zip(mean_result["result_int"], mean_result["remainder_int"])
        ]
        self.assertEqual(rebuilt, mean_result["accum_int"])

    def test_missing_id_continues_and_records_position(self):
        engine = self._engine()
        self._prepare(engine)
        req = next(req for req in self.requests if req.request_id == "rq_missing_01")
        result = engine.run_gather(self.table, req)
        self.assertEqual(result["missing_count"], 1)
        self.assertEqual(result["found_count"], 2)
        self.assertEqual(result["missing_failures"][0]["code"], E_ID_MISSING)
        self.assertEqual(result["missing_failures"][0]["position"], 1)

    def test_stage_errors_and_first_breach(self):
        engine = self._engine()
        self._prepare(engine)
        req = next(req for req in self.requests if req.request_id == "rq_mean_01")
        result = engine.run_gather(self.table, req)
        for stage in ("quantize", "gather", "accumulate", "divide", "round"):
            self.assertIn(stage, result["stage_errors"])
        self.assertIn(result["first_breach_stage"], (None, "quantize", "gather", "accumulate", "divide", "round"))
        self.assertGreaterEqual(result["stage_errors"]["quantize"]["max_abs_error"], 0.0)

    def test_budget_pending(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["point_budget"] = 3
        engine = GatherEngine(cfg, snapshot_dir=None)
        outcome = engine.run(self.table, self.requests, force_rebuild=True, concurrency_override=1)
        pending_codes = [f["code"] for f in outcome["failures"] if f["code"] == E_POINT_BUDGET_PENDING]
        self.assertTrue(pending_codes)
        self.assertTrue(outcome["pending_request_ids"])
        self.assertEqual(outcome["budget_consumed"], 3)

    def test_invalidation_and_reuse(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["point_budget"] = 1000
        engine = GatherEngine(cfg, snapshot_dir=None)
        first = engine.run(self.table, self.requests, force_rebuild=True, concurrency_override=1)
        self.assertFalse(first["pending_request_ids"])
        self.assertEqual(len(first["reuse_ledger"]["recomputed"]), len(self.requests))
        untouched = self.table.copy()
        second = engine.run(untouched, self.requests, concurrency_override=1)
        self.assertEqual(
            sorted(second["reuse_ledger"]["reused"]),
            sorted(str(req.request_id) for req in self.requests),
        )
        target_row = self.cfg and 100
        edit = {
            "op": "replace_row",
            "row_id": target_row,
            "block_id": 0,
            "offset": 0,
            "vector": [9.9] * self.cfg["dim"],
        }
        changed_table = apply_edit(self.table, edit)
        engine.invalidate_for_edit(changed_table, edit, {target_row}, set(), False)
        third = engine.run(changed_table, self.requests, concurrency_override=1)
        affected = {
            str(req.request_id)
            for req in self.requests
            if target_row in req.id_sequence
        }
        self.assertEqual(set(third["reuse_ledger"]["recomputed"]), affected)
        self.assertTrue(set(third["reuse_ledger"]["reused"]).isdisjoint(affected))

    def test_fingerprints_change_with_gear(self):
        engine_a = self._engine()
        engine_a.prepare_quantization(self.table, {"quant_level": 16, "fractional_bits": 8})
        fp_a = engine_a.quant_cache.rows[100]["row_fingerprint"]
        engine_b = GatherEngine(self.cfg, snapshot_dir=None)
        engine_b.prepare_quantization(self.table, {"quant_level": 12, "fractional_bits": 4})
        fp_b = engine_b.quant_cache.rows[100]["row_fingerprint"]
        self.assertNotEqual(fp_a, fp_b)
        self.assertNotEqual(engine_a.layout_fp, engine_b.layout_fp)
        self.assertNotEqual(
            block_layout_fingerprint(engine_a.block_fps),
            block_layout_fingerprint(engine_b.block_fps),
        )


class EquivalenceTests(SampleFixture):
    def test_incremental_equals_full_single_and_concurrent(self):
        edits = load_edits(self.info["edits_path"])
        final_table, final_cfg, overrides, _ = apply_state_edits(self.table, edits, self.cfg)
        final_requests = apply_agg_overrides(copy.deepcopy(self.requests), overrides)
        reports = {}
        for label, concurrency in (("single", 1), ("concurrent", final_cfg["concurrency"])):
            inc_engine, inc_outcome, _, _, _, _ = run_incremental(
                self.cfg,
                self.table,
                copy.deepcopy(self.requests),
                edits,
                snapshot_dir=os.path.join(self.tmp, f"inc_{label}"),
                concurrency_override=concurrency,
            )
            _, full_outcome = run_full(
                final_cfg,
                final_table,
                copy.deepcopy(final_requests),
                snapshot_dir=os.path.join(self.tmp, f"full_{label}"),
                concurrency_override=concurrency,
                expected_version=inc_outcome["global_version"],
            )
            reports[label] = (inc_outcome, full_outcome)
            for key in ("results", "failures", "reuse_ledger", "stage_versions"):
                self.assertEqual(inc_outcome[key], full_outcome[key], f"{label}:{key}")
        self.assertEqual(
            reports["single"][0]["results"],
            reports["concurrent"][0]["results"],
        )
        self.assertEqual(
            reports["single"][0]["failures"],
            reports["concurrent"][0]["failures"],
        )

    def test_snapshot_resume_uses_config_hash(self):
        snap = os.path.join(self.tmp, "resume-snap")
        resume_cfg = copy.deepcopy(self.cfg)
        resume_cfg["point_budget"] = 1000
        engine = GatherEngine(resume_cfg, snapshot_dir=snap)
        outcome_a = engine.run(self.table, self.requests, force_rebuild=True, concurrency_override=1)
        engine_b = GatherEngine(resume_cfg, snapshot_dir=snap)
        self.assertIsNotNone(engine_b.manifest)
        outcome_b = engine_b.run(self.table, self.requests, concurrency_override=1)
        self.assertEqual(
            outcome_b["reuse_ledger"]["reused"],
            sorted(str(req.request_id) for req in self.requests),
        )
        other_cfg = copy.deepcopy(resume_cfg)
        other_cfg["tolerance"] = self.cfg["tolerance"] * 2
        engine_c = GatherEngine(other_cfg, snapshot_dir=snap)
        self.assertIsNone(engine_c.manifest)
        self.assertNotEqual(config_hash(self.cfg), config_hash(other_cfg))


class LayoutProbeTests(SampleFixture):
    def test_order_effect_observed_but_float_consistent(self):
        probe = run_probe(self.cfg)
        self.assertTrue(probe["order_effect_observed"])
        self.assertGreater(probe["block_merge_max_abs_diff"], 0)
        self.assertEqual(probe["fixedpoint_max_abs_diff"], 0)
        self.assertTrue(probe["float_reference_consistent"])
        self.assertEqual(probe["float64_reference_max_abs_diff"], 0.0)


if __name__ == "__main__":
    unittest.main()
