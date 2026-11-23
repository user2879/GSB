import copy
import unittest

from tests import _bootstrap  # noqa: F401
from qcalib.calibration import calibrate_column, quantile_permille
from qcalib.attribution import analyze_column
from qcalib.config import GroupConfig
from qcalib.kernel import rescale_column


def _cfg(**over):
    base = dict(
        group_id="g", columns=["x"], scheme="asymmetric",
        q_lo_permille=25, q_hi_permille=975, qbits=8, fbits=2,
        rounding="half_up", saturation="clip", abs_tol=0.1, rel_tol=0.05,
    )
    base.update(over)
    cfg = GroupConfig(**base)
    cfg.params_fingerprint = "test"
    return cfg


class QuantileTest(unittest.TestCase):
    def test_endpoints_and_linear_interpolation(self):
        data = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(quantile_permille(data, 0), 1.0)
        self.assertEqual(quantile_permille(data, 1000), 4.0)
        # position = 0.5*(3)=1.5 -> 2.5
        self.assertAlmostEqual(quantile_permille(data, 500), 2.5)

    def test_tie_break_lower_index(self):
        data = [1.0, 1.0, 1.0, 2.0]
        # 并列处返回低下标值，不抛异常且确定
        self.assertEqual(quantile_permille(data, 200), 1.0)


class CalibKernelTest(unittest.TestCase):
    def setUp(self):
        self.vector = [0.1, -0.2, 0.5, 1.0, None, 3.0, -4.0]
        self.row_ids = [f"r{i}" for i in range(len(self.vector))]

    def test_deterministic_repeat(self):
        cfg = _cfg()
        c1 = calibrate_column(cfg, "x", self.vector, self.row_ids)
        c2 = calibrate_column(cfg, "x", self.vector, self.row_ids)
        self.assertEqual(c1.to_dict(), c2.to_dict())

    def test_fbits_changes_integers_and_saturation(self):
        cfg = _cfg()
        calib = calibrate_column(cfg, "x", self.vector, self.row_ids)
        per_fbits = {}
        for fbits in (1, 3, 5):
            c = _cfg(fbits=fbits)
            cal = calibrate_column(c, "x", self.vector, self.row_ids)
            r, _ = rescale_column(c, cal, self.vector, self.row_ids)
            ints = tuple(e.quantized for e in r.elements if e.quantized is not None)
            per_fbits[fbits] = (cal.scale, ints, r.n_saturated)
        scales = {v[0] for v in per_fbits.values()}
        self.assertGreater(len(scales), 1)  # 尺度随 fbits 改变
        ints_sets = {v[1] for v in per_fbits.values()}
        self.assertGreater(len(ints_sets), 1)  # 中间整数真实改变

    def test_saturation_records_direction_and_raw(self):
        cfg = _cfg(saturation="clip")
        cal = calibrate_column(cfg, "x", self.vector, self.row_ids)
        r, _ = rescale_column(cfg, cal, self.vector, self.row_ids)
        saturated = [e for e in r.elements if e.saturated]
        self.assertTrue(saturated)
        for e in saturated:
            self.assertIn(e.direction, ("low", "high"))
            self.assertIsNotNone(e.raw_int)
            self.assertNotEqual(e.raw_int, e.quantized)

    def test_reject_strategy_skips_accumulation(self):
        clip_cfg = _cfg(saturation="clip")
        reject_cfg = _cfg(saturation="reject")
        clip_r, _ = rescale_column(
            clip_cfg, calibrate_column(clip_cfg, "x", self.vector, self.row_ids),
            self.vector, self.row_ids)
        rej_r, _ = rescale_column(
            reject_cfg, calibrate_column(reject_cfg, "x", self.vector, self.row_ids),
            self.vector, self.row_ids)
        self.assertGreaterEqual(clip_r.n_saturated, rej_r.n_saturated)
        rejected = [e for e in rej_r.elements if e.direction == "reject"]
        self.assertTrue(rejected)
        self.assertTrue(all(e.quantized is None for e in rejected))

    def test_accumulator_overflow_reports_position(self):
        cfg = _cfg()
        cal = calibrate_column(cfg, "x", self.vector, self.row_ids)
        r, failures = rescale_column(cfg, cal, self.vector, self.row_ids, accumulator_bits=3)
        self.assertTrue(any(f.code == "E_ACC_OVERFLOW" for f in failures.items))
        self.assertTrue(r.accumulator_overflow)
        self.assertTrue(r.overflow_positions)

    def test_attribution_zero_branch_and_components(self):
        cfg = _cfg()
        vector = [0.0, 0.05, 2.0]
        rows = [f"z{i}" for i in range(3)]
        cal = calibrate_column(cfg, "x", vector, rows)
        r, _ = rescale_column(cfg, cal, vector, rows)
        report = analyze_column(cfg, cal, r)
        zero_elem = next(e for e in report.elements if e.original == 0.0)
        self.assertEqual(zero_elem.tolerance_branch, "absolute-zero")
        for e in report.elements:
            if e.out_of_tolerance:
                self.assertIn(e.dominant_source,
                              ("quantile", "scale_quant", "saturation", "dequant_round"))
                self.assertEqual(set(e.components),
                                 {"quantile", "scale_quant", "saturation", "dequant_round"})


if __name__ == "__main__":
    unittest.main()
