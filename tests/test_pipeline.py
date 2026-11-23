import unittest

from tspworkbench.config import ModelSpec, QuantSpec
from tspworkbench.models import fit_model
from tspworkbench.quantize import quantize, quantize_list, compute_metrics
from tspworkbench.errors import E_NON_FINITE
from tspworkbench.windows import _impute


class DeterminismTests(unittest.TestCase):
    def test_models_deterministic_and_shared_interface(self):
        train = [float(i) for i in range(20)]
        specs = [
            ModelSpec("naive", 0.2, 0, {}),
            ModelSpec("moving_average", 0.3, 5, {}),
            ModelSpec("autoregressive", 0.5, 3, {}),
        ]
        for spec in specs:
            a = fit_model(spec, train)
            b = fit_model(spec, train)
            self.assertEqual(a.predict(6), b.predict(6))
            self.assertEqual(a.residuals, b.residuals)
            self.assertEqual(a.train_window, train)

    def test_quant_gear_changes_integers_and_metrics(self):
        preds = [1.23456, 2.99999, -0.5]
        actual = [1.2, 3.0, -0.4]
        q1 = QuantSpec(frac_bits=4, gear=1)   # scale 32
        q2 = QuantSpec(frac_bits=4, gear=2)   # scale 64
        self.assertNotEqual(quantize_list(preds, q1), quantize_list(preds, q2))
        m1 = compute_metrics(quantize_list(preds, q1), quantize_list(actual, q1), q1)
        m2 = compute_metrics(quantize_list(preds, q2), quantize_list(actual, q2), q2)
        self.assertNotEqual((m1.mae_q, m1.rmse_q), (m2.mae_q, m2.rmse_q))

    def test_near_zero_branch(self):
        q = QuantSpec(frac_bits=2, gear=0)  # scale 4
        # 真值 0.1 -> 量化为 0，触发近零分支
        pred_q = quantize_list([0.9], q)
        actual_q = quantize_list([0.1], q)
        self.assertEqual(actual_q, [0])
        m = compute_metrics(pred_q, actual_q, q)
        self.assertTrue(m.near_zero_used)
        self.assertEqual(m.mape_bp, -1)

    def test_non_finite_quantization_rejected(self):
        with self.assertRaises(Exception) as ctx:
            quantize(float("nan"), QuantSpec(2, 0))
        self.assertEqual(ctx.exception.code, E_NON_FINITE)

    def test_imputation_rules_only_use_window(self):
        vals = [None, 1.0, None, 3.0]
        ff, pos = _impute(vals, "forward_fill")
        self.assertEqual(ff, [0.0, 1.0, 1.0, 3.0])
        self.assertEqual(pos, [0, 2])
        mf, _ = _impute(vals, "mean_fill")
        self.assertEqual(mf, [2.0, 1.0, 2.0, 3.0])
        zf, _ = _impute(vals, "zero_fill")
        self.assertEqual(zf, [0.0, 1.0, 0.0, 3.0])


if __name__ == "__main__":
    unittest.main()
