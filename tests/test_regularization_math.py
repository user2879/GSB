import unittest

import numpy as np

from regbench.models import (
    _regularized_gradient,
    _regularized_loss,
    _subgradient_l1,
)


class RegularizationMathTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.x = rng.normal(size=(16, 5))
        self.x[:, 0] = 1.0
        self.y = rng.normal(size=16)
        self.w = rng.normal(size=5)
        self.w[2] = 0.0  # 覆盖零系数次梯度分支

    def test_l2_gradient_matches_finite_difference(self):
        penalty = 0.7
        analytic = _regularized_gradient(
            "ridge", self.x, self.y, self.w, penalty, 0.0
        )
        numeric = self._finite_difference(
            lambda weights: _regularized_loss(
                "ridge", self.x, self.y, weights, penalty, 0.0
            )
        )
        np.testing.assert_allclose(analytic, numeric, atol=1e-6)

    def test_elasticnet_blends_l1_and_l2_explicitly(self):
        penalty, rho = 0.5, 0.4
        loss = _regularized_loss(
            "elasticnet", self.x, self.y, self.w, penalty, rho
        )
        residual = self.x.dot(self.w) - self.y
        data_loss = 0.5 * float(np.mean(residual**2))
        body = self.w[1:]
        expected_reg = penalty * (
            rho * np.sum(np.abs(body))
            + 0.5 * (1.0 - rho) * np.sum(body**2)
        )
        self.assertAlmostEqual(loss, data_loss + expected_reg, places=10)

        grad = _regularized_gradient(
            "elasticnet", self.x, self.y, self.w, penalty, rho
        )
        data_grad = self.x.T.dot(residual) / self.x.shape[0]
        expected_grad = data_grad.copy()
        expected_grad[1:] += penalty * (
            rho * _subgradient_l1(body) + (1.0 - rho) * body
        )
        np.testing.assert_allclose(grad, expected_grad, atol=1e-12)

    def test_l1_zero_subgradient_fixed_to_zero(self):
        sub = _subgradient_l1(np.array([0.0, 0.3, -0.2]))
        np.testing.assert_array_equal(sub, np.array([0.0, 1.0, -1.0]))

    def test_intercept_is_not_regularized(self):
        penalty = 1.0
        grad = _regularized_gradient(
            "lasso", self.x, self.y, self.w, penalty, 0.0
        )
        residual = self.x.dot(self.w) - self.y
        self.assertAlmostEqual(
            grad[0], float(np.mean(residual)), places=12
        )

    def _finite_difference(self, fn, eps=1e-7):
        grad = np.zeros_like(self.w)
        for index in range(self.w.shape[0]):
            step = np.zeros_like(self.w)
            step[index] = eps
            grad[index] = (fn(self.w + step) - fn(self.w - step)) / (2 * eps)
        return grad


if __name__ == "__main__":
    unittest.main()
