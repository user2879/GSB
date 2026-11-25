import unittest

from arbiter.fixedpoint import FixedPoint
from arbiter.numpy_checks import cross_check_quantization, vectorized_cost_grid


class NumpyCheckTest(unittest.TestCase):
    def test_quantization_oracle_agrees(self):
        fp = FixedPoint(3)
        values = ["1.2345", "10", "-7.891", "100.001", "0.0009", "555.5"]
        result = cross_check_quantization(values, fp)
        self.assertTrue(result["consistent"])
        self.assertEqual(result["checked"], len(values))

    def test_cost_grid_shape_and_values(self):
        grid = vectorized_cost_grid([1, 2, 4], [5, 10], avg_obs=3)
        self.assertEqual(grid.shape, (3, 2))
        self.assertEqual(grid[1, 0], 2 * 5 * 3)
        self.assertEqual(grid[2, 1], 4 * 10 * 3)


if __name__ == "__main__":
    unittest.main()
