import unittest

from rexapp.fixedpoint import classify_band, drift, is_boundary, quantize_score


class FixedPointTest(unittest.TestCase):
    def test_half_up_quantization(self):
        self.assertEqual(quantize_score(0.12345, 4), 1235)
        self.assertEqual(quantize_score(0.12344, 4), 1234)
        self.assertEqual(quantize_score(1.0, 2), 100)

    def test_equal_threshold_goes_to_higher_band(self):
        thresholds = [5000, 7000]
        labels = ["low", "medium", "high"]
        self.assertEqual(classify_band(5000, thresholds, labels), "medium")
        self.assertEqual(classify_band(7000, thresholds, labels), "high")
        self.assertEqual(classify_band(4999, thresholds, labels), "low")

    def test_boundary_and_drift(self):
        thresholds = [5495]
        self.assertTrue(is_boundary(5495, thresholds))
        value = drift(0.5495238095238095, 5495, 4)
        self.assertLess(value, 0)


if __name__ == "__main__":
    unittest.main()
