import unittest

from repairbench.fixedpoint import (
    INT64_MAX,
    INT64_MIN,
    ErrorLedger,
    FixedPoint,
)


class TestFixedPoint(unittest.TestCase):
    def test_scale_range(self):
        for e in range(0, 7):
            self.assertEqual(FixedPoint(e).scale, 10**e)
        with self.assertRaises(ValueError):
            FixedPoint(7)
        with self.assertRaises(ValueError):
            FixedPoint(-1)

    def test_truncation_toward_zero(self):
        fp = FixedPoint(3)
        self.assertEqual(fp.quantize_bytes(1023), 999)   # 1023*1000//1024
        self.assertEqual(fp.quantize_bytes(1024), 1000)
        self.assertEqual(fp.dequantize_bytes(1000), 1024)
        self.assertEqual(fp.mul(-1500, 2000), -3000)     # 向零截断
        self.assertEqual(fp.div(1000, 3), 333333)

    def test_saturation(self):
        fp = FixedPoint(6)
        self.assertEqual(fp.quantize_units(2**50), INT64_MAX)
        self.assertEqual(fp.quantize_units(-(2**50)), INT64_MIN)
        self.assertEqual(fp.div(5, 0), INT64_MAX)
        self.assertEqual(fp.div(-5, 0), INT64_MIN)

    def test_error_ledger_levels(self):
        fp = FixedPoint(3)
        led = ErrorLedger(fp)
        led.add_block_quantization("b1", "n1", 2)
        led.add_batch_split("b1", "n1", 500)
        led.add_queue_delay("b1", "n1", 1)
        led.add_block_quantization("b2", "n2", 3)
        self.assertEqual(led.block_byte_error_quanta("b1"), 2)
        self.assertEqual(led.node_bound("n1").total(), 503)
        glob = led.global_bound()
        self.assertEqual(glob.total(), 506)
        self.assertEqual(glob.dominant(), "batch_split_rounding")
        report = led.report()
        self.assertEqual(report["global"]["dominant"], "batch_split_rounding")
        self.assertIn("total_bytes_bound", report["global"])


if __name__ == "__main__":
    unittest.main()
