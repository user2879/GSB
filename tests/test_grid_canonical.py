import unittest

from cfgdist.config.grid import Grid
from cfgdist.errors import PlatformError
from cfgdist.util.canonical import canonical_bytes, digest


class GridTests(unittest.TestCase):
    def test_rejects_non_power_of_ten_exponent(self):
        for bad in (-1, 13, 99):
            with self.assertRaises(PlatformError) as ctx:
                Grid.from_q(bad)
            self.assertEqual(ctx.exception.code, "E_GRID")

    def test_rejects_non_integer(self):
        with self.assertRaises(PlatformError) as ctx:
            Grid.from_q(2.5)  # type: ignore[arg-type]
        self.assertEqual(ctx.exception.code, "E_GRID")

    def test_scale_is_power_of_ten(self):
        self.assertEqual(Grid.from_q(3).scale, 1000)


class CanonicalTests(unittest.TestCase):
    def test_key_order_does_not_change_bytes(self):
        a = canonical_bytes({"x": 1, "y": [1, 2]})
        b = canonical_bytes({"y": [1, 2], "x": 1})
        self.assertEqual(a, b)

    def test_digest_stable(self):
        self.assertEqual(digest({"a": 1}), digest({"a": 1}))
        self.assertNotEqual(digest({"a": 1}), digest({"a": 2}))


if __name__ == "__main__":
    unittest.main()
