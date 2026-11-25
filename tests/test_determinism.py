import unittest

from arbplatform.determinism import canonical, digest_of


class DeterminismTest(unittest.TestCase):
    def test_key_order_independent(self):
        a = {"z": 1, "a": [1, 2, 3], "m": {"y": 1, "x": 2}}
        b = {"m": {"x": 2, "y": 1}, "a": [1, 2, 3], "z": 1}
        self.assertEqual(canonical(a), canonical(b))
        self.assertEqual(digest_of(a), digest_of(b))

    def test_list_order_matters(self):
        self.assertNotEqual(canonical([1, 2]), canonical([2, 1]))


if __name__ == "__main__":
    unittest.main()
