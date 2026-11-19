from __future__ import annotations

import hashlib
import unittest

from glyphforge.demo import canonical_report, run_demo


class DemoTests(unittest.TestCase):
    def test_two_runs_are_byte_identical(self) -> None:
        first = canonical_report(run_demo())
        second = canonical_report(run_demo())
        self.assertEqual(first, second)
        self.assertEqual(hashlib.sha256(first).hexdigest(), hashlib.sha256(second).hexdigest())


if __name__ == "__main__":
    unittest.main()
