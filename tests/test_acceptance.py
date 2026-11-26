from __future__ import annotations

import unittest

from edge_shard_diag.acceptance import run_acceptance


class AcceptanceTests(unittest.TestCase):
    def test_full_local_acceptance(self) -> None:
        report = run_acceptance("artifacts/test-acceptance")
        self.assertTrue(report["ok"])
        self.assertEqual(report["steps"]["receipts"]["counts"], {"d1": 4, "d2": 3, "d3": 4})
        self.assertTrue(report["steps"]["incremental"]["all_unchanged_bytes_identical"])
        self.assertTrue(report["steps"]["crash_recovery"]["child_was_sigkilled"])
        self.assertIn("shard-0002", report["steps"]["crash_recovery"]["recovery"]["unknown_shards"])


if __name__ == "__main__":
    unittest.main()
