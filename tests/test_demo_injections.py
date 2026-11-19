from __future__ import annotations

import unittest

from config_ledger.demo import run_demo_case

from .helpers import temp_workspace


class DemoInjectionTests(unittest.TestCase):
    def test_three_required_injections_produce_their_business_codes(self) -> None:
        with temp_workspace() as temp:
            invalid_json = run_demo_case(
                sample="divergence", inject="invalid_json", runtime_dir=temp
            )
            codes = invalid_json["structured_log"]["counts"]
            self.assertGreaterEqual(codes.get("E_OBSERVE", 0), 1)

            unreachable = run_demo_case(
                sample="divergence", inject="port_unreachable", runtime_dir=temp
            )
            codes = unreachable["structured_log"]["counts"]
            self.assertGreaterEqual(codes.get("E_PORT", 0), 1)
            self.assertGreaterEqual(codes.get("E_OBSERVE", 0), 1)

            snapshot = run_demo_case(
                sample="divergence", inject="snapshot_mismatch", runtime_dir=temp
            )
            codes = snapshot["structured_log"]["counts"]
            self.assertEqual(codes.get("E_SNAPSHOT", 0), 1)
            self.assertIn("digest mismatch", snapshot["snapshot_error"])


if __name__ == "__main__":
    unittest.main()
