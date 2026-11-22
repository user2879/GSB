from __future__ import annotations

import unittest
from pathlib import Path

from search_intent_workbench.faults import run_fault_suite


ROOT = Path(__file__).resolve().parents[1]


class FaultInjectionTests(unittest.TestCase):
    def test_all_four_isolated_injections_are_detected(self) -> None:
        report = run_fault_suite(ROOT / "data" / "injected" / "injections.json")
        self.assertTrue(report["summary"]["all_detected"])
        self.assertEqual(report["summary"]["injection_count"], 4)
        self.assertEqual(report["summary"]["hit_count"], 4)
        by_id = {item["injection_id"]: item for item in report["injections"]}
        self.assertEqual(set(by_id), {"I01", "I02", "I03", "I04"})
        self.assertTrue(by_id["I01"]["detected"])
        self.assertIn("E_CONFIG_INVALID", by_id["I01"]["observed_codes"])
        self.assertTrue(by_id["I02"]["detected"])
        self.assertIn("E_CONFIG_INVALID", by_id["I02"]["observed_codes"])
        self.assertEqual(by_id["I03"]["execution"]["processed_query_count"], 16)
        self.assertEqual(by_id["I03"]["execution"]["failed_query_ids"], ["q_fault_turn"])
        self.assertEqual(by_id["I04"]["execution"]["processed_query_count"], 16)
        self.assertEqual(by_id["I04"]["execution"]["failed_query_ids"], ["q_fault_oov"])
        self.assertEqual(by_id["I03"]["execution"]["run_failure_count"], 1)
        self.assertEqual(by_id["I04"]["execution"]["run_failure_count"], 1)


if __name__ == "__main__":
    unittest.main()
