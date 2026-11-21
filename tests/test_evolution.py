from __future__ import annotations

import unittest

from mvcc_workbench.ledger import Ledger
from mvcc_workbench.workbench import Workbench

from tests.helpers import base_dataset, evolved_dataset, temp_db


class EvolutionTests(unittest.TestCase):
    def test_evolution_replaces_view_and_preserves_unaffected_bytes(self) -> None:
        db = temp_db("mvcc-evolution-test-")
        with Ledger(db) as ledger:
            cold = Workbench(base_dataset(), ledger).execute()
        before_reason = {
            item["query_id"]: item["reason_digest"]
            for item in cold["decisions"]
        }
        with Ledger(db) as ledger:
            evolved = Workbench(evolved_dataset(), ledger).execute()
            self.assertEqual(evolved["status"], "EVOLUTION_REPLACED")
            self.assertEqual(evolved["code"], "E_REPLAY")
            self.assertEqual(evolved["generation"], 2)
            location = evolved["diagnostic"]["location"]
            self.assertEqual(location["read_point_id"], "RP01")
            self.assertEqual(location["row_key"], "acct:1")
            self.assertIn("field", location)
            affected = evolved["affected"]
            self.assertEqual(
                affected["affected_query_ids"],
                ["Q01", "Q02", "Q03", "Q04", "Q05", "Q06"],
            )
            changed = {item["query_id"]: item for item in affected["changed_queries"]}
            self.assertEqual(changed["Q01"]["before_visible_version_no"], 3)
            self.assertEqual(changed["Q01"]["after_visible_version_no"], 4)
            unchanged = {item["query_id"]: item for item in affected["unchanged_checks"]}
            self.assertTrue(unchanged["Q07"]["byte_equal"])
            self.assertEqual(unchanged["Q07"]["before_reason_digest"], before_reason["Q07"])
            self.assertEqual(unchanged["Q08"]["before_reason_digest"], before_reason["Q08"])
        with Ledger(db) as ledger:
            replay = Workbench(evolved_dataset(), ledger).execute()
            self.assertEqual(replay["status"], "REPLAY_OK")
            self.assertEqual(replay["generation"], 2)
            self.assertEqual(len(ledger.completed_rounds(2)), 3)


if __name__ == "__main__":
    unittest.main()
