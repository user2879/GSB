from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from mwlab.config import declarations_from_document
from mwlab.ledger import Ledger

from .helpers import load_config


class LedgerVersionTests(unittest.TestCase):
    def test_format_policy_and_group_changes_increment_parameter_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = Ledger(Path(directory) / "ledger.sqlite3")
            try:
                base_document = load_config("session.json")
                base = declarations_from_document(base_document)
                first = ledger.start_execution(base, "0" * 64, [])
                self.assertEqual(first["parameter_version"], 1)

                format_document = copy.deepcopy(base_document)
                for group in format_document["consumer_groups"]:
                    if group["name"] == "analytics":
                        group["format_version"] = "v2"
                format_changed = declarations_from_document(format_document)
                second = ledger.start_execution(format_changed, "0" * 64, [])
                self.assertEqual(second["parameter_version"], 2)
                self.assertTrue(second["parameter_changed"])

                policy_document = copy.deepcopy(format_document)
                for topic in policy_document["topics"]:
                    if topic["name"] == "orders":
                        topic["confirmation_policy"]["max_allowed_retries"] = 1
                policy_changed = declarations_from_document(policy_document)
                third = ledger.start_execution(policy_changed, "0" * 64, [])
                self.assertEqual(third["parameter_version"], 3)
                self.assertTrue(third["parameter_changed"])

                group_document = copy.deepcopy(policy_document)
                group_document["consumer_groups"].append({"name": "shadow", "format_version": "v3"})
                group_changed = declarations_from_document(group_document)
                fourth = ledger.start_execution(group_changed, "0" * 64, [])
                self.assertEqual(fourth["parameter_version"], 4)
                self.assertTrue(fourth["parameter_changed"])

                versions = ledger.connection.execute(
                    "SELECT version FROM parameter_versions ORDER BY version"
                ).fetchall()
                self.assertEqual([row[0] for row in versions], [1, 2, 3, 4])
            finally:
                ledger.close()


if __name__ == "__main__":
    unittest.main()

