import sqlite3
import tempfile
import unittest
from pathlib import Path

from arcore.ledger import (
    FORMAT_VERSION,
    commit_round,
    last_committed_round_id,
    load_committed_rounds,
    mark_round_unknown,
    open_ledger,
    replay_rounds,
    unknown_round_ids,
)


class LedgerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "ledger.sqlite3"

    def tearDown(self):
        self.tmp.cleanup()

    def _payload(self, round_id, value=1):
        return {"round_id": round_id, "result": value,
                "nested": {"k": value}}

    def test_commit_and_replay_without_diffs(self):
        connection, _ = open_ledger(self.path)
        commit_round(connection, 1, self._payload(1), {"s": 1}, 100)
        diffs = replay_rounds(connection, [self._payload(1)])
        self.assertEqual(diffs, [])
        self.assertEqual(last_committed_round_id(connection), 1)
        connection.close()

    def test_replay_detects_field_difference(self):
        connection, _ = open_ledger(self.path)
        commit_round(connection, 1, self._payload(1), {}, 100)
        diffs = replay_rounds(connection, [self._payload(1, value=2)])
        self.assertTrue(any(d["field_path"].endswith("result") for d in diffs))
        connection.close()

    def test_unknown_round_requires_rerun_and_no_exit_report(self):
        connection, _ = open_ledger(self.path)
        commit_round(connection, 1, self._payload(1), {}, 100)
        mark_round_unknown(connection, 2)
        connection.close()
        recovered, _ = open_ledger(self.path)
        self.assertEqual(unknown_round_ids(recovered), [2])
        self.assertEqual(last_committed_round_id(recovered), 1)
        committed = load_committed_rounds(recovered)
        self.assertEqual([p["round_id"] for p in committed], [1])
        recovered.close()

    def test_digest_mismatch_triggers_wholesale_replacement(self):
        connection, _ = open_ledger(self.path)
        commit_round(connection, 1, self._payload(1), {}, 100)
        connection.close()
        # Corrupt a committed payload while leaving the stored digest intact.
        corrupted = sqlite3.connect(str(self.path))
        corrupted.execute(
            "UPDATE rounds SET payload_json = ? WHERE round_id = 1",
            ('{"round_id": 1, "result": 999}',))
        corrupted.commit()
        corrupted.close()
        reopened, notes = open_ledger(self.path)
        self.assertTrue(any("replaced wholesale" in note for note in notes))
        self.assertEqual(last_committed_round_id(reopened), 0)
        reopened.close()

    def test_format_version_mismatch_triggers_replacement(self):
        connection, _ = open_ledger(self.path)
        connection.execute(
            "UPDATE ledger_meta SET value = ? WHERE key = 'format_version'",
            (str(FORMAT_VERSION + 9),))
        connection.commit()
        connection.close()
        reopened, notes = open_ledger(self.path)
        self.assertTrue(any("replaced wholesale" in note for note in notes))
        reopened.close()


if __name__ == "__main__":
    unittest.main()
