import json
import os
import shutil
import tempfile
import unittest

from shardbench.engine import Engine
from shardbench.errors import ReplayError
from shardbench.logging_utils import StructuredLogger
from shardbench.replay import replay_round

from .test_validation import sample_paths


class ReplayDifferenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.config_path = os.path.join(self.tmp, "config.json")
        self.units_path = os.path.join(self.tmp, "units.json")
        shutil.copy(sample_paths()[0], self.config_path)
        shutil.copy(sample_paths()[1], self.units_path)
        self.ledger_path = os.path.join(self.tmp, "ledger.sqlite3")
        self.snapshot_path = os.path.join(self.tmp, "snapshot.json")
        self.log_path = os.path.join(self.tmp, "events.jsonl")
        logger = StructuredLogger(self.log_path)
        engine = Engine(
            self.config_path,
            self.units_path,
            self.ledger_path,
            self.snapshot_path,
            logger=logger,
        )
        engine.run_round()
        engine.close()
        logger.close()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_clean_ledger_replays_ok(self):
        result = replay_round(self.ledger_path, 1)
        self.assertEqual(result["status"], "ok")

    def test_tampered_target_reports_first_mismatch(self):
        import sqlite3

        conn = sqlite3.connect(self.ledger_path)
        row = conn.execute(
            "SELECT unit_id, targets_json FROM unit_results "
            "WHERE round_id=1 ORDER BY unit_id LIMIT 1"
        ).fetchone()
        unit_id, targets_json = row
        targets = json.loads(targets_json)
        tampered = targets + ["db9.t9"] if targets else ["db9.t9"]
        conn.execute(
            "UPDATE unit_results SET targets_json=? WHERE round_id=1 AND unit_id=?",
            (json.dumps(tampered), unit_id),
        )
        conn.commit()
        conn.close()

        with self.assertRaises(ReplayError) as ctx:
            replay_round(self.ledger_path, 1)
        error = ctx.exception
        self.assertEqual(error.code, "E_REPLAY")
        self.assertEqual(error.unit, unit_id)
        self.assertEqual(error.field, "targets")
        self.assertEqual(error.target, "db9.t9")


if __name__ == "__main__":
    unittest.main()
