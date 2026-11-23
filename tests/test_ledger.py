from __future__ import annotations

import json
import tempfile
import unittest

from mvcc_workbench.engine import DiagnosticEngine
from mvcc_workbench.errors import LedgerError
from mvcc_workbench.ledger import Ledger
from mvcc_workbench.loader import load_spec
from mvcc_workbench.model import conclusion_digest


class LedgerTests(unittest.TestCase):
    def test_commit_and_recover(self) -> None:
        spec = load_spec("data/demo_small.json")
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/ledger.sqlite3"
            ledger = Ledger(path)
            result = DiagnosticEngine(spec, ledger=ledger).run(round_id="complete-1")
            recovered = ledger.recover()
            self.assertEqual(recovered.code, "OK_RECOVERED")
            self.assertFalse(recovered.requires_rerun)
            self.assertEqual(recovered.ledger_cursor, 1)
            self.assertEqual(
                recovered.latest_complete.conclusion_digest,
                conclusion_digest(result.results),
            )
            self.assertEqual(len(recovered.latest_complete.results), 6)
            ledger.close()

    def test_pending_round_is_unknown_after_forced_exit(self) -> None:
        spec = load_spec("data/demo_small.json")
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/ledger.sqlite3"

            class SimulatedForcedExit(BaseException):
                pass

            ledger = Ledger(path)
            engine = DiagnosticEngine(spec, ledger=ledger)

            def crash() -> None:
                raise SimulatedForcedExit()

            with self.assertRaises(SimulatedForcedExit):
                engine.run(round_id="crashed-round", crash_hook=crash)
            ledger.close()

            reopened = Ledger(path)
            recovery = reopened.recover()
            self.assertEqual(recovery.code, "OK_RERUN_REQUIRED")
            self.assertTrue(recovery.requires_rerun)
            self.assertEqual(recovery.latest_complete, None)
            self.assertEqual(len(recovery.unknown_rounds), 1)
            self.assertEqual(
                recovery.unknown_rounds[0]["status"], "UNKNOWN_RERUN_REQUIRED"
            )
            self.assertEqual(len(recovery.unknown_rounds[0]["query_ids"]), 6)
            reopened.close()

    def test_content_digest_tamper_is_detected(self) -> None:
        spec = load_spec("data/demo_small.json")
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/ledger.sqlite3"
            ledger = Ledger(path)
            DiagnosticEngine(spec, ledger=ledger).run(round_id="tamper-round")
            ledger._conn.execute(
                "UPDATE rounds SET plan_json=? WHERE round_id=?",
                (json.dumps({"tampered": True}), "tamper-round"),
            )
            with self.assertRaises(LedgerError) as raised:
                ledger.load_latest_committed()
            self.assertEqual(raised.exception.code, "E_LEDGER")
            ledger.close()


if __name__ == "__main__":
    unittest.main()
