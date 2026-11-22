import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from configreg.collect import SessionParams
from configreg.errors import ReplayError
from configreg.ledger import Ledger, SessionIdentity
from configreg.manifest import load_manifest_pair
from configreg.session import Session


def params(seed: int = 777, precision: int = 4, **overrides) -> SessionParams:
    defaults = dict(
        sample_seed=seed,
        precision=precision,
        global_concurrency=2,
        per_node_concurrency=1,
        socket_timeout_ms=800,
        max_attempts=5,
        batch_size=2,
        max_elapsed_delta_ms=50,
        max_mismatch_nodes=0,
    )
    defaults.update(overrides)
    return SessionParams(**defaults)


class EndToEndSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.baseline, self.candidate, self.registry = load_manifest_pair(
            "data/baseline.json", "data/candidate.json", "data/registry.json"
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _session(self, ledger_name: str = "ledger.sqlite3", **overrides) -> Session:
        return Session(
            self.baseline,
            self.candidate,
            self.registry,
            params(**overrides),
            ledger_path=self.root / ledger_name,
            log_dir=self.root / "logs",
        )

    def test_full_run_writes_atomic_batches_and_replays(self) -> None:
        session = self._session()
        result = session.run()
        self.assertTrue(result.regression_report.regression_items)
        ledger_path = self.root / "ledger.sqlite3"
        conn = sqlite3.connect(ledger_path)
        batches = conn.execute("SELECT batch_id FROM batches ORDER BY batch_id").fetchall()
        self.assertGreaterEqual(len(batches), 2)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM snapshot").fetchone()[0], 1)
        conn.close()

        verifier = self._session()
        replay = verifier.replay_and_verify()
        self.assertTrue(replay["replayed"])
        self.assertIsNone(replay["mismatch"])
        verifier.ledger.close()

    def test_identity_change_replaces_entire_state(self) -> None:
        first = self._session(precision=4)
        first.run()
        first.ledger.close()
        second = self._session(precision=6)
        replacement = second.ensure_identity()
        self.assertTrue(replacement["replaced"])
        self.assertIn("base_rate", replacement["affected_items"])
        self.assertEqual(replacement["affected_nodes"], self.registry.node_ids())
        self.assertEqual(second.ledger.committed_batches(), [])
        events = second.ledger.list_replacements()
        self.assertEqual(len(events), 1)
        second.ledger.close()

    def test_tampered_ledger_raises_replay_with_location(self) -> None:
        session = self._session()
        session.run()
        session.ledger.close()
        ledger_path = self.root / "ledger.sqlite3"
        conn = sqlite3.connect(ledger_path)
        row = conn.execute("SELECT batch_id, item_key, payload FROM item_verdicts LIMIT 1").fetchone()
        payload = json.loads(row[2])
        payload["status"] = "ok" if payload["status"] == "E_REGRESSION" else "E_REGRESSION"
        conn.execute(
            "UPDATE item_verdicts SET payload=? WHERE batch_id=? AND item_key=?",
            (json.dumps(payload, sort_keys=True), row[0], row[1]),
        )
        conn.commit()
        conn.close()

        verifier = self._session()
        with self.assertRaises(ReplayError) as ctx:
            verifier.replay_and_verify()
        self.assertEqual(ctx.exception.code, "E_REPLAY")
        mismatch = ctx.exception.first_mismatch
        self.assertEqual(mismatch["item"], row[1])
        self.assertEqual(mismatch["field"], "status")
        verifier.ledger.close()

    def test_forced_termination_resume_marks_unknown_and_reruns(self) -> None:
        crashing = self._session(batch_size=2)
        crashing.crash_after_batches = 1
        with self.assertRaises(KeyboardInterrupt):
            crashing.run()
        crashing.ledger.close()

        conn = sqlite3.connect(self.root / "ledger.sqlite3")
        committed_items = {row[0] for row in conn.execute("SELECT DISTINCT item_key FROM diff_records")}
        self.assertEqual(len(conn.execute("SELECT * FROM batches").fetchall()), 1)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM snapshot").fetchone()[0], 0)
        conn.close()

        resumed = self._session(batch_size=2)
        result = resumed.run()
        self.assertTrue(result.unknown_items)
        self.assertNotIn(result.unknown_items[0], committed_items)
        conn = sqlite3.connect(self.root / "ledger.sqlite3")
        all_items = {row[0] for row in conn.execute("SELECT DISTINCT item_key FROM diff_records")}
        self.assertEqual(all_items, set(self.baseline.items) | set(self.candidate.items))
        conn.close()

        verifier = self._session(batch_size=2)
        replay = verifier.replay_and_verify()
        self.assertIsNone(replay["mismatch"])
        verifier.ledger.close()


class LedgerIdentityTests(unittest.TestCase):
    def test_identity_mismatch_fields(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            ledger = Ledger(Path(name) / "x.sqlite3")
            identity = SessionIdentity(1, 2, "a", "b", 4, 1, 7)
            ledger.initialize_identity(identity)
            changed = SessionIdentity(1, 2, "a", "b", 6, 1, 7)
            ok, fields = ledger.identity_matches(changed)
            self.assertFalse(ok)
            self.assertEqual(fields, ["precision"])
            ledger.close()


if __name__ == "__main__":
    unittest.main()
