import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from degprop.engine import DiagnosticsEngine
from degprop.errors import ReplayError, SnapshotError
from degprop.ledger import Ledger
from degprop.loader import load_topology, topology_digest

ROOT = Path(__file__).resolve().parents[1]
TOPO = ROOT / "samples" / "topology.json"
EVENTS = ROOT / "samples" / "events.ndjson"


class LedgerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ledger_path = Path(self.tmp.name) / "session.db"

    def tearDown(self):
        self.tmp.cleanup()

    def test_batch_persisted_and_replay_verified(self):
        with DiagnosticsEngine(TOPO, EVENTS, self.ledger_path) as engine:
            engine.run_batch(engine.events[:8], 1)
        with DiagnosticsEngine(TOPO, EVENTS, self.ledger_path) as engine:
            result = engine.resume_and_replay()
        self.assertTrue(result["verified"])
        self.assertEqual(result["replayed"], 8)

    def test_digest_mismatch_replaces_whole_snapshot(self):
        with DiagnosticsEngine(TOPO, EVENTS, self.ledger_path) as engine:
            engine.run_batch(engine.events[:8], 1)
            digest = engine.digest
        topology = load_topology(TOPO)
        with self.assertRaises(SnapshotError) as ctx:
            Ledger.open_or_replace(self.ledger_path,
                                   topology.topology_version,
                                   digest + "ff", raise_on_mismatch=True)
        self.assertEqual(ctx.exception.code, "E_SNAPSHOT")
        # 异常路径不删库；随后以替换模式用错误摘要重开，触发整体替换。
        ledger, replaced = Ledger.open_or_replace(
            self.ledger_path, topology.topology_version,
            digest + "ff", raise_on_mismatch=False)
        self.assertTrue(replaced["replaced"])
        self.assertEqual(ledger.cursor()["last_seq"], 0)
        ledger.close()

    def test_format_version_mismatch_replaces(self):
        with DiagnosticsEngine(TOPO, EVENTS, self.ledger_path) as engine:
            engine.run_batch(engine.events[:8], 1)
        conn = sqlite3.connect(self.ledger_path)
        conn.execute("UPDATE meta SET value='999' WHERE key='format_version'")
        conn.commit()
        conn.close()
        topology = load_topology(TOPO)
        ledger, replaced = Ledger.open_or_replace(
            self.ledger_path, topology.topology_version,
            topology_digest(topology))
        self.assertTrue(replaced["replaced"])
        self.assertIn("format_version", replaced["reason"])
        ledger.close()

    def test_replay_mismatch_locates_first_event_and_field(self):
        with DiagnosticsEngine(TOPO, EVENTS, self.ledger_path) as engine:
            engine.run_batch(engine.events[:8], 1)
        conn = sqlite3.connect(self.ledger_path)
        conn.execute(
            "UPDATE call_outcomes SET bucket=1 WHERE seq=2")
        conn.commit()
        conn.close()
        with self.assertRaises(ReplayError) as ctx:
            with DiagnosticsEngine(TOPO, EVENTS, self.ledger_path) as engine:
                engine.resume_and_replay()
        self.assertEqual(ctx.exception.code, "E_REPLAY")
        self.assertEqual(ctx.exception.line, 2)
        self.assertEqual(ctx.exception.field, "bucket")

    def test_batch_digest_written_in_single_transaction(self):
        with DiagnosticsEngine(TOPO, EVENTS, self.ledger_path) as engine:
            engine.run_batch(engine.events[:8], 1)
        conn = sqlite3.connect(self.ledger_path)
        rows = conn.execute("SELECT key FROM meta WHERE key LIKE 'batch_digest.%'").fetchall()
        self.assertEqual(len(rows), 1)
        outcomes = conn.execute("SELECT COUNT(*) FROM call_outcomes").fetchone()[0]
        self.assertEqual(outcomes, 8)
        conn.close()


if __name__ == "__main__":
    unittest.main()
