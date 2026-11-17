import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from graphidx import (
    BudgetError,
    GraphError,
    IndexIntegrityError,
    JsonLogger,
    MaintenanceSession,
)

from tests.helpers import SAMPLES, make_session, sample_changesets

ROOT = Path(__file__).resolve().parent.parent


class TestCorruptGraphInjection(unittest.TestCase):
    def test_corrupt_graph_file_raises_e_graph_and_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            corrupt = Path(tmp) / "corrupt.json"
            corrupt.write_text(
                (SAMPLES / "graph.json").read_text(encoding="utf-8") + "\n{corrupted",
                encoding="utf-8",
            )
            log = Path(tmp) / "inject.jsonl"
            logger = JsonLogger(log)
            with self.assertRaises(GraphError) as ctx:
                MaintenanceSession.from_files(
                    corrupt, SAMPLES / "indexes.json", SAMPLES / "views.json", logger=logger
                )
            err = ctx.exception
            self.assertEqual(err.code, "E_GRAPH")
            logger.error(err.code, err.message, injection="corrupt-graph", details=err.details)
            record = json.loads(log.read_text(encoding="utf-8").strip().splitlines()[-1])
            self.assertEqual(record["code"], "E_GRAPH")
            self.assertEqual(record["injection"], "corrupt-graph")

    def test_semantically_invalid_graph_raises_e_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text(json.dumps({
                "nodes": [{"id": 1, "label": "A"}, {"id": 1, "label": "B"}],
                "edges": [],
            }), encoding="utf-8")
            with self.assertRaises(GraphError):
                MaintenanceSession.from_files(bad, SAMPLES / "indexes.json", SAMPLES / "views.json")


class TestTamperedIndexInjection(unittest.TestCase):
    def test_tampered_snapshot_entry_raises_e_index_and_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp) / "snap.json"
            session = make_session()
            session.save_snapshot(snapshot)
            snap = json.loads(snapshot.read_text(encoding="utf-8"))
            snap["entries"]["by_label"]["Person"]["digest"] = "sha256:" + "f" * 64
            snapshot.write_text(json.dumps(snap), encoding="utf-8")
            log = Path(tmp) / "inject.jsonl"
            logger = JsonLogger(log)
            with self.assertRaises(IndexIntegrityError) as ctx:
                MaintenanceSession.from_snapshot(snapshot, logger=logger)
            err = ctx.exception
            self.assertEqual(err.code, "E_INDEX")
            logger.error(err.code, err.message, injection="tamper-index", details=err.details)
            record = json.loads(log.read_text(encoding="utf-8").strip().splitlines()[-1])
            self.assertEqual(record["code"], "E_INDEX")


class TestZeroBudgetInjection(unittest.TestCase):
    def test_zero_budget_raises_e_budget_and_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "inject.jsonl"
            logger = JsonLogger(log)
            session = make_session(budget=0, logger=logger)
            with self.assertRaises(BudgetError) as ctx:
                session.apply_changeset(sample_changesets()[0])
            err = ctx.exception
            self.assertEqual(err.code, "E_BUDGET")
            logger.error(err.code, err.message, injection="zero-budget", details=err.details)
            record = json.loads(log.read_text(encoding="utf-8").strip().splitlines()[-1])
            self.assertEqual(record["code"], "E_BUDGET")


class TestCli(unittest.TestCase):
    def run_cli(self, *argv):
        return subprocess.run(
            [sys.executable, str(ROOT / "demo.py"), *argv],
            capture_output=True, text=True, cwd=ROOT,
        )

    def test_cli_inject_commands_exit_zero_and_report_codes(self):
        expected = {
            "corrupt-graph": "E_GRAPH",
            "tamper-index": "E_INDEX",
            "zero-budget": "E_BUDGET",
        }
        for kind, code in expected.items():
            proc = self.run_cli("inject", kind)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn(code, proc.stdout)

    def test_cli_full_flow(self):
        proc = self.run_cli("all")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn('"status": "ok"', proc.stdout)


if __name__ == "__main__":
    unittest.main()
