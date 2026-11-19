import json
import tempfile
import time
import unittest
from pathlib import Path

from rexapp.app import build_engine, run_interactive
from rexapp.commands import apply_command, load_commands
from rexapp.config import load_config
from rexapp.documents import load_documents
from rexapp.errors import ErrorCode

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


class ConcurrencyTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config(SAMPLES / "config.json")
        loaded = load_documents(SAMPLES / "documents.jsonl", set(self.config.entity_vocab))
        self.documents = {doc.doc_id: doc for doc in loaded.documents}
        self.failures = loaded.failures

    def test_threadpool_matches_single_thread_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            parallel = build_engine(self.config, self.documents, self.failures, Path(tmp) / "p")
            parallel.run_full()
            single_cfg = self.config.with_updates(thread_pool_workers=1)
            sequential = build_engine(single_cfg, self.documents, self.failures, Path(tmp) / "s")
            sequential.run_full()
            self.assertEqual(parallel.export_results(), sequential.export_results())
            self.assertEqual(parallel.build_report()["failures"], sequential.build_report()["failures"])

    def test_sort_key_is_stable(self):
        engine = build_engine(self.config, self.documents, self.failures, Path(tempfile.mkdtemp()) / "x")
        keys = [engine.sort_key(doc_id, "confidence") for doc_id in sorted(self.documents)]
        self.assertEqual(keys, sorted(keys))


class BudgetTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config(SAMPLES / "config.json")
        loaded = load_documents(SAMPLES / "documents.jsonl", set(self.config.entity_vocab))
        self.documents = {doc.doc_id: doc for doc in loaded.documents}
        self.commands = load_commands(SAMPLES / "commands.json")

    def test_pending_response_when_budget_exceeded(self):
        tight = self.config.with_updates(interaction_budget_ms=0)

        def slow_hook(doc_id: str, stage: str) -> None:
            time.sleep(0.01)

        with tempfile.TemporaryDirectory() as tmp:
            engine = build_engine(tight, self.documents, [], Path(tmp) / "b", work_hook=slow_hook)
            command = self.commands[0]
            new_config, new_documents, effect = apply_command(command, tight, self.documents)
            response = engine.apply_interaction("budget-cmd", effect, new_config, new_documents)
            self.assertFalse(response.latest)
            self.assertEqual(response.code, ErrorCode.E_INTERACT_BUDGET_PENDING)
            self.assertEqual(response.status, "pending")
            pending = engine.drain_pending(effect.affected_docs)
            self.assertEqual(pending, [])

    def test_latest_response_not_faked(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = build_engine(self.config, self.documents, [], Path(tmp) / "l")
            engine.run_full()
            command = self.commands[2]
            new_config, new_documents, effect = apply_command(command, self.config, self.documents)
            response = engine.apply_interaction("latest-cmd", effect, new_config, new_documents)
            self.assertTrue(response.latest)
            self.assertIsNone(response.code)


class SnapshotAndExitTest(unittest.TestCase):
    def test_snapshot_version_changes_and_exit_code_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            outcome = run_interactive(SAMPLES, Path(tmp))
            self.assertEqual(outcome["exit_code"], 0)
            snapshot_path = Path(tmp) / "snapshots" / "snapshot.json"
            self.assertTrue(snapshot_path.exists())
            payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
            self.assertIn("results", payload)
            self.assertIn("cache", payload)

    def test_atomic_snapshot_skips_unchanged_content(self):
        from rexapp.snapshots import SnapshotStore

        with tempfile.TemporaryDirectory() as tmp:
            store = SnapshotStore(Path(tmp))
            first = store.commit({"a": 1})
            second = store.commit({"a": 1})
            third = store.commit({"a": 2})
            self.assertTrue(first["changed"])
            self.assertFalse(second["changed"])
            self.assertTrue(third["changed"])
            self.assertEqual(third["version"], 2)


if __name__ == "__main__":
    unittest.main()
