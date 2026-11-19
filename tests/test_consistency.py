import tempfile
import unittest
from pathlib import Path

from rexapp.app import build_engine
from rexapp.commands import apply_command, load_commands
from rexapp.config import load_config
from rexapp.documents import Document, load_documents

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def full_state(config, documents, failures, tmp_dir):
    engine = build_engine(config, documents, failures, Path(tmp_dir) / "snap")
    engine.run_full()
    return engine.export_results(), engine.build_report()["failures"]


class IncrementalConsistencyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base_config = load_config(SAMPLES / "config.json")
        loaded = load_documents(SAMPLES / "documents.jsonl", set(self.base_config.entity_vocab))
        self.documents = {doc.doc_id: doc for doc in loaded.documents}
        self.failures = loaded.failures
        self.commands = load_commands(SAMPLES / "commands.json")

    def tearDown(self):
        self.tmp.cleanup()

    def test_each_command_matches_cold_full_run(self):
        config = self.base_config
        documents = dict(self.documents)
        engine = build_engine(config, documents, self.failures, Path(self.tmp.name) / "inc")
        for command in self.commands:
            new_config, new_documents, effect = apply_command(command, config, documents)
            response = engine.apply_interaction(effect.command_id, effect, new_config, new_documents)
            if not response.latest:
                engine.drain_pending(effect.affected_docs)
            config, documents = new_config, new_documents

            cold_results, cold_failures = full_state(
                config, documents, self.failures, tempfile.mkdtemp(dir=self.tmp.name)
            )
            self.assertEqual(engine.export_results(), cold_results, f"mismatch after {command['command_id']}")
            self.assertEqual(
                engine.build_report()["failures"], cold_failures, f"failure mismatch after {command['command_id']}"
            )

    def test_untouched_stage_is_reused_not_recomputed(self):
        config = self.base_config
        documents = dict(self.documents)
        engine = build_engine(config, documents, self.failures, Path(self.tmp.name) / "reuse")
        engine.run_full()
        recomputed_before = engine.cache.recomputed_count
        command = self.commands[2]  # event threshold change -> only confidence stage
        new_config, new_documents, effect = apply_command(command, config, documents)
        engine.apply_interaction(effect.command_id, effect, new_config, new_documents)
        audit = [
            entry
            for entry in engine.cache.audit
            if entry.stage == "entity" and entry.doc_id in effect.affected_docs
        ]
        self.assertTrue(any(entry.action == "reuse" for entry in audit))
        self.assertGreater(engine.cache.reused_count, 0)
        self.assertEqual(engine.cache.recomputed_count, recomputed_before + len(documents))

    def test_empty_result_and_all_failure_edges(self):
        config = self.base_config
        empty: dict[str, Document] = {}
        engine = build_engine(config, empty, [], Path(self.tmp.name) / "empty")
        outcome = engine.run_full()
        self.assertEqual(outcome["results"], [])
        cold, _ = full_state(config, empty, [], tempfile.mkdtemp(dir=self.tmp.name))
        self.assertEqual(engine.export_results(), cold)

    def test_single_document_repeated_replacement(self):
        config = self.base_config
        source = list(self.documents.values())[1]
        initial = Document(source.doc_id, source.words, source.positions, source.text)
        one_doc = {initial.doc_id: initial}
        engine = build_engine(config, one_doc, [], Path(self.tmp.name) / "replace")
        engine.run_full()
        for round_index in range(3):
            words = tuple(list(source.words) + [f"v{round_index}"])
            positions = tuple((index, index + 1) for index in range(len(words)))
            replaced = Document(source.doc_id, words, positions, " ".join(words))
            engine.documents[replaced.doc_id] = replaced
            engine.cache.invalidate_document(replaced.doc_id, "test")
            engine._results.pop(replaced.doc_id, None)
            engine.run_full()
            cold, _ = full_state(config, {replaced.doc_id: replaced}, [], tempfile.mkdtemp(dir=self.tmp.name))
            self.assertEqual(engine.export_results(), cold)


if __name__ == "__main__":
    unittest.main()
