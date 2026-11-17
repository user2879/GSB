"""Unit tests for spec validation, state machine, ledger and recovery."""

import json
import os
import shutil
import stat
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eventbus.bus import EventBus
from eventbus.errors import BusError, E_LEDGER, E_PROGRESS, E_SPEC, E_STATE
from eventbus.ledger import open_ledger
from eventbus.machine import SubscriberMachine
from eventbus.spec import load_specs

SOURCES = {
    "sources": [
        {
            "source_id": "src-a",
            "event_types": ["t1", "t2"],
            "event_count": 4,
            "event_seqs": [1, 2, 3, 4],
            "first_event_ms": 1000,
            "interval_ms": 10,
        }
    ]
}

SUBSCRIBERS = {
    "set_id": "subs",
    "set_version": 1,
    "subscribers": [
        {"subscriber_id": "sub-1", "event_types": ["t1"], "timeout_ms": 100},
        {"subscriber_id": "sub-2", "event_types": ["t1", "t2"], "timeout_ms": 200},
    ],
}


class WorkspaceMixin(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="eventbus-test-")
        self.sources_path = os.path.join(self.dir, "sources.json")
        self.subs_path = os.path.join(self.dir, "subscribers.json")
        self.ledger_path = os.path.join(self.dir, "ledger.db")
        self._write(self.sources_path, SOURCES)
        self._write(self.subs_path, SUBSCRIBERS)

    def tearDown(self):
        for name in os.listdir(self.dir):
            os.chmod(os.path.join(self.dir, name), 0o644)
        shutil.rmtree(self.dir, ignore_errors=True)

    @staticmethod
    def _write(path, doc):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(doc, handle)

    def load(self):
        return load_specs(self.sources_path, self.subs_path)

    def open(self, format_version=1):
        spec = self.load()
        return open_ledger(self.ledger_path, format_version, spec.set_id,
                           spec.set_version, spec.subscriber_snapshot)


class SpecValidationTest(WorkspaceMixin):
    def expect_spec_error(self, mutate, field_prefix):
        doc = json.loads(json.dumps(SOURCES))
        mutate(doc)
        self._write(self.sources_path, doc)
        with self.assertRaises(BusError) as ctx:
            self.load()
        self.assertEqual(ctx.exception.code, E_SPEC)
        self.assertTrue(ctx.exception.field.startswith(field_prefix),
                        ctx.exception.field)

    def test_duplicate_source_id(self):
        def mutate(doc):
            doc["sources"].append(dict(doc["sources"][0]))
        self.expect_spec_error(mutate, "sources[1].source_id")

    def test_non_monotonic_seqs(self):
        def mutate(doc):
            doc["sources"][0]["event_seqs"] = [1, 2, 2, 4]
        self.expect_spec_error(mutate, "sources[0].event_seqs")

    def test_seq_count_mismatch(self):
        def mutate(doc):
            doc["sources"][0]["event_count"] = 5
        self.expect_spec_error(mutate, "sources[0].event_seqs")

    def test_bad_timeout(self):
        doc = json.loads(json.dumps(SUBSCRIBERS))
        doc["subscribers"][0]["timeout_ms"] = 0
        self._write(self.subs_path, doc)
        with self.assertRaises(BusError) as ctx:
            self.load()
        self.assertEqual(ctx.exception.code, E_SPEC)
        self.assertEqual(ctx.exception.field, "subscribers[0].timeout_ms")

    def test_unknown_event_type(self):
        doc = json.loads(json.dumps(SUBSCRIBERS))
        doc["subscribers"][0]["event_types"] = ["nope"]
        self._write(self.subs_path, doc)
        with self.assertRaises(BusError) as ctx:
            self.load()
        self.assertEqual(ctx.exception.code, E_SPEC)
        self.assertEqual(ctx.exception.field, "subscribers[0].event_types[0]")

    def test_corrupt_source_file(self):
        with open(self.sources_path, "w", encoding="utf-8") as handle:
            handle.write('{"sources": [')
        with self.assertRaises(BusError) as ctx:
            self.load()
        self.assertEqual(ctx.exception.code, E_SPEC)


class StateMachineTest(unittest.TestCase):
    def test_illegal_transition(self):
        machine = SubscriberMachine("sub-1")
        with self.assertRaises(BusError) as ctx:
            machine.transition("committed", 1)
        self.assertEqual(ctx.exception.code, E_STATE)
        self.assertEqual(machine.state, "idle")

    def test_happy_path(self):
        machine = SubscriberMachine("sub-1")
        machine.transition("processing", 1)
        machine.transition("committed", 1)
        machine.transition("processing", 2)
        machine.transition("failed", 2)
        machine.transition("processing", 2)
        self.assertEqual(machine.state, "processing")


class LedgerTest(WorkspaceMixin):
    def test_duplicate_committed_rejected(self):
        ledger, _ = self.open()
        ledger.record_transition("sub-1", "src-a", 1, "idle", "processing", 1)
        ledger.commit_with_progress("sub-1", "src-a", 1, "processing", 2)
        with self.assertRaises(BusError) as ctx:
            ledger.commit_with_progress("sub-1", "src-a", 1, "processing", 3)
        self.assertEqual(ctx.exception.code, E_STATE)
        self.assertEqual(ledger.get_progress("sub-1"), 1)
        ledger.close()

    def test_progress_overflow_detected(self):
        ledger, _ = self.open()
        ledger.record_transition("sub-1", "src-a", 1, "idle", "processing", 1)
        ledger.commit_with_progress("sub-1", "src-a", 1, "processing", 2)
        ledger.close()
        import sqlite3
        connection = sqlite3.connect(self.ledger_path)
        connection.execute("UPDATE progress SET committed_seq=99 WHERE subscriber_id='sub-1'")
        connection.commit()
        connection.close()
        ledger, _ = self.open()
        with self.assertRaises(BusError) as ctx:
            ledger.check_progress_consistency()
        self.assertEqual(ctx.exception.code, E_PROGRESS)
        ledger.close()

    def test_corrupt_ledger_detected(self):
        ledger, _ = self.open()
        ledger.record_transition("sub-1", "src-a", 1, "idle", "processing", 1)
        ledger.close()
        import sqlite3
        connection = sqlite3.connect(self.ledger_path)
        connection.execute("UPDATE transitions SET to_state='bogus'")
        connection.commit()
        connection.close()
        with self.assertRaises(BusError) as ctx:
            self.open()
        self.assertEqual(ctx.exception.code, E_LEDGER)

    def test_version_bump_replaces_wholesale(self):
        ledger, _ = self.open()
        ledger.record_transition("sub-1", "src-a", 1, "idle", "processing", 1)
        ledger.commit_with_progress("sub-1", "src-a", 1, "processing", 2)
        ledger.close()
        doc = json.loads(json.dumps(SUBSCRIBERS))
        doc["set_version"] = 2
        self._write(self.subs_path, doc)
        ledger, replacement = self.open()
        self.assertIsNotNone(replacement)
        self.assertEqual(replacement.replaced["replaced_set_id"], "subs")
        self.assertEqual(replacement.replaced["set_version_old"], 1)
        self.assertEqual(replacement.replaced["set_version_new"], 2)
        self.assertTrue(os.path.exists(replacement.backup_path))
        mode = stat.S_IMODE(os.stat(replacement.backup_path).st_mode)
        self.assertEqual(mode & 0o222, 0)
        self.assertEqual(ledger.all_transitions(), [])
        self.assertIsNone(ledger.get_progress("sub-1"))
        ledger.close()


class RecoveryTest(WorkspaceMixin):
    def test_recovery_marks_unknown_and_reprocesses(self):
        spec = self.load()
        ledger, _ = self.open()
        bus = EventBus(spec, ledger)
        bus.recover()
        bus.process_all()
        ledger.record_transition("sub-1", "src-a", 3, "committed", "processing", 999)
        ledger.close()
        ledger, _ = self.open()
        bus = EventBus(spec, ledger)
        report = bus.recover()
        self.assertEqual(report.unknown_transitions, 1)
        self.assertEqual(report.progress["sub-1"], 3)
        bus.replay_check()
        commits = bus.process_all()
        self.assertEqual(commits, 0)
        ledger.close()

    def test_replay_check_passes_after_full_run(self):
        spec = self.load()
        ledger, _ = self.open()
        bus = EventBus(spec, ledger)
        bus.recover()
        bus.process_all()
        ledger.close()
        ledger, _ = self.open()
        bus = EventBus(spec, ledger)
        bus.recover()
        bus.replay_check()
        ledger.close()


if __name__ == "__main__":
    unittest.main()
