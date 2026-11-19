from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from srebudget.clock import LogicalClock
from srebudget.engine import DiagnosticEngine
from srebudget.errors import SreError
from srebudget.loader import load_chain, load_trace
from srebudget.replay import ReplayVerifier
from srebudget.runner import BatchRunner
from srebudget.samples import write_samples
from srebudget.storage import SessionStore


class SnapshotReplayTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tempdir.name)
        chain_path, trace_path = write_samples(self.data_dir)
        self.chain = load_chain(chain_path)
        self.trace = load_trace(trace_path, self.chain)
        self.db_path = self.data_dir / "session.sqlite3"

    def tearDown(self):
        self.tempdir.cleanup()

    def test_snapshot_whole_replacement_and_conclusion_invalidation(self):
        with SessionStore(self.db_path) as store:
            session_id = store.create_session(self.chain, self.trace, session_id="snap")
            clock = LogicalClock(0)
            runner = BatchRunner(
                store,
                session_id,
                self.chain,
                self.trace,
                clock=clock,
                follow_trace_time=True,
            )
            runner.run(max_batches=1, batch_size=2)
            snapshot = runner.create_snapshot()
            self.assertEqual(snapshot.cursor, 2)
            runner.run(batch_size=4)
            self.assertEqual(runner.engine.cursor, 12)
            restore = store.restore_snapshot(
                snapshot.snapshot_id, runner.engine, self.chain, self.trace
            )
            self.assertEqual(restore.cursor, 2)
            self.assertEqual(restore.invalidated_conclusions, 10)
            self.assertEqual(restore.affected_services, tuple(sorted(restore.affected_services)))
            self.assertEqual(
                runner.engine.export_state(), snapshot.state
            )
            valid = store.list_conclusions(session_id, valid_only=True)
            self.assertEqual(len(valid), 2)
            self.assertEqual(
                [row["call_index"] for row in valid], [1, 2]
            )

            resumed = BatchRunner(
                store,
                session_id,
                self.chain,
                self.trace,
                clock=LogicalClock(0),
                follow_trace_time=True,
            )
            self.assertEqual(resumed.engine.cursor, 2)
            outcome = resumed.run(batch_size=4)
            self.assertEqual(outcome.processed_calls, 10)
            self.assertEqual(resumed.engine.cursor, 12)

            store.corrupt_snapshot_digest(snapshot.snapshot_id)
            with self.assertRaises(SreError) as raised:
                store.load_snapshot(snapshot.snapshot_id)
            self.assertEqual(raised.exception.code, "E_SNAPSHOT")
            self.assertEqual(raised.exception.details["field_path"], "content_digest")

    def test_chain_change_invalidates_all_old_conclusions(self):
        with SessionStore(self.db_path) as store:
            session_id = store.create_session(self.chain, self.trace, session_id="chain")
            clock = LogicalClock(0)
            runner = BatchRunner(
                store,
                session_id,
                self.chain,
                self.trace,
                clock=clock,
                follow_trace_time=True,
            )
            runner.run(batch_size=4)
            snapshot = runner.create_snapshot()

            changed_data = self.chain.canonical_dict()
            changed_data["version"] = "chain-2026.11.20-r2"
            changed = __import__("srebudget.loader", fromlist=["load_chain_data"]).load_chain_data(changed_data)
            changed_engine = DiagnosticEngine(changed, LogicalClock(0))
            result = store.restore_snapshot(
                snapshot.snapshot_id, changed_engine, changed, self.trace
            )
            self.assertEqual(result.mode, "whole_state_reset")
            self.assertEqual(result.reason, "chain_version_changed")
            self.assertEqual(result.cursor, 0)
            self.assertEqual(result.invalidated_conclusions, len(self.trace))
            self.assertEqual(
                store.list_conclusions(session_id, valid_only=True), []
            )
            self.assertEqual(
                store.get_session(session_id)["chain_digest"], changed.digest
            )


class CrashRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tempdir.name)
        chain_path, trace_path = write_samples(self.data_dir)
        self.chain = load_chain(chain_path)
        self.trace = load_trace(trace_path, self.chain)
        self.db_path = self.data_dir / "crash.sqlite3"

    def tearDown(self):
        self.tempdir.cleanup()

    def test_uncommitted_batch_is_unknown_and_rerun_from_committed_cursor(self):
        with SessionStore(self.db_path) as store:
            session_id = store.create_session(self.chain, self.trace, session_id="crash")
            runner = BatchRunner(
                store,
                session_id,
                self.chain,
                self.trace,
                clock=LogicalClock(0),
                follow_trace_time=True,
            )
            runner.run(max_batches=1, batch_size=2)
            store.start_batch_attempt(session_id, start_call=3, end_call_exclusive=5)
            self.assertEqual(store.get_session(session_id)["committed_cursor"], 2)

        with SessionStore(self.db_path) as reopened:
            runner = BatchRunner(
                reopened,
                session_id,
                self.chain,
                self.trace,
                clock=LogicalClock(0),
                follow_trace_time=True,
            )
            report = runner.recovery
            self.assertEqual(report.committed_cursor, 2)
            self.assertTrue(report.rerun_required)
            self.assertEqual(len(report.recovered_batches), 1)
            self.assertEqual(len(report.unknown_batches), 1)
            self.assertEqual(
                report.unknown_batches[0]["unknown_call_indexes"], [3, 4]
            )
            outcome = runner.run(batch_size=5)
            self.assertEqual(outcome.processed_calls, 10)
            self.assertEqual(runner.engine.cursor, 12)
            self.assertEqual(
                len(reopened.list_conclusions(session_id, valid_only=True)),
                12,
            )
            self.assertEqual(len(reopened.list_unknown(session_id)), 1)


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tempdir.name)
        chain_path, trace_path = write_samples(self.data_dir)
        self.chain = load_chain(chain_path)
        self.trace = load_trace(trace_path, self.chain)
        self.db_path = self.data_dir / "replay.sqlite3"

    def tearDown(self):
        self.tempdir.cleanup()

    def _snapshot(self, prefix: int = 5):
        with SessionStore(self.db_path) as store:
            session_id = store.create_session(self.chain, self.trace, session_id="replay")
            clock = LogicalClock(0)
            engine = DiagnosticEngine(self.chain, clock)
            for call in self.trace.calls[:prefix]:
                clock.follow(call.timestamp_ms)
                engine.process_call(call)
            return store.create_snapshot(session_id, engine, self.chain, self.trace)

    def test_database_snapshot_replays_every_remaining_call(self):
        snapshot = self._snapshot(5)
        with SessionStore(self.db_path) as store:
            loaded = store.load_snapshot(snapshot.snapshot_id)
            report = ReplayVerifier(self.chain, self.trace, loaded).run()
        self.assertEqual(report.compared_calls, 7)
        self.assertEqual(report.final_cursor, 12)

    def test_divergence_localizes_first_call_service_and_field(self):
        snapshot = self._snapshot(5)
        modified_state = __import__("copy").deepcopy(snapshot.state)
        for item in modified_state["services"]:
            if item["id"] == "orders":
                item["open_count"] = 99
        modified = replace(snapshot, state=modified_state)
        with self.assertRaises(SreError) as raised:
            ReplayVerifier(self.chain, self.trace, modified).run()
        error = raised.exception
        self.assertEqual(error.code, "E_REPLAY")
        self.assertEqual(error.details["kind"], "divergence")
        self.assertEqual(error.details["call_index"], 6)
        self.assertEqual(error.details["service"], "orders")
        self.assertEqual(error.details["field"], "circuit.open_count")

    def test_window_parameter_change_is_reported_as_incompatibility(self):
        snapshot = self._snapshot(5)
        data = self.chain.canonical_dict()
        for item in data["services"]:
            if item["id"] == "db":
                item["budget"]["ratio_bp"] = 4000
        changed = __import__("srebudget.loader", fromlist=["load_chain_data"]).load_chain_data(data)
        with self.assertRaises(SreError) as raised:
            ReplayVerifier(changed, self.trace, snapshot).run()
        self.assertEqual(raised.exception.code, "E_REPLAY")
        self.assertEqual(raised.exception.details["kind"], "incompatible")
        self.assertEqual(raised.exception.details["basis"], "window_parameters_changed")


if __name__ == "__main__":
    unittest.main()
