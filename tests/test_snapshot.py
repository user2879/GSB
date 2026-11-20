from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from isolation_lab.errors import BusinessError
from isolation_lab.locks import LockManager
from isolation_lab.models import SchedulerParams
from isolation_lab.scheduler import Engine, compare_run_outcomes
from isolation_lab.snapshot import replay_snapshot, restore_snapshot, save_snapshot
from isolation_lab.scripts import load_script
from isolation_lab.storage import SnapshotStore, Storage


ROOT = Path(__file__).resolve().parent.parent


def run_and_snapshot(snapshot_file: Path, script_file: Path, snapshot_id: str, limit: int = 2):
    script = load_script(script_file)
    storage = Storage(":memory:")
    result = Engine(storage, SchedulerParams(global_concurrency_limit=limit)).run(script)
    store = SnapshotStore(snapshot_file)
    save_snapshot(store, snapshot_id, script, result)
    return script, result, storage, store


class SnapshotTests(unittest.TestCase):
    def test_every_sample_snapshot_replays(self) -> None:
        samples = (
            "current-v3.json",
            "legacy-v1.json",
            "legacy-v2.json",
            "lock-timeout-v3.json",
            "deadlock-v3.json",
        )
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_file = Path(tmp) / "snapshots.sqlite"
            for index, filename in enumerate(samples):
                snapshot_id = f"sample-{index}"
                _script, _result, storage, store = run_and_snapshot(
                    snapshot_file,
                    ROOT / "samples" / filename,
                    snapshot_id,
                )
                report, replay = replay_snapshot(store, snapshot_id)
                self.assertTrue(report.reproducible, f"non-reproducible: {filename}")
                self.assertIsNone(report.session_id)
                self.assertIsNone(report.op_no)
                storage.close()
                store.close()

    def test_restore_replaces_state_and_clears_live_locks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_file = Path(tmp) / "snapshots.sqlite"
            script, result, storage, store = run_and_snapshot(
                snapshot_file,
                ROOT / "samples" / "current-v3.json",
                "normal-final",
            )
            storage.close()

            target = Storage(":memory:")
            timeout_script = load_script(ROOT / "samples" / "lock-timeout-v3.json")
            Engine(
                target,
                SchedulerParams(global_concurrency_limit=1, clock_quantum_ms=2),
            ).run(timeout_script)
            target_state_before = target.get_context()

            live_locks = LockManager()
            acquired = live_locks.acquire(
                session_id="ghost",
                row_key="acct:alpha",
                mode="X",
                isolation_level="repeatable_read",
                now=0,
                op_no=1,
                op_started_virtual_ms=0,
                remaining_budget_ms=10,
                operation_timeout_ms=10,
            )
            self.assertEqual(acquired.status, "granted")
            self.assertEqual(live_locks.entry_count(), 1)

            report = restore_snapshot(
                store,
                "normal-final",
                target_storage=target,
                target_lock_manager=live_locks,
            )
            self.assertEqual(report.mode, "replace")
            self.assertFalse(report.script_digest_match)
            self.assertFalse(report.param_digest_match)
            self.assertTrue(report.parameter_version_match)
            self.assertEqual(report.cleared_lock_entries, 1)
            self.assertEqual(live_locks.entry_count(), 0)
            self.assertIn("clock_quantum_ms", report.changed_parameter_items)
            self.assertIn("global_concurrency_limit", report.changed_parameter_items)
            self.assertTrue(report.changed_sessions)
            self.assertEqual(report.data_version, result.global_data_version)
            self.assertEqual(target.all_rows(), result.final_rows)
            context = target.get_context()
            self.assertEqual(context["script_id"], result.script_id)
            self.assertEqual(context["outcomes"], [item.to_dict() for item in result.outcomes])
            self.assertEqual(len(target.read_ledger(run_id=result.run_id)), len(result.trace))
            self.assertNotEqual(context["script_digest"], target_state_before["script_digest"])
            target.close()
            store.close()

    def test_replay_reports_first_mismatched_session_operation_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_file = Path(tmp) / "snapshots.sqlite"
            script, result, storage, store = run_and_snapshot(
                snapshot_file,
                ROOT / "samples" / "current-v3.json",
                "corrupt-me",
            )
            storage.close()
            snapshot = store.load("corrupt-me")
            snapshot["outcomes"][0]["operations"][0]["value"] = 999
            store.save(snapshot)
            report, _replay = replay_snapshot(store, "corrupt-me", raise_on_error=False)
            self.assertFalse(report.reproducible)
            self.assertEqual(report.session_id, "N1")
            self.assertEqual(report.op_no, 1)
            self.assertIn("operations[0].value", report.field)
            self.assertEqual(report.expected, 999)
            self.assertEqual(report.actual, 10)

            with self.assertRaises(BusinessError) as captured:
                replay_snapshot(store, "corrupt-me", raise_on_error=True)
            error = captured.exception
            self.assertEqual(error.code, "E_REPLAY")
            self.assertEqual(error.session_id, "N1")
            self.assertEqual(error.op_no, 1)
            self.assertIn("operations[0].value", error.path)
            self.assertEqual(error.details["snapshot_id"], "corrupt-me")
            store.close()

    def test_ledger_reads_records_from_all_script_versions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "ledger.sqlite"
            storage = Storage(db)
            for filename in ("legacy-v1.json", "legacy-v2.json", "current-v3.json"):
                script = load_script(ROOT / "samples" / filename)
                result = Engine(storage, SchedulerParams(global_concurrency_limit=2)).run(script)
                self.assertGreater(len(result.trace), 0)
            records = storage.read_ledger()
            self.assertEqual({record.source_version for record in records}, {1, 2, 3})
            self.assertEqual({record.normalized_version for record in records}, {3})
            self.assertEqual(
                [record.trace_event_seq for record in records[:5]],
                [1, 2, 3, 4, 5],
            )
            run_ids = {record.run_id for record in records}
            self.assertEqual(len(run_ids), 3)
            filtered = storage.read_ledger(run_id=records[-1].run_id)
            self.assertTrue(all(record.run_id == records[-1].run_id for record in filtered))
            storage.close()


if __name__ == "__main__":
    unittest.main()
