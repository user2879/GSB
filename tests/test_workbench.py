from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from imv.errors import BusinessError
from imv.store import LedgerStore
from imv.workbench import Workbench


ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"


class WorkbenchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _workbench(self, name: str, **kwargs) -> Workbench:
        return Workbench(
            SAMPLES / "base_table.json",
            SAMPLES / "views.json",
            SAMPLES / "changes.ndjson",
            self.root / name,
            start_watchdog=kwargs.pop("start_watchdog", False),
            auto_rebuild_on_expiry=kwargs.pop("auto_rebuild_on_expiry", False),
            **kwargs,
        )

    def test_incremental_is_byte_identical_across_batch_boundaries(self) -> None:
        signatures = []
        for batch_size in (1, 2, 3, 10):
            workbench = self._workbench(f"boundary-{batch_size}", reset=True)
            try:
                while workbench.cursor < len(workbench.changes.changes):
                    report = workbench.advance_batch(batch_size)
                    self.assertEqual(report["status"], "committed")
                authority = workbench.authoritative_recompute()
                self.assertTrue(authority["byte_identical"])
                signatures.append(
                    {
                        view_id: runtime.result_json.encode("utf-8")
                        for view_id, runtime in workbench.runtime.items()
                    }
                )
            finally:
                workbench.close()
        for signature in signatures[1:]:
            self.assertEqual(signature, signatures[0])

    def test_lease_expiry_hides_view_then_rebuild_restores_replay(self) -> None:
        workbench = self._workbench("lease", reset=True)
        try:
            workbench.advance_all()
            before = workbench.fast_snapshot()
            before_view = next(view for view in before["views"] if view["view_id"] == "v_region")
            self.assertTrue(before_view["visible"])
            simulation = workbench.simulate_lease_expiry("v_region", ttl_ms=80)
            self.assertTrue(simulation["expired"])
            hidden = workbench.fast_snapshot()
            hidden_view = next(view for view in hidden["views"] if view["view_id"] == "v_region")
            self.assertFalse(hidden_view["visible"])
            self.assertTrue(hidden_view["inconsistent"])
            with self.assertRaises(BusinessError) as caught:
                workbench.advance_batch()
            self.assertEqual(caught.exception.code, "E_LEASE")
            trace = workbench.rebuild_view("v_region", "test rebuild")
            self.assertEqual(trace["status"], "rebuilt")
            recovered = workbench.fast_snapshot()
            recovered_view = next(view for view in recovered["views"] if view["view_id"] == "v_region")
            self.assertTrue(recovered_view["visible"])
            self.assertFalse(recovered_view["inconsistent"])
            self.assertEqual(recovered_view["lease"]["state"], "completed")
            self.assertTrue(workbench.authoritative_recompute()["byte_identical"])
            self.assertEqual(workbench.replay_verify()["status"], "matched")
        finally:
            workbench.close()

    def test_snapshot_tamper_is_detected_and_restored_on_reopen(self) -> None:
        workbench = self._workbench("snapshot", reset=True)
        try:
            workbench.advance_all()
            expected = {view_id: runtime.result_digest for view_id, runtime in workbench.runtime.items()}
        finally:
            workbench.close()
        store = LedgerStore(self.root / "snapshot" / "ledger.sqlite3")
        try:
            tamper = store.tamper_result_digest()
        finally:
            store.close()
        connection = sqlite3.connect(self.root / "snapshot" / "ledger.sqlite3")
        try:
            connection.execute("UPDATE meta SET value='7' WHERE key='cursor'")
            connection.commit()
        finally:
            connection.close()
        reopened = self._workbench("snapshot")
        try:
            snapshot_events = [event for event in reopened.startup_events if event.get("code") == "E_SNAPSHOT"]
            self.assertTrue(snapshot_events)
            self.assertEqual({view_id: runtime.result_digest for view_id, runtime in reopened.runtime.items()}, expected)
            self.assertEqual(reopened.cursor, 10)
            self.assertEqual(reopened.replay_verify()["status"], "matched")
            self.assertTrue(tamper["tampered_digest"] != tamper["old_digest"])
        finally:
            reopened.close()

    def test_definition_and_scale_replacement_marks_old_version_stale(self) -> None:
        workbench = self._workbench("replacement", reset=True)
        try:
            workbench.advance_all()
            before_version = workbench.param_version
            report = workbench.apply_views_file(SAMPLES / "views_v2.json")
            self.assertEqual(report["status"], "replaced")
            summary = report["summary"]
            self.assertEqual(summary["old_param_version"], before_version)
            self.assertEqual(summary["new_param_version"], before_version + 1)
            self.assertEqual(summary["added"], ["v_active"])
            self.assertEqual(summary["removed"], ["v_product"])
            self.assertEqual(summary["changed"], ["v_region"])
            self.assertEqual(summary["before"]["scale_factor"], 100)
            self.assertEqual(summary["after"]["scale_factor"], 10)
            self.assertEqual(sorted(workbench.runtime), ["v_active", "v_region", "v_region_product"])
            self.assertTrue(workbench.authoritative_recompute()["byte_identical"])
            self.assertEqual(workbench.replay_verify()["status"], "matched")
            connection = sqlite3.connect(self.root / "replacement" / "ledger.sqlite3")
            connection.row_factory = sqlite3.Row
            try:
                old_results = connection.execute(
                    "SELECT stale FROM view_results WHERE param_version=?", (before_version,)
                ).fetchall()
                self.assertTrue(old_results)
                self.assertTrue(all(row["stale"] == 1 for row in old_results))
                old_leases = connection.execute(
                    "SELECT stale FROM leases WHERE param_version=?", (before_version,)
                ).fetchall()
                self.assertTrue(old_leases)
                self.assertTrue(all(row["stale"] == 1 for row in old_leases))
                current_versions = connection.execute(
                    "SELECT version FROM param_versions WHERE is_current=1"
                ).fetchall()
                self.assertEqual([row["version"] for row in current_versions], [before_version + 1])
            finally:
                connection.close()
        finally:
            workbench.close()

    def test_anomaly_injections_emit_required_business_codes(self) -> None:
        workbench = self._workbench("injections", reset=True)
        try:
            workbench.advance_all()
            lease = workbench.inject_lease_conflict()
            data = workbench.inject_base_corruption()
            snapshot = workbench.inject_snapshot_tamper()
            self.assertEqual(lease["code"], "E_LEASE")
            self.assertEqual(data["code"], "E_DATA")
            self.assertEqual(snapshot["code"], "E_SNAPSHOT")
            self.assertTrue(snapshot["recovered"])
            log_codes = {event.get("code") for event in workbench.event_log(500)}
            self.assertTrue({"E_LEASE", "E_DATA", "E_SNAPSHOT"}.issubset(log_codes))
            self.assertEqual(workbench.replay_verify()["status"], "matched")
        finally:
            workbench.close()

    def test_normal_reopen_checks_snapshot_and_replays(self) -> None:
        workbench = self._workbench("reopen", reset=True)
        try:
            workbench.advance_all()
        finally:
            workbench.close()
        reopened = self._workbench("reopen")
        try:
            audit = reopened._last_snapshot_audit
            self.assertIsNotNone(audit)
            self.assertTrue(audit.matches)
            self.assertFalse(audit.recovered)
            self.assertEqual(reopened.cursor, 10)
            self.assertTrue(reopened.authoritative_recompute()["byte_identical"])
            self.assertEqual(reopened.replay_verify()["status"], "matched")
            batches = reopened.ledger_batches(limit=10)
            self.assertEqual(batches[0]["ledger_format_version"], 1)
            self.assertEqual(len(batches[0]["content_digest"]), 64)
        finally:
            reopened.close()

    def test_view_processing_order_does_not_change_result_bytes(self) -> None:
        signatures = []
        for concurrency in (1, 3):
            workbench = self._workbench(f"order-{concurrency}", reset=True, global_concurrency=concurrency)
            try:
                workbench.advance_all()
                signatures.append(
                    {view_id: runtime.result_json for view_id, runtime in workbench.runtime.items()}
                )
            finally:
                workbench.close()
        self.assertEqual(signatures[0], signatures[1])

    def test_previ_and_replay_errors_identify_first_view_group_and_field(self) -> None:
        workbench = self._workbench("differences", reset=True)
        try:
            workbench.advance_all()
            runtime = workbench.runtime["v_region"]
            payload = json.loads(runtime.result_json)
            payload["groups"][0]["values"]["amount_sum"] += 1
            runtime.result_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            with self.assertRaises(BusinessError) as caught:
                workbench.authoritative_recompute()
            self.assertEqual(caught.exception.code, "E_PREVI")
            self.assertEqual(caught.exception.location["view_id"], "v_region")
            self.assertEqual(caught.exception.location["group_key"], payload["groups"][0]["key"])
            self.assertEqual(caught.exception.location["field"], "amount_sum")
            with self.assertRaises(BusinessError) as replay_error:
                workbench.replay_verify()
            self.assertEqual(replay_error.exception.code, "E_REPLAY")
            self.assertEqual(replay_error.exception.location["view_id"], "v_region")
            self.assertEqual(replay_error.exception.location["field"], "amount_sum")
        finally:
            workbench.close()


if __name__ == "__main__":
    unittest.main()
