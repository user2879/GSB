from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from config_arbiter.errors import ReplayError, SnapshotError
from config_arbiter.ledger import Ledger
from config_arbiter.snapshot import load_snapshot, replay_ledger, restore_workspace, write_snapshot
from config_arbiter.util import canonical_sha256, canonical_text
from tests.support import build_engine, read_json, write_json


class SnapshotTests(unittest.TestCase):
    def _prepare(self, root: Path):
        engine, ledger, specs, environment_path, views_path = build_engine(root)
        engine.run_all()
        snapshot_path = root / "snapshot.json"
        write_snapshot(snapshot_path, specs, ledger)
        return engine, ledger, specs, environment_path, views_path, snapshot_path

    def test_snapshot_contains_versions_summaries_and_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine, ledger, _, _, _, snapshot_path = self._prepare(root)
            try:
                payload = load_snapshot(snapshot_path)
                self.assertEqual(payload["format_version"], 1)
                self.assertEqual(payload["environment_set_version"], 3)
                self.assertEqual(payload["source_priority_version"], 7)
                self.assertEqual(payload["scale_policy_version"], "scale-policy/1000.1")
                self.assertEqual(len(payload["completed_view_ids"]), 3)
                self.assertIn("ops-authority", payload["view_summaries"])
                self.assertRegex(payload["content_digest"], r"^[0-9a-f]{64}$")
            finally:
                ledger.close()

    def test_matching_restore_replays_three_views(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine, ledger, _, environment_path, views_path, snapshot_path = self._prepare(root)
            ledger.close()
            report = restore_workspace(
                snapshot_path=snapshot_path,
                environment_path=environment_path,
                views_path=views_path,
                db_path=root / "ledger.sqlite",
            )
            self.assertEqual(report["mode"], "replayed")
            self.assertEqual(report["replay"]["compared_count"], 3)

    def test_version_or_state_mismatch_triggers_whole_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine, ledger, _, environment_path, views_path, snapshot_path = self._prepare(root)
            ledger.close()
            environment_document = read_json(environment_path)
            environment_document["environments"][0]["sources"][0]["observations"][0]["value"] = 9999
            write_json(environment_path, environment_document)

            report = restore_workspace(
                snapshot_path=snapshot_path,
                environment_path=environment_path,
                views_path=views_path,
                db_path=root / "ledger.sqlite",
            )
            self.assertEqual(report["mode"], "replaced")
            self.assertEqual(report["reason"], "snapshot_state_or_version_mismatch")
            restored_document = read_json(environment_path)
            self.assertEqual(
                restored_document["environments"][0]["sources"][0]["observations"][0]["value"],
                3001.25,
            )
            ledger = Ledger(root / "ledger.sqlite")
            try:
                self.assertEqual(len(ledger.all_current_views()), 3)
                self.assertEqual(ledger.verify_integrity(), ledger.meta()["content_digest"])
            finally:
                ledger.close()

    def test_replay_mismatch_reports_first_view_key_and_field(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine, ledger, specs, environment_path, views_path, snapshot_path = self._prepare(root)
            current = ledger.current_view("ops-authority")
            result = current["result"]
            result["keys"]["http.timeout_ms"]["authoritative"]["value_fixed"] += 1
            tampered_json = canonical_text(result)
            tampered_sha = canonical_sha256(result)
            try:
                ledger.connection.execute("BEGIN IMMEDIATE")
                ledger.connection.execute(
                    "UPDATE view_runs SET result_json = ?, result_sha256 = ? WHERE view_id = ? AND version = ?",
                    (tampered_json, tampered_sha, "ops-authority", current["current_version"]),
                )
                ledger.connection.execute(
                    "UPDATE views SET result_sha256 = ? WHERE view_id = ?",
                    (tampered_sha, "ops-authority"),
                )
                digest = ledger._compute_content_digest(ledger.connection)
                ledger.connection.execute(
                    "UPDATE ledger_meta SET content_digest = ? WHERE singleton = 1",
                    (digest,),
                )
                ledger.connection.commit()
            finally:
                ledger.close()

            with self.assertRaises(ReplayError) as raised:
                restore_workspace(
                    snapshot_path=snapshot_path,
                    environment_path=environment_path,
                    views_path=views_path,
                    db_path=root / "ledger.sqlite",
                )
            error = raised.exception
            self.assertEqual(error.code, "E_REPLAY")
            self.assertEqual(error.details["view_id"], "ops-authority")
            self.assertEqual(error.details["key"], "http.timeout_ms")
            self.assertEqual(error.details["field"], "authoritative.value_fixed")

    def test_snapshot_content_digest_tamper_is_e_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine, ledger, _, _, _, snapshot_path = self._prepare(root)
            try:
                payload = read_json(snapshot_path)
                payload["view_summaries"]["ops-authority"]["view_version"] = 999
                snapshot_path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(SnapshotError) as raised:
                    load_snapshot(snapshot_path)
                self.assertEqual(raised.exception.code, "E_SNAPSHOT")
                self.assertIn("content digest mismatch", raised.exception.message)
            finally:
                ledger.close()

    def test_snapshot_conclusion_summary_mismatch_replaces_whole_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine, ledger, _, environment_path, views_path, snapshot_path = self._prepare(root)
            ledger.close()
            payload = read_json(snapshot_path)
            payload["view_summaries"]["ops-authority"]["view_aggregation_error_upper_bound_fixed_units"] = 999999
            without_digest = {key: value for key, value in payload.items() if key != "content_digest"}
            payload["content_digest"] = canonical_sha256(without_digest)
            write_json(snapshot_path, payload)

            report = restore_workspace(
                snapshot_path=snapshot_path,
                environment_path=environment_path,
                views_path=views_path,
                db_path=root / "ledger.sqlite",
            )
            self.assertEqual(report["mode"], "replaced")
            self.assertEqual(report["reason"], "snapshot_conclusion_summary_mismatch")
            restored = load_snapshot(snapshot_path)
            self.assertNotEqual(
                restored["view_summaries"]["ops-authority"]["view_aggregation_error_upper_bound_fixed_units"],
                999999,
            )


if __name__ == "__main__":
    unittest.main()
