from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

from h2mux.declaration import load_declaration, load_script, parse_declaration
from h2mux.errors import SnapshotError
from h2mux.replay import replay_from_snapshot
from h2mux.session import H2Session
from h2mux.snapshot import SnapshotStore, load_snapshot_document, restore_snapshot
from h2mux.util import digest_value


ROOT = Path(__file__).resolve().parents[1]


class SnapshotReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.declaration = load_declaration(ROOT / "samples" / "session.json")
        self.script = load_script(ROOT / "samples" / "frames.json", self.declaration)
        self.store = SnapshotStore(self.directory)

    def make_snapshot(self, stop: int = 12, name: str = "stage.json") -> Path:
        session = H2Session(self.declaration, self.script)
        session.run(stop_after_events=stop)
        return self.store.save(session, name)

    def test_snapshot_round_trip_and_match_replay(self) -> None:
        path = self.make_snapshot()
        document = load_snapshot_document(path)
        self.assertEqual(document.raw["snapshot_format_version"], 1)
        self.assertEqual(document.declaration_version, 7)
        report = replay_from_snapshot(
            str(path),
            self.script,
            current_declaration=self.declaration,
            current_strategy="weighted_round_robin",
        )
        self.assertEqual(report.status, "MATCH")
        self.assertIsNone(report.first_difference)
        self.assertGreater(report.compared_record_count, 0)

    def test_same_path_save_preserves_read_only_backup(self) -> None:
        session = H2Session(self.declaration, self.script)
        session.run(stop_after_events=8)
        first = self.store.save(session, "same.json")
        first_digest = load_snapshot_document(first).content_digest
        session.run(stop_after_events=14)
        second = self.store.save(session, "same.json")
        backup = Path(str(second) + ".bak")
        self.assertTrue(backup.exists())
        mode = stat.S_IMODE(os.stat(backup).st_mode)
        self.assertFalse(mode & stat.S_IWRITE)
        self.assertEqual(load_snapshot_document(backup).content_digest, first_digest)

    def test_tampered_content_digest_is_rejected(self) -> None:
        path = self.make_snapshot()
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["state"]["ledger"]["records"][0]["conn_send_window_after"] += 1
        path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaises(SnapshotError) as context:
            load_snapshot_document(path)
        self.assertEqual(context.exception.code, "E_SNAPSHOT")

    def test_declaration_version_change_triggers_whole_replacement_and_replay_basis(self) -> None:
        path = self.make_snapshot()
        current = parse_declaration({**self.declaration.to_dict(), "declaration_version": 8})
        restoration = restore_snapshot(
            path,
            self.script,
            current_declaration=current,
            current_strategy="weighted_round_robin",
        )
        self.assertTrue(restoration.replacement_required)
        self.assertIn("DECLARATION_VERSION_CHANGED", restoration.replacement_reasons)
        self.assertTrue(
            any(item["path"] == "$.declaration_version" for item in restoration.changed_fields)
        )
        self.assertEqual(restoration.session.declaration.declaration_version, 7)
        replay = replay_from_snapshot(
            str(path),
            self.script,
            current_declaration=current,
            current_strategy="weighted_round_robin",
        )
        self.assertEqual(replay.status, "NOT_REPRODUCIBLE")
        self.assertEqual(replay.reason, "DECLARATION_VERSION_MISMATCH")
        self.assertTrue(replay.basis["changed_declaration_fields"])

    def test_strategy_change_is_reported_with_specific_basis(self) -> None:
        path = self.make_snapshot()
        replay = replay_from_snapshot(
            str(path),
            self.script,
            current_declaration=self.declaration,
            current_strategy="window_priority",
        )
        self.assertEqual(replay.status, "NOT_REPRODUCIBLE")
        self.assertEqual(replay.reason, "STRATEGY_CHANGED")
        self.assertEqual(replay.basis["current_strategy"], "window_priority")
        self.assertEqual(replay.basis["snapshot_strategy"], "weighted_round_robin")

    def test_snapshot_after_strategy_switch_replays_with_history(self) -> None:
        session = H2Session(self.declaration, self.script)
        session.run(stop_after_events=10)
        session.set_strategy("window_priority", script_index=10)
        session.run(stop_after_events=17)
        path = self.store.save(session, "switched.json")
        report = replay_from_snapshot(
            str(path),
            self.script,
            current_declaration=self.declaration,
            current_strategy="window_priority",
        )
        self.assertEqual(report.status, "MATCH")

    def test_prefix_tampering_reports_first_frame_and_field(self) -> None:
        path = self.make_snapshot()
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["state"]["ledger"]["records"][0]["conn_send_window_after"] += 1
        content = {key: value for key, value in raw.items() if key != "content_digest"}
        raw["content_digest"] = digest_value(content)
        path.write_text(json.dumps(raw), encoding="utf-8")
        report = replay_from_snapshot(
            str(path),
            self.script,
            current_declaration=self.declaration,
            current_strategy="weighted_round_robin",
        )
        self.assertEqual(report.status, "MISMATCH")
        self.assertEqual(report.reason, "SNAPSHOT_PREFIX_MISMATCH")
        self.assertIsNotNone(report.first_difference)
        self.assertIn("conn_send_window", report.first_difference["field"])
        self.assertIn("frame_index", report.first_difference)
        self.assertIn("stream_id", report.first_difference)

    def test_tail_truncation_is_attributed_separately(self) -> None:
        truncated_script = load_script(ROOT / "samples" / "frames_truncated.json", self.declaration)
        session = H2Session(self.declaration, truncated_script)
        session.run(stop_after_events=2)
        path = self.store.save(session, "truncated.json")
        report = replay_from_snapshot(
            str(path),
            truncated_script,
            current_declaration=self.declaration,
            current_strategy="weighted_round_robin",
        )
        self.assertEqual(report.status, "TRUNCATED")
        self.assertEqual(report.reason, "SCRIPT_TAIL_TRUNCATED")
        self.assertTrue(report.tail_truncated)


if __name__ == "__main__":
    unittest.main()
