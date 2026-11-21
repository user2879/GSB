from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from streamagg.engine import Engine
from streamagg.errors import BusinessError
from streamagg.ledger import Ledger
from streamagg.replay import validate_replay
from streamagg.snapshot import SnapshotStore, restore_engine
from streamagg.spec import load_spec


ROOT = Path(__file__).resolve().parents[1]


class SnapshotAndReplayTests(unittest.TestCase):
    def test_snapshot_roundtrip_marks_pending_unknown_and_replays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = load_spec(ROOT / "samples" / "scenario1_ordered.json")
            store = SnapshotStore(root / "snapshots")
            ledger = Ledger(root / "ledger.sqlite3")
            engine = Engine(spec, session_id="snapshot-roundtrip", ledger=ledger)
            engine.run_all()
            view = engine.view_id
            manifest = dict(ledger.get_view_manifest(engine.session_id, view))
            pending_before = ledger.connection.execute(
                "SELECT COUNT(*) AS c FROM window_state WHERE session_id = ? AND view_id = ? AND status = 'PENDING'",
                (engine.session_id, view),
            ).fetchone()["c"]
            self.assertGreater(pending_before, 0)
            saved = store.save(engine, snapshot_id="roundtrip")
            ledger.close()

            reopened = Ledger(root / "ledger.sqlite3")
            restored, loaded, unknown = restore_engine(
                saved.path,
                store=store,
                ledger=reopened,
            )
            self.assertEqual(len(unknown), pending_before)
            self.assertIsNotNone(loaded.comparison)
            self.assertFalse(loaded.comparison.replacement_required)
            self.assertEqual(restored.spec.rules_version, 1)
            self.assertEqual(len(restored.finalized_windows), 15)
            self.assertEqual(len(restored.decisions), 36)
            self.assertEqual(len(restored.traces), 36)
            unknown_rows = reopened.connection.execute(
                "SELECT COUNT(*) AS c FROM window_state WHERE session_id = ? AND view_id = ? AND status = 'UNKNOWN'",
                (restored.session_id, restored.view_id),
            ).fetchone()["c"]
            self.assertEqual(unknown_rows, pending_before)
            report = validate_replay(restored, reopened)
            self.assertTrue(report["consistent"])
            reopened.close()

    def test_tampered_snapshot_digest_reports_e_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = load_spec(ROOT / "samples" / "scenario1_ordered.json")
            store = SnapshotStore(root / "snapshots")
            with Ledger(root / "ledger.sqlite3") as ledger:
                engine = Engine(spec, session_id="tamper", ledger=ledger)
                engine.run_all()
                saved = store.save(engine, snapshot_id="tampered")
            envelope = json.loads(saved.path.read_text(encoding="utf-8"))
            envelope["state"]["watermarks"]["p0"] = 1
            saved.path.write_text(
                json.dumps(envelope, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            with self.assertRaises(BusinessError) as caught:
                store.load(saved.path)
            self.assertEqual(caught.exception.code, "E_SNAPSHOT")
            self.assertEqual(caught.exception.field, "manifest.content_digest")

    def test_tampered_ledger_batch_digest_reports_e_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = load_spec(ROOT / "samples" / "scenario1_ordered.json")
            with Ledger(root / "ledger.sqlite3") as ledger:
                engine = Engine(spec, session_id="ledger-digest", ledger=ledger)
                engine.run_all()
                ledger.connection.execute(
                    "UPDATE ledger_batches SET content_digest = ? WHERE session_id = ? AND view_id = ? AND batch_no = 1",
                    ("0" * 64, engine.session_id, engine.view_id),
                )
                ledger.connection.commit()
                with self.assertRaises(BusinessError) as caught:
                    ledger.verify_content_digest(engine.session_id, engine.view_id)
            self.assertEqual(caught.exception.code, "E_SNAPSHOT")
            self.assertEqual(caught.exception.field, "ledger.content_digest")

    def test_replay_reports_first_mismatching_window_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = load_spec(ROOT / "samples" / "scenario1_ordered.json")
            with Ledger(root / "ledger.sqlite3") as ledger:
                engine = Engine(spec, session_id="replay-mismatch", ledger=ledger)
                engine.run_all()
                row = ledger.connection.execute(
                    "SELECT * FROM window_state WHERE session_id = ? AND view_id = ? AND status = 'COMMITTED' ORDER BY partition_key, window_start_ms LIMIT 1",
                    (engine.session_id, engine.view_id),
                ).fetchone()
                ledger.connection.execute(
                    "UPDATE window_state SET mean_fixed = mean_fixed + 7 WHERE session_id = ? AND view_id = ? AND rule_id = ? AND partition_key = ? AND window_start_ms = ? AND window_end_ms = ?",
                    (
                        engine.session_id,
                        engine.view_id,
                        row["rule_id"],
                        row["partition_key"],
                        row["window_start_ms"],
                        row["window_end_ms"],
                    ),
                )
                ledger.connection.commit()
                with self.assertRaises(BusinessError) as caught:
                    validate_replay(engine, ledger)
            error = caught.exception
            self.assertEqual(error.code, "E_REPLAY")
            self.assertEqual(error.field, "window.mean_fixed")
            self.assertEqual(error.details["partition_key"], "p0")
            self.assertEqual(error.details["window_start_ms"], 1700000000000)
            self.assertEqual(error.details["expected"], row["mean_fixed"] + 7)
            self.assertEqual(error.details["actual"], row["mean_fixed"])

    def test_replay_detects_fixed_scale_manifest_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = load_spec(ROOT / "samples" / "scenario1_ordered.json")
            with Ledger(root / "ledger.sqlite3") as ledger:
                engine = Engine(spec, session_id="scale-mismatch", ledger=ledger)
                engine.run_all()
                changed_spec = type(spec)(
                    format_version=spec.format_version,
                    sources=spec.sources,
                    rules=spec.rules,
                    fixed_scale=spec.fixed_scale + 1,
                    lateness_budget_count=spec.lateness_budget_count,
                    lateness_budget_amount=spec.lateness_budget_amount,
                    dominant_threshold_bp=spec.dominant_threshold_bp,
                    order_backward_limit_ms=spec.order_backward_limit_ms,
                )
                mismatched_engine = Engine(
                    changed_spec,
                    session_id="scale-mismatch",
                    ledger=None,
                    view_id=engine.view_id,
                )
                with self.assertRaises(BusinessError) as caught:
                    validate_replay(mismatched_engine, ledger)
            self.assertEqual(caught.exception.code, "E_REPLAY")
            self.assertEqual(caught.exception.field, "fixed_scale")

    def test_manifest_mismatch_requires_wholesale_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = load_spec(ROOT / "samples" / "scenario1_ordered.json")
            store = SnapshotStore(root / "snapshots")
            with Ledger(root / "ledger.sqlite3") as ledger:
                engine = Engine(spec, session_id="manifest", ledger=ledger)
                engine.run_all()
                saved = store.save(engine, snapshot_id="manifest")
            loaded = store.load(
                saved.path,
                current_manifest={
                    "rules_version": 9,
                    "content_digest": "different",
                    "spec_digest": "different",
                },
            )
            self.assertIsNotNone(loaded.comparison)
            self.assertTrue(loaded.comparison.replacement_required)
            self.assertFalse(loaded.comparison.rules_version_matches)
            self.assertFalse(loaded.comparison.content_digest_matches)


if __name__ == "__main__":
    unittest.main()
