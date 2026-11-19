from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from bucketpack.engine import run_full
from bucketpack.errors import E_LEDGER, E_REPLAY, BusinessError
from bucketpack.ledger import Ledger, schedule_map_from_state

from .common import tiny_spec


class LedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = tiny_spec()
        self.state = run_full(self.spec)
        self.schedule = schedule_map_from_state(self.state)

    def test_persist_and_recover_complete_batch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ledger.sqlite"
            with Ledger(path) as ledger:
                persisted = ledger.persist_state(self.state, "batch-1")
                self.assertEqual(persisted["status"], "complete")
                self.assertTrue(ledger.batch_content_digest("batch-1"))
                report = ledger.recover_batch(
                    "batch-1",
                    self.state.packs,
                    spec_digest=self.spec.digest,
                    snapshot_digest=self.state.result["content_digest"],
                    schedule=self.schedule,
                )
                self.assertEqual(report["status"], "complete")
                self.assertEqual(report["unknown_packs"], [])
                batch = ledger.read_batches()[0]
                self.assertEqual(batch["committed_count"], len(self.state.packs))

    def test_partial_batch_marks_unknown_then_replays(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ledger.sqlite"
            with Ledger(path) as ledger:
                ledger.begin_batch(
                    "partial",
                    self.state.packs,
                    spec_digest=self.spec.digest,
                    snapshot_digest=self.state.result["content_digest"],
                )
                first = self.state.packs[0]
                ledger.commit_pack("partial", first, self.schedule.get(first.pack_id))
            with Ledger(path) as ledger:
                report = ledger.recover_batch(
                    "partial",
                    self.state.packs,
                    spec_digest=self.spec.digest,
                    snapshot_digest=self.state.result["content_digest"],
                    schedule=self.schedule,
                )
                self.assertEqual(report["status"], "partial")
                self.assertIn(self.state.packs[1].pack_id, report["unknown_packs"])
                by_id = {pack.pack_id: pack for pack in self.state.packs}
                for pack_id in report["unknown_packs"]:
                    ledger.commit_pack("partial", by_id[pack_id], self.schedule.get(pack_id))
                ledger.finish_batch("partial", snapshot_digest=self.state.result["content_digest"])
                final = ledger.recover_batch(
                    "partial",
                    self.state.packs,
                    spec_digest=self.spec.digest,
                    snapshot_digest=self.state.result["content_digest"],
                    schedule=self.schedule,
                )
                self.assertEqual(final["status"], "complete")

    def test_version_mismatch_replaces_ledger_and_raises_replay(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ledger.sqlite"
            with Ledger(path) as ledger:
                ledger.persist_state(self.state, "versioned")
            with sqlite3.connect(path) as conn:
                conn.execute("UPDATE ledger_meta SET value='0' WHERE key='format_version'")
                conn.commit()
            with Ledger(path) as ledger:
                with self.assertRaises(BusinessError) as caught:
                    ledger.recover_batch(
                        "versioned",
                        self.state.packs,
                        spec_digest=self.spec.digest,
                        snapshot_digest=self.state.result["content_digest"],
                        schedule=self.schedule,
                    )
                self.assertEqual(caught.exception.code, E_REPLAY)
                self.assertEqual(ledger.format_version(), 1)
                self.assertEqual(ledger.read_batches()[0]["status"], "COMPLETE")

    def test_content_mismatch_replaces_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ledger.sqlite"
            with Ledger(path) as ledger:
                ledger.persist_state(self.state, "content")
            first_pack = min(self.state.packs, key=lambda item: item.pack_id)
            with sqlite3.connect(path) as conn:
                conn.execute(
                    "UPDATE packs SET summary_json='{}' WHERE batch_id=? AND pack_id=?",
                    ("content", first_pack.pack_id),
                )
                conn.commit()
            with Ledger(path) as ledger:
                with self.assertRaises(BusinessError) as caught:
                    ledger.recover_batch(
                        "content",
                        self.state.packs,
                        spec_digest=self.spec.digest,
                        snapshot_digest=self.state.result["content_digest"],
                        schedule=self.schedule,
                    )
                self.assertEqual(caught.exception.code, E_REPLAY)
                self.assertEqual(
                    caught.exception.details["first_differing_pack_id"],
                    first_pack.pack_id,
                )
                self.assertEqual(ledger.read_batches()[0]["status"], "COMPLETE")

    def test_injected_ledger_write_failure(self) -> None:
        with (
            tempfile.TemporaryDirectory() as td,
            Ledger(Path(td) / "ledger.sqlite", fail_on_write=True) as ledger,
            self.assertRaises(BusinessError) as caught,
        ):
            ledger.persist_state(self.state, "fault")
        self.assertEqual(caught.exception.code, E_LEDGER)


if __name__ == "__main__":
    unittest.main()
