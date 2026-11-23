from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from cacheproof.canonical import digest_value
from cacheproof.errors import DiagnosticError
from cacheproof.spec import load_spec
from cacheproof.store import CacheProofStore


ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "samples" / "cache_spec.json"


def reseal_proof(proof: dict) -> None:
    payload = {key: value for key, value in proof.items() if key != "content_digest"}
    proof["content_digest"] = digest_value(payload)


def reseal_snapshot(snapshot: dict) -> None:
    payload = {
        key: value for key, value in snapshot.items() if key != "content_digest"
    }
    snapshot["content_digest"] = digest_value(payload)


class SnapshotReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="cacheproof-snapshot-")
        self.workspace = Path(self.temp.name) / "workspace"
        self.store = CacheProofStore(
            self.workspace,
            spec=load_spec(SAMPLE),
            create=True,
            auto_snapshot=False,
        )

    def tearDown(self) -> None:
        self.store.close()
        self.temp.cleanup()

    def _write_snapshot(self, snapshot: dict) -> None:
        self.store.paths.session_snapshot.write_text(
            json.dumps(snapshot, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    def test_stale_snapshot_replays_and_is_replaced_wholesale(self) -> None:
        saved = self.store.save_snapshot()
        self.store.put_entry("seg-alpha", "alpha/after", 16, "after")
        restored = self.store.restore_snapshot()
        self.assertTrue(restored["snapshot_replaced"])
        self.assertEqual(restored["replayed_operations"], 1)
        self.assertEqual(self.store.verify_all()["code"], "OK")
        again = self.store.restore_snapshot()
        self.assertFalse(again["snapshot_replaced"])
        self.assertEqual(again["replayed_operations"], 0)

    def test_tampered_snapshot_entry_reports_first_e_replay_field(self) -> None:
        self.store.save_snapshot()
        snapshot = json.loads(
            self.store.paths.session_snapshot.read_text(encoding="utf-8")
        )
        alpha = next(
            proof for proof in snapshot["segments"] if proof["segment_id"] == "seg-alpha"
        )
        entry = next(item for item in alpha["entries"] if item["key"] == "alpha/a")
        entry["digest"] = "sha256:" + "2" * 64
        reseal_proof(alpha)
        reseal_snapshot(snapshot)
        self._write_snapshot(snapshot)
        with self.assertRaises(DiagnosticError) as caught:
            self.store.restore_snapshot()
        error = caught.exception
        self.assertEqual(error.code, "E_REPLAY")
        self.assertEqual(error.segment_id, "seg-alpha")
        self.assertEqual(error.entry_key, "alpha/a")
        self.assertEqual(error.field, "entries.digest")

    def test_tampered_snapshot_digest_is_e_snapshot(self) -> None:
        self.store.save_snapshot()
        snapshot = json.loads(
            self.store.paths.session_snapshot.read_text(encoding="utf-8")
        )
        snapshot["content_digest"] = "sha256:" + "0" * 64
        self._write_snapshot(snapshot)
        with self.assertRaises(DiagnosticError) as caught:
            self.store.restore_snapshot()
        self.assertEqual(caught.exception.code, "E_SNAPSHOT")
        self.assertEqual(caught.exception.field, "snapshot.content_digest")

    def test_tampered_operation_business_code_reports_e_replay(self) -> None:
        self.store.save_snapshot()
        result = self.store.put_entry("seg-alpha", "alpha/code", 8, "code")
        self.store.connection.execute(
            "UPDATE operations SET result_code='E_CAPACITY' WHERE operation_seq=?",
            (result.operation_seq,),
        )
        with self.assertRaises(DiagnosticError) as caught:
            self.store.restore_snapshot()
        self.assertEqual(caught.exception.code, "E_REPLAY")
        self.assertEqual(caught.exception.segment_id, "seg-alpha")
        self.assertEqual(caught.exception.entry_key, "alpha/code")
        self.assertEqual(caught.exception.field, "result_code")


if __name__ == "__main__":
    unittest.main()

