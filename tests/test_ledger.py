from __future__ import annotations

import copy
import sqlite3
import tempfile
import unittest
from pathlib import Path

from speaker_diag.canonical import canonical_bytes, canonical_sha256, sha256_hex
from speaker_diag.errors import DiagnosticError
from speaker_diag.ledger import Ledger
from speaker_diag.pipeline import commit_receipts, normalize_local_results
from speaker_diag.receipts import (
    NormalizedReceipt,
    context_for,
    normalize_receipt,
    strip_to_r1,
)
from speaker_diag.spec import load_workload_data
from speaker_diag.synthetic import synthesize

from tests.helpers import tiny_spec, tiny_workload


class LedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)
        self.spec = tiny_spec()
        self.bundle = synthesize(self.spec)
        self.raw, self.normalized = normalize_local_results(self.spec, self.bundle)
        self.truth = {window.window_id: window for window in self.bundle.windows}
        self.path = self.dir / "ledger.sqlite3"
        self.ledger = Ledger(
            self.path,
            session_version=self.spec.session_version,
            spec_digest=self.spec.spec_digest,
            session_snapshot=self.spec.to_dict(),
            reset=True,
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _commit(self, batch_ids: list[str]) -> list[dict[str, object]]:
        ordinal = {request.batch_id: request.request_index for request in self.spec.requests}
        return [self.ledger.commit_batch(self.normalized[batch_id], self.truth, ordinal=ordinal[batch_id]) for batch_id in batch_ids]

    def test_duplicate_batch_is_rejected(self) -> None:
        self._commit(["B001"])
        with self.assertRaises(DiagnosticError) as caught:
            self.ledger.commit_batch(self.normalized["B001"], self.truth, ordinal=0)
        self.assertEqual(caught.exception.code, "E_REPLAY")
        self.assertEqual(caught.exception.field, "batch_id")

    def test_integer_ledger_format_version_is_stored(self) -> None:
        connection = sqlite3.connect(self.path)
        try:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
        finally:
            connection.close()
        self.assertEqual(version, self.ledger.content_manifest()["ledger_schema_version"])

    def test_recovery_marks_unknown_then_rerun_verifies(self) -> None:
        self._commit(["B001"])
        expected = [request.batch_id for request in self.spec.requests]
        recovery = self.ledger.recover(expected)
        self.assertEqual(recovery["rerun_order"], ["B002", "B003"])
        self.assertEqual([item["batch_id"] for item in recovery["unknown_batches"]], ["B002", "B003"])
        rerun = {batch_id: self.normalized[batch_id] for batch_id in recovery["rerun_order"]}
        commit_receipts(self.ledger, self.spec, self.bundle, rerun)
        verified = self.ledger.verify_replay(self.normalized, expected_order=expected)
        self.assertEqual(verified["verified_batches"], 3)
        self.assertEqual(self.ledger.content_manifest()["unknown_batches"], [])

    def test_replay_mismatch_reports_first_batch_window_field(self) -> None:
        expected = [request.batch_id for request in self.spec.requests]
        self._commit(expected)
        bad_data = copy.deepcopy(self.normalized["B002"].data)
        bad_data["windows"][0]["attribution"] = "spk_9"
        encoded = canonical_bytes(bad_data)
        bad = NormalizedReceipt(
            data=bad_data,
            canonical=encoded,
            sha256=sha256_hex(encoded),
            attribution_digest=canonical_sha256(bad_data),
            source_format="r3",
            filled_fields=(),
        )
        regenerated = dict(self.normalized)
        regenerated["B002"] = bad
        with self.assertRaises(DiagnosticError) as caught:
            self.ledger.verify_replay(regenerated, expected_order=expected)
        exc = caught.exception
        self.assertEqual(exc.code, "E_REPLAY")
        self.assertEqual(exc.details["batch_id"], "B002")
        self.assertEqual(exc.details["window_id"], bad_data["windows"][0]["window_id"])
        self.assertEqual(exc.field, "windows[0].attribution")

    def test_snapshot_restore_and_session_replacement(self) -> None:
        expected = [request.batch_id for request in self.spec.requests]
        self._commit(expected)
        snapshot_path = self.dir / "snapshot.json"
        self.ledger.export_snapshot(snapshot_path)
        restored_path = self.dir / "restored.sqlite3"
        restored = Ledger.restore_snapshot(restored_path, snapshot_path)
        self.assertEqual(restored.verify_replay(self.normalized, expected_order=expected)["verified_batches"], 3)

        # Force a different session/spec identity while keeping this isolated test deterministic.
        raw = tiny_workload(port1=39003, port2=39004)
        raw["session_version"] = 2
        new_spec = load_workload_data(raw)
        replaced = Ledger(
            self.path,
            session_version=new_spec.session_version,
            spec_digest=new_spec.spec_digest,
            session_snapshot=new_spec.to_dict(),
            allow_replace=True,
        )
        manifest = replaced.content_manifest()
        self.assertEqual(manifest["session_version"], 2)
        self.assertEqual(manifest["batch_count"], 0)

    def test_replay_rejects_wrong_batch_order(self) -> None:
        expected = [request.batch_id for request in self.spec.requests]
        self._commit(expected)
        reordered = [expected[1], expected[0], expected[2]]
        with self.assertRaises(DiagnosticError) as caught:
            self.ledger.verify_replay(self.normalized, expected_order=reordered)
        self.assertEqual(caught.exception.code, "E_REPLAY")
        self.assertEqual(caught.exception.field, "ordinal")
        self.assertEqual(caught.exception.details["reason"], "batch_order_mismatch")

    def test_r1_ledger_row_replays_against_r3_semantic_output(self) -> None:
        expected = [request.batch_id for request in self.spec.requests]
        r1_receipt = strip_to_r1(self.raw["B001"].receipt)
        r1_normalized = normalize_receipt(r1_receipt, context=context_for(self.spec, "B001"))
        self.ledger.commit_batch(r1_normalized, self.truth, ordinal=0)
        self._commit(["B002", "B003"])
        verified = self.ledger.verify_replay(self.normalized, expected_order=expected)
        self.assertEqual(verified["verified_batches"], 3)


if __name__ == "__main__":
    unittest.main()
