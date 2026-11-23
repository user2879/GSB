from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from cacheproof.canonical import canonical_text
from cacheproof.diagnostics import dashboard_snapshot, render_text_dashboard
from cacheproof.errors import DiagnosticError
from cacheproof.spec import load_spec
from cacheproof.store import CacheProofStore


ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "samples" / "cache_spec.json"


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="cacheproof-test-")
        self.workspace = Path(self.temp.name) / "workspace"
        self.spec = load_spec(SAMPLE)
        self.store = CacheProofStore(
            self.workspace, spec=self.spec, create=True, auto_snapshot=False
        )

    def tearDown(self) -> None:
        self.store.close()
        self.temp.cleanup()

    def test_proofs_verify_and_are_independent_of_workspace_order(self) -> None:
        first = self.store.verify_all()
        self.assertEqual(first["code"], "OK")
        with tempfile.TemporaryDirectory(prefix="cacheproof-second-") as other:
            with CacheProofStore(
                Path(other) / "workspace",
                spec=load_spec(SAMPLE),
                create=True,
                auto_snapshot=False,
            ) as second_store:
                for segment_id in ("seg-beta", "seg-alpha"):
                    self.assertEqual(
                        canonical_text(self.store.build_proof(segment_id)),
                        canonical_text(second_store.build_proof(segment_id)),
                    )

    def test_content_files_are_byte_identical_across_workspaces(self) -> None:
        def file_bytes(workspace: Path) -> dict[str, bytes]:
            entries = workspace / "entries"
            return {
                path.relative_to(entries).as_posix(): path.read_bytes()
                for path in entries.rglob("*.bin")
            }

        first = file_bytes(self.workspace)
        with tempfile.TemporaryDirectory(prefix="cacheproof-files-") as other:
            second_workspace = Path(other) / "workspace"
            with CacheProofStore(
                second_workspace,
                spec=load_spec(SAMPLE),
                create=True,
                auto_snapshot=False,
            ):
                second = file_bytes(second_workspace)
        self.assertEqual(first, second)

    def test_reinitializing_same_declaration_does_not_duplicate_files(self) -> None:
        before = sorted(
            path.relative_to(self.workspace).as_posix()
            for path in (self.workspace / "entries").rglob("*.bin")
        )
        with CacheProofStore(
            self.workspace,
            spec=load_spec(SAMPLE),
            create=True,
            auto_snapshot=False,
        ) as reopened:
            self.assertEqual(reopened.verify_all()["code"], "OK")
        after = sorted(
            path.relative_to(self.workspace).as_posix()
            for path in (self.workspace / "entries").rglob("*.bin")
        )
        self.assertEqual(before, after)

    def test_capacity_rejection_does_not_change_proof(self) -> None:
        before = canonical_text(self.store.build_proof("seg-alpha"))
        with self.assertRaises(DiagnosticError) as caught:
            self.store.put_entry("seg-alpha", "alpha/too-big", 500, "too-big")
        error = caught.exception
        self.assertEqual(error.code, "E_CAPACITY")
        self.assertEqual(error.details["required_bytes"], 500)
        self.assertEqual(error.details["available_bytes"], 144)
        self.assertEqual(before, canonical_text(self.store.build_proof("seg-alpha")))
        rejected = self.store.connection.execute(
            "SELECT * FROM operations WHERE status='rejected'"
        ).fetchone()
        self.assertIsNone(rejected["committed_cursor"])

    def test_put_delete_and_idempotent_put(self) -> None:
        put = self.store.put_entry("seg-alpha", "alpha/c", 40, "alpha-c")
        self.assertEqual(put.committed_cursor, 1)
        proof = self.store.build_proof("seg-alpha")
        self.assertEqual(proof["used_bytes"], 152)
        self.assertEqual(proof["entry_count"], 3)

        repeated = self.store.put_entry("seg-alpha", "alpha/c", 40, "alpha-c")
        self.assertTrue(repeated.idempotent)
        self.assertEqual(self.store.build_proof("seg-alpha")["used_bytes"], 152)
        files = list((self.workspace / "entries").rglob("*.bin"))
        self.assertEqual(len(files), 5)

        deleted = self.store.delete_entry("seg-alpha", "alpha/a")
        self.assertEqual(deleted.committed_cursor, 2)
        proof = self.store.build_proof("seg-alpha")
        self.assertEqual(proof["used_bytes"], 104)
        self.assertEqual(proof["remaining_bytes"], 152)
        self.assertEqual(len(list((self.workspace / "entries").rglob("*.bin"))), 4)
        self.assertEqual(self.store.verify_all()["code"], "OK")

    def test_capacity_override_reports_e_capacity(self) -> None:
        with self.assertRaises(DiagnosticError) as caught:
            self.store.verify_all(capacity_override={"seg-alpha": 111})
        self.assertEqual(caught.exception.code, "E_CAPACITY")
        self.assertEqual(caught.exception.segment_id, "seg-alpha")
        self.assertEqual(caught.exception.details["required_bytes"], 112)

    def test_view_rebuild_is_whole_and_stable(self) -> None:
        orders = self.store.get_view("view-orders")
        self.assertFalse(orders["rebuilt"])
        self.store.put_entry("seg-alpha", "alpha/new", 32, "alpha-new")
        self.store.delete_entry("seg-alpha", "alpha/a")
        rebuilt = self.store.get_view("view-orders")
        self.assertTrue(rebuilt["rebuilt"])
        self.assertEqual(rebuilt["view_version"], 2)
        segments = rebuilt["differences"]["segments"]
        self.assertEqual(segments[0]["segment_id"], "seg-alpha")
        self.assertEqual([item["key"] for item in segments[0]["added"]], ["alpha/new"])
        self.assertEqual([item["key"] for item in segments[0]["deleted"]], ["alpha/a"])
        again = self.store.get_view("view-orders")
        self.assertFalse(again["rebuilt"])
        self.assertEqual(again["differences"], rebuilt["differences"])

    def test_audit_view_lists_digest_change(self) -> None:
        audit = self.store.get_view("view-audit")
        self.assertTrue(audit["rebuilt"])
        alpha = next(
            segment
            for segment in audit["differences"]["segments"]
            if segment["segment_id"] == "seg-alpha"
        )
        self.assertEqual([item["key"] for item in alpha["digest_changed"]], ["alpha/a", "alpha/b"])

    def test_dashboard_refresh_is_read_only_and_stably_ordered(self) -> None:
        before = self.store.verify_all()["ledger"]
        first = dashboard_snapshot(self.store)
        second = dashboard_snapshot(self.store)
        after = self.store.verify_all()["ledger"]
        self.assertEqual(before, after)
        self.assertEqual(canonical_text(first), canonical_text(second))
        self.assertEqual(render_text_dashboard(first), render_text_dashboard(second))
        self.assertEqual(
            [segment["segment_id"] for segment in first["segments"]],
            ["seg-alpha", "seg-beta"],
        )

    def test_ledger_digest_tamper_is_e_snapshot(self) -> None:
        self.store.connection.execute(
            "UPDATE ledger_state SET content_digest = ? WHERE singleton = 1",
            ("sha256:" + "f" * 64,),
        )
        with self.assertRaises(DiagnosticError) as caught:
            self.store.verify_all()
        self.assertEqual(caught.exception.code, "E_SNAPSHOT")
        self.assertEqual(caught.exception.field, "ledger.content_digest")

    def test_truncated_content_is_e_integrity(self) -> None:
        row = self.store.connection.execute(
            "SELECT * FROM entries WHERE segment_id='seg-alpha' AND entry_key='alpha/a'"
        ).fetchone()
        path = self.store.paths.absolute_entry_path(row["file_relpath"])
        with path.open("r+b") as handle:
            handle.truncate(10)
        with self.assertRaises(DiagnosticError) as caught:
            self.store.verify_all()
        self.assertEqual(caught.exception.code, "E_INTEGRITY")
        self.assertEqual(caught.exception.segment_id, "seg-alpha")
        self.assertEqual(caught.exception.entry_key, "alpha/a")


if __name__ == "__main__":
    unittest.main()
