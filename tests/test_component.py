from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from docdb_index.canonical import canonical_json, loads_strict
from docdb_index.engine import (
    apply_changes,
    diagnose,
    initialize_session,
    manifest,
    query_index,
)
from docdb_index.errors import (
    BoundError,
    DocumentError,
    LedgerError,
    ReplayError,
    SegmentError,
    StaleError,
)
from docdb_index.inject import inject_anomaly
from docdb_index.replay import replay_validate
from docdb_index.snapshot import (
    create_snapshot,
    load_snapshot_manifest,
    restore_snapshot,
    snapshot_state,
)
from docdb_index.storage import load_segments, open_session
from docdb_index.validation import validate_declarations
from docdb_index.verify import verify_structure, verify_session

ROOT = Path(__file__).resolve().parents[1]


def make_specs(count: int = 24, capacity: int = 4, version: int = 5):
    documents = []
    for number in range(1, count + 1):
        documents.append(
            {
                "id": f"d-{number:04d}",
                "profile": {
                    "region": ("east", "north", "south", "west")[number % 4],
                    "rank": number,
                },
                "metrics": {
                    "score": (number * 7) % 101,
                    "other": number * 2,
                },
                "attributes": {"active": number % 3 != 0},
            }
        )
    root = {
        "collections": [
            {
                "name": "items",
                "version": version,
                "primary_key": "id",
                "documents": documents,
                "indexes": [
                    {
                        "path": "metrics.score",
                        "segment_capacity": capacity,
                        "allow_null": False,
                    }
                ],
            }
        ]
    }
    return validate_declarations(root)


def digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ComponentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="docdb-tests-")
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def init(self, name: str, specs=None, **kwargs) -> Path:
        session = self.root / name
        initialize_session(session, specs or make_specs(), **kwargs)
        return session

    def test_document_validation_and_no_partial_state(self) -> None:
        invalid = {
            "collections": [
                {
                    "name": "bad",
                    "version": 1,
                    "primary_key": "id",
                    "documents": [{"id": "a", "meta": {"x": 1}}],
                    "indexes": [
                        {
                            "path": "meta.bad-path",
                            "segment_capacity": 3,
                            "allow_null": False,
                        }
                    ],
                }
            ]
        }
        with self.assertRaises(DocumentError) as raised:
            validate_declarations(invalid)
        self.assertEqual(raised.exception.code, "E_DOC")
        self.assertEqual(raised.exception.details["field_path"], "meta.bad-path")

        session = self.root / "invalid-session"
        input_path = self.root / "invalid.json"
        input_path.write_text(json.dumps(invalid), encoding="utf-8")
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "docdb_index",
                "init",
                "--session",
                str(session),
                "--input",
                str(input_path),
                "--reset",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(process.returncode, 2)
        payload = json.loads(process.stdout)
        self.assertEqual(payload["code"], "E_DOC")
        self.assertFalse((session / "session.db").exists())

        existing = self.init("existing-session")
        before_digest = digest_file(existing / "session.db")
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "docdb_index",
                "init",
                "--session",
                str(existing),
                "--input",
                str(input_path),
                "--reset",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(process.returncode, 2)
        self.assertEqual(digest_file(existing / "session.db"), before_digest)
        self.assertEqual(manifest(existing)["collection_versions"], {"items": 5})

        duplicate = {
            "collections": [
                {
                    "name": "dup",
                    "version": 1,
                    "primary_key": "id",
                    "documents": [
                        {"id": "same", "metrics": {"score": 1}},
                        {"id": "same", "metrics": {"score": 2}},
                    ],
                    "indexes": [
                        {
                            "path": "metrics.score",
                            "segment_capacity": 2,
                            "allow_null": True,
                        }
                    ],
                }
            ]
        }
        with self.assertRaises(DocumentError) as duplicate_error:
            validate_declarations(duplicate)
        self.assertEqual(duplicate_error.exception.details["primary_key"], "same")

    def test_initial_build_is_byte_deterministic(self) -> None:
        specs = make_specs()
        first = self.init("first", specs)
        second = self.init("second", specs)
        first_manifest = manifest(first)
        second_manifest = manifest(second)
        self.assertEqual(first_manifest["segment_digests"], second_manifest["segment_digests"])
        self.assertEqual(first_manifest["segments"], second_manifest["segments"])

        blobs = []
        for session in (first, second):
            conn = open_session(session)
            try:
                rows = conn.execute(
                    "SELECT segment_id, content_blob FROM segments ORDER BY segment_id"
                ).fetchall()
                blobs.append([(row["segment_id"], bytes(row["content_blob"])) for row in rows])
            finally:
                conn.close()
        self.assertEqual(blobs[0], blobs[1])

    def test_incremental_only_replaces_affected_segments(self) -> None:
        session = self.init("incremental")
        before = manifest(session)
        conn = open_session(session)
        try:
            before_segments = {
                segment.segment_id: segment
                for segment in load_segments(conn, "items", "metrics.score", with_entries=True)
            }
        finally:
            conn.close()
        report = apply_changes(
            session,
            "items",
            [
                {
                    "op": "set_path",
                    "pk": "d-0001",
                    "path": "metrics.score",
                    "value": 5000,
                }
            ],
            batch_id="one-update",
            batch_size=1,
        )
        after = manifest(session)
        index_report = report["indexes"][0]
        affected = set(index_report["affected_segment_ids"])
        self.assertTrue(affected)
        self.assertIn("d-0001", canonical_json(report["indexes"][0]["derivation_chain"]))

        after_chunks = {
            item["segment_id"]: item for item in after["segments"]
        }
        for segment_id, old in before_segments.items():
            if segment_id in affected:
                continue
            self.assertIn(segment_id, after_chunks)
            self.assertEqual(after_chunks[segment_id]["digest"], old.digest)
            self.assertEqual(after_chunks[segment_id]["version"], old.version)
        conn = open_session(session)
        try:
            after_blobs = {
                row["segment_id"]: bytes(row["content_blob"])
                for row in conn.execute("SELECT segment_id, content_blob FROM segments")
            }
        finally:
            conn.close()
        for segment_id in index_report["unchanged_segment_ids"]:
            self.assertEqual(
                before_chunks_for(before, segment_id)["digest"],
                after_chunks[segment_id]["digest"],
            )
            self.assertEqual(before_segments[segment_id].content_blob, after_blobs[segment_id])

    def test_batch_order_and_batch_size_do_not_change_state(self) -> None:
        specs = make_specs()
        left = self.init("left", specs)
        right = self.init("right", specs)
        changes = [
            {"op": "set_path", "pk": "d-0001", "path": "metrics.score", "value": 2200},
            {"op": "set_path", "pk": "d-0002", "path": "metrics.score", "value": -20},
            {"op": "delete", "pk": "d-0003"},
            {"op": "set_path", "pk": "d-0024", "path": "metrics.score", "value": 900},
            {
                "op": "insert",
                "document": {
                    "id": "ins-0001",
                    "profile": {"region": "east", "rank": 99},
                    "metrics": {"score": 1500, "other": 0},
                    "attributes": {"active": True},
                },
            },
        ]
        apply_changes(left, "items", changes, batch_id="b", batch_size=1)
        apply_changes(right, "items", list(reversed(changes)), batch_id="b", batch_size=100)
        left_manifest = manifest(left)
        right_manifest = manifest(right)
        self.assertEqual(left_manifest["segments"], right_manifest["segments"])
        self.assertEqual(left_manifest["ledger_digest"], right_manifest["ledger_digest"])
        left_query = query_index(
            left, "items", "metrics.score", lo=-1000, hi=10000, limit=1000
        )
        right_full = query_index(
            right, "items", "metrics.score", lo=-1000, hi=10000, limit=1000, scan="full"
        )
        self.assertEqual(left_query["items"], right_full["items"])

    def test_stale_isolation_reclaim_and_ledger_idempotence(self) -> None:
        session = self.init("stale")
        changes = [
            {"op": "set_path", "pk": "d-0001", "path": "metrics.score", "value": 777},
            {"op": "delete", "pk": "d-0002"},
            {"op": "set_path", "pk": "d-0009", "path": "metrics.score", "value": -9},
        ]
        apply_changes(
            session,
            "items",
            changes,
            batch_id="retain",
            retain_stale=True,
        )
        report = diagnose(session, "items", "metrics.score")
        self.assertGreaterEqual(sum(item["stale_entry_count"] for item in report["segments"]), 2)
        self.assertGreaterEqual(report["stale_ledger_count"], 3)
        reasons = {row["reason"] for row in report["stale_ledger"]}
        self.assertIn("document_delete", reasons)
        self.assertIn("key_migration", reasons)

        indexed = query_index(
            session, "items", "metrics.score", lo=-10000, hi=10000, limit=1000
        )
        full = query_index(
            session,
            "items",
            "metrics.score",
            lo=-10000,
            hi=10000,
            limit=1000,
            scan="full",
        )
        self.assertEqual(indexed["items"], full["items"])
        self.assertGreaterEqual(indexed["filtered_stale"], 2)
        filtered_report = diagnose(session, "items", "metrics.score")
        self.assertGreaterEqual(filtered_report["isolation_filtered_total"], 2)

        ledger_before = filtered_report["stale_ledger"]
        reclaim_report = apply_changes(
            session,
            "items",
            [],
            batch_id="reclaim",
            retain_stale=False,
        )
        self.assertGreaterEqual(
            reclaim_report["indexes"][0]["stale_entries_reclaimed"],
            2,
        )
        second_reclaim = apply_changes(
            session,
            "items",
            [],
            batch_id="reclaim-again",
            retain_stale=False,
        )
        self.assertEqual(second_reclaim["indexes"][0]["stale_entries_reclaimed"], 0)
        after = diagnose(session, "items", "metrics.score")
        self.assertEqual(sum(item["stale_entry_count"] for item in after["segments"]), 0)
        self.assertEqual(after["stale_ledger"], ledger_before)
        self.assertEqual(len({canonical_json(row) for row in after["stale_ledger"]}), len(after["stale_ledger"]))
        self.assertEqual(canonical_json(after), canonical_json(diagnose(session, "items", "metrics.score")))
        verify_structure(session)

    def test_pagination_and_duplicate_keys_are_stable(self) -> None:
        documents = [
            {"id": "a", "metrics": {"score": 10}},
            {"id": "c", "metrics": {"score": 10}},
            {"id": "b", "metrics": {"score": 10}},
            {"id": "d", "metrics": {"score": 20}},
            {"id": "e", "metrics": {"score": 30}},
        ]
        specs = validate_declarations(
            {
                "collections": [
                    {
                        "name": "tiny",
                        "version": 1,
                        "primary_key": "id",
                        "documents": documents,
                        "indexes": [
                            {
                                "path": "metrics.score",
                                "segment_capacity": 2,
                                "allow_null": False,
                            }
                        ],
                    }
                ]
            }
        )
        session = self.init("tiny", specs)
        first_page = query_index(
            session, "tiny", "metrics.score", offset=0, limit=2
        )
        second_page = query_index(
            session, "tiny", "metrics.score", offset=2, limit=2
        )
        all_items = query_index(
            session, "tiny", "metrics.score", offset=0, limit=20
        )
        self.assertEqual(
            first_page["items"] + second_page["items"],
            all_items["items"][:4],
        )
        self.assertEqual(first_page["next_offset"], 2)
        equal = query_index(session, "tiny", "metrics.score", eq=10, limit=20)
        self.assertEqual(
            [item["primary_key"] for item in equal["items"]],
            ["a", "b", "c"],
        )

    def test_stale_version_rejects_query_and_apply(self) -> None:
        session = self.init("version")
        conn = sqlite3.connect(session / "session.db")
        conn.execute("UPDATE collections SET collection_version=99 WHERE name='items'")
        conn.commit()
        conn.close()
        with self.assertRaises(StaleError) as raised:
            query_index(session, "items", "metrics.score", lo=0, hi=100)
        self.assertEqual(raised.exception.code, "E_STALE")
        self.assertEqual(raised.exception.details["collection_version"], 99)
        self.assertEqual(raised.exception.details["index_version"], 5)
        self.assertEqual(
            raised.exception.details["stale_dimension"],
            "index_version_vs_collection_version",
        )
        with self.assertRaises(StaleError):
            apply_changes(
                session,
                "items",
                [{"op": "delete", "pk": "d-0001"}],
                batch_id="blocked",
            )

    def test_apply_failure_is_atomic(self) -> None:
        session = self.init("atomic")
        before = manifest(session)
        with self.assertRaises(DocumentError):
            apply_changes(
                session,
                "items",
                [
                    {
                        "op": "set_path",
                        "pk": "d-0001",
                        "path": "metrics.score",
                        "value": 333,
                    },
                    {"op": "delete", "pk": "missing"},
                ],
                batch_id="will-fail",
            )
        after = manifest(session)
        self.assertEqual(before["collection_versions"], after["collection_versions"])
        self.assertEqual(before["index_versions"], after["index_versions"])
        self.assertEqual(before["segment_digests"], after["segment_digests"])
        conn = sqlite3.connect(session / "session.db")
        count = conn.execute("SELECT COUNT(*) FROM audit_batches").fetchone()[0]
        conn.close()
        self.assertEqual(count, 0)

    def test_snapshot_round_trip_and_itemized_differences(self) -> None:
        session = self.init("snapshot")
        baseline = snapshot_state(session)
        snapshot = create_snapshot(session, label="snap-000001")
        manifest_data = load_snapshot_manifest(snapshot["path"])
        self.assertEqual(manifest_data["content_digest"], snapshot["content_digest"])
        self.assertEqual(manifest_data["collection_versions"], {"items": 5})

        apply_changes(
            session,
            "items",
            [{"op": "set_path", "pk": "d-0001", "path": "metrics.score", "value": 1234}],
            batch_id="after-snapshot",
        )
        changed = snapshot_state(session)
        self.assertNotEqual(baseline["segment_digests"], changed["segment_digests"])
        second = create_snapshot(session, label="snap-000002")
        self.assertTrue(Path(snapshot["path"]).is_dir())
        self.assertTrue(second["read_only"])
        self.assertTrue((Path(session) / "snapshots" / "snap-000001" / "manifest.json").is_file())

        restored = restore_snapshot(session, snapshot["path"])
        self.assertTrue(restored["replaced"])
        self.assertFalse(restored["same_state"])
        self.assertTrue(restored["differences"])
        self.assertTrue(
            any(item["dimension"] == "segment_changes" for item in restored["differences"])
        )
        self.assertEqual(snapshot_state(session), baseline)
        self.assertEqual(manifest(session)["collection_versions"], {"items": 5})
        self.assertTrue(Path(restored["rollback_copy"]).is_file())

    def _prepared_for_injection(self, name: str) -> Path:
        session = self.init(name)
        apply_changes(
            session,
            "items",
            [
                {"op": "set_path", "pk": "d-0001", "path": "metrics.score", "value": 999},
                {"op": "delete", "pk": "d-0002"},
            ],
            batch_id="create-ledger",
        )
        return session

    def test_three_business_fault_codes(self) -> None:
        segment_session = self._prepared_for_injection("fault-segment")
        inject_anomaly(
            segment_session,
            collection_name="items",
            index_path="metrics.score",
            kind="segment",
        )
        with self.assertRaises(SegmentError) as segment_error:
            verify_session(segment_session, include_replay=False)
        self.assertEqual(segment_error.exception.code, "E_SEGMENT")
        self.assertEqual(segment_error.exception.details["field"], "digest")

        bound_session = self._prepared_for_injection("fault-bound")
        inject_anomaly(
            bound_session,
            collection_name="items",
            index_path="metrics.score",
            kind="bound",
        )
        with self.assertRaises(BoundError) as bound_error:
            verify_session(bound_session, include_replay=False)
        self.assertEqual(bound_error.exception.code, "E_BOUND")
        self.assertEqual(bound_error.exception.details["field"], "entry_count")

        ledger_session = self._prepared_for_injection("fault-ledger")
        inject_anomaly(
            ledger_session,
            collection_name="items",
            index_path="metrics.score",
            kind="ledger",
        )
        with self.assertRaises(LedgerError) as ledger_error:
            verify_session(ledger_session, include_replay=False)
        self.assertEqual(ledger_error.exception.code, "E_LEDGER")

        for session, code in (
            (segment_session, "E_SEGMENT"),
            (bound_session, "E_BOUND"),
            (ledger_session, "E_LEDGER"),
        ):
            lines = (session / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertIn(code, [json.loads(line)["code"] for line in lines])

    def test_replay_detects_first_tampered_segment(self) -> None:
        session = self.init("replay")
        apply_changes(
            session,
            "items",
            [{"op": "delete", "pk": "d-0001"}],
            batch_id="replay-batch",
        )
        report = replay_validate(session)
        self.assertGreaterEqual(report["batches_checked"], 1)
        verify_session(session)

        conn = sqlite3.connect(session / "session.db")
        table_name = conn.execute(
            "SELECT table_name FROM segments ORDER BY segment_id LIMIT 1"
        ).fetchone()[0]
        conn.execute(f'UPDATE "{table_name}" SET value_json=? WHERE seq=1', ("99999",))
        conn.commit()
        conn.close()
        with self.assertRaises(ReplayError) as raised:
            replay_validate(session)
        self.assertEqual(raised.exception.code, "E_REPLAY")
        self.assertEqual(raised.exception.details["field"], "digest")
        self.assertIn("segment_id", raised.exception.details)


def before_chunks_for(manifest_payload, segment_id):
    for item in manifest_payload["segments"]:
        if item["segment_id"] == segment_id:
            return item
    raise AssertionError(f"segment {segment_id} not found")


if __name__ == "__main__":
    unittest.main()
