from __future__ import annotations

import base64
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from chunkdeps.engine import Workbench
from chunkdeps.errors import (
    E_AUDIT,
    E_CHUNK,
    E_INTEGRITY,
    E_STATE,
    E_VERSION,
    BusinessError,
)
from chunkdeps.util import b64encode, digest_bytes, iter_file_chunks


class EngineTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = Path(tempfile.mkdtemp(prefix="chunkdeps-test-"))
        self.root = self.temp / "workbench"
        self.source = self.temp / "sample.bin"
        self.data = (b"deterministic-chunk-data:" * 380) + b"tail"
        self.source.write_bytes(self.data)
        self.workbench = Workbench(self.root)

    def tearDown(self) -> None:
        try:
            self.workbench.close()
        except Exception:
            pass
        shutil.rmtree(self.temp, ignore_errors=True)

    def create(self, key: str, chunk_size: int, tags: dict[str, str] | None = None, expected=None):
        return self.workbench.create_session(
            {
                "object_key": key,
                "chunk_size": chunk_size,
                "total_size": len(self.data),
                "tags": tags or {},
                "expected_base_version": expected,
            }
        )

    def upload_all(self, session_id: str, chunk_size: int, data: bytes | None = None) -> list[dict]:
        payload = self.data if data is None else data
        results = []
        count = (len(payload) + chunk_size - 1) // chunk_size
        for index in range(count):
            part = payload[index * chunk_size : (index + 1) * chunk_size]
            results.append(
                self.workbench.upload_chunk(
                    {
                        "session_id": session_id,
                        "chunk_index": index,
                        "data_b64": b64encode(part),
                        "digest": digest_bytes(part),
                    }
                )
            )
        return results

    def put(self, key: str, chunk_size: int, tags: dict[str, str] | None = None, expected=None):
        session = self.create(key, chunk_size, tags, expected)
        self.upload_all(session["session_id"], chunk_size)
        return self.workbench.commit_session({"session_id": session["session_id"]})


class DeterministicChunkTests(EngineTestBase):
    def test_split_is_byte_deterministic_and_contiguous(self) -> None:
        first = list(iter_file_chunks(self.source, 257))
        second = list(iter_file_chunks(self.source, 257))
        self.assertEqual(first, second)
        self.assertEqual([index for index, _, _ in first], list(range(len(first))))
        self.assertEqual(b"".join(payload for _, payload, _ in first), self.data)
        for index, payload, digest in first:
            self.assertEqual(digest, digest_bytes(payload))
            expected_size = 257 if index < len(first) - 1 else len(self.data) - 257 * (len(first) - 1)
            self.assertEqual(len(payload), expected_size)

    def test_invalid_chunk_size_empty_file_and_wrong_chunk_size(self) -> None:
        with self.assertRaises(BusinessError) as caught:
            self.workbench.create_session(
                {"object_key": "bad/size", "chunk_size": 0, "total_size": 10}
            )
        self.assertEqual(caught.exception.code, E_CHUNK)
        with self.assertRaises(BusinessError) as caught:
            self.workbench.create_session(
                {"object_key": "bad/empty", "chunk_size": 10, "total_size": 0}
            )
        self.assertEqual(caught.exception.code, E_CHUNK)
        session = self.create("bad/chunk", 512)
        with self.assertRaises(BusinessError) as caught:
            self.workbench.upload_chunk(
                {
                    "session_id": session["session_id"],
                    "chunk_index": 0,
                    "data_b64": b64encode(b"too short"),
                }
            )
        self.assertEqual(caught.exception.code, E_CHUNK)
        self.assertIn("expected_size", caught.exception.details)


class StateMachineTests(EngineTestBase):
    def test_abort_retains_chunks_resume_skips_bitmap_and_commit_is_terminal(self) -> None:
        session = self.create("state/machine.bin", 333, {"case": "state"})
        session_id = session["session_id"]
        for index in (0, 2, 1):
            part = self.data[index * 333 : (index + 1) * 333]
            self.workbench.upload_chunk(
                {
                    "session_id": session_id,
                    "chunk_index": index,
                    "data_b64": b64encode(part),
                }
            )
        aborted = self.workbench.abort_session({"session_id": session_id})
        self.assertEqual(aborted["session"]["state"], "aborted")
        self.assertEqual(aborted["chunks_retained"], 3)
        self.assertEqual(len(aborted["session"]["bitmap"]), session["chunk_count"])

        resumed = self.workbench.resume_session({"session_id": session_id})
        self.assertEqual(resumed["state"], "uploading")
        self.assertEqual(len(resumed["skipped_indices"]), 3)
        self.assertNotIn(0, resumed["missing_indices"])
        self.assertEqual(resumed["transition"], {"from_state": "aborted", "to_state": "uploading"})

        for index in resumed["missing_indices"]:
            part = self.data[index * 333 : (index + 1) * 333]
            self.workbench.upload_chunk(
                {
                    "session_id": session_id,
                    "chunk_index": index,
                    "data_b64": b64encode(part),
                }
            )
        committed = self.workbench.commit_session({"session_id": session_id})
        self.assertEqual(committed["session"]["state"], "committed")
        self.assertTrue(committed["session"]["completed"])
        with self.assertRaises(BusinessError) as caught:
            self.workbench.resume_session({"session_id": session_id})
        self.assertEqual(caught.exception.code, E_STATE)
        with self.assertRaises(BusinessError) as caught:
            self.workbench.abort_session({"session_id": session_id})
        self.assertEqual(caught.exception.code, E_STATE)
        with self.assertRaises(BusinessError) as caught:
            self.workbench.commit_session({"session_id": session_id})
        self.assertEqual(caught.exception.code, E_STATE)

        detail = self.workbench.get_session({"session_id": session_id})
        states = [(event["from_state"], event["to_state"]) for event in detail["events"]]
        self.assertIn(("created", "uploading"), states)
        self.assertIn(("uploading", "aborted"), states)
        self.assertIn(("aborted", "uploading"), states)
        self.assertIn(("verifying", "committed"), states)
        self.assertEqual([chunk["chunk_index"] for chunk in detail["chunks"]], list(range(session["chunk_count"])))


class IntegrityTests(EngineTestBase):
    def test_integrity_failure_keeps_session_and_never_makes_partial_object(self) -> None:
        session = self.create("integrity/object.bin", 401)
        self.upload_all(session["session_id"], 401)
        first = self.workbench.get_session({"session_id": session["session_id"]})["chunks"][0]
        path = self.workbench.ledger.resolve(first["file_path"])
        path.write_bytes(path.read_bytes()[:-7])
        with self.assertRaises(BusinessError) as caught:
            self.workbench.commit_session({"session_id": session["session_id"]})
        self.assertEqual(caught.exception.code, E_INTEGRITY)
        detail = self.workbench.get_session({"session_id": session["session_id"]})
        self.assertEqual(detail["state"], "uploading")
        self.assertEqual(detail["version"], None)
        self.assertEqual(self.workbench.list_objects()["version_count"], 0)
        self.assertEqual(list(self.workbench.ledger.temp_dir.glob("*")), [])

        part = self.data[:401]
        self.workbench.upload_chunk(
            {
                "session_id": session["session_id"],
                "chunk_index": 0,
                "data_b64": b64encode(part),
                "digest": digest_bytes(part),
            }
        )
        committed = self.workbench.commit_session({"session_id": session["session_id"]})
        self.assertTrue(committed["session"]["completed"])
        self.assertEqual(committed["version"]["content_digest"], digest_bytes(self.data))


class DependencyTests(EngineTestBase):
    def test_retransmit_invalidates_descendants_in_stable_topology_and_preserves_unaffected(self) -> None:
        self.put("branch/a.txt", 512, {"v": "1"})
        self.put("other/control.txt", 700, {"v": "control"})
        before = {
            node["node_id"]: node
            for node in self.workbench.get_graph()["nodes"]
        }
        session = self.create("branch/a.txt", 600, {"v": "2"})
        self.upload_all(session["session_id"], 600)
        part = self.data[:600]
        reupload = self.workbench.upload_chunk(
            {
                "session_id": session["session_id"],
                "chunk_index": 0,
                "data_b64": b64encode(part),
                "digest": digest_bytes(part),
            }
        )
        invalidation = reupload["invalidation"]
        affected = invalidation["affected"]
        positions = {entry["node_id"]: index for index, entry in enumerate(affected)}
        chunk_id = f"chunk:{session['session_id']}:00000000"
        manifest_id = f"manifest:{session['session_id']}"
        stage_id = f"stage-metadata:{session['session_id']}"
        summary_id = f"session-summary:{session['session_id']}"
        self.assertLess(positions[chunk_id], positions[manifest_id])
        self.assertLess(positions[manifest_id], positions[stage_id])
        self.assertLess(positions[stage_id], positions[summary_id])
        self.assertTrue(affected[0]["direct"])
        self.assertTrue(any(not entry["direct"] for entry in affected))
        for entry in affected:
            self.assertEqual(
                entry["entry_version_after"], entry["entry_version_before"] + 1
            )
            self.assertIsNone(entry["digest_after"])

        after = {
            node["node_id"]: node
            for node in self.workbench.get_graph()["nodes"]
        }
        for node_id in (
            "metadata:other/control.txt:v1",
            "object-key:other/control.txt",
            "prefix-view:other",
        ):
            self.assertEqual(node_id in before, node_id in after)
            self.assertEqual(before[node_id]["entry_version"], after[node_id]["entry_version"])
            self.assertEqual(before[node_id]["cached_digest"], after[node_id]["cached_digest"])
        self.assertTrue(self.workbench.get_graph()["validation"]["valid"])


class VersionAuditSnapshotTests(EngineTestBase):
    def test_whole_version_replacement_audit_and_snapshot_restore(self) -> None:
        v1 = self.put("versioned/item.bin", 512, {"release": "one"})
        self.assertEqual(v1["version"]["version"], 1)
        v2 = self.put(
            "versioned/item.bin", 777, {"release": "two", "extra": "yes"}, expected=1
        )
        self.assertEqual(v2["version"]["version"], 2)
        self.assertEqual(v2["replacement"], "whole")
        self.assertEqual(v2["previous_version"]["status"], "expired")
        changed = {item["field"] for item in v2["field_changes"]}
        self.assertIn("chunk_size", changed)
        self.assertIn("tags", changed)
        self.assertIn("metadata_digest", changed)
        versions = self.workbench.list_objects({"object_key": "versioned/item.bin"})["versions"]
        self.assertEqual([item["status"] for item in versions], ["expired", "active"])
        self.assertEqual(self.workbench.audit()["passed"], True)

        snapshot = self.workbench.snapshot_save()
        self.assertGreater(snapshot["object_count"], 0)
        self.workbench.delete_object({"object_key": "versioned/item.bin"})
        self.assertEqual(self.workbench.list_objects()["active_count"], 0)
        restored = self.workbench.snapshot_restore()
        self.assertEqual(restored["mode"], "whole-state-replacement")
        self.assertFalse(restored["incremental_merge"])
        self.assertIn("versioned/item.bin", restored["comparison"]["metadata_version_changes"])
        self.assertTrue(restored["comparison"]["metadata_version_changes"])
        self.assertEqual(self.workbench.audit()["passed"], True)
        active = self.workbench.list_objects({"object_key": "versioned/item.bin"})["objects"]
        self.assertEqual(active[0]["version"], 2)

        session = self.workbench.ledger.session_row(active[0]["session_id"])
        first = self.workbench.ledger.chunk_row(session["session_id"], 0)
        path = self.workbench.ledger.resolve(first["file_path"])
        path.write_bytes(path.read_bytes()[:-3])
        with self.assertRaises(BusinessError) as caught:
            self.workbench.audit()
        self.assertEqual(caught.exception.code, E_AUDIT)
        self.assertEqual(caught.exception.details["first_deviation"]["object_key"], "versioned/item.bin")
        self.assertEqual(caught.exception.details["first_deviation"]["chunk_index"], 0)


class FaultAndRecoveryTests(EngineTestBase):
    def test_fault_codes_are_structured_and_logged(self) -> None:
        session = self.create("fault/chunk.bin", 512)
        part = self.data[:512]
        with self.assertRaises(BusinessError) as caught:
            self.workbench.upload_chunk(
                {
                    "session_id": session["session_id"],
                    "chunk_index": 0,
                    "data_b64": b64encode(part),
                    "inject": "truncate-chunk",
                }
            )
        self.assertEqual(caught.exception.code, E_CHUNK)

        integrity = self.create("fault/integrity.bin", 512)
        self.upload_all(integrity["session_id"], 512)
        with self.assertRaises(BusinessError) as caught:
            self.workbench.commit_session(
                {"session_id": integrity["session_id"], "inject": "corrupt-metadata"}
            )
        self.assertEqual(caught.exception.code, E_INTEGRITY)
        retained = self.workbench.get_session({"session_id": integrity["session_id"]})
        self.assertEqual(retained["state"], "uploading")

        a = self.create("fault/version.bin", 512, expected=0)
        self.upload_all(a["session_id"], 512)
        b = self.create("fault/version.bin", 700, expected=0)
        self.upload_all(b["session_id"], 700)
        self.workbench.commit_session({"session_id": b["session_id"]})
        with self.assertRaises(BusinessError) as caught:
            self.workbench.commit_session(
                {"session_id": a["session_id"], "expected_base_version": 0}
            )
        self.assertEqual(caught.exception.code, E_VERSION)
        self.assertEqual(caught.exception.details["actual_base_version"], 1)

        injected = self.create("fault/version-injected.bin", 512)
        self.upload_all(injected["session_id"], 512)
        with self.assertRaises(BusinessError) as caught:
            self.workbench.commit_session(
                {"session_id": injected["session_id"], "inject": "version-conflict"}
            )
        self.assertEqual(caught.exception.code, E_VERSION)
        self.assertEqual(caught.exception.details["injection"], "version-conflict")

        log_path = self.root / "logs" / "events.jsonl"
        codes = []
        for line in log_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("code"):
                codes.append(record["code"])
        self.assertIn(E_CHUNK, codes)
        self.assertIn(E_INTEGRITY, codes)
        self.assertIn(E_VERSION, codes)

    def test_restart_marks_merge_unknown_and_requires_rerun(self) -> None:
        stable = self.put("recovery/stable.bin", 600)
        session = self.create("recovery/pending.bin", 700)
        self.upload_all(session["session_id"], 700)
        self.workbench.conn.execute(
            """
            UPDATE sessions
               SET state='verifying', commit_unknown=0, recovery_status='running'
             WHERE session_id=?
            """,
            (session["session_id"],),
        )
        self.workbench.close()
        self.workbench = Workbench(self.root)
        status = self.workbench.recovery_status()
        unknown = [item["session_id"] for item in status["unknown_merge_sessions"]]
        self.assertIn(session["session_id"], unknown)
        self.assertFalse(status["exit_report_available"])
        before = self.workbench.list_objects({"object_key": "recovery/pending.bin"})
        self.assertEqual(before["version_count"], 0)
        self.assertIn(
            stable["version"]["object_key"],
            [item["object_key"] for item in self.workbench.list_objects()["objects"]],
        )
        rerun = self.workbench.commit_session({"session_id": session["session_id"]})
        self.assertEqual(rerun["version"]["version"], 1)
        after = self.workbench.list_objects({"object_key": "recovery/pending.bin"})
        self.assertEqual(after["version_count"], 1)


if __name__ == "__main__":
    unittest.main()
