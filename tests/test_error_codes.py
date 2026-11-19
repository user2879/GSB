import json
import os
import shutil
import tempfile
import unittest

from syncpipe.cli import _run_scenario
from syncpipe.errors import ALL_CODES
from syncpipe.generator import generate_all
from syncpipe.storage import Storage
from syncpipe.validator import validate_directory

SEED = 20261119


class ErrorCodeCoverageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.streams = generate_all(SEED)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_all_seven_codes_are_defined(self):
        self.assertEqual(
            sorted(ALL_CODES),
            sorted([
                "E_BASE_VERSION",
                "E_PEER_GAP",
                "E_SNAPSHOT_HASH",
                "E_LOG_TRUNCATED",
                "E_VERSION_DIVERGE",
                "E_CONFLICT_UNRESOLVED",
                "E_CHAIN_BROKEN",
            ]),
        )

    def test_e_base_version_emitted_in_catchup_stream(self):
        root = os.path.join(self.tmp, "catchup")
        summary = _run_scenario(root, self.streams["catchup_game"], SEED)
        records = [e for e in summary["errors"] if e["code"] == "E_BASE_VERSION"]
        self.assertTrue(records)
        for record in records:
            self.assertIsNotNone(record["event_loc"])
            self.assertIn("peer_id", record["event_loc"])

    def test_e_peer_gap_on_broken_sequence_and_illegal_tick(self):
        good = self.streams["normal_game"]
        broken = dict(good[5])
        broken["local_seq"] = 999
        illegal_tick = dict(good[6])
        illegal_tick["tick"] = -3
        stream = good[:5] + [broken, illegal_tick] + good[7:]
        root = os.path.join(self.tmp, "gap")
        summary = _run_scenario(root, stream, SEED)
        codes = {e["code"] for e in summary["errors"]}
        self.assertIn("E_PEER_GAP", codes)
        gap_records = [e for e in summary["errors"] if e["code"] == "E_PEER_GAP"]
        self.assertTrue(any("gap" in e["reason"] or "illegal_tick" in e["reason"]
                            for e in gap_records))

    def test_e_snapshot_hash_and_e_log_truncated_and_chain_and_diverge(self):
        root = os.path.join(self.tmp, "normal")
        _run_scenario(root, self.streams["normal_game"], SEED)
        storage = Storage(root)

        # E_SNAPSHOT_HASH
        snap = os.path.join(root, Storage.snapshot_filename(2))
        with open(snap, "rb") as handle:
            payload = json.loads(handle.read())
        payload["state"]["entities"]["X"] = {"entity_id": "X", "fields": {}}
        with open(snap, "wb") as handle:
            handle.write(json.dumps(payload, sort_keys=True).encode())
        report = validate_directory(root, SEED)
        self.assertIn("E_SNAPSHOT_HASH", {e["code"] for e in report["errors"]})

        # E_LOG_TRUNCATED
        root2 = os.path.join(self.tmp, "kill")
        _run_scenario(root2, self.streams["kill_game"], SEED)
        segment = Storage(root2).list_log_segments()[0]
        with open(segment, "ab") as handle:
            handle.write(b'{"broken":')
        report2 = validate_directory(root2, SEED)
        self.assertIn("E_LOG_TRUNCATED", {e["code"] for e in report2["errors"]})

        # E_CHAIN_BROKEN
        root3 = os.path.join(self.tmp, "chain")
        _run_scenario(root3, self.streams["normal_game"], SEED)
        storage3 = Storage(root3)
        manifest = storage3.load_manifest()
        manifest["versions"][2]["parent_version"] = 4242
        storage3.write_manifest(manifest)
        with self.assertRaises(Exception) as ctx:
            storage3.verify_parent_chain(manifest)
        self.assertEqual(ctx.exception.code, "E_CHAIN_BROKEN")

        # E_VERSION_DIVERGE
        root4 = os.path.join(self.tmp, "other")
        _run_scenario(root4, self.streams["conflict_game"], SEED)
        with self.assertRaises(Exception) as ctx:
            Storage.detect_replacement(
                storage3.load_manifest(), Storage(root4).load_manifest()
            )
        self.assertEqual(ctx.exception.code, "E_VERSION_DIVERGE")

    def test_e_conflict_unresolved_in_kill_stream(self):
        root = os.path.join(self.tmp, "kill")
        summary = _run_scenario(root, self.streams["kill_game"], SEED)
        records = [e for e in summary["errors"] if e["code"] == "E_CONFLICT_UNRESOLVED"]
        self.assertTrue(records)
        for record in records:
            self.assertIsNotNone(record["version"])
            self.assertIsNotNone(record["event_loc"])


if __name__ == "__main__":
    unittest.main()
