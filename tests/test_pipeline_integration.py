import json
import os
import shutil
import tempfile
import unittest

from syncpipe.cli import _run_scenario
from syncpipe.engine import Engine, rebuild_from_anchor
from syncpipe.errors import (
    E_CHAIN_BROKEN,
    E_LOG_TRUNCATED,
    E_SNAPSHOT_HASH,
    E_VERSION_DIVERGE,
)
from syncpipe.generator import generate_all
from syncpipe.recovery import recover
from syncpipe.state import canonical_bytes
from syncpipe.storage import Storage
from syncpipe.validator import validate_directory

SEED = 20261119


class PipelineIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.streams = generate_all(SEED)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _synthesize(self, name):
        root = os.path.join(self.tmp, name)
        summary = _run_scenario(root, self.streams[name], SEED)
        return root, summary

    def test_all_scenarios_validate_clean(self):
        for name in ("normal_game", "conflict_game", "catchup_game", "kill_game"):
            root, _ = self._synthesize(name)
            report = validate_directory(root, SEED)
            self.assertTrue(report["ok"], msg=json.dumps(report["errors"]))

    def test_compaction_preserves_reconstructibility(self):
        root, summary = self._synthesize("normal_game")
        storage = Storage(root)
        manifest = storage.load_manifest()
        engine = Engine(storage, SEED)
        engine.manifest = manifest
        compaction = engine.compact()
        self.assertIn(0, compaction["kept_snapshots"])
        self.assertIn(summary["head_version"], compaction["kept_snapshots"])
        # Rebuild head from the oldest retained anchor so logs are replayed.
        oldest_anchor = min(storage.available_snapshot_versions())
        self.assertLess(oldest_anchor, summary["head_version"])
        rebuilt, state, replayed = rebuild_from_anchor(
            storage, summary["head_version"], anchor_version=oldest_anchor
        )
        direct = storage.read_snapshot_file(summary["head_version"])
        self.assertEqual(canonical_bytes(state), canonical_bytes(direct["state"]))
        self.assertTrue(
            any(entry["version"] > oldest_anchor for entry in replayed)
        )
        self.assertTrue(validate_directory(root, SEED)["ok"])

    def test_catchup_equals_direct_snapshot(self):
        root, summary = self._synthesize("catchup_game")
        storage = Storage(root)
        anchors = [v for v in storage.available_snapshot_versions() if v < summary["head_version"]]
        anchor = min(anchors)
        rebuilt, state, _ = rebuild_from_anchor(storage, summary["head_version"])
        direct = storage.read_snapshot_file(summary["head_version"])
        self.assertEqual(state, direct["state"])
        self.assertLess(anchor, summary["head_version"])

    def test_snapshot_tampering_is_detected(self):
        root, _ = self._synthesize("normal_game")
        snap_path = os.path.join(root, Storage.snapshot_filename(4))
        with open(snap_path, "rb") as handle:
            snapshot = json.loads(handle.read())
        snapshot["state"]["entities"]["INJECTED"] = {"entity_id": "INJECTED", "fields": {}}
        with open(snap_path, "wb") as handle:
            handle.write(canonical_bytes(snapshot) + b"\n")
        report = validate_directory(root, SEED)
        self.assertFalse(report["ok"])
        self.assertEqual(report["errors"][0]["code"], E_SNAPSHOT_HASH)
        self.assertEqual(report["first_mismatch"]["version"], 4)

    def test_log_truncation_is_quarantined_and_reported(self):
        root, _ = self._synthesize("kill_game")
        storage = Storage(root)
        segment = storage.list_log_segments()[1]
        with open(segment, "ab") as handle:
            handle.write(b'{"version": 1, "tick": 1, "broken')
            handle.flush()
            os.fsync(handle.fileno())
        entries, errors = storage.read_all_log_entries()
        self.assertTrue(errors)
        self.assertEqual(errors[0]["code"], E_LOG_TRUNCATED)
        quarantine = os.listdir(os.path.join(root, "quarantine"))
        self.assertTrue(quarantine)
        report = validate_directory(root, SEED)
        self.assertFalse(report["ok"])
        self.assertEqual(report["first_mismatch"]["kind"], "log_truncation")

    def test_chain_broken_detects_parent_gap(self):
        root, _ = self._synthesize("normal_game")
        storage = Storage(root)
        manifest = storage.load_manifest()
        manifest["versions"][3]["parent_version"] = 999
        storage.write_manifest(manifest)
        with self.assertRaises(Exception) as ctx:
            storage.verify_parent_chain(manifest)
        self.assertEqual(ctx.exception.code, E_CHAIN_BROKEN)

    def test_directory_replacement_detects_version_divergence(self):
        root_a, _ = self._synthesize("normal_game")
        root_b, _ = self._synthesize("conflict_game")
        storage_a = Storage(root_a)
        storage_b = Storage(root_b)
        with self.assertRaises(Exception) as ctx:
            Storage.detect_replacement(
                storage_a.load_manifest(),
                storage_b.load_manifest(),
            )
        self.assertEqual(ctx.exception.code, E_VERSION_DIVERGE)

    def test_recovery_after_forced_kill_matches_reference(self):
        # Simulate a worker that stopped early without a shutdown marker by
        # running a full reference then replaying only its durable prefix.
        ref_root, ref_summary = self._synthesize("kill_game")
        kill_root = os.path.join(self.tmp, "kill_prefix")
        storage = Storage(kill_root)
        ref_storage = Storage(ref_root)
        ref_manifest = ref_storage.load_manifest()
        stop_at = max(6, ref_summary["head_version"] // 2)

        # Copy prefix snapshots/logs to build a crashed-worker directory.
        os.makedirs(os.path.join(kill_root, "snapshots"), exist_ok=True)
        os.makedirs(os.path.join(kill_root, "logs"), exist_ok=True)
        prefix_manifest = {
            "schema": ref_manifest["schema"],
            "seed": SEED,
            "head_version": stop_at,
            "versions": [v for v in ref_manifest["versions"] if v["version"] <= stop_at],
            "snapshot_files": [
                f for f in ref_manifest["snapshot_files"] if f["version"] <= stop_at
            ],
            "compactions": [],
            "shutdown": False,
        }
        for item in prefix_manifest["snapshot_files"]:
            shutil.copy(
                os.path.join(ref_root, item["file"]),
                os.path.join(kill_root, item["file"]),
            )
        entries, _ = ref_storage.read_all_log_entries()
        for entry in entries:
            if entry["version"] <= stop_at:
                storage.append_log_entry(entry)
        storage.close_log()
        storage.write_manifest(prefix_manifest)
        self.assertFalse(storage.has_shutdown_marker())

        report = recover(storage)
        self.assertTrue(report["forced_termination"])
        self.assertFalse(report["clean_shutdown"])
        expected = next(
            v for v in ref_manifest["versions"] if v["version"] == report["recovered_version"]
        )
        self.assertEqual(report["recovered_state_hash"], expected["state_hash"])

    def test_two_runs_are_byte_identical(self):
        root_a = os.path.join(self.tmp, "a")
        root_b = os.path.join(self.tmp, "b")
        _run_scenario(root_a, self.streams["conflict_game"], SEED)
        _run_scenario(root_b, self.streams["conflict_game"], SEED)
        for relative in ("manifest.json", "shutdown.marker"):
            with open(os.path.join(root_a, relative), "rb") as handle:
                bytes_a = handle.read()
            with open(os.path.join(root_b, relative), "rb") as handle:
                bytes_b = handle.read()
            self.assertEqual(bytes_a, bytes_b)


if __name__ == "__main__":
    unittest.main()
