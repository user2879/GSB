import json
import tempfile
import unittest
from pathlib import Path

from pcl.entries import CURRENT_ENTRY_VERSION, Entry, decode_entry, encode_entry
from pcl.errors import CorruptEntryError
from pcl.samples import write_sample_workspace
from pcl.store import CacheStore, UpgradeInterrupted


class StoreUpgradeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.manifest_path, self.paths = write_sample_workspace(self.root)
        self.store = CacheStore(self.root / "cache")

    def tearDown(self):
        self.tmp.cleanup()

    def _legacy(self, key, fmt):
        data = self.paths[key].read_bytes()
        from pcl.digest import digest_bytes

        entry = Entry(
            key=key,
            digest=digest_bytes(data),
            version=1,
            size=len(data),
            stage_id="stage-foundation",
        )
        self.store._entry_path(key).write_bytes(
            encode_entry(entry, fmt)
        )

    def test_upgrade_rewrites_legacy_and_skips_current(self):
        from pcl.manifest import load_manifest

        manifest = load_manifest(self.manifest_path, self.root)
        self._legacy("base-lib", 1)
        self._legacy("config", 2)
        self._legacy("codegen", 3)
        report = self.store.open_and_upgrade(manifest)
        self.assertEqual({u["key"] for u in report["upgraded"]},
                         {"base-lib", "config"})
        for key in ("base-lib", "config", "codegen"):
            entry = self.store.read_raw_entry(key)
            self.assertEqual(entry.format_version, CURRENT_ENTRY_VERSION)
        self.assertIn("refs", self.store.backfilled["base-lib"])
        self.assertIn("compressed", self.store.backfilled["config"])

    def test_kill_resumes_only_from_checkpoint(self):
        self._legacy("base-lib", 1)
        self._legacy("config", 1)
        self._legacy("codegen", 2)
        with self.assertRaises(UpgradeInterrupted):
            self.store.open_and_upgrade(simulate_kill_after=1)
        cp = json.loads(
            (self.root / "cache" / "upgrade_checkpoint.json").read_text()
        )
        self.assertEqual(cp["processed"], ["base-lib"])
        already = self.store.read_raw_entry("base-lib")
        self.assertEqual(already.format_version, CURRENT_ENTRY_VERSION)

        second_store = CacheStore(self.root / "cache")
        report = second_store.open_and_upgrade()
        cumulative = {u["key"] for u in report["upgraded"]}
        self.assertEqual(cumulative, {"base-lib", "codegen", "config"})
        # This process only touched the two pending entries; base-lib was
        # skipped because the checkpoint already marked it processed.
        this_run = set(report["processed"]) - set(cp["processed"])
        self.assertEqual(this_run, {"codegen", "config"})
        self.assertEqual(
            second_store.upgrade_log,
            [u for u in report["upgraded"] if u["key"] in this_run],
        )
        verification = second_store.upgrade_verification()
        self.assertTrue(verification["all_current"])
        self.assertTrue(verification["processed_covers_disk"])

    def test_corrupt_entry_reports_key_and_offset(self):
        self._legacy("base-lib", 1)
        path = self.store._entry_path("base-lib")
        path.write_bytes(b"PCL1\x01\x00")
        with self.assertRaises(CorruptEntryError) as ctx:
            self.store.open_and_upgrade()
        self.assertEqual(ctx.exception.details["key"], "base-lib")
        self.assertEqual(ctx.exception.details["offset"], 0)

    def test_atomic_tmp_files_never_left_behind(self):
        self._legacy("base-lib", 1)
        self.store.open_and_upgrade()
        leftovers = [
            p.name for p in self.store.cache_dir.iterdir()
            if p.name.endswith(".tmp")
        ]
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
