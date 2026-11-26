import tempfile
import unittest
from pathlib import Path

from intent_router.ledger import Ledger, CacheEntry
from intent_router.snapshot import SnapshotBundle, SnapshotStore, config_hash


class SnapshotTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = SnapshotStore(Path(self.tmp.name) / "snap")

    def tearDown(self):
        self.tmp.cleanup()

    def _bundle(self, version):
        ledger = Ledger()
        ledger.put(
            CacheEntry(
                key="k1",
                kind="score",
                text_id="T1",
                referenced_entry_fp={"E1": "fp"},
                text_fp="tfp",
                chain_fp="cfp",
                gear_key="fb8-g2",
                payload={"x": version},
            )
        )
        return SnapshotBundle(
            version=version,
            config_hash=config_hash({"v": version}),
            gear_key="fb8-g2",
            head_revision="R1",
            products={"scores": {"T1": {"v": version}}},
            pending=[],
            consumed={"global": 1},
            ledger=ledger.to_snapshot(),
        )

    def test_commit_and_reload_roundtrip(self):
        self.store.commit(self._bundle(1))
        loaded = self.store.load()
        self.assertEqual(loaded.version, 1)
        ledger = self.store.load_ledger()
        self.assertIn("k1", ledger.entries)

    def test_version_conflict_detection(self):
        self.store.commit(self._bundle(1))
        with self.assertRaises(RuntimeError):
            self.store.commit(self._bundle(3), expected_version=99)

    def test_recovery_only_uses_committed_snapshot(self):
        self.store.commit(self._bundle(1))
        # Simulate restart: a brand new store over the same path reloads state.
        reopened = SnapshotStore(self.store.root)
        bundle = reopened.load()
        self.assertIsNotNone(bundle)
        self.assertEqual(bundle.products["scores"]["T1"]["v"], 1)
        self.assertEqual(bundle.pending, [])

    def test_missing_snapshot_returns_empty_ledger(self):
        empty = SnapshotStore(Path(self.tmp.name) / "none")
        self.assertIsNone(empty.load())
        self.assertEqual(empty.load_ledger().entries, {})


if __name__ == "__main__":
    unittest.main()
