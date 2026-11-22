import tempfile
import unittest
from pathlib import Path

from dns_workbench.app import DeterministicClock
from dns_workbench.errors import VersionError
from dns_workbench.ledger import CacheLedger
from dns_workbench.records import CacheEntry, canonical_digest
from dns_workbench.snapshot import HitEvent, SnapshotStore
from dns_workbench.wire import TYPE_A


def entry(name: str, version: str = "v1", seq: int = 0) -> CacheEntry:
    item = CacheEntry(name=name, qtype=TYPE_A, value=f"A 10.0.0.{len(name)}", ttl=60,
                      source_version=version, content_digest="", seq=seq)
    item.content_digest = canonical_digest(item.normalized_payload())
    return item


class LedgerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = DeterministicClock()
        self.ledger = CacheLedger(capacity=16, clock=self.clock)

    def test_wholesale_replacement_marks_stale_and_lists_diff(self) -> None:
        self.ledger.initialize(zone_version=1, entries=[entry("a.test"), entry("b.test")])
        diff = self.ledger.replace(zone_version=2, entries=[entry("a.test"), entry("c.test", "v2")])
        self.assertTrue(diff.stale)
        self.assertEqual(diff.added, [("c.test", TYPE_A)])
        self.assertEqual(diff.removed, [("b.test", TYPE_A)])
        self.assertTrue(self.ledger.history[-1].stale)
        self.assertEqual(self.ledger.current().generation, 2)

    def test_format_version_change_triggers_replacement(self) -> None:
        self.ledger.initialize(zone_version=1, entries=[entry("a.test", "v1")])
        diff = self.ledger.replace(zone_version=1, entries=[entry("a.test", "v2")])
        self.assertTrue(diff.stale)
        self.assertEqual(self.ledger.current().format_versions, ("v2",))

    def test_zone_version_downgrade_rejected(self) -> None:
        self.ledger.initialize(zone_version=3, entries=[entry("a.test")])
        with self.assertRaises(VersionError):
            self.ledger.ensure_version(2, ("v1",))

    def test_no_change_no_replacement(self) -> None:
        self.ledger.initialize(zone_version=1, entries=[entry("a.test")])
        diff = self.ledger.replace(zone_version=1, entries=[entry("a.test")])
        self.assertFalse(diff.stale)


class SnapshotTest(unittest.TestCase):
    def test_save_and_verify_then_detect_drift(self) -> None:
        clock = DeterministicClock()
        ledger = CacheLedger(capacity=16, clock=clock)
        ledger.initialize(zone_version=1, entries=[entry("a.test", "v1"), entry("b.test", "v2")])
        with tempfile.TemporaryDirectory() as tmp:
            store = SnapshotStore(Path(tmp) / "ledger.sqlite3")
            snapshot_id = store.save(ledger, created_at_ms=clock.now_ms(),
                                     hits=[HitEvent(clock.now_ms(), "a.test", TYPE_A, 55)])
            report = store.verify_against(snapshot_id, ledger)
            self.assertTrue(report.matched)
            saved = store.load(snapshot_id)
            self.assertEqual(len(saved["entries"]), 2)
            self.assertEqual(saved["hits"][0]["remaining_ttl_s"], 55)
            ledger.cache().put(entry("c.test", "v3"))
            drifted = store.verify_against(snapshot_id, ledger)
            self.assertFalse(drifted.matched)
            self.assertEqual(drifted.diff["added"], [["c.test", TYPE_A]])
            store.close()

    def test_format_versions_persisted(self) -> None:
        clock = DeterministicClock()
        ledger = CacheLedger(capacity=16, clock=clock)
        ledger.initialize(zone_version=1, entries=[entry("a.test", "v3")])
        with tempfile.TemporaryDirectory() as tmp:
            store = SnapshotStore(Path(tmp) / "s.sqlite3")
            sid = store.save(ledger, created_at_ms=1)
            snapshots = store.list_snapshots()
            self.assertEqual(snapshots[0]["format_versions"], ["v3"])
            self.assertEqual(snapshots[0]["snapshot_id"], sid)
            store.close()


if __name__ == "__main__":
    unittest.main()
