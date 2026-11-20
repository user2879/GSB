from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from edgecache.cache import EdgeCache
from edgecache.errors import EdgeError
from edgecache.ledger import Ledger
from edgecache.model import canonical_trace_bytes
from edgecache.spec import load_spec

from tests.helpers import SAMPLE_V1, SAMPLE_V2, load_raw, write_raw


class PropagationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_dependency_order_is_stable_when_content_order_changes(self) -> None:
        original = load_spec(SAMPLE_V1)
        document = load_raw(SAMPLE_V1)
        document["content"].reverse()
        changed_path = write_raw(self.path / "reordered.json", document)
        reordered = load_spec(changed_path)
        self.assertEqual(original.content_version, reordered.content_version)
        plan_a = original.content.propagation_plan(["/data/root"])
        plan_b = reordered.content.propagation_plan(["/data/root"])
        self.assertEqual(canonical_trace_bytes(plan_a), canonical_trace_bytes(plan_b))
        self.assertEqual(
            ["/data/left", "/data/right", "/page/home"],
            [row["affected_path"] for row in plan_b],
        )

    def test_trigger_is_not_part_of_downstream_set(self) -> None:
        spec = load_spec(SAMPLE_V2)
        affected = spec.content.descendant_order("/data/root")
        self.assertNotIn("/data/root", affected)
        self.assertEqual(["/data/left", "/data/right", "/page/home"], affected)


class LedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name)
        self.spec1 = load_spec(SAMPLE_V1)
        self.spec2 = load_spec(SAMPLE_V2)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _entries(self, spec, ledger_version: int) -> list[dict]:
        cache = EdgeCache(
            node_id="edge-a",
            capacity=10,
            eviction_rule="stable_fifo",
            policy=spec.policy,
            ledger_version=ledger_version,
        )
        for record in spec.content.records:
            cache.put(
                ledger_version=ledger_version,
                path=record.path,
                etag=record.etag,
                content_version=spec.content_version,
                ttl_ms=record.ttl_ms,
                content_size=len(record.content),
            )
        return [
            {"node_id": "edge-a", **entry}
            for entry in cache.entry_state(ledger_version=ledger_version)
        ]

    def test_monotonic_versions_and_stale_current_stats(self) -> None:
        ledger = Ledger(self.path / "ledger.sqlite")
        try:
            v1 = ledger.begin_round(self.spec1, round_kind="test")
            self.assertEqual(1, v1)
            ledger.complete_round(
                v1,
                entries=self._entries(self.spec1, v1),
                evictions=[],
                propagation=[],
                progress_cursor=5,
            )
            v2 = ledger.begin_round(self.spec2, round_kind="content_update")
            self.assertEqual(2, v2)
            ledger.complete_round(
                v2,
                entries=self._entries(self.spec2, v2),
                evictions=[],
                propagation=self.spec2.content.propagation_plan(["/data/root"]),
                progress_cursor=8,
            )
            self.assertEqual(v2, ledger.current_version())
            view = ledger.view_model()
            self.assertEqual(v2, view["ledger_version"])
            self.assertTrue(all(not entry["stale"] for entry in view["entries"]))
            self.assertEqual({self.spec2.content_version}, {entry["content_version"] for entry in view["entries"]})
            self.assertEqual([True, False], [item["stale"] for item in view["versions"]])
            self.assertTrue(all(entry["stale"] for entry in ledger.entries_for_version(v1)))
            self.assertEqual(3, view["current_stats"]["propagation_count"])
        finally:
            ledger.close()

    def test_pending_round_becomes_unknown_after_reopen(self) -> None:
        path = self.path / "pending.sqlite"
        ledger = Ledger(path)
        version = ledger.begin_round(self.spec1, round_kind="interrupted")
        ledger.close()
        reopened = Ledger(path)
        try:
            self.assertEqual([version], reopened.recover_pending())
            self.assertIsNone(reopened.current_version())
            self.assertEqual([version], [item["ledger_version"] for item in reopened.unknown_rounds()])
        finally:
            reopened.close()

    def test_mixed_entry_version_is_rejected(self) -> None:
        ledger = Ledger(self.path / "mixed.sqlite")
        try:
            version = ledger.begin_round(self.spec1, round_kind="test")
            entries = self._entries(self.spec1, version)
            entries[0]["content_version"] = "cv-wrong"
            with self.assertRaises(EdgeError) as context:
                ledger.complete_round(
                    version,
                    entries=entries,
                    evictions=[],
                    propagation=[],
                    progress_cursor=1,
                )
            self.assertEqual("E_STALE", context.exception.code)
            self.assertEqual("pending", ledger.version_row(version)["status"])
        finally:
            ledger.close()


if __name__ == "__main__":
    unittest.main()
