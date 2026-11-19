import shutil
import tempfile
import unittest
from pathlib import Path

from dwb_diag.config import DataLayout
from dwb_diag.engine import DiagnosticEngine, build_demo_views
from dwb_diag.errors import (
    PageFormatError,
    ReplayError,
    SnapshotMismatchError,
)
from dwb_diag.fault_injection import (
    clone_layout,
    inject_buffer_truncate,
    inject_replay_tamper,
    inject_snapshot_stale,
    replay_compare,
)
from dwb_diag.recovery import RecoveryEngine
from dwb_diag.storage import CrashPlan, SimulatedCrash, Storage
from dwb_diag.views import ViewCatalog, ViewManager


def build_session(root: Path) -> DiagnosticEngine:
    eng = DiagnosticEngine(root)
    build_demo_views(eng.catalog)
    eng.catalog.save()
    eng.views = ViewManager(eng.catalog, eng.storage.keyspace)
    eng.submit({"user:1": 1, "order:1": 100})
    try:
        eng.submit(
            {"user:2": 2, "user:3": 3, "order:7": 7, "order:8": 8},
            crash_plan=CrashPlan(commit_id=2, phase="main_overwrite"),
        )
    except SimulatedCrash:
        pass
    report = eng.recover()
    eng.save_snapshot("s1", report)
    return eng


class TestSnapshot(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.eng = build_session(Path(self.tmp.name) / "sess")

    def tearDown(self):
        self.tmp.cleanup()

    def test_snapshot_match_then_stale_full_replace(self):
        result = self.eng.open_snapshot("s1")
        self.assertEqual(result["status"], "match")

        inject_snapshot_stale(self.eng.layout)
        with self.assertRaises(SnapshotMismatchError) as ctx:
            self.eng.open_snapshot("s1")
        self.assertEqual(ctx.exception.code, "E_SNAPSHOT")
        detail = ctx.exception.detail
        self.assertTrue(Path(detail["backup"]).exists())
        # 备份为应用层只读副本, 且结构化日志记录了备份动作
        self.assertTrue(
            any(e["event"] == "snapshot_backup_readonly" for e in self.eng.logger.events)
        )
        self.assertIn("changed_pages", detail)

    def test_replay_ok_and_tamper_e_replay(self):
        snap = self.eng.snapshots.latest()
        scratch = Path(self.tmp.name) / "replay-ok"
        out = self.eng.replay(snap, scratch)
        self.assertEqual(out["status"], "replay_ok")

        target = sorted(self.eng.storage.versions.pages)[0]
        # 失效轨迹被篡改: 重放推导与快照轨迹不一致 -> E_REPLAY(invalidation)
        bad_invalidation = [
            dict(item, sequence=["tampered_view"]) for item in snap.invalidation
        ] or [{"page_no": target, "sequence": ["tampered_view"]}]
        with self.assertRaises(ReplayError) as ctx:
            replay_compare(
                self.eng.storage.main.iter_digests(),
                self.eng._snapshot_main_digests(snap),
                bad_invalidation,
                snap.invalidation,
            )
        self.assertEqual(ctx.exception.code, "E_REPLAY")
        self.assertEqual(ctx.exception.detail["field"], "invalidation")

    def test_replay_digest_mismatch_locates_first_page(self):
        snap = self.eng.snapshots.latest()
        target = sorted(self.eng.storage.versions.pages)[0]
        wrong = dict(self.eng._snapshot_main_digests(snap))
        wrong[str(target)] = "f" * 64
        with self.assertRaises(ReplayError) as ctx:
            replay_compare(
                self.eng.storage.main.iter_digests(),
                wrong,
                snap.invalidation,
                snap.invalidation,
            )
        self.assertEqual(ctx.exception.code, "E_REPLAY")
        self.assertEqual(ctx.exception.detail["field"], "digest")
        self.assertEqual(ctx.exception.detail["page_no"], target)


class TestInjectionCodes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.eng = build_session(Path(self.tmp.name) / "sess")

    def tearDown(self):
        self.tmp.cleanup()

    def test_buffer_truncate_e_page(self):
        inject_root = Path(self.tmp.name) / "trunc"
        shutil.copytree(self.eng.layout.root, inject_root)
        layout = DataLayout(inject_root)
        inject_buffer_truncate(layout)
        from dwb_diag.buffer import DoubleWriteBuffer

        dwb = DoubleWriteBuffer(layout.buffer_file, self.eng.capacity)
        with self.assertRaises(PageFormatError) as ctx:
            dwb.validate_strict()
        self.assertEqual(ctx.exception.code, "E_PAGE")
        self.assertIsNotNone(ctx.exception.detail["offset"])

    def test_structured_log_contains_codes(self):
        # 制造一次 E_SNAPSHOT 并确认结构化日志记录
        inject_snapshot_stale(self.eng.layout)
        try:
            self.eng.open_snapshot("s1")
        except SnapshotMismatchError:
            pass
        self.assertIn("E_SNAPSHOT", self.eng.logger.codes())


if __name__ == "__main__":
    unittest.main()
