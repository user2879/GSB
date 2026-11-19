import tempfile
import unittest
from pathlib import Path

from dwb_diag.config import DataLayout, DeterministicClock
from dwb_diag.engine import DiagnosticEngine, build_demo_views
from dwb_diag.errors import LostPageError
from dwb_diag.fault_injection import inject_main_corrupt
from dwb_diag.paging import Page
from dwb_diag.recovery import RecoveryEngine
from dwb_diag.storage import CrashPlan, SimulatedCrash
from dwb_diag.util import StructuredLogger
from dwb_diag.buffer import DoubleWriteBuffer
from dwb_diag.versions import VersionMap
from dwb_diag.keyspace import Keyspace
from dwb_diag.paging import MainDataFile


class TestWriteAndCrash(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.eng = DiagnosticEngine(self.root)
        build_demo_views(self.eng.catalog)
        self.eng.catalog.save()
        from dwb_diag.views import ViewManager

        self.eng.views = ViewManager(self.eng.catalog, self.eng.storage.keyspace)

    def tearDown(self):
        self.tmp.cleanup()

    def test_three_phase_markers_in_order(self):
        self.eng.submit({"user:1": 10, "order:2": 20})
        phases = [m.phase for m in self.eng.storage.markers.markers]
        self.assertEqual(phases, ["buffer_write", "main_overwrite", "commit_mark"])
        # 所有标记带序号与单调时钟毫秒
        for marker in self.eng.storage.markers.markers:
            self.assertGreaterEqual(marker.ts_ms, 0)
            self.assertGreaterEqual(marker.seq, 1)

    def test_double_write_order_buffer_first(self):
        self.eng.submit({"user:1": 10})
        dwb = DoubleWriteBuffer(self.eng.layout.buffer_file, self.eng.capacity)
        scan = dwb.validate_strict()
        self.assertGreaterEqual(len(scan.mirrors), 1)
        # 主文件页摘要与缓冲镜像摘要一致
        for mirror in scan.mirrors:
            frame = self.eng.storage.main.read_slot(mirror.page_no)
            self.assertEqual(Page.digest_hex(frame), mirror.digest.hex())

    def test_crash_during_main_overwrite_is_repairable(self):
        self.eng.submit({"user:1": 1})
        try:
            self.eng.submit(
                {"user:2": 2, "user:3": 3, "order:9": 9},
                crash_plan=CrashPlan(commit_id=2, phase="main_overwrite"),
            )
            self.fail("expected SimulatedCrash")
        except SimulatedCrash as exc:
            self.assertEqual(exc.phase, "main_overwrite")
        # 恢复前键空间被闸门禁止
        self.assertFalse(self.eng.storage.keyspace.ready)
        report = self.eng.recover()
        self.assertTrue(self.eng.storage.keyspace.ready)
        self.assertEqual(report.max_seq >= 2, True)
        # 修复后所有页与最新缓冲镜像一致
        scan = self.eng.storage.dwb.scan(strict_digest=False)
        for pno, mirror in scan.latest_per_page.items():
            self.assertEqual(
                Page.digest_hex(self.eng.storage.main.read_slot(pno)),
                mirror.digest.hex(),
            )

    def test_crash_during_buffer_phase(self):
        with self.assertRaises(SimulatedCrash):
            self.eng.submit(
                {"user:1": 1},
                crash_plan=CrashPlan(commit_id=1, phase="buffer_write"),
            )
        report = self.eng.recover()
        # 缓冲阶段崩溃: 没有完整新镜像, 主文件也未被改动 => 无修复
        self.assertEqual(report.repaired, [])

    def test_deterministic_repeated_recovery(self):
        self.eng.submit({"user:1": 1, "order:1": 2})
        try:
            self.eng.submit(
                {"user:2": 3}, crash_plan=CrashPlan(commit_id=2, phase="main_overwrite")
            )
        except SimulatedCrash:
            pass
        r1 = self.eng.recover().to_dict()
        # 重新打开一个全新引擎指向同一目录, 再恢复一次
        eng2 = DiagnosticEngine(self.root)
        r2 = eng2.recover().to_dict()
        self.assertEqual(r1["final_page_digests"], r2["final_page_digests"])
        self.assertEqual(r1["repaired"], r2["repaired"])
        self.assertEqual(r1["max_seq"], r2["max_seq"])
        self.assertIn("per-page comparison", r1["truth_source"])


class TestLostPage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_lost_page_without_zero_fill(self):
        layout = DataLayout(self.root).ensure()
        clock = DeterministicClock()
        logger = StructuredLogger(layout.log_file, clock)
        from dwb_diag.storage import Storage

        storage = Storage(layout, 1024, 8, clock, logger)
        storage.keyspace.apply_change  # 仅确保对象可用
        # 直接构造: 版本映射声明页 2 存在, 但主文件槽位损坏且缓冲无镜像
        frame = Page(1024).encode(b'{"v":1,"entries":{"x":1}}')
        digest = Page.digest_hex(frame)
        storage.versions.upsert(2, 5, digest)
        storage.versions.save()
        storage.main.ensure_file()
        # 主文件写一个被破坏的槽位
        bad = bytearray(frame)
        bad[20] ^= 0xFF
        storage.main.write_slot(2, bytes(bad))
        # 缓冲保持空
        keyspace = Keyspace(storage.main)
        engine = RecoveryEngine(
            storage.main, storage.dwb, storage.versions, keyspace, logger
        )
        with self.assertRaises(LostPageError) as ctx:
            engine.run()
        self.assertEqual(ctx.exception.code, "E_LOST")
        self.assertEqual(ctx.exception.detail["page_no"], 2)
        self.assertEqual(ctx.exception.detail["max_seq"], 0)
        # 不得零填充: 损坏内容保持原样
        self.assertNotEqual(storage.main.read_slot(2), b"\x00" * 1024)


class TestMainCorruptRecovery(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.eng = DiagnosticEngine(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_injected_corruption_gets_repaired(self):
        self.eng.submit({"user:1": 7})
        target = sorted(self.eng.storage.versions.pages)[0]
        inject_main_corrupt(self.eng.layout, target, self.eng.page_size)
        report = self.eng.recover()
        self.assertEqual([e.page_no for e in report.repaired], [target])
        self.assertEqual(report.repaired[0].reason, "digest_mismatch")


if __name__ == "__main__":
    unittest.main()
