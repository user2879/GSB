import json
import os
import shutil
import tempfile
import unittest

from repairbench.errors import (
    E_INTEGRITY,
    E_OVERSUB,
    E_REPLAY,
    E_SNAPSHOT,
    WorkbenchError,
)
from repairbench.replay import replay_group
from repairbench.workbench import Session

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "samples", "groups.json")
EVOLVE = os.path.join(os.path.dirname(__file__), "..", "samples", "evolve_alpha.json")


class SessionCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.session = Session(self.dir)
        self.session.load(SAMPLE)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)


class TestRunAndReplay(SessionCase):
    def test_run_all_and_replay_consistent(self):
        result = self.session.run_all()
        self.assertIn("alpha", result)
        report = replay_group(self.session, "alpha")
        self.assertTrue(report["consistent"])
        self.assertGreater(report["batches_compared"], 0)

    def test_error_report_three_segments(self):
        result = self.session.run_all()
        glob = result["alpha"]["error_report"]["global"]
        for key in ("quantization_trunc_quanta", "batch_split_rounding_quanta",
                    "queue_delay_conv_quanta", "dominant"):
            self.assertIn(key, glob)

    def test_verdict_classes_in_ranking(self):
        result = self.session.run_all()
        classes = result["alpha"]["ranking"]["classes"]
        self.assertGreater(classes["ERROR_INDISTINGUISHABLE"], 0)
        self.assertGreater(classes["EXACT_SEPARABLE"], 0)
        self.assertGreater(classes["INSUFFICIENT_EVIDENCE"], 0)


class TestCrashRecovery(SessionCase):
    def test_crash_then_recover_and_replay(self):
        self.session.run_all(crash_after_commits=2)
        # 强制终止后：账本中应存在未提交的 pending 批次
        pending = self.session.ledger.pending_batches()
        self.assertGreater(len(pending), 0)
        committed_before = len(self.session.ledger.committed_batches())

        # 重启会话恢复：仅依据已提交记录，pending 标记 UNKNOWN 并回收配额
        session2 = Session(self.dir)
        info = session2.load(SAMPLE, recover=True)
        self.assertGreater(len(info["unknown_batches"]), 0)
        states = {r["state"] for r in session2.ledger.committed_batches()}
        self.assertIn("UNKNOWN", states)
        self.assertIn("COMMITTED", states)
        self.assertEqual(len(session2.ledger.pending_batches()), 0)
        self.assertGreaterEqual(
            len(session2.ledger.committed_batches()), committed_before
        )
        # 恢复后重放一致
        report = replay_group(session2, "alpha")
        self.assertTrue(report["consistent"])
        self.assertGreater(len(report["rerun_batches"]), 0)


class TestEvolution(SessionCase):
    def test_evolve_replaces_view_and_marks_stale(self):
        self.session.run_all()
        with open(EVOLVE, encoding="utf-8") as fh:
            new_spec = json.load(fh)
        result = self.session.evolve("alpha", new_spec)
        self.assertEqual(result["version"], 2)
        impact = result["impact"]
        self.assertEqual(impact["nodes"]["added"], ["a-n4"])
        self.assertEqual(impact["nodes"]["removed"], ["a-n2"])
        self.assertEqual(impact["nodes"]["changed"], ["a-n1", "a-n3"])
        self.assertIn("concurrency_limit", impact["param_fields_changed"])
        self.assertIn("global_budget_bytes", impact["param_fields_changed"])
        # 被删除节点 a-n2 上的块被延误
        self.assertIn("a-b3", impact["delayed_blocks"])
        self.assertIn("a-b4", impact["delayed_blocks"])
        # 旧版本结论整体标记陈旧，当前视图整体替换
        ctx = self.session.groups["alpha"]
        self.assertEqual(len(ctx.stale.entries), 1)
        self.assertTrue(ctx.stale.entries[0]["stale"])
        self.assertIsNone(ctx.last_run)
        view = self.session.current_view("alpha")
        self.assertEqual(view["version"], 2)
        node_ids = {n["node_id"] for n in view["spec"]["nodes"]}
        self.assertEqual(node_ids, {"a-n1", "a-n3", "a-n4"})

    def test_evolve_then_run_and_replay(self):
        self.session.run_all()
        with open(EVOLVE, encoding="utf-8") as fh:
            new_spec = json.load(fh)
        self.session.evolve("alpha", new_spec)
        self.session.run_group("alpha")
        report = replay_group(self.session, "alpha")
        self.assertTrue(report["consistent"])


class TestFaultInjection(SessionCase):
    def test_inject_corrupt_block_integrity(self):
        self.session.inject_corrupt_block("a-b1")
        with self.assertRaises(WorkbenchError) as cm:
            self.session.run_all()
        self.assertEqual(cm.exception.code, E_INTEGRITY)
        codes = [e["code"] for e in self.session.log.read_all()]
        self.assertIn(E_INTEGRITY, codes)

    def test_inject_oversub(self):
        with self.assertRaises(WorkbenchError) as cm:
            self.session.inject_oversubscription()
        self.assertEqual(cm.exception.code, E_OVERSUB)
        codes = [e["code"] for e in self.session.log.read_all()]
        self.assertIn(E_OVERSUB, codes)

    def test_inject_snapshot_mismatch(self):
        self.session.run_all()
        self.session.inject_snapshot_mismatch()
        session2 = Session(self.dir)
        info = session2.load(SAMPLE)
        self.assertEqual(info["status"], "replaced")  # 整体替换而非增量合并
        codes = [e["code"] for e in session2.log.read_all()]
        self.assertIn(E_SNAPSHOT, codes)

    def test_replay_detects_ledger_tamper(self):
        self.session.run_all()
        # 篡改账本中一条已提交批次的带宽占用
        self.session.ledger._conn.execute(
            "UPDATE batches SET bandwidth_kbps = bandwidth_kbps + 1 "
            "WHERE batch_seq = (SELECT MIN(batch_seq) FROM batches)"
        )
        self.session.ledger._conn.commit()
        with self.assertRaises(WorkbenchError) as cm:
            replay_group(self.session, "alpha")
        self.assertEqual(cm.exception.code, E_REPLAY)
        self.assertEqual(cm.exception.context["field"], "bandwidth_kbps")
        self.assertIn("batch_id", cm.exception.context)
        self.assertIn("block_id", cm.exception.context)


if __name__ == "__main__":
    unittest.main()
