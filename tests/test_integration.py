import json
import os
import tempfile
import unittest

from configbench.errors import (FieldError, HandshakeError, LatencyError,
                                PortError, PreviewError)
from configbench.manifest import load_all
from configbench.platform import Platform

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class IntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry, cls.manifest, cls.samples = load_all(
            os.path.join(ROOT, "samples/registry.json"),
            os.path.join(ROOT, "samples/manifest.json"),
            os.path.join(ROOT, "samples/samples.json"))
        cls.tmp = tempfile.mkdtemp()

    def _platform(self, ledger_name, max_global=4, per_node_quota=2):
        p = Platform(self.registry, self.manifest, self.samples,
                     os.path.join(self.tmp, ledger_name),
                     max_global=max_global, per_node_quota=per_node_quota)
        p.start_nodes()
        self.addCleanup(p.close)
        return p

    def test_full_publish_two_paths_consistent(self):
        p = self._platform("main.db")
        pub1 = p.publish_version(1)
        self.assertEqual(pub1["receipts"]["success"], 3)
        self.assertEqual(pub1["receipts"]["total"], 3)
        # 再次查询缓存与权威重算逐字节一致。
        preview = p.query_preview()
        self.assertEqual(len(preview["nodes"]), 3)
        recomp = p.authoritative_recompute(1)
        self.assertEqual(len(recomp["authoritative"]), 3)
        cmp1 = p.compare_paths(1)
        self.assertTrue(cmp1["match"])
        # 发布 v2：只有受影响节点产生新任务，未变产物字节不变。
        pub2 = p.publish_version(2)
        affected = set(pub2["propagation"]["affected_nodes"])
        self.assertEqual(affected, {"node-alpha", "node-beta"})
        cmp2 = p.compare_paths(2)
        self.assertTrue(cmp2["match"])
        self.assertEqual(pub2["receipts"]["success"], 2)
        # 未受影响节点保留 v1 版本，不产生新任务。
        self.assertEqual(p.cache["node-gamma"].config_version, 1)
        # 同一切片重复聚合摘要稳定（逐字节），只比对 v2 受影响节点。
        again = p.authoritative_recompute(2)
        again_digests = sorted(a["summary_digest"] for a in again["authoritative"].values())
        cached_v2 = sorted(
            p.cache[a["node_id"]].summary_digest
            for a in again["authoritative"].values())
        self.assertEqual(cached_v2, again_digests)

    def test_attempt_outcomes_and_concurrency_limits(self):
        p = self._platform("sched.db", max_global=2, per_node_quota=1)
        pub = p.publish_version(1)
        for result in pub["results"]:
            self.assertTrue(result.attempts)
            self.assertEqual(result.attempts[0]["attempt_no"], 1)
            self.assertIn(result.outcome,
                          ("SUCCESS", "TIMEOUT", "CONN_REFUSED", "FIELD", "EXHAUSTED"))
            self.assertGreaterEqual(result.wait_queued_ms, 0)
        self.assertLessEqual(pub["peaks"]["peak_global"], 2)
        self.assertTrue(all(q <= 1 for q in pub["peaks"]["peak_node"].values()))

    def test_latency_budget_violation_recorded_not_dropped(self):
        p = self._platform("lat.db")
        p.publish_version(1)
        # 人为构造超预算：把预算调为不可能达到的 -1 不可能，改为 0 预算对象。
        import dataclasses
        p.manifest = dataclasses.replace(p.manifest, latency_budget_ms=0)
        import time as _time
        # 注入确定性开销，确保实测严格为正，验证“超预算即 E_LATENCY”。
        p.pre_query_hook = lambda: _time.sleep(0.01)
        with self.assertRaises(LatencyError) as ctx:
            p.query_preview("q-over-1")
        self.assertEqual(ctx.exception.code, "E_LATENCY")
        report = p.latency_report()
        # 超预算请求仍在账本中（计数包含它），且有独立违规事件。
        self.assertTrue(any(r["request_id"] == "q-over-1"
                            for _, _, r in p.ledger.events("latency")))
        self.assertEqual(report["violations"][0]["request_id"], "q-over-1")
        self.assertEqual(report["overall"]["scale"], 1000)

    def test_preview_mismatch_detected(self):
        p = self._platform("mismatch.db")
        p.publish_version(1)
        # 篡改快速缓存（模拟脏缓存），compare_paths 必须定位首个不一致。
        p.cache["node-alpha"].summary_digest = "deadbeef"
        with self.assertRaises(PreviewError) as ctx:
            p.compare_paths(1)
        self.assertEqual(ctx.exception.code, "E_PREVIEW")
        self.assertEqual(ctx.exception.extra["node_id"], "node-alpha")

    def test_ledger_persistence_across_reopen(self):
        path = os.path.join(self.tmp, "persist.db")
        p1 = Platform(self.registry, self.manifest, self.samples, path)
        p1.start_nodes()
        p1.publish_version(1)
        p1.close()
        # 新进程视角：同一账本恢复，缓存由账本重建。
        p2 = Platform(self.registry, self.manifest, self.samples, path)
        rec = p2.recover()
        self.assertEqual(rec["unknown_tasks"], [])
        self.assertEqual(sorted(rec["restored_nodes"]),
                         sorted(self.registry.nodes))
        self.assertTrue(all(m["match"] for m in rec["replay_mismatches"]))
        cmp = p2.compare_paths(1)
        self.assertTrue(cmp["match"])
        p2.close()


if __name__ == "__main__":
    unittest.main()
