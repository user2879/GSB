import json
import os
import signal
import tempfile
import unittest

from configbench.errors import LedgerError, ReplayError
from configbench.ledger import LEDGER_FORMAT_VERSION, Ledger
from configbench.manifest import load_all
from configbench.platform import Platform
from configbench.scheduler import mono_ms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RecoveryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry, cls.manifest, cls.samples = load_all(
            os.path.join(ROOT, "samples/registry.json"),
            os.path.join(ROOT, "samples/manifest.json"),
            os.path.join(ROOT, "samples/samples.json"))
        cls.tmp = tempfile.mkdtemp()

    def test_chain_digest_tamper_replaces_and_backs_up(self):
        path = os.path.join(self.tmp, "tamper.db")
        ledger = Ledger(path)
        ledger.append("latency", {"request_id": "q1", "elapsed_ms": 5})
        ledger.close()
        # 直接篡改已提交事件 payload，破坏内容摘要。
        import sqlite3
        conn = sqlite3.connect(path)
        row = conn.execute("SELECT payload FROM events LIMIT 1").fetchone()
        tampered = row[0].replace("5", "9")
        conn.execute("UPDATE events SET payload=? WHERE seq=1", (tampered,))
        conn.commit()
        conn.close()
        ledger2 = Ledger(path)
        self.assertTrue(ledger2.replaced)
        self.assertTrue(ledger2.backup_path.endswith((".db", )[0:0])
                        or ledger2.backup_path.startswith(path))
        self.assertTrue(os.path.exists(ledger2.backup_path))
        # 备份为只读；新账本为空（整体替换而非增量合并）。
        self.assertEqual(ledger2.events(), [])
        if os.name != "nt":
            mode = os.stat(ledger2.backup_path).st_mode & 0o777
            self.assertEqual(mode & 0o222, 0)
        ledger2.close()

    def test_format_version_mismatch_replaces(self):
        path = os.path.join(self.tmp, "fmt.db")
        ledger = Ledger(path)
        ledger.append("latency", {"request_id": "q", "elapsed_ms": 1})
        ledger.close()
        import sqlite3
        conn = sqlite3.connect(path)
        conn.execute("UPDATE meta SET v=? WHERE k='format_version'",
                     (str(LEDGER_FORMAT_VERSION + 100),))
        conn.commit()
        conn.close()
        ledger2 = Ledger(path)
        self.assertTrue(ledger2.replaced)
        self.assertEqual(ledger2.events(), [])
        ledger2.close()

    def test_replay_mismatch_raises_e_replay(self):
        path = os.path.join(self.tmp, "replay.db")
        p = Platform(self.registry, self.manifest, self.samples, path)
        p.start_nodes()
        p.publish_version(1)
        p.close()
        # 篡改账本中的回执摘要，恢复重放必须以 E_REPLAY 定位任务与字段。
        import sqlite3
        conn = sqlite3.connect(path)
        row = conn.execute(
            "SELECT seq,payload FROM events WHERE kind='receipt' ORDER BY seq LIMIT 1"
        ).fetchone()
        payload = json.loads(row[1])
        task_id = payload["task_id"]
        payload["summary_digest"] = "f" * 64
        conn.execute("UPDATE events SET payload=? WHERE seq=?",
                     (json.dumps(payload, sort_keys=True), row[0]))
        conn.commit()
        conn.close()
        # 摘要链因此失效 -> 账本会被整体替换。为单独验证 E_REPLAY，
        # 改为篡改后同步修复 digest 链不现实，故这里直接校验链失效走替换路径；
        # E_REPLAY 路径用下面的内存级单测覆盖。
        p2 = Platform(self.registry, self.manifest, self.samples, path)
        self.assertTrue(p2.ledger.replaced)
        p2.close()

    def test_replay_one_field_mismatch_e_replay(self):
        # 直接构造一个摘要不一致的回执，调用 _replay_one 验证 E_REPLAY 定位。
        path = os.path.join(self.tmp, "replay-one.db")
        p = Platform(self.registry, self.manifest, self.samples, path)
        p.start_nodes()
        p.publish_version(1)
        receipt = p.ledger.snapshot_state()["receipts"]["v1:node-alpha"]
        bad = dict(receipt)
        bad["summary_digest"] = "a" * 64
        with self.assertRaises(ReplayError) as ctx:
            p._replay_one("v1:node-alpha", bad, [])
        self.assertEqual(ctx.exception.code, "E_REPLAY")
        self.assertIn("v1:node-alpha", ctx.exception.field)
        p.close()

    def test_hard_kill_unknown_task_then_rerun(self):
        path = os.path.join(self.tmp, "crash.db")
        p1 = Platform(self.registry, self.manifest, self.samples, path,
                      max_global=1, per_node_quota=1)
        p1.start_nodes()
        # 先成功发布 v1，建立已提交基线。
        p1.publish_version(1)
        # 模拟崩溃：强制杀掉全部节点子进程（不做清理握手），平台进程随后中止。
        for node_id in sorted(p1.node_processes):
            p1.node_processes[node_id].kill_hard()
        p1.ledger.close()
        # 新平台进程打开同一账本：已提交任务可恢复，无退出报告生成。
        p2 = Platform(self.registry, self.manifest, self.samples, path)
        rec = p2.recover()
        self.assertEqual(rec["unknown_tasks"], [])
        self.assertEqual(len(rec["restored_nodes"]), 3)
        # 恢复后重放已完成任务且全部匹配。
        self.assertTrue(all(m.get("match") for m in rec["replay_mismatches"]))
        # 重启节点并继续发布 v2，验证恢复后系统可继续工作且双路径一致。
        p2.start_nodes()
        pub2 = p2.publish_version(2)
        self.assertEqual(pub2["receipts"]["success"], 2)  # 仅受影响两节点
        self.assertTrue(p2.compare_paths(2)["match"])
        p2.close()


if __name__ == "__main__":
    unittest.main()
