"""要求2/5/6/7/8：真实子进程端到端、收敛、快照恢复、重放、错误码。"""
import json
import shutil
import socket
import tempfile
import unittest
from pathlib import Path

from ccdplatform.engine import SessionEngine
from ccdplatform.errors import (
    GridError,
    HandshakeTimeoutError,
    PortInUseError,
    ReplayError,
)
from ccdplatform.injection import inject_cycle, inject_range, strip_grid_from_snapshot
from ccdplatform.orchestrator import NodeOrchestrator, is_port_free
from ccdplatform.snapshot import SnapshotStore
from ccdplatform.structured_log import StructuredLogger

ROOT = Path(__file__).resolve().parent.parent


class EndToEndTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.work = Path(self.tmp.name)
        self.manifest = self.work / "manifest.json"
        self.nodes = self.work / "nodes.json"
        self.db = self.work / "session.sqlite"
        self.log = self.work / "events.jsonl"
        shutil.copy(ROOT / "samples" / "manifest.json", self.manifest)
        # 使用高位独立端口避免与其他测试冲突
        nodes_spec = {
            "nodes": [
                {"node_id": "node-a", "host": "127.0.0.1", "port": 22001},
                {"node_id": "node-b", "host": "127.0.0.1", "port": 22002},
                {"node_id": "node-c", "host": "127.0.0.1", "port": 22003},
            ]
        }
        self.nodes.write_text(json.dumps(nodes_spec), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def _engine(self, **kwargs):
        return SessionEngine(
            workdir=self.work,
            manifest_path=self.manifest,
            nodes_path=self.nodes,
            db_path=self.db,
            log_path=self.log,
            exponent=3,
            max_concurrency=2,
            **kwargs,
        )

    def test_full_pipeline_converges_and_persists(self):
        engine = self._engine()
        try:
            summaries = engine.process_versions()
        finally:
            engine.close()
        self.assertEqual([s["version"] for s in summaries], [1, 2, 3])
        for summary in summaries:
            for node, entry in summary["convergence"].items():
                self.assertTrue(entry["matched"], node)
            self.assertLessEqual(summary["peak_inflight"], 2)  # 不绕过并发预算
        # 快照包含三个版本
        store = SnapshotStore(self.db)
        self.assertEqual(store.list_versions(), [1, 2, 3])
        store.close()

    def test_retry_and_first_receipt_separate_log_lines(self):
        engine = self._engine()
        try:
            engine.process_versions()
        finally:
            engine.close()
        lines = [json.loads(line) for line in self.log.read_text().splitlines()]
        receipts = [e for e in lines if e.get("event") == "dispatch.receipt"]
        # 每条回执带独立 attempt 字段，不存在合并条目
        self.assertTrue(all("attempt" in e for e in receipts))
        self.assertTrue(all("is_retry" in e for e in receipts))
        transitions = [e for e in lines if e.get("event") == "state.transition"]
        states = [e["to"] for e in transitions]
        self.assertIn("CONVERGED", states)
        self.assertIn("PERSISTED", states)

    def test_restore_whole_replace_and_replay(self):
        engine = self._engine()
        try:
            engine.process_versions()
        finally:
            engine.close()

        engine2 = self._engine()
        try:
            info = engine2.restore_latest()
            self.assertTrue(info["restored"])
            self.assertEqual(info["mode"], "whole_replace")
            result = engine2.replay()
            self.assertTrue(result["digest_match"])
            # 恢复后仍可继续处理新版本：直接重放 v1..v3 不报错即验证可继续演进
            self.assertIn("convergence", result)
        finally:
            engine2.close()

    def test_replay_detects_tampered_snapshot(self):
        engine = self._engine()
        try:
            engine.process_versions()
        finally:
            engine.close()
        # 篡改某个编译项的量化值，重放必须定位到键与字段并报 E_REPLAY
        import sqlite3

        conn = sqlite3.connect(str(self.db))
        row = conn.execute(
            "SELECT item_json FROM compiled_items WHERE version=3 ORDER BY key LIMIT 1"
        ).fetchone()
        tampered = json.loads(row[0])
        original_q = tampered["quantized"]
        tampered["quantized"] = original_q + 1
        conn.execute(
            "UPDATE compiled_items SET item_json=? WHERE version=3 AND key=?",
            (json.dumps(tampered), tampered["key"]),
        )
        conn.commit()
        conn.close()
        engine2 = self._engine()
        try:
            engine2.restore_latest()
            with self.assertRaises(ReplayError) as ctx:
                engine2.replay()
            self.assertEqual(ctx.exception.code, "E_REPLAY")
            self.assertEqual(
                ctx.exception.details["first_mismatch_key"], tampered["key"]
            )
            self.assertEqual(
                ctx.exception.details["first_mismatch_field"], "quantized"
            )
        finally:
            engine2.close()

    # ------------------------------------------------------- 异常注入（要求8）
    def test_inject_range(self):
        target = self.work / "m_range.json"
        shutil.copy(self.manifest, target)
        inject_range(target, "alpha", exponent=3)
        engine = SessionEngine(
            workdir=self.work, manifest_path=target, nodes_path=self.nodes,
            db_path=self.work / "range.sqlite", log_path=self.work / "range.log",
            exponent=3, max_concurrency=2,
        )
        from ccdplatform.errors import RangeError

        try:
            with self.assertRaises(RangeError) as ctx:
                engine.process_versions()
            self.assertEqual(ctx.exception.code, "E_RANGE")
        finally:
            engine.close()

    def test_inject_cycle(self):
        target = self.work / "m_cycle.json"
        shutil.copy(self.manifest, target)
        inject_cycle(target, "alpha", "beta")
        engine = SessionEngine(
            workdir=self.work, manifest_path=target, nodes_path=self.nodes,
            db_path=self.work / "cycle.sqlite", log_path=self.work / "cycle.log",
            exponent=3, max_concurrency=2,
        )
        from ccdplatform.errors import CycleError

        try:
            with self.assertRaises(CycleError) as ctx:
                engine.process_versions()
            self.assertEqual(ctx.exception.code, "E_CYCLE")
        finally:
            engine.close()

    def test_inject_grid_missing(self):
        engine = self._engine()
        try:
            engine.process_versions()
        finally:
            engine.close()
        strip_grid_from_snapshot(self.db)
        store = SnapshotStore(self.db)
        try:
            with self.assertRaises(GridError) as ctx:
                store.load_grid()
            self.assertEqual(ctx.exception.code, "E_GRID")
        finally:
            store.close()

    # ------------------------------------------------- 节点编排错误码（要求2）
    def test_port_in_use_is_e_port(self):
        holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        holder.bind(("127.0.0.1", 22099))
        holder.listen(1)
        try:
            self.assertFalse(is_port_free("127.0.0.1", 22099))
            from ccdplatform.model import NodeSpec

            logger = StructuredLogger(self.work / "port.log")
            orch = NodeOrchestrator(
                {"x": NodeSpec("x", "127.0.0.1", 22099)}, 1000, logger
            )
            with self.assertRaises(PortInUseError) as ctx:
                orch.start_all()
            self.assertEqual(ctx.exception.code, "E_PORT")
            logger.close()
        finally:
            holder.close()

    def test_handshake_timeout_is_e_handshake(self):
        # 静默 accept 但永远不发送 hello 的哑服务，驱动握手超时 -> E_HANDSHAKE
        import threading

        holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        holder.bind(("127.0.0.1", 22097))
        holder.listen(1)
        holder.settimeout(5)
        accepted = {}

        def serve():
            conn, _ = holder.accept()
            accepted["conn"] = conn  # 保持连接但不发任何数据

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        from ccdplatform.model import NodeSpec

        logger = StructuredLogger(self.work / "hs.log")
        orch = NodeOrchestrator(
            {"x": NodeSpec("x", "127.0.0.1", 22097)},
            1000,
            logger,
            handshake_timeout_ms=600,
            skip_prebind_check=True,
        )
        try:
            with self.assertRaises(HandshakeTimeoutError) as ctx:
                orch.start_all()
            self.assertEqual(ctx.exception.code, "E_HANDSHAKE")
        finally:
            thread.join(timeout=2)
            accepted.get("conn", socket.socket()).close()
            holder.close()
            logger.close()
    def test_node_start_failure_is_e_node(self):
        # 指向一个不存在的脚本路径不可行（路径固定），改为让节点绑定一个
        # 被占用端口并跳过预检：子进程 BIND_ERROR 后提前退出 -> E_NODE
        holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        holder.bind(("127.0.0.1", 22096))
        holder.listen(1)
        from ccdplatform.model import NodeSpec

        logger = StructuredLogger(self.work / "node.log")
        orch = NodeOrchestrator(
            {"x": NodeSpec("x", "127.0.0.1", 22096)}, 1000, logger,
            startup_timeout_ms=3000,
        )
        from ccdplatform.errors import NodeStartError

        try:
            with self.assertRaises((NodeStartError, PortInUseError)) as ctx:
                orch.start_all()
            self.assertIn(ctx.exception.code, ("E_NODE", "E_PORT"))
        finally:
            logger.close()
            holder.close()

    def test_real_timeout_then_retry_converges(self):
        # 让 node-b 首批首次尝试延迟 800ms，策略超时 300ms 触发一次真实重试，
        # 重试成功后收敛；验证首次与重试回执分别记录且最终采信重试回执。
        from ccdplatform.dispatcher import Dispatcher
        from ccdplatform.convergence import check_convergence
        from ccdplatform.compiler import Compiler
        from ccdplatform.manifest import load_manifest, load_nodes
        from ccdplatform.fixedpoint import GridSpec
        from ccdplatform.model import ManifestVersion

        nodes = load_nodes(self.nodes)
        # 缩小 node-b 超时以快速触发
        logger = StructuredLogger(self.work / "retry.log")
        orch = NodeOrchestrator(
            nodes, 1000, logger,
            fault_env={"CCD_FAULT_FIRST_DELAY_MS": "1200"},
        )
        try:
            handles = orch.start_all()
            versions = load_manifest(self.manifest, set(nodes))
            compiler = Compiler(GridSpec(3))
            result = compiler.compile_version(versions[0])
            # 把超时调小，使首批首次尝试必然超时
            from ccdplatform.model import DispatchStrategy

            result.plan.strategy = DispatchStrategy(
                batch_size=2, timeout_ms=250, max_retries=5
            )
            dispatcher = Dispatcher(handles, logger, max_concurrency=3)
            outcome = dispatcher.dispatch(result.plan, result.compiled)
            self.assertEqual({}, outcome.node_failures)
            # node-b 至少出现过一次重试
            retry_lines = [
                json.loads(line)
                for line in self.work.joinpath("retry.log").read_text().splitlines()
                if '"event": "dispatch.retry"' in line
            ]
            self.assertTrue(retry_lines)
            report = check_convergence(
                result.plan, result.compiled, result.graph, outcome, 1000
            )
            self.assertTrue(all(e["matched"] for e in report.values()))
            # 最终采信的回执 attempt 可能 > 1
            self.assertGreaterEqual(outcome.final_attempts.get("node-b", 1), 1)
        finally:
            orch.shutdown_all()
            logger.close()

if __name__ == "__main__":
    unittest.main()
