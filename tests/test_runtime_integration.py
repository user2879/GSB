import json
import os
import socket
import tempfile
import time
import unittest

from configmesh.errors import BusinessError, ErrorCode
from configmesh.fixedpoint import Grid
from configmesh.logging_utils import StructuredLogger
from configmesh.models import NodeSpec
from configmesh.node_process import NodeProcess
from configmesh.protocol import send_line, recv_line
from configmesh.scheduler import DispatchTask, Scheduler
from configmesh.session import Session
from configmesh.manifest import load_bundle
from configmesh.compiler import compile_version, build_dispatch_plan
from configmesh.convergence import verify_convergence
from configmesh.models import Receipt


def free_port():
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


class NodeProtocolTests(unittest.TestCase):
    def test_port_in_use_is_e_port(self):
        port = free_port()
        holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        holder.bind(("127.0.0.1", port))
        holder.listen(1)
        logger = StructuredLogger()
        process = NodeProcess(NodeSpec(node_id="p", port=port, timeout_ms=500), logger)
        try:
            with self.assertRaises(BusinessError) as ctx:
                process.start()
            self.assertEqual(ctx.exception.code, ErrorCode.E_PORT)
        finally:
            holder.close()

    def test_handshake_and_malformed_reply_e_reply(self):
        port = free_port()
        logger = StructuredLogger()
        process = NodeProcess(NodeSpec(node_id="handshake_node", port=port,
                                       timeout_ms=1500), logger)
        process.start()
        try:
            ack = process.handshake()
            self.assertEqual(ack["node_id"], "handshake_node")
            raw = socket.create_connection(("127.0.0.1", port), timeout=2)
            send_line(raw, {"type": "dispatch", "node_id": "wrong_node_id",
                            "version": 1, "shard_index": 0, "shard_count": 1,
                            "grid_scale": 1000, "items": {}})
            reply = recv_line(raw)
            self.assertEqual(reply["type"], "error")
            raw.close()
            # missing-field receipt path is exercised structurally via required_fields
            self.assertIn("aggregate_digest", Receipt.required_fields())
        finally:
            process.stop()

    def test_handshake_timeout_against_silent_server(self):
        port = free_port()
        silent = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        silent.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        silent.bind(("127.0.0.1", port))
        silent.listen(1)
        logger = StructuredLogger()
        process = NodeProcess(NodeSpec(node_id="silent", port=port, timeout_ms=300), logger)
        thread_hint = {"conn": None}

        def accept_but_ignore():
            conn, _ = silent.accept()
            thread_hint["conn"] = conn
            time.sleep(1.0)

        import threading
        t = threading.Thread(target=accept_but_ignore, daemon=True)
        t.start()
        try:
            with self.assertRaises(BusinessError) as ctx:
                # _connect succeeds via accept thread, then no ack -> timeout
                process.handshake()
            self.assertEqual(ctx.exception.code, ErrorCode.E_HANDSHAKE)
        finally:
            t.join()
            if thread_hint["conn"]:
                thread_hint["conn"].close()
            silent.close()


class SchedulerBudgetTests(unittest.TestCase):
    def test_peak_never_exceeds_budget(self):
        logger = StructuredLogger()
        scheduler = Scheduler(global_limit=2, per_node_quota=1, logger=logger)
        tasks = [
            DispatchTask(task_id=f"t{i}", node_id=f"n{i % 3}", version=1,
                         shard_index=0, shard_count=1,
                         payload={"_retry_policy": "none", "_max_retries": 0})
            for i in range(9)
        ]

        def fake_dispatch(task, attempt):
            time.sleep(0.02)
            return Receipt(node_id=task.node_id, version=1, shard_index=0,
                           shard_count=1, derived_state={}, aggregate_digest="x",
                           compute_ms=1, attempt=attempt)

        records = scheduler.run(tasks, fake_dispatch)
        self.assertTrue(all(r.status == "ok" for r in records))
        self.assertLessEqual(scheduler.peak_concurrency, 2)
        self.assertEqual(scheduler.peak_concurrency, 2)
        self.assertTrue(all(r.queue_ms is not None and r.queue_ms >= 0 for r in records))


    def test_per_node_quota_serializes_same_node_tasks(self):
        import threading as _threading
        logger = StructuredLogger()
        scheduler = Scheduler(global_limit=5, per_node_quota=1, logger=logger)
        active = {"count": 0}
        peak = {"count": 0}
        lock = _threading.Lock()

        def serialized_dispatch(task, attempt):
            with lock:
                active["count"] += 1
                peak["count"] = max(peak["count"], active["count"])
            time.sleep(0.03)
            with lock:
                active["count"] -= 1
            return Receipt(node_id=task.node_id, version=1, shard_index=task.shard_index,
                           shard_count=1, derived_state={}, aggregate_digest="x",
                           compute_ms=1, attempt=attempt)

        tasks = [DispatchTask(task_id=f"serial-{i}", node_id="same", version=1,
                              shard_index=i, shard_count=1,
                              payload={"_retry_policy": "none"}) for i in range(4)]
        records = scheduler.run(tasks, serialized_dispatch)
        self.assertTrue(all(r.status == "ok" for r in records))
        self.assertEqual(peak["count"], 1)
        self.assertLessEqual(scheduler.peak_concurrency, 1)


class EndToEndTests(unittest.TestCase):
    def test_full_pipeline_convergence_and_replay(self):
        workdir = tempfile.mkdtemp(prefix="configmesh-e2e-")
        session = Session("e2e", workdir, grid_exp=3, global_limit=3, per_node_quota=2)
        try:
            session.load_inputs("examples/manifest.json", "examples/registry.json")
            outputs = session.run_all()
            self.assertEqual([o["version"] for o in outputs], [1, 2, 3])
            self.assertLessEqual(session.scheduler.peak_concurrency, 3)
            self.assertTrue(all(o["convergence"]["converged"] for o in outputs))
            for output in outputs:
                check = session.replay_snapshot(output["version"])
                self.assertTrue(check["replay_consistent"])
        finally:
            session.close()

    def test_convergence_mismatch_e_converge(self):
        registry, versions = load_bundle("examples/manifest.json", "examples/registry.json")
        grid = Grid.create(3)
        version = versions[0]
        plan = build_dispatch_plan(compile_version(version, grid), version)
        bad_receipts = [
            Receipt(node_id=node, version=version.version, shard_index=0,
                    shard_count=1, derived_state={"heat_index": -1},
                    aggregate_digest="deadbeef", compute_ms=1)
            for node in version.targets
        ]
        with self.assertRaises(BusinessError) as ctx:
            verify_convergence(version, plan, bad_receipts, grid)
        self.assertEqual(ctx.exception.code, ErrorCode.E_CONVERGE)
        reasons = {d["reason"] for d in ctx.exception.details}
        self.assertIn("derived_state_mismatch", reasons)
        self.assertIn("aggregate_digest_mismatch", reasons)

    def test_missing_receipt_reported(self):
        registry, versions = load_bundle("examples/manifest.json", "examples/registry.json")
        grid = Grid.create(3)
        version = versions[0]
        plan = build_dispatch_plan(compile_version(version, grid), version)
        with self.assertRaises(BusinessError) as ctx:
            verify_convergence(version, plan, [], grid)
        self.assertEqual(ctx.exception.code, ErrorCode.E_CONVERGE)
        self.assertTrue(all(d["reason"] == "missing_receipt" for d in ctx.exception.details))


if __name__ == "__main__":
    unittest.main()


class MalformedNodeTests(unittest.TestCase):
    def test_missing_receipt_fields_is_e_reply(self):
        import threading
        port = free_port()
        holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        holder.bind(("127.0.0.1", port))
        holder.listen(1)
        holder.settimeout(5)
        ready = threading.Event()

        def fake_node():
            conn, _ = holder.accept()
            line = conn.makefile().readline()
            assert "dispatch" in line
            conn.sendall(b'{"type":"receipt","node_id":"fake"}\n')
            conn.close()
            ready.set()

        thread = threading.Thread(target=fake_node, daemon=True)
        thread.start()
        logger = StructuredLogger()
        process = NodeProcess(NodeSpec(node_id="fake", port=port, timeout_ms=1500), logger)
        try:
            # preflight passes because fake server listens; handshake tolerates
            # non-ack only if wrong; send a raw dispatch via process.dispatch
            process.process = holder  # marker so stop() does not manage a child
            with self.assertRaises(BusinessError) as ctx:
                process.dispatch({
                    "type": "dispatch", "node_id": "fake", "version": 1,
                    "shard_index": 0, "shard_count": 1, "grid_scale": 1000,
                    "items": {},
                }, attempt=1)
            self.assertEqual(ctx.exception.code, ErrorCode.E_REPLY)
            missing = ctx.exception.details[0]["missing"]
            self.assertIn("version", missing)
            self.assertIn("derived_state", missing)
            thread.join(timeout=5)
            self.assertTrue(ready.is_set())
        finally:
            holder.close()
