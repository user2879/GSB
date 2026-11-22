"""验收测试：规格校验、缓存确定性、租约、编排、快照/恢复/重放与异常码。

运行：python -m unittest discover -s tests -v
"""

import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from cdn_diag import inject  # noqa: E402
from cdn_diag.cache import NodeCache  # noqa: E402
from cdn_diag.errors import (  # noqa: E402
    GraphError,
    HandshakeError,
    LeaseError,
    PortError,
    ProtoError,
    ReplayError,
    SnapshotError,
    SpecError,
    VersionError,
)
from cdn_diag.lease import LEASE_CONFLICT_RULE, LeaseRegistry  # noqa: E402
from cdn_diag.logx import JsonlLogger  # noqa: E402
from cdn_diag.orchestrator import Orchestrator  # noqa: E402
from cdn_diag.spec import (  # noqa: E402
    Policy,
    load_content_spec,
    load_policy,
    load_requests,
)

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(PROJECT, "samples")


def make_workspace():
    ws = tempfile.mkdtemp(prefix="cdn_diag_test_")
    os.makedirs(os.path.join(ws, "samples"))
    os.makedirs(os.path.join(ws, "ledger"))
    os.makedirs(os.path.join(ws, "logs"))
    for name in ("content_spec.json", "policy.json", "requests.json"):
        shutil.copyfile(os.path.join(SAMPLES, name), os.path.join(ws, "samples", name))
    return ws


def write_json(path, doc):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)


class TestSpecValidation(unittest.TestCase):
    def _spec_doc(self):
        return {
            "spec_version": 1,
            "contents": [
                {"path": "/a", "etag": "v1", "body_base64": "QQ==", "ttl_ms": 1000},
                {"path": "/b", "derived_from": ["/a"], "combine": "concat", "ttl_ms": 1000},
            ],
        }

    def test_valid_spec_loads(self):
        ws = make_workspace()
        self.addCleanup(shutil.rmtree, ws, True)
        path = os.path.join(ws, "spec.json")
        write_json(path, self._spec_doc())
        spec = load_content_spec(path)
        self.assertEqual(spec.contents["/b"].body, b"A")
        self.assertTrue(spec.contents["/b"].etag.startswith("D"))

    def test_duplicate_path_rejected(self):
        ws = make_workspace()
        self.addCleanup(shutil.rmtree, ws, True)
        doc = self._spec_doc()
        doc["contents"].append(dict(doc["contents"][0]))
        path = os.path.join(ws, "spec.json")
        write_json(path, doc)
        with self.assertRaises(SpecError) as ctx:
            load_content_spec(path)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertEqual(ctx.exception.details["first_path"], "/a")
        self.assertEqual(ctx.exception.details["field"], "path")

    def test_bad_ttl_rejected(self):
        ws = make_workspace()
        self.addCleanup(shutil.rmtree, ws, True)
        doc = self._spec_doc()
        doc["contents"][0]["ttl_ms"] = 0
        path = os.path.join(ws, "spec.json")
        write_json(path, doc)
        with self.assertRaises(SpecError) as ctx:
            load_content_spec(path)
        self.assertEqual(ctx.exception.details["field"], "ttl_ms")

    def test_missing_derived_ref_rejected(self):
        ws = make_workspace()
        self.addCleanup(shutil.rmtree, ws, True)
        doc = self._spec_doc()
        doc["contents"][1]["derived_from"] = ["/nope"]
        path = os.path.join(ws, "spec.json")
        write_json(path, doc)
        with self.assertRaises(SpecError) as ctx:
            load_content_spec(path)
        self.assertEqual(ctx.exception.details["field"], "derived_from")

    def test_cycle_at_load_rejected(self):
        ws = make_workspace()
        self.addCleanup(shutil.rmtree, ws, True)
        doc = self._spec_doc()
        doc["contents"][0]["derived_from"] = ["/b"]
        doc["contents"][0]["combine"] = "concat"
        doc["contents"][0].pop("etag")
        doc["contents"][0].pop("body_base64")
        path = os.path.join(ws, "spec.json")
        write_json(path, doc)
        with self.assertRaises(SpecError):
            load_content_spec(path)

    def test_duplicate_cache_key_fields_rejected(self):
        ws = make_workspace()
        self.addCleanup(shutil.rmtree, ws, True)
        doc = json.load(open(os.path.join(SAMPLES, "policy.json")))
        doc["cache_key_fields"] = ["node_id", "node_id"]
        path = os.path.join(ws, "policy.json")
        write_json(path, doc)
        with self.assertRaises(SpecError) as ctx:
            load_policy(path)
        self.assertEqual(ctx.exception.details["field"], "cache_key_fields")


class TestCacheDeterminism(unittest.TestCase):
    def _run_sequence(self):
        cache = NodeCache("edge-a", 2, "lru", ("node_id", "path"))
        events = []
        ops = [
            ("store", "/a", 1000), ("store", "/b", 1100), ("lookup", "/a", 1200),
            ("store", "/c", 1300), ("lookup", "/b", 1400), ("lookup", "/a", 1500),
            ("lookup", "/a", 99000),
        ]
        for op in ops:
            if op[0] == "store":
                entry, evicted = cache.store(op[1], "v1-" + op[1], b"x", 30000, op[2], "miss_fetch")
                events.append(("store", op[1], entry.entry_version,
                               [e.cache_key for e in evicted]))
            else:
                status, entry, remaining, evicted = cache.lookup(op[1], op[2])
                events.append((status, op[1], remaining,
                               evicted.cache_key if evicted else None))
        return events

    def test_identical_runs_byte_identical(self):
        self.assertEqual(self._run_sequence(), self._run_sequence())

    def test_ttl_decrement_and_expiry(self):
        events = self._run_sequence()
        hit = events[2]
        self.assertEqual(hit[0], "hit")
        self.assertEqual(hit[2], 30000 - 200)
        expired = events[-1]
        self.assertEqual(expired[0], "expired")
        self.assertIsNotNone(expired[3])

    def test_lru_eviction_order(self):
        events = self._run_sequence()
        # /c 写入时容量为 2，LRU 受害者是 /b（/a 刚被访问）
        self.assertEqual(events[3][3], ["edge-a|/b"])


class TestLease(unittest.TestCase):
    def test_renew_version_mismatch(self):
        reg = LeaseRegistry()
        lease, _ = reg.issue("edge-a", {"/a": "v1"}, 60000)
        with self.assertRaises(LeaseError) as ctx:
            reg.renew(lease["lease_id"], lease["lease_version"] + 1, 60000)
        self.assertEqual(ctx.exception.code, "E_LEASE")

    def test_expired_lease_rejected_on_pull(self):
        reg = LeaseRegistry()
        lease, _ = reg.issue("edge-a", {"/a": "v1"}, 30)
        time.sleep(0.05)
        with self.assertRaises(LeaseError):
            reg.validate_for_pull(lease["lease_id"], lease["lease_version"], "/a", "v1")
        self.assertEqual(reg.reclaim_expired(), [])

    def test_conflict_newer_version_wins(self):
        reg = LeaseRegistry()
        old, _ = reg.issue("edge-a", {"/a": "v1"}, 60000)
        _, conflicts = reg.issue("edge-a", {"/a": "v2"}, 60000)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["rule"], LEASE_CONFLICT_RULE)
        self.assertEqual(conflicts[0]["revoked_lease_id"], old["lease_id"])
        with self.assertRaises(LeaseError):
            reg.validate_for_pull(old["lease_id"], old["lease_version"], "/a", "v1")


class TestOrchestratorErrors(unittest.TestCase):
    def _orchestrator(self, ws, **policy_kw):
        policy = Policy(
            policy_version=1,
            cache_key_fields=("node_id", "path"),
            nodes=(),
            global_inflight=2,
            per_node_inflight=1,
            pull_timeout_ms=500,
            max_retries=1,
            lease_ttl_ms=1000,
            handshake_timeout_ms=policy_kw.get("handshake_timeout_ms", 300),
            port_range=policy_kw.get("port_range", (49300, 49310)),
        )
        logger = JsonlLogger(os.path.join(ws, "logs", "test.jsonl"))
        return Orchestrator(ws, policy, logger)

    def test_port_exhausted(self):
        ws = make_workspace()
        self.addCleanup(shutil.rmtree, ws, True)
        blocker = socket.socket()
        blocker.bind(("127.0.0.1", 49300))
        blocker.listen(1)
        self.addCleanup(blocker.close)
        orch = self._orchestrator(ws, port_range=(49300, 49300))
        with self.assertRaises(PortError) as ctx:
            orch._alloc_port()
        self.assertEqual(ctx.exception.code, "E_PORT")

    def test_handshake_timeout(self):
        ws = make_workspace()
        self.addCleanup(shutil.rmtree, ws, True)
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        self.addCleanup(srv.close)

        def silent():
            conn, _ = srv.accept()
            time.sleep(2)
            conn.close()

        threading.Thread(target=silent, daemon=True).start()
        orch = self._orchestrator(ws, handshake_timeout_ms=300)
        handle = types.SimpleNamespace(name="fake", port=port)
        with self.assertRaises(HandshakeError):
            orch._handshake(handle, "fake")

    def test_handshake_missing_field(self):
        ws = make_workspace()
        self.addCleanup(shutil.rmtree, ws, True)
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        self.addCleanup(srv.close)

        def bad_ack():
            conn, _ = srv.accept()
            conn.makefile("rb")
            conn.sendall(b'{"type": "hello_ack"}\n')
            time.sleep(0.2)
            conn.close()

        threading.Thread(target=bad_ack, daemon=True).start()
        orch = self._orchestrator(ws)
        handle = types.SimpleNamespace(name="fake", port=port)
        with self.assertRaises(ProtoError) as ctx:
            orch._handshake(handle, "fake")
        self.assertEqual(ctx.exception.details["missing_field"], "protocol")


class IntegrationBase(unittest.TestCase):
    """完整流程基类：每个用例独立临时工作区。"""

    def setUp(self):
        self.ws = make_workspace()
        self.addCleanup(shutil.rmtree, self.ws, True)
        self.logger = JsonlLogger(os.path.join(self.ws, "logs", "test.jsonl"))
        policy = load_policy(os.path.join(self.ws, "samples", "policy.json"))
        self.batches = load_requests(os.path.join(self.ws, "samples", "requests.json"))
        self.orch = Orchestrator(self.ws, policy, self.logger)
        self.orch.start()
        self.addCleanup(self.orch.stop)

    def run_full_flow(self):
        self.orch.run_batches(self.batches)
        self.orch.evolve_content("/style.css", "v2-style", b"STYLE-V2", tick_ms=100000)
        self.orch.save_snapshot()
        self.orch.kill_nodes()
        diff, replaced = self.orch.restore()
        self.orch.restart_nodes(restore_entries=True)
        self.orch.kill_nodes()
        self.orch.restart_nodes(restore_entries=False)
        return diff, replaced


class TestIntegration(IntegrationBase):
    def test_happy_path_replay_ok(self):
        diff, replaced = self.run_full_flow()
        self.assertFalse(replaced)
        self.assertEqual(diff, {"added": [], "removed": [], "changed": []})
        verdict = self.orch.replay()
        self.assertEqual(verdict["verdict"], "OK")
        self.assertGreater(verdict["compared_fields"], 50)

    def test_replay_detects_tampered_ledger(self):
        self.run_full_flow()
        # 篡改账本中一条请求事件的 etag，重放必须定位首个不一致
        row = self.orch.ledger.conn.execute(
            "SELECT seq, payload FROM events WHERE kind='request' ORDER BY seq LIMIT 1"
        ).fetchone()
        payload = json.loads(row["payload"])
        payload["etag"] = "tampered-etag"
        self.orch.ledger.conn.execute(
            "UPDATE events SET payload=? WHERE seq=?",
            (json.dumps(payload, ensure_ascii=False, sort_keys=True), row["seq"]),
        )
        self.orch.ledger.conn.commit()
        with self.assertRaises(ReplayError) as ctx:
            self.orch.replay()
        self.assertEqual(ctx.exception.code, "E_REPLAY")
        self.assertEqual(ctx.exception.details["field"], "etag")
        self.assertEqual(ctx.exception.details["node"], payload["node"])
        self.assertEqual(ctx.exception.details["path"], payload["path"])

    def test_inject_cycle_e_graph(self):
        with self.assertRaises(GraphError) as ctx:
            self.orch.evolve_content(
                "/index.html", "v2-index", b"INDEX-V2",
                tick_ms=100000,
                derived_from=inject.cycle_derivation_override(),
            )
        self.assertEqual(ctx.exception.code, "E_GRAPH")

    def test_inject_truncate_e_snapshot(self):
        self.orch.run_batches(self.batches)
        self.orch.save_snapshot()
        inject.truncate_snapshot(self.orch.snapshot_path)
        self.orch.ledger.close()
        from cdn_diag.ledger import restore_snapshot
        from cdn_diag.spec import load_content_spec

        with self.assertRaises(SnapshotError):
            restore_snapshot(
                self.orch.snapshot_path,
                self.orch.ledger_path,
                load_content_spec(self.orch.spec_path),
            )

    def test_inject_digest_e_version(self):
        self.orch.run_batches(self.batches)
        self.orch.save_snapshot()
        inject.tamper_snapshot_digest(self.orch.snapshot_path)
        self.orch.ledger.close()
        from cdn_diag.ledger import restore_snapshot
        from cdn_diag.spec import load_content_spec

        with self.assertRaises(VersionError):
            restore_snapshot(
                self.orch.snapshot_path,
                self.orch.ledger_path,
                load_content_spec(self.orch.spec_path),
            )

    def test_version_drift_full_replace(self):
        """磁盘声明与快照不一致 -> 整体替换并列出变更路径。"""
        self.orch.run_batches(self.batches)
        self.orch.save_snapshot()
        # 演进内容但不落盘到工作声明：直接改磁盘工作副本制造漂移
        spec_path = self.orch.spec_path
        doc = json.load(open(spec_path))
        doc["contents"].append(
            {"path": "/new.txt", "etag": "v1-new",
             "body_base64": "TkVX", "ttl_ms": 5000}
        )
        for item in doc["contents"]:
            if item["path"] == "/app.js":
                item["etag"] = "v9-drift"
        write_json(spec_path, doc)
        self.orch.ledger.close()
        from cdn_diag.ledger import restore_snapshot
        from cdn_diag.spec import load_content_spec

        ledger, diff, replaced = restore_snapshot(
            self.orch.snapshot_path,
            self.orch.ledger_path,
            load_content_spec(spec_path),
            logger=self.logger,
        )
        self.assertTrue(replaced)
        self.assertEqual(diff["added"], ["/new.txt"])
        self.assertEqual(diff["changed"], ["/app.js", "/page/app"])
        self.assertEqual(ledger.entries_for("edge-a"), [])
        self.assertEqual(ledger.leases(), [])
        # 旧快照只读保留
        mode = os.stat(self.orch.snapshot_path).st_mode
        self.assertFalse(mode & 0o200)


if __name__ == "__main__":
    unittest.main()
