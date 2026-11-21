import json
import os
import sqlite3
import sys
import tempfile
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hotgray.errors import (
    HandshakeError, PortError, ReplyError, ReplayError, SnapshotError,
)
from hotgray.gateway import Gateway
from hotgray.samples import build_sample_declaration
from hotgray.snapshot import authoritative_recompute

BASE_PORT = 18050


class GatewayIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="hotgray-it-")
        cls.decl_path = os.path.join(cls.tmp, "decl.json")
        cls.db_path = os.path.join(cls.tmp, "ledger.sqlite3")
        with open(cls.decl_path, "w", encoding="utf-8") as handle:
            json.dump(build_sample_declaration(BASE_PORT), handle)
        cls.gateway = Gateway(cls.decl_path, cls.db_path)
        cls.gateway.start()

    @classmethod
    def tearDownClass(cls):
        cls.gateway.stop()

    def test_01_handshake_and_param_versions(self):
        for controller in self.gateway.backends.values():
            self.assertIn(0, controller.version_digests)
            self.assertEqual(len(controller.version_digests[0]), 64)

    def test_02_routing_and_ledger(self):
        stream = self.gateway.declaration["request_stream"]
        for request in stream[:10]:
            self.gateway.admit_request(request)
        self.gateway.flush_all()
        self.assertGreaterEqual(self.gateway.store.count_hits(), 10)
        for hit in self.gateway.store.all_hits():
            self.assertEqual(len(hit["content_digest"]), 64)

    def test_03_hotswap_version_chain(self):
        swaps = self.gateway.hotswap_gray_backends()
        for swap in swaps:
            self.assertEqual(swap["new_seq"], 1)
            controller = self.gateway.backends[swap["backend_id"]]
            self.assertEqual(controller.active_seq, 1)
            self.assertIn(0, controller.version_digests)
            self.assertIn(1, controller.version_digests)
            self.assertNotEqual(controller.version_digests[0],
                                controller.version_digests[1])
        for request in self.gateway.declaration["request_stream"][10:]:
            self.gateway.admit_request(request)
        self.gateway.flush_all()

    def test_04_split_batch_recorded(self):
        batches = self.gateway.store.conn.execute(
            "SELECT DISTINCT batch_id FROM hits WHERE split_reason IS NOT NULL"
        ).fetchall()
        self.assertTrue(batches, "hot swap mid-batch must produce a split")
        rows = self.gateway.store.all_hits()
        for batch_id in {b[0] for b in batches}:
            sub_seqs = {}
            for row in rows:
                if row["batch_id"] == batch_id:
                    sub_seqs.setdefault(row["sub_batch"],
                                        set()).add(row["backend_seq"])
            for seqs in sub_seqs.values():
                self.assertEqual(len(seqs), 1, "no mixed versions in sub-batch")

    def test_05_rebuild_disjoint_and_verify(self):
        upper = self.gateway.store.max_ledger_seq()
        event = self.gateway.rebuild_snapshot()
        snapshot = event["snapshot"]
        self.assertEqual(snapshot["ledger_seq_to"], upper)
        self.assertEqual(snapshot["incoming_during_rebuild"], 0)
        recomputed = authoritative_recompute(
            self.gateway.declaration, self.gateway.store
        )
        from hotgray.snapshot import compare_snapshots

        compare_snapshots(snapshot["body"], recomputed["groups"])

    def test_06_rebuild_admits_new_requests(self):
        import threading as _threading

        before = self.gateway.store.max_ledger_seq()
        gate = _threading.Event()
        self.gateway.rebuild_gate = gate
        event = self.gateway.rebuild_snapshot(wait=False)
        time.sleep(0.2)
        extra = {
            "sample_id": "sample-live-001",
            "rule_id": self.gateway.declaration["rules"][0]["rule_id"],
            "client_key": "client-live",
        }
        self.gateway.admit_request(extra)
        self.gateway.flush_all()
        gate.set()
        event["thread"].join(timeout=10)
        self.gateway.rebuild_gate = None
        self.assertFalse(event["thread"].is_alive())
        self.assertEqual(event["snapshot"]["ledger_seq_to"], before)
        self.assertGreaterEqual(
            event["snapshot"]["incoming_during_rebuild"], 1
        )

    def test_07_port_error(self):
        decl = build_sample_declaration(BASE_PORT + 10)
        db = os.path.join(self.tmp, "port.sqlite3")
        gw = Gateway(self.decl_path_for(decl, db), db)
        import socket

        holder = socket.socket()
        holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        holder.bind(("127.0.0.1", decl["backends"][0]["port"]))
        holder.listen(1)
        try:
            with self.assertRaises(PortError) as ctx:
                gw.start()
            self.assertEqual(ctx.exception.code, "E_PORT")
        finally:
            holder.close()
            for controller in gw.backends.values():
                if controller.process is not None and controller.process.poll() is None:
                    controller.process.kill()
                    controller.process.wait(timeout=3)
            gw.stop()

    def test_08_handshake_timeout(self):
        import socket

        controller = list(self.gateway.backends.values())[0]
        raw = socket.socket()
        raw.bind(("127.0.0.1", 0))
        raw.listen(1)
        port = raw.getsockname()[1]
        controller.port = port
        controller.process = None
        old_sock = controller.sock
        with self.assertRaises(HandshakeError) as ctx:
            controller._connect()
            controller.handshake()
        self.assertEqual(ctx.exception.code, "E_HANDSHAKE")
        raw.close()
        controller.sock = old_sock
        controller.port = controller.decl["port"]

    def test_09_missing_reply_field(self):
        from hotgray.protocol import validate_reply

        with self.assertRaises(ReplyError) as ctx:
            validate_reply({"type": "batch_ack"})
        self.assertEqual(ctx.exception.code, "E_REPLY")

    def decl_path_for(self, decl, db):
        path = os.path.join(self.tmp, f"decl-{os.path.basename(db)}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(decl, handle)
        return path


class FaultInjectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hotgray-fault-")
        self.decl_path = os.path.join(self.tmp, "decl.json")
        self.db_path = os.path.join(self.tmp, "ledger.sqlite3")
        with open(self.decl_path, "w", encoding="utf-8") as handle:
            json.dump(build_sample_declaration(18080), handle)
        self.gateway = Gateway(self.decl_path, self.db_path)
        self.gateway.start()
        for request in self.gateway.declaration["request_stream"]:
            self.gateway.admit_request(request)
        self.gateway.flush_all()
        self.gateway.rebuild_snapshot()

    def tearDown(self):
        self.gateway.stop()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def test_corrupt_reply_snapshot_error(self):
        seq = self._conn().execute(
            "SELECT ledger_seq FROM hits ORDER BY ledger_seq LIMIT 1"
        ).fetchone()[0]
        conn = self._conn()
        conn.execute("UPDATE hits SET output_digest=? WHERE ledger_seq=?",
                     ("0" * 64, seq))
        conn.commit()
        conn.close()
        with self.assertRaises(SnapshotError) as ctx:
            authoritative_recompute(self.gateway.declaration,
                                    self.gateway.store)
        self.assertEqual(ctx.exception.code, "E_SNAPSHOT")

    def test_missing_param_version_replay_error(self):
        row = self._conn().execute(
            "SELECT backend_id, backend_seq FROM param_versions "
            "ORDER BY backend_seq DESC LIMIT 1"
        ).fetchone()
        conn = self._conn()
        conn.execute("DELETE FROM param_versions WHERE backend_id=? "
                     "AND backend_seq=?", row)
        conn.commit()
        conn.close()
        with self.assertRaises(ReplayError) as ctx:
            authoritative_recompute(self.gateway.declaration,
                                    self.gateway.store)
        self.assertEqual(ctx.exception.code, "E_REPLAY")

    def test_ledger_digest_mismatch_replay_error(self):
        seq = self._conn().execute(
            "SELECT ledger_seq FROM hits ORDER BY ledger_seq LIMIT 1"
        ).fetchone()[0]
        conn = self._conn()
        conn.execute("UPDATE hits SET content_digest=? WHERE ledger_seq=?",
                     ("f" * 64, seq))
        conn.commit()
        conn.close()
        with self.assertRaises(ReplayError) as ctx:
            authoritative_recompute(self.gateway.declaration,
                                    self.gateway.store)
        self.assertEqual(ctx.exception.code, "E_REPLAY")


if __name__ == "__main__":
    unittest.main()
