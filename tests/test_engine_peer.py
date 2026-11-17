from __future__ import annotations

import json
import socket
import tempfile
import unittest
from pathlib import Path

from tunnelvpn.codec import encode_message
from tunnelvpn.engine import TunnelEngine, send_inbound_payload
from tunnelvpn.peer import read_json_line, send_frame
from tunnelvpn.replay import ReplayVerifier
from tunnelvpn.snapshot import SnapshotStore, restore_snapshot

from tests.helpers import PeerGroup, config


class EnginePeerIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg, _ = config(
            self.root,
            peer_behaviors=("normal", "digest-error", "delay-script", "unreachable"),
            peer_timeout_ms=200,
            connect_timeout_ms=100,
            fragment_timeout_ms=120,
        )
        self.peers = PeerGroup(self.cfg, self.root / "peer-logs")
        self.peers.__enter__()
        self.engine = TunnelEngine(
            self.cfg, log_path=self.root / "logs" / "engine.jsonl"
        )
        self.engine.start()

    def tearDown(self) -> None:
        self.engine.stop()
        self.peers.__exit__(None, None, None)
        self.tmp.cleanup()

    def test_normal_and_mtu_fragment_forwarding(self) -> None:
        routes = self.cfg.route_map()
        small = b"normal-session"
        response = send_inbound_payload(routes["alpha"], small, timeout=10)
        self.assertEqual(response["code"], "OK")
        self.assertEqual(response["fragment_count"], 1)
        self.assertEqual(response["payload_bytes"], len(small))

        large = b"fragment-" * 80
        response = send_inbound_payload(routes["alpha"], large, timeout=10)
        self.assertEqual(response["code"], "OK")
        self.assertGreater(response["fragment_count"], 1)
        self.assertEqual(response["payload_bytes"], len(large))
        self.assertEqual(len(self.engine.ledger.records()), 2)

        store = SnapshotStore(self.root / "snapshots")
        snapshot = store.save_engine(self.engine)
        restored = restore_snapshot(snapshot, self.cfg)
        self.assertTrue(restored.report.whole_replaced)
        report = ReplayVerifier(snapshot, current_config=self.cfg).verify()
        self.assertEqual(report.code, "OK", report.first_mismatch)

    def test_business_codes_and_structured_logs(self) -> None:
        routes = self.cfg.route_map()
        corrupt = send_inbound_payload(routes["corrupt"], b"digest", timeout=10)
        dead = send_inbound_payload(routes["dead"], b"peer", timeout=10)
        slow = send_inbound_payload(routes["slow"], b"timeout", timeout=10)
        self.assertEqual(corrupt["code"], "E_DIGEST")
        self.assertEqual(dead["code"], "E_PEER")
        self.assertEqual(slow["code"], "E_TIMEOUT")

        codes = {
            record.business_code
            for record in self.engine.ledger.records()
        }
        self.assertEqual(codes, {"E_DIGEST", "E_PEER", "E_TIMEOUT"})
        self.assertEqual(
            self.engine.ledger.statistics()["business_code_counts"],
            {"E_DIGEST": 1, "E_PEER": 1, "E_TIMEOUT": 1},
        )
        log_lines = (self.root / "logs" / "engine.jsonl").read_text(encoding="utf-8").splitlines()
        events = [json.loads(line) for line in log_lines]
        self.assertTrue(any(event["code"] == "E_DIGEST" for event in events))
        self.assertTrue(any(event["code"] == "E_PEER" for event in events))
        self.assertTrue(any(event["code"] == "E_TIMEOUT" for event in events))

    def test_missing_fragment_reports_e_frag_with_offset(self) -> None:
        route = self.cfg.route_map()["alpha"]
        payload = b"fragment-injection" * 40
        encoded = encode_message(
            tunnel_id=route.id,
            sequence=700,
            cipher=route.cipher,
            mtu=route.mtu,
            payload=payload,
        )
        self.assertGreaterEqual(len(encoded.frames), 3)
        missing_index = 1
        missing_offset = missing_index * (route.mtu - 84)
        with socket.create_connection(
            (route.target.host, route.target.port), timeout=2
        ) as sock:
            sock.settimeout(3)
            for index, frame in enumerate(encoded.frames):
                if index != missing_index:
                    send_frame(sock, frame)
            sock.shutdown(socket.SHUT_WR)
            reply = read_json_line(sock)
        self.assertEqual(reply["code"], "E_FRAG")
        self.assertIn(missing_offset, reply["missing_offsets"])

    def test_session_ledger_record_is_unique(self) -> None:
        route = self.cfg.route_map()["alpha"]
        response = send_inbound_payload(route, b"one-record", timeout=10)
        self.assertEqual(response["code"], "OK")
        records = self.engine.ledger.records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].session_id, response["session_id"])
        self.assertEqual(records[0].forwarded_bytes, len(b"one-record"))
        self.assertEqual(records[0].queue_count, 0)

    def test_restored_engine_can_continue_without_session_id_reuse(self) -> None:
        route = self.cfg.route_map()["alpha"]
        first = send_inbound_payload(route, b"before-restart", timeout=10)
        self.assertEqual(first["code"], "OK")
        store = SnapshotStore(self.root / "snapshots")
        snapshot = store.save_engine(self.engine)
        restored = restore_snapshot(snapshot, self.cfg)
        self.engine.stop()
        restarted = TunnelEngine(
            restored.config,
            ledger=restored.ledger,
            log_path=self.root / "logs" / "restarted.jsonl",
            sequence_watermarks=restored.sequence_watermarks,
            budget_usage=restored.budget_usage,
            restored_sessions=restored.sessions,
        )
        restarted.start()
        self.engine = restarted
        second = send_inbound_payload(route, b"after-restart", timeout=10)
        self.assertEqual(second["code"], "OK")
        self.assertEqual(second["session_id"], "alpha-000002")
        self.assertEqual(second["sequence"], 1)
        self.assertEqual(len(restarted.ledger.records()), 2)


if __name__ == "__main__":
    unittest.main()
