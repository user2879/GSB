from __future__ import annotations

import base64
import copy
import tempfile
import unittest
from pathlib import Path

from tunnelvpn.codec import encode_message
from tunnelvpn.config import parse_config
from tunnelvpn.errors import ErrorCode, TunnelError
from tunnelvpn.metrics import Ledger, QueueEvent, SessionRecord
from tunnelvpn.replay import ReplayVerifier
from tunnelvpn.session import SessionState
from tunnelvpn.snapshot import Snapshot, SnapshotStore, restore_snapshot

from tests.helpers import config, raw_config


def completed_fixture(tmp_path: Path):
    cfg, _ = config(tmp_path, peer_behaviors=("normal",))
    route = cfg.routes[0]
    payload = b"snapshot-replay-payload-" * 8
    encoded = encode_message(
        tunnel_id=route.id,
        sequence=0,
        cipher=route.cipher,
        mtu=route.mtu,
        payload=payload,
    )
    state = SessionState(
        session_id="alpha-000001",
        route_id=route.id,
        peer_id=route.peer_id,
        sequence=0,
        start_ms=100,
        status="COMPLETED",
        processing_started_ms=105,
        end_ms=250,
        payload_bytes=len(payload),
        forwarded_bytes=len(payload),
        wire_bytes=encoded.wire_bytes + 4 * len(encoded.frames),
        queue_events=[QueueEvent(100, 110)],
        business_code="OK",
        encapsulation_digest=encoded.payload_digest,
        peer_payload_digest=encoded.payload_digest,
        fragment_count=len(encoded.frames),
        payload_b64=base64.b64encode(payload).decode("ascii"),
    )
    record = SessionRecord(
        session_id=state.session_id,
        route_id=state.route_id,
        peer_id=state.peer_id,
        sequence=state.sequence,
        start_ms=state.start_ms,
        end_ms=state.end_ms,
        payload_bytes=state.payload_bytes,
        forwarded_bytes=state.forwarded_bytes,
        wire_bytes=state.wire_bytes,
        queue_events=tuple(state.queue_events),
        status="COMPLETED",
        business_code="OK",
        encapsulation_digest=state.encapsulation_digest,
        fragment_count=state.fragment_count,
        peer_payload_digest=state.peer_payload_digest,
    )
    ledger = Ledger([record])
    payload_doc = {
        "created_utc": "2026-11-18T00:00:00.000+00:00",
        "route_version": cfg.route_version,
        "config": cfg.to_dict(),
        "sequence_watermarks": {"alpha": 0},
        "budget_usage": {
            "global": {"used_bytes": state.wire_bytes},
            "routes": {"alpha": {"used_bytes": state.wire_bytes}},
        },
        "sessions": [state.to_dict()],
        "ledger": {"records": ledger.to_dicts(), "digest": ledger.digest()},
        "statistics": ledger.statistics(),
        "peer_behaviors": [peer.to_dict() for peer in cfg.peers],
    }
    return cfg, payload_doc


class SnapshotReplayTests(unittest.TestCase):
    def test_snapshot_roundtrip_checksum_and_listing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, payload = completed_fixture(root)
            store = SnapshotStore(root / "snapshots")
            path = store.save_payload(payload)
            loaded = store.load(path)
            self.assertEqual(loaded.route_version, cfg.route_version)
            self.assertEqual(loaded.ledger().digest(), payload["ledger"]["digest"])
            infos = store.list_snapshots()
            self.assertEqual(len(infos), 1)
            self.assertTrue(infos[0].complete)
            self.assertEqual(infos[0].session_count, 1)

    def test_truncated_snapshot_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SnapshotStore(root)
            store.latest_path.write_text('{"format":', encoding="utf-8")
            with self.assertRaises(TunnelError) as caught:
                store.load_latest()
            self.assertEqual(caught.exception.code, ErrorCode.E_STATE.value)

    def test_version_mismatch_whole_replacement_and_stable_diff(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, payload = completed_fixture(root)
            store = SnapshotStore(root / "snapshots")
            snapshot = store.load(store.save_payload(payload))
            current_raw, _ = raw_config(
                root,
                route_version="test-v2",
                route_bandwidth_kbps=32,
                route_max_sessions=7,
                global_bandwidth_kbps=128,
                peer_behaviors=("normal",),
            )
            current = parse_config(current_raw)
            restored = restore_snapshot(snapshot, current)
            report = restored.report.to_dict()
            self.assertTrue(report["whole_replaced"])
            self.assertTrue(report["version_changed"])
            self.assertEqual(report["changed_tunnel_ids"], ["alpha"])
            route_change = report["route_changes"][0]
            fields = {field["field"]: field for field in route_change["fields"]}
            self.assertIn("bandwidth_kbps", fields)
            self.assertIn("max_sessions", fields)
            self.assertEqual(fields["bandwidth_kbps"]["snapshot"], 16)
            self.assertEqual(fields["bandwidth_kbps"]["current"], 32)
            self.assertEqual(restored.config.route_version, "test-v1")
            self.assertEqual(len(restored.ledger.records()), 1)
            self.assertEqual(report["global_changes"][0]["field"], "bandwidth_kbps")

    def test_in_flight_session_becomes_unknown_and_requires_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, payload = completed_fixture(root)
            running = SessionState(
                session_id="alpha-000002",
                route_id="alpha",
                peer_id=cfg.routes[0].peer_id,
                sequence=1,
                start_ms=300,
                status="RUNNING",
                processing_started_ms=310,
                payload_b64=base64.b64encode(b"not-finished").decode("ascii"),
            ).to_dict()
            payload = copy.deepcopy(payload)
            payload["sessions"].append(running)
            payload["sequence_watermarks"]["alpha"] = 1
            store = SnapshotStore(root / "snapshots")
            snapshot = store.load(store.save_payload(payload))
            restored = restore_snapshot(snapshot, cfg)
            target = next(
                session for session in restored.sessions if session.session_id == "alpha-000002"
            )
            self.assertEqual(target.status, "UNKNOWN")
            self.assertTrue(target.requires_rerun)
            self.assertEqual(target.business_code, "UNKNOWN")

    def test_restore_rejects_statistics_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, payload = completed_fixture(root)
            tampered = copy.deepcopy(payload)
            tampered["statistics"]["global"]["session_count"] = 99
            store = SnapshotStore(root / "snapshots")
            snapshot = store.load(store.save_payload(tampered))
            with self.assertRaises(TunnelError) as caught:
                restore_snapshot(snapshot, cfg)
            self.assertEqual(caught.exception.code, "E_STATE")

    def test_replay_is_identical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, payload = completed_fixture(root)
            store = SnapshotStore(root / "snapshots")
            snapshot = store.load(store.save_payload(payload))
            report = ReplayVerifier(snapshot, current_config=cfg).verify()
            self.assertEqual(report.code, "OK")
            self.assertTrue(report.ok)
            self.assertEqual(
                report.expected_statistics["ledger_digest"],
                report.replay_statistics["ledger_digest"],
            )

    def test_replay_reports_first_session_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, payload = completed_fixture(root)
            tampered = copy.deepcopy(payload)
            tampered["sessions"][0]["business_code"] = "E_PEER"
            tampered["sessions"][0]["status"] = "FAILED"
            snapshot = Snapshot(root / "tampered.json", tampered, "unused")
            report = ReplayVerifier(snapshot, current_config=cfg).verify()
            self.assertEqual(report.code, "E_REPLAY")
            self.assertIsNotNone(report.first_mismatch)
            mismatch = report.first_mismatch.to_dict()
            self.assertEqual(mismatch["tunnel_id"], "alpha")
            self.assertEqual(mismatch["session_id"], "alpha-000001")
            self.assertEqual(mismatch["field"], "outcome")


if __name__ == "__main__":
    unittest.main()
