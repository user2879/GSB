from __future__ import annotations

import json
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path

from cluster_lease.canonical import canonical_json
from cluster_lease.errors import LeaseError
from cluster_lease.loader import load_inputs
from cluster_lease.models import ClusterSpec
from cluster_lease.orchestrator import ServiceOrchestrator, ServiceProcess
from cluster_lease.state_machine import ArbitrationEngine

from .helpers import write_bundle


class LoaderTests(unittest.TestCase):
    def test_valid_and_injections(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = write_bundle(Path(temporary))
            bundle = load_inputs(paths["clusters"], paths["holders"], paths["events"])
            self.assertEqual(14, len(bundle.events))
            self.assertEqual(2, len(bundle.clusters))
            with self.assertRaises(LeaseError) as missing_ref:
                load_inputs(
                    paths["clusters"],
                    paths["holders"],
                    paths["events"],
                    injection="missing-ref",
                )
            self.assertEqual("E_REF", missing_ref.exception.code)
            self.assertIn(".targets", missing_ref.exception.field or "")
            with self.assertRaises(LeaseError) as timing:
                load_inputs(
                    paths["clusters"],
                    paths["holders"],
                    paths["events"],
                    injection="timing",
                )
            self.assertEqual("E_TIMING", timing.exception.code)
            self.assertIn("lease_duration_ms", timing.exception.field or "")

    def test_first_spec_violation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = write_bundle(Path(temporary))
            raw = json.loads(Path(paths["clusters"]).read_text(encoding="utf-8"))
            raw["clusters"][1]["port"] = raw["clusters"][0]["port"]
            Path(paths["clusters"]).write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(LeaseError) as error:
                load_inputs(paths["clusters"], paths["holders"], paths["events"])
            self.assertEqual("E_SPEC", error.exception.code)
            self.assertEqual("clusters[1].port", error.exception.field)


class EngineTests(unittest.TestCase):
    def spec(self) -> ClusterSpec:
        return ClusterSpec("alpha", "127.0.0.1", 39171, 5000, 1000, 1)

    @staticmethod
    def request(event_seq: int, timestamp: int, holder: str, event_type: str) -> dict[str, object]:
        return {
            "cluster_id": "alpha",
            "event_seq": event_seq,
            "event_type": event_type,
            "holder_id": holder,
            "timestamp_ms": timestamp,
            "type": "lease",
        }

    def test_priority_expiry_release_and_determinism(self) -> None:
        requests = [
            self.request(1, 0, "mid", "apply"),
            self.request(2, 1000, "mid", "renew"),
            self.request(3, 5999, "high", "apply"),
            self.request(4, 6002, "mid", "renew"),
            self.request(5, 6100, "high", "release"),
            self.request(6, 6200, "mid", "apply"),
        ]
        outputs = []
        for _ in range(2):
            engine = ArbitrationEngine(
                self.spec(), {"high": 9, "mid": 5}, clock_mode="logical"
            )
            outputs.append([engine.process(request) for request in requests])
        self.assertEqual([canonical_json(item) for item in outputs[0]], [canonical_json(item) for item in outputs[1]])
        self.assertEqual("granted", outputs[0][0]["decision"])
        self.assertEqual("renewed", outputs[0][1]["decision"])
        self.assertEqual("higher_priority_takeover", outputs[0][2]["reason"])
        self.assertEqual("rejected", outputs[0][3]["decision"])
        self.assertEqual("released", outputs[0][4]["decision"])
        self.assertEqual("granted", outputs[0][5]["decision"])

    def test_real_clock_expiry(self) -> None:
        spec = ClusterSpec("real", "127.0.0.1", 39182, 120, 50, 1)
        engine = ArbitrationEngine(spec, {"h": 1}, clock_mode="real")
        engine.process(
            {
                "cluster_id": "real",
                "event_seq": 1,
                "event_type": "apply",
                "holder_id": "h",
                "timestamp_ms": 0,
                "type": "lease",
            }
        )
        time.sleep(0.18)
        status = engine.status(0)
        self.assertEqual(0, status["used_capacity"])
        self.assertEqual("lease_expired", status["transitions"][0]["reason"])


class ProtocolErrorTests(unittest.TestCase):
    def test_handshake_timeout_and_missing_reply_field(self) -> None:
        spec = ClusterSpec("probe", "127.0.0.1", 39183, 1000, 200, 1)
        process = ServiceProcess(
            spec,
            {"h": 1},
            parameter_version=1,
            runtime_dir=tempfile.mkdtemp(),
            handshake_timeout_ms=120,
        )

        silent_client, silent_server = socket.socketpair()
        def silent() -> None:
            silent_server.recv(4096)
            time.sleep(0.4)
        thread = threading.Thread(target=silent, daemon=True)
        thread.start()
        with self.assertRaises(LeaseError) as timeout_error:
            process._handshake(silent_client)
        self.assertEqual("E_HANDSHAKE", timeout_error.exception.code)
        silent_client.close()
        silent_server.close()

        bad_client, bad_server = socket.socketpair()
        def bad() -> None:
            bad_server.recv(4096)
            bad_server.sendall(b'{"ok":true,"type":"hello","cluster_id":"probe"}\n')
        thread = threading.Thread(target=bad, daemon=True)
        thread.start()
        with self.assertRaises(LeaseError) as reply_error:
            process._handshake(bad_client)
        self.assertEqual("E_REPLY", reply_error.exception.code)
        self.assertEqual("handshake.protocol", reply_error.exception.field)
        bad_client.close()
        bad_server.close()

    def test_port_conflict(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        listener.listen(1)
        spec = ClusterSpec("occupied", "127.0.0.1", port, 1000, 200, 1)
        orchestrator = ServiceOrchestrator(
            tempfile.mkdtemp(), clock_mode="logical", startup_timeout_ms=1000
        )
        try:
            with self.assertRaises(LeaseError) as error:
                orchestrator.start([spec], {"occupied": {"h": 1}}, {"occupied": 1})
            self.assertEqual("E_PORT", error.exception.code)
        finally:
            orchestrator.stop()
            listener.close()


if __name__ == "__main__":
    unittest.main()
