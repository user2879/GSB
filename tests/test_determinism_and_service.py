from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cluster_diag.determinism import compute_aggregate_canonical, compute_aggregate_summary
from cluster_diag.errors import DiagError, E_PORT, StructuredLogger
from cluster_diag.spec import load_spec
from cluster_diag.transport import ServiceManager

from .support import base_payload, free_ports, write_spec


class DeterminismTests(unittest.TestCase):
    def test_repeated_aggregate_is_byte_identical_and_revision_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = write_spec(Path(temp) / "spec.json", base_payload(free_ports(2)))
            spec = load_spec(path)
        first = compute_aggregate_canonical(spec, cluster_id="edge-b", revision_version=1, task_id="core-network")
        second = compute_aggregate_canonical(spec, cluster_id="edge-b", revision_version=1, task_id="core-network")
        third = compute_aggregate_canonical(spec, cluster_id="edge-b", revision_version=2, task_id="core-network")
        self.assertEqual(first, second)
        self.assertNotEqual(first, third)
        summary = compute_aggregate_summary(spec, cluster_id="edge-b", revision_version=1, task_id="core-network")
        self.assertEqual(summary["value_scale"], 1_000_000)


class SimulatorServiceTests(unittest.TestCase):
    def test_jsonl_service_is_deterministic_and_detects_port_conflict(self) -> None:
        ports = free_ports(2)
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            path = write_spec(base / "spec.json", base_payload(ports))
            spec = load_spec(path)
            logger = StructuredLogger(base / "logs" / "events.jsonl")
            with ServiceManager(spec, base, logger) as manager:
                client = manager.client("edge-b")
                first = client.distribute(1, "core-network", "repeat-1")
                second = client.distribute(1, "core-network", "repeat-2")
                self.assertEqual(first["aggregate_summary_canonical"], second["aggregate_summary_canonical"])
                self.assertEqual(first["aggregate_summary_digest"], second["aggregate_summary_digest"])
                self.assertIsInstance(first["compute_ms"], int)
                self.assertGreaterEqual(first["compute_ms"], 0)

                conflicting = ServiceManager(spec, base, logger)
                with self.assertRaises(DiagError) as raised:
                    conflicting.start()
                self.assertEqual(raised.exception.code, E_PORT)
                conflicting.close()
                self.assertEqual(client.ping()["type"], "pong")


if __name__ == "__main__":
    unittest.main()
