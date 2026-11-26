from __future__ import annotations

import dataclasses
import json
import socket
import tempfile
import threading
import unittest
from pathlib import Path

from affinity_scheduler.errors import E_PORT, E_REPLY, E_TIMEOUT, PortError, ReplyError, TimeoutError
from affinity_scheduler.model import PLUGIN_NAMES
from affinity_scheduler.samples import sample_spec, write_sample
from affinity_scheduler.scheduler import conclusion_bytes, schedule_full
from affinity_scheduler.scoring import ScoringManager, WorkerHandle
from affinity_scheduler.spec import load_spec


class WorkerIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.sample = write_sample(self.root, port_base=40300)
        self.spec = load_spec(self.sample["spec"])

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_full_schedule_and_filter_order(self) -> None:
        with ScoringManager(self.spec) as manager:
            state = schedule_full(self.spec, manager)
        self.assertEqual(len(state["results"]), 4)
        pod1 = state["results"]["pod-0001"]
        traces = {row["node_name"]: row["steps"] for row in pod1["filter_trace"]}
        self.assertEqual(traces["node-b"][0]["plugin"], "taint_toleration")
        self.assertEqual(traces["node-b"][0]["result"], "reject")
        self.assertEqual(traces["node-c"][0]["result"], "pass")
        self.assertEqual(traces["node-c"][1]["plugin"], "node_affinity")
        self.assertEqual(traces["node-c"][1]["result"], "reject")
        for result in state["results"].values():
            self.assertEqual(result["conclusion_digest"], result["conclusion_digest"].lower())
            if result["status"] == "bound":
                self.assertEqual(list(result["receipts"]), sorted(result["receipts"]))

    def test_node_enumeration_order_is_irrelevant(self) -> None:
        reversed_spec_value = sample_spec(1, 40300)
        reversed_spec_value["nodes"] = list(reversed(reversed_spec_value["nodes"]))
        reversed_path = self.root / "config" / "spec-reversed.json"
        reversed_path.write_text(
            json.dumps(reversed_spec_value, indent=2) + "\n", encoding="utf-8"
        )
        reversed_spec = load_spec(reversed_path)
        with ScoringManager(self.spec) as manager:
            first = schedule_full(self.spec, manager)
            second = schedule_full(reversed_spec, manager)
        for pod_id in sorted(first["results"]):
            self.assertEqual(
                conclusion_bytes(first["results"][pod_id]),
                conclusion_bytes(second["results"][pod_id]),
            )

    def test_port_occupied_reports_e_port(self) -> None:
        port = self.spec.service.groups[0].port
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", port))
        listener.listen(1)
        try:
            manager = ScoringManager(self.spec)
            with self.assertRaises(PortError) as caught:
                manager.start()
            self.assertEqual(caught.exception.code, E_PORT)
        finally:
            listener.close()

    def _manager_with_socketpair(self) -> tuple[ScoringManager, socket.socket, socket.socket]:
        short_service = dataclasses.replace(self.spec.service, timeout_ms=100)
        spec = dataclasses.replace(self.spec, service=short_service)
        manager = ScoringManager(spec)
        client, server = socket.socketpair()
        handle = WorkerHandle(
            group_name=spec.service.groups[0].name,
            port=spec.service.groups[0].port,
            process=None,  # type: ignore[arg-type]
            sock=client,
            buffer=bytearray(),
        )
        manager.workers[handle.group_name] = handle
        return manager, client, server

    def test_socket_timeout_reports_e_timeout(self) -> None:
        manager, client, server = self._manager_with_socketpair()
        try:
            with self.assertRaises(TimeoutError) as caught:
                manager.score(
                    pod_id="pod-0001",
                    node_name="node-a",
                    parameter_version=1,
                    scale_exponent=3,
                    weights_myriad=self.spec.weights_myriad,
                    features={
                        "has_taints": False,
                        "untolerated_prefer_taints": 0,
                        "has_node_affinity": True,
                        "node_soft_match_weight": 1,
                        "has_pod_anti_affinity": False,
                        "anti_soft_conflicts": 0,
                        "remaining_cpu": 6,
                        "remaining_memory": 15360,
                        "allocatable_cpu": 8,
                        "allocatable_memory": 16384,
                    },
                )
            self.assertEqual(caught.exception.code, E_TIMEOUT)
        finally:
            client.close()
            server.close()

    def test_missing_reply_fields_report_e_reply(self) -> None:
        manager, client, server = self._manager_with_socketpair()

        def server_thread() -> None:
            try:
                server.recv(65536)
                server.sendall(b"{}\n")
            finally:
                server.close()

        thread = threading.Thread(target=server_thread, daemon=True)
        thread.start()
        try:
            with self.assertRaises(ReplyError) as caught:
                manager.score(
                    pod_id="pod-0001",
                    node_name="node-a",
                    parameter_version=1,
                    scale_exponent=3,
                    weights_myriad=self.spec.weights_myriad,
                    features={
                        "has_taints": False,
                        "untolerated_prefer_taints": 0,
                        "has_node_affinity": True,
                        "node_soft_match_weight": 1,
                        "has_pod_anti_affinity": False,
                        "anti_soft_conflicts": 0,
                        "remaining_cpu": 6,
                        "remaining_memory": 15360,
                        "allocatable_cpu": 8,
                        "allocatable_memory": 16384,
                    },
                )
            self.assertEqual(caught.exception.code, E_REPLY)
        finally:
            client.close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
