from __future__ import annotations

import concurrent.futures
import tempfile
import threading
import unittest
from pathlib import Path

from gray_merge_lab.engine import GatewayEngine, InProcessTransport, ModelBank
from gray_merge_lab.errors import LabError
from gray_merge_lab.gateway import (
    GatewayClient,
    GatewayTCPServer,
    TcpBackendTransport,
    WorkerManager,
)
from gray_merge_lab.ledger import Ledger
from tests.helpers import free_ports, test_spec


class GatewayIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.spec = test_spec(
            ports=free_ports(2),
            sample_count=4,
            batch_size=2,
            weight_bp=10000,
            client_key="gateway-client",
        )
        self.manager = WorkerManager(self.spec.backends).start()
        self.resources = []

    def tearDown(self):
        for close in reversed(self.resources):
            try:
                close()
            except Exception:
                pass
        self.manager.stop()
        self.temp.cleanup()

    def _engine_and_server(self, name: str):
        bank = ModelBank(self.spec)
        transport = TcpBackendTransport(bank)
        ledger = Ledger(self.root / f"{name}.sqlite3")
        engine = GatewayEngine(
            self.spec,
            ledger,
            scale_exp=3,
            snapshot_path=self.root / f"{name}.snapshot.json",
            transport=transport,
        )
        server = GatewayTCPServer(engine, host="127.0.0.1", port=0)
        server.start_background()
        self.resources.append(transport.close)
        self.resources.append(ledger.close)
        self.resources.append(lambda: (server.shutdown(), server.server_close()))
        return engine, ledger, server

    @staticmethod
    def _projection(rows):
        return [
            {
                key: item[key]
                for key in (
                    "batch_id",
                    "rule_id",
                    "destination",
                    "client_bucket",
                    "client_key",
                    "sample_ids",
                    "output_digest",
                    "backend_param_digest",
                    "backend_param_version",
                    "merge_seq",
                )
            }
            for item in rows
        ]

    def test_loopback_gateway_worker_inference_and_batch_plan(self):
        engine, ledger, server = self._engine_and_server("single")
        with GatewayClient(server.host, server.port) as client:
            for sample in self.spec.samples:
                response = client.accept(sample.declaration())
                self.assertTrue(response["ok"])
            flush = client.flush()
        self.assertEqual(flush["samples"], 4)
        self.assertEqual(flush["batches"], 2)
        self.assertEqual(len(ledger.verify_integrity()["content_digest"]), 71)
        self.assertEqual(len(ledger.batches()), 2)
        for row in flush["batch_rows"]:
            self.assertEqual(row["destination"], "canary")
            self.assertEqual(row["backend_param_version"], 1)
            self.assertEqual(row["merge_seq"], 2)

        in_process_bank = ModelBank(self.spec)
        in_process = InProcessTransport(in_process_bank, duration_provider=lambda *_: 1)
        first_batch = flush["batch_rows"][0]
        sample_map = {sample.sample_id: sample for sample in self.spec.samples}
        samples = [sample_map[sample_id] for sample_id in first_batch["sample_ids"]]
        result = in_process.infer("canary", samples, first_batch["batch_id"])
        self.assertEqual(result.output_digest, first_batch["output_digest"])
        self.assertEqual(result.backend_param_digest, first_batch["backend_param_digest"])

    def test_concurrent_and_reversed_submission_have_same_partition(self):
        _, _, server_a = self._engine_and_server("concurrent")
        _, _, server_b = self._engine_and_server("reversed")
        clients = [GatewayClient(server_a.host, server_a.port) for _ in range(3)]
        self.resources.extend(client.close for client in clients)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                futures = [
                    pool.submit(client.accept, sample.declaration())
                    for client, sample in zip(clients, self.spec.samples[:3])
                ]
                for future in futures:
                    self.assertTrue(future.result()["ok"])
            self.assertTrue(clients[0].accept(self.spec.samples[3].declaration())["ok"])
            concurrent_flush = clients[0].flush()
        finally:
            for client in clients:
                client.close()

        with GatewayClient(server_b.host, server_b.port) as client:
            for sample in reversed(self.spec.samples):
                self.assertTrue(client.accept(sample.declaration())["ok"])
            sequential_flush = client.flush()

        self.assertEqual(
            self._projection(concurrent_flush["batch_rows"]),
            self._projection(sequential_flush["batch_rows"]),
        )
        self.assertEqual(concurrent_flush["sample_to_batch"], sequential_flush["sample_to_batch"])

    def test_hot_update_reaches_worker_and_reinference_uses_new_version(self):
        engine, ledger, server = self._engine_and_server("hot")
        with GatewayClient(server.host, server.port) as client:
            for sample in self.spec.samples:
                self.assertTrue(client.accept(sample.declaration())["ok"])
            flush = client.flush()
        batch_id = flush["batch_rows"][0]["batch_id"]
        summary = engine.hot_update_backend("canary", {"seed": 777, "hidden_dim": 8})
        self.assertEqual(summary["version"], 2)
        reinfer = engine.reinfer_existing_batch(batch_id)
        self.assertEqual(reinfer["backend_param_version"], 2)
        self.assertEqual(reinfer["adopted_merge_seq"], 3)
        self.assertEqual(len(ledger.receipt_rows(batch_id)), 4)

    def test_gateway_rejects_non_loopback_host(self):
        engine, _, _ = self._engine_and_server("host")
        with self.assertRaises(LabError) as caught:
            GatewayTCPServer(engine, host="0.0.0.0", port=0)
        self.assertEqual(caught.exception.code, "E_SPEC")
        self.assertEqual(caught.exception.field, "host")

    def test_gateway_flush_reports_spec_error_for_duplicate_sequence(self):
        _, _, server = self._engine_and_server("invalid")
        with GatewayClient(server.host, server.port) as client:
            duplicate = self.spec.samples[1].declaration()
            duplicate["sample_seq"] = self.spec.samples[0].sample_seq
            self.assertTrue(client.accept(self.spec.samples[0].declaration())["ok"])
            self.assertTrue(client.accept(duplicate)["ok"])
            with self.assertRaises(LabError) as caught:
                client.flush()
        self.assertEqual(caught.exception.code, "E_SPEC")
        self.assertEqual(caught.exception.field, "samples[1].sample_seq")


if __name__ == "__main__":
    unittest.main()

