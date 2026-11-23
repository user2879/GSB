from __future__ import annotations

import socket
import unittest

from config_pipeline.errors import E_PORT, E_REPLY, PipelineError
from config_pipeline.manifest import Cluster
from config_pipeline.planner import ShardPlan
from config_pipeline.simulator import SimulatorProcess
from tests.helpers import free_port


def shard_for(port: int) -> ShardPlan:
    return ShardPlan(
        stable_id="c0::task::s000000",
        cluster_id="c0",
        task_id="task",
        shard_index=0,
        revision_version=1,
        config_keys=("scaled",),
        config_values={"scaled": 13},
        config_relations={"scaled": ("base", "scaled")},
        resource_units=1,
        task_concurrency=1,
        cluster_seed=7,
        cluster_fingerprint="cluster-fingerprint",
    )


class SimulatorTests(unittest.TestCase):
    def test_handshake_dispatch_and_deterministic_replay(self) -> None:
        port = free_port()
        cluster = Cluster(
            id="c0",
            host="127.0.0.1",
            port=port,
            nodes=("n1",),
            concurrency=1,
            seed=7,
        )
        process = SimulatorProcess(cluster, handshake_timeout=5.0)
        process.start()
        try:
            self.assertEqual(process.hello_reply["service"], "deterministic-cluster-sim")
            first = process.dispatch(shard_for(port))
            second = process.dispatch(shard_for(port))
            self.assertEqual(first["aggregate_digest"], second["aggregate_digest"])
            self.assertEqual(first["aggregate_summary"], second["aggregate_summary"])
            self.assertGreaterEqual(first["elapsed_ms"], 0)
        finally:
            process.stop()

    def test_occupied_port_is_e_port(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = int(listener.getsockname()[1])
        cluster = Cluster(
            id="busy",
            host="127.0.0.1",
            port=port,
            nodes=("n",),
            concurrency=1,
            seed=1,
        )
        try:
            with self.assertRaises(PipelineError) as caught:
                SimulatorProcess(cluster, handshake_timeout=1.0).start()
            self.assertEqual(caught.exception.code, E_PORT)
            self.assertEqual(caught.exception.field, "port")
        finally:
            listener.close()

    def test_missing_reply_field_is_e_reply(self) -> None:
        shard = shard_for(free_port())
        with self.assertRaises(PipelineError) as caught:
            SimulatorProcess.validate_dispatch_reply(
                shard,
                {
                    "ok": True,
                    "cluster_id": shard.cluster_id,
                    "shard_id": shard.stable_id,
                    "revision_version": shard.revision_version,
                    "aggregate_summary": {"x": 1},
                    "aggregate_digest": "bad",
                },
            )
        self.assertEqual(caught.exception.code, E_REPLY)
        self.assertEqual(caught.exception.field, "reply.elapsed_ms")


if __name__ == "__main__":
    unittest.main()
