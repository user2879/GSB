"""Launch and supervise real cache-node subprocesses.

Each node is a separate ``python -m edge_cache.node`` process bound to its
own loopback port.  The pool writes a JSON spec file per node (node config +
shard manifest slice), waits for a successful ``ping`` before declaring the
node ready, and guarantees best-effort shutdown on :meth:`close`.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any

from .config import BatchConfig, NodeConfig
from .errors import E_NODE_UNAVAILABLE, EdgeCacheError, Failure
from .protocol import NodeClient


@dataclass
class RunningNode:
    config: NodeConfig
    process: subprocess.Popen[bytes]
    client: NodeClient
    spec_path: str


class NodePool:
    def __init__(self, config: BatchConfig, work_dir: str,
                 clock_schedule_path: str | None = None) -> None:
        self.config = config
        self.work_dir = work_dir
        self.clock_schedule_path = clock_schedule_path
        self._nodes: dict[str, RunningNode] = {}
        os.makedirs(work_dir, exist_ok=True)

    def start_all(self) -> None:
        for node_config in self.config.nodes:
            self.start_node(node_config)

    def start_node(self, node_config: NodeConfig) -> RunningNode:
        spec = {
            "node": node_config.__dict__,
            "shards": [
                {
                    "shard_id": shard.shard_id,
                    "index": shard.index,
                    "size_units": shard.size_units,
                    "seed": shard.seed,
                    "sample_size": shard.sample_size,
                    "upstream": list(shard.upstream),
                }
                for shard in self.config.shards
            ],
        }
        fd, spec_path = tempfile.mkstemp(
            prefix=f"node-{node_config.node_id}-", suffix=".json", dir=self.work_dir
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(spec, handle)
        env = dict(os.environ)
        if self.clock_schedule_path:
            env["EDGE_CLOCK_SCHEDULE"] = self.clock_schedule_path
        cmd = [sys.executable, "-m", "edge_cache.node", spec_path]
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        client = NodeClient(node_config.port)
        try:
            client.connect()
            response = client.request({"type": "ping"})
            if not response.get("ok"):
                raise EdgeCacheError(
                    Failure(E_NODE_UNAVAILABLE, "node ping failed",
                            {"node_id": node_config.node_id, "response": response})
                )
        except EdgeCacheError:
            process.kill()
            raise
        running = RunningNode(config=node_config, process=process,
                              client=client, spec_path=spec_path)
        self._nodes[node_config.node_id] = running
        return running

    def client(self, node_id: str) -> NodeClient:
        try:
            return self._nodes[node_id].client
        except KeyError as exc:
            raise EdgeCacheError(
                Failure(E_NODE_UNAVAILABLE, "node is not running", {"node_id": node_id})
            ) from exc

    def is_running(self, node_id: str) -> bool:
        running = self._nodes.get(node_id)
        return running is not None and running.process.poll() is None

    def snapshot_all(self) -> dict[str, dict[str, Any]]:
        snapshots: dict[str, dict[str, Any]] = {}
        for node_id, running in self._nodes.items():
            response = running.client.request({"type": "snapshot"})
            if not response.get("ok"):
                raise EdgeCacheError(
                    Failure(E_NODE_UNAVAILABLE, "node snapshot failed",
                            {"node_id": node_id, "response": response})
                )
            snapshots[node_id] = response
        return snapshots

    def restore_all(self, snapshots: dict[str, dict[str, Any]]) -> None:
        for node_id, payload in snapshots.items():
            if node_id not in self._nodes:
                continue
            response = self._nodes[node_id].client.request({
                "type": "restore",
                "leases": payload.get("leases", []),
                "algebra_log": payload.get("algebra_log", []),
                "reset_clock": bool(payload.get("reset_clock", False)),
            })
            if not response.get("ok"):
                raise EdgeCacheError(
                    Failure(E_NODE_UNAVAILABLE, "node restore failed",
                            {"node_id": node_id, "response": response})
                )

    def shutdown_node(self, node_id: str) -> None:
        running = self._nodes.pop(node_id, None)
        if running is None:
            return
        try:
            running.client.request({"type": "shutdown"})
        except EdgeCacheError:
            running.process.kill()
        try:
            running.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            running.process.kill()
            running.process.wait(timeout=5)
        running.client.close()

    def close(self) -> None:
        for node_id in list(self._nodes):
            self.shutdown_node(node_id)

    def __enter__(self) -> "NodePool":
        self.start_all()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
