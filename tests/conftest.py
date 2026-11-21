from __future__ import annotations

import copy
import socket
import time
from typing import Any

import pytest

from modeldash.nodes import DispatchResult


def free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def base_raw(*, ports: list[int] | None = None) -> dict[str, Any]:
    ports = ports or [free_port(), free_port(), free_port()]
    return {
        "format_version": 1,
        "spec_version": 1,
        "seed": 17,
        "input": {"size": 6, "scale": 13},
        "nodes": [
            {"id": "n1", "host": "127.0.0.1", "port": ports[0], "quota": 1, "shard_delay_ms": 0},
            {"id": "n2", "host": "127.0.0.1", "port": ports[1], "quota": 1, "shard_delay_ms": 0},
            {"id": "n3", "host": "127.0.0.1", "port": ports[2], "quota": 1, "shard_delay_ms": 0},
        ],
        "configs": [
            {
                "version": 1,
                "priority": 1,
                "target_nodes": ["n1", "n2", "n3"],
                "shards": 3,
                "items": {"alpha": 1, "beta": 2.5, "gamma": "v1"},
            },
            {
                "version": 2,
                "priority": 2,
                "target_nodes": ["n1", "n2", "n3"],
                "shards": 3,
                "items": {"alpha": 9, "beta": 4.5, "gamma": "v2"},
            },
        ],
        "budget": {
            "global_concurrency": 2,
            "timeout_ms": 1000,
            "retry_backoff_ms": [5, 10],
            "max_retries": 2,
        },
        "tasks": [
            {"id": "t1", "config_version": 1, "node": "n1", "priority": 1, "release_after_ms": 0},
            {"id": "t2", "config_version": 2, "node": "n2", "priority": 3, "release_after_ms": 0},
        ],
    }


@pytest.fixture
def raw_spec() -> dict[str, Any]:
    return base_raw()


@pytest.fixture
def clone():
    return copy.deepcopy


class FakeDispatcher:
    """Deterministic in-process dispatcher with concurrency accounting."""

    def __init__(
        self,
        *,
        delay_ms: int = 5,
        fail_shards: dict[str, int] | None = None,
        always_fail: bool = False,
    ) -> None:
        self.delay_ms = delay_ms
        self.fail_shards = dict(fail_shards or {})
        self.always_fail = always_fail
        self.lock = __import__("threading").Lock()
        self.active = 0
        self.active_by_node: dict[str, int] = {}
        self.max_active = 0
        self.max_active_by_node: dict[str, int] = {}
        self.attempts: dict[tuple[str, int], int] = {}
        self.task_attempts: dict[str, int] = {}

    def dispatch(self, task: Any, shard_index: int, request_id: str) -> DispatchResult:
        from modeldash.errors import DispatchError

        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.active_by_node[task.node_id] = self.active_by_node.get(task.node_id, 0) + 1
            self.max_active_by_node[task.node_id] = max(
                self.max_active_by_node.get(task.node_id, 0), self.active_by_node[task.node_id]
            )
            key = (task.task_id, shard_index)
            self.attempts[key] = self.attempts.get(key, 0) + 1
            task_attempt = self.task_attempts.get(task.task_id, 0) + 1
            self.task_attempts[task.task_id] = task_attempt
        try:
            time.sleep(self.delay_ms / 1000.0)
            if self.always_fail or task_attempt <= self.fail_shards.get(task.task_id, 0):
                raise DispatchError("E_INJECTED", task.node_id, "injected fake failure")
            output = f"out-v{task.config_version}-s{shard_index}"
            input_digest = f"input-s{shard_index}"
            return DispatchResult(
                output_summary=output,
                input_digest=input_digest,
                compute_ms=self.delay_ms,
                items_used=["alpha"] if shard_index == 0 else ["beta"],
                response={},
                dispatched_ms=self.delay_ms,
            )
        finally:
            with self.lock:
                self.active -= 1
                self.active_by_node[task.node_id] -= 1
