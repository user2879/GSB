"""受全局并发上限与每节点配额约束的并行采集调度（需求 6）。

约束与语义
----------
* 全局信号量保证任一时刻在途采集数 <= ``global_concurrency``。
* 每节点信号量保证同一节点在途采集数 <= ``per_node_concurrency``。
* 任务在队列中等待两个配额的单调时钟毫秒数分别记录，调度结果与
  提交顺序、线程调度顺序无关（最终按任务键排序输出）。
* 每次尝试拥有独立 request_id；套接字超时 / 回执缺失按策略重试，
  尝试历史逐条保留，只有最终成功的那一次进入差异判定，杜绝双计入。
"""
from __future__ import annotations

import threading
import time
import uuid
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from .manifest import NodeSpec
from .nodepool import NodeProtocolError, collect_once
import socket


@dataclass(frozen=True)
class CollectTask:
    task_key: str
    node: NodeSpec
    payload: dict[str, Any]


@dataclass(frozen=True)
class Attempt:
    attempt_no: int
    request_id: str
    started_monotonic_ms: int
    ended_monotonic_ms: int
    socket_timeout_ms: int
    outcome: str
    error: str | None
    response: dict[str, Any] | None = None


@dataclass
class TaskResult:
    task_key: str
    node_id: str
    ok: bool
    response: dict[str, Any] | None
    attempts: list[Attempt] = field(default_factory=list)
    enqueued_monotonic_ms: int = 0
    wait_started_ms: int = 0
    wait_ended_ms: int = 0
    global_wait_ms: int = 0
    node_wait_ms: int = 0
    error: str | None = None


def now_ms() -> int:
    return time.monotonic_ns() // 1_000_000


class Scheduler:
    def __init__(
        self,
        global_concurrency: int,
        per_node_concurrency: int,
        socket_timeout_ms: int,
        max_attempts: int,
        retry_backoff_ms: int = 20,
    ):
        if global_concurrency < 1 or per_node_concurrency < 1:
            raise ValueError("并发上限必须为正整数")
        if max_attempts < 1:
            raise ValueError("最大尝试次数必须 >= 1")
        self.global_limit = global_concurrency
        self.per_node_limit = per_node_concurrency
        self.socket_timeout_ms = socket_timeout_ms
        self.max_attempts = max_attempts
        self.retry_backoff_ms = retry_backoff_ms
        self._global = threading.BoundedSemaphore(global_concurrency)
        self._node_slots: dict[str, threading.BoundedSemaphore] = {}
        self._slot_lock = threading.Lock()

    def _node_semaphore(self, node_id: str) -> threading.BoundedSemaphore:
        with self._slot_lock:
            sem = self._node_slots.get(node_id)
            if sem is None:
                sem = threading.BoundedSemaphore(self.per_node_limit)
                self._node_slots[node_id] = sem
            return sem

    def run(self, tasks: list[CollectTask], progress: Callable[[int, int], None] | None = None) -> list[TaskResult]:
        ordered = sorted(tasks, key=lambda task: task.task_key)
        results: list[TaskResult | None] = [None] * len(ordered)
        done_counter = {"count": 0}
        counter_lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=max(1, min(self.global_limit * 2, 32))) as pool:
            futures = {}
            for index, task in enumerate(ordered):
                futures[pool.submit(self._run_one, task)] = index
            for future in futures:
                index = futures[future]
                results[index] = future.result()
                if progress is not None:
                    with counter_lock:
                        done_counter["count"] += 1
                        progress(done_counter["count"], len(ordered))
        return [result for result in results if result is not None]

    def _run_one(self, task: CollectTask) -> TaskResult:
        result = TaskResult(task_key=task.task_key, node_id=task.node.node_id, ok=False, response=None)
        result.enqueued_monotonic_ms = now_ms()
        node_sem = self._node_semaphore(task.node.node_id)
        result.wait_started_ms = now_ms()
        with self._global:
            result.global_wait_ms = now_ms() - result.wait_started_ms
            with node_sem:
                result.wait_ended_ms = now_ms()
                result.node_wait_ms = result.wait_ended_ms - result.wait_started_ms - result.global_wait_ms
                result.node_wait_ms = max(0, result.node_wait_ms)
                for attempt_no in range(1, self.max_attempts + 1):
                    request = dict(task.payload)
                    request["request_id"] = f"{task.task_key}#try{attempt_no}-{uuid.uuid4().hex[:10]}"
                    attempt = self._attempt(task, request, attempt_no)
                    result.attempts.append(attempt)
                    if attempt.outcome == "ok":
                        result.ok = True
                        result.response = attempt.response  # type: ignore[attr-defined]
                        return result
                    if attempt_no < self.max_attempts:
                        jitter = random.uniform(0.0, self.retry_backoff_ms / 1000.0)
                        time.sleep(self.retry_backoff_ms / 1000.0 + jitter)
        result.error = result.attempts[-1].error if result.attempts else "未执行任何尝试"
        return result

    def _attempt(self, task: CollectTask, request: dict[str, Any], attempt_no: int) -> Attempt:
        started = now_ms()
        try:
            response = collect_once(task.node, request, self.socket_timeout_ms / 1000.0)
        except (NodeProtocolError, OSError, TimeoutError) as exc:
            ended = now_ms()
            return Attempt(
                attempt_no=attempt_no,
                request_id=request["request_id"],
                started_monotonic_ms=started,
                ended_monotonic_ms=ended,
                socket_timeout_ms=self.socket_timeout_ms,
                outcome="timeout" if isinstance(exc, (socket.timeout, TimeoutError)) else "error",
                error=f"{type(exc).__name__}: {exc}",
            )
        ended = now_ms()
        return Attempt(
            attempt_no=attempt_no,
            request_id=request["request_id"],
            started_monotonic_ms=started,
            ended_monotonic_ms=ended,
            socket_timeout_ms=self.socket_timeout_ms,
            outcome="ok",
            error=None,
            response=response,
        )
