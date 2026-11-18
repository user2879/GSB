"""受预算约束的并行下发器。

约束实现
--------
* 全局并发上限 ``max_concurrency``：真实工作线程数，用有界信号量保证，
  配额耗尽时任务在队列中等待，绝不可能绕过上限。
* 每节点单次下发超时：socket 超时 + 策略重试次数。
* 首次回执与每次重试回执分别落日志、分别建 :class:`Receipt` 记录；
  收敛时只采信 ``accepted=True`` 的最终回执。
* 排队与每次尝试都记录单调时钟起止毫秒。
"""
from __future__ import annotations

import json
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .model import (
    CompiledItem,
    DispatchPlan,
    DispatchRecord,
    Receipt,
)
from .orchestrator import NodeHandle
from .structured_log import StructuredLogger


@dataclass
class DispatchOutcome:
    version: int
    receipts: Dict[str, Receipt]
    all_receipts: List[Receipt] = field(default_factory=list)
    records: List[DispatchRecord] = field(default_factory=list)
    final_attempts: Dict[str, int] = field(default_factory=dict)
    node_failures: Dict[str, str] = field(default_factory=dict)


class Dispatcher:
    def __init__(
        self,
        handles: Dict[str, NodeHandle],
        logger: StructuredLogger,
        max_concurrency: int = 2,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.handles = handles
        self.logger = logger
        self.max_concurrency = max(1, max_concurrency)
        self.clock = clock
        self._semaphore = threading.BoundedSemaphore(self.max_concurrency)
        self._inflight_lock = threading.Lock()
        self._inflight = 0
        self._peak_inflight = 0

    def now_ms(self) -> int:
        return int(round(self.clock() * 1000))

    def _payload_for(
        self,
        node_id: str,
        keys: List[str],
        compiled: Dict[str, CompiledItem],
    ) -> list:
        # 只把本批实际下发到该节点的上游引用带给节点，避免引用不可达键
        reachable = set(keys)
        payload = []
        for key in keys:
            item = compiled[key]
            if item.is_derived:
                refs = [
                    list(pair)
                    for pair in item.coef_quantized
                    if pair[0] in reachable
                ]
                derivation = {"refs": refs} if refs else None
            else:
                derivation = None
            payload.append(
                {
                    "key": item.key,
                    "quantized": item.quantized,
                    "derivation": derivation,
                }
            )
        return payload

    def _drain_stale(self, handle: NodeHandle, seq: int, attempt: int) -> list:
        """排空并返回属于更早 attempt 的迟到回执（只记录，不采信）。"""
        stale: list = []
        handle.conn.setblocking(False)
        try:
            while True:
                try:
                    raw = handle.recv()
                except (BlockingIOError, OSError):
                    break
                if raw.get("type") == "receipt":
                    stale.append(raw)
                    self.logger.log(
                        "dispatch.stale_receipt",
                        node=handle.node_id,
                        seq=raw.get("seq"),
                        attempt=raw.get("attempt"),
                        ignored_for_seq=seq,
                        ignored_for_attempt=attempt,
                    )
        finally:
            handle.conn.setblocking(True)
        return stale

    def _send_one_batch(
        self,
        handle: NodeHandle,
        version: int,
        seq: int,
        attempt: int,
        payload: list,
        timeout_ms: int,
    ) -> Receipt:
        # 重试前先排空上一次尝试可能迟到的回执，避免陈旧报文被当作本次结果
        if attempt > 1:
            self._drain_stale(handle, seq, attempt)
        handle.conn.settimeout(timeout_ms / 1000.0)
        message = {
            "type": "dispatch",
            "node": handle.node_id,
            "version": version,
            "seq": seq,
            "attempt": attempt,
            "items": payload,
        }
        handle.send(message)
        # 读取直到拿到与本次 seq/attempt 匹配的回执
        while True:
            raw = handle.recv()
            if raw.get("type") != "receipt":
                raise ConnectionError(f"节点 {handle.node_id} 返回了非回执报文")
            if int(raw.get("seq", -1)) == seq and int(raw.get("attempt", -1)) == attempt:
                break
            self.logger.log(
                "dispatch.stale_receipt",
                node=handle.node_id,
                seq=raw.get("seq"),
                attempt=raw.get("attempt"),
                ignored_for_seq=seq,
                ignored_for_attempt=attempt,
            )
        handle.conn.settimeout(None)
        return Receipt(
            node_id=handle.node_id,
            version=int(raw["version"]),
            digest=str(raw["digest"]),
            seq=int(raw["seq"]),
            attempt=int(raw.get("attempt", attempt)),
            accepted=False,
        )

    def _dispatch_node(
        self,
        node_id: str,
        plan: DispatchPlan,
        compiled: Dict[str, CompiledItem],
        seq_counter: Dict[str, int],
    ) -> DispatchOutcome:
        outcome = DispatchOutcome(version=plan.version, receipts={})
        handle = self.handles[node_id]
        strategy = plan.strategy
        last_receipt: Optional[Receipt] = None
        for batch_index, batch in enumerate(plan.node_batches.get(node_id, [])):
            payload = self._payload_for(node_id, batch, compiled)
            with self._inflight_lock:
                seq_counter[node_id] = seq_counter.get(node_id, 0) + 1
                seq = seq_counter[node_id]
            queued_ms = self.now_ms()
            # 配额获取：任务在此排队，全局在飞任务永不超过并发上限
            with self._semaphore:
                started_ms = self.now_ms()
                with self._inflight_lock:
                    self._inflight += 1
                    self._peak_inflight = max(self._peak_inflight, self._inflight)
                self.logger.log(
                    "dispatch.dequeued",
                    node=node_id,
                    version=plan.version,
                    seq=seq,
                    batch_index=batch_index,
                    queued_monotonic_ms=queued_ms,
                    started_monotonic_ms=started_ms,
                    inflight=self._inflight,
                )
                attempt = 0
                while True:
                    attempt += 1
                    t_attempt_start = self.now_ms()
                    try:
                        receipt = self._send_one_batch(
                            handle,
                            plan.version,
                            seq,
                            attempt,
                            payload,
                            strategy.timeout_ms,
                        )
                        ended_ms = self.now_ms()
                        record = DispatchRecord(
                            node_id=node_id,
                            version=plan.version,
                            attempt=attempt,
                            queued_monotonic_ms=queued_ms,
                            started_monotonic_ms=started_ms,
                            ended_monotonic_ms=ended_ms,
                            outcome="ok",
                            seq=seq,
                        )
                        outcome.records.append(record)
                        # 首次与重试分别成条，绝不合并
                        self.logger.log(
                            "dispatch.receipt",
                            node=node_id,
                            version=plan.version,
                            seq=seq,
                            attempt=attempt,
                            is_retry=attempt > 1,
                            digest=receipt.digest,
                            started_monotonic_ms=t_attempt_start,
                            ended_monotonic_ms=ended_ms,
                        )
                        outcome.all_receipts.append(receipt)
                        last_receipt = receipt
                        break
                    except (socket.timeout, TimeoutError, OSError, ConnectionError,
                            json.JSONDecodeError) as exc:
                        ended_ms = self.now_ms()
                        record = DispatchRecord(
                            node_id=node_id,
                            version=plan.version,
                            attempt=attempt,
                            queued_monotonic_ms=queued_ms,
                            started_monotonic_ms=started_ms,
                            ended_monotonic_ms=ended_ms,
                            outcome="timeout" if isinstance(exc, (socket.timeout, TimeoutError))
                            else "error",
                            seq=seq,
                            detail=str(exc),
                        )
                        outcome.records.append(record)
                        self.logger.log(
                            "dispatch.retry" if attempt <= strategy.max_retries
                            else "dispatch.failed",
                            node=node_id,
                            version=plan.version,
                            seq=seq,
                            attempt=attempt,
                            max_retries=strategy.max_retries,
                            started_monotonic_ms=t_attempt_start,
                            ended_monotonic_ms=ended_ms,
                            error=str(exc),
                        )
                        if attempt > strategy.max_retries:
                            with self._inflight_lock:
                                self._inflight -= 1
                            outcome.node_failures[node_id] = (
                                f"重试 {strategy.max_retries} 次后仍失败: {exc}"
                            )
                            return outcome
                        time.sleep(min(0.05 * attempt, 0.25))
                with self._inflight_lock:
                    self._inflight -= 1
        if last_receipt is not None:
            # 多批次节点：采信最大 seq（最终批次）的回执作为节点最终状态
            accepted = Receipt(**{**last_receipt.__dict__, "accepted": True})
            prev = outcome.receipts.get(node_id)
            if prev is None or accepted.seq >= prev.seq:
                outcome.receipts[node_id] = accepted
            outcome.final_attempts[node_id] = max(
                outcome.final_attempts.get(node_id, 0), last_receipt.attempt
            )
            self.logger.log(
                "dispatch.accepted",
                node=node_id,
                version=plan.version,
                seq=last_receipt.seq,
                attempt=last_receipt.attempt,
                digest=last_receipt.digest,
            )
        return outcome

    def dispatch(
        self,
        plan: DispatchPlan,
        compiled: Dict[str, CompiledItem],
    ) -> DispatchOutcome:
        node_ids = sorted(plan.node_batches)
        seq_counter: Dict[str, int] = {}
        merged = DispatchOutcome(version=plan.version, receipts={})
        self.logger.log(
            "dispatch.begin",
            version=plan.version,
            nodes=node_ids,
            max_concurrency=self.max_concurrency,
        )
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            futures = [
                pool.submit(self._dispatch_node, node_id, plan, compiled, seq_counter)
                for node_id in node_ids
            ]
            for future in futures:
                part = future.result()
                merged.receipts.update(part.receipts)
                merged.all_receipts.extend(part.all_receipts)
                merged.records.extend(part.records)
                merged.final_attempts.update(part.final_attempts)
                merged.node_failures.update(part.node_failures)
        self.logger.log(
            "dispatch.end",
            version=plan.version,
            peak_inflight=self._peak_inflight,
            concurrency_budget=self.max_concurrency,
            accepted_nodes=sorted(merged.receipts),
        )
        return merged

    @property
    def peak_inflight(self) -> int:
        return self._peak_inflight
