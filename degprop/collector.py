"""受全局/每故障域双重并发约束的真实回环采集调度。"""
from __future__ import annotations

import json
import logging
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .models import LOOPBACK, Outcome, Topology
from .outcome import plan_outcome

logger = logging.getLogger("degprop.collector")


@dataclass
class CallRecord:
    seq: int
    upstream: str
    downstream: str
    client_key: str
    domain_id: str
    outcome: str
    bucket: int
    attempts: int
    wait_start_ms: int
    wait_end_ms: int
    wait_ms: int
    latency_ms: int
    connect_port: int
    port_kind: str = "live"
    error: str | None = None

    @property
    def is_success(self) -> bool:
        return self.outcome in (Outcome.SUCCESS.value, Outcome.SLOW_SUCCESS.value)


@dataclass
class SchedulerStats:
    max_global_in_flight: int = 0
    max_domain_in_flight: dict[str, int] = field(default_factory=dict)
    queued_calls: int = 0


class CallScheduler:
    def __init__(self, topology: Topology,
                 port_map: dict[str, dict[str, int]]) -> None:
        self.topology = topology
        self.port_map = port_map
        params = topology.params
        self._global = threading.BoundedSemaphore(params.global_concurrency)
        self._domain = {
            did: threading.BoundedSemaphore(quota.in_flight_quota)
            for did, quota in topology.domains.items()
        }
        self._in_flight_global = 0
        self._in_flight_domain: dict[str, int] = {did: 0 for did in topology.domains}
        self._lock = threading.Lock()
        self.stats = SchedulerStats(max_domain_in_flight={did: 0 for did in topology.domains})

    def run(self, events) -> list[CallRecord]:
        records: list[CallRecord | None] = [None] * len(events)
        with ThreadPoolExecutor(max_workers=self.topology.params.global_concurrency,
                                thread_name_prefix="call") as pool:
            futures = {pool.submit(self._dispatch, event): idx
                       for idx, event in enumerate(events)}
            for future, idx in futures.items():
                records[idx] = future.result()
        return [r for r in records if r is not None]

    def _dispatch(self, event) -> CallRecord:
        service = self.topology.services[event.downstream]
        wait_start = time.monotonic_ns() // 1_000_000
        # 先取全局，再取域配额；任一耗尽即真实排队等待。
        self._global.acquire()
        try:
            self._domain[service.domain_id].acquire()
            wait_end = time.monotonic_ns() // 1_000_000
            try:
                with self._lock:
                    self._in_flight_global += 1
                    self.stats.max_global_in_flight = max(
                        self.stats.max_global_in_flight, self._in_flight_global)
                    self._in_flight_domain[service.domain_id] += 1
                    self.stats.max_domain_in_flight[service.domain_id] = max(
                        self.stats.max_domain_in_flight[service.domain_id],
                        self._in_flight_domain[service.domain_id])
                    if wait_end - wait_start > 0:
                        self.stats.queued_calls += 1
                return self._call(event, service, wait_start, wait_end)
            finally:
                with self._lock:
                    self._in_flight_global -= 1
                    self._in_flight_domain[service.domain_id] -= 1
                self._domain[service.domain_id].release()
        finally:
            self._global.release()

    def _call(self, event, service, wait_start: int, wait_end: int) -> CallRecord:
        plan = plan_outcome(service.seed, event.seq)
        ports = self.port_map[service.service_id]
        connect_port = ports["port"] if plan.connect_port_kind == "live" \
            else ports["refused_port"]
        started = time.monotonic_ns()
        attempts = 0
        last_error: str | None = None
        outcome = plan.outcome
        # 重试只重发同一 seq 的相同请求，不改变单次调用语义与确定性结局。
        for attempt in range(self.topology.params.max_retries + 1):
            attempts = attempt + 1
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.topology.params.socket_timeout_ms / 1000.0)
            try:
                sock.connect((LOOPBACK, connect_port))
                if plan.connect_port_kind == "refused":
                    last_error = "unexpected_live_connection"
                    continue
                request = json.dumps({"seq": event.seq,
                                      "client_key": event.client_key}).encode("utf-8")
                sock.sendall(request + b"\n")
                data = sock.recv(256)
                if not data:
                    last_error = "empty_response"
                    continue
                response = json.loads(data.decode("utf-8"))
                outcome = response["outcome"]
                last_error = None
                break
            except ConnectionRefusedError as exc:
                # 真实连接错误：连接拒绝不重试（重试不会改变拒绝语义）。
                last_error = f"ConnectionRefusedError: {exc}"
                outcome = Outcome.REFUSED.value
                break
            except socket.timeout as exc:
                # 真实套接字超时，来源只能是 settimeout 到期。
                last_error = f"socket.timeout: {exc}"
                outcome = Outcome.TIMEOUT.value
                if attempt >= self.topology.params.max_retries:
                    break
            except OSError as exc:
                last_error = f"OSError: {exc}"
                if attempt >= self.topology.params.max_retries:
                    break
            finally:
                sock.close()
        latency_ms = (time.monotonic_ns() - started) // 1_000_000
        return CallRecord(
            seq=event.seq, upstream=event.upstream, downstream=event.downstream,
            client_key=event.client_key, domain_id=service.domain_id,
            outcome=outcome, bucket=plan.bucket, attempts=attempts,
            wait_start_ms=wait_start, wait_end_ms=wait_end,
            wait_ms=wait_end - wait_start, latency_ms=latency_ms,
            connect_port=connect_port, port_kind=plan.connect_port_kind,
            error=last_error,
        )
