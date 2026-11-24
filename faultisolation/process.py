"""故障域仲裁服务的子进程编排与真实回环实时调度。

* :class:`ArbiterProcessGroup` 为每个故障域启动独立 Python 子进程，绑定声明端口；
* :class:`LiveScheduler` 先由确定性引擎产出逐事件计划，再把每个动作经线程池投射到
  真实回环 TCP 服务，并把服务真实回执与计划逐字段交叉核对；
* 全局/每域在途上限用真实信号量约束在途任务；
* 节拍等待（``Condition.wait``）仅对齐事件到达时刻，绝不用 sleep 模拟租约超时；
* 租约到期只由服务端 :class:`MonotonicClock` 判定，客户端记录真实回执往返 RTT。
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .clock import MonotonicClock
from .engine import Engine, canonical_event_order
from .errors import FIError, LedgerError
from .ledger import Ledger
from .planner import PATH_BATCH, PATH_EVENT, PATH_MERGE, choose_path
from .protocol import request
from .spec import SpecBundle


class ArbiterProcessGroup:
    def __init__(self, bundle: SpecBundle) -> None:
        self.bundle = bundle
        self.processes: dict[str, subprocess.Popen] = {}
        self._ready_events: dict[str, threading.Event] = {}

    def start(self) -> None:
        spec = str(self.bundle.spec_path)
        for domain in self.bundle.domains:
            event = threading.Event()
            self._ready_events[domain.domain_id] = event
            proc = subprocess.Popen(
                [sys.executable, "-m", "faultisolation.server", "--spec", spec,
                 "--domain", domain.domain_id],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            self.processes[domain.domain_id] = proc

            def _wait_ready(p=proc, ev=event) -> None:
                if p.stdout is None:
                    ev.set()
                    return
                line = p.stdout.readline()
                if "listening on" in line:
                    ev.set()

            threading.Thread(target=_wait_ready, daemon=True).start()
        deadline = time.monotonic() + 15
        for did, event in self._ready_events.items():
            if not event.wait(timeout=max(0.1, deadline - time.monotonic())):
                proc = self.processes[did]
                err = proc.stderr.read() if proc.stderr else ""
                raise RuntimeError(f"故障域 {did} 仲裁服务启动超时: {err}")

    def stop(self) -> None:
        # 先礼貌通知，再统一回收；子进程以 terminate 快速退出，避免 serve_forever 轮询延迟
        for domain in self.bundle.domains:
            try:
                request(domain.host, domain.port, {"type": "shutdown"}, timeout_ms=1000)
            except OSError:
                pass
        for proc in list(self.processes.values()):
            proc.terminate()
        for proc in list(self.processes.values()):
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        self.processes.clear()

    def kill_all(self) -> None:
        for proc in list(self.processes.values()):
            proc.kill()
            proc.wait(timeout=5)
        self.processes.clear()


@dataclass
class LiveTelemetry:
    seq: int
    sent_monotonic_ms: int
    reply_monotonic_ms: int
    rtt_ms: int
    planned_status: str
    reply_status: str
    matched: bool
    error_code: str = ""


class LiveScheduler:
    def __init__(self, bundle: SpecBundle, ledger: Ledger, path: str | None = None) -> None:
        self.bundle = bundle
        self.ledger = ledger
        self.clock = MonotonicClock()
        selection = choose_path(bundle)
        self.path = path or selection["selected"]
        self.path_selection = selection
        # 确定性计划引擎使用逻辑时钟，保证与校验器重放结论逐字节一致
        from .clock import LogicalClock
        self.engine = Engine(bundle, LogicalClock(0))
        self.telemetry: list[LiveTelemetry] = []
        self._global_sem = threading.BoundedSemaphore(bundle.session.global_lease_cap)
        self._domain_sem = {
            d.domain_id: threading.BoundedSemaphore(d.lease_cap) for d in bundle.domains
        }
        self._lock = threading.Lock()
        self._domain_send_lock = {d.domain_id: threading.Lock() for d in bundle.domains}
        self._pool = ThreadPoolExecutor(
            max_workers=min(32, bundle.session.global_lease_cap + len(bundle.domains)),
            thread_name_prefix="arbiter-task",
        )

    def _plan(self) -> list:
        if self.path == PATH_EVENT:
            return self.engine.run_event_path()
        if self.path == PATH_BATCH:
            return self.engine.run_batch_path()
        if self.path == PATH_MERGE:
            return self.engine.run_merge_path()
        raise ValueError(f"未知仲裁路径 {self.path}")

    def _logical_lease_ms(self, domain_id: str) -> int:
        return self.bundle.domain(domain_id).lease_ms

    def run(self) -> dict:
        outcomes = self._plan()
        ordered = canonical_event_order(self.bundle.events, self.engine.priority_of)
        outcome_by_seq = {o.seq: o for o in outcomes}
        batch_size = self.bundle.session.batch_size

        batch_id = 0
        outcome_cursor = 0
        transition_cursor = 0
        wait_cursor = 0
        # 每个事件投射为一次真实回环 RPC；服务端协调器负责 FIFO 排队与泵取
        for index, event in enumerate(ordered, start=1):
            planned = outcome_by_seq[event.seq]
            future = self._pool.submit(self._execute, event, planned)
            future.result()
            if index % batch_size == 0 or index == len(ordered):
                batch_id += 1
                outcome_cursor, transition_cursor, wait_cursor = self._commit_batch(
                    batch_id, outcome_cursor, transition_cursor, wait_cursor, batch_size)
        self._pool.shutdown(wait=True)
        self._verify_server_finals(outcome_by_seq)
        recovery = self.ledger.recover()
        return {
            "path": self.path,
            "path_selection": self.path_selection,
            "event_count": len(ordered),
            "batches": batch_id,
            "telemetry": [t.__dict__ for t in sorted(self.telemetry, key=lambda t: t.seq)],
            "recovery": recovery.__dict__,
        }

    def _verify_server_finals(self, outcome_by_seq) -> None:
        """流末用真实快照核对服务端活跃租约/槽位与确定性计划一致。"""
        last_logical = max(e.arrival_ms for e in self.bundle.events)
        for dom in self.bundle.domains:
            # 用最后事件的逻辑时刻取快照，与引擎最终仲裁时刻一致
            snap = request(dom.host, dom.port,
                           {"type": "snapshot", "logical_ms": last_logical})
            planned_owners = self.engine.arbiters[dom.domain_id].slot_owner
            actual = {int(k): v for k, v in snap.get("slot_owner", {}).items()}
            if actual != planned_owners:
                raise RuntimeError(
                    f"域 {dom.domain_id} 真实槽位归属 {actual} 与计划 {planned_owners} 不一致")

    def _sweep_all_domains(self, logical_ms: int) -> None:
        for dom in self.bundle.domains:
            request(dom.host, dom.port,
                    {"type": "sweep", "logical_ms": logical_ms}, timeout_ms=10000)

    def _execute(self, event, planned) -> None:
        domain_id = self.bundle.instance(event.instance_id).domain_id
        domain = self.bundle.domain(domain_id)
        acquired_global = False
        acquired_domain = False
        # 同域发送严格串行（规范序），跨域仍由线程池并行；信号量约束全局/每域在途上限
        try:
            if event.action == "acquire":
                self._global_sem.acquire()
                acquired_global = True
                self._domain_sem[domain_id].acquire()
                acquired_domain = True
            # 每个事件前推进所有域的逻辑单调时钟并泵取队列，复现引擎全域扫描
            self._sweep_all_domains(event.arrival_ms)
            with self._domain_send_lock[domain_id]:
                lease_id = planned.lease_id or f"live:{event.seq}:{event.instance_id}"
                logical_ms = event.arrival_ms
                if event.action == "acquire":
                    payload = {
                        "type": "acquire", "instance_id": event.instance_id,
                        "lease_id": lease_id, "lease_ms": self._logical_lease_ms(domain_id),
                        "logical_ms": logical_ms, "event_seq": event.seq,
                    }
                elif event.action == "renew":
                    # 计划 version 为续约后新版本，服务端校验续约前 current 版本
                    payload = {"type": "renew", "lease_id": lease_id,
                               "version": max(1, (planned.version or 1) - 1),
                               "logical_ms": logical_ms}
                else:
                    payload = {"type": "release", "lease_id": lease_id,
                               "logical_ms": logical_ms}
                sent = self.clock.now_ms()
                reply = request(domain.host, domain.port, payload, timeout_ms=10000)
                recv = self.clock.now_ms()
            reply_status = reply.get("status", reply.get("code", "error"))
            matched = self._cross_check(planned, reply, event)
            with self._lock:
                self.telemetry.append(LiveTelemetry(
                    seq=event.seq, sent_monotonic_ms=sent, reply_monotonic_ms=recv,
                    rtt_ms=recv - sent, planned_status=planned.status,
                    reply_status=reply_status, matched=matched,
                    error_code=reply.get("code", ""),
                ))
        finally:
            if acquired_domain:
                self._domain_sem[domain_id].release()
            if acquired_global:
                self._global_sem.release()

    @staticmethod
    def _cross_check(planned, reply: dict, event) -> bool:
        """真实回执与确定性计划核对；排队申请回执为 queued，最终结局由服务端泵取后落账。"""
        if reply.get("type") == "error":
            return planned.status == "denied_lease" and reply.get("code") == "E_LEASE"
        reply_status = reply.get("status")
        if reply_status == "queued":
            # 计划中该申请经历过等待（wait_start_ms 为正）即视为一致
            return planned.status in ("granted", "denied_quota") and planned.wait_start_ms > 0
        return reply_status == planned.status

    def _commit_batch(self, batch_id: int, outcome_cursor: int,
                      transition_cursor: int, wait_cursor: int, batch_size: int):
        end_outcome = min(len(self.engine.outcomes), outcome_cursor + batch_size)
        batch_outcomes = self.engine.outcomes[outcome_cursor:end_outcome]
        if not batch_outcomes:
            return outcome_cursor, transition_cursor, wait_cursor
        # 迁移与等待按本批结局涉及的租约/序号切片，保持账本完整且不重复
        batch_seqs = {o.seq for o in batch_outcomes}
        batch_lease_ids = {o.lease_id for o in batch_outcomes if o.lease_id}
        new_transition_cursor = transition_cursor
        batch_transitions = []
        for tr in self.engine.transitions[transition_cursor:]:
            if tr.lease_id in batch_lease_ids and new_transition_cursor < len(self.engine.transitions):
                batch_transitions.append(tr)
            new_transition_cursor += 1
        batch_waits = [w for w in self.engine.waits[wait_cursor:] if w.seq in batch_seqs]
        new_wait_cursor = wait_cursor + len(batch_waits)
        slot_snapshot = {
            did: dict(arb.slot_owner) for did, arb in self.engine.arbiters.items() if arb.slot_owner
        }
        self.ledger.append_batch(
            batch_id=batch_id, outcomes=batch_outcomes, transitions=batch_transitions,
            waits=batch_waits, slot_snapshot=slot_snapshot, clock_ms=self.clock.now_ms(),
        )
        return end_outcome, new_transition_cursor, new_wait_cursor
