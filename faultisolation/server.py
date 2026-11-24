"""基于真实回环 TCP 的故障域仲裁服务（每域独立子进程）。

协议为 TCP 上的 JSON 行（见 :mod:`faultisolation.protocol`）。服务复用与确定性引擎
完全相同的 :class:`DomainArbiter` 状态机与 :class:`DomainCoordinator` 排队器：

* 物理单调时钟 :class:`MonotonicClock` 测量真实回执往返 RTT；
* 仲裁逻辑时刻取自请求注入的 ``logical_ms``（事件到达时刻），保证与离线重放一致；
* 容量不足由协调器 FIFO 入队，到期回收/释放后自动泵取，严格不允许优先级插队；
* 不使用任何线程延时或 sleep 模拟超时，到期只来自单调时钟比较。
"""

from __future__ import annotations

import argparse
import socketserver
import sys
import threading
from typing import Any

from .clock import MonotonicClock
from .coordinator import DomainCoordinator
from .engine import DomainArbiter
from .errors import FIError, LeaseError
from .protocol import decode_line, encode_line
from .spec import InstanceSpec, load_spec


class LogicalPassthroughClock:
    """返回最近一次请求注入的逻辑单调毫秒；无注入时使用物理单调时钟。"""

    def __init__(self, physical: MonotonicClock) -> None:
        self.physical = physical
        self._logical: int | None = None

    def set_logical(self, value: int | None) -> None:
        # 逻辑单调时钟只前进不回退；同域发送锁保证事件按规范序到达
        if value is None:
            self._logical = None
        else:
            value = int(value)
            self._logical = value if self._logical is None or value >= self._logical else self._logical

    def now_ms(self) -> int:
        return self.physical.now_ms() if self._logical is None else self._logical


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        arbiter: DomainArbiter = self.server.arbiter  # type: ignore[attr-defined]
        coordinator: DomainCoordinator = self.server.coordinator  # type: ignore[attr-defined]
        instances: dict[str, InstanceSpec] = self.server.instances  # type: ignore[attr-defined]
        physical: MonotonicClock = self.server.physical_clock  # type: ignore[attr-defined]
        logical_clock: LogicalPassthroughClock = arbiter.clock  # type: ignore[attr-defined]
        for raw_line in self.rfile:
            line = raw_line.rstrip(b"\n")
            if not line:
                continue
            physical_sent = physical.now_ms()
            try:
                req = decode_line(line)
                with self.server.domain_lock:
                    if "logical_ms" in req:
                        logical_clock.set_logical(int(req["logical_ms"]))
                    now = arbiter.clock.now_ms()
                    reply = self._dispatch(arbiter, coordinator, instances, req, now)
            except FIError as exc:
                reply = {"type": "error", "code": exc.code, "message": exc.message,
                         "context": exc.context}
            except Exception as exc:
                reply = {"type": "error", "code": "E_PROTO", "message": str(exc)}
            physical_recv = physical.now_ms()
            reply["physical_clock_ms"] = physical_recv
            reply["rtt_ms"] = physical_recv - physical_sent
            self.wfile.write(encode_line(reply))
            self.wfile.flush()
            if req.get("type") == "shutdown":
                return

    @staticmethod
    def _dispatch(arbiter: DomainArbiter, coordinator: DomainCoordinator,
                  instances: dict[str, InstanceSpec], req: dict[str, Any], now: int) -> dict[str, Any]:
        kind = req.get("type")
        if kind == "acquire":
            instance = instances[req["instance_id"]]
            lease_id = req["lease_id"]
            lease_ms = req.get("lease_ms")
            # 到期回收已由事件前的显式 sweep 完成，这里只泵取队列并提交申请
            pumped = coordinator.pump(now)
            status, outcome = coordinator.submit_acquire(lease_id, instance, int(req["event_seq"]), now)
            if status == "queued":
                return {"type": "ack", "status": "queued", "lease_id": lease_id,
                        "version": 0, "slots": [], "expiry_ms": 0, "clock_ms": now,
                        "pumped_seqs": [o.seq for o in pumped]}
            assert outcome is not None
            return {
                "type": "ack", "status": outcome.status, "lease_id": lease_id,
                "version": outcome.version, "slots": [int(x) for x in outcome.slots.split(",") if x],
                "expiry_ms": outcome.expiry_ms, "clock_ms": now,
                "pumped_seqs": [o.seq for o in pumped],
            }
        if kind == "renew":
            try:
                status, lease, expiry = arbiter.renew(req["lease_id"], int(req["version"]), now)
            except LeaseError as exc:
                return {"type": "error", "code": exc.code, "message": exc.message,
                        "context": exc.context, "clock_ms": now}
            return {
                "type": "ack", "status": status, "lease_id": req["lease_id"],
                "version": lease.version if lease else int(req["version"]),
                "slots": list(lease.slots) if lease else [], "expiry_ms": expiry,
                "clock_ms": now,
            }
        if kind == "release":
            status, lease = arbiter.release(req["lease_id"], now)
            pumped = coordinator.pump(now)
            return {
                "type": "ack", "status": status, "lease_id": req["lease_id"],
                "version": lease.version if lease else 0,
                "slots": list(lease.slots) if lease else [],
                "expiry_ms": lease.expiry_ms if lease else 0, "clock_ms": now,
                "pumped_seqs": [o.seq for o in pumped],
            }
        if kind == "sweep":
            expired = arbiter.sweep_expired(now)
            pumped = coordinator.pump(now)
            return {"type": "ack", "status": "swept", "expired_lease_ids": expired,
                    "pumped_seqs": [o.seq for o in pumped], "clock_ms": now}
        if kind == "snapshot":
            # 快照只读：到期回收已由业务消息/sweep 显式触发，这里不再改变状态
            snap = {
                "domain_id": arbiter.spec.domain_id,
                "active": sorted(
                    (
                        {
                            "lease_id": l.lease_id, "instance_id": l.instance_id,
                            "slots": list(l.slots), "version": l.version,
                            "state": l.state, "expiry_ms": l.expiry_ms,
                        }
                        for l in arbiter.active.values()
                    ),
                    key=lambda item: item["lease_id"],
                ),
                "slot_owner": {str(k): v for k, v in sorted(arbiter.slot_owner.items())},
                "free_slots": sorted(arbiter.free_slots),
            }
            snap.update({"type": "ack", "clock_ms": now,
                         "queued_seqs": [p.event_seq for p in coordinator.queue]})
            return snap
        if kind == "shutdown":
            return {"type": "ack", "status": "shutdown", "clock_ms": now}
        return {"type": "error", "code": "E_PROTO", "message": f"未知消息类型 {kind!r}"}


class ArbiterTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, domain, instances) -> None:
        super().__init__((domain.host, domain.port), _Handler)
        self.physical_clock = MonotonicClock()
        self.arbiter = DomainArbiter(domain, LogicalPassthroughClock(self.physical_clock))
        self.coordinator = DomainCoordinator(self.arbiter)
        self.instances = {i.instance_id: i for i in instances if i.domain_id == domain.domain_id}
        self.domain_lock = threading.Lock()


def serve(spec_path: str, domain_id: str) -> None:
    domains, instances, _, _ = load_spec(spec_path)
    domain = next(d for d in domains if d.domain_id == domain_id)
    server = ArbiterTCPServer(domain, instances)
    print(f"arbiter {domain_id} listening on {domain.host}:{domain.port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="故障域租约仲裁服务子进程入口")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--domain", required=True)
    args = parser.parse_args(argv)
    serve(args.spec, args.domain)
    return 0


if __name__ == "__main__":
    sys.exit(main())
