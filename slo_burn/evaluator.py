"""本地评估器子进程：租约签发、乐观续约与到期回收的唯一权威。

通信：AF_UNIX 长度前缀 JSON 帧；时钟：time.monotonic_ns 真实单调时钟，
对外统一暴露毫秒。子进程独立持有租约表，父进程崩溃不污染其状态；
子进程重启则所有租约按到期回收（租约不持久化，配额在恢复流程显式回收）。
"""

from __future__ import annotations

import json
import socket
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .errors import LeaseError

GRANT = "grant"
RENEW = "renew"
REVOKE = "revoke"
REAP = "reap"
SNAPSHOT = "snapshot"
SHUTDOWN = "shutdown"


def now_ms() -> int:
    return time.monotonic_ns() // 1_000_000


@dataclass
class Lease:
    lease_id: str
    slot_id: int
    service_id: str
    version: int
    granted_ms: int
    expires_ms: int
    state: str  # active | expired | revoked


class LeaseAuthority:
    """租约权威状态机（运行在评估器子进程内）。"""

    def __init__(self, global_limit: int, per_service_limit: int) -> None:
        self.global_limit = global_limit
        self.per_service_limit = per_service_limit
        self._lock = threading.Lock()
        self._leases: dict[str, Lease] = {}
        self._slots: dict[int, str | None] = {i: None for i in range(global_limit)}
        self._service_versions: dict[tuple[str, int], int] = {}
        self._event_seq = 0

    def _next_event(self) -> int:
        self._event_seq += 1
        return self._event_seq

    def _reap_expired_locked(self, at_ms: int) -> list[Lease]:
        reaped: list[Lease] = []
        for lease in self._leases.values():
            if lease.state == "active" and lease.expires_ms <= at_ms:
                lease.state = "expired"
                if self._slots.get(lease.slot_id) == lease.lease_id:
                    self._slots[lease.slot_id] = None
                reaped.append(lease)
        return reaped

    def handle(self, request: dict) -> dict:
        kind = request["kind"]
        at_ms = now_ms()
        with self._lock:
            if kind == GRANT:
                self._reap_expired_locked(at_ms)
                return self._grant_locked(request, at_ms)
            if kind == RENEW:
                # 续约不做静默回收：状态机自行依据 expires_ms 判定
                return self._renew_locked(request, at_ms)
            if kind == REVOKE:
                return self._revoke_locked(request, at_ms)
            if kind == REAP:
                # 显式 REAP 是到期回收的唯一事件来源，保证裁决日志可留痕
                reaped = self._reap_expired_locked(at_ms)
                return {
                    "ok": True,
                    "event_seq": self._next_event(),
                    "reaped": [self._lease_payload(lease) for lease in reaped],
                    "at_ms": at_ms,
                }
            if kind == SNAPSHOT:
                return {
                    "ok": True,
                    "at_ms": at_ms,
                    "slots": dict(self._slots),
                    "leases": [
                        self._lease_payload(lease) for lease in self._leases.values()
                    ],
                }
            raise LeaseError(f"未知租约请求类型: {kind}")

    def _grant_locked(self, request: dict, at_ms: int) -> dict:
        service_id = request["service_id"]
        ttl_ms = int(request["ttl_ms"])
        if ttl_ms <= 0:
            raise LeaseError("ttl_ms 必须为正整数")
        active_for_service = [
            lease
            for lease in self._leases.values()
            if lease.service_id == service_id and lease.state == "active"
        ]
        if len(active_for_service) >= self.per_service_limit:
            raise LeaseError(
                "每服务租约上限已达，拒绝签发",
                service_id=service_id,
                per_service_limit=self.per_service_limit,
            )
        free_slots = [slot for slot, owner in self._slots.items() if owner is None]
        if not free_slots:
            raise LeaseError(
                "全局并发槽位已满，拒绝签发",
                global_limit=self.global_limit,
            )
        slot_id = min(free_slots)
        version_key = (service_id, slot_id)
        version = self._service_versions.get(version_key, 0) + 1
        self._service_versions[version_key] = version
        lease_id = f"lease-{service_id}-slot{slot_id}-v{version}"
        lease = Lease(
            lease_id=lease_id,
            slot_id=slot_id,
            service_id=service_id,
            version=version,
            granted_ms=at_ms,
            expires_ms=at_ms + ttl_ms,
            state="active",
        )
        self._leases[lease_id] = lease
        self._slots[slot_id] = lease_id
        return {
            "ok": True,
            "event_seq": self._next_event(),
            "lease": self._lease_payload(lease),
            "at_ms": at_ms,
        }

    def _renew_locked(self, request: dict, at_ms: int) -> dict:
        lease_id = request["lease_id"]
        expected_version = int(request["version"])
        ttl_ms = int(request["ttl_ms"])
        lease = self._leases.get(lease_id)
        if lease is None:
            raise LeaseError("租约不存在，续约被拒绝", lease_id=lease_id)
        if lease.version != expected_version:
            raise LeaseError(
                "租约版本号不符，乐观续约被拒绝且不延长到期",
                lease_id=lease_id,
                expected_version=expected_version,
                actual_version=lease.version,
                expires_ms=lease.expires_ms,
            )
        if lease.state != "active":
            raise LeaseError(
                f"租约状态为 {lease.state}，续约被拒绝且不延长到期",
                lease_id=lease_id,
                expires_ms=lease.expires_ms,
            )
        lease.granted_ms = at_ms
        lease.expires_ms = at_ms + ttl_ms
        return {
            "ok": True,
            "event_seq": self._next_event(),
            "lease": self._lease_payload(lease),
            "at_ms": at_ms,
        }

    def _revoke_locked(self, request: dict, at_ms: int) -> dict:
        lease_id = request["lease_id"]
        expected_version = int(request["version"])
        lease = self._leases.get(lease_id)
        if lease is None:
            raise LeaseError("租约不存在，撤销被拒绝", lease_id=lease_id)
        if lease.version != expected_version:
            raise LeaseError(
                "租约版本号不符，撤销被拒绝",
                lease_id=lease_id,
                expected_version=expected_version,
                actual_version=lease.version,
            )
        if lease.state == "active":
            lease.state = "revoked"
            if self._slots.get(lease.slot_id) == lease_id:
                self._slots[lease.slot_id] = None
        return {
            "ok": True,
            "event_seq": self._next_event(),
            "lease": self._lease_payload(lease),
            "at_ms": at_ms,
        }

    @staticmethod
    def _lease_payload(lease: Lease) -> dict:
        return {
            "lease_id": lease.lease_id,
            "slot_id": lease.slot_id,
            "service_id": lease.service_id,
            "version": lease.version,
            "granted_ms": lease.granted_ms,
            "expires_ms": lease.expires_ms,
            "state": lease.state,
        }


def _send_frame(connection: socket.socket, payload: dict) -> None:
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    connection.sendall(struct.pack(">I", len(data)) + data)


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise ConnectionError("评估器连接已关闭")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_frame(connection: socket.socket) -> dict:
    header = _recv_exact(connection, 4)
    (size,) = struct.unpack(">I", header)
    return json.loads(_recv_exact(connection, size).decode("utf-8"))


def serve(socket_path: str, global_limit: int, per_service_limit: int) -> None:
    """子进程入口：在 UNIX socket 上串行处理租约请求。"""
    path = Path(socket_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    authority = LeaseAuthority(global_limit, per_service_limit)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(path))
        server.listen(128)
        while True:
            connection, _ = server.accept()
            try:
                request = _recv_frame(connection)
                if request.get("kind") == SHUTDOWN:
                    _send_frame(connection, {"ok": True, "shutdown": True})
                    connection.close()
                    return
                try:
                    response = authority.handle(request)
                except LeaseError as exc:
                    response = {"ok": False, "error": exc.to_dict()}
                _send_frame(connection, response)
            except (ConnectionError, OSError):
                pass
            finally:
                try:
                    connection.close()
                except OSError:
                    pass


def _main() -> None:
    import sys
    from pathlib import Path

    if __package__ in (None, ""):
        project_root = Path(__file__).resolve().parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from slo_burn.evaluator import serve as _serve
    else:
        _serve = serve
    if len(sys.argv) != 4:
        raise SystemExit("usage: evaluator.py <socket_path> <global_limit> <per_service_limit>")
    _serve(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))


if __name__ == "__main__":
    _main()
