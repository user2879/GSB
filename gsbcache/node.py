"""A local loopback cache node process."""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .content import NodeStore
from .errors import AppError, PortError, ProtocolError, ReplyError, SpecError
from .protocol import PROTOCOL_VERSION, encode_line, read_line
from .spec import load_spec
from .util import is_int, monotonic_ns, now_ms, positive_int


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return True
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
            kernel32.GetExitCodeProcess.restype = wintypes.BOOL
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel32.CloseHandle.restype = wintypes.BOOL
            handle = kernel32.OpenProcess(0x1000, False, int(pid))
            if not handle:
                return False
            try:
                code = wintypes.DWORD()
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return False
                return code.value == 259  # STILL_ACTIVE
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return True
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _watch_parent(server: "CacheNode", parent_pid: int) -> None:
    while not server.stop_event.wait(0.2):
        if not _process_alive(parent_pid):
            server.stop_event.set()
            if server._listener is not None:
                try:
                    server._listener.close()
                except OSError:
                    pass
            return


class CacheNode:
    def __init__(self, spec, node, data_dir: str | Path) -> None:
        self.spec = spec
        self.node = node
        self.store = NodeStore(Path(data_dir), spec, node)
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.leases: dict[str, dict[str, Any]] = {}
        self.capacity_used = 0
        self.reaped: list[dict[str, Any]] = []
        self._listener: socket.socket | None = None
        self._executor = ThreadPoolExecutor(max_workers=32, thread_name_prefix=f"node-{node.node_id}")
        self._reaper = threading.Thread(target=self._reap_loop, name=f"reaper-{node.node_id}", daemon=True)

    def _capacity_used_locked(self) -> int:
        return sum(len(lease["artifact_keys"]) for lease in self.leases.values())

    def _reap_locked(self) -> list[dict[str, Any]]:
        now = monotonic_ns()
        expired: list[dict[str, Any]] = []
        for lease_id, lease in list(self.leases.items()):
            if now >= lease["deadline_mono_ns"]:
                del self.leases[lease_id]
                event = {
                    "event": "node_lease_expired",
                    "lease_id": lease_id,
                    "artifact_keys": list(lease["artifact_keys"]),
                    "expired_at_ms": now_ms(),
                }
                self.reaped.append(event)
                expired.append(event)
        self.capacity_used = self._capacity_used_locked()
        return expired

    def _reap_loop(self) -> None:
        while not self.stop_event.wait(0.005):
            with self.lock:
                self._reap_locked()

    def _inventory_locked(self) -> dict[str, str]:
        return self.store.inventory()

    def _reusable_locked(self, key: str, digests: dict[str, str]) -> dict[str, Any]:
        return self.store.reusability(key, digests)

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        op = request.get("type")
        if request.get("protocol", PROTOCOL_VERSION) != PROTOCOL_VERSION:
            return self._error("E_PROTOCOL", "protocol version mismatch", expected=PROTOCOL_VERSION)
        if op == "hello":
            return {
                "ok": True,
                "type": "hello_ack",
                "protocol": PROTOCOL_VERSION,
                "node_id": self.node.node_id,
                "seed": self.node.seed,
                "capacity_slots": self.node.capacity_slots,
                "lease_limit": self.node.lease_limit,
            }
        if op == "inventory":
            with self.lock:
                self._reap_locked()
                digests = self._inventory_locked()
                reusable = [key for key in self.spec.all_artifact_keys if self._reusable_locked(key, digests)["reusable"]]
                return {
                    "ok": True,
                    "type": "inventory",
                    "protocol": PROTOCOL_VERSION,
                    "node_id": self.node.node_id,
                    "digests": digests,
                    "reusable_keys": reusable,
                }
        if op == "status":
            with self.lock:
                self._reap_locked()
                return {
                    "ok": True,
                    "type": "status",
                    "protocol": PROTOCOL_VERSION,
                    "node_id": self.node.node_id,
                    "capacity_slots": self.node.capacity_slots,
                    "capacity_used": self.capacity_used,
                    "lease_limit": self.node.lease_limit,
                    "active_lease_count": len(self.leases),
                    "leases": {lease_id: self._public_lease(lease) for lease_id, lease in self.leases.items()},
                    "reaped_events": list(self.reaped[-20:]),
                }
        if op == "lease":
            return self._lease(request)
        if op == "renew":
            return self._renew(request)
        if op == "release":
            return self._release(request)
        if op == "shutdown":
            self.stop_event.set()
            return {"ok": True, "type": "shutdown_ack", "node_id": self.node.node_id}
        return self._error("E_PROTOCOL", "unsupported request type", field="type", actual=op)

    @staticmethod
    def _public_lease(lease: dict[str, Any]) -> dict[str, Any]:
        return {
            "artifact_keys": list(lease["artifact_keys"]),
            "expires_at_ms": lease["expires_at_ms"],
            "digests": dict(lease["digests"]),
        }

    def _lease(self, request: dict[str, Any]) -> dict[str, Any]:
        keys = request.get("artifact_keys")
        lease_ms = request.get("lease_ms")
        request_id = request.get("request_id")
        sequence = request.get("sequence")
        if not isinstance(keys, list) or not keys or any(not isinstance(key, str) or not key for key in keys):
            raise ReplyError("artifact_keys must be a non-empty array", node_id=self.node.node_id, field="artifact_keys")
        if len(set(keys)) != len(keys):
            raise ReplyError("artifact_keys must be unique", node_id=self.node.node_id, field="artifact_keys")
        if not positive_int(lease_ms):
            raise ReplyError("lease_ms must be a positive integer", node_id=self.node.node_id, field="lease_ms")
        if not isinstance(request_id, str) or not request_id:
            raise ReplyError("request_id must be a non-empty string", node_id=self.node.node_id, field="request_id")
        if not positive_int(sequence):
            raise ReplyError("sequence must be a positive integer", node_id=self.node.node_id, field="sequence")
        with self.lock:
            self._reap_locked()
            unknown = [key for key in keys if key not in self.spec.artifacts]
            if unknown:
                raise ReplyError(
                    "unknown artifact key",
                    node_id=self.node.node_id,
                    artifact_key=unknown[0],
                    field="artifact_keys",
                )
            active_keys = {key for lease in self.leases.values() for key in lease["artifact_keys"]}
            conflicts = sorted(set(keys) & active_keys)
            if conflicts:
                raise ReplyError(
                    "artifact key already leased on this node",
                    node_id=self.node.node_id,
                    artifact_key=conflicts[0],
                    field="artifact_keys",
                    reason="node_key_conflict",
                )
            if len(self.leases) + 1 > self.node.lease_limit:
                raise ReplyError(
                    "per-node lease limit exhausted",
                    node_id=self.node.node_id,
                    field="lease_limit",
                    reason="lease_limit_exhausted",
                    lease_limit=self.node.lease_limit,
                    active_leases=len(self.leases),
                )
            if self.capacity_used + len(keys) > self.node.capacity_slots:
                raise ReplyError(
                    "node capacity slots exhausted",
                    node_id=self.node.node_id,
                    field="capacity_slots",
                    reason="capacity_exhausted",
                    capacity_slots=self.node.capacity_slots,
                    capacity_used=self.capacity_used,
                    requested=len(keys),
                )
            digests = self._inventory_locked()
            reusable = [key for key in keys if self._reusable_locked(key, digests)["reusable"]]
            if len(reusable) != len(keys):
                bad = next(key for key in keys if key not in reusable)
                detail = self._reusable_locked(bad, digests)
                return self._error(
                    "E_REUSE_MISMATCH",
                    "artifact is not reusable on this node",
                    node_id=self.node.node_id,
                    artifact_key=bad,
                    expected_digest=detail.get("expected_digest"),
                    actual_digest=detail.get("actual_digest"),
                    reason=detail.get("reason"),
                    bad_references=detail.get("bad_references", []),
                )
            lease_id = "lease_" + uuid.uuid4().hex
            deadline = monotonic_ns() + int(lease_ms) * 1_000_000
            expires_at = now_ms() + int(lease_ms)
            lease = {
                "lease_id": lease_id,
                "request_id": request_id,
                "artifact_keys": list(keys),
                "digests": {key: digests[key] for key in keys},
                "expires_at_ms": expires_at,
                "deadline_mono_ns": deadline,
            }
            self.leases[lease_id] = lease
            self.capacity_used = self._capacity_used_locked()
            return {
                "ok": True,
                "type": "lease_granted",
                "protocol": PROTOCOL_VERSION,
                "lease_id": lease_id,
                "request_id": request_id,
                "sequence": sequence,
                "node_id": self.node.node_id,
                "granted_key": keys[0],
                "granted_keys": list(keys),
                "expires_at_ms": expires_at,
                "expires_monotonic_ms": deadline // 1_000_000,
                "digests": {key: digests[key] for key in keys},
                "reusable_keys": reusable,
            }

    def _renew(self, request: dict[str, Any]) -> dict[str, Any]:
        lease_id = request.get("lease_id")
        extend_ms = request.get("extend_ms")
        if not isinstance(lease_id, str) or not lease_id:
            raise ReplyError("lease_id must be a non-empty string", node_id=self.node.node_id, field="lease_id")
        if not positive_int(extend_ms):
            raise ReplyError("extend_ms must be a positive integer", node_id=self.node.node_id, field="extend_ms")
        with self.lock:
            self._reap_locked()
            lease = self.leases.get(lease_id)
            if lease is None:
                return self._error(
                    "E_REPLY",
                    "lease does not exist or already expired",
                    node_id=self.node.node_id,
                    lease_id=lease_id,
                    field="lease_id",
                )
            lease["deadline_mono_ns"] = monotonic_ns() + int(extend_ms) * 1_000_000
            lease["expires_at_ms"] = now_ms() + int(extend_ms)
            return {
                "ok": True,
                "type": "lease_renewed",
                "protocol": PROTOCOL_VERSION,
                "lease_id": lease_id,
                "node_id": self.node.node_id,
                "expires_at_ms": lease["expires_at_ms"],
                "expires_monotonic_ms": lease["deadline_mono_ns"] // 1_000_000,
            }

    def _release(self, request: dict[str, Any]) -> dict[str, Any]:
        lease_id = request.get("lease_id")
        if not isinstance(lease_id, str) or not lease_id:
            raise ReplyError("lease_id must be a non-empty string", node_id=self.node.node_id, field="lease_id")
        with self.lock:
            self._reap_locked()
            existed = self.leases.pop(lease_id, None) is not None
            self.capacity_used = self._capacity_used_locked()
            return {
                "ok": True,
                "type": "lease_released",
                "protocol": PROTOCOL_VERSION,
                "lease_id": lease_id,
                "node_id": self.node.node_id,
                "released": existed,
                "reason": request.get("reason", "client_release"),
            }

    @staticmethod
    def _error(code: str, message: str, **details: Any) -> dict[str, Any]:
        payload = {"ok": False, "code": code, "message": message}
        payload.update(details)
        return payload

    def serve(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind((self.node.host, self.node.port))
            listener.listen(128)
        except OSError as exc:
            listener.close()
            raise PortError(
                "failed to bind loopback port",
                node_id=self.node.node_id,
                host=self.node.host,
                port=self.node.port,
                error=str(exc),
            ) from exc
        self._listener = listener
        listener.settimeout(0.2)
        self._reaper.start()
        print(
            json.dumps(
                {
                    "type": "ready",
                    "node_id": self.node.node_id,
                    "host": self.node.host,
                    "port": self.node.port,
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
        try:
            while not self.stop_event.is_set():
                try:
                    conn, _ = listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self.stop_event.is_set():
                        break
                    raise
                self._executor.submit(self._serve_connection, conn)
        finally:
            self.stop_event.set()
            try:
                listener.close()
            except OSError:
                pass
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._reaper.join(timeout=1.0)

    def _serve_connection(self, conn: socket.socket) -> None:
        with conn:
            conn.settimeout(3.0)
            while not self.stop_event.is_set():
                try:
                    request = read_line(conn)
                except socket.timeout:
                    break
                except ProtocolError as exc:
                    try:
                        conn.sendall(encode_line({"ok": False, "code": "E_PROTOCOL", "message": exc.message, **exc.details}))
                    except OSError:
                        pass
                    break
                try:
                    response = self.handle(request)
                except AppError as exc:
                    response = {"ok": False, "code": exc.code, "message": exc.message, **exc.details}
                except Exception as exc:  # Defensive protocol boundary.
                    response = {"ok": False, "code": "E_REPLY", "message": "node handler failed", "error": repr(exc)}
                try:
                    conn.sendall(encode_line(response))
                except OSError:
                    break
                if request.get("type") == "shutdown":
                    break


def run_node(spec_path: str, node_id: str, data_dir: str, parent_pid: int | None = None) -> int:
    spec = load_spec(spec_path)
    if node_id not in spec.node_by_id:
        raise SpecError("unknown node_id", node_id=node_id, field="node_id")
    node = spec.node_by_id[node_id]
    server = CacheNode(spec, node, data_dir)

    def _stop(_signum=None, _frame=None):
        server.stop_event.set()
        if server._listener is not None:
            try:
                server._listener.close()
            except OSError:
                pass

    try:
        signal.signal(signal.SIGTERM, _stop)
        signal.signal(signal.SIGINT, _stop)
    except (ValueError, OSError):
        pass
    if parent_pid:
        threading.Thread(target=_watch_parent, args=(server, int(parent_pid)), daemon=True).start()
    try:
        server.serve()
        return 0
    except PortError as exc:
        print(json.dumps(exc.to_dict(), separators=(",", ":")), flush=True)
        return exc.exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one local JSON-lines cache node")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--parent-pid", type=int, default=None)
    args = parser.parse_args(argv)
    try:
        return run_node(args.spec, args.node_id, args.data_dir, args.parent_pid)
    except AppError as exc:
        print(json.dumps(exc.to_dict(), separators=(",", ":")), flush=True)
        return exc.exit_code
    except Exception as exc:
        print(
            json.dumps(
                {"ok": False, "code": "E_INTERNAL", "message": "node startup failed", "error": repr(exc)},
                separators=(",", ":"),
            ),
            flush=True,
        )
        return 11


if __name__ == "__main__":
    raise SystemExit(main())
