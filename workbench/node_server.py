from __future__ import annotations

import argparse
import json
import os
import socketserver
import threading
import time
from typing import Any

import numpy as np

from .constants import E_CAPACITY_EXCEEDED, E_LEASE_LIMIT_EXCEEDED, E_PROTOCOL_VIOLATION
from .hashing import prefixed_digest
from .watermark import build_watermark, projected_hard_rejected


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _error(request_id: Any, code: str, message: str, **extra: Any) -> dict[str, Any]:
    error = {"code": code, "message": message}
    error.update(extra)
    return {"ok": False, "request_id": request_id, "error": error}


class NodeState:
    def __init__(self, args: argparse.Namespace) -> None:
        self.node_id = args.node_id
        self.port = args.port
        self.seed = int(args.seed)
        self.capacity_slots = int(args.capacity_slots)
        self.lease_limit = int(args.lease_limit)
        self.capacity_version = int(args.capacity_version)
        self.quota_version = int(args.quota_version)
        self.clock_mode = args.clock_mode
        policies = json.loads(args.policies_json or "{}")
        self.policies = {int(key): value for key, value in policies.items()}
        self.residents: dict[str, dict[str, Any]] = {}
        self.rng = np.random.RandomState(self.seed % (2**32 - 1))
        self.last_now_ms = 0
        self.request_count = 0
        self.protocol_violation_count = 0
        self.lock = threading.RLock()

    def watermark(self) -> Any:
        policy = self.policies.get(self.capacity_version)
        if policy is None:
            return None
        return build_watermark(
            capacity_slots=self.capacity_slots,
            capacity_version=self.capacity_version,
            soft_permille=int(policy["soft_permille"]),
            hard_permille=int(policy["hard_permille"]),
        )

    def occupancy(self) -> int:
        return sum(int(item["slots"]) for item in self.residents.values())

    def process_line(self, line: bytes) -> dict[str, Any]:
        try:
            request = json.loads(line.decode("utf-8"))
        except Exception as exc:  # malformed JSON is a protocol violation
            with self.lock:
                self.protocol_violation_count += 1
            return _error(None, E_PROTOCOL_VIOLATION, f"invalid JSON line: {exc}")

        request_id = request.get("request_id") if isinstance(request, dict) else None
        started = time.monotonic_ns()
        with self.lock:
            try:
                if not isinstance(request, dict):
                    response = _error(request_id, E_PROTOCOL_VIOLATION, "request must be a JSON object")
                    self.protocol_violation_count += 1
                else:
                    response = self._handle_locked(request)
            except Exception as exc:
                self.protocol_violation_count += 1
                response = _error(request_id, E_PROTOCOL_VIOLATION, f"handler failure: {type(exc).__name__}: {exc}")

            if self.clock_mode == "system":
                response["elapsed_ms"] = max(0, (time.monotonic_ns() - started) // 1_000_000)
                response["clock_source"] = "system.monotonic_ns"
            else:
                response.setdefault("elapsed_ms", self._seeded_elapsed_locked(request if isinstance(request, dict) else {}))
                response["clock_source"] = "seeded-monotonic-v1"
            response.setdefault("node_id", self.node_id)
            response.setdefault("quota_version", self.quota_version)
            response.setdefault("capacity_version", self.capacity_version)
            return response

    def _seeded_elapsed_locked(self, request: dict[str, Any]) -> int:
        self.request_count += 1
        jitter = int(self.rng.randint(0, 3))
        slots = request.get("slots", 1)
        if not _is_int(slots) or slots <= 0:
            slots = 1
        return 1 + jitter + max(0, (int(slots) - 1) // 4)

    def _common_meta(self, request_id: Any) -> dict[str, Any]:
        return {
            "ok": True,
            "request_id": request_id,
            "node_id": self.node_id,
            "quota_version": self.quota_version,
            "capacity_version": self.capacity_version,
        }

    def _handle_locked(self, request: dict[str, Any]) -> dict[str, Any]:
        request_id = request.get("request_id")
        req_type = request.get("type")
        if req_type not in {"write", "read", "release", "quota_change"}:
            self.protocol_violation_count += 1
            return _error(
                request_id,
                E_PROTOCOL_VIOLATION,
                "type must be write, read, release, or quota_change",
                allowed_types=["write", "read", "release", "quota_change"],
            )

        now_ms = request.get("now_ms")
        if now_ms is not None:
            if not _is_int(now_ms) or now_ms < 0:
                self.protocol_violation_count += 1
                return _error(request_id, E_PROTOCOL_VIOLATION, "now_ms must be a non-negative integer")
            if now_ms < self.last_now_ms:
                self.protocol_violation_count += 1
                return _error(
                    request_id,
                    E_PROTOCOL_VIOLATION,
                    "monotonic node clock cannot move backwards",
                    last_now_ms=self.last_now_ms,
                    now_ms=now_ms,
                )
            self.last_now_ms = now_ms

        if req_type == "write":
            return self._write(request)
        if req_type == "read":
            return self._read(request)
        if req_type == "release":
            return self._release(request)
        return self._quota_change(request)

    def _write(self, request: dict[str, Any]) -> dict[str, Any]:
        request_id = request.get("request_id")
        key = request.get("key")
        slots = request.get("slots", 1)
        ttl_ms = request.get("ttl_ms")
        now_ms = request.get("now_ms", 0)
        priority = request.get("priority", 0)
        stage_id = request.get("stage_id")

        if not isinstance(key, str) or not key:
            self.protocol_violation_count += 1
            return _error(request_id, E_PROTOCOL_VIOLATION, "write requires non-empty string key")
        if not _is_int(slots) or slots <= 0:
            self.protocol_violation_count += 1
            return _error(request_id, E_PROTOCOL_VIOLATION, "slots must be a positive integer")
        if not _is_int(ttl_ms) or ttl_ms <= 0:
            self.protocol_violation_count += 1
            return _error(request_id, E_PROTOCOL_VIOLATION, "ttl_ms must be a positive integer")
        if not _is_int(now_ms) or now_ms < 0:
            self.protocol_violation_count += 1
            return _error(request_id, E_PROTOCOL_VIOLATION, "now_ms must be a non-negative integer")
        if not _is_int(priority):
            self.protocol_violation_count += 1
            return _error(request_id, E_PROTOCOL_VIOLATION, "priority must be an integer")
        if "content" not in request and not isinstance(request.get("digest"), str):
            self.protocol_violation_count += 1
            return _error(request_id, E_PROTOCOL_VIOLATION, "write requires content or digest")

        old = self.residents.get(key)
        old_slots = int(old["slots"]) if old else 0
        new_occupancy = self.occupancy() - old_slots + slots
        if old is None and len(self.residents) >= self.lease_limit:
            return _error(
                request_id,
                E_LEASE_LIMIT_EXCEEDED,
                "node lease limit reached",
                active_leases=len(self.residents),
                lease_limit=self.lease_limit,
            )
        if new_occupancy > self.capacity_slots:
            return _error(
                request_id,
                E_CAPACITY_EXCEEDED,
                "integer capacity slots exhausted",
                occupancy=self.occupancy(),
                requested_slots=slots,
                capacity_slots=self.capacity_slots,
            )
        watermark = self.watermark()
        if watermark is not None and projected_hard_rejected(self.occupancy() - old_slots, slots, watermark.hard_limit):
            return _error(
                request_id,
                E_CAPACITY_EXCEEDED,
                "projected occupancy exceeds hard watermark",
                occupancy=self.occupancy(),
                requested_slots=slots,
                hard_limit=watermark.hard_limit,
                capacity_version=self.capacity_version,
            )

        if "content" in request:
            digest = prefixed_digest(request["content"])
        else:
            digest = str(request["digest"])
        overwritten_digest = old["content_digest"] if old else None
        generation = int(old.get("lease_generation", 0)) + 1 if old else 1
        lease_id = f"lease:{self.node_id}:{key}:g{generation}"
        record = {
            "node_id": self.node_id,
            "key": key,
            "slots": slots,
            "priority": priority,
            "stage_id": stage_id if isinstance(stage_id, str) else None,
            "content": request.get("content"),
            "content_digest": digest,
            "lease_id": lease_id,
            "lease_generation": generation,
            "issued_ms": now_ms,
            "expires_ms": now_ms + ttl_ms,
            "ttl_ms": ttl_ms,
            "capacity_version": self.capacity_version,
            "quota_version": self.quota_version,
            "write_request_id": request_id,
        }
        self.residents[key] = record
        meta = self._common_meta(request_id)
        meta.update({
            "content_digest": digest,
            "result": {
                "key": key,
                "created": old is None,
                "overwritten": old is not None,
                "overwritten_digest": overwritten_digest,
                "slots": slots,
                "lease_id": lease_id,
                "issued_ms": record["issued_ms"],
                "expires_ms": record["expires_ms"],
                "occupancy": self.occupancy(),
                "active_leases": len(self.residents),
                "lease_limit": self.lease_limit,
                "hard_limit": watermark.hard_limit if watermark else None,
            },
        })
        return meta

    def _read(self, request: dict[str, Any]) -> dict[str, Any]:
        request_id = request.get("request_id")
        key = request.get("key")
        now_ms = request.get("now_ms", self.last_now_ms)
        renew_ttl = request.get("renew_ttl_ms")
        if not isinstance(key, str) or not key:
            self.protocol_violation_count += 1
            return _error(request_id, E_PROTOCOL_VIOLATION, "read requires key")
        if renew_ttl is not None and (not _is_int(renew_ttl) or renew_ttl <= 0):
            self.protocol_violation_count += 1
            return _error(request_id, E_PROTOCOL_VIOLATION, "renew_ttl_ms must be positive")

        record = self.residents.get(key)
        meta = self._common_meta(request_id)
        if record is None:
            meta.update({"found": False, "content_digest": None, "result": {"key": key, "found": False}})
            return meta

        renewed = False
        renewal_reason = None
        if renew_ttl is not None:
            if now_ms < int(record["expires_ms"]):
                record["expires_ms"] = int(now_ms) + int(renew_ttl)
                record["ttl_ms"] = int(renew_ttl)
                renewed = True
            else:
                renewal_reason = "lease_expired"

        meta.update({
            "found": True,
            "content_digest": record["content_digest"],
            "result": {
                "key": key,
                "found": True,
                "renewed": renewed,
                "renewal_reason": renewal_reason,
                "lease_id": record["lease_id"],
                "expires_ms": record["expires_ms"],
                "content": record.get("content"),
            },
        })
        return meta

    def _release(self, request: dict[str, Any]) -> dict[str, Any]:
        request_id = request.get("request_id")
        key = request.get("key")
        lease_id = request.get("lease_id")
        if key is None and lease_id is None:
            self.protocol_violation_count += 1
            return _error(request_id, E_PROTOCOL_VIOLATION, "release requires key or lease_id")

        record = None
        if isinstance(key, str) and key in self.residents:
            record = self.residents[key]
            if lease_id is not None and lease_id != record["lease_id"]:
                self.protocol_violation_count += 1
                return _error(request_id, E_PROTOCOL_VIOLATION, "lease_id does not match resident lease")
        elif isinstance(lease_id, str):
            for candidate in self.residents.values():
                if candidate["lease_id"] == lease_id:
                    record = candidate
                    key = candidate["key"]
                    break

        meta = self._common_meta(request_id)
        if record is None:
            meta.update({
                "found": False,
                "released": False,
                "result": {"key": key, "lease_id": lease_id, "released": False},
            })
            return meta

        released_slots = int(record["slots"])
        released_digest = record["content_digest"]
        del self.residents[key]
        meta.update({
            "found": True,
            "released": True,
            "content_digest": released_digest,
            "result": {
                "key": key,
                "lease_id": record["lease_id"],
                "released_slots": released_slots,
                "released_digest": released_digest,
                "occupancy": self.occupancy(),
                "active_leases": len(self.residents),
            },
        })
        return meta

    def _quota_change(self, request: dict[str, Any]) -> dict[str, Any]:
        request_id = request.get("request_id")
        quota_version = request.get("quota_version")
        lease_limit = request.get("lease_limit")
        capacity_version = request.get("capacity_version")
        if quota_version is None and lease_limit is None and capacity_version is None:
            self.protocol_violation_count += 1
            return _error(request_id, E_PROTOCOL_VIOLATION, "quota_change requires a version or limit field")
        if quota_version is not None and (not _is_int(quota_version) or quota_version <= 0):
            self.protocol_violation_count += 1
            return _error(request_id, E_PROTOCOL_VIOLATION, "quota_version must be positive")
        if lease_limit is not None and (not _is_int(lease_limit) or lease_limit <= 0):
            self.protocol_violation_count += 1
            return _error(request_id, E_PROTOCOL_VIOLATION, "lease_limit must be positive")
        if capacity_version is not None:
            if not _is_int(capacity_version) or capacity_version <= 0:
                self.protocol_violation_count += 1
                return _error(request_id, E_PROTOCOL_VIOLATION, "capacity_version must be positive")
            if capacity_version not in self.policies:
                self.protocol_violation_count += 1
                return _error(request_id, E_PROTOCOL_VIOLATION, "unknown capacity_version")

        before = {
            "quota_version": self.quota_version,
            "lease_limit": self.lease_limit,
            "capacity_version": self.capacity_version,
        }
        if quota_version is not None:
            self.quota_version = quota_version
        if lease_limit is not None:
            self.lease_limit = lease_limit
        if capacity_version is not None:
            self.capacity_version = capacity_version
        after = {
            "quota_version": self.quota_version,
            "lease_limit": self.lease_limit,
            "capacity_version": self.capacity_version,
        }
        meta = self._common_meta(request_id)
        meta["result"] = {"before": before, "after": after}
        return meta


class JsonLineHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        state: NodeState = self.server.state  # type: ignore[attr-defined]
        while True:
            line = self.rfile.readline()
            if not line:
                return
            response = state.process_line(line)
            self.wfile.write((json.dumps(response, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"))
            self.wfile.flush()


class ThreadedTcpServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], handler: type[JsonLineHandler], state: NodeState) -> None:
        super().__init__(address, handler)
        self.state = state


def watch_parent(parent_pid: int | None) -> None:
    if not parent_pid:
        return

    def alive_windows(pid: int) -> bool:
        import ctypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            return False
        exit_code = ctypes.c_ulong()
        kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        kernel32.CloseHandle(handle)
        return exit_code.value == still_active

    def alive_posix(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def loop() -> None:
        while True:
            alive = alive_windows(parent_pid) if os.name == "nt" else alive_posix(parent_pid)
            if not alive:
                os._exit(0)
            time.sleep(0.5)

    thread = threading.Thread(target=loop, name="parent-watchdog", daemon=True)
    thread.start()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic cache node JSON-lines TCP server")
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--capacity-slots", required=True, type=int)
    parser.add_argument("--lease-limit", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--capacity-version", required=True, type=int)
    parser.add_argument("--quota-version", default=1, type=int)
    parser.add_argument("--clock-mode", default="seeded", choices=["seeded", "system"])
    parser.add_argument("--policies-json", default="{}")
    parser.add_argument("--parent-pid", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    watch_parent(args.parent_pid)
    state = NodeState(args)
    server = ThreadedTcpServer(("127.0.0.1", args.port), JsonLineHandler, state)
    actual_port = server.server_address[1]
    print(json.dumps({"type": "READY", "node_id": args.node_id, "port": actual_port}), flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
