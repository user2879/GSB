from __future__ import annotations

import argparse
import base64
import json
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .content import shard_digest
from .errors import FieldError
from .ledger import Ledger
from .protocol import PROTOCOL_VERSION, encode_line


READY_PREFIX = "@@CACHEFLOW_READY@@"
ERROR_PREFIX = "@@CACHEFLOW_ERROR@@"


class NodeService:
    def __init__(
        self,
        *,
        node_id: str,
        host: str,
        port: int,
        seed: str = "node-seed",
        ledger_path: str | Path | None = None,
    ) -> None:
        self.node_id = node_id
        self.host = host
        self.port = port
        self.seed = seed
        self.ledger_path = None if ledger_path is None else Path(ledger_path)
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self.leases: dict[str, dict[str, Any]] = {}
        self.shards: dict[tuple[str, int], bytes] = {}
        self._recover()

    def _recover(self) -> None:
        if self.ledger_path is None:
            return
        rows = Ledger.read_node_recovery(self.ledger_path, self.node_id)
        ttl_by_lease = self._ttls_from_events()
        now = time.monotonic()
        for lease_id, row in rows.items():
            ttl_ms = ttl_by_lease.get(lease_id, 600_000)
            self.leases[lease_id] = {
                "state": row["state"],
                "generation": int(row["generation"]),
                "artifact_key": row["artifact_key"],
                "quota_version": int(row["quota_version"]),
                "ttl_ms": ttl_ms,
                "expires_at": now + ttl_ms / 1000.0,
                "recovered": True,
            }

    def _ttls_from_events(self) -> dict[str, int]:
        if self.ledger_path is None or not self.ledger_path.exists():
            return {}
        try:
            conn = Ledger.read_only_connection(self.ledger_path)
        except Exception:
            return {}
        try:
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "events" not in tables:
                return {}
            result: dict[str, int] = {}
            for row in conn.execute("SELECT event_json FROM events WHERE event_type='grant'"):
                event = json.loads(row["event_json"])
                result[str(event["lease_id"])] = int(event.get("ttl_ms", 600_000))
            return result
        except Exception:
            return {}
        finally:
            conn.close()

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        request_id = request.get("request_id")
        try:
            if request.get("type") == "hello":
                return {
                    "type": "hello_ack",
                    "protocol": PROTOCOL_VERSION,
                    "node_id": self.node_id,
                    "request_id": request_id,
                }
            if request.get("type") != "request":
                raise FieldError("expected type=request", field="type")
            op = request.get("op")
            if not isinstance(op, str) or not op:
                raise FieldError("op is required", field="op")
            handler = getattr(self, f"op_{op}", None)
            if handler is None:
                return self._error(op, "E_OP", f"unknown op: {op}", request_id)
            response = handler(request)
            response.setdefault("ok", True)
            response["op"] = op
            response["request_id"] = request_id
            return response
        except FieldError as exc:
            return {
                "ok": False,
                "op": request.get("op"),
                "request_id": request_id,
                "error": {"code": "E_FIELD", "message": exc.message, "field": exc.field},
            }
        except Exception as exc:  # pragma: no cover - defensive node boundary
            return {
                "ok": False,
                "op": request.get("op"),
                "request_id": request_id,
                "error": {"code": "E_NODE", "message": repr(exc)},
            }

    @staticmethod
    def _error(op: str, code: str, message: str, request_id: Any, **details: Any) -> dict[str, Any]:
        error: dict[str, Any] = {"code": code, "message": message}
        error.update(details)
        return {"ok": False, "op": op, "request_id": request_id, "error": error}

    @staticmethod
    def _require(request: dict[str, Any], field: str) -> Any:
        if field not in request or request[field] is None:
            raise FieldError(f"missing field: {field}", field=field)
        return request[field]

    def op_status(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            return {
                "node_id": self.node_id,
                "leases": [
                    {"lease_id": lease_id, **{k: v for k, v in row.items() if k != "expires_at"}}
                    for lease_id, row in sorted(self.leases.items())
                ],
            }

    def op_grant(self, request: dict[str, Any]) -> dict[str, Any]:
        lease_id = str(self._require(request, "lease_id"))
        ttl_ms = int(request.get("ttl_ms", 600_000))
        if ttl_ms <= 0:
            raise FieldError("ttl_ms must be positive", field="ttl_ms")
        if request.get("block_forever"):
            event = threading.Event()
            event.wait(float(request.get("block_timeout_seconds", 30.0)))
        with self._condition:
            if lease_id in self.leases and self.leases[lease_id]["state"] not in {"released", "expired"}:
                current = int(self.leases[lease_id]["generation"])
                return self._error(
                    "grant", "E_GEN", "active lease already exists", request.get("request_id"),
                    current_generation=current,
                )
            self.leases[lease_id] = {
                "state": "granted",
                "generation": 1,
                "artifact_key": str(request.get("artifact_key", "")),
                "quota_version": int(request.get("quota_version", 1)),
                "ttl_ms": ttl_ms,
                "expires_at": time.monotonic() + ttl_ms / 1000.0,
            }
            self._condition.notify_all()
            return {"lease_id": lease_id, "generation": 1, "state": "granted"}

    def _active_lease(self, request: dict[str, Any], op: str) -> tuple[str, dict[str, Any]] | dict[str, Any]:
        lease_id = str(self._require(request, "lease_id"))
        with self._lock:
            lease = self.leases.get(lease_id)
            if lease is None:
                return self._error(op, "E_GEN", "unknown lease", request.get("request_id"))
        return lease_id, lease

    def op_write(self, request: dict[str, Any]) -> dict[str, Any]:
        lease_id = str(self._require(request, "lease_id"))
        shard_index = int(self._require(request, "shard_index"))
        content_b64 = str(self._require(request, "content_b64"))
        artifact_key = str(self._require(request, "artifact_key"))
        try:
            content = base64.b64decode(content_b64, validate=True)
        except Exception as exc:
            raise FieldError("content_b64 is invalid base64", field="content_b64") from exc
        digest = shard_digest(self.node_id, artifact_key, shard_index, content)
        with self._lock:
            if lease_id not in self.leases:
                return self._error("write", "E_GEN", "unknown lease", request.get("request_id"))
            self.shards[(lease_id, shard_index)] = content
        return {"lease_id": lease_id, "shard_index": shard_index, "digest": digest, "bytes": len(content)}

    def op_read(self, request: dict[str, Any]) -> dict[str, Any]:
        lease_id = str(self._require(request, "lease_id"))
        shard_index = int(self._require(request, "shard_index"))
        with self._lock:
            content = self.shards.get((lease_id, shard_index))
            if content is None:
                return self._error("read", "E_FIELD", "shard is not present on node", request.get("request_id"))
            artifact_key = str(request.get("artifact_key", self.leases.get(lease_id, {}).get("artifact_key", "")))
        digest = shard_digest(self.node_id, artifact_key, shard_index, content)
        return {
            "lease_id": lease_id,
            "shard_index": shard_index,
            "digest": digest,
            "content_b64": base64.b64encode(content).decode("ascii"),
        }

    def _tagged_transition(self, request: dict[str, Any], op: str, new_state: str) -> dict[str, Any]:
        lease_id = str(self._require(request, "lease_id"))
        carried = int(self._require(request, "generation"))
        with self._condition:
            lease = self.leases.get(lease_id)
            if lease is None:
                return self._error(op, "E_GEN", "unknown lease", request.get("request_id"), current_generation=0)
            current = int(lease["generation"])
            if carried != current:
                return self._error(
                    op,
                    "E_STALE",
                    f"carried label {carried} != current label {current}",
                    request.get("request_id"),
                    current_generation=current,
                )
            current += 1
            lease["generation"] = current
            lease["state"] = new_state
            if op == "renew":
                ttl_ms = int(request.get("ttl_ms", lease.get("ttl_ms", 600_000)))
                lease["ttl_ms"] = ttl_ms
                lease["expires_at"] = time.monotonic() + ttl_ms / 1000.0
            self._condition.notify_all()
            return {"lease_id": lease_id, "generation": current, "state": new_state}

    def op_renew(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._tagged_transition(request, "renew", "renewed")

    def op_release(self, request: dict[str, Any]) -> dict[str, Any]:
        response = self._tagged_transition(request, "release", "released")
        if response.get("ok", True) and "error" not in response:
            lease_id = response["lease_id"]
            with self._lock:
                for key in [key for key in self.shards if key[0] == lease_id]:
                    self.shards.pop(key, None)
        return response

    def op_expire(self, request: dict[str, Any]) -> dict[str, Any]:
        lease_id = str(self._require(request, "lease_id"))
        wait = bool(request.get("wait", True))
        with self._condition:
            lease = self.leases.get(lease_id)
            if lease is None:
                return self._error("expire", "E_GEN", "unknown lease", request.get("request_id"), current_generation=0)
            carried = request.get("generation")
            current = int(lease["generation"])
            if carried is not None and int(carried) != current:
                return self._error("expire", "E_STALE", "stale expiry generation", request.get("request_id"), current_generation=current)
            while time.monotonic() < float(lease["expires_at"]):
                if not wait:
                    return {"lease_id": lease_id, "reaped": False, "generation": current, "state": lease["state"]}
                timeout = max(0.001, float(lease["expires_at"]) - time.monotonic())
                self._condition.wait(timeout=min(timeout, 0.25))
                if lease_id not in self.leases:
                    return self._error("expire", "E_GEN", "lease vanished", request.get("request_id"))
            if lease["state"] not in {"granted", "renewed"}:
                return {"lease_id": lease_id, "reaped": False, "generation": current, "state": lease["state"]}
            current += 1
            lease["generation"] = current
            lease["state"] = "expired"
            return {"lease_id": lease_id, "reaped": True, "generation": current, "state": "expired"}


def _handle_connection(conn: socket.socket, address: tuple[str, int], service: NodeService) -> None:
    buffer = b""
    conn.settimeout(30.0)
    try:
        while True:
            while b"\n" not in buffer:
                chunk = conn.recv(65536)
                if not chunk:
                    return
                buffer += chunk
            line, buffer = buffer.split(b"\n", 1)
            try:
                request = json.loads(line.decode("utf-8"))
                if not isinstance(request, dict):
                    raise ValueError("not object")
                response = service.handle(request)
            except Exception as exc:
                response = {"ok": False, "error": {"code": "E_FIELD", "message": repr(exc)}}
            conn.sendall(encode_line(response))
    except (socket.timeout, OSError):
        return
    finally:
        try:
            conn.close()
        except OSError:
            pass


def serve(service: NodeService) -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind((service.host, service.port))
    except OSError as exc:
        error = {
            "code": "E_PORT",
            "message": f"cannot bind {service.host}:{service.port}: {exc}",
            "node_id": service.node_id,
            "port": service.port,
        }
        print(ERROR_PREFIX + json.dumps(error, sort_keys=True), file=sys.stderr, flush=True)
        raise SystemExit(2) from exc
    server.listen(64)
    ready = {"node_id": service.node_id, "host": service.host, "port": service.port, "pid": __import__("os").getpid()}
    print(READY_PREFIX + json.dumps(ready, sort_keys=True), flush=True)
    try:
        while True:
            conn, address = server.accept()
            thread = threading.Thread(target=_handle_connection, args=(conn, address, service), daemon=True)
            thread.start()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cacheflow-node")
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--seed", default="node-seed")
    parser.add_argument("--ledger", default=None)
    args = parser.parse_args(argv)
    service = NodeService(
        node_id=args.node_id,
        host=args.host,
        port=args.port,
        seed=args.seed,
        ledger_path=args.ledger,
    )
    serve(service)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
