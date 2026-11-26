"""Training-shard child process and TCP JSONL worker."""

from __future__ import annotations

import argparse
import json
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any

from .errors import DiagnosticError
from .model import LayerSpec, ShardSpec
from .params import compute_batch_summary
from .protocol import PROTOCOL_VERSION, recv_json_line, send_json_line


def shard_from_dict(payload: dict[str, Any]) -> ShardSpec:
    layers = tuple(
        LayerSpec(
            layer_id=str(layer["layer_id"]),
            input_dim=int(layer["input_dim"]),
            output_dim=int(layer["output_dim"]),
            scale=int(layer["scale"]),
            depends_on=tuple(str(dep) for dep in layer.get("depends_on", [])),
        )
        for layer in payload["layers"]
    )
    return ShardSpec(
        shard_id=str(payload["shard_id"]),
        host=str(payload["host"]),
        port=int(payload["port"]),
        lease_limit=int(payload["lease_limit"]),
        concurrency_limit=int(payload["concurrency_limit"]),
        seed=int(payload["seed"]),
        parameter_version=int(payload["parameter_version"]),
        sample_dim=int(payload["sample_dim"]),
        lease_ttl_ms=int(payload["lease_ttl_ms"]),
        layers=layers,
    )


class WorkerLeaseBook:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._leases: dict[str, dict[str, int]] = {}

    def _prune(self) -> None:
        now = time.monotonic_ns() // 1_000_000
        for lease_id in [
            key for key, value in self._leases.items() if now >= value["expires_at_ms"]
        ]:
            self._leases.pop(lease_id, None)

    def issue(self, lease_id: str, ttl_ms: int) -> dict[str, int]:
        with self._lock:
            self._prune()
            if lease_id in self._leases:
                raise DiagnosticError("E_LEASE", f"lease {lease_id} already exists", field="lease_id")
            now = time.monotonic_ns() // 1_000_000
            record = {
                "lease_version": 1,
                "expires_at_ms": now + ttl_ms,
                "renew_count": 0,
            }
            self._leases[lease_id] = record
            return dict(record)

    def renew(self, lease_id: str, lease_version: int, ttl_ms: int) -> dict[str, int]:
        with self._lock:
            self._prune()
            record = self._leases.get(lease_id)
            if record is None:
                raise DiagnosticError("E_LEASE", f"unknown or expired lease {lease_id}", field="lease_id")
            if record["lease_version"] != lease_version:
                raise DiagnosticError(
                    "E_LEASE",
                    f"lease version mismatch: expected {record['lease_version']}, got {lease_version}",
                    field="lease_version",
                )
            now = time.monotonic_ns() // 1_000_000
            record["lease_version"] += 1
            record["renew_count"] += 1
            record["expires_at_ms"] = now + ttl_ms
            return dict(record)

    def release(self, lease_id: str) -> None:
        with self._lock:
            self._leases.pop(lease_id, None)


class ShardServer:
    def __init__(
        self,
        shard: ShardSpec,
        *,
        connect_timeout_s: float = 30.0,
    ) -> None:
        self.shard = shard
        self.connect_timeout_s = connect_timeout_s
        self.leases = WorkerLeaseBook()
        self.stop_event = threading.Event()
        self._listener: socket.socket | None = None

    def bind(self) -> socket.socket:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind((self.shard.host, self.shard.port))
            listener.listen(128)
            listener.settimeout(0.2)
        except OSError as exc:
            listener.close()
            raise DiagnosticError(
                "E_PORT",
                f"cannot bind {self.shard.host}:{self.shard.port}: {exc}",
                field="port",
                details={"port": self.shard.port},
            ) from exc
        self._listener = listener
        return listener

    def serve_forever(self) -> None:
        listener = self._listener or self.bind()
        while not self.stop_event.is_set():
            try:
                connection, _ = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            thread = threading.Thread(
                target=self._serve_connection, args=(connection,), daemon=True
            )
            thread.start()
        try:
            listener.close()
        except OSError:
            pass

    def _serve_connection(self, connection: socket.socket) -> None:
        try:
            connection.settimeout(self.connect_timeout_s)
            request = recv_json_line(connection, self.connect_timeout_s, peer="master")
            response = self.handle(request)
            send_json_line(connection, response)
        except DiagnosticError as exc:
            try:
                send_json_line(connection, {"ok": False, **exc.to_dict()})
            except OSError:
                pass
        except Exception as exc:  # pragma: no cover - defensive boundary
            try:
                send_json_line(
                    connection,
                    {
                        "ok": False,
                        "code": "E_PROTOCOL",
                        "message": f"unhandled worker error: {exc}",
                    },
                )
            except OSError:
                pass
        finally:
            try:
                connection.close()
            except OSError:
                pass

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        request_type = request.get("type")
        if request_type == "handshake":
            requested_shard = request.get("shard_id")
            if requested_shard != self.shard.shard_id:
                raise DiagnosticError(
                    "E_FIELD", "handshake shard_id does not match worker", field="shard_id"
                )
            return {
                "ok": True,
                "shard_id": self.shard.shard_id,
                "pid": os.getpid(),
                "protocol_version": PROTOCOL_VERSION,
            }
        if request_type == "ping":
            return {"ok": True, "shard_id": self.shard.shard_id, "pid": os.getpid()}
        if request_type == "batch":
            for field in ("shard_id", "batch_id", "sample_ids", "parameter_version"):
                if field not in request:
                    raise DiagnosticError(
                        "E_FIELD", f"batch request is missing {field}", field=field
                    )
            if request["shard_id"] != self.shard.shard_id:
                raise DiagnosticError("E_FIELD", "batch shard_id mismatch", field="shard_id")
            try:
                batch_id = str(request["batch_id"])
                sample_ids = [int(value) for value in request["sample_ids"]]
                version = int(request["parameter_version"])
            except (TypeError, ValueError) as exc:
                raise DiagnosticError("E_FIELD", f"invalid batch field: {exc}") from exc
            started = time.monotonic_ns()
            summary = compute_batch_summary(
                self.shard, batch_id, sample_ids, version
            )
            elapsed_ms = max(0, (time.monotonic_ns() - started) // 1_000_000)
            response = {
                "ok": True,
                **summary,
                "elapsed_ms": int(elapsed_ms),
                "pid": os.getpid(),
            }
            if "checkpoint_id" in request:
                checkpoint_id = request["checkpoint_id"]
                if not isinstance(checkpoint_id, str) or not checkpoint_id:
                    raise DiagnosticError(
                        "E_FIELD", "checkpoint_id must be a non-empty string", field="checkpoint_id"
                    )
                response["checkpoint_id"] = checkpoint_id
            return response
        if request_type == "lease_issue":
            lease_id = request.get("lease_id")
            ttl_ms = request.get("ttl_ms")
            if not isinstance(lease_id, str) or not lease_id:
                raise DiagnosticError("E_FIELD", "lease_issue requires lease_id", field="lease_id")
            if isinstance(ttl_ms, bool) or not isinstance(ttl_ms, int) or ttl_ms <= 0:
                raise DiagnosticError("E_FIELD", "lease_issue requires positive ttl_ms", field="ttl_ms")
            record = self.leases.issue(lease_id, ttl_ms)
            return {
                "ok": True,
                "shard_id": self.shard.shard_id,
                "lease_id": lease_id,
                "lease_version": record["lease_version"],
                "expires_at_ms": record["expires_at_ms"],
            }
        if request_type == "lease_renew":
            lease_id = request.get("lease_id")
            version = request.get("lease_version")
            ttl_ms = request.get("ttl_ms")
            if not isinstance(lease_id, str) or not lease_id:
                raise DiagnosticError("E_FIELD", "lease_renew requires lease_id", field="lease_id")
            if isinstance(version, bool) or not isinstance(version, int):
                raise DiagnosticError("E_FIELD", "lease_renew requires lease_version", field="lease_version")
            if isinstance(ttl_ms, bool) or not isinstance(ttl_ms, int) or ttl_ms <= 0:
                raise DiagnosticError("E_FIELD", "lease_renew requires positive ttl_ms", field="ttl_ms")
            record = self.leases.renew(lease_id, version, ttl_ms)
            return {
                "ok": True,
                "shard_id": self.shard.shard_id,
                "lease_id": lease_id,
                "lease_version": record["lease_version"],
                "expires_at_ms": record["expires_at_ms"],
            }
        if request_type == "lease_release":
            lease_id = request.get("lease_id")
            if not isinstance(lease_id, str) or not lease_id:
                raise DiagnosticError("E_FIELD", "lease_release requires lease_id", field="lease_id")
            self.leases.release(lease_id)
            return {
                "ok": True,
                "shard_id": self.shard.shard_id,
                "lease_id": lease_id,
                "state": "released",
            }
        if request_type == "shutdown":
            self.stop_event.set()
            return {"ok": True, "shard_id": self.shard.shard_id, "state": "stopping"}
        raise DiagnosticError("E_FIELD", f"unknown request type {request_type!r}", field="type")


def run_from_file(path: str | Path) -> int:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    shard = shard_from_dict(payload)
    server = ShardServer(shard)
    try:
        server.bind()
    except DiagnosticError as exc:
        print(json.dumps({"event": "error", **exc.to_dict()}, separators=(",", ":")), flush=True)
        return 2
    print(
        json.dumps(
            {
                "event": "ready",
                "shard_id": shard.shard_id,
                "host": shard.host,
                "port": shard.port,
                "pid": os.getpid(),
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    server.serve_forever()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GSB training shard worker")
    parser.add_argument("--spec", required=True, help="shard declaration JSON file")
    args = parser.parse_args(argv)
    return run_from_file(args.spec)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
