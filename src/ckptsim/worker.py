from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from .canonical import canonical_json
from .errors import CkptError
from .model import build_batch_samples, compute_batch, generate_parameter_set
from .protocol import recv_json_line, send_json_line
from .spec import ParameterVersion, ShardSpec, load_layers, load_shards, spec_error


class WorkerLeaseRegistry:
    def __init__(self, ttl_ms: int) -> None:
        self.ttl_ms = ttl_ms
        self._lock = threading.Lock()
        self._leases: dict[str, dict[str, Any]] = {}

    def register(self, lease_id: str, version: int, batch_id: str) -> None:
        now = time.monotonic_ns()
        with self._lock:
            existing = self._leases.get(lease_id)
            if existing is not None and existing["expires_ns"] > now:
                if existing["version"] != version:
                    raise CkptError(
                        "E_LEASE",
                        "lease already exists with a different version",
                        batch_id=batch_id,
                    )
                existing["expires_ns"] = now + self.ttl_ms * 1_000_000
                return
            self._leases[lease_id] = {
                "version": version,
                "expires_ns": now + self.ttl_ms * 1_000_000,
                "batch_id": batch_id,
            }

    def renew(self, lease_id: str, expected_version: int, ttl_ms: int) -> dict[str, Any]:
        now = time.monotonic_ns()
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None or lease["expires_ns"] <= now:
                self._leases.pop(lease_id, None)
                raise CkptError("E_LEASE", "lease is unknown or expired")
            if lease["version"] != expected_version:
                raise CkptError(
                    "E_LEASE",
                    f"lease version mismatch: expected {expected_version}, actual {lease['version']}",
                    field="lease_version",
                )
            lease["version"] = expected_version + 1
            lease["expires_ns"] = now + ttl_ms * 1_000_000
            return {
                "ok": True,
                "lease_id": lease_id,
                "lease_version": lease["version"],
                "expires_ms": lease["expires_ns"] // 1_000_000,
            }


class WorkerState:
    def __init__(
        self,
        shard: ShardSpec,
        versions: dict[int, ParameterVersion],
        lease_ttl_ms: int,
    ) -> None:
        self.shard = shard
        self.versions = versions
        self.parameter_cache: dict[int, Any] = {}
        self.lease_registry = WorkerLeaseRegistry(lease_ttl_ms)
        self.lease_ttl_ms = lease_ttl_ms

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        operation = request.get("op")
        if operation == "health":
            return {"ok": True, "shard_id": self.shard.shard_id, "pid": os.getpid()}
        if operation == "renew":
            return self._renew(request)
        if operation == "batch":
            return self._batch(request)
        raise CkptError("E_FIELD", f"unknown operation {operation!r}", field="op")

    def _require(self, request: dict[str, Any], field: str, expected_type: type) -> Any:
        if field not in request:
            raise CkptError("E_FIELD", f"request is missing {field}", field=field)
        value = request[field]
        if expected_type is int:
            if type(value) is not int:
                raise CkptError(
                    "E_FIELD", f"request field {field} must be an integer", field=field
                )
        elif not isinstance(value, expected_type):
            raise CkptError(
                "E_FIELD", f"request field {field} has the wrong type", field=field
            )
        return value

    def _renew(self, request: dict[str, Any]) -> dict[str, Any]:
        shard_id = self._require(request, "shard_id", str)
        if shard_id != self.shard.shard_id:
            raise CkptError(
                "E_FIELD",
                f"request targets shard {shard_id!r}, this worker serves {self.shard.shard_id!r}",
                shard_id=shard_id,
                field="shard_id",
            )
        lease_id = self._require(request, "lease_id", str)
        version = self._require(request, "lease_version", int)
        ttl_ms = request.get("lease_ttl_ms", self.lease_ttl_ms)
        if type(ttl_ms) is not int or ttl_ms <= 0:
            raise CkptError("E_FIELD", "lease_ttl_ms must be positive", field="lease_ttl_ms")
        return self.lease_registry.renew(lease_id, version, ttl_ms)

    def _batch(self, request: dict[str, Any]) -> dict[str, Any]:
        started_ns = time.monotonic_ns()
        shard_id = self._require(request, "shard_id", str)
        if shard_id != self.shard.shard_id:
            raise CkptError(
                "E_FIELD",
                f"request targets shard {shard_id!r}, this worker serves {self.shard.shard_id!r}",
                shard_id=shard_id,
                field="shard_id",
            )
        batch_id = self._require(request, "batch_id", str)
        parameter_version = self._require(request, "parameter_version", int)
        sample_ids = self._require(request, "sample_ids", list)
        if (
            not sample_ids
            or any(type(value) is not int or value < 0 for value in sample_ids)
            or len(set(sample_ids)) != len(sample_ids)
        ):
            raise CkptError(
                "E_FIELD",
                "sample_ids must be a non-empty unique integer array",
                shard_id=shard_id,
                batch_id=batch_id,
                field="sample_ids",
            )
        version = self.versions.get(parameter_version)
        if version is None:
            raise CkptError(
                "E_FIELD",
                f"unknown parameter_version {parameter_version}",
                shard_id=shard_id,
                batch_id=batch_id,
                field="parameter_version",
            )
        lease_id = self._require(request, "lease_id", str)
        lease_version = self._require(request, "lease_version", int)
        lease_ttl_ms = request.get("lease_ttl_ms", self.lease_ttl_ms)
        if type(lease_ttl_ms) is not int or lease_ttl_ms <= 0:
            raise CkptError(
                "E_FIELD",
                "lease_ttl_ms must be positive",
                shard_id=shard_id,
                batch_id=batch_id,
                field="lease_ttl_ms",
            )
        self.lease_registry.register(lease_id, lease_version, batch_id)
        samples = request.get("samples")
        if samples is not None and not isinstance(samples, dict):
            raise CkptError(
                "E_FIELD",
                "samples must be an object keyed by root layer",
                shard_id=shard_id,
                batch_id=batch_id,
                field="samples",
            )
        try:
            computation = compute_batch(
                self.shard,
                version,
                batch_id,
                sorted(sample_ids),
                samples=samples,
                cache=self.parameter_cache,
            )
        except ValueError as exc:
            raise CkptError(
                "E_FIELD",
                str(exc),
                shard_id=shard_id,
                batch_id=batch_id,
                field="samples",
            ) from exc
        duration_ms = (time.monotonic_ns() - started_ns) // 1_000_000
        return replace(computation, duration_ms=duration_ms).to_response()


def _recv_line(sock: socket.socket) -> dict[str, Any]:
    return recv_json_line(sock)


def _serve_connection(state: WorkerState, connection: socket.socket) -> None:
    with connection:
        connection.settimeout(30.0)
        try:
            request = _recv_line(connection)
            response = state.handle(request)
        except CkptError as exc:
            response = {"ok": False, "error": exc.to_dict()}
        except Exception as exc:  # keep the protocol line-shaped on all failures
            response = {
                "ok": False,
                "error": {"code": "E_FIELD", "detail": f"worker failed: {exc}"},
            }
        try:
            send_json_line(connection, response)
        except OSError:
            pass


def run_worker(
    *,
    shards_path: str,
    layers_path: str,
    shard_id: str,
    ready_file: str | None,
    lease_ttl_ms: int,
) -> int:
    shard_config = load_shards(shards_path)
    layer_config = load_layers(layers_path)
    shards = shard_config.by_id()
    if shard_id not in shards:
        raise spec_error(f"unknown shard {shard_id!r}", item=shard_id, line=1)
    shard = shards[shard_id]
    versions = layer_config.by_version()
    if shard.parameter_version not in versions:
        raise spec_error(
            f"shard {shard_id!r} references missing parameter_version {shard.parameter_version}",
            item=shard_id,
            line=shard.line,
        )
    state = WorkerState(shard, versions, lease_ttl_ms)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        listener.bind((shard.host, shard.port))
        listener.listen(128)
        listener.settimeout(0.25)
    except OSError as exc:
        listener.close()
        raise CkptError(
            "E_PORT",
            f"cannot bind {shard.host}:{shard.port}: {exc}",
            item=shard.shard_id,
        ) from exc
    stop = threading.Event()

    def _signal_handler(signum: int, frame: Any) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    ready = {
        "ok": True,
        "event": "ready",
        "shard_id": shard.shard_id,
        "host": shard.host,
        "port": shard.port,
        "pid": os.getpid(),
    }
    if ready_file:
        path = Path(ready_file)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(canonical_json(ready) + "\n", encoding="utf-8")
        temporary.replace(path)
    print(canonical_json(ready), flush=True)
    try:
        while not stop.is_set():
            try:
                connection, _ = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                if stop.is_set():
                    break
                raise
            threading.Thread(
                target=_serve_connection,
                args=(state, connection),
                name=f"ckptsim-{shard.shard_id}-conn",
                daemon=True,
            ).start()
    finally:
        listener.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one deterministic TCP training shard")
    parser.add_argument("--shards", required=True)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--shard-id", required=True)
    parser.add_argument("--ready-file")
    parser.add_argument("--lease-ttl-ms", type=int, default=30_000)
    args = parser.parse_args(argv)
    try:
        return run_worker(
            shards_path=args.shards,
            layers_path=args.layers,
            shard_id=args.shard_id,
            ready_file=args.ready_file,
            lease_ttl_ms=args.lease_ttl_ms,
        )
    except CkptError as exc:
        print(canonical_json({"ok": False, "error": exc.to_dict()}), file=sys.stderr, flush=True)
        return 3
    except Exception as exc:
        print(
            canonical_json(
                {"ok": False, "error": {"code": "E_FIELD", "detail": str(exc)}}
            ),
            file=sys.stderr,
            flush=True,
        )
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
