from __future__ import annotations

import argparse
import json
import os
import socketserver
import threading
from pathlib import Path
from typing import Any, Optional

from .checksum import block_group_checksum
from .errors import E_FIELD, E_LEASE, E_PORT, E_SPEC, MeshError
from .protocol import encode_line, generation_from_label, label_for
from .spec import load_declaration
from .timeutil import monotonic_ms


class NodeService:
    def __init__(self, node: dict[str, Any], quota_version: int) -> None:
        self.node_id = node["node_id"]
        self.node = node
        self.quota_version = int(quota_version)
        self.checksum_profile = node["checksum"]
        self.leases: dict[str, dict[str, Any]] = {}
        self.request_cache: dict[str, dict[str, Any]] = {}
        self.lock = threading.RLock()

    def _base(self, payload: dict[str, Any], started: float) -> dict[str, Any]:
        checksum = block_group_checksum(self.checksum_profile)
        return {
            "ok": True,
            "node_id": self.node_id,
            "lease_id": payload.get("lease_id"),
            "quota_version": self.quota_version,
            "checksum_digest": checksum["checksum_digest"],
            "checksum": checksum,
            "compute_ms": checksum["compute_ms"],
            "elapsed_ms": max(0, int(monotonic_ms() - started)),
            "pid": os.getpid(),
        }

    def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        started = monotonic_ms()
        request_id = payload.get("request_id")
        if isinstance(request_id, str) and request_id in self.request_cache:
            cached = dict(self.request_cache[request_id])
            cached["idempotent_replay"] = True
            return cached

        op = payload.get("op")
        if op == "ping":
            response = self._base(payload, started)
            response.update({"type": "pong", "generation": 0, "lease_id": None})
            return response

        if op not in {"grant", "renew", "release"}:
            raise MeshError(f"unknown node operation: {op!r}", code=E_FIELD, field="op")

        lease_id = payload.get("lease_id")
        if not isinstance(lease_id, str) or not lease_id:
            raise MeshError("lease_id is required", code=E_FIELD, field="lease_id")

        with self.lock:
            lease = self.leases.get(lease_id)
            if op == "grant":
                if lease is not None:
                    response = self._reject_generation(
                        payload,
                        started,
                        lease,
                        "lease already exists on node; retry is idempotent only with same request_id",
                        code=E_LEASE,
                    )
                    return self._cache(request_id, response)
                generation = 1
                lease = {
                    "lease_id": lease_id,
                    "state": "granted",
                    "generation": generation,
                    "label": label_for(lease_id, generation),
                    "slot_id": payload.get("slot_id"),
                }
                self.leases[lease_id] = lease
                response = self._base(payload, started)
                response.update(
                    {
                        "state": lease["state"],
                        "generation": generation,
                        "label": lease["label"],
                        "advanced": True,
                    }
                )
                return self._cache(request_id, response)

            if lease is None:
                checksum = block_group_checksum(self.checksum_profile)
                response = {
                    "ok": False,
                    "code": E_LEASE,
                    "error": f"unknown lease_id {lease_id}",
                    "node_id": self.node_id,
                    "lease_id": lease_id,
                    "generation": None,
                    "checksum_digest": checksum["checksum_digest"],
                    "compute_ms": checksum["compute_ms"],
                    "quota_version": self.quota_version,
                }
                return self._cache(request_id, response)

            expected = self._expected_generation(payload, lease_id)
            current = int(lease["generation"])
            if expected != current:
                response = self._reject_generation(
                    payload,
                    started,
                    lease,
                    "stale generation label",
                )
                response["details"] = {
                    "expected_generation": expected,
                    "current_generation": current,
                }
                return self._cache(request_id, response)

            before_state = lease["state"]
            before_generation = current
            if op == "renew":
                if before_state not in {"granted", "renewed"}:
                    return self._reject_state(payload, started, lease, "renew")
                lease["state"] = "renewed"
            else:
                if before_state not in {"granted", "renewed"}:
                    return self._reject_state(payload, started, lease, "release")
                lease["state"] = "released"

            lease["generation"] = current + 1
            lease["label"] = label_for(lease_id, lease["generation"])
            response = self._base(payload, started)
            response.update(
                {
                    "state": lease["state"],
                    "before_state": before_state,
                    "generation": lease["generation"],
                    "generation_before": before_generation,
                    "label": lease["label"],
                    "advanced": True,
                }
            )
            return self._cache(request_id, response)

    def _expected_generation(self, payload: dict[str, Any], lease_id: str) -> int:
        if payload.get("label") is not None:
            return generation_from_label(payload["label"], lease_id)
        generation = payload.get("generation")
        if not isinstance(generation, int) or isinstance(generation, bool) or generation <= 0:
            raise MeshError("generation or label is required", code=E_FIELD, field="generation")
        return int(generation)

    def _reject_generation(
        self,
        payload: dict[str, Any],
        started: float,
        lease: dict[str, Any],
        message: str,
        code: str = E_LEASE,
    ) -> dict[str, Any]:
        response = self._base(payload, started)
        response.update(
            {
                "ok": False,
                "code": code,
                "error": message,
                "state": lease["state"],
                "generation": lease["generation"],
                "label": lease["label"],
                "advanced": False,
            }
        )
        response.pop("checksum", None)
        return response

    def _reject_state(
        self, payload: dict[str, Any], started: float, lease: dict[str, Any], op: str
    ) -> dict[str, Any]:
        response = self._reject_generation(
            payload, started, lease, f"illegal {op} from state {lease['state']}", code="E_STATE"
        )
        response["code"] = "E_STATE"
        return response

    def _cache(self, request_id: Any, response: dict[str, Any]) -> dict[str, Any]:
        if isinstance(request_id, str):
            self.request_cache[request_id] = dict(response)
        return response


class LeaseTcpHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        service: NodeService = self.server.service  # type: ignore[attr-defined]
        timeout_ms = getattr(self.server, "socket_timeout_ms", 2000)
        self.connection.settimeout(max(0.001, timeout_ms / 1000.0))
        while True:
            try:
                line = self.rfile.readline()
            except (TimeoutError, OSError):
                return
            if not line:
                return
            try:
                payload = json.loads(line.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("request is not an object")
            except Exception as exc:
                response = {"ok": False, "code": E_FIELD, "error": f"invalid JSON request: {exc}"}
                self._write(response)
                return

            if payload.get("op") == "hello":
                self._write(
                    {
                        "ok": True,
                        "type": "hello_ack",
                        "node_id": service.node_id,
                        "pid": os.getpid(),
                    }
                )
                continue
            if payload.get("op") == "shutdown":
                self._write({"ok": True, "node_id": service.node_id, "type": "shutdown_ack"})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return

            try:
                response = service.handle(payload)
            except MeshError as exc:
                response = exc.to_dict()
                response["node_id"] = service.node_id
            except Exception as exc:  # pragma: no cover - defensive node boundary
                response = {"ok": False, "code": "E_IO", "error": str(exc), "node_id": service.node_id}
            self._write(response)

    def _write(self, response: dict[str, Any]) -> None:
        try:
            self.wfile.write(encode_line(response))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass


class ReusableThreadingServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], handler: type, service: NodeService, timeout_ms: int):
        self.service = service
        self.socket_timeout_ms = timeout_ms
        super().__init__(address, handler)


def write_ready(path: Optional[str], payload: dict[str, Any]) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(temporary, target)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="LeaseMesh storage node subprocess")
    parser.add_argument("--declarations", required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--ready-file")
    args = parser.parse_args(argv)

    try:
        spec = load_declaration(args.declarations)
        node = next((item for item in spec["nodes"] if item["node_id"] == args.node_id), None)
        if node is None:
            raise MeshError(f"node not found: {args.node_id}", code=E_SPEC, field="nodes")
        service = NodeService(node, int(spec["quota_version"]))
        try:
            server = ReusableThreadingServer(
                (node["host"], int(node["port"])),
                LeaseTcpHandler,
                service,
                int(spec.get("socket_timeout_ms", 2000)),
            )
        except OSError as exc:
            payload = {
                "ok": False,
                "code": E_PORT,
                "error": f"cannot bind {node['host']}:{node['port']}: {exc}",
                "field": "port",
                "port": node["port"],
                "node_id": args.node_id,
            }
            write_ready(args.ready_file, payload)
            print(json.dumps(payload), flush=True)
            return 2

        write_ready(
            args.ready_file,
            {
                "ok": True,
                "code": "OK",
                "pid": os.getpid(),
                "node_id": args.node_id,
                "host": node["host"],
                "port": node["port"],
            },
        )
        server.serve_forever(poll_interval=0.05)
        server.server_close()
        return 0
    except MeshError as exc:
        write_ready(args.ready_file, exc.to_dict())
        print(json.dumps(exc.to_dict()), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
