"""Independent cache-node subprocess speaking JSON lines over 127.0.0.1 TCP."""

from __future__ import annotations

import argparse
import json
import os
import socketserver
import sys
import threading
import time
from typing import Any

from .content import derive_shard, manifest_digest
from .errors import SpecError
from .model import ContentSpec, Manifest, NodeSpec
from .spec import load_manifest


class WorkerFailure(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class WorkerState:
    def __init__(self, manifest: Manifest, node: NodeSpec) -> None:
        self.manifest = manifest
        self.manifest_digest = manifest_digest(manifest)
        self.node = node
        self.contents: dict[str, ContentSpec] = manifest.content_map
        self.quota_version = node.quota_version
        self.lease_versions: dict[str, int] = {}
        self.lock = threading.RLock()

    def response(self, op: str, **fields: Any) -> dict[str, Any]:
        result = {
            "ok": True,
            "op": op,
            "node_id": self.node.node_id,
            "pid": os.getpid(),
            "mono_ms": time.monotonic_ns() // 1_000_000,
            "quota_version": self.quota_version,
            "manifest_digest": self.manifest_digest,
        }
        result.update(fields)
        return result

    def validate_items(self, batch_id: str, items: Any) -> list[dict[str, Any]]:
        if not isinstance(items, list) or not items:
            raise WorkerFailure("E_SPEC", "items must be a non-empty array")
        results: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                raise WorkerFailure("E_SPEC", "batch item must be an object")
            lease_id = item.get("lease_id")
            content_key = item.get("content_key")
            shard_index = item.get("shard_index")
            expected_digest = item.get("expected_digest")
            if not isinstance(lease_id, str) or not lease_id:
                raise WorkerFailure("E_SPEC", "lease_id must be a non-empty string")
            if content_key not in self.contents:
                raise WorkerFailure("E_SPEC", "unknown content_key")
            if not isinstance(shard_index, int) or isinstance(shard_index, bool):
                raise WorkerFailure("E_SPEC", "shard_index must be an integer")
            content = self.contents[content_key]
            if not 0 <= shard_index < content.shard_count:
                raise WorkerFailure("E_SPEC", "shard_index out of range")
            started = time.monotonic_ns()
            computed = derive_shard(self.node, content, shard_index, self.manifest.config.sample_scale)
            if expected_digest is not None and expected_digest != computed.digest:
                raise WorkerFailure("E_DIGEST", "dispatcher digest does not match node computation")
            self.lease_versions.setdefault(lease_id, 1)
            elapsed_ms = (time.monotonic_ns() - started) // 1_000_000
            results.append(
                {
                    "lease_id": lease_id,
                    "content_key": content_key,
                    "shard_index": shard_index,
                    "digest": computed.digest,
                    "checksum": computed.checksum,
                    "sample_count": computed.sample_count,
                    "elapsed_mono_ms": int(elapsed_ms),
                    "lease_version": self.lease_versions[lease_id],
                }
            )
        return results

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        op = request.get("op")
        with self.lock:
            if op == "ping":
                return self.response("ping")
            if op == "set_quota":
                version = request.get("quota_version")
                if not isinstance(version, int) or isinstance(version, bool) or version <= 0:
                    raise WorkerFailure("E_LEASE", "quota_version must be a positive integer")
                if version < self.quota_version:
                    raise WorkerFailure("E_LEASE", "quota_version cannot decrease")
                previous_version = self.quota_version
                self.quota_version = version
                return self.response("set_quota", previous_version=previous_version, applied_version=version)
            requested_quota = request.get("quota_version")
            if requested_quota is not None and requested_quota != self.quota_version:
                raise WorkerFailure(
                    "E_LEASE",
                    f"stale quota version {requested_quota}; node is at {self.quota_version}",
                )
            if op == "validate_batch":
                batch_id = request.get("batch_id")
                if not isinstance(batch_id, str) or not batch_id:
                    raise WorkerFailure("E_SPEC", "batch_id must be a non-empty string")
                started = time.monotonic_ns()
                items = self.validate_items(batch_id, request.get("items"))
                elapsed_ms = (time.monotonic_ns() - started) // 1_000_000
                return self.response(
                    "validate_batch",
                    batch_id=batch_id,
                    items=items,
                    elapsed_mono_ms=int(elapsed_ms),
                )
            if op == "renew":
                lease_id = request.get("lease_id")
                expected = request.get("expected_version")
                if not isinstance(lease_id, str) or not lease_id:
                    raise WorkerFailure("E_LEASE", "lease_id must be a non-empty string")
                if not isinstance(expected, int) or isinstance(expected, bool) or expected <= 0:
                    raise WorkerFailure("E_LEASE", "expected_version must be a positive integer")
                current = self.lease_versions.get(lease_id)
                if current is None or current != expected:
                    raise WorkerFailure(
                        "E_LEASE",
                        f"lease version mismatch: expected {expected}, current {current}",
                    )
                self.lease_versions[lease_id] = expected + 1
                return self.response("renew", lease_id=lease_id, lease_version=expected + 1)
            if op == "release":
                lease_id = request.get("lease_id")
                if not isinstance(lease_id, str) or not lease_id:
                    raise WorkerFailure("E_LEASE", "lease_id must be a non-empty string")
                self.lease_versions.pop(lease_id, None)
                return self.response("release", lease_id=lease_id)
            if op == "restore":
                leases = request.get("leases")
                if not isinstance(leases, list):
                    raise WorkerFailure("E_LEASE", "leases must be an array")
                restored = 0
                for item in leases:
                    if not isinstance(item, dict):
                        raise WorkerFailure("E_LEASE", "restore item must be an object")
                    lease_id = item.get("lease_id")
                    version = item.get("lease_version")
                    if not isinstance(lease_id, str) or not lease_id:
                        raise WorkerFailure("E_LEASE", "restore lease_id is invalid")
                    if not isinstance(version, int) or isinstance(version, bool) or version <= 0:
                        raise WorkerFailure("E_LEASE", "restore lease_version is invalid")
                    self.lease_versions[lease_id] = version
                    restored += 1
                return self.response("restore", restored=restored)
            if op == "shutdown":
                threading.Thread(target=self._server.shutdown, daemon=True).start()
                return self.response("shutdown")
            raise WorkerFailure("E_SPEC", f"unknown op {op!r}")


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        state: WorkerState = self.server.state  # type: ignore[attr-defined]
        while True:
            raw = self.rfile.readline(16 * 1024 * 1024 + 1)
            if not raw:
                return
            if len(raw) > 16 * 1024 * 1024:
                self._send({"ok": False, "code": "E_SPEC", "message": "request too large"})
                return
            try:
                request = json.loads(raw.decode("utf-8"))
                if not isinstance(request, dict):
                    raise WorkerFailure("E_SPEC", "request must be an object")
                if request.get("op") == "hello":
                    if request.get("node_id") != state.node.node_id:
                        raise WorkerFailure("E_NODE", "node_id mismatch")
                    response = state.response("hello", lease_count=len(state.lease_versions))
                else:
                    response = state.handle(request)
            except WorkerFailure as exc:
                response = {
                    "ok": False,
                    "code": exc.code,
                    "message": exc.message,
                    "node_id": state.node.node_id,
                    "pid": os.getpid(),
                    "mono_ms": time.monotonic_ns() // 1_000_000,
                }
            except Exception as exc:  # defensive boundary for the child process
                response = {
                    "ok": False,
                    "code": "E_NODE",
                    "message": f"unhandled worker error: {exc}",
                    "node_id": state.node.node_id,
                    "pid": os.getpid(),
                    "mono_ms": time.monotonic_ns() // 1_000_000,
                }
            self._send(response)

    def _send(self, response: dict[str, Any]) -> None:
        self.wfile.write(
            json.dumps(response, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        )
        self.wfile.flush()


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = os.name != "nt"
    daemon_threads = True


def serve(manifest: Manifest, node: NodeSpec) -> None:
    state = WorkerState(manifest, node)
    try:
        server = _Server((node.host, node.port), _Handler)
    except OSError as exc:
        print(
            json.dumps(
                {"event": "error", "code": "E_PORT", "node_id": node.node_id, "message": str(exc)},
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(2)
    state._server = server  # type: ignore[attr-defined]
    server.state = state  # type: ignore[attr-defined]
    print(
        json.dumps(
            {
                "event": "ready",
                "node_id": node.node_id,
                "host": node.host,
                "port": node.port,
                "pid": os.getpid(),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.05)
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="edge-cache node subprocess")
    parser.add_argument("--sample", required=True)
    parser.add_argument("--node-id", required=True)
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.sample)
        node = manifest.node_map.get(args.node_id)
        if node is None:
            raise SpecError("unknown node_id", line=1, field="node_id", path="node_id")
        serve(manifest, node)
        return 0
    except SpecError as exc:
        print(json.dumps({"event": "error", **exc.to_dict()}, ensure_ascii=False), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
