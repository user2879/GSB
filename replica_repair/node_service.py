"""Loopback JSON-lines replica node service and client."""

from __future__ import annotations

import base64
import binascii
import json
import socket
import socketserver
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import BlockError, FormatError, IntegrityError, NetworkError, RepairError, StateError, error_from_dict
from .manifest_repo import block_file
from .model import CURRENT_MANIFEST_VERSION, Manifest, NodeRecord, validate_digest
from .util import (
    atomic_write_bytes,
    canonical_json_dumps,
    sha256_bytes,
    validate_simple_id,
)


MAX_PROTOCOL_LINE = 32 * 1024 * 1024


@dataclass(frozen=True)
class BlockObservation:
    node_id: str
    block_id: str
    size_bytes: int
    digest: str
    expected_size_bytes: int
    expected_digest: str
    manifest_mismatch: bool


class NodeStore:
    def __init__(self, root: Path | str, manifest_path: Path | str, node_id: str) -> None:
        self.root = Path(root)
        self.manifest_path = Path(manifest_path)
        self.node_id = validate_simple_id(node_id, "$.node_id")
        self._lock = threading.RLock()
        self._manifest: Manifest | None = None
        self._node: NodeRecord | None = None
        self.reload_manifest()

    def reload_manifest(self) -> Manifest:
        with self._lock:
            manifest = Manifest.from_file(self.manifest_path)
            if manifest.format_version != CURRENT_MANIFEST_VERSION:
                raise FormatError(
                    "node service requires a current-format manifest",
                    path="$.format_version",
                    details={"found": manifest.format_version, "required": CURRENT_MANIFEST_VERSION},
                )
            nodes = manifest.node_map()
            if self.node_id not in nodes:
                raise FormatError(
                    "node_id is absent from the manifest",
                    path="$.nodes",
                    details={"node_id": self.node_id},
                )
            self._manifest = manifest
            self._node = nodes[self.node_id]
            return manifest

    def manifest(self) -> Manifest:
        with self._lock:
            assert self._manifest is not None
            return self._manifest

    def node(self) -> NodeRecord:
        with self._lock:
            assert self._node is not None
            return self._node

    def block_record(self, block_id: str):
        block_id = validate_simple_id(block_id, "$.block_id")
        with self._lock:
            blocks = self.node().block_map()
            if block_id not in blocks:
                raise BlockError(
                    "block is not present in this node manifest",
                    path=f"$.nodes[{self.node_id}].blocks[{block_id}]",
                    details={"node_id": self.node_id, "block_id": block_id},
                )
            return blocks[block_id]

    def block_path(self, block_id: str) -> Path:
        return block_file(self.root, self.node_id, block_id)

    def read_block(self, block_id: str) -> tuple[bytes, str]:
        block = self.block_record(block_id)
        path = self.block_path(block_id)
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise BlockError(
                "block file is missing",
                path=f"$.nodes[{self.node_id}].blocks[{block_id}]",
                details={"node_id": self.node_id, "block_id": block_id, "file": str(path)},
            ) from exc
        if len(data) != block.size_bytes:
            raise BlockError(
                "block file is truncated or oversized",
                path=f"$.nodes[{self.node_id}].blocks[{block_id}].size_bytes",
                details={
                    "node_id": self.node_id,
                    "block_id": block_id,
                    "expected": block.size_bytes,
                    "actual": len(data),
                },
            )
        return data, sha256_bytes(data)

    def get_digest(self, block_id: str) -> dict[str, Any]:
        block = self.block_record(block_id)
        data, actual = self.read_block(block_id)
        return {
            "ok": True,
            "node_id": self.node_id,
            "block_id": block_id,
            "size_bytes": len(data),
            "actual_digest": actual,
            "manifest_digest": block.digest,
            "manifest_size_bytes": block.size_bytes,
        }

    def get_block(self, block_id: str) -> dict[str, Any]:
        block = self.block_record(block_id)
        data, actual = self.read_block(block_id)
        return {
            "ok": True,
            "node_id": self.node_id,
            "block_id": block_id,
            "size_bytes": len(data),
            "actual_digest": actual,
            "manifest_digest": block.digest,
            "manifest_size_bytes": block.size_bytes,
            "data_b64": base64.b64encode(data).decode("ascii"),
        }

    def put_block(
        self,
        block_id: str,
        data: bytes,
        expected_size: int,
        expected_digest: str,
    ) -> dict[str, Any]:
        block = self.block_record(block_id)
        if not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size <= 0:
            raise FormatError("expected size must be a positive integer", path="$.size_bytes")
        validate_digest(expected_digest, "$.digest")
        if len(data) != expected_size:
            raise BlockError(
                "received block size is invalid",
                path="$.data_b64",
                details={
                    "node_id": self.node_id,
                    "block_id": block_id,
                    "expected": expected_size,
                    "actual": len(data),
                },
            )
        actual = sha256_bytes(data)
        if actual != expected_digest:
            raise IntegrityError(
                "received block digest is invalid",
                path="$.digest",
                details={
                    "node_id": self.node_id,
                    "block_id": block_id,
                    "expected": expected_digest,
                    "actual": actual,
                },
            )
        path = self.block_path(block_id)
        atomic_write_bytes(path, data)
        persisted = path.read_bytes()
        if sha256_bytes(persisted) != expected_digest:
            raise IntegrityError(
                "persisted block failed post-write verification",
                path=f"$.nodes[{self.node_id}].blocks[{block_id}].digest",
                details={"node_id": self.node_id, "block_id": block_id},
            )
        return {
            "ok": True,
            "node_id": self.node_id,
            "block_id": block_id,
            "size_bytes": len(persisted),
            "digest": expected_digest,
            "path": str(path),
            "expected_manifest_digest": block.digest,
        }


class _ThreadingServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


class _NodeRequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        while True:
            line = self.rfile.readline(MAX_PROTOCOL_LINE + 1)
            if not line:
                return
            if len(line) > MAX_PROTOCOL_LINE:
                self._write(
                    {
                        "ok": False,
                        "error": {
                            "code": "E_FORMAT",
                            "message": "protocol line exceeds maximum size",
                            "path": "$",
                        },
                    }
                )
                return
            try:
                request = json.loads(line.decode("utf-8"))
                if not isinstance(request, dict):
                    raise FormatError("request must be a JSON object", path="$")
                response = self.server.dispatch(request)  # type: ignore[attr-defined]
            except RepairError as exc:
                response = exc.to_dict()
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                response = FormatError("request is not valid JSON", path="$").to_dict()
                response["error"]["details"] = {"reason": type(exc).__name__}
            except Exception as exc:  # defensive protocol boundary
                response = StateError(
                    "unexpected node service failure",
                    details={"reason": type(exc).__name__, "message": str(exc)},
                ).to_dict()
            if not self._write(response):
                return

    def _write(self, payload: dict[str, Any]) -> bool:
        try:
            self.wfile.write(canonical_json_dumps(payload).encode("utf-8") + b"\n")
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            return False


class NodeServer:
    def __init__(
        self,
        root: Path | str,
        manifest_path: Path | str,
        node_id: str,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        if host != "127.0.0.1":
            raise NetworkError("node service must bind to loopback", path="$.host")
        if not isinstance(port, int) or isinstance(port, bool) or port < 0 or port > 65535:
            raise NetworkError("port must be in range 0..65535", path="$.port")
        if port != 0 and port < 1024:
            raise NetworkError("explicit ports must be high ports (>=1024)", path="$.port")
        self.store = NodeStore(root, manifest_path, node_id)
        self.host = host
        self.requested_port = port
        try:
            self._server = _ThreadingServer((host, port), _NodeRequestHandler)
        except OSError as exc:
            raise NetworkError(
                "cannot bind loopback port",
                path="$.port",
                details={"host": host, "port": port, "reason": str(exc)},
            ) from exc
        self._server.dispatch = self._dispatch  # type: ignore[attr-defined]
        self.port = int(self._server.server_address[1])
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        return self.host, self.port

    def _dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        operation = request.get("op")
        if operation in {"ping", "health"}:
            manifest = self.store.manifest()
            return {
                "ok": True,
                "node_id": self.store.node_id,
                "group_id": manifest.group_id,
                "manifest_version": manifest.manifest_version,
            }
        if operation == "reload_manifest":
            manifest = self.store.reload_manifest()
            return {
                "ok": True,
                "node_id": self.store.node_id,
                "manifest_version": manifest.manifest_version,
            }
        if operation == "get_digest":
            return self.store.get_digest(_require_block_id(request))
        if operation == "get_block":
            return self.store.get_block(_require_block_id(request))
        if operation == "put_block":
            block_id = _require_block_id(request)
            size = request.get("size_bytes")
            digest = request.get("digest")
            if "data_b64" not in request:
                raise FormatError("required field is missing", path="$.data_b64")
            try:
                data = base64.b64decode(request["data_b64"], validate=True)
            except (binascii.Error, ValueError, TypeError) as exc:
                raise FormatError("data_b64 is not valid base64", path="$.data_b64") from exc
            return self.store.put_block(block_id, data, size, digest)
        raise FormatError(
            "unknown operation",
            path="$.op",
            details={"operation": operation, "supported": ["ping", "reload_manifest", "get_digest", "get_block", "put_block"]},
        )

    def start(self) -> "NodeServer":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name=f"replica-node-{self.store.node_id}",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._thread is None:
            self._server.server_close()
            return
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
        self._thread = None

    def __enter__(self) -> "NodeServer":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


def _require_block_id(request: dict[str, Any]) -> str:
    if "block_id" not in request:
        raise FormatError("required field is missing", path="$.block_id")
    return validate_simple_id(request["block_id"], "$.block_id")


class NodeClient:
    def __init__(self, host: str, port: int, *, timeout: float = 5.0) -> None:
        self.host = host
        self.port = int(port)
        self.timeout = timeout

    def call(self, operation: str, **fields: Any) -> dict[str, Any]:
        request = {"op": operation}
        request.update(fields)
        payload = canonical_json_dumps(request).encode("utf-8") + b"\n"
        try:
            with socket.create_connection((self.host, self.port), timeout=self.timeout) as sock:
                sock.settimeout(self.timeout)
                sock.sendall(payload)
                chunks: list[bytes] = []
                total = 0
                while True:
                    chunk = sock.recv(65536)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    total += len(chunk)
                    if total > MAX_PROTOCOL_LINE:
                        raise NetworkError("node response exceeds maximum size", path="$")
                    if b"\n" in chunk:
                        break
        except RepairError:
            raise
        except OSError as exc:
            raise NetworkError(
                "cannot communicate with node service",
                path="$",
                details={"host": self.host, "port": self.port, "reason": str(exc)},
            ) from exc
        line = b"".join(chunks).split(b"\n", 1)[0]
        if not line:
            raise NetworkError(
                "node service returned an empty response",
                path="$",
                details={"host": self.host, "port": self.port},
            )
        try:
            response = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise NetworkError("node response is not valid JSON", path="$") from exc
        if not isinstance(response, dict):
            raise NetworkError("node response must be an object", path="$")
        if not response.get("ok"):
            error = response.get("error")
            if not isinstance(error, dict):
                raise NetworkError("node response is missing an error object", path="$")
            raise error_from_dict(error)
        return response

    def fetch_observation(
        self,
        block_id: str,
        expected_size: int,
        expected_digest: str,
        *,
        strict_manifest: bool,
    ) -> BlockObservation:
        digest_response = self.call("get_digest", block_id=block_id)
        block_response = self.call("get_block", block_id=block_id)
        if "data_b64" not in block_response:
            raise IntegrityError(
                "block response is missing data_b64",
                path="$.data_b64",
                details={"block_id": block_id},
            )
        try:
            data = base64.b64decode(block_response["data_b64"], validate=True)
        except (binascii.Error, ValueError, TypeError) as exc:
            raise IntegrityError(
                "block response contains invalid base64",
                path="$.data_b64",
                details={"block_id": block_id},
            ) from exc
        actual = sha256_bytes(data)
        declared = block_response.get("actual_digest")
        if declared != actual:
            raise IntegrityError(
                "pulled block digest does not match returned bytes",
                path="$.actual_digest",
                details={
                    "block_id": block_id,
                    "declared": declared,
                    "computed": actual,
                },
            )
        if len(data) != expected_size:
            raise BlockError(
                "pulled block size does not match manifest",
                path="$.size_bytes",
                details={"block_id": block_id, "expected": expected_size, "actual": len(data)},
            )
        if digest_response.get("actual_digest") != actual:
            raise IntegrityError(
                "digest query and data pull disagree",
                path="$.actual_digest",
                details={"block_id": block_id, "digest_query": digest_response.get("actual_digest"), "computed": actual},
            )
        mismatch = actual != expected_digest
        if mismatch and strict_manifest:
            raise IntegrityError(
                "pulled block content does not match manifest digest",
                path=f"$.block[{block_id}].digest",
                details={
                    "node_id": block_response.get("node_id"),
                    "block_id": block_id,
                    "expected": expected_digest,
                    "actual": actual,
                },
            )
        return BlockObservation(
            node_id=str(block_response.get("node_id")),
            block_id=block_id,
            size_bytes=len(data),
            digest=actual,
            expected_size_bytes=expected_size,
            expected_digest=expected_digest,
            manifest_mismatch=mismatch,
        )

    def fetch_block_bytes(self, block_id: str, expected_size: int, expected_digest: str) -> bytes:
        response = self.call("get_block", block_id=block_id)
        if "data_b64" not in response:
            raise IntegrityError("block response is missing data_b64", path="$.data_b64")
        try:
            data = base64.b64decode(response["data_b64"], validate=True)
        except (binascii.Error, ValueError, TypeError) as exc:
            raise IntegrityError("block response contains invalid base64", path="$.data_b64") from exc
        if len(data) != expected_size:
            raise BlockError(
                "source block size is invalid",
                path="$.size_bytes",
                details={"block_id": block_id, "expected": expected_size, "actual": len(data)},
            )
        actual = sha256_bytes(data)
        if actual != expected_digest:
            raise IntegrityError(
                "source block digest is invalid",
                path="$.digest",
                details={"block_id": block_id, "expected": expected_digest, "actual": actual},
            )
        return data

    def put_block(self, block_id: str, data: bytes, expected_digest: str) -> dict[str, Any]:
        return self.call(
            "put_block",
            block_id=block_id,
            size_bytes=len(data),
            digest=expected_digest,
            data_b64=base64.b64encode(data).decode("ascii"),
        )

    def reload_manifest(self) -> dict[str, Any]:
        return self.call("reload_manifest")

    def ping(self) -> dict[str, Any]:
        return self.call("ping")
