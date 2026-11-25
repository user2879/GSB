"""JSON-lines loopback TCP protocol, node server and node client.

Each request/response is one JSON object terminated by ``\\n`` on a real TCP
socket connected to 127.0.0.1. Socket timeouts are always explicit so blocked
handshakes and missing receipts surface as E_HANDSHAKE / E_FIELD rather than
hanging the orchestrator.
"""

import json
import socket
import threading
import time

from .errors import HandshakeError, PortError, ProtocolFieldError
from .samples import decode_batch, fixed_point_aggregate, sample_digest
from .util import now_ms

REQUIRED_RECEIPT_FIELDS = (
    "type",
    "node_id",
    "shard_id",
    "config_version",
    "capacity_version",
    "summary",
    "elapsed_ms",
)


def _send_message(sock, message):
    data = (json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8")
    sock.sendall(data)


def _recv_message(sock):
    buffer = bytearray()
    while not buffer.endswith(b"\n"):
        chunk = sock.recv(65536)
        if not chunk:
            raise ProtocolFieldError("connection closed before complete message")
        buffer.extend(chunk)
    try:
        return json.loads(buffer.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ProtocolFieldError("invalid JSON message: %s" % exc)


class NodeServer:
    """Threaded TCP server executing the node-side fixed-point workload."""

    def __init__(self, node, effective_config):
        self.node = node
        self.effective_config = dict(effective_config)
        self.server_socket = None
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._lease_versions = {}
        self._stopped = False

    def start(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind((self.node.host, self.node.port))
        except OSError as exc:
            server.close()
            raise PortError(
                "node port %d is unavailable: %s" % (self.node.port, exc),
                object_id=self.node.id,
                field="port",
            )
        server.listen(16)
        server.settimeout(0.2)
        self.server_socket = server
        self._thread = threading.Thread(
            target=self._accept_loop, name="node-%s" % self.node.id, daemon=True
        )
        self._thread.start()

    def _accept_loop(self):
        while not self._stop.is_set():
            try:
                client, _ = self.server_socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(
                target=self._handle_connection,
                args=(client,),
                name="node-%s-conn" % self.node.id,
                daemon=True,
            ).start()

    def _handle_connection(self, client):
        with client:
            client.settimeout(30.0)
            try:
                request = _recv_message(client)
                response = self._dispatch(request)
                _send_message(client, response)
            except (OSError, ProtocolFieldError, KeyError, TypeError, ValueError):
                return

    def _dispatch(self, request):
        message_type = request.get("type")
        if message_type == "handshake":
            return {
                "type": "handshake_ack",
                "node_id": self.node.id,
                "capacity_version": self.node.capacity_version,
                "fixed_scale": self.node.fixed_scale,
            }
        if message_type == "dispatch":
            return self._dispatch_shard(request)
        if message_type == "renew":
            return self._renew_lease(request)
        if message_type == "cancel":
            with self._lock:
                self._lease_versions.pop(request.get("shard_id"), None)
            return {"type": "cancel_ack", "shard_id": request.get("shard_id")}
        raise ProtocolFieldError("unknown request type %r" % message_type)

    def _dispatch_shard(self, request):
        for field in ("shard_id", "config_version", "lease_id", "lease_version",
                      "batch", "sample_digest"):
            if field not in request:
                raise ProtocolFieldError(
                    "dispatch missing field %s" % field,
                    object_id=request.get("shard_id"),
                    field=field,
                )
        groups, values = decode_batch(request["batch"])
        actual_digest = sample_digest(groups, values)
        if actual_digest != request["sample_digest"]:
            raise ProtocolFieldError(
                "sample digest mismatch", object_id=request["shard_id"],
                field="sample_digest",
            )
        started = now_ms()
        summary = fixed_point_aggregate(
            groups,
            values,
            self.node.fixed_scale,
            self.effective_config,
            self.node.compute_iters,
        )
        elapsed = now_ms() - started
        with self._lock:
            self._lease_versions[request["shard_id"]] = request["lease_version"]
        return {
            "type": "receipt",
            "node_id": self.node.id,
            "shard_id": request["shard_id"],
            "config_version": request["config_version"],
            "capacity_version": self.node.capacity_version,
            "lease_id": request["lease_id"],
            "lease_version": request["lease_version"],
            "summary": summary,
            "elapsed_ms": elapsed,
        }

    def _renew_lease(self, request):
        for field in ("shard_id", "lease_id", "lease_version"):
            if field not in request:
                raise ProtocolFieldError("renew missing field %s" % field,
                                         field=field)
        with self._lock:
            known = self._lease_versions.get(request["shard_id"])
        if known != request["lease_version"]:
            return {
                "type": "renew_reject",
                "node_id": self.node.id,
                "shard_id": request["shard_id"],
                "reason": "lease_version_mismatch",
                "expected_version": known,
            }
        with self._lock:
            self._lease_versions[request["shard_id"]] = request["lease_version"] + 1
        return {
            "type": "renew_ack",
            "node_id": self.node.id,
            "shard_id": request["shard_id"],
            "lease_version": request["lease_version"] + 1,
        }

    def stop(self):
        if self._stopped:
            return
        self._stopped = True
        self._stop.set()
        if self.server_socket is not None:
            try:
                self.server_socket.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)


class NodeClient:
    """Short-lived socket client used by worker threads."""

    def __init__(self, node, handshake_timeout_ms, io_timeout_ms):
        self.node = node
        self.handshake_timeout = handshake_timeout_ms / 1000.0
        self.io_timeout = io_timeout_ms / 1000.0

    def _connect(self, timeout):
        deadline = time.monotonic() + timeout
        last_error = None
        while time.monotonic() < deadline:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(max(0.05, min(timeout, deadline - time.monotonic())))
            try:
                sock.connect((self.node.host, self.node.port))
                return sock
            except ConnectionRefusedError as exc:
                # The child interpreter is still starting; retry inside the
                # real socket timeout window instead of failing immediately.
                last_error = exc
                sock.close()
                time.sleep(0.05)
            except socket.timeout as exc:
                sock.close()
                raise HandshakeError(
                    "connection to node %s timed out" % self.node.id,
                    object_id=self.node.id,
                    field="handshake",
                ) from exc
            except OSError as exc:
                sock.close()
                raise PortError(
                    "cannot connect to node %s on %s:%d: %s"
                    % (self.node.id, self.node.host, self.node.port, exc),
                    object_id=self.node.id,
                    field="port",
                ) from exc
        raise HandshakeError(
            "node %s did not accept a connection within %.0f ms: %s"
            % (self.node.id, timeout * 1000.0, last_error),
            object_id=self.node.id,
            field="handshake",
        )

    def handshake(self):
        sock = self._connect(self.handshake_timeout)
        try:
            sock.settimeout(self.handshake_timeout)
            start = time.monotonic()
            _send_message(sock, {"type": "handshake", "node_id": self.node.id})
            try:
                response = _recv_message(sock)
            except (socket.timeout, ProtocolFieldError, OSError) as exc:
                raise HandshakeError(
                    "handshake timed out for node %s: %s" % (self.node.id, exc),
                    object_id=self.node.id,
                    field="handshake",
                )
            if response.get("type") != "handshake_ack":
                raise HandshakeError(
                    "bad handshake acknowledgement from node %s" % self.node.id,
                    object_id=self.node.id,
                    field="type",
                )
            if time.monotonic() - start > self.handshake_timeout:
                raise HandshakeError(
                    "handshake exceeded timeout on node %s" % self.node.id,
                    object_id=self.node.id,
                    field="handshake",
                )
            return response
        finally:
            sock.close()

    def dispatch(self, shard_id, config_version, lease_id, lease_version, batch,
                 digest):
        request = {
            "type": "dispatch",
            "shard_id": shard_id,
            "config_version": config_version,
            "lease_id": lease_id,
            "lease_version": lease_version,
            "batch": batch,
            "sample_digest": digest,
        }
        return self._request(request, self.io_timeout, REQUIRED_RECEIPT_FIELDS)

    def renew(self, shard_id, lease_id, lease_version):
        request = {
            "type": "renew",
            "shard_id": shard_id,
            "lease_id": lease_id,
            "lease_version": lease_version,
        }
        return self._request(request, self.io_timeout, ("type", "shard_id"))

    def cancel(self, shard_id, lease_id):
        request = {"type": "cancel", "shard_id": shard_id, "lease_id": lease_id}
        return self._request(request, self.io_timeout, ("type", "shard_id"))

    def _request(self, request, timeout, required_fields):
        sock = self._connect(timeout)
        try:
            sock.settimeout(timeout)
            _send_message(sock, request)
            try:
                response = _recv_message(sock)
            except socket.timeout as exc:
                raise HandshakeError(
                    "socket timeout waiting for response",
                    object_id=request.get("shard_id"),
                    field="receipt",
                ) from exc
        finally:
            sock.close()
        for field in required_fields:
            if field not in response:
                raise ProtocolFieldError(
                    "response missing required field %s" % field,
                    object_id=request.get("shard_id"),
                    field=field,
                )
        return response
