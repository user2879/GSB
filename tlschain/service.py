from __future__ import annotations

import argparse
import json
import socket
import sys
from typing import Any

from .canonical import canonical_text
from .errors import DomainError
from .models import ScriptMessage, Spec
from .spec import load_spec


MAX_LINE_BYTES = 1_048_576


def _send_line(sock: socket.socket, value: dict[str, Any]) -> None:
    sock.sendall(canonical_text(value).encode("utf-8") + b"\n")


def _recv_line(sock: socket.socket, buffer: bytearray) -> bytes | None:
    while b"\n" not in buffer:
        chunk = sock.recv(65536)
        if not chunk:
            return None
        buffer.extend(chunk)
        if len(buffer) > MAX_LINE_BYTES:
            raise DomainError("E_REPLY", "socket line exceeds maximum size", field_path="reply.line")
    line, _, rest = buffer.partition(b"\n")
    buffer[:] = rest
    return bytes(line)


def _message_wire(message: ScriptMessage) -> dict[str, Any]:
    return message.wire_dict()


def _find_group(spec: Spec, client_seq: int) -> dict[str, Any] | None:
    for group in spec.message_groups():
        if group["request"].client_seq == client_seq:
            return group
    return None


def _handle_request(
    request: dict[str, Any],
    spec: Spec,
    drop_client_seq: int | None,
    omit_reply_field: str | None,
) -> dict[str, Any] | None:
    if request.get("op") != "client_message":
        return {
            "status": "ERROR",
            "code": "E_REPLY",
            "message": "unsupported operation",
        }
    supplied = request.get("message", {})
    client_seq = supplied.get("client_seq", request.get("client_seq"))
    if not isinstance(client_seq, int) or isinstance(client_seq, bool):
        return {
            "status": "ERROR",
            "code": "E_REPLY",
            "message": "client_seq must be an integer",
            "message_seq": request.get("seq"),
        }
    group = _find_group(spec, client_seq)
    if group is None:
        return {
            "status": "ERROR",
            "code": "E_REPLY",
            "message": f"unknown client_seq: {client_seq}",
            "message_seq": request.get("seq"),
        }
    expected = group["request"]
    for field in ("seq", "sender", "client_seq", "arrival_ms", "type", "cert_ids"):
        if supplied.get(field) != expected.as_dict().get(field):
            return {
                "status": "ERROR",
                "code": "E_REPLY",
                "message": f"request field mismatch: {field}",
                "message_seq": expected.seq,
                "field": field,
            }
    if drop_client_seq == client_seq:
        return None
    replies = [_message_wire(message) for message in group["replies"]]
    if omit_reply_field and replies:
        replies[0].pop(omit_reply_field, None)
    return {
        "status": "OK",
        "client_seq": client_seq,
        "request_seq": expected.seq,
        "replies": replies,
    }


def serve(
    cert_path: str,
    handshake_path: str,
    host: str,
    port: int,
    drop_client_seq: int | None = None,
    omit_reply_field: str | None = None,
) -> int:
    try:
        spec = load_spec(cert_path, handshake_path)
    except DomainError as exc:
        sys.stdout.write(canonical_text({"status": "ERROR", **exc.as_dict()}) + "\n")
        sys.stdout.flush()
        return 2
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.bind((host, port))
        listener.listen(1)
    except OSError as exc:
        sys.stdout.write(
            canonical_text(
                {
                    "status": "ERROR",
                    "code": "E_PORT",
                    "message": f"cannot bind {host}:{port}: {exc}",
                    "cert_id": None,
                    "field": "socket.port",
                }
            )
            + "\n"
        )
        sys.stdout.flush()
        listener.close()
        return 3
    sys.stdout.write(canonical_text({"status": "READY", "host": host, "port": port}) + "\n")
    sys.stdout.flush()
    listener.settimeout(max(5.0, spec.socket_timeout_ms / 1000.0))
    try:
        connection, _address = listener.accept()
    except OSError:
        listener.close()
        return 4
    buffer = bytearray()
    try:
        connection.settimeout(max(5.0, spec.socket_timeout_ms / 1000.0))
        while True:
            line = _recv_line(connection, buffer)
            if line is None:
                break
            if not line.strip():
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                _send_line(
                    connection,
                    {
                        "status": "ERROR",
                        "code": "E_REPLY",
                        "message": "request is not valid JSON",
                    },
                )
                continue
            response = _handle_request(request, spec, drop_client_seq, omit_reply_field)
            if response is not None:
                _send_line(connection, response)
    except (OSError, DomainError):
        pass
    finally:
        connection.close()
        listener.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic loopback TLS handshake service")
    parser.add_argument("--certs", required=True)
    parser.add_argument("--handshake", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--drop-client-seq", type=int)
    parser.add_argument("--omit-reply-field")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return serve(
        args.certs,
        args.handshake,
        args.host,
        args.port,
        args.drop_client_seq,
        args.omit_reply_field,
    )


if __name__ == "__main__":
    raise SystemExit(main())
