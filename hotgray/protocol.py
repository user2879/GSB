"""Newline-delimited JSON framing over loopback TCP sockets.

Every message is a single JSON object terminated by ``\\n``. All socket reads
and writes are guarded by explicit timeouts so a wedged backend raises a real
timeout error instead of hanging the gateway.
"""

import json

from .errors import HandshakeError, ReplyError

REQUIRED_REPLY_FIELDS = ("type", "backend_id", "seq", "param_digest",
                         "sample_ids", "output_digests", "batch_digest",
                         "elapsed_ms")


def send_message(sock, message):
    try:
        sock.sendall(json.dumps(message, ensure_ascii=False).encode("utf-8") + b"\n")
    except TimeoutError as exc:
        raise HandshakeError(f"socket send timed out: {exc}") from exc
    except OSError as exc:
        raise HandshakeError(f"socket send failed: {exc}") from exc


def recv_message(sock, timeout):
    try:
        sock.settimeout(timeout)
        line = sock.makefile("rb", buffering=0).readline()
    except TimeoutError as exc:
        raise HandshakeError(f"socket read timed out after {timeout}s") from exc
    except OSError as exc:
        raise HandshakeError(f"socket read failed: {exc}") from exc
    if not line:
        raise HandshakeError("backend closed the connection unexpectedly")
    try:
        return json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReplyError(f"malformed JSON line from backend: {exc}")


def validate_reply(reply, expected_type=None, expected_seq=None):
    if not isinstance(reply, dict):
        raise ReplyError("reply is not a JSON object")
    missing = [f for f in REQUIRED_REPLY_FIELDS if f not in reply]
    if missing:
        raise ReplyError(f"reply missing fields: {missing}", field=missing[0])
    if expected_type is not None and reply.get("type") != expected_type:
        raise ReplyError(
            f"unexpected reply type {reply.get('type')!r}, want {expected_type!r}",
            field="type",
        )
    if expected_seq is not None and reply.get("seq") != expected_seq:
        raise ReplyError(
            f"reply seq {reply.get('seq')} != expected {expected_seq}", field="seq"
        )
    return True
