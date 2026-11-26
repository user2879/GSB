from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from typing import Any

from .canonical import canonical_text
from .errors import ObserverError
from .spec import ScriptEntry, SessionSpec, load_session
from .transport import bind_listener


def deterministic_response(spec: SessionSpec, entry: ScriptEntry) -> dict[str, Any]:
    base = {
        "type": "response",
        "event_seq": entry.event_seq,
        "peer_id": f"peer-{spec.session_id}",
        "script_version": spec.script_version,
        "duration_ms": entry.delay_ms,
        "kind": entry.type,
        "ack_count": entry.ack_count,
        "dup_count": entry.dup_count,
        "order": list(entry.order),
    }
    if entry.type == "loss":
        base["disposition"] = "simulated_loss"
        base["ack_count"] = 0
    return base


def _write_line(conn: socket.socket, value: dict[str, Any]) -> None:
    conn.sendall((canonical_text(value) + "\n").encode("utf-8"))


def serve(spec: SessionSpec) -> int:
    listener = bind_listener(spec.peer_host, spec.peer_port)
    actual_port = listener.getsockname()[1]
    print(canonical_text({"type": "ready", "port": actual_port, "pid": __import__("os").getpid()}), flush=True)
    listener.settimeout(max(0.001, spec.handshake_timeout_ms / 1000.0))
    try:
        conn, address = listener.accept()
    except (socket.timeout, TimeoutError) as exc:
        print(canonical_text({"code": "E_TIMEOUT", "message": "peer accept timed out"}), file=sys.stderr, flush=True)
        listener.close()
        return 3
    with listener:
        with conn:
            conn.settimeout(max(0.001, spec.handshake_timeout_ms / 1000.0))
            hello = {
                "type": "hello",
                "peer_id": f"peer-{spec.session_id}",
                "script_digest": spec.script_digest,
                "script_version": spec.script_version,
                "address": list(address),
            }
            _write_line(conn, hello)
            reader = conn.makefile("r", encoding="utf-8", newline="\n")
            client_hello_line = reader.readline()
            if not client_hello_line:
                return 4
            try:
                client_hello = json.loads(client_hello_line)
            except json.JSONDecodeError:
                return 4
            if client_hello.get("type") != "hello_ack":
                return 4
            conn.settimeout(max(0.001, spec.read_timeout_ms / 1000.0))
            while True:
                line = reader.readline()
                if not line:
                    break
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    _write_line(conn, {"type": "error", "code": "E_FORMAT", "message": "invalid JSON line"})
                    continue
                if message.get("type") == "shutdown":
                    _write_line(conn, {"type": "bye"})
                    break
                event_seq = message.get("event_seq")
                entry = spec.entry_for_seq(int(event_seq)) if isinstance(event_seq, int) and not isinstance(event_seq, bool) else None
                if entry is None:
                    _write_line(conn, {"type": "error", "code": "E_SCRIPT", "event_seq": event_seq, "message": "no script entry"})
                    continue
                if entry.delay_ms:
                    time.sleep(entry.delay_ms / 1000.0)
                _write_line(conn, deterministic_response(spec, entry))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic local TCP peer subprocess")
    parser.add_argument("--spec", required=True)
    args = parser.parse_args(argv)
    try:
        spec = load_session(args.spec)
        return serve(spec)
    except ObserverError as exc:
        print(canonical_text(exc.to_dict()), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
