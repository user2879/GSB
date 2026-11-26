from __future__ import annotations

import argparse
import json
import socket
import sys
import traceback
from typing import Any

from .scoring import compute_score_reply
from .util import canonical_json


MAX_LINE = 1024 * 1024


def _serve(host: str, port: int) -> int:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        listener.bind((host, port))
        listener.listen(8)
    except OSError as exc:
        print(
            canonical_json({"error": "E_PORT", "message": str(exc)}),
            file=sys.stderr,
            flush=True,
        )
        listener.close()
        return 2
    print(
        canonical_json({"ready": True, "host": host, "port": port}),
        flush=True,
    )
    try:
        while True:
            connection, _ = listener.accept()
            with connection:
                connection.settimeout(30)
                buffer = bytearray()
                while True:
                    newline = buffer.find(b"\n")
                    if newline < 0:
                        try:
                            chunk = connection.recv(65536)
                        except socket.timeout:
                            continue
                        if not chunk:
                            break
                        buffer.extend(chunk)
                        if len(buffer) > MAX_LINE:
                            _send(connection, {"error": "E_REPLY", "message": "line too long"})
                            break
                        continue
                    line = bytes(buffer[:newline])
                    del buffer[: newline + 1]
                    if not line.strip():
                        continue
                    try:
                        request: Any = json.loads(line)
                    except json.JSONDecodeError as exc:
                        _send(
                            connection,
                            {"error": "E_REPLY", "message": f"invalid JSON: {exc.msg}"},
                        )
                        continue
                    if request == {"command": "shutdown"}:
                        return 0
                    try:
                        reply = compute_score_reply(request)
                    except (KeyError, TypeError, ValueError) as exc:
                        _send(
                            connection,
                            {"error": "E_REPLY", "message": str(exc)},
                        )
                        continue
                    except BaseException:
                        _send(
                            connection,
                            {
                                "error": "E_REPLY",
                                "message": "internal scoring failure",
                                "trace": traceback.format_exc(limit=2),
                            },
                        )
                        continue
                    _send(connection, reply)
    finally:
        listener.close()
    return 0


def _send(connection: socket.socket, payload: dict[str, Any]) -> None:
    connection.sendall((canonical_json(payload) + "\n").encode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic affinity score worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args(argv)
    if args.host != "127.0.0.1":
        print(
            canonical_json(
                {"error": "E_PORT", "message": "worker only binds 127.0.0.1"}
            ),
            file=sys.stderr,
            flush=True,
        )
        return 2
    return _serve(args.host, args.port)


if __name__ == "__main__":
    raise SystemExit(main())
